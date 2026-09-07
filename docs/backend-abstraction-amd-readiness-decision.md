# バックエンド抽象層の AMD（ROCm/HIP）readiness 設計記録（#1340）

イシュー #1340「将来の AMD（ROCm/HIP）バックエンド追加に備えたバックエンド抽象層の境界（warp 幅の実行時パラメータ化・シャッフルのマスク差異吸収・cooperative launch の区別・stream 優先 API）を設計記録として書く」に対応する。親: #1333（Phase 4 横断データパス）・ルート: #1269。

本ドキュメントは **コード変更を伴わない設計記録のみ**である。実装（`crates/**`）・テスト・依存・ガードレール閾値・数値一致許容誤差はいずれも変更しない。ROCm/HIP は REQ-2 の受け入れ基準上、引き続き対象外（`docs/spec/04-requirements.md` REQ-2・`docs/backend-matrix.md` §3.4「ROCm・Vulkan は対象外」）であり、本ドキュメントは対象範囲の変更ではなく、将来 ROCm/HIP バックエンドを追加する場合に必要となる抽象層の境界を先に言語化しておく「備え（readiness）」の記録である。仕様変更が必要になった場合は正本である `docs/spec/`（fandhe-ai-spec リポジトリ）側への提案とし、本リポでは `docs/spec/` を編集しない。

棚卸し時点の HEAD SHA: `a659d0497f09cbafc2f31b821ff133d96bcb80b7`（2026-09-07）。`file_path:line` は同 SHA 時点のもの。後続の変更で行番号がずれる可能性があるため、参照する際は当該コミット、または近傍のコミットで再確認すること。

## 判断サマリ

**warp／wave 幅は実行時デバイス属性として抽象層（`tensor-core::DeviceInfo`）へ移し、カーネル文字列にはレンダリング時 `#define` で数値注入する。シャッフル／warp 同期はバックエンド別の薄いマクロで吸収する。カーネル起動 API は「通常起動」「persistent（CTA 間の待ち合わせなし）」「cooperative（grid 全体同期）」の 3 区分とし、cooperative は専用 API 経由に限定して通常起動へフォールバックしない。Tensor Core 系 ISA（`mma.sync`／`ldmatrix`／`cp.async`／`wmma::`）はハードウェア命令セット自体が異なるため抽象化対象外とし、引き続きバックエンド別カーネル族として扱う（抽象層が担うのは経路選択インターフェースのみ）。**

この判断は本ドキュメントの記述にとどまり、`crates/tensor-core`・`crates/backend-cuda`・`crates/backend-metal` への実装は行わない（§6「起票案」参照）。

## 1. 背景

現行の CUDA（`crates/backend-cuda`）・Metal（`crates/backend-metal`）カーネルは、warp／simdgroup 幅 32 を次の形でコード中に埋め込んでいる。

- カーネル文字列（NVRTC ソース・MSL）中のレーン導出式・シャッフル呼び出しへの数値リテラル埋め込み
- Rust 側のブロック次元定数（`RMSNORM_BLOCK_DIM`・`SOFTMAX_BLOCK_DIM`・`SIMDGROUP_THREADGROUP_WIDTH` 等）

一方、AMD GPU は世代によって warp（HIP では「wavefront」または「wave」と呼ぶ）幅が異なる（RDNA = 32・CDNA = 64。`.claude/skills/amd-rocm/references/hip/cpp-language-extensions.md:22`「`warpSize` (32 on RDNA, 64 on CDNA)」）。また `__shfl_xor()` は CUDA の `__shfl_xor_sync()` と異なりマスク引数を取らない（`cpp-language-extensions.md:27`）。grid 全体の同期を要するカーネルは通常起動と別の API（`hipLaunchCooperativeKernel`）を要する（`.claude/skills/amd-rocm/references/hip/cooperative-groups.md:36`）。これらは現行実装の「32 固定・フルマスク固定・通常起動のみ」という前提と衝突する。

## 2. 棚卸し（AC-1）

32 固定・warp 幅前提の箇所を、意味論のカテゴリ別に整理する。列「判断」は §3 の判断表に対応する。

### (A) レーン／warp 番号の導出式（`tid / 32`・`tid % 32`）

| file_path:line | 内容 |
|---|---|
| `crates/backend-cuda/src/kernels_mse.rs:101-102`, `:144-145` | `lane = threadIdx.x % 32`／`warp_id = threadIdx.x / 32` |
| `crates/backend-cuda/src/kernels_rmsnorm.rs:441-442` | 同上（rmsnorm 版） |
| `crates/backend-cuda/src/kernels_mma.rs:1264-1265` | `warp_id = tid / 32`／`lane = tid % 32`（`MMA_WARP_M` は `:267`） |
| `crates/backend-cuda/src/kernels_mma_tf32.rs:349-350` | 同上（TF32 版。`MMA_TF32_WARP_M`／`MMA_TF32_WARP_N` は `:183-184`） |
| `crates/backend-cuda/src/kernels_mma_tf32x3.rs:220-221` | 同上（TF32x3 版） |
| `crates/backend-cuda/src/kernels_wmma_opt.rs:888`, `:2367`, `:2881` | `warp_id = tid / 32` |
| `crates/backend-cuda/src/kernels_wmma_opt.rs:3518-3519` | `warp_grid_m = base.block_m / 32`／`warp_grid_n = base.block_n / 32`（Rust 側ホストコード） |
| `crates/backend-cuda/src/kernels.rs:596` | `warp_id = tid / 32`（WMMA TF32 basic） |

### (B) butterfly reduction の初期 offset = 16（warp 幅 32 の半分を前提）

| file_path:line | 内容 |
|---|---|
| `crates/backend-cuda/src/kernels_mse.rs:112`, `:123`, `:153`, `:164` | `for (int offset = 16; offset > 0; offset >>= 1)` |
| `crates/backend-cuda/src/kernels_softmax.rs:214`, `:295` | 同上 |
| `crates/backend-cuda/src/kernels_rmsnorm.rs:268`, `:364`, `:483`, `:495` | 同上 |
| `crates/backend-metal/src/shaders/softmax.metal:69-73`, `:79-83` | `simd_shuffle_xor(v, 16u)` 〜 `simd_shuffle_xor(v, 1u)`（5 段展開済み） |
| `crates/backend-metal/src/shaders/rmsnorm.metal:88`, `:221-228` | `RMSNORM_SIMD_WIDTH = 32u` を参照するループで `simd_shuffle_xor` |

### (C) フルマスク `0xffffffff`（`__shfl_xor_sync`／`__syncwarp`。CUDA 固有引数）

| file_path:line | 内容 |
|---|---|
| `crates/backend-cuda/src/kernels_softmax.rs:206`, `:215-216`, `:232`, `:296-297` | `__syncwarp(0xffffffffu)`／`__shfl_xor_sync(0xffffffffu, …)` |
| `crates/backend-cuda/src/kernels_rmsnorm.rs:260`, `:269`, `:295`, `:365`, `:484`, `:496` | 同上 |
| `crates/backend-cuda/src/kernels_mse.rs:113`, `:124`, `:154`, `:165` | `__shfl_xor_sync(0xffffffff, …)`（マスク引数のみ、`__syncwarp` 呼び出しはこのファイルにはない） |

### (D) Rust 側ブロック次元定数が warp 幅と同値で、warp 内演算（シャッフル・`__syncwarp`）を前提とするもの

| file_path:line | 内容 |
|---|---|
| `crates/backend-cuda/src/kernels_rmsnorm.rs:144` | `pub const RMSNORM_BLOCK_DIM: u32 = 32;`（`:140` docコメントに「`__shfl_xor_sync`／`__syncwarp` による warp 内 reduction を前提とするため 32 固定」の旨を明記） |
| `crates/backend-cuda/src/kernels_softmax.rs:118-120` | `pub const SOFTMAX_BLOCK_DIM: u32 = 32;`（doc コメント「`kernels_rmsnorm.rs::RMSNORM_BLOCK_DIM` と同じ理由」） |
| `crates/backend-metal/src/gemm.rs:50` | `const SIMDGROUP_THREADGROUP_WIDTH: usize = 32;`（gemm.metal 側のストライドとコメントで結合。`:4136-4240` に一致検証テストあり） |
| `crates/backend-metal/src/rmsnorm.rs:33` | `const RMSNORM_THREADGROUP_WIDTH: usize = 32;` |
| `crates/backend-metal/src/softmax.rs:36` | `const SOFTMAX_THREADGROUP_WIDTH: usize = 32;` |

参考として、以下は warp 幅とは独立のブロッキング設計値（タイル寸法）であり、意味論上「32」を共有するのみで本棚卸しの対象外（誤検出防止のため明記）:

- `crates/backend-cuda/src/kernels.rs:91`（`TILE = 32`）
- `crates/backend-cuda/src/kernels_mma.rs:240` 付近（`MMA_BK = 32`）
- `crates/backend-cuda/src/kernels_tiled_pipeline_128x64.rs:947`（`BANKS = 32`。共有メモリバンク数であり warp 幅と無関係）
- `crates/backend-cuda/src/kernels_transpose.rs:55-58`（32×32 タイルの転置。バンクコンフリクト回避の文脈）

### (E) Metal 側 simdgroup 幅 32 前提（実測取得と手動定数の二重管理）

| file_path:line | 内容 |
|---|---|
| `crates/backend-metal/src/gemm.rs:498` | `thread_execution_width: u32`（`MTLComputePipelineState` フィールド） |
| `crates/backend-metal/src/gemm.rs:1467` | `thread_execution_width: pipeline.threadExecutionWidth() as u32`（実機から実測取得） |
| `crates/backend-metal/src/error.rs:244` 付近 | `MetalError::UnexpectedThreadExecutionWidth { expected, actual }`（実測値が期待値と不一致なら fail-closed でエラー化） |
| `crates/backend-metal/src/shaders/gemm.metal:550` | `i += 32u; // 32 = Rust 側 SIMDGROUP_THREADGROUP_WIDTH（gemm.rs）と一致` |
| `crates/backend-metal/src/shaders/gemm.metal:1087-1088`, `:1816-1817`, `:2116-2117` | `simd_id * 32 + simd_lane`／`WM * WN * 32` |
| `crates/backend-metal/src/tile.rs:2257` 付近 | `thread_count()` が `wm * wn * 32` に一致することを検査するテスト |

Apple GPU の simdgroup 幅は 32 固定であり、実測（`threadExecutionWidth()`）と手動定数（`SIMDGROUP_THREADGROUP_WIDTH` 等）の一致を実行時に検証する「検証付き定数」パターンを既に採用している。

### (F) バックエンド固有 ISA（warp 幅の問題ではなくフラグメントレイアウトが命令セット固有。抽象化対象外）

| file_path:line | 内容 |
|---|---|
| `crates/backend-cuda/src/kernels_mma.rs:1237`, `:1467`, `:1493`, `:1601` 付近 | `cp.async`／`ldmatrix`／`mma.sync` |
| `crates/backend-cuda/src/kernels_mma_tf32.rs:327`, `:522`, `:586` 付近 | 同上（TF32 版） |
| `crates/backend-cuda/src/kernels_mma_tf32x3.rs:196`, `:358`, `:398` 付近 | 同上（TF32x3 版） |
| `crates/backend-cuda/src/kernels.rs:603-661` 付近 | `wmma::` ネームスペース API |
| `crates/backend-cuda/src/kernels_tiled_pipeline*.rs` | `cp.async` 多段パイプライン |

## 3. 抽象層へ移す項目・移さない項目（AC-1 判断表）

| 項目 | 判断 | 置き場所・方式 |
|------|------|--------------|
| warp／wave 幅 | **移す**（実行時デバイス属性） | `crates/tensor-core/src/device.rs` の `DeviceInfo`（`#[non_exhaustive]`。既存フィールドは `device`・`name`・`total_memory_bytes`・`compute_units` 等）へ `warp_width: Option<u32>` を追加する案。CUDA は既存の `multiprocessor_count()`（`crates/backend-cuda/src/device.rs:229`）と同じ層で `CU_DEVICE_ATTRIBUTE_WARP_SIZE`（`cudarc =0.19.8` の `sys` に存在。属性値は実装時に cudarc のバージョン固定ドキュメントで再確認する）から取得する案。Metal は既存の `threadExecutionWidth()` 実測＋不一致エラー（`gemm.rs:1467`・`error.rs:244`）を「検証付き定数」パターンとしてそのまま `DeviceInfo` 経由に一般化する。CPU は `None` |
| カーネル文字列への幅注入 | **移す**（レンダリング時 `#define`） | `kernels_tiled_pipeline*.rs` の `render_source` が既に整数パラメータを `#define` プレフィクスとしてソースへ埋め込む方式を採用している（NVRTC ソース生成への外部入力混入防止のため、文字列ではなく数値〈`u32`〉のみを埋め込む設計を踏襲する）。この方式を横展開し、`WARP_SIZE`／`WARP_HALF`（butterfly 初期 offset）をレンダリング時定数として注入する。§2 (A)(B)(D) のリテラル `32`・`16` が置換対象になる |
| シャッフル・warp 同期のマスク差異 | **移す**（薄いマクロ） | `WARP_SHFL_XOR(v, off)`／`WARP_SYNC()` 相当をソースプレフィクスのマクロとして定義する案。CUDA 側は `__shfl_xor_sync(FULL_MASK, …)`（マスク型は wave64 で 64 bit 化が必要になりうる。`porting-cuda-to-hip.md:24`「lane-mask bit operations may need 64-bit integers on 64-wide warps」）、HIP 側は `__shfl_xor(v, off)`（マスク引数なし。`cpp-language-extensions.md:27`）として定義する。**`__syncwarp` に対応する HIP API は本ドキュメントが参照した `.claude/skills/amd-rocm/references/hip/*.md` の範囲では確認できなかった**ため、「要出典確認」として断定しない（§6 起票案でロックステップ実行前提の妥当性を含め確認する） |
| ブロック次元定数（rmsnorm／softmax の 1 行 = 1 warp 型） | **移す**（幅から導出） | `RMSNORM_BLOCK_DIM`／`SOFTMAX_BLOCK_DIM` を固定値ではなく `DeviceInfo::warp_width` から導出する形へ変更する案（RDNA なら 32・CDNA なら 64）。`derive_persistent_grid_*`（`crates/backend-cuda/src/rmsnorm.rs:96/115/138`）は SM 数ベースの grid-stride 導出であり幅に依存しないため、この変更でも契約は不変 |
| Metal の simdgroup 幅 | **移す（同じパラメータ経路を通すが値は 32 固定）** | Apple GPU は世代によらず simdgroup 幅 32 固定。抽象層の `warp_width` には常に 32 を報告させ、既存の実測検証（`UnexpectedThreadExecutionWidth`）はそのまま維持する |
| タイル寸法の 32（`TILE`・`MMA_BK`・`BANKS` 等） | **移さない** | warp 幅と独立のブロッキング設計値であり、§2 (D) 直後の「参考」欄で述べたとおり誤って同一視しない |
| Tensor Core ISA（§2 (F)） | **移さない** | バックエンド別カーネル族として維持する。抽象層が担うのは「Tensor Core 経路の選択インターフェース」（既存の `MatrixUnit` 分岐相当）のみで、カーネル本体（`mma.sync`／`wmma::` 等）は移さない。HIP では MFMA／rocWMMA 相当の別カーネル族になる想定だが、本ドキュメントではその実装可否を評価しない |
| 数値一致複合判定・FMA 契約・カーネル側手動境界チェック | **不変** | `.claude/rules/coding-rust.md`「バックエンド間数値一致は統一複合判定〈相対誤差 1e-3 未満または絶対誤差 1e-5 未満〉」「性能下限・最適化の達成を理由に、シェーダ・カーネル側の手動境界チェックを省略しない」は本設計記録の対象外であり、変更しない旨をここに明記する |
| 依存の追加・更新（`cudarc` の feature 追加を含む） | **不変** | 本ドキュメントは依存を追加・更新しない。`=x.y.z` 完全固定方針（`.claude/rules/deps-policy.md`）は不変 |

## 4. カーネル起動種別の区別（AC-2）

| 種別 | 定義 | 現行コードの該当 | 起動 API |
|------|------|----------------|---------|
| 通常起動 | CTA（thread block）間の待ち合わせがない | 全カーネルの大半（`stream.launch_builder(func).launch(cfg)` 方式。例: `crates/backend-cuda/src/elementwise.rs:186`・`crates/backend-cuda/src/gemm_mma.rs:828`・`crates/backend-cuda/src/gemm.rs` の多数箇所） | CUDA `cuLaunchKernel` 相当（`launch_builder`）／HIP `hipModuleLaunchKernel`（module-API 経由。`porting-cuda-to-hip.md:25` が挙げる stream-based API に相当） |
| persistent（同期なし） | grid 次元を SM 数などに固定し、atomic 操作でタイルを動的配布するが、CTA 間で完了を待ち合わせる同期はない | `crates/backend-cuda/src/kernels_tiled_pipeline_128x64.rs:733`（`s_tile = atomicAdd(tile_counter, 1u)`。`docs/perf/cuda-gemm-tiled-pipeline-persistent.md` 参照）、`crates/backend-cuda/src/rmsnorm.rs` の `derive_persistent_grid_*`（grid-stride ループ、SM 数ベースで幅に依存しない） | 通常起動 API で正しい。複数 CTA の同時実行（co-residency）は性能上の期待にすぎず、正しさの前提条件ではないことをここに明記する |
| cooperative（grid 全体同期） | fixup 等の目的で、あるカーネル実行内から他 CTA の完了を明示的に待ち合わせる（`grid.sync()` 相当を含む） | 本ドキュメント執筆時点の HEAD（`a659d049`）の `crates/backend-cuda/src/*.rs` には `launch_cooperative` の使用は見つからなかった（`grep -n launch_cooperative crates/backend-cuda/src/*.rs` が空）。Stream-K の fixup（`docs/cuda-streamk-decision.md`。関連イシュー #1357 は本ドキュメント執筆時点で open）が「他 CTA の部分和をスピン待ちする」設計を採る場合はこの区分に該当しうるが、現行実装は該当しない | CUDA `cuLaunchCooperativeKernel`（`cudarc =0.19.8` の `src/driver/safe/launch.rs` に `launch_cooperative` として存在し、シグネチャは **`unsafe fn`**）／HIP `hipLaunchCooperativeKernel`＋デバイスのcooperative launch対応可否の属性検査（`cooperative-groups.md:24,36`）。Metal に相当する概念はなく、Metal バックエンドでは cooperative 区分は使用されず通常起動へ退化する |

### 方針

抽象層のカーネル起動入口は `LaunchKind { Normal, Persistent, Cooperative }` 相当の区分を持たせる案とし、`Cooperative` を選ぶ場合は次の 3 点を満たすことを設計上の要件として記録する。

1. デバイス属性でcooperative launch対応可否を fail-closed に検査する（未対応デバイスでは `Cooperative` を要求された時点でエラーとし、通常起動へ黙って fallback しない）
2. 占有可能な CTA 数（occupancy）以下に grid を強制する（cooperative launch は全 CTA の同時常駐を要求するため）
3. `Cooperative` を要求されたカーネルは通常起動 API へは絶対に落とさない（同期意味論が異なるため、フォールバックは正しさを壊す）

`cudarc` の `launch_cooperative` は `unsafe fn` であるため、これを採用する場合は `.claude/rules/coding-rust.md`「`unsafe` は FFI 境界の必要最小限に留め、理由をコメントで明記しレビュー必須」に基づき、実装時にユーザー承認・レビュー（security-auditor）が必要になることをここに記録する。本ドキュメントでは結線の計画は立てない。

## 5. stream 優先 API の境界

現行 CUDA 実装は既に stream 中心の設計である。`crates/backend-cuda/src/device.rs:118` の `default_stream()` を起点に、全カーネル起動・`synchronize`・H2D/D2H 転送が `Arc<CudaStream>` 経由で行われる（同期契約の詳細は `docs/backend-cuda-async-execution-design.md`）。Metal も同型で、コマンドバッファ共有・エンコーダ共有による同期境界を持つ（`docs/backend-metal-command-batching-design.md`）。

この構成を踏まえ、抽象層が公開する境界は「コンパイル済みカーネル関数ハンドル ＋ stream（キュー）ハンドル」までとし、コンテキスト／モジュール管理（`crates/backend-cuda/src/context_cache.rs`・`module_cache.rs`・NVRTC JIT）はバックエンド内部の実装詳細として抽象層の外に置く。

HIP では `hipCtx*`／`hipModule*` は legacy 扱いで、新規コードは `hipSetDevice` とstream-based APIが推奨される（`porting-cuda-to-hip.md:25`）が、`hipModuleLaunchKernel`／HIPRTC（HIP版NVRTC相当）自体は存在するため、NVRTC JIT ＋ module_cache という現行方式に対応するHIP側の経路は存在する、という事実のみをここに記録する（`.claude/skills/amd-rocm` の参照範囲を超えて実装可否を断定しない）。

## 6. スコープ外・起票案

### スコープ外（本イシューでは扱わない）

- ROCm/HIP 対応を REQ-2 の受け入れ基準に含める要件化（正本 `docs/spec/` 側の変更が必要であり、本リポでは提案のみ可能）
- `crates/tensor-core`・`crates/backend-cuda`・`crates/backend-metal` へのコード変更・実測

### 起票案（ユーザー承認後に起票する。本イシューでは起票しない）

1. `DeviceInfo::warp_width: Option<u32>` の追加と CUDA（`CU_DEVICE_ATTRIBUTE_WARP_SIZE`）・Metal（`threadExecutionWidth()` の再利用）実装
2. rmsnorm／softmax／mse カーネルの `#define WARP_SIZE` レンダリング時注入化と、既存ビット一致テストの回帰確認
3. cooperative 起動入口（`LaunchKind::Cooperative`）の `unsafe` 採用承認と、Stream-K fixup（#1357）との連携可否の評価

## 7. 出典

- `.claude/skills/amd-rocm/references/hip/porting-cuda-to-hip.md`（warp 幅・legacy driver/module API）
- `.claude/skills/amd-rocm/references/hip/cpp-language-extensions.md`（`warpSize`・`__shfl_xor()`）
- `.claude/skills/amd-rocm/references/hip/cooperative-groups.md`（`hipLaunchCooperativeKernel`）
- `docs/backend-switching-design.md`（cfg ベースバックエンド切替の既存設計）
- `docs/backend-matrix.md` §3.4（ROCm・Vulkan 対象外の既存記述）
- `docs/spec/04-requirements.md` REQ-2
- `docs/cuda-streamk-decision.md`・`docs/perf/cuda-gemm-tiled-pipeline-persistent.md`（persistent／Stream-K 実装の現況）
- `docs/backend-cuda-async-execution-design.md`・`docs/backend-metal-command-batching-design.md`（stream・コマンドバッファの既存同期契約）
- `cudarc =0.19.8`（`src/driver/safe/launch.rs` の `launch_cooperative` シグネチャ。Cargo.lock 記載バージョン）
