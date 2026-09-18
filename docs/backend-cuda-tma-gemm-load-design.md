# TMA（cp.async.bulk.tensor）による GEMM ロード経路の設計（Phase B・イシュー #1589）

## 0. 位置づけ・スコープ

- **対応イシュー**: #1589（GEMM 性能改善ツリー・ルート #479・Phase B 親 #490 配下）。#490 本文が「A-3（#483）の TMA プローブ成功を条件に B-12（TMA producer/consumer パイプライン設計）・B-13（試作カーネル接続）・B-14（A/B 計測と採否）を条件付き起票する」としていた運用のうち、**B-12 相当（設計確定）を本 issue が担う**。
- **前提が満たされた経緯**: #483（`crates/backend-cuda/tests/tma_probe_real_device.rs`）は #1574（低レイヤー診断・2026-09-12）で初めて DGX Spark GB10 実機を通過し、cluster／cta 両 variant が `compute_121`／`compute_121a`／`compute_121f` の全組み合わせで NVRTC コンパイル成功、両 variant とも実行プローブ（`compute_121` を選択）で 16×16 タイル転送が bit 一致した（出典: `docs/perf/logs/lowlayer-diagnosis-2026-09-12/dgx/tma_probe_real_device.log`。§8 に全文転記）。`docs/cuda-tensor-core-design.md` §12 の判定基準に照らすと「cluster・cta のいずれか一方でもコンパイル・実行が成功」＝**起票要**が確定する。
- **本文書は設計のみ**であり、カーネル・ホストコードの実装は含めない（受入基準 1）。実装着手前に必要な承認事項は §5 に列挙し、後続実装 issue（B-13／B-14 相当。本 issue では起票しない）の前提として残す（受入基準 2）。
- **契約**: 数値一致は run-to-run bit 同一。tolerance・baseline の変更は対象外（変更が必要になれば別途ユーザー承認。`.claude/rules/coding-rust.md`）。
- **非信頼データの扱い**: Issue 本文中に命令文は含まれていなかった（矛盾指示なし）。外部 artifact（CUTLASS ソース調査等）は出典として転記するのみで内容を取得・実行しない。

## 1. 背景・目的

- 現状の CUDA f32 本番経路は `ops.rs::CudaBackendOps::gemm` → `context_cache::cached_gemm` → `CudaGemm::run_tiled_f32` → `select_tiled_f32_kernel`（`tiled_f32_kernel_kind`: `n%4==0 && k%4==0` かつ A オフセット 4 要素整列で `Pipeline`、それ以外は classic `TILED_F32`）→ `kernels_tiled_pipeline.rs`（64×64×16・3 stage cp.async）／`kernels_tiled_pipeline_128x64.rs`（N≥1024 かつ K≥1024 で選択・8×4 レジスタブロック・A フラグメント XOR スウィズル）。f16 は `TypedOps<f16>`（#1797）→ `CudaGemmAuto::run_f16` → `kernels_mma.rs`（64×128×32・3 stage・cp.async + ldmatrix）。
- 期待効果は限定的であることを正直に記す: `docs/perf/cuda-gemm-reuse-phase-breakdown.md` で matmul カーネル単体は candle fresh を既に上回っており（N=1024 で 1.59 倍・N=4096 で 1.47 倍）、candle 比ゲート未達（`docs/perf/cuda-gemm-candle-gate-remeasurement.md` §16: N=1024 0.429 倍・N=2048 0.496 倍・N=4096 1.478 倍〈達成〉）は主に固定費（H2D／D2H・ホスト側オーケストレーション）由来であり、カーネル自体のロード命令削減がゲート達成を直接には約束しない。
- TMA の狙いは (a) cp.async ループのロード発行命令数削減（現行はスレッドごとに複数回の `cp.async.cg.shared.global` を発行）、(b) アドレス計算・境界チェックのハードウェアオフロード（`cuTensorMapEncodeTiled` で事前検証済みの境界を hardware が解釈）、(c) レジスタ圧の低減（ロード用インデックス変数の削減）である。これらはカーネル純粋実行時間の改善候補であり、固定費支配的な現状の candle 比ゲートへの寄与は間接的・限定的と見込む。

## 2. 前提・制約（調査で確定した事実）

| # | 事実 | 出典（`file:line`） | 設計への影響 |
|---|------|------|-------------|
| F1 | cta／cluster 両 variant が 3 arch（`compute_121`／`121a`／`121f`）で NVRTC コンパイル成功。両 variant とも `compute_121` を選択して実行し 16×16 タイル転送が bit 一致（`bitwise_match=true`） | `docs/perf/logs/lowlayer-diagnosis-2026-09-12/dgx/tma_probe_real_device.log`（§8 に全文） | `shared::cta`（クラスタ launch 不要・単一 CTA 内で完結）を採用する。cluster／multicast は `ClusterShape=1x1x1` 前提の範囲を出ないため不採用（CUTLASS の SM120 系譜が `shared::cta` opcode を発行する設計であることは `docs/cuda-tensor-core-design.md` §12 冒頭「CUTLASS 側の根拠」で既に確認済み） |
| F2 | `setmaxnreg` プローブ（#484・`docs/cuda-tensor-core-design.md` §13）は GB10 実機実測が本 issue 時点でも「未了」のまま | `docs/cuda-tensor-core-design.md:341`（§13「実機実行: 未了」節） | producer/consumer の非対称レジスタ配分・warp specialization に依存しない設計とする。対称レジスタ・単一 elected thread が TMA を発行し、既存ループ末尾の `__syncthreads()` で WAR（write-after-read）を保証する保守的構成を初期形とする。producer/consumer 分離は `setmaxnreg` 使用可否が確定した後の後続候補として §6 に記録する |
| F3 | `cuTensorMapEncodeTiled`・`CUtensorMap`（`sys::CUtensorMap`。`align(128)`・`opaque:[u64;16]`）・`CUtensorMapSwizzle::{CU_TENSOR_MAP_SWIZZLE_NONE,...}`・`CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_FLOAT32`／`_TFLOAT32` は cudarc 0.19.8（`cuda-13000` feature）の既存 `driver::sys` API | `crates/backend-cuda/tests/tma_probe_real_device.rs:501-518,610-624`（`TensorMapArg` ラッパー・`cuTensorMapEncodeTiled` 呼び出し） | 新規依存の追加なし（`.claude/rules/deps-policy.md` 対象外）。ただし raw FFI の呼び出しは `unsafe`（§5 承認事項 1） |
| F4 | `CUtensorMap` はカーネル引数として `__grid_constant__` 値渡し。プローブは `#[repr(transparent)]` のマーカーラッパー型 `TensorMapArg` に `unsafe impl cudarc::driver::DeviceRepr` を付与し、cluster 版は `cuLaunchKernelEx`（クラスタ属性指定のため raw FFI）・cta 版は `cudarc::driver::result::launch_kernel`（`cuLaunchKernel` の薄いラッパー。ともに `unsafe`）で起動する。cudarc の safe `launch_builder()` は本プローブでは使用していない | `crates/backend-cuda/tests/tma_probe_real_device.rs:507-518,637-650,841-861` | 採用する cta（クラスタ属性不要・単一 CTA 起動）は、`PushKernelArg<&T: DeviceRepr>` を満たす `TensorMapArg` を safe `launch_builder().arg(&wrapper)` で起動できる見込みだが、プローブ自体はこの経路を検証していない。**実機確認項目**として §6 の事前登録ゲートへ含める。`unsafe impl DeviceRepr` 自体は既存プローブと同じ理由（マーカートレイト・メソッド非公開・値の中身は driver 側 ABI 契約）で妥当と見込むが、本番導入時に改めて security-auditor 監査が要る（§5 承認事項 1） |
| F5 | 現行 `kernels_tiled_pipeline.rs`（64×64）は `__shared__` 静的配列 + 行パディング（`TP_A_PAD = TP_BK + 4`・`TP_B_PAD = TP_BN + 4`）でバンク衝突を回避する。`kernels_tiled_pipeline_128x64.rs`（128×64）は既にパディングではなく **A のみ XOR スウィズル**（`swz(row, chunk) = chunk ^ ((row >> 3) & 3)`。16 バイトチャンク単位の行内全単射）へ移行済み（B は現状 `SWIZZLE_NONE` 相当） | `crates/backend-cuda/src/kernels_tiled_pipeline.rs:143-146`（`TP_A_PAD`/`TP_B_PAD`）／`crates/backend-cuda/src/kernels_tiled_pipeline_128x64.rs:56-76,298-300`（XOR スウィズル設計コメント・`swizzled_chunk_a`） | TMA は tensor map が記述する box を smem へ**密に**（パディングなしで）書き込むため、行パディングによる衝突回避（64×64 版）は使えなくなる。ハードウェア swizzle モード（`CU_TENSOR_MAP_SWIZZLE_{32B,64B,128B}`）への置換が必要。128×64 版は既に XOR スウィズルへ移行済みのため、その導出過程（アドレスビット XOR とバンク差の対応）をハードウェア swizzle モードの選定根拠に転用できる見込み |
| F6 | GB10: SM 数 48・`MAX_SHARED_MEMORY_PER_BLOCK_OPTIN` 101,376 B・静的（非 opt-in）smem 上限 49,152 B | `docs/perf/sm121-device-attributes.md:60-61,85` | 密レイアウト時の per-stage 使用量: 64×64 版 `(BM*BK + BK*BN)*4 = (64*16+16*64)*4 = 8,192 B`（`TP_SMEM_BYTES_PER_STAGE` の非パディング相当。`STAGES=3` で 24,576 B）・128×64 版は既に密（`TP128_SMEM_BYTES_PER_STAGE = (128*16+16*64)*4 = 12,288 B`。`STAGES=3` で 36,864 B。`crates/backend-cuda/src/kernels_tiled_pipeline_128x64.rs:109,201`）。いずれも静的上限 49,152 B 以内に収まる見積り（swizzle アトム整列によるパディングを追加してもこの余裕は残る想定）。実測での確認は §6 の事前登録ゲートへ含める |
| F7 | `tiled_pipeline_alignment_ok`（`n%4==0 && k%4==0`）・`tiled_pipeline_offset_aligned`（A オフセット 4 要素整列）が既存の cp.async 適用ゲート | `crates/backend-cuda/src/gemm.rs:685,710,789-797` | TMA の global 制約（行ストライド 16 B 倍数・ベースアドレス 16 B 整列・box 内側次元 16 B 倍数）は既存ゲートと同じ整列単位（f32 4 要素＝16 B）に対応する。ゲート式自体の追加緩和は不要と見込む。実際の `cuTensorMapEncodeTiled` 制約（ストライド境界の詳細）との厳密な突合は実装時に行う |
| F8 | 既存の opt-in カーネル変種（persistent #1346／Stream-K #1358／128×64 #1343）は `internal-diagnostics` feature 限定の `compile_tiled_pipeline_*_variant` API と、計算本体文字列（`TP_TILE_CORE` 等）の共有により bit 同一を機構的に担保する設計パターンが確立済み | `crates/backend-cuda/src/gemm.rs:859,965,1029-1032,1132-1148,1470,1671-1675` | TMA 版も同型: ロード段（cp.async → TMA）のみ差し替え、計算本体（アキュムレータのレジスタブロック計算）は既存カーネルと文字列共有し bit 同一を機構的に保証する |
| F9 | TF32 の `mma.sync` 経路（`kernels_mma_tf32.rs`）は #839 で凍結継続中 | `docs/cuda-tensor-core-design.md` §15 冒頭 | TF32 経路への TMA 適用（`CU_TENSOR_MAP_DATA_TYPE_TFLOAT32` の存在自体は F3 で確認済み）は「データ型の存在確認のみ・暗黙変換の意味論は未検証・凍結中のため本設計の対象外」と明記する |
| F10 | 低レイヤー診断（#1574）の issue ツリー草案（§7）が本 issue（#1589）を A-6 行として指す | `docs/perf/lowlayer-diagnosis-2026-09-12.md` §7 | 相互参照を追記する（§9） |

## 3. スコープ・対象カーネル（判断）

- **主対象（Stage 1）**: FP32 SIMT `kernels_tiled_pipeline.rs`（64×64×16）。理由: facade 公開面（f32 固定 dispatch）の本番既定経路であること・`TP_TILE_CORE` 共有による bit 同一の機構的担保が既に確立していること（F8）・opt-in 変種の追加パターンをそのまま流用できること。
- **Stage 2**: `kernels_tiled_pipeline_128x64.rs`（本番で N,K≥1024 の形状を担う）へ横展開。128×64 版は既に XOR スウィズルへ移行済みのため、TMA のハードウェア swizzle モードへの置換はこちらの方が設計上の連続性が高い（F5）。
- **Stage 3（#490 の元の B-12 文言が指していた対象）**: f16 `mma.sync`（`kernels_mma.rs`）。`ldmatrix` 向けの 128B swizzle 適合・B 行（256 B/行）の box 分割が必要になるため、Stage 1／2 の実測結果（特に swizzle モード選定・smem 収支）を見てから着手する。#490 が当初 `kernels_mma.rs` を名指ししていた点との差異と、本設計が f32 経路を先行させる理由をここに明記する。
- **対象外**: TF32 経路（F9・凍結中）・cluster／multicast スコープ（F1）・warp specialization／`setmaxnreg` 依存設計（F2）・Metal／CPU バックエンド。

## 4. 設計の要点

### 4.1 命令列・パイプライン

- 命令: `cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes`（プローブ検証済み opcode）。付随して `mbarrier.init`（期待到着数 1）／`mbarrier.arrive.expect_tx`／`mbarrier.try_wait.parity` ループ／`fence.proxy.async.shared::cta`（generic proxy → async proxy の順序制約）を使う。
- 構造: 現行の「prologue で `STAGES-1` 段先行発行 → 本体ループで `wait` → compute → 次段発行 → `__syncthreads()`」という骨格（`kernels_tiled_pipeline.rs` の cp.async 版と同型）を維持しつつ、次の 2 点のみを差し替える:
  1. 各段の A・B ロードは、現行の「各スレッドが `cp.async.cg.shared.global` を複数回発行するループ」を、「`threadIdx.x==0 && threadIdx.y==0` の 1 スレッド（elected thread）のみが A box・B box 各 1 回ずつ `cp.async.bulk.tensor` を発行する」へ置き換える（`mbarrier.arrive.expect_tx` に渡す期待バイト数は A box + B box の合計）。
  2. 現行の `cp.async.wait_group STAGES-2` を、「段 `t % STAGES` に対応する full mbarrier を parity `(t / STAGES) & 1` で待つ」へ置き換える。
- WAR（write-after-read）保証は既存ループ末尾の `__syncthreads()` に委ねる保守的構成を初期形とする（producer/consumer 分離による非同期オーバーラップの深追いは F2 の理由により見送る）。
- mbarrier は段ごとに 1 本（`__shared__ uint64_t full[STAGES];`）を持つ。ポーリングには上限回数を設け（プローブと同様のタイムアウト機構）、上限到達時は失敗フラグを立ててハングを fail-closed に顕在化させる（silent green を許さない。`.claude/rules/coding-rust.md` の品質基準に整合）。

### 4.2 smem レイアウト・swizzle 選定

- box 定義: A = `[BM 行][BK=16 f32]`（内側 64 B）・B = `[BK=16 行][BN=64 f32]`（内側 256 B。64×64／128×64 とも `BN=64` で共通）。
- **A（64×64 版・行差 2 行 = 4 行 stride = 256 B。128×64 版・行差 8 行 = 512 B）**: 密レイアウト化によりパディングが失われるため、`CU_TENSOR_MAP_SWIZZLE_64B`（64×64 版の候補）・`CU_TENSOR_MAP_SWIZZLE_128B`（128×64 版の候補）を初期候補として、アドレスビット XOR とバンク差の対応（128×64 版が既に採用している `swz(row, chunk) = chunk ^ ((row>>3)&3)` の導出過程。F5）を基に机上導出する。**この導出は仮説であり実機の bit 一致テスト・純カーネル時間実測を経て最終選定を確定する**（推測値を確定事項として書かない）。
- **B（内側 256 B）**: 128 B swizzle のスパン（128 B）を超えるため初期形は `CU_TENSOR_MAP_SWIZZLE_NONE`（現行 B 読みパターンの衝突特性を変えない）とする。box を 128 B×2 分割してハードウェア swizzle を適用する案は代替候補として記録するに留める。
- 各 stage の smem バッファは swizzle アトム境界（最大 1024 B）へ `__align__` する。F6 の見積りに従えば静的 smem（49,152 B）の範囲に収まる想定であり、`cuFuncSetAttribute` による動的 smem opt-in は原則不要と見込む。超過が判明した場合のみ動的 smem 化を検討する（§5 承認事項 3 の対象）。

### 4.3 REQ-8（境界検査）との整合

- ハードウェアの OOB ゼロ充填（`CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE`／`_NAN` 等の選択）は現行の `src_size=0` ゼロ充填と同じ意味論を意図するが、`.claude/rules/coding-rust.md`「性能下限・最適化の達成を理由に手動境界チェックを省略しない」を守るため:
  1. エピローグの guarded store（出力書き込み時の境界チェック）は不変のまま維持する。
  2. tensor map の `globalDim`／`globalStrides` は、ホスト側で検証済みの `m`／`n`／`k`（`InvalidShape` による fail-closed・`gemm.rs` の既存 `validate_tiled_pipeline_k_bound` と同型の i32 上限検査）からのみ生成する。外部入力（形状）を直接 `cuTensorMapEncodeTiled` へ渡さない。
  3. カーネル内で発行前にタイル起点座標のガード（`block_row0 < m` 等の既存ガードと同型）を維持する。
- **推測で埋めない検証項目**: 部分 OOB の box であっても `mbarrier.arrive.expect_tx` に渡す `complete_tx` バイト数が box 全体のサイズと等しいままか（PTX ISA・実機実行で確認する。仮に box 全体分を要求するなら、境界タイルでの mbarrier 待機ロジックに影響しうる）。

### 4.4 ホスト側: tensor map の生成・寿命・受け渡し

- `cuTensorMapEncodeTiled` はホスト固定費（呼び出しごとのオーバーヘッド）を伴うため、`CudaGemm` 内に上限付きキャッシュを設ける。キーはエンコード入力（デバイスポインタ・`globalDim`・`globalStrides`・box 形状・swizzle モード・dtype）の組。tensor map はキーの純関数であるため、メモリプール（`SizeClassPool`）の再利用で同一デバイスポインタが別テンソルとして再登場しても、キーが変われば別エントリとして扱われ安全である。
- カーネル引数は `#[repr(transparent)]` ラッパー + `unsafe impl DeviceRepr`（プローブと同型）を、safe `launch_builder().arg(&wrapper)` で渡す方針を第一候補とする（F4 の実機確認項目）。tensor map を global メモリに常駐させ `prefetch.tensormap` で先読みする案は実装複雑度に対して得られる効果が不明瞭なため不採用とする。
- 適用ゲートは既存の `tiled_pipeline_alignment_ok`／`tiled_pipeline_offset_aligned`（F7）と同一の整列条件を流用する。NVRTC コンパイル失敗時は既存の cp.async 版・classic 版への fail-closed フォールバックを維持する。

### 4.5 数値契約

- 出力は現行 cp.async 版と **bit 同一**であることを目指す（計算本体文字列の共有・オペランド同一・累積順序同一。F8）。tolerance／baseline／REQ-2 判定は不変。
- 後続実装（B-13 相当）へ引き渡す検証構成: Linux で実行可能な静的テスト（計算本体文字列の共有検査・REQ-8 ガード文字列の残存検査・swizzle 選定定数のドリフト検出）と、実機 `#[ignore]` bit 一致テストの 2 層。既存の `compile_tiled_pipeline_*_variant` 系の自己検証パターン（F8）をそのまま踏襲する。

## 5. 承認事項（受入基準 2）

実装着手前にユーザー承認が必要な事項を列挙する。本 issue ではこれらの承認取得・実装は行わない。

1. **本番コードへの新規 `unsafe` の追加**: `cuTensorMapEncodeTiled` raw FFI 呼び出し・`unsafe impl DeviceRepr`（`TensorMapArg` 相当のラッパー型）。`cuFuncSetAttribute`（動的 smem opt-in が必要になった場合のみ）も対象。`.claude/rules/security.md` に従い security-auditor 監査必須。
2. **B-13（試作カーネル接続）・B-14（A/B 計測と本番採否）相当の後続 issue 起票**: `.claude/rules/out-of-scope-tracking.md` に従い、本 issue のクローズ後にユーザー承認を得たうえで起票する（本 issue では起票しない）。
3. **本番結線の可否判断**: GB10 実機での事前登録ゲート（§6）を全通過するまで、実装は `internal-diagnostics` feature 限定の opt-in（F8 と同型）に留める。`select_tiled_f32_kernel`／`CudaGemm::new` 等の本番選択ロジックへは結線しない。
4. **tolerance・baseline の変更なし**: 本設計の適用によって数値一致契約・REQ-2 判定式・baseline 定数を変更する必要が生じた場合は、それ自体を別途ユーザー承認事項として扱う（`.claude/rules/coding-rust.md`）。

## 6. 後続への申し送り・事前登録ゲート（B-13／B-14 相当）

既存の opt-in 変種評価（#1344 の A〜D・#1358 の A〜D）と同形の事前登録ゲートを、後続実装の受入基準として申し送る。

- **ゲート A（数値契約）**: 実機 bit 一致（cp.async 版 vs TMA 版。N=256〜4096・端あり形状を含む・NN／NT／TN／TT の 4 転置パターン）。
- **ゲート B（parity 非後退）**: `tests/parity_nonregression.rs` 等の既存 parity テストが、既知の無関係な fail を除き 0 fail のまま。
- **ゲート C（純カーネル時間）**: GPU-only 計測（`--phases` 相当の GPU-only 区間切り出し）による 5 回中央値比較。採用条件は「N≥1024 のいずれかの形状で 1.05 倍以上の改善」かつ「全計測形状で 1.00 倍未満（後退）がないこと」。
- **ゲート D（本番ディスパッチ非後退）**: 結線後の同一 HEAD base/after を framework-compare gemm cuda で比較し、checksum 完全一致・非後退を確認する。
- **no-go 条件**: ゲート A の不一致が 1 件でもある場合、またはゲート C で後退する形状が 1 件でもある場合は REJECT とし、opt-in 実装のみを維持する（既存の #1358 Stream-K・#1347 persistent 版と同じ判断様式）。
- **実機到達性**: 本 issue の実装セッションは DGX Spark GB10 実機への到達手段を持たない（`docs/real-hardware-verification-env.md` の到達可否と同型の制約）。上記ゲートの実測は後続実装 issue（承認後起票）が到達可能な環境で行う。

## 7. リスクと安全側判断

- **F2（`setmaxnreg` 未実測）を理由に対称レジスタ設計を選ぶ**: producer/consumer 分離による性能上振れの機会を見送るが、`setmaxnreg` 拒否時にフォールバックが破綻するリスクを避ける安全側の判断。`setmaxnreg` 実測確定後に非対称設計を再検討できるよう、本設計は対称レジスタ版が独立して成立する構成とする。
- **swizzle モード選定（§4.2）は机上導出のみ**: 実機での bit 一致・性能検証なしに本番へ反映しない。導出過程自体は 128×64 版の既存 XOR スウィズル実装（F5）と整合させ、無根拠な新規パラメータを持ち込まない。
- **F9（TF32 凍結）に触れない**: `CU_TENSOR_MAP_DATA_TYPE_TFLOAT32` の存在確認に留め、暗黙変換の意味論検証や `kernels_mma_tf32.rs` への適用は本設計・後続 B-13/B-14 のいずれのスコープにも含めない。

## 8. 出典・実測ログ全文

`docs/perf/logs/lowlayer-diagnosis-2026-09-12/dgx/tma_probe_real_device.log`（GB10 実機・2026-09-12・#1574 実施分）:

```text
    Finished `release` profile [optimized] target(s) in 0.03s
     Running tests/tma_probe_real_device.rs (target/release/deps/tma_probe_real_device-367b695ccec3673b)

running 3 tests
test tma_execution_probe ... tma_compile_probe variant=cluster arch=compute_121 result=success (selected for execution probe)
tma_execution_probe variant=cluster arch=compute_121 result=success tile=16x16 global=64x64 bitwise_match=true
ok
test tma_execution_probe_cta ... tma_compile_probe variant=cta arch=compute_121 result=success (selected for execution probe)
tma_execution_probe_cta variant=cta arch=compute_121 result=success tile=16x16 global=64x64 bitwise_match=true
ok
test tma_nvrtc_compile_probe ... environment: name="NVIDIA GB10" compute_capability=(12, 1) arch="compute_121"
tma_compile_probe variant=cluster arch=compute_121 result=success
tma_compile_probe variant=cluster arch=compute_121a result=success
tma_compile_probe variant=cluster arch=compute_121f result=success
tma_compile_probe variant=cta arch=compute_121 result=success
tma_compile_probe variant=cta arch=compute_121a result=success
tma_compile_probe variant=cta arch=compute_121f result=success
ok

test result: ok. 3 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.99s
```

環境: `name="NVIDIA GB10" compute_capability=(12, 1)`（内部ホスト名は含めない）。関連: `docs/perf/lowlayer-diagnosis-2026-09-12.md` §7（issue ツリー草案）。

## 9. スコープ外事項（既知の限界。実装は行わない）

- `crates/backend-cuda/tests/tma_probe_real_device.rs` 冒頭コメントの「本ファイルのカーネルソース・`cuTensorMapEncodeTiled` 呼び出しパラメータは実機コンパイル・実行を一度も通過していない」という記述は、#1574 の実測（本文書 §1・§8）により陳腐化している。本 issue は docs-only 方針のためこのコメントの是正（実装ファイルへの変更）は行わず、後続実装 issue（B-13 相当）でのコメント更新に引き継ぐ。
- 本文書の swizzle モード選定・smem 収支見積り（§4.2・F6）は机上導出・実測前の仮説であり、実機値で置き換わるまでは確定事項として扱わない。

## 10. 実装記録（イシュー #1975。段階的承認の C 案〈`internal-diagnostics` 限定・本番非結線・PR 内 security-auditor 監査・tolerance／baseline／依存不変・GB10 検証は #1976〉に基づく Stage 1 opt-in 実装）

以下は本 issue で実装した内容と、時間制約により実装計画から縮小したスコープを記録する。

### 10.1 実装ファイル・シンボル

- `crates/backend-cuda/src/kernels_tiled_pipeline.rs`（追記のみ。既存 `TP_TILE_CORE` 等の断片は無変更）: `TP_TMA_A_BOX_BYTES`／`TP_TMA_B_BOX_BYTES`／`TP_TMA_EXPECT_TX_BYTES`／`TP_TMA_SMEM_ALIGN`／`TP_TMA_POLL_LIMIT`（Rust 定数＋const assert 群）・`TmaSwizzleA`（`None`／`B64` の 2 値 enum）・`render_tma_defines`／`render_tma_source`・`TP_TMA_HELPER`（`CUtensorMap` typedef・`TP_TMA_A_AT` マクロ）・`TP_TMA_PREFIX`／`TP_TMA_TILE_CORE`／`TP_TMA_SUFFIX`（カーネル本体）・`tiled_pipeline_tma_f32_source`（本番結線相当の既定段数固定アクセサ。`pub fn`・無条件コンパイル）・`tma_swizzled_chunk_a`（B64 仮説のホストモデル。**`#[cfg(test)]` 限定**）・`tiled_pipeline_tma_f32_source_with_stages`（**`#[cfg(test)]` 限定**。§10.4 参照）。静的テスト 9 件（`mod tests` 内。GPU 非依存・通常 CI で実行）。
- `crates/backend-cuda/src/gemm.rs`: `#[cfg(feature = "internal-diagnostics")] mod tma_tiled_pipeline { ... }`（`TensorMapArg`〈`DeviceRepr` newtype〉・`TmaBoxSpec`＋`encode_tensor_map_2d_f32`〈`cuTensorMapEncodeTiled` FFI 呼び出し〉・`TmaTiledPipelineFunction`・`CudaGemm::compile_tiled_pipeline_tma_variant`／`launch_tiled_pipeline_tma_f32`／`run_tiled_pipeline_tma_f32`）。GPU 非依存単体テスト 3 件。
- `crates/backend-cuda/src/lib.rs`: `#[cfg(feature = "internal-diagnostics")] pub use gemm::TmaTiledPipelineFunction;`・`#[cfg(feature = "internal-diagnostics")] pub use kernels_tiled_pipeline::TmaSwizzleA;`（`PersistentTiledPipelineFunction`・`StreamKTiledPipelineFunction` と同一の公開面ゲート方針）。
- `crates/backend-cuda/tests/cpu_cuda_tiled_pipeline_tma_parity.rs`（新規）: `#[ignore]` 実機テスト 5 件（`None` 腕の bit 一致・k=0 no-op・決定性・事前転置入力・`B64` 腕の仮説検証〈不一致を記録するのみで CI 失敗にしない〉）＋環境適応スモーク 1 件（`#[ignore]` なし・通常 CI で実行・CUDA/NVRTC/CC9.0 未満は早期 return）。
- `crates/backend-cuda/Cargo.toml`: `[[test]] name = "cpu_cuda_tiled_pipeline_tma_parity"`・`required-features = ["internal-diagnostics"]`（既存の persistent／Stream-K 版と同一パターン）。
- `crates/backend-cuda/tests/tma_probe_real_device.rs`: 冒頭コメントの「実機コンパイル・実行を一度も通過していない」という陳腐化した記述を、#1574 実測済みの事実（§1・§8）へ是正。§9 の申し送りを解消。

### 10.2 座標系・swizzle・REQ-8

- 座標系は**要素座標・内側次元先行**（計画 §2 の調査どおり）: A は `globalDim=[k,m]`・`boxDim=[TP_BK,TP_BM]`・座標 `{t*TP_BK, block_row0}`、B は `globalDim=[n,k]`・`boxDim=[TP_BN,TP_BK]`・座標 `{block_col0, t*TP_BK}`。
- swizzle は `None`（恒等アクセス。正しさ確認用の fail-safe ベースライン）・`B64`（`chunk ^ ((row>>1)&3)` の仮説式。物理列 `= swz*4 + (kk%4)`）の 2 値をパラメータ化した。B（`bs_tile`）は行幅 256B が 64B swizzle アトムを跨ぐため常に `None` 相当（swizzle 対象外）に固定する。
- REQ-8: プロローグは `s < num_k_tiles`（タイル境界）に加え `block_row0 < m && block_col0 < n && kt < k`（タイル起点ガード）を発行前に検査する。エピローグは既存 `if (r < m && cc < n)` の guarded store を不変維持し、mbarrier ポーリングが `TP_TMA_POLL_LIMIT` 回で完了しなかった場合は `timed_out` フラグをスティッキーに立てて以後の待機・発行をスキップし（無期限ハング防止）、エピローグで NaN センチネル（0x7fc00000）を書く（`tests/tma_probe_real_device.rs` と同じ fail-closed 方式。bit 一致テストで確実に検出される）。
- 本番結線（`select_tiled_f32_kernel`／`CudaGemm::new`）は本節のカーネルを一切参照しない（`internal-diagnostics` feature 限定・opt-in API のみ）。既存の固定ハッシュテスト（`EXPECTED_NON_PERSISTENT_FRAGMENTS_FNV1A64` 等）は無変更のまま green（本番ソース不変の機械的証拠）。

### 10.3 新規 `unsafe` の一覧（PR 内 security-auditor 監査対象。承認済みスコープ内）

1. `unsafe impl cudarc::driver::DeviceRepr for TensorMapArg {}`（`crates/backend-cuda/src/gemm.rs::tma_tiled_pipeline`）。
2. `sys::cuTensorMapEncodeTiled(...)` の FFI 呼び出し（`encode_tensor_map_2d_f32`）。
3. `self.stream.launch_builder(&func.func)...launch(cfg)` の safe `launch_builder` 経路（`unsafe` ブロックは既存カーネル群と同一パターン。raw `cuLaunchKernel(Ex)` へは到達しない — 計画 §3.3 が「低リスク」と見込んだ safe `launch_builder().arg(&wrapper)` 経路をそのまま採用できた）。

`cuFuncSetAttribute`（動的 smem）・`mem::zeroed`（`CUtensorMap { opaque: [0u64; 16] }` の明示ゼロ初期化で代替）は使わない。

### 10.4 計画からのスコープ縮小（時間制約による）

- **意味論プローブ 3 件**（要素座標の非ゼロ確認・部分 OOB box の fill 意味論・`B64` swizzle の smem 物理配置ダンプ）は未実装。`tma_swizzled_chunk_a` を `#[cfg(test)]` 限定に留めた（外部プローブから参照するには `pub` へ戻し `lib.rs` から re-export する作業が必要。#1976 へ引き継ぐ）。
- **`tiled_pipeline_tma_f32_source_with_stages`**（任意段数）は実装したが、`compile_tiled_pipeline_tma_variant` は既定 `TP_DEFAULT_STAGES` 固定のみを呼ぶ（persistent／Stream-K 版のような段数比較ベンチ example を追加しなかったため、`_with_stages` を本番結線から呼ぶ経路が存在しない）。`#[cfg(test)]` 限定として dead-code を回避した。
- **ベンチ example への `--tma off|none|b64` 列追加**（計画 Step 6）は未実施。
- **tensor map のキャッシュ**（計画 §3.3 の「起動ごとにエンコード」からの改善候補）は未実装のまま。`launch_tiled_pipeline_tma_f32` は毎起動 `encode_tensor_map_2d_f32` を呼ぶ（GPU-only 区間の計測ではホスト側 encode 費用は含まれない）。
- **転置パターン（NT/TN/TT）**: 計画 §3.5 のとおり、事前転置済みホスト入力を NN として渡すケース 1 形状のみを検証する（`tiled_pipeline_tma_none_matches_pipeline_with_pretransposed_host_input`）。API レベルで転置を扱う専用入口は追加していない。

### 10.5 検証状態

- CI ゲート（`.claude/rules/coding-rust.md` が要求する `cargo clippy --workspace --all-targets --all-features -- -D warnings`）は green（新規警告ゼロ）。`cargo fmt --all --check`・`cargo check --workspace --all-features`・`cargo test -p fandhe-ai-backend-cuda --all-features`（1042 件 pass・新規 `#[ignore]` 5 件・環境適応スモーク 1 件 pass）も green。
- **`internal-diagnostics` feature なしの `cargo build`／`cargo clippy` は dead-code 警告が 14 件増える**（`TP_TMA_*` 定数・`TmaSwizzleA`・`render_tma_defines`／`render_tma_source`・`TILED_PIPELINE_TMA_F32_SOURCE_{NONE,B64}`・`tiled_pipeline_tma_f32_source`）。これは新規に持ち込んだ欠陥ではなく、同ファイル内の既存 opt-in カーネル群（`TP_KERNEL_PERSISTENT_PREFIX`・`tiled_pipeline_persistent_f32_source`・`tiled_pipeline_streamk_f32_source` 等。`compile_tiled_pipeline_persistent_variant` 等の呼び出し元自体が `internal-diagnostics` 限定のため feature なしでは呼び手が存在しない）が既に持つのと同種・同数量級の dead-code である（`git stash` 比較で確認: feature なし clippy はベースライン 62 件 → 本実装後 76 件）。計画 §3.1 の対処案（`#[cfg(any(test, feature = "internal-diagnostics"))]` でカーネルソース関数自体をゲートする案）は、他の兄弟カーネル族との一貫性を優先しあえて適用しなかった（実装判断・§10.4 のスコープ縮小とは別軸）。**実際の CI 品質ゲートは `--all-features` 付きのため、この dead-code は CI を破壊しない**。
- GB10 実機での `#[ignore]` テスト実行・意味論プローブ・純カーネル時間実測は **未実施のまま #1976 へ引き継ぐ**（`docs/perf/logs/cuda-tma-stage1-1975/README.md` にランブックを整備済み）。

### 10.6 レビュー是正（PR 内自己レビューでの指摘 3 件。コミット `82c02615`）

1. **`k == 0` の fail-closed バグ**: `TmaBoxSpec::validate` がゼロ次元を拒否するため、当初実装の `run_tiled_pipeline_tma_f32`／`launch_tiled_pipeline_tma_f32` は `k == 0` で `InvalidShape` を返していた（§3.3 の「`k == 0` はカーネル内 no-op へ委ねる」という当初記述が誤りだった）。`m == 0 || n == 0` の直後に `k == 0` の早期 return を追加（`launch_` は `c_dev` を明示 `memset_zeros`・`run_` は全ゼロ `Vec` を返す）。
2. **`TP_TMA_SMEM_ALIGN` を `128` から `1024` へ修正**: `B64` swizzle 仮説（`chunk ^ ((row>>1)&3)`）はハードウェアが smem **絶対**アドレスのビット [7,9) を [4,6) へ XOR するという前提に立ち、各段の A タイル先頭アドレスが 512 バイト整列でなければ成立しない。`__align__(128)` はそれより緩い制約（128 バイト整列）しか保証しないため、`1024` バイト整列＋`TP_TMA_A_BOX_BYTES` が 512 の倍数であることの const assert を追加し、整列崩れによる `B64` 仮説の誤帰属（#1976 の意味論プローブが「仮説不成立」と誤記録するリスク）を防いだ。
3. **`#[allow(clippy::too_many_arguments)]` の撤去**（§5 承認事項・計画 R7 で新規追加を明示的に禁止していた）: `launch_tiled_pipeline_tma_f32` の `m`/`n`/`k` を `dims: (u32, u32, u32)` へまとめ、引数 6 個（clippy 既定閾値 7 以下）に収めた。
