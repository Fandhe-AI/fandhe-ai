# CUDA Tensor Core（WMMA/mma）実装知見・制約 集約インデックス

- 対応イシュー: #65（TASK-11.1f、親 #59 TASK-11.1 の再分解サブタスク）
- 受け入れ条件: 「知見メモがコードコメント・ドキュメントに残されている」
- 位置づけ: 本文書は兄弟サブタスク（#60〜64・#186・#187）で得た知見・制約・未検証事項の**集約インデックス**である。詳細が既存ドキュメント・コードコメントに既にある項目は要約と参照のみを示し、重複記述しない（`.claude/rules/code-comment-style.md`「陳腐化しやすい実装詳細の重複を書かない」）。TASK-11.3（#70・`docs/matrix-unit-dispatch.md` への証跡整備）の材料として使うことを想定する。
- **本文書自体は新規実測・カーネル変更を一切含まない**。すべて既存ドキュメント・コードコメントからの集約である。

## 1. sm_121（GB10）固有制約

- **命令セット系譜**: SM12x（sm_120/121）は Ampere（SM80）以来の WMMA／`mma.sync` プログラミングモデルを維持し、データセンター系 Blackwell（SM100）の `tcgen05` や Hopper（SM90）の `wgmma` を要求しない。詳細根拠・出典は [`docs/cuda-tensor-core-design.md`](./cuda-tensor-core-design.md) 2 節を参照。
- **対応 fragment shape**: f16 は `m16n16k16`／`m32n8k16`／`m8n32k16`（累算 f16 または f32）、TF32 は `m16n16k8`（compute capability 8.0 以降）。sm_121 はいずれの要件も満たす（同 2 節・4.1 節）。
- **NVRTC の `compute_121` 受理可否**: **未検証**。`crates/backend-cuda` は `CudaContext` から取得した compute capability を `--gpu-architecture=compute_XY` に反映する構成のため機構自体は存在するが、sm_121 に対する実際のコンパイル成否は実機での NVRTC 実行でのみ確認できる（同 2 節）。#61（TASK-11.1b）着手時に優先解消すべき事項として記録されている（同 8 節「未検証事項」1）。

## 2. NVRTC 制約

### 2.1 ヘッダ非同梱と `<mma.h>` の実行時 include パス解決

- NVRTC は CUDA ヘッダを同梱しないため、`<mma.h>` を使う WMMA カーネル（[`kernels_wmma.rs`](../crates/backend-cuda/src/kernels_wmma.rs)）のコンパイルには `nvrtcCreateProgram` 呼び出し時に CUDA toolkit の include パスを渡す必要がある。
- **ビルド時 vs 実行時の切り分け**: NVRTC 呼び出しはバイナリのビルド時ではなく実行時に発生するため、**ビルド成立自体（`cargo build --workspace --locked`）は toolkit 非搭載環境でも保たれる**（`cudarc` 動的ロード契約、`.claude/rules/deps-policy.md`）。一方、**実行時に toolkit の include パスが解決できない環境では `<mma.h>` を使うカーネルのコンパイルが失敗する**。詳細は[設計メモ](./cuda-tensor-core-design.md) 3.2 節。
- **NVRTC プロビジョニング実測**（#186・TASK-11.1g）: 本リポジトリの CUDA toolkit 非搭載環境（pip 配布の `nvidia-cuda-*` wheel）では `crt/mma.h` が wheel に含まれないため、apt パッケージから `crt/mma.h` のみを取得し wheel の include 配下に重ね合わせる手順で NVRTC コンパイルを成立させた実績がある。**`LD_LIBRARY_PATH`／`CUDA_INCLUDE_PATH` はビルド時だけでなく実行時（バイナリ起動時）にも必要**（NVRTC は `CudaGemm::new`／`CudaWmmaGemm::new` 呼び出し時、すなわちバイナリ実行中にカーネルソースをコンパイルするため。ビルドと実行を別シェルセッションで行うと「NVRTC 非搭載」への誤 skip 判定を招く）。詳細手順は [`docs/perf/cuda-tensor-core-tolerance-evaluation.md`](./perf/cuda-tensor-core-tolerance-evaluation.md) 1 節「NVRTC プロビジョニング」を参照。

### 2.2 `NvrtcUnavailable` graceful skip 分岐（検証済み）

- CUDA driver は存在するが NVRTC が存在しない環境（本リポの複数実装セッションで確認: RTX 3060・compute capability 8.6）では `compile_ptx` が `CudaError::NvrtcUnavailable` を返す。この分岐は環境適応テスト（`crates/backend-cuda/tests/gemm_mma.rs` 等）・`cargo run -p backend-cuda --example gemm_mma_bench` の両方で実際に動作を確認済み（[`docs/perf/cuda-gemm-mma-pipeline.md`](./perf/cuda-gemm-mma-pipeline.md) 「本実装セッションで検証済みの事項」）。

### 2.3 静的共有メモリ per-block 48KiB 上限とタイル構成縮小の判断

- 全 compute capability 共通の静的共有メモリ上限（per-block 48KiB。動的共有メモリ opt-in `cudaFuncSetAttribute` を追加で呼ばない限り超過するとコンパイル・起動が失敗する）に対し、`mma.sync`/`ldmatrix`/`cp.async` パイプライン（#187）の実装計画候補値（ブロックタイル 128×128・BK=32・3 ステージ）は `(128×32+32×128)×2B×3 ≈ 49152B ≈ 48KiB` とほぼ上限に達し、コンパイル検証ができない環境では危険側に倒れる。よって当初 `BM=32`・`BN=64`・`BK=32`・3 ステージ（共有メモリ 18432B ≈ 18KiB）に縮小した（[`kernels_mma.rs`](../crates/backend-cuda/src/kernels_mma.rs) 冒頭コメント「タイル構成」）。
- **B-3（#494）でのブロックタイル拡大**: B-2（レジスタブロッキング。#493）でブロックスレッド数が 512→128 へ減り `cp.async` 協調ロードの並列度が低下したため、`BK=32` を維持したまま `BM=64`・`BN=128`（warp 構成 `2x8`=16 warp=512 スレッド。共有メモリ 36864B ≈ 36KiB。per-block 48KiB 上限に対し余裕を残す）へ拡大した。`BK` 不変のためアキュムレート順序は BM/BN に依存せず B-1/B-2 時点と bit 一致の出力を維持する（parity 非後退契約は変更不要）。候補算出・SMEM/レジスタ予算・段階的計測手順は [`docs/perf/cuda-gemm-mma-block-tile.md`](./perf/cuda-gemm-mma-block-tile.md) を参照。sm_121 実機属性は未実測のため候補判断は全 compute capability 共通の保証値ベースであり、実機再確認は #502 へ引き継ぐ。

### 2.4 インライン PTX の NVRTC 受理は未検証

- `mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32`・`ldmatrix.sync.aligned.m8n8.x4/x2.trans.shared.b16`・`cp.async.cg.shared.global`／`.commit_group`／`.wait_group` を NVRTC が受理するかどうかは、本実装セッション中に一度も構文検証を通過していない（driver はあるが NVRTC が存在しない環境のため）。sm_86（実機到達確認済み）・sm_121（設計上のターゲット）のいずれでも未検証（[`docs/perf/cuda-gemm-mma-pipeline.md`](./perf/cuda-gemm-mma-pipeline.md) 「未検証の事項（#64／#65 へ引き継ぐ）」）。

## 3. 実装判断の記録

| 判断事項 | 内容 | 出典 |
|---------|------|------|
| WMMA タイル構成の縮小 | ブロックタイル = warp タイル = fragment タイル = `m16n16k16`（1 ブロック = 1 warp = 32 スレッド、fragment 1 個のみ）に縮小。設計メモ候補値（128×128・64×64・2×2 warp）からの意図的な逸脱 | [`kernels_wmma.rs`](../crates/backend-cuda/src/kernels_wmma.rs) 冒頭コメント「タイル構成」 |
| mma パイプラインのタイル構成縮小（当初・B-1〜B-2） | `BM=32`・`BN=64`・`BK=32`・3 ステージ、1 warp = C の `16x8` タイル 1 個のみ（warp 内 M/N 方向の追加タイルループなし） | [`kernels_mma.rs`](../crates/backend-cuda/src/kernels_mma.rs) 冒頭コメント「タイル構成」。2.3 節参照 |
| mma パイプラインのブロックタイル拡大（B-3・#494） | `BM=64`・`BN=128`・`BK=32`・3 ステージへ拡大（warp あたり 2x2 レジスタブロッキングは #493 のまま不変）。ブロックスレッド数を 128→512（B-1 相当）へ回復 | 同上・2.3 節参照 |
| 縮小の共通理由 | 実機未接続・コンパイル未検証環境での「索引演算の複雑度を最小化する安全側判断」。`kernels_mma.rs` は `kernels_wmma.rs` の判断をそのまま踏襲 | 両ファイル冒頭コメント |
| XOR swizzle 不採用 | バンクコンフリクト低減目的の XOR swizzle（実装計画「段階 3」）は、索引演算が最も複雑でありながらコンパイル未検証環境では誤りを検出できないため不採用。将来、性能実測が可能な環境で導入を検討する | [`docs/perf/cuda-gemm-mma-pipeline.md`](./perf/cuda-gemm-mma-pipeline.md)「スコープ外」 |
| `ldm`（leading dimension）制約 | half 入力の `load_matrix_sync` は ldm が 8 要素の倍数、f32 の `store_matrix_sync` は ldm が 4 要素の倍数を要求。WMMA 実装は共有メモリタイル行幅を fragment 次元と同じ 16 要素に固定することで、追加パディングなしに両制約を同時に満たす（設計メモが挙げる 24 要素幅パディング候補は #63 スコープの将来最適化として保留） | [`kernels_wmma.rs`](../crates/backend-cuda/src/kernels_wmma.rs) 冒頭コメント「ldm 制約」 |
| 共有メモリのアライメント | `load_matrix_sync`／`store_matrix_sync` に渡すベースポインタは 256 bit（32 バイト）境界へのアライメントを要求する。nvcc の既定の `__shared__` 変数配置はこれを保証しないため、`__align__(32)` を各タイル宣言に明示（省略時は実機で misaligned address によるカーネル起動失敗・誤演算のリスク） | 同上 |
| f32 アキュムレート固定 | f16 入出力・f32 内部アキュムレートを WMMA・mma パイプライン双方で統一。PyTorch の f16 GEMM が cuBLAS 内部で FP32 アキュムレートするのと精度前提を揃える方針を PoC-v2-3 から継承 | [`kernels_wmma.rs`](../crates/backend-cuda/src/kernels_wmma.rs)・[`kernels_mma.rs`](../crates/backend-cuda/src/kernels_mma.rs) 冒頭コメント「数値契約」、設計メモ 4.1 節 |
| `cp.async` 16 バイト整列制約とホスト側追加検証 | `cp.async.cg.shared.global` はコピー粒度 16 バイト（f16 8 要素）固定。グローバル側の行ストライド（A は `k`、B は `n`）が 8 の倍数でない場合は整列しない可能性があるため、`gemm_mma.rs::CudaMmaGemm::run_f16` がホスト側で `k % 8 == 0 && n % 8 == 0` を追加検証し、満たさない形状は `CudaError::InvalidShape` で拒否する | [`kernels_mma.rs`](../crates/backend-cuda/src/kernels_mma.rs) 冒頭コメント「整列制約」 |
| REQ-8 境界検査の維持方針 | WMMA・mma パイプラインいずれも、性能下限・最適化達成を理由に手動境界チェックを省略しない（`.claude/rules/coding-rust.md`）。WMMA は guarded load／guarded store の条件分岐、mma パイプラインは `cp.async` の `src_size` オペランドをゼロにする方式＋クランプ済み添字でポインタ自体を範囲外にしない方式、という異なる実装手段を採るが方針は共通 | 両ファイル冒頭コメント「境界検査（REQ-8。省略禁止）」節 |

## 4. 経路別 検証状態マトリクス

**未検証事項を検証済みと書かない**（TASK-11.3 証跡整備の前提となるため整合性を最優先する）。

| 経路 | 実装ファイル | 検証済み | 未検証 |
|------|------------|---------|--------|
| tiled f32 | `kernels.rs`／`gemm.rs` | PoC-v2-3 実機実測（1.832 TFLOPS、M=N=K=4096） | - |
| WMMA f16 | `kernels_wmma.rs`／`gemm_wmma.rs` | ビルド（toolkit 非搭載環境）・clippy・命令実在テスト（`mod tests`）・環境適応テスト（`NvrtcUnavailable` graceful skip） | NVRTC の `<mma.h>` 実行時コンパイル成否（sm_121 実機）・実機数値一致・実機 TFLOPS（[`cuda-tensor-core-measurement.md`](./perf/cuda-tensor-core-measurement.md)「状態: 実測未実施」） |
| WMMA TF32 | `kernels_wmma.rs`／`gemm_wmma.rs`（TF32 側） | 同上 | 同上。加えて誤差分布は #186 で RTX 3060（sm_86）実測済みだが sm_121 は未確認（5 節参照） |
| WMMA f16／TF32 opt 版（`kernels_wmma_opt.rs`） | 同上 opt 経路 | ビルド・clippy・命令実在テスト・ダブルバッファリング構造保持テスト | 実機コンパイル・実機性能・実機数値一致 |
| mma.sync／ldmatrix／cp.async パイプライン | `kernels_mma.rs`／`gemm_mma.rs` | ビルド・clippy・`#[cfg(test)]` によるコンパイル時定数整合検査（`const _: () = assert!(...)`）・命令実在検査・境界チェック実在検査・driver 到達→NVRTC 不在検出→graceful skip の実機実行確認（RTX 3060） | インライン PTX の NVRTC 受理（2.4 節）・`ldmatrix` レーン→共有メモリアドレス対応と `mma.sync` フラグメントのレーン→レジスタ対応の整合（実行時数値照合でのみ確認可能）・`cp.async` 16 バイト整列制約下の実スループット・sm_121 固有挙動（[`cuda-gemm-mma-pipeline.md`](./perf/cuda-gemm-mma-pipeline.md)「未検証の事項」） |

### 4.1 数値一致閾値の既知の重大な未解決事項（#186 実測）

- #186（TASK-11.1g）で RTX 3060（compute capability 8.6）実機の誤差分布を実測した結果、**TF32 経路は全形状で現行閾値（相対 1e-3 未満 または 絶対 1e-5 未満）を著しく超過し、f16 経路も大きな K で閾値を超過することが判明した**（[`cuda-tensor-core-tolerance-evaluation.md`](./perf/cuda-tensor-core-tolerance-evaluation.md) 4 節「結論」）。
- 閾値定数自体は #186 では**一切変更していない**（ユーザー承認なしの緩和は行わない方針、`.claude/rules/security.md` A08）。改定候補は同ドキュメント 4 節に記載されているが、いずれも REQ-2 改定（正本 spec リポジトリ側での対応）が必要でありスコープ外。
- 本事実は sm_121（GB10）実機ではなく sm_86（RTX 3060）実測に基づくものであり、Tensor Core の世代差（mantissa 丸め方式・累算精度）による差異が出る可能性があるため、**sm_121 にそのまま適用しない**（同ドキュメント 5 節「制約事項」）。TASK-11.3 の証跡整備・#66（ディスパッチ規則）の実装時は、この未解決事項を前提として扱うこと。

## 5. TASK-11.2（#66）・TASK-11.3（#70）への引き渡し

- **ディスパッチ規則の設計**: TASK-11.2a としてすでに [`docs/dispatch-rules-design.md`](./dispatch-rules-design.md) が存在する（HW 判定・形状判定・dtype ゲート・フォールバック連鎖・決定表を含む）。本文書はその実装（#66 本体）の対象ではないため重複記述しない。
- **compute capability ゲート**: WMMA 経路は cc 7.0 以降、TF32 経路は cc 8.0 以降、`cp.async`／`ldmatrix` を使う mma パイプラインは `MIN_COMPUTE_CAPABILITY_MAJOR = 8`（`gemm_mma.rs`）。sm_121 はいずれも満たす（1 節・[`kernels_mma.rs`](../crates/backend-cuda/src/kernels_mma.rs) 冒頭コメント「命令選定・sm_80+ ゲート」）。
- **命令実在の証跡位置**: `kernels_wmma.rs`／`kernels_wmma_opt.rs`／`kernels_mma.rs` それぞれの末尾 `#[cfg(test)]` モジュールが、tensor core 命令文字列の実在検査・タイル定数とカーネルソース `#define` の整合検査・REQ-8 境界チェック実在検査を実施している。
- **ベンチログテンプレート**: [`docs/perf/cuda-tensor-core-measurement.md`](./perf/cuda-tensor-core-measurement.md)（#64。WMMA TF32／f16 実機実測テンプレート）・[`docs/perf/cuda-gemm-mma-pipeline.md`](./perf/cuda-gemm-mma-pipeline.md)（#187。mma パイプライン実測テンプレート）。いずれも「実測未実施」状態で、実機（DGX Spark GB10 等）実行後に記入する運用。
- **フォールバック条件**（[`cuda-tensor-core-design.md`](./cuda-tensor-core-design.md) 7 節から転記・要約）: (1) toolkit 非搭載・NVRTC が `<mma.h>` を解決できない環境、(2) compute capability が WMMA/TF32/mma パイプラインの要件を満たさない環境、(3) M/N/K がタイル最小単位に満たない極小形状、の 3 条件で tiled 経路へフォールバックする。
- **4.1 節の数値一致閾値未解決事項**は、TASK-11.3 の証跡整備・#66 のディスパッチ規則実装いずれにおいても前提条件として扱うこと（精度重視の用途では TF32/f16 Tensor Core 経路を選択しない設計も #186 の改定候補として提示されている）。

## 関連ドキュメント一覧

| ドキュメント | 対応イシュー | 内容 |
|------------|------------|------|
| [`docs/cuda-tensor-core-design.md`](./cuda-tensor-core-design.md) | #60（TASK-11.1a） | sm_121 系譜・方式 A/B 比較・NVRTC ヘッダ問題・fragment/タイル構成候補値・境界検査設計・未検証事項一覧 |
| [`docs/dispatch-rules-design.md`](./dispatch-rules-design.md) | #66 前段（TASK-11.2a） | ディスパッチ規則（HW/形状/dtype 判定・フォールバック連鎖）の設計 |
| [`docs/perf/cuda-tensor-core-measurement.md`](./perf/cuda-tensor-core-measurement.md) | #64（TASK-11.1e） | WMMA TF32／f16 実機実測テンプレート（記入待ち） |
| [`docs/perf/cuda-gemm-mma-pipeline.md`](./perf/cuda-gemm-mma-pipeline.md) | #187（TASK-11.1h） | `mma.sync`/`ldmatrix`/`cp.async` パイプライン実測テンプレート（記入待ち）・スコープ外事項 |
| [`docs/perf/cuda-tensor-core-tolerance-evaluation.md`](./perf/cuda-tensor-core-tolerance-evaluation.md) | #186（TASK-11.1g） | RTX 3060 実機誤差分布実測・閾値超過の判明・NVRTC プロビジョニング手順 |
| [`crates/backend-cuda/src/kernels_wmma.rs`](../crates/backend-cuda/src/kernels_wmma.rs) | #61（TASK-11.1b） | WMMA f16 カーネルソース・タイル縮小理由・`ldm`／アライメント制約 |
| [`crates/backend-cuda/src/kernels_mma.rs`](../crates/backend-cuda/src/kernels_mma.rs) | #187（TASK-11.1h） | mma.sync パイプラインカーネルソース・48KiB 上限判断・整列制約・XOR swizzle 不採用 |

## 参考文献

新規の外部文献調査は行っていない。出典は [`docs/cuda-tensor-core-design.md`](./cuda-tensor-core-design.md)「参考文献」節を参照。
