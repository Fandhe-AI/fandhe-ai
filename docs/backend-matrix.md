# バックエンド別ビルド・実行マトリクス（#57・TASK-2.4）

イシュー #57「docs(backend): TASK-2.4 ビルド・実行マトリクスの文書化」（親: TASK-2.4、`docs/spec/05-tasks.md` 112〜117 行）に対応する。
`docs/spec/04-requirements.md` REQ-2 受け入れ基準「各バックエンドのビルド・実行の可否と詰まりポイントをマトリクスとして文書化すること。v2 ではマシン構成を PoC-v2-5 の実施環境（Apple M4 Max〈CPU・Metal〉・NVIDIA DGX Spark GB10〈CPU・CUDA〉）へ差し替え、v1 の『バックエンド別ビルド・実行マトリクス』を土台としつつ更新すること」を満たすためのドキュメント。

v1 の土台は `docs/spec/03-poc/poc-4-multi-backend/README.md`「バックエンド別ビルド・実行マトリクス」節（Burn 基盤・Apple Silicon Mac 単機）。v2 は REQ-1 全面改定（`burn`・`cubecl`・`ndarray` 禁止）により基盤が完全自作コアへ差し替わっており、マシン構成も PoC-v2-5 実施環境（Apple M4 Max・NVIDIA DGX Spark GB10）へ更新した。前提タスク TASK-2.2（#52・数値一致回帰テスト実装）・TASK-2.3（#56・CUDA 非搭載環境ビルドの CI 検証）はいずれも完了済みであり、本ドキュメントはその成果を集約する。

## 1. 位置づけ・スコープ外

- 切替構成（cfg ベース・feature フラグなし）の設計根拠は [`docs/backend-switching-design.md`](./backend-switching-design.md) が正本であり、本ドキュメントでは重複記述しない。
- 実機（Apple Silicon）テスト実行手順の正本は [`docs/backend-metal-real-device-testing.md`](./backend-metal-real-device-testing.md)。CUDA Tensor Core 経路の詳細知見は [`docs/cuda-tensor-core-knowledge.md`](./cuda-tensor-core-knowledge.md) が正本。いずれも要点のみ引用し詳細は参照に委ねる。
- **既定 feature 構成の決定（CUDA を既定で有効にするか等）は TASK-2.5（#58）の担当であり、本ドキュメントでは決定しない**（`docs/spec/05-tasks.md` TASK-2.5「前提タスク: TASK-2.4」）。TASK-2.5 は本ファイルへ決定事項を追記する前提の構成にしている。
- ROCm・Vulkan は REQ-2 で対象外（下記表に「対象外」行として明記）。

## 2. 検証環境

REQ-2 受け入れ基準が指定する 2 実機に加え、CI（self-hosted・実機非搭載）を検証実体として扱う。

| 環境 | 構成 | 検証実体 |
|------|------|---------|
| Apple M4 Max（macOS・Apple Silicon） | CPU・Metal | PoC-v2-4／PoC-v2-5 実測 + `make test-ignored-metal`（[`docs/backend-metal-real-device-testing.md`](./backend-metal-real-device-testing.md)） |
| NVIDIA DGX Spark GB10（aarch64 Linux・CUDA 13.0・sm_121） | CPU・CUDA | PoC-v2-3／PoC-v2-5 実測 + `make test-ignored-cuda`・`#[ignore]` 実機テスト |
| self-hosted CI（Linux・CUDA toolkit 非搭載・macOS runner 未登録） | ビルド検証・非実機テスト | `build` ジョブ（Linux ホスト + `aarch64-apple-darwin` クロスターゲット lib-only ビルド）・`build-no-cuda-toolkit` ジョブ（TASK-1.7d #35・TASK-2.3 #56）・`test` ジョブ（`.github/workflows/ci.yml`） |

## 3. バックエンド別ビルド・実行マトリクス

各セル: **ビルド可否／実行可否／検証手段／詰まりポイント**。実行可否は本ドキュメント執筆時点で新規実測を行っておらず、既存の実測記録（PoC 番号・イシュー/PR 番号つき出典）を転記する。未実測の項目は「未実測」と明記し推定で埋めない（TASK-2.4 は 0.5 人日の集約タスクであり新規実測を含まない）。

### 3.1 CPU（`rayon` 無条件依存）

| 項目 | 内容 |
|------|------|
| ビルド可否 | **OK（全環境）**。`rayon` は cfg 分岐なしの無条件依存（`.claude/rules/deps-policy.md`） |
| 実行可否 | **OK（全環境）**。PoC-v2-1（`docs/spec/03-poc/poc-v2-1-tensor-cpu-gemm/README.md`）で採用実測（naive/blocked 比 約 6〜8.5 倍改善） |
| 検証手段 | `test`／`build`／`build-no-cuda-toolkit` の各 CI ジョブで全環境共通に実行される（`.github/workflows/ci.yml`）。数値一致は `crates/backend-cpu/tests/gemm_parity.rs`・`gemm_blis_parity.rs`（`backend_cpu::parity` 経由） |
| 詰まりポイント | なし。CPU–GPU 数値一致の前提となる FMA 契約（`f32::mul_add`）の唯一の参照点は `backend_cpu::matmul_reference_fma`（下記 4 節） |

### 3.2 CUDA（`cudarc` 無条件依存・動的ロード）

| 項目 | 内容 |
|------|------|
| ビルド可否 | **OK（CUDA toolkit 搭載・非搭載いずれの環境でも成立）**。`cudarc` は動的ロード方式（`dynamic-loading` feature）のため、コンパイル時に toolkit を要求しない。PoC-v2-3（`docs/spec/03-poc/poc-v2-3-cuda-gemm/README.md`）で実証、CI `build-no-cuda-toolkit` ジョブ（TASK-2.3・#56）が `scripts/check-cuda-toolkit-absent.sh assert` で「toolkit 非搭載であること自体」を fail-closed 検証したうえで継続監視 |
| 実行可否 | **DGX Spark GB10（toolkit 搭載）: OK**。PoC-v2-3／PoC-v2-5 実機実測（tiled f32 経路: 1.832 TFLOPS、M=N=K=4096）。**toolkit 非搭載環境: 型付きエラーへ縮退（panic しない）**。`CudaDevice::is_available()` が `false` に縮退し `CudaError::DriverUnavailable` を返す契約（`build-no-cuda-toolkit` ジョブ、`.github/workflows/ci.yml` 294〜306 行コメント） |
| 検証手段 | `make test-ignored-cuda`（実機専用 `#[ignore]` テスト）。toolkit 非搭載側は CI `build-no-cuda-toolkit` ジョブの `cargo build --workspace --locked` → `cargo test --workspace --locked` |
| 詰まりポイント（NVRTC・Tensor Core 経路。詳細は [`docs/cuda-tensor-core-knowledge.md`](./cuda-tensor-core-knowledge.md)） | (1) **NVRTC ヘッダ非同梱**: `<mma.h>` を使う WMMA カーネルは実行時（`nvrtcCreateProgram` 呼び出し時）に toolkit の include パス解決が必要。ビルド成立とは別軸（同文書 2.1 節）。(2) **静的共有メモリ per-block 48KiB 上限**: `mma.sync`/`ldmatrix`/`cp.async` パイプラインのタイル構成を `BM=32・BN=64・BK=32・3 ステージ`（18KiB）へ縮小して回避（同文書 2.3 節）。(3) **sm_121（GB10）固有の未検証事項**: `compute_121` の NVRTC 受理可否・インライン PTX（`mma.sync`/`ldmatrix`/`cp.async`）の NVRTC 受理可否はいずれも未検証（同文書 1 節・2.4 節）。(4) **数値一致閾値の重大な未解決事項（#186）**: RTX 3060（sm_86）実機実測で TF32 経路は全形状、f16 経路も大きな K で現行複合判定閾値（相対 1e-3 未満 または 絶対 1e-5 未満）を超過することが判明。sm_121 では未確認。閾値自体は変更していない（ユーザー承認必須事項のため。同文書 4.1 節） |
| 未実測 | WMMA f16／TF32・mma パイプラインの sm_121 実機コンパイル成否・実機数値一致・実機 TFLOPS（[`docs/cuda-tensor-core-knowledge.md`](./cuda-tensor-core-knowledge.md) 4 節「経路別 検証状態マトリクス」に詳細） |

### 3.3 Metal（`objc2`／`objc2-foundation`／`objc2-metal`、`cfg(target_os = "macos")` 分離）

| 項目 | 内容 |
|------|------|
| ビルド可否 | **macOS: OK**。**非 macOS（Linux 等）: 該当コード・依存ごとビルド対象外**（`[target.'cfg(target_os = "macos")'.dependencies]` 分離。`docs/backend-switching-design.md`）。CI（Linux self-hosted）では `aarch64-apple-darwin` クロスターゲットへの lib-only ビルド（`cargo build --workspace --locked --target aarch64-apple-darwin`）で Metal 有効経路のコンパイル可能性のみ検証する。macOS self-hosted runner は未登録のため実機ビルドは CI 対象外（`.github/workflows/ci.yml` `build` ジョブ） |
| 実行可否 | **Apple M4 Max: OK**。PoC-v2-4／PoC-v2-5 実測 |
| 検証手段 | `make test-ignored-metal`（`--release` 推奨。`cargo test -p backend-metal --release -- --ignored --nocapture`。[`docs/backend-metal-real-device-testing.md`](./backend-metal-real-device-testing.md)）。CI 側の型検査代替は `make check-cross-metal-tests`（`cargo check -p backend-metal --tests --target aarch64-apple-darwin`）。数値一致は `crates/backend-metal/tests/cpu_metal_parity.rs`（`backend_cpu::parity` 経由） |
| 詰まりポイント | (1) **macOS runner 未登録**: 実機実行は Linux CI では検証不能。代替として `aarch64-apple-darwin` クロスターゲットの型検査（`cargo check`。リンクを行わないため macOS SDK 不要）に限定（[`docs/backend-metal-real-device-testing.md`](./backend-metal-real-device-testing.md)「Linux CI での型検査」節）。(2) **`--workspace --all-targets` 不可**: `bench-harness` の `dev-dependencies`（`criterion`）経由で `alloca`（macOS ネイティブ C ビルド要）を引き込み、macOS クロスコンパイラ非搭載の self-hosted runner では `cc: error: unrecognized command-line option '-arch'` 等で失敗する。`-p backend-metal --tests` に限定することで回避（同文書「`--workspace --all-targets` ではなく」節）。(3) **debug ビルドの著しい低速化**: `cpu_metal_parity.rs::k4096_stress_poc_v2_5`（M=N=512, K=4096）は debug では著しく遅いため `--release` を実行手順の既定にしている（同文書「実行コマンド」節）。(4) **precise math 明示**: `MTLCompileOptions.mathFloatingPointFunctions=Precise` と `metal::precise::exp`／`metal::precise::tanh` の明示使用が数値一致複合判定の前提（REQ-2 (b)。`mathMode=Safe` のみでは `metal::fast` 経路にディスパッチされ判定余裕が薄くなる） |
| 未実測 | 本ドキュメント執筆時点で `make test-ignored-metal`／`make check-cross-metal-tests` が実際に green になることの実測はユーザー側実機確認に依存する（Linux 環境からは実行不能。[`docs/backend-metal-real-device-testing.md`](./backend-metal-real-device-testing.md)「実機実行の実測状況」節） |

### 3.4 ROCm・Vulkan（対象外）

| 項目 | 内容 |
|------|------|
| ビルド可否／実行可否 | **対象外**。REQ-2 は本要件の対象に含めないと明記（`docs/spec/04-requirements.md` REQ-2 受け入れ基準「ROCm・Vulkan バックエンドは本要件では対象外とする」）。ROCm は PoC-10 で格上げ判断基準を整理済みの条件付き Won't、Vulkan は未調査 |

## 4. バックエンド間数値一致の前提条件（REQ-2 (a)(b)(c)）

3 バックエンドの詰まりポイントに共通する前提条件を集約する（詳細は REQ-2 本文・`docs/backend-switching-design.md` を参照。本ドキュメントでは重複記述しない）。

| 前提条件 | 内容 | 一本化されている実体 |
|---------|------|---------------------|
| (a) FMA 契約統一 | CPU 参照実装は `f32::mul_add`（乗算後に丸めず加算まで一度に行う）を用い、GPU 側（CUDA NVRTC・Metal `simdgroup_multiply_accumulate`）の既定 FMA 契約と丸め方針を揃える。PoC-v2-5 で `acc += a * b`（未適用）だと K=4096 ストレスケースで 262,144 セル中 7 セルが複合判定を外れるが、`mul_add` 差し替え後は完全一致（fail_cells=0）と実測確認済み | `backend_cpu::matmul_reference_fma` |
| (b) Metal precise math 明示 | `MTLCompileOptions.mathFloatingPointFunctions=Precise` とシェーダ側 `metal::precise::*` の明示使用 | `crates/backend-metal` シェーダソース（3.3 節参照） |
| 統一複合判定 | 相対誤差 1e-3 未満 または 絶対誤差 1e-5 未満（全ペア共通）。TASK-2.2（#52）で CPU-Metal・CPU-CUDA 各ペアの回帰テストとして実装済み | `backend_cpu::parity::{compare, assert_parity}` |
| f16 の扱い | naive f16 経路は複合判定の適用が実質的な許容誤差変更にあたるため対象外（既定方針）。WMMA f16（TASK-11.1b・#61 の受け入れ条件）はユーザー承認済みの明示的な例外として複合判定を適用 | `crates/backend-cuda/tests/cpu_cuda_parity.rs`（対象外方針）・`cpu_cuda_wmma_parity.rs`（例外適用） |

**閾値の独自定義・緩和はしない**（`.claude/rules/security.md`・`.claude/rules/coding-rust.md`）。3.2 節の #186（TF32／f16 Tensor Core 経路の閾値超過）は既知の未解決事項として記載するに留め、本ドキュメントでは解決しない。

## 5. CI 検証ジョブとの対応

| CI ジョブ | 対応する詰まりポイント検証 | 出典 |
|-----------|---------------------------|------|
| `build`（Linux ホスト + `aarch64-apple-darwin` クロスターゲット） | CPU・CUDA の Linux ビルド、Metal の型検査（lib-only／`-p backend-metal --tests`） | TASK-2.1b・#50・PR #238 |
| `build-no-cuda-toolkit` | CUDA toolkit 非搭載環境でのビルド・実行成立（`DriverUnavailable` への型付き縮退） | TASK-1.7d・#35、TASK-2.3・#56 |
| `test` | CPU 経路の全環境共通ビルド・実行、数値一致回帰テスト（非 `#[ignore]` 分） | TASK-1.7〜1.9d |
| `deps-forbidden` | 依存禁止リスト（`burn`・`cubecl`・`candle`・`tch`・`ndarray`）の `Cargo.lock` 機械検査 | TASK-1.2 |

実機専用（`#[ignore]`）テストは通常 CI では実行しない。実行は `make test-ignored-cuda`（DGX Spark GB10）・`make test-ignored-metal`（Apple M4 Max）をユーザー側実機で行う運用（`.claude/rules/ci.md`「実機依存」節）。

## 6. 既知の未解決事項（TASK-2.5・自己修復ループ参照時の前提）

- **#186（TASK-11.1g）**: TF32／f16 Tensor Core 経路の数値一致閾値が sm_86 実測で著しく超過している。sm_121 では未確認。閾値改定には REQ-2 改定（正本 spec リポジトリ側での対応）が必要（`out-of-scope-tracking.md` に基づき既に #186 として追跡済み。新規起票は不要）
- **sm_121 の NVRTC 受理可否**: `compute_121` アーキテクチャフラグ・インライン PTX 命令の NVRTC 受理可否はいずれも実機未検証
- **macOS self-hosted runner 未登録**: Metal 実機 CI ジョブは未整備。追加する場合は runner ラベルで対象 runner を明示する方針（`.claude/rules/ci.md`）
- 上記はいずれも本ドキュメントの新規スコープではなく、既存イシュー・ドキュメントでの追跡状況をそのまま集約したものである

## 出典一覧

- `docs/spec/04-requirements.md` REQ-2（マルチ GPU バックエンド対応）
- `docs/spec/05-tasks.md` TASK-2.1〜2.5
- `docs/spec/03-poc/poc-4-multi-backend/README.md`（v1 土台マトリクス）
- `docs/spec/03-poc/poc-v2-1-tensor-cpu-gemm/README.md`・`poc-v2-3-cuda-gemm/README.md`・`poc-v2-5-backend-numeric-parity/README.md`
- [`docs/backend-switching-design.md`](./backend-switching-design.md)（TASK-2.1c・#51）
- [`docs/backend-metal-real-device-testing.md`](./backend-metal-real-device-testing.md)（TASK-1.8e・#42）
- [`docs/cuda-tensor-core-knowledge.md`](./cuda-tensor-core-knowledge.md)（TASK-11.1f・#65）
- `.github/workflows/ci.yml`（`build`・`build-no-cuda-toolkit`・`test`・`deps-forbidden` ジョブ）
- `Makefile`（`test-ignored-cuda`・`test-ignored-metal`・`check-cross-metal-tests`・`build-no-cuda` ターゲット）
- `crates/backend-cpu/src/parity.rs`・`crates/backend-cuda/tests/cpu_cuda_{parity,wmma_parity,mma_parity}.rs`・`crates/backend-metal/tests/cpu_metal_parity.rs`
