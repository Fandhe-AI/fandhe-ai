# バックエンド別ビルド・実行マトリクス（#57・TASK-2.4）

イシュー #57「docs(backend): TASK-2.4 ビルド・実行マトリクスの文書化」（親: TASK-2.4、`docs/spec/05-tasks.md` 112〜117 行）に対応する。
`docs/spec/04-requirements.md` REQ-2 受け入れ基準「各バックエンドのビルド・実行の可否と詰まりポイントをマトリクスとして文書化すること。v2 ではマシン構成を PoC-v2-5 の実施環境（Apple M4 Max〈CPU・Metal〉・NVIDIA DGX Spark GB10〈CPU・CUDA〉）へ差し替え、v1 の『バックエンド別ビルド・実行マトリクス』を土台としつつ更新すること」を満たすためのドキュメント。

v1 の土台は `docs/spec/03-poc/poc-4-multi-backend/README.md`「バックエンド別ビルド・実行マトリクス」節（Burn 基盤・Apple Silicon Mac 単機）。v2 は REQ-1 全面改定（`burn`・`cubecl`・`ndarray` 禁止）により基盤が完全自作コアへ差し替わっており、マシン構成も PoC-v2-5 実施環境（Apple M4 Max・NVIDIA DGX Spark GB10）へ更新した。前提タスク TASK-2.2（#52・数値一致回帰テスト実装）・TASK-2.3（#56・CUDA 非搭載環境ビルドの CI 検証）はいずれも完了済みであり、本ドキュメントはその成果を集約する。

## 1. 位置づけ・スコープ外

- 切替構成（cfg ベース・feature フラグなし）の設計根拠は [`docs/backend-switching-design.md`](./backend-switching-design.md) が正本であり、本ドキュメントでは重複記述しない。
- 実機（Apple Silicon）テスト実行手順の正本は [`docs/backend-metal-real-device-testing.md`](./backend-metal-real-device-testing.md)。CUDA Tensor Core 経路の詳細知見は [`docs/cuda-tensor-core-knowledge.md`](./cuda-tensor-core-knowledge.md) が正本。いずれも要点のみ引用し詳細は参照に委ねる。
- **既定 feature 構成の決定（CUDA を既定で有効にするか等）は TASK-2.5（#58）の担当**（`docs/spec/05-tasks.md` TASK-2.5「前提タスク: TASK-2.4」）。決定事項は 7 節に追記済み。
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
| 実測済み（#380） | `make test-ignored-metal`（`#[ignore]` 52 件）・`make check-cross-metal-tests` はいずれも Apple Silicon 実機（M4 Max・macOS 26.6・`stable-aarch64-apple-darwin`）で green を実測確認済み（[`docs/backend-metal-real-device-testing.md`](./backend-metal-real-device-testing.md)「実機実行の実測状況」節） |
| ベンチ実測完了（#381〜#386。トラッキング親 #379） | f32 GEMM 4 段（naive/tiled/simdgroup/dynamic-tile。#381）: size=4096 で simdgroup 1.7432 TFLOPS・dynamic-tile 3.0283 TFLOPS（[`docs/perf/metal-gemm-dynamic-tile.md`](./perf/metal-gemm-dynamic-tile.md)）。境界形状 TFLOPS・`METAL_SIMDGROUP_MIN_DIM` 妥当性判定（#382）: クロスオーバー実測 384（変更提案を記録。実施は別レビュー・別 PR・ユーザー承認）（[`docs/perf/dispatch-boundary-measurement.md`](./perf/dispatch-boundary-measurement.md)）。f16 GEMM 対 PyTorch MPS f16（#383）: size=2048 で 21.6%・size=4096 で 18.6%（[`docs/perf/metal-f16-vs-mps-f16.md`](./perf/metal-f16-vs-mps-f16.md)）。起動コスト（#384）: cold ≒ warm（[`docs/perf/startup-cost-measurement.md`](./perf/startup-cost-measurement.md)「Metal 実測結果」節）。ピークメモリ（#385）: 対理論比 1.000（[`docs/peak-memory-coefficient-decision.md`](./peak-memory-coefficient-decision.md)）。REQ-8「Metal f16 初期リリース」下限 15% 確定（#386。人間承認済み。[`docs/perf/performance-floor-decision.md`](./perf/performance-floor-decision.md) §8） |

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

## 7. 既定 feature 構成の決定（TASK-2.5・#58）

`docs/spec/05-tasks.md` TASK-2.5「`cuda` feature をライブラリの既定 feature に含めるかを、CI 実行時間・配布物のバイナリサイズ・依存クレート数への影響評価を踏まえて決定する」に対応する。担当区分は「共同（影響評価は Claude Code、既定化の可否判断は人間）」。

### 7.1 決定

**バックエンド切替は feature フラグなし cfg ベース構成（2 節・[`docs/backend-switching-design.md`](./backend-switching-design.md)）を既定として維持し、新規 `cuda` feature は追加しない。** CUDA サポートはすべてのビルドに無条件で組み込み、CUDA の要否は実行時の動的ロード可否（`cudarc` dynamic-loading）で決まる。

これは REQ-2 が「未検証のまま残る」と明記した残存課題「バックエンド有効化構成（feature 追加の要否を含む）の決定」（`docs/spec/04-requirements.md` REQ-2 受け入れ基準）を、下記 7.2 の実測に基づき解消するものである。

### 7.2 影響評価（実測値）

計測環境: 本 worktree（Linux・self-hosted 相当、`rustc` は workspace `Cargo.toml` 指定の edition 2024 対応版）。`cudarc =0.19.8`（`driver`／`nvrtc`／`dynamic-loading`／`cuda-13000`／`f16` feature、`Cargo.toml` `[workspace.dependencies]`）を対象とする。3 指標とも本イシューで新規実測し、既存の推定記述は含めない。

| 指標 | 実測値 | 計測方法・出典 |
|------|--------|----------------|
| 依存クレート数 | `Cargo.lock` 総パッケージ数 113 に対し、`cudarc` を依存グラフから完全に除いた場合（`backend-cuda` だけでなく、`cudarc` に直接依存する `bench-harness` の `stream.synchronize()` 用途も含めて除去した場合）に**削減されるパッケージは `cudarc` 自身と `libloading` の 2 個のみ（1.8%）**。`cudarc` の他の直接依存（`half`・`rand`・`num-traits`・`libc`・`libm`・`zerocopy`・`proc-macro2`・`syn` 等）はいずれも `rayon`・`half`（workspace 許容依存）・`serde` derive 等の既存経路と共有されており除去されない。なお `backend-cuda` を除いても `cudarc` 自体は残る点に注意（`bench-harness`（`crates/bench-harness/Cargo.toml`）が同期方式統一〈TASK-8.1b〉のため `cudarc` に直接依存しており、`cuda` feature を `backend-cuda` クレート単体に付けても `cudarc` を切り離せない） | `cargo metadata --locked --format-version 1` の resolve グラフで、`cudarc` ノードへの再帰を止めた到達可能集合と、通常の到達可能集合（workspace member 起点、normal+build+dev 全 kind）を比較（scratchpad の解析スクリプト）。`cargo tree -i cudarc`・`cargo tree -i libloading` で cudarc への直接依存元（`backend-cuda`・`bench-harness`）と `libloading` が cudarc 専有であることを確認 |
| バイナリサイズ | CPU のみ構成（`tensor-core`+`backend-cpu`）と CPU+CUDA 構成（+`backend-cuda`、`CudaDevice::is_available()` を実際に呼び出し driver 動的ロード経路をリンカのデッドコード除去対象から外した構成）の release バイナリ差分は **stripped で 14,712 バイト（約 14.4 KiB、344,200 → 358,912 バイト）・unstripped で 23,248 バイト（約 22.7 KiB、447,176 → 470,424 バイト）**。参考値として、CUDA API を一切呼ばない（型名参照のみの）構成では差分が 80 バイトまで縮む（デッドコード除去がほぼ全体を除去するため。実利用時の下限ではなく非代表値として記録）。`libcudarc` の rlib 自体は 7.26 MiB（`libbackend_cuda` rlib は 712,874 バイト）だが、`cudarc` は動的ロード方式で実 CUDA 共有ライブラリへ静的リンクしないため、実行ファイルへの増分は数十 KiB 規模にとどまる | scratchpad に最小 bin クレート 2 個（`tensor-core`+`backend-cpu` の path 依存のみ／同 +`backend-cuda`）を作成し `cargo build --release` 後のファイルサイズを比較。B 側は `CudaDevice::is_available()` を呼び出す構成で計測（型名参照のみだとデッドコード除去で過小評価になるため是正）。参考値として `target/release/deps/libcudarc-*.rlib`・`libbackend_cuda-*.rlib` サイズを記録 |
| CI 実行時間 | ローカル `cargo build --release --workspace --timings`（クリーンビルド）で、全 88 コンパイル単位の per-unit 所要時間合計 32.14 秒中、`cudarc`＋`backend-cuda` の合計は **3.57 秒（約 11.1%）**。直近の CI 実行（run #31137297795、PR #267 マージ時）では `cargo build (linux / aarch64-apple-darwin)` ジョブ 57 秒・`build-no-cuda-toolkit`（CUDA toolkit 非搭載検証）ジョブ 1 分 43 秒・`cargo test` ジョブ 62 秒（最長は `cargo deny check licenses sources` の 4 分 37 秒で CUDA 依存量とは無関係）。CUDA を feature 化した場合に削減しうる CI 時間の上限は、全体ビルド時間の 1 割程度にとどまる | `target/cargo-timings/cargo-timing.html` 埋め込みの `UNIT_DATA` を解析（scratchpad のスクリプト）。`gh run view 31137297795 --json jobs` でジョブ別所要時間を取得（読み取りのみ） |

計測用の比較 bin クレート・解析スクリプトはいずれも scratchpad 限定で作成し、本リポジトリへはコミットしていない。

### 7.3 判断根拠

- 3 指標とも CUDA 無条件組み込みの実コストは小さい。依存クレート数は 113 中 2 個（1.8%）、バイナリサイズは実利用時（driver ロード経路を実際に呼び出す構成）で 14〜23 KiB 程度（`cudarc` が動的ロード方式のため実 CUDA ライブラリへの静的リンクを持たないため）、CI 時間はビルド全体の約 1 割にとどまる。`cuda` feature を新設して opt-out 可能にしても、削減できる実コストは限定的である。
- さらに、`cudarc` は `backend-cuda` だけでなく `bench-harness`（TASK-8.1b・同期方式統一）からも直接依存されているため、**`backend-cuda` クレート単体を feature でくくっても `cudarc` 自体は依存グラフから外れない**。`cudarc` を真に opt-out にするには `bench-harness` の同期方式統一機能も feature 分岐させる必要があり、TASK-2.5 が想定する「`cuda` feature の要否」は実装上 `backend-cuda` 単体の feature 化では完結しない。これは feature 化のコストを追加で押し上げる要因であり、7.1 節の決定（feature 化しない）を補強する。
- 一方で feature 化した場合に失うものは大きい: [`docs/backend-switching-design.md`](./backend-switching-design.md)「なぜ feature フラグを使わないか」節が指摘する (1) PoC-v2-5 が feature フラグなし構成を直接実証済みであること、(2) v1 の「型エイリアス + Cargo feature」前提は REQ-1 全面改定で消滅していること、(3) feature 組合せ増加による検証マトリクスの組合せ的増大・feature 指定漏れによる経路欠落リスク、の 3 論点は本イシューの実測でも覆らない。
- CUDA toolkit 非搭載環境でのビルド・実行成立は `build-no-cuda-toolkit` ジョブ（TASK-2.3・#56）が `scripts/check-cuda-toolkit-absent.sh assert` による fail-closed 検証で継続監視しており、`cuda` feature という opt-out 手段がなくても非 CUDA 環境の利用者に実害はない（`CudaDevice::is_available()` が `false` に縮退し `CudaError::DriverUnavailable` を返す型付きエラー契約。3.2 節）。

### 7.4 担当区分と承認

影響評価（7.2 節）は Claude Code が本イシューで実施した。既定化の可否判断（7.1 節の決定案の採否）は人間が行う事項であり、**本イシューに対応する PR のレビュー・マージをもって承認とする**（TASK-2.5 の担当区分「共同」に対応）。

### 7.5 再検討トリガー

- 配布形態が変化した場合（例: crates.io への公開、バイナリ配布形態の追加）は、依存クレート数・バイナリサイズの実コストが変わりうるため 7.2 節の実測を再実施する。
- 7.2 節の実測値が大幅に悪化した場合（例: `cudarc` のメジャーバージョン更新で静的リンク依存が増える等）も同様に再評価する。
- 再評価は REQ-2 改定（正本 spec リポジトリ側での対応）とセットで行う。REQ-2 残存課題「バックエンド有効化構成の決定」は本節により実装リポ側では解消済みとして扱うが、spec 本文（`docs/spec/04-requirements.md`）側への反映は `docs/spec/` 編集禁止のため本ドキュメントでは行わない。正本側への反映提案は本イシューに対応する PR 本文に記載し、ユーザー判断に委ねる。

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
