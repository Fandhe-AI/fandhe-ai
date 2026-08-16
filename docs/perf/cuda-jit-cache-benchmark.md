# CUDA JIT キャッシュ 初回コンパイル／2 回目ロード／スループット計測記録（#534・Phase C-12）

イシュー #534「JIT キャッシュ導入前後の初回コンパイル時間・2 回目ロード時間・スループットを計測」の実測記録テンプレート。
親イシュー #503（Phase C: CUDA JIT shape 特化・コンパイルキャッシュ・静的タイル選定）の C-12。
ディスクキャッシュ I/O（C-1〜C-3・#504／#506／#509）・回帰テスト整備（C-10・#529・PR #694）を踏まえ、その効果（初期化レイテンシ・スループット）を実測する。

## 状態: 未実測・実機実行待ち（DGX Spark GB10 実測は本イシューの実機セッションへ引き継ぐ）

本実装セッションには `docs/real-hardware-verification-env.local.md`（実ホスト名。`.gitignore` 対象・実機接続情報）が存在せず、DGX Spark GB10 実機へ到達できない。`docs/perf/cuda-jit-template-expansion.md`「状態」節・`docs/perf/cuda-jit-cache-*` 系の先例と同じ位置づけで、本ファイルはベンチコード（`crates/backend-cuda/src/jit_cache_bench_tests.rs`）とドキュメント骨子のみを用意し、実測値は空欄のまま実機セッションでの追記対象とする。

- 通常 CI で機械検証済みの事項（§4）: ベンチコードのコンパイル成立（`cargo build --workspace`）・`cargo test --workspace --all-features`（`#[ignore]` テストはコンパイル検査のみ）・fmt／clippy
- 実機必須・未実測の事項（§3）: NVRTC 実コンパイル時間・キャッシュ I/O のレイテンシ・モジュールロード時間・GEMM スループット

## 1. 計測対象と「JIT キャッシュ導入前後」の対応付け

C-4（本番 GEMM ディスパッチ経路〈`gemm_auto.rs::CudaGemmAuto::run_f16`〉への「ミス→コンパイル→store→hit」結線・プロセス内 LRU。#511）は本イシュー時点で未実装（open）。したがって「本番経路の前後比較」は原理的に行えず、本ベンチはキャッシュ I/O プリミティブ（[`crates/backend-cuda/src/nvrtc.rs`](../../crates/backend-cuda/src/nvrtc.rs) の `compile_ptx`・`store_cache_entry_in`・`load_cache_entry_in`。いずれも module-private）を、C-10 の回帰テスト（[`jit_cache_regression_tests.rs::get_or_compile`](../../crates/backend-cuda/src/jit_cache_regression_tests.rs)）と同型の直叩きで計測する。

| 計測 | 対応する経路 | 「導入前後」の意味 |
|------|-------------|-------------------|
| 初回コンパイル時間 | `compile_ptx`（NVRTC 実コンパイル）→ `store_cache_entry_in` | 「導入前」相当（毎プロセスでコンパイルする従来コスト） |
| 2 回目ロード時間 | `load_cache_entry_in`（キャッシュヒット） | 「導入後」の再起動時コスト |
| スループット | フレッシュコンパイル PTX と キャッシュロード PTX それぞれから起動した GEMM の bit 一致・TFLOPS | キャッシュ導入による性能非後退の実証 |

**本番経路への実効果（C-4 結線後に見込まれる起動コスト短縮）は、C-4 完了後の再計測事項として引き継ぐ**（本ファイル §5「引き継ぎ」参照）。

### 1.1 なぜ全ケースで実プロダクションカーネルソース 1 種類を使い回すか

`kernels_mma.rs::RenderedMmaKernel` は生ソース文字列を外部（`kernels_mma` モジュール外）へ返す公開メソッドを一切持たない設計（PR #643 codex-review 再々指摘〈P0〉。`cuda-jit-template-expansion.md` §1.3 参照）。ベンチコードは `nvrtc` モジュールの子孫であって `kernels_mma` の子孫ではないため、shape 特化構成（`gemm_auto.rs::specialized_mma_config`）が生成する形状ごとに異なるソース文字列をこの API から取得することはできない（意図された不変条件であり、本イシューのスコープでこのカプセル化を弱めない）。

そのため本ベンチは C-10 回帰テストと同じ選択で、既定構成（全次元 `Dynamic`）の実プロダクションカーネルソース（`kernels_mma::mma_f16_source()`）を全ケース共通で使う。`CudaKernelDescriptor` の `shape`／`CompiledDims` のみを変えてキャッシュキーの一意性を作り、複数の「キャッシュエントリ」を区別する目的に限定する。**したがって「初回コンパイル vs 2 回目ロード」の時間差は NVRTC コンパイル対象ソースの複雑度（shape 特化の有無）には依存せず、単一の実カーネルソースに対する測定である**。

## 2. 計測方法

`crates/backend-cuda/src/jit_cache_bench_tests.rs`（`nvrtc` モジュールの子モジュール。C-10 の `jit_cache_regression_tests.rs` と同じ配置理由。同ファイル冒頭ドキュメンテーションコメント参照）に 2 個の `#[ignore]` テストを追加した。

- `jit_cache_bench_cold_compile_vs_warm_load_latency`: `STATIC_MNK`（1024³・4096³）・`DYNAMIC_ALL`（4096³。既定プリセット）の 3 descriptor × 5 trial で、trial ごとに新規キャッシュルートを使い「コンパイル → store → 2 回目ロード」を計測する。区間別（compile／store／warm load）に `Instant` で計測し、`bench_harness::stats::median_q1_q3` で中央値・Q1・Q3 を求める
- `jit_cache_bench_module_load_and_throughput_parity`: 4096³ 形状で (a) フレッシュコンパイル PTX と (b) キャッシュ経由でロードした PTX それぞれのモジュールロード時間（5 回計測中央値）を記録し、両者から起動した GEMM（f16・4096³）の出力が bit 一致することを assert する。スループット（TFLOPS）は `bench_harness::protocol::run`（warmup 20 回・計測 20 回。TASK-8.1 下限）で計測し記録のみに留める（hard assert にしない。§2.1 参照）

いずれも `#[ignore]`（実機必須。`.claude/rules/coding-rust.md`「実機依存テストは `#[ignore]` で分離」）。実行コマンド（実機・§3）:

```bash
cargo test -p backend-cuda --release --lib -- --ignored --nocapture jit_cache_bench
```

### 2.1 スループット差を hard assert にしない理由

フレッシュコンパイル PTX とキャッシュロード PTX は byte 一致することを assert 済みのため、同一 NVRTC/ドライバ実装であれば理論上 GEMM 実行性能は同一のはずである。しかし GPU クロック挙動・他プロセス競合等の環境揺らぎを TFLOPS の hard assert に持ち込むと flaky 化するため（実装計画 §8「リスクと安全側の倒し方」）、TFLOPS は記録のみに留め、gating は「出力が bit 一致すること」（決定的で揺らがない検証）に限定する。

## 3. 実測結果（DGX Spark GB10・実機実行後に追記）

### 3.1 実行環境

| 項目 | 値 |
|------|-----|
| GPU | `<cuda-node>`（実ホスト名は `docs/real-hardware-verification-env.local.md` 参照。`.gitignore` 対象） |
| compute capability | 未実測 |
| driver / CUDA バージョン | 未実測 |
| NVRTC バージョン | 未実測（`nvrtc_version()` 実測値） |
| rustc バージョン | 未実測 |
| 計測リビジョン | 未実測（`git rev-parse HEAD`） |
| 実施日 | 未実測 |
| GPU 占有（計測前後） | 未実測（`nvidia-smi --query-compute-apps` で確認する。`docs/real-hardware-verification-env.md` §6.1） |

### 3.2 初回コンパイル時間・2 回目ロード時間（5 回計測中央値・Q1/Q3）

| descriptor | compile 中央値 (s) | store 中央値 (s) | cold 合計中央値 (s) | warm load 中央値 (s) | 速度比 (cold/warm) |
|-----------|---------------------|-------------------|----------------------|------------------------|----------------------|
| `c12-static-mnk-1024`（1024³・STATIC_MNK） | 未実測 | 未実測 | 未実測 | 未実測 | 未実測 |
| `c12-static-mnk-4096`（4096³・STATIC_MNK） | 未実測 | 未実測 | 未実測 | 未実測 | 未実測 |
| `c12-dynamic-all-4096`（4096³・DYNAMIC_ALL 既定） | 未実測 | 未実測 | 未実測 | 未実測 | 未実測 |

### 3.3 モジュールロード時間・スループット（4096³）

| 項目 | フレッシュコンパイル PTX | キャッシュロード PTX |
|------|---------------------------|------------------------|
| モジュールロード中央値 (s)（5 回計測） | 未実測 | 未実測 |
| GEMM スループット（TFLOPS。記録のみ・非 gating） | 未実測 | 未実測 |
| 出力 bit 一致 | 未実測（テストの `assert_eq!` が担保） | 同左 |

## 4. REQ-13（起動コスト）実測との突合

`docs/perf/startup-cost-measurement.md`「CUDA 実測結果」節・「コールド／ウォームの検証」節の実測（run1: first_kernel 中央値 cold 530.805ms／warm 322.028ms、差 208.777ms。run2: cold 510.810ms／warm 308.080ms、差 202.730ms）は、**本イシューのキャッシュ実測とは別レイヤの計測**である点を明記する。

- **startup-cost-measurement.md の cold/warm 差**: `CUDA_CACHE_PATH`（**ドライバ側 PTX→SASS JIT キャッシュ**）の付け替えによる差。同ドキュメント「コールド／ウォームの検証」節が明記するとおり「NVRTC の source→PTX コンパイルはキャッシュ状態に関係なく毎プロセス発生」し、この実測はドライバ側キャッシュの寄与のみを捉えている（NVRTC コンパイルコスト自体は cold/warm いずれでも発生するため、この差には現れない）
- **本イシューのキャッシュ実測（§3.2）**: 本リポジトリ実装の **NVRTC PTX テキストのディスクキャッシュ**（`RUST_AI_CUDA_CACHE_DIR` 系。`nvrtc.rs`）による「NVRTC source→PTX コンパイルそのものを回避できるか」の計測であり、startup-cost-measurement.md が「本ハーネスの計測粒度では分離できない」としていた NVRTC source→PTX と driver PTX→SASS の寄与のうち、**前者（NVRTC コンパイル区間）を直接計測する**

したがって両者は加算的な関係にある可能性が高い（本リポキャッシュがヒットした場合、NVRTC コンパイル区間の短縮 ＋ ドライバ側 JIT キャッシュがヒットしていれば PTX→SASS 区間の短縮、の両方が本番経路〈C-4 結線後〉で得られる見込み）が、本イシューの計測プリミティブ直叩き方式では本番経路の実際の合算効果までは実証できない（§1 参照）。

## 5. 引き継ぎ・スコープ外

- **本番経路での再計測（C-4・#511 完了後）**: C-4 がミス→コンパイル→store→hit 導線を `CudaGemmAuto::run_f16` へ結線した後、本イシューと同じ 5 回計測中央値プロトコルで「プロセス再起動をまたいだ本番経路の初回起動 vs 2 回目起動」を再計測することが望ましい。本イシューの成果物（§3 の一次データ）はその際のキャッシュ I/O プリミティブ単体コストの参照値として使える
- **#539（手順ドキュメント）との役割分担**: #539 は実機検証手順の文書化を担当し、本イシュー（#534）は計測コード・一次実測データの記録を担当する。役割の重複はない

## 6. 通常 CI で機械検証済みの事項

- `cargo fmt --all -- --check`
- `cargo clippy --workspace --all-targets --all-features -- -D warnings`
- `cargo test -p backend-cuda`（非 ignore テスト全 green。新規モジュールはコンパイル検査。`cargo test -p backend-cuda --lib -- --list` で `nvrtc::jit_cache_bench_tests::*` の 2 テストが登録されていることを確認済み）
- `cargo test --workspace --all-features`（CI の test ジョブ相当。全 green）
- `cargo build --workspace`（`build-no-cuda-toolkit` 契約: `#[ignore]` 分離によりビルド成立を確認）
- `git diff --stat`（`kernels_*.rs`・tolerance 定数・parity ベースライン fixture に差分がないことを確認済み。`crates/backend-cuda/src/nvrtc.rs` への追加は `#[cfg(test)] mod` 登録 1 行〈コメント込み 11 行〉のみ）
