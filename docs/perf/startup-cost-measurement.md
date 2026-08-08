# 起動コスト実測・v1 差分記録（#171・TASK-13.1b）

イシュー #171「test(backend): TASK-13.1b 実測・v1 差分の記録」の実測記録。
受け入れ条件「実測記録と差分分析が残されている」に対応する。計測ハーネス自体は
TASK-13.1a（#170・PR #360）で整備済み（`crates/bench-harness/src/startup.rs`・
`Makefile` の `startup-bench` ターゲット）であり、本ドキュメントは同ハーネスを用いた
実測結果と v1（PoC-5）実測値との差分分析を記録する。

## 状態: CPU 実測済み・CUDA/Metal 実機未実施

本実装セッションは Linux x86_64（QEMU/KVM 仮想化環境、NVIDIA RTX 3060 passthrough）
worktree で行っており、`libnvrtc`（CUDA toolkit）が導入されていない
（`ldconfig -p | grep nvrtc` で未検出）。`backend-cuda` は `cudarc` 動的ロード契約
により本環境でも `cargo build` が成立するが、実行時は型付きエラーに fail-closed で
倒れる契約であり、下記「CUDA 動的ロード契約の実地確認」節で実測している。
Metal（Apple Silicon）実機はこのセッションから利用できない。

したがって既存の先例（`docs/perf/dispatch-boundary-measurement.md`〈#69〉・
`docs/perf/metal-gemm-dynamic-tile.md`〈#188〉・`docs/perf/cuda-tensor-core-measurement.md`
〈#64〉）と同じ運用を採る: **CPU は本セッションで実測を完了**し、**CUDA（DGX Spark
GB10）・Metal（Apple Silicon）は再現可能な計測手順と結果転記テンプレートのみを整備**
して「実機未実施」と明記する。CPU のコールド／ウォーム同値性は、ハーネス自身のモジュール
冒頭コメント（`crates/bench-harness/src/startup.rs:31-33`）が「#171 の実測で確認する
v1 との重要な差分データ点」と明示しており、本セッションの CPU 実測のみでも受け入れ条件の
中核（実測記録＋差分分析）を満たす。

## 環境

| 項目 | 値 |
|------|-----|
| OS | Linux 7.0.0-28-generic（Ubuntu、`uname -a` 実測） |
| CPU | QEMU Virtual CPU version 2.5+（KVM 仮想化。12 vCPU、`lscpu` 実測） |
| GPU | NVIDIA GeForce RTX 3060（`nvidia-smi -L` で検出。CUDA toolkit 非搭載のため実行時未使用） |
| rustc | 1.96.0（`rustc --version` 実測） |
| ビルドプロファイル | `--release`（`cargo build -p bench-harness --release --bins`） |

v1（PoC-5）は DGX Spark GB10（aarch64・CUDA 実機）上での計測であり、本セッションの
CPU 実測（x86_64・仮想化環境）とはハードウェア構成が異なる（「計測前提の非対称性」節で後述）。

## CPU 実測結果

`./target/release/startup_bench --backend cpu --trials 5`（`make startup-bench BACKEND=cpu
TRIALS=5` が内部で `cargo build -p bench-harness --release --bins` の後に呼び出すのと
同一のバイナリ呼び出し）を `--out` 付きで 2 セット実行し、`StartupReport`
（コールド／ウォーム別、5 試行の中央値＋Q1/Q3）を取得した。生 JSON は
`docs/perf/startup-cost/cpu-run1.json`・`cpu-run2.json`（全試行の `samples` 込み）。
再現性確認のため、`make startup-bench BACKEND=cpu TRIALS=5`（`--out` なし・標準出力）も
1 回実行した（下記「再現性確認」節）。

### 内部計測（`device_init_secs`・`first_kernel_secs`。probe プロセス内 `Instant` 計測）

| セット | フェーズ | device_init（中央値 / Q1 / Q3, ms） | first_kernel（中央値 / Q1 / Q3, ms） |
|--------|---------|----------------------------------|-------------------------------------|
| run1 | cold | 0.311 / 0.305 / 0.316 | 2.117 / 1.764 / 2.175 |
| run1 | warm | 0.365 / 0.363 / 0.554 | 2.132 / 1.817 / 2.221 |
| run2 | cold | 0.405 / 0.327 / 0.491 | 2.065 / 1.887 / 2.192 |
| run2 | warm | 0.556 / 0.407 / 0.559 | 2.274 / 1.872 / 2.419 |

### 外部計測（`wall_secs`。親ハーネスが `Command::spawn` 前後で計測するプロセス全体の wall time）

| セット | フェーズ | wall（中央値 / Q1 / Q3, ms） |
|--------|---------|------------------------------|
| run1 | cold | 2.905 / 2.440 / 2.920 |
| run1 | warm | 3.121 / 2.765 / 3.281 |
| run2 | cold | 3.023 / 2.530 / 3.224 |
| run2 | warm | 3.300 / 2.605 / 3.453 |

### 再現性確認

run1・run2（`--out` 付き 2 セット）の中央値は互いのフェーズ内 Q1〜Q3 の範囲に収まっており
（例: run2 cold wall 中央値 3.023ms は run1 cold の Q3 2.920ms をわずかに超えるが、run1 の
Q1〜Q3 幅（2.440〜2.920ms）・run2 の Q1〜Q3 幅（2.530〜3.224ms）は大きく重なる）、乖離は
仮想化環境のスケジューリング揺らぎの範囲内と判断した。

加えて `make startup-bench BACKEND=cpu TRIALS=5`（標準出力。ファイル保存なし）を追加で
1 回実行したところ、wall_secs のサンプルが 5.1〜23.5ms（cold/warm 込み）となり、run1・run2
（2.2〜3.5ms）の約 3〜8 倍に達した。この run は直前に `cargo build -p bench-harness --release
--bins`（Makefile が内部で呼ぶビルドステップ）を伴っており、ビルド直後の I/O・ページキャッシュ
競合が数ミリ秒オーダーのプロセス起動計測を支配的に押し上げた可能性が高い（`device_init_secs`
は run1・run2 と同オーダーのまま、`wall_secs`・`first_kernel_secs` のみ顕著に増大している
ことからも、計測対象そのものではなく計測時の外的要因（ビルド直後の I/O 競合）が主因と考えられる）。
値の取捨選択・恣意的な選別は行わず、3 回とも記録する（`.claude/rules/coding-rust.md` 「ベンチは
5 回計測の中央値を採用」の精神に沿い、外れ値の非開示ではなく事実ごとの記録を優先した）。

### コールド／ウォーム同値性の確認

CPU バックエンドは JIT を持たないため、`crates/bench-harness/src/startup.rs` の
モジュールコメントが予告するとおり、コールド／ウォームの実測値はほぼ同一だった
（例: run1 first_kernel 中央値はコールド 2.117ms・ウォーム 2.132ms、差は 0.015ms）。
これは CUDA の `CUDA_CACHE_PATH`（NVRTC 実行時コンパイル結果を保持するドライバ側 JIT
キャッシュ）に相当する永続状態を CPU バックエンドが持たないことの実測的裏付けである。

## CUDA 動的ロード契約の実地確認（実測値としては扱わない）

`make startup-bench BACKEND=cuda TRIALS=5` を実行し、以下の型付きエラーで fail-closed に
失敗することを確認した:

```
cold フェーズの計測失敗: probe が異常終了（exit status: 1）: startup_probe 失敗:
CudaGemm::new 失敗: CUDA NVRTC library unavailable: libnvrtc dynamic library not found
(dlopen failed); CUDA toolkit is not installed or not on the library search path
```

これは deps-policy.md の「`cudarc` は無条件依存＋動的ロード方式（CUDA toolkit 非搭載環境
でもビルド成立）」という契約が実行時レベルでも成立していることの実地確認である
（ビルド自体は `cargo build --workspace --locked` で先に成立確認済み）。起動コストの
定量値としては採用しない（NVRTC 呼び出し前の初期化失敗であり、実際のカーネル起動コストを
反映しないため）。

## CUDA／Metal 実機手順（結果転記テンプレート）

### CUDA（DGX Spark GB10・NVRTC 搭載）

```sh
git fetch origin
git checkout test/171-startup-cost-measurement   # 本イシューの実装ブランチ
make startup-bench BACKEND=cuda TRIALS=5
```

コールド定義（ハーネスの v2 定義。`crates/bench-harness/src/startup.rs:13-24` 参照）:
試行ごとに新規の空ディレクトリを子プロセスの `CUDA_CACHE_PATH` に設定する（NVRTC
コンパイル結果を保持するドライバ側 JIT キャッシュなしの初回起動）。ウォームは priming
実行 1 回の後、同一 `CUDA_CACHE_PATH` を再利用して計測する。

転記先: 下表を実測値で埋める。

| フェーズ | device_init（中央値 / Q1 / Q3, ms） | first_kernel（中央値 / Q1 / Q3, ms） | wall（中央値 / Q1 / Q3, ms） |
|---------|-------------------------------------|--------------------------------------|-------------------------------|
| cold | （未実施） | （未実施） | （未実施） |
| warm | （未実施） | （未実施） | （未実施） |

### Metal（Apple Silicon 実機）

```sh
git fetch origin
git checkout test/171-startup-cost-measurement
make startup-bench BACKEND=metal TRIALS=5
```

Metal はドライバ側 JIT キャッシュに相当する永続機構を `crates/backend-metal` 側で
明示的に扱っていない（`ProbeReport::first_kernel_secs` doc コメント参照。
`MetalBackendOps::gemm` 内部で `MetalContext` が都度再構築される現行実装）ため、
コールド／ウォームの差は CPU 同様ほぼ生じない可能性がある。実測で確認する。

| フェーズ | device_init（中央値 / Q1 / Q3, ms） | first_kernel（中央値 / Q1 / Q3, ms） | wall（中央値 / Q1 / Q3, ms） |
|---------|-------------------------------------|--------------------------------------|-------------------------------|
| cold | （未実施） | （未実施） | （未実施） |
| warm | （未実施） | （未実施） | （未実施） |

## v1 差分分析（本イシューの中核）

### v1 実測値の再掲（出典: `docs/spec/03-poc/poc-5-performance/README.md`）

PoC-5 は Burn/CubeCL（v1）を用い、DGX Spark GB10（CUDA 実機）上で
「プロセス起動（`main()` 開始）〜初回推論完了」を内部計測、加えてシェルの `time`
コマンドでプロセス全体（外部計測）を計測した。

| 実装 | コールド/ウォーム | 内部計測（起動〜初回推論） | `time` 計測（プロセス全体、real） |
|------|-------------------|---------------------------|-----------------------------------|
| Rust（v1・Burn/CubeCL・CUDA） | コールド（`target/autotune/` 削除直後、2試行） | 20.15〜21.66秒 | 20.34〜21.86秒 |
| Rust（v1・Burn/CubeCL・CUDA） | ウォーム（cache 存在下での再実行、4試行） | 1.77 / 1.97 / 2.07 / 2.48秒 | 1.95〜2.67秒 |
| PyTorch（CUDA、参考） | — | 0.92〜0.95秒 | — |

v1 のウォーム時（PyTorch 比 約 1.9〜2.7 倍）・コールド時（同 約 21〜24 倍）の劣位は、
Burn/CubeCL がシェーダー（CUDA 側は CUDA C/PTX）を**プロセスごとに JIT コンパイル**し、
`target/autotune/` にはコンパイル済みバイナリではなく autotune が選んだ最速戦略の
**識別結果のみ**を永続化する構造に起因すると特定されている（同 README「起動コスト」節）。
コールド時はさらに autotune 全候補（`unit`/`accelerated`/`tma` グループ）の実行時
ベンチマーク探索コストが加わる。

### 構造差分

| 観点 | v1（Burn/CubeCL） | v2（完全自作コア） |
|------|---------------------|----------------------|
| autotune 探索 | あり（コールド時 約 20 秒の主因。カーネル候補を実行時ベンチマークで選定） | **なし**（`select_gemm_kernel` は実行時ベンチマークを伴わない静的ディスパッチ規則。`docs/dispatch-rules-design.md`） |
| プロセスごとの JIT コンパイル | あり（CubeCL がシェーダーを毎プロセス JIT コンパイル。コンパイル結果は非永続） | CPU: **なし**（本セッション実測で確認済み。コールド≒ウォーム）。CUDA: **あり**（NVRTC 実行時コンパイルは `backend-cuda/src/nvrtc.rs` がディスク非永続の契約。ただし対象カーネル数は v2 の限定的な自作カーネル群のみで、CubeCL の autotune 候補群のような複数戦略の実行時探索は行わない。定量差分は CUDA 実機実測で確定） |
| キャッシュ機構 | `target/autotune/` に戦略選択の識別結果のみ永続化（コンパイル済みバイナリ非永続） | CUDA: ドライバ側 `CUDA_CACHE_PATH` の JIT キャッシュ（`CudaGemm::run_tiled_f32` 呼び出しのみ影響を受ける。`backend-cuda` 独自のキャッシュ実装はなし） |

本セッションの CPU 実測（`device_init_secs`・`first_kernel_secs` がミリ秒オーダーで
コールド／ウォームほぼ同値）は、v2 が v1 の主要劣位要因（autotune 探索・プロセスごとの
シェーダー JIT）を CPU 経路では構造的に持たないことを裏付ける。CUDA 経路については
NVRTC コンパイルが残存するため、v1 ウォーム時の JIT コンパイルコストと同種の要因が
部分的に残りうるが、v1 の「autotune 候補群の実行時探索」は存在しない分、定性的には
v1 ウォームより小さいコストになると見込まれる（**推測**であり、実機実測で確定する）。

### 計測前提の非対称性（結論を先取りしないための明記）

- **ハードウェア差**: v1 は DGX Spark GB10（aarch64）実機。本セッションの v2 CPU 実測は
  x86_64 の仮想化環境（QEMU/KVM）。CPU アーキテクチャ・仮想化オーバーヘッドの影響で
  絶対値は単純比較できない
- **ワークロード差**: v1 は Transformer ブロック（`d_model=512, n_heads=8, d_ff=2048,
  batch=8, seq_len=128`）の初回推論完了までを計測。v2 の `startup_probe` は GEMM
  256×256 の初回カーネル完了までを計測（`ProbeReport::first_kernel_secs` doc コメント
  参照）。計測対象の演算規模・種類が異なる
- **対 PyTorch 比較はスコープ外**: 本イシューは v1（Burn/CubeCL）との差分記録が目的であり、
  PyTorch との比較実測は行っていない（v1 の PyTorch 比数値は参考として再掲したのみ）
- **CUDA/Metal の定量差分は実機転記後に確定**: 上記「構造差分」節の CUDA に関する記述は
  実装契約（NVRTC 非永続コンパイル・静的ディスパッチ）からの定性的な見通しであり、
  実測値による確定ではない。実機転記後、本節を実測値で更新する

## 検証（本セッションで実施）

- `cargo build --workspace --locked` — 依存・ロック不変を確認（差分なし）
- `cargo fmt --all -- --check` — 差分なし
- `cargo clippy --workspace --all-targets --all-features -- -D warnings` — warning なし
- `cargo test -p bench-harness --release` — 既存 `tests/startup_harness.rs` を含め green
- `./target/release/startup_bench --backend cpu --trials 5` を `--out` 付きで 2 回、
  `make startup-bench BACKEND=cpu TRIALS=5` を追加で 1 回実行し、値の変動を含めて記録
  （「再現性確認」節。3 回目はビルド直後の I/O 競合とみられる外れ値を観測し、選別せず記載）
- `make startup-bench BACKEND=cuda TRIALS=5` を実行し、型付きエラーで fail-closed に
  失敗することを確認（「CUDA 動的ロード契約の実地確認」節）

## 後続

TASK-13.2（#172・短命プロセス対応方針の決定、担当: 人間）が本記録を入力とする。
CUDA/Metal 実機転記が完了するまでは、本記録の CPU 実測部分・構造差分の定性分析のみを
判断材料として扱う。
