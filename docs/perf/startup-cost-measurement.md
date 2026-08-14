# 起動コスト実測・v1 差分記録（#171・TASK-13.1b）

イシュー #171「test(backend): TASK-13.1b 実測・v1 差分の記録」の実測記録。
受け入れ条件「実測記録と差分分析が残されている」に対応する。計測ハーネス自体は
TASK-13.1a（#170・PR #360）で整備済み（`crates/bench-harness/src/startup.rs`・
`Makefile` の `startup-bench` ターゲット）であり、本ドキュメントは同ハーネスを用いた
実測結果と v1（PoC-5）実測値との差分分析を記録する。

## 状態: CPU・Metal・CUDA 実測済み

CPU 実測は Linux x86_64（QEMU/KVM 仮想化環境、NVIDIA RTX 3060 passthrough）
worktree で行っており、`libnvrtc`（CUDA toolkit）が導入されていない
（`ldconfig -p | grep nvrtc` で未検出）。`backend-cuda` は `cudarc` 動的ロード契約
により本環境でも `cargo build` が成立するが、実行時は型付きエラーに fail-closed で
倒れる契約であり、下記「CUDA 動的ロード契約の実地確認」節で実測している。
Metal（Apple Silicon）実機はイシュー #384 で、CUDA（DGX Spark GB10）実機は
イシュー #391 でそれぞれ実測完了した（下記「Metal 実測結果」節・「CUDA 実測結果」節）。

CPU は本ドキュメント初版（#171）のセッションで実測を完了し、Metal・CUDA は
それぞれ後続イシュー（#384・#391）の実機セッションで実測を完了した。3 バックエンド
すべての実測が揃ったことで、REQ-13 の受け入れ条件（実測記録＋差分分析）を
全バックエンド横断で満たす。CPU のコールド／ウォーム同値性は、ハーネス自身の
モジュール冒頭コメント（`crates/bench-harness/src/startup.rs:31-33`）が「#171 の
実測で確認する v1 との重要な差分データ点」と明示しており、CUDA は本イシュー（#391）
が「ドライバ JIT キャッシュ機構が実効的に働く」ことを初めて検証する対象である
（下記「CUDA 実測結果」節「コールド／ウォームの検証」参照）。

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

## Metal 実測結果

イシュー #384（親 #379「Metal 実機検証・ベンチ計測トラッキング」配下）の実測。
Apple Silicon 実機（Apple M4 Max）で `./target/release/startup_bench --backend metal
--trials 5` を `--out` 付きで 2 セット実行し、`StartupReport`（コールド／ウォーム別、
5 試行の中央値＋Q1/Q3）を取得した。生 JSON は
`docs/perf/startup-cost/metal-run1.json`・`metal-run2.json`（全試行の `samples` 込み）。
再現性確認のため `make startup-bench BACKEND=metal TRIALS=5`（`--out` なし・標準出力）
も 1 回実行した。

### 実行環境

| 項目 | 値 |
|------|-----|
| チップ | Apple M4 Max（`sysctl -n machdep.cpu.brand_string` 実測） |
| OS | macOS 26.6（build 25G72、`sw_vers` 実測） |
| toolchain | stable（`rust-toolchain.toml` 準拠） |
| rustc | 1.96.0（`rustc --version` 実測） |
| MSL コンパイラ版 | 取得不可（`xcrun metal --version` が `missing Metal Toolchain` エラー。本ホストに Metal Toolchain 単体コンポーネントが別途導入されていない。`backend-metal` のビルド・実行自体は Xcode 付属ランタイムで成立しており計測に影響なし） |
| 計測リビジョン | `3f7203975887ef3836a003db888b56c29232ccf6`（`perf/384-metal-startup-cost` ブランチの計測時点の HEAD。本 PR マージ前で main 未マージ。`git rev-parse HEAD`） |
| ビルドプロファイル | `--release`（`cargo build -p bench-harness --release --bins`） |
| 実施日 | 2026-08-10 |
| 実行コマンド | `./target/release/startup_bench --backend metal --trials 5 --out docs/perf/startup-cost/metal-run{1,2}.json` および `make startup-bench BACKEND=metal TRIALS=5` |

### 内部計測（`device_init_secs`・`first_kernel_secs`。probe プロセス内 `Instant` 計測）

| セット | フェーズ | device_init（中央値 / Q1 / Q3, ms） | first_kernel（中央値 / Q1 / Q3, ms） |
|--------|---------|----------------------------------|-------------------------------------|
| run1 | cold | 35.280 / 34.823 / 38.207 | 42.649 / 41.903 / 44.634 |
| run1 | warm | 34.589 / 34.418 / 35.476 | 42.026 / 41.194 / 43.779 |
| run2 | cold | 37.786 / 33.870 / 38.084 | 43.885 / 42.251 / 44.475 |
| run2 | warm | 35.312 / 35.073 / 35.556 | 42.785 / 41.828 / 43.533 |

### 外部計測（`wall_secs`。親ハーネスが `Command::spawn` 前後で計測するプロセス全体の wall time）

| セット | フェーズ | wall（中央値 / Q1 / Q3, ms） |
|--------|---------|------------------------------|
| run1 | cold | 46.914 / 45.999 / 48.883 |
| run1 | warm | 46.181 / 45.582 / 48.045 |
| run2 | cold | 47.984 / 46.510 / 48.665 |
| run2 | warm | 47.102 / 46.032 / 47.959 |

### near-cold 観測点

run1 cold の**第 1 サンプル**（出典: `metal-run1.json` の `samples[0]`）は
`wall_secs = 0.393817417`（393.817ms）・`device_init_secs = 0.038207084`（38.207ms、
中央値と同オーダー）・`first_kernel_secs = 0.110939709`（110.940ms）であり、
run1 cold の中央値（wall 46.914ms・first_kernel 42.649ms）と比べて wall で約 8.4 倍、
first_kernel で約 2.6 倍大きい。`device_init_secs` は中央値と同水準にとどまる一方
`first_kernel_secs`・`wall_secs` のみ突出しており、`MetalGemm::new` が `gemm.metal`
を毎プロセス runtime コンパイルする経路（後述「構造差分」節）のうち、システム側
Metal コンパイラキャッシュが本セッションで未温だった最初の呼び出しでコンパイル
コストが顕在化したと解釈できる。run2 の第 1 サンプル（`metal-run2.json`）では
同様の突出は見られず（wall 48.866ms、run2 cold の Q3 48.665ms とほぼ同水準）、
run1 実行によりシステムキャッシュが温まった結果と整合する。値の取捨選択はせず、
中央値表には全 5 試行を含めたまま算出した値を掲載している。

### `make startup-bench BACKEND=metal TRIALS=5` 実行結果（再現性確認）

標準出力（`--out` なし）で得た中央値は run1・run2 の Q1〜Q3 範囲に収まった。

| フェーズ | device_init（中央値, ms） | first_kernel（中央値, ms） | wall（中央値, ms） |
|---------|---------------------------|------------------------------|----------------------|
| cold | 34.648 | 41.727 | 46.597 |
| warm | 33.655 | 41.438 | 45.795 |

### コールド／ウォームの検証

既存文書（実装当時）の予告「Metal はコールド／ウォーム差がほぼ生じない可能性がある。
実測で確認する」に対する回答: **中央値ベースでは cold ≒ warm** であり、上表のとおり
device_init・first_kernel・wall のいずれも cold/warm 間の差はミリ秒未満（run1 cold
first_kernel 42.649ms に対し warm 42.026ms、差 0.623ms）で、run1・run2 間の揺らぎと
同程度に収まる。

ただし、この「cold ≒ warm」を「Metal に毎プロセスのシェーダーコンパイルコストが
存在しない証拠」と読み違えてはならない。ハーネスのコールド定義（`startup.rs:23-24,
737-740`）は **`CUDA_CACHE_PATH` の付け替えのみ**を行う実装であり、Metal 側には
対応する制御機構がない（Metal に対しては無作用）。したがって本ハーネスの
cold/warm フラグは Metal 実行に対して実質的に同一条件（両方とも「ハーネスからは
何も制御していない」状態）を計測しているにすぎない。cold≒warm という結果は
「ハーネスのコールド定義が Metal に効かないことの帰結」であって「Metal のシェーダー
コンパイルコストがゼロであること」の証拠ではない。

加えて、macOS はコンパイル済み Metal 関数を**システムレベル**（MTLCompilerService・
ユーザーごとキャッシュ）で保持し、本ハーネスはそのキャッシュ機構に一切触れない
（削除もしない）。したがって本節の計測は「本セッション内で 2 回目以降に観測される
定常状態」を捉えたものであり、**真のコールド**（システムキャッシュも含め完全に
未温な状態）は測定していない。近似的な観測点は上記「near-cold 観測点」節の
run1 cold 第 1 サンプルのみであり、これも「本プロセスにとっての 1 回目」を意味する
だけでシステムキャッシュの状態までは保証しない限界がある。

## CUDA 実測結果

イシュー #391（REQ-13・本ドキュメント CUDA 節の実機実測）の実測。DGX Spark GB10
実機（`<cuda-node>`。`docs/real-hardware-verification-env.md` 2 節）で
`./target/release/startup_bench --backend cuda --trials 5` を `--out` 付きで 2 セット
実行し、`StartupReport`（コールド／ウォーム別、5 試行の中央値＋Q1/Q3）を取得した。
生 JSON は `docs/perf/startup-cost/cuda-run1.json`・`cuda-run2.json`（全試行の
`samples` 込み・回収後無編集）。再現性確認のため `make startup-bench BACKEND=cuda
TRIALS=5`（`--out` なし・標準出力）も 1 回実行した。

### 実行環境

| 項目 | 値 |
|------|-----|
| GPU | NVIDIA GB10（sm_121）・driver 580.159.03（`nvidia-smi` 実測） |
| OS | Linux 6.17.0-1026-nvidia aarch64（`uname -srm` 実測） |
| CUDA (nvcc) | release 13.0, V13.0.88（`nvcc --version` 実測。`docs/real-hardware-verification-env.md` 2.1 節の既存記録と一致） |
| toolchain | stable（`rust-toolchain.toml` 準拠） |
| rustc | 1.97.0（`rustc --version` 実測） |
| 計測リビジョン | `c9c06ef124632a2816f407bef2f9f3b33540255c`（origin/main 現在の HEAD。`.rev-stamp` で転送先ノード上での一致を確認済み。本 PR〈`perf/391-cuda-startup-cost`〉のドキュメント変更コミット前の時点） |
| ビルドプロファイル | `--release`（`cargo build -p bench-harness --release --bins`） |
| 実施日 | 2026-08-10 |
| 実行コマンド | `./target/release/startup_bench --backend cuda --trials 5 --out docs/perf/startup-cost/cuda-run{1,2}.json` および `make startup-bench BACKEND=cuda TRIALS=5` |
| GPU 占有状況 | 計測前後とも常駐 2 プロセスのみ（実名・使用量はローカル版 `docs/real-hardware-verification-env.local.md` 参照）・`utilization.gpu` 0%（`docs/real-hardware-verification-env.md` 6.1 節の手順で確認） |

### 内部計測（`device_init_secs`・`first_kernel_secs`。probe プロセス内 `Instant` 計測）

両値とも `process_start`（probe プロセス内で最初に取得する `Instant`）からの
**累積経過時間**であり、`device_init_secs` と `first_kernel_secs` は連続する
区間の加算対象ではない（`crates/bench-harness/src/bin/startup_probe.rs::run_cuda`
L138・L159。`first_kernel_secs` は driver 初期化コストを内包した「起動〜初回
カーネル完了まで」の累積値）。実測値は全試行で
`device_init_secs < first_kernel_secs < wall_secs` の順序が成立しており
（`wall_secs` は親ハーネス側の `Command::spawn` 前後計測で probe プロセス
起動自体のオーバーヘッドを含むぶん更に大きい）、この累積関係と整合する。

| セット | フェーズ | device_init（中央値 / Q1 / Q3, ms） | first_kernel（中央値 / Q1 / Q3, ms） |
|--------|---------|----------------------------------|-------------------------------------|
| run1 | cold | 204.316 / 204.001 / 204.761 | 530.805 / 530.191 / 538.994 |
| run1 | warm | 198.326 / 197.217 / 198.869 | 322.028 / 321.890 / 322.898 |
| run2 | cold | 196.358 / 193.666 / 199.981 | 510.810 / 507.336 / 510.860 |
| run2 | warm | 192.543 / 191.217 / 192.653 | 308.080 / 307.476 / 310.231 |

### 外部計測（`wall_secs`。親ハーネスが `Command::spawn` 前後で計測するプロセス全体の wall time）

| セット | フェーズ | wall（中央値 / Q1 / Q3, ms） |
|--------|---------|------------------------------|
| run1 | cold | 641.353 / 641.037 / 648.973 |
| run1 | warm | 432.287 / 429.987 / 434.082 |
| run2 | cold | 621.301 / 618.455 / 621.382 |
| run2 | warm | 419.264 / 418.324 / 419.710 |

### `make startup-bench BACKEND=cuda TRIALS=5` 実行結果（再現性確認）

標準出力（`--out` なし）で得た中央値は run1・run2 の Q1〜Q3 範囲に収まった。

| フェーズ | device_init（中央値, ms） | first_kernel（中央値, ms） | wall（中央値, ms） |
|---------|---------------------------|------------------------------|----------------------|
| cold | 195.654 | 506.808 | 616.082 |
| warm | 195.463 | 311.509 | 423.122 |

### コールド／ウォームの検証

Metal（#384）の「ハーネスのコールド定義は Metal に無作用」というフレーミングは
CUDA には当てはまらない。CUDA はハーネスのコールド定義（試行ごとに新規の空
ディレクトリを `CUDA_CACHE_PATH` に設定する。`startup.rs` モジュール冒頭「コールド／
ウォームの v2 定義」節）が実際に機能する対象であり、本イシューが初めてこの機構を
実効的に検証した。

実測結果: **cold と warm で明確な差が生じた**（run1 first_kernel 中央値: cold
530.805ms・warm 322.028ms、差 208.777ms。run2 も同様に cold 510.810ms・warm
308.080ms、差 202.730ms）。device_init（driver 初期化・`CudaDevice::new` の
dlopen＋初期化コスト）は cold/warm でほぼ同水準（run1: 204.316ms / 198.326ms、
run2: 196.358ms / 192.543ms）であり、cold/warm 差は主に `first_kernel_secs`
（`CudaGemm::new`〜初回カーネル完了。NVRTC コンパイル＋転送を含む）側に現れている。

ただし `crates/bench-harness/src/startup.rs`（モジュール冒頭）と
`crates/backend-cuda/src/nvrtc.rs` の契約により、**NVRTC の source→PTX
コンパイルはキャッシュ状態に関係なく毎プロセス発生**し、`CUDA_CACHE_PATH` が
効くのはドライバ側 PTX→SASS JIT キャッシュのみである。したがって本実測が
示すのは「**ドライバ JIT キャッシュ（PTX→SASS）の寄与は無視できない規模で
存在する**」ことであり、「NVRTC コンパイルコストが cold/warm 差の全てを
説明する」あるいは「コンパイルコストが無い」という解釈のどちらも誤りである。
両者の寄与の内訳（NVRTC source→PTX と driver PTX→SASS の分離計測）は本
ハーネスの計測粒度では分離できず、本イシューのスコープ外とする。

`device_init_secs` は CUDA 経路のみ `CudaDevice::new`（dlopen ＋ driver 初期化）を
明示的に含む（`startup.rs` の `ProbeReport::device_init_secs` doc、
`bin/startup_probe.rs::run_cuda`）。CPU（約 0.3〜0.6ms・ハンドル構築のみ）・
Metal（約 35ms）とは性質が異なる参照点であり、CUDA の約 195〜205ms は
dlopen・CUDA コンテキスト生成のコストを反映した実測値である。

`first_kernel_secs` には `CudaGemm::run_tiled_f32` の `clone_htod`／`clone_dtoh`
（H2D/D2H 転送）が含まれる（`startup.rs` の既知の注記）。カーネル単体時間では
ない点は CPU・Metal 実測と同様に留意する。

外れ値の選別は行っていない（両セットとも 5 試行全てが Q1〜Q3 の狭い幅に収まって
おり、near-cold 観測点のような突出サンプルは観測されなかった）。

## CUDA 動的ロード契約の実地確認（実測値としては扱わない）

本節は CUDA toolkit 非搭載の CPU 実測環境（本ドキュメント「環境」節・Linux x86_64
QEMU/KVM）で実施した確認であり、下記「CUDA 実測結果」節（DGX Spark GB10・CUDA
toolkit 搭載環境での実測）とは対象環境が異なる。両者は矛盾するものではなく、
「`cudarc` は toolkit 非搭載でもビルドは成立し実行時に fail-closed で倒れる」
（本節）と「toolkit 搭載環境では実際に起動・計測できる」（CUDA 実測結果節）という
異なる環境での契約確認である。

`make startup-bench BACKEND=cuda TRIALS=5` を（CUDA toolkit 非搭載の CPU 実測環境で）
実行し、以下の型付きエラーで fail-closed に失敗することを確認した:

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

## 実機再現手順（CUDA・Metal ともに実測済み）

### CUDA（DGX Spark GB10・NVRTC 搭載・イシュー #391 で実測済み）

転送・実行方法の正本は `docs/real-hardware-verification-env.md`（2〜4 節）。
git clone/fetch はノード側で使えないため rsync で転送する（同ドキュメント 3 節）。
実行前に `export CUDA_NODE="<実ホスト名>"`（山括弧のクォート必須。未クォートだと `export` 自体がリダイレクトと誤解釈される。同ドキュメント冒頭の注記・
`docs/real-hardware-verification-env.local.md` 参照）を設定する。

```sh
# Mac 側 worktree ルートで実行（.rev-stamp によるリビジョン記録・rsync 転送）
git rev-parse HEAD > .rev-stamp
rsync -a --delete --delete-excluded --filter=':- .gitignore' \
  --exclude '.git/' --exclude '.codex/' --exclude '.env*' \
  --exclude '.claude/settings.local.json' --exclude '.venv*/' \
  ./ "$CUDA_NODE":~/work/rust-ai-library-run/
rm .rev-stamp

# ノード側で実行（--out は同期ツリー外の $HOME/work/ に置く。--delete-excluded で
# 消えないようにするため）
ssh "$CUDA_NODE" 'cd ~/work/rust-ai-library-run && \
  env PATH=$HOME/.cargo/bin:/usr/local/cuda/bin:$PATH \
      CARGO_TARGET_DIR=$HOME/work/target-rust-ai-library \
  make startup-bench BACKEND=cuda TRIALS=5'
```

コールド定義（ハーネスの v2 定義。`crates/bench-harness/src/startup.rs` モジュール
冒頭「コールド／ウォームの v2 定義」節参照）: 試行ごとに新規の空ディレクトリを
子プロセスの `CUDA_CACHE_PATH` に設定する（NVRTC コンパイル結果を保持するドライバ側
JIT キャッシュなしの初回起動）。ウォームは priming 実行 1 回の後、同一
`CUDA_CACHE_PATH` を再利用して計測する。

実測結果は上記「CUDA 実測結果」節を参照（結論: cold と warm で明確な差が生じた。
ただし解釈上の注意点は同節「コールド／ウォームの検証」を参照）。

| フェーズ | device_init（中央値 / Q1 / Q3, ms） | first_kernel（中央値 / Q1 / Q3, ms） | wall（中央値 / Q1 / Q3, ms） |
|---------|-------------------------------------|--------------------------------------|-------------------------------|
| cold（run1） | 204.316 / 204.001 / 204.761 | 530.805 / 530.191 / 538.994 | 641.353 / 641.037 / 648.973 |
| warm（run1） | 198.326 / 197.217 / 198.869 | 322.028 / 321.890 / 322.898 | 432.287 / 429.987 / 434.082 |

### Metal（Apple Silicon 実機・イシュー #384 で実測済み）

```sh
git fetch origin
git checkout perf/384-metal-startup-cost   # 本イシューの実装ブランチ（計測 SHA: 3f7203975887ef3836a003db888b56c29232ccf6 由来）
make startup-bench BACKEND=metal TRIALS=5
```

Metal はドライバ側 JIT キャッシュに相当する永続機構を `crates/backend-metal` 側で
明示的に扱っていない（`ProbeReport::first_kernel_secs` doc コメント参照。
`MetalBackendOps::gemm` 内部で `MetalContext` が都度再構築される現行実装）ため、
コールド／ウォームの差は CPU 同様ほぼ生じないと予想していた。実測結果は上記
「Metal 実測結果」節を参照（結論: 中央値ベースでは cold ≒ warm。ただし解釈上の
注意点は同節「コールド／ウォームの検証」を参照）。

| フェーズ | device_init（中央値 / Q1 / Q3, ms） | first_kernel（中央値 / Q1 / Q3, ms） | wall（中央値 / Q1 / Q3, ms） |
|---------|-------------------------------------|--------------------------------------|-------------------------------|
| cold（run1） | 35.280 / 34.823 / 38.207 | 42.649 / 41.903 / 44.634 | 46.914 / 45.999 / 48.883 |
| warm（run1） | 34.589 / 34.418 / 35.476 | 42.026 / 41.194 / 43.779 | 46.181 / 45.582 / 48.045 |

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
| プロセスごとの JIT コンパイル | あり（CubeCL がシェーダーを毎プロセス JIT コンパイル。コンパイル結果は非永続） | CPU: **なし**（本セッション実測で確認済み。コールド≒ウォーム）。CUDA: **あり**（NVRTC の source→PTX コンパイルはキャッシュ状態に関係なく毎プロセス発生する契約。`backend-cuda/src/nvrtc.rs`。イシュー #391 実測で cold/warm 差 約 203〜209ms〈first_kernel 中央値〉を確認。ただし対象カーネル数は v2 の限定的な自作カーネル群のみで、CubeCL の autotune 候補群のような複数戦略の実行時探索は行わない）。**Metal: あり**（`MetalGemm::new` が `gemm.metal` を毎プロセス runtime コンパイル。ただし CUDA 同様 autotune 探索は無い。イシュー #384 実測で near-cold 観測点〈第 1 サンプル〉のみコンパイルコストと見られる突出〈first_kernel 約 2.6 倍〉を確認） |
| キャッシュ機構 | `target/autotune/` に戦略選択の識別結果のみ永続化（コンパイル済みバイナリ非永続） | CUDA: ドライバ側 `CUDA_CACHE_PATH` の JIT キャッシュ（`CudaGemm::run_tiled_f32` 呼び出しのみ影響を受ける。`backend-cuda` 独自のキャッシュ実装はなし）。**Metal: OS 側 Metal コンパイラキャッシュ**（MTLCompilerService・ユーザーごとキャッシュ。`backend-metal` 側に独自キャッシュ実装はなく、本ハーネスもこのキャッシュに触れない） |

本セッションの CPU 実測（`device_init_secs`・`first_kernel_secs` がミリ秒オーダーで
コールド／ウォームほぼ同値）は、v2 が v1 の主要劣位要因のうち **autotune 探索**を
CPU 経路では構造的に持たないことを裏付ける。ただし「プロセスごとのシェーダー JIT」
自体は v2 でも Metal・CUDA の両バックエンドに残存する（イシュー #384 の Metal 実測
で near-cold 観測点に compile コストとみられる突出を確認、イシュー #391 の CUDA
実測で cold/warm 間に明確な差〈first_kernel 中央値差 約 203〜209ms〉を確認したことが
その裏付け）。v1 との定量比較で言えることは、v2 は v1 のような**複数戦略の実行時
ベンチマーク探索**（autotune）を持たない点であり、単一シェーダーの毎プロセス
コンパイル自体は Metal・CUDA いずれのバックエンドでも構造的に残る。

**v1 との定量対比（CUDA・本イシューの実質的価値）**: v1（Burn/CubeCL・CUDA）の
ウォーム時内部計測は 1.77〜2.48**秒**（1770〜2480ms）だったのに対し、v2（自作コア・
CUDA）の `first_kernel` 中央値（ウォーム、run1/run2）は 308.080〜322.028**ms** で
あり、**約 5.5〜8.0 倍**高速である（`1.77s / 0.322s ≈ 5.5`〜`2.48s / 0.308s ≈ 8.0`）。
v1 のコールド時内部計測 20.15〜21.66 秒（autotune 探索込み）に対し、v2 の
`first_kernel` 中央値（コールド）は 506.808〜530.805ms であり、**約 38〜43 倍**
高速である。ただしこの差の大半は v1 側の autotune 探索コスト（コールド時の主因）
に起因するものであり、後述「計測前提の非対称性」節のとおりワークロード・
ハードウェア構成が異なるため単純な性能比較として断定はできない。それでも
autotune を持たない静的ディスパッチ（v2）が起動コストの観点で優位という
定性的な結論を数値で裏付ける結果である。

### 計測前提の非対称性（結論を先取りしないための明記）

- **ハードウェア差**: v1 は DGX Spark GB10（aarch64）実機。本セッションの v2 CPU 実測は
  x86_64 の仮想化環境（QEMU/KVM）。CPU アーキテクチャ・仮想化オーバーヘッドの影響で
  絶対値は単純比較できない
- **ワークロード差**: v1 は Transformer ブロック（`d_model=512, n_heads=8, d_ff=2048,
  batch=8, seq_len=128`）の初回推論完了までを計測。v2 の `startup_probe` は GEMM
  256×256 の初回カーネル完了までを計測（`ProbeReport::first_kernel_secs` doc コメント
  参照）。計測対象の演算規模・種類が異なる。CUDA 実測（#391）も同一の GEMM 256×256 を
  用いており、上記「v1 との定量対比」の倍率はこの非対称性を含んだ数値である
- **Metal 分**: v1 = DGX Spark GB10（CUDA 実機）・Transformer ブロック初回推論。
  v2 Metal 実測（#384） = Apple M4 Max・256×256 GEMM。ハードウェア（CUDA GPU vs
  Apple GPU）・ワークロードの両方が異なり絶対値の単純比較はできない。加えて
  `startup_probe.rs` の `GEMM_SIZE` 固定・`MetalBackendOps::gemm` 内部で
  `MetalContext` が呼び出しごとに再構築される現行実装（既知の制約。PoC 段階の
  簡略化であり本イシューでは変更しない）が計測値に混入している点は CPU・CUDA
  分析時と同様に留意する
- **対 PyTorch 比較はスコープ外**: 本イシューは v1（Burn/CubeCL）との差分記録が目的であり、
  PyTorch との比較実測は行っていない（v1 の PyTorch 比数値は参考として再掲したのみ）
- **CUDA/Metal の定量差分は実測値で確定済み**: 上記「構造差分」節の CUDA・Metal に
  関する記述は、それぞれイシュー #391・#384 の実機実測値（上記「CUDA 実測結果」・
  「Metal 実測結果」節）で裏付けられている（実装契約からの定性的な見通しの段階は
  解消済み）

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

### 検証（イシュー #384・Apple M4 Max 実機で実施）

- `./target/release/startup_bench --backend metal --trials 5` を `--out` 付きで 2 回
  （`metal-run1.json`・`metal-run2.json`）、`make startup-bench BACKEND=metal TRIALS=5`
  を追加で 1 回実行（「Metal 実測結果」節）
- `make startup-bench BACKEND=bogus`・`make startup-bench TRIALS=abc` が引数検証の
  許可リストにより exit 2 で fail-closed に失敗することを確認（既存防御の非回帰確認）
- `cargo fmt --all -- --check` — 差分なし
- `cargo clippy --workspace --all-targets --all-features -- -D warnings` — warning なし
- `cargo build --workspace --locked` — `Cargo.lock` 変更なしで成功
- `cargo test -p bench-harness --release` — 既存 `tests/startup_harness.rs`（CPU
  cold/warm E2E）を含め green
- `git diff --stat origin/main` — `docs/perf/**` のみ

### 検証（イシュー #391・DGX Spark GB10 実機で実施）

- 計測前後で `nvidia-smi --query-compute-apps` を確認し、常駐 2 プロセス
  （実名はローカル版 `docs/real-hardware-verification-env.local.md` 参照）のみ・`utilization.gpu` 0% であることを確認（「CUDA 実測結果」
  節「実行環境」表）
- `./target/release/startup_bench --backend cuda --trials 5` を `--out` 付きで 2 回
  （`cuda-run1.json`・`cuda-run2.json`）実行し、いずれも `ProbeTimeout`・
  `ProbeExitFailure` なく正常終了することを確認
- `make startup-bench BACKEND=cuda TRIALS=5`（`--out` なし・標準出力）を追加で 1 回
  実行し、中央値が run1・run2 の Q1〜Q3 範囲に収まることを確認（再現性確認）
- 回収した `cuda-run1.json`・`cuda-run2.json` をパース確認（`samples` 5 件・
  cold/warm 2 レポート、手編集なし）
- 既存の `#[ignore]` E2E テスト（`tests/startup_harness.rs::
  cuda_cold_and_warm_phases_are_reproducibly_measurable_on_real_hardware`）を
  `cargo test -p bench-harness --release -- --ignored` で実機実行し pass を確認
- rsync 転送後、ノード側で `find . -name ".env*" -o -name "settings.local.json"` が
  空であることを確認（秘密情報非混入の確認。`docs/real-hardware-verification-env.md`
  3 節）
- ローカル（Mac・worktree）で `cargo build --workspace --locked`・
  `cargo fmt --all -- --check`・`cargo clippy --workspace --all-targets
  --all-features -- -D warnings`・`cargo test -p bench-harness --release` を実施
  （docs のみの変更のため差分なし・green を確認）
- `git diff --stat origin/main` — `docs/perf/**` のみ

## 後続

TASK-13.2（#172・短命プロセス対応方針の決定、担当: 人間）が本記録を入力とする。
CPU・Metal・CUDA の 3 バックエンドすべての実機実測が完了し、本記録は起動コスト
実測に関する REQ-13 の受け入れ条件を満たした状態にある。

- **下流ドキュメント同期**: `docs/short-lived-process-decision.md`（再判定トリガー
  「CUDA と Metal の両方の転記完了」が本イシューで満たされた旨の反映）・
  `docs/backend-matrix.md` への反映は、#387（総括反映）・#172（人間判断）の
  スコープとし本 PR には含めない
- **行番号アンカーの参照ずれ**: 本改訂により本文書内の行番号がずれるため、本文書を
  行番号アンカーで引用している下流ドキュメントがあれば、今後は節見出し参照へ寄せる
  ことが望ましい
- **bench-harness 側 Metal `#[ignore]` E2E テスト**: `tests/startup_harness.rs` には
  CUDA 版（`cuda_cold_and_warm_phases_are_reproducibly_measurable_on_real_hardware`。
  本イシューで実機実行し pass 確認済み）が既存だが、Metal 版は未追加。追加は
  本 PR のスコープ外（受入条件は記録のみ）。後続候補としてここに記す
- **NVRTC source→PTX と driver PTX→SASS の寄与分離**: 本イシューの CUDA 実測は
  cold/warm 差（ドライバ JIT キャッシュの寄与を含む）を確認したが、NVRTC コンパイル
  自体のコストとドライバ側キャッシュの寄与を分離計測してはいない（「CUDA 実測結果」
  節「コールド／ウォームの検証」参照）。分離計測が必要になった場合の後続候補として
  ここに記す
- **GEMM TFLOPS 計測系の残差（#391 のスコープ外）**: `cuda_floor_bench` example・
  `#[ignore]` テストバイナリ上の GEMM TFLOPS 計測で #440・#444 から引き継がれていた
  以下 3 件は、本イシュー（プロセス起動コスト計測）とは計測経路・プロトコルが異なり、
  本 PR では解決しない: (1) tiled f32 @4096 の直列再実行値（1.187〜1.237 TFLOPS）と
  `cuda_floor_bench` 逐次計測値（約 1.977 TFLOPS）の約 1.6〜1.7 倍の残差、
  (2) `wmma_f16` 経路 2048 形状の run 間ばらつき（6.27〜8.70 TFLOPS）、
  (3) `tensor_core_real_device.rs::tensor_core_tflops_record` のフレーキー性。
  本イシューの計測（`startup.rs::run_cold`／`run_warm` が probe を厳密に逐次 spawn
  し、計測時 GPU utilization 0%・常駐 2 プロセスのみだった）から言える範囲は
  「起動コスト計測自体は並列競合アーティファクトから自由」ということのみであり、
  上記 3 件の残差を解決するものではない
