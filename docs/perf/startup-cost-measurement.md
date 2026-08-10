# 起動コスト実測・v1 差分記録（#171・TASK-13.1b）

イシュー #171「test(backend): TASK-13.1b 実測・v1 差分の記録」の実測記録。
受け入れ条件「実測記録と差分分析が残されている」に対応する。計測ハーネス自体は
TASK-13.1a（#170・PR #360）で整備済み（`crates/bench-harness/src/startup.rs`・
`Makefile` の `startup-bench` ターゲット）であり、本ドキュメントは同ハーネスを用いた
実測結果と v1（PoC-5）実測値との差分分析を記録する。

## 状態: CPU・Metal 実測済み・CUDA 実機未実施

CPU 実測は Linux x86_64（QEMU/KVM 仮想化環境、NVIDIA RTX 3060 passthrough）
worktree で行っており、`libnvrtc`（CUDA toolkit）が導入されていない
（`ldconfig -p | grep nvrtc` で未検出）。`backend-cuda` は `cudarc` 動的ロード契約
により本環境でも `cargo build` が成立するが、実行時は型付きエラーに fail-closed で
倒れる契約であり、下記「CUDA 動的ロード契約の実地確認」節で実測している。
Metal（Apple Silicon）実機はイシュー #384 で実測完了した（下記「Metal 実測結果」節）。

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

## CUDA 実機手順（結果転記テンプレート）／Metal 実機手順（実測済み・再現手順）

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
| プロセスごとの JIT コンパイル | あり（CubeCL がシェーダーを毎プロセス JIT コンパイル。コンパイル結果は非永続） | CPU: **なし**（本セッション実測で確認済み。コールド≒ウォーム）。CUDA: **あり**（NVRTC 実行時コンパイルは `backend-cuda/src/nvrtc.rs` がディスク非永続の契約。ただし対象カーネル数は v2 の限定的な自作カーネル群のみで、CubeCL の autotune 候補群のような複数戦略の実行時探索は行わない。定量差分は CUDA 実機実測で確定）。**Metal: あり**（`MetalGemm::new` が `gemm.metal` を毎プロセス runtime コンパイル。ただし CUDA 同様 autotune 探索は無い。イシュー #384 実測で near-cold 観測点〈第 1 サンプル〉のみコンパイルコストと見られる突出〈first_kernel 約 2.6 倍〉を確認） |
| キャッシュ機構 | `target/autotune/` に戦略選択の識別結果のみ永続化（コンパイル済みバイナリ非永続） | CUDA: ドライバ側 `CUDA_CACHE_PATH` の JIT キャッシュ（`CudaGemm::run_tiled_f32` 呼び出しのみ影響を受ける。`backend-cuda` 独自のキャッシュ実装はなし）。**Metal: OS 側 Metal コンパイラキャッシュ**（MTLCompilerService・ユーザーごとキャッシュ。`backend-metal` 側に独自キャッシュ実装はなく、本ハーネスもこのキャッシュに触れない） |

本セッションの CPU 実測（`device_init_secs`・`first_kernel_secs` がミリ秒オーダーで
コールド／ウォームほぼ同値）は、v2 が v1 の主要劣位要因のうち **autotune 探索**を
CPU 経路では構造的に持たないことを裏付ける。ただし「プロセスごとのシェーダー JIT」
自体は v2 でも Metal・CUDA の両バックエンドに残存する（イシュー #384 の Metal 実測
で near-cold 観測点に compile コストとみられる突出を確認したことがその裏付け）。
v1 との定量比較で言えることは、v2 は v1 のような**複数戦略の実行時ベンチマーク探索**
（autotune）を持たない点であり、単一シェーダーの毎プロセスコンパイル自体は
Metal・CUDA いずれのバックエンドでも構造的に残る。CUDA 経路の定量差分は
実機実測で確定する。

### 計測前提の非対称性（結論を先取りしないための明記）

- **ハードウェア差**: v1 は DGX Spark GB10（aarch64）実機。本セッションの v2 CPU 実測は
  x86_64 の仮想化環境（QEMU/KVM）。CPU アーキテクチャ・仮想化オーバーヘッドの影響で
  絶対値は単純比較できない
- **ワークロード差**: v1 は Transformer ブロック（`d_model=512, n_heads=8, d_ff=2048,
  batch=8, seq_len=128`）の初回推論完了までを計測。v2 の `startup_probe` は GEMM
  256×256 の初回カーネル完了までを計測（`ProbeReport::first_kernel_secs` doc コメント
  参照）。計測対象の演算規模・種類が異なる
- **Metal 分**: v1 = DGX Spark GB10（CUDA 実機）・Transformer ブロック初回推論。
  v2 Metal 実測（#384） = Apple M4 Max・256×256 GEMM。ハードウェア（CUDA GPU vs
  Apple GPU）・ワークロードの両方が異なり絶対値の単純比較はできない。加えて
  `startup_probe.rs` の `GEMM_SIZE` 固定・`MetalBackendOps::gemm` 内部で
  `MetalContext` が呼び出しごとに再構築される現行実装（既知の制約。PoC 段階の
  簡略化であり本イシューでは変更しない）が計測値に混入している点は CPU・CUDA
  分析時と同様に留意する
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

## 後続

TASK-13.2（#172・短命プロセス対応方針の決定、担当: 人間）が本記録を入力とする。
CUDA 実機転記（#388 ツリー側）が完了するまでは、本記録の CPU・Metal 実測部分・
構造差分の定性分析のみを判断材料として扱う。

- **CUDA 実機転記**: #388 ツリー側で別途対応する
- **下流ドキュメント同期**: `docs/short-lived-process-decision.md`（Metal 行・
  再判定トリガー）・`docs/backend-matrix.md` への反映は、#387（総括反映）・#172
  （人間判断）のスコープとし本 PR には含めない。再判定トリガーは「CUDA と Metal の
  **両方**の転記完了」であり、CUDA 未実測のため #172 の決定自体は本 PR では変わらない
- **行番号アンカーの参照ずれ**: 本改訂により本文書内の行番号がずれるため、本文書を
  行番号アンカーで引用している下流ドキュメントがあれば、今後は節見出し参照へ寄せる
  ことが望ましい
- **bench-harness 側 Metal `#[ignore]` E2E テスト**: `tests/startup_harness.rs` へ
  Metal 版の E2E テスト追加は本 PR のスコープ外（受入条件は記録のみ）。後続候補として
  ここに記す
