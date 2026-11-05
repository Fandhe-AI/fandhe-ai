<!--
役割: 実機計測で共有 CARGO_TARGET_DIR（$HOME/work/target-fandhe-ai）の workspace メンバー成果物が前の回から再利用される問題（イシュー #4044）の決定記録。単一の正。
呼び出し文脈: 手順は docs/real-hardware-verification-env.md §4.11 が要約して参照し、run スクリプトの雛形は docs/perf/logs/hw-parity-run-template/、
  再現の根拠ログは docs/perf/logs/shared-target-reuse-repro-4044/、Phase 10 の各計測 README の「#4044 追記」節が §4 を参照する。
範囲: ドキュメントと手動実行専用スクリプトのみ。tolerance 定数・baseline・閾値・過去の回収ログ・過去の RULE は変えない。実機での取り直しはしない（所有者の判断）。
-->

# 実機計測で共有 target の成果物が再利用される問題（#4044）

2026-11-05。#4031 の測り直し（`docs/perf/logs/metal-batch-norm-2e20-4031-rerun/`）で、テストバイナリがリンクした workspace メンバーの rlib（検証対象の Metal カーネルを含む `libfandhe_ai_backend_metal`・`libfandhe_ai_tensor_core`）が前の回のビルド成果物の再利用だったことが分かった。2 回目（rerun2）は全メンバーの `cargo clean -p` と成果物の更新時刻の検査を足して解消した。本書は、(1) 再利用が起きる条件を開発機で再現して記録し、(2) その対策を計測手順と run スクリプトの雛形の標準にし、(3) Phase 10 の過去の 16 計測への影響を棚卸しする。

## 1. 背景

- 実機ノードは、ビルドキャッシュを同期ツリー外の共有 `CARGO_TARGET_DIR` に置く（`docs/real-hardware-verification-env.md` §3 注意）。転送は `rsync -a` で、送信側のソースの mtime を保つ
- cargo の workspace メンバー（`path` 依存）の freshness 判定は、ソースの mtime と、前回ビルドの dep-info の mtime の比較である
- 展開先のディレクトリ名を変えても、成果物のファイル名のハッシュが変わらない例があった（rerun と rerun2。rerun2 README「成果物の確認」）

## 2. 再現（AC1）

根拠ログは `docs/perf/logs/shared-target-reuse-repro-4044/`（`repro.sh` を開発機 Linux・cargo 1.98.1・`--offline` で実行。`summary.tsv`・`rlibs.tsv`・`build_*.log`・`SHA256SUMS`）。実機は使っていない。

### 2.1 条件

- 比較する 2 コミット: X = `b1f3dd069`、Y = `679e86372`（`tensor-core` の `Device` に `Rocm(usize)` を足したコミット。`crates/tensor-core/src/device.rs` に差分がある）
- 対象: `cargo build -v -p fandhe-ai-tensor-core -p fandhe-ai-backend-cpu`（dev プロファイル）。target は専用の一時ディレクトリ
- X のツリーのソース mtime は 2026-01-02、Y のツリーは 2026-01-01（どちらもビルド時刻より過去）。実運用の「後で作った worktree の方が先にノードでビルドされた」状況を模す

### 2.2 結果

| シナリオ | 操作 | tensor-core | backend-cpu | tensor-core の rlib |
|---|---|---|---|---|
| S1 x | 展開先 D へ X を `rsync -a` してビルド | Compiling | Compiling | `46a806826c516d36`（先頭 16 桁） |
| **S1 y** | 同じ D へ Y を `rsync -a --delete` してビルド | **Fresh** | Compiling | **`46a806826c516d36`（X のまま）** |
| **S2** | 別の展開先 D2 へ Y を `rsync -a` してビルド | **Fresh** | **Fresh** | **同じ（X のまま。ファイル名のハッシュも同じ `1037abea4461a376`）** |
| S3 | mtime を保たない（`--no-times`）で X、続けて Y を送る | Compiling | Compiling | `0cffce6cca73b9a3`（Y の内容） |
| S4 | 再現状態から全 13 メンバーを `cargo clean -p` してビルド | Compiling | Compiling | `0cffce6cca73b9a3`（Y の内容） |
| S5 | Y のツリーの mtime を現在時刻に更新してから送る | Compiling | Compiling | `0cffce6cca73b9a3`（Y の内容） |

- S1 y は、**内容が違うのに Fresh と判定され、X の rlib がそのまま残る**（SHA-256 が X と同じ）。再利用は再現した
- S2: 展開先を変えても Fresh のまま、rlib のファイル名のハッシュも変わらない。メンバーの成果物の同一性はワークスペース内の相対パスで決まり、展開先の絶対パスに依らない。**展開先を変えるだけでは再利用は防げない**
- S1 y で backend-cpu は Compiling だった（fingerprint ログは `UnitDependencyInfoChanged` を示す。理由は本書では追っていない）。つまり、一部のメンバーだけが再コンパイルされ、古い tensor-core にリンクされる状態が起こりうる
- S3・S5 は、mtime が新しければ（git checkout が変更ファイルの mtime を更新する通常の流れ）再コンパイルされることを示す
- S4: `cargo clean -p` は dev プロファイルで `target/debug/deps` の lib と `.fingerprint` を消し（`clean_s4.log`: tensor-core で 9 ファイル、backend-cpu で 24 ファイル）、tensor-core の成果物は残らなかった（`after_clean_leftovers.log` が空）。ビルド後の rlib は Y の内容になった

### 2.3 仕組み（観察と整合する範囲）

1. cargo はメンバーのソースの mtime が前回ビルドの dep-info より古ければ Fresh とする。内容のハッシュは見ない（stable cargo では mtime 比較のみ。checksum による判定は nightly の機能で、stable の実機では使えないため対策に採らない）
2. `rsync -a` は mtime を保つので、送信側のソースが既存の成果物より古ければ Fresh になる
3. メンバーの成果物のファイル名ハッシュは展開先の絶対パスに依らないので、別の展開先でも同じ成果物を使う

Phase 10 の観察との整合:

- (b) CUDA の #3857・#3864 は何も再コンパイルしなかった（build.log が `Finished … in 0.05s`／`0.06s`）。S1 y と整合する（ただし、その時点のソース mtime と dep-info の関係はログに無く、確かめていない）
- (a) Metal の Phase 10 の大半の回は全メンバーを再コンパイルした。各回が新しく作った worktree から送られ、ソースの mtime が新しかったためと**推定**するが、送信側の mtime の記録が無く**未検証**

## 3. 対策（AC2）

標準手順は `docs/real-hardware-verification-env.md` §4.11。要点:

1. ビルドの前に、`cargo metadata --no-deps` の全メンバーへ `cargo clean [--release] -p <名前>`（プロファイルは計測と同じ）
2. ビルド後、テストバイナリと全メンバーの lib 成果物（`lib<lib 名>-*.rlib`・`.rmeta`）の更新時刻が、ロック取得（ノード上の owner ファイルの mtime）より秒単位で厳密に後であることを確かめ、`build_artifacts.txt` に保存する。外れれば undetermined
3. 実行したバイナリが確かめたバイナリと同じことを確かめる

雛形は `docs/perf/logs/hw-parity-run-template/run_parity_template.sh`（R1〜R5・複数テスト・プロファイル切り替え・`--self-test`）。rerun2 の R5 の 2 つの欠陥を雛形で直した。

- 手書き glob `libfandhe_ai_*` が facade 自身の lib（`libfandhe_ai-<hash>.rlib`）に一致しなかった。Phase 10 の facade テストはこれをリンクしている。雛形は `cargo metadata` の lib ターゲット名から作る
- release 専用だった。雛形は `PROFILE`（dev／release）で `debug/deps`・`release/deps` と clean の引数を切り替える

計測済みの過去の run スクリプト（rerun2 を含む）は回収物なので直さない。事前登録の雛形 `RULE.txt`（`create-graph-gpu-parity-template`・`train-gpu-parity-template`）には追加キー `build_freshness:` を足した。これらは実測に使っていない雛形で、既存行は変えていない。計測済み回の RULE は一切変えていない。

mtime を転送で更新する案（`rsync --no-times` 等）は、送信側と実機で挙動が変わりうるうえ、検査にならないため採らず、clean と時刻検査を標準とした。

## 4. Phase 10 の過去の計測の棚卸し（AC3）

対象は `docs/perf/logs/` の 16 ディレクトリ（#3856〜#3866）。すべて dev（`test`）プロファイル。観察は各ディレクトリの `rev-stamp.txt`・`env_info.txt`・`build.log`。

### 4.1 判定基準

| 分類 | 意味 |
|---|---|
| A: 再コンパイルを確認 | `build.log` に、テストがリンクする全メンバー（その機体のビルドの対象）の `Compiling` がある |
| B: 再利用あり、ただし前任とライブラリのコードが同一 | 一部のメンバーが `Compiling` でなく再利用された。ただし、同種の機体の記録上の前任の計測の事前登録コミットとの間で、`crates/`（`tests/`・`examples/`・`benches/` を除く）・`Cargo.toml`・`Cargo.lock`・`rust-toolchain.toml` の差分が 0 件。再利用された成果物が、測ったコードと同じコードから作られたと判断できる |
| C: 再利用の可能性を排除できない | 上の根拠が成り立たない（再利用された成果物の由来を記録から確かめられない） |

- 差分は隣り合う事前登録コミットの組すべてで取った（`git diff --name-only <前任> <今回> -- crates Cargo.toml Cargo.lock rust-toolchain.toml`）。cuda 7 組・metal 7 組でライブラリのコードの差分は 0 件。`tests/` の差分は #3860・#3861 の train テンプレート回（`crates/facade/tests/train_gpu_parity.rs`）の 1 件ずつだけで、どちらも全メンバーが再コンパイルされた回（A）である
- 機体の同一性は `env_info.txt` がマスク済みで特定できない場合がある（`<cuda-node>` は GB10 の総称で機体は 1 台に固定されない）。差分が 0 件なのは同種の記録上の全回の間なので、どの機体が前任でも結論は変わらない
- **`Compiling` 行の限界**: `Compiling <pkg>` はパッケージ単位で、テストターゲットだけのコンパイルでも出る。`Fresh` は `-v` なしでは出ない。判定の主根拠は git 差分で、`Compiling` 行は補強である
- **B の限界**: 判断は「記録上の前任」に依る。計測の間に記録に無い他セッションのビルドが同じ target にあった場合（#4031 で展開ツリーが別セッションに上書きされた事例がある）、再利用された成果物が記録上の前任のものとは限らない

### 4.2 表

| ディレクトリ | 機体 | 事前登録コミット | 実行（UTC・11-04） | build.log の `Compiling`（メンバー） | 前任との lib 差分 | 分類 |
|---|---|---|---|---|---|---|
| create-graph-c-cuda-parity-3856 | cuda | 6852ef6bd | 12:46 | 8 メンバー全部 | （最初の回） | A |
| create-graph-b-cuda-parity-3857 | cuda | 48d0263ac | 12:51 | 0 件（`Finished … 0.05s`）。テストバイナリ `create_graph_gpu_parity-43f86a2c2940f8f0` が 3856 と同一 | 0 件（tests も 0 件） | B |
| nn-layers-cuda-parity-3860 | cuda | 77c2cbd58 | 12:56 | `fandhe-ai` のみ | 0 件 | B |
| param-groups-resident-cuda-parity-3862 | cuda | 6ba9db57f | 13:01 | tensor-core・backend-cuda・backend-cpu・autodiff・bench-harness と crates.io の依存（libc 等）、続けて `fandhe-ai`。`backend-rocm`・`onnx-interop` は無し | 0 件 | B（依存まで再コンパイルしており、target が別だった可能性。`backend-rocm`・`onnx-interop` は `Compiling` が無く再利用された可能性） |
| onnx-train-cuda-parity-3864 | cuda | a05e3caeb | 13:04 | 0 件（`Finished … 0.06s`）。テストバイナリ `train_gpu_parity-7f8cdf980f4362cb` | 0 件 | **C**（このテストバイナリは 3862 のビルドにも既に存在した。作った回が記録に無く、そのときのテストのコードを確かめられない。**#4046 が 2026-11-05 に測り直し、pass**。記録は `docs/perf/logs/onnx-train-cuda-parity-3864-rerun/`。テストバイナリと全メンバーの lib 成果物がロック取得より後に書かれたことを確かめた） |
| gpu-linalg-optin-cuda-parity-3866 | cuda | 49a470f4b | 15:03 | `fandhe-ai` のみ（新しいテストターゲット） | 0 件 | B（前任との間隔が約 2 時間で、記録に無いビルドがありうる） |
| nn-train-template-cuda-parity-3860 | cuda | bf118b365 | 19:05 | 8 メンバー全部 | lib 0 件・tests 1 件 | A |
| optim-train-template-cuda-parity-3862 | cuda | f81eedd9c | 19:09 | 8 メンバー全部 | 0 件 | A |
| create-graph-c-metal-parity-3858 | metal | 0c170cee2 | 13:58 | 9 メンバー全部 | （最初の回） | A |
| create-graph-b-metal-parity-3859 | metal | bbfcaee6b | 14:04 | 9 メンバー全部 | 0 件 | A |
| nn-layers-metal-parity-3861 | metal | 6c29359cc | 14:11 | 9 メンバー全部と `fandhe-ai` 15 回 | 0 件 | A |
| param-groups-resident-metal-parity-3863 | metal | 39cb7d101 | 14:16 | 9 メンバー全部（大半が 2 回） | 0 件 | A |
| onnx-train-metal-parity-3865 | metal | 55fa6e3e3 | 14:19 | 9 メンバー全部 | 0 件 | A |
| gpu-linalg-optin-metal-parity-3866 | metal | 49a470f4b | 15:03 | backend-cpu・backend-rocm・bench-harness・`fandhe-ai` のみ。backend-metal・tensor-core・autodiff 等は再利用 | 0 件 | B |
| nn-train-template-metal-parity-3861 | metal | c71ef53a2 | 19:05 | 9 メンバー全部 | lib 0 件・tests 1 件 | A |
| optim-train-template-metal-parity-3863 | metal | 31656bccc | 19:09 | 9 メンバー全部 | 0 件 | A |

集計: A 10 件・B 5 件・C 1 件（元の 16 計測の分類。C の 1 件は #4046 の測り直しで pass。分類そのものは変えていない）。

### 4.3 読み方

- ライブラリのコードは、記録上のどの隣り合う組でも同一だった。したがって B の回は、記録上の前任が実際の前任なら、測ったコードと再利用された成果物のコードが同じで、結果への影響は無い。この判断は上の「B の限界」に依る
- C の 1 件（#3864 の CUDA、ONNX の学習）は、テストバイナリの由来を記録から確かめられない。ライブラリのコードは同一なので、影響が出るとすればテストのコードが古い場合に限られる。**同じ族（`train_gpu_parity`）の 19:05・19:09 の回は全メンバーを再コンパイルしており、そこでは nn・optim の族を測ったが、onnx の族は測っていない**。所有者の判断で #4046 が取り直し、pass だった（`docs/perf/logs/onnx-train-cuda-parity-3864-rerun/`）
- #4031 の rerun（無効）と rerun2（有効）は本書の対象外（release プロファイル。rerun2 が正）

### 4.4 各 README への反映

16 ディレクトリの `README.md` の末尾に「#4044 追記」節を足した（分類と本節へのリンクのみ。既存の本文と結果は変えていない）。`README.md` はどのディレクトリでも `SHA256SUMS` に載っていないことを確かめてから追記した。追跡表 `docs/compat-matrix-ari-tracker.md` の §1 注記に 1 行足した（「実機 parity」列の値は変えていない）。

## 5. 範囲外・残件

- C（#3864 の CUDA）の実機での取り直しは #4046 で実施済み（pass。§4.2）
- 計測済みの過去の run スクリプト（rerun2 を含む）の `libfandhe_ai_*` glob の漏れは、回収物なので直していない
- 雛形の macOS（bash 3.2）での `--self-test`・実機での実行は未検証。次回の `<metal-node>` 計測時に確かめる
- 作業ノードの片付け（`rust-ai-library-run-4031-rerun2/` 等）は本書の範囲外で、所有者の指示待ち
- tolerance 定数・baseline・ガードレール閾値は変えていない
