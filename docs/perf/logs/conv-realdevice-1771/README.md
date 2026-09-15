# イシュー #1771 実機ランブック（Conv1d／Conv2d の CUDA／Metal parity・実測）

## 位置づけ

本ディレクトリは Conv1d／Conv2d 関連 4 イシュー（#1766「CUDA Conv2d
im2col／col2im」・#1767「CUDA Conv1d」・#1768「Metal Conv2d
im2col／col2im」・#1769「Metal Conv1d」）に分散していた実行手順・
事前登録判定規則を統合した**正式な実測記録の受け皿**である
（`docs/perf/logs/{cuda-conv2d-1766, cuda-conv1d-1767,
metal-conv2d-1768, metal-conv1d-1769}/` の各 README は本ディレクトリ
への forward pointer のみを追記済み）。

本 PR の実行環境（Linux コンテナ／worktree）には DGX Spark GB10・
Apple Silicon いずれの実機への到達手段（`docs/real-hardware-
verification-env.local.md`・`CUDA_NODE` 環境変数・`~/.ssh/config` の
いずれも確認できず）もないため、**実測値は一切含まれていない**。
実測は GB10（CUDA）・Mac（Metal）それぞれの実機を持つセッションへ
申し送る。

**PR #1882 レビュー指摘の是正**: `im2col_col2im_parity`（CUDA／
Metal）・`conv2d_backend_parity`・`conv1d_backend_parity`、および
既存の nn 層 forward テスト（`{cuda,metal}_sequential_conv2d_
matches_cpu`）に `bits=`／`fold_bits=` 出力が無く、事前登録判定規則
4) の run-to-run 決定性チェック（`grep` 抽出 → `diff`）が空の抽出
結果同士の自明一致で無検出になり得た指摘を受け、該当テストすべてに
`print_fold_bits`（FNV-1a fold チェックサム）出力を追加した。あわせ
て「事前登録判定規則」4) の手順を、抽出行数の非空・run1/run2 間の
一致を診断のうえ diff する `check_determinism.sh` へ差し替えた。

## 対象テスト一覧

### 1) 既存 4 イシューの `#[ignore]` テスト（変更なし・非後退確認用）

| バックエンド | コマンド |
|---|---|
| CUDA | `cargo test -p fandhe-ai-backend-cuda --release --test im2col_col2im_parity -- --ignored --nocapture --test-threads=1` |
| CUDA | `cargo test -p fandhe-ai --release --test conv2d_backend_parity -- --ignored --nocapture --test-threads=1` |
| CUDA | `cargo test -p fandhe-ai --release --test conv1d_backend_parity -- --ignored --nocapture --test-threads=1` |
| Metal | `cargo test -p fandhe-ai-backend-metal --release --test im2col_col2im_parity -- --ignored --nocapture --test-threads=1` |
| Metal | `cargo test -p fandhe-ai --release --test conv2d_backend_parity -- --ignored --nocapture --test-threads=1 metal_` |
| Metal | `cargo test -p fandhe-ai --release --test conv1d_backend_parity -- --ignored --nocapture --test-threads=1 metal_` |

`conv2d_backend_parity.rs`／`conv1d_backend_parity.rs` は `cuda_*`／
`metal_*` の両テストを同一バイナリに含み、`cuda_*` 側は
`cfg(target_os = "macos")` で除外されない（実機必須の `.expect()` で
panic する設計のため）。Metal 実機でフィルタなしに `--ignored` を
実行すると CUDA 実機必須テストまで拾って ANY_FAILED が水増しされる
ため、Metal 側コマンドには `metal_` 部分一致フィルタを付ける
（`run_ignored_tests_metal.sh` も同様。PR #1882 レビュー指摘）。
`--test-threads=1` は事前登録判定規則 4)（run-to-run 決定性）のため
共通で付与する（並列実行だと `bits=`／`fold_bits=` 行の出力順が
起動ごとに変わりうる）。

### 2) イシュー #1771 で新設した nn 層（`compat::Sequential`）`#[ignore]` テスト

`crates/facade/tests/nn_conv_backend_parity.rs`（CUDA・Metal 各 6 件）:

- `{cuda,metal}_sequential_conv2d_matches_cpu`（既存。forward のみ）
- `{cuda,metal}_sequential_conv2d_backward_matches_cpu`（新設。
  weight／bias／入力勾配）
- `{cuda,metal}_sequential_conv1d_forward_matches_cpu`（新設）
- `{cuda,metal}_sequential_conv1d_backward_matches_cpu`（新設）
- `{cuda,metal}_sequential_conv1d_matches_manual_reshape_conv2d_bit_exact`
  （新設。「特化」契約の nn 層版）
- `{cuda,metal}_sequential_conv2d_sgd_steps_record_only`（新設。
  **record-only**。5 step SGD の loss・最終パラメータ比較）

実行コマンド（`--test-threads=1` は事前登録判定規則 4) のため共通で
付与し、Metal は同一バイナリに含まれる `cuda_*` テストを除外するため
`metal_` 部分一致フィルタを付ける。PR #1882 レビュー指摘）:

```bash
# CUDA
cargo test -p fandhe-ai --release --test nn_conv_backend_parity -- --ignored --nocapture --test-threads=1

# Metal
cargo test -p fandhe-ai --release --test nn_conv_backend_parity -- --ignored --nocapture --test-threads=1 metal_
```

## 実行手順

### GB10（CUDA）

`docs/real-hardware-verification-env.md` §3（rsync 転送。`--filter=':-
.gitignore'` に加え `--exclude 'real-hardware-verification-env.local.md'
--exclude '.env*' --exclude '.claude/settings.local.json'` を必ず付ける）
→ `.rev-stamp` でリビジョン確認 → §6.1 で他プロセスの GPU 占有がない
ことを確認 → `run_ignored_tests_cuda.sh` を実行する。

### Mac（Metal）

リポジトリルートで `run_ignored_tests_metal.sh` を直接実行する
（`uname -sm` が `Darwin arm64` であることを起動時に検証する）。

## 保存すべきログ

- `cuda/ignored/*.log`（`run_ignored_tests_cuda.sh` の出力。事前登録
  判定規則 4) の run-to-run 決定性確認のため 2 回実行し、それぞれ
  `cuda/ignored-run1`／`cuda/ignored-run2` 等の別ディレクトリへ保存
  する）
- `metal/ignored/*.log`（`run_ignored_tests_metal.sh` の出力。同上
  2 回分を保存する）
- `check_determinism.sh` の出力（上記 2 回分の run1/run2 ディレクト
  リ対を `cuda`／`metal` それぞれで比較した結果。PASS／FAIL 行）
- `env_info.txt`（本ディレクトリのテンプレートに実測値を記入。
  **内部ホスト名は書かない**。`hostname: masked` のまま残す）

## 事前登録判定規則

### 1) im2col／col2im（bit 完全一致契約。`docs/conv-ops-design.md` §7）

- **im2col**: CPU 参照実装（`backend-cpu::im2col::im2col`）と byte
  単位完全一致（算術を含まない純粋コピー演算）
- **col2im**: CPU 参照実装（`f64` 逐次和・1 回 `f32` downcast）と
  byte 単位完全一致（CUDA は `double` ネイティブ・Metal は binary64
  ソフトウェアエミュレーション。NaN のみクラス一致）

### 2) conv forward／backward（GEMM 段を含む全体。REQ-2 統一複合判定）

- CPU 参照実装（`fandhe_ai::tape()`）との複合判定（相対誤差 1e-3
  未満 または 絶対誤差 1e-5 未満）が全 fail 0 件であること
- nn 層（`compat::Sequential`）経由でも同一契約が成立する（`Var::
  conv2d`／`conv1d` の薄いラッパーのため GPU 差分の発生源は GEMM 段
  のみ）

### 3) conv1d ↔ 手動 reshape conv2d（「特化」契約）

- 同一 GPU tape 上で forward・d_input・d_weight・d_bias が**bit 完全
  一致**すること（`conv1d` は reshape 併合のみで新規カーネルを持た
  ないため機構的に成立する契約）

### 4) run-to-run 決定性

- 同一入力で 2 回起動しても `bits=`／`fold_bits=` 行が完全一致する
  こと。**`check_determinism.sh --expect-logs <name1,...> <run1_dir> <run2_dir>` を使う**
  （素の `grep -E 'bits=|fold_bits=' | diff` だけでは、対象テストの
  出力に抽出対象行が 1 件も無い場合に空の抽出結果同士が自明に一致
  してしまい起動間の変化を検出できない〈PR #1882 レビュー指摘〉。
  本スクリプトは diff の前に (a) 各ログの抽出行数が 0 でないこと・
  (b) 2 回の起動間で抽出行数が一致すること を検査してから diff す
  る。`--self-test` で GPU 実機なしにロジック自体を検証できる）。
  `run_ignored_tests_{cuda,metal}.sh` を出力先を変えて 2 回実行し
  （1 回目の `cuda/ignored`／`metal/ignored` を `-run1` へ退避してか
  ら 2 回目を実行し `-run2` へ退避する等）、対応するログディレクトリ
  同士を渡す。`--expect-logs` には本 README「対象テスト一覧」1)〜2)
  の 4〈CUDA〉／4〈Metal〉グループのログ名（`run_ignored_tests_
  {cuda,metal}.sh` の `run_case` 呼び出し名と同一）を渡す:
  `im2col_col2im_parity,conv2d_backend_parity,conv1d_backend_parity,nn_conv_backend_parity`。
  これにより次の 2 点が機械的に保証される（**PR #1882 追加レビュー
  指摘の是正**）:
  - 走査は run1／run2 の `*.log` の**和集合**に対して行う（旧実装は
    run1 側の `*.log` のみを走査しており、run2 側にのみ存在する
    ログ〈run1 側の取りこぼし等〉を一度も検査せず見逃していた）
  - `--expect-logs` で列挙した 4 グループのログ名それぞれが run1・
    run2 の**両方**に存在し抽出行が 1 件以上あることを検査する
    （旧実装は「run1／run2 に実在するログ」しか把握しておらず、ある
    期待ログが run1・run2 の**両方**から丸ごと欠落した場合〈ビルド
    失敗・実行忘れ・出力先パスの取り違え等〉は走査対象にすら現れず
    検出できなかった）

  **既知の残存限界**: 既存ログ自体は存在し run1／run2 間の抽出行数
  も一致するが、本来あるべき `print_fold_bits` 呼び出し 1 件が
  run1・run2 の**両方で同一に**欠落しているケース（ログファイル単位
  ではなくログ内の特定ケース単位の印字漏れが両起動で再現する場合）
  は、run1 と run2 の相互比較だけでは原理的に検出できない（比較対象
  となる「期待される行数・ラベル集合」の run1/run2 から独立した正が
  このランブックには無いため）。対象は本番コードではなく診断専用の
  ログ・スクリプトであることを踏まえ、ログファイル単位の欠落検出
  （上記 2 点）に留める。

### 5) 学習ループ（record-only。`{cuda,metal}_sequential_conv2d_sgd_steps_record_only`）

- **ADOPT／REJECT の判定対象にしない**。GEMM 由来の差が step をまた
  いで累積しうるため。各 step の loss（`bits=` 行）・最終パラメータの
  REQ-2 複合判定結果・`fold_bits` を記録するのみ

### 総合判定

上記 1)〜4) がすべて成立すれば `verdict=PASS`、いずれか不成立なら
`verdict=FAIL`（該当箇所を明記。**tolerance／`BASELINES` は変更せず
エスカレーションのみ**。`docs/conv-ops-design.md` §7 の baseline 方式
適用可否はユーザー承認が必要）。実機到達不能のまま出荷する場合は
`verdict=undetermined` とし、申し送り先（親 #1645）を明記する。
