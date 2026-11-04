# 二階微分（create_graph）の C 区分 8 族の Metal 実機 parity（#3858）

対象は `crates/facade/tests/create_graph_gpu_parity.rs` の `#[ignore]` 付き Metal テスト 8 本（#3789〜#3796 の族ごとに 1 本）と、
経路の自己検査 1 本（scaffold。測定対象外）。機体は `<metal-node>`（Mac mini・Apple M4・メモリ 16GB。`docs/real-hardware-verification-env.md` §7）。
GB10 版は #3856（`../create-graph-c-cuda-parity-3856/`）。雛形は `../create-graph-gpu-parity-template/RULE.txt`（#3776・#3777）、
測り方は `docs/autodiff-higher-order-grad-decision.md` §48。

- 事前登録: [`RULE.txt`](./RULE.txt)（宣言 2026-11-04T13:57Z。実機実行より前にコミット 0c170cee2 で push した。実行後に変えていない）
- 実行スクリプト: [`run_metal_parity.sh`](./run_metal_parity.sh)（Mac mini 上で手動実行。CI からは呼ばない。`--self-test` 付き）。
  GB10 版 `../create-graph-c-cuda-parity-3856/run_cuda_parity.sh` の複製で、Darwin 限定・`<metal-node>` マスク・Mac の環境情報・
  外部 timeout なし（macOS に `timeout` が無い。Metal 前例 #3166・#3169 と同じ）・各テスト前の並走ビルド検査に変えた
- tolerance 定数（`RELATIVE_TOLERANCE`／`ABSOLUTE_RESCUE_THRESHOLD`）・parity baseline（空のまま）・判定規則・テストコードは不変

## 状態

**実測済み: 測定対象 8 本中 8 本 pass、fail 0、undetermined 0**（2026-11-04・`<metal-node>`）。自己検査（scaffold）も pass。

- 実行コミット（`rev-stamp.txt`）: `0c170cee25c00bd5d52cb2b7848378a3691f5024`・`dirty=0`。事前登録と実行スクリプトを含むコミットそのもので、
  `crates/`・`Cargo.toml`・`Cargo.lock`・`rust-toolchain.toml` は対象コミット `b82b1e5d1` と同一（送信前に `git diff --quiet` で確認）。
  転送は rsync だけで、ノード側で `.rev-stamp` を読んで送信側の HEAD と照合した。
- スクリプト対応: `env_info.txt` の `script_sha256`（`ba2da5ce…1ca5`）は、上記コミットの `run_metal_parity.sh` の SHA-256 と一致する。
- 実機ロック: `env_info.txt` の実行前・実行後とも `hw_lock: held owner=#3858 2026-11-04T13:58:26Z`。終了後にロックが外れていることを確かめた。
  ほかの利用者のプロセスは止めていない。並走する cargo／rustc は事前ビルド前・各テスト前とも無し。メモリの空き率は実行前後とも 72%。
- 回収物は手で編集していない。`SHA256SUMS` はこのディレクトリで `shasum -a 256 -c SHA256SUMS` が通る（22 ファイルすべて OK）。
  ホスト名・`/Users/`・ユーザー名の残存は 0 件（`<metal-node>`／`<user>`／`<home>` へマスク済み。送信側でも grep で確認）。
- デバイス経路: 各テストは `fandhe_ai::tape_for(Device::Metal)` を `expect` で受ける（`create_graph_gpu_parity.rs` の `tapes_for`）。
  Metal が使えなければ `Err` で panic するため、CPU への無言のフォールバックで pass することは無い。route H のケースの pass は
  経路の parity であり、カーネル同士の parity ではない（ホスト関数で計算する Op を含む）。
- S3（split-K 非発火）: 8 族とも `NoMatMul` または `Rule`（RNN・GRU・LSTM。k は最大 20 で、`crates/backend-metal/src/tile.rs:2939-2942` の
  `k/16 < min_k_tiles16 = 8` により split-K は発火しない）。`Listed`・`Judge::Baseline` の族は無く、baseline の Metal 行は要らなかった。
  形式検査（`check_split_k_not_fired`）は各テストの中で通っている。
- 各テストは 0.15〜0.66 秒で終わった（`test result` 行の `finished in`）。

## 結果表

判定契約は全ケース `Judge::Strict`（Q1 同デバイス bit 一致＋GPU・CPU の REQ-2 複合判定／Q2 cg.grad・Q3 H v〈ゼロでない〉・
Q4 孫テープ再生の REQ-2 統一複合判定／Q5 拒否は `Err(Backward)` かつ子テープが空）。

| # | 実装 issue | テスト（`--exact`） | ケース（route） | S3 | 結果 |
|---|---|---|---|---|---|
| 0 | #3777（自己検査） | `create_graph_scaffold_metal_matches_cpu` | scaffold_cubic（K） | NoMatMul | pass |
| 1 | #3789 | `create_graph_select_reduce_minmax_metal_matches_cpu` | max・min・amax・amin（K） | NoMatMul | pass |
| 2 | #3790 | `create_graph_select_reduce_order_metal_matches_cpu` | kthvalue・median_all・quantile_linear（H） | NoMatMul | pass |
| 3 | #3791 | `create_graph_stat_varnorm_metal_matches_cpu` | var・std・norm_l1・norm_l2・norm_p（H） | NoMatMul | pass |
| 4 | #3792 | `create_graph_stat_reduce_lse_nan_metal_matches_cpu` | logsumexp・nansum・nanmean（H） | NoMatMul | pass |
| 5 | #3793 | `create_graph_structural_metal_matches_cpu` | interpolate nearest・bilinear（K）、bicubic・area（H）、max_unpool・dropout（K） | NoMatMul | pass |
| 6 | #3794 | `create_graph_norm_reparam_metal_matches_cpu` | lrn・weight_norm・spectral_norm（H） | NoMatMul | pass |
| 7 | #3795 | `create_graph_rnn_gru_cell_metal_matches_cpu` | rnn・gru（K） | Rule | pass |
| 8 | #3796 | `create_graph_lstm_cell_metal_matches_cpu` | lstm_cell_h・lstm_cell_c（K） | Rule | pass |

各行の根拠は `summary.tsv`（exit code・`running 1 test` の件数・`test result` 行）と `logs/<群>/<テスト名>.log`（末尾に `rc=0`）。
対象の実装 issue #3789〜#3796 は既に CLOSED。CUDA（GB10）は #3856 が測る。

## 回収物

| ファイル | 内容 |
|---|---|
| `rev-stamp.txt` | 展開ツリーの `.rev-stamp`（2 行形式） |
| `env_info.txt` | 実行前後の環境（macOS・CPU・メモリ量と空き率・rustc・負荷・ロック保持） |
| `build.log` | 事前ビルド（`--no-run`）のログ |
| `summary.tsv` | 9 行の判定表 |
| `logs/<群>/` | 各テストの全出力（`--nocapture`）と終了コード |
| `SHA256SUMS` | 上記の SHA-256 |
