# 二階微分（create_graph）の C 区分 8 族の CUDA 実機 parity（#3856）

対象は `crates/facade/tests/create_graph_gpu_parity.rs` の `#[ignore]` 付き CUDA テスト 8 本（#3789〜#3796 の族ごとに 1 本）と、
経路の自己検査 1 本（scaffold。測定対象外）。機体は `<cuda-node>`（DGX Spark GB10。`docs/real-hardware-verification-env.md`）。
雛形は `../create-graph-gpu-parity-template/RULE.txt`（#3776）、測り方は `docs/autodiff-higher-order-grad-decision.md` §48。

- 事前登録: [`RULE.txt`](./RULE.txt)（宣言 2026-11-04T12:44Z。実機実行より前にコミット 6852ef6bd で push した。実行後に変えていない）
- 実行スクリプト: [`run_cuda_parity.sh`](./run_cuda_parity.sh)（GB10 上で手動実行。CI からは呼ばない。`--self-test` 付き）。
  `../cuda-autodiff-parity-3168/run_cuda_parity.sh` の複製に、2 行形式の `.rev-stamp` 検査・実機ロックの取得と解放・`<home>` マスクを足した
- tolerance 定数（`RELATIVE_TOLERANCE`／`ABSOLUTE_RESCUE_THRESHOLD`）・parity baseline（空のまま）・判定規則・テストコードは不変

## 状態

**実測済み: 測定対象 8 本中 8 本 pass、fail 0、undetermined 0**（2026-11-04・`<cuda-node>`）。自己検査（scaffold）も pass。

- 実行コミット（`rev-stamp.txt`）: `6852ef6bd067a74e3090b4f390f59a5a70f8b4a5`・`dirty=0`。事前登録と実行スクリプトを含むコミットそのもので、
  `crates/`・`Cargo.toml`・`Cargo.lock`・`rust-toolchain.toml` は対象コミット `b82b1e5d1` と同一（送信前に `git diff --quiet` で確認）。
  転送は rsync だけで、ノード側で `.rev-stamp` を読んで送信側の HEAD と照合した。
- スクリプト対応: `env_info.txt` の `script_sha256`（`c4be6b6b…d1c0`）は、上記コミットの `run_cuda_parity.sh` の SHA-256 と一致する。
- 実機ロック: `env_info.txt` の実行前・実行後とも `hw_lock: held owner=#3856 2026-11-04T12:46:48Z`。終了後にロックが外れていることを確かめた。
  ほかのサービスのプロセスは止めていない（常駐 compute プロセスは実行前後とも 3 件。GPU 使用率は前後とも 0%）。並走する cargo／rustc は無し。
- 回収物は手で編集していない。`SHA256SUMS` はこのディレクトリで `shasum -a 256 -c SHA256SUMS`（Linux は `sha256sum -c`）が通る。
  ホスト名・`/home/`・ユーザー名の残存は 0 件（`<cuda-node>`／`<user>`／`<home>` へマスク済み）。
- デバイス経路: 各テストは `fandhe_ai::tape_for(Device::Cuda(0))` を `expect` で受ける（`create_graph_gpu_parity.rs` の `tapes_for`）。
  CUDA が使えなければ `Err` で panic するため、CPU への無言のフォールバックで pass することは無い。route H のケースの pass は
  経路の parity であり、カーネル同士の parity ではない（ホスト関数で計算する Op を含む）。
- 各テストは 0.36〜1.07 秒で終わった（`test result` 行の `finished in`）。

## 結果表

判定契約は全ケース `Judge::Strict`（Q1 同デバイス bit 一致＋GPU・CPU の REQ-2 複合判定／Q2 cg.grad・Q3 H v〈ゼロでない〉・
Q4 孫テープ再生の REQ-2 統一複合判定／Q5 拒否は `Err(Backward)` かつ子テープが空）。

| # | 実装 issue | テスト（`--exact`） | ケース（route） | 結果 |
|---|---|---|---|---|
| 0 | #3776（自己検査） | `create_graph_scaffold_cuda_matches_cpu` | scaffold_cubic（K） | pass |
| 1 | #3789 | `create_graph_select_reduce_minmax_cuda_matches_cpu` | max・min・amax・amin（K） | pass |
| 2 | #3790 | `create_graph_select_reduce_order_cuda_matches_cpu` | kthvalue・median_all・quantile_linear（H） | pass |
| 3 | #3791 | `create_graph_stat_varnorm_cuda_matches_cpu` | var・std・norm_l1・norm_l2・norm_p（H） | pass |
| 4 | #3792 | `create_graph_stat_reduce_lse_nan_cuda_matches_cpu` | logsumexp・nansum・nanmean（H） | pass |
| 5 | #3793 | `create_graph_structural_cuda_matches_cpu` | interpolate nearest・bilinear（K）、bicubic・area（H）、max_unpool・dropout（K） | pass |
| 6 | #3794 | `create_graph_norm_reparam_cuda_matches_cpu` | lrn・weight_norm・spectral_norm（H） | pass |
| 7 | #3795 | `create_graph_rnn_gru_cell_cuda_matches_cpu` | rnn・gru（K） | pass |
| 8 | #3796 | `create_graph_lstm_cell_cuda_matches_cpu` | lstm_cell_h・lstm_cell_c（K） | pass |

各行の根拠は `summary.tsv`（exit code・`running 1 test` の件数・`test result` 行）と `logs/<群>/<テスト名>.log`（末尾に `rc=0`）。
対象の実装 issue #3789〜#3796 は既に CLOSED。Metal（Mac mini）は #3858 が測る。

## 回収物

| ファイル | 内容 |
|---|---|
| `rev-stamp.txt` | 展開ツリーの `.rev-stamp`（2 行形式） |
| `env_info.txt` | 実行前後の環境（GPU・ドライバ・nvcc・rustc・負荷・常駐件数・ロック保持） |
| `build.log` | 事前ビルド（`--no-run`）のログ |
| `summary.tsv` | 9 行の判定表 |
| `logs/<群>/` | 各テストの全出力（`--nocapture`）と終了コード |
| `SHA256SUMS` | 上記の SHA-256 |
