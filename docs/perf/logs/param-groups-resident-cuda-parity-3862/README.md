# param groups と DeviceParamStore の GPU 常駐更新の CUDA 実機 parity（#3862）

対象は param groups と `DeviceParamStore` の GPU 常駐更新（実装 #3837・#3838）。`docs/autodiff-param-groups-decision.md` §19.4 が
Phase 10 で測ると定めた CUDA 側の `#[ignore]` 3 本を、`<cuda-node>`（DGX Spark GB10。`docs/real-hardware-verification-env.md`）で測った。
雛形は `../train-gpu-parity-template/RULE.txt`（#3778）。

- 事前登録: [`RULE.txt`](./RULE.txt)（宣言 2026-11-04T13:00Z。実機実行より前にコミット 6ba9db57f で push した。実行後に変えていない）
- 実行スクリプト: [`run_cuda_parity.sh`](./run_cuda_parity.sh)（#3860 版 `../nn-layers-cuda-parity-3860/run_cuda_parity.sh` の複製で、
  package をターゲット引数に含める変更だけを入れた。GB10 上で手動実行。CI からは呼ばない。`--self-test` 付き）
- tolerance 定数（`RELATIVE_TOLERANCE`／`ABSOLUTE_RESCUE_THRESHOLD`）・parity baseline・判定規則・テストコードは不変

## 状態

**実測済み: 測定対象 3 本中 3 本 pass、fail 0、undetermined 0**（2026-11-04・`<cuda-node>`）。

**要対応（所有者へ報告）: 雛形 `crates/facade/tests/train_gpu_parity.rs` の optim 族は登録 0 件。** `train_parity_optim_cuda_matches_cpu` は
`undetermined: 族 optim の登録が 0 件`（`crates/facade/tests/common/train_gpu_parity.rs:193`）で panic し、exit 101・判定 undetermined だった。
RULE.txt で事前に宣言したとおり完了判定には数えていない。雛形に optim 族の行を足すかどうかは所有者の判断を待つ。

- 数値契約は決定記録 §19.5 の (3)「CUDA + groups → CPU 参照と REQ-2 統一複合判定」。GPU の bit 一致・run-to-run の bit 同一は
  観測値で合否の基準ではない。今回のテスト出力には bit 一致の観測行は無かった（記録していない）。
- 実行コミット（`rev-stamp.txt`）: `6ba9db57f141c69effbc6f9da1d3517d08f1fe58`・`dirty=0`。事前登録と実行スクリプトを含むコミットそのもので、
  `crates/`・`Cargo.toml`・`Cargo.lock`・`rust-toolchain.toml` は対象コミット `b82b1e5d1` と同一（送信前に `git diff --quiet` で確認）。
  転送は rsync だけで、ノード側で `.rev-stamp` とスクリプトの SHA-256 を読んで送信側と照合した。
- スクリプト対応: `env_info.txt` の `script_sha256`（`88d3740b…1634`）は、上記コミットの `run_cuda_parity.sh` の SHA-256 と一致する。
- 実機ロック: `env_info.txt` の実行前・実行後とも `hw_lock: held owner=#3862 2026-11-04T13:01:11Z`。終了後にロックが外れていることを確かめた。
  ほかのサービスのプロセスは止めていない（常駐 compute プロセスは実行前後とも 3 件。GPU 使用率は実行前 0%・実行後 7%）。並走する cargo／rustc は無し。
- 回収物は手で編集していない。`SHA256SUMS` はこのディレクトリで `shasum -a 256 -c SHA256SUMS`（Linux は `sha256sum -c`）が通る。
  ホスト名・`/home/`・ユーザー名の残存は 0 件（`<cuda-node>`／`<user>`／`<home>` へマスク済み）。
- 各テストは 0.24〜1.31 秒で終わった（`test result` 行の `finished in`）。#3837 の `cuda_reports_grouped_capability` は `#[ignore]` ではなく対象外。

## 結果表

| # | 実装 issue | package・`--test` | テスト（`--exact`） | 腕 | judge | 結果 |
|---|---|---|---|---|---|---|
| 1 | #3837 | `fandhe-ai-backend-cuda`・`grouped_optimizer_step_real_device` | `sgd_grouped_matches_cpu_grouped` | CUDA grouped カーネル 対 CPU grouped 参照 | Req2 | pass |
| 2 | #3837 | 同上 | `adam_grouped_matches_cpu_grouped_and_rejects_bad_totals` | 同上（不正な合計の拒否を含む） | Req2 | pass |
| 3 | #3838 | `fandhe-ai`・`device_param_store_param_groups_backend_parity` | `cuda_param_groups_match_cpu_for_sgd_adam_adamw` | `Device::Cuda(0)` の常駐更新（SGD・Adam・AdamW を 100 step）対 CPU ホスト参照 | Req2 | pass |
| 4 | #3778（雛形） | `fandhe-ai`・`train_gpu_parity` | `train_parity_optim_cuda_matches_cpu` | — | — | undetermined（登録 0 件。完了判定に数えない） |

judge は RULE.txt の参考の写しで、判定契約の正は各テストのコード。Req2 は REQ-2 統一複合判定（相対誤差 1e-3 未満 または 絶対誤差 1e-5 未満）。
各行の根拠は `summary.tsv` と `logs/<群>/<テスト名>.log`（末尾に `rc=<終了コード>`）。対象の実装 issue は既に CLOSED。Metal（Mac mini）は #3863 が測る。

## 回収物

| ファイル | 内容 |
|---|---|
| `rev-stamp.txt` | 展開ツリーの `.rev-stamp`（2 行形式） |
| `env_info.txt` | 実行前後の環境（GPU・ドライバ・nvcc・rustc・負荷・常駐件数・ロック保持） |
| `build.log` | 事前ビルド（`--no-run`。3 ターゲット）のログ |
| `summary.tsv` | 4 行の判定表 |
| `logs/<群>/` | 各テストの全出力（`--nocapture`）と終了コード |
| `SHA256SUMS` | 上記の SHA-256 |
