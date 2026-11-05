# param groups と DeviceParamStore の GPU 常駐更新の Metal 実機 parity（#3863）

対象は param groups と `DeviceParamStore` の GPU 常駐更新（実装 #3837・#3838）で、`docs/autodiff-param-groups-decision.md` §19.4 が
Phase 10 で測ると定めた Metal 側の `#[ignore]` 3 本を CPU 参照と突き合わせた。機体は `<metal-node>`（Mac mini・Apple M4・メモリ 16GB。
`docs/real-hardware-verification-env.md` §7）。GB10 版は #3862。雛形は `../train-gpu-parity-template/RULE.txt`（#3778）。

- 事前登録: [`RULE.txt`](./RULE.txt)（宣言 2026-11-04T14:15Z。実機実行より前にコミット 39cb7d101 で push した。実行後に変えていない）
- 実行スクリプト: [`run_metal_parity.sh`](./run_metal_parity.sh)（Mac mini 上で手動実行。CI からは呼ばない。`--self-test` 付き）。
  `../nn-layers-metal-parity-3861/run_metal_parity.sh`（#3861）の複製に、GB10 版 #3862 と同じく package をケースごとに書く変更を足した
- tolerance 定数（`RELATIVE_TOLERANCE`／`ABSOLUTE_RESCUE_THRESHOLD`）・parity baseline・判定規則・テストコードは不変

## 状態

**実測済み: 測定対象 3 本中 3 本 pass、fail 0、undetermined 0**（2026-11-04・`<metal-node>`）。

**要対応（所有者へ報告。GB10 の #3862 と同じ）: 雛形 `crates/facade/tests/train_gpu_parity.rs` の optim 族は登録 0 件。**
`train_parity_optim_metal_matches_cpu` は `undetermined: 族 optim の登録が 0 件`（`crates/facade/tests/common/train_gpu_parity.rs:193`）で
panic し、exit 101・判定 undetermined だった。RULE.txt で事前に宣言したとおり §19.4 の 3 本を測定対象にし、雛形の optim 族は
完了判定に数えていない。所有者の判断（2026-11-04）により、雛形への optim 族の追加は #4026 で行う。雛形（学習ステップ）は #4026 の登録待ちで未実測であり、#3863 は #4026 のマージ後に雛形テストを再計測してから閉じる（この記録の PR は `Refs`。RULE.txt はこの判断より前に固定したため変えていない）。

- 実行コミット（`rev-stamp.txt`）: `39cb7d10160115ebdbbe23958ce8c43ae390c115`・`dirty=0`。事前登録と実行スクリプトを含むコミットそのもので、
  `crates/`・`Cargo.toml`・`Cargo.lock`・`rust-toolchain.toml` は対象コミット `b82b1e5d1` と同一（送信前に `git diff --quiet` で確認）。
  転送は rsync だけで、ノード側で `.rev-stamp` を読んで送信側の HEAD と照合した。
- スクリプト対応: `env_info.txt` の `script_sha256`（`40d242ae…0e5e`）は、上記コミットの `run_metal_parity.sh` の SHA-256 と一致する。
- 実機ロック: `env_info.txt` の実行前・実行後とも `hw_lock: held owner=#3863 2026-11-04T14:16:10Z`。終了後にロックが外れていることを確かめた。
  ほかの利用者のプロセスは止めていない。並走する cargo／rustc は事前ビルド前・各テスト前とも無し。メモリの空き率は実行前後とも 72%。
- 回収物は手で編集していない。`SHA256SUMS` はこのディレクトリで `shasum -a 256 -c SHA256SUMS` が通る（12 ファイルすべて OK）。
  ホスト名・`/Users/`・ユーザー名の残存は 0 件（`<metal-node>`／`<user>`／`<home>` へマスク済み。送信側でも grep で確認）。
- デバイス経路: backend-metal の 2 本は `MetalBackendOps::new()` の `memory_ops().unwrap()` と `assert!(supports_grouped_optimizer_step())`、
  facade の 1 本は `tape_for(Device::Metal).unwrap()` で受ける。Metal が使えなければ panic するため、CPU への無言のフォールバックで pass する
  ことは無い（早期 return・skip の分岐が無いことを対象ファイルで確かめた）。
- 数値契約は §19.5 の (3)（Metal + groups → CPU 参照と REQ-2 統一複合判定）。3 本とも `tolerance::composite_close_f32` で要素ごとに判定する。
  GPU の bit 一致は合否の基準にしていない。
- 事前ビルド（3 ターゲットの `--no-run`）は 7.83 秒・9.92 秒・0.41 秒（`build.log`）。各テストは 0.23〜0.77 秒で終わった。

## 結果表

| # | 実装 issue | テスト（`-p` パッケージ・`--test` ターゲット／`--exact`） | 判定 | 結果 |
|---|---|---|---|---|
| 1 | #3837 | `fandhe-ai-backend-metal`・`grouped_optimizer_step_parity`／`sgd_grouped_matches_cpu_grouped` | Req2 | pass |
| 2 | #3837 | `fandhe-ai-backend-metal`・`grouped_optimizer_step_parity`／`adam_grouped_matches_cpu_grouped` | Req2 | pass |
| 3 | #3838 | `fandhe-ai`・`device_param_store_param_groups_backend_parity`／`metal_param_groups_match_cpu_for_sgd_adam_adamw` | Req2 | pass |
| 4 | #3778（雛形） | `fandhe-ai`・`train_gpu_parity`／`train_parity_optim_metal_matches_cpu` | - | undetermined（登録 0 件。完了判定外） |

Metal 版の adam テストは CUDA 版（`adam_grouped_matches_cpu_grouped_and_rejects_bad_totals`）と違い、不正な合計の拒否を含まない
（テスト名・内容は対象コミットのまま）。各行の根拠は `summary.tsv`（exit code・`running 1 test` の件数・`test result` 行）と
`logs/<群>/<テスト名>.log`（末尾に `rc=`）。対象の実装 issue #3837・#3838 は既に CLOSED。CUDA（GB10）は #3862（PR #4023・マージ済み）で 3 本中 3 本 pass。

## 回収物

| ファイル | 内容 |
|---|---|
| `rev-stamp.txt` | 展開ツリーの `.rev-stamp`（2 行形式） |
| `env_info.txt` | 実行前後の環境（macOS・CPU・メモリ量と空き率・rustc・負荷・ロック保持） |
| `build.log` | 事前ビルド（3 ターゲットの `--no-run`）のログ |
| `summary.tsv` | 4 行の判定表 |
| `logs/<群>/` | 各テストの全出力（`--nocapture`）と終了コード |
| `SHA256SUMS` | 上記の SHA-256 |

## #4044 追記（共有 target の成果物の再利用の棚卸し）

- 分類: **A（再コンパイルを確認）**
- 根拠: build.log に 9 メンバー全部の `Compiling` がある（大半が 2 回）。
- 判定基準・全 16 計測の表・限界は [`docs/real-hardware-shared-target-reuse-decision.md`](../../../real-hardware-shared-target-reuse-decision.md) §4。本追記は結果を変えない（既存の本文・回収ログは無改変）。
