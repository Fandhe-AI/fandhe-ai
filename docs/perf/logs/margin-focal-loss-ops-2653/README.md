# MultiMargin・MultiLabelMargin・MultiLabelSoftMargin・sigmoid focal loss（#2653）CUDA／Metal 実機未実測の申し送り

> CUDA は #3161 で GB10 実測済み（2026-10-19・コミット `b640875f`・pass 2/2）。結果は末尾「CUDA 実測記入欄（#3161）」節。

`docs/autodiff-margin-focal-loss-ops-decision.md` §10 参照。本実装エージェント実行環境は CUDA／Metal 実機に
到達できないため、`crates/facade/tests/margin_focal_loss_ops_backend_parity.rs` のうち CUDA（`Device::Cuda(0)`）・
Metal（`Device::Metal`。`cfg(target_os = "macos")` 限定）を対象とする計 4 テストは `#[ignore]` のまま未実測である。

- CUDA: `cuda_bit_exact_forward_matches_cpu_reference`・`cuda_bit_exact_backward_matches_cpu_reference`
- Metal: 上記と対称の `metal_*` 2 件

## 測定コマンド案

```sh
# CUDA（DGX Spark GB10 等の実機上で）
cargo test -p fandhe-ai --test margin_focal_loss_ops_backend_parity -- --ignored --nocapture cuda_

# Metal（Apple Silicon 実機上で）
cargo test -p fandhe-ai --test margin_focal_loss_ops_backend_parity -- --ignored --nocapture metal_
```

CPU（`CpuBackendOps`）版は同テストファイルの属性なしテスト（`cpu_*`）で既に検証済み（green）。

## 期待結果

- 本実装は新規 GPU カーネルを持たない。新規 Op 4 種は常にホスト参照実装（`eval::margin_focal_loss`）で計算されるため、
  CUDA／Metal の tape でも CPU tape と **bit 一致**する見込み（REQ-2 複合判定を使う経路はない）。
- 形状は小さく（`[3, 4]`）、Metal の split-K 経路が発動する形状は含まない。

この前提が崩れる場合は本 README の「期待結果」を更新し、判定を外れた事実を PR へ記録すること
（tolerance の単独緩和は行わない。`.claude/rules/coding-rust.md`）。

## 実測記入欄

| 日付 | 実機 | コマンド | 結果 | 備考 |
|---|---|---|---|---|
| 2026-10-19 | DGX Spark GB10 | `cargo test -p fandhe-ai --test margin_focal_loss_ops_backend_parity -- --ignored --nocapture --test-threads=1 cuda_`（`cuda/RULE.txt`） | pass（2/2） | `.rev-stamp` `b640875f`・dirty=0。ログは `cuda/` |
| 未実測 | Mac mini（Apple M4） | 上記 Metal | 未実測 | Linux 環境から Mac mini に届かない（接続先情報なし）。Mac 側へ申し送り（#3162）。事前登録規則は `metal/RULE.txt` |

## Metal 実測記入欄（#3162・準備済み・未実測）

- 判定規則・実行コマンド・期待件数（2 件）は事前登録規則 `metal/RULE.txt` を正とする（実機実行より前に固定済み）。
- 本セッションは Linux 環境で Mac mini（Apple M4）に到達できなかったため実機を実行していない。
  pass／fail・`.rev-stamp`・`env_info.txt`・ログは作っておらず、結果は「未実測」のままである。
- Mac 側の手順: 展開ツリーの `.rev-stamp` に HEAD の全桁 SHA を書いて転送（`docs/real-hardware-verification-env.md` §3・§7）、
  ノード側で `.rev-stamp` 一致と他の cargo／rustc 不在を確認、`RULE.txt` のコマンドを実行、
  ログ・env_info（before／after）をホスト名・ユーザー名・個人パスをマスクして `metal/` へ置き、
  `metal/rev-stamp.txt` に `.rev-stamp` を写し、本節と上の記入欄へ pass／fail 件数を記録する。
- fail の場合は tolerance・baseline・入力を変えず、失敗件数・テスト名・最大誤差を記録する。

## CUDA 実測記入欄（#3161）

- 判定規則・fail の扱い・実行コマンド・期待件数（2 件）は事前登録規則 `cuda/RULE.txt` を正とする（実測前に固定済み。事後に変更していない）。
- 実行コマンド: 上記 `RULE.txt` のとおり独立プロセスで 1 回（並列度 1・`--release` なし・`timeout 1800`）。再実行・選別はしていない。
- 対象コミット: `b640875f`（`cuda/rev-stamp.txt`: 1 行目 sha・2 行目 `dirty=0`。実行前・各ファイル実行後・実行後の 8 回の読み出しがすべて一致）。
- 環境: NVIDIA GB10（aarch64・ドライバ 580.173.02）・CUDA 13.0（nvcc 13.0.88）・rustc 1.97.0。詳細は `cuda/env_info.txt`（実行前後）。
- 備考: 常駐サービスの GPU プロセス 3 件（実行前後で同一 pid・`utilization.gpu` 0%）以外の GPU 負荷なし。作業ツリーは初回転送でビルドキャッシュが冷えていたため、判定外のウォームアップ（`--no-run`、`cuda/build.log`）で 6 テストファイル分をビルドしてからテストを実行した（テスト自体は 1 回ずつ）。

| ファイル | 関数 | 結果 | 備考 |
|---|---|---|---|
| `crates/facade/tests/margin_focal_loss_ops_backend_parity.rs` | `cuda_bit_exact_backward_matches_cpu_reference` | pass | |
| `crates/facade/tests/margin_focal_loss_ops_backend_parity.rs` | `cuda_bit_exact_forward_matches_cpu_reference` | pass | |

- 合計 2/2 pass（fail 0）。実行時間 0.21s と短く、経路確認（GPU フック経由かフォールバックか）は追従課題。フォールバック経路を通ったとは断定しない。
- tolerance 定数・baseline・判定規則・テストコードは不変。
