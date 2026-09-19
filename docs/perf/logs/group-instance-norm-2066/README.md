# GroupNorm／InstanceNorm 実機 parity 実測 申し送り（イシュー #2066）

本エージェント実行環境に CUDA（DGX Spark GB10 等）・Metal（Apple Silicon）
いずれの実機への到達手段もないため、`crates/facade/tests/
group_instance_norm_backend_parity.rs` に追加した `#[ignore]` テスト群
（`metal_group_norm_forward_matches_cpu`／`cuda_group_norm_forward_matches_cpu`／
`metal_instance_norm_forward_matches_cpu`／`cuda_instance_norm_forward_matches_cpu`）
は未実行のまま本 PR をマージする。実機セッションで以下を実行し、本ファイルへ
結果を追記すること。

`GroupNorm`／`InstanceNorm` は既存の最終軸限定 `Var::layer_norm`（`nn::norm`。
イシュー #1596。CPU／CUDA／Metal 3 バックエンドとも forward カーネル実装済み）
を reshape で挟むだけの合成であり、新規カーネルは一切追加していない。したがって
本実測の主目的は「reshape 経由の呼び出しでも既存 `layer_norm` カーネルの正しさ・
数値契約が壊れていないこと」の確認であり、性能実測（A/B・採否判定）は対象外。

## 実行コマンド

```sh
# Linux 実行可能な単体テスト（GPU 不要。CPU 上の CpuBackendOps vs NaiveOps 突合）
cargo test -p fandhe-ai-autodiff --lib normalization
cargo test -p fandhe-ai --test group_instance_norm_backend_parity

# 実機必須テスト（コンパイル確認のみは --no-run で可能）
cargo test -p fandhe-ai --test group_instance_norm_backend_parity -- --ignored --nocapture

# 一括実行（他の #[ignore] テストと同時実行する場合）
make test-ignored-cuda    # 対象リポジトリの Makefile に定義があれば（CUDA 側）
make test-ignored-metal   # 同上（Metal 側）
```

## 保存すべきファイル

- 上記コマンドの標準出力（各テスト名・pass/fail）
- `env_info.txt`（内部ホスト名は含めない。GPU 型番・CUDA/driver バージョン、
  または macOS／Apple Silicon 型番の要点のみ）

## 事前登録判定規則

- `cuda_group_norm_forward_matches_cpu`／`cuda_instance_norm_forward_matches_cpu`／
  `metal_group_norm_forward_matches_cpu`／`metal_instance_norm_forward_matches_cpu`:
  `fandhe_ai_backend_cpu::parity::assert_parity`（REQ-2 統一複合判定「相対誤差
  1e-3 未満 または 絶対誤差 1e-5 未満」）で CPU tape 経由の結果と一致すること。
  `GroupNorm`／`InstanceNorm` 自体は算術を追加しないため、既存 `layer_norm`
  カーネルの CUDA／Metal 実装が満たす契約（`docs/norm-ops-design.md`）がそのまま
  引き継がれる。
- FAIL が生じた場合は**是正せず記録のみ**とする（本 PR の受け入れ条件は
  Linux 実行可能テストの green のみであり、実機実測は申し送り事項）。

## 実測結果（記入欄）

（未実測。実機セッションで追記すること）
