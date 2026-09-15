# Metal Pooling（MaxPool／AvgPool／AdaptiveAvgPool）実機実測ログ置き場

イシュー #1730（親 #1607・設計 `docs/pooling-ops-design.md` §15
「Metal 実装（イシュー #1730）」）の Apple Silicon 実機実測を保存する
ディレクトリ。**本エージェント実行環境（Linux）には Apple Silicon
実機への到達手段がなく、本 PR 時点では実測は未実施**（本 README の
み・記入欄）。

## 位置づけ

- カーネル・ホストモデルの正しさ（bit 完全一致）は Linux 実行可能な
  `crates/backend-metal/tests/pooling_source_evidence.rs`・
  `crates/backend-metal/src/pooling_model.rs` の単体テストで既に
  機械検証済み（本 PR で all green）。
- 本ディレクトリが対象とするのは、実機（Apple Silicon GPU）での
  `MetalPooling::run_*` 実行が
  `crates/backend-metal/tests/pooling_parity.rs` の `#[ignore]`
  テスト群を pass することの確認のみ（性能実測は範囲外——Pooling は
  REQ-8 性能下限の対象外演算のため）。

## 実行コマンド（Mac 実機）

```sh
cargo test -p fandhe-ai-backend-metal --release --test pooling_parity -- --ignored --nocapture
```

既存 `#[ignore]` テスト群（他演算含む）の非後退確認:

```sh
make test-ignored-metal
```

## 判定規則

- `pooling_parity.rs` の全テスト（`max_pool2d_bit_exact_against_host_model`・
  `max_pool2d_special_values_bit_exact`・
  `avg_pool2d_bit_exact_against_host_model`・
  `adaptive_avg_pool2d_bit_exact_against_host_model`・
  `zero_batch_returns_empty_without_device_dispatch`・
  `oversized_shape_returns_size_limit_exceeded_error`）が pass する
  こと（bit 完全一致・決定性）。
- 既存 `#[ignore]` テスト群（`make test-ignored-metal`）が非後退で
  あること。

## 保存すべきログ

- 上記 2 コマンドの標準出力（内部ホスト名は含めない）。
- `env_info.txt`（`docs/real-hardware-verification-env.md` の手順に
  従い OS・チップ・rustc バージョン等を記録。内部ホスト名は含めない）。
