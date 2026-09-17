# `log_softmax` backward の Metal カーネル 実機実測ランブック（イシュー #1952）

本実装エージェントの実行環境（Linux）には Apple Silicon 実機への到達
手段がないため、`crate::log_softmax_backward::MetalLogSoftmaxBackward`
の実機 `#[ignore]` テストは未実行のまま Mac セッションへ申し送る。

設計・数値契約の正は `docs/backend-metal-reduce-sum-design.md` §12。

## 実行コマンド

Linux（CI・型検査のみ。実機なしでコンパイル可能性を担保）:

```sh
cargo check -p fandhe-ai-backend-metal --tests --target aarch64-apple-darwin
cargo check -p fandhe-ai --tests --target aarch64-apple-darwin
```

Apple Silicon 実機（`--release` 推奨）:

```sh
cargo test -p fandhe-ai-backend-metal --release \
  --test log_softmax_backward_source_evidence
cargo test -p fandhe-ai-backend-metal --release \
  --test log_softmax_backward_parity -- --ignored --nocapture
cargo test -p fandhe-ai --release \
  --test softmax_backend_parity -- --ignored --nocapture
```

## 保存すべきログ一覧

- `parity_run1.log`: `log_softmax_backward_parity.rs` の全 4 テスト
  （`--ignored --nocapture`）の生ログ。
- `facade_parity_run1.log`: `softmax_backend_parity.rs` の macOS 限定
  `#[ignore]` テスト 2 件（`metal_log_softmax_forward_matches_cpu`・
  `metal_log_softmax_backward_matches_cpu`）の生ログ。
- `env_info.txt`: 実行環境（macOS バージョン・チップ世代・Xcode/Metal
  ツールチェイン版）。内部ホスト名は含めない。

## 事前登録判定規則

1. `log_softmax_backward_parity.rs::
   metal_log_softmax_backward_matches_cpu_composite_judgment` が全
   ケース（7 形状 × dim）で REQ-2 統一複合判定 fail 0 であること。
2. `metal_log_softmax_backward_matches_cpu_bit_exact_when_y_is_zero`／
   `..._when_y_is_neg_infinity` が bit 完全一致（fail 0）であること。
3. `backend_ops_log_softmax_backward_matches_direct_api_call` が
   `BackendOps` 経由と直接呼び出しで完全一致すること。
4. 上記いずれも run-to-run で決定的（同一入力に対し複数回実行しても
   bit 同一）であること。
5. 性能実測（純カーネル時間・framework-compare 等）は本イシューの
   スコープ外（正しさ検証のみ）。実施する場合は 5 回計測中央値を
   採用する（`.claude/rules/coding-rust.md`）。

FAIL が生じた場合は是正せず本 README に記録のみ行い、原因調査は
別イシューへ切り出す（事後緩和はしない）。
