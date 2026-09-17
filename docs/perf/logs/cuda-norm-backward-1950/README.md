# RMSNorm／LayerNorm backward の CUDA カーネル（イシュー #1950）実機実測申し送り

本エージェント実行環境には DGX Spark GB10（またはその他 CUDA 実機）への到達手段がないため、
`crates/backend-cuda/tests/norm_backward_parity.rs` の `#[ignore]` テストは未実行のまま
本ディレクトリへ申し送る。Linux（CUDA 非搭載）で実行可能な範囲（環境適応スモーク・
静的検査・単体テスト・fmt／clippy）はすべて green を確認済み（`docs/norm-ops-design.md` §10）。

## 実行コマンド

```sh
cargo test -p fandhe-ai-backend-cuda --release --test norm_backward_parity -- --ignored --nocapture
cargo test -p fandhe-ai --release --test norm_backend_parity -- --ignored cuda_ --nocapture
```

非後退確認（既存 3 テストバイナリ。無変更のはず）:

```sh
cargo test -p fandhe-ai-backend-cuda --release --test rmsnorm_backward_parity -- --ignored --nocapture
cargo test -p fandhe-ai-backend-cuda --release --test rmsnorm_parity -- --ignored --nocapture
cargo test -p fandhe-ai-backend-cuda --release --test layer_norm_parity -- --ignored --nocapture
```

## 事前登録判定規則（実測後の事後緩和はしない）

1. `norm_backward_parity.rs` の新規 `#[ignore]` テスト（
   `rmsnorm_backward_matches_cpu_across_shapes`・
   `layer_norm_backward_matches_cpu_across_shapes`・
   `norm_backward_zero_element_contract`・
   `norm_backward_numerical_stability_and_determinism`）全 pass ＝
   REQ-2 統一複合判定（相対誤差 1e-3 未満 または 絶対誤差 1e-5 未満）成立。
2. `crates/facade/tests/norm_backend_parity.rs` の
   `cuda_rms_norm_backward_matches_cpu`／`cuda_layer_norm_backward_matches_cpu`
   （facade 経由の横断 parity）全 pass。
3. 上記コマンドの「非後退確認」3 本（既存テスト。本イシューでは無変更）が
   引き続き pass すること（forward・既存 backward API を壊していないことの確認）。
4. FAIL が出た場合、tolerance／baseline は変更せず記録のみとする（是正は別 PR）。
5. 性能 A/B（純カーネル時間・framework-compare）は本イシューの受入条件に
   含まれない（記録のみ任意。`docs/norm-ops-design.md` §9「対象外事項」）。

## 記入欄

- 実行日時:（未実測）
- GB10 実機コミット sha:（未実測）
- 判定結果:（未実測）
- 備考:（未実測）
