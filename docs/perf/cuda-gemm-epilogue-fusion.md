# CUDA GEMM epilogue 融合（bias・activation）計測記録（#599）

イシュー #599「feat(backend-cuda): elementwise カーネル（add / relu 系）を追加し
`gemm_bias_act` を実融合化」の実測記録。実装計画 8 章「実機検証」に対応する。

## 現状: 実機検証未完・ブロック中

本イシューの実装（`crates/backend-cuda/src/kernels_elementwise.rs`・
`elementwise.rs`・`kernels.rs::TILED_BIAS_ACT_F32`・
`gemm.rs::CudaGemm::run_tiled_bias_act_f32`・`ops.rs::CudaBackendOps` の
elementwise 5 演算実装・`gemm_bias_act` オーバーライド）は本セッションの
実行環境（CUDA 非搭載サンドボックス。`cargo test -p fandhe-ai-backend-cuda` は全て
`BackendError::CudaUnavailable` 環境適応経路で通過）で完結させた。

`docs/real-hardware-verification-env.md` §2 が示す CUDA 実機
（DGX Spark GB10 等。`docs/real-hardware-verification-env.local.md` の
実値を要する）への SSH 接続は本セッションから到達できないため、以下は
**未実施**である（実測値を捏造しない。`.claude/rules/coding-rust.md`
「ベンチは 5 回計測の中央値を採用」・security.md の実測原則に従う）:

- `cargo test -p fandhe-ai-backend-cuda --release -- --ignored --nocapture`
  （`tests/gemm_bias_act_parity.rs`・`tests/backend_ops_real_device.rs` の
  実機 `#[ignore]` テスト全体）
- 既存 `parity_nonregression.rs`（B-0・イシュー #491）による GEMM 数値
  一致の非後退確認（既存 GEMM カーネル・`kernels::TILED_F32` は本イシューで
  一切変更していないため理論上は非後退のはずだが、実機実測による確認は
  未実施）
- 融合 vs 非融合合成（`gemm`→`add`→`relu`）の 5 回計測中央値ベンチ

## 実機検証時の再現コマンド（未実施・手順のみ記録）

```bash
# docs/real-hardware-verification-env.md §2 の手順で CUDA_NODE へ
# ブランチを転送したうえで、実機上で実行する。

# 1. 新規テスト（elementwise・gemm_bias_act の CPU-CUDA 数値一致・
#    融合 vs 非融合合成の bit 完全一致・bias 形状グリッド・k=0 縮退）
cargo test -p fandhe-ai-backend-cuda --release --test gemm_bias_act_parity \
  -- --ignored --nocapture

# 2. 既存実機テスト全体（回帰確認）
cargo test -p fandhe-ai-backend-cuda --release -- --ignored --nocapture

# 3. B-0 parity 非後退契約（既存 GEMM カーネル不変更のため非後退のはず）
cargo test -p fandhe-ai-backend-cuda --release --test parity_nonregression \
  -- --ignored --nocapture
```

## 実測すべき項目（実機到達後に本節を実測値で置き換える）

| 項目 | 状態 |
|------|------|
| CPU-CUDA `gemm_bias_act` 数値一致（REQ-2 複合判定） | 未実施 |
| CUDA 上の融合 vs 非融合合成の bit 完全一致 | 未実施 |
| elementwise 5 演算の CPU-CUDA 数値一致 | 未実施 |
| bias 形状グリッド（`[n]`・`[1]`・`[1,n]`・不整合拒否・m/n/k=0 縮退・非 contiguous） | 未実施 |
| `assert_tolerance_constants_pinned`（B-0） | 未実施 |
| 融合 vs 非融合の 5 回計測中央値ベンチ | 未実施 |
| `parity_nonregression.rs` の fail_count／mean_abs_diff 非後退確認 | 未実施 |

## マージ可否についての注記

実機検証未完のため、本 PR のマージ可否はレビュー・ユーザー判断に委ねる
（実装計画 8 章の安全側判断）。CI で実行可能な範囲（環境適応スモーク
テスト・fmt／clippy／単体テスト）はすべて green であることを
`cargo fmt --all -- --check`・
`cargo clippy --workspace --all-targets --all-features -- -D warnings`・
`cargo test --workspace` で確認済み。
