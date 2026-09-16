# Metal f32 sum reduction（イシュー #1895）実機実測 申し送り

本エージェント実行環境には Apple Silicon 実機への到達手段がないため、
`crates/backend-metal/tests/reduce_parity.rs`（`#[ignore]`）は未実行
のまま Mac セッションへ申し送る。設計・実装記録は
`docs/backend-metal-reduce-sum-design.md` を参照。

## Linux でここまで完了していること

- `cargo test -p fandhe-ai-backend-metal --lib reduce_model`:
  ホストモデル ⇔ CPU 参照実装（`fandhe_ai_backend_cpu::reduction::
  sum`）の bit 完全一致（全 11 テスト green）
- `cargo test -p fandhe-ai-backend-metal --test
  reduce_source_evidence`: MSL ソース文字列証跡（全 7 テスト green）
- `cargo check -p fandhe-ai-backend-metal --tests --target
  aarch64-apple-darwin`: `reduce.rs`／`reduce_parity.rs` を含む
  クロス型検査 green（既存の無関係な warning のみ・新規 error なし）
- `cargo clippy --workspace --all-targets --all-features -- -D
  warnings`: green（`#[allow]` 追加なし）

## Mac 実機で実行すべきコマンド

```sh
cargo test -p fandhe-ai-backend-metal --release --test reduce_parity \
  -- --ignored --nocapture 2>&1 | tee reduce_parity.log
```

## 保存すべきログ

- `reduce_parity.log`: 上記コマンドの標準出力・標準エラー全文
- `env_info.txt`: 実行環境情報（`.claude/rules/security.md`・
  `docs/real-hardware-verification-env.md` に従い内部ホスト名は
  含めない。macOS バージョン・チップ種別・`rustc --version` 程度）

## 判定規則

- `reduce_parity.rs` の 3 テスト
  （`metal_sum_all_matches_cpu_bit_exact`／
  `metal_sum_axis_matches_cpu_bit_exact`／
  `metal_sum_axis_empty_cases_match_cpu`）が全 pass すること
  （bit 完全一致契約・NaN クラス一致・run-to-run 決定性）
- 性能実測（純カーネル時間等）は本イシューのスコープ外
  （`MetalBackendOps::sum` 未結線のため framework-compare 等の
  実践規模計測は #1896 以降が対象）
