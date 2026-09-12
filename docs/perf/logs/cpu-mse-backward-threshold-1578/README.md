# mse_loss_backward 逐次フォールバックしきい値（イシュー #1578）実測ログ

事前登録した規則・全体の判定・原因分析は
`docs/perf/cpu-mse-backward-sequential-threshold.md` を正とする。
本ディレクトリはその実測生ログを保存する。

## 構成

- `m4max/`・`gb10/`: 機体別（内部ホスト名は含めない）
  - `run{1..5}.log`: Phase 0 マイクロベンチ（`mse_backward_threshold_sweep`。
    forced-seq／forced-par の中央値 ns を機体ごと・サイズごとに記録）
    のプロセス起動 5 回分
  - `compare-train-1578-cpu.md`: Phase 1（framework-compare train A/B・
    `--device cpu` の reuse セル。事前登録した必須判定）の結果表
  - `compare-train-1578-cpu-fresh-reference.md`: 同 fresh セル（対照・
    参考。判定には用いない）
  - `uptime-1578-cpu-1578.log`: Phase 1 各 run 前後の `uptime`
  - `env_info.txt`: 機体属性・実行時の共有負荷（record_only。専有ゲート
    なし）

## 再現

Phase 0:
```
cargo test -p fandhe-ai-backend-cpu --release --lib \
  mse::tests::mse_backward_threshold_sweep -- --ignored --nocapture
```

Phase 1（`scripts/bench/framework-compare/run_ab_1578.sh`。
`AB_BEFORE_FACADE_PATH`／`AB_AFTER_FACADE_PATH` に before（`origin/main`
の `crates/facade`）／after（本ブランチの `crates/facade`）の絶対パスを
指定）:
```
AB_BEFORE_FACADE_PATH=/path/to/before/crates/facade \
AB_AFTER_FACADE_PATH=/path/to/after/crates/facade \
AB_DEVICE=cpu AB_ROUNDS=5 \
  bash scripts/bench/framework-compare/run_ab_1578.sh 1578
```
