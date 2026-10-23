# cpu-gemm-reuse-release-3226

イシュー #3226（reuse が fresh より遅い逆転の是正）の実測記録置き場。設計・結果の本文は
`docs/perf/cpu-gemm-reuse-fresh-inversion.md` §8。

- `RULE.txt`: 事前登録規則（計測より前に固定。事後に緩めない）
- `diag-gb10/run1.jsonl`: GB10 での補助 diag（1 run・record-only・判定外。末尾行は取得時の uptime）
- 未実施: 5 round の A/B（GB10 14 セル・Mac mini 12 セル）と `.rev-stamp` 突合。実施時は
  `gb10/`・`macmini/`・`rev-stamp-verification-3226.md` をここへ追加する
