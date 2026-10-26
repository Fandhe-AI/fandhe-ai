# DIAG-r5 の分類（イシュー #3610。手順は `../DIAG-r5.txt`）

分類規則は `DIAG-r5.txt` の [分類規則]（計測前に固定）をそのまま当てた。規則は事後に変えていない。各 run の `DIAG3610 summary` 行の転記（`run-<ツリー>-<n>.log`）。

| ツリー | run | rc | mismatches | matched_full | unmatched | idx_ge_20 | in_window | plain_params_diff | replay_bad |
|---|---|---|---|---|---|---|---|---|---|
| before（`f7fe6b4df`） | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | false |
| before | 2 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | false |
| before | 3 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | false |
| before | 4 | 101 | 2 | 2 | 0 | 1 | 2 | 0 | false |
| before | 5 | 101 | 1 | 1 | 0 | 1 | 1 | 0 | false |
| main_prefix（`c1204755e`） | 1 | 101 | 3 | 3 | 0 | 3 | 3 | 0 | false |
| main_prefix | 2 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | false |
| main_prefix | 3 | 101 | 1 | 1 | 0 | 0 | 1 | 0 | false |
| main_prefix | 4 | 101 | 6 | 6 | 0 | 5 | 6 | 0 | false |
| main_prefix | 5 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | false |
| after（`616be4ff7`） | 1〜5 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | false |

- 集計: before + main_prefix の 10 run 中 5 run が失敗（rc=101）。ずれの総数は 13 件（before 3・main_prefix 10）。13 件すべてが `full-reference-idx` と `replay-live-idx` の両方で**ちょうど 1 つの index と bit 一致**し、その index は両者で同じだった。unmatched は 0。
- パラメータ: 全 15 run・全 30 repeat（計 450 repeat）で `plain-params-equal=true`（20 step の参照パラメータと bit 一致）。`plain_params_diff` は 0。
- after: 5 run すべて rc=0、mismatches = 0、plain_params_diff = 0。
- 補足: ずれた loss の一致 index は 15〜176 に分布する。うち 10 件は 20 以上（`short` の外）で、r2 の `equals-reference-idx []` は、r2 が参照の先頭 20 件にしか照合していなかったことで説明できる。残る 3 件は index 15・16・17（replay が序盤の step にいる間の読み取り）で、20 件の範囲内だったが、plain 自身の同じ位置の参照（idx 0 など）とは別の位置だった。

## 当てはまった場合

- **(a)**: 当てはまる。before と main_prefix で失敗があり、unmatched = 0、plain_params_diff = 0、after は 5/5 で mismatches = 0 かつ plain_params_diff = 0。
- (b) 当てはまらない（unmatched = 0）。
- (c) 当てはまらない（plain_params_diff = 0）。
- (d) 当てはまらない（失敗が出た）。
- (e) 当てはまらない（after は 5/5 で一致）。

## 結論の範囲

- r2 の食い違いは診断の照合範囲（先頭 20 件）による見かけ上のもので、ずれた plain の loss は replay スレッドのいずれかの step の loss と bit 一致した（H1 を支持。H2・H3 は棄却）。
- ずれた値が「相手スレッドの step の loss」であることは実測で示された。一方、`partial`／`finalize` のどちらの起動がどの順で割り込んだか（finalize が graph の partial を読む、という起動順そのもの）は loss の値からは直接観測できない。機構の記述は「実測と整合する推定」であり、起動順の直接確認ではない。
- `in_window`（一致 index が replay の進捗窓 `[before, after+1]` に入る）は 13/13 だが、窓が広い run があり（例: `[52, 111]`）、補強にとどまる。
