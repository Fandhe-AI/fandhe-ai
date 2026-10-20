# gb10（実測済み・2026-10-20・イシュー #3190）

イシュー #2102 の収録先。Stage 1（T = 16384。`../../optin-ab-gb10-3190/on-reduction-gb10.patch`）を #3190 で実測し、`*2102-gb10-3190*` に収録した。
**GB10 機体判定: undetermined**（正式系列 5/5 round が load1 < 8.0）。非後退でないセルは gemm 2048/fresh 1.0192 と infer reuse 1.0049 で、いずれも 5/5 round 一貫ではない（REJECT 条件に当たらない）。
ほかは ratio <= 1.00・checksum 完全一致。総合は Mac mini（#3184）待ち。

- label: `2102-gb10`
- 手順・判定: `../README.md`・`../RULE.txt`
- Stage 0 の生ログは `../../elemental-reduction-threshold-2101/gb10/` へ収録し、`../aggregate_sweep.py` で集計する。
- Stage 1 の成果物（compare 表・JSONL・`load_gate.log`〈各 round の `gate_ok`〉・tree・sha）を本ディレクトリへ置く。
- 内部ホスト名・絶対パス・ユーザー名は `<home>` 等にマスクする。
