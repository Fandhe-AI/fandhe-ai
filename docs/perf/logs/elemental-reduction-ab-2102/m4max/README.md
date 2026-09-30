# m4max（未実測・Mac セッションで実施）

イシュー #2102 の収録先。**現時点で実測値はない**（基盤を実装したホストは判定対象機体ではない）。数値を推定で書かない。

- label: `2102-m4max`
- 手順・判定: `../README.md`・`../RULE.txt`
- Stage 0 の生ログは `../../elemental-reduction-threshold-2101/m4max/` へ収録し、`../aggregate_sweep.py` で集計する。
- Stage 1 の成果物（compare 表・JSONL・`load_gate.log`〈各 round の `gate_ok`〉・tree・sha）を本ディレクトリへ置く。
- 内部ホスト名・絶対パス・ユーザー名は `<home>` 等にマスクする。
