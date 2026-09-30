# m4max（未実測・Mac セッションで実施）

本ディレクトリはイシュー #2101 の Phase 0 実測の収録先である。**現時点で実測値はない**
（実装を行ったホストは判定対象機体ではないため）。数値をここへ推定で書かない。

## 手順

`../RULE.txt` の計測コマンドを 5 プロセス独立に実行し、次を収録する。

- `run1.log`〜`run5.log`: 各プロセスの標準エラー出力（生ログ）
- `aggregate.md`: (演算, n, 腕) ごとの代表値・r(n)・checksum 一致の表
- `env_info.txt`: OS・CPU・rayon スレッド数・load average の推移

Metal の parity 確認（`cargo test -p fandhe-ai-backend-metal -- --ignored` の reduce 系）も同セッションで実施する。
内部ホスト名・絶対パス・ユーザー名は `<home>` 等にマスクしてから置く。

## 集計・A/B（#2102）

Phase 0 の生ログの集計は `../../elemental-reduction-ab-2102/aggregate_sweep.py` で行う（判定規則は `../../elemental-reduction-ab-2102/RULE.txt`）。
