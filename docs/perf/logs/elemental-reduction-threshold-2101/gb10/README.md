# gb10（実測済み・2026-10-20・イシュー #3190）

本ディレクトリはイシュー #2101 の Phase 0 実測の収録先である。GB10 の 5 プロセス実測（`run1.log`〜`run5.log`・`env_info.txt`）を #3190 で収録した。
集計は GB10 単機（`../../optin-ab-gb10-3190/aggregate_sweep_single.py`。継承の `aggregate_sweep.py` は 2 機体必須のため）で `aggregate.md` に出した:
**判定 PROCEED・T_gb10 = 16384**（checksum は全セル一致・T 未満の max r = 1.0000）。最終 T は将来 min(T_macmini, T_gb10)。

## 手順

`../RULE.txt` の計測コマンドを 5 プロセス独立に実行し、次を収録する。

- `run1.log`〜`run5.log`: 各プロセスの標準エラー出力（生ログ）
- `aggregate.md`: (演算, n, 腕) ごとの代表値・r(n)・checksum 一致の表
- `env_info.txt`: OS・CPU・rayon スレッド数・load average の推移

CUDA の parity 確認（`cargo test -p fandhe-ai-backend-cuda -- --ignored` の reduce 系）も同セッションで実施する。
内部ホスト名・絶対パス・ユーザー名は `<home>` 等にマスクしてから置く。

## 集計・A/B（#2102）

Phase 0 の生ログの集計は `../../elemental-reduction-ab-2102/aggregate_sweep.py` で行う（判定規則は `../../elemental-reduction-ab-2102/RULE.txt`）。
