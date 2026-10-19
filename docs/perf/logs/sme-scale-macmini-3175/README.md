# sme-scale-macmini-3175

イシュー #3175（親 #3174）: Mac mini（Apple M4）での SME の多スレッド頭打ちと micro-call 固定費の計測ログ置き場。結論は `docs/perf/cpu-sme-thread-scaling-macmini.md`、判定規則は `RULE.txt`（計測前に単独コミット済み）。

## 成果物

| ファイル | 内容 |
|---|---|
| `RULE.txt` | 事前登録規則 |
| `orchestrate_macmini.sh` | 5 run x スレッド数 x 系列の起動・負荷ゲート（`--dry-run` あり） |
| `aggregate.py` | 厳密パースの集計（`--self-test` あり。標準ライブラリのみ） |
| `env_info.txt` | チップ・macOS・rustc・perflevel・FEAT_SME・`.rev-stamp` |
| `load_gate.log` | 各プロセスの START（ゲート結果）/ END（exit）記録 |
| `thread_t{1,2,4,10}_run{1..5}.log`（`.raw.log`） | 系列 A: SME/NEON のスレッド掃引 |
| `microcall_run{1..5}.log`（`.raw.log`） | 系列 B: micro-call 固定費 |
| `ecore_t1_run{1..5}.log`（`.raw.log`） | 系列 C: `taskpolicy -b` による E コア側 |
| `aggregate.md` | `aggregate.py` の出力 |

`.log` は `.raw.log` から行頭アンカーの grep で抽出した行のみ。系列 D（`sme_vs_neon_ab_r4_grid` の参考回帰）は RULE.txt 6 節の「省略可」に従い実施しない。

## 手順

1. `docs/real-hardware-verification-env.md` §7.1 の rsync 方式でツリーを Mac mini へ転送（`.rev-stamp` にコミット sha を書く）。ホスト名は Git 管理外の `docs/real-hardware-verification-env.local.md` にあり、ここには書かない
2. `cargo test -p fandhe-ai-backend-cpu --release --lib --no-run` で事前ビルド（計測中はビルドを並走させない）
3. `nohup sh docs/perf/logs/sme-scale-macmini-3175/orchestrate_macmini.sh > run.out 2>&1 &` で切り離して起動し、`ps` で生存確認（macOS に `setsid` は無い）
4. 完了後にこのディレクトリを回収し、`python3 -I aggregate.py > aggregate.md`
