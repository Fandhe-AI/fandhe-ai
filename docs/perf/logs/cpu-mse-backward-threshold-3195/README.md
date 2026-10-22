# #3195 MSE backward 逐次しきい値の再測定（T = 4096）

状態: **未計測**（規則とハーネスのみ。実機計測は未実施。判定は undetermined 相当で、
`MSE_BACKWARD_PARALLEL_MIN_ELEMS` は 0 のまま）。

| 項目 | 内容 |
|---|---|
| 規則 | `RULE.txt`（計測前に固定） |
| ハーネス | `scripts/bench/framework-compare/run_ab_mse_backward_threshold_cpu.sh` |
| 候補 | T = 4096（単一） |
| before 腕 | main HEAD の `crates/facade`（定数 0） |
| after 腕 | 同一コミット + `mse.rs` 定数 1 行のみ変更（計測専用・非コミット） |
| 判定セル | train 64 × {fresh, reuse}（cpu）。`--phases` は記録のみ |
| 機体 | Mac mini（`macmini/`）・GB10（`gb10/`）。各 5 round |
| 過去記録 | #1578（T = 1<<18・REJECT）は不変。本件はそのデータを読み直さない |

## 未計測の理由と次の手順
- 本ブランチは自動運転の実装フェーズで作成され、実機（Mac mini・GB10）計測は未実施。推定値では埋めない。
- 計測手順は `docs/real-hardware-verification-env.md` §3・§4.4・§7 に従い、`.rev-stamp` とツリー指紋で腕を同定し、
  Stage 0（`mse_backward_threshold_sweep` 5 プロセス）→ Stage 1（ハーネス）の順に実行し、成果物を
  `macmini/`・`gb10/` へ回収して本 README に判定表を追記する。
- 系列開始前の preflight は**手動で記録する**（ハーネスは記録も未実施時の停止もしない）。各機体で
  `ps -axo pid,etime,command | grep -E 'cargo|rustc|bench' | grep -v grep` の出力を、他セッションの該当プロセスが
  0 件であることを確認したうえで `preflight-procs.txt`（`macmini/`・`gb10/` 配下）へ保存してから Stage 0 を開始する。
  1 件でも残っていれば計測を開始しない（RULE.txt §5）。
