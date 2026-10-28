# #3672 download_owned の before／after A/B（計測部品と回収物）

CPU の `matmul_resident` が `MemoryOps::download_owned`（Vec 移動）へ切り替わったこと（PR #3677）で、
GB10 の N=4096 の reuse 逆転（#3646 §9.7 H1: download 区間）が縮むかを、実機 A/B で判定する。
判定の正は同ディレクトリの `RULE.txt`（実測前固定）と `RULE.v2.txt`（解釈の固定。閾値は不変）。
結果と結論は `docs/perf/cpu-gemm-reuse-fresh-inversion.md` §11 に書く。

## 構成

| ファイル | 役割 |
|---|---|
| `RULE.txt` / `RULE.v2.txt` | 事前登録規則（編集しない）。sha256 は各 `env_info.txt` に記録 |
| `build_ab_arms.sh` | 両腕の木から bench-fandhe（両腕とも path patch）と診断テストのバイナリ 4 本を hw-lock の前に作る。ハード条件（診断テストと framework-compare が両腕で同一）を強制 |
| `orchestrate.sh` | 1 機体分の正式系列。腕ごとに診断 1 ＋ bench reuse `--phases`（N=256..4096）を 5 run、run ごとに起動順を反転（ABBA） |
| `session-scripts/session-3672.sh` | hw-lock・GB10 の専有 poller（30 秒間隔）・orchestrate・`excl_check.py --series main`・機体別 judge |
| `ab_judge.py` | RULE を機械的に適用する判定器（標準ライブラリのみ・fail-closed・`--self-test` 付き） |
| `gb10/` `macmini/` | 回収した生ログ（マスク済み）と機体別 judge |
| `judge-3672.md` | 両機体をまたぐ最終判定（`ab_judge.py` の出力そのまま） |
| `rev-stamp-verification-3672.md` | 腕の同定（`.rev-stamp`・ツリー指紋・バイナリ sha256） |

専有の合否は #3646 の `docs/perf/logs/cpu-gemm-resident-reuse-3646/excl_check.py` を `--series main` で再利用する（コピー・編集しない）。

## 腕

| 腕 | sha | 取得 |
|---|---|---|
| before | `a7aac2f0227add243845418e0897ac0675c5b622`（#3677 のマージベース） | `git archive` |
| after | `f7ce20fc91730216e8f3f60514b181f32ebfee48`（#3677 の PR head。tree は `e03f81957` と同一） | `git archive` |

registry の `fandhe-ai =0.12.0` は before に使わない。両腕とも各木の `crates/facade` への path patch でビルドする。

## セル集合の解釈（RULE.v2）

RULE.txt の「非処置セル: 上記以外の全セル」は、機体ごとに bench の `iter_total` × N 5 点と、診断の全 8 腕の `iter_total` × N 5 点と読む。
`iter_total` 以外の区間は記録のみ（`DL_before`・`X` の計算にだけ使う）。ノイズで中央値 ratio が 1.00 を超えた非処置セルも ADOPT を妨げる（緩めない）。

## 実行手順（機体ごと）

1. ローカルで `git archive` を 3 本（runner・before・after）作り転送・別ディレクトリに展開し、各木に `.rev-stamp`（1 行目 sha・2 行目 `dirty=0`）を置く。
2. `build_ab_arms.sh <before木> <after木> <bin>` を hw-lock の前に実行する。
3. `session-scripts/session-3672.sh <gb10|macmini> <label>` を切り離して起動する（GB10 は `setsid nohup`、Mac mini は `nohup` サブシェル）。
4. 回収・マスク・`ab_judge.py --host gb10=… --host macmini=…` で最終判定。

## 状態

2026-10-28 に GB10 専有（`<cuda-node>`）と Mac mini（`<metal-node>`）で各 1 回（ラベル `s1`）実測済み。
**判定は REJECT**（`judge-3672.md`）。処置 2 セルは ratio 0.80（bench N=4096）・0.82（診断 resident_public N=4096）で 5/5 run < 1.00、
rho_REM 1.28、checksum 全件一致だが、RULE の「5/5 run で後退したセルがある」に非処置 3 セルが当たった
（GB10 診断 resident_split N=512、Mac mini 診断 resident_public N=256 と gemm_device_download N=4096）。
詳細・保守性の判断・限界は `docs/perf/cpu-gemm-reuse-fresh-inversion.md` §11。

実行の経緯: 起動のやり直しなし。計測後に `ab_judge.py` を 1 点修正した（判定に使わない区間行の `median_s` が 0.0 でも入力不備にしない。
判定式・閾値は不変。`rev-stamp-verification-3672.md`）。`RULE.v2.txt` の `fixed-at-utc`（13:30Z）は記入時の誤りで、実際のコミットは 14:00Z 頃（同 md で訂正）。
`orchestrate.sh` の最終行の案内文は絶対パスを含むため、回収時にマスクした。
