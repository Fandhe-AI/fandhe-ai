# SME 既定 ON の新しい事前登録 A/B を Mac mini と GB10 で実測する（イシュー #4156）

## 状態

**事前登録・未実測**。計測部品（A/B 本体・オーケストレーション・厳密集計）と事前登録規則 `RULE.txt` を固定した段階で、実機の値は一つもない。

**計測はまだ始められない**。`RULE.txt` §9（所有者決定の転記欄）が `<未決>` で、P-OWNER が不成立になるため。根拠:

- #4143 の記録（`docs/perf/cpu-gemm-sme-fmopa-microkernel.md` §5.16、`docs/cpu-gemm-sme-unsafe-audit.md` §12.14）は「案・所有者回答待ち／承認欄 未取得」。
- #4142 の記録（`docs/perf/noise-band-decision.md`）も「案・所有者承認待ち／所有者承認 未取得」。
- 両イシューは回答のないまま CLOSED になっている。決定を推定・代筆しない。

所有者の回答コメントが得られたら、`RULE.txt` §9 へ列挙値と回答コメント URL を転記し（計測前に限る追記）、`prereg_append.txt` を作ってから実機で測る。

## 目的

2026-10-28 の所有者判断（perf doc §5.15）は「SME 既定は OFF のまま。ON に戻すのは新しい事前登録 A/B に合格したときだけ」。
#3232 の A/B では Mac mini の到達 10 セルがすべて ON 優位だったが、SME 非搭載の GB10 の 2 セルが 5/5 で後退して総合 REJECT になり、
#3613 の 3 腕 A/B でその 2 セルはノイズ帯（非再現）だった。本 A/B は GB10 の数え方と僅差セルのノイズ帯を先に決めたうえで測り直し、
ADOPT／REJECT を記録する。**既定の切り替え（定数を true にすること）は本イシューの範囲外**（Phase 4 の #4165 と所有者承認）。

## 収録物

| パス | 内容 |
|---|---|
| `RULE.txt` | 事前登録規則（計測前に単独コミットして固定。訂正は末尾への追記のみ。§9 は所有者決定の転記欄） |
| `on-arm.patch` | `SME_PRODUCTION_ENABLED` を `false` から `true` にする 1 行（#3613 の同名パッチと同一内容。sha256 は `RULE.txt` に固定） |
| `orchestrate_macmini.sh`・`orchestrate_gb10.sh` | P-OWNER → ロック →（GB10 は外側ゲート）→ P-TREE → P-R0 → P-RT → A/B → P-BIT → マスク収録（`--dry-run` あり） |
| `aggregate.py` | 厳密集計（標準ライブラリのみ。`python3 -I aggregate.py --self-test`） |
| `prereg_append.txt`・`macmini/`・`gb10/`・`aggregate-*.md` | 実測後に収録（現時点では無い）。`aggregate-*.md` は機械出力で手で直さない |
| `../../../../scripts/bench/framework-compare/run_ab_sme_default_on_4156_cpu.sh` | 2 機共通の A/B 本体。#3232 の `run_ab_sme_default_on_cpu.sh` の派生（元は凍結して変更しない） |

腕は before = OFF（main のクリーンな clone・定数 false）、after = ON（その clone に `on-arm.patch` を当てた計測専用ツリー）。
#3232 と逆の組み立てで、比 = ON / OFF（> 1.00 なら後退）の向きは変わらない。

## 手順（実機セッション。Linux の自動運転では実行しない）

1. 所有者の回答を `RULE.txt` §9 へ転記してコミット・push し、追記後の `RULE.txt` の sha256 と UTC 時刻を `prereg_append.txt`（`rule_sha256=` と `prepared_at=`）に記録する。時刻は hwlock の取得時刻より前でなければならない。
2. 開発機で `git bundle` を作って実機へ転送し、クリーンな clone を BASE とする。登録基準 SHA と BASE の間で `crates/`・`scripts/bench/`・`Cargo.*`・`rust-toolchain.toml` に差分がないことを確認する（#3613 §8 の教訓）。
3. `RAYON_NUM_THREADS` を unset し、他の cargo・bench・python が動いていないことを確かめる。計測中はビルドを並走させない。
4. GB10 を先に: `BASE=<clone> WORK=<作業> [TORCH_PY=<python>] bash docs/perf/logs/sme-default-on-ab-4156/orchestrate_gb10.sh`（`PATH` に `${HOME}/.cargo/bin`。専有できる機体を選ぶ。compute プロセスが常駐する機体では外側ゲートを通らない）。
5. Mac mini: `BASE=<clone> WORK=<作業> TORCH_PY=<venv python> bash .../orchestrate_macmini.sh`。
6. 回収後 `python3 -I aggregate.py gb10 gb10 > aggregate-gb10.md`、続けて `macmini macmini > aggregate-macmini.md`。perf doc §5.17 に結果を転記する。

## マスク規則・実機ロック

- ホスト名は `masked`（公開文書では `<metal-node>`・`<cuda-node>`）、`$HOME` は `<home>`、ユーザー名・絶対パス・IP は除去する。オーケストレーションは収録前に grep で漏れを検査し、0 件でなければ収録しない。
- 実機ロック `~/work/.fandhe-hw-lock` を取得してから測り、取得・解放時刻を `hwlock.log` に残す。
- Mac mini の値と M4 Max（開発機）の値を混ぜない。

## 検証状況

- Linux 上で検証済み: `aggregate.py --self-test`、`bash -n`、`--dry-run`、`git apply --check`、パッチ sha256。
- **未検証**: 実機での A/B・P-RT・P-BIT・マスク収録（Mac mini と GB10 が必要）。

## スコープ外

- 既定の切り替え（#4165）。SME の unsafe 独立監査（#4157）。N=256 のスレッド分配（#4162）。6 フレームワークでの 1 位の確定（Phase 6）。
