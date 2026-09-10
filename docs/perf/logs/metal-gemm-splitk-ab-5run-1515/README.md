# split-K A/B 5 run 正式確定（イシュー #1515）

`docs/perf/metal-gemm-splitk-ab.md` §10・`docs/backend-metal-splitk-decision.md`
§4 が引き継ぐ「split-K vs classic 経路の A/B を 5 run 完走させ ADOPT/REJECT
を正式確定する」計測のスキャフォールド（Linux 本ラン分。実測は **Apple
M4 Max 実機を持つ Mac セッションで実施**する）。

## 位置づけ（#1475 との違い）

- `docs/perf/logs/metal-gemm-splitk-ab-1475/` の 3 run は、is-optimized-away
  是正前バイナリ（`seed_offset` 不一致・B′ の `should_split_k` 未呼び出し）
  による計測のため、本 5 run は **新規**（run1〜run5）として実施する。
  #1475 の 3 run とは混在させない（旧ログは編集せずそのまま保持する）。
- **専有ゲート（load average の閾値判定）は受け入れ条件にしない**
  （ルート #1509 のユーザー指示）。`gemm_splitk_ab_bench` へ
  `--max-load-avg` を渡さず **record_only** 運用（判定なし・load average
  等を記録するのみ）で実行する。共有負荷下（他プロセス並走を許容）で
  あることを `env_guard_mode=record_only`・`env_guard_load_avg` 行・
  `runN_monitor.log`（計測中の負荷推移）・`runN_procs.txt`（並走プロセス
  watchlist の件数）で記録する。

## 事前登録判定規則

`docs/perf/metal-gemm-splitk-ab.md` §10.2 を正とする（本 README は要約の
み）。腕定義・計測境界・ADOPT/REJECT の定数（対象 9 形状: 中央値 ≥ 1.5 か
つ 5/5 run で run 内比 > 1.0／対照 3 形状: 中央値 ≥ 0.95）は #1475 の事前
登録から**変更しない**。REJECT は共有負荷下でも有効な REJECT として扱う
（専有ゲート不成立を理由に REJECT を undetermined へ格下げしない）。

## Mac セッションでの実行手順

1. リポジトリ最新化（`git pull` 等）・`docs/real-hardware-verification-env.md`
   の実機接続手順に従う。
2. 5 回、別プロセス起動で実行する（1 run = 1 起動。差し替え禁止）:

   ```sh
   cd docs/perf/logs/metal-gemm-splitk-ab-5run-1515
   for i in 1 2 3 4 5; do
     ./orchestrate.sh "$i"
   done
   ```

   各 run は `runN.log`（本体出力）・`runN_monitor.log`（10 秒間隔の負荷
   サンプラー）・`runN_procs.txt`（watchlist プロセス件数）・
   `uptime_before_runN.txt`・`pmset_therm_{before,after}_runN.txt` を生成
   する。中断した場合は `env_info.txt` に経緯を記録し、run を黙って差し
   替えない。`orchestrate.sh` は計測開始前に当該 run 番号の既存成果物を
   検出すると何も書かずに非ゼロ終了する（同番号の同時実行もロックで拒否
   する）ため、中断後の再開は**未実施の run 番号のみ**を指定する。同番号
   を取り直したい場合は成果物を手動で別名へ退避してから再実行する。

3. 集計する:

   ```sh
   python3 aggregate.py --gate-mode=record_only \
     --monitor-logs=run1_monitor.log,run2_monitor.log,run3_monitor.log,run4_monitor.log,run5_monitor.log \
     run1.log run2.log run3.log run4.log run5.log > aggregate.md
   ```

4. `aggregate.md` の内容を `docs/perf/metal-gemm-splitk-ab.md` §10.4 の
   記入欄へ転記する。判定（ADOPT/REJECT/undetermined）は `aggregate.py`
   の機械判定をそのまま採用し、人間側で緩めない。
5. `env_info.txt` の記入欄（機種・OS・rustc・base sha・作業ブランチ・
   run ごとの load1 範囲・中断有無）を埋める。内部ホスト名は書かない。
6. イシュー #1515 へ結果をコメントする。ADOPT なら #1516（本番結線）の
   ブロッカーを解除し、REJECT なら理由を記録し結線しないまま完了とする。

## ファイル構成

| ファイル | 内容 |
|---------|------|
| `orchestrate.sh` | 1 run 分の実行ラッパー（`--dry-run` あり） |
| `aggregate.py` | 集計・判定スクリプト（`--self-test` あり） |
| `env_info.txt` | 機種・OS・実行構成の記入欄（実測未実施のため未記入） |
| `run{1..5}.log` | 各 run の `gemm_splitk_ab_bench` 出力（未生成） |
| `run{1..5}_monitor.log` | 各 run の負荷推移サンプラーログ（未生成） |
| `run{1..5}_procs.txt` | 各 run 開始時の並走プロセス watchlist 件数（未生成） |
| `aggregate.md` | 集計結果（未生成） |

## 注意

- 内部ホスト名・ユーザー名・絶対パスを成果物（ログ・env_info・README）
  に書かない。
- 本番結線（`crates/backend-metal/src/` の変更）は本イシューのスコープ外
  （#1516 へ引き継ぐ）。`orchestrate.sh`・`aggregate.py` はいずれも
  `crates/backend-metal/src/` を変更しない。
