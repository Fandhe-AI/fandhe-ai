# SME 既定 ON の新しい事前登録 A/B を Mac mini と GB10 で実測する（イシュー #4156）

## 状態

**事前登録・未実測**。計測部品（A/B 本体・オーケストレーション・厳密集計）と事前登録規則 `RULE.txt` を固定した段階で、実機の値は一つもない。

**計測はまだ始められない**。`RULE.txt` §9（所有者決定の転記欄）が `<未決>` で、P-OWNER が不成立になるため。根拠:

- #4143 は 2026-11-10 に所有者が承認済み（案 (c) GB10-GATE-ONLY-STRUCTURAL・UNDETERMINED-HANDLING = return-to-owner・#4142 依存の了承）: https://github.com/Fandhe-AI/fandhe-ai/issues/4143#issuecomment-6757498148 。記録は `docs/perf/cpu-gemm-sme-fmopa-microkernel.md` §5.16、`docs/cpu-gemm-sme-unsafe-audit.md` §12.14。
- #4142 も 2026-11-10 に所有者が承認済み（案 A: NB-RANGE5・NB-SEP を採用）: https://github.com/Fandhe-AI/fandhe-ai/issues/4142#issuecomment-6757497844 。記録は `docs/perf/noise-band-decision.md`。
- ただし `RULE.txt` §9 は事前登録時点で `<未決>` のまま固定している。決定を推定・代筆せず、§9 への転記は実測者が測る前に行う（下記）。

#4142・#4143 は承認済み。実測者は上記の回答コメント URL とともに `RULE.txt` §9 へ列挙値を転記し（計測前に限る追記）、`prereg_append.txt` を作ってから実機で測る。

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
| `orchestrate_macmini.sh`・`orchestrate_gb10.sh` | P-OWNER → ロック →（GB10 は外側ゲート）→ P-TREE → P-R0 → P-RT →（Mac mini は mediaanalysisd の確認）→ A/B → P-BIT → マスク収録（`--dry-run` あり） |
| `test_foreign_gate_4156.sh` | Mac mini 腕の他ジョブの前提ゲートと mediaanalysisd の確認の self-test（実機に接続しない。`bash test_foreign_gate_4156.sh`） |
| `aggregate.py` | 厳密集計（標準ライブラリのみ。`python3 -I aggregate.py --self-test`） |
| `prereg_append.txt`・`macmini/`・`gb10/`・`aggregate-*.md` | 実測後に収録（現時点では無い）。`aggregate-*.md` は機械出力で手で直さない |
| `../../../../scripts/bench/framework-compare/run_ab_sme_default_on_4156_cpu.sh` | 2 機共通の A/B 本体。#3232 の `run_ab_sme_default_on_cpu.sh` の派生（元は凍結して変更しない） |

腕は before = OFF（main のクリーンな clone・定数 false）、after = ON（その clone に `on-arm.patch` を当てた計測専用ツリー）。
#3232 と逆の組み立てで、比 = ON / OFF（> 1.00 なら後退）の向きは変わらない。

## 手順（実機セッション。Linux の自動運転では実行しない）

1. 所有者の回答を `RULE.txt` §9 へ転記してコミット・push し、追記後の `RULE.txt` の sha256 と UTC 時刻を `prereg_append.txt`（`rule_sha256=` と `prepared_at=`）に記録する。時刻は hwlock の取得時刻より前でなければならない。`prereg_append.txt` も同じ要領でコミット・push する（未追跡のまま残すと A/B 本体の `verify_no_untracked_build_inputs` が BASE 内の未追跡ファイルを検出して停止する）。`prepared_at` は `YYYY-MM-DDTHH:MM:SSZ` 形式で、形式不正・未来時刻ならロック取得前に停止し、取得直後にも `prepared_at < acquired` を再確認する。
2. 開発機で `git bundle` を作って実機へ転送し、クリーンな clone を BASE とする。登録基準 SHA と BASE の間で `crates/`・`scripts/bench/`・`Cargo.*`・`rust-toolchain.toml` に差分がないことを確認する（#3613 §8 の教訓）。
3. `RAYON_NUM_THREADS` を unset し、他の cargo・bench・python が動いていないことを確かめる。計測中はビルドを並走させない。
4. GB10 を先に: `BASE=<clone> WORK=<作業> [TORCH_PY=<python>] bash docs/perf/logs/sme-default-on-ab-4156/orchestrate_gb10.sh`（`PATH` に `${HOME}/.cargo/bin`。専有できる機体を選ぶ。compute プロセスが常駐する機体では外側ゲートを通らない）。
5. Mac mini: `BASE=<clone> WORK=<作業> TORCH_PY=<venv python> FOREIGN_GATE=<foreign_job_gate.py の絶対パス> bash .../orchestrate_macmini.sh`。`WORK` は名前が `.noindex` で終わるディレクトリにする（Spotlight の索引を避ける。#4151 の RULE.v2.txt 項 10 (e)）。hw-lock は先に取らない（スクリプトが取る）。
6. 回収後 `python3 -I aggregate.py gb10 gb10 > aggregate-gb10.md`、続けて `macmini macmini > aggregate-macmini.md`。perf doc §5.17 に結果を転記する。

## 他ジョブの前提ゲートと mediaanalysisd の確認（Mac mini。2026-11-11 追加）

`RULE.txt` の固定後に足した。ゲートを厳しくする方向だけの追加で、判定式・閾値・round 数・ほかの段の手順は変えていない。
この版で測る系列は #4156 の RULE.v2 で登録する（GB10 セッションが書く）。GB10（`AB_HOST=gb10`）の経路は変えていない。

- **他ジョブの前提ゲート**: 部品は #4160 と同じ `scripts/bench/framework-compare/ab_foreign_gate.sh` で、判定は `docs/perf/logs/macmini-foreign-job-gate/foreign_job_gate.py`（#4227。G1〜G4）。どちらも変えていない。
  - A/B 本体の各 round の前に、他ジョブが CLEAN かつ load1 < 3.0 になるまで、同じループで待つ（30 秒間隔・最大 60 回。load1 の待ちと同じ上限）。
  - round の後（PyTorch の同席計測まで終えた直後）に 1 回確かめる。
  - 通過の行は従来どおり `round<N> gate=pass gate_ok=1 ...` の 1 行。通過以外は行末に `reason=foreign-job|load1|foreign-gate-error|load1-unavailable` を足す。
  - 終了後の FOREIGN・ゲート失敗は、同じ round の `gate=invalid gate_ok=0 ... reason=foreign-after|foreign-gate-error phase=after` を足す。
  - 開始前の不通過は `aggregate.py` の P-GATE で正式系列から外れる。
  - 終了後の FOREIGN は P-GATE では数えない（P-GATE は pass の行だけを読む）。代わりに A/B 本体が非 0 で終わり、`run_ab exit=0` が欠けて P-RUNAB で正式系列から外れる。`aggregate.py` は変えていない。
  - ゲートの出力は A/B 本体が `results/raw/foreign-gate/4156-sme-cpu-<LABEL>/` に残し、`orchestrate_macmini.sh` が `macmini/foreign-gate/` へ収録する。ゲートの sha256 は `macmini/foreign_gate.txt`。
- **mediaanalysisd の確認**: #4146 の `RULE.v2.txt` 項 8 (c) と同じ基準。A/B の直前に 2 分間隔で 3 回、%CPU が 5 未満であることを確かめる。
  - 名前は basename の完全一致で見る。複数あれば最大値で判定し、プロセスがなければ通過。`ps` の失敗・出力が空・非数値は不成立。
  - 不成立なら A/B を始めずに止める。記録は `${WORK}/macmini/mediaanalysisd.txt` に残る。
- **起動前に止まる場合**: `FOREIGN_GATE` が未指定・相対パス・存在しない・禁止文字を含むときは、ロックを取る前に止まる。

## マスク規則・実機ロック

- ホスト名は `masked`（公開文書では `<metal-node>`・`<cuda-node>`）、`$HOME` は `<home>`、ユーザー名・絶対パス・IP は除去する。オーケストレーションは収録前に grep で漏れを検査し、0 件でなければ収録しない。
- 実機ロック `~/work/.fandhe-hw-lock` を取得してから測り、取得・解放時刻を `hwlock.log` に残す。
- 漏れ検査のユーザー名の項は語境界つき（#4156 所有者判断 2026-11-11）。語の文字は英数字・下線・ハイフンで、前後がそのどれでもないときだけ当たる。ほかの項（`/Users/`・`/home/`・ホスト名・IPv4）は変えていない。正規表現は両オーケストレーションの `leak_re` に置いた。`bash orchestrate_gb10.sh --self-test-leak`（Mac 版も同じ）で、植えた値が変更後も検出されることを確かめられる。植えた値はユーザー名・ホーム・ホスト名・IP・絶対パスで、合成値と実行環境の実値の 2 組を使う。
- 記録のみ（直さない）: `orchestrate_gb10.sh` はロックの owner を `#4156 <UTC>` の 2 語で書く（ロック取得の直後の行）。他の実機スクリプトの雛形は 3 語の形を要求するため、同じロックを見る別のスクリプトがこの owner を読むと形式の違いが出る。本 A/B の判定には関係しないので、所有者判断により変えない。
- Mac mini の値と M4 Max（開発機）の値を混ぜない。

## 検証状況

- Linux 上で検証済み: `aggregate.py --self-test`、`bash -n`、`--dry-run`、`git apply --check`、パッチ sha256。
- 漏れ検査の self-test（`--self-test-leak`）は次の 2 環境で通った: 開発機（macOS の grep・bash 3.2）、`<cuda-node>`（GNU grep）。<cuda-node> では、固定前の試走の収録物 43 ファイルに当て、誤検出が 0 件になることも確かめた。
- 変異の確認: 次の 2 つの変異のどちらでも self-test が落ちることを確かめた。
  - ユーザー名の項を外す
  - 境界を外す、または境界からハイフンを外す
- 他ジョブの前提ゲートと mediaanalysisd の確認（2026-11-11）は、開発機で `bash test_foreign_gate_4156.sh` を通した。実機には接続していない。
  - 確かめた場合: 全 CLEAN・終了後 FOREIGN・開始前 60 回 FOREIGN・load1 超過・ゲート失敗（前・後）・出力不一致・load1 が読めない・緩和閾値の系列・GB10 の行が従来どおり。
  - 変更なしの `aggregate.py` の判定: 全 CLEAN は ADOPT、開始前の不通過は undetermined、終了後の FOREIGN は `run_ab exit=1` で undetermined。
  - mediaanalysisd の確認: 境界 5.0・部分一致の名前・複数・非数値・`ps` の失敗。
  - orchestrate: `FOREIGN_GATE` の不正でロック前に止まること・呼び出し順・dry-run。
  - 変異の確認: 次の 5 つの変異のどれでも self-test が落ちることを確かめた。
    - round 後の呼び出しを外す
    - 開始前の GATE_NOT_OK を数えない
    - 終了後の GATE_NOT_OK を数えない
    - mediaanalysisd を 5 以下で通す
    - mediaanalysisd を部分一致で拾う
  - <metal-node> で `foreign_job_gate.py` を 1 回だけ動かした。出力 10 行は漏れ検査（`leak_re`）に 0 件で当たった（計測ではない）。
- **未検証**: 実機での A/B・P-RT・P-BIT・マスク収録（Mac mini と GB10 が必要）。

## スコープ外

- 既定の切り替え（#4165）。SME の unsafe 独立監査（#4157）。N=256 のスレッド分配（#4162）。6 フレームワークでの 1 位の確定（Phase 6）。
