# SME 既定 ON の新しい事前登録 A/B を Mac mini と GB10 で実測する（イシュー #4156）

## 状態

**実測済み・総合判定 ADOPT**（2026-11-13 集計。下の「結果」）。GB10 腕は `gb10/`、Mac mini 腕は `macmini-formal-1/`（1 系列目で全ゲートを通過）。判定に使った規則は `RULE.v2.txt`（v2）。既定の切り替えは本イシューの範囲外で、定数は `false` のまま（#4165 の入力として記録する）。v1 の `RULE.txt` は経緯として残し、道具（両オーケストレーション・`aggregate.py`）は読まない。

- v2 で変えた点と理由は `RULE.v2.txt` の冒頭に書いた。判定式・閾値・5 round・5/5 一貫性・checksum 完全一致・処置は v1 のまま。
  - 登録基準を `d551b31` から `18938d84c`（#4258 のマージコミット）へ移した。v1 の固定後に道具の PR（#4256・#4258）が入ったため。
  - §9 に所有者決定を転記した（GB10-GATE-ONLY-STRUCTURAL・NOISE-BAND = adopted・UNDETERMINED-HANDLING = return-to-owner）。根拠は #4143 の https://github.com/Fandhe-AI/fandhe-ai/issues/4143#issuecomment-6757498148 と #4142 の https://github.com/Fandhe-AI/fandhe-ai/issues/4142#issuecomment-6757497844 。記録は `docs/perf/cpu-gemm-sme-fmopa-microkernel.md` §5.16、`docs/cpu-gemm-sme-unsafe-audit.md` §12.14、`docs/perf/noise-band-decision.md`。
  - §11 に道具の sha256、§12 に Mac mini 腕の準備、§13 に露出の申告を足した。
  - 計測対象（target）を `319a276b0`（#4263 のマージコミット）と明記した。`18938d84c` との差分は #4263 の `crates/backend-cuda` だけ（テスト 3 件・テスト専用の構築関数・`CudaDevice::new` の挙動を変えない切り出し）で、`crates/backend-cpu`・`scripts/bench/` は変わっていない。§1 の差分検査はこの版で行う。
- `prereg_append.txt`（`RULE.v2.txt` の sha256 と準備時刻）も v2 の PR でコミットした。
- `RULE.v2.txt` §12 (c)（Mac mini 腕で他ジョブのゲートに当たったときの系列数の上限）は **3 系列**（所有者判断 2026-11-12。#4166 と同じ形で、`formal-1`〜`formal-3`）。機械が読む行 `MAC-SERIES-CAP: 3` を置き、両オーケストレーション（ロック取得の前）と `aggregate.py` が P-CAP として検査する（空欄・不正値・欠落・重複は不成立）。
- 計測は v2 の PR のマージ後に始める。`RULE.v2.txt` を変えると sha256 が変わるため、`prereg_append.txt` を取り直してから両機体を測る（`aggregate.py` は両機体の `prereg_append.txt` を現在の `RULE.v2.txt` と照合する）。

## 目的

2026-10-28 の所有者判断（perf doc §5.15）は「SME 既定は OFF のまま。ON に戻すのは新しい事前登録 A/B に合格したときだけ」。
#3232 の A/B では Mac mini の到達 10 セルがすべて ON 優位だったが、SME 非搭載の GB10 の 2 セルが 5/5 で後退して総合 REJECT になり、
#3613 の 3 腕 A/B でその 2 セルはノイズ帯（非再現）だった。本 A/B は GB10 の数え方と僅差セルのノイズ帯を先に決めたうえで測り直し、
ADOPT／REJECT を記録する。**既定の切り替え（定数を true にすること）は本イシューの範囲外**（Phase 4 の #4165 と所有者承認）。

## 収録物

| パス | 内容 |
|---|---|
| `RULE.v2.txt` | 事前登録規則 v2（判定に使う規則。計測前に単独 PR でコミットして固定。訂正は末尾への追記のみ。v1 から変えた点は冒頭） |
| `RULE.txt` | 事前登録規則 v1（経緯として残す。§9 は `<未決>` のまま。道具は読まない） |
| `prereg_append.txt` | `RULE.v2.txt` の sha256（`rule_sha256=`）と準備時刻（`prepared_at=`。両機体の hwlock 取得より前）。v2 の PR でコミット |
| `on-arm.patch` | `SME_PRODUCTION_ENABLED` を `false` から `true` にする 1 行（#3613 の同名パッチと同一内容。sha256 は `RULE.v2.txt` に固定。v1 と同じ値） |
| `orchestrate_macmini.sh`・`orchestrate_gb10.sh` | P-OWNER → P-CAP → ロック →（GB10 は外側ゲート）→ P-TREE → P-R0 → P-RT →（Mac mini は mediaanalysisd の確認）→ A/B → P-BIT → マスク収録（`--dry-run`・`--self-test-leak`・`--self-test-cap` あり） |
| `test_foreign_gate_4156.sh` | Mac mini 腕の他ジョブの前提ゲートと mediaanalysisd の確認の self-test（実機に接続しない。`bash test_foreign_gate_4156.sh`） |
| `aggregate.py` | 厳密集計（標準ライブラリのみ。`python3 -I aggregate.py --self-test`） |
| `gb10/` | GB10 腕の収録物（`orchestrate_gb10.sh` のマスク収録の写し） |
| `macmini-formal-1/` | Mac mini 腕の系列 1（`formal-1`）の収録物（`orchestrate_macmini.sh` が BASE の下の `macmini/` へ収録したものの写し）。`kit/` は系列の外の準備の記録（道具のピンの照合・dry・torch の venv・起動の終了行）。系列の分け方は下の「Mac mini 腕」 |
| `aggregate-gb10.md`・`aggregate-macmini-formal-1.md` | `python3 -I aggregate.py gb10 gb10`・`python3 -I aggregate.py macmini-formal-1 macmini` の機械出力（手で直さない）。Mac mini 側の総合判定は `gb10/` を合わせた両機体の総合 |
| `../../../../scripts/bench/framework-compare/run_ab_sme_default_on_4156_cpu.sh` | 2 機共通の A/B 本体。#3232 の `run_ab_sme_default_on_cpu.sh` の派生（元は凍結して変更しない） |

腕は before = OFF（main のクリーンな clone・定数 false）、after = ON（その clone に `on-arm.patch` を当てた計測専用ツリー）。
#3232 と逆の組み立てで、比 = ON / OFF（> 1.00 なら後退）の向きは変わらない。

## 手順（実機セッション。Linux の自動運転では実行しない）

1. `RULE.v2.txt` §9 の転記・§12 (c) の記入と `prereg_append.txt` は v2 の PR でコミット済み。実測者は転記しない。§13 へ追記したときだけ、追記後の `RULE.v2.txt` の sha256 と UTC 時刻で `prereg_append.txt` を取り直してコミット・push する（未追跡のまま残すと A/B 本体の `verify_no_untracked_build_inputs` が BASE 内の未追跡ファイルを検出して停止する）。`prepared_at` は `YYYY-MM-DDTHH:MM:SSZ` 形式で、hwlock の取得時刻より前。形式不正・未来時刻ならロック取得前に停止し、取得直後にも `prepared_at < acquired` を再確認する。
2. 開発機で `git bundle` を作って実機へ転送し、クリーンな clone を BASE とする。BASE は `RULE.v2.txt` と `prereg_append.txt` を含む main のコミット。計測対象 `319a276b0`（#4263 のマージコミット。`RULE.v2.txt` 冒頭の「計測対象 (target)」）と BASE の間で `crates/`・`scripts/bench/`・`Cargo.*`・`rust-toolchain.toml` に差分がないこと（#3613 §8 の教訓）と、`RULE.v2.txt` §11 のピンが BASE で一致することを、送る前に確かめて記録する。実機ではツリー指紋（送信側と実機側）を突き合わせる。
3. `RAYON_NUM_THREADS` を unset し、他の cargo・bench・python が動いていないことを確かめる。計測中はビルドを並走させない。
4. GB10 を先に: `BASE=<clone> WORK=<作業> [TORCH_PY=<python>] bash docs/perf/logs/sme-default-on-ab-4156/orchestrate_gb10.sh`（`PATH` に `${HOME}/.cargo/bin`。専有できる機体を選ぶ。compute プロセスが常駐する機体では外側ゲートを通らない）。
5. Mac mini: `BASE=<clone> WORK=<作業> TORCH_PY=<venv python> FOREIGN_GATE=<foreign_job_gate.py の絶対パス> bash .../orchestrate_macmini.sh`。`WORK` は名前が `.noindex` で終わるディレクトリにする（Spotlight の索引を避ける。#4151 の RULE.v2.txt 項 10 (e)）。hw-lock は先に取らない（スクリプトが取る）。
6. 回収後 `python3 -I aggregate.py gb10 gb10 > aggregate-gb10.md`、続けて `macmini macmini > aggregate-macmini.md`。perf doc §5.17 に結果を転記する。

## 他ジョブの前提ゲートと mediaanalysisd の確認（Mac mini。2026-11-11 追加）

v1 の `RULE.txt` の固定後に足した。ゲートを厳しくする方向だけの追加で、判定式・閾値・round 数・ほかの段の手順は変えていない。
`RULE.v2.txt` §4（P-GATE）に登録した。GB10（`AB_HOST=gb10`）の経路は変えていない。
Mac mini 腕の準備（`__pycache__` の扱い・系列の前に誰がプロセスを止めるか・系列数の上限）は `RULE.v2.txt` §12。

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
- P-CAP（2026-11-12）は開発機（bash 3.2）で次を通した。実機には接続していない。
  - 両オーケストレーションの `--self-test-cap`（14 件）: `3`・`1`・`12` は成立。空欄記号・`0`・`-1`・`3.5`・非数値・`03`・空・`+3`・欠落・重複・別の行に空欄記号が残る場合は不成立。
  - `aggregate.py --self-test`: `3` は両機体とも正式、上の不正値はどれも両機体で undetermined、空欄でも checksum 不一致の FAIL は保つ。
  - `test_foreign_gate_4156.sh`: 両オーケストレーションが、上限を空欄に戻した写しではロックを取る前に P-CAP で止まり、`3` では P-CAP を通って次の検査へ進む。呼び出しが P-OWNER の後でロック取得の前にあること。
  - 変異の確認: 次の 6 つの変異のどれでも、いずれかの self-test が落ちることを確かめた。
    - Mac mini 側の P-CAP の呼び出しを外す
    - GB10 側の P-CAP の呼び出しを外す
    - `check_series_cap` を常に成立にする
    - `aggregate.py` で P-CAP の理由を捨てる
    - `aggregate.py` で空欄記号を見ない
    - `check_series_cap` で空欄記号を見ない（Mac mini 側）
- v2 の固定前に、`<cuda-node>` で v2 の道具の捨て試走を 1 回流した（2026-11-11）。全段が exit 0 で、マスク収録まで通った。見たのは終了コードの行だけで、出力は開かずに消した（`RULE.v2.txt` §13）。
- 正式系列（2026-11-12）: 両機体で実機の A/B・P-RT・P-BIT・マスク収録が通り、マスク後の収録物に `aggregate.py` を流した（下の「結果」「実行の記録」）。

## 結果

**総合判定: ADOPT**（`aggregate-macmini-formal-1.md` の総合。評価順 FAIL > undetermined（前提ゲート不成立）> REJECT > ADOPT）。

- 所有者決定（§9）: GB10-MODE = GB10-GATE-ONLY-STRUCTURAL・NOISE-BAND = adopted・UNDETERMINED-HANDLING = return-to-owner。
- 前提ゲート: 両機体とも「すべて成立」（`aggregate-*.md` の「前提ゲート」節）。checksum 不一致（FAIL）は両機体ともない。
- Mac mini（`<metal-node>`。系列 `formal-1`）: 到達 10 セル（gemm 256〜2048 × fresh/reuse・train 64 × fresh/reuse）はすべて 5/5 で ON/OFF ≤ 1.00。ON/OFF 中央値比は gemm 0.463〜0.760・train 0.880〜0.913。ガード（非到達）の infer 64 は 0.997（fresh）・0.959（reuse）で、5/5 一貫の後退・分離した後退はない。Mac mini 側の判定は ADOPT。
- GB10（`<cuda-node>`）: 12 セルすべてガード（非到達）。5/5 一貫の後退・分離した後退はない。拘束する方式 GB10-GATE-ONLY-STRUCTURAL は pass（参考の GB10-RECORD・GB10-NBSEP も pass）。
- P-BIT: 両機体とも ON/OFF の `^out[` 行が 6726847 行で sha256 一致・`cmp` 差分 0。両機体の sha256 も同じ値（`b8c399e9…a694af`）。
- 1 位化の記録（§6。判定には使わない・PyTorch だけとの比較）: Mac mini の ON 腕は gemm 512/1024/2048 × fresh/reuse の 6 セルで PyTorch 比 1.078〜1.568、NB-SEP 成立（「1 位（確定）」。OFF 腕は 0.516〜0.775）。GB10 は 6 セルとも NB-SEP 不成立で「改善傾向・確定せず」。6 フレームワークでの 1 位化は Phase 6。
- 処置（§8）: 本イシューでは定数を変えない（`SME_PRODUCTION_ENABLED` は `false` のまま）。ADOPT を Phase 4（#4165）の入力として記録する。既定 ON の前提として SME の unsafe 独立監査（#4157）を併記する。
- #3232 との違い: #3232 は GB10 の 2 セルの 5/5 後退で総合 REJECT だった。本 A/B では GB10 の数え方（#4143）を先に決めて測り、GB10 に 5/5 一貫の後退は出なかった。#3635 以降 `reuse` は常駐入力経路のため、#3232 の値と並べて比べない。

## 実行の記録

### GB10 腕（`<cuda-node>`）

- 計測対象（BASE）: `85498f32be4bb8ea70302bd85ec2b8b4329d862f`（#4264 のマージコミット。`gb10/rev-stamp.txt`）。`git bundle` で送ってクリーンな clone を BASE とした。ノード上で GitHub への clone・fetch はしていない。
  - 収録した `gb10/tree_fingerprint.txt`（`orchestrate_gb10.sh` が BASE の `git ls-files` から計算）は、開発機で `85498f32b` の追跡ファイル一覧から同じ方法（`LC_ALL=C` の sort）で計算した値と一致した（2026-11-13）。
  - `gb10/tree_verify.txt`（ON 腕の差分ファイル一覧）は `crates/backend-cpu/src/gemm_blis/mod.rs` の 1 件。
- 系列: 2026-11-12T04:57:01Z〜05:01:51Z（`run_ab exit=0`）。
- 実機ロック: `orchestrate_gb10.sh` が 04:57:01Z に取得し、05:01:51Z に解放した。
  - 収録した `gb10/hwlock.log` は、収録先へ複製した時点の写しなので、最後の行は `released-pending at=2026-11-12T05:01:51Z`。
  - 解放の行（`released at=2026-11-12T05:01:51Z`）は、ノードの作業ディレクトリ側の `hwlock.log` にだけある。
- 収録前の漏れ検査（`orchestrate_gb10.sh` の `leak_re`）: 0 件（`collect exit=0`）。
- push 前の再検査（2026-11-13・開発機）:
  - 回収した 43 ファイルに、実機の実値で組んだ `leak_re` を当てた。0 件。
  - 開発機のユーザー名・ホスト名も 0 件。
  - 43 ファイルの sha256 は、ノード上の収録先の複製とすべて一致した。
- 道具（`RULE.v2.txt` §11 で固定した 8 本）は変えていない。
- 値を見ていないこと:
  - 系列の途中でも、Mac mini 腕の系列が終わるまでの間も、収録物の値は開いていない。
  - 見たのは次だけ: ファイル名・空ファイルの数・各ファイルの sha256・`hwlock.log` の時刻・終了行（`run_ab exit=`・`collect exit=`）・漏れ検査の件数と `<home>` を含む行の数・`rev-stamp.txt` の SHA・`tree_fingerprint.txt` のハッシュ（照合のため）・`tree_verify.txt` のファイル名。
  - GB10 腕のセッションは `aggregate.py` を流していない。
- ノードの作業ディレクトリ `<home>/work/sme-4156-gb10` は、記録 PR のマージまで残す。

### Mac mini 腕（`<metal-node>`）

- 計測対象（BASE）: `85498f32be4bb8ea70302bd85ec2b8b4329d862f`（#4264 のマージコミット。`macmini-formal-1/rev-stamp.txt`）。
  - 開発機で `85498f32b` から `git clone --depth 1` したクリーンな clone（origin を外したもの）を rsync で送り、BASE とした。README の手順 2 の `git bundle` ではないが、BASE がクリーンな clone であることは同じ。ノード上で GitHub への clone・fetch はしていない。shallow clone から ON 腕の複製・`on-arm.patch` の適用・差分 1 件の確認が通ることは、送る前に開発機で試した。
  - 計測対象 `319a276b0` は `85498f32b` の祖先で、間の `crates/`・`scripts/bench/`・`Cargo.*`・`rust-toolchain.toml` の差分は 0 件（開発機で確認）。
  - 収録した `macmini-formal-1/tree_fingerprint.txt` は、開発機で `85498f32b` の追跡ファイル一覧から同じ方法（`LC_ALL=C` の sort）で計算した値と一致した（`557ae30e…643da9`。GB10 腕とも同じ値）。`tree_verify.txt` は `crates/backend-cpu/src/gemm_blis/mod.rs` の 1 件。
  - `RULE.v2.txt` §11 の 8 本のピンは、開発機の clone とノード上の BASE の両方で一致した（`macmini-formal-1/kit/pin-check.txt`）。ツリーの外の kit に置いた `foreign_job_gate.py` も一致した（`kit/kit-pin-check.txt`）。BASE の未コミット変更は 0 件（`kit/base-status-lines.txt`）。
- 系列の分け方（所有者判断 2026-11-12。#4156 のコメント https://github.com/Fandhe-AI/fandhe-ai/issues/4156#issuecomment-6820151024 の A）: 系列ごとに空の新しい WORK（`<home>/work/w4156-formal-k.noindex`）と、`85498f32b` から作り直した BASE（`<home>/work/b4156-formal-k.noindex`）を使い、収録は `macmini-formal-k/`。`formal-1` で全ゲートを通ったため、`formal-2`・`formal-3` は回していない（§12 (c) の上限 3 のうち 1 系列）。
- 1 系列目の前の dry（2026-11-12T09:50:30Z）: hw-lock を手で取り（owner `#4156 dry <UTC>`）、`orchestrate_macmini.sh --dry-run` を流した。rc=0、P-OWNER・P-CAP とも成立、WORK は作られていない（`kit/dry-1.txt`）。dry は計画の表示と P-OWNER・P-CAP だけで、収録と集計の分け方はコードを読んで確かめた（収録先 `${BASE}/…/macmini` は BASE を作り直すため系列ごとに新しい、`aggregate.py` の subdir は任意）。
- torch の venv（所有者判断 2026-11-12）: `<home>/work/k4156.noindex/venv-torch`（WORK・BASE の外）。uv の Python 3.12.15 で作り、`pip install torch==2.14.0 numpy==2.5.4` だけを入れた（2026-11-12T09:59:15Z〜09:59:25Z。hw-lock なし・系列の前）。版は torch 2.14.0・numpy 2.5.4 で RULE.v2 のとおり（`kit/venv-python.txt`・`kit/venv-versions.txt`・`kit/venv-pip-freeze.txt`・`kit/pip.log`）。リポジトリの依存は変えていない。
- 系列 `formal-1`: 2026-11-12T10:42:13Z〜10:54:44Z（`kit/formal-1.status` は `end rc=0`）。所有者が Mac mini の準備を終えた連絡を受けてから起動した。止めるよう名指しされたプロセスはなく（§12 (b)）、計測セッションからは何も止めていない。
- 実機ロック: `orchestrate_macmini.sh` が 10:42:13Z に取得し、10:54:44Z に解放した。収録した `macmini-formal-1/hwlock.log` は GB10 腕と同じく収録先へ複製した時点の写しなので、最後の行は `released-pending at=2026-11-12T10:54:44Z`。解放の行（`released at=2026-11-12T10:54:44Z`）はノードの WORK 側の `hwlock.log` にだけある。`aggregate.py` はこの形で止まらなかった。
- 判定行（`macmini-formal-1/load_gate.log`）: `rt_on exit=0 skip=sme_production_disabled_after_gb10_reject`・`rt_off exit=0`・`mediaanalysisd exit=0`・`run_ab exit=0`・`bitdump exit=0 cmp_exit=0`・`collect exit=0`。オーケストレーションの標準出力は `P-TREE ok`・`P-R0 ok`・`完了`（`kit/formal-1.out`）。
- 他ジョブの前提ゲート: 5 round とも、開始前のゲートが CLEAN かつ load1 < 3.0 で `gate=pass gate_ok=1`（待ちは round 1 から順に 90・0・30・0・30 秒）、終了後の確認も 5 round とも CLEAN（`macmini-formal-1/foreign-gate/`。ゲートの sha256 は `foreign_gate.txt` でピンと一致）。hw-lock を取らずに GPU を使う別のジョブは見ていない。
- 集計の前の揃い方の確認（系列が 12 分半と短かったため。値は開かずに確認）: round 1〜5 の判定行が揃っている、結果の jsonl の行数（gemm 40・train 10・infer 10 を両腕、torch 30）が GB10 腕と同じ、ファイルの組み合わせが GB10 腕と同じ（Mac mini だけの `foreign-gate/`・`foreign_gate.txt`・`mediaanalysisd.txt` と GB10 だけの `load_gate_outer.log` を除く）、`torch_failed_total=0`、skip の記録が空。
- 漏れ検査: 収録前の `leak_re` は 0 件（`collect exit=0`）。回収後に開発機で `<home>` などへの置換をかけ直し（3 ファイル。kit の照合結果・dry・pip.log の絶対パス）、ノードのホスト名（2 形）・ユーザー名・ホームのパス・`/Users/`・IPv4・開発機のユーザー名を全 75 ファイルに当てて 0 件。
- `__pycache__` は消していない（§12 (a)）。`aggregate.py` は `-B` と `PYTHONDONTWRITEBYTECODE=1` で流し、`__pycache__` を作っていない。
- 露出の申告（§13 の Mac mini 分）: #4156 のコメント https://github.com/Fandhe-AI/fandhe-ai/issues/4156#issuecomment-6820151024 の B（1 系列目より前に統合セッションが投稿）。系列の途中は、プロセスの生存・hw-lock の owner・終了の行だけを見た。値・集計は、全系列が終わり GB10 の記録（`test/4156-record`）を受け取った後の集計で初めて見た。GB10 の記録もその時点まで開いていない。
- ノードの `<home>/work/b4156-formal-1.noindex`（BASE）・`w4156-formal-1.noindex`（WORK）・`k4156.noindex`（kit）は、記録 PR のマージまで残す。

## スコープ外

- 既定の切り替え（#4165）。SME の unsafe 独立監査（#4157）。N=256 のスレッド分配（#4162）。6 フレームワークでの 1 位の確定（Phase 6）。
