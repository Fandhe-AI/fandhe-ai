# N=256 で SME のスレッド数と分配を変えて A/B する（イシュー #4162）

## 状態

**PR-1（道具立て）時点は実測なし**。機構（既定 OFF）・腕の patch・RULE 雛形・実行キット・集計を入れた段階で、実機の値は一つもない。判定は PR-3 で記録する。

| PR | 内容 | 状態 |
|---|---|---|
| PR-1 | 機構・patch・`RULE.template.txt`・`orchestrate_macmini.sh`・`aggregate.py`・doc | 本 PR（`Refs #4162`。イシューは閉じない） |
| PR-2 | Mac mini のセッションが `RULE.template.txt` を埋めて `RULE.txt` として単独コミット | 未着手 |
| PR-3 | Mac mini での実測と判定の記録（`Closes #4162`） | 未着手 |

## 目的と根拠

Mac mini（Apple M4・4P+6E）の CPU GEMM N=256 は、SME ON でも PyTorch 比 0.888 でまだ約 12% 足りない
（`../sme-default-on-ab-3232/aggregate-macmini.md`）。N=256 の SME 経路はグローバル rayon プール（10 スレッド）と `jobs_per_worker=2` で動く。
`../../cpu-sme-thread-scaling-macmini.md` §2・§3（#3175）では (256,256,128) の SME は T=4 が最大で、T=10 は 0.90 倍に落ちる。
小形状ではワーカー起床・fork-join の固定費と job の粒度が効いている可能性があるため、スレッド数（専用プール）と分配（jpw）を事前登録で測る。

イシュー本文は #4151 の区間別内訳を対象に腕を選ぶとしているが、#4151 の PR-3（内訳の実測）はまだ結果がない。
そのため腕は既存の #3175 実測だけから事前登録した。#4151 の結果が先に main に入った場合は RULE の参考として追記するだけで、腕の組は変えない。

**既定は切り替えない**（Phase 4 の #4165 と所有者承認の範囲）。SME 自体が既定 OFF なので、ADOPT でも本番に効くのは #4156 で SME 既定 ON が ADOPT された場合だけ。

## 収録物

| パス | 内容 |
|---|---|
| `RULE.template.txt` | 事前登録規則の雛形。PR-2 で `<未記入>` を埋めて `RULE.txt` にする |
| `on-arm.patch` | `SME_PRODUCTION_ENABLED` を true にする 1 行（`../sme-default-on-ab-4156/on-arm.patch` と同一。sha256 一致） |
| `arm-{t4,t6,jpw1,jpw4,t4-jpw1}.patch` | `sme_small_sched.rs` の定数行だけを変える（ENABLED=true と THREADS／JPW） |
| `patch_sha256.txt` | 全 patch の sha256 |
| `orchestrate_macmini.sh` | 7 腕の実行キット（`--dry-run` あり。bash 3.2 互換）。各 round を他ジョブの前提ゲートで前後から挟み、系列の前に Safari・mediaanalysisd を確かめる（2026-11-13 追加） |
| `test_foreign_gate_4162.sh` | 他ジョブの前提ゲートと系列の前の確認の self-test（実機に接続しない。`bash test_foreign_gate_4162.sh`） |
| `aggregate.py` | 厳密集計（標準ライブラリのみ。`--self-test`・`--check-template`・`--fingerprint`） |

機構の実装は `crates/backend-cpu/src/gemm_blis/sme_small_sched.rs`。aarch64 版 `dispatch_two_d_dynamic` の SME 分岐の中だけから呼ばれ、NEON・x86・scalar は通らない。
main では `ENABLED=false`・`THREADS=0`・`JPW=0` で、SME 自体も既定 OFF のため本番の挙動は変わらない。

## 腕・セル・判定

| 腕 | THREADS | JPW | 狙い |
|---|---|---|---|
| base | 0（10） | 0（2） | 0.888 の基準（SME ON） |
| aa | 0 | 0 | base と同一バイナリ。ノイズ幅の記録だけ |
| t4 | 4 | 0 | P コア数・#3175 の T=4 最大 |
| t6 | 6 | 0 | 中間 |
| jpw1 | 0 | 1 | 分配の粗化 |
| jpw4 | 0 | 4 | 分配の細分化 |
| t4-jpw1 | 4 | 1 | 組み合わせ |

判定セルは gemm 256 の fresh／reuse。ガードは gemm 512 fresh／reuse と train size=64 fresh／reuse。
候補は判定 2 セルとも 5/5 round で比 ≤ 1.00 かつガードに 5/5 後退なし（checksum・P-BIT 一致）で ADOPT 適格。適格のうち幾何平均が最小のものを選ぶ。
詳細は `RULE.template.txt`。

## 手順（実機セッション。Linux の自動運転では実行しない）

1. PR-2: `RULE.template.txt` の PARAMS（`fixed-at-utc`・`tree_sha`・`tree_fingerprint`・`aggregate.py`／`orchestrate_macmini.sh` の sha256・P-DIAG）を埋め、`RULE.txt` として単独コミットして push する。
2. BASE（クリーンな clone）で `RAYON_NUM_THREADS` を unset し、他の cargo・bench・python が動いていないことを確かめる。Safari を終了しておく（Safari アプリ本体が動いていると系列を始めない）。
3. `BASE=<clone> WORK=<作業> TORCH_PY=<venv python> FOREIGN_GATE=<foreign_job_gate.py の絶対パス> bash docs/perf/logs/sme-n256-sched-ab-4162/orchestrate_macmini.sh`。`FOREIGN_GATE` は `docs/perf/logs/macmini-foreign-job-gate/foreign_job_gate.py` を tree の外の kit に置いたもの。hw-lock は先に取らない（スクリプトが取る）。系列の前の確認が不成立なら `${WORK}/prestart.txt` が残る。所有者へ報告し、再試行は新しい `WORK` で行う。
4. `python3 -I aggregate.py macmini > aggregate-macmini.md`。perf doc §5.18 に結果を転記する。

## 他ジョブの前提ゲートと系列の前の確認（2026-11-13 追加）

PR-1 の後、`RULE.txt` の固定（PR-2）の前に足した。ゲートを厳しくする方向だけの追加で、判定式・閾値・round 数・腕・セル・ほかの段の手順・`aggregate.py`・patch は変えていない。
形は #4156 の Mac mini 腕（PR #4258）と同じ。#4156 は round のループが A/B 本体にあるのでそこで挟んだが、本キットは round のループが `orchestrate_macmini.sh` にあるので、ここで挟む（`scripts/bench/` は変えていない）。

- **他ジョブの前提ゲート**: 部品は #4160・#4156 と同じ `scripts/bench/framework-compare/ab_foreign_gate.sh` で、判定は `docs/perf/logs/macmini-foreign-job-gate/foreign_job_gate.py`（#4227。G1 caffeinate・G2 %CPU・G3 名前・G4 GPU。ps と ioreg を 1 回ずつ読む）。どちらも変えていない。
  - 各 round の前に、他ジョブが CLEAN かつ load1 < 3.0 になるまで、同じループで待つ（30 秒間隔・最大 60 回。従来の load1 の待ちと同じ上限）。
  - round の後（7 腕と PyTorch の同席計測まで終えた直後）に 1 回確かめる。
  - 通過の行は従来どおり `round<N> gate=pass gate_ok=1 ...` の 1 行。通過以外は行末に `reason=foreign-job|load1|foreign-gate-error|load1-unavailable` を足す（ゲート失敗は `phase=before` も）。
  - 終了後の FOREIGN・ゲート失敗は、同じ round の `gate=invalid gate_ok=0 ... reason=foreign-after|foreign-gate-error phase=after` を足す。
  - 不通過でも系列は最後まで進める（従来の timeout と同じ。部分系列で判定する経路は作らない）。
  - 開始前の不通過は `aggregate.py` の P-GATE で undetermined になる。
  - 終了後の FOREIGN は、同じ round の 2 行目を `aggregate.py` の `read_load_gate` が重複として扱い、最後の行の `gate_ok=0` で P-GATE も不成立になるので undetermined になる（#4156 は `run_ab exit=` の P-RUNAB に頼ったが、本キットには `run_ab` が無い）。`aggregate.py` は変えていない。
  - ゲートの出力は `macmini/foreign-gate/`、ゲートの sha256 は `macmini/foreign_gate.txt`（絶対パスは書かない）。収録前の mask と漏れ検査の対象に入る。
- **系列の前の確認**: ビルドの後・最初のセルの前（hw-lock の下）に、次の順に確かめる。記録は `macmini/series-start.txt`。
  - Safari: `ps` の comm の basename がちょうど `Safari` のプロセス（Safari アプリ本体）が 0 件（#4160 の `RULE.v2.txt` 項 4 (b) 1 と同じ）。補助プロセス（`com.apple.Safari.SafeBrowsing.Service` など）は条件にせず、名前に Safari を含むプロセスの basename と %CPU を記録するだけ。
  - mediaanalysisd: #4146 の `RULE.v2.txt` 項 8 (c) と同じ基準（#4258 と同じ関数）。2 分間隔で 3 回、%CPU が 5 未満。名前は basename の完全一致で見る。複数あれば最大値で判定し、プロセスがなければ通過。
  - `ps` の失敗・出力が空・非数値は不成立。
  - 不成立なら 1 セルも測らずに止める。記録は `${WORK}/prestart.txt`（`result=not-started reason=safari|mediaanalysisd`）に移る。系列は始まっていない。
- **起動前に止まる場合**: `FOREIGN_GATE` が未指定・相対パス・存在しない・禁止文字を含むときは、ロックを取る前に止まる。
- **PR-2 への申し送り**: `RULE.template.txt` は変えていない（#4258 も `RULE.txt` を変えず、記述は後の RULE.v2 で足した）。`RULE.txt` を固定するときに、§4 の前提ゲートへ上の 2 つ（他ジョブの前提ゲート・系列の前の確認）を書き足すかを決める。`orchestrate_macmini.sh` の sha256 はこの版の値で埋める（`python3 -I aggregate.py --fingerprint`）。

## マスク規則・実機ロック

- ホスト名は `<metal-node>`、`$HOME` は `<home>`。ユーザー名・絶対パス・IP は除去し、収録前の grep が 0 件でなければ収録しない。
- 実機ロック `~/work/.fandhe-hw-lock` を取得してから測り、取得・解放時刻を `hwlock.log` に残す。

## 検証状況

- Linux で検証済み: lib テスト（`select`・帯の境界・専用プール／jpw の bit 完全一致）、`aggregate.py --self-test`／`--check-template`、`bash -n`、`--dry-run`、patch の `git apply --check`、aarch64-apple-darwin のコンパイル確認。
- 他ジョブの前提ゲートと系列の前の確認（2026-11-13）は、Linux で `bash test_foreign_gate_4162.sh` を通した。実機には接続していない。
  - 確かめた場合: 全 CLEAN・終了後 FOREIGN・開始前 60 回 FOREIGN（60 回で打ち切り）・load1 超過・ゲート失敗（前・後）・出力不一致・load1 が読めない。
  - 変更なしの `aggregate.py` の判定: 全 CLEAN は ADOPT まで進み、開始前の不通過と終了後の FOREIGN は undetermined。
  - Safari: 補助プロセスだけなら通過・アプリ本体で不成立・`ps` の失敗と空で不成立。mediaanalysisd: 境界 5.0・部分一致の名前・複数・非数値・`ps` の失敗・空。
  - 系列の前の確認が不成立なら prestart を残して止まり、先へ進まないこと。
  - orchestrate: `FOREIGN_GATE` の不正でロック前に止まること・呼び出し順・上書き禁止の対象・dry-run。
  - 変異の確認: 次の 10 の変異のどれでも self-test が落ちることを確かめた。round 後の呼び出しを外す・開始前の GATE_NOT_OK を数えない・終了後の GATE_NOT_OK を数えない・終了後の行を書かない・mediaanalysisd を 5 以下で通す・mediaanalysisd を部分一致で拾う・Safari を部分一致で拾う・系列の前の確認で止まらない・系列の前の確認を呼ばない・`FOREIGN_GATE` のパス検査を外す。
- **未検証**: Mac mini 上での `#[ignore]` テスト（P-FIRE・P-BIT）、A/B の実行、P-RT、`foreign_job_gate.py` と Safari・mediaanalysisd の確認を実機の `ps` で動かすこと。

## 限界

- 候補が 5 つあるため多重比較になる（補正はしない）。
- 専用プールのワーカーとグローバルプールの spin が混在する場面での干渉は、ガードセルが帯の外でプールが生成されないため検出できない。
- macOS には pinning の手段がなく、T=4 が P コアだけで動く保証はない（#3175 §5）。
- NB-RANGE5・NB-SEP は #4142 が所有者承認待ちのため記録だけにする。承認済みとは書かない。

## スコープ外

PR-2・PR-3、#4151 の実測、既定の切り替え（#4165）、SME の unsafe 独立監査（#4157）、#4156 の SME 既定 ON の判定。
