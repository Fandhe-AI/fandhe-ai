# N≥1024 限定 steel 候補 T0U-LU-FB の Mac mini A/B（イシュー #4160）

結論の記入先は `docs/perf/metal-gemm-steel-candidates.md` §11。本ディレクトリは事前登録規則の雛形・計測基盤・（将来の）実測ログ。

## 状態: 実測済み・総合 ADOPT（RULE.v2 の系列 `formal-v2-2`。2026-11-11・Mac mini `<metal-node>` 単機）

- 規則: `RULE.v2.txt`（PR #4252 で固定）。旧 `RULE.txt` の 3 系列がすべて UNDETERMINED だったことを受けた所有者判断（2026-11-11「準備だけ整えて測り直す」）による改訂で、変えたのは各系列の前の準備だけ（ゲート・判定式・閾値・ラウンド数・対象ツリー・`aggregate.py` は同じ）。
- 結果: `formal-v2-2` が全ゲート通過（純カーネル 39/39・セル A/B 5/5・両スクリプト rc=0）。RULE.v2 項 3 によりここで止め、この系列で判定した。`aggregate.py`（sha256 `e8ae78af…`・`--expect-rev afbfeafba…`）の総合は **ADOPT**。
  - ADOPT の意味は「Phase 4（#4166）で所有者が結線を判断する価値がある」に限る（RULE 項 1）。`tile::STEEL_LARGE_N_ENABLED` は `false` のままで、結線・M4 Max での確認・既定反転の承認は #4166 の担当。
  - 純カーネル（kernel_gpu の 5 run 中央値）: sl/cur は N=1024 で -7.0%、N=2048 で -10.3%、N=4096 で -9.5%（各 5/5 run で速い・bit 一致 sl=ref・parity pass・到達成立）。N=256／512 の no-op は成立（解決タイル同一・bit 一致）。
  - セル A/B（非後退ゲート）: 状態 ok。5/5 round 一貫して 2% を超えて後退するセルはない。checksum は全セルで完全一致。
  - スコアボード比（candle 相手）: N=2048 は NB-SEP 成立（after の比 1.005〜1.014、before 0.936〜0.946）。N=1024 は不成立（after 0.900〜1.080）。G4096 は記録のみ（不成立）。「1 位に変わった」と書けるのは opt-in を ON にした状態の G2048 だけで、main の既定は変わっていない。

| 系列 | 時刻（UTC） | 純カーネル（ゲート） | セル A/B（ゲート） | 当たったゲート | aggregate.py |
|---|---|---|---|---|---|
| `series-formal-v2-1`（LABEL fv2-1） | 08:44:52〜09:18:53 | 39/39 pass | 5/6 pass | round2 `after-N4096-fresh` 直後に foreign-after（`secd` %CPU 33.3。G2） | rc=0・総合 undetermined（参考系列） |
| `formal-v2-2` 試み 1 | 09:31:39〜09:35:48 | — | — | 起動前のツリー照合で不一致（下記）。計測プロセスは起動していない（RULE.v2 項 4 (c)。系列に数えない） | — |
| `series-formal-v2-2`（LABEL fv2-2） | 09:43:41〜10:16:25 | 39/39 pass | 5/5 pass | なし | rc=0・**総合 ADOPT**（判定に使う系列） |

- 系列の前の準備（RULE.v2 項 4）: 各系列の前に所有者から準備の連絡を受け、項 4 (b) の確認を通してから起動した。確認の記録は各系列の `env/series-start.txt`（試み 1 は `drive-v2/prestart-v2-2-1.txt`）。Safari アプリ本体・G1/G3 に当たるプロセスは、どの確認でも 0 件だった。
  - 後で見つかった常駐: `formal-v2-1` の後に Mac mini を調べると、Magnet・Parsec（parsecd）・Siri AI（31 日前から）、ループバックだけで待ち受ける node（26 日前から）、別アカウントの colima（lima VM。21 日前から）が動いていた。いずれも %CPU 0.0 で、`formal-v2-1` の間も動いていた（ゲートには当たっていない）。
  - `formal-v2-2` の間に残っていたもの: Parsec、Siri AI（止めた直後に macOS が再起動した）、別アカウントの colima（権限がなく止めていない。所有者了承のうえ開始）。
- 逸脱の記録（2 件）:
  1. **計測セッションによるプロセスの停止**: RULE.v2 項 4 (c) は「ほかのプロセスを計測セッションが止めることはしない」と定めているが、所有者の明示の指示（2026-11-11。止める対象も所有者が選んだ）により、`formal-v2-2` の前に計測セッションが Magnet・Siri AI・node の待ち受けを SIGTERM で止めた。Magnet と node は止めたまま（再起動していない）。
  2. **Python のキャッシュの削除**: `formal-v2-2` 試み 1 は、ツリー照合で `scripts/bench/framework-compare/__pycache__/checksum_contract.cpython-315.pyc` が 1 件増えていたため始めなかった（`formal-v2-1` の A/B の比較スクリプトが 09:18Z に作った Python のキャッシュ。ソースは不変で manifest 照合は一致。旧系列 1〜3 の後には作られておらず、理由は未確認）。所有者判断（「pyc を消して再試行」）でこの 1 件だけを消し、指紋が配置時の値に戻ったことを確かめてから試み 2 として起動した（`drive-v2/pyc-removal-v2-2.txt`）。この所有者の選択を項 4 (c) の「再度の連絡」として扱った。試み 2 の後には作られていない。
- 固定前の確認の通し（RULE.v2 項 6）: `drive-v2/precheck-dry-1.txt`（計測は起動していない）。項 6 は収録先を「記録 PR の `drive/`」と書いているが、旧系列の `drive/` と分けるため `drive-v2/` に置いた。
- 開示: RULE.v2 項 0 のとおり（参考値を見た後の判断・試走 1 の判定表 3 行・旧系列 3 の N=256 の 2 行）。v2 の各系列の値は、`formal-v2-2` が全ゲート通過と決まった後に集計して開いた。

### v2 で収録した出力

- `series-formal-v2-<k>/`: 旧系列と同じ構成（純カーネルの出力・`ab/`・`env/`・`foreign-gate.tar.gz`／`foreign-gate-summary.tsv`・`aggregate.md`）。`env/series-start.txt` が RULE.v2 項 4 (b) の確認の記録。
- `drive-v2/`: 駆動ログ（`formal-v2.drive.log`・各 `*.drive.out`）・系列 1 本の起動スクリプト（`run4160v2.sh`）・固定前の確認の通し・試み 1 の記録・pyc の削除記録。

## 旧 `RULE.txt` の 3 系列（参考。総合 UNDETERMINED。2026-11-11）


- 規則: `RULE.txt`（PR #4248 で固定。固定コミット `afbfeafba`、対象ツリー `19eca6fa3`）。所有者判断「別ラベルで最大 3 系列」（RULE 項 14 (d)）に従い、全ゲート通過の系列が出るまで回した。
- 結果: 3 系列とも、セル A/B の他ジョブの前提ゲートに当たり、全ゲート通過にならなかった。RULE 項 14 (d) により **3 系列とも参考系列・総合 UNDETERMINED**。`tile::STEEL_LARGE_N_ENABLED` は `false` のまま据え置く（結線は #4166 で保留のまま）。
- 純カーネル計測（orchestrate）は 3 系列とも load1 ゲートと他ジョブのゲートを全部通った（各 39/39）。`aggregate.md` の「正式系列: はい」「ADOPT_CANDIDATE」は**純カーネル部分だけ**の表示で、総合判定ではない（総合は aggregate.md 自身も undetermined）。ここから ADOPT は導かない。

| 系列 | 時刻（UTC） | 純カーネル（ゲート） | セル A/B（ゲート） | 当たったゲート | aggregate.py |
|---|---|---|---|---|---|
| `series-formal-1`（LABEL f1） | 04:59:18〜05:35:55 | 39/39 pass | 5/6 pass | round3 `before-N1024-fresh` 直後に foreign-after（`com.apple.Safari.SafeBrowsing.Service` %CPU 94.5。G2） | rc=0・総合 undetermined |
| `series-formal-2`（f2） | 05:40:05〜06:13:50 | 39/39 pass | 5/7 pass | round1 `candle-N1024-fresh` 直後（同 Safari 94.5）・round2 `after-N512-reuse` 直後（`runningboardd` 62.4・`PerfPowerServices` 27.6） | rc=0・総合 undetermined |
| `series-formal-3`（f3） | 06:18:05〜07:30:12（途中で停止） | 39/39 pass | round4 で timeout と foreign-after | 06:46:45Z に起動した他プロセス（同じアカウントの Python の待ち受け。G3）で round4 のセルの前が 30 分待って timeout | rc=1（`aggregate-error.txt`。セル行が 5 に満たない） |

- 系列 3 は所有者判断（2026-11-11「系列 3 を止めて記録」）により、A/B の round5 の途中（07:30:12Z）で止めた（RULE 項 14 (d) からの逸脱。経緯は `drive/abort.txt`）。`series-formal-3/aggregate-kern-only.md` は、純カーネル部分を見るために `ab/` を除いた写しで集計した参考の派生出力で、判定には使わない。
- 純カーネル（参考）: sl/cur は N=1024 で -6.9%／-6.8%／-6.7%、N=2048 で -10.4%／-10.6%／-10.5%、N=4096 で -9.7%／-9.8%／-9.5%（系列 1／2／3。各 5/5 run で速い。bit 一致 sl=ref・parity pass。cur_aa/cur は ±0.3% 以内）。
- セル A/B（参考。系列 1・2）: after/before の中央値比は N=2048 fresh 0.940／0.945・reuse 0.932／0.931、N=4096 fresh 0.905／0.907・reuse 0.913／0.917、N=1024 fresh 0.969／0.999・reuse 0.969／0.970。checksum は全セルで完全一致。スコアボード比の NB-SEP は負荷ゲート不成立のため判定不能。
- 開示: (1) 固定前の試走 1 で純カーネルの判定表 3 行を見た（RULE 項 14 (g)）。(2) 系列 3 の停止を確かめる際に、A/B の N=256 の値 2 行（判定対象外）を見た。系列 1・2 の値は、3 系列の結果が決まった後に集計して開いた。

### 収録した出力

- `series-formal-<k>/`: 純カーネル計測の出力（`orchestrate_macmini.sh` の OUT）、`ab/`（セル A/B の出力。`results-*-f<k>-metal-gemm.jsonl`・`load-gate-4160-metal-f<k>.log` ほか）、`env/`（mediaanalysisd の確認・系列の開始・状態・系列直前のツリー照合・スクリプトの標準出力）、`aggregate.md`（系列 1・2）。
- 他ジョブのゲートの各回の出力（純カーネル 74〜76 件、A/B 260〜290 件／系列）は件数が多いため、`foreign-gate.tar.gz`・`ab/foreign-gate.tar.gz`（名前順・mtime 0 の決定的な tar.gz）にまとめた。1 ファイル 1 行の要約は `foreign-gate-summary.tsv`・`ab/foreign-gate-summary.tsv`。aggregate.py はこれらを読まない。
- `drive/`: 駆動ログ（`formal.drive.log`）・配置時のキットの sha256（`kit-verify.txt`）・A/B の出力を除いた配置時の指紋・停止後の照合（`post-abort-check.txt`）・停止の記録（`abort.txt`）。
- 固定前の試走 1 の出力はリポジトリに入れていない（RULE 項 14 (g)）。

## 収録物

| パス | 内容 |
|---|---|
| `RULE.template.txt` | 事前登録規則の雛形。記入欄 4 つ（`fixed-at-utc`・`tree_sha`・`tree_fingerprint`・`aggregate_script_sha256`）を埋めて `RULE.txt` として**単独コミット**で固定してから計測する |
| `orchestrate_macmini.sh` | 純カーネル計測の起動（ビルド → 前提ゲート → 5 run × {trace・kern} と candle excl。`--dry-run` あり。上書き禁止・`.rev-stamp` 必須・load1 ゲート） |
| `test_foreign_gate_4160.sh` | 他ジョブの前提ゲートの self-test（偽ゲートで通し実行。`bash test_foreign_gate_4160.sh` → `self-test ok`。約 2 分） |
| `aggregate.py` | 厳格な集計（標準ライブラリのみ。`python3 -I aggregate.py --self-test`／`python3 -I aggregate.py <dir> --expect-rev <SHA> > aggregate.md`）。`RULE.txt` の記入欄残り・集計器 sha256 不一致は停止。`ab/` が無ければセル A/B は unavailable として総合を undetermined にする |

診断テストは `crates/backend-metal/src/gemm_steel_large_n_diag_tests.rs`（出力行の書式は `aggregate.py` の正規表現と 1 対 1）。
セル A/B の runner は `scripts/bench/framework-compare/run_ab_steel_large_n_gemm_metal.sh`（`run_ab_frag_interleave_gemm_metal.sh` の派生）。

## 他ジョブの前提ゲート（2026-11-10 の所有者指示）

Mac mini では hw-lock を取らない他ジョブや `mediaanalysisd` が計測と重なりうる（#4151・#4153・#4154）。両スクリプトは
`docs/perf/logs/macmini-foreign-job-gate/foreign_job_gate.py`（`FOREIGN_GATE` で絶対パスを渡す。必須）で各計測プロセスを前後から挟む。
関数は `scripts/bench/framework-compare/ab_foreign_gate.sh`（両者で共用）。

- 前: 他ジョブ CLEAN かつ load1 < 3.0 を 30 秒間隔・最大 60 回待つ（従来の load1 の待ちと同じ上限）。揃わなければ `gate=timeout reason=foreign-job|load1`、ゲート失敗は `gate=invalid reason=foreign-gate-error phase=before`。
- 後: 1 回確認し、FOREIGN は同じ label の `gate=invalid reason=foreign-after phase=after`、失敗は `reason=foreign-gate-error phase=after`。
- いずれも既存の GATE-TIMEOUT／GATE-INVALID と同じく記録して続行し、`aggregate.py`（変更なし）が正式系列から外す。通過の行の書式は従来のまま。
- セル A/B は round 先頭（`round<N> gate=...` の 1 行）に加え、各セルのプロセスの前後でも確認する（通過は行を足さず、異常だけ `round<N> gate=... cell=<arm>-N<size>-<mode>` を足す）。
- 各回の出力（マスク済み）は `<OUT>/foreign-gate/`（セル A/B は `results/raw/foreign-gate/4160-metal-<label>/`）。ゲートの sha256 は `env_info.txt`／同ディレクトリに残す。

## 再現（Mac mini 上。`docs/real-hardware-verification-env.md`）

1. 規則を固定したコミット（RULE.txt のみを足す単独コミット）を `git archive` し、`.rev-stamp` を 2 行（1 行目 = そのコミットの SHA、2 行目 = `parent=<その親の SHA>`。親が RULE の `tree_sha`）で置き、rsync で転送する。ツリー指紋を両側で突合する。
2. セル A/B 用に before／after の 2 clone を作り、after は `crates/backend-metal/src/tile.rs` の `STEEL_LARGE_N_ENABLED` を false→true の 1 行だけ書き換える（コミットしない）。
3. 純カーネル計測: `FOREIGN_GATE=<foreign_job_gate.py の絶対パス> CARGO_TARGET_DIR=$HOME/work/target-fandhe-ai OUT=<出力ディレクトリ> sh orchestrate_macmini.sh`
4. セル A/B（3 を終えてから。同時に走らせない）:
   `FOREIGN_GATE=<abs> AB_BEFORE_FACADE_PATH=<abs>/before/crates/facade AB_AFTER_FACADE_PATH=<abs>/after/crates/facade bash run_ab_steel_large_n_gemm_metal.sh 4160`
5. 回収・マスク（ホスト名は `<metal-node>`）・集計。いずれも nohup のサブシェルで切り離して起動し、生存を確認する。

## 申し送り

- PR-2（Mac mini）: 雛形を埋めた `RULE.txt` を単独コミットで main に固定する。
- PR-3（Mac mini）: 5 run の系列・`aggregate.md`・§11 の記入欄・`loss-attribution-matrix.md` の M-MTL-G1024／G2048 行の更新（`Closes #4160`）。
  - 実施状況（2026-11-11）: 旧 `RULE.txt` の 3 系列は総合 UNDETERMINED（PR #4249）。RULE.v2（PR #4252）の `formal-v2-2` で総合 ADOPT となり、系列・`aggregate.md`・§11.5・§11.7 を記録した（本記録 PR で #4160 を閉じる）。
  - `loss-attribution-matrix.md` は変えていない。ADOPT は opt-in を ON にした状態の判定で、main の既定（`STEEL_LARGE_N_ENABLED=false`）とスコアボードの値は変わっていないため。同表の `M-MTL-G1024／G2048` は M4 Max の行で、Mac mini の行は `MM-MTL-G1024／G2048`。結線後の行の付け直しは §11.7 のとおり #4168 の担当。
- `STEEL_LARGE_N_ENABLED` の既定を true にする結線は Phase 4（#4166）の担当で、所有者の承認が要る。
- NB-RANGE5／NB-SEP の定義（#4142 `docs/perf/noise-band-decision.md`）は #4142 で所有者承認済み（2026-11-10。https://github.com/Fandhe-AI/fandhe-ai/issues/4142#issuecomment-6757497844 ）。NB-RANGE5 は腕ごとの round 別比の最小〜最大（`aggregate.py` は `range_before`／`range_after` を別々に出す）。
