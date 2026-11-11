# N≥1024 限定 steel 候補 T0U-LU-FB の Mac mini A/B（イシュー #4160）

結論の記入先は `docs/perf/metal-gemm-steel-candidates.md` §11。本ディレクトリは事前登録規則の雛形・計測基盤・（将来の）実測ログ。

## 状態: 実測済み・総合 UNDETERMINED（3 系列とも全ゲート通過にならず。2026-11-11・Mac mini `<metal-node>`）

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
  - 実施状況（2026-11-11）: 系列・`aggregate.md`・§11.5 は記録した。総合 UNDETERMINED のため `loss-attribution-matrix.md` は変えていない。なお同表の `M-MTL-G1024／G2048` は M4 Max の行で、Mac mini の行は `MM-MTL-G1024／G2048`。行の付け直しは §11.7 のとおり #4168 の担当。#4160 を閉じるかは所有者の判断待ち。
- `STEEL_LARGE_N_ENABLED` の既定を true にする結線は Phase 4（#4166）の担当で、所有者の承認が要る。
- NB-RANGE5／NB-SEP の定義（#4142 `docs/perf/noise-band-decision.md`）は #4142 で所有者承認済み（2026-11-10。https://github.com/Fandhe-AI/fandhe-ai/issues/4142#issuecomment-6757497844 ）。NB-RANGE5 は腕ごとの round 別比の最小〜最大（`aggregate.py` は `range_before`／`range_after` を別々に出す）。
