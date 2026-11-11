# N≥1024 限定 steel 候補 T0U-LU-FB の Mac mini A/B（イシュー #4160）

結論の記入先は `docs/perf/metal-gemm-steel-candidates.md` §11。本ディレクトリは事前登録規則の雛形・計測基盤・（将来の）実測ログ。

## 状態: 未実測（PR-1 の範囲。機構と計測基盤のみ）

- `tile::STEEL_LARGE_N_ENABLED`（既定 `false`）と形状限定の選択 `tile::select_steel_large_n` を追加した。既定挙動は不変。
- 実測・判定・Issue のクローズは Mac mini が要るため後続（下記「申し送り」）。

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
- `STEEL_LARGE_N_ENABLED` の既定を true にする結線は Phase 4（#4166）の担当で、所有者の承認が要る。
- NB-RANGE5／NB-SEP の定義（#4142 `docs/perf/noise-band-decision.md`）は #4142 で所有者承認済み（2026-11-10。https://github.com/Fandhe-AI/fandhe-ai/issues/4142#issuecomment-6757497844 ）。NB-RANGE5 は腕ごとの round 別比の最小〜最大（`aggregate.py` は `range_before`／`range_after` を別々に出す）。
