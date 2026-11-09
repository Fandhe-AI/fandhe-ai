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
| `aggregate.py` | 厳格な集計（標準ライブラリのみ。`python3 -I aggregate.py --self-test`／`python3 -I aggregate.py <dir> --expect-rev <SHA> > aggregate.md`）。`RULE.txt` の記入欄残り・集計器 sha256 不一致は停止。`ab/` が無ければセル A/B は unavailable として総合を undetermined にする |

診断テストは `crates/backend-metal/src/gemm_steel_large_n_diag_tests.rs`（出力行の書式は `aggregate.py` の正規表現と 1 対 1）。
セル A/B の runner は `scripts/bench/framework-compare/run_ab_steel_large_n_gemm_metal.sh`（`run_ab_frag_interleave_gemm_metal.sh` の派生）。

## 再現（Mac mini 上。`docs/real-hardware-verification-env.md`）

1. 規則を固定したコミットを `git archive` して `.rev-stamp` を置き、rsync で転送する。ツリー指紋を両側で突合する。
2. セル A/B 用に before／after の 2 clone を作り、after は `crates/backend-metal/src/tile.rs` の `STEEL_LARGE_N_ENABLED` を false→true の 1 行だけ書き換える（コミットしない）。
3. 純カーネル計測: `CARGO_TARGET_DIR=$HOME/work/target-fandhe-ai OUT=<出力ディレクトリ> sh orchestrate_macmini.sh`
4. セル A/B（3 を終えてから。同時に走らせない）:
   `AB_BEFORE_FACADE_PATH=<abs>/before/crates/facade AB_AFTER_FACADE_PATH=<abs>/after/crates/facade bash run_ab_steel_large_n_gemm_metal.sh 4160`
5. 回収・マスク（ホスト名は `<metal-node>`）・集計。いずれも nohup のサブシェルで切り離して起動し、生存を確認する。

## 申し送り

- PR-2（Mac mini）: 雛形を埋めた `RULE.txt` を単独コミットで main に固定する。
- PR-3（Mac mini）: 5 run の系列・`aggregate.md`・§11 の記入欄・`loss-attribution-matrix.md` の M-MTL-G1024／G2048 行の更新（`Closes #4160`）。
- `STEEL_LARGE_N_ENABLED` の既定を true にする結線は Phase 4（#4166）の担当で、所有者の承認が要る。
- NB-RANGE5／NB-SEP の定義（#4142 `docs/perf/noise-band-decision.md`）は記録時点で所有者承認待ち。
