# m=64 細タイル候補の Mac mini A/B（イシュー #3209）

結論は `docs/perf/metal-gemm-m64-thin-tile-candidates.md` §7。本ディレクトリは事前登録規則と計測基盤。
**診断のみ**（`tile.rs` の定数・選択ロジック・tolerance・parity baseline・FMA 契約・シェーダは不変）。

## 状態: 基盤のみ出荷・Mac mini 実測は未実施（unavailable）

- 本ディレクトリに実測ログ（`trace_run*.log`・`kern_run*.log`・`parity.log`・`aggregate.md` 等）は**まだない**。
  RULE 項 15 により数値は埋めず、候補の判定は undetermined とする。推定値では埋めない。
- 実測は下記「再現」で Mac mini 上で行い、同ディレクトリへ回収して `aggregate.py` で集計する。
- 規則の固定: `RULE.txt` の sha256 = `d5bfebcacf27b6b95da47297cc71ba66f2ce3e44a76d3e50c107e8cd26a26e40`、
  規則を固定したコミット = `b5da0c2b0f4a080edf183ce7cb7fa6c7eb922d2b`（計測基盤はそのあとの別コミット。
  規則本文は固定後に変更していない）。

## 収録物

| パス | 内容 |
|---|---|
| `RULE.txt` | 事前登録規則（計測前に単独コミット） |
| `orchestrate_macmini.sh` | 純カーネル計測の起動（ビルド → 前提ゲート → 5 run × {trace・kern}。load1 < 3.0 ゲート・上書き禁止・`.rev-stamp` 必須） |
| `aggregate.py` | 厳格な集計（標準ライブラリのみ。`python3 -I aggregate.py --self-test`／`python3 -I aggregate.py <dir> --expect-rev <SHA> --cell-ab <ok｜anomaly｜fail｜unavailable> > aggregate.md`）。self-test の fixture は実行時に一時ディレクトリへ生成する（欠損・run 数不足・非有限値・hash 不一致・正式系列でない・rev 不一致を検証） |
| （実測後）`trace_run{1..5}.log`・`kern_run{1..5}.log` | `DEVICE`／`TRACE`／`CAND` 行 |
| （実測後）`prereq_bitident.log`・`parity.log` | 前提ゲート（既存 bit 一致テスト 92 件・CPU 参照 parity） |
| （実測後）`bitident_report.log` | 候補別 bit 一致の記録（`BITIDENT` 行。不一致候補は `ok=false` で記録して継続し、集計側が NOT_ADOPTABLE として除外する。既存の厳密診断 `prereq_bitident.log` は 1 件の不一致で panic するため失敗しても系列は止めず、rc を `load_gate.log` へ残す） |
| （実測後）`load_gate.log`・`env_info.txt`・`rev-stamp.txt`・`tree-fingerprint.txt`・`build_tests.log` | 負荷ゲート・機体情報・腕の同定・ビルドログ（ホスト名・ユーザー名・`$HOME` はマスク） |
| （実測後）`ab/` | セル A/B runner の出力（`results-*.jsonl`・`sha-*.txt`・`tree-*.txt`・`compare-*.md／.err`・`load-gate-*.log`・`cooldown-*.log`・`skipped-*.log`・`uptime-*.log`） |
| `aggregate.md` | 機械集計（実測後） |

診断テストは `crates/backend-metal/src/gemm_m64_thin_tile_ab_diag_tests.rs`（出力行の書式は `aggregate.py` の正規表現と 1 対 1）、
encode-only の診断入口は `MetalGemm::diag_encode_strided_tiled`（`#[cfg(test)]`。本番経路は不変）。
セル A/B の runner は `scripts/bench/framework-compare/run_ab_m64_thin_tile_metal.sh`。

## 規則の解釈・実装上の注意

- **NN の 8 非整列形状は実効次元で測る**: `diag_encode_tiled_nn` は 8 の倍数以外を拒否する。本番 `dispatch_auto` も
  8 の倍数へゼロ詰めした実効次元で選択・dispatch するため、(64,10,256)・(256,64,10)・(256,10,64) は
  (64,16,256)・(256,64,16)・(256,16,64) として測る（ゼロ詰めは FMA の累積値を変えないので CPU 参照との比較も
  実効次元で成立する）。出力行の `shape=` は生の形状、`eff=` が実効次元。到達追跡（`TRACE`）も実効次元で行う。
- **`cur` は classic 構成**: (64,256,784) は本番 `dispatch_auto` では split-K が先に取るが、`cur` は
  `select_for_verified_device` の classic 構成を明示指定して測る（RULE 項 4 のとおり）。
- 候補の総合判定で undetermined が 1 件でもあれば undetermined を最優先にする（`aggregate.py` 冒頭）。

## 再現（Mac mini 上。`docs/real-hardware-verification-env.md` §7）

1. 計測前コミットを `git archive` して `.rev-stamp` を置き、rsync で転送する（before／after／diag の 3 clone が必要。
   セル A/B runner は `.git` を必要とするため git bundle から clone する。after は
   `crates/backend-metal/src/tile.rs` の `M64_THIN_TILE_ENABLED` を false→true の 1 行だけ書き換えた計測専用 checkout で、
   main へはコミットしない）。`.rev-stamp` とツリー指紋を `git archive` と突合して記録する。
2. 純カーネル計測（ビルドは計測前に全部済む）:
   `CARGO_TARGET_DIR=$HOME/work/target-fandhe-ai OUT=<出力ディレクトリ> sh orchestrate_macmini.sh`
3. セル A/B（2 を終えてから。同時に走らせない）:
   `AB_BEFORE_FACADE_PATH=<abs>/before/crates/facade AB_AFTER_FACADE_PATH=<abs>/after/crates/facade bash run_ab_m64_thin_tile_metal.sh 3209`
4. 回収・マスク・集計（`aggregate.py`）。2 時間を超える場合は 2 と 3 を別セッションに分ける。
   nohup のサブシェルで切り離して起動し、生存を確認する。openrsync の `--delete-excluded` は DGX 側等の
   既存 target を消すため注意する。

## 開発機（M4 Max）でのスモーク

診断 3 テストを `--ignored --test-threads=1` で 1 回実行し、行の書式・assert・`aggregate.py` の正規表現との一致を確認した
（ここの値は判定に使わず、本ディレクトリにも収録しない）。
