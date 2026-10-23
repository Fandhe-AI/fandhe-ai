# m=64 細タイル候補の Mac mini A/B（イシュー #3209）

結論は `docs/perf/metal-gemm-m64-thin-tile-candidates.md` §7。本ディレクトリは事前登録規則と計測基盤。
**診断のみ**（`tile.rs` の定数・選択ロジック・tolerance・parity baseline・FMA 契約・シェーダは不変）。

## 状態: Mac mini 実測済み（正式系列）

- 計測対象リビジョン `bd2ce9464783edfdc395a923a76d4b88f6529ab3`（`rev-stamp.txt`）。展開ツリー指紋は開発機（`git archive` 展開）と
  Mac mini で一致（`tree-fingerprint.txt`）。機体は Mac mini（Apple M4・GPU 10 コア・`verified_tile_table=M4Base`。`env_info.txt`）。
- 正式系列: 純カーネルは全 gate pass・`series done`（`load_gate.log`）、セル A/B は 5 round 全 gate pass・`series=official`（`ab/load-gate-*.log`）。
  再実行・run 差し替え・セル除外はしていない（外れ値も除外していない）。
- 判定（機械判定。`aggregate.md`）: T1〜T4 は規則 11 で ADOPT。ゲート写像は **undetermined**
  （到達追跡で (64,256,784) が split-K 先取りのため T1 に到達せず、机上の前提 2 件と食い違った。ゲート反転で変わるのは (784,64,256)→T3 のみ）。
- `--cell-ab anomaly` の導出: RULE 項 13 の字義どおり。`ab/skipped-*.log` は空・checksum は 4 セルとも完全一致（fail ではない）。
  train 64/fresh の round 別比が 1.0456／1.0123／1.0005／1.0063／1.0066 と 5/5 round すべて after が遅い向きだったため anomaly
  （差は 0.9%。ノイズ帯として扱い、ゲート判定は ADOPT にしない）。`compare_gemm_ab.py --threshold 1.00` の exit 3 は
  「比が 1.00 を超えるセルがある」の意味で fail ではない。
- 結論と数値表は `docs/perf/metal-gemm-m64-thin-tile-candidates.md` §7。結線（ADOPT 候補の写像）は別 PR・所有者判断で、本ディレクトリは判定の記録のみ。
- 規則の固定: `RULE.txt` の sha256 = `d5bfebcacf27b6b95da47297cc71ba66f2ce3e44a76d3e50c107e8cd26a26e40`、
  規則を固定したコミット = `b5da0c2b0f4a080edf183ce7cb7fa6c7eb922d2b`（規則本文は固定後に変更していない）。
- マスク: ホスト名・ユーザー名・`$HOME` の絶対パスは `<home>`／`masked` に置換済み（`grep` で 0 件を確認）。バイナリ（`bench-fandhe-3209-*`）は収録していない
  （sha256 は `ab/sha-3209-*.txt`）。

## 収録物

| パス | 内容 |
|---|---|
| `RULE.txt` | 事前登録規則（計測前に単独コミット） |
| `orchestrate_macmini.sh` | 純カーネル計測の起動（ビルド → 前提ゲート → 5 run × {trace・kern}。load1 < 3.0 ゲート・上書き禁止・`.rev-stamp` 必須） |
| `aggregate.py` | 厳格な集計（標準ライブラリのみ。`python3 -I aggregate.py --self-test`／`python3 -I aggregate.py <dir> --expect-rev <SHA> --cell-ab <ok｜anomaly｜fail｜unavailable> > aggregate.md`）。self-test の fixture は実行時に一時ディレクトリへ生成する（欠損・run 数不足・非有限値・hash 不一致・正式系列でない・rev 不一致を検証） |
| `trace_run{1..5}.log`・`kern_run{1..5}.log` | `DEVICE`／`TRACE`／`CAND` 行 |
| `prereq_bitident.log`・`parity.log` | 前提ゲート（既存 bit 一致テスト 92 件・CPU 参照 parity） |
| `bitident_report.log` | 候補別 bit 一致の記録（`BITIDENT` 行。不一致候補は `ok=false` で記録して継続し、集計側が NOT_ADOPTABLE として除外する。既存の厳密診断 `prereq_bitident.log` は 1 件の不一致で panic するため失敗しても系列は止めず、rc を `load_gate.log` へ残す） |
| `load_gate.log`・`env_info.txt`・`rev-stamp.txt`・`tree-fingerprint.txt`・`build_tests.log` | 負荷ゲート・機体情報・腕の同定・ビルドログ（ホスト名・ユーザー名・`$HOME` はマスク） |
| `ab/` | セル A/B runner の出力（`results-*.jsonl`・`sha-*.txt`・`tree-*.txt`・`compare-*.md／.err`・`load-gate-*.log`・`cooldown-*.log`・`skipped-*.log`・`uptime-*.log`） |
| `aggregate.md` | 機械集計 |

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
