# m=64 GEMM の job 分割・再 pack・確保の計数と単スレッド GFLOPS・A/B ログ（イシュー #3177）

事前登録規則・A/B patch・オーケストレーション・集計・Mac mini と GB10 の実測ログ。結論は
[`docs/perf/cpu-gemm-m64-jobgrid-diagnosis.md`](../../cpu-gemm-m64-jobgrid-diagnosis.md)。

## 収録物

| ファイル | 内容 |
|---|---|
| `RULE.txt` | 事前登録規則（計測前に固定。編集していない） |
| `ab_patch_{jpw1,cap4,cap8}.diff` | A/B 腕の差分（各 1 ファイル 1 行。計測用の複製ツリーにだけ当て、main へは当てていない。sha256 は `<machine>/patch_sha256.txt`） |
| `orchestrate.sh` | 機体名（`macmini`／`gb10`）を引数にした一括実行（`--dry-run` あり。bash 3.2 互換。負荷ゲート閾値は Mac mini 5.0・GB10 10.0） |
| `aggregate.py` | fail-closed 集計（`--self-test` あり。欠損・非数値・checksum 不一致・`rayon_threads` の食い違い・計数の run 間不一致を検出） |
| `aggregate.md` | `python3 -I aggregate.py > aggregate.md` の出力（手で直さない） |
| `{macmini,gb10}/` | 実測ログ（マスク済み）。`.rev-stamp`・`env_info.txt`・`uname.txt`・`load_gate.log`・`tree_verify.txt`・`patch_sha256.txt`・`build-*.log`・`count/run{1..5}`・`st/run{1..5}`・`serial_ref/run{1..5}`・`ab/run{N}-{arm}`・`DONE` |

## 手順

1. 転送元（origin/main に本ディレクトリの patch・スクリプトを加えたコミット）の HEAD とツリー指紋（`git archive` の sha256）を `.rev-stamp` に書き、
   `~/work/jobgrid-3177/base`（専用ディレクトリ。共有の `rust-ai-library-run` を `--delete` で壊さないため）へ rsync（`.gitignore` フィルタ・`.env*`・`*.local.md` 除外）。
2. 各機体で `BASE=…/base WORK=…/work bash base/docs/perf/logs/cpu-jobgrid-3177/orchestrate.sh <machine>`。
   Mac mini は `zsh -lc` 経由の `nohup` サブシェル、GB10 は PATH を明示した `setsid nohup`。2 機体は別マシンなので並行起動した（機体内は逐次）。
3. 計測後にログを rsync で回収し、`build-*.log` のホーム絶対パスを `<home>` へ置換して収録（`count/*.raw.log` と `env_info.txt` は実行時にマスク済み）。

## 状態（正式系列）

- 両機体とも負荷ゲートを全 35 プロセスが通過（`load_gate.log`）。再実行・差し替えはしていない（`orchestrate-stdout.log` は終了コード 0 の「完了」）。
- 計測バイナリの腕ツリーでは `cargo test` を実行していない（cap 腕の下限テストは構造上落ちるため）。計数は base ツリーの `--lib` テスト。
- 逸脱: なし（規則の解釈変更・再実行なし）。ただし次を事実として残す。
  - Mac mini の最初のゲート通過は load1 4.5（直前のビルド由来。閾値 5.0 未満）で、全プロセスが約 5 秒で完走した。run1 の単スレッド値が全形状で低い。
  - GB10 は共有ノードで load1 約 3.6 が常在（閾値 10.0 未満）。単スレッド L1 fwd は 36／105 GFLOPS の 2 水準に割れた（コア種との関係は未検証）。
- `.rev-stamp` の 1 行目は計測した複製元の HEAD、2 行目はそのツリーの指紋。ノード側の `base` は同ツリーを rsync したもの（腕ツリーとの差分が 1 ファイルだけであることは `tree_verify.txt`）。
