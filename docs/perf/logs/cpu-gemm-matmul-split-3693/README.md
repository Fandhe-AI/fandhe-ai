# #3693: matmul_resident の後退（T2）を matmul 区間の内側で切り分ける診断

**状態: 未実測（部品の実装まで。実機計測は未実施）**。GB10・Mac mini とも実測値は無く、他機体（開発機の M4 Max・Linux）の値で代替していない。結論の欄は「未実測」のまま。
背景・事実・限界は `docs/perf/cpu-gemm-reuse-fresh-inversion.md` §15。

## 構成

| ファイル | 役割 |
|---|---|
| `RULE.txt` | 事前登録規則（単独コミットで最初に固定。以後編集しない。固定時刻はそのコミットのコミット時刻） |
| `before-arm.patch` | `resident_input.rs` の呼び出し 1 行を `download_owned(out)` から `download(&out)` に戻す固定パッチ（before の木。main には入れない） |
| `make_before_tree.sh` | after の展開木に上のパッチを当てて before の木を作り、`.rev-stamp` を置く |
| `build_split_arms.sh` | 2 本の木の差が `resident_input.rs` 1 ファイルだけ・診断テストが同一であることを強制し、診断テストバイナリ `diag-before`／`diag-after` と `arms.txt` を作る（bench は作らない） |
| `orchestrate.sh` | 1 機体・1 系列。5 run・ABBA。`--first` で系列の先頭腕（木）を指定 |
| `session-scripts/session-3693.sh` | hw-lock・GB10 の poller と前後スナップショット・orchestrate・`excl_check.py`・参考判定 |
| `split_judge.py` | RULE を機械的に適用する判定器（標準ライブラリのみ・fail-closed・`--self-test` 付き） |

診断テストは `crates/facade/tests/cpu_gemm_resident_matmul_split_diag.rs`（CI で走る bit 一致テスト＋`#[ignore]` の実機診断 `cpu_gemm_resident_matmul_split_phases`）。

## 実行手順（実ホスト名は git 管理外の `.local.md` から引き、公開物には書かない）

1. after の sha を固定し `git archive` で after の木を作り、`.rev-stamp`（1 行目 sha・2 行目 `dirty=0`）を置く。`make_before_tree.sh <after木> <before木>` で before の木を作る。ツリー指紋を記録する
2. 各機体の `$HOME/work/r3693/{runner,before,after}` へ tar を ssh で流して展開する（DGX では openrsync の `--delete-excluded` の落とし穴があるため rsync は使わない）。ノード上に `*.local.md`・`.env*`・`settings.local.json` が無いことを `find` で確かめる
3. hw-lock を取る前に `build_split_arms.sh <before木> <after木> $HOME/work/r3693/bin` を 1 回実行する（計測中はビルドしない）
4. 系列を 30 分以上あけて切り離し起動する（GB10 は `setsid nohup`、Mac mini は `nohup` のサブシェル）。完了待ちの grep は `^done\. rc=` に行頭アンカーを付ける
   - `session-3693.sh <gb10|macmini> s1 before`・`s2 after`・`s3 before`・（ゲート不通過があれば）`s4 after`・`s5 before`
5. マスクして回収（実ホスト名・HOME・ユーザー名の grep が 0 件）し、`gb10/sN`・`macmini/sN` に置く。大きな生出力はコミットしない
6. `python3 -I split_judge.py --host macmini=<s1>,<s2>,<s3> --host gb10=<s1>,<s2>,<s3> --out judge-3693.md`（系列は起動順。手で書き換えない）

## 検証済み（実機なし）

- `python3 -I split_judge.py --self-test` が PASS（主因 1 つ・複数・特定できない・健全性 A／B 不成立・D_iter 非正・系列不足・checksum 不一致・欠損／重複・RULE sha 不一致・GB10 記録のみ）
- `before-arm.patch` を after 相当の木へ適用し、差が `resident_input.rs` の 1 行だけになること
- 診断テストの CI 部分（N=64・128 の bit 一致）と `--ignored` 実行の JSON 行の区間名が RULE と一致すること（手元の M4 Max での確認のみ。値は記録にも判定にも使っていない）
