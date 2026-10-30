# #3701: T2 の主因 S1（gemm_device）の内側を分けて測る（計測専用パッチ）

**状態: 部品まで実装済み・実機は未実測**（部品のコミットまで。Mac mini・GB10 とも `judge-3701.md` はまだ無い。他機体や M4 Max の値では代替しない）。
背景・コード読解・限界は `docs/perf/cpu-gemm-reuse-fresh-inversion.md` §16。規則は `RULE.txt`（部品より前に単独コミットで固定。以後編集しない）。
イシュー本文の所有者承認（2026-10-30）の記載は、本ディレクトリでは検証済みの承認としては扱わない。

## 構成

| ファイル | 役割 |
|---|---|
| `RULE.txt` | 事前登録規則（単独コミットで最初に固定。固定時刻はそのコミットのコミット時刻） |
| `probe-3701.patch` | 計測専用パッチ。backend-cpu に `split_probe.rs`（thread_local の受け渡し）と `gemm_device`・`download_owned`・`download` の計時点を足し、診断テストの `probe_take()` の本体を置き換える。**main には入れない**（main の本番コードに計時点は無い） |
| `before-arm.patch` | #3693 と同じ固定パッチ（`resident_input.rs` の 1 行を `download_owned(out)` から `download(&out)` に戻す。#3693 のものと逐語で同一） |
| `make_after_tree.sh` | runner の木に `probe-3701.patch` を当てて after の木を作る。入力に `split_probe.rs` が無いことを先に検査する |
| `make_before_tree.sh` | after の木に `before-arm.patch` を当てて before の木を作る（#3693 の複製） |
| `build_split_arms.sh` | ハード条件 4 つ（差が `resident_input.rs` 1 ファイルだけ・診断テストが同一・両 `.rev-stamp` が patch の sha256 先頭 12 桁を含む・両木に `split_probe.rs`）を強制して `diag-before`／`diag-after` を作る |
| `orchestrate.sh` | 1 機体・1 系列。5 run・ABBA |
| `session-scripts/session-3701.sh` | hw-lock・GB10 の poller・orchestrate・`excl_check.py`・参考判定 |
| `split_inner_judge.py` | RULE を機械的に適用する判定器（標準ライブラリのみ・fail-closed・`--self-test` 付き。段 1・段 2 の 2 段） |

診断テストは `crates/facade/tests/cpu_gemm_s1_inner_split_diag.rs`（CI で走る bit 一致テスト＋`#[ignore]` の実機診断 `cpu_gemm_s1_inner_split_phases`）。
main では `probe_take()` が常に `None` を返すスタブで、実機診断はパッチ無しなら panic する（fail-closed）。

## #3693 との差分

| 項目 | #3693 | #3701 |
|---|---|---|
| 木 | before／after（差は 1 行） | runner・after（runner + probe）・before（after + 1 行）。after と before の差は #3693 と同じ 1 行 |
| 診断テスト | `cpu_gemm_resident_matmul_split_diag.rs`（変更しない） | 複製して `probe_take()` と内側の区間の記録を足す |
| 区間 | 外側のみ | 外側 + 内側（`gd_*`・`do_*`・`dc_*`） |
| 判定器 | `split_judge.py` | `split_inner_judge.py`（段 2 を追加。内側の区間の欠落・木と経路の不整合は入力不備） |
| 判定の閾値 | k >= 12・系列 3・係数 0.5 | 同じ（新しい閾値は作らない） |

## 実行手順（実ホスト名は git 管理外の `.local.md` から引き、公開物には書かない）

1. 部品のコミットを runner の sha に固定し、`git archive` で runner の木を作って `.rev-stamp`（1 行目 sha・2 行目 `dirty=0`）を置く
2. `make_after_tree.sh <runner木> <after木>` → `make_before_tree.sh <after木> <before木>`。ツリー指紋を記録する
3. 各機体の `$HOME/work/r3701/{runner,before,after}` へ tar を ssh で流して展開する（DGX では rsync の `--delete-excluded` の落とし穴があるため rsync は使わない）。ノード上に `*.local.md`・`.env*`・`settings.local.json` が無いことを `find` で確かめる
4. hw-lock を取る前に `build_split_arms.sh <before木> <after木> $HOME/work/r3701/bin` を 1 回実行する（計測中はビルドしない）
5. 系列を 30 分以上あけて切り離し起動する（GB10 は `setsid nohup`、Mac mini は `nohup` のサブシェル）。完了待ちの grep は `^done\. rc=` に行頭アンカーを付ける
   - `session-3701.sh <gb10|macmini> s1 before`・`s2 after`・`s3 before`・（ゲート不通過があれば）`s4 after`・`s5 before`
6. マスクして回収（実ホスト名・HOME・ユーザー名の grep が 0 件）し、`gb10/sN`・`macmini/sN` に置く
7. `python3 -I split_inner_judge.py --host macmini=<s1>,<s2>,<s3> --host gb10=<s1>,<s2>,<s3> --out judge-3701.md`（系列は起動順。手で書き換えない）

## 本番が不変であることの機械検査

- `git diff --name-only origin/main...HEAD` の `crates/` 配下は `crates/facade/tests/cpu_gemm_s1_inner_split_diag.rs` の 1 ファイルだけ
- `git diff origin/main...HEAD -- crates/backend-cpu crates/tensor-core crates/autodiff` は空

## 検証済み（実機なし）

- `python3 -I split_inner_judge.py --self-test` が PASS（S1a・S1b・S2m の単独主因・複数併記・段 1 で S1 主因かつ段 2 で特定できない・特定できない・健全性 A／B 不成立・D_iter 非正・系列不足・checksum 不一致・欠損／重複・RULE sha 不一致・内側の区間の欠落・木と経路の不整合・GB10 記録のみ）
- `probe-3701.patch` を HEAD の展開木へ適用し、`cargo clippy -p fandhe-ai-backend-cpu -p fandhe-ai --all-targets -- -D warnings`・`cargo test -p fandhe-ai-backend-cpu --lib download`・診断テストの CI 部分が通ること
- パッチ適用後の `--ignored` 実行の JSON 行の区間名が RULE と一致すること（手元の M4 Max での確認のみ。値は記録にも判定にも使っていない）
- パッチ無しの木で `--ignored` を実行すると panic すること
- `make_after_tree.sh` → `make_before_tree.sh` → `build_split_arms.sh` をローカルで通し、split_probe.rs の無い木を before に渡すとハード条件違反で中止すること

## 未実測

- Mac mini（判定対象）・GB10（記録のみ）とも、実機での 3 系列は未実行。`judge-3701.md`・`rev-stamp-verification-3701.md`・`macmini/`・`gb10/`・`arms/` は実測後に追加する
