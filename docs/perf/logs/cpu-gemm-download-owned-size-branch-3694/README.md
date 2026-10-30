# #3694: matmul_resident のサイズ分岐（D'）の A/B

## 状態（先に結論）

**実測済み・判定は REJECT**（事前登録規則 `RULE.txt` を機械的に適用した結果。`judge-3694.md`）。
GB10（専有機体）と Mac mini の各 3 系列（系列間 31 分以上・hw-lock・ゲート全通過）で測った。

- 3 対の判定: 対 1 = REJECT・対 2 = 判定不能・対 3 = REJECT。RULE の合成（1 対でも REJECT で REJECT）により **REJECT**
- REJECT の根拠（経路上セルの 5/5 run 後退）: 対 1 は Mac mini の bench `reuse iter_total` N=256、対 3 は GB10 の診断 `resident_reupload` N=256（いずれも `judge-3694.md` の「REJECT 条件に当たった項目」）
- T2（Mac mini 診断 `resident_public` N=256）の中央値 ratio（after/before）は 1.0020・0.9942・0.9939（対 1・2・3）。5 run の各値は `judge-3694.md` に名指しで載せた。N=256 は出力 262144 B で、サイズ分岐は after でも `download`（コピー）を選ぶため、D2H の呼び方は before と同じ（ただし a7aac2f0 から main までに backend-cpu・tensor-core へ #3672 以降の追加があり、ライブラリ全体が同一ではない）
- 処置セル（GB10 N=4096）は 3 対とも改善（診断・bench とも中央値 ratio 0.81〜0.84）。ADOPT を妨げたのは N=256〜1024 の経路上セルの中央値 ratio > 1.00 と、上記 2 件の 5/5 後退
- 結論は書き換えない。**修正（サイズ分岐）を本番に残すかは所有者の判断事項**。RULE の「結論後」節に従い、本番のサイズ分岐は本 PR に残さず（`crates/` の差分 0）、パッチ `size-branch.patch` と結果を残した

## 中身

| ファイル | 内容 |
|---|---|
| `RULE.txt` | 事前登録規則（単独コミットで固定済み。編集しない） |
| `size-branch.patch` | 本番のサイズ分岐（閾値 262144 B 以下は `download`・超えると `download_owned`）と単体・CPU bit 一致テストの差分（main には入れていない） |
| `make_before_tree.sh` | before の木（a7aac2f0 の木に計測ハーネスだけを after から置換）を作る。置換内容の指紋の決め方は下記 |
| `orchestrate.sh` | 1 機体・1 系列の計測（#3681 の写し。`--first` で先頭腕を系列ごとに交互・小 N の記録セルを別ループで測る） |
| `session-scripts/session-3694.sh` | hw-lock・GB10 の専有確認（poller・`excl_check.py`）・orchestrate・参考表示を直列に実行する |
| `judge_3694.py` | RULE「3 系列の合成」を適用する判定器（#3681 の `path_judge.py` を sha256 固定で読む。`--self-test` あり） |
| `judge-3694.md` | 判定器の出力（手で書き換えない） |
| `rev-stamp-verification-3694.md` | 腕の同定記録（木・指紋・バイナリ・規則 sha・系列の時刻と順・使わなかった系列） |
| `arms/{gb10,macmini}-arms.txt` | `build_ab_arms.sh` の出力（腕の rev-stamp・バイナリの sha256） |
| `gb10/s{1..3}`・`macmini/s{1..3}` | マスク済みの生ログ（各系列の `<機体>.tar.gz`〈`<機体>/` 配下の診断・bench・`small/` をまとめたもの〉と `steps.log`・GB10 は poller・`excl_status.txt`）。`<機体>/` は 1 系列 180 ファイル前後あり、PR が Cursor Bugbot の上限を超えたため系列ごとに tar.gz にした（中身は変えていない） |

小 N（64・128）の記録セル用に `crates/facade/tests/cpu_gemm_resident_reuse_diag.rs` へ `cpu_gemm_resident_reuse_phases_small_n`（`#[ignore]`）を足してある。既存の関数・反復数は変えていない。

## 実行手順（再現用）

1. `git archive` で after の木と a7aac2f0 の木を作り、それぞれ `.rev-stamp`（1 行目 sha・2 行目 `dirty=0`）を置く
2. `make_before_tree.sh <base_tree> <after_tree> <before_tree>` で before の木を作る。`.rev-stamp` 1 行目は `a7aac2f0…+harness-overlay(<12 桁>)`。12 桁は「置換した全ファイルを相対パスで `LC_ALL=C sort` し `<sha256>  <path>` の一覧全体を sha256 した値」の先頭（RULE に定義が無いため本書で定めた）
3. 各機体の `$HOME/work/r3694/{runner,before,after}` へ ssh の tar で展開し、ツリー指紋が一致することを確かめる（rsync は使わない）
4. hw-lock の前に `build_ab_arms.sh <before> <after> <bin>` を 1 回実行する（計測中にビルドしない）
5. `session-scripts/session-3694.sh <gb10|macmini> <label> <before|after>` を s1（before 先頭）・s2（after 先頭）・s3（before 先頭）で 31 分以上あけて切り離し起動する
6. 回収・マスク後に、各系列ディレクトリで `tar -xzf <機体>.tar.gz` を実行して `<機体>/` を展開し、`python3 -I judge_3694.py --host gb10=gb10/s1/gb10,gb10/s2/gb10,gb10/s3/gb10 --host macmini=macmini/s1/macmini,... --out judge-3694.md`

## 検証した事項

- 生ログを tar.gz にまとめた後、展開したファイル一覧がまとめる前と一致し、展開した木から `judge_3694.py` を再実行した出力が `judge-3694.md` と完全に一致した（`diff` が空）

- `bash -n`・`shellcheck`（3 スクリプト）・引数の拒否ケース（相対パス・不正な機体名・不正な label・不正な `--first`）・`judge_3694.py --self-test`（3 対 ADOPT／1 対 REJECT／混在／系列不足／REJECT 優先／ゲート不通過を飛ばして次を使う／各 sha 不一致・`rule_3681_sha256` 欠落・rev-stamp 不正で入力不備／小 N 欠落でも判定不変）
- サイズ分岐を当てた状態で `cargo test -p fandhe-ai-autodiff --lib size_branch`・`cargo test -p fandhe-ai --test resident_input_inference`（閾値を超える形状 (256, 8, 257) を含む CPU の bit 一致）・`cargo clippy -p fandhe-ai-autodiff -p fandhe-ai --all-targets -- -D warnings` が通った
- ローカル（M4 Max。値は使わない）で両腕のビルドとセル集合の一致を確認した（`rev-stamp-verification-3694.md`）
- 両機体の木の指紋・バイナリの sha256・`env_info.txt` の `rule_sha256`／`rule_3681_sha256` を照合した

## 限界・未検証

- 判定器は計測後に表示名の関数だけを足した版で出力した（判定ロジックは不変。`rev-stamp-verification-3694.md`）
- 1 回目の GB10 は専有機体ではない機体に誤投入し、ゲート不通過（`excl=fail`）のため判定にも記録にも使っていない。専有機体で最初からやり直した
- 小 N（N=64・128）は記録のみ（`judge-3694.md` の別表）。判定に使っていない
- 他機体（M4 Max・Linux）の値で代替していない。ここでの REJECT は RULE の式（経路上セルの 5/5 run 後退）を機械的に適用した結果で、因果（サイズ分岐が N=256 を遅くした）を示すものではない。N=256 では D2H の呼び方が before と同じため、後退は run 間のばらつきか、a7aac2f0 以降のライブラリ差の可能性があるが、本計測ではそれらを分離していない
- 直し方の再提案・RULE の見直しは所有者判断
