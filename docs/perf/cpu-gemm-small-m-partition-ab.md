# m が小さい GEMM の分割方式 A/B の結論（イシュー #3201）

## 1. 目的と結論

MLP（784→256→10・batch 64）の forward／backward は m の小さい細長 GEMM になる。#3199 の G1（job 統合と A の pack 共有）、
#3200 の G2（常駐 W の事前 pack キャッシュ）、既存の GB10 affinity を、両機体の実機で A/B した。判定規則は計測前に
固定した [`RULE.txt`](logs/cpu-small-m-gemm-ab-3201/RULE.txt)・[`RULE-addendum.txt`](logs/cpu-small-m-gemm-ab-3201/RULE-addendum.txt)
で、閾値・セル・ガードセルの扱いは変えていない。

**結論: 検証した全腕（Mac mini 2 腕・GB10 5 腕）が REJECT**。正式系列（全腕 5/5 round が負荷ゲート pass）・機構発火の前提成立・
checksum 完全一致・bit 一致テスト 24 件 pass のもとでの判定であり、判定不能や未計測の腕はない。既定は反転しない
（反転の要否は #3227 の担当。本 PR は既定を変えていない）。

| 事実 | 内容 |
|---|---|
| G1（merge）は forward 形状で遅い | `l1_fwd`・`l2_fwd` は両機体で 1.15〜1.98 倍遅い（Mac mini 1.153・1.982、GB10 1.300・1.613） |
| G1 は dW／dX では中立〜やや速い | GB10 の `l1_dw` 0.926・`l1_dx` 0.982。Mac mini は `l1_dw` 0.958・`l1_dx` 1.002 で境界 |
| G2（prepack）の上乗せは小さい | merge+prepack／merge は 0.85〜1.06。常駐 W を通る infer/reuse で Mac mini 0.880・GB10 0.945 |
| train／infer の 4 セルは両機体で悪化 | merge 系は GB10 の train/fresh で 1.20 前後、infer で 1.15〜1.30 |
| GB10 affinity は単体 6 形状を大きく改善 | off-aff／off は 0.54〜0.90。ただし train/fresh が 1.065 で、規則（全セル 1.00 以下）により REJECT |

## 2. 計測条件

- 規則の固定: `RULE.txt` の `fixed-at: 2026-10-23`。判定条件・閾値・round 数は雛形から変えていない。
- 計測コミット: `.rev-stamp`（`logs/cpu-small-m-gemm-ab-3201/{macmini,gb10}/.rev-stamp`）の 1 行目
  `2ba22c432882e3c89a2de45cd499fa3dadf4742e`、ツリー指紋（`git archive` の sha256）`e7203fcc250277293b205dc896a816437a6f5ed1a0088522072d06f0bc4fe1b1`。
  `crates/` は origin/main `9e5b4d23f` と同一で、腕ツリーの定数反転は計測専用の複製に patch を当てて作った（main へはコミットしない）。
- 機体: Mac mini（Apple M4・10 コア。負荷ゲート load1 < 3.0。計測時の最大 2.60）、GB10（20 コア。load1 < 10.0。最大 9.21）。
  どちらも 5 round すべて `gate=pass`（`load_gate.log`）。
- セル: 単体 6 形状（`cpu_jobgrid_bench`。正方 512 は参照用で判定に使わない）と、framework-compare の train／infer（size=64）×
  fresh／reuse の 4 セル。各セル 5 round の中央値。腕の起動順は round ごとにローテーションし、腕ごとに独立プロセスで測った
  （`ORDER` 行）。同じ round の中で PyTorch CPU（fresh のみ）も測った。
- 腕の構成は `tree_verify.txt`・`patch_sha256.txt` で検証した。各腕ツリーと BASE のファイル一覧は一致し、変更ファイルは
  patch 対象の 1〜3 ファイルと完全一致する。`bench-fandhe` は腕ごとの crates/facade への path patch を `cargo tree`
  で確認した（`tree-<arm>.txt`）。

## 3. 結果（中央値 µs と比＝腕／off）

表の全体と勝ち run 数は [`aggregate.md`](logs/cpu-small-m-gemm-ab-3201/aggregate.md)（`python3 -I aggregate.py` の出力）。
ここには判定の根拠になった比を抜粋する。1.00 超が REJECT の根拠。

### 3.1 Mac mini

| セル | off（µs） | merge | merge+prepack |
|---|---|---|---|
| l1_fwd | 163.1 | 1.153 | 1.088 |
| l1_dw | 158.3 | 0.958 | 1.006 |
| l1_dx | 161.9 | 1.002 | 0.938 |
| l2_fwd | 25.3 | 1.982 | 1.954 |
| l2_dw | 23.0 | 1.191 | 1.129 |
| l2_dx | 26.5 | 1.447 | 1.228 |
| train/fresh | 693.3 | 1.015 | 1.024 |
| train/reuse | 677.0 | 0.997 | 1.056 |
| infer/fresh | 208.2 | 1.125 | 1.092 |
| infer/reuse | 205.2 | 1.111 | 0.978 |

### 3.2 GB10

| セル | off（µs） | merge | merge+prepack | off-aff | merge-aff | merge+prepack-aff |
|---|---|---|---|---|---|---|
| l1_fwd | 112.9 | 1.300 | 1.331 | 0.825 | 0.686 | 0.701 |
| l1_dw | 136.5 | 0.926 | 0.895 | 0.648 | 0.628 | 0.661 |
| l1_dx | 109.4 | 0.982 | 0.993 | 0.822 | 0.607 | 0.588 |
| l2_fwd | 24.1 | 1.613 | 1.695 | 0.544 | 0.649 | 0.650 |
| l2_dw | 23.4 | 0.999 | 1.009 | 0.898 | 0.526 | 0.528 |
| l2_dx | 32.3 | 0.974 | 0.960 | 0.720 | 0.486 | 0.484 |
| train/fresh | 957.7 | 1.214 | 1.204 | 1.065 | 1.070 | 1.161 |
| train/reuse | 848.6 | 1.006 | 1.035 | 0.957 | 1.010 | 1.007 |
| infer/fresh | 174.4 | 1.295 | 1.294 | 0.786 | 0.804 | 0.754 |
| infer/reuse | 187.6 | 1.212 | 1.145 | 0.803 | 0.748 | 0.718 |

## 4. 判定（腕ごと・機体ごと）

| 機体 | 腕 | 判定 | 比が 1.00 超のセル |
|---|---|---|---|
| Mac mini | merge | **REJECT** | l1_fwd・l1_dx・l2_fwd・l2_dw・l2_dx・train/fresh・infer/fresh・infer/reuse |
| Mac mini | merge+prepack | **REJECT** | l1_fwd・l1_dw・l2_fwd・l2_dw・l2_dx・train/fresh・train/reuse・infer/fresh |
| GB10 | merge | **REJECT** | l1_fwd・l2_fwd・train/fresh・train/reuse・infer/fresh・infer/reuse |
| GB10 | merge+prepack | **REJECT** | l1_fwd・l2_fwd・l2_dw・train/fresh・train/reuse・infer/fresh・infer/reuse |
| GB10 | off-aff | **REJECT** | train/fresh（1.065） |
| GB10 | merge-aff | **REJECT** | train/fresh（1.070）・train/reuse（1.010） |
| GB10 | merge+prepack-aff | **REJECT** | train/fresh（1.161）・train/reuse（1.007） |

未計測・判定不能の腕はない。FAIL（checksum 不一致）もない。

## 5. 確認項目

- checksum: 単体 6 形状＋正方 512 は全腕・全 round で同一の `checksum=0x…`。train／infer は (task, mode) ごとに全腕・全 round
  で完全一致。両機体とも不一致 0 件（`aggregate.md` の「checksum 完全一致」）。
- bit 一致: `small_m::tests`・`prepack::tests` の 24 件（+ ignored 1）が両機体で pass（`bitmatch.log`。ゲート OFF のまま
  `_with(true)` 経路を強制するテスト群。BASE ツリーで実行）。ops 層の `ops::resident_w_prepack_tests` はフィルタ誤記
  （`ops_resident_w_prepack_tests` はモジュールパスと不一致）で両機体の `bitmatch.log` には含まれない。フィルタを
  `orchestrate.sh` で是正したうえで、開発機（Mac）で同 3 スイートを再実行し 32 件 pass（+ ignored 2）を確認した
  （ops 層 8 件 pass＋ignored 1。`bitmatch-ops-local.log`）。両機体の実測ログ自体は再計測していない。
- 機構発火の前提（`prereq/<arm>.log`）: G1 は merge 系の腕で適格形状の行が `gate=true` かつ `a_shared_bands == row_bands`、
  OFF 系で全行 `gate=false`。G2 は prepack 系の腕で 4 入口すべて `builds=1`・`hits>=1`、それ以外で `cache_len=0`。
  GB10 の affinity は `EXPECT_GB10_AFFINITY_POOL_ACTIVE=1／0` で期待どおり（`enabled`・専用プール有効・大コア 10 個）。
- `rayon_threads` は腕内で一定（Mac mini 10、GB10 20）。`RAYON_NUM_THREADS` は unset で、設定されていれば orchestrate.sh が拒否する。

## 6. 参考値（判定に使わない）

### 6.1 #3199（再現）と #3200（工夫）の効果の分離

- 再現（merge／off）: 1 節の表のとおり forward 形状で悪化。A の pack を共有して job 数を減らす利得より、行帯単位の共有
  （共有 pack のための同期・並列度低下）の損失が大きい形状がある、というのが数値の示す事実。原因の切り分けは未実施。
- 工夫（merge+prepack／merge）: Mac mini 0.85〜1.06、GB10 0.95〜1.05。常駐 W を通る infer/reuse が最も良い
  （Mac mini 0.880・GB10 0.945）。単体 6 形状は生のスライスを渡すため G2 は構造上発火せず、merge+prepack と merge は
  同一の挙動になる（差は計測ばらつき）。train/fresh・infer/fresh で G2 が発火したかは本計測では確認していない
  （事前確認は report テストの入口のみ）。

### 6.2 GB10 affinity との相互作用（affinity 単独の再判定ではない）

X-aff／X 比は単体 6 形状で 0.38〜0.90（off-aff/off のみでは 0.54〜0.90、mergeprepack-aff/mergeprepack の l2_fwd が最小 0.384）と大きく改善する。merge 系と組み合わせた方が off-aff より改善幅が大きいセルが多い
（例: `l1_fwd` off-aff 0.825 に対し merge-aff／merge 0.527）。一方 train/fresh は off-aff／off 1.065 で、train 系 4 セルの
判定を落としている。affinity 単独は #2117 で REJECT 済みで、その判定は変えない。

### 6.3 PyTorch 比（PyTorch 中央値 ÷ fandhe 中央値。1.00 以上で fandhe が同等以上）

train／infer の 4 セル。単体 6 形状の PyTorch 比は未計測（RULE-addendum 3）。

| 機体 | セル | off | merge | merge+prepack | off-aff | merge-aff | merge+prepack-aff |
|---|---|---|---|---|---|---|---|
| Mac mini | train/fresh | 0.174 | 0.171 | 0.170 | — | — | — |
| Mac mini | train/reuse | 0.178 | 0.178 | 0.168 | — | — | — |
| Mac mini | infer/fresh | 0.155 | 0.138 | 0.142 | — | — | — |
| Mac mini | infer/reuse | 0.157 | 0.142 | 0.161 | — | — | — |
| GB10 | train/fresh | 0.397 | 0.327 | 0.330 | 0.373 | 0.371 | 0.342 |
| GB10 | train/reuse | 0.448 | 0.445 | 0.433 | 0.468 | 0.444 | 0.445 |
| GB10 | infer/fresh | 0.753 | 0.581 | 0.582 | 0.958 | 0.936 | 0.998 |
| GB10 | infer/reuse | 0.700 | 0.578 | 0.611 | 0.872 | 0.935 | 0.975 |

（Mac mini に affinity 腕はない。）0.11.0 時点の本リポ記録（Mac mini infer 0.25・train 0.20、GB10 infer 0.60・train 0.35）と
比べ、本計測の off 腕は Mac mini でさらに低い（0.155〜0.178）。同一ホスト・同一 round 内の比だが、Mac mini は他用途と共有されうる
ホストで、torch の版（2.14.0）も異なる可能性があるため、絶対値の比較は参考に留める。

### 6.4 正方 512（参照・判定外）

GB10 で merge 0.727・merge+prepack 0.755（改善）、off-aff 1.074・merge-aff 1.143・merge+prepack-aff 1.202（悪化）。
Mac mini は 1.02 前後。本件の対象形状ではない。

## 7. #3227 への引き継ぎ

| ゲート | Mac mini | GB10 | 機体別の既定にできるか |
|---|---|---|---|
| G1 `SMALL_M_JOB_MERGE_ENABLED` | REJECT | REJECT | 両機体とも REJECT のため既定 ON の根拠なし。割れていない |
| G2 `RESIDENT_W_PREPACK_ENABLED`（G1 と併用） | REJECT | REJECT | 同上。G2 単独の腕は規則にない（#3200 の上乗せ分は 6.1） |
| GB10 affinity（X-aff） | 対象外 | REJECT（全腕 train 系のみが理由） | 単体 6 形状と infer 4 セルは大幅改善。train/fresh の後退が 6.5〜16.1%。反転の可否は #3227 で判断（規則の閾値は事後に動かさない） |

- 判定が割れたゲートはない。機体別既定（ブランド・コア数の検証済み型）の検討が要るのは affinity のみで、それは既に GB10 専用の
  ゲートである。
- G1 の悪化は `l1_fwd`・`l2_fwd`（m=64 の forward）に集中し、dW／dX は中立〜改善。形状別に G1 を絞る案（選択条件の見直し）は
  実装変更を伴うため本 PR の対象外。要否は #3227 または新規イシューで決める。

## 8. 逸脱・未計測・制約

- 初回起動は事前確認の検査（`probe=` 行の取り出し）の不具合で、計測を始める前に停止した。計測は 1 件も行っていない。
  `orchestrate.sh` を修正（commit `2ba22c432`）して新しいコミットから最初からやり直した。本結論の数値はすべて修正後の 1 回の系列で、
  run の差し替え・追加はない。RULE.txt・RULE-addendum.txt・判定式・閾値は不変（`RULE.txt` の「計測コミット」は
  `.rev-stamp` で同定する運用のとおり修正後の HEAD）。
- `compare_gemm_ab.py` による train／infer の照合は実施していない（`aggregate.py` が同じ 5 回中央値・checksum 契約で判定）。
- 計測は共有ホスト（Mac mini の最大 load1 2.60 は閾値 3.0 に近い）で、負荷ゲートの範囲内。ばらつきは `aggregate.md` の勝ち run 数で確認できる。
- tolerance・FMA 契約・parity baseline は変更していない。
