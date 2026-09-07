# REQ-2 スケール付き絶対誤差救済項・比較対象側 fail 判定不能規定の spec 提案 draft

> **本文書は spec リポジトリ（Fandhe-AI/fandhe-ai-spec）への REQ-2 改定提案の draft であり、
> 正本ではない。** `RELATIVE_TOLERANCE`／`ABSOLUTE_RESCUE_THRESHOLD`（`crates/backend-cpu/src/parity.rs:32,37`）・
> `PARITY_REL_TOL`／`PARITY_ABS_TOL`（`scripts/bench/framework-compare/bench-common/src/parity.rs:49,54`）・
> 判定式（`compare`／`element_error`／`_parity_check`）・`BASELINES`・`docs/spec/`（正本 submodule）は
> 本 draft の作成にあたって一切変更していない。
>
> **spec リポジトリへの実起票は、イシュー #1241 でのユーザー承認後に限る。** 承認前に
> `gh issue create -R Fandhe-AI/fandhe-ai-spec` を実行しない（本エージェントは実行していない）。
>
> 先例: `docs/spec-proposal-req2-req8-revision.md`（#580／#996。限定救済項〈案 1′〉の既存 spec 提案 draft）・
> `docs/cuda-tensor-core-parity-judgment-decision.md`（#1106 → spec PR #63 で反映済み）。

## 1. 位置づけ

イシューツリー #1234（ルート）→ Phase 1 親 #1235 配下 #1240 の成果物。#1236 配下の
机上評価（#1237『`docs/perf/candle-parity-tolerance-candidates.md`』・#1238
『`docs/perf/candle-parity-tolerance-baseline-impact.md`』）・決定記録 draft
（#1239『`docs/candle-parity-tolerance-contract-decision.md`』）を入力とし、本文書は
「spec 側へ提示する提案文」に閉じる。承認記録は #1241、承認後の実装（Phase 2）は
#1243 配下（#1245／#1252／#1258）が扱う。

## 2. spec 側 issue の起票用本文（そのまま貼り付け可能な形式）

以下は、#1241 でのユーザー承認後に `gh issue create -R Fandhe-AI/fandhe-ai-spec` へ
そのまま渡せる形で用意した起票用本文である。係数値・採否は #1241 で確定した値へ
起票時に差し替える（本 draft では未確定のためプレースホルダのまま残す）。

---

**タイトル案**:

```
docs(requirements): REQ-2 統一複合判定へのスケール付き絶対誤差救済項の追加と比較対象側 fail の「判定不能」規定を提案する（実装リポ Fandhe-AI/fandhe-ai#1234 提案）
```

**本文案**:

```markdown
## 背景

実装リポ（Fandhe-AI/fandhe-ai）の framework-compare GEMM ゲート（candle 比 5 回計測
中央値）で、N=2048・正方・入力 U[-0.5,0.5)・固定シードの条件において、比較対象側
（candle-core =0.11.0 の CUDA cuBLAS／CPU gemm crate 経路）出力が REQ-2 統一複合判定
（相対誤差 1e-3 未満 または 絶対誤差 1e-5 未満）を各 2/4,194,304 要素外れ、5 run とも
完全に決定的に「判定不能」となることが確認された（実装リポ
`docs/perf/cuda-gemm-candle-gate-remeasurement.md` §5.1・§11.4、
`docs/perf/cpu-gemm-candle-gate-remeasurement.md` §5.2・§12.3）。fandhe-ai 側は同条件で
全 run `parity_fail_count=0`。

原因は実装リポイシュー #1184（同 cuda doc §5.3 追記）が fail 4 要素の実値・厳密真値
突合で特定済み: 参照実装（k 昇順 f32 FMA 逐次累積）・candle 側カーネルのどちらの丸め
誤差も通常の累積誤差水準（`√K・ulp(max|partial|)` オーダー、比 0.08〜0.85 倍）に収まって
いるが、行列積のキャンセレーションにより最終値（`exact`）が部分和絶対値最大
（`max|partial|`）の 1/400〜1/2600 まで縮小する。この桁縮小により、通常水準の丸め誤差
フロアが最終値そのものと同程度の大きさになり、相対誤差 1e-3・絶対誤差 1e-5 の両閾値を
同時に割り込む。片側の実装が恒常的に他方より誤差が大きいという構造はない
（4 要素中 2 要素は参照実装側、2 要素は candle 側の誤差がそれぞれ大きい）。

## 提案 (a): 統一複合判定へのスケール付き絶対誤差救済項の OR 追加

判定式へ次の項を OR 追加する（既存 2 条件・既存定数は不変。単調性により既存 pass 要素は
pass のまま）:

```
pass ⇔ rel < 1e-3 ∨ diff < 1e-5 ∨ diff <= c・u・K・S_A・S_B
```

（`u = 2^-24`〈unit roundoff〉・`S_A = max|A|`・`S_B = max|B|`・`K` は内積長）

係数 `c` は実装リポでの机上評価で 2 案を検討中（#1241 で確定した値をここに反映）:
`c=0.5`〈bound 1.526e-05〉／`c=1.0`〈bound 3.052e-05〉（K=2048・`S_A・S_B=0.25` 時点）。
いずれも上記 fail 4 要素すべてを救済することを机上算出で確認済み
（実装リポ `docs/perf/candle-parity-tolerance-candidates.md` §4.3）。

**REQ-2 2026-09-02 追記との関係**: 本提案は当該追記の項 4（判定式・定数は当該追記では
変更しない）を変更するものではなく（既存 2 条件・既存定数は不変のまま OR 追加のみ）、
項 6（非 Tensor Core 経路〈f32 FMA〉は統一複合判定をそのまま適用する）の**改定**に該当
する。既存の限定救済項（案 1′。CUDA Tensor Core 経路〈TF32／f16〉限定・`internal-diagnostics`
feature 限定）の拡張ではなく、f32 SIMT／FMA 経路（非 Tensor Core 経路）を対象とする別の
改定である。

**案 1′ との相違点**:

1. スケーリング則が異なる: 案 1′ は `√K` 形式（`τ・S_A・S_B・√K`）だが、本提案は線形 `K`
   形式（`c・u・K・S_A・S_B`）。案 1′ 同型の √K 形式（係数 `c=2.0`〈u=2^-24 相当〉）は本件の
   fail 4 要素を**1 件も救済しない**ことを机上算出で確認済み
   （実装リポ同 candidates.md §4.2「A-3」行）。観測された誤差フロアが `√K・ulp(max|partial|)`
   である一方 `max|partial|` 自体が `√K・S_A・S_B` オーダーであるため、実効的には K に
   ほぼ線形に効くことが線形形式を推す根拠。
2. 対象経路が異なる: 案 1′ は CUDA Tensor Core 経路（TF32／f16）限定だが、本提案は
   一般の f32 SIMT/FMA 経路（非 Tensor Core 経路）が対象。
3. 誤差の起源が異なる: 案 1′ は TF32 丸め起因の恒常的な不合格を救済するが、本提案は
   f32 FMA 経路のキャンセレーション起因で「たまたま」複合判定を割る形状・入力の組み合わせ
   に対する救済。

**昇格条件の継承（限界の明示）**: 本提案の根拠は K=2048・正方・U[-0.5,0.5)・固定シードの
**1 条件のみ**の fail 要素（実装リポ `candidates.md` §2.2 が明記する外挿禁止方針）である。
REQ-2 の 2026-08-29 追記が案 1′ に課す一般契約への昇格条件（同符号・強相殺・外れ値入力での
上界確認、`S_A・S_B` がテンソル全体スケールであり局所性を欠く限界）は、本提案にもそのまま
当てはまる。したがって本提案は「一般契約として即時確定」ではなく、案 1′ と同型の
**適用スコープ限定（呼び出し経路限定・opt-in）＋昇格条件**の構造を選択肢として提示する。

**適用スコープの選択肢**（実装リポ側での実装コスト・影響範囲は下記「実装リポ側の対応
issue」参照）:

- (a-1) 本体 `compare`／`assert_parity`／`ParityBaseline` を含む一般契約として反映する
- (a-2) framework-compare ハーネス（比較対象の妥当性検証）限定で反映する

いずれを採るかで spec 側の反映文言が変わるため、両案を併記する。

## 提案 (b): 比較対象側（candle 等）fail の「判定不能」規定

現状、「比較対象側（fandhe-ai ではない側）の fail を判定不能として扱う」という規定は
REQ-2 に存在しない。実装リポの `scripts/bench/framework-compare/compare_gemm_gate.py::_parity_check`
はどちら側の fail かを区別せず、`fail_count > 0` であれば一律「判定不能」として扱う実装に
なっている（比較対象・自社側いずれが fail しても同じ扱い）。この挙動はハーネス
（`compare_gemm_gate.py`・`summarize.py`・README「目標達成ゲート」節）と実装リポの
ゲート受け入れ条件（#1031／#1117）に閉じており、spec 側の規定ではない。

2 つの提案の形を示す（結論は出さない。spec 側・ユーザーが判断する論点として提示する）:

- (b-1) REQ-2（または REQ-8 のゲート運用注記）に「第三者比較対象が統一複合判定を外れた
  場合は、データ妥当性上の『判定不能』であり fandhe-ai 側の REQ-2 違反ではない」ことを
  明文化する
- (b-2) 比較対象の妥当性検証に限り、提案 (a) の OR 項をハーネス限定で適用することを認める

**spec 側へ投げる論点（結論を出さない）**: framework-compare の `bench-common::parity` の
定数（`PARITY_REL_TOL`／`PARITY_ABS_TOL`）は本体 `backend-cpu::parity` の定数値へ
ピン留めされている（`bench-common/src/parity.rs` 該当 doc comment。値のドリフトを
テストで検知する設計）。このハーネス限定への OR 追加が REQ-5 の「テスト許容誤差の単独
緩和」に該当するかどうかは、spec 側・ユーザーが判断すべき事項として明示的に問う。
該当するなら (b) にも spec 側の明文化が必要になり、該当しないなら (b-1) の明文化のみで
足りる可能性がある。

## 受け入れ基準への影響

- **REQ-2**: 2026-09-02 追記の項 4（判定式・定数不変）・項 6（非 Tensor Core 経路は統一
  複合判定のまま）の改定に該当する（`docs/spec/04-requirements.md` の該当行。
  submodule コミット `0ca67cd18e55ff7ea0cd480903a8e4a6fae86131` 時点で項 4 は該当行、
  項 6 は該当行に対応。内容一致で対応箇所を特定されたい）
- **REQ-5**: 許容誤差の変更は「REQ-2 改定とセットで人間承認を経てのみ可能」の既存規定に
  従う。本提案はその REQ-2 改定側の提案である
- **REQ-8**: CUDA 行の限定条件「REQ-2 改定〈spec #56〉解決後に再確認」は Tensor Core
  経路（TF32／f16）に関する別の改定を指しており、本提案（f32 SIMT 経路対象）が解消する
  ものではない
- **REQ-7**（PyTorch 参照値比較の別指標）は対象外・不変

## 実装リポ側の対応 issue

ルート #1234 → Phase 1 #1235（#1236: #1237／#1238／#1239／#1240／#1241）→ Phase 2 #1243
（#1245／#1252／#1258）。原因分析: #1184。ゲート未達成確定: #1031（CUDA）・#1117（CPU）。
fail 要素ダンプ計装: #1183。

## スコープ境界

本提案に含まれないもの:

- spec リポジトリへの実起票そのもの（#1241 承認後に実施）
- tolerance 契約の実装・`BASELINES` の実機再測定（Phase 2 の対象）
- 他形状・他シードでの候補判定の追加机上計算（外挿は行わない方針。#1237 §2.2）
- burn/cpu（fail=5）の実値取得（実装リポ側で対象外と明記済み）

## 添付文書

- 実装リポ main 上の本提案 draft: `https://github.com/Fandhe-AI/fandhe-ai/blob/main/docs/spec-proposal-req2-candle-parity-tolerance.md`
- 決定記録 draft（候補比較・推奨案・ユーザー承認待ち事項）: `https://github.com/Fandhe-AI/fandhe-ai/blob/main/docs/candle-parity-tolerance-contract-decision.md`
```

---

## 3. 背景の事実（出典付き）

### 3.1 N=2048 判定不能の決定性

- 形状: N=2048（正方行列。M=N=K=2048）。入力: xorshift64\* 同一シード・同一生成式
- 対象: `candle-core =0.11.0` の CUDA GEMM 経路（`bench-candle gemm cuda 2048 fresh`）
- 実測値: `parity_fail_count=2`・`parity_total=4194304`・`parity_max_abs_err=3.623962e-05`・
  `parity_max_rel_err=2.811288e-01`
- 正式系列・参考系列それぞれ 5 run・計 10 run すべてで上記 4 値が一致（完全に決定的）。
  v0.6.0 横並び再計測（環境 10。別セッション・別バイナリビルド）とも一致
- fandhe-ai 側（reuse）は N=2048 で `parity_fail_count=0`（10 run 全件）

出典: `docs/perf/cuda-gemm-candle-gate-remeasurement.md` §5.1・§5.2・§11.4・§12.5、
`docs/perf/cpu-gemm-candle-gate-remeasurement.md` §5.2・§12.3。

### 3.2 #1184 の実値表（要約転記）

`FRAMEWORK_COMPARE_PARITY_DUMP=1` により GB10 実機で取得した fail 4 要素の厳密真値突合
（`docs/perf/logs/cuda-gemm-candle-parity-1184/truth-2048.txt`）:

| idx | device | row,col | exact | ref | actual | \|ref−exact\| | \|actual−exact\| | max\|partial\| | √K・ulp(max\|partial\|) |
|---|---|---|---|---|---|---|---|---|---|
| 13850 | cuda | 6,1562 | 2.166853e-03 | 2.168937e-03 | 2.157688e-03 | 2.084e-06 | 9.165e-06 | 3.969 | 1.079e-05 |
| 4130484 | cuda | 2016,1716 | 9.197426e-03 | 9.188101e-03 | 9.199142e-03 | 9.325e-06 | 1.717e-06 | 6.099 | 2.158e-05 |
| 1372466 | cpu | 670,306 | 9.918718e-03 | 9.920587e-03 | 9.933233e-03 | 1.869e-06 | 1.452e-05 | 5.613 | 2.158e-05 |
| 1633751 | cpu | 797,1495 | 5.382382e-03 | 5.374012e-03 | 5.385637e-03 | 8.370e-06 | 3.255e-06 | 4.748 | 2.158e-05 |

結論（同ドキュメント §5.3 追記）: 4 要素すべてで `|ref−exact|`・`|actual−exact|` は
`√K・ulp(max|partial|)` と同水準（比 0.08〜0.85 倍）にあり、通常の累積丸め誤差の範囲に
収まっている。片側優位の構造はない。最終値（`exact`）は部分和絶対値最大
（`max|partial|`）の 1/400〜1/2600 まで縮小しており、「0 近傍」の実体はこのキャンセレー
ションによる相対的な桁縮小である。

出典: `docs/perf/cuda-gemm-candle-gate-remeasurement.md` §5.3 追記（イシュー #1184）。

### 3.3 fandhe-ai 側は 0 fail（正式系列・参考系列とも）

正式系列 `fandhe-ai =0.7.0`・参考系列いずれも、N=1024/2048/4096 の CUDA GEMM ゲート
全 run で `parity_fail_count=0`。CPU GEMM ゲートも同様。

出典: `docs/perf/cuda-gemm-candle-gate-remeasurement.md` §11.4・§12.5、
`docs/perf/cpu-gemm-candle-gate-remeasurement.md` §12.3〜§12.4、
`docs/candle-parity-tolerance-contract-decision.md` §6 (b)。

### 3.4 「tolerance は緩めない」で据え置いてきた経緯

`docs/perf/cuda-gemm-candle-gate-remeasurement.md` §5.4 は「tolerance は緩めない」
方針のまま、本体の数値一致契約（相対誤差 1e-3 未満 または 絶対誤差 1e-5 未満）を不変
のまま適用し、N=2048 を「判定不能」のまま記録してきた。判定方式の変更は「ユーザー判断
事項」として列挙するに留め、当該イシューでは実施していない。

## 4. 提案 (a) の詳細（決定記録 draft からの転記・要約）

判定式（決定記録 draft §5 第一候補 A-1）: `pass ⇔ rel < 1e-3 ∨ diff < 1e-5 ∨ diff <= c・u・K・S_A・S_B`
（`u = 2^-24`・`S_A = max|A|`・`S_B = max|B|`・`K` 縮約長）。既存 2 条件・既存定数は不変、
OR 追加のみ。単調性により既存 pass 要素は pass のまま変わらない。

**係数 2 案**（K=2048・`S_A・S_B=0.25` 時点。出典: `candle-parity-tolerance-candidates.md` §4.3）:

| 案 | `c`（`u=2^-24`） | bound(K=2048) | N=512 | N=4096 |
|---|---:|---:|---:|---:|
| (i) 最小緩和 | 0.5 | 1.526e-05 | 3.8e-06（1e-5 比 0.38 倍） | 3.05e-05（1e-5 比 3.05 倍） |
| (ii) 2 倍余裕 | 1.0 | 3.052e-05 | 7.6e-06（1e-5 比 0.76 倍） | 6.10e-05（1e-5 比 6.10 倍） |

いずれも未承認であり、#1241 の判断を待つ（本 draft では確定しない）。

**代替案 B-2**（`t・ulp(Σ|ab|)`。決定記録 draft §5）: spec 側が案 1′（A-3 と同型の √K 形式）
に対して指摘する「`S_A・S_B` はテンソル全体スケールであり要素ごとの局所性を失う」という
限界に応える選択肢として残すが、参照実装側に O(MNK) の追加パスが要り、`BASELINES` では
上界代用のため全救済判定が確定できない。第一候補としては推さない。

**REQ-2 2026-09-02 追記との関係**: 本提案は追記項 4（判定式・定数不変）と**項 6（非 Tensor
Core 経路は統一複合判定のまま）の改定**に該当する。案 1′（限定救済項）の拡張ではなく、
f32 SIMT/FMA 経路を対象とする別の改定である。

**案 1′ との 3 つの相違**:

1. 線形 K vs √K（決定記録 draft §5「スケーリング則の注記」: √K 形式は本ダンプを 1 件も
   救済しない）
2. 一般 f32 経路 vs Tensor Core 限定
3. キャンセレーション起因フロア vs TF32 丸め起因

**昇格条件の継承（正直な限界の提示）**: 根拠は K=2048・正方・U[-0.5,0.5)・固定シードの
1 条件のみ（#1237 §2.2 の外挿禁止）。REQ-2 に既に書かれている「一般契約への昇格条件」
（同符号・強相殺・外れ値入力での上界確認、`S_A・S_B` のテンソル全体スケール限界）は本提案
にもそのまま当てはまる。したがって spec 側へは「一般契約として即時確定」ではなく、案 1′
と同型の**適用スコープ限定（呼び出し経路限定・opt-in）＋昇格条件**の構造を選択肢として
併記する。

**適用スコープの選択肢**（決定記録 draft §8-3 と対応）:

- (a-1) 本体 `compare`／`assert_parity`／`ParityBaseline` を含む一般契約
- (a-2) framework-compare ハーネス（比較対象検証）限定

spec 文言はどちらを採るかで変わるため両案の文案を §2 に併記した。

## 5. 提案 (b) の詳細

**現状**: 「比較対象側の fail → 判定不能」は spec に存在せず、
`compare_gemm_gate.py::_parity_check`（`scripts/bench/framework-compare/compare_gemm_gate.py:219-`）・
`summarize.py`・README「目標達成ゲート」節・#1031／#1117 受け入れ条件に閉じている
（本計画の grep で確認。spec は candle／framework-compare に言及しない）。

提案の形を 2 つに分けて提示する:

- (b-1) REQ-2（または REQ-8 のゲート運用注記）に「第三者比較対象が統一複合判定を外れた
  場合はデータ妥当性上の『判定不能』であり fandhe-ai 側の REQ-2 違反ではない」ことを
  明文化する
- (b-2) 比較対象の妥当性検証に限りハーネス限定の拡張判定（提案 (a) の OR 項）を認める

**spec 側へ投げる論点（結論を出さない）**: `bench-common` の定数は本体定数へピン留め
されている（`bench-common/src/parity.rs:1074` 付近の doc comment）ため、ハーネス限定の
OR 追加が REQ-5 の「テスト許容誤差の単独緩和」に該当するか否か。該当するなら (b) にも
spec 文言が必要、該当しないなら (b-1) の明文化のみで足りる可能性がある。この判断は
spec 側・ユーザー承認事項として明示的な問いとして記述する。

## 6. 受け入れ基準への影響

- **REQ-2**: 2026-09-02 追記項 4・項 6 の改定。統一複合判定本文
  （`docs/spec/04-requirements.md:74`。submodule コミット
  `0ca67cd18e55ff7ea0cd480903a8e4a6fae86131` 時点）。項 4 は同ファイル 90 行目
  「統一複合判定そのもの（判定式・`RELATIVE_TOLERANCE`・`ABSOLUTE_RESCUE_THRESHOLD` に
  相当する定数）は本追記では変更しない」、項 6 は同 92 行目「非 Tensor Core 経路
  （f32 FMA）は従前どおり統一複合判定をそのまま適用する（変更なし）」に対応する
  （行番号は上記 submodule コミット時点。内容一致で対応箇所を特定されたい）
- **REQ-5**: 許容誤差の変更は「REQ-2 改定とセットで人間承認を経てのみ可能」の規定
  （`docs/spec/04-requirements.md:140,145`）に従う。本提案はその REQ-2 改定側
- **REQ-8**: CUDA 行の限定条件「REQ-2 改定〈spec #56〉解決後に再確認」
  （`docs/spec/04-requirements.md:194-195`）は Tensor Core 経路に関する別の改定であり、
  本提案が解消するものではないと明記する
- **REQ-7**（PyTorch 参照値比較の別指標）: 対象外・不変

**未確定候補としての `diff` 形式文案**: 上記行番号・引用は本文書作成時点
（submodule コミット `0ca67cd18e55ff7ea0cd480903a8e4a6fae86131`）のものであり、
`docs/spec-proposal-req2-req8-revision.md` §1.6 と同形式の「そのまま転記できる差分文案」
ではない。実際の spec 側改定文言は spec リポ側での起票・レビュー時に確定する。

## 7. 実装リポ側の対応 issue 参照

- ルート: #1234
- Phase 1: #1235（配下 #1236: #1237／#1238／#1239／#1240／#1241）
- Phase 2（#1241 承認後）: #1243（配下 #1245／#1252／#1258）
- 原因分析: #1184
- ゲート未達成確定: #1031（CUDA）・#1117（CPU）
- fail 要素ダンプ計装: #1183

## 8. スコープ外・未変更事項の確認

本文書作成にあたり、`crates/`・`scripts/`・`.github/`・`docs/spec/` への差分は無い
（`git diff --stat origin/main -- crates/ scripts/ .github/ docs/spec/` は出力なし）。

対象外事項:

- spec リポジトリへの実起票（#1241 承認後に実施。本文書の作成では実行していない）
- tolerance 契約の実装・`BASELINES` 再測定・GB10 再計測（Phase 2: #1245／#1252／#1258）
- 他形状・他シードでの候補判定の追加机上計算（#1237 §2.2 の外挿禁止方針）
- burn/cpu（fail=5）の実値取得（実装リポ側で本イシューのスコープ外と明記済み）

## 9. 関連ドキュメント

- `docs/candle-parity-tolerance-contract-decision.md`（イシュー #1239。候補比較・推奨案・
  ユーザー承認待ち事項の決定記録 draft）
- `docs/perf/candle-parity-tolerance-candidates.md`（イシュー #1237。候補 A/B の fail 数
  机上算出）
- `docs/perf/candle-parity-tolerance-baseline-impact.md`（イシュー #1238。`BASELINES` への
  影響机上確認）
- `docs/perf/logs/cuda-gemm-candle-parity-1184/`（イシュー #1184。fail 要素ダンプ生データ・
  厳密真値突合結果・env_info）
- `docs/perf/cuda-gemm-candle-gate-remeasurement.md` §5・§11.4・§12.5（N=2048 判定不能の
  実測記録）・`docs/perf/cpu-gemm-candle-gate-remeasurement.md` §5.2・§12.3（CPU 版）
- `docs/spec-proposal-req2-req8-revision.md`（限定救済項〈案 1′〉の既存 spec 提案 draft）
- `.claude/rules/coding-rust.md`「テスト・ベンチ」節（tolerance 単独緩和の禁止）
- `.claude/rules/out-of-scope-tracking.md`「仕様変更が必要な場合」節
