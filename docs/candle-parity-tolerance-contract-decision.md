# tolerance 契約変更の決定記録（draft）

> **本文書は draft であり採用決定ではない。** `RELATIVE_TOLERANCE`／`ABSOLUTE_RESCUE_THRESHOLD`・
> `PARITY_REL_TOL`／`PARITY_ABS_TOL`・判定式（`compare`／`_parity_check`）・
> `crates/backend-cuda/tests/common/parity_baseline.rs::BASELINES`・`docs/spec/`（正本 submodule）
> は本 draft の作成にあたって一切変更していない。採否・判定形式・係数値・適用スコープ・
> spec 提案の起票可否は、いずれもイシュー #1241 でのユーザー承認を経て初めて確定する
> （`.claude/rules/coding-rust.md`・`.claude/rules/security.md`「自己修復ループ固有のガードレール」）。

## 1. 位置づけ

イシューツリー #1234（ルート）→ Phase 1 親 #1236 配下、前段イシュー #1237
（`docs/perf/candle-parity-tolerance-candidates.md`）・#1238
（`docs/perf/candle-parity-tolerance-baseline-impact.md`）に続く本イシュー #1239 の成果物。
並走イシュー #1240（spec 提案 draft。本文書公開時点で PR・コメントなし）とは独立に、
本文書は候補判定の比較・推奨案（draft）・ユーザー承認待ち事項の整理に閉じる。
承認記録は #1241、承認後の実装（Phase 2）は #1243 配下 #1245／#1252／#1258 が扱う。

## 2. 背景

- framework-compare の GEMM ゲート（candle 比 5 回計測中央値）は、N=2048・正方・入力
  U[-0.5,0.5)・固定シードの条件で candle 側（CUDA cuBLAS・CPU gemm crate）出力が
  REQ-2 統一複合判定（相対誤差 1e-3 未満 または 絶対誤差 1e-5 未満）を各 2/4,194,304 要素
  外れ、5 run とも完全に決定的に「判定不能」となる
  （`docs/perf/cuda-gemm-candle-gate-remeasurement.md` §5.1・§11.4、
  `docs/perf/cpu-gemm-candle-gate-remeasurement.md` §12.3）。fandhe-ai 側は同条件で
  全 run `parity_fail_count=0`（同 §11.2・§12.2）
- 原因はイシュー #1184（`cuda-gemm-candle-gate-remeasurement.md` §5.3 追記）が実値・厳密真値
  突合で特定した: 参照実装・candle 側双方の丸め誤差は通常の累積誤差
  （`√K・ulp(max|partial|)` 水準）の範囲に収まっているが、行列積のキャンセレーションにより
  最終値が部分和絶対値最大の 1/400〜1/2600 に縮小し、丸め誤差フロアが相対・絶対の両閾値を
  同時に割り込む
- 「tolerance は緩めない」方針のまま据え置いてきた経緯があり（同 §5.4）、
  #1031（CUDA）・#1117（CPU）両ゲートの正式系列 `fandhe-ai =0.7.0` 再計測でも N=2048 は
  「判定不能」のまま未達成が確定している
  （`cuda-gemm-candle-gate-remeasurement.md` §11.5、`cpu-gemm-candle-gate-remeasurement.md` §12.4）
- #1237 は上記 fail 4 要素（cuda 2・cpu 2）の実値ダンプのみを入力に、候補判定（スケール付き
  絶対誤差 A／ULP ベース B）を現行複合判定へ OR 追加した場合の fail 数を机上算出し、#1238 は
  同じ候補定義を fandhe-ai 本体側の parity 非後退契約（`ParityBaseline::BASELINES`。45 行）へ
  適用した場合の影響（no-op／全救済／部分・未確定の 3 クラス分類・契約 5 項目への影響・同時
  更新箇所一覧）を机上確認した。両 issue とも「推奨案・採否は本 issue（#1239）が扱う」と明記
  して引き継いでいる

## 3. 候補の定義

記号（#1237 §3 に準拠。以下 `d = |ref − actual|`〈実行時適用可能〉、`K`〈GEMM の内積長〉、
`S_A・S_B`〈A・B 各行列の絶対値最大の積〉、`Σ|ab|`〈要素積絶対値の和〉、`max|partial|`
〈f32 FMA 逐次累積の部分和絶対値の最大〉）:

- **候補 A（スケール付き絶対誤差。pass 条件 `d <= bound`）**: A-1 `bound = c・u・K・S_A・S_B`
  （`u` は unit roundoff。以下 `u=2^-24` 表記に統一——`(c, eps=2^-23)` と `(2c, u=2^-24)` は
  同一 bound のため重複行を潰す。#1237 の元表は machine epsilon `eps_f32=2^-23` 表記も併記して
  いたが、本表では unit roundoff 側の `c` に正規化する）・A-2（`S_A・S_B` を実測値に置換）・
  A-3（`√K` スケール）・A-4（`K・u・Σ|ab|`。古典的前進誤差上界）
- **候補 B（ULP ベース。pass 条件 `err <= t・ulp(base)`）**: B-1（`base=max|partial|`。部分和
  トレースが要る）・B-2（`base=Σ|ab|` の実行時 1 パス代替）・B-3（`base=exact`〈出力値自体〉。
  診断専用で実行時には使えない）

各候補の記号・式の完全な定義は #1237 §3 を正とする（本節は要約のみ）。

## 4. 候補比較表

出典: `bound`／fail 数は `docs/perf/candle-parity-tolerance-candidates.md` §4.2・§4.3・§4.4、
`BASELINES` 分類は `docs/perf/candle-parity-tolerance-baseline-impact.md` §5。K=2048・
`S_A・S_B=0.25`（#1237 の入力条件 U[-0.5,0.5) 前提）。

| 候補 | bound(K=2048) | candle cuda fail | candle cpu fail | 許容誤差上限（1e-5 比。N=512/1024/2048/4096） | 実行時適用可否 | Phase 2 実装コスト | `BASELINES` 分類（no-op／全救済／部分・未確定／分類不能） |
|---|---:|---:|---:|---|---|---|---|
| A-1 `c=0.125`（`u=2^-24`） | 3.815e-06 | 2/2 | 2/2 | 0.10x/0.19x/0.38x/0.76x | 適用可能（3 スカラのみ） | 判定式へ OR 追加のみ（1 パス追加不要） | 34/0/11/0 |
| A-1 `c=0.25`（`u=2^-24`） | 7.629e-06 | 2/2 | 2/2 | 0.19x/0.38x/0.76x/1.53x | 同上 | 同上 | 31/0/14/0 |
| A-1 `c=0.5`（`u=2^-24`） | 1.526e-05 | **0/2** | **0/2** | 0.38x/0.76x/1.53x/3.05x | 同上 | 同上 | 25/0/19/1 |
| A-1 `c=1.0`（`u=2^-24`） | 3.052e-05 | **0/2** | **0/2** | 0.76x/1.53x/3.05x/6.10x | 同上 | 同上 | 22/0/22/1 |
| A-1 `c=2.0`（`u=2^-24`） | 6.104e-05 | **0/2** | **0/2** | 1.53x/3.05x/6.10x/12.21x | 同上 | 同上 | 17/0/27/1 |
| A-2（実測 `S_A・S_B`。`c=2, u=2^-24` 相当——#1237 元表の `c=1, eps=2^-23` と同一 bound。§3 の換算則 `(c, eps=2^-23) ≡ (2c, u=2^-24)` 適用） | 6.092e-05（代表値。要素依存。candidates.md §4.2 の `c=1 eps=2^-23` 行の値をそのまま使用） | **0/2** | **0/2** | 実測依存 | 適用可能（`S_A・S_B` の事前 1 回計算が要る） | O(1) 空間の全体 `max|A|・max|B|` 導出（#1238 §3.3 で確定） | 未算出（#1238 は候補 A-1 系のみ集計。§9 参照） |
| A-3（√K スケール。`c=2.0, u=2^-24` 相当） | 1.349e-06 | 2/2 | 2/2 | 全 N で no-op 相当 | 適用可能（**1 件も救済しない**） | 判定式へ OR 追加のみ | 45/0/0/0（**全行 no-op**） |
| A-4（`K・u・Σ|ab|`。古典的前進誤差上界） | 1.568e-02 | **0/2** | **0/2** | 現行絶対閾値比 1568 倍（**緩すぎる参考値**） | 適用可能だが不採用 | — | 集計省略（緩すぎるため対象外。#1238 §5 末尾） |
| B-1（`max\|partial\|` 基準。`t=32〜48`） | — | `t=32`: 1/2・`t=48`: 0/2 | `t=32`: 0/2・`t=48`: 0/2 | — | **部分和トレースが要る**（既存カーネルは最終値のみ返す） | GPU カーネル側の部分和トレース機構が要る（実装可否未検討。#1254 へ引き継ぎ） | 机上分類不能（要素単位ダンプ・部分和トレースが `BASELINES` に存在しない） |
| B-2（`Σ\|ab\|` 基準。`t=1`） | — | 0/2 | 1/2 | — | 実行時 1 パス追加で計算可能 | 1 パス追加で実装可能 | 上界代用のため「部分／未確定」に分類（全救済判定不能。#1238 §4） |
| B-2（`t=2`） | — | 0/2 | 0/2 | — | 同上 | 同上 | 同上 |
| B-3（`exact` 基準） | — | 全 `t` で 2/2〜0/2（`t≥1e5` で救済） | 同上 | — | **実行時には使えない（診断専用）** | 実装不可（真値は実行時に得られない） | 対象外 |

**境界の要点**（#1237 §6・#1238 §5 の再掲）:

- A-1 系は `u=2^-24` 表記で `c=0.5` 以上、A-2 は代表値で `c=2（u=2^-24 相当。#1237 元表の
  `c=1, eps=2^-23` と同一 bound）` 時点で 4 要素すべて救済する。これは現行絶対閾値 `1e-5` の
  1.5〜6 倍程度の緩和幅に相当する（A-1 `c=0.5` の 1.53 倍〜A-2 代表値の 6.09 倍）
- A-3（√K スケール）は本ダンプの fail 要素を 1 件も救済しない一方、`BASELINES` 側では
  全 45 行が no-op に分類される（後述 §5「スケーリング則の注記」で理由を扱う）
- A-1 系列で `BASELINES` の行を丸ごと「全救済」と**確定できた**のは `c=2.0（u=2^-23 系）` の
  1 行（`MmaTf32VsWmmaStaged 512³ seed=6002`）のみで、他は「no-op」（同行内の全 fail 要素が
  非救済と確定）か「部分・未確定」のいずれかに分類されている（#1238 §5「代表的な観察」）。
  「部分・未確定」は `baseline_max_abs_diff_ceiling` が行単位の保守的な集計値であるために
  no-op（`bound < 1e-5`）にも全救済（`ceiling <= bound`）にも該当しなかった行を指す分類であり
  （#1238 §4.2 `classify`）、**「行内の一部要素が救済されたことを確認した」分類ではない**。
  実際の救済要素数は 0 件（＝全 fail 要素が非救済のまま）から全件未満までのいずれもあり得、
  要素単位ダンプ（GB10 実機実測）なしには行内の fail 要素が 1 件でも救済されているか自体も
  本表からは確定できない点に注意する
- B-2（`sum_abs_ab` 基準）は `t=1` の時点で 4 要素中 3 件（cuda 2・cpu 1）を救済し、
  `t=2` 以上で全件救済するが、実行時は上界代用のため `BASELINES` 側では全救済を確定できない

**適用限界（#1237 §2.2 の必読事項の再掲）**: 上記は K=2048・正方・U[-0.5,0.5)・固定シードの
**1 条件のみ**の fail 要素が根拠であり、「候補で置き換える」判定（現行 pass 要素が候補判定の
下で新たに fail に転じうるか）は評価不能・スコープ外。他形状・他シードへの外挿は行わない。

## 5. 推奨案と定数値（draft の提案）

**第一候補: A-1（`pass ⇔ rel < 1e-3 ∨ diff < 1e-5 ∨ diff <= c・u・K・S_A・S_B`。既存 2 条件への
OR 追加・既存定数〈`RELATIVE_TOLERANCE`／`ABSOLUTE_RESCUE_THRESHOLD`〉は不変）。** 係数 `c`
（`u=2^-24` 表記）は次の 2 案を提示し、選択は #1241 に委ねる。

| 案 | `c`（`u=2^-24`） | bound(K=2048) | 現行 1e-5 比の余裕 | N=512 | N=4096 | `BASELINES` 分類（no-op/全救済/部分・未確定/分類不能） |
|---|---:|---:|---|---:|---:|---|
| (i) 最小緩和 | 0.5 | 1.526e-05 | max d（1.265e-05）に対し約 1.2 倍 | 3.8e-06（no-op 相当） | 3.05e-05（1e-5 比 3.05 倍） | 25/0/19/1 |
| (ii) 2 倍余裕 | 1.0 | 3.052e-05 | max d に対し約 2.4 倍 | 7.6e-06（no-op 相当） | 6.10e-05（1e-5 比 6.10 倍） | 22/0/22/1 |

推奨理由（事実ベース）:

- いずれも #1184 の 4 要素すべてを救済する（§4 表）
- 実行時に `K`・`S_A`・`S_B` の 3 スカラのみで計算可能（`S_A・S_B` は #1238 §3.3 が確定した
  O(1) 空間の凸関数性質による全体 `max|A|・max|B|` 導出方式を使えば追加コストは事前 1 回の
  スキャンのみ）
- `BASELINES` 45 行のうち「全救済」と**確定できた**行は 0 件（(i) 25/0/19/1・(ii) 22/0/22/1。
  no-op と確定した行〈(i) 25 件・(ii) 22 件〉は `ParityBaseline` の回帰検出力を損なわないが、
  「部分・未確定」に分類された行〈(i) 19 件・(ii) 22 件〉は #1238 の机上算出が行単位の保守的な
  `ceiling` 集計値のみに基づく分類（要素単位ダンプ非使用。#1238 §4.2）のため、行内の fail 要素が
  1 件も救済されていない可能性・一部のみ救済されている可能性・すべて救済されている可能性の
  いずれも排除できておらず、回帰検出力への実際の影響は要素単位ダンプまたは実測での再確認が
  必要な未確定事項として残る（分類不能 1 件も同様）
- spec 側の 2026-08-29 限定救済項（案 1′。`docs/spec-proposal-req2-req8-revision.md` §1.5.1・
  §1.6）と同型の `max(1e-5, …)` 構造であり、非後退の骨格が既存の spec 提案と整合する

**代替案: B-2（`t=2・ulp(Σ_k|a_k b_k|)`）。** spec 側が案 1′（A-3 と同型の √K 形式）に対して
指摘する「`S_A・S_B` はテンソル全体スケールであり要素ごとの局所性を失う」という限界に応える
選択肢として残す。ただし参照実装側に O(MNK) の追加パスが要り、`BASELINES` では上界代用のため
全救済判定が確定できない（§4 表）。局所性を優先する場合の選択肢として提示するに留め、
第一候補としては推さない（実装コストと `BASELINES` 側の判定確度の両面で A-1 に劣るため）。

B-1・B-3 は実行時適用不可のため不採用（B-1 は GPU カーネル側の部分和トレース機構が未実装、
B-3 は実行時に真値を得られない診断専用指標）。

### スケーリング則の注記（√K 形式との関係）

観測された誤差フロアは `√K・ulp(max|partial|)`（#1184 の分析）であり、`max|partial|` 自体が
おおむね `√K・S_A・S_B` オーダーであるため、実効的には fail 要素の丸め誤差フロアは K に
ほぼ**線形**に効く。spec 案 1′（A-3 と同型の √K 形式）は係数 `c=1` で本ダンプの fail 要素を
**1 件も救済しない**（§4 表）。案 1′ 同型の判定式で本ダンプを救済するには
`τ_f32 ≥ max d / (S_A・S_B・√K) = 1.265e-5 / (0.25 × 45.25) ≈ 1.1e-6` が必要——**この値は
K=2048 の 1 点でのみ成立する手計算であり #1237 の表には無い（#1237 表外）**。K が変わると
この係数は外れる可能性が高いため、線形 K 形式（A-1）を推奨する根拠の一つとする。

## 6. fandhe-ai 側 0 fail 不変・既存 baseline 非後退の根拠

- **(a) OR 追加の単調性**: 候補判定を既存複合判定へ OR で追加する限り、既存の pass 要素は
  pass のまま変わらない（`fail_count` は単調非増加。#1238 §4.1）。よって候補追加が
  fandhe-ai 側の `parity_fail_count=0` を後退させることは構造的にない
- **(b) fandhe-ai 側の実測 0 fail**: 正式系列 `fandhe-ai =0.7.0`・参考系列いずれも、
  N=1024/2048/4096 の CUDA GEMM ゲート全 run で `parity_fail_count=0`
  （`cuda-gemm-candle-gate-remeasurement.md` §11.4・§12.5）。CPU GEMM ゲートも同様
  （`cpu-gemm-candle-gate-remeasurement.md` §12.3〜§12.4）
- **(c) `assert_no_parity_regression` 5 項目への影響**（#1238 §6 の再掲）:

  | 検査項目 | 影響 |
  |---|---|
  | `baseline_provenance_unconfirmed` fail-closed 契約 | 無影響（候補追加とは独立） |
  | `total` 完全一致 | 無影響（要素数不変） |
  | `fail_count <= baseline_fail_count` | 恒常成立（OR 追加の単調性） |
  | `mean_abs_diff <= ceiling` | 無影響（bit 同一。候補は判定式のみを変え計算対象の値は変えない） |
  | `max_abs_diff`／`max_rel_err <= ceiling`（`Some` のみ） | 無影響（同上） |

- **(d) ただし `BASELINES` 自体は本 draft では再測定していない**: 候補判定を実装した新しい
  比較器では `fail_count` 等の意味が変わるため、`BASELINES` の実機再測定が Phase 2 で必要
  （#1238 §8）。§4・§5 で示した分類・値域は「OR 追加した場合の理論上の値域」であり確定値では
  ない。**本 draft は `BASELINES` の追加・更新を一切行っていない**

## 7. spec 側提案の要否（並走 #1240 との対応）

REQ-2「2026-09-02 追記」は次の 3 点を規定する（`docs/spec/04-requirements.md`。編集しない）:

1. 判定式・定数は当該追記で変更しない
2. 限定救済項（案 1′）は昇格しない
3. 非 Tensor Core 経路（本 draft が対象とする f32 SIMT candle 比較）は統一複合判定のまま

この整理により、**本体 `compare`／`assert_parity` の判定式へ A-1／B-2 を OR 追加する場合は
spec 再改定（#1240 案 (a) 相当）が前提**となる。一方、**framework-compare ハーネス限定
（`bench-common::parity`／`compare_gemm_gate.py`）で判定不能条件のみを緩和する場合でも、
「比較対象側（candle）の fail を判定不能とする」現行規定を変更する提案（#1240 案 (b) 相当）が
要る**。

本 draft はこの要否の整理と論点（線形 K 形式 vs √K 形式・`S_A・S_B` の局所性限界・
非 Tensor Core 経路への適用可否）のみを渡し、spec 提案本文の draft 自体は #1240 の成果物と
する。spec リポジトリへの実起票は #1241 承認後に限る。

## 8. ユーザー承認待ち事項（#1241 で記録）

1. **契約変更の採否**: tolerance 契約（判定式への候補判定 OR 追加）自体を変更するか、
   現状（「tolerance は緩めない」・N=2048 判定不能のまま据え置き）を維持するか
2. **判定形式と係数**: A-1（第一候補）か B-2（代替案）か。A-1 の場合、`c=0.5`（案 i）／
   `c=1.0`（案 ii）／その他の値か
3. **適用スコープ**: framework-compare ハーネス限定（`bench-common` + `compare_gemm_gate.py`）
   か、本体 `compare`／`assert_parity`／`ParityBaseline` にも反映するか
4. **spec 提案の起票可否**: #1240（案 (a)／(b)）の起票を進めるか
5. **既存定数の扱い**: `RELATIVE_TOLERANCE`／`ABSOLUTE_RESCUE_THRESHOLD`／`PARITY_REL_TOL`／
   `PARITY_ABS_TOL` は変更せず維持（OR 追加のみ）でよいか

## 9. Phase 2 の反映範囲（#1238 §7.1〜§7.5 の Phase 2 issue への対応付け）

| Phase 2 issue（想定） | 対応する #1238 の同時更新箇所 |
|---|---|
| #1247 | `bench-common::parity`（`element_error`・`PARITY_REL_TOL`／`PARITY_ABS_TOL` 周辺）への判定式追加、および定数ピンが判定式そのものを検査しない盲点（§7.1）を埋める判定式ピンテストの新設 |
| #1250 | `compare_gemm_gate.py::_parity_check`・`summarize.py` の判定不能条件の更新（§7.2） |
| #1254 | 本体 `compare` の新入口追加（シグネチャ非破壊。§8 の設計制約）・判定式レプリカ群（§7.2）・リテラル閾値レプリカ（§7.3）・`extract_f64_const`／`_extract_f64_const` 両方への新定数追加（§7.5）・規約文言更新（§7.4） |
| #1256 | GB10 実機で `BASELINES` の非後退確認、必要なら再測定（人間承認必須） |
| #1260／#1262 | CUDA／CPU N=2048 再計測で判定不能が解消したことの記録 |

**#1241 で本 draft が却下された場合、Phase 2 は「対応不要」としてクローズする。**

## 10. 未変更事項の確認

本 draft の作成にあたり、以下は一切変更していない:

```
$ git diff --stat origin/main -- crates/ scripts/ .github/ docs/spec/
(出力なし)
```

- `crates/backend-cpu/src/parity.rs::RELATIVE_TOLERANCE`（1e-3）・`ABSOLUTE_RESCUE_THRESHOLD`（1e-5）
- `scripts/bench/framework-compare/bench-common/src/parity.rs::PARITY_REL_TOL`（1e-3）・
  `PARITY_ABS_TOL`（1e-5）
- `crates/backend-cuda/tests/common/parity_baseline.rs::BASELINES`（45 行）
- `scripts/bench/framework-compare/compare_gemm_gate.py`・`summarize.py`（判定ロジック・
  判定不能条件）
- `docs/spec/`（正本 submodule。編集しない）

## 11. 関連ドキュメント

- `docs/perf/candle-parity-tolerance-candidates.md`（イシュー #1237。候補 A/B の fail 数机上算出）
- `docs/perf/logs/candle-parity-tolerance-candidates-1237/`（同上の生出力・env_info）
- `docs/perf/candle-parity-tolerance-baseline-impact.md`（イシュー #1238。`BASELINES` への影響
  机上確認）
- `docs/perf/logs/candle-parity-tolerance-baseline-impact-1238/`（同上の生出力・env_info）
- `docs/perf/logs/cuda-gemm-candle-parity-1184/`（イシュー #1184。fail 要素ダンプ生データ・
  厳密真値突合結果・env_info）
- `docs/perf/cuda-gemm-candle-gate-remeasurement.md` §5・§11.4・§12.5（N=2048 判定不能の
  実測記録）・`docs/perf/cpu-gemm-candle-gate-remeasurement.md` §5.2・§12.3（CPU 版）
- `docs/spec-proposal-req2-req8-revision.md`（限定救済項〈案 1′〉の既存 spec 提案 draft）
- `docs/cuda-tensor-core-parity-judgment-decision.md`・`docs/cuda-tf32-optin-api-decision.md`
  （決定記録の章立て・承認ステータス節の書式先例）
- `.claude/rules/coding-rust.md`「テスト・ベンチ」節（tolerance 単独緩和の禁止・TF32/f16
  判定方式の正）
