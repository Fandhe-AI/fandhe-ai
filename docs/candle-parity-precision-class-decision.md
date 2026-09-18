# 比較対象の精度クラスと PyTorch cpu N=4096 fail 要素の扱いの設計判断（draft）

> **本文書は draft（未承認）である。採否・適用スコープ・係数値・PyTorch 扱い方の 4 点はイシュー #1989 でのユーザー承認後に確定版化する。コード変更なし。**

## 1. 位置づけ

イシューツリー ルート #1966（Phase 5）→ 親 #1982（判定不能 6 セル）→ #1983 → #1984 → #1985 → **本イシュー #1986** → #1989（ユーザー承認）→ #1987（実装）→ #1988（再計測）。

前作は #1239（`docs/candle-parity-tolerance-contract-decision.md` 確定版。2026-09-08 ユーザー承認。イシューツリー #1234（旧ルート）→ Phase 1 親 #1236 配下）であり、本文書はその続編として位置づける。

#1984 は burn cuda TF32 経路の fail 要素を `Fraction` 厳密真値と突合して線形 K 形 bound（`u=2^-11`・`c=0.5`）での全数救済を確認した事実の記録。#1985 は PyTorch cpu N=4096 の fail 1 要素を同じく厳密真値と突合して、`c=1.0` 線形 K で救済される事実と、比較データの妥当性上の「判定不能」であることを記録した。

本文書は上記 2 件の実測結果から、ハーネス側の精度クラス導入案（burn の TF32 行を `u=2^-11` で特別扱い）と PyTorch cpu 4096 の 3 択（T1：現状維持／T2：係数変更／T3：形式変更）を整理し、推奨案（未承認）を示す。実装・確定版化は #1989 のユーザー承認を経て #1987（burn 精度クラス結線）・#1988（PyTorch 判定・framework-compare 再計測）へ引き継ぐ。

## 2. 背景

### fail セルの事実

framework-compare 0.9.0 系列（2026-09-18 registry ピン）で、スコアボード規則上「判定不能」となるセルが存在する。詳細は出典を参照（`docs/perf/logs/parity-burn-tf32-truth-1984/README.md` §(c)・`docs/perf/logs/parity-torch-cpu-truth-1985/README.md` 集計部）：

| 項目 | セル | fail_count（母集団） | max_abs_err | 現行 bound（u=2^-24, c=0.5） | u=2^-11 bound（c=0.5） | 注釈 |
|------|------|---:|---:|---:|---:|---|
| burn cuda gemm fresh | N=256/512/1024/2048/4096 | 10538/42361/169929/681407/2728488 | 1.58e-3〜7.12e-3 | 1.91e-6／3.81e-6／7.63e-6／1.53e-5／3.05e-5 | 1.56e-2／3.13e-2／6.25e-2／1.25e-1／2.50e-1 | #1984。JSONL `parity_scaled_abs_bound`（`docs/perf/logs/framework-compare-cuda-tf32-sweep-1983/results-cuda-0.9.0-2026-09-18.jsonl` line 27-31）から読取。全 N で u=2^-11 線形 K bound 内に収まり全数救済 |
| PyTorch cpu gemm | N=4096 | 1 | 8.96e-05 | 3.05e-5 | （N/A） | #1985。idx=343838・fail 要素の d=3.37e-5。線形 K 形は c=1.0 で救済・c=0.5 で fail |

### 比較対象の精度クラスの背景

#1984 の統計（`docs/perf/logs/parity-burn-tf32-truth-1984/README.md` §(c)・truth-summary.md）から：

- **burn TF32 実測値の丸め誤差水準**: 中央値 4.0e-4〜1.6e-3 は TF32 単位丸め `u=2^-11`（4.9e-4）にほぼ一致
- **f32 FMA 参照実装との比較**: `|ref−exact|` は 1e-12〜2.5e-5、中央値 1.3e-7〜2.0e-6。全解析行（N≤1024 母集団＋N=2048/4096 サンプル 4096 件）で参照がより正確
- **救済率**:
  - 線形 K 形（`u=2^-11`, `c=0.5`）: N≤1024 母集団 100%・N=2048/4096 サンプル内 100%。JSONL `parity_max_abs_err` が bound を全 N で下回る客観的根拠あり
  - √K 形（`u=2^-11`, `c=0.5`）: N≤1024 母集団 98.2〜98.4%・N=2048/4096 サンプル内 98.2%／98.0%

### PyTorch cpu N=4096 fail 1 要素の背景

#1985 の分析（`docs/perf/logs/parity-torch-cpu-truth-1985/README.md` §「fail 要素の真値突合」・§「救済可否表」）から：

- **idx=343838 (row 83, col 3870)**:
  - `|actual−truth| = 4.014e-06`（PyTorch）< `|ref−truth| = 2.972e-05`（f32 FMA 参照）
  - PyTorch 側が真値（`-1.325068847e-02`）に近い珍しいケース
  - f32 FMA 参照は bit 一致（`ref_fma_bit_match=True`）で二重丸め境界ではない
- **救済可否**:
  - `d = |ref−actual| = 3.373e-05` に対して:
    - `c=0.5` 線形 K（現行）: bound=3.051e-05 < d → fail
    - `c=1.0` 線形 K: bound=6.103e-05 > d → **救済**
    - `c=1.5` 線形 K: bound=9.155e-05 > d → 救済
    - √K 形（いずれの c でも fail）

## 3. 候補の定義

### 精度クラス導入案（P）

判定式内の `ScaledAbsTolerance` の `u`（unit roundoff）を比較対象フレームワークごとに切り替える案。出典：`scripts/bench/framework-compare/bench-common/src/parity.rs::ScaledAbsTolerance` (line 106)。

- **burn cuda 行のみ例外化**: `tf32: true` を検出した場合（JSONL の `"tf32":true` フィール値）、当該セルの判定で `u=2^-11`（TF32 単位丸め）を使用
- **他の比較対象・fandhe-ai 行は不変**: candle・PyTorch・fandhe-ai（`verify_strict` 含む）は従来どおり `u=2^-24`（f32 unit roundoff）
- **係数・形式は不変**: `c=0.5`・線形 K は現行から変更なし（√K は対象外のまま）
- **実装場所**: `scripts/bench/framework-compare/bench-common/src/parity.rs::ScaledAbsTolerance::bound()` の `u` の選択分岐
- **fandhe-ai 側への影響**: 「判定不能」セルが救済対象に変わるだけで、fandhe-ai の `verify_strict` 行は引き続き 0 fail のまま

### 対案（P′）

`u` を不変のまま burn 行を「判定不能」として維持（現状維持）

### PyTorch cpu N=4096 の扱い（3 択）

#### 案 T1：現状維持

- 当該 1 要素を「spec 上正当な判定不能」として記録のまま維持
- **根拠**: spec REQ-2（2026-09-12 追記 (b-1)）が「第三者比較対象の出力が統一複合判定を外れた場合は比較データの妥当性上の判定不能」と定めていること
- 係数・形式の変更なし。spec 再改定不要

#### 案 T2：係数 c を 1.0 へ変更

- 現行 `c=0.5` を `c=1.0`（係数を 2 倍）へ変更すると、PyTorch 1 要素も救済される
- **代償**: `bound = c·u·K·S_A·S_B` は他条件が同じなら c に単調増加するため、c=0.5 で救済済みの要素（burn 線形 K 全数救済・candle 2 要素）は c=1.0 でも必ず救済される（既存救済は失われない）。実際の懸念は許容範囲が 2 倍に広がることによる**回帰の見逃し**（fandhe-ai 側で将来生じうる真の後退を救済してしまう）であり、c=0.5 で承認済みの前作 #1241 の再承認が必要
- spec 再改定必要（係数変更）

#### 案 T3：√K 形へ変更

- PyTorch 1 要素は `√K` 形では `c=1.5` でも fail（救済不可）
- 消去法の対象候補

## 4. 候補比較表

| 候補 | 適用対象 | burn 5 セル | PyTorch 1 要素 | fandhe-ai 側 影響 | spec 改定 | ハーネス改定 箇所 | 推奨度 |
|------|--------|---:|---:|---|---|---|---|
| P（精度クラス） | burn cuda TF32 のみ `u=2^-11` | 全数救済（客観的根拠: JSONL bound 内） | 非対応（fandhe-ai 判定不変） | 0 fail 不変 | **必要**（(b-2) は `u = 2^-24` を逐語で固定。`tf32: true` 行の `u=2^-11` は (b) 形式の spec 提案〈#1989 承認後〉が前提） | `ScaledAbsTolerance` に精度クラス選択 | 高 |
| P′ | 不変 | 判定不能維持 | 判定不能維持 | 0 fail 不変 | 不要 | なし | 参考（保守性の観点で変化なし） |
| T1（PyTorch 現状維持） | N/A | P/P′ に従属 | 判定不能維持 | 0 fail 不変 | 不要 | なし（P または P′ に従属） | 高（spec 定義の踏襲） |
| T2（c=1.0 化） | 全体 | 全数救済維持（bound は c に単調増加。c=0.5 の救済は c=1.0 でも保持） | 救済 | 0 fail 不変（ただし許容範囲 2 倍化により将来の後退を見逃すリスク） | 必要（係数再承認） | `PARITY_SCALED_ABS_COEFF=1.0`・テスト定数・`bench_py.py` の係数リテラル | 低（回帰見逃しリスク） |
| T3（√K 化） | N/A | √K 救済率は 98% 台（全数救済不可能） | fail のまま | 0 fail 不変 | 必要（形式再承認） | 判定式変更・テスト複数箇所 | 非推奨 |

## 5. 推奨案（未承認）

### 精度クラス導入（案 P）を推奨候補

- **burn TF32 精度クラス導入** (P): `u=2^-11` の機械的事実（実測 max_abs_err が TF32 単位丸め水準・線形 K bound で全数救済の客観的根拠）に基づく
- **実装リスク が低い**: スカラ `u` フィールドの条件分岐のみで、判定式・tolerance 定数・fandhe-ai 側判定は不変
- **fandhe-ai 側への影響なし**: 「判定不能」セルの救済であり、fandhe-ai の 0 fail（`verify_strict` 含む）は維持される

### PyTorch cpu N=4096 の扱い（案 T1 を先頭に併記）

#### 第一候補：T1（現状維持）
- **根拠**: spec REQ-2（2026-09-12 (b-1)）が「第三者比較対象の妥当性上の判定不能」を明記
- **代償なし**: 係数・形式の変更が不要で、既存承認を揺るがさない
- **推奨として併記**

#### 参考：T2（係数 c=1.0）
- PyTorch 1 要素の救済は可能。bound は c に単調増加するため burn 全数救済・candle 2 要素の既存救済は c=1.0 でも維持される（再検証は不要）
- 代償は許容範囲の 2 倍化であり、fandhe-ai 側で将来生じうる真の後退（c=0.5 では fail になる差）を救済して見逃す回帰検出力の低下
- **事実として記録**: 実測では線形 K 形 c=1.0 で救済されるが、これだけで係数変更を決定するには裏付けが限定的

#### 非推奨：T3（√K 化）
- PyTorch fail のまま（救済不可）のため消去法対象

## 6. fandhe-ai 側 0 fail 不変の根拠

- `ScaledAbsTolerance::from_inputs` の `u` フィールド化は比較対象行（burn TF32・candle・PyTorch）のみに適用。出典：`scripts/bench/framework-compare/bench-common/src/parity.rs::from_inputs` (line 138)
- fandhe-ai 行（all mode）は `verify_strict`（本体 parity テストの厳密ゼロ fail 要求）を回避できず、引き続き既存 2 条件（`PARITY_REL_TOL=1e-3` ∨ `PARITY_ABS_TOL=1e-5`）のみで判定
- 精度クラス導入は「比較対象の精度水準の認識」の形式化であり、本体の統一複合判定の緩和ではない

## 7. spec 側提案の要否

### 案 P（精度クラス） → spec 提案**必要**（(b) 形式・#1989 承認後に起票）
- REQ-2 の 2026-09-12 追記 (b-2) は第 3 項を `diff <= 0.5・u・K・S_A・S_B`（`u = 2^-24`〈unit roundoff〉）と**逐語で固定**している（`docs/spec/04-requirements.md` REQ-2 (b-2)）。比較対象行に限って `u` を `2^-11` へ切り替える案 P はこの逐語規定からの逸脱であり、ハーネス限定であっても spec 側の追記（「比較対象の精度クラスに応じた `u` の選択を認める」旨）が前提になる
- (b-2) が言及する比較対象は candle のみで、burn（TF32 既定降格）は spec が現時点でカバーしていない拡張である
- 提案は前作 §7 と同じ (b) 形式（実装リポ側 doc を出典として spec に短い規定を追記）とし、起票は #1989 でのユーザー承認後に限る（本 draft では起票しない）

### 案 T1（PyTorch 現状維持） → spec 提案**不要**
- spec REQ-2 既存の (b-1) 定義に従うもので、追加の明文化は不要

### 案 T2（係数 c=1.0） → spec 提案**必要**（係数再承認）
- 2026-09-12 (b-2) で承認済みの `u=2^-24`・`c=0.5`・線形 K を c=1.0 へ改定するため、spec への反映（人間承認）が前提
- 既存 (b-2) の係数改定として (b) 形式で別途起票（本 draft では起票しない）

## 8. ユーザー承認記録（承認待ち）

| 項目 | 案 | 決定 | 記録 |
|------|------|------|------|
| 精度クラス導入の採否 | P vs P′ | 承認待ち | |
| burn TF32 行への適用 | u=2^-11 機械的事実の受け入れ | 承認待ち | |
| PyTorch cpu N=4096 の扱い | T1（現状維持） vs T2（c=1.0） | 承認待ち | |
| spec 側提案の起票可否 | P: 必要（(b) 形式）/ T1: 不要 / T2: 必要 | 承認待ち（P または T2 選択時） | |

### 承認記録ブロック（決定後に記入）

（承認待ち）本文書の §8 承認記録は #1989 でのユーザー判断を受けて記入予定

## 9. Phase 2 の反映範囲

### #1987：burn TF32 精度クラス実装（案 P 採用時）

**修正対象ファイル**: `scripts/bench/framework-compare/bench-common/src/parity.rs`・`scripts/bench/framework-compare/bench-burn/src/main.rs`

- `ScaledAbsTolerance` に精度クラス選択フィールド（例: `precision_class: PrecisionClass` enum〈`f32`／`tf32`〉）を追加
  - または、より簡潔に `u: f64` フィールドで直接 unit roundoff を受け取る設計
- 精度クラスは判定処理より**前**に渡す。既存 `bench-burn::run_gemm` は `reference.verify(&out)` で判定した後に `Record`（`tf32: cli.device == "cuda"`）を構築して JSONL へ出力するため、判定内部で JSONL の `tf32` を読む設計はデータ依存順序が逆になる
- `bench-burn/src/main.rs`（`run_gemm` および `Record` を構築する他 2 経路）で `cli.device` から決まる精度クラスを `verify`（または `compute`／`verify_with_sink_and_tol`）の引数として事前に供給し、JSONL の `tf32` 列は従来どおり記録専用に保つ
- テスト: 既存 `gemm_cpu_parity_zero_fail_without_scaled_rescue`（fandhe-ai 0 fail 不変検証）に burn TF32 全数救済検証を追加

### #1988：PyTorch および framework-compare 再計測（案 T1 または T2）

**修正対象ファイル**:
- T1（現状維持）: 修正なし（記録のみ）
- T2（c=1.0）: `PARITY_SCALED_ABS_COEFF=1.0` に変更・テスト境界値更新に加え、PyTorch 計測経路 `docs/perf/logs/lowlayer-diagnosis-2026-09-12/bench_py.py::parity`（`bound = 0.5 * (2.0 ** -24) * n * sa * sb` と係数 `0.5` を直接定義）の係数更新（または共通定数の参照化）も反映範囲に含める。Rust 側定数の変更だけでは PyTorch の bound は変わらず、目的の 1 要素は救済されない。GB10 / M4 Max での framework-compare 再計測

## 10. 未変更事項の確認

本候補すべてで不変：

- `PARITY_REL_TOL=1e-3`（`RELATIVE_TOLERANCE` 机上検証テスト不変）。出典：`scripts/bench/framework-compare/bench-common/src/parity.rs` (line 56)
- `PARITY_ABS_TOL=1e-5`（`ABSOLUTE_RESCUE_THRESHOLD` 机上検証テスト不変）。出典：同ファイル (line 61)
- 本体判定式（`crates/backend-cpu/src/parity.rs::compare` ほか）及び `ParityBaseline`
- `docs/spec/04-requirements.md` REQ-2 の定義（2026-09-12 追記 (b-1)/(b-2) 不変）

## 11. 関連ドキュメント

- 前作：`docs/candle-parity-tolerance-contract-decision.md`（2026-09-08 ユーザー承認確定版）
- 実測データ：`docs/perf/logs/parity-burn-tf32-truth-1984/`（#1984 の burn TF32 真値突合）
- 実測データ：`docs/perf/logs/parity-torch-cpu-truth-1985/`（#1985 の PyTorch cpu N=4096 真値突合）
- 判定式出典：`scripts/bench/framework-compare/bench-common/src/parity.rs::ScaledAbsTolerance`・`PARITY_SCALED_ABS_COEFF` (line 84)・`F32_UNIT_ROUNDOFF` (line 90)
- spec 参考：`docs/spec/04-requirements.md` REQ-2 節（2026-09-12 追記 (b-1)・(b-2)）
