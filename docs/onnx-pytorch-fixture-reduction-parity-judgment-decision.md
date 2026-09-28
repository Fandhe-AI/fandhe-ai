# ONNX PyTorch fixture 突合（縮約系）の判定方式決定記録

イシュー #2329（親 #2185）・PR #2343 の codex-review 指摘対応として、
`crates/onnx-interop/tests/onnx_interp_pytorch_cnn_fixture.rs`
（`Conv`・`AveragePool`・`GlobalAveragePool`・`BatchNormalization` の
PyTorch 実生成 ONNX fixture 突合）の縮約系判定方式を確定する。

## 1. 背景・spec との切り分け

親 #2185 の受け入れ条件チェックボックスは「PyTorch ONNX export との import
往復で bit 同一を確認する」と記していた。一方 `docs/spec/
04-requirements.md` REQ-7 の受け入れ基準本体（`04-requirements.md:182`
「数値一致基準（v2 方式）」の箇条書き）は「PyTorch 参照値に対し判定式
`abs_err / (|ref| + 1e-6) ≤ 1e-3` を満たすこと（PoC-v2-6 事前固定基準と
同一）」とのみ定めており、bit 同一（`to_bits()` 完全一致）は要求して
いない。同「詳細」節（`04-requirements.md:174`）も PoC-v2-6 の実測根拠
として「PyTorch 参照値との最大相対誤差 0.000000」を記すのみである。
`docs/spec/04-requirements.md` 全体で「ビット」（bit 完全一致）に言及する
箇所は REQ-2（v1 Burn 基盤上でのバックエンド間 3 系統一致。`04-requirements.
md:7`・`:19`・`:22`・`:69`）に限られ、REQ-7 の受け入れ基準には現れない。
REQ-7 事前固定式 `abs_err / (|ref| + 1e-6) <= 1e-3`
（`tests/model_zoo_parity.rs::assert_req7` 以来の既存式）はこの受け入れ
基準の実装である。

したがって「bit 同一」という要求は spec（正本）由来ではなく、#2185
（実装リポ issue）本文が独自に付け足したチェックボックス表現にすぎない。
本改定は spec の緩和ではなく、実装リポ issue 側の受け入れ条件表現を
spec の定めへ整合させる変更である。#2185 のチェックボックス文言の更新は
別途 main セッションが行う。

## 2. 縮約系で bit 同一が原理的に成立しない理由

`Conv`・`AveragePool`・`GlobalAveragePool`・`BatchNormalization`（dynamo
分解経路の `ReduceMean` を含む）は、PyTorch の CPU 実行系（MKL-DNN／
oneDNN 等のブロック化・SIMD 縮約順序）と本クレートの直接ループ実装
（row-major 順の `f32::mul_add`／`f64` 逐次蓄積。`ops::conv`・`ops::pool`・
`ops::batch_norm`・`ops::global_average_pool`）とで縮約（累算）の結合順序
が異なる。浮動小数点加算は結合則を満たさないため、この結合順序差は原理的
に bit 完全一致を生まない（実測でも 14 ケース中 9 ケースで bit 不一致が
生じた。§4 参照）。一方 `MaxPool`・`Flatten` は選択・形状操作のみで縮約を
伴わないため bit 完全一致が成立し、実測でも全ケースで成立した。

## 3. 決定: 併用方式（2026-09-28 ユーザー承認）

縮約系の受け入れ条件から bit 同一を外し、次の **併用方式**へ正式移行する
（`Expectation::Req7BaselineNonRegression`）。

1. **REQ-7 事前固定式**（`abs_err / (|ref| + 1e-6) <= 1e-3` の
   `fail_count == 0`）を**必須条件として維持**する。既存の tolerance 定数
   （`1e-3`）自体は変更しない。
2. その上で、**ケースごとの実測上限 baseline**
   （`REDUCTION_BASELINES: &[ReductionBaseline]`）に対する fail-closed
   非後退判定を行う。baseline は次を保持する:
   - `total`（比較対象要素数の完全一致検査）
   - `baseline_fail_count`（REQ-7 式が必須条件のため常に `0`。項目 1 と
     独立に非後退性を機械検査する）
   - `baseline_max_abs_diff_ceiling`（`f32`）
   - `baseline_max_rel_err_ceiling`（`f32`）
   - `baseline_mean_abs_diff_ceiling`（`f64`。`abs_diff` を `f64` で
     蓄積してから要素数で割った値）

ceiling は**実測値そのもの**（余裕係数を掛けない）。実測で bit 一致した
ケース（`conv2d_basic`・`conv2d_nobias`〈ts/dynamo 双方〉・`gap1d`〈`ts`
のみ〉）も本方式の対象に含め、ceiling を `0` として記録する——「bit 一致
したケースだけ別枠の `BitExact` 扱いにする」という以前の設計（PR #2343
初版）は、縮約系の受け入れ条件を op 種別ではなくケースごとの実測結果で
分岐させてしまい、表の書き換えで縮約系ケースを緩い方式へ動かす／その逆へ
動かす、という双方向の書き換えミスを構造的に検出できなかった。本方式では
「縮約系 op を含む演算列かどうか」という構造的な条件のみで `Expectation`
を分岐させ、`assert_expectation_matches_op_types` が機械検査する
（`REDUCTION_OP_TYPES = ["Conv", "AveragePool", "GlobalAveragePool",
"BatchNormalization", "ReduceMean"]`）。

`MaxPool`・`Flatten` は縮約を伴わないため、従来どおり
`Expectation::BitExact`（フォールバックなしの bit 完全一致のみ）を維持
する。

baseline の追加・更新は実測値のみ・人間承認必須という fail-closed 設計は
`crates/backend-cuda/tests/common/parity_baseline.rs::ParityBaseline`／
`crates/backend-metal/tests/common/splitk_parity_baseline.rs::
SplitKParityBaseline` と同型（`.claude/rules/coding-rust.md`「結合順序が
単一の連続 K ループと異なるカーネルの parity テスト判定方式」節・
`docs/cuda-tensor-core-parity-judgment-decision.md`・
`docs/backend-metal-splitk-parity-judgment-decision.md` を参考にした）。
`reduction_baselines_are_well_formed`（`#[ignore]` なし。通常 CI で常時
実行）が次を機械検査する: `REDUCTION_BASELINES` の `(case_name,
exporter_name)` 集合が `EXPECTATIONS` の `Req7BaselineNonRegression`
エントリと過不足なく一致すること（重複行・取りこぼしの検出）／各行の
`baseline_fail_count == 0`／各行の ceiling が有限かつ非負（NaN・負値の
混入拒否）／各行の `total > 0`。

## 4. 実測値（baseline の出典）

実測コマンド・実測環境・全 28 行（縮約系 14 ケース × 2 exporter）の値は
`docs/perf/logs/onnx-cnn-ops-pytorch-fixture-2329/README.md`「ケース ×
exporter の実測結果」表を正とする（本 doc では二重管理しない）。全ケースが
REQ-7 事前固定式・baseline 双方の判定を通過した
（`cargo test -p fandhe-ai-onnx-interop --test onnx_interp_pytorch_cnn_fixture`
で確認済み。40 テスト全 pass・`#[ignore]` 0 件）。

## 5. 計算経路の決定性（CPU feature 検出との無関係性）

baseline を CI（GitHub ホステッド `ubuntu-latest`・x86_64）・ローカル
（x86_64）で共通の単一値とできるかどうかは、判定対象の計算経路が
ランタイム環境（CPU feature・スレッド数）に依存して結果が変わりうるかに
かかっている。調査の結果、次を確認した:

- `ops::conv`・`ops::pool`・`ops::batch_norm`・`ops::global_average_pool`
  （`crates/onnx-interop/src/ops/conv.rs`・`pool.rs`・`batch_norm.rs`・
  `global_average_pool.rs`）はいずれも `rayon`・SIMD intrinsics・
  `is_x86_feature_detected!` 等のランタイム分岐を含まない単純な逐次ループ
  （`f32::mul_add`／`f64` 蓄積、走査順は入力の row-major 順に固定）である。
  `is_x86_feature_detected!`・`target_feature` は `crates/backend-cpu`
  （AVX2/AVX-512/NEON 系 SIMD GEMM カーネル）のみに存在し、
  `fandhe-ai-onnx-interop` クレートからは到達しない。
- dynamo 分解経路の `ReduceMean`（`onnx::interp_ext::compute_reduce_mean`）
  は `fandhe_ai_autodiff::Var::mean` → `default_ops::NaiveOps::sum`
  （`Tape::new()` の既定 `BackendOps`）→ `autodiff::eval::sum` を経由する。
  `eval::sum` の全軸縮約は `Vec<f32>::into_iter().sum()`（標準ライブラリの
  `Sum for f32` 実装。`iter.fold(0., Add::add)` による逐次左畳み込み）で
  あり、`rayon` 並列縮約は使わない。Rust は明示的な `fast-math`／
  `reassoc` 指定なしに浮動小数点加算を並べ替えないため、コンパイル最適化
  レベル（debug/release）を問わず加算順序は固定される。
- `fandhe-ai-onnx-interop` クレート自体の `Cargo.toml` に `rayon` の依存
  記載はない。

以上より、本 fixture の判定経路には CPU feature 検出・スレッド数依存の
非決定性が存在しない。baseline は経路ごとに分ける必要がなく、CI・
ローカルを問わず単一の値で成立する。

## 6. 変更しないもの

- REQ-7 事前固定式・tolerance 定数（`1e-3`）自体
- `MaxPool`・`Flatten` の bit 完全一致判定（`Expectation::BitExact`）
- R1〜R3・R5 の既存記録（`docs/perf/logs/onnx-cnn-ops-pytorch-fixture-2329/
  README.md` 参照）
- external data（dynamo exporter の companion `.data` file）への対応要否
  ——別イシューで対応予定（ユーザー決定済み）

## 7. 関連 Issue・PR

- イシュー #2329（親 #2185）
- PR #2343（codex-review 指摘: レビュースレッド
  `PRRT_kwDOTuUCJc6mhmHL`・`PRRT_kwDOTuUCJc6mhpBB`・`PRRT_kwDOTuUCJc6mh384`。
  Cursor Bugbot: `PRRT_kwDOTuUCJc6mh4GY`）
- 2026-09-28 ユーザー承認: 本 doc §3 の併用方式
