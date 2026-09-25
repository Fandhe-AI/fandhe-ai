# floor・ceil・round・sign・reciprocal・rsqrt・erf・pow_scalar の設計判断記録

イシュー #2145（親 #2131）。`docs/autodiff-rearrange-ops-decision.md`・
`docs/autodiff-bool-ops-exposure-decision.md` と同型の記録。

## §0 結論

要素ごとのスカラー単項演算 8 種（`floor`／`ceil`／`round`／`sign`／
`reciprocal`／`rsqrt`／`erf`／`pow_scalar`）を、**`fandhe_ai_autodiff`
のうち facade が再エクスポートしない自由関数モジュール
`scalar_unary_ops`**（`crates/autodiff/src/scalar_unary_ops.rs`）として
実装した（案 C。§2 参照。`bool_ops`〈#2141〉・`rearrange_ops`
〈#2143〉と同じ判断枠組み）。`Var` に inherent の `pub fn` は追加して
いない。新規 `Op` は追加していない——いずれも既存の `Op::ScalarUnary`
（`tensor_core::ScalarUnaryOp` の 8 variant への dispatch。うち
`PowScalar` は #1634 で定義済みで本モジュールは入口関数の追加のみ）へ
の薄い委譲のみで構成した。GPU 専用カーネルはスコープ外（既定の
`Unsupported` → ホスト参照実装フォールバック）。facade 公開（`Var`
への委譲メソッド追加）は承認待ちのまま対象外とし、`crates/facade/
src/lib.rs::VarScalarUnaryOpsHoldDoctestGuard`（正のプローブ
doctest）と `crates/facade/tests/api_surface.rs` のソース走査・
workspace インベントリ（4 テスト）で多層固定している。

## §1 背景

イシュー #2145・親 #2131 にはコメントが 0 件で、facade 公開の承認
記録はない（2026-09-25 時点。着手前に `gh issue view 2145/2131
--comments` で確認済み）。親 #2131 はこのツリーでの facade 公開面の
拡張を「設計判断記録 → 承認 → 実装」の 2 段階と定めているため、本
実装は内部クレート限定に倒す（`rearrange_ops`〈#2143〉・`bool_ops`
〈#2141〉と同じ枠組み）。

## §2 設計判断（API 配置）

- **案 A（`Var` の inherent メソッド）**: facade へ即座に到達する
  （`crates/facade/src/lib.rs` は `pub use fandhe_ai_autodiff::{…, Var,
  …};` で `Var` を再エクスポートしているため）。承認が無いため不採用
- **案 C（自由関数）**: 採用。承認後の撤去・委譲が単純
- モジュール名は `scalar_unary_ops`（`tensor_core::ScalarUnaryOp` の
  dispatch を薄く包む自由関数群である契約をそのまま名前へ反映。
  `rearrange_ops`／`bool_ops` と対になる命名）

各関数は `crate::var::Var::scalar_unary`（`pub(crate)`。forward の
実体化・バックエンド dispatch・tape 記録を担う共通経路）へ
`ScalarUnaryOp` の該当 variant を渡すだけの 1 行委譲であり、新規
`Op`・`BackendOps` メソッドは追加していない。

**`crates/backend-cpu/src/scalar_elementwise.rs` は変更していない**:
同モジュールの汎用ループ（`scalar_unary_slice`）は `ScalarUnaryOp::
apply` を直接呼ぶため、`tensor-core` 側に variant を追加するだけで
CPU 参照実装が自動的に新 7 kind をカバーする（専用スライス関数の
追加が不要という同モジュールの設計動機どおり）。

## §3 数値契約

forward 数式の単一情報源は `tensor_core::scalar_op::ScalarUnaryOp::
apply`（`crates/tensor-core/src/scalar_op.rs`）である。

- `floor`／`ceil`は IEEE のまま伝播する。`round` は偶数丸め
  （`f32::round_ties_even`。PyTorch `torch.round` と同じタイブレーク
  規則で `f32::round`〈0 から遠い側への丸め〉ではない）。`sign` は
  `x > 0 → 1`／`x < 0 → -1`／`±0` を含むそれ以外 → `0`
  （`f32::signum` は `±0` に対し `±1` を返すため使わない）。`NaN`
  入力は `NaN` を返す。いずれも区分定数のため勾配は恒等的に `0`
  （`ScalarUnaryOp::is_piecewise_constant()`）
- `reciprocal`（`1/x`）・`rsqrt`（`1/sqrt(x)`）は定義域外・`0` 入力
  でも IEEE のまま（`inf`／`NaN`）panic しない。勾配係数はそれぞれ
  `-y^2`・`-0.5*y/x`（forward 記録値 `y` を再利用。既存 `Sqrt`／`Exp`
  と同型の閉形式）
- `erf` は `Gelu`（誤差関数版）と同じ `f64` 精度の自作近似
  （Abramowitz–Stegun 7.1.26。`erf_f64`）を `f64` で計算してから
  `f32` へ 1 回だけ downcast する。勾配は `2/√π・exp(-x²)`
  （`erf_grad`。同じく `f64` で計算し 1 回 downcast）
- `pow_scalar`（`x.powf(exponent)`）は既存 `PowScalar` variant
  （#1634／#1686）をそのまま使う。`exponent == 0.0` は勾配が常に `0`
  にマスクされる

**区分定数 4 種の勾配 VJP は乗算を経由しない**（`crates/autodiff/src/
grad.rs::vjp` の `Op::ScalarUnary` 分岐）。`is_piecewise_constant()`
が `true` の場合、`vjp_elementwise_mul(upstream, 0 係数)` を経由せず
入力 shape のゼロテンソルを直接生成する。これは
`ScalarBinaryOp::is_comparison`（比較演算の VJP。PR #1823）と同じ
理由で、upstream が `inf`／`NaN` を含む場合の `0.0 * inf = NaN`
汚染を避けるための設計である。

## §4 GPU 専用カーネル

新 7 kind（`PowScalar` を除く）の GPU 専用カーネルはスコープ外。
`crates/backend-cuda/src/kernels_scalar_op.rs::unary_expr`・
`crates/backend-metal/src/scalar_op_source.rs::unary_expr` はいずれも
新 7 kind に対し明示 `None` arm を持つ（末尾ワイルドカードのみに
頼らない。新 kind 追加のたびにこの match を見直すことを強制する
既存方針）。既定の `BackendError::Unsupported` から `autodiff::eval::
scalar` のホスト参照実装へフォールバックする。

## §5 PyTorch との差異

差異なし（`round` の偶数丸め・`sign(±0) = 0`・`reciprocal`／`rsqrt`
の非 panic はいずれも PyTorch と同じ挙動）。

## §6 テストの配置

- 単体テスト（forward 委譲確認）は `crates/autodiff/src/
  scalar_unary_ops.rs` 内の `#[cfg(test)] mod tests` に置いた
  （`bool_ops.rs`／`rearrange_ops.rs` と同じ配置）
- 勾配テスト（数値微分突合・区分定数の恒等ゼロ検証・`inf` upstream
  汚染回帰）は既存の `crates/autodiff/src/grad.rs` の `#[cfg(test)]
  mod tests`（`numeric_grad_cases`／`scalar_unary_piecewise_constant_
  grad_is_always_zero`／`scalar_unary_piecewise_constant_backward_
  does_not_nan_with_inf_upstream`）に追加した（既存 8 kind と同じ
  数値微分ハーネスを再利用するため）
- バックエンド間 parity は `crates/facade/tests/
  scalar_unary_ops_backend_parity.rs`（`fandhe_ai_autodiff::
  scalar_unary_ops::*` を直接 use。CPU と NaiveOps の forward bit
  完全一致・backward 突合〈属性なし 2 件〉＋CUDA／Metal の
  `#[ignore]`〈forward・backward 各 2 件。未実測。`docs/perf/logs/
  scalar-unary-ops-2145/README.md` 参照〉）
- CPU 参照実装の bit 同一性は `crates/backend-cpu/tests/
  scalar_op_parity.rs`（既存ハーネスへ新 7 kind を追加）
- `tensor-core` 側の forward 単体テスト（`apply`／`is_piecewise_
  constant`／`erf_f64` の数値検証）は `crates/tensor-core/src/
  scalar_op.rs` 内の `#[cfg(test)] mod tests`

## §7 実機実測の申し送り

CUDA（DGX Spark GB10）・Metal 実機は本エージェント実行環境に無いため
未実測。`docs/perf/logs/scalar-unary-ops-2145/README.md` へ測定
コマンド案・期待結果を申し送る。

## §8 実装記録（イシュー #2145・2026-09-25）

- `crates/tensor-core/src/scalar_op.rs`: `ScalarUnaryOp` へ
  `Floor`／`Ceil`／`Round`／`Sign`／`Reciprocal`／`Rsqrt`／`Erf` の
  7 variant を追加（`PowScalar` は #1634 で追加済みのため対象外）。
  `apply`・`kind_str`・`is_piecewise_constant()`・`erf_grad()` と
  単体テストを追加
- `crates/autodiff/src/eval/scalar.rs`: `unary_grad_factor` に新
  7 kind の分岐を追加（区分定数 4 種は `0.0`・`reciprocal`／
  `rsqrt`／`erf` はそれぞれの閉形式）
- `crates/autodiff/src/grad.rs`: `Op::ScalarUnary` の VJP に
  `is_piecewise_constant()` 分岐を追加（ゼロテンソル直接生成による
  `inf`／`NaN` 汚染回避）＋回帰テスト 2 件
- `crates/autodiff/src/scalar_unary_ops.rs`（新規）: 8 個の入口関数・
  単体テスト（`each_entry_forwards_to_expected_scalar_unary_op`）
- `crates/autodiff/src/lib.rs`: `pub mod scalar_unary_ops;` を追加
  （アルファベット順維持）
- `crates/autodiff/tests/scalar_unary_parity.rs`（新規）: 数値微分
  突合・エッジケース
- `crates/backend-cpu/tests/scalar_op_parity.rs`: 新 7 kind を既存
  ハーネスへ追加
- `crates/backend-cuda/src/kernels_scalar_op.rs`・`crates/
  backend-metal/src/scalar_op_source.rs`: 新 7 kind に対し明示
  `None` arm＋回帰テスト（`new_2145_unary_kinds_are_unsupported`）
- `crates/facade/src/lib.rs`: `VarScalarUnaryOpsHoldDoctestGuard`
  （正のプローブ doctest。`VarBoolOpsHoldDoctestGuard`／
  `VarRearrangeOpsHoldDoctestGuard` と同型）
- `crates/facade/tests/api_surface.rs`:
  `scalar_unary_ops_hold_doctest_globs_all_pub_modules`・
  `scalar_unary_ops_hold_doctest_probe_body_matches_fixed_contract`・
  `facade_does_not_reexport_or_declare_scalar_unary_ops`・
  `workspace_declares_scalar_unary_ops_fn_names_only_in_autodiff_
  scalar_unary_ops`（4 テスト。後者は `onnx-interop/src/ops/
  activation.rs::erf`〈ONNX `Erf` 演算子の無関係な実装。1 件〉を
  期待集合に明示的に含める——`erf` という名前の衝突が実在するため
  `bool_ops` 系より 1 段複雑）
- `crates/facade/tests/scalar_unary_ops_backend_parity.rs`（新規）:
  CPU と NaiveOps の forward bit 一致・backward 突合（属性なし
  2 件）＋CUDA／Metal の `#[ignore]`（forward・backward 各 2 件。
  未実測。`docs/perf/logs/scalar-unary-ops-2145/README.md` 参照）
- `docs/README.md`: 本 doc・perf log README の索引行を追加

## §9 承認事項（未承認として列挙）

1. facade 公開（`Var::floor`／`ceil`／`round`／`sign`／
   `reciprocal`／`rsqrt`／`erf`／`pow_scalar` の委譲メソッド追加と
   保留ガードの撤去）
2. GPU 専用カーネル（CUDA／Metal の `floor`／`ceil`／`round`／
   `sign`／`reciprocal`／`rsqrt`／`erf` 専用実装）
3. `create_graph`（二階微分）対応

## §10 スコープ外

- facade 公開（上記承認事項 1 と同じ）
- GPU 専用カーネル（上記承認事項 2 と同じ）
- `create_graph`（二階微分）対応（上記承認事項 3 と同じ。`crate::
  create_graph::scalar_unary_replayable` は新 7 kind に触れておらず
  末尾ワイルドカードで `false` のまま）
