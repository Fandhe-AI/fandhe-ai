# 低精度 Linear forward（TypedOps<f16／bf16>・opt-in・f32 master weight）設計・実装記録

イシュー #1960（親 #1626／#1648）。`crate::typed_ops::TypedOps<T>`（`BackendOps::typed_ops_f16`／`typed_ops_bf16` capability accessor。イシュー #1687）は `gemm`／`add`／`relu` 等の低レベル演算のみを提供し、`autodiff` から直接利用する箇所はこれまで存在しなかった（`docs/backend-dtype-dispatch-design.md` §8「`Var`／`Tape` の dtype 一般化はスコープ外」）。本イシューは Linear 層に限定して forward を f16／bf16 で計算し、パラメータ（master weight）と勾配は f32 のまま保つ経路を **opt-in** で追加する。

## 1. 背景・要件

- 既定は f32 のまま。既存 f32 経路（`LinearVars::forward`／`forward_with_activation`）は bit 同一（コード経路・出力とも不変）。
- 低精度経路は事前登録した許容（`crates/tensor-core/src/low_precision.rs` の単体テスト、`crates/autodiff/src/var.rs::linear_act_tests` の bit 一致オラクルテスト）で検証し、既存 tolerance 定数（`RELATIVE_TOLERANCE`／`ABSOLUTE_RESCUE_THRESHOLD`）・baseline は変更しない。

## 2. 数値方式

- **forward**（低精度）: `x`・`weight`・`bias`（いずれも f32 のテープ値）を要素ごとに `half::f16::from_f32`／`half::bf16::from_f32`（IEEE 最近接偶数丸め）で降格 → `TypedOps<T>::gemm` →（bias があれば）`TypedOps<T>::add`（NumPy 互換 broadcast。3 バックエンドとも f32 版と同一規則）→（`Activation::Relu` なら）`TypedOps<T>::relu` → 結果を `to_f32` で昇格し `Tensor<f32>` としてテープへ格納する。`Var`／`Tape` の dtype は f32 のまま（`docs/backend-dtype-dispatch-design.md` §8 を維持）。
- **backward**（f32）: 既存 `Op::LinearAct` の VJP（`matmul_vjp` + `reduce_bias_grad`〈f64 アキュムレータ契約〉+ ReLU マスク `out_value > 0`）をそのまま使う。入力・重みは f32 master 値、マスクは低精度 forward 出力から取る（straight-through。丸めの微分は恒等扱い）。**PyTorch autocast との意図的な差異**: autocast は backward の matmul も低精度で行うが、本実装は backward を常に f32 に保つ。
- **f32 master weight**: `Linear` の `weight`／`bias`（`Tensor<f32>`）・`Gradients`・optimizer は一切変更しない。低精度コピーは forward 呼び出しごとの一時値で、保持しない。
- **fail-closed**: `typed_ops_f16()`／`typed_ops_bf16()` が `None` のバックエンドでは f32 へ静かにフォールバックせず `BackendError::Unsupported` を返す（opt-in が精度について嘘をつかないため）。`dtype` が `F16`／`Bf16` 以外、`Activation` が `None`／`Relu` 以外（`#[non_exhaustive]`）も `BackendError::InvalidArgument` で拒否する。f16 の表現範囲超過（±inf）は IEEE 挙動としてそのまま伝播させる（`amp::GradScaler::has_non_finite` 等との併用を想定）。

## 3. 配置（依存変更・facade 公開面拡張を避けるための構成）

- `autodiff` は `half` に直接依存しておらず、`tensor-core` は既に `half` 依存を持つ。依存変更を避けるため、**f32⇔低精度の変換と `TypedOps` ディスパッチは `tensor-core`（`crates/tensor-core/src/low_precision.rs`）に置く**。
- `facade` は `Var` と `nn::LinearVars` を再エクスポートしており、`LinearVars` は pub フィールド（`weight`／`bias`）の struct literal で外部（`crates/facade/tests/mha_backend_parity.rs` 等）から構築されている。よって **`LinearVars` へのフィールド追加（破壊的）・`Var`／`LinearVars` への新規 `pub fn`（facade 公開面拡張。`docs/compat-api-scope.md` §5 の承認手続きが必要）は行わない**。opt-in 入口は facade 非再エクスポートの `fandhe_ai_autodiff::nn::linear` モジュールの自由関数 `linear_forward_low_precision(vars: &LinearVars, input: &Var, act: Activation, dtype: ScalarDType) -> Result<Var, AutodiffError>` にする。

## 4. Op 記録

- `Op::LinearAct` に `compute_dtype: ScalarDType` フィールドを追加した（新 variant は作らない）。既存 `Var::linear_act`（fresh 融合経路）は `ScalarDType::F32` を記録するのみで計算経路は不変。新設 `Var::linear_act_low_precision`（`pub(crate)`）が `F16`／`Bf16` を記録する。
- `Op::LinearAct` は checkpoint 非適格（`is_checkpoint_eligible() == false`。常に `push_eager` で実体化済みの値をそのまま使い、再計算されない）のため、`compute_dtype` が低精度でも再計算時に精度が食い違う経路は存在しない。将来 `LinearAct` を checkpoint 適格へ拡張する場合は、再計算ロジックが `compute_dtype` を見て同じ低精度経路を再実行する必要がある旨を `tape.rs` の doc comment に明記した。
- `grad::vjp` の `Op::LinearAct` 分岐は `compute_dtype` を計算には使わず、未対応 `Activation` を拒否するエラーメッセージの診断情報としてのみ読む（フィールドが dead_code 扱いにならないための正当な用途）。

## 5. 検証

- `crates/tensor-core/src/low_precision.rs`: accessor `None`（`Unsupported`）・非対応 dtype（`InvalidArgument`）・shape 不整合（accessor 取得より先に `ShapeMismatch`）の単体テスト 5 件。
- `crates/autodiff/src/var.rs::linear_act_tests`: `linear_act` が `ScalarDType::F32` を記録すること、`linear_act_low_precision` が低精度カーネル未実装バックエンド（`Tape::new()` 既定の naive 参照実装）で `AutodiffError::Backend(BackendError::Unsupported(_))` を返すことを確認する 2 件。
- 既存 f32 経路（`LinearVars::forward`／`forward_with_activation`・`crates/autodiff/tests/` 配下・`crates/facade` の融合 vs 非融合 bit 一致テスト）は無変更で pass することを確認済み（`git diff` が f32 経路の計算・記録内容に触れていないことをレビューで担保。`Op::LinearAct` の既存フィールドは順序・型とも不変で `compute_dtype` を末尾に追加したのみ）。
- 本イシューは性能目標を持たない（CPU の `TypedOps<f16/bf16>` は f32 昇格方式のためむしろ遅い）。ベンチは追加しない。

## 6. スコープ外

- facade 公開（`LinearVars`／`Var` の `pub fn` 化・`ScalarDType` 再エクスポート）→ 引き続き `docs/compat-api-scope.md` §5 のユーザー承認が前提（未取得。§7 で追加した `compat::AmpDType` は `ScalarDType` を facade へ直接持ち出さない facade ローカルの薄い写像であり、この非公開方針とは矛盾しない）。
- `DeviceParamStore` 常駐経路・`linear_forward_device` の低精度化、backward の低精度化（真の混合精度）、Linear 以外の層、fusion の dtype 対応。
- 真の f16 カーネルによる CPU 高速化（現 CPU `TypedOps` は f32 昇格方式のソフトウェア変換。`crates/backend-cpu/src/typed_f16.rs`／`typed_bf16.rs` 参照）。
- CUDA／Metal 実機での低精度 Linear forward の facade parity 実測（本実装エージェントの実行環境に実機への到達手段がないため未実施。既存 `typed_ops_f16`／`typed_ops_bf16` 自体の実機実測状況は `docs/backend-dtype-dispatch-design.md` の該当節を参照）。
- `compat::Sequential::evaluate`／`predict`／`predict_resident` の低精度化（fit の学習ループのみが対象。§7 実装計画 §8）。

## 7. GradScaler 統合・fit opt-in（#1961）

イシュー #1961・親 #1958。上記 §6 が当初スコープ外としていた「`compat::Sequential` の precision 設定・`fit()` 連携」を、`fandhe_ai::optim::GradScaler`（#1721／#1722）と統合したうえで実装した。

### 7.1 §5 手続きの適用根拠（`docs/compat-api-scope.md` §5）

AMP は Tier 2（`docs/compat-api-scope.md` §1.3「AMP」行・#1625）、`compile()`／`fit()` は Tier 1（同 §1.2・#1618）に列挙済みの機能であり、§5「Tier 1／Tier 2 に列挙済みの機能の実装は本節の再適用を要しない（1 節の各 issue の承認事項に従う）」に従う。#1625・#1618 のいずれも 2026-09-12 のユーザー承認コメントで facade 公開面（`fandhe_ai`／`compat`）の §5 手続きに基づく範囲拡張を承認済み（範囲外は tolerance／baseline 変更・依存追加・unsafe 監査省略のみ）。#1961 の概要自体が「`compat::Sequential::fit` からの opt-in」を成果物として明示しているため、この根拠に基づき facade 公開面（`compat::{AmpConfig, AmpDType}`・`Sequential::compile_with_amp`／`amp_loss_scale`）を追加した。

**残る soft spot**: `ScalarDType` 自体の facade 再エクスポート・`LinearVars`／`Var` への `pub fn` 追加（§6 参照）は引き続き未承認のまま。`AmpDType`（facade ローカルの `#[non_exhaustive] enum`）は `ScalarDType` を internal に写像するのみで、この非公開方針を回避しない。

### 7.2 演算列（AMP 有効時）

`compat::Sequential::compile_with_amp`（`crates/facade/src/compat/training.rs`）は `Compiled` に `amp: Option<AmpState>`（`dtype: ScalarDType`・`scaler: GradScaler`）を追加し、`run_fit` のバッチループを次の順序へ分岐させる:

1. `SequentialVars::forward_with_precision`（`sequential.rs`。既存 `forward` を内部実装化し `low_precision: Option<ScalarDType>` 引数を追加。`None` のときは既存実装と完全に同一の演算列・bit 同一）が `Linear` 層のみ `linear_forward_low_precision`（#1960）へ切り替える。`Linear` 以外の層は常に f32
2. **scale 前**の素の loss を `History::loss` へ記録する（非有限でもそのまま記録して overflow を可視化する）
3. `scaler.scale_loss` → `tape.backward` → `trainable_grads` → `scaler.unscale`
4. `should_skip_step()` が `true` なら `optimizer.step`／`apply_parameters` を両方スキップ（`optimizer` の `step_count` も進めない）
5. `false` なら unscale 済み勾配で `optimizer.step` → `apply_parameters`
6. 最後に必ず `scaler.update(found_non_finite)`

AMP 無効（`compiled.amp.is_none()`）のときは既存 f32 経路をそのまま通る（`if let Some(amp) = compiled.amp.as_mut() { .. } else { .. }` の `else` 分岐が AMP 導入前の演算列と完全に同一）。

### 7.3 正しさの検証

- **AC-a（bit 一致）**: `crates/facade/tests/compat_sequential_fit_amp.rs`。`compile()`（AMP なし）が AMP 配線導入前と bit 同一（`fit_without_amp_is_unchanged_by_amp_wiring`）・`compile_with_amp` が手動 `GradScaler` + `linear_forward_low_precision` ループと bit 完全一致（F16／Bf16 各 1 件）・skip／backoff の挙動が手動ループと bit 一致（`fit_amp_skip_and_backoff_matches_gradscaler_bit_exact`。最初の step が skip されパラメータ不変のまま scale が半減することを直接検証）・AMP 状態が `fit` 呼び出しをまたいで継続（`fit(4)+fit(4) == fit(8)`）・`compile_with_amp` の検証失敗時に既存 `compiled` 状態を保持することを確認済み（6 件全 pass）。
- **AC-b（MNIST 規模・REQ-2 統一複合判定）**: `crates/facade/tests/mnist_amp_low_precision_parity.rs`。784→256(ReLU)→10・batch 64・20 step（shuffle なし）の loss 系列（f32 `compile()` vs AMP `compile_with_amp`）を `fandhe_ai_backend_cpu::parity::compare`（既存 tolerance・判定式は不変）で突合。**主判定（F16）・副判定（Bf16）とも `fail_count=0/20` で達成**（実測: F16 `max_abs_diff=8.583069e-6`・`max_rel_err=2.525904e-5`／Bf16 `max_abs_diff=4.190207e-5`・`max_rel_err=1.189803e-4`。両系列とも全点有限・単調に減少）。backward は AMP 有無に関わらず常に f32 のため、GradScaler 結線自体の正しさは AC-a の bit 一致テストが独立に証明する。
- 学習ループ例（CI 実行可能なサンプル兼テスト）: `crates/facade/tests/optim_amp_low_precision_fit.rs`。`fandhe_ai` のみを import し `compile_with_amp` → `fit` の 2 呼び出しで低精度 forward + GradScaler を使った学習ループを組めることを示す。

### 7.4 スコープ外（実装計画 §8）

`compat::Sequential::evaluate`／validation／`predict`／`predict_resident` の低精度化、`SequentialVars` の公開低精度 forward、`ScalarDType` の facade 再エクスポート（引き続き §6・#1939 の範囲）、`DeviceParamStore` 常駐経路（`step_adam` 等）への AMP・低精度結線、backward の低精度化、Linear 以外の層、fit の勾配 clip 連携、CUDA／Metal 実機での実測（`fit` は CPU `tape()` 固定）、CPU `TypedOps<f16/bf16>` の真の低精度カーネル化（性能目標なし・ベンチ追加なし）。
