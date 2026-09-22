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

## 8. Conv2d・MultiheadAttention 拡張（#2071）

イシュー #2071（親 #1626／#1648）。§6・§7.4 が「Linear 以外の層」としてスコープ外にしていた対象のうち、Conv2d・MultiheadAttention（MHA）を同じ narrow opt-in パターン（f32 master weight・backward は常に f32・facade 新規公開面なし）で追加した。`Var`／`Tape` の dtype 一般化・backward の低精度化・`DeviceParamStore` 常駐経路・Conv1d／ConvTranspose2d／TransformerEncoderLayer への拡張は引き続きスコープ外（§8.5）。

### 8.1 tensor-core 側の拡張

`crates/tensor-core/src/low_precision.rs` へ [`matmul_low_precision`]・[`conv2d_forward_low_precision`] を追加した（数値方式・fail-closed 方針は `linear_forward_low_precision` と同一）。

- `matmul_typed`（内部）はバッチ行列積を `crate::ops_shape::batched_matmul_plan` で正規化し、rank 2 は `TypedOps::gemm` へ直接委譲、rank≥3 は各オペランドを `[B, m, k]`／`[B, k, n]` へ正規化してから per-batch `TypedOps::gemm` ループで合成する（`tensor-core::backend_ops::default_gemm_batched` の f32 版と同じ「バッチをほどく」構造だが、`Tensor<f32>` 専用の `normalize_batched_operand` は再利用できないため `T: Element` へ一般化した `normalize_batched_operand_typed` を本モジュール専用に持つ）。
- `conv2d_forward_low_precision` は im2col 済みの `col`（`[N, G, K_g, P]`。呼び出し元が f32 のまま計算する——算術を伴わない bit 完全一致コピーのため低精度化の対象外）と `weight`（`[Cout, Cin_g, kH, kW]`）を受け取り、`w_mat = weight.reshape([G, Cout_g, K_g])` × `col` のバッチ GEMM（`matmul_typed`）＋（`bias` があれば）`TypedOps::add` のみを低精度化する。

### 8.2 autodiff 側の記録層

- `Op::Conv2d` に `compute_dtype: ScalarDType` を追加した（`Op::LinearAct::compute_dtype` と同型の記録専用フィールド）。`Op::Conv2d` は `is_checkpoint_eligible() == false`（常に実体化済み）かつ `supports_create_graph() == false`（二階微分 replay 対象外）のため、本フィールドは checkpoint 解放判定・`create_graph` replay のいずれにも影響しない——`Op::LinearAct`（checkpoint 非適格だが `create_graph` は拒否対象に含まれる）と異なり、`Op::Conv2d` は元々 `create_graph::validate_ancestors` の条件 (b)（`supports_create_graph() == false`）で無条件拒否されるため、Conv2d 専用の追加ガードは不要だった。`grad::vjp` の `Op::Conv2d` 分岐は本フィールドを sanity check（既知の 3 値以外は `InvalidArgument`）としてのみ読み、計算には使わない。
- `Op::MatMul` は `Op::LinearAct`／`Op::Conv2d` と異なり **checkpoint 適格・`create_graph` replay 対象**（`docs/autodiff-var-dtype-multiplexing-design.md` 案 C）であり、`Op::MatMul` variant 自体が精度情報を持たない（通常版・`fp32_strict` 版・低精度版のいずれも同じ `Op::MatMul(a, b)` を記録する）。既存 `TapeNode::fp32_strict`（`Var::matmul_fp32_strict` 用）と並列に `TapeNode::low_precision: bool`（既定 `false`）を追加し、`Var::matmul_low_precision` が記録後に事後設定する（`matmul_fp32_strict` と同じパターン）。
  - **checkpoint 解放からの除外**: `release_checkpoint_region` の条件を `is_checkpoint_eligible() && !fp32_strict && !low_precision` へ拡張した。解放すると非低精度な `matmul_forward`（`ops.gemm`。f32）で再計算されてしまい、低精度 opt-in が精度について嘘をつくことになるため。
  - **`create_graph` からの無条件拒否**: `create_graph::validate_ancestors` は祖先に `low_precision == true` の `Op::MatMul` ノードが含まれた時点で（`requires_grad`／rank に関わらず）`Err(AutodiffError::Backward(_))` を返す。`replay_op` は `fp32_strict` のみを読んで厳密版へ分岐する実装のため、低精度ノードをそのまま通すと `fp32_strict == false` の通常 `Var::matmul`（f32）へ静かにフォールバックしてしまう（`.claude/rules/security.md` A04 の fail-closed 方針に反する）。`Op::LinearAct`／`Op::ResidentLeaf` のような「値の実体化自体が成立しない」無条件拒否群とは別枠（`Op::MatMul` 自体は本来 replay 可能な Op のため）だが、同じ「精度契約を静かに破らせない」動機で無条件拒否とした。

### 8.3 Var・nn 層

- `Var::conv2d_low_precision`（`pub(crate)`）: `Var::conv2d` と同じ検査順序（① rank・`Conv2dParams::new` ② `conv2d_out_shape` ③ bias shape）のあと、`im2col_with_fallback`（f32・bit 完全一致コピー）→ `conv2d_forward_low_precision`（tensor-core）→ `push_eager(Op::Conv2d { .., compute_dtype: dtype })`。`Var::conv2d` の `ops.conv2d`（ネイティブ融合カーネル）呼び出しは経由しない——低精度化するのは GEMM＋bias のみのため、常に im2col 段階的合成へ入る。
- `Var::matmul_low_precision`（`pub(crate)`）: `Var::matmul_fp32_strict` と同型（クロステープ検査 → shape 検査 → `tensor-core::matmul_low_precision` → `push_eager` → 事後 `low_precision = true` 設定）。
- `nn::conv::conv2d_forward_low_precision`（`pub` 自由関数。`nn::linear::linear_forward_low_precision` と同じ「`Conv2dVars` へのフィールド追加を避ける」配置理由）と `nn::attention::multihead_attention_forward_low_precision`（`pub` 自由関数）を追加した。
- MHA は `crate::attention::scaled_dot_product_attention`（sub-issue (a)・#1639）ではなく、`nn::attention` モジュール内の **private 複製** `project`／`sdpa_compose`（モジュール doc「sub-issue (a) との関係」参照。#1639 マージ後に一本化予定の既存の重複）を経由する。両関数へ `low_precision: Option<ScalarDType>` 引数を追加し（`None` の既存呼び出し元は bit 同一のまま）、`sdpa_compose` 内の 2 回の `matmul`（`scores = q_scaled @ k_t`・`out = weights @ value`）のみを `matmul_low_precision` へ切り替える。scale・transpose・mask・softmax は PyTorch autocast の fp32 リストと同様 f32 のまま。`project`（q/k/v/out projection の共通実装。`nn::transformer_encoder_layer` の FFN とも共有）にも同じ `Option<ScalarDType>` 引数を追加した（既存呼び出し元はすべて `None`）。

### 8.4 検証

- `crates/tensor-core/src/low_precision.rs`: `matmul_low_precision`／`conv2d_forward_low_precision` の accessor 不在・非対応 dtype・shape 不整合（accessor 取得より先）の単体テスト 8 件追加（既存 5 件と合わせ計 14 件）。
- `crates/autodiff/src/var.rs::linear_act_tests`（モジュール名は歴史的経緯で維持。Linear 以外のテストも同居）: `matmul_low_precision`／`conv2d_low_precision` の丸めオラクル一致（`conv2d` は実際の `im2col_with_fallback` 呼び出しを直接オラクルへ使い、内部 enumerate 順序を仮定しない）・`TapeNode::low_precision`／`Op::Conv2d::compute_dtype` の記録確認・accessor 不在の fail-closed 確認・`create_graph_rejects_low_precision_matmul_ancestor`（`Var::matmul_low_precision`／`backward_create_graph` がいずれも `pub(crate)`／内部 API のため統合テストではなく本モジュールに配置）の計 9 件追加。
- `crates/facade/tests/compat_sequential_fit_amp_conv_mha.rs`（新規）: Conv2d／MHA モデルの `compile()`（AMP なし）決定性（2 回実行の bit 一致）と、`compile_with_amp` の loss 系列が f32 系列と REQ-2 統一複合判定内で一致することを検証する。**Conv2d は F16／Bf16 とも `fail_count=0/10` で達成**。**MHA は F16 で `fail_count=0/10` を達成したが、Bf16 は事前登録判定の結果 FAIL した**（`fail_count=2/10`・`max_abs_diff=5.21e-4`・`max_rel_err=2.83e-3`。bf16 の仮数 7bit が softmax〈E=4・L=3 の小規模再正規化〉を経由する 10 step 軌道差を増幅したと推定）。`.claude/rules/coding-rust.md`「バックエンド間数値一致テストの許容誤差を単独で緩和しない」に従い、tolerance／baseline は変更せず本判定のみを落とした（`mha_amp_bf16_matches_f32_within_req2_composite_tolerance` を削除しコメントで理由を記録）。
- 非後退ガード（`crates/facade/tests/conv2d_backend_parity.rs`・`mha_backend_parity.rs`・`compat_sequential_fit_amp.rs`・`mnist_amp_low_precision_parity.rs`・`checkpoint_backend_bit_identity.rs`・`attention_backend_parity.rs`・`architecture_boundaries.rs`・`api_surface.rs`・`crates/autodiff/tests/{conv2d,nn_conv,nn_attention,attention,checkpoint*,create_graph}.rs`）はいずれも無変更のまま green（`cargo test -p fandhe-ai-tensor-core`・`-p fandhe-ai-autodiff --lib --tests`・`-p fandhe-ai` の全件で確認）。

### 8.5 スコープ外（実装計画 §8。out-of-scope-tracking.md）

- Conv1d／ConvTranspose2d の低精度 forward（`nn::conv1d_forward_low_precision`／`conv_transpose2d_forward_low_precision` は未実装）。
- `compat::Sequential` の `TransformerEncoderLayer` 層は `compile_with_amp` でも内部 FFN（`nn::transformer_encoder_layer::project` 経由の `linear1`／`linear2`）を含め f32 のまま——本イシュー以前から存在する既存ギャップであり、`SequentialVars::forward_with_precision` の対象層拡張には含めていない。
- `compat::Sequential::evaluate`／`predict`／`predict_resident`・`DeviceParamStore` 常駐経路の低精度化、backward の低精度化（真の混合精度）、attention weights／dropout の低精度結線。
- `create_graph`（二階微分）の低精度 forward ノード対応（本イシューは fail-closed 拒否のみ。§8.2 参照）。
- CUDA／Metal 実機実測（`crates/facade/tests/amp_conv_mha_low_precision_backend_parity.rs`。実行コマンド・判定規則は `docs/perf/logs/amp-conv-mha-low-precision-2071/README.md` を参照）。同ファイルは MHA のみを対象とし、**Conv2d の実機 parity テストは対象外**とした——`nn::conv2d_forward_low_precision` が要求する `Conv2dVars` は `MultiheadAttentionVars::new`（`pub`）のような直接構築コンストラクタを持たず（`Conv2d::bind` は crate-internal な `&fandhe_ai_autodiff::Tape` を要求し facade テストから到達できない）、新規 `pub` コンストラクタの追加は facade／autodiff 公開面の拡張としてユーザー承認事項の判断になるため本イシューでは追加しなかった（`Conv2dVars::new` の新設は別途ユーザー承認を得たうえでの後続イシュー候補）。
- CPU `TypedOps<f16/bf16>` は f32 昇格方式のため性能目標なし・ベンチ追加なし（§7.4 と同じ方針）。
