//! 演算ごとの勾配関数（VJP: vector-Jacobian product）と `Op` 単位の
//! ディスパッチ入口 `vjp()`。
//!
//! TASK-1.5a（#16）が記録したテープ構造（`tape::Op`/`TapeNode`）に対し、
//! 「出力側勾配（upstream）→ 各入力 `NodeId` への勾配」の変換層を提供
//! する（spec 根拠: `docs/spec/05-tasks.md` TASK-1.5、
//! `docs/spec/03-poc/poc-v2-2-autodiff/code/rust/src/tape.rs` の
//! backward 実装）。`Tape::backward`（`backward.rs`・TASK-1.5c・#18）は
//! ノード列を発生順とは逆順に走査しながら本モジュールの `vjp()` を
//! 呼び、返り値（入力 `NodeId` ごとの勾配寄与）を蓄積する。**勾配の
//! 蓄積そのものは本モジュールの責務ではない**（`backward.rs` 側で
//! 複数の出力先から同一入力ノードへ流入する勾配を合算する）。
//!
//! 値計算は `eval.rs`（クレート非公開の暫定 CPU 参照実装）のヘルパー
//! を再利用し、forward と勾配計算で数式の実体を 2 か所に別実装しない
//! （PoC-v2-2 の方針を踏襲）。ただし `MatMul`／`LinearAct`／
//! `LinearResident` の GEMM 系 VJP（`matmul_vjp`・`Op::LinearResident`
//! の `d_weight`）はイシュー #1211 で `eval::matmul`（scalar 参照実装）
//! から `BackendOps::gemm_fp32_strict`（forward と同じ CPU BLIS／CUDA／
//! Metal カーネルを使うが、CUDA の TF32 opt-in フラグ
//! （`set_cuda_tf32_gemm_enabled`）の状態に関わらず常に FP32 厳密で
//! 計算する入口。`ops.gemm` をそのまま使うと backward が opt-in フラグ
//! に暗黙追従してしまい、`docs/cuda-tf32-optin-api-decision.md`・
//! `backend-cuda::precision` モジュール冒頭コメントの「学習経路は
//! スコープ外のまま FP32」契約に反するため区別する。codex-review
//! 指摘・PR #1223）へ切り替え済み。backward の支配的コスト（`docs/perf/
//! train-step-phase-breakdown.md` §11・§15）を forward と同じ既定 FMA
//! 契約・並列実装で計算するための変更で、`eval::matmul` は
//! `NaiveOps`／`TestOps`（compat・テスト経路）に限り引き続き使われる
//! （`docs/perf/train-backward-gemm-wiring.md`）。

use fandhe_ai_tensor_core::{
    Activation, BackendError, BackendOps, BceKind, HuberKind, ScalarBinaryOp, ScalarUnaryOp,
    ScatterReduce, ShapeError, Tensor, VectorNormOrd, row_norm_layout,
};

use crate::error::AutodiffError;
use crate::eval::{self, build_tensor, dense_vec};
use crate::tape::{
    NodeId, Op, ResidentBiasTarget, ResidentResolver, TapeId, TapeNode, materialize_fallible,
};
use crate::var::Reduction;

/// elementwise VJP（`Op::Mul`／`Op::Exp`／`Op::Tanh`／`Op::Sigmoid` の
/// 乗算、`backward.rs::accumulate` の fan-out 勾配合算）を
/// `BackendOps`（forward と同じ CPU 並列／CUDA／Metal カーネル）経由で
/// 計算するか、ホスト逐次参照実装（`eval::mul`／`eval::add`）のまま
/// にするかを切り替えるゲート（イシュー #1583）。#1211 が GEMM 系 VJP
/// （`matmul_vjp`）へ適用した「backward を forward と同じ実装で計算
/// する」方針の elementwise 版。
///
/// いずれも単一 IEEE 演算（乗算／加算 1 回）のみで縮約を含まないため、
/// `ops.mul`／`ops.add` と `eval::mul`／`eval::add` は run-to-run・
/// バックエンド間を問わず bit 同一（`.claude/rules/coding-rust.md`
/// 「バックエンド間数値一致は複合判定」が対象とする縮約系演算には
/// 該当しない）。**対象外**（本ゲートの影響を受けない・ホスト経路の
/// まま不変）: `Op::Relu`／`LinearAct`／`LinearResident` のマスク演算
/// （[`elementwise_mul_mask`]。#1577 の stride 対応 host 経路。
/// `BackendOps` にマスク演算面がなく追加は公開 trait 拡張のため別途
/// ユーザー承認事項）、`Op::Add` の broadcast 縮約（[`reduce_bias_grad`]
/// ／[`reduce_to_shape`]。f64 アキュムレータ統一・Metal 側
/// `BackendOps::sum` 未実装のため対象外）。
///
/// 実測・出荷判断の経緯は `docs/perf/elementwise-vjp-backend-ops.md`
/// を参照（事前登録規則はイシュー #1583 のコメントに固定済み）。
pub(crate) const ELEMENTWISE_VJP_VIA_BACKEND_OPS: bool = false;

/// [`ELEMENTWISE_VJP_VIA_BACKEND_OPS`] に従い `g ⊙ rhs`（elementwise
/// 積。broadcast 前提だが本関数の呼び出し元はいずれも同 shape で渡す）
/// を `ops.mul` または `eval::mul` で計算する。
///
/// フォールバックは [`BackendError::Unsupported`] の場合のみ
/// `eval::mul` へ切り替える（バックエンドがそもそも当該演算を持たない
/// 場合の救済。`matmul_vjp` と異なり無条件フォールバックを許すのは、
/// elementwise 積が単一 IEEE 演算で forward／backward・バックエンド間
/// を問わず bit 同一であり、フォールバックしても「backward だけ別の
/// 数値経路になる」ことがないため）。他のエラー（デバイス割当失敗等）
/// は `AutodiffError::Backend` として fail-closed に伝播する
/// （`.claude/rules/security.md` A08）。戻り値の shape が
/// `broadcast_shape(g, rhs)` と一致しない場合も fail-closed で
/// エラーにする（バックエンド実装のバグを静かに呑み込まない）。
fn vjp_elementwise_mul(
    ops: &dyn BackendOps,
    g: &Tensor<f32>,
    rhs: &Tensor<f32>,
) -> Result<Tensor<f32>, AutodiffError> {
    vjp_elementwise_mul_via(ops, g, rhs, ELEMENTWISE_VJP_VIA_BACKEND_OPS)
}

/// [`vjp_elementwise_mul`] の実体。ゲート値を引数として受け取ることで
/// ビルド時定数 [`ELEMENTWISE_VJP_VIA_BACKEND_OPS`] の値に関わらず
/// 両分岐を単体テストできるようにする（イシュー #1583）。
fn vjp_elementwise_mul_via(
    ops: &dyn BackendOps,
    g: &Tensor<f32>,
    rhs: &Tensor<f32>,
    via_backend_ops: bool,
) -> Result<Tensor<f32>, AutodiffError> {
    if !via_backend_ops {
        return Ok(eval::mul(g, rhs));
    }
    match ops.mul(g, rhs) {
        Ok(out) => {
            let expected = fandhe_ai_tensor_core::broadcast_shape(g.shape(), rhs.shape())
                .map_err(|err| AutodiffError::Backend(BackendError::ShapeMismatch(err)))?;
            if out.shape() != expected.as_slice() {
                return Err(AutodiffError::Backend(BackendError::ShapeMismatch(
                    ShapeError::ShapeMismatch {
                        lhs: out.shape().to_vec(),
                        rhs: expected,
                    },
                )));
            }
            Ok(out)
        }
        Err(BackendError::Unsupported(_)) => Ok(eval::mul(g, rhs)),
        Err(other) => Err(AutodiffError::Backend(other)),
    }
}

/// [`ELEMENTWISE_VJP_VIA_BACKEND_OPS`] に従い `a + b`（同 shape 前提の
/// elementwise 和。`backward.rs::accumulate` の fan-out 勾配合算専用）
/// を `ops.add` または `eval::add` で計算する。エラー処理・フォール
/// バック方針は [`vjp_elementwise_mul`] と同一。`pub(crate)`:
/// `backward.rs::accumulate` から呼ばれる。
pub(crate) fn vjp_elementwise_add(
    ops: &dyn BackendOps,
    a: &Tensor<f32>,
    b: &Tensor<f32>,
) -> Result<Tensor<f32>, AutodiffError> {
    vjp_elementwise_add_via(ops, a, b, ELEMENTWISE_VJP_VIA_BACKEND_OPS)
}

/// [`vjp_elementwise_add`] の実体。[`vjp_elementwise_mul_via`] と同じ
/// 理由でゲート値を引数化する（イシュー #1583）。
fn vjp_elementwise_add_via(
    ops: &dyn BackendOps,
    a: &Tensor<f32>,
    b: &Tensor<f32>,
    via_backend_ops: bool,
) -> Result<Tensor<f32>, AutodiffError> {
    if !via_backend_ops {
        return Ok(eval::add(a, b));
    }
    match ops.add(a, b) {
        Ok(out) => {
            let expected = fandhe_ai_tensor_core::broadcast_shape(a.shape(), b.shape())
                .map_err(|err| AutodiffError::Backend(BackendError::ShapeMismatch(err)))?;
            if out.shape() != expected.as_slice() {
                return Err(AutodiffError::Backend(BackendError::ShapeMismatch(
                    ShapeError::ShapeMismatch {
                        lhs: out.shape().to_vec(),
                        rhs: expected,
                    },
                )));
            }
            Ok(out)
        }
        Err(BackendError::Unsupported(_)) => Ok(eval::add(a, b)),
        Err(other) => Err(AutodiffError::Backend(other)),
    }
}

/// ノード 1 個分の VJP。`upstream`（出力側勾配）と記録済みノード列
/// `nodes` から、各入力 `NodeId` への勾配寄与を返す。`out_value` は
/// 当該ノードの forward 記録値で、`Exp`/`Tanh`/`Sigmoid`/`Max` が
/// 再計算を避けて再利用する（`Sigmoid` は TASK-9.1b・#92 で追加）。
/// `Op::Leaf` は入力を持たないため空 `Vec` を返す。
///
/// **TASK-12.1d（#164）**: `Add`／`Sum` の入力 shape は `TapeNode.shape`
/// （実体化なしに算出済み。`tape.rs`）から直接読み、実体化を要求しない
/// （`docs/fusion-graph-design.md` §3.5.1）。`MatMul`／`Mul`／`Relu`／
/// `Max`／`MseLoss`／`CrossEntropyLoss` は入力の実際の値を要するため、
/// forward 記録済みの未実体化ノードを [`materialize_fallible`]（層 1。
/// `run_fused` の失敗のうち `Unsupported` 以外は `?` で伝播する）経由で
/// 読む（`Var::value`〈層 2〉は呼ばない。§3.5.2）。
///
/// `resident`（イシュー #1022）: `Op::LinearResident` の VJP が
/// `weight`／`bias` のデバイス常駐バッファを取得するための
/// [`ResidentResolver`]。素の [`crate::tape::Tape::backward`] からは
/// `None` が渡り、`DeviceParamStore::backward`（`optim::device_store`）
/// 経由の呼び出し（`Tape::backward_with_resident`）でのみ `Some` になる
/// （`tape::Op::LinearResident` doc「素の `Tape::backward`（resolver
/// なし）では型付きエラー」参照）。`Op::ResidentLeaf` 自身は `Op::Leaf`
/// と同じく入力を持たないため `resident` を参照しない。
#[allow(clippy::too_many_arguments)]
pub(crate) fn vjp(
    op: &Op,
    out_value: &Tensor<f32>,
    upstream: &Tensor<f32>,
    nodes: &[TapeNode],
    ops: &dyn BackendOps,
    resident: Option<&dyn ResidentResolver>,
    // イシュー #1212 codex-review P0 追加是正: `Op::LinearResident` の
    // VJP が `ResidentResolver::fill_resident_weight_grad` へ「どの
    // テープ・どの世代を今まさに微分しているか」を伝えるため
    // （`tape::ResidentResolver::fill_resident_weight_grad` doc 参照）。
    // `backward_impl`（`backward.rs`）から差分対象 `Tape` 自身の
    // `id`／`epoch()` をそのまま渡す。
    tape_id: TapeId,
    tape_epoch: u64,
) -> Result<Vec<(NodeId, Tensor<f32>)>, AutodiffError> {
    // `Op` は `CrossEntropyLoss` の `targets: Tensor<i32>` payload
    // ゆえに `Copy` を持たない（`tape.rs::Op` doc 参照）。旧
    // `match *op`（`Copy` 前提の値コピー）を `op.clone()` に置き換え、
    // それ以外の分岐は変更しない。
    let contributions = match op.clone() {
        Op::Leaf => Vec::new(),
        Op::MatMul(a, b) => {
            let a_val = materialize_fallible(nodes, ops, a)?;
            let b_val = materialize_fallible(nodes, ops, b)?;
            let (da, db) = matmul_vjp(ops, a_val, b_val, upstream)?;
            vec![(a, da), (b, db)]
        }
        Op::Add(a, b) => {
            let a_shape = &nodes[a.0].shape;
            let b_shape = &nodes[b.0].shape;
            // イシュー #1566・PR #1659→#1665→#1666 取り込み後の追加
            // ユーザー承認（2026-09-12）: `Op::Add` の broadcast 縮約
            // のうち bias パターン（`upstream: [m, n]` → `[n]`／
            // `[1, n]` の行方向縮約。`reduce_bias_grad` の shape 構造
            // 判定と同一条件）に限り `reduce_bias_grad`（f64 相当の
            // アキュムレータ。`Op::LinearAct`／`Op::LinearResident` の
            // bias フォールバックと共通）へ委譲する。`LinearVars::
            // forward`（`nn/linear.rs`。`matmul → add` の非融合合成。
            // `nn::Linear` の既定 forward 経路）の bias 勾配はこの
            // `Op::Add` の VJP を経由するため、これまで `LinearAct`／
            // `LinearResident`（同一の bias 縮約が f64 相当）と
            // 数値方式が食い違っていた（`[1e8, 1.0, -1e8]` で結果が
            // 変わる）。条件を満たさない broadcast 形状（bias パターン
            // 以外の一般的な `Op::Add` 縮約）は `reduce_bias_grad` が
            // 内部で `reduce_to_shape`（`f32` 逐次和・任意 rank・任意軸
            // 対応）へそのまま委譲するため挙動を変えない（`reduce_bias_
            // grad` doc 参照）。
            let da = reduce_bias_grad(upstream, a_shape);
            let db = reduce_bias_grad(upstream, b_shape);
            vec![(a, da), (b, db)]
        }
        Op::Mul(a, b) => {
            let a_val = materialize_fallible(nodes, ops, a)?;
            let b_val = materialize_fallible(nodes, ops, b)?;
            let da = reduce_to_shape(&vjp_elementwise_mul(ops, upstream, b_val)?, a_val.shape());
            let db = reduce_to_shape(&vjp_elementwise_mul(ops, upstream, a_val)?, b_val.shape());
            vec![(a, da), (b, db)]
        }
        Op::Relu(a) => {
            // 劣勾配は x = 0 で 0 とする（PoC-v2-2 準拠）。NaN 入力は
            // マスク不成立（`v > 0.0` が false）となり勾配 0 を返す。
            // `upstream`（reuse backward の下流層からは非連続転置 view
            // でありうる）・`a_val` とも `elementwise_mul_mask` が
            // stride 対応で読む（イシュー #1577）。
            let a_val = materialize_fallible(nodes, ops, a)?;
            let da = elementwise_mul_mask(upstream, a_val, |v| v > 0.0);
            vec![(a, da)]
        }
        Op::Exp(a) => {
            // d/dx exp(x) = exp(x)。forward 記録値 `out_value` を
            // 再利用し `exp` を再計算しない。
            let da = vjp_elementwise_mul(ops, upstream, out_value)?;
            vec![(a, da)]
        }
        Op::Tanh(a) => {
            // d/dx tanh(x) = 1 - tanh(x)^2。同じく `out_value` を再利用。
            let factor = tanh_grad_factor(out_value);
            let da = vjp_elementwise_mul(ops, upstream, &factor)?;
            vec![(a, da)]
        }
        Op::Sigmoid(a) => {
            // d/dx sigmoid(x) = sigmoid(x) * (1 - sigmoid(x))。
            // `Exp`/`Tanh` と同じく forward 記録値 `out_value`
            // （= sigmoid(x)）を再利用し再計算しない（TASK-9.1b・#92）。
            let factor = sigmoid_grad_factor(out_value);
            let da = vjp_elementwise_mul(ops, upstream, &factor)?;
            vec![(a, da)]
        }
        Op::ScalarUnary { op: sop, input } => {
            // イシュー #1634: `ScalarUnaryOp` の汎用 VJP。`eval::scalar::
            // unary_grad_factors` が forward 記録値 `out_value` を
            // 再利用しつつ入力値 `x_val` から係数テンソルを組み立て、
            // 既存の `vjp_elementwise_mul`（イシュー #1583 のゲート
            // 付き乗算。`Exp`/`Tanh`/`Sigmoid` と同じ経路）へ渡す。
            let x_val = materialize_fallible(nodes, ops, input)?;
            let factor = eval::scalar::unary_grad_factors(x_val, out_value, sop);
            let da = vjp_elementwise_mul(ops, upstream, &factor)?;
            vec![(input, da)]
        }
        Op::ScalarBinary { op: sop, a, b } => {
            // イシュー #1634: `ScalarBinaryOp` の汎用 VJP。`eval::scalar::
            // binary_grad_factors` が broadcast 後 shape（= `out_value`
            // の shape）で `(da 係数, db 係数)` を返し、`Op::Add` と
            // 同じ `reduce_bias_grad`（f64 相当のアキュムレータ）で
            // 元の `a`/`b` shape へ縮約する（§3.5 設計方針）。
            let a_val = materialize_fallible(nodes, ops, a)?;
            let b_val = materialize_fallible(nodes, ops, b)?;
            if sop.is_comparison() {
                // 比較演算（`gt`／`ge`／`lt`／`le`／`eq`／`ne`）は区分
                // 定数で両入力の勾配が恒等的にゼロ（`binary_partials`
                // が常に `(0.0, 0.0)` を返す設計）。ここで
                // `vjp_elementwise_mul(upstream, 0 係数)` を経由すると
                // `upstream` が `inf`／`NaN` を含む場合（例:
                // `sum(x * x.gt(c))` の直通経路が生む `inf`）に
                // `0.0 * inf = NaN` へ汚染されてしまう（codex-review
                // 指摘・PR #1823）。乗算を経由せず各入力 shape の
                // ゼロテンソルを直接返し、upstream の値に関わらず
                // 常に有限のゼロ勾配を保証する。
                let da = build_tensor(vec![0.0f32; a_val.numel()], a_val.shape());
                let db = build_tensor(vec![0.0f32; b_val.numel()], b_val.shape());
                vec![(a, da), (b, db)]
            } else {
                let (factor_a, factor_b) =
                    eval::scalar::binary_grad_factors(a_val, b_val, out_value, sop);
                let da = reduce_bias_grad(
                    &vjp_elementwise_mul(ops, upstream, &factor_a)?,
                    a_val.shape(),
                );
                let db = reduce_bias_grad(
                    &vjp_elementwise_mul(ops, upstream, &factor_b)?,
                    b_val.shape(),
                );
                vec![(a, da), (b, db)]
            }
        }
        Op::Softmax { input, dim } => {
            // d/dx softmax(x) = y ⊙ (g − Σ_dim(g ⊙ y))（`y` = forward
            // 記録値 `out_value` = softmax(x)。`Exp`/`Sigmoid` と同じ
            // 「再計算しない」方針）。軸方向の縮約は f64 アキュムレータ
            // （要素積は f32 で確定してから f64 へ昇格。
            // `.claude/rules/coding-rust.md`「勾配の長軸縮約は f64
            // アキュムレータで統一する」）。
            let da = softmax_vjp_along(out_value, upstream, dim);
            vec![(input, da)]
        }
        Op::LogSoftmax { input, dim } => {
            // d/dx log_softmax(x) = g − exp(y) ⊙ Σ_dim(g)（`y` = forward
            // 記録値 `out_value` = log_softmax(x)）。軸方向の縮約
            // （`Σ_dim(g)`）は f64 アキュムレータ。
            let da = log_softmax_vjp_along(out_value, upstream, dim);
            vec![(input, da)]
        }
        Op::Cumsum { input, dim } => {
            // d_x[i] = Σ_{j>=i} g[j]（`dim` 方向の逆順累積和。イシュー
            // #1731）。`out_value` は使わない（`cumsum` の VJP は
            // upstream のみから決まる線形演算のため）。
            let da = cumsum_vjp_along(upstream, dim);
            vec![(input, da)]
        }
        Op::Cumprod { input, dim } => {
            // 除算を用いない厳密形（`cumprod_vjp_along` doc 参照。
            // イシュー #1731）。forward 記録値 `out_value` ではなく
            // 入力 `x` 自体が必要なため `Op::Max` と同型に
            // `materialize_fallible` で取得する。
            let x_val = materialize_fallible(nodes, ops, input)?;
            let da = cumprod_vjp_along(x_val, upstream, dim);
            vec![(input, da)]
        }
        Op::Sum { input, dim } => {
            let input_shape = &nodes[input.0].shape;
            let da = unreduce_broadcast(upstream, input_shape, dim);
            vec![(input, da)]
        }
        Op::Max { input, dim } => {
            let input_val = materialize_fallible(nodes, ops, input)?;
            let da = max_vjp(input_val, dim, out_value, upstream);
            vec![(input, da)]
        }

        Op::Var {
            input,
            dim,
            correction,
        } => {
            let input_val = materialize_fallible(nodes, ops, input)?;
            let da = var_vjp(input_val, dim, correction, upstream);
            vec![(input, da)]
        }
        Op::VectorNorm { input, ord, dim } => {
            let input_val = materialize_fallible(nodes, ops, input)?;
            let da = vector_norm_vjp(input_val, ord, dim, upstream);
            vec![(input, da)]
        }
        Op::Std {
            input,
            dim,
            correction,
        } => {
            let input_val = materialize_fallible(nodes, ops, input)?;
            let da = std_vjp(input_val, dim, correction, upstream);
            vec![(input, da)]
        }
        Op::Min { input, dim } => {
            // `extremum_first_match_vjp`（`Op::Max` の `max_vjp` と
            // 共有する実体）を直接呼ぶ: forward 記録値 `out_value` と
            // `==` 一致する最初の位置へ上流勾配を置くだけで最大／最小
            // どちらの縮約かに依存しないため（`Op::Min` doc 参照）。
            let input_val = materialize_fallible(nodes, ops, input)?;
            let da = extremum_first_match_vjp(input_val, dim, out_value, upstream);
            vec![(input, da)]
        }
        Op::Mean { input, dim } => {
            // `d(mean)/d(x_i) = 1/n`（`n` は forward〈`Var::mean`〉と
            // 同じ縮約対象要素数）。`Sum` の VJP（複製）を `n` で割った
            // ものに等しいため `unreduce_broadcast` を再利用する
            // （`mean_vjp` 参照）。
            let input_shape = &nodes[input.0].shape;
            let da = mean_vjp(upstream, input_shape, dim);

            vec![(input, da)]
        }
        Op::MseLoss {
            pred,
            target,
            reduction,
        } => {
            let pred_val = materialize_fallible(nodes, ops, pred)?;
            let target_val = materialize_fallible(nodes, ops, target)?;
            let n = pred_val.numel();
            let (dpred, dtarget) = if n == 0 {
                // `mse_loss_vjp` と同じゼロ除算回避（`scale` 計算前に
                // 早期 return。融合カーネル呼び出しを回避することで
                // `n == 0` を渡すバックエンド実装契約を単純に保つ）。
                let zeros = build_tensor(vec![0f32; 0], pred_val.shape());
                (zeros.clone(), zeros)
            } else {
                let g_value = dense_vec(upstream).first().copied().unwrap_or(0.0);
                let scale = mse_loss_scale(g_value, n, reduction);
                match ops.mse_loss_backward(pred_val, target_val, scale) {
                    Ok(dpred) => {
                        if dpred.shape() != pred_val.shape() {
                            return Err(AutodiffError::Backend(BackendError::ShapeMismatch(
                                fandhe_ai_tensor_core::ShapeError::ShapeMismatch {
                                    lhs: dpred.shape().to_vec(),
                                    rhs: pred_val.shape().to_vec(),
                                },
                            )));
                        }
                        // `dTarget = −dPred`（`backend_ops.rs::BackendOps::
                        // mse_loss_backward` doc 参照）。カーネル側は
                        // `dPred` のみを計算する契約のため、符号反転は
                        // ホスト側の単純な逐次 map（新規 GPU カーネル
                        // 起動・D2H を増やさない）で行う。
                        let dtarget_data: Vec<f32> =
                            dense_vec(&dpred).iter().map(|&v| -v).collect();
                        let dtarget = build_tensor(dtarget_data, dpred.shape());
                        (dpred, dtarget)
                    }
                    Err(BackendError::Unsupported(_)) => {
                        mse_loss_vjp(pred_val, target_val, upstream, reduction)
                    }
                    Err(other) => return Err(AutodiffError::Backend(other)),
                }
            };
            vec![(pred, dpred), (target, dtarget)]
        }
        Op::HuberLoss {
            pred,
            target,
            kind,
            delta,
            reduction,
        } => {
            let pred_val = materialize_fallible(nodes, ops, pred)?;
            let target_val = materialize_fallible(nodes, ops, target)?;
            let n = pred_val.numel();
            let (dpred, dtarget) = if n == 0 {
                // `huber_loss_vjp` と同じゼロ除算回避（`Op::MseLoss`
                // 分岐と同じ理由で `scale` 計算前に早期 return）。
                let zeros = build_tensor(vec![0f32; 0], pred_val.shape());
                (zeros.clone(), zeros)
            } else {
                let g_value = dense_vec(upstream).first().copied().unwrap_or(0.0);
                let scale = huber_loss_scale(g_value, n, reduction);
                match ops.huber_loss_backward(pred_val, target_val, kind, delta, scale) {
                    Ok(dpred) => {
                        if dpred.shape() != pred_val.shape() {
                            return Err(AutodiffError::Backend(BackendError::ShapeMismatch(
                                fandhe_ai_tensor_core::ShapeError::ShapeMismatch {
                                    lhs: dpred.shape().to_vec(),
                                    rhs: pred_val.shape().to_vec(),
                                },
                            )));
                        }
                        // `dTarget = −dPred`（`Op::MseLoss` 分岐と同じ理由。
                        // `backend_ops.rs::BackendOps::huber_loss_backward`
                        // doc 参照）。
                        let dtarget_data: Vec<f32> =
                            dense_vec(&dpred).iter().map(|&v| -v).collect();
                        let dtarget = build_tensor(dtarget_data, dpred.shape());
                        (dpred, dtarget)
                    }
                    Err(BackendError::Unsupported(_)) => {
                        huber_loss_vjp(pred_val, target_val, upstream, kind, delta, reduction)
                    }
                    Err(other) => return Err(AutodiffError::Backend(other)),
                }
            };
            vec![(pred, dpred), (target, dtarget)]
        }
        Op::BceLoss {
            input,
            target,
            kind,
            reduction,
        } => {
            let input_val = materialize_fallible(nodes, ops, input)?;
            let target_val = materialize_fallible(nodes, ops, target)?;
            let n = input_val.numel();
            let (dinput, dtarget) = if n == 0 {
                // `mse_loss` の `Op::MseLoss` 分岐と同じゼロ除算回避
                // （`scale` 計算前に早期 return）。
                let zeros = build_tensor(vec![0f32; 0], input_val.shape());
                (zeros.clone(), zeros)
            } else {
                let g_value = dense_vec(upstream).first().copied().unwrap_or(0.0);
                let scale = bce_loss_scale(g_value, n, reduction);
                match ops.bce_loss_backward(input_val, target_val, kind, scale) {
                    Ok(dinput) => {
                        if dinput.shape() != input_val.shape() {
                            return Err(AutodiffError::Backend(BackendError::ShapeMismatch(
                                fandhe_ai_tensor_core::ShapeError::ShapeMismatch {
                                    lhs: dinput.shape().to_vec(),
                                    rhs: input_val.shape().to_vec(),
                                },
                            )));
                        }
                        // `dTarget` は `dInput` と対称な単純合成
                        // （符号反転）ではないため、`BackendOps::
                        // bce_loss_backward` の doc 契約どおりホスト側の
                        // 逐次 map で求める（新規 GPU カーネル起動・
                        // D2H を増やさない）。
                        let input_data = dense_vec(input_val);
                        let target_data = dense_vec(target_val);
                        let dtarget_data: Vec<f32> = input_data
                            .iter()
                            .zip(target_data.iter())
                            .map(|(&p, &y)| scale * eval::bce_elem_grad_target(p, y, kind))
                            .collect();
                        let dtarget = build_tensor(dtarget_data, dinput.shape());
                        (dinput, dtarget)
                    }
                    Err(BackendError::Unsupported(_)) => {
                        bce_loss_vjp(input_val, target_val, kind, upstream, reduction)
                    }
                    Err(other) => return Err(AutodiffError::Backend(other)),
                }
            };
            vec![(input, dinput), (target, dtarget)]
        }
        Op::CrossEntropyLoss {
            logits,
            targets,
            class_dim,
            reduction,
        } => {
            let logits_val = materialize_fallible(nodes, ops, logits)?;
            let dlogits =
                cross_entropy_loss_vjp(logits_val, &targets, class_dim, reduction, upstream);
            // `targets` は非追跡（`Var`/`NodeId` を持たない）ため勾配
            // 寄与を返すのは `logits` の 1 系統のみ（`tape::Op::
            // CrossEntropyLoss` doc 参照）。
            vec![(logits, dlogits)]
        }
        // デバイス常駐パラメータの葉（イシュー #1022）。`Op::Leaf` と同じく
        // 入力を持たないため寄与なし（`tape::Op::ResidentLeaf` doc 参照）。
        Op::ResidentLeaf { .. } => Vec::new(),
        // デバイス常駐 weight（・bias）で forward した Linear 相当ノード
        // （イシュー #1022）。`resident`（`ResidentResolver`）経由でしか
        // `weight` の `DeviceBuffer<f32>` を取得できないため、`None` の
        // 場合は型付きエラーで拒否する（`tape::Op::LinearResident` doc
        // 「素の `Tape::backward`（resolver なし）では型付きエラー」）。
        Op::LinearResident {
            input,
            weight,
            bias,
            act,
        } => {
            let Some(resident) = resident else {
                return Err(AutodiffError::InvalidArgument(
                    "grad::vjp: Op::LinearResident requires DeviceParamStore::backward (a plain \
                     Tape::backward cannot resolve the resident weight buffer)"
                        .to_string(),
                ));
            };
            // イシュー #1022 P1 是正（codex-review 指摘）: `weight`／
            // `bias` の `NodeId` は `DeviceParamStore::
            // register_resident_params`／`snapshot_resident_params` が発行した
            // `ResidentLeaf` から来るが、`ResidentLeaf` 自体はライフタイム
            // 引数のみで `Tape` の同一性を保証しない（`optim::device_store::
            // ResidentLeaf::tape_id` 検証は `linear_forward` 側の別途対応。
            // `optim/device_store.rs` モジュール冒頭参照）。ここでは
            // 縦深防御として `nodes[weight.0]` の直接添字アクセス（別
            // テープの葉が混入した場合に範囲外添字 panic・無関係ノード
            // 誤読の余地があった）を `nodes.get(...)` へ置き換え、
            // fail-closed に拒否する（`.claude/rules/security.md` A08）。
            let weight_node = nodes.get(weight.0).ok_or_else(|| {
                AutodiffError::InvalidArgument(
                    "grad::vjp: Op::LinearResident.weight node_id is out of range for this tape \
                     (contract violation: leaf registered on a different Tape?)"
                        .to_string(),
                )
            })?;
            let (store_id, slot) = match &weight_node.op {
                Op::ResidentLeaf { store_id, slot } => (*store_id, *slot),
                _ => {
                    return Err(AutodiffError::InvalidArgument(
                        "grad::vjp: Op::LinearResident.weight does not point to an \
                         Op::ResidentLeaf node (contract violation)"
                            .to_string(),
                    ));
                }
            };
            let w_dev = resident.resident_buffer(store_id, slot)?;
            let x_val = materialize_fallible(nodes, ops, input)?;

            // epilogue activation のマスク段（イシュー #1044）。`act ==
            // Relu` の場合、フォワードで融合した ReLU の劣勾配
            // （`Op::Relu` の VJP と同じ `out_value > 0` 規約。`out_value`
            // は bias 加算後・activation 適用後の forward 記録値なので
            // ここから直接マスクを復元でき、前活性化の再計算・追加ノード
            // を必要としない）を先に適用し、以降は「非融合の
            // `Op::LinearResident`（`act: None`）の VJP と同じ勾配 `g`」
            // として扱う。`upstream` はこの層が最終出力層でない限り
            // 下流の `Op::LinearResident` d_input が返す非連続転置 view
            // でありうるが、`elementwise_mul_mask` が stride 対応で
            // 読むためコピーは発生しない（イシュー #1577）。
            let masked_upstream;
            let g: &Tensor<f32> = match act {
                Activation::None => upstream,
                Activation::Relu => {
                    masked_upstream = elementwise_mul_mask(upstream, out_value, |v| v > 0.0);
                    &masked_upstream
                }
                // `Activation` は `#[non_exhaustive]`（`tensor-core::
                // backend_ops`）のため、autodiff クレート外から見た未知の
                // 将来 variant に対しては、誤った勾配（マスクなし）を
                // 静かに返すのではなく fail-closed で拒否する
                // （`.claude/rules/security.md` A08）。
                _ => {
                    return Err(AutodiffError::InvalidArgument(format!(
                        "grad::vjp: Op::LinearResident has an unsupported Activation variant \
                         ({act:?}); the VJP mask is only defined for None/Relu"
                    )));
                }
            };

            // d_weight = x^T @ g（既存 `matmul_vjp` の `dB` と同一式。
            // `x`・`g` はいずれもホスト常駐）。イシュー #1211:
            // `ops.gemm_fp32_strict`（forward と同じ CPU BLIS／CUDA／
            // Metal カーネルを経由するが CUDA の TF32 opt-in フラグには
            // 追従しない入口。冒頭コメント参照）を経由するため、
            // `x_t`（`transpose2d` の zero-copy view）はバックエンド側の
            // `gemm`（`CpuBackendOps::gemm` は `gemm_fp32_strict` の既定
            // 実装がそのまま委譲する）実装依存で扱いが変わる。CPU は
            // イシュー #1213 で dense な転置 view（`strides() == [1,
            // shape()[0]]`）を判定できる限り `contiguous()` の再パック
            // コピーを経由せず BLIS packing 側で直接吸収する専用入口
            // （TN パターン。CPU 実装クレートの `gemm_blis_parallel_tn`）
            // へ渡す（`narrow` 後の転置・TT は一般 stride 非対応のため
            // 従来どおり `contiguous()` フォールバック）。CUDA（イシュー
            // #1214）も同型の判定で GPU 側 smem 転置カーネル → 既存 NN
            // GEMM カーネルへ渡す専用入口（`CudaGemm::run_tiled_f32_tn`）
            // を持つ。Metal（イシュー #1215）は片側転置（NT/TN）を
            // `layout::classify_2d` で分類できる場合に限り、`contiguous()`
            // を経由せず classic strided カーネル入口
            // （`gemm::MetalGemm::dispatch_strided_bias_act_prepared`）へ
            // 分岐する（既存 NN 経路 `dispatch_auto` とは別カーネルの
            // ため、数値契約は bit 一致ではなく REQ-2 統一複合判定。
            // `docs/matmul-vjp-zero-copy-decision.md` §4.4）。TT・分類
            // 不能形状は Metal でも従来どおり `contiguous()` を経由する。
            let x_t = transpose2d(x_val);

            // イシュー #1212: d_weight をホストへ戻さずデバイス常駐の
            // まま `resolver`（`DeviceParamStore`）の grad staging へ
            // 直接書き込めるか試みる（`ResidentResolver::
            // fill_resident_weight_grad`。既定 `Ok(false)`）。成功した
            // 場合、`weight` の勾配は `contributions` に含めない
            // （`Gradients::get()` からは「未到達」と区別できなくなる
            // が、公開 API から `Op::ResidentLeaf` の `Var` を得る経路は
            // 元々存在しないため実害はない。`tape::Op::ResidentLeaf`
            // doc・`optim::device_store::ResidentLeaf` doc 参照。
            // `DeviceParamStore::step` は自身の grad staging を直接
            // 参照するため `Gradients` 経由の読み出しを必要としない）。
            // `Unsupported`（バックエンドが `gemm_fp32_strict_into`／
            // `MemoryOps` を実装しない。現時点で CUDA／Metal はここに
            // 該当する）の場合のみ、従来どおりホスト経路
            // （`ops.gemm_fp32_strict`）へフォールバックする（判定迂回
            // を作らない。`.claude/rules/security.md` A08）。
            // イシュー #1212 codex-review P0 追加是正: `weight`
            // （`Op::LinearResident.weight`）は今まさに差分している
            // `weight_node` の `NodeId` そのもの。`tape_id`／
            // `tape_epoch` と併せて resident 書き込みの由来として
            // 実装側（`DeviceParamStore`）へ渡す（`ResidentResolver::
            // fill_resident_weight_grad` doc 参照）。
            //
            // イシュー #1563: この `fill_resident_weight_grad`
            // （encode-only。Metal では GPU コマンドバッファへ積むだけ
            // で同期しない）を、下の `gemm_resident_lhs`（d_input。
            // Metal では同期点を持つ）より **前** に呼ぶ。d_weight と
            // d_input は独立な計算（`x^T @ g` と `W @ g^T`）であり
            // どちらを先に encode しても出力は bit 同一（構造的に
            // 保証される順序無依存性）。この順序により、同じ層の
            // d_weight の GPU コマンドが d_input の同期点へ「合流」し、
            // 層ごとに開きっぱなしだったコマンドバッファが d_input の
            // 同期 1 回で一緒に flush・wait される（Metal の同期境界
            // 回収。`docs/backend-metal-command-batching-design.md`
            // §7.4）。CPU／CUDA は本経路に同期境界を持たないため本質
            // 的な影響はない（CUDA の `gemm_fp32_strict_into` NT/TN は
            // 内部 `stream.synchronize()` を持つが性能中立）。
            //
            // イシュー #1566: bias の `Op::ResidentLeaf` 解決を
            // `fill_resident_weight_grad` 呼び出しより前に行う（bias も
            // 同時に resident staging へ書き込めるか試みるため。
            // `docs/backend-metal-command-batching-design.md` §10
            // 「案 A′」）。bias が `Some` でも `Op::ResidentLeaf` でない
            // ／`store_id` が weight と異なる場合は `bias_target` を
            // `None` のままにし、bias は常にホスト `reduce_bias_grad`
            // フォールバックへ回す（`fill_resident_weight_grad` は
            // weight のみを試み `bias_filled: false` を返す）。
            // `nodes.get(...)` は `weight` と同じ理由（範囲外添字 panic
            // 防止・fail-closed）で経由する。
            let bias_node = match bias {
                Some(bias_id) => Some(nodes.get(bias_id.0).ok_or_else(|| {
                    AutodiffError::InvalidArgument(
                        "grad::vjp: Op::LinearResident.bias node_id is out of range for this \
                         tape (contract violation: leaf registered on a different Tape?)"
                            .to_string(),
                    )
                })?),
                None => None,
            };
            let bias_target = match (bias, bias_node) {
                (Some(bias_id), Some(node)) => match &node.op {
                    Op::ResidentLeaf {
                        store_id: bias_store_id,
                        slot: bias_slot,
                    } if *bias_store_id == store_id => Some(ResidentBiasTarget {
                        slot: *bias_slot,
                        node_id: bias_id,
                        shape: node.shape.clone(),
                    }),
                    // 別 store の葉、または `Op::ResidentLeaf` 以外
                    // （理論上到達しないはず——`DeviceParamStore::
                    // linear_forward` は bias も `ResidentLeaf` としてのみ
                    // 受け付ける——だが fail-closed に「resident 化を
                    // 試みない」側へ倒す。誤った勾配を書き込むより安全）。
                    _ => None,
                },
                _ => None,
            };

            let outcome = resident.fill_resident_weight_grad(
                ops,
                store_id,
                slot,
                tape_id,
                tape_epoch,
                weight,
                &x_t,
                g,
                bias_target,
            )?;

            // d_input^T = W @ g^T（`W: [k,n]`・`g: [m,n]` → `g^T: [n,m]`
            // → `tmp: [k,m]`）。`W` はデバイス常駐のまま
            // `ops.gemm_resident_lhs` へ渡し、ホストへ download しない
            // （本イシューの受け入れ条件の中核）。イシュー #1563: 本
            // 呼び出しの同期点は、直前に encode-only で積んだ同じ層の
            // d_weight のコマンドバッファも合流させて完了させる合流点
            // になる（`crates/backend-metal/src/ops.rs::
            // gemm_resident_lhs` doc 参照）。
            let g_t = transpose2d(g);
            let tmp = ops
                .gemm_resident_lhs(w_dev, &g_t)
                .map_err(AutodiffError::Backend)?;
            let d_input = transpose2d(&tmp);

            let mut contributions = vec![(input, d_input)];
            if !outcome.weight_filled {
                let d_weight = ops
                    .gemm_fp32_strict(&x_t, g)
                    .map_err(AutodiffError::Backend)?;
                contributions.push((weight, d_weight));
            }
            if let (Some(bias_id), Some(bias_node)) = (bias, bias_node)
                && !outcome.bias_filled
            {
                // bias の勾配は `Op::Add` の VJP と同じ縮約の基本形
                // （行方向ブロードキャストの逆演算）だが、`reduce_bias_
                // grad`（f64 アキュムレータ経由。上記 doc 参照）へ委譲
                // する——resident 経由で書き込めた場合（`outcome.
                // bias_filled`）はここへ来ない（weight と対称の
                // 「resident 成功時は無駄な計算をスキップする」最適化。
                // `fill_resident_weight_grad` doc 参照）が、resident
                // 非対応バックエンドのフォールバックがここに来るため、
                // resident 経路（f64 逐次和）と数値方式を揃える
                // 必要がある（イシュー #1566・PR #1659 codex-review P1）。
                let d_bias = reduce_bias_grad(g, &bias_node.shape);
                contributions.push((bias_id, d_bias));
            }
            contributions
        }
        Op::LinearAct {
            input,
            weight,
            bias,
            act,
        } => {
            let w_val = materialize_fallible(nodes, ops, weight)?;
            let x_val = materialize_fallible(nodes, ops, input)?;

            // epilogue activation のマスク段（`Op::LinearResident` と同じ
            // `out_value > 0` 規約。イシュー #1044）。`upstream` が
            // 非連続転置 view の場合の扱いは `Op::LinearResident` 分岐
            // の同型コメント（イシュー #1577）を参照。
            let masked_upstream;
            let g: &Tensor<f32> = match act {
                Activation::None => upstream,
                Activation::Relu => {
                    masked_upstream = elementwise_mul_mask(upstream, out_value, |v| v > 0.0);
                    &masked_upstream
                }
                // `Activation` は `#[non_exhaustive]`（`tensor-core::
                // backend_ops`）のため、autodiff クレート外から見た未知の
                // 将来 variant に対しては、誤った勾配（マスクなし）を
                // 静かに返すのではなく fail-closed で拒否する
                // （`.claude/rules/security.md` A08）。
                _ => {
                    return Err(AutodiffError::InvalidArgument(format!(
                        "grad::vjp: Op::LinearAct has an unsupported Activation variant \
                         ({act:?}); the VJP mask is only defined for None/Relu"
                    )));
                }
            };

            let (d_input, d_weight) = matmul_vjp(ops, x_val, w_val, g)?;
            let mut contributions = vec![(input, d_input), (weight, d_weight)];
            if let Some(bias_id) = bias {
                let bias_shape = &nodes[bias_id.0].shape;
                // `Op::LinearResident` の resident フォールバック（上記
                // `reduce_bias_grad` doc 参照）と数値方式を揃える
                // （fresh〈本 Op〉/reuse 間の一致。イシュー #1566・PR
                // #1659 codex-review P1）。
                let d_bias = reduce_bias_grad(g, bias_shape);
                contributions.push((bias_id, d_bias));
            }
            contributions
        }
        // view ノード（イシュー #1047・親 #1043「カーネル融合・autodiff
        // 実行モデルの強化」）。`Reshape`/`Transpose` は逆写像も同じ演算
        // 族（reshape は「元の shape へ戻す」・transpose は対合）で
        // 表現でき、いずれも zero-copy（`Tensor::reshape`/`transpose`
        // が `storage: Arc<Storage>` を共有するのみ）。中間バッファを
        // 持たないという本イシューの受け入れ条件は、forward（`tape.rs::
        // resolve_view`）だけでなく backward（本 VJP）でも成立する。
        Op::Reshape { input } => {
            let input_shape = &nodes[input.0].shape;
            // `upstream` は out_shape（このノード自身の shape）を持つ。
            // 非 contiguous（例: 上流に `transpose` が挟まる）な場合は
            // zero-copy な `reshape` が `ShapeError::NonContiguousReshape`
            // を返しうるため、その場合に限り `contiguous()`（明示コピー）
            // を経由してから戻す（勾配バッファ側の話であり、view ノード
            // 自身が確保を持つわけではない。zero-copy を優先する順序を
            // 明記する）。
            let da = match upstream.reshape(input_shape) {
                Ok(t) => t,
                Err(_) => upstream.contiguous().reshape(input_shape).unwrap_or_else(|_| {
                    debug_assert!(
                        false,
                        "grad::vjp: Op::Reshape の逆伝播で reshape が失敗した（forward 側の契約違反）"
                    );
                    upstream.clone()
                }),
            };
            vec![(input, da)]
        }
        Op::Transpose { input, dim0, dim1 } => {
            // transpose は対合（同じ軸で 2 回適用すると恒等）のため、
            // 逆伝播も同じ `dim0`/`dim1` で `upstream` を transpose する
            // だけで閉じる（zero-copy。`tape::Op::Transpose` doc 参照）。
            let da = upstream.transpose(dim0, dim1).unwrap_or_else(|_| {
                debug_assert!(
                    false,
                    "grad::vjp: Op::Transpose の逆伝播で transpose が失敗した（forward 側の契約違反）"
                );
                upstream.clone()
            });
            vec![(input, da)]
        }
        Op::RmsNorm { input, weight, eps } => {
            // forward（`Var::rms_norm`）記録値 `out_value` からは `weight`
            // に 0 要素があると `x`（正規化前入力）を逆算できないため、
            // `input`（と `weight` があれば `weight`）を実体化し直して
            // 行内統計（`rstd`）を再計算する（`eval::row_rms_stats` と
            // 同じ縮約精度契約。`.claude/rules/coding-rust.md`）。
            let x_val = materialize_fallible(nodes, ops, input)?.clone();
            let w_val = match weight {
                Some(w) => Some(materialize_fallible(nodes, ops, w)?.clone()),
                None => None,
            };
            let x_shape = x_val.shape().to_vec();
            let (rows, hidden) = row_norm_layout(&x_shape).unwrap_or_else(|_| {
                debug_assert!(
                    false,
                    "grad::vjp: Op::RmsNorm の row_norm_layout が forward 側の契約に反して失敗した"
                );
                (0, 0)
            });
            let x_slice = dense_vec(&x_val);
            let w_slice = w_val.as_ref().map(dense_vec);
            let dy_slice = dense_vec(upstream);
            let (dx, dw) =
                rmsnorm_vjp_rows(&x_slice, w_slice.as_deref(), eps, rows, hidden, &dy_slice);
            let mut contributions = vec![(input, build_tensor(dx, &x_shape))];
            if let (Some(w), Some(dw)) = (weight, dw) {
                contributions.push((w, build_tensor(dw, &[hidden])));
            }
            contributions
        }
        Op::LayerNorm {
            input,
            weight,
            bias,
            eps,
        } => {
            // `Op::RmsNorm` と同じ理由で `input`／`weight` を実体化し直し
            // `mean`／`rstd` を再計算する（`eval::row_ln_stats`）。
            let x_val = materialize_fallible(nodes, ops, input)?.clone();
            let w_val = match weight {
                Some(w) => Some(materialize_fallible(nodes, ops, w)?.clone()),
                None => None,
            };
            let x_shape = x_val.shape().to_vec();
            let (rows, hidden) = row_norm_layout(&x_shape).unwrap_or_else(|_| {
                debug_assert!(
                    false,
                    "grad::vjp: Op::LayerNorm の row_norm_layout が forward 側の契約に反して失敗した"
                );
                (0, 0)
            });
            let x_slice = dense_vec(&x_val);
            let w_slice = w_val.as_ref().map(dense_vec);
            let dy_slice = dense_vec(upstream);
            let (dx, dw, db) = layer_norm_vjp_rows(
                &x_slice,
                w_slice.as_deref(),
                bias.is_some(),
                eps,
                rows,
                hidden,
                &dy_slice,
            );
            let mut contributions = vec![(input, build_tensor(dx, &x_shape))];
            if let (Some(w), Some(dw)) = (weight, dw) {
                contributions.push((w, build_tensor(dw, &[hidden])));
            }
            if let (Some(b), Some(db)) = (bias, db) {
                contributions.push((b, build_tensor(db, &[hidden])));
            }
            contributions
        }
        // RNN（tanh 版）セル 1 step（イシュー #1647・設計 `docs/autodiff-
        // rnn-cell-tape-design.md` 決定 1・5）。`out_value` は forward
        // 記録済みの `h_t`（= `tanh(pre)`）。`Op::Tanh` と同じ
        // `tanh_grad_factor` を再利用したのち、`gate_affine_vjp` の
        // 単方向版 `affine_vjp` を `x`/`w_ih` 側・`h_prev`/`w_hh` 側の
        // 2 回に分けて呼ぶ（RNN は列ブロック分割を持たないため
        // `col_start = 0`・`total_cols = H`）。
        Op::RnnCell {
            x,
            h_prev,
            w_ih,
            w_hh,
            b_ih,
            b_hh,
        } => {
            let x_val = materialize_fallible(nodes, ops, x)?;
            let h_prev_val = materialize_fallible(nodes, ops, h_prev)?;
            let w_ih_val = materialize_fallible(nodes, ops, w_ih)?;
            let w_hh_val = materialize_fallible(nodes, ops, w_hh)?;
            let total_cols = w_ih_val.shape().get(1).copied().unwrap_or(0);
            let factor = tanh_grad_factor(out_value);
            let d_pre = eval::mul(upstream, &factor);
            let (dx, dw_ih, db_ih) = affine_vjp(ops, x_val, w_ih_val, &d_pre, 0, total_cols)?;
            let (dh_prev, dw_hh, db_hh) =
                affine_vjp(ops, h_prev_val, w_hh_val, &d_pre, 0, total_cols)?;
            let mut contributions = vec![(x, dx), (h_prev, dh_prev), (w_ih, dw_ih), (w_hh, dw_hh)];
            if let Some(b_ih_id) = b_ih {
                contributions.push((b_ih_id, db_ih));
            }
            if let Some(b_hh_id) = b_hh {
                contributions.push((b_hh_id, db_hh));
            }
            contributions
        }
        // LSTM セルの `c_t` ノード（決定 1b）。`upstream` は
        // `backward_impl` の fan-in 蓄積により、`Op::LstmHidden` からの
        // `dc_from_h` 寄与と（多 step の場合）次 step の `Op::LstmCell`
        // からの `dc_prev` 寄与が既に合算された `dc` である。
        Op::LstmCell {
            x,
            h_prev,
            c_prev,
            w_ih,
            w_hh,
            b_ih,
            b_hh,
            gates_ifg,
        } => {
            let x_val = materialize_fallible(nodes, ops, x)?;
            let h_prev_val = materialize_fallible(nodes, ops, h_prev)?;
            let c_prev_val = materialize_fallible(nodes, ops, c_prev)?;
            let w_ih_val = materialize_fallible(nodes, ops, w_ih)?;
            let w_hh_val = materialize_fallible(nodes, ops, w_hh)?;
            let total_cols = w_ih_val.shape().get(1).copied().unwrap_or(0);
            let (d_pre_ifg, dc_prev) =
                match ops.lstm_cell_backward(&gates_ifg, c_prev_val, upstream) {
                    Ok(v) => v,
                    Err(BackendError::Unsupported(_)) => {
                        eval::lstm_cell_backward(&gates_ifg, c_prev_val, upstream)
                    }
                    Err(other) => return Err(AutodiffError::Backend(other)),
                };
            let (dx, dw_ih, db_ih) = affine_vjp(ops, x_val, w_ih_val, &d_pre_ifg, 0, total_cols)?;
            let (dh_prev, dw_hh, db_hh) =
                affine_vjp(ops, h_prev_val, w_hh_val, &d_pre_ifg, 0, total_cols)?;
            let mut contributions = vec![
                (x, dx),
                (h_prev, dh_prev),
                (c_prev, dc_prev),
                (w_ih, dw_ih),
                (w_hh, dw_hh),
            ];
            if let Some(b_ih_id) = b_ih {
                contributions.push((b_ih_id, db_ih));
            }
            if let Some(b_hh_id) = b_hh {
                contributions.push((b_hh_id, db_hh));
            }
            contributions
        }
        // LSTM セルの `h_t` ノード（決定 1b・決定 1b 追記）。`cell` が
        // 指す先が必ず `Op::LstmCell` である push 順序契約（`tape::
        // Op::LstmCell` doc）を利用し、`nodes[cell.0].op` から
        // `x`/`h_prev`/`w_ih`/`w_hh`/`b_ih`/`b_hh` の `NodeId` を読み出す
        // （決定 1b 追記。`cell` 以外を指すことは想定しないため `_` 分岐
        // では型付きエラーで fail-closed に拒否し、パニックしない）。
        Op::LstmHidden { cell, gate_o } => {
            let c_val = materialize_fallible(nodes, ops, cell)?;
            let (d_pre_o, dc) = match ops.lstm_hidden_backward(c_val, &gate_o, upstream) {
                Ok(v) => v,
                Err(BackendError::Unsupported(_)) => {
                    eval::lstm_hidden_backward(c_val, &gate_o, upstream)
                }
                Err(other) => return Err(AutodiffError::Backend(other)),
            };
            let cell_node = nodes.get(cell.0).ok_or_else(|| {
                AutodiffError::InvalidArgument(
                    "grad::vjp: Op::LstmHidden.cell node_id is out of range for this tape \
                     (contract violation)"
                        .to_string(),
                )
            })?;
            let (x, h_prev, w_ih, w_hh, b_ih, b_hh) = match &cell_node.op {
                Op::LstmCell {
                    x,
                    h_prev,
                    w_ih,
                    w_hh,
                    b_ih,
                    b_hh,
                    ..
                } => (*x, *h_prev, *w_ih, *w_hh, *b_ih, *b_hh),
                _ => {
                    return Err(AutodiffError::InvalidArgument(
                        "grad::vjp: Op::LstmHidden.cell does not point to an Op::LstmCell node \
                         (contract violation: push order invariant broken)"
                            .to_string(),
                    ));
                }
            };
            let x_val = materialize_fallible(nodes, ops, x)?;
            let h_prev_val = materialize_fallible(nodes, ops, h_prev)?;
            let w_ih_val = materialize_fallible(nodes, ops, w_ih)?;
            let w_hh_val = materialize_fallible(nodes, ops, w_hh)?;
            let total_cols = w_ih_val.shape().get(1).copied().unwrap_or(0);
            let hidden = d_pre_o.shape().get(1).copied().unwrap_or(0);
            let col_start = total_cols.saturating_sub(hidden);
            let (dx, dw_ih, db_ih) =
                affine_vjp(ops, x_val, w_ih_val, &d_pre_o, col_start, total_cols)?;
            let (dh_prev, dw_hh, db_hh) =
                affine_vjp(ops, h_prev_val, w_hh_val, &d_pre_o, col_start, total_cols)?;
            let mut contributions = vec![
                (cell, dc),
                (x, dx),
                (h_prev, dh_prev),
                (w_ih, dw_ih),
                (w_hh, dw_hh),
            ];
            if let Some(b_ih_id) = b_ih {
                contributions.push((b_ih_id, db_ih));
            }
            if let Some(b_hh_id) = b_hh {
                contributions.push((b_hh_id, db_hh));
            }
            contributions
        }
        // GRU セル 1 step（決定 1c・5。`reset_after=True` 規約）。`pre_i`
        // 側（`x`/`w_ih`）と `pre_h` 側（`h_prev`/`w_hh`）は独立した
        // GEMM のため、`d_pre_i`/`d_pre_h` をそれぞれ `affine_vjp` へ
        // 個別に渡す（RNN／LSTM の「1 個の d_pre を共有」とは異なる）。
        // `h_prev` への寄与は `dh_prev_direct`（`z` 経由の直接項）と
        // `d_pre_h` の affine 逆伝播の 2 系統あり、`backward.rs::
        // accumulate` が同一 `NodeId` への複数寄与を合算する契約
        // （本 `Vec` 内に 2 エントリを push するだけでよい）。
        Op::GruCell {
            x,
            h_prev,
            w_ih,
            w_hh,
            b_ih,
            b_hh,
            gates_rzn,
            q,
        } => {
            let x_val = materialize_fallible(nodes, ops, x)?;
            let h_prev_val = materialize_fallible(nodes, ops, h_prev)?;
            let w_ih_val = materialize_fallible(nodes, ops, w_ih)?;
            let w_hh_val = materialize_fallible(nodes, ops, w_hh)?;
            let total_cols = w_ih_val.shape().get(1).copied().unwrap_or(0);
            let (d_pre_i, d_pre_h, dh_prev_direct) =
                match ops.gru_backward(&gates_rzn, &q, h_prev_val, upstream) {
                    Ok(v) => v,
                    Err(BackendError::Unsupported(_)) => {
                        eval::gru_backward(&gates_rzn, &q, h_prev_val, upstream)
                    }
                    Err(other) => return Err(AutodiffError::Backend(other)),
                };
            let (dx, dw_ih, db_ih) = affine_vjp(ops, x_val, w_ih_val, &d_pre_i, 0, total_cols)?;
            let (dh_prev_affine, dw_hh, db_hh) =
                affine_vjp(ops, h_prev_val, w_hh_val, &d_pre_h, 0, total_cols)?;
            let mut contributions = vec![
                (x, dx),
                (w_ih, dw_ih),
                (w_hh, dw_hh),
                (h_prev, dh_prev_direct),
                (h_prev, dh_prev_affine),
            ];
            if let Some(b_ih_id) = b_ih {
                contributions.push((b_ih_id, db_ih));
            }
            if let Some(b_hh_id) = b_hh {
                contributions.push((b_hh_id, db_hh));
            }
            contributions
        }
        // 線形代数（イシュー #1621・`docs/autodiff-linalg-design.md`
        // §3.4）。三角解法・特異値スケーリングを要する VJP 本体は
        // `eval::linalg`（`f64` 内部計算。数式の実体を二重管理しない）
        // に集約し、ここでは各 Op の入出力（forward 記録値・upstream・
        // 兄弟ノードの forward 値）を渡すだけに徹する。
        Op::Inv { input } => {
            // `out_value`（forward が返す `f32` 記録値の `A^{-1}`）は
            // 再利用せず、`input` から改めて `f64` で計算する
            // （`eval::linalg::inv_vjp` doc 参照。codex-review 指摘）。
            let a_val = materialize_fallible(nodes, ops, input)?;
            let da = eval::linalg::inv_vjp(a_val, upstream)?;
            vec![(input, da)]
        }
        Op::Solve { a, b } => {
            // `out_value`（forward の解 `X` の `f32` 記録値）は再利用
            // せず、`a`／`b` から改めて `f64` で計算する
            // （`eval::linalg::solve_vjp` doc 参照。codex-review 指摘）。
            let a_val = materialize_fallible(nodes, ops, a)?;
            let b_val = materialize_fallible(nodes, ops, b)?;
            let (da, db) = eval::linalg::solve_vjp(a_val, b_val, upstream)?;
            vec![(a, da), (b, db)]
        }
        Op::Det { input } => {
            let a_val = materialize_fallible(nodes, ops, input)?;
            let g_scalar = dense_vec(upstream).first().copied().unwrap_or(0.0);
            let da = eval::linalg::det_vjp(a_val, g_scalar)?;
            vec![(input, da)]
        }
        Op::Cholesky { input } => {
            let da = eval::linalg::cholesky_vjp(out_value, upstream)?;
            vec![(input, da)]
        }
        // reduced QR の多出力ノード（イシュー #1621・`tape::Op::QrQ`
        // doc「多出力の扱い」）。`QrQ` は `dQ = upstream`・`dR = 0`、
        // `QrR` はその逆として部分寄与を計算し、`Tape::backward` が
        // `input` ノードへ合算する。
        Op::QrQ { input, r } => {
            let dr_zero = build_tensor(vec![0.0; r.numel()], r.shape());
            let da = eval::linalg::qr_vjp(out_value, &r, upstream, &dr_zero)?;
            vec![(input, da)]
        }
        Op::QrR { input, q } => {
            let dq_zero = build_tensor(vec![0.0; q.numel()], q.shape());
            let da = eval::linalg::qr_vjp(&q, out_value, &dq_zero, upstream)?;
            vec![(input, da)]
        }
        // reduced SVD の多出力ノード（`tape::Op::SvdU` doc）。3 ノード
        // それぞれが自身のコタンジェントのみ非ゼロとして部分寄与を返す。
        Op::SvdU { input, s, vh } => {
            let da = eval::linalg::svd_vjp(out_value, &s, &vh, Some(upstream), None, None)?;
            vec![(input, da)]
        }
        Op::SvdS { input, u, vh } => {
            let da = eval::linalg::svd_vjp(&u, out_value, &vh, None, Some(upstream), None)?;
            vec![(input, da)]
        }
        Op::SvdVh { input, u, s } => {
            let da = eval::linalg::svd_vjp(&u, &s, out_value, None, None, Some(upstream))?;
            vec![(input, da)]
        }
        Op::MatrixNorm { input, ord } => {
            let a_val = materialize_fallible(nodes, ops, input)?;
            let g_scalar = dense_vec(upstream).first().copied().unwrap_or(0.0);
            let da = eval::linalg::matrix_norm_vjp(a_val, ord, g_scalar)?;
            vec![(input, da)]
        }
        // `Var::permute` が記録する view ノード（イシュー #1597）。
        // 逆写像は逆置換（`inverse_permutation`）で `upstream` を
        // permute するだけで閉じる（zero-copy。`tape::Op::Permute`
        // doc 参照）。
        Op::Permute { input, perm } => {
            let inv = inverse_permutation(&perm);
            let da = upstream.permute(&inv).unwrap_or_else(|_| {
                debug_assert!(
                    false,
                    "grad::vjp: Op::Permute の逆伝播で permute が失敗した（forward 側の契約違反）"
                );
                upstream.clone()
            });
            vec![(input, da)]
        }
        // `Var::broadcast_to`（`Var::expand` はこれへ委譲）が記録する
        // view ノード（イシュー #1597）。`upstream` は out_shape
        // （ブロードキャスト後）を持つため、`Op::Add`/`Op::Mul` の
        // 暗黙ブロードキャストと**同じ数値契約**で入力 shape へ縮約
        // する（`tape::Op::BroadcastTo` doc 参照）。`reduce_bias_grad`
        // は `[1, n]` 行方向縮約等の特定パターンに限り f64 アキュムレ
        // ータ経路（`eval::reduce_bias_grad_rows`）へ委譲し、それ以外
        // は `reduce_to_shape`（f32 逐次和）へフォールバックする関数
        // であり、`Op::Add` の暗黙 broadcast 縮約（232〜233 行目）と
        // 同一の関数を呼ぶことで明示 broadcast・暗黙 broadcast 間の
        // 数値方式の食い違い（codex-review P1 是正）を防ぐ。
        Op::BroadcastTo { input } => {
            let input_shape = &nodes[input.0].shape;
            let da = reduce_bias_grad(upstream, input_shape);
            vec![(input, da)]
        }
        // `Var::cat` が記録するノード（イシュー #1598）。VJP は各入力
        // へ `upstream.narrow(dim, off_i, len_i)`（zero-copy view）を
        // 分配する（「Concat の VJP は Split（Narrow）」）。同一
        // `NodeId` の重複（`cat(&[x, x])`）は呼び出し元（`backward.rs::
        // accumulate`）が合算するため、ここでは単純に列挙する。
        Op::Concat { inputs, dim } => {
            let mut off = 0usize;
            let mut contributions = Vec::with_capacity(inputs.len());
            for input in inputs {
                let len = nodes[input.0].shape[dim];
                let da = upstream.narrow(dim, off, len).unwrap_or_else(|_| {
                    debug_assert!(
                        false,
                        "grad::vjp: Op::Concat の逆伝播で narrow が失敗した（forward 側の契約違反）"
                    );
                    upstream.clone()
                });
                contributions.push((input, da));
                off += len;
            }
            contributions
        }
        // `Var::narrow`（`split`／`split_with_sizes`／`chunk` の実体）が
        // 記録するノード（イシュー #1598）。「Split の VJP は Concat」
        // の原則どおり、選択されなかった前後の区間を zero-pad した
        // テンソルと `upstream` を `dim` で連結し入力 shape へ戻す
        // （`concat_with_fallback` を `Var::cat` の forward と共用。
        // 空区間はスキップして 1 要素連結〈恒等コピー〉に落とす）。
        Op::Narrow {
            input,
            dim,
            start,
            len,
        } => {
            let input_shape = nodes[input.0].shape.clone();
            let before_len = start;
            let after_len = input_shape[dim] - start - len;
            let mut before_shape = input_shape.clone();
            before_shape[dim] = before_len;
            let mut after_shape = input_shape.clone();
            after_shape[dim] = after_len;
            let before =
                eval::build_tensor(vec![0f32; before_shape.iter().product()], &before_shape);
            let after = eval::build_tensor(vec![0f32; after_shape.iter().product()], &after_shape);
            let mut pieces: Vec<&Tensor<f32>> = Vec::with_capacity(3);
            if before_len > 0 {
                pieces.push(&before);
            }
            pieces.push(upstream);
            if after_len > 0 {
                pieces.push(&after);
            }
            let da = concat_with_fallback(ops, &pieces, dim, &input_shape)?;
            vec![(input, da)]
        }
        // `Var::where_cond` が記録するノード（イシュー #1637）。
        // `cond` は forward 時点で実体化済みの f32 マスク
        // （`out_shape` ちょうど）を Op が保持する。本体は
        // [`where_vjp`]（単体テストから直接呼べるよう分離。
        // `matmul_vjp` と同じ切り出し方針）。
        Op::Where { cond, a, b } => {
            let a_shape = nodes[a.0].shape.clone();
            let b_shape = nodes[b.0].shape.clone();
            let (da, db) = where_vjp(&cond, upstream, &a_shape, &b_shape);
            vec![(a, da), (b, db)]
        }
        // `Var::masked_fill` が記録するノード（イシュー #1637）。
        // 本体は [`masked_fill_vjp`]。
        Op::MaskedFill { input, mask } => {
            let d_input = masked_fill_vjp(&mask, upstream);
            vec![(input, d_input)]
        }
        // `Var::pad`（イシュー #1756）。「pad の forward ⟷ narrow の
        // VJP・pad の VJP ⟷ narrow の forward」という双対性
        // （`Op::Concat`⟷`Op::Narrow` の双対性と同型）に基づき、`Op::
        // Narrow` の forward と全く同じ `narrow` 呼び出しを各軸へ
        // 連鎖適用してパディング領域を落とす（zero-copy view チェーン。
        // `value` は forward 記録値に焼き込まれ VJP には不要なため
        // `Op::Pad` は保持しない）。`narrow` の失敗は forward 側で
        // 検査済みの shape 契約違反を意味するため `Op::Narrow` の VJP
        // と同じ `?` 経由の `AutodiffError` 化で扱う。
        Op::Pad { input, pads } => {
            let input_shape = nodes[input.0].shape.clone();
            let mut d_input = upstream.clone();
            for (axis, &(before, _after)) in pads.iter().enumerate() {
                let len = input_shape[axis];
                d_input = d_input
                    .narrow(axis, before, len)
                    .map_err(AutodiffError::Shape)?;
            }
            vec![(input, d_input)]
        }
        // `Var::gather`（`Var::index_select` も同一 Op へ委譲。イシュー
        // #1776）。「Gather の VJP は scatter_add」の原則
        // （`Op::Concat`⟷`Op::Narrow` の双対性と同型）: 入力 shape の
        // ゼロテンソルへ `upstream` を `index` の位置へ加算する。
        Op::Gather { input, dim, index } => {
            let input_shape = nodes[input.0].shape.clone();
            let zeros = Tensor::zeros(&input_shape).map_err(AutodiffError::Shape)?;
            let d_input = scatter_with_fallback(
                ops,
                &zeros,
                dim,
                &index,
                upstream,
                ScatterReduce::Add,
                &input_shape,
            )?;
            vec![(input, d_input)]
        }
        // `Var::embedding`（`nn::Embedding`。イシュー #1604）。
        // `Op::Gather` と同じ「Gather の VJP は scatter_add」の原則で
        // `weight` shape のゼロテンソルへ `upstream` を `index` の
        // 位置へ加算し、`padding_idx` が `Some(p)` の場合のみ行 `p`
        // をゼロで上書きする（forward は当該行の現在値をそのまま
        // 返すが、勾配は流さないという PyTorch `nn.Embedding
        // (padding_idx=..)` の意味論。`with_row_zeroed` 参照）。
        Op::Embedding {
            weight,
            index,
            padding_idx,
        } => {
            let weight_shape = nodes[weight.0].shape.clone();
            let zeros = Tensor::zeros(&weight_shape).map_err(AutodiffError::Shape)?;
            let scattered = scatter_with_fallback(
                ops,
                &zeros,
                0,
                &index,
                upstream,
                ScatterReduce::Add,
                &weight_shape,
            )?;
            let d_weight = match padding_idx {
                Some(p) => with_row_zeroed(&scattered, p, weight_shape[1])?,
                None => scattered,
            };
            vec![(weight, d_weight)]
        }
        // `Var::scatter`／`Var::scatter_add`（イシュー #1776）。
        // `d_src` は `upstream`（shape=`input_shape`）を `index`
        // （shape=`src_shape`）で gather して求める。`scatter_out_shape`
        // は `dim` 以外の軸で `index_shape[axis] <= input_shape[axis]`
        // （縮小）を許容するが、`gather_out_shape` は `dim` 以外の軸の
        // 完全一致を要求するため、縮小されている軸がある場合は
        // `upstream` を該当軸の先頭 `src_shape[axis]` 要素へ narrow
        // してから gather する（scatter の非 `dim` 軸は index の座標系
        // がそのまま input の座標系となる〈オフセットなし〉ため、
        // 先頭からの narrow で forward の書き込み対象範囲と一致する。
        // codex-review 指摘）。
        //
        // `d_input` は `reduce` で分岐: `Add` は恒等（線形性）、
        // `Overwrite` は書き込まれた位置のみ upstream を 0 で上書きする
        // （`grad_self.scatter_(dim, index, 0)` と同じ式）。
        //
        // `Overwrite` の `d_src` はさらに、forward の決定的集約契約
        // （`ScatterReduce::Overwrite` doc: 同一出力位置への複数回
        // 書き込みは行優先走査順で「最後に処理された値」だけが出力に
        // 残る）に従い、上書きされて消えた重複書き込みへは 0 を返す
        // 必要がある（`scatter_overwrite_last_writer_mask`）。さもないと
        // 例えば `input=[0]／index=[0,0]／src=[2,3]` のように出力が
        // 実際には最後の書き手（`src[1]=3`）のみを反映するにも
        // 関わらず両方の `src` 要素へ勾配が流れ、数値微分と不整合に
        // なる（codex-review 指摘）。`Add` は線形なので重複書き込みが
        // あっても各 `src` 要素はそのまま upstream を受け取る
        // （マスク不要）。
        Op::Scatter {
            input,
            dim,
            index,
            src,
            reduce,
        } => {
            let src_shape = nodes[src.0].shape.clone();
            let input_shape = nodes[input.0].shape.clone();

            let mut d_src_upstream = upstream.clone();
            for (axis, &in_s) in input_shape.iter().enumerate() {
                if axis == dim {
                    continue;
                }
                let idx_s = src_shape[axis];
                if idx_s < in_s {
                    d_src_upstream = d_src_upstream
                        .narrow(axis, 0, idx_s)
                        .map_err(AutodiffError::Shape)?;
                }
            }
            let raw_d_src = gather_with_fallback(ops, &d_src_upstream, dim, &index, &src_shape)?;

            let (d_input, d_src) = match reduce {
                ScatterReduce::Add => (upstream.clone(), raw_d_src),
                ScatterReduce::Overwrite => {
                    let zeros_src = Tensor::zeros(&src_shape).map_err(AutodiffError::Shape)?;
                    let d_input = scatter_with_fallback(
                        ops,
                        upstream,
                        dim,
                        &index,
                        &zeros_src,
                        ScatterReduce::Overwrite,
                        &input_shape,
                    )?;
                    let mask = scatter_overwrite_last_writer_mask(&index, dim, &input_shape);
                    let d_src = elementwise_mul_mask(&raw_d_src, &mask, |m| m != 0.0);
                    (d_input, d_src)
                }
                // `ScatterReduce` は `#[non_exhaustive]`（`tensor-core`
                // 側で将来 variant を追加しうる。`Activation` と同じ
                // 非破壊拡張方針）。未知 variant は fail-closed に
                // エラーを返す（`backend-cpu::ops::linear_forward_
                // device` の `Activation` 未知 variant 分岐と同型）。
                _ => {
                    return Err(AutodiffError::InvalidArgument(format!(
                        "Op::Scatter の VJP: 未知の ScatterReduce variant {reduce:?}"
                    )));
                }
            };
            vec![(input, d_input), (src, d_src)]
        }
        // `Var::sort`／`Var::topk`（`Var::argsort` はノードを記録しない
        // ため到達しない。イシュー #1733）。forward が
        // `values = gather(input, dim, index)` と数学的に同一
        // （並べ替え・選択は算術演算を含まない要素の並べ替えのみ）で
        // あるため、VJP は `Op::Gather` と同じ「scatter_add で零テンソル
        // へ upstream を index の位置へ加算する」式を使う。同一出力
        // 位置（`index` 内）に重複する添字は存在しない（sort／topk は
        // 各出力位置ごとに `input` の異なる要素を選ぶ全単射・部分
        // 単射のため）ので `ScatterReduce::Add` と `Overwrite` は同値
        // だが、`Op::Gather` VJP と実装を揃え `Add`（`f64` 経路も
        // `0.0 + x` で bit 一致）を使う。
        Op::Sort { input, dim, index } | Op::Topk { input, dim, index } => {
            let input_shape = nodes[input.0].shape.clone();
            let zeros = Tensor::zeros(&input_shape).map_err(AutodiffError::Shape)?;
            let d_input = scatter_with_fallback(
                ops,
                &zeros,
                dim,
                &index,
                upstream,
                ScatterReduce::Add,
                &input_shape,
            )?;
            vec![(input, d_input)]
        }
        // `Var::interpolate`（イシュー #1757）。「各出力要素が単一の
        // 入力要素を参照する演算（gather／sort／topk と同型）の VJP
        // は scatter_add」の原則を、空間軸のみを対象に適用する:
        // `input`／`upstream` を `[outer, sp]`（`outer` = 先頭の残り
        // 軸の積・`sp` = 空間軸の積）へ reshape してから
        // `nearest_src_index_map`（forward と同じ添字式を共有する
        // 単一情報源）で構築した index を使い `scatter_with_fallback`
        // （`dim=1`・`ScatterReduce::Add`）で入力空間位置へ加算し、
        // 最後に `input_shape` へ reshape し直す。
        Op::Interpolate { input, size, mode } => {
            let input_shape = nodes[input.0].shape.clone();
            let rank = input_shape.len();
            let spatial_start = rank - size.len();
            let out_shape = upstream.shape().to_vec();
            // `outer`（先頭の残り軸——batch 等——の積）も `sp_in_numel`／
            // `sp_out_numel`（下記）と同じ理由で無検査 `.iter().product()`
            // のままにしない（呼び出し元が構築時点で検査済みという
            // 不変条件に暗黙に依存しない・Cursor Bugbot 指摘と同型の
            // 懸念への一貫した対処。PR #1834 レビュー）。
            let outer: usize = input_shape[..spatial_start]
                .iter()
                .try_fold(1usize, |acc, &d| acc.checked_mul(d))
                .ok_or(AutodiffError::Shape(ShapeError::ElementCountOverflow))?;

            // `outer == 0`（先頭の残り軸——batch 等——が空）の場合、
            // `d_input` の全要素数は `outer * sp_in_numel == 0` で
            // 自明にゼロ勾配となる。`interpolate_out_shape` の契約
            // （forward はこの場合も出力の先頭軸が 0 になる空 shape
            // で成功する。`ops_shape.rs::interpolate_out_shape` doc
            // 参照）により、`outer==0` のときは forward が `size` に
            // **単一の**巨大な値を指定しても success する（出力全体の
            // 要素数積が先頭の `0` に短絡し `checked_numel_for::<f32>`
            // のバイトサイズ検査が実際のバッファ非確保ゆえ問題にならない
            // ため。`outer != 0` の場合はこの短絡が起きず、単一軸でも
            // `checked_numel_for::<f32>` のバイトサイズ上限で拒否され
            // うる——`ops_shape.rs::
            // interpolate_out_shape_rejects_byte_size_overflow_without_
            // numel_overflow` 参照）。ただし `size` が**複数軸**かつ
            // 非ゼロ次元同士の部分積が `usize` を overflow する組合せ
            // （例: `size=[usize::MAX, 2]`）は、`outer==0` でも
            // `interpolate_out_shape` 自身が単一情報源として事前拒否
            // する（`ops_shape.rs::interpolate_out_shape` の `nonzero_
            // dims` 検査。イシュー #1834 Cursor Bugbot 指摘・是正）ため
            // 本関数へは到達しない。空間軸の積（`sp_in_numel`／
            // `sp_out_numel`）自体の計算を `outer == 0` 判定より前に
            // 行うと、`out_shape` の空間軸に `usize::MAX` 級の値が
            // 単一軸で入りうる（`outer==0` のときは forward が単一軸の
            // 巨大 `size` を受理するため）ケースで積計算自体が
            // overflow する（debug は panic・release は wrap。本番経路
            // panic 禁止規約 `.claude/rules/coding-rust.md`。Cursor
            // Bugbot 指摘）。
            // `nearest_src_index_map` も `outer` に依存せず空間軸の
            // 全出力位置（`sp_out_numel` 個）分の index 行を無条件に
            // 確保するため、`outer==0` のまま呼ぶと `size=[usize::MAX]`
            // 等で capacity overflow で panic しうる（イシュー #1834
            // codex-review P1 是正）。よって `sp_in_numel`／
            // `sp_out_numel` の計算・index 構築より前に `outer` のみで
            // early return し、この不要な大量確保・走査・積 overflow
            // を通常の大きな size で同時に避ける（多層防御。`interpolate_
            // out_shape` 側の拒否をすり抜ける経路が万一あっても本関数
            // 自身が独立に安全側へ倒れる）。
            if outer == 0 {
                let d_input = Tensor::zeros(&input_shape).map_err(AutodiffError::Shape)?;
                return Ok(vec![(input, d_input)]);
            }

            // `outer != 0` に絞ったこの時点でも、`input_shape`／
            // `out_shape` の空間軸だけを取り出した部分積は無検査
            // `.iter().product()` のままでは overflow しうる
            // （Cursor Bugbot 指摘・PR #1834 レビュー）。`outer` を
            // 含む全軸の積（テンソル全体の要素数）は生成時点
            // （`Tensor::broadcast_to`／`Var::broadcast_to` 等）で
            // `checked_numel` 相当により overflow しないことを検査
            // 済みという不変条件に暗黙に依存せず、ここでも自前で
            // `checked_mul` を用い明示的に検査する（`.claude/rules/
            // security.md` A03 の fail-closed 方針・本ファイル既存の
            // `checked_numel_for` 系ヘルパーと同型）。
            let sp_in_numel: usize = input_shape[spatial_start..]
                .iter()
                .try_fold(1usize, |acc, &d| acc.checked_mul(d))
                .ok_or(AutodiffError::Shape(ShapeError::ElementCountOverflow))?;
            let sp_out_numel: usize = out_shape[spatial_start..]
                .iter()
                .try_fold(1usize, |acc, &d| acc.checked_mul(d))
                .ok_or(AutodiffError::Shape(ShapeError::ElementCountOverflow))?;

            match mode {
                fandhe_ai_tensor_core::InterpolateMode::Nearest => {
                    // scatter_add の index dtype は `i32`（`Tensor<i32>`）
                    // のため、空間軸の要素数が `i32::MAX` を超える場合は
                    // 添字が表現できない（REQ-8 の縦深防御。`Var::
                    // gather`／`scatter` の `IndexRangeOverflow` 契約と
                    // 同種の事前検査）。
                    if sp_in_numel > i32::MAX as usize {
                        return Err(AutodiffError::InvalidArgument(format!(
                            "Op::Interpolate の VJP: 空間軸要素数 {sp_in_numel} が \
                             i32::MAX を超え scatter_add の index dtype (i32) に \
                             収まらない"
                        )));
                    }
                    let index =
                        nearest_src_index_map(&input_shape, &out_shape, spatial_start, outer);
                    let upstream2d = upstream
                        .contiguous()
                        .reshape(&[outer, sp_out_numel])
                        .map_err(AutodiffError::Shape)?;
                    let zeros2d =
                        Tensor::zeros(&[outer, sp_in_numel]).map_err(AutodiffError::Shape)?;
                    let d_input2d = scatter_with_fallback(
                        ops,
                        &zeros2d,
                        1,
                        &index,
                        &upstream2d,
                        ScatterReduce::Add,
                        &[outer, sp_in_numel],
                    )?;
                    let d_input = d_input2d
                        .reshape(&input_shape)
                        .map_err(AutodiffError::Shape)?;
                    vec![(input, d_input)]
                }
                // `Var::interpolate`（`Bilinear`。イシュー #1762）。
                // 各出力位置は 4 個の入力近傍への重み付き寄与を持つため
                // （`Nearest` の 1 対 1 対応とは異なる）、scatter_add の
                // index／src を「出力位置 × 4 コーナー」の平坦化 2 階
                // テンソル（shape `[outer, sp_out * 4]`）として構築する。
                // コーナー順は forward と共有する単一情報源
                // （`bilinear_src_index_and_weight_map`。`tensor-core::
                // interpolate::bilinear_src_coord` を呼ぶ）が固定する
                // `(y0,x0),(y0,x1),(y1,x0),(y1,x1)` で、scatter_add は
                // 出力位置 major・コーナー minor の逐次和として決定的に
                // 集約する（`ScatterReduce::Add` の決定的集約契約。
                // `.claude/rules/coding-rust.md`）。重複コーナー
                // （境界・`in_size==1`）はそのまま複数回加算される
                // （forward の重みの和が 1 のまま保たれるのと対）。
                fandhe_ai_tensor_core::InterpolateMode::Bilinear { align_corners } => {
                    if sp_in_numel > i32::MAX as usize {
                        return Err(AutodiffError::InvalidArgument(format!(
                            "Op::Interpolate の VJP: 空間軸要素数 {sp_in_numel} が \
                             i32::MAX を超え scatter_add の index dtype (i32) に \
                             収まらない"
                        )));
                    }
                    // `sp_out_numel * 4`（各出力位置あたり 4 コーナー）・
                    // `outer * (sp_out_numel * 4)`（`index`／`src` の
                    // 実際の確保長）は `nearest_src_index_map` と同じ
                    // 理由で `checked_mul` により独立に検査する（巨大な
                    // `size` に対する capacity overflow panic 防止。
                    // `.claude/rules/coding-rust.md`）。
                    let sp_out_x4 = sp_out_numel
                        .checked_mul(4)
                        .ok_or(AutodiffError::Shape(ShapeError::ElementCountOverflow))?;
                    outer
                        .checked_mul(sp_out_x4)
                        .ok_or(AutodiffError::Shape(ShapeError::ElementCountOverflow))?;

                    let (index_row, weight_row) = bilinear_src_index_and_weight_map(
                        &input_shape,
                        &out_shape,
                        spatial_start,
                        align_corners,
                    );

                    let upstream2d = upstream
                        .contiguous()
                        .reshape(&[outer, sp_out_numel])
                        .map_err(AutodiffError::Shape)?;
                    let upstream_data = eval::dense_vec(&upstream2d);

                    let mut index_data = Vec::with_capacity(outer * sp_out_x4);
                    let mut src_data = Vec::with_capacity(outer * sp_out_x4);
                    for o in 0..outer {
                        index_data.extend_from_slice(&index_row);
                        let row_base = o * sp_out_numel;
                        for p in 0..sp_out_numel {
                            let u = upstream_data[row_base + p];
                            for k in 0..4 {
                                src_data.push(u * weight_row[p * 4 + k]);
                            }
                        }
                    }
                    let index = eval::build_index_tensor(index_data, &[outer, sp_out_x4]);
                    let src2d =
                        Tensor::new(src_data, &[outer, sp_out_x4]).map_err(AutodiffError::Shape)?;
                    let zeros2d =
                        Tensor::zeros(&[outer, sp_in_numel]).map_err(AutodiffError::Shape)?;
                    let d_input2d = scatter_with_fallback(
                        ops,
                        &zeros2d,
                        1,
                        &index,
                        &src2d,
                        ScatterReduce::Add,
                        &[outer, sp_in_numel],
                    )?;
                    let d_input = d_input2d
                        .reshape(&input_shape)
                        .map_err(AutodiffError::Shape)?;
                    vec![(input, d_input)]
                }
                // `InterpolateMode` は `#[non_exhaustive]`（`tensor-core`
                // 側で将来 variant を追加しうる。`Op::Scatter` VJP の
                // 未知 `ScatterReduce` variant 分岐と同型の fail-closed
                // 処理）。
                _ => {
                    return Err(AutodiffError::InvalidArgument(format!(
                        "Op::Interpolate の VJP: 未知の InterpolateMode variant {mode:?}"
                    )));
                }
            }
        }
        // `Var::contiguous` が記録するノード（イシュー #1620。
        // `crate::einsum` の permute 後 reshape 前の明示実体化）。
        // メモリレイアウトのみが変わり値は変わらないため、VJP は
        // upstream をそのまま入力へ渡す恒等パススルー（`tape::
        // Op::Contiguous` doc 参照）。
        Op::Contiguous { input } => {
            vec![(input, upstream.clone())]
        }
        // `Var::one_hot`（**非微分演算**。イシュー #1755）。クラス id
        // から 0/1 行列を作る操作は入力に対し微分不能なため、
        // `upstream` の値を無視して常に入力 shape の**明示ゼロ**
        // テンソルを返す（`Op` doc 参照: 寄与なし〈`vec![]`〉ではなく
        // 「ゼロ勾配が流れる」ことを `Gradients::get` で観測可能にする
        // ための意図的な設計）。
        Op::OneHot { input, .. } => {
            let input_shape = nodes[input.0].shape.clone();
            let d_input = Tensor::zeros(&input_shape).map_err(AutodiffError::Shape)?;
            vec![(input, d_input)]
        }
    };
    Ok(contributions)
}

/// [`Op::Where`] の VJP 本体（イシュー #1637）。`cond` は `out_shape`
/// ちょうど（forward 時点で broadcast 済み）の f32 マスク。`Op::Mul`
/// と同じ「まず out_shape で計算してから `reduce_to_shape` で入力
/// shape へ縮約する」契約に従う: `da = reduce(mask_keep(g, cond, c !=
/// 0.0), a_shape)`・`db = reduce(mask_keep(g, cond, c == 0.0),
/// b_shape)`。
fn where_vjp(
    cond: &Tensor<f32>,
    upstream: &Tensor<f32>,
    a_shape: &[usize],
    b_shape: &[usize],
) -> (Tensor<f32>, Tensor<f32>) {
    let da = elementwise_mul_mask(upstream, cond, |c| c != 0.0);
    let db = elementwise_mul_mask(upstream, cond, |c| c == 0.0);
    (reduce_to_shape(&da, a_shape), reduce_to_shape(&db, b_shape))
}

/// [`Op::MaskedFill`] の VJP 本体（イシュー #1637）。fill 位置
/// （`mask != 0.0`）の勾配は 0。`mask` は `input` と同 shape のため
/// broadcast 縮約は不要（`Op::Relu` の VJP と同型）。
fn masked_fill_vjp(mask: &Tensor<f32>, upstream: &Tensor<f32>) -> Tensor<f32> {
    elementwise_mul_mask(upstream, mask, |m| m == 0.0)
}

/// [`Op::Concat`] の forward（`Var::cat`）と [`Op::Narrow`] の VJP
/// （直上）が共用する連結ヘルパー（イシュー #1598）。`ops.concat` →
/// `Unsupported` のときのみ `eval::concat` へフォールバックする
/// （`Op::Softmax` の「バックエンド実装 → フォールバック」二段構成と
/// 同型。判定迂回経路を作らない。`.claude/rules/security.md` A08）。
///
/// バックエンド実装（`Unsupported` 以外）が返した出力 shape を
/// `out_shape` と照合し、不一致は
/// `AutodiffError::Backend(BackendError::ShapeMismatch(..))` を返す
/// （実装バグの黙認防止）。
pub(crate) fn concat_with_fallback(
    ops: &dyn BackendOps,
    inputs: &[&Tensor<f32>],
    dim: usize,
    out_shape: &[usize],
) -> Result<Tensor<f32>, AutodiffError> {
    match ops.concat(inputs, dim) {
        Ok(v) => {
            if v.shape() != out_shape {
                return Err(AutodiffError::Backend(BackendError::ShapeMismatch(
                    ShapeError::ShapeMismatch {
                        lhs: v.shape().to_vec(),
                        rhs: out_shape.to_vec(),
                    },
                )));
            }
            Ok(v)
        }
        Err(BackendError::Unsupported(_)) => Ok(eval::concat(inputs, dim, out_shape)),
        Err(other) => Err(AutodiffError::Backend(other)),
    }
}

/// [`Op::Where`] の forward（`Var::where_cond`）が使う
/// 「バックエンド実装 → フォールバック」ヘルパー（イシュー #1637）。
/// [`concat_with_fallback`] と同型: `ops.where_cond` →
/// `Unsupported` のときのみ `eval::where_cond` へフォールバックし、
/// それ以外のエラーは伝播する（判定迂回経路を作らない）。バックエンド
/// 実装が返した出力 shape を `out_shape` と照合し、不一致は
/// `AutodiffError::Backend(BackendError::ShapeMismatch(..))` を返す。
pub(crate) fn where_cond_with_fallback(
    ops: &dyn BackendOps,
    cond: &Tensor<f32>,
    a: &Tensor<f32>,
    b: &Tensor<f32>,
    out_shape: &[usize],
) -> Result<Tensor<f32>, AutodiffError> {
    match ops.where_cond(cond, a, b) {
        Ok(v) => {
            if v.shape() != out_shape {
                return Err(AutodiffError::Backend(BackendError::ShapeMismatch(
                    ShapeError::ShapeMismatch {
                        lhs: v.shape().to_vec(),
                        rhs: out_shape.to_vec(),
                    },
                )));
            }
            Ok(v)
        }
        Err(BackendError::Unsupported(_)) => Ok(eval::where_cond(cond, a, b, out_shape)),
        Err(other) => Err(AutodiffError::Backend(other)),
    }
}

/// [`Op::ScalarUnary`] の forward（`Var::scalar_unary`）が使う
/// 「バックエンド実装 → フォールバック」ヘルパー（イシュー #1634）。
/// [`where_cond_with_fallback`] と同型: `ops.scalar_unary` →
/// `Unsupported` のときのみ `eval::scalar::unary` へフォールバックし、
/// それ以外のエラーは伝播する（判定迂回経路を作らない）。バックエンド
/// 実装が返した出力 shape を `a` の shape と照合し、不一致は
/// `AutodiffError::Backend(BackendError::ShapeMismatch(..))` を返す
/// （単項のため shape は不変契約）。
///
/// 呼び出し元 `Var::scalar_unary` は #1710 で公開メソッド（`Var::sqrt`
/// 等）から、#1711 で `Var::log` 等からも到達可能になったため
/// `#[allow(dead_code)]` は撤去済み（`tape::Op::ScalarUnary` doc と
/// 同じ経緯）。
pub(crate) fn scalar_unary_with_fallback(
    ops: &dyn BackendOps,
    op: ScalarUnaryOp,
    a: &Tensor<f32>,
) -> Result<Tensor<f32>, AutodiffError> {
    match ops.scalar_unary(op, a) {
        Ok(v) => {
            if v.shape() != a.shape() {
                return Err(AutodiffError::Backend(BackendError::ShapeMismatch(
                    ShapeError::ShapeMismatch {
                        lhs: v.shape().to_vec(),
                        rhs: a.shape().to_vec(),
                    },
                )));
            }
            Ok(v)
        }
        Err(BackendError::Unsupported(_)) => Ok(eval::scalar::unary(a, op)),
        Err(other) => Err(AutodiffError::Backend(other)),
    }
}

/// [`Op::ScalarBinary`] の forward（`Var::scalar_binary`）が使う
/// 「バックエンド実装 → フォールバック」ヘルパー（イシュー #1634）。
/// [`scalar_unary_with_fallback`] の 2 項版で設計方針は同一。
/// `out_shape`（呼び出し元が `broadcast_shape(a, b)` で事前計算）と
/// 戻り shape を照合する。
///
/// 呼び出し元 `Var::scalar_binary` は #1710 で公開メソッド
/// （`Var::sub`／`div`／`pow`）から到達可能になったため
/// `#[allow(dead_code)]` は撤去済み（[`scalar_unary_with_fallback`]
/// と同じ経緯）。
pub(crate) fn scalar_binary_with_fallback(
    ops: &dyn BackendOps,
    op: ScalarBinaryOp,
    a: &Tensor<f32>,
    b: &Tensor<f32>,
    out_shape: &[usize],
) -> Result<Tensor<f32>, AutodiffError> {
    match ops.scalar_binary(op, a, b) {
        Ok(v) => {
            if v.shape() != out_shape {
                return Err(AutodiffError::Backend(BackendError::ShapeMismatch(
                    ShapeError::ShapeMismatch {
                        lhs: v.shape().to_vec(),
                        rhs: out_shape.to_vec(),
                    },
                )));
            }
            Ok(v)
        }
        Err(BackendError::Unsupported(_)) => Ok(eval::scalar::binary(a, b, op)),
        Err(other) => Err(AutodiffError::Backend(other)),
    }
}

/// [`Op::MaskedFill`] の forward（`Var::masked_fill`）が使う
/// 「バックエンド実装 → フォールバック」ヘルパー（イシュー #1637）。
/// [`where_cond_with_fallback`] と同型。出力 shape は `x` と恒等。
pub(crate) fn masked_fill_with_fallback(
    ops: &dyn BackendOps,
    x: &Tensor<f32>,
    mask: &Tensor<f32>,
    value: f32,
) -> Result<Tensor<f32>, AutodiffError> {
    match ops.masked_fill(x, mask, value) {
        Ok(v) => {
            if v.shape() != x.shape() {
                return Err(AutodiffError::Backend(BackendError::ShapeMismatch(
                    ShapeError::ShapeMismatch {
                        lhs: v.shape().to_vec(),
                        rhs: x.shape().to_vec(),
                    },
                )));
            }
            Ok(v)
        }
        Err(BackendError::Unsupported(_)) => Ok(eval::masked_fill(x, mask, value)),
        Err(other) => Err(AutodiffError::Backend(other)),
    }
}

/// [`Op::Pad`] の forward（`Var::pad`）が使う「バックエンド実装 →
/// フォールバック」ヘルパー（イシュー #1756）。[`masked_fill_with_fallback`]
/// と同型: `ops.pad` → `Unsupported` のときのみ `eval::pad` へ
/// フォールバックし、それ以外のエラーは伝播する（判定迂回経路を
/// 作らない）。バックエンド実装が返した出力 shape を `out_shape` と
/// 照合し、不一致は `AutodiffError::Backend(BackendError::
/// ShapeMismatch(..))` を返す。
pub(crate) fn pad_with_fallback(
    ops: &dyn BackendOps,
    input: &Tensor<f32>,
    pads: &[(usize, usize)],
    value: f32,
    out_shape: &[usize],
) -> Result<Tensor<f32>, AutodiffError> {
    match ops.pad(input, pads, value) {
        Ok(v) => {
            if v.shape() != out_shape {
                return Err(AutodiffError::Backend(BackendError::ShapeMismatch(
                    ShapeError::ShapeMismatch {
                        lhs: v.shape().to_vec(),
                        rhs: out_shape.to_vec(),
                    },
                )));
            }
            Ok(v)
        }
        Err(BackendError::Unsupported(_)) => Ok(eval::pad(input, pads, value, out_shape)),
        Err(other) => Err(AutodiffError::Backend(other)),
    }
}

/// [`Op::Gather`] の forward（`Var::gather`／`Var::index_select` 経由）
/// および [`Op::Gather`] の VJP（`Op::Scatter { reduce: Add }` を
/// 経由せず直接 [`Op::Gather`] を再利用する `d_src` 計算）が使う
/// 「バックエンド実装 → フォールバック」ヘルパー（イシュー #1776）。
/// [`where_cond_with_fallback`] と同型: `ops.gather` →
/// `Unsupported` のときのみ `eval::gather` へフォールバックし、
/// それ以外のエラーは伝播する（判定迂回経路を作らない）。バックエンド
/// 実装が返した出力 shape を `out_shape` と照合し、不一致は
/// `AutodiffError::Backend(BackendError::ShapeMismatch(..))` を返す。
pub(crate) fn gather_with_fallback(
    ops: &dyn BackendOps,
    input: &Tensor<f32>,
    dim: usize,
    index: &Tensor<i32>,
    out_shape: &[usize],
) -> Result<Tensor<f32>, AutodiffError> {
    match ops.gather(input, dim, index) {
        Ok(v) => {
            if v.shape() != out_shape {
                return Err(AutodiffError::Backend(BackendError::ShapeMismatch(
                    ShapeError::ShapeMismatch {
                        lhs: v.shape().to_vec(),
                        rhs: out_shape.to_vec(),
                    },
                )));
            }
            Ok(v)
        }
        Err(BackendError::Unsupported(_)) => Ok(eval::gather(input, dim, index, out_shape)),
        Err(other) => Err(AutodiffError::Backend(other)),
    }
}

/// [`Op::Scatter`] の forward（`Var::scatter`／`scatter_add` 経由）
/// および [`Op::Gather`] の VJP（`d_input` 計算）が使う
/// 「バックエンド実装 → フォールバック」ヘルパー（イシュー #1776）。
/// [`gather_with_fallback`] と同型: `ops.scatter` → `Unsupported` の
/// ときのみ `eval::scatter`（[`ScatterReduce`] の決定的集約契約に
/// 厳密に従う）へフォールバックする。
pub(crate) fn scatter_with_fallback(
    ops: &dyn BackendOps,
    input: &Tensor<f32>,
    dim: usize,
    index: &Tensor<i32>,
    src: &Tensor<f32>,
    reduce: ScatterReduce,
    out_shape: &[usize],
) -> Result<Tensor<f32>, AutodiffError> {
    match ops.scatter(input, dim, index, src, reduce) {
        Ok(v) => {
            if v.shape() != out_shape {
                return Err(AutodiffError::Backend(BackendError::ShapeMismatch(
                    ShapeError::ShapeMismatch {
                        lhs: v.shape().to_vec(),
                        rhs: out_shape.to_vec(),
                    },
                )));
            }
            Ok(v)
        }
        Err(BackendError::Unsupported(_)) => Ok(eval::scatter(input, dim, index, src, reduce)),
        Err(other) => Err(AutodiffError::Backend(other)),
    }
}

/// [`Op::Embedding`] の VJP（`padding_idx` 行のゼロ上書き）が使う
/// ヘルパー（イシュー #1604）。`Tensor<f32>` に可変スライス API が
/// ないため、`host_slice()`（strided／owned のいずれでも密な `Vec`
/// を返す）で値を取り出し、行 `row`（`[row*cols, (row+1)*cols)` の
/// 半開区間。`row < t.shape()[0]` は呼び出し元（`vjp` の
/// `Op::Embedding` 分岐）が forward 時点で検証済みの `padding_idx`
/// をそのまま渡す契約——`Var::embedding` の入口検査を `.claude/
/// rules/coding-rust.md` の「境界検査を省略しない」方針に従い
/// 再度ここで信頼する）を 0.0 で上書きしてから新しい `Tensor` を
/// 構築して返す。
fn with_row_zeroed(t: &Tensor<f32>, row: usize, cols: usize) -> Result<Tensor<f32>, AutodiffError> {
    let mut data = t.host_slice().into_owned();
    let start = row * cols;
    let end = start + cols;
    for v in &mut data[start..end] {
        *v = 0.0;
    }
    Tensor::new(data, t.shape()).map_err(AutodiffError::Shape)
}

/// [`Op::Sort`] の forward（`Var::sort`／`argsort` 経由）が使う
/// 「バックエンド実装 → フォールバック」ヘルパー（イシュー #1733）。
/// [`gather_with_fallback`] と同型: `ops.sort` → `Unsupported` の
/// ときのみ `eval::sort`（[`fandhe_ai_tensor_core::BackendOps::sort`]
/// の順序契約に厳密に従う）へフォールバックし、それ以外のエラーは
/// 伝播する（判定迂回経路を作らない）。バックエンド実装が返した
/// `values`／`index` の shape を `out_shape` と照合し、不一致は
/// `AutodiffError::Backend(BackendError::ShapeMismatch(..))` を返す。
pub(crate) fn sort_with_fallback(
    ops: &dyn BackendOps,
    input: &Tensor<f32>,
    dim: usize,
    descending: bool,
    out_shape: &[usize],
) -> Result<(Tensor<f32>, Tensor<i32>), AutodiffError> {
    match ops.sort(input, dim, descending) {
        Ok((values, index)) => {
            if values.shape() != out_shape || index.shape() != out_shape {
                return Err(AutodiffError::Backend(BackendError::ShapeMismatch(
                    ShapeError::ShapeMismatch {
                        lhs: values.shape().to_vec(),
                        rhs: out_shape.to_vec(),
                    },
                )));
            }
            Ok((values, index))
        }
        Err(BackendError::Unsupported(_)) => Ok(eval::sort(input, dim, descending)?),
        Err(other) => Err(AutodiffError::Backend(other)),
    }
}

/// [`Op::Topk`] の forward（`Var::topk` 経由）が使う「バックエンド
/// 実装 → フォールバック」ヘルパー（イシュー #1733）。
/// [`sort_with_fallback`] と同型: `ops.topk` → `Unsupported` の
/// ときのみ `eval::topk` へフォールバックする。
pub(crate) fn topk_with_fallback(
    ops: &dyn BackendOps,
    input: &Tensor<f32>,
    dim: usize,
    k: usize,
    largest: bool,
    out_shape: &[usize],
) -> Result<(Tensor<f32>, Tensor<i32>), AutodiffError> {
    match ops.topk(input, dim, k, largest) {
        Ok((values, index)) => {
            if values.shape() != out_shape || index.shape() != out_shape {
                return Err(AutodiffError::Backend(BackendError::ShapeMismatch(
                    ShapeError::ShapeMismatch {
                        lhs: values.shape().to_vec(),
                        rhs: out_shape.to_vec(),
                    },
                )));
            }
            Ok((values, index))
        }
        Err(BackendError::Unsupported(_)) => Ok(eval::topk(input, dim, k, largest, out_shape)?),
        Err(other) => Err(AutodiffError::Backend(other)),
    }
}

/// 縮約対象の要素数が 0 か（`min`／`argmax`／`argmin` は単位元を
/// 持たないためエラーとする対象。`backend-cpu::reduction::max`
/// の空縮約判定と同じ「`axis_len == 0` かつ `total_out > 0`」規則を
/// ホストフォールバック側でも再現する。イシュー #1720）。
fn is_empty_reduction(shape: &[usize], dim: Option<usize>) -> bool {
    match dim {
        None => shape.iter().product::<usize>() == 0,
        Some(axis) => {
            let axis_len = shape.get(axis).copied().unwrap_or(0);
            let total_out: usize = shape
                .iter()
                .enumerate()
                .filter(|&(i, _)| i != axis)
                .map(|(_, &d)| d)
                .product();
            axis_len == 0 && total_out > 0
        }
    }
}

/// [`Op::Min`] の forward（`Var::min` 経由）が使う「バックエンド実装
/// → フォールバック」ヘルパー（イシュー #1720）。[`sort_with_fallback`]
/// と同型: `ops.min` → `Unsupported` のときのみ空縮約を検査してから
/// `eval::min` へフォールバックし、それ以外のエラーは伝播する
/// （判定迂回経路を作らない）。空縮約は
/// `AutodiffError::InvalidArgument` を返す（`min` は単位元を持たない
/// ため。`backend-cpu::reduction::min` の `EmptyReduction` と同じ
/// 方針をホスト経路でも守る）。
pub(crate) fn min_with_fallback(
    ops: &dyn BackendOps,
    input: &Tensor<f32>,
    dim: Option<usize>,
    out_shape: &[usize],
) -> Result<Tensor<f32>, AutodiffError> {
    match ops.min(input, dim) {
        Ok(v) => {
            if v.shape() != out_shape {
                return Err(AutodiffError::Backend(BackendError::ShapeMismatch(
                    ShapeError::ShapeMismatch {
                        lhs: v.shape().to_vec(),
                        rhs: out_shape.to_vec(),
                    },
                )));
            }
            Ok(v)
        }
        Err(BackendError::Unsupported(_)) => {
            if is_empty_reduction(input.shape(), dim) {
                return Err(AutodiffError::InvalidArgument(
                    "min: cannot compute min of an empty reduction".into(),
                ));
            }
            Ok(eval::min(input, dim, out_shape))
        }
        Err(other) => Err(AutodiffError::Backend(other)),
    }
}

/// [`Var::argmax`]／[`Var::argmin`] のどちらを計算するかを区別する
/// （イシュー #1720）。`ops.argmax`／`argmin`・`eval::argmax`／`argmin`
/// のどちらを呼ぶかを [`argext_with_fallback`] が切り替える。
pub(crate) enum ArgExtremum {
    Max,
    Min,
}

/// [`Op::Sort`]／[`Op::Topk`] と異なり `argmax`／`argmin` は非微分
/// 演算（テープにノードを追加しない）のため対応する `Op` variant を
/// 持たない。[`sort_with_fallback`] と同型の「バックエンド実装 →
/// フォールバック」ヘルパー（イシュー #1720）で、`kind` に応じて
/// `ops.argmax`／`argmin` → `Unsupported` のときのみ空縮約を検査して
/// から `eval::argmax`／`argmin` へフォールバックする。
pub(crate) fn argext_with_fallback(
    ops: &dyn BackendOps,
    input: &Tensor<f32>,
    dim: Option<usize>,
    out_shape: &[usize],
    kind: ArgExtremum,
) -> Result<Tensor<i32>, AutodiffError> {
    let backend_result = match kind {
        ArgExtremum::Max => ops.argmax(input, dim),
        ArgExtremum::Min => ops.argmin(input, dim),
    };
    match backend_result {
        Ok(v) => {
            if v.shape() != out_shape {
                return Err(AutodiffError::Backend(BackendError::ShapeMismatch(
                    ShapeError::ShapeMismatch {
                        lhs: v.shape().to_vec(),
                        rhs: out_shape.to_vec(),
                    },
                )));
            }
            Ok(v)
        }
        Err(BackendError::Unsupported(_)) => {
            if is_empty_reduction(input.shape(), dim) {
                let op_name = match kind {
                    ArgExtremum::Max => "argmax",
                    ArgExtremum::Min => "argmin",
                };
                return Err(AutodiffError::InvalidArgument(format!(
                    "{op_name}: cannot compute {op_name} of an empty reduction"
                )));
            }
            let result = match kind {
                ArgExtremum::Max => eval::argmax(input, dim, out_shape),
                ArgExtremum::Min => eval::argmin(input, dim, out_shape),
            };
            Ok(result?)
        }
        Err(other) => Err(AutodiffError::Backend(other)),
    }
}

/// [`Op::OneHot`] の forward（`Var::one_hot` 経由）が使う「バックエンド
/// 実装 → フォールバック」ヘルパー（イシュー #1755）。[`gather_with_
/// fallback`] と同型: `ops.one_hot` → `Unsupported` のときのみ
/// `eval::one_hot` へフォールバックし、それ以外のエラーは伝播する
/// （判定迂回経路を作らない。`.claude/rules/security.md` A08）。
pub(crate) fn one_hot_with_fallback(
    ops: &dyn BackendOps,
    index: &Tensor<i32>,
    num_classes: usize,
    out_shape: &[usize],
) -> Result<Tensor<f32>, AutodiffError> {
    match ops.one_hot(index, num_classes) {
        Ok(v) => {
            if v.shape() != out_shape {
                return Err(AutodiffError::Backend(BackendError::ShapeMismatch(
                    ShapeError::ShapeMismatch {
                        lhs: v.shape().to_vec(),
                        rhs: out_shape.to_vec(),
                    },
                )));
            }
            Ok(v)
        }
        Err(BackendError::Unsupported(_)) => Ok(eval::one_hot(index, num_classes, out_shape)),
        Err(other) => Err(AutodiffError::Backend(other)),
    }
}

/// [`Var::unique`] が使う「バックエンド実装 → フォールバック」ヘルパー
/// （イシュー #1734）。[`gather_with_fallback`] と同型: `ops.unique` →
/// `Unsupported` のときのみ `eval::unique` へフォールバックし、それ
/// 以外のエラーは伝播する（判定迂回経路を作らない）。バックエンドが
/// 返した出力に [`BackendOps::unique`] doc の出力不変条件（rank 1・
/// `len <= numel`・totalOrder で非減少・隣接に `==` な要素がない）を
/// 事後検査し、違反は shape 自体の不正（rank／len）は
/// `AutodiffError::Backend(BackendError::ShapeMismatch(..))`、順序の
/// 不正（totalOrder 非減少・隣接重複なし）は
/// `AutodiffError::InvalidArgument(..)` として区別して拒否する
/// （両者は異なる契約違反であり同一 variant では誤解を招くため。
/// review 指摘）。3 バックエンド実装が独立に契約を守っているかを
/// 呼び出し元でも検証する二重検査方針（`.claude/rules/security.md`
/// A08）。
pub(crate) fn unique_with_fallback(
    ops: &dyn BackendOps,
    x: &Tensor<f32>,
) -> Result<Tensor<f32>, AutodiffError> {
    // `x.numel()`（内部で無検査の `.iter().product()` を使い
    // `overflow-checks` 有効ビルドで panic しうる）・`ops.unique`
    // （バックエンド実装が内部で `numel()`/`contiguous()` を呼ぶ）を
    // 呼ぶ前に要素数積のオーバーフローを検査する（PR #1828
    // codex-review P1 是正: `transpose` 済みの非 contiguous view
    // （例: `[0, 2, usize::MAX]` → `transpose(0, 2)` → `[usize::MAX,
    // 2, 0]`）は `Tensor::new`/`transpose` 単体では要素数積を
    // 再検査しないため到達しうる。`Var::broadcast_to` と同じ自前
    // `checked_mul` 実装。`tensor-core` は `checked_numel` を非公開に
    // している）。
    if x.shape()
        .iter()
        .try_fold(1usize, |acc, &d| acc.checked_mul(d))
        .is_none()
    {
        return Err(AutodiffError::Shape(ShapeError::ElementCountOverflow));
    }
    let numel = x.numel();
    let v = match ops.unique(x) {
        Ok(v) => v,
        Err(BackendError::Unsupported(_)) => eval::unique(x),
        Err(other) => return Err(AutodiffError::Backend(other)),
    };
    validate_unique_output(&v, numel)?;
    Ok(v)
}

/// [`unique_with_fallback`] の出力不変条件検査（[`BackendOps::unique`]
/// doc の契約を正とする）。rank 1・`len <= numel` に加え、隣接ペアが
/// totalOrder で非減少（`Ordering::Greater` でない）かつ `==` でない
/// （NaN は `total_cmp` で `Equal` と判定されても IEEE `==` では
/// `false` になりうるため両者を併用する。「厳密増加」ではなく
/// 「非減少 ＋ 隣接 `==` なし」が正しい述語である点に注意——同一 bit の
/// NaN が隣接して現れうる）ことを検査する。
fn validate_unique_output(v: &Tensor<f32>, numel: usize) -> Result<(), AutodiffError> {
    // shape 違反（rank != 1 または len > numel）は真に shape の契約
    // 違反のため `ShapeMismatch` のまま報告する。一方 totalOrder 順序
    // 違反（rank・len 自体は正しいのに非減少でない／隣接 `==` が
    // 混入した）は shape 不一致ではないため `ShapeMismatch` を流用
    // すると誤解を招く（review 指摘）。`AutodiffError::InvalidArgument`
    // （`error.rs` doc: 既存 `ShapeError` variant に意味的に適合しない
    // 契約違反の集約先）で区別して報告する。
    if v.shape().len() != 1 || v.shape()[0] > numel {
        return Err(AutodiffError::Backend(BackendError::ShapeMismatch(
            ShapeError::ShapeMismatch {
                lhs: v.shape().to_vec(),
                rhs: vec![numel],
            },
        )));
    }
    let data = dense_vec(v);
    let order_ok = data.windows(2).all(|w| {
        let (a, b) = (w[0], w[1]);
        a.total_cmp(&b) != std::cmp::Ordering::Greater && a != b
    });
    if order_ok {
        Ok(())
    } else {
        Err(AutodiffError::InvalidArgument(format!(
            "BackendOps::unique の出力が totalOrder で非減少・隣接 `==` \
             なしという不変条件に違反している（shape={:?}）",
            v.shape()
        )))
    }
}

/// [`Var::interpolate`] が使う「バックエンド実装 → フォールバック」
/// ヘルパー（イシュー #1757）。[`gather_with_fallback`] と同型:
/// `ops.interpolate` → `Unsupported` のときのみホスト参照実装
/// （`eval::interpolate_nearest`。`mode` の未知 variant——将来の
/// bilinear〈#1762〉等——は `eval` 側に対応する再計算経路がないため
/// `AutodiffError::InvalidArgument` を返す）へフォールバックし、
/// それ以外のエラーは伝播する（判定迂回経路を作らない。
/// `.claude/rules/security.md` A08）。
pub(crate) fn interpolate_with_fallback(
    ops: &dyn BackendOps,
    input: &Tensor<f32>,
    size: &[usize],
    mode: fandhe_ai_tensor_core::InterpolateMode,
    out_shape: &[usize],
) -> Result<Tensor<f32>, AutodiffError> {
    match ops.interpolate(input, size, mode) {
        Ok(v) => {
            if v.shape() != out_shape {
                return Err(AutodiffError::Backend(BackendError::ShapeMismatch(
                    ShapeError::ShapeMismatch {
                        lhs: v.shape().to_vec(),
                        rhs: out_shape.to_vec(),
                    },
                )));
            }
            Ok(v)
        }
        Err(BackendError::Unsupported(_)) => match mode {
            fandhe_ai_tensor_core::InterpolateMode::Nearest => {
                eval::interpolate_nearest(input, size).map_err(AutodiffError::Shape)
            }
            fandhe_ai_tensor_core::InterpolateMode::Bilinear { align_corners } => {
                eval::interpolate_bilinear(input, size, align_corners).map_err(AutodiffError::Shape)
            }
            // `InterpolateMode` は `#[non_exhaustive]`（`tensor-core`
            // 側で将来 variant を追加しうる。`ScatterReduce` の
            // `Op::Scatter` VJP 未知 variant 分岐と同型）。
            _ => Err(AutodiffError::InvalidArgument(format!(
                "Var::interpolate の Unsupported フォールバック: 未知の \
                 InterpolateMode variant {mode:?} に対応するホスト参照実装がない"
            ))),
        },
        Err(other) => Err(AutodiffError::Backend(other)),
    }
}

/// [`Op::Interpolate`]（[`fandhe_ai_tensor_core::InterpolateMode::
/// Nearest`]）の VJP が使う scatter_add index 構築（イシュー #1757）。
/// forward のホスト参照実装（`eval::interpolate_nearest`）と**同じ
/// 添字式**（`eval::nearest_src_coord`。単一情報源）を使い、出力の
/// 各空間位置が対応する入力の空間位置（flat 添字。`sp_in` 軸内）を
/// 求める。
///
/// 戻り値は `input`／`upstream` を `[outer, sp]`（`outer` = 先頭の
/// 残り軸の積・`sp` = 空間軸の積）へ reshape した後の 2 階テンソルに
/// 対する `scatter_with_fallback` の `index` 引数として使える形
/// （shape `[outer, sp_out]`。全 `outer` 行が同一の空間添字パターンを
/// 持つ——`outer` 軸〈batch／channel 等〉は素通しで src/dst の対応が
/// 変わらないため）。
pub(crate) fn nearest_src_index_map(
    in_shape: &[usize],
    out_shape: &[usize],
    spatial_start: usize,
    outer: usize,
) -> Tensor<i32> {
    let sp_in = &in_shape[spatial_start..];
    let sp_out = &out_shape[spatial_start..];
    let sp_out_numel: usize = sp_out.iter().product();
    let sp_in_strides = eval::row_major_strides(sp_in);

    let mut row = vec![0i32; sp_out_numel];
    for (flat, slot) in row.iter_mut().enumerate() {
        let coords = eval::unravel(flat, sp_out);
        let mut pos = 0usize;
        for (axis, &stride) in sp_in_strides.iter().enumerate() {
            let src_c = eval::nearest_src_coord(coords[axis], sp_in[axis], sp_out[axis]);
            pos += src_c * stride;
        }
        // `pos` は `sp_in` 内の flat 添字（`< sp_in.iter().product()`）。
        // `Var::interpolate` の forward 検査（`interpolate_out_shape`
        // 経由の `checked_numel`）により `sp_in` 全体の要素数は
        // `usize` の範囲でオーバーフローしないことが保証されているが、
        // `i32` への切り詰め自体は独立に検証する（REQ-8 の縦深防御。
        // `nearest_src_index_map` 呼び出し元 `vjp` の `Op::Interpolate`
        // 腕が事前に `sp_in_numel <= i32::MAX` を検査する契約——
        // 超過時は `AutodiffError::InvalidArgument` を返し本関数へは
        // 到達しない）。
        *slot = pos as i32;
    }

    let mut data = Vec::with_capacity(outer * sp_out_numel);
    for _ in 0..outer {
        data.extend_from_slice(&row);
    }
    eval::build_index_tensor(data, &[outer, sp_out_numel])
}

/// [`Op::Interpolate`]（[`fandhe_ai_tensor_core::InterpolateMode::
/// Bilinear`]）の VJP が使う scatter_add index／重み構築（イシュー
/// #1762）。forward のホスト参照実装（`eval::interpolate_bilinear`）と
/// **同じ座標式**（`fandhe_ai_tensor_core::interpolate::
/// bilinear_src_coord`。単一情報源）を使い、出力の各空間位置に対応する
/// 4 個の入力近傍（flat 添字。`sp_in` 軸内）とその補間重みを求める。
///
/// 戻り値は「出力位置 1 個あたり 4 コーナー」を平坦化した行
/// （長さ `sp_out * 4`。コーナー順は `(y0,x0),(y0,x1),(y1,x0),(y1,x1)`
/// 固定——`nearest_src_index_map` と同様、全 `outer` 行が同一パターン
/// を持つため呼び出し元がこの行を複製して 2 階テンソルを構築する）:
/// - `.0`（`index_row: Vec<i32>`）: `sp_in` 内の flat 添字
/// - `.1`（`weight_row: Vec<f32>`）: 対応する補間重み（`upstream` との
///   乗算は呼び出し元が行う——本関数は `outer` に依存しないため）
///
/// `in_shape`／`out_shape` の空間軸は `spatial_start..` の**ちょうど 2
/// 軸**（`(H, W)`。呼び出し元 `vjp` の `Op::Interpolate` 腕が
/// `interpolate_out_shape_for_mode` 経由で事前保証済み——`Var::
/// interpolate` が forward 時点で拒否するため `Bilinear` の `Op` は
/// 必ず 2 空間軸を持つ）。
pub(crate) fn bilinear_src_index_and_weight_map(
    in_shape: &[usize],
    out_shape: &[usize],
    spatial_start: usize,
    align_corners: bool,
) -> (Vec<i32>, Vec<f32>) {
    debug_assert_eq!(
        out_shape.len() - spatial_start,
        2,
        "bilinear_src_index_and_weight_map: caller must guarantee exactly 2 spatial axes"
    );
    let sp_in = &in_shape[spatial_start..];
    let sp_out = &out_shape[spatial_start..];
    let sp_out_numel: usize = sp_out.iter().product();
    let sp_in_strides = eval::row_major_strides(sp_in);
    let stride_h = sp_in_strides[0];
    let stride_w = sp_in_strides[1];
    let (in_h, in_w) = (sp_in[0], sp_in[1]);
    let (out_h, out_w) = (sp_out[0], sp_out[1]);
    let scale_h = fandhe_ai_tensor_core::bilinear_scale(in_h, out_h, align_corners);
    let scale_w = fandhe_ai_tensor_core::bilinear_scale(in_w, out_w, align_corners);

    let mut index_row = vec![0i32; sp_out_numel * 4];
    let mut weight_row = vec![0f32; sp_out_numel * 4];
    for p in 0..sp_out_numel {
        // `sp_out` はちょうど 2 軸のため flat 添字は `(y, x)` へ直接
        // 分解できる（`eval::unravel` の 2 軸特殊化——ここでは
        // `p / out_w`／`p % out_w` で十分）。
        let y = p / out_w;
        let x = p % out_w;
        let cy = fandhe_ai_tensor_core::bilinear_src_coord(y, in_h, scale_h, align_corners);
        let cx = fandhe_ai_tensor_core::bilinear_src_coord(x, in_w, scale_w, align_corners);
        let l0x = 1.0 - cx.lambda1;
        let l0y = 1.0 - cy.lambda1;
        // コーナー順固定: (y0,x0),(y0,x1),(y1,x0),(y1,x1)（forward の
        // 添字読み出し順・`bilinear_blend` の引数順と一致させる）。
        let corners = [
            (cy.i0, cx.i0, l0y * l0x),
            (cy.i0, cx.i1, l0y * cx.lambda1),
            (cy.i1, cx.i0, cy.lambda1 * l0x),
            (cy.i1, cx.i1, cy.lambda1 * cx.lambda1),
        ];
        for (k, &(iy, ix, w)) in corners.iter().enumerate() {
            let pos = iy * stride_h + ix * stride_w;
            // `pos` は `sp_in` 内の flat 添字。`i32` への切り詰めは
            // 呼び出し元（`vjp` の `Op::Interpolate` Bilinear 分岐）が
            // 事前に `sp_in_numel <= i32::MAX` を検査済みのため安全
            // （`nearest_src_index_map` の同種コメントと同じ契約）。
            index_row[p * 4 + k] = pos as i32;
            weight_row[p * 4 + k] = w;
        }
    }
    (index_row, weight_row)
}

/// [`Op::Scatter`]（`reduce = Overwrite`）の VJP 補助（イシュー
/// #1776・codex-review 指摘）。forward の決定的集約契約
/// （`ScatterReduce::Overwrite` doc: `index`／`src` を行優先で走査し、
/// 同一出力位置への複数回書き込みは「最後に処理された値」だけが
/// 残る）に従い、`d_src` は各出力位置の最後の書き手にのみ流し、
/// 上書きされて消えた重複書き込みへは 0 を返す必要がある。`index`
/// （`src` と同 shape）を forward と同じ行優先走査順で処理し「各
/// 出力位置（`input_shape` 上の線形添字）ごとの最後の書き手 flat
/// 添字」を求め、その添字だけ `1.0`・それ以外 `0.0` の `index` と
/// 同 shape のマスクを返す（`gather_scatter.rs::scatter` の
/// `resolve_pos` と同じ位置計算式の独立実装——`autodiff` は具体
/// バックエンドクレートへ依存できないため重複実装する。
/// `.claude/rules/coding-rust.md`）。`index` の値は呼び出し元
/// （`Var::scatter`／`scatter_add` の forward）が範囲検査済みの前提。
fn scatter_overwrite_last_writer_mask(
    index: &Tensor<i32>,
    dim: usize,
    input_shape: &[usize],
) -> Tensor<f32> {
    let index_shape = index.shape().to_vec();
    let index_numel: usize = index_shape.iter().product();
    if index_numel == 0 {
        return build_tensor(Vec::new(), &index_shape);
    }
    let index_data = eval::dense_vec_i32(index);
    let input_strides = eval::row_major_strides(input_shape);

    // 出力位置（線形添字）ごとの最後の書き手 flat 添字。行優先で
    // 走査し後勝ちで上書きするだけで「最後の書き手」が求まる。
    let mut last_writer: std::collections::HashMap<usize, usize> =
        std::collections::HashMap::with_capacity(index_numel);
    for (flat, &dim_idx_raw) in index_data.iter().enumerate() {
        let coords = eval::unravel(flat, &index_shape);
        let dim_idx = dim_idx_raw as usize;
        let mut pos = 0usize;
        for (axis, &stride) in input_strides.iter().enumerate() {
            let coord = if axis == dim { dim_idx } else { coords[axis] };
            pos += coord * stride;
        }
        last_writer.insert(pos, flat);
    }
    let mut mask = vec![0f32; index_numel];
    for &flat in last_writer.values() {
        mask[flat] = 1.0;
    }
    build_tensor(mask, &index_shape)
}

/// `Op::RmsNorm` の VJP 本体（イシュー #1596）:
/// `x̂ = x·r`（`r` = `rstd`）・`dx̂ = dy·w`（`w` なしは `dy`）・
/// `dx = r·(dx̂ − x̂·mean(dx̂·x̂))`・`dw = Σ_rows dy·x̂`。
///
/// 行内（`hidden` 軸）の `mean(dx̂·x̂)` は `softmax_vjp_along`
/// （`crates/autodiff/tests` 側の先例。要素積を `f32` で確定してから
/// `f64` へ昇格して蓄積し、`rstd` との最終乗算・`dy` からの減算は
/// `f64` のまま保持して 1 回だけ `f32` へ downcast する）と同じ overflow
/// 回避方針を踏襲する。`dw` の行方向（`rows` 軸）蓄積は
/// `.claude/rules/coding-rust.md`「勾配の長軸縮約の要素積は `f32` で
/// 確定してから `f64` へ昇格して蓄積する」契約に厳密に従う（コメント
/// が挙げる代表例そのもの）。`rows == 0 || hidden == 0` は
/// [`eval::rmsnorm_rows`] と同じ早期 return で空／ゼロ出力を返す。
fn rmsnorm_vjp_rows(
    x: &[f32],
    w: Option<&[f32]>,
    eps: f32,
    rows: usize,
    hidden: usize,
    dy: &[f32],
) -> (Vec<f32>, Option<Vec<f32>>) {
    let mut dx = vec![0.0f32; x.len()];
    let mut dw_acc: Option<Vec<f64>> = w.map(|_| vec![0.0f64; hidden]);
    if rows == 0 || hidden == 0 {
        let dw = dw_acc.map(|v| v.into_iter().map(|a| a as f32).collect());
        return (dx, dw);
    }
    let inv_n = 1.0f64 / hidden as f64;
    for r in 0..rows {
        let row = &x[r * hidden..(r + 1) * hidden];
        let dy_row = &dy[r * hidden..(r + 1) * hidden];
        let rstd = eval::row_rms_stats(row, eps, inv_n);
        let dxhat_at = |i: usize| -> f32 {
            match w {
                Some(w) => dy_row[i] * w[i],
                None => dy_row[i],
            }
        };
        let mut dot_acc = 0.0f64;
        for (i, &xv) in row.iter().enumerate() {
            let xhat = xv * rstd;
            let term = dxhat_at(i) * xhat;
            dot_acc += term as f64;
        }
        let mean_dot = dot_acc * inv_n;
        let dx_row = &mut dx[r * hidden..(r + 1) * hidden];
        for (i, (&xv, dxv)) in row.iter().zip(dx_row.iter_mut()).enumerate() {
            let xhat = xv * rstd;
            let dxhat = dxhat_at(i);
            let d = (rstd as f64) * (dxhat as f64 - (xhat as f64) * mean_dot);
            *dxv = d as f32;
        }
        if let Some(dw_acc) = dw_acc.as_mut() {
            for (i, (&xv, &dyv)) in row.iter().zip(dy_row.iter()).enumerate() {
                let xhat = xv * rstd;
                let term = dyv * xhat;
                dw_acc[i] += term as f64;
            }
        }
    }
    let dw = dw_acc.map(|v| v.into_iter().map(|a| a as f32).collect());
    (dx, dw)
}

/// `Op::LayerNorm` の VJP 本体（イシュー #1596）:
/// `x̂ = (x−μ)·r`・`dx̂ = dy·w`・`dx = r·(dx̂ − mean(dx̂) − x̂·mean(dx̂·x̂))`・
/// `dw = Σ_rows dy·x̂`・`db = Σ_rows dy`。[`rmsnorm_vjp_rows`] と同じ
/// f64 縮約方針（行内 `mean(dx̂)`／`mean(dx̂·x̂)` は要素を `f32` で確定
/// してから `f64` 蓄積・`dw`／`db` の行方向蓄積も同型）。`has_bias` は
/// forward で `bias` が `Some` だったか（`weight` の有無とは独立）を
/// 表し、`db` を計算するかどうかを決める。
fn layer_norm_vjp_rows(
    x: &[f32],
    w: Option<&[f32]>,
    has_bias: bool,
    eps: f32,
    rows: usize,
    hidden: usize,
    dy: &[f32],
) -> (Vec<f32>, Option<Vec<f32>>, Option<Vec<f32>>) {
    let mut dx = vec![0.0f32; x.len()];
    let mut dw_acc: Option<Vec<f64>> = w.map(|_| vec![0.0f64; hidden]);
    let mut db_acc: Option<Vec<f64>> = if has_bias {
        Some(vec![0.0f64; hidden])
    } else {
        None
    };
    if rows == 0 || hidden == 0 {
        let dw = dw_acc.map(|v| v.into_iter().map(|a| a as f32).collect());
        let db = db_acc.map(|v| v.into_iter().map(|a| a as f32).collect());
        return (dx, dw, db);
    }
    let n = hidden as f64;
    for r in 0..rows {
        let row = &x[r * hidden..(r + 1) * hidden];
        let dy_row = &dy[r * hidden..(r + 1) * hidden];
        let (mean, rstd) = eval::row_ln_stats(row, eps, hidden);
        // `mean`／`rstd` を `f64` のまま偏差計算に使い、`x̂` を確定する
        // 直前の 1 回だけ `f32` へ downcast する（forward `eval::
        // layer_norm_rows` と同じ理由。codex-review 指摘: `mean` の
        // 早期丸めは forward・backward 双方の `x̂` を歪める）。
        let xhat_at = |i: usize| -> f32 { ((row[i] as f64 - mean) * rstd) as f32 };
        let dxhat_at = |i: usize| -> f32 {
            match w {
                Some(w) => dy_row[i] * w[i],
                None => dy_row[i],
            }
        };
        let mut sum_dxhat = 0.0f64;
        let mut dot_acc = 0.0f64;
        for i in 0..row.len() {
            let xhat = xhat_at(i);
            let dxhat = dxhat_at(i);
            sum_dxhat += dxhat as f64;
            let term = dxhat * xhat;
            dot_acc += term as f64;
        }
        // `mean` と同じ理由（`row_ln_stats` doc 参照）で、事前丸めした
        // 逆数との積ではなく `hidden` による直接除算で求める。
        let mean_dxhat = sum_dxhat / n;
        let mean_dot = dot_acc / n;
        let dx_row = &mut dx[r * hidden..(r + 1) * hidden];
        for (i, dxv) in dx_row.iter_mut().enumerate() {
            let xhat = xhat_at(i);
            let dxhat = dxhat_at(i);
            let d = rstd * (dxhat as f64 - mean_dxhat - (xhat as f64) * mean_dot);
            *dxv = d as f32;
        }
        if let Some(dw_acc) = dw_acc.as_mut() {
            for (i, &dyv) in dy_row.iter().enumerate() {
                let xhat = xhat_at(i);
                let term = dyv * xhat;
                dw_acc[i] += term as f64;
            }
        }
        if let Some(db_acc) = db_acc.as_mut() {
            for (acc, &dyv) in db_acc.iter_mut().zip(dy_row.iter()) {
                *acc += dyv as f64;
            }
        }
    }
    let dw = dw_acc.map(|v| v.into_iter().map(|a| a as f32).collect());
    let db = db_acc.map(|v| v.into_iter().map(|a| a as f32).collect());
    (dx, dw, db)
}

/// [`Op::Permute`] の VJP（`vjp` 内）が使う逆置換の算出（イシュー
/// #1597）。`perm[k] = p` は「出力軸 `k` が入力軸 `p` を指す」ことを
/// 表すため、逆写像 `inv` は `inv[p] = k` を満たす（`perm ∘ inv ==
/// identity`）。`Var::permute`（`var.rs`）が push 前に `perm` を
/// `0..rank` の順列として検査済み（長さ一致・範囲内・重複なし）のため、
/// 本関数は常に `perm` と同じ長さの妥当な順列を返す（infallible）。
fn inverse_permutation(perm: &[usize]) -> Vec<usize> {
    let mut inv = vec![0usize; perm.len()];
    for (k, &p) in perm.iter().enumerate() {
        inv[p] = k;
    }
    inv
}

/// ゲート演算（RNN／LSTM／GRU セル）の GEMM 部分の VJP 共通ヘルパー
/// （イシュー #1647・設計 `docs/autodiff-rnn-cell-tape-design.md` 決定
/// 5 の列ブロック配置に対応）。`d_pre_blk`（あるゲートブロックの
/// pre-activation 勾配。`[B, w]`）と、その GEMM の片側オペランド
/// （`input_val: [B, D]`・`weight_val: [D, total_cols]`）から、
/// `d_input = d_pre_blk · (weight[:, blk])ᵀ`（`[B, D]`）・
/// `d_weight`（`weight` と同じ `[D, total_cols]`。ブロック外はゼロ埋め）・
/// `d_bias`（`[total_cols]`。同じくブロック外はゼロ埋め）を計算する。
///
/// 全幅（`col_start == 0 && d_pre_blk` の列数 `== total_cols`）の場合は
/// `narrow`／embed を経由せず `weight_val`／結果をそのまま使う（RNN・
/// LSTM の `LstmCell` 分岐が該当。GRU・`LstmHidden` は常に部分幅）。
///
/// [`matmul_vjp`] と同じ `BackendOps::gemm_fp32_strict`（`eval::matmul`
/// へのフォールバックなし。A08）を経由する。
///
/// 戻り値 `(d_input, d_weight, d_bias)`。`clippy::type_complexity` 回避
/// のため [`AffineVjpOutput`] という名前を与える。
type AffineVjpOutput = (Tensor<f32>, Tensor<f32>, Tensor<f32>);

fn affine_vjp(
    ops: &dyn BackendOps,
    input_val: &Tensor<f32>,
    weight_val: &Tensor<f32>,
    d_pre_blk: &Tensor<f32>,
    col_start: usize,
    total_cols: usize,
) -> Result<AffineVjpOutput, AutodiffError> {
    let block_width = d_pre_blk.shape().get(1).copied().unwrap_or(0);
    let in_dim = weight_val.shape().first().copied().unwrap_or(0);
    let full_width = col_start == 0 && block_width == total_cols;

    let weight_blk_owned;
    let weight_blk: &Tensor<f32> = if full_width {
        weight_val
    } else {
        weight_blk_owned = weight_val
            .narrow(1, col_start, block_width)
            .map(|t| t.contiguous())
            .unwrap_or_else(|_| {
                debug_assert!(
                    false,
                    "affine_vjp: narrow(1, col_start, block_width) が失敗した（呼び出し元の \
                     列ブロック整合違反）"
                );
                weight_val.contiguous()
            });
        &weight_blk_owned
    };
    let weight_blk_t = transpose2d(weight_blk);
    let d_input = ops
        .gemm_fp32_strict(d_pre_blk, &weight_blk_t)
        .map_err(AutodiffError::Backend)?;

    let input_t = transpose2d(input_val);
    let d_weight_blk = ops
        .gemm_fp32_strict(&input_t, d_pre_blk)
        .map_err(AutodiffError::Backend)?;
    let d_weight_full = if full_width {
        d_weight_blk
    } else {
        embed_columns_2d(&d_weight_blk, in_dim, total_cols, col_start)
    };

    // 勾配の長軸縮約は f64（`.claude/rules/coding-rust.md`）。
    // `d_pre_blk: [rows, block_width]` → `[block_width]` は行方向
    // （軸 0）縮約であり `reduce_bias_grad` の row-axis 判定を満たす
    // ため f64 アキュムレータ経路（`eval::reduce_bias_grad_rows`）へ
    // 委譲する（RNN／LSTM／GRU 共通 bias 勾配。イシュー #1647
    // codex-review P1 是正: 旧 `reduce_to_shape` は f32 逐次和のため
    // 大きく相殺する上流勾配で桁落ちする）。
    let d_bias_blk = reduce_bias_grad(d_pre_blk, &[block_width]);
    let d_bias_full = if full_width {
        d_bias_blk
    } else {
        embed_columns_1d(&d_bias_blk, total_cols, col_start)
    };

    Ok((d_input, d_weight_full, d_bias_full))
}

/// `partial: [rows, block_width]` を `[rows, total_cols]` の零行列の
/// `[col_start, col_start+block_width)` 列範囲へ埋め込む（[`affine_vjp`]
/// の重み勾配の列ブロック配置を復元するためのホスト側 scatter）。
fn embed_columns_2d(
    partial: &Tensor<f32>,
    rows: usize,
    total_cols: usize,
    col_start: usize,
) -> Tensor<f32> {
    let block_width = partial.shape().get(1).copied().unwrap_or(0);
    let partial_data = dense_vec(partial);
    let mut out = vec![0f32; rows * total_cols];
    for r in 0..rows {
        for c in 0..block_width {
            out[r * total_cols + col_start + c] = partial_data[r * block_width + c];
        }
    }
    build_tensor(out, &[rows, total_cols])
}

/// `partial: [block_width]` を `[total_cols]` の零ベクトルの
/// `[col_start, col_start+block_width)` 範囲へ埋め込む（[`affine_vjp`]
/// の bias 勾配の列ブロック配置を復元するためのホスト側 scatter）。
fn embed_columns_1d(partial: &Tensor<f32>, total_cols: usize, col_start: usize) -> Tensor<f32> {
    let block_width = partial.shape().first().copied().unwrap_or(0);
    let partial_data = dense_vec(partial);
    let mut out = vec![0f32; total_cols];
    out[col_start..col_start + block_width].copy_from_slice(&partial_data);
    build_tensor(out, &[total_cols])
}

/// 2 次元 `matmul` の転置。shape 検査は forward（`Var::matmul` →
/// `matmul_out_shape`）が済ませた 2 次元前提であり、`transpose(0, 1)`
/// は構造的に失敗しえない。それでも本番経路で `unwrap()`/`expect()`
/// を使わない方針（`.claude/rules/coding-rust.md`）のため、失敗時は
/// `debug_assert!` で契約違反を検知しつつ入力をそのまま返す
/// （到達すれば forward 側の shape 検査ロジックにバグがある）。
fn transpose2d(tensor: &Tensor<f32>) -> Tensor<f32> {
    match tensor.transpose(0, 1) {
        Ok(t) => t,
        Err(_) => {
            debug_assert!(
                false,
                "transpose2d: matmul VJP の rank-2 前提が崩れた（forward 側の契約違反）"
            );
            tensor.clone()
        }
    }
}

/// `MatMul(A, B)` の VJP: `dA = g @ Bᵀ`、`dB = Aᵀ @ g`
/// （`A: [m,k]`・`B: [k,n]`・`g: [m,n]`）。イシュー #1211: forward と
/// 同じ `BackendOps::gemm_fp32_strict`（CPU は BLIS 並列 GEMM・CUDA/Metal
/// はデバイス GEMM。`gemm` と同じカーネルを使うが、CUDA の TF32 opt-in
/// フラグ〈`set_cuda_tf32_gemm_enabled`〉には追従せず常に FP32 厳密で
/// 計算する。backward は `docs/cuda-tf32-optin-api-decision.md`・
/// `backend-cuda::precision` モジュール冒頭コメントの契約でスコープ外の
/// まま FP32 のため区別する。codex-review 指摘・PR #1223）を経由する
/// ため、backward の支配的コスト（`docs/perf/
/// train-step-phase-breakdown.md` §11・§15）がバックエンド既定の並列・
/// デバイス実装の恩恵を受ける。FMA 契約は各バックエンドの `gemm` 既定
/// 契約に従う（`coding-rust.md` の FMA 契約統一方針。forward の
/// `Var::matmul` と同一カーネルを通るため経路間で分岐しない）。
///
/// `transpose2d`（下記。`Tensor::transpose` の zero-copy stride view）
/// で作った転置オペランドはそのまま `ops.gemm_fp32_strict` へ渡す。
/// CPU（イシュー #1213）は片側転置（NT/TN）かつ dense な転置格納
/// （`strides() == [1, shape()[0]]`）と判定できる場合に限り
/// `contiguous()` の再パックコピーを経由せず BLIS packing 側で直接
/// 吸収する専用入口（CPU 実装クレートの `gemm_blis_parallel_nt`／
/// `gemm_blis_parallel_tn`）へ分岐する。両方転置（TT）・一般 stride
/// （`narrow` 後の転置等）は CPU でも従来どおり `contiguous()` を経由
/// する（一般 stride 化はスコープ外。`docs/matmul-vjp-zero-copy-
/// decision.md` §3.2・§4.2・§4.3 追補）。CUDA（イシュー #1214）も同型の
/// NT/TN 専用入口（GPU 側 smem 転置カーネル → 既存 NN GEMM カーネル。
/// `CudaGemm::run_tiled_f32_nt`／`run_tiled_f32_tn`）を持つ。Metal
/// （イシュー #1215）も同型の NT/TN 判定で classic strided カーネル
/// 入口（`gemm::MetalGemm::dispatch_strided_bias_act_prepared`）へ分岐
/// する専用経路を持つが、既存 NN 経路（`dispatch_auto`）とは別カーネル
/// のため数値契約は bit 一致ではなく REQ-2 統一複合判定
/// （`docs/matmul-vjp-zero-copy-decision.md` §4.4）。
///
/// エラーは fail-closed で `AutodiffError::Backend` として伝播し、
/// `eval::matmul` への暗黙フォールバックは設けない（forward と backward
/// で数値経路が分岐する判定迂回を作らないため。`.claude/rules/
/// security.md` A08）。
fn matmul_vjp(
    ops: &dyn BackendOps,
    a: &Tensor<f32>,
    b: &Tensor<f32>,
    g: &Tensor<f32>,
) -> Result<(Tensor<f32>, Tensor<f32>), AutodiffError> {
    if a.shape().len() == 2 && b.shape().len() == 2 {
        let b_t = transpose2d(b);
        let a_t = transpose2d(a);
        let da = ops
            .gemm_fp32_strict(g, &b_t)
            .map_err(AutodiffError::Backend)?;
        let db = ops
            .gemm_fp32_strict(&a_t, g)
            .map_err(AutodiffError::Backend)?;
        return Ok((da, db));
    }

    // rank≥3（バッチ次元を含む）の matmul VJP（イシュー #1715）。
    // `transpose_last2`（末尾 2 軸のみ転置する zero-copy view。バッチ軸
    // は不変）で転置オペランドを作り、`gemm_batched_fp32_strict`
    // （`gemm_fp32_strict` のバッチ版。TF32 opt-in に追従しない）で
    // フル（broadcast 後）バッチ形状の勾配を計算したのち、
    // `reduce_batch_axes_f64` で `a`／`b` それぞれの元のバッチ形状
    // （broadcast されていた軸）へ `f64` アキュムレータで縮約する
    // （`.claude/rules/coding-rust.md` の「勾配の長軸縮約は `f64`
    // アキュムレータで統一する」）。バッチ形状が forward と同一
    // （broadcast なし）の場合は縮約が恒等になり per-batch 2 次元 VJP
    // と bit 一致する。
    let b_t = transpose_last2(b);
    let a_t = transpose_last2(a);
    let da_full = ops
        .gemm_batched_fp32_strict(g, &b_t)
        .map_err(AutodiffError::Backend)?;
    let db_full = ops
        .gemm_batched_fp32_strict(&a_t, g)
        .map_err(AutodiffError::Backend)?;
    let da = reduce_batch_axes_f64(&da_full, a.shape())?;
    let db = reduce_batch_axes_f64(&db_full, b.shape())?;
    Ok((da, db))
}

/// [`matmul_vjp`] の rank≥3 経路が使う、末尾 2 軸のみを転置する
/// zero-copy view（イシュー #1715）。バッチ軸（先頭 rank−2 軸）は
/// 不変のまま、行列積の m/k・k/n に相当する末尾 2 軸だけを入れ替える
/// （`transpose2d` の 2 次元専用版をバッチ次元へ一般化したもの）。
/// `tensor.shape().len() < 2` は forward（`Var::matmul` →
/// `matmul_out_shape`）が rank≥2 を保証済みのため構造的に到達しない
/// （`transpose2d` と同じ `debug_assert!` + フォールバック方針）。
fn transpose_last2(tensor: &Tensor<f32>) -> Tensor<f32> {
    let rank = tensor.shape().len();
    if rank < 2 {
        debug_assert!(
            false,
            "transpose_last2: matmul VJP の rank≥2 前提が崩れた（forward 側の契約違反）"
        );
        return tensor.clone();
    }
    match tensor.transpose(rank - 2, rank - 1) {
        Ok(t) => t,
        Err(_) => {
            debug_assert!(
                false,
                "transpose_last2: transpose が失敗した（forward 側の契約違反）"
            );
            tensor.clone()
        }
    }
}

/// [`matmul_vjp`] のバッチ軸縮約専用ヘルパー（イシュー #1715）。
///
/// `g`（`ops.gemm_batched_fp32_strict` が返す broadcast 後フル
/// バッチ形状の勾配）を、`target_shape`（`a`／`b` 自身の broadcast 前
/// バッチ形状）へ、先頭のバッチ軸（末尾 2 軸＝行列積の m/k・k/n に
/// 相当する軸は broadcast されない契約のため対象外）についてのみ和を
/// 取って縮約する。`reduce_to_shape`（`f32` 逐次和。bias 縮約以外の
/// 一般 broadcast 用）とはアキュムレータの数値方式が異なる別関数と
/// して独立に保つ: 本関数は要素を直接 `f64` へ昇格して縮約全体を
/// `f64` で行い、最後に 1 回だけ `f32` へ downcast する
/// （`.claude/rules/coding-rust.md` の「勾配の長軸縮約は `f64`
/// アキュムレータで統一する」方針をバッチ軸縮約にも適用する）。
///
/// 空の勾配から `target_shape` のゼロテンソルを復元する経路は、`g` が
/// 空でも `target_shape` 自体は確保可能とは限らない（例: 1 要素を
/// `[1, H, 1]`〈H = isize::MAX / 4 + 1〉へ broadcast した `a` と空の
/// `b = [0, 1, 1]` では `da_full = [0, H, 1]` は空だが縮約先は H 個の
/// f32）ため、`Tensor::zeros` の確保前検証（バイトサイズ ≤ isize::MAX）
/// を型付きエラーとして伝播する（PR #1810 codex-review P1 是正）。
fn reduce_batch_axes_f64(
    g: &Tensor<f32>,
    target_shape: &[usize],
) -> Result<Tensor<f32>, AutodiffError> {
    let g_shape = g.shape().to_vec();
    if g_shape == target_shape {
        return Ok(g.clone());
    }
    debug_assert!(
        g_shape.len() >= target_shape.len(),
        "reduce_batch_axes_f64: broadcast 後 shape の rank は入力 rank 以上のはず（契約違反）"
    );
    let rank_diff = g_shape.len() - target_shape.len();
    let mut padded_target = vec![1usize; rank_diff];
    padded_target.extend_from_slice(target_shape);

    // 勾配が空（いずれかの軸が 0 サイズ）なら縮約結果は `target_shape`
    // のゼロテンソルで確定するため、軸ごとの縮約ループへ入らずに返す
    // （PR #1810 codex-review P2 是正）。空の `k`／`m` 軸を持つ入力は
    // 実データなしで巨大なバッチ軸（例: `a = [1, 0, 1]`・
    // `b = [2^40, 1, 0]`）を構成でき、`inner == 0` でも `axis_len`
    // 回の空ループを回すと最適化なしビルドで実質停止する。
    if g.numel() == 0 {
        return Tensor::zeros(target_shape)
            .map_err(|err| AutodiffError::Backend(BackendError::ShapeMismatch(err)));
    }

    let data = dense_vec(g);
    let mut acc: Vec<f64> = data.iter().map(|&x| x as f64).collect();
    let mut cur_shape = g_shape;
    for axis in 0..cur_shape.len() {
        if padded_target[axis] == 1 && cur_shape[axis] != 1 {
            let outer: usize = cur_shape[..axis].iter().product();
            let axis_len = cur_shape[axis];
            let inner: usize = cur_shape[axis + 1..].iter().product();
            let mut reduced = vec![0f64; outer * inner];
            for o in 0..outer {
                for a in 0..axis_len {
                    for i in 0..inner {
                        let src = (o * axis_len + a) * inner + i;
                        reduced[o * inner + i] += acc[src];
                    }
                }
            }
            acc = reduced;
            cur_shape[axis] = 1;
        }
    }
    let out: Vec<f32> = acc.iter().map(|&x| x as f32).collect();
    Ok(build_tensor(out, target_shape))
}

/// ブロードキャストの逆演算。`add`/`mul` の VJP が返す勾配は forward
/// 出力の shape（ブロードキャスト後）を持つため、元の入力 shape
/// （`target_shape`）へ縮約する必要がある。NumPy 風ブロードキャスト
/// は「先頭に新設された軸」と「入力側が size 1 だった軸」を複製する
/// ため、その逆演算は同じ軸集合を合計で潰せばよい（PoC-v2-2 の
/// `sum_rows_data` を任意 shape・任意軸へ一般化した実装）。
fn reduce_to_shape(g: &Tensor<f32>, target_shape: &[usize]) -> Tensor<f32> {
    let g_shape = g.shape().to_vec();
    if g_shape == target_shape {
        return g.clone();
    }
    debug_assert!(
        g_shape.len() >= target_shape.len(),
        "reduce_to_shape: broadcast 後 shape の rank は入力 rank 以上のはず（契約違反）"
    );
    let rank_diff = g_shape.len() - target_shape.len();
    let mut padded_target = vec![1usize; rank_diff];
    padded_target.extend_from_slice(target_shape);

    let mut data = dense_vec(g);
    let mut cur_shape = g_shape;
    for axis in 0..cur_shape.len() {
        if padded_target[axis] == 1 && cur_shape[axis] != 1 {
            let outer: usize = cur_shape[..axis].iter().product();
            let axis_len = cur_shape[axis];
            let inner: usize = cur_shape[axis + 1..].iter().product();
            let mut reduced = vec![0f32; outer * inner];
            for o in 0..outer {
                for a in 0..axis_len {
                    for i in 0..inner {
                        let src = (o * axis_len + a) * inner + i;
                        reduced[o * inner + i] += data[src];
                    }
                }
            }
            data = reduced;
            cur_shape[axis] = 1;
        }
    }
    build_tensor(data, target_shape)
}

/// bias 勾配専用の縮約ディスパッチ（イシュー #1566・PR #1659 codex-review
/// P1 是正・2026-09-12 ユーザー承認 A の横展開）。
///
/// `Op::LinearResident` は resident 経由の成功時（`outcome.bias_filled`）
/// `eval::reduce_bias_grad_rows`（`f64` アキュムレータ。ホスト経路）・
/// GPU カーネル `gemm_bias_grad_reduce_f32`（binary64 加算の 64bit 整数
/// エミュレーション。ホストと bit 一致）のいずれか
/// で bias を計算する（`docs/backend-metal-command-batching-design.md`
/// §10.8）。`outcome.bias_filled == false`（CPU／CUDA 等 resident 非対応
/// バックエンド、または weight tying で bias 自身が非 resident 扱いに
/// なった場合のフォールバック）だけが `g` を単純な `f32` 逐次和
/// （`reduce_to_shape`）で縮約していたため、**同一の `Op::LinearResident`
/// が実行環境（バックエンド／resident 対応可否）によって異なる縮約方式
/// を使う**という不整合があった（`[1e8, 1.0, -1e8]` のような相殺入力で
/// Metal resident 経路と CPU／CUDA フォールバックが食い違う。codex-review
/// 指摘）。
///
/// `Op::LinearAct`（`Op::LinearResident` の resident 化を伴わない同型の
/// bias 縮約）にも同じ理由で適用し、両 Op 間の縮約方式を揃える
/// （fresh〈`LinearAct`〉と reuse〈`LinearResident` フォールバック〉が
/// 同じ形状パターンで異なる数値を返さないようにする）。
///
/// **`Op::Add` への横展開（2026-09-12 ユーザー承認・PR #1659→#1665→
/// #1666 取り込み後の追加是正）**: `nn::Linear` の既定 forward 経路
/// （`LinearVars::forward`。`matmul → add` の非融合合成。`nn/linear.rs`
/// doc「bias 加算は `Var::add` の broadcast に委ねる」参照）は `Op::Add`
/// の VJP を経由するため、`Op::Add(a, b)` 側でも本関数へ委譲する
/// （`da`／`db` 双方。`Op::Add` は可換なので bias がどちらの引数に来ても
/// 対称に扱える）。これにより、同じ `Linear` 層が `LinearVars::forward`
/// （fresh・非融合）と `forward_with_activation`（`Op::LinearAct`。
/// epilogue 融合）・`DeviceParamStore::linear_forward_with_activation`
/// （`Op::LinearResident`。reuse）のいずれで forward されても bias
/// 勾配の数値方式が揃う。`Op::Add` は bias 以外の一般的な broadcast
/// （bias パターンに一致しない任意 shape の加算）にも使われる汎用 Op
/// のため、下記の shape 構造判定を満たさない呼び出しは本関数の内部で
/// 既存の `reduce_to_shape` へそのまま委譲され挙動を変えない（bias
/// パターンに限定した横展開であり、汎用 `Op::Add`・`reduce_to_shape`
/// 本体自体は不変）。
///
/// 適用条件は `g` が rank-2 `[m, n]` かつ `target_shape` が「軸 0
/// （行／batch 軸）方向の縮約」を表す形状（末尾次元が `n` と一致し、
/// それより前の全次元が `1`。典型例: `[n]`・`[1, n]`。`nn::Linear` の
/// bias `[out_features]` を含む）の場合に限る。`eval::reduce_bias_grad_
/// rows` は「rank-2 入力を行 `0..m` で縮約し列ごとの和を返す」という
/// 固定の契約（`gemm_bias_grad_reduce_f32` の `MatrixLayout` と同型）
/// のため、**軸 1（列）方向を縮約する broadcast 形状（例: `g: [2, 2]`
/// に対する `target_shape: [2, 1]`。各行の bias が列方向へ複製される
/// パターン）には適用できない**——列ごとの和という異なる縮約軸の結果を
/// 返してしまい、値そのものが誤りになる（PR #1659 codex-review P2
/// 是正。回帰テスト `reduce_bias_grad_does_not_misapply_row_reduction_
/// to_column_broadcast_bias` 参照）。この判定は総要素数の一致だけでは
/// 検出できない（`[2, 1]` も総要素数 `2` で `g` の列数 `2` と一致して
/// しまうため、旧実装は shape 構造を見ずに誤って f64 経路へ分岐して
/// いた）。`nn::Linear`〈`from_parameters` が bias を `[out_features]`
/// 厳密一致にしか構築しない〉経由では軸 1 縮約の broadcast bias は
/// 到達しない（`pub(crate) fn linear_act` を直接呼ぶ経路・`Op::Add`
/// 経由で `Var::add` に非 bias 形状の broadcast を直接渡す経路限定。
/// `var.rs` doc「`linear_act` は `[n]` と厳密一致しない broadcast
/// 可能な bias も受理する」参照）。`Op::Add` への横展開後もこの shape
/// 構造判定自体は不変であり、`[2, 1]` のような軸 1 縮約は `Op::Add`
/// 経由でも同様に誤適用を回避する。適用条件を満たさない場合は既存の
/// `reduce_to_shape`（`f32` 逐次和・任意 rank・任意軸対応）のまま
/// 維持し挙動を変えない（安全側）。
fn reduce_bias_grad(g: &Tensor<f32>, target_shape: &[usize]) -> Tensor<f32> {
    let g_shape = g.shape();
    let is_row_axis_reduction = g_shape.len() == 2
        && target_shape.last() == Some(&g_shape[1])
        && target_shape[..target_shape.len().saturating_sub(1)]
            .iter()
            .all(|&d| d == 1);
    if is_row_axis_reduction {
        let data = eval::reduce_bias_grad_rows(g);
        return build_tensor(data, target_shape);
    }
    reduce_to_shape(g, target_shape)
}

/// 同 shape の 2 テンソルに対する要素ごとの条件付き選択
/// （`g` をそのまま通すか 0 にするかを `mask_src` の値で決める）。
/// `Relu` の VJP（`g ⊙ 1[x > 0]`）専用の最小実装。
///
/// **イシュー #1577**: reuse 学習（`DeviceParamStore` 経由・
/// `Op::LinearResident`）の backward では、下流層の VJP が返す
/// `d_input = transpose2d(&tmp)`（本ファイル内 `Op::LinearResident`
/// 分岐）が stride `[1, m]` のゼロコピー転置 view であり、単一寄与
/// なら `backward.rs::accumulate` がコピーせずそのまま上流層の
/// `upstream` になる。旧実装は `dense_vec`（`eval::dense_vec` →
/// `Tensor::contiguous()`）が非連続入力を要素ごと `get(&index)`
/// （rank 検査・軸ごとの範囲検査を伴う）で走査するため、連続入力比で
/// 大幅に劣化していた（実測は `docs/perf/
/// lowlayer-diagnosis-2026-09-12.md` §4・`docs/perf/
/// train-reuse-relu-mask-stride.md`）。
///
/// 本実装は `g`・`mask_src`（`Op::LinearAct`／`Op::LinearResident` では
/// `out_value` が `materialize_fallible` 経由で view になりうる）の
/// 双方を独立に [`Tensor::as_view_slice`]（全 strides が非負な限り
/// `contiguous()` を経由せず storage を借用で読む。`as_slice` が成功
/// するケース〈真に contiguous〉も同じ formula で正しく読める＝分岐を
/// 増やさず包含する）で読み、`get()` の rank・範囲検査コストを避けて
/// 出力を 1 パスで構築する。`as_view_slice` が `None`（負 stride 等。
/// 現行公開 API の `transpose`/`narrow`/`broadcast_to` はいずれも負
/// stride を生成しないため到達しないが将来拡張への fail-safe）の
/// 場合や shape 不一致・オフセット計算のオーバーフロー等、想定外の
/// 状態を検知した場合は、静かに 0 で埋めたり判定を迂回したりせず、
/// `dense_vec` を使う既存の走査へ**経路全体を丸ごと**フォールバック
/// する（数値的に同一のコピー経路であり、`.claude/rules/security.md`
/// A08 が禁じる判定迂回ではない）。出力は要素ごとの選択（算術なし）
/// のため走査順に依存せず bit 同一（run-to-run・変更前後とも）を
/// 維持する。
fn elementwise_mul_mask(
    g: &Tensor<f32>,
    mask_src: &Tensor<f32>,
    keep: impl Fn(f32) -> bool,
) -> Tensor<f32> {
    let shape = g.shape().to_vec();
    if let Some(out) = try_elementwise_mul_mask_strided(g, mask_src, &shape, &keep) {
        return build_tensor(out, &shape);
    }
    let g_data = dense_vec(g);
    let mask_data = dense_vec(mask_src);
    let out: Vec<f32> = g_data
        .iter()
        .zip(mask_data.iter())
        .map(|(&gv, &mv)| if keep(mv) { gv } else { 0.0 })
        .collect();
    build_tensor(out, &shape)
}

/// [`elementwise_mul_mask`] の stride 対応主経路。読み出しに失敗しうる
/// 要因（shape 不一致・オフセット計算オーバーフロー）を検出した場合は
/// `None` を返し、呼び出し元が `dense_vec` 経路へ丸ごとフォールバック
/// する（部分的に誤った値を返さない）。
fn try_elementwise_mul_mask_strided(
    g: &Tensor<f32>,
    mask_src: &Tensor<f32>,
    shape: &[usize],
    keep: &impl Fn(f32) -> bool,
) -> Option<Vec<f32>> {
    if mask_src.shape() != shape {
        // 既存実装（`dense_vec` の zip）は shape 不一致時に短い方へ
        // 暗黙に切り詰めていた。多次元 index による読み出しはこの
        // 前提を要求するため、不一致時は無条件でフォールバックし
        // 既存の暗黙切り詰め挙動をそのまま保つ。
        return None;
    }
    let numel: usize = shape.iter().product();
    let g_op = MaskReadOperand::classify(g);
    let mask_op = MaskReadOperand::classify(mask_src);

    // fresh 経路（`Op::Relu`／`Op::LinearAct` の `upstream` が
    // `matmul_vjp` の連続な GEMM 出力である通常ケース）を含む、
    // 両オペランドとも連続な最頻ケースの高速経路。`read(idx, flat)`
    // 経由の enum ディスパッチ・オフセット計算を経由せず、借用スライス
    // 2 本の `zip`／`map`／`collect` に落とすことでコンパイラの自動
    // ベクトル化を妨げない（`MaskReadOperand::read` 経由の一般化した
    // 経路は非連続 view 専用に限定する）。
    if let (MaskReadOperand::Contig(g_s), MaskReadOperand::Contig(m_s)) = (&g_op, &mask_op) {
        return Some(
            g_s.iter()
                .zip(m_s.iter())
                .map(|(&gv, &mv)| if keep(mv) { gv } else { 0.0 })
                .collect(),
        );
    }

    if shape.len() == 2 {
        let (rows, cols) = (shape[0], shape[1]);
        // reuse backward の実際のホットパス（下流層 d_input が
        // `transpose2d` のゼロコピー view・上流層 `out_value` は連続な
        // forward 記録値、またはその逆）を狙い撃ちした専用経路。片方が
        // `Contig`（行優先の連続スライスを直接インデックス）・片方が
        // `View`（行ごとの基準オフセット `i * s0` を 1 回だけ計算し、
        // 列方向は `+ j * s1` の加算のみ）に限定して読み出すことで、
        // 一般化した `MaskReadOperand::read` 経由（列ごとに strides を
        // ゼロから内積するオーバーヘッド）より高速化する（実測は
        // `docs/perf/train-reuse-relu-mask-stride.md` §5）。
        if let (
            MaskReadOperand::Contig(g_s),
            MaskReadOperand::View {
                span: m_span,
                strides: m_strides,
            },
        ) = (&g_op, &mask_op)
        {
            let (ms0, ms1) = (m_strides[0], m_strides[1]);
            let mut out = Vec::with_capacity(numel);
            for i in 0..rows {
                let row_start = i * cols;
                let g_row = g_s.get(row_start..row_start + cols)?;
                let m_base = i * ms0;
                for (j, &gv) in g_row.iter().enumerate() {
                    let mv = *m_span.get(m_base + j * ms1)?;
                    out.push(if keep(mv) { gv } else { 0.0 });
                }
            }
            return Some(out);
        }
        if let (
            MaskReadOperand::View {
                span: g_span,
                strides: g_strides,
            },
            MaskReadOperand::Contig(m_s),
        ) = (&g_op, &mask_op)
        {
            let (gs0, gs1) = (g_strides[0], g_strides[1]);
            let mut out = Vec::with_capacity(numel);
            for i in 0..rows {
                let row_start = i * cols;
                let m_row = m_s.get(row_start..row_start + cols)?;
                let g_base = i * gs0;
                for (j, &mv) in m_row.iter().enumerate() {
                    let gv = *g_span.get(g_base + j * gs1)?;
                    out.push(if keep(mv) { gv } else { 0.0 });
                }
            }
            return Some(out);
        }
    }

    let mut out = Vec::with_capacity(numel);

    if shape.len() == 2 {
        // 上記 2 分岐（片方 `Contig`・片方 `View`）に該当しない rank-2
        // （両方 `View`／`Owned` を含むケース）向けの一般経路。固定長
        // 2 要素の index 配列のみでスタック上で完結し、一般 N-d 経路の
        // `Vec<usize>` 繰り上げより軽い。
        let (rows, cols) = (shape[0], shape[1]);
        for i in 0..rows {
            for j in 0..cols {
                let idx = [i, j];
                let flat = i * cols + j;
                let gv = g_op.read(&idx, flat)?;
                let mv = mask_op.read(&idx, flat)?;
                out.push(if keep(mv) { gv } else { 0.0 });
            }
        }
        return Some(out);
    }

    // 一般 N-d: index ベクタを行優先（最終軸が最速）で繰り上げる。
    let mut idx = vec![0usize; shape.len()];
    for flat in 0..numel {
        let gv = g_op.read(&idx, flat)?;
        let mv = mask_op.read(&idx, flat)?;
        out.push(if keep(mv) { gv } else { 0.0 });
        for axis in (0..shape.len()).rev() {
            idx[axis] += 1;
            if idx[axis] < shape[axis] {
                break;
            }
            idx[axis] = 0;
        }
    }
    Some(out)
}

/// [`elementwise_mul_mask`] が読む 1 オペランド分の抽象。
///
/// `as_slice()`（真に contiguous）が成功すれば `Contig` として最優先で
/// 扱う（`try_elementwise_mul_mask_strided` の全 contig 高速経路・
/// rank-2／一般 N-d 経路いずれからも `flat` 添字で直接読める）。次に
/// `as_view_slice()`（`transpose`/`narrow`/`broadcast_to` の非負
/// stride view を含む）が成功すれば `View` として **`usize` へ変換
/// 済みの** strides 付きで借用を保持する。いずれも失敗した場合のみ
/// `dense_vec`（コピー）を保持する `Owned` へフォールバックする。
///
/// `View` のオフセット計算（[`Self::read`]）は要素ごとに `checked_mul`/
/// `checked_add`/`isize`↔`usize` 変換を経由せず、プレーンな `usize`
/// 乗算・加算のみを行う。安全性の根拠: `as_view_slice()` が `Some` を
/// 返した時点で全 strides が非負であることが確定しており（`classify`
/// で 1 回だけ `usize` へ変換）、かつ同メソッドは
/// `span = 1 + Σ (shape_i − 1)·stride_i` を `checked_add`/`checked_mul`
/// で検証済みである。したがって shape 範囲内の任意の `idx` に対し
/// `Σ idx_i·stride_i < span == span.len()` が保証され、本メソッド内で
/// 改めて overflow を心配する必要はない（対象テンソルの要素数は
/// 学習用途の実用範囲で `usize::MAX` に遠く及ばない）。境界外
/// アクセスの検出自体は最終的な `span.get(off)` の 1 回の `Option`
/// 判定に集約し、そこで `None` になった場合のみ
/// `try_elementwise_mul_mask_strided` 全体が `dense_vec` 経路へ
/// フォールバックする（`.claude/rules/coding-rust.md` の `unwrap`／
/// `expect` 非使用方針を保ちつつ、要素ごとの checked 演算チェーンに
/// よる速度低下〈初版実装で実測。`docs/perf/
/// train-reuse-relu-mask-stride.md` §5 参照〉を避ける）。
enum MaskReadOperand<'a> {
    Contig(&'a [f32]),
    View {
        span: &'a [f32],
        strides: Vec<usize>,
    },
    Owned(Vec<f32>),
}

impl<'a> MaskReadOperand<'a> {
    fn classify(t: &'a Tensor<f32>) -> Self {
        if let Some(s) = t.as_slice() {
            return MaskReadOperand::Contig(s);
        }
        if let Some(span) = t.as_view_slice() {
            // `as_view_slice()` が `Some` を返した時点で strides は
            // 全て非負が保証されるため、この `usize::try_from` は
            // 通常失敗しない。万一の不整合（`Tensor` 側の契約違反）
            // に備え、フォールバック先である `Owned` へ迂回する。
            let strides: Option<Vec<usize>> = t
                .strides()
                .iter()
                .map(|&s| usize::try_from(s).ok())
                .collect();
            if let Some(strides) = strides {
                return MaskReadOperand::View { span, strides };
            }
        }
        MaskReadOperand::Owned(dense_vec(t))
    }

    /// `Contig`／`Owned` の場合は `flat`（行優先の平坦 index。
    /// `as_slice()`／`dense_vec` の走査順と一致）で、`View` の場合は
    /// `idx`（strides との内積でオフセットを計算。プレーン `usize`
    /// 演算のみ・型定義側 doc 参照）で読む。境界外アクセスを検知
    /// した場合（`View` の `span.get` が `None` を返す場合。通常到達
    /// しない防御的経路）は `None` を返し、呼び出し元の
    /// `try_elementwise_mul_mask_strided` 全体を `dense_vec` 経路へ
    /// フォールバックさせる。両オペランドとも `Contig` の最頻ケースは
    /// この汎用経路を経由せず、呼び出し元の専用高速経路で処理する
    /// （enum ディスパッチのオーバーヘッドを避けるため）。
    #[inline]
    fn read(&self, idx: &[usize], flat: usize) -> Option<f32> {
        match self {
            MaskReadOperand::Contig(s) => s.get(flat).copied(),
            MaskReadOperand::View { span, strides } => {
                let mut off = 0usize;
                for (&i, &s) in idx.iter().zip(strides.iter()) {
                    off += i * s;
                }
                span.get(off).copied()
            }
            MaskReadOperand::Owned(v) => v.get(flat).copied(),
        }
    }
}

/// `Tanh` の VJP 係数 `1 - tanh(x)^2` を forward 記録値 `out_value`
/// （= `tanh(x)`）から計算する（再計算を避ける）。
fn tanh_grad_factor(out_value: &Tensor<f32>) -> Tensor<f32> {
    let shape = out_value.shape().to_vec();
    let data = dense_vec(out_value);
    let out: Vec<f32> = data.iter().map(|&v| 1.0 - v * v).collect();
    build_tensor(out, &shape)
}

/// `Sigmoid` の VJP 係数 `sigmoid(x) * (1 - sigmoid(x))` を forward
/// 記録値 `out_value`（= `sigmoid(x)`）から計算する（TASK-9.1b・#92。
/// `tanh_grad_factor` と同型の out_value 再利用パターン）。
fn sigmoid_grad_factor(out_value: &Tensor<f32>) -> Tensor<f32> {
    let shape = out_value.shape().to_vec();
    let data = dense_vec(out_value);
    let out: Vec<f32> = data.iter().map(|&v| v * (1.0 - v)).collect();
    build_tensor(out, &shape)
}

/// `Op::Softmax` の VJP 本体: `dx = y ⊙ (g − Σ_dim(g ⊙ y))`。
/// `eval::softmax_along`／`log_softmax_along` と同じ「外側（outer）×
/// 走査軸（axis_len）× 内側（inner）」の 3 段走査（`dim` は forward
/// 側で範囲検査済みの前提）。`Σ_dim(g ⊙ y)` の要素積は `f32` で確定
/// してから `f64` へ昇格して蓄積し（`.claude/rules/coding-rust.md`
/// 「勾配の長軸縮約の要素積は f32 で確定してから f64 へ昇格」）、
/// `g` からの減算・`y` との最終乗算も（`log_softmax_vjp_along` と同じ
/// 理由で）`f64` のまま保持し、最終書き出しで 1 回だけ `f32` へ
/// downcast する（縮約値を先に `f32` へ戻すと、有限の `f32` 入力でも
/// 減算・乗算の結果が overflow しうるため）。
fn softmax_vjp_along(out_value: &Tensor<f32>, upstream: &Tensor<f32>, axis: usize) -> Tensor<f32> {
    let shape = out_value.shape().to_vec();
    // `log_softmax_vjp_along` 直下と同じ早期 return（部分積オーバー
    // フロー回避。`eval::softmax_along` 冒頭のコメント参照）。
    if shape.contains(&0) {
        return build_tensor(Vec::new(), &shape);
    }
    let outer: usize = shape[..axis].iter().product();
    let axis_len = shape[axis];
    let inner: usize = shape[axis + 1..].iter().product();
    let y = dense_vec(out_value);
    let g = dense_vec(upstream);
    let mut out = vec![0f32; y.len()];
    for o in 0..outer {
        for i in 0..inner {
            let mut dot_acc: f64 = 0.0;
            for a in 0..axis_len {
                let idx = (o * axis_len + a) * inner + i;
                let term = g[idx] * y[idx];
                dot_acc += term as f64;
            }
            // `dot_acc`（f64）を乗算前に `f32` へ downcast すると、
            // `y * (g - dot)` が有限の `f32` 入力でも overflow しうる
            // （例: `y=[0.25,0.75]`・上流勾配 `g=[3e38,-3e38]` で正しい
            // 入力勾配 `[~1.125e38, ...]` が `[inf, ...]` になる）。
            // `log_softmax_vjp_along` と同じ f64 アキュムレータ契約
            // （`.claude/rules/coding-rust.md`）に従い、`g` からの減算・
            // `y` との最終乗算まで f64 で保持し、最終書き出しでのみ
            // `f32` へ downcast する。
            for a in 0..axis_len {
                let idx = (o * axis_len + a) * inner + i;
                let d = (y[idx] as f64) * (g[idx] as f64 - dot_acc);
                out[idx] = d as f32;
            }
        }
    }
    build_tensor(out, &shape)
}

/// `Op::LogSoftmax` の VJP 本体: `dx = g − exp(y) ⊙ Σ_dim(g)`。
/// `softmax_vjp_along` と同じ 3 段走査・f64 縮約方針（本関数の縮約は
/// 要素積ではなく単純和のため、`g` の各要素をそのまま `f64` へ昇格して
/// 蓄積する）。`Σ_dim(g)` との乗算（`exp(y) ⊙ Σ_dim(g)`）・`g` からの
/// 減算も f64 のまま行い、最終書き出しで 1 回だけ `f32` へ downcast
/// する（縮約値を先に `f32` へ戻すと、有限の `f32` 入力でも乗算結果が
/// overflow しうるため）。
fn log_softmax_vjp_along(
    out_value: &Tensor<f32>,
    upstream: &Tensor<f32>,
    axis: usize,
) -> Tensor<f32> {
    let shape = out_value.shape().to_vec();
    // `softmax_vjp_along` 直上と同じ早期 return（部分積オーバーフロー
    // 回避。`eval::softmax_along` 冒頭のコメント参照）。
    if shape.contains(&0) {
        return build_tensor(Vec::new(), &shape);
    }
    let outer: usize = shape[..axis].iter().product();
    let axis_len = shape[axis];
    let inner: usize = shape[axis + 1..].iter().product();
    let y = dense_vec(out_value);
    let g = dense_vec(upstream);
    let mut out = vec![0f32; y.len()];
    for o in 0..outer {
        for i in 0..inner {
            let mut sum_acc: f64 = 0.0;
            for a in 0..axis_len {
                let idx = (o * axis_len + a) * inner + i;
                sum_acc += g[idx] as f64;
            }
            // `sum_acc`（f64）を乗算前に `f32` へ downcast すると、
            // `exp(y) * sum_g` が有限の `f32` 入力でも overflow しうる
            // （例: `y=[0,0]`・上流勾配 `g=[2e38,2e38]` で正しい入力勾配
            // `[0,0]` が `[-inf,-inf]` になる）。`.claude/rules/
            // coding-rust.md` の f64 アキュムレータ契約に従い、
            // `exp(y)` との乗算・`g` からの減算まで f64 で保持し、
            // 最終書き出しでのみ `f32` へ downcast する。
            for a in 0..axis_len {
                let idx = (o * axis_len + a) * inner + i;
                let d = g[idx] as f64 - (y[idx].exp() as f64) * sum_acc;
                out[idx] = d as f32;
            }
        }
    }
    build_tensor(out, &shape)
}

/// `Op::Cumsum` の VJP 本体: `d_x[i] = Σ_{j>=i} g[j]`（`dim` 方向の
/// 逆順累積和。イシュー #1731）。`eval::cumsum_along` と同じ
/// 「外側（outer）× 走査軸（axis_len）× 内側（inner）」の 3 段走査
/// だが、走査は `dim` の添字降順（末尾から先頭へ）で行う。forward と
/// 同じ f64 アキュムレータ契約（`.claude/rules/coding-rust.md`）で
/// lane ごとに `f64` の和を保持し、各ステップで直前の要素を加算して
/// から、その時点の和を書き出す（末尾要素の勾配は `g[n-1]` そのもの、
/// 以降は逆順に累積していく）。
fn cumsum_vjp_along(upstream: &Tensor<f32>, axis: usize) -> Tensor<f32> {
    let shape = upstream.shape().to_vec();
    // `softmax_vjp_along` と同じ早期 return（部分積オーバーフロー
    // 回避）。
    if shape.contains(&0) {
        return build_tensor(Vec::new(), &shape);
    }
    let outer: usize = shape[..axis].iter().product();
    let axis_len = shape[axis];
    let inner: usize = shape[axis + 1..].iter().product();
    let g = dense_vec(upstream);
    let mut out = vec![0f32; g.len()];
    for o in 0..outer {
        for i in 0..inner {
            let mut acc: f64 = 0.0;
            // `dim` の添字降順（`axis_len - 1` から `0` へ）に走査し、
            // `out[a] = Σ_{j>=a} g[j]` を逐次構築する。
            for a in (0..axis_len).rev() {
                let idx = (o * axis_len + a) * inner + i;
                acc += g[idx] as f64;
                out[idx] = acc as f32;
            }
        }
    }
    build_tensor(out, &shape)
}

/// `Op::Cumprod` の VJP 本体（イシュー #1731）。除算を用いない
/// 厳密形 `d_x[a] = L[a] · S[a]` を計算する:
/// - `L[a] = Π_{k<a} x[k]`（排他的 prefix 積。`L[0] = 1`）。先頭から
///   末尾へ順に累積する。
/// - `S[axis_len-1] = g[axis_len-1]`・
///   `S[a] = g[a] + x[a+1]·S[a+1]`（後ろ向き Horner 型再帰）。末尾から
///   先頭へ逆順に累積する。
///
/// `S` は `dim` の降順、`L`（および最終的な `out = L · S`）は昇順で
/// 走査する必要があるため、まず `S` を全 `a` について 1 パスで
/// 計算してから（lane あたり `axis_len` 個のアキュムレータを一時保持）、
/// 2 パス目で `L` を累積しながら `out[a] = L[a] · S[a]` を書き出す
/// （`axis_len` は参照実装が想定する規模〈小〜中程度の軸長〉では
/// 許容できる追加メモリ）。零要素が 0 個・1 個・2 個以上のいずれの
/// 場合でも同一式で厳密に成り立つ（PyTorch の「零なしなら除算形」
/// 高速経路は意図的に不採用。計画立案時に中央差分との数値突合で
/// 0/1/2 零ケースを確認済み）。O(axis_len) per lane。
///
/// **中間積のオーバーフロー対策（codex-review 指摘・PR #1819。
/// [`WideFloat`] 導入）**: `L`／`S` の各アキュムレータは素朴な `f64`
/// ではなく [`WideFloat`]（仮数・指数分離の拡張レンジ浮動小数点数）
/// で保持する。当初 `f64` アキュムレータで実装していたところ、軸上に
/// 極端に大きい要素（例: `2^100`）が連続する区間で `S`（Horner 型
/// 再帰）の中間値が `f64` 表現域（約 `1.8e308`）を超えて `inf` に
/// 発散し、その後段で小さい要素（例: `2^-100`）を掛けても `inf *
/// 有限非零 = inf` のため復元できず、真に有限であるはずの勾配
/// （`入力 = [0] + [2^-100]×11 + [2^100]×11` の `dx[0]` 等）が誤って
/// `inf`／`NaN` になる反例が見つかった（当初の「乗算の一方が厳密に
/// `0.0` の場合のみ明示的に `0.0` を返す」零遮断ガードは、この
/// 「suffix 積自体が発散する」ケースを救えなかった）。[`WideFloat`]
/// は仮数を常に `[0.5, 1.0)` へ正規化し指数を `i64`（`checked_add`／
/// `checked_sub` による加算・減算・比較）のみで扱うため、真の数学的
/// な値が有限である限り乗算・加算いずれの
/// 中間結果も表現域を超えない。最終的に [`WideFloat::to_f64`] で
/// 1 回だけ実数値へ変換し `f32` へ downcast する段階でのみ、真に
/// 表現域外（`f32` にも `f64` にも収まらない）の場合に限り `inf` を
/// 返す——これは正しい IEEE 754 の意味論であり、本関数が解消する
/// 「中間発散による誤 `inf`／`NaN`」とは区別される。零要素の遮断
/// （`L[a] = 0` 以降の勾配が厳密に `0.0`）も、[`WideFloat::mul`] が
/// 零オペランドを明示的に検査して常に厳密な `WideFloat::ZERO` を
/// 返す設計により、特別扱いのガード無しで自動的に成り立つ
/// （零 × 有限は `f64` 表現域内かどうかによらず厳密に零）。
fn cumprod_vjp_along(input: &Tensor<f32>, upstream: &Tensor<f32>, axis: usize) -> Tensor<f32> {
    let shape = input.shape().to_vec();
    if shape.contains(&0) {
        return build_tensor(Vec::new(), &shape);
    }
    let outer: usize = shape[..axis].iter().product();
    let axis_len = shape[axis];
    let inner: usize = shape[axis + 1..].iter().product();
    let x = dense_vec(input);
    let g = dense_vec(upstream);
    let mut out = vec![0f32; x.len()];
    let mut s_values = vec![WideFloat::ZERO; axis_len];
    for o in 0..outer {
        for i in 0..inner {
            // 1 パス目: `S[a]` を末尾（`axis_len - 1`）から先頭（`0`）
            // へ逆順に構築する（[`WideFloat`] アキュムレータ。零遮断・
            // オーバーフロー耐性はいずれも `WideFloat::mul`／`add` の
            // 内部実装が担保する）。
            let mut s_running = WideFloat::ZERO;
            for a in (0..axis_len).rev() {
                let idx = (o * axis_len + a) * inner + i;
                s_running = if a + 1 == axis_len {
                    WideFloat::from_f64(g[idx] as f64)
                } else {
                    let idx_next = (o * axis_len + (a + 1)) * inner + i;
                    let x_next = WideFloat::from_f64(x[idx_next] as f64);
                    let term = x_next.mul(s_running);
                    WideFloat::from_f64(g[idx] as f64).add(term)
                };
                s_values[a] = s_running;
            }
            // 2 パス目: `L[a]`（排他的 prefix 積）を先頭から末尾へ順に
            // 累積しながら `out[a] = L[a] · S[a]` を書き出す。
            let mut l_acc = WideFloat::from_f64(1.0);
            for (a, &s_a) in s_values.iter().enumerate() {
                let idx = (o * axis_len + a) * inner + i;
                out[idx] = l_acc.mul(s_a).to_f64() as f32;
                let x_a = WideFloat::from_f64(x[idx] as f64);
                l_acc = l_acc.mul(x_a);
            }
        }
    }
    build_tensor(out, &shape)
}

/// 拡張レンジ浮動小数点数（仮数 `mantissa: f64`〈絶対値 `[0.5, 1.0)`
/// または厳密ゼロ〉・指数 `exponent: i64` の組で `value = mantissa ·
/// 2^exponent` を表す）。[`cumprod_vjp_along`] の `S`（Horner 型後方
/// 再帰）・`L`（排他的 prefix 積）専用のオーバーフロー安全アキュムレ
/// ータ（codex-review 指摘・PR #1819）。詳細な動機は
/// [`cumprod_vjp_along`] doc の「中間積のオーバーフロー対策」節を
/// 参照。`Copy` な軽量値型（`f64` 1 個 + `i64` 1 個）で、`cumprod_vjp_
/// along` の 2 パス走査（`O(axis_len)` per lane）へそのまま組み込める。
///
/// **指数型を `i32` ではなく `i64` にする理由（codex-review 追加指摘・
/// PR #1819）**: `i32` は最大 `2^31 - 1 ≈ 2.15e9` までしか表現できず、
/// 軸長 `axis_len` に上限検査がない状態で `x = [0] + [2^127] × N`
/// （`N ≈ 17,000,000`）のような入力を与えると、suffix 積の指数が
/// `127 * N ≈ 2.16e9` に達し `i32` の表現域を超える（overflow checks
/// 有効時は `checked_add` なしの素朴な `+` が `debug` ビルドで
/// panic、無効時は指数が折り返して誤った勾配を返す）。`i64` の表現域
/// は `±9.22e18` あり、1 要素あたりの指数変化量の理論上限（`f64` の
/// 正規化数・非正規化数を合わせた全表現域は `[-1074, 1023]`。実際に
/// `WideFloat` の `mantissa` は正規化済みで概ね指数 1 個あたり
/// `|Δexponent| <= 1074` に収まる）を `axis_len` 倍しても
/// `axis_len * 1074 < 2^63`（すなわち `axis_len < 8.58e15`）が成り立つ
/// 限り `checked_add`／`checked_sub` がオーバーフローしない。
/// `axis_len` が `f32` テンソル 1 本として物理的に確保可能な要素数
/// （メモリ上限で自然に抑えられる）を大きく超えない限りこの条件は
/// 常に成立する。万一 `checked_add`／`checked_sub` がオーバーフロー
/// した場合も、以下の実装は panic せず `i64::MAX`／`i64::MIN` へ
/// 飽和させ、[`WideFloat::to_f64`] がそれを `±inf`／`0.0` へ正しく
/// 変換する（本番経路で決して panic しない）。
///
/// `cumprod_vjp_along` 以外から使われる想定はない（`pub` にしない。
/// `compat-api-scope.md` §5 の公開面拡張手続きの対象外に保つ）。
#[derive(Clone, Copy, Debug)]
struct WideFloat {
    mantissa: f64,
    exponent: i64,
}

impl WideFloat {
    const ZERO: WideFloat = WideFloat {
        mantissa: 0.0,
        exponent: 0,
    };

    /// `f64` から構築する。有限の非零値は [`frexp`] で `[0.5, 1.0)`
    /// へ正規化し、`0.0`・非有限値（`cumprod_vjp_along` の入力契約上
    /// 想定していないが、伝播のみ安全に行えるよう保険で扱う）は
    /// `mantissa` にそのまま保持する（`is_non_finite`／`is_zero` が
    /// この 2 状態を区別する）。
    fn from_f64(v: f64) -> WideFloat {
        if v == 0.0 || !v.is_finite() {
            return WideFloat {
                mantissa: v,
                exponent: 0,
            };
        }
        // `frexp` は `f64` の表現域内に収まる指数（`i32` で十分表現
        // できる）しか返さないため、ここでの `i64` への拡大変換は
        // 常に無損失（`as i64` で桁あふれしない）。
        let (mantissa, exponent) = frexp(v);
        WideFloat {
            mantissa,
            exponent: exponent as i64,
        }
    }

    fn is_zero(&self) -> bool {
        self.mantissa == 0.0
    }

    /// `NaN`／`inf` を保持しているかどうか。保持していれば `mantissa`
    /// が生のスカラー値そのものであり `exponent` は意味を持たない
    /// （[`Self::from_f64`] の非有限分岐参照）。
    fn is_non_finite(&self) -> bool {
        !self.mantissa.is_finite()
    }

    /// `mantissa · 2^exponent` を実数値へ変換する（1 回だけの最終
    /// downcast 対象）。真の値が `f64` の表現域を超える場合は IEEE 754
    /// の意味論どおり `inf`（極端に絶対値が小さい場合は `0.0`）を
    /// 返す——これは [`WideFloat`] が防ぐ「中間発散による誤 `inf`」
    /// とは異なり、最終結果自体が真に表現域外である場合の正しい挙動
    /// である。
    fn to_f64(self) -> f64 {
        if self.is_non_finite() {
            return self.mantissa;
        }
        if self.is_zero() {
            return 0.0;
        }
        // `exponent`（`i64`）は `mul`／`add`／`normalize` の
        // `checked_add`／`checked_sub` が飽和させた `i64::MAX`／
        // `i64::MIN` を保持している可能性がある。`f64` の全表現域は
        // 2 進指数で `[-1074, 1023]` に収まるため、それを大きく超える
        // 指数（マージンを見て `±1100` の外）は「真の値が `f64` の
        // 表現域を天文学的規模で超えている」ことを意味し、IEEE 754
        // の意味論どおり `±inf`（下回る場合は `0.0`）へ飽和させる。
        // `exponent as i32` へキャストするのは、この範囲チェックで
        // `[-1100, 1100]` に収まることを確認した後のみであり、桁あふれ
        // しない。
        if self.exponent > 1100 {
            return if self.mantissa > 0.0 {
                f64::INFINITY
            } else {
                f64::NEG_INFINITY
            };
        }
        if self.exponent < -1100 {
            return if self.mantissa > 0.0 { 0.0 } else { -0.0 };
        }
        ldexp(self.mantissa, self.exponent as i32)
    }

    fn mul(self, other: WideFloat) -> WideFloat {
        if self.is_non_finite() || other.is_non_finite() {
            return WideFloat::from_f64(self.to_f64() * other.to_f64());
        }
        // 零オペランドは厳密に `WideFloat::ZERO` を返す（相手がどれほど
        // 大きな指数を保持していても、`0 * 有限 = 0` を表現域と無関係に
        // 厳密に成り立たせる。旧実装の零遮断ガードが担っていた役割を
        // ここで代替する）。
        if self.is_zero() || other.is_zero() {
            return WideFloat::ZERO;
        }
        // 仮数の絶対値はいずれも `[0.5, 1.0)` のため積は `(0.25, 1.0)`
        // に収まり、この乗算自体が `f64` の表現域を超えることはない。
        // 指数の加算は `checked_add`（`i64`）で行い、万一オーバー
        // フローしても panic せず `i64::MAX`／`i64::MIN` へ飽和させる
        // （型の doc comment に根拠を記載。`to_f64` がこれを正しく
        // `±inf`／`0.0` へ変換する）。
        let mantissa = self.mantissa * other.mantissa;
        let exponent = match self.exponent.checked_add(other.exponent) {
            Some(e) => e,
            None => {
                if self.exponent > 0 {
                    i64::MAX
                } else {
                    i64::MIN
                }
            }
        };
        normalize(mantissa, exponent)
    }

    fn add(self, other: WideFloat) -> WideFloat {
        if self.is_non_finite() || other.is_non_finite() {
            return WideFloat::from_f64(self.to_f64() + other.to_f64());
        }
        if self.is_zero() {
            return other;
        }
        if other.is_zero() {
            return self;
        }
        let (hi, lo) = if self.exponent >= other.exponent {
            (self, other)
        } else {
            (other, self)
        };
        // 指数差の計算も `checked_sub`（`i64`）で行い、万一
        // オーバーフロー（`hi`／`lo` いずれかが `mul` の飽和で
        // `i64::MAX`／`i64::MIN` を保持している場合等）しても panic
        // しない。`hi.exponent >= lo.exponent` のため理論上の差は
        // `<= 0` だが、飽和値どうしの差は `i64` の表現域自体を超え
        // うるためガードする（オーバーフロー時は「差が極端に大きい」
        // 側〈`i64::MIN`〉へ倒せば、直後の `< -1100` 分岐で安全に
        // `0.0` 扱いになる）。
        let diff: i64 = lo.exponent.checked_sub(hi.exponent).unwrap_or(i64::MIN); // <= 0（オーバーフロー時は i64::MIN）
        // `diff` が極端に小さい（指数差が大きすぎる）場合、`lo` の
        // 寄与は `hi` に対して丸めで消える桁のため `0.0` として扱う
        // （2 進指数差 1100 は `f64` の全表現域〈約 2^-1074〜2^1024〉
        // より広く、早期 return は最適化であって正しさの条件では
        // ない。`pow2` 自体も極端に負の指数では正しく underflow
        // して `0.0` を返すため、この分岐がなくても結果は変わらない）。
        // `diff < -1100` を満たさない場合は `i32` の表現域内に収まる
        // ことが保証されるため `as i32` は無損失。
        let scaled_lo = if diff < -1100 {
            0.0
        } else {
            lo.mantissa * pow2(diff as i32)
        };
        // `hi.mantissa` の絶対値は `[0.5, 1.0)`、`scaled_lo` の絶対値は
        // `<= 1.0`（`diff <= 0` のため）なので、和は `(-2.0, 2.0)` に
        // 収まり `f64` の表現域を超えない。
        let mantissa = hi.mantissa + scaled_lo;
        normalize(mantissa, hi.exponent)
    }
}

/// `mantissa`（絶対値が `2.0` 未満の任意の有限値。ゼロを含む）を
/// `[0.5, 1.0)` へ正規化しつつ `exponent` を調整する
/// （[`WideFloat::mul`]／[`WideFloat::add`] の後処理として使う）。
fn normalize(mantissa: f64, exponent: i64) -> WideFloat {
    if mantissa == 0.0 {
        return WideFloat::ZERO;
    }
    let (m, e) = frexp(mantissa);
    // `e`（`frexp` が返す小さな補正量。`mantissa` の絶対値は呼び出し元
    // の契約上 `(-2.0, 2.0)` のため `e ∈ {-1, 0, 1}` 相当）を `i64` へ
    // 拡大してから `checked_add` する。`exponent` が `mul`／`add` の
    // 飽和により既に `i64::MAX`／`i64::MIN` の場合でも panic せず、
    // 同じ側へ飽和させる（`to_f64` がこれを正しく `±inf`／`0.0` へ
    // 変換する）。
    let widened =
        exponent
            .checked_add(e as i64)
            .unwrap_or(if e >= 0 { i64::MAX } else { i64::MIN });
    WideFloat {
        mantissa: m,
        exponent: widened,
    }
}

/// `value`（有限・非零）を `value == mantissa * 2^exponent`
/// （`mantissa` の絶対値は `[0.5, 1.0)`）へ分解する（C 標準ライブラリ
/// `frexp` 相当）。Rust 標準ライブラリに同等 API がなく、`libm` は
/// 許容依存 9 区分（`deps-policy.md`）に含まれないため、IEEE 754
/// binary64 のビットレイアウトから手動で計算する（`unsafe` 不使用。
/// `to_bits`/`from_bits` はいずれも安全 API）。
fn frexp(value: f64) -> (f64, i32) {
    debug_assert!(value.is_finite() && value != 0.0);
    const MANTISSA_MASK: u64 = 0x000f_ffff_ffff_ffff;
    const SIGN_MASK: u64 = 0x8000_0000_0000_0000;
    let bits = value.to_bits();
    let sign = bits & SIGN_MASK;
    let biased_exp = ((bits >> 52) & 0x7ff) as i32;
    let mantissa_bits = bits & MANTISSA_MASK;

    if biased_exp == 0 {
        if mantissa_bits == 0 {
            // `value == ±0.0`（呼び出し前提で除外済みだが、万一到達
            // しても安全に `0.0` を返す）。
            return (0.0, 0);
        }
        // 非正規化数（絶対値 `< 2^-1022`）: 仮数ビットの先頭ゼロ数
        // から正規化に必要なシフト量を直接求める（ループ不要）。
        // `mantissa_bits` は 52bit フィールドに収まる非零値のため
        // `leading_zeros()`（64bit 幅基準）は `12..=63`。
        let leading_zeros = mantissa_bits.leading_zeros();
        let shift = leading_zeros - 11; // 1..=52
        let normalized_mantissa_bits = (mantissa_bits << shift) & MANTISSA_MASK;
        let exponent = -1010 - leading_zeros as i32;
        let new_bits = sign | (1022u64 << 52) | normalized_mantissa_bits;
        return (f64::from_bits(new_bits), exponent);
    }

    // 正規化数: 指数フィールドを `1022`（`value ∈ [0.5, 1.0)` に対応
    // する biased exponent）へ置き換えるだけで、仮数ビットはそのまま
    // 使える。
    let exponent = biased_exp - 1022;
    let new_bits = sign | (1022u64 << 52) | mantissa_bits;
    (f64::from_bits(new_bits), exponent)
}

/// `2^exponent` を計算する。`exponent ∈ [-1022, 1023]`（`f64` の
/// 正規化数として厳密に表現できる範囲）は `f64::from_bits` による
/// ビット構成で誤差なく求める。範囲外（オーバーフロー・アンダー
/// フロー）は 2 分割した半分ずつの掛け算で段階的に構成し、IEEE 754
/// の飽和（`inf`）・下限丸め（`0.0`）へ正しく帰着させる（[`ldexp`]
/// の内部で呼ばれる。範囲外の結果はどのみち [`WideFloat`] が防ぐ
/// 「中間発散」ではなく最終値自体の真の表現域外を意味するため実害
/// はない）。
fn pow2(exponent: i32) -> f64 {
    if (-1022..=1023).contains(&exponent) {
        let biased = (exponent + 1023) as u64;
        return f64::from_bits(biased << 52);
    }
    let half = exponent / 2;
    pow2_saturating(half) * pow2_saturating(exponent - half)
}

/// [`pow2`] の範囲外分岐専用ヘルパ。ビット構成できない指数
/// （`(-1022..=1023)` の外）を、`inf`／`0.0` への飽和へ直接倒す
/// （半分に分割してもなお範囲外になりうる極端な `exponent` に対する
/// 終端条件）。
fn pow2_saturating(exponent: i32) -> f64 {
    if (-1022..=1023).contains(&exponent) {
        let biased = (exponent + 1023) as u64;
        return f64::from_bits(biased << 52);
    }
    if exponent > 1023 { f64::INFINITY } else { 0.0 }
}

/// `mantissa * 2^exponent` を計算する（C 標準ライブラリ `ldexp`
/// 相当）。指数を半分ずつ 2 回に分けて乗算することで、`f64` 表現域
/// の境界付近（`exponent` が `f64::MAX_EXP` 近傍）でも早期の
/// オーバーフロー・アンダーフローを避け、真に表現域内の値をより正確
/// に復元する（[`WideFloat::to_f64`] からのみ呼ばれる最終変換）。
fn ldexp(mantissa: f64, exponent: i32) -> f64 {
    let e1 = exponent / 2;
    let e2 = exponent - e1;
    mantissa * pow2(e1) * pow2(e2)
}

/// `Sum` の VJP: 出力側勾配 `g` を入力 shape へブロードキャストして
/// 複製する（`sum` の逆演算は複製、`reduce_out_shape` が縮約軸を
/// 除去済み〈keepdim なし〉のため、`dim: Some(axis)` はいったん
/// size-1 軸を挿入してから `broadcast_to` する）。
fn unreduce_broadcast(g: &Tensor<f32>, input_shape: &[usize], dim: Option<usize>) -> Tensor<f32> {
    match dim {
        None => {
            let value = dense_vec(g).first().copied().unwrap_or(0.0);
            match Tensor::full(input_shape, value) {
                Ok(t) => t,
                Err(_) => {
                    debug_assert!(
                        false,
                        "unreduce_broadcast: dim=None の full() 構築が失敗した（契約違反）"
                    );
                    g.clone()
                }
            }
        }
        Some(axis) => {
            let mut inserted_shape = g.shape().to_vec();
            inserted_shape.insert(axis, 1);
            let reshaped = match g.contiguous().reshape(&inserted_shape) {
                Ok(t) => t,
                Err(_) => {
                    debug_assert!(
                        false,
                        "unreduce_broadcast: reduce_out_shape 逆算の reshape が失敗した（契約違反）"
                    );
                    return g.clone();
                }
            };
            match reshaped.broadcast_to(input_shape) {
                Ok(t) => t.contiguous(),
                Err(_) => {
                    debug_assert!(
                        false,
                        "unreduce_broadcast: 挿入軸からの broadcast_to が失敗した（契約違反）"
                    );
                    reshaped
                }
            }
        }
    }
}

/// `Mean{input, dim}` の VJP（イシュー #1719）。`d(mean)/d(x_i) = 1/n`
/// （`n` は forward〈`Var::mean`〉と同じ縮約対象要素数）で、連鎖律
/// により上流勾配 `g` の各要素を `n` で割ってから
/// [`unreduce_broadcast`]（`Sum` の VJP＝複製）へ渡せば良い（`Sum` の
/// VJP を `1/n` でスケールしたものが `Mean` の VJP と一致するため、
/// 別実装を持たず合成する）。`n == 0` は forward（`Var::mean`）が
/// 事前に `AutodiffError::InvalidArgument` で拒否しているため
/// backward 側へは到達しない契約——到達した場合は 0 除算
/// （`v / 0 == NaN`）を避け `g` を無加工のまま複製する安全側
/// フォールバックとする（`unreduce_broadcast` 自身の契約違反
/// フォールバック方針と同じ）。
fn mean_vjp(g: &Tensor<f32>, input_shape: &[usize], dim: Option<usize>) -> Tensor<f32> {
    let n: usize = match dim {
        None => input_shape.iter().product(),
        Some(axis) => input_shape.get(axis).copied().unwrap_or(0),
    };
    if n == 0 {
        debug_assert!(false, "mean_vjp: n が 0（契約違反）");
        return unreduce_broadcast(g, input_shape, dim);
    }
    let scaled_data: Vec<f32> = dense_vec(g).into_iter().map(|v| v / n as f32).collect();
    let scaled = build_tensor(scaled_data, g.shape());
    unreduce_broadcast(&scaled, input_shape, dim)
}

/// `Max` の VJP: 出力側勾配 `g` を、縮約軸に沿った最大値の位置のみへ
/// 伝播する。**同値タイは「最初に現れる最大要素 1 箇所のみ」へ伝播
/// する**（PyTorch `amax` の均等分配とは異なる、決定的な選択。
/// PoC-v2-2 のビット一致決定性方針・`train_repro` と整合させるための
/// 設計判断）。`out_value` は forward 記録済みの縮約後最大値で、走査中
/// に現れる要素と exact 一致するかで argmax 位置を判定する（同一デー
/// タ・同一 reduction 経路のため bit 一致する）。
///
/// Issue #224（先勝ち挙動の再確認。compat 層〈REQ-9〉実装時に要再確認
/// としていた事項）の結論: **本挙動を維持する（変更なし）**。
///
/// **イシュー #1718（amax／amin 勾配分配方式の確定）で最終確定**:
/// `Var::max`／`min`／`max_dims`（`max(dim)`／`min(dim)` 族の意味論）は
/// 本先勝ち決定的方式を**維持**する。根拠は 2 点——(a) 本方式は
/// crates.io 公開全版（v0.3.0〜）で出荷済みの勾配値であり、均等分配へ
/// 変更すると勾配値そのものが変わる破壊的変更になる、(b) PyTorch
/// 自身も `torch.max(input, dim)`／`min(dim)`（添字を返す族）は均等
/// 分配ではなく返した添字 1 箇所のみへ勾配を伝播する仕様であり、本
/// リポの `argmax`／`argmin`「タイは最初の添字」契約（`eval::
/// arg_extremum`）と内部整合する。PyTorch `torch.amax`／`amin`
/// （添字を返さない縮約）相当の均等分配 API を追加する場合は、本
/// ヘルパーを差し替えず独立の `Op`／VJP として実装する方針とした
/// （`docs/autodiff-amax-grad-distribution-decision.md` 参照。未実装・
/// 後続 issue 提案のまま）。
///
/// **`Op::Min` との共有（イシュー #1720）**: 本関数の実体は
/// 「`out_value` と `==` 一致する最初の位置へ `g` を置く」だけで
/// 最大／最小どちらの縮約かに依存しない。そのため実体を
/// [`extremum_first_match_vjp`] へ改称し、本関数は既存呼び出し元
/// （テスト・`Op::Max` アーム）の名前を変えないための薄いラッパーと
/// して残す（#1718 の確定により、このヘルパーは今後も `Max`／`Min`
/// 共有のまま先勝ち決定的方式に固定される）。
fn max_vjp(
    input: &Tensor<f32>,
    dim: Option<usize>,
    out_value: &Tensor<f32>,
    g: &Tensor<f32>,
) -> Tensor<f32> {
    extremum_first_match_vjp(input, dim, out_value, g)
}

/// [`max_vjp`] doc 参照。`out_value`（縮約後の最大値または最小値）と
/// `==` 一致する `dim` 軸上の最初の要素へ `g` を伝播する、最大／最小
/// 非依存の VJP 実体（イシュー #1720 で `max_vjp` から改称・共有化）。
fn extremum_first_match_vjp(
    input: &Tensor<f32>,
    dim: Option<usize>,
    out_value: &Tensor<f32>,
    g: &Tensor<f32>,
) -> Tensor<f32> {
    let in_shape = input.shape().to_vec();
    let in_data = dense_vec(input);
    let g_data = dense_vec(g);
    let out_data = dense_vec(out_value);
    let mut grad = vec![0f32; in_data.len()];
    match dim {
        None => {
            if let (Some(target), Some(gv)) = (out_data.first(), g_data.first())
                && let Some(idx) = in_data.iter().position(|&v| v == *target)
            {
                grad[idx] = *gv;
            }
        }
        Some(axis) => {
            let outer: usize = in_shape[..axis].iter().product();
            let axis_len = in_shape[axis];
            let inner: usize = in_shape[axis + 1..].iter().product();
            for o in 0..outer {
                for i in 0..inner {
                    let out_idx = o * inner + i;
                    // `out_data`/`g_data` の要素数は `reduce_out_shape`
                    // の契約上 `outer * inner` と一致するはずだが、
                    // 本ファイルの他ヘルパー（`transpose2d`／
                    // `unreduce_broadcast` 等）と同様、契約違反時に
                    // release ビルドで境界外アクセス panic させず
                    // `debug_assert!` で検知しつつ安全側（当該要素の
                    // 勾配は 0 のまま）へフォールバックする
                    // （coding-rust.md「本番経路で unwrap/expect を
                    // 使わない」方針の趣旨に揃える）。
                    let (Some(&target), Some(&g_val)) =
                        (out_data.get(out_idx), g_data.get(out_idx))
                    else {
                        debug_assert!(
                            false,
                            "max_vjp: out_value/g の要素数が reduce_out_shape の想定と不一致（契約違反）"
                        );
                        continue;
                    };
                    for a in 0..axis_len {
                        let src = (o * axis_len + a) * inner + i;
                        match in_data.get(src) {
                            Some(&v) if v == target => {
                                grad[src] = g_val;
                                break;
                            }
                            Some(_) => {}
                            None => {
                                debug_assert!(
                                    false,
                                    "max_vjp: input の要素数が in_shape と不一致（契約違反）"
                                );
                            }
                        }
                    }
                }
            }
        }
    }
    build_tensor(grad, &in_shape)
}

/// `Op::Var`（`Var::var`）の VJP: `da_i = g · 2(x_i − mean) / (n −
/// correction)`（イシュー #1723）。`mean` は `out_value`（forward の
/// `f32` downcast 済み記録値）を再利用せず `input` から改めて `f64` で
/// 計算する——`matrix_norm_vjp`（`eval/linalg.rs`）と同じ理由（forward
/// の丸め誤差を backward へ持ち込まない。`.claude/rules/coding-rust.md`
/// の内部精度契約）。`softmax_vjp_along` と同じ「外側（outer）×走査軸
/// （axis_len）×内側（inner）」の 3 段走査（`dim=None` は
/// `outer=inner=1`）。`2(x_i − mean)` の計算・`g` との乗算・
/// `(n − correction)` による除算はすべて `f64` のまま保持し、最後に
/// 1 回だけ `f32` へ downcast する（縮約値を先に `f32` へ戻すと、
/// 有限の `f32` 入力でも乗算結果が overflow しうるため）。
fn var_vjp(
    input: &Tensor<f32>,
    dim: Option<usize>,
    correction: usize,
    g: &Tensor<f32>,
) -> Tensor<f32> {
    let shape = input.shape().to_vec();
    let (outer, axis_len, inner) = match dim {
        None => (1usize, input.numel(), 1usize),
        Some(axis) => (
            shape[..axis].iter().product(),
            shape[axis],
            shape[axis + 1..].iter().product(),
        ),
    };
    let data = dense_vec(input);
    let g_data = dense_vec(g);
    let n = axis_len as f64;
    let denom = n - correction as f64;
    let mut out = vec![0f32; data.len()];
    for o in 0..outer {
        for i in 0..inner {
            let mut mean_acc = 0.0f64;
            for a in 0..axis_len {
                let src = (o * axis_len + a) * inner + i;
                mean_acc += data[src] as f64;
            }
            let mean = mean_acc / n;
            let out_idx = o * inner + i;
            let g_val = g_data.get(out_idx).copied().unwrap_or_else(|| {
                debug_assert!(
                    false,
                    "var_vjp: g の要素数が reduce_out_shape の想定と不一致（契約違反）"
                );
                0.0
            }) as f64;
            for a in 0..axis_len {
                let src = (o * axis_len + a) * inner + i;
                let d = data[src] as f64 - mean;
                out[src] = (g_val * 2.0 * d / denom) as f32;
            }
        }
    }
    build_tensor(out, &shape)
}

/// `Op::Std`（`Var::std`）の VJP: `da_i = g · (x_i − mean) / ((n −
/// correction) · std)`（イシュー #1723 レビュー是正）。[`var_vjp`] と
/// 同じ「`input` から改めて `f64` で計算する」方針だが、`std`
/// （forward の `f32` downcast 済み記録値 `out_value`）も再利用せず
/// `mean`／`sq_acc` から `f64` で改めて `sqrt` する——`var_vjp` の
/// `2(x_i − mean) / denom` を `f32` へ downcast してから `0.5 / std`
/// （`Var::sqrt` の VJP）を掛ける合成では、`2(x_i − mean) / denom`
/// 自体が `f32` の範囲を超えて overflow しうる（`std` は最終的に
/// 有限でも、途中の `var` の勾配項は無限大になりうるため。
/// codex-review P2 指摘）。本関数は `(x_i − mean)` と `denom · std` の
/// 除算を最後まで `f64` に保つことでこれを回避する。`std == 0`
/// （縮約対象が全て同値の定数列）の要素は PyTorch `std_backward`
/// （`FunctionsManual.cpp`）の
/// `masked_fill_(result == 0, 0)` と同じ規約でゼロ勾配へ明示的に
/// マスクする（`0.0 / 0.0` の `NaN` を伝播させない。判定は forward
/// が実際に返す `f32` 丸め後の値と同じ丸めで行う）。これは
/// `Var::var(..).sqrt()`（新規 `Op` を追加しない合成）を使った場合の
/// 挙動——`Var::sqrt` の `y == 0` 規約により `0.0 / 0.0 = NaN` を
/// 返す——とは意図的に異なる（`Var::std` の doc「数値規約」参照。
/// codex-review 指摘。PR #1826 レビュー是正）。
fn std_vjp(
    input: &Tensor<f32>,
    dim: Option<usize>,
    correction: usize,
    g: &Tensor<f32>,
) -> Tensor<f32> {
    let shape = input.shape().to_vec();
    let (outer, axis_len, inner) = match dim {
        None => (1usize, input.numel(), 1usize),
        Some(axis) => (
            shape[..axis].iter().product(),
            shape[axis],
            shape[axis + 1..].iter().product(),
        ),
    };
    let data = dense_vec(input);
    let g_data = dense_vec(g);
    let n = axis_len as f64;
    let denom = n - correction as f64;
    let mut out = vec![0f32; data.len()];
    for o in 0..outer {
        for i in 0..inner {
            let mut mean_acc = 0.0f64;
            for a in 0..axis_len {
                let src = (o * axis_len + a) * inner + i;
                mean_acc += data[src] as f64;
            }
            let mean = mean_acc / n;
            let mut sq_acc = 0.0f64;
            for a in 0..axis_len {
                let src = (o * axis_len + a) * inner + i;
                let d = data[src] as f64 - mean;
                sq_acc += d * d;
            }
            let std = (sq_acc / denom).sqrt();
            // PyTorch `std_backward`（FunctionsManual.cpp）は forward の
            // `f32` 出力 `result` が 0 の要素を `masked_fill_(result == 0,
            // 0)` で明示的にゼロ勾配へ落としてから `var_backward` へ渡す
            // （`0 / 0` の NaN 伝播を避ける）。ここでは `std` を 1 回
            // `f32` へ downcast した値（forward が実際に返す `result` と
            // 同じ丸め）で判定し、同じ規約に揃える（codex-review 指摘。
            // PR #1826 レビュー是正）。
            let std_is_zero = (std as f32) == 0.0;
            let out_idx = o * inner + i;
            let g_val = g_data.get(out_idx).copied().unwrap_or_else(|| {
                debug_assert!(
                    false,
                    "std_vjp: g の要素数が reduce_out_shape の想定と不一致（契約違反）"
                );
                0.0
            }) as f64;
            let denom_std = denom * std;
            for a in 0..axis_len {
                let src = (o * axis_len + a) * inner + i;
                if std_is_zero {
                    out[src] = 0.0;
                    continue;
                }
                let d = data[src] as f64 - mean;
                out[src] = (g_val * d / denom_std) as f32;
            }
        }
    }
    build_tensor(out, &shape)
}

/// `Op::VectorNorm`（`Var::norm_l1`／`norm_l2`）の VJP（イシュー
/// #1723）。[`var_vjp`] と同じ 3 段走査。
///
/// - `L1`: `da_i = g · sign(x_i)`（`sign(0) = 0`。`ScalarUnaryOp::Abs`
///   の劣勾配規約〈`eval::scalar`〉と同じ）。総和のみのため `f64`
///   縮約は不要。
/// - `L2`: `‖x‖` を `input` から改めて `f64` で再計算し
///   （`matrix_norm_vjp` の Fro ノルムと同じ理由）、`da_i = g · x_i /
///   ‖x‖`。`‖x‖ == 0` の出力要素は勾配 0、`‖x‖` が `NaN`（入力に `NaN`
///   を含む）の場合は明示的に `NaN` を伝播する（`matrix_norm_vjp` の
///   NaN／ゼロ分岐構造を踏襲。`norm > 0.0` は NaN に対して常に偽になる
///   ため、この分岐がないと NaN が黙って 0 として扱われ数値異常が
///   隠れる）。
fn vector_norm_vjp(
    input: &Tensor<f32>,
    ord: VectorNormOrd,
    dim: Option<usize>,
    g: &Tensor<f32>,
) -> Tensor<f32> {
    let shape = input.shape().to_vec();
    let (outer, axis_len, inner) = match dim {
        None => (1usize, input.numel(), 1usize),
        Some(axis) => (
            shape[..axis].iter().product(),
            shape[axis],
            shape[axis + 1..].iter().product(),
        ),
    };
    let data = dense_vec(input);
    let g_data = dense_vec(g);
    let mut out = vec![0f32; data.len()];
    for o in 0..outer {
        for i in 0..inner {
            let out_idx = o * inner + i;
            let g_val = g_data.get(out_idx).copied().unwrap_or_else(|| {
                debug_assert!(
                    false,
                    "vector_norm_vjp: g の要素数が reduce_out_shape の想定と不一致（契約違反）"
                );
                0.0
            });
            match ord {
                VectorNormOrd::L1 => {
                    for a in 0..axis_len {
                        let src = (o * axis_len + a) * inner + i;
                        let v = data[src];
                        let sign = if v > 0.0 {
                            1.0
                        } else if v < 0.0 {
                            -1.0
                        } else {
                            0.0
                        };
                        out[src] = g_val * sign;
                    }
                }
                VectorNormOrd::L2 => {
                    let mut sq_acc = 0.0f64;
                    for a in 0..axis_len {
                        let src = (o * axis_len + a) * inner + i;
                        let v = data[src] as f64;
                        sq_acc += v * v;
                    }
                    let norm = sq_acc.sqrt();
                    for a in 0..axis_len {
                        let src = (o * axis_len + a) * inner + i;
                        let v = data[src] as f64;
                        out[src] = if norm.is_nan() {
                            f32::NAN
                        } else if norm > 0.0 {
                            (g_val as f64 * v / norm) as f32
                        } else {
                            0.0
                        };
                    }
                }
                // `VectorNormOrd` は `#[non_exhaustive]`。`eval::
                // vector_norm_along` と同じ安全側フォールバック
                // （寄与なし。到達しない想定）。
                _ => {}
            }
        }
    }
    build_tensor(out, &shape)
}

/// `MseLoss{pred, target, reduction}` の VJP: `dPred = g · 2(pred −
/// target) / n`（mean）／`g · 2(pred − target)`（sum）、
/// `dTarget = −dPred`（`g` はスカラー上流勾配）。#190 で sum 縮約を
/// 追加（`reduction` 分岐は forward の `eval::mse_loss` と対称）。
/// `n == 0` は mean・sum ともゼロ除算を避け zeros を返す。PoC-v2-2 は
/// `target` 側の勾配計算をスキップしていたが、本実装は数学的に完全な
/// VJP を返す（`target` 側を使うか捨てるかは #18 の勾配蓄積側の責務で
/// あり、ここでは両方提供する）。
fn mse_loss_vjp(
    pred: &Tensor<f32>,
    target: &Tensor<f32>,
    g: &Tensor<f32>,
    reduction: Reduction,
) -> (Tensor<f32>, Tensor<f32>) {
    let shape = pred.shape().to_vec();
    let n = pred.numel();
    if n == 0 {
        let zeros = build_tensor(vec![0f32; 0], &shape);
        return (zeros.clone(), zeros);
    }
    let g_value = dense_vec(g).first().copied().unwrap_or(0.0);
    let pred_data = dense_vec(pred);
    let target_data = dense_vec(target);
    let scale = mse_loss_scale(g_value, n, reduction);
    let dpred_data: Vec<f32> = pred_data
        .iter()
        .zip(target_data.iter())
        .map(|(&p, &t)| scale * (p - t))
        .collect();
    let dtarget_data: Vec<f32> = dpred_data.iter().map(|&v| -v).collect();
    let dpred = build_tensor(dpred_data, &shape);
    let dtarget = build_tensor(dtarget_data, &shape);
    (dpred, dtarget)
}

/// `mse_loss_vjp`（ホスト参照実装）と融合カーネル経路（`vjp()` の
/// `Op::MseLoss` 分岐。イシュー #1045）の双方が使う `scale` 算出の
/// 共有ロジック: `dPred = scale·(pred−target)`（`Mean` は `g·2/n`、
/// `Sum` は `g·2`）。`BackendOps::mse_loss_backward` の呼び出し元が
/// このスケールを事前計算して渡す契約（`backend_ops.rs` doc 参照）
/// であり、フォールバック（`mse_loss_vjp`）と融合カーネル経路とで
/// 同一の数式を 2 か所に別実装しないための切り出し。
fn mse_loss_scale(g_value: f32, n: usize, reduction: Reduction) -> f32 {
    match reduction {
        Reduction::Mean => g_value * 2.0 / n as f32,
        Reduction::Sum => g_value * 2.0,
    }
}

/// `HuberLoss{pred, target, kind, delta, reduction}` の VJP:
/// `dPred = scale · grad_elem(d)`（`d = pred − target`。`grad_elem` は
/// `eval::huber_elem_grad`）、`dTarget = −dPred`（イシュー #1739。
/// `mse_loss_vjp` と同型）。`n == 0` は mean・sum ともゼロ除算を避け
/// zeros を返す。
fn huber_loss_vjp(
    pred: &Tensor<f32>,
    target: &Tensor<f32>,
    g: &Tensor<f32>,
    kind: HuberKind,
    delta: f32,
    reduction: Reduction,
) -> (Tensor<f32>, Tensor<f32>) {
    let shape = pred.shape().to_vec();
    let n = pred.numel();
    if n == 0 {
        let zeros = build_tensor(vec![0f32; 0], &shape);
        return (zeros.clone(), zeros);
    }
    let g_value = dense_vec(g).first().copied().unwrap_or(0.0);
    let pred_data = dense_vec(pred);
    let target_data = dense_vec(target);
    let scale = huber_loss_scale(g_value, n, reduction);
    let dpred_data: Vec<f32> = pred_data
        .iter()
        .zip(target_data.iter())
        .map(|(&p, &t)| scale * eval::huber_elem_grad(p - t, kind, delta))
        .collect();
    let dtarget_data: Vec<f32> = dpred_data.iter().map(|&v| -v).collect();
    let dpred = build_tensor(dpred_data, &shape);
    let dtarget = build_tensor(dtarget_data, &shape);
    (dpred, dtarget)
}

/// `huber_loss_vjp`（ホスト参照実装）と融合カーネル経路（`vjp()` の
/// `Op::HuberLoss` 分岐）の双方が使う `scale` 算出の共有ロジック
/// （`mse_loss_scale` と同型。イシュー #1739）:
/// `dPred = scale·grad_elem(pred−target)`（`Mean` は `g/n`、`Sum` は
/// `g`。`MseLoss` と異なり係数 2 は付かない——`grad_elem` 自体が
/// `d(0.5·d²)/dd = d` を含むため）。
fn huber_loss_scale(g_value: f32, n: usize, reduction: Reduction) -> f32 {
    match reduction {
        Reduction::Mean => g_value / n as f32,
        Reduction::Sum => g_value,
    }
}

/// `BceLoss{input, target, kind, reduction}` のホスト参照 VJP（`ops.
/// bce_loss_backward` が `Unsupported` のときのみ呼ばれる。イシュー
/// #1737）。`eval::bce_elem_grad_input`／`bce_elem_grad_target`
/// （forward 要素式 `eval::bce_elem_loss` と対をなす意味論の正）に
/// `bce_loss_scale` を乗じて `dInput`／`dTarget` を構成する。`n == 0`
/// は `mse_loss_vjp` と同じくゼロ除算を避け zeros を返す。
fn bce_loss_vjp(
    input: &Tensor<f32>,
    target: &Tensor<f32>,
    kind: BceKind,
    g: &Tensor<f32>,
    reduction: Reduction,
) -> (Tensor<f32>, Tensor<f32>) {
    let shape = input.shape().to_vec();
    let n = input.numel();
    if n == 0 {
        let zeros = build_tensor(vec![0f32; 0], &shape);
        return (zeros.clone(), zeros);
    }
    let g_value = dense_vec(g).first().copied().unwrap_or(0.0);
    let input_data = dense_vec(input);
    let target_data = dense_vec(target);
    let scale = bce_loss_scale(g_value, n, reduction);
    let dinput_data: Vec<f32> = input_data
        .iter()
        .zip(target_data.iter())
        .map(|(&p, &y)| scale * eval::bce_elem_grad_input(p, y, kind))
        .collect();
    let dtarget_data: Vec<f32> = input_data
        .iter()
        .zip(target_data.iter())
        .map(|(&p, &y)| scale * eval::bce_elem_grad_target(p, y, kind))
        .collect();
    let dinput = build_tensor(dinput_data, &shape);
    let dtarget = build_tensor(dtarget_data, &shape);
    (dinput, dtarget)
}

/// `mse_loss_vjp`（ホスト参照実装）と融合カーネル経路（`vjp()` の
/// `Op::BceLoss` 分岐。イシュー #1737）の双方が使う `scale` 算出の
/// 共有ロジック: `dInput = scale·bce_elem_grad_input(..)`（`Mean` は
/// `g/n`、`Sum` は `g`。`mse_loss_scale` の `2` 倍係数〈二乗誤差由来〉
/// は BCE には現れないため異なる式）。`BackendOps::bce_loss_backward`
/// の呼び出し元がこのスケールを事前計算して渡す契約
/// （`backend_ops.rs` doc 参照）。
fn bce_loss_scale(g_value: f32, n: usize, reduction: Reduction) -> f32 {
    match reduction {
        Reduction::Mean => g_value / n as f32,
        Reduction::Sum => g_value,
    }
}

/// `CrossEntropyLoss(logits, targets)` の VJP:
/// `d loss / d logits[..., c, ...] = (softmax(logits)[..., c, ...] − 1{c == t}) × g`
/// （`g` はサンプルごとのスカラー係数。`Mean` は `g = upstream / N`、
/// `Sum` は `g = upstream`。`N` はサンプル数 `= targets.numel()`）。
/// `eval::softmax_along` を再利用し、forward（`eval::cross_entropy_loss`
/// の log-sum-exp）と数式の実体を分離しない（`grad.rs` 冒頭 doc）。
/// `targets` は非追跡のため戻り値は `logits` 側の勾配のみ（呼び出し元
/// `vjp()` の `CrossEntropyLoss` 分岐参照）。
fn cross_entropy_loss_vjp(
    logits: &Tensor<f32>,
    targets: &Tensor<i32>,
    class_dim: usize,
    reduction: Reduction,
    upstream: &Tensor<f32>,
) -> Tensor<f32> {
    let shape = logits.shape().to_vec();
    let outer: usize = shape[..class_dim].iter().product();
    let axis_len = shape[class_dim];
    let inner: usize = shape[class_dim + 1..].iter().product();
    let n = outer * inner;

    let softmax = eval::softmax_along(logits, class_dim);
    let mut grad = dense_vec(&softmax);
    let target_data = eval::dense_vec_i32(targets);

    let g_value = dense_vec(upstream).first().copied().unwrap_or(0.0);
    let scale = match reduction {
        Reduction::Mean if n > 0 => g_value / n as f32,
        Reduction::Mean => 0.0,
        Reduction::Sum => g_value,
    };

    for o in 0..outer {
        for i in 0..inner {
            let t = target_data[o * inner + i];
            // forward（`Var::cross_entropy_loss`）が事前検査済みの前提
            // （`0 <= t < axis_len`）。範囲外は契約違反であり
            // `debug_assert!` で検知しつつ onehot 減算をスキップする
            // 安全側フォールバック（`eval::cross_entropy_loss` と同型の
            // 契約違反対応）。
            if t >= 0 && (t as usize) < axis_len {
                let idx = (o * axis_len + t as usize) * inner + i;
                grad[idx] -= 1.0;
            } else {
                debug_assert!(
                    false,
                    "cross_entropy_loss_vjp: target 添字が範囲外（契約違反）"
                );
            }
        }
    }
    let scaled: Vec<f32> = grad.iter().map(|&v| v * scale).collect();
    build_tensor(scaled, &shape)
}

#[cfg(test)]
mod tests {
    //! 受け入れ条件「各演算の解析勾配が数値微分と一致する」の直接検証。
    //!
    //! 各演算について、固定の重みテンソル `s`（forward 出力と同じ
    //! shape）によるスカラー射影 `L(x) = Σ (op(x) ⊙ s)` を定義し、
    //! 解析側（`vjp` 経由）と数値側（中央差分。f64 で集計）を突合する。
    //! `Tape`/`Var` を経由せず `eval.rs` の値計算を直接叩くため、
    //! `Tape::backward`（#18）が未実装でも検証できる。
    //!
    //! **判定基準**: 要素ごとに「相対誤差
    //! `|ad − num| / max(|ad|, |num|, τ)` が 1e-2 以下」または
    //! 「絶対誤差が 1e-3 以下」（`h = 1e-3`・`τ = 1e-4`）。PoC-v2-2 の
    //! 1e-4 は f64 前提（`torch.autograd.gradcheck` と同じ理由で f64
    //! 必須と PoC 自身が明記）だが、本実装は f32 のため中央差分の
    //! 丸め誤差床 `≈ ε_f32 · |L| / h ≈ 1e-4` を踏まえた本イシュー
    //! 新規の grad-check 専用閾値とする（バックエンド間数値一致判定
    //! 〈相対 1e-3 / 絶対 1e-5〉とは別系統）。
    //!
    //! **承認記録（#223・承認済み）**: `CLAUDE.md` Conventions・
    //! `.claude/rules/delegation-impl.md` の「テスト許容誤差の変更は
    //! ユーザー承認必須」規定に基づく本閾値（新規 grad-check 専用
    //! `REL_TOL`/`ABS_TOL`）の承認は完了している。
    //! - 承認者: ユーザー／承認日: 2026-08-09
    //! - 承認記録: <https://github.com/Fandhe-AI/fandhe-ai/issues/223#issuecomment-5230026874>
    //! - 判断材料: 全 grad-check テストの実測誤差マージン採取で、
    //!   最も僅差のケースでも絶対誤差側に約 3.4 倍の余裕
    //!   （実測 `diff ≈ 2.9e-4` に対し `ABS_TOL = 1e-3`）を確認
    //!
    //! 値（`REL_TOL`/`ABS_TOL`/`TAU`/`H`）の変更が必要になった場合は
    //! 改めてユーザー承認が必須であり、#223 系譜の新規 Issue で追跡する。
    //!
    //! **キンク・タイ回避**: ReLU は `|x| >= 10h` の固定入力のみ、
    //! Max は同値タイのない固定入力のみを使う（固定値のため再生成
    //! ガードは不要。PoC-v2-2 `grad_check.rs` と同方針）。

    use super::*;

    const H: f64 = 1e-3;
    const TAU: f32 = 1e-4;
    const REL_TOL: f32 = 1e-2;
    const ABS_TOL: f32 = 1e-3;

    fn t(data: &[f32], shape: &[usize]) -> Tensor<f32> {
        Tensor::new(data.to_vec(), shape)
            .expect("test fixture: shape とデータ長は事前に一致させている")
    }

    fn assert_grad_close(label: &str, analytic: &Tensor<f32>, numeric: &Tensor<f32>) {
        let a = dense_vec(analytic);
        let n = dense_vec(numeric);
        assert_eq!(
            a.len(),
            n.len(),
            "{label}: analytic/numeric の要素数が一致しない"
        );
        for (i, (&av, &nv)) in a.iter().zip(n.iter()).enumerate() {
            let diff = (av - nv).abs();
            let rel = diff / av.abs().max(nv.abs()).max(TAU);
            assert!(
                rel <= REL_TOL || diff <= ABS_TOL,
                "{label}[{i}]: analytic={av} numeric={nv} diff={diff} rel={rel}"
            );
        }
    }

    /// `L(x) = Σ (forward(x) ⊙ s)` の f64 集計によるスカラー射影値。
    fn scalar_dot(a: &Tensor<f32>, s: &Tensor<f32>) -> f64 {
        dense_vec(a)
            .iter()
            .zip(dense_vec(s).iter())
            .map(|(&x, &y)| x as f64 * y as f64)
            .sum()
    }

    /// 単一入力 `x` に対する `L` の中央差分勾配（要素ごと・f64 集計）。
    fn numeric_grad_unary(
        x: &Tensor<f32>,
        s: &Tensor<f32>,
        forward: impl Fn(&Tensor<f32>) -> Tensor<f32>,
    ) -> Tensor<f32> {
        let shape = x.shape().to_vec();
        let mut data = dense_vec(x);
        let mut grad = vec![0f32; data.len()];
        for i in 0..data.len() {
            let orig = data[i] as f64;
            data[i] = (orig + H) as f32;
            let lp = scalar_dot(&forward(&build_tensor(data.clone(), &shape)), s);
            data[i] = (orig - H) as f32;
            let lm = scalar_dot(&forward(&build_tensor(data.clone(), &shape)), s);
            data[i] = orig as f32;
            grad[i] = ((lp - lm) / (2.0 * H)) as f32;
        }
        build_tensor(grad, &shape)
    }

    // --- ScalarUnary／ScalarBinary（イシュー #1634） ---
    //
    // `scalar_unary_with_fallback`／`scalar_binary_with_fallback`
    // （`ops.scalar_unary`／`scalar_binary` が常に `Unsupported` を返す
    // `test_ops()`〈`TestOps`。イシュー #1634 では `scalar_unary`／
    // `scalar_binary` を実装しないため既定 `Unsupported` のまま〉経由で
    // `eval::scalar::unary`／`binary` へのフォールバックを実地に叩く）と
    // `vjp()` ディスパッチの解析勾配を中央差分と突合する。

    /// 微分可能点（kink を避けた固定値）でのみ検証する unary variant
    /// （代表値・入力）。`LeakyRelu`／`Elu`／`Softplus` は `x == 0`
    /// 近傍の折れ点を避けるため `x` を離す。`Sqrt`/`Log*` は定義域が
    /// 正のため正の値のみ使う。
    fn unary_numeric_grad_cases() -> Vec<(ScalarUnaryOp, Vec<f32>)> {
        let general = vec![-1.7_f32, -0.6, 0.9, 2.3];
        let positive = vec![0.4_f32, 1.1, 2.7, 5.0];
        vec![
            (ScalarUnaryOp::Neg, general.clone()),
            (ScalarUnaryOp::Abs, vec![-1.7, -0.6, 0.9, 2.3]), // 0 を避ける（kink）
            (ScalarUnaryOp::Sqrt, positive.clone()),
            (ScalarUnaryOp::Log, positive.clone()),
            (ScalarUnaryOp::Log2, positive.clone()),
            (ScalarUnaryOp::Log10, positive.clone()),
            (ScalarUnaryOp::Sin, general.clone()),
            (ScalarUnaryOp::Cos, general.clone()),
            (ScalarUnaryOp::Tan, vec![-0.7, -0.2, 0.3, 0.8]),
            (ScalarUnaryOp::Relu, vec![-1.7, -0.6, 0.9, 2.3]),
            (ScalarUnaryOp::Exp, general.clone()),
            (ScalarUnaryOp::Tanh, general.clone()),
            (ScalarUnaryOp::Sigmoid, general.clone()),
            (ScalarUnaryOp::Gelu, general.clone()),
            (ScalarUnaryOp::GeluTanh, general.clone()),
            (ScalarUnaryOp::Silu, general.clone()),
            (ScalarUnaryOp::Hardswish, vec![-4.0, -1.0, 1.0, 4.0]),
            (
                ScalarUnaryOp::LeakyRelu {
                    negative_slope: 0.1,
                },
                vec![-1.7, -0.6, 0.9, 2.3],
            ),
            (
                ScalarUnaryOp::Elu { alpha: 1.3 },
                vec![-1.7, -0.6, 0.9, 2.3],
            ),
            (
                ScalarUnaryOp::Softplus {
                    beta: 1.0,
                    threshold: 20.0,
                },
                general.clone(),
            ),
            (
                ScalarUnaryOp::Clamp {
                    min: -1.0,
                    max: 1.0,
                },
                vec![-2.0, -0.3, 0.3, 2.0], // 境界 ±1.0 は避ける
            ),
            (ScalarUnaryOp::PowScalar { exponent: 2.5 }, positive),
        ]
    }

    #[test]
    fn scalar_unary_analytic_grad_matches_numeric_for_all_variants() {
        let ops = test_ops();
        for (op, xs) in unary_numeric_grad_cases() {
            let x = t(&xs, &[xs.len()]);
            let s = t(&vec![1.0; xs.len()], &[xs.len()]);

            let y = scalar_unary_with_fallback(&ops, op, &x).unwrap();
            let factor = eval::scalar::unary_grad_factors(&x, &y, op);
            let analytic = eval::mul(&factor, &s);

            let numeric = numeric_grad_unary(&x, &s, |xt| {
                scalar_unary_with_fallback(&ops, op, xt).unwrap()
            });

            assert_grad_close(&format!("scalar_unary({op:?})"), &analytic, &numeric);
        }
    }

    /// kink（劣勾配）点は中央差分がまたいで不定になるため、解析値を
    /// 直接検証する（§3.4 数値規約）。
    #[test]
    fn scalar_unary_subgradient_at_kink_points() {
        assert_eq!(
            eval::scalar::unary_grad_factor(ScalarUnaryOp::Relu, 0.0, 0.0),
            0.0
        );
        assert_eq!(
            eval::scalar::unary_grad_factor(ScalarUnaryOp::Abs, 0.0, 0.0),
            0.0
        );
        let clamp = ScalarUnaryOp::Clamp {
            min: -1.0,
            max: 1.0,
        };
        // 境界上は 1（通過）。
        assert_eq!(eval::scalar::unary_grad_factor(clamp, -1.0, -1.0), 1.0);
        assert_eq!(eval::scalar::unary_grad_factor(clamp, 1.0, 1.0), 1.0);
        // 範囲外は 0。
        assert_eq!(eval::scalar::unary_grad_factor(clamp, -2.0, -1.0), 0.0);
    }

    fn binary_numeric_grad_cases() -> Vec<(ScalarBinaryOp, Vec<f32>, Vec<f32>)> {
        let a = vec![1.3_f32, -0.7, 2.1, 0.4];
        let b = vec![0.6_f32, 1.8, -1.1, 2.4]; // Div/Pow の分母・底が 0 に近すぎない
        vec![
            (ScalarBinaryOp::Add, a.clone(), b.clone()),
            (ScalarBinaryOp::Sub, a.clone(), b.clone()),
            (ScalarBinaryOp::Mul, a.clone(), b.clone()),
            (ScalarBinaryOp::Div, a.clone(), b.clone()),
            (
                ScalarBinaryOp::Pow,
                vec![1.3, 0.7, 2.1, 0.4], // Pow は a > 0 前提（ln(a) を使うため）
                b.clone(),
            ),
            (
                ScalarBinaryOp::Maximum,
                a.clone(),
                vec![0.1, -1.5, 3.0, -0.2],
            ),
            (ScalarBinaryOp::Minimum, a, vec![0.1, -1.5, 3.0, -0.2]),
        ]
    }

    #[test]
    fn scalar_binary_analytic_grad_matches_numeric_for_all_variants() {
        let ops = test_ops();
        for (op, a_data, b_data) in binary_numeric_grad_cases() {
            let len = a_data.len();
            let a = t(&a_data, &[len]);
            let b = t(&b_data, &[len]);
            let s = t(&vec![1.0; len], &[len]);

            let y = scalar_binary_with_fallback(&ops, op, &a, &b, &[len]).unwrap();
            let (factor_a, factor_b) = eval::scalar::binary_grad_factors(&a, &b, &y, op);
            let analytic_da = eval::mul(&factor_a, &s);
            let analytic_db = eval::mul(&factor_b, &s);

            let numeric_da = numeric_grad_unary(&a, &s, |xt| {
                scalar_binary_with_fallback(&ops, op, xt, &b, &[len]).unwrap()
            });
            let numeric_db = numeric_grad_unary(&b, &s, |xt| {
                scalar_binary_with_fallback(&ops, op, &a, xt, &[len]).unwrap()
            });

            assert_grad_close(
                &format!("scalar_binary({op:?}) da"),
                &analytic_da,
                &numeric_da,
            );
            assert_grad_close(
                &format!("scalar_binary({op:?}) db"),
                &analytic_db,
                &numeric_db,
            );
        }
    }

    #[test]
    fn scalar_binary_maximum_minimum_tie_and_comparison_zero_grad() {
        let (da, db) = eval::scalar::binary_partials(ScalarBinaryOp::Maximum, 1.0, 1.0, 1.0);
        assert_eq!((da, db), (0.5, 0.5));
        let (da, db) = eval::scalar::binary_partials(ScalarBinaryOp::Gt, 1.0, 2.0, 0.0);
        assert_eq!((da, db), (0.0, 0.0));
    }

    #[test]
    fn scalar_binary_pow_db_masked_at_zero_base() {
        let (_, db) = eval::scalar::binary_partials(ScalarBinaryOp::Pow, 0.0, 2.0, 0.0);
        assert_eq!(
            db, 0.0,
            "a==0 では ln(a) 発散を避けるため db=0 にマスクする"
        );
    }

    // --- PR #1686 codex-review／Bugbot 指摘の回帰テスト ---
    // （`docs/scalar-op-dispatch-design.md`「PR #1686 codex-review／Bugbot
    // 指摘の是正」参照。以下は既存 db マスク〈a==0〉と対になる da
    // マスク〈b==0〉・unary PowScalar の exponent==0 マスク・Div の
    // overflow/underflow 耐性・Maximum/Minimum の NaN 規約）。

    #[test]
    fn scalar_binary_pow_da_masked_at_zero_exponent_even_when_base_is_also_zero() {
        // a=0, b=0 はガードなしだと `da = b * a.powf(b-1)` =
        // `0.0 * 0.0.powf(-1.0)` = `0.0 * inf` = `NaN` になっていた
        // （forward `y = 0.0.powf(0.0) = 1.0` は IEEE 754 の 0^0 規約
        // どおり有限値のため、勾配だけが不正に NaN 化する非対称な bug）。
        let (da, db) = eval::scalar::binary_partials(ScalarBinaryOp::Pow, 0.0, 0.0, 1.0);
        assert_eq!(
            (da, db),
            (0.0, 0.0),
            "a==0 かつ b==0 では da/db とも 0 にマスクする（b==0 は定数関数のため da=0、\
             a==0 は ln(0) 発散回避のため db=0）"
        );
        // a!=0, b=0 でも da は定数関数（a^0=1）の勾配として 0。
        let (da2, _) = eval::scalar::binary_partials(ScalarBinaryOp::Pow, 3.0, 0.0, 1.0);
        assert_eq!(da2, 0.0, "b==0 では a の値によらず da=0（定数関数）");
    }

    #[test]
    fn scalar_unary_pow_scalar_grad_masked_at_zero_exponent_even_when_x_is_also_zero() {
        // x=0, exponent=0 はガードなしだと `exponent * x.powf(exponent-1)`
        // = `0.0 * 0.0.powf(-1.0)` = `NaN` になっていた（binary Pow の
        // da と同型の bug）。
        let grad =
            eval::scalar::unary_grad_factor(ScalarUnaryOp::PowScalar { exponent: 0.0 }, 0.0, 1.0);
        assert_eq!(
            grad, 0.0,
            "exponent==0 では x の値によらず勾配 0（定数関数）"
        );
    }

    #[test]
    fn scalar_binary_div_db_avoids_underflow_and_overflow_of_b_squared() {
        // a=b=1e-30: 素朴な `-a/(b*b)` は `b*b` が 0 へ underflow して
        // `-inf` になっていたが、数学的な値は `-1/b = -1e30`（有限）。
        let (_, db_small) = eval::scalar::binary_partials(ScalarBinaryOp::Div, 1e-30, 1e-30, 1.0);
        assert!(
            db_small.is_finite(),
            "a=b=1e-30 の db は有限値であるべき: {db_small}"
        );
        let rel = (db_small - (-1e30)).abs() / 1e30;
        assert!(rel < 1e-3, "db_small={db_small} が -1e30 から乖離しすぎ");

        // a=b=1e20: 素朴な `-a/(b*b)` は `b*b` が inf へ overflow して
        // `-0.0`（実質ゼロ）になっていたが、数学的な値は
        // `-1/b = -1e-20`（有限の非ゼロ値）。
        let (_, db_large) = eval::scalar::binary_partials(ScalarBinaryOp::Div, 1e20, 1e20, 1.0);
        assert!(
            db_large.is_finite() && db_large != 0.0,
            "a=b=1e20 の db は有限の非ゼロ値であるべき: {db_large}"
        );
        let rel_large = (db_large - (-1e-20)).abs() / 1e-20;
        assert!(
            rel_large < 1e-3,
            "db_large={db_large} が -1e-20 から乖離しすぎ"
        );
    }

    #[test]
    fn scalar_binary_maximum_minimum_nan_input_yields_zero_grad_for_both_sides() {
        // NaN 入力では IEEE 754 比較（`>`／`<`）がすべて false になり
        // タイ分割（0.5/0.5）へ落ちていた（Bugbot 指摘）。forward が
        // NaN を明示伝播する契約（`nan_propagating_max`/`_min`）と
        // `Clamp` の NaN 規約に揃え、両入力の勾配をゼロにする。
        let (da, db) = eval::scalar::binary_partials(ScalarBinaryOp::Maximum, f32::NAN, 1.0, 0.0);
        assert_eq!((da, db), (0.0, 0.0), "Maximum(NaN, 1.0) は勾配ゼロ");
        let (da, db) = eval::scalar::binary_partials(ScalarBinaryOp::Maximum, 1.0, f32::NAN, 0.0);
        assert_eq!((da, db), (0.0, 0.0), "Maximum(1.0, NaN) は勾配ゼロ");
        let (da, db) =
            eval::scalar::binary_partials(ScalarBinaryOp::Maximum, f32::NAN, f32::NAN, 0.0);
        assert_eq!((da, db), (0.0, 0.0), "Maximum(NaN, NaN) は勾配ゼロ");

        let (da, db) = eval::scalar::binary_partials(ScalarBinaryOp::Minimum, f32::NAN, 1.0, 0.0);
        assert_eq!((da, db), (0.0, 0.0), "Minimum(NaN, 1.0) は勾配ゼロ");
        let (da, db) = eval::scalar::binary_partials(ScalarBinaryOp::Minimum, 1.0, f32::NAN, 0.0);
        assert_eq!((da, db), (0.0, 0.0), "Minimum(1.0, NaN) は勾配ゼロ");
    }

    #[test]
    fn scalar_unary_elu_grad_is_computed_from_input_not_rounded_forward_output() {
        // 旧実装は forward 出力 `y = alpha * expm1(x)` から `y + alpha` で
        // 復元していたため、`alpha` が大きく `x` が負のとき `y` が
        // `-alpha` へ丸まり係数が `0.0` になって勾配が消失していた
        // （PR #1686 codex-review 2 回目の指摘 P2）。
        let alpha = 1e8_f32;
        let x = -20.0_f32;
        let y = ScalarUnaryOp::Elu { alpha }.apply(x);
        assert_eq!(y, -alpha, "前提: forward 出力は -alpha へ丸まる");
        let g = eval::scalar::unary_grad_factor(ScalarUnaryOp::Elu { alpha }, x, y);
        let expected = alpha * x.exp(); // 約 0.20611536
        assert!(
            (g - expected).abs() <= expected * 1e-5,
            "ELU 勾配は入力から計算した alpha*exp(x)={expected} に一致すべき: {g}"
        );
        assert!(g > 0.0, "勾配が消失してはならない");
    }

    // --- Var::scalar_unary／scalar_binary の Tape 経由エンドツーエンド ---

    #[test]
    fn var_scalar_unary_backward_matches_manual_relu_subgradient() {
        let tape = crate::tape::Tape::new_with_ops(crate::test_support::test_ops());
        let x = tape.var(&t(&[1.5, -2.0, 0.0, 3.0], &[4]));
        let y = x.scalar_unary(ScalarUnaryOp::Relu).unwrap();
        let loss = y.sum(None).unwrap();
        let grads = tape.backward(&loss).unwrap();
        let dx = grads.get(&x).unwrap().unwrap();
        let expected = [1.0, 0.0, 0.0, 1.0];
        for (i, &e) in expected.iter().enumerate() {
            assert_eq!(dx.get(&[i]).unwrap(), e, "d(relu)/dx[{i}]");
        }
    }

    #[test]
    fn var_scalar_binary_backward_matches_manual_add_grad() {
        let tape = crate::tape::Tape::new_with_ops(crate::test_support::test_ops());
        let a = tape.var(&t(&[1.0, 2.0, 3.0], &[3]));
        let b = tape.var(&t(&[10.0, 20.0, 30.0], &[3]));
        let y = a.scalar_binary(&b, ScalarBinaryOp::Add).unwrap();
        let loss = y.sum(None).unwrap();
        let grads = tape.backward(&loss).unwrap();
        let da = grads.get(&a).unwrap().unwrap();
        let db = grads.get(&b).unwrap().unwrap();
        for i in 0..3 {
            assert_eq!(da.get(&[i]).unwrap(), 1.0);
            assert_eq!(db.get(&[i]).unwrap(), 1.0);
        }
    }

    // --- Var::log／log2／log10／sin／cos／tan／abs／neg（イシュー #1711）
    // Tape 経由エンドツーエンド ---

    #[test]
    fn var_log_family_forward_and_backward_matches_manual_grad() {
        let tape = crate::tape::Tape::new_with_ops(crate::test_support::test_ops());
        let xs = [1.0_f32, 2.0, 4.0, 10.0];
        let x = tape.var(&t(&xs, &[4]));
        let y_ln = x.log().unwrap();
        let y_log2 = x.log2().unwrap();
        let y_log10 = x.log10().unwrap();
        for (i, &xv) in xs.iter().enumerate() {
            assert!((y_ln.value().get(&[i]).unwrap() - xv.ln()).abs() < 1e-5);
            assert!((y_log2.value().get(&[i]).unwrap() - xv.log2()).abs() < 1e-5);
            assert!((y_log10.value().get(&[i]).unwrap() - xv.log10()).abs() < 1e-5);
        }

        let loss = y_ln.sum(None).unwrap();
        let grads = tape.backward(&loss).unwrap();
        let dx = grads.get(&x).unwrap().unwrap();
        for (i, &xv) in xs.iter().enumerate() {
            assert!(
                (dx.get(&[i]).unwrap() - 1.0 / xv).abs() < 1e-5,
                "d(ln)/dx[{i}] は 1/x に一致すべき"
            );
        }
    }

    #[test]
    fn var_log_nonpositive_input_yields_inf_or_nan_without_panic() {
        let tape = crate::tape::Tape::new_with_ops(crate::test_support::test_ops());
        let x = tape.var(&t(&[0.0, -1.0], &[2]));
        let y = x.log().unwrap();
        assert_eq!(y.value().get(&[0]).unwrap(), f32::NEG_INFINITY);
        assert!(y.value().get(&[1]).unwrap().is_nan());
    }

    #[test]
    fn var_sin_cos_tan_backward_matches_manual_grad() {
        let tape = crate::tape::Tape::new_with_ops(crate::test_support::test_ops());
        let xs = [-0.7_f32, 0.3, 0.8];
        let x_sin = tape.var(&t(&xs, &[3]));
        let y_sin = x_sin.sin().unwrap();
        let loss_sin = y_sin.sum(None).unwrap();
        let grads_sin = tape.backward(&loss_sin).unwrap();
        let dx_sin = grads_sin.get(&x_sin).unwrap().unwrap();

        let x_cos = tape.var(&t(&xs, &[3]));
        let y_cos = x_cos.cos().unwrap();
        let loss_cos = y_cos.sum(None).unwrap();
        let grads_cos = tape.backward(&loss_cos).unwrap();
        let dx_cos = grads_cos.get(&x_cos).unwrap().unwrap();

        let x_tan = tape.var(&t(&xs, &[3]));
        let y_tan = x_tan.tan().unwrap();
        let loss_tan = y_tan.sum(None).unwrap();
        let grads_tan = tape.backward(&loss_tan).unwrap();
        let dx_tan = grads_tan.get(&x_tan).unwrap().unwrap();

        for (i, &xv) in xs.iter().enumerate() {
            assert!((y_sin.value().get(&[i]).unwrap() - xv.sin()).abs() < 1e-5);
            assert!(
                (dx_sin.get(&[i]).unwrap() - xv.cos()).abs() < 1e-5,
                "d(sin)/dx"
            );
            assert!((y_cos.value().get(&[i]).unwrap() - xv.cos()).abs() < 1e-5);
            assert!(
                (dx_cos.get(&[i]).unwrap() - (-xv.sin())).abs() < 1e-5,
                "d(cos)/dx"
            );
            assert!((y_tan.value().get(&[i]).unwrap() - xv.tan()).abs() < 1e-4);
            let expected_dtan = 1.0 / (xv.cos() * xv.cos());
            assert!(
                (dx_tan.get(&[i]).unwrap() - expected_dtan).abs() < 1e-3,
                "d(tan)/dx"
            );
        }
    }

    /// `abs`（劣勾配）の forward・backward を検証する。`neg` とは分離
    /// 可能な独立テスト単位とする（`docs/compat-api-scope.md` §1.2 の
    /// 範囲判断参照）。
    #[test]
    fn var_abs_backward_subgradient_sign() {
        let tape = crate::tape::Tape::new_with_ops(crate::test_support::test_ops());
        let x = tape.var(&t(&[-2.0, 0.0, 3.0], &[3]));
        let y = x.abs().unwrap();
        let loss = y.sum(None).unwrap();
        let grads = tape.backward(&loss).unwrap();
        let dx = grads.get(&x).unwrap().unwrap();
        assert_eq!(y.value().get(&[0]).unwrap(), 2.0);
        assert_eq!(y.value().get(&[1]).unwrap(), 0.0);
        assert_eq!(y.value().get(&[2]).unwrap(), 3.0);
        assert_eq!(dx.get(&[0]).unwrap(), -1.0);
        assert_eq!(dx.get(&[1]).unwrap(), 0.0, "劣勾配は x==0 で 0");
        assert_eq!(dx.get(&[2]).unwrap(), 1.0);
    }

    /// `neg` の forward（`-0.0` の符号ビット反転を含む）・backward を
    /// 検証する（`abs` とは分離可能な独立テスト単位）。
    #[test]
    fn var_neg_forward_and_backward() {
        let tape = crate::tape::Tape::new_with_ops(crate::test_support::test_ops());
        let x = tape.var(&t(&[1.5, -0.0, -2.0], &[3]));
        let y = x.neg().unwrap();
        let loss = y.sum(None).unwrap();
        let grads = tape.backward(&loss).unwrap();
        let dx = grads.get(&x).unwrap().unwrap();
        assert_eq!(y.value().get(&[0]).unwrap(), -1.5);
        let neg_zero = y.value().get(&[1]).unwrap();
        assert_eq!(neg_zero, 0.0);
        assert!(
            neg_zero.is_sign_positive(),
            "neg(-0.0) は符号ビット反転で +0.0 になるべき"
        );
        assert_eq!(y.value().get(&[2]).unwrap(), 2.0);
        for i in 0..3 {
            assert_eq!(dx.get(&[i]).unwrap(), -1.0, "d(neg)/dx は定数 -1");
        }
    }

    // --- Var::sub／div／pow／sqrt（イシュー #1710）: Tape 経由
    //     エンドツーエンド ---
    //
    // forward の解析式・VJP 係数自体は `scalar_unary_analytic_grad_
    // matches_numeric_for_all_variants`／`scalar_binary_analytic_grad_
    // matches_numeric_for_all_variants`（`Sqrt`／`Sub`／`Div`／`Pow` を
    // 含む）が既に中央差分と突合済みのため、ここでは `Var` 公開
    // メソッドが `Op::ScalarUnary`／`ScalarBinary` を正しく記録し
    // `Tape::backward` が期待どおりの勾配（broadcast 縮約を含む）を
    // 返すことを手計算値で検証する。

    #[test]
    fn var_sub_backward_matches_manual_grad_with_broadcast() {
        // [2,3] - [3]（bias broadcast）。d(a-b)/da = 1・d(a-b)/db = -1
        // で、db は broadcast された行方向に `reduce_bias_grad`
        // （f64 相当アキュムレータ）で縮約される（`Op::Add` と同型）。
        let tape = crate::tape::Tape::new_with_ops(crate::test_support::test_ops());
        let a = tape.var(&t(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]));
        let b = tape.var(&t(&[10.0, 20.0, 30.0], &[3]));
        let y = a.sub(&b).unwrap();
        assert_eq!(
            dense_vec(&y.to_tensor()),
            vec![-9.0, -18.0, -27.0, -6.0, -15.0, -24.0]
        );
        let loss = y.sum(None).unwrap();
        let grads = tape.backward(&loss).unwrap();
        let da = grads.get(&a).unwrap().unwrap();
        let db = grads.get(&b).unwrap().unwrap();
        for i in 0..6 {
            assert_eq!(dense_vec(da)[i], 1.0, "d(a-b)/da[{i}]");
        }
        for i in 0..3 {
            // 2 行分（各 1.0）を合算するため db = -2.0。
            assert_eq!(dense_vec(db)[i], -2.0, "d(a-b)/db[{i}]");
        }
    }

    #[test]
    fn var_div_backward_matches_manual_grad() {
        // d(a/b)/da = 1/b・d(a/b)/db = -a/b^2。
        let tape = crate::tape::Tape::new_with_ops(crate::test_support::test_ops());
        let a = tape.var(&t(&[6.0, 9.0], &[2]));
        let b = tape.var(&t(&[2.0, 3.0], &[2]));
        let y = a.div(&b).unwrap();
        assert_eq!(dense_vec(&y.to_tensor()), vec![3.0, 3.0]);
        let loss = y.sum(None).unwrap();
        let grads = tape.backward(&loss).unwrap();
        let da = grads.get(&a).unwrap().unwrap();
        let db = grads.get(&b).unwrap().unwrap();
        assert_grad_close("div da", da, &t(&[0.5, 1.0 / 3.0], &[2]));
        assert_grad_close("div db", db, &t(&[-1.5, -1.0], &[2]));
    }

    #[test]
    fn var_pow_backward_matches_manual_grad() {
        // y = a^b（a > 0）。da = b*a^(b-1)・db = y*ln(a)。
        let tape = crate::tape::Tape::new_with_ops(crate::test_support::test_ops());
        let a = tape.var(&t(&[2.0, 3.0], &[2]));
        let b = tape.var(&t(&[3.0, 2.0], &[2]));
        let y = a.pow(&b).unwrap();
        assert_eq!(dense_vec(&y.to_tensor()), vec![8.0, 9.0]);
        let loss = y.sum(None).unwrap();
        let grads = tape.backward(&loss).unwrap();
        let da = grads.get(&a).unwrap().unwrap();
        let db = grads.get(&b).unwrap().unwrap();
        assert_grad_close("pow da", da, &t(&[12.0, 6.0], &[2]));
        assert_grad_close(
            "pow db",
            db,
            &t(&[8.0 * 2.0_f32.ln(), 9.0 * 3.0_f32.ln()], &[2]),
        );
    }

    #[test]
    fn var_sqrt_forward_and_backward_matches_manual_grad() {
        // y = sqrt(x)。dy/dx = 0.5/y。
        let tape = crate::tape::Tape::new_with_ops(crate::test_support::test_ops());
        let x = tape.var(&t(&[4.0, 9.0], &[2]));
        let y = x.sqrt().unwrap();
        assert_eq!(dense_vec(&y.to_tensor()), vec![2.0, 3.0]);
        let loss = y.sum(None).unwrap();
        let grads = tape.backward(&loss).unwrap();
        let dx = grads.get(&x).unwrap().unwrap();
        assert_grad_close("sqrt dx", dx, &t(&[0.25, 1.0 / 6.0], &[2]));
    }

    #[test]
    fn var_sqrt_negative_input_yields_nan_without_panic() {
        // 定義域外（`x < 0`）は IEEE `NaN`（PyTorch `torch.sqrt` と同じ
        // 規約。設計 §7）で panic しない。
        let tape = crate::tape::Tape::new_with_ops(crate::test_support::test_ops());
        let x = tape.var(&t(&[-1.0], &[1]));
        let y = x.sqrt().unwrap();
        assert!(dense_vec(&y.to_tensor())[0].is_nan());
    }

    #[test]
    fn var_gelu_forward_and_backward_matches_manual_grad() {
        // GELU（誤差関数版）: y = 0.5*x*(1+erf(x/sqrt(2)))。
        // x=0 で y=0・dy/dx=0.5（Φ(0)=0.5・φ(0)=1/sqrt(2π)≈0.3989 より
        // dy/dx = Φ(0) + 0*φ(0) = 0.5）。
        let tape = crate::tape::Tape::new_with_ops(crate::test_support::test_ops());
        let x = tape.var(&t(&[0.0, 1.0], &[2]));
        let y = x.gelu().unwrap();
        let out = dense_vec(&y.to_tensor());
        assert!((out[0] - 0.0).abs() < 1e-6, "gelu(0) = {}", out[0]);
        assert!((out[1] - 0.841_344_7).abs() < 1e-5, "gelu(1) = {}", out[1]);
        let loss = y.sum(None).unwrap();
        let grads = tape.backward(&loss).unwrap();
        let dx = grads.get(&x).unwrap().unwrap();
        assert_grad_close(
            "gelu dx",
            dx,
            &t(
                &[0.5, fandhe_ai_tensor_core::scalar_op::gelu_erf_grad(1.0)],
                &[2],
            ),
        );
    }

    #[test]
    fn var_gelu_tanh_forward_and_backward_matches_manual_grad() {
        // GELU（tanh 近似版）: x=0 で y=0（奇関数なので tanh(0)=0）。
        let tape = crate::tape::Tape::new_with_ops(crate::test_support::test_ops());
        let x = tape.var(&t(&[0.0, 1.0], &[2]));
        let y = x.gelu_tanh().unwrap();
        let out = dense_vec(&y.to_tensor());
        assert!((out[0] - 0.0).abs() < 1e-6, "gelu_tanh(0) = {}", out[0]);
        let loss = y.sum(None).unwrap();
        let grads = tape.backward(&loss).unwrap();
        let dx = grads.get(&x).unwrap().unwrap();
        assert_grad_close(
            "gelu_tanh dx",
            dx,
            &t(
                &[
                    fandhe_ai_tensor_core::scalar_op::gelu_tanh_grad(0.0),
                    fandhe_ai_tensor_core::scalar_op::gelu_tanh_grad(1.0),
                ],
                &[2],
            ),
        );
    }

    #[test]
    fn var_softplus_forward_and_backward_matches_manual_grad() {
        // softplus(0) = ln(2)（既定 beta=1・threshold=20 では恒等分岐
        // 〈x*beta > threshold〉に入らない）。dy/dx = sigmoid(beta*x)。
        let tape = crate::tape::Tape::new_with_ops(crate::test_support::test_ops());
        let x = tape.var(&t(&[0.0, 1.0], &[2]));
        let y = x.softplus(1.0, 20.0).unwrap();
        let out = dense_vec(&y.to_tensor());
        assert!(
            (out[0] - std::f32::consts::LN_2).abs() < 1e-5,
            "softplus(0) = {}",
            out[0]
        );
        let loss = y.sum(None).unwrap();
        let grads = tape.backward(&loss).unwrap();
        let dx = grads.get(&x).unwrap().unwrap();
        let sigmoid = |v: f32| 1.0 / (1.0 + (-v).exp());
        assert_grad_close("softplus dx", dx, &t(&[sigmoid(0.0), sigmoid(1.0)], &[2]));
    }

    #[test]
    fn var_softplus_identity_branch_gradient_is_one() {
        // beta=2.0・threshold=1.0 では x=1.0 のとき x*beta=2.0 > 1.0 で
        // 恒等分岐（y=x・dy/dx=1.0）に入る。
        let tape = crate::tape::Tape::new_with_ops(crate::test_support::test_ops());
        let x = tape.var(&t(&[1.0], &[1]));
        let y = x.softplus(2.0, 1.0).unwrap();
        assert_eq!(dense_vec(&y.to_tensor()), vec![1.0]);
        let loss = y.sum(None).unwrap();
        let grads = tape.backward(&loss).unwrap();
        let dx = grads.get(&x).unwrap().unwrap();
        assert_grad_close("softplus identity-branch dx", dx, &t(&[1.0], &[1]));
    }

    #[test]
    fn var_softplus_rejects_non_positive_or_non_finite_beta_and_non_finite_threshold() {
        let tape = crate::tape::Tape::new_with_ops(crate::test_support::test_ops());
        let x = tape.var(&t(&[1.0], &[1]));
        assert!(matches!(
            x.softplus(0.0, 20.0),
            Err(AutodiffError::InvalidArgument(_))
        ));
        assert!(matches!(
            x.softplus(-1.0, 20.0),
            Err(AutodiffError::InvalidArgument(_))
        ));
        assert!(matches!(
            x.softplus(f32::NAN, 20.0),
            Err(AutodiffError::InvalidArgument(_))
        ));
        assert!(matches!(
            x.softplus(f32::INFINITY, 20.0),
            Err(AutodiffError::InvalidArgument(_))
        ));
        assert!(matches!(
            x.softplus(1.0, f32::NAN),
            Err(AutodiffError::InvalidArgument(_))
        ));
        assert!(matches!(
            x.softplus(1.0, f32::INFINITY),
            Err(AutodiffError::InvalidArgument(_))
        ));
        assert!(x.softplus(1.0, 20.0).is_ok());
    }

    #[test]
    fn var_sub_div_pow_reject_cross_tape_and_non_broadcastable_shape() {
        // cross-tape は fail-closed（`check_same_tape`）。
        let tape_a = crate::tape::Tape::new_with_ops(crate::test_support::test_ops());
        let tape_b = crate::tape::Tape::new_with_ops(crate::test_support::test_ops());
        let a = tape_a.var(&t(&[1.0], &[1]));
        let b = tape_b.var(&t(&[1.0], &[1]));
        assert!(a.sub(&b).is_err());
        assert!(a.div(&b).is_err());
        assert!(a.pow(&b).is_err());

        // 非 broadcast 可能 shape も fail-closed（`broadcast_shape`）。
        let tape = crate::tape::Tape::new_with_ops(crate::test_support::test_ops());
        let x = tape.var(&t(&[1.0; 6], &[2, 3]));
        let y = tape.var(&t(&[1.0; 4], &[2, 2]));
        assert!(matches!(x.sub(&y).unwrap_err(), AutodiffError::Shape(_)));
        assert!(matches!(x.div(&y).unwrap_err(), AutodiffError::Shape(_)));
        assert!(matches!(x.pow(&y).unwrap_err(), AutodiffError::Shape(_)));
    }

    // --- Var::clamp／比較演算（イシュー #1712）: Tape 経由
    //     エンドツーエンド ---
    //
    // 解析式・VJP 係数自体は `ScalarUnaryOp::Clamp`／`ScalarBinaryOp`
    // 比較 6 種の中央差分突合（`tensor-core::scalar_op` の
    // 単体テスト・`scalar_binary_maximum_minimum_tie_and_comparison_
    // zero_grad` 等。#1634／#1686）が既に済んでいるため、ここでは
    // `Var` 公開メソッドが `Op::ScalarUnary`／`ScalarBinary` を正しく
    // 記録し `Tape::backward` が期待どおりの勾配（ゼロ勾配・
    // broadcast 縮約を含む）を返すことを手計算値で検証する。

    #[test]
    fn var_clamp_forward_and_backward_matches_manual_grad() {
        // clamp(x, -1, 1)。境界上（-1・1）は勾配 1、範囲外は 0。
        let tape = crate::tape::Tape::new_with_ops(crate::test_support::test_ops());
        let x = tape.var(&t(&[-2.0, -1.0, 0.5, 1.0, 2.0], &[5]));
        let y = x.clamp(-1.0, 1.0).unwrap();
        assert_eq!(dense_vec(&y.to_tensor()), vec![-1.0, -1.0, 0.5, 1.0, 1.0]);
        let loss = y.sum(None).unwrap();
        let grads = tape.backward(&loss).unwrap();
        let dx = grads.get(&x).unwrap().unwrap();
        assert_eq!(dense_vec(dx), vec![0.0, 1.0, 1.0, 1.0, 0.0]);
    }

    #[test]
    fn var_clamp_min_greater_than_max_is_constant_with_zero_grad() {
        // min > max は PyTorch と同じく常に max を返す定数関数
        // （panic しない）。勾配は全域でゼロ。
        let tape = crate::tape::Tape::new_with_ops(crate::test_support::test_ops());
        let x = tape.var(&t(&[-10.0, 0.0, 10.0], &[3]));
        let y = x.clamp(5.0, 1.0).unwrap();
        assert_eq!(dense_vec(&y.to_tensor()), vec![1.0, 1.0, 1.0]);
        let loss = y.sum(None).unwrap();
        let grads = tape.backward(&loss).unwrap();
        let dx = grads.get(&x).unwrap().unwrap();
        assert_eq!(dense_vec(dx), vec![0.0, 0.0, 0.0]);
    }

    #[test]
    fn var_clamp_nan_propagates_with_zero_grad() {
        let tape = crate::tape::Tape::new_with_ops(crate::test_support::test_ops());
        let x = tape.var(&t(&[f32::NAN], &[1]));
        let y = x.clamp(-1.0, 1.0).unwrap();
        assert!(dense_vec(&y.to_tensor())[0].is_nan());
        let loss = y.sum(None).unwrap();
        let grads = tape.backward(&loss).unwrap();
        let dx = grads.get(&x).unwrap().unwrap();
        assert_eq!(dense_vec(dx), vec![0.0]);
    }

    #[test]
    fn var_comparisons_forward_return_zero_or_one() {
        // NaN・tie（a[1]==b[1] の 2.0）を含める（IEEE 754 準拠の
        // 確認。`eq`/`ne` は NaN を含む比較で特別扱い: eq=0・ne=1）。
        let tape = crate::tape::Tape::new_with_ops(crate::test_support::test_ops());
        let a = tape.var(&t(&[1.0, 2.0, 3.0, f32::NAN], &[4]));
        let b = tape.var(&t(&[2.0, 2.0, 1.0, 1.0], &[4]));
        assert_eq!(
            dense_vec(&a.gt(&b).unwrap().to_tensor()),
            vec![0.0, 0.0, 1.0, 0.0]
        );
        assert_eq!(
            dense_vec(&a.ge(&b).unwrap().to_tensor()),
            vec![0.0, 1.0, 1.0, 0.0]
        );
        assert_eq!(
            dense_vec(&a.lt(&b).unwrap().to_tensor()),
            vec![1.0, 0.0, 0.0, 0.0]
        );
        assert_eq!(
            dense_vec(&a.le(&b).unwrap().to_tensor()),
            vec![1.0, 1.0, 0.0, 0.0]
        );
        assert_eq!(
            dense_vec(&a.eq(&b).unwrap().to_tensor()),
            vec![0.0, 1.0, 0.0, 0.0]
        );
        assert_eq!(
            dense_vec(&a.ne(&b).unwrap().to_tensor()),
            vec![1.0, 0.0, 1.0, 1.0]
        );
    }

    #[test]
    fn var_comparison_backward_yields_zero_grads_with_broadcast() {
        // [2,3] gt [3]（broadcast）でも da／db は共に全ゼロ
        // （`reduce_bias_grad` の縮約経路を通っても値がゼロのまま
        // 一致することを固定）。
        let tape = crate::tape::Tape::new_with_ops(crate::test_support::test_ops());
        let a = tape.var(&t(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]));
        let b = tape.var(&t(&[2.0, 2.0, 2.0], &[3]));
        let y = a.gt(&b).unwrap();
        let loss = y.sum(None).unwrap();
        let grads = tape.backward(&loss).unwrap();
        let da = grads.get(&a).unwrap().unwrap();
        let db = grads.get(&b).unwrap().unwrap();
        assert_eq!(dense_vec(da), vec![0.0; 6]);
        assert_eq!(dense_vec(db), vec![0.0; 3]);
    }

    #[test]
    fn var_comparison_zero_grad_does_not_block_other_paths() {
        // y = x * (x > c) という典型的なマスク合成で、比較側の
        // ゼロ勾配が x の他経路の勾配（マスク値との積）を妨げない
        // ことを確認する。c は定数（tape に登録しない）ではなく
        // 同一 tape 上の leaf とし、d(x*mask)/dx = mask + x*d(mask)/dx
        // = mask + x*0 = mask（gt 側の勾配が常にゼロのため mask 項の
        // みが残る）ことを固定する。
        let tape = crate::tape::Tape::new_with_ops(crate::test_support::test_ops());
        let x = tape.var(&t(&[-1.0, 2.0, 3.0], &[3]));
        let c = tape.var(&t(&[0.0, 0.0, 0.0], &[3]));
        let mask = x.gt(&c).unwrap(); // [0, 1, 1]
        let y = x.mul(&mask).unwrap(); // [0, 2, 3]
        assert_eq!(dense_vec(&y.to_tensor()), vec![0.0, 2.0, 3.0]);
        let loss = y.sum(None).unwrap();
        let grads = tape.backward(&loss).unwrap();
        let dx = grads.get(&x).unwrap().unwrap();
        // dy/dx = mask（gt 側の勾配は寄与せずゼロ）。
        assert_eq!(dense_vec(dx), vec![0.0, 1.0, 1.0]);
    }

    #[test]
    fn var_comparison_backward_does_not_propagate_upstream_inf_nan() {
        // codex-review 指摘（PR #1823）: 比較演算の VJP が upstream への
        // 乗算（`0.0 * 係数`）経由でゼロ勾配化していると、その比較演算
        // ノード自体への upstream（＝下流から流入する勾配）が `inf`／
        // `NaN` を含む場合に `0.0 * inf = NaN` へ汚染される。
        //
        // `y = x * x.gt(c)`（`x = inf, c = 0`）で loss = sum(y) を取ると、
        // `Op::Mul` の VJP により mask（`x.gt(c)`）自身への upstream は
        // `x = inf` になる（`db = upstream(=1) * x_val(=inf)`）。この
        // `inf` が `mask` ノードの `Op::ScalarBinary(Gt)` 分岐へ upstream
        // として渡されたとき、修正前は `0.0 係数 * inf = NaN` が
        // `x`／`c` の最終勾配へ混入していた。修正後は乗算を経由せず
        // 直接ゼロテンソルを返すため、`x` の最終勾配は `mul` 側の寄与
        // （`mask_val = 1.0`）のみが残り有限値のまま、`c` の最終勾配は
        // 有限のゼロのままになる。
        let tape = crate::tape::Tape::new_with_ops(crate::test_support::test_ops());
        let x = tape.var(&t(&[f32::INFINITY], &[1]));
        let c = tape.var(&t(&[0.0], &[1]));
        let mask = x.gt(&c).unwrap(); // [1.0]（inf > 0）
        let y = x.mul(&mask).unwrap(); // [inf]
        let loss = y.sum(None).unwrap();
        let grads = tape.backward(&loss).unwrap();
        let dx = dense_vec(grads.get(&x).unwrap().unwrap());
        let dc = dense_vec(grads.get(&c).unwrap().unwrap());
        // `mul` 側の寄与（mask_val = 1.0）のみが残り、`gt` 側の寄与は
        // 有限のゼロ（修正前は NaN で全体を汚染していた）。
        assert!(dx[0].is_finite(), "dx が NaN 化した: {dx:?}");
        assert_eq!(dx, vec![1.0]);
        assert!(dc[0].is_finite(), "dc が NaN 化した: {dc:?}");
        assert_eq!(dc, vec![0.0]);
    }

    #[test]
    fn var_clamp_and_comparisons_reject_cross_tape_and_non_broadcastable_shape() {
        // cross-tape は fail-closed（`check_same_tape`）。clamp は
        // 単項のため cross-tape の概念がなく、非 broadcast 可能
        // shape の検査対象外（比較演算のみ検証）。
        let tape_a = crate::tape::Tape::new_with_ops(crate::test_support::test_ops());
        let tape_b = crate::tape::Tape::new_with_ops(crate::test_support::test_ops());
        let a = tape_a.var(&t(&[1.0], &[1]));
        let b = tape_b.var(&t(&[1.0], &[1]));
        assert!(a.gt(&b).is_err());
        assert!(a.ge(&b).is_err());
        assert!(a.lt(&b).is_err());
        assert!(a.le(&b).is_err());
        assert!(a.eq(&b).is_err());
        assert!(a.ne(&b).is_err());

        // 非 broadcast 可能 shape も fail-closed（`broadcast_shape`）。
        let tape = crate::tape::Tape::new_with_ops(crate::test_support::test_ops());
        let x = tape.var(&t(&[1.0; 6], &[2, 3]));
        let y = tape.var(&t(&[1.0; 4], &[2, 2]));
        assert!(matches!(x.gt(&y).unwrap_err(), AutodiffError::Shape(_)));
        assert!(matches!(x.ge(&y).unwrap_err(), AutodiffError::Shape(_)));
        assert!(matches!(x.lt(&y).unwrap_err(), AutodiffError::Shape(_)));
        assert!(matches!(x.le(&y).unwrap_err(), AutodiffError::Shape(_)));
        assert!(matches!(x.eq(&y).unwrap_err(), AutodiffError::Shape(_)));
        assert!(matches!(x.ne(&y).unwrap_err(), AutodiffError::Shape(_)));
    }

    /// `Op::ScalarUnary`／`ScalarBinary` は `is_checkpoint_eligible ==
    /// false`（最小・安全側の判断。`tape.rs::Op` doc 参照）。
    #[test]
    fn scalar_unary_and_binary_ops_are_not_checkpoint_eligible() {
        assert!(
            !crate::tape::Op::ScalarUnary {
                op: ScalarUnaryOp::Relu,
                input: NodeId(0),
            }
            .is_checkpoint_eligible()
        );
        assert!(
            !crate::tape::Op::ScalarBinary {
                op: ScalarBinaryOp::Add,
                a: NodeId(0),
                b: NodeId(1),
            }
            .is_checkpoint_eligible()
        );
    }

    #[test]
    fn scalar_binary_rejects_non_broadcastable_shape_via_var() {
        let tape = crate::tape::Tape::new_with_ops(crate::test_support::test_ops());
        let a = tape.var(&t(&[1.0; 6], &[2, 3]));
        let b = tape.var(&t(&[1.0; 4], &[2, 2]));
        let err = a.scalar_binary(&b, ScalarBinaryOp::Add).unwrap_err();
        assert!(matches!(err, AutodiffError::Shape(_)));
    }

    // --- MatMul ---

    #[test]
    fn matmul_grad_matches_numeric() {
        let a = t(&[1.0, 2.0, -1.0, 0.5, 3.0, -2.0], &[2, 3]);
        let b = t(&[0.5, -1.0, 2.0, 1.0, -0.5, 1.5], &[3, 2]);
        let s = t(&[1.0, -0.5, 0.3, 2.0], &[2, 2]);

        let g = s.clone();
        let (da, db) = matmul_vjp(&test_ops(), &a, &b, &g).unwrap();

        let num_da = numeric_grad_unary(&a, &s, |x| eval::matmul(x, &b));
        let num_db = numeric_grad_unary(&b, &s, |x| eval::matmul(&a, x));

        assert_grad_close("matmul dA", &da, &num_da);
        assert_grad_close("matmul dB", &db, &num_db);
    }

    /// イシュー #1046 受け入れ条件 (a) の機械検証: `matmul_vjp` は
    /// `transpose2d`（zero-copy view）で作った転置オペランドを
    /// `eval::matmul` へ渡すが、`eval::matmul_operand` が
    /// `layout::classify_2d` で分類して直接読み出すため、ホスト側
    /// 転置コピー（`eval::MATMUL_HOST_REPACK_COUNT`）が発生しない
    /// ことを確認する。
    #[test]
    fn matmul_vjp_does_not_repack_transposed_operands() {
        let a = t(&[1.0, 2.0, -1.0, 0.5, 3.0, -2.0], &[2, 3]);
        let b = t(&[0.5, -1.0, 2.0, 1.0, -0.5, 1.5], &[3, 2]);
        let g = t(&[1.0, -0.5, 0.3, 2.0], &[2, 2]);

        let before = eval::MATMUL_HOST_REPACK_COUNT.with(|c| c.get());
        let _ = matmul_vjp(&test_ops(), &a, &b, &g).unwrap();
        let after = eval::MATMUL_HOST_REPACK_COUNT.with(|c| c.get());

        assert_eq!(
            before, after,
            "matmul_vjp: test_ops()（TestOps → eval::matmul 委譲。#1211 で \
             本番経路は BackendOps::gemm_fp32_strict 経由になったが、この \
             compat 経路の \
             ゼロコピー保証は変わらない）が転置オペランド（transpose2d の \
             zero-copy view）でホスト側転置コピーへフォールバックした \
             （MATMUL_HOST_REPACK_COUNT が増加した）"
        );
    }

    // --- Add（同 shape・bias broadcast・スカラー broadcast） ---

    #[test]
    fn add_grad_same_shape_matches_numeric() {
        let a = t(&[1.0, -2.0, 3.0, 0.5], &[2, 2]);
        let b = t(&[0.5, 1.5, -1.0, 2.0], &[2, 2]);
        let s = t(&[1.0, -1.0, 2.0, 0.5], &[2, 2]);

        let da = reduce_to_shape(&s, a.shape());
        let db = reduce_to_shape(&s, b.shape());
        let num_da = numeric_grad_unary(&a, &s, |x| eval::add(x, &b));
        let num_db = numeric_grad_unary(&b, &s, |x| eval::add(&a, x));

        assert_grad_close("add(same) dA", &da, &num_da);
        assert_grad_close("add(same) dB", &db, &num_db);
    }

    #[test]
    fn add_grad_bias_broadcast_matches_numeric() {
        let a = t(&[1.0, -2.0, 3.0, 0.5, -1.0, 2.0], &[2, 3]);
        let b = t(&[0.5, 1.5, -1.0], &[3]);
        let s = t(&[1.0, -1.0, 2.0, 0.5, 1.0, -0.5], &[2, 3]);

        let da = reduce_to_shape(&s, a.shape());
        let db = reduce_to_shape(&s, b.shape());
        let num_da = numeric_grad_unary(&a, &s, |x| eval::add(x, &b));
        let num_db = numeric_grad_unary(&b, &s, |x| eval::add(&a, x));

        assert_grad_close("add(bias) dA", &da, &num_da);
        assert_grad_close("add(bias) dB", &db, &num_db);
    }

    #[test]
    fn add_grad_scalar_broadcast_matches_numeric() {
        let a = t(&[1.0, -2.0, 3.0, 0.5], &[2, 2]);
        let b = t(&[2.0], &[]);
        let s = t(&[1.0, -1.0, 2.0, 0.5], &[2, 2]);

        let da = reduce_to_shape(&s, a.shape());
        let db = reduce_to_shape(&s, b.shape());
        let num_da = numeric_grad_unary(&a, &s, |x| eval::add(x, &b));
        let num_db = numeric_grad_unary(&b, &s, |x| eval::add(&a, x));

        assert_grad_close("add(scalar) dA", &da, &num_da);
        assert_grad_close("add(scalar) dB", &db, &num_db);
    }

    // --- Mul（同 shape・bias broadcast・スカラー broadcast） ---

    #[test]
    fn mul_grad_same_shape_matches_numeric() {
        let a = t(&[1.0, -2.0, 3.0, 0.5], &[2, 2]);
        let b = t(&[0.5, 1.5, -1.0, 2.0], &[2, 2]);
        let s = t(&[1.0, -1.0, 2.0, 0.5], &[2, 2]);

        let da = reduce_to_shape(&eval::mul(&s, &b), a.shape());
        let db = reduce_to_shape(&eval::mul(&s, &a), b.shape());
        let num_da = numeric_grad_unary(&a, &s, |x| eval::mul(x, &b));
        let num_db = numeric_grad_unary(&b, &s, |x| eval::mul(&a, x));

        assert_grad_close("mul(same) dA", &da, &num_da);
        assert_grad_close("mul(same) dB", &db, &num_db);
    }

    #[test]
    fn mul_grad_bias_broadcast_matches_numeric() {
        let a = t(&[1.0, -2.0, 3.0, 0.5, -1.0, 2.0], &[2, 3]);
        let b = t(&[0.5, 1.5, -1.0], &[3]);
        let s = t(&[1.0, -1.0, 2.0, 0.5, 1.0, -0.5], &[2, 3]);

        let da = reduce_to_shape(&eval::mul(&s, &b), a.shape());
        let db = reduce_to_shape(&eval::mul(&s, &a), b.shape());
        let num_da = numeric_grad_unary(&a, &s, |x| eval::mul(x, &b));
        let num_db = numeric_grad_unary(&b, &s, |x| eval::mul(&a, x));

        assert_grad_close("mul(bias) dA", &da, &num_da);
        assert_grad_close("mul(bias) dB", &db, &num_db);
    }

    #[test]
    fn mul_grad_scalar_broadcast_matches_numeric() {
        let a = t(&[1.0, -2.0, 3.0, 0.5], &[2, 2]);
        let b = t(&[2.0], &[]);
        let s = t(&[1.0, -1.0, 2.0, 0.5], &[2, 2]);

        let da = reduce_to_shape(&eval::mul(&s, &b), a.shape());
        let db = reduce_to_shape(&eval::mul(&s, &a), b.shape());
        let num_da = numeric_grad_unary(&a, &s, |x| eval::mul(x, &b));
        let num_db = numeric_grad_unary(&b, &s, |x| eval::mul(&a, x));

        assert_grad_close("mul(scalar) dA", &da, &num_da);
        assert_grad_close("mul(scalar) dB", &db, &num_db);
    }

    // --- Relu（正負混在。|x| >= 10h でキンク回避） ---

    #[test]
    fn relu_grad_matches_numeric() {
        let a = t(&[2.0, -3.0, 0.5, -0.02, 1.5, -1.5], &[2, 3]);
        let s = t(&[1.0, -1.0, 2.0, 0.5, -0.5, 1.0], &[2, 3]);

        let g = s.clone();
        let da = elementwise_mul_mask(&g, &a, |v| v > 0.0);
        let num_da = numeric_grad_unary(&a, &s, eval::relu);

        assert_grad_close("relu dA", &da, &num_da);
    }

    #[test]
    fn relu_subgradient_at_zero_is_zero() {
        // x = 0 における劣勾配は 0 とする（PoC-v2-2 準拠。中央差分は
        // キンクで数値的に不安定なため、ここは解析式の直接検証のみ）。
        let a = t(&[0.0], &[1]);
        let g = t(&[3.0], &[1]);
        let da = elementwise_mul_mask(&g, &a, |v| v > 0.0);
        assert_eq!(dense_vec(&da), vec![0.0]);
    }

    // --- イシュー #1577: elementwise_mul_mask の stride 対応 ---
    //
    // 新実装（`try_elementwise_mul_mask_strided` を経由する
    // `elementwise_mul_mask`）と、旧実装をそのまま残した参照実装
    // （`dense_vec` を zip するだけの経路）の出力を `to_bits()` で
    // 完全一致比較する。数値的に同一の値を出す契約（bit 同一）を
    // 直接検証する。

    /// `dense_vec` 経由の参照実装（旧 `elementwise_mul_mask` そのもの）。
    /// 新実装との bit 同一性を突き合わせる基準として使う。
    fn elementwise_mul_mask_reference(
        g: &Tensor<f32>,
        mask_src: &Tensor<f32>,
        keep: impl Fn(f32) -> bool,
    ) -> Tensor<f32> {
        let shape = g.shape().to_vec();
        let g_data = dense_vec(g);
        let mask_data = dense_vec(mask_src);
        let out: Vec<f32> = g_data
            .iter()
            .zip(mask_data.iter())
            .map(|(&gv, &mv)| if keep(mv) { gv } else { 0.0 })
            .collect();
        build_tensor(out, &shape)
    }

    fn assert_bits_eq(label: &str, actual: &Tensor<f32>, expected: &Tensor<f32>) {
        let a = dense_vec(actual);
        let e = dense_vec(expected);
        assert_eq!(a.len(), e.len(), "{label}: 要素数不一致");
        for (i, (&av, &ev)) in a.iter().zip(e.iter()).enumerate() {
            assert_eq!(
                av.to_bits(),
                ev.to_bits(),
                "{label}[{i}]: actual={av:?}（bits={:#x}） expected={ev:?}（bits={:#x}）",
                av.to_bits(),
                ev.to_bits()
            );
        }
    }

    #[test]
    fn mask_stride_transpose_view_matches_reference() {
        // reuse backward が生む `d_input = transpose2d(&tmp)` を再現
        // （`tmp: [k, m]` 連続 → 転置後 `[m, k]`・strides `[1, k]`）。
        let tmp = t(&[1.0, -2.0, 3.0, -4.0, 5.0, -6.0], &[2, 3]);
        let g = transpose2d(&tmp); // shape [3, 2]、strides [1, 3]
        let mask_src = t(&[1.0, -1.0, 0.0, 2.0, -2.0, 0.5], &[3, 2]);

        let actual = elementwise_mul_mask(&g, &mask_src, |v| v > 0.0);
        let expected = elementwise_mul_mask_reference(&g, &mask_src, |v| v > 0.0);
        assert_bits_eq("transpose view (g 側)", &actual, &expected);
    }

    #[test]
    fn mask_stride_mask_src_side_non_contiguous() {
        // `mask_src` 側だけが非連続（`out_value` が view の場合の
        // 想定。`g` は連続）。
        let g = t(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[3, 2]);
        let mask_tmp = t(&[1.0, -1.0, 0.0, 2.0, -2.0, 0.5], &[2, 3]);
        let mask_src = transpose2d(&mask_tmp); // shape [3, 2]

        let actual = elementwise_mul_mask(&g, &mask_src, |v| v > 0.0);
        let expected = elementwise_mul_mask_reference(&g, &mask_src, |v| v > 0.0);
        assert_bits_eq("transpose view (mask_src 側)", &actual, &expected);
    }

    #[test]
    fn mask_stride_narrow_offset_view() {
        // `narrow` 後の view（offset != 0・かつ真に非連続）。列方向
        // （dim 1）の `narrow` は、行方向（dim 0）の `narrow` と異なり
        // 元の行幅（stride 4）が残ったまま shape が縮む（`[3,4]` の
        // 列 1..3 を切り出すと shape `[3,2]`・strides `[4,1]` となり、
        // 新 shape の標準行優先 stride `[2,1]` とは一致しない）ため
        // `as_slice()` が `None` を返す（`Tensor::is_contiguous` 契約）。
        // 行方向の `narrow` は新 shape でも標準行優先 stride のまま
        // 残り `as_slice()` が成功してしまう（`Contig` 分類）ため、
        // 本テストの意図（`View` 分類・rank-2 `Contig`×`View` 専用
        // 経路のオフセット付きケース）を検証するには列方向でなければ
        // ならない（advisor 指摘。行方向版は誤って `Contig`×`Contig`
        // 高速経路しか検証していなかった）。
        let base = t(
            &[
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
            ],
            &[3, 4],
        );
        let g = base
            .narrow(1, 1, 2)
            .expect("narrow: 事前に範囲内であることを確認済み");
        assert!(
            g.as_slice().is_none(),
            "narrow(dim=1) は非連続 view のはず（本テストが検証したい前提。\
release ビルドの `cargo test --release` でも前提崩れを検知できるよう \
`debug_assert!` ではなく `assert!` を使う）"
        );
        let mask_src = t(&[-1.0, 1.0, 0.0, -2.0, 3.0, -3.0], &[3, 2]);

        let actual = elementwise_mul_mask(&g, &mask_src, |v| v > 0.0);
        let expected = elementwise_mul_mask_reference(&g, &mask_src, |v| v > 0.0);
        assert_bits_eq("narrow(dim=1) view", &actual, &expected);
    }

    #[test]
    fn mask_stride_both_operands_transposed_rank2() {
        // `g`・`mask_src` の両方が非連続 view（`View`×`View`）の
        // rank-2 ケース。rank-2 専用経路のうち「片方 `Contig`・片方
        // `View`」の 2 分岐（advisor 指摘で追加）のどちらにも該当
        // しないため、`try_elementwise_mul_mask_strided` 内の
        // 一般化した `read(idx, flat)` 経由の rank-2 経路（`MaskReadOperand::
        // read` の `View` アーム）を確実に踏む。
        let tmp_g = t(&[1.0, -2.0, 3.0, -4.0, 5.0, -6.0], &[2, 3]);
        let g = transpose2d(&tmp_g); // shape [3, 2]、非連続 view
        let tmp_mask = t(&[1.0, -1.0, 0.0, 2.0, -2.0, 0.5], &[2, 3]);
        let mask_src = transpose2d(&tmp_mask); // shape [3, 2]、非連続 view
        assert!(
            g.as_slice().is_none() && mask_src.as_slice().is_none(),
            "両オペランドとも非連続 view のはず（本テストが検証したい前提。\
release ビルドでも検知できるよう `assert!` を使う）"
        );

        let actual = elementwise_mul_mask(&g, &mask_src, |v| v > 0.0);
        let expected = elementwise_mul_mask_reference(&g, &mask_src, |v| v > 0.0);
        assert_bits_eq("View×View rank-2", &actual, &expected);
    }

    #[test]
    fn mask_stride_broadcast_zero_stride_view() {
        // `broadcast_to` が生む stride 0 の軸を含む view。
        let row = t(&[1.0, -1.0, 2.0], &[1, 3]);
        let g = row
            .broadcast_to(&[2, 3])
            .expect("broadcast_to: shape 互換性は事前に確認済み");
        let mask_src = t(&[1.0, -1.0, 1.0, -1.0, 1.0, -1.0], &[2, 3]);

        let actual = elementwise_mul_mask(&g, &mask_src, |v| v > 0.0);
        let expected = elementwise_mul_mask_reference(&g, &mask_src, |v| v > 0.0);
        assert_bits_eq("broadcast (stride 0) view", &actual, &expected);
    }

    #[test]
    fn mask_stride_rank1_and_rank3() {
        // rank-1（transpose2d 適用対象外だが narrow で非連続を作る）。
        let base1 = t(&[1.0, 2.0, 3.0, 4.0, 5.0], &[5]);
        let g1 = base1
            .narrow(0, 1, 3)
            .expect("narrow: 事前に範囲内であることを確認済み");
        let mask1 = t(&[-1.0, 1.0, -1.0], &[3]);
        let actual1 = elementwise_mul_mask(&g1, &mask1, |v| v > 0.0);
        let expected1 = elementwise_mul_mask_reference(&g1, &mask1, |v| v > 0.0);
        assert_bits_eq("rank-1 narrow", &actual1, &expected1);

        // rank-3: 2x2x3 を transpose(0, 2) で非連続にする。
        let base3 = t(
            &[
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
            ],
            &[2, 2, 3],
        );
        let g3 = base3
            .transpose(0, 2)
            .expect("transpose: rank-3 は 0,2 とも範囲内");
        let mask3 = t(
            &[
                1.0, -1.0, 0.0, 2.0, -2.0, 0.5, -0.5, 1.5, -1.5, 3.0, -3.0, 0.25,
            ],
            &[3, 2, 2],
        );
        let actual3 = elementwise_mul_mask(&g3, &mask3, |v| v > 0.0);
        let expected3 = elementwise_mul_mask_reference(&g3, &mask3, |v| v > 0.0);
        assert_bits_eq("rank-3 transpose", &actual3, &expected3);
    }

    #[test]
    fn mask_stride_empty_tensor() {
        let g = t(&[], &[0, 3]);
        let mask_src = t(&[], &[0, 3]);
        let actual = elementwise_mul_mask(&g, &mask_src, |v| v > 0.0);
        assert_eq!(dense_vec(&actual), Vec::<f32>::new());
    }

    #[test]
    fn mask_stride_nan_and_signed_zero_and_subnormal() {
        // NaN（マスク不成立で 0 を返す規約）・-0.0・subnormal を含む
        // 転置 view で bit 同一性を確認する。
        let tmp = t(
            &[f32::NAN, -0.0, f32::MIN_POSITIVE / 2.0, 1.0, -1.0, 0.0],
            &[2, 3],
        );
        let g = transpose2d(&tmp); // shape [3, 2]
        let mask_src = t(&[1.0, f32::NAN, -0.0, 1.0, 0.0, -1.0], &[3, 2]);

        let actual = elementwise_mul_mask(&g, &mask_src, |v| v > 0.0);
        let expected = elementwise_mul_mask_reference(&g, &mask_src, |v| v > 0.0);
        assert_bits_eq("NaN / -0.0 / subnormal", &actual, &expected);
    }

    /// マイクロベンチ（`#[ignore]`。手動実行専用。
    /// `docs/perf/lowlayer-diagnosis-2026-09-12.md` §4 の
    /// `diag_elementwise_mask_bench` と同構成。64×256 の連続入力 と
    /// `[256,64]→transpose2d` の非連続転置 view を 1000 回反復した
    /// 中央値を、新実装（`elementwise_mul_mask`）・旧参照実装
    /// （`elementwise_mul_mask_reference`。`dense_vec` zip 経路）の
    /// 双方・連続／非連続の計 4 系列で比較する。stderr 出力のみで
    /// assert は行わない（実測記録は `docs/perf/
    /// train-reuse-relu-mask-stride.md`）。
    /// 実行例:
    /// `cargo test -p fandhe-ai-autodiff --release -- --ignored
    /// --nocapture mask_stride_microbench`
    #[test]
    #[ignore = "手動実行専用のマイクロベンチ（stderr 出力のみ）"]
    fn mask_stride_microbench() {
        use std::time::Instant;

        const ROWS: usize = 64;
        const COLS: usize = 256;
        const ITERS: usize = 1000;

        let contiguous_data: Vec<f32> = (0..ROWS * COLS).map(|i| ((i % 7) as f32) - 3.0).collect();
        let contiguous = t(&contiguous_data, &[ROWS, COLS]);
        let mask_contig = t(&contiguous_data, &[ROWS, COLS]);

        let transposed_src_data: Vec<f32> =
            (0..COLS * ROWS).map(|i| ((i % 7) as f32) - 3.0).collect();
        let transposed_src = t(&transposed_src_data, &[COLS, ROWS]);
        let non_contig = transpose2d(&transposed_src); // shape [ROWS, COLS]
        let mask_non_contig = t(&contiguous_data, &[ROWS, COLS]);

        let mut contig_times = Vec::with_capacity(ITERS);
        for _ in 0..ITERS {
            let start = Instant::now();
            let out = elementwise_mul_mask(&contiguous, &mask_contig, |v| v > 0.0);
            std::hint::black_box(&out);
            contig_times.push(start.elapsed());
        }
        let mut non_contig_times = Vec::with_capacity(ITERS);
        for _ in 0..ITERS {
            let start = Instant::now();
            let out = elementwise_mul_mask(&non_contig, &mask_non_contig, |v| v > 0.0);
            std::hint::black_box(&out);
            non_contig_times.push(start.elapsed());
        }
        // 旧実装（`dense_vec` zip 経路）との対照。新実装の連続経路が
        // 旧実装の連続経路を大きく下回っていないか（退行していないか）
        // を直接確認するための参考値。
        let mut reference_contig_times = Vec::with_capacity(ITERS);
        for _ in 0..ITERS {
            let start = Instant::now();
            let out = elementwise_mul_mask_reference(&contiguous, &mask_contig, |v| v > 0.0);
            std::hint::black_box(&out);
            reference_contig_times.push(start.elapsed());
        }
        let mut reference_non_contig_times = Vec::with_capacity(ITERS);
        for _ in 0..ITERS {
            let start = Instant::now();
            let out = elementwise_mul_mask_reference(&non_contig, &mask_non_contig, |v| v > 0.0);
            std::hint::black_box(&out);
            reference_non_contig_times.push(start.elapsed());
        }

        contig_times.sort();
        non_contig_times.sort();
        reference_contig_times.sort();
        reference_non_contig_times.sort();
        eprintln!(
            "mask_stride_microbench: contiguous median={:?} non_contiguous(transpose view) median={:?} reference_contiguous(dense_vec zip) median={:?} reference_non_contiguous(dense_vec zip) median={:?}",
            contig_times[ITERS / 2],
            non_contig_times[ITERS / 2],
            reference_contig_times[ITERS / 2],
            reference_non_contig_times[ITERS / 2]
        );
    }

    // --- Exp ---

    #[test]
    fn exp_grad_matches_numeric() {
        let a = t(&[0.5, -1.0, 1.5, -0.3], &[2, 2]);
        let s = t(&[1.0, -1.0, 0.5, 2.0], &[2, 2]);

        let out_value = eval::exp(&a);
        let g = s.clone();
        let da = eval::mul(&g, &out_value);
        let num_da = numeric_grad_unary(&a, &s, eval::exp);

        assert_grad_close("exp dA", &da, &num_da);
    }

    // --- Tanh ---

    #[test]
    fn tanh_grad_matches_numeric() {
        let a = t(&[0.5, -1.0, 1.5, -0.3], &[2, 2]);
        let s = t(&[1.0, -1.0, 0.5, 2.0], &[2, 2]);

        let out_value = eval::tanh(&a);
        let g = s.clone();
        let factor = tanh_grad_factor(&out_value);
        let da = eval::mul(&g, &factor);
        let num_da = numeric_grad_unary(&a, &s, eval::tanh);

        assert_grad_close("tanh dA", &da, &num_da);
    }

    // --- Sigmoid（飽和域を含む） ---

    #[test]
    fn sigmoid_grad_matches_numeric() {
        let a = t(&[0.5, -1.0, 1.5, -0.3], &[2, 2]);
        let s = t(&[1.0, -1.0, 0.5, 2.0], &[2, 2]);

        let out_value = eval::sigmoid(&a);
        let g = s.clone();
        let factor = sigmoid_grad_factor(&out_value);
        let da = eval::mul(&g, &factor);
        let num_da = numeric_grad_unary(&a, &s, eval::sigmoid);

        assert_grad_close("sigmoid dA", &da, &num_da);
    }

    #[test]
    fn sigmoid_grad_saturated_region_matches_numeric() {
        // |x| が大きい飽和域（勾配 ≈ 0）でも中央差分と一致することを
        // 確認する（`eval::sigmoid` の数値安定形が飽和域で NaN/Inf を
        // 出さないことの間接検証も兼ねる）。
        let a = t(&[8.0, -8.0, 15.0, -15.0], &[2, 2]);
        let s = t(&[1.0, -1.0, 0.5, 2.0], &[2, 2]);

        let out_value = eval::sigmoid(&a);
        let g = s.clone();
        let factor = sigmoid_grad_factor(&out_value);
        let da = eval::mul(&g, &factor);
        let num_da = numeric_grad_unary(&a, &s, eval::sigmoid);

        assert_grad_close("sigmoid(saturated) dA", &da, &num_da);
    }

    // --- Sum（dim: None / Some(0) / Some(1)） ---

    #[test]
    fn sum_grad_dim_none_matches_numeric() {
        let a = t(&[1.0, -2.0, 3.0, 0.5, -1.0, 2.0], &[2, 3]);
        let s = t(&[2.0], &[]);

        let g = s.clone();
        let da = unreduce_broadcast(&g, a.shape(), None);
        let num_da = numeric_grad_unary(&a, &s, |x| eval::sum(x, None, &[]));

        assert_grad_close("sum(None) dA", &da, &num_da);
    }

    #[test]
    fn sum_grad_dim_0_matches_numeric() {
        let a = t(&[1.0, -2.0, 3.0, 0.5, -1.0, 2.0], &[2, 3]);
        let s = t(&[1.0, -1.0, 2.0], &[3]);

        let g = s.clone();
        let da = unreduce_broadcast(&g, a.shape(), Some(0));
        let num_da = numeric_grad_unary(&a, &s, |x| eval::sum(x, Some(0), &[3]));

        assert_grad_close("sum(dim=0) dA", &da, &num_da);
    }

    #[test]
    fn sum_grad_dim_1_matches_numeric() {
        let a = t(&[1.0, -2.0, 3.0, 0.5, -1.0, 2.0], &[2, 3]);
        let s = t(&[1.0, -1.0], &[2]);

        let g = s.clone();
        let da = unreduce_broadcast(&g, a.shape(), Some(1));
        let num_da = numeric_grad_unary(&a, &s, |x| eval::sum(x, Some(1), &[2]));

        assert_grad_close("sum(dim=1) dA", &da, &num_da);
    }

    // --- Max（dim: None / Some(0) / Some(1)。同値タイなし） ---

    #[test]
    fn max_grad_dim_none_matches_numeric() {
        let a = t(&[1.0, -2.0, 5.0, 0.5, -1.0, 2.0], &[2, 3]);
        let s = t(&[2.0], &[]);

        let out_value = eval::max(&a, None, &[]);
        let g = s.clone();
        let da = max_vjp(&a, None, &out_value, &g);
        let num_da = numeric_grad_unary(&a, &s, |x| eval::max(x, None, &[]));

        assert_grad_close("max(None) dA", &da, &num_da);
    }

    #[test]
    fn max_grad_dim_0_matches_numeric() {
        let a = t(&[1.0, -2.0, 5.0, 0.5, -1.0, 2.0], &[2, 3]);
        let s = t(&[1.0, -1.0, 2.0], &[3]);

        let out_value = eval::max(&a, Some(0), &[3]);
        let g = s.clone();
        let da = max_vjp(&a, Some(0), &out_value, &g);
        let num_da = numeric_grad_unary(&a, &s, |x| eval::max(x, Some(0), &[3]));

        assert_grad_close("max(dim=0) dA", &da, &num_da);
    }

    #[test]
    fn max_grad_dim_1_matches_numeric() {
        let a = t(&[1.0, -2.0, 5.0, 0.5, -1.0, 2.0], &[2, 3]);
        let s = t(&[1.0, -1.0], &[2]);

        let out_value = eval::max(&a, Some(1), &[2]);
        let g = s.clone();
        let da = max_vjp(&a, Some(1), &out_value, &g);
        let num_da = numeric_grad_unary(&a, &s, |x| eval::max(x, Some(1), &[2]));

        assert_grad_close("max(dim=1) dA", &da, &num_da);
    }

    // --- Max（同値タイ。#224: 先勝ち決定的挙動の回帰固定） ---
    //
    // タイ発生時は最大値位置が複数あり数値微分（中央差分）が定義でき
    // ないため、上記の同値タイなしケースとは異なり厳密値アサーション
    // （数値微分比較なし）で「最初に現れる最大要素 1 箇所のみに勾配が
    // 伝播し、他はゼロになる」先勝ち挙動そのものを固定する。

    #[test]
    fn max_grad_dim_none_tie_first_wins() {
        // 最大値 5.0 がインデックス 1・3 の 2 箇所に現れるタイケース。
        let a = t(&[1.0, 5.0, 3.0, 5.0], &[4]);
        let g = t(&[2.0], &[]);

        let out_value = eval::max(&a, None, &[]);
        let da = max_vjp(&a, None, &out_value, &g);
        let grad = dense_vec(&da);

        assert_eq!(
            grad,
            vec![0.0, 2.0, 0.0, 0.0],
            "max(None) タイ時は最初に現れる最大要素（idx=1）のみへ伝播するはず"
        );
        // 勾配総量が上流勾配 g と一致すること（先勝ちでも保存量は保たれる）。
        assert_eq!(grad.iter().sum::<f32>(), 2.0);
    }

    #[test]
    fn max_grad_dim_axis_tie_first_wins() {
        // shape [2, 3]。行 0 は列 0・2 が 5.0 でタイ、行 1 はタイなし。
        let a = t(&[5.0, 1.0, 5.0, 1.0, -2.0, 4.0], &[2, 3]);
        let g = t(&[3.0, 7.0], &[2]);

        let out_value = eval::max(&a, Some(1), &[2]);
        let da = max_vjp(&a, Some(1), &out_value, &g);
        let grad = dense_vec(&da);

        assert_eq!(
            grad,
            vec![3.0, 0.0, 0.0, 0.0, 0.0, 7.0],
            "max(dim=1) タイ行（行 0）は軸方向で最初の最大要素（列 0）のみへ伝播するはず"
        );
        // 各 (outer) スライスごとに勾配総量が上流勾配 g[outer] と一致すること。
        assert_eq!(grad[0..3].iter().sum::<f32>(), 3.0);
        assert_eq!(grad[3..6].iter().sum::<f32>(), 7.0);
    }

    // --- Min（dim: None / Some(0) / Some(1)。同値タイなし。イシュー
    // #1720。`extremum_first_match_vjp` を `Max` と共有する回帰） ---

    #[test]
    fn min_grad_dim_none_matches_numeric() {
        let a = t(&[1.0, -2.0, 5.0, 0.5, -1.0, 2.0], &[2, 3]);
        let s = t(&[2.0], &[]);

        let out_value = eval::min(&a, None, &[]);
        let g = s.clone();
        let da = extremum_first_match_vjp(&a, None, &out_value, &g);
        let num_da = numeric_grad_unary(&a, &s, |x| eval::min(x, None, &[]));

        assert_grad_close("min(None) dA", &da, &num_da);
    }

    #[test]
    fn min_grad_dim_0_matches_numeric() {
        let a = t(&[1.0, -2.0, 5.0, 0.5, -1.0, 2.0], &[2, 3]);
        let s = t(&[1.0, -1.0, 2.0], &[3]);

        let out_value = eval::min(&a, Some(0), &[3]);
        let g = s.clone();
        let da = extremum_first_match_vjp(&a, Some(0), &out_value, &g);
        let num_da = numeric_grad_unary(&a, &s, |x| eval::min(x, Some(0), &[3]));

        assert_grad_close("min(dim=0) dA", &da, &num_da);
    }

    #[test]
    fn min_grad_dim_1_matches_numeric() {
        let a = t(&[1.0, -2.0, 5.0, 0.5, -1.0, 2.0], &[2, 3]);
        let s = t(&[1.0, -1.0], &[2]);

        let out_value = eval::min(&a, Some(1), &[2]);
        let g = s.clone();
        let da = extremum_first_match_vjp(&a, Some(1), &out_value, &g);
        let num_da = numeric_grad_unary(&a, &s, |x| eval::min(x, Some(1), &[2]));

        assert_grad_close("min(dim=1) dA", &da, &num_da);
    }

    // --- Min（同値タイ。先勝ち決定的挙動の回帰固定。`Max` と対称） ---

    #[test]
    fn min_grad_dim_none_tie_first_wins() {
        // 最小値 -2.0 がインデックス 1・3 の 2 箇所に現れるタイケース。
        let a = t(&[1.0, -2.0, 3.0, -2.0], &[4]);
        let g = t(&[2.0], &[]);

        let out_value = eval::min(&a, None, &[]);
        let da = extremum_first_match_vjp(&a, None, &out_value, &g);
        let grad = dense_vec(&da);

        assert_eq!(
            grad,
            vec![0.0, 2.0, 0.0, 0.0],
            "min(None) タイ時は最初に現れる最小要素（idx=1）のみへ伝播するはず"
        );
        assert_eq!(grad.iter().sum::<f32>(), 2.0);
    }

    #[test]
    fn min_grad_dim_axis_tie_first_wins() {
        // shape [2, 3]。行 0 は列 0・2 が -5.0 でタイ、行 1 はタイなし。
        let a = t(&[-5.0, 1.0, -5.0, 1.0, -2.0, 4.0], &[2, 3]);
        let g = t(&[3.0, 7.0], &[2]);

        let out_value = eval::min(&a, Some(1), &[2]);
        let da = extremum_first_match_vjp(&a, Some(1), &out_value, &g);
        let grad = dense_vec(&da);

        assert_eq!(
            grad,
            vec![3.0, 0.0, 0.0, 0.0, 7.0, 0.0],
            "min(dim=1) タイ行（行 0）は軸方向で最初の最小要素（列 0）のみへ伝播するはず"
        );
        assert_eq!(grad[0..3].iter().sum::<f32>(), 3.0);
        assert_eq!(grad[3..6].iter().sum::<f32>(), 7.0);
    }

    // --- MseLoss（pred/target 両勾配） ---

    #[test]
    fn mse_loss_grad_mean_matches_numeric() {
        let pred = t(&[1.0, -2.0, 3.0, 0.5], &[2, 2]);
        let target = t(&[0.5, -1.0, 2.5, 1.0], &[2, 2]);
        let s = t(&[3.0], &[]);

        let g = s.clone();
        let (dpred, dtarget) = mse_loss_vjp(&pred, &target, &g, Reduction::Mean);
        let num_dpred =
            numeric_grad_unary(&pred, &s, |x| eval::mse_loss(x, &target, Reduction::Mean));
        let num_dtarget =
            numeric_grad_unary(&target, &s, |x| eval::mse_loss(&pred, x, Reduction::Mean));

        assert_grad_close("mse(mean) dPred", &dpred, &num_dpred);
        assert_grad_close("mse(mean) dTarget", &dtarget, &num_dtarget);
    }

    #[test]
    fn mse_loss_grad_sum_matches_numeric() {
        // sum 縮約（#190）。scale が `2/n` ではなく `2` になる分岐を
        // mean と同じ数値微分ハーネスで検証する。
        let pred = t(&[1.0, -2.0, 3.0, 0.5], &[2, 2]);
        let target = t(&[0.5, -1.0, 2.5, 1.0], &[2, 2]);
        let s = t(&[3.0], &[]);

        let g = s.clone();
        let (dpred, dtarget) = mse_loss_vjp(&pred, &target, &g, Reduction::Sum);
        let num_dpred =
            numeric_grad_unary(&pred, &s, |x| eval::mse_loss(x, &target, Reduction::Sum));
        let num_dtarget =
            numeric_grad_unary(&target, &s, |x| eval::mse_loss(&pred, x, Reduction::Sum));

        assert_grad_close("mse(sum) dPred", &dpred, &num_dpred);
        assert_grad_close("mse(sum) dTarget", &dtarget, &num_dtarget);
    }

    #[test]
    fn mse_loss_grad_n_zero_is_zero() {
        // numel() == 0 はゼロ除算を避け zeros を返す（ガード条件の
        // 直接検証。中央差分は空テンソルに対して定義できないため
        // 解析式のみで確認する）。mean/sum いずれも同じ早期 return
        // 経路（`n == 0` 分岐）を通るため mean のみ代表して検証する。
        let pred = build_tensor(Vec::new(), &[0]);
        let target = build_tensor(Vec::new(), &[0]);
        let g = t(&[1.0], &[]);
        let (dpred, dtarget) = mse_loss_vjp(&pred, &target, &g, Reduction::Mean);
        assert!(dense_vec(&dpred).is_empty());
        assert!(dense_vec(&dtarget).is_empty());
    }

    // --- CrossEntropyLoss（#191。PyTorch 参照値との突合は
    //     `tests/nn_cross_entropy.rs`、ここでは既存の
    //     `numeric_grad_unary`/`assert_grad_close`〈中央差分〉基盤に
    //     揃えた eval レベルの grad check を行う） ---

    #[test]
    fn cross_entropy_loss_grad_matches_numeric() {
        let logits = t(&[1.0, -2.0, 3.0, 0.5, -1.0, 2.0], &[2, 3]);
        let targets = fandhe_ai_tensor_core::Tensor::new(vec![2i32, 0], &[2])
            .expect("test fixture: shape とデータ長は事前に一致させている");
        // forward 出力は既に scalar shape [] のため、`s` も scalar
        // （`mse_loss_grad_matches_numeric` と同じ「射影 s がスカラー」
        // パターン）。
        let s = t(&[3.0], &[]);

        let g = s.clone();
        let dlogits = cross_entropy_loss_vjp(&logits, &targets, 1, Reduction::Mean, &g);
        let num_dlogits = numeric_grad_unary(&logits, &s, |x| {
            eval::cross_entropy_loss(x, &targets, 1, Reduction::Mean)
        });

        assert_grad_close("cross_entropy_loss(mean) dLogits", &dlogits, &num_dlogits);
    }

    #[test]
    fn cross_entropy_loss_grad_sum_matches_numeric() {
        let logits = t(&[1.0, -2.0, 3.0, 0.5, -1.0, 2.0], &[2, 3]);
        let targets = fandhe_ai_tensor_core::Tensor::new(vec![2i32, 0], &[2])
            .expect("test fixture: shape とデータ長は事前に一致させている");
        let s = t(&[3.0], &[]);

        let g = s.clone();
        let dlogits = cross_entropy_loss_vjp(&logits, &targets, 1, Reduction::Sum, &g);
        let num_dlogits = numeric_grad_unary(&logits, &s, |x| {
            eval::cross_entropy_loss(x, &targets, 1, Reduction::Sum)
        });

        assert_grad_close("cross_entropy_loss(sum) dLogits", &dlogits, &num_dlogits);
    }

    // --- reduce_to_shape（中間軸縮約。ランク同一で先頭・末尾以外の
    //     軸を broadcast 元へ潰す経路。add/mul の bias broadcast テスト
    //     は末尾軸・スカラーテストは rank 0 のみで、この経路は未カバー
    //     だった） ---

    #[test]
    fn reduce_to_shape_middle_axis_matches_numeric() {
        // g: [2,4,3] を target_shape [2,1,3]（中間軸 dim=1 が size 1）
        // へ縮約する。add(a, b) の b 側勾配として同じ経路を通す
        // （a: [2,4,3]、b: [2,1,3] からの broadcast）。
        let a = t(
            &[
                1.0, -2.0, 3.0, 0.5, -1.0, 2.0, 1.5, -0.5, 2.5, -1.5, 0.5, -2.5, 3.0, -1.0, 0.5,
                -0.5, 1.0, -1.5, 2.0, -2.0, 0.5, -0.5, 1.5, -1.0,
            ],
            &[2, 4, 3],
        );
        let b = t(&[0.5, -1.0, 2.0, 1.0, -0.5, 1.5], &[2, 1, 3]);
        let s = t(
            &[
                1.0, -1.0, 2.0, 0.5, 1.0, -0.5, 2.0, -2.0, 0.5, -0.5, 1.5, -1.0, 0.2, -0.2, 0.4,
                0.1, 0.2, -0.1, 0.4, -0.4, 0.1, -0.1, 0.3, -0.2,
            ],
            &[2, 4, 3],
        );

        let db = reduce_to_shape(&s, b.shape());
        let num_db = numeric_grad_unary(&b, &s, |x| eval::add(&a, x));

        assert_grad_close("reduce_to_shape(middle axis) dB", &db, &num_db);
    }

    // --- reduce_bias_grad（イシュー #1566・PR #1659 codex-review P2 是正） ---

    /// `reduce_bias_grad` が「軸 0（行）方向の bias 縮約」用の f64 経路
    /// （`eval::reduce_bias_grad_rows`）を、**軸 1（列）方向の
    /// broadcast bias**（`target_shape` の末尾次元が `g` の列数と
    /// 一致しない形状。例: `g: [2, 2]` に対する `target_shape: [2, 1]`）
    /// へ誤って適用しないことを確認する回帰テスト（codex-review 指摘。
    /// `[2, 1]` は総要素数が `2` で `g` の列数 `2` と偶然一致するため、
    /// 総要素数のみで判定する実装だと誤って行縮約の f64 経路へ分岐し
    /// てしまっていた）。
    #[test]
    fn reduce_bias_grad_does_not_misapply_row_reduction_to_column_broadcast_bias() {
        // g = [[1, 2], [3, 4]]（行優先）。target_shape = [2, 1] は
        // 各行の bias 値が 2 列へ複製される broadcast（軸 1 縮約）。
        // 正しい勾配は行ごとの和: row0 = 1+2=3・row1 = 3+4=7。
        // 行縮約（列ごとの和 col0=1+3=4・col1=2+4=6）を誤って適用すると
        // 全く異なる値になる。
        let g = t(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);

        let got = reduce_bias_grad(&g, &[2, 1]);
        let expected = reduce_to_shape(&g, &[2, 1]);

        assert_eq!(got.shape(), &[2, 1]);
        assert_eq!(
            got.contiguous().as_slice().unwrap(),
            expected.contiguous().as_slice().unwrap(),
            "reduce_bias_grad は軸 1 縮約（[2, 1]）には reduce_to_shape をそのまま使う              はず（f64 行縮約経路を誤適用してはいけない）"
        );
        assert_eq!(
            got.contiguous().as_slice().unwrap(),
            &[3.0f32, 7.0],
            "軸 1 縮約の正しい値（行ごとの和）と一致するはず"
        );
    }

    /// 対照: 軸 0（行）方向の標準的な bias 縮約（`target_shape` の末尾
    /// 次元が `g` の列数と一致し、それより前の次元がすべて `1`）は
    /// 引き続き `eval::reduce_bias_grad_rows`（f64 経路）へ委譲される
    /// ことを、`[n]`・`[1, n]` の両形状で確認する（値は
    /// `reduce_to_shape`〈こちらは `f32` 逐次和〉と一致する範囲——
    /// 相殺による桁落ちがない入力なので両経路の値自体は一致するが、
    /// 経路選択の正しさを shape 網羅で確認する意図）。
    #[test]
    fn reduce_bias_grad_applies_row_reduction_for_rank1_and_leading_one_targets() {
        let g = t(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);

        let got_rank1 = reduce_bias_grad(&g, &[2]);
        assert_eq!(got_rank1.shape(), &[2]);
        assert_eq!(
            got_rank1.contiguous().as_slice().unwrap(),
            &[4.0f32, 6.0],
            "target_shape=[2] は列ごとの和（col0=1+3=4・col1=2+4=6）のはず"
        );

        let got_leading_one = reduce_bias_grad(&g, &[1, 2]);
        assert_eq!(got_leading_one.shape(), &[1, 2]);
        assert_eq!(
            got_leading_one.contiguous().as_slice().unwrap(),
            &[4.0f32, 6.0],
            "target_shape=[1, 2] も同じ列ごとの和になるはず（先頭次元 1 個の reshape）"
        );
    }

    // --- vjp() ディスパッチの疎通確認（#18 との継ぎ目契約） ---
    //
    // Low 指摘: MatMul のみが vjp() 経由で疎通確認されており、他の
    // 8 演算（Add/Mul/Relu/Exp/Tanh/Sum/Max/MseLoss）は内部ヘルパーを
    // 直接呼ぶ形でしか検証されていなかった。各 match アームの配線
    // （`nodes[a.0]`/`nodes[b.0]` の対応順序）を通しで検証するため、
    // 全 9 演算（Leaf を除く）を vjp() 経由でテストする。Sigmoid
    // （TASK-9.1b・#92）追加により対象は 10 演算に拡大。
    // CrossEntropyLoss（#191）追加により対象は 11 演算に拡大
    // （Add/Mul/Relu/Exp/Tanh/Sigmoid/Sum/Max/MseLoss/MatMul/
    // CrossEntropyLoss）。

    fn leaf_node(value: Tensor<f32>) -> TapeNode {
        // `TapeNode`（TASK-12.1d・#164）は `shape` を独立フィールドとして
        // 持ち、`value` は `OnceCell` になった。テスト用の葉ノードは
        // 常に実体化済み（`OnceCell::from`）として構築する。
        let shape = value.shape().to_vec();
        TapeNode {
            op: Op::Leaf,
            shape,
            value: std::cell::OnceCell::from(value),
            lazy_chain_size: 0,
            recompute: false,
            recompute_failed: std::cell::Cell::new(false),
            requires_grad: true,
        }
    }

    /// `vjp()` の第 5 引数（`ops: &dyn BackendOps`）用テストフィクスチャ。
    /// 本モジュールのテストはすべて `leaf_node` で葉ノード（常に実体化
    /// 済み）のみを組み立てるため `materialize_fallible` は早期リターン
    /// し、`ops` の実体は使われない（`crate::test_support::TestOps` を
    /// 形式的に渡すのみ）。
    fn test_ops() -> crate::test_support::TestOps {
        crate::test_support::TestOps
    }

    #[test]
    fn vjp_dispatch_matmul_returns_both_inputs() {
        let a = t(&[1.0, 2.0, -1.0, 0.5, 3.0, -2.0], &[2, 3]);
        let b = t(&[0.5, -1.0, 2.0, 1.0, -0.5, 1.5], &[3, 2]);
        let out_value = eval::matmul(&a, &b);
        let (expected_da, expected_db) =
            matmul_vjp(&test_ops(), &a, &b, &t(&[1.0, -0.5, 0.3, 2.0], &[2, 2])).unwrap();
        let nodes = vec![leaf_node(a), leaf_node(b)];
        let op = Op::MatMul(NodeId(0), NodeId(1));
        let g = t(&[1.0, -0.5, 0.3, 2.0], &[2, 2]);

        let grads = vjp(
            &op,
            &out_value,
            &g,
            &nodes,
            &test_ops(),
            None,
            TapeId::for_test(0),
            0,
        )
        .unwrap();

        assert_eq!(grads.len(), 2);
        assert_eq!(grads[0].0, NodeId(0));
        assert_eq!(grads[1].0, NodeId(1));
        assert_eq!(dense_vec(&grads[0].1), dense_vec(&expected_da));
        assert_eq!(dense_vec(&grads[1].1), dense_vec(&expected_db));
    }

    #[test]
    fn vjp_dispatch_add_returns_both_inputs_in_order() {
        let a = t(&[1.0, -2.0, 3.0, 0.5], &[2, 2]);
        let b = t(&[0.5, 1.5, -1.0, 2.0], &[2, 2]);
        let g = t(&[1.0, -1.0, 2.0, 0.5], &[2, 2]);
        let out_value = eval::add(&a, &b);
        let nodes = vec![leaf_node(a), leaf_node(b)];
        let op = Op::Add(NodeId(0), NodeId(1));

        let grads = vjp(
            &op,
            &out_value,
            &g,
            &nodes,
            &test_ops(),
            None,
            TapeId::for_test(0),
            0,
        )
        .unwrap();

        assert_eq!(grads.len(), 2);
        assert_eq!(grads[0].0, NodeId(0));
        assert_eq!(grads[1].0, NodeId(1));
        assert_eq!(dense_vec(&grads[0].1), dense_vec(&g));
        assert_eq!(dense_vec(&grads[1].1), dense_vec(&g));
    }

    #[test]
    fn vjp_dispatch_mul_returns_both_inputs_in_order() {
        let a = t(&[1.0, -2.0, 3.0, 0.5], &[2, 2]);
        let b = t(&[0.5, 1.5, -1.0, 2.0], &[2, 2]);
        let g = t(&[1.0, -1.0, 2.0, 0.5], &[2, 2]);
        let out_value = eval::mul(&a, &b);
        let nodes = vec![leaf_node(a.clone()), leaf_node(b.clone())];
        let op = Op::Mul(NodeId(0), NodeId(1));

        let grads = vjp(
            &op,
            &out_value,
            &g,
            &nodes,
            &test_ops(),
            None,
            TapeId::for_test(0),
            0,
        )
        .unwrap();

        assert_eq!(grads.len(), 2);
        assert_eq!(grads[0].0, NodeId(0));
        assert_eq!(grads[1].0, NodeId(1));
        assert_eq!(dense_vec(&grads[0].1), dense_vec(&eval::mul(&g, &b)));
        assert_eq!(dense_vec(&grads[1].1), dense_vec(&eval::mul(&g, &a)));
    }

    #[test]
    fn vjp_dispatch_relu_returns_single_input() {
        let a = t(&[2.0, -3.0, 0.5, -0.02], &[2, 2]);
        let g = t(&[1.0, -1.0, 2.0, 0.5], &[2, 2]);
        let out_value = eval::relu(&a);
        let nodes = vec![leaf_node(a.clone())];
        let op = Op::Relu(NodeId(0));

        let grads = vjp(
            &op,
            &out_value,
            &g,
            &nodes,
            &test_ops(),
            None,
            TapeId::for_test(0),
            0,
        )
        .unwrap();

        assert_eq!(grads.len(), 1);
        assert_eq!(grads[0].0, NodeId(0));
        assert_eq!(
            dense_vec(&grads[0].1),
            dense_vec(&elementwise_mul_mask(&g, &a, |v| v > 0.0))
        );
    }

    #[test]
    fn vjp_dispatch_exp_returns_single_input() {
        let a = t(&[0.5, -1.0, 1.5, -0.3], &[2, 2]);
        let g = t(&[1.0, -1.0, 0.5, 2.0], &[2, 2]);
        let out_value = eval::exp(&a);
        let nodes = vec![leaf_node(a)];
        let op = Op::Exp(NodeId(0));

        let grads = vjp(
            &op,
            &out_value,
            &g,
            &nodes,
            &test_ops(),
            None,
            TapeId::for_test(0),
            0,
        )
        .unwrap();

        assert_eq!(grads.len(), 1);
        assert_eq!(grads[0].0, NodeId(0));
        assert_eq!(
            dense_vec(&grads[0].1),
            dense_vec(&eval::mul(&g, &out_value))
        );
    }

    #[test]
    fn vjp_dispatch_tanh_returns_single_input() {
        let a = t(&[0.5, -1.0, 1.5, -0.3], &[2, 2]);
        let g = t(&[1.0, -1.0, 0.5, 2.0], &[2, 2]);
        let out_value = eval::tanh(&a);
        let nodes = vec![leaf_node(a)];
        let op = Op::Tanh(NodeId(0));

        let grads = vjp(
            &op,
            &out_value,
            &g,
            &nodes,
            &test_ops(),
            None,
            TapeId::for_test(0),
            0,
        )
        .unwrap();

        assert_eq!(grads.len(), 1);
        assert_eq!(grads[0].0, NodeId(0));
        let expected = eval::mul(&g, &tanh_grad_factor(&out_value));
        assert_eq!(dense_vec(&grads[0].1), dense_vec(&expected));
    }

    #[test]
    fn vjp_dispatch_sigmoid_returns_single_input() {
        let a = t(&[0.5, -1.0, 1.5, -0.3], &[2, 2]);
        let g = t(&[1.0, -1.0, 0.5, 2.0], &[2, 2]);
        let out_value = eval::sigmoid(&a);
        let nodes = vec![leaf_node(a)];
        let op = Op::Sigmoid(NodeId(0));

        let grads = vjp(
            &op,
            &out_value,
            &g,
            &nodes,
            &test_ops(),
            None,
            TapeId::for_test(0),
            0,
        )
        .unwrap();

        assert_eq!(grads.len(), 1);
        assert_eq!(grads[0].0, NodeId(0));
        let expected = eval::mul(&g, &sigmoid_grad_factor(&out_value));
        assert_eq!(dense_vec(&grads[0].1), dense_vec(&expected));
    }

    #[test]
    fn vjp_dispatch_sum_returns_single_input() {
        let a = t(&[1.0, -2.0, 3.0, 0.5, -1.0, 2.0], &[2, 3]);
        let g = t(&[1.0, -1.0, 2.0], &[3]);
        let out_value = eval::sum(&a, Some(0), &[3]);
        let nodes = vec![leaf_node(a)];
        let op = Op::Sum {
            input: NodeId(0),
            dim: Some(0),
        };

        let grads = vjp(
            &op,
            &out_value,
            &g,
            &nodes,
            &test_ops(),
            None,
            TapeId::for_test(0),
            0,
        )
        .unwrap();

        assert_eq!(grads.len(), 1);
        assert_eq!(grads[0].0, NodeId(0));
        let expected = unreduce_broadcast(&g, &[2, 3], Some(0));
        assert_eq!(dense_vec(&grads[0].1), dense_vec(&expected));
    }

    #[test]
    fn vjp_dispatch_max_returns_single_input() {
        let a = t(&[1.0, -2.0, 5.0, 0.5, -1.0, 2.0], &[2, 3]);
        let g = t(&[1.0, -1.0, 2.0], &[3]);
        let out_value = eval::max(&a, Some(0), &[3]);
        let nodes = vec![leaf_node(a.clone())];
        let op = Op::Max {
            input: NodeId(0),
            dim: Some(0),
        };

        let grads = vjp(
            &op,
            &out_value,
            &g,
            &nodes,
            &test_ops(),
            None,
            TapeId::for_test(0),
            0,
        )
        .unwrap();

        assert_eq!(grads.len(), 1);
        assert_eq!(grads[0].0, NodeId(0));
        let expected = max_vjp(&a, Some(0), &out_value, &g);
        assert_eq!(dense_vec(&grads[0].1), dense_vec(&expected));
    }

    #[test]
    fn vjp_dispatch_mse_loss_mean_returns_both_inputs_in_order() {
        let pred = t(&[1.0, -2.0, 3.0, 0.5], &[2, 2]);
        let target = t(&[0.5, -1.0, 2.5, 1.0], &[2, 2]);
        let g = t(&[3.0], &[]);
        let out_value = eval::mse_loss(&pred, &target, Reduction::Mean);
        let nodes = vec![leaf_node(pred.clone()), leaf_node(target.clone())];
        let op = Op::MseLoss {
            pred: NodeId(0),
            target: NodeId(1),
            reduction: Reduction::Mean,
        };

        let grads = vjp(
            &op,
            &out_value,
            &g,
            &nodes,
            &test_ops(),
            None,
            TapeId::for_test(0),
            0,
        )
        .unwrap();

        assert_eq!(grads.len(), 2);
        assert_eq!(grads[0].0, NodeId(0));
        assert_eq!(grads[1].0, NodeId(1));
        let (expected_dpred, expected_dtarget) = mse_loss_vjp(&pred, &target, &g, Reduction::Mean);
        assert_eq!(dense_vec(&grads[0].1), dense_vec(&expected_dpred));
        assert_eq!(dense_vec(&grads[1].1), dense_vec(&expected_dtarget));
    }

    #[test]
    fn vjp_dispatch_mse_loss_sum_returns_both_inputs_in_order() {
        // sum 縮約（#190）でも `Op::MseLoss` ディスパッチが reduction を
        // 正しく `mse_loss_vjp` へ引き渡すことを確認する。
        let pred = t(&[1.0, -2.0, 3.0, 0.5], &[2, 2]);
        let target = t(&[0.5, -1.0, 2.5, 1.0], &[2, 2]);
        let g = t(&[3.0], &[]);
        let out_value = eval::mse_loss(&pred, &target, Reduction::Sum);
        let nodes = vec![leaf_node(pred.clone()), leaf_node(target.clone())];
        let op = Op::MseLoss {
            pred: NodeId(0),
            target: NodeId(1),
            reduction: Reduction::Sum,
        };

        let grads = vjp(
            &op,
            &out_value,
            &g,
            &nodes,
            &test_ops(),
            None,
            TapeId::for_test(0),
            0,
        )
        .unwrap();

        assert_eq!(grads.len(), 2);
        assert_eq!(grads[0].0, NodeId(0));
        assert_eq!(grads[1].0, NodeId(1));
        let (expected_dpred, expected_dtarget) = mse_loss_vjp(&pred, &target, &g, Reduction::Sum);
        assert_eq!(dense_vec(&grads[0].1), dense_vec(&expected_dpred));
        assert_eq!(dense_vec(&grads[1].1), dense_vec(&expected_dtarget));
    }

    #[test]
    fn vjp_dispatch_huber_loss_mean_returns_both_inputs_in_order() {
        let pred = t(&[1.0, -2.0, 3.0, 0.5], &[2, 2]);
        let target = t(&[0.5, -1.0, 2.5, 1.0], &[2, 2]);
        let g = t(&[3.0], &[]);
        let out_value = eval::huber_loss(&pred, &target, HuberKind::Huber, 1.0, Reduction::Mean);
        let nodes = vec![leaf_node(pred.clone()), leaf_node(target.clone())];
        let op = Op::HuberLoss {
            pred: NodeId(0),
            target: NodeId(1),
            kind: HuberKind::Huber,
            delta: 1.0,
            reduction: Reduction::Mean,
        };

        let grads = vjp(
            &op,
            &out_value,
            &g,
            &nodes,
            &test_ops(),
            None,
            TapeId::for_test(0),
            0,
        )
        .unwrap();

        assert_eq!(grads.len(), 2);
        assert_eq!(grads[0].0, NodeId(0));
        assert_eq!(grads[1].0, NodeId(1));
        let (expected_dpred, expected_dtarget) =
            huber_loss_vjp(&pred, &target, &g, HuberKind::Huber, 1.0, Reduction::Mean);
        assert_eq!(dense_vec(&grads[0].1), dense_vec(&expected_dpred));
        assert_eq!(dense_vec(&grads[1].1), dense_vec(&expected_dtarget));
    }

    #[test]
    fn vjp_dispatch_huber_loss_sum_returns_both_inputs_in_order() {
        // sum 縮約（イシュー #1739）でも `Op::HuberLoss` ディスパッチが
        // reduction を正しく `huber_loss_vjp` へ引き渡すことを確認する
        // （`vjp_dispatch_mse_loss_sum_returns_both_inputs_in_order` と
        // 同型）。`SmoothL1` kind で異なる `delta` も併せて検証する。
        let pred = t(&[1.0, -2.0, 3.0, 0.5], &[2, 2]);
        let target = t(&[0.5, -1.0, 2.5, 1.0], &[2, 2]);
        let g = t(&[3.0], &[]);
        let out_value = eval::huber_loss(&pred, &target, HuberKind::SmoothL1, 2.0, Reduction::Sum);
        let nodes = vec![leaf_node(pred.clone()), leaf_node(target.clone())];
        let op = Op::HuberLoss {
            pred: NodeId(0),
            target: NodeId(1),
            kind: HuberKind::SmoothL1,
            delta: 2.0,
            reduction: Reduction::Sum,
        };

        let grads = vjp(
            &op,
            &out_value,
            &g,
            &nodes,
            &test_ops(),
            None,
            TapeId::for_test(0),
            0,
        )
        .unwrap();

        assert_eq!(grads.len(), 2);
        assert_eq!(grads[0].0, NodeId(0));
        assert_eq!(grads[1].0, NodeId(1));
        let (expected_dpred, expected_dtarget) =
            huber_loss_vjp(&pred, &target, &g, HuberKind::SmoothL1, 2.0, Reduction::Sum);
        assert_eq!(dense_vec(&grads[0].1), dense_vec(&expected_dpred));
        assert_eq!(dense_vec(&grads[1].1), dense_vec(&expected_dtarget));
    }

    #[test]
    fn huber_loss_grad_n_zero_is_zero() {
        // `n == 0`（空テンソル）は mean・sum ともゼロ除算を避け zeros を
        // 返す（`mse_loss_vjp` の同種契約と同型。イシュー #1739）。
        let pred = t(&[], &[0]);
        let target = t(&[], &[0]);
        let g = t(&[1.0], &[]);
        let (dpred, dtarget) =
            huber_loss_vjp(&pred, &target, &g, HuberKind::Huber, 1.0, Reduction::Mean);
        assert_eq!(dense_vec(&dpred), Vec::<f32>::new());
        assert_eq!(dense_vec(&dtarget), Vec::<f32>::new());
    }

    /// `pred`／`target` の一方が NaN のとき、forward（`eval::huber_loss`
    /// の `else` 分岐は `NaN − 0.5·delta = NaN`）と backward
    /// （`eval::huber_elem_grad`）で NaN 伝播の有無が食い違わないことを
    /// 確認する（イシュー #1739 レビュー指摘。`abs_d < delta` は NaN
    /// 比較で常に false のため、明示チェックなしでは backward が
    /// `copysign` 系の有限値〈±1／±delta〉を返してしまう）。
    #[test]
    fn huber_loss_grad_propagates_nan() {
        let pred = t(&[f32::NAN], &[1]);
        let target = t(&[0.0], &[1]);
        let g = t(&[1.0], &[]);
        for kind in [HuberKind::Huber, HuberKind::SmoothL1] {
            for reduction in [Reduction::Mean, Reduction::Sum] {
                let (dpred, dtarget) = huber_loss_vjp(&pred, &target, &g, kind, 1.0, reduction);
                assert!(
                    dense_vec(&dpred)[0].is_nan(),
                    "kind={kind:?} reduction={reduction:?}: dPred は NaN を伝播すべき"
                );
                assert!(
                    dense_vec(&dtarget)[0].is_nan(),
                    "kind={kind:?} reduction={reduction:?}: dTarget は NaN を伝播すべき"
                );
            }
        }
    }

    /// Huber／SmoothL1 の解析的 VJP（`huber_loss_vjp`）を中央差分
    /// （`numeric_grad_unary`）と突合する（イシュー #1739）。損失は
    /// `d = pred − target` のみの区分関数（`eval::huber_elem_loss`）で
    /// `C¹`（連続微分可能）だが `C²` ではないため、標本点は折れ点
    /// `|d| == delta` から中央差分の刻み幅 `H` の数倍以上離した値を
    /// 選ぶ（`pred`/`target` 双方に負の差分・delta≠1 のケースを含む）。
    #[test]
    fn huber_loss_grad_matches_numeric() {
        let pred = t(&[1.5, -3.0, 0.25, -0.8, 2.2, 0.0], &[6]);
        let target = t(&[0.0, 0.0, 0.0, 0.3, -1.0, 0.0], &[6]);
        let s = t(&[1.0], &[]); // スカラー出力への射影は恒等（係数 1）。

        for kind in [HuberKind::Huber, HuberKind::SmoothL1] {
            for delta in [0.5f32, 1.0, 2.0] {
                for reduction in [Reduction::Mean, Reduction::Sum] {
                    let analytic_pred = numeric_grad_unary(&pred, &s, |x| {
                        eval::huber_loss(x, &target, kind, delta, reduction)
                    });
                    let analytic_target = numeric_grad_unary(&target, &s, |x| {
                        eval::huber_loss(&pred, x, kind, delta, reduction)
                    });
                    let (dpred, dtarget) =
                        huber_loss_vjp(&pred, &target, &s, kind, delta, reduction);
                    assert_grad_close(
                        &format!("huber_loss({kind:?}, delta={delta}, {reduction:?}) dPred"),
                        &dpred,
                        &analytic_pred,
                    );
                    assert_grad_close(
                        &format!("huber_loss({kind:?}, delta={delta}, {reduction:?}) dTarget"),
                        &dtarget,
                        &analytic_target,
                    );
                }
            }
        }
    }

    #[test]
    fn vjp_dispatch_cross_entropy_loss_returns_single_input() {
        // `targets` は非追跡（`NodeId` を持たない Op payload）のため、
        // `MseLoss`（pred/target 2 系統）とは異なり寄与は `logits` の
        // 1 系統のみ（`grads.len() == 1`）であることが配線検証の要点
        // （`tape::Op::CrossEntropyLoss` doc 参照）。
        let logits = t(&[1.0, -2.0, 3.0, 0.5, -1.0, 2.0], &[2, 3]);
        let targets = fandhe_ai_tensor_core::Tensor::new(vec![2i32, 0], &[2])
            .expect("test fixture: shape とデータ長は事前に一致させている");
        let g = t(&[3.0], &[]);
        let out_value = eval::cross_entropy_loss(&logits, &targets, 1, Reduction::Mean);
        let nodes = vec![leaf_node(logits.clone())];
        let op = Op::CrossEntropyLoss {
            logits: NodeId(0),
            targets: targets.clone(),
            class_dim: 1,
            reduction: Reduction::Mean,
        };

        let grads = vjp(
            &op,
            &out_value,
            &g,
            &nodes,
            &test_ops(),
            None,
            TapeId::for_test(0),
            0,
        )
        .unwrap();

        assert_eq!(grads.len(), 1);
        assert_eq!(grads[0].0, NodeId(0));
        let expected = cross_entropy_loss_vjp(&logits, &targets, 1, Reduction::Mean, &g);
        assert_eq!(dense_vec(&grads[0].1), dense_vec(&expected));
    }

    // --- RmsNorm / LayerNorm（イシュー #1596） ---

    #[test]
    fn rmsnorm_grad_matches_numeric_no_weight() {
        let x = t(&[1.0, 2.0, -1.0, 0.5, -0.5, 2.0], &[2, 3]);
        let s = t(&[1.0, -2.0, 0.5, 2.0, -1.0, 0.3], &[2, 3]);
        let eps = 1e-5f32;
        let (rows, hidden) = row_norm_layout(&[2, 3]).unwrap();

        let x_slice = dense_vec(&x);
        let s_slice = dense_vec(&s);
        let (da, dw) = rmsnorm_vjp_rows(&x_slice, None, eps, rows, hidden, &s_slice);
        assert!(dw.is_none());
        let da = build_tensor(da, &[2, 3]);

        let num_da =
            numeric_grad_unary(&x, &s, |xt| eval::rmsnorm_rows(xt, None, eps, rows, hidden));
        assert_grad_close("rmsnorm dx (no weight)", &da, &num_da);
    }

    #[test]
    fn rmsnorm_grad_matches_numeric_with_weight() {
        let x = t(&[1.0, 2.0, -1.0, 0.5, -0.5, 2.0], &[2, 3]);
        let w = t(&[2.0, -1.0, 0.5], &[3]);
        let s = t(&[1.0, -2.0, 0.5, 2.0, -1.0, 0.3], &[2, 3]);
        let eps = 1e-5f32;
        let (rows, hidden) = row_norm_layout(&[2, 3]).unwrap();

        let x_slice = dense_vec(&x);
        let w_slice = dense_vec(&w);
        let s_slice = dense_vec(&s);
        let (da, dw) = rmsnorm_vjp_rows(&x_slice, Some(&w_slice), eps, rows, hidden, &s_slice);
        let da = build_tensor(da, &[2, 3]);
        let dw = build_tensor(dw.expect("weight present"), &[3]);

        let num_da = numeric_grad_unary(&x, &s, |xt| {
            eval::rmsnorm_rows(xt, Some(&w_slice), eps, rows, hidden)
        });
        assert_grad_close("rmsnorm dx (weighted)", &da, &num_da);

        let num_dw = numeric_grad_unary(&w, &s, |wt| {
            eval::rmsnorm_rows(&x, Some(&dense_vec(wt)), eps, rows, hidden)
        });
        assert_grad_close("rmsnorm dw", &dw, &num_dw);
    }

    #[test]
    fn layer_norm_grad_matches_numeric_no_affine() {
        let x = t(&[1.0, 2.0, -1.0, 0.5, -0.5, 2.0], &[2, 3]);
        let s = t(&[1.0, -2.0, 0.5, 2.0, -1.0, 0.3], &[2, 3]);
        let eps = 1e-5f32;
        let (rows, hidden) = row_norm_layout(&[2, 3]).unwrap();

        let x_slice = dense_vec(&x);
        let s_slice = dense_vec(&s);
        let (da, dw, db) = layer_norm_vjp_rows(&x_slice, None, false, eps, rows, hidden, &s_slice);
        assert!(dw.is_none());
        assert!(db.is_none());
        let da = build_tensor(da, &[2, 3]);

        let num_da = numeric_grad_unary(&x, &s, |xt| {
            eval::layer_norm_rows(xt, None, None, eps, rows, hidden)
        });
        assert_grad_close("layer_norm dx (no affine)", &da, &num_da);
    }

    #[test]
    fn layer_norm_grad_matches_numeric_with_weight_and_bias() {
        let x = t(&[1.0, 2.0, -1.0, 0.5, -0.5, 2.0], &[2, 3]);
        let w = t(&[2.0, -1.0, 0.5], &[3]);
        let b = t(&[0.1, -0.2, 0.3], &[3]);
        let s = t(&[1.0, -2.0, 0.5, 2.0, -1.0, 0.3], &[2, 3]);
        let eps = 1e-5f32;
        let (rows, hidden) = row_norm_layout(&[2, 3]).unwrap();

        let x_slice = dense_vec(&x);
        let w_slice = dense_vec(&w);
        let s_slice = dense_vec(&s);
        let (da, dw, db) =
            layer_norm_vjp_rows(&x_slice, Some(&w_slice), true, eps, rows, hidden, &s_slice);
        let da = build_tensor(da, &[2, 3]);
        let dw = build_tensor(dw.expect("weight present"), &[3]);
        let db = build_tensor(db.expect("bias present"), &[3]);

        let num_da = numeric_grad_unary(&x, &s, |xt| {
            eval::layer_norm_rows(xt, Some(&w_slice), Some(&dense_vec(&b)), eps, rows, hidden)
        });
        assert_grad_close("layer_norm dx (affine)", &da, &num_da);

        let num_dw = numeric_grad_unary(&w, &s, |wt| {
            eval::layer_norm_rows(
                &x,
                Some(&dense_vec(wt)),
                Some(&dense_vec(&b)),
                eps,
                rows,
                hidden,
            )
        });
        assert_grad_close("layer_norm dw", &dw, &num_dw);

        let num_db = numeric_grad_unary(&b, &s, |bt| {
            eval::layer_norm_rows(&x, Some(&w_slice), Some(&dense_vec(bt)), eps, rows, hidden)
        });
        assert_grad_close("layer_norm db", &db, &num_db);
    }

    #[test]
    fn vjp_dispatch_rms_norm_returns_input_and_weight() {
        let x = t(&[1.0, 2.0, -1.0, 0.5], &[1, 4]);
        let w = t(&[1.0, 1.0, 1.0, 1.0], &[4]);
        let g = t(&[1.0, -2.0, 0.5, 2.0], &[1, 4]);
        let eps = 1e-5f32;
        let out_value = eval::rmsnorm_rows(&x, Some(&dense_vec(&w)), eps, 1, 4);
        let nodes = vec![leaf_node(x.clone()), leaf_node(w.clone())];
        let op = Op::RmsNorm {
            input: NodeId(0),
            weight: Some(NodeId(1)),
            eps,
        };

        let grads = vjp(
            &op,
            &out_value,
            &g,
            &nodes,
            &test_ops(),
            None,
            TapeId::for_test(0),
            0,
        )
        .unwrap();

        assert_eq!(grads.len(), 2);
        assert_eq!(grads[0].0, NodeId(0));
        assert_eq!(grads[1].0, NodeId(1));
    }

    #[test]
    fn vjp_dispatch_layer_norm_returns_input_weight_bias() {
        let x = t(&[1.0, 2.0, -1.0, 0.5], &[1, 4]);
        let w = t(&[1.0, 1.0, 1.0, 1.0], &[4]);
        let b = t(&[0.0, 0.0, 0.0, 0.0], &[4]);
        let g = t(&[1.0, -2.0, 0.5, 2.0], &[1, 4]);
        let eps = 1e-5f32;
        let out_value =
            eval::layer_norm_rows(&x, Some(&dense_vec(&w)), Some(&dense_vec(&b)), eps, 1, 4);
        let nodes = vec![
            leaf_node(x.clone()),
            leaf_node(w.clone()),
            leaf_node(b.clone()),
        ];
        let op = Op::LayerNorm {
            input: NodeId(0),
            weight: Some(NodeId(1)),
            bias: Some(NodeId(2)),
            eps,
        };

        let grads = vjp(
            &op,
            &out_value,
            &g,
            &nodes,
            &test_ops(),
            None,
            TapeId::for_test(0),
            0,
        )
        .unwrap();

        assert_eq!(grads.len(), 3);
        assert_eq!(grads[0].0, NodeId(0));
        assert_eq!(grads[1].0, NodeId(1));
        assert_eq!(grads[2].0, NodeId(2));
    }

    // --- Softmax / LogSoftmax（イシュー #1594） ---
    //
    // softmax の行和は常に 1（一様重みでは L(x) = Σ softmax(x) が
    // 定数となり勾配が恒等的に 0 になってしまい検証が空になる）ため、
    // 射影重み `s` はすべて非一様にする。

    #[test]
    fn softmax_grad_matches_numeric_dim1() {
        let x = t(&[1.0, 2.0, -1.0, 0.5, -0.5, 2.0], &[2, 3]);
        let s = t(&[1.0, -2.0, 0.5, 2.0, -1.0, 0.3], &[2, 3]);

        let out_value = eval::softmax_along(&x, 1);
        let da = softmax_vjp_along(&out_value, &s, 1);
        let num_da = numeric_grad_unary(&x, &s, |x| eval::softmax_along(x, 1));

        assert_grad_close("softmax dim1", &da, &num_da);
    }

    #[test]
    fn softmax_grad_matches_numeric_dim0() {
        let x = t(&[1.0, 2.0, -1.0, 0.5, -0.5, 2.0], &[2, 3]);
        let s = t(&[1.0, -2.0, 0.5, 2.0, -1.0, 0.3], &[2, 3]);

        let out_value = eval::softmax_along(&x, 0);
        let da = softmax_vjp_along(&out_value, &s, 0);
        let num_da = numeric_grad_unary(&x, &s, |x| eval::softmax_along(x, 0));

        assert_grad_close("softmax dim0", &da, &num_da);
    }

    #[test]
    fn softmax_grad_matches_numeric_3d_middle_axis() {
        let x = t(
            &[
                1.0, -1.0, 2.0, 0.5, -0.5, 1.5, 0.3, -0.2, 1.0, -1.0, 2.0, 0.1,
            ],
            &[2, 3, 2],
        );
        let s = t(
            &[
                1.0, -0.5, 2.0, 0.3, -1.0, 0.7, 0.4, -0.8, 1.2, -0.3, 0.6, -1.5,
            ],
            &[2, 3, 2],
        );

        let out_value = eval::softmax_along(&x, 1);
        let da = softmax_vjp_along(&out_value, &s, 1);
        let num_da = numeric_grad_unary(&x, &s, |x| eval::softmax_along(x, 1));

        assert_grad_close("softmax 3d middle axis", &da, &num_da);
    }

    #[test]
    fn log_softmax_grad_matches_numeric_dim1() {
        let x = t(&[1.0, 2.0, -1.0, 0.5, -0.5, 2.0], &[2, 3]);
        let s = t(&[1.0, -2.0, 0.5, 2.0, -1.0, 0.3], &[2, 3]);

        let out_value = eval::log_softmax_along(&x, 1);
        let da = log_softmax_vjp_along(&out_value, &s, 1);
        let num_da = numeric_grad_unary(&x, &s, |x| eval::log_softmax_along(x, 1));

        assert_grad_close("log_softmax dim1", &da, &num_da);
    }

    #[test]
    fn log_softmax_grad_matches_numeric_dim0() {
        let x = t(&[1.0, 2.0, -1.0, 0.5, -0.5, 2.0], &[2, 3]);
        let s = t(&[1.0, -2.0, 0.5, 2.0, -1.0, 0.3], &[2, 3]);

        let out_value = eval::log_softmax_along(&x, 0);
        let da = log_softmax_vjp_along(&out_value, &s, 0);
        let num_da = numeric_grad_unary(&x, &s, |x| eval::log_softmax_along(x, 0));

        assert_grad_close("log_softmax dim0", &da, &num_da);
    }

    #[test]
    fn log_softmax_grad_matches_numeric_3d_middle_axis() {
        let x = t(
            &[
                1.0, -1.0, 2.0, 0.5, -0.5, 1.5, 0.3, -0.2, 1.0, -1.0, 2.0, 0.1,
            ],
            &[2, 3, 2],
        );
        let s = t(
            &[
                1.0, -0.5, 2.0, 0.3, -1.0, 0.7, 0.4, -0.8, 1.2, -0.3, 0.6, -1.5,
            ],
            &[2, 3, 2],
        );

        let out_value = eval::log_softmax_along(&x, 1);
        let da = log_softmax_vjp_along(&out_value, &s, 1);
        let num_da = numeric_grad_unary(&x, &s, |x| eval::log_softmax_along(x, 1));

        assert_grad_close("log_softmax 3d middle axis", &da, &num_da);
    }

    #[test]
    fn vjp_dispatch_softmax_returns_single_input() {
        let a = t(&[1.0, 2.0, -1.0, 0.5], &[2, 2]);
        let g = t(&[1.0, -2.0, 0.5, 2.0], &[2, 2]);
        let out_value = eval::softmax_along(&a, 1);
        let nodes = vec![leaf_node(a)];
        let op = Op::Softmax {
            input: NodeId(0),
            dim: 1,
        };

        let grads = vjp(
            &op,
            &out_value,
            &g,
            &nodes,
            &test_ops(),
            None,
            TapeId::for_test(0),
            0,
        )
        .unwrap();

        assert_eq!(grads.len(), 1);
        assert_eq!(grads[0].0, NodeId(0));
        let expected = softmax_vjp_along(&out_value, &g, 1);
        assert_eq!(dense_vec(&grads[0].1), dense_vec(&expected));
    }

    #[test]
    fn vjp_dispatch_log_softmax_returns_single_input() {
        let a = t(&[1.0, 2.0, -1.0, 0.5], &[2, 2]);
        let g = t(&[1.0, -2.0, 0.5, 2.0], &[2, 2]);
        let out_value = eval::log_softmax_along(&a, 1);
        let nodes = vec![leaf_node(a)];
        let op = Op::LogSoftmax {
            input: NodeId(0),
            dim: 1,
        };

        let grads = vjp(
            &op,
            &out_value,
            &g,
            &nodes,
            &test_ops(),
            None,
            TapeId::for_test(0),
            0,
        )
        .unwrap();

        assert_eq!(grads.len(), 1);
        assert_eq!(grads[0].0, NodeId(0));
        let expected = log_softmax_vjp_along(&out_value, &g, 1);
        assert_eq!(dense_vec(&grads[0].1), dense_vec(&expected));
    }

    // codex-review 指摘（PR #1664）の回帰検証: `sum_acc`（f64）を
    // `exp(y)` との乗算前に `f32` へ downcast する実装では、有限の
    // `f32` 上流勾配でも overflow しうる（`logits=[0,0]` すなわち
    // `y=log_softmax([0,0])=[-ln(2),-ln(2)]`・上流勾配 `g=[2e38,2e38]`
    // で、正しい入力勾配 `[0,0]`〈`Σ_dim(g)=4e38` に対し `exp(y)=0.5`
    // なので `g - exp(y)*Σg = 2e38 - 0.5*4e38 = 0` のはずが、`sum_g`
    // を `f32` へ戻してから `exp(y) as f32 * sum_g` を計算すると
    // `0.5 * 4e38 = 2e38` は有限だが、`f32::MAX ≈ 3.4e38` に近い値の
    // 掛け算・加減算が丸め誤差で `-inf` を生む経路がある）。
    // `exp(y)` との乗算・`g` からの減算まで f64 で保持することで
    // overflow を避ける。
    #[test]
    fn log_softmax_vjp_along_large_upstream_grad_does_not_overflow() {
        let logits = t(&[0.0, 0.0], &[1, 2]);
        let y = eval::log_softmax_along(&logits, 1);
        let g = t(&[2e38, 2e38], &[1, 2]);
        let dx = log_softmax_vjp_along(&y, &g, 1);
        for (c, v) in dense_vec(&dx).iter().enumerate() {
            assert!(
                v.is_finite(),
                "dx[{c}] = {v} は有限であるべき（overflow 回帰）"
            );
            assert!(v.abs() < 1.0, "dx[{c}] = {v}（期待値は 0 近傍）");
        }
    }

    // codex-review 指摘（PR #1664）の回帰検証: 縮約後の `dot`（`Σ_dim
    // (g ⊙ y)`）を `y[idx] * (g[idx] - dot)` の減算まで `f32` で行う
    // 実装では、有限で表現可能な入力勾配が overflow して `inf`/`-inf`
    // になる。`y=[0.25,0.75]`・上流勾配 `g=[3e38,-3e38]` では
    // `dot=-1.5e38` に対し `g[0]-dot=4.5e38` が `f32::MAX`（約 3.4e38）
    // を超えて `f32` では overflow するが、正しい入力勾配
    // `y[0]*(g[0]-dot)=0.25*4.5e38=1.125e38` は有限。`log_softmax_vjp_
    // along` と同じく `g` からの減算・`y` との最終乗算まで `f64` で
    // 保持することで overflow を避ける。
    #[test]
    fn softmax_vjp_along_large_upstream_grad_does_not_overflow() {
        let y = t(&[0.25, 0.75], &[1, 2]);
        let g = t(&[3e38, -3e38], &[1, 2]);
        let dx = softmax_vjp_along(&y, &g, 1);
        let dx = dense_vec(&dx);
        for (c, v) in dx.iter().enumerate() {
            assert!(
                v.is_finite(),
                "dx[{c}] = {v} は有限であるべき（overflow 回帰）"
            );
        }
        assert!(
            (dx[0] - 1.125e38).abs() < 1e33,
            "dx[0] = {}（期待値 1.125e38 近傍）",
            dx[0]
        );
        assert!(
            (dx[1] + 1.125e38).abs() < 1e33,
            "dx[1] = {}（期待値 -1.125e38 近傍）",
            dx[1]
        );
    }

    // codex-review 指摘（PR #1664）の回帰検証: `eval::softmax_along`
    // 冒頭コメント参照。`shape[axis+1..]` 等の部分積は `checked_numel`
    // が通した shape（要素数積は `0`）でも overflow しうるため、
    // `softmax_vjp_along`／`log_softmax_vjp_along` も同じ早期 return
    // で部分積計算前に安全側へ倒れることを確認する。
    #[test]
    fn softmax_vjp_along_empty_tensor_with_overflow_prone_inner_does_not_panic() {
        let shape = [0usize, 0, usize::MAX, 2];
        let y = Tensor::<f32>::new(Vec::new(), &shape)
            .expect("要素数積は 0 のため構築は成功する契約（checked_numel）");
        let g = Tensor::<f32>::new(Vec::new(), &shape)
            .expect("要素数積は 0 のため構築は成功する契約（checked_numel）");
        let dx = softmax_vjp_along(&y, &g, 1);
        assert_eq!(dx.shape(), &shape);
        assert_eq!(dx.numel(), 0);
    }

    #[test]
    fn log_softmax_vjp_along_empty_tensor_with_overflow_prone_inner_does_not_panic() {
        let shape = [0usize, 0, usize::MAX, 2];
        let y = Tensor::<f32>::new(Vec::new(), &shape)
            .expect("要素数積は 0 のため構築は成功する契約（checked_numel）");
        let g = Tensor::<f32>::new(Vec::new(), &shape)
            .expect("要素数積は 0 のため構築は成功する契約（checked_numel）");
        let dx = log_softmax_vjp_along(&y, &g, 1);
        assert_eq!(dx.shape(), &shape);
        assert_eq!(dx.numel(), 0);
    }

    // --- イシュー #1583: elementwise VJP の BackendOps 経由化 ---
    //
    // `vjp_elementwise_mul`／`vjp_elementwise_add` はビルド時定数
    // `ELEMENTWISE_VJP_VIA_BACKEND_OPS` の値でゲートされるため、
    // `_via` バリアントへ両方の値を明示的に渡して両分岐を検証する
    // （`ELEMENTWISE_VJP_VIA_BACKEND_OPS` 自体の現在値に関わらず
    // テストが両分岐をカバーする）。

    /// `ops.mul`／`ops.add` を任意のエラーで応答させ、フォールバック・
    /// エラー伝播の分岐をテストするためだけの `BackendOps` モック。
    /// `mul`/`add` 以外は到達しないため `unreachable!` で明示的に失敗
    /// させる（静かな 0 埋め等の判定迂回を作らない。security.md A08）。
    struct MockOps {
        mul_result: Option<Result<Tensor<f32>, BackendError>>,
        add_result: Option<Result<Tensor<f32>, BackendError>>,
    }

    impl BackendOps for MockOps {
        fn device(&self) -> fandhe_ai_tensor_core::Device {
            fandhe_ai_tensor_core::Device::Cpu
        }
        fn gemm(&self, _a: &Tensor<f32>, _b: &Tensor<f32>) -> Result<Tensor<f32>, BackendError> {
            unreachable!("MockOps::gemm はイシュー #1583 テストでは使わない")
        }
        fn add(&self, a: &Tensor<f32>, b: &Tensor<f32>) -> Result<Tensor<f32>, BackendError> {
            match &self.add_result {
                Some(Ok(t)) => Ok(t.clone()),
                Some(Err(e)) => Err(clone_backend_error(e)),
                None => Ok(crate::eval::add(a, b)),
            }
        }
        fn mul(&self, a: &Tensor<f32>, b: &Tensor<f32>) -> Result<Tensor<f32>, BackendError> {
            match &self.mul_result {
                Some(Ok(t)) => Ok(t.clone()),
                Some(Err(e)) => Err(clone_backend_error(e)),
                None => Ok(crate::eval::mul(a, b)),
            }
        }
        fn relu(&self, _a: &Tensor<f32>) -> Result<Tensor<f32>, BackendError> {
            unreachable!("MockOps::relu はイシュー #1583 テストでは使わない")
        }
        fn exp(&self, _a: &Tensor<f32>) -> Result<Tensor<f32>, BackendError> {
            unreachable!("MockOps::exp はイシュー #1583 テストでは使わない")
        }
        fn tanh(&self, _a: &Tensor<f32>) -> Result<Tensor<f32>, BackendError> {
            unreachable!("MockOps::tanh はイシュー #1583 テストでは使わない")
        }
        fn sum(&self, _a: &Tensor<f32>, _dim: Option<usize>) -> Result<Tensor<f32>, BackendError> {
            unreachable!("MockOps::sum はイシュー #1583 テストでは使わない")
        }
        fn max(&self, _a: &Tensor<f32>, _dim: Option<usize>) -> Result<Tensor<f32>, BackendError> {
            unreachable!("MockOps::max はイシュー #1583 テストでは使わない")
        }
    }

    /// `BackendError` は `Clone` を持たないため、テスト用に必要な
    /// variant のみ手動で複製する（`Unsupported`／`ShapeMismatch`）。
    fn clone_backend_error(e: &BackendError) -> BackendError {
        match e {
            BackendError::Unsupported(msg) => BackendError::Unsupported(msg.clone()),
            BackendError::ShapeMismatch(err) => BackendError::ShapeMismatch(err.clone()),
            other => panic!("clone_backend_error: 未対応の variant {other:?}"),
        }
    }

    /// [`unique_with_fallback`] の回帰テスト（PR #1828 codex-review
    /// P1 是正確認・イシュー #1734）。`backend-cpu::unique`／
    /// `backend-cuda::ops::unique`／`backend-metal::ops::unique` の
    /// 同型回帰テストと同じ手法: `[0, 2, usize::MAX]` を
    /// `transpose(0, 2)` すると `[usize::MAX, 2, 0]` になり、最終的な
    /// 要素数は 0 だが中間積 `usize::MAX * 2` が `usize` の範囲を
    /// 超える。`unique_with_fallback` が `x.numel()`（内部で無検査の
    /// `.iter().product()` を使う）／`ops.unique(x)`（バックエンド
    /// 実装が内部で `numel()`/`contiguous()` を呼ぶ）を呼ぶ前に
    /// 要素数積のオーバーフローを検査し、`overflow-checks` 有効
    /// ビルドでも panic せず `AutodiffError::Shape(ShapeError::
    /// ElementCountOverflow)` を返すことを確認する。チェックは
    /// `ops.unique` 呼び出し前に完了するため `MockOps` の `unique`
    /// （既定実装。到達しない）には依存しない。
    #[test]
    fn unique_with_fallback_rejects_transposed_shape_with_overflowing_intermediate_product() {
        let input_base = Tensor::<f32>::new(Vec::new(), &[0usize, 2usize, usize::MAX]).unwrap();
        let input = input_base.transpose(0, 2).unwrap();
        assert_eq!(input.shape(), &[usize::MAX, 2, 0]);

        let mock = MockOps {
            mul_result: None,
            add_result: None,
        };
        let err = unique_with_fallback(&mock, &input).unwrap_err();
        assert!(matches!(
            err,
            AutodiffError::Shape(ShapeError::ElementCountOverflow)
        ));
    }

    #[test]
    fn vjp_elementwise_mul_via_false_uses_eval_reference() {
        let g = t(&[1.0, 2.0, 3.0, -4.0], &[2, 2]);
        let rhs = t(&[5.0, -6.0, 0.5, 2.0], &[2, 2]);
        let got = vjp_elementwise_mul_via(&test_ops(), &g, &rhs, false).unwrap();
        let expected = eval::mul(&g, &rhs);
        assert_eq!(dense_vec(&got), dense_vec(&expected));
    }

    /// ゲート `true`・`TestOps`（`ops.mul` が `eval::mul` へ委譲する
    /// 参照実装）経由でも `eval::mul` 直呼びと bit 完全一致する
    /// （NaN／-0.0 を含む。単一 IEEE 演算のため）。
    #[test]
    fn vjp_elementwise_mul_via_true_matches_eval_bit_exact() {
        let g = t(&[f32::NAN, -0.0, 1.0, 2.0], &[2, 2]);
        let rhs = t(&[3.0, 4.0, -0.0, f32::NAN], &[2, 2]);
        let got = vjp_elementwise_mul_via(&test_ops(), &g, &rhs, true).unwrap();
        let expected = eval::mul(&g, &rhs);
        for (a, b) in dense_vec(&got).iter().zip(dense_vec(&expected).iter()) {
            assert_eq!(a.to_bits(), b.to_bits());
        }
    }

    /// ブロードキャストを伴う `g ⊙ rhs` も `eval::mul` と一致する
    /// （`Op::Mul` の呼び出しパターン: `upstream` は forward 出力
    /// shape、`rhs` は入力側 shape で異なりうる）。
    #[test]
    fn vjp_elementwise_mul_via_true_handles_broadcast() {
        let g = t(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
        let rhs = t(&[10.0, 20.0, 30.0], &[3]);
        let got = vjp_elementwise_mul_via(&test_ops(), &g, &rhs, true).unwrap();
        let expected = eval::mul(&g, &rhs);
        assert_eq!(got.shape(), expected.shape());
        assert_eq!(dense_vec(&got), dense_vec(&expected));
    }

    /// `ops.mul` が `BackendError::Unsupported` を返した場合のみ
    /// `eval::mul` へフォールバックすることを確認する（vjp_elementwise_
    /// mul_via doc 参照）。
    #[test]
    fn vjp_elementwise_mul_via_true_falls_back_to_eval_on_unsupported() {
        let g = t(&[1.0, 2.0], &[2]);
        let rhs = t(&[3.0, 4.0], &[2]);
        let mock = MockOps {
            mul_result: Some(Err(BackendError::Unsupported("test".to_string()))),
            add_result: None,
        };
        let got = vjp_elementwise_mul_via(&mock, &g, &rhs, true).unwrap();
        let expected = eval::mul(&g, &rhs);
        assert_eq!(dense_vec(&got), dense_vec(&expected));
    }

    /// `Unsupported` 以外のエラー（例: デバイス割当失敗を模した
    /// `ShapeMismatch`）は暗黙にフォールバックせず
    /// `AutodiffError::Backend` として伝播する（fail-closed。
    /// security.md A08）。
    #[test]
    fn vjp_elementwise_mul_via_true_propagates_non_unsupported_error() {
        let g = t(&[1.0, 2.0], &[2]);
        let rhs = t(&[3.0, 4.0], &[2]);
        let mock = MockOps {
            mul_result: Some(Err(BackendError::ShapeMismatch(
                ShapeError::ShapeMismatch {
                    lhs: vec![2],
                    rhs: vec![3],
                },
            ))),
            add_result: None,
        };
        let err = vjp_elementwise_mul_via(&mock, &g, &rhs, true).unwrap_err();
        assert!(matches!(
            err,
            AutodiffError::Backend(BackendError::ShapeMismatch(_))
        ));
    }

    /// バックエンド実装が誤った shape のテンソルを返した場合、
    /// 静かに受け入れず fail-closed でエラーにする
    /// （`vjp_elementwise_mul_via` doc 参照）。
    #[test]
    fn vjp_elementwise_mul_via_true_rejects_wrong_output_shape() {
        let g = t(&[1.0, 2.0], &[2]);
        let rhs = t(&[3.0, 4.0], &[2]);
        let wrong_shape = t(&[1.0, 2.0, 3.0], &[3]);
        let mock = MockOps {
            mul_result: Some(Ok(wrong_shape)),
            add_result: None,
        };
        let err = vjp_elementwise_mul_via(&mock, &g, &rhs, true).unwrap_err();
        assert!(matches!(
            err,
            AutodiffError::Backend(BackendError::ShapeMismatch(_))
        ));
    }

    #[test]
    fn vjp_elementwise_add_via_false_uses_eval_reference() {
        let a = t(&[1.0, 2.0, 3.0, -4.0], &[2, 2]);
        let b = t(&[5.0, -6.0, 0.5, 2.0], &[2, 2]);
        let got = vjp_elementwise_add_via(&test_ops(), &a, &b, false).unwrap();
        let expected = eval::add(&a, &b);
        assert_eq!(dense_vec(&got), dense_vec(&expected));
    }

    /// `backward.rs::accumulate` の fan-out 合算（同 shape の 2 項和）を
    /// 模した bit 完全一致確認（NaN／-0.0 込み）。
    #[test]
    fn vjp_elementwise_add_via_true_matches_eval_bit_exact() {
        let a = t(&[f32::NAN, -0.0, 1.0, 2.0], &[2, 2]);
        let b = t(&[3.0, 4.0, -0.0, f32::NAN], &[2, 2]);
        let got = vjp_elementwise_add_via(&test_ops(), &a, &b, true).unwrap();
        let expected = eval::add(&a, &b);
        for (x, y) in dense_vec(&got).iter().zip(dense_vec(&expected).iter()) {
            assert_eq!(x.to_bits(), y.to_bits());
        }
    }

    #[test]
    fn vjp_elementwise_add_via_true_falls_back_to_eval_on_unsupported() {
        let a = t(&[1.0, 2.0], &[2]);
        let b = t(&[3.0, 4.0], &[2]);
        let mock = MockOps {
            mul_result: None,
            add_result: Some(Err(BackendError::Unsupported("test".to_string()))),
        };
        let got = vjp_elementwise_add_via(&mock, &a, &b, true).unwrap();
        let expected = eval::add(&a, &b);
        assert_eq!(dense_vec(&got), dense_vec(&expected));
    }

    #[test]
    fn vjp_elementwise_add_via_true_propagates_non_unsupported_error() {
        let a = t(&[1.0, 2.0], &[2]);
        let b = t(&[3.0, 4.0], &[2]);
        let mock = MockOps {
            mul_result: None,
            add_result: Some(Err(BackendError::ShapeMismatch(
                ShapeError::ShapeMismatch {
                    lhs: vec![2],
                    rhs: vec![3],
                },
            ))),
        };
        let err = vjp_elementwise_add_via(&mock, &a, &b, true).unwrap_err();
        assert!(matches!(
            err,
            AutodiffError::Backend(BackendError::ShapeMismatch(_))
        ));
    }

    /// `ELEMENTWISE_VJP_VIA_BACKEND_OPS` の現在の出荷値（ドリフト検出。
    /// `docs/perf/elementwise-vjp-backend-ops.md` の verdict と一致する
    /// ことを確認する。値を変える場合は同 doc の verdict 更新とセットで
    /// 変更すること）。
    #[test]
    fn elementwise_vjp_via_backend_ops_gate_matches_documented_default() {
        // `ELEMENTWISE_VJP_VIA_BACKEND_OPS` はビルド時定数のため、素の
        // `assert!` へ渡すと clippy::assertions_on_constants
        // （`-D warnings` 下でエラー）に抵触する。`const { }` ブロックへ
        // 包んでコンパイル時評価であることを明示し、ドリフト検出の意図
        // （ゲート既定値と doc の verdict の一致を機械的に固定する）を
        // 保ったまま clippy を通す。
        const { assert!(!ELEMENTWISE_VJP_VIA_BACKEND_OPS) };
    }

    // `Op::Mul`／`Op::Exp`／`Op::Tanh`／`Op::Sigmoid` の既存 grad-check
    // テスト群（本ファイル冒頭。数値微分との突合）はゲート値に関わらず
    // 現行ビルド設定（`ELEMENTWISE_VJP_VIA_BACKEND_OPS`）の下で実行され、
    // 既に green であることを既存テスト実行で確認済み（`vjp` 経由の
    // 統合経路のカバレッジは既存テストが担う。本節は `vjp_elementwise_
    // *_via` 単体のカバレッジを補う）。
    // --- Permute / BroadcastTo（イシュー #1597） ---

    #[test]
    fn inverse_permutation_roundtrip() {
        for perm in [
            vec![0usize, 1, 2],
            vec![2, 0, 1],
            vec![1, 0],
            vec![0],
            vec![3, 1, 0, 2],
        ] {
            let inv = inverse_permutation(&perm);
            for (k, &p) in perm.iter().enumerate() {
                assert_eq!(inv[p], k, "perm={perm:?} inv={inv:?} で往復しない");
            }
        }
    }

    #[test]
    fn permute_grad_matches_numeric() {
        let x = t(
            &[
                1.0, -2.0, 3.0, 0.5, -1.0, 2.0, 0.25, -0.75, 1.5, -0.5, 2.5, -1.25,
            ],
            &[2, 3, 2],
        );
        let perm = [2usize, 0, 1];
        let out_value = x.permute(&perm).unwrap();
        let s = t(
            &[
                1.0, -0.5, 0.3, 2.0, -1.0, 0.5, -0.2, 1.2, 0.7, -0.3, 1.1, -0.9,
            ],
            out_value.shape(),
        );

        let g = s.clone();
        let inv = inverse_permutation(&perm);
        let da = g.permute(&inv).unwrap();

        let num_da = numeric_grad_unary(&x, &s, |v| v.permute(&perm).unwrap());
        assert_grad_close("permute dx", &da, &num_da);
    }

    #[test]
    fn broadcast_to_grad_matches_numeric_new_leading_axis() {
        // (a) 先頭軸新設: [3] → [2,3]
        let x = t(&[1.0, -2.0, 0.5], &[3]);
        let out_shape = [2usize, 3];
        let out_value = x.broadcast_to(&out_shape).unwrap();
        let s = t(&[1.0, -0.5, 0.3, 2.0, -1.0, 0.5], &out_shape);

        let da = reduce_to_shape(&s, x.shape());
        let num_da = numeric_grad_unary(&x, &s, |v| v.broadcast_to(&out_shape).unwrap());
        assert_grad_close("broadcast_to (new leading axis) dx", &da, &num_da);

        // out_value は forward zero-copy の確認（値そのものの検証は
        // shape 一致で足りる。broadcast_to の値契約自体は tensor-core
        // 側で検証済み）。
        assert_eq!(out_value.shape(), &out_shape);
    }

    #[test]
    fn broadcast_to_grad_matches_numeric_size_one_axis() {
        // (b) size-1 軸拡張: [2,1] → [2,3]
        let x = t(&[1.0, -2.0], &[2, 1]);
        let out_shape = [2usize, 3];
        let s = t(&[1.0, -0.5, 0.3, 2.0, -1.0, 0.5], &out_shape);

        let da = reduce_to_shape(&s, x.shape());
        let num_da = numeric_grad_unary(&x, &s, |v| v.broadcast_to(&out_shape).unwrap());
        assert_grad_close("broadcast_to (size-1 axis) dx", &da, &num_da);
    }

    #[test]
    fn broadcast_to_grad_matches_numeric_scalar() {
        // (c) rank 0 スカラー → [2,2]
        let x = t(&[2.0], &[]);
        let out_shape = [2usize, 2];
        let s = t(&[1.0, -0.5, 0.3, 2.0], &out_shape);

        let da = reduce_to_shape(&s, x.shape());
        let num_da = numeric_grad_unary(&x, &s, |v| v.broadcast_to(&out_shape).unwrap());
        assert_grad_close("broadcast_to (scalar) dx", &da, &num_da);
    }

    #[test]
    fn vjp_dispatch_permute_returns_single_input() {
        let a = t(&[1.0, -2.0, 3.0, 0.5, -1.0, 2.0], &[2, 3]);
        let perm = vec![1usize, 0];
        let out_value = a.permute(&perm).unwrap();
        let g = t(&[1.0, -1.0, 2.0, 0.5, -0.5, 1.5], out_value.shape());
        let nodes = vec![leaf_node(a)];
        let op = Op::Permute {
            input: NodeId(0),
            perm: perm.clone(),
        };

        let grads = vjp(
            &op,
            &out_value,
            &g,
            &nodes,
            &test_ops(),
            None,
            TapeId::for_test(0),
            0,
        )
        .unwrap();

        assert_eq!(grads.len(), 1);
        assert_eq!(grads[0].0, NodeId(0));
        let expected = g.permute(&inverse_permutation(&perm)).unwrap();
        assert_eq!(dense_vec(&grads[0].1), dense_vec(&expected));
    }

    #[test]
    fn vjp_dispatch_broadcast_to_returns_single_input() {
        let a = t(&[1.0, -2.0, 0.5], &[3]);
        let out_shape = [2usize, 3];
        let out_value = a.broadcast_to(&out_shape).unwrap();
        let g = t(&[1.0, -1.0, 2.0, 0.5, -0.5, 1.5], &out_shape);
        let nodes = vec![leaf_node(a)];
        let op = Op::BroadcastTo { input: NodeId(0) };

        let grads = vjp(
            &op,
            &out_value,
            &g,
            &nodes,
            &test_ops(),
            None,
            TapeId::for_test(0),
            0,
        )
        .unwrap();

        assert_eq!(grads.len(), 1);
        assert_eq!(grads[0].0, NodeId(0));
        let expected = reduce_to_shape(&g, &[3]);
        assert_eq!(dense_vec(&grads[0].1), dense_vec(&expected));
    }

    /// `Op::Contiguous`（イシュー #1620）の VJP は upstream をそのまま
    /// 単一入力へ渡す恒等パススルー（`tape::Op::Contiguous` doc参照）。
    #[test]
    fn vjp_dispatch_contiguous_returns_single_input() {
        let a = t(&[1.0, -2.0, 3.0, 0.5], &[2, 2]);
        let out_value = a.clone();
        let g = t(&[1.0, -1.0, 2.0, 0.5], &[2, 2]);
        let nodes = vec![leaf_node(a)];
        let op = Op::Contiguous { input: NodeId(0) };

        let grads = vjp(
            &op,
            &out_value,
            &g,
            &nodes,
            &test_ops(),
            None,
            TapeId::for_test(0),
            0,
        )
        .unwrap();

        assert_eq!(grads.len(), 1);
        assert_eq!(grads[0].0, NodeId(0));
        assert_eq!(dense_vec(&grads[0].1), dense_vec(&g));
    }

    /// codex-review P2 指摘の是正（PRRT_kwDOTuUCJc6hxBOl。設計 `docs/
    /// autodiff-rnn-cell-tape-design.md` 決定 11(h)）: `Op::GruCell` の
    /// `q`（決定 1c: `pre_h` の n 列ブロック。GEMM 再計算を避けるため
    /// backward で `∂n/∂r` の復元に直接使う payload）を意図的に
    /// 破損させると、backward の結果が変化することを確認する構造
    /// テスト。`eval::gru_backward` の式（`dr = d_pre_n * q_val` →
    /// `d_pre_r`。本ファイル冒頭の doc 参照）により、`q` は r ゲート
    /// 列ブロックの勾配にのみ影響するため、`col_start=0` の全幅 embed
    /// を経由する `w_ih`（`affine_vjp` の `d_weight` 戻り値。r 列
    /// ブロックを含む全幅）が `q` の値に応じて変化するはずである。
    /// 変化しなければ `q` が実際には使われていない（GEMM 再計算に
    /// フォールバックしている、または死んでいる）ことを意味する。
    #[test]
    fn vjp_gru_cell_backward_is_sensitive_to_stored_q_payload() {
        // D=2, hidden=1, B=1（`total_cols = gates(=3) * hidden = 3`）。
        let x = t(&[1.0, -0.5], &[1, 2]);
        let h_prev = t(&[0.3], &[1, 1]);
        let w_ih = t(&[0.1, 0.2, -0.1, 0.05, 0.3, -0.2], &[2, 3]);
        let w_hh = t(&[0.2, -0.1, 0.05], &[1, 3]);
        // gates_rzn（活性化後の r,z,n。値域は sigmoid/tanh 範囲内）。
        let gates_rzn = t(&[0.6, 0.4, 0.2], &[1, 3]);
        let dh = t(&[1.0], &[1, 1]);
        // Op::GruCell 分岐は `out_value` を参照しない（本ファイル上部の
        // `Op::GruCell` 分岐実装参照）ためプレースホルダで足りる。
        let out_value = t(&[0.0], &[1, 1]);

        let nodes = vec![
            leaf_node(x),
            leaf_node(h_prev),
            leaf_node(w_ih),
            leaf_node(w_hh),
        ];

        let q_correct = t(&[0.5], &[1, 1]);
        let q_corrupted = t(&[9.0], &[1, 1]);

        let op_correct = Op::GruCell {
            x: NodeId(0),
            h_prev: NodeId(1),
            w_ih: NodeId(2),
            w_hh: NodeId(3),
            b_ih: None,
            b_hh: None,
            gates_rzn: gates_rzn.clone(),
            q: q_correct,
        };
        let op_corrupted = Op::GruCell {
            x: NodeId(0),
            h_prev: NodeId(1),
            w_ih: NodeId(2),
            w_hh: NodeId(3),
            b_ih: None,
            b_hh: None,
            gates_rzn,
            q: q_corrupted,
        };

        let grads_correct = vjp(
            &op_correct,
            &out_value,
            &dh,
            &nodes,
            &test_ops(),
            None,
            TapeId::for_test(0),
            0,
        )
        .unwrap();
        let grads_corrupted = vjp(
            &op_corrupted,
            &out_value,
            &dh,
            &nodes,
            &test_ops(),
            None,
            TapeId::for_test(0),
            0,
        )
        .unwrap();

        let dw_ih_correct = grads_correct
            .iter()
            .find(|(id, _)| *id == NodeId(2))
            .map(|(_, g)| dense_vec(g))
            .expect("w_ih への寄与が存在するはず");
        let dw_ih_corrupted = grads_corrupted
            .iter()
            .find(|(id, _)| *id == NodeId(2))
            .map(|(_, g)| dense_vec(g))
            .expect("w_ih への寄与が存在するはず");

        assert_ne!(
            dw_ih_correct, dw_ih_corrupted,
            "q を破損させても w_ih 勾配が変化しない: q payload が backward で実際に \
             使われていない（GEMM 再計算・死んだ payload 等の）疑いがある"
        );
    }

    /// codex-review P2 指摘の是正（PRRT_kwDOTuUCJc6hxBOl。設計 `docs/
    /// autodiff-rnn-cell-tape-design.md` 決定 11(j)）: `Op::LstmHidden`
    /// の VJP が `cell`（`NodeId`）経由で `nodes[cell.0].op` を参照し、
    /// そこに保持された `Op::LstmCell.w_ih`／`w_hh` の**実データ**を
    /// 読んでいることを確認する構造テスト。`cell` が指す先の
    /// `Op::LstmCell` ノードの `w_ih`／`w_hh` leaf データだけを差し替え
    /// (`x`／`h_prev`／`gates_ifg`／`gate_o` 等は完全に同一のまま)、
    /// `affine_vjp` が返す `dx`（`w_ih_val` に依存）・`dh_prev`
    /// （`w_hh_val` に依存）が変化することを確認する。変化しなければ
    /// `cell` 参照が実際には読まれていない（固定値・別経路へのフォール
    /// バック等）ことを意味する。
    #[test]
    fn vjp_lstm_hidden_reads_referenced_cell_node_weight_data() {
        let x = t(&[1.0, -0.5], &[1, 2]);
        let h_prev = t(&[0.3, -0.2], &[1, 2]);
        let c_prev = t(&[0.1, 0.4], &[1, 2]);
        let w_ih_a = t(
            &[
                0.1, 0.2, -0.1, 0.05, 0.3, -0.2, 0.15, -0.05, 0.2, -0.3, 0.1, 0.25, 0.05, -0.1,
                0.2, -0.15,
            ],
            &[2, 8],
        );
        let w_hh_a = t(
            &[
                0.2, -0.1, 0.05, 0.1, -0.2, 0.3, 0.1, 0.05, -0.1, 0.2, 0.15, -0.05, 0.1, 0.2,
                -0.05, 0.15,
            ],
            &[2, 8],
        );
        // w_ih_b／w_hh_b は w_ih_a／w_hh_a と全要素 +1.0 だけ異なる
        // （shape 同一・データのみ破損させた「別の」重み）。
        let w_ih_b = t(
            &dense_vec(&w_ih_a)
                .iter()
                .map(|v| v + 1.0)
                .collect::<Vec<_>>(),
            &[2, 8],
        );
        let w_hh_b = t(
            &dense_vec(&w_hh_a)
                .iter()
                .map(|v| v + 1.0)
                .collect::<Vec<_>>(),
            &[2, 8],
        );
        let gates_ifg = t(&[0.6, 0.4, 0.3, 0.7, -0.2, 0.5], &[1, 6]);
        let gate_o = t(&[0.55, 0.45], &[1, 2]);
        let c_t = t(&[0.2, -0.1], &[1, 2]);
        let h_t = t(&[0.1, 0.05], &[1, 2]);
        let dh = t(&[1.0, -1.0], &[1, 2]);

        let build_nodes = |w_ih: Tensor<f32>, w_hh: Tensor<f32>| {
            let cell_op = Op::LstmCell {
                x: NodeId(0),
                h_prev: NodeId(1),
                c_prev: NodeId(2),
                w_ih: NodeId(3),
                w_hh: NodeId(4),
                b_ih: None,
                b_hh: None,
                gates_ifg: gates_ifg.clone(),
            };
            // `Op::LstmCell` ノードは常に実体化済み（`tape.rs::Op::
            // LstmCell` doc の push_eager 契約）のため、テスト用にも
            // `OnceCell::from` で事前に値を設定する。
            let cell_node = TapeNode {
                op: cell_op,
                shape: c_t.shape().to_vec(),
                value: std::cell::OnceCell::from(c_t.clone()),
                lazy_chain_size: 0,
                recompute: false,
                recompute_failed: std::cell::Cell::new(false),
                requires_grad: true,
            };
            vec![
                leaf_node(x.clone()),
                leaf_node(h_prev.clone()),
                leaf_node(c_prev.clone()),
                leaf_node(w_ih),
                leaf_node(w_hh),
                cell_node,
            ]
        };

        let nodes_a = build_nodes(w_ih_a, w_hh_a);
        let nodes_b = build_nodes(w_ih_b, w_hh_b);
        let op_hidden = Op::LstmHidden {
            cell: NodeId(5),
            gate_o,
        };

        let grads_a = vjp(
            &op_hidden,
            &h_t,
            &dh,
            &nodes_a,
            &test_ops(),
            None,
            TapeId::for_test(0),
            0,
        )
        .unwrap();
        let grads_b = vjp(
            &op_hidden,
            &h_t,
            &dh,
            &nodes_b,
            &test_ops(),
            None,
            TapeId::for_test(0),
            0,
        )
        .unwrap();

        let dx_a = grads_a
            .iter()
            .find(|(id, _)| *id == NodeId(0))
            .map(|(_, g)| dense_vec(g))
            .expect("x への寄与が存在するはず");
        let dx_b = grads_b
            .iter()
            .find(|(id, _)| *id == NodeId(0))
            .map(|(_, g)| dense_vec(g))
            .expect("x への寄与が存在するはず");
        let dh_prev_a = grads_a
            .iter()
            .find(|(id, _)| *id == NodeId(1))
            .map(|(_, g)| dense_vec(g))
            .expect("h_prev への寄与が存在するはず");
        let dh_prev_b = grads_b
            .iter()
            .find(|(id, _)| *id == NodeId(1))
            .map(|(_, g)| dense_vec(g))
            .expect("h_prev への寄与が存在するはず");

        assert_ne!(
            dx_a, dx_b,
            "cell 参照先の w_ih データを差し替えても dx が変化しない: LstmHidden の \
             VJP が cell 経由の w_ih を実際に読んでいない疑いがある"
        );
        assert_ne!(
            dh_prev_a, dh_prev_b,
            "cell 参照先の w_hh データを差し替えても dh_prev が変化しない: LstmHidden \
             の VJP が cell 経由の w_hh を実際に読んでいない疑いがある"
        );
    }

    // --- Where／MaskedFill（イシュー #1637） ---

    /// `where_vjp` の解析勾配が数値微分と一致することを確認する
    /// （同 shape・分岐反転なしの固定 `cond`）。`cond` 自体は摂動対象
    /// 外（定数マスク）のため `numeric_grad_unary` をそのまま使える。
    #[test]
    fn where_grad_matches_numeric_same_shape() {
        let cond = t(&[1.0, 0.0, 1.0, 0.0], &[2, 2]);
        let a = t(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
        let b = t(&[10.0, 20.0, 30.0, 40.0], &[2, 2]);
        let s = t(&[1.0, -0.5, 0.3, 2.0], &[2, 2]);

        let g = s.clone();
        let (da, db) = where_vjp(&cond, &g, &[2, 2], &[2, 2]);

        let num_da = numeric_grad_unary(&a, &s, |x| eval::where_cond(&cond, x, &b, &[2, 2]));
        let num_db = numeric_grad_unary(&b, &s, |x| eval::where_cond(&cond, &a, x, &[2, 2]));

        assert_grad_close("where dA", &da, &num_da);
        assert_grad_close("where dB", &db, &num_db);
    }

    /// broadcast（`a: [2,2]`, `b: [2]`）で `db` が行方向へ縮約される
    /// ことを確認する（`Op::Mul` の broadcast VJP と同じ縮約契約）。
    #[test]
    fn where_grad_broadcast_reduces_to_input_shape() {
        let cond = t(&[1.0, 0.0, 0.0, 1.0], &[2, 2]);
        let g = t(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);

        let (da, db) = where_vjp(&cond, &g, &[2, 2], &[2]);

        assert_eq!(da.shape(), &[2, 2]);
        assert_eq!(db.shape(), &[2]);
        // da: cond!=0 の位置のみ g を通す。
        assert_eq!(dense_vec(&da), vec![1.0, 0.0, 0.0, 4.0]);
        // db: cond==0 の位置のみ g を通し、行方向（broadcast 元軸）で
        // 合算する。cond=[[1,0],[0,1]]・g=[[1,2],[3,4]] より
        // masked=[[0,2],[3,0]]・列ごとの和=[0+3, 2+0]=[3, 2]。
        assert_eq!(dense_vec(&db), vec![3.0, 2.0]);
    }

    /// 同一 `Var` を `a`／`b` 両方に指定した場合（`where(c, x, x)`）、
    /// `accumulate` が合算する前提のもと、`da + db == g`（全域で
    /// upstream をそのまま通す）ことを確認する。
    #[test]
    fn where_grad_same_var_both_sides_sums_to_upstream() {
        let cond = t(&[1.0, 0.0, 1.0, 0.0], &[2, 2]);
        let g = t(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);

        let (da, db) = where_vjp(&cond, &g, &[2, 2], &[2, 2]);
        let sum: Vec<f32> = dense_vec(&da)
            .iter()
            .zip(dense_vec(&db).iter())
            .map(|(&a, &b)| a + b)
            .collect();
        assert_eq!(sum, dense_vec(&g));
    }

    /// NaN が非選択側に留まる（選択側の値・勾配へ伝播しない）ことを
    /// 確認する。`cond` が `1.0` の位置では `b` 側に NaN があっても
    /// forward 出力・`da` は NaN の影響を受けない。
    #[test]
    fn where_forward_and_grad_isolate_nan_to_unselected_side() {
        let cond = t(&[1.0, 0.0], &[2]);
        let a = t(&[1.0, 2.0], &[2]);
        let b = t(&[f32::NAN, 20.0], &[2]);
        let out_shape = [2usize];

        let value = eval::where_cond(&cond, &a, &b, &out_shape);
        assert_eq!(dense_vec(&value), vec![1.0, 20.0]);

        let g = t(&[1.0, 1.0], &[2]);
        let (da, db) = where_vjp(&cond, &g, &out_shape, &out_shape);
        assert_eq!(dense_vec(&da), vec![1.0, 0.0]);
        // db[0] は `cond[0] != 0.0` により 0 になるはず（NaN の位置は
        // 選択されていないため upstream を通さない）。
        assert_eq!(dense_vec(&db)[0], 0.0);
        assert_eq!(dense_vec(&db)[1], 1.0);
    }

    /// `masked_fill_vjp` の解析勾配が数値微分と一致することを確認する
    /// （fill 位置の勾配は 0、それ以外は upstream をそのまま通す）。
    #[test]
    fn masked_fill_grad_matches_numeric() {
        let mask = t(&[1.0, 0.0, 1.0, 0.0], &[2, 2]);
        let x = t(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
        let s = t(&[1.0, -0.5, 0.3, 2.0], &[2, 2]);
        let value = -9.0f32;

        let g = s.clone();
        let dx = masked_fill_vjp(&mask, &g);

        let num_dx = numeric_grad_unary(&x, &s, |t| eval::masked_fill(t, &mask, value));

        assert_grad_close("masked_fill dX", &dx, &num_dx);
    }

    /// fill 位置の勾配が厳密に 0 であることを直接確認する。
    #[test]
    fn masked_fill_grad_zero_at_filled_positions() {
        let mask = t(&[1.0, 0.0, 1.0, 0.0], &[2, 2]);
        let g = t(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);

        let dx = masked_fill_vjp(&mask, &g);

        assert_eq!(dense_vec(&dx), vec![0.0, 2.0, 0.0, 4.0]);
    }

    // --- イシュー #1734 review 指摘: `validate_unique_output` の 2
    // 分岐（`ShapeMismatch`／`InvalidArgument`）を直接検証する。
    // 3 バックエンドとも契約を守るため通常経路では到達しないが、
    // 事後検査自体が正しく機能することを踏まえ、バックエンド実装を
    // 経由せず不変条件検査関数を直接叩いて両分岐を踏む。

    /// rank != 1（2 次元）の出力は shape 自体の契約違反として
    /// `AutodiffError::Backend(BackendError::ShapeMismatch(..))` を
    /// 返すことを確認する。
    #[test]
    fn validate_unique_output_rejects_wrong_rank() {
        let v = t(&[1.0, 2.0], &[1, 2]);
        let err = validate_unique_output(&v, 4).unwrap_err();
        assert!(matches!(
            err,
            AutodiffError::Backend(BackendError::ShapeMismatch(_))
        ));
    }

    /// `len > numel`（入力要素数を超える出力長）も同じく
    /// `ShapeMismatch` として拒否されることを確認する。
    #[test]
    fn validate_unique_output_rejects_len_exceeding_numel() {
        let v = t(&[1.0, 2.0, 3.0], &[3]);
        let err = validate_unique_output(&v, 2).unwrap_err();
        assert!(matches!(
            err,
            AutodiffError::Backend(BackendError::ShapeMismatch(_))
        ));
    }

    /// rank・len 自体は正しいが totalOrder で非減少でない（降順が
    /// 混入した）出力は、shape 違反ではなく
    /// `AutodiffError::InvalidArgument` として区別して拒否されること
    /// を確認する。
    #[test]
    fn validate_unique_output_rejects_non_monotonic_order() {
        let v = t(&[3.0, 1.0, 2.0], &[3]);
        let err = validate_unique_output(&v, 3).unwrap_err();
        assert!(matches!(err, AutodiffError::InvalidArgument(_)));
    }

    /// 非減少ではあるが隣接に `==`（重複）が残る出力も
    /// `InvalidArgument` として拒否されることを確認する（`unique` は
    /// 重複除去済みでなければならない）。
    #[test]
    fn validate_unique_output_rejects_adjacent_duplicate() {
        let v = t(&[1.0, 2.0, 2.0, 3.0], &[4]);
        let err = validate_unique_output(&v, 4).unwrap_err();
        assert!(matches!(err, AutodiffError::InvalidArgument(_)));
    }

    /// 契約を満たす出力（非減少・重複なし・len <= numel）は
    /// 受理されることを確認する（両分岐が誤検出しないことの対照）。
    #[test]
    fn validate_unique_output_accepts_valid_output() {
        let v = t(&[1.0, 2.0, 3.0], &[3]);
        assert!(validate_unique_output(&v, 5).is_ok());
    }

    // --- WideFloat（cumprod VJP のオーバーフロー安全アキュムレータ。
    //     codex-review 指摘・PR #1819） ---

    /// `frexp` のラウンドトリップ性質（`mantissa * 2^exponent == v`・
    /// `mantissa` の絶対値が `[0.5, 1.0)`）を、正規化数・非正規化数
    /// （絶対値の上限・下限付近）・負値・2 のべき乗ちょうどの値の
    /// 各ケースで確認する。
    #[test]
    fn frexp_roundtrips_and_normalizes_mantissa() {
        let cases: &[f64] = &[
            1.0,
            2.0,
            0.5,
            3.0,
            -3.0,
            1e300,
            1e-300,
            f64::MIN_POSITIVE,       // 最小の正規化数（2^-1022）
            f64::MIN_POSITIVE / 2.0, // 非正規化数
            f64::MIN_POSITIVE * 1.5, // 正規化数の下限付近
            5e-324,                  // 最小の非正規化数（1 ULP）
            -5e-324,
            f64::MAX,
            -f64::MAX,
            2f64.powi(100),
            2f64.powi(-100),
        ];
        for &v in cases {
            let (m, e) = frexp(v);
            assert!(
                m.abs() >= 0.5 && m.abs() < 1.0,
                "frexp({v}): mantissa {m} が [0.5, 1.0) の範囲外"
            );
            // 検証には（ネイティブ `f64::powi` 経由の素朴な再構成では
            // なく）本モジュール自身の `ldexp` を使う。`f64::powi` は
            // `2^-1073` のような極端な非正規化数境界で `0.0` へ潰れて
            // しまい（`frexp` が解消しようとしている精度問題そのもの
            // が再構成側に混入する）、意図した回帰検出にならない。
            let reconstructed = ldexp(m, e);
            assert_eq!(
                reconstructed, v,
                "frexp({v}) のラウンドトリップが不一致（mantissa={m}, exponent={e}）"
            );
            assert_eq!(
                m.is_sign_negative(),
                v.is_sign_negative(),
                "frexp({v}): 符号不一致"
            );
        }
    }

    /// [`WideFloat`] の乗算・加算が、オーバーフローしない通常範囲では
    /// ネイティブ `f64` 演算と一致することを確認する（回帰: `WideFloat`
    /// 導入が既存の精度を劣化させていないことの直接検証）。
    #[test]
    fn wide_float_mul_add_match_native_f64_in_normal_range() {
        let pairs: &[(f64, f64)] = &[
            (1.5, 2.5),
            (-3.0, 7.0),
            (0.1, 0.2),
            (123456.789, -0.0001234),
            (1e10, 1e-5),
        ];
        for &(a, b) in pairs {
            let wa = WideFloat::from_f64(a);
            let wb = WideFloat::from_f64(b);
            assert_eq!(
                wa.mul(wb).to_f64(),
                a * b,
                "WideFloat::mul({a}, {b}) が f64 と不一致"
            );
            assert_eq!(
                wa.add(wb).to_f64(),
                a + b,
                "WideFloat::add({a}, {b}) が f64 と不一致"
            );
        }
    }

    /// [`WideFloat`] は零オペランドとの乗算を、相手がどれほど大きな
    /// 指数を持っていても厳密に `WideFloat::ZERO` に固定する（cumprod
    /// VJP の零遮断規約。`f64` ネイティブなら `0.0 * inf = NaN` に
    /// なる状況でも `NaN` を生まないことを確認する）。
    #[test]
    fn wide_float_zero_dominates_even_with_extreme_exponent() {
        // `2^1000` 自体は `f64` の表現域内（最大 `≈ 1.8e308 ≈ 2^1024`）
        // だが、それを `WideFloat::mul` で自乗すると真の値は `2^2000`
        // となり `f64` 表現域を超える。`WideFloat` は指数を `i64` の
        // 加算として保持するのみで、この乗算自体は内部でオーバー
        // フローしない（`to_f64()` した時点で初めて inf になる）。
        let big = WideFloat::from_f64(2f64.powi(1000));
        let huge = big.mul(big);
        assert!(huge.to_f64().is_infinite());
        let zero = WideFloat::ZERO;
        let product = zero.mul(huge);
        assert!(product.is_zero());
        assert_eq!(product.to_f64(), 0.0);
    }

    /// [`WideFloat`] は中間積が `f64` 表現域を大きく超えても内部では
    /// オーバーフローしない（真の値が最終的に有限へ戻るケースで、
    /// 素朴な `f64` 乗算なら `inf` のまま復元できない状況を再現する）。
    #[test]
    fn wide_float_survives_intermediate_overflow_and_recovers() {
        let big = WideFloat::from_f64(2f64.powi(600));
        // 素朴な `f64` ならここで `big_squared` は inf になる
        // （`2^600 * 2^600 = 2^1200` は `f64` 表現域〈約 `2^1024`〉超）。
        assert!((big.to_f64() * big.to_f64()).is_infinite());
        let big_squared = big.mul(big);
        // `WideFloat` は指数を分離して保持するため、この時点でも
        // 有限（`to_f64()` すると真に表現域外なので inf になるのは
        // 正しい——ここで確認したいのは、続けて小さい係数を掛けたとき
        // に「inf のまま戻らない」現象が起きないこと）。
        let small = WideFloat::from_f64(2f64.powi(-600));
        let recovered = big_squared.mul(small);
        assert_eq!(
            recovered.to_f64(),
            2f64.powi(600),
            "big^2 * small = 2^600 へ有限で復元できるはず"
        );
    }

    /// [`WideFloat`] の指数（`i64`）が旧実装（`i32`）の表現域
    /// （`±2.147e9`）を大きく超えても panic せず正しく飽和することを
    /// 確認する回帰テスト（codex-review 追加指摘・PR #1819）。指摘は
    /// `x = [0] + [2^127] × 17,000,000` のような入力（約 68 MB）で
    /// suffix 積の指数が `127 * 17e6 ≈ 2.16e9` に達し `i32` を超える
    /// というものだったが、ここでは大量の要素を確保せずに同じ現象
    /// （指数が `i32::MAX` を大きく超える）を「2 分累乗」で軽量に
    /// 再現する: `2^70` を 25 回自乗すると、指数は `70 * 2^25 ≈
    /// 2.35e9` となり `i32::MAX`（`≈ 2.147e9`）を超えるが、`i64::MAX`
    /// （`≈ 9.22e18`）には遠く及ばない。旧 `i32` 実装なら
    /// `overflow-checks` 有効時に `+` が panic し、無効時は指数が
    /// 折り返して誤った（符号や桁が滅茶苦茶な）有限値を返していた
    /// はずの状況である。
    #[test]
    fn wide_float_exponent_survives_i32_overflow_without_panicking() {
        let mut acc = WideFloat::from_f64(2f64.powi(70));
        for _ in 0..25 {
            acc = acc.mul(acc); // 指数は毎回 2 倍（i64 の checked_add で panic しない）
        }
        // 真の値は `2^(70 * 2^25)` であり、これは `f64` の表現域
        // （最大 `≈ 2^1024`）を天文学的規模で超えるため、`to_f64()`
        // が符号付き `inf` へ飽和するのが数学的に正しい挙動。
        let value = acc.to_f64();
        assert!(
            value.is_infinite() && value.is_sign_positive(),
            "指数オーバーフロー後も符号付き inf へ正しく飽和するはず（実際: {value}）"
        );

        // 小さい係数を掛けて指数を大きく下げても、悪化前と同様に
        // 有限へ正しく復元できる（`WideFloat::mul` の指数飽和が
        // 以後の演算を破壊しないことの確認）。
        let tiny = WideFloat::from_f64(2f64.powi(-70));
        let mut shrink = acc;
        // 24 回掛けて指数を `70 * 2^25 - 70 * 24` 程度まで下げる
        // （それでもなお表現域外に留まる規模のため inf のまま）。
        for _ in 0..24 {
            shrink = shrink.mul(tiny);
        }
        let shrunk_value = shrink.to_f64();
        assert!(
            !shrunk_value.is_nan(),
            "指数飽和後の演算が NaN を生んではならない（実際: {shrunk_value}）"
        );
    }

    /// `cumprod_vjp_along` の統合テスト: 軸上に極端に大きい要素
    /// （`2^127`）が数千個連続し、先頭が `0` である入力に対して、
    /// suffix 積の指数が `i32` の表現域（`±2.147e9`）を超えて真に
    /// `f64` 表現域外まで発散しうる状況でも backward が panic せず、
    /// `NaN` を生まないことを確認する（codex-review 追加指摘・PR
    /// #1819 の再現条件を、実メモリを大量消費しない小規模な軸長で
    /// 近似したもの）。`x[0] = 0` より後段（`a >= 1`）は排他的
    /// prefix 積 `L[a]` が厳密に `0` を含むため、`WideFloat::mul` の
    /// 零遮断規約により勾配は厳密に `0.0` になる（真の値がどれほど
    /// 発散していても `0 * 有限 = 0` が表現域と無関係に成り立つ）。
    /// `dx[0]` は `S[0]`（`x[1..]` が全て `2^127` の Horner 型後方
    /// 再帰の和）そのものであり、`2^127` の連続乗算は数要素で真に
    /// `f64` 表現域（`最大 ≈ 2^1024`）を超えるため、符号付き `inf`
    /// へ正しく飽和するのが数学的に正しい（`0.0` や有限値へ誤って
    /// wrap しないことが本テストの主眼であり、「有限または 0」を
    /// 要求するものではない）。
    #[test]
    fn cumprod_vjp_along_handles_large_magnitude_run_without_panicking() {
        const AXIS_LEN: usize = 4096;
        let mut x = vec![2f64.powi(127) as f32; AXIS_LEN];
        x[0] = 0.0;
        let input = build_tensor(x, &[AXIS_LEN]);
        let upstream = build_tensor(vec![1.0f32; AXIS_LEN], &[AXIS_LEN]);

        let dx = cumprod_vjp_along(&input, &upstream, 0);
        let dx_values = dense_vec(&dx);
        assert_eq!(dx_values.len(), AXIS_LEN);
        for (i, &v) in dx_values.iter().enumerate() {
            assert!(!v.is_nan(), "index {i} の勾配が NaN になってはならない");
        }
        for (i, &v) in dx_values.iter().enumerate().skip(1) {
            assert_eq!(
                v, 0.0,
                "index {i}: 先頭の 0 を含む prefix 積により厳密に 0.0 のはず（実際: {v}）"
            );
        }
        assert!(
            dx_values[0].is_infinite() && dx_values[0].is_sign_positive(),
            "dx[0] は真に f64 表現域外へ発散するため符号付き inf のはず（実際: {}）",
            dx_values[0]
        );
    }

    // イシュー #1834 codex-review P1 是正の回帰テスト（2 件）。

    #[test]
    fn nearest_src_index_map_does_not_overflow_with_huge_in_shape_and_small_out() {
        // `outer=1`（空でない先頭軸）・空間軸の入力サイズが巨大
        // （`2^63`）・出力サイズは小さい（`3`）組み合わせ。`row` の
        // 確保自体は `sp_out_numel=3` で軽量だが、内部で呼ぶ
        // `eval::nearest_src_coord` の中間積 `dst * in_size` が
        // `usize` を overflow しうる（backend-cpu クレートの `interpolate::nearest_src_coord` と同型の bug。是正後は
        // `u128` 昇格により正しい添字を返す）。
        // `in_size` 自体は `i32::MAX` を超えるため、本関数が返す `i32`
        // 添字（`pos as i32`）は `usize` の `pos` を正しく計算した上での
        // 意図的な切り詰めであり、値そのものが `i32` の範囲へ収まる
        // ことは呼び出し元〈VJP `Op::Interpolate` 分岐〉が事前に
        // `sp_in_numel <= i32::MAX` を検査する契約の範囲外（本テストは
        // その契約を満たさない `in_size` を意図的に渡す）。ここで
        // 検証したいのは `usize` の中間積 `dst * in_size` 自体が
        // overflow せず（是正前は debug ビルドで overflow panic して
        // いた）、`pos as i32` へ到達する前に panic しないことのみ。
        let in_size = 1usize << 63;
        let index = nearest_src_index_map(&[1, in_size], &[1, 3], 1, 1);
        assert_eq!(index.shape(), &[1, 3]);
        let data = eval::dense_vec_i32(&index);
        assert_eq!(data.len(), 3);
    }

    #[test]
    fn interpolate_vjp_empty_leading_axis_with_huge_size_does_not_panic() {
        // `Op::Interpolate` VJP の `outer == 0` 早期 return 経路
        // （`grad.rs` 本体の `Op::Interpolate` 分岐）を `vjp` 経由で
        // 直接検証する。`outer == 0` のまま `nearest_src_index_map` を
        // 呼ぶと `size=[usize::MAX]` 等で `sp_out_numel` が巨大になり
        // capacity overflow panic しうる（本テストが是正前に再現した
        // 不具合の直接検証）。
        let input_shape = vec![0usize, 1];
        let input = build_tensor(Vec::new(), &input_shape);
        let nodes = vec![leaf_node(input)];
        let op = Op::Interpolate {
            input: NodeId(0),
            size: vec![usize::MAX],
            mode: fandhe_ai_tensor_core::InterpolateMode::Nearest,
        };
        let out_value = build_tensor(Vec::new(), &[0, usize::MAX]);
        let upstream = build_tensor(Vec::new(), &[0, usize::MAX]);

        let contributions = vjp(
            &op,
            &out_value,
            &upstream,
            &nodes,
            &test_ops(),
            None,
            TapeId::for_test(0),
            0,
        )
        .expect("outer==0 の場合 panic せず成功するはず");
        assert_eq!(contributions.len(), 1);
        let (node, d_input) = &contributions[0];
        assert_eq!(*node, NodeId(0));
        assert_eq!(d_input.shape(), &input_shape);
        assert_eq!(d_input.numel(), 0);
    }

    /// Cursor Bugbot 指摘（イシュー #1834・PR レビュー）の回帰テスト。
    /// `interpolate_vjp_empty_leading_axis_with_huge_size_does_not_panic`
    /// （上記）は空間軸が 1 軸のみのため `sp_out_numel` の計算が単一
    /// 値の読み取りに留まり overflow を経由しない。本テストは空間軸
    /// **2 軸**（`usize::MAX` と `2`）を持つ入力 `[0, usize::MAX, 2]`
    /// を使い、是正前のコード（`outer == 0` の早期 return より前に
    /// `out_shape[spatial_start..].iter().product()` を計算していた）
    /// では `usize::MAX * 2` が `usize` の範囲を超え overflow panic
    /// （debug ビルド）していたことを再現する。**本テストは `vjp()`
    /// を `Var::interpolate`（`interpolate_out_shape` による事前検査を
    /// 経由する通常経路）ではなく直接呼び出す**ため、`interpolate_
    /// out_shape` 側の検査を経由しない。この入力・`size` の組合せ
    /// （`shape=[0, usize::MAX, 2]`・`size=[usize::MAX, 2]`）は
    /// `ops_shape.rs::interpolate_out_shape` 自身が現在は事前拒否する
    /// （`nonzero_dims` 検査。イシュー #1834 Cursor Bugbot 指摘・是正）
    /// ため、`Var::interpolate` 経由では forward の時点で
    /// `ElementCountOverflow` となり本テストが再現する `vjp()` 単体
    /// 呼び出しの状況（forward 成功後に VJP だけが呼ばれる状況）には
    /// 到達しない。それでも本テストを維持するのは、`vjp()` 自身が
    /// 「呼び出し元が検査済み」という不変条件に暗黙に依存せず独立に
    /// 安全側へ倒れることを担保するため（多層防御。上記
    /// `interpolate_vjp_nonempty_leading_axis_with_overflowing_spatial_
    /// product_returns_error` と同じ設計判断）。是正後は `outer == 0`
    /// の早期 return が積計算そのものより前に位置するため、この
    /// 組み合わせでも panic せずゼロ勾配を返す。
    #[test]
    fn interpolate_vjp_empty_leading_axis_with_overflowing_spatial_product_does_not_panic() {
        let input_shape = vec![0usize, usize::MAX, 2];
        let input = build_tensor(Vec::new(), &input_shape);
        let nodes = vec![leaf_node(input)];
        let op = Op::Interpolate {
            input: NodeId(0),
            size: vec![usize::MAX, 2],
            mode: fandhe_ai_tensor_core::InterpolateMode::Nearest,
        };
        let out_value = build_tensor(Vec::new(), &input_shape);
        let upstream = build_tensor(Vec::new(), &input_shape);

        let contributions = vjp(
            &op,
            &out_value,
            &upstream,
            &nodes,
            &test_ops(),
            None,
            TapeId::for_test(0),
            0,
        )
        .expect("outer==0 の場合、空間軸積が overflow する組み合わせでも panic せず成功するはず");
        assert_eq!(contributions.len(), 1);
        let (node, d_input) = &contributions[0];
        assert_eq!(*node, NodeId(0));
        assert_eq!(d_input.shape(), &input_shape);
        assert_eq!(d_input.numel(), 0);
    }

    /// Cursor Bugbot 指摘（PR #1834 レビュー）の回帰テスト。上記 2 件は
    /// いずれも `outer == 0`（先頭軸が空）の経路のみを検証しており、
    /// `outer == 0` ガードを通過した**後**の `sp_in_numel`／
    /// `sp_out_numel` 計算自体が無検査 `.iter().product()` のままでは
    /// overflow しうる、という指摘の経路（`outer != 0`）は未検証
    /// だった。
    ///
    /// `Var::interpolate`／`Tensor::broadcast_to` を経由する通常の
    /// 構築経路では、テンソル全軸の積オーバーフロー検査
    /// （`checked_numel` 相当）が `outer` を含む shape 全体に対して
    /// 事前に行われるため、`outer != 0` のとき空間軸だけの部分積が
    /// 単独で overflow する状態は実際には構築不能（本ファイル冒頭の
    /// 是正コメント参照）。本テストはその不変条件を迂回し、`vjp()`
    /// 自身が「呼び出し元が正しく検査済みである」という前提に暗黙に
    /// 依存せず自前でも overflow を検査することを直接確認する
    /// （`TapeNode` を `leaf_node`／`build_tensor` を介さず直接構築し、
    /// `shape` フィールドにのみ検査を経ていない巨大値を注入する。
    /// `Op::Interpolate` の VJP 分岐は `nodes[input.0].value` を参照
    /// せず `.shape` のみを読むため、`value` 自体は shape と無関係な
    /// 軽量なダミーで構わない）。
    #[test]
    fn interpolate_vjp_nonempty_leading_axis_with_overflowing_spatial_product_returns_error() {
        // 空間軸 2 本の積 `(1<<32) * (1<<32) == 1<<64` は 64bit `usize`
        // の範囲（`usize::MAX == (1<<64) - 1`）をちょうど 1 超え
        // overflow する。先頭軸は `1`（`outer == 1` で非ゼロ）。
        let input_shape = vec![1usize, 1usize << 32, 1usize << 32];
        let dummy_value = Tensor::scalar(0.0f32);
        let node = TapeNode {
            op: Op::Leaf,
            shape: input_shape.clone(),
            value: std::cell::OnceCell::from(dummy_value),
            lazy_chain_size: 0,
            recompute: false,
            recompute_failed: std::cell::Cell::new(false),
            requires_grad: true,
        };
        let nodes = vec![node];
        let op = Op::Interpolate {
            input: NodeId(0),
            size: vec![3, 3],
            mode: fandhe_ai_tensor_core::InterpolateMode::Nearest,
        };
        let out_shape = vec![1usize, 3, 3];
        let out_value = build_tensor(vec![0.0f32; 9], &out_shape);
        let upstream = build_tensor(vec![0.0f32; 9], &out_shape);

        let err = vjp(
            &op,
            &out_value,
            &upstream,
            &nodes,
            &test_ops(),
            None,
            TapeId::for_test(0),
            0,
        )
        .expect_err(
            "outer != 0 でも空間軸の部分積 overflow は panic ではなく \
             型付きエラーを返すはず",
        );
        assert!(
            matches!(err, AutodiffError::Shape(ShapeError::ElementCountOverflow)),
            "sp_in_numel の checked_mul overflow は ElementCountOverflow を返すはず（実際: {err:?}）"
        );
    }
}
