//! `interp::run_with_ops` から呼ばれる `BackendOps` 経由の device 実行ヘルパ
//! （非公開。イシュー #2077・親 #2076「ONNX import モデルの GPU 実行」）。
//!
//! `interp` 側の各 `compute_*` は、`ops: Option<&dyn BackendOps>` が
//! `Some` かつ対象 op（`MatMul`／`Gemm`／`Add`／`Mul`／`Div`／`Sqrt`／
//! `Relu`／`Sigmoid`／`Softmax`／`LayerNormalization`）の f32 経路である
//! 場合に限り本モジュールの `device_*` 関数を呼ぶ。各関数は次の 3 分岐を
//! 返す:
//!
//! - `Ok(Some(value))`: device 実行に成功した（呼び出し元はホスト実装を
//!   スキップしこの値を採用する）
//! - `Ok(None)`: device 側で扱えない（未実装カーネル・非対応 shape・
//!   要素数 0 等）ため、呼び出し元は**既存のホスト実装（`ops::*`）へ
//!   そのままフォールバックする**（判定・検証をここで重複させない。
//!   opt-in OFF 時と同じホスト検証・同じエラー面を保つため）
//! - `Err(InterpError::Backend { .. })`: `BackendError::Unsupported`／
//!   `BackendError::ShapeMismatch` 以外の実行時エラー（driver 不在・
//!   カーネル起動失敗・デバイスメモリ確保失敗等）。ホストへの黙示
//!   フォールバックはしない（GPU 故障を隠蔽しない。OWASP A08・
//!   `.claude/rules/security.md`）
//!
//! `ShapeMismatch` を `Unsupported` と同列にフォールバック対象とする
//! 設計判断: 各 `device_*` は呼び出し前に必要最小限の形状検査（rank・
//! 空入力・LayerNorm の単一正規化軸等）を行うが、境界ケースの完全な
//! 検証はホスト実装（`ops::*`）が既に持つロジックと二重管理しない
//! ため、device 側カーネルが返す shape 起因のエラーもホストへ委ねる
//! （ホストが同じ入力に対して発する型付きエラー——`OpError::
//! GemmDimMismatch`／`MatMulDimMismatch`等——と opt-in ON／OFF で
//! 一致させるため。`docs/onnx-gpu-execution-decision.md` §3.2 参照）。
//!
//! 数値契約: device 実行の結果は同一 shape のホスト実装（`ops::*`。
//! 逐次 `mul_add` 参照実装）とは一般に bit 一致しない（CUDA／Metal の
//! カーネルは異なる結合順序で部分和を求めるため。`cpu_row_kernel_naive_
//! parity.rs` と同型の関係）。REQ-2 統一複合判定（相対誤差 1e-3 未満
//! または絶対誤差 1e-5 未満）で突合する（`.claude/rules/coding-rust.md`）。

use fandhe_ai_tensor_core::{BackendError, BackendOps, ScalarBinaryOp, ScalarUnaryOp, Tensor};

use crate::ops::{GemmAttrs, LayerNormAttrs, normalize_axis};

use super::interp::InterpError;

/// `f` を実行し、`BackendError::Unsupported`／`BackendError::ShapeMismatch`
/// は `Ok(None)`（呼び出し元がホストへフォールバック）、それ以外は
/// [`InterpError::Backend`] として伝播する（モジュール冒頭コメント参照）。
fn try_device<F>(node: &str, f: F) -> Result<Option<Tensor<f32>>, InterpError>
where
    F: FnOnce() -> Result<Tensor<f32>, BackendError>,
{
    match f() {
        Ok(t) => Ok(Some(t)),
        Err(BackendError::Unsupported(_)) | Err(BackendError::ShapeMismatch(_)) => Ok(None),
        Err(e) => Err(InterpError::Backend {
            node: node.to_string(),
            message: e.to_string(),
        }),
    }
}

/// `MatMul` の device 経路（イシュー #2077 実装計画 §3.3）。
///
/// rank 2×2 は [`BackendOps::gemm_fp32_strict`]、rank≥3 を含む場合は
/// [`BackendOps::gemm_batched_fp32_strict`]（既定実装が NumPy 互換
/// バッチブロードキャストを自前で正規化するため、呼び出し側での
/// 事前正規化は不要）を使う。ONNX `MatMul` の 1-D 特例（軸挿入・除去。
/// `ops::matmul` 冒頭コメント参照）は `gemm_batched_fp32_strict` の
/// 契約に無いため常にホストへ委ねる（`rank < 2` は `Ok(None)`）。
pub(super) fn device_matmul(
    node: &str,
    ops: &dyn BackendOps,
    a: &Tensor<f32>,
    b: &Tensor<f32>,
) -> Result<Option<Tensor<f32>>, InterpError> {
    if a.rank() < 2 || b.rank() < 2 {
        return Ok(None);
    }
    if a.numel() == 0 || b.numel() == 0 {
        return Ok(None);
    }
    if a.rank() == 2 && b.rank() == 2 {
        try_device(node, || ops.gemm_fp32_strict(a, b))
    } else {
        try_device(node, || ops.gemm_batched_fp32_strict(a, b))
    }
}

/// `Gemm` の device 経路。`transA`／`transB` はホスト側と同じ
/// `transpose(0,1).contiguous()` で実体化してから
/// [`BackendOps::gemm_fp32_strict`] を呼び（TF32 opt-in の適用範囲を
/// 無断拡張しない。`docs/cuda-tf32-optin-api-decision.md`）、`alpha`
/// 乗算・`beta*C` 加算は `ops::gemm`（ホスト実装）と同じ順序
/// （`alpha * acc` の後に `+= beta * cv`）で自前適用する（GPU 側
/// epilogue 融合〈`gemm_bias_act`〉は使わない。本イシューのスコープ外）。
pub(super) fn device_gemm(
    node: &str,
    ops: &dyn BackendOps,
    a: &Tensor<f32>,
    b: &Tensor<f32>,
    c: Option<&Tensor<f32>>,
    attrs: &GemmAttrs,
) -> Result<Option<Tensor<f32>>, InterpError> {
    if a.rank() != 2 || b.rank() != 2 {
        return Ok(None);
    }
    let a_eff = if attrs.trans_a {
        match a.transpose(0, 1) {
            Ok(t) => t.contiguous(),
            Err(_) => return Ok(None),
        }
    } else {
        a.contiguous()
    };
    let b_eff = if attrs.trans_b {
        match b.transpose(0, 1) {
            Ok(t) => t.contiguous(),
            Err(_) => return Ok(None),
        }
    } else {
        b.contiguous()
    };
    let (m, k) = (a_eff.shape()[0], a_eff.shape()[1]);
    let (k2, n) = (b_eff.shape()[0], b_eff.shape()[1]);
    if k != k2 || m == 0 || n == 0 || k == 0 {
        // 内部次元不一致・空入力はホストの既存検証（`OpError::
        // GemmDimMismatch` 等）へ委ねる。
        return Ok(None);
    }

    let raw = match try_device(node, || ops.gemm_fp32_strict(&a_eff, &b_eff))? {
        Some(t) => t,
        None => return Ok(None),
    };
    let raw_c = raw.contiguous();
    let Some(raw_slice) = raw_c.as_slice() else {
        return Ok(None);
    };

    let mut out: Vec<f32> = raw_slice.iter().map(|&v| attrs.alpha * v).collect();
    if let Some(c) = c {
        let Ok(c_b) = c.broadcast_to(&[m, n]) else {
            return Ok(None);
        };
        let c_c = c_b.contiguous();
        let Some(c_slice) = c_c.as_slice() else {
            return Ok(None);
        };
        for (o, &cv) in out.iter_mut().zip(c_slice.iter()) {
            *o += attrs.beta * cv;
        }
    }

    let t = Tensor::new(out, &[m, n]).map_err(InterpError::from)?;
    Ok(Some(t))
}

/// `Add`／`Mul` の device 経路。[`BackendOps::add`]／[`BackendOps::mul`]
/// は既存契約（`crates/backend-cpu/src/elementwise.rs` 冒頭コメント）
/// どおり NumPy 互換ブロードキャストへ自前対応するため、事前の
/// `broadcast_with`／`contiguous()` 実体化は不要（そのまま渡す）。
pub(super) fn device_add(
    node: &str,
    ops: &dyn BackendOps,
    a: &Tensor<f32>,
    b: &Tensor<f32>,
) -> Result<Option<Tensor<f32>>, InterpError> {
    if a.numel() == 0 || b.numel() == 0 {
        return Ok(None);
    }
    try_device(node, || ops.add(a, b))
}

pub(super) fn device_mul(
    node: &str,
    ops: &dyn BackendOps,
    a: &Tensor<f32>,
    b: &Tensor<f32>,
) -> Result<Option<Tensor<f32>>, InterpError> {
    if a.numel() == 0 || b.numel() == 0 {
        return Ok(None);
    }
    try_device(node, || ops.mul(a, b))
}

/// `Div` の device 経路（[`BackendOps::scalar_binary`]・
/// [`ScalarBinaryOp::Div`]）。
pub(super) fn device_div(
    node: &str,
    ops: &dyn BackendOps,
    a: &Tensor<f32>,
    b: &Tensor<f32>,
) -> Result<Option<Tensor<f32>>, InterpError> {
    if a.numel() == 0 || b.numel() == 0 {
        return Ok(None);
    }
    try_device(node, || ops.scalar_binary(ScalarBinaryOp::Div, a, b))
}

/// `Sqrt` の device 経路（[`BackendOps::scalar_unary`]・
/// [`ScalarUnaryOp::Sqrt`]）。
pub(super) fn device_sqrt(
    node: &str,
    ops: &dyn BackendOps,
    x: &Tensor<f32>,
) -> Result<Option<Tensor<f32>>, InterpError> {
    if x.numel() == 0 {
        return Ok(None);
    }
    try_device(node, || ops.scalar_unary(ScalarUnaryOp::Sqrt, x))
}

/// `Relu` の device 経路。**`BackendOps::relu`（NaN を 0 に潰す。
/// `crates/backend-cpu/src/elementwise.rs` 冒頭コメント）は使わない**
/// ——ONNX `Relu`（`ops::relu`）・[`ScalarUnaryOp::Relu`] はいずれも
/// `NaN` を伝播する契約であり、数値意味論を変えないため。CUDA／Metal は
/// 本イシュー時点で `ScalarUnaryOp::Relu` 未実装（`Unsupported` →
/// ホストへフォールバック）。
pub(super) fn device_relu(
    node: &str,
    ops: &dyn BackendOps,
    x: &Tensor<f32>,
) -> Result<Option<Tensor<f32>>, InterpError> {
    if x.numel() == 0 {
        return Ok(None);
    }
    try_device(node, || ops.scalar_unary(ScalarUnaryOp::Relu, x))
}

/// `Sigmoid` の device 経路（[`ScalarUnaryOp::Sigmoid`]。CUDA／Metal は
/// 本イシュー時点で未実装のため `Unsupported` → ホストへフォールバック）。
pub(super) fn device_sigmoid(
    node: &str,
    ops: &dyn BackendOps,
    x: &Tensor<f32>,
) -> Result<Option<Tensor<f32>>, InterpError> {
    if x.numel() == 0 {
        return Ok(None);
    }
    try_device(node, || ops.scalar_unary(ScalarUnaryOp::Sigmoid, x))
}

/// `Softmax` の device 経路（[`BackendOps::softmax`]）。`axis` の正規化
/// のみここで行い、最終軸限定の判定は `BackendOps::softmax` 自身の
/// 既定契約（非最終軸は `Unsupported` を返す。`crates/tensor-core/src/
/// backend_ops.rs::softmax` doc）に委ねる（二重管理しない）。
pub(super) fn device_softmax(
    node: &str,
    ops: &dyn BackendOps,
    x: &Tensor<f32>,
    axis: i64,
) -> Result<Option<Tensor<f32>>, InterpError> {
    if x.numel() == 0 {
        return Ok(None);
    }
    let rank = x.rank();
    let Some(dim) = normalize_axis(axis, rank) else {
        return Ok(None);
    };
    try_device(node, || ops.softmax(x, dim))
}

/// `LayerNormalization` の device 経路（[`BackendOps::layer_norm`]）。
/// `BackendOps::layer_norm` は最終軸 1 軸のみを正規化集合とする契約
/// （`w`／`b` は `[cols]` 形状）だが、ONNX の `axis` は `x.shape()[axis..]`
/// という**複数軸にまたがりうる**正規化集合を許す
/// （`ops::layer_normalization` 冒頭コメント・`rank3_multi_dim_
/// normalized_set_axis1` テスト参照）。したがって正規化後の `axis` が
/// 「最終軸 1 つのみ」（`axis == rank - 1`）の場合に限り device 経路を
/// 試みる（複数軸にまたがる場合は常にホストへ委ねる。誤った縮約軸で
/// 実行しないための必須ガード）。
pub(super) fn device_layer_norm(
    node: &str,
    ops: &dyn BackendOps,
    x: &Tensor<f32>,
    scale: &Tensor<f32>,
    bias: Option<&Tensor<f32>>,
    attrs: &LayerNormAttrs,
) -> Result<Option<Tensor<f32>>, InterpError> {
    if !attrs.epsilon.is_finite() {
        // 非有限 epsilon はホストの `OpError::InvalidEpsilon` へ委ねる。
        return Ok(None);
    }
    if attrs.epsilon < 0.0 {
        // 負の epsilon は ONNX 仕様上合法でホスト実装（`ops::
        // layer_normalization`）は許容・透過するが、CPU/CUDA/Metal の
        // `validate_layer_norm_launch` はいずれも起動前 fail-closed 検証
        // で負値を `InvalidEps`（→ `BackendError::KernelLaunchFailed`）
        // として拒否する契約（各バックエンドの `layer_norm.rs`）。
        // `try_device` は `KernelLaunchFailed` をフォールバック対象
        // （`Unsupported`／`ShapeMismatch`）に含めないため、ここで拒否
        // せずに呼び出すと opt-in ON 時のみランが中断し、opt-in OFF
        // （ホスト実行）では成功するという不整合が生じる
        // （cursor(Bugbot) 指摘・PR #2222）。ホストが許容する入力は
        // device 経路の可否に関わらず常に成功させるため、ここで
        // 明示的にホストへ委ねる。
        return Ok(None);
    }
    let rank = x.rank();
    if rank == 0 {
        return Ok(None);
    }
    let Some(axis) = normalize_axis(attrs.axis, rank) else {
        return Ok(None);
    };
    if axis != rank - 1 {
        // 正規化集合が複数軸にまたがる: device の単一最終軸契約では
        // 扱えないため常にホストへ委ねる。
        return Ok(None);
    }
    if x.numel() == 0 {
        return Ok(None);
    }
    let cols = x.shape()[rank - 1];
    if cols == 0 {
        return Ok(None);
    }
    let Ok(scale_full) = scale.broadcast_to(&[cols]) else {
        return Ok(None);
    };
    let scale_c = scale_full.contiguous();
    let bias_c = match bias {
        Some(b) => match b.broadcast_to(&[cols]) {
            Ok(t) => Some(t.contiguous()),
            Err(_) => return Ok(None),
        },
        None => None,
    };
    try_device(node, || {
        ops.layer_norm(x, Some(&scale_c), bias_c.as_ref(), attrs.epsilon)
    })
}
