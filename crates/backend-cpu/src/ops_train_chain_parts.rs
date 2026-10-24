//! 学習ステップ入口（常駐チェーン）の CPU 部品（イシュー #3253。決定記録
//! `docs/compat-resident-train-step-decision.md` §13・設計 `docs/train-step-resident-parts-design.md`）。
//!
//! # 役割と呼び出し文脈
//! `autodiff::DeviceParamStore::train_step_device_chain` はバックエンド非依存の 1 本のループで、
//! MSE・ReLU マスク・dW（bias 縮約の融合）・d_input の各部品を `BackendOps` 越しに呼ぶ。Metal（#3251）・
//! CUDA（#3252）は専用実装を持ち、CPU は本モジュールの部品を `ops.rs` の `impl BackendOps` から
//! 委譲することで**同じループを再利用する**（チェーンを 2 本にしない。決定記録 §11）。
//!
//! # bit 一致の方針
//! 比較先は同じ CPU のホスト版（Tape 経路）。新しい縮約や stride 読みの核は書かず、ホスト経路と
//! **同じ核を同じ選択入力で呼ぶ**ことで bit 一致を構造的に保つ。FMA 契約・`f64` アキュムレータ契約・
//! tolerance・baseline は変えない。CPU は呼び出し内で完了するため `DispatchFailureCell` は使わず、
//! エラーは同期的に `Err` で返す。
//!
//! # 境界検査（REQ-8・OWASP A03）
//! shape・offset の積と和は `checked_*`、スライス取得は `get(range)` で行い panic させない。
//! 検査はすべて書き込みより前に済ませ、`Err` のとき `out` は一切変わらない。

use fandhe_ai_tensor_core::buffer::{DeviceBuffer, DeviceBufferView};
use fandhe_ai_tensor_core::device::{BackendError, Device};
use fandhe_ai_tensor_core::{DeviceMatrixLayout, MseReduction, ShapeError};

use super::shared_cpu_memory;
use crate::gemm_blis::{gemm_blis_parallel, gemm_blis_parallel_nt, gemm_blis_parallel_tn};
use crate::memory::CpuBufferHandle;
use crate::mse;

fn invalid(msg: String) -> BackendError {
    BackendError::InvalidArgument(msg)
}

fn shape_mismatch(lhs: &[usize], rhs: &[usize]) -> BackendError {
    BackendError::ShapeMismatch(ShapeError::ShapeMismatch {
        lhs: lhs.to_vec(),
        rhs: rhs.to_vec(),
    })
}

/// MSE の最終スカラー（`Mean` は `sum_sq / numel`、`numel == 0` は 0.0）。ホスト版
/// `CpuBackendOps::mse_loss` と本部品が同じ式を共有する（別式で書くと bit がずれる）。
pub(super) fn mse_reduce_value(
    sum_sq: f32,
    numel: usize,
    reduction: MseReduction,
    ctx: &str,
) -> Result<f32, BackendError> {
    match reduction {
        MseReduction::Mean => Ok(if numel == 0 {
            0.0
        } else {
            sum_sq / numel as f32
        }),
        MseReduction::Sum => Ok(sum_sq),
        _ => Err(BackendError::Unsupported(format!(
            "{ctx}: unsupported MseReduction variant {reduction:?}"
        ))),
    }
}

/// `_into` 版の出力 `out` の事前検査（device → shape → 実体長）。CUDA の `check_into_out` と同型。
pub(super) fn check_cpu_into_out(
    out: &DeviceBuffer<f32>,
    expected_shape: &[usize],
) -> Result<(), BackendError> {
    if out.device() != Device::Cpu {
        return Err(BackendError::DeviceMismatch);
    }
    if out.shape() != expected_shape {
        return Err(shape_mismatch(out.shape(), expected_shape));
    }
    let handle = out
        .downcast_handle::<CpuBufferHandle>()
        .ok_or(BackendError::DeviceMismatch)?;
    if handle.data().len() != out.numel() {
        return Err(BackendError::ShapeMismatch(
            ShapeError::ElementCountMismatch {
                expected: out.numel(),
                actual: handle.data().len(),
            },
        ));
    }
    Ok(())
}

/// `out` の先頭 `len` 要素への可変スライス（`check_cpu_into_out` 後に使う）。
pub(super) fn out_slice_mut(
    out: &mut DeviceBuffer<f32>,
    len: usize,
) -> Result<&mut [f32], BackendError> {
    let handle = out
        .downcast_handle_mut::<CpuBufferHandle>()
        .ok_or(BackendError::DeviceMismatch)?;
    handle
        .data_mut()
        .get_mut(..len)
        .ok_or_else(|| invalid(format!("out slice length {len} exceeds backing buffer")))
}

/// view の範囲をスライスとして取り出す（範囲外は panic ではなく `InvalidArgument`）。
fn view_slice<'a>(v: &DeviceBufferView<'a>) -> Result<&'a [f32], BackendError> {
    if v.device() != Device::Cpu {
        return Err(BackendError::DeviceMismatch);
    }
    let handle = v
        .buffer()
        .downcast_handle::<CpuBufferHandle>()
        .ok_or(BackendError::DeviceMismatch)?;
    let end = v
        .offset()
        .checked_add(v.numel())
        .ok_or_else(|| invalid("view offset + numel overflowed usize".into()))?;
    handle
        .data()
        .get(v.offset()..end)
        .ok_or_else(|| invalid(format!("view range [{}, {end}) exceeds buffer", v.offset())))
}

fn buffer_slice(b: &DeviceBuffer<f32>) -> Result<&[f32], BackendError> {
    if b.device() != Device::Cpu {
        return Err(BackendError::DeviceMismatch);
    }
    let handle = b
        .downcast_handle::<CpuBufferHandle>()
        .ok_or(BackendError::DeviceMismatch)?;
    handle
        .data()
        .get(..b.numel())
        .ok_or_else(|| invalid("buffer length is smaller than its numel".into()))
}

fn wrap(data: Vec<f32>, shape: Vec<usize>) -> Result<DeviceBuffer<f32>, BackendError> {
    shared_cpu_memory().wrap_vec(data, shape)
}

fn unsupported_empty(name: &str) -> BackendError {
    BackendError::Unsupported(format!(
        "{name}: empty/degenerate shape is left to the allocating path"
    ))
}

/// `mse_loss_device`／`mse_loss_device_into_tracked` の共有本体。出力は shape `[]`。
/// `out` が `Some` で `numel == 0` のときは契約どおり `Unsupported`（確保版へ任せる）。
pub(super) fn mse_loss_device_core(
    pred: &DeviceBuffer<f32>,
    target: &DeviceBuffer<f32>,
    reduction: MseReduction,
    out: Option<&mut DeviceBuffer<f32>>,
) -> Result<Option<DeviceBuffer<f32>>, BackendError> {
    if pred.device() != Device::Cpu || target.device() != Device::Cpu {
        return Err(BackendError::DeviceMismatch);
    }
    if pred.shape() != target.shape() {
        return Err(shape_mismatch(pred.shape(), target.shape()));
    }
    let numel = pred.numel();
    let ctx = "mse_loss_device";
    if out.is_some() && numel == 0 {
        return Err(unsupported_empty("mse_loss_device_into_tracked"));
    }
    if let Some(o) = out.as_deref() {
        check_cpu_into_out(o, &[])?;
    }
    let sum_sq = mse::mse_sum_sq_f32(buffer_slice(pred)?, buffer_slice(target)?)?;
    let value = mse_reduce_value(sum_sq, numel, reduction, ctx)?;
    match out {
        None => wrap(vec![value], vec![]).map(Some),
        Some(o) => {
            let dst = out_slice_mut(o, 1)?;
            dst[0] = value;
            Ok(None)
        }
    }
}

/// `mse_loss_backward_device`／`..._into_tracked` の共有本体。出力は `pred.shape()`。
pub(super) fn mse_loss_backward_device_core(
    pred: &DeviceBuffer<f32>,
    target: &DeviceBuffer<f32>,
    scale: f32,
    out: Option<&mut DeviceBuffer<f32>>,
) -> Result<Option<DeviceBuffer<f32>>, BackendError> {
    if pred.device() != Device::Cpu || target.device() != Device::Cpu {
        return Err(BackendError::DeviceMismatch);
    }
    if pred.shape() != target.shape() {
        return Err(shape_mismatch(pred.shape(), target.shape()));
    }
    let numel = pred.numel();
    if out.is_some() && numel == 0 {
        return Err(unsupported_empty("mse_loss_backward_device_into_tracked"));
    }
    if let Some(o) = out.as_deref() {
        check_cpu_into_out(o, pred.shape())?;
    }
    let (p, t) = (buffer_slice(pred)?, buffer_slice(target)?);
    match out {
        None => {
            let mut dst = fandhe_ai_tensor_core::alloc::take_zeroed_f32(numel);
            mse::mse_loss_backward_f32(p, t, scale, &mut dst)?;
            wrap(dst, pred.shape().to_vec()).map(Some)
        }
        Some(o) => {
            let dst = out_slice_mut(o, numel)?;
            // 全要素を代入する核だが、確保版（`take_zeroed_f32`）と前提をそろえる。
            dst.fill(0.0);
            mse::mse_loss_backward_f32(p, t, scale, dst)?;
            Ok(None)
        }
    }
}

/// ReLU の backward マスク（乗算ではなく選択）の共有本体。`out[i] = mask[i] > 0 ? g[i] : +0.0`。
/// 参照は `autodiff::grad::elementwise_mul_mask(.., |v| v > 0.0)`。
pub(super) fn relu_backward_mask_device_core(
    g: DeviceBufferView<'_>,
    g_layout: DeviceMatrixLayout,
    mask_src: DeviceBufferView<'_>,
    out: Option<&mut DeviceBuffer<f32>>,
) -> Result<Option<DeviceBuffer<f32>>, BackendError> {
    let g_shape = g.shape();
    if g_shape.len() != 2 {
        return Err(BackendError::ShapeMismatch(ShapeError::RankMismatch {
            expected: 2,
            actual: g_shape.len(),
        }));
    }
    let (rows, cols) = match g_layout {
        DeviceMatrixLayout::RowMajor => (g_shape[0], g_shape[1]),
        DeviceMatrixLayout::Transposed => (g_shape[1], g_shape[0]),
        _ => {
            return Err(BackendError::Unsupported(format!(
                "relu_backward_mask_device: unsupported layout {g_layout:?}"
            )));
        }
    };
    let logical = [rows, cols];
    if mask_src.shape() != logical {
        return Err(shape_mismatch(mask_src.shape(), &logical));
    }
    let numel = rows.checked_mul(cols).ok_or(BackendError::ShapeMismatch(
        ShapeError::ElementCountOverflow,
    ))?;
    if out.is_some() && numel == 0 {
        return Err(unsupported_empty("relu_backward_mask_device_into_tracked"));
    }
    if let Some(o) = out.as_deref() {
        check_cpu_into_out(o, &logical)?;
    }
    let (gs, ms) = (view_slice(&g)?, view_slice(&mask_src)?);
    let fill = |dst: &mut [f32]| match g_layout {
        DeviceMatrixLayout::Transposed => {
            // 格納 [cols, rows] の (j, i) を論理 (i, j) として読む。
            for i in 0..rows {
                for j in 0..cols {
                    let idx = i * cols + j;
                    dst[idx] = if ms[idx] > 0.0 { gs[j * rows + i] } else { 0.0 };
                }
            }
        }
        _ => {
            for ((o, &gv), &mv) in dst.iter_mut().zip(gs.iter()).zip(ms.iter()) {
                *o = if mv > 0.0 { gv } else { 0.0 };
            }
        }
    };
    match out {
        None => {
            let mut dst = vec![0.0f32; numel];
            fill(&mut dst);
            wrap(dst, logical.to_vec()).map(Some)
        }
        Some(o) => {
            fill(out_slice_mut(o, numel)?);
            Ok(None)
        }
    }
}

/// `eval::reduce_bias_grad_rows` と同一アルゴリズムの bias 縮約（`.claude/rules/coding-rust.md` の
/// `f64` アキュムレータ契約）。論理 `b: [k, n]` の行方向和を行昇順に `f64` で逐次加算し、最後に
/// 1 回だけ `f32` へ downcast する。`k == 1` は値をそのまま複製する（`-0.0` を保持する）。
/// 列ごとに独立した累積なので、列ループを外側に置いても加算順序（行昇順）は変わらない。
fn reduce_bias_rows_f64(
    b: &[f32],
    b_layout: DeviceMatrixLayout,
    k: usize,
    n: usize,
    dst: &mut [f32],
) {
    for (col, d) in dst.iter_mut().enumerate().take(n) {
        let at = |row: usize| match b_layout {
            DeviceMatrixLayout::Transposed => b[col * k + row],
            _ => b[row * n + col],
        };
        *d = if k == 1 {
            at(0)
        } else {
            let mut acc = 0f64;
            for row in 0..k {
                acc += f64::from(at(row));
            }
            acc as f32
        };
    }
}

/// dW の GEMM（`gemm_fp32_strict_into_device`）の本体。`bias` が `Some` なら必ず書いて `Ok(true)`。
#[allow(clippy::too_many_arguments)]
pub(super) fn gemm_fp32_strict_into_device_core(
    a: DeviceBufferView<'_>,
    a_layout: DeviceMatrixLayout,
    b: DeviceBufferView<'_>,
    b_layout: DeviceMatrixLayout,
    out: &mut DeviceBuffer<f32>,
    out_offset: usize,
    bias: Option<(usize, usize)>,
) -> Result<bool, BackendError> {
    const NAME: &str = "gemm_fp32_strict_into_device";
    if a.device() != Device::Cpu || b.device() != Device::Cpu || out.device() != Device::Cpu {
        return Err(BackendError::DeviceMismatch);
    }
    let (a_s, b_s) = (a.shape(), b.shape());
    for s in [a_s, b_s] {
        if s.len() != 2 {
            return Err(BackendError::ShapeMismatch(ShapeError::RankMismatch {
                expected: 2,
                actual: s.len(),
            }));
        }
    }
    let (m, k) = match a_layout {
        DeviceMatrixLayout::RowMajor => (a_s[0], a_s[1]),
        DeviceMatrixLayout::Transposed => (a_s[1], a_s[0]),
        _ => {
            return Err(BackendError::Unsupported(format!(
                "{NAME}: unsupported a_layout"
            )));
        }
    };
    let (kb, n) = match b_layout {
        DeviceMatrixLayout::RowMajor => (b_s[0], b_s[1]),
        DeviceMatrixLayout::Transposed => (b_s[1], b_s[0]),
        _ => {
            return Err(BackendError::Unsupported(format!(
                "{NAME}: unsupported b_layout"
            )));
        }
    };
    if k != kb {
        return Err(shape_mismatch(a_s, b_s));
    }
    // ホストは (Transposed, Transposed) を contiguous() 実体化の別経路で解く。1 対 1 の核が無く、
    // 別の核で黙って計算しないため `Unsupported` とする（チェーンは使わない形）。
    if matches!(
        (a_layout, b_layout),
        (
            DeviceMatrixLayout::Transposed,
            DeviceMatrixLayout::Transposed
        )
    ) {
        return Err(BackendError::Unsupported(format!(
            "{NAME}: (Transposed, Transposed) has no one-to-one host kernel"
        )));
    }
    let mn = m
        .checked_mul(n)
        .ok_or_else(|| invalid(format!("{NAME}: m * n overflowed usize")))?;
    let end = out_offset
        .checked_add(mn)
        .ok_or_else(|| invalid(format!("{NAME}: out_offset + m * n overflowed usize")))?;
    if end > out.numel() {
        return Err(invalid(format!(
            "{NAME}: write range [{out_offset}, {end}) exceeds out buffer length {}",
            out.numel()
        )));
    }
    let bias_range = match bias {
        Some((boff, nb)) => {
            if nb != n {
                return Err(invalid(format!("{NAME}: bias width {nb} != n {n}")));
            }
            let bend = boff
                .checked_add(n)
                .ok_or_else(|| invalid(format!("{NAME}: bias_offset + n overflowed usize")))?;
            if bend > out.numel() {
                return Err(invalid(format!(
                    "{NAME}: bias range [{boff}, {bend}) exceeds out buffer length {}",
                    out.numel()
                )));
            }
            // weight 範囲との重なりは禁止（空範囲は重ならない）。
            if n > 0 && mn > 0 && boff < end && out_offset < bend {
                return Err(invalid(format!(
                    "{NAME}: bias range [{boff}, {bend}) overlaps weight range \
                     [{out_offset}, {end})"
                )));
            }
            Some((boff, bend))
        }
        None => None,
    };
    let (a_data, b_data) = (view_slice(&a)?, view_slice(&b)?);
    // 書き込み前に out の実体長を検査する（Err 時は weight も bias も書かない契約）。
    check_out_backing(out)?;
    let all = out_slice_mut(out, out.numel())?;

    let dst = &mut all[out_offset..end];
    // BLIS 核は C を load して書き戻すため、再利用する出力は 0 埋めが必須（ホストと同じ）。
    dst.fill(0.0);
    // 退化形状（m・n・k のいずれかが 0）は全 0 のまま（ホストは別分岐だが結果は全 0 で一致）。
    if m > 0 && n > 0 && k > 0 {
        let r = match (a_layout, b_layout) {
            (DeviceMatrixLayout::Transposed, _) => {
                gemm_blis_parallel_tn(a_data, b_data, dst, m, n, k)
            }
            (_, DeviceMatrixLayout::Transposed) => {
                gemm_blis_parallel_nt(a_data, b_data, dst, m, n, k)
            }
            _ => gemm_blis_parallel(a_data, b_data, dst, m, n, k),
        };
        r.map_err(|e| BackendError::KernelLaunchFailed(e.to_string()))?;
    }
    if let Some((boff, bend)) = bias_range {
        reduce_bias_rows_f64(b_data, b_layout, k, n, &mut all[boff..bend]);
        return Ok(true);
    }
    Ok(false)
}

fn check_out_backing(out: &DeviceBuffer<f32>) -> Result<(), BackendError> {
    check_cpu_into_out(out, out.shape())
}

/// d_input の GEMM（`W @ op(B)`）の共有本体。`W: [p, q]` 行優先、`B` の論理形は `[q, r]`。出力 `[p, r]`。
pub(super) fn gemm_resident_lhs_device_core(
    w: DeviceBufferView<'_>,
    b: DeviceBufferView<'_>,
    b_layout: DeviceMatrixLayout,
    out: Option<&mut DeviceBuffer<f32>>,
) -> Result<Option<DeviceBuffer<f32>>, BackendError> {
    let (w_s, b_s) = (w.shape(), b.shape());
    for s in [w_s, b_s] {
        if s.len() != 2 {
            return Err(BackendError::ShapeMismatch(ShapeError::RankMismatch {
                expected: 2,
                actual: s.len(),
            }));
        }
    }
    let (p, q) = (w_s[0], w_s[1]);
    let (qb, r) = match b_layout {
        DeviceMatrixLayout::RowMajor => (b_s[0], b_s[1]),
        DeviceMatrixLayout::Transposed => (b_s[1], b_s[0]),
        _ => {
            return Err(BackendError::Unsupported(
                "gemm_resident_lhs_device: unsupported b_layout".into(),
            ));
        }
    };
    if q != qb {
        return Err(shape_mismatch(w_s, b_s));
    }
    let numel = p.checked_mul(r).ok_or(BackendError::ShapeMismatch(
        ShapeError::ElementCountOverflow,
    ))?;
    if out.is_some() && (numel == 0 || q == 0) {
        return Err(unsupported_empty("gemm_resident_lhs_device_into_tracked"));
    }
    let out_shape = [p, r];
    if let Some(o) = out.as_deref() {
        check_cpu_into_out(o, &out_shape)?;
    }
    let (w_data, b_data) = (view_slice(&w)?, view_slice(&b)?);
    let run = |dst: &mut [f32]| -> Result<(), BackendError> {
        dst.fill(0.0);
        if numel == 0 || q == 0 {
            return Ok(());
        }
        let res = match b_layout {
            DeviceMatrixLayout::Transposed => gemm_blis_parallel_nt(w_data, b_data, dst, p, r, q),
            _ => gemm_blis_parallel(w_data, b_data, dst, p, r, q),
        };
        res.map_err(|e| BackendError::KernelLaunchFailed(e.to_string()))
    };
    match out {
        None => {
            let mut dst = fandhe_ai_tensor_core::alloc::take_zeroed_f32(numel);
            run(&mut dst)?;
            wrap(dst, out_shape.to_vec()).map(Some)
        }
        Some(o) => {
            run(out_slice_mut(o, numel)?)?;
            Ok(None)
        }
    }
}
