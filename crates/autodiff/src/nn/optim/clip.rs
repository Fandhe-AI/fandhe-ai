//! gradient clipping（global L2 norm 方式・要素値 clip 方式の 2 系統。
//! 親イシュー #192・#195（norm 方式）・#1753（value 方式・親 #1631）。
//!
//! **norm 方式**（[`clip_grad_norm`]）は PyTorch
//! `torch.nn.utils.clip_grad_norm_` と同一の定義・適用順序に揃える:
//! 複数テンソルの勾配をひとつの仮想ベクトルとみなした L2 ノルム
//! （global norm）を計算し、`max_norm` を超えている場合のみ全テンソルへ
//! 同一スケール係数を掛けて縮小する（方向を保存したまま大きさだけ抑える）。
//!
//! **value 方式**（[`clip_grad_value`]）は PyTorch
//! `torch.nn.utils.clip_grad_value_` と同一の定義に揃える: 各勾配要素を
//! 独立に `[-clip_value, clip_value]` へクランプする（テンソル間の相関
//! を見ないため norm 方式と異なりスケーリングを伴わず、範囲内の要素は
//! bit 同一のまま返る）。
//!
//! 本モジュールは `Gradients`/`Var`（`tape.rs`・`var.rs`）に依存しない
//! 純関数として実装する。`optim/mod.rs` の適用順序契約
//! （backward → unscale → clip → optimizer step）に従い、呼び出し元
//! （SGD/AdamW 実装・#193/#194）が `Gradients::get` で取り出した
//! `&Tensor<f32>` 群をここへ渡す運用を想定する。

use fandhe_ai_tensor_core::Tensor;

use crate::error::AutodiffError;

/// `grad` の要素を行優先メモリ順の `Vec` として読み出す
/// （`as_slice` は非 contiguous なら `None` を返すため `contiguous()`
/// フォールバックを内包する）。
///
/// `contiguous()` 直後は `is_contiguous() == true` が
/// `tensor-core::Tensor::contiguous` の契約（`tensor.rs` doc）で保証
/// されるため理論上 `as_slice` は必ず `Some` を返すが、本番経路で
/// `unwrap`/`expect` を使わない方針（`.claude/rules/coding-rust.md`）
/// に沿い、契約違反時も panic せず `AutodiffError::Backward` を返す
/// （到達すれば `tensor-core` 側のバグであり fail-closed で検知する）。
fn read_contiguous(grad: &Tensor<f32>) -> Result<Vec<f32>, AutodiffError> {
    if let Some(slice) = grad.as_slice() {
        return Ok(slice.to_vec());
    }
    let owned = grad.contiguous();
    owned.as_slice().map(<[f32]>::to_vec).ok_or_else(|| {
        AutodiffError::Backward(
            "contiguous() 後も as_slice が None を返した（tensor-core 契約違反）".to_string(),
        )
    })
}

/// 複数テンソルの勾配を横断した global L2 ノルム。
///
/// `sqrt(sum_i sum_j grad_i[j]^2)`（PyTorch `clip_grad_norm_` の
/// `norm_type=2.0` 定義と同一）。二乗和の蓄積は `f32::mul_add`
/// （`g*g + acc`）で行い、CPU 参照実装の FMA 契約
/// （`.claude/rules/coding-rust.md`）に揃える。テンソルの走査順は
/// 引数スライスの順序・各テンソル内は `read_contiguous` のメモリ順
/// （行優先。非 contiguous なら `contiguous()` フォールバック）で固定
/// され、結果は決定的。
///
/// 空スライス（勾配なし）は 0.0 を返す（正常系。PyTorch も
/// パラメータ集合が空なら 0 を返す）。
///
/// # Errors
///
/// 二乗和・平方根の結果が非有限（NaN/Inf 勾配・オーバーフロー）に
/// なった場合は `AutodiffError::InvalidArgument` を返す
/// （PyTorch `error_if_nonfinite=True` 相当。壊れた勾配のまま
/// clip・optimizer step へ進めない fail-closed 判断。
/// `.claude/rules/security.md` A08 整合性）。
pub fn global_grad_norm(grads: &[&Tensor<f32>]) -> Result<f32, AutodiffError> {
    let mut sum_sq = 0.0f32;
    for grad in grads {
        let values = read_contiguous(grad)?;
        for value in values {
            sum_sq = value.mul_add(value, sum_sq);
        }
    }

    let norm = sum_sq.sqrt();
    if !norm.is_finite() {
        return Err(AutodiffError::InvalidArgument(format!(
            "global grad norm は非有限（勾配に NaN/Inf が含まれるか、二乗和がオーバーフローした）: {norm}"
        )));
    }
    Ok(norm)
}

/// [`clip_grad_norm`] の結果。
pub struct ClipGradResult {
    /// clip 後の勾配（`grads` と同じ順序・shape）。`scaled == false`
    /// の場合も呼び出し側の統一的な扱いのため複製して返す。
    pub grads: Vec<Tensor<f32>>,
    /// clip 前の global L2 ノルム（ログ・監視用途に呼び出し元へ公開する）。
    pub total_norm: f32,
    /// スケール係数を実際に適用したかどうか（`total_norm <= max_norm`
    /// なら `false`）。
    pub scaled: bool,
}

/// global norm 方式の gradient clipping。
///
/// `clip_coef = max_norm / (total_norm + 1e-6)`（PyTorch
/// `clip_grad_norm_` と同一のゼロ除算回避定数）。`clip_coef >= 1.0`
/// （= `total_norm <= max_norm`）の場合は勾配を無変更のまま返す
/// （不要なスケーリングによる丸め誤差混入を避ける。PyTorch も
/// `clip_coef_clamped = min(clip_coef, 1.0)` で同じ挙動）。
///
/// # Errors
///
/// - `max_norm` が非有限または 0 以下 → `AutodiffError::InvalidArgument`
///   （fail-closed。負値・0 は「常に勾配をゼロへ潰す」設定であり、
///   呼び出し側の誤り混入を早期に検出する）
/// - [`global_grad_norm`] が Err を返した場合はそのまま伝播する
pub fn clip_grad_norm(
    grads: &[&Tensor<f32>],
    max_norm: f32,
) -> Result<ClipGradResult, AutodiffError> {
    if !max_norm.is_finite() || max_norm <= 0.0 {
        return Err(AutodiffError::InvalidArgument(format!(
            "max_norm は有限かつ正の値でなければならない: {max_norm}"
        )));
    }

    let total_norm = global_grad_norm(grads)?;

    // ゼロ除算回避定数 1e-6 は PyTorch `clip_grad_norm_` 実装と同一
    // （total_norm がちょうど 0（勾配なし・全ゼロ勾配）でも安全に
    // 割り算できるようにする）。
    let clip_coef = max_norm / (total_norm + 1e-6);
    let scaled = clip_coef < 1.0;

    let mut out_grads = Vec::with_capacity(grads.len());
    for grad in grads {
        if !scaled {
            out_grads.push((*grad).clone());
            continue;
        }
        let values = read_contiguous(grad)?;
        let scaled_data: Vec<f32> = values.iter().map(|&v| v * clip_coef).collect();
        // shape はスケール前の `grad.shape()` から取得しており
        // `scaled_data` の要素数は `values`（= `grad` の全要素）と
        // 常に一致するため、通常は `ShapeError` にはならない。
        // それでも `?` で型付きエラーとして伝播し `unwrap`/`expect`
        // は使わない（`.claude/rules/coding-rust.md`）。
        let tensor = Tensor::from_slice(&scaled_data, grad.shape())?;
        out_grads.push(tensor);
    }

    Ok(ClipGradResult {
        grads: out_grads,
        total_norm,
        scaled,
    })
}

/// value 方式の gradient clipping（PyTorch `torch.nn.utils.clip_grad_value_`
/// 相当）。各勾配要素を独立に `[-clip_value, clip_value]` へクランプする。
///
/// [`clip_grad_norm`] と異なりテンソル間の相関（global L2 norm）を見ず、
/// 各要素を独立にクランプするためスケーリングを伴わない（範囲内の要素は
/// 無変更のまま返す。丸め誤差混入が原理的に起きない）。
///
/// `f32::clamp`（NaN 境界で panic しうる API）は使わず `v.max(-clip_value)
/// .min(clip_value)` で実装する（本番経路に panic 可能性を残さない方針。
/// `clip_value` は下記の事前検査で有限かつ正であることが保証されており
/// 到達時点で NaN 境界は発生しないが、検査ロジックとクランプロジックを
/// 疎結合に保つため）。
///
/// # Errors
///
/// - `clip_value` が非有限または 0 以下 → `AutodiffError::InvalidArgument`
///   （fail-closed。[`clip_grad_norm`] の `max_norm` 検査と同型。0 は
///   「常に勾配をゼロへ潰す」設定であり、呼び出し側の誤り混入を早期に
///   検出する）
/// - 勾配要素に非有限（NaN/±Inf）が含まれる →
///   `AutodiffError::InvalidArgument`。`±Inf.max(-c).min(c)` は `±c` へ、
///   `NaN.max(-c).min(c)` は `-c` へ**静かに正規化されて壊れた勾配を
///   隠蔽してしまう**ため、クランプ前に全要素の有限性を検査する
///   （`.claude/rules/security.md` A08 整合性。[`global_grad_norm`] が
///   非有限入力に対し Err を返す契約と揃える）
pub fn clip_grad_value(
    grads: &[&Tensor<f32>],
    clip_value: f32,
) -> Result<Vec<Tensor<f32>>, AutodiffError> {
    if !clip_value.is_finite() || clip_value <= 0.0 {
        return Err(AutodiffError::InvalidArgument(format!(
            "clip_value は有限かつ正の値でなければならない: {clip_value}"
        )));
    }

    let mut out_grads = Vec::with_capacity(grads.len());
    for grad in grads {
        let values = read_contiguous(grad)?;
        if let Some(&non_finite) = values.iter().find(|v| !v.is_finite()) {
            return Err(AutodiffError::InvalidArgument(format!(
                "clip_grad_value に非有限の勾配要素が含まれる（NaN/Inf をクランプで\
                 隠蔽しない fail-closed 判断）: {non_finite}"
            )));
        }
        let clipped: Vec<f32> = values
            .iter()
            .map(|&v| v.max(-clip_value).min(clip_value))
            .collect();
        // shape は `grad.shape()` から取得しており `clipped` の要素数は
        // `values`（= `grad` の全要素）と常に一致するため、通常は
        // `ShapeError` にはならない。それでも `?` で型付きエラーとして
        // 伝播し `unwrap`/`expect` は使わない
        // （`.claude/rules/coding-rust.md`）。
        let tensor = Tensor::from_slice(&clipped, grad.shape())?;
        out_grads.push(tensor);
    }

    Ok(out_grads)
}
