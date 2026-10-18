//! loss 5 種（mse・huber/smooth_l1・bce・nll・kl_div）と backward の ROCm 実行入口
//! （`backend-cuda` の `mse.rs`・`huber.rs`・`bce.rs`・`nll.rs`・`kl_div.rs` の移植。
//! イシュー #3436・分解案 R5）。
//!
//! 役割: [`RocmLoss`] が HIP/HIPRTC をロードして `kernels_loss` の 1 ソースをコンパイルし、
//! ホストのスライスから forward のスカラー損失・backward の勾配 `Vec` を返す。`Tensor` 化・
//! shape 検査・`Mean`／`Sum` から `factor` への変換・enum から `i32` への写像は
//! `RocmBackendOps`（`ops.rs`）の責務で、本型はスライス入出力のみ。`ops.rs` の 10 メソッドが
//! [`cached_loss`] 経由で呼ぶ。ホスト側の検証（長さ一致・`i32` 収容・nll の添字値域）は
//! 起動前に純関数で行い（カーネル側の REQ-8 と二重化）、ROCm がなくても単体テストできる。
//!
//! forward は 2 回起動する: `(num_blocks,1,1)×(256,1,1)` の partial → `(1,1,1)×(256,1,1)` の
//! finalize。`num_blocks` は [`num_blocks`] が `min(ceil_div(n, 256), 1024)` で決め、`partial`
//! の確保長と同じ値から導く（ずれると finalize が境界外を読む）。backward は 1 スレッド 1 要素
//! （nll は 1 サンプル 1 書き込みで、出力を 0 初期化してから起動する）。要素数 0 のときは HIP に
//! 触れない（forward は `0.0`、backward は空または 0 埋め）。

use std::cell::RefCell;
use std::ffi::c_void;
use std::rc::Rc;

use crate::error::RocmError;
use crate::kernels_loss::{
    BCE_BACKWARD_SPEC, BCE_PARTIAL_SPEC, FINALIZE_SPEC, HUBER_BACKWARD_SPEC, HUBER_PARTIAL_SPEC,
    KL_DIV_BACKWARD_SPEC, KL_DIV_PARTIAL_SPEC, LOSS_BLOCK_DIM, LOSS_MAX_BLOCKS, MSE_BACKWARD_SPEC,
    MSE_PARTIAL_SPEC, NLL_BACKWARD_SPEC, NLL_PARTIAL_SPEC, render_loss_source,
};
use crate::launch::{Arg, KernelHost, KernelSpec};
use crate::target::GfxTarget;
use crate::typed_buffer::TypedDeviceBuffer;
use crate::warp_geometry::WarpGeometry;

const OP_MSE: &str = "mse_loss";
const OP_HUBER: &str = "huber_loss";
const OP_BCE: &str = "bce_loss";
const OP_KL_DIV: &str = "kl_div_loss";
const OP_NLL: &str = "nll_loss";

fn fit_i32(op: &'static str, value: usize, name: &str) -> Result<i32, RocmError> {
    i32::try_from(value).map_err(|_| RocmError::InvalidOpArgs {
        op,
        detail: format!("{name}={value} は i32（カーネル引数型）に収まりません"),
    })
}

/// 2 つの入力の長さ一致と `i32` 収容を検証し、要素数を返す。
pub(crate) fn validate_pair(
    op: &'static str,
    a_len: usize,
    b_len: usize,
) -> Result<usize, RocmError> {
    if a_len != b_len {
        return Err(RocmError::InvalidOpArgs {
            op,
            detail: format!("入力の長さ不一致: {a_len} と {b_len}"),
        });
    }
    fit_i32(op, a_len, "numel")?;
    Ok(a_len)
}

/// kind が allowlist（0 か 1）に入ることを検証する。未知の値は fail-closed で拒否する。
pub(crate) fn validate_kind(op: &'static str, kind: i32) -> Result<(), RocmError> {
    if matches!(kind, 0 | 1) {
        Ok(())
    } else {
        Err(RocmError::InvalidOpArgs {
            op,
            detail: format!("kind={kind} は 0 か 1 のみ受け付けます"),
        })
    }
}

/// forward 1 段目の起動ブロック数 `min(ceil_div(n, LOSS_BLOCK_DIM), LOSS_MAX_BLOCKS)`。
///
/// `n >= 1` が前提（0 は呼び出し側が早期リターンする）。`partial` の確保長と finalize の
/// `num_partials` はこの値から導く。
pub(crate) fn num_blocks(n: usize) -> usize {
    n.div_ceil(LOSS_BLOCK_DIM as usize)
        .min(LOSS_MAX_BLOCKS as usize)
}

/// 検証済みの nll 寸法（カーネル引数は `i32`）。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct NllDims {
    /// `outer * inner`。
    pub(crate) n_samples: usize,
    /// `outer * num_classes * inner`。
    pub(crate) numel: usize,
    pub(crate) outer_i: i32,
    pub(crate) classes_i: i32,
    pub(crate) inner_i: i32,
    pub(crate) n_i: i32,
}

/// `targets` のうち `0 <= t < num_classes` を外れる最初の値。
pub(crate) fn first_bad_target(targets: &[i32], num_classes: usize) -> Option<i32> {
    targets
        .iter()
        .copied()
        .find(|&t| t < 0 || (t as usize) >= num_classes)
}

/// nll の寸法を checked 演算で求め、長さ・`i32` 収容・添字値域を検証する。
///
/// `input_len` は forward のとき `Some`（`outer*C*inner` と一致が必要）、backward は `None`。
pub(crate) fn validate_nll(
    op: &'static str,
    input_len: Option<usize>,
    targets: &[i32],
    outer: usize,
    num_classes: usize,
    inner: usize,
) -> Result<NllDims, RocmError> {
    let ovf = |what: &str| RocmError::InvalidOpArgs {
        op,
        detail: format!("{what} が usize を超えます"),
    };
    let n_samples = outer.checked_mul(inner).ok_or_else(|| ovf("outer*inner"))?;
    let numel = n_samples
        .checked_mul(num_classes)
        .ok_or_else(|| ovf("outer*num_classes*inner"))?;
    if let Some(len) = input_len
        && len != numel
    {
        return Err(RocmError::InvalidOpArgs {
            op,
            detail: format!("input の長さ不一致: 期待 {numel}, 実際 {len}"),
        });
    }
    if targets.len() != n_samples {
        return Err(RocmError::InvalidOpArgs {
            op,
            detail: format!(
                "targets の長さ不一致: 期待 {n_samples}, 実際 {}",
                targets.len()
            ),
        });
    }
    let dims = NllDims {
        n_samples,
        numel,
        outer_i: fit_i32(op, outer, "outer")?,
        classes_i: fit_i32(op, num_classes, "num_classes")?,
        inner_i: fit_i32(op, inner, "inner")?,
        n_i: fit_i32(op, n_samples, "n_samples")?,
    };
    fit_i32(op, numel, "numel")?;
    if let Some(bad) = first_bad_target(targets, num_classes) {
        return Err(RocmError::InvalidOpArgs {
            op,
            detail: format!("target index {bad} が範囲 [0, {num_classes}) の外です"),
        });
    }
    Ok(dims)
}

/// コンパイル済みの loss カーネル群を保持し、実行する。
pub struct RocmLoss {
    host: KernelHost,
}

impl RocmLoss {
    /// HIP/HIPRTC をロードし loss 用の 1 ソースを `target` 向けにコンパイルする。
    ///
    /// wave 幅は `GfxTarget` の固定値から決める。ROCm 非搭載では `LibraryUnavailable`、
    /// デバイスがなければ `NoDevice`（panic しない）。
    pub fn new(target: GfxTarget) -> Result<Self, RocmError> {
        let src = render_loss_source(WarpGeometry::for_target(target))?;
        Ok(Self {
            host: KernelHost::new(&src, target)?,
        })
    }

    /// mse forward: `factor * Σ (pred - target)^2`（`factor` は `Mean` なら `1/n`・`Sum` なら `1`）。
    pub fn run_mse_loss_f32(
        &self,
        pred: &[f32],
        target: &[f32],
        factor: f32,
    ) -> Result<f32, RocmError> {
        let n = validate_pair(OP_MSE, pred.len(), target.len())?;
        if n == 0 {
            return Ok(0.0);
        }
        let _guard = self.host.select()?;
        let dp = TypedDeviceBuffer::<f32>::from_slice(self.host.runtime(), pred)?;
        let dt = TypedDeviceBuffer::<f32>::from_slice(self.host.runtime(), target)?;
        self.reduce(OP_MSE, &MSE_PARTIAL_SPEC, n, factor, |partial| {
            vec![
                Arg::Ptr(dp.as_device_ptr()),
                Arg::Ptr(dt.as_device_ptr()),
                Arg::Ptr(partial),
                Arg::I32(n as i32),
            ]
        })
    }

    /// mse backward: `dpred[i] = scale * (pred[i] - target[i])`。
    pub fn run_mse_backward_f32(
        &self,
        pred: &[f32],
        target: &[f32],
        scale: f32,
    ) -> Result<Vec<f32>, RocmError> {
        let n = validate_pair(OP_MSE, pred.len(), target.len())?;
        if n == 0 {
            return Ok(Vec::new());
        }
        let _guard = self.host.select()?;
        let dp = TypedDeviceBuffer::<f32>::from_slice(self.host.runtime(), pred)?;
        let dt = TypedDeviceBuffer::<f32>::from_slice(self.host.runtime(), target)?;
        let dout = TypedDeviceBuffer::<f32>::new_zeroed(self.host.runtime(), n)?;
        let args = [
            Arg::Ptr(dp.as_device_ptr()),
            Arg::Ptr(dt.as_device_ptr()),
            Arg::Ptr(dout.as_device_ptr()),
            Arg::I32(n as i32),
            Arg::F32(scale),
        ];
        self.host.run(OP_MSE, &MSE_BACKWARD_SPEC, n, &args)?;
        dout.to_vec()
    }

    /// huber／smooth_l1 forward（`kind`: 0=Huber・1=SmoothL1）。
    pub fn run_huber_loss_f32(
        &self,
        pred: &[f32],
        target: &[f32],
        kind: i32,
        delta: f32,
        factor: f32,
    ) -> Result<f32, RocmError> {
        let n = validate_pair(OP_HUBER, pred.len(), target.len())?;
        validate_kind(OP_HUBER, kind)?;
        if n == 0 {
            return Ok(0.0);
        }
        let _guard = self.host.select()?;
        let dp = TypedDeviceBuffer::<f32>::from_slice(self.host.runtime(), pred)?;
        let dt = TypedDeviceBuffer::<f32>::from_slice(self.host.runtime(), target)?;
        self.reduce(OP_HUBER, &HUBER_PARTIAL_SPEC, n, factor, |partial| {
            vec![
                Arg::Ptr(dp.as_device_ptr()),
                Arg::Ptr(dt.as_device_ptr()),
                Arg::Ptr(partial),
                Arg::I32(n as i32),
                Arg::I32(kind),
                Arg::F32(delta),
            ]
        })
    }

    /// huber／smooth_l1 backward: `dpred[i] = scale * grad(pred[i] - target[i])`。
    pub fn run_huber_backward_f32(
        &self,
        pred: &[f32],
        target: &[f32],
        kind: i32,
        delta: f32,
        scale: f32,
    ) -> Result<Vec<f32>, RocmError> {
        let n = validate_pair(OP_HUBER, pred.len(), target.len())?;
        validate_kind(OP_HUBER, kind)?;
        if n == 0 {
            return Ok(Vec::new());
        }
        let _guard = self.host.select()?;
        let dp = TypedDeviceBuffer::<f32>::from_slice(self.host.runtime(), pred)?;
        let dt = TypedDeviceBuffer::<f32>::from_slice(self.host.runtime(), target)?;
        let dout = TypedDeviceBuffer::<f32>::new_zeroed(self.host.runtime(), n)?;
        let args = [
            Arg::Ptr(dp.as_device_ptr()),
            Arg::Ptr(dt.as_device_ptr()),
            Arg::Ptr(dout.as_device_ptr()),
            Arg::I32(n as i32),
            Arg::I32(kind),
            Arg::F32(delta),
            Arg::F32(scale),
        ];
        self.host.run(OP_HUBER, &HUBER_BACKWARD_SPEC, n, &args)?;
        dout.to_vec()
    }

    /// bce forward（`kind`: 0=Probabilities・1=Logits）。
    pub fn run_bce_loss_f32(
        &self,
        input: &[f32],
        target: &[f32],
        kind: i32,
        factor: f32,
    ) -> Result<f32, RocmError> {
        self.run_kind_forward(OP_BCE, &BCE_PARTIAL_SPEC, input, target, kind, factor)
    }

    /// bce backward: `dinput[i] = scale * grad(input[i], target[i])`。
    pub fn run_bce_backward_f32(
        &self,
        input: &[f32],
        target: &[f32],
        kind: i32,
        scale: f32,
    ) -> Result<Vec<f32>, RocmError> {
        self.run_kind_backward(OP_BCE, &BCE_BACKWARD_SPEC, input, target, kind, scale)
    }

    /// kl_div forward（`kind`: 0=Probabilities・1=LogProbabilities）。
    pub fn run_kl_div_loss_f32(
        &self,
        input: &[f32],
        target: &[f32],
        kind: i32,
        factor: f32,
    ) -> Result<f32, RocmError> {
        self.run_kind_forward(OP_KL_DIV, &KL_DIV_PARTIAL_SPEC, input, target, kind, factor)
    }

    /// kl_div backward: `dinput[i] = scale * grad(target[i])`。
    pub fn run_kl_div_backward_f32(
        &self,
        input: &[f32],
        target: &[f32],
        kind: i32,
        scale: f32,
    ) -> Result<Vec<f32>, RocmError> {
        self.run_kind_backward(OP_KL_DIV, &KL_DIV_BACKWARD_SPEC, input, target, kind, scale)
    }

    /// nll forward: `factor * Σ_s -input[(o*C + t_s)*inner + i]`（`input` は `[outer, C, inner]`）。
    pub fn run_nll_loss_f32(
        &self,
        input: &[f32],
        targets: &[i32],
        outer: usize,
        num_classes: usize,
        inner: usize,
        factor: f32,
    ) -> Result<f32, RocmError> {
        let d = validate_nll(
            OP_NLL,
            Some(input.len()),
            targets,
            outer,
            num_classes,
            inner,
        )?;
        if d.n_samples == 0 {
            return Ok(0.0);
        }
        let _guard = self.host.select()?;
        let di = TypedDeviceBuffer::<f32>::from_slice(self.host.runtime(), input)?;
        let dt = TypedDeviceBuffer::<i32>::from_slice(self.host.runtime(), targets)?;
        self.reduce(OP_NLL, &NLL_PARTIAL_SPEC, d.n_samples, factor, |partial| {
            vec![
                Arg::Ptr(di.as_device_ptr()),
                Arg::Ptr(dt.as_device_ptr()),
                Arg::Ptr(partial),
                Arg::I32(d.outer_i),
                Arg::I32(d.classes_i),
                Arg::I32(d.inner_i),
                Arg::I32(d.n_i),
            ]
        })
    }

    /// nll backward: `dinput[(o*C + t_s)*inner + i] = -scale`、ほかは 0（長さ `outer*C*inner`）。
    pub fn run_nll_backward_f32(
        &self,
        targets: &[i32],
        outer: usize,
        num_classes: usize,
        inner: usize,
        scale: f32,
    ) -> Result<Vec<f32>, RocmError> {
        let d = validate_nll(OP_NLL, None, targets, outer, num_classes, inner)?;
        if d.n_samples == 0 {
            return Ok(vec![0.0; d.numel]);
        }
        let _guard = self.host.select()?;
        let dt = TypedDeviceBuffer::<i32>::from_slice(self.host.runtime(), targets)?;
        // 書き込み先が一意なので atomic は不要。ターゲット位置以外は 0 のまま残す。
        let dout = TypedDeviceBuffer::<f32>::new_zeroed(self.host.runtime(), d.numel)?;
        let args = [
            Arg::Ptr(dt.as_device_ptr()),
            Arg::Ptr(dout.as_device_ptr()),
            Arg::I32(d.outer_i),
            Arg::I32(d.classes_i),
            Arg::I32(d.inner_i),
            Arg::I32(d.n_i),
            Arg::F32(scale),
        ];
        self.host
            .run(OP_NLL, &NLL_BACKWARD_SPEC, d.n_samples, &args)?;
        dout.to_vec()
    }

    fn run_kind_forward(
        &self,
        op: &'static str,
        spec: &KernelSpec,
        input: &[f32],
        target: &[f32],
        kind: i32,
        factor: f32,
    ) -> Result<f32, RocmError> {
        let n = validate_pair(op, input.len(), target.len())?;
        validate_kind(op, kind)?;
        if n == 0 {
            return Ok(0.0);
        }
        let _guard = self.host.select()?;
        let di = TypedDeviceBuffer::<f32>::from_slice(self.host.runtime(), input)?;
        let dt = TypedDeviceBuffer::<f32>::from_slice(self.host.runtime(), target)?;
        self.reduce(op, spec, n, factor, |partial| {
            vec![
                Arg::Ptr(di.as_device_ptr()),
                Arg::Ptr(dt.as_device_ptr()),
                Arg::Ptr(partial),
                Arg::I32(n as i32),
                Arg::I32(kind),
            ]
        })
    }

    fn run_kind_backward(
        &self,
        op: &'static str,
        spec: &KernelSpec,
        input: &[f32],
        target: &[f32],
        kind: i32,
        scale: f32,
    ) -> Result<Vec<f32>, RocmError> {
        let n = validate_pair(op, input.len(), target.len())?;
        validate_kind(op, kind)?;
        if n == 0 {
            return Ok(Vec::new());
        }
        let _guard = self.host.select()?;
        let di = TypedDeviceBuffer::<f32>::from_slice(self.host.runtime(), input)?;
        let dt = TypedDeviceBuffer::<f32>::from_slice(self.host.runtime(), target)?;
        let dout = TypedDeviceBuffer::<f32>::new_zeroed(self.host.runtime(), n)?;
        let args = [
            Arg::Ptr(di.as_device_ptr()),
            Arg::Ptr(dt.as_device_ptr()),
            Arg::Ptr(dout.as_device_ptr()),
            Arg::I32(n as i32),
            Arg::I32(kind),
            Arg::F32(scale),
        ];
        self.host.run(op, spec, n, &args)?;
        dout.to_vec()
    }

    /// 2 段縮約を起動する。`work` は 1 段目が処理する要素数（mse 等は numel・nll は n_samples）、
    /// `partial_args` は `partial` のデバイスポインタを受けて 1 段目の引数列を作る。
    ///
    /// 呼び出し元が `select()` のガードと入力バッファを保持している間に呼ぶこと。`partial` の
    /// 長さと 1 段目のグリッド数と finalize の `num_partials` は同じ `num_blocks(work)` から導く。
    fn reduce(
        &self,
        op: &'static str,
        partial_spec: &KernelSpec,
        work: usize,
        factor: f32,
        partial_args: impl FnOnce(*mut c_void) -> Vec<Arg>,
    ) -> Result<f32, RocmError> {
        let blocks = num_blocks(work);
        let blocks_u32 = fit_u32(op, blocks)?;
        let dpartial = TypedDeviceBuffer::<f32>::new_zeroed(self.host.runtime(), blocks)?;
        let dout = TypedDeviceBuffer::<f32>::new_zeroed(self.host.runtime(), 1)?;
        let block = (LOSS_BLOCK_DIM, 1, 1);
        let args1 = partial_args(dpartial.as_device_ptr());
        self.host
            .run_geometry(op, partial_spec, (blocks_u32, 1, 1), block, &args1)?;
        let args2 = [
            Arg::Ptr(dpartial.as_device_ptr()),
            Arg::Ptr(dout.as_device_ptr()),
            Arg::I32(blocks_u32 as i32),
            Arg::F32(factor),
        ];
        self.host
            .run_geometry(op, &FINALIZE_SPEC, (1, 1, 1), block, &args2)?;
        dout.to_vec()?
            .into_iter()
            .next()
            .ok_or_else(|| RocmError::InvalidOpArgs {
                op,
                detail: "finalize の出力が空です".into(),
            })
    }
}

fn fit_u32(op: &'static str, value: usize) -> Result<u32, RocmError> {
    u32::try_from(value).map_err(|_| RocmError::InvalidOpArgs {
        op,
        detail: format!("ブロック数 {value} が u32 に収まりません"),
    })
}

type LossCache = RefCell<Vec<(GfxTarget, Rc<RocmLoss>)>>;

thread_local! {
    /// target ごとにコンパイル済みの loss カーネル群を遅延保持する。失敗はキャッシュしない。
    static LOSS_CACHE: LossCache = const { RefCell::new(Vec::new()) };
}

/// thread_local キャッシュから `target` 用の [`RocmLoss`] を取得する（無ければ構築して入れる）。
///
/// `RocmBackendOps`（`Copy`・`Send`）が `!Send` の実行型を直接持てないための方式で、
/// `ops.rs` の GEMM キャッシュと同じ。呼び出し元は `ops.rs` の loss 10 メソッド。
pub(crate) fn cached_loss(target: GfxTarget) -> Result<Rc<RocmLoss>, RocmError> {
    let hit = LOSS_CACHE
        .try_with(|c| {
            c.borrow()
                .iter()
                .find(|(t, _)| *t == target)
                .map(|(_, l)| Rc::clone(l))
        })
        .map_err(|_| RocmError::InvalidKernelConfig {
            detail: "loss キャッシュ（thread_local）が破棄済みです".into(),
        })?;
    if let Some(l) = hit {
        return Ok(l);
    }
    // 借用を解いてから構築する（構築中に再入しても二重借用にならない）。
    let l = Rc::new(RocmLoss::new(target)?);
    let _ = LOSS_CACHE.try_with(|c| c.borrow_mut().push((target, Rc::clone(&l))));
    Ok(l)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn num_blocks_is_clamped_and_covers_input() {
        assert_eq!(num_blocks(1), 1);
        assert_eq!(num_blocks(256), 1);
        assert_eq!(num_blocks(257), 2);
        assert_eq!(num_blocks(256 * 1024), 1024);
        assert_eq!(num_blocks(256 * 1024 + 3), 1024);
        assert_eq!(num_blocks(usize::MAX), 1024);
    }

    #[test]
    fn pair_validation() {
        assert_eq!(validate_pair("t", 5, 5).unwrap(), 5);
        assert_eq!(validate_pair("t", 0, 0).unwrap(), 0);
        assert!(validate_pair("t", 5, 4).is_err());
        assert!(validate_pair("t", i32::MAX as usize + 1, i32::MAX as usize + 1).is_err());
        assert!(validate_pair("t", i32::MAX as usize, i32::MAX as usize).is_ok());
    }

    #[test]
    fn kind_allowlist() {
        assert!(validate_kind("t", 0).is_ok());
        assert!(validate_kind("t", 1).is_ok());
        assert!(validate_kind("t", 2).is_err());
        assert!(validate_kind("t", -1).is_err());
    }

    #[test]
    fn nll_validation_accepts_and_rejects() {
        let d = validate_nll("t", Some(2 * 3 * 4), &[0; 8], 2, 3, 4).unwrap();
        assert_eq!((d.n_samples, d.numel), (8, 24));
        // backward は input 長なし。
        assert_eq!(validate_nll("t", None, &[2; 8], 2, 3, 4).unwrap().numel, 24);
        // 空。
        assert_eq!(
            validate_nll("t", Some(0), &[], 0, 3, 4).unwrap().n_samples,
            0
        );
        // 長さ不一致。
        assert!(validate_nll("t", Some(23), &[0; 8], 2, 3, 4).is_err());
        assert!(validate_nll("t", Some(24), &[0; 7], 2, 3, 4).is_err());
        // 範囲外の添字。
        assert!(validate_nll("t", Some(24), &[3; 8], 2, 3, 4).is_err());
        assert!(validate_nll("t", None, &[-1; 8], 2, 3, 4).is_err());
        // C == 0 でサンプルがあれば全 t が範囲外。
        assert!(validate_nll("t", Some(0), &[0; 8], 2, 0, 4).is_err());
    }

    #[test]
    fn nll_validation_overflow_is_typed_error() {
        assert!(validate_nll("t", None, &[], usize::MAX, 1, 2).is_err());
        assert!(validate_nll("t", None, &[], 1 << 40, 1 << 40, 1).is_err());
        let big = i32::MAX as usize + 1;
        assert!(matches!(
            validate_nll("t", None, &[], 0, big, 0),
            Err(RocmError::InvalidOpArgs { .. })
        ));
    }

    #[test]
    fn first_bad_target_finds_boundaries() {
        assert_eq!(first_bad_target(&[0, 1, 2], 3), None);
        assert_eq!(first_bad_target(&[0, 3], 3), Some(3));
        assert_eq!(first_bad_target(&[0, -2], 3), Some(-2));
        assert_eq!(first_bad_target(&[i32::MAX], 3), Some(i32::MAX));
    }
}
