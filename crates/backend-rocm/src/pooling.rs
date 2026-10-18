//! MaxPool2d／AvgPool2d／AdaptiveAvgPool2d の ROCm 実行入口（`backend-cuda::pooling` の移植。
//! イシュー #3438・R7）。
//!
//! 役割: [`RocmPooling`] が HIP/HIPRTC をロードして `kernels_pooling` のソースをコンパイルし、行優先
//! contiguous のホストスライスから結果の `Vec` を返す。`BackendOps` の 3 メソッドの shape 再検査
//! （`pool2d_out_shape`／`adaptive_pool2d_out_shape`）・avg の dilation ≠ 1 の `Unsupported`・`Tensor` 化は
//! `RocmBackendOps`（`ops/conv_pool.rs`）側の責務で、本層は出力 shape の導出・検証と実行だけを持つ。
//!
//! 検証は HIP に触れない純関数 `validate_*` に集約する（A04。CUDA の `validate_and_shape` 系と同内容）:
//! kernel・stride・dilation ≥ 1、`padding <= kernel/2`、空間軸 ≥ 1、dilation による空窓の拒否、
//! adaptive の `output_size >= 1`、max の `H*W <= i32::MAX`、全スカラー引数と numel の `i32` 収容。
//! `i32` 超過は `SizeLimitExceeded`（`ops` が `Unsupported` へ写してホスト参照実装へ委ねる）、
//! 形状・パラメータの不整合は `InvalidOpArgs`。

use std::cell::RefCell;
use std::rc::Rc;

use crate::error::RocmError;
use crate::im2col::fit_i32_limit;
use crate::kernels_pooling::{
    ADAPTIVE_POOL_SPEC, AVG_POOL_SPEC, MAX_POOL_SPEC, render_pooling_source,
};
use crate::launch::{Arg, KernelHost};
use crate::target::GfxTarget;
use crate::typed_buffer::TypedDeviceBuffer;

const OP: &str = "pooling";

fn bad(detail: String) -> RocmError {
    RocmError::InvalidOpArgs { op: OP, detail }
}

fn checked_numel(shape: &[usize; 4]) -> Result<usize, RocmError> {
    // 0 軸があれば積は 0（その前の巨大軸で overflow しても拒否しない）。
    if shape.contains(&0) {
        return Ok(0);
    }
    shape
        .iter()
        .try_fold(1usize, |acc, &d| acc.checked_mul(d))
        .ok_or_else(|| bad("要素数が usize を超えます".into()))
}

fn ge_one(name: &str, v: [usize; 2]) -> Result<(), RocmError> {
    if v[0] == 0 || v[1] == 0 {
        return Err(bad(format!("{name} の各成分は 1 以上が必要です: {v:?}")));
    }
    Ok(())
}

/// `(in + 2p - d*(k-1) - 1) / s + 1`（`tensor-core::pool_out_len` と同じ式。overflow・負の分子は拒否）。
fn pool_out_len(
    in_len: usize,
    kernel: usize,
    stride: usize,
    padding: usize,
    dilation: usize,
) -> Result<usize, RocmError> {
    let overflow = || bad("出力長の計算が overflow しました".into());
    let in_plus_2p = padding
        .checked_mul(2)
        .and_then(|two_p| in_len.checked_add(two_p))
        .ok_or_else(overflow)?;
    let dk = dilation.checked_mul(kernel - 1).ok_or_else(overflow)?;
    let numerator = in_plus_2p
        .checked_sub(dk)
        .and_then(|v| v.checked_sub(1))
        .ok_or_else(|| {
            bad(format!(
                "窓が入力より大きく出力長が負になります (in={in_len}, kernel={kernel}, \
                 padding={padding}, dilation={dilation})"
            ))
        })?;
    Ok(numerator / stride + 1)
}

/// 検証結果（起動引数）。`out_shape` は `[N, C, H_out, W_out]`。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct PoolPlan {
    pub(crate) out_shape: [usize; 4],
    pub(crate) numel_in: usize,
    pub(crate) numel_out: usize,
    pub(crate) n_batch: i32,
    pub(crate) c: i32,
    pub(crate) h_in: i32,
    pub(crate) w_in: i32,
    pub(crate) h_out: i32,
    pub(crate) w_out: i32,
}

fn plan_from(in_shape: [usize; 4], out_shape: [usize; 4]) -> Result<PoolPlan, RocmError> {
    let numel_in = checked_numel(&in_shape)?;
    let numel_out = checked_numel(&out_shape)?;
    if numel_out == 0 {
        // 起動しない。軸サイズの `i32` 収容は不要なので、巨大な N 等でも拒否しない。
        return Ok(PoolPlan {
            out_shape,
            numel_in,
            numel_out,
            n_batch: 0,
            c: 0,
            h_in: 0,
            w_in: 0,
            h_out: 0,
            w_out: 0,
        });
    }
    let f = |v: usize| fit_i32_limit(OP, v);
    let plan = PoolPlan {
        out_shape,
        numel_in,
        numel_out,
        n_batch: f(in_shape[0])?,
        c: f(in_shape[1])?,
        h_in: f(in_shape[2])?,
        w_in: f(in_shape[3])?,
        h_out: f(out_shape[2])?,
        w_out: f(out_shape[3])?,
    };
    fit_i32_limit(OP, numel_out)?;
    Ok(plan)
}

/// `dilation` を持つ窓（max）と持たない窓（avg。dilation=1）の共通検証。
fn validate_window(
    in_shape: [usize; 4],
    kernel: [usize; 2],
    stride: [usize; 2],
    padding: [usize; 2],
    dilation: [usize; 2],
) -> Result<[usize; 4], RocmError> {
    let [n, c, h_in, w_in] = in_shape;
    ge_one("kernel", kernel)?;
    ge_one("stride", stride)?;
    ge_one("dilation", dilation)?;
    for i in 0..2 {
        let limit = kernel[i] / 2;
        if padding[i] > limit {
            return Err(bad(format!(
                "padding[{i}]={} が floor(kernel[{i}]/2)={limit} を超えます",
                padding[i]
            )));
        }
    }
    if h_in == 0 || w_in == 0 {
        return Err(bad(format!(
            "空間軸は 1 以上が必要です: h_in={h_in}, w_in={w_in}"
        )));
    }
    // kernel=2 で dilation が入力より大きいと窓が入力を飛び越え、有効タップが 0 の窓ができる。
    for (i, dim) in [h_in, w_in].into_iter().enumerate() {
        if kernel[i] == 2 && dilation[i] > dim {
            return Err(bad(format!(
                "dilation[{i}]={} が入力サイズ {dim} を超え、窓が空になります",
                dilation[i]
            )));
        }
    }
    let h_out = pool_out_len(h_in, kernel[0], stride[0], padding[0], dilation[0])?;
    let w_out = pool_out_len(w_in, kernel[1], stride[1], padding[1], dilation[1])?;
    Ok([n, c, h_out, w_out])
}

/// max の検証。`H*W` が `i32` に収まること（索引 `h * w_in + w` が `int` のため）。
pub(crate) fn validate_max_pool(
    in_shape: [usize; 4],
    kernel: [usize; 2],
    stride: [usize; 2],
    padding: [usize; 2],
    dilation: [usize; 2],
) -> Result<PoolPlan, RocmError> {
    let out_shape = validate_window(in_shape, kernel, stride, padding, dilation)?;
    let hw = in_shape[2]
        .checked_mul(in_shape[3])
        .ok_or(RocmError::SizeLimitExceeded {
            op: OP,
            n: usize::MAX,
            limit: i32::MAX as usize,
        })?;
    fit_i32_limit(OP, hw)?;
    let plan = plan_from(in_shape, out_shape)?;
    fit_i32_limit_all(&[kernel, stride, padding, dilation])?;
    Ok(plan)
}

/// 実行時にカーネルへ渡すスカラー引数が全て `i32` に収まることを HIP 取得前に検査する。
///
/// `run_*` 内の変換は HIP ロード後のため、ここで弾かないと ROCm 非搭載時に
/// `Unsupported` ではなく `DeviceUnavailable` となり、ホストフォールバックの契約
/// （`ops/conv_pool.rs`）を満たさない。
fn fit_i32_limit_all(groups: &[[usize; 2]]) -> Result<(), RocmError> {
    for g in groups {
        for &v in g {
            fit_i32_limit(OP, v)?;
        }
    }
    Ok(())
}

/// avg の検証（dilation は 1 固定）。
pub(crate) fn validate_avg_pool(
    in_shape: [usize; 4],
    kernel: [usize; 2],
    stride: [usize; 2],
    padding: [usize; 2],
) -> Result<PoolPlan, RocmError> {
    let out_shape = validate_window(in_shape, kernel, stride, padding, [1, 1])?;
    let plan = plan_from(in_shape, out_shape)?;
    fit_i32_limit_all(&[kernel, stride, padding])?;
    Ok(plan)
}

/// adaptive の検証。
pub(crate) fn validate_adaptive_pool(
    in_shape: [usize; 4],
    output_size: [usize; 2],
) -> Result<PoolPlan, RocmError> {
    let [n, c, h_in, w_in] = in_shape;
    if h_in == 0 || w_in == 0 {
        return Err(bad(format!(
            "空間軸は 1 以上が必要です: h_in={h_in}, w_in={w_in}"
        )));
    }
    ge_one("output_size", output_size)?;
    plan_from(in_shape, [n, c, output_size[0], output_size[1]])
}

fn check_len(op: &str, input_len: usize, plan: &PoolPlan) -> Result<(), RocmError> {
    if input_len != plan.numel_in {
        return Err(bad(format!(
            "{op}: input の長さ不一致: 期待 {}, 実際 {input_len}",
            plan.numel_in
        )));
    }
    Ok(())
}

/// コンパイル済みの pooling カーネル 3 種を保持し、実行する。
pub struct RocmPooling {
    host: KernelHost,
}

impl RocmPooling {
    /// HIP/HIPRTC をロードしカーネルを `target` 向けにコンパイルする。
    ///
    /// ROCm 非搭載では `LibraryUnavailable`、デバイスがなければ `NoDevice`（panic しない）。
    pub fn new(target: GfxTarget) -> Result<Self, RocmError> {
        let src = render_pooling_source()?;
        Ok(Self {
            host: KernelHost::new(&src, target)?,
        })
    }

    /// MaxPool2d。値と (n, c) 平面内の flat 索引 `h * W + w` を返す。
    pub fn run_max_pool2d_f32(
        &self,
        input: &[f32],
        in_shape: [usize; 4],
        kernel: [usize; 2],
        stride: [usize; 2],
        padding: [usize; 2],
        dilation: [usize; 2],
    ) -> Result<(Vec<f32>, Vec<i32>), RocmError> {
        let plan = validate_max_pool(in_shape, kernel, stride, padding, dilation)?;
        if plan.numel_out == 0 {
            return Ok((Vec::new(), Vec::new()));
        }
        check_len("max_pool2d", input.len(), &plan)?;
        let f = |v: usize| fit_i32_limit(OP, v);
        let _guard = self.host.select()?;
        let rt = || self.host.runtime();
        let din = TypedDeviceBuffer::<f32>::from_slice(rt(), input)?;
        let dout = TypedDeviceBuffer::<f32>::new_zeroed(rt(), plan.numel_out)?;
        let didx = TypedDeviceBuffer::<i32>::new_zeroed(rt(), plan.numel_out)?;
        let mut args = vec![
            Arg::Ptr(din.as_device_ptr()),
            Arg::Ptr(dout.as_device_ptr()),
            Arg::Ptr(didx.as_device_ptr()),
        ];
        args.extend(
            [
                plan.n_batch,
                plan.c,
                plan.h_in,
                plan.w_in,
                plan.h_out,
                plan.w_out,
                f(kernel[0])?,
                f(kernel[1])?,
                f(stride[0])?,
                f(stride[1])?,
                f(padding[0])?,
                f(padding[1])?,
                f(dilation[0])?,
                f(dilation[1])?,
                f(plan.numel_out)?,
            ]
            .map(Arg::I32),
        );
        self.host.run(OP, &MAX_POOL_SPEC, plan.numel_out, &args)?;
        Ok((dout.to_vec()?, didx.to_vec()?))
    }

    /// AvgPool2d（dilation は 1 固定）。`count_include_pad` が true なら divisor は `kh*kw`。
    pub fn run_avg_pool2d_f32(
        &self,
        input: &[f32],
        in_shape: [usize; 4],
        kernel: [usize; 2],
        stride: [usize; 2],
        padding: [usize; 2],
        count_include_pad: bool,
    ) -> Result<Vec<f32>, RocmError> {
        let plan = validate_avg_pool(in_shape, kernel, stride, padding)?;
        if plan.numel_out == 0 {
            return Ok(Vec::new());
        }
        check_len("avg_pool2d", input.len(), &plan)?;
        let f = |v: usize| fit_i32_limit(OP, v);
        let _guard = self.host.select()?;
        let rt = || self.host.runtime();
        let din = TypedDeviceBuffer::<f32>::from_slice(rt(), input)?;
        let dout = TypedDeviceBuffer::<f32>::new_zeroed(rt(), plan.numel_out)?;
        let mut args = vec![
            Arg::Ptr(din.as_device_ptr()),
            Arg::Ptr(dout.as_device_ptr()),
        ];
        args.extend(
            [
                plan.n_batch,
                plan.c,
                plan.h_in,
                plan.w_in,
                plan.h_out,
                plan.w_out,
                f(kernel[0])?,
                f(kernel[1])?,
                f(stride[0])?,
                f(stride[1])?,
                f(padding[0])?,
                f(padding[1])?,
                i32::from(count_include_pad),
                f(plan.numel_out)?,
            ]
            .map(Arg::I32),
        );
        self.host.run(OP, &AVG_POOL_SPEC, plan.numel_out, &args)?;
        dout.to_vec()
    }

    /// AdaptiveAvgPool2d。
    pub fn run_adaptive_avg_pool2d_f32(
        &self,
        input: &[f32],
        in_shape: [usize; 4],
        output_size: [usize; 2],
    ) -> Result<Vec<f32>, RocmError> {
        let plan = validate_adaptive_pool(in_shape, output_size)?;
        if plan.numel_out == 0 {
            return Ok(Vec::new());
        }
        check_len("adaptive_avg_pool2d", input.len(), &plan)?;
        let _guard = self.host.select()?;
        let rt = || self.host.runtime();
        let din = TypedDeviceBuffer::<f32>::from_slice(rt(), input)?;
        let dout = TypedDeviceBuffer::<f32>::new_zeroed(rt(), plan.numel_out)?;
        let mut args = vec![
            Arg::Ptr(din.as_device_ptr()),
            Arg::Ptr(dout.as_device_ptr()),
        ];
        args.extend(
            [
                plan.n_batch,
                plan.c,
                plan.h_in,
                plan.w_in,
                plan.h_out,
                plan.w_out,
                fit_i32_limit(OP, plan.numel_out)?,
            ]
            .map(Arg::I32),
        );
        self.host
            .run(OP, &ADAPTIVE_POOL_SPEC, plan.numel_out, &args)?;
        dout.to_vec()
    }
}

type PoolCache = RefCell<Vec<(GfxTarget, Rc<RocmPooling>)>>;

thread_local! {
    /// target ごとにコンパイル済みの実行型を遅延保持する（`softmax.rs` と同型。失敗はキャッシュしない）。
    static POOL_CACHE: PoolCache = const { RefCell::new(Vec::new()) };
}

/// `target` 用の [`RocmPooling`] を取得する（無ければ構築して入れる）。`ops/conv_pool.rs` から呼ばれる。
pub(crate) fn cached_pooling(target: GfxTarget) -> Result<Rc<RocmPooling>, RocmError> {
    let hit = POOL_CACHE
        .try_with(|c| {
            c.borrow()
                .iter()
                .find(|(t, _)| *t == target)
                .map(|(_, v)| Rc::clone(v))
        })
        .map_err(|_| RocmError::InvalidKernelConfig {
            detail: "pooling キャッシュ（thread_local）が破棄済みです".into(),
        })?;
    if let Some(v) = hit {
        return Ok(v);
    }
    // 借用を解いてから構築する（再入しても二重借用にならない）。
    let v = Rc::new(RocmPooling::new(target)?);
    let _ = POOL_CACHE.try_with(|c| c.borrow_mut().push((target, Rc::clone(&v))));
    Ok(v)
}

#[cfg(test)]
mod tests {
    use super::*;

    const IN: [usize; 4] = [2, 3, 8, 8];

    #[test]
    fn max_pool_out_shape_and_dilation() {
        let p = validate_max_pool(IN, [3, 3], [2, 2], [1, 1], [1, 1]).unwrap();
        assert_eq!(p.out_shape, [2, 3, 4, 4]);
        // dilation=2, kernel=3 → 有効窓 5。(8 - 4 - 1)/1 + 1 = 4。
        let p = validate_max_pool(IN, [3, 3], [1, 1], [0, 0], [2, 2]).unwrap();
        assert_eq!(p.out_shape, [2, 3, 4, 4]);
        assert_eq!(p.numel_in, 2 * 3 * 8 * 8);
    }

    #[test]
    fn rejects_invalid_parameters() {
        type Win = [usize; 2];
        let bad_cases: [(Win, Win, Win, Win); 5] = [
            ([0, 3], [1, 1], [0, 0], [1, 1]),
            ([3, 3], [0, 1], [0, 0], [1, 1]),
            ([3, 3], [1, 1], [0, 0], [1, 0]),
            ([3, 3], [1, 1], [2, 0], [1, 1]), // padding > kernel/2
            ([9, 3], [1, 1], [0, 0], [1, 1]), // 窓が入力より大きい
        ];
        for (k, s, p, d) in bad_cases {
            assert!(
                matches!(
                    validate_max_pool(IN, k, s, p, d),
                    Err(RocmError::InvalidOpArgs { .. })
                ),
                "{k:?} {s:?} {p:?} {d:?}"
            );
        }
        // 空間軸 0。
        assert!(validate_max_pool([1, 1, 0, 4], [1, 1], [1, 1], [0, 0], [1, 1]).is_err());
        assert!(validate_avg_pool([1, 1, 4, 0], [1, 1], [1, 1], [0, 0]).is_err());
        assert!(validate_adaptive_pool([1, 1, 0, 4], [2, 2]).is_err());
    }

    #[test]
    fn rejects_empty_window_from_large_dilation() {
        // kernel=2・padding=1: dilation > in だと両タップとも padding 位置に落ち、有効タップが 0 の窓ができる。
        assert!(validate_max_pool([1, 1, 3, 3], [2, 2], [1, 1], [1, 0], [4, 1]).is_err());
        assert!(validate_max_pool([1, 1, 3, 3], [2, 2], [1, 1], [1, 0], [3, 1]).is_ok());
    }

    #[test]
    fn avg_ignores_dilation_and_adaptive_shape() {
        let p = validate_avg_pool(IN, [2, 2], [2, 2], [1, 1]).unwrap();
        assert_eq!(p.out_shape, [2, 3, 5, 5]);
        let p = validate_adaptive_pool(IN, [3, 5]).unwrap();
        assert_eq!(p.out_shape, [2, 3, 3, 5]);
        assert!(validate_adaptive_pool(IN, [0, 5]).is_err());
    }

    #[test]
    fn empty_batch_gives_empty_output_without_overflow() {
        let p = validate_avg_pool([0, 3, 8, 8], [2, 2], [2, 2], [0, 0]).unwrap();
        assert_eq!((p.numel_in, p.numel_out), (0, 0));
        // 0 軸の前に巨大軸があっても積の overflow で拒否しない。
        let big = usize::MAX / 2 + 1;
        let p = validate_adaptive_pool([big, 4, 3, 3], [2, 2]);
        assert!(p.is_err()); // numel_out = big*4*2*2 は overflow（0 軸なし）。
        let p = validate_adaptive_pool([big, 0, 3, 3], [2, 2]).unwrap();
        assert_eq!((p.numel_in, p.numel_out), (0, 0));
    }

    #[test]
    fn i32_limits_map_to_size_limit() {
        let big = i32::MAX as usize + 1;
        // max: H*W が i32 を超える。
        assert!(matches!(
            validate_max_pool([1, 1, big, 1], [1, 1], [1, 1], [0, 0], [1, 1]),
            Err(RocmError::SizeLimitExceeded { .. })
        ));
        assert!(matches!(
            validate_max_pool([1, 1, 65_536, 65_536], [1, 1], [1, 1], [0, 0], [1, 1]),
            Err(RocmError::SizeLimitExceeded { .. })
        ));
        // avg: 出力 numel が i32 を超える。
        assert!(matches!(
            validate_avg_pool([big, 1, 1, 1], [1, 1], [1, 1], [0, 0]),
            Err(RocmError::SizeLimitExceeded { .. })
        ));
        // max の索引域境界: H*W == i32::MAX は許容（numel_out は別途収まる形状で確認）。
        assert!(
            validate_max_pool([1, 1, 1, i32::MAX as usize], [1, 1], [1, 1], [0, 0], [1, 1]).is_ok()
        );
    }

    /// 起動スカラー（kernel/stride/dilation。padding は kernel/2 以下に縛られ kernel 側で捕捉）の i32 超過は HIP 取得前の検証で
    /// `SizeLimitExceeded`（= ops 層で Unsupported → ホストフォールバック）になる。
    #[test]
    fn launch_scalars_beyond_i32_rejected_before_hip() {
        let big = i32::MAX as usize + 1;
        let one = [1usize, 1];
        let in_shape = [1, 1, 1, 1];
        let is_limit =
            |r: Result<PoolPlan, RocmError>| matches!(r, Err(RocmError::SizeLimitExceeded { .. }));
        assert!(is_limit(validate_avg_pool(in_shape, one, [big, 1], [0, 0])));
        assert!(is_limit(validate_max_pool(
            in_shape,
            one,
            [big, 1],
            [0, 0],
            one
        )));
        assert!(is_limit(validate_max_pool(
            in_shape,
            one,
            one,
            [0, 0],
            [big, 1]
        )));
    }

    #[test]
    fn length_check() {
        let plan = validate_adaptive_pool([1, 1, 2, 2], [1, 1]).unwrap();
        assert!(check_len("t", 4, &plan).is_ok());
        assert!(check_len("t", 3, &plan).is_err());
    }
}
