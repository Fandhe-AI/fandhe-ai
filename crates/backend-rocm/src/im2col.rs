//! Conv2d の im2col／col2im の ROCm 実行入口（`backend-cuda::im2col` の移植。イシュー #3438・R7）。
//!
//! 役割: [`RocmIm2col`] が HIP/HIPRTC をロードして `kernels_im2col` のソースをコンパイルし、行優先
//! contiguous のホストスライスから結果の `Vec` を返す。`BackendOps::im2col`／`col2im` の shape 再検査
//! （`im2col_out_shape`）・空出力の早期リターン・`Unsupported` の事前確定・`Tensor` 化は `RocmBackendOps`
//! 側（`ops/conv_pool.rs`）の責務で、本層は起動引数の導出と実行だけを持つ。
//!
//! ホスト側で fail-closed に検証する（A04）: rank 4・`groups` の整合（`Cin % groups`・`K_g`・`P`）・
//! スライス長・全スカラー引数と numel の `i32` 収容。`i32` 超過は `SizeLimitExceeded`（`ops` が
//! `Unsupported` へ写してホスト参照実装へ委ねる）、形状の不整合は `InvalidOpArgs`。HIP に触れない純関数
//! [`plan_im2col`]／[`plan_col2im`] に集約しているので、ROCm 非搭載でも検査できる。

use std::cell::RefCell;
use std::rc::Rc;

use fandhe_ai_tensor_core::{Conv2dParams, conv_out_len};

use crate::error::RocmError;
use crate::kernels_im2col::{COL2IM_SPEC, IM2COL_SPEC, render_im2col_source};
use crate::launch::{Arg, KernelHost, KernelSpec};
use crate::target::GfxTarget;
use crate::typed_buffer::TypedDeviceBuffer;

const OP: &str = "im2col";

fn bad(detail: String) -> RocmError {
    RocmError::InvalidOpArgs { op: OP, detail }
}

/// `i32`（カーネル引数型）への収容。超過は `SizeLimitExceeded`（`Unsupported` へ写す対象）。
pub(crate) fn fit_i32_limit(op: &'static str, value: usize) -> Result<i32, RocmError> {
    i32::try_from(value).map_err(|_| RocmError::SizeLimitExceeded {
        op,
        n: value,
        limit: i32::MAX as usize,
    })
}

fn checked_numel(shape: &[usize]) -> Result<usize, RocmError> {
    // 0 軸があれば積は 0（その前の巨大軸で overflow しても拒否しない）。
    if shape.contains(&0) {
        return Ok(0);
    }
    shape
        .iter()
        .try_fold(1usize, |acc, &d| acc.checked_mul(d))
        .ok_or_else(|| bad("要素数が usize を超えます".into()))
}

/// 起動引数（`kernels_im2col.rs` の 19 個のスカラー引数のうち numel を除く 18 個）。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Im2colDims {
    pub(crate) n_batch: i32,
    pub(crate) cin: i32,
    pub(crate) groups: i32,
    pub(crate) cin_g: i32,
    pub(crate) h_in: i32,
    pub(crate) w_in: i32,
    pub(crate) h_out: i32,
    pub(crate) w_out: i32,
    pub(crate) kh: i32,
    pub(crate) kw: i32,
    pub(crate) sh: i32,
    pub(crate) sw: i32,
    pub(crate) ph: i32,
    pub(crate) pw: i32,
    pub(crate) dh: i32,
    pub(crate) dw: i32,
    pub(crate) k_g: i32,
    pub(crate) p: i32,
}

/// テスト専用: [`Im2colDims`] を `long long` 相当（`i64`）へ広げたもの（ホストモデルが使う）。
#[cfg(test)]
#[derive(Debug, Clone, Copy)]
pub(crate) struct Dims64 {
    pub(crate) cin: i64,
    pub(crate) groups: i64,
    pub(crate) cin_g: i64,
    pub(crate) h_in: i64,
    pub(crate) w_in: i64,
    pub(crate) h_out: i64,
    pub(crate) w_out: i64,
    pub(crate) kh: i64,
    pub(crate) kw: i64,
    pub(crate) sh: i64,
    pub(crate) sw: i64,
    pub(crate) ph: i64,
    pub(crate) pw: i64,
    pub(crate) dh: i64,
    pub(crate) dw: i64,
    pub(crate) k_g: i64,
    pub(crate) p: i64,
}

#[cfg(test)]
impl Im2colDims {
    pub(crate) fn map_i64(&self) -> Dims64 {
        Dims64 {
            cin: self.cin.into(),
            groups: self.groups.into(),
            cin_g: self.cin_g.into(),
            h_in: self.h_in.into(),
            w_in: self.w_in.into(),
            h_out: self.h_out.into(),
            w_out: self.w_out.into(),
            kh: self.kh.into(),
            kw: self.kw.into(),
            sh: self.sh.into(),
            sw: self.sw.into(),
            ph: self.ph.into(),
            pw: self.pw.into(),
            dh: self.dh.into(),
            dw: self.dw.into(),
            k_g: self.k_g.into(),
            p: self.p.into(),
        }
    }
}

/// 検証結果。`numel_in`／`numel_col` は入力 `[N,Cin,H,W]`／col `[N,G,K_g,P]` の要素数。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Im2colPlan {
    pub(crate) dims: Im2colDims,
    pub(crate) numel_in: usize,
    pub(crate) numel_col: usize,
}

/// `in_shape`（`[N,Cin,H,W]`）・`col_shape`（`[N,G,K_g,P]`）・`params` の整合を検査して起動引数を導出する。
fn derive(
    in_shape: &[usize],
    col_shape: &[usize],
    params: &Conv2dParams,
) -> Result<Im2colPlan, RocmError> {
    if in_shape.len() != 4 || col_shape.len() != 4 {
        return Err(bad(format!(
            "rank 4 が必要です: input {}, col {}",
            in_shape.len(),
            col_shape.len()
        )));
    }
    let numel_in = checked_numel(in_shape)?;
    let numel_col = checked_numel(col_shape)?;
    let (n, cin, h_in, w_in) = (in_shape[0], in_shape[1], in_shape[2], in_shape[3]);
    let (groups, k_g, p) = (col_shape[1], col_shape[2], col_shape[3]);
    if groups == 0 || groups != params.groups() || cin % groups != 0 {
        return Err(bad(format!(
            "groups 不整合: col の G={groups}, params.groups={}, Cin={cin}",
            params.groups()
        )));
    }
    if col_shape[0] != n {
        return Err(bad(format!(
            "バッチ不一致: input {n}, col {}",
            col_shape[0]
        )));
    }
    let cin_g = cin / groups;
    let [kh, kw] = params.kernel_size();
    let [sh, sw] = params.stride();
    let [ph, pw] = params.padding();
    let [dh, dw] = params.dilation();
    let expected_k_g = cin_g
        .checked_mul(kh)
        .and_then(|v| v.checked_mul(kw))
        .ok_or_else(|| bad("K_g が usize を超えます".into()))?;
    if k_g != expected_k_g {
        return Err(bad(format!("K_g 不一致: 期待 {expected_k_g}, col {k_g}")));
    }
    let h_out = conv_out_len(h_in, kh, sh, ph, dh)
        .map_err(|e| bad(format!("conv_out_len(h) 失敗: {e:?}")))?;
    let w_out = conv_out_len(w_in, kw, sw, pw, dw)
        .map_err(|e| bad(format!("conv_out_len(w) 失敗: {e:?}")))?;
    if h_out.checked_mul(w_out) != Some(p) {
        return Err(bad(format!(
            "P 軸不一致: h_out*w_out={} p={p}",
            h_out.saturating_mul(w_out)
        )));
    }
    let f = |v: usize| fit_i32_limit(OP, v);
    Ok(Im2colPlan {
        dims: Im2colDims {
            n_batch: f(n)?,
            cin: f(cin)?,
            groups: f(groups)?,
            cin_g: f(cin_g)?,
            h_in: f(h_in)?,
            w_in: f(w_in)?,
            h_out: f(h_out)?,
            w_out: f(w_out)?,
            kh: f(kh)?,
            kw: f(kw)?,
            sh: f(sh)?,
            sw: f(sw)?,
            ph: f(ph)?,
            pw: f(pw)?,
            dh: f(dh)?,
            dw: f(dw)?,
            k_g: f(k_g)?,
            p: f(p)?,
        },
        numel_in,
        numel_col,
    })
}

/// im2col の検証（起動する numel は col 側。`i32` に収まること）。
pub(crate) fn plan_im2col(
    in_shape: &[usize],
    col_shape: &[usize],
    params: &Conv2dParams,
) -> Result<Im2colPlan, RocmError> {
    let plan = derive(in_shape, col_shape, params)?;
    fit_i32_limit(OP, plan.numel_col)?;
    Ok(plan)
}

/// col2im の検証（起動する numel は入力側。`i32` に収まること）。
pub(crate) fn plan_col2im(
    in_shape: &[usize],
    col_shape: &[usize],
    params: &Conv2dParams,
) -> Result<Im2colPlan, RocmError> {
    let plan = derive(in_shape, col_shape, params)?;
    fit_i32_limit(OP, plan.numel_in)?;
    Ok(plan)
}

/// コンパイル済みの im2col／col2im カーネルを保持し、実行する。
pub struct RocmIm2col {
    host: KernelHost,
}

impl RocmIm2col {
    /// HIP/HIPRTC をロードしカーネルを `target` 向けにコンパイルする。
    ///
    /// ROCm 非搭載では `LibraryUnavailable`、デバイスがなければ `NoDevice`（panic しない）。
    pub fn new(target: GfxTarget) -> Result<Self, RocmError> {
        let src = render_im2col_source()?;
        Ok(Self {
            host: KernelHost::new(&src, target)?,
        })
    }

    /// im2col。`input` は `in_shape = [N,Cin,H,W]` の行優先 contiguous、`col_shape = [N,G,K_g,P]`。
    pub fn run_im2col_f32(
        &self,
        input: &[f32],
        in_shape: &[usize],
        col_shape: &[usize],
        params: &Conv2dParams,
    ) -> Result<Vec<f32>, RocmError> {
        let plan = plan_im2col(in_shape, col_shape, params)?;
        if plan.numel_col == 0 {
            return Ok(Vec::new());
        }
        if input.len() != plan.numel_in {
            return Err(bad(format!(
                "input の長さ不一致: 期待 {}, 実際 {}",
                plan.numel_in,
                input.len()
            )));
        }
        self.launch(&IM2COL_SPEC, input, &plan, plan.numel_col)
    }

    /// col2im（im2col の随伴）。`d_col` は `col_shape` の contiguous、戻り値は `in_shape` の要素数。
    pub fn run_col2im_f32(
        &self,
        d_col: &[f32],
        col_shape: &[usize],
        in_shape: &[usize],
        params: &Conv2dParams,
    ) -> Result<Vec<f32>, RocmError> {
        let plan = plan_col2im(in_shape, col_shape, params)?;
        if plan.numel_in == 0 {
            return Ok(Vec::new());
        }
        if d_col.len() != plan.numel_col {
            return Err(bad(format!(
                "d_col の長さ不一致: 期待 {}, 実際 {}",
                plan.numel_col,
                d_col.len()
            )));
        }
        self.launch(&COL2IM_SPEC, d_col, &plan, plan.numel_in)
    }

    /// 共通の確保・起動・読み戻し。`numel_out` は出力要素数（`plan_*` が `i32` 収容を検証済み）。
    fn launch(
        &self,
        spec: &KernelSpec,
        src: &[f32],
        plan: &Im2colPlan,
        numel_out: usize,
    ) -> Result<Vec<f32>, RocmError> {
        let numel_i = fit_i32_limit(OP, numel_out)?;
        let _guard = self.host.select()?;
        let rt = || self.host.runtime();
        let dsrc = TypedDeviceBuffer::<f32>::from_slice(rt(), src)?;
        let dout = TypedDeviceBuffer::<f32>::new_zeroed(rt(), numel_out)?;
        let d = &plan.dims;
        let mut args = vec![
            Arg::Ptr(dsrc.as_device_ptr()),
            Arg::Ptr(dout.as_device_ptr()),
        ];
        args.extend(
            [
                d.n_batch, d.cin, d.groups, d.cin_g, d.h_in, d.w_in, d.h_out, d.w_out, d.kh, d.kw,
                d.sh, d.sw, d.ph, d.pw, d.dh, d.dw, d.k_g, d.p, numel_i,
            ]
            .map(Arg::I32),
        );
        self.host.run(OP, spec, numel_out, &args)?;
        dout.to_vec()
    }
}

type Im2colCache = RefCell<Vec<(GfxTarget, Rc<RocmIm2col>)>>;

thread_local! {
    /// target ごとにコンパイル済みの実行型を遅延保持する（`softmax.rs` と同型。失敗はキャッシュしない。
    /// 生ポインタを持つ実行型は `!Send` のため thread_local に置く）。
    static IM2COL_CACHE: Im2colCache = const { RefCell::new(Vec::new()) };
}

/// `target` 用の [`RocmIm2col`] を取得する（無ければ構築して入れる）。`ops/conv_pool.rs` から呼ばれる。
pub(crate) fn cached_im2col(target: GfxTarget) -> Result<Rc<RocmIm2col>, RocmError> {
    let hit = IM2COL_CACHE
        .try_with(|c| {
            c.borrow()
                .iter()
                .find(|(t, _)| *t == target)
                .map(|(_, v)| Rc::clone(v))
        })
        .map_err(|_| RocmError::InvalidKernelConfig {
            detail: "im2col キャッシュ（thread_local）が破棄済みです".into(),
        })?;
    if let Some(v) = hit {
        return Ok(v);
    }
    // 借用を解いてから構築する（再入しても二重借用にならない）。
    let v = Rc::new(RocmIm2col::new(target)?);
    let _ = IM2COL_CACHE.try_with(|c| c.borrow_mut().push((target, Rc::clone(&v))));
    Ok(v)
}

#[cfg(test)]
mod tests {
    use super::*;
    use fandhe_ai_tensor_core::im2col_out_shape;

    fn params(
        k: [usize; 2],
        s: [usize; 2],
        p: [usize; 2],
        d: [usize; 2],
        g: usize,
    ) -> Conv2dParams {
        Conv2dParams::new(k, s, p, d, g).unwrap()
    }

    #[test]
    fn plan_derives_launch_dims() {
        let pr = params([3, 3], [2, 1], [1, 0], [1, 2], 2);
        let in_shape = [2, 4, 9, 8];
        let col = im2col_out_shape(&in_shape, &pr).unwrap();
        let plan = plan_im2col(&in_shape, &col, &pr).unwrap();
        assert_eq!(plan.numel_in, 2 * 4 * 9 * 8);
        assert_eq!(plan.numel_col, col.iter().product::<usize>());
        let d = plan.dims;
        assert_eq!((d.groups, d.cin_g, d.k_g), (2, 2, 18));
        assert_eq!(d.h_out * d.w_out, d.p);
        assert_eq!(plan_col2im(&in_shape, &col, &pr).unwrap(), plan);
    }

    #[test]
    fn plan_rejects_inconsistent_shapes() {
        let pr = params([3, 3], [1, 1], [1, 1], [1, 1], 1);
        let in_shape = [1, 2, 5, 5];
        let col = im2col_out_shape(&in_shape, &pr).unwrap();
        assert!(plan_im2col(&in_shape, &col, &pr).is_ok());
        // rank 不一致。
        assert!(matches!(
            plan_im2col(&[1, 2, 5], &col, &pr),
            Err(RocmError::InvalidOpArgs { .. })
        ));
        // groups が割り切れない／params と col の G が不一致。
        let pr3 = params([3, 3], [1, 1], [1, 1], [1, 1], 3);
        assert!(matches!(
            plan_im2col(&in_shape, &col, &pr3),
            Err(RocmError::InvalidOpArgs { .. })
        ));
        // K_g・P・バッチの不一致。
        for idx in [0usize, 2, 3] {
            let mut c = col.clone();
            c[idx] += 1;
            assert!(
                matches!(
                    plan_im2col(&in_shape, &c, &pr),
                    Err(RocmError::InvalidOpArgs { .. })
                ),
                "axis {idx}"
            );
        }
    }

    #[test]
    fn plan_reports_i32_overflow_as_size_limit() {
        let pr = params([1, 1], [1, 1], [0, 0], [1, 1], 1);
        let big = i32::MAX as usize + 1;
        // 空間軸が i32 を超える（numel は 1 * 1 * big * 1 = big で col 側も同じ）。
        let in_shape = [1, 1, big, 1];
        let col = im2col_out_shape(&in_shape, &pr).unwrap();
        assert!(matches!(
            plan_im2col(&in_shape, &col, &pr),
            Err(RocmError::SizeLimitExceeded { .. })
        ));
        assert!(matches!(
            plan_col2im(&in_shape, &col, &pr),
            Err(RocmError::SizeLimitExceeded { .. })
        ));
    }

    #[test]
    fn empty_output_does_not_overflow_before_short_circuit() {
        // N=0 の col numel は 0。plan は積の overflow なしで通る。
        let pr = params([3, 3], [1, 1], [1, 1], [1, 1], 1);
        let in_shape = [0, 2, 5, 5];
        let col = im2col_out_shape(&in_shape, &pr).unwrap();
        let plan = plan_im2col(&in_shape, &col, &pr).unwrap();
        assert_eq!((plan.numel_in, plan.numel_col), (0, 0));
    }
}
