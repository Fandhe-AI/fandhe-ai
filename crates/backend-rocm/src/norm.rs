//! 正規化 6 演算（rmsnorm・layer_norm・両 backward・batch_norm の train/infer）の ROCm 実行入口
//! （`backend-cuda` の `rmsnorm.rs`・`layer_norm.rs`・`norm_backward.rs`・`batch_norm.rs` の移植。
//! イシュー #3437・分解案 R6・親 #3150）。
//!
//! 役割: [`RocmNorm`] が HIP/HIPRTC をロードして `kernels_norm` のソースをコンパイルし、ホストの
//! スライス（`[rows, hidden]` または `[n, c, spatial]` の行優先）を受けて `Vec` を返す。
//! `BackendOps::rmsnorm` 等への結線（`row_norm_layout`／`batch_norm_layout` による形状導出・
//! contiguous 化・`Tensor` 化）は `RocmBackendOps`（`ops.rs`）の責務で、本型はスライス入出力のみ。
//!
//! ホスト側で eps・長さ・checked 乗算・`i32` 収容を起動前に検証する（カーネル側の手動境界チェック
//! REQ-8 と二重化。CPU の `validate_*_launch`・CUDA の `validate_*_launch` に揃える）。検証関数は
//! `pub(crate)` で、テスト専用のホスト側モデル経路（`norm_model.rs`）も同じ検証を通る。
//! 起動は `launch::KernelHost::run_wave_blocks`（行・チャネル単位。1 block = 1 wave）または `run`
//! （dw・dwdb・infer の 1 次元）で、`unsafe` はこのファイルに持たない（K1 に集約済み）。
//!
//! 省略された weight／bias／統計にも必ず `hidden`（または `c`）要素のダミーバッファを渡す
//! （CUDA と同じ predicated-load 対策）。`rstd`／`mean` の double スクラッチは起動前に 0 で
//! 初期化したバッファに置く（未初期化メモリを起動に渡さない。A04）。0 要素は HIP に触らず戻る。

use crate::error::RocmError;
use crate::kernels_norm::{
    BATCH_NORM_INFER_SPEC, BATCH_NORM_TRAIN_SPEC, LAYER_NORM_BWD_DWDB_SPEC, LAYER_NORM_BWD_DX_SPEC,
    LAYER_NORM_SPEC, RMSNORM_BWD_DW_SPEC, RMSNORM_BWD_DX_SPEC, RMSNORM_SPEC, render_norm_source,
};
use crate::launch::{Arg, KernelHost};
use crate::module::DeviceBuffer;
use crate::target::GfxTarget;
use crate::typed_buffer::TypedDeviceBuffer;
use crate::warp_geometry::WarpGeometry;

const OP_RMSNORM: &str = "rmsnorm";
const OP_LAYER_NORM: &str = "layer_norm";
const OP_RMSNORM_BWD: &str = "rmsnorm_backward";
const OP_LAYER_NORM_BWD: &str = "layer_norm_backward";
const OP_BATCH_NORM_TRAIN: &str = "batch_norm_train";
const OP_BATCH_NORM_INFER: &str = "batch_norm_infer";

/// `rmsnorm_backward` の戻り値 `(dx, dw)`。`dw` は weight があるときだけ `Some`（長さ `hidden`）。
pub type RmsNormBackwardHost = (Vec<f32>, Option<Vec<f32>>);
/// `layer_norm_backward` の戻り値 `(dx, dw, db)`。`dw` は weight、`db` は `has_bias` のときだけ `Some`。
pub type LayerNormBackwardHost = (Vec<f32>, Option<Vec<f32>>, Option<Vec<f32>>);
/// `batch_norm_train` の戻り値 `(out, batch_mean, batch_var)`（統計は長さ `c`・biased ÷M）。
pub type BatchNormTrainHost = (Vec<f32>, Vec<f32>, Vec<f32>);

/// 起動前に長さを突き合わせる任意引数の長さ群（`None` は「その引数なし」）。
#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct OptLens {
    pub(crate) w: Option<usize>,
    pub(crate) b: Option<usize>,
    pub(crate) dy: Option<usize>,
    pub(crate) mean: Option<usize>,
    pub(crate) var: Option<usize>,
}

/// 検証済みの行単位の寸法（カーネル引数は `i32`）。0 要素のときは `rows_i` が 0。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct RowDims {
    pub(crate) rows: usize,
    pub(crate) hidden: usize,
    pub(crate) numel: usize,
    pub(crate) rows_i: i32,
    pub(crate) hidden_i: i32,
}

/// 検証済みのチャネル単位の寸法（カーネル引数は `i32`。`numel` は `long long` 引数だが CUDA に合わせ
/// `i32` 内に制限する）。0 要素のとき `spatial_i`・`m_i` は 0。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct ChannelDims {
    pub(crate) c: usize,
    pub(crate) numel: usize,
    pub(crate) c_i: i32,
    pub(crate) spatial_i: i32,
    pub(crate) m_i: i32,
}

fn fit_i32(op: &'static str, value: usize, name: &str) -> Result<i32, RocmError> {
    i32::try_from(value).map_err(|_| RocmError::InvalidOpArgs {
        op,
        detail: format!("{name}={value} は i32（カーネル引数型）に収まりません"),
    })
}

fn bad(op: &'static str, detail: String) -> RocmError {
    RocmError::InvalidOpArgs { op, detail }
}

fn check_eps(op: &'static str, eps: f32) -> Result<(), RocmError> {
    if !eps.is_finite() || eps < 0.0 {
        return Err(bad(op, format!("eps は有限かつ 0 以上が必要です: {eps}")));
    }
    Ok(())
}

fn check_len(
    op: &'static str,
    name: &str,
    got: Option<usize>,
    want: usize,
) -> Result<(), RocmError> {
    match got {
        Some(g) if g != want => Err(bad(
            op,
            format!("{name} の長さ不一致: 期待 {want}, 実際 {g}"),
        )),
        _ => Ok(()),
    }
}

/// 行単位演算（rmsnorm・layer_norm とその backward）の検証。`rows*hidden == x_len` と
/// 各任意引数の長さ（`w`／`b` は `hidden`、`dy` は `rows*hidden`）・eps・`i32` 収容を確かめる。
pub(crate) fn validate_row_norm(
    op: &'static str,
    x_len: usize,
    rows: usize,
    hidden: usize,
    lens: OptLens,
    eps: f32,
) -> Result<RowDims, RocmError> {
    check_eps(op, eps)?;
    let numel = rows.checked_mul(hidden).ok_or_else(|| {
        bad(
            op,
            format!("rows*hidden が usize を超えます: {rows}*{hidden}"),
        )
    })?;
    if x_len != numel {
        return Err(bad(
            op,
            format!("x の長さ不一致: rows*hidden={numel}, x.len()={x_len}"),
        ));
    }
    check_len(op, "weight", lens.w, hidden)?;
    check_len(op, "bias", lens.b, hidden)?;
    check_len(op, "dy", lens.dy, numel)?;
    // `hidden` は 0 要素でも dw／db のゼロ埋め確保に使うため常に収容を要求する。
    let hidden_i = fit_i32(op, hidden, "hidden")?;
    let (rows_i, _) = if numel == 0 {
        (0, 0)
    } else {
        (fit_i32(op, rows, "rows")?, fit_i32(op, numel, "numel")?)
    };
    Ok(RowDims {
        rows,
        hidden,
        numel,
        rows_i,
        hidden_i,
    })
}

/// batch_norm の検証。`n*c*spatial == x_len`、`w`／`b`／`mean`／`var` は長さ `c`、eps、`i32` 収容。
pub(crate) fn validate_batch_norm(
    op: &'static str,
    x_len: usize,
    dims: (usize, usize, usize),
    lens: OptLens,
    eps: f32,
) -> Result<ChannelDims, RocmError> {
    let (n, c, spatial) = dims;
    check_eps(op, eps)?;
    let numel = n
        .checked_mul(c)
        .and_then(|v| v.checked_mul(spatial))
        .ok_or_else(|| {
            bad(
                op,
                format!("n*c*spatial が usize を超えます: {n}*{c}*{spatial}"),
            )
        })?;
    if x_len != numel {
        return Err(bad(
            op,
            format!("x の長さ不一致: n*c*spatial={numel}, x.len()={x_len}"),
        ));
    }
    check_len(op, "weight", lens.w, c)?;
    check_len(op, "bias", lens.b, c)?;
    check_len(op, "mean", lens.mean, c)?;
    check_len(op, "var", lens.var, c)?;
    // `c` は 0 要素でも統計のゼロ埋め確保に使うため常に収容を要求する。
    let c_i = fit_i32(op, c, "c")?;
    if numel == 0 {
        return Ok(ChannelDims {
            c,
            numel,
            c_i,
            spatial_i: 0,
            m_i: 0,
        });
    }
    // numel > 0 なので n・spatial は非 0 で、m = n*spatial <= numel。
    fit_i32(op, numel, "numel")?;
    let m = n * spatial;
    Ok(ChannelDims {
        c,
        numel,
        c_i,
        spatial_i: fit_i32(op, spatial, "spatial")?,
        m_i: fit_i32(op, m, "m")?,
    })
}

/// `n` 要素のゼロ埋め `Vec<f32>`（確保失敗は panic せず型付きエラー）。
pub(crate) fn zeros_f32(op: &'static str, n: usize) -> Result<Vec<f32>, RocmError> {
    let mut v: Vec<f32> = Vec::new();
    v.try_reserve_exact(n)
        .map_err(|_| bad(op, format!("{n} 要素のホストバッファを確保できません")))?;
    v.resize(n, 0.0);
    Ok(v)
}

/// rmsnorm backward の 0 要素戻り値（`dx` 空・`dw` は weight があればゼロ埋めの `[hidden]`）。
pub(crate) fn rmsnorm_backward_empty(
    has_weight: bool,
    hidden: usize,
) -> Result<RmsNormBackwardHost, RocmError> {
    let dw = if has_weight {
        Some(zeros_f32(OP_RMSNORM_BWD, hidden)?)
    } else {
        None
    };
    Ok((Vec::new(), dw))
}

/// layer_norm backward の 0 要素戻り値（`dx` 空・`dw`／`db` は該当時ゼロ埋めの `[hidden]`）。
pub(crate) fn layer_norm_backward_empty(
    has_weight: bool,
    has_bias: bool,
    hidden: usize,
) -> Result<LayerNormBackwardHost, RocmError> {
    let dw = if has_weight {
        Some(zeros_f32(OP_LAYER_NORM_BWD, hidden)?)
    } else {
        None
    };
    let db = if has_bias {
        Some(zeros_f32(OP_LAYER_NORM_BWD, hidden)?)
    } else {
        None
    };
    Ok((Vec::new(), dw, db))
}

/// batch_norm train の 0 要素戻り値（出力空・統計はゼロ埋めの `[c]`）。
pub(crate) fn batch_norm_train_empty(c: usize) -> Result<BatchNormTrainHost, RocmError> {
    Ok((
        Vec::new(),
        zeros_f32(OP_BATCH_NORM_TRAIN, c)?,
        zeros_f32(OP_BATCH_NORM_TRAIN, c)?,
    ))
}

/// コンパイル済みの正規化カーネルを保持し、実行する。
pub struct RocmNorm {
    host: KernelHost,
    geom: WarpGeometry,
}

impl RocmNorm {
    /// HIP/HIPRTC をロードし正規化カーネルを `target` 向けにコンパイルする。wave 幅は
    /// `WarpGeometry::for_target`（allowlist の固定値）から得る。
    ///
    /// ROCm 非搭載では `LibraryUnavailable`、デバイスがなければ `NoDevice`（panic しない）。
    pub fn new(target: GfxTarget) -> Result<Self, RocmError> {
        let geom = WarpGeometry::for_target(target);
        let src = render_norm_source(geom)?;
        Ok(Self {
            host: KernelHost::new(&src, target)?,
            geom,
        })
    }

    /// weight／bias があれば転送し、なければ `len` 要素のゼロのダミーバッファを作る。
    fn f32_or_dummy(
        &self,
        s: Option<&[f32]>,
        len: usize,
    ) -> Result<TypedDeviceBuffer<f32>, RocmError> {
        match s {
            Some(s) => TypedDeviceBuffer::from_slice(self.host.runtime(), s),
            None => TypedDeviceBuffer::new_zeroed(self.host.runtime(), len),
        }
    }

    /// 行ごとの RMSNorm（`x * rsqrt(mean(x^2) + eps) [* w]`。二乗和は f64）。
    pub fn run_rmsnorm_f32(
        &self,
        x: &[f32],
        w: Option<&[f32]>,
        eps: f32,
        rows: usize,
        hidden: usize,
    ) -> Result<Vec<f32>, RocmError> {
        let lens = OptLens {
            w: w.map(<[f32]>::len),
            ..OptLens::default()
        };
        let d = validate_row_norm(OP_RMSNORM, x.len(), rows, hidden, lens, eps)?;
        if d.numel == 0 {
            return Ok(Vec::new());
        }
        // CPU の `run_rmsnorm_f32` と同じ f32 の逆数（カーネルへは f32 引数で渡す）。
        let inv_n = 1.0f32 / hidden as f32;
        let _guard = self.host.select()?;
        let rt = self.host.runtime();
        let dx = TypedDeviceBuffer::<f32>::from_slice(rt.clone(), x)?;
        let dw = self.f32_or_dummy(w, hidden)?;
        let dout = TypedDeviceBuffer::<f32>::new_zeroed(rt, d.numel)?;
        let args = [
            Arg::Ptr(dx.as_device_ptr()),
            Arg::Ptr(dw.as_device_ptr()),
            Arg::Ptr(dout.as_device_ptr()),
            Arg::I32(d.rows_i),
            Arg::I32(d.hidden_i),
            Arg::F32(eps),
            Arg::F32(inv_n),
            Arg::I32(i32::from(w.is_some())),
        ];
        self.host
            .run_wave_blocks(OP_RMSNORM, &RMSNORM_SPEC, d.rows, self.geom.width(), &args)?;
        dout.to_vec()
    }

    /// 行ごとの LayerNorm（二パス分散・biased ÷N。統計は f64）。
    pub fn run_layer_norm_f32(
        &self,
        x: &[f32],
        w: Option<&[f32]>,
        b: Option<&[f32]>,
        eps: f32,
        rows: usize,
        hidden: usize,
    ) -> Result<Vec<f32>, RocmError> {
        let lens = OptLens {
            w: w.map(<[f32]>::len),
            b: b.map(<[f32]>::len),
            ..OptLens::default()
        };
        let d = validate_row_norm(OP_LAYER_NORM, x.len(), rows, hidden, lens, eps)?;
        if d.numel == 0 {
            return Ok(Vec::new());
        }
        let _guard = self.host.select()?;
        let rt = self.host.runtime();
        let dx = TypedDeviceBuffer::<f32>::from_slice(rt.clone(), x)?;
        let dw = self.f32_or_dummy(w, hidden)?;
        let db = self.f32_or_dummy(b, hidden)?;
        let dout = TypedDeviceBuffer::<f32>::new_zeroed(rt, d.numel)?;
        let args = [
            Arg::Ptr(dx.as_device_ptr()),
            Arg::Ptr(dw.as_device_ptr()),
            Arg::Ptr(db.as_device_ptr()),
            Arg::Ptr(dout.as_device_ptr()),
            Arg::I32(d.rows_i),
            Arg::I32(d.hidden_i),
            Arg::F32(eps),
            Arg::I32(i32::from(w.is_some())),
            Arg::I32(i32::from(b.is_some())),
        ];
        self.host.run_wave_blocks(
            OP_LAYER_NORM,
            &LAYER_NORM_SPEC,
            d.rows,
            self.geom.width(),
            &args,
        )?;
        dout.to_vec()
    }

    /// RMSNorm の逆伝播 `(dx, dw)`（統計はカーネル内で再計算。`dw` は weight があるときだけ）。
    pub fn run_rmsnorm_backward_f32(
        &self,
        x: &[f32],
        w: Option<&[f32]>,
        dy: &[f32],
        eps: f32,
        rows: usize,
        hidden: usize,
    ) -> Result<RmsNormBackwardHost, RocmError> {
        let lens = OptLens {
            w: w.map(<[f32]>::len),
            dy: Some(dy.len()),
            ..OptLens::default()
        };
        let d = validate_row_norm(OP_RMSNORM_BWD, x.len(), rows, hidden, lens, eps)?;
        if d.numel == 0 {
            return rmsnorm_backward_empty(w.is_some(), hidden);
        }
        let _guard = self.host.select()?;
        let rt = self.host.runtime();
        let dx_in = TypedDeviceBuffer::<f32>::from_slice(rt.clone(), x)?;
        let dw_in = self.f32_or_dummy(w, hidden)?;
        let ddy = TypedDeviceBuffer::<f32>::from_slice(rt.clone(), dy)?;
        let ddx = TypedDeviceBuffer::<f32>::new_zeroed(rt.clone(), d.numel)?;
        // 行ごとの rstd（f64）。dx カーネルが書き、dw カーネルが読む。起動前に 0 で初期化する。
        let rstd = DeviceBuffer::<f64>::from_slice(rt.clone(), &vec![0.0f64; rows])?;
        let dx_args = [
            Arg::Ptr(dx_in.as_device_ptr()),
            Arg::Ptr(dw_in.as_device_ptr()),
            Arg::Ptr(ddy.as_device_ptr()),
            Arg::Ptr(ddx.as_device_ptr()),
            Arg::Ptr(rstd.as_device_ptr()),
            Arg::I32(d.rows_i),
            Arg::I32(d.hidden_i),
            Arg::F32(eps),
            Arg::I32(i32::from(w.is_some())),
        ];
        self.host.run_wave_blocks(
            OP_RMSNORM_BWD,
            &RMSNORM_BWD_DX_SPEC,
            d.rows,
            self.geom.width(),
            &dx_args,
        )?;
        let dw = if w.is_some() {
            let ddw = TypedDeviceBuffer::<f32>::new_zeroed(rt, hidden)?;
            let dw_args = [
                Arg::Ptr(dx_in.as_device_ptr()),
                Arg::Ptr(ddy.as_device_ptr()),
                Arg::Ptr(rstd.as_device_ptr()),
                Arg::Ptr(ddw.as_device_ptr()),
                Arg::I32(d.rows_i),
                Arg::I32(d.hidden_i),
            ];
            // dx カーネルの完了（同期済み）後に起動する。
            self.host
                .run(OP_RMSNORM_BWD, &RMSNORM_BWD_DW_SPEC, hidden, &dw_args)?;
            Some(ddw.to_vec()?)
        } else {
            None
        };
        Ok((ddx.to_vec()?, dw))
    }

    /// LayerNorm の逆伝播 `(dx, dw, db)`。`dw` は weight、`db` は `has_bias` のときだけ。
    #[allow(clippy::too_many_arguments)] // backend-cpu の `run_batch_norm_infer_f32` と同じ。引数は BackendOps の契約どおり
    pub fn run_layer_norm_backward_f32(
        &self,
        x: &[f32],
        w: Option<&[f32]>,
        has_bias: bool,
        dy: &[f32],
        eps: f32,
        rows: usize,
        hidden: usize,
    ) -> Result<LayerNormBackwardHost, RocmError> {
        let lens = OptLens {
            w: w.map(<[f32]>::len),
            dy: Some(dy.len()),
            ..OptLens::default()
        };
        let d = validate_row_norm(OP_LAYER_NORM_BWD, x.len(), rows, hidden, lens, eps)?;
        if d.numel == 0 {
            return layer_norm_backward_empty(w.is_some(), has_bias, hidden);
        }
        let _guard = self.host.select()?;
        let rt = self.host.runtime();
        let dx_in = TypedDeviceBuffer::<f32>::from_slice(rt.clone(), x)?;
        let dw_in = self.f32_or_dummy(w, hidden)?;
        let ddy = TypedDeviceBuffer::<f32>::from_slice(rt.clone(), dy)?;
        let ddx = TypedDeviceBuffer::<f32>::new_zeroed(rt.clone(), d.numel)?;
        let mean = DeviceBuffer::<f64>::from_slice(rt.clone(), &vec![0.0f64; rows])?;
        let rstd = DeviceBuffer::<f64>::from_slice(rt.clone(), &vec![0.0f64; rows])?;
        let dx_args = [
            Arg::Ptr(dx_in.as_device_ptr()),
            Arg::Ptr(dw_in.as_device_ptr()),
            Arg::Ptr(ddy.as_device_ptr()),
            Arg::Ptr(ddx.as_device_ptr()),
            Arg::Ptr(mean.as_device_ptr()),
            Arg::Ptr(rstd.as_device_ptr()),
            Arg::I32(d.rows_i),
            Arg::I32(d.hidden_i),
            Arg::F32(eps),
            Arg::I32(i32::from(w.is_some())),
        ];
        self.host.run_wave_blocks(
            OP_LAYER_NORM_BWD,
            &LAYER_NORM_BWD_DX_SPEC,
            d.rows,
            self.geom.width(),
            &dx_args,
        )?;
        let (dw, db) = if w.is_some() || has_bias {
            let ddw = TypedDeviceBuffer::<f32>::new_zeroed(rt.clone(), hidden)?;
            let ddb = TypedDeviceBuffer::<f32>::new_zeroed(rt, hidden)?;
            let args = [
                Arg::Ptr(dx_in.as_device_ptr()),
                Arg::Ptr(ddy.as_device_ptr()),
                Arg::Ptr(mean.as_device_ptr()),
                Arg::Ptr(rstd.as_device_ptr()),
                Arg::Ptr(ddw.as_device_ptr()),
                Arg::Ptr(ddb.as_device_ptr()),
                Arg::I32(d.rows_i),
                Arg::I32(d.hidden_i),
                Arg::I32(i32::from(w.is_some())),
                Arg::I32(i32::from(has_bias)),
            ];
            self.host
                .run(OP_LAYER_NORM_BWD, &LAYER_NORM_BWD_DWDB_SPEC, hidden, &args)?;
            (
                if w.is_some() {
                    Some(ddw.to_vec()?)
                } else {
                    None
                },
                if has_bias { Some(ddb.to_vec()?) } else { None },
            )
        } else {
            (None, None)
        };
        Ok((ddx.to_vec()?, dw, db))
    }

    /// BatchNorm の train（バッチ統計。チャネルごとに 1 wave）。`(out, mean, var)` を返す。
    #[allow(clippy::too_many_arguments)] // backend-cpu の `run_batch_norm_train_f32` と同じ引数構成
    pub fn run_batch_norm_train_f32(
        &self,
        x: &[f32],
        w: Option<&[f32]>,
        b: Option<&[f32]>,
        eps: f32,
        n: usize,
        c: usize,
        spatial: usize,
    ) -> Result<BatchNormTrainHost, RocmError> {
        let lens = OptLens {
            w: w.map(<[f32]>::len),
            b: b.map(<[f32]>::len),
            ..OptLens::default()
        };
        let d = validate_batch_norm(OP_BATCH_NORM_TRAIN, x.len(), (n, c, spatial), lens, eps)?;
        if d.numel == 0 {
            return batch_norm_train_empty(c);
        }
        let _guard = self.host.select()?;
        let rt = self.host.runtime();
        let dx = TypedDeviceBuffer::<f32>::from_slice(rt.clone(), x)?;
        let dw = self.f32_or_dummy(w, c)?;
        let db = self.f32_or_dummy(b, c)?;
        let dout = TypedDeviceBuffer::<f32>::new_zeroed(rt.clone(), d.numel)?;
        let dmean = TypedDeviceBuffer::<f32>::new_zeroed(rt.clone(), c)?;
        let dvar = TypedDeviceBuffer::<f32>::new_zeroed(rt, c)?;
        let args = [
            Arg::Ptr(dx.as_device_ptr()),
            Arg::Ptr(dw.as_device_ptr()),
            Arg::Ptr(db.as_device_ptr()),
            Arg::Ptr(dout.as_device_ptr()),
            Arg::Ptr(dmean.as_device_ptr()),
            Arg::Ptr(dvar.as_device_ptr()),
            Arg::I32(d.c_i),
            Arg::I32(d.spatial_i),
            Arg::I32(d.m_i),
            Arg::F32(eps),
            Arg::I32(i32::from(w.is_some())),
            Arg::I32(i32::from(b.is_some())),
        ];
        self.host.run_wave_blocks(
            OP_BATCH_NORM_TRAIN,
            &BATCH_NORM_TRAIN_SPEC,
            c,
            self.geom.width(),
            &args,
        )?;
        Ok((dout.to_vec()?, dmean.to_vec()?, dvar.to_vec()?))
    }

    /// BatchNorm の infer（固定統計。1 スレッド = 1 要素）。
    #[allow(clippy::too_many_arguments)] // backend-cpu の `run_batch_norm_infer_f32` と同じ引数構成
    pub fn run_batch_norm_infer_f32(
        &self,
        x: &[f32],
        mean: &[f32],
        var: &[f32],
        w: Option<&[f32]>,
        b: Option<&[f32]>,
        eps: f32,
        n: usize,
        c: usize,
        spatial: usize,
    ) -> Result<Vec<f32>, RocmError> {
        let lens = OptLens {
            w: w.map(<[f32]>::len),
            b: b.map(<[f32]>::len),
            mean: Some(mean.len()),
            var: Some(var.len()),
            ..OptLens::default()
        };
        let d = validate_batch_norm(OP_BATCH_NORM_INFER, x.len(), (n, c, spatial), lens, eps)?;
        if d.numel == 0 {
            return Ok(Vec::new());
        }
        let _guard = self.host.select()?;
        let rt = self.host.runtime();
        let dx = TypedDeviceBuffer::<f32>::from_slice(rt.clone(), x)?;
        let dmean = TypedDeviceBuffer::<f32>::from_slice(rt.clone(), mean)?;
        let dvar = TypedDeviceBuffer::<f32>::from_slice(rt.clone(), var)?;
        let dw = self.f32_or_dummy(w, c)?;
        let db = self.f32_or_dummy(b, c)?;
        let dout = TypedDeviceBuffer::<f32>::new_zeroed(rt, d.numel)?;
        let args = [
            Arg::Ptr(dx.as_device_ptr()),
            Arg::Ptr(dmean.as_device_ptr()),
            Arg::Ptr(dvar.as_device_ptr()),
            Arg::Ptr(dw.as_device_ptr()),
            Arg::Ptr(db.as_device_ptr()),
            Arg::Ptr(dout.as_device_ptr()),
            Arg::I32(d.c_i),
            Arg::I32(d.spatial_i),
            Arg::I64(d.numel as i64),
            Arg::F32(eps),
            Arg::I32(i32::from(w.is_some())),
            Arg::I32(i32::from(b.is_some())),
        ];
        self.host
            .run(OP_BATCH_NORM_INFER, &BATCH_NORM_INFER_SPEC, d.numel, &args)?;
        dout.to_vec()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn lens_w(h: usize) -> OptLens {
        OptLens {
            w: Some(h),
            ..OptLens::default()
        }
    }

    #[test]
    fn row_validation_accepts_ordinary_and_empty() {
        let d = validate_row_norm("t", 24, 3, 8, lens_w(8), 1e-5).unwrap();
        assert_eq!(
            (d.rows, d.hidden, d.numel, d.rows_i, d.hidden_i),
            (3, 8, 24, 3, 8)
        );
        assert!(validate_row_norm("t", 24, 3, 8, OptLens::default(), 0.0).is_ok());
        // 0 要素（rows=0 でも hidden=0 でも通り、rows_i は 0）。
        assert_eq!(
            validate_row_norm("t", 0, 0, 8, lens_w(8), 1e-5)
                .unwrap()
                .rows_i,
            0
        );
        assert_eq!(
            validate_row_norm("t", 0, 3, 0, lens_w(0), 1e-5)
                .unwrap()
                .numel,
            0
        );
    }

    #[test]
    fn row_validation_rejects_bad_eps_lengths_and_overflow() {
        for eps in [-1.0, f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            assert!(
                matches!(
                    validate_row_norm("t", 24, 3, 8, OptLens::default(), eps),
                    Err(RocmError::InvalidOpArgs { .. })
                ),
                "{eps}"
            );
        }
        let none = OptLens::default();
        assert!(validate_row_norm("t", 23, 3, 8, none, 1e-5).is_err());
        assert!(validate_row_norm("t", 24, 3, 8, lens_w(7), 1e-5).is_err());
        let b = OptLens { b: Some(9), ..none };
        assert!(validate_row_norm("t", 24, 3, 8, b, 1e-5).is_err());
        let dy = OptLens {
            dy: Some(23),
            ..none
        };
        assert!(validate_row_norm("t", 24, 3, 8, dy, 1e-5).is_err());
        assert!(validate_row_norm("t", 0, usize::MAX, 2, none, 1e-5).is_err());
        // i32 に収まらない。
        let big = i32::MAX as usize + 1;
        assert!(validate_row_norm("t", big, 1, big, none, 1e-5).is_err());
        assert!(validate_row_norm("t", 0, 0, big, none, 1e-5).is_err());
        assert!(validate_row_norm("t", big, big, 1, none, 1e-5).is_err());
    }

    #[test]
    fn channel_validation_accepts_ordinary_and_empty() {
        let w = OptLens {
            w: Some(3),
            b: Some(3),
            mean: Some(3),
            var: Some(3),
            dy: None,
        };
        let d = validate_batch_norm("t", 24, (2, 3, 4), w, 1e-5).unwrap();
        assert_eq!((d.c, d.numel, d.c_i, d.spatial_i, d.m_i), (3, 24, 3, 4, 8));
        let e = validate_batch_norm("t", 0, (0, 3, 4), OptLens::default(), 1e-5).unwrap();
        assert_eq!((e.numel, e.spatial_i, e.m_i), (0, 0, 0));
        // c=0 は n が巨大でも空として通る（m は検査しない）。
        assert!(validate_batch_norm("t", 0, (1 << 40, 0, 1), OptLens::default(), 1e-5).is_ok());
    }

    #[test]
    fn channel_validation_rejects_bad_inputs() {
        let none = OptLens::default();
        assert!(validate_batch_norm("t", 23, (2, 3, 4), none, 1e-5).is_err());
        for eps in [-1.0, f32::NAN, f32::INFINITY] {
            assert!(validate_batch_norm("t", 24, (2, 3, 4), none, eps).is_err());
        }
        for lens in [
            OptLens { w: Some(2), ..none },
            OptLens { b: Some(2), ..none },
            OptLens {
                mean: Some(2),
                ..none
            },
            OptLens {
                var: Some(4),
                ..none
            },
        ] {
            assert!(validate_batch_norm("t", 24, (2, 3, 4), lens, 1e-5).is_err());
        }
        assert!(validate_batch_norm("t", 0, (0, usize::MAX, 2), none, 1e-5).is_err());
        let big = i32::MAX as usize + 1;
        assert!(validate_batch_norm("t", 0, (0, big, 1), none, 1e-5).is_err());
        assert!(validate_batch_norm("t", big, (1, 1, big), none, 1e-5).is_err());
    }

    #[test]
    fn zero_element_contracts() {
        assert_eq!(
            rmsnorm_backward_empty(true, 3).unwrap(),
            (Vec::new(), Some(vec![0.0; 3]))
        );
        assert_eq!(
            rmsnorm_backward_empty(false, 3).unwrap(),
            (Vec::new(), None)
        );
        assert_eq!(
            layer_norm_backward_empty(true, false, 2).unwrap(),
            (Vec::new(), Some(vec![0.0; 2]), None)
        );
        assert_eq!(
            layer_norm_backward_empty(false, true, 2).unwrap(),
            (Vec::new(), None, Some(vec![0.0; 2]))
        );
        assert_eq!(
            batch_norm_train_empty(2).unwrap(),
            (Vec::new(), vec![0.0; 2], vec![0.0; 2])
        );
    }
}
