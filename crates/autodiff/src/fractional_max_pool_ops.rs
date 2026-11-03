//! `fractional_max_pool2d`・`fractional_max_pool3d` の自由関数（イシュー #3826・親 #3749
//! 「FractionalMaxPool」。設計・実装記録は `docs/autodiff-pool3d-ops-decision.md` §20・§24）。
//!
//! **facade 公開状況**: `Var::fractional_max_pool2d`／`Var::fractional_max_pool3d`（本モジュールの
//! グローバル RNG 経路への 1 行委譲）として公開する。委譲本体は
//! `crates/facade/tests/api_surface.rs` の `var_fractional_max_pool_methods_are_thin_delegations` が
//! 固定する。モジュール自体・標本を明示する入口（`*_with_samples`。PyTorch の `_random_samples` 相当）は
//! `pub(crate)` のままで再エクスポートしない（Q2(a)）。層型・`Sequential::add_fractional_max_pool*`
//! は #3827 の担当。
//!
//! **PyTorch 相当**: `F.fractional_max_pool{2,3}d(x, kernel_size, output_size=..., return_indices=True)`。
//! NCHW／NCDHW 固定（バッチなし入力・`output_ratio` は v1 では受け付けない。Q3(a)）。
//!
//! **乱数契約**: 標本は `[N, C, R]`（R は空間軸数）をホストのグローバル RNG（`rng::rand`）から
//! **すべての検査を通った後に一括で**引く（消費数はちょうど `N·C·R`。`N·C = 0` は消費しない）。
//! `eval()` でも毎回引く（Q4(a)）。決定的になるのは `manual_seed` の下だけで、PyTorch と同じシードでも
//! 同じ値は出ない。xorshift64* は暗号用途ではない。
//!
//! **経路**: ① 形状・`kernel_size`・`output_size`・索引上限の検査（確保・RNG 消費・tape 操作より前。
//! エラー時に孤児ノードを残さない）→ ② 標本の取得（グローバル RNG 経路のみ）→ ③ 標本の検証と
//! ホストでの窓表の構築（整数）→ ④ 入力の実体化 → `BackendOps::fractional_max_pool{2,3}d_windows`
//! （`Unsupported` のときだけ共有ホストカーネルへフォールバックし、他のエラーは伝播する。戻り値の
//! shape も検証する）→ ⑤ `push_eager`。Op は既存の `Op::MaxPool2d`／`Op::MaxPool3d { input, index }` を
//! 再利用する（VJP は索引だけで決まり、checkpoint／recompute の対象外なので標本を引き直さない）。
//!
//! **数値契約**: 選択のみ（値は入力要素と bit 一致・タイ先勝ち・NaN 伝播）。同じ標本なら
//! バックエンド間で bit 一致する（規則の正は `fandhe_ai_tensor_core::fractional_max_pool`）。

use fandhe_ai_tensor_core::fractional_max_pool::{
    self as fmp, FractionalMaxPoolError, FractionalMaxPoolWindows,
};
use fandhe_ai_tensor_core::{BackendError, ShapeError, Tensor};

use crate::error::AutodiffError;
use crate::tape::{Op, materialize_fallible};
use crate::var::Var;

fn map_err(e: FractionalMaxPoolError) -> AutodiffError {
    match e {
        FractionalMaxPoolError::Shape(s) => AutodiffError::Shape(s),
        FractionalMaxPoolError::InvalidArgument(m) => AutodiffError::InvalidArgument(m),
    }
}

fn verify_shape(actual: &[usize], expected: &[usize]) -> Result<(), AutodiffError> {
    if actual == expected {
        Ok(())
    } else {
        Err(AutodiffError::Backend(BackendError::ShapeMismatch(
            ShapeError::ShapeMismatch {
                lhs: actual.to_vec(),
                rhs: expected.to_vec(),
            },
        )))
    }
}

fn materialize_one<'t>(x: &Var<'t>) -> Result<Tensor<f32>, AutodiffError> {
    let nodes = x.tape().nodes.borrow();
    let ops = x.tape().ops();
    Ok(materialize_fallible(&nodes, ops, x.node_id())?.clone())
}

/// 窓表が確定した後の共通経路（実体化 → backend／ホスト → shape 再検証 → `push_eager`）。
/// `to_op` は 2d が `Op::MaxPool2d`、3d が `Op::MaxPool3d` を作る。
fn run_windows<'t, const R: usize>(
    input: &Var<'t>,
    windows: &FractionalMaxPoolWindows<R>,
    backend: impl FnOnce(&Tensor<f32>) -> Result<(Tensor<f32>, Tensor<i32>), BackendError>,
    to_op: impl FnOnce(crate::tape::NodeId, Tensor<i32>) -> Op,
) -> Result<(Var<'t>, Tensor<i32>), AutodiffError> {
    let x = materialize_one(input)?;
    let out_shape = windows.out_shape();
    let (value, index) = match backend(&x) {
        Ok((v, i)) => {
            verify_shape(v.shape(), &out_shape)?;
            verify_shape(i.shape(), &out_shape)?;
            (v, i)
        }
        Err(BackendError::Unsupported(_)) => {
            let (v, i) = fmp::fractional_max_pool_host(&x.contiguous().host_slice(), windows)
                .map_err(AutodiffError::Shape)?;
            (
                Tensor::new(v, &out_shape).map_err(AutodiffError::Shape)?,
                Tensor::new(i, &out_shape).map_err(AutodiffError::Shape)?,
            )
        }
        Err(other) => return Err(AutodiffError::Backend(other)),
    };
    let id = input
        .tape()
        .push_eager(to_op(input.node_id(), index.clone()), value);
    Ok((Var::from_raw(input.tape(), id), index))
}

/// 検査 → （グローバル RNG なら）標本を引く → 窓表。標本を明示する経路は `samples` を渡す。
fn build_windows<const R: usize>(
    in_shape: &[usize],
    kernel: [usize; R],
    output: [usize; R],
    samples: Option<&Tensor<f32>>,
) -> Result<FractionalMaxPoolWindows<R>, AutodiffError> {
    fmp::fractional_max_pool_check::<R>(in_shape, kernel, output).map_err(map_err)?;
    match samples {
        Some(s) => fmp::fractional_max_pool_windows(
            in_shape,
            kernel,
            output,
            &s.contiguous().host_slice(),
            s.shape(),
        ),
        None => {
            let drawn = fandhe_ai_tensor_core::rng::rand(&[in_shape[0], in_shape[1], R])
                .map_err(AutodiffError::Shape)?;
            fmp::fractional_max_pool_windows(
                in_shape,
                kernel,
                output,
                &drawn.host_slice(),
                drawn.shape(),
            )
        }
    }
    .map_err(map_err)
}

/// 2D fractional max pooling（`F.fractional_max_pool2d(..., return_indices=True)` 相当）。
/// 標本はグローバル RNG から引く。`input`: `[N, C, H, W]`。値と索引を返す。
pub(crate) fn fractional_max_pool2d<'t>(
    input: &Var<'t>,
    kernel_size: [usize; 2],
    output_size: [usize; 2],
) -> Result<(Var<'t>, Tensor<i32>), AutodiffError> {
    let w = build_windows::<2>(&input.shape(), kernel_size, output_size, None)?;
    finish_2d(input, &w)
}

/// 標本を明示する 2D 版（PyTorch の `_random_samples` 相当）。`samples`: `[N, C, 2]`（`[0]=W`・`[1]=H`）。
#[cfg(test)]
pub(crate) fn fractional_max_pool2d_with_samples<'t>(
    input: &Var<'t>,
    kernel_size: [usize; 2],
    output_size: [usize; 2],
    samples: &Tensor<f32>,
) -> Result<(Var<'t>, Tensor<i32>), AutodiffError> {
    let w = build_windows::<2>(&input.shape(), kernel_size, output_size, Some(samples))?;
    finish_2d(input, &w)
}

fn finish_2d<'t>(
    input: &Var<'t>,
    w: &FractionalMaxPoolWindows<2>,
) -> Result<(Var<'t>, Tensor<i32>), AutodiffError> {
    run_windows(
        input,
        w,
        |x| input.tape().ops().fractional_max_pool2d_windows(x, w),
        |id, index| Op::MaxPool2d { input: id, index },
    )
}

/// 3D fractional max pooling。標本はグローバル RNG から引く。`input`: `[N, C, D, H, W]`。
pub(crate) fn fractional_max_pool3d<'t>(
    input: &Var<'t>,
    kernel_size: [usize; 3],
    output_size: [usize; 3],
) -> Result<(Var<'t>, Tensor<i32>), AutodiffError> {
    let w = build_windows::<3>(&input.shape(), kernel_size, output_size, None)?;
    finish_3d(input, &w)
}

/// 標本を明示する 3D 版。`samples`: `[N, C, 3]`（`[0]=T`・`[1]=H`・`[2]=W`）。
#[cfg(test)]
pub(crate) fn fractional_max_pool3d_with_samples<'t>(
    input: &Var<'t>,
    kernel_size: [usize; 3],
    output_size: [usize; 3],
    samples: &Tensor<f32>,
) -> Result<(Var<'t>, Tensor<i32>), AutodiffError> {
    let w = build_windows::<3>(&input.shape(), kernel_size, output_size, Some(samples))?;
    finish_3d(input, &w)
}

fn finish_3d<'t>(
    input: &Var<'t>,
    w: &FractionalMaxPoolWindows<3>,
) -> Result<(Var<'t>, Tensor<i32>), AutodiffError> {
    run_windows(
        input,
        w,
        |x| input.tape().ops().fractional_max_pool3d_windows(x, w),
        |id, index| Op::MaxPool3d { input: id, index },
    )
}
#[cfg(test)]
mod tests {
    use super::*;
    use crate::Tape;

    const REFERENCE: &str = include_str!(
        "../tests/fixtures/fractional-max-pool-pytorch-reference/fractional_max_pool_reference.json"
    );

    fn tensor(data: Vec<f32>, shape: &[usize]) -> Tensor<f32> {
        Tensor::new(data, shape).expect("test fixture: shape とデータ長は一致させている")
    }

    fn floats(v: &serde_json::Value) -> Vec<f32> {
        v.as_array()
            .expect("array")
            .iter()
            .map(|x| x.as_f64().expect("number") as f32)
            .collect()
    }

    fn usizes(v: &serde_json::Value) -> Vec<usize> {
        v.as_array()
            .expect("array")
            .iter()
            .map(|x| x.as_u64().expect("uint") as usize)
            .collect()
    }

    fn bits(v: &[f32]) -> Vec<u32> {
        v.iter().map(|x| x.to_bits()).collect()
    }

    /// 次元数で 2d／3d を呼び分ける（テスト専用）。
    fn run_with_samples<'t>(
        x: &Var<'t>,
        dim: usize,
        k: &[usize],
        o: &[usize],
        samples: &Tensor<f32>,
    ) -> Result<(Var<'t>, Tensor<i32>), AutodiffError> {
        if dim == 2 {
            fractional_max_pool2d_with_samples(x, [k[0], k[1]], [o[0], o[1]], samples)
        } else {
            fractional_max_pool3d_with_samples(x, [k[0], k[1], k[2]], [o[0], o[1], o[2]], samples)
        }
    }

    /// PyTorch 2.14.0 の実行値 fixture（`_random_samples` 明示）と、値（bit 一致）・索引（完全一致）・
    /// 入力勾配（重み付き和の backward。整数値の scatter-add なので bit 一致）を突合する。
    #[test]
    fn matches_pytorch_reference_fixture() {
        let doc: serde_json::Value = serde_json::from_str(REFERENCE).expect("fixture json");
        let cases = doc["cases"].as_array().expect("cases");
        assert!(cases.len() >= 5);
        for c in cases {
            let name = c["name"].as_str().expect("name");
            let in_shape = usizes(&c["in_shape"]);
            let dim = c["dim"].as_u64().expect("dim") as usize;
            let samples = tensor(floats(&c["samples"]), &[in_shape[0], in_shape[1], dim]);
            let tape = Tape::new();
            let x = tape.var(&tensor(floats(&c["input"]), &in_shape));
            let (y, idx) = run_with_samples(
                &x,
                dim,
                &usizes(&c["kernel"]),
                &usizes(&c["output"]),
                &samples,
            )
            .unwrap_or_else(|e| panic!("{name}: {e:?}"));
            assert_eq!(
                bits(&y.to_tensor().host_slice()),
                bits(&floats(&c["output_values"])),
                "{name}: values"
            );
            let want_idx: Vec<i32> = c["indices"]
                .as_array()
                .expect("indices")
                .iter()
                .map(|v| v.as_i64().expect("int") as i32)
                .collect();
            assert_eq!(idx.host_slice().to_vec(), want_idx, "{name}: indices");
            let w = tape.var_no_grad(&tensor(floats(&c["weights"]), y.to_tensor().shape()));
            let loss = y.mul(&w).unwrap().sum(None).unwrap();
            let grads = tape.backward(&loss).unwrap();
            let dx = grads.get(&x).unwrap().unwrap().host_slice().to_vec();
            assert_eq!(
                bits(&dx),
                bits(&floats(&c["input_grad"])),
                "{name}: input_grad"
            );
        }
    }

    /// 実行可能性の式は PyTorch 実測（probes）と一致する。標本の範囲外だけは意図的な差分
    /// （PyTorch は受理、本実装は拒否。設計記録 §24）。
    #[test]
    fn feasibility_matches_pytorch_probes_and_out_of_range_samples_are_rejected() {
        let doc: serde_json::Value = serde_json::from_str(REFERENCE).expect("fixture json");
        for p in doc["probes"].as_array().expect("probes") {
            let in_shape = usizes(&p["in_shape"]);
            let dim = p["dim"].as_u64().expect("dim") as usize;
            let samples = floats(&p["samples"]);
            let in_range = samples.iter().all(|s| (0.0..1.0).contains(s));
            let torch_ok = p["accepted"].as_bool().expect("accepted");
            let samples = tensor(samples, &[in_shape[0], in_shape[1], dim]);
            let tape = Tape::new();
            let n: usize = in_shape.iter().product();
            let x = tape.var(&tensor(vec![0.5; n], &in_shape));
            let r = run_with_samples(
                &x,
                dim,
                &usizes(&p["kernel"]),
                &usizes(&p["output"]),
                &samples,
            );
            assert_eq!(r.is_ok(), torch_ok && in_range, "probe {p}");
        }
    }

    /// 検査・標本検証で拒否した入力は tape にノードを残さない。
    #[test]
    fn rejected_inputs_leave_no_orphan_nodes() {
        let tape = Tape::new();
        let x = tape.var(&tensor(vec![0.0; 16], &[1, 1, 4, 4]));
        let r3 = tape.var(&tensor(vec![0.0; 16], &[1, 4, 4]));
        let bad_shape = tensor(vec![0.1; 3], &[1, 1, 3]);
        let nan = tensor(vec![f32::NAN, 0.1], &[1, 1, 2]);
        let before = tape.nodes.borrow().len();
        assert!(fractional_max_pool2d(&x, [0, 1], [1, 1]).is_err());
        assert!(fractional_max_pool2d(&x, [2, 2], [4, 4]).is_err());
        assert!(fractional_max_pool2d(&r3, [1, 1], [1, 1]).is_err());
        assert!(fractional_max_pool2d_with_samples(&x, [2, 2], [2, 2], &bad_shape).is_err());
        assert!(fractional_max_pool2d_with_samples(&x, [2, 2], [2, 2], &nan).is_err());
        assert!(fractional_max_pool3d(&x, [1, 1, 1], [1, 1, 1]).is_err());
        assert_eq!(tape.nodes.borrow().len(), before);
    }

    /// 重なり窓（W 軸のみ）で同じ入力要素が勝つと勾配は加算される。記録ノードは 1 つだけ。
    #[test]
    fn overlapping_windows_accumulate_gradients_hand_computed() {
        let tape = Tape::new();
        let x = tape.var(&tensor(vec![0.0, 5.0, 1.0], &[1, 1, 1, 3]));
        let samples = tensor(vec![0.4, 0.4], &[1, 1, 2]);
        let before = tape.nodes.borrow().len();
        let (y, idx) = fractional_max_pool2d_with_samples(&x, [1, 2], [1, 2], &samples).unwrap();
        assert_eq!(tape.nodes.borrow().len(), before + 1);
        assert_eq!(y.to_tensor().host_slice().to_vec(), vec![5.0, 5.0]);
        assert_eq!(idx.host_slice().to_vec(), vec![1, 1]);
        let loss = y.sum(None).unwrap();
        let g = tape.backward(&loss).unwrap();
        assert_eq!(
            g.get(&x).unwrap().unwrap().host_slice().to_vec(),
            vec![0.0, 2.0, 0.0]
        );
    }
}
