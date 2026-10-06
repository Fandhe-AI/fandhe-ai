//! `max_unpool1d`／`max_unpool2d`／`max_unpool3d` の自由関数（`F.max_unpool1d/2d/3d` 相当。
//! イシュー #2644・親 #2625「Phase 4」・ルート #2499）。
//!
//! **facade 非公開（保留）**: 公開形（`Var::max_unpool1d/2d/3d` の委譲メソッド）は未承認で、承認依頼は
//! #2677（公開自体は承認後の #2678）。層化（`nn::MaxUnpool*`・`Sequential::add_*`）は #2679 の対象で
//! 本イシューでは作らない。本モジュールは内部クレート限定の入口で、`Var` に inherent メソッドを足さない。
//! 保留は `crates/facade/src/lib.rs` の `ConvTranspose3dMaxUnpoolHoldDoctestGuard` と
//! `crates/facade/tests/api_surface.rs` の否定ガードが機械的に固定する
//! （`docs/autodiff-conv-transpose3d-max-unpool-decision.md`）。
//!
//! **PyTorch 相当**: `F.max_unpool{1,2,3}d(input, indices, kernel_size, stride, padding, output_size)`。
//! 入力 `[N, C, 空間...]`・`indices` は同 shape の `Tensor<i32>`（`Var::max_pool1d`／`max_pool2d`／
//! `pool3d_ops::max_pool3d` などが返す `(n, c)` 平面内 flat 添字をそのまま渡せる）。出力は
//! `[N, C, 出力空間...]`（既定長 `(in − 1)·stride − 2·padding + kernel`。`output_size` は空間軸のみで
//! PyTorch 実測どおり `default ± stride` の開区間）。`stride = None` は `kernel_size`。
//!
//! **経路**: ① shape・サイズ検査（`tensor_core::max_unpool::max_unpool_layout`。tape 操作・実体化より
//! 前）→ ② 索引値の範囲検査（負値・`>= 出力平面長` は `AutodiffError::InvalidArgument`。
//! `Var::scatter` と同じ慣例）→ ③ 入力を `[N·C, L_in]` へ平坦化して
//! `scatter_with_fallback`（既存 `BackendOps::scatter` フック。`Unsupported` のときだけホスト
//! `eval::scatter`。他のエラーは伝播）で `ScatterReduce::Overwrite` → ④ 戻り shape 再検証 →
//! ⑤ `Op::MaxUnpool` を `push_eager`（1 ノード）。VJP は `grad.rs`（`gather`＋last-writer マスク）。
//!
//! **数値契約**: 算術を含まないコピーなので値は入力要素と bit 一致（NaN／inf も保存）。重複索引の
//! forward は `ScatterReduce::Overwrite` の決定的契約（row-major 走査で最後の書き手が残る。PyTorch の
//! 重複索引時の勝者は未規定〈実測では書き手により異なる〉）。高階微分（`create_graph`）・activation
//! checkpoint・f64 経路は対象外。`N = 0`／`C = 0` は受理して空出力を返す（PyTorch は `C = 0` を拒否）。

use fandhe_ai_tensor_core::max_unpool::{MaxUnpoolLayout, max_unpool_layout};
use fandhe_ai_tensor_core::{ScatterReduce, Tensor};

use crate::error::AutodiffError;
use crate::grad::scatter_with_fallback;
use crate::tape::{Op, materialize_fallible};
use crate::var::Var;

/// 1d／2d／3d 共通の本体。`kernel_size.len()` が空間 rank。
fn max_unpool_impl<'t>(
    input: &Var<'t>,
    indices: &Tensor<i32>,
    kernel_size: &[usize],
    stride: Option<&[usize]>,
    padding: &[usize],
    output_size: Option<&[usize]>,
) -> Result<Var<'t>, AutodiffError> {
    let layout: MaxUnpoolLayout = max_unpool_layout(
        &input.shape(),
        indices.shape(),
        kernel_size,
        stride,
        padding,
        output_size,
    )
    .map_err(AutodiffError::Shape)?;
    let in_shape = layout.in_shape();
    let out_shape = layout.out_shape();
    let (flat_in, flat_out) = (layout.flat_in(), layout.flat_out());

    // 索引値の範囲検査（scatter／gather へ渡す前。REQ-8: 手動境界検査を省略しない）。
    let idx_c = indices.contiguous();
    let out_plane = layout.out_plane();
    for v in idx_c.host_slice().iter() {
        let vi = i64::from(*v);
        if vi < 0 || (vi as usize) >= out_plane {
            return Err(AutodiffError::InvalidArgument(format!(
                "max_unpool: indices の値 {vi} が [0, {out_plane}) の範囲外（出力の (n, c) 平面長）"
            )));
        }
    }

    let x = {
        let nodes = input.tape().nodes.borrow();
        materialize_fallible(&nodes, input.tape().ops(), input.node_id())?.clone()
    };
    let value = if flat_in[0] == 0 {
        // 空バッチ／空チャンネル: scatter を呼ばず空テンソルを直接返す。
        Tensor::zeros(&out_shape).map_err(AutodiffError::Shape)?
    } else {
        let x_flat = x
            .contiguous()
            .reshape(&flat_in)
            .map_err(AutodiffError::Shape)?;
        let idx_flat = idx_c.reshape(&flat_in).map_err(AutodiffError::Shape)?;
        let zeros = Tensor::zeros(&flat_out).map_err(AutodiffError::Shape)?;
        scatter_with_fallback(
            input.tape().ops(),
            &zeros,
            1,
            &idx_flat,
            &x_flat,
            ScatterReduce::Overwrite,
            &flat_out,
        )?
        .reshape(&out_shape)
        .map_err(AutodiffError::Shape)?
    };
    let index_record = idx_c.reshape(&in_shape).map_err(AutodiffError::Shape)?;
    let id = input.tape().push_eager(
        Op::MaxUnpool {
            input: input.node_id(),
            index: index_record,
        },
        value,
    );
    Ok(Var::from_raw(input.tape(), id))
}

/// 1D max unpooling（`F.max_unpool1d` 相当）。`input`: `[N, C, L]`・`indices`: 同 shape の
/// `(n, c)` 平面内添字（`w`）。`stride = None` は `kernel_size`。`output_size` は出力長
/// （`None` は既定長）。検査順は共通本体の経路（モジュール doc）を参照。
pub fn max_unpool1d<'t>(
    input: &Var<'t>,
    indices: &Tensor<i32>,
    kernel_size: usize,
    stride: Option<usize>,
    padding: usize,
    output_size: Option<usize>,
) -> Result<Var<'t>, AutodiffError> {
    let stride_arr = stride.map(|s| [s]);
    let output_arr = output_size.map(|o| [o]);
    max_unpool_impl(
        input,
        indices,
        &[kernel_size],
        stride_arr.as_ref().map(|s| s.as_slice()),
        &[padding],
        output_arr.as_ref().map(|o| o.as_slice()),
    )
}

/// 2D max unpooling（`F.max_unpool2d` 相当）。`input`: `[N, C, H, W]`・`indices`: 同 shape の
/// `(n, c)` 平面内添字（`h·W_out + w`。`W_out` は **出力** の幅）。
pub fn max_unpool2d<'t>(
    input: &Var<'t>,
    indices: &Tensor<i32>,
    kernel_size: [usize; 2],
    stride: Option<[usize; 2]>,
    padding: [usize; 2],
    output_size: Option<[usize; 2]>,
) -> Result<Var<'t>, AutodiffError> {
    max_unpool_impl(
        input,
        indices,
        &kernel_size,
        stride.as_ref().map(|s| s.as_slice()),
        &padding,
        output_size.as_ref().map(|o| o.as_slice()),
    )
}

/// 3D max unpooling（`F.max_unpool3d` 相当）。`input`: `[N, C, D, H, W]`・`indices`: 同 shape の
/// `(n, c)` 平面内添字（`d·H_out·W_out + h·W_out + w`。`H_out`／`W_out` は **出力** の寸法）。
pub fn max_unpool3d<'t>(
    input: &Var<'t>,
    indices: &Tensor<i32>,
    kernel_size: [usize; 3],
    stride: Option<[usize; 3]>,
    padding: [usize; 3],
    output_size: Option<[usize; 3]>,
) -> Result<Var<'t>, AutodiffError> {
    max_unpool_impl(
        input,
        indices,
        &kernel_size,
        stride.as_ref().map(|s| s.as_slice()),
        &padding,
        output_size.as_ref().map(|o| o.as_slice()),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tape::Tape;

    fn t(data: Vec<f32>, shape: &[usize]) -> Tensor<f32> {
        Tensor::new(data, shape).expect("test fixture: shape 一致")
    }

    fn ti(data: Vec<i32>, shape: &[usize]) -> Tensor<i32> {
        Tensor::new(data, shape).expect("test fixture: shape 一致")
    }

    #[test]
    fn unpool1d_places_values_at_indices() {
        let tape = Tape::new();
        let x = tape.var(&t(vec![5.0, 6.0, 7.0], &[1, 1, 3]));
        let idx = ti(vec![1, 2, 5], &[1, 1, 3]);
        let y = max_unpool1d(&x, &idx, 2, None, 0, None).unwrap();
        let out = y.to_tensor();
        assert_eq!(out.shape(), &[1, 1, 6]);
        assert_eq!(
            out.host_slice().into_owned(),
            vec![0.0, 5.0, 6.0, 0.0, 0.0, 7.0]
        );
    }

    #[test]
    fn unpool2d_indices_address_output_plane() {
        let tape = Tape::new();
        let x = tape.var(&t(vec![1.0, 2.0, 3.0, 4.0], &[1, 1, 2, 2]));
        // 出力は 4×4。平面内添字 h·4 + w。
        let idx = ti(vec![0, 7, 8, 15], &[1, 1, 2, 2]);
        let y = max_unpool2d(&x, &idx, [2, 2], None, [0, 0], None).unwrap();
        let out = y.to_tensor().host_slice().into_owned();
        assert_eq!(out.len(), 16);
        assert_eq!((out[0], out[7], out[8], out[15]), (1.0, 2.0, 3.0, 4.0));
        assert_eq!(out.iter().filter(|v| **v != 0.0).count(), 4);
    }

    #[test]
    fn rejects_invalid_indices_and_arguments_without_orphan_nodes() {
        let tape = Tape::new();
        let x = tape.var(&t(vec![1.0, 2.0, 3.0], &[1, 1, 3]));
        let before = tape.len();
        for bad in [vec![0, 1, 6], vec![0, 1, -1]] {
            assert!(matches!(
                max_unpool1d(&x, &ti(bad, &[1, 1, 3]), 2, None, 0, None),
                Err(AutodiffError::InvalidArgument(_))
            ));
        }
        let ok = ti(vec![0, 1, 2], &[1, 1, 3]);
        assert!(matches!(
            max_unpool1d(&x, &ti(vec![0, 1], &[1, 1, 2]), 2, None, 0, None),
            Err(AutodiffError::Shape(_))
        ));
        assert!(max_unpool1d(&x, &ok, 0, None, 0, None).is_err());
        assert!(max_unpool1d(&x, &ok, 2, Some(0), 0, None).is_err());
        assert!(max_unpool1d(&x, &ok, 2, None, 0, Some(8)).is_err());
        assert!(max_unpool2d(&x, &ok, [2, 2], None, [0, 0], None).is_err());
        assert_eq!(tape.len(), before);
    }

    #[test]
    fn empty_batch_yields_empty_output() {
        let tape = Tape::new();
        let x = tape.var(&t(vec![], &[0, 2, 3]));
        let y = max_unpool1d(&x, &ti(vec![], &[0, 2, 3]), 2, None, 0, None).unwrap();
        assert_eq!(y.to_tensor().shape(), &[0, 2, 6]);
    }
}
