//! ONNX `LayerNormalization`（opset-17）オペ（TASK-7.3d・#85）。
//!
//! `axis`（ONNX 負軸表記対応）以降の trailing 次元群（`X.shape[axis..]`）を
//! 正規化集合として、集合ごとに平均・分散（母分散。除数は集合の要素数
//! `inner_size`。ddof=0）を計算し `Y = (X - mean) / sqrt(var + epsilon) * Scale + B`
//! を返す。`Scale`／`B`（`B` は省略可）は正規化集合の shape
//! （`X.shape[axis..]`）へ [`Tensor::broadcast_to`] でブロードキャストする
//! （`gemm.rs` の `C` 引数と同じ委譲パターン）。
//!
//! ONNX 仕様が定義する任意出力 `Mean`／`InvStdDev`（学習時の逆伝播用）と
//! `stash_type` 属性（本クレートは `f32` 単一 dtype 実装のため実質無効）は
//! 本関数のスコープ外（イシュー #85 計画のスコープ外節・#274 へ追記候補）。
//! ONNX proto デコード・グラフ実行エンジンとの結線は本モジュールの責務外
//! （`ops/mod.rs` ドキュメンテーションコメント参照。#78/#274 待ち）。
//!
//! codegen 経路の `sanitize_ident`・絶対パス埋め込み要件（TASK-7.3 spec）は
//! codegen 方式採用時のみ該当し、本関数は純粋関数（インタープリタ向け）
//! であり該当しない。

use fandhe_ai_tensor_core::Tensor;

use super::error::OpError;
use super::normalize_axis;

/// `LayerNormalization` の属性。ONNX 仕様の既定値（`axis = -1`・`epsilon = 1e-5`）を
/// [`Default`] に反映する。
#[derive(Debug, Clone, Copy)]
pub struct LayerNormAttrs {
    pub axis: i64,
    pub epsilon: f32,
}

impl Default for LayerNormAttrs {
    fn default() -> Self {
        LayerNormAttrs {
            axis: -1,
            epsilon: 1e-5,
        }
    }
}

/// `LayerNormalization(x, scale, bias?)` を計算する。
///
/// - `axis` は `x` の rank に対して正規化する（範囲外は [`OpError::AxisOutOfRange`]）。
///   正規化後の軸が範囲内でも正規化集合（`x.shape()[axis..]`）の要素数積が 0
///   （例: `shape=[2,0], axis=1`）の場合は分散の除数 0 割りを避けるため
///   [`OpError::EmptyNormalizedSet`] を返す。
/// - `epsilon` は非有限値（`NaN`／`inf`）を [`OpError::InvalidEpsilon`] で拒否する。
///   `epsilon` はモデル属性（外部入力）であり、`Div`/`Sqrt` が実行時データに対して
///   採る IEEE 754 透過方針（`arith.rs`）とは異なり、非有限な属性値は分散計算全体を
///   静かに `NaN`／`inf` へ汚染するため事前検査で弾く（OWASP A03。`security.md`）。
///   負値・0 は ONNX 仕様上明示的に禁止されていないため許容し、透過する
///   （通常入力では `var + epsilon >= 0` が成り立つが、異常な負 `epsilon` で
///   `sqrt` が `NaN` を返す場合は `Div`/`Sqrt` と同じ IEEE 754 透過方針に従う）。
/// - `scale`／`bias` は正規化集合の shape（`x.shape()[axis..]`）へ
///   [`Tensor::broadcast_to`] でブロードキャストする（不可の場合 [`OpError::Shape`]）。
///   `bias` を省略した場合はバイアス項なし（`+ 0`）として扱う。
pub fn layer_normalization(
    x: &Tensor<f32>,
    scale: &Tensor<f32>,
    bias: Option<&Tensor<f32>>,
    attrs: &LayerNormAttrs,
) -> Result<Tensor<f32>, OpError> {
    if !attrs.epsilon.is_finite() {
        return Err(OpError::InvalidEpsilon {
            op: "LayerNormalization",
            epsilon: attrs.epsilon,
        });
    }

    let rank = x.rank();
    let axis = normalize_axis(attrs.axis, rank).ok_or(OpError::AxisOutOfRange {
        op: "LayerNormalization",
        axis: attrs.axis,
        rank,
    })?;

    let normalized_shape = &x.shape()[axis..];
    let inner_size: usize = normalized_shape.iter().product();
    let outer_size: usize = x.shape()[..axis].iter().product();

    // `inner_size == 0`（正規化集合が空。例: shape=[2,0], axis=1）は ONNX 仕様上未定義に
    // 近く、分散の除数 0 割りで意味のない出力（NaN）を静かに生成しうる。0 要素テンソルは
    // shape 経由で `Tensor::new`/`zeros` の時点で許容されうるため、ここで明示的に拒否する。
    // `axis` 自体は `[0, rank)` の範囲内であり [`OpError::AxisOutOfRange`] とは原因が
    // 異なるため、専用 variant（[`OpError::EmptyNormalizedSet`]）で区別する。
    if inner_size == 0 {
        return Err(OpError::EmptyNormalizedSet {
            op: "LayerNormalization",
            axis,
        });
    }

    let scale_b = scale.broadcast_to(normalized_shape)?.contiguous();
    let scale_slice = scale_b
        .as_slice()
        .ok_or(OpError::NonContiguousInternal("LayerNormalization(Scale)"))?;

    let bias_b = match bias {
        Some(b) => Some(b.broadcast_to(normalized_shape)?.contiguous()),
        None => None,
    };
    let bias_slice = match &bias_b {
        Some(b) => Some(
            b.as_slice()
                .ok_or(OpError::NonContiguousInternal("LayerNormalization(B)"))?,
        ),
        None => None,
    };

    let xc = x.contiguous();
    let x_slice = xc
        .as_slice()
        .ok_or(OpError::NonContiguousInternal("LayerNormalization(X)"))?;

    let mut out = vec![0f32; outer_size * inner_size];
    let inv_n = 1.0 / inner_size as f32;
    for o in 0..outer_size {
        let block = &x_slice[o * inner_size..(o + 1) * inner_size];

        // 平均: 単純総和（`Gemm` の内積累積と異なり乗算を伴わないため `mul_add` 対象外）。
        let mean = block.iter().sum::<f32>() * inv_n;

        // 母分散（除数 = inner_size・ddof=0）。二乗差の累積は `Gemm` の内積累積と同じ
        // 乗算加算パターンのため、丸め方針統一（FMA 契約。`coding-rust.md`）に従い
        // `f32::mul_add` を用いる。
        let mut sq_acc = 0f32;
        for &v in block {
            let diff = v - mean;
            sq_acc = diff.mul_add(diff, sq_acc);
        }
        let var = sq_acc * inv_n;
        let inv_std = 1.0 / (var + attrs.epsilon).sqrt();

        let out_block = &mut out[o * inner_size..(o + 1) * inner_size];
        for i in 0..inner_size {
            let normalized = (block[i] - mean) * inv_std;
            // Scale 乗算・Bias 加算も分散計算の二乗差累積（上記）と同じ乗算加算
            // パターンのため、FMA 契約統一方針（`coding-rust.md`）に従い
            // `mul_add` で統一する（レビュー指摘 #85: ファイル内の一貫性）。
            out_block[i] = match bias_slice {
                Some(bs) => normalized.mul_add(scale_slice[i], bs[i]),
                None => normalized * scale_slice[i],
            };
        }
    }

    Tensor::new(out, xc.shape()).map_err(OpError::from)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// テスト専用のスカラー参照実装。本体実装（`mul_add` 累積・ブロードキャスト経路）
    /// とは独立した素朴なループで平均・分散・正規化を再計算し、実装バグの見落としを
    /// 防ぐ（`coding-rust.md` の許容誤差方針・イシュー #85 計画 Step 2）。
    fn reference_layer_norm(
        x: &[f32],
        normalized_shape: &[usize],
        outer: usize,
        scale: &[f32],
        bias: Option<&[f32]>,
        epsilon: f32,
    ) -> Vec<f32> {
        let inner: usize = normalized_shape.iter().product();
        let mut out = vec![0f32; outer * inner];
        for o in 0..outer {
            let block = &x[o * inner..(o + 1) * inner];
            let mean: f32 = block.iter().sum::<f32>() / inner as f32;
            let var: f32 =
                block.iter().map(|&v| (v - mean) * (v - mean)).sum::<f32>() / inner as f32;
            let inv_std = 1.0 / (var + epsilon).sqrt();
            for i in 0..inner {
                let n = (block[i] - mean) * inv_std * scale[i];
                out[o * inner + i] = match bias {
                    Some(b) => n + b[i],
                    None => n,
                };
            }
        }
        out
    }

    fn assert_close(a: f32, b: f32) {
        let tol = 1e-5_f32.max(b.abs() * 1e-3);
        assert!(
            (a - b).abs() <= tol,
            "assert_close failed: a={a}, b={b}, diff={}",
            (a - b).abs()
        );
    }

    #[test]
    fn basic_2d_axis_minus_one() {
        // 行 [1,2,3,4]: mean=2.5, var=1.25（母分散）。
        let x = Tensor::<f32>::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], &[2, 4]).unwrap();
        let scale = Tensor::<f32>::new(vec![1.0, 1.0, 1.0, 1.0], &[4]).unwrap();
        let attrs = LayerNormAttrs::default();
        let y = layer_normalization(&x, &scale, None, &attrs).unwrap();
        assert_eq!(y.shape(), &[2, 4]);

        let reference = reference_layer_norm(
            &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            &[4],
            2,
            &[1.0, 1.0, 1.0, 1.0],
            None,
            1e-5,
        );
        for i in 0..2 {
            for j in 0..4 {
                assert_close(y.get(&[i, j]).unwrap(), reference[i * 4 + j]);
            }
        }
        // 手計算の直接検証（mean=2.5, var=1.25, epsilon=1e-5 は無視できるほど小さい）。
        let inv_std = 1.0 / 1.25_f32.sqrt();
        assert_close(y.get(&[0, 0]).unwrap(), (1.0 - 2.5) * inv_std);
        assert_close(y.get(&[0, 3]).unwrap(), (4.0 - 2.5) * inv_std);
    }

    #[test]
    fn positive_and_negative_axis_are_equivalent() {
        let x = Tensor::<f32>::new((1..=8).map(|v| v as f32).collect(), &[2, 4]).unwrap();
        let scale = Tensor::<f32>::new(vec![1.0, 1.0, 1.0, 1.0], &[4]).unwrap();
        let y_pos = layer_normalization(
            &x,
            &scale,
            None,
            &LayerNormAttrs {
                axis: 1,
                epsilon: 1e-5,
            },
        )
        .unwrap();
        let y_neg = layer_normalization(
            &x,
            &scale,
            None,
            &LayerNormAttrs {
                axis: -1,
                epsilon: 1e-5,
            },
        )
        .unwrap();
        for i in 0..2 {
            for j in 0..4 {
                assert_eq!(y_pos.get(&[i, j]).unwrap(), y_neg.get(&[i, j]).unwrap());
            }
        }
    }

    #[test]
    fn rank3_multi_dim_normalized_set_axis1() {
        // [2,3,4] を axis=1 で正規化 -> 正規化集合は shape[1..] = [3,4]（2 次元にまたがる）。
        // 末尾 1 次元のみ正規化する誤実装（axis=-1 相当の実装を誤って axis=1 に流用した場合）
        // はこのテストで shape 不一致（scale の shape [3,4] を [4] とみなして broadcast_to
        // が失敗する）か数値不一致で検出される。
        let data: Vec<f32> = (0..24).map(|v| v as f32).collect();
        let x = Tensor::<f32>::new(data.clone(), &[2, 3, 4]).unwrap();
        let scale = Tensor::<f32>::new(vec![1.0; 12], &[3, 4]).unwrap();
        let bias = Tensor::<f32>::new(vec![0.0; 12], &[3, 4]).unwrap();
        let attrs = LayerNormAttrs {
            axis: 1,
            epsilon: 1e-5,
        };
        let y = layer_normalization(&x, &scale, Some(&bias), &attrs).unwrap();
        assert_eq!(y.shape(), &[2, 3, 4]);

        let reference = reference_layer_norm(&data, &[3, 4], 2, &[1.0; 12], Some(&[0.0; 12]), 1e-5);
        let mut idx = 0;
        for i in 0..2 {
            for j in 0..3 {
                for k in 0..4 {
                    assert_close(y.get(&[i, j, k]).unwrap(), reference[idx]);
                    idx += 1;
                }
            }
        }
    }

    #[test]
    fn scale_and_bias_are_applied() {
        // scale=2, bias=1 の単純値検証。x=[1,2,3,4] は axis=-1 で正規化。
        let x = Tensor::<f32>::new(vec![1.0, 2.0, 3.0, 4.0], &[1, 4]).unwrap();
        let scale = Tensor::<f32>::new(vec![2.0, 2.0, 2.0, 2.0], &[4]).unwrap();
        let bias = Tensor::<f32>::new(vec![1.0, 1.0, 1.0, 1.0], &[4]).unwrap();
        let attrs = LayerNormAttrs::default();
        let y = layer_normalization(&x, &scale, Some(&bias), &attrs).unwrap();
        let inv_std = 1.0 / 1.25_f32.sqrt();
        assert_close(y.get(&[0, 0]).unwrap(), (1.0 - 2.5) * inv_std * 2.0 + 1.0);
        assert_close(y.get(&[0, 3]).unwrap(), (4.0 - 2.5) * inv_std * 2.0 + 1.0);
    }

    #[test]
    fn bias_none_means_zero() {
        let x = Tensor::<f32>::new(vec![1.0, 2.0, 3.0, 4.0], &[1, 4]).unwrap();
        let scale = Tensor::<f32>::new(vec![1.0, 1.0, 1.0, 1.0], &[4]).unwrap();
        let y_none = layer_normalization(&x, &scale, None, &LayerNormAttrs::default()).unwrap();
        let zero_bias = Tensor::<f32>::new(vec![0.0, 0.0, 0.0, 0.0], &[4]).unwrap();
        let y_zero =
            layer_normalization(&x, &scale, Some(&zero_bias), &LayerNormAttrs::default()).unwrap();
        for j in 0..4 {
            assert_eq!(y_none.get(&[0, j]).unwrap(), y_zero.get(&[0, j]).unwrap());
        }
    }

    #[test]
    fn axis_out_of_range_rejected() {
        let x = Tensor::<f32>::zeros(&[2, 3]).unwrap();
        let scale = Tensor::<f32>::zeros(&[3]).unwrap();
        let attrs = LayerNormAttrs {
            axis: 5,
            epsilon: 1e-5,
        };
        let err = layer_normalization(&x, &scale, None, &attrs).unwrap_err();
        assert!(matches!(err, OpError::AxisOutOfRange { axis: 5, .. }));
    }

    #[test]
    fn empty_normalized_set_rejected() {
        // shape=[2,0], axis=1 -> 正規化集合 shape[1..] = [0] は要素数積 0（inner_size=0）。
        // axis 自体は rank=2 に対し範囲内のため AxisOutOfRange ではなく専用 variant を返す。
        let x = Tensor::<f32>::zeros(&[2, 0]).unwrap();
        let scale = Tensor::<f32>::zeros(&[0]).unwrap();
        let attrs = LayerNormAttrs {
            axis: 1,
            epsilon: 1e-5,
        };
        let err = layer_normalization(&x, &scale, None, &attrs).unwrap_err();
        assert!(matches!(err, OpError::EmptyNormalizedSet { axis: 1, .. }));
    }

    #[test]
    fn scale_shape_mismatch_rejected() {
        let x = Tensor::<f32>::zeros(&[2, 4]).unwrap();
        // 正規化集合 shape は [4] だが scale は [3] でブロードキャスト不可。
        let scale = Tensor::<f32>::zeros(&[3]).unwrap();
        let err = layer_normalization(&x, &scale, None, &LayerNormAttrs::default()).unwrap_err();
        assert!(matches!(err, OpError::Shape(_)));
    }

    #[test]
    fn non_finite_epsilon_rejected() {
        let x = Tensor::<f32>::zeros(&[2, 4]).unwrap();
        let scale = Tensor::<f32>::zeros(&[4]).unwrap();
        for bad in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            let attrs = LayerNormAttrs {
                axis: -1,
                epsilon: bad,
            };
            let err = layer_normalization(&x, &scale, None, &attrs).unwrap_err();
            assert!(matches!(err, OpError::InvalidEpsilon { .. }));
        }
    }

    #[test]
    fn outer_size_zero_produces_empty_output() {
        // shape=[0,4], axis=1 -> outer_size=0（正規化対象ブロックが 0 件）・
        // inner_size=4（正規化集合自体は空でないため EmptyNormalizedSet 対象外）。
        // ループが 1 度も実行されないため 0 除算は発生せず、shape=[0,4] の空
        // テンソルがそのまま返ることを確認する（レビュー指摘 #85: エッジケース）。
        let x = Tensor::<f32>::zeros(&[0, 4]).unwrap();
        let scale = Tensor::<f32>::new(vec![1.0, 1.0, 1.0, 1.0], &[4]).unwrap();
        let attrs = LayerNormAttrs {
            axis: 1,
            epsilon: 1e-5,
        };
        let y = layer_normalization(&x, &scale, None, &attrs).unwrap();
        assert_eq!(y.shape(), &[0, 4]);
    }

    #[test]
    fn scale_bias_degenerate_shape_broadcasts_across_outer_normalized_dim() {
        // 正規化集合 shape[axis..] = [3,4] に対し Scale/Bias を縮退形状 [4]
        // （先頭次元 3 を暗黙拡張）で与えるケース。`Tensor::broadcast_to` の
        // 単方向ブロードキャストで [3,4] へ拡張されることを確認する
        // （レビュー指摘 #85: 縮退形状ブロードキャストの直接カバレッジ）。
        let data: Vec<f32> = (0..24).map(|v| v as f32).collect();
        let x = Tensor::<f32>::new(data.clone(), &[2, 3, 4]).unwrap();
        // rank 縮退（先頭次元の暗黙拡張）ケース。
        let scale_rank_deg = Tensor::<f32>::new(vec![2.0, 2.0, 2.0, 2.0], &[4]).unwrap();
        let bias_rank_deg = Tensor::<f32>::new(vec![1.0, 1.0, 1.0, 1.0], &[4]).unwrap();
        // size-1 次元拡張（同 rank・先頭次元が明示的に 1）ケース。上記とは
        // `Tensor::broadcast_to` 内の別経路（stride-0 拡張 vs 右詰め拡張）を通る。
        let scale_size1_deg = Tensor::<f32>::new(vec![2.0, 2.0, 2.0, 2.0], &[1, 4]).unwrap();
        let bias_size1_deg = Tensor::<f32>::new(vec![1.0, 1.0, 1.0, 1.0], &[1, 4]).unwrap();
        let scale_full = Tensor::<f32>::new(vec![2.0; 12], &[3, 4]).unwrap();
        let bias_full = Tensor::<f32>::new(vec![1.0; 12], &[3, 4]).unwrap();
        let attrs = LayerNormAttrs {
            axis: 1,
            epsilon: 1e-5,
        };
        let y_rank_deg =
            layer_normalization(&x, &scale_rank_deg, Some(&bias_rank_deg), &attrs).unwrap();
        let y_size1_deg =
            layer_normalization(&x, &scale_size1_deg, Some(&bias_size1_deg), &attrs).unwrap();
        let y_full = layer_normalization(&x, &scale_full, Some(&bias_full), &attrs).unwrap();
        for i in 0..2 {
            for j in 0..3 {
                for k in 0..4 {
                    let full = y_full.get(&[i, j, k]).unwrap();
                    assert_eq!(y_rank_deg.get(&[i, j, k]).unwrap(), full);
                    assert_eq!(y_size1_deg.get(&[i, j, k]).unwrap(), full);
                }
            }
        }
    }

    #[test]
    fn epsilon_default_matches_onnx_spec() {
        assert_eq!(LayerNormAttrs::default().epsilon, 1e-5);
        assert_eq!(LayerNormAttrs::default().axis, -1);
    }
}
