//! `grad::logsumexp_lane_masks`（イシュー #3792）の分類が `logsumexp_vjp` の出力パターンと
//! 一致することを確かめる単体テスト（`grad.rs` から `#[path]` で取り込む）。

use super::*;

#[test]
fn classification_matches_vjp_output_pattern() {
    let inf = f32::INFINITY;
    // 5 レーン（dim=Some(1)）: +inf 混在・+inf と -inf・全 -inf・有限・NaN。
    let data = vec![inf, 1.0, inf, -inf, -inf, -inf, 0.5, 1.5, f32::NAN, 1.0];
    let x = Tensor::new(data, &[5, 2]).unwrap();
    let g = Tensor::new(vec![1.0f32; 5], &[5]).unwrap();
    let masks = logsumexp_lane_masks(&x, Some(1));
    let vjp = dense_vec(&logsumexp_vjp(&x, Some(1), &g));
    #[allow(clippy::needless_range_loop)] // 3 つの並列配列（masks.*・vjp）を同じ添字で突き合わせる
    for i in 0..8 {
        if masks.posinf_elem[i] {
            assert!(vjp[i].is_finite() && vjp[i] > 0.0, "{i}");
            assert!((vjp[i] - 1.0 / masks.inf_count[i]).abs() < 1e-7);
        } else if masks.zero_elem[i] {
            assert_eq!(vjp[i], 0.0, "{i}");
        } else {
            assert!(!masks.guard[i], "{i}");
        }
    }
    // NaN レーンはどのマスクにも入らず、VJP は NaN。
    assert!(!masks.guard[8] && !masks.guard[9]);
    assert!(vjp[8].is_nan() && vjp[9].is_nan());
    assert!(masks.guard[0] && masks.guard[1] && masks.posinf_elem[0] && masks.zero_elem[1]);
    assert!(masks.zero_elem[4] && masks.zero_elem[5] && !masks.posinf_elem[4]);
}
