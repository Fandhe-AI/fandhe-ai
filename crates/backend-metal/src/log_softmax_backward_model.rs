//! `shaders/log_softmax_backward.metal`（`log_softmax` backward の
//! Metal カーネル。イシュー #1952・親 #1947）のホスト側逐語モデル。
//!
//! `crate::reduce_model`・`crate::scan_model` と同じ設計判断:
//! `crate::soft_f64` の binary64 ソフトウェアエミュレーション
//! （[`crate::soft_f64::widen_f32_bits`]／[`crate::soft_f64::
//! add_f64_bits`]／[`crate::soft_f64::sub_f64_bits`]／[`crate::
//! soft_f64::mul_f64_bits`]／[`crate::soft_f64::narrow_f64_bits`]）を
//! `log_softmax_backward.metal::lsb_f64_*` と同じ演算列で呼び出す
//! ことで、GPU 側カーネルが正しい binary64 逐次演算列を実行することを
//! Mac 実機に到達できない環境（本実装環境。Linux・CI）でも機械的に
//! 裏付ける（`objc2` 系 FFI に触れないため `cfg(target_os = "macos")`
//! を付けない）。
//!
//! # 数式・演算順序（ホスト参照実装 `fandhe_ai_autodiff::grad::
//! log_softmax_vjp_along` との対応）
//!
//! `dx = g − exp(y)·Σ_dim(g)`。ホスト参照実装は次の演算列で計算する
//! （`crates/autodiff/src/grad.rs::log_softmax_vjp_along` doc 参照）:
//!
//! 1. `Σ_dim(g)`: `dim` 軸を `0.0f64` から index 昇順に **`f32→f64`
//!    ウィデン → `f64` 加算**で逐次和する（`f32` のまま加算しない）。
//! 2. 各要素について `e = y[idx].exp()`（**`f32` 精度**で計算してから
//!    `f64` へウィデン）・`term = e_f64 * sum_f64`（`f64` 乗算・1 回の
//!    丸め）・`d = g[idx]_f64 - term`（`f64` 減算・1 回の丸め）を計算し、
//!    最後に 1 回だけ `d as f32` で `f32` へ downcast する。
//!
//! **bit 一致を主張する範囲**は (1) の縮約と (2) の `f64` 連鎖
//! （ウィデン → 乗算 → 減算 → narrow）のみであり、**`exp(y)` 自体の
//! 丸めは bit 一致を主張しない**（Metal `precise::exp` とホスト
//! `f32::exp` の丸めは規格上一致が保証されない）。よって実機の最終
//! 出力（`exp` の丸め差を含む）は REQ-2 統一複合判定（相対誤差 1e-3
//! 未満 または 絶対誤差 1e-5 未満）で検証し、bit 完全一致は
//! [`lane_sum_bits`]（縮約のみ）と [`apply_bits`]（`exp` の bit 表現を
//! 入力として与えたときの `f64` 連鎖のみ）の 2 関数単位でのみ主張する
//! （`docs/backend-metal-reduce-sum-design.md` 追補節参照）。
//!
//! Rust は `a - b*c` を FMA へ縮約しない（`f64::mul_add` を明示的に
//! 呼ばない限り乗算と加減算は別命令になる。`rustc` の既定動作）ため、
//! [`apply_bits`] の「乗算 → 減算」2 段はホスト参照実装の
//! `(y[idx].exp() as f64) * sum_acc` → `g[idx] as f64 - (..)` という
//! 2 段の演算と同一の丸め契約になる。
//!
//! # 結線について
//!
//! 本モジュールは [`crate::log_softmax_backward::
//! MetalLogSoftmaxBackward`] の正しさを裏付けるホスト側参照実装・
//! [`plan_log_softmax_backward`]（`ops.rs::MetalBackendOps::
//! log_softmax_backward` が起動前に呼ぶ事前検証ヘルパー）を提供する。

use crate::reduce_model::{self, ReduceAxisPlan, ReducePrepareError};
use crate::soft_f64::{add_f64_bits, mul_f64_bits, narrow_f64_bits, sub_f64_bits, widen_f32_bits};

/// 1 lane（`dim` 軸を除いた出力位置）分の `Σ_dim(g)` を、`0.0` から
/// index 昇順に「ウィデン → `f64` 加算」で逐次計算し、**narrow せず**
/// `f64` の bit 表現（`u64`）のまま返す（`log_softmax_backward.metal::
/// lsb_f64_lane_sum` の 1 lane 分ループ本体と 1 対 1 対応。narrow しない
/// 理由は `crate::reduce_model::sum_all_soft_f64` のチャンク部分和と同じ
/// —— 中間値を `f32` へ丸めてから次段へ渡すと二重丸めで契約が崩れる）。
pub fn lane_sum_bits(g_lane_bits: &[u32]) -> u64 {
    let mut acc: u64 = 0; // widen_f32_bits(0.0) == 0（+0.0）。
    for &bits in g_lane_bits {
        acc = add_f64_bits(acc, widen_f32_bits(bits));
    }
    acc
}

/// 1 要素分の `dx = g − exp(y)·sum` を、`g`／`exp(y)` の bit 表現と
/// `sum`（[`lane_sum_bits`] の戻り値。narrow していない `f64` bit）から
/// 計算する（`log_softmax_backward.metal::lsb_f64_apply` の 1 要素分
/// 本体と 1 対 1 対応）。`exp(y)` は呼び出し元が計算済みの値を渡す
/// 契約（モジュール doc「bit 一致を主張する範囲」参照。本関数自体は
/// `exp` を計算しない）。
pub fn apply_bits(g_bits: u32, exp_bits: u32, sum_bits: u64) -> u32 {
    let g64 = widen_f32_bits(g_bits);
    let e64 = widen_f32_bits(exp_bits);
    let term = mul_f64_bits(e64, sum_bits);
    let d = sub_f64_bits(g64, term);
    narrow_f64_bits(d)
}

/// `y`（forward 記録値 `log_softmax(x, dim)`）・`g`（上流勾配。`y` と
/// 同一 shape）・`shape`／`dim` から `dx` 全体を計算する（テスト専用の
/// Rust 側参照実装。本体経路は `crate::log_softmax_backward::
/// MetalLogSoftmaxBackward::run_f32` が GPU 側で直接計算するため、
/// 本関数は突合用。`crate::reduce_model::sum_axis_soft_f64` と同型）。
///
/// `exp(y)` は Rust の `f32::exp`（ホスト `libm` 実装）で計算する。
/// これは GPU 側 `precise::exp` と bit 一致するとは限らないため、本
/// 関数を経由した突合は REQ-2 統一複合判定（bit 完全一致ではない）で
/// 行う契約（モジュール doc 参照）。
#[cfg(test)]
pub fn log_softmax_backward_soft_f64(
    y: &[f32],
    g: &[f32],
    shape: &[usize],
    dim: usize,
) -> Vec<f32> {
    let outer: usize = shape[..dim].iter().product();
    let axis_len = shape[dim];
    let inner: usize = shape[dim + 1..].iter().product();
    let mut out = vec![0f32; y.len()];
    for o in 0..outer {
        for i in 0..inner {
            let lane_bits: Vec<u32> = (0..axis_len)
                .map(|a| g[(o * axis_len + a) * inner + i].to_bits())
                .collect();
            let sum_bits = lane_sum_bits(&lane_bits);
            for a in 0..axis_len {
                let idx = (o * axis_len + a) * inner + i;
                let exp_bits = y[idx].exp().to_bits();
                out[idx] = f32::from_bits(apply_bits(g[idx].to_bits(), exp_bits, sum_bits));
            }
        }
    }
    out
}

/// `log_softmax_backward` の起動計画。`crate::reduce_model::
/// plan_reduce_axis` をそのまま再利用して `outer`／`axis_len`／
/// `inner`／`lanes` を求め（`reduce_sum_axis_f32` と同じ添字規約の
/// ため）、加えてカーネル `uint` 引数として渡す `numel`
/// （`lanes * axis_len`）が [`crate::reduce_model::
/// REDUCE_KERNEL_ARG_LIMIT`] に収まることを検証する（`sum` 系は
/// `numel` を直接カーネル引数に渡さないため `plan_reduce_axis` 自体は
/// この検査を含まない。本関数が追加で行う）。
pub fn plan_log_softmax_backward(
    shape: &[usize],
    dim: usize,
) -> Result<ReduceAxisPlan, ReducePrepareError> {
    let plan = reduce_model::plan_reduce_axis(shape, dim)?;
    let numel =
        plan.lanes
            .checked_mul(plan.axis_len)
            .ok_or(ReducePrepareError::SizeLimitExceeded {
                what: "numel (lanes * axis_len overflow)",
                value: usize::MAX,
                limit: reduce_model::REDUCE_KERNEL_ARG_LIMIT,
            })?;
    if numel > reduce_model::REDUCE_KERNEL_ARG_LIMIT {
        return Err(ReducePrepareError::SizeLimitExceeded {
            what: "numel",
            value: numel,
            limit: reduce_model::REDUCE_KERNEL_ARG_LIMIT,
        });
    }
    Ok(plan)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `Σ_dim(g)` の縮約が `f64` 参照実装（`f64` へ都度ウィデンしてから
    /// 加算する素朴な逐次和）と bit 完全一致することを確認する
    /// （`lane_sum_bits` 単体・narrow 前）。
    #[test]
    fn lane_sum_bits_matches_naive_f64_accumulation() {
        let lanes: [&[f32]; 4] = [
            &[1.0, 2.0, 3.0, 4.0],
            &[-1.0, 1.0, -1.0, 1.0],
            &[0.0],
            &[1e20, -1e20, 1.0, 0.0],
        ];
        for lane in lanes {
            let bits: Vec<u32> = lane.iter().map(|v| v.to_bits()).collect();
            let actual = lane_sum_bits(&bits);
            let mut expected: f64 = 0.0;
            for &v in lane {
                expected += v as f64;
            }
            assert_eq!(actual, expected.to_bits());
        }
    }

    /// 相殺列（`Neumaier`/`Kahan` 単純実装では丸め誤差が残る列）でも
    /// `f64` 逐次和と一致することを確認する（`.claude/rules/
    /// coding-rust.md` の「f32 のみの補償和は相殺列で一致しない」
    /// 指摘への回帰）。
    #[test]
    fn lane_sum_bits_matches_cancelling_sequence() {
        let lane: [f32; 5] = [
            (1u64 << 48) as f32,
            (1u64 << 24) as f32,
            1.0,
            -((1u64 << 48) as f32),
            -((1u64 << 24) as f32),
        ];
        let bits: Vec<u32> = lane.iter().map(|v| v.to_bits()).collect();
        let actual = lane_sum_bits(&bits);
        let mut expected: f64 = 0.0;
        for &v in &lane {
            expected += v as f64;
        }
        assert_eq!(actual, expected.to_bits());
    }

    /// `apply_bits` が `g_f64 - e_f64 * sum_f64` の `f64` 参照演算
    /// （乗算 → 減算の非 FMA 2 段）と bit 完全一致することを確認する。
    #[test]
    fn apply_bits_matches_naive_f64_mul_sub() {
        let cases: [(f32, f32, f64); 5] = [
            (0.1, 0.9, 1.0),
            (-3.0, 0.5, -2.0),
            (0.0, 1.0, 0.0),
            (1e10, 1e-3, 1e5),
            (-1e10, 1e-3, -1e5),
        ];
        for (g, e, sum) in cases {
            let sum_bits = sum.to_bits();
            let actual = apply_bits(g.to_bits(), e.to_bits(), sum_bits);
            let expected = (g as f64 - (e as f64) * sum) as f32;
            assert_eq!(actual, expected.to_bits());
        }
    }

    /// 大きな上流勾配（有限入力）でも overflow せず `f64` 連鎖のまま
    /// 保持することを確認する（ホスト参照実装
    /// `log_softmax_vjp_along_large_upstream_grad_does_not_overflow`
    /// の趣旨と同じ回帰ケース。`apply_bits` 自体は `exp(y)` の計算を
    /// 行わないため、丸め誤差の影響を受けない厳密値
    /// `exp_bits = 0.5f32`（2 のべき乗のため `0.5 * sum` は丸め誤差
    /// なしで正確に半分になる）を直接与え、`term` との厳密な相殺で
    /// `d = g - 0.5*sum = 2e38 - 2e38 = 0` になることを確認する
    /// （`f32` のまま `g - exp(y)*sum` を計算する実装では、有限入力
    /// でも `exp(y)*sum` の乗算自体が `f32` の最大値を超え `inf` に
    /// なり `d` が `-inf` になる overflow 回帰）。
    #[test]
    fn apply_bits_large_upstream_grad_does_not_overflow() {
        let exp_bits = 0.5f32.to_bits(); // 2 のべき乗なので乗算が厳密。
        let g: f32 = 2e38;
        let g_bits = g.to_bits();
        // `sum` は 2 要素とも `g` のケースを [`lane_sum_bits`] と同じ
        // 演算列（`widen(g_bits)` を 2 回加算）で求める——`g` は
        // f32 の最大値付近のため表現精度が粗く（絶対誤差 ~2^104 ≈
        // 2e31）、独立に計算した数学的 `2.0*2e38f64` とは一致しない
        // （widen 後の実際のビット値からズレる）。
        let sum_bits = lane_sum_bits(&[g_bits, g_bits]);
        let out_bits = apply_bits(g_bits, exp_bits, sum_bits);
        let out = f32::from_bits(out_bits);
        assert!(
            out.is_finite(),
            "dx={out} は有限であるべき（overflow 回帰）"
        );
        assert_eq!(out, 0.0, "dx={out}（厳密に 0 になるはず）");
    }

    /// `y=0` 行（`exp(0)==1.0` は厳密丸め）での `log_softmax_backward_
    /// soft_f64` が `f64` 参照実装（`grad::log_softmax_vjp_along` と
    /// 同一式）と bit 完全一致することを確認する（`exp` の丸め差の
    /// 影響を受けない行での bit 一致主張。モジュール doc 参照）。
    #[test]
    fn log_softmax_backward_soft_f64_matches_f64_reference_when_y_is_zero() {
        let shape = [2, 3];
        let y = vec![0.0f32; 6];
        let g = vec![1.0, -2.0, 3.0, -4.0, 5.0, -6.0];
        let actual = log_softmax_backward_soft_f64(&y, &g, &shape, 1);

        let outer = shape[0];
        let axis_len = shape[1];
        let mut expected = vec![0f32; 6];
        for o in 0..outer {
            let mut sum_acc: f64 = 0.0;
            for a in 0..axis_len {
                sum_acc += g[o * axis_len + a] as f64;
            }
            for a in 0..axis_len {
                let idx = o * axis_len + a;
                let d = g[idx] as f64 - (y[idx].exp() as f64) * sum_acc;
                expected[idx] = d as f32;
            }
        }
        assert_eq!(actual, expected);
    }

    /// `y=-inf` 行（`exp(-inf)==0.0` は厳密）でも `dx == g` になる
    /// （`sum` との積が厳密ゼロのため）ことを確認する。
    #[test]
    fn log_softmax_backward_soft_f64_matches_when_y_is_neg_infinity() {
        let shape = [1, 3];
        let y = vec![f32::NEG_INFINITY; 3];
        let g = vec![1.5, -2.5, 3.5];
        let actual = log_softmax_backward_soft_f64(&y, &g, &shape, 1);
        assert_eq!(actual, g);
    }

    /// `plan_log_softmax_backward` が `numel` 超過を検出することを
    /// 確認する（`plan_reduce_axis` 自体は `numel` 検査を含まないため、
    /// 本関数が追加で行う検査の直接検証）。
    #[test]
    fn plan_log_softmax_backward_rejects_numel_overflow() {
        // axis_len=100000・lanes=50000 はいずれも単独では
        // `REDUCE_KERNEL_ARG_LIMIT`（`u32::MAX`）未満だが、積
        // `numel = lanes * axis_len = 5_000_000_000` が超過する
        // （`plan_reduce_axis` 自体の個別フィールド検査ではなく
        // 本関数が追加する `numel` 検査を直接検証する）。
        let shape = [100_000usize, 50_000usize];
        let result = plan_log_softmax_backward(&shape, 0);
        assert!(matches!(
            result,
            Err(ReducePrepareError::SizeLimitExceeded { what: "numel", .. })
        ));
    }

    /// 通常形状では `plan_log_softmax_backward` が成功し
    /// `plan_reduce_axis` と同じ `outer`／`axis_len`／`inner`／`lanes`
    /// を返すことを確認する。
    #[test]
    fn plan_log_softmax_backward_normal_shape() {
        let shape = [2, 3, 4];
        let plan = plan_log_softmax_backward(&shape, 1).expect("valid shape");
        assert_eq!(plan.outer, 2);
        assert_eq!(plan.axis_len, 3);
        assert_eq!(plan.inner, 4);
        assert_eq!(plan.lanes, 8);
    }
}
