//! `crates/backend-metal/src/shaders/gemm.metal::gemm_bias_grad_reduce_f32`
//! の `m >= 2` 蓄積アルゴリズム（`bias_scale_sum_add`／`bias_pow2_floor`／
//! `bias_kahan_add`）を Rust へ逐語的に移植したホスト参照モデル（イシュー
//! #1566・PR #1659→#1665 取り込み後の codex-review 追加指摘 P1 是正）。
//!
//! Metal 実機なしでは MSL カーネル自体を実行できないため、Linux で常時
//! 実行可能なこの Rust 版で数値契約（中間 overflow 回避・2 のべき乗
//! `scale` による相殺精度）を検証する。関数本体は `gemm.metal` の対応
//! 関数と 1 対 1 対応させてあり、`gemm.metal` を変更した場合は本ファイル
//! も追従させること（構造的な二重管理だが、MSL を直接 Rust から呼べない
//! 制約上の妥協。`docs/backend-metal-command-batching-design.md` §10.10
//! 参照）。
//!
//! # 検証対象（codex-review 指摘の再現・是正確認）
//!
//! 当初実装（`scale` を「列内の最大絶対値そのもの」とする版）は、
//! `x / scale`・`acc *= ratio` が一般の実数比を経由するため丸めを伴い、
//! `[1e8, -100000008.0, 8.0]`（真値 `0`）のような相殺入力で REQ-2
//! （相対誤差 1e-3 未満 または 絶対誤差 1e-5 未満）を満たさない誤差
//! （実測 約 2.04）が生じた。本ファイルの
//! [`preserves_cancelling_contribution_with_rounding_prone_values`] が
//! この再現・是正確認を直接担う。

use bench_harness::rng::Xorshift64Star;

/// `gemm.metal::bias_kahan_add` の逐語移植（Neumaier 改良版 Kahan 補償和の
/// 1 ステップ）。
fn bias_kahan_add(sum: &mut f32, comp: &mut f32, value: f32) {
    let t = *sum + value;
    if sum.abs() >= value.abs() {
        *comp += (*sum - t) + value;
    } else {
        *comp += (value - t) + *sum;
    }
    *sum = t;
}

/// `gemm.metal::bias_pow2_floor` の逐語移植: `ax`（`> 0` の有限値を仮定）
/// 以下の最大の 2 のべき乗を、ビットパターンの仮数部（下位 23 bit）を
/// ゼロクリアするだけで exact に求める（`ceil` ではなく `floor` を使う
/// 理由は `gemm.metal` 側コメント参照。`f32::MAX` 付近でも overflow
/// しない）。
fn bias_pow2_floor(ax: f32) -> f32 {
    let bits = ax.to_bits() & 0xFF80_0000u32;
    f32::from_bits(bits)
}

/// `gemm.metal::bias_scale_sum_add` の逐語移植。
fn bias_scale_sum_add(scale: &mut f32, acc: &mut f32, comp: &mut f32, x: f32) {
    if x.is_nan() || acc.is_nan() || scale.is_nan() {
        *scale = 1.0;
        *acc = f32::NAN;
        *comp = 0.0;
        return;
    }
    let ax = x.abs();
    if ax.is_infinite() {
        let sign = if x > 0.0 { 1.0f32 } else { -1.0f32 };
        if scale.is_infinite() {
            if *acc != sign {
                *acc = f32::NAN;
                *comp = 0.0;
            }
            return;
        }
        *scale = f32::INFINITY;
        *acc = sign;
        *comp = 0.0;
        return;
    }
    if scale.is_infinite() {
        return;
    }
    if ax > *scale {
        let new_scale = bias_pow2_floor(ax);
        if *scale > 0.0 {
            let ratio = *scale / new_scale;
            *acc *= ratio;
            *comp *= ratio;
        }
        *scale = new_scale;
        bias_kahan_add(acc, comp, x / *scale);
    } else if *scale > 0.0 {
        bias_kahan_add(acc, comp, x / *scale);
    }
}

/// `gemm.metal::gemm_bias_grad_reduce_f32` の `m >= 2` 経路（1 列分の
/// 縮約ループ本体）の逐語移植。`m == 1` の直接コピー特殊扱いはこの
/// ループより手前で分岐する別経路のため、本関数の対象外
/// （`reduce_bias_grad_rows_host`／`reduce_bias_grad_rows` と同型）。
fn bias_scale_sum_reduce(xs: &[f32]) -> f32 {
    let mut scale = 0.0f32;
    let mut acc = 0.0f32;
    let mut comp = 0.0f32;
    for &x in xs {
        bias_scale_sum_add(&mut scale, &mut acc, &mut comp, x);
    }
    scale * (acc + comp)
}

/// (a) codex-review 指摘の直接再現・是正確認: 当初実装（`scale` が
/// 任意の最大絶対値）では丸めにより REQ-2 を割り込んでいた相殺入力。
/// 2 のべき乗 `scale` 版では `x / scale`・`ratio` 双方が exact になる
/// ため、真値 `0` と REQ-2 統一複合判定で一致するはず。
#[test]
fn preserves_cancelling_contribution_with_rounding_prone_values() {
    let xs = [1.0e8f32, -100000008.0, 8.0];
    let got = bias_scale_sum_reduce(&xs);
    let f64_truth: f64 = xs.iter().map(|&v| f64::from(v)).sum();
    assert!(
        f64_truth == 0.0,
        "test fixture: 真値が 0 であることの前提確認（f64_truth={f64_truth}）"
    );
    fandhe_ai_backend_cpu::assert_parity(
        "bias_scale_sum_reduce は相殺入力 [1e8, -100000008.0, 8.0] で真値 0 と \
         REQ-2 複合判定で一致するはず（codex-review 指摘の是正確認）",
        &[got],
        &[0.0f32],
    );
}

/// (b) 当初の P1 指摘そのもの: 有限入力の中間 overflow 再現テスト。
/// `f32::MAX + f32::MAX` は素朴な逐次和では `+inf` へ overflow するが、
/// scale 方式（2 のべき乗版）は overflow せず有限のまま真値 `0` を
/// 返すはず。
#[test]
fn avoids_intermediate_overflow_for_finite_max_magnitude_inputs() {
    let xs = [f32::MAX, f32::MAX, -f32::MAX, -f32::MAX];
    let got = bias_scale_sum_reduce(&xs);
    assert!(
        got.is_finite(),
        "scale 方式は中間 overflow を回避し有限値を返すはず（got={got}）"
    );
    fandhe_ai_backend_cpu::assert_parity(
        "bias_scale_sum_reduce は [MAX, MAX, -MAX, -MAX] で真値 0 と REQ-2 複合判定で \
         一致するはず（中間 overflow 回避の確認）",
        &[got],
        &[0.0f32],
    );
}

/// (c) 相殺後の非ゼロ寄与が保持されることの確認（f64 アキュムレータ版
/// `reduce_bias_grad_rows`〈#1566〉の回帰テストと同種の入力）。
#[test]
fn preserves_non_cancelling_residual_contribution() {
    let xs = [1.0e8f32, 1.0, -1.0e8];
    let got = bias_scale_sum_reduce(&xs);
    fandhe_ai_backend_cpu::assert_parity(
        "bias_scale_sum_reduce は [1e8, 1.0, -1e8] で真値 1.0 と REQ-2 複合判定で \
         一致するはず",
        &[got],
        &[1.0f32],
    );
}

/// (d) 乱数列（大小さまざまな桁の値が混在）に対し、`f64` 逐次和
/// （真値の近似参照）と REQ-2 統一複合判定で一致することを確認する。
/// 決定的シード PRNG（`.claude/rules/coding-rust.md`「学習系回帰
/// テストには決定的シード設定ユーティリティを使う」）で桁の異なる値
/// （`[-1, 1)` の一様分布に `10^{-3..8}` のスケールを乱数選択で掛ける）
/// を生成する。
#[test]
fn matches_f64_reference_sum_for_mixed_magnitude_random_sequence() {
    let mut rng = Xorshift64Star::new(0xB1A5_5CA1_E000_0001);
    const SCALES: [f64; 12] = [
        1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7, 1e8,
    ];
    const LEN: usize = 200;
    let mut xs = Vec::with_capacity(LEN);
    for _ in 0..LEN {
        // `Xorshift64Star::next_f32` は `[-1.0, 1.0)` を返す契約
        // （`rng.rs` doc 参照）。符号付き係数はそのまま使い、桁選択用の
        // もう 1 個の乱数は絶対値を `[0, 1)` へ丸めてスケール表の添字へ
        // 変換する。
        let signed = rng.next_f32();
        let scale_pick = rng.next_f32().abs();
        let scale_idx = ((scale_pick * SCALES.len() as f32) as usize).min(SCALES.len() - 1);
        let v = f64::from(signed) * SCALES[scale_idx];
        xs.push(v as f32);
    }

    let got = bias_scale_sum_reduce(&xs);
    let f64_truth: f64 = xs.iter().map(|&v| f64::from(v)).sum();
    fandhe_ai_backend_cpu::assert_parity(
        &format!(
            "bias_scale_sum_reduce は乱数列（混在桁）で f64 真値 {f64_truth} と REQ-2 \
             複合判定で一致するはず"
        ),
        &[got],
        &[f64_truth as f32],
    );
}

/// `bias_pow2_floor` 単体の性質確認: `ax` 以下の最大の 2 のべき乗を
/// 返し、`ax` が既に厳密な 2 のべき乗の場合は `ax` 自身を返す（exact
/// 除算・比の前提となる不変条件）。`f32::MAX` でも overflow せず有限
/// 値を返すことも確認する（`ceil` 版なら `+inf` になっていたはずの
/// ケース）。
#[test]
fn pow2_floor_is_exact_and_never_overflows_for_finite_input() {
    assert_eq!(bias_pow2_floor(1.0), 1.0);
    assert_eq!(bias_pow2_floor(2.0), 2.0);
    assert_eq!(bias_pow2_floor(3.0), 2.0);
    assert_eq!(bias_pow2_floor(1e8), 67108864.0); // 2^26 <= 1e8 < 2^27
    assert_eq!(bias_pow2_floor(100000008.0), 67108864.0); // 同じ 2^26 帯域

    let floor_of_max = bias_pow2_floor(f32::MAX);
    assert!(
        floor_of_max.is_finite(),
        "f32::MAX の pow2_floor は有限のはず（floor_of_max={floor_of_max}）"
    );
    assert_eq!(floor_of_max, 2f32.powi(127));
    assert!(floor_of_max <= f32::MAX);
}
