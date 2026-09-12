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
//!
//! # Tier A（イシュー #1666・codex-review P1 是正）
//!
//! REQ-2 判定（上記）は入力の条件数によっては成立しない場合があり、
//! 事前判定できない「除外範囲」が生じるという codex-review 指摘を受け、
//! 判定契約を **Tier A（全入力に常に適用する理論上界）**・**Tier B
//! （REQ-2 複合判定。Tier A の上界が事前に REQ-2 閾値以下と分かる列
//! にのみ適用）** の 2 層構造へ改めた（正本
//! `docs/metal-grad-reduction-parity-judgment-decision.md` 予定・
//! `docs/backend-metal-command-batching-design.md` §10.13）。Tier A は
//! `[assert_tier_a]`・`[tier_a_holds_for_cancelling_extreme_magnitude_
//! sequence]` 等（下記）が担う。上記の「REQ-2」表記はこの 2 層構造の
//! うち Tier B を指す。

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

/// Tier A（イシュー #1666・codex-review P1「除外範囲を事前判定できる
/// 検証可能な契約にせよ」を受けた 2 層契約。正本は
/// `docs/metal-grad-reduction-parity-judgment-decision.md` 予定・
/// `docs/perf/train-resident-grad-device-update.md` §10.x 参照）:
/// `f32` unit roundoff（`ε32 = 2^-24`。round-to-nearest の 1 ulp 相対
/// 誤差上限）。
const EPS32: f64 = 1.0 / (1u64 << 24) as f64;

/// Tier A の安全側定数 `C = 3`。導出:
///
/// `gemm_bias_grad_reduce_f32` は 1 出力列につき 1 thread が `m` 行を
/// 逐次処理する構成であり、`rmsnorm.metal` のような thread 間 butterfly
/// 結合を**持たない**（段数 = 1）。各項 `x_i / scale`（`scale` は 2 の
/// べき乗）は指数部のシフトのみで exact（丸めなし）、再スケール時の
/// `acc *= ratio`／`comp *= ratio`（`ratio` も 2 のべき乗同士の比）も
/// exact、最終読み出しの `scale * (acc + comp)`（`scale` は 2 のべき乗）
/// も exact——誤差源は `bias_kahan_add`（Neumaier 改良版 Kahan 補償和）
/// 本体の丸めのみに帰着する。古典的な Neumaier／Kahan-Babuska 補償和の
/// 前方誤差上界（Higham, *Accuracy and Stability of Numerical
/// Algorithms*）は `|E_n| ≤ (2u + O(n u²)) Σ|t_i|` であり、
/// `Σ|t_i| = Σ|x_i| / scale`・最終読み出しの exact な `scale` 倍を
/// 適用すると `|Δ| ≤ (2ε32 + O(m ε32²)) Σ|x_i|` になる（`u = ε32`。
/// `O(m ε32²)` 項は `m` が現実的な行数〈高々数千〜数万〉である限り
/// `ε32²=2^-48` により無視できるほど小さい）。ホスト参照（`eval::
/// reduce_bias_grad_rows` 等）側の `f64` → `f32` 最終 downcast 1 回も
/// `≤ 0.5 ulp` 相対誤差（`0.5 ε32`）を追加しうる（`|S_f64| ≤ Σ|x_i|`
/// のため `Σ|x_i|` 基準でも同じ上界に収まる）。合計 `2 + 0.5 = 2.5` を
/// 安全側に切り上げ、`C = 段数(1) × 2 + downcast 余裕 1 = 3` とする。
const TIER_A_C: f64 = 3.0;

/// `xs` に対し Tier A（`|Δ| ≤ C·ε32·Σ|x_i|`。理論上界・全入力へ常に
/// 適用）を検証し、観測比 `|Δ| / (ε32·Σ|x_i|)`（`Σ|x_i| == 0` の場合は
/// `0.0`）を返す。`Σ|x_i|` は `f32::MAX` 級の入力でも overflow しない
/// よう `f64` で計算する。
fn assert_tier_a(label: &str, xs: &[f32]) -> f64 {
    let y_metal = bias_scale_sum_reduce(xs);
    let sum_f64: f64 = xs.iter().map(|&v| f64::from(v)).sum();
    let y_f64_downcast = sum_f64 as f32;
    let sum_abs: f64 = xs.iter().map(|&v| f64::from(v).abs()).sum();
    let delta = (f64::from(y_metal) - f64::from(y_f64_downcast)).abs();
    let bound = TIER_A_C * EPS32 * sum_abs;
    let ratio = if sum_abs > 0.0 {
        delta / (EPS32 * sum_abs)
    } else {
        0.0
    };
    assert!(
        delta <= bound,
        "{label}: Tier A 上界超過（|Δ|={delta:e}, bound=C·ε32·Σ|x_i|={bound:e}, \
         観測比={ratio:.4}, C={TIER_A_C}）"
    );
    ratio
}

/// Tier A 単体: codex-review 指摘の直接検証入力
/// `[2^48, 2^24, 1, -2^48, -2^24]`（真値 `1`。`2^48`・`2^24` は `f32`
/// で exact に表現できる 2 のべき乗のため、真値の exactness 自体は
/// 本テストの前提として崩れない）。
#[test]
fn tier_a_holds_for_cancelling_extreme_magnitude_sequence() {
    let xs = [
        2f32.powi(48),
        2f32.powi(24),
        1.0,
        -(2f32.powi(48)),
        -(2f32.powi(24)),
    ];
    let ratio = assert_tier_a("[2^48, 2^24, 1, -2^48, -2^24]", &xs);
    println!(
        "[tier_a_holds_for_cancelling_extreme_magnitude_sequence] observed_ratio={ratio:.3e} \
         (C={TIER_A_C})"
    );
}

/// Tier A 単体: 既存 3 ケース（(a)/(b)/(c)。上記 Tier B〈REQ-2〉テストと
/// 同じ入力）でも Tier A が成立することを確認する。
#[test]
fn tier_a_holds_for_existing_rounding_prone_and_overflow_cases() {
    let cases: [(&str, &[f32]); 3] = [
        (
            "preserves_cancelling_contribution",
            &[1.0e8f32, -100000008.0, 8.0],
        ),
        (
            "avoids_intermediate_overflow",
            &[f32::MAX, f32::MAX, -f32::MAX, -f32::MAX],
        ),
        (
            "preserves_non_cancelling_residual",
            &[1.0e8f32, 1.0, -1.0e8],
        ),
    ];
    let mut max_ratio = 0.0f64;
    for (label, xs) in cases {
        let ratio = assert_tier_a(label, xs);
        max_ratio = max_ratio.max(ratio);
    }
    println!(
        "[tier_a_holds_for_existing_rounding_prone_and_overflow_cases] max_observed_ratio=\
         {max_ratio:.3e} (C={TIER_A_C})"
    );
}

/// Tier A 単体: 高条件数（κ = Σ|x_i| / |S|）の乱数列を多数生成し、
/// 観測比 `|Δ| / (ε32·Σ|x_i|)` の最大値が `C=3` の範囲内に収まることを
/// 確認する（イシュー #1666 の依頼「乱数高 κ 列（m=4096 程度・符号
/// 混在・振幅 2^-20〜2^20 の対数一様）数十本」）。決定的シード PRNG
/// （`.claude/rules/coding-rust.md`）で系列ごとに独立したシードを使う。
#[test]
fn tier_a_holds_for_high_kappa_random_columns() {
    const M: usize = 4096;
    const TRIALS: usize = 30;
    let mut max_ratio = 0.0f64;
    let mut max_ratio_trial = 0usize;
    for trial in 0..TRIALS {
        let mut rng = Xorshift64Star::new(0xC0FF_EE00_0000_0001u64 ^ (trial as u64));
        let mut xs = Vec::with_capacity(M);
        for _ in 0..M {
            // 符号は独立の乱数draw、振幅は `next_f32()`（`[-1, 1)`）を
            // `[-20, 20)` へ線形写像した指数で対数一様に生成する
            // （`2^-20`〜`2^20` の振幅レンジ）。
            let sign = if rng.next_f32() >= 0.0 { 1.0f64 } else { -1.0 };
            let exponent = f64::from(rng.next_f32()) * 20.0;
            let magnitude = 2f64.powf(exponent);
            xs.push((sign * magnitude) as f32);
        }
        let ratio = assert_tier_a(&format!("high_kappa_random[trial={trial}]"), &xs);
        if ratio > max_ratio {
            max_ratio = ratio;
            max_ratio_trial = trial;
        }
    }
    println!(
        "[tier_a_holds_for_high_kappa_random_columns] max_observed_ratio={max_ratio:.3e} \
         (trial={max_ratio_trial}, m={M}, trials={TRIALS}, C={TIER_A_C})"
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
