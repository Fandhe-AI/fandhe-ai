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
//! `[1e8, -100000008.0, 8.0]`（`S_ref = 0`）のような相殺入力で REQ-2
//! （相対誤差 1e-3 未満 または 絶対誤差 1e-5 未満）を満たさない誤差
//! （実測 約 2.04）が生じた。本ファイルの
//! [`preserves_cancelling_contribution_with_rounding_prone_values`] が
//! この再現・是正確認を直接担う。
//!
//! # Tier A／Tier B の 2 層契約（イシュー #1666・codex-review 是正）
//!
//! REQ-2 判定（上記）は入力の条件数によっては成立しない場合があり、
//! 事前判定できない「除外範囲」が生じるという codex-review 指摘（P1）
//! を受け、判定契約を次の 2 層構造へ改めた（正本
//! `docs/metal-grad-reduction-parity-judgment-decision.md` 予定・
//! `docs/backend-metal-command-batching-design.md` §10.13）:
//!
//! - **参照値**: `S_ref` は `xs` をホスト `f64` で index 順に逐次
//!   加算した和（`reduce_bias_grad_rows_host`／`eval::reduce_bias_
//!   grad_rows` と同じ縮約順序。厳密和・真値ではなく「参照実装が
//!   計算する値」という位置づけ）。`y_ref` は `S_ref` を 1 回
//!   downcast した `f32`。
//! - **Tier A（全入力に常に適用）**: `y_metal`（本ファイルの
//!   `bias_scale_sum_reduce`。`gemm_bias_grad_reduce_f32` の逐語移植）
//!   と `y_ref` の差が明示的な理論上界
//!   `|y_metal − y_ref| ≤ (3 + n·ε32) · ε32 · Σ|x_i|`
//!   （`ε32 = 2^-24`・`n` は縮約要素数〈行数 `m`〉・有効範囲
//!   `n < 2^24`。`O` 記法は使わない）を満たす。`Σ|x_i|` はホスト
//!   `f64` で index 順に累積する（`sum_abs` 関数参照）。
//! - **Tier B（REQ-2 複合判定）**: `(3 + n·ε32)·ε32·Σ|x_i| ≤
//!   max(1e-3·|S_ref|, 1e-5)` が入力から事前に成立する列にのみ
//!   `REQ-2` 統一複合判定（`fandhe_ai_backend_cpu::assert_parity`）を
//!   適用する。不成立列は Tier A のみで検証する（本ファイルの
//!   `tier_b_applicable` 参照）。
//!
//! `[assert_tier_a]`／`[tier_a_holds_for_cancelling_extreme_magnitude_
//! sequence]` 等（下記）がこの契約を実装する。

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
/// ため、S_ref=0 と REQ-2 統一複合判定で一致するはず。
#[test]
fn preserves_cancelling_contribution_with_rounding_prone_values() {
    let xs = [1.0e8f32, -100000008.0, 8.0];
    let got = bias_scale_sum_reduce(&xs);
    let s_ref: f64 = xs.iter().map(|&v| f64::from(v)).sum();
    assert!(
        s_ref == 0.0,
        "test fixture: S_ref が 0 であることの前提確認（s_ref={s_ref}）"
    );
    fandhe_ai_backend_cpu::assert_parity(
        "bias_scale_sum_reduce は相殺入力 [1e8, -100000008.0, 8.0] で S_ref=0 と \
         REQ-2 複合判定で一致するはず（codex-review 指摘の是正確認）",
        &[got],
        &[0.0f32],
    );
}

/// (b) 当初の P1 指摘そのもの: 有限入力の中間 overflow 再現テスト。
/// `f32::MAX + f32::MAX` は素朴な逐次和では `+inf` へ overflow するが、
/// scale 方式（2 のべき乗版）は overflow せず有限のまま S_ref=0 を
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
        "bias_scale_sum_reduce は [MAX, MAX, -MAX, -MAX] で S_ref=0 と REQ-2 複合判定で \
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
        "bias_scale_sum_reduce は [1e8, 1.0, -1e8] で S_ref=1.0 と REQ-2 複合判定で \
         一致するはず",
        &[got],
        &[1.0f32],
    );
}

/// (d) 乱数列（大小さまざまな桁の値が混在）に対し、`f64` 逐次和
/// （`S_ref`。`xs` をホスト `f64` で index 順に逐次加算した和）と
/// REQ-2 統一複合判定で一致することを確認する。
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
    let s_ref: f64 = xs.iter().map(|&v| f64::from(v)).sum();
    fandhe_ai_backend_cpu::assert_parity(
        &format!(
            "bias_scale_sum_reduce は乱数列（混在桁）で S_ref={s_ref} と REQ-2 \
             複合判定で一致するはず"
        ),
        &[got],
        &[s_ref as f32],
    );
}

/// `f32` unit roundoff（`ε32 = 2^-24`。round-to-nearest の 1 ulp 相対
/// 誤差上限）。Tier A／Tier B の 2 層契約（イシュー #1666・codex-review
/// 是正。定義は本ファイル冒頭 doc 参照）で共通に使う。
const EPS32: f64 = 1.0 / (1u64 << 24) as f64;

/// Tier A の明示式 `|y_metal − y_ref| ≤ (3 + n·ε32) · ε32 · Σ|x_i|` の
/// 加法定数 `3`。導出:
///
/// `gemm_bias_grad_reduce_f32` は 1 出力列につき 1 thread が `n`（行数
/// `m`）要素を逐次処理する構成であり、`rmsnorm.metal` のような thread
/// 間 butterfly 結合を**持たない**（段数 = 1）。各項 `x_i / scale`
/// （`scale` は 2 のべき乗）は指数部のシフトのみで exact（丸めなし）、
/// 再スケール時の `acc *= ratio`／`comp *= ratio`（`ratio` も 2 の
/// べき乗同士の比）も exact、最終読み出しの `scale * (acc + comp)`
/// （`scale` は 2 のべき乗）も exact——誤差源は `bias_kahan_add`
/// （Neumaier 改良版 Kahan 補償和）本体の丸めのみに帰着する。古典的な
/// Neumaier／Kahan-Babuska 補償和の前方誤差上界（Higham, *Accuracy
/// and Stability of Numerical Algorithms*）は、`u` を unit roundoff
/// として `|E_n| ≤ 2u Σ|t_i| + n(n-1)u² Σ|t_i|` の形（`O` 記法を使わず
/// 明示すると `n(n-1)u² ≤ n²u²` で抑えられる）であり、`u = ε32`・
/// `Σ|t_i| = Σ|x_i| / scale`・最終読み出しの exact な `scale` 倍を
/// 適用すると `|Δ| ≤ (2 + n·ε32) · ε32 · Σ|x_i|` になる（`n²ε32² =
/// n·ε32·ε32·n` を安全側に `n·ε32` 倍として単項化。有効範囲
/// `n < 2^24` では `n·ε32 < 1` のため単調に効く）。ホスト参照
/// （`S_ref`）側の `f64` → `f32` 最終 downcast 1 回も `≤ 0.5 ulp`
/// 相対誤差（`0.5 ε32`）を追加しうる（`|S_ref| ≤ Σ|x_i|` のため
/// `Σ|x_i|` 基準でも同じ上界に収まる）。合計 `2 + 0.5 = 2.5` を安全側に
/// 切り上げ、`(3 + n·ε32) · ε32 · Σ|x_i|` を Tier A の明示式とする。
const TIER_A_ADDITIVE_CONST: f64 = 3.0;

/// Tier A 上界 `(3 + n·ε32) · ε32 · Σ|x_i|`（`n = xs.len()`。有効範囲
/// `n < 2^24`）を計算する。
fn tier_a_bound(n: usize, sum_abs: f64) -> f64 {
    assert!(
        n < (1usize << 24),
        "Tier A 上界の有効範囲外（n={n} ≥ 2^24）"
    );
    (TIER_A_ADDITIVE_CONST + (n as f64) * EPS32) * EPS32 * sum_abs
}

/// `xs` の参照値 `S_ref`（ホスト `f64` で index 順に逐次加算した和。
/// `reduce_bias_grad_rows_host`／`eval::reduce_bias_grad_rows` と同じ
/// 縮約順序）と `y_ref`（`S_ref` を 1 回 downcast した `f32`）を返す。
fn s_ref_and_y_ref(xs: &[f32]) -> (f64, f32) {
    let s_ref: f64 = xs.iter().map(|&v| f64::from(v)).sum();
    (s_ref, s_ref as f32)
}

/// `Σ|x_i|`（ホスト `f64` で index 順に累積。`f32::MAX` 級の入力でも
/// overflow しないよう `f64` で計算する）。
fn sum_abs(xs: &[f32]) -> f64 {
    xs.iter().map(|&v| f64::from(v).abs()).sum()
}

/// Tier B 述語: `(3 + n·ε32)·ε32·Σ|x_i| ≤ max(1e-3·|S_ref|, 1e-5)`
/// （REQ-2 統一複合判定の閾値を Tier A 上界が事前に下回るか。条件数
/// `κ = Σ|x_i| / |S_ref|` の上限と等価）。成立する列にのみ Tier B
/// （`assert_parity`）を適用する契約。
fn tier_b_applicable(bound: f64, s_ref: f64) -> bool {
    bound <= (1e-3 * s_ref.abs()).max(1e-5)
}

/// `xs` に対し Tier A（全入力へ常に適用する明示式の理論上界）を検証し、
/// 観測比 `|Δ| / (ε32·Σ|x_i|)`（`Σ|x_i| == 0` の場合は `0.0`）を返す。
fn assert_tier_a(label: &str, xs: &[f32]) -> f64 {
    let y_metal = bias_scale_sum_reduce(xs);
    let (_s_ref, y_ref) = s_ref_and_y_ref(xs);
    let sa = sum_abs(xs);
    let bound = tier_a_bound(xs.len(), sa);
    let delta = (f64::from(y_metal) - f64::from(y_ref)).abs();
    let ratio = if sa > 0.0 { delta / (EPS32 * sa) } else { 0.0 };
    assert!(
        delta <= bound,
        "{label}: Tier A 上界超過（|Δ|={delta:e}, bound=(3+n·ε32)·ε32·Σ|x_i|={bound:e},          観測比={ratio:.4}, n={}）",
        xs.len()
    );
    ratio
}

/// Tier A 単体: codex-review 指摘の直接検証入力
/// `[2^48, 2^24, 1, -2^48, -2^24]`（`S_ref = 1`。`2^48`・`2^24` は
/// `f32` で exact に表現できる 2 のべき乗のため、`S_ref` の exactness 自体は
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
    println!("[tier_a_holds_for_cancelling_extreme_magnitude_sequence] observed_ratio={ratio:.3e}");
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
        "[tier_a_holds_for_existing_rounding_prone_and_overflow_cases] max_observed_ratio=         {max_ratio:.3e}"
    );
}

/// Tier A 単体: 高条件数（κ = Σ|x_i| / |S_ref|）の乱数列を多数生成し、
/// 観測比 `|Δ| / (ε32·Σ|x_i|)` の最大値が Tier A 上界の範囲内に収まる
/// ことを確認する（イシュー #1666 の依頼「乱数高 κ 列（m=4096 程度・
/// 符号混在・振幅 2^-20〜2^20 の対数一様）数十本」）。決定的シード
/// PRNG（`.claude/rules/coding-rust.md`）で系列ごとに独立したシードを
/// 使う。
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
        "[tier_a_holds_for_high_kappa_random_columns] max_observed_ratio={max_ratio:.3e}          (trial={max_ratio_trial}, m={M}, trials={TRIALS})"
    );
}

/// Tier B 述語（`tier_b_applicable`）が、条件数によって Tier B の
/// 適用可否を正しく振り分けることを機械検査する:
///
/// - `[2^48, 2^24, 1, -2^48, -2^24]`（`Σ|x_i| ≈ 5.6×10^14`・
///   `S_ref = 1`。条件数 `κ ≈ 5.6×10^14` が極端に高い）は Tier A の
///   上界自体が REQ-2 閾値（`max(1e-3·|S_ref|, 1e-5) = 1e-3`）を
///   大きく超えるため Tier B 不成立——Tier A のみで検証する契約になる
///   （Tier A 自体は成立する。上記
///   `tier_a_holds_for_cancelling_extreme_magnitude_sequence` 参照）。
/// - 条件数の小さい列（`[3.0, 4.0, -2.0]`。`Σ|x_i| = 9`・
///   `S_ref = 5`・`κ = 1.8`）は Tier A 上界が REQ-2 閾値を十分下回る
///   ため Tier B 成立——`assert_parity`（REQ-2 統一複合判定）を実際に
///   適用して pass することを確認する。
///
/// **注記**: `[1e8, 1.0, -1e8]`（`preserves_non_cancelling_residual_
/// contribution` の入力）は条件数 `κ = Σ|x_i|/|S_ref| ≈ 2×10^8` が
/// 極めて高く、Tier A 明示式では Tier B 不成立（`bound ≈ 36` が
/// `1e-3` を大きく超える）と判定される——実際の丸め誤差はゼロ
/// （`delta = 0`）で REQ-2 自体には経験的に一致するが、これは Tier A
/// 上界（最悪ケース保証）が緩いことの帰結であり、Tier B 契約（事前に
/// 保証できる列）の対象ではない。既存テスト
/// `preserves_non_cancelling_residual_contribution` は Tier B の
/// 事前保証とは独立に、この入力で `assert_parity` が経験的に成立する
/// ことを確認する記録として維持する（削除・変更しない）。
#[test]
fn tier_b_predicate_selects_expected_cases_and_matches_assert_parity() {
    // 高条件数（Tier B 不成立の予定）。
    let xs_high_kappa = [
        2f32.powi(48),
        2f32.powi(24),
        1.0,
        -(2f32.powi(48)),
        -(2f32.powi(24)),
    ];
    let (s_ref_hk, _y_ref_hk) = s_ref_and_y_ref(&xs_high_kappa);
    let sa_hk = sum_abs(&xs_high_kappa);
    let bound_hk = tier_a_bound(xs_high_kappa.len(), sa_hk);
    assert!(
        !tier_b_applicable(bound_hk, s_ref_hk),
        "[2^48, 2^24, 1, -2^48, -2^24] は高条件数（κ={:e}）のため Tier B 不成立の          はず（bound={bound_hk:e}, threshold={:e}）",
        sa_hk / s_ref_hk.abs(),
        (1e-3 * s_ref_hk.abs()).max(1e-5)
    );
    // Tier A のみで検証する（Tier B 不成立でも Tier A は常に成立する契約）。
    let ratio_hk = assert_tier_a("tier_b_predicate/high_kappa", &xs_high_kappa);
    println!(
        "[tier_b_predicate_selects_expected_cases_and_matches_assert_parity]          high_kappa: tier_b_applicable=false observed_ratio={ratio_hk:.3e}"
    );

    // 低条件数（Tier B 成立の予定）。
    let xs_low_kappa = [3.0f32, 4.0, -2.0];
    let (s_ref_lk, _y_ref_lk) = s_ref_and_y_ref(&xs_low_kappa);
    let sa_lk = sum_abs(&xs_low_kappa);
    let bound_lk = tier_a_bound(xs_low_kappa.len(), sa_lk);
    assert!(
        tier_b_applicable(bound_lk, s_ref_lk),
        "[3.0, 4.0, -2.0] は低条件数（κ={:e}）のため Tier B 成立のはず          （bound={bound_lk:e}, threshold={:e}）",
        sa_lk / s_ref_lk.abs(),
        (1e-3 * s_ref_lk.abs()).max(1e-5)
    );
    // Tier A・Tier B 双方を検証する（Tier B 成立列は REQ-2 複合判定を
    // 実際に適用できる）。
    let ratio_lk = assert_tier_a("tier_b_predicate/low_kappa", &xs_low_kappa);
    let got_lk = bias_scale_sum_reduce(&xs_low_kappa);
    fandhe_ai_backend_cpu::assert_parity(
        "[3.0, 4.0, -2.0]（Tier B 成立列）は S_ref=5.0 と REQ-2 複合判定で          一致するはず",
        &[got_lk],
        &[s_ref_lk as f32],
    );
    println!(
        "[tier_b_predicate_selects_expected_cases_and_matches_assert_parity]          low_kappa: tier_b_applicable=true observed_ratio={ratio_lk:.3e}"
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
