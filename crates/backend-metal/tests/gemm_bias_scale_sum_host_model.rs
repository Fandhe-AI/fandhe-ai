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
//!   `|y_metal − y_ref| ≤ (4 + n·ε32) · ε32 · Σ|x_i|`
//!   （`ε32 = 2^-24`・`n` は縮約要素数〈行数 `m`〉・有効範囲
//!   `n < 2^24`。`O` 記法は使わない）を満たす。`Σ|x_i|` はホスト
//!   `f64` で index 順に累積する（`sum_abs` 関数参照）。
//! - **Tier B（REQ-2 複合判定）**: `(4 + n·ε32)·ε32·Σ|x_i| ≤
//!   max(1e-3·|S_ref|, 1e-5)` が入力から事前に成立する列にのみ
//!   `REQ-2` 統一複合判定（`fandhe_ai_backend_cpu::assert_parity`）を
//!   適用する。不成立列は Tier A のみで検証する（本ファイルの
//!   `tier_b_applicable` 参照）。
//!
//! `[assert_tier_a]`／`[tier_a_holds_for_cancelling_extreme_magnitude_
//! sequence]` 等（下記）がこの契約を実装する。
//!
//! # 入力上限・非有限値のクラス一致（イシュー #1666・codex-review 追加
//! P1 是正）
//!
//! - Tier A の有効範囲は縮約要素数 `n < 2^24`（`tier_a_bound` の
//!   `assert!` で機械検査。上限定数 `BIAS_GRAD_MAX_ROWS` は
//!   `crate::layout` 側で定義し、`m >= 2^24` を `BackendError::
//!   InvalidArgument` として fail-closed 拒否する。`layout.rs` の
//!   契約テスト参照）。
//! - Tier A は `y_metal`・`y_ref` の**両方が有限の場合にのみ**適用する
//!   （`tier_a_bound`／`assert_tier_a` doc 参照）。`y_ref` が非有限の
//!   場合は `y_metal` が同クラス（`NaN`↔`NaN`・`±inf` は同符号）に
//!   到達することを [`assert_class_match`] で検証する
//!   （[`nonfinite_class_matching_cases`] 参照）。
//! - 2 のべき乗 `scale` 除算の underflow（商が厳密 `0.0` へ丸められる
//!   ケース。codex-review 追加 P2）は「列合計 `≤ n·2^-150·Σ|x_i| ≤
//!   2^-126·Σ|x_i|` は Tier A の加法定数 `4`（安全余裕分）に
//!   吸収済み」と規定する
//!   （[`tier_a_holds_for_pow2_scale_underflow_case_single_tiny_term`]／
//!   [`tier_a_holds_for_pow2_scale_underflow_case_two_tiny_terms`]
//!   参照）。

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
///
/// **契約（イシュー #1666・codex-review 追加 P1 是正）**: `x` は常に
/// 有限であることを呼び出し元（`bias_scale_sum_reduce`）が保証する
/// （列内に非有限値を検出した場合は本関数を一切呼ばず、素朴な `f32`
/// 逐次和 `naive_acc` へ切り替える。`bias_scale_sum_reduce` doc
/// 「非有限値のクラス一致伝播」参照）。このため本関数自体は
/// `NaN`／`±inf` の特殊扱いを持たず、`bias_pow2_floor` へも有限値
/// のみが渡る（`gemm.metal::bias_scale_sum_add` と同一構成）。
fn bias_scale_sum_add(scale: &mut f32, acc: &mut f32, comp: &mut f32, x: f32) {
    let ax = x.abs();
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
///
/// **非有限値のクラス一致伝播（イシュー #1666・codex-review 追加 P1
/// 是正）**: 契約は「`y_ref`（ホスト `f64` 逐次和を 1 回 downcast した
/// `f32`）と `y_metal` の両方が有限なら Tier A／B、`y_ref` が非有限
/// なら `y_metal` は同クラス（`NaN`↔`NaN`・`±inf` は同符号）」
/// （`docs/backend-metal-command-batching-design.md` §10.13）。scale
/// 方式（2 のべき乗）は列内の要素がすべて有限であることを前提に
/// 中間 overflow を回避する設計のため、列内に `NaN`／`±inf` を含む
/// 場合は scale 方式を経由せず、**素朴な `f32` 逐次和**
/// （`naive_acc += x` を単純な `+` 演算子で行う）へ切り替える。
/// `f32`／`f64` いずれの IEEE 754 逐次和も `NaN`／`±inf` の伝播規則
/// （`inf + finite == inf`・`inf + inf == inf`〈同符号〉・
/// `inf + (-inf) == NaN`・`NaN + 任意 == NaN`〈sticky〉）は精度に
/// 依存せず同一構造のため、素朴な `f32` 逐次和は `f64` 逐次和と常に
/// 同じクラス（同符号の `±inf`、または `NaN`）に到達する（有限項の
/// 大小・順序に関わらず、列内の `NaN`／`±inf` の出現パターンのみで
/// クラスが決まるため）。
fn bias_scale_sum_reduce(xs: &[f32]) -> f32 {
    let mut scale = 0.0f32;
    let mut acc = 0.0f32;
    let mut comp = 0.0f32;
    let mut naive_acc = 0.0f32;
    let mut has_nonfinite = false;
    for &x in xs {
        naive_acc += x;
        if x.is_nan() || x.is_infinite() {
            has_nonfinite = true;
        }
        if !has_nonfinite {
            bias_scale_sum_add(&mut scale, &mut acc, &mut comp, x);
        }
    }
    if has_nonfinite {
        naive_acc
    } else {
        scale * (acc + comp)
    }
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

/// Tier A の明示式 `|y_metal − y_ref| ≤ (4 + n·ε32) · ε32 · Σ|x_i|` の
/// 加法定数 `4`（2026-09-12 codex 指摘によりコーディネータが `C=3` →
/// `C=4` へ確定し直した。以下は改訂後の導出）:
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
/// （`S_ref`）側の `f64` → `f32` 最終 downcast 1 回の誤差は、基準を
/// `Σ|x_i|` に統一すると（codex 指摘・是正）`0.5u` ではなく **`1u`**
/// を計上する必要がある: downcast の丸め誤差は `|S_ref|` を基準に
/// `≤ 0.5 ulp`（`0.5 ε32 · |S_ref|`）だが、`|S_ref| ≤ Σ|x_i|` という
/// 不等式だけでは `0.5 ε32 · |S_ref| ≤ 0.5 ε32 · Σ|x_i|` の関係しか
/// 導けず、`S_ref` 自体が `Σ|x_i|` よりはるかに小さい高条件数の入力
/// （本ファイルの高 κ テスト群）では downcast 誤差の絶対値が
/// `Σ|x_i|` の何倍にもなりうる場合と紙一重になるため、安全側に
/// `1 ε32 · Σ|x_i|`（２倍の安全係数）を計上する。合計
/// `2（Kahan/Neumaier 本体）+ 1（downcast）+ 1（安全余裕）= 4` を
/// Tier A の加法定数とし、`(4 + n·ε32) · ε32 · Σ|x_i|` を明示式とする。
const TIER_A_ADDITIVE_CONST: f64 = 4.0;

/// Tier A 上界 `(4 + n·ε32) · ε32 · Σ|x_i|`（`n = xs.len()`。有効範囲
/// `n < 2^24`）を計算する。
///
/// **適用条件（イシュー #1666・codex-review 追加 P1 是正）**: Tier A は
/// `y_metal`・`y_ref` の**両方が有限の場合にのみ**適用する契約
/// （`docs/backend-metal-command-batching-design.md` §10.13）。いずれか
/// が非有限の場合は本関数・[`assert_tier_a`] の対象外であり、代わりに
/// [`assert_class_match`]（非有限値のクラス一致検証）を使う。
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

/// Tier B 述語: `(4 + n·ε32)·ε32·Σ|x_i| ≤ max(1e-3·|S_ref|, 1e-5)`
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
    assert!(
        y_metal.is_finite() && y_ref.is_finite(),
        "{label}: Tier A は y_metal・y_ref の両方が有限の場合にのみ適用する契約          （y_metal={y_metal}, y_ref={y_ref}）。非有限の場合は assert_class_match を使う"
    );
    let sa = sum_abs(xs);
    let bound = tier_a_bound(xs.len(), sa);
    let delta = (f64::from(y_metal) - f64::from(y_ref)).abs();
    let ratio = if sa > 0.0 { delta / (EPS32 * sa) } else { 0.0 };
    assert!(
        delta <= bound,
        "{label}: Tier A 上界超過（|Δ|={delta:e}, bound=(4+n·ε32)·ε32·Σ|x_i|={bound:e},          観測比={ratio:.4}, n={}）",
        xs.len()
    );
    ratio
}

/// 非有限値のクラス一致（イシュー #1666・codex-review 追加 P1 是正）を
/// 機械検査する: `y_ref`（`S_ref` の 1 回 downcast）が非有限の場合、
/// `y_metal`（[`bias_scale_sum_reduce`]）が同クラス（`NaN`↔`NaN`・
/// `±inf` は同符号）に到達することを確認する。
fn assert_class_match(label: &str, xs: &[f32]) {
    let y_metal = bias_scale_sum_reduce(xs);
    let (s_ref, y_ref) = s_ref_and_y_ref(xs);
    assert!(
        !y_ref.is_finite(),
        "{label}: assert_class_match は y_ref が非有限の入力専用（y_ref={y_ref},          S_ref={s_ref}）。有限なら assert_tier_a を使う"
    );
    if y_ref.is_nan() {
        assert!(
            y_metal.is_nan(),
            "{label}: y_ref=NaN のクラスに y_metal が一致しない（y_metal={y_metal}）"
        );
    } else {
        assert!(
            y_metal.is_infinite() && y_metal.signum() == y_ref.signum(),
            "{label}: y_ref={y_ref}（±inf）のクラスに y_metal が一致しない          （y_metal={y_metal}）"
        );
    }
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
/// 極めて高く、Tier A 明示式では Tier B 不成立（`bound ≈ 48` が
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

/// 非有限値のクラス一致伝播（イシュー #1666・codex-review 追加 P1
/// 是正）を機械検査する 6 ケース（コーディネータ指定の入力を逐語検査）。
/// `y_ref` が非有限になる列（(1)〜(5)）は [`assert_class_match`] で
/// クラス一致を確認し、`y_ref` が有限になる列（(6)）は [`assert_tier_a`]
/// で通常の Tier A 上界検証を行う（両者有限の場合にのみ Tier A が
/// 適用可能という条件そのものの確認を兼ねる）。
#[test]
fn nonfinite_class_matching_cases() {
    // (1) [f32::MAX, f32::MAX] -> 両者 +inf（f64 逐次和が f32 表現範囲を
    // 超えて downcast 時に +inf へ overflow する。scale 方式側も最終
    // `scale * (acc + comp)` の f32 乗算自体が同じ理由で +inf へ overflow
    // する——素朴な f32 逐次和と同一の IEEE 754 overflow 規則に従うため）。
    assert_class_match("[f32::MAX, f32::MAX]", &[f32::MAX, f32::MAX]);

    // (2) [-f32::MAX, -f32::MAX] -> 両者 -inf（(1) の符号反転）。
    assert_class_match("[-f32::MAX, -f32::MAX]", &[-f32::MAX, -f32::MAX]);

    // (3) [inf, -inf] -> 両者 NaN（`inf + (-inf) == NaN`）。
    assert_class_match("[inf, -inf]", &[f32::INFINITY, f32::NEG_INFINITY]);

    // (4) [NaN, 1] -> NaN（`NaN` は sticky に伝播する）。
    assert_class_match("[NaN, 1]", &[f32::NAN, 1.0]);

    // (5) [inf, 1, -5] -> +inf（有限項の値によらず符号付き無限大が支配する）。
    assert_class_match("[inf, 1, -5]", &[f32::INFINITY, 1.0, -5.0]);

    // (6) [f32::MAX, f32::MAX, -f32::MAX] -> 有限（S_ref = f32::MAX が
    // f32 の表現範囲内に収まる）。両者有限のため Tier A（通常の理論上界）
    // を適用する。
    let xs_finite = [f32::MAX, f32::MAX, -f32::MAX];
    let (s_ref, y_ref) = s_ref_and_y_ref(&xs_finite);
    assert!(
        y_ref.is_finite(),
        "test fixture: [f32::MAX, f32::MAX, -f32::MAX] の y_ref は有限のはず          （y_ref={y_ref}, S_ref={s_ref}）"
    );
    let ratio = assert_tier_a("[f32::MAX, f32::MAX, -f32::MAX]", &xs_finite);
    println!("[nonfinite_class_matching_cases] case(6) observed_ratio={ratio:.3e}");
}

/// 2 のべき乗 `scale` 除算の underflow（codex-review 追加 P2 是正・
/// イシュー #1666）: 商 `x / scale` が非正規化数域を超えて厳密な `0.0`
/// へ丸められる（gradual underflow の範囲外）要素を含む列でも、契約は
/// 「列合計 `≤ n·2^-150·Σ|x_i| ≤ 2^-126·Σ|x_i|` は Tier A の加法定数
/// `3`（`0.5u` の安全余裕）に吸収済み」と規定する。本テストはこの
/// 契約どおり Tier A が成立することを機械検査する。
///
/// `[2^127, 2^-149, -2^127]`: `scale = 2^127`（列内最大絶対値）。中央
/// 要素 `2^-149 / 2^127 = 2^-276` は `f32` の最小非正規化数
/// （`2^-149`）を大幅に下回るため商は厳密に `0.0` へ丸められる
/// （underflow）——`bias_kahan_add` への寄与が失われるが、この要素
/// 自体が `Σ|x_i|`（`≈ 2 * 2^127`）に対して無視できるほど小さいため
/// Tier A の理論上界（余裕分 `0.5u` 相当）に吸収される。
#[test]
fn tier_a_holds_for_pow2_scale_underflow_case_single_tiny_term() {
    let xs = [2f32.powi(127), 2f32.powi(-149), -(2f32.powi(127))];
    let ratio = assert_tier_a("[2^127, 2^-149, -2^127]", &xs);
    println!(
        "[tier_a_holds_for_pow2_scale_underflow_case_single_tiny_term] observed_ratio={ratio:.3e}"
    );
}

/// `[2^100, 2^-140, 2^-140, -2^100]`: `scale = 2^100`。`2^-140 / 2^100 =
/// 2^-240` も同様に厳密 `0.0` へ underflow する微小項を 2 個含む
/// （相殺後の真値は `0`）。
#[test]
fn tier_a_holds_for_pow2_scale_underflow_case_two_tiny_terms() {
    let xs = [
        2f32.powi(100),
        2f32.powi(-140),
        2f32.powi(-140),
        -(2f32.powi(100)),
    ];
    let ratio = assert_tier_a("[2^100, 2^-140, 2^-140, -2^100]", &xs);
    println!(
        "[tier_a_holds_for_pow2_scale_underflow_case_two_tiny_terms] observed_ratio={ratio:.3e}"
    );
}
