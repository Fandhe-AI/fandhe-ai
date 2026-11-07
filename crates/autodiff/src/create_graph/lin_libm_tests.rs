//! `Lin::powf_bounded`・`Lin::pow_grad_bounded` の誤差規則のテスト（イシュー #4084・設計記録
//! `docs/autodiff-higher-order-grad-decision.md` §81.10）。`lin.rs` の `#[path]` で読み込む子モジュール。
//!
//! 役割: libm の `powf` が結果の正規化数の範囲で 2 ulp 以内（[`LIBM_F32_MAX_ULP`]。所有者決定 #4083 の設計値）に
//! 収まるという仮定の下で、規則の上界が f32 構造の実測の偏差を覆うことを確かめる。**乱数掃引は証明ではない**。
//! 仮定を確かめた環境（x86_64 Linux glibc）以外では既定で `ignore` にし、計測テスト [`powf_ulp_measurement`]
//! （`--ignored`）を aarch64 Linux・macOS でも実行できる形にしてある（それらの値は未実測）。
//! 2 ulp を超える実測が出たら定数を上げず、`ignore` も足さず、失敗のまま報告する（#4083）。

use std::hint::black_box;

use super::tests::F32Rev;
use super::*;

/// 乱数（SplitMix64。決定的）。
struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
    /// `[0, 1)` の一様乱数。
    fn unit(&mut self) -> f64 {
        (self.next() >> 11) as f64 / (1u64 << 53) as f64
    }
    fn range(&mut self, lo: f64, hi: f64) -> f64 {
        lo + (hi - lo) * self.unit()
    }
}

/// 参照 `f64::powf` に対する f32 `powf` の誤差を ulp で返す。`ulp` は参照値の 2 進桁（`2^-126` 未満は `2^-149`）。
/// 参照が `2^128` 以上で f32 が `inf` なら誤差 0（overflow として別に数える）。戻りの 2 つ目は overflow か。
fn powf_ulp_err(x: f32, e: f32) -> (f64, bool) {
    let reference = f64::from(x).powf(f64::from(e));
    let actual = black_box(x).powf(black_box(e));
    if !reference.is_finite() || reference <= 0.0 {
        return (0.0, false);
    }
    let two128 = 2.0f64.powi(128);
    if reference >= two128 && actual.is_infinite() {
        return (0.0, true);
    }
    let actual64 = if actual.is_infinite() {
        two128
    } else {
        f64::from(actual)
    };
    let exp = ((reference.to_bits() >> 52) & 0x7ff) as i32 - 1023;
    let ulp = 2.0f64.powi((exp - 23).max(-149));
    ((actual64 - reference).abs() / ulp, false)
}

#[derive(Clone, Copy, Debug)]
struct Stat {
    count: u64,
    max_ulp: f64,
    max_at: (f32, f32),
    overflow: u64,
}

const STRATA: [&str; 6] = [
    "全域",
    "非正規化数",
    "オーバーフロー近傍",
    "底が1の近く",
    "大きい指数",
    "実用域",
];

/// 結果の log2 を `t` に狙って、底 `x`（f32）から指数 `e = t / log2 x` を逆算する。
fn aim(x: f32, t: f64) -> (f32, f32) {
    let l = f64::from(x).log2();
    (x, (t / l) as f32)
}

fn near_one(rng: &mut Rng, kmax: u64) -> f32 {
    let k = 1 + rng.next() % kmax;
    if rng.next() & 1 == 0 {
        1.0 + (k as f32) * 2.0f32.powi(-23)
    } else {
        1.0 - (k as f32) * 2.0f32.powi(-24)
    }
}

fn ref_sign(rng: &mut Rng) -> f64 {
    if rng.next() & 1 == 0 { 1.0 } else { -1.0 }
}

/// 層 `s` の入力を 1 件作る。
fn gen_case(s: usize, rng: &mut Rng) -> (f32, f32) {
    const FIXED_E: [f32; 9] = [0.5, 0.75, 1.0, 1.5, 2.0, 3.0, -0.75, -1.75, 1.0 / 3.0];
    let wide_base = |rng: &mut Rng| {
        let lx = rng.range(0.01, 40.0) * ref_sign(rng);
        (lx.exp() as f32).max(f32::MIN_POSITIVE)
    };
    match s {
        0 => {
            let x = wide_base(rng);
            aim(x, rng.range(-150.0, 128.0))
        }
        1 => {
            let x = wide_base(rng);
            aim(x, rng.range(-152.0, -125.0))
        }
        2 => {
            let x = wide_base(rng);
            aim(x, rng.range(127.0, 128.05))
        }
        3 => {
            let x = near_one(rng, 4096);
            aim(x, rng.range(-150.0, 128.0))
        }
        4 => {
            let x = near_one(rng, 8);
            let t = rng.range(20.0, 128.0) * ref_sign(rng);
            aim(x, t)
        }
        _ => {
            let x = 2.0f64.powf(rng.range(-50.0, 50.0)) as f32;
            if rng.next() & 1 == 0 {
                (x, rng.range(-8.0, 8.0) as f32)
            } else {
                (x, FIXED_E[(rng.next() % 9) as usize])
            }
        }
    }
}

/// 全層を `n` 件ずつ掃引する。
fn powf_ulp_sweep(n: u64, seed: u64) -> Vec<Stat> {
    (0..STRATA.len())
        .map(|s| {
            let mut rng = Rng(seed ^ ((s as u64 + 1) << 56));
            let mut st = Stat {
                count: 0,
                max_ulp: 0.0,
                max_at: (0.0, 0.0),
                overflow: 0,
            };
            for _ in 0..n {
                let (x, e) = gen_case(s, &mut rng);
                if !(x > 0.0 && e.is_finite()) {
                    continue;
                }
                let (u, ov) = powf_ulp_err(x, e);
                st.count += 1;
                st.overflow += u64::from(ov);
                if u > st.max_ulp {
                    st.max_ulp = u;
                    st.max_at = (x, e);
                }
            }
            st
        })
        .collect()
}

/// 通常の掃引の件数（各層）。debug ビルドでも数秒以内。
const SWEEP_N: u64 = 1 << 17;

/// `cargo test` で既定実行するのは、#4083 の仮定を確かめた環境（x86_64 Linux glibc）だけ。
macro_rules! assumption_env_test {
    ($(#[$m:meta])* fn $name:ident() $body:block) => {
        $(#[$m])*
        #[test]
        #[cfg_attr(
            not(all(target_os = "linux", target_arch = "x86_64", target_env = "gnu")),
            ignore = "未実測の環境。計測は powf_ulp_measurement"
        )]
        fn $name() $body
    };
}

assumption_env_test! {
    /// 通常の掃引で、全層の最大誤差が仮定 [`LIBM_F32_MAX_ULP`] 以内。
    fn lin_powf_ulp_within_assumption() {
        for (name, st) in STRATA.iter().zip(powf_ulp_sweep(SWEEP_N, 0x4084)) {
            assert!(st.count > SWEEP_N / 2, "{name}: 件数 {}", st.count);
            assert!(
                st.max_ulp <= LIBM_F32_MAX_ULP,
                "{name}: 最大 {} ulp が仮定 {LIBM_F32_MAX_ULP} を超えた (x,e)={:?}",
                st.max_ulp,
                st.max_at
            );
        }
    }
}

/// 計測（全環境で `--ignored` 実行可能。`--release` 推奨）。層ごとの統計を `POWF_ULP` 行で出し、2 ulp 以内を assert する。
/// 2 ulp を超えたら定数を上げず、層・件数・最大 ulp・入力・環境を #4084 へ報告する。
#[test]
#[ignore = "計測。cargo test --release --lib powf_ulp_measurement -- --ignored --nocapture"]
fn powf_ulp_measurement() {
    let per: u64 = std::env::var("POWF_ULP_N")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(1 << 28);
    let threads = std::thread::available_parallelism().map_or(1, |n| n.get()) as u64;
    let each = per.div_ceil(threads);
    let parts: Vec<Vec<Stat>> = std::thread::scope(|sc| {
        let hs: Vec<_> = (0..threads)
            .map(|t| sc.spawn(move || powf_ulp_sweep(each, 0xF00D ^ (t << 32))))
            .collect();
        hs.into_iter().map(|h| h.join().unwrap()).collect()
    });
    let env = format!(
        "os={} arch={} env={}",
        std::env::consts::OS,
        std::env::consts::ARCH,
        if cfg!(target_env = "gnu") {
            "gnu"
        } else if cfg!(target_env = "musl") {
            "musl"
        } else {
            "other"
        }
    );
    let mut worst = 0.0f64;
    for (s, name) in STRATA.iter().enumerate() {
        let (mut count, mut ov, mut mx, mut at) = (0, 0, 0.0f64, (0.0, 0.0));
        for p in &parts {
            count += p[s].count;
            ov += p[s].overflow;
            if p[s].max_ulp > mx {
                mx = p[s].max_ulp;
                at = p[s].max_at;
            }
        }
        worst = worst.max(mx);
        println!(
            "POWF_ULP {env} 層={name} 件数={count} 最大ulp={mx:.4} overflow={ov} 入力(x,e)={at:?}"
        );
    }
    assert!(
        worst <= LIBM_F32_MAX_ULP,
        "最大 {worst} ulp が仮定 {LIBM_F32_MAX_ULP} を超えた（定数を上げず #4084 へ報告する。{env}）"
    );
}

// ---- 環境に依らない決定的なテスト ----

fn is_rejected(l: Lin) -> bool {
    !l.bound().is_finite()
}

fn plain(x: f64) -> Lin {
    lin_reset();
    Lin::k(x)
}

fn rej(m: &str) -> AutodiffError {
    AutodiffError::Backward(m.into())
}

#[test]
fn powe_and_pow_grad_stay_unbounded() {
    lin_reset();
    let x = Lin::k(2.0);
    assert!(is_rejected(x.powe(1.5)));
    assert!(is_rejected(x.pow_grad(1.5)));
    assert!(entry_check_lin(x.powe(1.5), 1.0, "powe", rej).is_err());
}

#[test]
fn powf_bounded_rejects_out_of_domain() {
    for x in [0.0, -1.0, f64::INFINITY, f64::NAN, 4.0e38] {
        assert!(is_rejected(plain(x).powf_bounded(2.0, 2.0)), "x={x}");
    }
    let x = plain(2.0);
    assert!(is_rejected(x.powf_bounded(f64::INFINITY, 1.0)));
    assert!(is_rejected(x.powf_bounded(1.0, f32::NAN)));
    assert!(is_rejected(
        plain(2.0).powf_bounded(127.9999999, 127.9999999)
    ));
    assert!(is_rejected(plain(2.0).powf_bounded(200.0, 200.0)));
}

#[test]
fn powf_bounded_nan_base_keeps_taint() {
    lin_reset();
    let l = Lin::k(f64::NAN).powf_bounded(2.0, 2.0);
    assert!(l.input_nan());
    assert!(entry_check_lin(l, f32::NAN, "nan", rej).is_ok());
}

#[test]
fn powf_bounded_rejects_large_amplification() {
    // 相殺で底の相対誤差 ρ を大きくした入力は、指数が大きくなくても拒否する（ρ > 1/2）。
    lin_reset();
    let tiny = Lin::k(1.0e7).plus(Lin::k(0.3)).minus(Lin::k(1.0e7));
    assert!(is_rejected(tiny.powf_bounded(1.5, 1.5)));
    // ρ が小さくても |e|·ρ が大きければ Z 超過で拒否する。
    lin_reset();
    let b = Lin::k(1.0).plus(Lin::k(0.1)).times(Lin::k(0.7));
    assert!(!is_rejected(b.powf_bounded(2.0, 2.0)));
    assert!(is_rejected(b.powf_bounded(1.0e5, 1.0e5)));
}

#[test]
fn powf_bounded_subnormal_result_gets_absolute_term() {
    let l = plain(2.0).powf_bounded(-148.0, -148.0);
    let floor = LIBM_F32_MAX_ULP * LIN_ULP_SUBNORMAL;
    assert!(l.bound() >= floor, "bound={:e}", l.bound());
    let s = black_box(2.0f32).powf(-148.0);
    assert!(entry_check_lin(l, s, "sub", rej).is_ok());
    let l = plain(2.0).powf_bounded(-160.0, -160.0);
    assert!(l.bound() >= floor && l.bound().is_finite());
}

#[test]
fn powf_bounded_exponent_mismatch_widens_bound() {
    let e = 1.0 / 3.0;
    let es = e as f32;
    let l = plain(10.0).powf_bounded(e, es);
    let de = (f64::from(es) - e).abs();
    assert!(l.bound() >= l.v.abs() * de * 10.0f64.ln());
}

#[test]
fn pow_grad_bounded_zero_exponent_is_constant_zero() {
    lin_reset();
    let g = Lin::k(3.0).pow_grad_bounded(0.0);
    assert_eq!((g.v, g.bound()), (0.0, 0.0));
    assert!(is_rejected(Lin::k(3.0).pow_grad_bounded(1e-60)));
}

#[test]
fn powf_bounded_shared_event_cancels() {
    // y / y は共有した丸めの打ち消しで、自身の除算の丸めだけが残る。
    lin_reset();
    let a = Lin::k(1.7).times(Lin::k(2.3));
    let y = a.powf_bounded(1.5, 1.5);
    let q = y.over(y);
    assert!(q.bound() <= 2.0 * LIN_U, "bound={:e}", q.bound());
}

// ---- soundness ----

fn slack() -> f64 {
    1.0 + 16.0 * lin_events() as f64 * LIN_U
}

/// 最大比 `dev/bound` の下限（上界が実測の偏差に比べて極端に膨らまないことの確認。実測を見て決めた値）。
const POWF_RATIO_FLOOR: f64 = 0.1;

assumption_env_test! {
    /// 層ごとの入力を直接 `Lin::powf_bounded` に通し、受理された全件で `|f32 − v| ≤ bound·(1 + 16·N·u)`。
    /// 上界の過大な膨張を検出するため、最大比 `dev/bound` の下限も課す。
    fn lin_powf_bound_covers_f32_deviation_strata() {
        let mut rng = Rng(0x4084_0001);
        let (mut accepted, mut ratio_max) = (0u64, 0.0f64);
        for (s, name) in STRATA.iter().enumerate() {
            let mut acc_s = 0u64;
            for _ in 0..SWEEP_N {
                let (x, e) = gen_case(s, &mut rng);
                if !(x > 0.0 && e.is_finite()) {
                    continue;
                }
                lin_reset();
                let l = Lin::k(f64::from(x)).powf_bounded(f64::from(e), e);
                if is_rejected(l) {
                    continue;
                }
                let f = black_box(x).powf(black_box(e));
                let dev = (f64::from(f) - l.v).abs();
                let b = l.bound() * slack();
                assert!(dev <= b, "層={} x={x} e={e} f32={f} dev={dev:e} bound={b:e}", name);
                accepted += 1;
                acc_s += 1;
                if b > 0.0 {
                    ratio_max = ratio_max.max(dev / b);
                }
            }
            assert!(acc_s > 100, "層={} 受理が少なすぎる {acc_s}", name);
        }
        assert!(accepted > 6 * SWEEP_N / 4, "受理 {accepted}");
        assert!(ratio_max >= POWF_RATIO_FLOOR, "最大比 {ratio_max} が下限 {POWF_RATIO_FLOOR} 未満（上界が過大）");
    }
}

/// 合成プログラムの結果 3 種（f32 構造・逆順和の f32・`Lin`）の突合。受理されたら true。
fn check_cover(label: &str, s: f32, srev: f32, l: Lin) -> bool {
    if is_rejected(l) {
        return false;
    }
    let b = l.bound() * slack();
    for (n, f) in [("f32", s), ("rev", srev)] {
        let dev = (f64::from(f) - l.v).abs();
        assert!(
            dev <= b,
            "{label} {n}: dev={dev:e} bound={b:e} f={f} v={}",
            l.v
        );
    }
    true
}

fn prog_sum_pow<T: Num>(x: f64, y: f64, e: f64) -> T {
    T::k(x).plus(T::k(y)).powf_bounded(e, e as f32)
}
fn prog_prod_pow<T: Num>(x: f64, y: f64, z: f64, e: f64) -> T {
    T::k(x)
        .times(T::k(y))
        .powf_bounded(e, e as f32)
        .times(T::k(z))
}
/// Triplet 形 `(Σ uᵢ^p)^(1/p)` と `(u/n)^(p−1)` の積。
fn prog_triplet<T: Num>(us: &[f64], p: f64) -> T {
    let p32 = p as f32;
    let terms: Vec<T> = us.iter().map(|&u| T::k(u).powf_bounded(p, p32)).collect();
    let s = T::total(&terms).powf_bounded(1.0 / p, 1.0 / p32);
    let g = T::k(us[0])
        .over(T::k(us.len() as f64))
        .powf_bounded(p - 1.0, p32 - 1.0);
    s.times(g)
}
/// LRN 形 `(k + α·Σx²)^(−β)` と、その VJP 係数。
fn prog_lrn<T: Num>(xs: &[f64], k: f64, alpha: f64, beta: f64, grad: bool) -> T {
    let sq: Vec<T> = xs.iter().map(|&x| T::k(x).times(T::k(x))).collect();
    let base = T::k(k).plus(T::k(alpha).times(T::total(&sq)));
    if grad {
        base.pow_grad_bounded(-beta)
    } else {
        base.powf_bounded(-beta, (-beta) as f32)
    }
}

macro_rules! run3 {
    ($label:expr, $f:ident ( $($a:expr),* )) => {{
        let s: f32 = $f::<f32>($($a),*);
        let r: F32Rev = $f::<F32Rev>($($a),*);
        lin_reset();
        let l: Lin = $f::<Lin>($($a),*);
        check_cover($label, s, r.0, l)
    }};
}

assumption_env_test! {
    /// 合成プログラム（相殺・Triplet 形・LRN 形・Z が上限に近い入力）で、f32 構造と逆順和が `Lin` の上界に収まる。
    fn lin_powf_bound_covers_composed_programs() {
        let mut rng = Rng(0x4084_0002);
        let (mut acc, mut tried) = (0u32, 0u32);
        for _ in 0..4000 {
            let x = rng.range(0.1, 50.0);
            let y = rng.range(0.1, 50.0);
            let z = rng.range(-3.0, 3.0);
            let e = rng.range(-3.0, 3.0);
            let n = 1 + (rng.next() % 6) as usize;
            let us: Vec<f64> = (0..n).map(|_| rng.range(0.05, 4.0)).collect();
            let p = [1.5, 2.0, 3.0, 2.7, 1.2, 0.7][(rng.next() % 6) as usize];
            let xs: Vec<f64> = (0..n).map(|_| rng.range(-3.0, 3.0)).collect();
            let (k, alpha, beta) = (rng.range(0.5, 3.0), rng.range(1e-4, 1.0), rng.range(0.25, 1.0));
            tried += 5;
            acc += u32::from(run3!("sum_pow", prog_sum_pow(x, y, e)));
            acc += u32::from(run3!("prod_pow", prog_prod_pow(x, y, z, e)));
            acc += u32::from(run3!("triplet", prog_triplet(&us, p)));
            acc += u32::from(run3!("lrn", prog_lrn(&xs, k, alpha, beta, false)));
            acc += u32::from(run3!("lrn_grad", prog_lrn(&xs, k, alpha, beta, true)));
        }
        assert!(acc > tried * 9 / 10, "受理 {acc}/{tried}");
        let mut near_z = 0;
        for e in [50.0, 200.0, 400.0, 800.0, 1500.0] {
            for _ in 0..200 {
                let x = rng.range(0.9995, 1.0005);
                let y = rng.range(0.0, 1.0e-4);
                near_z += u32::from(run3!("near_z", prog_sum_pow(x, y, e)));
            }
        }
        assert!(near_z > 100, "Z 近傍の受理 {near_z}");
    }
}
