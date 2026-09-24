//! `nn::init`（PyTorch `torch.nn.init.*` 相当。イシュー #2140）の統合
//! テスト。
//!
//! `nn::init` の public 関数群はプロセスグローバルな決定的 RNG
//! （`fandhe_ai_tensor_core::rng::manual_seed`）に従属するため、他の
//! 統合テストファイルとの競合はない（`cargo test` は統合テストファイル
//! ごとに別プロセス）が、本ファイル内のテスト同士は既定の並列実行で
//! 競合しうるため、ファイル局所 `Mutex` で直列化する
//! （`crates/facade/tests/rng_tensor_generation.rs` と同型のパターン）。
//!
//! 既存の個別シード API（`Linear::new(.., seed)` 等）との独立性は
//! `nn/init.rs` の単体テスト（`linear_new_is_unaffected_by_global_
//! manual_seed_state` 相当）で既に固定済みのため、本ファイルでは
//! 「`manual_seed` を変えても `Linear::new` の出力が変わらない」ことを
//! 改めて回帰確認する 1 件のみ追加する。

use std::sync::Mutex;

use fandhe_ai_autodiff::AutodiffError;
use fandhe_ai_autodiff::nn::init::{self, FanMode, Nonlinearity};
use fandhe_ai_autodiff::nn::{Conv2d, Linear};
use fandhe_ai_tensor_core::ShapeError;
use fandhe_ai_tensor_core::rng::manual_seed;

fn test_lock() -> &'static Mutex<()> {
    static LOCK: Mutex<()> = Mutex::new(());
    &LOCK
}

// ---------------------------------------------------------------------
// 決定性
// ---------------------------------------------------------------------

#[test]
fn uniform_is_deterministic_under_same_manual_seed() {
    let _guard = test_lock().lock().unwrap_or_else(|p| p.into_inner());

    manual_seed(11);
    let a = init::uniform(&[8, 8], -1.0, 1.0).unwrap();
    manual_seed(11);
    let b = init::uniform(&[8, 8], -1.0, 1.0).unwrap();
    assert_eq!(a.host_slice(), b.host_slice());

    manual_seed(12);
    let c = init::uniform(&[8, 8], -1.0, 1.0).unwrap();
    assert_ne!(a.host_slice(), c.host_slice());
}

#[test]
fn normal_is_deterministic_under_same_manual_seed() {
    let _guard = test_lock().lock().unwrap_or_else(|p| p.into_inner());

    manual_seed(21);
    let a = init::normal(&[4, 4], 0.0, 1.0).unwrap();
    manual_seed(21);
    let b = init::normal(&[4, 4], 0.0, 1.0).unwrap();
    assert_eq!(a.host_slice(), b.host_slice());
}

#[test]
fn linear_new_output_is_unaffected_by_global_manual_seed_state() {
    let _guard = test_lock().lock().unwrap_or_else(|p| p.into_inner());

    manual_seed(999);
    let before = Linear::new(8, 4, true, 7).unwrap();
    manual_seed(1);
    // `nn::init::*` 呼び出しでグローバル RNG を消費してから比較する。
    let _ = init::uniform(&[16], -1.0, 1.0).unwrap();
    let after = Linear::new(8, 4, true, 7).unwrap();

    assert_eq!(before.weight().host_slice(), after.weight().host_slice());
}

// ---------------------------------------------------------------------
// 分布・値域（固定 seed・大標本・粗い閾値。既存 `try_normal_init` の
// `normal_init_large_sample_has_roughly_standard_normal_statistics` と
// 同程度の粒度）
// ---------------------------------------------------------------------

#[test]
fn uniform_values_are_within_range() {
    let _guard = test_lock().lock().unwrap_or_else(|p| p.into_inner());
    manual_seed(3);
    let t = init::uniform(&[5000], -2.0, 3.0).unwrap();
    for &v in t.host_slice().iter() {
        assert!((-2.0..3.0).contains(&v), "out of range: {v}");
    }
}

#[test]
fn uniform_low_equals_high_is_constant() {
    let _guard = test_lock().lock().unwrap_or_else(|p| p.into_inner());
    manual_seed(3);
    let t = init::uniform(&[10], 1.5, 1.5).unwrap();
    for &v in t.host_slice().iter() {
        assert_eq!(v, 1.5);
    }
}

#[test]
fn uniform_rejects_low_greater_than_high() {
    let _guard = test_lock().lock().unwrap_or_else(|p| p.into_inner());
    let err = init::uniform(&[4], 1.0, 0.0).unwrap_err();
    assert!(matches!(err, AutodiffError::InvalidArgument(_)));
}

#[test]
fn normal_large_sample_has_roughly_expected_statistics() {
    let _guard = test_lock().lock().unwrap_or_else(|p| p.into_inner());
    manual_seed(5);
    let t = init::normal(&[20_000], 2.0, 0.5).unwrap();
    let data = t.host_slice();
    let n = data.len() as f64;
    let mean: f64 = data.iter().map(|&v| v as f64).sum::<f64>() / n;
    let var: f64 = data
        .iter()
        .map(|&v| {
            let d = v as f64 - mean;
            d * d
        })
        .sum::<f64>()
        / n;
    assert!((mean - 2.0).abs() < 0.05, "mean out of range: {mean}");
    assert!((var - 0.25).abs() < 0.05, "var out of range: {var}");
}

#[test]
fn normal_std_zero_returns_constant_mean_and_does_not_consume_rng() {
    let _guard = test_lock().lock().unwrap_or_else(|p| p.into_inner());
    manual_seed(9);
    let before = init::uniform(&[4], 0.0, 1.0).unwrap();
    manual_seed(9);
    let t = init::normal(&[4], 3.0, 0.0).unwrap();
    for &v in t.host_slice().iter() {
        assert_eq!(v, 3.0);
    }
    let after = init::uniform(&[4], 0.0, 1.0).unwrap();
    assert_eq!(before.host_slice(), after.host_slice());
}

#[test]
fn normal_rejects_negative_std() {
    let _guard = test_lock().lock().unwrap_or_else(|p| p.into_inner());
    let err = init::normal(&[4], 0.0, -1.0).unwrap_err();
    assert!(matches!(err, AutodiffError::InvalidArgument(_)));
}

#[test]
fn constant_fills_all_elements_and_does_not_consume_rng() {
    let _guard = test_lock().lock().unwrap_or_else(|p| p.into_inner());
    manual_seed(17);
    let before = init::uniform(&[4], 0.0, 1.0).unwrap();
    manual_seed(17);
    let t = init::constant(&[3, 3], 0.25).unwrap();
    for &v in t.host_slice().iter() {
        assert_eq!(v, 0.25);
    }
    let after = init::uniform(&[4], 0.0, 1.0).unwrap();
    assert_eq!(before.host_slice(), after.host_slice());
}

// ---------------------------------------------------------------------
// fan 計算・gain
// ---------------------------------------------------------------------

#[test]
fn calculate_fan_in_and_fan_out_rank2() {
    let (fan_in, fan_out) = init::calculate_fan_in_and_fan_out(&[8, 4]).unwrap();
    assert_eq!(fan_in, 4);
    assert_eq!(fan_out, 8);
}

#[test]
fn calculate_fan_in_and_fan_out_rank4_conv() {
    // [out_channels, in_channels, kh, kw]
    let (fan_in, fan_out) = init::calculate_fan_in_and_fan_out(&[16, 3, 3, 3]).unwrap();
    assert_eq!(fan_in, 3 * 3 * 3);
    assert_eq!(fan_out, 16 * 3 * 3);
}

#[test]
fn calculate_fan_in_and_fan_out_rejects_rank_below_2() {
    let err = init::calculate_fan_in_and_fan_out(&[8]).unwrap_err();
    assert!(matches!(
        err,
        AutodiffError::Shape(ShapeError::RankMismatch {
            expected: 2,
            actual: 1
        })
    ));
}

#[test]
fn calculate_gain_matches_known_values() {
    assert_eq!(init::calculate_gain(Nonlinearity::Linear), 1.0);
    assert_eq!(init::calculate_gain(Nonlinearity::Sigmoid), 1.0);
    assert!((init::calculate_gain(Nonlinearity::Tanh) - 5.0 / 3.0).abs() < 1e-6);
    assert!((init::calculate_gain(Nonlinearity::Relu) - std::f32::consts::SQRT_2).abs() < 1e-6);
    assert!((init::calculate_gain(Nonlinearity::Selu) - 0.75).abs() < 1e-6);
    let leaky = init::calculate_gain(Nonlinearity::LeakyRelu(0.0));
    assert!((leaky - std::f32::consts::SQRT_2).abs() < 1e-6);
}

// ---------------------------------------------------------------------
// xavier / kaiming
// ---------------------------------------------------------------------

#[test]
fn xavier_uniform_values_are_within_bound() {
    let _guard = test_lock().lock().unwrap_or_else(|p| p.into_inner());
    manual_seed(31);
    let shape = [64, 32];
    let (fan_in, fan_out) = init::calculate_fan_in_and_fan_out(&shape).unwrap();
    let bound = (6.0 / (fan_in + fan_out) as f32).sqrt();
    let t = init::xavier_uniform(&shape, 1.0).unwrap();
    for &v in t.host_slice().iter() {
        assert!(v.abs() <= bound + 1e-6, "out of bound: {v}");
    }
}

#[test]
fn xavier_normal_has_expected_variance() {
    let _guard = test_lock().lock().unwrap_or_else(|p| p.into_inner());
    manual_seed(32);
    let shape = [200, 200];
    let (fan_in, fan_out) = init::calculate_fan_in_and_fan_out(&shape).unwrap();
    let expected_std = (2.0 / (fan_in + fan_out) as f32).sqrt();
    let t = init::xavier_normal(&shape, 1.0).unwrap();
    let data = t.host_slice();
    let n = data.len() as f64;
    let var: f64 = data.iter().map(|&v| (v as f64).powi(2)).sum::<f64>() / n;
    assert!(
        (var.sqrt() - expected_std as f64).abs() < 0.02,
        "std mismatch: {} vs {}",
        var.sqrt(),
        expected_std
    );
}

#[test]
fn xavier_uniform_rejects_negative_gain() {
    let _guard = test_lock().lock().unwrap_or_else(|p| p.into_inner());
    // レビュー指摘（PR #2239・init.rs:460）: 負の gain は bound が非負区間の
    // 半幅であるという前提を壊すため拒否する。RNG を消費していないことも
    // 併せて確認する（`normal_std_zero_returns_constant_mean_and_does_not_
    // consume_rng` と同型のパターン）。
    manual_seed(33);
    let before = init::uniform(&[4], 0.0, 1.0).unwrap();
    manual_seed(33);
    let err = init::xavier_uniform(&[8, 8], -1.0).unwrap_err();
    assert!(matches!(err, AutodiffError::InvalidArgument(_)));
    let after = init::uniform(&[4], 0.0, 1.0).unwrap();
    assert_eq!(before.host_slice(), after.host_slice());
}

#[test]
fn xavier_normal_rejects_negative_gain() {
    let _guard = test_lock().lock().unwrap_or_else(|p| p.into_inner());
    manual_seed(34);
    let before = init::uniform(&[4], 0.0, 1.0).unwrap();
    manual_seed(34);
    let err = init::xavier_normal(&[8, 8], -1.0).unwrap_err();
    assert!(matches!(err, AutodiffError::InvalidArgument(_)));
    let after = init::uniform(&[4], 0.0, 1.0).unwrap();
    assert_eq!(before.host_slice(), after.host_slice());
}

#[test]
fn xavier_uniform_zero_gain_returns_zeros() {
    // gain == 0（負ではない境界値）は受理され、bound == 0 のため全要素 0
    // になることを固定する。
    let _guard = test_lock().lock().unwrap_or_else(|p| p.into_inner());
    manual_seed(35);
    let t = init::xavier_uniform(&[8, 8], 0.0).unwrap();
    for &v in t.host_slice().iter() {
        assert_eq!(v, 0.0);
    }
}

#[test]
fn kaiming_uniform_values_are_within_bound() {
    let _guard = test_lock().lock().unwrap_or_else(|p| p.into_inner());
    manual_seed(41);
    let shape = [64, 32];
    let (fan_in, _fan_out) = init::calculate_fan_in_and_fan_out(&shape).unwrap();
    let gain = init::calculate_gain(Nonlinearity::Relu);
    let std = gain / (fan_in as f32).sqrt();
    let bound = std * 3f32.sqrt();
    let t = init::kaiming_uniform(&shape, 0.0, FanMode::FanIn, Nonlinearity::Relu).unwrap();
    for &v in t.host_slice().iter() {
        assert!(v.abs() <= bound + 1e-6, "out of bound: {v}");
    }
}

#[test]
fn kaiming_normal_fan_out_mode_uses_fan_out() {
    let _guard = test_lock().lock().unwrap_or_else(|p| p.into_inner());
    manual_seed(42);
    let shape = [64, 32];
    let (_fan_in, fan_out) = init::calculate_fan_in_and_fan_out(&shape).unwrap();
    let gain = init::calculate_gain(Nonlinearity::Relu);
    let expected_std = gain / (fan_out as f32).sqrt();
    let t = init::kaiming_normal(&shape, 0.0, FanMode::FanOut, Nonlinearity::Relu).unwrap();
    let data = t.host_slice();
    let n = data.len() as f64;
    let var: f64 = data.iter().map(|&v| (v as f64).powi(2)).sum::<f64>() / n;
    assert!(
        (var.sqrt() - expected_std as f64).abs() < 0.03,
        "std mismatch: {} vs {}",
        var.sqrt(),
        expected_std
    );
}

#[test]
fn kaiming_rejects_zero_fan() {
    let _guard = test_lock().lock().unwrap_or_else(|p| p.into_inner());
    // in_features(=shape[1]) == 0 -> fan_in == 0
    let err = init::kaiming_uniform(&[4, 0], 0.0, FanMode::FanIn, Nonlinearity::Relu).unwrap_err();
    assert!(matches!(err, AutodiffError::InvalidArgument(_)));
}

/// PyTorch `kaiming_uniform_(tensor, a=..., nonlinearity='leaky_relu')` と
/// 同じ意味論で `a` が gain（ひいては bound）を決定することを確認する
/// （codex-review 指摘。PR #2239）。`nonlinearity` に `LeakyRelu(_)` を
/// 渡した場合、埋め込まれた負勾配ではなく `a` が採用される。
#[test]
fn kaiming_uniform_uses_a_as_leaky_relu_negative_slope() {
    let _guard = test_lock().lock().unwrap_or_else(|p| p.into_inner());
    manual_seed(43);
    let shape = [64, 32];
    let a = 0.2f32;
    let (fan_in, _fan_out) = init::calculate_fan_in_and_fan_out(&shape).unwrap();
    // `a` が gain へ反映される契約なので、期待 gain は `a` から直接計算する
    // （`LeakyRelu` に埋め込む値〈ここでは `0.0`。故意に `a` と不一致にし、
    // `a` 側が優先されることを検証する〉ではなく `a` が使われる）。
    let expected_gain = (2.0 / (1.0 + a * a)).sqrt();
    let expected_std = expected_gain / (fan_in as f32).sqrt();
    let expected_bound = expected_std * 3f32.sqrt();
    let t = init::kaiming_uniform(&shape, a, FanMode::FanIn, Nonlinearity::LeakyRelu(0.0)).unwrap();
    for &v in t.host_slice().iter() {
        assert!(
            v.abs() <= expected_bound + 1e-6,
            "out of bound（a が gain に反映されていない可能性）: {v}"
        );
    }
}

/// `nonlinearity` が `LeakyRelu` 以外（例: `Relu`）の場合、PyTorch と
/// 同じく `a` は gain 計算に一切影響しない（有限性のみ検証される）こと
/// を確認する。
#[test]
fn kaiming_normal_ignores_a_for_non_leaky_relu_nonlinearity() {
    let _guard = test_lock().lock().unwrap_or_else(|p| p.into_inner());
    manual_seed(44);
    let shape = [200, 200];
    let (_fan_in, fan_out) = init::calculate_fan_in_and_fan_out(&shape).unwrap();
    let expected_gain = init::calculate_gain(Nonlinearity::Relu);
    let expected_std = expected_gain / (fan_out as f32).sqrt();
    // a = 5.0（Relu の gain 計算には無関係な値）を渡しても std は変わらない。
    let t = init::kaiming_normal(&shape, 5.0, FanMode::FanOut, Nonlinearity::Relu).unwrap();
    let data = t.host_slice();
    let n = data.len() as f64;
    let var: f64 = data.iter().map(|&v| (v as f64).powi(2)).sum::<f64>() / n;
    assert!(
        (var.sqrt() - expected_std as f64).abs() < 0.03,
        "std mismatch（a が誤って gain に影響した可能性）: {} vs {}",
        var.sqrt(),
        expected_std
    );
}

// ---------------------------------------------------------------------
// orthogonal
// ---------------------------------------------------------------------

#[test]
fn orthogonal_produces_orthonormal_rows_when_rows_le_cols() {
    let _guard = test_lock().lock().unwrap_or_else(|p| p.into_inner());
    manual_seed(51);
    let (rows, cols) = (4usize, 8usize);
    let q = init::orthogonal(&[rows, cols], 1.0).unwrap();
    let data = q.host_slice();
    // Q Qᵀ ≈ I_rows（rows < cols のケース）。
    for i in 0..rows {
        for j in 0..rows {
            let mut dot = 0.0f64;
            for k in 0..cols {
                dot += data[i * cols + k] as f64 * data[j * cols + k] as f64;
            }
            let expected = if i == j { 1.0 } else { 0.0 };
            assert!(
                (dot - expected).abs() < 1e-3,
                "Q Qᵀ[{i},{j}] = {dot}, expected {expected}"
            );
        }
    }
}

#[test]
fn orthogonal_produces_orthonormal_columns_when_rows_ge_cols() {
    let _guard = test_lock().lock().unwrap_or_else(|p| p.into_inner());
    manual_seed(52);
    let (rows, cols) = (8usize, 4usize);
    let q = init::orthogonal(&[rows, cols], 1.0).unwrap();
    let data = q.host_slice();
    // Qᵀ Q ≈ I_cols（rows >= cols のケース）。
    for i in 0..cols {
        for j in 0..cols {
            let mut dot = 0.0f64;
            for k in 0..rows {
                dot += data[k * cols + i] as f64 * data[k * cols + j] as f64;
            }
            let expected = if i == j { 1.0 } else { 0.0 };
            assert!(
                (dot - expected).abs() < 1e-3,
                "Qᵀ Q[{i},{j}] = {dot}, expected {expected}"
            );
        }
    }
}

#[test]
fn orthogonal_scales_by_gain_squared() {
    let _guard = test_lock().lock().unwrap_or_else(|p| p.into_inner());
    manual_seed(53);
    let gain = 2.0f32;
    let q = init::orthogonal(&[6, 3], gain).unwrap();
    let data = q.host_slice();
    let mut dot = 0.0f64;
    for k in 0..6 {
        dot += (data[k * 3] as f64).powi(2);
    }
    assert!(
        (dot - (gain as f64).powi(2)).abs() < 1e-2,
        "column norm² = {dot}, expected {}",
        (gain as f64).powi(2)
    );
}

#[test]
fn orthogonal_reshapes_rank_above_2() {
    let _guard = test_lock().lock().unwrap_or_else(|p| p.into_inner());
    manual_seed(54);
    // [out_channels, in_channels, kh, kw] -> flatten [8, 3*2*2]
    let q = init::orthogonal(&[8, 3, 2, 2], 1.0).unwrap();
    assert_eq!(q.shape(), &[8, 3, 2, 2]);
}

#[test]
fn orthogonal_rejects_rank_below_2() {
    let err = init::orthogonal(&[8], 1.0).unwrap_err();
    assert!(matches!(
        err,
        AutodiffError::Shape(ShapeError::RankMismatch {
            expected: 2,
            actual: 1
        })
    ));
}

/// codex-review 指摘の回帰（イシュー #2140・PR #2239）: `orthogonal` は
/// 巨大だが算術上有効な shape（`checked_mul` はオーバーフローしない）を
/// 渡されても `Err` を返し、allocation panic／プロセス abort に至らない
/// ことを確認する。`rows * cols == 2^62` は `f32` 換算バイト数
/// （`2^62 * 4 == 2^64`）が `usize` の乗算そのもので折り返る規模であり、
/// `fill_normal`（既存の `try_alloc` 経由）内部の `Vec::try_reserve_exact`
/// が `Layout` 計算のみで `CapacityOverflow` を検出し、実際にアロケータを
/// 呼び出すことなく即座に `Err` を返す（実メモリ確保を一切試みないため
/// 環境依存性がなく、テストは高速かつ決定的に完了する）。`crate::eval::
/// linalg::qr` 側の `Mat::try_zeros`／`try_vec_zeroed` 等の新規フォール
/// ブル化は `mat_try_zeros_rejects_isize_overflowing_byte_size`／
/// `try_vec_zeroed_rejects_isize_overflowing_byte_size`
/// （`crates/autodiff/src/eval/linalg.rs`）で個別に固定済み——`qr` の
/// 入力段階で使う実データ `Tensor` を伴わずに確保可否だけを検証できる
/// のはこの 2 関数のみで、`orthogonal` 経由のエンドツーエンド呼び出し
/// では `fill_normal` が同じオーダーの shape で必ず先に `Err` を返す
/// ため `qr` 内部の新規分岐そのものへは到達しない。それでも `orthogonal`
/// 自体がパイプライン全体を通して panic せず `Err` へ収束することを
/// 固定する意味で本テストを維持する。
#[test]
fn orthogonal_rejects_huge_shape_without_panicking() {
    let err = init::orthogonal(&[1usize << 40, 1usize << 22], 1.0).unwrap_err();
    assert!(matches!(err, AutodiffError::InvalidArgument(_)));
}

// ---------------------------------------------------------------------
// trunc_normal
// ---------------------------------------------------------------------

#[test]
fn trunc_normal_all_elements_within_bounds() {
    let _guard = test_lock().lock().unwrap_or_else(|p| p.into_inner());
    manual_seed(61);
    let t = init::trunc_normal(&[2000], 0.0, 1.0, -0.5, 0.5).unwrap();
    for &v in t.host_slice().iter() {
        assert!((-0.5..=0.5).contains(&v), "out of range: {v}");
    }
}

#[test]
fn trunc_normal_rejects_a_greater_equal_b() {
    let err = init::trunc_normal(&[4], 0.0, 1.0, 1.0, 1.0).unwrap_err();
    assert!(matches!(err, AutodiffError::InvalidArgument(_)));
}

#[test]
fn trunc_normal_std_zero_within_range_returns_constant() {
    let _guard = test_lock().lock().unwrap_or_else(|p| p.into_inner());
    manual_seed(62);
    let t = init::trunc_normal(&[4], 0.2, 0.0, -1.0, 1.0).unwrap();
    for &v in t.host_slice().iter() {
        assert_eq!(v, 0.2);
    }
}

#[test]
fn trunc_normal_std_zero_mean_outside_range_is_rejected() {
    let err = init::trunc_normal(&[4], 5.0, 0.0, -1.0, 1.0).unwrap_err();
    assert!(matches!(err, AutodiffError::InvalidArgument(_)));
}

#[test]
fn trunc_normal_rejects_window_exceeding_attempt_cap() {
    // `[a, b] = [20, 21]` は `N(0, 1)` の裾のさらに外側で受理確率が
    // 実質ゼロのため、要素あたり試行上限
    // （`TRUNC_NORMAL_MAX_ATTEMPTS_PER_ELEMENT`）に達し fail-closed で
    // 打ち切られることを固定する（Low follow-up・`docs/
    // facade-nn-init-exposure-decision.md` §6）。要素数を小さく保ち
    // テスト時間を試行上限×要素数程度に抑える。
    let _guard = test_lock().lock().unwrap_or_else(|p| p.into_inner());
    manual_seed(63);
    let err = init::trunc_normal(&[4], 0.0, 1.0, 20.0, 21.0).unwrap_err();
    assert!(matches!(err, AutodiffError::InvalidArgument(_)));
}

// ---------------------------------------------------------------------
// 層互換（既存層コンストラクタとの互換確認。`with_init` は存在しないため
// `from_parameters` 経由で確認する。§計画「3.2 層との互換」参照）
// ---------------------------------------------------------------------

#[test]
fn linear_from_parameters_accepts_kaiming_normal_weight_and_zero_bias() {
    let _guard = test_lock().lock().unwrap_or_else(|p| p.into_inner());
    manual_seed(71);
    // `nn::Linear::weight` は `[in_features, out_features]`
    // （PyTorch `nn.Linear.weight` の `[out_features, in_features]` とは
    // 転置の関係にある。`calculate_fan_in_and_fan_out` は PyTorch と
    // 同じ `[out, in, ..]` 規約を前提とするため、fandhe の `Linear`
    // レイアウトへ適用する場合は fan の意味が入れ替わる点に注意
    // ——本テストは forward が通ることの確認に限り、fan 方向の
    // 統計的な正しさは xavier/kaiming の専用テストで別途検証済み）。
    let weight = init::kaiming_normal(&[8, 4], 0.0, FanMode::FanIn, Nonlinearity::Relu).unwrap();
    let bias = init::constant(&[4], 0.0).unwrap();
    let linear = Linear::from_parameters(weight, Some(bias)).unwrap();

    let tape = fandhe_ai_autodiff::Tape::new();
    let x = tape.var(&fandhe_ai_tensor_core::Tensor::new(vec![1.0; 8], &[1, 8]).unwrap());
    let y = linear.bind(&tape).forward(&x).unwrap();
    assert_eq!(y.to_tensor().shape(), &[1, 4]);
}

#[test]
fn conv2d_from_parameters_accepts_kaiming_uniform_weight() {
    let _guard = test_lock().lock().unwrap_or_else(|p| p.into_inner());
    manual_seed(72);
    // [out_channels=4, in_channels=2, kh=3, kw=3]
    let weight =
        init::kaiming_uniform(&[4, 2, 3, 3], 0.0, FanMode::FanIn, Nonlinearity::Relu).unwrap();
    let conv = Conv2d::from_parameters(weight, None, [1, 1], [0, 0], [1, 1], 1).unwrap();

    let tape = fandhe_ai_autodiff::Tape::new();
    let x =
        tape.var(&fandhe_ai_tensor_core::Tensor::new(vec![1.0; 2 * 5 * 5], &[1, 2, 5, 5]).unwrap());
    let y = conv.bind(&tape).forward(&x).unwrap();
    assert_eq!(y.to_tensor().shape(), &[1, 4, 3, 3]);
}
