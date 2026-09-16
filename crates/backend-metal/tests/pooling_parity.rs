//! イシュー #1730: MaxPool／AvgPool／AdaptiveAvgPool（MSL）の
//! CPU-Metal 数値一致検証。
//!
//! `batch_norm_parity.rs`（#1736）と同じ構成方針: Metal 実機
//! （Apple Silicon）依存のため `#![cfg(target_os = "macos")]` で
//! ファイル全体を macOS 限定にし、各テストに `#[ignore]` を付けて
//! 通常 CI では実行しない。**本 PR 時点では `MetalBackendOps` への
//! 結線（Layer B）が未実施のため**（`crate::pooling` モジュール doc
//! 参照）、参照実装は `fandhe_ai_backend_cpu`（CPU クレート）ではなく
//! `fandhe_ai_backend_metal::pooling_model`（本クレート内のホスト
//! 逐語モデル）を使う。Max は値・索引とも **bit 完全一致**、
//! Avg／Adaptive は soft-f64 が CPU `f64` 逐次和と bit 完全一致する
//!設計のため同じく **bit 完全一致**を判定基準とする（`shaders/
//! pooling.metal` 冒頭コメント「数値方式」参照）。
//!
//! 実行コマンド（Mac 実機。`#[ignore]` テストのみ）:
//!
//! ```sh
//! cargo test -p fandhe-ai-backend-metal --release --test pooling_parity -- --ignored --nocapture
//! ```

#![cfg(target_os = "macos")]

use bench_harness::rng::Xorshift64Star;
use fandhe_ai_backend_metal::pooling_model::{
    self, adaptive_avg_pool2d_soft_f64, avg_pool2d_soft_f64, max_pool2d_model,
};
use fandhe_ai_backend_metal::{MetalContext, MetalPooling};

/// max pooling parity のケース表 1 行（入力 shape・kernel・stride・
/// padding・dilation の順）。clippy `type_complexity` 是正の `type`
/// 定義（`#[allow]` で抑止しない方針 `.claude/rules/coding-rust.md`）。
type MaxPoolCase<'a> = (
    &'a [usize],
    (usize, usize),
    (usize, usize),
    (usize, usize),
    (usize, usize),
);

/// avg pooling parity のケース表 1 行（`MaxPoolCase` の末尾に
/// `count_include_pad` を加えたもの）。
type AvgPoolCase<'a> = (
    &'a [usize],
    (usize, usize),
    (usize, usize),
    (usize, usize),
    (usize, usize),
    bool,
);

fn gen_input(rng: &mut Xorshift64Star, numel: usize) -> Vec<f32> {
    (0..numel).map(|_| rng.next_f32() * 5.0).collect()
}

/// MaxPool の値・索引が run-to-run 決定的で `pooling_model` と
/// bit 完全一致することを、重なり窓・非正方・padding・dilation の
/// 組み合わせを網羅して検証する。
#[test]
#[ignore]
fn max_pool2d_bit_exact_against_host_model() {
    let ctx = MetalContext::new().expect("Metal context 構築に失敗");
    let pooling = MetalPooling::new(&ctx).expect("MetalPooling 構築に失敗");
    let mut rng = Xorshift64Star::new(0x1730_0001);

    let cases: &[MaxPoolCase] = &[
        (&[2, 3, 5, 5], (2, 2), (2, 2), (0, 0), (1, 1)),
        (&[2, 3, 5, 5], (3, 3), (1, 1), (1, 1), (1, 1)),
        (&[1, 2, 7, 4], (3, 2), (2, 1), (1, 0), (1, 1)),
        (&[1, 1, 8, 8], (2, 2), (2, 2), (0, 0), (1, 1)),
        (&[3, 4, 1, 10], (1, 2), (1, 2), (0, 0), (1, 1)), // 1d 併合形状
        (&[1, 1, 9, 9], (3, 3), (1, 1), (1, 1), (2, 2)),
    ];

    for &(shape, kernel, stride, padding, dilation) in cases {
        let numel: usize = shape.iter().product();
        let x = gen_input(&mut rng, numel);

        let dims =
            pooling_model::derive_pool_dims(shape, kernel, stride, padding, dilation, false, true)
                .expect("derive_pool_dims failed");
        let (want_out, want_idx) = max_pool2d_model(&x, &dims).expect("host model failed");

        let (got_out, got_idx) = pooling
            .run_max_pool2d_f32(&ctx, &x, shape, kernel, stride, padding, dilation)
            .expect("run_max_pool2d_f32 failed");

        assert_eq!(got_idx, want_idx, "shape={shape:?}: index mismatch");
        assert_eq!(got_out.len(), want_out.len());
        for (i, (&g, &w)) in got_out.iter().zip(want_out.iter()).enumerate() {
            if w.is_nan() {
                assert!(g.is_nan(), "shape={shape:?} i={i}: expected NaN, got {g}");
            } else {
                assert_eq!(
                    g.to_bits(),
                    w.to_bits(),
                    "shape={shape:?} i={i}: {g} != {w}"
                );
            }
        }

        // run-to-run 決定性。
        let (got_out2, got_idx2) = pooling
            .run_max_pool2d_f32(&ctx, &x, shape, kernel, stride, padding, dilation)
            .expect("run_max_pool2d_f32 (2nd run) failed");
        assert_eq!(got_idx, got_idx2);
        for (a, b) in got_out.iter().zip(got_out2.iter()) {
            if a.is_nan() {
                assert!(b.is_nan());
            } else {
                assert_eq!(a.to_bits(), b.to_bits());
            }
        }
    }
}

/// MaxPool の特殊値契約（NaN 最初の索引で固定・全 `-inf` 窓・タイの
/// 先勝ち）を実機で検証する。
#[test]
#[ignore]
fn max_pool2d_special_values_bit_exact() {
    let ctx = MetalContext::new().expect("Metal context 構築に失敗");
    let pooling = MetalPooling::new(&ctx).expect("MetalPooling 構築に失敗");

    // タイの先勝ち。
    {
        let shape = [1usize, 1, 1, 4];
        let x = vec![5.0f32, 5.0, 5.0, 1.0];
        let (out, idx) = pooling
            .run_max_pool2d_f32(&ctx, &x, &shape, (1, 4), (1, 1), (0, 0), (1, 1))
            .unwrap();
        assert_eq!(out, vec![5.0]);
        assert_eq!(idx, vec![0]);
    }

    // 最初の NaN で固定。
    {
        let shape = [1usize, 1, 1, 4];
        let x = vec![1.0f32, f32::NAN, f32::NAN, 9.0];
        let (out, idx) = pooling
            .run_max_pool2d_f32(&ctx, &x, &shape, (1, 4), (1, 1), (0, 0), (1, 1))
            .unwrap();
        assert!(out[0].is_nan());
        assert_eq!(idx, vec![1]);
    }

    // 全 -inf 窓。
    {
        let shape = [1usize, 1, 1, 3];
        let x = vec![f32::NEG_INFINITY; 3];
        let (out, idx) = pooling
            .run_max_pool2d_f32(&ctx, &x, &shape, (1, 3), (1, 1), (0, 0), (1, 1))
            .unwrap();
        assert_eq!(out, vec![f32::NEG_INFINITY]);
        assert_eq!(idx, vec![0]);
    }
}

/// AvgPool の soft-f64 縮約が `pooling_model` の soft-f64 参照と
/// bit 完全一致することを、`count_include_pad` 両値・相殺列込みで
/// 検証する。
#[test]
#[ignore]
fn avg_pool2d_bit_exact_against_host_model() {
    let ctx = MetalContext::new().expect("Metal context 構築に失敗");
    let pooling = MetalPooling::new(&ctx).expect("MetalPooling 構築に失敗");
    let mut rng = Xorshift64Star::new(0x1730_0002);

    let cases: &[AvgPoolCase] = &[
        (&[2, 3, 5, 5], (3, 3), (2, 2), (1, 1), (1, 1), true),
        (&[2, 3, 5, 5], (3, 3), (2, 2), (1, 1), (1, 1), false),
        (&[1, 2, 7, 4], (3, 2), (1, 1), (1, 0), (1, 1), true),
        (&[3, 4, 1, 10], (1, 2), (1, 2), (0, 0), (1, 1), true), // 1d 併合形状
    ];

    for &(shape, kernel, stride, padding, dilation, count_include_pad) in cases {
        let numel: usize = shape.iter().product();
        let x = gen_input(&mut rng, numel);

        let dims = pooling_model::derive_pool_dims(
            shape,
            kernel,
            stride,
            padding,
            dilation,
            false,
            count_include_pad,
        )
        .expect("derive_pool_dims failed");
        let want = avg_pool2d_soft_f64(&x, &dims).expect("host model failed");

        let got = pooling
            .run_avg_pool2d_f32(
                &ctx,
                &x,
                shape,
                kernel,
                stride,
                padding,
                dilation,
                count_include_pad,
            )
            .expect("run_avg_pool2d_f32 failed");

        assert_eq!(got.len(), want.len());
        for (i, (&g, &w)) in got.iter().zip(want.iter()).enumerate() {
            assert_eq!(
                g.to_bits(),
                w.to_bits(),
                "shape={shape:?} count_include_pad={count_include_pad} i={i}: {g} != {w}"
            );
        }
    }

    // 相殺列（対消滅）: f32 単純和では丸め誤差が乗るが soft-f64 は
    // ホスト f64 参照と一致する。
    {
        let shape = [1usize, 1, 1, 4];
        let x = vec![2f32.powi(24), 1.0, -(2f32.powi(24)), 1.0];
        let dims =
            pooling_model::derive_pool_dims(&shape, (1, 4), (1, 1), (0, 0), (1, 1), false, true)
                .unwrap();
        let want = avg_pool2d_soft_f64(&x, &dims).unwrap();
        let got = pooling
            .run_avg_pool2d_f32(&ctx, &x, &shape, (1, 4), (1, 1), (0, 0), (1, 1), true)
            .unwrap();
        assert_eq!(got[0].to_bits(), want[0].to_bits());
    }
}

/// AdaptiveAvgPool の soft-f64 縮約が `pooling_model` の soft-f64
/// 参照と bit 完全一致することを、`output_size` が入力長以下／超過
/// する両ケースで検証する。
#[test]
#[ignore]
fn adaptive_avg_pool2d_bit_exact_against_host_model() {
    let ctx = MetalContext::new().expect("Metal context 構築に失敗");
    let pooling = MetalPooling::new(&ctx).expect("MetalPooling 構築に失敗");
    let mut rng = Xorshift64Star::new(0x1730_0003);

    let cases: &[(&[usize], (usize, usize))] = &[
        (&[1, 2, 5, 5], (3, 3)),
        (&[2, 3, 7, 4], (2, 2)),
        (&[1, 1, 2, 2], (4, 4)), // output_size > input（拡大方向）
    ];

    for &(shape, output_size) in cases {
        let numel: usize = shape.iter().product();
        let x = gen_input(&mut rng, numel);

        let dims = pooling_model::derive_adaptive_dims(shape, output_size).expect("derive failed");
        let want = adaptive_avg_pool2d_soft_f64(&x, &dims).expect("host model failed");

        let got = pooling
            .run_adaptive_avg_pool2d_f32(&ctx, &x, shape, output_size)
            .expect("run_adaptive_avg_pool2d_f32 failed");

        assert_eq!(got.len(), want.len());
        for (i, (&g, &w)) in got.iter().zip(want.iter()).enumerate() {
            assert_eq!(
                g.to_bits(),
                w.to_bits(),
                "shape={shape:?} output_size={output_size:?} i={i}: {g} != {w}"
            );
        }
    }
}

/// `N == 0` の入力が device 非接触で空 `Vec` を返すことを検証する
/// （`MetalPooling::run_*` の早期リターン契約）。
#[test]
#[ignore]
fn zero_batch_returns_empty_without_device_dispatch() {
    let ctx = MetalContext::new().expect("Metal context 構築に失敗");
    let pooling = MetalPooling::new(&ctx).expect("MetalPooling 構築に失敗");
    let shape = [0usize, 2, 4, 4];
    let x: Vec<f32> = Vec::new();

    let (out, idx) = pooling
        .run_max_pool2d_f32(&ctx, &x, &shape, (2, 2), (2, 2), (0, 0), (1, 1))
        .unwrap();
    assert!(out.is_empty());
    assert!(idx.is_empty());

    let avg = pooling
        .run_avg_pool2d_f32(&ctx, &x, &shape, (2, 2), (2, 2), (0, 0), (1, 1), true)
        .unwrap();
    assert!(avg.is_empty());

    let adaptive = pooling
        .run_adaptive_avg_pool2d_f32(&ctx, &x, &shape, (2, 2))
        .unwrap();
    assert!(adaptive.is_empty());
}

/// `PoolingSizeLimitExceeded`（`u32` 上限超過）が型付きエラーとして
/// 伝播することを検証する（偽の巨大形状で `derive_pool_dims` を
/// 到達させる）。
#[test]
#[ignore]
fn oversized_shape_returns_size_limit_exceeded_error() {
    let ctx = MetalContext::new().expect("Metal context 構築に失敗");
    let pooling = MetalPooling::new(&ctx).expect("MetalPooling 構築に失敗");

    // plane_in = h_in * w_in > i32::MAX（MaxPool 索引契約超過）。
    let shape = [1usize, 1, 1 << 16, 1 << 16];
    let x: Vec<f32> = Vec::new();
    let err = pooling
        .run_max_pool2d_f32(&ctx, &x, &shape, (1, 1), (1, 1), (0, 0), (1, 1))
        .unwrap_err();
    assert!(
        matches!(
            err,
            fandhe_ai_backend_metal::MetalError::PoolingSizeLimitExceeded { .. }
        ),
        "expected PoolingSizeLimitExceeded, got {err:?}"
    );
}
