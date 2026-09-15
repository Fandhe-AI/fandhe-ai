//! イシュー #1736: BatchNorm1d／2d 順伝播カーネル（MSL・soft-f64・
//! persistent simdgroup〈train〉・単純 elementwise〈infer〉）の
//! CPU-Metal 数値一致検証。
//!
//! `layer_norm_parity.rs`（#1596）と同じ構成方針を踏襲する: Metal
//! 実機（Apple Silicon）依存のため `#![cfg(target_os = "macos")]` で
//! ファイル全体を macOS 限定にし、各テストに `#[ignore]` を付けて
//! 通常 CI では実行しない。判定式・許容誤差は再定義せず
//! `fandhe_ai_backend_cpu::parity` を唯一の参照とする（`.claude/rules/
//! coding-rust.md`）。CPU との bit 一致は主張しない（`shaders/
//! batch_norm.metal` 冒頭コメント「REQ-2 判定契約」参照——soft-f64 の
//! `mul`/`add` による二重丸め・Newton-Raphson `rsqrt` のため CPU の
//! `f64` FMA・ハードウェア `sqrt` とは bit 一致しない）。
//!
//! 実行コマンド（Mac 実機。`#[ignore]` テストのみ）:
//!
//! ```sh
//! cargo test -p fandhe-ai-backend-metal --release --test batch_norm_parity -- --ignored --nocapture
//! ```

#![cfg(target_os = "macos")]

use bench_harness::rng::Xorshift64Star;
use fandhe_ai_backend_cpu::parity::assert_parity;
use fandhe_ai_backend_metal::{MetalBatchNorm, MetalContext};
use fandhe_ai_tensor_core::{BackendOps, Tensor};

/// テスト専用 `f64` 参照実装（GPU の soft-f64・CPU の `f64` 逐次和の
/// いずれとも独立した実装で突き合わせることで、両実装共通のバグを
/// 検出できるようにする。`layer_norm_parity.rs::
/// f64_layer_norm_reference` のチャネル版）。
fn f64_batch_norm_train_reference(
    x: &[f32],
    w: Option<&[f32]>,
    b: Option<&[f32]>,
    eps: f32,
    n: usize,
    c: usize,
    spatial: usize,
) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let mut out = vec![0.0f32; x.len()];
    let mut mean_out = vec![0.0f32; c];
    let mut var_out = vec![0.0f32; c];
    if n == 0 || c == 0 || spatial == 0 {
        return (out, mean_out, var_out);
    }
    let m = n * spatial;
    let idx = |i: usize, ch: usize| -> usize {
        let batch = i / spatial;
        let sp = i % spatial;
        batch * (c * spatial) + ch * spatial + sp
    };
    for ch in 0..c {
        let mean: f64 = (0..m).map(|i| x[idx(i, ch)] as f64).sum::<f64>() / m as f64;
        let var: f64 = (0..m)
            .map(|i| (x[idx(i, ch)] as f64 - mean).powi(2))
            .sum::<f64>()
            / m as f64;
        let rstd = 1.0f64 / (var + eps as f64).sqrt();
        mean_out[ch] = mean as f32;
        var_out[ch] = var as f32;
        let wv = w.map_or(1.0f32, |w| w[ch]);
        let bv = b.map_or(0.0f32, |b| b[ch]);
        for i in 0..m {
            let xhat = ((x[idx(i, ch)] as f64 - mean) * rstd) as f32;
            out[idx(i, ch)] = xhat.mul_add(wv, bv);
        }
    }
    (out, mean_out, var_out)
}

fn f64_batch_norm_infer_reference(
    x: &[f32],
    mean: &[f32],
    var: &[f32],
    w: Option<&[f32]>,
    b: Option<&[f32]>,
    eps: f32,
    n: usize,
    c: usize,
    spatial: usize,
) -> Vec<f32> {
    let mut out = vec![0.0f32; x.len()];
    if n == 0 || c == 0 || spatial == 0 {
        return out;
    }
    let idx = |i: usize, ch: usize| -> usize {
        let batch = i / spatial;
        let sp = i % spatial;
        batch * (c * spatial) + ch * spatial + sp
    };
    let m = n * spatial;
    for ch in 0..c {
        let mean_c = mean[ch] as f64;
        let rstd = 1.0f64 / (var[ch] as f64 + eps as f64).sqrt();
        let wv = w.map_or(1.0f32, |w| w[ch]);
        let bv = b.map_or(0.0f32, |b| b[ch]);
        for i in 0..m {
            let xhat = ((x[idx(i, ch)] as f64 - mean_c) * rstd) as f32;
            out[idx(i, ch)] = xhat.mul_add(wv, bv);
        }
    }
    out
}

#[allow(clippy::too_many_arguments)]
fn assert_train_parity(
    ctx: &MetalContext,
    bn: &MetalBatchNorm,
    seed_x: u64,
    seed_w: u64,
    seed_b: u64,
    n: usize,
    c: usize,
    spatial: usize,
    with_weight: bool,
    with_bias: bool,
    eps: f32,
) {
    let x_data = Xorshift64Star::new(seed_x).fill_vec(n * c * spatial);
    let w_data = with_weight.then(|| Xorshift64Star::new(seed_w).fill_vec(c));
    let b_data = with_bias.then(|| Xorshift64Star::new(seed_b).fill_vec(c));

    let gpu = bn
        .run_batch_norm_train_f32(
            ctx,
            &x_data,
            w_data.as_deref(),
            b_data.as_deref(),
            eps,
            n,
            c,
            spatial,
        )
        .expect("MetalBatchNorm::run_batch_norm_train_f32 must succeed on Metal-equipped runner");
    let (expected_out, expected_mean, expected_var) = f64_batch_norm_train_reference(
        &x_data,
        w_data.as_deref(),
        b_data.as_deref(),
        eps,
        n,
        c,
        spatial,
    );

    let label = format!(
        "batch_norm train n={n} c={c} spatial={spatial} with_weight={with_weight} with_bias={with_bias} eps={eps}"
    );
    assert_parity(&format!("{label} out"), &gpu.out, &expected_out);
    assert_parity(&format!("{label} mean"), &gpu.mean, &expected_mean);
    assert_parity(&format!("{label} var"), &gpu.var, &expected_var);
}

#[test]
#[ignore = "Metal 実機（Apple Silicon）依存。CI では実行しない"]
fn batch_norm_train_matches_f64_reference_across_shapes_and_affine_combinations() {
    let ctx = MetalContext::new().expect("Metal デバイス・コマンドキューの初期化に失敗した");
    let bn = MetalBatchNorm::new(&ctx).expect("BatchNorm パイプラインの構築に失敗した");

    // `spatial` を変えて rank 2（spatial=1）・rank 3／4（spatial>1）を
    // 網羅する。`M = n*spatial` が 32 の倍数でない・M=1（単一要素縮約）
    // ・大形状（4097 は LayerNorm の `ONEPASS_MAX_HIDDEN` 相当の境界
    // 意識だが BatchNorm には該当しない——単に十分大きい形状として
    // 含める）を含む。
    let cases: &[(usize, usize, usize)] = &[
        (1, 3, 1),    // M=1（n*spatial=1）
        (2, 3, 1),    // rank2 相当・M=2
        (3, 5, 4),    // M=12（32 の倍数でない）
        (4, 3, 8),    // M=32 ちょうど
        (5, 2, 7),    // M=35（32 の倍数でない）
        (2, 3, 2049), // M=4098（大形状）
    ];
    let mut seed = 3000u64;
    for &(n, c, spatial) in cases {
        for with_weight in [false, true] {
            for with_bias in [false, true] {
                seed += 1;
                assert_train_parity(
                    &ctx,
                    &bn,
                    seed,
                    seed + 500,
                    seed + 900,
                    n,
                    c,
                    spatial,
                    with_weight,
                    with_bias,
                    1e-5,
                );
            }
        }
    }
}

#[test]
#[ignore = "Metal 実機（Apple Silicon）依存。CI では実行しない"]
fn batch_norm_infer_matches_f64_reference() {
    let ctx = MetalContext::new().expect("Metal デバイス・コマンドキューの初期化に失敗した");
    let bn = MetalBatchNorm::new(&ctx).expect("BatchNorm パイプラインの構築に失敗した");

    let (n, c, spatial) = (4usize, 3usize, 5usize);
    let x_data = Xorshift64Star::new(4100).fill_vec(n * c * spatial);
    let mean_data = Xorshift64Star::new(4200).fill_vec(c);
    // running_var は非負であるべきなので絶対値を取る。
    let var_data: Vec<f32> = Xorshift64Star::new(4300)
        .fill_vec(c)
        .into_iter()
        .map(f32::abs)
        .collect();
    let w_data = Xorshift64Star::new(4400).fill_vec(c);
    let b_data = Xorshift64Star::new(4500).fill_vec(c);
    let eps = 1e-5f32;

    let gpu_out = bn
        .run_batch_norm_infer_f32(
            &ctx,
            &x_data,
            &mean_data,
            &var_data,
            Some(&w_data),
            Some(&b_data),
            eps,
            n,
            c,
            spatial,
        )
        .expect("MetalBatchNorm::run_batch_norm_infer_f32 must succeed on Metal-equipped runner");
    let expected = f64_batch_norm_infer_reference(
        &x_data,
        &mean_data,
        &var_data,
        Some(&w_data),
        Some(&b_data),
        eps,
        n,
        c,
        spatial,
    );

    assert_parity("batch_norm infer n=4 c=3 spatial=5", &gpu_out, &expected);
}

#[test]
#[ignore = "Metal 実機（Apple Silicon）依存。CI では実行しない"]
fn batch_norm_train_extreme_values_stay_finite() {
    let ctx = MetalContext::new().expect("Metal デバイス・コマンドキューの初期化に失敗した");
    let bn = MetalBatchNorm::new(&ctx).expect("BatchNorm パイプラインの構築に失敗した");

    let (n, c, spatial) = (2usize, 2usize, 4usize);
    // チャネル 0: 極端に大きい正負が混在。チャネル 1: 通常値。
    let x = vec![
        2e20f32, -2e20f32, 1e-4, -1e-4, 3.0, -1.0, 0.5, -0.5, // batch 0
        1e19f32, -1e19f32, 2e-4, -2e-4, 1.0, -2.0, 0.3, -0.3, // batch 1
    ];
    let result = bn
        .run_batch_norm_train_f32(&ctx, &x, None, None, 1e-5, n, c, spatial)
        .expect("must succeed");
    for (i, &v) in result.out.iter().enumerate() {
        assert!(v.is_finite(), "out[{i}] は有限であるべき: {v}");
    }
    for (ch, (&mean, &var)) in result.mean.iter().zip(result.var.iter()).enumerate() {
        assert!(mean.is_finite(), "mean[{ch}] は有限であるべき: {mean}");
        assert!(var.is_finite(), "var[{ch}] は有限であるべき: {var}");
        assert!(var >= 0.0, "var[{ch}] は非負であるべき: {var}");
    }
}

#[test]
#[ignore = "Metal 実機（Apple Silicon）依存。CI では実行しない"]
fn batch_norm_train_nan_propagates_only_to_owning_channel() {
    let ctx = MetalContext::new().expect("Metal デバイス・コマンドキューの初期化に失敗した");
    let bn = MetalBatchNorm::new(&ctx).expect("BatchNorm パイプラインの構築に失敗した");

    let (n, c, spatial) = (2usize, 2usize, 2usize);
    // channel_index(i=0, ch=0, c=2, spatial=2) -> batch=0*(2*2)+0*2+0=0。
    let mut x = vec![1.0f32; n * c * spatial];
    x[0] = f32::NAN; // ch=0, batch=0, sp=0 のみ NaN。
    let result = bn
        .run_batch_norm_train_f32(&ctx, &x, None, None, 1e-5, n, c, spatial)
        .expect("must succeed");

    assert!(
        result.mean[0].is_nan(),
        "NaN を含むチャネル 0 の mean は NaN であるべき"
    );
    assert!(
        !result.mean[1].is_nan(),
        "NaN を含まないチャネル 1 の mean は NaN であってはならない: {}",
        result.mean[1]
    );
    let idx = |i: usize, ch: usize| -> usize {
        let batch = i / spatial;
        let sp = i % spatial;
        batch * (c * spatial) + ch * spatial + sp
    };
    for i in 0..(n * spatial) {
        assert!(
            !result.out[idx(i, 1)].is_nan(),
            "チャネル 1 の出力に NaN が伝播してはならない"
        );
    }
}

#[test]
#[ignore = "Metal 実機（Apple Silicon）依存。CI では実行しない"]
fn batch_norm_train_is_deterministic_across_runs() {
    let ctx = MetalContext::new().expect("Metal デバイス・コマンドキューの初期化に失敗した");
    let bn = MetalBatchNorm::new(&ctx).expect("BatchNorm パイプラインの構築に失敗した");

    let (n, c, spatial) = (3usize, 4usize, 5usize);
    let x = Xorshift64Star::new(5000).fill_vec(n * c * spatial);
    let w = Xorshift64Star::new(5100).fill_vec(c);
    let b = Xorshift64Star::new(5200).fill_vec(c);

    let first = bn
        .run_batch_norm_train_f32(&ctx, &x, Some(&w), Some(&b), 1e-5, n, c, spatial)
        .expect("must succeed");
    for _ in 0..4 {
        let repeat = bn
            .run_batch_norm_train_f32(&ctx, &x, Some(&w), Some(&b), 1e-5, n, c, spatial)
            .expect("must succeed");
        assert_eq!(
            first.out, repeat.out,
            "run-to-run で out が bit 単位で一致しない（決定性の回帰）"
        );
        assert_eq!(
            first.mean, repeat.mean,
            "run-to-run で mean が bit 単位で一致しない（決定性の回帰）"
        );
        assert_eq!(
            first.var, repeat.var,
            "run-to-run で var が bit 単位で一致しない（決定性の回帰）"
        );
    }
}

#[test]
#[ignore = "Metal 実機（Apple Silicon）依存。CI では実行しない"]
fn metal_backend_ops_batch_norm_train_matches_cpu_backend_ops_req2() {
    use fandhe_ai_backend_cpu::CpuBackendOps;
    use fandhe_ai_backend_metal::MetalBackendOps;

    let n = 4usize;
    let c = 3usize;
    let spatial = 5usize;
    let x_data = Xorshift64Star::new(6000).fill_vec(n * c * spatial);
    let w_data = Xorshift64Star::new(6100).fill_vec(c);
    let b_data = Xorshift64Star::new(6200).fill_vec(c);
    let x = Tensor::new(x_data, &[n, c, spatial]).expect("valid tensor");
    let w = Tensor::new(w_data, &[c]).expect("valid tensor");
    let b = Tensor::new(b_data, &[c]).expect("valid tensor");

    let cpu_ops = CpuBackendOps;
    let metal_ops = MetalBackendOps;

    let cpu_out = cpu_ops
        .batch_norm_train(&x, Some(&w), Some(&b), 1e-5)
        .expect("CpuBackendOps::batch_norm_train must succeed");
    let metal_out = metal_ops
        .batch_norm_train(&x, Some(&w), Some(&b), 1e-5)
        .expect("MetalBackendOps::batch_norm_train must succeed");

    assert_parity(
        "MetalBackendOps vs CpuBackendOps batch_norm_train out",
        metal_out
            .output
            .as_slice()
            .expect("output は contiguous のはず"),
        cpu_out
            .output
            .as_slice()
            .expect("output は contiguous のはず"),
    );
    assert_parity(
        "MetalBackendOps vs CpuBackendOps batch_norm_train mean",
        metal_out
            .batch_mean
            .as_slice()
            .expect("mean は contiguous のはず"),
        cpu_out
            .batch_mean
            .as_slice()
            .expect("mean は contiguous のはず"),
    );
    assert_parity(
        "MetalBackendOps vs CpuBackendOps batch_norm_train var",
        metal_out
            .batch_var
            .as_slice()
            .expect("var は contiguous のはず"),
        cpu_out
            .batch_var
            .as_slice()
            .expect("var は contiguous のはず"),
    );
}

#[test]
#[ignore = "Metal 実機（Apple Silicon）依存。CI では実行しない"]
fn batch_norm_train_accepts_non_contiguous_input_via_backend_ops() {
    use fandhe_ai_backend_metal::MetalBackendOps;

    // 転置で非連続にした後、`MetalBackendOps::batch_norm_train`
    // （呼び出し側で `.contiguous()` する契約。`ops.rs` 参照）へ渡す。
    let (n, c, spatial) = (4usize, 3usize, 1usize);
    let x_data = Xorshift64Star::new(7000).fill_vec(c * n);
    // 転置前 shape [c, n] を作り permute で [n, c] へ（非連続）。
    let x_t = Tensor::new(x_data, &[c, n]).expect("valid tensor");
    let x_noncontig = x_t.permute(&[1, 0]).expect("valid permute");
    assert!(
        x_noncontig.as_slice().is_none(),
        "テストの前提: permute 後は非連続であるべき"
    );

    let metal_ops = MetalBackendOps;
    let out = metal_ops
        .batch_norm_train(&x_noncontig, None, None, 1e-5)
        .expect("non-contiguous 入力でも成功するはず（内部で contiguous 化する）");
    assert_eq!(out.output.shape(), &[n, c]);
    let _ = spatial;
}

#[test]
#[ignore = "Metal 実機（Apple Silicon）依存。CI では実行しない"]
fn batch_norm_train_rejects_weight_length_mismatch() {
    let ctx = MetalContext::new().expect("Metal デバイス・コマンドキューの初期化に失敗した");
    let bn = MetalBatchNorm::new(&ctx).expect("BatchNorm パイプラインの構築に失敗した");

    let (n, c, spatial) = (2usize, 3usize, 1usize);
    let x = vec![0.0f32; n * c * spatial];
    let bad_w = vec![1.0f32; c + 1];
    let err = bn
        .run_batch_norm_train_f32(&ctx, &x, Some(&bad_w), None, 1e-5, n, c, spatial)
        .expect_err("weight 長さ不一致は拒否されるはず");
    assert!(
        matches!(
            err,
            fandhe_ai_backend_metal::MetalError::InvalidBatchNormShape { .. }
        ),
        "weight 長さ不一致は InvalidBatchNormShape であるべき: {err:?}"
    );
}

#[test]
#[ignore = "Metal 実機（Apple Silicon）依存。CI では実行しない"]
fn batch_norm_train_and_infer_empty_axis_return_without_touching_device() {
    // `n == 0 || c == 0 || spatial == 0` はカーネル起動なしで空出力・
    // ゼロ統計を返す早期 return 契約（`crate::batch_norm_model::
    // batch_norm_train_host_model` と同じ。Linux 単体テストの
    // `train_host_model_handles_zero_axis_early_return` と対の実機
    // 確認）。デバイス初期化自体は必要（`MetalBatchNorm::new`）だが、
    // カーネルディスパッチには到達しないことを間接的に確認する。
    let ctx = MetalContext::new().expect("Metal デバイス・コマンドキューの初期化に失敗した");
    let bn = MetalBatchNorm::new(&ctx).expect("BatchNorm パイプラインの構築に失敗した");

    let train = bn
        .run_batch_norm_train_f32(&ctx, &[], None, None, 1e-5, 0, 3, 4)
        .expect("n=0 は早期 return で成功するはず");
    assert!(train.out.is_empty());
    assert_eq!(train.mean, vec![0.0f32; 3]);
    assert_eq!(train.var, vec![0.0f32; 3]);

    let infer = bn
        .run_batch_norm_infer_f32(&ctx, &[], &[], &[], None, None, 1e-5, 0, 3, 4)
        .expect("n=0 は早期 return で成功するはず");
    assert!(infer.is_empty());
}
