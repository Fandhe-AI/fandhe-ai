//! イシュー #1895: `crate::reduce::MetalReduce`（f32 `sum` reduction。
//! 全要素・単一軸）の CPU-Metal 数値一致検証。
//!
//! `MetalBackendOps::sum` への結線はイシュー #1896 で完了済み。本
//! ファイルは結線後も起動 API 直叩き（`MetalReduce::run_sum_all_f32`／
//! `run_sum_axis_f32`）の検証として維持する（`dispatch_boundary.rs` と
//! 同じ「起動 API 直接構築」方針）。`BackendOps` 経由（0 サイズ契約・
//! 検査順序込み）の検証は `tests/backend_ops_real_device.rs::
//! backend_ops_sum_matches_cpu_bit_exact` が担う。
//!
//! macOS 実機（Apple Silicon）でのみコンパイル・実行する
//! （`scan_parity.rs` と同方針。`#![cfg(target_os = "macos")]` により
//! Linux CI ではコンパイル対象外・`#[ignore]` により通常の `cargo test`
//! からも除外される）。
//!
//! **契約は bit 完全一致**（`fandhe_ai_backend_cpu::reduction::sum` と
//! 同一の演算順序を binary64 ソフトウェアエミュレーションで逐語再現。
//! `shaders/reduce.metal`／`crate::reduce_model` doc 参照）。
//!
//! Linux CI での型検査（実機なしでもコンパイル可能性を担保）:
//!
//! ```sh
//! cargo check -p fandhe-ai-backend-metal --tests --target aarch64-apple-darwin
//! ```
//!
//! 実行コマンド（Apple Silicon 実機。`--release` 推奨）:
//!
//! ```sh
//! cargo test -p fandhe-ai-backend-metal --release --test reduce_parity -- --ignored --nocapture
//! ```

#![cfg(target_os = "macos")]

use bench_harness::rng::Xorshift64Star;
use fandhe_ai_backend_metal::reduce::ArgExtKind;
use fandhe_ai_backend_metal::{MetalContext, MetalReduce};
use fandhe_ai_tensor_core::Tensor;

fn cpu_sum_all(x: &[f32]) -> f32 {
    let t = Tensor::new(x.to_vec(), &[x.len()]).expect("tensor");
    fandhe_ai_backend_cpu::reduction::sum(&t, None)
        .expect("cpu sum always succeeds")
        .as_slice()
        .expect("contiguous")[0]
}

fn cpu_sum_axis(x: &[f32], shape: &[usize], dim: usize) -> Vec<f32> {
    let t = Tensor::new(x.to_vec(), shape).expect("tensor");
    fandhe_ai_backend_cpu::reduction::sum(&t, Some(dim))
        .expect("cpu sum always succeeds")
        .as_slice()
        .expect("contiguous")
        .to_vec()
}

fn gen_data(n: usize, seed: u64) -> Vec<f32> {
    let mut rng = Xorshift64Star::new(seed);
    (0..n).map(|_| rng.next_f32() * 1024.0).collect()
}

/// [`MetalReduce::run_sum_all_f32`] が CPU 参照実装と bit 完全一致する
/// ことを複数サイズ（`REDUCE_SUM_CHUNK` 境界前後・0・大形状）で確認する。
#[test]
#[ignore = "Metal 実機（Apple Silicon）必須"]
fn metal_sum_all_matches_cpu_bit_exact() {
    let ctx = MetalContext::new().expect("Metal デバイス・コマンドキューの初期化に失敗した");
    let reduce = MetalReduce::new(&ctx).expect("MetalReduce::new に失敗した");

    for &n in &[0usize, 1, 4095, 4096, 4097, 8192, 3 * 4096 + 1, 1 << 20] {
        let data = gen_data(n, 0x1895_0000 + n as u64);
        let metal_out = reduce
            .run_sum_all_f32(&ctx, &data)
            .expect("metal sum must succeed on Metal-equipped runner");
        let cpu_out = cpu_sum_all(&data);
        assert_eq!(
            metal_out.to_bits(),
            cpu_out.to_bits(),
            "n={n}: bit 不一致（metal={metal_out:?}, cpu={cpu_out:?}）"
        );

        // run-to-run 決定性。
        let metal_out2 = reduce.run_sum_all_f32(&ctx, &data).expect("metal sum run2");
        assert_eq!(
            metal_out2.to_bits(),
            metal_out.to_bits(),
            "n={n}: run-to-run で bit 同一のはず"
        );
    }
}

/// [`MetalReduce::run_sum_axis_f32`] が CPU 参照実装と bit 完全一致する
/// ことを複数 rank・複数 `dim` で確認する。
#[test]
#[ignore = "Metal 実機（Apple Silicon）必須"]
fn metal_sum_axis_matches_cpu_bit_exact() {
    let ctx = MetalContext::new().expect("Metal デバイス・コマンドキューの初期化に失敗した");
    let reduce = MetalReduce::new(&ctx).expect("MetalReduce::new に失敗した");

    let cases: &[(&[usize], usize)] = &[
        (&[5], 0),
        (&[3, 4], 0),
        (&[3, 4], 1),
        (&[2, 3, 5], 0),
        (&[2, 3, 5], 1),
        (&[2, 3, 5], 2),
        (&[2, 3, 4, 2], 2),
        (&[1, 4097], 1),
        (&[4097, 1], 0),
    ];

    for &(shape, dim) in cases {
        let numel: usize = shape.iter().product();
        let data = gen_data(numel, 0x1895_1000 + numel as u64);
        let outer: usize = shape[..dim].iter().product();
        let axis_len = shape[dim];
        let inner: usize = shape[dim + 1..].iter().product();

        let metal_out = reduce
            .run_sum_axis_f32(&ctx, &data, outer, axis_len, inner)
            .expect("metal sum axis must succeed on Metal-equipped runner");
        let cpu_out = cpu_sum_axis(&data, shape, dim);

        assert_eq!(
            metal_out.len(),
            cpu_out.len(),
            "shape={shape:?} dim={dim}: 出力長不一致"
        );
        for (i, (&m, &c)) in metal_out.iter().zip(cpu_out.iter()).enumerate() {
            assert_eq!(
                m.to_bits(),
                c.to_bits(),
                "shape={shape:?} dim={dim} idx={i}: bit 不一致（metal={m:?}, cpu={c:?}）"
            );
        }

        // run-to-run 決定性。
        let metal_out2 = reduce
            .run_sum_axis_f32(&ctx, &data, outer, axis_len, inner)
            .expect("metal sum axis run2");
        assert_eq!(
            metal_out2.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            metal_out.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            "shape={shape:?} dim={dim}: run-to-run で bit 同一のはず"
        );
    }
}

/// 空縮約（`axis_len == 0`）・空出力（`outer == 0`）の早期リターン経路
/// が CPU 参照実装と一致することを確認する。
#[test]
#[ignore = "Metal 実機（Apple Silicon）必須"]
fn metal_sum_axis_empty_cases_match_cpu() {
    let ctx = MetalContext::new().expect("Metal デバイス・コマンドキューの初期化に失敗した");
    let reduce = MetalReduce::new(&ctx).expect("MetalReduce::new に失敗した");

    // axis_len=0・outer=1・inner=3 → 3 出力要素、値すべて 0.0。
    let out = reduce
        .run_sum_axis_f32(&ctx, &[], 1, 0, 3)
        .expect("metal sum axis (empty axis_len)");
    assert_eq!(out, vec![0.0f32; 3]);

    // outer=0（空出力）。
    let out_empty = reduce
        .run_sum_axis_f32(&ctx, &[], 0, 5, 3)
        .expect("metal sum axis (empty outer)");
    assert!(out_empty.is_empty());
}

// ---- argmax／argmin（イシュー #1951）----

fn cpu_argext(x: &[f32], shape: &[usize], dim: Option<usize>, kind: ArgExtKind) -> Vec<i32> {
    let t = Tensor::new(x.to_vec(), shape).expect("tensor");
    let out = match kind {
        ArgExtKind::Max => fandhe_ai_backend_cpu::reduction::argmax(&t, dim),
        ArgExtKind::Min => fandhe_ai_backend_cpu::reduction::argmin(&t, dim),
    }
    .expect("cpu argext succeeds for non-empty reduction");
    out.as_slice().expect("contiguous").to_vec()
}

/// [`MetalReduce::run_arg_all_f32`] が CPU 参照実装（`fandhe_ai_backend_cpu::
/// reduction::{argmax, argmin}`）と各種サイズ（`REDUCE_SUM_CHUNK` 境界
/// 前後・大形状）で添字完全一致することを確認する。
#[test]
#[ignore = "Metal 実機（Apple Silicon）必須"]
fn metal_arg_all_matches_cpu_exact() {
    let ctx = MetalContext::new().expect("Metal デバイス・コマンドキューの初期化に失敗した");
    let reduce = MetalReduce::new(&ctx).expect("MetalReduce::new に失敗した");

    for &n in &[1usize, 2, 4095, 4096, 4097, 8192, 3 * 4096 + 17, 1 << 20] {
        let data = gen_data(n, 0x1951_0000 + n as u64);
        for kind in [ArgExtKind::Max, ArgExtKind::Min] {
            let metal_out = reduce
                .run_arg_all_f32(&ctx, &data, kind)
                .expect("metal argext all must succeed on Metal-equipped runner");
            let cpu_out = cpu_argext(&data, &[n], None, kind)[0];
            assert_eq!(
                metal_out, cpu_out,
                "n={n} kind={kind:?}: 添字不一致（metal={metal_out}, cpu={cpu_out}）"
            );

            // run-to-run 決定性。
            let metal_out2 = reduce
                .run_arg_all_f32(&ctx, &data, kind)
                .expect("metal argext all run2");
            assert_eq!(
                metal_out2, metal_out,
                "n={n} kind={kind:?}: run-to-run で添字同一のはず"
            );
        }
    }
}

/// [`MetalReduce::run_arg_axis_f32`] が CPU 参照実装と複数 shape・
/// 複数 dim で添字完全一致することを確認する。
#[test]
#[ignore = "Metal 実機（Apple Silicon）必須"]
fn metal_arg_axis_matches_cpu_exact() {
    let ctx = MetalContext::new().expect("Metal デバイス・コマンドキューの初期化に失敗した");
    let reduce = MetalReduce::new(&ctx).expect("MetalReduce::new に失敗した");

    let cases: &[(&[usize], usize)] = &[
        (&[5], 0),
        (&[3, 4], 0),
        (&[3, 4], 1),
        (&[2, 3, 5], 0),
        (&[2, 3, 5], 1),
        (&[2, 3, 5], 2),
        (&[2, 3, 4, 2], 2),
    ];

    for &(shape, dim) in cases {
        let numel: usize = shape.iter().product();
        let data = gen_data(numel, 0x1951_1000 + numel as u64);
        let outer: usize = shape[..dim].iter().product();
        let axis_len = shape[dim];
        let inner: usize = shape[dim + 1..].iter().product();

        for kind in [ArgExtKind::Max, ArgExtKind::Min] {
            let metal_out = reduce
                .run_arg_axis_f32(&ctx, &data, outer, axis_len, inner, kind)
                .expect("metal argext axis must succeed on Metal-equipped runner");
            let cpu_out = cpu_argext(&data, shape, Some(dim), kind);

            assert_eq!(
                metal_out.len(),
                cpu_out.len(),
                "shape={shape:?} dim={dim} kind={kind:?}: 出力長不一致"
            );
            assert_eq!(
                metal_out, cpu_out,
                "shape={shape:?} dim={dim} kind={kind:?}: 添字不一致"
            );

            // run-to-run 決定性。
            let metal_out2 = reduce
                .run_arg_axis_f32(&ctx, &data, outer, axis_len, inner, kind)
                .expect("metal argext axis run2");
            assert_eq!(
                metal_out2, metal_out,
                "shape={shape:?} dim={dim} kind={kind:?}: run-to-run で添字同一のはず"
            );
        }
    }
}

/// タイ（同値）・NaN 混入時の挙動が CPU 参照実装と一致することを確認
/// する（先勝ちタイ規則・全 NaN は添字 0）。
#[test]
#[ignore = "Metal 実機（Apple Silicon）必須"]
fn metal_arg_all_tie_and_nan_match_cpu() {
    let ctx = MetalContext::new().expect("Metal デバイス・コマンドキューの初期化に失敗した");
    let reduce = MetalReduce::new(&ctx).expect("MetalReduce::new に失敗した");

    let cases: &[Vec<f32>] = &[
        vec![1.0f32; 4097 * 2],
        vec![f32::NAN, 3.0, f32::NAN, 1.0, f32::NAN],
        vec![f32::NAN; 4097],
        vec![-0.0f32, 0.0f32, 1.0f32],
    ];

    for data in cases {
        for kind in [ArgExtKind::Max, ArgExtKind::Min] {
            let metal_out = reduce
                .run_arg_all_f32(&ctx, data, kind)
                .expect("metal argext all (tie/nan) must succeed");
            let cpu_out = cpu_argext(data, &[data.len()], None, kind)[0];
            assert_eq!(
                metal_out, cpu_out,
                "kind={kind:?} data={data:?}: 添字不一致（metal={metal_out}, cpu={cpu_out}）"
            );
        }
    }
}
