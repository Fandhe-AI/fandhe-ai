//! `gemm.metal::gemm_bias_grad_reduce_f32`（bias 勾配の行方向縮約。
//! binary64 逐次加算の 64bit 整数エミュレーション。イシュー #1566・
//! PR #1659）の Metal 実機 bit 一致テスト。
//!
//! 検証対象は「GPU カーネルの出力 ＝ ホスト参照実装
//! `layout::reduce_bias_grad_rows_host`（`f64` 逐次和 → 1 回 `f32` へ
//! downcast）＝ ホスト側逐語モデル `fandhe_ai_backend_metal::soft_f64::
//! sequential_sum_f32`」の 3 者 bit 完全一致（NaN はクラス一致）。
//! codex-review（PR #1659）が挙げた相殺列・中間 overflow・subnormal・
//! 非有限値の名指し入力を各列に埋め込み、f32 のみの補償和では一致しなかった
//! ケースがカーネルでも一致することを直接確認する。
//!
//! 実行コマンド（Apple Silicon 実機。`--release` 推奨）:
//!
//! ```sh
//! cargo test -p fandhe-ai-backend-metal --release --test gemm_bias_grad_reduce_bit_match -- --ignored --nocapture
//! ```

#![cfg(target_os = "macos")]

use bench_harness::rng::Xorshift64Star;
use fandhe_ai_backend_metal::MetalBackendOps;
use fandhe_ai_backend_metal::layout::{MatrixLayout, reduce_bias_grad_rows_host};
use fandhe_ai_backend_metal::soft_f64::{f32_bits_match, sequential_sum_f32};
use fandhe_ai_tensor_core::{BackendOps, DispatchFailureCell, Tensor};

/// codex-review（PR #1659）で挙げられた名指しケース。各列（長さは
/// テスト側で `batch` 行へ末尾 0 埋め）。
fn named_columns() -> Vec<Vec<f32>> {
    let p = |e: i32| 2f32.powi(e);
    vec![
        vec![p(48), p(24), 1.0, -p(48), -p(24)],
        vec![p(50), p(25), 1.0, -p(50), -p(25)],
        vec![1e8, 1.0, -99999992.0],
        vec![100000000.0, -100000008.0, 8.0],
        vec![f32::from_bits(1), 1.0],
        vec![f32::from_bits(1), f32::from_bits(1), -f32::from_bits(1)],
        vec![f32::MAX, f32::MAX, -f32::MAX, -f32::MAX],
        vec![f32::MAX, f32::MAX],
        vec![f32::MAX, f32::MAX, -f32::MAX],
        vec![p(127), p(-149), -p(127)],
        vec![f32::INFINITY, f32::NEG_INFINITY],
        vec![f32::INFINITY, -f32::MAX, 1.0],
        vec![f32::NAN, 1.0],
        vec![1.0, f32::NAN],
        vec![-0.0, -0.0],
        vec![0.0, -0.0],
        vec![1.0, p(-30), p(-30), p(-30), -1.0],
    ]
}

/// `batch × d_out` の `g`（行優先）を、列ごとに `columns` を縦に並べて作る。
/// 余った列は乱数、余った行は `0.0`。
fn build_g(batch: usize, d_out: usize, columns: &[Vec<f32>], seed: u64) -> Vec<f32> {
    let mut rng = Xorshift64Star::new(seed);
    let random = rng.fill_vec(batch * d_out);
    let mut g = random;
    for (col, values) in columns.iter().enumerate().take(d_out) {
        for row in 0..batch {
            g[row * d_out + col] = values.get(row).copied().unwrap_or(0.0);
        }
    }
    g
}

/// 3 者比較。NaN はクラス一致・それ以外は bit 一致。
fn assert_three_way(kernel: &[f32], host: &[f32], model: &[f32], ctx: &str) {
    assert_eq!(kernel.len(), host.len(), "{ctx}: length");
    assert_eq!(kernel.len(), model.len(), "{ctx}: length");
    for (i, ((k, h), m)) in kernel.iter().zip(host).zip(model).enumerate() {
        assert!(
            f32_bits_match(*k, *h),
            "{ctx}: 列 {i} で GPU カーネルとホスト参照実装が不一致（kernel={k:e} bits={:#010x}, \
             host={h:e} bits={:#010x}）",
            k.to_bits(),
            h.to_bits()
        );
        assert!(
            f32_bits_match(*m, *h),
            "{ctx}: 列 {i} でホスト逐語モデルとホスト参照実装が不一致（model={m:e}, host={h:e}）"
        );
    }
}

/// 名指しケース（相殺・overflow・subnormal・非有限値）と乱数列で、
/// GPU カーネル出力がホスト参照実装・逐語モデルと bit 完全一致する。
#[test]
#[ignore = "Metal 実機（Apple Silicon）依存。CI では実行しない"]
fn bias_grad_reduce_kernel_matches_host_f64_bit_exactly() {
    let ops = MetalBackendOps::new();
    let mem = ops
        .memory_ops()
        .expect("Metal MemoryOps must be available on a Metal-equipped test runner");
    let token = DispatchFailureCell::new();
    let named = named_columns();

    for &(batch, d_in, d_out, seed) in &[
        (5usize, 3usize, 17usize, 0x1566_0001u64),
        (8, 9, 33, 0x1566_0002),
        (64, 129, 96, 0x1566_0003),
        (257, 4, 40, 0x1566_0004),
    ] {
        let g_data = build_g(batch, d_out, &named, seed);
        let g = Tensor::new(g_data.clone(), &[batch, d_out]).unwrap();
        let x = Tensor::new(
            Xorshift64Star::new(seed ^ 0xABCD).fill_vec(batch * d_in),
            &[batch, d_in],
        )
        .unwrap();
        let x_t = x.transpose(0, 1).unwrap();

        let g_layout = MatrixLayout {
            rows: batch,
            cols: d_out,
            ld: d_out,
            transposed: false,
        };
        let host =
            reduce_bias_grad_rows_host(&g_data, &g_layout).expect("valid layout/data in test");
        let model: Vec<f32> = (0..d_out)
            .map(|col| {
                let column: Vec<f32> = (0..batch).map(|row| g_data[row * d_out + col]).collect();
                sequential_sum_f32(&column)
            })
            .collect();

        let weight_mn = d_in * d_out;
        let bias_offset = weight_mn + 1;
        let total = bias_offset + d_out;
        let seed_tensor = Tensor::new(vec![f32::NAN; total], &[total]).unwrap();
        let mut staging = mem.upload(&seed_tensor).unwrap();
        let bias_written = ops
            .gemm_fp32_strict_into_with_bias_reduce_tracked(
                &x_t,
                &g,
                &mut staging,
                0,
                Some((bias_offset, d_out)),
                &token,
            )
            .expect("gemm_fp32_strict_into_with_bias_reduce_tracked must succeed for NT/TN");
        assert!(bias_written, "NT/TN 経路は bias を Ok(true) で書くはず");

        let readback = mem.download(&staging).unwrap();
        let readback_c = readback.contiguous();
        let kernel = &readback_c.as_slice().unwrap()[bias_offset..bias_offset + d_out];
        assert_three_way(
            kernel,
            &host,
            &model,
            &format!("batch={batch} d_in={d_in} d_out={d_out}"),
        );
    }
}

/// `m == 1`（直接コピー経路）で符号付きゼロ・NaN・inf がそのまま保持される。
#[test]
#[ignore = "Metal 実機（Apple Silicon）依存。CI では実行しない"]
fn bias_grad_reduce_kernel_single_row_copies_bits() {
    let ops = MetalBackendOps::new();
    let mem = ops.memory_ops().expect("Metal MemoryOps must be available");
    let token = DispatchFailureCell::new();
    let d_in = 3usize;
    let g_data = vec![
        -0.0f32,
        0.0,
        f32::NAN,
        f32::INFINITY,
        f32::NEG_INFINITY,
        f32::from_bits(1),
        -1.5,
        f32::MAX,
    ];
    let d_out = g_data.len();
    let g = Tensor::new(g_data.clone(), &[1, d_out]).unwrap();
    let x_t = Tensor::new(vec![1.0f32; d_in], &[1, d_in])
        .unwrap()
        .transpose(0, 1)
        .unwrap();
    let g_layout = MatrixLayout {
        rows: 1,
        cols: d_out,
        ld: d_out,
        transposed: false,
    };
    let host = reduce_bias_grad_rows_host(&g_data, &g_layout).unwrap();
    let bias_offset = d_in * d_out;
    let mut staging = mem
        .upload(&Tensor::new(vec![f32::NAN; bias_offset + d_out], &[bias_offset + d_out]).unwrap())
        .unwrap();
    let written = ops
        .gemm_fp32_strict_into_with_bias_reduce_tracked(
            &x_t,
            &g,
            &mut staging,
            0,
            Some((bias_offset, d_out)),
            &token,
        )
        .unwrap();
    assert!(written);
    let readback = mem.download(&staging).unwrap();
    let readback_c = readback.contiguous();
    let kernel = &readback_c.as_slice().unwrap()[bias_offset..bias_offset + d_out];
    let model: Vec<f32> = g_data.iter().map(|v| sequential_sum_f32(&[*v])).collect();
    // `m == 1` は直接コピーのため逐語モデル（`0.0 + x` を経由）とは `-0.0` で
    // 異なりうる。カーネル対ホストのみ bit 比較し、モデルは参考。
    for (i, (k, h)) in kernel.iter().zip(&host).enumerate() {
        assert!(f32_bits_match(*k, *h), "列 {i}: kernel={k:e} host={h:e}");
    }
    assert_eq!(model.len(), d_out);
}
