//! イシュー #1953（親 #1947）: LayerNorm／RMSNorm backward カーネル
//! （`crate::norm_backward::MetalNormBackward`。recompute-in-backward・
//! forward カーネルとは独立の新設エントリ）の CPU-Metal 数値一致検証。
//!
//! `tests/layer_norm_parity.rs` と同じ構成方針を踏襲する: ファイル全体を
//! `#![cfg(target_os = "macos")]` にし、各テストに `#[ignore]` を付けて
//! 通常 CI では実行しない。判定式・許容誤差は再定義せず
//! `fandhe_ai_backend_cpu::parity::assert_parity`（REQ-2 統一複合判定）を
//! 唯一の参照とする（`.claude/rules/coding-rust.md`）。CUDA 側 #1950 と
//! 同じ理由（行内統計の縮約順序が GPU の並列 reduction であり、CPU の
//! 逐次走査と異なる）で dx／dw／db いずれも bit 完全一致は主張しない
//! （`crates/backend-metal/src/shaders/norm_backward.metal` 冒頭コメント
//! 「数値契約」参照）。
//!
//! CPU 参照実装は `fandhe_ai_autodiff::grad::rmsnorm_vjp_rows`／
//! `layer_norm_vjp_rows`（private）を呼べないため、同じアルゴリズムを
//! 本ファイル内に複製する（`crates/backend-cuda/tests/norm_backward_
//! parity.rs` と同じ理由）。
//!
//! **本エージェント実行環境には Apple Silicon 実機への到達手段がない
//! ため、`#[ignore]` テストは未実測のまま
//! `docs/perf/logs/metal-norm-backward-1953/` へ記入欄を残す**。
//!
//! 実行コマンド（Apple Silicon 実機。`#[ignore]` テストのみ）:
//!
//! ```sh
//! cargo test -p fandhe-ai-backend-metal --release --test norm_backward_parity -- --ignored --nocapture
//! ```

#![cfg(target_os = "macos")]

use bench_harness::rng::Xorshift64Star;
use fandhe_ai_backend_cpu::parity::assert_parity;
use fandhe_ai_backend_metal::{MetalContext, MetalNormBackward};
use fandhe_ai_tensor_core::{BackendOps, Tensor};

/// `fandhe_ai_autodiff::eval::warp_reduce_f64`（private）のテスト専用
/// 複製。GPU の 32 レーン + butterfly 縮約順序を再現する。
fn warp_reduce_f64(hidden: usize, mut contribute: impl FnMut(usize, f64) -> f64) -> f64 {
    const LANES: usize = 32;
    let mut lanes = [0.0f64; LANES];
    for (lane, slot) in lanes.iter_mut().enumerate() {
        let mut idx = lane;
        while idx < hidden {
            *slot = contribute(idx, *slot);
            idx += LANES;
        }
    }
    let mut offset = 16usize;
    while offset > 0 {
        let snapshot = lanes;
        for (lane, slot) in lanes.iter_mut().enumerate() {
            *slot = snapshot[lane] + snapshot[lane ^ offset];
        }
        offset >>= 1;
    }
    lanes[0]
}

/// `fandhe_ai_autodiff::grad::rmsnorm_vjp_rows`（private）と同一
/// アルゴリズムのテスト専用複製。
fn cpu_rmsnorm_backward_reference(
    x: &[f32],
    w: Option<&[f32]>,
    eps: f32,
    rows: usize,
    hidden: usize,
    dy: &[f32],
) -> (Vec<f32>, Option<Vec<f32>>) {
    let mut dx = vec![0.0f32; x.len()];
    let mut dw_acc: Option<Vec<f64>> = w.map(|_| vec![0.0f64; hidden]);
    if rows == 0 || hidden == 0 {
        let dw = dw_acc.map(|v| v.into_iter().map(|a| a as f32).collect());
        return (dx, dw);
    }
    let inv_n = 1.0f64 / hidden as f64;
    for r in 0..rows {
        let row = &x[r * hidden..(r + 1) * hidden];
        let dy_row = &dy[r * hidden..(r + 1) * hidden];
        let mut acc = 0.0f64;
        for &v in row {
            let v = v as f64;
            acc = v.mul_add(v, acc);
        }
        let rstd = (1.0f64 / acc.mul_add(inv_n, eps as f64).sqrt()) as f32;
        let dxhat_at = |i: usize| -> f32 {
            match w {
                Some(w) => dy_row[i] * w[i],
                None => dy_row[i],
            }
        };
        let dot_acc = warp_reduce_f64(hidden, |i, acc| {
            let xhat = row[i] * rstd;
            let term = dxhat_at(i) * xhat;
            acc + term as f64
        });
        let mean_dot = dot_acc * inv_n;
        let dx_row = &mut dx[r * hidden..(r + 1) * hidden];
        for (i, (&xv, dxv)) in row.iter().zip(dx_row.iter_mut()).enumerate() {
            let xhat = xv * rstd;
            let dxhat = dxhat_at(i);
            let d = (rstd as f64) * (dxhat as f64 - (xhat as f64) * mean_dot);
            *dxv = d as f32;
        }
        if let Some(dw_acc) = dw_acc.as_mut() {
            for (i, (&xv, &dyv)) in row.iter().zip(dy_row.iter()).enumerate() {
                let xhat = xv * rstd;
                let term = dyv * xhat;
                dw_acc[i] += term as f64;
            }
        }
    }
    let dw = dw_acc.map(|v| v.into_iter().map(|a| a as f32).collect());
    (dx, dw)
}

/// `fandhe_ai_autodiff::grad::layer_norm_vjp_rows`（private）と同一
/// アルゴリズムのテスト専用複製。
#[allow(clippy::too_many_arguments)]
fn cpu_layer_norm_backward_reference(
    x: &[f32],
    w: Option<&[f32]>,
    has_bias: bool,
    eps: f32,
    rows: usize,
    hidden: usize,
    dy: &[f32],
) -> (Vec<f32>, Option<Vec<f32>>, Option<Vec<f32>>) {
    let mut dx = vec![0.0f32; x.len()];
    let mut dw_acc: Option<Vec<f64>> = w.map(|_| vec![0.0f64; hidden]);
    let mut db_acc: Option<Vec<f64>> = has_bias.then(|| vec![0.0f64; hidden]);
    if rows == 0 || hidden == 0 {
        let dw = dw_acc.map(|v| v.into_iter().map(|a| a as f32).collect());
        let db = db_acc.map(|v| v.into_iter().map(|a| a as f32).collect());
        return (dx, dw, db);
    }
    let n = hidden as f64;
    for r in 0..rows {
        let row = &x[r * hidden..(r + 1) * hidden];
        let dy_row = &dy[r * hidden..(r + 1) * hidden];
        let sum = warp_reduce_f64(hidden, |i, acc| acc + row[i] as f64);
        let mean = sum / n;
        let sq_acc = warp_reduce_f64(hidden, |i, acc| {
            let d = row[i] as f64 - mean;
            d.mul_add(d, acc)
        });
        let var = sq_acc / n;
        let rstd = 1.0f64 / (var + eps as f64).sqrt();
        let xhat_at = |i: usize| -> f32 { ((row[i] as f64 - mean) * rstd) as f32 };
        let dxhat_at = |i: usize| -> f32 {
            match w {
                Some(w) => dy_row[i] * w[i],
                None => dy_row[i],
            }
        };
        let sum_dxhat = warp_reduce_f64(hidden, |i, acc| acc + dxhat_at(i) as f64);
        let dot_acc = warp_reduce_f64(hidden, |i, acc| {
            let term = dxhat_at(i) * xhat_at(i);
            acc + term as f64
        });
        let mean_dxhat = sum_dxhat / n;
        let mean_dot = dot_acc / n;
        let dx_row = &mut dx[r * hidden..(r + 1) * hidden];
        for (i, dxv) in dx_row.iter_mut().enumerate() {
            let xhat = xhat_at(i);
            let dxhat = dxhat_at(i);
            let d = rstd * (dxhat as f64 - mean_dxhat - (xhat as f64) * mean_dot);
            *dxv = d as f32;
        }
        if let Some(dw_acc) = dw_acc.as_mut() {
            for (i, &dyv) in dy_row.iter().enumerate() {
                dw_acc[i] += (dyv * xhat_at(i)) as f64;
            }
        }
        if let Some(db_acc) = db_acc.as_mut() {
            for (acc, &dyv) in db_acc.iter_mut().zip(dy_row.iter()) {
                *acc += dyv as f64;
            }
        }
    }
    let dw = dw_acc.map(|v| v.into_iter().map(|a| a as f32).collect());
    let db = db_acc.map(|v| v.into_iter().map(|a| a as f32).collect());
    (dx, dw, db)
}

fn gen_data(rng: &mut Xorshift64Star, n: usize) -> Vec<f32> {
    rng.fill_vec(n).into_iter().map(|v| v * 2.0).collect()
}

fn shapes() -> Vec<(usize, usize)> {
    vec![(1, 1), (2, 31), (3, 32), (4, 100), (5, 257)]
}

/// 環境適応スモーク（属性なし。通常 CI で実行。Metal デバイス不在では
/// `MetalContext::new` が `Err` を返すため panic しないことのみ検証）。
#[test]
fn rmsnorm_backward_smoke_does_not_panic_without_device() {
    let Ok(ctx) = MetalContext::new() else {
        return;
    };
    let Ok(nb) = MetalNormBackward::new(&ctx) else {
        return;
    };
    let x = vec![1.0f32, 2.0, 3.0, 4.0];
    let dy = vec![0.1f32, 0.2, 0.3, 0.4];
    let _ = nb.run_rmsnorm_backward_f32(&ctx, &x, None, &dy, 1e-5, 1, 4);
}

#[test]
#[ignore]
fn rmsnorm_backward_matches_cpu_reference_across_shapes() {
    let ctx = MetalContext::new().expect("Metal device required");
    let nb = MetalNormBackward::new(&ctx).expect("pipeline compile failed");
    let mut rng = Xorshift64Star::new(0xC0FF_EE00_1234_5678);
    for &(rows, hidden) in &shapes() {
        for has_weight in [false, true] {
            let x = gen_data(&mut rng, rows * hidden);
            let dy = gen_data(&mut rng, rows * hidden);
            let w = has_weight.then(|| gen_data(&mut rng, hidden));
            let (dx_gpu, dw_gpu) = nb
                .run_rmsnorm_backward_f32(&ctx, &x, w.as_deref(), &dy, 1e-5, rows, hidden)
                .expect("kernel launch failed");
            let (dx_cpu, dw_cpu) =
                cpu_rmsnorm_backward_reference(&x, w.as_deref(), 1e-5, rows, hidden, &dy);
            assert_parity(
                &format!("rmsnorm dx rows={rows} hidden={hidden}"),
                &dx_gpu,
                &dx_cpu,
            );
            match (dw_gpu, dw_cpu) {
                (Some(a), Some(e)) => assert_parity("rmsnorm dw", &a, &e),
                (None, None) => {}
                _ => panic!("rmsnorm dw Some/None mismatch"),
            }
        }
    }
}

#[test]
#[ignore]
fn layer_norm_backward_matches_cpu_reference_across_shapes() {
    let ctx = MetalContext::new().expect("Metal device required");
    let nb = MetalNormBackward::new(&ctx).expect("pipeline compile failed");
    let mut rng = Xorshift64Star::new(0xABCD_EF01_2345_6789);
    for &(rows, hidden) in &shapes() {
        for has_weight in [false, true] {
            for has_bias in [false, true] {
                let x = gen_data(&mut rng, rows * hidden);
                let dy = gen_data(&mut rng, rows * hidden);
                let w = has_weight.then(|| gen_data(&mut rng, hidden));
                let (dx_gpu, dw_gpu, db_gpu) = nb
                    .run_layer_norm_backward_f32(
                        &ctx,
                        &x,
                        w.as_deref(),
                        has_bias,
                        &dy,
                        1e-5,
                        rows,
                        hidden,
                    )
                    .expect("kernel launch failed");
                let (dx_cpu, dw_cpu, db_cpu) = cpu_layer_norm_backward_reference(
                    &x,
                    w.as_deref(),
                    has_bias,
                    1e-5,
                    rows,
                    hidden,
                    &dy,
                );
                assert_parity(
                    &format!("layer_norm dx rows={rows} hidden={hidden}"),
                    &dx_gpu,
                    &dx_cpu,
                );
                match (dw_gpu, dw_cpu) {
                    (Some(a), Some(e)) => assert_parity("layer_norm dw", &a, &e),
                    (None, None) => {}
                    _ => panic!("layer_norm dw Some/None mismatch"),
                }
                match (db_gpu, db_cpu) {
                    (Some(a), Some(e)) => assert_parity("layer_norm db", &a, &e),
                    (None, None) => {}
                    _ => panic!("layer_norm db Some/None mismatch"),
                }
            }
        }
    }
}

/// `BackendOps::rmsnorm_backward`／`layer_norm_backward`（`MetalBackendOps`
/// override）経由の到達確認（`context_cache::cached_norm_backward` の
/// 結線・`grad::vjp` から呼ばれる実経路を検証）。
#[test]
#[ignore]
fn backend_ops_norm_backward_reaches_metal_override() {
    use fandhe_ai_backend_metal::MetalBackendOps;
    let ops = MetalBackendOps;
    let x = Tensor::new(vec![1.0f32, 2.0, 3.0, 4.0], &[1, 4]).unwrap();
    let dy = Tensor::new(vec![0.1f32, 0.2, 0.3, 0.4], &[1, 4]).unwrap();
    let (dx, dw) = ops
        .rmsnorm_backward(&x, None, &dy, 1e-5)
        .expect("rmsnorm_backward should succeed on Metal");
    assert_eq!(dx.shape(), &[1, 4]);
    assert!(dw.is_none());

    let (dx2, dw2, db2) = ops
        .layer_norm_backward(&x, None, false, &dy, 1e-5)
        .expect("layer_norm_backward should succeed on Metal");
    assert_eq!(dx2.shape(), &[1, 4]);
    assert!(dw2.is_none());
    assert!(db2.is_none());
}
