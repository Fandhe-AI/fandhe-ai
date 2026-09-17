//! イシュー #1950: RMSNorm／LayerNorm backward の CUDA カーネル
//! （`crate::norm_backward::CudaNormBackward`。recompute-in-backward・
//! forward カーネルとは独立の新設エントリ）の CPU-CUDA 数値一致検証。
//!
//! `tests/layer_norm_parity.rs`・`tests/rmsnorm_backward_parity.rs` と同じ
//! 構成方針を踏襲する: 環境適応スモーク（属性なし。通常 CI で実行し、
//! CUDA 非搭載環境では `fandhe_ai_backend_cuda::CudaError::
//! DriverUnavailable`／`NvrtcUnavailable` を確認して panic しないことのみ
//! 検証）と、実機必須の形状網羅（`#[ignore]`。DGX Spark GB10 等）を
//! 分離する。判定式・許容誤差は再定義せず
//! `fandhe_ai_backend_cpu::parity::assert_parity`（REQ-2 統一複合判定）を
//! 唯一の参照とする（`.claude/rules/coding-rust.md`）。dx は
//! 符号付き項の行内総和（`dot`／`sum_dxhat`）こそ [`warp_reduce_f64`]
//! で GPU の butterfly 縮約と同一順序に揃えてあるが（イシュー #1950・
//! PR #1995 codex-review P1 是正）、RMSNorm の二乗和（`rstd`
//! 導出）は符号なし項のみで相殺が生じないため意図して単純逐次和の
//! ままであり、rstd 自体の丸め誤差が dx へ伝播するため bit 一致は
//! 主張しない（`kernels_norm_backward.rs` 冒頭コメント参照）。
//!
//! CPU 参照実装は `fandhe_ai_autodiff::grad::rmsnorm_vjp_rows`／
//! `layer_norm_vjp_rows`（private）を呼べないため、同じアルゴリズムを
//! 本ファイル内に複製する（doc comment に明記）。
//!
//! **本エージェント実行環境には CUDA 実機への到達手段がないため、
//! `#[ignore]` テストは未実測のまま `docs/perf/logs/cuda-norm-backward-1950/`
//! へ記入欄を残す**（`docs/norm-ops-design.md` 実機実測状況節）。
//!
//! 実行コマンド（DGX Spark GB10 等 CUDA 実機。`#[ignore]` テストのみ）:
//!
//! ```sh
//! cargo test -p fandhe-ai-backend-cuda --release --test norm_backward_parity -- --ignored --nocapture
//! ```

use bench_harness::rng::Xorshift64Star;
use fandhe_ai_backend_cuda::{CudaDevice, CudaError, CudaNormBackward, NormBackwardShape};

mod common;

/// `fandhe_ai_autodiff::eval::warp_reduce_f64`（private・`autodiff` クレート
/// 内限定）のテスト専用複製。CUDA
/// `kernels_norm_backward.rs`（レーンストライド `idx = lane; idx += 32`
/// で分担してから offset 16→1 の butterfly で合流）と同一の縮約順序
/// を再現する（イシュー #1950・PR #1995 codex-review P1 是正）。
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
/// アルゴリズムのテスト専用複製（doc comment 冒頭参照）。`rstd` は
/// `row_rms_stats` 相当（`f64` 二乗和 → 1 回 `f32` downcast）で導出する。
/// `dot_acc`（符号付き項の行内総和）は上記 [`warp_reduce_f64`] で
/// CUDA カーネルと同一順序に揃える（イシュー #1950・PR #1995
/// codex-review P1 是正）。
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
/// アルゴリズムのテスト専用複製。`mean`／`rstd` は `row_ln_stats` 相当
/// （[`warp_reduce_f64`] butterfly reduction・`f64` のまま保持）で
/// 導出する。`sum_dxhat`／`dot_acc`（符号付き項の行内総和）も同じ
/// [`warp_reduce_f64`] で CUDA カーネルと同一順序に揃える（イシュー
/// #1950・PR #1995 codex-review P2 是正: 従来はこの doc comment の
/// 記述に反し `mean`／`sq_acc`／`dot_acc` が単純逐次和のままで、相殺を
/// 含む入力では実際の `eval::row_ln_stats`／CUDA butterfly 縮約と
/// 乖離した基準値を返していた）。
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
    let mut db_acc: Option<Vec<f64>> = if has_bias {
        Some(vec![0.0f64; hidden])
    } else {
        None
    };
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
            let xhat = xhat_at(i);
            let dxhat = dxhat_at(i);
            let term = dxhat * xhat;
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
                let xhat = xhat_at(i);
                let term = dyv * xhat;
                dw_acc[i] += term as f64;
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

/// Linux 実行可能（CUDA 実機不要）: 本ファイルの `cpu_rmsnorm_backward_
/// reference`／`cpu_layer_norm_backward_reference`（`warp_reduce_f64` 経由）
/// が `crates/autodiff/src/grad.rs` の
/// `rmsnorm_grad_dot_reduction_matches_gpu_butterfly_order_on_cancelling_input`／
/// `layer_norm_grad_sum_dxhat_reduction_matches_gpu_butterfly_order_on_cancelling_input`
/// と同一アルゴリズムであることを、同一の相殺入力・同一の期待値
/// （`dx[1] == 0.96875f32`）で直接検証する。
/// `norm_backward_cancelling_input_detects_reduction_order_regression`
/// （実機必須）が「回帰を検出できる」前提としている CPU 参照値の正しさを、
/// 実機なしでも担保する。
#[test]
fn cpu_reference_matches_host_vjp_butterfly_order_on_cancelling_input() {
    let hidden = 32usize;
    let mut dy = vec![0.0f32; hidden];
    dy[0] = 1e20;
    dy[1] = 1.0;
    dy[2] = -1e20;

    let rms_x = vec![1.0f32; hidden];
    let (rms_dx, rms_dw) = cpu_rmsnorm_backward_reference(&rms_x, None, 0.0, 1, hidden, &dy);
    assert!(rms_dw.is_none());
    assert_eq!(rms_dx[1], 0.96875f32);
    for &v in &rms_dx[3..] {
        assert_eq!(v, -0.03125f32);
    }

    let ln_x = vec![0.0f32; hidden];
    let (ln_dx, ln_dw, ln_db) =
        cpu_layer_norm_backward_reference(&ln_x, None, false, 1.0, 1, hidden, &dy);
    assert!(ln_dw.is_none());
    assert!(ln_db.is_none());
    assert_eq!(ln_dx[1], 0.96875f32);
    for &v in &ln_dx[3..] {
        assert_eq!(v, -0.03125f32);
    }
}

#[allow(clippy::too_many_arguments)]
fn assert_rmsnorm_backward_parity(
    nb: &CudaNormBackward,
    seed_x: u64,
    seed_w: u64,
    seed_dy: u64,
    rows: usize,
    hidden: usize,
    with_weight: bool,
    eps: f32,
) {
    let x = Xorshift64Star::new(seed_x).fill_vec(rows * hidden);
    let w = if with_weight {
        Some(Xorshift64Star::new(seed_w).fill_vec(hidden))
    } else {
        None
    };
    let dy = Xorshift64Star::new(seed_dy).fill_vec(rows * hidden);

    let (gpu_dx, gpu_dw) = nb
        .run_rmsnorm_backward_f32(&x, w.as_deref(), &dy, eps, rows, hidden)
        .expect("CudaNormBackward::run_rmsnorm_backward_f32 must succeed on CUDA-equipped runner");
    let (cpu_dx, cpu_dw) = cpu_rmsnorm_backward_reference(&x, w.as_deref(), eps, rows, hidden, &dy);

    let label =
        format!("rmsnorm_backward dx rows={rows} hidden={hidden} with_weight={with_weight}");
    fandhe_ai_backend_cpu::parity::assert_parity(&label, &gpu_dx, &cpu_dx);
    match (gpu_dw, cpu_dw) {
        (Some(gpu_dw), Some(cpu_dw)) => {
            fandhe_ai_backend_cpu::parity::assert_parity(
                &format!("rmsnorm_backward dw rows={rows} hidden={hidden}"),
                &gpu_dw,
                &cpu_dw,
            );
        }
        (None, None) => {}
        (gpu, cpu) => panic!("dw Some/None mismatch: gpu={gpu:?} cpu={cpu:?}"),
    }
}

#[allow(clippy::too_many_arguments)]
fn assert_layer_norm_backward_parity(
    nb: &CudaNormBackward,
    seed_x: u64,
    seed_w: u64,
    seed_dy: u64,
    rows: usize,
    hidden: usize,
    with_weight: bool,
    with_bias: bool,
    eps: f32,
) {
    let x = Xorshift64Star::new(seed_x).fill_vec(rows * hidden);
    let w = if with_weight {
        Some(Xorshift64Star::new(seed_w).fill_vec(hidden))
    } else {
        None
    };
    let dy = Xorshift64Star::new(seed_dy).fill_vec(rows * hidden);

    let (gpu_dx, gpu_dw, gpu_db) = nb
        .run_layer_norm_backward_f32(
            &x,
            w.as_deref(),
            with_bias,
            &dy,
            eps,
            NormBackwardShape { rows, hidden },
        )
        .expect(
            "CudaNormBackward::run_layer_norm_backward_f32 must succeed on CUDA-equipped runner",
        );
    let (cpu_dx, cpu_dw, cpu_db) =
        cpu_layer_norm_backward_reference(&x, w.as_deref(), with_bias, eps, rows, hidden, &dy);

    let label =
        format!("layer_norm_backward dx rows={rows} hidden={hidden} with_weight={with_weight}");
    fandhe_ai_backend_cpu::parity::assert_parity(&label, &gpu_dx, &cpu_dx);
    match (gpu_dw, cpu_dw) {
        (Some(gpu_dw), Some(cpu_dw)) => {
            fandhe_ai_backend_cpu::parity::assert_parity(
                &format!("layer_norm_backward dw rows={rows} hidden={hidden}"),
                &gpu_dw,
                &cpu_dw,
            );
        }
        (None, None) => {}
        (gpu, cpu) => panic!("dw Some/None mismatch: gpu={gpu:?} cpu={cpu:?}"),
    }
    match (gpu_db, cpu_db) {
        (Some(gpu_db), Some(cpu_db)) => {
            fandhe_ai_backend_cpu::parity::assert_parity(
                &format!("layer_norm_backward db rows={rows} hidden={hidden}"),
                &gpu_db,
                &cpu_db,
            );
        }
        (None, None) => {}
        (gpu, cpu) => panic!("db Some/None mismatch: gpu={gpu:?} cpu={cpu:?}"),
    }
}

/// 環境適応スモーク（属性なし。通常 CI で実行）。CUDA／NVRTC 非搭載環境
/// では既知の variant を確認して panic しないことのみ検証し、実機なら
/// 小規模な parity まで実行する（`layer_norm_parity.rs` と同じ分岐）。
#[test]
fn norm_backward_smoke_env_adaptive() {
    let device = match CudaDevice::new(0) {
        Ok(device) => device,
        Err(CudaError::DriverUnavailable { .. }) => return,
        Err(other) => panic!("unexpected error variant for CudaDevice::new: {other}"),
    };
    match CudaNormBackward::new(&device) {
        Ok(nb) => {
            common::parity_baseline::assert_tolerance_constants_pinned();
            assert_rmsnorm_backward_parity(&nb, 7001, 7002, 7003, 1, 8, false, 1e-5);
            assert_rmsnorm_backward_parity(&nb, 7004, 7005, 7006, 3, 128, true, 1e-5);
            assert_layer_norm_backward_parity(&nb, 7007, 7008, 7009, 1, 8, false, false, 1e-5);
            assert_layer_norm_backward_parity(&nb, 7010, 7011, 7012, 3, 128, true, true, 1e-5);
        }
        Err(CudaError::NvrtcUnavailable { .. }) => {}
        Err(other) => panic!("unexpected error variant for CudaNormBackward::new: {other}"),
    }
}

/// 実機必須の形状網羅（受け入れ条件の本体）。
///
/// hidden の網羅: 1（最小）・7（`%4≠0`）・31（`<32`）・32・33（`%4≠0`）・
/// 64・768・4099（大規模・非 4 の倍数）。rows の網羅: 1・3・257。
/// `w`／`b` の `Some`/`None` 全組合せ。
#[test]
#[ignore = "CUDA 実機（DGX Spark GB10 等）必須"]
fn rmsnorm_backward_matches_cpu_across_shapes() {
    common::parity_baseline::assert_tolerance_constants_pinned();

    let device = CudaDevice::new(0).expect("CUDA device must be available on ignored test runner");
    let nb = CudaNormBackward::new(&device)
        .expect("CudaNormBackward::new must succeed on CUDA-equipped test runner");

    let hidden_cases: &[usize] = &[1, 7, 31, 32, 33, 64, 768, 4099];
    let rows_cases: &[usize] = &[1, 3, 257];
    let mut seed = 8000u64;
    for &hidden in hidden_cases {
        for &rows in rows_cases {
            for with_weight in [false, true] {
                seed += 1;
                assert_rmsnorm_backward_parity(
                    &nb,
                    seed,
                    seed + 500,
                    seed + 900,
                    rows,
                    hidden,
                    with_weight,
                    1e-5,
                );
            }
        }
    }
}

#[test]
#[ignore = "CUDA 実機（DGX Spark GB10 等）必須"]
fn layer_norm_backward_matches_cpu_across_shapes() {
    common::parity_baseline::assert_tolerance_constants_pinned();

    let device = CudaDevice::new(0).expect("CUDA device must be available on ignored test runner");
    let nb = CudaNormBackward::new(&device)
        .expect("CudaNormBackward::new must succeed on CUDA-equipped test runner");

    let hidden_cases: &[usize] = &[1, 7, 31, 32, 33, 64, 768, 4099];
    let rows_cases: &[usize] = &[1, 3, 257];
    let mut seed = 9000u64;
    for &hidden in hidden_cases {
        for &rows in rows_cases {
            for with_weight in [false, true] {
                for with_bias in [false, true] {
                    seed += 1;
                    assert_layer_norm_backward_parity(
                        &nb,
                        seed,
                        seed + 500,
                        seed + 900,
                        rows,
                        hidden,
                        with_weight,
                        with_bias,
                        1e-5,
                    );
                }
            }
        }
    }
}

/// `rows == 0`／`hidden == 0` の 0 要素契約（実機必須。driver 非接触の
/// 早期 return 経路も CUDA コンテキスト構築自体は必要なため `#[ignore]`）。
#[test]
#[ignore = "CUDA 実機（DGX Spark GB10 等）必須"]
fn norm_backward_zero_element_contract() {
    let device = CudaDevice::new(0).expect("CUDA device must be available on ignored test runner");
    let nb = CudaNormBackward::new(&device)
        .expect("CudaNormBackward::new must succeed on CUDA-equipped test runner");

    let (dx, dw) = nb
        .run_rmsnorm_backward_f32(&[], Some(&[1.0, 2.0]), &[], 1e-5, 0, 2)
        .expect("rows==0 must succeed");
    assert!(dx.is_empty());
    assert_eq!(dw, Some(vec![0.0f32; 2]));

    // `validate_norm_backward_launch` は `numel == 0` の早期 return 分岐
    // より前に `w.len() == hidden` を検査する（`norm_backward.rs`
    // 参照）。`hidden == 0` のこのケースでは `weight` も長さ 0
    // （`Some(&[])`）で渡す必要がある——さもないと意図した 0 要素
    // 早期 return 経路ではなく `InvalidRmsNormShape` に落ちてしまう
    // （イシュー #1950・PR #1995 codex-review P2／Cursor Bugbot 重複
    // 指摘の是正）。不正な weight 長の拒否自体は
    // `validate_norm_backward_launch_rejects_w_len_mismatch`
    // （`norm_backward.rs` 内の Linux 実行可能な単体テスト）が別途
    // 検証済み。
    let (dx, dw, db) = nb
        .run_layer_norm_backward_f32(
            &[],
            Some(&[]),
            true,
            &[],
            1e-5,
            NormBackwardShape { rows: 3, hidden: 0 },
        )
        .expect("hidden==0 must succeed");
    assert!(dx.is_empty());
    assert_eq!(dw, Some(vec![0.0f32; 0]));
    assert_eq!(db, Some(vec![0.0f32; 0]));
}

/// 相殺入力・NaN 混入・run-to-run 決定性（実機必須）。
///
/// PR #1995（codex-review P2 是正）: 従来は有限性・run-to-run 決定性のみを
/// 検証しており、本 PR で修正した host／CUDA 間の縮約順序不一致（相殺
/// 入力での誤差）を検出できていなかった。`assert_rmsnorm_backward_parity`／
/// `assert_layer_norm_backward_parity`（`cpu_*_backward_reference` との
/// `assert_parity` REQ-2 統一複合判定）と同じ相殺入力
/// `[1e30, 1.0, -1e30, 0.0]` に対して CPU 参照値との数値一致を確認する
/// （bit 一致は主張しない。ファイル冒頭 doc comment 参照）。
#[test]
#[ignore = "CUDA 実機（DGX Spark GB10 等）必須"]
fn norm_backward_numerical_stability_and_determinism() {
    let device = CudaDevice::new(0).expect("CUDA device must be available on ignored test runner");
    let nb = CudaNormBackward::new(&device)
        .expect("CudaNormBackward::new must succeed on CUDA-equipped test runner");

    // 相殺入力（REQ-2 判定。bit 一致は主張しない）。
    let x: Vec<f32> = vec![1e30, 1.0, -1e30, 0.0];
    let dy: Vec<f32> = vec![1.0, 1.0, 1.0, 1.0];
    let (rms_dx1, rms_dw1) = nb
        .run_rmsnorm_backward_f32(&x, None, &dy, 1e-5, 1, 4)
        .expect("cancelling-input rmsnorm backward must succeed");
    for &v in &rms_dx1 {
        assert!(v.is_finite(), "expected finite rmsnorm dx, got {v}");
    }
    let (cpu_rms_dx, cpu_rms_dw) = cpu_rmsnorm_backward_reference(&x, None, 1e-5, 1, 4, &dy);
    fandhe_ai_backend_cpu::parity::assert_parity(
        "rmsnorm_backward dx cancelling-input",
        &rms_dx1,
        &cpu_rms_dx,
    );
    assert_eq!(
        rms_dw1, cpu_rms_dw,
        "weight なし: dw は両側とも None のはず"
    );
    let (rms_dx2, _) = nb
        .run_rmsnorm_backward_f32(&x, None, &dy, 1e-5, 1, 4)
        .expect("re-run must succeed");
    assert_eq!(rms_dx1, rms_dx2, "run-to-run bit 同一契約（決定性）");

    let (ln_dx1, ln_dw1, ln_db1) = nb
        .run_layer_norm_backward_f32(
            &x,
            None,
            false,
            &dy,
            1e-5,
            NormBackwardShape { rows: 1, hidden: 4 },
        )
        .expect("cancelling-input layer_norm backward must succeed");
    for &v in &ln_dx1 {
        assert!(v.is_finite(), "expected finite layer_norm dx, got {v}");
    }
    let (cpu_ln_dx, cpu_ln_dw, cpu_ln_db) =
        cpu_layer_norm_backward_reference(&x, None, false, 1e-5, 1, 4, &dy);
    fandhe_ai_backend_cpu::parity::assert_parity(
        "layer_norm_backward dx cancelling-input",
        &ln_dx1,
        &cpu_ln_dx,
    );
    assert_eq!(ln_dw1, cpu_ln_dw, "weight なし: dw は両側とも None のはず");
    assert_eq!(ln_db1, cpu_ln_db, "bias なし: db は両側とも None のはず");
    let (ln_dx2, _, _) = nb
        .run_layer_norm_backward_f32(
            &x,
            None,
            false,
            &dy,
            1e-5,
            NormBackwardShape { rows: 1, hidden: 4 },
        )
        .expect("re-run must succeed");
    assert_eq!(ln_dx1, ln_dx2, "run-to-run bit 同一契約（決定性）");
}

/// 縮約順序不一致を実際に検出できる相殺入力での CPU-CUDA 突合（実機必須）。
///
/// PR #1995（codex-review P2 是正）: 上記
/// `norm_backward_numerical_stability_and_determinism` が使う相殺入力
/// `[1e30, 1.0, -1e30, 0.0]` は `rstd` が約 `1e-30` まで縮むため、`dx` の
/// 差分も極小になり `assert_parity` の絶対誤差救済（1e-5 未満）で縮約順序が
/// 一致していなくても素通りしてしまう（この入力では回帰を検出できない）。
/// `crates/autodiff/src/grad.rs` の
/// `rmsnorm_grad_dot_reduction_matches_gpu_butterfly_order_on_cancelling_input`／
/// `layer_norm_grad_sum_dxhat_reduction_matches_gpu_butterfly_order_on_cancelling_input`
/// と同一の入力（RMSNorm: `x=[1.0; 32]`・`eps=0.0`。LayerNorm:
/// `x=[0.0; 32]`・`eps=1.0`。共通: `dy=[1e20, 1.0, -1e20, 0.0, ...]`）を
/// 本ファイルの `warp_reduce_f64`／`cpu_*_backward_reference`（CUDA と同一の
/// butterfly 縮約順序）へ通すと、host 側の `dx[1]` は厳密に
/// `0.96875f32`・`dx[3..]` は `-0.03125f32` になる（先頭からの単純逐次和
/// では `0.0`／`1.0` のまま）。CUDA 側がこの縮約順序不一致を起こせば
/// `dx` の差分は `0.03125` 程度になり、絶対誤差救済閾値 `1e-5` を明確に
/// 超えるため `assert_parity` が確実に fail する。
#[test]
#[ignore = "CUDA 実機（DGX Spark GB10 等）必須"]
fn norm_backward_cancelling_input_detects_reduction_order_regression() {
    let device = CudaDevice::new(0).expect("CUDA device must be available on ignored test runner");
    let nb = CudaNormBackward::new(&device)
        .expect("CudaNormBackward::new must succeed on CUDA-equipped test runner");
    let hidden = 32usize;

    let mut dy = vec![0.0f32; hidden];
    dy[0] = 1e20;
    dy[1] = 1.0;
    dy[2] = -1e20;

    // RMSNorm: x=[1.0; 32]・eps=0.0（rstd=1.0。#1950 回帰と同一の入力）。
    let rms_x = vec![1.0f32; hidden];
    let (rms_dx, rms_dw) = nb
        .run_rmsnorm_backward_f32(&rms_x, None, &dy, 0.0, 1, hidden)
        .expect("cancelling-input rmsnorm backward must succeed");
    let (cpu_rms_dx, cpu_rms_dw) =
        cpu_rmsnorm_backward_reference(&rms_x, None, 0.0, 1, hidden, &dy);
    assert_eq!(
        cpu_rms_dx[1], 0.96875f32,
        "CPU 参照値自体が butterfly 縮約の dot=1.0 を反映しているはず（回帰の前提）"
    );
    fandhe_ai_backend_cpu::parity::assert_parity(
        "rmsnorm_backward dx cancelling-input (hidden=32)",
        &rms_dx,
        &cpu_rms_dx,
    );
    assert_eq!(rms_dw, cpu_rms_dw, "weight なし: dw は両側とも None のはず");

    // LayerNorm: x=[0.0; 32]・eps=1.0（xhat=0 を全要素で恒等に満たす。
    // #1950 回帰と同一の入力）。
    let ln_x = vec![0.0f32; hidden];
    let (ln_dx, ln_dw, ln_db) = nb
        .run_layer_norm_backward_f32(
            &ln_x,
            None,
            false,
            &dy,
            1.0,
            NormBackwardShape { rows: 1, hidden },
        )
        .expect("cancelling-input layer_norm backward must succeed");
    let (cpu_ln_dx, cpu_ln_dw, cpu_ln_db) =
        cpu_layer_norm_backward_reference(&ln_x, None, false, 1.0, 1, hidden, &dy);
    assert_eq!(
        cpu_ln_dx[1], 0.96875f32,
        "CPU 参照値自体が butterfly 縮約の sum_dxhat=1.0 を反映しているはず（回帰の前提）"
    );
    fandhe_ai_backend_cpu::parity::assert_parity(
        "layer_norm_backward dx cancelling-input (hidden=32)",
        &ln_dx,
        &cpu_ln_dx,
    );
    assert_eq!(ln_dw, cpu_ln_dw, "weight なし: dw は両側とも None のはず");
    assert_eq!(ln_db, cpu_ln_db, "bias なし: db は両側とも None のはず");
}

// --- BackendOps::rmsnorm_backward／layer_norm_backward 独立エントリ ---

/// 環境適応スモーク（属性なし。通常 CI で実行）。`CudaBackendOps` 経由の
/// 独立エントリが直接 `CudaNormBackward` 呼び出しと一致することを確認する。
#[test]
fn backend_ops_norm_backward_smoke_env_adaptive() {
    use fandhe_ai_tensor_core::device::BackendError;
    use fandhe_ai_tensor_core::{BackendOps, Tensor};

    let rows = 3usize;
    let hidden = 8usize;
    let x_data = Xorshift64Star::new(4101).fill_vec(rows * hidden);
    let dy_data = Xorshift64Star::new(4102).fill_vec(rows * hidden);
    let x = Tensor::new(x_data.clone(), &[rows, hidden]).expect("valid tensor");
    let dy = Tensor::new(dy_data.clone(), &[rows, hidden]).expect("valid tensor");

    let cuda = fandhe_ai_backend_cuda::CudaBackendOps::new(0);
    match cuda.rmsnorm_backward(&x, None, &dy, 1e-5) {
        Ok((dx, dw)) => {
            let (expected_dx, expected_dw) =
                cpu_rmsnorm_backward_reference(&x_data, None, 1e-5, rows, hidden, &dy_data);
            assert_eq!(dx.shape(), &[rows, hidden]);
            assert!(dw.is_none() && expected_dw.is_none());
            fandhe_ai_backend_cpu::parity::assert_parity(
                "BackendOps::rmsnorm_backward vs cpu reference",
                dx.as_slice().expect("contiguous"),
                &expected_dx,
            );
        }
        Err(BackendError::CudaUnavailable(msg)) => {
            assert!(!msg.is_empty(), "error detail message must not be empty");
        }
        Err(other) => {
            panic!("unexpected error variant for CudaBackendOps::rmsnorm_backward: {other}")
        }
    }
}
