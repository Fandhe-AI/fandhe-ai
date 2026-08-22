//! イシュー #592: 融合 RMSNorm 順伝播カーネル（NVRTC・warp 内 reduction・
//! persistent block）の CPU-CUDA 数値一致検証。
//!
//! `gemm_bias_act_parity.rs` と同じ構成方針を踏襲する: 環境適応スモーク
//! （属性なし。通常 CI で実行し、CUDA 非搭載環境では
//! `fandhe_ai_backend_cuda::CudaError::DriverUnavailable` を確認して panic しない
//! ことのみ検証）と、実機必須の形状網羅（`#[ignore]`。DGX Spark GB10 等）を
//! 分離する。判定式・許容誤差は再定義せず `fandhe_ai_backend_cpu::parity` を唯一の
//! 参照とする（`.claude/rules/coding-rust.md`）。
//!
//! CPU 参照実装は本ファイル内のテスト専用関数（`f32::mul_add` 使用）で
//! ある。`backend-cpu` 側 RMSNorm 実装は別スコープ（#607）のため依存しない
//! （実装計画 §5「Step 5」）。
//!
//! 実行コマンド（DGX Spark GB10 等 CUDA 実機。`#[ignore]` テストのみ）:
//!
//! ```sh
//! cargo test -p fandhe-ai-backend-cuda --release --test rmsnorm_parity -- --ignored --nocapture
//! ```

use bench_harness::rng::Xorshift64Star;
use fandhe_ai_backend_cuda::{CudaDevice, CudaError, CudaRmsNorm};

mod common;

/// テスト専用 CPU 参照実装（`f32::mul_add` を使用し、GPU 側 `fmaf` と
/// 丸め方針を揃える。`.claude/rules/coding-rust.md`）。
/// `out = x * rsqrt(mean(x^2, axis=-1) + eps) * w`（`w` が `None` の場合は
/// 乗算をスキップ）。
fn cpu_rmsnorm_reference(
    x: &[f32],
    w: Option<&[f32]>,
    eps: f32,
    rows: usize,
    hidden: usize,
) -> Vec<f32> {
    let mut out = vec![0.0f32; x.len()];
    if hidden == 0 {
        return out;
    }
    let inv_n = 1.0f32 / hidden as f32;
    for r in 0..rows {
        let row = &x[r * hidden..(r + 1) * hidden];
        let mut acc = 0.0f32;
        for &v in row {
            acc = v.mul_add(v, acc);
        }
        let rstd = 1.0f32 / (acc.mul_add(inv_n, eps)).sqrt();
        let out_row = &mut out[r * hidden..(r + 1) * hidden];
        for i in 0..hidden {
            let mut normed = row[i] * rstd;
            if let Some(w) = w {
                normed *= w[i];
            }
            out_row[i] = normed;
        }
    }
    out
}

fn assert_rmsnorm_parity(
    rmsnorm: &CudaRmsNorm,
    seed_x: u64,
    seed_w: u64,
    rows: usize,
    hidden: usize,
    with_weight: bool,
    eps: f32,
) {
    let x_data = Xorshift64Star::new(seed_x).fill_vec(rows * hidden);
    let w_data = if with_weight {
        Some(Xorshift64Star::new(seed_w).fill_vec(hidden))
    } else {
        None
    };

    // `hiddens` の網羅（呼び出し元 `rmsnorm_matches_cpu_across_shapes`）は
    // 1 パス（≤8192）・2 パス（16384）の両経路を意図的にまたぐ形状を含む
    // （実装計画 §5「Step 5」）。経路自体は `rmsnorm.rs` 内の単体テスト
    // （`rmsnorm_route_*`）で既に検証済みのため、ここでは出力の数値一致
    // のみを比較する。
    let gpu_out = rmsnorm
        .run_rmsnorm_f32(&x_data, w_data.as_deref(), eps, rows, hidden)
        .expect("CudaRmsNorm::run_rmsnorm_f32 must succeed on CUDA-equipped test runner");
    let cpu_out = cpu_rmsnorm_reference(&x_data, w_data.as_deref(), eps, rows, hidden);

    assert_eq!(gpu_out.len(), cpu_out.len());
    fandhe_ai_backend_cpu::parity::assert_parity(
        &format!(
            "rmsnorm cpu-cuda parity rows={rows} hidden={hidden} with_weight={with_weight} \
             eps={eps}"
        ),
        &gpu_out,
        &cpu_out,
    );
}

/// 環境適応スモーク（属性なし。通常 CI で実行）。CUDA 不在
/// （`CudaError::DriverUnavailable`）に加え、driver は存在するが NVRTC が
/// 見つからない（`CudaError::NvrtcUnavailable`。本セッション実行環境で
/// 実際に観測した構成）も早期 return の対象とする
/// （`gemm_bias_act_parity.rs::gemm_bias_act_parity_smoke_env_adaptive`
/// と同じ分岐パターンを、カーネルコンパイルの失敗可能性まで含めて拡張）。
/// 実機なら形状網羅ケースまで実行する。
///
/// いずれの `match` も環境不在を表す既知の variant
/// （`DriverUnavailable`／`NvrtcUnavailable`）のみを早期 return の対象と
/// し、それ以外（カーネルのコンパイルエラー・関数名不一致・属性取得
/// エラー等）は `panic!` する。全 `Err` を無条件に環境不在扱いすると、
/// CUDA/NVRTC が利用可能な CI 環境でも実際のバグが握りつぶされテストが
/// 誤って成功してしまう（codex-review 指摘・PR #706 レビュー
/// r3793473253。`device_init.rs`／`transpose_parity.rs` と同じ厳密な
/// variant match パターンを踏襲する）。
#[test]
fn rmsnorm_parity_smoke_env_adaptive() {
    let device = match CudaDevice::new(0) {
        Ok(device) => device,
        Err(CudaError::DriverUnavailable { .. }) => return,
        Err(other) => panic!("unexpected error variant for CudaDevice::new: {other}"),
    };
    match CudaRmsNorm::new(&device) {
        Ok(rmsnorm) => {
            common::parity_baseline::assert_tolerance_constants_pinned();
            assert_rmsnorm_parity(&rmsnorm, 801, 802, 1, 8, false, 1e-5);
            assert_rmsnorm_parity(&rmsnorm, 803, 804, 3, 1024, true, 1e-5);
        }
        Err(CudaError::NvrtcUnavailable { .. }) => {
            // NVRTC 非搭載環境（driver はあるが nvrtc が無い）。panic
            // しないことのみ確認する。
        }
        Err(other) => panic!("unexpected error variant for CudaRmsNorm::new: {other}"),
    }
}

/// 実機必須の形状網羅（受け入れ条件の本体）。
///
/// hidden の網羅: 8（極小）・1024（1 パス中位）・4096（1 パス上位）・
/// 4097（vec4 端要素・`hidden % 4 != 0`）・8192（1 パス上限付近）・
/// 16384（2 パス強制）。rows の網羅: 1・3・33（persistent 行ループを
/// grid 超で回す）。
#[test]
#[ignore = "CUDA 実機（DGX Spark GB10 等）必須"]
fn rmsnorm_matches_cpu_across_shapes() {
    common::parity_baseline::assert_tolerance_constants_pinned();

    let device = CudaDevice::new(0).expect("CUDA device must be available on real-device runner");
    let rmsnorm = CudaRmsNorm::new(&device).expect("RMSNorm kernel compile must succeed");

    let hiddens: &[usize] = &[8, 1024, 4096, 4097, 8192, 16384];
    let rows_cases: &[usize] = &[1, 3, 33];

    let mut seed = 1000u64;
    for &hidden in hiddens {
        for &rows in rows_cases {
            for &with_weight in &[false, true] {
                seed += 1;
                let seed_w = seed + 500;
                assert_rmsnorm_parity(&rmsnorm, seed, seed_w, rows, hidden, with_weight, 1e-5);
            }
        }
    }

    // eps=0.0（`run_fused` 経由の canonical プランと同じ eps。有限値の
    // 境界ケース）。
    assert_rmsnorm_parity(&rmsnorm, 9001, 9002, 4, 256, false, 0.0);
}

/// `run_fused`（`ops.rs::CudaBackendOps::run_fused`）経由の canonical
/// RMSNorm プラン実行を CPU per-op 合成（`mul → sum → rsqrt → broadcast →
/// mul`）と突き合わせる。CUDA 非搭載環境では `BackendError::CudaUnavailable`
/// を確認して早期 return する env-adaptive 分岐
/// （`ops.rs` の既存テストパターン踏襲）。
#[test]
fn rmsnorm_run_fused_matches_cpu_composed_env_adaptive() {
    use fandhe_ai_tensor_core::device::BackendError;
    use fandhe_ai_tensor_core::{BackendOps, DType, FusedOpKind, FusionPlan, Tensor};

    let hidden = 16usize;
    let x_data = Xorshift64Star::new(9101).fill_vec(hidden);
    let x = Tensor::new(x_data.clone(), &[hidden]).expect("valid tensor");

    // canonical RMSNorm プラン（leaf 0=x, 1=Mul(0,0), 2=Sum{None}(1),
    // 3=Rsqrt(2), 4=Broadcast{None}(3), 5=Mul(4,0)）。
    let ops = vec![
        FusedOpKind::Input { leaf_index: 0 },
        FusedOpKind::Mul { lhs: 0, rhs: 0 },
        FusedOpKind::Sum {
            input: 1,
            axis: None,
        },
        FusedOpKind::Rsqrt { input: 2 },
        FusedOpKind::Broadcast {
            input: 3,
            axis: None,
        },
        FusedOpKind::Mul { lhs: 4, rhs: 0 },
    ];
    let plan = FusionPlan::from_ops(ops, vec![hidden], DType::F32, 1)
        .expect("canonical RMSNorm plan must construct");

    let cuda = fandhe_ai_backend_cuda::CudaBackendOps::new(0);
    match cuda.run_fused(&plan, &[&x]) {
        Ok(fused_out) => {
            // CPU per-op 合成（プランと同じ意味論: mean 化・eps なし）。
            let sq: Vec<f32> = x_data.iter().map(|v| v * v).collect();
            let sum: f32 = sq.iter().sum();
            let rstd = 1.0f32 / sum.sqrt();
            let composed: Vec<f32> = x_data.iter().map(|v| v * rstd).collect();

            assert_eq!(fused_out.shape(), &[hidden]);
            fandhe_ai_backend_cpu::parity::assert_parity(
                "rmsnorm run_fused vs cpu composed (canonical plan)",
                fused_out.as_slice().expect("contiguous"),
                &composed,
            );
        }
        Err(BackendError::CudaUnavailable(msg)) => {
            assert!(!msg.is_empty(), "error detail message must not be empty");
        }
        Err(other) => panic!("unexpected error variant for CudaBackendOps::run_fused: {other}"),
    }
}

/// CPU-CUDA 直接突合（イシュー #607）: `fandhe_ai_backend_cpu::rmsnorm::
/// run_rmsnorm_f32`（NEON/rayon 参照実装）を GPU 出力と直接比較する。
/// 実機必須（`#[ignore]`。CI ではコンパイルのみ）。既存の
/// `cpu_rmsnorm_reference`（テスト専用ローカル参照実装）と数学的に同一
/// だが、本テストは実クレート API の呼び出し経路自体を検証する
/// （実装計画 §4「Step 5」）。
#[test]
#[ignore = "CUDA 実機（DGX Spark GB10 等）必須"]
fn rmsnorm_matches_backend_cpu_directly() {
    common::parity_baseline::assert_tolerance_constants_pinned();

    let device = CudaDevice::new(0).expect("CUDA device must be available on real-device runner");
    let rmsnorm = CudaRmsNorm::new(&device).expect("RMSNorm kernel compile must succeed");

    let rows = 3usize;
    let hidden = 4097usize; // NEON 端要素を含む。
    let eps = 1e-5f32;
    let x_data = Xorshift64Star::new(31_001).fill_vec(rows * hidden);
    let w_data = Xorshift64Star::new(31_002).fill_vec(hidden);

    let gpu_out = rmsnorm
        .run_rmsnorm_f32(&x_data, Some(&w_data), eps, rows, hidden)
        .expect("CudaRmsNorm::run_rmsnorm_f32 must succeed on CUDA-equipped test runner");
    let cpu_out =
        fandhe_ai_backend_cpu::rmsnorm::run_rmsnorm_f32(&x_data, Some(&w_data), eps, rows, hidden)
            .expect("fandhe_ai_backend_cpu::rmsnorm::run_rmsnorm_f32 must succeed");

    fandhe_ai_backend_cpu::parity::assert_parity(
        "rmsnorm cpu(backend_cpu)-cuda direct parity",
        &gpu_out,
        &cpu_out,
    );
}
