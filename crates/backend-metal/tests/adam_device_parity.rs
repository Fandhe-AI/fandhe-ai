//! `MetalBackendOps::adam_step_device`／`adam_step_device_tracked`
//! （イシュー #2070・in-place デバイス常駐 Adam・AdamW 更新）の実機
//! テスト。
//!
//! `MetalBackendOps` は `crates/backend-metal/src/lib.rs` 側で
//! `cfg(target_os = "macos")` ゲートされている（`backend-metal` クレート
//! 自体は全 OS でビルド対象になる）ため、本テストファイルにも同じ
//! `cfg(target_os = "macos")` を付けて非 macOS の CI（GitHub ホステッド
//! ubuntu-latest）でコンパイル対象から除外する（`sgd_device_parity.rs`
//! と同じ理由）。`cfg(target_os = "macos")` はコンパイル対象を絞るのみで
//! 実機の有無までは保証しないため、各 `#[test]` には理由付き
//! `#[ignore]` を付け macOS 実機で `--ignored` を明示指定したときのみ
//! 実行する:
//!
//! ```sh
//! cargo test -p fandhe-ai-backend-metal --release --test adam_device_parity -- --ignored --nocapture
//! ```
//!
//! 事前登録判定規則（`docs/perf/logs/adam-device-step-metal-2070/
//! README.md`）: 1. REQ-2 統一複合判定（相対誤差 1e-3 未満 または
//! 絶対誤差 1e-5 未満） 2. bit 完全一致（`to_bits()` 全要素一致。REQ-2
//! とは独立に判定） 3. run-to-run 5 run bit 同一 4. SGD 常駐経路の
//! 非後退（`sgd_device_parity.rs` 自体で担保） 5. 性能判定なし。
#![cfg(target_os = "macos")]

use fandhe_ai_backend_metal::MetalBackendOps;
use fandhe_ai_tensor_core::{
    AdamStepConfig, AdamStepKind, BackendOps, DispatchFailureCell, Tensor,
};

const N: usize = 17;

fn assert_close(actual: f32, expected: f32, ctx: &str) {
    let abs_diff = (actual - expected).abs();
    let rel_diff = abs_diff / expected.abs().max(1e-12);
    assert!(
        abs_diff < 1e-5 || rel_diff < 1e-3,
        "{ctx}: actual={actual} expected={expected} abs_diff={abs_diff} rel_diff={rel_diff}"
    );
}

/// `param`／`m`／`v` の初期値・勾配列を決定的に生成する。
fn seed_data() -> (Vec<f32>, Vec<f32>) {
    let init: Vec<f32> = (0..N).map(|i| (i as f32) * 0.1 - 0.8).collect();
    let grad_seed: Vec<f32> = (0..N).map(|i| ((i as f32) * 0.37).sin() * 0.05).collect();
    (init, grad_seed)
}

/// Metal・CPU 双方へ同じ `AdamStepConfig` 列を適用し、最終
/// `param.to_bits()` 列を返す（`beta^t` は `f64` 逐次積で追跡）。
fn run_adam(kind: AdamStepKind, weight_decay: f32, steps: usize) -> (Vec<f32>, Vec<f32>) {
    let metal_ops = MetalBackendOps::new();
    let metal_mem = metal_ops
        .memory_ops()
        .expect("MetalBackendOps must implement MemoryOps");
    let cpu_ops = fandhe_ai_backend_cpu::CpuBackendOps::new();
    let cpu_mem = cpu_ops
        .memory_ops()
        .expect("CpuBackendOps must implement MemoryOps");

    let (init, grad_seed) = seed_data();

    let mut metal_param = metal_mem
        .upload(&Tensor::new(init.clone(), &[N]).unwrap())
        .unwrap();
    let mut metal_m = metal_mem.alloc_zeroed(&[N]).unwrap();
    let mut metal_v = metal_mem.alloc_zeroed(&[N]).unwrap();

    let mut cpu_param = cpu_mem.upload(&Tensor::new(init, &[N]).unwrap()).unwrap();
    let mut cpu_m = cpu_mem.alloc_zeroed(&[N]).unwrap();
    let mut cpu_v = cpu_mem.alloc_zeroed(&[N]).unwrap();

    let lr = 0.001f64;
    let beta1 = 0.9f64;
    let beta2 = 0.999f64;
    let mut beta1_pow_t = 1.0f64;
    let mut beta2_pow_t = 1.0f64;

    for step in 0..steps {
        beta1_pow_t *= beta1;
        beta2_pow_t *= beta2;
        let step_size = (lr / (1.0 - beta1_pow_t)) as f32;
        let bias_correction2_sqrt = ((1.0 - beta2_pow_t).sqrt()) as f32;
        let decay_factor = (1.0 - lr * weight_decay as f64) as f32;

        let grad: Vec<f32> = grad_seed
            .iter()
            .map(|g| g * (1.0 + step as f32 * 0.001))
            .collect();
        let grad_tensor = Tensor::new(grad, &[N]).unwrap();
        let metal_grad = metal_mem.upload(&grad_tensor).unwrap();
        let cpu_grad = cpu_mem.upload(&grad_tensor).unwrap();

        let config = AdamStepConfig {
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            weight_decay,
            decay_factor,
            step_size,
            bias_correction2_sqrt,
            kind,
        };

        metal_ops
            .adam_step_device(
                &mut metal_param,
                &metal_grad,
                &mut metal_m,
                &mut metal_v,
                &config,
            )
            .expect("metal adam_step_device must succeed on real hardware");
        cpu_ops
            .adam_step_device(&mut cpu_param, &cpu_grad, &mut cpu_m, &mut cpu_v, &config)
            .unwrap();
    }

    let metal_result = metal_mem.download(&metal_param).unwrap();
    let cpu_result = cpu_mem.download(&cpu_param).unwrap();
    (
        metal_result.as_slice().unwrap().to_vec(),
        cpu_result.as_slice().unwrap().to_vec(),
    )
}

fn run_and_check(kind: AdamStepKind, weight_decay: f32, steps: usize, ctx: &str) {
    let (metal, cpu) = run_adam(kind, weight_decay, steps);
    for i in 0..N {
        assert_close(metal[i], cpu[i], &format!("{ctx} index {i}"));
    }
}

#[test]
#[ignore = "Metal 実機（Apple Silicon）依存。CI では実行しない"]
fn coupled_wd_zero_matches_cpu_reference() {
    run_and_check(AdamStepKind::Coupled, 0.0, 100, "coupled_wd_zero");
}

#[test]
#[ignore = "Metal 実機（Apple Silicon）依存。CI では実行しない"]
fn coupled_wd_nonzero_matches_cpu_reference() {
    run_and_check(AdamStepKind::Coupled, 0.01, 100, "coupled_wd_nonzero");
}

#[test]
#[ignore = "Metal 実機（Apple Silicon）依存。CI では実行しない"]
fn decoupled_wd_zero_matches_cpu_reference() {
    run_and_check(AdamStepKind::Decoupled, 0.0, 100, "decoupled_wd_zero");
}

#[test]
#[ignore = "Metal 実機（Apple Silicon）依存。CI では実行しない"]
fn decoupled_wd_nonzero_matches_cpu_reference() {
    run_and_check(AdamStepKind::Decoupled, 0.01, 100, "decoupled_wd_nonzero");
}

/// 複数スレッドグループにまたがる奇数要素数（threadgroup width 256 の
/// 非倍数）で境界検査（REQ-8）を実機で検証する。
#[test]
#[ignore = "Metal 実機（Apple Silicon）依存。CI では実行しない"]
fn odd_numel_multi_threadgroup_matches_cpu_reference() {
    let metal_ops = MetalBackendOps::new();
    let metal_mem = metal_ops.memory_ops().unwrap();
    let cpu_ops = fandhe_ai_backend_cpu::CpuBackendOps::new();
    let cpu_mem = cpu_ops.memory_ops().unwrap();

    let numel = 65_549usize;
    let init: Vec<f32> = (0..numel).map(|i| ((i as f32) * 0.001).sin()).collect();

    let mut metal_param = metal_mem
        .upload(&Tensor::new(init.clone(), &[numel]).unwrap())
        .unwrap();
    let mut metal_m = metal_mem.alloc_zeroed(&[numel]).unwrap();
    let mut metal_v = metal_mem.alloc_zeroed(&[numel]).unwrap();
    let mut cpu_param = cpu_mem
        .upload(&Tensor::new(init, &[numel]).unwrap())
        .unwrap();
    let mut cpu_m = cpu_mem.alloc_zeroed(&[numel]).unwrap();
    let mut cpu_v = cpu_mem.alloc_zeroed(&[numel]).unwrap();

    for step in 0..5 {
        let grad: Vec<f32> = (0..numel)
            .map(|i| ((i as f32 + step as f32) * 0.002).cos() * 0.05)
            .collect();
        let grad_tensor = Tensor::new(grad, &[numel]).unwrap();
        let metal_grad = metal_mem.upload(&grad_tensor).unwrap();
        let cpu_grad = cpu_mem.upload(&grad_tensor).unwrap();
        let config = AdamStepConfig {
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            weight_decay: 0.0,
            decay_factor: 1.0,
            step_size: 0.001,
            bias_correction2_sqrt: 1.0,
            kind: AdamStepKind::Coupled,
        };
        metal_ops
            .adam_step_device(
                &mut metal_param,
                &metal_grad,
                &mut metal_m,
                &mut metal_v,
                &config,
            )
            .unwrap();
        cpu_ops
            .adam_step_device(&mut cpu_param, &cpu_grad, &mut cpu_m, &mut cpu_v, &config)
            .unwrap();
    }

    let metal_result = metal_mem.download(&metal_param).unwrap();
    let cpu_result = cpu_mem.download(&cpu_param).unwrap();
    let metal_slice = metal_result.as_slice().unwrap();
    let cpu_slice = cpu_result.as_slice().unwrap();
    for i in 0..numel {
        assert_close(
            metal_slice[i],
            cpu_slice[i],
            &format!("odd_numel index {i}"),
        );
    }
}

/// 事前登録判定規則 2: bit 完全一致（REQ-2 とは独立に判定。FAIL は
/// 是正せず `docs/perf/logs/adam-device-step-metal-2070/README.md` へ
/// 記録する）。
#[test]
#[ignore = "Metal 実機（Apple Silicon）依存。CI では実行しない"]
fn coupled_wd_nonzero_bit_identical_to_cpu_reference() {
    let (metal, cpu) = run_adam(AdamStepKind::Coupled, 0.01, 100);
    for i in 0..N {
        assert_eq!(
            metal[i].to_bits(),
            cpu[i].to_bits(),
            "index {i}: Metal と CPU 参照実装が bit 不一致"
        );
    }
}

#[test]
#[ignore = "Metal 実機（Apple Silicon）依存。CI では実行しない"]
fn decoupled_wd_nonzero_bit_identical_to_cpu_reference() {
    let (metal, cpu) = run_adam(AdamStepKind::Decoupled, 0.01, 100);
    for i in 0..N {
        assert_eq!(
            metal[i].to_bits(),
            cpu[i].to_bits(),
            "index {i}: Metal と CPU 参照実装が bit 不一致"
        );
    }
}

/// 事前登録判定規則 3: 同一入力 5 run の `to_bits()` 列が一致する
/// （run-to-run 決定性。checksum 完全一致の実装形）。
#[test]
#[ignore = "Metal 実機（Apple Silicon）依存。CI では実行しない"]
fn run_to_run_bit_identical_across_5_runs() {
    let mut runs = Vec::with_capacity(5);
    for _ in 0..5 {
        let (metal, _cpu) = run_adam(AdamStepKind::Coupled, 0.01, 100);
        runs.push(metal.iter().map(|f| f.to_bits()).collect::<Vec<_>>());
    }
    for (i, run) in runs.iter().enumerate().skip(1) {
        assert_eq!(
            &runs[0], run,
            "run 0 と run {i} の to_bits() 列が不一致（run-to-run 非決定性）"
        );
    }
}

/// `adam_step_device_tracked` 経由も同期 `adam_step_device` と bit 一致
/// し、`token` に失敗が記録されないことを検証する（`sgd_device_parity.rs`
/// に対応するケースがない新規契約。`DeviceParamStore::step_adam` の
/// 呼び出し経路）。
#[test]
#[ignore = "Metal 実機（Apple Silicon）依存。CI では実行しない"]
fn tracked_variant_matches_untracked() {
    let metal_ops = MetalBackendOps::new();
    let metal_mem = metal_ops.memory_ops().unwrap();

    let (init, grad) = seed_data();
    let config = AdamStepConfig {
        beta1: 0.9,
        beta2: 0.999,
        eps: 1e-8,
        weight_decay: 0.01,
        decay_factor: 1.0 - 0.001 * 0.01,
        step_size: 0.001,
        bias_correction2_sqrt: 1.0,
        kind: AdamStepKind::Coupled,
    };

    let mut untracked_param = metal_mem
        .upload(&Tensor::new(init.clone(), &[N]).unwrap())
        .unwrap();
    let mut untracked_m = metal_mem.alloc_zeroed(&[N]).unwrap();
    let mut untracked_v = metal_mem.alloc_zeroed(&[N]).unwrap();
    let grad_buf_untracked = metal_mem
        .upload(&Tensor::new(grad.clone(), &[N]).unwrap())
        .unwrap();
    metal_ops
        .adam_step_device(
            &mut untracked_param,
            &grad_buf_untracked,
            &mut untracked_m,
            &mut untracked_v,
            &config,
        )
        .unwrap();
    let untracked_result = metal_mem.download(&untracked_param).unwrap();

    let mut tracked_param = metal_mem.upload(&Tensor::new(init, &[N]).unwrap()).unwrap();
    let mut tracked_m = metal_mem.alloc_zeroed(&[N]).unwrap();
    let mut tracked_v = metal_mem.alloc_zeroed(&[N]).unwrap();
    let grad_buf_tracked = metal_mem.upload(&Tensor::new(grad, &[N]).unwrap()).unwrap();
    let token = DispatchFailureCell::new();
    metal_ops
        .adam_step_device_tracked(
            &mut tracked_param,
            &grad_buf_tracked,
            &mut tracked_m,
            &mut tracked_v,
            &config,
            &token,
        )
        .unwrap();
    // ホスト実体化（`download`）が同期点となり、バッチが実行される
    // （`sgd.rs`／`context.rs` の非同期契約と同じ）。
    let tracked_result = metal_mem.download(&tracked_param).unwrap();
    assert!(!token.is_set(), "token に失敗が記録されているべきではない");

    let untracked_slice = untracked_result.as_slice().unwrap();
    let tracked_slice = tracked_result.as_slice().unwrap();
    for i in 0..N {
        assert_eq!(
            untracked_slice[i].to_bits(),
            tracked_slice[i].to_bits(),
            "index {i}: tracked と untracked が bit 不一致"
        );
    }
}

#[test]
#[ignore = "Metal 実機（Apple Silicon）依存。CI では実行しない"]
fn adam_step_device_rejects_shape_mismatch() {
    let ops = MetalBackendOps::new();
    let mem = ops.memory_ops().unwrap();
    let mut param_buf = mem
        .upload(&Tensor::new(vec![1.0, 2.0], &[2]).unwrap())
        .unwrap();
    let grad_buf = mem
        .upload(&Tensor::new(vec![1.0, 2.0, 3.0], &[3]).unwrap())
        .unwrap();
    let mut m_buf = mem.alloc_zeroed(&[2]).unwrap();
    let mut v_buf = mem.alloc_zeroed(&[2]).unwrap();
    let config = AdamStepConfig {
        beta1: 0.9,
        beta2: 0.999,
        eps: 1e-8,
        weight_decay: 0.0,
        decay_factor: 1.0,
        step_size: 0.001,
        bias_correction2_sqrt: 1.0,
        kind: AdamStepKind::Coupled,
    };
    let err = ops
        .adam_step_device(&mut param_buf, &grad_buf, &mut m_buf, &mut v_buf, &config)
        .unwrap_err();
    assert!(matches!(
        err,
        fandhe_ai_tensor_core::BackendError::ShapeMismatch(_)
    ));
}

#[test]
#[ignore = "Metal 実機（Apple Silicon）依存。CI では実行しない"]
fn adam_step_device_empty_tensor_is_a_no_op() {
    let ops = MetalBackendOps::new();
    let mem = ops.memory_ops().unwrap();
    let mut param_buf = mem
        .upload(&Tensor::new(Vec::<f32>::new(), &[0]).unwrap())
        .unwrap();
    let grad_buf = mem
        .upload(&Tensor::new(Vec::<f32>::new(), &[0]).unwrap())
        .unwrap();
    let mut m_buf = mem.alloc_zeroed(&[0]).unwrap();
    let mut v_buf = mem.alloc_zeroed(&[0]).unwrap();
    let config = AdamStepConfig {
        beta1: 0.9,
        beta2: 0.999,
        eps: 1e-8,
        weight_decay: 0.0,
        decay_factor: 1.0,
        step_size: 0.001,
        bias_correction2_sqrt: 1.0,
        kind: AdamStepKind::Coupled,
    };
    ops.adam_step_device(&mut param_buf, &grad_buf, &mut m_buf, &mut v_buf, &config)
        .unwrap();
    let out = mem.download(&param_buf).unwrap();
    assert_eq!(out.numel(), 0);
}
