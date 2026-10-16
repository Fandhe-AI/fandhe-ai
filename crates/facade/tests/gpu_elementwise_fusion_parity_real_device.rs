//! イシュー #3385: facade の B-1 opt-in setter で elementwise 融合を ON にした
//! 状態の実機 parity テスト（CUDA／Metal 実機必須。`#[ignore]`）。
//!
//! 実行コマンド:
//!
//! ```sh
//! cargo test -p fandhe-ai --release --test gpu_elementwise_fusion_parity_real_device \
//!   -- --ignored --nocapture
//! ```
//!
//! 検証内容: (1) CPU tape 出力との統一複合判定
//! （`fandhe_ai_backend_cpu::parity::assert_parity`。tolerance は不変）、
//! (2) 同じ入力の OFF 出力との bit 一致（融合 ≡ per-op の契約）、
//! (3) `BackendOps::run_fused` 直接呼び出しが `Ok` を返すこと。B-1 には経路
//! カウンタがないため、(3) を「融合カーネルへルーティングされた」ことの代替
//! 確認とする。実測と結果の記録は Phase 8 に委ねる（本ファイルは未実行）。

use bench_harness::rng::Xorshift64Star;
use fandhe_ai::{Device, Tape};
use fandhe_ai_backend_cpu::parity::assert_parity;
use fandhe_ai_tensor_core::{BackendOps, DType, FusedOpKind, FusionPlan, Tensor};

const NUMELS: [usize; 6] = [1, 255, 256, 257, 4096, 65536];

fn values(t: &Tensor<f32>) -> Vec<f32> {
    t.contiguous()
        .as_slice()
        .expect("contiguous() 直後は必ず as_slice() が Some を返す")
        .to_vec()
}

fn bits(v: &[f32]) -> Vec<u32> {
    v.iter().map(|x| x.to_bits()).collect()
}

fn build_4_chain_plan(numel: usize) -> FusionPlan {
    FusionPlan::from_ops(
        vec![
            FusedOpKind::Input { leaf_index: 0 },
            FusedOpKind::Input { leaf_index: 1 },
            FusedOpKind::Add { lhs: 0, rhs: 1 },
            FusedOpKind::Relu { input: 2 },
            FusedOpKind::Exp { input: 3 },
            FusedOpKind::Tanh { input: 4 },
        ],
        vec![numel],
        DType::F32,
        2,
    )
    .expect("test fixture: valid plan")
}

/// `add → relu → exp → tanh` を `tape` 上で組んで実体化する。
fn run_chain(tape: &Tape, a: &Tensor<f32>, b: &Tensor<f32>) -> Vec<f32> {
    let a = tape.var(a);
    let b = tape.var(b);
    values(&a.add(&b).unwrap().relu().exp().tanh().to_tensor())
}

/// ゲート状態を原状復帰する RAII ガード。
struct Restore<F: Fn(bool)> {
    original: bool,
    set: F,
}

impl<F: Fn(bool)> Drop for Restore<F> {
    fn drop(&mut self) {
        (self.set)(self.original);
    }
}

fn check_device<F: Fn(bool), O: BackendOps>(device: Device, set: F, ops: &O) {
    for (i, &n) in NUMELS.iter().enumerate() {
        let mut rng = Xorshift64Star::new(3_385_000 + i as u64);
        let a = Tensor::new(rng.fill_vec(n), &[n]).expect("valid tensor");
        let b = Tensor::new(rng.fill_vec(n), &[n]).expect("valid tensor");

        let cpu_ref = run_chain(&fandhe_ai::tape(), &a, &b);

        set(false);
        let off = run_chain(
            &fandhe_ai::tape_for(device).expect("tape_for（実機到達）"),
            &a,
            &b,
        );
        set(true);
        let on = run_chain(
            &fandhe_ai::tape_for(device).expect("tape_for（実機到達）"),
            &a,
            &b,
        );

        assert_eq!(bits(&off), bits(&on), "n={n}: 融合 ON は OFF と bit 一致");
        assert_parity(&format!("n={n}: CPU 参照"), &on, &cpu_ref);

        let direct = ops
            .run_fused(&build_4_chain_plan(n), &[&a, &b])
            .unwrap_or_else(|e| panic!("n={n}: run_fused が Ok を返さない: {e}"));
        assert_eq!(
            bits(&values(&direct)),
            bits(&on),
            "n={n}: 直接呼び出しと一致"
        );
    }
}

#[test]
#[ignore = "CUDA 実機（DGX Spark GB10 等）必須"]
fn cuda_api_enabled_fusion_matches_cpu_and_per_op() {
    let _restore = Restore {
        original: fandhe_ai::cuda_elementwise_fusion_enabled(),
        set: fandhe_ai::set_cuda_elementwise_fusion_enabled,
    };
    check_device(
        Device::Cuda(0),
        fandhe_ai::set_cuda_elementwise_fusion_enabled,
        &fandhe_ai_backend_cuda::CudaBackendOps::new(0),
    );
}

#[cfg(target_os = "macos")]
#[test]
#[ignore = "Metal 実機（M4 Max 等）必須"]
fn metal_api_enabled_fusion_matches_cpu_and_per_op() {
    let _restore = Restore {
        original: fandhe_ai::metal_elementwise_fusion_enabled(),
        set: fandhe_ai::set_metal_elementwise_fusion_enabled,
    };
    check_device(
        Device::Metal,
        fandhe_ai::set_metal_elementwise_fusion_enabled,
        &fandhe_ai_backend_metal::MetalBackendOps::new(),
    );
}
