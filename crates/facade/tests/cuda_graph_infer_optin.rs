//! `fandhe_ai::set_cuda_graph_infer_enabled`／`fandhe_ai::cuda_graph_infer_enabled`
//! （イシュー #3116。公開形は #3115 の
//! `docs/facade-graph-tier-b-optin-exposure-decision.md` §3・§4）の受入テスト。
//!
//! `cuda_graph_step_optin.rs` と同じ理由で、setter/getter の往復自体は
//! CUDA driver を要さない（`fandhe_ai_backend_cuda::graph` の `AtomicBool`
//! はプロセスグローバルで `CudaDevice` 初期化を経由しない）ため、CUDA
//! 非搭載の CI でも常に実行できる。本テストは CUDA デバイスを初期化しない。
//!
//! ON 時の実 capture・replay と CPU 参照に対する parity は
//! `cuda_graph_infer_parity_real_device.rs`（`#[ignore]`。実機必須）が担い、
//! 責務を分離する。

use fandhe_ai::compat::Sequential;
use fandhe_ai_tensor_core::Tensor;

/// フラグはプロセスグローバルのため、並列テスト間で直列化・原状復帰する
/// RAII ガード（`cuda_graph_step_optin.rs::GraphStepFlagGuard` と同型）。
struct InferFlagGuard {
    _lock: std::sync::MutexGuard<'static, ()>,
    original: bool,
}

impl InferFlagGuard {
    fn acquire() -> Self {
        static LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());
        let lock = LOCK.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
        let original = fandhe_ai::cuda_graph_infer_enabled();
        Self {
            _lock: lock,
            original,
        }
    }
}

impl Drop for InferFlagGuard {
    fn drop(&mut self) {
        fandhe_ai::set_cuda_graph_infer_enabled(self.original);
    }
}

/// 既定無効の契約を、明示的に `false` へ戻した直後の観測で確認する
/// （他テストが有効化したまま残す可能性があるため、起動直後の真の初期値
/// そのものは検証しない。`cuda_graph_step_optin.rs::default_is_disabled`
/// と同じ理由）。
#[test]
fn default_is_disabled() {
    let _guard = InferFlagGuard::acquire();
    fandhe_ai::set_cuda_graph_infer_enabled(false);
    assert!(!fandhe_ai::cuda_graph_infer_enabled());
}

/// setter/getter の往復契約。
#[test]
fn set_true_then_false_round_trips() {
    let _guard = InferFlagGuard::acquire();
    fandhe_ai::set_cuda_graph_infer_enabled(true);
    assert!(fandhe_ai::cuda_graph_infer_enabled());
    fandhe_ai::set_cuda_graph_infer_enabled(false);
    assert!(!fandhe_ai::cuda_graph_infer_enabled());
}

fn bits(t: &Tensor<f32>) -> Vec<u32> {
    t.contiguous()
        .as_slice()
        .expect("contiguous() 直後は必ず as_slice() が Some を返す")
        .iter()
        .map(|v| v.to_bits())
        .collect()
}

/// CPU のみの環境で OFF → ON → OFF と切り替えても panic せず、CPU `predict`
/// の出力が bit 完全に一致する（非対応環境では効果を持たない契約）。
#[test]
fn toggling_on_cpu_keeps_output_bit_identical() {
    let _guard = InferFlagGuard::acquire();
    let model = Sequential::new()
        .add_linear(8, 16, 0x3116_0001)
        .unwrap()
        .add_relu()
        .add_linear(16, 4, 0x3116_0002)
        .unwrap();
    let data: Vec<f32> = (0..5 * 8)
        .map(|i| ((i * 37 % 101) as f32 - 50.0) / 50.0)
        .collect();
    let input = Tensor::new(data, &[5, 8]).expect("test fixture: shape とデータ長は一致");

    fandhe_ai::set_cuda_graph_infer_enabled(false);
    let off1 = model.predict(&input).unwrap();
    fandhe_ai::set_cuda_graph_infer_enabled(true);
    let on = model.predict(&input).unwrap();
    fandhe_ai::set_cuda_graph_infer_enabled(false);
    let off2 = model.predict(&input).unwrap();

    assert_eq!(bits(&off1), bits(&on));
    assert_eq!(bits(&off1), bits(&off2));
}
