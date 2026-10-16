//! イシュー #3116: `fandhe_ai::set_cuda_graph_infer_enabled(true)` で推論
//! チェーン capture を API から有効化した状態の実機 parity テスト
//! （CUDA 実機必須。`#[ignore]`）。
//!
//! 実行方法（プロセス内で最初に CUDA を初期化するのが本テストになるよう、
//! 単独・単スレッドで実行する。stream 種別は ordinal ごとに sticky なため）:
//!
//! ```sh
//! cargo test -p fandhe-ai --release --test cuda_graph_infer_parity_real_device \
//!   -- --ignored --nocapture --test-threads=1
//! ```
//!
//! 検証内容: (1) capture 1 回・replay 2 回という結線（黙って非 capture に
//! フォールバックしても parity は合格してしまうため必須）、(2) 3 回の出力が
//! bit 一致、(3) CPU `predict` 参照に対する統一複合判定
//! （`fandhe_ai_backend_cpu::parity::assert_parity`。tolerance は不変）、
//! (4) 非 capture 経路との bit 一致。実測と結果の記録は Phase 8 に委ねる。

use fandhe_ai::Device;
use fandhe_ai::compat::Sequential;
use fandhe_ai_backend_cpu::parity::assert_parity;
use fandhe_ai_backend_cuda::graph::{infer_graph_stats, override_infer_graph_for_scope};
use fandhe_ai_tensor_core::Tensor;

const BATCH: usize = 64;
const D_IN: usize = 784;
const D_HIDDEN: usize = 256;
const D_OUT: usize = 10;

fn values(t: &Tensor<f32>) -> Vec<f32> {
    t.contiguous()
        .as_slice()
        .expect("contiguous() 直後は必ず as_slice() が Some を返す")
        .to_vec()
}

fn bits(v: &[f32]) -> Vec<u32> {
    v.iter().map(|x| x.to_bits()).collect()
}

#[test]
#[ignore = "CUDA 実機（DGX Spark GB10 等）必須。--test-threads=1 で単独実行"]
fn api_enabled_infer_capture_matches_cpu_reference() {
    // CUDA デバイス初期化より前に有効化する（設定タイミングの契約）。
    fandhe_ai::set_cuda_graph_infer_enabled(true);

    let model = Sequential::new()
        .add_linear(D_IN, D_HIDDEN, 0x1111_1111)
        .unwrap()
        .add_relu()
        .add_linear(D_HIDDEN, D_OUT, 0x2222_2222)
        .unwrap();
    let data: Vec<f32> = (0..BATCH * D_IN)
        .map(|i| ((i * 2654435761usize % 1009) as f32 / 1009.0) - 0.5)
        .collect();
    let input = Tensor::new(data, &[BATCH, D_IN]).expect("test fixture: shape とデータ長は一致");

    let cpu_ref = values(&model.predict(&input).unwrap());

    let tape = fandhe_ai::tape_for(Device::Cuda(0))
        .expect("CUDA device 0 must be available on ignored test runner");
    let store = model.init_device_param_store(&tape).unwrap();
    drop(tape);

    let before = infer_graph_stats();
    let first = values(&model.predict_resident(&store, &input).unwrap());
    for i in 1..3 {
        let again = values(&model.predict_resident(&store, &input).unwrap());
        assert_eq!(bits(&first), bits(&again), "{i} 回目が 1 回目と bit 不一致");
    }
    let after = infer_graph_stats();
    assert_eq!(after.captured - before.captured, 1, "capture は 1 回");
    assert_eq!(after.replayed - before.replayed, 2, "replay は 2 回");

    assert_parity("api-enabled infer capture vs CPU", &first, &cpu_ref);

    // 非 capture 経路との bit 一致（thread-local override は hidden API）。
    let plain = {
        let _g = override_infer_graph_for_scope(false);
        values(&model.predict_resident(&store, &input).unwrap())
    };
    assert_eq!(
        bits(&first),
        bits(&plain),
        "capture と非 capture が bit 不一致"
    );

    fandhe_ai::set_cuda_graph_infer_enabled(false);
}
