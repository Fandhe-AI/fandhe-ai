//! イシュー #2115（親 #1580 系・`docs/perf/infer-chain-graphcapture-cuda-ab.md`）:
//! 推論 forward チェーンの CUDA Graph capture／replay
//! （`CudaBackendOps::linear_chain_forward_captured`）の実機検証（R1）。
//!
//! **実行手順**: opt-in は最初の CUDA デバイス初期化より前に環境変数で
//! 与える必要がある（created stream は ordinal ごとに sticky。
//! `crates/backend-cuda/src/device.rs`）。
//!
//! ```sh
//! FANDHE_AI_CUDA_GRAPH_INFER=1 cargo test -p fandhe-ai-backend-cuda --release \
//!   --test infer_graph_capture_real_device -- --ignored --nocapture --test-threads=1
//! ```
//!
//! `internal-diagnostics` feature を付けずにビルドすること（同 feature は常に legacy
//! stream に固定するため capture 不能）。環境変数が未設定（`infer_graph_enabled() == false`）の場合、各テストは
//! 理由を標準エラーへ出して**早期 return する**（`make test-ignored-cuda` の
//! 一括実行で ON 前提テストが誤って落ちないため）。R1 の判定では
//! 「`INFER_CAPTURE_R1: PASS` 行が全テスト分出ていること」を確認する
//! （skip は PASS 行を出さない）。
//!
//! 検証項目（design: `docs/inference-chain-single-sync-design.md` §11）:
//! 1. 1 回目 `captured`+1・2 回目以降 `replayed`+1、かつ
//!    `override_infer_graph_for_scope(false)` の非 capture チェーン出力と
//!    `to_bits` で完全一致する
//! 2. weight を同一アドレスのまま更新（`upload_into`）した後の replay に
//!    新しい weight が反映される（アドレス焼き込みの正当性）
//! 3. batch を変えると別 key で再 capture される
//! 4. ワーカースレッドでキャッシュを埋めて終了しても abort しない
//! 5. 不適用入力（rank≠2）は `Ok(None)`

use fandhe_ai_backend_cuda::graph::{
    infer_graph_enabled, infer_graph_stats, override_infer_graph_for_scope,
};
use fandhe_ai_backend_cuda::{CudaBackendOps, CudaDevice};
use fandhe_ai_tensor_core::buffer::{DeviceBuffer, DeviceBufferView, MemoryOps};
use fandhe_ai_tensor_core::{Activation, BackendOps, DispatchFailureCell, Tensor};

const D_IN: usize = 784;
const D_HIDDEN: usize = 256;
const D_OUT: usize = 10;

/// 決定的疑似乱数（`[-0.5, 0.5)`）。
fn fill(seed: u64, len: usize) -> Vec<f32> {
    let mut state = seed;
    (0..len)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            ((state >> 11) as f64 / (1u64 << 53) as f64) as f32 - 0.5
        })
        .collect()
}

fn bits(t: &Tensor<f32>) -> Vec<u32> {
    t.contiguous()
        .as_slice()
        .unwrap()
        .iter()
        .map(|v| v.to_bits())
        .collect()
}

/// flat params の各区間（`DeviceParamStore` の連結バッファと同型）。
const W1_OFF: usize = 0;
const B1_OFF: usize = W1_OFF + D_IN * D_HIDDEN;
const W2_OFF: usize = B1_OFF + D_HIDDEN;
const B2_OFF: usize = W2_OFF + D_HIDDEN * D_OUT;
const TOTAL: usize = B2_OFF + D_OUT;

struct Fixture {
    ops: CudaBackendOps,
    params: DeviceBuffer<f32>,
}

impl Fixture {
    fn new(seed: u64) -> Self {
        let device =
            CudaDevice::new(0).expect("CUDA device 0 must be available on ignored test runner");
        let ops = CudaBackendOps::new(device.ordinal());
        let mem = ops.memory_ops().expect("CUDA memory_ops");
        let flat = Tensor::new(fill(seed, TOTAL), &[TOTAL]).unwrap();
        let params = mem.upload(&flat).unwrap();
        Self { ops, params }
    }

    fn mem(&self) -> &dyn MemoryOps {
        self.ops.memory_ops().expect("CUDA memory_ops")
    }

    /// 全層の (w, bias, act)。
    fn layers(
        &self,
    ) -> Vec<(
        DeviceBufferView<'_>,
        Option<DeviceBufferView<'_>>,
        Activation,
    )> {
        let p = &self.params;
        vec![
            (
                DeviceBufferView::new(p, W1_OFF, &[D_IN, D_HIDDEN]).unwrap(),
                Some(DeviceBufferView::new(p, B1_OFF, &[D_HIDDEN]).unwrap()),
                Activation::Relu,
            ),
            (
                DeviceBufferView::new(p, W2_OFF, &[D_HIDDEN, D_OUT]).unwrap(),
                Some(DeviceBufferView::new(p, B2_OFF, &[D_OUT]).unwrap()),
                Activation::None,
            ),
        ]
    }

    /// 非 capture の既存チェーン（`upload` → 層ごと
    /// `linear_forward_device_tracked` → `download`）。
    fn reference(&self, input: &Tensor<f32>) -> Tensor<f32> {
        let token = DispatchFailureCell::new();
        let mut cur = self.mem().upload(input).unwrap();
        for (w, b, act) in self.layers() {
            cur = self
                .ops
                .linear_forward_device_tracked(&cur, w, b, act, &token)
                .unwrap();
        }
        self.mem().download(&cur).unwrap()
    }

    fn captured(&self, input: &Tensor<f32>) -> Option<Tensor<f32>> {
        self.ops
            .linear_chain_forward_captured(input, &self.layers())
            .unwrap()
    }
}

fn input(batch: usize, seed: u64) -> Tensor<f32> {
    Tensor::new(fill(seed, batch * D_IN), &[batch, D_IN]).unwrap()
}

/// opt-in 未設定、または対象デバイスが capture 不能な legacy stream
/// （`internal-diagnostics` ビルドでは常にそうなる）なら理由を出して false
/// （呼び出し側は return する。PASS 行は出さない）。
fn on_or_skip(name: &str) -> bool {
    if !infer_graph_enabled() {
        eprintln!(
            "INFER_CAPTURE_R1: SKIP {name} (set FANDHE_AI_CUDA_GRAPH_INFER=1 before process start)"
        );
        return false;
    }
    let capturable = CudaDevice::new(0)
        .map(|d| d.is_capturable_stream())
        .unwrap_or(false);
    if !capturable {
        eprintln!(
            "INFER_CAPTURE_R1: SKIP {name} (device 0 is not on a capturable stream; build without \
             the `internal-diagnostics` feature and set the env var before the first device init)"
        );
    }
    capturable
}

#[test]
#[ignore = "CUDA 実機必須。FANDHE_AI_CUDA_GRAPH_INFER=1 で起動すること"]
fn chain_capture_then_replay_is_bit_identical_to_uncaptured_chain() {
    if !on_or_skip("chain_capture_then_replay") {
        return;
    }
    let fx = Fixture::new(0xA11CE);
    let x = input(64, 0xBEEF);
    let expected = {
        let _off = override_infer_graph_for_scope(false);
        assert!(fx.captured(&x).is_none(), "OFF は不適用（Ok(None)）");
        fx.reference(&x)
    };

    let s0 = infer_graph_stats();
    let first = fx.captured(&x).expect("capturable stream 上では Some");
    let s1 = infer_graph_stats();
    assert_eq!(s1.captured - s0.captured, 1, "1 回目は capture");
    assert_eq!(s1.replayed - s0.replayed, 0);
    assert_eq!(
        bits(&first),
        bits(&expected),
        "capture 直後の出力が bit 不一致"
    );

    for i in 0..3 {
        let again = fx.captured(&x).unwrap();
        assert_eq!(bits(&again), bits(&expected), "replay {i} が bit 不一致");
    }
    let s2 = infer_graph_stats();
    assert_eq!(s2.captured - s1.captured, 0);
    assert_eq!(s2.replayed - s1.replayed, 3);
    assert_eq!(s2.graph_launches - s0.graph_launches, 4);
    println!("INFER_CAPTURE_R1: PASS chain_capture_then_replay");
}

#[test]
#[ignore = "CUDA 実機必須。FANDHE_AI_CUDA_GRAPH_INFER=1 で起動すること"]
fn replay_reflects_in_place_weight_update() {
    if !on_or_skip("replay_reflects_in_place_weight_update") {
        return;
    }
    let mut fx = Fixture::new(0xB0B);
    let x = input(64, 0xCAFE);
    let before = fx.captured(&x).unwrap();
    // 同一アドレス（flat params）のまま weight を書き換える（`step()` 相当）。
    let new_w = Tensor::new(fill(0xD00D, D_IN * D_HIDDEN), &[D_IN * D_HIDDEN]).unwrap();
    fx.ops
        .memory_ops()
        .unwrap()
        .upload_into(&new_w, &mut fx.params, W1_OFF)
        .unwrap();

    let s0 = infer_graph_stats();
    let after = fx.captured(&x).unwrap();
    let s1 = infer_graph_stats();
    assert_eq!(s1.replayed - s0.replayed, 1, "更新後も同じ graph を replay");
    let expected = {
        let _off = override_infer_graph_for_scope(false);
        fx.reference(&x)
    };
    assert_eq!(
        bits(&after),
        bits(&expected),
        "replay に新 weight が反映されない"
    );
    assert_ne!(
        bits(&before),
        bits(&after),
        "テスト自体が空振り（更新が出力に効いていない）"
    );
    println!("INFER_CAPTURE_R1: PASS replay_reflects_in_place_weight_update");
}

#[test]
#[ignore = "CUDA 実機必須。FANDHE_AI_CUDA_GRAPH_INFER=1 で起動すること"]
fn different_batch_recaptures_under_a_different_key() {
    if !on_or_skip("different_batch_recaptures") {
        return;
    }
    let fx = Fixture::new(0xFEED);
    let x64 = input(64, 1);
    let x128 = input(128, 2);
    let _ = fx.captured(&x64).unwrap();
    let s0 = infer_graph_stats();
    let out = fx.captured(&x128).unwrap();
    let s1 = infer_graph_stats();
    assert_eq!(s1.captured - s0.captured, 1, "batch 変更は再 capture");
    let expected = {
        let _off = override_infer_graph_for_scope(false);
        fx.reference(&x128)
    };
    assert_eq!(bits(&out), bits(&expected));
    // 元の batch は evict されていなければ replay になる。
    let s2 = infer_graph_stats();
    let _ = fx.captured(&x64).unwrap();
    let s3 = infer_graph_stats();
    assert_eq!(s3.replayed - s2.replayed, 1);
    println!("INFER_CAPTURE_R1: PASS different_batch_recaptures");
}

#[test]
#[ignore = "CUDA 実機必須。FANDHE_AI_CUDA_GRAPH_INFER=1 で起動すること"]
fn worker_thread_exit_with_populated_cache_does_not_abort() {
    if !on_or_skip("worker_thread_exit") {
        return;
    }
    // `DeviceBuffer` は `!Send` のため、Fixture はワーカー内で構築・破棄する。
    // キャッシュ（thread-local）は Fixture の破棄後、スレッド終了時の
    // TLS 破棄まで生き残る（graph が解放済み weight を指す状態での破棄）。
    let x = input(32, 3);
    let x2 = x.clone();
    let worker_bits = std::thread::spawn(move || {
        let fx = Fixture::new(0x7EA);
        let a = fx.captured(&x2).unwrap();
        let b = fx.captured(&x2).unwrap();
        assert_eq!(bits(&a), bits(&b));
        bits(&a)
    })
    .join()
    .expect("ワーカースレッドの終了（TLS 破棄）で panic/abort しない");
    // 終了後もこのスレッドでチェーンが使える（ordinal が壊れていない）。
    let fx = Fixture::new(0x7EA);
    let expected = {
        let _off = override_infer_graph_for_scope(false);
        fx.reference(&x)
    };
    assert_eq!(worker_bits, bits(&expected));
    assert_eq!(bits(&fx.captured(&x).unwrap()), bits(&expected));
    println!("INFER_CAPTURE_R1: PASS worker_thread_exit");
}

#[test]
#[ignore = "CUDA 実機必須。FANDHE_AI_CUDA_GRAPH_INFER=1 で起動すること"]
fn inapplicable_inputs_return_none() {
    if !on_or_skip("inapplicable_inputs") {
        return;
    }
    let fx = Fixture::new(0x1234);
    let rank1 = Tensor::new(fill(9, D_IN), &[D_IN]).unwrap();
    assert!(fx.captured(&rank1).is_none(), "rank≠2 は不適用");
    let wrong_k = Tensor::new(fill(9, 4 * (D_IN - 1)), &[4, D_IN - 1]).unwrap();
    assert!(fx.captured(&wrong_k).is_none(), "shape 連鎖不一致は不適用");
    let empty = Tensor::new(Vec::new(), &[0, D_IN]).unwrap();
    assert!(fx.captured(&empty).is_none(), "m == 0 は不適用");
    println!("INFER_CAPTURE_R1: PASS inapplicable_inputs");
}
