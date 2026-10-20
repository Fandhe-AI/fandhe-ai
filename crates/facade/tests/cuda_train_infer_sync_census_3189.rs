//! イシュー #3189（`docs/perf/train-step-phase-breakdown.md` §19）: GB10 の CUDA で train／infer の
//! stream 同期回数・launch 回数を数えるための「計数用ドライバ」。
//!
//! 本ファイル自体は回数を数えない。`nsys profile --trace=cuda` の下で 1 プロセス 1 窓（`CENSUS_WINDOW`）
//! × 1 step 数（`CENSUS_STEPS`）を実行し、`orchestrate_gb10.sh` が `nsys stats` の CUDA API 集計から
//! 回数を取る。1 step あたりの回数は K=20 と K=120 の差分法（`(count(K=120) - count(K=20)) / 100`。
//! 初期化・初回コンパイル・確保の分を差分で消す。`docs/perf/logs/cuda-train-infer-gb10-3189/RULE.txt` 規則 1・2）。
//!
//! # 窓
//!
//! | `CENSUS_WINDOW` | 1 step の中身 |
//! |---|---|
//! | `fwd` | tape 構築 → `forward_resident` → `mse_loss` → loss 読み出し → `abandon_pending_forward` |
//! | `fwd_bwd` | `fwd` ＋ `backward_device_param_store`（step は呼ばない） |
//! | `full` | bench-fandhe `measure_train_reuse_phases` と同じ 1 step（backward ＋ `step_device_param_store`） |
//! | `infer_reuse` | `predict_resident` ＋ host 読み出し |
//!
//! モデル・データ・LR・seed は `mnist_scale_train_reuse_bench.rs`（bench-fandhe と同一手順）と同値。
//!
//! # 構成
//!
//! - CPU スモーク（`#[ignore]` なし・CI 実行）: 同じ 4 窓を CPU で回し、`PendingForwardUnconsumed`・
//!   勾配 staging の鮮度検査・poison に当たらないことを確かめる。
//! - CUDA 計数用（`#[ignore]`）: 環境変数で窓と step 数を受け取り、`CENSUS_JSON` を 1 行出す。
//!
//! 環境変数は固定集合だけを受け付け、それ以外は panic で拒否する（OWASP A03）。診断のみで本番コード・
//! tolerance・baseline は変更しない。

use bench_harness::rng::Xorshift64Star;
use fandhe_ai::compat::Sequential;
use fandhe_ai::{Device, DeviceParamStore, SgdConfig, Tensor};
use fandhe_ai_autodiff::nn::loss::{MseLoss, Reduction};

// `mnist_scale_train_reuse_bench.rs` と同値。
const BATCH: usize = 64;
const D_IN: usize = 784;
const D_HIDDEN: usize = 256;
const D_OUT: usize = 10;
const SEED_X: u64 = 0xDA7A_0001;
const SEED_Y: u64 = 0xDA7A_0002;
const SEED_L1: u64 = 0x1111_1111;
const SEED_L2: u64 = 0x2222_2222;
const LR: f32 = 0.01;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Window {
    Fwd,
    FwdBwd,
    Full,
    InferReuse,
}

impl Window {
    fn name(self) -> &'static str {
        match self {
            Window::Fwd => "fwd",
            Window::FwdBwd => "fwd_bwd",
            Window::Full => "full",
            Window::InferReuse => "infer_reuse",
        }
    }

    /// 固定集合の照合のみ。未知の値は拒否する。
    fn parse(s: &str) -> Option<Self> {
        match s {
            "fwd" => Some(Window::Fwd),
            "fwd_bwd" => Some(Window::FwdBwd),
            "full" => Some(Window::Full),
            "infer_reuse" => Some(Window::InferReuse),
            _ => None,
        }
    }
}

const ALL_WINDOWS: [Window; 4] = [
    Window::Fwd,
    Window::FwdBwd,
    Window::Full,
    Window::InferReuse,
];

fn tensor(data: Vec<f32>, shape: &[usize]) -> Tensor<f32> {
    Tensor::new(data, shape).expect("形状とデータ長は一致させている")
}

fn build_model() -> Sequential {
    Sequential::new()
        .add_linear(D_IN, D_HIDDEN, SEED_L1)
        .expect("Linear 構築")
        .add_relu()
        .add_linear(D_HIDDEN, D_OUT, SEED_L2)
        .expect("Linear 構築")
}

fn fnv(bits: impl Iterator<Item = u32>) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for b in bits {
        for byte in b.to_le_bytes() {
            h ^= u64::from(byte);
            h = h.wrapping_mul(0x0000_0100_0000_01b3);
        }
    }
    h
}

/// 窓を `steps` 回実行して `(loss_bits, checksum)` を返す。
fn run_window(device: Device, window: Window, steps: usize) -> (u32, u64) {
    let model = build_model();
    let x_data = tensor(
        Xorshift64Star::new(SEED_X).fill_vec(BATCH * D_IN),
        &[BATCH, D_IN],
    );
    let y_data = tensor(
        Xorshift64Star::new(SEED_Y).fill_vec(BATCH * D_OUT),
        &[BATCH, D_OUT],
    );
    let init_tape = fandhe_ai::tape_for(device).expect("tape_for");
    let mut store: DeviceParamStore = model.init_device_param_store(&init_tape).expect("store");
    let _ = init_tape
        .sync_device_param_store_to_host(&store)
        .expect("init sync");
    drop(init_tape);

    let config = SgdConfig::new(LR);
    let mut loss_bits = 0u32;
    let mut infer_sum = 0u64;
    for _ in 0..steps {
        if window == Window::InferReuse {
            let out = model.predict_resident(&store, &x_data).expect("predict");
            let host = out.contiguous();
            infer_sum = fnv(host
                .as_slice()
                .expect("contiguous 直後は as_slice が Some")
                .iter()
                .map(|v| v.to_bits()));
            continue;
        }
        let tape = fandhe_ai::tape_for(device).expect("tape_for");
        let x = tape.var(&x_data);
        let y = tape.var(&y_data);
        let pred = model
            .forward_resident(&tape, &x, &mut store)
            .expect("forward_resident");
        let loss = MseLoss::new(Reduction::Mean)
            .forward(&pred, &y)
            .expect("mse");
        loss_bits = loss.to_tensor().get(&[]).expect("scalar").to_bits();
        match window {
            Window::Fwd => {
                // step しない窓なので、保留中の forward を放棄して次の forward を可能にする。
                let _ = store.abandon_pending_forward();
            }
            Window::FwdBwd => {
                let _grads = tape
                    .backward_device_param_store(&loss, &store)
                    .expect("backward");
                let _ = store.abandon_pending_forward();
            }
            Window::Full => {
                let grads = tape
                    .backward_device_param_store(&loss, &store)
                    .expect("backward");
                tape.step_device_param_store(&mut store, &grads, &config)
                    .expect("step");
            }
            Window::InferReuse => unreachable!("continue 済み"),
        }
    }
    let final_tape = fandhe_ai::tape_for(device).expect("tape_for");
    let synced = final_tape
        .sync_device_param_store_to_host(&store)
        .expect("final sync");
    let train_sum = fnv(synced.iter().flat_map(|t| {
        t.contiguous()
            .as_slice()
            .expect("contiguous 直後は as_slice が Some")
            .iter()
            .map(|v| v.to_bits())
            .collect::<Vec<_>>()
    }));
    let sum = if window == Window::InferReuse {
        infer_sum
    } else {
        train_sum
    };
    (loss_bits, sum)
}

#[test]
fn census_windows_run_repeatedly_on_cpu_without_state_machine_errors() {
    for w in ALL_WINDOWS {
        let a = run_window(Device::Cpu, w, 3);
        let b = run_window(Device::Cpu, w, 3);
        assert_eq!(a, b, "窓 {} が CPU で決定的でない", w.name());
    }
}

#[test]
fn census_window_parse_rejects_unknown() {
    assert!(Window::parse("fwd").is_some());
    assert!(Window::parse("full ").is_none());
    assert!(Window::parse("").is_none());
    assert!(Window::parse("../x").is_none());
}

#[test]
#[ignore = "CUDA 実機必須の計数用ドライバ（nsys の下で orchestrate_gb10.sh が起動）"]
fn cuda_train_infer_sync_census() {
    let kind = std::env::var("CENSUS_KIND").unwrap_or_else(|_| "cuda".to_string());
    assert_eq!(kind, "cuda", "CENSUS_KIND は cuda のみ（CPU 混入の拒否）");
    let window = std::env::var("CENSUS_WINDOW")
        .ok()
        .as_deref()
        .and_then(Window::parse)
        .expect("CENSUS_WINDOW は fwd|fwd_bwd|full|infer_reuse のみ");
    let steps: usize = match std::env::var("CENSUS_STEPS").as_deref() {
        Ok("20") => 20,
        Ok("120") => 120,
        other => panic!("CENSUS_STEPS は 20|120 のみ: {other:?}"),
    };
    let (loss_bits, checksum) = run_window(Device::Cuda(0), window, steps);
    println!(
        "CENSUS_JSON {{\"window\":\"{}\",\"steps\":{steps},\"loss_bits\":{loss_bits},\"checksum_bits\":{checksum}}}",
        window.name()
    );
}
