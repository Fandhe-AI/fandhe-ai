//! イシュー #2155 の事前登録 A/B マイクロベンチ（`#[ignore]`。実機実測
//! 専用）。
//!
//! `docs/perf/logs/gpu-min-log-softmax-2155/README.md`「事前登録判定
//! 規則」の (3)（5 run 中央値・非後退 ratio<=1.00）・(2)（GPU 経路の
//! 5 run 間 checksum 完全一致）を計測する。公開 API（`fandhe_ai::
//! tape_for`・`Var::min`・`Var::log_softmax`）のみを使い、base の
//! commit（イシュー #2155 適用前。GPU カーネル未結線でホスト
//! フォールバック経由）へ同じファイルを持ち込めば A/B が取れる設計
//! （`elementwise_vjp_bench.rs` と同じ「公開 API 経由の forward
//! 単体計測」方針。backward は対象外——`min`／`log_softmax` forward
//! の GPU カーネル化がスコープのため）。
//!
//! `FANDHE_BENCH_DEVICE` 環境変数（`cpu`〈既定〉／`metal`／`cuda`）で
//! 対象デバイスを切り替える（`elementwise_vjp_bench.rs` と同じ方式）。
//! 出力は before/after で diff 可能な固定形式
//! （`bench[<case>].median_s=`・`bench[<case>].checksum=`）とする。
//!
//! 固定形状（事前登録）: `min` は `[4096, 4096]` の `dim=None`／
//! `dim=0`／`dim=1`、`log_softmax` は `[4096, 4096]` の `dim=1` と
//! `[256, 32768]` の `dim=1`（CUDA 2 パス・Metal 2 パス経路を強制）。

use bench_harness::median_q1_q3;
use bench_harness::rng::Xorshift64Star;
use fandhe_ai::{Device, Tensor};
use std::time::Instant;

const WARMUP: usize = 2;
const RUNS: usize = 5;

fn random_tensor(seed: u64, shape: &[usize]) -> Tensor<f32> {
    let numel: usize = shape.iter().product();
    let data = Xorshift64Star::new(seed).fill_vec(numel);
    Tensor::new(data, shape).unwrap()
}

/// `FANDHE_BENCH_DEVICE`（`cpu`〈既定〉／`metal`／`cuda`）からデバイスを
/// 解決する（`elementwise_vjp_bench.rs::device_from_env` と同型）。
fn device_from_env() -> Device {
    match std::env::var("FANDHE_BENCH_DEVICE").as_deref() {
        Ok("cpu") | Err(_) => Device::Cpu,
        #[cfg(target_os = "macos")]
        Ok("metal") => Device::Metal,
        Ok("cuda") => Device::Cuda(0),
        Ok(other) => panic!("FANDHE_BENCH_DEVICE: 未対応の値 '{other}'"),
    }
}

/// 出力テンソルの `to_bits()` を fold した診断用チェックサム（FNV-1a
/// 相当。`elementwise_vjp_bench.rs::fold_bits` と同一実装）。
fn fold_bits(t: &Tensor<f32>) -> u64 {
    let mut acc: u64 = 0xcbf29ce484222325;
    for &v in t.host_slice().iter() {
        let bits = v.to_bits() as u64;
        acc ^= bits;
        acc = acc.wrapping_mul(0x100000001b3);
    }
    acc
}

/// 1 ケース分の forward 計測: `WARMUP` 回のウォームアップ後、`RUNS`
/// 回計測し中央値・各 run の checksum を出力する（判定規則 (2)「5 run
/// 間で checksum 完全一致」は出力の `run[<label>][<i>].checksum=` 行を
/// 突合すれば確認できる）。
fn run_case(label: &str, device: Device, mut build: impl FnMut(Device) -> Tensor<f32>) {
    let mut samples = Vec::with_capacity(RUNS);
    let mut checksums = Vec::with_capacity(RUNS);

    for i in 0..(WARMUP + RUNS) {
        let start = Instant::now();
        let out = build(device);
        let elapsed = start.elapsed().as_secs_f64();
        let checksum = fold_bits(&out);
        std::hint::black_box(&out);

        if i >= WARMUP {
            samples.push(elapsed);
            checksums.push(checksum);
            println!("run[{label}][{}].checksum={checksum:#018x}", i - WARMUP);
        }
    }

    let q = median_q1_q3(&samples).unwrap();
    println!("bench[{label}].median_s={:.9}", q.median);
    let all_equal = checksums.windows(2).all(|w| w[0] == w[1]);
    println!("bench[{label}].checksum_stable={all_equal}");
}

#[test]
#[ignore = "実機実測専用（FANDHE_BENCH_DEVICE で対象デバイス指定。イシュー #2155）"]
fn min_forward_fixed_shapes() {
    let device = device_from_env();
    let shape = [4096usize, 4096];
    let data = random_tensor(0x2155_0001, &shape);

    for (label, dim) in [
        ("min_all", None),
        ("min_dim0", Some(0usize)),
        ("min_dim1", Some(1)),
    ] {
        run_case(label, device, |device| {
            let tape = fandhe_ai::tape_for(device).unwrap_or_else(|err| {
                panic!("tape_for({device:?}) に失敗した（実機非到達の可能性）: {err}")
            });
            let x = tape.var(&data);
            x.min(dim).unwrap().to_tensor()
        });
    }
}

#[test]
#[ignore = "実機実測専用（FANDHE_BENCH_DEVICE で対象デバイス指定。イシュー #2155）"]
fn log_softmax_forward_fixed_shapes() {
    let device = device_from_env();

    for (label, shape) in [
        ("log_softmax_4096x4096", [4096usize, 4096usize]),
        ("log_softmax_256x32768", [256usize, 32768usize]),
    ] {
        let data = random_tensor(0x2155_0002, &shape);
        run_case(label, device, |device| {
            let tape = fandhe_ai::tape_for(device).unwrap_or_else(|err| {
                panic!("tape_for({device:?}) に失敗した（実機非到達の可能性）: {err}")
            });
            let x = tape.var(&data);
            x.log_softmax(1).unwrap().to_tensor()
        });
    }
}
