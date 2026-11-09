//! G-CUDA-INF（GB10・推論 batch 64）の導出値 `kernel_wait`（約 63 µs）を、GPU 実行の幅と
//! 待ち（起動遅延・間隙・同期復帰）に分けて記録する診断 example（イシュー #4150。GB10 実機専用）。
//!
//! # 役割と呼び出し文脈
//!
//! - `docs/perf/logs/diag-0140-cuda-inf-wait-4150/run_round.sh` が 1 round につき 1 プロセスで起動し、
//!   標準出力の `CHAIN4150` 行を同ディレクトリの `aggregate_4150.py` が厳密にパースして内訳にする。
//!   規則の正は同ディレクトリの `RULE.txt`（計測前に固定）。record_only の内訳であり合否は判定しない。
//! - チェーンは `tests/infer_chain_phase_diag_3189.rs`（#3189・`docs/perf/infer-reuse-phase-breakdown.md` §12）と
//!   同じ公開 trait の呼び出し列（upload → L1 784→256＋bias＋ReLU → L2 256→10＋bias → download）。
//!   #4147（`cuda_mlp_event_bench_4147`）は純 GEMM 入口の event 計測で、本 example は bias・活性化融合込みの
//!   実物 `linear_forward_device` を挟む点が境界差。
//!
//! # 腕
//!
//! - P: host `Instant` のみ（3189 と同じ。`kernel_wait_derived = download - download_again` の再現）。
//! - E: P と同じチェーンに CUDA event を 4 本差す（GPU 側の幅。host 時計も同時に取る）。
//! - I: drain 後に event 対で `linear_forward_device` 1 回を層ごとに挟む（単体の GPU 実行幅）。
//! - F: 固定費（idle sync・1x1 往復・1x1 発行・event 対の床）。
//!
//! 反復ごとに P/E/I の実行順を巡回させて位置バイアスを均す。
//!
//! # 意図的な逸脱
//!
//! `internal-diagnostics` 有効時、`CudaDevice` は常に Legacy（NULL）stream を使う（`device.rs`）。
//! NULL stream はプロセス内で単一のため、本 example が `CudaDevice::new(0)` から得た stream に
//! event を記録すると `CudaBackendOps` の本番チェーンと同じ stream に差し込める。冒頭で
//! `cu_stream()` が null であることを検査し、違えば停止する。
//!
//! # 不変事項
//!
//! 本番ディスパッチ・NVRTC カーネル文字列・REQ-8 の手動境界検査・FMA 契約・tolerance は変更しない。
//! 新規 `unsafe` は 0 件（event は cudarc の safe API）。入力は `--round <1..=999>` のみ（A03）。
//!
//! # 実行方法
//!
//! `cargo run --release -p fandhe-ai-backend-cuda --features internal-diagnostics
//! --example cuda_infer_chain_event_diag_4150 -- --round 1`

use std::time::Instant;

use bench_harness::rng::Xorshift64Star;
use cudarc::driver::sys::CUevent_flags;
use fandhe_ai_backend_cuda::{CudaBackendOps, CudaDevice};
use fandhe_ai_tensor_core::buffer::{DeviceBuffer, DeviceBufferView};
use fandhe_ai_tensor_core::{Activation, BackendOps, Tensor};

const BATCH: usize = 64;
const D_IN: usize = 784;
const D_HIDDEN: usize = 256;
const D_OUT: usize = 10;
const WARMUP: usize = 50;
const ITERS: usize = 200;
const W1S: [usize; 2] = [D_IN, D_HIDDEN];
const B1S: [usize; 1] = [D_HIDDEN];
const W2S: [usize; 2] = [D_HIDDEN, D_OUT];
const B2S: [usize; 1] = [D_OUT];
const W11S: [usize; 2] = [1, 1];
/// 腕の個数（P・E・I。F は各反復の末尾に固定順で測る）。
const ARMS: usize = 3;

/// 中央値（偶数個は中央 2 要素の平均）。単体テストと将来の host 側要約用。
#[cfg(test)]
fn median(samples: &[f64]) -> f64 {
    let mut v = samples.to_vec();
    v.sort_by(|a, b| a.partial_cmp(b).expect("samples must be NaN-free"));
    let n = v.len();
    if n % 2 == 1 {
        v[n / 2]
    } else {
        (v[n / 2 - 1] + v[n / 2]) / 2.0
    }
}

/// `i` 回目の反復で腕の実行順を巡回させる（位置バイアスを均す）。
fn rotated_order(count: usize, i: usize) -> Vec<usize> {
    (0..count).map(|j| (j + i) % count).collect()
}

/// `--round <n>` だけを受け付ける。不正・未知の引数は `Err`（呼び出し側が停止する）。
fn parse_args(args: &[String]) -> Result<u32, String> {
    match args {
        [flag, val] if flag == "--round" => {
            let n: u32 = val
                .parse()
                .map_err(|_| format!("--round が数値でない: {val}"))?;
            if (1..=999).contains(&n) {
                Ok(n)
            } else {
                Err(format!("--round は 1..=999: {n}"))
            }
        }
        _ => Err("使い方: --round <1..=999>".to_string()),
    }
}

/// 集計が厳格パースする 1 行を作る。値は µs（小数 3 桁）。
fn format_line(arm: char, iter: usize, kv: &[(&str, f64)]) -> String {
    let mut s = format!("CHAIN4150 arm={arm} iter={iter}");
    for (k, v) in kv {
        s.push_str(&format!(" {k}={v:.3}"));
    }
    s
}

fn us(secs: f64) -> f64 {
    secs * 1e6
}

fn tensor(seed: u64, shape: &[usize]) -> Tensor<f32> {
    let n: usize = shape.iter().product();
    Tensor::new(Xorshift64Star::new(seed).fill_vec(n), shape)
        .expect("形状とデータ長は一致させている")
}

/// FNV-1a（決定的・依存なし）。checksum 表示用で、bit 一致の判定は配列比較で行う。
fn bits_hash(v: &[u32]) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for x in v {
        for b in x.to_le_bytes() {
            h ^= u64::from(b);
            h = h.wrapping_mul(0x0000_0100_0000_01b3);
        }
    }
    h
}

fn bits(t: &Tensor<f32>) -> Vec<u32> {
    t.contiguous()
        .as_slice()
        .expect("contiguous 直後は as_slice が Some")
        .iter()
        .map(|v| v.to_bits())
        .collect()
}

fn timed<T>(f: impl FnOnce() -> T) -> (T, f64) {
    let t = Instant::now();
    let out = f();
    (out, t.elapsed().as_secs_f64())
}

/// 常駐 w・bias（本番では `DeviceParamStore` が保持する resident leaf に相当）。
struct Params {
    w1: DeviceBuffer<f32>,
    b1: DeviceBuffer<f32>,
    w2: DeviceBuffer<f32>,
    b2: DeviceBuffer<f32>,
}

fn upload_params(ops: &CudaBackendOps) -> Params {
    let mem = ops.memory_ops().expect("CUDA MemoryOps");
    let up = |seed: u64, shape: &[usize]| mem.upload(&tensor(seed, shape)).expect("常駐 upload");
    Params {
        w1: up(0x1111_1111, &W1S),
        b1: up(0x1111_1111 ^ 0xB1A5, &B1S),
        w2: up(0x2222_2222, &W2S),
        b2: up(0x2222_2222 ^ 0xB1A5, &B2S),
    }
}

type Event = cudarc::driver::CudaEvent;

/// E 腕で差す event（upload 後・L1 発行直後・L2 発行直後。upload 前の e0 は別引数）。
struct Probe<'a> {
    e0: &'a Event,
    after_upload: &'a Event,
    after_l1: &'a Event,
    after_l2: &'a Event,
}

struct ChainOut {
    out: Tensor<f32>,
    h2d_upload: f64,
    l1_issue: f64,
    l2_issue: f64,
    download: f64,
    download_again: f64,
}

fn l1(ops: &CudaBackendOps, p: &Params, a: &DeviceBuffer<f32>) -> DeviceBuffer<f32> {
    ops.linear_forward_device(
        a,
        DeviceBufferView::new(&p.w1, 0, &W1S).expect("w1 view"),
        Some(DeviceBufferView::new(&p.b1, 0, &B1S).expect("b1 view")),
        Activation::Relu,
    )
    .expect("L1")
}

fn l2(ops: &CudaBackendOps, p: &Params, h: &DeviceBuffer<f32>) -> DeviceBuffer<f32> {
    ops.linear_forward_device(
        h,
        DeviceBufferView::new(&p.w2, 0, &W2S).expect("w2 view"),
        Some(DeviceBufferView::new(&p.b2, 0, &B2S).expect("b2 view")),
        Activation::None,
    )
    .expect("L2")
}

fn tiny(ops: &CudaBackendOps, a: &DeviceBuffer<f32>, w: &DeviceBuffer<f32>) -> DeviceBuffer<f32> {
    ops.linear_forward_device(
        a,
        DeviceBufferView::new(w, 0, &W11S).expect("w11 view"),
        None,
        Activation::None,
    )
    .expect("1x1")
}

/// 本番チェーンと同じ呼び出し列を 1 回回す。`probe` があれば event を差す（P 腕は None）。
/// event の記録は各層の発行直後（drop より前）に置き、host 側の drop 遅延が GPU 側時計へ入らないようにする。
fn chain(
    ops: &CudaBackendOps,
    stream: &cudarc::driver::CudaStream,
    p: &Params,
    input: &Tensor<f32>,
    probe: Option<&Probe<'_>>,
) -> ChainOut {
    let mem = ops.memory_ops().expect("CUDA MemoryOps");
    if let Some(pr) = probe {
        pr.e0.record(stream).expect("rec");
    }
    let (a, h2d_upload) = timed(|| mem.upload(input).expect("入力 H2D"));
    if let Some(pr) = probe {
        pr.after_upload.record(stream).expect("rec");
    }
    // 本番どおり L1 発行の直後に入力 a を drop し、その費用も l1_issue に含める。
    let (h, l1_issue) = timed(|| {
        let h = l1(ops, p, &a);
        if let Some(pr) = probe {
            pr.after_l1.record(stream).expect("rec");
        }
        drop(a);
        h
    });
    let (y, l2_issue) = timed(|| {
        let y = l2(ops, p, &h);
        if let Some(pr) = probe {
            pr.after_l2.record(stream).expect("rec");
        }
        drop(h);
        y
    });
    let (out, download) = timed(|| mem.download(&y).expect("D2H"));
    let (_again, download_again) = timed(|| mem.download(&y).expect("D2H（2 回目）"));
    drop(y);
    ChainOut {
        out,
        h2d_upload,
        l1_issue,
        l2_issue,
        download,
        download_again,
    }
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let round = match parse_args(&args) {
        Ok(r) => r,
        Err(e) => {
            println!("ERROR4150 {e}");
            std::process::exit(2);
        }
    };
    let device = match CudaDevice::new(0) {
        Ok(d) => d,
        Err(e) => {
            println!("ERROR4150 CudaDevice::new failed: {e}");
            std::process::exit(2);
        }
    };
    let stream = device.stream().clone();
    if !stream.cu_stream().is_null() {
        println!(
            "ERROR4150 stream が NULL stream でない（internal-diagnostics の Legacy 前提が崩れた）"
        );
        std::process::exit(2);
    }
    let (maj, min) = device.compute_capability();
    println!(
        "DEVICE4150 compute_capability={maj}.{min} round={round} warmup={WARMUP} iters={ITERS}"
    );
    let ctx = stream.context().clone();
    // `new_event(None)` は DISABLE_TIMING になるため DEFAULT を明示する。event は区間外で事前生成する。
    let mk = || {
        ctx.new_event(Some(CUevent_flags::CU_EVENT_DEFAULT))
            .expect("event")
    };
    let (e0, e1, e2, e3) = (mk(), mk(), mk(), mk());
    let ms_us = |a: &Event, b: &Event| f64::from(a.elapsed_ms(b).expect("elapsed")) * 1e3;

    let ops = CudaBackendOps::new(0);
    let params = upload_params(&ops);
    let input = tensor(0x9000, &[BATCH, D_IN]);
    let mem = ops.memory_ops().expect("CUDA MemoryOps");
    let a_pre = mem.upload(&input).expect("事前 upload");
    // 1x1 層（固定費用）。
    let t11 = tensor(0x4150, &[1, 1]);
    let a11 = mem.upload(&t11).expect("1x1 a");
    let w11 = mem.upload(&t11).expect("1x1 w");

    // 参照出力と warmup。以降の全反復が bit 一致することを hard assert する。
    let reference = chain(&ops, &stream, &params, &input, None).out;
    let ref_bits = bits(&reference);
    for _ in 0..WARMUP {
        let c = chain(&ops, &stream, &params, &input, None);
        assert_eq!(bits(&c.out), ref_bits, "warmup で出力が bit 不一致");
        let h = l1(&ops, &params, &a_pre);
        let _ = l2(&ops, &params, &h);
        let _ = tiny(&ops, &a11, &w11);
    }
    stream.synchronize().expect("sync");

    for i in 0..ITERS {
        for arm in rotated_order(ARMS, i) {
            stream.synchronize().expect("drain");
            match arm {
                0 => {
                    let c = chain(&ops, &stream, &params, &input, None);
                    assert_eq!(bits(&c.out), ref_bits, "P 腕で出力が bit 不一致");
                    let kw = (c.download - c.download_again).max(0.0);
                    let total = c.h2d_upload + c.l1_issue + c.l2_issue + c.download;
                    println!(
                        "{}",
                        format_line(
                            'P',
                            i,
                            &[
                                ("h2d_upload_us", us(c.h2d_upload)),
                                ("l1_issue_us", us(c.l1_issue)),
                                ("l2_issue_us", us(c.l2_issue)),
                                ("download_us", us(c.download)),
                                ("download_again_us", us(c.download_again)),
                                ("kernel_wait_derived_us", us(kw)),
                                ("total_us", us(total)),
                            ]
                        )
                    );
                }
                1 => {
                    let pr = Probe {
                        e0: &e0,
                        after_upload: &e1,
                        after_l1: &e2,
                        after_l2: &e3,
                    };
                    let c = chain(&ops, &stream, &params, &input, Some(&pr));
                    assert_eq!(bits(&c.out), ref_bits, "E 腕で出力が bit 不一致");
                    println!(
                        "{}",
                        format_line(
                            'E',
                            i,
                            &[
                                ("up_gpu_us", ms_us(&e0, &e1)),
                                ("l1_span_us", ms_us(&e1, &e2)),
                                ("l2_span_us", ms_us(&e2, &e3)),
                                ("chain_gpu_us", ms_us(&e1, &e3)),
                                ("download_us", us(c.download)),
                                ("l1_issue_us", us(c.l1_issue)),
                                ("l2_issue_us", us(c.l2_issue)),
                            ]
                        )
                    );
                }
                _ => {
                    // I: 層ごとに event 対で実物の linear_forward_device を挟む。
                    e0.record(&stream).expect("rec");
                    let (h, l1_host) = timed(|| l1(&ops, &params, &a_pre));
                    e1.record(&stream).expect("rec");
                    stream.synchronize().expect("sync");
                    let l1_iso = ms_us(&e0, &e1);
                    e0.record(&stream).expect("rec");
                    let (y, l2_host) = timed(|| l2(&ops, &params, &h));
                    e1.record(&stream).expect("rec");
                    stream.synchronize().expect("sync");
                    let l2_iso = ms_us(&e0, &e1);
                    let out = mem.download(&y).expect("D2H");
                    assert_eq!(bits(&out), ref_bits, "I 腕で出力が bit 不一致");
                    drop((h, y));
                    println!(
                        "{}",
                        format_line(
                            'I',
                            i,
                            &[
                                ("l1_iso_us", l1_iso),
                                ("l2_iso_us", l2_iso),
                                ("l1_host_us", us(l1_host)),
                                ("l2_host_us", us(l2_host)),
                            ]
                        )
                    );
                }
            }
        }
        // F: 固定費（#4149 の cuda_fixed と定義を揃える）。
        stream.synchronize().expect("drain");
        let (_, sync_idle) = timed(|| stream.synchronize().expect("sync"));
        let (y, launch_floor) = timed(|| tiny(&ops, &a11, &w11));
        stream.synchronize().expect("sync");
        drop(y);
        let y2 = tiny(&ops, &a11, &w11);
        let (_t, tiny_rt) = timed(|| mem.download(&y2).expect("D2H"));
        drop(y2);
        stream.synchronize().expect("drain");
        let (_, pair_host) = timed(|| {
            e0.record(&stream).expect("rec");
            e1.record(&stream).expect("rec");
        });
        stream.synchronize().expect("sync");
        let pair_gpu = ms_us(&e0, &e1);
        println!(
            "{}",
            format_line(
                'F',
                i,
                &[
                    ("sync_idle_us", us(sync_idle)),
                    ("launch_floor_1x1_us", us(launch_floor)),
                    ("tiny_roundtrip_us", us(tiny_rt)),
                    ("event_record_pair_us", us(pair_host)),
                    ("event_pair_gpu_us", pair_gpu),
                ]
            )
        );
    }
    println!("CHECKSUM4150 value={}", bits_hash(&ref_bits));
    println!("done4150");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn median_odd_and_even() {
        assert_eq!(median(&[3.0, 1.0, 2.0]), 2.0);
        assert_eq!(median(&[4.0, 1.0, 3.0, 2.0]), 2.5);
    }

    #[test]
    fn rotation_covers_every_arm() {
        for i in 0..6 {
            let mut o = rotated_order(ARMS, i);
            o.sort_unstable();
            assert_eq!(o, vec![0, 1, 2]);
        }
        assert_eq!(rotated_order(3, 1), vec![1, 2, 0]);
    }

    #[test]
    fn args_are_validated() {
        let s = |v: &[&str]| v.iter().map(|x| x.to_string()).collect::<Vec<_>>();
        assert_eq!(parse_args(&s(&["--round", "5"])), Ok(5));
        assert!(parse_args(&s(&[])).is_err());
        assert!(parse_args(&s(&["--round", "0"])).is_err());
        assert!(parse_args(&s(&["--round", "1000"])).is_err());
        assert!(parse_args(&s(&["--round", "x"])).is_err());
        assert!(parse_args(&s(&["--other", "1"])).is_err());
    }

    #[test]
    fn line_format_is_stable() {
        assert_eq!(
            format_line('E', 3, &[("a_us", 1.5), ("b_us", 2.0)]),
            "CHAIN4150 arm=E iter=3 a_us=1.500 b_us=2.000"
        );
    }

    #[test]
    fn hash_is_deterministic() {
        assert_eq!(bits_hash(&[1, 2]), bits_hash(&[1, 2]));
        assert_ne!(bits_hash(&[1, 2]), bits_hash(&[2, 1]));
    }
}
