//! CUDA 小形状 GEMM（N=256 が判定対象）の起動固定費を多層分解する診断
//! テスト（イシュー #2109。親 #2099 Phase 3 負けセル対処の sub）。
//!
//! # 背景
//!
//! スコアボード 2026-09-19 版（`docs/perf/logs/framework-compare-
//! precision-class-remeasure-1988/scoreboard/gen_1988.out`）で GB10 の
//! CUDA gemm N=256（セル G-CUDA-G256）は 2 位・対 candle 0.83× だった
//! （fandhe-ai reuse 92.2 µs／fresh 89.3 µs、candle fresh 76.4 µs）。
//! なお Issue 本文の「0.72×」は G-CPU-G256 の値で転記違いの可能性が
//! ある（`docs/perf/loss-attribution-matrix.md` §8）ため、本診断は
//! G-CUDA-G256 = 0.83× を対象とする。小形状ではカーネル自体の時間より
//! launch 回数・同期・確保・解放といった固定費が支配的になりうるが、
//! その内訳は未計測だった（`docs/analysis/candle-0.11.0-cuda-path.md`
//! §9-3 の「dispatch 多段」は未検証の仮説）。
//!
//! # 配置理由（`gemm_reuse_phase_diag_tests.rs` と同型の判断）
//!
//! `context_cache::{cached_device, cached_gemm, cached_allocator}`・
//! `CudaGemm::launch_tiled_f32_pooled`・診断カウンタ（いずれも
//! `pub(crate)`）へ到達するため、`lib.rs` の `#[cfg(test)]` 兄弟
//! モジュールとして置く。既存の `gemm_reuse_phase_diag_tests` は
//! N=1024〜4096 専用で teardown（drop）区間と device 側時間を測って
//! いない。本ファイルは N ∈ {128, 256, 512} でそれらを補う。
//!
//! # 区間の定義（1 反復あたり。出力は `DIAG2109` 接頭辞の 1 行 1 区間）
//!
//! - L0 `ops_total`: `BackendOps::gemm_fp32_strict` の wall（本番の入口）
//! - L1 `gemm_total`: `CudaGemm::run_tiled_f32` の wall
//! - L2: `run_f32_kernel` と同じ順の手動分解（`h2d_a`／`h2d_b`／`alloc_c`
//!   ／`launch_issue`／`readback`〈カーネル完了待ち + D2H。本番同様
//!   `memory::readback` 内の同期 1 回のみ〉／`teardown`／`driver_scope`／
//!   `l2_sum`）。追加同期を含まない本番相当経路であり、`l2_sum` はこの経路
//!   の合計（PR #2454 指摘: 追加の `stream.synchronize()` を除く）
//! - L2S: kernel_wait と d2h の分離用の**非本番**補助系列（`kernel_wait`
//!   ／`d2h`）。起動後に追加同期を 1 回挟むため本番の同期回数と異なり、
//!   `l2_sum` には含めない。H2・H3 の内訳算出専用
//! - D: device 側 event 計時（別パス。L2 の値を乱さない）。event 間隔は
//!   device idle（host の投入待ち）を含む点に注意。`dev_kernel_b2b` は
//!   同一カーネルを連続キューイングした時間差から求める推定値
//! - E: floor（`sync_idle`／`tiny_roundtrip`／`event_create_drop`／
//!   `h2d_prealloc`／`h2d_clone`）
//!
//! # 断言の方針
//!
//! 件数（launch 回数等）は決定的なので厳密に断言し、時間は断言しない
//! （環境揺らぎによる flaky 化防止）。一方、所要時間を比較する全腕の**出力**
//! は計測窓の外で `run_tiled_f32` の参照出力と bit 一致を確認し、不一致は
//! panic とする（PR #2454 指摘。[`verify_bit_identical`]・[`OutputVerifier`]。
//! 実施結果は `DIAG2109 n=.. verify arm=.. checks=.. mode=bit_exact` 行）。readback の宛先確保
//! 方式の A/B は #2107 の担当であり、本ファイルは本番 `memory::readback`
//! を 1 区間として測るだけである。
//!
//! # 実行方法
//!
//! `cargo test --release -p fandhe-ai-backend-cuda --lib
//! gemm_small_launch_cost_diag -- --ignored --exact --test-threads=1
//! --nocapture`（同一 GPU 上のスレッド競合を避けるため
//! `--test-threads=1` 必須）。実機依存テストはすべて `#[ignore]`。

use std::time::Instant;

use bench_harness::{Quartiles, median_q1_q3, rng::Xorshift64Star};
use cudarc::driver::sys::CUevent_flags;
use fandhe_ai_tensor_core::{BackendOps, Tensor};

use crate::context_cache::{cached_allocator, cached_device, cached_gemm};
use crate::gemm::{
    DiagTiledF32Kernel, GemmLaunchDiagCounters, gemm_launch_diag_reset, gemm_launch_diag_snapshot,
};
use crate::ops::CudaBackendOps;

/// 測定対象サイズ。判定対象は 256（RULE.txt と同値）。
const SIZES: [usize; 3] = [128, 256, 512];
const WARMUP: usize = 50;
const MEASURED: usize = 200;

fn gen_square_ab(seed: u64, n: usize) -> (Vec<f32>, Vec<f32>) {
    let mut rng = Xorshift64Star::new(seed);
    let a = rng.fill_vec(n * n);
    let b = rng.fill_vec(n * n);
    (a, b)
}

fn quartiles_us(samples_secs: &[f64]) -> Quartiles {
    let us: Vec<f64> = samples_secs.iter().map(|s| s * 1e6).collect();
    median_q1_q3(&us).expect("samples must be non-empty and NaN-free")
}

/// `DIAG2109 n=.. layer=.. phase=.. median_us=.. q1_us=.. q3_us=..` を返す。
fn format_phase_line(n: usize, layer: &str, phase: &str, q: Quartiles) -> String {
    format!(
        "DIAG2109 n={n} layer={layer} phase={phase} median_us={:.3} q1_us={:.3} q3_us={:.3}",
        q.median, q.q1, q.q3
    )
}

fn emit(n: usize, layer: &str, phase: &str, samples_secs: &[f64]) {
    println!(
        "{}",
        format_phase_line(n, layer, phase, quartiles_us(samples_secs))
    );
}

fn checksum_bits(c: &[f32]) -> u64 {
    let s: f64 = c.iter().map(|v| *v as f64).sum();
    s.to_bits()
}

/// 出力一致検証の不一致種別（[`verify_bit_identical`] の戻り値）。
#[derive(Debug, PartialEq)]
enum OutputMismatch {
    /// 要素数が異なる。
    Len { expected: usize, got: usize },
    /// `got` に非有限値（NaN／inf）がある（参照側の非有限も同じ扱い）。
    NonFinite { index: usize, bits: u32 },
    /// 有限だが bit 表現が異なる（`-0.0` と `0.0` も不一致）。
    Bits {
        index: usize,
        expected: u32,
        got: u32,
    },
}

/// 所要時間を比較する各腕の出力が参照と **bit 一致**することを判定する純関数
/// （PR #2454 指摘: 比較対象の出力検証。GPU なしで単体テストできるよう
/// 判定だけを切り出している）。
///
/// # 判定方式（bit 一致を採る根拠）
///
/// 比較する全腕（L0 `gemm_fp32_strict`・L1 `run_tiled_f32`・L2／L2S／D の
/// `launch_tiled_f32_pooled(Select)`）は同一入力に対し同一の
/// `select_tiled_f32_kernel(0, m, n, k)` を選ぶ単一の連続 K ループ（f32 FMA。
/// split-K・atomics なし）で決定的なため、結合順序の差による誤差が生じず
/// bit 一致が成立するはずである。ゆえに統一複合判定（相対 1e-3 未満または
/// 絶対 1e-5 未満。`fandhe_ai_backend_cpu::assert_parity`）ではなく、より厳しい
/// bit 一致（`f32::to_bits`）を採る。許容誤差の定数は新設・変更しない。
/// 非有限値は bit が同じでも不一致とする（NaN 同士が通ることを防ぐ）。
fn verify_bit_identical(expected: &[f32], got: &[f32]) -> Result<(), OutputMismatch> {
    if expected.len() != got.len() {
        return Err(OutputMismatch::Len {
            expected: expected.len(),
            got: got.len(),
        });
    }
    for (index, (e, g)) in expected.iter().zip(got).enumerate() {
        if !e.is_finite() {
            return Err(OutputMismatch::NonFinite {
                index,
                bits: e.to_bits(),
            });
        }
        if !g.is_finite() {
            return Err(OutputMismatch::NonFinite {
                index,
                bits: g.to_bits(),
            });
        }
        if e.to_bits() != g.to_bits() {
            return Err(OutputMismatch::Bits {
                index,
                expected: e.to_bits(),
                got: g.to_bits(),
            });
        }
    }
    Ok(())
}

/// 腕ごとの出力一致検証の実施回数を数え、不一致は panic（fail-closed）にする。
///
/// 検証は必ず計測区間の外（`Instant` 停止後・計測窓の外）で呼ぶ。実施結果は
/// `DIAG2109 n=.. verify arm=.. checks=..` 行として出力し、`aggregate.py` が
/// 全腕・checks>0 を要求する（RULE.txt 1・2。検証が走らなかった腕を採用しない）。
struct OutputVerifier {
    n: usize,
    /// 全腕が突き合わせる参照（本番 `run_tiled_f32` の出力）。
    reference: Vec<f32>,
    checks: std::collections::BTreeMap<&'static str, u64>,
}

impl OutputVerifier {
    /// 計測前に 1 回だけ `run_tiled_f32` を実行して参照を作る（有限性は
    /// 以降の比較で全要素検査される。ここでは自己比較で先に検査する）。
    fn new(n: usize, reference: Vec<f32>) -> Self {
        assert_eq!(
            verify_bit_identical(&reference, &reference),
            Ok(()),
            "n={n}: 参照出力が有限でない"
        );
        assert_eq!(reference.len(), n * n, "n={n}: 参照出力の要素数");
        assert_ne!(
            checksum_bits(&reference),
            0.0f64.to_bits(),
            "n={n}: 参照出力の checksum が 0"
        );
        Self {
            n,
            reference,
            checks: Default::default(),
        }
    }

    /// 出力 `got` が参照と bit 一致することを確認する。
    fn check(&mut self, arm: &'static str, got: &[f32]) {
        Self::record(self.n, &mut self.checks, arm, &self.reference, got);
    }

    /// 出力 `got` が任意の期待値 `expected` と bit 一致することを確認する
    /// （tiny 起動・H2D の往復など、参照と形状が異なる腕用）。
    fn check_against(&mut self, arm: &'static str, expected: &[f32], got: &[f32]) {
        Self::record(self.n, &mut self.checks, arm, expected, got);
    }

    fn record(
        n: usize,
        checks: &mut std::collections::BTreeMap<&'static str, u64>,
        arm: &'static str,
        expected: &[f32],
        got: &[f32],
    ) {
        if let Err(e) = verify_bit_identical(expected, got) {
            panic!("n={n} arm={arm}: 出力が参照と一致しない: {e:?}");
        }
        *checks.entry(arm).or_insert(0) += 1;
    }

    fn emit(&self) {
        for (arm, c) in &self.checks {
            println!(
                "DIAG2109 n={} verify arm={arm} checks={c} mode=bit_exact",
                self.n
            );
        }
    }
}

/// 検証が必須の腕（`aggregate.py` の `VERIFY_ARMS` と一致させる）。
const VERIFY_ARMS: [&str; 10] = [
    "L0_ops",
    "L1_gemm",
    "L2",
    "L2S",
    "D_dev_trial",
    "D_b2b",
    "E_tiny",
    "E_h2d_prealloc",
    "E_h2d_clone",
    "count_run",
];

/// 生の `CudaSlice<f32>` を D2H して同期する（`memory::readback` と同じ
/// `clone_dtoh` → `synchronize` の順）。計測区間の外の検証専用。
fn dtoh_sync(
    stream: &std::sync::Arc<cudarc::driver::CudaStream>,
    s: &cudarc::driver::CudaSlice<f32>,
) -> Vec<f32> {
    let host = stream.clone_dtoh(s).expect("verify dtoh");
    stream.synchronize().expect("verify sync");
    host
}

/// L2 手動分解 1 反復の各区間（秒）。
#[derive(Default)]
struct L2Sample {
    h2d_a: f64,
    h2d_b: f64,
    alloc_c: f64,
    launch_issue: f64,
    /// 本番相当の `memory::readback`（カーネル完了待ち 1 回 + D2H）。
    /// `split_sync = true` の補助試行では 0（`kernel_wait`／`d2h` へ分離）。
    readback: f64,
    /// 補助試行（`split_sync = true`）のみ: 追加同期によるカーネル完了待ち。
    kernel_wait: f64,
    /// 補助試行のみ: 同期後の D2H（readback 内同期は完了済みで軽い）。
    d2h: f64,
    teardown: f64,
    /// 本番 `run_f32_kernel` が全区間を包む `with_driver_call`（capture 排他
    /// スコープ）の入退場費用。全操作を同一スコープ内で実行し、スコープ全体の
    /// 所要から各区間合計を引いた残差として求め l2_sum へ含める
    /// （L1 − l2_sum＝H4b の帰属を本番経路に合わせる。PR #2454 指摘）。
    driver_scope: f64,
}

impl L2Sample {
    fn sum(&self) -> f64 {
        self.h2d_a
            + self.h2d_b
            + self.alloc_c
            + self.launch_issue
            + self.readback
            + self.teardown
            + self.driver_scope
    }
}

/// `split_sync = false` は本番 `run_f32_kernel` と同じ同期回数（readback 内
/// の 1 回のみ）で測る。`true` は kernel_wait と d2h を分離するための補助
/// 試行で追加同期を挟む（`sum()` の対象外・非本番）。
fn l2_trial(
    split_sync: bool,
    device: &crate::device::CudaDevice,
    gemm: &crate::gemm::CudaGemm,
    allocator: &crate::pool::CudaAllocator,
    a: &[f32],
    b: &[f32],
    n: u32,
) -> (L2Sample, Vec<f32>) {
    let stream = device.stream().clone();
    let mut s = L2Sample::default();

    // 本番 `run_f32_kernel` と同じく H2D・確保・起動・readback・解放の全操作を
    // `with_driver_call`（capture 排他スコープ）の内側で実行する。各区間は
    // クロージャ内で個別計測し、スコープ全体の所要から区間合計を引いた残差を
    // 入退場費用（`driver_scope`）とする（PR #2454 指摘）。
    let scope_t = Instant::now();
    let out = crate::context_cache::with_driver_call(device.ordinal(), || {
        let t = Instant::now();
        let a_dev = gemm.upload_h2d_new(a)?;
        s.h2d_a = t.elapsed().as_secs_f64();

        let t = Instant::now();
        let b_dev = gemm.upload_h2d_new(b)?;
        s.h2d_b = t.elapsed().as_secs_f64();

        let t = Instant::now();
        let mut c_dev = allocator.alloc_uninit_f32((n as usize) * (n as usize))?;
        s.alloc_c = t.elapsed().as_secs_f64();

        let t = Instant::now();
        gemm.launch_tiled_f32_pooled(
            &a_dev,
            &b_dev,
            &mut c_dev,
            n,
            n,
            n,
            DiagTiledF32Kernel::Select,
        )?;
        s.launch_issue = t.elapsed().as_secs_f64();

        if split_sync {
            let t = Instant::now();
            stream.synchronize().expect("kernel wait");
            s.kernel_wait = t.elapsed().as_secs_f64();
        }

        let t = Instant::now();
        let out = crate::memory::readback(&stream, &c_dev.as_view())?;
        let rb = t.elapsed().as_secs_f64();
        if split_sync {
            s.d2h = rb;
        } else {
            s.readback = rb;
        }

        let t = Instant::now();
        drop(c_dev);
        drop(a_dev);
        drop(b_dev);
        s.teardown = t.elapsed().as_secs_f64();
        Ok(out)
    })
    .expect("l2 scoped trial");
    let scope_total = scope_t.elapsed().as_secs_f64();
    s.driver_scope = (scope_total
        - (s.h2d_a
            + s.h2d_b
            + s.alloc_c
            + s.launch_issue
            + s.kernel_wait
            + s.readback
            + s.d2h
            + s.teardown))
        .max(0.0);

    (s, out)
}

/// device 側 event 計時 1 反復（ms）。区間は device idle を含む。
struct DevSample {
    h2d_a_ms: f64,
    h2d_b_ms: f64,
    kernel_seg_ms: f64,
    d2h_ms: f64,
    span_ms: f64,
}

fn dev_trial(
    device: &crate::device::CudaDevice,
    gemm: &crate::gemm::CudaGemm,
    allocator: &crate::pool::CudaAllocator,
    a: &[f32],
    b: &[f32],
    n: u32,
) -> (DevSample, Vec<f32>) {
    let stream = device.stream().clone();
    let ctx = stream.context().clone();
    // `new_event(None)` は DISABLE_TIMING になるため DEFAULT を明示する。
    // event は計測区間の外で事前生成し、生成コストを区間へ混ぜない。
    let mk = || {
        ctx.new_event(Some(CUevent_flags::CU_EVENT_DEFAULT))
            .expect("event")
    };
    let (e0, e1, e2, e3, e4) = (mk(), mk(), mk(), mk(), mk());
    stream.synchronize().expect("drain");

    e0.record(&stream).expect("rec");
    let a_dev = gemm.upload_h2d_new(a).expect("h2d a");
    e1.record(&stream).expect("rec");
    let b_dev = gemm.upload_h2d_new(b).expect("h2d b");
    e2.record(&stream).expect("rec");
    let mut c_dev = allocator
        .alloc_uninit_f32((n as usize) * (n as usize))
        .expect("alloc c");
    gemm.launch_tiled_f32_pooled(
        &a_dev,
        &b_dev,
        &mut c_dev,
        n,
        n,
        n,
        DiagTiledF32Kernel::Select,
    )
    .expect("launch");
    e3.record(&stream).expect("rec");
    let out = crate::memory::readback(&stream, &c_dev.as_view()).expect("readback");
    e4.record(&stream).expect("rec");
    stream.synchronize().expect("sync");

    let ms = |x: &cudarc::driver::CudaEvent, y: &cudarc::driver::CudaEvent| -> f64 {
        x.elapsed_ms(y).expect("elapsed") as f64
    };
    let sample = DevSample {
        h2d_a_ms: ms(&e0, &e1),
        h2d_b_ms: ms(&e1, &e2),
        kernel_seg_ms: ms(&e2, &e3),
        d2h_ms: ms(&e3, &e4),
        span_ms: ms(&e0, &e4),
    };
    // 出力は event 区間の外で呼び出し側が参照と突き合わせる。
    (sample, out)
}

/// 同一カーネルを `launches` 回連続でキューイングした device 側時間（ms）。
/// 1 回版との差から back-to-back の純カーネル時間を推定する（launch
/// latency の影響を差し引くための推定であり厳密値ではない）。
fn kernel_queue_ms(
    device: &crate::device::CudaDevice,
    gemm: &crate::gemm::CudaGemm,
    a_dev: &cudarc::driver::CudaSlice<f32>,
    b_dev: &cudarc::driver::CudaSlice<f32>,
    c_dev: &mut crate::pool::PooledCudaHandle<f32>,
    n: u32,
    launches: usize,
) -> f64 {
    let stream = device.stream().clone();
    let ctx = stream.context().clone();
    let s = ctx
        .new_event(Some(CUevent_flags::CU_EVENT_DEFAULT))
        .expect("event");
    let e = ctx
        .new_event(Some(CUevent_flags::CU_EVENT_DEFAULT))
        .expect("event");
    stream.synchronize().expect("drain");
    s.record(&stream).expect("rec");
    for _ in 0..launches {
        gemm.launch_tiled_f32_pooled(a_dev, b_dev, c_dev, n, n, n, DiagTiledF32Kernel::Select)
            .expect("launch");
    }
    e.record(&stream).expect("rec");
    // cuEventElapsedTime は両イベント完了が前提（未完了だと NOT_READY）。
    // 計測値には含まれない同期のため、経過時間の取得前に stream を同期する。
    stream.synchronize().expect("sync before elapsed");
    s.elapsed_ms(&e).expect("elapsed") as f64
}

fn run_size(n: usize) {
    let device = cached_device(0).expect("CUDA device");
    let gemm = cached_gemm(&device).expect("CudaGemm");
    let allocator = cached_allocator(&device).expect("CudaAllocator");
    let ops = CudaBackendOps::new(0);
    let (a, b) = gen_square_ab(0x2109_0000 ^ (n as u64), n);
    let nn = n as u32;

    #[cfg(feature = "internal-diagnostics")]
    println!(
        "DIAG2109 n={n} kernel={:?}",
        gemm.tiled_f32_kernel_for(nn, nn)
    );

    // --- L0 / L1（本番入口・CudaGemm 入口）---
    let ta = Tensor::new(a.clone(), &[n, n]).expect("tensor a");
    let tb = Tensor::new(b.clone(), &[n, n]).expect("tensor b");
    let mut l0 = Vec::with_capacity(MEASURED);
    let mut l1 = Vec::with_capacity(MEASURED);
    // 全腕が突き合わせる参照を計測前に 1 回だけ作る（PR #2454 指摘）。
    let mut verifier = OutputVerifier::new(
        n,
        gemm.run_tiled_f32(&a, &b, nn, nn, nn)
            .expect("reference run_tiled_f32"),
    );
    let checksum = checksum_bits(&verifier.reference);
    for i in 0..(WARMUP + MEASURED) {
        let t = Instant::now();
        let out = ops.gemm_fp32_strict(&ta, &tb).expect("ops gemm");
        let d0 = t.elapsed().as_secs_f64();
        let t = Instant::now();
        let c = gemm
            .run_tiled_f32(&a, &b, nn, nn, nn)
            .expect("run_tiled_f32");
        let d1 = t.elapsed().as_secs_f64();
        if i >= WARMUP {
            l0.push(d0);
            l1.push(d1);
        }
        // 検証は d0・d1 の計測後（計測窓の外）。不一致は panic（fail-closed）。
        verifier.check("L1_gemm", &c);
        verifier.check("L0_ops", out.as_slice().expect("L0 出力は contiguous"));
        drop(out);
    }
    emit(n, "L0", "ops_total", &l0);
    emit(n, "L1", "gemm_total", &l1);
    println!("DIAG2109 n={n} checksum_bits=0x{checksum:016x}");

    // --- L2 ---
    let mut cols: [Vec<f64>; 8] = Default::default();
    for i in 0..(WARMUP + MEASURED) {
        let (s, out) = l2_trial(false, &device, &gemm, &allocator, &a, &b, nn);
        verifier.check("L2", &out);
        if i >= WARMUP {
            let v = [
                s.h2d_a,
                s.h2d_b,
                s.alloc_c,
                s.launch_issue,
                s.readback,
                s.teardown,
                s.driver_scope,
                s.sum(),
            ];
            for (c, x) in cols.iter_mut().zip(v) {
                c.push(x);
            }
        }
    }
    for (name, c) in [
        "h2d_a",
        "h2d_b",
        "alloc_c",
        "launch_issue",
        "readback",
        "teardown",
        "driver_scope",
        "l2_sum",
    ]
    .iter()
    .zip(cols.iter())
    {
        emit(n, "L2", name, c);
    }

    // --- L2S（kernel_wait／d2h 分離。追加同期を含む非本番の補助系列）---
    let mut split: [Vec<f64>; 2] = Default::default();
    for i in 0..(WARMUP + MEASURED) {
        let (s, out) = l2_trial(true, &device, &gemm, &allocator, &a, &b, nn);
        verifier.check("L2S", &out);
        if i >= WARMUP {
            split[0].push(s.kernel_wait);
            split[1].push(s.d2h);
        }
    }
    emit(n, "L2S", "kernel_wait", &split[0]);
    emit(n, "L2S", "d2h", &split[1]);

    // --- D（device 側 event）---
    let mut d: [Vec<f64>; 5] = Default::default();
    for i in 0..(WARMUP + MEASURED) {
        let (s, out) = dev_trial(&device, &gemm, &allocator, &a, &b, nn);
        verifier.check("D_dev_trial", &out);
        if i >= WARMUP {
            let v = [s.h2d_a_ms, s.h2d_b_ms, s.kernel_seg_ms, s.d2h_ms, s.span_ms];
            for (c, x) in d.iter_mut().zip(v) {
                c.push(x * 1e-3);
            }
        }
    }
    for (name, c) in [
        "dev_h2d_a",
        "dev_h2d_b",
        "dev_kernel_seg",
        "dev_d2h",
        "dev_span",
    ]
    .iter()
    .zip(d.iter())
    {
        emit(n, "D", name, c);
    }
    // back-to-back 純カーネル時間の推定（(t9 - t1) / 8）。
    {
        let a_dev = gemm.upload_h2d_new(&a).expect("h2d a");
        let b_dev = gemm.upload_h2d_new(&b).expect("h2d b");
        let mut c_dev = allocator.alloc_uninit_f32(n * n).expect("alloc c");
        let mut est = Vec::with_capacity(MEASURED);
        for i in 0..(WARMUP + MEASURED) {
            let t1 = kernel_queue_ms(&device, &gemm, &a_dev, &b_dev, &mut c_dev, nn, 1);
            let t9 = kernel_queue_ms(&device, &gemm, &a_dev, &b_dev, &mut c_dev, nn, 9);
            if i >= WARMUP {
                est.push(((t9 - t1) / 8.0).max(0.0) * 1e-3);
            }
        }
        emit(n, "D", "dev_kernel_b2b", &est);
        // b2b は出力を読まずに連続起動するため、全反復後の c_dev（最後の 9 回連続
        // 起動の結果）を計測窓の外で 1 回読み戻して参照と突き合わせる。
        let out = crate::memory::readback(device.stream(), &c_dev.as_view()).expect("b2b readback");
        verifier.check("D_b2b", &out);
    }

    // --- E（floor）---
    let stream = device.stream().clone();
    let ctx = stream.context().clone();
    let mut sync_idle = Vec::with_capacity(MEASURED);
    let mut tiny = Vec::with_capacity(MEASURED);
    let mut ev = Vec::with_capacity(MEASURED);
    let mut h2d_pre = Vec::with_capacity(MEASURED);
    let mut h2d_clone = Vec::with_capacity(MEASURED);
    let one = [1.0f32];
    let ta1 = stream.clone_htod(&one).expect("tiny a");
    let tb1 = stream.clone_htod(&one).expect("tiny b");
    let mut tc1 = allocator.alloc_uninit_f32(1).expect("tiny c");
    let mut pre = stream.alloc_zeros::<f32>(n * n).expect("prealloc");
    // h2d 系の腕は計測窓の中で結果を読まないため、計測前に同じ API 呼び出しの
    // 往復（H2D → D2H）が入力と bit 一致することを 1 回確認する。
    stream.memcpy_htod(&a, &mut pre).expect("h2d prealloc");
    verifier.check_against("E_h2d_prealloc", &a, &dtoh_sync(&stream, &pre));
    let cl0 = stream.clone_htod(&a).expect("h2d clone");
    verifier.check_against("E_h2d_clone", &a, &dtoh_sync(&stream, &cl0));
    drop(cl0);
    for i in 0..(WARMUP + MEASURED) {
        stream.synchronize().expect("drain");
        let t = Instant::now();
        stream.synchronize().expect("sync idle");
        let x0 = t.elapsed().as_secs_f64();

        let t = Instant::now();
        gemm.launch_tiled_f32_pooled(&ta1, &tb1, &mut tc1, 1, 1, 1, DiagTiledF32Kernel::Select)
            .expect("tiny launch");
        stream.synchronize().expect("tiny sync");
        let x1 = t.elapsed().as_secs_f64();

        let t = Instant::now();
        let e = ctx.new_event(None).expect("event");
        drop(e);
        let x2 = t.elapsed().as_secs_f64();

        let t = Instant::now();
        stream.memcpy_htod(&a, &mut pre).expect("h2d prealloc");
        let x3 = t.elapsed().as_secs_f64();

        let t = Instant::now();
        let cl = stream.clone_htod(&a).expect("h2d clone");
        drop(cl);
        let x4 = t.elapsed().as_secs_f64();

        if i >= WARMUP {
            sync_idle.push(x0);
            tiny.push(x1);
            ev.push(x2);
            h2d_pre.push(x3);
            h2d_clone.push(x4);
        }
    }
    emit(n, "E", "sync_idle", &sync_idle);
    emit(n, "E", "tiny_roundtrip", &tiny);
    emit(n, "E", "event_create_drop", &ev);
    emit(n, "E", "h2d_prealloc", &h2d_pre);
    emit(n, "E", "h2d_clone_drop", &h2d_clone);
    // tiny 起動（1x1x1 の 1.0 * 1.0）の出力は 1.0 でなければならない。全反復後に
    // 計測窓の外で読み戻す。
    let tiny_out = crate::memory::readback(&stream, &tc1.as_view()).expect("tiny readback");
    verifier.check_against("E_tiny", &[1.0f32], &tiny_out);

    // --- 件数 ---
    gemm_launch_diag_reset();
    let count_out = gemm.run_tiled_f32(&a, &b, nn, nn, nn).expect("count run");
    let c = gemm_launch_diag_snapshot();
    // 件数の snapshot 取得後（件数窓の外）で出力を参照と突き合わせる。
    verifier.check("count_run", &count_out);
    println!(
        "DIAG2109 n={n} counts driver_call_scopes={} h2d_calls={} h2d_bytes={} pool_allocs={} \
         kernel_launches={} d2h_calls={} d2h_bytes={} stream_syncs={}",
        c.driver_call_scopes,
        c.h2d_calls,
        c.h2d_bytes,
        c.pool_allocs,
        c.kernel_launches,
        c.d2h_calls,
        c.d2h_bytes,
        c.stream_syncs
    );

    // 検証が必須の全腕で 1 回以上実施されたことを確認し（実施漏れは fail-closed）、
    // 実施結果を出力する（aggregate.py が全腕・checks>0 を要求する）。
    assert_eq!(
        verifier.checks.keys().copied().collect::<Vec<_>>(),
        {
            let mut v = VERIFY_ARMS.to_vec();
            v.sort_unstable();
            v
        },
        "n={n}: 出力検証の腕の集合が期待と一致しない"
    );
    verifier.emit();

    let _ = allocator.release_cached();
}

/// 実機診断: 全区間を計測して `DIAG2109` 行を出力する。時間の大小関係は
/// 断言しない。比較する全腕（L0／L1／L2／L2S／D／b2b／tiny／H2D／件数走行）の
/// 出力は計測窓の外で参照と bit 一致を検証し、不一致は panic（fail-closed）。
#[test]
#[ignore]
fn gemm_small_launch_cost_diag() {
    for n in SIZES {
        run_size(n);
    }
}

fn expected_counts(n: usize) -> GemmLaunchDiagCounters {
    let bytes = (n * n * 4) as u64;
    GemmLaunchDiagCounters {
        driver_call_scopes: 1,
        h2d_calls: 2,
        h2d_bytes: 2 * bytes,
        pool_allocs: 1,
        kernel_launches: 1,
        d2h_calls: 1,
        d2h_bytes: bytes,
        stream_syncs: 1,
    }
}

/// 実機: 1 回の GEMM が踏む driver 境界の件数を厳密に断言する
/// （`run_tiled_f32` と本番入口 `gemm_fp32_strict` の両方）。件数は
/// 決定的なので flaky にならない。
#[test]
#[ignore]
fn gemm_small_launch_counts_exact() {
    let n = 256usize;
    let nn = n as u32;
    let device = cached_device(0).expect("CUDA device");
    let gemm = cached_gemm(&device).expect("CudaGemm");
    let (a, b) = gen_square_ab(0x2109_0001, n);

    // ウォームアップ（初回のモジュール確保等が件数へ混ざらないようにする）。
    let reference = gemm.run_tiled_f32(&a, &b, nn, nn, nn).expect("warmup");
    assert_eq!(verify_bit_identical(&reference, &reference), Ok(()));

    let before = gemm_launch_diag_snapshot();
    let out = gemm
        .run_tiled_f32(&a, &b, nn, nn, nn)
        .expect("run_tiled_f32");
    let d = gemm_launch_diag_snapshot().since(before);
    assert_eq!(d, expected_counts(n), "run_tiled_f32 の件数");
    assert_eq!(verify_bit_identical(&reference, &out), Ok(()));

    let ops = CudaBackendOps::new(0);
    let ta = Tensor::new(a, &[n, n]).expect("tensor a");
    let tb = Tensor::new(b, &[n, n]).expect("tensor b");
    let _ = ops.gemm_fp32_strict(&ta, &tb).expect("warmup ops");
    let before = gemm_launch_diag_snapshot();
    let ops_out = ops.gemm_fp32_strict(&ta, &tb).expect("gemm_fp32_strict");
    let d = gemm_launch_diag_snapshot().since(before);
    assert_eq!(d, expected_counts(n), "gemm_fp32_strict の件数");
    assert_eq!(
        verify_bit_identical(&reference, ops_out.as_slice().expect("contiguous")),
        Ok(())
    );
}

// ---- GPU 不要の単体テスト（非 ignore）----

#[test]
fn diag_counters_reset_snapshot_and_since() {
    gemm_launch_diag_reset();
    assert_eq!(
        gemm_launch_diag_snapshot(),
        GemmLaunchDiagCounters::default()
    );
    let hi = GemmLaunchDiagCounters {
        kernel_launches: 3,
        h2d_bytes: 10,
        ..Default::default()
    };
    let lo = GemmLaunchDiagCounters {
        kernel_launches: 1,
        h2d_bytes: 20,
        ..Default::default()
    };
    let d = hi.since(lo);
    assert_eq!(d.kernel_launches, 2);
    // saturating: 逆転しても 0 で止まる。
    assert_eq!(d.h2d_bytes, 0);
}

#[test]
fn phase_line_format_is_machine_readable() {
    let q = median_q1_q3(&[1.0, 2.0, 3.0]).expect("quartiles");
    let line = format_phase_line(256, "L2", "h2d_a", q);
    assert_eq!(
        line,
        format!(
            "DIAG2109 n=256 layer=L2 phase=h2d_a median_us={:.3} q1_us={:.3} q3_us={:.3}",
            q.median, q.q1, q.q3
        )
    );
    assert!(line.starts_with("DIAG2109 "));
}

#[test]
fn quartiles_us_converts_seconds_to_microseconds() {
    let q = quartiles_us(&[1e-6, 2e-6, 3e-6]);
    assert!((q.median - 2.0).abs() < 1e-9);
}

#[test]
fn checksum_bits_is_deterministic_and_order_sensitive_to_values() {
    let a = checksum_bits(&[1.0, 2.0, 3.0]);
    assert_eq!(a, checksum_bits(&[1.0, 2.0, 3.0]));
    assert_ne!(a, checksum_bits(&[1.0, 2.0, 4.0]));
}

#[test]
fn verify_bit_identical_accepts_identical_outputs() {
    let x = [1.0f32, -2.5, 0.0, 3.25e-3];
    assert_eq!(verify_bit_identical(&x, &x), Ok(()));
    assert_eq!(verify_bit_identical(&[], &[]), Ok(()));
}

#[test]
fn verify_bit_identical_rejects_single_element_difference() {
    let x = [1.0f32, 2.0, 3.0, 4.0];
    let mut y = x;
    // 最下位 bit 1 つの差（許容誤差内でも bit 一致判定では不一致）。
    y[2] = f32::from_bits(y[2].to_bits() + 1);
    assert_eq!(
        verify_bit_identical(&x, &y),
        Err(OutputMismatch::Bits {
            index: 2,
            expected: 3.0f32.to_bits(),
            got: y[2].to_bits(),
        })
    );
    // 符号付きゼロも bit が異なるため不一致。
    assert!(matches!(
        verify_bit_identical(&[0.0], &[-0.0]),
        Err(OutputMismatch::Bits { index: 0, .. })
    ));
}

#[test]
fn verify_bit_identical_rejects_non_finite_even_if_bits_equal() {
    let nan = [1.0f32, f32::NAN];
    assert!(matches!(
        verify_bit_identical(&nan, &nan),
        Err(OutputMismatch::NonFinite { index: 1, .. })
    ));
    assert!(matches!(
        verify_bit_identical(&[1.0, 2.0], &[1.0, f32::NAN]),
        Err(OutputMismatch::NonFinite { index: 1, .. })
    ));
    assert!(matches!(
        verify_bit_identical(&[f32::INFINITY], &[f32::INFINITY]),
        Err(OutputMismatch::NonFinite { index: 0, .. })
    ));
}

#[test]
fn verify_bit_identical_rejects_length_mismatch() {
    assert_eq!(
        verify_bit_identical(&[1.0, 2.0], &[1.0]),
        Err(OutputMismatch::Len {
            expected: 2,
            got: 1
        })
    );
}

#[test]
#[should_panic(expected = "出力が参照と一致しない")]
fn output_verifier_panics_on_mismatch() {
    let mut v = OutputVerifier::new(2, vec![1.0, 2.0, 3.0, 4.0]);
    v.check("L0_ops", &[1.0, 2.0, 3.0, 5.0]);
}

#[test]
fn output_verifier_counts_checks_per_arm() {
    let mut v = OutputVerifier::new(2, vec![1.0, 2.0, 3.0, 4.0]);
    v.check("L0_ops", &[1.0, 2.0, 3.0, 4.0]);
    v.check("L0_ops", &[1.0, 2.0, 3.0, 4.0]);
    v.check_against("E_tiny", &[1.0], &[1.0]);
    assert_eq!(v.checks.get("L0_ops"), Some(&2));
    assert_eq!(v.checks.get("E_tiny"), Some(&1));
}

#[test]
#[should_panic(expected = "参照出力が有限でない")]
fn output_verifier_rejects_non_finite_reference() {
    let _ = OutputVerifier::new(1, vec![f32::NAN]);
}
