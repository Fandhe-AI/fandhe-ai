//! TASK-12.2a（#167）: elementwise 連鎖・fan-out・transpose 混在
//! ワークロードでの融合効果の実測ハーネス。
//!
//! ## 前提: `run_fused` 結線
//!
//! 本ハーネスは `CpuBackendOps::run_fused`（`crates/backend-cpu/src/ops.rs`）
//! が融合カーネル（`fused_elementwise::run_fused_elementwise`）へ結線
//! されていることを前提とする。この結線は本イシュー（#167）の実装時に
//! 追加した（TASK-12.1 系列〈#163・#164〉が「相手のスコープ」と委ね合い
//! 未結線のまま残っていたギャップ。`crates/backend-cpu/src/ops.rs` の
//! `run_fused` ドキュメンテーションコメント参照）。結線なしでは融合条件・
//! 非融合条件が区別不能で本ハーネスの前提が成立しない。
//!
//! ## A/B 比較方式（PoC-9 の feature 切替の v2 対応物）
//!
//! - **融合条件**: `CpuBackendOps`（`run_fused` オーバーライド込み）を
//!   `Tape::new_with_ops` へ渡す。`autodiff::tape` の遅延評価 2 層
//!   （`push_lazy`／`materialize_*`）が 4 段以上の elementwise 連鎖を
//!   `run_fused` へ回す。
//! - **非融合条件**: `NonFusedCpuOps`（本ファイル定義）——`CpuBackendOps`
//!   の全 per-op メソッドへ委譲しつつ `run_fused` はオーバーライドせず
//!   デフォルト `Unsupported` のまま残すラッパー。`crates/autodiff/tests/
//!   fusion_backend_integration.rs` の `AlwaysUnsupportedFused` と同型
//!   （同ファイルは `common::NaiveOps` へ委譲するが、本ハーネスは実際の
//!   `CpuBackendOps` カーネルへ委譲する点のみ異なる）。`run_fused` が
//!   `Unsupported` を返すと `autodiff::tape` は per-op フォールバックへ
//!   倒れる契約（`docs/kernel-fusion.md` §1 (a)）。
//!
//! 両条件とも同一カーネル・同一実行系（`CpuBackendOps` の per-op 実体）
//! を使い、融合の有無だけが異なる within-harness 比較になる。
//!
//! ## 計測プロトコル
//!
//! `bench_harness::protocol::run`（warmup 20 回・計測 20 回・中央値/Q1/Q3。
//! TASK-8.1 準拠）。入力は `bench_harness::rng::Xorshift64Star`（決定的
//! シード）で `[-1, 1)` へ生成する（`exp` のオーバーフロー回避）。
//!
//! `#[ignore]` として通常 CI から除外する（`tests/gemm_epilogue_perf.rs`
//! と同方針）。受け入れ条件「実測記録が残されている」の実測記録は
//! `docs/perf/cpu-elementwise-fusion-effect.md`。
//!
//! 実行例（AVX2+FMA を有効化してビルド）:
//! ```text
//! RUSTFLAGS="-C target-feature=+avx2,+fma" cargo test -p backend-cpu \
//!     --release --test fusion_effect_perf -- --ignored --nocapture
//! ```

use autodiff::{Tape, Var};
use backend_cpu::CpuBackendOps;
use backend_cpu::parity::assert_parity;
use bench_harness::rng::Xorshift64Star;
use bench_harness::{MeasurementConfig, run};
use tensor_core::device::{BackendError, Device};
use tensor_core::{BackendOps, FusionPlan, Tensor};

/// `CpuBackendOps` の全 per-op メソッドへ委譲しつつ `run_fused` は
/// オーバーライドしない（デフォルト `Unsupported` のまま）ラッパー。
/// 非融合条件（per-op フォールバック経路）を、融合条件と同一カーネル
/// 実体を使いながら区別するための唯一の差分点。
struct NonFusedCpuOps {
    inner: CpuBackendOps,
}

impl BackendOps for NonFusedCpuOps {
    fn device(&self) -> Device {
        self.inner.device()
    }
    fn gemm(&self, a: &Tensor<f32>, b: &Tensor<f32>) -> Result<Tensor<f32>, BackendError> {
        self.inner.gemm(a, b)
    }
    fn gemm_bias_act(
        &self,
        a: &Tensor<f32>,
        b: &Tensor<f32>,
        bias: Option<&Tensor<f32>>,
        act: tensor_core::Activation,
    ) -> Result<Tensor<f32>, BackendError> {
        self.inner.gemm_bias_act(a, b, bias, act)
    }
    fn add(&self, a: &Tensor<f32>, b: &Tensor<f32>) -> Result<Tensor<f32>, BackendError> {
        self.inner.add(a, b)
    }
    fn mul(&self, a: &Tensor<f32>, b: &Tensor<f32>) -> Result<Tensor<f32>, BackendError> {
        self.inner.mul(a, b)
    }
    fn relu(&self, a: &Tensor<f32>) -> Result<Tensor<f32>, BackendError> {
        self.inner.relu(a)
    }
    fn exp(&self, a: &Tensor<f32>) -> Result<Tensor<f32>, BackendError> {
        self.inner.exp(a)
    }
    fn tanh(&self, a: &Tensor<f32>) -> Result<Tensor<f32>, BackendError> {
        self.inner.tanh(a)
    }
    fn sum(&self, a: &Tensor<f32>, dim: Option<usize>) -> Result<Tensor<f32>, BackendError> {
        self.inner.sum(a, dim)
    }
    fn max(&self, a: &Tensor<f32>, dim: Option<usize>) -> Result<Tensor<f32>, BackendError> {
        self.inner.max(a, dim)
    }
    // `run_fused` はデフォルト実装（`Unsupported`）のまま override しない。
    // これが非融合条件（per-op フォールバック経路）の唯一の実現手段。
    fn run_fused(
        &self,
        _plan: &FusionPlan,
        _leaves: &[&Tensor<f32>],
    ) -> Result<Tensor<f32>, BackendError> {
        Err(BackendError::Unsupported(
            "fusion_effect_perf: NonFusedCpuOps は意図的に run_fused を \
             override しない（非融合条件の実現手段）"
                .into(),
        ))
    }
}

fn seeded_tensor_unit_range(seed: u64, shape: &[usize]) -> Tensor<f32> {
    // `[-1, 1)` へスケール（exp のオーバーフロー回避。§3 計測プロトコル）。
    let numel: usize = shape.iter().product();
    let mut rng = Xorshift64Star::new(seed);
    let data: Vec<f32> = (0..numel).map(|_| rng.next_f32() * 2.0 - 1.0).collect();
    Tensor::new(data, shape).unwrap()
}

/// `ew4`（連鎖 4 段）: `add -> mul -> tanh -> mul`。
/// PoC-9 の `ew4` は sigmoid を含むが、sigmoid は v2 の遅延対象外の
/// 組込み複合演算（`docs/kernel-fusion.md` 表 3）のため tanh へ置換する。
fn build_ew4<'t>(x: Var<'t>, y: Var<'t>) -> Var<'t> {
    let h1 = x.add(&y).unwrap();
    let h2 = h1.mul(&y).unwrap();
    let h3 = h2.tanh();
    h3.mul(&x).unwrap()
}

/// `ew6`（連鎖 6 段）: `add -> mul -> tanh -> mul -> add -> tanh`。
fn build_ew6<'t>(x: Var<'t>, y: Var<'t>) -> Var<'t> {
    let h1 = x.add(&y).unwrap();
    let h2 = h1.mul(&y).unwrap();
    let h3 = h2.tanh();
    let h4 = h3.mul(&x).unwrap();
    let h5 = h4.add(&y).unwrap();
    h5.tanh()
}

/// `ew_fanout`: `a = x + y; b = a * a; c = b + x; tanh(c)`。
/// 中間 `a`・葉 `x` を 2 回消費する fan-out パターン（PoC-9 `ew_fanout`
/// 対応。`docs/fusion-graph-design.md` §2.4 の fan-out 融合対象）。
fn build_ew_fanout<'t>(x: Var<'t>, y: Var<'t>) -> Var<'t> {
    let a = x.add(&y).unwrap();
    let b = a.mul(&a).unwrap();
    let c = b.add(&x).unwrap();
    c.tanh()
}

/// `ew4` 相当の 4 段連鎖を 2D テンソルで構成する（`ew_transpose`・
/// `ew_2d_contig` の対照構築に使う共通ヘルパー）。
fn build_ew4_2d<'t>(x: Var<'t>, y: Var<'t>) -> Var<'t> {
    build_ew4(x, y)
}

/// warmup+計測クロージャの共通実行部。`out_of()` は毎回テープを新規
/// 構築し `to_tensor()` で実体化するクロージャ（Tape 構築 → 連鎖記録 →
/// 実体化を 1 イテレーション内で完結させる。§3「各イテレーションは
/// 『Tape 構築 → 連鎖記録 → to_tensor() 実体化』を閉包内で完結」）。
fn measure_pattern<F: FnMut() -> Tensor<f32>>(mut out_of: F) -> bench_harness::Measurement {
    let config = MeasurementConfig::default(); // warmup 20・iters 20（TASK-8.1 下限）
    run(&config, || {
        std::hint::black_box(out_of());
    })
    .expect("計測に失敗")
}

/// 1D パターン（`ew4`／`ew6`／`ew_fanout`）の融合条件・非融合条件を
/// 計測し、サニティ検査（REQ-2 複合判定での最終値一致）を行った上で
/// 標準出力へ記録する。
fn measure_1d_pattern(
    name: &str,
    n: usize,
    seed_x: u64,
    seed_y: u64,
    build: impl for<'a> Fn(Var<'a>, Var<'a>) -> Var<'a>,
) {
    let shape = [n];
    let x_data = seeded_tensor_unit_range(seed_x, &shape);
    let y_data = seeded_tensor_unit_range(seed_y, &shape);

    // サニティ検査: 融合条件・非融合条件の最終値が REQ-2 複合判定で
    // 一致することを 1 回実行で確認する（ガードレール・許容誤差は
    // 変更しない。`.claude/rules/coding-rust.md`）。
    let fused_tape = Tape::new_with_ops(Box::new(CpuBackendOps::new()));
    let x_fused = fused_tape.var(&x_data);
    let y_fused = fused_tape.var(&y_data);
    let fused_out = build(x_fused, y_fused).to_tensor();

    let nonfused_tape = Tape::new_with_ops(Box::new(NonFusedCpuOps {
        inner: CpuBackendOps::new(),
    }));
    let x_nonfused = nonfused_tape.var(&x_data);
    let y_nonfused = nonfused_tape.var(&y_data);
    let nonfused_out = build(x_nonfused, y_nonfused).to_tensor();

    assert_parity(
        &format!("{name}: fused vs non-fused sanity check"),
        fused_out.contiguous().as_slice().expect("contiguous"),
        nonfused_out.contiguous().as_slice().expect("contiguous"),
    );

    // 計測本体: 融合条件。
    let fused_measurement = measure_pattern(|| {
        let tape = Tape::new_with_ops(Box::new(CpuBackendOps::new()));
        let x = tape.var(&x_data);
        let y = tape.var(&y_data);
        build(x, y).to_tensor()
    });

    // 計測本体: 非融合条件（per-op フォールバック）。
    let nonfused_measurement = measure_pattern(|| {
        let tape = Tape::new_with_ops(Box::new(NonFusedCpuOps {
            inner: CpuBackendOps::new(),
        }));
        let x = tape.var(&x_data);
        let y = tape.var(&y_data);
        build(x, y).to_tensor()
    });

    let speedup = nonfused_measurement.median_secs / fused_measurement.median_secs;
    println!(
        "{name} (N={n}): non-fused median={:.6}s (q1={:.6}, q3={:.6}) / \
         fused median={:.6}s (q1={:.6}, q3={:.6}) / speedup={speedup:.3}x",
        nonfused_measurement.median_secs,
        nonfused_measurement.q1_secs,
        nonfused_measurement.q3_secs,
        fused_measurement.median_secs,
        fused_measurement.q1_secs,
        fused_measurement.q3_secs,
    );
}

/// 2D パターン（`ew_transpose`／`ew_2d_contig`）の計測。`transposed` が
/// `true` の場合は葉 `x` を転置 view（非 contiguous）として与え、
/// `run_fused_elementwise` が非 contiguous 葉を `Unsupported` で拒否し
/// per-op フォールバックへ倒れる（`docs/kernel-fusion.md` 限界表 4 行目）
/// ことを実測で裏付ける。
fn measure_2d_pattern(name: &str, d: usize, seed_x: u64, seed_y: u64, transposed: bool) {
    let shape = [d, d];
    let x_base = seeded_tensor_unit_range(seed_x, &shape);
    let x_data = if transposed {
        x_base.transpose_2d().expect("2D transpose")
    } else {
        x_base
    };
    let y_data = seeded_tensor_unit_range(seed_y, &shape);

    let fused_tape = Tape::new_with_ops(Box::new(CpuBackendOps::new()));
    let x_fused = fused_tape.var(&x_data);
    let y_fused = fused_tape.var(&y_data);
    let fused_out = build_ew4_2d(x_fused, y_fused).to_tensor();

    let nonfused_tape = Tape::new_with_ops(Box::new(NonFusedCpuOps {
        inner: CpuBackendOps::new(),
    }));
    let x_nonfused = nonfused_tape.var(&x_data);
    let y_nonfused = nonfused_tape.var(&y_data);
    let nonfused_out = build_ew4_2d(x_nonfused, y_nonfused).to_tensor();

    assert_parity(
        &format!("{name}: fused vs non-fused sanity check"),
        fused_out.contiguous().as_slice().expect("contiguous"),
        nonfused_out.contiguous().as_slice().expect("contiguous"),
    );

    let fused_measurement = measure_pattern(|| {
        let tape = Tape::new_with_ops(Box::new(CpuBackendOps::new()));
        let x = tape.var(&x_data);
        let y = tape.var(&y_data);
        build_ew4_2d(x, y).to_tensor()
    });

    let nonfused_measurement = measure_pattern(|| {
        let tape = Tape::new_with_ops(Box::new(NonFusedCpuOps {
            inner: CpuBackendOps::new(),
        }));
        let x = tape.var(&x_data);
        let y = tape.var(&y_data);
        build_ew4_2d(x, y).to_tensor()
    });

    let speedup = nonfused_measurement.median_secs / fused_measurement.median_secs;
    println!(
        "{name} (D={d}, transposed_leaf={transposed}): non-fused median={:.6}s \
         (q1={:.6}, q3={:.6}) / fused median={:.6}s (q1={:.6}, q3={:.6}) / speedup={speedup:.3}x",
        nonfused_measurement.median_secs,
        nonfused_measurement.q1_secs,
        nonfused_measurement.q3_secs,
        fused_measurement.median_secs,
        fused_measurement.q1_secs,
        fused_measurement.q3_secs,
    );
}

#[test]
#[ignore = "性能計測ハーネス。--release かつ RUSTFLAGS で AVX2+FMA を有効化して個別実行する想定"]
fn fusion_effect_perf_all_patterns() {
    // 主サイズ: N=1e7（f32 40MB/テンソル）。共有ホスト（QEMU 仮想 CPU・
    // 複数エージェント並列実行中）での実行時間を抑えるため、計画（§3）が
    // 許容する縮小サイズ（PoC-9 の N=4e7 主サイズから縮小）を採用する。
    // 縮小した旨は実測記録（`docs/perf/cpu-elementwise-fusion-effect.md`）
    // に明記する。
    const N_PRIMARY: usize = 10_000_000;
    const D_PRIMARY: usize = 2048;

    measure_1d_pattern("ew4", N_PRIMARY, 401, 402, build_ew4);
    measure_1d_pattern("ew6", N_PRIMARY, 403, 404, build_ew6);
    measure_1d_pattern("ew_fanout", N_PRIMARY, 405, 406, build_ew_fanout);

    // transpose 混在: 融合効果が出ない（速度比 ≈ 1.0）ことの実測裏付け。
    measure_2d_pattern("ew_transpose", D_PRIMARY, 407, 408, true);
    // 対照条件: 同一連鎖を contiguous 葉で実行し transpose の影響を分離。
    measure_2d_pattern("ew_2d_contig", D_PRIMARY, 407, 408, false);
}
