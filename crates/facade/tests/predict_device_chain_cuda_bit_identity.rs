//! `DeviceParamStore::predict_device_chain`（イシュー #1688・親 #1581／
//! #1580・#1689・`docs/inference-chain-single-sync-design.md`）の CUDA
//! 実機上での bit 同一・数値一致回帰。
//!
//! `crates/facade/tests/predict_device_chain_cpu_bit_exact.rs`（CPU 版・
//! 非 ignore）の CUDA 版。CPU は `BackendOps::linear_forward_device` を
//! 実装済みのため常に chain 経路（`predict_device_chain`）を通るのに対し、
//! CUDA も #1688 で `linear_forward_device_tracked`（default メソッド。
//! `docs/inference-chain-single-sync-design.md` §9）が結線済みのため
//! 同様に chain 経路を通る。本ファイルは実機（DGX Spark GB10 等）必須の
//! ため全テストを `#[ignore]` で分離する（`.claude/rules/ci.md`
//! 「実機依存」節・`crates/backend-cuda/tests/linear_forward_device_
//! real_device.rs` と同じ方針）。
//!
//! ```sh
//! cargo test -p fandhe-ai --release --test predict_device_chain_cuda_bit_identity -- --ignored --nocapture
//! ```
//!
//! テスト構成（イシュー #1689 実装計画 §4.1）:
//! 1. 小形状 2 種（`Linear→ReLU→Linear`・`Linear→Linear`。CPU 版と同一
//!    構成）＋ bench 形状（batch 64・784→256→ReLU→10。`bench-fandhe`
//!    の `build_model`／`mlp_data` と同型）で chain 経路（`predict_
//!    resident`）と旧経路（`forward_resident`）が bit 完全一致・
//!    run-to-run bit 完全一致することを確認する
//!    （`predict_device_chain_matches_legacy_path_bit_exact_on_cuda_*`）。
//! 2. 同モデルの CUDA 出力を CPU `Sequential::predict`（内部で default
//!    CPU tape を構築する既存経路）と REQ-2 統一複合判定
//!    （`fandhe_ai_backend_cpu::assert_parity` へ委譲。バックエンド間
//!    数値一致テストの許容誤差を単独で緩和しない）で突合する
//!    （`predict_resident_matches_cpu_reference_on_cuda`）。
//! 3. `predict_resident_bit_dump_cuda`: bench 形状の出力を
//!    `out[<i>].bits=<hex>` 形式で 1 要素 1 行印字する（イシュー #1689
//!    R2〈before/after ツリー間の cross-tree bit diff〉用。
//!    `docs/perf/logs/infer-chain-single-sync-cuda-1689/run_bitdump.sh`
//!    から `--exact --nocapture` 単独実行される想定。期待行数は
//!    `BATCH * D_OUT` から機械的に決まる〈本ファイルでは 64*10=640〉）。

use fandhe_ai::Device;
use fandhe_ai::compat::Sequential;
use fandhe_ai_tensor_core::Tensor;

fn tensor(data: Vec<f32>, shape: &[usize]) -> Tensor<f32> {
    Tensor::new(data, shape).expect("test fixture: shape とデータ長は事前に一致させている")
}

fn dense_vec(t: &Tensor<f32>) -> Vec<f32> {
    t.contiguous()
        .as_slice()
        .expect("contiguous() 直後は必ず as_slice() が Some を返す")
        .to_vec()
}

/// `dense_vec` の各要素を [`f32::to_bits`] で比較する（`predict_device_
/// chain_cpu_bit_exact.rs::bits_vec` と同一根拠: `assert_eq!` の
/// `Vec<f32>` 比較は符号付きゼロを区別できないため、bit 完全一致の
/// 検証には `to_bits` によるビット列比較を用いる）。
fn bits_vec(t: &Tensor<f32>) -> Vec<u32> {
    dense_vec(t).into_iter().map(f32::to_bits).collect()
}

/// `model` を CUDA 上で chain 経路（`predict_resident`）・旧経路
/// （`forward_resident` 経由）の双方で forward し、出力が bit 完全一致
/// することを確認する共通ロジック（`predict_device_chain_cpu_bit_exact.
/// rs::assert_chain_matches_legacy_bit_exact` の CUDA 版。`tape_for`
/// が `Device::Cuda(0)` を明示する点のみ異なる）。
fn assert_chain_matches_legacy_bit_exact_on_cuda(model: &Sequential, input: &Tensor<f32>) {
    let device = Device::Cuda(0);

    // chain 経路（`predict_resident`）。#1688 で CUDA も `linear_forward_
    // device_tracked` が結線済みのためこちらを通る。
    let chain_init_tape = fandhe_ai::tape_for(device)
        .expect("CUDA device 0 must be available on ignored test runner");
    let chain_store = model.init_device_param_store(&chain_init_tape).unwrap();
    drop(chain_init_tape);
    let chain_output = model.predict_resident(&chain_store, input).unwrap();

    // run-to-run bit 同一: 同一 store・同一入力で 2 回目も完全に同じ
    // 出力になることを確認する（decision 6 の 1 点目）。
    let chain_output_again = model.predict_resident(&chain_store, input).unwrap();
    assert_eq!(
        bits_vec(&chain_output),
        bits_vec(&chain_output_again),
        "predict_resident（CUDA）の run-to-run 出力が bit 一致しない"
    );

    // 旧経路（`forward_resident`）。`register_resident_params` を通す
    // ため `&mut store` が必要。`abandon_pending_forward` で forward 後
    // の pending を明示的にクリアしてから比較する。
    let legacy_init_tape = fandhe_ai::tape_for(device).unwrap();
    let mut legacy_store = model.init_device_param_store(&legacy_init_tape).unwrap();
    drop(legacy_init_tape);
    let legacy_tape = fandhe_ai::tape_for(device).unwrap();
    let input_var = legacy_tape.var(input);
    let legacy_output = model
        .forward_resident(&legacy_tape, &input_var, &mut legacy_store)
        .unwrap()
        .to_tensor();
    legacy_store.abandon_pending_forward();

    assert_eq!(
        bits_vec(&chain_output),
        bits_vec(&legacy_output),
        "chain 経路（predict_device_chain）と旧経路（forward_resident）の出力（CUDA）が bit 一致しない"
    );
}

/// モデル A: `Linear(bias)→ReLU→Linear(bias)`。chain の `ReLU` 融合を
/// CUDA 上で経由する経路を検証する（`predict_device_chain_cpu_bit_
/// exact.rs` モデル A の CUDA 版）。
#[test]
#[ignore = "CUDA 実機（DGX Spark GB10 等）必須"]
fn predict_device_chain_matches_legacy_path_bit_exact_on_cuda_relu_fusion() {
    let model = Sequential::new()
        .add_linear(4, 8, 0x1001)
        .unwrap()
        .add_relu()
        .add_linear(8, 3, 0x2002)
        .unwrap();

    let input = tensor(
        (0..2 * 4).map(|i| (i as f32) * 0.1 - 0.5).collect(),
        &[2, 4],
    );

    assert_chain_matches_legacy_bit_exact_on_cuda(&model, &input);
}

/// モデル B: `Linear→Linear`（層間に活性化なし。`Activation::None` の
/// まま 2 層を辿る非融合の複数層経路）を CUDA 上で検証する。
#[test]
#[ignore = "CUDA 実機（DGX Spark GB10 等）必須"]
fn predict_device_chain_matches_legacy_path_bit_exact_on_cuda_no_activation_fusion() {
    let model = Sequential::new()
        .add_linear(4, 8, 0x3003)
        .unwrap()
        .add_linear(8, 3, 0x4004)
        .unwrap();

    let input = tensor(
        (0..2 * 4).map(|i| (i as f32) * 0.07 + 0.2).collect(),
        &[2, 4],
    );

    assert_chain_matches_legacy_bit_exact_on_cuda(&model, &input);
}

const BATCH: usize = 64;
const D_IN: usize = 784;
const D_HIDDEN: usize = 256;
const D_OUT: usize = 10;

// `scripts/bench/framework-compare/bench-common/src/lib.rs::{SEED_X,
// SEED_L1, SEED_L2}` と同一の値（cross-workspace 依存を避けリテラルで
// 複製。bench-common は独立 workspace のため facade からは参照不可）。
// 以前はこのテスト独自のシード（`SEED_INPUT = 0xC0FFEE`・`SEED_L1 =
// 0x5EED_0001`・`SEED_L2 = 0x5EED_0002`）を使っており、R1/R2 が実際に
// `bench-fandhe --task infer` が計測する入力・重みと異なるデータに対する
// bit 一致検証になっていた（codex-review 指摘）。bench-fandhe の
// `mlp_data`／`build_model` と同一シードへ揃えることで、R1（単一ツリー
// bit 一致）・R2（cross-tree bit diff）が性能計測対象の出力そのものの
// bit 同一の裏付けになるようにする。
const SEED_X: u64 = 0xDA7A_0001;
const SEED_L1: u64 = 0x1111_1111;
const SEED_L2: u64 = 0x2222_2222;

/// `scripts/bench/framework-compare/bench-common/src/lib.rs::Xorshift64Star`
/// の複製（cross-workspace 依存を避けリテラルで複製。上記コメント参照）。
///
/// **注意**: `bench_harness::rng::Xorshift64Star`（本クレートの
/// `bench-harness` 依存）とはシフト定数・`[0, 1)` から `f32` への写像が
/// 異なる別実装であり、同一シードでも異なる系列を生成する（Bugbot 指摘。
/// `bench-harness` 側は `<<13`/`>>7`/`<<17` シフト・`[-1, 1)` 写像、
/// `bench-common` 側は `>>12`/`<<25`/`>>27` シフト・`fill_vec` で
/// `next_f32() - 0.5` による `[-0.5, 0.5)` 写像）。`bench-fandhe --task
/// infer` の `mlp_data` が生成する入力テンソルと bit 完全に同一の値を
/// このテストでも生成するため、`bench_harness::rng::Xorshift64Star` は
/// 使わず本構造体（`bench-common` の実装をそのまま複製したもの）だけを
/// 入力生成に用いる（重みは `add_linear(D_IN, D_HIDDEN, SEED_L1)` 等が
/// facade 自身の RNG で決定的に生成するため、この乱数系列の相違とは
/// 無関係にすでに一致している）。
struct BenchCommonXorshift64Star {
    state: u64,
}

impl BenchCommonXorshift64Star {
    fn new(seed: u64) -> Self {
        Self {
            state: if seed == 0 {
                0x9E37_79B9_7F4A_7C15
            } else {
                seed
            },
        }
    }

    fn next_u64(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.state = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }

    /// `[0, 1)` の一様分布の f32 を返す（`bench-common` の
    /// `next_f32` と同一の写像。写像自体は `[-0.5, 0.5)` への平行移動を
    /// 含まない点に注意。平行移動は `fill_vec` 側で行う）。
    fn next_f32(&mut self) -> f32 {
        ((self.next_u64() >> 40) as f32) / (1u32 << 24) as f32
    }

    /// `[-0.5, 0.5)` の範囲の f32 ベクトルを生成する（`bench-common::
    /// Xorshift64Star::fill_vec` と bit 完全一致する式）。
    fn fill_vec(&mut self, n: usize) -> Vec<f32> {
        (0..n).map(|_| self.next_f32() - 0.5).collect()
    }
}

/// `scripts/bench/framework-compare/bench-fandhe/src/main.rs::build_model`
/// ／`mlp_data` と同型の bench 形状モデル（784→256→ReLU→10・
/// `bench-common` の `Xorshift64Star` と bit 完全一致する決定的シード）
/// を組み立てる。事前登録規則（イシュー #1689・`docs/perf/infer-chain-
/// single-sync-cuda-ab.md`）が対象とする `bench-fandhe --task infer` の
/// 実形状で chain/legacy 一致・parity を検証するための共通フィクスチャ。
/// シード値（`SEED_X`／`SEED_L1`／`SEED_L2`）に加えて乱数生成アルゴリズム
/// 自体も `bench-common` と揃えてあるため（`BenchCommonXorshift64Star`
/// 参照）、`bench-fandhe --task infer` が実際に計測する入力・モデルと
/// 同一データを生成する。
fn bench_shape_model_and_input() -> (Sequential, Tensor<f32>) {
    let model = Sequential::new()
        .add_linear(D_IN, D_HIDDEN, SEED_L1)
        .unwrap()
        .add_relu()
        .add_linear(D_HIDDEN, D_OUT, SEED_L2)
        .unwrap();
    let mut rng = BenchCommonXorshift64Star::new(SEED_X);
    let input = tensor(rng.fill_vec(BATCH * D_IN), &[BATCH, D_IN]);
    (model, input)
}

/// bench 形状（batch=64・784→256→ReLU→10）で chain/legacy 経路の bit
/// 完全一致・run-to-run bit 完全一致を確認する（イシュー #1689 実装計画
/// §4.1 の「bench 形状」対応。事前登録規則が対象とする実際の推論チェーン
/// 形状で決定 6 を直接検証する）。
#[test]
#[ignore = "CUDA 実機（DGX Spark GB10 等）必須"]
fn predict_device_chain_matches_legacy_path_bit_exact_on_cuda_bench_shape() {
    let (model, input) = bench_shape_model_and_input();
    assert_chain_matches_legacy_bit_exact_on_cuda(&model, &input);
}

/// bench 形状モデルの CUDA `predict_resident`（chain 経路）出力を CPU
/// `Sequential::predict`（内部で default CPU tape を構築する既存経路。
/// `crates/facade/src/compat/sequential.rs::predict`）と REQ-2 統一
/// 複合判定で突合する（`.claude/rules/coding-rust.md`「バックエンド間
/// 数値一致は統一複合判定」・独自閾値を書かず `fandhe_ai_backend_cpu::
/// assert_parity` へ委譲する）。
#[test]
#[ignore = "CUDA 実機（DGX Spark GB10 等）必須"]
fn predict_resident_matches_cpu_reference_on_cuda() {
    let (model, input) = bench_shape_model_and_input();

    let cuda_init_tape = fandhe_ai::tape_for(Device::Cuda(0))
        .expect("CUDA device 0 must be available on ignored test runner");
    let cuda_store = model.init_device_param_store(&cuda_init_tape).unwrap();
    drop(cuda_init_tape);
    let cuda_output = model.predict_resident(&cuda_store, &input).unwrap();

    let cpu_output = model.predict(&input).unwrap();

    let actual = cuda_output.contiguous();
    let expected = cpu_output.contiguous();
    fandhe_ai_backend_cpu::assert_parity(
        "predict_resident（CUDA chain 経路）vs Sequential::predict（CPU 参照）",
        actual.as_slice().unwrap(),
        expected.as_slice().unwrap(),
    );
}

/// R2（cross-tree bit diff）用の生値ダンプ。bench 形状モデルの CUDA
/// `predict_resident`（chain 経路）出力を `out[<i>].bits=<hex>` 形式で
/// 1 要素 1 行印字する。`docs/perf/logs/infer-chain-single-sync-cuda-
/// 1689/run_bitdump.sh` が before／after 2 ツリーでこのテストを個別実行
/// し、標準出力から `^out\[` 行を抽出して diff する（`--exact
/// --nocapture` 単独実行が前提。期待行数は `BATCH * D_OUT` = 640 行）。
#[test]
#[ignore = "CUDA 実機（DGX Spark GB10 等）必須"]
fn predict_resident_bit_dump_cuda() {
    let (model, input) = bench_shape_model_and_input();

    let cuda_init_tape = fandhe_ai::tape_for(Device::Cuda(0))
        .expect("CUDA device 0 must be available on ignored test runner");
    let cuda_store = model.init_device_param_store(&cuda_init_tape).unwrap();
    drop(cuda_init_tape);
    let output = model.predict_resident(&cuda_store, &input).unwrap();

    for (i, bits) in bits_vec(&output).into_iter().enumerate() {
        println!("out[{i}].bits={bits:#010x}");
    }
}

/// `bits_vec`（`f32::to_bits`）が符号付きゼロを区別できることを確認する
/// （`predict_device_chain_cpu_bit_exact.rs::bits_vec_distinguishes_
/// signed_zero` の複製。実機非依存・非 ignore のためコンパイル・実行
/// の両方を通常 CI で検証する）。
#[test]
fn bits_vec_distinguishes_signed_zero() {
    let positive_zero = tensor(vec![0.0_f32, 1.0, -2.0], &[3]);
    let negative_zero = tensor(vec![-0.0_f32, 1.0, -2.0], &[3]);

    assert_eq!(
        dense_vec(&positive_zero),
        dense_vec(&negative_zero),
        "IEEE 754 の `==` は +0.0 と -0.0 を等しいと判定するはず"
    );
    assert_ne!(
        bits_vec(&positive_zero),
        bits_vec(&negative_zero),
        "bits_vec は +0.0 と -0.0 を異なるビット列として区別できるはず"
    );
    assert_eq!(bits_vec(&positive_zero)[0], 0.0_f32.to_bits());
    assert_eq!(bits_vec(&negative_zero)[0], (-0.0_f32).to_bits());
}

/// `BenchCommonXorshift64Star` が `scripts/bench/framework-compare/
/// bench-common/src/lib.rs::Xorshift64Star` と bit 完全一致する出力を
/// 生成することを固定値で回帰検証する（Bugbot 指摘：`bench_harness::
/// rng::Xorshift64Star` とはシフト定数・写像が異なる別実装であるため、
/// 複製が乖離しないことをこのテストで機械的に担保する。期待値は
/// `bench-common` 側の `Xorshift64Star::new(SEED_X).fill_vec(5)` を
/// 実際に実行して得た値〈2026-09-13 実測〉）。通常 CI（`#[ignore]`
/// なし）で実行する。
#[test]
fn bench_common_xorshift64_star_matches_upstream_bit_exact() {
    let mut rng = BenchCommonXorshift64Star::new(SEED_X);
    let actual = rng.fill_vec(5);
    let expected: Vec<f32> = vec![-0.20255262, 0.09809947, 0.413486, -0.00042378902, 0.2202937];
    assert_eq!(
        actual.into_iter().map(f32::to_bits).collect::<Vec<_>>(),
        expected.into_iter().map(f32::to_bits).collect::<Vec<_>>(),
        "BenchCommonXorshift64Star が bench-common::Xorshift64Star と bit 一致しない"
    );
}
