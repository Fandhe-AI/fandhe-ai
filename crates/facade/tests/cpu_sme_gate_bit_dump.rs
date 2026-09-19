//! イシュー #2049: `SME_PRODUCTION_ENABLED`（`crates/backend-cpu/src/
//! gemm_blis/mod.rs:3085`。既定 `false`。aarch64 SME `fmopa` マイクロ
//! カーネルを `GemmDriverVariant::TwoDDynamic` へ形状条件付きで結線
//! するかどうかの単一 const ゲート。`TWO_D_DYNAMIC_PRODUCTION_ENABLED`・
//! `thread_limit::BIG_CORE_LIMIT_ENABLED` と同型のロールバック機構
//! （同ファイル doc comment 参照）で、採否記録は
//! `docs/perf/cpu-gemm-sme-fmopa-microkernel.md`）の on/off で CPU 側
//! 出力が bit 完全一致することを、before（`SME_PRODUCTION_ENABLED =
//! false`）／after（同 `true`）の 2 ツリーで本ファイルを実行し
//! 標準出力を diff することで確認する診断用テスト。
//!
//! **本ファイル単体では `SME_PRODUCTION_ENABLED` を切り替えない**
//! （常に checkout 済みのツリーの値のまま実行される）。GB10 実機での
//! before/after 2 ツリー比較手順は
//! `docs/perf/logs/cpu-gemm-sme-fmopa-1587/gb10/run_bitdump.sh` を
//! 参照。
//!
//! # 対象ラベル（`docs/perf/cpu-gemm-sme-fmopa-microkernel.md` §5.4.2
//! 「正誤（PR #2016 レビュー指摘による再分類）」の層別 GEMM 形状表を
//! 根拠に選定。`sme_shape_eligible(m_total, n, k)` は `m_total >= 256
//! && n >= 256 && k >= 64`）:
//!
//! - **`gemm512`／`gemm1024`／`gemm2048`**（到達形状。gemm cpu タスクの
//!   正方形状 `m=n=k`）: [`fandhe_ai::tape_for`]`(Device::Cpu)` 上の
//!   `Var::matmul`（NN・rank 2）で `CpuBackendOps::gemm` →
//!   `gemm_blis_parallel_with_transpose` → `dispatch_two_d_dynamic` へ
//!   到達する。512/1024/2048 はいずれも `m=n=k` が `SME_MIN_M/N=256`・
//!   `SME_MIN_K=64` を上回るため 3 形状とも SME 到達（§5.4.2 表の
//!   `min>=512 かつ k>=512` は R4 採用候補 2 組いずれにも含まれる）。
//! - **`train.step<s>.{loss,grad<p>,param<p>}`**（到達形状。train
//!   size=64 の reuse 経路）: MLP 784→256（ReLU）→10・バッチ 64 を
//!   `DeviceParamStore`（`Sequential::forward_resident`）で 3 step
//!   学習する。L1 の d_weight GEMM（TN・`fill_resident_weight_grad` →
//!   `gemm_fp32_strict_into` → `gemm_blis_parallel_tn` →
//!   `dispatch_two_d_dynamic`）は `(m, n, k) = (784, 256, 64)` で
//!   `sme_shape_eligible` を満たす（§5.4.2 表「L1 d_weight」行）。
//!   L1 forward・L2 forward／d_weight／d_input はいずれも非到達
//!   （同表の他行）だが、`param_grads_to_host`／
//!   `sync_device_param_store_to_host` の戻り値は全パラメータ・全勾配
//!   を含むため、本テストは到達・非到達の両方の演算結果を跨いだ
//!   bit 一致を一括で検証する。
//! - **`infer`**（非到達対照。同表「infer forward」行）: 同じ MLP 形状
//!   （バッチ 64）を `fandhe_ai::compat::Sequential::predict`
//!   （tape 不要経路。`CpuBackendOps` 固定・`Linear`→`ReLU` の
//!   `gemm_bias_act` 融合を含む）で 1 回推論する。forward の `m`
//!   （バッチ数 64）が `SME_MIN_M=256` を下回るため、ゲートの値に
//!   関わらず常に非 SME 経路（既存 `NeonKernel`）が選ばれる想定の
//!   非到達対照。
//!
//! 出力行形式は `out[<label>][<i>].bits=0x........`（1 要素 1 行。
//! `{:#010x}` = "0x" + 0 パディング 8 桁の小文字 16 進）。
//!
//! 実行コマンド（`#[ignore]` は before/after ツリー間 diff 専用のため。
//! `--release` は速度のためで数値契約には影響しない。2048² 形状を
//! 含むため非 release 実行は現実的な時間で終わらない）:
//!
//! ```sh
//! cargo test -p fandhe-ai --release --test cpu_sme_gate_bit_dump \
//!   -- --ignored --nocapture --exact dump_cpu_sme_gate_bits
//! ```
//!
//! # 期待行数の内訳（`EXPECTED_TOTAL_LINES` としてテスト内で算出・
//! 自己検証する）
//!
//! - `gemm512`: 512 * 512 = 262,144
//! - `gemm1024`: 1024 * 1024 = 1,048,576
//! - `gemm2048`: 2048 * 2048 = 4,194,304
//! - `train`（`TRAIN_STEPS=3` step 分。1 step あたり
//!   `loss`(1) + `grad`(`w1`+`b1`+`w2`+`b2` = 784*256 + 256 + 256*10 +
//!   10 = 203,530) + `param`(同 203,530) = 407,061 行）:
//!   3 * 407,061 = 1,221,183
//! - `infer`: 64 (バッチ) * 10 (`D_OUT`) = 640
//! - 合計: 262,144 + 1,048,576 + 4,194,304 + 1,221,183 + 640 =
//!   **6,726,847**

use std::io::{self, BufWriter, Write};

use bench_harness::rng::Xorshift64Star;
use fandhe_ai::compat::Sequential;
use fandhe_ai::{Device, SgdConfig as FacadeSgdConfig};
use fandhe_ai_autodiff::nn::loss::{MseLoss, Reduction};
use fandhe_ai_tensor_core::Tensor;

/// gemm タスク（正方 NN・`sme_shape_eligible` 到達形状）の対象サイズ。
/// `docs/perf/cpu-gemm-sme-fmopa-microkernel.md` §5.4.2 の gemm cpu
/// 到達セル（512/1024/2048）と同一。
const GEMM_SIZES: [usize; 3] = [512, 1024, 2048];

/// train（reuse 経路。L1 d_weight のみ SME 到達）の MLP 形状。
/// `docs/perf/cpu-gemm-sme-fmopa-microkernel.md` §5.4.2 の層別 GEMM
/// 形状表（784→256→10・バッチ 64）と同一。
const D_IN: usize = 784;
const D_HIDDEN: usize = 256;
const D_OUT: usize = 10;
const BATCH: usize = 64;
const TRAIN_STEPS: usize = 3;
const LR: f32 = 0.01;

const SEED_GEMM: u64 = 0x05F5_E0FF;
const SEED_TRAIN_DATA: u64 = 0xC0FFEE;
const SEED_TRAIN_L1: u64 = 0x1111_1111;
const SEED_TRAIN_L2: u64 = 0x2222_2222;
const SEED_INFER_MODEL_L1: u64 = 0x3333_3333;
const SEED_INFER_MODEL_L2: u64 = 0x4444_4444;
const SEED_INFER_INPUT: u64 = 0xBEEF;

/// train 1 step あたりのパラメータ要素数
/// （`w1`(784*256) + `b1`(256) + `w2`(256*10) + `b2`(10)）。
/// `grad` と `param` は同じ形状集合のためどちらも本定数を使う。
const TRAIN_PARAM_ELEMS: usize = D_IN * D_HIDDEN + D_HIDDEN + D_HIDDEN * D_OUT + D_OUT;
/// train 1 step あたりの出力行数（`loss` 1 行 + `grad` 全要素 +
/// `param` 全要素）。
const TRAIN_STEP_LINES: usize = 1 + TRAIN_PARAM_ELEMS + TRAIN_PARAM_ELEMS;
/// train 全 `TRAIN_STEPS` step 分の出力行数。
const TRAIN_LINES: usize = TRAIN_STEPS * TRAIN_STEP_LINES;

/// infer（非到達対照）の出力行数（バッチ * `D_OUT`）。
const INFER_LINES: usize = BATCH * D_OUT;

/// gemm 3 形状（512/1024/2048 の正方 `m*n`）の合計出力行数。
const GEMM_LINES: usize =
    GEMM_SIZES[0] * GEMM_SIZES[0] + GEMM_SIZES[1] * GEMM_SIZES[1] + GEMM_SIZES[2] * GEMM_SIZES[2];

/// 本テストが印字する総行数（ファイル冒頭 doc の内訳と一致する
/// ことをテスト内で `assert_eq!` する）。
const EXPECTED_TOTAL_LINES: usize = GEMM_LINES + TRAIN_LINES + INFER_LINES;

fn tensor(data: Vec<f32>, shape: &[usize]) -> Tensor<f32> {
    Tensor::new(data, shape).expect("test fixture: shape とデータ長は事前に一致させている")
}

fn scalar(t: &Tensor<f32>) -> f32 {
    t.get(&[]).expect("test fixture: スカラー shape [] のはず")
}

/// `label` の下、`values` の全要素を
/// `out[<label>][<i>].bits=0x........` 形式で 1 要素 1 行印字する。
/// 数百万行規模（`gemm2048` 単体で 4,194,304 行）を高速に書き出すため
/// `println!`（呼び出しごとに標準出力ロック）ではなく、呼び出し元が
/// 共有する `BufWriter` ロック済みハンドルへ直接書く。
fn print_bits<W: Write>(out: &mut W, label: &str, values: &[f32]) -> usize {
    for (i, v) in values.iter().enumerate() {
        writeln!(out, "out[{label}][{i}].bits=0x{:08x}", v.to_bits())
            .expect("test fixture: stdout への書き込みは失敗しない想定");
    }
    values.len()
}

/// `Tensor` を contiguous 化してスライスを取り出す（`grad`／`param`
/// テンソルは resident staging 由来で非連続な場合があるため、
/// `crates/facade/tests/cpu_reuse_step_grad_bit_dump.rs` と同じ手順で
/// 安全に平坦化する）。
fn contiguous_bits<W: Write>(out: &mut W, label: &str, t: &Tensor<f32>) -> usize {
    let contiguous = t.contiguous();
    let slice = contiguous.as_slice().unwrap_or(&[]);
    print_bits(out, label, slice)
}

/// gemm512／gemm1024／gemm2048: `Device::Cpu` 上の `Var::matmul`
/// （NN・rank 2 の正方形状）で `CpuBackendOps::gemm` を直接経由する。
fn dump_gemm_cells<W: Write>(out: &mut W) -> usize {
    let mut rng = Xorshift64Star::new(SEED_GEMM);
    let mut total = 0usize;
    for &n in &GEMM_SIZES {
        let a = tensor(rng.fill_vec(n * n), &[n, n]);
        let b = tensor(rng.fill_vec(n * n), &[n, n]);

        let tape = fandhe_ai::tape_for(Device::Cpu).expect("CPU device must be available");
        let av = tape.var(&a);
        let bv = tape.var(&b);
        let cv = av
            .matmul(&bv)
            .expect("test fixture: 正方形状の matmul は成功するはず");
        let c = cv.to_tensor();

        let label = format!("gemm{n}");
        total += contiguous_bits(out, &label, &c);
    }
    total
}

/// bias あり 2 層 MLP（`Linear` → `ReLU` → `Linear`）。
/// `docs/perf/cpu-gemm-sme-fmopa-microkernel.md` §5.4.2 の層別 GEMM
/// 形状表（784→256→10）と同一構成。
fn build_train_model() -> Sequential {
    Sequential::new()
        .add_linear(D_IN, D_HIDDEN, SEED_TRAIN_L1)
        .unwrap()
        .add_relu()
        .add_linear(D_HIDDEN, D_OUT, SEED_TRAIN_L2)
        .unwrap()
}

fn gen_train_data(seed: u64) -> (Tensor<f32>, Tensor<f32>) {
    let mut rng = Xorshift64Star::new(seed);
    let x = rng.fill_vec(BATCH * D_IN);
    let y = rng.fill_vec(BATCH * D_OUT);
    (tensor(x, &[BATCH, D_IN]), tensor(y, &[BATCH, D_OUT]))
}

/// train.step<s>.{loss,grad<p>,param<p>}: reuse 経路
/// （`DeviceParamStore`／`forward_resident`）で `TRAIN_STEPS` 回 SGD
/// 学習し、各 step の loss・重み勾配（`Tape::param_grads_to_host`）・
/// 更新後パラメータ（`Tape::sync_device_param_store_to_host`）を印字
/// する。`crates/facade/tests/cpu_reuse_step_grad_bit_dump.rs` と同型の
/// 手順（呼び出し窓の契約も同じ）。
fn dump_train_cells<W: Write>(out: &mut W) -> usize {
    let model = build_train_model();
    let (x_data, y_data) = gen_train_data(SEED_TRAIN_DATA);

    let init_tape = fandhe_ai::tape_for(Device::Cpu).expect("CPU device must be available");
    let mut store = model.init_device_param_store(&init_tape).unwrap();
    drop(init_tape);

    let config = FacadeSgdConfig::new(LR);
    let mut total = 0usize;

    for step in 0..TRAIN_STEPS {
        let tape = fandhe_ai::tape_for(Device::Cpu).expect("CPU device must be available");
        let x = tape.var(&x_data);
        let y = tape.var(&y_data);

        let pred = model.forward_resident(&tape, &x, &mut store).unwrap();
        let loss = MseLoss::new(Reduction::Mean).forward(&pred, &y).unwrap();
        let loss_label = format!("train.step{step}.loss");
        total += print_bits(out, &loss_label, &[scalar(&loss.to_tensor())]);

        let grads = tape.backward_device_param_store(&loss, &store).unwrap();

        // 呼び出し窓: backward 直後・`step_device_param_store` の前
        // （`DeviceParamStore::param_grads_to_host` doc の契約。
        // `cpu_reuse_step_grad_bit_dump.rs` と同じ）。
        let step_grads = tape
            .param_grads_to_host(&store, &grads)
            .expect("param_grads_to_host: 呼び出し窓内で呼んでいるはず");
        for (p, t) in step_grads.iter().enumerate() {
            let label = format!("train.step{step}.grad{p}");
            total += contiguous_bits(out, &label, t);
        }

        tape.step_device_param_store(&mut store, &grads, &config)
            .unwrap();

        let step_synced = tape.sync_device_param_store_to_host(&store).unwrap();
        for (p, t) in step_synced.iter().enumerate() {
            let label = format!("train.step{step}.param{p}");
            total += contiguous_bits(out, &label, t);
        }
    }
    total
}

/// 非到達対照 infer size=64: `Sequential::predict`（tape 不要経路。
/// forward の `m`（バッチ 64）が `SME_MIN_M=256` を下回るため常に
/// 非 SME 経路）の出力（バッチ * `D_OUT`）を印字する。train とは
/// 独立の乱数系列・独立のモデルインスタンスを使う。
fn dump_infer_cell<W: Write>(out: &mut W) -> usize {
    let model = Sequential::new()
        .add_linear(D_IN, D_HIDDEN, SEED_INFER_MODEL_L1)
        .unwrap()
        .add_relu()
        .add_linear(D_HIDDEN, D_OUT, SEED_INFER_MODEL_L2)
        .unwrap();

    let mut rng = Xorshift64Star::new(SEED_INFER_INPUT);
    let input = tensor(rng.fill_vec(BATCH * D_IN), &[BATCH, D_IN]);

    let output = model.predict(&input).unwrap();
    contiguous_bits(out, "infer", &output)
}

#[test]
#[ignore = "before/after ツリー間 diff 専用（通常 CI では不要）。--release --nocapture 必須"]
fn dump_cpu_sme_gate_bits() {
    // ファイル冒頭 doc に記載した内訳（262,144 + 1,048,576 + 4,194,304
    // + 1,221,183 + 640 = 6,726,847）が `EXPECTED_TOTAL_LINES` の算出式
    // と一致することを、実際の印字前に固定する（本テストが `#[ignore]`
    // のため通常 CI では走らない。`run_bitdump.sh` 側のリテラル
    // 6726847 とのドリフト検出も兼ねる）。
    assert_eq!(GEMM_LINES, 262_144 + 1_048_576 + 4_194_304);
    assert_eq!(TRAIN_LINES, 1_221_183);
    assert_eq!(INFER_LINES, 640);
    assert_eq!(EXPECTED_TOTAL_LINES, 6_726_847);

    let stdout = io::stdout();
    let mut out = BufWriter::new(stdout.lock());

    let mut total = 0usize;
    total += dump_gemm_cells(&mut out);
    total += dump_train_cells(&mut out);
    total += dump_infer_cell(&mut out);

    out.flush()
        .expect("test fixture: stdout の flush は失敗しない想定");

    // テスト自身による行数の自己検証（ファイル冒頭 doc の内訳・
    // `EXPECTED_TOTAL_LINES` の算出式と、実際に印字した行数の合計が
    // 一致することを確認する）。
    assert_eq!(
        total, EXPECTED_TOTAL_LINES,
        "印字した行数がファイル冒頭 doc の内訳（{EXPECTED_TOTAL_LINES}）と一致しない"
    );
}
