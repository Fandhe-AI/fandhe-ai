//! イシュー #2115: 推論チェーンの CUDA Graph capture（opt-in
//! `FANDHE_AI_CUDA_GRAPH_INFER`・既定 OFF）の bit 同一検証（R2）用の
//! 生値ダンプ。`predict_device_chain_cuda_bit_identity.rs` の
//! `predict_resident_bit_dump_cuda`（batch 64 固定）を、判定セルの batch
//! （64／1024／4096。`bench-fandhe --infer-batch`）へ広げたもの。
//!
//! **必ず `--exact` で 1 テストずつ、OFF（環境変数未設定）と ON
//! （`FANDHE_AI_CUDA_GRAPH_INFER=1`）の 2 プロセスで実行する**。opt-in は
//! 最初の CUDA デバイス初期化前に環境変数で決まり（created stream は
//! ordinal ごとに sticky）、同一プロセス内では切り替えられないため。
//! 標準出力から `^out\[` 行を抽出して OFF／ON を diff し、全行一致を
//! R2 の合格とする（`docs/perf/logs/cuda-infer-chain-graphcapture-2115/
//! README.md`）。ON では 1 回目 capture・2 回目以降 replay になるため、
//! 各テストは 3 回 `predict_resident` を呼び、3 回の出力が bit 一致する
//! ことも同時に検証してからダンプする。
//!
//! ON では predict_resident が capture 1 回・replay 2 回を実際に辿ることを
//! `infer_graph_stats` の差分で検証する（結線が外れて OFF 経路へ落ちた場合に
//! bit 一致だけで合格する取りこぼしを防ぐ）。OFF では capture／replay が 0
//! 回であることを検証する。
//!
//! 判定は fail-closed とする（テストの終了状態と各ダンプの行数 `batch * 10`
//! を確認する。`grep` 単独ではテスト失敗で空ダンプ同士が一致し偽合格になる）:
//!
//! ```sh
//! b=1024
//! for mode in off on; do
//!   if [ "$mode" = on ]; then export FANDHE_AI_CUDA_GRAPH_INFER=1; else unset FANDHE_AI_CUDA_GRAPH_INFER; fi
//!   cargo test -p fandhe-ai --release --test predict_device_chain_cuda_graph_bit_identity \
//!     -- --ignored --exact --nocapture "predict_resident_bit_dump_cuda_graph_batch_$b" \
//!     > "raw-$mode-$b.txt" || { echo "R2-FAIL: test failed mode=$mode"; break; }
//!   grep '^out\[' "raw-$mode-$b.txt" > "dump-$mode-$b.txt"
//!   [ "$(wc -l < "dump-$mode-$b.txt")" -eq $((b * 10)) ] || { echo "R2-FAIL: dump 行数不正 mode=$mode"; break; }
//! done
//! diff "dump-off-$b.txt" "dump-on-$b.txt" && echo R2-PASS
//! ```

use fandhe_ai::Device;
use fandhe_ai::compat::Sequential;
use fandhe_ai_backend_cuda::graph::{infer_graph_enabled, infer_graph_stats};
use fandhe_ai_tensor_core::Tensor;

fn tensor(data: Vec<f32>, shape: &[usize]) -> Tensor<f32> {
    Tensor::new(data, shape).expect("test fixture: shape とデータ長は事前に一致させている")
}

fn bits_vec(t: &Tensor<f32>) -> Vec<u32> {
    t.contiguous()
        .as_slice()
        .expect("contiguous() 直後は必ず as_slice() が Some を返す")
        .iter()
        .map(|v| v.to_bits())
        .collect()
}

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

/// bench 形状モデル（784→256→ReLU→10）と `batch` 行の入力
/// （`bench-fandhe --task infer --infer-batch` と同一の重み・入力系列）。
fn bench_shape_model_and_input(batch: usize) -> (Sequential, Tensor<f32>) {
    let model = Sequential::new()
        .add_linear(D_IN, D_HIDDEN, SEED_L1)
        .unwrap()
        .add_relu()
        .add_linear(D_HIDDEN, D_OUT, SEED_L2)
        .unwrap();
    let mut rng = BenchCommonXorshift64Star::new(SEED_X);
    let input = tensor(rng.fill_vec(batch * D_IN), &[batch, D_IN]);
    (model, input)
}

/// `predict_resident` を 3 回呼び（ON では capture→replay→replay）、3 回が
/// bit 一致することを確認したうえで `out[<i>].bits=<hex>` を 1 要素 1 行
/// 印字する。
fn dump(batch: usize) {
    let on = infer_graph_enabled();
    let (model, input) = bench_shape_model_and_input(batch);
    let init_tape = fandhe_ai::tape_for(Device::Cuda(0))
        .expect("CUDA device 0 must be available on ignored test runner");
    let store = model.init_device_param_store(&init_tape).unwrap();
    drop(init_tape);
    let stats_before = infer_graph_stats();
    let first = model.predict_resident(&store, &input).unwrap();
    for i in 1..3 {
        let again = model.predict_resident(&store, &input).unwrap();
        assert_eq!(
            bits_vec(&first),
            bits_vec(&again),
            "predict_resident の {i} 回目が 1 回目と bit 不一致（replay の不安定）"
        );
    }
    // predict_resident → capture 経路の結線検証（bit 一致だけでは結線が
    // 外れて OFF 経路へ落ちても合格してしまう）。
    let stats_after = infer_graph_stats();
    let captured = stats_after.captured - stats_before.captured;
    let replayed = stats_after.replayed - stats_before.replayed;
    if on {
        assert_eq!(captured, 1, "ON: 1 回目の capture が 1 回であること");
        assert_eq!(replayed, 2, "ON: 2・3 回目が replay 2 回であること");
    } else {
        assert_eq!((captured, replayed), (0, 0), "OFF: capture／replay は 0 回");
    }
    for (i, bits) in bits_vec(&first).into_iter().enumerate() {
        println!("out[{i}].bits={bits:#010x}");
    }
}

#[test]
#[ignore = "CUDA 実機（DGX Spark GB10 等）必須。--exact で単独実行（OFF/ON の 2 プロセス）"]
fn predict_resident_bit_dump_cuda_graph_batch_64() {
    dump(64);
}

#[test]
#[ignore = "CUDA 実機（DGX Spark GB10 等）必須。--exact で単独実行（OFF/ON の 2 プロセス）"]
fn predict_resident_bit_dump_cuda_graph_batch_1024() {
    dump(1024);
}

#[test]
#[ignore = "CUDA 実機（DGX Spark GB10 等）必須。--exact で単独実行（OFF/ON の 2 プロセス）"]
fn predict_resident_bit_dump_cuda_graph_batch_4096() {
    dump(4096);
}
