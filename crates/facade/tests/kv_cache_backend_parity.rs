//! KV キャッシュ付き attention（`MultiheadAttentionVars::
//! forward_with_cache`・イシュー #2084・親 #2059）の facade 横断 parity
//! テスト（`mha_backend_parity.rs` と同型）。
//!
//! facade は `add_stateful_attention`／`StatefulAttention` 相当の新規
//! `pub fn` を追加していない（`docs/kv-cache-design.md` §6 承認事項 2。
//! `crates/facade/tests/api_surface.rs` の否定ガードで固定）ため、
//! `mha_backend_parity.rs` と同じ到達経路（`LinearVars`〈pub フィールド〉
//! を自前構築し `MultiheadAttentionVars::new` へ渡す）を使う。
//!
//! - 属性なし: `fandhe_ai::tape()`（`CpuBackendOps`）で「prefill →
//!   decode N ステップ」の各出力を `fandhe_ai_autodiff::Tape::new()`
//!   （`NaiveOps`）と突合し、さらに CPU 本番 ops 上での「全系列
//!   再計算」との REQ-2 一致（`assert_parity`）も検証する。
//! - `#[ignore]`: Metal／CUDA の decode 列と CPU を突合する（実機必須）。

use fandhe_ai::Device;
use fandhe_ai_autodiff::Var;
use fandhe_ai_autodiff::nn::{KvCache, LinearVars, MultiheadAttentionVars};
use fandhe_ai_backend_cpu::parity::assert_parity;
use fandhe_ai_tensor_core::Tensor;

/// `mha_backend_parity.rs::VarSource` と同型（`fandhe_ai::Tape`・
/// `fandhe_ai_autodiff::Tape` のいずれからも `var()` を呼べるようにする）。
trait VarSource {
    fn make_var(&self, tensor: &Tensor<f32>) -> Var<'_>;
}

impl VarSource for fandhe_ai::Tape {
    fn make_var(&self, tensor: &Tensor<f32>) -> Var<'_> {
        self.var(tensor)
    }
}

impl VarSource for fandhe_ai_autodiff::Tape {
    fn make_var(&self, tensor: &Tensor<f32>) -> Var<'_> {
        self.var(tensor)
    }
}

const B: usize = 2;
const E: usize = 4;
const NUM_HEADS: usize = 2;
const TOTAL_LEN: usize = 5;
const PREFILL_LEN: usize = 2;

fn t(data: Vec<f32>, shape: &[usize]) -> Tensor<f32> {
    Tensor::new(data, shape).expect("fixture: shape とデータ長は事前に一致させている")
}

fn seq(seed: i64, len: usize) -> Vec<f32> {
    (0..len)
        .map(|i| ((seed + i as i64) % 17 - 8) as f32 * 0.03)
        .collect()
}

fn full_sequence() -> Tensor<f32> {
    t(seq(211, B * TOTAL_LEN * E), &[B, TOTAL_LEN, E])
}

fn q_weight() -> Tensor<f32> {
    t(seq(1, E * E), &[E, E])
}
fn q_bias() -> Tensor<f32> {
    t(seq(2, E), &[E])
}
fn k_weight() -> Tensor<f32> {
    t(seq(3, E * E), &[E, E])
}
fn k_bias() -> Tensor<f32> {
    t(seq(4, E), &[E])
}
fn v_weight() -> Tensor<f32> {
    t(seq(5, E * E), &[E, E])
}
fn v_bias() -> Tensor<f32> {
    t(seq(6, E), &[E])
}
fn out_weight() -> Tensor<f32> {
    t(seq(7, E * E), &[E, E])
}
fn out_bias() -> Tensor<f32> {
    t(seq(8, E), &[E])
}

/// `mha_backend_parity.rs::build_vars` と同型（8 パラメータを `tape`
/// へ登録し `MultiheadAttentionVars` を構築する）。
fn build_vars<'t>(tape: &'t impl VarSource) -> MultiheadAttentionVars<'t> {
    let q = LinearVars {
        weight: tape.make_var(&q_weight()),
        bias: Some(tape.make_var(&q_bias())),
    };
    let k = LinearVars {
        weight: tape.make_var(&k_weight()),
        bias: Some(tape.make_var(&k_bias())),
    };
    let v = LinearVars {
        weight: tape.make_var(&v_weight()),
        bias: Some(tape.make_var(&v_bias())),
    };
    let out = LinearVars {
        weight: tape.make_var(&out_weight()),
        bias: Some(tape.make_var(&out_bias())),
    };
    MultiheadAttentionVars::new(NUM_HEADS, q, k, v, out)
        .expect("fixture: 4 層とも [E,E]・bias 全 Some・E % NUM_HEADS == 0")
}

// --- CpuBackendOps vs NaiveOps（decode 列の各ステップ突合） -----------

#[test]
fn cpu_prefill_then_decode_matches_naive_reference() {
    let x = full_sequence();
    let mut cache_cpu = KvCache::new();
    let mut cache_naive = KvCache::new();

    // prefill。
    {
        let cpu_tape = fandhe_ai::tape();
        let vars_cpu = build_vars(&cpu_tape);
        let x_cpu = cpu_tape.make_var(&x);
        let x_prefill_cpu = x_cpu.narrow(1, 0, PREFILL_LEN).unwrap();
        let out_cpu = vars_cpu
            .forward_with_cache(
                &x_prefill_cpu,
                &x_prefill_cpu,
                &x_prefill_cpu,
                &mut cache_cpu,
            )
            .unwrap()
            .to_tensor();

        let naive_tape = fandhe_ai_autodiff::Tape::new();
        let vars_naive = build_vars(&naive_tape);
        let x_naive = naive_tape.make_var(&x);
        let x_prefill_naive = x_naive.narrow(1, 0, PREFILL_LEN).unwrap();
        let out_naive = vars_naive
            .forward_with_cache(
                &x_prefill_naive,
                &x_prefill_naive,
                &x_prefill_naive,
                &mut cache_naive,
            )
            .unwrap()
            .to_tensor();

        assert_parity(
            "KV キャッシュ prefill: CpuBackendOps vs NaiveOps",
            out_cpu.as_slice().expect("contiguous"),
            out_naive.as_slice().expect("contiguous"),
        );
    }

    // decode（1 トークンずつ）。
    for step in PREFILL_LEN..TOTAL_LEN {
        let cpu_tape = fandhe_ai::tape();
        let vars_cpu = build_vars(&cpu_tape);
        let x_cpu = cpu_tape.make_var(&x);
        let x_t_cpu = x_cpu.narrow(1, step, 1).unwrap();
        let out_cpu = vars_cpu
            .forward_with_cache(&x_t_cpu, &x_t_cpu, &x_t_cpu, &mut cache_cpu)
            .unwrap()
            .to_tensor();

        let naive_tape = fandhe_ai_autodiff::Tape::new();
        let vars_naive = build_vars(&naive_tape);
        let x_naive = naive_tape.make_var(&x);
        let x_t_naive = x_naive.narrow(1, step, 1).unwrap();
        let out_naive = vars_naive
            .forward_with_cache(&x_t_naive, &x_t_naive, &x_t_naive, &mut cache_naive)
            .unwrap()
            .to_tensor();

        assert_parity(
            &format!("KV キャッシュ decode（step={step}）: CpuBackendOps vs NaiveOps"),
            out_cpu.as_slice().expect("contiguous"),
            out_naive.as_slice().expect("contiguous"),
        );
    }

    assert_eq!(cache_cpu.seq_len(), TOTAL_LEN);
    assert_eq!(cache_naive.seq_len(), TOTAL_LEN);
}

/// CPU 本番 ops 上でも「全系列再計算」と「prefill → decode」が REQ-2
/// 統一複合判定で一致することを検証する（`docs/kv-cache-design.md`
/// §3.5。CPU GEMM の shape 依存ブロッキングのため bit 一致は仮説に
/// 留める）。
#[test]
fn cpu_prefill_then_decode_matches_full_recompute_on_cpu_backend() {
    let x = full_sequence();

    let full_tape = fandhe_ai::tape();
    let vars_full = build_vars(&full_tape);
    let x_full = full_tape.make_var(&x);
    let full_out = vars_full
        .forward(&x_full, &x_full, &x_full, None, true)
        .unwrap()
        .to_tensor();

    let mut cache = KvCache::new();
    let mut outputs: Vec<Vec<f32>> = Vec::new();
    let mut shapes: Vec<usize> = Vec::new();
    {
        let tape = fandhe_ai::tape();
        let vars = build_vars(&tape);
        let xv = tape.make_var(&x);
        let x_prefill = xv.narrow(1, 0, PREFILL_LEN).unwrap();
        let out = vars
            .forward_with_cache(&x_prefill, &x_prefill, &x_prefill, &mut cache)
            .unwrap()
            .to_tensor();
        shapes.push(out.shape()[1]);
        outputs.push(out.contiguous().as_slice().expect("contiguous").to_vec());
    }
    for step in PREFILL_LEN..TOTAL_LEN {
        let tape = fandhe_ai::tape();
        let vars = build_vars(&tape);
        let xv = tape.make_var(&x);
        let x_t = xv.narrow(1, step, 1).unwrap();
        let out = vars
            .forward_with_cache(&x_t, &x_t, &x_t, &mut cache)
            .unwrap()
            .to_tensor();
        shapes.push(out.shape()[1]);
        outputs.push(out.contiguous().as_slice().expect("contiguous").to_vec());
    }

    // [B, TOTAL_LEN, E] を [B, len_i, E] の列から手動連結する
    // （`nn_kv_cache.rs::concat_seq` と同じ考え方。facade テストは別
    // クレートのためヘルパーを共有できない）。
    let mut combined = vec![0.0f32; B * TOTAL_LEN * E];
    for bi in 0..B {
        let mut offset = 0usize;
        for (chunk, &len) in outputs.iter().zip(shapes.iter()) {
            for li in 0..len {
                for ei in 0..E {
                    let src = (bi * len + li) * E + ei;
                    let dst = (bi * TOTAL_LEN + offset + li) * E + ei;
                    combined[dst] = chunk[src];
                }
            }
            offset += len;
        }
    }

    assert_parity(
        "KV キャッシュ prefill+decode vs 全系列再計算: CpuBackendOps",
        &combined,
        full_out.contiguous().as_slice().expect("contiguous"),
    );
}

// --- 実機横断（`#[ignore]`。Metal／CUDA。decode 列を CPU と突合）------

/// 実機 decode 列の突合対象一式（cache 内部状態だけでなく、QK/mask/
/// softmax/V/出力射影を経た attention 最終出力まで含める。旧実装は
/// `cache.k()` のみを比較しており、出力射影〈`out_proj`〉を含む
/// attention 本体の実機数値一致を検証していなかった〈codex-review
/// P2・PR #2240〉）。
struct DecodeSequenceResult {
    /// 各 decode ステップの attention 最終出力（`out_proj` 通過後）を
    /// `[B, TOTAL_LEN, E]` へ手動連結したもの（`cpu_prefill_then_decode_
    /// matches_full_recompute_on_cpu_backend` の `combined` と同じ
    /// 連結方式）。
    combined_output: Vec<f32>,
    /// prefill＋decode 全ステップ後の cache 内部状態（QK 側の入力）。
    cache_k: Tensor<f32>,
    cache_v: Tensor<f32>,
}

fn decode_sequence_on(device: Device) -> DecodeSequenceResult {
    let x = full_sequence();
    let tape = fandhe_ai::tape_for(device).expect("実機が利用可能な前提のテストのため成功するはず");
    let vars = build_vars(&tape);
    let mut cache = KvCache::new();

    let mut outputs: Vec<Vec<f32>> = Vec::new();
    let mut shapes: Vec<usize> = Vec::new();

    let x0 = tape.make_var(&x);
    let x_prefill = x0.narrow(1, 0, PREFILL_LEN).unwrap();
    let out_prefill = vars
        .forward_with_cache(&x_prefill, &x_prefill, &x_prefill, &mut cache)
        .unwrap()
        .to_tensor();
    shapes.push(out_prefill.shape()[1]);
    outputs.push(
        out_prefill
            .contiguous()
            .as_slice()
            .expect("contiguous")
            .to_vec(),
    );

    for step in PREFILL_LEN..TOTAL_LEN {
        let xv = tape.make_var(&x);
        let x_t = xv.narrow(1, step, 1).unwrap();
        let out_t = vars
            .forward_with_cache(&x_t, &x_t, &x_t, &mut cache)
            .unwrap()
            .to_tensor();
        shapes.push(out_t.shape()[1]);
        outputs.push(out_t.contiguous().as_slice().expect("contiguous").to_vec());
    }

    // `[B, TOTAL_LEN, E]` へ手動連結する（`cpu_prefill_then_decode_
    // matches_full_recompute_on_cpu_backend` と同じ考え方。facade
    // テストは別クレートのためヘルパーを共有できない）。
    let mut combined_output = vec![0.0f32; B * TOTAL_LEN * E];
    for bi in 0..B {
        let mut offset = 0usize;
        for (chunk, &len) in outputs.iter().zip(shapes.iter()) {
            for li in 0..len {
                for ei in 0..E {
                    let src = (bi * len + li) * E + ei;
                    let dst = (bi * TOTAL_LEN + offset + li) * E + ei;
                    combined_output[dst] = chunk[src];
                }
            }
            offset += len;
        }
    }

    DecodeSequenceResult {
        combined_output,
        cache_k: cache
            .k()
            .expect("prefill 済みのため cache は非空のはず")
            .clone(),
        cache_v: cache
            .v()
            .expect("prefill 済みのため cache は非空のはず")
            .clone(),
    }
}

fn assert_decode_sequence_parity(
    label: &str,
    actual: &DecodeSequenceResult,
    cpu: &DecodeSequenceResult,
) {
    assert_parity(
        &format!("KV キャッシュ decode 列（attention 最終出力）: {label} vs CPU"),
        &actual.combined_output,
        &cpu.combined_output,
    );
    assert_parity(
        &format!("KV キャッシュ decode 列（最終 cache.k）: {label} vs CPU"),
        actual.cache_k.contiguous().as_slice().expect("contiguous"),
        cpu.cache_k.contiguous().as_slice().expect("contiguous"),
    );
    assert_parity(
        &format!("KV キャッシュ decode 列（最終 cache.v）: {label} vs CPU"),
        actual.cache_v.contiguous().as_slice().expect("contiguous"),
        cpu.cache_v.contiguous().as_slice().expect("contiguous"),
    );
}

// `Device::Metal` variant 自体が `cfg(target_os = "macos")` 限定
// （`mha_backend_parity.rs` と同じ理由でコンパイル自体を macOS 限定に
// する必要がある）。
#[cfg(target_os = "macos")]
#[test]
#[ignore = "Metal 実機（Apple Silicon）依存。CI では実行しない"]
fn metal_decode_cache_matches_cpu() {
    let metal = decode_sequence_on(Device::Metal);
    let cpu = decode_sequence_on(Device::Cpu);
    assert_decode_sequence_parity("Metal", &metal, &cpu);
}

#[test]
#[ignore = "CUDA 実機（DGX Spark GB10 等）必須"]
fn cuda_decode_cache_matches_cpu() {
    let cuda = decode_sequence_on(Device::Cuda(0));
    let cpu = decode_sequence_on(Device::Cpu);
    assert_decode_sequence_parity("CUDA", &cuda, &cpu);
}
