//! AMP 低精度 forward（イシュー #2071）の MultiheadAttention CUDA／
//! Metal 実機 parity テスト（`mha_backend_parity.rs` と同型）。
//!
//! `nn::multihead_attention_forward_low_precision`（`pub` 自由関数。
//! `Var::matmul_low_precision` 自体は `pub(crate)` のため facade の
//! 統合テストからは直接呼べない）を `tape_for(Device::Cuda(0))`／
//! `tape_for(Device::Metal)`（`cfg(target_os = "macos")` 限定）で実行
//! し、CPU（`fandhe_ai::tape()`）の同一 dtype（F16／Bf16）低精度
//! forward と `fandhe_ai_backend_cpu::parity::assert_parity`（REQ-2
//! 統一複合判定。既存 `typed_ops_f16_parity.rs` と同方式）で突き合わ
//! せる。`MultiheadAttentionVars::new`（`pub`。`mha_backend_parity.rs`
//! doc 参照）で `LinearVars`（pub フィールド）を自前構築する——facade
//! の `Tape` は内部 `fandhe_ai_autodiff::Tape` を `pub(crate)` に
//! 保持するのみで `MultiheadAttention::bind`（crate-internal）を
//! facade テストから直接呼べないため。
//!
//! **Conv2d 側は対象外**: `nn::conv2d_forward_low_precision` が要求
//! する `Conv2dVars` は `pub weight`／`pub bias` に加え非公開の
//! `stride`／`padding`／`dilation`／`groups` を持ち、`Conv2d::bind`
//! （crate-internal。facade 非公開の `&fandhe_ai_autodiff::Tape` を
//! 要求）以外に構築する公開経路がない（`MultiheadAttentionVars::new`
//! のような直接構築コンストラクタが存在しない）。新規 `pub`
//! コンストラクタの追加は本イシューのスコープ外の判断（facade／
//! autodiff 公開面の拡張はユーザー承認事項）のため、Conv2d の実機
//! parity は本ファイルでは対象外とし、`docs/perf/logs/
//! amp-conv-mha-low-precision-2071/README.md` へ申し送る（out-of-
//! scope-tracking.md に従いスコープ外追跡）。
//!
//! 実機到達手段が本セッションにないため未実測のまま出荷する。実行
//! コマンド・判定規則・記入欄は同 README を参照。

use fandhe_ai::Device;
use fandhe_ai_autodiff::Var;
use fandhe_ai_autodiff::nn::{
    LinearVars, MultiheadAttentionVars, multihead_attention_forward_low_precision,
};
use fandhe_ai_backend_cpu::parity::assert_parity;
use fandhe_ai_tensor_core::{ScalarDType, Tensor};

/// `fandhe_ai::Tape`（newtype）・`fandhe_ai_autodiff::Tape`（生の型）の
/// いずれからも `var()` を呼べるようにする（`mha_backend_parity.rs::
/// VarSource` と同じ理由・同じ構成）。
trait VarSource {
    fn make_var(&self, tensor: &Tensor<f32>) -> Var<'_>;
}

impl VarSource for fandhe_ai::Tape {
    fn make_var(&self, tensor: &Tensor<f32>) -> Var<'_> {
        self.var(tensor)
    }
}

const B: usize = 2;
const L: usize = 3;
const E: usize = 4;
const NUM_HEADS: usize = 2;

fn leaf(seed: u64, shape: &[usize]) -> Tensor<f32> {
    let numel: usize = shape.iter().product();
    let data = bench_harness::rng::Xorshift64Star::new(seed).fill_vec(numel);
    Tensor::new(data, shape).expect("leaf: shape 一致")
}

fn contiguous_slice(t: &Tensor<f32>) -> Vec<f32> {
    t.contiguous()
        .as_slice()
        .expect("contiguous() 後は as_slice が必ず Some を返す")
        .to_vec()
}

fn mha_low_precision_forward_on(device: Device, dtype: ScalarDType) -> Tensor<f32> {
    let tape = fandhe_ai::tape_for(device).expect("実機が利用可能な前提のテストのため成功するはず");
    let q = LinearVars {
        weight: tape.make_var(&leaf(10, &[E, E])),
        bias: Some(tape.make_var(&leaf(11, &[E]))),
    };
    let k = LinearVars {
        weight: tape.make_var(&leaf(12, &[E, E])),
        bias: Some(tape.make_var(&leaf(13, &[E]))),
    };
    let v = LinearVars {
        weight: tape.make_var(&leaf(14, &[E, E])),
        bias: Some(tape.make_var(&leaf(15, &[E]))),
    };
    let out = LinearVars {
        weight: tape.make_var(&leaf(16, &[E, E])),
        bias: Some(tape.make_var(&leaf(17, &[E]))),
    };
    let vars = MultiheadAttentionVars::new(NUM_HEADS, q, k, v, out)
        .expect("test fixture: MultiheadAttentionVars::new が失敗した");
    let x = tape.make_var(&leaf(1, &[B, L, E]));
    multihead_attention_forward_low_precision(&vars, &x, &x, &x, None, false, dtype)
        .expect("低精度 MHA forward が実機で失敗した")
        .to_tensor()
}

#[test]
#[ignore = "CUDA 実機（DGX Spark GB10 等）必須"]
fn cuda_mha_low_precision_forward_f16_matches_cpu() {
    let cpu = mha_low_precision_forward_on(Device::Cpu, ScalarDType::F16);
    let cuda = mha_low_precision_forward_on(Device::Cuda(0), ScalarDType::F16);
    assert_parity(
        "multihead_attention_forward_low_precision(F16): CUDA vs CPU",
        &contiguous_slice(&cuda),
        &contiguous_slice(&cpu),
    );
}

#[test]
#[ignore = "CUDA 実機（DGX Spark GB10 等）必須"]
fn cuda_mha_low_precision_forward_bf16_matches_cpu() {
    let cpu = mha_low_precision_forward_on(Device::Cpu, ScalarDType::Bf16);
    let cuda = mha_low_precision_forward_on(Device::Cuda(0), ScalarDType::Bf16);
    assert_parity(
        "multihead_attention_forward_low_precision(Bf16): CUDA vs CPU",
        &contiguous_slice(&cuda),
        &contiguous_slice(&cpu),
    );
}

#[test]
#[ignore = "Metal 実機必須"]
#[cfg(target_os = "macos")]
fn metal_mha_low_precision_forward_f16_matches_cpu() {
    let cpu = mha_low_precision_forward_on(Device::Cpu, ScalarDType::F16);
    let metal = mha_low_precision_forward_on(Device::Metal, ScalarDType::F16);
    assert_parity(
        "multihead_attention_forward_low_precision(F16): Metal vs CPU",
        &contiguous_slice(&metal),
        &contiguous_slice(&cpu),
    );
}

#[test]
#[ignore = "Metal 実機必須"]
#[cfg(target_os = "macos")]
fn metal_mha_low_precision_forward_bf16_matches_cpu() {
    let cpu = mha_low_precision_forward_on(Device::Cpu, ScalarDType::Bf16);
    let metal = mha_low_precision_forward_on(Device::Metal, ScalarDType::Bf16);
    assert_parity(
        "multihead_attention_forward_low_precision(Bf16): Metal vs CPU",
        &contiguous_slice(&metal),
        &contiguous_slice(&cpu),
    );
}
