//! `nn::EmbeddingBag`（イシュー #2161・親 #2131）の 3 バックエンド
//! 受け入れ条件対応テスト（`dropout_variants_backend_parity.rs`と同型）。
//!
//! **facade `Tape` newtype を経由しない理由**（`infer_device_chain_
//! metal.rs` 冒頭 doc と同型のパターン）: `EmbeddingBag::bind` は
//! `&fandhe_ai_autodiff::Tape`（生の型）を要求するが、facade `Tape`
//! （newtype。内部フィールドは `pub(crate)` でクレート外から取り出せ
//! ない）はこれを渡せない。`EmbeddingBagVars` も `mode`／`padding_idx`
//! が非公開のため `bind` を経由しない直接構築もできない。このため
//! 本ファイルは `fandhe_ai_autodiff::Tape::new_with_ops(Box::new(<各
//! バックエンドの BackendOps>::new(..)))` で facade の `tape()`／
//! `tape_for()` と同じバックエンド結線を独立に再現する。
//!
//! Sum／Mean は `BackendOps::sum` 経由（REQ-2 の統一複合判定
//! `assert_parity` で判定）、Max は forward が bit 一致する
//! （`BackendOps::max` は縮約順序に依存しない要素ごと比較のため）。
//! backward はいずれのモードも REQ-2 の統一複合判定で判定する
//! （`sum`／`mean`／`max` の VJP は縮約軸を持つため）。

use std::sync::Mutex;

use fandhe_ai_autodiff::Tape;
use fandhe_ai_autodiff::nn::{EmbeddingBag, EmbeddingBagMode};
use fandhe_ai_backend_cpu::CpuBackendOps;
use fandhe_ai_backend_cpu::parity::assert_parity;
use fandhe_ai_tensor_core::Tensor;

fn test_lock() -> &'static Mutex<()> {
    static LOCK: Mutex<()> = Mutex::new(());
    &LOCK
}

fn cpu_tape() -> Tape {
    Tape::new_with_ops(Box::new(CpuBackendOps::new()))
}

fn contiguous_slice(t: &Tensor<f32>) -> Vec<f32> {
    t.contiguous()
        .as_slice()
        .expect("contiguous() 後は as_slice が必ず Some を返す")
        .to_vec()
}

fn weight(num_embeddings: usize, embedding_dim: usize) -> Tensor<f32> {
    let data: Vec<f32> = (0..num_embeddings * embedding_dim)
        .map(|v| (v as f32) * 0.1 - 1.0)
        .collect();
    Tensor::new(data, &[num_embeddings, embedding_dim]).expect("shape 一致")
}

fn run_forward(mode: EmbeddingBagMode) -> (Vec<f32>, Vec<f32>) {
    let w = weight(6, 4);
    let ids = Tensor::<i32>::new(vec![0, 1, 2, 3, 4, 5], &[2, 3]).expect("shape 一致");
    let bag = EmbeddingBag::from_parameters(w, mode, None).expect("rank 2・num_embeddings > 0");

    let cpu = cpu_tape();
    let cpu_out = bag
        .bind(&cpu)
        .forward(&ids)
        .expect("正常な rank 2 ids")
        .to_tensor();

    let naive = Tape::new();
    let naive_out = bag
        .bind(&naive)
        .forward(&ids)
        .expect("正常な rank 2 ids")
        .to_tensor();

    (contiguous_slice(&cpu_out), contiguous_slice(&naive_out))
}

#[test]
fn cpu_embedding_bag_sum_forward_matches_naive_reference() {
    let _guard = test_lock().lock().unwrap_or_else(|p| p.into_inner());
    let (cpu, naive) = run_forward(EmbeddingBagMode::Sum);
    assert_parity("embedding_bag(sum) forward: CPU vs NaiveOps", &cpu, &naive);
}

#[test]
fn cpu_embedding_bag_mean_forward_matches_naive_reference() {
    let _guard = test_lock().lock().unwrap_or_else(|p| p.into_inner());
    let (cpu, naive) = run_forward(EmbeddingBagMode::Mean);
    assert_parity("embedding_bag(mean) forward: CPU vs NaiveOps", &cpu, &naive);
}

#[test]
fn cpu_embedding_bag_max_forward_matches_naive_reference_bit_exact() {
    let _guard = test_lock().lock().unwrap_or_else(|p| p.into_inner());
    let (cpu, naive) = run_forward(EmbeddingBagMode::Max);
    assert_parity("embedding_bag(max) forward: CPU vs NaiveOps", &cpu, &naive);
    assert_eq!(
        cpu, naive,
        "embedding_bag(max) forward: 要素ごと最大値比較は丸めを伴わないため bit 同一のはず"
    );
}

#[test]
fn cpu_embedding_bag_sum_backward_matches_naive_reference() {
    let _guard = test_lock().lock().unwrap_or_else(|p| p.into_inner());
    let w = weight(4, 3);
    let ids = Tensor::<i32>::new(vec![0, 1, 2, 0], &[2, 2]).expect("shape 一致");

    let cpu = cpu_tape();
    let cpu_bag = EmbeddingBag::from_parameters(w.clone(), EmbeddingBagMode::Sum, None)
        .expect("有効な weight");
    let cpu_vars = cpu_bag.bind(&cpu);
    let cpu_out = cpu_vars.forward(&ids).expect("正常な rank 2 ids");
    let cpu_loss = cpu_out.sum(None).expect("全軸縮約は失敗しない");
    let cpu_grads = cpu
        .backward(&cpu_loss)
        .expect("weight は requires_grad の葉");
    let cpu_dw = cpu_grads
        .get(&cpu_vars.weight)
        .expect("weight は requires_grad=true の葉")
        .expect("weight は loss に到達する");

    let naive = Tape::new();
    let naive_bag =
        EmbeddingBag::from_parameters(w, EmbeddingBagMode::Sum, None).expect("有効な weight");
    let naive_vars = naive_bag.bind(&naive);
    let naive_out = naive_vars.forward(&ids).expect("正常な rank 2 ids");
    let naive_loss = naive_out.sum(None).expect("全軸縮約は失敗しない");
    let naive_grads = naive
        .backward(&naive_loss)
        .expect("weight は requires_grad の葉");
    let naive_dw = naive_grads
        .get(&naive_vars.weight)
        .expect("weight は requires_grad=true の葉")
        .expect("weight は loss に到達する");

    let cpu_slice = contiguous_slice(cpu_dw);
    let naive_slice = contiguous_slice(naive_dw);
    assert_parity(
        "embedding_bag(sum) backward: CPU vs NaiveOps",
        &cpu_slice,
        &naive_slice,
    );
}

// --- 実機横断（`#[ignore]`。Metal／CUDA） ---------------------------------

#[cfg(target_os = "macos")]
#[test]
#[ignore = "Metal 実機（Apple Silicon）依存。CI では実行しない"]
fn metal_embedding_bag_sum_forward_matches_cpu() {
    use fandhe_ai_backend_metal::MetalBackendOps;

    let _guard = test_lock().lock().unwrap_or_else(|p| p.into_inner());
    let w = weight(6, 4);
    let ids = Tensor::<i32>::new(vec![0, 1, 2, 3, 4, 5], &[2, 3]).expect("shape 一致");
    let bag = EmbeddingBag::from_parameters(w, EmbeddingBagMode::Sum, None).expect("有効な weight");

    let metal = Tape::new_with_ops(Box::new(MetalBackendOps::new()));
    let metal_out = bag
        .bind(&metal)
        .forward(&ids)
        .expect("正常な rank 2 ids")
        .to_tensor();
    let cpu_out = bag
        .bind(&cpu_tape())
        .forward(&ids)
        .expect("正常な rank 2 ids")
        .to_tensor();

    let metal_slice = contiguous_slice(&metal_out);
    let cpu_slice = contiguous_slice(&cpu_out);
    assert_parity(
        "embedding_bag(sum) forward: Metal vs CPU",
        &metal_slice,
        &cpu_slice,
    );
}

#[test]
#[ignore = "CUDA 実機（DGX Spark GB10）依存。CI では実行しない"]
fn cuda_embedding_bag_sum_forward_matches_cpu() {
    use fandhe_ai_backend_cuda::CudaBackendOps;

    let _guard = test_lock().lock().unwrap_or_else(|p| p.into_inner());
    let w = weight(6, 4);
    let ids = Tensor::<i32>::new(vec![0, 1, 2, 3, 4, 5], &[2, 3]).expect("shape 一致");
    let bag = EmbeddingBag::from_parameters(w, EmbeddingBagMode::Sum, None).expect("有効な weight");

    let cuda = Tape::new_with_ops(Box::new(CudaBackendOps::new(0)));
    let cuda_out = bag
        .bind(&cuda)
        .forward(&ids)
        .expect("正常な rank 2 ids")
        .to_tensor();
    let cpu_out = bag
        .bind(&cpu_tape())
        .forward(&ids)
        .expect("正常な rank 2 ids")
        .to_tensor();

    let cuda_slice = contiguous_slice(&cuda_out);
    let cpu_slice = contiguous_slice(&cpu_out);
    assert_parity(
        "embedding_bag(sum) forward: CUDA vs CPU",
        &cuda_slice,
        &cpu_slice,
    );
}
