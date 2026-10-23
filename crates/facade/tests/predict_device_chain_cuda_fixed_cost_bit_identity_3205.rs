//! イシュー #3205: 推論チェーンの層出力バッファ再利用（`DeviceParamStore::chain_scratch`・
//! `BackendOps::linear_forward_device_into_tracked`）の CUDA 実機での bit 一致回帰。
//!
//! 再利用経路は出力バッファを memset せず前回の内容の上へ融合カーネルで書き直す。カーネルは論理形の
//! 全要素を無条件に代入する（`kernels.rs::gemm_tiled_bias_act_f32`）ため、前回の値が残らず
//! 確保経路（`alloc_zeroed`）と bit 一致するはずである。本ファイルはそれを実機で確かめる。
//!
//! 全テストは実機必須のため `#[ignore]` で分離する（`.claude/rules/ci.md`「実機依存」節）。
//!
//! ```sh
//! cargo test -p fandhe-ai --release --test predict_device_chain_cuda_fixed_cost_bit_identity_3205 -- --ignored --nocapture --test-threads=1
//! ```
//!
//! - `reuse_matches_legacy_bit_exact_on_cuda_*`: 1 回目（スクラッチ未確保）・2 回目以降（再利用）・
//!   別バッチ形状を挟んだ後（作り直し）・旧経路（`forward_resident`）の出力を `to_bits()` で全要素比較する。
//!   入力は呼び出しごとに変えて「前回の内容が残らない」ことも検査する。
//! - `predict_resident_bit_dump_cuda_3205`: 再利用経路（3 回目）の出力を `out[<i>].bits=<hex>` で印字する。
//!   before／after 2 ツリーのダンプを別バイナリ間で sha256 突合する（RULE.txt 条件 3。checksum は
//!   bit 一致の証拠にしない）。before ツリーでは再利用が無いので確保経路の出力になる。

use fandhe_ai::Device;
use fandhe_ai::compat::Sequential;
use fandhe_ai_tensor_core::Tensor;

const D_IN: usize = 784;
const D_HIDDEN: usize = 256;
const D_OUT: usize = 10;

/// 決定的な入力生成（xorshift64*。依存を増やさないための最小実装。`[-0.5, 0.5)`）。
fn fill(seed: u64, n: usize) -> Vec<f32> {
    let mut state = if seed == 0 {
        0x9E37_79B9_7F4A_7C15
    } else {
        seed
    };
    (0..n)
        .map(|_| {
            state ^= state >> 12;
            state ^= state << 25;
            state ^= state >> 27;
            let r = state.wrapping_mul(0x2545_F491_4F6C_DD1D);
            ((r >> 40) as f32) / (1u32 << 24) as f32 - 0.5
        })
        .collect()
}

fn input(seed: u64, batch: usize) -> Tensor<f32> {
    Tensor::new(fill(seed, batch * D_IN), &[batch, D_IN]).expect("shape とデータ長は一致")
}

fn bits(t: &Tensor<f32>) -> Vec<u32> {
    t.contiguous()
        .as_slice()
        .expect("contiguous 直後は as_slice が Some")
        .iter()
        .map(|v| v.to_bits())
        .collect()
}

fn model() -> Sequential {
    Sequential::new()
        .add_linear(D_IN, D_HIDDEN, 0x1111_1111)
        .unwrap()
        .add_relu()
        .add_linear(D_HIDDEN, D_OUT, 0x2222_2222)
        .unwrap()
}

/// 旧経路（`forward_resident`）の出力。
fn legacy(model: &Sequential, x: &Tensor<f32>) -> Vec<u32> {
    let init = fandhe_ai::tape_for(Device::Cuda(0)).expect("CUDA device 0 が必要");
    let mut store = model.init_device_param_store(&init).unwrap();
    drop(init);
    let tape = fandhe_ai::tape_for(Device::Cuda(0)).unwrap();
    let var = tape.var(x);
    let out = model
        .forward_resident(&tape, &var, &mut store)
        .unwrap()
        .to_tensor();
    store.abandon_pending_forward();
    bits(&out)
}

fn assert_reuse_bit_exact(batch: usize) {
    let model = model();
    let init = fandhe_ai::tape_for(Device::Cuda(0)).expect("CUDA device 0 が必要");
    let store = model.init_device_param_store(&init).unwrap();
    drop(init);

    let x1 = input(0xA1, batch);
    let x2 = input(0xB2, batch);
    let other = input(0xC3, batch / 2 + 1);

    let first_x1 = bits(&model.predict_resident(&store, &x1).unwrap());
    let reuse_x2 = bits(&model.predict_resident(&store, &x2).unwrap());
    let reuse_x1 = bits(&model.predict_resident(&store, &x1).unwrap());
    assert_eq!(
        first_x1, reuse_x1,
        "再利用経路が 1 回目（確保経路）と bit 不一致"
    );
    assert_ne!(
        first_x1, reuse_x2,
        "入力が違うのに出力が同一（前回の内容が残っている疑い）"
    );

    // 別バッチ形状を挟んでスクラッチが作り直されても bit 一致する。
    let _ = model.predict_resident(&store, &other).unwrap();
    let rebuilt_x2 = bits(&model.predict_resident(&store, &x2).unwrap());
    assert_eq!(reuse_x2, rebuilt_x2, "作り直し後の出力が bit 不一致");
    let reused_again = bits(&model.predict_resident(&store, &x2).unwrap());
    assert_eq!(reuse_x2, reused_again, "run-to-run で bit 不一致");

    // 旧経路との bit 一致。
    assert_eq!(
        first_x1,
        legacy(&model, &x1),
        "旧経路（forward_resident）と bit 不一致 x1"
    );
    assert_eq!(
        reuse_x2,
        legacy(&model, &x2),
        "旧経路（forward_resident）と bit 不一致 x2"
    );
}

#[test]
#[ignore = "CUDA 実機（DGX Spark GB10 等）必須"]
fn reuse_matches_legacy_bit_exact_on_cuda_batch_64() {
    assert_reuse_bit_exact(64);
}

#[test]
#[ignore = "CUDA 実機（DGX Spark GB10 等）必須"]
fn reuse_matches_legacy_bit_exact_on_cuda_batch_1024() {
    assert_reuse_bit_exact(1024);
}

#[test]
#[ignore = "CUDA 実機（DGX Spark GB10 等）必須"]
fn reuse_matches_legacy_bit_exact_on_cuda_batch_4096() {
    assert_reuse_bit_exact(4096);
}

/// 別バイナリ間 bit 比較用ダンプ（RULE.txt 条件 3）。3 回目の呼び出し（after では再利用経路）の出力。
#[test]
#[ignore = "CUDA 実機（DGX Spark GB10 等）必須"]
fn predict_resident_bit_dump_cuda_3205() {
    let model = model();
    let init = fandhe_ai::tape_for(Device::Cuda(0)).expect("CUDA device 0 が必要");
    let store = model.init_device_param_store(&init).unwrap();
    drop(init);
    let x = input(0xD4, 64);
    let other = input(0xE5, 64);
    let _ = model.predict_resident(&store, &other).unwrap();
    let _ = model.predict_resident(&store, &x).unwrap();
    let out = model.predict_resident(&store, &x).unwrap();
    for (i, b) in bits(&out).into_iter().enumerate() {
        println!("out[{i}].bits={b:#010x}");
    }
}
