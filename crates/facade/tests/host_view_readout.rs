//! `VarHostView`（facade 公開面。イシュー #1335）の読み出し検証。
//!
//! CPU は既定 `fandhe_ai::tape()` 上で通常実行。Metal は実機依存の
//! `#[ignore]` テストとし、`crates/facade/tests/tape_reuse.rs` と同じ
//! `#[cfg(target_os = "macos")]` + `tape_for(Device::Metal)` 構成に
//! 揃える（`.claude/rules/coding-rust.md`「実機依存テストは `#[ignore]`
//! で分離」）。

use fandhe_ai_tensor_core::Tensor;

fn t(data: Vec<f32>, shape: &[usize]) -> Tensor<f32> {
    Tensor::new(data, shape).expect("test tensor は shape が一致する")
}

fn bits(data: &[f32]) -> Vec<u32> {
    data.iter().map(|v| v.to_bits()).collect()
}

/// facade 公開面 `fandhe_ai::tape()`（既定 CPU）上で `Var::host_view()`
/// が `to_tensor().contiguous().as_slice().to_vec()` と bit 同一である
/// ことを確認する（AC2・facade 公開面での検証）。
#[test]
fn host_view_matches_to_tensor_on_default_cpu_tape() {
    let tape = fandhe_ai::tape();
    let a = tape.var(&t(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]));
    let b = tape.var(&t(vec![5.0, 6.0, 7.0, 8.0], &[2, 2]));
    let y = a.matmul(&b).expect("matmul は shape 一致で成功する");

    let expected = y.to_tensor().contiguous().as_slice().unwrap().to_vec();
    let view = y.host_view();

    assert_eq!(
        bits(&view),
        bits(&expected),
        "facade 公開面の host_view は to_tensor() 経路と bit 同一のはず"
    );
}

/// Metal 実機（Apple Silicon）で `tape_for(Device::Metal)` 上の
/// `matmul` 結果に対し `host_view()` が `to_tensor()` 経路と bit 同一
/// であることを確認する（イシュー #1335 実装計画 §5.2）。
///
/// ```sh
/// cargo test -p fandhe-ai --test host_view_readout -- --ignored --nocapture
/// ```
#[cfg(target_os = "macos")]
#[test]
#[ignore = "Metal 実機（Apple Silicon）依存。CI では実行しない"]
fn host_view_matches_to_tensor_on_metal_tape() {
    let tape =
        fandhe_ai::tape_for(fandhe_ai::Device::Metal).expect("Metal 実機セッションでのみ実行する");
    let a = tape.var(&t(
        (0..(37 * 65)).map(|i| (i as f32) * 0.01 - 3.0).collect(),
        &[37, 65],
    ));
    let b = tape.var(&t(
        (0..(65 * 33)).map(|i| (i as f32) * 0.02 - 1.0).collect(),
        &[65, 33],
    ));
    let y = a.matmul(&b).expect("matmul は shape 一致で成功する");

    let expected = y.to_tensor().contiguous().as_slice().unwrap().to_vec();
    let view = y.host_view();

    assert_eq!(
        bits(&view),
        bits(&expected),
        "Metal 実機でも host_view は to_tensor() 経路と bit 同一のはず"
    );
}
