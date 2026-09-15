//! facade 経由の `Tensor<T>` 値表示（`Debug`／`Display`。イシュー #1754）
//! の到達性・view 一致・打ち切りの統合テスト。
//!
//! `fandhe_ai::Tensor` は `crates/facade/src/lib.rs` の
//! `pub use fandhe_ai_tensor_core::Tensor` 経由の再エクスポートのみで、
//! 表示実装自体は `tensor-core::tensor_fmt`（非公開モジュール）にある。
//! 本テストは facade のみを import して型検査・実行することで、
//! 「facade 利用者が `{}`／`{:?}` をそのまま使える」ことを機械的に
//! 固定する（新規公開アイテムは増えないため `api_surface.rs` の
//! 到達性テストとは別に、facade 経由の実行結果を直接検証する）。

use fandhe_ai::Tensor;

#[test]
fn display_and_debug_reachable_via_facade() {
    let t = Tensor::new(vec![1.0f32, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
    assert_eq!(format!("{}", t), "tensor([[1, 2], [3, 4]])");
    let dbg = format!("{:?}", t);
    assert!(dbg.starts_with("Tensor {"), "{dbg}");
    // `Debug` は要素側も `T::fmt`（`Debug`）へ委譲するため `f32` は
    // `1.0` 表記になる（`Display` の `1` とは異なる。§3.2 参照）。
    assert!(dbg.contains("data: [[1.0, 2.0], [3.0, 4.0]]"), "{dbg}");
}

#[test]
fn view_transpose_display_matches_contiguous_via_facade() {
    let t = Tensor::new(vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]).unwrap();
    let tr = t.transpose_2d().unwrap();
    assert!(!tr.is_contiguous());
    assert_eq!(format!("{}", tr), format!("{}", tr.contiguous()));
}

#[test]
fn truncation_bounds_output_for_large_tensor_via_facade() {
    // `Tape: Debug`（`docs/public-api-design.md` §7 の公開契約）越しに
    // 巨大テンソルがダンプされても出力サイズが有界であることを
    // facade 経由でも確認する（DoS 耐性。`.claude/rules/security.md`
    // A04 観点）。
    let t: Tensor<f32> = Tensor::zeros(&[200, 200]).unwrap();
    let s = format!("{}", t);
    assert!(s.len() < 2000, "output too long: {} bytes", s.len());
    assert!(s.contains("..."), "{s}");
}
