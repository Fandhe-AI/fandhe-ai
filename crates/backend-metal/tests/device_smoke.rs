//! `backend-metal` のデバイス・コマンドキュー・バッファ基盤（TASK-1.8a・#38）に対する
//! 実機スモークテスト。
//!
//! macOS 実機（Apple Silicon）でのみコンパイル・実行する。CI（self-hosted・Linux）では
//! `#![cfg(target_os = "macos")]` によりコンパイル対象外になり、`#[ignore]` により通常の
//! `cargo test` からも除外される（実機依存テストの分離。`.claude/rules/coding-rust.md`）。
//! 実行するには macOS 実機で以下を叩く:
//!
//! ```sh
//! cargo test -p fandhe-ai-backend-metal -- --ignored --nocapture
//! ```
//!
//! 実行手順・テスト一覧の正本は `docs/backend-metal-real-device-testing.md`
//! （TASK-1.8e・#42）を参照する。

#![cfg(target_os = "macos")]

use fandhe_ai_backend_metal::{MetalBuffer, MetalContext, MetalError};

/// デバイス初期化 → キュー生成 → バッファ確保 → データ書き込み → readback が一致することを
/// 確認する（受け入れ条件「macOS でデバイス初期化・バッファ確保が動作する」の直接検証）。
#[test]
#[ignore = "Metal 実機（Apple Silicon）依存。CI では実行しない"]
fn device_and_buffer_roundtrip() {
    let ctx = MetalContext::new().expect("Metal デバイス・コマンドキューの初期化に失敗した");

    let data = vec![1.0f32, 2.0, 3.0, 4.5, -6.25];
    let buf = MetalBuffer::new_with_data(&ctx, &data).expect("データ入りバッファの確保に失敗した");
    assert_eq!(buf.len(), data.len());
    assert_eq!(buf.read_to_vec(), data);

    let zeroed = MetalBuffer::new_zeroed(&ctx, 16).expect("ゼロ初期化バッファの確保に失敗した");
    assert_eq!(zeroed.len(), 16);
    assert_eq!(zeroed.read_to_vec(), vec![0.0f32; 16]);
}

/// 長さ 0 のバッファ確保が FFI 呼び出し前に型付きエラーで拒否されることを確認する
/// （OWASP A03 観点。`.claude/rules/security.md`）。
#[test]
#[ignore = "Metal 実機（Apple Silicon）依存。CI では実行しない"]
fn zero_length_allocation_is_rejected() {
    let ctx = MetalContext::new().expect("Metal デバイス・コマンドキューの初期化に失敗した");

    let empty: Vec<f32> = Vec::new();
    let err = MetalBuffer::new_with_data(&ctx, &empty).unwrap_err();
    assert!(matches!(err, MetalError::ZeroLengthAllocation));

    let err = MetalBuffer::new_zeroed(&ctx, 0).unwrap_err();
    assert!(matches!(err, MetalError::ZeroLengthAllocation));
}

/// 要素数 × `size_of::<f32>()` が `usize` の範囲でオーバーフローする場合に型付きエラーで
/// 拒否されることを確認する（OWASP A03 観点）。
#[test]
#[ignore = "Metal 実機（Apple Silicon）依存。CI では実行しない"]
fn allocation_size_overflow_is_rejected() {
    let ctx = MetalContext::new().expect("Metal デバイス・コマンドキューの初期化に失敗した");

    let huge_len = usize::MAX / 2;
    let err = MetalBuffer::new_zeroed(&ctx, huge_len).unwrap_err();
    assert!(matches!(
        err,
        MetalError::AllocationSizeOverflow { len } if len == huge_len
    ));
}
