//! イシュー #1647: `shaders/rnn_cell.metal` に REQ-8 境界検査・決定性
//! （`atomic` 系不使用）が実在することを機械検査する証跡テスト。
//!
//! `mse_source_evidence.rs` と同じ位置づけ: `include_str!` によるビルド
//! 時文字列埋め込みへの contains 検査のみで完結するため、Metal 実機・
//! `cfg(target_os = "macos")` を必要とせず Linux CI 上でも green になる
//! （Linux 上での唯一の CUDA 側 `kernels_rnn_cell.rs::tests` に対応する
//! 証跡）。

/// `crates/backend-metal/src/shaders/rnn_cell.metal` のソース全文。
const RNN_CELL_METAL_SOURCE: &str = include_str!("../src/shaders/rnn_cell.metal");

/// REQ-8: 5 カーネルすべてが `if (idx < numel)` の手動境界チェックを
/// 持つことをロックする。
#[test]
fn rnn_cell_metal_source_has_bound_checks_for_all_five_kernels() {
    // 冒頭コメントにも同じ文字列が 1 回登場するため、実コード側の
    // 5 箇所（カーネル本体）と合わせて 6 回以上であることを検査する
    // （`>=` ではなく厳密な期待値を保つため、冒頭コメントの 1 回分を
    // 明示的に加算する）。
    let occurrences = RNN_CELL_METAL_SOURCE.matches("if (idx < numel)").count();
    let expected_kernel_occurrences = 5;
    let comment_occurrences = 1;
    assert_eq!(
        occurrences,
        expected_kernel_occurrences + comment_occurrences,
        "5 カーネル分の `if (idx < numel)` 手動境界チェック（+ 冒頭コメント 1 回）が \
         見つかりません（実際: {occurrences} 箇所）"
    );
}

/// 決定性: pointwise カーネルはブロック間の縮約を持たないため
/// `atomic` 系命令（`atomic_fetch_add` 等）を一切使わない
/// （`kernels_rnn_cell.rs` の CUDA 側証跡と同じ契約）。
#[test]
fn rnn_cell_metal_source_uses_no_atomics() {
    assert!(
        !RNN_CELL_METAL_SOURCE.to_lowercase().contains("atomic"),
        "rnn_cell.metal に atomic 系命令が含まれています（決定性契約違反）"
    );
}

/// FMA 契約統一: 積が和へ流れる箇所（`c = f*c_prev + i*g` 等）は
/// `fma(...)` を用いる。
#[test]
fn rnn_cell_metal_source_uses_fma() {
    assert!(
        RNN_CELL_METAL_SOURCE.contains("fma("),
        "rnn_cell.metal に FMA 契約統一の `fma(...)` 呼び出しが見つかりません"
    );
}

/// 意味論の正（precise math 方針）: `metal::precise::exp`／
/// `metal::precise::tanh` を明示使用する（`shaders/elementwise.metal`
/// 冒頭コメント「意味論の正」と同じ方針）。
#[test]
fn rnn_cell_metal_source_uses_precise_math() {
    assert!(
        RNN_CELL_METAL_SOURCE.contains("metal::precise::exp"),
        "rnn_cell.metal に `metal::precise::exp` が見つかりません"
    );
    assert!(
        RNN_CELL_METAL_SOURCE.contains("metal::precise::tanh"),
        "rnn_cell.metal に `metal::precise::tanh` が見つかりません"
    );
}

/// 5 カーネルすべてのエントリポイント名が実在することをロックする
/// （`rnn_cell.rs::MetalRnnCell::new` が `pipeline::make_pipeline` へ
/// 渡す関数名と 1:1 対応）。
#[test]
fn rnn_cell_metal_source_declares_all_five_kernel_entry_points() {
    for name in [
        "kernel void lstm_pointwise_f32",
        "kernel void lstm_hidden_backward_f32",
        "kernel void lstm_cell_backward_f32",
        "kernel void gru_pointwise_f32",
        "kernel void gru_backward_f32",
    ] {
        assert!(
            RNN_CELL_METAL_SOURCE.contains(name),
            "rnn_cell.metal にカーネルエントリポイント `{name}` が見つかりません"
        );
    }
}
