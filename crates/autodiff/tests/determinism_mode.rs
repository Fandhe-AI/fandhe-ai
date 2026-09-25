//! `fandhe_ai_autodiff::determinism`（イシュー #2157・親 #2131）の
//! end-to-end 統合テスト。
//!
//! グローバル `AtomicBool` 状態を扱うため、本バイナリでは**単一
//! `#[test]` 関数に全アサーションを集約**する（同一テストバイナリ内の
//! `#[test]` は既定で並列実行されるため、複数関数に分けると他テストの
//! `set_deterministic` 呼び出しと競合しうる。`crates/autodiff/src/
//! determinism.rs` のインライン単体テストは `Mutex` で直列化している
//! が、本ファイルは既定 `false` の確認を含むため単一関数化がより単純
//! で確実）。
//!
//! 検証内容:
//! 1. 既定値が `false` であること。
//! 2. `set_deterministic(true)` で `true` になり、2 回目の呼び出しも
//!    冪等であること。
//! 3. `set_deterministic(false)` で `false` に戻ること。
//! 4. no-op 契約（`docs/autodiff-determinism-mode-design.md` §0・§3）
//!    の直接検証: `Tape::new()` 上で MLP 相当の forward
//!    （matmul → relu → mse_loss）・backward を実行し、決定性モード
//!    ON／OFF で出力値・勾配が bit 完全一致すること。

use fandhe_ai_autodiff::Tape;
use fandhe_ai_autodiff::determinism::{is_deterministic, set_deterministic};
use fandhe_ai_tensor_core::Tensor;

fn t(data: Vec<f32>, shape: &[usize]) -> Tensor<f32> {
    Tensor::new(data, shape).expect("test fixture: shape とデータ長は事前に一致させている")
}

/// `Tape::new()` 上で matmul → relu → mse_loss の forward・backward を
/// 実行し、`x` に対する勾配・loss 値を `(f32 の to_bits, Vec<u32>)` で
/// 返す（呼び出し元が決定性モード ON／OFF の 2 回分を bit 比較する）。
fn run_mlp_like_graph() -> (u32, Vec<u32>) {
    let tape = Tape::new();

    let x = tape.var(&t(vec![1.0, -2.0, 3.0, -4.0, 5.0, -6.0], &[2, 3]));
    let w = tape.var(&t(
        vec![0.5, -0.25, 0.125, 1.0, -1.0, 0.75, 0.25, -0.5, 1.5],
        &[3, 3],
    ));
    let target = tape.var(&t(vec![0.0; 6], &[2, 3]));

    let hidden = x.matmul(&w).expect("shape 適合済みの matmul");
    let activated = hidden.relu();
    let loss = activated
        .mse_loss(&target)
        .expect("shape 適合済みの mse_loss");

    let loss_bits = loss
        .to_tensor()
        .get(&[])
        .expect("mse_loss はスカラー")
        .to_bits();

    let grads = tape.backward(&loss).expect("backward は成功する");
    let dx = grads
        .get(&x)
        .expect("backward は成功する")
        .expect("x は loss へ到達する");
    let dx_bits: Vec<u32> = dx
        .as_slice()
        .expect("x は contiguous")
        .iter()
        .map(|v| v.to_bits())
        .collect();

    (loss_bits, dx_bits)
}

#[test]
fn set_deterministic_default_toggle_and_noop_contract() {
    // 1. 既定値は `false`。
    assert!(!is_deterministic(), "既定値は false のはず");

    // 4a. 決定性モード OFF での forward/backward 結果を確定する。
    let (loss_off, dx_off) = run_mlp_like_graph();

    // 2. `set_deterministic(true)` で `true` になり、2 回目の呼び出しも
    //    冪等。
    set_deterministic(true);
    assert!(is_deterministic(), "true 設定後は true のはず");
    set_deterministic(true);
    assert!(is_deterministic(), "2 回目の true 設定も冪等のはず");

    // 4b. 決定性モード ON での forward/backward 結果（no-op 契約により
    //     OFF と bit 完全一致するはず）。
    let (loss_on, dx_on) = run_mlp_like_graph();
    assert_eq!(
        loss_off, loss_on,
        "決定性モード ON/OFF で loss が bit 一致しない（no-op 契約違反）"
    );
    assert_eq!(
        dx_off, dx_on,
        "決定性モード ON/OFF で勾配が bit 一致しない（no-op 契約違反）"
    );

    // 3. `set_deterministic(false)` で `false` に戻る。
    set_deterministic(false);
    assert!(!is_deterministic(), "false 設定後は false のはず");
}
