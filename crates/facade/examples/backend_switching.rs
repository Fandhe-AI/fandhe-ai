//! `fandhe_ai::tape_for(Device)` によるバックエンド切替の最小例（イシュー #874）。
//!
//! バックエンド切替は feature フラグなしの cfg ベース（PoC-v2-5 実証
//! 構成。`.claude/rules/coding-rust.md`）で、`Device::Cuda(_)`／
//! `Device::Metal` は実行時にドライバ・デバイスの存在検証を行い、
//! 不在なら `BackendError` を返す fail-fast 設計（`crates/facade/src/
//! lib.rs` モジュール doc「`Device::Cuda(_)`／`Device::Metal` の構築規則」
//! 参照）。GitHub ホステッド CI（`ubuntu-latest`。実機 CUDA/Metal 非搭載）
//! でも `cargo run --example` が成功するよう、`Device::Cuda(0)` が
//! 失敗した場合は既定バックエンド（`Device::Cpu`。常に利用可能）へ
//! フォールバックして続行する。
//!
//! この「失敗したら CPU へフォールバックする」処理は本 example 側の
//! 選択であり、`fandhe_ai::tape_for` 自体は自動デバイス選択を行わない
//! （`Device::available()` 相当の集約入口は facade のスコープ外。
//! `crates/facade/src/lib.rs` 参照）。

use fandhe_ai::{Device, tape_for};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let tape = match tape_for(Device::Cuda(0)) {
        Ok(tape) => {
            println!("connected to Device::Cuda(0)");
            tape
        }
        Err(err) => {
            // driver 不在・範囲外 ordinal 等は fail-fast で `BackendError`
            // が返る（`panic!`/`unwrap()` しない。`.claude/rules/coding-rust.md`）。
            // ここでは CPU へフォールバックして example の実行を継続する。
            println!("Device::Cuda(0) unavailable ({err}); falling back to Device::Cpu");
            tape_for(Device::Cpu)?
        }
    };

    let input = tape.var(&fandhe_ai::Tensor::new(
        vec![1.0_f32, 2.0, 3.0, 4.0],
        &[1, 4],
    )?);
    let loss = input.sum(None)?;
    let grads = tape.backward(&loss)?;
    // 入力は loss に直接寄与しているため勾配が必ず存在するはずだが、本番経路で
    // `unwrap()`/`expect()` を使わない方針（`.claude/rules/coding-rust.md`）に
    // 合わせ `?` で型付きエラーとして伝播する。
    let input_grad = grads
        .get(&input)?
        .ok_or("input has no gradient after backward")?;

    println!("input grad shape: {:?}", input_grad.shape());
    Ok(())
}
