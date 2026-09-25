//! 決定性モード（イシュー #2157・親 #2131）。PyTorch
//! `torch.use_deterministic_algorithms` に相当する、プロセスワイドな
//! opt-in 状態を管理する。
//!
//! **facade 非公開（意図的）**: `crate::matrix_ops`／`crate::reduce_ops`
//! モジュール doc と同じ理由・同じ判断枠組みによる。`Var` は facade
//! （`fandhe_ai` クレート）から直接再エクスポートされるため、`Var` への
//! inherent メソッド追加は即座に facade 公開面へ出てしまう。本モジュール
//! は `Var` を経由しない自由関数のみで構成し、facade 公開（crate ルート
//! `fandhe_ai::set_deterministic`／`fandhe_ai::is_deterministic`）は
//! 承認事項として保留する（`docs/autodiff-determinism-mode-design.md`
//! §6）。承認後は facade 側の保留ガード（`crates/facade/src/lib.rs::
//! DeterminismHoldDoctestGuard`）を撤去し、薄い委譲関数を追加する。
//!
//! **no-op 契約**（`docs/autodiff-determinism-mode-design.md` §0・§3。
//! 必読）: `crates/backend-cpu` の棚卸し結果、`Tape::new()`（`NaiveOps`）
//! ・`Tape::new_with_ops(Box::new(CpuBackendOps::new()))` いずれの本番
//! 経路にも、結果がスレッド数・実行順に依存する非決定的な縮約は
//! 見つからなかった（rayon 縮約は固定チャンク分割・チャンク番号順の
//! 逐次結合、または出力要素ごとの排他書き込み。atomic 蓄積は本番
//! 未結線）。そのため [`set_deterministic`] は状態を記録するのみで、
//! 実行時の分岐・拒否は一切行わない。呼び出し元のない
//! `AutodiffError` variant・使われないヘルパーは追加しない
//! （`.claude/rules/coding-rust.md` の `#[allow(dead_code)]` 濫用禁止
//! 方針）。
//!
//! **保証範囲**: CPU バックエンド（`Tape::new()`・`Tape::new_with_ops`
//! への `CpuBackendOps` 注入）に限る。CUDA／Metal の `Tape` に対しても
//! 本モジュールの関数は等しく no-op で、GPU 経路の決定性は本イシューの
//! 棚卸し・保証の対象外（未検証。`docs/autodiff-determinism-mode-
//! design.md` §5・§7）。複数スレッドから並行にグローバル RNG
//! （`crate::eval` が経由する `fandhe_ai_tensor_core::rng`）を消費
//! する順序も対象外（既存の RNG 契約どおり保証しない）。
//!
//! 将来 CPU・GPU いずれかに非決定的な経路が追加された場合の規則は
//! `docs/autodiff-determinism-mode-design.md` §3.3・§7 を参照
//! （`AutodiffError` への `#[non_exhaustive]` variant 追加・決定性
//! モード中の拒否）。

use std::sync::atomic::{AtomicBool, Ordering};

/// [`set_deterministic`]／[`is_deterministic`] が読み書きする
/// プロセスワイドの決定性モード状態（既定 `false`）。`SeqCst`。
/// facade `set_cuda_gemm_precision` 系の `AtomicBool` 先例
/// （`crates/facade/src/interop/onnx.rs::CUDA_ONNX_GPU_EXEC` 等）と
/// 同型のグローバル opt-in 状態パターンを踏襲する。
static DETERMINISTIC: AtomicBool = AtomicBool::new(false);

/// 決定性モードを設定する（既定 `false`）。
///
/// 本クレートの no-op 契約（モジュール doc 参照）により、現時点では
/// 状態を記録するのみで実行時の分岐・拒否は行わない。`Tape::new()`・
/// `Tape::new_with_ops(Box::new(CpuBackendOps::new()))` の forward・
/// backward は `true`／`false` いずれでも bit 完全一致する
/// （`crates/autodiff/tests/determinism_mode.rs` で固定）。
pub fn set_deterministic(enabled: bool) {
    DETERMINISTIC.store(enabled, Ordering::SeqCst);
}

/// 現在の決定性モード状態を返す（既定 `false`）。
pub fn is_deterministic() -> bool {
    DETERMINISTIC.load(Ordering::SeqCst)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;

    // グローバル `AtomicBool` を直接触るテストのため、他の並列テストと
    // の競合を避けるため単一プロセス内で直列化する（同一クレート内
    // `#[test]` はデフォルトで並列実行されるため）。
    static SERIAL: Mutex<()> = Mutex::new(());

    /// 既定値 `false` → `set(true)` で `true` → 2 回目の `set(true)` も
    /// 冪等 → `set(false)` で `false` に戻ることを確認する
    /// （`AtomicBool` の単純な read-your-writes 契約の直接検証）。
    #[test]
    fn set_deterministic_round_trips_and_is_idempotent() {
        let _guard = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
        set_deterministic(false);
        assert!(!is_deterministic());

        set_deterministic(true);
        assert!(is_deterministic());

        // 2 回目の `true` も冪等（副作用のない store の繰り返し）。
        set_deterministic(true);
        assert!(is_deterministic());

        set_deterministic(false);
        assert!(!is_deterministic());
    }
}
