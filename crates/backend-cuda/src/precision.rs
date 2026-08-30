//! CUDA GEMM の TF32 Tensor Core 経路を opt-in で選択する公開スイッチ
//! （イシュー #1042。親ツリー #1029 Phase 2）。
//!
//! `CudaGemm::run_wmma_tf32`（`gemm.rs`）は GB10 実機で誤差分布を実測済み
//! （`docs/perf/cuda-tensor-core-tolerance-opt-remeasurement.md`・
//! `cuda-tensor-core-tolerance-gb10-scale-sweep.md`）だが、`ops.rs` の
//! 公開経路（`CudaBackendOps::gemm`）は既定で FP32 厳密（`run_tiled_f32`）
//! のみを使う。本モジュールはプロセスワイドな `AtomicBool` フラグで
//! TF32 経路への切り替えを制御する（candle の
//! `MM_F32_REDUCED_PRECISION` と同型の opt-in 方式）。
//!
//! **契約（REQ-2 複合判定・既定 OFF）**:
//! - 既定値は `false`（FP32 厳密）。フラグ OFF 時の `CudaBackendOps::gemm`
//!   の経路・出力は本イシュー導入前と bit-exact に不変（`ops.rs::gemm`
//!   のドキュメンテーションコメント参照）。
//! - opt-in（`true`）時も、バックエンド間数値一致は
//!   `.claude/rules/coding-rust.md` の統一複合判定「相対誤差 1e-3 未満
//!   または 絶対誤差 1e-5 未満」（TF32 前提へ改定済みの REQ-2）の範囲内で
//!   動作する。許容誤差そのものは変更しない。
//! - opt-in 時に TF32 カーネルが使用不能（cc<8.0・NVRTC コンパイル失敗等）
//!   の場合は **fail-closed**: `run_wmma_tf32` が返す
//!   `CudaError::WmmaUnavailable` をそのまま伝播し、FP32 への黙示
//!   フォールバックはしない（#994 の診断コンストラクタと同じ方針。
//!   明示 opt-in の計測条件を静かに崩さない）。
//! - 適用範囲は `CudaBackendOps::gemm`（素の f32 GEMM）のみ。
//!   `gemm_bias_act`・`gemm_resident_*`・学習経路は本イシューのスコープ外
//!   のまま FP32 で動作する（`docs/cuda-tf32-optin-api-decision.md` 参照）。
//!
//! プロセスワイドである理由: `facade` の公開 API
//! （`fandhe_ai::set_cuda_tf32_gemm_enabled`）はデバイスハンドルを介さない
//! グローバルスイッチとして設計する（`Device` 単位のインスタンスを都度
//! 引き回す設計は呼び出し側の負担が大きく、candle の前例
//! （プロセスグローバル環境変数相当の設定）を踏襲する）。`AtomicBool`
//! （`Ordering::SeqCst`。頻度が低い設定変更のため緩い順序による最適化は
//! 不要と判断）はスレッド間で安全に共有できるため、`Mutex` 等は要さない。

use std::sync::atomic::{AtomicBool, Ordering};

/// TF32 Tensor Core 経路の opt-in フラグ本体。既定 `false`（FP32 厳密）。
static TF32_GEMM_ENABLED: AtomicBool = AtomicBool::new(false);

/// CUDA GEMM（`CudaBackendOps::gemm`）の TF32 Tensor Core 経路を有効化・
/// 無効化する。プロセスワイドな設定であり、以降の全スレッド・全
/// `CudaBackendOps` インスタンスの `gemm` 呼び出しに反映される
/// （`facade::set_cuda_tf32_gemm_enabled` から委譲される。モジュール冒頭
/// コメントの契約を参照）。
pub fn set_tf32_gemm_enabled(enabled: bool) {
    TF32_GEMM_ENABLED.store(enabled, Ordering::SeqCst);
}

/// 現在の TF32 Tensor Core 経路の opt-in 状態を返す（既定 `false`）。
pub fn tf32_gemm_enabled() -> bool {
    TF32_GEMM_ENABLED.load(Ordering::SeqCst)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// フラグはプロセスグローバルのため、他のテストとの競合を避けて
    /// 直列化・原状復帰する RAII ガード（イシュー #1042 実装計画
    /// §3「テスト」節。`cargo test` の既定並列実行下でも安全に検証する）。
    struct FlagGuard {
        _lock: std::sync::MutexGuard<'static, ()>,
        original: bool,
    }

    impl FlagGuard {
        fn acquire() -> Self {
            static LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());
            // 直前のテストが panic してポイズンされていても、原状復帰の
            // ためだけに使うロックなので握り潰して継続する。
            let lock = LOCK.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
            let original = tf32_gemm_enabled();
            Self {
                _lock: lock,
                original,
            }
        }
    }

    impl Drop for FlagGuard {
        fn drop(&mut self) {
            set_tf32_gemm_enabled(self.original);
        }
    }

    #[test]
    fn default_is_disabled_when_no_prior_test_left_it_enabled() {
        let _guard = FlagGuard::acquire();
        // 既定値そのものの検証はプロセス起動直後の状態に依存するため、
        // ここでは「明示的に false へ戻した直後は false を観測できる」
        // という setter/getter の往復契約を検証する（他テストが有効化
        // したまま残す可能性があるため、真の初期値検証はしない）。
        set_tf32_gemm_enabled(false);
        assert!(!tf32_gemm_enabled());
    }

    #[test]
    fn set_true_then_false_round_trips() {
        let _guard = FlagGuard::acquire();
        set_tf32_gemm_enabled(true);
        assert!(tf32_gemm_enabled());
        set_tf32_gemm_enabled(false);
        assert!(!tf32_gemm_enabled());
    }
}
