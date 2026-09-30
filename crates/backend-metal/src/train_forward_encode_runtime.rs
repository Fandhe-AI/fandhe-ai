//! Metal の train forward（`BackendOps::gemm_resident_rhs_act`）を
//! encode-only 合流へ切り替える opt-in スイッチ（イシュー #2113・
//! #1980 §17.4 の施策 2）。
//!
//! # 役割
//!
//! `DeviceParamStore::linear_forward_with_activation`（autodiff。train reuse
//! forward の入口）は Metal では `gemm_resident_rhs_act` の既定合成
//! （`gemm_resident_rhs` → `self.relu`）を通り、`act == Relu` で 2 回
//! `waitUntilCompleted` する。本フラグが ON のとき `ops.rs` の Metal
//! オーバーライドは gemm と relu を同一コマンドバッファへ encode-only で積み
//! 同期を 1 回へ合流する（カーネル・入力は不変で bit 同一。encode 総数も不変）。
//!
//! # 契約
//!
//! - **既定は OFF**（[`TRAIN_FORWARD_ENCODE_ONLY_DEFAULT_ENABLED`] が
//!   `false`）。M4 Max・GB10 の両機体 A/B が ADOPT になるまで既定を変えない
//!   （別 PR で定数のみ切り替える。`docs/perf/logs/
//!   metal-train-forward-encodeonly-2113/RULE.txt`）。
//! - フラグは**プロセスワイド**（`AtomicBool`・`SeqCst`。`split_k_runtime.rs`
//!   と同型）。facade の公開面・環境変数による切替は設けない。
//! - A/B（`scripts/bench/framework-compare/run_ab_train_forward_encode_metal.sh`）
//!   は after 用の計測専用 worktree でこの定数だけを `true` に書き換えて行う
//!   （main へはコミットしない）。スクリプトが定数を sed で読むため、宣言の
//!   書式（`pub(crate) const ... : bool = <true|false>;` の 1 行）を変えない。
//!
//! `cfg(target_os = "macos")` を付けず Linux でも常時コンパイルし、既定値の
//! ドリフト検出テストを CI で実行する（`split_k_runtime.rs` と同じ判断）。

use std::sync::atomic::{AtomicBool, Ordering};

/// encode-only 合流の既定値の単一情報源（本モジュール冒頭「契約」参照）。
pub(crate) const TRAIN_FORWARD_ENCODE_ONLY_DEFAULT_ENABLED: bool = false;

static TRAIN_FORWARD_ENCODE_ONLY: AtomicBool =
    AtomicBool::new(TRAIN_FORWARD_ENCODE_ONLY_DEFAULT_ENABLED);

/// 現在の状態を返す（`ops.rs::gemm_resident_rhs_act` が呼び出しごとに読む）。
pub(crate) fn train_forward_encode_only_enabled() -> bool {
    TRAIN_FORWARD_ENCODE_ONLY.load(Ordering::SeqCst)
}

/// テスト・A/B 診断専用の切替入口（`lib.rs` から `#[doc(hidden)]` で再公開）。
/// プロセスワイドのため、並列実行されるテストは呼び出し側で直列化すること。
pub(crate) fn set_train_forward_encode_only_enabled(enabled: bool) {
    TRAIN_FORWARD_ENCODE_ONLY.store(enabled, Ordering::SeqCst);
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{Mutex, MutexGuard};

    fn lock() -> MutexGuard<'static, ()> {
        static LOCK: Mutex<()> = Mutex::new(());
        LOCK.lock().unwrap_or_else(|p| p.into_inner())
    }

    /// 既定値が `false`（opt-in・既定 OFF）であることを固定するドリフト検出。
    #[test]
    fn default_is_off() {
        assert!(
            !std::hint::black_box(TRAIN_FORWARD_ENCODE_ONLY_DEFAULT_ENABLED),
            "既定は OFF（両機体 A/B が ADOPT になるまで true にしない。イシュー #2113）"
        );
    }

    #[test]
    fn set_and_get_round_trip() {
        let _g = lock();
        let original = train_forward_encode_only_enabled();
        set_train_forward_encode_only_enabled(true);
        assert!(train_forward_encode_only_enabled());
        set_train_forward_encode_only_enabled(false);
        assert!(!train_forward_encode_only_enabled());
        set_train_forward_encode_only_enabled(original);
    }
}
