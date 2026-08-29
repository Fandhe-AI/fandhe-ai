//! バックエンド跨ぎで共有する遅延失敗トークン（イシュー #1017・
//! `docs/backend-metal-command-batching-design.md` §3.7）。
//!
//! Metal のコマンドバッファ共有バッチ（`backend-metal::context::
//! MetalContext::encode`／`synchronize`）は複数 dispatch を 1 個の
//! コマンドバッファへまとめ、GPU 側の実行時エラー（fault・OOM・
//! discarded work）は `waitUntilCompleted` 後の状態検査でしか判明しない。
//! エラーが判明する時点（ホスト実体化: `download`／`zero_fill`／`Drop`）
//! と、その原因となった dispatch を発行した呼び出し元
//! （`fandhe_ai_autodiff::optim::device_store::DeviceParamStore`）が
//! 別の呼び出しであるため、dispatch 発行時に本セルを**同一ロック区間で**
//! 登録しておき、バッチ実行完了検査時にそこへ書き込むことで、呼び出し元
//! が次回のエントリ検査（`DeviceParamStore::check_not_poisoned`）で
//! 確実に検出できるようにする（登録漏れが起こる「encode 後に別 API で
//! 登録する」方式は採用しない。設計文書 §3.7 (2)）。
//!
//! [`crate::backend_ops::BackendOps::sgd_step_device_tracked`]
//! （デフォルトメソッド。非破壊拡張）の追加引数として渡す。実際に
//! 登録・検査するのは Metal 実装（`backend-metal::ops::
//! MetalBackendOps::sgd_step_device_tracked`）のみで、CPU／CUDA は
//! 同期実行のため実行時エラーが即座に判明し本セルを使わない。

use std::sync::{Arc, Mutex};

use crate::device::BackendError;

/// dispatch 発行時に登録し、バッチ実行完了検査時に書き込む共有セル。
///
/// `Arc<Mutex<Option<BackendError>>>` の薄いラッパー。`clone()` した
/// 全インスタンスが同一の内部状態を共有する（複数の dispatch 呼び出し
/// 元が同一バッチへ登録する経路を想定）。
///
/// `BackendError` は `#[non_exhaustive]` な多様な variant を持つため
/// `Clone` を導出しない。そのため [`Self::set`] は
/// **first-writer-wins**（既に値が設定済みなら上書きしない）とする:
/// 最初に検出された実行時エラーを保持し続ける方が、後続の伝播処理
/// （`backend-metal::batch_state::propagate_failure` が同一バッチの
/// 全トークンへ同じ診断メッセージを書き込む等）で情報が失われない。
///
/// `Mutex` の poison（他スレッドの panic）は「既にエラーが立っている」
/// 扱いに倒す fail-closed（`.claude/rules/security.md` A08。縮退運転で
/// 実行時エラーを見逃さない）。
#[derive(Clone, Debug, Default)]
pub struct DispatchFailureCell {
    inner: Arc<Mutex<Option<BackendError>>>,
}

impl DispatchFailureCell {
    /// 未設定状態の新規セルを構築する。
    pub fn new() -> Self {
        Self::default()
    }

    /// まだ値が設定されていなければ `err` を設定する
    /// （first-writer-wins。モジュール冒頭コメント参照）。
    pub fn set(&self, err: BackendError) {
        let mut guard = match self.inner.lock() {
            Ok(guard) => guard,
            // poison 時も書き込みは試みる（fail-closed の意図は「消えない」
            // ことであり、`is_set`/`take` 側で確実に検出できればよい）。
            Err(poisoned) => poisoned.into_inner(),
        };
        if guard.is_none() {
            *guard = Some(err);
        }
    }

    /// エラーが設定済みか判定する（`Mutex` poison も「設定済み」扱いの
    /// fail-closed）。
    pub fn is_set(&self) -> bool {
        match self.inner.lock() {
            Ok(guard) => guard.is_some(),
            Err(_) => true,
        }
    }

    /// 設定済みのエラーを取り出す（消費。以後 [`Self::is_set`] は
    /// `false` に戻る）。`Mutex` poison 時は `into_inner()` で内部値を
    /// そのまま取り出す（poison 前に既に `set` 済みであれば `Some` を
    /// 返し、`set` される前に他スレッドが panic した到達しにくい経路
    /// では `None` を返す。呼び出し元は `is_set()` を先に見て poison を
    /// 検出する契約であり、`take()` 自体は panic させないことのみを
    /// 保証する）。**poison 自体は `take()` 後も解除されない**ため
    /// （`Mutex::lock()` は同一 `Mutex` が poison している限り恒久的に
    /// `Err` を返す標準ライブラリの仕様）、`None` を取り出した場合でも
    /// 以後 [`Self::is_set`] は `true` のまま固定される
    /// （fail-closed。「エラーは無いが恒久的に設定済み扱い」という
    /// 安全側の縮退であり、値を取りこぼしたことにはならない）。
    pub fn take(&self) -> Option<BackendError> {
        let mut guard = match self.inner.lock() {
            Ok(guard) => guard,
            Err(poisoned) => poisoned.into_inner(),
        };
        guard.take()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::panic;
    use std::sync::Arc as StdArc;

    fn err(msg: &str) -> BackendError {
        BackendError::KernelLaunchFailed(msg.to_string())
    }

    #[test]
    fn new_cell_is_unset() {
        let cell = DispatchFailureCell::new();
        assert!(!cell.is_set());
        assert!(cell.take().is_none());
    }

    #[test]
    fn set_then_is_set_and_take() {
        let cell = DispatchFailureCell::new();
        cell.set(err("boom"));
        assert!(cell.is_set());
        match cell.take() {
            Some(BackendError::KernelLaunchFailed(msg)) => assert_eq!(msg, "boom"),
            other => panic!("unexpected: {other:?}"),
        }
        // take() は消費するため以後は未設定に戻る。
        assert!(!cell.is_set());
    }

    #[test]
    fn set_is_first_writer_wins() {
        let cell = DispatchFailureCell::new();
        cell.set(err("first"));
        cell.set(err("second"));
        match cell.take() {
            Some(BackendError::KernelLaunchFailed(msg)) => assert_eq!(msg, "first"),
            other => panic!("unexpected: {other:?}"),
        }
    }

    #[test]
    fn clone_shares_underlying_state() {
        let cell = DispatchFailureCell::new();
        let cloned = cell.clone();
        cell.set(err("shared"));
        assert!(cloned.is_set());
        match cloned.take() {
            Some(BackendError::KernelLaunchFailed(msg)) => assert_eq!(msg, "shared"),
            other => panic!("unexpected: {other:?}"),
        }
    }

    /// `Mutex` poison（他スレッドが臨界区間内で panic）後も
    /// `is_set`/`take` が fail-closed（既にエラーが立っている扱い）に
    /// 倒れることを検証する（モジュール冒頭コメント「A08」参照）。
    #[test]
    fn poison_is_treated_as_set_fail_closed() {
        let cell = DispatchFailureCell::new();
        let shared: StdArc<Mutex<Option<BackendError>>> = StdArc::clone(&cell.inner);

        let result = panic::catch_unwind(panic::AssertUnwindSafe(|| {
            let _guard = shared.lock().unwrap();
            panic!("simulated poison while holding the lock");
        }));
        assert!(result.is_err());

        assert!(cell.is_set(), "poisoned mutex must be treated as set");
        // poison 後も take() は panic せず、合成値を含め None を返さない
        // か、あるいは実際に格納された値を返す（今回は書き込み前に panic
        // したため None の可能性があるが、poison 自体は panic しない）。
        let _ = cell.take();
    }
}
