//! コマンドバッファ共有バッチの純粋ロジック（イシュー #1017・
//! `docs/backend-metal-command-batching-design.md`）。
//!
//! `context.rs::MetalContext::encode`／`flush`／`synchronize` が積む
//! 「1 コマンドバッファに複数 dispatch をまとめ、ホスト実体化まで
//! `waitUntilCompleted` を遅延する」状態遷移のうち、`objc2-metal` の
//! FFI 型に触れない部分（ラベル記録・自動 flush 判定・失敗伝播・
//! 診断メッセージ整形）をここへ切り出す。`pad`／`tile`／`row_kernel` と
//! 同じ設計判断（モジュール冒頭コメント参照）で `cfg(target_os =
//! "macos")` を付けず、Linux（本実装環境・CI）でも単体テストが回る
//! ようにする。macOS ビルドでは `context.rs` からのみ呼ばれるため
//! `pub(crate)` に留める。
//!
//! `#![cfg_attr(not(target_os = "macos"), allow(dead_code))]`
//! （`generic_cache.rs`／`row_kernel.rs` と同じ理由）: 呼び出し元
//! `context.rs` 自体が `cfg(target_os = "macos")` 限定のため、非 macOS
//! ビルド（`cargo build`／`cargo clippy` の非テストパス）では本モジュール
//! の各項目が「クレート内から到達不能」と判定され dead_code lint が
//! 誤検知する。

#![cfg_attr(not(target_os = "macos"), allow(dead_code))]

use fandhe_ai_tensor_core::DispatchFailureCell;
use fandhe_ai_tensor_core::device::BackendError;

/// 1 コマンドバッファに積む dispatch 数の上限（安全弁）。到達したら
/// 呼び出し元（`context.rs::MetalContext::encode`）が自動 `flush`
/// （commit のみ・待たない）する。学習ループのパラメータ数（数個〜
/// 数十個程度）に対し十分大きく、通常経路では到達しない想定だが、
/// コマンドバッファの無制限な肥大化を防ぐための上限として置く
/// （設計文書 §3.6「安全弁」）。
pub(crate) const MAX_DISPATCHES_PER_BATCH: usize = 256;

/// 開いているバッチの診断メタデータ（ラベル列と dispatch 数）。
///
/// `objc2-metal` の型に触れない純粋なカウンタ・ラベル保持であり、
/// `context.rs::Batch` がこれをフィールドとして持つ。
#[derive(Debug, Default)]
pub(crate) struct BatchMeta {
    labels: Vec<&'static str>,
}

impl BatchMeta {
    pub(crate) fn new() -> Self {
        Self::default()
    }

    /// 1 dispatch 分のラベルを記録する（`context.rs::MetalContext::encode`
    /// が呼び出し元から受け取ったラベルをそのまま渡す）。
    pub(crate) fn record_dispatch(&mut self, label: &'static str) {
        self.labels.push(label);
    }

    /// これまでに記録した dispatch 数。
    pub(crate) fn dispatch_count(&self) -> usize {
        self.labels.len()
    }

    /// [`MAX_DISPATCHES_PER_BATCH`] へ到達したか（自動 flush 判定）。
    pub(crate) fn should_auto_flush(&self) -> bool {
        self.dispatch_count() >= MAX_DISPATCHES_PER_BATCH
    }

    /// 記録済みラベル列（[`format_failure_message`] へ渡す診断用）。
    pub(crate) fn labels(&self) -> &[&'static str] {
        &self.labels
    }
}

/// `waitUntilCompleted()` 後にコマンドバッファの `status` が
/// `MTLCommandBufferStatus::Error` だった場合の診断メッセージを、
/// バッチに含まれていた dispatch のラベル列とあわせて整形する。
///
/// 1 個のコマンドバッファに複数 dispatch（例: GEMM の呼び出し中に別
/// スレッドの SGD dispatch が偶然同居していた等）が積まれていた場合、
/// 元の `NSError::localizedDescription`（`message`）だけでは「どの
/// dispatch が原因か・巻き込まれたか」が呼び出し元に伝わらない
/// （`.claude/rules/security.md` A08。エラーを握り潰さず追跡可能にする
/// 方針）。
pub(crate) fn format_failure_message(labels: &[&'static str], message: &str) -> String {
    if labels.is_empty() {
        return format!("Metal command buffer completed with an error: {message}");
    }
    format!(
        "Metal command buffer completed with an error while executing [{}]: {message}",
        labels.join(", ")
    )
}

/// バッチ実行時エラーを、そのバッチに登録済みの全 [`DispatchFailureCell`]
/// （`context.rs::Batch::tokens`）へ伝播する。
///
/// `BackendError` は `#[non_exhaustive]` な多様な variant を持つため
/// `Clone` を導出しない（`crate::dispatch_failure` モジュールコメント
/// 参照）。トークンごとに同一の診断メッセージを持つ
/// [`BackendError::KernelLaunchFailed`] を新規構築して `set` する
/// （`DispatchFailureCell::set` は first-writer-wins のため、既に別の
/// エラーが登録済みのトークンは上書きされない）。
pub(crate) fn propagate_failure(tokens: &[DispatchFailureCell], message: &str) {
    for token in tokens {
        token.set(BackendError::KernelLaunchFailed(message.to_string()));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn batch_meta_tracks_dispatch_count_and_labels() {
        let mut meta = BatchMeta::new();
        assert_eq!(meta.dispatch_count(), 0);
        meta.record_dispatch("sgd_step_f32");
        meta.record_dispatch("elementwise_add");
        assert_eq!(meta.dispatch_count(), 2);
        assert_eq!(meta.labels(), &["sgd_step_f32", "elementwise_add"]);
    }

    #[test]
    fn batch_meta_auto_flush_triggers_at_limit() {
        let mut meta = BatchMeta::new();
        for _ in 0..MAX_DISPATCHES_PER_BATCH - 1 {
            meta.record_dispatch("sgd_step_f32");
            assert!(!meta.should_auto_flush());
        }
        meta.record_dispatch("sgd_step_f32");
        assert!(meta.should_auto_flush());
    }

    #[test]
    fn format_failure_message_includes_labels_in_order() {
        let msg = format_failure_message(&["sgd_step_f32", "gemm_tiled"], "GPU fault");
        assert!(msg.contains("sgd_step_f32, gemm_tiled"));
        assert!(msg.contains("GPU fault"));
    }

    #[test]
    fn format_failure_message_handles_empty_labels() {
        let msg = format_failure_message(&[], "GPU fault");
        assert_eq!(
            msg,
            "Metal command buffer completed with an error: GPU fault"
        );
    }

    #[test]
    fn propagate_failure_sets_all_tokens_first_writer_wins() {
        let t1 = DispatchFailureCell::new();
        let t2 = DispatchFailureCell::new();
        // t1 は既に別のエラーが登録済み（poison 検出等の先行経路を模擬）。
        t1.set(BackendError::KernelLaunchFailed(
            "earlier error".to_string(),
        ));

        propagate_failure(&[t1.clone(), t2.clone()], "batch failure");

        match t1.take() {
            Some(BackendError::KernelLaunchFailed(msg)) => assert_eq!(msg, "earlier error"),
            other => panic!("t1 should keep the first error, got {other:?}"),
        }
        match t2.take() {
            Some(BackendError::KernelLaunchFailed(msg)) => assert_eq!(msg, "batch failure"),
            other => panic!("t2 should receive the propagated error, got {other:?}"),
        }
    }

    #[test]
    fn propagate_failure_on_empty_tokens_is_noop() {
        propagate_failure(&[], "unreachable");
    }
}
