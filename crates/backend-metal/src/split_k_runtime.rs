//! `MetalGemm::dispatch_auto` の split-K 2 パス経路（イシュー #1516 で
//! 本番結線・#1544 で既定有効化）を実行時に無効化する opt-out スイッチ
//! （イシュー #1545）。
//!
//! # 背景・2 段ゲートの関係
//!
//! split-K 経路への到達は次の 2 段すべてが揃って初めて成立する
//! （`gemm.rs::dispatch_auto_with_route_impl` の分岐条件を参照。かつては
//! コンパイル時定数 `tile::SPLIT_K_DISPATCH_AUTO_PRODUCTION_ENABLED` を
//! 含む 3 段ゲートだったが、#1547 で当該定数とドリフト検出テストを撤去し
//! 本モジュールの実行時トグルへ一本化した）。
//!
//! 1. `MetalGemm::split_k_auto_enabled`（インスタンス単位フィールド。
//!    `MetalGemm::new` は常に `true` 固定で渡す。`new_with_split_k_auto`
//!    で個別インスタンスごとに明示 `true`／`false` を指定できる A/B
//!    診断用の入口）。
//! 2. 本モジュールの [`split_k_enabled`]（**プロセスワイドな実行時
//!    フラグ**。イシュー #1545 で新設）。
//!
//! 上記 1. はコンパイル時・構築時に固定される値であり、実行中の
//! プロセスから split-K を一時的に無効化する手段がなかった
//! （`docs/backend-metal-splitk-parity-judgment-decision.md` の baseline
//! 非後退方式は split-K 到達形状で classic 経路と bit 一致しないため、
//! 運用上「今だけ classic に固定したい」場面〈例: candle 比ゲート再計測
//! での切り分け〉に対応できない）。本モジュールはその間隙を埋める
//! 実行時トグルであり、`crates/backend-cuda/src/precision.rs`
//! （`AtomicU8` によるプロセスワイド精度モード切替）と同型の設計を
//! `AtomicBool` で踏襲する。
//!
//! # 契約
//!
//! - **既定値は `true`**（#1544 の本番既定と同一。フラグ導入前後で
//!   デフォルト挙動は完全に不変）。
//! - `false` の間、`dispatch_auto`（`gemm.rs::dispatch_auto_with_route_impl`
//!   の分岐条件に本フラグが AND で加わる）は 1. の値に関わらず常に
//!   classic 経路（`GemmRoute::Classic`）へ固定され、結線前
//!   （`fandhe-ai =0.8.0` 相当）と bit 同一の出力を返す（fail-closed に
//!   「安全な既知の経路」へ倒す設計。split-K 側で問題が起きても `false`
//!   にすれば必ず classic へ戻せる）。
//! - `true` に戻すと `split_k_auto_enabled` の値で決まる従来どおりの
//!   経路選択に戻る（本フラグが `true` であること自体は split-K 到達を
//!   保証しない。1. が `true` かつ `tile::should_split_k` が対象形状と
//!   判定した場合のみ split-K に到達する）。
//! - **プロセスワイド**（`Device` 単位ではない）。`context_cache::
//!   cached_gemm` が保持する `MetalGemm` シングルトンはシェーダの
//!   実行時コンパイルを伴う重い構築コストを持つため、本フラグの
//!   切り替えのために `MetalGemm` を作り直すことはしない
//!   （`split_k_auto_enabled` フィールド自体は不変のまま、呼び出しの
//!   都度本フラグを読み取ることで対応する）。
//! - `AtomicBool`（`Ordering::SeqCst`。頻度が低い設定変更のため緩い
//!   順序による最適化は不要と判断。`precision.rs` の `AtomicU8` と同じ
//!   判断）でスレッド間で安全に共有できるため `Mutex` 等は要さない。
//!
//! `facade::set_metal_split_k_gemm_enabled`／`metal_split_k_gemm_enabled`
//! から委譲される（composition root。`docs/compat-api-scope.md` §0）。

use std::sync::atomic::{AtomicBool, Ordering};

/// split-K 実行時トグル本体。既定 `true`（#1544 の本番既定と同一）。
static SPLIT_K_RUNTIME_ENABLED: AtomicBool = AtomicBool::new(true);

/// split-K 2 パス経路への実行時分岐を有効化・無効化する。
/// `false` にすると `dispatch_auto` は常に classic 経路へ固定される
/// （本モジュール冒頭コメントの契約参照）。
pub fn set_split_k_enabled(enabled: bool) {
    SPLIT_K_RUNTIME_ENABLED.store(enabled, Ordering::SeqCst);
}

/// 現在の実行時トグル状態を返す（既定 `true`）。
pub fn split_k_enabled() -> bool {
    SPLIT_K_RUNTIME_ENABLED.load(Ordering::SeqCst)
}

/// `SPLIT_K_RUNTIME_ENABLED` を操作するテスト間で共有する直列化ロック
/// （`cfg(test)` 限定）。`precision.rs::test_support` と同じ理由:
/// 複数のテストファイル（＝複数のテストバイナリ）が同一のプロセス
/// グローバルフラグを書き換え合うと `cargo test` の既定並列実行下で
/// レースしうるため、単一の `Mutex` を経由して直列化する。
#[cfg(test)]
pub(crate) mod test_support {
    use std::sync::Mutex;

    /// フラグ操作テスト全体で共有する単一ロックを返す。
    pub(crate) fn split_k_runtime_test_lock() -> &'static Mutex<()> {
        static LOCK: Mutex<()> = Mutex::new(());
        &LOCK
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// フラグはプロセスグローバルのため、他のテストとの競合を避けて
    /// 直列化・原状復帰する RAII ガード（`precision.rs::tests::FlagGuard`
    /// と同型）。
    pub(crate) struct FlagGuard {
        _lock: std::sync::MutexGuard<'static, ()>,
        original: bool,
    }

    impl FlagGuard {
        pub(crate) fn acquire() -> Self {
            // 直前のテストが panic してポイズンされていても、原状復帰の
            // ためだけに使うロックなので握り潰して継続する。
            let lock = test_support::split_k_runtime_test_lock()
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            let original = split_k_enabled();
            Self {
                _lock: lock,
                original,
            }
        }
    }

    impl Drop for FlagGuard {
        fn drop(&mut self) {
            set_split_k_enabled(self.original);
        }
    }

    #[test]
    fn default_is_enabled_when_no_prior_test_left_it_disabled() {
        let _guard = FlagGuard::acquire();
        // 既定値そのものの検証はプロセス起動直後の状態に依存するため、
        // ここでは「明示的に true へ戻した直後は true を観測できる」
        // という setter/getter の往復契約を検証する（他テストが無効化
        // したまま残す可能性があるため、真の初期値検証はしない。
        // `precision.rs::tests::default_is_disabled_when_no_prior_test_
        // left_it_enabled` と同じ設計判断）。
        set_split_k_enabled(true);
        assert!(split_k_enabled());
    }

    #[test]
    fn set_true_then_false_round_trips() {
        let _guard = FlagGuard::acquire();
        set_split_k_enabled(true);
        assert!(split_k_enabled());
        set_split_k_enabled(false);
        assert!(!split_k_enabled());
        set_split_k_enabled(true);
        assert!(split_k_enabled());
    }

    #[test]
    fn disabling_then_restoring_round_trips_via_guard() {
        // 初期値の取得から復元確認までを同一のロック区間内（`guard` が
        // 生存している間）に収める。当初は `_guard` をブロックで早期
        // drop し、ロック解放後に外側で読んだ `outer_original` と比較して
        // いたが、その両読み取り（ロック取得前の初期値読み取り・ロック
        // 解放後の復元確認）はいずれも本ロックの保護区間外であり、
        // 並列実行される他テスト（例: `set_true_then_false_round_trips`）
        // がその隙間でフラグを書き換えると比較が偽陰性・偽陽性になり
        // うる欠陥だった（PR #1546 codex-review P2 指摘）。`guard` を
        // スコープ末尾まで生存させることで、初期値の捕捉・書き換え・
        // 復元操作の検証をすべてロック保持中に行う。
        let guard = FlagGuard::acquire();
        let original = guard.original;
        set_split_k_enabled(false);
        assert!(!split_k_enabled());
        // `Drop for FlagGuard` が行う復元処理（`set_split_k_enabled(self.
        // original)`）と同じ操作をロックを保持したまま直接検証する
        // （`guard` をここで drop してロック解放後に確認すると上記と同じ
        // 欠陥が再発するため、drop を待たずに検証する）。
        set_split_k_enabled(original);
        assert_eq!(split_k_enabled(), original);
        // スコープ末尾で `guard` が drop され、同じ復元処理が冪等に
        // もう一度走る（ロック保持中に既に検証済みのため実害なし）。
    }
}
