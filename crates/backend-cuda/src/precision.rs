//! CUDA GEMM の精度モードを opt-in で選択する公開スイッチ
//! （イシュー #1042・#1355。親ツリー #1029 Phase 2・#1354）。
//!
//! `CudaGemm::run_wmma_tf32`（`gemm.rs`）は GB10 実機で誤差分布を実測済み
//! （`docs/perf/cuda-tensor-core-tolerance-opt-remeasurement.md`・
//! `cuda-tensor-core-tolerance-gb10-scale-sweep.md`）だが、`ops.rs` の
//! 公開経路（`CudaBackendOps::gemm`）は既定で FP32 厳密（`run_tiled_f32`）
//! のみを使う。本モジュールはプロセスワイドな `AtomicU8` フラグで
//! GEMM の精度モードを 3 択（[`CudaGemmPrecision`]）で切り替える
//! （candle の `MM_F32_REDUCED_PRECISION` と同型の opt-in 方式）。
//!
//! **契約（REQ-2 複合判定・既定 `Fp32Strict`）**:
//! - 既定値は [`CudaGemmPrecision::Fp32Strict`]（FP32 厳密）。フラグが
//!   `Fp32Strict` の間の `CudaBackendOps::gemm` の経路・出力は本モジュール
//!   導入前と bit-exact に不変（`ops.rs::gemm` のドキュメンテーション
//!   コメント参照）。
//! - `Tf32`／`Tf32x3` いずれの opt-in 時も、バックエンド間数値一致は
//!   `.claude/rules/coding-rust.md` の統一複合判定「相対誤差 1e-3 未満
//!   または 絶対誤差 1e-5 未満」（TF32 前提へ改定済みの REQ-2）の範囲内で
//!   動作する。許容誤差そのものは変更しない。
//! - `Tf32x3`（3×TF32・split-single 法。hi/lo 分割・3 回の `mma.sync`
//!   累積で f32 相当精度を Tensor Core 上で近似する。CUTLASS
//!   `mma_tensor_op_fast_f32` 相当）は f32 SIMT と **bit 一致しない**
//!   （`.claude/rules/coding-rust.md` の FMA 契約統一節の **例外**。
//!   ユーザー承認 2026-09-06・#1338 コメント。詳細は
//!   `docs/cuda-tf32x3-split-single-decision.md`）。
//! - opt-in 時にモード固有のカーネルが使用不能（cc<8.0・NVRTC コンパイル
//!   失敗・整列制約不成立等）の場合は **fail-closed**: 型付きエラーを
//!   そのまま伝播し、FP32 への黙示フォールバックはしない（#994 の診断
//!   コンストラクタと同じ方針。明示 opt-in の計測条件を静かに崩さない）。
//! - 適用範囲は `CudaBackendOps::gemm`（素の f32 GEMM）のみ。
//!   `gemm_bias_act`・`gemm_resident_*`・学習経路は本モジュールのスコープ
//!   外のまま FP32 で動作する（`docs/cuda-tf32-optin-api-decision.md`
//!   参照）。
//!
//! プロセスワイドである理由: `facade` の公開 API
//! （`fandhe_ai::set_cuda_tf32_gemm_enabled`／`set_cuda_gemm_precision`）は
//! デバイスハンドルを介さないグローバルスイッチとして設計する（`Device`
//! 単位のインスタンスを都度引き回す設計は呼び出し側の負担が大きく、
//! candle の前例（プロセスグローバル環境変数相当の設定）を踏襲する）。
//! `AtomicU8`（`Ordering::SeqCst`。頻度が低い設定変更のため緩い順序に
//! よる最適化は不要と判断）はスレッド間で安全に共有できるため、`Mutex`
//! 等は要さない。

use std::sync::atomic::{AtomicU8, Ordering};

/// CUDA GEMM（`CudaBackendOps::gemm`）が使う精度モード（イシュー
/// #1355。3 択）。
///
/// `#[non_exhaustive]` を付け、将来モードを追加する際に既存の
/// `match` 呼び出し元を破壊的変更なしで拡張できるようにする
/// （公開 API 非破壊はガードレール条件。CLAUDE.md「Conventions」）。
#[repr(u8)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum CudaGemmPrecision {
    /// FP32 厳密経路（`run_tiled_f32`）。既定値。
    Fp32Strict = 0,
    /// TF32 Tensor Core 単発経路（`run_wmma_tf32`。#1042）。
    Tf32 = 1,
    /// 3×TF32（split-single 法）経路（`run_tf32x3`。#1355）。
    Tf32x3 = 2,
}

impl CudaGemmPrecision {
    /// `AtomicU8` から読み取ったバイト値をモードへ変換する。未知の
    /// バイト（将来のプロセス間不整合・メモリ破損等、通常到達しない
    /// 経路）は安全側（既定 FP32 厳密）に倒す（fail-closed）。
    fn from_u8(raw: u8) -> Self {
        match raw {
            1 => Self::Tf32,
            2 => Self::Tf32x3,
            _ => Self::Fp32Strict,
        }
    }
}

/// GEMM 精度モードフラグ本体。既定 `Fp32Strict`（バイト値 0）。
static GEMM_PRECISION: AtomicU8 = AtomicU8::new(CudaGemmPrecision::Fp32Strict as u8);

/// CUDA GEMM（`CudaBackendOps::gemm`）の精度モードを設定する。
/// プロセスワイドな設定であり、以降の全スレッド・全 `CudaBackendOps`
/// インスタンスの `gemm` 呼び出しに反映される（`facade::
/// set_cuda_gemm_precision` から委譲される。モジュール冒頭コメントの
/// 契約を参照）。
pub fn set_gemm_precision(mode: CudaGemmPrecision) {
    GEMM_PRECISION.store(mode as u8, Ordering::SeqCst);
}

/// 現在の GEMM 精度モードを返す（既定 `Fp32Strict`）。
pub fn gemm_precision() -> CudaGemmPrecision {
    CudaGemmPrecision::from_u8(GEMM_PRECISION.load(Ordering::SeqCst))
}

/// `set_tf32_gemm_enabled`／`tf32_gemm_enabled`（#1042 由来）の互換
/// ラッパー。公開 API 非破壊のため既存シグネチャを維持する。
///
/// - `set_tf32_gemm_enabled(true)` は [`CudaGemmPrecision::Tf32`] へ
///   設定する。
/// - `set_tf32_gemm_enabled(false)` は **現在のモードに関わらず**
///   [`CudaGemmPrecision::Fp32Strict`] へ戻す（`Tf32x3` からの呼び出しも
///   含む）。「無効化すれば必ず FP32 厳密へ戻る」という既存契約を
///   3 モード化後も保つ。
pub fn set_tf32_gemm_enabled(enabled: bool) {
    set_gemm_precision(if enabled {
        CudaGemmPrecision::Tf32
    } else {
        CudaGemmPrecision::Fp32Strict
    });
}

/// 現在のモードが単発 TF32（[`CudaGemmPrecision::Tf32`]）のときのみ
/// `true` を返す（`Tf32x3` では `false`。既存 API 名の素直な意味
/// 「単発 TF32 が有効か」をそのまま維持し、3 モード化前の呼び出し元の
/// 意味論を変えない）。
pub fn tf32_gemm_enabled() -> bool {
    gemm_precision() == CudaGemmPrecision::Tf32
}

/// `GEMM_PRECISION` を操作するテスト間で共有する直列化ロック
/// （`cfg(test)` 限定）。
///
/// 当初は本モジュールの `tests::FlagGuard` と `ops.rs::tests::
/// Tf32FlagGuard` がそれぞれ独立した `static LOCK: Mutex<()>` を
/// 持っていたため、`cargo test` の既定並列実行下で異なるテストバイナリ
/// 内のテストが別々のロックを取得しつつ同一のプロセスグローバル
/// `GEMM_PRECISION` を書き換え合い、直列化が効かないレースが起こり
/// うる不具合があった（codex-review P2・Cursor Bugbot Medium 指摘。
/// PR #1091）。両モジュールの RAII ガードは本関数が返す単一の `Mutex`
/// を経由することで直列化を統一する。
#[cfg(test)]
pub(crate) mod test_support {
    use std::sync::Mutex;

    /// フラグ操作テスト全体で共有する単一ロックを返す。
    pub(crate) fn tf32_flag_test_lock() -> &'static Mutex<()> {
        static LOCK: Mutex<()> = Mutex::new(());
        &LOCK
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// フラグはプロセスグローバルのため、他のテストとの競合を避けて
    /// 直列化・原状復帰する RAII ガード（イシュー #1042 実装計画
    /// §3「テスト」節。`cargo test` の既定並列実行下でも安全に検証する）。
    /// ロック本体は `test_support::tf32_flag_test_lock()`（`ops.rs::tests::
    /// Tf32FlagGuard` と共有）を使う。
    struct FlagGuard {
        _lock: std::sync::MutexGuard<'static, ()>,
        original: CudaGemmPrecision,
    }

    impl FlagGuard {
        fn acquire() -> Self {
            // 直前のテストが panic してポイズンされていても、原状復帰の
            // ためだけに使うロックなので握り潰して継続する。
            let lock = test_support::tf32_flag_test_lock()
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            let original = gemm_precision();
            Self {
                _lock: lock,
                original,
            }
        }
    }

    impl Drop for FlagGuard {
        fn drop(&mut self) {
            set_gemm_precision(self.original);
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
        assert_eq!(gemm_precision(), CudaGemmPrecision::Fp32Strict);
    }

    #[test]
    fn set_true_then_false_round_trips() {
        let _guard = FlagGuard::acquire();
        set_tf32_gemm_enabled(true);
        assert!(tf32_gemm_enabled());
        set_tf32_gemm_enabled(false);
        assert!(!tf32_gemm_enabled());
    }

    #[test]
    fn set_gemm_precision_round_trips_across_all_three_modes() {
        let _guard = FlagGuard::acquire();
        for mode in [
            CudaGemmPrecision::Fp32Strict,
            CudaGemmPrecision::Tf32,
            CudaGemmPrecision::Tf32x3,
        ] {
            set_gemm_precision(mode);
            assert_eq!(gemm_precision(), mode);
        }
    }

    #[test]
    fn tf32_gemm_enabled_is_true_only_for_single_pass_tf32() {
        let _guard = FlagGuard::acquire();
        set_gemm_precision(CudaGemmPrecision::Fp32Strict);
        assert!(!tf32_gemm_enabled());
        set_gemm_precision(CudaGemmPrecision::Tf32);
        assert!(tf32_gemm_enabled());
        set_gemm_precision(CudaGemmPrecision::Tf32x3);
        assert!(
            !tf32_gemm_enabled(),
            "tf32_gemm_enabled() は Tf32x3 では false を返す契約（本モジュール冒頭コメント参照）"
        );
    }

    #[test]
    fn set_tf32_gemm_enabled_false_resets_from_tf32x3_to_fp32_strict() {
        let _guard = FlagGuard::acquire();
        set_gemm_precision(CudaGemmPrecision::Tf32x3);
        set_tf32_gemm_enabled(false);
        assert_eq!(
            gemm_precision(),
            CudaGemmPrecision::Fp32Strict,
            "set_tf32_gemm_enabled(false) はどのモードからでも Fp32Strict へ戻す契約"
        );
    }

    #[test]
    fn from_u8_maps_unknown_byte_to_fp32_strict_fail_closed() {
        assert_eq!(CudaGemmPrecision::from_u8(0), CudaGemmPrecision::Fp32Strict);
        assert_eq!(CudaGemmPrecision::from_u8(1), CudaGemmPrecision::Tf32);
        assert_eq!(CudaGemmPrecision::from_u8(2), CudaGemmPrecision::Tf32x3);
        assert_eq!(
            CudaGemmPrecision::from_u8(255),
            CudaGemmPrecision::Fp32Strict
        );
    }
}
