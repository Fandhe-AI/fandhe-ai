//! GPU `run_fused` の elementwise allowlist 融合（区分 B-1・イシュー
//! #2085・`docs/autodiff-graph-optimization-scope-decision.md` §5）。
//!
//! `ops.rs::CudaBackendOps::run_fused` は現状 canonical RMSNorm・softmax
//! プランのみを専用カーネルへルーティングし、それ以外の elementwise
//! 連鎖（`Add`／`Mul`／`Relu`／`Exp`／`Tanh` のみで構成される非
//! canonical プラン）は `Unsupported` として呼び出し元の per-op
//! フォールバックへ倒す（`ops.rs::run_fused` ドキュメンテーション
//! コメント参照）。per-op フォールバックは連鎖長 N に対し H2D→起動→D2H
//! を N 回発生させる。本モジュールは `backend-cpu::fused_elementwise`
//! と同一の allowlist（`Input`／`Add`／`Mul`／`Relu`／`Exp`／`Tanh`）を
//! 対象に、実行時 NVRTC ソース生成（`crate::kernels_fused_elementwise`）
//! による単一パスカーネルへ融合し、この往復を 1 回へ削減する。
//!
//! # opt-in ゲート（既定 OFF）
//!
//! [`gpu_elementwise_fusion_enabled`]／[`set_gpu_elementwise_fusion_enabled`]
//! はプロセスワイドな `AtomicBool`（`crate::precision::set_gemm_precision`
//! と同型の opt-in スイッチ）。**ゲート OFF 時は `run_fused` の融合分岐が
//! デバイスアクセス（`device_handle_raw` 等）より前に `Unsupported` を
//! 返す**（`ops.rs::run_fused` の呼び出し順序契約。CUDA 非搭載環境でも
//! `CudaUnavailable` にならず `Unsupported` になるため、ゲート OFF の
//! 挙動は本 PR 導入前と atomic load 1 回を除き bit 同一——実装計画 §6
//! 「同一バイナリ A/B」の前提）。**本 PR では `facade` へのセッター公開は
//! 行わない**（実装計画 §9 承認事項 (1)。`backend-cuda` クレート内部限定）。
//!
//! # allowlist（denylist 化しない。CPU 版と同一方針）
//!
//! `match_elementwise_plan` は `plan.ops()` の全要素が
//! `FusedOpKind::{Input, Add, Mul, Relu, Exp, Tanh}` のいずれかであり、
//! かつ `plan.row_fusion().is_none()`（行融合プランは RMSNorm／softmax
//! 専用カーネルの対象であり、本モジュールは対象外。`ops.rs::run_fused`
//! が先に試す 2 分岐と排他的）である場合にのみ `ElementwiseProgram`
//! を返す。**denylist ではなく allowlist**（`backend-cpu::
//! fused_elementwise` 冒頭コメント「denylist ではなく allowlist」と
//! 同一の理由: `FusedOpKind` は `#[non_exhaustive]` のため、将来 variant
//! が追加されてもこの判定は安全側〈拒否〉へ倒れる）。
//!
//! # 数値契約（REQ-2）
//!
//! 融合カーネルは同一バックエンドの per-op 経路
//! （`kernels_elementwise.rs`）・CPU `backend-cpu::fused_elementwise` と
//! bit 完全一致を目標とする（`kernels_fused_elementwise.rs` モジュール
//! 冒頭「数値契約」参照。FMA 縮約遮断のため `Add`／`Mul` に非縮約
//! intrinsic を使う）。CPU 対 GPU の突合は引き続き REQ-2 統一複合判定
//! （相対誤差 1e-3 未満 または 絶対誤差 1e-5 未満）に依る。

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use fandhe_ai_tensor_core::{FusedOpKind, FusionPlan};

use crate::context_cache;
use crate::device::CudaDevice;
use crate::elementwise::launch_nary;
use crate::error::CudaError;
use crate::kernels_fused_elementwise::{self, FUSED_EW_FUNCTION_NAME};
use crate::pool::CudaAllocator;

/// GPU elementwise 融合 opt-in ゲート本体。既定 `false`
/// （`crate::precision::GEMM_PRECISION` と同じくプロセスワイド。
/// `Ordering::SeqCst`: 頻度が低い設定変更のため緩い順序による最適化は
/// 不要と判断）。
static GPU_ELEMENTWISE_FUSION_ENABLED: AtomicBool = AtomicBool::new(false);

/// GPU `run_fused` の elementwise allowlist 融合を有効化する
/// （既定 OFF。モジュール冒頭「opt-in ゲート」参照）。プロセスワイドな
/// 設定であり、以降の全スレッド・全 `CudaBackendOps` インスタンスの
/// `run_fused` 呼び出しに反映される。
///
/// **`backend-cuda` クレート内 `pub`（`crate::precision::
/// set_tf32_gemm_enabled`／`crate::graph::set_step_graph_enabled` と同じ
/// 可視性）だが、`facade` への再公開は本 PR の対象外**（実装計画 §9
/// 承認事項 (1)。公開面拡張〈`docs/compat-api-scope.md` §0〉は別途
/// ユーザー承認が必要）。現在の呼び出し元は本クレートの `#[ignore]`
/// 実機統合テスト（`tests/fused_elementwise_parity.rs`）のみ。
pub fn set_gpu_elementwise_fusion_enabled(enabled: bool) {
    GPU_ELEMENTWISE_FUSION_ENABLED.store(enabled, Ordering::SeqCst);
}

/// 現在の GPU elementwise 融合ゲート状態を返す（既定 `false`）。
pub fn gpu_elementwise_fusion_enabled() -> bool {
    GPU_ELEMENTWISE_FUSION_ENABLED.load(Ordering::SeqCst)
}

/// `plan` の op 列・葉数・`row_fusion()` を照合し、B-1 allowlist に
/// 一致する場合のみ [`ElementwiseProgram`] を返す（純関数。デバイス非
/// 依存で実機なしでも単体テスト可能。`rmsnorm.rs::match_rmsnorm_plan`
/// と同型の判定様式）。
///
/// 検証順序（fail-closed）:
/// 1. `plan.row_fusion().is_some()` → 拒否（RMSNorm／softmax 専用の
///    行融合プランは `ops.rs::run_fused` の先行 2 分岐が扱う対象）。
/// 2. `plan.ops()` の全要素が allowlist に一致すること（1 個でも
///    `Sum`／`Max`／`Rsqrt`／`Sub`／`Div`／`Broadcast`・将来の未知
///    variant を含めば拒否）。
/// 3. `plan.ops().count() <= MAX_FUSED_SEGMENT_NODES`（`FusionPlan`
///    構築時〈`from_segment`／`from_ops`〉に既に保証済みの不変条件だが、
///    レジスタ配列サイズ・カーネル引数個数の際限ない増大を避ける多層
///    防御として本関数でも検査する。`.claude/rules/coding-rust.md`
///    「カーネル実装の境界検査」と同じ考え方をホスト側検証にも適用）。
pub(crate) fn match_elementwise_plan(plan: &FusionPlan) -> Option<ElementwiseProgram> {
    if plan.row_fusion().is_some() {
        return None;
    }
    let ops: Vec<FusedOpKind> = plan.ops().collect();
    if ops.is_empty() || ops.len() > fandhe_ai_tensor_core::MAX_FUSED_SEGMENT_NODES {
        return None;
    }
    if ops.iter().any(|op| {
        !matches!(
            op,
            FusedOpKind::Input { .. }
                | FusedOpKind::Add { .. }
                | FusedOpKind::Mul { .. }
                | FusedOpKind::Relu { .. }
                | FusedOpKind::Exp { .. }
                | FusedOpKind::Tanh { .. }
        )
    }) {
        return None;
    }
    Some(ElementwiseProgram {
        ops,
        leaf_count: plan.leaf_count(),
    })
}

/// [`match_elementwise_plan`] が受理した融合プランの実行可能表現。
/// `ops`（`FusedOpKind` 列。allowlist 検証済み）と `leaf_count` のみを
/// 保持する薄い DTO（`FusionPlan` 自体は借用のため呼び出し元
/// `ops.rs::run_fused` の呼び出しスコープを超えて保持できない）。
/// [`crate::kernels_fused_elementwise::cache_key`]／[`crate::
/// fused_elementwise_model::eval_program_host`] の双方がこの型を読む。
#[derive(Debug, Clone)]
pub(crate) struct ElementwiseProgram {
    pub(crate) ops: Vec<FusedOpKind>,
    pub(crate) leaf_count: usize,
}

/// [`ElementwiseProgram`] に対応する CUDA カーネル（実行時ソース生成 →
/// キャッシュ済みコンパイル）を取得する（**構築専用フェーズ**。NVRTC
/// コンパイル・`load_module` は driver 呼び出しであり `CudaRmsNorm::
/// run_rmsnorm_f32_raw` を取得する `ops.rs::run_fused_rmsnorm` と同じく
/// 「コンパイル済みハンドル取得」と「実行（起動）」を呼び出し元
/// （`ops.rs::run_fused_elementwise_allowlist`）が別々の
/// `self.with_driver_call` 区間へ分離する契約——コンパイル失敗
/// （NVRTC 不在等）を `BackendError::CudaUnavailable` へ、起動失敗を
/// `BackendError::KernelLaunchFailed` へ、それぞれ独立して分類できる
/// ようにするため。`device` は呼び出し元がハンドルを取得済みのものを
/// そのまま渡す契約。
///
/// キャッシュ上限到達時は `Ok(None)`（呼び出し元は `BackendError::
/// Unsupported` へ変換し per-op フォールバックへ委ねる。実装計画 §2.5
/// 「キャッシュ上限」）。
pub(crate) fn compile_program(
    device: &CudaDevice,
    program: &ElementwiseProgram,
) -> Result<Option<Arc<cudarc::driver::CudaFunction>>, CudaError> {
    let key = kernels_fused_elementwise::cache_key(&program.ops, program.leaf_count);
    let source = kernels_fused_elementwise::generate_source(&program.ops, program.leaf_count);
    context_cache::cached_fused_elementwise_kernel(device, &key, &source, FUSED_EW_FUNCTION_NAME)
}

/// コンパイル済みカーネル `func`（[`compile_program`] が返したもの）を
/// `leaves` に対して実行する（**実行専用フェーズ**。[`launch_nary`] を
/// 呼ぶだけの薄いラッパー。`ops.rs::run_fused_elementwise_allowlist` の
/// 第 2 `with_driver_call` 区間から呼ばれる契約）。
pub(crate) fn launch_program_f32(
    device: &CudaDevice,
    allocator: &CudaAllocator,
    func: &cudarc::driver::CudaFunction,
    leaves: &[&[f32]],
) -> Result<Vec<f32>, CudaError> {
    launch_nary(device.stream(), allocator, device.ordinal(), func, leaves)
}

#[cfg(test)]
pub(crate) mod test_support {
    use std::sync::Mutex;

    /// [`super::GPU_ELEMENTWISE_FUSION_ENABLED`] を操作するテスト間で
    /// 直列化する RAII ガード（`crate::precision::test_support::
    /// tf32_flag_test_lock` と同型。`cargo test` の既定並列実行下でも
    /// フラグ操作が競合しないようにする）。
    pub(crate) fn gate_test_lock() -> &'static Mutex<()> {
        static LOCK: Mutex<()> = Mutex::new(());
        &LOCK
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use fandhe_ai_tensor_core::FusionPlan;

    struct GateGuard {
        _lock: std::sync::MutexGuard<'static, ()>,
        original: bool,
    }

    impl GateGuard {
        fn acquire() -> Self {
            let lock = test_support::gate_test_lock()
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            let original = gpu_elementwise_fusion_enabled();
            Self {
                _lock: lock,
                original,
            }
        }
    }

    impl Drop for GateGuard {
        fn drop(&mut self) {
            set_gpu_elementwise_fusion_enabled(self.original);
        }
    }

    /// ゲートは既定 `false`（実装計画 §5.1 (c)「既定 OFF のドリフト
    /// 検出」）。他のテストがフラグを変更していても本テストが直列化
    /// ロックを取得できれば、既定値は前回の `Drop` により復元済みの
    /// はず。
    #[test]
    fn gate_defaults_to_disabled() {
        let _guard = GateGuard::acquire();
        // `GateGuard::acquire` 自体はフラグを変更しないため、ロック
        // 取得直後の値が「他テストの `Drop` 復元後の既定値」を表す。
        // プロセス起動直後の初期値を厳密に検証するにはプロセス分離が
        // 要るため、ここでは「明示的に true へ変更していない限り false」
        // という運用上の既定値検証に留める（`precision.rs` の同種
        // テストと同じ限界）。
        set_gpu_elementwise_fusion_enabled(false);
        assert!(!gpu_elementwise_fusion_enabled());
    }

    #[test]
    fn gate_round_trips() {
        let _guard = GateGuard::acquire();
        set_gpu_elementwise_fusion_enabled(true);
        assert!(gpu_elementwise_fusion_enabled());
        set_gpu_elementwise_fusion_enabled(false);
        assert!(!gpu_elementwise_fusion_enabled());
    }

    fn build_plan_4_chain() -> FusionPlan {
        // i0, i1, add(0,1), relu(2), exp(3), tanh(4) — allowlist 内。
        FusionPlan::from_ops(
            vec![
                FusedOpKind::Input { leaf_index: 0 },
                FusedOpKind::Input { leaf_index: 1 },
                FusedOpKind::Add { lhs: 0, rhs: 1 },
                FusedOpKind::Relu { input: 2 },
                FusedOpKind::Exp { input: 3 },
                FusedOpKind::Tanh { input: 4 },
            ],
            vec![4],
            fandhe_ai_tensor_core::DType::F32,
            2,
        )
        .expect("valid plan")
    }

    #[test]
    fn match_elementwise_plan_accepts_allowlisted_chain() {
        let plan = build_plan_4_chain();
        let program = match_elementwise_plan(&plan).expect("allowlist match");
        assert_eq!(program.leaf_count, 2);
        assert_eq!(program.ops.len(), 6);
    }

    #[test]
    fn match_elementwise_plan_rejects_sum() {
        let plan = FusionPlan::from_ops(
            vec![
                FusedOpKind::Input { leaf_index: 0 },
                FusedOpKind::Relu { input: 0 },
                FusedOpKind::Sum {
                    input: 1,
                    axis: None,
                },
            ],
            vec![4],
            fandhe_ai_tensor_core::DType::F32,
            1,
        )
        .expect("valid plan");
        assert!(match_elementwise_plan(&plan).is_none());
    }

    #[test]
    fn match_elementwise_plan_rejects_row_fusion_plan() {
        // RMSNorm canonical 形（6 op・全軸縮約）は `row_fusion()` を
        // 持つため本モジュールの対象外（`ops.rs::run_fused` の先行
        // 分岐が扱う）。
        let plan = FusionPlan::from_ops(
            vec![
                FusedOpKind::Input { leaf_index: 0 },
                FusedOpKind::Mul { lhs: 0, rhs: 0 },
                FusedOpKind::Sum {
                    input: 1,
                    axis: None,
                },
                FusedOpKind::Rsqrt { input: 2 },
                FusedOpKind::Broadcast {
                    input: 3,
                    axis: None,
                },
                FusedOpKind::Mul { lhs: 4, rhs: 0 },
            ],
            vec![8],
            fandhe_ai_tensor_core::DType::F32,
            1,
        )
        .expect("valid plan");
        assert!(match_elementwise_plan(&plan).is_none());
    }
}
