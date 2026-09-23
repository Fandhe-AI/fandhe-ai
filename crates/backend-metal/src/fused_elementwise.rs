//! GPU `run_fused` の elementwise allowlist 融合（区分 B-1・イシュー
//! #2085・`docs/autodiff-graph-optimization-scope-decision.md` §5。
//! CUDA 側 `backend-cuda::fused_elementwise` の Metal 対応版）。
//!
//! `ops.rs::MetalBackendOps::run_fused` は現状 canonical RMSNorm・
//! softmax プランのみを専用カーネルへルーティングし、それ以外の
//! elementwise 連鎖は `Unsupported` として呼び出し元の per-op
//! フォールバックへ倒す。本モジュールは `backend-cpu::fused_elementwise`
//! と同一の allowlist（`Input`／`Add`／`Mul`／`Relu`／`Exp`／`Tanh`）を
//! 対象に、実行時 MSL ソース生成（`crate::fused_elementwise_source`）
//! による単一パスカーネルへ融合し、往復を 1 回へ削減する。
//!
//! # opt-in ゲート（既定 OFF）
//!
//! [`gpu_elementwise_fusion_enabled`]／[`set_gpu_elementwise_fusion_enabled`]
//! はプロセスワイドな `AtomicBool`（CUDA 側 `backend-cuda::
//! fused_elementwise` と同型の opt-in スイッチ）。**ゲート OFF 時は
//! `run_fused` の融合分岐がデバイスアクセス（`context_cache::
//! cached_context` 等）より前に `Unsupported` を返す**（`ops.rs::
//! run_fused` の呼び出し順序契約。CUDA 側と同じく「ゲート OFF の挙動は
//! 本 PR 導入前と atomic load 1 回を除き bit 同一」という同一バイナリ
//! A/B の前提を保つ）。**本 PR では `facade` へのセッター公開は行わない**
//! （CUDA 側と同じ承認範囲。実装計画 §9 承認事項 (1)）。
//!
//! # allowlist（denylist 化しない。CUDA 側・CPU 版と同一方針）
//!
//! `match_elementwise_plan` は CUDA 側 `backend-cuda::
//! fused_elementwise::match_elementwise_plan` と同一の判定ロジック
//! （独立実装。両クレートは `tensor-core` 経由でのみ結合する設計方針
//! のため相互参照しない）。
//!
//! # 数値契約（REQ-2）
//!
//! 融合カーネルは同一バックエンドの per-op 経路
//! （`shaders/elementwise.metal`）・CPU `backend-cpu::fused_elementwise`
//! と bit 完全一致を目標とする（`fused_elementwise_source.rs` モジュール
//! 冒頭「数値契約」参照）。CPU 対 GPU の突合は引き続き REQ-2 統一複合
//! 判定に依る。

// 実際の呼び出し元（`ops.rs::MetalBackendOps::run_fused_elementwise_
// allowlist`）は `cfg(target_os = "macos")` 限定のため、Linux 単体
// ビルド（`cargo build`／`cargo clippy` の非テストパス）では
// `match_elementwise_plan`／`ElementwiseProgram` が「クレート内から
// 到達不能」と判定され dead_code lint が誤検知する。`row_kernel.rs`
// と同じ理由・同じ対処（`pub` へ広げず non-macOS ビルドに限定した
// `allow` で個別に抑制する。codex-review P1 指摘・PR #714 の先例）。
#![cfg_attr(not(target_os = "macos"), allow(dead_code))]

use std::sync::atomic::{AtomicBool, Ordering};

use fandhe_ai_tensor_core::{FusedOpKind, FusionPlan};

/// GPU elementwise 融合 opt-in ゲート本体。既定 `false`（CUDA 側
/// `backend-cuda::fused_elementwise::GPU_ELEMENTWISE_FUSION_ENABLED` と
/// 同型）。
static GPU_ELEMENTWISE_FUSION_ENABLED: AtomicBool = AtomicBool::new(false);

/// GPU `run_fused` の elementwise allowlist 融合を有効化する（既定
/// OFF。モジュール冒頭「opt-in ゲート」参照）。プロセスワイドな設定
/// であり、以降の全スレッド・全 `MetalBackendOps` インスタンスの
/// `run_fused` 呼び出しに反映される。
///
/// **`backend-metal` クレート内 `pub`（`facade` への再公開は本 PR の
/// 対象外。CUDA 側 `set_gpu_elementwise_fusion_enabled` と同じ可視性
/// 方針）**。現在の呼び出し元は本クレートの `#[ignore]` 実機統合
/// テスト（`tests/fused_elementwise_parity.rs`）のみ。
pub fn set_gpu_elementwise_fusion_enabled(enabled: bool) {
    GPU_ELEMENTWISE_FUSION_ENABLED.store(enabled, Ordering::SeqCst);
}

/// 現在の GPU elementwise 融合ゲート状態を返す（既定 `false`）。
pub fn gpu_elementwise_fusion_enabled() -> bool {
    GPU_ELEMENTWISE_FUSION_ENABLED.load(Ordering::SeqCst)
}

/// `plan` の op 列・葉数・`row_fusion()` を照合し、B-1 allowlist に
/// 一致する場合のみ [`ElementwiseProgram`] を返す（純関数。デバイス非
/// 依存で実機なしでも単体テスト可能。CUDA 側
/// `backend-cuda::fused_elementwise::match_elementwise_plan` と同一
/// ロジック）。検証順序・根拠は CUDA 側モジュールの同名関数 doc
/// コメントを参照（独立実装だが判定規則は同一）。
///
/// `MAX_FUSED_SEGMENT_NODES` との比較は `plan.ops()`（`Input` 葉を
/// 含む線形化列）の全長ではなく、`Input` を除いた演算・縮約ノード数
/// に対して行う（codex-review・Cursor Bugbot 指摘・PR #2232。CUDA 側
/// `match_elementwise_plan` doc コメントの「検証順序」3. 参照。両
/// クレート同一の是正）。加えて `plan.leaf_count() <=
/// MAX_FUSED_SEGMENT_NODES + 1` も検査する（演算・縮約ノード数の上限
/// だけでは、`FusionPlan::from_ops` が許す「大半が未使用の `Input`
/// を大量に持つプラン」を拒否できない。カーネル引数数〈葉引数 +
/// `out` + `numel`〉の際限ない増大を防ぐ多層防御。CUDA 側 doc
/// コメントの「検証順序」4. 参照）。
pub(crate) fn match_elementwise_plan(plan: &FusionPlan) -> Option<ElementwiseProgram> {
    if plan.row_fusion().is_some() {
        return None;
    }
    let ops: Vec<FusedOpKind> = plan.ops().collect();
    let segment_node_count = ops
        .iter()
        .filter(|op| !matches!(op, FusedOpKind::Input { .. }))
        .count();
    if ops.is_empty()
        || segment_node_count > fandhe_ai_tensor_core::MAX_FUSED_SEGMENT_NODES
        || plan.leaf_count() > fandhe_ai_tensor_core::MAX_FUSED_SEGMENT_NODES + 1
    {
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

/// [`match_elementwise_plan`] が受理した融合プランの実行可能表現
/// （CUDA 側 `ElementwiseProgram` と同一構造の独立型）。
#[derive(Debug, Clone)]
pub(crate) struct ElementwiseProgram {
    pub(crate) ops: Vec<FusedOpKind>,
    pub(crate) leaf_count: usize,
}

// パイプライン取得（実行時 MSL 生成 → キャッシュ済みコンパイル）は
// `objc2` 系 FFI（`MetalContext`／`MtlPipeline`）に触れるため、本モジュール
// （Linux でも単体テストが回る非 macOS 限定モジュール。`lib.rs` 参照）
// には置かず、`cfg(target_os = "macos")` 限定の `ops.rs::MetalBackendOps::
// run_fused_elementwise_allowlist` 内へ直接インライン化する
// （`row_kernel.rs` が canonical プラン照合のみを Linux 側へ切り出し、
// カーネル起動自体は `ops.rs`／`rmsnorm.rs`／`softmax.rs` 側に残す設計と
// 同じ役割分担）。

#[cfg(test)]
pub(crate) mod test_support {
    use std::sync::Mutex;

    /// [`super::GPU_ELEMENTWISE_FUSION_ENABLED`] を操作するテスト間で
    /// 直列化する RAII ガード（`crate::precision` 等が使う直列化 Mutex
    /// と同型）。
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

    #[test]
    fn gate_defaults_to_disabled() {
        let _guard = GateGuard::acquire();
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

    /// codex-review・Cursor Bugbot 指摘（PR #2232）の回帰テスト。CUDA 側
    /// `fused_elementwise::tests::
    /// match_elementwise_plan_accepts_wide_plan_with_many_leaves` と同型
    /// のプラン（演算・縮約ノード 6 個・葉 7 個の 13-entry）が誤って
    /// per-op フォールバックへ落ちないことを検証する。
    #[test]
    fn match_elementwise_plan_accepts_wide_plan_with_many_leaves() {
        let plan = FusionPlan::from_ops(
            vec![
                FusedOpKind::Input { leaf_index: 0 },  // 0
                FusedOpKind::Input { leaf_index: 1 },  // 1
                FusedOpKind::Input { leaf_index: 2 },  // 2
                FusedOpKind::Input { leaf_index: 3 },  // 3
                FusedOpKind::Input { leaf_index: 4 },  // 4
                FusedOpKind::Input { leaf_index: 5 },  // 5
                FusedOpKind::Input { leaf_index: 6 },  // 6
                FusedOpKind::Add { lhs: 0, rhs: 1 },   // 7
                FusedOpKind::Add { lhs: 2, rhs: 3 },   // 8
                FusedOpKind::Add { lhs: 4, rhs: 5 },   // 9
                FusedOpKind::Add { lhs: 9, rhs: 6 },   // 10
                FusedOpKind::Mul { lhs: 7, rhs: 8 },   // 11
                FusedOpKind::Add { lhs: 11, rhs: 10 }, // 12
            ],
            vec![4],
            fandhe_ai_tensor_core::DType::F32,
            7,
        )
        .expect("valid wide plan");
        assert_eq!(plan.ops().count(), 13);
        let program = match_elementwise_plan(&plan).expect("wide plan must be accepted");
        assert_eq!(program.leaf_count, 7);
        assert_eq!(program.ops.len(), 13);
    }

    /// codex-review・Cursor Bugbot 指摘の対応レビューで追加。CUDA 側
    /// `fused_elementwise::tests::
    /// match_elementwise_plan_rejects_excessive_leaf_count` と同型:
    /// 演算・縮約ノード数の上限だけでは拒否できない「大半が未使用の
    /// `Input` を大量に持つプラン」を、葉数上限（検証順序の doc
    /// コメント参照）が拒否することを確認する。
    #[test]
    fn match_elementwise_plan_rejects_excessive_leaf_count() {
        let leaf_count = fandhe_ai_tensor_core::MAX_FUSED_SEGMENT_NODES + 2;
        let mut ops: Vec<FusedOpKind> = (0..leaf_count)
            .map(|leaf_index| FusedOpKind::Input { leaf_index })
            .collect();
        ops.push(FusedOpKind::Add { lhs: 0, rhs: 1 });
        let plan =
            FusionPlan::from_ops(ops, vec![4], fandhe_ai_tensor_core::DType::F32, leaf_count)
                .expect("valid plan with mostly-unused leaves");
        assert_eq!(plan.leaf_count(), leaf_count);
        assert!(match_elementwise_plan(&plan).is_none());
    }
}
