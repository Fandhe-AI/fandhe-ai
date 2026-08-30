//! テンソル型・演算グラフ／カーネル融合機構の完全自作コア。
//!
//! Burn 等の既存 ML フレームワークに依存せず、テンソルの形状・ストレージ表現と
//! 演算グラフ（カーネル融合機構）を本クレートで自作する（REQ-1 v2。
//! `.claude/rules/coding-rust.md`）。`autodiff` クレートはこのテンソル表現の上に
//! 動的テープを構築し、`backend-cpu` / `backend-cuda` / `backend-metal` は
//! ここで定義する演算グラフのノードを各バックエンドのカーネルへ変換して実行する。
//!
//! TASK-1.4a でテンソル型のデータ構造（`Tensor<T>` の stride レイアウト・
//! `Arc<Storage<T>>` 所有権モデル・生成系／zero-copy view API）を実装済み。
//! TASK-1.4b（#12）で NumPy 互換ブロードキャスト（`broadcast_shape`・
//! `Tensor::broadcast_to`／`broadcast_with`。stride 0 による zero-copy view）
//! を追加した。`ops_shape`（TASK-1.4c・#13）は matmul・elementwise・
//! reduction 等の演算実行時 shape 検査を、`autodiff`（`Var`）・backend
//! 入口（`DeviceBuffer`）の双方から再利用可能な純粋関数群として提供する。
//! PoC-v2-1 数値突合の総合テスト（TASK-1.4d・#14）は `tests/` に整備済み:
//! `tests/poc_v2_1_parity.rs` が PoC-v2-1 のテストベクタ移植・
//! `evidence/dump_case_512.bin` との数値突合（`docs/spec` submodule
//! checkout が必要なため `#[ignore]` でローカル実行に分離）を、
//! `tests/tensor_views.rs` が複数 API 組合せの統合テストを担う。
//! 演算グラフ本体（カーネル融合機構）は後続タスクで追加する（spec 根拠:
//! `docs/spec/05-tasks.md` TASK-1.4、`docs/public-api-design.md` §2）。
//!
//! `device`（TASK-1.9a・#44）は `backend-cpu`／`backend-cuda`／
//! `backend-metal` が実装するデバイス列挙・選択の共通 trait
//! （[`device::DeviceProvider`]）・共通型（[`device::Device`]・
//! [`device::DeviceInfo`]）・エラー型（[`device::BackendError`]）を提供する。
//! 3 バックエンドクレートはいずれも `tensor-core` に依存するため、trait
//! 定義をここに置き各バックエンド側で実装する依存逆転構成を取る
//! （`docs/public-api-design.md` §4）。
//!
//! `buffer`（TASK-1.9b・#45）はデバイス常駐バッファ（[`buffer::DeviceBuffer`]）
//! と、各バックエンドが実装する確保・アップロード・ダウンロードの共通
//! 入口（[`buffer::MemoryOps`]）を提供する。解放は各バックエンドの具体
//! ハンドル型（`Box<dyn buffer::BufferHandle>` の中身）の `Drop` に一本化
//! する（明示 `free()` API は設けない）。`backend_ops`（TASK-1.9c・#46）は
//! ホスト常駐 `Tensor<f32>` ベースのカーネルディスパッチ
//! （[`backend_ops::BackendOps`]・[`backend_ops::ops_for`]）を提供する。
//!
//! `dispatch`（TASK-11.2b・#68）は行列演算ユニット（CUDA Tensor Core・
//! Metal `simdgroup_matrix`）経路の決定的な自作ディスパッチ規則
//! （[`dispatch::select_gemm_kernel`]）を提供する。REQ-11 v2 が定める
//! 「利用者向け明示切替 API を提供しない」方針のとおり、本モジュールは
//! `backend-cuda`／`backend-metal` の GEMM 自動経路入口が内部で呼ぶ規則
//! エンジンであり、`tensor-core` の公開 API 利用者が直接切り替える経路
//! ではない（設計は `docs/dispatch-rules-design.md`〈#67〉、決定表は
//! `dispatch` モジュールのドキュメンテーションコメント参照）。
//!
//! `typed`（TASK-10.1b・#99）は基盤層（実行時 rank・shape）の上に積む
//! **後続レイヤー**として、コンパイル時に shape が既知の固定次元層
//! （全結合層の重み・bias 等）に限定した const generics 型レベル検証
//! （[`typed::FixedVec`]・[`typed::FixedMat`]・[`typed::BatchedFeatures`]）
//! を提供する（REQ-10。基盤層自体の rank は実行時のまま変更しない設計
//! 判断は §2.5 参照。`docs/public-api-design.md`）。
//!
//! `memory_stats`（TASK-14.1a・#174）はアロケータ計測フックの共通シグネチャ
//! （[`memory_stats::MemoryStats`]）と、`backend-cpu`／`backend-cuda`／
//! `backend-metal` が共通利用する計測実装（[`memory_stats::AllocationTracker`]・
//! [`memory_stats::TrackedAllocation`]）を提供する。`buffer::MemoryOps` とは
//! 独立したトレイトとして新設し、本イシューでは `backend-cpu` のみが実装する
//! （CUDA/Metal への組み込みは #175。REQ-14）。
//!
//! `pool`（TASK-#201・REQ-14 14-3）は既存 `MemoryOps` 実装を包む opt-in
//! デコレータ（[`pool::PooledMemory`]）として、サイズクラス別（バイトサイズ
//! 完全一致）バッファプールを提供する。総量上限・自動破棄（グローバル
//! LRU）を最初から組み込み、上限超過時の破棄が `memory_stats` の計測
//! （`allocated_bytes`／`peak_allocated_bytes`）へそのまま反映される設計
//! （v1 で GEMM 4096³ のピークメモリが理論値の約 17 倍に蓄積した教訓を
//! 踏まえ、既定を無制限成長にしない安全側判断。`pool` モジュールコメント
//! 参照）。
//!
//! `fusion`（TASK-12.1a・#161 の設計〈`docs/fusion-graph-design.md`〉を
//! TASK-12.1b・#162 で実装したもの）は演算グラフのカーネル融合機構
//! （REQ-12）の中間表現（elementwise 5 演算に閉じた `FusionGraph`）と、
//! 融合可能な elementwise 連鎖（4〜6 段程度）を検出する判定アルゴリズム
//! （`fusion::detect_fusion`）を提供する。`FusionGraph`／`detect_fusion`
//! は `pub(crate)`（設計書 §2.5「配置は `tensor-core` の 1 か所に閉じる」）
//! のまま変更しない。TASK-12.1c・#163 で融合カーネル生成向け公開 DTO
//! （[`FusionPlan`]・[`FusedOpKind`]・[`FusedNodeIndex`]・
//! [`FusionPlanError`]。下記 re-export）を追加した——`backend-cpu`／
//! `backend-cuda`／`backend-metal` が `pub(crate)` の内部融合 IR を経由
//! せず融合グラフの内容を読み取る唯一の経路である（設計書 §3.4）。
//! TASK-12.1d・#164 で `BackendOps::run_fused` trait メソッド追加・
//! `autodiff` 側の遅延評価統合（`crates/autodiff/src/tape.rs` の
//! `Tape::push_lazy`／`materialize_fallible`／`materialize_non_fallible`）
//! を実装した。`autodiff` は `tensor-core` 内部の `pub(crate)`
//! `FusionGraph`／`detect_fusion` を経由せず、自身が保持する遅延ノード
//! 連鎖を直接 `FusedOpKind` 列へ変換して [`FusionPlan::from_ops`] を
//! 呼ぶ構成のため（`tensor-core` → `autodiff` の逆依存を作れないため）、
//! `FusionGraph`／`detect_fusion` は本クレート内では `plan.rs` の
//! `#[cfg(test)]`（`from_segment` の単体テスト）からのみ使用される。
//!
//! `layout`（`backend-metal` 専用モジュール〈#1040〉の 2 次元 view 転置
//! 分類・先頭次元 collapse）は、イシュー #1046 で `autodiff::eval::matmul`
//! と共用するため一時的に本クレートへ移設したが、`pub mod layout` が
//! `fandhe-ai-tensor-core`（crates.io 公開クレート）の公開面へ内部
//! レイアウト型を露出させてしまう問題が codex-review で指摘された
//! （`#[doc(hidden)]` は rustdoc 表示を隠すのみで Rust の可視性・semver
//! 上の公開面は変えないため契約にできない。AGENTS.md「内部表現の公開
//! API への漏出は P1」。PR #1077）。そのため PR #1077 で `backend-metal`
//! （`crates/backend-metal/src/layout.rs`）・`autodiff`
//! （`crates/autodiff/src/layout.rs`）それぞれのクレート内非公開
//! モジュールへ分類ロジックを複製する形へ差し戻し、本クレートからは
//! `layout` モジュール自体を削除した。両モジュールは共通のシェーダ
//! 添字契約（`backend-metal::shaders::gemm.metal` の
//! `gemm_tiled_bias_act`）を正とする双子モジュールであり、変更する際は
//! 両方に反映する（設計判断の記録は `docs/matmul-vjp-zero-copy-decision.md`）。

mod backend_ops;
mod broadcast;
pub mod buffer;
pub mod device;
pub mod dispatch;
mod dispatch_failure;
mod element;
mod error;
mod fusion;
pub mod memory_stats;
mod ops_shape;
pub mod pool;
// プールの共通コアロジック（サイズクラス・フリーリスト・統計）。`backend-cuda`／
// `backend-metal` が具体ハンドル型で実装を組み立てるためのクレート横断内部面で
// あり、サポート対象の公開 API ではない（PR #1063 codex-review P1 対応。公開契約
// は `PoolStats` の再エクスポートのみ。`docs/device-memory-pool-design.md` §3.1・
// §8。`#[doc(hidden)]` により docs.rs・rustdoc から隠し、semver 互換性の対象外で
// あることを明示する）。
#[doc(hidden)]
pub mod pool_core;
mod tensor;
pub mod typed;

pub use backend_ops::{Activation, BackendOps, SgdStepConfig, ops_for};
pub use broadcast::broadcast_shape;
pub use buffer::{BufferHandle, DeviceBuffer, DeviceBufferView, MemoryOps};
pub use device::{BackendError, Device, DeviceInfo, DeviceProvider, enumerate_all, select_from};
pub use dispatch::{DType, DeviceCaps, GemmShape, KernelKind, select_gemm_kernel};
pub use dispatch_failure::DispatchFailureCell;
pub use element::Element;
pub use error::ShapeError;
// `MAX_FUSED_CHAIN_LEN`（#404）: `fandhe_ai_autodiff::tape` の push 時上限適用が
// 参照する単一真実源（`fusion/detect.rs` の doc comment 参照）。
// `RowFusionMeta`（#588）: 行方向 reduction＋broadcast 融合プランの行
// メタデータ（`axis`／`row_len`。`fusion/plan.rs` の doc comment 参照）。
// 1 パス／2 パス判定の閾値定数（旧 `MAX_SINGLE_PASS_ROW_LEN`）は
// codex-review PR #687 P2 是正で backend 非依存層から削除済み
// （閾値判定は各バックエンドの責務。`RowFusionMeta` doc 参照）。
pub use fusion::{
    FusedNodeIndex, FusedOpKind, FusionPlan, FusionPlanError, MAX_FUSED_CHAIN_LEN,
    MAX_FUSED_SEGMENT_NODES, RowFusionMeta,
};
pub use memory_stats::{AllocationTracker, MemoryStats, TrackedAllocation};
pub use ops_shape::{
    elementwise_out_shape, matmul_out_shape, reduce_out_shape, require_same_shape,
};
pub use pool::{PoolConfig, PoolZeroFill, PooledMemory};
// `pool_core::SizeClassPoolConfig` は `pool::PoolConfig`（crates.io 0.4.0
// 公開済み）と紛らわしい命名衝突を避けるため意図的に非公開のまま
// （`pool_core.rs` モジュールコメント「命名の差異」参照）。`PoolStats` の
// みを再公開する（`backend_ops::BackendOps::device_memory_pool_stats` の
// 戻り値型。CUDA〈#1020〉・Metal〈#1021〉共通の統計スナップショット型）。
pub use pool_core::PoolStats;
pub use tensor::Tensor;
pub use typed::{BatchedFeatures, FixedMat, FixedVec};
