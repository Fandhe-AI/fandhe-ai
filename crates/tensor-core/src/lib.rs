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

mod backend_ops;
mod broadcast;
pub mod buffer;
pub mod device;
pub mod dispatch;
mod element;
mod error;
pub mod memory_stats;
mod ops_shape;
pub mod pool;
mod tensor;
pub mod typed;

pub use backend_ops::{BackendOps, ops_for};
pub use broadcast::broadcast_shape;
pub use buffer::{BufferHandle, DeviceBuffer, MemoryOps};
pub use device::{BackendError, Device, DeviceInfo, DeviceProvider, enumerate_all, select_from};
pub use dispatch::{DType, DeviceCaps, GemmShape, KernelKind, select_gemm_kernel};
pub use element::Element;
pub use error::ShapeError;
pub use memory_stats::{AllocationTracker, MemoryStats, TrackedAllocation};
pub use ops_shape::{
    elementwise_out_shape, matmul_out_shape, reduce_out_shape, require_same_shape,
};
pub use pool::{PoolConfig, PoolZeroFill, PooledMemory};
pub use tensor::Tensor;
pub use typed::{BatchedFeatures, FixedMat, FixedVec};
