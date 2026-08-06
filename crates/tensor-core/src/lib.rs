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
//! する（明示 `free()` API は設けない）。`BackendOps`（カーネル
//! ディスパッチ本体）は後続タスク（TASK-1.9c・#46）で追加する。
//!
//! `dispatch`（TASK-11.2b・#68）は行列演算ユニット（CUDA Tensor Core・
//! Metal `simdgroup_matrix`）経路の決定的な自作ディスパッチ規則
//! （[`dispatch::select_gemm_kernel`]）を提供する。REQ-11 v2 が定める
//! 「利用者向け明示切替 API を提供しない」方針のとおり、本モジュールは
//! `backend-cuda`／`backend-metal` の GEMM 自動経路入口が内部で呼ぶ規則
//! エンジンであり、`tensor-core` の公開 API 利用者が直接切り替える経路
//! ではない（設計は `docs/dispatch-rules-design.md`〈#67〉、決定表は
//! `dispatch` モジュールのドキュメンテーションコメント参照）。

mod broadcast;
pub mod buffer;
pub mod device;
pub mod dispatch;
mod element;
mod error;
mod ops_shape;
mod tensor;

pub use broadcast::broadcast_shape;
pub use buffer::{BufferHandle, DeviceBuffer, MemoryOps};
pub use device::{BackendError, Device, DeviceInfo, DeviceProvider, enumerate_all, select_from};
pub use dispatch::{DType, DeviceCaps, GemmShape, KernelKind, select_gemm_kernel};
pub use element::Element;
pub use error::ShapeError;
pub use ops_shape::{
    elementwise_out_shape, matmul_out_shape, reduce_out_shape, require_same_shape,
};
pub use tensor::Tensor;
