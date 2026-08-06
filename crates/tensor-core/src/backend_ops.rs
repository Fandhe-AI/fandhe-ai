//! カーネルディスパッチ機構（TASK-1.9c・#46）。
//!
//! 単一の計算記述（[`BackendOps`] を受け取る関数）から CPU／CUDA／Metal
//! いずれのバックエンドのカーネルへも呼び分けられるようにする入口。
//! `device`（TASK-1.9a・#44）と同じ依存逆転構成を踏襲する: trait 定義を
//! 3 バックエンドクレートが依存できる本クレートに置き、各バックエンド
//! クレート（`backend-cpu`／`backend-cuda`／`backend-metal`）側で実装する
//! （`tensor-core` → `backend-*` の逆依存は作らない）。
//!
//! シグネチャは `docs/public-api-design.md` §4.2 の `BackendOps` trait案を
//! 正本としつつ、以下の点で拡張・簡略化している（同文書「TASK-1.9 実装
//! イシューで本文書との突合を行うこと」に対応。突合結果は同文書にも
//! 注記する）:
//!
//! - **`DeviceBuffer`／`upload`／`download` を含めない**。§4.2 が示す
//!   デバイス常駐バッファ型・転送 API は TASK-1.9b（#45）の担当であり、
//!   本イシュー時点で `tensor-core`・3 バックエンドクレートいずれにも
//!   存在しない（実装開始時に `git fetch origin main` で確認済み）。
//!   本イシューの受け入れ条件は「同一コードで 3 バックエンドのカーネルが
//!   呼び分けられる」（機構的な呼び分け）であり、既存カーネル入口
//!   （CPU `gemm_blis_parallel`・CUDA `CudaGemm::run_tiled_f32`・Metal
//!   `MetalGemm::dispatch_auto`）がいずれもホスト常駐 `&[f32]` を受け取り
//!   内部で H2D／D2H 転送を完結させる契約であるため、`DeviceBuffer` なしで
//!   本受け入れ条件を満たせる。§4.2 の `DeviceBuffer` 版シグネチャへの
//!   移行（`upload`／`download` の追加）は #45 のマージ後、`BackendOps` の
//!   非破壊拡張（デフォルトメソッド追加等）として TASK-1.9d（#47）以降で
//!   検討する
//! - 各メソッドはホスト常駐 [`Tensor<f32>`](crate::Tensor) を受け取り
//!   [`Tensor<f32>`](crate::Tensor) を返す（§4.2 の `DeviceBuffer<f32>` を
//!   `Tensor<f32>` に読み替えた形）。CPU 実装は転送コストが発生しないため
//!   このままで問題なく、CUDA／Metal 実装は各メソッド内で
//!   `Tensor::as_slice` → カーネル呼び出し（内部で H2D／D2H）→
//!   `Tensor::new` で完結させる
//! - 未実装カーネル（CUDA／Metal の elementwise・reduction。TASK-1.9c 時点
//!   では両バックエンドとも GEMM カーネルのみ実装済み）は
//!   [`crate::device::BackendError::Unsupported`]（本イシューで追加した
//!   非破壊拡張 variant）を返す fail-safe 実装とする。GPU 側
//!   elementwise・reduction カーネルの実装自体は本イシューのスコープ外
//!   （out-of-scope-tracking.md 対象。引き継ぎ先はユーザー承認を得て別
//!   Issue で追跡する）
//!
//! ディスパッチ規則（形状・HW 判定による経路選択）は TASK-11.2b（#68）の
//! 担当でありスコープ外（`docs/dispatch-rules-design.md`。TASK-11.2a・
//! #67）。既定デバイス選択ロジック（CUDA 既定有効化の構成決定含む）も
//! ユーザー承認必須のためスコープ外（`device` モジュールと同方針）。
//! 3 バックエンド横断の統合テストは TASK-1.9d（#47）が本格的に担当し、
//! 本イシューは受け入れ条件検証に必要な最小限のテストに留める。

use crate::Tensor;
use crate::device::{BackendError, Device};

/// 各バックエンド（CPU／CUDA／Metal）が実装するカーネル入口
/// （`docs/public-api-design.md` §4.2。差分はモジュール冒頭コメント参照）。
///
/// object-safe に設計している（`&dyn BackendOps` として扱える。
/// [`ops_for`] が複数バックエンドを横断して選択する際に使用する）。
/// v1 は PoC-v2-5 実測 API（`MetalOps`）のスコープに合わせて `f32` 固定
/// とする（f16 経路のジェネリック化は §4.2 6-8 のとおり保留）。
///
/// 公開 API はすべて safe。`unsafe` は各バックエンド実装内部の FFI 境界
/// （`cudarc`・`objc2` 系呼び出し）に閉じ込める
/// （`.claude/rules/coding-rust.md`）。
pub trait BackendOps {
    /// このインスタンスが対応する [`Device`]（呼び出し元がログ・
    /// エラーメッセージで識別するために使う）。
    fn device(&self) -> Device;

    /// 行列積 `C = A @ B` を計算する（`A: [m, k]`・`B: [k, n]` の 2 次元
    /// テンソルのみ受け付ける。shape 不整合は
    /// [`BackendError::ShapeMismatch`]）。
    fn gemm(&self, a: &Tensor<f32>, b: &Tensor<f32>) -> Result<Tensor<f32>, BackendError>;

    // elementwise（`docs/public-api-design.md` §4.2 と同じ 5 演算）
    fn add(&self, a: &Tensor<f32>, b: &Tensor<f32>) -> Result<Tensor<f32>, BackendError>;
    fn mul(&self, a: &Tensor<f32>, b: &Tensor<f32>) -> Result<Tensor<f32>, BackendError>;
    fn relu(&self, a: &Tensor<f32>) -> Result<Tensor<f32>, BackendError>;
    fn exp(&self, a: &Tensor<f32>) -> Result<Tensor<f32>, BackendError>;
    fn tanh(&self, a: &Tensor<f32>) -> Result<Tensor<f32>, BackendError>;

    // reduction（`docs/public-api-design.md` §4.2 と同じ 2 演算）
    fn sum(&self, a: &Tensor<f32>, dim: Option<usize>) -> Result<Tensor<f32>, BackendError>;
    fn max(&self, a: &Tensor<f32>, dim: Option<usize>) -> Result<Tensor<f32>, BackendError>;
}

/// 複数の `&dyn BackendOps` を横断して `device` に一致する実装を選択する。
///
/// `device::select_from`（TASK-1.9a）と同型の注入式ディスパッチ:
/// `tensor-core` は `backend-cpu`／`backend-cuda`／`backend-metal` を直接
/// 参照できないため、呼び出し側（結線を担う上位クレート・テスト）が
/// `ops` を注入する。本関数こそが受け入れ条件「同一コードで 3 バック
/// エンドのカーネルが呼び分けられる」の直接の実装であり、`device` の
/// variant にのみ基づいて対応実装を返す（形状・HW ヒューリスティクスは
/// 一切持ち込まない。TASK-11.2b・#68 のスコープ）。
///
/// 対応する実装が `ops` に含まれない場合は
/// [`BackendError::DeviceUnavailable`] を返す（`device::select_from` と
/// 同じエラー variant・同じ意味論。「対応 provider／ops 未登録」を表す）。
pub fn ops_for<'a>(
    ops: &[&'a dyn BackendOps],
    device: Device,
) -> Result<&'a dyn BackendOps, BackendError> {
    ops.iter()
        .find(|candidate| candidate.device() == device)
        .copied()
        .ok_or_else(|| {
            BackendError::DeviceUnavailable(format!(
                "no BackendOps registered for device {device:?}"
            ))
        })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::ShapeError;

    /// テスト専用のモック `BackendOps`。実バックエンドに依存せず
    /// `ops_for` の選択ロジックを検証するために `tensor-core` 内で定義
    /// する（実バックエンドの検証は各バックエンドクレートの結合テスト
    /// で行う。`device` モジュールの `MockProvider` と同じ位置付け）。
    struct MockOps(Device);

    impl BackendOps for MockOps {
        fn device(&self) -> Device {
            self.0
        }

        fn gemm(&self, _a: &Tensor<f32>, _b: &Tensor<f32>) -> Result<Tensor<f32>, BackendError> {
            Err(BackendError::Unsupported("mock: gemm".into()))
        }

        fn add(&self, _a: &Tensor<f32>, _b: &Tensor<f32>) -> Result<Tensor<f32>, BackendError> {
            Err(BackendError::Unsupported("mock: add".into()))
        }

        fn mul(&self, _a: &Tensor<f32>, _b: &Tensor<f32>) -> Result<Tensor<f32>, BackendError> {
            Err(BackendError::Unsupported("mock: mul".into()))
        }

        fn relu(&self, _a: &Tensor<f32>) -> Result<Tensor<f32>, BackendError> {
            Err(BackendError::Unsupported("mock: relu".into()))
        }

        fn exp(&self, _a: &Tensor<f32>) -> Result<Tensor<f32>, BackendError> {
            Err(BackendError::Unsupported("mock: exp".into()))
        }

        fn tanh(&self, _a: &Tensor<f32>) -> Result<Tensor<f32>, BackendError> {
            Err(BackendError::Unsupported("mock: tanh".into()))
        }

        fn sum(&self, _a: &Tensor<f32>, _dim: Option<usize>) -> Result<Tensor<f32>, BackendError> {
            Err(BackendError::Unsupported("mock: sum".into()))
        }

        fn max(&self, _a: &Tensor<f32>, _dim: Option<usize>) -> Result<Tensor<f32>, BackendError> {
            Err(BackendError::Unsupported("mock: max".into()))
        }
    }

    /// object-safe であることの型検査を兼ねる（`Box<dyn BackendOps>` が
    /// 構築できることをコンパイル時に確認する）。
    fn assert_object_safe(_ops: &dyn BackendOps) {}

    #[test]
    fn ops_for_dispatches_to_matching_device() {
        let cpu = MockOps(Device::Cpu);
        let cuda = MockOps(Device::Cuda(0));
        let ops: Vec<&dyn BackendOps> = vec![&cpu, &cuda];

        let selected = ops_for(&ops, Device::Cuda(0)).expect("cuda ops registered");
        assert_eq!(selected.device(), Device::Cuda(0));
        assert_object_safe(selected);

        let selected = ops_for(&ops, Device::Cpu).expect("cpu ops registered");
        assert_eq!(selected.device(), Device::Cpu);
    }

    #[test]
    fn ops_for_missing_device_returns_device_unavailable() {
        let cpu = MockOps(Device::Cpu);
        let ops: Vec<&dyn BackendOps> = vec![&cpu];

        // `ops_for` の `Ok` 側は `&dyn BackendOps` を含み `Debug` を実装
        // しないため `expect_err` は使わず、`is_err`／`matches!` で
        // `Err` 経路のみ検査する。
        let result = ops_for(&ops, Device::Cuda(0));
        assert!(result.is_err());
        assert!(matches!(result, Err(BackendError::DeviceUnavailable(_))));
    }

    #[test]
    fn unsupported_error_carries_shape_error_independently() {
        // `BackendError::Unsupported` が既存 variant（`ShapeMismatch` 等）と
        // 独立して構築・表示できることを確認する（非破壊追加の検証）。
        let err = BackendError::Unsupported("elementwise add on cuda".into());
        assert!(err.to_string().contains("elementwise add on cuda"));

        let shape_err = BackendError::ShapeMismatch(ShapeError::RankMismatch {
            expected: 2,
            actual: 1,
        });
        assert!(!shape_err.to_string().is_empty());
    }
}
