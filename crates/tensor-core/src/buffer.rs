//! デバイス常駐バッファとメモリ操作抽象（TASK-1.9b・#45）。
//!
//! TASK-1.9a（#44）で `device::DeviceProvider`（デバイス列挙・選択）を
//! 追加したのに続き、本モジュールは「確保・アップロード・ダウンロード・
//! 解放」の共通抽象を提供する。`docs/public-api-design.md` §4.2 が指摘する
//! 課題（各バックエンドの GEMM 実装内部（`backend-cuda/src/gemm.rs` の
//! `clone_htod`/`alloc_zeros`/`clone_dtoh`、`backend-metal/src/buffer.rs` の
//! `MetalBuffer`）にホスト⇔デバイス転送が埋め込まれ、演算ごとにホスト
//! 往復が発生する構造）を解消する土台として、`backend-cpu`/`backend-cuda`/
//! `backend-metal` が共通実装する [`MemoryOps`] トレイトと、その戻り値
//! である [`DeviceBuffer`] を定義する。カーネルディスパッチ本体
//! （`BackendOps`。§4.2）は TASK-1.9c（#46）のスコープであり、
//! `MemoryOps` はその supertrait となる想定（本イシューでは結線しない）。
//!
//! # 依存逆転構成
//!
//! `device.rs`（#44）と同じ理由（`tensor-core` → `backend-*` の逆依存は
//! 作れない）により、[`DeviceBuffer`] は実データ（CUDA デバイスポインタ・
//! Metal `MTLBuffer` 等）への不透明ハンドルを `Box<dyn BufferHandle>` で
//! 保持する。具体型は各バックエンドクレートが定義し、`downcast_handle`
//! （`Any` ダウンキャスト）経由でのみ自分自身の具体型を取り出す。
//!
//! `BufferHandle: Debug + 'static`（`Any` を直接 supertrait にしない）
//! とし、`as_any()` を明示メソッドとして要求する設計にしている理由:
//! `&dyn BufferHandle` から `&dyn Any` への trait-upcasting coercion は
//! 対応 Rust バージョンに依存する機能であり、ツールチェーンのバージョン
//! ガンブルを避けるため（`Any` を supertrait にすると
//! `downcast_ref::<H>()` を呼ぶ前に `&dyn BufferHandle` → `&dyn Any` の
//! 暗黙変換が必要になり、これは 2024 年時点の安定 Rust では
//! `#[feature(trait_upcasting)]` なしには保証されない）。`as_any()` を
//! 経由すれば変換は各バックエンド実装が持つ具体型のメソッド呼び出しで
//! 完結し、ツールチェーン非依存でダウンキャストできる。
//!
//! # 解放方針（RAII 一本化）
//!
//! 明示 `free()` API は設けない（二重解放・解放漏れの構造的排除）。
//! 各バックエンドの具体ハンドル型が `Drop` で実データを解放する
//! （CUDA: `CudaSlice<f32>` の `Drop` がストリーム上で解放、Metal:
//! `Retained<MTLBuffer>` の `Drop`、CPU: `Vec<f32>` の `Drop`）。
//! `DeviceBuffer` 自体は `Box<dyn BufferHandle>` を所有するのみで、
//! `Drop` を独自実装しない（`Box` の既定 drop 実装が中身の `Drop` を
//! 正しく呼ぶ）。
//!
//! # 空テンソル（numel == 0）の契約
//!
//! `numel() == 0` のバッファは FFI を呼ばず「空ハンドル」（デバイス確保
//! なし）で表現する契約とする。Metal の `MetalBuffer` は zero-length
//! 確保を `MetalError::ZeroLengthAllocation` として拒否し（`buffer.rs`
//! 参照）、CUDA も 0 バイトの `cuMemAlloc` を一部環境の driver が拒否
//! しうる（`backend-cuda/src/gemm.rs` の `k == 0` 早期 return コメント
//! 参照）。この統一契約により、各バックエンドの `MemoryOps` 実装は
//! `numel == 0` を確保・転送呼び出し前の早期 return として扱い、
//! `download` は空 `Tensor` を返す（FFI 経路には一切触れない）。
//!
//! # download の同期契約
//!
//! `download` は復帰時点でホストデータが確定していることを全バックエンド
//! 共通の契約とする。CUDA の `cuMemcpyDtoHAsync`（`clone_dtoh` が内部で
//! 使う。`cudarc-0.19.8/src/driver/safe/core.rs` の `memcpy_dtoh` 実装
//! 参照）は非同期であるため、`backend-cuda::CudaMemory::download` は
//! `clone_dtoh` の直後に `stream.synchronize()` を挟む（既存
//! `backend-cuda/src/gemm.rs` がカーネル起動後の readback で行っている
//! 同期パターンと同じ手順を `download` にも適用する）。
//!
//! # Send/Sync 境界
//!
//! v1 では `DeviceBuffer`/`BufferHandle` に `Send`/`Sync` 境界を要求しない
//! （Metal `Retained<MTLBuffer>` のスレッド安全性を過剰に約束しないため）。
//! 必要になった時点（TASK-1.9c 以降）で再検討する。

use std::any::Any;
use std::fmt;

use crate::device::{BackendError, Device};
use crate::element::Element;
use crate::tensor::Tensor;

/// 各バックエンドが定義する不透明バッファハンドルの共通境界。
///
/// `Debug` を要求するのは診断用（`DeviceBuffer` の `Debug` 導出・エラー
/// メッセージ）。`as_any` はモジュール冒頭コメントのとおり
/// trait-upcasting に依存しないダウンキャスト経路を提供する
/// （実装は各バックエンドの具体型で `fn as_any(&self) -> &dyn Any { self }`
/// の 1 行で済む）。
pub trait BufferHandle: fmt::Debug + 'static {
    /// `downcast_handle` から呼ばれる。実装は常に `self` を返すだけでよい。
    fn as_any(&self) -> &dyn Any;

    /// `as_any` の可変版。`pool::PoolZeroFill::zero_fill`（TASK-#201）が
    /// プールから再利用した直後（まだ他に共有されていない排他所有の
    /// 段階）のバッファへ書き込むために使う。実装は `as_any` と同じく
    /// 常に `self` を返すだけでよい（この可変アクセス経路のみで
    /// ゼロ初期化を行い、FFI 境界以外での新規 `unsafe` 追加を避ける。
    /// `.claude/rules/coding-rust.md`「`unsafe` は FFI 境界等の必要
    /// 最小限に留める」方針）。
    fn as_any_mut(&mut self) -> &mut dyn Any;
}

/// デバイス上に確保されたバッファへのハンドル。
///
/// `device`/`shape` はメタデータとして `tensor-core`/backend 入口から
/// 直接参照できるが、実データは `handle`（`Box<dyn BufferHandle>`）の
/// 内部に閉じ込められ、具体型は `downcast_handle` を呼んだバックエンド
/// 自身のみが取り出せる。`T: Element` は `docs/public-api-design.md`
/// §4.2 のシグネチャをそのまま維持するための型パラメータであり、
/// `handle` 自体は要素型を持たない（`PhantomData<T>` で型情報のみ保持）。
#[derive(Debug)]
pub struct DeviceBuffer<T: Element> {
    device: Device,
    shape: Vec<usize>,
    handle: Box<dyn BufferHandle>,
    _marker: std::marker::PhantomData<T>,
}

impl<T: Element> DeviceBuffer<T> {
    /// 新規 `DeviceBuffer` を構築する。バックエンド実装（`CpuMemory`／
    /// `CudaMemory`／`MetalMemory` の `MemoryOps` 実装）の `alloc_zeroed`／
    /// `upload` から呼ばれる、具体ハンドルを持つバックエンド専用の構築
    /// 入口（`tensor-core`/backend 入口の他コードから直接構築しない）。
    pub fn new(device: Device, shape: Vec<usize>, handle: Box<dyn BufferHandle>) -> Self {
        Self {
            device,
            shape,
            handle,
            _marker: std::marker::PhantomData,
        }
    }

    /// このバッファが属するデバイス。
    pub fn device(&self) -> Device {
        self.device
    }

    /// shape（各軸のサイズ）。
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    /// 全要素数。
    pub fn numel(&self) -> usize {
        self.shape.iter().product()
    }

    /// 要素数が 0 か判定する（モジュール冒頭「空テンソルの契約」参照。
    /// `clippy::len_without_is_empty` 対応として `numel`/`is_empty` の
    /// ペアで公開する。`MetalBuffer::is_empty`（`backend-metal/src/buffer.rs`）
    /// と同じ設計判断）。
    pub fn is_empty(&self) -> bool {
        self.numel() == 0
    }

    /// バックエンド自身の具体ハンドル型 `H` へダウンキャストする。
    ///
    /// `H` が `handle` の実型と一致しない場合（他バックエンドの
    /// `DeviceBuffer` を誤って渡した等）は `None` を返す（`unwrap`/
    /// `expect` を要求しない。呼び出し元は `BackendError::DeviceMismatch`
    /// 等へ変換する想定）。
    pub fn downcast_handle<H: BufferHandle>(&self) -> Option<&H> {
        self.handle.as_any().downcast_ref::<H>()
    }

    /// 内部ハンドルの所有権を取り出す（`downcast_handle` は参照しか返さない）。
    ///
    /// `pool`（TASK-#201）が確保直後の `DeviceBuffer` を `PooledBufferHandle`
    /// で包み直す（返却時にプールへ戻せるようにする）ために所有権移転が
    /// 必要であり、そのための唯一の入口として `pub(crate)` 限定で追加する。
    /// `tensor-core` の外からハンドルの所有権を直接奪える一般公開 API には
    /// しない（本モジュール冒頭「解放方針（RAII 一本化）」を壊さないため、
    /// 呼び出し元は取り出した `Box<dyn BufferHandle>` を必ず別の
    /// `DeviceBuffer`／`BufferHandle` 実装（`PooledBufferHandle` 等）に
    /// 移し替え、解放責務を引き継ぐことを前提とする）。
    pub(crate) fn into_handle(self) -> Box<dyn BufferHandle> {
        self.handle
    }
}

/// 各バックエンド（CPU/CUDA/Metal）が実装するメモリ操作の共通入口
/// （`docs/public-api-design.md` §4.2 の `upload`/`download` を土台に、
/// `alloc_zeroed` を追加した v1・f32 固定版）。
///
/// object-safe に設計している（`device::DeviceProvider` と同じく `&dyn
/// MemoryOps` として扱える）。TASK-1.9c（#46）の `BackendOps` はこれを
/// supertrait として拡張し、カーネルディスパッチ（`gemm`/`add`/... ）を
/// 追加する想定。
///
/// f32 固定である理由: `docs/public-api-design.md` §4.2「`BackendOps` が
/// f32 専用である理由」と同じ（PoC-v2-5 実測 API が f32 のみ）。GPU 推論
/// で使う `half::f16` 経路の入口設計は TASK-1.9 実装時の決定事項として
/// 未決（§6-8）であり、本イシューのスコープ外として記録する
/// （`.claude/rules/out-of-scope-tracking.md`）。
pub trait MemoryOps {
    /// `shape` 分のゼロ初期化されたデバイスバッファを確保する。
    /// `numel() == 0` の場合は FFI を呼ばず空ハンドルを返す
    /// （モジュール冒頭「空テンソルの契約」参照）。
    fn alloc_zeroed(&self, shape: &[usize]) -> Result<DeviceBuffer<f32>, BackendError>;

    /// ホスト常駐の `tensor` をデバイスへアップロードする。
    ///
    /// 非 contiguous な `tensor`（`transpose`/`narrow` 後の view）は
    /// 内部で `tensor.contiguous()` により実体化してから転送する
    /// （`upload` はそもそもコピーを伴う明示 API であり、
    /// `docs/public-api-design.md` §2.2.1 の「暗黙コピー禁止」論点
    /// （`reshape` の非 contiguous ケース）とは別枠。`reshape` は
    /// zero-copy が期待される API のため暗黙コピーを避けるが、`upload`
    /// はホスト→デバイスの実データ転送そのものであるため、この契約とは
    /// 衝突しない）。
    fn upload(&self, tensor: &Tensor<f32>) -> Result<DeviceBuffer<f32>, BackendError>;

    /// デバイス常駐の `buffer` をホストへダウンロードする。
    ///
    /// 復帰時点でホストデータが確定していることを全バックエンド共通の
    /// 契約とする（モジュール冒頭「download の同期契約」参照）。
    /// `numel() == 0` の場合は FFI を呼ばず空 `Tensor` を返す。
    fn download(&self, buffer: &DeviceBuffer<f32>) -> Result<Tensor<f32>, BackendError>;
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::Cell;
    use std::rc::Rc;

    /// テスト専用のモックハンドル。`drop_count` を通じて `Drop` が
    /// 何回呼ばれたかを検証する（RAII 一本化方針の受け入れ条件検証。
    /// `Rc<Cell<usize>>` はシングルスレッドテストで十分なため
    /// `Send`/`Sync` を要求しない設計判断（モジュール冒頭コメント）と
    /// 整合する）。
    #[derive(Debug)]
    struct MockHandle {
        drop_count: Rc<Cell<usize>>,
        payload: Vec<f32>,
    }

    impl BufferHandle for MockHandle {
        fn as_any(&self) -> &dyn Any {
            self
        }

        fn as_any_mut(&mut self) -> &mut dyn Any {
            self
        }
    }

    impl Drop for MockHandle {
        fn drop(&mut self) {
            self.drop_count.set(self.drop_count.get() + 1);
        }
    }

    /// 他バックエンドを模した無関係なハンドル型（downcast 失敗系の検証用）。
    #[derive(Debug)]
    struct OtherHandle;

    impl BufferHandle for OtherHandle {
        fn as_any(&self) -> &dyn Any {
            self
        }

        fn as_any_mut(&mut self) -> &mut dyn Any {
            self
        }
    }

    #[test]
    fn shape_numel_and_device_accessors() {
        let handle: Box<dyn BufferHandle> = Box::new(MockHandle {
            drop_count: Rc::new(Cell::new(0)),
            payload: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        });
        let buf: DeviceBuffer<f32> = DeviceBuffer::new(Device::Cpu, vec![2, 3], handle);

        assert_eq!(buf.device(), Device::Cpu);
        assert_eq!(buf.shape(), &[2, 3]);
        assert_eq!(buf.numel(), 6);
        assert!(!buf.is_empty());
    }

    #[test]
    fn empty_shape_buffer_reports_zero_numel() {
        let handle: Box<dyn BufferHandle> = Box::new(MockHandle {
            drop_count: Rc::new(Cell::new(0)),
            payload: Vec::new(),
        });
        let buf: DeviceBuffer<f32> = DeviceBuffer::new(Device::Cpu, vec![0, 3], handle);

        assert_eq!(buf.numel(), 0);
        assert!(buf.is_empty());
    }

    #[test]
    fn downcast_handle_succeeds_for_matching_type() {
        let handle: Box<dyn BufferHandle> = Box::new(MockHandle {
            drop_count: Rc::new(Cell::new(0)),
            payload: vec![42.0],
        });
        let buf: DeviceBuffer<f32> = DeviceBuffer::new(Device::Cpu, vec![1], handle);

        let downcast = buf.downcast_handle::<MockHandle>();
        assert!(downcast.is_some());
        assert_eq!(downcast.unwrap().payload, vec![42.0]);
    }

    #[test]
    fn downcast_handle_fails_for_mismatched_type() {
        let handle: Box<dyn BufferHandle> = Box::new(OtherHandle);
        let buf: DeviceBuffer<f32> = DeviceBuffer::new(Device::Cpu, vec![1], handle);

        assert!(buf.downcast_handle::<MockHandle>().is_none());
    }

    #[test]
    fn drop_releases_handle_exactly_once() {
        let counter = Rc::new(Cell::new(0));
        let handle: Box<dyn BufferHandle> = Box::new(MockHandle {
            drop_count: Rc::clone(&counter),
            payload: vec![1.0; 16],
        });
        let buf: DeviceBuffer<f32> = DeviceBuffer::new(Device::Cpu, vec![16], handle);

        assert_eq!(counter.get(), 0, "構築直後はまだ drop されていない");
        drop(buf);
        assert_eq!(
            counter.get(),
            1,
            "DeviceBuffer の drop は内部ハンドルを 1 回だけ解放する"
        );
    }

    /// 受け入れ条件「各バックエンドで確保・転送・解放がリークなく動作する」
    /// の tensor-core 側の受け皿: N 回の確保→解放サイクル（空テンソルを
    /// 含む）で、都度ちょうど 1 回だけ `Drop` が呼ばれ、カウンタが
    /// 単調に増加することを検証する（実機依存の CUDA/Metal リーク検証
    /// （`#[ignore]` 分離。計画 §6 参照）を補う、環境非依存の解放回数検証）。
    #[test]
    fn repeated_alloc_drop_cycles_release_every_handle_exactly_once() {
        let counter = Rc::new(Cell::new(0));
        for i in 0..50 {
            // 偶数回は空テンソル（numel == 0）の解放経路も同じ counter で
            // 検証する（モジュール冒頭「空テンソルの契約」の解放側カバー）。
            let (shape, payload) = if i % 2 == 0 {
                (vec![0], Vec::new())
            } else {
                (vec![4], vec![0.0f32; 4])
            };
            let handle: Box<dyn BufferHandle> = Box::new(MockHandle {
                drop_count: Rc::clone(&counter),
                payload,
            });
            let buf: DeviceBuffer<f32> = DeviceBuffer::new(Device::Cpu, shape, handle);
            drop(buf);
            assert_eq!(counter.get(), i + 1, "サイクル {i} 回目で解放回数がずれた");
        }
        assert_eq!(counter.get(), 50);
    }
}
