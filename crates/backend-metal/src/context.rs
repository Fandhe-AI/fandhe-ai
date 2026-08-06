//! Metal デバイス・コマンドキューの基盤（TASK-1.8a・#38）。
//!
//! `tensor-core` の演算グラフノードを MSL カーネルへディスパッチする前段
//! として、システムデフォルトの Metal デバイス取得とコマンドキュー生成を
//! 一箇所にまとめる。MSL ライブラリのコンパイル・パイプライン構築・
//! ディスパッチ経路は本イシューのスコープ外（TASK-1.8b・#39 で
//! `MetalContext` を土台にして追加する）。
//!
//! **移植元**: `docs/spec/03-poc/poc-v2-4-metal-gemm/code/rust/src/metal_gemm.rs`
//! の `MetalGemm::new`（デバイス・キュー取得部分）。PoC は `Option` 返しの
//! `expect` 呼び出しだったが、本実装は [`MetalError`] を返す `Result` 化
//! （coding-rust.md「本番経路で unwrap/expect を使わない」）。

use objc2::rc::{Retained, autoreleasepool};
use objc2::runtime::ProtocolObject;
use objc2_metal::{
    MTLCommandBuffer, MTLCommandBufferStatus, MTLCommandEncoder, MTLCommandQueue,
    MTLComputeCommandEncoder, MTLCreateSystemDefaultDevice, MTLDevice, MTLGPUFamily,
};
use tensor_core::dispatch::DeviceCaps;

use crate::error::MetalError;

pub(crate) type MtlDevice = ProtocolObject<dyn MTLDevice>;
pub(crate) type MtlQueue = ProtocolObject<dyn MTLCommandQueue>;

/// Metal デバイスとコマンドキューを保持するハンドル。
///
/// [`crate::buffer::MetalBuffer`] の確保・[`MetalContext::dispatch_sync`]
/// の同期実行はいずれも本構造体が保持する `device` / `queue` を介して
/// 行う。TASK-1.8b（#39）以降のパイプライン構築・エンコーダ結線は
/// 本構造体を土台にして追加される想定であり、公開フィールドは持たせず
/// アクセサ（[`MetalContext::device`] / [`MetalContext::queue`]）経由に
/// 限定する。
///
/// TASK-11.2b（#68）で `caps`（[`DeviceCaps`]）を追加した。`MTLDevice::
/// supportsFamily(MTLGPUFamily::Apple7)` の判定結果を `new` 時に 1 回
/// キャッシュし、[`crate::gemm::MetalGemm::dispatch_backend_auto`] から
/// `tensor_core::dispatch::select_gemm_kernel` へそのまま渡せるようにする
/// （`docs/dispatch-rules-design.md` §2.1「判定タイミング: デバイス初期化
/// 時に 1 回」。ディスパッチ呼び出しごとに `supportsFamily` を再照会
/// しない）。
pub struct MetalContext {
    device: Retained<MtlDevice>,
    queue: Retained<MtlQueue>,
    caps: DeviceCaps,
}

impl MetalContext {
    /// システムデフォルトの Metal デバイスを取得し、コマンドキューを
    /// 生成する。デバイスが見つからない・キュー生成に失敗した場合は
    /// [`MetalError`] を返す（PoC-v2-4 の `Option` 返しを型付きエラーへ
    /// 置き換え）。
    ///
    /// `MTLDevice::supportsFamily(MTLGPUFamily::Apple7)`（`simdgroup_matrix`
    /// 対応可否の判定材料。`docs/dispatch-rules-design.md` §2 表）もここで
    /// 1 回だけ評価し [`DeviceCaps`] へキャッシュする（[`Self::caps`]）。
    pub fn new() -> Result<Self, MetalError> {
        let device = MTLCreateSystemDefaultDevice().ok_or(MetalError::DeviceUnavailable)?;
        let queue = device
            .newCommandQueue()
            .ok_or(MetalError::CommandQueueCreation)?;
        // `supportsFamily` は objc2-metal が safe メソッドとして提供する
        // （`MTLDevice.rs` 生成コードに `unsafe` プレフィックスなし。
        // `device.rs::probe_all` が同様に他の `MTLDevice` メソッドを
        // unsafe ブロックなしで呼んでいるのと同じ扱い）。判定失敗
        // （このメソッド自体は bool を返すため失敗はしないが、将来
        // API が変わり得ない前提を置かない）時は非対応（Apple7 未満）
        // 扱いに倒す fail-safe とする（§2.2）。
        let apple7_supported = device.supportsFamily(MTLGPUFamily::Apple7);
        let caps = DeviceCaps::metal(apple7_supported);
        Ok(Self {
            device,
            queue,
            caps,
        })
    }

    /// [`crate::buffer::MetalBuffer`] の確保・パイプライン構築
    /// （TASK-1.8b・#39 以降）から参照される Metal デバイスハンドル。
    pub fn device(&self) -> &MtlDevice {
        &self.device
    }

    /// コマンドバッファ生成に使うコマンドキュー
    /// （TASK-1.8b・#39 のディスパッチ経路から参照される）。
    pub fn queue(&self) -> &MtlQueue {
        &self.queue
    }

    /// `new` 時にキャッシュした GPU family 判定結果
    /// （[`crate::gemm::MetalGemm::dispatch_backend_auto`] が
    /// `select_gemm_kernel` へ渡す `DeviceCaps`。TASK-11.2b・#68）。
    pub fn caps(&self) -> DeviceCaps {
        self.caps
    }

    /// コンピュートエンコーダを生成し `encode` にディスパッチ内容の記録
    /// を委ね、`commit()` + `waitUntilCompleted()` で完了を待つ同期実行
    /// ヘルパ。同期方式は PoC-v2-4 の計測境界（`GemmCase::dispatch`）と
    /// 同一にし、v1 系と揃える（バックエンド間比較の計測条件を崩さない
    /// ため。呼び出し元は TASK-1.8b・#39 のカーネルディスパッチ実装）。
    ///
    /// `commandBuffer()` / `computeCommandEncoder()` は autoreleased な
    /// オブジェクトを返す。Rust バイナリ（test/bench 実行ファイル等）には
    /// Cocoa アプリのような周囲の autorelease pool が存在しないため、
    /// `autoreleasepool` で明示的に囲まないと繰り返しディスパッチ
    /// （特にベンチマークループ）のたびに Metal の一時オブジェクトが
    /// プロセス寿命分蓄積する。`commit()` は完了を待つだけで成功を返す
    /// ため、`waitUntilCompleted()` 後にコマンドバッファの `status` を
    /// 確認しない場合 GPU 側の fault・OOM・discarded work が `Ok(())`
    /// として握り潰される（出力バッファの古い／不完全な内容を読む無言の
    /// 数値誤りにつながるため、[`MetalError::CommandBufferExecutionFailed`]
    /// として呼び出し元へ返す）。
    pub fn dispatch_sync<F>(&self, encode: F) -> Result<(), MetalError>
    where
        F: FnOnce(&ProtocolObject<dyn MTLComputeCommandEncoder>),
    {
        autoreleasepool(|_pool| {
            let cmd_buf = self
                .queue
                .commandBuffer()
                .ok_or(MetalError::CommandBufferCreation)?;
            let encoder = cmd_buf
                .computeCommandEncoder()
                .ok_or(MetalError::ComputeEncoderCreation)?;

            encode(&encoder);
            encoder.endEncoding();
            cmd_buf.commit();
            cmd_buf.waitUntilCompleted();

            if cmd_buf.status() == MTLCommandBufferStatus::Error {
                let message = cmd_buf
                    .error()
                    .map(|error| error.localizedDescription().to_string())
                    .unwrap_or_else(|| "no NSError attached to failed command buffer".to_string());
                return Err(MetalError::CommandBufferExecutionFailed { message });
            }
            Ok(())
        })
    }
}
