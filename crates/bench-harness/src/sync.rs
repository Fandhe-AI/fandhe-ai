//! バックエンド間で統一する「同期方式」の契約と 3 バックエンド実装（TASK-8.1b）。
//!
//! PoC-5 実測により、ホスト転送を伴う同期と伴わない同期が混在すると
//! 計測値が約 101% ⇔ 約 115〜119% まで変動することが分かっている
//! （`docs/spec/04-requirements.md:288`）。REQ-8 の受け入れ基準
//! （`docs/spec/04-requirements.md:188`）は「ホスト転送を伴わない完了待ち」
//! への統一を求めており、本モジュールはその契約を [`SyncPoint`] trait として
//! 明文化し、CPU・CUDA・Metal 向けの実装を提供する。
//!
//! 計測コア（TASK-8.1a・#27）は計測区間の終端でここの実装を呼び出し、
//! 「カーネル起動 → 完了待ち」までを計測に含め、結果のホスト転送
//! （`memcpy_dtoh` 等）は計測区間の外に置く（PoC-v2-3
//! `docs/spec/03-poc/poc-v2-3-cuda-gemm/code/rust/src/cuda/mod.rs:228-240` と
//! 同方式）。この前提（計測区間内にホスト転送を置かない）を崩す呼び出し方は
//! 同期方式統一の意図を無効化するため、[`SyncPoint::wait_idle`] の呼び出し側は
//! 遵守すること。
//!
//! バックエンド抽象層 `BackendOps`（`docs/public-api-design.md` 4 章）は
//! TASK-1.9（Phase 3）まで未実装のため、本モジュールは backend クレートに
//! 依存せず自己完結で実装する（統合は TASK-1.9 側に引き継ぐ。イシュー #28
//! 計画の「スコープ外」節を参照）。

use std::fmt;

/// 完了待ちの失敗を呼び出し元へ伝える型付きエラー。
///
/// CUDA（`cudarc::driver::result::DriverError`）・Metal（`NSError`）の
/// バックエンド固有エラーを、計測コア側がバックエンドを問わず統一的に
/// 扱えるようラップする（本番経路で `unwrap`/`expect` を使わない方針。
/// `.claude/rules/coding-rust.md`）。
#[derive(Debug)]
pub enum SyncError {
    /// CUDA `stream.synchronize()` の失敗。メッセージは `cudarc` 側のエラー文字列。
    Cuda(String),
    /// Metal コマンドバッファが `MTLCommandBufferStatus::Error` で完了した場合。
    /// メッセージは `NSError.localizedDescription`（取得できない場合は既定文言）。
    Metal(String),
}

impl fmt::Display for SyncError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SyncError::Cuda(msg) => write!(f, "CUDA 同期に失敗しました: {msg}"),
            SyncError::Metal(msg) => write!(f, "Metal 同期に失敗しました: {msg}"),
        }
    }
}

impl std::error::Error for SyncError {}

/// 「ホスト転送を伴わない完了待ち」を表す統一契約（REQ-8）。
///
/// 実装（[`CpuSync`]・[`CudaStreamSync`]・[`MetalCommandBufferSync`]）は
/// いずれも `wait_idle` の呼び出しでバックエンド上の未完了処理が完了する
/// ことのみを保証し、結果データのホスト側 `Vec` への転送は行わない。
/// 計測コアはカーネル起動直後に `wait_idle` を呼び、その後にホスト転送を
/// 行うことで計測区間からホスト転送を除外する（モジュールドキュメント参照）。
pub trait SyncPoint {
    /// バックエンド上の未完了処理が完了するまでホスト側をブロックする。
    fn wait_idle(&self) -> Result<(), SyncError>;
}

/// CPU バックエンド向けの同期実装。
///
/// CPU は（`rayon` によるスレッドプール内の同期を除き）ベンチ計測の対象と
/// なるカーネル呼び出しに対してバックグラウンド実行を持たないため、
/// `wait_idle` は no-op として REQ-8 の「該当なし」を明示する
/// （`docs/spec/04-requirements.md:188`）。
#[derive(Debug, Default, Clone, Copy)]
pub struct CpuSync;

impl SyncPoint for CpuSync {
    fn wait_idle(&self) -> Result<(), SyncError> {
        Ok(())
    }
}

/// CUDA バックエンド向けの同期実装。
///
/// `stream.synchronize()` はホスト-デバイス間のメモリ転送を行わず、
/// ストリームに投入済みの全カーネルの完了のみを待つ（PoC-v2-3 実測方式。
/// `docs/spec/03-poc/poc-v2-3-cuda-gemm/code/rust/src/cuda/mod.rs:228-240`）。
/// `cudarc` は `dynamic-loading` feature を使うため、`CudaStream` の生成自体は
/// CUDA toolkit 搭載環境でのみ成立するが、本型・`SyncPoint` 実装のコンパイルは
/// toolkit 非搭載環境でも成立する（無条件依存。`.claude/rules/deps-policy.md`）。
pub struct CudaStreamSync {
    stream: std::sync::Arc<cudarc::driver::CudaStream>,
}

impl CudaStreamSync {
    /// 呼び出し元（計測コア）がカーネル起動に使った `CudaStream` をそのまま渡す。
    /// ストリームの所有権は呼び出し元側に残るため `Arc` で共有する。
    pub fn new(stream: std::sync::Arc<cudarc::driver::CudaStream>) -> Self {
        Self { stream }
    }
}

impl SyncPoint for CudaStreamSync {
    fn wait_idle(&self) -> Result<(), SyncError> {
        // `DriverError` は `Display` を実装しないため `Debug` 経由で文字列化する
        // （PoC-v2-3 でも同様に `CUresult` の生値を Debug で扱っている）。
        self.stream
            .synchronize()
            .map_err(|e| SyncError::Cuda(format!("{e:?}")))
    }
}

/// Metal バックエンド向けの同期実装。
///
/// macOS 専用（`objc2`/`objc2-metal` は `cfg(target_os = "macos")` 限定。
/// `.claude/rules/deps-policy.md`）。commit 済みコマンドバッファの
/// `waitUntilCompleted()` を呼び、結果バッファの読み出し（ホスト転送）は
/// 計測区間の外（`wait_idle` 呼び出し後）で行う（PoC-v2-5 実測方式。
/// `docs/spec/03-poc/poc-v2-5-backend-numeric-parity/code/rust/src/metal_backend.rs:187-188`）。
#[cfg(target_os = "macos")]
pub struct MetalCommandBufferSync {
    command_buffer:
        objc2::rc::Retained<objc2::runtime::ProtocolObject<dyn objc2_metal::MTLCommandBuffer>>,
}

#[cfg(target_os = "macos")]
impl MetalCommandBufferSync {
    /// 呼び出し元（計測コア）が `commit()` 済みのコマンドバッファを渡す。
    /// `wait_idle` 側では commit を行わない（commit 前に渡された場合の
    /// 挙動は Metal 側の未定義動作となるため、呼び出し元が commit 責務を持つ）。
    pub fn new(
        command_buffer: objc2::rc::Retained<
            objc2::runtime::ProtocolObject<dyn objc2_metal::MTLCommandBuffer>,
        >,
    ) -> Self {
        Self { command_buffer }
    }
}

#[cfg(target_os = "macos")]
impl SyncPoint for MetalCommandBufferSync {
    fn wait_idle(&self) -> Result<(), SyncError> {
        use objc2_metal::{MTLCommandBuffer, MTLCommandBufferStatus};

        // `waitUntilCompleted` はコマンドバッファが Completed または Error の
        // いずれかの終端状態に達するまでホスト側をブロックする（Apple 公式仕様）。
        // 戻り値を持たないため、完了後に `status()` を見てエラーを判別する。
        self.command_buffer.waitUntilCompleted();

        if self.command_buffer.status() == MTLCommandBufferStatus::Error {
            let message = self
                .command_buffer
                .error()
                .map(|e| e.localizedDescription().to_string())
                .unwrap_or_else(|| "unknown Metal command buffer error".to_string());
            return Err(SyncError::Metal(message));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cpu_sync_is_noop_success() {
        let sync = CpuSync;
        assert!(sync.wait_idle().is_ok());
    }

    #[test]
    fn cpu_sync_default_and_copy() {
        // `Default`/`Copy` を落とさないことを確認する（ゼロコスト no-op である
        // ことを型レベルで担保するための最小テスト）。
        let a: CpuSync = Default::default();
        let b = a;
        assert!(a.wait_idle().is_ok());
        assert!(b.wait_idle().is_ok());
    }
}
