//! Metal バッファ確保・アップロード・readback（TASK-1.8a・#38）。
//!
//! [`crate::context::MetalContext`] が保持するデバイスを用いて
//! `MTLResourceOptions::StorageModeShared`（CPU/GPU 共有メモリ）で
//! `f32` バッファを確保する。TASK-1.8b（#39）以降のカーネルディスパッチ
//! はここで確保したバッファをエンコーダへ結線する（`raw()` 経由）。
//!
//! **StorageModeShared を選ぶ理由**（PoC-v2-4 実測。
//! `docs/spec/03-poc/poc-v2-4-metal-gemm/code/rust/src/metal_gemm.rs:173-175`）:
//! Apple Silicon は UMA（統合メモリ）のため、discrete GPU 向けの明示転送
//! （`MTLResourceOptions::StorageModePrivate` + blit エンコーダ）は不要と
//! 判断した。
//!
//! **移植元**: 同ファイルの `new_buffer_with_data` / `new_buffer_zeroed` /
//! `read_result`。PoC の `expect` 呼び出しを [`MetalError`] へ置き換え、
//! バイト長算出を `checked_mul` にして長さ 0・オーバーフローを FFI 呼び出し
//! 前に拒否する（外部フォーマット由来の形状がこの経路へ流入しうるための
//! 前段検証。OWASP A03・`.claude/rules/security.md`）。

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{MTLBuffer, MTLDevice, MTLResourceOptions};
use std::ffi::c_void;

use crate::context::MetalContext;
use crate::error::MetalError;

pub(crate) type MtlBuffer = ProtocolObject<dyn MTLBuffer>;

/// `f32` 要素を保持する Metal バッファのラッパー。
///
/// 確保時に記録した要素数（`len`）のみを [`MetalBuffer::read_to_vec`] で
/// 読み出す。これにより `contents()` から得る生ポインタの読み出し範囲が
/// 確保時に検証済みのバイト数を超えないことを保証する（REQ-8 の境界検査
/// 方針を FFI readback 経路にも適用）。
///
/// `Debug` を導出しているのは `tests/device_smoke.rs`（`#[ignore]`
/// 分離だが `cargo test` / `cargo clippy --all-targets` は非実行対象
/// テストも通常コンパイルするため）の `Result<MetalBuffer, MetalError>`
/// に対する `unwrap_err()` が `T: Debug` を要求するため（レビュー指摘。
/// 内部 `Retained<MtlBuffer>` はポインタ相当の識別子として出力される）。
#[derive(Debug)]
pub struct MetalBuffer {
    buffer: Retained<MtlBuffer>,
    len: usize,
}

/// `len` 要素分の `f32` バッファのバイト長を検証付きで算出する。
/// 長さ 0・オーバーフローを FFI 呼び出し前に拒否する
/// （`crate::error::MetalError` バリアント参照）。
fn checked_byte_len(len: usize) -> Result<usize, MetalError> {
    if len == 0 {
        return Err(MetalError::ZeroLengthAllocation);
    }
    len.checked_mul(std::mem::size_of::<f32>())
        .ok_or(MetalError::AllocationSizeOverflow { len })
}

impl MetalBuffer {
    /// `data` の内容を Metal バッファへアップロードして確保する。
    ///
    /// # Safety 境界（`unsafe` 使用箇所 1/3）
    /// `newBufferWithBytes_length_options` は `data` の先頭ポインタから
    /// `bytes_len` バイトを読み取って複製する。`bytes_len` は直前の
    /// `checked_byte_len(data.len())` により `data` の実バイト長と一致する
    /// ことを検証済みであり、`data` は本関数の呼び出し中生存しているため
    /// 範囲外読み出しは発生しない（PoC-v2-4 と同じ呼び出し形。
    /// `newBufferWithBytes_length_options` は入力を即座に複製し保持しない
    /// ため呼び出し後の `data` の生存は不要）。
    pub fn new_with_data(ctx: &MetalContext, data: &[f32]) -> Result<Self, MetalError> {
        let len = data.len();
        let bytes_len = checked_byte_len(len)?;

        // SAFETY: `&[f32]` の先頭ポインタは Rust のスライス仕様上つねに
        // 非 null（長さ 0 でも非 null が保証される）。`checked_byte_len`
        // が長さ 0 を事前拒否しているためこの分岐に到達する時点で
        // `data` は非空だが、非 null 性自体はスライス長に関わらず常に
        // 成立する不変条件であり `expect` で表現すべき失敗系ではない
        // （coding-rust.md: 本番経路で `unwrap`/`expect` を使わない）ため
        // `new_unchecked` を用いる。ポインタ・長さともに確保直前に検証済み。
        let ptr = unsafe { std::ptr::NonNull::new_unchecked(data.as_ptr() as *mut c_void) };

        // SAFETY: 上記コメント参照。ポインタ・長さともに確保直前に検証済み。
        let buffer = unsafe {
            ctx.device().newBufferWithBytes_length_options(
                ptr,
                bytes_len,
                MTLResourceOptions::StorageModeShared,
            )
        }
        .ok_or(MetalError::BufferAllocation { bytes: bytes_len })?;

        Ok(Self { buffer, len })
    }

    /// `len` 要素分（ゼロ初期化）の Metal バッファを確保する。
    /// GEMM の出力バッファ（C）等、ホスト側の初期値を持たない確保に使う
    /// （TASK-1.8b・#39 のディスパッチ経路から呼ばれる想定）。
    pub fn new_zeroed(ctx: &MetalContext, len: usize) -> Result<Self, MetalError> {
        let bytes_len = checked_byte_len(len)?;

        let buffer = ctx
            .device()
            .newBufferWithLength_options(bytes_len, MTLResourceOptions::StorageModeShared)
            .ok_or(MetalError::BufferAllocation { bytes: bytes_len })?;

        Ok(Self { buffer, len })
    }

    /// 確保済みの要素数。
    pub fn len(&self) -> usize {
        self.len
    }

    /// 要素数が 0 かどうか（`new_with_data` / `new_zeroed` は長さ 0 を
    /// 確保前に拒否するため、生存インスタンスでは常に `false` になる。
    /// clippy `len_without_is_empty` 対応のため公開する）。
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// TASK-1.8b（#39）のエンコーダ結線（`setBuffer_offset_atIndex` 等）
    /// から参照される生バッファへの参照。
    pub fn raw(&self) -> &MtlBuffer {
        &self.buffer
    }

    /// バッファの内容をホストへ読み出す。
    ///
    /// **呼び出し前提（イシュー #1017）**: 呼び出し元は、このバッファへの
    /// 書き込みが完了していることを `context.rs::MetalContext::
    /// synchronize` 等で保証済みであること。本メソッド自体はシグネチャに
    /// `&MetalContext` を持たないため同期を行わない（公開 API 非破壊。
    /// `.claude/rules/security.md`）。クレート内の呼び出し元
    /// （`memory.rs::download_inner`）は `read_to_vec` 直前に
    /// `self.context.synchronize()` を挟む契約とすることでこの前提を
    /// 満たす。
    ///
    /// # Safety 境界（`unsafe` 使用箇所 2/3）
    /// `contents()` は `StorageModeShared` バッファの CPU 可視アドレスを
    /// 返す（確保時に `MTLResourceOptions::StorageModeShared` を指定して
    /// いるため CPU から直接参照可能）。読み出す要素数は確保時に記録した
    /// `self.len`（確保時の `checked_byte_len` で検証済みのバイト数に
    /// 対応する要素数）に限定しており、確保バイト数を超えて読むことは
    /// ない。
    pub fn read_to_vec(&self) -> Vec<f32> {
        let ptr = self.buffer.contents();
        // SAFETY: 上記コメント参照。`self.len` は確保時に検証済みの要素数。
        let slice: &[f32] =
            unsafe { std::slice::from_raw_parts(ptr.as_ptr() as *const f32, self.len) };
        slice.to_vec()
    }

    /// バッファの内容を全要素 0 で上書きする。
    ///
    /// `fandhe_ai_tensor_core::pool::PooledMemory<MetalMemory>`（TASK-#201・
    /// REQ-14 14-3）がプールから再利用したバッファへ、`alloc_zeroed` の
    /// 「全要素 0」契約を再適用するために呼ぶ（`memory.rs::PoolZeroFill`
    /// 実装から呼ばれる想定。Metal 実機検証は #175 完了後）。
    ///
    /// # Safety 境界（`unsafe` 使用箇所 3/3。`read_to_vec` の書き込み版）
    /// `contents()` は `StorageModeShared` バッファの CPU 可視アドレスを
    /// 返す（確保時に `MTLResourceOptions::StorageModeShared` を指定して
    /// いるため CPU から直接書き込み可能。`new_with_data`/`new_zeroed`
    /// モジュールコメント参照）。書き込む要素数は確保時に記録した
    /// `self.len`（確保時の `checked_byte_len` で検証済みのバイト数に
    /// 対応する要素数）に限定しており、確保バイト数を超えて書くことは
    /// ない。呼び出し元（`PooledMemory::alloc_zeroed`）はプールから
    /// 取り出した直後・呼び出し元へ返す前の排他所有段階でのみ本メソッドを
    /// 呼ぶため、ホスト側からの他のエイリアスは存在しない。GPU 側との
    /// 同時アクセスについては、本メソッド自体は GPU 実行完了の待機を
    /// 行わない（`MTLCommandBuffer` の完了同期は呼び出し元が担う。
    /// イシュー #1017 以降は `memory.rs::PoolZeroFill::zero_fill` が本
    /// メソッド呼び出し直前に `context.rs::MetalContext::synchronize`
    /// を挟む契約に変わった）。プールへ返却されるバッファは
    /// `download`／GEMM 出力読み出しが完了した後に `Drop` されたもので
    /// あることが呼び出し元（`PooledMemory`）側の運用契約であり、本
    /// メソッドはその契約が守られている前提で安全である（契約破棄時の
    /// 挙動は他の `unsafe` 読み書き経路と同様に呼び出し元の責務とする）。
    pub(crate) fn zero_fill(&self) {
        let ptr = self.buffer.contents();
        // SAFETY: 上記コメント参照。`self.len` は確保時に検証済みの要素数。
        let slice: &mut [f32] =
            unsafe { std::slice::from_raw_parts_mut(ptr.as_ptr() as *mut f32, self.len) };
        slice.fill(0.0);
    }
}
