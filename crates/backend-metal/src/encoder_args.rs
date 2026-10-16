//! compute encoder へのバッファ・スカラー引数結線の共有ヘルパー
//! （イシュー #3137・監査記録 `docs/gpu-linalg-kernel-launch-unsafe-audit.md`
//! §3 の案 A）。
//!
//! # 役割と呼び出し文脈
//!
//! objc2-metal の `setBuffer_offset_atIndex`／`setBytes_length_atIndex` は
//! `unsafe fn` である。`rnn_cell.rs`・`sort.rs` に個別にあった同型の
//! ヘルパー（MT-1・MT-2）を本モジュールへ移設し、`linalg_kernel.rs` を含む
//! 呼び出し元が共有する（正味の新規 `unsafe` ブロックは 0）。各呼び出し元は
//! `ctx.dispatch_sync` 内で本ヘルパーを呼び、コマンドバッファ完了までの
//! バッファ生存を関数スコープで保証する。

use objc2::runtime::ProtocolObject;
use objc2_metal::MTLComputeCommandEncoder;

use crate::buffer::MtlBuffer;

/// バッファを `index` バッファスロットへ結線する（MT-1）。
///
/// SAFETY: `setBuffer_offset_atIndex` は生存中の `MTLBuffer` への参照を保持する
/// のみで即座に読み書きしない。呼び出し元がコマンドバッファの完了まで `buf` の
/// 生存を保証する（`dispatch_sync` なら関数スコープで生存、`encode` なら
/// `resources` で retain）。`index` は対象 MSL の `[[buffer(i)]]` と一致させる
/// 責務を呼び出し側が負う。
pub(crate) fn set_buffer_arg(
    encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    buf: &MtlBuffer,
    index: usize,
) {
    unsafe {
        encoder.setBuffer_offset_atIndex(Some(buf), 0, index);
    }
}

/// `values`（`constant uint&` 系スカラー引数）を `start_index` から連番の
/// バッファインデックスへ順に `setBytes_length_atIndex` で複製する（MT-2）。
///
/// SAFETY: `setBytes_length_atIndex` は指定ポインタから指定バイト数を即座に
/// 複製して保持しない。値は呼び出し元スライスの要素で、本呼び出し中は生存する。
/// バイト数（4）は対象 MSL の `constant uint&` 宣言と一致させる責務を
/// 呼び出し側が負う。
pub(crate) fn set_u32_args(
    encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    values: &[u32],
    start_index: usize,
) {
    for (offset, v) in values.iter().enumerate() {
        unsafe {
            encoder.setBytes_length_atIndex(
                std::ptr::NonNull::from(v).cast(),
                std::mem::size_of::<u32>(),
                start_index + offset,
            );
        }
    }
}

/// `u32` 1 個を `index` へ複製する（[`set_u32_args`] への委譲。`unsafe` を
/// 増やさない）。
pub(crate) fn set_u32_arg(
    encoder: &ProtocolObject<dyn MTLComputeCommandEncoder>,
    value: u32,
    index: usize,
) {
    set_u32_args(encoder, &[value], index);
}
