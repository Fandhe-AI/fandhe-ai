//! GEMM + bias + activation 融合カーネルの ROCm 実行型（イシュー #3442・分解案 R11）。
//!
//! 役割: [`RocmGemmBiasAct`] が `kernels_gemm_bias_act` のソースをコンパイルして保持し、
//! デバイス常駐オペランド（`TypedDeviceBuffer` + 要素オフセット）を受けて
//! `C = act(A·B + bias)` を計算する。`ops/gemm_derived.rs`（`gemm_resident_rhs`・
//! `gemm_resident_lhs`・`gemm_fp32_strict_into`・`linear_forward_device`）が thread_local
//! キャッシュ経由で呼ぶ。形状・ハンドル種別の検査と `Tensor`／`DeviceBuffer` の組み立ては
//! 呼び出し側、本層は起動前の fail-closed な範囲検査と起動だけを持つ。
//!
//! # REQ-8（二重化した境界検査）
//! カーネル側の guarded load・書き込みガードに加え、ホスト側で (1) `validate_gemm_dims`
//! （m・n・k と各積が `i32` に収まる）、(2) 4 オペランドすべての `offset + len <= バッファ長`
//! （[`operand_range`]。checked）、(3) `launch_geometry` の grid 上限、を起動前に確かめる。
//!
//! # unsafe
//! 本ファイルに `unsafe` はない。起動は監査済みの `KernelHost::run_geometry`（K1）だけで、
//! `Arg::Ptr` は `TypedDeviceBuffer::as_device_ptr` の値のみ、オフセットは `Arg::I64` の数値で渡す。

use crate::error::RocmError;
use crate::gemm::{launch_geometry, validate_gemm_dims};
use crate::kernels_gemm::GemmF32Tile;
use crate::kernels_gemm_bias_act::{GEMM_BIAS_ACT_SPEC, render_gemm_bias_act_source};
use crate::launch::{Arg, KernelHost};
use crate::target::GfxTarget;
use crate::typed_buffer::TypedDeviceBuffer;

const OP: &str = "gemm_bias_act";

/// オペランド = デバイスバッファと、その先頭からの要素オフセット。
pub(crate) type Operand<'a> = (&'a TypedDeviceBuffer<f32>, usize);

/// 1 回の起動の引数一式（`run` の引数が多すぎるため束ねる）。
#[derive(Clone, Copy)]
pub(crate) struct GemmCall<'a> {
    /// `[m, k]` の左辺。
    pub(crate) a: Operand<'a>,
    /// `[k, n]` の右辺。
    pub(crate) b: Operand<'a>,
    /// `Some` のとき `[n]` の bias。
    pub(crate) bias: Option<Operand<'a>>,
    /// 書き込み先（`m*n` 要素を上書きする）。
    pub(crate) c: Operand<'a>,
    /// 行数。
    pub(crate) m: usize,
    /// 列数。
    pub(crate) n: usize,
    /// 縮約長。
    pub(crate) k: usize,
    /// epilogue で ReLU を適用するか。
    pub(crate) act_relu: bool,
}

fn bad(detail: String) -> RocmError {
    RocmError::InvalidOpArgs { op: OP, detail }
}

/// `[offset, offset + len)` がバッファ長 `buf_len` に収まることを確かめ、カーネル引数用の
/// `i64` オフセットを返す。HIP に触れない純関数（単体テスト可能）。
///
/// `offset + len` の overflow・`buf_len` 超過・`i64` へ収まらないオフセットはすべて拒否する。
pub(crate) fn operand_range(
    what: &str,
    buf_len: usize,
    offset: usize,
    len: usize,
) -> Result<i64, RocmError> {
    let end = offset
        .checked_add(len)
        .ok_or_else(|| bad(format!("{what}: offset + len が usize を超えます")))?;
    if end > buf_len {
        return Err(bad(format!(
            "{what}: 範囲 [{offset}, {end}) がバッファ長 {buf_len} を超えます"
        )));
    }
    i64::try_from(offset).map_err(|_| bad(format!("{what}: offset が i64 を超えます")))
}

/// コンパイル済みの融合 GEMM カーネルを保持し、実行する。
pub struct RocmGemmBiasAct {
    tile: GemmF32Tile,
    host: KernelHost,
    /// bias なしのときに渡す 1 要素のダミー（null を渡さない。c や a を流用しない）。
    dummy_bias: TypedDeviceBuffer<f32>,
}

impl RocmGemmBiasAct {
    /// HIP/HIPRTC をロードし、既定タイルの融合カーネルを `target` 向けにコンパイルする。
    ///
    /// ROCm 非搭載では `LibraryUnavailable`、デバイスがなければ `NoDevice`（panic しない）。
    pub fn new(target: GfxTarget) -> Result<Self, RocmError> {
        let tile = GemmF32Tile::DEFAULT;
        let src = render_gemm_bias_act_source(&tile)?;
        let host = KernelHost::new(&src, target)?;
        let dummy_bias = {
            let _guard = host.select()?;
            TypedDeviceBuffer::<f32>::from_slice(host.runtime(), &[0.0])?
        };
        Ok(Self {
            tile,
            host,
            dummy_bias,
        })
    }

    /// ホストのスライスをデバイスへ転送する（空スライスでも有効なバッファを返す）。
    pub(crate) fn upload(&self, src: &[f32]) -> Result<TypedDeviceBuffer<f32>, RocmError> {
        let _guard = self.host.select()?;
        TypedDeviceBuffer::<f32>::from_slice(self.host.runtime(), src)
    }

    /// 未初期化の出力バッファを確保する（カーネルが全有効要素を上書きする用途）。
    pub(crate) fn alloc(&self, len: usize) -> Result<TypedDeviceBuffer<f32>, RocmError> {
        let _guard = self.host.select()?;
        TypedDeviceBuffer::<f32>::new(self.host.runtime(), len)
    }

    /// デバイスバッファ全体をホストへ読み戻す。
    pub(crate) fn download(&self, buf: &TypedDeviceBuffer<f32>) -> Result<Vec<f32>, RocmError> {
        let _guard = self.host.select()?;
        buf.to_vec()
    }

    /// `c[c_off .. c_off + m*n] = act(a·b (+ bias))` を計算し、完了まで待つ。
    ///
    /// `a` は `[m, k]`、`b` は `[k, n]`、`bias` は `[n]`（`Some` のとき）。出力は**上書き**で、
    /// 累積ではない。`m == 0 || n == 0` は起動せず `Ok`（範囲検査は行う）。`k == 0` は
    /// カーネルが `num_tiles = 0` で 0（＋bias／ReLU）を書く。
    pub(crate) fn run(&self, call: &GemmCall<'_>) -> Result<(), RocmError> {
        let GemmCall {
            a,
            b,
            bias,
            c,
            m,
            n,
            k,
            act_relu,
        } = *call;
        let too_big = |what: &str| bad(format!("{what} の積が usize を超えます"));
        let mk = m.checked_mul(k).ok_or_else(|| too_big("m*k"))?;
        let kn = k.checked_mul(n).ok_or_else(|| too_big("k*n"))?;
        let mn = m.checked_mul(n).ok_or_else(|| too_big("m*n"))?;
        validate_gemm_dims(&self.tile, mk, kn, m, n, k)?;
        let a_off = operand_range("a", a.0.len(), a.1, mk)?;
        let b_off = operand_range("b", b.0.len(), b.1, kn)?;
        let (bias_buf, bias_off, has_bias) = match bias {
            Some((buf, off)) => (buf, operand_range("bias", buf.len(), off, n)?, 1),
            None => (&self.dummy_bias, 0, 0),
        };
        let c_off = operand_range("c", c.0.len(), c.1, mn)?;
        if m == 0 || n == 0 {
            return Ok(());
        }
        let (grid, block) = launch_geometry(&self.tile, m, n)?;
        // 検証済みのため i32 に収まる。
        let args = [
            Arg::Ptr(a.0.as_device_ptr()),
            Arg::Ptr(b.0.as_device_ptr()),
            Arg::Ptr(bias_buf.as_device_ptr()),
            Arg::Ptr(c.0.as_device_ptr()),
            Arg::I64(a_off),
            Arg::I64(b_off),
            Arg::I64(bias_off),
            Arg::I64(c_off),
            Arg::I32(m as i32),
            Arg::I32(n as i32),
            Arg::I32(k as i32),
            Arg::I32(has_bias),
            Arg::I32(i32::from(act_relu)),
        ];
        let _guard = self.host.select()?;
        self.host
            .run_geometry(OP, &GEMM_BIAS_ACT_SPEC, grid, block, &args)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn operand_range_boundaries() {
        // 末尾ちょうどは可。
        assert_eq!(operand_range("x", 10, 6, 4).unwrap(), 6);
        assert_eq!(operand_range("x", 10, 0, 10).unwrap(), 0);
        assert_eq!(operand_range("x", 0, 0, 0).unwrap(), 0);
        // 1 つ超えは不可。
        assert!(operand_range("x", 10, 6, 5).is_err());
        assert!(operand_range("x", 10, 11, 0).is_err());
        // usize の overflow は不可。
        assert!(operand_range("x", usize::MAX, usize::MAX, 1).is_err());
        assert!(operand_range("x", 10, usize::MAX, 2).is_err());
    }

    #[test]
    #[cfg(target_pointer_width = "64")]
    fn operand_range_rejects_offset_beyond_i64() {
        let off = (i64::MAX as usize) + 1;
        assert!(operand_range("x", usize::MAX, off, 0).is_err());
        assert!(operand_range("x", usize::MAX, off - 1, 0).is_ok());
    }
}
