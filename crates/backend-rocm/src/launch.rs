//! 添字系カーネル（scan・gather/scatter・interpolate・sort・unique）の共通起動基盤
//! （イシュー #3439・分解案 R8）。
//!
//! 役割: 各演算族の実行型（`scan.rs` 等）が共有する (1) 1 次元グリッド計算 [`grid_1d`]、
//! (2) HIP/HIPRTC のロード→コンパイル→module 保持→起動までを持つ [`KernelHost`]、
//! (3) 型タグ付きカーネル引数 [`Arg`]／[`ArgKind`]／[`KernelSpec`] を提供する。
//!
//! 起動の `unsafe` は [`KernelHost::run_geometry`] の 1 箇所に集約する（[`KernelHost::run`]（1 次元）と
//! [`KernelHost::run_wave_blocks`]（1 block = 1 wave の行・チャネル単位。R6・#3437）の両方がここを通る）。`kernelParams` の個数・種別
//! （ポインタ／i32／i64／f32）は呼び出し側が渡す [`KernelSpec`] と実行時に突き合わせ、
//! spec とカーネルソース署名の一致は各 `kernels_*.rs` のテスト（`kernel_param_kinds`）で固定する。
//! 起動は null stream で行い、完了を待ってから戻る（同期 API。引数の生存期間問題を作らない）。

use crate::error::RocmError;
use crate::hip::{self, DeviceGuard, HipRuntime};
use crate::hiprtc::{HiprtcRuntime, compile_code_object};
use crate::kernel_source::CompileSource;
use crate::module::Module;
use crate::target::GfxTarget;
use std::ffi::c_void;
use std::sync::Arc;

/// 添字系カーネル共通のブロックあたりスレッド数。
///
/// 64 の倍数にして wave64（MI300X）・wave32 のどちらでも端数 wave を作らない
/// （`GemmF32Tile::new` と同じ検査意図）。カーネル側は `__launch_bounds__(BLOCK_DIM)` で
/// 同じ値を宣言する。
pub(crate) const INDEX_OPS_BLOCK_DIM: u32 = 256;

const _: () = assert!(INDEX_OPS_BLOCK_DIM.is_multiple_of(64));

/// grid／block の 3 次元寸法。
pub(crate) type Dim3 = (u32, u32, u32);

/// 1 次元グリッド `(grid, block)`。`total` は非 0 であること。
///
/// `grid.x` と `grid.x * block` が `u32` に収まることを checked で確かめる（CUDA 版の
/// `(total as u32).div_ceil` のような切り捨てを持ち込まない）。
pub(crate) fn grid_1d(
    op: &'static str,
    total: usize,
    block: u32,
) -> Result<(Dim3, Dim3), RocmError> {
    if total == 0 || block == 0 {
        return Err(RocmError::InvalidOpArgs {
            op,
            detail: "グリッド計算は total・block とも非 0 が必要です".into(),
        });
    }
    let gx = total.div_ceil(block as usize);
    let limit = (u32::MAX / block) as usize;
    if gx > limit {
        return Err(RocmError::SizeLimitExceeded {
            op,
            n: total,
            limit: limit * block as usize,
        });
    }
    Ok(((gx as u32, 1, 1), (block, 1, 1)))
}

/// 1 block = 1 wave で `blocks` 個を起動するときの `(grid, block)`（R6 正規化・#3437）。
///
/// `blocks` は 1..=`i32::MAX`（カーネル側の `row`／`ch` は `int` で、CUDA 版の
/// `rows <= i32::MAX` と同じ上限）、`block_dim` は wave 幅（32 か 64）に限る。0 や範囲外は
/// 型付きエラー（fail-closed）。
pub(crate) fn wave_block_grid(
    op: &'static str,
    blocks: usize,
    block_dim: u32,
) -> Result<(Dim3, Dim3), RocmError> {
    if block_dim != 32 && block_dim != 64 {
        return Err(RocmError::InvalidOpArgs {
            op,
            detail: format!("block_dim は wave 幅（32 か 64）のみ: {block_dim}"),
        });
    }
    if blocks == 0 {
        return Err(RocmError::InvalidOpArgs {
            op,
            detail: "blocks は非 0 が必要です".into(),
        });
    }
    if blocks > i32::MAX as usize {
        return Err(RocmError::SizeLimitExceeded {
            op,
            n: blocks,
            limit: i32::MAX as usize,
        });
    }
    Ok(((blocks as u32, 1, 1), (block_dim, 1, 1)))
}

/// ブロックあたりスレッド数の HIP 上限。
const MAX_THREADS_PER_BLOCK: u128 = 1024;

/// 起動前の geometry 検査（純関数。HIP 非搭載でも単体で検査できる）。
///
/// 各次元が非 0、`block` の総スレッド数が 1024 以下であること。範囲外は `InvalidOpArgs`。
pub(crate) fn validate_geometry(
    op: &'static str,
    grid: Dim3,
    block: Dim3,
) -> Result<(), RocmError> {
    let bad = |detail: &str| RocmError::InvalidOpArgs {
        op,
        detail: detail.into(),
    };
    if [grid.0, grid.1, grid.2, block.0, block.1, block.2].contains(&0) {
        return Err(bad("grid・block の各次元は非 0 が必要です"));
    }
    let threads = u128::from(block.0) * u128::from(block.1) * u128::from(block.2);
    if threads > MAX_THREADS_PER_BLOCK {
        return Err(bad("block の総スレッド数が 1024 を超えます"));
    }
    Ok(())
}

/// カーネル引数の種別（ソース署名との照合用）。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ArgKind {
    /// デバイスポインタ。
    Ptr,
    /// C の `int`。
    I32,
    /// C の `long long`。
    I64,
    /// C の `float`。
    F32,
}

/// 起動時に渡す引数値（種別つき）。
#[derive(Debug, Clone, Copy)]
pub(crate) enum Arg {
    /// デバイスポインタ（`TypedDeviceBuffer::as_device_ptr`）。
    Ptr(*mut c_void),
    /// `int`。
    I32(i32),
    /// `long long`。
    I64(i64),
    /// `float`。
    F32(f32),
}

impl Arg {
    fn kind(&self) -> ArgKind {
        match self {
            Self::Ptr(_) => ArgKind::Ptr,
            Self::I32(_) => ArgKind::I32,
            Self::I64(_) => ArgKind::I64,
            Self::F32(_) => ArgKind::F32,
        }
    }
}

/// カーネルの名前と引数署名（ソースの宣言と一致させる定数）。
#[derive(Debug, Clone, Copy)]
pub(crate) struct KernelSpec {
    /// `extern "C"` のエントリ名。
    pub(crate) name: &'static str,
    /// 引数の種別列。
    pub(crate) args: &'static [ArgKind],
}

/// ソースをコンパイルして module を保持し、spec 経由でカーネルを起動する。
pub(crate) struct KernelHost {
    // Drop 順: module（アンロード）→ rt。module が `Arc` を持つため順序に依らず安全。
    module: Module,
    rt: Arc<HipRuntime>,
    /// module・バッファ・同期の対象デバイス ID（`new` が選択したもの）。
    device: i32,
}

impl KernelHost {
    /// HIP/HIPRTC をロードし、`src` を `target` 向けにコンパイル・ロードする。
    ///
    /// ROCm 非搭載では `LibraryUnavailable`、デバイスがなければ `NoDevice`（panic しない）。
    pub(crate) fn new(src: &impl CompileSource, target: GfxTarget) -> Result<Self, RocmError> {
        let rt = Arc::new(hip::load()?);
        let hiprtc = HiprtcRuntime::load()?;
        if rt.device_count()? < 1 {
            return Err(RocmError::NoDevice);
        }
        let device = 0;
        let module = {
            let _guard = rt.select_device(device)?;
            let code = compile_code_object(&hiprtc, src, target)?;
            Module::load(Arc::clone(&rt), code, device)?
        };
        Ok(Self { module, rt, device })
    }

    /// バッファ確保用のランタイム。
    pub(crate) fn runtime(&self) -> Arc<HipRuntime> {
        Arc::clone(&self.rt)
    }

    /// module の所有デバイスを選ぶガード。確保・起動・読み戻しの間保持する
    /// （バッファより先に宣言し、バッファの Drop 後に復元されるようにする）。
    pub(crate) fn select(&self) -> Result<DeviceGuard<'_>, RocmError> {
        self.rt.select_device(self.device)
    }

    /// `spec` のカーネルをちょうど `num_blocks` ブロック（各 [`INDEX_OPS_BLOCK_DIM`] スレッド）で
    /// 起動する。全軸縮約の 1 段目（grid-stride ループでブロック数がデータ量より小さい）用の薄い
    /// ラッパーで、起動の `unsafe` は [`Self::run`] の 1 か所のまま増やさない。
    /// `num_blocks == 0` は拒否する。
    pub(crate) fn run_blocks(
        &self,
        op: &'static str,
        spec: &KernelSpec,
        num_blocks: u32,
        args: &[Arg],
    ) -> Result<(), RocmError> {
        let total = (num_blocks as usize)
            .checked_mul(INDEX_OPS_BLOCK_DIM as usize)
            .filter(|_| num_blocks > 0)
            .ok_or_else(|| RocmError::InvalidOpArgs {
                op,
                detail: format!("ブロック数 {num_blocks} は 1 以上である必要があります"),
            })?;
        self.run(op, spec, total, args)
    }

    /// `spec` のカーネルを `total` スレッド（1 次元・block = [`INDEX_OPS_BLOCK_DIM`]）で起動し、
    /// 完了まで待つ。
    ///
    /// `args` の個数・種別が `spec` と一致しなければ `InvalidOpArgs`。`total == 0` は拒否
    /// （呼び出し側が 0 要素を早期リターンする契約）。
    pub(crate) fn run(
        &self,
        op: &'static str,
        spec: &KernelSpec,
        total: usize,
        args: &[Arg],
    ) -> Result<(), RocmError> {
        let (grid, block) = grid_1d(op, total, INDEX_OPS_BLOCK_DIM)?;
        self.run_geometry(op, spec, grid, block, args)
    }

    /// `spec` のカーネルを「`blocks` 個の block × 1 wave（`block_dim` = 32 か 64）」で起動し、
    /// 完了まで待つ（行・チャネル単位の正規化カーネル用。R6・#3437）。寸法は
    /// [`wave_block_grid`] が検証する。
    pub(crate) fn run_wave_blocks(
        &self,
        op: &'static str,
        spec: &KernelSpec,
        blocks: usize,
        block_dim: u32,
        args: &[Arg],
    ) -> Result<(), RocmError> {
        let (grid, block) = wave_block_grid(op, blocks, block_dim)?;
        self.run_geometry(op, spec, grid, block, args)
    }

    /// `spec` のカーネルを呼び出し側が導出した `(grid, block)` で起動し、完了まで待つ。
    ///
    /// 「1 block = 1 wave = 1 行」の softmax 系のように 1 次元・block 256 固定に合わない起動形
    /// 向け（イシュー #3435・R4）。`unsafe` の起動箇所は [`Self::run`] と共通のここ 1 か所
    /// （監査行 K1）。geometry は [`validate_geometry`] が起動前に検証する。
    pub(crate) fn run_geometry(
        &self,
        op: &'static str,
        spec: &KernelSpec,
        grid: Dim3,
        block: Dim3,
        args: &[Arg],
    ) -> Result<(), RocmError> {
        if args.len() != spec.args.len() || args.iter().zip(spec.args).any(|(a, k)| a.kind() != *k)
        {
            return Err(RocmError::InvalidOpArgs {
                op,
                detail: format!("{} の引数が署名と一致しません", spec.name),
            });
        }
        validate_geometry(op, grid, block)?;
        let func = self.module.function(spec.name)?;
        // 値をローカルへコピーし、その各要素へのポインタを `kernelParams` として渡す。
        let mut owned: Vec<Arg> = args.to_vec();
        let mut params: Vec<*mut c_void> = owned
            .iter_mut()
            .map(|a| match a {
                Arg::Ptr(p) => (p as *mut *mut c_void).cast(),
                Arg::I32(v) => (v as *mut i32).cast(),
                Arg::I64(v) => (v as *mut i64).cast(),
                Arg::F32(v) => (v as *mut f32).cast(),
            })
            .collect();
        // SAFETY: (U5) `params` の個数・種別は `spec`（ソース署名と同一であることを各
        // `kernels_*.rs` のテストで固定）と上で照合済みで、指す `owned` は下の `synchronize`
        // 完了まで本スコープで生存する。grid／block は呼び出し側が導出し、上の
        // `validate_geometry` が各次元の非 0 と block 総スレッド数の上限を検証済み。範囲外
        // アクセスは各カーネル内の手動境界チェック（REQ-8）が防ぐ。
        unsafe { func.launch(&self.rt, grid, block, &mut params)? };
        self.rt.synchronize()
    }
}

/// テスト専用: 1 次元グリッドの全スレッドの `gid`（`blockIdx.x * blockDim.x + threadIdx.x`）を
/// 順に `f` へ渡す。`gid >= total` のスレッドも渡す（端数ブロックのカーネル内ガードを
/// モデル側でも通すため）。`total == 0` は何も呼ばない（実機では起動しない契約）。
#[cfg(test)]
pub(crate) fn for_each_thread_gid(total: usize, mut f: impl FnMut(i64)) {
    if total == 0 {
        return;
    }
    let (grid, block) = grid_1d("model", total, INDEX_OPS_BLOCK_DIM).unwrap();
    for b in 0..i64::from(grid.0) {
        for t in 0..i64::from(block.0) {
            f(b * i64::from(block.0) + t);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn block_dim_is_wave_multiple() {
        assert_eq!(INDEX_OPS_BLOCK_DIM % 64, 0);
    }

    #[test]
    fn grid_covers_total_without_truncation() {
        let (g, b) = grid_1d("t", 1, 256).unwrap();
        assert_eq!((g, b), ((1, 1, 1), (256, 1, 1)));
        assert_eq!(grid_1d("t", 256, 256).unwrap().0, (1, 1, 1));
        assert_eq!(grid_1d("t", 257, 256).unwrap().0, (2, 1, 1));
        // u32 を超える total でも切り捨てずに覆う（超過は拒否）。
        let big = (u32::MAX as usize / 256) * 256;
        assert_eq!(grid_1d("t", big, 256).unwrap().0.0 as usize * 256, big);
    }

    #[test]
    fn run_blocks_total_maps_to_exact_grid() {
        // run_blocks(n) が run へ渡す total（n * BLOCK_DIM）から grid_1d がちょうど n ブロックを返す。
        for n in [1u32, 2, 4, 1024] {
            let total = n as usize * INDEX_OPS_BLOCK_DIM as usize;
            assert_eq!(
                grid_1d("t", total, INDEX_OPS_BLOCK_DIM).unwrap().0,
                (n, 1, 1)
            );
        }
    }

    #[test]
    fn grid_rejects_zero_and_overflow() {
        assert!(grid_1d("t", 0, 256).is_err());
        assert!(grid_1d("t", 1, 0).is_err());
        let limit = (u32::MAX / 256) as usize * 256;
        assert!(grid_1d("t", limit + 1, 256).is_err());
        assert!(grid_1d("t", usize::MAX, 256).is_err());
    }

    #[test]
    fn wave_block_grid_validates_blocks_and_block_dim() {
        assert_eq!(
            wave_block_grid("t", 3, 64).unwrap(),
            ((3, 1, 1), (64, 1, 1))
        );
        assert_eq!(
            wave_block_grid("t", i32::MAX as usize, 32).unwrap().0,
            (i32::MAX as u32, 1, 1)
        );
        assert!(matches!(
            wave_block_grid("t", 0, 64),
            Err(RocmError::InvalidOpArgs { .. })
        ));
        assert!(matches!(
            wave_block_grid("t", i32::MAX as usize + 1, 64),
            Err(RocmError::SizeLimitExceeded { .. })
        ));
        for bad in [0, 1, 16, 48, 128, 256, 1024] {
            assert!(wave_block_grid("t", 1, bad).is_err(), "{bad}");
        }
    }

    #[test]
    fn geometry_validation() {
        assert!(validate_geometry("t", (1, 1, 1), (64, 1, 1)).is_ok());
        assert!(validate_geometry("t", (65535, 1, 1), (1024, 1, 1)).is_ok());
        assert!(validate_geometry("t", (8, 8, 8), (16, 8, 8)).is_ok());
        for (g, b) in [
            ((0, 1, 1), (64, 1, 1)),
            ((1, 0, 1), (64, 1, 1)),
            ((1, 1, 1), (0, 1, 1)),
            ((1, 1, 1), (64, 0, 1)),
            ((1, 1, 1), (1025, 1, 1)),
            ((1, 1, 1), (32, 32, 2)),
            ((1, 1, 1), (u32::MAX, u32::MAX, u32::MAX)),
        ] {
            assert!(validate_geometry("t", g, b).is_err(), "{g:?} {b:?}");
        }
    }

    #[test]
    fn arg_kind_matches_variant() {
        assert_eq!(Arg::I32(1).kind(), ArgKind::I32);
        assert_eq!(Arg::I64(1).kind(), ArgKind::I64);
        assert_eq!(Arg::F32(1.0).kind(), ArgKind::F32);
        assert_eq!(Arg::Ptr(std::ptr::null_mut()).kind(), ArgKind::Ptr);
    }
}
