//! 平均二乗誤差（MSE）融合カーネルの起動 API（NVRTC コンパイル・保持・
//! 実行。イシュー #1045・親イシュー #1043）。
//!
//! `elementwise.rs::CudaElementwise`・`sgd.rs::CudaSgd` と同じ構成方針を
//! 踏襲する: [`CudaMse::new`] が `CudaDevice` から 3 カーネル
//! （`kernels_mse.rs`）を NVRTC コンパイルして保持し、以降は
//! [`CudaMse::run_mse_loss_f32`]／[`CudaMse::run_mse_backward_f32`] へ
//! ホスト側スライスを渡すだけで H2D → 起動 → 同期 → D2H を内部で完結
//! できる。`ops.rs::CudaBackendOps::mse_loss`／`mse_loss_backward` から
//! `BackendOps` の実装として呼ばれる。
//!
//! **target のデバイス再利用（イシュー #3203）**: 学習 1 step では forward と
//! backward が同じ `target` をそれぞれ H2D していた。[`CudaMse`] は直前に
//! 送った `target` のホスト側内容とデバイスバッファを 1 組だけ保持し、
//! **`f32::to_bits` 単位で完全一致したときだけ**H2D を省いて再利用する
//! （`==` は `-0.0 == 0.0` で符号ビットを取り違えるため使わない）。
//! キャッシュは `CudaMse`（= context 単位。`context_cache::cached_mse`）
//! ごとに持つので、別 ordinal・別 context と共有されない。ミス時は先に
//! 無効化してから書き込み、成功したときだけ登録する（H2D 失敗で古い鍵と
//! 半端な内容の組を残さない）。カーネル・起動構成は不変のため出力は
//! キャッシュの有無でビット同一。入力・ターゲット常駐ハンドル（#3249）や
//! 1 回同期の学習 step（#3252）とは独立の、公開 API を変えない最適化。

use std::sync::{Arc, Mutex};

use cudarc::driver::{CudaFunction, CudaSlice, CudaStream, LaunchConfig, PushKernelArg};

use crate::context_cache;
use crate::device::CudaDevice;
use crate::error::CudaError;
use crate::kernels_mse::{self, MSE_BLOCK_DIM, MSE_MAX_BLOCKS};
use crate::memory::readback;
use crate::nvrtc::compile_ptx;
use crate::pool::CudaAllocator;
use crate::warp_geometry::WarpGeometry;

/// `numel`（`pred`／`target`／`dpred` の要素数）が `i32::MAX` に収まる
/// ことを検証する（`elementwise.rs::validate_elementwise_len` と同じ
/// 理由。カーネル引数 `int numel` は C の 32bit 符号付き整数のため）。
pub(crate) fn validate_mse_len(len: usize) -> Result<(), CudaError> {
    if len > i32::MAX as usize {
        return Err(CudaError::InvalidElementwiseShape {
            detail: format!("mse_loss numel must fit in i32 (kernel argument type): numel={len}"),
        });
    }
    Ok(())
}

/// `pred_len`／`target_len` の一致と `i32::MAX` 上限の両方を検証する
/// （`elementwise.rs::validate_elementwise_binary_dims` と同じ構成）。
///
/// `run_mse_loss_f32`／`run_mse_backward_f32` は `CudaMse` の公開メソッド
/// であり、`ops.rs` の事前検証を経ない直接呼び出しに対しても長さ不一致を
/// `panic!`（`assert_eq!`）ではなく型付きエラーとして返す契約とする
/// （AGENTS.md「本番経路の panic 禁止」）。
pub(crate) fn validate_mse_binary_len(pred_len: usize, target_len: usize) -> Result<(), CudaError> {
    if pred_len != target_len {
        return Err(CudaError::InvalidElementwiseShape {
            detail: format!("mse length mismatch: pred_len={pred_len}, target_len={target_len}"),
        });
    }
    validate_mse_len(pred_len)
}

/// forward 1 段目（`mse_partial_f32`）の起動ブロック数を決定する。
/// `min(ceil_div(numel, MSE_BLOCK_DIM), MSE_MAX_BLOCKS)`（`kernels_mse.rs`
/// 冒頭コメント「forward の 2 段構成」の契約）。`numel > 0` を前提とする
/// （`numel == 0` は呼び出し元がカーネル起動自体を回避する）。
fn mse_num_blocks(numel: u32) -> u32 {
    numel.div_ceil(MSE_BLOCK_DIM).min(MSE_MAX_BLOCKS)
}

/// target キャッシュの対象とする最大要素数（`1 << 20` = 4 MiB）。
/// 学習の典型形状（64×10 等）より十分大きく、永続保持するホスト・デバイス
/// 各 1 本と O(n) のビット比較のコストに上限を与える。超える target は
/// キャッシュせず従来どおり一時バッファへ H2D する。
pub(crate) const MSE_TARGET_CACHE_MAX_NUMEL: usize = 1 << 20;

/// 直前に H2D した target のホスト側内容（鍵）とデバイスバッファ（値）。
/// 不変条件: `host.len()` とデバイスバッファの要素数は常に等しい。
struct CachedTarget<B> {
    host: Vec<f32>,
    dev: B,
}

/// `cached` と `target` が長さ・全要素のビット表現で一致するか。
/// `==` ではなく `to_bits` で比べる（`-0.0`／NaN payload を区別し、
/// カーネルが読むビット列が同一であることだけを再利用条件にする）。
pub(crate) fn target_bits_equal(cached: &[f32], target: &[f32]) -> bool {
    cached.len() == target.len()
        && cached
            .iter()
            .zip(target)
            .all(|(a, b)| a.to_bits() == b.to_bits())
}

/// キャッシュ `slot` を使って `target` のデバイスバッファを解決し、`use_dev`
/// に渡す（CUDA 非依存のジェネリック実装。状態遷移を単体テストするため
/// デバイス型 `B` と転送を差し込み式にしている）。`use_dev` の実行中も
/// `slot` の排他借用が続くので、他スレッドが同じバッファを書き換える
/// ことはない。
///
/// - ヒット（ビット一致）: 転送なしで `use_dev(&dev)`。
/// - ミス: 先に `slot` を空にし、同長なら `overwrite` で上書き、違えば
///   `upload` で新規確保。成功時のみ鍵を更新して登録する。失敗時は空の
///   まま `Err` を返す。
fn resolve_target<B, E, R>(
    slot: &mut Option<CachedTarget<B>>,
    target: &[f32],
    upload: impl FnOnce(&[f32]) -> Result<B, E>,
    overwrite: impl FnOnce(&mut B, &[f32]) -> Result<(), E>,
    use_dev: impl FnOnce(&B) -> Result<R, E>,
) -> Result<R, E> {
    if let Some(c) = slot.as_ref()
        && target_bits_equal(&c.host, target)
    {
        return use_dev(&c.dev);
    }
    let entry = match slot.take() {
        Some(mut c) if c.host.len() == target.len() => {
            overwrite(&mut c.dev, target)?;
            c.host.copy_from_slice(target);
            c
        }
        _ => CachedTarget {
            dev: upload(target)?,
            host: target.to_vec(),
        },
    };
    use_dev(&slot.insert(entry).dev)
}

/// MSE 3 カーネル（forward 2 段・backward 1 段。いずれも f32）の
/// コンパイル済みハンドルを保持する。
pub struct CudaMse {
    stream: Arc<CudaStream>,
    /// 構築元 `CudaDevice` の ordinal（イシュー #1349・codex-review P0
    /// 指摘・PR #1390 是正）。`Self::with_driver_call` が
    /// `context_cache::with_driver_call` を呼ぶ際のキーとして使う
    /// （`gemm.rs::CudaGemm::ordinal` と同じ役割）。
    ordinal: usize,
    allocator: Arc<CudaAllocator>,
    partial_f32: CudaFunction,
    finalize_f32: CudaFunction,
    backward_f32: CudaFunction,
    /// 直前の target のデバイス再利用キャッシュ（イシュー #3203。モジュール
    /// doc 参照）。`with_driver_call` 内でのみロックを取る。
    target_cache: Mutex<Option<CachedTarget<CudaSlice<f32>>>>,
}

impl CudaMse {
    /// `device` 上で MSE 3 カーネルを NVRTC コンパイルし保持するハンドル
    /// を構築する（`elementwise.rs::CudaElementwise::new` と同一手順）。
    pub fn new(device: &CudaDevice) -> Result<Self, CudaError> {
        let arch = device.arch();
        // warp 幅依存定数はデバイス属性から導出して数値 `#define` で注入する
        // （イシュー #2126。`kernels_mse.rs` 冒頭・`warp_geometry.rs` 参照）。
        let geom = WarpGeometry::for_cuda(device)?;

        let partial_ptx = compile_ptx(&kernels_mse::render_mse_partial_f32(geom)?, arch)?;
        let finalize_ptx = compile_ptx(&kernels_mse::render_mse_finalize_f32(geom)?, arch)?;
        let backward_ptx = compile_ptx(kernels_mse::MSE_BACKWARD_F32, arch)?;

        let partial_f32 = device
            .context()
            .load_module(partial_ptx)?
            .load_function("mse_partial_f32")?;
        let finalize_f32 = device
            .context()
            .load_module(finalize_ptx)?
            .load_function("mse_finalize_f32")?;
        let backward_f32 = device
            .context()
            .load_module(backward_ptx)?
            .load_function("mse_backward_f32")?;

        let allocator = context_cache::cached_allocator(device)?;

        Ok(Self {
            stream: device.stream().clone(),
            ordinal: device.ordinal(),
            allocator,
            partial_f32,
            finalize_f32,
            backward_f32,
            target_cache: Mutex::new(None),
        })
    }

    /// `target` のデバイスバッファを（可能なら再利用して）`f` に渡す。
    /// `run_mse_loss_f32`／`run_mse_backward_f32` の `with_driver_call`
    /// 内から呼ぶ。ロックは `f`（カーネル起動と readback による同期まで）
    /// の終了まで保持する。上限超過・Mutex poison のときはキャッシュを
    /// 使わず一時バッファへ H2D する（純粋な最適化なので縮退しても出力は
    /// 同一。panic・unwrap はしない）。
    fn with_target_dev<R>(
        &self,
        target: &[f32],
        f: impl FnOnce(&CudaSlice<f32>) -> Result<R, CudaError>,
    ) -> Result<R, CudaError> {
        if target.len() <= MSE_TARGET_CACHE_MAX_NUMEL
            && let Ok(mut guard) = self.target_cache.lock()
        {
            return resolve_target(
                &mut guard,
                target,
                |t| Ok(self.stream.clone_htod(t)?),
                |dev, t| Ok(self.stream.memcpy_htod(t, dev)?),
                f,
            );
        }
        let dev = self.stream.clone_htod(target)?;
        f(&dev)
    }

    /// `CudaMse` の driver 呼び出し（H2D 転送・カーネル起動・D2H
    /// readback）を CUDA Graph capture 排他へ参加させる共通ヘルパー
    /// （`elementwise.rs::CudaElementwise::with_driver_call` と同じ設計。
    /// codex-review P0 指摘対応・PR #1390 是正）。
    fn with_driver_call<T>(
        &self,
        f: impl FnOnce() -> Result<T, CudaError>,
    ) -> Result<T, CudaError> {
        context_cache::with_driver_call(self.ordinal, f)
    }

    /// forward: `reduction(Σ(pred[i]−target[i])²)`。`pred.len() ==
    /// target.len()` は呼び出し元（`ops.rs`）が検証済みの契約。
    /// `numel == 0` はカーネル起動を回避し `0.0` を返す（`Mean`／`Sum`
    /// いずれも空和の契約。`backend-cpu::mse` と同じ）。
    pub fn run_mse_loss_f32(
        &self,
        pred: &[f32],
        target: &[f32],
        factor: f32,
    ) -> Result<f32, CudaError> {
        validate_mse_binary_len(pred.len(), target.len())?;
        let numel = pred.len();
        if numel == 0 {
            return Ok(0.0);
        }

        // codex-review P0 指摘対応（PR #1390 是正）: `Self::with_driver_call`
        // で本体（H2D・確保・起動・readback）全体を capture 排他へ参加
        // させる（`elementwise.rs::run_binary` と同じ「1 回の呼び出しに
        // まとめて包む」方式）。
        self.with_driver_call(|| {
            let pred_dev = self.stream.clone_htod(pred)?;
            self.with_target_dev(target, |target_dev| {
                let num_blocks = mse_num_blocks(numel as u32);
                // `partial_f32` は起動する `num_blocks` 個のブロックそれぞれが
                // `partial[blockIdx.x]` を必ず 1 回書く（`kernels_mse.rs` 冒頭
                // コメント参照）ため `alloc_uninit_f32` を使える（`pool.rs`
                // `alloc_uninit_f32` doc の適用条件）。
                let mut partial_dev = self.allocator.alloc_uninit_f32(num_blocks as usize)?;

                let numel_i = numel as i32;
                let partial_cfg = LaunchConfig {
                    grid_dim: (num_blocks, 1, 1),
                    block_dim: (MSE_BLOCK_DIM, 1, 1),
                    shared_mem_bytes: 0,
                };
                // SAFETY: `pred_dev`／`target_dev`（キャッシュ経由でも長さは鍵と一致し
                // `numel` 要素）は H2D 済み
                // デバイスバッファ、`partial_dev` は `num_blocks` 要素確保済みで
                // カーネルが `blockIdx.x`（`0..num_blocks`）ごとに 1 回だけ書く
                // （`kernels_mse.rs::MSE_PARTIAL_F32` 参照）。カーネル内の
                // grid-stride ループは `idx < numel` を維持する（REQ-8）ため
                // OOB 読み出しは起きない。
                unsafe {
                    self.stream
                        .launch_builder(&self.partial_f32)
                        .arg(&pred_dev)
                        .arg(target_dev)
                        .arg(&mut partial_dev.as_view_mut())
                        .arg(&numel_i)
                        .launch(partial_cfg)?;
                }

                let mut out_dev = self.allocator.alloc_uninit_f32(1)?;
                let num_partials_i = num_blocks as i32;
                let finalize_cfg = LaunchConfig {
                    grid_dim: (1, 1, 1),
                    block_dim: (MSE_BLOCK_DIM, 1, 1),
                    shared_mem_bytes: 0,
                };
                // SAFETY: `partial_dev` は上記で `num_blocks` 要素すべてが書き
                // 込み済み、`out_dev` は `mse_finalize_f32` が単一ブロックの
                // lane 0 で `out[0]` を必ず 1 回書く（`kernels_mse.rs` 参照）ため
                // `alloc_uninit_f32(1)` を使える。`num_partials_i` は起動時に
                // 確保した `partial_dev` の長さと同一の値を渡す（本関数内の
                // 単一の `num_blocks` 由来。呼び出し元がずらせない）。
                unsafe {
                    self.stream
                        .launch_builder(&self.finalize_f32)
                        .arg(&partial_dev.as_view())
                        .arg(&mut out_dev.as_view_mut())
                        .arg(&num_partials_i)
                        .arg(&factor)
                        .launch(finalize_cfg)?;
                }

                let host: Vec<f32> = readback(&self.stream, &out_dev.as_view())?;
                Ok(host.first().copied().unwrap_or(0.0))
            })
        })
    }

    /// backward: `dPred[i] = scale·(pred[i]−target[i])`。`dTarget` は
    /// 呼び出し元（`ops.rs`／`fandhe_ai_autodiff::grad::vjp`）がホスト側
    /// で符号反転して得る契約（`backend_ops.rs::BackendOps::
    /// mse_loss_backward` doc 参照）のため、本関数は `dPred` のみを
    /// 計算する。`numel == 0` は空 `Vec` を返す。
    ///
    /// **ストリーム順序契約（イシュー #1692）**: 本関数は launch 直後の
    /// 明示 `synchronize()` を持たず、末尾の `readback` 呼び出し 1 箇所
    /// （D2H＋同期）へ完了待ちを集約する。`docs/backend-cuda-async-
    /// execution-design.md` §2.3・§16 が定める「ホスト `Tensor` を返す
    /// `BackendOps` 演算は戻り値の D2H が構造的な同期点」契約に既に
    /// 準拠しており、本関数自体への変更は不要と確認済み。
    pub fn run_mse_backward_f32(
        &self,
        pred: &[f32],
        target: &[f32],
        scale: f32,
    ) -> Result<Vec<f32>, CudaError> {
        validate_mse_binary_len(pred.len(), target.len())?;
        let numel = pred.len();
        if numel == 0 {
            return Ok(Vec::new());
        }

        // codex-review P0 指摘対応（PR #1390 是正）: `run_mse_loss_f32`
        // と同じ理由で `Self::with_driver_call` へ参加させる。
        self.with_driver_call(|| {
            let pred_dev = self.stream.clone_htod(pred)?;
            self.with_target_dev(target, |target_dev| {
                // イシュー #1020: `mse_backward_f32` は `if (idx < numel)` ガード
                // 内で `dpred[idx]` を必ず埋める（`kernels_mse.rs` 参照）ため
                // `alloc_uninit_f32` を使う（`elementwise.rs::run_unary` と同じ
                // 適用条件）。
                let mut dpred_dev = self.allocator.alloc_uninit_f32(numel)?;

                let numel_i = numel as i32;
                let cfg = LaunchConfig {
                    grid_dim: (numel.div_ceil(MSE_BLOCK_DIM as usize) as u32, 1, 1),
                    block_dim: (MSE_BLOCK_DIM, 1, 1),
                    shared_mem_bytes: 0,
                };
                // SAFETY: `pred_dev`／`target_dev`／`dpred_dev` はいずれも
                // `numel` 要素のデバイスバッファであり、カーネル内の手動境界
                // チェック（`if (idx < numel)`。REQ-8）と合わせて OOB 読み書き
                // が起きない根拠とする。グリッド次元は `div_ceil` で numel を
                // 包含するよう構築しており、末尾ブロックの余剰スレッドは
                // カーネル内境界チェックで弾かれる。
                unsafe {
                    self.stream
                        .launch_builder(&self.backward_f32)
                        .arg(&pred_dev)
                        .arg(target_dev)
                        .arg(&mut dpred_dev.as_view_mut())
                        .arg(&numel_i)
                        .arg(&scale)
                        .launch(cfg)?;
                }

                readback(&self.stream, &dpred_dev.as_view())
            })
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    type Fake = Vec<f32>;

    /// 呼び出し回数を数えるフェイク転送で `resolve_target` を駆動する。
    #[derive(Default)]
    struct Calls {
        upload: usize,
        overwrite: usize,
    }

    fn run(
        slot: &mut Option<CachedTarget<Fake>>,
        calls: &mut Calls,
        target: &[f32],
        fail: bool,
    ) -> Result<Fake, ()> {
        let (u, o) = (&mut calls.upload, &mut calls.overwrite);
        resolve_target(
            slot,
            target,
            |t| {
                *u += 1;
                if fail { Err(()) } else { Ok(t.to_vec()) }
            },
            |d, t| {
                *o += 1;
                if fail {
                    return Err(());
                }
                d.copy_from_slice(t);
                Ok(())
            },
            |d| Ok(d.clone()),
        )
    }

    #[test]
    fn bits_equal_distinguishes_signed_zero_nan_and_len() {
        assert!(target_bits_equal(&[], &[]));
        assert!(target_bits_equal(&[1.0, 2.0], &[1.0, 2.0]));
        assert!(!target_bits_equal(&[1.0], &[1.0, 2.0]));
        assert!(!target_bits_equal(&[0.0], &[-0.0]));
        let n1 = f32::from_bits(0x7fc0_0000);
        let n2 = f32::from_bits(0x7fc0_0001);
        assert!(target_bits_equal(&[n1], &[n1]));
        assert!(!target_bits_equal(&[n1], &[n2]));
    }

    #[test]
    fn hit_miss_and_resize_transitions() {
        let mut slot = None;
        let mut c = Calls::default();
        let a = [1.0, 2.0, 3.0];
        let b = [1.0, 2.0, 4.0];
        assert_eq!(run(&mut slot, &mut c, &a, false).unwrap(), a);
        assert_eq!((c.upload, c.overwrite), (1, 0));
        assert_eq!(run(&mut slot, &mut c, &a, false).unwrap(), a);
        assert_eq!((c.upload, c.overwrite), (1, 0), "A->A はヒット");
        assert_eq!(run(&mut slot, &mut c, &b, false).unwrap(), b);
        assert_eq!((c.upload, c.overwrite), (1, 1), "同長ミスは上書き");
        let longer = [9.0; 5];
        assert_eq!(run(&mut slot, &mut c, &longer, false).unwrap(), longer);
        assert_eq!((c.upload, c.overwrite), (2, 1), "長さ違いは再確保");
        assert_eq!(run(&mut slot, &mut c, &a, false).unwrap(), a);
        assert_eq!((c.upload, c.overwrite), (3, 1), "B->A は再びミス");
    }

    #[test]
    fn failed_transfer_invalidates_cache() {
        let mut slot = None;
        let mut c = Calls::default();
        let a = [1.0, 2.0];
        let b = [3.0, 4.0];
        run(&mut slot, &mut c, &a, false).unwrap();
        assert!(run(&mut slot, &mut c, &b, true).is_err());
        assert!(slot.is_none(), "失敗後は空");
        // 古い A にヒットせず必ず再転送される。
        let before = c.upload;
        assert_eq!(run(&mut slot, &mut c, &a, false).unwrap(), a);
        assert_eq!(c.upload, before + 1);
    }
}
