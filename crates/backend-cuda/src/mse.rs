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
//!
//! **デバイス入出力版（イシュー #3222。設計 `docs/train-step-resident-parts-
//! design.md` §3）**: `launch_mse_loss_device_f32`／
//! `launch_mse_backward_device_f32` は pred・target・出力を
//! 呼び出し元（`ops.rs::CudaBackendOps::mse_loss_device`／
//! `mse_loss_backward_device`）が確保済みのデバイス引数で受け、起動だけを
//! ストリームへ積む。`readback`・`synchronize` は行わない（#1692 の
//! ストリーム順序契約。起動エラーは次の同期点の poison／世代検査から観測
//! される）。ホスト版と同じ起動ヘルパ（`launch_partial_finalize`／
//! `launch_backward`）を共有するため、カーネル・起動構成・縮約順序が同一で
//! 出力はホスト版と bit 一致する（`unsafe` ブロックもヘルパの 3 か所のまま
//! 増えない。ホスト版から移設しただけ）。
//! forward の `partial` はプールへ返すと「finalize がまだ読んでいる間に返却・
//! 再利用される」race になる（readback の同期が無いため）ので、`CudaMse` が
//! 持つ永続ワークスペースを使う。条件: (a) ロックは partial・finalize の両
//! 起動をまたいで保持する、(b) プールへは渡さない、(c) 再利用はストリーム
//! 順序で安全（次回の partial 書き込みは前回の finalize 読み出しより後に
//! 積まれる）、(d) partial カーネルは起動する全ブロックが
//! `partial[blockIdx.x]` を必ず書くので前回の内容は結果に影響しない。

use std::sync::{Arc, Mutex};

use cudarc::driver::{CudaFunction, CudaSlice, CudaStream, LaunchConfig, PushKernelArg};

use crate::context_cache;
use crate::device::CudaDevice;
use crate::error::CudaError;
use crate::kernels_mse::{self, MSE_BLOCK_DIM, MSE_MAX_BLOCKS};
use crate::memory::{CudaArg, CudaArgMut, readback};
use crate::nvrtc::compile_ptx;
use crate::pool::CudaAllocator;
use crate::warp_geometry::WarpGeometry;

/// `m` のロックを `f` の間だけ保持する。汚染は `detail` 付きの型付きエラーにする
/// （`with_partial_workspace_locked` の実体。CUDA なしでもテストできるよう汎用化）。
pub(crate) fn with_mutex_locked<T, R>(
    m: &Mutex<T>,
    detail: &str,
    f: impl FnOnce() -> R,
) -> Result<R, CudaError> {
    let _guard = m.lock().map_err(|_| CudaError::InvalidElementwiseShape {
        detail: detail.to_string(),
    })?;
    Ok(f())
}

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
    /// デバイス版 forward 専用の `partial` 永続ワークスペース（イシュー
    /// #3222。モジュール doc 参照）。`MSE_MAX_BLOCKS` 要素で遅延確保し、
    /// 以後は使い回す。
    partial_workspace: Mutex<Option<CudaSlice<f32>>>,
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
            partial_workspace: Mutex::new(None),
        })
    }

    /// 永続 `partial` ワークスペースのロックを `f` の間だけ保持する（イシュー #3596）。
    ///
    /// 学習 graph（`graph.rs::launch_train_chain` の capture 結果）は本ワークスペースの
    /// アドレスを焼き込む。replay・初回 launch は `launch_mse_loss_device_f32` を通らないため、
    /// 非 capture の forward が「partial → finalize」の 2 起動の間にいる最中に graph が
    /// 同一 stream へ割り込むと、finalize が graph の partial を読んで他 step の loss を返す。
    /// これを防ぐため graph の起動側（`graph.rs::with_train_graph_workspaces_locked`）が
    /// 本メソッドで同じ mutex を取る。mutex 汚染は型付きエラー。
    pub(crate) fn with_partial_workspace_locked<R>(
        &self,
        f: impl FnOnce() -> R,
    ) -> Result<R, CudaError> {
        with_mutex_locked(
            &self.partial_workspace,
            "mse partial workspace mutex poisoned",
            f,
        )
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

    /// forward の 2 起動（`mse_partial_f32` → `mse_finalize_f32`）を積む
    /// 共通ヘルパ（ホスト版 `run_mse_loss_f32` とデバイス版
    /// `launch_mse_loss_device_f32` だけが呼ぶ。非公開）。
    /// 長さ（`pred`／`target` は `numel`、`partial` は `num_blocks` 以上、
    /// `out` は 1 以上、`numel <= i32::MAX`）を fail-closed に再検査して
    /// から起動し、`unsafe` の SAFETY 前提を実装で担保する。同期はしない。
    /// `numel == 0` は起動せず `Ok`（出力は呼び出し元が 0.0 で用意する）。
    fn launch_partial_finalize(
        &self,
        pred: &CudaArg<'_>,
        target: &CudaArg<'_>,
        partial: &mut CudaArgMut<'_>,
        out: &mut CudaArgMut<'_>,
        numel: usize,
        factor: f32,
    ) -> Result<(), CudaError> {
        validate_mse_len(numel)?;
        if pred.len() != numel || target.len() != numel {
            return Err(CudaError::InvalidElementwiseShape {
                detail: format!(
                    "mse buffer length must equal numel: pred_len={}, target_len={}, numel={numel}",
                    pred.len(),
                    target.len()
                ),
            });
        }
        if numel == 0 {
            return Ok(());
        }
        let num_blocks = mse_num_blocks(numel as u32);
        if partial.len() < num_blocks as usize || out.len() < 1 {
            return Err(CudaError::InvalidElementwiseShape {
                detail: format!(
                    "mse workspace too small: partial_len={}, num_blocks={num_blocks}, out_len={}",
                    partial.len(),
                    out.len()
                ),
            });
        }

        let numel_i = numel as i32;
        let partial_cfg = LaunchConfig {
            grid_dim: (num_blocks, 1, 1),
            block_dim: (MSE_BLOCK_DIM, 1, 1),
            shared_mem_bytes: 0,
        };
        // SAFETY: `pred`／`target` は上記検査で `numel` 要素、`partial` は
        // `num_blocks` 要素以上であり、カーネルは `blockIdx.x`
        // （`0..num_blocks`）ごとに `partial[blockIdx.x]` を 1 回だけ書く
        // （`kernels_mse.rs::MSE_PARTIAL_F32`）。カーネル内の grid-stride
        // ループは `idx < numel` を維持する（REQ-8）ため OOB 読み出しは
        // 起きない。引数順は `(pred, target, partial, int numel)`。
        unsafe {
            let mut builder = self.stream.launch_builder(&self.partial_f32);
            pred.push(&mut builder);
            target.push(&mut builder);
            partial.push(&mut builder);
            builder.arg(&numel_i);
            builder.launch(partial_cfg)?;
        }

        let num_partials_i = num_blocks as i32;
        let finalize_cfg = LaunchConfig {
            grid_dim: (1, 1, 1),
            block_dim: (MSE_BLOCK_DIM, 1, 1),
            shared_mem_bytes: 0,
        };
        // SAFETY: `partial` は上記で `num_blocks` 要素すべてが書き込み済み
        // （同一ストリームの順序で partial 起動の後に実行される）。`out` は
        // 1 要素以上を上記で確認済みで、`mse_finalize_f32` は単一ブロックの
        // lane 0 で `out[0]` を必ず 1 回書く（`kernels_mse.rs`）。
        // `num_partials_i` は本関数内の単一の `num_blocks` 由来で、`partial`
        // の長さ検査と同じ値を渡す（呼び出し元がずらせない）。引数順は
        // `(partial, out, int num_partials, float factor)`。
        unsafe {
            let mut builder = self.stream.launch_builder(&self.finalize_f32);
            partial.push(&mut builder);
            out.push(&mut builder);
            builder.arg(&num_partials_i);
            builder.arg(&factor);
            builder.launch(finalize_cfg)?;
        }
        Ok(())
    }

    /// backward の 1 起動（`mse_backward_f32`）を積む共通ヘルパ
    /// （ホスト版 `run_mse_backward_f32` とデバイス版
    /// `launch_mse_backward_device_f32` だけが呼ぶ。非公開）。
    /// 長さを fail-closed に再検査してから起動する。同期はしない。
    fn launch_backward(
        &self,
        pred: &CudaArg<'_>,
        target: &CudaArg<'_>,
        dpred: &mut CudaArgMut<'_>,
        numel: usize,
        scale: f32,
    ) -> Result<(), CudaError> {
        validate_mse_len(numel)?;
        if pred.len() != numel || target.len() != numel || dpred.len() != numel {
            return Err(CudaError::InvalidElementwiseShape {
                detail: format!(
                    "mse buffer length must equal numel: pred_len={}, target_len={}, \
                     dpred_len={}, numel={numel}",
                    pred.len(),
                    target.len(),
                    dpred.len()
                ),
            });
        }
        if numel == 0 {
            return Ok(());
        }

        let numel_i = numel as i32;
        let cfg = LaunchConfig {
            grid_dim: (numel.div_ceil(MSE_BLOCK_DIM as usize) as u32, 1, 1),
            block_dim: (MSE_BLOCK_DIM, 1, 1),
            shared_mem_bytes: 0,
        };
        // SAFETY: `pred`／`target`／`dpred` はいずれも上記検査で `numel` 要素の
        // デバイスバッファであり、カーネル内の手動境界チェック
        // （`if (idx < numel)`。REQ-8）と合わせて OOB 読み書きが起きない根拠
        // とする。グリッド次元は `div_ceil` で numel を包含し、末尾ブロックの
        // 余剰スレッドはカーネル内境界チェックで弾かれる。引数順は
        // `(pred, target, dpred, int numel, float scale)`。
        unsafe {
            let mut builder = self.stream.launch_builder(&self.backward_f32);
            pred.push(&mut builder);
            target.push(&mut builder);
            dpred.push(&mut builder);
            builder.arg(&numel_i);
            builder.arg(&scale);
            builder.launch(cfg)?;
        }
        Ok(())
    }

    /// デバイス版 forward（イシュー #3222）: `out[0] = factor·Σ(pred−target)²`
    /// を積む。`out` は呼び出し元確保の 1 要素バッファ。`readback`・
    /// `synchronize` はしない（モジュール doc 参照）。`partial` は永続
    /// ワークスペース（ロックは 2 起動をまたぐ）。`numel == 0` は起動しない
    /// （`out` は呼び出し元が 0.0 で用意済みの契約）。
    pub(crate) fn launch_mse_loss_device_f32(
        &self,
        pred: &CudaArg<'_>,
        target: &CudaArg<'_>,
        out: &mut CudaArgMut<'_>,
        numel: usize,
        factor: f32,
    ) -> Result<(), CudaError> {
        self.with_driver_call(|| {
            let mut guard =
                self.partial_workspace
                    .lock()
                    .map_err(|_| CudaError::InvalidElementwiseShape {
                        detail: "mse partial workspace mutex poisoned".to_string(),
                    })?;
            if guard.is_none() {
                // Graph capture 中の確保は graph に記録されてしまうため拒否する（イシュー #3260 の
                // 同期点ガード再監査。`gemm.rs::workspace_slot` と同型。定常状態では確保済みで通らない）。
                if crate::context_cache::is_capturing_on_current_thread(self.ordinal) {
                    return Err(CudaError::CaptureExclusionRejected {
                        detail: "mse partial workspace must be allocated during graph capture"
                            .to_string(),
                    });
                }
                *guard = Some(self.stream.alloc_zeros::<f32>(MSE_MAX_BLOCKS as usize)?);
                // 永続ワークスペースの初回確保も数える（#3252。定常状態では 0）。
                crate::gemm::diag_count_mem_alloc();
            }
            let Some(ws) = guard.as_mut() else {
                return Err(CudaError::InvalidElementwiseShape {
                    detail: "mse partial workspace missing".to_string(),
                });
            };
            let mut partial = CudaArgMut::SliceMut(ws);
            self.launch_partial_finalize(pred, target, &mut partial, out, numel, factor)
        })
    }

    /// デバイス版 backward（イシュー #3222）: `dpred[i] = scale·(pred[i]−
    /// target[i])` を積む。同期しない。
    pub(crate) fn launch_mse_backward_device_f32(
        &self,
        pred: &CudaArg<'_>,
        target: &CudaArg<'_>,
        dpred: &mut CudaArgMut<'_>,
        numel: usize,
        scale: f32,
    ) -> Result<(), CudaError> {
        self.with_driver_call(|| self.launch_backward(pred, target, dpred, numel, scale))
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

                // `out_dev` は `mse_finalize_f32` が単一ブロックの lane 0 で
                // `out[0]` を必ず 1 回書く（`kernels_mse.rs` 参照）ため
                // `alloc_uninit_f32(1)` を使える。
                let mut out_dev = self.allocator.alloc_uninit_f32(1)?;
                let pred_arg = CudaArg::Slice(&pred_dev);
                let target_arg = CudaArg::Slice(target_dev);
                let mut partial_arg = CudaArgMut::View(partial_dev.as_view_mut());
                let mut out_arg = CudaArgMut::View(out_dev.as_view_mut());
                self.launch_partial_finalize(
                    &pred_arg,
                    &target_arg,
                    &mut partial_arg,
                    &mut out_arg,
                    numel,
                    factor,
                )?;

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

                let pred_arg = CudaArg::Slice(&pred_dev);
                let target_arg = CudaArg::Slice(target_dev);
                let mut dpred_arg = CudaArgMut::View(dpred_dev.as_view_mut());
                self.launch_backward(&pred_arg, &target_arg, &mut dpred_arg, numel, scale)?;

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

    #[test]
    fn with_mutex_locked_releases_after_ok_and_panic() {
        let m = Mutex::new(0u32);
        assert_eq!(with_mutex_locked(&m, "x", || 7).unwrap(), 7);
        assert!(m.try_lock().is_ok(), "Ok 後はロックが解放される");
        let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _ = with_mutex_locked(&m, "x", || panic!("boom"));
        }));
        assert!(r.is_err());
        // panic でガードが落ちると mutex は汚染され、以後は型付きエラーになる。
        match with_mutex_locked(&m, "poisoned-detail", || 1) {
            Err(CudaError::InvalidElementwiseShape { detail }) => {
                assert_eq!(detail, "poisoned-detail")
            }
            other => panic!("unexpected: {other:?}"),
        }
    }

    #[test]
    fn with_mutex_locked_holds_lock_during_closure() {
        let m = Mutex::new(());
        with_mutex_locked(&m, "x", || {
            assert!(m.try_lock().is_err(), "f の間はロック保持");
        })
        .unwrap();
    }
}
