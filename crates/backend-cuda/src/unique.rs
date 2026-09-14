//! `unique`（`torch.unique(input, sorted=True)` の values のみ。
//! イシュー #1734）の起動 API（NVRTC コンパイル・保持・実行）。
//!
//! `gather_scatter.rs::CudaGatherScatter` と同じ構成方針を踏襲する:
//! [`CudaUnique::new`] が `CudaDevice` から `bitonic_step_u32`
//! （`kernels_unique.rs`）を NVRTC コンパイルして保持し、以降は
//! [`CudaUnique::run_unique_f32`] へホスト側スライスを渡すだけで
//! GPU 実行できる。`ops.rs::CudaBackendOps::unique` から `BackendOps`
//! の実装として呼ばれる。
//!
//! # アルゴリズム
//!
//! 1. `n = x.len()`。`n == 0`／`n == 1` は GPU 起動なしで自明に処理
//!    する（バッファ確保・カーネル起動のオーバーヘッドを避ける）。
//! 2. `f32` 値を [`crate::unique_model::total_order_key`] で `u32`
//!    キーへ変換し、`padded`（次の 2 のべき乗）長になるよう
//!    `u32::MAX`（totalOrder 最大キー）でパディングする。
//! 3. `padded` が `i32::MAX`（カーネル引数 `int` の範囲）を超える場合は
//!    [`CudaError::UniqueSizeLimitExceeded`] を返す（`ops.rs` はこの
//!    variant のみを `Unsupported` へ写像し `Var::unique` のホスト
//!    フォールバックへ委ねる）。
//! 4. `keys` を H2D 転送し、標準的なビットニックソート（`k`／`j` の
//!    入れ子ループ）で `bitonic_step_u32` を同一ストリーム上へ繰り返し
//!    投入する（同一ストリームの FIFO 順序保証によりステップ間の
//!    依存が正しく直列化される。`docs/backend-cuda-async-execution-
//!    design.md` §3 と同じ前提）。
//! 5. `readback` で 1 回だけ D2H（`memory.rs::readback` の唯一の同期点
//!    契約に従う）し、先頭 `n` 要素を `key_to_f32` で復元してから
//!    ホスト側で `dedup_by(==)`（GPU 側 prefix-sum 圧縮は行わない——
//!    正直な設計判断。`docs/unique-facade-exposure-decision.md` §3.1
//!    「GPU 側 prefix-sum 圧縮は対象外」参照）する。

use std::sync::Arc;

use cudarc::driver::{CudaFunction, CudaStream, LaunchConfig, PushKernelArg};

use crate::context_cache;
use crate::device::CudaDevice;
use crate::error::CudaError;
use crate::kernels_unique::{self, UNIQUE_BLOCK_DIM};
use crate::memory::readback;
use crate::nvrtc::compile_ptx;
use crate::unique_model::{key_to_f32, total_order_key};

/// `value` が `i32::MAX` に収まることを検証する（カーネル引数 `int` は
/// C の 32bit 符号付き整数のため。`gather_scatter.rs::
/// validate_i32_bound` と同じ理由の複製）。
fn validate_i32_bound(value: usize, name: &str) -> Result<i32, CudaError> {
    i32::try_from(value).map_err(|_| CudaError::InvalidUniqueShape {
        detail: format!("unique dimension must fit in i32 (kernel argument type): {name}={value}"),
    })
}

/// `bitonic_step_u32` カーネルのコンパイル済みハンドルを保持する。
pub struct CudaUnique {
    stream: Arc<CudaStream>,
    /// `gather_scatter.rs::CudaGatherScatter::ordinal` と同じ役割
    /// （`Self::with_driver_call` が `context_cache::with_driver_call`
    /// を呼ぶ際のキー）。
    ordinal: usize,
    bitonic_step_u32: CudaFunction,
}

impl CudaUnique {
    /// `device` 上で `bitonic_step_u32` カーネルを NVRTC コンパイルし
    /// 保持するハンドルを構築する（`gather_scatter.rs::
    /// CudaGatherScatter::new` と同一手順）。
    pub fn new(device: &CudaDevice) -> Result<Self, CudaError> {
        let arch = device.arch();
        let ptx = compile_ptx(kernels_unique::BITONIC_STEP_U32, arch)?;
        let bitonic_step_u32 = device
            .context()
            .load_module(ptx)?
            .load_function("bitonic_step_u32")?;

        Ok(Self {
            stream: device.stream().clone(),
            ordinal: device.ordinal(),
            bitonic_step_u32,
        })
    }

    /// `CudaUnique` の driver 呼び出しを CUDA Graph capture 排他へ
    /// 参加させる共通ヘルパー（`gather_scatter.rs::CudaGatherScatter::
    /// with_driver_call` と同じ設計）。
    fn with_driver_call<T>(
        &self,
        f: impl FnOnce() -> Result<T, CudaError>,
    ) -> Result<T, CudaError> {
        context_cache::with_driver_call(self.ordinal, f)
    }

    /// `torch.unique(input, sorted=True)` の values（本ファイル冒頭
    /// コメント「アルゴリズム」節参照）。
    pub fn run_unique_f32(&self, x: &[f32]) -> Result<Vec<f32>, CudaError> {
        let n = x.len();
        if n == 0 {
            return Ok(Vec::new());
        }
        if n == 1 {
            return Ok(vec![x[0]]);
        }

        let padded = n
            .checked_next_power_of_two()
            .ok_or(CudaError::UniqueSizeLimitExceeded {
                n,
                limit: usize::MAX,
            })?;
        // `padded` がカーネル引数 `int` の範囲を超える場合のみ
        // `UniqueSizeLimitExceeded`（`ops.rs` が `Unsupported` へ写像し
        // ホストフォールバックへ委ねる。本チェック自体はサイズ上限の
        // 検査であり `InvalidUniqueShape`〈内部契約違反〉とは区別する）。
        if i32::try_from(padded).is_err() {
            return Err(CudaError::UniqueSizeLimitExceeded {
                n,
                limit: i32::MAX as usize,
            });
        }

        let mut keys: Vec<u32> = x.iter().map(|&v| total_order_key(v)).collect();
        keys.resize(padded, u32::MAX);

        let padded_i = validate_i32_bound(padded, "padded")?;

        let sorted_keys: Vec<u32> = self.with_driver_call(|| {
            let mut keys_dev = self.stream.clone_htod(&keys)?;

            let mut k = 2usize;
            while k <= padded {
                let mut j = k / 2;
                while j >= 1 {
                    let j_i = validate_i32_bound(j, "j")?;
                    let k_i = validate_i32_bound(k, "k")?;
                    let cfg = LaunchConfig {
                        grid_dim: ((padded as u32).div_ceil(UNIQUE_BLOCK_DIM), 1, 1),
                        block_dim: (UNIQUE_BLOCK_DIM, 1, 1),
                        shared_mem_bytes: 0,
                    };
                    // SAFETY: `keys_dev` は `padded` 要素確保済み（直上の
                    // `clone_htod` の結果）。カーネルは `i < padded`・
                    // `ixj < padded`（REQ-8）を維持したまま `keys_dev`
                    // 内の要素同士を read-modify-write するのみで範囲外
                    // アクセスはない（`kernels_unique.rs::
                    // BITONIC_STEP_U32` 参照）。
                    unsafe {
                        self.stream
                            .launch_builder(&self.bitonic_step_u32)
                            .arg(&mut keys_dev)
                            .arg(&j_i)
                            .arg(&k_i)
                            .arg(&padded_i)
                            .launch(cfg)?;
                    }
                    j /= 2;
                }
                k *= 2;
            }
            readback::<u32, _>(&self.stream, &keys_dev)
        })?;

        let mut out: Vec<f32> = sorted_keys[..n].iter().map(|&k| key_to_f32(k)).collect();
        out.dedup_by(|cur, prev| *cur == *prev);
        Ok(out)
    }
}
