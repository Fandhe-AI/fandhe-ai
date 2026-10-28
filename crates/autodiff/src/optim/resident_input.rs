//! 常駐入力ハンドル [`ResidentInput`] と、それを受ける行列積 [`matmul_resident`]
//! （イシュー #3250 P1・P4。公開形の決定は `docs/compat-resident-train-step-decision.md` §4）。
//!
//! # 役割
//! 推論（`Sequential::predict_resident`）と行列積（`Var::matmul`）はホストの `Tensor` しか
//! 受けず、呼ぶたびに入力の H2D が計測窓に入る。本モジュールは入力を **窓の外で 1 回だけ**
//! デバイスへ置く不透明ハンドルを提供し、窓の中を「演算＋出力の D2H 1 回」に絞る。
//! 消えるのは H2D だけで、D2H と同期の 1 回は残る（CPU は出力 `Vec` を `Tensor` へ移すため
//! D2H 相当のコピーも消える。#3672）。
//!
//! # 呼び出し元・境界
//! - 作成: `DeviceParamStore::upload_resident_input`（`device_store.rs`）。facade の
//!   `Tape::upload_resident` が薄く委譲する
//! - 推論: `DeviceParamStore::predict_device_chain_resident`（チェーン本体は
//!   ホスト入力版と共有）。facade の `Sequential::predict_resident_input` が呼ぶ
//! - 行列積: 本モジュールの [`matmul_resident`]。facade の `Tape::matmul_resident` が呼ぶ
//! - 生の autodiff `Tape` は facade から到達できないため、本モジュールの関数は facade が
//!   内部で `&Tape` を渡して使う前例（`DeviceParamStore::predict_device_chain`）に倣う
//!
//! # 寿命・束縛・スレッド
//! - ハンドルは作成時の store（`store_id`）とデバイスに束縛される。推論は store を検査し、
//!   行列積は device と世代だけを検査する（store を取らないため）
//! - 古い世代（CUDA の `invalidate` 後）のバッファはバックエンドの入口が
//!   `StaleDeviceGeneration` で拒否する
//! - `DeviceBuffer` が `Send` でないため、ハンドルは自動で `!Send`（`unsafe impl Send` は足さない）
//!
//! # エラー
//! | 状況 | エラー |
//! |---|---|
//! | 別 store のハンドル（推論） | `InvalidArgument` |
//! | 別デバイス | `DeviceMismatch` |
//! | rank が 2 でない | `InvalidArgument`（行列積）・`InvalidArgument`（推論） |
//! | 内側の次元の不一致 | `ShapeMismatch` |
//! | バックエンドが非対応 | `Unsupported`（ホスト経路へは落とさない。落とすと取り除く H2D が戻る） |

use fandhe_ai_tensor_core::buffer::DeviceBuffer;
use fandhe_ai_tensor_core::device::{BackendError, Device};
use fandhe_ai_tensor_core::{ShapeError, Tensor};

use crate::tape::Tape;

/// デバイスに常駐する入力テンソルの不透明ハンドル。フィールドは非公開で、
/// 中身（バッファ）は公開しない。
pub struct ResidentInput {
    buf: DeviceBuffer<f32>,
    store_id: u64,
    device: Device,
}

impl ResidentInput {
    /// `DeviceParamStore::upload_resident_input` だけが呼ぶ構築入口。
    pub(super) fn new(buf: DeviceBuffer<f32>, store_id: u64, device: Device) -> Self {
        Self {
            buf,
            store_id,
            device,
        }
    }

    /// 入力の shape。
    pub fn shape(&self) -> &[usize] {
        self.buf.shape()
    }

    /// 入力が常駐するデバイス。
    pub fn device(&self) -> Device {
        self.device
    }

    pub(crate) fn store_id(&self) -> u64 {
        self.store_id
    }

    pub(crate) fn buffer(&self) -> &DeviceBuffer<f32> {
        &self.buf
    }
}

impl std::fmt::Debug for ResidentInput {
    /// バッファの中身は出さず、shape と device だけを出す。
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ResidentInput")
            .field("shape", &self.buf.shape())
            .field("device", &self.device)
            .finish()
    }
}

/// 常駐した `a: [m, k]`・`b: [k, n]` の行列積を計算し、結果を D2H 1 回でホストへ返す
/// （イシュー #3250 P4）。カーネルは `BackendOps::gemm_device`（`gemm` と同じ核。
/// 現在の精度モードの `gemm` と bit 一致）で、ホスト経路へは落とさない。
///
/// 同じハンドルを `a` と `b` の両方に渡してもよい（共有借用）。
pub fn matmul_resident(
    tape: &Tape,
    a: &ResidentInput,
    b: &ResidentInput,
) -> Result<Tensor<f32>, BackendError> {
    let device = tape.ops().device();
    if a.device != device || b.device != device {
        return Err(BackendError::DeviceMismatch);
    }
    for r in [a, b] {
        if r.shape().len() != 2 {
            return Err(BackendError::InvalidArgument(format!(
                "matmul_resident: operands must be rank 2, got shape {:?}",
                r.shape()
            )));
        }
    }
    if a.shape()[1] != b.shape()[0] {
        return Err(BackendError::ShapeMismatch(ShapeError::ShapeMismatch {
            lhs: a.shape().to_vec(),
            rhs: b.shape().to_vec(),
        }));
    }
    let mem = tape.ops().memory_ops().ok_or_else(|| {
        BackendError::Unsupported("matmul_resident: backend does not implement MemoryOps".into())
    })?;
    let out = tape.ops().gemm_device(&a.buf, &b.buf)?;
    // 出力は以後使わないため所有権ごと渡す（CPU は Vec を Tensor へ移しコピーしない。#3672）
    mem.download_owned(out)
}

/// 学習ステップのチェーン（`DeviceParamStore::train_step_device_chain`）が積んだ loss への
/// 不透明なハンドル（イシュー #3251。決定記録 `docs/compat-resident-train-step-decision.md`
/// §4 の P6 `ResidentLoss` の autodiff 内部側）。
///
/// loss の値はデバイス上のスカラーとして store が保持し、本ハンドルは作成時の
/// `store_id` と世代だけを持つ。[`DeviceParamStore::read_train_step_loss`](super::DeviceParamStore::read_train_step_loss)
/// が D2H と同期を 1 回だけ行って `f32` を返す。次の step が積まれると世代が進み、
/// 古いハンドルは `InvalidArgument` で拒否される（上書きされた loss を黙って返さない）。
/// フィールドは非公開で、facade の公開面には出さない（公開入口は #3254）。
#[derive(Debug)]
pub struct TrainStepLoss {
    store_id: u64,
    generation: u64,
}

impl TrainStepLoss {
    /// `DeviceParamStore::train_step_device_chain` だけが呼ぶ構築入口。
    pub(super) fn new(store_id: u64, generation: u64) -> Self {
        Self {
            store_id,
            generation,
        }
    }

    pub(super) fn store_id(&self) -> u64 {
        self.store_id
    }

    pub(super) fn generation(&self) -> u64 {
        self.generation
    }
}
