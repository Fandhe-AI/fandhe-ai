//! elementwise 系の ROCm 実行入口（`backend-cuda::elementwise`／`fused_elementwise` の移植。
//! イシュー #3433・分解案 R2）。
//!
//! 役割: HIP/HIPRTC をロードして静的ソースをコンパイルし、ホストのスライスを受けて `Vec` を返す
//! 3 つの実行型を提供する。`BackendOps` への結線（ブロードキャスト・形状再検査・`Tensor` 化・
//! `Unsupported` の事前確定）は `ops/ew.rs` の責務で、本ファイルはスライス入出力のみ。
//!
//! - `RocmElementwise`: `add`・`mul`・`relu`・`exp`・`tanh`・`where_cond`・`masked_fill`
//!   （`kernels_elementwise`）と、デバイス常駐版（`binary_resident`・`unary_resident`）。
//! - `RocmScalarOp`: `ScalarUnaryOp`／`ScalarBinaryOp`（`kernels_scalar_op`）。
//! - `RocmFusedElementwise`: 融合プランを `[kind, a, b]` の `i32` 3 つ組へ符号化した
//!   `FusedProgram` を、静的インタプリタカーネル（`kernels_fused_elementwise`）で実行する。
//!   `encode_program` は HIP に依存しない純関数で、allowlist 検証（fail-closed）を担う。
//!
//! 起動は `KernelHost::run` に集約され、呼び出しごとに `synchronize` する（Metal と同じ
//! 「呼び出しごとに同期」。非同期にストリームへ積む CUDA とは異なる。trait doc が許す差異）。
//! 新しい `unsafe` はなく、監査済みの `KernelHost::run`・`TypedDeviceBuffer` を再利用する。
//! ホスト側で `i32` 範囲・要素数の checked 演算を検証してから起動する（カーネル側 REQ-8 と二重化）。

use fandhe_ai_tensor_core::{
    FusedOpKind, FusionPlan, MAX_FUSED_SEGMENT_NODES, ScalarBinaryOp, ScalarUnaryOp,
};

use crate::error::RocmError;
use crate::kernels_elementwise::{
    ADD_SPEC, EXP_SPEC, MASKED_FILL_SPEC, MUL_SPEC, RELU_SPEC, TANH_SPEC, WHERE_SPEC,
    render_elementwise_source,
};
use crate::kernels_fused_elementwise::{
    FUSED_SPEC, MAX_FUSED_OPS, OP_ADD, OP_EXP, OP_INPUT, OP_MUL, OP_RELU, OP_TANH,
    render_fused_source,
};
use crate::kernels_scalar_op::{binary_spec, render_scalar_source, unary_payload, unary_spec};
use crate::launch::{Arg, KernelHost, KernelSpec};
use crate::target::GfxTarget;
use crate::typed_buffer::TypedDeviceBuffer;

const OP: &str = "elementwise";

/// `i32`（カーネル引数型）への収容検査。
pub(crate) fn fit_i32(op: &'static str, value: usize, name: &str) -> Result<i32, RocmError> {
    i32::try_from(value).map_err(|_| RocmError::InvalidOpArgs {
        op,
        detail: format!("{name}={value} は i32（カーネル引数型）に収まりません"),
    })
}

/// 入力スライス長の一致を検査する。
fn check_len(
    op: &'static str,
    name: &str,
    expected: usize,
    actual: usize,
) -> Result<(), RocmError> {
    if expected != actual {
        return Err(RocmError::InvalidOpArgs {
            op,
            detail: format!("{name} の長さ不一致: 期待 {expected}, 実際 {actual}"),
        });
    }
    Ok(())
}

/// 二項カーネルの種別（`add`・`mul`）。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum BinaryKind {
    Add,
    Mul,
}

impl BinaryKind {
    fn spec(self) -> &'static KernelSpec {
        match self {
            Self::Add => &ADD_SPEC,
            Self::Mul => &MUL_SPEC,
        }
    }
}

/// 単項カーネルの種別（`relu`・`exp`・`tanh`）。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum UnaryKind {
    Relu,
    Exp,
    Tanh,
}

impl UnaryKind {
    fn spec(self) -> &'static KernelSpec {
        match self {
            Self::Relu => &RELU_SPEC,
            Self::Exp => &EXP_SPEC,
            Self::Tanh => &TANH_SPEC,
        }
    }
}

/// コンパイル済みの elementwise 基本カーネルを保持し、実行する。
pub struct RocmElementwise {
    host: KernelHost,
}

impl RocmElementwise {
    /// HIP/HIPRTC をロードし基本カーネルを `target` 向けにコンパイルする。
    ///
    /// ROCm 非搭載では `LibraryUnavailable`、デバイスがなければ `NoDevice`（panic しない）。
    pub fn new(target: GfxTarget) -> Result<Self, RocmError> {
        let src = render_elementwise_source()?;
        Ok(Self {
            host: KernelHost::new(&src, target)?,
        })
    }

    /// `a + b`（同長のスライス）。
    pub fn run_add_f32(&self, a: &[f32], b: &[f32]) -> Result<Vec<f32>, RocmError> {
        self.run_binary(BinaryKind::Add, a, b)
    }

    /// `a * b`（同長のスライス）。
    pub fn run_mul_f32(&self, a: &[f32], b: &[f32]) -> Result<Vec<f32>, RocmError> {
        self.run_binary(BinaryKind::Mul, a, b)
    }

    /// `x > 0 ? x : 0`。
    pub fn run_relu_f32(&self, a: &[f32]) -> Result<Vec<f32>, RocmError> {
        self.run_unary(UnaryKind::Relu, a)
    }

    /// `expf(x)`。
    pub fn run_exp_f32(&self, a: &[f32]) -> Result<Vec<f32>, RocmError> {
        self.run_unary(UnaryKind::Exp, a)
    }

    /// `tanhf(x)`。
    pub fn run_tanh_f32(&self, a: &[f32]) -> Result<Vec<f32>, RocmError> {
        self.run_unary(UnaryKind::Tanh, a)
    }

    /// `cond != 0 ? a : b`（3 本とも同長）。
    pub fn run_where_f32(&self, cond: &[f32], a: &[f32], b: &[f32]) -> Result<Vec<f32>, RocmError> {
        let n = cond.len();
        check_len(OP, "a", n, a.len())?;
        check_len(OP, "b", n, b.len())?;
        let n_i = fit_i32(OP, n, "numel")?;
        if n == 0 {
            return Ok(Vec::new());
        }
        let _guard = self.host.select()?;
        let rt = self.host.runtime();
        let dc = TypedDeviceBuffer::<f32>::from_slice(rt.clone(), cond)?;
        let da = TypedDeviceBuffer::<f32>::from_slice(rt.clone(), a)?;
        let db = TypedDeviceBuffer::<f32>::from_slice(rt.clone(), b)?;
        let dout = TypedDeviceBuffer::<f32>::new_zeroed(rt, n)?;
        let args = [
            Arg::Ptr(dc.as_device_ptr()),
            Arg::Ptr(da.as_device_ptr()),
            Arg::Ptr(db.as_device_ptr()),
            Arg::Ptr(dout.as_device_ptr()),
            Arg::I32(n_i),
        ];
        self.host.run(OP, &WHERE_SPEC, n, &args)?;
        dout.to_vec()
    }

    /// `mask != 0 ? value : x`（同長）。
    pub fn run_masked_fill_f32(
        &self,
        x: &[f32],
        mask: &[f32],
        value: f32,
    ) -> Result<Vec<f32>, RocmError> {
        let n = x.len();
        check_len(OP, "mask", n, mask.len())?;
        let n_i = fit_i32(OP, n, "numel")?;
        if n == 0 {
            return Ok(Vec::new());
        }
        let _guard = self.host.select()?;
        let rt = self.host.runtime();
        let dx = TypedDeviceBuffer::<f32>::from_slice(rt.clone(), x)?;
        let dm = TypedDeviceBuffer::<f32>::from_slice(rt.clone(), mask)?;
        let dout = TypedDeviceBuffer::<f32>::new_zeroed(rt, n)?;
        let args = [
            Arg::Ptr(dx.as_device_ptr()),
            Arg::Ptr(dm.as_device_ptr()),
            Arg::F32(value),
            Arg::Ptr(dout.as_device_ptr()),
            Arg::I32(n_i),
        ];
        self.host.run(OP, &MASKED_FILL_SPEC, n, &args)?;
        dout.to_vec()
    }

    pub(crate) fn run_binary(
        &self,
        kind: BinaryKind,
        a: &[f32],
        b: &[f32],
    ) -> Result<Vec<f32>, RocmError> {
        check_len(OP, "b", a.len(), b.len())?;
        let n = a.len();
        let n_i = fit_i32(OP, n, "numel")?;
        if n == 0 {
            return Ok(Vec::new());
        }
        let _guard = self.host.select()?;
        let rt = self.host.runtime();
        let da = TypedDeviceBuffer::<f32>::from_slice(rt.clone(), a)?;
        let db = TypedDeviceBuffer::<f32>::from_slice(rt.clone(), b)?;
        let dout = TypedDeviceBuffer::<f32>::new_zeroed(rt, n)?;
        let args = [
            Arg::Ptr(da.as_device_ptr()),
            Arg::Ptr(db.as_device_ptr()),
            Arg::Ptr(dout.as_device_ptr()),
            Arg::I32(n_i),
        ];
        self.host.run(OP, kind.spec(), n, &args)?;
        dout.to_vec()
    }

    pub(crate) fn run_unary(&self, kind: UnaryKind, a: &[f32]) -> Result<Vec<f32>, RocmError> {
        let n = a.len();
        let n_i = fit_i32(OP, n, "numel")?;
        if n == 0 {
            return Ok(Vec::new());
        }
        let _guard = self.host.select()?;
        let rt = self.host.runtime();
        let da = TypedDeviceBuffer::<f32>::from_slice(rt.clone(), a)?;
        let dout = TypedDeviceBuffer::<f32>::new_zeroed(rt, n)?;
        let args = [
            Arg::Ptr(da.as_device_ptr()),
            Arg::Ptr(dout.as_device_ptr()),
            Arg::I32(n_i),
        ];
        self.host.run(OP, kind.spec(), n, &args)?;
        dout.to_vec()
    }

    /// デバイス常駐の二項演算。入力は同長で、結果も新しいデバイスバッファのまま返す（H2D/D2H なし）。
    /// 長さ 0 は呼び出し側（`ops/ew.rs`）が HIP に触れずに処理する契約。
    pub(crate) fn binary_resident(
        &self,
        kind: BinaryKind,
        a: &TypedDeviceBuffer<f32>,
        b: &TypedDeviceBuffer<f32>,
    ) -> Result<TypedDeviceBuffer<f32>, RocmError> {
        check_len(OP, "b", a.len(), b.len())?;
        let n = a.len();
        let n_i = fit_i32(OP, n, "numel")?;
        let _guard = self.host.select()?;
        // 出力は kernel が全有効要素を上書きするため、ゼロ初期化の H2D 転送は行わない（常駐 API 契約）。
        let dout = TypedDeviceBuffer::<f32>::new(self.host.runtime(), n)?;
        let args = [
            Arg::Ptr(a.as_device_ptr()),
            Arg::Ptr(b.as_device_ptr()),
            Arg::Ptr(dout.as_device_ptr()),
            Arg::I32(n_i),
        ];
        self.host.run(OP, kind.spec(), n, &args)?;
        Ok(dout)
    }

    /// デバイス常駐の単項演算（[`Self::binary_resident`] の単項版）。
    pub(crate) fn unary_resident(
        &self,
        kind: UnaryKind,
        a: &TypedDeviceBuffer<f32>,
    ) -> Result<TypedDeviceBuffer<f32>, RocmError> {
        let n = a.len();
        let n_i = fit_i32(OP, n, "numel")?;
        let _guard = self.host.select()?;
        // 出力は kernel が全有効要素を上書きするため、ゼロ初期化の H2D 転送は行わない（常駐 API 契約）。
        let dout = TypedDeviceBuffer::<f32>::new(self.host.runtime(), n)?;
        let args = [
            Arg::Ptr(a.as_device_ptr()),
            Arg::Ptr(dout.as_device_ptr()),
            Arg::I32(n_i),
        ];
        self.host.run(OP, kind.spec(), n, &args)?;
        Ok(dout)
    }

    /// テスト専用の upload（`ops/ew.rs` の `upload_resident` が使う）。
    #[cfg(test)]
    pub(crate) fn upload(&self, src: &[f32]) -> Result<TypedDeviceBuffer<f32>, RocmError> {
        let _guard = self.host.select()?;
        TypedDeviceBuffer::<f32>::from_slice(self.host.runtime(), src)
    }
}

/// コンパイル済みの scalar-op カーネル群を保持し、実行する。
pub struct RocmScalarOp {
    host: KernelHost,
}

impl RocmScalarOp {
    /// HIP/HIPRTC をロードし scalar-op カーネルを `target` 向けにコンパイルする。
    pub fn new(target: GfxTarget) -> Result<Self, RocmError> {
        let src = render_scalar_source()?;
        Ok(Self {
            host: KernelHost::new(&src, target)?,
        })
    }

    /// `op(a)`。対象外の kind は `InvalidOpArgs`（呼び出し側が事前に `Unsupported` を確定する契約）。
    pub fn run_unary_f32(&self, op: ScalarUnaryOp, a: &[f32]) -> Result<Vec<f32>, RocmError> {
        const OPN: &str = "scalar_unary";
        let spec = unary_spec(op).ok_or_else(|| RocmError::InvalidOpArgs {
            op: OPN,
            detail: format!("対象外の kind です: {op:?}"),
        })?;
        let n = a.len();
        let n_i = fit_i32(OPN, n, "numel")?;
        if n == 0 {
            return Ok(Vec::new());
        }
        let _guard = self.host.select()?;
        let rt = self.host.runtime();
        let da = TypedDeviceBuffer::<f32>::from_slice(rt.clone(), a)?;
        let dout = TypedDeviceBuffer::<f32>::new_zeroed(rt, n)?;
        let mut args = vec![
            Arg::Ptr(da.as_device_ptr()),
            Arg::Ptr(dout.as_device_ptr()),
            Arg::I32(n_i),
        ];
        // ペイロードはソースではなくカーネル引数で渡す（A03）。
        let (count, values) = unary_payload(op);
        args.extend(values.iter().take(count).map(|v| Arg::F32(*v)));
        self.host.run(OPN, spec, n, &args)?;
        dout.to_vec()
    }

    /// `op(a, b)`（同長）。対象外の kind は `InvalidOpArgs`。
    pub fn run_binary_f32(
        &self,
        op: ScalarBinaryOp,
        a: &[f32],
        b: &[f32],
    ) -> Result<Vec<f32>, RocmError> {
        const OPN: &str = "scalar_binary";
        let spec = binary_spec(op).ok_or_else(|| RocmError::InvalidOpArgs {
            op: OPN,
            detail: format!("対象外の kind です: {op:?}"),
        })?;
        check_len(OPN, "b", a.len(), b.len())?;
        let n = a.len();
        let n_i = fit_i32(OPN, n, "numel")?;
        if n == 0 {
            return Ok(Vec::new());
        }
        let _guard = self.host.select()?;
        let rt = self.host.runtime();
        let da = TypedDeviceBuffer::<f32>::from_slice(rt.clone(), a)?;
        let db = TypedDeviceBuffer::<f32>::from_slice(rt.clone(), b)?;
        let dout = TypedDeviceBuffer::<f32>::new_zeroed(rt, n)?;
        let args = [
            Arg::Ptr(da.as_device_ptr()),
            Arg::Ptr(db.as_device_ptr()),
            Arg::Ptr(dout.as_device_ptr()),
            Arg::I32(n_i),
        ];
        self.host.run(OPN, spec, n, &args)?;
        dout.to_vec()
    }
}

/// 符号化済みの融合プログラム（`[kind, a, b]` の `i32` 3 つ組を `n_ops` 個並べたもの）。
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct FusedProgram {
    /// `3 * n_ops` 個の `i32`。
    pub(crate) words: Vec<i32>,
    /// 演算（`Input` を含む）の個数。
    pub(crate) n_ops: usize,
    /// 葉の個数。
    pub(crate) leaf_count: usize,
}

/// 融合プランを検証して符号化する（HIP に触れない純関数。fail-closed）。
///
/// `Err(理由)` は `BackendError::Unsupported` に写す（呼び出し側 autodiff が演算ごとの実行へ
/// フォールバックする契約）。allowlist は `Input`・`Add`・`Mul`・`Relu`・`Exp`・`Tanh` のみ
/// （`FusedOpKind` は `#[non_exhaustive]` のため未知 variant は拒否する）。行融合（RMSNorm・softmax）
/// プランは本カーネルの対象外。
pub(crate) fn encode_program(plan: &FusionPlan) -> Result<FusedProgram, String> {
    if plan.row_fusion().is_some() {
        return Err("row fusion（RMSNorm・softmax）プランは未対応です".into());
    }
    let ops: Vec<FusedOpKind> = plan.ops().collect();
    if ops.is_empty() || ops.len() > MAX_FUSED_OPS {
        return Err(format!(
            "ops 数 {} が 1..={MAX_FUSED_OPS} の範囲外です",
            ops.len()
        ));
    }
    let leaf_count = plan.leaf_count();
    if leaf_count > MAX_FUSED_SEGMENT_NODES + 1 {
        return Err(format!("leaf 数 {leaf_count} が上限を超えます"));
    }
    let mut words = Vec::with_capacity(ops.len() * 3);
    let mut compute_nodes = 0usize;
    for (i, op) in ops.iter().enumerate() {
        let operand = |idx: usize| -> Result<i32, String> {
            if idx >= i {
                return Err(format!(
                    "ノード {i} のオペランド {idx} がトポロジカル順でありません"
                ));
            }
            i32::try_from(idx).map_err(|_| "オペランド添字が i32 を超えます".to_string())
        };
        let (kind, a, b) = match *op {
            FusedOpKind::Input { leaf_index } => {
                if leaf_index >= leaf_count {
                    return Err(format!(
                        "leaf_index {leaf_index} が leaf 数 {leaf_count} 以上です"
                    ));
                }
                (OP_INPUT, leaf_index as i32, 0)
            }
            FusedOpKind::Add { lhs, rhs } => (OP_ADD, operand(lhs)?, operand(rhs)?),
            FusedOpKind::Mul { lhs, rhs } => (OP_MUL, operand(lhs)?, operand(rhs)?),
            FusedOpKind::Relu { input } => (OP_RELU, operand(input)?, 0),
            FusedOpKind::Exp { input } => (OP_EXP, operand(input)?, 0),
            FusedOpKind::Tanh { input } => (OP_TANH, operand(input)?, 0),
            // Sub・Div・Rsqrt・Sum・Max・Broadcast と未知の将来 variant。
            other => return Err(format!("未対応の融合演算です: {other:?}")),
        };
        if kind != OP_INPUT {
            compute_nodes += 1;
        }
        words.extend_from_slice(&[kind, a, b]);
    }
    if compute_nodes > MAX_FUSED_SEGMENT_NODES {
        return Err(format!(
            "演算ノード数 {compute_nodes} が上限 {MAX_FUSED_SEGMENT_NODES} を超えます"
        ));
    }
    Ok(FusedProgram {
        words,
        n_ops: ops.len(),
        leaf_count,
    })
}

/// コンパイル済みの融合インタプリタカーネルを保持し、実行する。
pub struct RocmFusedElementwise {
    host: KernelHost,
}

impl RocmFusedElementwise {
    /// HIP/HIPRTC をロードし融合カーネルを `target` 向けにコンパイルする。
    pub fn new(target: GfxTarget) -> Result<Self, RocmError> {
        let src = render_fused_source()?;
        Ok(Self {
            host: KernelHost::new(&src, target)?,
        })
    }

    /// 符号化済みプログラムを実行する。`leaves` は全 leaf を連結した `[leaf_count * numel]`。
    pub(crate) fn run_encoded(
        &self,
        program: &FusedProgram,
        leaves: &[f32],
        numel: usize,
    ) -> Result<Vec<f32>, RocmError> {
        const OPN: &str = "fused_elementwise";
        let expected =
            program
                .leaf_count
                .checked_mul(numel)
                .ok_or(RocmError::SizeLimitExceeded {
                    op: OPN,
                    n: numel,
                    limit: usize::MAX / program.leaf_count.max(1),
                })?;
        check_len(OPN, "leaves", expected, leaves.len())?;
        check_len(OPN, "program", program.n_ops * 3, program.words.len())?;
        let numel_i = fit_i32(OPN, numel, "numel")?;
        let n_ops_i = fit_i32(OPN, program.n_ops, "n_ops")?;
        let leaf_count_i = fit_i32(OPN, program.leaf_count, "leaf_count")?;
        if numel == 0 {
            return Ok(Vec::new());
        }
        let _guard = self.host.select()?;
        let rt = self.host.runtime();
        let dleaves = TypedDeviceBuffer::<f32>::from_slice(rt.clone(), leaves)?;
        let dprog = TypedDeviceBuffer::<i32>::from_slice(rt.clone(), &program.words)?;
        let dout = TypedDeviceBuffer::<f32>::new_zeroed(rt, numel)?;
        let args = [
            Arg::Ptr(dleaves.as_device_ptr()),
            Arg::Ptr(dprog.as_device_ptr()),
            Arg::Ptr(dout.as_device_ptr()),
            Arg::I32(numel_i),
            Arg::I32(n_ops_i),
            Arg::I32(leaf_count_i),
        ];
        self.host.run(OPN, &FUSED_SPEC, numel, &args)?;
        dout.to_vec()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use fandhe_ai_tensor_core::DType;

    fn plan(ops: Vec<FusedOpKind>, leaf_count: usize) -> FusionPlan {
        FusionPlan::from_ops(ops, vec![4], DType::F32, leaf_count).unwrap()
    }

    #[test]
    fn fit_i32_rejects_overflow() {
        assert_eq!(fit_i32("t", 7, "n").unwrap(), 7);
        assert!(fit_i32("t", i32::MAX as usize + 1, "n").is_err());
    }

    #[test]
    fn encodes_allowlisted_plan() {
        use FusedOpKind::*;
        let p = plan(
            vec![
                Input { leaf_index: 0 },
                Input { leaf_index: 1 },
                Add { lhs: 0, rhs: 1 },
                Relu { input: 2 },
                Mul { lhs: 3, rhs: 0 },
                Exp { input: 4 },
                Tanh { input: 5 },
            ],
            2,
        );
        let e = encode_program(&p).unwrap();
        assert_eq!((e.n_ops, e.leaf_count), (7, 2));
        assert_eq!(
            e.words,
            vec![
                OP_INPUT, 0, 0, OP_INPUT, 1, 0, OP_ADD, 0, 1, OP_RELU, 2, 0, OP_MUL, 3, 0, OP_EXP,
                4, 0, OP_TANH, 5, 0
            ]
        );
    }

    #[test]
    fn rejects_non_allowlisted_ops() {
        use FusedOpKind::*;
        for tail in [
            Sub { lhs: 0, rhs: 1 },
            Div { lhs: 0, rhs: 1 },
            Rsqrt { input: 0 },
        ] {
            let p = plan(
                vec![Input { leaf_index: 0 }, Input { leaf_index: 1 }, tail],
                2,
            );
            assert!(encode_program(&p).is_err(), "{tail:?}");
        }
        let p = plan(
            vec![
                Input { leaf_index: 0 },
                Sum {
                    input: 0,
                    axis: None,
                },
            ],
            1,
        );
        assert!(encode_program(&p).is_err());
    }

    #[test]
    fn rejects_over_limit_plans() {
        use FusedOpKind::*;
        // 演算ノード 13 個（上限 12 超過）。
        let mut ops = vec![Input { leaf_index: 0 }];
        for i in 0..=MAX_FUSED_SEGMENT_NODES {
            ops.push(Relu { input: i });
        }
        let p = FusionPlan::from_ops(ops, vec![4], DType::F32, 1).unwrap();
        assert!(encode_program(&p).is_err());
    }

    #[test]
    fn max_plan_encodes() {
        use FusedOpKind::*;
        // 演算 12 個・葉 13 個。
        let mut ops: Vec<FusedOpKind> = (0..13).map(|l| Input { leaf_index: l }).collect();
        let mut prev = 0usize;
        for l in 1..13 {
            ops.push(Add { lhs: prev, rhs: l });
            prev = ops.len() - 1;
        }
        let p = FusionPlan::from_ops(ops, vec![4], DType::F32, 13).unwrap();
        let e = encode_program(&p).unwrap();
        assert_eq!(e.n_ops, 25);
        assert_eq!(e.leaf_count, 13);
    }
}
