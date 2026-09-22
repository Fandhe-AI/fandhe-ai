//! `onnx::interp::run_with_ops`（イシュー #2077・親 #2076「ONNX import
//! モデルの GPU 実行」）の `BackendOps` dispatcher 統合テスト。
//!
//! CUDA／Metal 実機を必要とせず（`BackendOps` はテスト double または
//! `CpuBackendOps` を使う）、次を CI で検証する:
//!
//! 1. `run`（`dev_ops = None`）は本イシュー導入前と bit 完全に不変
//!    （実装計画契約 (a)）
//! 2. `run_with_ops(CpuBackendOps)` は対象 op（`MatMul`／`Gemm`／`Add`／
//!    `Mul`／`Div`／`Sqrt`／`Softmax`／`LayerNormalization`）を device 側
//!    （ここでは `CpuBackendOps`）へディスパッチする（[`DispatchReport`]
//!    で確認）
//! 3. `BackendError::Unsupported` を返す `BackendOps` 実装
//!    （`UnsupportedOps`）ではホスト実装へフォールバックし、全ノードが
//!    `host_nodes` に記録される
//! 4. `BackendError::Unsupported`／`ShapeMismatch` 以外のエラーを返す
//!    `BackendOps` 実装（`FailingOps`）では [`InterpError::Backend`] が
//!    伝播する（ホストへの黙示フォールバックはしない。OWASP A08）
//! 5. `Relu` の `NaN` 伝播（[`fandhe_ai_tensor_core::ScalarUnaryOp::Relu`]
//!    契約）が opt-in ON でも維持される（`BackendOps::relu` の NaN→0
//!    契約を誤って使っていないことの回帰）
//!
//! 数値契約: 加算・乗算・除算・平方根（reduction を伴わない elementwise
//! op）は `CpuBackendOps` と `ops::*`（ホスト参照実装）が同一の素朴な
//! 演算のため bit 完全一致を要求する。GEMM／MatMul／Softmax／
//! LayerNormalization は reduction の結合順序が異なる（`CpuBackendOps`
//! は本番 BLIS ブロッキング／rayon 並列カーネル、ホストは逐次
//! `mul_add` ループ）ため bit 一致を要求せず、`cpu_row_kernel_naive_
//! parity.rs` と同じ REQ-2 統一複合判定（`fandhe_ai_backend_cpu::parity::
//! assert_parity`）で突合する（`.claude/rules/coding-rust.md`）。

use std::cell::RefCell;
use std::collections::HashMap;

use fandhe_ai_backend_cpu::CpuBackendOps;
use fandhe_ai_backend_cpu::parity::assert_parity;
use fandhe_ai_onnx_interop::onnx::graph::Graph;
use fandhe_ai_onnx_interop::onnx::interp::{
    InterpError, Value, run, run_with_ops, run_with_ops_report,
};
use fandhe_ai_onnx_interop::onnx::proto::{AttributeProto, NodeProto};
use fandhe_ai_tensor_core::{
    BackendError, BackendOps, Device, ScalarBinaryOp, ScalarUnaryOp, Tensor,
};

// ---- グラフ構築ヘルパ（`interp.rs` 内部テストの `node`／`node_with_attrs` と同型） ----

fn node(op_type: &str, name: &str, input: Vec<&str>, output: Vec<&str>) -> NodeProto {
    NodeProto {
        input: input.into_iter().map(String::from).collect(),
        output: output.into_iter().map(String::from).collect(),
        name: name.to_string(),
        op_type: op_type.to_string(),
        attribute: vec![],
        domain: String::new(),
    }
}

fn node_with_attrs(
    op_type: &str,
    name: &str,
    input: Vec<&str>,
    output: Vec<&str>,
    attribute: Vec<AttributeProto>,
) -> NodeProto {
    let mut n = node(op_type, name, input, output);
    n.attribute = attribute;
    n
}

fn attr_i64(name: &str, i: i64) -> AttributeProto {
    AttributeProto {
        name: name.to_string(),
        i,
        ..Default::default()
    }
}

/// 単一ノードのグラフ（initializer なし。全入力を feed で渡す）。
fn single_node_graph(n: NodeProto, inputs: &[&str]) -> Graph {
    let outputs = n.output.clone();
    Graph {
        nodes: vec![n],
        initializers: HashMap::new(),
        inputs: inputs.iter().map(|s| s.to_string()).collect(),
        outputs,
    }
}

fn feed_f32(name: &str, data: Vec<f32>, shape: &[usize]) -> (String, Value) {
    (
        name.to_string(),
        Value::F32(Tensor::<f32>::new(data, shape).unwrap()),
    )
}

fn as_f32_slice(v: &Value) -> Vec<f32> {
    match v {
        Value::F32(t) => t.contiguous().as_slice().unwrap().to_vec(),
        other => panic!("Value::F32 を期待したが {other:?}"),
    }
}

// ---- テスト double: `BackendOps` ----

/// `CpuBackendOps` へ委譲しつつ、呼び出されたメソッド名を記録する
/// `BackendOps` 実装（`run_with_ops_report` とは独立に、実際に device
/// 経路の kernel が呼ばれたことを直接検証するための double）。
struct RecordingOps {
    inner: CpuBackendOps,
    calls: RefCell<Vec<&'static str>>,
}

impl RecordingOps {
    fn new() -> Self {
        RecordingOps {
            inner: CpuBackendOps::new(),
            calls: RefCell::new(Vec::new()),
        }
    }

    fn record(&self, name: &'static str) {
        self.calls.borrow_mut().push(name);
    }
}

impl BackendOps for RecordingOps {
    fn device(&self) -> Device {
        self.inner.device()
    }
    fn gemm(&self, a: &Tensor<f32>, b: &Tensor<f32>) -> Result<Tensor<f32>, BackendError> {
        self.inner.gemm(a, b)
    }
    fn gemm_fp32_strict(
        &self,
        a: &Tensor<f32>,
        b: &Tensor<f32>,
    ) -> Result<Tensor<f32>, BackendError> {
        self.record("gemm_fp32_strict");
        self.inner.gemm_fp32_strict(a, b)
    }
    fn gemm_batched_fp32_strict(
        &self,
        a: &Tensor<f32>,
        b: &Tensor<f32>,
    ) -> Result<Tensor<f32>, BackendError> {
        self.record("gemm_batched_fp32_strict");
        self.inner.gemm_batched_fp32_strict(a, b)
    }
    fn add(&self, a: &Tensor<f32>, b: &Tensor<f32>) -> Result<Tensor<f32>, BackendError> {
        self.record("add");
        self.inner.add(a, b)
    }
    fn mul(&self, a: &Tensor<f32>, b: &Tensor<f32>) -> Result<Tensor<f32>, BackendError> {
        self.record("mul");
        self.inner.mul(a, b)
    }
    fn relu(&self, a: &Tensor<f32>) -> Result<Tensor<f32>, BackendError> {
        self.inner.relu(a)
    }
    fn exp(&self, a: &Tensor<f32>) -> Result<Tensor<f32>, BackendError> {
        self.inner.exp(a)
    }
    fn tanh(&self, a: &Tensor<f32>) -> Result<Tensor<f32>, BackendError> {
        self.inner.tanh(a)
    }
    fn sum(&self, a: &Tensor<f32>, dim: Option<usize>) -> Result<Tensor<f32>, BackendError> {
        self.inner.sum(a, dim)
    }
    fn max(&self, a: &Tensor<f32>, dim: Option<usize>) -> Result<Tensor<f32>, BackendError> {
        self.inner.max(a, dim)
    }
    fn scalar_unary(
        &self,
        op: ScalarUnaryOp,
        a: &Tensor<f32>,
    ) -> Result<Tensor<f32>, BackendError> {
        self.record("scalar_unary");
        self.inner.scalar_unary(op, a)
    }
    fn scalar_binary(
        &self,
        op: ScalarBinaryOp,
        a: &Tensor<f32>,
        b: &Tensor<f32>,
    ) -> Result<Tensor<f32>, BackendError> {
        self.record("scalar_binary");
        self.inner.scalar_binary(op, a, b)
    }
    fn softmax(&self, x: &Tensor<f32>, dim: usize) -> Result<Tensor<f32>, BackendError> {
        self.record("softmax");
        self.inner.softmax(x, dim)
    }
    fn layer_norm(
        &self,
        x: &Tensor<f32>,
        weight: Option<&Tensor<f32>>,
        bias: Option<&Tensor<f32>>,
        eps: f32,
    ) -> Result<Tensor<f32>, BackendError> {
        self.record("layer_norm");
        self.inner.layer_norm(x, weight, bias, eps)
    }
}

/// 全メソッドが `BackendError::Unsupported` を返す `BackendOps`
/// （`crates/tensor-core/src/backend_ops.rs::tests::MockOps` と同型の
/// テスト double）。device 経路が常にホストへフォールバックすることを
/// 検証する。
struct UnsupportedOps;

impl BackendOps for UnsupportedOps {
    fn device(&self) -> Device {
        Device::Cpu
    }
    fn gemm(&self, _a: &Tensor<f32>, _b: &Tensor<f32>) -> Result<Tensor<f32>, BackendError> {
        Err(BackendError::Unsupported("test: gemm".into()))
    }
    fn add(&self, _a: &Tensor<f32>, _b: &Tensor<f32>) -> Result<Tensor<f32>, BackendError> {
        Err(BackendError::Unsupported("test: add".into()))
    }
    fn mul(&self, _a: &Tensor<f32>, _b: &Tensor<f32>) -> Result<Tensor<f32>, BackendError> {
        Err(BackendError::Unsupported("test: mul".into()))
    }
    fn relu(&self, _a: &Tensor<f32>) -> Result<Tensor<f32>, BackendError> {
        Err(BackendError::Unsupported("test: relu".into()))
    }
    fn exp(&self, _a: &Tensor<f32>) -> Result<Tensor<f32>, BackendError> {
        Err(BackendError::Unsupported("test: exp".into()))
    }
    fn tanh(&self, _a: &Tensor<f32>) -> Result<Tensor<f32>, BackendError> {
        Err(BackendError::Unsupported("test: tanh".into()))
    }
    fn sum(&self, _a: &Tensor<f32>, _dim: Option<usize>) -> Result<Tensor<f32>, BackendError> {
        Err(BackendError::Unsupported("test: sum".into()))
    }
    fn max(&self, _a: &Tensor<f32>, _dim: Option<usize>) -> Result<Tensor<f32>, BackendError> {
        Err(BackendError::Unsupported("test: max".into()))
    }
    // `gemm_fp32_strict`／`scalar_unary`／`scalar_binary`／`softmax`／
    // `layer_norm` はすべて既定実装（`Unsupported` fail-safe）のまま。
}

/// `gemm_fp32_strict` が `Unsupported`／`ShapeMismatch` 以外のエラー
/// （`KernelLaunchFailed`）を返す `BackendOps`。`InterpError::Backend`
/// への fail-closed 伝播（ホストへの黙示フォールバックをしないこと）を
/// 検証する。他の必須メソッドは `CpuBackendOps` へ委譲する（テスト対象
/// ノード以外に影響を与えないため）。
struct FailingGemmOps {
    inner: CpuBackendOps,
}

impl BackendOps for FailingGemmOps {
    fn device(&self) -> Device {
        self.inner.device()
    }
    fn gemm(&self, a: &Tensor<f32>, b: &Tensor<f32>) -> Result<Tensor<f32>, BackendError> {
        self.inner.gemm(a, b)
    }
    fn gemm_fp32_strict(
        &self,
        _a: &Tensor<f32>,
        _b: &Tensor<f32>,
    ) -> Result<Tensor<f32>, BackendError> {
        Err(BackendError::KernelLaunchFailed(
            "test: simulated device failure".into(),
        ))
    }
    fn add(&self, a: &Tensor<f32>, b: &Tensor<f32>) -> Result<Tensor<f32>, BackendError> {
        self.inner.add(a, b)
    }
    fn mul(&self, a: &Tensor<f32>, b: &Tensor<f32>) -> Result<Tensor<f32>, BackendError> {
        self.inner.mul(a, b)
    }
    fn relu(&self, a: &Tensor<f32>) -> Result<Tensor<f32>, BackendError> {
        self.inner.relu(a)
    }
    fn exp(&self, a: &Tensor<f32>) -> Result<Tensor<f32>, BackendError> {
        self.inner.exp(a)
    }
    fn tanh(&self, a: &Tensor<f32>) -> Result<Tensor<f32>, BackendError> {
        self.inner.tanh(a)
    }
    fn sum(&self, a: &Tensor<f32>, dim: Option<usize>) -> Result<Tensor<f32>, BackendError> {
        self.inner.sum(a, dim)
    }
    fn max(&self, a: &Tensor<f32>, dim: Option<usize>) -> Result<Tensor<f32>, BackendError> {
        self.inner.max(a, dim)
    }
}

// ---- テスト本体 ----

#[test]
fn run_without_ops_matmul_unchanged() {
    // 契約 (a): `dev_ops = None` の `run` は本イシュー導入前と同一実装
    // を通ることを、単純な MatMul グラフで確認する（回帰の固定化）。
    let n = node("MatMul", "n_matmul", vec!["a", "b"], vec!["y"]);
    let graph = single_node_graph(n, &["a", "b"]);
    let mut feeds = HashMap::new();
    feeds.extend([
        feed_f32("a", vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]),
        feed_f32("b", vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0], &[3, 2]),
    ]);
    let result = run(&graph, feeds).expect("run は成功するはず");
    let y = as_f32_slice(&result["y"]);
    // gemm.rs の plain_matmul_no_bias と同一想定値。
    assert_eq!(y, vec![58.0, 64.0, 139.0, 154.0]);
}

#[test]
fn run_with_ops_matmul_reaches_device_and_matches_within_req2() {
    let n = node("MatMul", "n_matmul", vec!["a", "b"], vec!["y"]);
    let graph = single_node_graph(n, &["a", "b"]);
    let ops = RecordingOps::new();

    let mut feeds = HashMap::new();
    feeds.extend([
        feed_f32("a", vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]),
        feed_f32("b", vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0], &[3, 2]),
    ]);
    let result = run_with_ops(&graph, feeds, &ops).expect("run_with_ops は成功するはず");
    let actual = as_f32_slice(&result["y"]);

    assert_eq!(
        ops.calls.borrow().as_slice(),
        &["gemm_fp32_strict"],
        "MatMul（rank 2x2）は gemm_fp32_strict へ到達するはず"
    );
    assert_parity(
        "MatMul: CpuBackendOps::gemm_fp32_strict vs ops::matmul",
        &actual,
        &[58.0, 64.0, 139.0, 154.0],
    );
}

#[test]
fn run_with_ops_gemm_batched_matmul_reaches_device() {
    // rank 3 の MatMul は `gemm_batched_fp32_strict` へ到達する。
    let n = node("MatMul", "n_matmul3d", vec!["a", "b"], vec!["y"]);
    let graph = single_node_graph(n, &["a", "b"]);
    let ops = RecordingOps::new();

    let a = Tensor::<f32>::new(vec![1.0, 0.0, 0.0, 1.0], &[1, 2, 2]).unwrap();
    let b = Tensor::<f32>::new(vec![2.0, 3.0, 4.0, 5.0], &[1, 2, 2]).unwrap();
    let mut feeds = HashMap::new();
    feeds.insert("a".to_string(), Value::F32(a));
    feeds.insert("b".to_string(), Value::F32(b));

    let result = run_with_ops(&graph, feeds, &ops).expect("run_with_ops は成功するはず");
    let actual = as_f32_slice(&result["y"]);

    assert_eq!(ops.calls.borrow().as_slice(), &["gemm_batched_fp32_strict"]);
    assert_parity(
        "MatMul(3D): CpuBackendOps::gemm_batched_fp32_strict vs ops::matmul",
        &actual,
        &[2.0, 3.0, 4.0, 5.0],
    );
}

#[test]
fn run_with_ops_gemm_alpha_beta_bias_reaches_device_and_matches_host() {
    let n = node_with_attrs(
        "Gemm",
        "n_gemm",
        vec!["a", "b", "c"],
        vec!["y"],
        vec![
            AttributeProto {
                name: "alpha".to_string(),
                f: 2.0,
                ..Default::default()
            },
            AttributeProto {
                name: "beta".to_string(),
                f: 0.5,
                ..Default::default()
            },
        ],
    );
    let graph = single_node_graph(n, &["a", "b", "c"]);
    let ops = RecordingOps::new();

    let mut feeds = HashMap::new();
    feeds.extend([
        feed_f32("a", vec![1.0, 0.0, 0.0, 1.0], &[2, 2]),
        feed_f32("b", vec![2.0, 3.0, 4.0, 5.0], &[2, 2]),
        feed_f32("c", vec![1.0, 1.0], &[2]),
    ]);
    let result = run_with_ops(&graph, feeds, &ops).expect("run_with_ops は成功するはず");
    let actual = as_f32_slice(&result["y"]);

    assert_eq!(ops.calls.borrow().as_slice(), &["gemm_fp32_strict"]);
    // alpha * (A@B) + beta * bias(broadcast) と同じ手計算値（gemm.rs
    // alpha_beta_and_bias_broadcast テストと同一想定値）。
    assert_eq!(
        actual,
        vec![
            2.0 * 2.0 + 0.5,
            2.0 * 3.0 + 0.5,
            2.0 * 4.0 + 0.5,
            2.0 * 5.0 + 0.5
        ]
    );
}

#[test]
fn run_with_ops_elementwise_ops_reach_device_bit_identical() {
    // Add/Mul/Div/Sqrt は reduction を伴わない elementwise であり、
    // `CpuBackendOps` とホスト実装（`ops::*`）は同一演算のため bit 完全
    // 一致を期待する。
    for (op_type, expected_call, a, b, expected) in [
        ("Add", "add", vec![1.0, 2.0], vec![3.0, 4.0], vec![4.0, 6.0]),
        ("Mul", "mul", vec![1.0, 2.0], vec![3.0, 4.0], vec![3.0, 8.0]),
        (
            "Div",
            "scalar_binary",
            vec![6.0, 9.0],
            vec![3.0, 3.0],
            vec![2.0, 3.0],
        ),
    ] {
        let n = node(op_type, &format!("n_{op_type}"), vec!["a", "b"], vec!["y"]);
        let graph = single_node_graph(n, &["a", "b"]);
        let ops = RecordingOps::new();
        let mut feeds = HashMap::new();
        feeds.extend([feed_f32("a", a, &[2]), feed_f32("b", b, &[2])]);
        let result = run_with_ops(&graph, feeds, &ops).expect("run_with_ops は成功するはず");
        let actual = as_f32_slice(&result["y"]);
        assert_eq!(ops.calls.borrow().as_slice(), &[expected_call], "{op_type}");
        assert_eq!(actual, expected, "{op_type}");
    }

    // Sqrt（単項）。
    let n = node("Sqrt", "n_sqrt", vec!["x"], vec!["y"]);
    let graph = single_node_graph(n, &["x"]);
    let ops = RecordingOps::new();
    let mut feeds = HashMap::new();
    feeds.extend([feed_f32("x", vec![4.0, 9.0], &[2])]);
    let result = run_with_ops(&graph, feeds, &ops).expect("run_with_ops は成功するはず");
    let actual = as_f32_slice(&result["y"]);
    assert_eq!(ops.calls.borrow().as_slice(), &["scalar_unary"]);
    assert_eq!(actual, vec![2.0, 3.0]);
}

#[test]
fn run_with_ops_softmax_last_axis_reaches_device() {
    let n = node_with_attrs(
        "Softmax",
        "n_softmax",
        vec!["x"],
        vec!["y"],
        vec![attr_i64("axis", -1)],
    );
    let graph = single_node_graph(n, &["x"]);
    let ops = RecordingOps::new();
    let mut feeds = HashMap::new();
    feeds.extend([feed_f32("x", vec![1.0, 2.0, 3.0, 4.0], &[2, 2])]);
    let result = run_with_ops(&graph, feeds, &ops).expect("run_with_ops は成功するはず");
    let actual = as_f32_slice(&result["y"]);
    assert_eq!(ops.calls.borrow().as_slice(), &["softmax"]);

    let x = Tensor::<f32>::new(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
    let expected = fandhe_ai_onnx_interop::ops::softmax(&x, -1).unwrap();
    let expected = expected.contiguous();
    assert_parity(
        "Softmax: CpuBackendOps::softmax vs ops::softmax",
        &actual,
        expected.as_slice().unwrap(),
    );
}

#[test]
fn run_with_ops_softmax_non_last_axis_stays_on_host() {
    // 非最終軸は `BackendOps::softmax` の既定契約（`Unsupported`）に
    // より、opt-in ON でもホスト実装へフォールバックする。
    let n = node_with_attrs(
        "Softmax",
        "n_softmax_axis0",
        vec!["x"],
        vec!["y"],
        vec![attr_i64("axis", 0)],
    );
    let graph = single_node_graph(n, &["x"]);
    let ops = RecordingOps::new();
    let mut feeds = HashMap::new();
    feeds.extend([feed_f32("x", vec![1.0, 2.0, 3.0, 4.0], &[2, 2])]);
    let (result, report) =
        run_with_ops_report(&graph, feeds, &ops).expect("run_with_ops_report は成功するはず");
    let actual = as_f32_slice(&result["y"]);
    // `ops.softmax(x, 0)` 自体は呼ばれる（`BackendOps::softmax` の既定
    // 契約どおり内部で `Unsupported` を返すことで判別する設計。
    // `RecordingOps::calls` は「呼ばれたか」を記録するため非空になる）が、
    // 最終的な出力・`DispatchReport` は host 実行であることを示す。
    assert_eq!(ops.calls.borrow().as_slice(), &["softmax"]);
    assert_eq!(report.host_nodes, vec!["n_softmax_axis0".to_string()]);
    assert!(report.device_nodes.is_empty());

    let x = Tensor::<f32>::new(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
    let expected = fandhe_ai_onnx_interop::ops::softmax(&x, 0).unwrap();
    let expected = expected.contiguous();
    assert_eq!(actual, expected.as_slice().unwrap());
}

#[test]
fn run_with_ops_layer_normalization_last_axis_reaches_device() {
    let n = node_with_attrs(
        "LayerNormalization",
        "n_ln",
        vec!["x", "scale"],
        vec!["y"],
        vec![attr_i64("axis", -1)],
    );
    let graph = single_node_graph(n, &["x", "scale"]);
    let ops = RecordingOps::new();
    let mut feeds = HashMap::new();
    feeds.extend([
        feed_f32("x", vec![1.0, 2.0, 3.0, 4.0], &[1, 4]),
        feed_f32("scale", vec![1.0, 1.0, 1.0, 1.0], &[4]),
    ]);
    let result = run_with_ops(&graph, feeds, &ops).expect("run_with_ops は成功するはず");
    let actual = as_f32_slice(&result["y"]);
    assert_eq!(ops.calls.borrow().as_slice(), &["layer_norm"]);

    let x = Tensor::<f32>::new(vec![1.0, 2.0, 3.0, 4.0], &[1, 4]).unwrap();
    let scale = Tensor::<f32>::new(vec![1.0, 1.0, 1.0, 1.0], &[4]).unwrap();
    let attrs = fandhe_ai_onnx_interop::ops::LayerNormAttrs {
        axis: -1,
        epsilon: 1e-5,
    };
    let expected =
        fandhe_ai_onnx_interop::ops::layer_normalization(&x, &scale, None, &attrs).unwrap();
    let expected = expected.contiguous();
    assert_parity(
        "LayerNormalization: CpuBackendOps::layer_norm vs ops::layer_normalization",
        &actual,
        expected.as_slice().unwrap(),
    );
}

#[test]
fn run_with_ops_layer_normalization_multi_axis_stays_on_host() {
    // axis が最終軸以外（正規化集合が複数軸にまたがる）の場合は常に
    // ホストへ委ねる（`interp_device::device_layer_norm` の必須ガード）。
    let n = node_with_attrs(
        "LayerNormalization",
        "n_ln_multi",
        vec!["x", "scale"],
        vec!["y"],
        vec![attr_i64("axis", 1)],
    );
    let graph = single_node_graph(n, &["x", "scale"]);
    let ops = RecordingOps::new();
    let data: Vec<f32> = (0..24).map(|v| v as f32).collect();
    let mut feeds = HashMap::new();
    feeds.insert(
        "x".to_string(),
        Value::F32(Tensor::<f32>::new(data, &[2, 3, 4]).unwrap()),
    );
    feeds.insert(
        "scale".to_string(),
        Value::F32(Tensor::<f32>::new(vec![1.0; 12], &[3, 4]).unwrap()),
    );
    let (_, report) =
        run_with_ops_report(&graph, feeds, &ops).expect("run_with_ops_report は成功するはず");
    assert!(ops.calls.borrow().is_empty());
    assert_eq!(report.host_nodes, vec!["n_ln_multi".to_string()]);
}

#[test]
fn run_with_ops_unsupported_ops_falls_back_to_host_for_all_dispatch_eligible_nodes() {
    // `UnsupportedOps` は全メソッドが `Unsupported` を返すため、opt-in ON
    // でも全ノードがホストへフォールバックし、`run`（opt-in OFF）と同一
    // 結果になる。
    let n = node("MatMul", "n_matmul", vec!["a", "b"], vec!["y"]);
    let graph = single_node_graph(n, &["a", "b"]);
    let mut feeds_a = HashMap::new();
    feeds_a.extend([
        feed_f32("a", vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]),
        feed_f32("b", vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0], &[3, 2]),
    ]);
    let mut feeds_b = HashMap::new();
    feeds_b.extend([
        feed_f32("a", vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]),
        feed_f32("b", vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0], &[3, 2]),
    ]);

    let ops = UnsupportedOps;
    let (result_on, report) = run_with_ops_report(&graph, feeds_a, &ops)
        .expect("UnsupportedOps でも run_with_ops はホストへフォールバックして成功するはず");
    let result_off = run(&graph, feeds_b).expect("run は成功するはず");

    assert_eq!(report.host_nodes, vec!["n_matmul".to_string()]);
    assert!(report.device_nodes.is_empty());
    assert_eq!(
        as_f32_slice(&result_on["y"]),
        as_f32_slice(&result_off["y"])
    );
}

#[test]
fn run_with_ops_non_unsupported_backend_error_propagates_as_backend_error() {
    // `FailingGemmOps` は `gemm_fp32_strict` で `KernelLaunchFailed` を
    // 返す。`Unsupported`／`ShapeMismatch` 以外のエラーはホストへの黙示
    // フォールバックをせず `InterpError::Backend` として伝播する
    // （OWASP A08。実装計画 §3.2）。
    let n = node("MatMul", "n_matmul", vec!["a", "b"], vec!["y"]);
    let graph = single_node_graph(n, &["a", "b"]);
    let mut feeds = HashMap::new();
    feeds.extend([
        feed_f32("a", vec![1.0, 2.0, 3.0, 4.0], &[2, 2]),
        feed_f32("b", vec![1.0, 0.0, 0.0, 1.0], &[2, 2]),
    ]);
    let ops = FailingGemmOps {
        inner: CpuBackendOps::new(),
    };
    let err = run_with_ops(&graph, feeds, &ops).expect_err("device 故障は伝播するはず");
    match err {
        InterpError::Backend { node, message } => {
            assert_eq!(node, "n_matmul");
            assert!(
                message.contains("simulated device failure"),
                "message={message}"
            );
        }
        other => panic!("InterpError::Backend を期待したが {other:?}"),
    }
}

#[test]
fn run_with_ops_relu_preserves_nan_propagation_contract() {
    // `BackendOps::relu`（NaN を 0 に潰す）ではなく
    // `scalar_unary(ScalarUnaryOp::Relu)`（NaN 伝播）を経由することを、
    // NaN 入力の出力が NaN のままであることで確認する（ONNX `Relu`・
    // `ops::relu` の契約と同じ。`interp_device::device_relu` 冒頭コメント
    // 参照）。
    let n = node("Relu", "n_relu", vec!["x"], vec!["y"]);
    let graph = single_node_graph(n, &["x"]);
    let ops = RecordingOps::new();
    let mut feeds = HashMap::new();
    feeds.extend([feed_f32("x", vec![f32::NAN, -1.0, 2.0], &[3])]);
    let result = run_with_ops(&graph, feeds, &ops).expect("run_with_ops は成功するはず");
    let actual = as_f32_slice(&result["y"]);
    assert_eq!(ops.calls.borrow().as_slice(), &["scalar_unary"]);
    assert!(actual[0].is_nan(), "NaN は伝播するはず（0 に潰されない）");
    assert_eq!(actual[1], 0.0);
    assert_eq!(actual[2], 2.0);
}

#[test]
fn scalar_binary_div_kind_matches_ops_semantics() {
    // `Div` の device 経路が `ScalarBinaryOp::Div` を使い、`ops::div`
    // （IEEE 754 透過）と同じ 0 除算セマンティクスを保つことを確認する。
    let n = node("Div", "n_div", vec!["a", "b"], vec!["y"]);
    let graph = single_node_graph(n, &["a", "b"]);
    let ops = RecordingOps::new();
    let mut feeds = HashMap::new();
    feeds.extend([
        feed_f32("a", vec![1.0, -1.0], &[2]),
        feed_f32("b", vec![0.0, 0.0], &[2]),
    ]);
    let result = run_with_ops(&graph, feeds, &ops).expect("run_with_ops は成功するはず");
    let actual = as_f32_slice(&result["y"]);
    assert_eq!(ops.calls.borrow().as_slice(), &["scalar_binary"]);
    assert!(actual[0].is_infinite() && actual[0] > 0.0);
    assert!(actual[1].is_infinite() && actual[1] < 0.0);
}

/// `PoolStats`／`Ordering` 等の未使用 import が残らないための素通し
/// （`ScalarUnaryOp`／`ScalarBinaryOp` の再エクスポート面を直接参照する
/// ことで、facade 側 `interp_device` の enum 選択が誤っていないかも
/// 間接的に確認する）。
#[test]
fn scalar_unary_binary_enum_variants_are_reachable() {
    let _ = ScalarUnaryOp::Sqrt;
    let _ = ScalarUnaryOp::Relu;
    let _ = ScalarBinaryOp::Div;
}
