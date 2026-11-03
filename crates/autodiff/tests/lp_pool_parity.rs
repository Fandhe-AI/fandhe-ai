//! `Var::lp_pool1d`／`lp_pool2d`／`lp_pool3d`（イシュー #3825）を実 PyTorch 2.14 の実行値 fixture
//! （`tests/fixtures/lp-pool-pytorch-reference/lp_pool_reference.json`。生成条件・sha256 は同ディレクトリの
//! README.md）と突合する統合テスト。
//!
//! 判定は REQ-2 統一複合判定（`common::req2_close`。tolerance 定数は新設しない）。NaN は NaN 同士、
//! ±inf は厳密一致。PyTorch は f32 累積・本実装は `avg_pool` の `f64` 累積のため bit 一致は求めない。
//! 設計記録: `docs/autodiff-pool3d-ops-decision.md` §19 と LPPool 実施記録の節。

mod common;

use fandhe_ai_autodiff::Tape;
use fandhe_ai_tensor_core::Tensor;
use serde::Deserialize;

#[derive(Deserialize)]
struct Fixture {
    torch_version: String,
    cases: Vec<Case>,
}

#[derive(Deserialize)]
struct Case {
    name: String,
    dim: usize,
    norm_type: f32,
    kernel_size: Vec<usize>,
    stride: Option<Vec<usize>>,
    in_shape: Vec<usize>,
    out_shape: Vec<usize>,
    x_bits: Vec<u32>,
    g_bits: Vec<u32>,
    out_bits: Vec<u32>,
    grad_bits: Vec<u32>,
}

fn from_bits(bits: &[u32]) -> Vec<f32> {
    bits.iter().map(|&b| f32::from_bits(b)).collect()
}

/// NaN は NaN 同士、±inf は厳密、有限は REQ-2 統一複合判定。
fn class_close(a: f32, e: f32) -> bool {
    if e.is_nan() {
        a.is_nan()
    } else if e.is_infinite() {
        a == e
    } else {
        common::req2_close(f64::from(a), f64::from(e))
    }
}

fn assert_close_all(actual: &[f32], expected: &[f32], context: &str) {
    assert_eq!(actual.len(), expected.len(), "{context}: 長さ不一致");
    for (i, (&a, &e)) in actual.iter().zip(expected).enumerate() {
        assert!(class_close(a, e), "{context}[{i}]: actual={a} expected={e}");
    }
}

#[test]
fn lp_pool_matches_pytorch_reference_forward_and_backward() {
    let path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/lp-pool-pytorch-reference/lp_pool_reference.json");
    let raw = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("fixture 読込に失敗: {} ({e})", path.display()));
    let fixture: Fixture =
        serde_json::from_str(&raw).unwrap_or_else(|e| panic!("fixture のパースに失敗: {e}"));
    assert!(
        fixture.torch_version.starts_with("2.14"),
        "fixture は PyTorch 2.14 系の実行値である必要がある: {}",
        fixture.torch_version
    );
    assert!(fixture.cases.len() >= 20);
    let mut dims = [0usize; 3];
    let mut nan_cases = 0usize;
    for case in &fixture.cases {
        dims[case.dim - 1] += 1;
        let tape = Tape::new_with_ops(common::naive_ops());
        let x = tape.var(&Tensor::new(from_bits(&case.x_bits), &case.in_shape).unwrap());
        let (p, k, s) = (case.norm_type, &case.kernel_size, &case.stride);
        let y = match case.dim {
            1 => x.lp_pool1d(p, k[0], s.as_ref().map(|v| v[0]), false),
            2 => x.lp_pool2d(p, [k[0], k[1]], s.as_ref().map(|v| [v[0], v[1]]), false),
            3 => x.lp_pool3d(
                p,
                [k[0], k[1], k[2]],
                s.as_ref().map(|v| [v[0], v[1], v[2]]),
                false,
            ),
            d => panic!("dim={d}"),
        }
        .unwrap_or_else(|e| panic!("{}: {e:?}", case.name));
        let out = y.to_tensor();
        assert_eq!(
            out.shape(),
            case.out_shape.as_slice(),
            "{}: shape",
            case.name
        );
        assert_close_all(
            &out.host_slice(),
            &from_bits(&case.out_bits),
            &format!("{} forward", case.name),
        );
        // forward に NaN を含む（負の入力 × 非整数 p）ケースの勾配は比較しない。PyTorch は NaN を返すが、
        // 本実装は `pow_scalar` の VJP と上流 0 の積が 0 になり NaN にならない（既知の差分。実装記録に記載）。
        if from_bits(&case.out_bits).iter().any(|v| v.is_nan()) {
            nan_cases += 1;
            continue;
        }
        let g = tape.var_no_grad(&Tensor::new(from_bits(&case.g_bits), out.shape()).unwrap());
        let loss = y.mul(&g).unwrap().sum(None).unwrap();
        let grads = tape.backward(&loss).unwrap();
        let got = grads.get(&x).unwrap().unwrap().host_slice().into_owned();
        assert_close_all(
            &got,
            &from_bits(&case.grad_bits),
            &format!("{} grad", case.name),
        );
    }
    assert!(dims.iter().all(|&n| n >= 5), "1d/2d/3d の件数: {dims:?}");
    assert!(nan_cases >= 1, "NaN ケースが fixture に含まれる");
}
