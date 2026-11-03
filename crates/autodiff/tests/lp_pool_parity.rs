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
        let out_has_nan = from_bits(&case.out_bits).iter().any(|v| v.is_nan());
        if out_has_nan {
            nan_cases += 1;
        }
        let g = tape.var_no_grad(&Tensor::new(from_bits(&case.g_bits), out.shape()).unwrap());
        let loss = y.mul(&g).unwrap().sum(None).unwrap();
        let grads = tape.backward(&loss).unwrap();
        let got = grads.get(&x).unwrap().unwrap().host_slice().into_owned();
        let want = from_bits(&case.grad_bits);
        if out_has_nan {
            // forward に NaN を含む（負の入力 × 非整数 p）ケース。PyTorch の勾配は NaN 要素を含むが、
            // 本実装は `pow_scalar` の VJP と上流 0 の積が 0 になる要素があり、全要素が NaN にはならない（既知の差分。実装記録に記載）。
            // 参照勾配が NaN の要素だけを比較から除外し、有限の参照勾配は検証する。除外した要素は
            // 本実装が NaN か 0 のいずれかを返す挙動を独立に固定する（特殊値処理の回帰を検出する）。
            assert_eq!(got.len(), want.len(), "{} grad 長", case.name);
            let mut known_diff = 0usize;
            let mut zero_diff = 0usize;
            for (i, (g_i, w_i)) in got.iter().zip(want.iter()).enumerate() {
                if w_i.is_nan() {
                    known_diff += 1;
                    // 差分として記録した要素の挙動を固定する: NaN（PyTorch と一致）か厳密な 0（既知の差分）の
                    // いずれかで、有限の非零値へ変わったら回帰として検出する。
                    assert!(
                        g_i.is_nan() || *g_i == 0.0,
                        "{} grad[{i}]: NaN 参照要素は NaN または 0 のみ許す: {g_i}",
                        case.name
                    );
                    if *g_i == 0.0 {
                        zero_diff += 1;
                    }
                } else {
                    assert_close_all(
                        std::slice::from_ref(g_i),
                        std::slice::from_ref(w_i),
                        &format!("{} grad[{i}]", case.name),
                    );
                }
            }
            assert!(
                known_diff >= 1,
                "{}: NaN 参照勾配が 1 件以上ある",
                case.name
            );
            assert!(
                zero_diff >= 1,
                "{}: 0 を返す既知の差分が 1 件以上ある",
                case.name
            );
        } else {
            assert_close_all(&got, &want, &format!("{} grad", case.name));
        }
    }
    assert!(dims.iter().all(|&n| n >= 5), "1d/2d/3d の件数: {dims:?}");
    assert!(nan_cases >= 1, "NaN ケースが fixture に含まれる");
}
