//! `functional_ops::{vjp, hvp, vmap}`（イシュー #2877・親 #2841）の第三者比較テスト。
//! 契約の正は `docs/autodiff-functional-transforms-design.md` §6（数値一致の判定方式）・§18。
//!
//! - F0: fixture の健全性（PyTorch 版数・各セクション非空・`INDETERMINATE` の実在）。
//! - F1／F2／F3: 実 PyTorch 2.14.0 の実行値 fixture
//!   （`tests/fixtures/functional-transforms-pytorch-reference/functional_transforms_reference.json`。
//!   生成条件は同ディレクトリの `README.md`）と REQ-2 統一複合判定（`common::req2_close`。
//!   tolerance 定数は新設しない）で全要素突合する。forward 値（`out`）も先に突合し、
//!   プログラムが PyTorch 側と同じ関数であることを確認する。
//! - F4: 判定不能の自前参照（vjp は `jacobian` の転置積・hvp は `hessian · v`・vmap は
//!   バッチなしのスライス単位実行）が fixture と独立に成立すること。
//!
//! **判定不能の扱い**: 第三者（PyTorch）の出力が統一複合判定を外れた場合は比較データの妥当性上の
//! 「判定不能」であり fandhe-ai 側の REQ-2 違反ではない（`.claude/rules/coding-rust.md`・spec REQ-2
//! 2026-09-12 追記）。`INDETERMINATE` へ載せたケースは PyTorch とのゲートから外し、代わりに自前の参照で
//! 同じ統一複合判定を要求する（ゲートを外すのではなく付け替える）。項目は実際に fail を観測し、
//! `expected_f64` と比べて PyTorch の f32 値自体が真値から外れていることを確認した場合だけ足す。
//!
//! 実 `CpuBackendOps` との一致は `crates/facade/tests/functional_ops_backend_parity.rs`、
//! CUDA／Metal 実機（`#[ignore]`）は #2881 が担当する。

mod common;

use std::collections::HashMap;
use std::path::PathBuf;

use fandhe_ai_autodiff::functional_ops::{hvp, vjp, vmap};
use fandhe_ai_autodiff::jacobian_ops::{hessian, jacobian};
use fandhe_ai_autodiff::{AutodiffError, Tape, Var};
use fandhe_ai_tensor_core::Tensor;
use serde::Deserialize;

/// PyTorch 側の f32 値が真値から外れ、判定不能と確認できたケース
/// （変換名 `"vjp"`／`"hvp"`／`"vmap"`・ケース名・理由）。初期値は空。
const INDETERMINATE: &[(&str, &str, &str)] = &[];

#[derive(Deserialize)]
struct Packed {
    shape: Vec<usize>,
    bits: Vec<u32>,
}

impl Packed {
    fn values(&self) -> Vec<f32> {
        self.bits.iter().map(|&b| f32::from_bits(b)).collect()
    }
    fn tensor(&self) -> Tensor<f32> {
        let n = self
            .shape
            .iter()
            .try_fold(1usize, |a, &d| a.checked_mul(d))
            .expect("fixture: 要素数がオーバーフローしない");
        assert_eq!(n, self.bits.len(), "fixture: shape とデータ長が一致する");
        Tensor::new(self.values(), &self.shape).expect("fixture: shape とデータ長は一致している")
    }
}

/// 診断用の f64 真値（ゲートには使わない）。
#[derive(Deserialize)]
struct PackedF64 {
    shape: Vec<usize>,
    values: Vec<f64>,
}

#[derive(Deserialize)]
struct VjpCase {
    name: String,
    x: Packed,
    consts: HashMap<String, Packed>,
    out: Packed,
    u: Packed,
    expected: Packed,
    expected_f64: PackedF64,
}

#[derive(Deserialize)]
struct HvpCase {
    name: String,
    x: Packed,
    consts: HashMap<String, Packed>,
    out: Packed,
    v: Packed,
    expected: Packed,
    expected_f64: PackedF64,
}

#[derive(Deserialize)]
struct VmapCase {
    name: String,
    x: Packed,
    in_dim: usize,
    consts: HashMap<String, Packed>,
    expected: Packed,
    expected_f64: PackedF64,
}

#[derive(Deserialize)]
struct Fixture {
    torch_version: String,
    vjp_cases: Vec<VjpCase>,
    hvp_cases: Vec<HvpCase>,
    vmap_cases: Vec<VmapCase>,
}

fn load_fixture() -> Fixture {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(
        "tests/fixtures/functional-transforms-pytorch-reference/functional_transforms_reference.json",
    );
    let raw = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("fixture 読込に失敗: {} ({e})", path.display()));
    serde_json::from_str(&raw).unwrap_or_else(|e| panic!("fixture のパースに失敗: {e}"))
}

fn new_tape() -> Tape {
    Tape::new_with_ops(common::naive_ops())
}

fn host(x: &Tensor<f32>) -> Vec<f32> {
    x.host_slice().into_owned()
}

fn consts_on<'t>(tape: &'t Tape, consts: &HashMap<String, Packed>) -> HashMap<String, Var<'t>> {
    consts
        .iter()
        .map(|(k, v)| (k.clone(), tape.var_no_grad(&v.tensor())))
        .collect()
}

fn is_indeterminate(kind: &str, name: &str) -> bool {
    INDETERMINATE
        .iter()
        .any(|&(k, n, _)| k == kind && n == name)
}

/// forward 値の突合（プログラムが PyTorch 側と同じ関数であることの確認）。
fn assert_close_packed(actual: &Tensor<f32>, expected: &Packed, ctx: &str) {
    assert_eq!(actual.shape(), expected.shape.as_slice(), "{ctx}: shape");
    let a = host(actual);
    let e = expected.values();
    assert_eq!(a.len(), e.len(), "{ctx}: 要素数");
    for (i, (&x, &y)) in a.iter().zip(e.iter()).enumerate() {
        assert!(
            common::req2_close(x as f64, y as f64),
            "{ctx}[{i}]: actual={x} expected={y}"
        );
    }
}

/// 不一致要素の一覧（要素番号・実測・参照）。
fn mismatches(actual: &[f32], reference: &[f32]) -> Vec<usize> {
    actual
        .iter()
        .zip(reference.iter())
        .enumerate()
        .filter(|&(_, (&a, &r))| !common::req2_close(a as f64, r as f64))
        .map(|(i, _)| i)
        .collect()
}

/// 変換結果のゲート。通常は PyTorch 値と全要素突合する。`INDETERMINATE` のケースは PyTorch との
/// ゲートを自前参照（`own`）へ付け替える（付け替え対象が実際に PyTorch と食い違っていることも検査し、
/// 陳腐化した項目を残さない）。fail 時は `expected_f64` を併記して診断する。
fn gate(
    kind: &str,
    name: &str,
    actual: &Tensor<f32>,
    expected: &Packed,
    truth: &PackedF64,
    own: impl FnOnce() -> Vec<f32>,
) {
    assert_eq!(
        actual.shape(),
        expected.shape.as_slice(),
        "{kind} {name}: shape"
    );
    let a = host(actual);
    let e = expected.values();
    assert_eq!(a.len(), e.len(), "{kind} {name}: 要素数");
    assert_eq!(
        truth.shape, expected.shape,
        "{kind} {name}: f64 真値の shape"
    );
    assert_eq!(
        truth.values.len(),
        e.len(),
        "{kind} {name}: f64 真値の要素数"
    );
    let bad = mismatches(&a, &e);
    for &i in &bad {
        eprintln!(
            "[{kind} {name}][{i}] fandhe={} pytorch_f32={} pytorch_f64_truth={}",
            a[i], e[i], truth.values[i]
        );
    }
    if is_indeterminate(kind, name) {
        assert!(
            !bad.is_empty(),
            "{kind} {name}: INDETERMINATE に載っているが PyTorch と一致している（陳腐化した項目）"
        );
        let r = own();
        assert_eq!(r.len(), a.len(), "{kind} {name}: 自前参照の要素数");
        let own_bad = mismatches(&a, &r);
        assert!(
            own_bad.is_empty(),
            "{kind} {name}: 自前参照とも不一致 {own_bad:?}"
        );
    } else {
        assert!(
            bad.is_empty(),
            "{kind} {name}: PyTorch と {} 要素が統一複合判定を外れた（上の診断を確認。\
             PyTorch 値が f64 真値から外れている場合に限り判定不能として登録する）",
            bad.len()
        );
    }
}

/// fixture 側 `gen_reference.py` の vjp プログラムと同名・同式。
fn build_vjp_program<'t>(
    name: &str,
    x: &Var<'t>,
    c: &HashMap<String, Var<'t>>,
) -> Result<Var<'t>, AutodiffError> {
    match name {
        "vec_elementwise" => x.tanh().mul(x)?.add(&x.exp()),
        "scalar_sum" => x.sigmoid().mul(x)?.sum(None),
        "matmul_tanh" => Ok(x.matmul(&c["w"])?.tanh()),
        "transpose_out" => x.transpose(0, 1),
        "broadcast_out" => x.broadcast_to(&[2, 3]),
        "independent_rows" => Var::cat(&[x.mul(x)?, c["k"]], 0),
        "mean_dim" => x.exp().mean(Some(1)),
        "mlp" => {
            let h = x.matmul(&c["w1"])?.add(&c["b1"])?.relu();
            h.matmul(&c["w2"])
        }
        "scalar_in" => x.exp().mul(x),
        other => panic!("未知の vjp プログラム: {other}"),
    }
}

/// fixture 側 `gen_reference.py` の hvp プログラムと同名・同式。
fn build_hvp_program<'t>(
    name: &str,
    x: &Var<'t>,
    c: &HashMap<String, Var<'t>>,
) -> Result<Var<'t>, AutodiffError> {
    match name {
        "quadratic" => x.mul(&c["a"].matmul(x)?)?.sum(None),
        "linear" => c["k"].mul(x)?.sum(None),
        "tanh_sum" => x.tanh().mul(x)?.sum(None),
        "exp_mean" => x.exp().mul(x)?.mean(None),
        "mlp" => {
            let h = x.matmul(&c["w1"])?.add(&c["b1"])?.tanh();
            h.matmul(&c["w2"])?.sigmoid().sum(None)
        }
        "scalar_in" => x.exp().mul(x),
        "loss_1x1" => x.mul(x)?.sum(None)?.reshape(&[1, 1]),
        "cat_transpose" => {
            let z = Var::cat(&[x.sigmoid(), x.tanh()], 1)?.transpose(0, 1)?;
            z.mul(&z)?.sum(None)
        }
        "relu_cubic" => {
            let r = x.relu();
            r.mul(&r)?.mul(x)?.sum(None)
        }
        other => panic!("未知の hvp プログラム: {other}"),
    }
}

/// fixture 側 `gen_reference.py` の vmap プログラム（スライス 1 枚に適用する本体）と同名・同式。
fn build_vmap_program<'t>(
    name: &str,
    s: &Var<'t>,
    c: &HashMap<String, Var<'t>>,
) -> Result<Var<'t>, AutodiffError> {
    match name {
        "elementwise" => s.tanh().mul(s),
        "in_dim1" => s.exp().mul(s)?.add(s),
        "matmul_closure" => Ok(s.matmul(&c["w"])?.tanh()),
        "transpose" => s.transpose(0, 1),
        "to_scalar" => s.sum(None),
        "expand" => s.broadcast_to(&[2, 3]),
        "cat" => Var::cat(&[*s, s.mul(s)?], 0),
        other => panic!("未知の vmap プログラム: {other}"),
    }
}

// ---- 自前参照（判定不能ケースのゲート先。F4 で fixture と独立に検証する）----------

/// `Σ_i u[i] · J[i, j]`（`J = jacobian(y, x)` の転置積）。
fn own_vjp(case: &VjpCase) -> Vec<f32> {
    let tape = new_tape();
    let x = tape.var(&case.x.tensor());
    let c = consts_on(&tape, &case.consts);
    let y = build_vjp_program(&case.name, &x, &c).expect("program");
    let jac = host(&jacobian(&tape, &y, &x).expect("jacobian"));
    let u = case.u.values();
    let n: usize = case.x.shape.iter().product();
    assert_eq!(jac.len(), u.len() * n);
    (0..n)
        .map(|j| {
            (0..u.len())
                .map(|i| u[i] as f64 * jac[i * n + j] as f64)
                .sum::<f64>() as f32
        })
        .collect()
}

/// `Σ_j H[i, j] · v[j]`（`H = hessian(loss, x)`）。
fn own_hvp(case: &HvpCase) -> Vec<f32> {
    let tape = new_tape();
    let child = new_tape();
    let x = tape.var(&case.x.tensor());
    let c = consts_on(&tape, &case.consts);
    let loss = build_hvp_program(&case.name, &x, &c).expect("program");
    let h = host(&hessian(&tape, &loss, &x, &child).expect("hessian"));
    let v = case.v.values();
    let n = v.len();
    assert_eq!(h.len(), n * n);
    (0..n)
        .map(|i| {
            (0..n)
                .map(|j| h[i * n + j] as f64 * v[j] as f64)
                .sum::<f64>() as f32
        })
        .collect()
}

/// バッチなしで `in_dim` 軸のスライスごとに別テープで実行し、先頭軸へ積んだ結果。
fn own_vmap(case: &VmapCase) -> Vec<f32> {
    let shape = &case.x.shape;
    let d = case.in_dim;
    let data = case.x.values();
    let outer: usize = shape[..d].iter().product();
    let n = shape[d];
    let inner: usize = shape[d + 1..].iter().product();
    let mut slice_shape = shape.clone();
    slice_shape.remove(d);
    let mut result = Vec::new();
    for b in 0..n {
        let mut sl = Vec::with_capacity(outer * inner);
        for o in 0..outer {
            let base = (o * n + b) * inner;
            sl.extend_from_slice(&data[base..base + inner]);
        }
        let tape = new_tape();
        let s = tape.var(&Tensor::new(sl, &slice_shape).expect("slice"));
        let c = consts_on(&tape, &case.consts);
        let y = build_vmap_program(&case.name, &s, &c).expect("program");
        result.extend(host(&y.to_tensor()));
    }
    result
}

// =====================================================================
// F0: fixture の健全性
// =====================================================================

#[test]
fn f0_fixture_is_sane_and_indeterminate_entries_exist() {
    let fx = load_fixture();
    assert!(
        fx.torch_version.starts_with("2.14.0"),
        "{}",
        fx.torch_version
    );
    assert!(!fx.vjp_cases.is_empty());
    assert!(!fx.hvp_cases.is_empty());
    assert!(!fx.vmap_cases.is_empty());
    for &(kind, name, reason) in INDETERMINATE {
        let exists = match kind {
            "vjp" => fx.vjp_cases.iter().any(|c| c.name == name),
            "hvp" => fx.hvp_cases.iter().any(|c| c.name == name),
            "vmap" => fx.vmap_cases.iter().any(|c| c.name == name),
            _ => false,
        };
        assert!(
            exists,
            "INDETERMINATE が fixture に無い項目を指す: {kind} {name}"
        );
        assert!(!reason.is_empty(), "{kind} {name}: 理由が空");
    }
}

// =====================================================================
// F1: vjp × PyTorch fixture
// =====================================================================

#[test]
fn f1_vjp_matches_pytorch_fixture() {
    let fx = load_fixture();
    for case in &fx.vjp_cases {
        let tape = new_tape();
        let x = tape.var(&case.x.tensor());
        let c = consts_on(&tape, &case.consts);
        let y = build_vjp_program(&case.name, &x, &c).expect("program");
        assert_close_packed(
            &y.to_tensor(),
            &case.out,
            &format!("vjp {} forward", case.name),
        );
        let g =
            vjp(&tape, &y, &x, &case.u.tensor()).unwrap_or_else(|e| panic!("{}: {e:?}", case.name));
        gate(
            "vjp",
            &case.name,
            &g,
            &case.expected,
            &case.expected_f64,
            || own_vjp(case),
        );
    }
}

// =====================================================================
// F2: hvp × PyTorch fixture
// =====================================================================

#[test]
fn f2_hvp_matches_pytorch_fixture() {
    let fx = load_fixture();
    for case in &fx.hvp_cases {
        let tape = new_tape();
        let child = new_tape();
        let x = tape.var(&case.x.tensor());
        let c = consts_on(&tape, &case.consts);
        let loss = build_hvp_program(&case.name, &x, &c).expect("program");
        assert_close_packed(
            &loss.to_tensor(),
            &case.out,
            &format!("hvp {} forward", case.name),
        );
        let hv = hvp(&tape, &loss, &x, &case.v.tensor(), &child)
            .unwrap_or_else(|e| panic!("{}: {e:?}", case.name));
        gate(
            "hvp",
            &case.name,
            &hv,
            &case.expected,
            &case.expected_f64,
            || own_hvp(case),
        );
    }
}

// =====================================================================
// F3: vmap × PyTorch fixture
// =====================================================================

#[test]
fn f3_vmap_matches_pytorch_fixture() {
    let fx = load_fixture();
    for case in &fx.vmap_cases {
        let tape = new_tape();
        let x = tape.var(&case.x.tensor());
        let c = consts_on(&tape, &case.consts);
        let y = vmap(&tape, &x, case.in_dim, |s| {
            build_vmap_program(&case.name, s, &c)
        })
        .unwrap_or_else(|e| panic!("{}: {e:?}", case.name));
        gate(
            "vmap",
            &case.name,
            &y.to_tensor(),
            &case.expected,
            &case.expected_f64,
            || own_vmap(case),
        );
    }
}

// =====================================================================
// F4: 自前参照が fixture と独立に成立する（判定不能の付け替え先の健全性）
// =====================================================================

#[test]
fn f4_own_references_agree_with_the_transforms() {
    let fx = load_fixture();
    for case in &fx.vjp_cases {
        let tape = new_tape();
        let x = tape.var(&case.x.tensor());
        let c = consts_on(&tape, &case.consts);
        let y = build_vjp_program(&case.name, &x, &c).unwrap();
        let g = host(&vjp(&tape, &y, &x, &case.u.tensor()).unwrap());
        let r = own_vjp(case);
        assert!(mismatches(&g, &r).is_empty(), "vjp {}", case.name);
    }
    for case in &fx.hvp_cases {
        let tape = new_tape();
        let child = new_tape();
        let x = tape.var(&case.x.tensor());
        let c = consts_on(&tape, &case.consts);
        let loss = build_hvp_program(&case.name, &x, &c).unwrap();
        let g = host(&hvp(&tape, &loss, &x, &case.v.tensor(), &child).unwrap());
        let r = own_hvp(case);
        assert!(mismatches(&g, &r).is_empty(), "hvp {}", case.name);
    }
    for case in &fx.vmap_cases {
        let tape = new_tape();
        let x = tape.var(&case.x.tensor());
        let c = consts_on(&tape, &case.consts);
        let y = vmap(&tape, &x, case.in_dim, |s| {
            build_vmap_program(&case.name, s, &c)
        })
        .unwrap();
        let r = own_vmap(case);
        assert!(
            mismatches(&host(&y.to_tensor()), &r).is_empty(),
            "vmap {}",
            case.name
        );
    }
}
