//! 二階微分（`create_graph`・孫テープ再生）の CUDA 実機 parity テストの雛形
//! （イシュー #3776・親 #3719。測り方は `docs/autodiff-higher-order-grad-decision.md`
//! §48〈#3739〉）。Phase 5 の各族の実装 issue は [`CASES`] に 1 行足し、
//! Phase 10 が GB10（`<cuda-node>`）で測る。Metal 版は #3777 が同じファイルへ足す。
//!
//! 置き場が facade の理由: `crates/autodiff` は GPU バックエンドへ届かず、facade は
//! 全バックエンドを合成する公開経路で、既存の `create_graph_facade.rs`
//! （#3168・#3169 で実測済み）と同じ実行経路になるため（§48.4）。既存テスト・
//! そのログは変えない。
//!
//! 測る量（§48.4）: Q1 `first_order()` が同デバイスの素の `backward` と bit 一致、
//! Q2 子テープ上の `cg.grad`、Q3 二階 `H v`（ゼロでないことも assert）、
//! Q4 孫テープ再生（任意）は GPU と CPU の間を REQ-2 統一複合判定で比べる。
//! Q5 拒否の一致は GPU でも `Err(Backward)` かつ子テープが空であること。
//!
//! 判定: 比較は `fandhe_ai_backend_cpu::parity::compare`。tolerance 定数・baseline の
//! 値は変えない。`Baseline` 方式は `tests/common/create_graph_gpu_parity_baseline.rs`
//! の空の置き場を引き、行が無ければ undetermined で止まる。
//!
//! 実行: `#[ignore]` の GPU テストは CI で走らず、`cargo test -p fandhe-ai --test
//! create_graph_gpu_parity --no-run` でビルドだけ通る。実機手順は
//! `docs/real-hardware-verification-env.md` 4.9 節。RULE.txt の雛形は
//! `docs/perf/logs/create-graph-gpu-parity-template/RULE.txt`。
//! 登録表自体の健全性は CPU 同士の非 `#[ignore]` 自己検査で CI が守る。

#[path = "common/create_graph_gpu_parity_baseline.rs"]
mod baseline;

use fandhe_ai::{AutodiffError, Device, Tape, Tensor};
use fandhe_ai_backend_cpu::parity::compare;

/// 経路区分（§48.4）。`H` の pass は「経路の parity」でカーネル同士ではない。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[allow(dead_code)] // 各族の実装 issue が使う語彙。雛形では K だけ使う。
enum Route {
    /// デバイスカーネル経路。
    K,
    /// デバイスからホストへの実体化を含むフォールバック経路。
    H,
    /// opt-in ゲートの後ろ（既定 OFF なので H として測る）。
    G,
}

/// 判定方式。既定は `Strict`（fail_count == 0）。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[allow(dead_code)] // Baseline は所有者承認後に使う語彙。雛形は空の置き場だけ。
enum Judge {
    Strict,
    /// 結合順序の異なる経路を含む族。baseline に行が無ければ undetermined。
    Baseline,
}

/// 親・子テープ（同一デバイス）。Q4 の孫テープ再生が要る族は `grand` を使う。
struct Tapes {
    parent: Tape,
    child: Tape,
    #[allow(dead_code)] // Q4 を測る族の実装 issue が使う。
    grand: Tape,
}

/// 1 ケースが返す測定量。
struct Outputs {
    /// `cg.first_order()` の勾配（Q1）。
    first_order: Vec<f32>,
    /// 同デバイスの素の `Tape::backward` の勾配（Q1 の bit 比較相手）。
    plain: Vec<f32>,
    /// `cg.grad` の値（Q2）。
    grad: Vec<f32>,
    /// 固定方向 `probe_dir` の二階 `H v`（Q3。ゼロでないことも assert）。
    hv: Vec<f32>,
    /// 孫テープ再生の出力（Q4。測らないなら `None`）。
    replay: Option<Vec<f32>>,
}

/// 登録表の 1 行。**Op を足すときはここに 1 行足す**（テスト関数は族単位で別に置く）。
struct Case {
    /// ケース名。baseline の `case` と逐語一致させる。
    name: &'static str,
    /// 族名（`create_graph_<族>_cuda_matches_cpu` の `<族>`。§48.4 の命名表）。
    family: &'static str,
    route: Route,
    judge: Judge,
    /// 決定的な入力でテープを走らせ測定量を返す。形状・入力はこの関数内で固定する
    /// （S1: CI の小形状 + S2: 2 のべき乗でない中形状、S4: 決定的、S5: しきい値回避）。
    run: fn(&Tapes) -> Outputs,
    /// Q5。拒否入力で `Err(Backward)` かつ子が空（`leaf_count() == 0`）なら `true`。不要なら `None`。
    reject: Option<fn(&Tapes) -> bool>,
}

/// 登録表。雛形の自己検査用に、実装済みの `Mul`・`Sum` だけの 1 行を置く。
const CASES: &[Case] = &[Case {
    name: "scaffold_cubic",
    family: "scaffold",
    route: Route::K,
    judge: Judge::Strict,
    run: run_scaffold_cubic,
    reject: None,
}];

fn t(data: Vec<f32>, shape: &[usize]) -> Tensor<f32> {
    Tensor::<f32>::new(data, shape).expect("test fixture: tensor")
}

fn vals(t: &Tensor<f32>) -> Vec<f32> {
    t.host_slice().into_owned()
}

/// `loss = sum(x^3)`。二階が非自明（Hessian 対角 6x）な最小例。
fn run_scaffold_cubic(tp: &Tapes) -> Outputs {
    let x0: Vec<f32> = (0..17).map(|i| 0.25 + 0.125 * i as f32).collect();
    let dir: Vec<f32> = (0..17).map(|i| 1.0 - 0.0625 * i as f32).collect();

    let x = tp.parent.var(&t(x0, &[17]));
    let loss = x.mul(&x).unwrap().mul(&x).unwrap().sum(None).unwrap();
    let plain_grads = tp.parent.backward(&loss).expect("backward");
    let plain = vals(plain_grads.get(&x).unwrap().expect("grad"));

    let cg = tp
        .parent
        .backward_create_graph(&loss, &tp.child)
        .expect("create_graph");
    let first_order = vals(cg.first_order().get(&x).unwrap().expect("first_order"));
    let g = cg.grad(&x).unwrap().expect("grad");
    let grad = vals(&g.value());
    let v = tp.child.var_no_grad(&t(dir, &[17]));
    let hv_loss = g.mul(&v).unwrap().sum(None).unwrap();
    let grads2 = tp.child.backward(&hv_loss).expect("child backward");
    let xc = cg.child_var(&x).unwrap().expect("mirror");
    let hv = vals(grads2.get(&xc).unwrap().expect("second order"));
    Outputs {
        first_order,
        plain,
        grad,
        hv,
        replay: None,
    }
}

fn tapes_for(device: Option<Device>) -> Tapes {
    let mk = || match device {
        None => fandhe_ai::tape(),
        Some(d) => fandhe_ai::tape_for(d).expect("実機が利用可能な前提のテスト"),
    };
    Tapes {
        parent: mk(),
        child: mk(),
        grand: mk(),
    }
}

/// REQ-2 統一複合判定（`Baseline` 方式は空の置き場を引く）。fail 時は RULE.txt から
/// 引用できるよう fail_count・max_abs_diff・max_rel_err を出す。
fn judge(case: &Case, quantity: &str, gpu: &[f32], cpu: &[f32]) -> Result<(), String> {
    let report = compare(gpu, cpu).map_err(|e| format!("{}:{quantity}: {e:?}", case.name))?;
    match case.judge {
        Judge::Strict if report.passes() => Ok(()),
        Judge::Strict => Err(format!(
            "{}:{quantity}: REQ-2 fail (route={:?}, fail_count={}/{}, max_abs_diff={:.6e}, \
             max_rel_err={:.6e})",
            case.name,
            case.route,
            report.fail_count,
            report.total,
            report.max_abs_diff,
            report.max_rel_err
        )),
        Judge::Baseline => baseline::check_no_regression(baseline::lookup(case.name)?, &report),
    }
}

/// 1 ケースを `device`（None は CPU）と CPU 参照で突き合わせる。
fn check_case(case: &Case, device: Option<Device>) -> Result<(), String> {
    let gpu = (case.run)(&tapes_for(device));
    let cpu = (case.run)(&tapes_for(None));
    // Q1: 同デバイスの素の backward と bit 一致（構造的な性質）。
    let bits = |v: &[f32]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
    if bits(&gpu.first_order) != bits(&gpu.plain) {
        return Err(format!(
            "{}:Q1: first_order が backward と bit 不一致",
            case.name
        ));
    }
    judge(case, "Q2 grad", &gpu.grad, &cpu.grad)?;
    // 線形な Op で二階の検査が自明に通るのを防ぐ。
    if gpu.hv.iter().all(|&v| v == 0.0) {
        return Err(format!("{}:Q3: H v が全ゼロ", case.name));
    }
    judge(case, "Q3 hv", &gpu.hv, &cpu.hv)?;
    match (&gpu.replay, &cpu.replay) {
        (Some(g), Some(c)) => judge(case, "Q4 replay", g, c)?,
        (None, None) => {}
        _ => return Err(format!("{}:Q4: 片側だけ replay を返した", case.name)),
    }
    if let Some(reject) = case.reject
        && !reject(&tapes_for(device))
    {
        return Err(format!("{}:Q5: 拒否が CPU と一致しない", case.name));
    }
    Ok(())
}

/// Q5 の共通判定。`r` が `Err(Backward)` かつ `child` が空なら `true`。
#[allow(dead_code)] // 各族の `reject` が使う。
fn is_clean_rejection<T>(r: Result<T, AutodiffError>, child: &Tape) -> bool {
    matches!(r, Err(AutodiffError::Backward(_))) && child.leaf_count() == 0
}

/// 族 `family` の全ケースを `device` で測る。panic メッセージは RULE.txt 引用用。
fn run_family(family: &str, device: Option<Device>) {
    let cases: Vec<&Case> = CASES.iter().filter(|c| c.family == family).collect();
    assert!(
        !cases.is_empty(),
        "undetermined: 族 `{family}` の登録が 0 件"
    );
    let failures: Vec<String> = cases
        .iter()
        .filter_map(|c| check_case(c, device).err())
        .collect();
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

/// 実装 issue ごとに 1 本、`#[ignore]` の CUDA テストを足すマクロ
/// （`--exact` で 1 本ずつ実行するため。名前は `create_graph_<族>_cuda_matches_cpu`）。
macro_rules! cuda_family_test {
    ($name:ident, $family:literal) => {
        #[test]
        #[ignore = "CUDA 実機（DGX Spark GB10）が必要"]
        fn $name() {
            run_family($family, Some(Device::Cuda(0)));
        }
    };
}

cuda_family_test!(create_graph_scaffold_cuda_matches_cpu, "scaffold");

// --- CI で走る自己検査（GPU 不要。登録表と置き場の健全性だけを見る） ---

#[test]
fn registry_names_are_unique() {
    for (i, a) in CASES.iter().enumerate() {
        assert!(
            CASES[i + 1..].iter().all(|b| b.name != a.name),
            "{}",
            a.name
        );
    }
}

#[test]
fn scaffold_case_passes_cpu_against_cpu() {
    run_family("scaffold", None);
}

#[test]
fn baseline_is_an_empty_placeholder_and_fails_closed() {
    assert!(
        baseline::BASELINES.is_empty(),
        "値は実機実測と所有者承認まで入れない"
    );
    assert!(baseline::lookup("scaffold_cubic").is_err());
}
