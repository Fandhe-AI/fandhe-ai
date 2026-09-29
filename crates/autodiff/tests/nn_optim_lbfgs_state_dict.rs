//! `Lbfgs` の状態保存・復元 API（`state_dict`／`load_state_dict`／
//! `history_len`。イシュー #2366）の統合テスト。
//!
//! 受け入れ基準の対応: (1) 復元後の次 step が元インスタンスと bit 一致
//! （履歴が `history_size` 未満・到達の両方・line search 2 方式）、
//! (2) 件数・長さ・shape・非有限値・キー欠落／余剰の fail-closed 拒否
//! （拒否後に `self` の状態が変化しないことを含む）。キー配置の正は
//! `crates/autodiff/src/nn/optim/lbfgs.rs` 冒頭 doc「状態の保存・復元」節。

use std::collections::{BTreeSet, HashMap};

use fandhe_ai_autodiff::AutodiffError;
use fandhe_ai_autodiff::nn::optim::{Lbfgs, LbfgsConfig, LbfgsLineSearch};
use fandhe_ai_tensor_core::Tensor;

type State = HashMap<String, Tensor<f32>>;

fn t(data: Vec<f32>, shape: &[usize]) -> Tensor<f32> {
    Tensor::new(data, shape).expect("test fixture: shape とデータ長は事前に一致させている")
}

fn encode_u16x4(bits: u64) -> Tensor<f32> {
    let words: Vec<f32> = (0..4)
        .map(|i| (((bits >> (16 * i)) & 0xFFFF) as u32) as f32)
        .collect();
    t(words, &[4])
}

fn tensor_bits(tensor: &Tensor<f32>) -> (Vec<usize>, Vec<u32>) {
    let data = tensor
        .contiguous()
        .as_slice()
        .expect("test fixture: contiguous のはず")
        .iter()
        .map(|v| v.to_bits())
        .collect();
    (tensor.shape().to_vec(), data)
}

fn assert_state_dicts_bit_equal(a: &State, b: &State) {
    let ka: BTreeSet<&String> = a.keys().collect();
    let kb: BTreeSet<&String> = b.keys().collect();
    assert_eq!(ka, kb, "state_dict のキー集合が一致しない");
    for k in ka {
        assert_eq!(
            tensor_bits(&a[k]),
            tensor_bits(&b[k]),
            "キー `{k}` が不一致"
        );
    }
}

fn slot_shapes() -> Vec<Vec<usize>> {
    vec![vec![2, 2], vec![3]]
}

fn init_params() -> Vec<Tensor<f32>> {
    vec![
        t(vec![1.0, -2.0, 0.5, 3.0], &[2, 2]),
        t(vec![-1.5, 2.0, 0.25], &[3]),
    ]
}

/// 対角係数の異なる凸二次形式 + 弱い 4 次項（曲率は常に正）。
fn objective(params: &[Tensor<f32>]) -> (f32, Vec<Tensor<f32>>) {
    let a = [[0.5f32, 1.0, 1.5, 2.0], [0.7, 1.2, 0.9, 0.0]];
    let c = [[0.3f32, -0.2, 0.1, 0.4], [-0.5, 0.6, 0.2, 0.0]];
    let mut loss = 0.0f32;
    let mut grads = Vec::new();
    for (pi, p) in params.iter().enumerate() {
        let x = p.contiguous().as_slice().unwrap().to_vec();
        let mut g = Vec::with_capacity(x.len());
        for (i, &xi) in x.iter().enumerate() {
            let d = xi - c[pi][i];
            loss += a[pi][i] * d * d + 0.025 * xi * xi * xi * xi;
            g.push(2.0 * a[pi][i] * d + 0.1 * xi * xi * xi);
        }
        grads.push(t(g, p.shape()));
    }
    (loss, grads)
}

fn config(history_size: usize, line_search: LbfgsLineSearch) -> LbfgsConfig {
    LbfgsConfig {
        lr: 0.1,
        max_iter: 4,
        max_eval: Some(40),
        tolerance_grad: 1e-12,
        tolerance_change: 1e-15,
        history_size,
        line_search,
        ..LbfgsConfig::default()
    }
}

/// `steps` 回の outer step を実行した `Lbfgs` と最終パラメータを返す。
fn trained(cfg: LbfgsConfig, steps: usize) -> (Lbfgs, Vec<Tensor<f32>>) {
    let mut opt = Lbfgs::new(cfg).unwrap();
    let mut params = init_params();
    for _ in 0..steps {
        params = opt.step_closure(&params, objective).unwrap();
    }
    (opt, params)
}

fn roundtrip_matches(cfg: LbfgsConfig, steps: usize) -> usize {
    let (mut orig, params) = trained(cfg, steps);
    let sd = orig.state_dict().unwrap();
    let n = orig.history_len();
    let mut restored = Lbfgs::new(cfg).unwrap();
    restored.load_state_dict(sd, &slot_shapes(), n).unwrap();

    let out_a = orig.step_closure(&params, objective).unwrap();
    let out_b = restored.step_closure(&params, objective).unwrap();
    for (a, b) in out_a.iter().zip(out_b.iter()) {
        assert_eq!(
            tensor_bits(a),
            tensor_bits(b),
            "次 step の出力が bit 不一致"
        );
    }
    assert_state_dicts_bit_equal(&orig.state_dict().unwrap(), &restored.state_dict().unwrap());
    assert_eq!(orig.n_iter(), restored.n_iter());
    assert_eq!(orig.func_evals(), restored.func_evals());
    assert_eq!(
        orig.last_loss().map(f32::to_bits),
        restored.last_loss().map(f32::to_bits)
    );
    n
}

#[test]
fn roundtrip_bit_exact_history_below_capacity() {
    for ls in [LbfgsLineSearch::None, LbfgsLineSearch::StrongWolfe] {
        let n = roundtrip_matches(config(100, ls), 1);
        assert!(n > 0 && n < 100, "履歴が 0 < n < history_size でない: {n}");
    }
}

#[test]
fn roundtrip_bit_exact_history_at_capacity() {
    for ls in [LbfgsLineSearch::None, LbfgsLineSearch::StrongWolfe] {
        let n = roundtrip_matches(config(2, ls), 2);
        assert_eq!(n, 2, "履歴が history_size に到達していない");
    }
}

#[test]
fn unrun_state_roundtrips_and_adopts_new_shapes() {
    let cfg = config(100, LbfgsLineSearch::None);
    let orig = Lbfgs::new(cfg).unwrap();
    let sd = orig.state_dict().unwrap();
    let keys: BTreeSet<&str> = sd.keys().map(String::as_str).collect();
    assert_eq!(
        keys,
        BTreeSet::from(["n_iter.u64_u16x4", "func_evals.u64_u16x4", "t", "h_diag"])
    );
    let mut restored = Lbfgs::new(cfg).unwrap();
    restored.load_state_dict(sd, &[], 0).unwrap();
    // 元と同じく、次の step で params の shape を採用する。
    let params = vec![t(vec![1.0, 2.0], &[2])];
    let f = |p: &[Tensor<f32>]| {
        let x = p[0].contiguous().as_slice().unwrap().to_vec();
        let loss: f32 = x.iter().map(|v| v * v).sum();
        (loss, vec![t(x.iter().map(|v| 2.0 * v).collect(), &[2])])
    };
    let mut fresh = Lbfgs::new(cfg).unwrap();
    let a = fresh.step_closure(&params, f).unwrap();
    let b = restored.step_closure(&params, f).unwrap();
    assert_eq!(tensor_bits(&a[0]), tensor_bits(&b[0]));
}

#[test]
fn early_return_state_roundtrips_without_vectors() {
    // 初回評価で勾配収束判定を満たし、n_iter == 0 のまま終わる状態。
    let cfg = LbfgsConfig {
        tolerance_grad: 1e9,
        ..config(100, LbfgsLineSearch::None)
    };
    let mut orig = Lbfgs::new(cfg).unwrap();
    let params = init_params();
    orig.step_closure(&params, objective).unwrap();
    assert_eq!(orig.n_iter(), 0);
    assert!(orig.func_evals() >= 1);
    let sd = orig.state_dict().unwrap();
    let keys: BTreeSet<&str> = sd.keys().map(String::as_str).collect();
    assert_eq!(
        keys,
        BTreeSet::from([
            "n_iter.u64_u16x4",
            "func_evals.u64_u16x4",
            "t",
            "h_diag",
            "last_loss"
        ])
    );
    let mut restored = Lbfgs::new(cfg).unwrap();
    restored.load_state_dict(sd, &slot_shapes(), 0).unwrap();
    let a = orig.step_closure(&params, objective).unwrap();
    let b = restored.step_closure(&params, objective).unwrap();
    assert_eq!(tensor_bits(&a[0]), tensor_bits(&b[0]));
    assert_state_dicts_bit_equal(&orig.state_dict().unwrap(), &restored.state_dict().unwrap());
}

// ---- 拒否系 ----

const HIST: usize = 100;

fn valid() -> (State, usize) {
    let (opt, _) = trained(config(HIST, LbfgsLineSearch::None), 2);
    let n = opt.history_len();
    assert!(n >= 2, "拒否系の前提: 履歴 2 件以上（n={n}）");
    (opt.state_dict().unwrap(), n)
}

fn n_total() -> usize {
    7
}

/// 変異済み state を、学習済みターゲットへ load して `Err` になり、かつ
/// ターゲットの state_dict が load 前と bit 一致であることを確認する。
fn assert_rejected(state: State, slots: &[Vec<usize>], len: usize) -> AutodiffError {
    let (mut target, _) = trained(config(HIST, LbfgsLineSearch::None), 2);
    let before = target.state_dict().unwrap();
    let err = target
        .load_state_dict(state, slots, len)
        .expect_err("拒否されるべき state が受理された");
    assert!(
        matches!(
            err,
            AutodiffError::InvalidArgument(_) | AutodiffError::Shape(_)
        ),
        "想定外のエラー種別: {err:?}"
    );
    assert_state_dicts_bit_equal(&before, &target.state_dict().unwrap());
    err
}

fn reject_with(mutate: impl FnOnce(&mut State, usize)) {
    let (mut sd, n) = valid();
    mutate(&mut sd, n);
    assert_rejected(sd, &slot_shapes(), n);
}

#[test]
fn valid_state_is_accepted() {
    let (sd, n) = valid();
    let (mut target, _) = trained(config(HIST, LbfgsLineSearch::None), 1);
    target.load_state_dict(sd, &slot_shapes(), n).unwrap();
}

#[test]
fn rejects_history_len_mismatch() {
    let (sd, n) = valid();
    assert_rejected(sd.clone(), &slot_shapes(), n + 1);
    assert_rejected(sd.clone(), &slot_shapes(), n - 1);
    assert_rejected(sd, &slot_shapes(), usize::MAX);
}

#[test]
fn rejects_missing_keys() {
    for key in [
        "n_iter.u64_u16x4",
        "func_evals.u64_u16x4",
        "t",
        "h_diag",
        "last_loss",
        "d",
        "prev_flat_grad",
        "history.rho",
    ] {
        reject_with(|sd, _| {
            sd.remove(key);
        });
    }
    reject_with(|sd, n| {
        sd.remove(&format!("history.{}.y", n - 1));
    });
    reject_with(|sd, n| {
        sd.remove(&format!("history.{}.s", n - 1));
        sd.remove(&format!("history.{}.y", n - 1));
    });
}

#[test]
fn rejects_extra_and_noncanonical_keys() {
    reject_with(|sd, _| {
        sd.insert("unknown".into(), t(vec![0.0], &[1]));
    });
    reject_with(|sd, _| {
        sd.insert("history.01.s".into(), t(vec![0.0; 7], &[7]));
    });
    reject_with(|sd, _| {
        sd.insert("history.+1.s".into(), t(vec![0.0; 7], &[7]));
    });
    // 飛び番（0 と 2 以降のみ）。
    reject_with(|sd, n| {
        let s = sd.remove("history.1.s").unwrap();
        let y = sd.remove("history.1.y").unwrap();
        sd.insert(format!("history.{n}.s"), s);
        sd.insert(format!("history.{n}.y"), y);
    });
}

#[test]
fn rejects_wrong_lengths_and_shapes() {
    reject_with(|sd, _| {
        sd.insert("d".into(), t(vec![0.1; 6], &[6]));
    });
    reject_with(|sd, _| {
        sd.insert("d".into(), t(vec![0.1; 8], &[8]));
    });
    reject_with(|sd, _| {
        sd.insert("d".into(), t(vec![0.1; 7], &[1, 7]));
    });
    reject_with(|sd, _| {
        sd.insert("history.0.s".into(), t(vec![0.1; 6], &[6]));
    });
    reject_with(|sd, n| {
        sd.insert("history.rho".into(), t(vec![0.1; n - 1], &[n - 1]));
    });
    reject_with(|sd, _| {
        sd.insert("t".into(), t(vec![0.1; 2], &[2]));
    });
}

#[test]
fn rejects_non_finite_values() {
    for bad in [f32::NAN, f32::INFINITY] {
        for key in ["t", "h_diag", "last_loss"] {
            reject_with(|sd, _| {
                sd.insert(key.into(), t(vec![bad], &[1]));
            });
        }
        for key in ["d", "prev_flat_grad", "history.0.s", "history.1.y"] {
            reject_with(|sd, _| {
                let mut v = vec![0.5; n_total()];
                v[3] = bad;
                sd.insert(key.into(), t(v, &[n_total()]));
            });
        }
        reject_with(|sd, n| {
            let mut v = vec![0.5; n];
            v[0] = bad;
            sd.insert("history.rho".into(), t(v, &[n]));
        });
    }
}

#[test]
fn rejects_unreachable_states() {
    // n_iter に対して履歴が多すぎる。
    reject_with(|sd, n| {
        sd.insert("n_iter.u64_u16x4".into(), encode_u16x4(n as u64));
    });
    // n_iter >= 1 なのに last_loss がない（func_evals == 0）。
    reject_with(|sd, _| {
        sd.insert("func_evals.u64_u16x4".into(), encode_u16x4(0));
        sd.remove("last_loss");
    });
    // n_iter == 0 なのに d 等・履歴がある。
    reject_with(|sd, _| {
        sd.insert("n_iter.u64_u16x4".into(), encode_u16x4(0));
    });
    // func_evals == 0 なのに last_loss がある。
    reject_with(|sd, _| {
        sd.insert("n_iter.u64_u16x4".into(), encode_u16x4(0));
        sd.insert("func_evals.u64_u16x4".into(), encode_u16x4(0));
        // last_loss は残したまま（func_evals == 0 との矛盾）。
        for k in ["d", "prev_flat_grad", "history.rho"] {
            sd.remove(k);
        }
        for i in 0..2 {
            sd.remove(&format!("history.{i}.s"));
            sd.remove(&format!("history.{i}.y"));
        }
    });
}

#[test]
fn rejects_history_over_config_history_size() {
    let (sd, n) = valid();
    let (mut target, _) = trained(config(1, LbfgsLineSearch::None), 1);
    let before = target.state_dict().unwrap();
    assert!(target.load_state_dict(sd, &slot_shapes(), n).is_err());
    assert_state_dicts_bit_equal(&before, &target.state_dict().unwrap());
}

#[test]
fn rejects_bad_slot_shapes() {
    let (sd, n) = valid();
    assert_rejected(sd.clone(), &[], n);
    assert_rejected(sd.clone(), &[vec![usize::MAX, 2]], n);
    assert_rejected(sd, &[vec![3, 3]], n);
}

#[test]
fn rejects_bad_u16x4_encoding() {
    reject_with(|sd, _| {
        sd.insert("n_iter.u64_u16x4".into(), t(vec![1.0, 0.0, 0.0], &[3]));
    });
    reject_with(|sd, _| {
        sd.insert("n_iter.u64_u16x4".into(), t(vec![1.5, 0.0, 0.0, 0.0], &[4]));
    });
    reject_with(|sd, _| {
        sd.insert(
            "func_evals.u64_u16x4".into(),
            t(vec![70000.0, 0.0, 0.0, 0.0], &[4]),
        );
    });
}

/// `n_iter > func_evals` は `step` から到達不能なため拒否する。
#[test]
fn rejects_n_iter_exceeding_func_evals() {
    reject_with(|sd, _| {
        sd.insert("n_iter.u64_u16x4".into(), encode_u16x4(u64::MAX));
    });
    reject_with(|sd, n| {
        sd.insert("n_iter.u64_u16x4".into(), encode_u16x4(n as u64 + 2));
        sd.insert("func_evals.u64_u16x4".into(), encode_u16x4(n as u64 + 1));
    });
}

/// `history.rho` は正の有限値のみ到達可能（ys > 1e-10 のペアのみ生成）。
/// `h_diag` は負値のみ拒否する（0 は下記の受理テストを参照）。
#[test]
fn rejects_non_positive_h_diag_and_rho() {
    reject_with(|sd, _| {
        sd.insert("h_diag".into(), t(vec![-1.0], &[1]));
    });
    for v in [0.0f32, -1.0] {
        reject_with(|sd, n| {
            let mut rho = vec![1.0f32; n];
            rho[n - 1] = v;
            sd.insert("history.rho".into(), t(rho, &[n]));
        });
    }
}

/// `h_diag = ys / yy` は f32 で 0 にアンダーフローしうる到達可能状態のため、
/// 保存した `h_diag = 0` は復元でき、再保存が bit 一致する。
#[test]
fn zero_h_diag_is_restorable_and_round_trips() {
    let (mut sd, n) = valid();
    sd.insert("h_diag".into(), t(vec![0.0], &[1]));
    let (mut target, _params) = trained(config(HIST, LbfgsLineSearch::None), 1);
    target
        .load_state_dict(sd.clone(), &slot_shapes(), n)
        .unwrap();
    let saved = target.state_dict().unwrap();
    assert_state_dicts_bit_equal(&saved, &sd);
}

/// カウンタに上限定数はなく、`u64::MAX` の復元自体は受理し、
/// 次の `step` が型付きエラーで失敗する（panic／巻き戻りなし）。
#[test]
fn counter_at_u64_max_is_restorable_and_next_step_errors() {
    let (mut sd, n) = valid();
    sd.insert("n_iter.u64_u16x4".into(), encode_u16x4(u64::MAX));
    sd.insert("func_evals.u64_u16x4".into(), encode_u16x4(u64::MAX));
    let (mut target, params) = trained(config(HIST, LbfgsLineSearch::None), 1);
    target.load_state_dict(sd, &slot_shapes(), n).unwrap();
    let err = target
        .step_closure(&params, objective)
        .expect_err("オーバーフローは型付きエラー");
    assert!(matches!(err, AutodiffError::InvalidArgument(_)));
    assert_eq!(target.n_iter(), u64::MAX);
    assert_eq!(target.func_evals(), u64::MAX);
}
