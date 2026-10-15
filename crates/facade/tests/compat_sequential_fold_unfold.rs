//! `compat::Sequential::add_unfold`／`add_fold`（イシュー #3040・親 #3018。公開形の正は
//! `docs/compat-sequential-3d-fold-lrn-layers-decision.md` §3）の facade 公開面を検証する統合テスト。
//! `compat_sequential_pool3d.rs`（#3028）と同型。
//!
//! - 無効引数（0 次元・`2*padding` オーバーフロー・`Fold` の `output_size` の 0）が `Err`（panic しない）。
//! - `predict`／`forward` が `Var::unfold`／`fold` の直接呼び出しと bit 完全一致。
//! - 入力勾配が直接グラフと bit 完全一致。学習 1 step が参照グラフと bit 一致し `compile`／`fit` が動く。
//! - 常駐経路は `Unsupported`（fail-closed。決定記録 §5）。`save_model` は後続 issue（`-c`）まで
//!   `UnsupportedModel`（`-c` が本テストを成功系へ書き換える）。ONNX export は `UnsupportedLayer`。
//!
//! ホストフォールバック経路のため実機（CUDA／Metal）非依存。実機 parity は未実測のまま（#3171・#3172）。
#![cfg(unix)]

use fandhe_ai::compat::{FitConfig, Loss, ModelIoError, Optimizer, Sequential, save_model};
use fandhe_ai::interop::onnx::{OnnxError, OnnxModel};
use fandhe_ai::optim::{Sgd, SgdConfig};
use fandhe_ai::{AutodiffError, BackendError, Tensor};

mod common;
use common::temp_dir::TempDirGuard;

const SEED_C: u64 = 0x3040_0001;
const SEED_L: u64 = 0x3040_0002;

fn tensor(data: Vec<f32>, shape: &[usize]) -> Tensor<f32> {
    Tensor::new(data, shape).expect("test fixture: shape とデータ長は事前に一致させている")
}

fn dense_vec(t: &Tensor<f32>) -> Vec<f32> {
    t.contiguous()
        .as_slice()
        .expect("contiguous() 直後は必ず as_slice() が Some を返す")
        .to_vec()
}

fn ramp(n: usize, scale: f32, offset: f32) -> Vec<f32> {
    (0..n).map(|i| (i as f32) * scale + offset).collect()
}

const K: [usize; 2] = [2, 2];
const S: [usize; 2] = [1, 2];
const P: [usize; 2] = [1, 0];
const D: [usize; 2] = [1, 1];

/// Unfold 入力 `[N=2, C=3, H=4, W=5]`。
fn image() -> Tensor<f32> {
    tensor(ramp(2 * 3 * 4 * 5, 0.013, -0.7), &[2, 3, 4, 5])
}

/// `unfold(K, S, P, D)` の出力形状 `[2, 12, L]` に対応する Fold 入力。
/// H=4: (4+2*1-1*(2-1)-1)/1+1 = 5、W=5: (5+0-1-1)/2+1 = 2 → L=10。
fn columns() -> Tensor<f32> {
    tensor(ramp(2 * 12 * 10, 0.011, -1.1), &[2, 12, 10])
}

fn direct_unfold(x: &Tensor<f32>) -> Tensor<f32> {
    let tape = fandhe_ai::tape();
    let xv = tape.var(x);
    xv.unfold(K, S, P, D).unwrap().to_tensor()
}

// ---------------------------------------------------------------- 無効引数

#[test]
fn add_fold_unfold_reject_invalid_arguments() {
    let one = [1usize; 2];
    let z = [0usize; 2];
    assert!(Sequential::new().add_unfold([0, 2], one, z, one).is_err());
    assert!(Sequential::new().add_unfold(K, [1, 0], z, one).is_err());
    assert!(Sequential::new().add_unfold(K, one, z, [0, 1]).is_err());
    assert!(
        Sequential::new()
            .add_unfold(K, one, [usize::MAX, 0], one)
            .is_err()
    );
    assert!(
        Sequential::new()
            .add_fold([4, 4], [0, 2], one, z, one)
            .is_err()
    );
    assert!(
        Sequential::new()
            .add_fold([4, 4], K, [0, 1], z, one)
            .is_err()
    );
    assert!(
        Sequential::new()
            .add_fold([4, 4], K, one, z, [1, 0])
            .is_err()
    );
    assert!(
        Sequential::new()
            .add_fold([4, 4], K, one, [usize::MAX, 0], one)
            .is_err()
    );
    assert!(Sequential::new().add_fold([0, 4], K, one, z, one).is_err());
    assert!(Sequential::new().add_fold([4, 0], K, one, z, one).is_err());
}

// ---------------------------------------------------------------- forward / predict

#[test]
fn unfold_predict_and_forward_match_direct_call_bit_exact() {
    let model = Sequential::new().add_unfold(K, S, P, D).unwrap();
    let x = image();
    let predicted = model.predict(&x).unwrap();
    let tape = fandhe_ai::tape();
    let xv = tape.var(&x);
    let forwarded = model.forward(&tape, &xv).unwrap().to_tensor();
    let direct = direct_unfold(&x);
    assert_eq!(predicted.shape(), [2, 12, 10]);
    assert_eq!(dense_vec(&predicted), dense_vec(&direct));
    assert_eq!(dense_vec(&forwarded), dense_vec(&direct));
}

#[test]
fn fold_predict_and_forward_match_direct_call_bit_exact() {
    let model = Sequential::new().add_fold([4, 5], K, S, P, D).unwrap();
    let x = columns();
    let predicted = model.predict(&x).unwrap();
    let tape = fandhe_ai::tape();
    let xv = tape.var(&x);
    let forwarded = model.forward(&tape, &xv).unwrap().to_tensor();
    let direct = xv.fold([4, 5], K, S, P, D).unwrap().to_tensor();
    assert_eq!(predicted.shape(), [2, 3, 4, 5]);
    assert_eq!(dense_vec(&predicted), dense_vec(&direct));
    assert_eq!(dense_vec(&forwarded), dense_vec(&direct));
}

#[test]
fn unfold_then_fold_roundtrip() {
    // 非重複窓（stride = kernel・padding なし）では恒等になる。
    let x = tensor(ramp(2 * 3 * 4 * 6, 0.05, -1.0), &[2, 3, 4, 6]);
    let model = Sequential::new()
        .add_unfold([2, 2], [2, 2], [0, 0], [1, 1])
        .and_then(|m| m.add_fold([4, 6], [2, 2], [2, 2], [0, 0], [1, 1]))
        .unwrap();
    let y = model.predict(&x).unwrap();
    assert_eq!(y.shape(), x.shape());
    assert_eq!(dense_vec(&y), dense_vec(&x));

    // 重複窓（stride=1）では各要素が「それを含む窓の数」倍になる（ones 入力で確認）。
    let ones = tensor(vec![1.0; 4 * 4], &[1, 1, 4, 4]);
    let overlap = Sequential::new()
        .add_unfold([2, 2], [1, 1], [0, 0], [1, 1])
        .and_then(|m| m.add_fold([4, 4], [2, 2], [1, 1], [0, 0], [1, 1]))
        .unwrap();
    let got = dense_vec(&overlap.predict(&ones).unwrap());
    let want = vec![
        1.0, 2.0, 2.0, 1.0, 2.0, 4.0, 4.0, 2.0, 2.0, 4.0, 4.0, 2.0, 1.0, 2.0, 2.0, 1.0,
    ];
    assert_eq!(got, want);
}

#[test]
fn fold_unfold_reject_wrong_rank_or_shape_input() {
    let unfold = Sequential::new().add_unfold(K, S, P, D).unwrap();
    // バッチなし（rank 3）は非対応。
    assert!(
        unfold
            .predict(&tensor(ramp(3 * 4 * 5, 0.1, 0.0), &[3, 4, 5]))
            .is_err()
    );
    // 窓がパディング後の入力を超える。
    assert!(
        Sequential::new()
            .add_unfold([9, 9], [1, 1], [0, 0], [1, 1])
            .unwrap()
            .predict(&image())
            .is_err()
    );
    let fold = Sequential::new().add_fold([4, 5], K, S, P, D).unwrap();
    // rank 不正。
    assert!(fold.predict(&image()).is_err());
    // K 不整合（C·kH·kW で割り切れない 11）・L 不整合（10 ではない 9）。
    assert!(
        fold.predict(&tensor(ramp(2 * 11 * 10, 0.1, 0.0), &[2, 11, 10]))
            .is_err()
    );
    assert!(
        fold.predict(&tensor(ramp(2 * 12 * 9, 0.1, 0.0), &[2, 12, 9]))
            .is_err()
    );
}

// ---------------------------------------------------------------- backward

#[test]
fn unfold_backward_matches_direct_graph_bit_exact() {
    let x_data = image();
    let model = Sequential::new().add_unfold(K, S, P, D).unwrap();
    let tape1 = fandhe_ai::tape();
    let x1 = tape1.var(&x_data);
    let loss1 = model.forward(&tape1, &x1).unwrap().sum(None).unwrap();
    let grads1 = tape1.backward(&loss1).unwrap();
    let dx1 = grads1.get(&x1).unwrap().expect("入力へ勾配が到達するはず");

    let tape2 = fandhe_ai::tape();
    let x2 = tape2.var(&x_data);
    let loss2 = x2.unfold(K, S, P, D).unwrap().sum(None).unwrap();
    let grads2 = tape2.backward(&loss2).unwrap();
    let dx2 = grads2.get(&x2).unwrap().expect("入力へ勾配が到達するはず");
    assert_eq!(dense_vec(dx1), dense_vec(dx2));
}

#[test]
fn fold_backward_matches_direct_graph_bit_exact() {
    let x_data = columns();
    let model = Sequential::new().add_fold([4, 5], K, S, P, D).unwrap();
    let tape1 = fandhe_ai::tape();
    let x1 = tape1.var(&x_data);
    let loss1 = model.forward(&tape1, &x1).unwrap().sum(None).unwrap();
    let grads1 = tape1.backward(&loss1).unwrap();
    let dx1 = grads1.get(&x1).unwrap().expect("入力へ勾配が到達するはず");

    let tape2 = fandhe_ai::tape();
    let x2 = tape2.var(&x_data);
    let loss2 = x2.fold([4, 5], K, S, P, D).unwrap().sum(None).unwrap();
    let grads2 = tape2.backward(&loss2).unwrap();
    let dx2 = grads2.get(&x2).unwrap().expect("入力へ勾配が到達するはず");
    assert_eq!(dense_vec(dx1), dense_vec(dx2));
}

// ---------------------------------------------------------------- 学習 1 step

/// `Conv2d(1→2, k=2) → Unfold(k=2) → Flatten(1,2) → Linear(32→3)`（入力 `[N,1,4,4]`。
/// conv 出力 `[N,2,3,3]` → unfold `[N,8,4]` → flatten `[N,32]`）。`roundtrip` なら
/// `Unfold → Fold` の往復（`[N,2,3,3]` → Flatten(1,3) → Linear(18→3)）に置き換える。
fn mixed_model(roundtrip: bool) -> Sequential {
    let m = Sequential::new()
        .add_conv2d(1, 2, [2; 2], [1; 2], [0; 2], [1; 2], 1, SEED_C)
        .unwrap()
        .add_unfold([2; 2], [1; 2], [0; 2], [1; 2])
        .unwrap();
    if roundtrip {
        m.add_fold([3, 3], [2; 2], [1; 2], [0; 2], [1; 2])
            .unwrap()
            .add_flatten(1, 3)
            .add_linear(18, 3, SEED_L)
            .unwrap()
    } else {
        m.add_flatten(1, 2).add_linear(32, 3, SEED_L).unwrap()
    }
}

fn train_data() -> (Tensor<f32>, Tensor<f32>) {
    (
        tensor(ramp(4 * 16, 0.02, -0.5), &[4, 1, 4, 4]),
        tensor(ramp(4 * 3, 0.1, 0.0), &[4, 3]),
    )
}

#[test]
fn fold_unfold_do_not_change_trainable_parameters() {
    for roundtrip in [false, true] {
        let model = mixed_model(roundtrip);
        let params = model.trainable_parameters();
        assert_eq!(params.len(), 4, "conv(w,b) + linear(w,b)");
        assert_eq!(params[0].shape(), &[2, 1, 2, 2]);
        assert_eq!(params[2].shape(), &[if roundtrip { 18 } else { 32 }, 3]);
    }
}

#[test]
fn one_training_step_matches_reference_graph_bit_exact() {
    let (x, target) = train_data();
    for roundtrip in [false, true] {
        let mut model = mixed_model(roundtrip);
        let mut sgd = Sgd::new(SgdConfig::new(0.05)).unwrap();
        let updated = {
            let tape = fandhe_ai::tape();
            let bound = model.bind(&tape);
            let xv = tape.var(&x);
            let tv = tape.var(&target);
            let loss = bound.forward(&tape, &xv).unwrap().mse_loss(&tv).unwrap();
            let grads = tape.backward(&loss).unwrap();
            let grad_refs = bound.trainable_grads(&grads).unwrap();
            let param_refs = model.trainable_parameters();
            sgd.step(&param_refs, &grad_refs).unwrap()
        };
        model.apply_parameters(updated).unwrap();

        // 参照: 同一 seed の Unfold／Fold なし 2 分割モデルへ `Var` メソッドで直接挟む。
        let a = Sequential::new()
            .add_conv2d(1, 2, [2; 2], [1; 2], [0; 2], [1; 2], 1, SEED_C)
            .unwrap();
        let b = if roundtrip {
            Sequential::new()
                .add_flatten(1, 3)
                .add_linear(18, 3, SEED_L)
                .unwrap()
        } else {
            Sequential::new()
                .add_flatten(1, 2)
                .add_linear(32, 3, SEED_L)
                .unwrap()
        };
        let mut sgd_ref = Sgd::new(SgdConfig::new(0.05)).unwrap();
        let updated_ref = {
            let tape = fandhe_ai::tape();
            let ba = a.bind(&tape);
            let bb = b.bind(&tape);
            let xv = tape.var(&x);
            let tv = tape.var(&target);
            let h = ba.forward(&tape, &xv).unwrap();
            let u = h.unfold([2; 2], [1; 2], [0; 2], [1; 2]).unwrap();
            let p = if roundtrip {
                u.fold([3, 3], [2; 2], [1; 2], [0; 2], [1; 2]).unwrap()
            } else {
                u
            };
            let loss = bb.forward(&tape, &p).unwrap().mse_loss(&tv).unwrap();
            let grads = tape.backward(&loss).unwrap();
            let mut grad_refs = ba.trainable_grads(&grads).unwrap();
            grad_refs.extend(bb.trainable_grads(&grads).unwrap());
            let mut param_refs = a.trainable_parameters();
            param_refs.extend(b.trainable_parameters());
            sgd_ref.step(&param_refs, &grad_refs).unwrap()
        };

        let got: Vec<Vec<f32>> = model
            .trainable_parameters()
            .iter()
            .map(|t| dense_vec(t))
            .collect();
        let want: Vec<Vec<f32>> = updated_ref.iter().map(dense_vec).collect();
        assert_eq!(got, want, "roundtrip={roundtrip}");
    }
}

#[test]
fn compile_and_fit_run_with_fold_unfold() {
    let (x, target) = train_data();
    for roundtrip in [false, true] {
        let mut model = mixed_model(roundtrip);
        model
            .compile(Optimizer::Sgd(SgdConfig::new(0.05)), Loss::Mse)
            .unwrap();
        let w0 = dense_vec(model.trainable_parameters()[0]);
        model.fit(&x, &target, FitConfig::new(2, 4)).unwrap();
        assert_ne!(
            dense_vec(model.trainable_parameters()[0]),
            w0,
            "roundtrip={roundtrip}"
        );
    }
}

// ---------------------------------------------------------------- 常駐経路

#[test]
fn resident_path_rejects_fold_unfold_fail_closed() {
    let unfold = Sequential::new().add_unfold(K, S, P, D).unwrap();
    let fold = Sequential::new().add_fold([4, 5], K, S, P, D).unwrap();
    let linear_only = Sequential::new().add_linear(6, 6, SEED_L).unwrap();
    let init_tape = fandhe_ai::tape();
    let mut store = linear_only.init_device_param_store(&init_tape).unwrap();
    drop(init_tape);

    for (model, x) in [(&unfold, image()), (&fold, columns())] {
        let tape = fandhe_ai::tape();
        assert!(matches!(
            model.init_device_param_store(&tape).unwrap_err(),
            BackendError::Unsupported(_)
        ));
        let xv = tape.var(&x);
        assert!(matches!(
            model.forward_resident(&tape, &xv, &mut store).unwrap_err(),
            AutodiffError::Backend(BackendError::Unsupported(_))
        ));
        assert!(matches!(
            model.predict_resident(&store, &x).unwrap_err(),
            AutodiffError::Backend(BackendError::Unsupported(_))
        ));
    }
}

// ---------------------------------------------------------------- 保存・ONNX（fail-closed）

/// 保存対応は後続 `-c`（`LayerSpec` 実 variant 化）の担当。それまでは `UnsupportedModel` で
/// 書き込み前に拒否する。`-c` は本テストを成功系（ラウンドトリップ bit 一致）へ書き換えること。
#[test]
fn save_model_rejects_fold_unfold_as_unsupported_model() {
    let unfold = Sequential::new().add_unfold(K, S, P, D).unwrap();
    let fold = Sequential::new().add_fold([4, 5], K, S, P, D).unwrap();
    for (tag, model) in [("unfold", &unfold), ("fold", &fold)] {
        let guard = TempDirGuard::new(&format!("fold_unfold_reject_{tag}"));
        let dir = guard.path().join("m");
        let err = save_model(model, &dir).unwrap_err();
        assert!(
            matches!(err, ModelIoError::UnsupportedModel { .. }),
            "{tag}: {err:?}"
        );
        assert!(!dir.exists(), "拒否時は dir に何も作らない");
    }
}

#[test]
fn onnx_export_rejects_fold_unfold_as_unsupported() {
    let unfold = Sequential::new().add_unfold(K, S, P, D).unwrap();
    let fold = Sequential::new().add_fold([4, 5], K, S, P, D).unwrap();
    for m in [&unfold, &fold] {
        let err = OnnxModel::from_sequential(m).unwrap_err();
        assert!(matches!(err, OnnxError::UnsupportedLayer { .. }), "{err:?}");
    }
}

// ---------------------------------------------------------------- train／eval 非依存

#[test]
fn fold_unfold_is_mode_independent() {
    let mut model = Sequential::new()
        .add_unfold(K, S, P, D)
        .and_then(|m| m.add_fold([4, 5], K, S, P, D))
        .unwrap();
    let x = image();
    model.train();
    let a = dense_vec(&model.predict(&x).unwrap());
    model.eval();
    let b = dense_vec(&model.predict(&x).unwrap());
    assert_eq!(a, b);
}
