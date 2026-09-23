//! 層別 `requires_grad` 凍結（`Module::freeze`／`set_requires_grad`）の
//! 契約テスト（イシュー #2137）。設計・帰属は
//! `docs/autodiff-nograd-leaf-dinput-skip-decision.md`「実装記録
//! （#2137）」節を参照。
//!
//! `requires_grad` は葉登録時に確定するテープ側メタデータ
//! （`TapeNode::requires_grad`）に過ぎず、`backward.rs::accumulate`
//! 呼び出し前のゲート判定以外の経路（融合プラン選択・forward 演算列）
//! には一切影響しない（`tape.rs::push_eager`／`push_leaf` 参照）。
//! したがって凍結の有無で出力・非凍結パラメータの勾配は bit 完全に
//! 一致するはずであり、以下のテストはその前提を hard assert する。

mod common;

use fandhe_ai_autodiff::nn::activation::Relu;
use fandhe_ai_autodiff::nn::{Linear, Module, ModuleList, Sequential};
use fandhe_ai_autodiff::{AutodiffError, Tape};
use fandhe_ai_tensor_core::Tensor;

fn t(data: Vec<f32>, shape: &[usize]) -> Tensor<f32> {
    Tensor::new(data, shape).expect("test fixture: shape とデータ長は事前に一致させている")
}

/// T1: 既定契約。パラメータを持つ層（`Linear`）は `requires_grad() ==
/// true`。パラメータを持たない層（`Relu`。`set_requires_grad`／
/// `requires_grad` をオーバーライドせず trait 既定のまま）は
/// `set_requires_grad(false)` が `Ok` を返し（`named_parameters` が
/// 空のため既定実装の fail-closed 検査を通過する）、`requires_grad()`
/// は既定 `true` のまま変化しない（`Module::requires_grad` doc「既定
/// `true`」節。無状態層は保持する状態を持たないため）。
#[test]
fn default_contract_linear_is_true_and_parameterless_layer_set_requires_grad_is_noop_ok() {
    let linear = Linear::new(3, 2, true, 1).expect("seed=1 は有効な構築引数");
    assert!(linear.requires_grad());

    let mut relu = Relu;
    assert!(relu.named_parameters().is_empty());
    assert!(relu.set_requires_grad(false).is_ok());
    // パラメータを持たない層は状態を保持しないため、既定 `true` のまま。
    assert!(relu.requires_grad());
}

/// T2: `Linear` を freeze → bind → forward → backward すると、
/// (a) 出力は凍結なしと bit 一致する、(b) `grads.get(&weight/bias)` は
/// `Err(GradientTrackingDisabled)` を返す、(c) 入力 `x` の勾配（x は
/// 常に `requires_grad=true`）も凍結なしと bit 一致する。
#[test]
fn freeze_linear_output_and_input_grad_match_unfrozen_while_own_grad_is_disabled() {
    let x_data = t(vec![1.0, 2.0, -1.0, 0.5], &[2, 2]);

    // 経路 A: 凍結なし。
    let tape_a = Tape::new_with_ops(common::naive_ops());
    let linear_a = Linear::new(2, 3, true, 7).expect("seed=7 は有効");
    let x_a = tape_a.var(&x_data);
    let out_a = linear_a
        .bind(&tape_a)
        .forward(&x_a)
        .expect("shape が一致するため forward は成功する");
    let loss_a = out_a.sum(None).expect("全軸縮約は失敗しない");
    let grads_a = tape_a
        .backward(&loss_a)
        .expect("x が追跡対象のため成功する");
    let dx_a = grads_a
        .get(&x_a)
        .expect("x は requires_grad=true")
        .expect("x は loss に到達する");

    // 経路 B: `Linear::freeze()` 後。
    let tape_b = Tape::new_with_ops(common::naive_ops());
    let mut linear_b = Linear::new(2, 3, true, 7).expect("同じ seed で同じ初期値");
    linear_b
        .freeze()
        .expect("Linear は set_requires_grad をオーバーライド済み");
    assert!(!linear_b.requires_grad());
    let vars_b = linear_b.bind(&tape_b);
    let x_b = tape_b.var(&x_data);
    let out_b = vars_b
        .forward(&x_b)
        .expect("shape が一致するため forward は成功する");

    assert_eq!(
        out_a.to_tensor().as_slice().unwrap(),
        out_b.to_tensor().as_slice().unwrap(),
        "freeze は forward の値に影響しない（requires_grad は純粋なメタデータ）"
    );

    let loss_b = out_b.sum(None).expect("全軸縮約は失敗しない");
    let grads_b = tape_b
        .backward(&loss_b)
        .expect("x が追跡対象のため成功する");

    let dx_b = grads_b
        .get(&x_b)
        .expect("x は requires_grad=true")
        .expect("x は loss に到達する");
    assert_eq!(
        dx_a.as_slice().unwrap(),
        dx_b.as_slice().unwrap(),
        "凍結の有無で x の勾配は bit 一致する"
    );

    let err_w = grads_b
        .get(&vars_b.weight)
        .expect_err("凍結した weight は構造的に勾配を持たない");
    assert!(matches!(err_w, AutodiffError::GradientTrackingDisabled));
    let err_b = grads_b
        .get(vars_b.bias.as_ref().expect("bias=true で構築した"))
        .expect_err("凍結した bias は構造的に勾配を持たない");
    assert!(matches!(err_b, AutodiffError::GradientTrackingDisabled));
}

/// T3: `set_requires_grad(true)` で追跡が戻り、勾配が一度も凍結
/// しなかった場合と bit 一致する（凍結 → 解凍の往復が状態を破壊
/// しない）。
#[test]
fn unfreezing_restores_gradient_tracking_matching_never_frozen_path() {
    let x_data = t(vec![0.3, -0.7, 1.2, 0.4], &[2, 2]);

    // 経路 B: 凍結 → 解凍。
    let mut linear_b = Linear::new(2, 2, true, 3).expect("seed=3 は有効");
    linear_b.freeze().expect("freeze は成功する");
    linear_b
        .set_requires_grad(true)
        .expect("set_requires_grad(true) で解凍する");
    assert!(linear_b.requires_grad());

    let tape_b = Tape::new_with_ops(common::naive_ops());
    let vars_b = linear_b.bind(&tape_b);
    let x_b = tape_b.var(&x_data);
    let out_b = vars_b.forward(&x_b).expect("forward は成功する");
    let loss_b = out_b.sum(None).expect("全軸縮約は失敗しない");
    let grads_b = tape_b.backward(&loss_b).expect("解凍済みのため成功する");

    let dw_b = grads_b
        .get(&vars_b.weight)
        .expect("解凍済みのため Err にならない")
        .expect("weight は loss に到達する");
    let db_b = grads_b
        .get(vars_b.bias.as_ref().expect("bias=true"))
        .expect("解凍済みのため Err にならない")
        .expect("bias は loss に到達する");

    // 経路 C（参照）: 一度も凍結しなかった経路。
    let tape_c = Tape::new_with_ops(common::naive_ops());
    let linear_c = Linear::new(2, 2, true, 3).expect("同じ seed で同じ初期値");
    let vars_c = linear_c.bind(&tape_c);
    let x_c = tape_c.var(&x_data);
    let out_c = vars_c.forward(&x_c).expect("forward は成功する");
    let loss_c = out_c.sum(None).expect("全軸縮約は失敗しない");
    let grads_c = tape_c.backward(&loss_c).expect("backward は成功する");
    let dw_c = grads_c
        .get(&vars_c.weight)
        .expect("Err にならない")
        .expect("weight は loss に到達する");
    let db_c = grads_c
        .get(vars_c.bias.as_ref().expect("bias=true"))
        .expect("Err にならない")
        .expect("bias は loss に到達する");

    assert_eq!(dw_b.as_slice().unwrap(), dw_c.as_slice().unwrap());
    assert_eq!(db_b.as_slice().unwrap(), db_c.as_slice().unwrap());
}

/// T5: `Sequential`／`ModuleList` の `freeze()` が全子へ再帰する。
/// `get_mut(i)`／`layers_mut()[i]` で特定層だけ解凍できる。
#[test]
fn sequential_and_module_list_freeze_propagates_to_all_children() {
    let mut seq = Sequential::new()
        .add(Linear::new(4, 4, true, 10).expect("seed=10 は有効"))
        .add(Linear::new(4, 4, true, 11).expect("seed=11 は有効"));

    seq.freeze().expect("全子が Linear のため成功する");
    assert!(!seq.requires_grad());
    for layer in seq.layers() {
        assert!(!layer.requires_grad());
    }

    // 特定層だけ解凍する（`layers_mut()` 経由）。
    seq.layers_mut()[0]
        .set_requires_grad(true)
        .expect("Linear は失敗しない");
    assert!(seq.layers()[0].requires_grad());
    assert!(!seq.layers()[1].requires_grad());
    // `Sequential::requires_grad`（all）は 1 つでも false なら false。
    assert!(!seq.requires_grad());

    let mut list = ModuleList::new();
    list.push(Box::new(
        Linear::new(2, 2, true, 20).expect("seed=20 は有効"),
    ));
    list.push(Box::new(
        Linear::new(2, 2, true, 21).expect("seed=21 は有効"),
    ));
    list.set_requires_grad(false)
        .expect("全子が Linear のため成功する");
    for i in 0..list.len() {
        assert!(!list.get(i).unwrap().requires_grad());
    }
    list.get_mut(1).unwrap().freeze().unwrap(); // 既に凍結済みでも冪等。
    list.get_mut(0)
        .unwrap()
        .set_requires_grad(true)
        .expect("Linear は失敗しない");
    assert!(list.get(0).unwrap().requires_grad());
    assert!(!list.get(1).unwrap().requires_grad());
}

/// T6: 全層凍結かつ入力が `var_no_grad` の場合、`tape.backward` は
/// `Err(AutodiffError::Backward(..))`（#1748 §5.4 と同じ「追跡対象の
/// 祖先を持たない」契約）。入力が `tape.var`（既定 requires_grad=true）
/// なら backward は成功する（x 自身の勾配は得られる）。
#[test]
fn all_layers_frozen_with_no_grad_input_rejects_backward_but_succeeds_with_tracked_input() {
    let mut linear = Linear::new(2, 2, true, 5).expect("seed=5 は有効");
    linear.freeze().expect("Linear は失敗しない");

    let tape_notrack = Tape::new_with_ops(common::naive_ops());
    let x_notrack = tape_notrack.var_no_grad(&t(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]));
    let out_notrack = linear
        .bind(&tape_notrack)
        .forward(&x_notrack)
        .expect("forward は requires_grad と無関係に成功する");
    let loss_notrack = out_notrack.sum(None).expect("全軸縮約は失敗しない");
    let result = tape_notrack.backward(&loss_notrack);
    assert!(
        matches!(result, Err(AutodiffError::Backward(_))),
        "全層凍結 + 入力も追跡なしなら Err(Backward): {result:?}"
    );

    let tape_track = Tape::new_with_ops(common::naive_ops());
    let x_track = tape_track.var(&t(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]));
    let out_track = linear
        .bind(&tape_track)
        .forward(&x_track)
        .expect("forward は requires_grad と無関係に成功する");
    let loss_track = out_track.sum(None).expect("全軸縮約は失敗しない");
    let grads_track = tape_track
        .backward(&loss_track)
        .expect("x が requires_grad=true のため成功する");
    let dx = grads_track
        .get(&x_track)
        .expect("x は requires_grad=true")
        .expect("x は loss に到達する");
    assert_eq!(dx.shape(), &[2usize, 2]);
}

/// T7: fail-closed 既定。`named_parameters` だけをオーバーライドし
/// `set_requires_grad` を未実装のまま残した外部 `Module` 実装に対して
/// `freeze()` は `Err` を返す（`Module::set_requires_grad` 既定実装
/// doc「fail-closed」節）。これを含む `ModuleList` では `Err` となり、
/// 先に適用済みの子が元の `requires_grad` に戻る（ベストエフォート・
/// ロールバック）。
#[test]
fn fail_closed_default_rejects_freeze_and_module_list_rolls_back_on_failure() {
    struct HalfImplementedModule {
        param: Tensor<f32>,
    }

    impl Module for HalfImplementedModule {
        fn forward<'t>(
            &self,
            _tape: &'t Tape,
            input: &fandhe_ai_autodiff::Var<'t>,
        ) -> Result<fandhe_ai_autodiff::Var<'t>, AutodiffError> {
            Ok(*input)
        }

        fn named_parameters(&self) -> Vec<(String, &Tensor<f32>)> {
            vec![("param".to_string(), &self.param)]
        }
        // `set_requires_grad` は意図的にオーバーライドしない
        // （trait 既定の fail-closed 挙動をそのまま試す）。
    }

    let mut half = HalfImplementedModule {
        param: t(vec![1.0, 2.0], &[2]),
    };
    let err = half
        .freeze()
        .expect_err("named_parameters が非空なのに set_requires_grad 未実装は Err");
    assert!(matches!(err, AutodiffError::InvalidArgument(_)));

    // index 0: 正常に応答する `Linear`。index 1: 上記の半端な実装。
    let mut list = ModuleList::new();
    list.push(Box::new(Linear::new(2, 2, true, 9).expect("seed=9 は有効")));
    list.push(Box::new(HalfImplementedModule {
        param: t(vec![3.0, 4.0], &[2]),
    }));

    assert!(list.get(0).unwrap().requires_grad());
    let err = list
        .set_requires_grad(false)
        .expect_err("index 1 が Err を返すため全体も Err になる");
    assert!(matches!(err, AutodiffError::InvalidArgument(_)));
    // ロールバックにより index 0 は元の（凍結前の）状態へ戻る。
    assert!(
        list.get(0).unwrap().requires_grad(),
        "index 1 の失敗で index 0 の変更がロールバックされる"
    );
}

/// T9: `training`／`requires_grad` は独立の軸（`Module::
/// set_requires_grad` doc「`training`／`set_training` とは独立の軸」
/// 節）。`Linear` 自体はモード非依存（`training`／`set_training` は
/// trait 既定＝no-op のまま。モードの正はコンテナが持つ契約。
/// `module.rs` の trait doc 参照）のため、`freeze()`／
/// `set_training(false)` のどちらを呼んでも相手の状態を変えない
/// ことを両方向で確認する。
#[test]
fn freezing_and_training_mode_are_independent_axes() {
    let mut linear = Linear::new(2, 2, true, 1).expect("seed=1 は有効");
    assert!(linear.training());
    assert!(linear.requires_grad());

    linear.freeze().expect("Linear は失敗しない");
    assert!(
        linear.training(),
        "freeze は set_training を経由しないため training モードは不変"
    );
    assert!(!linear.requires_grad());

    // `Linear` は `set_training`/`training` を trait 既定のまま
    // （no-op／常に `true`）オーバーライドしないため、`set_training`
    // 自体は状態を変えない——ここでは「requires_grad が引き続き
    // 独立に false のまま」であることのみを確認する（`training()` は
    // 既定契約により呼び出し後も `true` のまま変化しない）。
    linear.set_training(false);
    assert!(
        linear.training(),
        "Linear の training は trait 既定の no-op"
    );
    assert!(
        !linear.requires_grad(),
        "set_training は requires_grad を変更しない"
    );
}

/// E1（転移学習 example）: backbone（`Linear` → `Relu`）を freeze、
/// head（`Linear`）は学習可能として、各層を明示的に `bind` した
/// グラフを組む（`Sequential::forward` は内部 `bind` の `Var` を外へ
/// 出さないため、勾配読み出し・手動更新には個別 `bind` を使う）。
/// 決定的な手動 GD（`named_parameters`/`set_parameter` を介さず、
/// `Linear::weight()`/`from_parameters` で直接値を書き換える最小
/// 構成）で head だけを数ステップ更新する: backbone のパラメータは
/// bit 不変、head は変化し、loss は単調減少する。1 ステップ目の head
/// 勾配は「一度も凍結しなかった」参照 run と bit 一致する。
#[test]
fn transfer_learning_example_freezes_backbone_and_trains_head_only() {
    const LR: f32 = 0.05;
    const STEPS: usize = 3;

    let x_data = t(vec![1.0, -0.5, 0.25, 2.0, -1.0, 0.1, 0.3, -0.2], &[2, 4]);
    let target_data = t(vec![0.0, 1.0, 1.0, 0.0], &[2, 2]);

    fn forward_loss<'t>(
        tape: &'t Tape,
        backbone: &Linear,
        head: &Linear,
        x: &Tensor<f32>,
        target: &Tensor<f32>,
    ) -> (
        fandhe_ai_autodiff::Var<'t>,
        fandhe_ai_autodiff::nn::LinearVars<'t>,
        fandhe_ai_autodiff::nn::LinearVars<'t>,
    ) {
        let backbone_vars = backbone.bind(tape);
        let head_vars = head.bind(tape);
        let x_var = tape.var(x);
        let h = backbone_vars
            .forward(&x_var)
            .expect("backbone forward は成功する");
        let h = h.relu();
        let out = head_vars.forward(&h).expect("head forward は成功する");
        let target_var = tape.var(target);
        let loss = out
            .mse_loss(&target_var)
            .expect("shape が一致するため成功する");
        (loss, backbone_vars, head_vars)
    }

    // --- 参照 run: 一度も凍結せず 1 ステップ目の head 勾配を取る。---
    let backbone_ref = Linear::new(4, 4, true, 200).expect("seed=200 は有効");
    let head_ref = Linear::new(4, 2, true, 201).expect("seed=201 は有効");
    let tape_ref = Tape::new_with_ops(common::naive_ops());
    let (loss_ref, _backbone_vars_ref, head_vars_ref) =
        forward_loss(&tape_ref, &backbone_ref, &head_ref, &x_data, &target_data);
    let grads_ref = tape_ref
        .backward(&loss_ref)
        .expect("head が追跡対象のため成功する");
    let head_weight_grad_ref = grads_ref
        .get(&head_vars_ref.weight)
        .expect("head は requires_grad=true")
        .expect("head.weight は loss に到達する")
        .clone();

    // --- 本番 run: backbone を freeze。---
    let mut backbone = Linear::new(4, 4, true, 200).expect("参照 run と同じ seed");
    backbone.freeze().expect("Linear は失敗しない");
    let mut head = Linear::new(4, 2, true, 201).expect("参照 run と同じ seed");

    let backbone_weight_before = backbone.weight().clone();
    let backbone_bias_before = backbone.bias().expect("bias=true").clone();

    let mut losses = Vec::with_capacity(STEPS);
    let mut first_step_head_weight_grad: Option<Tensor<f32>> = None;

    for step in 0..STEPS {
        let tape = Tape::new_with_ops(common::naive_ops());
        let (loss, _backbone_vars, head_vars) =
            forward_loss(&tape, &backbone, &head, &x_data, &target_data);
        losses.push(loss.to_tensor().as_slice().unwrap()[0]);
        let grads = tape.backward(&loss).expect("head が追跡対象のため成功する");

        // backbone は凍結済みのため、backbone の勾配取得は
        // 構造的に `Err(GradientTrackingDisabled)` になる
        // （T2 と同じ契約。ここでは freeze の効果を再確認する）。
        // `_backbone_vars` は `#[allow(unused)]` の代わりに束縛のみ
        // 保持し、勾配取得の対象にしないことで「backbone は触らない」
        // 手動更新ループの意図を明示する。

        let dw = grads
            .get(&head_vars.weight)
            .expect("head.weight は requires_grad=true")
            .expect("head.weight は loss に到達する")
            .clone();
        let db = grads
            .get(head_vars.bias.as_ref().expect("head bias=true"))
            .expect("head.bias は requires_grad=true")
            .expect("head.bias は loss に到達する")
            .clone();

        if step == 0 {
            first_step_head_weight_grad = Some(dw.clone());
        }

        // 手動 GD: head.weight -= LR * dw、head.bias -= LR * db。
        let new_weight_data: Vec<f32> = head
            .weight()
            .contiguous()
            .as_slice()
            .expect("contiguous 化済み")
            .iter()
            .zip(dw.contiguous().as_slice().expect("contiguous 化済み"))
            .map(|(w, g)| w - LR * g)
            .collect();
        let new_bias_data: Vec<f32> = head
            .bias()
            .expect("bias=true")
            .contiguous()
            .as_slice()
            .expect("contiguous 化済み")
            .iter()
            .zip(db.contiguous().as_slice().expect("contiguous 化済み"))
            .map(|(b, g)| b - LR * g)
            .collect();
        let new_weight = Tensor::new(new_weight_data, head.weight().shape())
            .expect("shape は元の weight と同一");
        let new_bias = Tensor::new(new_bias_data, head.bias().expect("bias=true").shape())
            .expect("shape は元の bias と同一");
        head = Linear::from_parameters(new_weight, Some(new_bias))
            .expect("from_parameters は shape 検査済みの値のみ渡すため成功する");
    }

    // 1 ステップ目の head.weight 勾配は「一度も凍結しなかった」参照
    // run と bit 一致する（凍結は backbone にのみ効き、head の勾配
    // 計算経路には影響しない）。
    assert_eq!(
        first_step_head_weight_grad
            .expect("STEPS >= 1")
            .as_slice()
            .unwrap(),
        head_weight_grad_ref.as_slice().unwrap(),
        "1 ステップ目の head 勾配は凍結の有無で bit 一致する"
    );

    // backbone のパラメータは全ステップを通じて bit 不変。
    assert_eq!(
        backbone.weight().as_slice().unwrap(),
        backbone_weight_before.as_slice().unwrap(),
        "凍結した backbone.weight は学習ループを通じて不変"
    );
    assert_eq!(
        backbone.bias().expect("bias=true").as_slice().unwrap(),
        backbone_bias_before.as_slice().unwrap(),
        "凍結した backbone.bias は学習ループを通じて不変"
    );

    // loss は単調減少する（決定的シード・小さい LR での GD）。
    for w in losses.windows(2) {
        assert!(
            w[1] < w[0],
            "loss は単調減少するはず: {:?} -> {:?} (losses={losses:?})",
            w[0],
            w[1]
        );
    }
}
