# L-BFGS（closure・strong Wolfe line search）設計判断記録

イシュー #2197（親 #2172「LBFGS」）。`docs/autodiff-param-groups-decision.md`
（#2173）・`docs/autodiff-optimizer-adadelta-adamax-nadam-radam-decision.md`
（#2171）と同型の記録。本 doc は `crates/autodiff/src/nn/optim/lbfgs.rs`
のモジュール doc・実装を正として要約したものであり、両者が食い違う
場合は実装側を正とする。

## §0 結論

`torch.optim.LBFGS.step(closure)` 相当を値型で提供する **内部クレート
限定**の optimizer として実装した: `fandhe_ai_autodiff::nn::optim::
{Lbfgs, LbfgsConfig, LbfgsLineSearch}`（`crates/autodiff/src/nn/optim/
lbfgs.rs`。`nn/optim/mod.rs:107` で再エクスポート）。既存 optimizer
（`AdamW` 等）が採る「`(param, grad)` の参照列 → 更新後 `Tensor<f32>`
列」という 1 step あたり勾配評価 1 回の値型・純関数パターンでは
line search 中の複数回評価を表現できないため、呼び出し元が用意する
**closure**（パラメータ列 → `(損失, 勾配列)`）を optimizer が内部で
複数回呼び出す `step` メソッドを別途設けた。

facade（`fandhe_ai::optim`）への公開・`compile()` 統合は**別イシュー
#2198**（facade 公開面拡張はユーザー承認事項。親 #2131 の「facade 公開
面の拡張は設計判断記録 → 承認 → 実装の 2 段」規則）として保留する。

## §1 使い方

`Lbfgs::new(LbfgsConfig)` で構築し、`try_step_closure`（可失敗
closure）または `step_closure`（不失敗 closure。内部で
`try_step_closure` へ委譲する薄いラッパー）へパラメータ列と closure を
渡す。closure は「現在の試行パラメータ列 → `(損失, 勾配列)`」を返す
関数で、`Lbfgs` は 1 回の `step` 呼び出し内で line search のため
**closure を複数回評価する**（固定ステップは反復ごとに 1 回程度、
strong Wolfe は `LbfgsConfig::line_search_steps` の予算内で複数回）。

既存 optimizer（`AdamW` 等）は呼び出し元が 1 回だけ forward/backward
した `Gradients` を渡す前提だが、L-BFGS の closure は **毎評価ごとに
新しい `Tape` を構築して forward/backward をやり直す**形になる（1 回の
`step` 呼び出しで forward/backward が複数回走る）。呼び出し元が
`Tape::backward` 等を駆動して勾配を作る責務を持つ点が、既存 optimizer
の「呼び出し元が層を再構築する」不変更新パターンとは異なる
（`nn/optim/mod.rs` 冒頭 doc 参照）。

```rust,ignore
use fandhe_ai_autodiff::nn::Linear;
use fandhe_ai_autodiff::nn::optim::{Lbfgs, LbfgsConfig, LbfgsLineSearch};
use fandhe_ai_autodiff::{AutodiffError, Tape};
use fandhe_ai_tensor_core::Tensor;

let mut opt = Lbfgs::new(LbfgsConfig {
    line_search: LbfgsLineSearch::StrongWolfe,
    ..LbfgsConfig::default()
})?;

// x_data／y_data は学習データ、weight0／bias0 は Linear の初期パラメータ
// （呼び出し元が用意する）。

// closure は呼ばれるたびに新しい Tape を構築し、forward から backward
// までをやり直して (損失, 勾配列) を返す（1 回の step 内で複数回
// 評価されうる）。
let eval = |params: &[Tensor<f32>]| -> Result<(f32, Vec<Tensor<f32>>), AutodiffError> {
    let tape = Tape::new();
    let x = tape.var(&x_data);
    let y = tape.var(&y_data);
    let linear = Linear::from_parameters(params[0].clone(), Some(params[1].clone()))?;
    let bound = linear.bind(&tape);
    let pred = bound.forward(&x)?;
    let loss = pred.mse_loss(&y)?;
    let no_grad = || AutodiffError::InvalidArgument("grad missing for leaf".to_string());
    let loss_value = loss.to_tensor().get(&[]).ok_or_else(no_grad)?;
    let grads = tape.backward(&loss)?;
    let w_grad = grads.get(&bound.weight)?.ok_or_else(no_grad)?.clone();
    let b_grad = grads
        .get(bound.bias.as_ref().ok_or_else(no_grad)?)?
        .ok_or_else(no_grad)?
        .clone();
    Ok((loss_value, vec![w_grad, b_grad]))
};

let updated = opt.try_step_closure(&[weight0, bias0], eval)?;
```

## §2 設定値と既定値（PyTorch と同一）

`LbfgsConfig`（`Default` 実装は `torch.optim.LBFGS` の既定値と同一）:

| フィールド | 既定値 | 意味 |
|---|---|---|
| `lr` | `1.0` | 学習率。固定ステップのステップ幅・strong Wolfe の初期試行ステップ幅に掛かる |
| `max_iter` | `20` | 1 回の `step` あたりの最大反復数 |
| `max_eval` | `None`（`max_iter * 5 / 4` へ解決） | 1 回の `step` あたりの closure 評価回数の上限 |
| `tolerance_grad` | `1e-7` | 勾配収束判定の閾値 |
| `tolerance_change` | `1e-9` | 変化量収束判定の閾値 |
| `history_size` | `100` | 曲率ペア `(s, y)` の履歴保持数 |
| `line_search` | `LbfgsLineSearch::None`（固定ステップ） | `None`／`StrongWolfe` の 2 択 |
| `line_search_steps` | `25` | strong Wolfe 1 回あたりの試行上限の追加キャップ |

`Lbfgs::new` は各フィールドを検証し、範囲外・非有限値・
`max_iter * 5` の overflow（`checked_mul` で検出）を `InvalidArgument`
として拒否する（本番経路 panic 禁止・`.claude/rules/coding-rust.md`）。

## §3 PyTorch との意図的な逸脱

`lbfgs.rs` モジュール冒頭 doc「PyTorch との対応・意図的な逸脱」節の
要約（詳細・理由は同節・discussion 参照箇所を正とする）:

- **戻り値**: `Result<_, AutodiffError>` を返す（本クレート全 optimizer
  共通の契約）。可失敗 closure 用 `try_step_closure` と不失敗 closure 用
  `step_closure` の 2 メソッドを提供する
- **空パラメータ列**: closure を呼ばずに `InvalidArgument`
- **closure 戻り値の検証**: 勾配数・shape の不一致、損失・勾配・
  「更新後の `x`」・入力 `params` 自体の非有限値を毎評価で fail-closed
  検出する（PyTorch は無検査。`.claude/rules/security.md` A03）
- **エラー時の状態不変**: `try_step_closure` 1 回の呼び出し内の全状態
  更新はローカル作業コピー上で行い、正常終了時にのみ `self` へ
  コミットする
- **`max_eval` の解決**: `None` の場合 `max_iter * 5 / 4`（PyTorch と
  同じ整数除算）。`checked_mul` で overflow を検出し `Lbfgs::new` が
  `InvalidArgument` を返す
- **固定ステップの `max_eval` 厳守**: PyTorch は予算をちょうど使い切った
  直後の反復でも closure をもう 1 回評価しうるが、本実装は
  `LbfgsConfig::max_eval` の doc が謳う「1 回の `step` あたりの closure
  評価回数の上限」を厳密な契約として扱い、予算切れ後の 1 回追加評価を
  行わない
- **`t == 0.0` では line search を行わない**: PyTorch の逐語移植は
  `t == 0` でも closure を評価しうるが、本実装は無駄な closure 呼び出し
  になる同一入力への再評価を省く
- **対象外**: `maximize`・複素数パラメータ・parameter group・sparse 勾配

## §4 数値型の方針

フラットベクトル上の縮約（`g·d`／`y·s`／`y·y`／`s·q`／`y·r`／`‖g‖₁`）は
`f64` アキュムレータで index 順に蓄積し最後に 1 回 `f32` へ丸める
（`.claude/rules/coding-rust.md` の縮約方針）。スカラー演算（`ro`／
`H_diag`／`al`／`t`／`gtd`／cubic 補間／Wolfe 条件比較）は PyTorch の
f32 テンソル演算に合わせ `f32`。`|loss - prev_loss| < tolerance_change`
のみ PyTorch が Python float（f64）で計算するため `loss`／`prev_loss`
を `f64` で保持する。ベクトル更新（`q -= al·y` 等）は `f32::mul_add`
（FMA 契約）を使う。

## §5 検証方法

- **PyTorch 参照 fixture**（`crates/autodiff/tests/fixtures/
  lbfgs-pytorch-reference/`）: 実 PyTorch 2.14.0+cpu 実行値
  （`torch/optim/lbfgs.py` の `LBFGS.step`／`_strong_wolfe`／
  `_cubic_interpolate` を実装前に読み、演算順の一致を確認済み）を
  `gen_reference.py` で 1 回生成し JSON としてコミット（CI は
  Python/PyTorch に依存しない。`.claude/rules/ci.md`「グローバル状態を
  汚す処理を workflow に書かない」）。目的関数は凸・良条件の最小二乗
  回帰。`lr` は 2 の冪（`1.0`／`0.5`）のみを使い、f64→f32 丸め値が
  純 f32 演算の結果と bit 一致するケースに限定する
- `crates/autodiff/tests/nn_optim_lbfgs.rs::lbfgs_matches_pytorch_reference`
  が上記 fixture との統一複合判定（`.claude/rules/coding-rust.md`
  tolerance）で照合する
- 固定ステップ・strong Wolfe 双方の line search 予算・`t == 0.0`
  ショートサーキット・エラー時の状態不変（`n_iter`／`func_evals`／
  履歴／`d`／`t`／`prev_flat_grad` が変化しないこと）を単体テストで
  固定
- 本実装はホスト `Tensor<f32>` 経路のみで新規 `Op`／`BackendOps`／VJP／
  GPU カーネルを追加していないため、CUDA／Metal parity の申し送りは
  不要

## §6 スコープ外

- facade（`fandhe_ai::optim`）への公開・`compat::Sequential::compile()`
  との統合は別イシュー **#2198**（facade 公開面拡張はユーザー承認
  事項）
- `crate::optim::device_store::DeviceParamStore` 常駐経路への対応
- param groups（`nn::optim::param_group::ParamGroupStep`。#2173）への
  対応
- `maximize`・複素数パラメータ・sparse 勾配（§3 参照）
