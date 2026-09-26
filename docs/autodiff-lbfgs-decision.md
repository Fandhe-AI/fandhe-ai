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
#2198 は着手時点でも承認未取得のため保留固定（多層ガード）で
`Closes` している。保留の詳細・再開条件は §7、承認後の実装設計は §8 を
参照。

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
  事項。§7・§8 参照）
- `crate::optim::device_store::DeviceParamStore` 常駐経路への対応
- param groups（`nn::optim::param_group::ParamGroupStep`。#2173）への
  対応
- `maximize`・複素数パラメータ・sparse 勾配（§3 参照）

## §7 facade 公開・`compile()` 統合の保留（#2198）

イシュー #2198（親 #2172・ルート #2131）は facade 公開面の拡張
（`compat::Optimizer::Lbfgs(LbfgsConfig)` variant の追加・
`fandhe_ai::optim::{Lbfgs, LbfgsConfig, LbfgsLineSearch}` の再
エクスポート）を求めるが、着手時点（2026-09-26）で #2198・親 #2172・
ルート #2131 のいずれにも所有者の明示承認コメントがない
（`docs/compat-api-scope.md` §5 経路 2「facade 公開面の拡張は設計判断
記録 → 承認 → 実装の 2 段」規則）。前例（#2136 → PR #2247・#2084 →
PR #2252）と同様に、多層防御のガード（保留固定）＋ドキュメント整備
のみで `Closes` する。

**追加した保留ガード**（承認が得られるまで facade 公開を機械的に
拒否する多層防御。`OptimizerExtHoldDoctestGuard`〈#2171〉と同型）:

- 正のプローブ doctest: `crates/facade/src/lib.rs::
  LbfgsHoldDoctestGuard`
- ソース走査（型名の再エクスポート・独自宣言）: `crates/facade/tests/
  api_surface.rs::facade_does_not_reexport_or_declare_lbfgs_items`
- ソース走査（`compat::Optimizer` enum への variant 追加）:
  `crates/facade/tests/api_surface.rs::
  compat_optimizer_enum_has_no_lbfgs_variant`（`Lbfgs` は型名の
  再エクスポートを伴わない enum variant 追加のため、上記 2 者とは別の
  走査が必要——`LbfgsHoldDoctestGuard` の `__fandhe_lbfgs_variant_probe`
  モジュールが対応する正のプローブ）
- 部分的な受け入れ裏付け（`fandhe_ai_autodiff` 直接 import の手動
  closure ループ統合テスト）: `crates/facade/tests/
  compat_sequential_lbfgs_manual.rs`

**再開条件**: #2198・親 #2172・ルート #2131 のいずれかに所有者の明示
承認コメントが付いた時点で、§8 の実装設計に従って facade 公開・
`compile()` 統合を実装し、上記の否定ガードを対応する正ガードへ置き
換える。

## §8 承認後の実装設計（事前提示）

承認が得られた場合の実装形を、実装時の判断コストを下げるためあらかじめ
記す（実装時に §7 のガードを撤去してから適用する）。

- **公開面**: `crates/facade/src/optim.rs` に
  `pub use fandhe_ai_autodiff::nn::optim::{Lbfgs, LbfgsConfig,
  LbfgsLineSearch};` を追加する。`crates/facade/src/compat/
  training.rs::Optimizer`（`#[non_exhaustive]`）に `Lbfgs(LbfgsConfig)`
  を追加する（非破壊）。
- **`OptimizerState` の網羅 match 6 か所**: `Debug`／`new`（`Lbfgs::new`
  へ委譲）／`lr`（`config().lr`）／`set_lr`（`Lbfgs::set_lr` へ委譲。
  `supports_lr_schedule` は他 optimizer 同様 true）／`step`（closure
  経路のため到達しないが防御的に `InvalidArgument`）／
  `supports_lr_schedule` にそれぞれ 1 arm を追加する。
- **`run_fit` のバッチ処理**: `OptimizerState::Lbfgs` の場合のみ非公開
  ヘルパー `lbfgs_batch_step` へ分岐する。手順は
  `trainable_parameters()` の snapshot 化 → `Lbfgs::try_step_closure`
  （closure 内で `apply_parameters(trial)` → forward → loss →
  backward → `trainable_grads` の owned clone）→ `Err` の場合は必ず
  `apply_parameters(snapshot)` で復元してからエラーを返す（本 doc §7
  の保留経路テスト `compat_sequential_lbfgs_manual.rs::
  lbfgs_step_on_sequential` が同じ復元契約をあらかじめ固定している）。
  `Ok` の場合は `apply_parameters(updated)` し、`History::loss` へは
  `lbfgs.last_loss()`（初回評価損失。PyTorch `orig_loss` 相当）を記録
  する。それ以外の optimizer のバッチ処理はバイト単位で不変とする。
- **AMP 非対応**: `compile_with_amp(Optimizer::Lbfgs(_), …)` は
  `InvalidArgument` を返し `self.compiled` を変更しない
  （construct-before-assign 規約。PyTorch の `GradScaler` も closure 型
  `LBFGS` 非対応と同じ判断）。
- **Dropout／BatchNorm**: closure 評価のたびに RNG・running stats が
  更新される（PyTorch と同じ意味論）ため拒否しない。エラー時の復元は
  trainable params のみで running stats は復元しない（他 optimizer の
  失敗経路と同じ）。
- **テスト**: 新規 `crates/facade/tests/compat_sequential_fit_lbfgs.rs`
  （facade 再エクスポートのみを使う契約）に、`compile`/`fit` と手動
  ループ（同一 seed・同一 `LbfgsConfig`）の bit 完全一致・MNIST 形状の
  定性的収束・NaN 入力の fail-closed（params 復元・`is_compiled()`
  維持）・AMP 拒否・再 compile での他 optimizer との切り替え・
  `LrSchedule` 併用を置く。
- **ガードの反転**: §7 の否定ガード 3 種を削除し、`Lbfgs` variant の
  存在を固定する正ガードへ置き換える。`optim_module_reexports_
  exactly_expected_surface` の期待集合に `Lbfgs`／`LbfgsConfig`／
  `LbfgsLineSearch` を追加する。
