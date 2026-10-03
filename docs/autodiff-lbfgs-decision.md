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
面の拡張は設計判断記録 → 承認 → 実装の 2 段」規則）で扱う。着手時点
（2026-09-26）は承認未取得のため保留固定（多層ガード）で `Closes`
していたが、**2026-09-27 に所有者が facade 公開面拡張を承認**
（`compat::Optimizer::Lbfgs(LbfgsConfig)` variant・`LbfgsConfig` の
facade 再エクスポート・`compile()`/`fit()` 統合の 3 点。#2172 コメント）
し、実装済みである。`Lbfgs`（optimizer 本体）・`LbfgsLineSearch`
（line search 方式選択）も **2026-10-04 にルート #2499 の一括承認で
イシュー #2502 が facade（`fandhe_ai::optim`）へ公開済み**（§8 の形。
保留ガードは全撤去。§9 末尾「#2502 による残り 2 型の公開」参照）。保留の
詳細・再開条件は §7、承認前の実装設計（事前提示）は §8、承認後の
実装記録は §9 を参照。

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

- facade（`fandhe_ai::optim`）への `LbfgsConfig` 再エクスポート・
  `compat::Sequential::compile()` との統合は別イシュー **#2198**
  （facade 公開面拡張はユーザー承認事項）で 2026-09-27 に承認・実装
  済み（§9 参照）。`Lbfgs`（optimizer 本体）・`LbfgsLineSearch`
  （line search 方式選択）自体の facade 再エクスポートは #2198 では
  スコープ外だったが、#2502（2026-10-04）で公開済み（§9 末尾参照）
- `crate::optim::device_store::DeviceParamStore` 常駐経路への対応
- param groups（`nn::optim::param_group::ParamGroupStep`。#2173）への
  対応
- `maximize`・複素数パラメータ・sparse 勾配（§3 参照）

## §7 facade 公開・`compile()` 統合の保留（#2198。2026-09-26 時点の記録）

**2026-09-27 追記**: 本節は着手時点（2026-09-26）の保留記録であり
歴史的経緯として残す。2026-09-27 に所有者が `compat::Optimizer::
Lbfgs(LbfgsConfig)` variant・`LbfgsConfig` の facade 再エクスポート・
`compile()`/`fit()` 統合を承認し実装済みである（承認範囲は §9 が
提示する 3 点のみで、本節が示す「3 型すべての再エクスポート」より
狭い）。`Lbfgs`／`LbfgsLineSearch` は当時は非公開（2026-10-04 の #2502 で公開し
保留ガードは全撤去。以下は当時の記録。下記の保留ガードは
この 2 型のみを対象とする形へ縮小済み。`compat_optimizer_enum_has_no_
lbfgs_variant` は variant 追加自体が承認されたため
`compat_optimizer_enum_has_lbfgs_variant`〈正のガード〉へ反転した）。

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

## §9 承認後の実装記録（2026-09-27。イシュー #2172 コメント）

所有者が facade 公開面の拡張を明示承認した
（https://github.com/Fandhe-AI/fandhe-ai/issues/2172#issuecomment-5852299614）。
**承認範囲は次の 3 点のみ**であり、§8 が事前提示した「`Lbfgs`／
`LbfgsConfig`／`LbfgsLineSearch` の 3 型すべてを再エクスポートする」案
より狭い:

1. `compat::Optimizer::Lbfgs(LbfgsConfig)` variant の追加
2. `LbfgsConfig` 型のみの facade 再エクスポート（`fandhe_ai::optim`）
3. `compile()`/`fit()` への LBFGS 統合

`Lbfgs`（optimizer 本体。closure 駆動）・`LbfgsLineSearch`（line search
方式選択）は承認事項に含まれないため、引き続き非公開のまま
`LbfgsHoldDoctestGuard`（`crates/facade/src/lib.rs`。プローブを
`{Lbfgs, LbfgsLineSearch}` の 2 型のみへ縮小）＋`crates/facade/tests/
api_surface.rs` のソース走査（`facade_does_not_reexport_or_declare_
lbfgs_items`。`NAMES` から `LbfgsConfig` を除外）で固定する。
`compat::Optimizer` enum への variant 追加自体は承認済みとなったため、
旧`compat_optimizer_enum_has_no_lbfgs_variant`（否定ガード）は
`compat_optimizer_enum_has_lbfgs_variant`（variant がちょうど 1 個
存在することを固定する正のガード）へ反転した。

### 実装内容

- `crates/facade/src/optim.rs`: `pub use fandhe_ai_autodiff::nn::
  optim::LbfgsConfig;`（単一識別子・波括弧なし。rustfmt が `{X}` を
  `X` へ整形するため、`tests/api_surface.rs::
  optim_module_reexports_exactly_expected_surface` の走査を単一識別子
  形にも対応するよう拡張した）。
- `crates/facade/src/compat/training.rs`:
  - `Optimizer::Lbfgs(LbfgsConfig)` variant を追加（非破壊。
    `#[non_exhaustive]` enum への variant 追加）。
  - `OptimizerState::Lbfgs(Lbfgs)`（`Lbfgs` は `fandhe_ai_autodiff::nn::
    optim::Lbfgs` を crate 内部専用に直接 import。facade からは一切
    再エクスポートしない）。`new`／`lr`／`set_lr`（`Lbfgs::set_lr` へ
    委譲。`supports_lr_schedule` は `true`）／`step`（closure 経由の
    ため到達しない防御的 `InvalidArgument`）の 4 か所へ 1 arm ずつ
    追加。
  - `Sequential::run_fit` のバッチ処理: `OptimizerState::Lbfgs` を
    検出した場合のみ非公開ヘルパー `lbfgs_batch_step` へ分岐する。
    手順は §8 の事前設計どおり——`trainable_parameters()` の snapshot
    化 → `Lbfgs::try_step_closure`（closure 内で `apply_parameters
    (trial)` → `forward_with_precision`（AMP 非対応のため常に
    `None`）→ `T::loss_for` → `tape.backward` → `trainable_grads` の
    owned clone）→ `Err` なら `apply_parameters(snapshot)` で復元して
    からエラーを返す → `Ok` なら `apply_parameters(updated)` し
    `History::loss` へ `lbfgs.last_loss()`（初回評価損失）を記録する。
  - **AMP 非対応**: `compile_with_amp(Optimizer::Lbfgs(_), …)` は
    `OptimizerState::new`／`GradScaler::new` を呼ぶ前に
    `InvalidArgument` を返し `self.compiled` を変更しない
    （construct-before-assign）。
  - **非対応の組み合わせ（fail-closed 拒否。§8 の事前設計を踏襲）**:
    `FitConfig` の `accumulate_steps > 1`、カスタム学習 step フック
    （イシュー #2184。フックは `&Sequential`〈不変参照〉しか受け取ら
    ないため trial パラメータ書き込みを駆動できない）。両者とも
    `fit_with_callbacks_named` の引数検査（モード変更・バッチループ
    より前）で拒否する。
  - **未対応のまま残る組み合わせ**: `OptimizerStateDict`（#2304）・
    param groups（#2173）はいずれも facade 側で別途保留中（未承認）の
    ため、`Optimizer::Lbfgs` 固有の追加対応は行っていない（両者の保留
    解除時に横断的に対応する）。
- **facade のみで使う場合の既知の制約（#2502 で解消済み。当時の記録）**:
  `LbfgsLineSearch` を facade
  から名指しできないため、`fandhe_ai` のみに依存する利用者は
  `LbfgsConfig::line_search` を明示指定できず、既定の固定ステップ
  （`LbfgsLineSearch::None`）のみで L-BFGS を使うことになる。strong
  Wolfe line search が必要な場合は引き続き `fandhe_ai_autodiff` への
  直接依存が必要（`crates/facade/tests/compat_sequential_lbfgs_
  manual.rs`）。

### `lbfgs_batch_step` の失敗時復元契約のテスト（codex-review 指摘・PR #2319）

§8 の「`run_fit` のバッチ処理」節が予定する「`Err` の場合は必ず
`apply_parameters(snapshot)` で復元してからエラーを返す」契約は、
実装時点では `crates/facade/tests/compat_sequential_lbfgs_manual.rs`
（内部 import 契約ファイル。closure を独自に組んで `Lbfgs::
try_step_closure` を直接呼ぶ手動ループ）でのみ検証しており、facade の
`compile()`/`fit()` 経由（`lbfgs_batch_step`）の失敗時復元は未検証
だった（PR #2319 codex-review P2 指摘）。

`crates/facade/src/compat/training.rs::lbfgs_fit_failure_tests::
lbfgs_fit_restores_params_and_keeps_compiled_after_multi_eval_failure`
（crate 内部の `#[cfg(test)]`。`lbfgs_batch_step` が非公開のため
外部統合テストクレートからは到達不能）を追加し、次を固定した:

- 極端に大きい `lr`（`1e30`）・`max_iter: 2` により、固定ステップの
  1 回目の closure 評価（元パラメータ・有限）は成功し、`x += t·d`
  更新後の 2 回目の評価で MSE loss が `f32::MAX` を超えて `inf` になる
  （決定的に再現可能。乱数の偶然性に依存しない）。
- `fit` がこの `InvalidArgument`（"closure returned non-finite loss"）
  を返す。
- `Sequential::trainable_parameters()` が `fit` 呼び出し前の snapshot
  と bit 完全一致で復元される（closure が既に trial パラメータを
  書き込んだ後の失敗であることをエラーメッセージで確認済み）。
- `Sequential::is_compiled()` が維持される。
- 同じモデルに対する `evaluate` が失敗前と同一の損失を返す（パラメータ
  復元の間接確認）・再 `compile`（正常な `lr`）後の `fit` が成功する
  （compiled 状態・モデル状態が壊れていないことの確認）。

**§8 の予定と実装の食い違いの有無**: 上記検証の結果、§8 が記述する
復元契約（`Err` → `apply_parameters(snapshot)`）は実装と完全に一致して
おり、§8 側の記述を訂正する必要はなかった。

### 学習曲線検証（残る受入条件の充足）

`crates/facade/tests/compat_sequential_fit_lbfgs.rs::
fit_lbfgs_learning_curve_matches_or_beats_sgd` が、MNIST 実寸を使った
合成回帰（`D_IN=784`・`D_HIDDEN=16`・`D_OUT=10`・`N=32`・フルバッチ・
`Loss::Mse`。CI 実行時間〈debug ビルドで数十秒以内目安〉を考慮し
`N`／`D_HIDDEN`／`epochs`／`max_iter` を縮小する一方、`D_IN`／`D_OUT`
は MNIST の実寸を使う）で `compile()`/`fit()`（`epochs=3`）を通じ
L-BFGS（固定ステップ・`lr=0.2`・`max_iter=20`）と SGD（`lr=0.05`）を
同一初期重みから学習し、`evaluate()` の最終損失を比較する。実測値
（2026-09-27・debug ビルド・`cargo test -p fandhe-ai --test
compat_sequential_fit_lbfgs -- --nocapture`）:

| optimizer | history.loss（epoch 1〜3） | 最終 evaluate 損失 | 所要時間 |
|---|---|---|---|
| L-BFGS（固定ステップ） | `[0.3956, 0.0673, 0.0319]` | `0.0205` | 約 1.04s |
| SGD | `[0.3956, 0.3832, 0.3719]` | `0.362` | 約 4.3ms |

テストバイナリ全体（ビルド含む）の所要時間は約 10.5s（`time` 実測）で
あり、CI タイムアウト（`test-timeout-minutes: 20`）に対し十分な余裕が
ある。

L-BFGS は 1 epoch（= 1 outer step。フルバッチ）あたり `max_iter=20`
回までの内部反復を行うため、同じ `epochs` 数でも SGD より大きく損失が
下がる（PyTorch の一般的な L-BFGS 運用と同じ——`step` を epoch ごとに
1 回呼び `max_iter` を大きく取る）。所要時間が SGD よりかなり長いのも
同じ理由（内部反復あたり 1 回の forward／backward）で想定どおり。


### #2502 による残り 2 型の公開（2026-10-04）

- 承認根拠: ルート #2499 の 2026-10-04 一括承認（本 doc §7〜§9 の推奨形）。
- 公開した名前: `fandhe_ai::optim::{Lbfgs, LbfgsLineSearch}`。`crates/facade/src/optim.rs`
  は §8 の波括弧形 `pub use fandhe_ai_autodiff::nn::optim::{Lbfgs, LbfgsConfig, LbfgsLineSearch};`
  の 1 行（newtype・独自メソッドの追加なし）。`LbfgsLineSearch` は `#[non_exhaustive]`
  なので将来の variant 追加も非破壊。`FitConfig`・`fandhe-ai =0.10.0` の既存 API は不変。
- 削除したガード: `LbfgsHoldDoctestGuard`（`lib.rs`）、`api_surface.rs` の
  `lbfgs_hold_doctest_globs_all_pub_modules`・`lbfgs_hold_doctest_probe_body_matches_fixed_contract`・
  `LBFGS_HOLD_PROBE_BODY`・`scan_lbfgs_reexports_and_declarations`・
  `facade_does_not_reexport_or_declare_lbfgs_items`（および `_detects_each_category`）。
- 正ガード: `optim_module_reexports_exactly_expected_surface` の期待集合へ
  `Lbfgs`・`LbfgsLineSearch` を追加（承認した形だけを許す）、
  `lbfgs_types_are_reachable_via_facade_only`（facade のみの import で strong Wolfe の
  手動 closure ループを実行）を新設。`compat_optimizer_enum_has_lbfgs_variant` は維持。
- 利用例テスト: `compat_sequential_fit_lbfgs.rs::fit_lbfgs_strong_wolfe_via_facade_decreases_loss`
  （facade のみで `LbfgsLineSearch::StrongWolfe` を指定した `compile`/`fit`）と
  `optim.rs` モジュール doc の doctest。§9 の「既知の制約」は解消した。
- 帰結: `Lbfgs` の inherent `state_dict`／`load_state_dict`／`history_len`（§10）も
  `fandhe_ai::optim::Lbfgs` から到達可能になった。`OptimizerStateDict` trait は
  再エクスポートしておらず（#2555 の範囲）、`OptimizerStateDictHoldDoctestGuard` は不変。
  直接 `load_state_dict` を呼ぶ場合の履歴長上限（`MAX_LBFGS_HISTORY`）は `load_model`
  経路側で強制されるため、呼び出し元が渡す `HashMap` の大きさに資源消費は依存する。

## §10 状態保存・復元 API（イシュー #2366）

`Lbfgs` の大域状態を保存・復元する専用 inherent API を内部クレート
（`fandhe_ai_autodiff::nn::optim::Lbfgs`。追加時点は facade 非公開。
#2502 以降は `fandhe_ai::optim::Lbfgs` から inherent メソッドとして
到達可能だが、`OptimizerStateDict` trait の facade 公開は #2555 の範囲）に追加した（`crates/autodiff/src/nn/optim/lbfgs.rs`）。
`docs/compat-model-io-decision.md` §2 item 2・§4・§13.5 の内部 API 部分の実装。

### API

| メソッド | 役割 |
|---|---|
| `state_dict(&self) -> Result<HashMap<String, Tensor<f32>>, AutodiffError>` | 状態の書き出し |
| `load_state_dict(&mut self, state, slot_shapes: &[Vec<usize>], expected_history_len: usize)` | fail-closed な復元（`Err` 時 `self` 不変） |
| `history_len(&self) -> usize` | 現在の履歴件数（manifest の `history_len` 用） |

`OptimizerStateDict` のトレイト実装にしなかった理由: `slot_shapes`
（キー配置に含まれず構築済みモデルから導出）と `expected_history_len`
（manifest 由来）を受け取る余地がトレイトのシグネチャにないため。

### キー配置（接頭辞なし。`optimizer.` は facade 側 safetensors の名前空間で #2373 が付与）

| キー | 形 | 出現条件 |
|---|---|---|
| `n_iter.u64_u16x4`・`func_evals.u64_u16x4` | `[4]` | 常に |
| `t`・`h_diag` | `[1]` 生 f32 | 常に |
| `last_loss` | `[1]` | `func_evals >= 1` |
| `d`・`prev_flat_grad` | `[N]` | `n_iter >= 1` |
| `history.{i}.s`（`old_stps`）・`history.{i}.y`（`old_dirs`） | `[N]` | `i in 0..n` |
| `history.rho` | `[n]`（index 0 が最古） | `n >= 1` |

空ベクトルはキーごと省く（長さ 0 テンソルを作らず、#2373 の safetensors
往復で 0 要素テンソルを扱わないため）。

### 復元時の不変条件（到達可能状態から導出。すべて状態変更前に検査）

- `n <= config.history_size`（追い出し判定が `==` のため、超過状態を許すと履歴が無限に増える）
- `n >= 1` ならば `n <= n_iter - 1`（曲率ペアは 2 回目以降の反復でのみ push される）
- `n_iter >= 1` ならば `func_evals >= 1`（`last_loss` 必須）・`d`／`prev_flat_grad` あり
- 履歴件数は実在キーから導出（正規表記の添字のみ）し `expected_history_len` と照合。宣言値で確保しない
- shape は厳密一致、全 f32 値は有限、`N` は checked 演算
- 未実行（`func_evals == 0`）の復元は `slot_shapes` を空に戻し、次の step で params の shape を採用させる

### スコープ外

（#2373 で facade に実装済み）履歴件数の固定上限・manifest の `history_len` との突き合わせ・safetensors
結線・`optimizer.` 接頭辞の付与は別イシュー #2373。
