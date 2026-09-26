# param groups（層別学習率・weight decay）設計判断記録

イシュー #2173（親 #2131「PyTorch／TF 置き換えの API 網羅」）。
`docs/autodiff-loss-ops-decision.md`（#2166）・`docs/autodiff-rnn-stacked-config-decision.md`
（#2164）と同型の記録。

## §0 結論

PyTorch `torch.optim.Optimizer.param_groups` 相当の機能（パラメータ
集合をグループに分け、グループごとに独立した学習率・weight decay を
適用する）を、**`fandhe_ai_autodiff::nn::optim::{ParamGroup,
ParamGroupStep}`**（`crates/autodiff/src/nn/optim/param_group.rs`）
として内部クレート限定で実装した。既存 6 optimizer（`AdamW`・`Adam`・
`RmsProp`・`Adagrad`・`Lamb`・`crate::optim::Sgd`）へ `ParamGroupStep`
を実装し、各 optimizer の既存 `step()` は新設した `pub(crate)
step_with_slot_hparams`（スロット単位の `lr`／`weight_decay` を受け取る
実装本体）への薄い委譲へ変更した。演算式の形・演算順は変えていない
ため、`groups = &[]` は既存 `step()` と **bit 完全一致**する。

facade（`fandhe_ai::optim`）への再エクスポート・`compat::Sequential`
への `compile_with_param_groups` 等の新設は、親 #2131 の「facade 公開
面の拡張は設計判断記録 → 承認 → 実装の 2 段」規則により**未承認のため
保留**する（§5「承認事項」）。

## §1 背景

イシュー #2173 と親 #2131 のいずれにも所有者の承認コメントはない
（着手前に `gh issue view --json comments` で確認済み）。加えてイシュー
本文自身が「facade `compile()` シグネチャ変更」を承認事項に挙げている。

`crates/facade/src/optim.rs` は `Sgd`／`Adam`／`AdamW`／`RmsProp`／
`Adagrad`／`Lamb` を**名前指定で再エクスポート**している（glob では
ない）。このため、これらの型へ inherent の `pub fn` を 1 つでも足すと
その時点で facade 公開面が広がる。本実装は `ParamGroupStep` を**別
trait** として実装することで、facade がこれら型を再エクスポートして
いても `ParamGroupStep`（内部クレート限定）が facade 経由でスコープに
入らない構造にした（trait メソッドは trait が import されて初めて
`.method()` 呼び出しが可能になる Rust の名前解決規則を利用）。

並走中の兄弟イシュー #2170（`compile()` の `Optimizer` enum 拡張・
`training.rs` を編集）・#2174（optimizer state_dict・各 optimizer
ファイルを編集）はどちらも本実装時点で OPEN。各 optimizer ファイルの
差分は最小限の shim に留め、コンフリクト面を減らした。

## §2 設計

### 2.1 `ParamGroup`

```rust
#[non_exhaustive]
pub struct ParamGroup {
    pub params: Vec<usize>,   // スロット添字列
    pub lr: f32,
    pub weight_decay: f32,
}
```

`params` は Tensor 参照ではなく**スロット添字**（`usize`）。呼び出し元
が `step_with_groups` の `params`／`grads` に渡す列の位置（`Sequential::
trainable_parameters`／`SequentialVars::trainable_grads` と同じ「位置
対応契約」）を指す。`#[non_exhaustive]` により将来のフィールド追加
（momentum 等）が非破壊になる。`ParamGroup::new` は検証を行わない
（範囲・重複の検証には呼び出し元 optimizer の `n_slots` が必要なため、
`resolve_slot_hparams` へ集約した）。

### 2.2 既定スロットの扱い

どのグループにも属さないスロットは、optimizer の現在の config
（`config.lr`／`config.weight_decay`）をそのまま使う。これにより
`groups = &[]` は既存 `step()` と bit 完全一致する（`crates/autodiff/
tests/nn_optim_param_groups.rs` で固定）。`set_lr`（LR scheduler 結線用）
は従来どおり config（既定グループ）の `lr` のみを書き換え、グループの
`lr` は絶対値のまま `set_lr` の影響を受けない。

### 2.3 検証規則（fail-closed）

`resolve_slot_hparams(groups, n_slots, default_lr, default_wd, who)` が
一括して検証する。状態変更前に完了する:

- グループの `params` が空 → `InvalidArgument`
- スロット添字が範囲外 → `InvalidArgument`
- 同一添字が同一グループ内、またはグループ間で重複 →
  `InvalidArgument`（PyTorch の「some parameters appear in more than
  one parameter group」と同じ）
- `lr`／`weight_decay` が非有限、または負値 → `InvalidArgument`

各 `ParamGroupStep` 実装は `params.len() == grads.len()` の検証 →
`resolve_slot_hparams` → 状態変更を伴う `step_with_slot_hparams` の順で
呼ぶため、いずれの検証違反も optimizer の状態（`m`／`v`／velocity 等）
を変更する前に検出する。

### 2.4 式の形（bit 一致契約）

各 optimizer の `step_with_slot_hparams` は既存 `step()` のループ本体を
そのまま移設し、`lr`／`weight_decay` の読み出しだけをスロット単位
`hparams[i]` へ差し替えた（`f32::mul_add` 等の演算順は変えていない）。
`beta1`／`beta2`／`eps`／`momentum`／`dampening`／`nesterov`／
`lr_decay`／trust ratio（Lamb）等の optimizer 固有ハイパーパラメータは
グループで上書きしない共有値のまま。状態（`m`／`v`／velocity／
`step_count`）も単一のまま（イシューのスコープ外）。

Lamb は `step_size`・`weight_decay` に加え、trust ratio の式
（`t_i = fma(lr*wd, x_i, s_i)`・`f = lr*‖x‖/‖t‖`）内の `lr` 参照もすべて
スロット単位の `hp.lr` に差し替えた（モジュール doc「実装形」節参照）。

Adagrad は `clr = lr / (1 + (step-1)*lr_decay)` の `lr` のみスロット
単位（`hp.lr`）にし、`lr_decay` は共有値のまま。

## §3 PyTorch との差分

- 未所属スロットは既定グループ扱い（PyTorch は全パラメータのグループ
  列挙が必須）
- グループはスロット添字で指定する（Tensor 参照ではない）
- グループで上書きできるのは `lr`・`weight_decay` のみ（`beta`・
  `momentum` 等は不可）
- `set_lr` は既定グループのみに効く（PyTorch のスケジューラは全
  グループに適用）

## §4 数値一致

既定経路（`groups = &[]`）は既存 `step()` と bit 完全一致する
（`crates/autodiff/tests/nn_optim_param_groups.rs` の各 optimizer 向け
テストで固定）。本実装は `Tensor<f32>` を介したホスト側 optimizer
step のみを対象とし、GPU カーネル・`Op`／`BackendOps`／VJP の追加は
一切行っていないため、CUDA／Metal の parity 申し送りは不要
（`crate::optim::device_store::DeviceParamStore` 常駐経路への group
対応もスコープ外。§6 参照）。

## §5 承認事項（未承認のため保留）

facade（`fandhe_ai::optim`）公開面の拡張は次のいずれも未承認。
`crates/facade/src/lib.rs::ParamGroupsHoldDoctestGuard`（正のプローブ
doctest）・`crates/facade/tests/api_surface.rs` の 4 テスト
（`param_groups_hold_doctest_globs_all_pub_modules`・
`param_groups_hold_doctest_probe_body_matches_fixed_contract`・
`facade_does_not_reexport_or_declare_param_groups`・
`workspace_declares_param_group_fn_names_only_in_allowed_locations`）が
機械的に固定する:

1. `compat::Sequential::compile_with_param_groups(optimizer, loss,
   param_groups: &[ParamGroup])` の追加（既存 `compile()` は不変。
   `compile()` は `&[]` への委譲とする案）
2. `ParamGroup`（と必要なら `ParamGroupStep`）の `fandhe_ai::optim` への
   再エクスポート
3. 未決の設計論点: `fit_with_callbacks` の `LrSchedule` とグループ lr の
   結線。案 A はグループ lr を `current_lr / compile_lr` の比でスケール
   する（PyTorch の initial_lr 方式に近いが、`compile_lr == 0` の扱いが
   要る）。案 B は param_groups 指定時に `LrSchedule` を fail-closed で
   拒否する。`History::lr` の意味も含め承認時に決める
4. 層単位でスロット添字を得る公開ヘルパー（例:
   `Sequential::param_indices_of_layer`）の要否

承認を得た日が来たら、`ParamGroupsHoldDoctestGuard`・対応する 4 テスト
を削除し、正のガード（実際の再エクスポート・facade テスト）へ置き換
える。

## §6 スコープ外

- Tensor 以外のパラメータ型のグループ
- optimizer 内部状態（`m`／`v`／velocity・`step_count`）のグループ追従
- `crate::optim::device_store::DeviceParamStore` 常駐経路の group 対応
- `Optimizer` enum 拡張（#2170 の担当）
- optimizer state_dict へのグループ保存（#2174 の担当）
- 本イシューと並行して追加された `Adadelta`／`Adamax`／`NAdam`／`RAdam`
  （#2171）への `ParamGroupStep` 実装（`param_group.rs` の実装対象は
  上記 6 optimizer のみ）
- 追跡 Issue の起票は承認なしに行わない規約（`out-of-scope-tracking.md`）
  に従い、必要なら PR 上でユーザーへ提案する

## §7 検証コマンドと結果

- `cargo fmt --all --check` → pass
- `cargo clippy --workspace --all-targets --all-features -- -D warnings`
  → pass（warning 0 件）
- `cargo test -p fandhe-ai-autodiff --lib nn::optim` → 81 passed
  （既存 optimizer 単体テスト、無修正で green）
- `cargo test -p fandhe-ai-autodiff --lib optim::sgd` → 22 passed
- `cargo test -p fandhe-ai-autodiff --tests` → 全 green（PyTorch fixture
  テストを含む既存統合テスト、無修正で green）
- `cargo test -p fandhe-ai-autodiff --test nn_optim_param_groups` →
  17 passed（新規。bit 一致・拒否ケース・状態非変更を固定）
- `cargo test -p fandhe-ai --doc` → 31 passed（`ParamGroupsHoldDoctestGuard`
  の正のプローブを含む）
- `cargo test -p fandhe-ai --test api_surface` → 201 passed（新規 4 テスト
  を含む）
- `cargo test -p fandhe-ai --test compat_sequential_param_groups` →
  4 passed（R4。CPU 学習ループでの bit 一致・層別 lr 収束・層凍結）
- `git diff --stat` で `crates/facade/src/compat/training.rs`・
  `crates/facade/src/optim.rs`・`Cargo.toml`／`Cargo.lock`・
  `docs/spec` に差分がないことを確認済み
