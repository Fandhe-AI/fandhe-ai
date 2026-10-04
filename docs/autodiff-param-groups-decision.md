# param groups（層別学習率・weight decay）設計判断記録

イシュー #2173（親 #2131「PyTorch／TF 置き換えの API 網羅」）。
`docs/autodiff-loss-ops-decision.md`（#2166）・`docs/autodiff-rnn-stacked-config-decision.md`
（#2164）と同型の記録。

## §0 結論

PyTorch `torch.optim.Optimizer.param_groups` 相当の機能（パラメータ
集合をグループに分け、グループごとに独立した学習率・weight decay を
適用する）を、**`fandhe_ai_autodiff::nn::optim::{ParamGroup,
ParamGroupStep}`**（`crates/autodiff/src/nn/optim/param_group.rs`）
として内部クレート限定で実装した。対象は計 10 optimizer——`AdamW`・
`Adam`・`RmsProp`・`Adagrad`・`Lamb`・`crate::optim::Sgd`（イシュー
#2173）に加え、`Adadelta`・`Adamax`・`NAdam`・`RAdam`（イシュー #2298。
#2171 で追加された 4 種への横展開。§7「#2298 追補」参照）——で、
`ParamGroupStep` を実装した。各 optimizer の既存 `step()` は新設した
`pub(crate) step_with_slot_hparams`（スロット単位の `lr`／
`weight_decay` を受け取る実装本体）への薄い委譲へ変更した。演算式の
形・演算順は変えていないため、`groups = &[]` は既存 `step()` と
**bit 完全一致**する。

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

**イシュー #2298 で追加した 4 種**（`docs/autodiff-optimizer-adadelta-
adamax-nadam-radam-decision.md` 参照）の lr 置換箇所:

- **Adadelta**: `param -= lr * delta` の `lr` を `hp.lr` に。
  `weight_decay` は coupled（`grad += wd*param`）のまま `hp.weight_decay`
  を使う
- **Adamax**: `clr = lr / (1 - beta1^step)` の `lr` を `hp.lr` に。
  `clr` 自体は f64 計算のままスロットループ内（要素ループの外）で
  求める
- **NAdam**: lr 依存の `coef_grad = -lr*(1-mu)/(1-mu_product)`・
  `coef_exp_avg = -lr*mu_next/(1-mu_product_next)` の `lr` を `hp.lr`
  に。共有状態 `mu_product` はスロットに依存しないためループ外で
  1 回だけ更新する（複数スロットへ誤って繰り返し乗算しないことが
  bit 一致契約の前提）
- **RAdam**: 更新式 `bias_corrected_exp_avg * lr * adaptive_lr * rect`
  （左結合）・`bias_corrected_exp_avg * lr` の `lr` を `hp.lr` に。
  `bc1`／`bc2`／`rho_inf`／`rho_t`／`rect` は lr に依存しない共有計算の
  ままループ外で 1 回だけ求める

### 2.5 `decoupled_weight_decay`（NAdam・RAdam）の扱い（R3）

`NAdam`／`RAdam` は `decoupled_weight_decay`（decoupled `param *= 1 -
lr*wd` か coupled `grad += wd*param` かを切り替えるフラグ）を config に
持つ。この値は**optimizer の config が定める意味のまま**
`ParamGroup` の `weight_decay` に適用され、**グループ側では切り替え
ない**（`ParamGroup` は `lr`／`weight_decay` のみを保持する純データ型
のため）。PyTorch の `param_groups` は `decoupled_weight_decay` も
グループごとに持てるが、本実装ではグループ上書き対象外とする意図的な
差分である（§3 に追記）。

## §3 PyTorch との差分

- 未所属スロットは既定グループ扱い（PyTorch は全パラメータのグループ
  列挙が必須）
- グループはスロット添字で指定する（Tensor 参照ではない）
- グループで上書きできるのは `lr`・`weight_decay` のみ（`beta`・
  `momentum` 等は不可）
- `set_lr` は既定グループのみに効く（PyTorch のスケジューラは全
  グループに適用）
- `NAdam`／`RAdam` の `decoupled_weight_decay` はグループ上書き不可
  （optimizer の config が定める意味のまま適用する。PyTorch の
  `param_groups` はこのフラグもグループごとに持てる。イシュー
  #2298・§2.5 参照）

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
- `Lbfgs`（closure 型で PyTorch LBFGS も param groups 非対応のため
  対象外）
- `NAdam`／`RAdam` の `decoupled_weight_decay` のグループ上書き
  （§2.5・§3 参照。イシュー #2298 でも対象外のまま）
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

## §8 #2298 追補（`Adadelta`／`Adamax`／`NAdam`／`RAdam` への横展開）

イシュー #2298（親 #2131）で `param_group.rs` の対象を上記 6 種から
`Adadelta`・`Adamax`・`NAdam`・`RAdam`（#2171 で追加）を加えた計 10 種
へ拡張した。実装形は §2.4「式の形（bit 一致契約）」・§2.5
「`decoupled_weight_decay` の扱い」に追記済み。facade 非公開の判断
（§5）・スコープ外（§6）は不変のまま維持する。

検証コマンドと結果:

- `cargo fmt --all --check` → pass
- `cargo clippy -p fandhe-ai-autodiff --lib -- -D warnings` → pass
  （`cargo clippy --workspace ...` は `backend-cuda` の無関係な
  pre-existing dead-code lint が `main` ブランチでも同様に fail する
  環境依存の既知事象のため、本イシューの変更対象クレートへ範囲を
  絞って確認した）
- `cargo test -p fandhe-ai-autodiff --lib nn::optim` → 196 passed
  （新規 4 件の `slot_hparams_len_mismatch_is_rejected` を含む）
- `cargo test -p fandhe-ai-autodiff --test nn_optim_param_groups` →
  33 passed（新規 16 件。R1・R2・R4）
- `cargo test -p fandhe-ai-autodiff --test nn_optim_adadelta --test
  nn_optim_adamax --test nn_optim_nadam --test nn_optim_radam` →
  各 3 passed（R5。無修正で green）
- `cargo test -p fandhe-ai-autodiff` → 全 green（1435 passed 他）
- `cargo test -p fandhe-ai --test api_surface` → 273 passed
  （`workspace_declares_param_group_fn_names_only_in_allowed_locations`
  の期待集合を 10 impl・4 `step_with_slot_hparams` 追加へ更新）
- `cargo test -p fandhe-ai --doc` → 46 passed（`ParamGroupsHoldDoctestGuard`・
  `OptimizerExtHoldDoctestGuard` を含む）
- `cargo test -p fandhe-ai --test compat_sequential_param_groups` →
  4 passed（無修正で green）
- `RUSTDOCFLAGS="-D warnings" cargo doc --workspace --no-deps --locked`
  → pass（`pub(crate)` の `step_with_slot_hparams`／`SlotHparams` へは
  pub な `step()` doc から intra-doc link せず `//` コメント・
  バッククォート表記に留めたため private-intra-doc-link 違反なし）
- `git diff --stat` で `crates/facade/src/**`・`Cargo.toml`／
  `Cargo.lock`・`docs/spec`・既存 fixture テスト 4 本に差分がないこと
  を確認済み

## §9 #2552 facade 公開形の着手時判定と承認依頼

イシュー #2552（親 #2551・ルート #2499）の記録。調査基準は `origin/main`
`2e6a166b`。**本節は推奨案の記録と承認依頼であり、承認を取得したことを
意味しない。** ルート #2499 の一括承認は「決定記録に書かれた推奨形」に
しか及ばないため、推奨形が無い論点は実装せずに停止する（#2732・#2728
と同型）。§5 の本文・見出しは変更しない。

### 9.1 着手時判定（§5 に確定した推奨形は無い）

| §5 項目 | 記録の文言 | 状態 |
|---|---|---|
| 1. `compile_with_param_groups` | 「追加」「委譲とする**案**」。戻り値型・エラー型・AMP／`LrSchedule` との組み合わせは未記載 | 案の段階 |
| 2. 再エクスポート | `ParamGroupStep` は「**必要なら**」 | 未決 |
| 3. `LrSchedule` とグループ lr | 「**未決の設計論点**」。案 A／案 B 併記、`History::lr` の意味は「承認時に決める」 | 複数案 |
| 4. スロット添字ヘルパー | 「**要否**」 | 未決 |
| エラー型 | §5 に記載なし（§2.3 は内部の `AutodiffError::InvalidArgument`） | 記載なし |

記録済みの事実: エラー型は既存の `AutodiffError::InvalidArgument`、
配置候補は `fandhe_ai::optim` と `compat::Sequential`。以降は推奨案
（承認待ち）である。

### 9.2 推奨案（承認待ち）

1. **エントリ**: `compat::Sequential` の inherent method として
   `pub fn compile_with_param_groups(&mut self, optimizer: Optimizer, loss:
   Loss, param_groups: &[ParamGroup]) -> Result<(), AutodiffError>` を追加
   する。`Vec<ParamGroup>` は非公開の `Compiled` に保持する。既存
   `compile()` のシグネチャ・意味は変えない（内部で `&[]` へ委譲する場合は
   bit 完全一致テストを必須とする）。状態への代入は全構築成功後に行う
   （`compile_with_amp` と同じ construct-before-assign）。
2. **再エクスポート**: `fandhe_ai::optim` へ `ParamGroup` と
   `ParamGroupStep` の両方を素の `pub use` で出す（手動学習ループで
   `step_with_groups` を呼ぶには trait が必要。`optim.rs` の 1 文 1 行規約
   に従う）。公開後は `step_with_groups` のシグネチャが 0.10.x の間固定
   される。trait は sealed ではないため、メソッド追加は default 実装付き
   に限る。
3. **`LrSchedule`**: 案 B を推奨する。`param_groups` が空でないとき
   `Callback::LrSchedule` が渡されたら、`fit_with_callbacks`（`_named`）の
   引数検査で状態変更前に `InvalidArgument` で拒否する（`RmsProp`／
   `Adagrad`／`Lamb` と `LrSchedule` の併用拒否と同型）。`History::lr` は
   現行どおり既定グループの lr の意味のまま。案 A（比でスケール。
   `compile_lr == 0` の扱いが要る）は後続の opt-in に回す。拒否を後で受理
   に変えるのは非破壊方向の変更である。
4. **スロット添字ヘルパー**: 追加しない。`compat::Sequential::
   named_parameters()` は `trainable_parameters()` と同一順序（層順・層内
   weight → bias。`sequential.rs` の順序契約 doc、#1758 のテストで固定）
   なので、キー（`"{index}.{name}"`）の列挙位置がそのままスロット添字に
   なる。対応関係を `compile_with_param_groups` の doc に書くだけにする
   （新規 API ゼロ）。
5. **エラー型**: 既存の `AutodiffError::InvalidArgument` のみ。新 variant・
   新エラー型は追加しない。検証規則（params が空・範囲外・重複・非有限・
   負値）は §2.3 のまま。

### 9.3 記録に無かった追加論点（推奨はいずれも fail-closed）

- `Optimizer::Lbfgs` と空でない groups: `ParamGroupStep` 未実装のため
  compile 時に `InvalidArgument`。
- `compile_with_amp` との併用: 組み合わせ用 API は今回追加しない（必要なら
  別の承認事項）。
- カスタム学習 step フック・`accumulate_steps > 1` と groups: 実装時に
  `optimizer.step` を迂回する経路を確認し、`step_with_groups` 経由で
  対応できなければ拒否する。
- スロット添字の検証時期: step 時の既存検証（`resolve_slot_hparams`）で
  必ず行う。compile 時に先行検証する場合も `pub(crate)` は公開せず、件数
  検査を facade 側に置く。
- compile 経路が届くのは `Optimizer` の 6 種（Sgd・AdamW・Adam・RmsProp・
  Adagrad・Lamb）。`Adadelta`／`Adamax`／`NAdam`／`RAdam` は variant が無く
  手動ループ（項目 2 の trait）でのみ使える。
- `DeviceParamStore` 常駐経路は従来どおりスコープ外（§6）。

### 9.4 `fandhe-ai =0.10.0` 公開 API の非破壊確認（推奨案ごと）

- 項目 1: inherent `pub fn` の追加のみ。`compile`・`compile_with_amp`・
  `fit*` のシグネチャと意味は不変。`FitConfig`（`Copy + Eq`）・
  `History`／`Optimizer`（`#[non_exhaustive]`）は変更しない。
- 項目 2: `pub use` の追加のみ（minor 変更）。glob import 利用者の自前
  同名定義との衝突の可能性を注記する。`crates/facade/src` に
  `ParamGroup`／`ParamGroupStep` の既存公開定義は無いことを確認済み。
- 項目 3: groups 無しの既定経路の `LrSchedule` 挙動は不変。拒否されるのは
  新 API 使用時のみ。
- 項目 4: 追加しないため変更ゼロ。
- エラー型: 既存の型・variant のみ。

### 9.5 ユーザーに決めてほしい事項

- (a) 9.2 項目 1 のシグネチャ
- (b) 項目 2 で `ParamGroup` と `ParamGroupStep` を両方出すか
- (c) 項目 3 を案 B（拒否）とするか案 A とするか
- (d) 項目 4 でヘルパーを追加しない方針
- (e) 9.3 の fail-closed 方針

推奨案を一括承認するか、論点ごとに代替を選ぶ。承認されるまで #2553
（facade 公開の実装）・#2554（保留ガード反転）は着手不可（blocked）。

### 9.6 本イシューで行わないこと

- `crates/facade/**` の変更、保留ガード（`ParamGroupsHoldDoctestGuard`・
  `api_surface.rs` の 4 テスト）の撤去・反転
- `docs/compat-api-scope.md` §5 への適用記録（公開を適用していないため）
- 追跡 Issue の起票（ユーザー承認が必要）

## §10 #2553 着手時判定（§9 未承認のため停止）

調査基準は `origin/main` の `10658775`（2026-10-04 確認）。

### 10.1 判定

- §5 に確定した推奨形は無い（§9.1 の表を参照）。§9.2 の推奨案は承認待ちであり、
  §5 に書かれた承認済みの形ではない
- #2553・#2552・ルート #2499 の各イシューに、承認コメントは付いていない
  （`gh api` でコメント一覧を確認）
- ルート #2499 の一括承認が及ぶのは §5 に書かれた推奨形だけであり、§9.2 には及ばない

### 10.2 結論

停止条項に従い、facade 公開（`compile_with_param_groups` の新設・
`fandhe_ai::optim` への再エクスポート）は実装せずに停止した。

- 残る判断事項は §9.5 の (a)〜(e)（再掲しない）
- 引き続き blocked: #2553 の実装本体と #2554（保留ガードの反転）。承認後に
  #2553 を reopen するか新しい実装イシューを起票するかは、ユーザーが判断する

### 10.3 本イシューで行わないこと

- `crates/facade/**` の変更、保留ガード（`ParamGroupsHoldDoctestGuard`・
  `api_surface.rs` の 4 テスト）の撤去・反転
- `docs/compat-api-scope.md` §5 への適用記録（公開を適用していないため）
- 追跡 Issue の起票（ユーザー承認が必要）

## §11 #2554 着手時判定（#2553 未実装・§9 未承認のため停止）

調査基準は `origin/main` の `28384093`（2026-10-04 確認）。本節は停止の記録であり、
承認を取得したことを意味しない。

### 11.1 判定

- 依存の #2553 は PR #2740 で §10 を記録してクローズされた。facade 公開
  （`compat::Sequential::compile_with_param_groups`・`fandhe_ai::optim::{ParamGroup, ParamGroupStep}`
  の再エクスポート）は `crates/facade/` に存在しない
  （`compile_with_param_groups` は保留ガードの doctest 足場と doc コメントにのみ現れる）
- 正ガードへの反転は「承認して公開した形だけを許す」検査であり、公開物が無い状態では
  反転先が存在しない
- §9.5 の (a)〜(e) は未承認のまま。#2499・#2551・#2552・#2553・#2554 に承認コメントは
  付いていない（`gh api` でコメント一覧を確認）。ルート #2499 の一括承認は §9.2 に及ばない
  （§10.1 と同じ）

### 11.2 結論

停止条項に従い、`ParamGroupsHoldDoctestGuard` と `crates/facade/tests/api_surface.rs` の 4 テスト
（`param_groups_hold_doctest_globs_all_pub_modules`・
`param_groups_hold_doctest_probe_body_matches_fixed_contract`・
`facade_does_not_reexport_or_declare_param_groups`・
`workspace_declares_param_group_fn_names_only_in_allowed_locations`）は撤去も反転もせず現状維持とした。

- 解除の順序: §9.5 のユーザー承認 → #2553 の reopen または新しい実装イシューでの facade 公開
  → 保留ガードの反転（#2554 の受入基準）。承認だけでは解除されない
- #2554 の受入基準 3 点（ガード反転・`compat-api-scope.md` §5 の適用記録・facade 経由の利用例テスト）
  は未達であり、公開の実装後に行う

### 11.3 本イシューで行わないこと

- `crates/facade/**` の変更、保留ガードの撤去・反転
- `docs/compat-api-scope.md` §5 への適用記録
- facade 経由の利用例 doctest・テストの追加（対象 API が存在しないため）
- 追跡 Issue の起票（ユーザー承認が必要）
