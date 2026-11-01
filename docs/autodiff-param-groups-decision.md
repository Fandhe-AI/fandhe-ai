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

> 追記: 本段落の保留は §12（#2553）で公開済み・§13（#2554）で正ガードとして仕上げ済み。
> 経緯は §12・§13 を参照（本段落は当時の記録として残す）。

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

> 追記: 上記は #2553 で公開・#2554 で正ガード仕上げ済み。§12・§13 参照。

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

## §12 #2553 実装記録（facade 公開）

### 12.1 承認の根拠

ルート #2499 のコメント（issuecomment-6033824965。リポジトリ所有者アカウントの投稿）が、#2551 を
「§9 の推奨案（§9.5 (a)〜(e) は推奨どおり）」で承認している。本実装はその範囲に収める。

### 12.2 公開した名前（3 件）

- `fandhe_ai::optim::ParamGroup`・`fandhe_ai::optim::ParamGroupStep`（`optim.rs` の素の `pub use`。別名なし）
- `compat::Sequential::compile_with_param_groups(&mut self, Optimizer, Loss, &[ParamGroup])`
  （`compat/training.rs` の inherent メソッド。既存 `compile()` は不変）

`SlotHparams`・`resolve_slot_hparams`・`step_with_slot_hparams` は `pub(crate)` のまま公開していない。

### 12.3 実装判断

| 論点 | 扱い |
|---|---|
| groups の保持 | 非公開 `Compiled::param_groups`。全構築成功後に代入（失敗時は直前の compile 状態を残す）。`compile`／`compile_with_amp`／復元／functional の構築箇所は空 |
| step の振り分け | `OptimizerState::step_dispatch`。groups が空なら既存 `step`（`compile()` 経路と bit 一致）、非空なら `step_with_groups`。`run_fit` の 3 箇所と functional の 1 箇所をすべてこの入口へ置換（迂回なし） |
| `accumulate_steps > 1` | 累積境界の step も `step_dispatch` を通すため対応（拒否しない） |
| カスタム step フック | 内部フック型へ `&[ParamGroup]` を追加し、`TrainStepOptimizer`（非公開フィールド）が groups を持つ。公開シグネチャは不変（非破壊） |
| `Lbfgs` | 空でない groups は `OptimizerState::new` より前に拒否 |
| `LrSchedule` | groups 非空なら `fit` 系入口で `InvalidArgument`（案 B。状態変更前）。`History::lr` の意味は現行のまま |
| 検証の段階 | compile 時は groups の中身を検証しない（compile 後に層を足せるため）。fit 入口でスロット添字の範囲を検査し、step 時に `resolve_slot_hparams` が空・重複・非有限・負値を optimizer 状態の変更前に検出する |
| AMP | 併用 API は追加しない（`compile_with_amp` は groups を持たない） |

### 12.4 記録に明記の無い外挿（要判断事項）

`save_model` の manifest／`CompiledSnapshot` に groups の欄が無く、§6 は optimizer state_dict への groups
保存をスコープ外としている。groups を黙って落とすと復元後に学習率が変わるため、`snapshot_of_compiled` は
groups 非空を `InvalidArgument`（`ModelIoError::Autodiff`）で拒否する（保存先には何も作らない）。
保存形式の拡張は行っていない。groups の保存・復元を認めるか否かは別途判断が必要。

### 12.5 保留ガードの更新

- 削除: `ParamGroupsHoldDoctestGuard`（`lib.rs`）と、そのドリフト検査 2 件・固定文言定数
- 縮小: `facade_does_not_reexport_or_declare_param_groups` を承認形だけを許す
  `facade_param_groups_public_surface_matches_approved_form` へ（許可 3 件がちょうど 1 件ずつであることも検査）
- 更新: `workspace_declares_param_group_fn_names_only_in_allowed_locations` の期待集合、
  `optim_module_reexports_exactly_expected_surface` の期待識別子、プローブ検出数の下限（17 → 16）
- #2554 に残す: 正ガードとしての仕上げ（正の doctest プローブ）、`compat-api-scope.md` §5 の適用記録、
  §5 への実装記録、facade 経由の利用例の拡充

### 12.6 対象外

AMP 併用 API、`LrSchedule` の案 A、groups の保存形式、`DeviceParamStore` 常駐経路、`Optimizer` enum に無い 4 種
（Adadelta・Adamax・NAdam・RAdam）の `compile()` 経路。GPU カーネル・`Op`・`BackendOps` は触らないため
実機依存テスト・perf ログの申し送りは不要。

### 12.7 検証

`cargo test -p fandhe-ai`（`compat_sequential_param_groups` 14 件・`api_surface`・doctest）、
`cargo test -p fandhe-ai-autodiff --test nn_optim_param_groups`（無修正で green）。
tolerance・baseline・依存・`docs/spec` は変更していない。

## §13 #2554 実装記録（正ガードの仕上げ）

### 13.1 承認の扱い

公開形の承認根拠は §12.1（ルート #2499 のコメントによる §9 推奨案の承認）の範囲だけである。#2554 は公開面を
増やさない（テスト・コメント・docs のみ）ため、新たな承認は要らず、本節も新たな承認を主張しない。

### 13.2 公開した名前

§12.2 の 3 件（`optim::ParamGroup`・`optim::ParamGroupStep`・`Sequential::compile_with_param_groups`）から
追加なし。`crates/facade/src` の差分はコメントのみ。

### 13.3 ガード反転の対応表

| 旧（#2173 の保留ガード） | #2553 での処置 | #2554 での最終形 |
|---|---|---|
| `ParamGroupsHoldDoctestGuard`（`lib.rs`）とそのドリフト検査 2 件 | 削除 | 新設せず。正の doctest プローブ `param_groups_usage_doctests_are_present_and_compiled`（`api_surface.rs`）が、`optim.rs` のモジュール doc と `compile_with_param_groups` の doc にある利用例がコンパイル・実行される形であること（`ignore`／`no_run`／`compile_fail`／`text` なし）を固定する |
| `facade_does_not_reexport_or_declare_param_groups` | `facade_param_groups_public_surface_matches_approved_form`（最小形）へ縮小 | `facade_param_groups_public_surface_matches_approved_contract` へ改名。鍵 3 種がちょうど 1 件であることの検査を `param_groups_inventory_violations` へ分離（欠落・二重化・未知の鍵を別メッセージで報告） |
| `param_groups_resolve_from_facade_paths_only` | 承認形の到達性テストへ更新 | `param_groups_types_are_reachable_via_facade_only` へ改名。公開済み 10 optimizer すべてで `step_with_groups` のシグネチャを fn ポインタへ束縛 |
| `param_groups_scan_flags_unapproved_forms` | 承認形を許す自己テストへ更新 | `facade_param_groups_public_surface_guards_detect_each_category` へ改名・拡充（別名・別ファイル・`impl Sequential` 外・二重宣言・型宣言・インベントリ欠落／二重化／未知・形状逸脱の各カテゴリを正例で踏む） |
| （新設） | なし | `param_groups_types_match_approved_shape`: `ParamGroup`（`#[non_exhaustive]`・`derive(Debug, Clone, PartialEq)`・pub フィールド 3 件・inherent `new` のみ・手書き trait impl なし）と `ParamGroupStep`（メソッド `step_with_groups` 1 件のみ・supertrait／関連型なし）の形状を固定 |
| `workspace_declares_param_group_fn_names_only_in_allowed_locations` | 期待集合を更新 | 変更なし（doc コメントのみ） |

否定の保証に `compile_fail` は使っていない（stable rustdoc はエラーコードを照合しないため。正のプローブ＋
インベントリで組む）。

### 13.4 追加した利用例テスト

`crates/facade/tests/compat_sequential_param_groups.rs::named_parameters_position_is_slot_index_for_groups`
（15 件目）。3 層モデルで `named_parameters()` と `trainable_parameters()` の位置・値が一致することと、
接頭辞から導いた中間層のスロットだけを `lr = 0`・`weight_decay = 0` のグループへ入れると、その層だけが
bit 不変で他が更新されることを固定する（§9.2 項目 4: スロット添字の公開ヘルパーを足さない代わりの契約）。
判定は bit 一致で、tolerance は新設していない。利用例 doctest は #2553 で追加済みの `optim.rs`
モジュール doc と `compile_with_param_groups` の doc を使う。

### 13.5 対象外

§12.6 と同じ。AMP 併用 API、`LrSchedule` の案 A、groups の保存形式（§12.4 の要判断事項）、
`DeviceParamStore` 常駐経路、`Optimizer` enum に無い 4 種（Adadelta・Adamax・NAdam・RAdam）の
`compile()` 経路、`Lbfgs` 併用、スロット添字の公開ヘルパー。GPU カーネル・`Op`・`BackendOps` は
触らないため実機依存テスト・perf ログの申し送りは不要。

### 13.6 検証

`cargo test -p fandhe-ai --test api_surface`・`--test compat_sequential_param_groups`（15 件）・`--doc`、
`cargo test -p fandhe-ai-autodiff --test nn_optim_param_groups`（無修正）、`cargo test --workspace`、
`RUSTDOCFLAGS="-D warnings" cargo doc --workspace --no-deps --locked`。再エクスポートへ別名を一時的に
足す変異で `facade_param_groups_public_surface_matches_approved_contract` が失敗することを手元で確認した。
`Cargo.toml`／`Cargo.lock`・tolerance・baseline・`docs/spec` は変更していない。

## §15 保守・運用チェックリスト（イシュー #3754）: param groups と `LrSchedule` の併用（A3-b）

本節は**設計記録であって承認ではない**。承認は #3769 の論点 4 の回答を待つ（実装は #3834）。実際に得ていない承認は
書かない。コードの引用は `origin/main` の `2b82c8740` のもの。公開形・推奨・戻す論点の書式は
`docs/decision-record-template.md` に従う。なお追跡表 `docs/compat-matrix-ari-tracker.md` の行番号は 20
（A3-b）で、issue 本文や #3834 が使う通し番号「項目 21」（ルート／#3769 側）とは 1 ずれる。

### 15.0 位置づけ・課題・推奨

- 現状: `compile_with_param_groups` で空でない groups を設定すると、`fit_with_callbacks` 系の入口が
  `Callback::LrSchedule` を `InvalidArgument` で拒否する（`crates/facade/src/compat/training.rs:2966-2980` の
  (2.06)。§9.2 項目 3 の案 B、§12.3 で実装）。層別学習率とスケジュールを同時には使えない。
  棚卸しでは `docs/compat-unlisted-holds-inventory.md:48`（A3-b）が保留、§12.6・§13.5 は対象外（案 A）。
- 拒否の理由: `LrSchedule::lr_for_epoch_begin`（`crates/facade/src/compat/callbacks.rs:693`）は**絶対値の lr**
  を返し、epoch 開始時に `optimizer.set_lr` が config の lr を書き換える（`training.rs:3273-3297`）。
  一方 `ParamGroup.lr` は所属スロットの config の lr を**置き換える**ので、拒否を外しただけでは group の
  スロットはスケジュールに追従せず、黙って意味が変わる。
- `LrScheduler` trait は `lr_at(step) -> f32` だけで（`crates/autodiff/src/nn/optim/lr_scheduler.rs:71-75`）、
  基準 lr を取り出す accessor がない。
- (2.05)（`training.rs:2947-2963`）は (2.06) より前に走り、`set_lr` を持たない RmsProp・Adagrad・Lamb
  （`training.rs:1259-1276`）と `LrSchedule` の組み合わせは groups の有無に関係なく拒否される。
- optimizer config は `lr >= 0` を受け付ける（例 `crates/autodiff/src/nn/optim/adam.rs:120`）ので、
  compile 時の lr が 0 になることは起こりうる。

| 案 | 内容 | 利点 | 欠点・影響 |
|---|---|---|---|
| A（推奨） | 全 group 共通の倍率。epoch 開始時の有効 lr を `group.lr × (現在の optimizer lr ／ base_lr)` にする。`base_lr` は compile 時の optimizer config の lr の控え（PyTorch の `initial_lr` に近い） | 新しい公開名がない（拒否を受理に変える挙動変更のみ）。乗法型スケジュール（StepLr・MultiStepLr・ExponentialLr・LinearWarmupLr・LambdaLr 等）で group ごとの比が保たれる | 加法項を持つスケジュール（`eta_min > 0` の `CosineAnnealingLr`・`CyclicLr`・`OneCycleLr`・`ReduceLrOnPlateau` の下限）は PyTorch の group ごとの評価と値が一致しない（公式ドキュメントの式による推測。ソースとの照合は未確認）。`base_lr == 0` は扱いが要る |
| B | group ごとのスケジュール（`Callback` の新 variant、`Vec<LrSchedule>` 等） | 加法型も PyTorch と同じ形で書ける | 新しい公開形が要り `Callback` の形も変わる。承認範囲が広い |
| C | 拒否を恒久の契約にする（対象外） | 変更なし | 層別 lr とスケジュールを同時に使えないまま残る |
| D | 拒否を外すだけ（group は固定 lr のまま） | 実装が最小 | group のスロットだけがスケジュールを無視し、黙って意味が変わるので不採用 |

**推奨は A（実装する）**。理由: 公開面を増やさない。§9.2 項目 3 が後続 opt-in に回した案 A の具体化で、
拒否を受理に変えるのは非破壊の方向（§9.2）。設計の詳細は次のとおり。

- **基準 lr**: compile 時の optimizer config の lr を非公開の `Compiled` に控える。`optimizer.lr()` は
  epoch ごとに上書きされるので基準にできない。`LrScheduler` への `base_lr()` 追加は退ける（既定実装のない
  メソッド追加は破壊的〈`docs/api-compat-0.14.0-decision.md` §8 の 2〉。既定実装つきでも自前スケジューラでは
  値が取れない）。初回 epoch のスケジュール値を基準にする案も退ける（warmup 系で倍率が崩れる）。
- **式と精度**: `((group.lr as f64) * (cur as f64) / (base as f64)) as f32`。f64 で計算し f32 への downcast は
  最後の 1 回だけ。`cur.to_bits() == base.to_bits()` のときは倍率計算を通さず groups をそのまま使い、
  既定経路の bit 一致を確実にする。
- **weight_decay** には倍率を掛けない（PyTorch のスケジューラも lr だけを書き換える）。`lr = 0` の group
  （凍結）は 0 のままで凍結が保たれる。
- **適用範囲**: epoch 開始時、LR 同期の後に有効 groups を 1 回組み立て、その epoch の全 step 経路
  （`step_dispatch` の各呼び出し〈`training.rs:3350`・`3505`・`3605`・`3662` 付近〉、`accumulate_steps > 1`
  の累積境界、`fit_with_train_step` のフック〈`training.rs:2509-2511` の doc の「compile 時の groups」を
  「有効 groups」へ改める〉）に同じものを渡す。組み立て方（`Cow` 等）は #3834 が決める。
- **複数の `LrSchedule`** は現行どおり後に並んだものが勝つ。倍率は同期後の `optimizer.lr()` から求めるので
  矛盾しない。
- **スケジュールつき fit の後の素の `fit()`**: group 外スロットの lr は持ち越される（現行どおり）。
  group の lr も同じ倍率を持ち越す（PyTorch で group の lr が書き換わったまま残るのと同じ。論点 5）。
- **`History::lr`** の意味は変えない（同期後の optimizer の lr）。doc には
  `group の lr = group.lr × history.lr[e] ／ base_lr` と書く。group ごとの記録欄は足さない（論点 7）。
- **`set_lr` を持たない 3 種**は (2.05) の拒否と順序を保つ。「全スロットが group に入っていれば受理」は
  退ける（`History::lr` の意味が定まらず、`set_lr` の追加は公開面の拡張で別承認になる）。
- **`Lbfgs`** は空でない groups を compile 時に拒否済み（`training.rs:2001-2004`）。**SWA** と groups は
  別条件で拒否済み（`training.rs:2861-2863`）。どちらも変更しない。

### 15.1 公開形

- シグネチャ案: **新しい公開名はない**。`compile_with_param_groups`・`ParamGroup`・`ParamGroupStep`・
  `LrSchedule`・`Callback`・`History` の形は変えない。変わるのは、groups が空でないときに
  `Callback::LrSchedule` を受理する挙動だけ。
- facade 内で触る箇所（#3834）: `compat/training.rs` の `Compiled` に非公開の基準 lr の控えを足す
  （`training.rs:1408-1411` の隣）／(2.06) の拒否を 15.3 の条件へ置き換える／epoch 開始時に有効 groups を組み立てる。
- 反転する対象: `api_surface.rs` に A3-b 専用の否定ガードや保留 doctest ガードは**ない**。拒否を固定している
  のは挙動テスト `crates/facade/tests/compat_sequential_param_groups.rs::lr_schedule_is_rejected_only_when_groups_present`
  （T4、`:441`）。#3834 で反転・更新するもの: T4（正のテストへ。名前も変える）、(2.06) のブロックとコメント、
  `compile_with_param_groups` の doc の「制約」箇条書き（`training.rs:1934-1935`）、`fit_with_callbacks` の doc の
  拒否一覧（`training.rs:2263` 付近）、`OptimizerState::set_lr` の doc、`docs/compat-api-scope.md` §5 の
  「`LrSchedule` 併用…fail-closed 拒否」（`:1827`）と対象外一覧（`:1842`）。
- 残す正ガード: `facade_param_groups_public_surface_matches_approved_contract`・`PARAM_GROUPS_APPROVED_KEYS`
  （3 件のまま）・`param_groups_types_match_approved_shape`・`param_groups_types_are_reachable_via_facade_only`・
  `workspace_declares_param_group_fn_names_only_in_allowed_locations` は変更せず、公開名が増えないことを固定し続ける。
- 自己テスト: 手書きループ（`step_with_groups` に倍率をかけた groups を渡す）との bit 一致を正のテストとして置く。

### 15.2 rustdoc と doctest の例

- `compile_with_param_groups` の doc に「学習率スケジュールとの関係」節を足す: 倍率の式、基準 lr が compile 時の
  config の lr であること、weight_decay に掛けないこと、加法型は PyTorch と一致しない可能性（未確認）、
  `History::lr` の意味。
- doctest 案: CPU で動く facade 単体の例。`StepLr` と 2 つの group を使い、1 epoch 後に group の lr の比が
  保たれることを見る。`ignore`／`no_run` は不要。既存の正プローブ `param_groups_usage_doctests_are_present_and_compiled`
  が、対象 doc の例がコンパイル・実行される形のままであることを確かめる。
- 失敗系は `compile_fail` に頼らず正のテストで確かめる（stable rustdoc はエラーコードを照合しない）。

### 15.3 エラー型

既存の `AutodiffError::InvalidArgument` だけを使い、新 variant は足さない。本番経路で `unwrap`／`expect` は使わない。
拒否は `fit_with_callbacks` 系の入口で、状態を変える前にこの順で評価する。

1. (2.05) を維持: `set_lr` を持たない optimizer と `LrSchedule` は groups の有無を問わず拒否。
2. 新 (2.06a): groups が空でなく `LrSchedule` があり、基準 lr の控えが 0 または非有限なら拒否。
   compile 時の拒否にはしない（スケジュールなしで lr = 0 を compile するのは現行で合法で、拒否すると破壊的）。
3. (2.06b): 既存のスロット範囲の検査はそのまま残す。

epoch 開始時の有効 lr が非有限（overflow）になる場合は、optimizer 状態を変える前に `InvalidArgument` で止める
（fail-closed）。step 時の `resolve_slot_hparams` の非有限検査は二重の防御として残す。拒否時は `self.compiled`
を入口の前の状態に戻す（現行の (2.05)／(2.06) と同じ）。

### 15.4 テスト計画

置き場は `crates/facade/tests/compat_sequential_param_groups.rs`（#3834）。GPU カーネル・`Op`・`BackendOps` に
触らないため実機の列はすべて該当なし。

| 受け入れ基準 | テスト（案） | 区分 | `<cuda-node>` | `<metal-node>` | `<rocm-node>` |
|---|---|---|---|---|---|
| groups 空 + スケジュールが現行と bit 一致 | 既存（T4 の後半）を維持 | CI | 該当なし | 該当なし | 該当なし |
| groups あり + スケジュールなしが現行と bit 一致（倍率を通さない分岐） | 既存テストを無変更で通す | CI | 該当なし | 該当なし | 該当なし |
| groups + `StepLr` が手書きループと bit 一致（Sgd・AdamW） | 新規（T4 を反転） | CI | 該当なし | 該当なし | 該当なし |
| `ReduceLrOnPlateau`（`LrSchedule::plateau`）でも bit 一致 | 新規 | CI | 該当なし | 該当なし | 該当なし |
| `lr = 0` の group はスケジュールがあっても bit 不変 | 新規 | CI | 該当なし | 該当なし | 該当なし |
| 基準 lr が 0 で groups + スケジュールを拒否し、パラメータも compile 状態も不変 | 新規 | CI | 該当なし | 該当なし | 該当なし |
| RmsProp・Adagrad・Lamb + groups + スケジュールが (2.05) で拒否のまま | 新規 | CI | 該当なし | 該当なし | 該当なし |
| `accumulate_steps > 1`・`fit_with_train_step` のフックにも有効 groups が渡る | 新規 | CI | 該当なし | 該当なし | 該当なし |
| スケジュールつき fit の後の素の `fit()` で倍率が持ち越される | 新規 | CI | 該当なし | 該当なし | 該当なし |
| 公開面が増えない | `cargo test -p fandhe-ai --test api_surface` を無変更で通す | CI | 該当なし | 該当なし | 該当なし |

Phase 10 で測る実機テストはない（追跡表の実機 parity 列は「該当 issue なし」のまま）。

### 15.5 数値契約

新しい GPU 演算はない。判定はすべて bit 一致で、新しい tolerance は置かない。REQ-2 の統一複合判定・
`RELATIVE_TOLERANCE`／`ABSOLUTE_RESCUE_THRESHOLD`・parity baseline・ガードレール閾値は変えない。FMA 契約・
f64 長軸縮約の契約にも影響しない（倍率はホスト側スカラー計算で、f64 で計算し 1 回 downcast する）。

### 15.6 互換への影響

判定は**追加のみ**（手順の正は `docs/api-compat-0.14.0-decision.md` §8、表は同 §4 の行 20 と `:91`）。
公開型・シグネチャ・derive・trait メソッドは変えず、変わるのは groups があるときの `Err` を `Ok` にする
受理方向の挙動だけ（§9.2）。groups 空の既定経路と、groups ありスケジュールなしの経路は bit 不変。
移行ノート（#3780）には「挙動の追加: groups ありで `LrSchedule` を受理する」とだけ書く想定（実装 PR が書く）。

### 15.7 適用記録・索引・追跡表

現時点の公開状態は未公開（拒否のまま）。適用記録の案（#3834 が `docs/compat-api-scope.md` §5 へ書く。承認 URL は
#3769 の回答コメントを待って埋める）: 「param groups 設定時の `Callback::LrSchedule` を全 group 共通の倍率
（group.lr × 現在 lr ／ compile 時の lr）で受理するよう変更。新しい公開名なし。公開状態: 公開済み（挙動のみ）」。

- [x] `docs/README.md` の索引を更新した
- [x] 追跡表（行 20）を更新した
- [ ] `docs/compat-api-scope.md` の適用記録（実装 PR が行う）

### 15.8 ベンチと性能への影響

既定経路（groups 空、または groups ありでスケジュールなし）は倍率を計算しない分岐を通り、コストは基準 lr の
bit 比較だけ。スケジュールありでも組み立ては epoch あたり 1 回で、groups の複製は O(スロット数)。前後比較は
不要と判断し、計測していないので数値は書かない。

### 15.9 依存の追加の有無

依存の追加はなく、新しい `unsafe` もない。許容 12 区分・禁止リストとの照合結果は変わらず、
`docs/license-matrix.md` の更新も不要。

### 15.10 所有者へ戻す論点（#3769 論点 4）

| 点 | 選択肢 | 推奨 | 確認がないときの扱い |
|---|---|---|---|
| 1. 適用の単位 | A 全 group 共通の倍率／B group ごとのスケジュール／C 拒否を恒久化（対象外） | A | 実装へ進めない（(2.06) の拒否と T4 を維持） |
| 2. 基準 lr | compile 時の config の lr の控え／初回 epoch のスケジュール値／`LrScheduler` の trait 拡張 | compile 時の控え | 実装へ進めない |
| 3. 基準 lr が 0 のとき | fit 入口で拒否（スケジュールがあるときだけ）／倍率 1 として groups を固定 | 拒否 | 実装へ進めない |
| 4. `set_lr` を持たない 3 種 | 拒否を維持／全スロットが group に入っていれば受理 | 拒否を維持 | 実装へ進めない |
| 5. スケジュール後の素の `fit()` | 倍率を持ち越す／groups を compile 時の lr に戻す | 持ち越す（group 外スロットと同じ） | 実装へ進めない |
| 6. AMP の入口（A3-a。#3914 の §14.10 点 4）との関係 | A3-b の受理は `compile_with_param_groups` に限り AMP の入口は A3-a の決定に従う／AMP の入口にも同じ受理を及ぼす | 前者（A3-a 点 4 が決まるまで AMP の入口では拒否を維持。後に入る実装 PR が条件とテストを足す） | 実装へ進めない |
| 7. `History` に group ごとの lr を記録するか | 足さない／`#[non_exhaustive]` の `History` にフィールドを追加 | 足さない（式で導ける） | 実装へ進めない |

### 15.11 セキュリティ・スコープ外・Closes／Refs

- A03: 外部入力の新しいパースはない。検査は状態変更の前に済ませ、非有限の lr は fail-closed で止める（15.3）。
- A08: ガードレール閾値・tolerance・baseline は変えず、自己修復ループの判定経路にも触れない。秘密情報・ホスト名は書かない。
- スコープ外: A3-a（#3752・#3832）、A3-c（#3755）、A3-d（#3756）、A3-e（Lbfgs）、`Optimizer` に入っていない
  4 種（#3757）、SWA と groups の組み合わせ、`set_lr` を持たない 3 種への `set_lr` 追加、group ごとの
  スケジュール（案 B）。新しい issue は起票しない（起票には承認が要る）。
- 本 PR は設計記録のみで `Closes #3754`。`Refs #3769`（承認待ち）・`Refs #3834`（実装）。

## §18 保守・運用チェックリスト（イシュー #3755）: param groups の保存形式（A3-c）

本節は**設計記録であって承認ではない**。承認は #3769 の論点 3 の回答を待つ（実装は書き出し側が #3835、読み込み側と互換検査が #3836）。
実際に得ていない承認は書かない。コードの引用は `origin/main` の `78d6b805d` のもの。公開形・推奨・戻す論点の書式は
`docs/decision-record-template.md` に従う。なお追跡表 `docs/compat-matrix-ari-tracker.md` の行番号は 21（A3-c）で、
issue 本文が使う「項目 22」とは 1 ずれる。

### 18.0 位置づけ・課題・推奨

- 現状: `compile_with_param_groups` で groups を設定した compile 状態は、`save_model` が `InvalidArgument`
  （`ModelIoError::Autodiff`）で拒否する（`crates/facade/src/compat/training.rs:1495-1497` の `snapshot_of_compiled`。
  §12.4 が要判断事項として残した外挿）。復元側の `compiled_from_snapshot` は groups を常に空で作る
  （`training.rs:1579`）。groups を黙って落とすと復元後に学習率が変わるため、今は拒否している。
- 保存先: autodiff の optimizer は groups を持たない（`ParamGroupStep::step_with_groups` が呼び出しごとに受け取る）。
  `OptimizerStateDict` 側に保存対象はなく、groups の保持先は facade の非公開 `Compiled::param_groups`
  （`training.rs:1411`）である。よって保存先は `save_model`／`load_model` の manifest の `compiled` 節になる。
- manifest の制約（`crates/facade/src/compat/model_io.rs`・`model_io/compiled.rs`）: `compiled` は
  `{"loss","optimizer":{"kind","config"[,"history_len"]},"optimizer_state_keys","amp"}` で、`parse_compiled` は
  `exact_fields` でキー集合の完全一致を要求する（未知キーは `ModelIoError::Manifest`）。`MAX_JSON_DEPTH = 4`
  （root → compiled → optimizer → config で使い切る）、`MAX_ARRAY_LEN = 2 * MAX_LAYERS`、`MAX_OBJECT_KEYS = 16`。
  いずれも承認済みの値で、変えるなら再承認が要る。`optimizer_state_keys` を平坦な配列にした理由も同じ深さ制約
  （`docs/compat-model-io-decision.md` §4）。
- 版数の先例: lbfgs の `history_len` 追加（#2373）、`compiled.loss` の allowlist 拡張（#2509）、以降の kind 追加は
  すべて `format_version` を 2 のまま据え置いた（Functional 形式は 1）。旧リーダーは未知キー・未知値を fail-closed で拒否する。
- Functional モデルには groups の入口がなく、`param_groups` は常に空。`functional_io.rs` は `render_compiled`／
  `parse_compiled` を共有する。
- 検証規則の正本 `resolve_slot_hparams`（`crates/autodiff/src/nn/optim/param_group.rs:136`）は `pub(crate)` で、
  facade からは呼べない。`ParamGroup::new(params, lr, weight_decay)` は検証しない。

**推奨は「実装する」**。理由: 拒否のままだと groups を使う学習を `save_model` で中断・再開できない（PyTorch の
`optimizer.state_dict()` は `param_groups` を含む）。公開名は増えず、既存の `save_model`／`load_model` が「拒否」から
「保存・復元」に変わるだけで、groups なしの出力は今とバイト単位で同じにできる。

| 案 | 内容 | 利点 | 欠点・影響 |
|---|---|---|---|
| A（推奨） | `compiled` に任意キー `param_groups` を 1 つ足す。値は CSR 形の object `{"lr":[f32…],"weight_decay":[f32…],"offsets":[0,…,N],"slots":[usize…]}`（group `g` のスロットは `slots[offsets[g]..offsets[g+1]]`）。**groups が空ならキー自体を書かない** | 深さは root(1) → compiled(2) → param_groups(3) → 配列(4) で上限内。group の順序と group 内 `params` の順序を保つので `ParamGroup` の `PartialEq` で往復を比べられる。groups なしの出力は今と同じバイト列 | 配列長の整合検査（`lr`・`weight_decay` は G 件、`offsets` は G+1 件）が要る |
| A' | `compiled` 直下に `param_groups: [{"lr","weight_decay"}]`・`param_group_slots`・`param_group_offsets` を並べる | A と同じ情報量 | キーが 3 つ増え、有無の整合検査（全部あるか全部ないか）が増える |
| B | スロット → group の写像 `param_group_of_slot: [idx\|null]` | 重複検査が要らない | group 内 `params` の順序が失われ、復元後の `ParamGroup` が昇順に並び替わる（`PartialEq` が往復で成り立たない） |
| C | safetensors の `optimizer.` 名前空間にテンソルで保存 | JSON を触らない | facade の safetensors 経路は f32 のみで、添字が 2^24 を超えると正確に表せない。optimizer の `load_state_dict` は余剰キーを拒否し、名前空間の衝突回避も別に要る |
| D（対象外） | 保存の拒否を恒久契約にする | 変更なし | groups を使う学習を `save_model` で再開できないまま残る |

- 数値の表現は既存の `config` と同じで、f32 は `{:?}` の正準形（`-0.0` も往復でビットが一致する）。スロット添字は JSON の非負整数。
- `compiled` のキー数は 4 → 5 で `MAX_OBJECT_KEYS` 内。`slots` の長さは重複禁止によりスロット数以下で、スロット数は
  `parameter_keys` が `MAX_ARRAY_LEN` で既に抑えている。上限定数は変えない。

版数は次の 3 案から選ぶ。

| 案 | 内容 | 評価 |
|---|---|---|
| (i)（推奨） | Sequential は `format_version` 2、Functional は 1 のまま据え置く | lbfgs（#2373）・loss 拡張（#2509）と同じ先例。groups なしのファイルは旧リーダーでも今どおり読める。groups を含む新ファイルは旧リーダーが `exact_fields` の未知キーとして `Manifest` で fail-closed に拒否する |
| (ii) | groups を含むときだけ `format_version` 3 を書く | 旧リーダーのエラーが「版が未対応」で分かりやすい。ただし書き手が版を 2 種類出し、`verify_round_trip` と版ごとの受理表が複雑になる |
| (iii) | 常に 3 へ上げる | groups を使わないファイルまで旧リーダーで読めなくなる（後退）。推奨しない |

### 18.1 公開形

- 公開名の追加はない。`save_model`／`load_model`／`ModelIoError` のシグネチャは不変。変わるのは挙動（拒否 → 保存・復元）と保存形式。
- #3835／#3836 のコード変更点:
  - `CompiledSnapshot`・`CompiledMeta`（非公開）に `param_groups: Vec<ParamGroup>` を足す
  - `snapshot_of_compiled`: 拒否をやめて groups を写す
  - `render_compiled`: groups が空でないときだけ `,"param_groups":{…}` を出す
  - `parse_compiled`: 有無で分けてからキー集合を照合する（lbfgs の `history_len` と同じ書き方）
  - `compiled_from_snapshot`: groups を復元する（「常に空」のコメントを更新）
  - `load_from_dir_with_limits`・`prepare_save`: スロット範囲の照合を足す
  - doc 更新: `compile_with_param_groups` の「# 制約」にある「groups が空でない compile 状態は `save_model` で保存できない」と、`optim.rs` のモジュール doc の同趣旨の記述
- ガード: `crates/facade/tests/api_surface.rs` に、この拒否を固定する否定ガード・保留 doctest ガードは**ない**。
  param groups の正ガード群（`facade_param_groups_public_surface_matches_approved_contract`・`param_groups_types_match_approved_shape`・
  `param_groups_types_are_reachable_via_facade_only` 等）は変更不要と見込むが、#3835 で全件を再実行して確認する。
  反転するのは挙動テスト `crates/facade/tests/compat_sequential_param_groups.rs::save_model_rejects_groups_and_keeps_empty_groups_working`
  （`:641`）で、往復の正テスト（改名案: `save_model_round_trips_param_groups_and_keeps_empty_groups_bit_identical`）へ置き換える。
  `param_groups_usage_doctests_are_present_and_compiled` は doc を更新したら再実行する。

### 18.2 rustdoc と doctest の例

- `compile_with_param_groups` の「# 制約」を「groups も `save_model` で保存・復元される（形式は決定記録 §18）」に直す。
  `save_model`／`load_model` の doc に groups の往復を 1 行足す。
- doctest は facade 単独・CPU で、一時ディレクトリへ保存 → 読み込み → `fit` 1 step の結果がビット一致（GPU 不要で `ignore`／`no_run` なし）。
- 失敗系は `compile_fail` に頼らず、正のテストで確かめる（`docs/facade-speculative-decoding-batching-design.md` §18.3 の方針）。

### 18.3 エラー型

- 新しい variant は作らない（`ModelIoError` は `#[non_exhaustive]`）。
  - スキーマ違反（未知キー・長さ不整合・空の group・重複・非有限・負値、`format_version` 1 や Functional manifest に groups がある）: `Manifest`
  - スロット範囲とモデルの不一致: `Mismatch`
  - 作れない組み合わせ（`Lbfgs` と groups、AMP と groups〈#3832 が実装されるまで〉）: `UnsupportedModel`
  - 配列長の超過: 既存の `TooLarge`
- 検証順（すべてアロケーションとモデル状態の変更より前。本番経路で `unwrap`／`expect` は使わない）:
  1. `param_groups` が無ければ groups は空（旧ファイル・`compile()` 由来）
  2. あれば `exact_fields` で 4 キーの完全一致、各値が配列であること
  3. 長さの整合: `lr.len() == weight_decay.len() == G`、`offsets.len() == G + 1`、`G >= 1`、`offsets[0] == 0`、`offsets` は狭義増加（空 group を拒否）、`offsets[G] == slots.len()`
  4. `slots` の重複を group 内・group 間ともに拒否。`lr`／`weight_decay` は有限かつ非負（`resolve_slot_hparams` と同じ規則）
  5. 組み合わせ（`Lbfgs`・AMP）と版（`format_version` 1・Functional）の検査
  6. モデル構築後に、どのスロット添字も `trainable_parameters().len()` 未満であること（`check_slot_shapes` の隣）
  7. 復元は construct-before-assign（`ParamGroup::new` で組み立て、全部成功してから代入）
- 保存側（`prepare_save`）も同じ検証を書き出し前に通し、「書いたものは必ず読める」契約を保つ。どこまで保存時に強制するかは 18.10 の点 4。
- 検証器は facade 側（`model_io/compiled.rs`）に置く。`resolve_slot_hparams` は `pub(crate)` なので、規則の食い違いを防ぐため、
  同じ不正入力を facade 公開の `step_with_groups` も拒否することを確かめる突合テストを足す。

### 18.4 テスト計画

すべて CI（ubuntu-latest）で回る。GPU カーネル・`Op`・`BackendOps` に触れないので実機列は「該当なし」。置き場は
`crates/facade/tests/compat_sequential_param_groups.rs`（往復・挙動）と `crates/facade/src/compat/model_io.rs`／`model_io/compiled.rs` の単体テスト（パーサ）。

| 受け入れ基準 | テスト | 区分 | `<cuda-node>` | `<metal-node>` | `<rocm-node>` |
|---|---|---|---|---|---|
| groups ありの往復で、パラメータ・optimizer 状態・groups（`PartialEq`）が一致し、再開後の `fit` が保存しない場合とビット一致 | 新規（上記の置き換え） | CI | 該当なし | 該当なし | 該当なし |
| groups なしの manifest が変更前とバイト単位で同じ | 既存のゴールデン文字列テストが無変更で通る | CI | 該当なし | 該当なし | 該当なし |
| `param_groups` キーのない旧ファイルが空の groups で読める | 新規（キーを除いた fixture） | CI | 該当なし | 該当なし | 該当なし |
| 長さ不整合・`offsets` の非単調・空 group・重複スロット・非有限や負の `lr`／`weight_decay` を `Manifest` で拒否 | 新規（パーサ単体） | CI | 該当なし | 該当なし | 該当なし |
| スロット範囲外を `Mismatch` で拒否し、部分状態を返さない | 新規 | CI | 該当なし | 該当なし | 該当なし |
| `Lbfgs`／AMP と groups、`format_version` 1 に groups、Functional manifest に groups を拒否 | 新規 | CI | 該当なし | 該当なし | 該当なし |
| facade の検証器と `step_with_groups` の拒否集合が一致する | 新規（突合テスト） | CI | 該当なし | 該当なし | 該当なし |
| Functional の保存・読み込みが不変 | 既存 `functional_io/tests.rs` が無変更で通る | CI | 該当なし | 該当なし | 該当なし |

実機で測るテストはない（実機依存の経路に触れないため）。

### 18.5 数値契約

新しい算術はなく、判定はビット一致のみ。REQ-2 の統一複合判定（相対誤差 1e-3 未満 または 絶対誤差 1e-5 未満）・tolerance 定数
（`RELATIVE_TOLERANCE`／`ABSOLUTE_RESCUE_THRESHOLD`）・parity baseline・ガードレール閾値は変えない。FMA 契約・f64 長軸縮約の契約への影響はない。
baseline の追加・更新はない。

### 18.6 互換への影響

**追加のみ**。`docs/api-compat-0.14.0-decision.md` §1.2 の A3-c 行（型より保存形式の互換が論点。`ModelIoError` は `#[non_exhaustive]`）と §8 の判定手順に従う。
`ModelIoError`・`ParamGroup`・`ParamGroupStep`・`save_model`／`load_model` のシグネチャは不変。保存形式は新リーダーが旧ファイルを読める（後方互換）。
旧リーダーは groups を含む新ファイルを fail-closed に拒否する（前方互換はない。lbfgs と同じ扱い）。挙動のみの変更（拒否 → 成功）は
移行ノート（`docs/migration/0.14.0.md`、雛形は #3780）の「挙動のみの変更」欄へ #3835 が書く。

### 18.7 適用記録・索引・追跡表

適用記録の案（#3835／#3836 が `docs/compat-api-scope.md` へ書く。承認 URL は #3769 の回答を待って入れる）:

```text
**適用記録（イシュー #3835・#3836。承認: <#3769 の回答コメント URL>）**: compile_with_param_groups の groups を save_model／load_model の manifest（compiled.param_groups・format_version 据え置き）で保存・復元。公開名の追加なし。公開状態: 未公開
```

- [ ] `docs/compat-api-scope.md` に適用記録を追記した（実装 PR が行う）
- [x] `docs/README.md` の索引を更新した
- [x] 追跡表 `docs/compat-matrix-ari-tracker.md`（行 21）を更新した

### 18.8 ベンチと性能への影響

既定経路（groups なしの保存・読み込み・学習）は分岐 1 つと空チェックだけで、算術も確保も増えない。計測しないので数値は書かず、前後比較は不要。

### 18.9 依存の追加の有無

- 依存追加: なし。手書きの JSON パーサを使い、`serde_json` は入れない。
- 許容 12 区分・禁止リストとの照合: 変化なし。
- 新規 `unsafe`: なし。

### 18.10 所有者へ戻す論点（#3769 論点 3）

| 点 | 選択肢 | 推奨 | 確認がないときの扱い |
|---|---|---|---|
| 1. 保存するか | 保存する／拒否を恒久契約にする | 保存する | 保存の拒否を維持（実装へ進めない） |
| 2. 版数 | (i) 据え置き／(ii) groups ありのときだけ 3／(iii) 常に 3 | (i) | 同上 |
| 3. 符号化 | A（CSR 形の 1 キー）／A'／B（スロット写像）／C（safetensors） | A | 同上 |
| 4. 保存・読み込みで強制する検証の範囲 | 両側でスロット範囲まで厳密に照合／スキーマだけ見てスロット範囲は fit 入口に委ねる | 両側で厳密に照合 | 同上 |
| 5. A3-a（AMP 併用）・A3-b（`LrSchedule` 併用）との関係 | AMP と groups は #3832 が実装されるまで `UnsupportedModel`、groups の `lr` は絶対値として保存／A3-b の形で意味が変わるならそのとき任意キーを足す | 左記 | 同上 |

### 18.11 セキュリティ・スコープ外

- A03: manifest は非信頼データ。長さ・添字・有限性の検証を確保とモデル変更より前に行い、既存の上限（`MAX_ARRAY_LEN`・`MAX_OBJECT_KEYS`・`MAX_JSON_DEPTH`・`MAX_MANIFEST_BYTES`）の中に収める。
  非信頼な `offsets`／`slots` の値で確保量を決めない。
- A04／A08: 保存側の自己検証で「読めないファイル」を作らない。閾値・tolerance・baseline・上限定数は変えない。
- 秘密情報は書かず、実機ホスト名は `<cuda-node>` 等で表す。
- スコープ外: A3-a（#3752／#3832）、A3-b（#3754）、A3-d（#3756）、`Lbfgs` 併用、`Optimizer` にない 4 種（#3757）、
  `OptimizerStateDict` への groups の追加（optimizer が groups を持たないため）、`LrSchedule` の状態保存。新しい issue は起票しない。
