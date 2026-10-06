# facade Functional API（多入力・多出力グラフ）設計判断記録

- 対象イシュー: #2664（親 #2663・ルート #2499 Phase 4）
- 基準コミット: `c74c93f0`（`origin/main`）
- 段階: **段階 0（docs のみ）**。`crates/**`・`Cargo.toml`／`Cargo.lock`・tolerance／baseline・ガードレール閾値・`docs/spec/` は変更しない
- 状態: 本記録の推奨案は**全項目「未承認」**。承認依頼は #2677 の一括依頼に委ね、公開は承認後の #2679 で行う。本記録は承認を代行せず、承認済みとも主張しない

## 0. 結論

後続 #2665（グラフ構築・forward）・#2666（結合層）・#2667（fit・保存）が従う設計を次の 1 案に確定する（いずれも未承認の推奨）。

| 論点 | 推奨（1 案） |
|---|---|
| (a) `Input`／`Model` 相当の型 | アリーナ型ビルダー `FunctionalBuilder`（仮称）が `Copy` な添字ハンドル `Node`（仮称）を返し、`build(&inputs, &outputs)` で `FunctionalModel`（仮称）を得る。層ノードの単位は `compat::Sequential` ブロック（§3） |
| (b) 既存型との関係 | `compat::Sequential` は無変更でブロックとして**包含**する。`FunctionalModel` は facade `nn::Module` を**実装しない**（§5） |
| (c) 保存形式 | 新しい `format` 名（仮称 `fandhe-ai.compat.functional`・版 1）と新しい保存入口。既存 `save_model`／`load_model` と既存ファイルは不変（§8） |
| (d) 非破壊性 | 追加は新しい型・自由関数のみ。既存シグネチャ・意味論・`FitConfig`・既存ファイル形式は不変（§9） |
| 公開形 | 3 類型のうち **モジュール再エクスポート**（`compat/mod.rs` の `pub use`）（§10） |
| 結合層 | 既存 `Var` 演算（`cat`／`add`／`mul`／`div`）の合成。新規 `Op`・`BackendOps` メソッドなし（§6） |
| 多出力 fit | compile で指定した単一 loss を全出力へ適用し出力の指定順に合計（§7） |

## 1. 背景・要件

- 現状 facade でモデルを組む手段は `compat::Sequential`（直列専用 `add_*` ビルダー）と facade `nn::Module`／`nn::{ModuleList, Sequential, ModuleDict}` のみで、分岐・合流（多入力・多出力）を表す API がない。参照モデル（ResNet・Transformer）は residual 加算を `Var::add` で手組みし `fit` 一括 API を使えていない（`docs/reference-models-decision.md` §10.3）
- 親 #2663 は Keras Functional API 相当と結合層を設計・実装する。本記録は最初の子として設計と公開形の推奨案を 1 つに決める
- Functional API は `docs/compat-api-scope.md` §1 の Tier 列挙にも §2 の対象外にも現れない。公開は同 §5 の手続き（ユーザー承認）を要する（本記録での適用記録は追記しない。先例: #2771・#2773 の docs のみ記録 PR）
- 共通契約: crates.io 出荷済み `fandhe-ai =0.10.0` の公開 API 非破壊（`FitConfig` は `Copy + Eq` 固定でフィールド追加不可）／tolerance・baseline・依存・ガードレール閾値・`docs/spec/` 不変／依存追加・新規 `unsafe` は承認範囲外／新しい facade 公開面は「設計判断記録 → ユーザー承認 → 実装」の 2 段

## 2. 現状のコード事実（基準 `c74c93f0`）

| 事実 | 出典 |
|---|---|
| `compat::Sequential` は直列ビルダー。`forward(&self, tape: &'t Tape, input: &Var<'t>)` は facade newtype `Tape` を取る | `crates/facade/src/compat/sequential.rs:163`・`:1794` |
| 学習は `bind(&tape) -> SequentialVars` が層ごとの `*Vars` を事前登録し、`SequentialVars::forward`（`:3227`）等と位置対応契約で optimizer へ渡す | 同 `:2121`・`:3172` |
| 複数の `compat::Sequential` を同一 tape 上でそれぞれ `bind` し `Var::add` で手組みする先例がある | `docs/reference-models-decision.md` §10.3 |
| `add_module`（利用者定義 `nn::Module`）はパラメータ持ちだと学習・保存で fail-closed、無状態は通過 | `sequential.rs:1773`・`docs/facade-nn-module-exposure-decision.md` §17 |
| facade `nn::Module::forward(&self, tape: TapeRef<'t>, input: &Var<'t>)` は単一入力・単一出力。`TapeRef` から `&Tape` へ戻る公開経路はなく、パラメータ勾配を公開経由で集める経路もない | `crates/facade/src/nn/module.rs:47`・`docs/reference-models-decision.md` §10.8・`docs/facade-nn-module-exposure-decision.md` §20 |
| 組み込み層型は facade 非公開で、`compat::Sequential::add_*` 経由でのみ到達する | `crates/facade/src/nn/mod.rs` モジュール doc |
| 生 `Tape`・`BackendOps` を署名に出す公開面は REQ-12 違反で `api_surface.rs` が機械固定 | `crates/facade/tests/api_surface.rs:201`（`compat_public_functions_do_not_accept_raw_autodiff_tape_argument`） |
| `fit<T: FitTarget>(&mut self, x, y, config: FitConfig)` は単一入力・単一目標。`FitConfig` は `#[derive(Debug, Clone, Copy, PartialEq, Eq)]`。`Compiled` は `pub(super)`、`OptimizerState` は private | `crates/facade/src/compat/training.rs:1255`・`:81`・`:712` |
| 保存形式: `manifest.json` + `model.<gen>.safetensors`。`FORMAT_NAME = "fandhe-ai.compat.sequential"`・`FORMAT_VERSION = 2`。上限 `MAX_MANIFEST_BYTES = 1 MiB`・`MAX_LAYERS = 4096`・`MAX_JSON_DEPTH = 4`・`MAX_ARRAY_LEN = 2 * MAX_LAYERS`・`MAX_OBJECT_KEYS = 16`。`save_model(&Sequential, dir)`／`load_model(dir) -> Result<Sequential, _>` は署名固定。保存前に `verify_round_trip` で往復検証 | `crates/facade/src/compat/model_io.rs:95`・`:101`・`:112`・`:118`・`:132`・`:137`・`:143`・`:264`・`:281`・`:1402`・`docs/compat-model-io-decision.md` |
| 結合に使える既存演算: `Var::cat(&[Var], dim)`・`Var::add`／`mul`／`div`（二項・broadcast あり）・`Var::stack`。定数は `Tape::var_no_grad` で作れる | `crates/autodiff/src/var.rs:663`・`:686`・`:887`・`:3129`・`:3187`・`crates/autodiff/src/tape.rs:2939` |
| facade 内 `pub(crate)` 実装 + 保留ガードの先例（`predict_batches` #2192。`*HoldDoctestGuard` + `api_surface.rs` の否定ガード） | `crates/facade/src/lib.rs`・`docs/facade-predict-batches-phase-metrics-decision.md` |

## 3. グラフ構築の型（推奨: アリーナ型ビルダー + ノードハンドル）

| 案 | 内容 | 判定 |
|---|---|---|
| **A（推奨）** | `FunctionalBuilder` がノード列を所有し `Copy` な `Node` を返す。Keras の `Input(...)` ↔ `builder.input()`、層適用 ↔ `builder.apply(block, node)`、結合 ↔ `builder.concatenate(&[..], dim)` 等、`Model(inputs, outputs)` ↔ `builder.build(&inputs, &outputs)` | 採用 |
| B | Keras 流シンボリックテンソル（`Rc<RefCell<..>>` 共有グラフ + `layer.call(&sym)`） | 不採用。組み込み層型が facade 非公開で `layer.call` は層型の公開（別承認）を前提にする |
| C | facade `nn::Module` へ多入力 forward を足す（trait 拡張） | 不採用。trait 変更で単一入力の既存実装者と整合せず、`TapeRef` 経由ではパラメータ勾配を集められない制約（`docs/facade-nn-module-exposure-decision.md` §17・§20）を引き継ぐ |

- A はノードが「自分より前の添字」しか参照できないため、**構築時点で DAG が保証され挿入順がそのままトポロジカル順**になる（循環検査・`Rc` が不要）
- 層ノードの単位は **`compat::Sequential` ブロック**（`apply` は値で受けて所有）。50 種超の `add_*`・`LayerSpec`（保存）・`bind`／`SequentialVars`（学習）・`add_module` を再利用でき、層語彙を二重に持たない（REQ-9 の薄いラッパー原則）
- 検証（すべて型付き `Err`・fail-closed・`unwrap`／`expect` なし）: 他ビルダー由来ハンドル・範囲外添字・`inputs`／`outputs` の空や重複・入力ノード以外を `inputs` に指定・出力へ到達しない入力・どの出力にも寄与しないノード（**推奨は拒否**）・結合入力 0 件
- `compile()` 済み `Sequential` を `apply` へ渡すと `InvalidArgument` で拒否。compile 状態（optimizer 内部状態・AMP）は Functional モデルが 1 つだけ持つため、ブロック側状態を黙って捨てる・二重保持するのを避ける
- 構築時の shape 推論は行わない（層側に shape 推論 API がない）。shape 不整合は forward 時に既存 `Var` 演算の型付きエラーで検出する
- 重み共有（同一ブロックの複数適用）は第 1 段対象外（`apply` が値で消費するため構造的に不可。BatchNorm running stats の二重更新等の意味論が未定）

## 4. forward・多入力・多出力

- 署名の推奨形: `forward(&self, tape: &'t Tape, inputs: &[Var<'t>]) -> Result<Vec<Var<'t>>, AutodiffError>`、`predict(&self, inputs: &[&Tensor<f32>]) -> Result<Vec<Tensor<f32>>, AutodiffError>`。facade `Tape` を取り、生 `Tape`・`BackendOps`・内部層型を署名に出さない（REQ-12。`api_surface.rs:201` と整合）
- ノードを挿入順に反復で 1 回ずつ評価（再帰を使わず深いグラフでのスタック枯渇を避ける）。入力件数不一致は `InvalidArgument`。多出力は `outputs` の指定順
- 状態系: `set_training`／`train`／`eval`／`training`（全ブロックへ伝播）・`named_parameters`／`state_dict`／`load_state_dict`（strict・fail-closed）・`trainable_parameters`／`apply_parameters`。パラメータキーは **全ブロックを通した層の通し番号 `i` による `{i}.{name}`** とし、§8 の `layers`／`parameter_keys`／safetensors キーと同一の採番を 1 か所の規則として定義する
- 常駐経路（`DeviceParamStore`／`predict_resident` 相当）は第 1 段では**入口を設けない**。GPU は既存 `Var` 演算の合成のため `tape_for(device)` で構築した tape 上で到達可能（新規 `Op`・`BackendOps` メソッドなし）

## 5. 既存 `compat::Sequential`／`nn::Module` との関係

- `compat::Sequential`: 変更なし。Functional 側がブロックとして包含する（逆方向依存なし）。`Sequential::add_*` へ結合層を足す案は「単一入力→単一出力」の層契約に合わず不採用
- facade `nn::Module`: `FunctionalModel` は**実装しない**。`Module::forward` は単一入力で `TapeRef` を取り、`TapeRef → &Tape` の公開経路がないためブロック（`Sequential::forward` は `&Tape`）へ委譲できない（`docs/reference-models-decision.md` §10.8 の公開ブリッジ案には依存しない）
- 利用者定義層はブロック内 `Sequential::add_module` 経由。既存契約（パラメータ持ちは学習・保存で fail-closed、無状態は通過）がそのまま効く
- 名前衝突: 配置は `compat` 側で、facade `nn::{Sequential, ModuleList, ModuleDict}` とは名前空間が別。仮称 `FunctionalBuilder`／`FunctionalModel`／`Node` は #2677 で確定する

## 6. 結合層（#2666 へ渡す仕様）

- Concatenate = `Var::cat`、Add = `Var::add` の左畳み込み、Multiply = `Var::mul` の左畳み込み、Average = Add の左畳み込み後に入力数 `n` で 1 回除算。除算は `Var::div`（`var.rs:887`）に `Tape::var_no_grad`（`tape.rs:2939`）で作った定数 `n` を渡す（`1/n` 乗算でなく `n` での除算を既定。PyTorch の平均が「和 ÷ 件数」のため）。`stack`＋`mean` は非 contiguous 入力を拒否するため使わない。実装時に `Var::div` の broadcast 挙動を再確認し API 名を確定する
- **新規 `Op`・`BackendOps` メソッドは追加しない**（`docs/autodiff-packed-sequence-decision.md` と同じ既存 Op 合成方式）
- 数値契約: 入力の index 順に f32 で逐次演算。高々数個の要素ごと演算で、`.claude/rules/coding-rust.md` の「長軸縮約の f64 アキュムレータ」の対象外。統一複合判定・FMA 契約は不変
- Add／Multiply／Average は**全入力 shape 完全一致を要求し broadcast を拒否**（暗黙 broadcast は結線ミスを成功させるため）。入力 1 件は拒否（2 件以上を要求）
- 置き場: 数値を伴う本体は内部クレート `autodiff` の自由関数モジュール（仮称 `merge_ops`）。compat 層に数値ロジックを持ち込まない（REQ-9）
- fixture: PyTorch に Functional API はないため `torch.cat`／`+`／`*`／平均の手組み参照を `crates/autodiff/tests/fixtures/<slug>-pytorch-reference/gen_reference.py` で生成し同 shape で比較する。実機（CUDA／Metal）parity は `#[ignore]` 分離し `docs/perf/logs/<slug>-2666/README.md` へ申し送る

## 7. 学習（#2667 へ渡す仕様）

- パラメータ収集: 全ブロックをノード順に `bind` し `trainable_parameters`／`trainable_vars`／`trainable_grads`／`apply_parameters` をノード順に連結。`fandhe_ai::optim` の位置対応契約を再利用（同一 tape 上の複数 `bind` は ResNet examples に先例）
- `compile(optimizer, loss)` は既存 `compat::{Optimizer, Loss}` を再利用。`fit`／`evaluate` は**新しい型のメソッド**として追加し、既存 `Sequential::fit*` の署名・演算列・`FitConfig` は変えない（既存経路の bit 同一を回帰テストで固定することを #2667 の要件とする）
- 署名の推奨: `fit<T: FitTarget>(&mut self, xs: &[&Tensor<f32>], ys: &[&Tensor<T>], config: FitConfig) -> Result<History, AutodiffError>`。多出力は単一 loss を全出力へ適用し出力の指定順に合計（Keras の単一 loss 指定時の既定）。公開後の署名変更は破壊的になるため最初から複数出力を受けられる形にする。出力別 loss・`loss_weights`・出力ごとに異なる目標 dtype は対象外
- 内部可視性: `Compiled` は `pub(super)`、`OptimizerState` は private（`training.rs:712`）。#2667 が兄弟モジュールから再利用するには crate 内部のみの可視性拡大が要る（公開面は不変）。**optimizer 状態の型を別に新設しない**
- 第 1 段で設けないもの: callbacks／metrics／重み付け／`train_step` フック／AMP／常駐更新
- fail-closed: 未 compile・サンプル数不一致・`batch_first=false` の MHA を含むブロック・パラメータ持ち利用者定義層は既存と同じ型付きエラー

## 8. 保存形式（#2667 へ渡す仕様）

| 案 | 内容 | 判定 |
|---|---|---|
| **S1（推奨）** | 新 `format` 名（仮称 `fandhe-ai.compat.functional`・版 1）と新入口（仮称 `save_functional_model`／`load_functional_model`） | 採用 |
| S2 | 既存形式を版 3 へ拡張 | 不採用 |

- 理由: `load_model` は戻り値が `Sequential` で署名固定のためグラフを返せない。`format` を分ければ既存 `load_model`（出荷済み 0.10.0 を含む）は `format` 不一致で `Manifest` エラー（fail-closed）となり、既存ファイル・既存リーダーへの影響がゼロ
- ディレクトリ構成・世代コミット・no-follow オープン・`safetensors_file` パターン検証・`compiled` object は既存 `model_io` の実装を共有（`docs/compat-model-io-decision.md` §12・§13 を正とする）
- トポロジは**既存上限の範囲内**で設計し、上限定数は変更しない・新設しない。`layers` は全ブロックを通した平坦な列（safetensors キー `{i}.{name}` の名前空間を不変に保つ）、`nodes` は `{index, op, inputs:[..], layer_start, layer_len, params:{..}}` の平坦配列、ほかに入力数・出力ノード列。深さは root → `nodes` → 要素 object → `inputs`／`params` の 4 段（`model_io.rs:2307` の `depth >= MAX_JSON_DEPTH` 判定で収まる見積り）、最上位キーは既存 10 + 3 = 13 個で `MAX_OBJECT_KEYS = 16` に収まる。`params` はスカラー値のみとし配列は要素 object 直下に置く（`model_io.rs:769` 付近の注記どおり）。#2667 で実コードに対し検証し、収まらなければ上限でなく表現側を平坦化する
- load 側検証: `op` は文字列 allowlist（未知は `UnsupportedModel`）／`inputs` は自ノードより小さい添字のみを load 時に検査（構築 API の保証に頼らない）／`layer_start`・`layer_len` は重複なし・隙間なし・全層被覆で、**この検査を safetensors のキーへ触れる前に完了**／件数は配列を数えた結果のみを信じ manifest の数値を確保量・ループ回数の根拠にしない／`add_module` を含むブロックは保存時に `UnsupportedModel`／保存前に `verify_round_trip` 方式の往復検証

## 9. 非破壊性

| 対象 | 扱い |
|---|---|
| `compat::Sequential` の全公開メソッド・`SequentialVars` | 不変 |
| `FitConfig`（`Copy + Eq`）・`Sequential::fit*` | 不変 |
| `save_model`／`load_model` と既存ファイル形式（`FORMAT_VERSION = 2`） | 不変 |
| facade `nn::Module` | required メソッド追加なし |
| root 再エクスポート群 | 不変 |
| 追加 | 新しい型・自由関数のみ（semver minor 相当） |

依存追加なし・新規 `unsafe` なし・tolerance／baseline／ガードレール閾値変更なし・新規 `Op`／`BackendOps` メソッドなし。

## 10. facade 公開形の推奨案（未承認）

| 公開形（3 類型） | 判定 |
|---|---|
| `Var` 委譲メソッド | 不採用。多入力グラフを表せない |
| **モジュール再エクスポート** | **推奨**。`compat/mod.rs` の `pub use` で `fandhe_ai::compat` へ型と保存入口を公開 |
| `Sequential::add_*` | 不採用。単一入力契約に合わない |

| 項目 | 内容 |
|---|---|
| 公開形 | モジュール再エクスポート（仮称 `FunctionalBuilder`・`FunctionalModel`・`Node`・`save_functional_model`・`load_functional_model`） |
| 非破壊性 | §9 のとおり追加のみ |
| 保留ガード | `FunctionalApiHoldDoctestGuard`（#2665 で `crates/facade/src/lib.rs` へ追加）＋ `api_surface.rs` の否定ガード（型名・自由関数名・`Sequential` への inherent メソッド追加の各経路を塞ぐ「正のプローブ + inventory」方式） |

既存ガードとの衝突（#2665 実装時に実名を再確認する）:

- (i) #2665 の時点: `crates/facade/src/compat/` へ `pub(crate)` の新規ファイル・型を足すだけで反応しうる inventory 型の走査（`api_surface.rs` の `compat_sequential_*` 系・facade src 全体のソース走査）。調査時点で確認できた `*_public_items_match_expected_set` は `nn` 配下向け（`:23749` 等）で、compat 配下向けの有無は実装時に確定する
- (ii) #2679 の反転時: `compat/mod.rs` の `pub use` 行を完全一致で固定している検査の有無と、保留ガードの正ガード化

## 11. 内部実装の配置と後続イシュー

| イシュー | 担当 | 配置 |
|---|---|---|
| #2666 | 結合演算 | 内部クレート `autodiff` の自由関数モジュール + PyTorch fixture |
| #2665 | グラフ型・forward | `crates/facade/src/compat/` 配下の `pub(crate)` 型（`predict_batches` #2192 と同じ方式）+ 保留ガード |
| #2667 | fit・保存 | 同上 + `model_io` 共有部品の再利用 |
| #2677 | 公開形の一括承認依頼 | §13 の承認事項を表にして依頼 |
| #2679 | 承認後の公開 | 保留ガードの反転 |

## 12. OWASP Top 10 観点（後続実装への要件）

- A03／A08: manifest のトポロジは非信頼入力。長さ・件数・添字を先に検証し、`op` は allowlist、`inputs` は前方参照禁止（循環を構文で排除）、層範囲は重複・隙間なし。manifest の数値を確保量・ループ回数の根拠にしない
- A04: 未知の `op`・未対応層・利用者定義層は型付きエラーで fail-closed。学習されないまま成功する状態・保存できたのに読めない状態を作らない
- A01／A05: 既存 `fs_guard`（no-follow・通常ファイル確認・世代コミット）を共有し、新しいファイル I/O 経路を別実装しない
- DoS: 既存の承認済み上限の範囲で設計し値を変えない。forward は反復評価
- A06: 依存追加なし・新規 `unsafe` なし
- REQ-12: 生 `Tape`・`BackendOps`・内部層型を公開署名に出さない

## 13. 承認事項（列挙のみ。未承認）

1. `docs/compat-api-scope.md` §5 による対象範囲への組み入れ
2. §10 の公開形と名前
3. 新しい `format` 名と保存入口
4. 多出力 fit の意味論（単一 loss の合計）
5. 結合層の broadcast 拒否
6. 第 1 段の対象外項目（§14）

承認依頼は #2677。承認が得られるまで本記録は未承認のまま保持する。

## 14. スコープ外

重み共有・構築時 shape 推論・出力別 loss／`loss_weights`・callbacks／metrics 等・常駐経路・ONNX export・GPU 専用カーネル・`FunctionalModel` の `nn::Module` 実装。新規 Issue は未承認のため起票しない（`.claude/rules/out-of-scope-tracking.md`）。

## 15. 出典

`docs/compat-api-scope.md`・`docs/reference-models-decision.md`・`docs/compat-model-io-decision.md`・`docs/facade-nn-module-exposure-decision.md`・`docs/facade-predict-batches-phase-metrics-decision.md`・`docs/autodiff-packed-sequence-decision.md`・§2 表の各コード行。
