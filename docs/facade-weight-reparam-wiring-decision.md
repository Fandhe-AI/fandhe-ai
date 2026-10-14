# weight_norm・spectral_norm の結線方式の決定記録（#3047）

- 対象: `Var::weight_norm`／`Var::spectral_norm`（#2851 で公開済み）を `Linear`／`Conv*` へ結線する方式。
- 種別: 決定記録のみ。**コード・`docs/spec`・依存・tolerance・baseline は変更しない。**
- 基準コミット: `33fe24a2`（本書の `file:line` はこのコミットでの値。行番号は変動するため §2 の再実測コマンドで確認する）。
- 承認根拠: 2026-10-11 の所有者判断 D-A（推奨形で一括承認。`compat-api-scope.md` §5 末尾の適用記録〈#3020〉）。D-A は「本記録に推奨形を書き、その形のまま実装へ進む」ことの承認であり、個別の追加承認ではない。
- 親: #3046（結線の実装）。ルート: #3018。後続: #3048（Linear）・#3049（Conv 系）・#3050（保存往復・学習テスト・保留ガード反転）。

## 0. 結論

| 項目 | 決定 |
|---|---|
| 結線方式 | **(a) 層内再パラメータ化**（既存の層型が `weight` スロットを分解パラメータへ置き換えた状態を内部に持つ）。層ラッパー型 (b) は採らない |
| 公開名 | facade `Sequential::parametrize_weight_norm(layer_index, dim)`・`Sequential::parametrize_spectral_norm(layer_index, n_power_iterations, eps, seed)`。追加のみ |
| 保存キー | `parametrizations.weight.original0`（g）・`original1`（v）／`parametrizations.weight.original`。`spectral_norm` の `u`／`v` は buffer（`parametrizations.weight.0._u`／`._v`） |
| manifest | kind を `{base}_weight_norm`／`{base}_spectral_norm` として 12 種新設。`format_version` は 2 のまま |
| `norm_except_dim` | 非公開のまま（§11） |
| 所有者へ戻す点 | **0 件**。実測で確定する技術点は §14.1 に列挙 |

## 1. 背景と要件

- `autodiff-lrn-weight-reparam-decision.md` §7・§12 は `Var` 委譲までを確定し、結線方式（§12.5）を保留していた。§14 は D-A による保留解除を記録済み。
- 比較軸は 5 つ: `named_parameters` の見え方・`save_model` のキーとスキーマ・`spectral_norm` の状態保存・optimizer／param groups との整合・常駐経路（`DeviceParamStore`）の fail-closed。
- 対象層は `Linear`、続いて `Conv1d`／`Conv2d`／`Conv3d`／`ConvTranspose1d`／`ConvTranspose2d`。

## 2. 現状事実（基準コミットでの実測）

| 事実 | 出典 |
|---|---|
| `Var::weight_norm(&self /*v*/, g, dim: Option<usize>)`／`Var::spectral_norm(&self, &mut SpectralNormState, training)` | `crates/autodiff/src/var.rs:5619,5626` |
| `SpectralNormState`: `dim`・`u: [h]`・`v: [w]`・`eps`・`n_power_iterations`。`#[non_exhaustive]`・フィールド非公開・乱数初期化なし。初期化は `from_vectors` ＋ `power_iterate(weight, 15)`。forward 失敗時は状態を進めない。`training=true` のときだけ更新し、勾配は `weight` にのみ流れる | `crates/autodiff/src/weight_reparam_ops.rs:124`、`autodiff-lrn-weight-reparam-decision.md` §12 |
| `weight_norm` は `g` が `norm_except_dim` の出力 shape と完全一致する場合のみ受理。`dim=Some(d)` は `d` 軸以外が 1 の keepdim 形、`None` は rank 0 | 同上 |
| `Linear.weight` は `[in, out]`（PyTorch と転置）。`Linear::bind(&self, tape) -> LinearVars` は `Result` を返さず、`LinearVars { pub weight, pub bias }` はフィールド pub。`Sequential::bind` も `Result` を返さない | `crates/autodiff/src/nn/linear.rs`、`crates/facade/src/compat/sequential.rs:2253` |
| `Module::named_parameters` の命名契約は `weight` → `bias` の順 | `crates/autodiff/src/nn/` |
| `Sequential` は `as_linear()`／`as_conv*()` の downcast フックに依存する（`trainable_parameters`・`apply_parameters`・常駐経路・`SequentialVars::forward`／`trainable_vars`／`trainable_grads`・`first_untracked_parametric_layer`・`contains_resident_unsupported_layer`・ONNX export `crates/onnx-interop/src/onnx/export_nn.rs`・旧複製 `crates/autodiff/src/compat/sequential.rs`） | `crates/facade/src/compat/sequential.rs`（2078／2420／2458／2538／2805／2945／3160／3223／3439／3678／3823 付近） |
| `Sequential` は `compiled: Option<Compiled>`（optimizer スロットが compile 時に決まる）と `specs: Vec<LayerSpec>` を持つ。`specs` は `model_io` が manifest を導く唯一の根拠 | 同上 |
| `model_io`: kind は文字列 allowlist・`params` は kind ごとの固定キー集合（未知・欠落キーは `Manifest` エラー）・`format_version` は 2・`buffer_keys` に BatchNorm running stats の先例（`expected_buffer_keys`）・未知 kind は `UnsupportedModel` | `crates/facade/src/compat/model_io.rs:736`、`compat-model-io-decision.md` |
| BatchNorm は `RefCell` で `&self` forward から状態を更新する先例 | `crates/autodiff/src/nn/batch_norm.rs:105` |
| インベントリテスト `workspace_declares_lrn_weight_reparam_fn_names_only_in_allowed_locations` が `fn weight_norm`／`fn spectral_norm`／`fn norm_except_dim` の宣言を許可位置の各 1 件に固定する | `crates/facade/src/lib.rs:6302,6316` |

再実測コマンド:

```bash
grep -rn 'as_linear\|as_conv' crates          # フック消費箇所
grep -n 'pub fn weight_norm\|pub fn spectral_norm' crates/autodiff/src/var.rs
grep -rn 'workspace_declares_lrn_weight_reparam' crates
grep -n 'expected_buffer_keys\|format_version' crates/facade/src/compat/model_io.rs
```

### 2.1 PyTorch との対応

- 新 API（`torch.nn.utils.parametrizations`）の `state_dict` キー: `weight_norm` は `parametrizations.weight.original0`（g）・`original1`（v）、`spectral_norm` は `parametrizations.weight.original`、buffer は `parametrizations.weight.0._u`・`._v`。
- `spectral_norm` の `dim` は通常の層で 0、`ConvTranspose*` で 1。
- `weight_norm` の `dim` は**層種に依らず利用者指定（既定 0）**。「ConvTranspose は dim=1」は `spectral_norm` にのみ当てはまる（#3049 本文の記述との差。§13 で申し送る）。
- 我々の `Linear.weight` は `[in, out]` のため、PyTorch の出力ごとの軸は我々の軸 1。Conv は PyTorch と同一レイアウト（`[out, in/g, k...]`）、`ConvTranspose` は `[in, out/g, k...]`。

## 3. 方式比較

| 軸 | (a) 層内再パラメータ化 | (b) 層ラッパー型 |
|---|---|---|
| 内容 | 既存の層型が `weight` スロットを分解パラメータへ置換した状態を持つ | `Module` を実装する新しい層型が内側の層を包み、forward のたびに再計算 |
| `named_parameters` | `weight` の位置に `parametrizations.weight.original0`／`original1`（または `original`）。`bias` は不変 | 同名にもできるが `as_*` フックと `SequentialVars` への新規配線が必要 |
| `save_model` | kind 追加と `params` キー追加で足りる | 新しい型ごとに kind・params・読み出し経路を新設 |
| `spectral_norm` 状態 | 層内 `RefCell`（BatchNorm 先例）。`set_training` で同期 | 内側の層と別に状態を持つ。同期は同等 |
| optimizer／param groups | スロット数・順序が層内で決まり、既存の共通順序契約を再利用 | 新規配線で順序契約を再実装 |
| 常駐経路の fail-closed | `weight()` が生テンソルを返すため暗黙の誤読リスクあり。ゲート漏れの機械検査が必須 | `as_linear()` が `None` になり既存 fail-closed が構造的に効く |
| 配線の増分 | 既存分岐内の小改修 | 新フック・bind 用 Vec・forward 分岐を新設。内側の層ごとに forward の重複 |
| PyTorch 整合 | キー・意味とも一致 | キー名は合わせられるが型構造が異なる |

## 4. 推奨

**(a) を採る。** 理由は、薄いラッパーに徹する方針（REQ-9）に沿い、forward・bind・学習経路を再利用できて配線の重複が最小になるため。

- 残るリスク: (b) が構造的に持つ fail-closed を (a) は持たない。弱点は §10 の「ゲート ＋ ソース走査インベントリテスト」で塞ぎ、**インベントリテストは #3050 の必須条件**とする。

## 5. 公開名（追加のみ）

- `Sequential::parametrize_weight_norm(self, layer_index: usize, dim: Option<usize>) -> Result<Self, AutodiffError>`
- `Sequential::parametrize_spectral_norm(self, layer_index: usize, n_power_iterations: usize, eps: f32, seed: u64) -> Result<Self, AutodiffError>`
- 作成済みの層へ後から適用するビルダー形（PyTorch の `parametrizations.*(module)` と同じ使い方）。`add_*` を 12 本以上増やさずに済む。
- 名前は `weight_norm`／`spectral_norm` を避けた別トークンにする（§2 のインベントリ衝突の回避）。
- 適用できる層は `Linear`／`Conv1d`／`Conv2d`／`Conv3d`／`ConvTranspose1d`／`ConvTranspose2d` の bias あり固定の層（`add_*` と同じ）。`ConvTranspose3d` は #3025 の層化結果に依存し、層が入った場合に限り追加する（条件付き）。
- 次は `InvalidArgument` で拒否する: 非対応の層（活性化・`add_module` 等）・範囲外の添字・二重適用・compile 済み（optimizer スロットが固定済みのため。compile 前限定）。
- autodiff 側の層にも追加のみの inherent メソッド（例 `Linear::with_weight_norm(self, dim)`・`with_spectral_norm(self, n, eps, seed)`。名称は #3048 で最終化）を置く。`fandhe-ai-autodiff` は内部クレートで facade が唯一のサポート API のため、結線の実体は autodiff 層、公開は facade の 2 メソッドとする。
- 既存シグネチャは変えない（`add_*`・`Linear::bind`・`LinearVars` のフィールド）。`bind` が infallible で再構成が fallible のため、bind では生の葉（`g`／`v` または `original`）のみ登録し、有効重みの再構成は `forward` で行う。`*Vars` への補助フィールド追加が構造体リテラルの構築箇所を壊す場合は、追加の別名 `bind_*` を足す案へ切り替える（#3048 で grep 実測して判断）。

## 6. 対象層ごとの軸の表

| 層 | weight shape（我々） | `weight_norm` の `dim`（利用者指定。PyTorch 既定 0 相当） | `spectral_norm` の `dim`（層から導出） | PyTorch との差 |
|---|---|---|---|---|
| Linear | `[in, out]` | `Some(1)` が PyTorch の `dim=0` 相当 | 1 | レイアウト転置のみ。`W_mat` は PyTorch と同一（rows=out）で u／v の役割・反復軌跡も一致 |
| Conv1d／2d／3d | `[out, in/g, k...]` | `Some(0)` | 0 | 同一 |
| ConvTranspose1d／2d | `[in, out/g, k...]` | `Some(0)` が PyTorch 既定（入力チャネル軸）。出力チャネル単位は `Some(1)` | **1** | `spectral_norm` のみ `dim=1` |

`g` の shape は、`dim=Some(d)` なら `d` 軸以外が 1 の keepdim 形、`None` なら rank 0。

## 7. 保存スキーマ

- `named_parameters`／`state_dict` のキー（`Sequential` では `{i}.` 接頭辞付き）は、再パラメータ化したスロットを weight の位置に置き換える。
  - `weight_norm`: `parametrizations.weight.original0`（g）→ `parametrizations.weight.original1`（v）→ `bias`
  - `spectral_norm`: `parametrizations.weight.original` → `bias`
- manifest の kind は `{base}_weight_norm`／`{base}_spectral_norm`（base は `linear`／`conv1d`／`conv2d`／`conv3d`／`conv_transpose1d`／`conv_transpose2d`。計 12 kind）を allowlist へ追加する。
  - `params` は base の既存キーに追加: `weight_norm` は `dim`（整数または `null`）、`spectral_norm` は `n_power_iterations`・`eps`（f32 の正準表記）。
  - 固定キー集合の規約を保ち、旧リーダーは未知 kind を `UnsupportedModel` で fail-closed 拒否できる。`format_version` は 2 のまま据え置く。
- `spectral_norm` の `u`／`v` は buffer として保存する（キー `{i}.parametrizations.weight.0._u`／`._v`）。`buffer_keys` に層構成から導いた期待値と完全一致で載せる（BatchNorm running stats〈#2371〉と同方式）。非有限値は保存時に拒否する。
- load は buffer キー集合と shape の一致を確認してから層を構築し、確保は検証後に限る。
- 保存・復元で `spectral_norm` の次の forward が bit 一致するのは `u`／`v` を保存する場合に限る。

## 8. `spectral_norm` の状態

- 状態は層内の `RefCell<SpectralNormState>`（BatchNorm と同じ先例）。
- training フラグは層の `set_training` で同期する。`eval()` では更新せず、`train()` の forward でのみ更新する。
- 初期化は決定的シード（`derive_seed`／`uniform_init` 系）で `u0`／`v0` を作り `power_iterate(weight, 15)` を適用する（乱数を持たない現行方針と整合）。
- forward が失敗しても状態は進まない（既存契約の継承）。勾配は original のみへ流れる。

## 9. optimizer・param groups

- スロット数: `weight_norm` は g・v の 2 本に bias、`spectral_norm` は 1 本のまま。
- スロット順は層順・層内で「再パラメータ化スロット → bias」。`trainable_parameters`／`trainable_vars`／`trainable_grads`／`apply_parameters` の共通順序契約（#294）を維持する。
- param groups はスロット添字で指定するため、`parametrize_*` は compile／param group 設定の**前**に行う。compile 済みは拒否する。

## 10. 常駐経路・その他の fail-closed

再パラメータ化された層に対し、次を型付きエラーで拒否する。

- `init_device_param_store`／`forward_resident`／`predict_resident`: `BackendError::Unsupported`。
- tape 不要の `predict` 系 host 経路（`forward_host`・Linear→ReLU 融合）: `supports_forward_host=false` 相当にするか拒否してフォールバックさせ、生の `weight()` を有効重みとして読む経路を通さない。
- ONNX export（`export_nn.rs`）。
- `weight()` アクセサは署名を変えず「生の一次テンソル」を返すと doc に明記する。`as_linear()`／`as_conv*()` の全消費箇所（§2）を、再パラメータ化の有無でゲートする。
- 機械検査: ソース走査型インベントリテスト（消費箇所の列挙とゲート有無の固定）を #3050 に割り当てる。先例は `workspace_declares_…`。stable rustdoc は `compile_fail` のコードを照合しないため、否定ガードは「正のプローブ ＋ インベントリ」で組む。
- 外部入力の検証（A03）: 新 kind の `params` はキー集合を固定し、数値は整数または f32 の正準表記のみ。shape は非信頼な整数から純粋算術で導き確保前に検証する。buffer の `u`／`v` の shape・非有限値も検証する（既存方針の踏襲）。

## 11. `norm_except_dim`

**非公開のままとする。** `g` の初期化は層内部が行う（`g = ‖v‖`。初期状態の有効重みが元の重みと一致）ため利用者が `g` を自作する必要がなく、#2849・D-0 の対象外確定と整合する。層の結線で公開を求める新たな根拠は生じない。公開が必要になった場合は別 issue で個別承認とする。

## 12. 非破壊性確認

追加のみ: 新規 inherent メソッド（`Sequential::parametrize_*`・autodiff 層側 `with_*`）・新規 manifest kind・新規 buffer キー。既存の `add_*` のシグネチャ・挙動・manifest スキーマは不変。`format_version` は据え置き、旧リーダーは未知 kind を `UnsupportedModel` で拒否する。`fandhe-ai =0.11.0` の公開面に対する「追加のみ」は `api_surface.rs` の正ガードで #3048 が固定する。新規 GPU カーネルはなく（再構成は既存 `Var` Op）、依存・`unsafe`・tolerance・baseline・`docs/spec` は変更しない。

## 13. 後続 issue への分解

- **#3048（Linear）**: (1) `Linear` へ再パラメータ化状態を追加 (2) `forward` で有効重みを再構成（`Var::weight_norm`／`Var::spectral_norm`）(3) `named_parameters`／`set_parameter`／`state_dict` 対応 (4) 状態の `RefCell` 保持と `set_training` 同期 (5) `Sequential::parametrize_*` と `LayerSpec` の拡張 (6) 常駐・host・ONNX 経路のゲート (7) `Var` メソッド直接使用との勾配一致テスト（`crates/facade/tests/`）(8) `api_surface.rs` の正ガード。
- **#3049（Conv 系）**: Linear と同型。`ConvTranspose` は軸の差（§6）を doc とテストで明示し、PyTorch 一致の根拠として `spectral_norm` の `dim=1` を検査する。**#3049 本文の「ConvTranspose は dim=1」は `spectral_norm` のみに当てはまり、`weight_norm` の `dim` は利用者指定（§6）。**
- **#3050**: 保存往復テスト（kind・buffer の完全一致と tamper 拒否）・学習テスト・`LrnWeightReparamHoldDoctestGuard` の該当分の反転・§10 のインベントリテスト。
- issue 本文の書き換えは本 issue の範囲外。

## 14. 所有者へ戻す点・スコープ外

承認依頼として戻す点は **0 件**（D-A は推奨形で実装へ進む）。

### 14.1 実装 issue で実測して確定する技術点

- rank 0 の `g`（`weight_norm` の `dim=None`）が safetensors で往復できるか（#3050）。
- `*Vars` へのフィールド追加が構造体リテラル構築を壊す場合の別名 `bind_*` への切替（#3048）。
- `ConvTranspose3d` の扱い（#3025 の層化に依存）。
- compile 後適用の拒否で足りるか。

### 14.2 スコープ外

`remove_parametrizations` 相当・旧 hook 方式（`torch.nn.utils.weight_norm`）・負の `dim`・`spectral_norm` の `dim` 利用者指定・他の層（RNN の重み等）・CUDA／Metal 実機再計測（新規 GPU カーネルなし）。

## 15. 出典

- `docs/autodiff-lrn-weight-reparam-decision.md` §7・§12・§14
- `docs/compat-model-io-decision.md`
- `docs/facade-nn-layer-type-exposure-decision.md`
- `docs/compat-api-scope.md` §5.1 行 15・§5 末尾の適用記録（#3020）
- 承認根拠: 2026-10-11 所有者判断 D-A（#3018 ツリー起票の根拠）。本記録は記録の反映であり新規の承認ではない。
