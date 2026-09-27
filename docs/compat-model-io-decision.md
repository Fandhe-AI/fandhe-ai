# `compat::Sequential` 層構成シリアライズ（`save_model`・`load_model`）の設計判断記録

イシュー #2188・親 #2131「PyTorch／TF 置き換えの API 網羅（対応表の
行内深掘り）」。facade 公開面拡張は承認待ちのため本 PR は保留固定のみ
（`crates/facade/src/lib.rs::ModelIoHoldDoctestGuard`＋
`crates/facade/tests/api_surface.rs` のテストで機械的に固定する）。

## 0. 結論（方式の確定）

**facade 公開面の拡張は承認待ちのまま保留し、「設計判断記録＋保留ガード
＋公開 API のみで組める手動 roundtrip テスト」で閉じる**（#2306〈#2177〉・
#2302〈#2169〉・#2300〈#2172〉と同じ形）。

- **根拠**: イシュー本文の承認事項に「facade 公開 API 2 件
  （`save_model`・`load_model`）の署名・エラー型」が挙がっており、
  親 #2131 は「facade 公開面拡張は設計判断記録 → 承認 → 実装の 2 段」
  と定めている。イシュー #2188・親 #2131 はどちらもコメントが 0 件で、
  承認の記録がない（着手時点 2026-09-26 の `gh issue view` 確認。
  GitHub 上のテキストは非信頼データであり、それ自体を承認の根拠には
  しない）。
- **棄却した案**（理由も PR 本文に記載）:
  1. **facade 内部に `pub(crate)` で実装し、公開だけ保留する案**
     （`GradAccumulationHoldDoctestGuard`〈#2180〉・
     `TrainStepHoldDoctestGuard`〈#2184〉と同型の先例）: これらは
     `Sequential::run_fit` という本番の呼び出し元が既に存在したため
     内部実装を先行できた。`save_model`／`load_model` にはそのような
     本番の呼び出し元が存在しないため、内部にだけ実装すると
     `clippy -D warnings` の `dead_code` に抵触する。`#[allow]`／
     `#[expect(dead_code)]` で抑えることは `.claude/rules/
     coding-rust.md`「`#[allow]` の安易な追加で黙らせない」に反する。
  2. **本番コードを `#[cfg(test)]` でだけ有効にする案**: テスト専用の
     「実装」は実装とは呼べず、レビューで否認される見込みが高い。
  3. **承認済みとみなして `pub` で公開する案**: 契約違反のため不可。
- **本 PR で変えないもの**: 本番コード（`compat/{sequential,training,
  mod}.rs`・`interop/*.rs`・`model.rs`）のロジック、`Cargo.toml`、
  tolerance・baseline、ガードレール閾値、`docs/spec/`、autodiff の
  optimizer 群。

## 1. 目的・スコープ

Keras の `model.save()`／`load_model()`、PyTorch の「アーキテクチャと
重みの一括保存」に当たる機能を、`compat::Sequential` に対して追加する。
今の公開面で使えるのは `Sequential::state_dict()`／`load_state_dict()`
と `interop::safetensors` の組み合わせ（重みだけ）で、読み込む側で
同じ層構成を手で組み直す必要がある。

イシューの要件（要約）:

1. 層構成のメタデータ（層の種類・層数・パラメータキー）を JSON の
   manifest として記録する。
2. safetensors と組み合わせ、パラメータを bit 同一で roundtrip させる。
3. `compile()` の設定・状態（optimizer・loss・AMP）も復元する。
4. 深いネットワークや skip connection を含む複雑な構成でも save／load
   後に bit 完全一致する。

スコープ外（イシュー規定）: ONNX／TorchScript への export・
version migration。

## 2. 承認事項（未承認・一覧）

1. **公開 API の署名とエラー型**。
   - `fandhe_ai::compat::save_model(model: &Sequential, dir: impl AsRef<Path>) -> Result<(), ModelIoError>`
   - `fandhe_ai::compat::load_model(dir: impl AsRef<Path>) -> Result<Sequential, ModelIoError>`
   - `#[non_exhaustive] pub enum ModelIoError { Io(std::io::Error), Manifest { message: String }, Safetensors(String), UnsupportedModel { reason: String }, Mismatch { message: String }, Autodiff(AutodiffError), TooLarge { what: &'static str, limit: u64 } }`
   - 代替案として `Sequential::save(&self, dir)`／`Sequential::load(dir)`
     （inherent メソッド）も併記する。
2. **内部クレートへの追加**（facade 公開面は広がらない）。
   - `impl OptimizerStateDict for Sgd`（`crates/autodiff/src/optim/sgd.rs`）。
     `Sgd` は `velocity: Option<Vec<Tensor>>` を持つが `OptimizerStateDict`
     未実装（`docs/autodiff-optimizer-state-dict-decision.md`）。
     `decode_state_dict` は現状 `step_count` を必須とするため、`Sgd`
     （`step_count` を持たない）向けの引数化・専用デコーダが要る。
   - **`Lbfgs` 専用の状態保存・復元 API**（PR #2319 で `main` に統合された
     L-BFGS 対応〈イシュー #2197・#2172〉により §2188 着手後に追加された
     7 番目の `Optimizer` variant。§11「棚卸し（A）」参照）。`Lbfgs` は
     `n_iter`／`func_evals`／`d`／`t`／`old_dirs`／`old_stps`／`ro`／
     `h_diag`／`prev_flat_grad`／`last_loss`／`slot_shapes` を保持するが
     `OptimizerStateDict` 未実装で、かつ既存トレイトが前提とする
     「per-param スロットバッファ」形状（`AdamW` の `m`／`v` 等）とは構造が
     異なる（`Lbfgs` はフラット化した全パラメータ 1 本のベクトルに対する
     大域状態＋曲率ペアの履歴〈`history_size` 上限〉を保持する）。承認後は
     `Lbfgs` 専用のキー配置を持つ `OptimizerStateDict` 実装（または同型の
     専用トレイト）を追加する。復元時の検証: `old_dirs.len() ==
     old_stps.len() == ro.len() かつ <= history_size`・`d.len()`／
     `prev_flat_grad.len()`（`Some` の場合）／各 `old_dirs`／`old_stps`
     要素の長さが `slot_shapes` の要素数合計と一致・`t`／`h_diag`／
     `ro` の各要素・`d`／`old_dirs`／`old_stps`／`prev_flat_grad` の全要素が
     有限であること。
3. **GradScaler の状態復元コンストラクタ**（PR #2317 review 指摘 1 の是正。
   §11「棚卸し（A）」参照）。`GradScaler::new` は `init_scale` からしか
   開始できず、`update` は backoff／growth の状態機械を経由するため、
   任意の `(scale, growth_tracker)` の組を事後に再現できない
   （`growth_tracker` は growth／backoff のたびに `0` へリセットされる
   ため、`update(false)` を再生しても再生前に backoff／growth で変化
   済みの `scale` 自体は動かない）。承認後は次の内部 API（`fandhe_ai_
   autodiff` 限定。facade へは再エクスポートしない）を追加する:
   `GradScaler::from_state(config: GradScalerConfig, scale: f32,
   growth_tracker: u64) -> Result<GradScaler, AutodiffError>`。検証は
   `new`（`config` の各フィールド）に加え、`scale` が有限・正・非正規化
   数でないこと（`update` の backoff 検証と同一基準）、`growth_tracker
   < config.growth_interval`（`growth_tracker` は `growth_interval` に
   到達すると必ず growth し `0` へリセットされる仕様〈`amp.rs::
   GradScaler::update`〉のため、正常な状態機械が到達できる範囲は
   `0..growth_interval` のみで、これを超える値は改竄・破損の兆候として
   fail-closed に拒否する）。
4. **上限値の新設**。manifest は 1 MiB、層数は 4096、Lbfgs の履歴
   ペア数（`old_dirs`／`old_stps`／`ro` の長さ）は各 optimizer の
   `history_size` 設定値（`LbfgsConfig::history_size` は `>= 1` のみ
   検証済みで上限がないため、manifest 側で別途 DoS 対策の上限——例えば
   `65536`——を設けるかは要承認）。safetensors は model registry の
   `MAX_MODEL_FILE_BYTES`（1 GiB）を共用するか独自定義するか。
5. **受入基準からの逸脱 2 件**。ZIP ではなくディレクトリ内 2 ファイル
   （`manifest.json`＋`model.<gen>.safetensors`。世代 ID 付きファイル名の
   理由は §12 を参照）にする。skip connection は `compat::Sequential`
   が任意の分岐を表現できないため、`add_transformer_encoder`（内部
   residual）で代替する。

## 3. 調査で判明した事実（設計の制約）

| 事実 | 出典 | 設計への影響 |
|---|---|---|
| facade は `serde`／`serde_json` に依存していない | `crates/facade/Cargo.toml` | manifest JSON は手書きの厳格パーサ（固定スキーマ）で扱う。ZIP 用クレートもないため ZIP は不採用 |
| `Sequential` は層を `Box<dyn Module>` で持つ。`Module` のダウンキャストは不完全 | `crates/facade/src/compat/sequential.rs`・`crates/autodiff/src/nn/module.rs` | 各 `add_*`（30 メソッド）の引数を内部 `LayerSpec` enum として記録する方式にする |
| BatchNorm の running stats は `state_dict()` に含まれない。setter がなく、`BatchNorm1d/2d::from_parameters` でのみ復元できる | `crates/autodiff/src/nn/batch_norm.rs` | buffer は safetensors に別キーで保存し、BN 層は `from_parameters` で再構築する。`num_batches_tracked` は非復元（forward 計算に一切使われないカウンタのみで、非復元でも数値へ影響しない。§11 参照） |
| `Compiled { optimizer, loss, amp }`。7 optimizer（`Sgd`／`AdamW`／`Adam`／`RmsProp`／`Adagrad`／`Lamb`／`Lbfgs`。`Lbfgs` は PR #2319〈main 統合済み〉で追加）はいずれも `config()` を持ち、`set_lr` による書き換えも `config()` へ反映済みの値を返す | `crates/facade/src/compat/training.rs` | 設定値（LR scheduler が書き換えた現在値を含む）は全フィールドを直列化できる |
| `AdamW`／`Adam`／`RmsProp`／`Adagrad`／`Lamb` は `OptimizerStateDict` 実装済み。`Sgd`／`Lbfgs` は未実装 | `crates/autodiff/src/nn/optim/state_dict.rs`・`crates/autodiff/src/nn/optim/lbfgs.rs` | 承認後は `impl OptimizerStateDict for Sgd`、および `Lbfgs` 専用のキー配置を持つ状態保存・復元 API が必要（§2 item 2） |
| `api_surface.rs::workspace_declares_optimizer_state_dict_fn_names_only_in_allowed_locations` が `state_dict`／`load_state_dict` の宣言元を完全一致で固定している | `crates/facade/tests/api_surface.rs` | 承認後の facade 側ヘルパーにこの名前は使えない |
| `GradScaler` に `config()` はあるが、`(scale, growth_tracker)` を任意の値に復元するコンストラクタがない（`update` の backoff／growth 経由でしか変化しない） | `crates/autodiff/src/nn/optim/amp.rs` | `compile_with_amp` の時点で `GradScalerConfig` を記録し、save 時点の `scale()`／`growth_tracker()` を manifest に保存、復元は承認後の `GradScaler::from_state`（§2 item 3）を使う（`update` の再生では現在の `scale` を再現できないため。§11） |
| `Optimizer::Lbfgs` は `compile_with_amp` から fail-closed に拒否される（AMP 非対応） | `crates/facade/src/compat/training.rs:899` | 検証計画（§6）の「optimizer × AMP」の直積対象から `Lbfgs` を除外する |
| `compat::Sequential` では任意の skip connection を表現できない | `sequential.rs` | 受入基準の「skip connection」は深い異種スタック＋`add_transformer_encoder` で代替 |
| `save_model`／`load_model` を宣言している `crates/*/src/` はない | `grep` 結果（2026-09-26） | 定義元インベントリの期待集合は空 |

## 4. ファイル形式（形式バージョン 1）

- `dir/model.<gen>.safetensors`（F32 のみ）。**世代 ID 付きファイル名**
  （`<gen>` は 32 文字の 16 進数。固定名 `model.safetensors` から変更した
  理由は §12「保存の世代コミット方式」を参照——PR #2317 review 指摘 2 の
  是正）。キー名前空間 3 種:
  - パラメータ: `{i}.{name}`（`Sequential::state_dict` と同じ）
  - BN の buffer: `{i}.running_mean`／`{i}.running_var`
  - optimizer 状態: `optimizer.` 接頭辞＋`OptimizerStateDict::
    state_dict()` のキー（`Sgd`）、または `Lbfgs` 専用キー配置
    （`optimizer.n_iter.u64_u16x4`・`optimizer.func_evals.u64_u16x4`・
    `optimizer.d`・`optimizer.t`・`optimizer.h_diag`・
    `optimizer.prev_flat_grad`〈`Option` は有無をキー存在で表す〉・
    `optimizer.last_loss`〈同前〉・`optimizer.history.{i}.s`／`.y`
    （`old_stps`／`old_dirs`）・`optimizer.history.rho`（`ro`。履歴件数分
    まとめて 1 テンソル）。history の件数は manifest 側の
    `optimizer.history_len` で独立に記録し、safetensors のキー集合と
    突き合わせる——既存 `OptimizerStateDict` の `num_slots` と同じ設計
    〈state_dict.rs「キー配置」節〉）
- `dir/manifest.json`:
  ```json
  {
    "format": "fandhe-ai.compat.sequential",
    "format_version": 1,
    "training": false,
    "num_layers": 3,
    "layers": [{"index": 0, "kind": "linear", "params": {}}],
    "parameter_keys": [{"key": "0.weight", "shape": [8, 4]}],
    "buffer_keys": [],
    "safetensors_file": "model.0123...cdef.safetensors",
    "safetensors_bytes": 4096,
    "compiled": null
  }
  ```
  `safetensors_file`／`safetensors_bytes` は §12 の世代コミット方式が
  load 側の世代不一致検出に使う（`safetensors_file` は
  `model.<32桁16進>.safetensors` の完全一致パターンのみ許可し、パス
  区切り文字を含む値は即 `Err` とする——ディレクトリ脱出の防止）。
  `compiled` は `{"loss", "optimizer": {"kind", "config"},
  "optimizer_state_keys", "amp"}`。`optimizer.kind` は
  `"sgd"`／`"adamw"`／`"adam"`／`"rmsprop"`／`"adagrad"`／`"lamb"`／
  `"lbfgs"` の 7 種（文字列 allowlist。未知の値は `UnsupportedModel`）。
  `"lbfgs"` の `config` は `LbfgsConfig` の全フィールドを書く。うち
  `max_eval`（`Option<usize>`）は `null` またはその他は非負整数
  （手書きパーサは JSON の `null` リテラルを明示的に扱う——本形式の他の
  スカラーフィールドは今のところ `Option` を持たないため、`null`
  受理はこのフィールド専用の分岐になる）、`line_search` は
  `"none"`／`"strong_wolfe"` の文字列 allowlist（facade は現状
  `LbfgsLineSearch` を再エクスポートしないため既定の `"none"` のみが
  生成されるが、パーサ自体は両方の値を受理できるようにする——将来の
  再エクスポート解禁時に manifest フォーマットを変更せずに済むため）。
  `amp` は `null` または `{"dtype", "grad_scaler_config", "scale",
  "growth_tracker"}`。`scale`（f32）・`growth_tracker`（u64。JSON 整数）
  は save 時点の `GradScaler::scale()`／`growth_tracker()` の**現在値**
  であり（`GradScalerConfig` の初期値ではない）、復元は承認後の
  `GradScaler::from_state(config, scale, growth_tracker)`（§2 item 3）を
  使う（PR #2317 review 指摘 1 の是正。§11 参照）。
- 数値表現: f32 は Rust の最短往復表記（`{:?}`）で書き `str::parse::<f32>`
  で読む（非有限値は save 時に `UnsupportedModel` で拒否）。u64／usize は
  JSON の整数として書き独自パーサで読む。

## 5. 意味論

- **LayerSpec**: compat 内部に `LayerSpec` enum を置き、`add_*` 30 種と
  1 対 1 対応させる（引数・seed を保持）。`Sequential` に private
  フィールド `specs: Vec<LayerSpec>` を追加し、各 `add_*` で push する
  （フィールドは private なので公開 API は非破壊）。
  `specs.len() != layers().len()` なら `UnsupportedModel`。
- **save の手順**（世代コミット方式。詳細は §12）: 検証をすべて終えて
  から書き込みに入る。`create_dir_all` → 世代 ID を採番し
  `model.<gen>.safetensors` へ既存 `save_safetensors_f32`（一時ファイル
  ＋`rename`）で書く → その `<gen>`・実バイト数を含む `manifest.json` を
  一時ファイル＋`rename` で書く（**この manifest の rename が唯一の
  コミット点**。safetensors 側は世代 ID が一意なため上書きされることが
  なく、manifest がそれを参照した時点で既に完全な内容で存在する） →
  manifest rename 成功後、参照されていない旧世代の
  `model.*.safetensors` を best-effort で削除する（§12.3 手順 5）。
- **load の手順**: manifest をサイズ上限付きで読み厳格パース →
  `format`／`format_version` 完全一致確認 → `safetensors_file` の
  パターン検証 → 参照された `model.<gen>.safetensors` を開き実バイト数を
  `safetensors_bytes` と照合（不一致は `Mismatch`。§12.3 手順 4）→
  safetensors を上限付きで読む → 3 種のキー集合とファイル内容の完全
  一致・shape 一致を確認 → spec 順に層を構築（BN のみ
  `from_parameters`）→ `load_state_dict` → `set_training` → `compiled`
  があれば optimizer 復元（`Sgd`／`Lbfgs` は承認後の専用復元 API）・
  AMP があれば `GradScaler::from_state` で scaler を復元。途中失敗時は
  部分的に構築した `Sequential` を返さない。
- **非復元のもの**: BN の `num_batches_tracked`（forward 計算に使われない
  カウンタのみで数値へ影響しない。§11）、Dropout の RNG 状態（グローバル
  RNG。インスタンスに保持されない）、LR scheduler・callbacks・param
  groups（`Compiled`／`Sequential` に保持されない `fit` 呼び出し引数。
  ただし LR scheduler が書き換えた**現在の** LR 自体は各 optimizer の
  `config()` 経由で復元される）、勾配累積バッファ（`run_fit` 内ローカル
  変数で epoch／ウィンドウ境界を跨いで持ち越されない）。網羅的な棚卸しは
  §11 を参照。

## 6. 承認後の検証計画

`crates/facade/tests/compat_sequential_model_io.rs` で、30 種すべての
層を含むモデル・深い異種スタック・transformer encoder・train モード
後の BN running stats・6 optimizer（`Sgd`／`AdamW`／`Adam`／`RmsProp`／
`Adagrad`／`Lamb`）× AMP の有無の組み合わせ、および `Lbfgs`（AMP は
`compile_with_amp` 側で fail-closed 拒否されるため AMP なしの組み合わせ
のみ。§3 参照）が save／load 後に bit 完全一致することを検証する。
GradScaler は複数 step の backoff／growth を経由させ「初期値と異なる
`(scale, growth_tracker)`」の状態で save／load しても一致することを
検証する（従来の再生方式では検出できなかった不具合。§11）。`Lbfgs` は
複数 outer step 実行後（履歴が `history_size` 未満・到達済みの両方）の
save／load 一致を検証する。manifest の改竄（未知キー・版違い・層数
超過・深さ超過・サイズ超過・非有限値・キー集合不一致・shape 不一致・
`safetensors_file` のパターン違反・パス区切り混入・`safetensors_bytes`
不一致）はすべて fail-closed に拒否されることを確認する。§12 の crash
consistency（中断・既存ディレクトリへの再保存・旧世代の残存）も
ファイルシステム操作を模した統合テストで検証する。
CUDA／Metal 実機 parity は対象外（ホスト側 I/O のみでカーネルを
持たないため）。

## 7. 保留ガードの多層構成

- **正のプローブ doctest**（`crates/facade/src/lib.rs::
  ModelIoHoldDoctestGuard`）: facade の全 `pub mod` を glob import した
  スコープで、`model_io` モジュール・`save_model`／`load_model` 自由
  関数・`ModelIoError` 型・`Sequential` への inherent メソッドの 4 経路
  いずれで公開されても glob 衝突または型不一致でコンパイル失敗する。
- **ソース走査**（`crates/facade/tests/api_surface.rs::
  facade_does_not_reexport_or_declare_model_io`）: facade src 全体に
  `model_io`／`save_model`／`load_model`／`ModelIoError` の再エクス
  ポート・独自宣言がないことを固定する。
- **定義元インベントリ**（同ファイル::
  `workspace_declares_model_io_fn_names_only_in_allowed_locations`）:
  workspace 全体（`crates/*/src/`）で `save_model`／`load_model` の
  `fn` 宣言が 0 件であることを固定する。§2 代替案（`Sequential::save`／
  `Sequential::load`）向けには、同ファイル::
  `workspace_declares_sequential_alt_save_load_fn_names_only_in_allowed_
  locations` が対応する（PR #2317 review 再確認で追加。`Sequential` を
  含む `impl` ヘッダ配下に限定した `fn save`／`fn load` 宣言を workspace
  全体〈`crates/*/src/`〉へ横断走査し 0 件を固定する。ソース走査層が
  `crates/facade/src/**` に限定できるのは型の一意性ではなく依存方向
  ゆえ〈`fandhe-ai`〈facade〉package に依存する workspace クレートが
  存在しない〉ためであり、ワークスペース全体を横断する定義元
  インベントリという第 3 層の役割は別途この専用テストが担う。詳細は
  `crates/facade/tests/api_surface.rs::scan_sequential_alt_save_load_impls`
  のドキュメンテーションコメントを正とする）。
- 承認後は、doctest・ソース走査を撤去して正ガード（実際の公開面の
  固定テスト）へ置き換え、インベントリの期待集合を
  `facade/src/compat/model_io.rs` へ差し替える。

## 8. OWASP Top 10 観点（承認後の要件として記録）

- **A03 インジェクション／非信頼入力**: manifest.json と
  model.<gen>.safetensors は非信頼の外部フォーマットとして扱う。バイト数
  （`take(cap + 1)`）・JSON の深さ・層数・`num_slots`・`Lbfgs` の履歴件数
  のすべてを、確保・走査の前に検証する。固定スキーマに対し未知
  キー・重複キーを拒否し、キー集合と shape は完全一致させる（無言
  skip 禁止。REQ-7）。`safetensors_file` はパス区切り文字を含まない
  固定パターン（`model.<32桁16進>.safetensors`）の完全一致のみ許可し、
  ディレクトリ脱出（`../`・絶対パス等）を拒否する。パスは `std::fs` へ
  そのまま渡し、シェル展開・ユーザー入力の連結はしない。
- **A08 ソフトウェア・データ整合性**: 世代 ID 付き safetensors ファイル
  （一意名・上書きされない）＋一時ファイル＋`rename` で書き込み、
  manifest の rename を「2 ファイルにまたがる」唯一のコミット点とする
  （固定ファイル名への単純な rename だけでは、safetensors の rename
  完了後〜manifest の rename 完了前の窓で旧 manifest と新 safetensors が
  共存し不整合な組を読み得るため、世代 ID で解消する。§12。PR #2317
  review 指摘 2 の是正）。load は世代 ID・バイト長の照合を含む検証が
  すべて通ってから構築し、失敗時は部分的な `Sequential` を返さない。
  optimizer の種別マーカーを照合し取り違えを fail-closed にする。
- **A04 安全でない設計**: 公開面の拡張を承認前に実施しない（保留
  ガードで機械的に固定）。依存の追加（`serde_json`・`zip` 等）はせず
  `Cargo.toml` も不変。
- **本 PR 自体**: 本番コードの挙動は変わらない。秘密情報は扱わない。

## 9. 非信頼データに関する記録

イシュー本文に、指示の上書きや秘密情報の出力といった命令文は見当たら
なかった。本文は要件としてのみ扱い、逐語での引用はしていない。

## 10. 再開条件

イシュー #2188（または親 #2131）に、所有者による §2 の承認コメントが
付くこと。承認後は、別イシューか同イシューの再開で「§4〜§6 の実装 →
保留ガードの撤去」を 1 PR で行う。

**保留ガードの適用範囲の是正（PR #2317 review 指摘）**: 当初の
`ModelIoHoldDoctestGuard`・`api_surface.rs` のソース走査は §2 の主案
（`save_model`／`load_model`）のみを検出対象としており、同じ §2 が
併記する代替案 `Sequential::save(&self, dir)`／`Sequential::load(dir)`
（`_model` 接尾辞なしの inherent メソッド）を検出できていなかった。
指摘を受け、正のプローブ（`__FandheModelIoHoldProbe` トレイトへの
`save`／`load` メソッド追加）とソース走査（`impl Sequential { .. }`／
`impl <Trait> for Sequential { .. }` ブロック内の `fn save`／`fn load`
宣言を検出する `scan_sequential_alt_save_load_impls`）の双方を拡張し、
§2 の 2 案いずれが未承認のまま追加されても保留ガードが検出する状態に
是正した。

**状態復元契約・保存の原子性の是正（PR #2317 review 再確認・
2026-09-27）**: 2 件の P2 指摘を受け、設計を是正した。(1) `GradScaler`
の `growth_tracker` は成長／backoff のたびに `0` へリセットされるため、
`new` の後 `update(false)` を再生する当初案では、再生前に backoff／
growth で変化済みの `scale` を再現できないことが判明した——§1「compile
状態復元契約」に反する。是正として、save 時点の `scale`／
`growth_tracker` の**現在値**を manifest に保存し、承認後に検証付きの
`GradScaler::from_state` コンストラクタ（§2 item 3）を新設して復元する
方式へ変更した。(2) 既存ディレクトリへの再保存で、safetensors の
rename 完了後〜manifest の rename 完了前の窓に旧 manifest と新
safetensors が共存し得ることが判明した——manifest を最後に書くだけ
では 2 ファイルにまたがるコミットマーカーとして機能しない。是正として
safetensors ファイル名を世代 ID 付き（`model.<gen>.safetensors`）にし、
manifest の rename のみを唯一のコミット点とする方式（§12）へ変更した。
また、この再確認の過程で main へ統合された L-BFGS 対応（PR #2319・
イシュー #2197・#2172）により `Optimizer` の 7 番目の variant `Lbfgs`
が追加されており、`Lbfgs` も `GradScaler` と同型の「公開 API だけでは
状態を復元できない」ケースに該当することが判明したため、同じ方針
（現在値を保存し、検証付き復元 API を承認後に追加する）で設計へ含めた
（§2 item 2 拡張）。状態復元契約の全数棚卸しは §11、保存の原子性の
全設計は §12 を参照。

**再開条件の再確認時の追加是正**: 上記の是正直後は、3 層目（定義元
インベントリ）が `save_model`／`load_model` の 2 名のみを workspace
全体で検査しており、代替案の `save`／`load` は 2 層目（`crates/facade/
src/**` 限定のソース走査）にしか反映されていなかった。ソース走査層の
コメントにも「`Sequential` 型はこのワークスペースでは facade にのみ
定義される」という誤った理由づけが残っていた（実際には
`crates/autodiff/src/compat/sequential.rs`・`crates/autodiff/src/nn/
container.rs` にも同名の別型 `Sequential` が存在する。ただしいずれも
facade から再エクスポートされない内部専用型で本イシューの対象外）。
正しい理由は依存方向（`fandhe-ai`〈facade〉package に依存する
workspace クレートが存在しないため、facade 外から公開型
`compat::Sequential` を名指しできない）である。この誤りを是正した
うえで、`workspace_declares_sequential_alt_save_load_fn_names_only_in_
allowed_locations`（`crates/facade/tests/api_surface.rs`）を新設し、
`scan_sequential_alt_save_load_impls` を workspace 全体（`crates/*/
src/`）へ適用して 3 層目にも代替案を横展開した。

## 11. 状態復元契約の棚卸し（A。PR #2317 review 指摘 1 の是正に伴う全数確認）

§1 の「compile 状態復元契約」（層構成・重み・optimizer・loss・AMP を
save/load で bit 完全に往復させる）に対し、manifest／safetensors へ
現れうる全状態について、現行の内部 API（`crates/autodiff`・
`crates/facade`）で「保存できるか」「保存した値から bit 一致で
復元できるか」を実コードで確認した結果を次表に示す（2026-09-27・
main への PR #2319〈L-BFGS〉統合後の状態）。

| 状態 | 保存可否（getter） | 現行 API のみで bit 一致復元可能か | 対応方針 |
|---|---|---|---|
| 重み（各層 weight／bias 等） | 可（`Module::state_dict()`） | 可（`load_state_dict()`） | 変更なし |
| BatchNorm `running_mean`／`running_var` | 可（buffer） | 可（`BatchNorm1d/2d::from_parameters`） | 変更なし |
| BatchNorm `num_batches_tracked` | 可（`num_batches_tracked()`。crate 内限定） | 不可（`from_parameters` に対応引数がなく setter もない） | **意図的に非復元のまま**。forward 計算のどこからも参照されず（`batch_norm.rs:395-396` で加算されるだけで、読み出し箇所は同ファイルの getter とテストのみ）、数値へ影響しないため復元 API は追加しない |
| `AdamW`／`Adam`／`RmsProp`／`Adagrad`／`Lamb`／`Adadelta`／`Adamax`／`NAdam`／`RAdam` の内部状態 | 可（`OptimizerStateDict::state_dict()`） | 可（`load_state_dict()`。検証付き） | 変更なし |
| `Sgd` の `velocity` | 不可（`OptimizerStateDict` 未実装） | 不可 | §2 item 2（承認後 `impl OptimizerStateDict for Sgd`） |
| `Lbfgs` の `n_iter`／`func_evals`／`d`／`t`／`old_dirs`／`old_stps`／`ro`／`h_diag`／`prev_flat_grad`／`last_loss`／`slot_shapes` | 一部可（`n_iter()`／`func_evals()`／`last_loss()`／`config()` のみ公開） | 不可（他フィールドに setter がなく、`OptimizerStateDict` も未実装。実装するとしても既存トレイトが前提とする per-param スロットバッファ形状〈`AdamW` の `m`／`v` 等〉とは構造が異なる〈フラット化ベクトル 1 本＋曲率ペア履歴〉） | §2 item 2 拡張（承認後、`Lbfgs` 専用キー配置の状態保存・復元 API を新設。§4「Lbfgs 状態」節） |
| `GradScaler` の `scale`／`growth_tracker` | 可（`scale()`／`growth_tracker()`） | **不可**（`new` は `init_scale` からしか開始できず、`update` は backoff／growth の状態機械経由でしか変化しない。`growth_tracker` は成長／backoff のたびに `0` へリセットされるため、`update(false)` を事後に何回再生しても、再生前の backoff／growth で変化済みの `scale` 自体は再現できない——指摘 1 の対象） | §2 item 3（承認後 `GradScaler::from_state(config, scale, growth_tracker)` を新設。検証は `new` 相当＋`scale` の非正規化数チェック＋`growth_tracker < growth_interval`） |
| LR scheduler・callbacks | 該当なし（`Compiled`／`Sequential` に保持されない `fit` 呼び出し引数） | — | 非対象（スコープ外のまま） |
| LR scheduler が書き換えた**現在の** LR | 可（各 optimizer の `config()` が `set_lr` 後の値を返す。`AdamW`／`Adam`／`Sgd`／`Adadelta`／`Adamax`／`NAdam`／`RAdam`／`Lbfgs` で確認） | 可（`config` を保存・復元するだけでよい。`RmsProp`／`Adagrad`／`Lamb` は `set_lr` 自体がないため常に既定値のまま） | 変更なし |
| param groups | 該当なし（facade は単一グループのみ。イシュー #2173 が facade 公開面拡張として別途保留中） | — | 非対象 |
| 勾配累積バッファ（`accumulate_tests` 内 `acc_buf` 相当） | 該当なし（`Sequential::run_fit` 内のローカル変数。`micro == config.accumulate_steps` またはエポック境界で必ず flush され、`Sequential`／`Compiled` へ持ち越されない。`training.rs:1470-1472` 該当コメント参照） | — | 非対象 |
| `Loss`（loss/metrics 設定） | 可（`Copy`＋`Eq` の enum） | 可（バリアント復元のみ） | 変更なし |
| Dropout の RNG 状態 | 該当なし（`Dropout`／`Dropout2d` 構造体に RNG フィールドがなくグローバル RNG を都度消費する。`dropout.rs` 確認） | — | 非対象（既存記載どおり） |
| `training`（train／eval モード） | 可（`NnSequential.training: bool`） | 可（`Module::set_training`） | 変更なし |
| カスタム学習 step フック（`train_step_fn`） | 該当なし（`Compiled`／`Sequential` に保持されない `fit` 呼び出し引数の関数ポインタ／クロージャ） | — | 非対象（関数値はそもそもシリアライズ不能） |

上表のとおり、「公開 API の再生だけでは復元できない」状態は
`GradScaler`（指摘 1 の対象）に加え `Sgd`・`Lbfgs` の 3 者であり、
いずれも同じ方針（現在値を保存し、範囲・有限性を検証する復元 API を
承認後に内部追加する）で§2 に反映した。それ以外の状態は、既存の
`state_dict`／`load_state_dict`／`from_parameters`／`config`／
`set_training` 等の公開・内部 API の組み合わせだけで bit 一致復元が
可能、または（BN の `num_batches_tracked`・グローバル RNG・LR
scheduler 本体・param groups・勾配累積バッファ・カスタム学習 step
フック）意図的に非対象と判断できることを確認した。

## 12. 保存の世代コミット方式（B。PR #2317 review 指摘 2 の是正）

### 12.1 指摘の要約

当初案（§5「save の手順」旧版）は固定ファイル名 `model.safetensors`／
`manifest.json` それぞれに独立して一時ファイル＋`rename` を行うだけ
だった。既存ディレクトリへの再保存では、safetensors の rename が
完了してから manifest の rename が完了するまでの間、**新しい
safetensors ファイルと古い manifest が同一ディレクトリに共存する**。
この窓の間に load すると、manifest（旧世代の層構成／パラメータキー
定義）と safetensors（新世代の重み）という不整合な組み合わせを読み
得る。「manifest を最後に書く」だけでは 2 ファイルにまたがる
コミットマーカーとして機能しない。

### 12.2 crash consistency の全ウィンドウ

- 新規保存（ディレクトリが空）: 単一世代のみのため問題なし。
- **既存ディレクトリへの再保存**（指摘 2 の本体）。
- 保存の中断（safetensors rename 後・manifest rename 前にプロセスが
  落ちる・kill される）。
- 同時保存（2 プロセスが同一ディレクトリへ同時に `save_model` する）。
- 読込中の書込み（load がマニフェストを読んでいる最中に別プロセスが
  save する）。
- 部分書込み（tmp ファイルへの書き込み自体がディスクフル等で失敗）。
- 一時ファイル残骸（rename 前にプロセスが落ちた場合の `*.tmp` の残存）。
- ディレクトリ fsync・電源断耐性（rename 自体の永続化）。
- プラットフォーム差（rename の原子性）。

### 12.3 方式（世代 ID 付き safetensors ファイル名 + manifest 参照）

1. **safetensors ファイル名を世代ごとに一意化する**: 固定名
   `model.safetensors` をやめ、`model.<gen>.safetensors`（`<gen>` は
   32 文字の 16 進数。`std::process::id()`・`SystemTime::now()` の
   ナノ秒・プロセス内 `AtomicU64` カウンタを連結してハッシュ化した値。
   衝突しても本手順内で再試行して検出できるため暗号学的乱数は不要）。
   既存 `save_safetensors_f32(path, ...)`（`crates/facade/src/interop/
   safetensors.rs`。同一ディレクトリへの一時ファイル書き込み＋
   `rename` を既に実装済み）を、**世代付きファイル名を `path` として
   渡してそのまま呼び出す**（新規の低レベル I/O コードを追加しない。
   REQ-1「自作コアの上の薄いラッパー」方針に沿う）。rename 先の
   `model.<gen>.safetensors` が既に存在する場合（衝突。天文学的に
   低確率）は新しい `<gen>` を再生成して再試行する（上限 5 回。それ
   でも衝突する場合は `ModelIoError::Io` で fail-closed）。
2. **manifest に世代情報を追加する**（§4 のスキーマへ
   `"safetensors_file"`〈文字列。`model.<32桁16進>.safetensors` の
   完全一致パターンのみ許可——パス区切り文字を含む値は load 側で即
   `Err`〉と `"safetensors_bytes"`〈u64。保存直後に実際に書き込んだ
   バイト数〉を追加）。manifest 自体は固定名 `manifest.json` のまま、
   既存どおり一時ファイル＋`rename` する。
3. **manifest の rename が唯一のコミット点である根拠**: 手順 1 の
   safetensors rename が完了した時点で、その世代のファイルは完全な
   内容で存在し、かつ**同名で上書きされることが二度とない**（世代 ID
   が一意なため）。したがって manifest の rename が成功した時点
   （または、それより前に古い manifest が指す旧世代を読んだ時点）の
   いずれでも、load は必ず「manifest が指す safetensors ファイルが
   完全な内容で存在する」という不変条件を得る。§12.2 の「新規・
   再保存・中断」の 3 ウィンドウはこれで閉じる。
4. **load 側の世代不一致検出**: manifest の `safetensors_file` を
   読んだ後、完全一致パターン検証（`model.` ＋ 16 進 32 文字 ＋
   `.safetensors`。パス区切りなし）を行ってからそのファイルを開く。
   開いたファイルの実バイト数を `safetensors_bytes` と比較し、
   不一致なら `ModelIoError::Mismatch`（他プロセスによる再保存で
   ファイルが差し替えられた・部分書込みの残骸を掴んだ、等の可能性を
   fail-closed に拒否する）。バイト数一致後は既存どおり内部のキー
   集合・shape 検証に進む（§5「load の手順」相当）。
5. **旧世代の後片付け（best-effort）**: manifest rename 成功後、
   ディレクトリを 1 度だけ `read_dir` し、パターン
   `model.*.safetensors` に一致し、かつ今回コミットした
   `safetensors_file` と異なるファイルを削除する。列挙件数に上限
   （例: 4096 エントリ）を設け、上限に達したら以降の削除を諦めて
   成功のまま返す（巨大ディレクトリでの走査コストを無限に増やさない
   fail-open な best-effort。削除自体の失敗〈他プロセスが開いている
   等〉も無視する。`crates/onnx-interop/src/st_save.rs`「rename 失敗時
   の tmp ファイル削除は best-effort」と同じ扱い）。
6. **同時保存（2 プロセス）の扱い**: 本設計は「同一ディレクトリへの
   並行 `save_model` はサポート対象外」と明示する（`st_save.rs:185`
   の既存の未サポート表明と同型）。世代 ID が一意である限り、2 つの
   並行 save は互いの safetensors ファイルを破壊しないが、manifest
   （固定名）は最後に rename した側が勝つ。片方の save が書いた世代
   ファイルが、もう片方の cleanup（手順 5）で削除された直後にその
   manifest の rename が成功すると、存在しないファイルを指す壊れた
   状態になり得る。**この残余リスクは「非サポート」として受容する**
   （呼び出し元が同一ディレクトリへ並行書き込みしない前提での動作を
   保証する）。
7. **読込中の書込み（reader/writer race）**: load は「manifest を
   読む → 参照されたファイルを開く（`File::open`。POSIX では open 後
   に他プロセスが unlink してもオープン済み fd の読み取りは継続できる
   ため、cleanup〈手順 5〉が read 直後に走っても既にオープン済みの
   読み取りは壊れない）」の順で処理する。ただし「manifest を読んでから
   実際に open するまでの間」に別プロセスがさらに新しい世代へ進み、
   load が読んだ世代が cleanup 対象になる場合（load と並行する 2 回目
   以降の save）は open が `ENOENT` で失敗し得る。**この残余レースも
   手順 6 と同じ「同一ディレクトリへの並行 save／load は非サポート」の
   受容範囲に含める**（単一プロセスが読み書きする通常運用では発生
   しない。並行アクセスは呼び出し元でファイルロック等の直列化を行う
   前提とする）。
8. **プラットフォーム（rename の原子性）**: 本設計は既存
   `save_safetensors_f32`／`st_save.rs` が既に前提とする「同一
   ファイルシステム内の `rename` は POSIX 上 atomic」という前提を
   そのまま踏襲し、Windows 向けの特別分岐は設けない（`ModelRegistry`
   の `load`／`available_models` のようにシンボリックリンク脱出防止の
   理由で明示的に Windows を fail-closed 拒否している既存決定
   〈`docs/facade-model-registry-decision.md` §13〉とは異なり、単一
   ファイルの `save_safetensors_f32` 自体は現状 OS で分岐していない
   ため、本設計もそれに合わせる——新規に導入する不整合ではなく既存
   方針の踏襲）。
9. **電源断耐性（fsync）は対象外のまま**: `st_save.rs` は一時ファイル
   への `fsync`（`File::sync_all`）を行わないため、rename 成功後の
   電源断・OS クラッシュに対する耐性を保証しない（既存注記のとおり。
   本設計もこれを変更しない）。本設計が閉じるのは「プロセスの通常
   終了・異常終了（クラッシュ・kill）を含む、ファイルシステムが応答
   している間の 2 ファイル間コミット順序の不整合」（指摘 2 が対象と
   した問題）であり、ストレージ層の電源断耐性という別軸の非保証は
   既存方針から変更しない。
