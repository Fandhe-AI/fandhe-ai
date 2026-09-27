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
5. **ファイル I/O のハードニング用内部 API**（facade 公開面は広がらない。
   PR #2317 review 再々確認・指摘 1・2 の是正に伴う新設。全数棚卸しは §13）。
   - **共有の no-follow リーフオープンヘルパー**: `crates/facade/src/
     model.rs` の `open_leaf_no_follow`／`open_flags` モジュール
     （`O_NOFOLLOW`／`O_NONBLOCK` の生値・Linux x86_64／aarch64・macOS
     限定）を `model.rs` 専用の非公開関数から facade 内部の共有
     ヘルパー（例: `crate::fs_guard` モジュール）へ抽出し、`model.rs`・
     新設 `model_io.rs` の双方が使う。既存呼び出し元（`model.rs`）の
     挙動は変えない（純粋な抽出。§13 参照）。
   - **`create_new` を使う一時ファイル書き込みヘルパー**（model_io 専用。
     既存 `crate::interop::safetensors::save_safetensors_f32`
     （`onnx-interop::st_save::save_safetensors_f32`）はそのまま使わない
     ——理由は §13「一時ファイル作成」節・§12.3 是正参照。代わりに
     バイト列のみを返す既存 `save_safetensors_f32_to_bytes`（副作用
     なし・変更不要）を呼び、model_io 側で
     `OpenOptions::new().write(true).create_new(true)`（Rust std のみ・
     プラットフォーム分岐不要。§13 参照）による一時ファイル作成 +
     `rename` を行う。manifest.json の一時ファイルも同じ
     `create_new` ヘルパーを使う（現行の `manifest.json` 一時ファイル
     手順を差し替える）。
6. **受入基準からの逸脱 2 件**。ZIP ではなくディレクトリ内 2 ファイル
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
  **ただしパターン一致だけでは同名のシンボリックリンク経由の脱出を
  防げない**（PR #2317 review 再々確認・指摘 1。`dir` はディレクトリごと
  非信頼として扱う——§13「信頼境界」参照）。パターン検証後、実際に
  開く際は §13 の no-follow リーフオープン手順（`fstat` による
  実体識別子照合を含む）を経由し、開いたファイルが `dir` 直下の通常
  ファイルであることを確認してから読む。`manifest.json` 自体の読み込みも
  同じ手順を経由する。
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
- **save の手順**（世代コミット方式。詳細は §12。symlink・所有権対策は
  §13）: 検証をすべて終えてから書き込みに入る。`create_dir_all` →
  読み込み専用で旧 `manifest.json` を §13 の no-follow 手順で試み読み
  （存在しない・パース不能なら「削除対象なし」として続行。手順 5 の
  削除対象決定に使う）→ 世代 ID を採番し `save_safetensors_f32_to_bytes`
  （既存・副作用なし）で得たバイト列を、§13 の `create_new` 一時ファイル
  ヘルパーで `model.<gen>.safetensors` へ書く（一時ファイル＋`rename`。
  `save_safetensors_f32` 自体は使わない——理由は §13） → その
  `<gen>`・実バイト数を含む `manifest.json` を同じ `create_new` ヘルパー
  で一時ファイル＋`rename` して書く（**この manifest の rename が唯一の
  コミット点**。safetensors 側は世代 ID が一意なため上書きされることが
  なく、manifest がそれを参照した時点で既に完全な内容で存在する） →
  manifest rename 成功後、直前に読んだ旧 `manifest.json` が指していた
  世代の `model.*.safetensors` **のみ**を、§13 の no-follow 削除手順で
  best-effort 削除する（§12.3 手順 5。パターン一致する全ファイルの削除
  ではない——削除所有権の契約は §13 参照）。
- **load の手順**: `dir` を §13 の no-follow 手順で開いた
  `manifest.json` からサイズ上限付きで読み厳格パース →
  `format`／`format_version` 完全一致確認 → `safetensors_file` の
  パターン検証 → 参照された `model.<gen>.safetensors` を同じく §13 の
  no-follow 手順で開き、開いたハンドルの実バイト数を
  `safetensors_bytes` と照合（不一致は `Mismatch`。§12.3 手順 4）→
  同じハンドルから safetensors を上限付きで読む → 3 種のキー集合と
  ファイル内容の完全一致・shape 一致を確認 → spec 順に層を構築（BN のみ
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

`crates/facade/tests/model_registry.rs` の symlink 系テスト（
`load_rejects_symlinked_leaf_file_escaping_root`・
`load_rejects_non_regular_leaf_unix_socket`・
`load_rejects_leaf_replaced_with_symlink_after_initial_write`）と同型の
テストを model_io 向けに追加し、§13 の脅威棚卸しを機械的に固定する
（PR #2317 review 再々確認・指摘 1 の是正）:
`load_model` は `manifest.json`・`model.<gen>.safetensors` のいずれかが
シンボリックリンク（`dir` 直下）である場合に拒否すること／非通常
ファイル（Unix ソケット・FIFO）である場合に拒否すること／検査後に
シンボリックリンクへ差し替えられた場合（TOCTOU）に拒否すること。
`save_model` は一時ファイル名の位置に既存のシンボリックリンク（有効・
dangling いずれも）が存在する場合に追従・上書きせず `Err` を返すこと
（`create_new` の効果を確認する）。削除所有権（§13）については、
直前の manifest が参照していた世代のみが削除され、無関係な命名規則
一致ファイル（テストが手動で作成した「よそ者」の
`model.deadbeef....safetensors`）が再保存後も残存することを検証する。

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

- **A01 アクセス制御の不備／パストラバーサル**: `dir` 配下のファイル名は
  固定文字列（`manifest.json`）または固定パターン（
  `model.<32桁16進>.safetensors`。パス区切り文字混入は即 `Err`）のみを
  扱い、それ以外の名前を `std::fs` へ渡さない。**パターン一致のみでは
  同名のシンボリックリンクによるディレクトリ脱出を防げない**ため
  （PR #2317 review 再々確認・指摘 1）、実際に開く・削除する段では必ず §13 の
  no-follow 手順（シンボリックリンク・非通常ファイルの拒否、TOCTOU 対策
  としての実体識別子照合）を経由する。`dir` はディレクトリごと非信頼
  として扱う（§13「信頼境界」）。
- **A03 インジェクション／非信頼入力**: manifest.json と
  model.<gen>.safetensors は非信頼の外部フォーマットとして扱う。バイト数
  （`take(cap + 1)`）・JSON の深さ・層数・`num_slots`・`Lbfgs` の履歴件数
  のすべてを、確保・走査の前に検証する。固定スキーマに対し未知
  キー・重複キーを拒否し、キー集合と shape は完全一致させる（無言
  skip 禁止。REQ-7）。パスはシェル展開・ユーザー入力の連結を経ずに
  `std::fs` へ渡すが、渡す前段の open 自体は上記 A01 の no-follow 手順を
  必須経路とする（単純にパスを渡すだけでは A01 の脱出を防げないため）。
- **A08 ソフトウェア・データ整合性**: 世代 ID 付き safetensors ファイル
  （一意名・上書きされない）＋ `create_new` による一時ファイル作成＋
  `rename` で書き込み、manifest の rename を「2 ファイルにまたがる」
  唯一のコミット点とする（固定ファイル名への単純な rename だけでは、
  safetensors の rename 完了後〜manifest の rename 完了前の窓で旧
  manifest と新 safetensors が共存し不整合な組を読み得るため、世代 ID で
  解消する。§12。PR #2317 review 指摘 2 の是正〈原子性〉。一時ファイル名の位置に
  攻撃者が事前配置したファイル（シンボリックリンクを含む）が存在する
  場合、`std::fs::write`（`create` + `truncate`。追従する）ではなく
  `create_new`（`O_EXCL` 相当。存在すれば symlink か否かを問わず
  `Err`）で作成することで、そこへの追従書き込み（内容の漏洩・破壊）を
  防ぐ（PR #2317 review 再々確認・指摘 1 の是正の一部。§13「一時ファイル作成」）。
  load は世代 ID・バイト長の照合を含む検証がすべて通ってから構築し、
  失敗時は部分的な `Sequential` を返さない。optimizer の種別マーカーを
  照合し取り違えを fail-closed にする。旧世代ファイルの削除は直前の
  manifest が参照していた 1 件に限定し（§12.3 手順 5・§13「削除
  所有権」）、パターン一致する無関係ファイルを削除しない（PR #2317
  review 再々確認・指摘 2 の是正）。
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

**ファイル I/O 脅威の全数棚卸しと是正（PR #2317 review 再々確認・
2026-09-27）**: 2 件の指摘を受け、設計を是正した。(1)
`safetensors_file` の値が固定パターン（`model.<32桁16進>.safetensors`）
に一致することの検証だけではディレクトリ脱出を防げない——同名の
シンボリックリンクを `File::open`（追従する）で開くと、対象ディレクトリ
外の任意ファイルを開いてしまう（P0）。是正として、`crates/facade/
src/model.rs`（`ModelRegistry`）が既に実装している no-follow オープン
（`symlink_metadata` による事前拒否 → `O_NOFOLLOW`／`O_NONBLOCK` 付き
オープン → 開いたハンドルの `fstat` による実体識別子〈`dev`／`ino`〉
照合 → サイズ上限付き読み取り）を facade 内部の共有ヘルパーへ抽出し、
`manifest.json`・`model.<gen>.safetensors` の読み込みへ適用する方式へ
変更した（§2 item 5・§13）。(2) manifest が参照する世代以外の
`model.*.safetensors` を無条件にパターン一致で削除する当初案は、
対象ディレクトリに置かれた無関係な同名パターンファイルも削除して
しまう（P2）。是正として、削除対象を「直前の manifest が参照していた
世代 1 件」に限定する方式へ変更し、クラッシュ等で孤立した世代が
削除されず残存する残余リスクを明示的に受容として記録した（§13
「削除所有権」）。この過程で一時ファイル作成手順自体にも
`std::fs::write`（追従する `create` + `truncate`）を使っていた欠陥が
判明したため（一時ファイル名の位置への事前配置シンボリックリンクへの
追従書き込み）、`create_new`（Rust std のみ・全対象プラットフォーム
共通）へ是正した（§2 item 5）。ファイル I/O 全般（シンボリックリンク・
ハードリンク・特殊ファイル・TOCTOU・パストラバーサル・サイズ上限・
削除所有権・一時ファイル・並行アクセス）の網羅的な棚卸しは新設 §13 を
参照。既存実装（`crates/facade/src/model.rs`・`crates/onnx-interop/
src/st_save.rs`・`crates/facade/tests/model_registry.rs`）との対応も
§13 に記録する。

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
   **既存 `save_safetensors_f32(path, ...)`（`crates/facade/src/
   interop/safetensors.rs` 経由の `onnx-interop::st_save::
   save_safetensors_f32` の再エクスポート）はそのまま呼び出さない**
   （PR #2317 review 再々確認・指摘 1 の是正で撤回。当初案「新規の
   低レベル I/O コードを追加しない」は、`st_save.rs` の一時ファイル
   作成が `std::fs::write`〈`OpenOptions::create(true).truncate(true)`
   相当。最終コンポーネントのシンボリックリンクに追従する〉であるため、
   `model_io` 用の tmp パスへ攻撃者が事前配置したシンボリックリンクへ
   追従して書き込んでしまう欠陥をそのまま継承する。是正として、
   バイト列化のみを行い副作用のない既存 `save_safetensors_f32_to_bytes`
   （変更不要）を呼び、model_io 側で `OpenOptions::new().write(true).
   create_new(true)`（Rust std のみ。追加依存なし。存在すれば
   シンボリックリンクか否かを問わず `Err` を返す——mktemp 相当の
   標準的な安全策。§13「一時ファイル作成」参照）による一時ファイル
   作成＋`rename` を行う（世代付きファイル名を使うため
   `save_safetensors_f32` を薄くラップするだけでは済まない、という
   意味で新規コードは避けられないが、範囲は「`create_new` によるオープン
   + 書き込み + `rename`」のみに限定し、独自のシリアライズ処理は追加
   しない）。rename 先の `model.<gen>.safetensors` が既に存在する場合
   （衝突。天文学的に低確率）は新しい `<gen>` を再生成して再試行する
   （上限 5 回。それでも衝突する場合は `ModelIoError::Io` で
   fail-closed）。
2. **manifest に世代情報を追加する**（§4 のスキーマへ
   `"safetensors_file"`〈文字列。`model.<32桁16進>.safetensors` の
   完全一致パターンのみ許可——パス区切り文字を含む値は load 側で即
   `Err`〉と `"safetensors_bytes"`〈u64。保存直後に実際に書き込んだ
   バイト数〉を追加）。manifest 自体は固定名 `manifest.json` のまま、
   手順 1 と同じ `create_new` ヘルパーで一時ファイル＋`rename` する
   （PR #2317 review 再々確認・指摘 1 の是正。旧版は `st_save.rs` と
   同型の `std::fs::write` ベースの一時ファイル作成だったため、同じ
   追従書き込みの欠陥を持っていた）。
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
   `.safetensors`。パス区切りなし）を行ってから、**§13 の no-follow
   手順でそのファイルを開く**（PR #2317 review 再々確認・指摘 1 の
   是正。パターン検証だけでは、対象ディレクトリ内に同名のシンボリック
   リンクが事前配置されていた場合の脱出を防げない）。開いたハンドルの
   `fstat` 実バイト数を `safetensors_bytes` と比較し、不一致なら
   `ModelIoError::Mismatch`（他プロセスによる再保存でファイルが
   差し替えられた・部分書込みの残骸を掴んだ、等の可能性を fail-closed
   に拒否する）。バイト数一致後は既存どおり内部のキー集合・shape 検証に
   進む（§5「load の手順」相当）。
5. **旧世代の後片付け（best-effort・所有権を限定）**: 手順 2 で新しい
   manifest を書く**前**に、既存の（今回上書きされる）
   `manifest.json` を §13 の no-follow 手順で読み、パースに成功すれば
   その `safetensors_file` が指すファイル名を「削除候補」として記憶
   しておく（存在しない・パース不能——初回保存・破損ディレクトリ等——
   なら削除候補なしとして扱う）。新 manifest の rename 成功後、記憶して
   おいた削除候補ファイル**1 件のみ**を、§13 の no-follow 手順（
   `symlink_metadata` でシンボリックリンク・非通常ファイルを除外して
   から `remove_file`）で best-effort 削除する（PR #2317 review 再々
   確認・指摘 2 の是正。当初案の「パターン
   `model.*.safetensors` に一致する現行世代以外を全削除」は、対象
   ディレクトリに置かれた無関係な同名パターンファイルまで削除して
   しまうため撤回した。削除所有権の契約・残余リスク〈中断でどの
   manifest からも参照されなくなった孤立世代は削除対象にならず残存
   する〉は §13「削除所有権」を参照）。削除自体の失敗〈他プロセスが
   開いている等〉は無視する（`crates/onnx-interop/src/st_save.rs`
   「rename 失敗時の tmp ファイル削除は best-effort」と同じ扱い）。
6. **同時保存（2 プロセス）の扱い**: 本設計は「同一ディレクトリへの
   並行 `save_model` はサポート対象外」と明示する（`st_save.rs:185`
   の既存の未サポート表明と同型）。世代 ID が一意である限り、2 つの
   並行 save は互いの safetensors ファイルを破壊しないが、manifest
   （固定名）は最後に rename した側が勝つ。削除所有権を「直前に自分が
   読んだ manifest が参照していた世代 1 件」に限定した是正
   （手順 5・§13）後も、2 つの並行 save が同じ「直前の manifest」を
   読んだ場合（片方の rename が完了する前にもう片方が読む）、両者が
   同じ世代を削除候補とみなし得る——これは同じファイルを 2 回削除
   しようとするだけで（2 回目は `NotFound` を無視する best-effort）、
   どちらの最終 manifest が参照する世代も誤って削除されないため、
   旧案（パターン一致で無条件削除）より悪化はしない。**この残余リスク
   （最終的にどちらの世代が勝つか・敗者側の世代が孤立して残ることを
   含む）は「非サポート」として受容する**（呼び出し元が同一ディレクトリ
   へ並行書き込みしない前提での動作を保証する）。
7. **読込中の書込み（reader/writer race）**: load は「manifest を
   読む → 参照されたファイルを §13 の no-follow 手順で開く（POSIX では
   open 後に他プロセスが unlink してもオープン済み fd の読み取りは
   継続できるため、cleanup〈手順 5〉が read 直後に走っても既に
   オープン済みの読み取りは壊れない）」の順で処理する。ただし
   「manifest を読んでから実際に open するまでの間」に別プロセスが
   さらに新しい世代へ進み、load が読んだ世代が cleanup 対象になる場合
   （load と並行する 2 回目以降の save）は open が `ENOENT` で失敗し
   得る。**この残余レースも手順 6 と同じ「同一ディレクトリへの並行
   save／load は非サポート」の受容範囲に含める**（単一プロセスが
   読み書きする通常運用では発生しない。並行アクセスは呼び出し元で
   ファイルロック等の直列化を行う前提とする）。
8. **プラットフォーム**: rename の原子性については、本設計は既存
   `save_safetensors_f32`／`st_save.rs` が既に前提とする「同一
   ファイルシステム内の `rename` は POSIX 上 atomic」という前提を
   そのまま踏襲する（Windows の `MoveFileEx` も `MOVEFILE_REPLACE_
   EXISTING` 指定で同等の原子的置換を提供するため、rename 自体の
   原子性については OS 分岐を設けない）。**一方 §13 の no-follow
   読み取り手順（シンボリックリンク経由の脱出対策。PR #2317 review
   再々確認・指摘 1 の是正）は Linux（x86_64／aarch64）・macOS 限定の
   実装であり、`ModelRegistry::load`（`docs/facade-model-registry-
   decision.md` §13）と同型の理由で、それ以外の OS（Windows 等）では
   `load_model` を fail-closed に拒否する**（`ModelIoError::Io` /
   `ErrorKind::Unsupported`）。これは既存 `save_safetensors_f32`／
   `st_save.rs` 自体が OS で分岐していないという旧版の前提を、load
   側についてのみ撤回する変更である。**`save_model` 側は非対称に
   Windows でも動作させる**——一時ファイル作成の対策（`create_new`。
   手順 1〜2・§13「一時ファイル作成」）は Rust std のみで完結し
   `OpenOptions::create_new` が Windows でも同じ「既存パス〈シンボリック
   リンクを含む〉があれば `Err`」という契約を提供するため、no-follow
   オープンのような OS 固有の生 flag 値を必要とせず、Windows を
   fail-closed にする理由がない。この非対称性（save は全 OS 対応・load
   は Linux／macOS 限定）は矛盾ではなく、両者が対処する脅威が異なる
   （書き込み側は「一時ファイル名への追従書き込み」、読み込み側は
   「配置済みファイルの追従オープン」）ことに起因する。
9. **電源断耐性（fsync）は対象外のまま**: `st_save.rs` は一時ファイル
   への `fsync`（`File::sync_all`）を行わないため、rename 成功後の
   電源断・OS クラッシュに対する耐性を保証しない（既存注記のとおり。
   本設計もこれを変更しない）。本設計が閉じるのは「プロセスの通常
   終了・異常終了（クラッシュ・kill）を含む、ファイルシステムが応答
   している間の 2 ファイル間コミット順序の不整合」（指摘 2 が対象と
   した問題）であり、ストレージ層の電源断耐性という別軸の非保証は
   既存方針から変更しない。

## 13. ファイル I/O 脅威の全数棚卸し（C。PR #2317 review 再々確認・
2026-09-27・指摘 1・2 の是正に伴う網羅確認）

### 13.1 信頼境界

`dir`（`save_model`／`load_model` の引数）はホスト側パス操作の
起点であり、**`dir` の中身（`manifest.json`・`model.*.safetensors`・
将来のツールが残した無関係なファイルを含む）は非信頼として扱う**。
`ModelRegistry`（`crates/facade/src/model.rs`）が「固定キャッシュ
ルート配下の `name`／`version` という**可変長パス**が非信頼」という
脅威モデルなのに対し、本モジュールは `dir` 自体は呼び出し元が指定する
（`ModelRegistry::new` のような固定ルート解決がない）が、**`dir` 直下
に置かれるファイル名は 2 種類の固定パターンのみ**（`manifest.json`・
`model.<32桁16進>.safetensors`）であり可変長の path traversal 要素を
持たない。したがって model_io の脅威は「`dir` 配下のどれか 1 段の
パスコンポーネントがシンボリックリンク・非通常ファイルである」ことに
集約され、`ModelRegistry` のような複数階層の canonicalize 突き合わせは
不要（構造が単純な分、対策も単純化できる。§13.2 参照）。`dir` 自体が
シンボリックリンクであることは許容する（`ModelRegistry::
load_succeeds_when_root_itself_is_a_symlink` と同じ考え方——利用者が
意図して symlink 越しにモデルを配置する運用を妨げない）。

### 13.2 no-follow リーフオープン手順（読み取り側の共通ヘルパー）

`manifest.json`・`model.<gen>.safetensors` の読み取りはいずれも次の
手順を経由する（`crates/facade/src/model.rs::open_leaf_no_follow`・
`resolve_model_file` と同型。共有ヘルパーへ抽出する方針は §2 item 5）。

1. 対象パス（`dir.join(leaf_name)`）を [`std::fs::symlink_metadata`]
   （リンクを辿らない）で検査し、シンボリックリンクなら拒否する。
   `is_file() == true` を明示要求し、FIFO・Unix ソケット・デバイス
   ファイル等を拒否する。
2. `open_leaf_no_follow`（Linux x86_64／aarch64・macOS 限定。
   `O_NOFOLLOW`〈最終コンポーネントのシンボリックリンク追跡をカーネル
   レベルで拒否〉＋`O_NONBLOCK`〈FIFO への差し替えによる無期限
   ブロックを防ぐ〉の生 flag 値を `custom_flags` で付与）で開く。
   それ以外の OS・アーキテクチャは fail-closed（`ErrorKind::
   Unsupported`。§12.3 手順 8）。
3. 開いたハンドルの `fstat`（[`std::fs::File::metadata`]）で
   `is_file()` を再確認し、Unix では手順 1 の `symlink_metadata` と
   `(dev, ino)` が一致することを検証する（検査と open の間の
   差し替え——TOCTOU——を実体識別子の一致で検出する）。
4. 開いたハンドルの `fstat` で得たファイルサイズを固定上限（manifest
   は 1 MiB、safetensors は `safetensors_bytes`〈manifest 記載値〉との
   一致。§2 item 4）と比較し、上回る場合は読み取りに入る前に拒否する。
5. 同じハンドルから `std::io::Read::take(上限 + 1)` で読む（許容量
   ちょうどで打ち切ると読み取り中の増大〈TOCTOU〉を検出できないため）。

### 13.3 脅威棚卸し表

| 脅威 | 対象（段階） | 本設計の対策（承認後の実装要件） | 既存実装の参照先 |
|---|---|---|---|
| シンボリックリンク（葉ファイル。`manifest.json`／`model.<gen>.safetensors`） | 読み込み | §13.2 の no-follow 手順（`symlink_metadata` 事前拒否 → `O_NOFOLLOW` オープン → `fstat` dev/ino 照合） | `crates/facade/src/model.rs::open_leaf_no_follow`・`resolve_model_file`／`crates/facade/tests/model_registry.rs::load_rejects_symlinked_leaf_file_escaping_root` |
| シンボリックリンク（対象ディレクトリ自身 `dir`） | 読み込み・書き込み共通 | 許容する（`dir` 自体が symlink であることは脅威モデル外。§13.1） | `model_registry.rs::load_succeeds_when_root_itself_is_a_symlink` |
| シンボリックリンク（途中のパス要素） | — | 該当なし（`dir` 直下 1 段のみを扱うレイアウトのため中間ディレクトリが存在しない。§13.1） | — |
| シンボリックリンク（一時ファイル名の位置に事前配置） | 書き込み | `create_new`（`O_EXCL` 相当。存在すれば symlink か否かを問わず `Err`）で作成し、追従書き込みを構造的に防ぐ | 新設（§2 item 5・§12.3 手順 1〜2） |
| ハードリンク | 読み込み・書き込み共通 | 対象外として受容（攻撃者が作成できるのは同一ファイルシステム上の既存ファイルへのリンクのみで、所有者・権限チェックを伴わない本モジュールの脅威モデル外） | `model.rs` モジュール doc「対象外として残る経路」節の理由をそのまま踏襲 |
| 特殊ファイル（FIFO・Unix ソケット・デバイス） | 読み込み | `symlink_metadata`／`fstat` の両方で `is_file() == true` を要求し拒否。`O_NONBLOCK` で FIFO への差し替えによる無期限ブロックも防ぐ | `model.rs::open_leaf_no_follow`／`model_registry.rs::load_rejects_non_regular_leaf_unix_socket` |
| 特殊ファイル（読み込み対象。削除候補） | 削除 | 削除前に `symlink_metadata` で通常ファイルであることを確認してから `remove_file`（§12.3 手順 5） | 新設。`unlink`／`remove_file` はシンボリックリンクを辿らずリンク自体を除去する POSIX 仕様のため、削除自体に脱出リスクはないが、契約を明示するため通常ファイル確認を行う |
| Windows reparse point／junction | 読み込み | no-follow の安全な実装を持たないため `load_model` を fail-closed 拒否（`ErrorKind::Unsupported`） | `model.rs`「Windows 対応状況」節・§12.3 手順 8 |
| Windows reparse point／junction | 書き込み | `create_new` は Rust std が Windows でも同じ「既存パスがあれば `Err`」契約を提供するため対応可能（§12.3 手順 8 の非対称性の根拠） | 新設 |
| TOCTOU（検査〜open の間の差し替え） | 読み込み | 開いたハンドルの `fstat` を `symlink_metadata` の実体識別子（`dev`／`ino`）と照合し、差し替えを検出する（パス再解決ではなく実体同一性で判定） | `model.rs`「検査と open のハンドル一体化」節 |
| TOCTOU（fstat 後の読み取り中の増大） | 読み込み | 同一ハンドルから `take(上限 + 1)` で読み、上限超過を検出する | `model.rs` 手順 6 |
| TOCTOU（一時ファイル作成） | 書き込み | `create_new` は「存在確認」と「作成」を単一のシステムコールで行うためレースが原理的に生じない（std ドキュメントが明記する atomic 操作） | 新設 |
| パストラバーサル（manifest 内の `safetensors_file`） | 読み込み | `model.<32桁16進>.safetensors` の完全一致パターンのみ許可。パス区切り文字・`..`・絶対パスを含む値は即 `Err`（ただし §13.2 の no-follow 手順と併用しない限りシンボリックリンク経由の脱出は防げない点に注意。§13.3 上段） | §4「ファイル形式」・§8 A01 |
| パストラバーサル（manifest 内の層構成・キー名等） | 読み込み | ファイルシステムへ渡さない値（層種別・パラメータキー名は state_dict のキー照合にのみ使う）のため対象外 | §4「ファイル形式」 |
| 読み込みサイズ上限（manifest） | 読み込み | 1 MiB 上限。`take(cap + 1)` で確保・パース前に検証（§2 item 4） | §8 A03 |
| 読み込みサイズ上限（safetensors ヘッダ長・データ長） | 読み込み | `fstat` 実バイト数と manifest の `safetensors_bytes` の事前照合（§12.3 手順 4）に加え、safetensors 自体のヘッダ検証は既存 `load_safetensors_f32_from_bytes` に一元化（複製・迂回しない） | `crate::interop::safetensors`（`onnx-interop::st_load`） |
| 削除の所有権 | 削除 | 「直前に自分が読んだ manifest が参照していた世代 1 件」のみを削除対象とする（§12.3 手順 5）。無関係な同名パターンファイルの削除・中断で孤立した世代の残存を許容する残余リスクは明示的に受容する | 新設（PR #2317 review 再々確認・指摘 2 の是正） |
| 一時ファイル（作成方式） | 書き込み | `create_new`（既存なら失敗）。世代 ID 由来の一意な名前と組み合わせ、通常運用での衝突確率を天文学的に低くする（§12.3 手順 1） | 新設。`st_save.rs` の `std::fs::write` ベース一時ファイル作成は踏襲しない（§12.3 手順 1 是正理由） |
| 一時ファイル（異常終了時の残骸） | 書き込み | rename 失敗時のみ best-effort で自身が作成した一時ファイルを削除する（作成自体に失敗した場合は削除対象がそもそも存在しない）。プロセスクラッシュによる残骸は次回 save の cleanup（手順 5）の対象外のまま残る——所有権を証明できないため削除しない | `st_save.rs`「rename 失敗時の tmp ファイル削除は best-effort」と同方針 |
| 同時保存・読込中の書込み | 読み込み・書き込み共通 | 「同一ディレクトリへの並行 `save_model`／`load_model` はサポート対象外」と明示し受容する（§12.3 手順 6〜7） | `st_save.rs:185`「同一 path への並行書き込みはサポート対象外」と同型 |

### 13.4 共有ヘルパーの抽出範囲

`open_flags`（`O_NOFOLLOW`／`O_NONBLOCK`／`ELOOP` の生値定数）と
`open_leaf_no_follow` を `model.rs` 専用の非公開実装から facade 内部の
共有モジュールへ抽出する（§2 item 5）。抽出は `model.rs` 側の既存
呼び出し・挙動を変えない純粋なリファクタリングであり、`model_io.rs`
（承認後の実装）は同じヘルパーを呼ぶだけで独自の `O_NOFOLLOW` 実装を
持たない（同じ脆弱性クラスの対策を 2 箇所に分散させない）。
