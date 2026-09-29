# `compat::Sequential` 層構成シリアライズ（`save_model`・`load_model`）の設計判断記録

イシュー #2188・親 #2131「PyTorch／TF 置き換えの API 網羅（対応表の
行内深掘り）」。facade 公開面拡張は承認待ちのため本 PR は保留固定のみ
（`crates/facade/src/lib.rs::ModelIoHoldDoctestGuard`＋
`crates/facade/tests/api_surface.rs` のテストで機械的に固定する）。

> **更新記録（イシュー #2369・親 #2362。2026-09-29）**: 親 #2362 でユーザー承認を受け、
> 本 doc §2 item 1 の**主案**（自由関数 `compat::save_model`／`compat::load_model` と
> `#[non_exhaustive] enum ModelIoError`）を `crates/facade/src/compat/model_io.rs` で
> 公開した（対応範囲は未 `compile` の `Linear` と活性化 7 種の最小構成。それ以外は
> `UnsupportedModel` で fail-closed。全 30 層は #2370、BN buffer は #2371、compile 状態は
> #2372・#2373、網羅テストは #2374〜#2376）。**代替案の inherent メソッド
> `Sequential::save`／`load` は承認範囲外のため保留ガードを維持**している（§7）。
> 以下 §0 は #2188 時点の「保留」判断の記録であり、経緯として残す。
> **#2369〜#2373 がすべてマージされるまで crates.io リリースを止める**（公開範囲が途中状態のため）。
>
> **更新記録（イシュー #2370・親 #2362。2026-09-29）**: 対応範囲を `compat::Sequential` の
> `add_*` 全 30 種へ広げた（`add_module` の利用者定義層のみ構成を記録できないため
> `UnsupportedModel` のまま）。kind 別 `params` スキーマ・f32 の JSON 表現・平坦化規則は §4、
> seed を保持しないこと・BN の暫定 fail-closed・層ごとのモード一致・save 側の自己検証は §5 を参照。
> 上限定数（`MAX_MANIFEST_BYTES`・`MAX_LAYERS`・`MAX_ARRAY_LEN`・`MAX_OBJECT_KEYS`・
> `MAX_JSON_DEPTH`）の値は変えていない（§2 item 4 の補足）。

> **更新記録（イシュー #2371・親 #2362。2026-09-29）**: BatchNorm1d／2d の running stats を
> `{i}.running_mean`／`{i}.running_var` として同じ safetensors へ保存・復元するようにした
> （#2370 の暫定 fail-closed〈stats が初期値のときだけ保存可〉を撤廃）。manifest の `buffer_keys` を
> 使い（§4）、load は `buffer_keys`・safetensors のキー集合・shape を層構成から導いた期待と
> 完全一致で照合してから `BatchNorm1d/2d::from_parameters` で BN 層を組み直す（§5）。
> `training` フラグは従来どおり manifest で往復し、load が全層を `set_training` で揃える。
> **`num_batches_tracked` は復元しない**（load 後は 0 から再開。forward のどこからも参照されず
> 数値に影響しないため。`save_model`／`load_model` の API doc にも明記。§11）。
> 上限定数の値は変えていない。**#2369〜#2373 がすべてマージされるまで crates.io リリースを止める**
> 契約は継続。

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

## 2. 承認事項（一覧。item 4 の上限値は 2026-09-29 承認済み）

1. **公開 API の署名とエラー型**。
   - `fandhe_ai::compat::save_model(model: &Sequential, dir: impl AsRef<Path>) -> Result<(), ModelIoError>`
   - `fandhe_ai::compat::load_model(dir: impl AsRef<Path>) -> Result<Sequential, ModelIoError>`
   - `#[non_exhaustive] pub enum ModelIoError { Io(std::io::Error), Manifest { message: String }, Safetensors(String), UnsupportedModel { reason: String }, Mismatch { message: String }, Autodiff(AutodiffError), TooLarge { what: &'static str, limit: u64 } }`
   - 代替案として `Sequential::save(&self, dir)`／`Sequential::load(dir)`
     （inherent メソッド）も併記する。
2. **内部クレートへの追加**（facade 公開面は広がらない）。
   - **【内部 API 実装済み（イシュー #2367）。`decode_slot_only_state_dict` で `step_count` 不要。`docs/autodiff-optimizer-state-dict-decision.md` §8】** `impl OptimizerStateDict for Sgd`（`crates/autodiff/src/optim/sgd.rs`）。
     `Sgd` は `velocity: Option<Vec<Tensor>>` を持つが `OptimizerStateDict`
     未実装（`docs/autodiff-optimizer-state-dict-decision.md`）。
     `decode_state_dict` は現状 `step_count` を必須とするため、`Sgd`
     （`step_count` を持たない）向けの引数化・専用デコーダが要る。
   - **【内部 API 実装済み（イシュー #2366）。キーは接頭辞なしで、facade 側が `optimizer.` を付与する。詳細は `docs/autodiff-lbfgs-decision.md` §10】**
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
     old_stps.len() == ro.len()`（manifest の `history_len` と
     safetensors 側の実キー数の完全一致。**`<= history_size` は
     非信頼値どうしの整合性確認にすぎず上限にはならない**——`history_size`
     自体も manifest 経由の値ではなく `compile` 時に呼び出し元が指定した
     設定値だが、これを唯一の根拠に確保量を決めない。実際の履歴件数は
     safetensors 内に実在するキー数（ファイル全体が §2 item 4 の固定
     上限で既に有界）で決まるため、パースは「まず固定上限までしか
     読まない安全なファイル読み取りの結果からキー集合を得て、その件数を
     `history_len` および `history_size` と突き合わせる」順で行い、
     `history_len` を信じて事前確保してから読むことはしない）・
     `d.len()`／`prev_flat_grad.len()`（`Some` の場合）／各 `old_dirs`／
     `old_stps` 要素の長さが `slot_shapes` の要素数合計と一致・`t`／
     `h_diag`／`ro` の各要素・`d`／`old_dirs`／`old_stps`／
     `prev_flat_grad` の全要素が有限であること（PR #2317 review 再確認
     ×3・2026-09-27 第 2 回是正。§13.5「状態復元値の再点検」参照）。
3. **GradScaler の状態復元コンストラクタ**（PR #2317 review 指摘 1 の是正。
   §11「棚卸し（A）」参照）。`GradScaler::new` は `init_scale` からしか
   開始できず、`update` は backoff／growth の状態機械を経由するため、
   任意の `(scale, growth_tracker)` の組を事後に再現できない
   （`growth_tracker` は growth／backoff のたびに `0` へリセットされる
   ため、`update(false)` を再生しても再生前に backoff／growth で変化
   済みの `scale` 自体は動かない）。承認後は次の内部 API（`fandhe_ai_
   autodiff` 限定。facade へは再エクスポートしない）を追加する:
   `grad_scaler_from_state(config: GradScalerConfig, scale: f32,
   growth_tracker: u64) -> Result<GradScaler, AutodiffError>`
   （`fandhe_ai_autodiff::nn::optim::amp` の自由関数）。
   **設計変更の記録（イシュー #2365・PR #2404 review 是正）**: 当初は
   `GradScaler::from_state` という inherent メソッドを承認していたが、
   facade は `GradScaler` 型を `pub use` で再エクスポートしており
   （`crates/facade/src/optim.rs`）、inherent メソッドは facade の
   公開面へ自動的に露出して「facade へは再エクスポートしない」契約に反する。
   このため名前・所在のみ自由関数 `grad_scaler_from_state` へ変更した
   （検証内容・引数・戻り値は承認済みの契約のまま不変。`nn::optim` の
   `pub use` にも載せない）。検証は
   `new`（`config` の各フィールド）に加え、`scale` が有限・正・非正規化
   数でないこと（`update` の backoff 検証と同一基準）、`growth_tracker
   < config.growth_interval`（`growth_tracker` は `growth_interval` に
   到達すると必ず growth し `0` へリセットされる仕様〈`amp.rs::
   GradScaler::update`〉のため、正常な状態機械が到達できる範囲は
   `0..growth_interval` のみで、これを超える値は改竄・破損の兆候として
   fail-closed に拒否する）。
4. **上限値の新設**（PR #2317 review 再確認 ×3・2026-09-27 第 2 回是正。
   §13.0 の原則「非信頼値は信頼できる固定上限で挟んでから照合にのみ
   使う」に基づき、以下は**すべてコード定数（信頼できる固定値）**として
   `model_io.rs`（または共有 `fs_guard`）に持つ。manifest 側の記載値は
   この固定定数と一致するかどうかの確認にのみ使い、確定前の資源確保・
   ループ回数の根拠にしない）。
   **#2369 の実装値（2026-09-29 ユーザー承認済み）**: `MAX_MANIFEST_BYTES = 1 MiB`・
   `MAX_LAYERS = 4096`（いずれも `model_io.rs` の private const）。**承認記録**: 親 #2362 の
   コメント <https://github.com/Fandhe-AI/fandhe-ai/issues/2362#issuecomment-5888987015>
   （2026-09-29）で、manifest サイズ上限 1 MiB・層数上限 4096・Lbfgs 履歴ペア数上限 65536
   を候補値どおり確定する承認を受けた。根拠: v1 manifest は 1 層 100〜150 B 程度で 4096 層でも
   約 0.6 MiB に収まる（`model_io.rs` の単体テスト `manifest_of_max_layers_fits_the_manifest_bound`
   が 5 桁次元の 4096 Linear で固定）。4096 は既存テストの 1714 層 `Sequential`
   （`sequential.rs`）を包含する 2 のべき乗。併せて構造上の値として JSON ネスト上限 4
   （v1 スキーマの最大ネスト。#2372・#2373 の `compiled` も 4 以内）・配列要素数上限
   `2 * MAX_LAYERS`・object キー数上限 16（重複キー検査の線形走査を有界にする）を持つが、
   これらはポリシー閾値ではなくスキーマから導いた値。値の変更は再承認が必要で、定数と
   本記述を同時に更新する。`MAX_TMP_NAME_ATTEMPTS = 8` は §12.3 で確定済みの値
   （`docs-site` と同値）。
   **#2370 の補足（値は不変）**: 全 30 層対応で `parameter_keys` は 1 層あたり最大 16 要素
   （`add_transformer_encoder`）になり、`MAX_ARRAY_LEN`（8192）は 4096 層より手前
   （TE のみなら 512 層）で先に超えうる。上限値は引き上げず（再承認が要るため）、
   `save_model` が書き込み前に manifest を描画して load と同じ厳格パーサで読み戻し
   （`verify_round_trip`）、超過するモデルを `TooLarge` で拒否する。「保存できたのに
   読めない」ファイルは生まれない。引き上げが必要になった場合は別途ユーザー承認を得る。
   - **safetensors ファイルサイズ上限**: `crates/facade/src/model.rs`
     の `MAX_MODEL_FILE_BYTES`（1 GiB。private const）を再利用する
     ——承認事項ではなく確定方針とする。理由: 用途が同一（非信頼な
     外部フォーマットファイルのサイズ上限）で、`ModelRegistry` の
     safetensors 読み込みと model_io の safetensors 読み込みは同じ
     脅威モデル（§13.1）を共有するため、新しい値を発明せず既存定数を
     `fs_guard` 共有モジュールへ `pub(crate)` として抽出し双方から使う
     （§2 item 5・§13.4）。
   - **manifest（JSON）サイズ上限**: **1 MiB で確定**（2026-09-29 ユーザー承認。
     #2362 コメント）。ワークスペース内を横断調査した結果、facade が依存
     できる範囲に同一用途の既存定数はない（`crates/self-repair/src/
     candidate.rs::MAX_CONTENT_BYTES`〈1 MiB〉・`crates/docs-site/src/
     nav.rs::MAX_INPUT_BYTES`〈1 MiB〉はいずれも同じ値だが facade とは
     無関係のクレート・用途〈self-repair の候補パッチ・docs-site の
     Markdown ソース〉のため転用しない）。
   - **層数上限**: **4096 で確定**（2026-09-29 ユーザー承認。#2362 コメント）。
   - **Lbfgs 履歴ペア数の上限**: **65536 で確定**（2026-09-29 ユーザー承認。
     #2362 コメント。`history_size` 自体・history エントリ総数のいずれも
     超過は型付きエラーで拒否）。`LbfgsConfig::history_size` は `>= 1` のみ
     検証済みで上限がないため、manifest の `history_len` を信じて確保しない
     ことに加えて設ける DoS 対策の固定上限である。**本 PR（#2369）は `Lbfgs` の
     読み書きを扱わない（最小構成は未 `compile` の `Linear` と活性化のみ）ため
     定数は未実装。実装は後続 issue（`compile` 状態の保存・復元を扱う
     #2372・#2373 の A 系）で、`Lbfgs` の状態を manifest／safetensors へ
     書き読みする際に適用し境界テストを加える**。上限の根拠・使い方は §2 item 2
     の是正済み記述を参照）。
5. **ファイル I/O のハードニング用内部 API**（facade 公開面は広がらない。
   PR #2317 review 再々確認・2026-09-27 第 2 回是正に伴う新設。全数棚卸しは
   §13）。
   - **共有の no-follow リーフオープンヘルパー**: `crates/facade/src/
     model.rs` の `open_leaf_no_follow`／`open_flags` モジュール
     （`O_NOFOLLOW`／`O_NONBLOCK` の生値・Linux x86_64／aarch64・macOS
     限定）を `model.rs` 専用の非公開関数から facade 内部の共有
     ヘルパー（例: `crate::fs_guard` モジュール）へ抽出し、`model.rs`・
     新設 `model_io.rs` の双方が使う。同モジュールへ `model.rs` の
     `MAX_MODEL_FILE_BYTES` も `pub(crate)` として合わせて抽出し、
     `model_io.rs` の safetensors サイズ上限として共用する（§2 item 4）。
     既存呼び出し元（`model.rs`）の挙動は変えない（純粋な抽出。§13 参照）。
   - **`create_new` を使う書き込みヘルパー**（model_io 専用。既存
     `crate::interop::safetensors::save_safetensors_f32`（`onnx-interop::
     st_save::save_safetensors_f32`）はそのまま使わない——理由は §13
     「一時ファイル作成」節・§12.3 是正参照。代わりにバイト列のみを
     返す既存 `save_safetensors_f32_to_bytes`（副作用なし・変更不要）を
     呼び、`model.<gen>.safetensors` は `OpenOptions::new().write(true).
     create_new(true)`（Rust std のみ・プラットフォーム分岐不要）で
     **最終ファイル名へ直接**書く（tmp＋`rename` を経由しない。世代 ID
     が一意であることに加え、`create_new` の `EEXIST` がそのまま衝突
     検出になるため——§12.3 手順 1 是正）。**`manifest.json` は固定名の
     ため引き続き `create_new` の一時ファイル＋`rename` を使う**
     （置換対象が固定名で存在する以上、原子的な置換には rename が要る。
     §12.3 手順 2）。この一時ファイルは、`rename` 成功前に手順が失敗
     した場合に限り §13.6 の所有権確認手順で削除する。
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
| `AdamW`／`Adam`／`RmsProp`／`Adagrad`／`Lamb`／`Sgd` は `OptimizerStateDict` 実装済み（`Sgd` は #2367）。`Lbfgs` は専用 inherent API 実装済み | `crates/autodiff/src/nn/optim/state_dict.rs`・`crates/autodiff/src/nn/optim/lbfgs.rs` | 承認後は `impl OptimizerStateDict for Sgd`、および `Lbfgs` 専用のキー配置を持つ状態保存・復元 API が必要（§2 item 2）。`Lbfgs` 側の内部 API は #2366 で実装済み（接頭辞なし。`docs/autodiff-lbfgs-decision.md` §10） |
| `api_surface.rs::workspace_declares_optimizer_state_dict_fn_names_only_in_allowed_locations` が `state_dict`／`load_state_dict` の宣言元を完全一致で固定している | `crates/facade/tests/api_surface.rs` | 承認後の facade 側ヘルパーにこの名前は使えない |
| `GradScaler` に `config()` はあるが、`(scale, growth_tracker)` を任意の値に復元するコンストラクタがない（`update` の backoff／growth 経由でしか変化しない） | `crates/autodiff/src/nn/optim/amp.rs` | `compile_with_amp` の時点で `GradScalerConfig` を記録し、save 時点の `scale()`／`growth_tracker()` を manifest に保存、復元は承認後の `grad_scaler_from_state`（§2 item 3）を使う（`update` の再生では現在の `scale` を再現できないため。§11） |
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
    "buffer_keys": [{"key": "1.running_mean", "shape": [6]}, {"key": "1.running_var", "shape": [6]}],
    "safetensors_file": "model.0123...cdef.safetensors",
    "safetensors_bytes": 4096,
    "compiled": null
  }
  ```
  **`buffer_keys`（#2371）**: 要素は `parameter_keys` と同じ `{"key","shape"}`。BatchNorm1d／2d の
  層 `i` ごとに `{i}.running_mean`・`{i}.running_var`（shape は `[num_features]`）を層順・
  mean → var の順で並べる。期待値は層構成（`num_features`）から導出し、manifest の記載値は
  完全一致の判定にだけ使う（ファイル I/O・確保量の根拠にしない。§13.5）。過不足・順序・
  キー名・shape の不一致は `Mismatch`、要素の型違い・未知フィールドは `Manifest`。
  buffer の値の有限性は検査しない（重みと同じく bit のまま往復し、発散したモデルも
  無言変換しない。REQ-7）。BN を含まないモデルでは `[]`。
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
  `grad_scaler_from_state(config, scale, growth_tracker)`（§2 item 3）を
  使う（PR #2317 review 指摘 1 の是正。§11 参照）。
- 数値表現: f32 は Rust の最短往復表記（`{:?}`）で書き `str::parse::<f32>`
  で読む（非有限値は save 時に `UnsupportedModel` で拒否）。u64／usize は
  JSON の整数として書き独自パーサで読む。
  **#2370 で確定した f32 の JSON 表現**: f32 は JSON 数値（小数点または指数を含む字句）で持つ。
  厳格パーサの字句解析は JSON 数値の完全な文法（`-?(0|[1-9][0-9]*)(\.[0-9]+)?([eE][+-]?[0-9]+)?`）を
  受理し、符号・小数点・指数のいずれも無い非負整数だけを `Json::Num(u64)`（先頭ゼロ・u64 桁あふれを
  拒否）、それ以外は生の字句を `Json::Real` として保持する（字句長は 64 バイト以内）。整数欄
  （`as_u64`／`as_usize`）は `Real` を拒否するため整数フィールドの厳格性は変わらない。f32 欄
  （`as_f32`）は `Real` だけを受理し、有限で、かつ**正準形**（読んだ値を `{:?}` で書き戻した
  文字列と一致。`0.50`・`1`・`1E0` は拒否）のものに限る。これにより改竄を検出でき、往復は bit 一致
  （`-0.0`・非正規化数を含む）になる。#2372 の optimizer config（`lr` 等）もこの表現を引き継ぐ。
- **kind 別 `params` スキーマ（#2370。kind は 30 種の文字列 allowlist。`params` は kind ごとに
  固定のキー集合で、未知キー・欠落キー・重複キー・型違いは `Manifest`、未知 kind は
  `UnsupportedModel`）**。`params` は `add_*` の引数から `seed` を除いたものだけで、`add_*` が
  内部で固定する値（linear／conv／MHA／TE の `bias=true`・pool の `ceil_mode=false`・TE の
  ReLU 活性化と LayerNorm eps 等）は書かない（load が同じ `add_*` を呼ぶため自動で再現される）。
  引数は渡された生の値のまま記録する（pool の `stride=None` を kernel で埋めない）。
  `[usize; 2]` は `*_h`／`*_w` の平坦な 2 キーへ展開する（`params` 内に配列を置くと
  `MAX_JSON_DEPTH = 4` を超えるため。全 kind で 16 キー以下）。`Option` は `null` または値で、
  pool2d の `stride` は `stride_h`／`stride_w` が「両方 `null`」か「両方整数」でなければ `Manifest`。

  | kind | params のキー | パラメータ（層内名: shape） |
  |---|---|---|
  | `linear` | `in_features`・`out_features` | `weight: [in, out]`・`bias: [out]` |
  | `relu`・`sigmoid`・`tanh`・`silu`・`hardswish`・`gelu`・`gelu_tanh` | なし | なし |
  | `leaky_relu` | `negative_slope`（f32） | なし |
  | `elu` | `alpha`（f32） | なし |
  | `softmax`・`log_softmax` | `dim` | なし |
  | `softplus` | `beta`・`threshold`（f32） | なし |
  | `flatten` | `start_dim`・`end_dim` | なし |
  | `dropout` | `p`（f32） | なし |
  | `conv2d` | `in_channels`・`out_channels`・`kernel_size_{h,w}`・`stride_{h,w}`・`padding_{h,w}`・`dilation_{h,w}`・`groups` | `weight: [out, in/groups, kh, kw]`・`bias: [out]` |
  | `conv1d` | `in_channels`・`out_channels`・`kernel_size`・`stride`・`padding`・`dilation`・`groups` | `weight: [out, in/groups, k]`・`bias: [out]` |
  | `layer_norm` | `normalized_size`・`eps`（f32） | `weight: [n]`・`bias: [n]` |
  | `rms_norm` | `normalized_size`・`eps`（f32） | `weight: [n]` |
  | `batch_norm1d`・`batch_norm2d` | `num_features`・`eps`・`momentum`（f32） | `weight: [c]`・`bias: [c]` |
  | `embedding` | `num_embeddings`・`embedding_dim`・`padding_idx`（`null` 可） | `weight: [num, dim]` |
  | `multihead_attention` | `embed_dim`・`num_heads` | `{q,k,v,out}_proj.{weight: [e, e], bias: [e]}` |
  | `transformer_encoder` | `d_model`・`num_heads`・`dim_feedforward` | `self_attn.*`（8）・`linear1.{weight: [d, dff], bias: [dff]}`・`linear2.{weight: [dff, d], bias: [d]}`・`norm1.*`・`norm2.*`（計 16） |
  | `max_pool2d` | `kernel_size_{h,w}`・`stride_{h,w}`（`null` 対可）・`padding_{h,w}`・`dilation_{h,w}` | なし |
  | `max_pool1d` | `kernel_size`・`stride`（`null` 可）・`padding`・`dilation` | なし |
  | `avg_pool2d` | `kernel_size_{h,w}`・`stride_{h,w}`（`null` 対可）・`padding_{h,w}`・`count_include_pad`（bool） | なし |
  | `avg_pool1d` | `kernel_size`・`stride`（`null` 可）・`padding`・`count_include_pad`（bool） | なし |
  | `adaptive_avg_pool2d` | `output_size_{h,w}` | なし |
  | `adaptive_avg_pool1d` | `output_size` | なし |

  期待キー・shape は非信頼な整数から純粋な算術だけで導く（`Vec` の事前確保に使わない。
  conv の `in/groups` は `groups >= 1` かつ割り切れることを `Manifest` として先に検査し、
  除算パニックを起こさない）。意味上の範囲（`p` が [0, 1] の外・`beta <= 0`・`kernel = 0`・
  `padding_idx >= num_embeddings`・`momentum` の範囲等）は load 時の `add_*` の既存検査が
  `ModelIoError::Autodiff` として拒否し、重複して実装しない。層を構築する（テンソルを確保する）のは、
  上限付きで読んだ safetensors の実 shape と期待キーが完全一致した後だけである。

## 5. 意味論

- **LayerSpec**: compat 内部に `LayerSpec` enum を置き、`add_*` 30 種と
  1 対 1 対応させる（引数・seed を保持）。`Sequential` に private
  フィールド `specs: Vec<LayerSpec>` を追加し、各 `add_*` で push する
  （フィールドは private なので公開 API は非破壊）。
  `specs.len() != layers().len()` なら `UnsupportedModel`。
- **#2370 で確定した LayerSpec の扱い**:
  - **seed は保持しない**（上の「引数・seed を保持」を改める。#2369 が残した「#2370 で再判断」の
    結論）。seed は初期化にしか使われず、重みは直後の `load_state_dict` で上書きされるため、
    manifest に載せても復元結果は変わらない。load は seed=0 固定で構築する。
  - **`add_module`（利用者定義層）は対象外**。構成を記録できないため `LayerSpec::Unsupported` の
    まま `save_model` が `UnsupportedModel` で拒否する。
  - **BatchNorm の running stats は #2371 で保存・復元する**。`running_mean`／`running_var` を
    `{i}.running_mean`／`{i}.running_var` として safetensors に保存し、manifest の `buffer_keys`
    にも記録する。load は `buffer_keys` と safetensors のキー集合・shape を完全一致で照合し、
    BN を `from_parameters(weight, bias, running_mean, running_var, eps, momentum)` で組み直す
    （weight／bias は safetensors の値を clone して渡し、直後の strict な `load_state_dict` でも
    同じ値を再設定する。buffer キーは `load_state_dict` が未知キーとして拒否するため先に取り除く）。
    層を BatchNorm として取り出せない場合の fail-closed は維持する。
  - **層ごとのモード一致**。`add_*` は push 後に層のモードを同期しない一方、load は
    `set_training(manifest.training)` で全層を揃える。モードで forward が変わる kind（dropout・
    batch_norm1d／2d）だけ、層の `training()` がモデル全体と一致することを要求し、不一致
    （`eval()` の後に積んだ等）は `UnsupportedModel`。他の kind へは適用しない
    （`Module::training` の既定が `true` のため eval モデルを誤って拒否する）。
  - **save 側の自己検証**（`verify_round_trip`）。`dir` に触れる前に manifest を描画し、load と
    同じ厳格パーサで読み戻して構成（kind・params〈f32 は bit 一致〉・parameter_keys・training）が
    一致することを確認する。配列長・キー数・深さ・サイズ・f32 正準形の違反をまとめて検出し、
    上限起因は `TooLarge`、それ以外の不一致（内部不整合）は `UnsupportedModel`。f32 引数が
    非有限のモデルは JSON へ書けないため保存前に `UnsupportedModel` で拒否する。
- **save の手順**（世代コミット方式。詳細は §12。symlink・所有権対策は
  §13）: 検証をすべて終えてから書き込みに入る。`create_dir_all` →
  世代 ID を採番し `save_safetensors_f32_to_bytes`（既存・副作用なし）
  で得たバイト列を、`create_new`（§13「一時ファイル作成」）で
  `model.<gen>.safetensors` へ直接書く（衝突時は既存エントリに触れず
  `<gen>` を再生成して再試行。上限 8 回。§12.3 手順 1） → その `<gen>`・
  実バイト数を含む `manifest.json` を `create_new` 一時ファイル＋`rename`
  して書く（一時ファイル名の衝突時も同じ上限で再試行する。§12.3 手順 2）
  （**この manifest の rename が唯一のコミット点**。safetensors 側は
  世代 ID が一意なため上書きされることがなく、manifest がそれを参照した
  時点で既に完全な内容で存在する）。**`save_model` は `dir` の既存内容を
  読まない**（§13.0 の原則。旧版は削除対象を決めるために旧
  `manifest.json` を読んでいたが、削除機能自体を撤回したため不要に
  なった——§12.3 手順 5・§13.6）。
- **load の手順**: `dir` を §13 の no-follow 手順で開いた
  `manifest.json` からサイズ上限付きで読み厳格パース →
  `format`／`format_version` 完全一致確認 → `safetensors_file` の
  パターン検証 → 参照された `model.<gen>.safetensors` を同じく §13 の
  no-follow 手順で開き、開いたハンドルの実バイト数を
  `safetensors_bytes` と照合（不一致は `Mismatch`。§12.3 手順 4）→
  同じハンドルから safetensors を上限付きで読む → 3 種のキー集合と
  ファイル内容の完全一致・shape 一致を確認 → spec 順に層を構築（BN のみ
  `from_parameters`。#2371 で結線済み）→ `load_state_dict` → `set_training` → `compiled`
  があれば optimizer 復元（`Sgd`／`Lbfgs` は承認後の専用復元 API）・
  AMP があれば `grad_scaler_from_state` で scaler を復元。途中失敗時は
  部分的に構築した `Sequential` を返さない。
- **非復元のもの**: BN の `num_batches_tracked`（forward 計算に使われない
  カウンタのみで数値へ影響しない。load 後は 0 から再開する。§11）、Dropout の RNG 状態（グローバル
  RNG。インスタンスに保持されない）、LR scheduler・callbacks・param
  groups（`Compiled`／`Sequential` に保持されない `fit` 呼び出し引数。
  ただし LR scheduler が書き換えた**現在の** LR 自体は各 optimizer の
  `config()` 経由で復元される）、勾配累積バッファ（`run_fit` 内ローカル
  変数で epoch／ウィンドウ境界を跨いで持ち越されない）。網羅的な棚卸しは
  §11 を参照。

## 6. 承認後の検証計画

`crates/facade/tests/compat_sequential_model_io.rs` で、30 種すべての
層を含むモデル・深い異種スタック・transformer encoder・train モード
後の BN running stats（#2371。`crates/facade/tests/compat_sequential_model_io_batch_norm.rs`）・6 optimizer（`Sgd`／`AdamW`／`Adam`／`RmsProp`／
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
`save_model` の一時ファイル・最終ファイル名の位置に既存のシンボリック
リンク（有効・dangling いずれも）が存在する場合の挙動は、§12.3 手順 1・2
の再試行方式（PR #2317 review 指摘〈P2〉の是正）に合わせ次の 2 ケースで
検証する（テストでは世代 ID・一時ファイル名の生成を差し替え可能な内部
フックで衝突を注入する）:

1. **一部の候補が衝突しても再試行の上限内に収まる場合**: `save_model` は
   衝突した既存のシンボリックリンクに追従・上書きせず（`create_new` が
   `AlreadyExists` を返すのみで、リンク自体にも参照先にも一切触れない）、
   別の世代 ID・一時ファイル名で保存に成功する。この場合、衝突を起こした
   シンボリックリンク自身は変更・削除されずそのまま残る。
2. **再試行の上限（8 回。[`MAX_TMP_NAME_ATTEMPTS`]。§12.3 手順 1）に
   達してもなお全候補が衝突する場合**: `save_model` は `Err` を返し、
   `dir` の既存エントリ（衝突を起こしたシンボリックリンクを含む）は
   一切変更されない。

また、既存 `manifest.json` がシンボリックリンクである場合、`save_model`
はそのリンクエントリ自体を新しい通常ファイルへ `rename` で置換し、
リンクの参照先ファイルには一切書き込まないこと（§12.3 手順 2 是正）を
検証する。**削除しない契約**
（§13.0・§13.6）については、再保存前に存在していた旧世代の
`model.<gen>.safetensors`（直前の manifest が参照していたもの）と、
無関係な命名規則一致ファイル（テストが手動で作成した「よそ者」の
`model.deadbeef....safetensors`）の**両方**が、再保存後も削除されずに
残存することを検証する（`save_model` は失敗時の自己所有一時ファイルを
除き `dir` の既存ファイルを一切削除しない）。

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
- **#2369 で実施した置き換え**（親 #2362）:
  - `ModelIoHoldDoctestGuard`: `model_io` モジュール・自由関数・エラー型のローカル定義
    （glob 衝突で doctest 自体が壊れるため）を撤去し、**代替案の inherent メソッド
    （`Sequential::save`／`load`／`save_model`／`load_model`）専用のプローブへ縮小**した
    （`api_surface.rs` の `MODEL_IO_HOLD_PROBE_BODY` も同期）。
  - ソース走査は**正ガード**へ反転: `facade_model_io_public_surface_matches_approved_contract`
    （`mod model_io;` は `compat/mod.rs` に private でちょうど 1 件・`pub use
    model_io::{ModelIoError, load_model, save_model};` がちょうど 1 文・`enum ModelIoError` と
    `fn save_model`／`fn load_model` は `compat/model_io.rs` に各 1 件・代替案は 0 件）と、その
    自己テスト `facade_model_io_public_surface_detects_each_category`。追加で
    `model_io_module_exposes_only_approved_surface`（`model_io.rs` の `pub` 項目は 3 件のみ）・
    `model_io_items_are_reachable_via_facade`（署名・`#[non_exhaustive]` の 7 variant）。
  - 定義元インベントリ `workspace_declares_model_io_fn_names_only_in_allowed_locations` の
    期待集合は `facade/src/compat/model_io.rs` の `save_model`・`load_model` 各 1 件へ差し替えた。
    代替案の `workspace_declares_sequential_alt_save_load_fn_names_only_in_allowed_locations` は
    承認範囲外のため 0 件固定を維持する。
  - `LOWERCASE_PUB_USE_LEAF_ALLOWLIST` に `save_model`・`load_model` を追加した
    （小文字始まりの `pub use` 葉は関数再エクスポートの契約）。

## 8. OWASP Top 10 観点（承認後の要件として記録）

- **A01 アクセス制御の不備／パストラバーサル**: `dir` 配下のファイル名は
  固定文字列（`manifest.json`）または固定パターン（
  `model.<32桁16進>.safetensors`。パス区切り文字混入は即 `Err`）のみを
  扱い、それ以外の名前を `std::fs` へ渡さない。**パターン一致のみでは
  同名のシンボリックリンクによるディレクトリ脱出を防げない**ため
  （PR #2317 review 再々確認・指摘 1）、実際に開く段では必ず §13 の
  no-follow 手順（シンボリックリンク・非通常ファイルの拒否、TOCTOU 対策
  としての実体識別子照合）を経由する（`save_model` は既存ファイルを
  削除しないため「削除する段」自体が存在しない。§13.0・§13.6）。`dir`
  はディレクトリごと非信頼として扱う（§13「信頼境界」）。
- **A03 インジェクション／非信頼入力**: manifest.json と
  model.<gen>.safetensors は非信頼の外部フォーマットとして扱う。バイト数
  はまず信頼できる固定上限と `fstat` 実長を比較し（確保・走査の前に
  検証。§13.2 手順 4）、非信頼値である manifest 記載のバイト数は
  この固定上限の範囲内での**一致確認**にのみ使う（§13.0）。JSON の
  深さ・層数・`num_slots`・`Lbfgs` の履歴件数も同様に、非信頼値を
  事前確保の根拠にせず固定上限・実測値との一致確認にのみ使う（§13.5）。
  固定スキーマに対し未知
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
  照合し取り違えを fail-closed にする。**旧世代ファイルは自動削除しない**
  （§13.0・§13.6・§10 末尾の 2026-09-27 第 2 回是正。旧版は「直前の
  manifest が参照していた 1 件に限定して削除」としていたが、この判定
  自体が非信頼な manifest の値を削除対象決定の根拠にしていたため
  さらに撤回した——非信頼 manifest が指す任意の通常ファイル名を
  「削除してよい対象」だと信じてしまう構造は、対象を 1 件に絞っても
  解消しない）。
- **A04 安全でない設計**: 公開面の拡張を承認前に実施しない（保留
  ガードで機械的に固定）。依存の追加（`serde_json`・`zip` 等）はせず
  `Cargo.toml` も不変。
- **本 PR 自体**: 本番コードの挙動は変わらない。秘密情報は扱わない。

## 9. 非信頼データに関する記録

イシュー本文に、指示の上書きや秘密情報の出力といった命令文は見当たら
なかった。本文は要件としてのみ扱い、逐語での引用はしていない。

## 10. 再開条件

**#2362 で再開済み（#2369）**: 2026-09-29 のユーザー承認を受け、親 #2362 配下で
「§4〜§6 の実装 → 保留ガードの撤去」を段階的に進める。#2369 は最小構成の公開と
保留ガードの正ガード化（§7）。以下は #2188 時点の再開条件の記録。

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
`grad_scaler_from_state` コンストラクタ（§2 item 3）を新設して復元する
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
「削除所有権」。**この「1 件限定」方式自体が本節末尾〈2026-09-27
第 2 回是正〉でさらに撤回され、自動削除機能をなくす方式へ変更された。
理由は本節末尾を参照**）。この過程で一時ファイル作成手順自体にも
`std::fs::write`（追従する `create` + `truncate`）を使っていた欠陥が
判明したため（一時ファイル名の位置への事前配置シンボリックリンクへの
追従書き込み）、`create_new`（Rust std のみ・全対象プラットフォーム
共通）へ是正した（§2 item 5）。ファイル I/O 全般（シンボリックリンク・
ハードリンク・特殊ファイル・TOCTOU・パストラバーサル・サイズ上限・
削除所有権・一時ファイル・並行アクセス）の網羅的な棚卸しは新設 §13 を
参照。既存実装（`crates/facade/src/model.rs`・`crates/onnx-interop/
src/st_save.rs`・`crates/facade/tests/model_registry.rs`）との対応も
§13 に記録する。

**非信頼値を削除・確保の根拠にしないための再是正（PR #2317 review
再確認・2026-09-27 第 2 回。P0 指摘 2 件）**: 上記の棚卸し（直前段落）
で導入した対策のうち 2 件が、根本原因（非信頼入力を破壊的操作・資源
確保の根拠にしている）を再発させていることが判明し、さらに是正した。
(1) 「直前の manifest が参照していた世代 1 件のみ削除」は、削除対象の
決定自体を**非信頼な旧 manifest の `safetensors_file` 値**に依存して
おり、`symlink_metadata` による事前検査は「対象が通常ファイルである
こと」しか証明せず「本モジュールが以前作成したファイルであること」は
何も証明しないため、命名パターンに合致する無関係な既存ファイルを
指定して削除させられる余地が残っていた（P0）。所有権を暗号学的に
証明する手段（鍵・MAC）は依存追加なしでは過剰であるため、是正として
**`save_model` から自動削除機能自体を撤回**した（§13.0 の原則・
§12.3 手順 5・§13.6）。唯一の例外は「今回の呼び出し自身が作成し、
同一呼び出し内で失敗した一時ファイル」で、これは fd を保持しているため
所有が証明できる（§13.6）。孤立した旧世代ファイルの残存は明示的に
受容し、利用者向けの手動掃除手順を文書化する方針とした。(2)
`safetensors_bytes`（同じく非信頼な manifest の値）は、`fstat` 実サイズ
との一致確認にしか使っていなかったつもりが、実際には読み取り時の
`take` 上限としてそのまま使われており、`fstat` の実サイズと
`safetensors_bytes` を一致させたうえで巨大なファイル＋巨大な記載値を
用意すれば、信頼できる固定上限を経由せずに大量のメモリを消費できて
しまう構造だった（P0）。是正として、`MAX_MODEL_FILE_BYTES`（`model.rs`
から共有抽出する既存の 1 GiB 定数）を**信頼できる固定上限**として先に
`fstat` 実サイズと比較し、そのあとで初めて非信頼値 `safetensors_bytes`
との一致確認を行い、実際の読み取りは常に固定上限以下と確認済みの
`fstat` 実サイズを根拠に行う三段構成へ変更した（§2 item 4・§12.3
手順 4・§13.2 手順 4〜5）。manifest 自体のサイズ上限も同じ構造の
承認事項として明示し、値を確定させずに済ませていた曖昧さを解消した。
この再是正に伴い、save・load・後片付けの全手順が扱う値について
「信頼／非信頼・使われる操作・非信頼な場合の信頼境界」を再点検した
表を新設 §13.5 に、削除の唯一の例外（自己所有一時ファイル）の安全な
削除手順を新設 §13.6 に記録した。

**世代ファイル衝突時の契約と検証計画の食い違いの是正（PR #2317 review・
2026-09-27・P2）**: §12.3 手順 1 は `model.<gen>.safetensors` の
`create_new` が `AlreadyExists` を返した場合に世代 ID を再生成して
再試行すると定める一方、当時の §6 検証計画は「最終ファイル名に既存の
シンボリックリンクがある場合に `save_model` が `Err` を返す」ことを
要求しており、シンボリックリンクも `AlreadyExists` を引き起こすため
両者は同時に満たせなかった（指摘の要約）。設計自体（衝突時は既存
エントリに一切触れずに再試行し、上限到達後にのみ `Err` を返す）を
正としたうえで §6 を是正し、(1) 再試行の上限内で成功する場合（衝突した
既存エントリは不変のまま、別世代で保存に成功する）と (2) 全再試行が
衝突する場合（`Err` を返し、既存エントリは不変）の 2 ケースへ書き換えた
（テストは世代 ID・一時ファイル名生成を差し替え可能な内部フックで衝突を
注入する）。再試行の上限回数は新規の数値を発明せず、リポジトリ内の同型
パターン（`create_new` 衝突時に次の候補名へ retry する一時ファイル
作成）である `crates/docs-site/src/build.rs::write_file_creating_parent`
の `MAX_TMP_NAME_ATTEMPTS`（8 回）にそのまま揃えた（旧版の「上限 5 回」は
出典のない値だったため置き換えた）。同じ是正の一環として、固定名
`manifest.json` 側の一時ファイル作成にも同じ粒度（衝突時の再試行・
同一上限）を明記し、既存 `manifest.json` がシンボリックリンクである
場合に `rename` がリンクエントリ自体を置換し参照先へは書き込まないこと
（POSIX `rename(2)` の宛先非追従の性質）も §12.3 手順 2・§13.3 へ明記した。
この是正のために文書内の全手順（save・load・一時ファイル・削除・各
検証段階）と §6 の検証計画を突き合わせ、他に食い違いがないことを
確認した（対照表は PR 説明・レビュー記録側に記載し、本文には結論のみを
反映する）。

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
| BatchNorm `running_mean`／`running_var` | 可（buffer） | 可（`BatchNorm1d/2d::from_parameters`） | #2371 で結線済み（`{i}.running_mean`／`{i}.running_var`＋manifest `buffer_keys`） |
| BatchNorm `num_batches_tracked` | 可（`num_batches_tracked()`。crate 内限定） | 不可（`from_parameters` に対応引数がなく setter もない） | **意図的に非復元のまま**。forward 計算のどこからも参照されず（`batch_norm.rs:395-396` で加算されるだけで、読み出し箇所は同ファイルの getter とテストのみ）、数値へ影響しないため復元 API は追加しない |
| `AdamW`／`Adam`／`RmsProp`／`Adagrad`／`Lamb`／`Adadelta`／`Adamax`／`NAdam`／`RAdam` の内部状態 | 可（`OptimizerStateDict::state_dict()`） | 可（`load_state_dict()`。検証付き） | 変更なし |
| `Sgd` の `velocity` | 内部 API 実装済み（#2367。manifest 結線は #2372） | 不可 | §2 item 2 |
| `Lbfgs` の `n_iter`／`func_evals`／`d`／`t`／`old_dirs`／`old_stps`／`ro`／`h_diag`／`prev_flat_grad`／`last_loss`／`slot_shapes` | 一部可（`n_iter()`／`func_evals()`／`last_loss()`／`config()` のみ公開） | 不可（他フィールドに setter がなく、`OptimizerStateDict` も未実装。実装するとしても既存トレイトが前提とする per-param スロットバッファ形状〈`AdamW` の `m`／`v` 等〉とは構造が異なる〈フラット化ベクトル 1 本＋曲率ペア履歴〉） | §2 item 2 拡張（承認後、`Lbfgs` 専用キー配置の状態保存・復元 API を新設。§4「Lbfgs 状態」節） |
| `GradScaler` の `scale`／`growth_tracker` | 可（`scale()`／`growth_tracker()`） | **不可**（`new` は `init_scale` からしか開始できず、`update` は backoff／growth の状態機械経由でしか変化しない。`growth_tracker` は成長／backoff のたびに `0` へリセットされるため、`update(false)` を事後に何回再生しても、再生前の backoff／growth で変化済みの `scale` 自体は再現できない——指摘 1 の対象） | §2 item 3（承認後 `grad_scaler_from_state(config, scale, growth_tracker)` を新設。検証は `new` 相当＋`scale` の非正規化数チェック＋`growth_tracker < growth_interval`） |
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
   （変更不要）を呼び、**`model.<gen>.safetensors` へ
   `OpenOptions::new().write(true).create_new(true)`（Rust std のみ。
   追加依存なし。存在すればシンボリックリンクか否かを問わず `Err` を
   返す）で直接書き込む（tmp ファイル＋`rename` を経由しない）**（PR
   #2317 review 再確認・2026-09-27 第 2 回是正。§2 item 5。世代 ID が
   一意である限り最終ファイル名への直接 `create_new` で「他プロセスの
   既存ファイルへの追従書き込み」も「衝突」も同時に防げるため、
   一時ファイル＋`rename` という中間段階自体が不要と判断した——rename
   を挟む旧案は、rename 先の衝突検出という点では `create_new` 直書きと
   同じ保証しか持たず、手順が 1 段増えるだけだった）。**`create_new` が
   `AlreadyExists` を返した場合（既存エントリの種類は問わない——通常
   ファイル・シンボリックリンク〈有効・dangling いずれも〉・ディレクトリ・
   特殊ファイルのいずれであっても同じ `AlreadyExists` になる。天文学的に
   低確率）は、その既存エントリには一切触れず（読まない・削除しない・
   追従しない）、新しい `<gen>` を再生成して再試行する**（上限
   [`MAX_TMP_NAME_ATTEMPTS`]〈8 回〉——新たに数値を決めず、リポジトリ内の
   同型パターン〈`create_new` が衝突したら次の候補名を生成して再試行する
   一時ファイル作成〉である `crates/docs-site/src/build.rs::
   write_file_creating_parent` の `MAX_TMP_NAME_ATTEMPTS` にそのまま揃える。
   上限に達してもなお衝突する場合は `ModelIoError::Io` で fail-closed とし
   `dir` の既存エントリは不変のまま返す——PR #2317 review 指摘（P2）の
   是正。§6「承認後の検証計画」・§13.3 の対応行も本節と同じ挙動に揃える）。
   部分書き込みで失敗した場合、このファイルは以後どの manifest からも
   参照されない孤立ファイルとして残る（§13.6「削除所有権」。自動削除は
   しない）。
2. **manifest に世代情報を追加する**（§4 のスキーマへ
   `"safetensors_file"`〈文字列。`model.<32桁16進>.safetensors` の
   完全一致パターンのみ許可——パス区切り文字を含む値は load 側で即
   `Err`〉と `"safetensors_bytes"`〈u64。保存直後に実際に書き込んだ
   バイト数〉を追加）。manifest 自体は固定名 `manifest.json` のまま、
   一時ファイル＋`rename` を使う（PR #2317 review 再々確認・指摘 1 の
   是正。旧版は `st_save.rs` と同型の `std::fs::write` ベースの一時
   ファイル作成だったため、同じ追従書き込みの欠陥を持っていた）。
   **一時ファイル名の衝突時の扱いは手順 1（safetensors の世代 ID
   衝突）と同じ粒度に揃える**（PR #2317 review 指摘〈P2〉の是正。
   一時ファイル名は `.manifest.json.tmp-{pid}-{カウンタ}-{nanos}`
   〈`crates/docs-site/src/build.rs::write_file_creating_parent` の
   命名パターンと同型。プロセス内 `AtomicU64` カウンタ＋
   `SystemTime::now()` のナノ秒＋`std::process::id()` を連結し、同一
   プロセス内の並行呼び出し・過去の残骸との衝突を避ける〉とし、
   `create_new` が `AlreadyExists` を返した場合は既存エントリに触れず
   次の候補名を生成して再試行する（上限は手順 1 と同一の
   [`MAX_TMP_NAME_ATTEMPTS`]〈8 回〉。上限に達してもなお衝突する場合は
   `ModelIoError::Io` で fail-closed とし、書き込み前の状態のまま
   返す）。一時ファイルの作成に成功した後の書き込み・`rename` の
   扱いは変更しない。
   **`rename` の置換先（固定名 `manifest.json`）が既存のシンボリック
   リンクである場合の挙動**（PR #2317 review 指摘〈P2〉の是正で明記。
   §13.3 に対応行を追加）: `std::fs::rename`（POSIX `rename(2)`相当）は
   宛先のディレクトリエントリ自体を置き換えるのであって、宛先が
   シンボリックリンクであってもリンクの参照先を辿って書き込むことは
   ない。したがって既存 `manifest.json` がシンボリックリンクであっても、
   `rename` はそのリンクエントリ自体を新しい通常ファイルへ置換するのみで、
   リンクの参照先ファイルには一切触れない・書き込まない。これは §13.2
   の読み取り側 no-follow 手順（シンボリックリンクを拒否する）とは
   対象が異なる（読み取り側は「シンボリックリンク越しに他のファイルを
   開いてしまう」ことが脅威だが、`rename` の置換先としてのシンボリック
   リンクは常に置換対象のエントリそのものであり、リンク先への意図しない
   書き込みという脅威が構造的に存在しない）ため、書き込み側でこのケースを
   拒否する理由はなく、通常の置換として扱ってよい（既存の通常ファイルの
   置換と同じ「最後に勝った manifest が有効になる」正常な更新経路。
   §12.3 手順 6）。**この宛先非追従の性質は Linux／macOS の POSIX
   `rename(2)` について確認したものであり、Windows の `MoveFileExW`
   （reparse point〈シンボリックリンク／junction〉が置換先にある場合の
   挙動）は一次資料で確定できない**。このため Windows（非 unix）では
   手順 8 のとおり `save_model` 自体を fail-closed にし、本手順の
   rename 経路へ到達させない（イシュー #2368 で決定。根拠・棄却案・
   緩和条件・#2369 実装要件は §12.4）。
3. **manifest の rename が唯一のコミット点である根拠**: 手順 1 の
   `create_new` による `model.<gen>.safetensors` への直接書き込みが
   完了した時点で（PR #2317 review 再々確認・指摘 1 の是正により、
   この書き込みは tmp＋rename を経由しない直接書き込みへ変更済み——
   §12.3 手順 1）、その世代のファイルは完全な内容で存在し、かつ**同名で
   上書きされることが二度とない**（世代 ID
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
   `fstat` 実バイト数をまず**信頼できる固定上限**（`MAX_MODEL_FILE_
   BYTES`。§2 item 4）と比較し、上回れば読み取りに入る前に
   `ModelIoError::TooLarge` で拒否する（固定上限との比較のみが資源
   確保量を決める根拠であり、manifest の `safetensors_bytes` は
   この時点ではまだ使わない）。固定上限の範囲内であることを確認して
   から、初めて非信頼値である manifest の `safetensors_bytes` との
   **一致**を確認する（不一致なら `ModelIoError::Mismatch`。他
   プロセスによる再保存でファイルが差し替えられた・部分書込みの残骸を
   掴んだ、等の可能性を fail-closed に拒否する）。一致確認後、同じ
   ハンドルから `Read::take(fstat 実長 + 1)` で読み、実際に読めた
   バイト数が `fstat` 実長とちょうど一致することを確認する（読み取り中
   の増大を検出する。§13.2 手順 4〜5・PR #2317 review 再確認・2026-09-27
   第 2 回是正——旧版は `safetensors_bytes`〈非信頼値〉を `take` の上限に
   使っており、巨大な `safetensors_bytes` を記載した非信頼 manifest と
   実際に巨大なファイルを用意すれば `MAX_MODEL_FILE_BYTES` を経由せず
   大量のメモリを消費し得た——指摘 2 の対象）。検証通過後は既存どおり
   内部のキー集合・shape 検証に進む（§5「load の手順」相当）。
5. **旧世代の後片付けは行わない（§13.0・§13.6）**: `save_model` は
   `dir` の既存ファイルを読まず、削除もしない（PR #2317 review 再確認・
   2026-09-27 第 2 回是正で撤回。旧版は「直前に読んだ旧 manifest が
   参照していた世代 1 件のみを削除」としていたが、これは非信頼な旧
   manifest の値——攻撃者が書き換えられる `safetensors_file`——を
   削除対象の決定根拠にしており、命名パターンに一致する無関係な
   ファイルを指定して削除させられる余地が残っていた〈指摘 1 の対象。
   `symlink_metadata` による実体確認は「削除してよい対象かどうか」を
   何も証明しない〉。所有権を暗号学的に証明する手段〈鍵・MAC〉は依存
   追加なしでは過剰なため、自動削除機能自体をなくした）。**結果として
   `manifest.json` から参照されなくなった世代の `model.<gen>.
   safetensors` は孤立ファイルとしてディスクに残り続ける**——これは
   明示的に受容する残余コストであり、利用者向けの掃除手順として
   「`manifest.json` の `safetensors_file` が指す世代以外の
   `model.*.safetensors` は安全に手動削除できる」旨を API ドキュメント
   （承認後に追加する `model_io` モジュール doc）に明記する。
6. **同時保存（2 プロセス）の扱い**: 本設計は「同一ディレクトリへの
   並行 `save_model` はサポート対象外」と明示する（`st_save.rs:185`
   の既存の未サポート表明と同型）。世代 ID が一意である限り、2 つの
   並行 save は互いの safetensors ファイルを破壊しないが、manifest
   （固定名）は最後に rename した側が勝つ。手順 5 で自動削除をなくした
   ため、削除に起因する競合（誤削除・二重削除）はそもそも発生しない。
   **残るのは「最後に勝った manifest 以外が参照していた世代が孤立して
   残ること」のみ**であり、これは手順 5 の残余コストと同じ性質の
   「非サポート」として受容する（呼び出し元が同一ディレクトリへ並行
   書き込みしない前提での動作を保証する）。
7. **読込中の書込み（reader/writer race）**: load は「manifest を
   読む → 参照されたファイルを §13 の no-follow 手順で開く」の順で
   処理する。手順 5 で自動削除をなくしたため、**load が読んだ世代が
   save の後片付けによって消される可能性はそもそもない**（PR #2317
   review 再確認・2026-09-27 第 2 回是正でこの race は解消済み。旧版は
   「manifest を読んでから open するまでの間に世代が cleanup 対象に
   なり `ENOENT` になり得る」残余レースを受容として記録していたが、
   cleanup 自体をなくしたため不要になった）。残る事象は「load が
   manifest を読んだ後、別プロセスの並行 save が新しい manifest への
   `rename` を先に終え、その後 load が今読んだ manifest の内容で
   safetensors を開こうとする」という通常の読み書き競合のみであり、
   世代 ID が一意なため参照先ファイル自体は存在し続ける（消えるのは
   「latest」の意味だけで、読んだ世代の内容自体は不変のまま安全に
   読める）。単一プロセスが読み書きする通常運用では発生せず、並行
   アクセスは呼び出し元でファイルロック等の直列化を行う前提とする。
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
   側についてのみ撤回する変更である。**`save_model` 側も同じく非 unix では
   fail-closed に拒否する**（`ModelIoError::Io` / `ErrorKind::
   Unsupported`。`dir` へのあらゆる副作用より前に判定する）。旧版は
   `create_new` が Windows でも既存パスで `Err` になることを根拠に
   「save は全 OS 対応・load は Linux／macOS 限定」という非対称性を
   認めていたが、イシュー #2368 で撤回した。`rename` の置換先が
   reparse point の場合の Windows 挙動を確定できないこと、Windows では
   `load_model` が拒否されるため roundtrip が成立しないこと、§13.6 を
   std だけで実装できないことが理由である（詳細は §12.4）。
9. **電源断耐性（fsync）は対象外のまま**: `st_save.rs` は一時ファイル
   への `fsync`（`File::sync_all`）を行わないため、rename 成功後の
   電源断・OS クラッシュに対する耐性を保証しない（既存注記のとおり。
   本設計もこれを変更しない）。本設計が閉じるのは「プロセスの通常
   終了・異常終了（クラッシュ・kill）を含む、ファイルシステムが応答
   している間の 2 ファイル間コミット順序の不整合」（指摘 2 が対象と
   した問題）であり、ストレージ層の電源断耐性という別軸の非保証は
   既存方針から変更しない。

### 12.4 Windows の rename 置換先 reparse point の扱い（イシュー #2368）

**決定: (a) Windows（`cfg(not(unix))`）では `save_model` も fail-closed に
する**（`ModelIoError::Io(io::Error::from(ErrorKind::Unsupported))`）。
§12.3 手順 8 の「save は全 OS・load は Linux／macOS」という非対称性は
撤回した。

**1. 調査結果（出典付き）**

| 項目 | 内容 | 出典 | 確度 |
|------|------|------|------|
| Rust std 1.98.1 の `fs::rename`（Windows） | まず `MoveFileExW(old, new, MOVEFILE_REPLACE_EXISTING)` を呼ぶ。`ERROR_ACCESS_DENIED` で失敗したときに限り、`old` を `DELETE` アクセス＋`FILE_FLAG_OPEN_REPARSE_POINT \| FILE_FLAG_BACKUP_SEMANTICS` で開き直し、`SetFileInformationByHandle(FileRenameInfoEx)`（`FILE_RENAME_FLAG_REPLACE_IF_EXISTS \| FILE_RENAME_FLAG_POSIX_SEMANTICS`）で再試行する 2 段構成 | rust-lang/rust タグ `1.98.1`・`library/std/src/sys/fs/windows.rs` `pub fn rename`（L1321〜L1387 付近。計画時点の確認。`rust-toolchain.toml` は stable 追従のため将来の版で変わりうる） | 明文（ソース） |
| `OpenOptions::create_new` | `CREATE_NEW` を選び `FILE_FLAG_OPEN_REPARSE_POINT` を自動付与するため、既存の symlink〈dangling を含む〉があれば `AlreadyExists` になる想定 | 同 `get_flags_and_attributes`（L317・L329〜L333 付近） | ソース上の推定。実機は未確認（W-save-4） |
| 置換先が**ファイル symlink** の場合 | リンク自体の置換か参照先への書き込みかを、MS Learn の明文では確定できない | MS Learn「MoveFileExW」・「Symbolic Link Effects on File Systems Functions」（主に移動元が symlink の場合を記述） | 未確認（W-save-2） |
| 置換先が**ディレクトリ symlink** の場合 | 上に加え、ディレクトリ属性を持つ置換先が `ERROR_ACCESS_DENIED` を経て POSIX semantics 経路へ落ちる場合の挙動も不明 | 同上 | 未確認（W-save-2・W-save-3） |
| 置換先が**junction** の場合 | 同上 | 同上 | 未確認（W-save-2・W-save-3） |

明文出典のない挙動は推定を事実として書かず「未確認」とし、Windows 実機での
確認を #2393 へ申し送る（本節 6）。

**2. 3 案の比較**

| 観点 | (a) fail-closed（採用） | (b) 宛先を `FILE_FLAG_OPEN_REPARSE_POINT` で検査して拒否 | (c) std の意味論のまま許可 |
|------|------|------|------|
| 参照先への書き込みリスク | なし（経路ごと遮断） | 検査〜`rename` 間の TOCTOU が構造的に残る（宛先は名前解決されるため、検査ハンドルと置換対象を一体化できない。`FILE_SHARE_DELETE` なしで保持すると rename 自体が共有違反になる） | 意味論が未確認のため評価不能 |
| `unsafe` | 不要 | 検査自体は std＋`MetadataExt::file_attributes()` で可能だが、§13.6 の Windows 実装で FFI が要る | §13.6 の Windows 実装で FFI が要る |
| 新規依存 | なし | なし | なし |
| §13.6 との整合 | 経路自体が不要 | `(dev, ino)` 相当（`file_index`）は 1.98.1 で nightly 限定のため kernel32 FFI が必要 | 同左 |
| roundtrip | 成立しない（ただし `load_model` も Windows で拒否済みで現状と同じ） | 成立しない | 成立しない（load が拒否） |

(b) は「意味論が安全なら検査は不要、危険なら検査では閉じられない」ため (c)
に支配される。(c) は前提の「std の意味論で安全」を明文出典・実機で確認できて
いないため採らない。

**3. 決定の帰結**

- `unsafe` FFI・新規依存は不要（FFI を要する案は選んでいない）。したがって
  security-auditor の追加監査要件は発生しない。#2369 の実装 PR は
  `security.md` のレビュー体制どおり通常の監査を行う
- Windows で実際に効くのは facade が Windows でビルド可能になる #2389〜#2391
  以降である

**4. 緩和条件（(c) へ移る条件。変更はユーザー承認必須）**

次の 3 点がすべて満たされること。

1. ファイル symlink／ディレクトリ symlink／junction／その他の reparse タグ
   （AppExecLink・cloud placeholder 等）のすべてで、`rename` が参照先へ書き込ま
   ないこと（リンク自体の置換またはエラー）を実機と明文出典で確認する
2. §13.6 の Windows 実装手段（`unsafe` を伴う場合は監査要件を別途定める）
3. `load_model` の Windows 対応

**5. #2369 実装要件（正本。実装済み: `model_io.rs` の `save_platform_supported(is_unix)`〈純関数。
`save_model` の冒頭で `dir` への副作用より前に呼ぶ〉・単体テスト
`non_unix_platform_is_rejected_as_unsupported`・`tests/compat_sequential_model_io.rs` の
`#[cfg(not(unix))]` テスト・公開 doc への記載）**

- `save_model` の冒頭で、`dir` へのあらゆる副作用（`create_dir_all`・
  `create_new`・`rename`）より前に、`cfg(not(unix))` のとき
  `Err(ModelIoError::Io(io::Error::from(ErrorKind::Unsupported)))` を返す
  （`dir` を一切変更しない）
- 判定は cfg 分岐する小さな関数（例: `save_model_platform_supported() ->
  io::Result<()>`）にまとめ、`load_model` の Unsupported 経路と同じ語彙の
  エラーにする
- 回帰テスト: (i) `#[cfg(not(unix))]` のテストで `save_model` が
  `ErrorKind::Unsupported` を返し `dir` のエントリが増えないことを検証する
  （Linux CI では実行されず、Windows 向けクロス clippy `--tests` による型検査
  のみ。facade のクロス clippy は #2391 以降に有効になる）。(ii) Linux でも
  拒否ロジックを検査したい場合は、`onnx-interop` の
  `windows_component_reject_reason`（`cfg(any(windows, test))`）の先例に倣い
  非 unix 判定の純関数を `cfg(any(not(unix), test))` で Linux のテストビルドに
  も含める
- 公開 doc（`model_io` モジュール doc・`save_model` doc）に「Windows では未対応
  （fail-closed）。理由と緩和条件は決定記録 §12.4」と記載する
  （`crates/facade/src/model.rs`「Windows 対応状況」節と同型）。intra-doc link は
  公開項目にだけ張る

**6. Windows 実機確認項目（#2393 への申し送り）**

- W-save-1: `save_model` が `ErrorKind::Unsupported` を返し `dir` に何も作らない
  こと（#2369・#2391 の後）
- W-save-2: 緩和検討用。`manifest.json` を ファイル symlink（有効／dangling）・
  ディレクトリ symlink・junction・その他の reparse タグ（可能なら AppExecLink・
  OneDrive placeholder）にして `std::fs::rename(tmp, manifest.json)` を実行し、
  参照先の内容・タイムスタンプが不変か、置換されたか、エラー（`ErrorKind`／OS
  エラー）かを記録する
- W-save-3: 上記で `MoveFileExW` が `ERROR_ACCESS_DENIED` を返し
  `FileRenameInfoEx` の POSIX semantics fallback 経路に入るケースの特定と、その
  経路での同じ観測
- W-save-4: `OpenOptions::create_new` が dangling symlink・junction の位置で
  `AlreadyExists` になること
- 記録事項: NTFS 必須（可能なら ReFS も）・Windows と Rust toolchain の版。結果の
  反映先は本節

## 13. ファイル I/O 脅威の全数棚卸し（C。PR #2317 review 再々確認・
2026-09-27・指摘 1・2 の是正に伴う網羅確認）

### 13.0 原則（PR #2317 review 再確認・2026-09-27 第 2 回の是正）

**`dir` 配下の全ファイルと manifest の全フィールドは非信頼である。**
破壊的操作（削除・上書き）・資源確保量（メモリ確保・読み取りバイト数）・
信頼判定（このファイルは本モジュールが書いたものだ、という判定）は、
非信頼値のみを根拠にしてはならない。非信頼値は、**信頼できる固定の
上限・固定パターンで挟んだうえで、照合（一致するかどうかの確認）にだけ
使う**。この原則から導かれる本設計の帰結は次の 2 点である（詳細は
以下の各節）。

1. **削除**: 所有権を暗号学的に証明する仕組み（鍵・MAC 等）は依存
   追加なしでは過剰であるため、**`save_model` は旧世代ファイルを
   自動削除しない**設計に改める。例外は「今回の `save_model` 呼び出し
   自身が `create_new` で作成し、同一呼び出し内で以降の手順が失敗した
   一時ファイル」のみであり、これは作成時に得た `File`（オープン済み
   fd）を手順の最後まで保持しているために所有が証明できる（§13.6
   「削除所有権」）。
2. **読み込み上限**: manifest・safetensors のいずれの読み取りも、
   非信頼値（manifest 記載のバイト数）を確保量やループ回数の根拠に
   しない。まず信頼できる固定上限（コード定数）に対して `fstat` の
   実測値を照合し、次に非信頼値（manifest 記載値）との**一致**だけを
   確認し、最後に実測値（固定上限で既に有界）を根拠に読む（§13.2）。

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
4. 開いたハンドルの `fstat` で得たファイルサイズを、**信頼できる固定
   上限**（manifest は承認済みの固定定数〈1 MiB。§2 item 4〉、
   safetensors は `MAX_MODEL_FILE_BYTES`〈`model.rs` から `fs_guard` へ
   共有抽出する既存の 1 GiB 定数。§2 item 4・§13.4〉）とまず比較し、
   上回る場合は読み取りに入る前に拒否する（この判定は非信頼値を一切
   経由しない）。safetensors についてはこの後さらに、非信頼値である
   manifest 記載の `safetensors_bytes` との**一致**を確認する（§12.3
   手順 4）。**この一致確認は上限判定ではなく等価判定にすぎず、
   `safetensors_bytes` 自体を確保量・読み取り量の根拠にはしない**
   （PR #2317 review 再確認・2026-09-27 第 2 回是正。指摘 2 の対象——
   旧版は `fstat` 実サイズと `safetensors_bytes` の一致を見ているだけで、
   `take` の上限に `safetensors_bytes`〈非信頼値〉を使っていたため、
   固定上限を経由しない大量メモリ消費を許していた）。
5. 同じハンドルから `std::io::Read::take(fstat 実長 + 1)`（fstat 実長は
   手順 4 で固定上限以下と確認済みの信頼できる値）で読み、実際に
   読めたバイト数が `fstat` 実長とちょうど一致することを確認する
   （読み取り中の増大〈TOCTOU〉を検出する）。

### 13.3 脅威棚卸し表

| 脅威 | 対象（段階） | 本設計の対策（承認後の実装要件） | 既存実装の参照先 |
|---|---|---|---|
| シンボリックリンク（葉ファイル。`manifest.json`／`model.<gen>.safetensors`） | 読み込み | §13.2 の no-follow 手順（`symlink_metadata` 事前拒否 → `O_NOFOLLOW` オープン → `fstat` dev/ino 照合） | `crates/facade/src/fs_guard.rs::open_leaf_no_follow`・`crates/facade/src/model.rs::resolve_model_file`／`crates/facade/tests/model_registry.rs::load_rejects_symlinked_leaf_file_escaping_root` |
| シンボリックリンク（対象ディレクトリ自身 `dir`） | 読み込み・書き込み共通 | 許容する（`dir` 自体が symlink であることは脅威モデル外。§13.1） | `model_registry.rs::load_succeeds_when_root_itself_is_a_symlink` |
| シンボリックリンク（途中のパス要素） | — | 該当なし（`dir` 直下 1 段のみを扱うレイアウトのため中間ディレクトリが存在しない。§13.1） | — |
| シンボリックリンク（一時ファイル名・最終ファイル名の位置に事前配置） | 書き込み | `create_new`（`O_EXCL` 相当。存在すれば symlink か否かを問わず `Err`）で作成し追従書き込みを構造的に防ぐ。衝突時は既存エントリに触れず新しい候補名で再試行する（上限 [`MAX_TMP_NAME_ATTEMPTS`]〈8 回〉。上限到達後もなお衝突する場合のみ `Err` を返し、既存エントリは不変） | 新設（§2 item 5・§12.3 手順 1〜2。PR #2317 review 指摘〈P2〉の是正） |
| シンボリックリンク（固定名 `manifest.json` への `rename` 置換先。Linux／macOS の POSIX `rename(2)` 限定） | 書き込み | `rename` は宛先ディレクトリエントリ自体を置換するのみで宛先シンボリックリンクの参照先を辿らないため、リンクエントリを新しい通常ファイルへ安全に置換できる（参照先ファイルには書き込まない）。読み取り側 no-follow 手順とは対象が異なる別種の安全性のため拒否は不要。Windows（非 unix）は置換先 reparse point の挙動を確定できないため `save_model` 自体を fail-closed とする（§12.4。イシュー #2368） | 新設（§12.3 手順 2・§12.4。PR #2317 review 指摘〈P2〉の是正） |
| ハードリンク | 読み込み・書き込み共通 | 対象外として受容（攻撃者が作成できるのは同一ファイルシステム上の既存ファイルへのリンクのみで、所有者・権限チェックを伴わない本モジュールの脅威モデル外） | `model.rs` モジュール doc「対象外として残る経路」節の理由をそのまま踏襲 |
| 特殊ファイル（FIFO・Unix ソケット・デバイス） | 読み込み | `symlink_metadata`／`fstat` の両方で `is_file() == true` を要求し拒否。`O_NONBLOCK` で FIFO への差し替えによる無期限ブロックも防ぐ | `fs_guard.rs::open_leaf_no_follow`／`model_registry.rs::load_rejects_non_regular_leaf_unix_socket` |
| 特殊ファイル（削除候補） | 削除 | **該当なし**（PR #2317 review 再確認・2026-09-27 第 2 回是正で自動削除機能自体を撤回。§13.0・§13.6） | — |
| Windows reparse point／junction | 読み込み | no-follow の安全な実装を持たないため `load_model` を fail-closed 拒否（`ErrorKind::Unsupported`） | `model.rs`「Windows 対応状況」節・§12.3 手順 8 |
| Windows reparse point／junction | 書き込み | `save_model` を fail-closed 拒否（`ErrorKind::Unsupported`。`dir` へ副作用を起こす前に判定）。`create_new` の契約と `rename` 置換先の挙動を Windows 実機で確認するまで許可しない | §12.4・§12.3 手順 8（実装は #2369） |
| TOCTOU（検査〜open の間の差し替え） | 読み込み | 開いたハンドルの `fstat` を `symlink_metadata` の実体識別子（`dev`／`ino`）と照合し、差し替えを検出する（パス再解決ではなく実体同一性で判定） | `model.rs`「検査と open のハンドル一体化」節 |
| TOCTOU（fstat 後の読み取り中の増大） | 読み込み | 同一ハンドルから `take(fstat 実長 + 1)` で読み、実読バイト数が `fstat` 実長と一致しなければ拒否する（§13.2 手順 5） | `model.rs` 手順 6 |
| TOCTOU（一時ファイル作成） | 書き込み | `create_new` は「存在確認」と「作成」を単一のシステムコールで行うためレースが原理的に生じない（std ドキュメントが明記する atomic 操作） | 新設 |
| パストラバーサル（manifest 内の `safetensors_file`） | 読み込み | `model.<32桁16進>.safetensors` の完全一致パターンのみ許可。パス区切り文字・`..`・絶対パスを含む値は即 `Err`（ただし §13.2 の no-follow 手順と併用しない限りシンボリックリンク経由の脱出は防げない点に注意。§13.3 上段） | §4「ファイル形式」・§8 A01 |
| パストラバーサル（manifest 内の層構成・キー名等） | 読み込み | ファイルシステムへ渡さない値（層種別・パラメータキー名は state_dict のキー照合にのみ使う）のため対象外 | §4「ファイル形式」 |
| 読み込みサイズ上限（manifest） | 読み込み | **信頼できる固定上限**（コード定数。§2 item 4・承認済み）に `fstat` 実長を比較してから読み取りに入る。`take(fstat 実長 + 1)` で確保・パース前に検証（§13.2 手順 4〜5） | §8 A03 |
| 読み込みサイズ上限（safetensors ヘッダ長・データ長） | 読み込み | まず `fstat` 実長を**信頼できる固定上限**（`MAX_MODEL_FILE_BYTES`。§2 item 4）と比較して拒否判定し、通過後に非信頼値である manifest の `safetensors_bytes` との**一致**を確認する（§12.3 手順 4）。`safetensors_bytes` 自体を確保量の根拠にはしない。safetensors 自体のヘッダ検証は既存 `load_safetensors_f32_from_bytes` に一元化（複製・迂回しない） | `crate::interop::safetensors`（`onnx-interop::st_load`） |
| 削除の所有権 | 削除 | **`save_model`／`load_model` は `dir` 内の既存ファイルを自動削除しない**（§13.0・§13.6）。旧世代・無関係ファイルはいずれも残存し、これは明示的に受容する残余コストとして API doc に記載する。所有が証明できる自己一時ファイル（作成時に得た fd を保持したまま同一呼び出し内で失敗した場合）のみ例外的に削除する | 新設（PR #2317 review 再確認・2026-09-27 第 2 回是正で自動削除機能を撤回） |
| 一時ファイル（作成方式） | 書き込み | `manifest.json` は `create_new` 一時ファイル（衝突時は既存エントリに触れず新しい候補名で再試行。上限 [`MAX_TMP_NAME_ATTEMPTS`]〈8 回〉。§12.3 手順 2）＋`rename`。`model.<gen>.safetensors` は世代 ID の一意性を利用し**最終ファイル名へ直接 `create_new`**（tmp／rename を経由しない。衝突時の再試行は同じ上限。§12.3 手順 1） | 新設。`st_save.rs` の `std::fs::write` ベース一時ファイル作成は踏襲しない（§12.3 手順 1 是正理由）。再試行上限は `crates/docs-site/src/build.rs::write_file_creating_parent` の `MAX_TMP_NAME_ATTEMPTS` に揃える（PR #2317 review 指摘〈P2〉の是正） |
| 一時ファイル（異常終了時の残骸） | 書き込み | **manifest の一時ファイルのみ**、`rename` 失敗時に限り、作成時に得た `File` ハンドルの `fstat` と削除直前の `symlink_metadata` の `(dev, ino)` が一致することを確認したうえで best-effort 削除する（§13.6「自己所有一時ファイルの削除」。差し替えを検出した場合は削除しない）。`model.<gen>.safetensors` は tmp を経由しないため、この経路の残骸自体が発生しない（部分書き込み失敗時は最終ファイル名のまま孤立するのみ）。プロセスクラッシュによる manifest 一時ファイルの残骸は削除主体が存在しないため残り続け、これも受容する | `st_save.rs`「rename 失敗時の tmp ファイル削除は best-effort」と同方針 |
| 同時保存・読込中の書込み | 読み込み・書き込み共通 | 「同一ディレクトリへの並行 `save_model`／`load_model` はサポート対象外」と明示し受容する（§12.3 手順 6〜7） | `st_save.rs:185`「同一 path への並行書き込みはサポート対象外」と同型 |

### 13.4 共有ヘルパーの抽出範囲

`open_flags`（`O_NOFOLLOW`／`O_NONBLOCK`／`ELOOP` の生値定数）と
`open_leaf_no_follow` を `model.rs` 専用の非公開実装から facade 内部の
共有モジュールへ抽出する（§2 item 5）。抽出は `model.rs` 側の既存
呼び出し・挙動を変えない純粋なリファクタリングであり、`model_io.rs`
（承認後の実装）は同じヘルパーを呼ぶだけで独自の `O_NOFOLLOW` 実装を
持たない（同じ脆弱性クラスの対策を 2 箇所に分散させない）。
`MAX_MODEL_FILE_BYTES`（1 GiB。現状 `model.rs` の private const）も
同じ抽出の対象とし `pub(crate)` へ格上げして共有する（§2 item 4）。

**書き込み側（#2369）**: `create_new`・一時ファイル名の再試行・自己所有一時ファイルの
`(dev, ino)` 照合削除は std のみで `compat/model_io.rs`（`cfg(unix)` 限定）に持ち、
`fs_guard` には置かない。読み取り側は `fs_guard::open_leaf_checked`（§13.2 手順 1〜4）と
`OpenedLeaf::read_exact_len`（手順 5）を新設して共有する（`model.rs` の手順を同ヘルパーへ
寄せる整理は挙動を変えない別作業として将来課題）。`load_model` の対応範囲
（Linux x86_64／aarch64・macOS）は unix 全体より狭いため、それ以外の unix では
`save_model` が成功しても `load_model` は拒否される（既知の非対称性）。

**抽出済み（イシュー #2364）**: `open_flags`・`open_leaf_no_follow`・
`MAX_MODEL_FILE_BYTES` は `crates/facade/src/fs_guard.rs`（`crate::fs_guard`。
`lib.rs` で素の `mod` 宣言のため公開面は不変）へ移動済み。

### 13.5 非信頼値の全数再点検（D。PR #2317 review 再確認・2026-09-27
第 2 回・指摘 1・2 の是正に伴う全手順再点検）

§13.0 の原則（非信頼値のみで破壊的操作・資源確保・信頼判定を行わない）
に対し、save／load／後片付けの各手順が扱う値をすべて洗い出し、
「信頼できる固定境界を経由しているか」を確認した結果を次表に示す。
GradScaler・Lbfgs の状態復元値（§2 item 2・3・§11）も非信頼な manifest
入力であるため同じ表に含める。

| 値 | 出所（信頼／非信頼） | 使われる操作 | 信頼境界（非信頼の場合） |
|---|---|---|---|
| `dir`（呼び出し元引数） | 信頼（呼び出し元がプロセス内で直接渡す。ネットワーク越しの入力ではない） | `create_dir_all`／全ファイル操作の起点 | 信頼済みのため境界不要。ただし配下の中身は非信頼（§13.1） |
| `dir` 配下のファイル名・種別（既存ファイル一覧） | **非信頼**（第三者が事前配置し得る） | 読み込み時の open 対象決定 | 固定パターン 2 種のみ（`manifest.json`／`model.<32桁16進>.safetensors`）に一致する名前しか扱わず、実際に開く際は §13.2 no-follow 手順（symlink・特殊ファイル拒否＋TOCTOU 検出）を経由する |
| `dir` 配下の既存ファイル（削除対象としての利用） | **非信頼** | （行わない） | **§13.0 の原則により削除の根拠に一切使わない**——`save_model` は既存ファイルを読みも削除もしない（§12.3 手順 5・§13.6） |
| manifest.json のバイト列そのもの | 非信頼 | パース前の読み取り量 | `fstat` 実長を**信頼できる固定上限**（承認済み。§2 item 4）と比較してから `take(実長 + 1)` で読む。実長自体は非信頼な manifest の内容ではなく OS が返す事実（fstat）であり、固定上限で挟まれているため境界内 |
| `format`／`format_version` | 非信頼 | 分岐（受理／拒否） | 固定文字列・固定整数との完全一致のみ許可。不一致は即拒否（確保・破壊操作を伴わない単純な等価判定） |
| `num_layers`／`layers`（層構成） | 非信頼 | 層オブジェクトの構築（メモリ確保を伴う） | manifest 全体が固定上限（承認済み）で有界なため要素数も有界。**加えて層数自体にも固定上限（4096・承認済み。§2 item 4）を課し、上限超過時は 1 層ずつ push する前に打ち切る**（事前に `num_layers` を信じて `Vec::with_capacity` しない） |
| `parameter_keys`／`buffer_keys`（キー集合・shape） | 非信頼 | state_dict とのキー突き合わせ（ファイルシステムへは渡さない） | 完全一致判定にのみ使用。ファイル I/O・確保量の根拠にしない（§13.3「パストラバーサル（層構成・キー名等）」） |
| `safetensors_file` | 非信頼 | open 対象パスの構築 | `model.<32桁16進>.safetensors` の完全一致パターン検証 → §13.2 no-follow 手順で open（パターン検証だけでは脱出を防げないため必ず両方を経由。§13.3） |
| `safetensors_bytes` | 非信頼 | safetensors 読み取り量の**確認**（確保量の決定には使わない） | `fstat` 実長を先に固定上限（`MAX_MODEL_FILE_BYTES`）と比較 → 通過後に非信頼値との**一致**のみ確認 → 一致後は `fstat` 実長（既に有界）を根拠に `take` する（§12.3 手順 4・§13.2 手順 4〜5） |
| `compiled.optimizer.kind` | 非信頼 | 分岐（optimizer 種別の決定） | 7 種の文字列 allowlist との完全一致のみ許可。未知の値は `UnsupportedModel`（確保操作を伴わない） |
| `compiled.optimizer.config`（各 optimizer 設定値） | 非信頼 | optimizer 構築のパラメータ | 既存 optimizer コンストラクタが行う範囲検証（`lr > 0` 等）をそのまま経由。新規の確保・破壊操作はない |
| `optimizer.history_len`（Lbfgs） | 非信頼 | 履歴件数の**期待値**（実際の確保量の根拠にはしない） | safetensors 内に実在する `optimizer.history.{i}.*` キー数（safetensors 自体が固定上限で有界）と**一致**するかどうかの確認にのみ使う。`history_len` を信じて事前確保しない（§2 item 2 是正）。加えて `<= history_size` はどちらも非信頼または呼び出し元設定値のため単なる整合性確認であり上限の代用にはしない |
| `amp.scale`（GradScaler 復元値） | 非信頼 | `grad_scaler_from_state` の引数 | 有限・正・非正規化数でないことを検証（§2 item 3）。確保・削除を伴わないスカラー値 |
| `amp.growth_tracker`（GradScaler 復元値） | 非信頼 | `grad_scaler_from_state` の引数（カウンタ） | `< config.growth_interval` を検証（§2 item 3）。カウンタ 1 個の代入のみで確保・削除を伴わないため、非信頼値どうしの範囲チェックで十分（`growth_interval` 自体は `GradScalerConfig` 側で `>= 1` 検証済みの呼び出し元設定値） |
| `d`／`t`／`h_diag`／`prev_flat_grad`／`old_dirs`／`old_stps`／`ro`（Lbfgs 状態テンソル） | 非信頼（safetensors 内容） | Lbfgs 内部状態への代入 | 各要素の有限性検証＋長さが `slot_shapes`（構築済みモデルから導出。§2 item 2）と一致することの確認のみ。confirmedな長さ以上には決して確保しない（safetensors 自体が固定上限で有界なため要素数の絶対上限も自動的に決まる） |
| `model.<gen>.safetensors` の rename／作成先パス | 信頼（コード側が生成する世代 ID。`<gen>` 自体は非信頼な外部入力から作られない） | `create_new` の対象パス | 該当なし（本モジュールが生成する値であり非信頼ではない） |

上表のとおり、非信頼値のみを根拠に削除・資源確保・信頼判定を行っている
行は 0 件であることを確認した（§13.0 の原則を満たす）。

### 13.6 自己所有一時ファイルの削除（§13.0 の唯一の削除例外）

`save_model` が `dir` の既存ファイルを削除しない（§12.3 手順 5）方針の
もとで、**唯一許容される削除**は「今回の呼び出し自身が `create_new` で
作成し、同一呼び出し内で以降の手順が失敗した一時ファイル」である
（manifest 用の一時ファイルのみ。`model.<gen>.safetensors` は tmp を
経由しないため対象外——§2 item 5・§12.3 手順 1 是正）。この削除が安全な
理由は「所有権が疑わしい非信頼ファイルの削除」ではなく「**このプロセス
がこの呼び出しの中で作成し、`File` ハンドル（オープン済み fd）を
握り続けているファイルの削除**」であるため、単なる名前・パターン一致
より強い証拠を持つ。それでも TOCTOU（作成後、削除までの間に別プロセス
がこのパスをシンボリックリンクへ差し替える可能性）を閉じるため、
削除直前に次を確認する:

1. 作成時に得た `File` ハンドルを手順の最後まで保持する（クローズ
   しない）。
2. 削除直前に対象パスを `std::fs::symlink_metadata`（リンクを辿らない）
   で検査し、通常ファイルであることを確認する。
3. 保持していたハンドルの `fstat`（`File::metadata`）と、手順 2 の
   `symlink_metadata` の実体識別子（Unix の `(dev, ino)`）が一致する
   ことを確認する。
4. 一致した場合のみ `remove_file` を呼ぶ（best-effort。失敗は無視する）。
   不一致——差し替え検出——の場合は削除しない（対象が別ファイルへ
   すり替わっている可能性があるため）。

この手順は §13.2 の no-follow 読み取り手順と同型の「検査と実体を
ハンドルで一体化する」設計であり、新しい対策パターンを追加しない。
