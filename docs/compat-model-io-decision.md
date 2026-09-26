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
   `impl OptimizerStateDict for Sgd`（`autodiff/src/optim/sgd.rs`）。
   `Sgd` は `velocity: Option<Vec<Tensor>>` を持つが `OptimizerStateDict`
   未実装（`docs/autodiff-optimizer-state-dict-decision.md`）。
   `decode_state_dict` は現状 `step_count` を必須とするため、`Sgd`
   （`step_count` を持たない）向けの引数化・専用デコーダが要る。
3. **上限値の新設**。manifest は 1 MiB、層数は 4096、GradScaler の
   tracker 再生回数は `2^24`。safetensors は model registry の
   `MAX_MODEL_FILE_BYTES`（1 GiB）を共用するか独自定義するか。
4. **受入基準からの逸脱 2 件**。ZIP ではなくディレクトリ内 2 ファイル
   （`manifest.json`＋`model.safetensors`）にする。skip connection は
   `compat::Sequential` が任意の分岐を表現できないため、
   `add_transformer_encoder`（内部 residual）で代替する。

## 3. 調査で判明した事実（設計の制約）

| 事実 | 出典 | 設計への影響 |
|---|---|---|
| facade は `serde`／`serde_json` に依存していない | `crates/facade/Cargo.toml` | manifest JSON は手書きの厳格パーサ（固定スキーマ）で扱う。ZIP 用クレートもないため ZIP は不採用 |
| `Sequential` は層を `Box<dyn Module>` で持つ。`Module` のダウンキャストは不完全 | `crates/facade/src/compat/sequential.rs`・`crates/autodiff/src/nn/module.rs` | 各 `add_*`（30 メソッド）の引数を内部 `LayerSpec` enum として記録する方式にする |
| BatchNorm の running stats は `state_dict()` に含まれない。setter がなく、`BatchNorm1d/2d::from_parameters` でのみ復元できる | `crates/autodiff/src/nn/batch_norm.rs` | buffer は safetensors に別キーで保存し、BN 層は `from_parameters` で再構築する。`num_batches_tracked` は非復元 |
| `Compiled { optimizer, loss, amp }`。6 optimizer はいずれも `config()` を持つ | `crates/facade/src/compat/training.rs` | 設定値は全フィールドを直列化できる |
| `AdamW`／`Adam`／`RmsProp`／`Adagrad`／`Lamb` は `OptimizerStateDict` 実装済み。`Sgd` は未実装 | `crates/autodiff/src/nn/optim/state_dict.rs` | 承認後は `impl OptimizerStateDict for Sgd` が必要 |
| `api_surface.rs::workspace_declares_optimizer_state_dict_fn_names_only_in_allowed_locations` が `state_dict`／`load_state_dict` の宣言元を完全一致で固定している | `crates/facade/tests/api_surface.rs` | 承認後の facade 側ヘルパーにこの名前は使えない |
| `GradScaler` に `config()` と状態復元用コンストラクタがない | `crates/autodiff/src/nn/optim/amp.rs` | `compile_with_amp` の時点で `GradScalerConfig` を記録し、復元は `new` の後 `update(false)` を `growth_tracker` 回再生する |
| `compat::Sequential` では任意の skip connection を表現できない | `sequential.rs` | 受入基準の「skip connection」は深い異種スタック＋`add_transformer_encoder` で代替 |
| `save_model`／`load_model` を宣言している `crates/*/src/` はない | `grep` 結果（2026-09-26） | 定義元インベントリの期待集合は空 |

## 4. ファイル形式（形式バージョン 1）

- `dir/model.safetensors`（F32 のみ）。キー名前空間 3 種:
  - パラメータ: `{i}.{name}`（`Sequential::state_dict` と同じ）
  - BN の buffer: `{i}.running_mean`／`{i}.running_var`
  - optimizer 状態: `optimizer.` 接頭辞＋`OptimizerStateDict::
    state_dict()` のキー
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
    "compiled": null
  }
  ```
  `compiled` は `{"loss", "optimizer": {"kind", "config"}, "optimizer_state_keys", "amp"}`。
- 数値表現: f32 は Rust の最短往復表記（`{:?}`）で書き `str::parse::<f32>`
  で読む（非有限値は save 時に `UnsupportedModel` で拒否）。u64／usize は
  JSON の整数として書き独自パーサで読む。

## 5. 意味論

- **LayerSpec**: compat 内部に `LayerSpec` enum を置き、`add_*` 30 種と
  1 対 1 対応させる（引数・seed を保持）。`Sequential` に private
  フィールド `specs: Vec<LayerSpec>` を追加し、各 `add_*` で push する
  （フィールドは private なので公開 API は非破壊）。
  `specs.len() != layers().len()` なら `UnsupportedModel`。
- **save の手順**: 検証をすべて終えてから書き込みに入る。
  `create_dir_all` → safetensors → manifest の順に一時ファイル＋
  `rename`（manifest がコミットマーカー）。
- **load の手順**: manifest をサイズ上限付きで読み厳格パース →
  `format`／`format_version` 完全一致確認 → safetensors を上限付きで
  読む → 3 種のキー集合とファイル内容の完全一致・shape 一致を確認 →
  spec 順に層を構築（BN のみ `from_parameters`）→ `load_state_dict` →
  `set_training` → `compiled` があれば optimizer 復元・AMP scaler
  再生。途中失敗時は部分的に構築した `Sequential` を返さない。
- **非復元のもの**: BN の `num_batches_tracked`、Dropout の RNG 状態
  （グローバル RNG）、LR scheduler・callbacks、param groups。

## 6. 承認後の検証計画

`crates/facade/tests/compat_sequential_model_io.rs` で、30 種すべての
層を含むモデル・深い異種スタック・transformer encoder・train モード
後の BN running stats・6 optimizer × AMP の有無の組み合わせが save／
load 後に bit 完全一致することを検証する。manifest の改竄（未知キー・
版違い・層数超過・深さ超過・サイズ超過・非有限値・キー集合不一致・
shape 不一致）はすべて fail-closed に拒否されることを確認する。
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
  `fn` 宣言が 0 件であることを固定する。
- 承認後は、doctest・ソース走査を撤去して正ガード（実際の公開面の
  固定テスト）へ置き換え、インベントリの期待集合を
  `facade/src/compat/model_io.rs` へ差し替える。

## 8. OWASP Top 10 観点（承認後の要件として記録）

- **A03 インジェクション／非信頼入力**: manifest.json と
  model.safetensors は非信頼の外部フォーマットとして扱う。バイト数
  （`take(cap + 1)`）・JSON の深さ・層数・`num_slots`・tracker 再生
  回数のすべてを、確保・走査の前に検証する。固定スキーマに対し未知
  キー・重複キーを拒否し、キー集合と shape は完全一致させる（無言
  skip 禁止。REQ-7）。パスは `std::fs` へそのまま渡し、シェル展開・
  ユーザー入力の連結はしない。
- **A08 ソフトウェア・データ整合性**: 一時ファイル＋`rename` で書き
  込み、manifest を最後に書いてコミットマーカーとする。load は検証が
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
