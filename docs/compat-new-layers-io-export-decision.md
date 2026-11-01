# 新しい層の保存・読込（model_io）と ONNX export の横断方針の設計判断記録

## §0 位置づけ・承認根拠・境界・推奨

- 本書は Phase 6 で `Sequential` に足す新しい層の、保存・読込（`save_model`／`load_model`。`crates/facade/src/compat/model_io.rs`）と ONNX export の扱いを横断で決める設計記録である（イシュー #3750）。兄弟の設計記録（#3740・#3742・#3744・#3746・#3747・#3748）は、どれもこの 2 点を「#3750 の横断方針に委ねる」としている。
- 追跡表 `docs/compat-matrix-ari-tracker.md` の対応（イシュー本文の項目番号は追跡表の通し番号と 1 ずれている。`docs/autodiff-pool3d-ops-decision.md`・`docs/autodiff-batch-norm-3d-decision.md` §0 と同じ注記）。更新するのは次の 5 行。

| イシュー本文の項目 | 追跡表の行 | 層 |
|---|---|---|
| 8 | # 7 | `add_upsample` の `scale_factor` |
| 10 | # 9 | C1 Hardshrink・Softshrink |
| 12 | # 11 | C3 BatchNorm3d |
| 15 | # 14 | C7 ChannelShuffle |
| 16 | # 15 | C9 AdaptiveAvgPool3d・AdaptiveMaxPool3d・LPPool・FractionalMaxPool |

- 流れ: 設計 #3750 → 承認 #3768（論点 4）→ 各層の実装 issue（#3814・#3817・#3819／#3820・#3822・#3823〜#3827）。
- **承認根拠: なし。未承認（#3768 待ち）。** 本書は何も承認しない。
- 書式は `docs/decision-record-template.md`（#3716）の必須 9 節（§1〜§9）に従い、横断規則を §10〜§12、所有者へ戻す論点を §13、周辺項目を §14 に置く。
- **設計のみ**で、コード・公開面・tolerance・baseline・ガードレール閾値・依存・`docs/spec` は変えない。行番号は origin/main の `969095335` を基準にする。実装 PR の前に引き直すこと。実機は `<cuda-node>`・`<metal-node>`・`<rocm-node>` で表す。
- 境界: Bilinear（# 10。`Sequential` に載らない）と CosineSimilarity・PairwiseDistance（# 12・13。2 入力）は `Sequential` の保存の対象外で、本書では扱わない。FractionalMaxPool（#3749）は設計記録がまだ無いため、規則だけを適用し kind 名・スキーマは決めない。

### 推奨（2 本に分ける）

- **model_io = 実装する。** 全層に §10 の規則で kind を足す。理由は、#3052 以降「保存に未対応で残るのは `add_module` だけ」という状態を保つため。fail-closed のまま放置すると、利用者が保存できない層が増える。
- **ONNX export = 対象外にする（0.14.0 では `UnsupportedLayer` で fail-closed のまま）。** 理由は 4 つ。
  1. 対応表の NN 行の判定に export は要らない（#3768 論点 4 の 2 つ目の選択肢と整合）。
  2. 既存の兄弟層（size 指定の Upsample・PixelShuffle・BN1d／2d・MaxPool 系）も export していない。
  3. 新しい op（Shrink・LpPool・Resize）には export_ops・interp・parity テストの追加と数値契約の承認が要る。Sigmoid は承認なしに腕を足して codex P1 でブロックされ除去した先例がある（`docs/facade-onnx-export-exposure-decision.md` §18.2。追跡表 # 26 で別途承認待ち）。
  4. `Module` trait への `as_*` フック追加は既定実装つきなら追加のみだが、公開面が広がる。

### 調査で確定した事実

- `ModelIoError` は `#[non_exhaustive]`（`model_io.rs:170`）。variant は `Io`・`Manifest`・`Safetensors`・`UnsupportedModel`・`Mismatch`・`Autodiff`・`TooLarge`。
- `FORMAT_VERSION = 2`（`:122`）・`LEGACY_FORMAT_VERSION = 1`（`:125`）。load はこの 2 つ以外を `Manifest` で拒否する（`:4205`）。
- kind は文字列の allowlist で、`spec_kind`（`:327`）と `spec_from_kind`（`:3342`）が受け持つ。未知 kind は `UnsupportedModel`。件数は単体テスト `all_thirty_kinds_are_covered_and_round_trip_through_params_schema`（アサートは `70`）で固定している。
- v1 manifest で拒否しているのは再パラメータ化層（buffer を持つ層）だけ（`:4155`、`Mismatch`）。無 buffer の kind は v1 でも受理される。
- f32 は最短往復表記 `{:?}` の正準形で持つ（`as_f32`・`put_real`）。**f64 の読み書き（`as_f64`）は存在しない。** 上限は `MAX_NUMBER_LEXEME_LEN = 64`・`MAX_OBJECT_KEYS = 16`・`MAX_JSON_DEPTH = 4`。`save_functional_model`（`functional_io.rs`）も `spec_kind` を使う。
- 先例（`docs/compat-model-io-decision.md`）: kind を追加しても `format_version` は 2 のまま（#2373・#2509・#2529・#2530・#2532・#2533・#3052 §4）。旧リーダーは未知 kind を `UnsupportedModel` で拒否する。rrelu・dropout の乱数状態は保存しない。意味の範囲検査は load 時の `add_*` に任せ（`Autodiff`）、構造の検査は manifest パーサが行う（`Manifest`）。
- ONNX: `ExportError` は `#[non_exhaustive]`（`crates/onnx-interop/src/onnx/export.rs`）、`OnnxError` も `#[non_exhaustive]`（`crates/facade/src/interop/onnx.rs`）。`export_parts_from_layers`（`export_nn.rs:230`）が対応する層は Linear・ReLU・Softmax・LayerNorm・GELU・Conv2d の 6 種で、それ以外は `UnsupportedLayer`。`layer_kind_name`（`:197`）が具体名を返すのは一部だけで、残りは `"unknown"`。`SUPPORTED_OP_TYPES`（`export_ops.rs:193`）に `Shrink`・`LpPool`・`Resize` は無い。`BatchNormalization` はあるが BN1d／2d 層の export の腕は無い。

## §1 公開形とガード反転計画

- 本書が新しく公開する名前は無い。`ModelIoError`・`OnnxError`・`ExportError` に variant を足さない。
- 反転するガードは無い。次は不変: `onnx_export_from_sequential_is_reachable_via_facade`・`interop_module_exposes_only_approved_onnx_surface`・`approved_onnx_surface_yields_no_offenses`（いずれも `crates/facade/tests/api_surface.rs`）。`ALLOWED_PUB_ITEMS` も不変。
- 層ごとの公開名の正ガードは、各兄弟の設計記録の担当として参照するだけにする。

## §2 rustdoc と doctest

- 実装 PR は `model_io.rs` のモジュール doc にある種別数（「70 種」）と、各 `add_*` の rustdoc の 1 行（「`save_model` で保存・復元できる（kind `x`）。ONNX export は `UnsupportedLayer`」）を更新する。
- 失敗系は `compile_fail` を使わず、`is_err` の通常テストで確かめる（stable rustdoc は `compile_fail` のコードを照合しない）。

## §3 エラー型（使い分けと検証順）

- 保存側: 非有限の f32／f64、上限超え、乱数で決まる値を記録できない構成は `UnsupportedModel`。`dir` に触れる前に判定する。
- load 側の検証順: kind の allowlist（無ければ `UnsupportedModel`）→ キー集合の完全一致（`Manifest`）→ 型・正準表記（`Manifest`）→ 構造上の範囲（`size_len` など。`Manifest`）→ 意味の範囲（`lambd < 0`・`groups = 0`・`norm_type <= 0` など）は `add_*` を経由して `Autodiff` → 期待キーと safetensors の不一致（`Mismatch`）→ 層の構築。テンソルの確保は、上限付きで読んだ実 shape が期待キーと一致した後だけ。
- v1 manifest に buffer を持つ新しい kind（`batch_norm3d`）が現れたら `Mismatch` で拒否する（reparam の先例 `:4155` と同じ variant）。
- ONNX: `ExportError::UnsupportedLayer` → `OnnxError::UnsupportedLayer`。本番経路で `unwrap`／`expect` を使わない。

## §4 テスト計画

model_io と export はホストで実行し、数値経路に依存しない。実機列は「該当なし（ホスト実行のみ）」とする。Phase 10 の実機 parity（#3860・#3861）は層の forward／VJP の担当で、本書の範囲外。

| 受け入れ基準 | テスト（実装 PR が追加） | 区分 | `<cuda-node>` | `<metal-node>` | `<rocm-node>` |
|---|---|---|---|---|---|
| 層ごとの往復 bit 一致 | `crates/facade/tests/compat_sequential_model_io_layers.rs` に層ごとに追加 | CI | 該当なし | 該当なし | 該当なし |
| BN3d の running stats の bit 一致と `training` の復元 | `compat_sequential_model_io_batch_norm.rs` に追加 | CI | 該当なし | 該当なし | 該当なし |
| 非信頼 manifest（キー欠落・未知キー・型違い・非正準表記・範囲外。BN3d は v1 での出現も） | `model_io.rs` の表駆動テストに層ごとに追加 | CI | 該当なし | 該当なし | 該当なし |
| kind 数の更新 | `all_thirty_kinds_are_covered_and_round_trip_through_params_schema` のアサート | CI | 該当なし | 該当なし | 該当なし |
| functional 側の対応 | `compat_functional_model_io.rs` に 1 件 | CI | 該当なし | 該当なし | 該当なし |
| export の fail-closed | `interop_onnx_export_sequential.rs` に `sequential_with_<layer>_is_rejected_with_unsupported_layer`（既存 Sigmoid と同形） | CI | 該当なし | 該当なし | 該当なし |

## §5 数値契約

- 保存の往復は bit 一致（f32・f64 とも正準表記）。REQ-2 の統一複合判定（相対誤差 1e-3 未満 または 絶対誤差 1e-5 未満）・tolerance 定数・parity baseline・ガードレール閾値は変えない。
- FMA 契約・f64 長軸縮約の契約への影響: なし。baseline の追加・更新: なし。

## §6 互換への影響

- 判定: **追加のみ**（`docs/api-compat-0.14.0-decision.md` §4・§8）。
- API（型ごと）: `ModelIoError`・`OnnxError`・`ExportError` は variant 追加なしで該当なし。`add_*` の追加は兄弟の記録に従い追加のみ。
- ファイル形式: `format_version` は 2 のまま。新しいリーダーは v1・v2 の旧ファイルをすべて読める。0.13.0 のリーダーは新しい kind を含むファイルを `UnsupportedModel` で拒否する（fail-closed。前方互換は保証しない）。先例どおりで、§8 の判定では追加のみ。移行ノート `docs/migration/0.14.0.md`（#3780）の「挙動のみの変更」欄に 1 行書く案を §13 の論点 5 で問う。

## §7 適用記録・索引・追跡表

- 適用記録の文案（書くのは実装 PR）: `**適用記録（イシュー #N・親 #P。承認: <#3768 回答 URL>）**: kind <x> を追加（format_version 2 据え置き）。ONNX export は UnsupportedLayer のまま。公開状態: 公開済み`
- [ ] `docs/compat-api-scope.md` に適用記録を追記した（実装 PR の担当。本 PR では未実施）
- [x] `docs/README.md` の索引を更新した
- [x] 追跡表の # 7・9・11・14・15 の「設計記録」列へ本書を足した（イシューが 5 行を指定する横断 issue のため、追跡表 §4.6 の「1 PR は自分の行だけ」の例外にあたる。「承認」「0.14.0 の区分」は変えない）

## §8 ベンチと性能への影響

- 保存と export は学習・推論の既定経路に入らないため影響なし。前後比較は不要。

## §9 依存の追加の有無

- 依存追加: なし。許容 12 区分・禁止リストと照合済み。新規 `unsafe`: なし。

## §10 model_io の横断規則（本書の中核）

- **R1 命名**: kind は `snake_case` で、`add_*` から `add_` を除いた名前にする。`LayerSpec::Unsupported { kind }` に現れる文字列と一致させる。既存 kind のスキーマは変えず、意味を広げるときは別の kind 名にする（例: `upsample` を拡張せず `upsample_scale_factor` を新設する）。
- **R2 `params`**: `add_*` の引数から `seed` を除いたもの。`add_*` が内部で固定する値は書かない。配列は `_d/_h/_w` か `size_len`／`size_i` に平坦化し、16 キー以内・深さ 4 以内に収める。
- **R3 版数**: `format_version` は 2 のまま。上げるのは、既存 kind のスキーマを変えるときか、同じ kind 名の読み方を変えなければならないときだけ（本ツリーには該当なし）。
- **R4 後方互換**: 新しいリーダーは旧ファイルを読める。未知 kind は `UnsupportedModel` で拒否し、黙って読み飛ばさない（REQ-7）。
- **R5 v1 の扱い**: buffer を持つ新しい kind は、v1 に現れたら `Mismatch` で拒否する（BN3d が対象。v1 の legacy 経路が初期 running stats で黙って補うのを防ぐ）。無 buffer の kind は既存の先例どおり v1 でも受理する。
- **R6 f64 の表現（新しい規則）**: 案 (i) `as_f64`／`put_real_f64` を足す。`as_f32` と同じく有限かつ `{:?}` 往復の正準形に限り、字句は 64 バイト以内。案 (ii) f64 を持つ層は当面 `UnsupportedModel` にする。推奨は (i)。効くのは upsample の倍率と、FractionalMaxPool の `output_ratio`（#3749 が採る場合）の 2 つ。
- **R7 乱数状態は保存しない**（rrelu・dropout の先例）。FractionalMaxPool の窓の乱数列にも適用し、eval で保存したときに決定的に往復できる範囲を #3749 の契約に合わせる。
- **R8 着地まで**: kind が入るまでは `LayerSpec` に variant を置き、保存は `UnsupportedModel` で fail-closed にする。
- **R9 件数**: kind 数のアサートは、各実装 PR が自分で足した分だけ加算する。到達数は適用記録に書く。並列 PR が衝突したら rebase して合算する。
- **R10 適用範囲**: `save_functional_model` にも同じ規則が及ぶ。

## §11 ONNX export の対応表

`layer_kind_name` は `"unknown"` のまま。`as_*` フックの追加は別承認の改善候補とする。いずれも現状は export の腕が無く `UnsupportedLayer`。

| 層 | ONNX の単体 op | 0.14.0 の扱い | 再開する場合の要件 |
|---|---|---|---|
| upsample（倍率） | `Resize`（scales） | `UnsupportedLayer` | size 指定も未対応のため、まとめて扱う |
| Hardshrink／Softshrink | `Shrink`（bias・lambd） | `UnsupportedLayer` | export_ops・interp の追加と数値契約の承認 |
| BatchNorm3d | `BatchNormalization`（allowlist にはある） | `UnsupportedLayer` | BN1d／2d も未対応のため、まとめて扱う |
| ChannelShuffle | 単体 op は無い。`Reshape→Transpose→Reshape` に分解できる | `UnsupportedLayer` | 分解 export の設計 |
| AdaptiveAvg／MaxPool3d | 無い（Global* のみ） | `UnsupportedLayer` | なし（op が無い） |
| LPPool | `LpPool`（`p` は整数） | `UnsupportedLayer` | `norm_type` が整数のときに限る |
| FractionalMaxPool | 無い | `UnsupportedLayer` | なし（op が無い） |

## §12 層ごとの適用表（kind の案。最終の確定は各実装 PR の適用記録）

| 層 | kind | `params` | 期待キー（buffer） |
|---|---|---|---|
| upsample（倍率） | `upsample_scale_factor` | `mode`・`align_corners`・`scale_len`（1..=3）・`scale_0..2`（f64。R6 (i)。(ii) なら当面 `UnsupportedModel`） | 0 |
| Hardshrink／Softshrink | `hardshrink`／`softshrink` | `lambd`（f32） | 0 |
| BatchNorm3d | `batch_norm3d` | `num_features`・`eps`・`momentum` | 1d／2d と同じ `running_mean`／`running_var`。R5 で v1 なら拒否 |
| ChannelShuffle | `channel_shuffle` | `groups` | 0 |
| AdaptiveAvg／MaxPool3d | `adaptive_avg_pool3d`／`adaptive_max_pool3d` | `output_size_{d,h,w}` | 0 |
| LPPool | `lp_pool1d`／`lp_pool2d`／`lp_pool3d` | `norm_type`（f32）・`kernel_size*`・`stride*`（`null` 対可） | 0 |
| FractionalMaxPool | （#3749 確定後に記入） | （同） | （同） |

- 増える数の見込みは +12（FractionalMaxPool を除く）。目安にとどめ、到達数は R9 で確定する。

## §13 所有者へ戻す論点（#3768 の論点 4 へ渡す）

| 点 | 選択肢 | 推奨 | 確認がないときの扱い |
|---|---|---|---|
| 1. model_io の範囲 | §10 の規則で全層に実装する／当面 fail-closed にする | 実装する | 確認があるまで実装へ進めない（保留のまま保護ガードを維持） |
| 2. f64 の表現 | (i) 正準 `{:?}` の f64 を足す／(ii) f64 を持つ層は当面 fail-closed | (i) | 確認があるまで実装へ進めない（保留のまま保護ガードを維持） |
| 3. v1 拒否の範囲 | buffer を持つ kind だけ／v2 以降に足した全 kind | buffer を持つ kind だけ | 確認があるまで実装へ進めない（保留のまま保護ガードを維持） |
| 4. ONNX export | 0.14.0 では含めない／Shrink・LpPool などを含める | 含めない | 確認があるまで実装へ進めない（保留のまま保護ガードを維持） |
| 5. 旧リーダーとの非互換（0.13.0 は新しい kind を拒否する）の移行ノート記載 | 書く／書かない | 書く | 確認があるまで実装へ進めない（保留のまま保護ガードを維持） |

## §14 セキュリティ観点・スコープ外

- A03: 非信頼の manifest は固定スキーマ・正準表記・上限の順で検証し、確保より前に済ませる。f64 の字句も 64 バイト以内にする。意味の検査は `add_*` に寄せて二重に実装しない。
- A04／A08: 未知 kind を黙って読み飛ばさない。v1 の legacy 経路で buffer が黙って化けるのを防ぐ（R5）。
- A06: 依存の追加なし。秘密情報・実機のホスト名を書かない。
- スコープ外（新しい issue は起票せず、所有者の承認事項として列挙するだけ）: ONNX export の新しい op（Shrink・LpPool・Resize・ChannelShuffle の分解）、BN1d／2d・Upsample(size) の export、`as_*` フックの追加、FractionalMaxPool の具体スキーマ（#3749）。
- `Closes`／`Refs`: 完了条件が設計記録の作成で実測を要しないため `Closes #3750` とする。
