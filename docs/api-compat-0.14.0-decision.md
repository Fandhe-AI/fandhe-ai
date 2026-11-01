# 0.14.0 公開 API 互換方針の決定記録

イシュー #3713（親 #3712・ルート #3711）。0.14.0 で対応表「部分的」9 行・29 項目を「ある」にするとき、公開型のどの変更が「追加のみ（非破壊）」でどれが「破壊的」かを判定するための入力表を作る。

- 基準面: `origin/main` の `496d23b46`（2026-10-31 確認）。コード・依存・tolerance・baseline・spec は変更しない
- 本 PR が書くのは §1（事実と判定）だけである。互換方針の決定・所有者承認は #3714 が §2 以降へ書く
- 判断は書かない。設計前のため、判断が割れる箇所は両方の場合の判定を併記する

## 1. 公開型の `#[non_exhaustive]` と構築可能性の棚卸し

### 1.0 目的・範囲・凡例

- 「公開型」は facade（`fandhe-ai`）が `pub` に出す型・trait・関数を指す。`facade` が唯一のサポートされる公開 API 面で、`tensor-core`・`autodiff`・`backend-*`・`onnx-interop` は内部クレートである（`docs/compat-api-scope.md` §0）
- 表の主行は facade から到達する型にする。facade が再エクスポートしない型（例: `onnx-interop` の `ExportError`、クレートルートの `LoadError`）は「内部。facade の安定性保証の外」と注記する
- 同名の型が autodiff と facade の双方にある場合（`Sequential`・`ExponentialMovingAverage`・`AveragedModel`）は facade 側を主行にする
- 行番号は基準面の実ファイルを `sed -n` で目視確認した値である。定義行の直前の attribute だけを根拠にした

### 1.1 判定規則

4 軸（`#[non_exhaustive]` の有無・外から構築できるか・derive・正ガード）から次のように判定する。cargo-semver-checks は導入しない。分類名は参照のみで、綴りの確認はこの PR では行っていない（**分類名は要確認**。#3714 が semver-checks のリファレンスで確認する）。

| 変更 | 条件 | 判定 | 参照する分類名 |
|---|---|---|---|
| enum に variant を足す | `#[non_exhaustive]` なし | 破壊的 | `enum_variant_added` |
| enum に variant を足す | `#[non_exhaustive]` あり | 追加のみ（該当 lint は発火しない） | - |
| struct にフィールドを足す | フィールドが外から構築できない（非公開フィールド、または `#[non_exhaustive]`） | 追加のみ | - |
| struct にフィールドを足す | 全フィールド `pub` かつ `#[non_exhaustive]` なし | 破壊的 | `constructible_struct_adds_field` |
| derive の追加・削除 | derive を外す（例: `Eq` の喪失） | 破壊的 | `derive_trait_impl_removed` |
| inherent メソッド・自由関数の追加 | 既存シグネチャ不変 | 追加のみ | -（削除は `inherent_method_missing`） |
| 既存関数のシグネチャ変更（引数追加・ジェネリック化） | - | 破壊的 | `function_parameter_count_changed` 等 |
| trait にメソッドを足す | 既定実装なし・sealed でない | 破壊的 | `trait_method_added` |
| trait にメソッドを足す | 既定実装あり | 追加のみ（この lint の対象外の見込み。要確認） | - |

### 1.2 29 項目と変わりうる公開型の対応

件数は 0.12.0 確定件数表（`docs/perf/framework-compare-feature-matrix-0.12.0.md:380-392`）の未公開＋保留 30 件から、f64 `jvp` 系を dtype 行と自動微分行で二重に数えている 1 件を引いた 29 件。内訳は dtype 2・自動微分 4・演算 1・NN 9・最適化 8・バックエンド 1・相互運用 3・事前学習 1・推論 1 の 30 で、項目 2 が dtype・自動微分の双方に効くため重複除去で 29。項目名と区分（A1〜A3-g・B1〜B5・C1〜C9）は `docs/compat-unlisted-holds-inventory.md` §4.1〜§4.3 に合わせた。

「型の変化」の判定は 1.1 の規則による。型を特定できなかった項目は推測で埋めず「未確認」と書く。

| N | 行 | 項目 | 変わりうる公開型・関数（候補） | 型への影響の見立て |
|---|---|---|---|---|
| 1 | dtype | 低精度演算の `Var` 以外への配置 | `Var`・`Tape`・`Tensor` の演算追加 | 設計で判断。関数追加なら追加のみ |
| 2 | dtype・自動微分 | `TapeF64`／`VarF64` の `jvp` 系 | `TapeF64`・`VarF64` の inherent メソッド追加 | 追加のみ（内部フィールドは `pub(crate)`。`crates/facade/src/lib.rs:2075`・`:2106`） |
| 3 | 自動微分 | `create_graph` の残る Op | 公開型の変化なし見込み。`AutodiffError` の variant 追加は `#[non_exhaustive]`（`crates/autodiff/src/error.rs:19`）で非破壊 | 挙動のみ |
| 4 | 自動微分 | 二階微分の拒否範囲（LayerNorm 等） | 同上 | 挙動のみ（拒否が受理に変わる） |
| 5 | 自動微分 | Embedding `padding_idx` の二階の意味論 | 同上 | 挙動のみ |
| 6 | 演算 | GPU 線形代数の opt-in 関数（#3405） | facade の自由関数・設定型の新設 | 追加のみ |
| 7 | NN | `add_upsample` の `scale_factor` 指定 | `Sequential::add_upsample`（`crates/facade/src/compat/sequential.rs:2435`）・`UpsampleSize` | 別名メソッドなら追加のみ。既存シグネチャ変更なら破壊的。`UpsampleSize` は `#[non_exhaustive]`（`crates/autodiff/src/nn/upsample.rs:21`） |
| 8 | NN | 独自 `Module` から組み込み層を呼ぶ経路 | `nn::Module` trait（open trait。`crates/facade/src/nn/module.rs:11` のコメントが「sealed にしない」と明記） | 既定実装なしのメソッド追加は破壊的。#3714 で判断 |
| 9 | NN | C1 Hardshrink／Softshrink | `Var` 演算・`Sequential::add_*` | 追加のみ |
| 10 | NN | C2 Bilinear | 層型・`Var` 演算・`add_*`（層型名の公開は #3022 の個別承認に従う） | 追加のみ |
| 11 | NN | C3 BatchNorm3d | `Var::batch_norm` の rank 5 対応・`add_batch_norm3d` | 挙動拡張＋追加のみ |
| 12 | NN | C5 CosineSimilarity | `Var` 演算 | 追加のみ |
| 13 | NN | C6 PairwiseDistance | `Var` 演算 | 追加のみ |
| 14 | NN | C7 ChannelShuffle | `Var` 演算・`add_*` | 追加のみ |
| 15 | NN | C9 Adaptive3d・LPPool・FractionalMaxPool | `Var` 演算・`add_*` | 追加のみ（乱数契約は型でなく契約の問題） |
| 16 | 最適化 | SWA の `FitConfig` 形 | `FitConfig`・`SwaCallback`・`AveragedModel` | `f32` フィールドを `FitConfig` へ足すと `Eq` を失い破壊的。bits で保持すれば不変（§1.4） |
| 17 | 最適化 | A1 EMA と AMP の併用 | `Callback::Ema`・`AmpConfig`・`EmaCallback` | 挙動のみ見込み |
| 18 | 最適化 | A2 EMA decay ウォームアップ | `EmaCallback` への設定追加 | `EmaCallback` はフィールド非公開で追加のみ。`FitConfig` に足すと `Eq` 喪失の恐れ |
| 19 | 最適化 | A3-a param groups と AMP | `ParamGroup`・`ParamGroupStep`・`AmpConfig` | `ParamGroup` は追加のみ。`ParamGroupStep` へのメソッド追加は #3714 で判断 |
| 20 | 最適化 | A3-b param groups と `LrSchedule` | `LrSchedule`・`ParamGroup`・`Callback::LrSchedule` | `LrSchedule` はフィールド非公開で追加のみ |
| 21 | 最適化 | A3-c param groups の保存形式 | `state_dict` の書式・`ModelIoError` | 型より保存形式の互換が論点。`ModelIoError` は `#[non_exhaustive]` |
| 22 | 最適化 | A3-d param groups と GPU 常駐更新 | `DeviceParamStore`（facade `lib.rs:273` で再エクスポート） | フィールド非公開（`crates/autodiff/src/optim/device_store.rs:420`）で追加のみ |
| 23 | 最適化 | A3-g `Optimizer` に 4 種を足す | `Optimizer`（`#[non_exhaustive]`） | 追加のみ（§1.4 の訂正） |
| 24 | バックエンド | ROCm の facade 公開 | `Device`・`available_devices`・`tape_for`・`select_from` | `Device::Rocm` の追加は破壊的（§1.5） |
| 25 | 相互運用 | B1 ONNX autograd の facade 公開 | `OnnxModel`・`OnnxValue`・`OnnxError` | 新規メソッドなら追加のみ。`OnnxValue` への variant 追加は破壊的 |
| 26 | 相互運用 | B3 ONNX export の `Sigmoid` | `OnnxExportOptions`・`OnnxError`（いずれも `#[non_exhaustive]`） | 挙動差のみ |
| 27 | 相互運用 | B5 safetensors の f32 以外 | `SaveError`・`LoadError`・`save_safetensors_f32`／`load_safetensors_f32` | dtype 別の入口を足すなら追加のみ。ジェネリック化は破壊的 |
| 28 | 事前学習・推論 | `hf-hub` の 2 段目（crates.io 公開） | `HfHub`（`crates/hf-hub/src/lib.rs:337`） | 新クレートの公開。facade の既存面は不変見込み |
| 29 | 推論 | `caches` で表せない内部状態のモデルの speculative decoding・連続バッチング | `SpeculativeConfig`・`GenerateConfig`・`BatchScheduler`・`SchedulerLimits`・`RequestId` | 新設 trait は追加のみ。既存 trait へのメソッド追加は破壊的 |

補足（29 に数えない横断項目）: 新しい層の `model_io` 保存と ONNX export は、件数表の NN 行 9 件に含まれない。`ModelIoError`・`OnnxError` は `#[non_exhaustive]` で variant 追加は非破壊。model_io の kind 文字列は型でなく保存形式の互換の論点。

NN 行の 9 件は N7〜N15（add_upsample・Module 経路・C1・C2・C3・C5・C6・C7・C9）、最適化行の 8 件は N16〜N23 である。

### 1.3 型別の棚卸し表

行番号は `path:line`。`#[non_exhaustive]` 列は attribute の行、`構築` 列は外のクレートから struct リテラルや variant で構築・網羅 `match` できるかを示す。「正ガード」は `crates/facade/tests/api_surface.rs` のテスト名で、grep で確認した範囲に限る（無いものは「確認できず」）。

| 型（定義） | 種別 | `#[non_exhaustive]` | 構築・網羅可能性 | derive | 正ガード | 触れる N | 判定 |
|---|---|---|---|---|---|---|---|
| `Device`（`crates/tensor-core/src/device.rs:47`。facade `lib.rs:322` で再エクスポート） | enum | なし（derive は `:46`） | 網羅 `match` 可能。`Metal` は `cfg(target_os = "macos")`（`:56-57`） | `Debug, Clone, Copy, PartialEq, Eq, Hash` | `device_variant_set_is_fixed_while_3157_on_hold`（`api_surface.rs:43039`） | 24 | variant 追加は破壊的（`enum_variant_added`）。`Rocm(usize)` なら `Copy`・`Eq`・`Hash` は保たれる |
| `BackendError`（`device.rs:202`） | enum | あり（`:200`） | 網羅不可 | `Debug` | 確認できず | 24 | 追加のみ |
| `Optimizer`（`crates/facade/src/compat/training.rs:240`） | enum | あり（`:238`） | 網羅不可 | `Debug, Clone, Copy, PartialEq`（`Eq` なし。`:239`） | 確認できず | 23 | 追加のみ。新 config 型を `Copy, PartialEq` で足せば derive も保たれる |
| `Loss`（`training.rs:194`） | enum | あり（`:192`） | 網羅不可 | `Debug, Clone, Copy, PartialEq, Eq` | 確認できず | なし | 本棚卸しの対象変更なし（参考） |
| `FitConfig`（`training.rs:295`） | struct | なし | フィールドは全て `pub(super)`（`:296-`）で外から構築できない。コンストラクタ経由 | `Debug, Clone, Copy, PartialEq, Eq`（`:294`） | `fit_config_keeps_copy_eq_for_0_9_0_compat`（`api_surface.rs:21921`） | 16・18 | フィールド追加そのものは非破壊。`f32` 等 `Eq` を持たない型を足すと `Eq` 喪失で破壊的（`derive_trait_impl_removed`） |
| `AmpConfig`（`training.rs:119`） | struct | なし | フィールド（`compute_dtype`・`grad_scaler`）は非公開（`:120-121`）で外から構築できない | `Debug, Clone, Copy, PartialEq`（`:118`） | 確認できず | 17・19 | フィールド追加は追加のみ |
| `Callback`（`crates/facade/src/compat/callbacks.rs:1660`） | enum | あり（`:1658`） | 網羅不可 | `Debug` | 確認できず | 17・20 | 追加のみ |
| `EmaCallback`（`callbacks.rs:1250`） | struct | なし | 構築はコンストラクタ経由（フィールドの非公開は §1.5 の未確認に含めて実コードで再確認する） | `Debug` | 確認できず | 17・18 | 追加のみ（フィールド非公開の場合） |
| `SwaCallback`（`callbacks.rs:1403`） | struct | なし | 同上 | `Debug` | 確認できず | 16 | 同上 |
| `LrSchedule`（`callbacks.rs:624`） | struct | なし | フィールド `kind`・`next_epoch` は非公開（`:625-626`） | なし | 確認できず | 20 | 追加のみ |
| `ParamGroup`（`crates/autodiff/src/nn/optim/param_group.rs:96`。facade `optim.rs:546` で再エクスポート） | struct | あり（`:94`） | `pub` フィールド（`params`・`lr`・`weight_decay`）を持つが、外からは `ParamGroup::new` でのみ構築できる | `Debug, Clone, PartialEq`（`:95`） | 確認できず（`api_surface.rs:17891` の `check_step_sig` は trait 側） | 19・20 | フィールド追加は追加のみ |
| `ParamGroupStep`（`param_group.rs:209`。facade `optim.rs:546`） | trait | - | sealed かは未確認 | - | `api_surface.rs:17891` 付近 | 19 | メソッド追加は `trait_method_added`（既定実装なし・非 sealed の場合。要確認） |
| `Var`（`crates/autodiff/src/var.rs:110`。facade `lib.rs:281`） | struct | なし | フィールド `tape`・`id` は非公開 | `Debug, Clone, Copy`（`:109`） | 確認できず | 1・9〜15 | inherent メソッド追加は追加のみ |
| `Tape`（facade `lib.rs:452`） | struct（newtype） | なし | フィールドは `pub(crate)` | なし | 確認できず | 1・2 | 追加のみ |
| `TapeRef`（`lib.rs:2001`） | struct | なし | フィールドは `pub(crate)` | `Clone, Copy`（`:2000`） | 確認できず | 8 | 追加のみ |
| `TapeF64`（`lib.rs:2075`）・`VarF64`（`:2106`） | struct（newtype） | なし | フィールドは `pub(crate)` | `VarF64` は `Clone, Copy`（`:2105`） | 確認できず | 2 | メソッド追加は追加のみ |
| `Module`（`crates/facade/src/nn/module.rs:45`） | trait | - | open（sealed にしない。`:11` のコメント） | - | `facade_nn_module_trait_methods_match_approved_set`（`api_surface.rs:7710`） | 8 | 既定実装なしのメソッド追加は破壊的（`trait_method_added`。要確認） |
| `Sequential`（facade `compat/sequential.rs:190`） | struct | なし | フィールドは非公開（`inner` ほか。`compat/sequential.rs:191` 付近） | 未確認 | 確認できず | 7・9〜15 | 追加のみ。autodiff 側の同名型（`crates/autodiff/src/compat/sequential.rs:59`・`nn/container.rs:480`）は内部・非保証 |
| `UpsampleSize`（`autodiff/src/nn/upsample.rs:23`） | enum | あり（`:21`） | 網羅不可 | `Debug, Clone, PartialEq` | 確認できず | 7 | 追加のみ |
| `ModelIoError`（`crates/facade/src/compat/model_io.rs:170`） | enum | あり（`:169`） | 網羅不可 | `Debug` | 確認できず | 21・補足 | 追加のみ |
| `OnnxModel`（`crates/facade/src/interop/onnx.rs:228`） | struct | なし | フィールド `graph` は非公開（`:229`） | `Debug` | 確認できず | 25 | メソッド追加は追加のみ |
| `OnnxValue`（`onnx.rs:515`） | enum | なし | 網羅 `match` 可能（`F32`・`I64`・`Bool`・`F16`） | `Clone, Debug` | 確認できず | 25 | variant 追加は破壊的（`enum_variant_added`） |
| `OnnxError`（`onnx.rs:558`） | enum | あり（`:556`） | 網羅不可 | `Debug` | 確認できず | 25・26 | 追加のみ |
| `OnnxExportOptions`（`onnx.rs:432`）・`OnnxExternalDataLimits`（`:485`） | struct | あり（`:430`・`:483`） | 外からはコンストラクタ経由 | 前者 `Clone, Debug, PartialEq, Eq`、後者 `Clone, Copy, Debug, PartialEq, Eq` | 確認できず | 26 | フィールド追加は追加のみ |
| `ExportError`（`crates/onnx-interop/src/onnx/export.rs:83`） | enum | あり（`:82`） | 網羅不可 | `Debug, PartialEq, Eq` | - | 26 | 内部。facade は `onnx.rs:150` で private に取り込み、公開は `OnnxError`（`ExportError::EmptyModel` 等を写像） |
| `SaveError`（`onnx-interop/src/st_save.rs:73`）・`LoadError`（`st_load.rs:48`） | enum | あり（`:71`・`:46`） | 網羅不可 | `Debug` | `interop_safetensors_reexports_exactly_expected_surface`（`safetensors.rs` のコメントが参照） | 27 | 追加のみ。facade の再エクスポートは `interop/safetensors.rs:95-98` |
| `LoadError`（`onnx-interop/src/require_keys.rs:47`） | enum | あり（`:46`） | - | `Debug, Clone, PartialEq, Eq` | - | なし | 内部。上の `st_load::LoadError` とは別型で facade は再エクスポートしない（`safetensors.rs:17-25` のコメント） |
| `InterpolateMode`（`crates/tensor-core/src/backend_ops.rs:744`）・`PadMode`（`pad_modes.rs:44`） | enum | あり（`:742`・`:43`） | 網羅不可 | 前者 `Debug, Clone, Copy, PartialEq, Eq, Hash`、後者 `Debug, Clone, Copy, PartialEq, Eq` | 確認できず | 参考（N7 の補間・pad 周辺） | 追加のみ |
| `GenerateConfig`（`autodiff/src/generate/mod.rs:117`）・`SpeculativeConfig`（`generate/speculative.rs:54`） | struct | あり（`:116`・`:53`） | 外からはコンストラクタ経由 | `Debug, Clone, PartialEq` | 確認できず | 29 | フィールド追加は追加のみ |
| `BatchScheduler`・`SchedulerLimits`・`RequestId`（`generate/scheduler.rs`。facade `inference/mod.rs:373`） | 未確認 | 未確認 | 未確認 | 未確認 | 確認できず | 29 | 未確認。実装時に #3714 側で確認する |
| `ModelError`（`crates/facade/src/model.rs:205`）・`DownloadOptions`（`:414`） | enum／struct | あり（`:203`・`:412`） | `DownloadOptions` はコンストラクタ経由 | `ModelError` は `Debug`、`DownloadOptions` は `Default` | 確認できず | 28 | 追加のみ（参考。`ModelRegistry` は `:684`） |
| `DeviceParamStore`（`autodiff/src/optim/device_store.rs:419`。facade `lib.rs:273`） | struct | なし | フィールドは非公開（`device` ほか。`device_store.rs:420`） | `Debug` | 確認できず | 22 | 追加のみ |
| `HfHub`（`crates/hf-hub/src/lib.rs:337`） | struct | なし | 未確認 | 未確認 | 確認できず | 28 | 新クレートの公開（`publish = false` から変更）。既存面は不変見込み |

### 1.4 訂正事項

- **`Optimizer` は既に `#[non_exhaustive]` である**（`crates/facade/src/compat/training.rs:238-240`）。依頼文の「破壊的」は誤りで、Adadelta・Adamax・NAdam・RAdam の variant 追加は非破壊と判定する。`Eq` は元から無く（`PartialEq` のみ。`:239`）、新しい config 型を `Copy, PartialEq` で足せば derive も保たれる
- **`FitConfig` が壊れる条件は `Eq` の喪失である**。フィールドはすべて `pub(super)` で外から構築できない（`training.rs:296` 以降）ため、フィールド追加そのものは非破壊。`f32` を足すと `Eq` が外れて破壊的になる。先例として `validation_split_bits: Option<u32>` が `f32::to_bits` で保持している（`training.rs:316` 付近の doc コメント。`docs/compat-fit-sample-weighting-decision.md` §11.1）。正ガードは `fit_config_keeps_copy_eq_for_0_9_0_compat`（`crates/facade/tests/api_surface.rs:21921`）

### 1.5 #3714 への申し送り

破壊的と判定した変更の候補（結論は書かない。判断は #3714 と Phase 3 の所有者承認）。

| 変更 | 関連 N | 論点 |
|---|---|---|
| `Device::Rocm(usize)` の追加 | 24 | `Device` は `#[non_exhaustive]` なし。保留ガード G1（`device_variant_set_is_fixed_while_3157_on_hold`。`api_surface.rs:43039`）が `docs/backend-rocm-device-exposure-decision.md` の所有者判断（Q1）まで variant 追加を塞いでいる。`non_exhaustive` を後付けする案があるか、そのタイミング自体が破壊的になる点を含めて判断が要る |
| `OnnxValue` への variant 追加 | 25 | 追加するかが未定。追加するなら破壊的。後付けの `#[non_exhaustive]` も破壊的 |
| `Module`（open trait）・`ParamGroupStep` へのメソッド追加 | 8・19 | 既定実装を付ければ追加のみの見込み（要確認）。付けなければ破壊的 |
| `FitConfig` への `Eq` を持たない型の追加 | 16・18 | bits 保持か、別の設定型へ分けるか |
| 既存関数のジェネリック化（safetensors 等） | 27 | 別名の入口を足すなら追加のみ |

未確認として残した項目（`EmaCallback`・`SwaCallback` のフィールド可視性、`BatchScheduler`・`SchedulerLimits`・`RequestId`、`ParamGroupStep` の sealed 有無、各型の正ガードの有無）は、#3714 の着手時に実コードで確認する。

### 1.6 変更していないもの

`crates/`・`docs/spec/`・tolerance・baseline・依存・CI は変更していない。cargo-semver-checks は導入していない。

## 2. 位置づけ・承認根拠・境界

§2 以降は #3714 が追記する（§1 は #3713 が書いた事実と判定規則で、本 PR では書き換えない）。

- **承認: 未承認（#3718 待ち）**。所有者の承認は取得していない。承認の URL は Phase 4 以降の最初の PR が転記する
- 本記録が決めるのは方針だけで、コード・公開面・依存・tolerance・baseline・spec・ガードレール閾値は変えない
- 保留ガード G1（`device_variant_set_is_fixed_while_3157_on_hold`）・G2 は、#3718 の回答と spec 改訂提案（#3715）の受理が揃うまで維持する（`docs/decision-record-template.md` の fail-closed 既定）

### 2.1 §1.5 の未確認事項の解決（実コード確認）

| 型 | 確認結果 | 判定 |
|---|---|---|
| `ParamGroupStep` | 素の `pub trait`（`crates/autodiff/src/nn/optim/param_group.rs:209`）。sealed でも supertrait 制約でもない | 既定実装なしのメソッド追加は破壊的（`trait_method_added`）。既定実装つきは追加のみ |
| `EmaCallback`・`SwaCallback` | フィールドはすべて非公開（`crates/facade/src/compat/callbacks.rs:1250`・`:1403`） | フィールド追加は追加のみ |
| `RequestId` | タプルのフィールドが非公開（`crates/autodiff/src/generate/scheduler.rs:72`） | 追加のみ |
| `SchedulerLimits` | `#[non_exhaustive]`（`scheduler.rs:76`） | フィールド追加は追加のみ |
| `BatchScheduler` | フィールド非公開（`scheduler.rs:329`）。`#[non_exhaustive]` は不要 | 追加のみ |
| `HfHub` | `pub struct`・フィールド `registry` は非公開（`crates/hf-hub/src/lib.rs:337`） | 追加のみ |
| `Module` | open trait（`crates/facade/src/nn/module.rs:45`） | `ParamGroupStep` と同じ扱い |

cargo-semver-checks の分類名は、開発機の 0.51.0 で `cargo semver-checks --explain <名前>` を実行して存在を確認した（ツールの導入ではなく参照のみ）。§1.1 の「要確認」は次で確定する。

- `trait_method_added` の説明は「without a default implementation」。既定実装つきのメソッド追加は追加のみ
- 使う分類名: `enum_variant_added`・`constructible_struct_adds_field`・`derive_trait_impl_removed`・`inherent_method_missing`・`function_parameter_count_changed`・`trait_added_supertrait`・`enum_marked_non_exhaustive`・`struct_marked_non_exhaustive`・`trait_method_added`

## 3. 論点・選択肢・推奨

### 3.1 破壊的変更の扱い

| 案 | 内容 | 評価 |
|---|---|---|
| (a) **推奨** | 承認済みの破壊的変更だけを 0.14.0 の 1 回に集約し、移行ノート（§5）を付ける | 下流の修正が 1 回で済む。0.x の先例（`Cargo.toml` の `[workspace.package]` コメント。0.1.0→0.2.0・0.2.0→0.3.0）と一致 |
| (b) | 破壊的変更を出さない。ROCm の facade 公開は見送り、バックエンド行は「部分的」のまま（または #3758 が比べる別案） | 項目 24 が承認されない場合の帰結 |
| (c) | `#[deprecated]` を併存させる段階移行 | `Device` は variant 追加を非破壊にできる形が無い（`docs/backend-rocm-device-exposure-decision.md` §3 の案 C 不成立）ため不成立 |

推奨 (a) の中身:

- 0.14.0 で破壊的になるのは `Device::Rocm` の追加と `Device` への `#[non_exhaustive]` 付与だけ
- §1.3 のほかの型はすべて「追加のみ」に設計を寄せる
- 「追加のみ」に寄せられない設計は実装せず所有者へ戻す
- 項目 24（ROCm の facade 公開）が Phase 3 の #3771 で承認されなければ、0.14.0 に破壊的変更は無い（(b)）

### 3.2 `#[non_exhaustive]`

- **推奨**: `Device` に限って付ける。付与自体が破壊的（`enum_marked_non_exhaustive`）なので、**同一リリースで当該型の破壊的変更が確定している型にだけ**付ける（二重に壊さない）。下流は 1 回の修正（`_` 腕の追加）で、以後の variant 追加から守られる。`Device` は `Metal` が `cfg(target_os = "macos")` で既に環境により variant 集合が変わる（`crates/tensor-core/src/device.rs:46-47` は現状 `#[non_exhaustive]` なし）
- ROCm が承認されない場合、0.14.0 では後付けしない（単独の後付けは破壊だけが残る）
- `OnnxValue`: variant 追加は既定で行わず、必要なら別型・別メソッドで足す。設計上 variant 追加が必要になれば #3759 が所有者へ戻す
- ほかの既存 enum・struct への遡及付与はしない（`docs/compat-api-scope.md` §0 の `FusedOpKind` の記述と同じ理由）
- 影響:
  - `_` 腕が必須になるのは別クレートの `facade::resolve_ops`（`crates/facade/src/lib.rs:2539`。`Device` を網羅 `match`。`docs/backend-rocm-device-exposure-decision.md` §2）だけで、`_` 腕は型付き `DeviceUnavailable` を返す。`tensor-core::select_from`（`crates/tensor-core/src/device.rs:171`）は `Device` と同一クレートなので `#[non_exhaustive]` の影響を受けず、`_` 腕を足さず網羅 `match` を維持する（variant 追加時にコンパイラが対応漏れを検出できる状態を保つ）
  - `api_surface.rs` は integration test（別クレート視点）なので、ワイルドカード腕なしの G1 はコンパイルできなくなる。反転の方針は「ワイルドカード腕つきの分類」と「ソース走査による variant 名の在庫テスト（`pb_has_non_exhaustive_before`〈`api_surface.rs:24949`〉と同型）」で集合を固定し、variant 追加がテストの失敗として表れるようにする。具体的なテスト名は #3758 の設計記録が確定する

### 3.3 版数

- **推奨 0.14.0**（`workspace.version` の lockstep。`docs/crates-io-publishing-order.md` §4。破壊の有無によらず 0.x の minor）
- 1.0.0 は不採用。29 項目の公開が途中で、安定宣言の条件（互換契約の保守体制）をまだ決めていないため
- 破壊的変更が無い場合も 0.14.0（追加が多数のため patch にしない）
- `=0.13.0` で固定している利用者（framework-compare の第 9 区分ピン等）は、明示的にピンを上げるまで影響を受けない
- 運用ゲートの先例: `docs/autodiff-var-dtype-multiplexing-design.md` 契約 4（`BREAKING CHANGE` 運用と版数運用は不可分）
- **spec との食い違い**: 0.x の minor での破壊は版上は合法だが、spec の「追加のみ」の文言（`docs/spec/04-requirements.md` の `:236`・`:238`・`:383`・`:405`。計画時点の行番号で、本 worktree では submodule 未展開のため再確認していない）と食い違う。**spec は編集しない**。(a) は #3715 の改訂提案が必要、(b) は不要。改訂提案を spec リポへ出すかは #3718 の論点 3

### 3.4 所有者へ戻す論点

| 論点 | 内容 | 推奨 | 確認がないときの扱い |
|---|---|---|---|
| 1 | 破壊的変更を 0.14.0 に集約するか（§3.1） | (a) | 確認があるまで実装へ進めない（保留ガード維持） |
| 2 | `Device` への `#[non_exhaustive]` 付与（§3.2） | `Device` のみ | 同上 |
| 3 | 版数と spec 文言の食い違い・改訂提案の要否（§3.3） | 0.14.0・#3715 で提案 | 同上 |

## 4. 型ごとの扱い

#3713 §1.3 の全行に対応づける。項目 24 以外で「破壊的」判定が出た場合は、§3.4 の表へ足して所有者へ戻す。

| 区分 | 型（N 番号） | 扱い |
|---|---|---|
| 破壊的・承認要 | `Device`（24） | `Rocm(usize)` 追加（`enum_variant_added`）と `#[non_exhaustive]` 後付け（`enum_marked_non_exhaustive`）。承認は #3718・#3771。未承認の間は追加しない |
| 追加のみを維持する設計が条件 | `Module`（8）・`ParamGroupStep`（19） | 既定実装つきのメソッドに限る。必須メソッドが要るなら新 trait を切る |
| 同上 | `FitConfig`（16・18） | `Copy`／`Eq` を維持する。`f32` は bits で保持（先例 `validation_split_bits`）。別の設定型へ逃がすのでも可 |
| 同上 | `OnnxValue`（25） | variant 追加は既定で禁止。別型・別メソッドで足す |
| 同上 | safetensors の入口（27） | ジェネリック化は不可。別名の入口を足す |
| 追加のみ | 上記以外の §1.3 の全行（`Optimizer`〈既に `#[non_exhaustive]`。§1.4〉・`FitConfig` のフィールド追加・`OnnxModel`・`OnnxError`・`OnnxExportOptions`・`SaveError`／`LoadError`・`GenerateConfig`・`SpeculativeConfig`・`BatchScheduler` 系・`ModelError`・`DownloadOptions`・`DeviceParamStore`・`HfHub` ほか） | フィールド・variant・メソッドの追加は §1.1 の規則で非破壊 |

## 5. 移行ノートの形

- 置き場所: `docs/migration/0.14.0.md`（次の版以降も同じ置き場所を使う）
- 雛形の作成は #3780（Phase 4）。Phase 5〜9 の実装 PR が該当欄を埋め、Phase 11 のバンプで確定する。公開用の写し（`site/` や注釈付き tag メッセージ）は Phase 11 の判断とし、本 issue では決めない
- 書式は CHANGELOG 相当の固定の節: 追加した API／破壊的変更と下流の直し方／挙動のみの変更／`#[deprecated]` の追加／既定値の変更なしの確認欄／対応表の行の変化
- 破壊的変更 1 件あたりの項目: 型・パス、変更の種類（semver-checks の分類名）、コンパイルエラーの出方、before／after のコード、承認 URL、関連 PR
- 作例（`Device::Rocm`）:
  - before: `Cpu`・`Cuda(i)`・`#[cfg(target_os = "macos")] Metal` の網羅 `match`
  - after: `_` 腕を足す。推奨は `_ => Err(...)` か既定バックエンドへのフォールバック
  - 承認で見送りになった場合は、空欄の理由を書く欄にする
- 先例: `docs/compat-api-scope.md` §4.2・§4.3 の `#[deprecated]` シム

## 6. 破壊的変更の検出方法

| 案 | 内容 | 評価 |
|---|---|---|
| (a) **推奨** | 既存機構で担保（導入 0 件）: `api_surface.rs` の正ガード・否定ガード、variant 在庫のソース走査テスト、doctest の網羅 `match`（失敗系は `compile_fail` に頼らず正のプローブで組む） | 依存・CI の変更なし |
| (b) 任意の補助 | Phase 11 の事前監査で cargo-semver-checks を手動の参考実行として使う。基準は crates.io の `0.13.0`（要ネットワーク）。開発機に導入済みで workspace の依存でも CI でもない。結果は「未実測」のまま、実行した PR が書く | 導入なし |
| (c) | CI ジョブ化 | **本 issue では推奨しない**。ツール導入と固定が依存・CI の変更になる。`main-protection` の required contexts 更新が要る（実装 Agent は ruleset を変更しない。`.claude/rules/ci.md`）。公開 7 クレートの基準版が複数要る。所有者承認事項 |

## 7. ガードレールとの関係

- 事実: `crates/guardrail/src/checks/api_stability.rs` の `public_api_broken` は、変更された `.rs` の `pub fn`／`pub struct`／`pub enum` の**行が baseline から消えたか**だけを見る行単位比較である。`Rocm(usize)` の variant 追加・`#[non_exhaustive]` の後付け・derive の喪失・trait への必須メソッド追加は検出できない
- CI が PR の diff に `guardrail check` を適用しているかは `ci.yml`・`scripts/run-guardrail-regression.sh` を grep して確認した範囲では、guardrail 自身の回帰検証（layer1／layer2）のみで、PR の公開 API diff への適用は見当たらない（未確認事項として残す）
- 方針:
  1. 承認済みの破壊的変更は人間が起こすレビュー済み PR で入れる。自己修復ループの自動適用対象にしない
  2. ガードレール閾値・5 条件・ポリシー除外リスト・`guardrail.toml` は変えない。承認を除外ルールに登録する経路を作らない（`.claude/rules/security.md` A08）
  3. 補償の統制は、破壊を試みるループの変更が `api_surface.rs` の保留ガード／在庫テストをビルド・テストゲートで落とす点に置く
- 検出強化（variant 追加・derive 喪失の検知）は本 issue の範囲外。所有者承認後の別 issue 提案として §11 に記す

## 8. 判定手順（設計記録の「互換への影響」が参照する正）

1. 変更が触れる公開型を §4（#3713 §1.3 の表）で引く。表に無ければ先に表へ足して判定する
2. §1.1 の規則で判定する。次のいずれかは破壊的: `#[non_exhaustive]` なし enum への variant 追加、構築可能な struct へのフィールド追加、derive の喪失（`Eq` 等）、既定実装なしの trait メソッド追加、既存シグネチャの変更、`#[non_exhaustive]` の後付け
3. 破壊的なら、§4 の承認済みの行にある場合だけ進める。**現時点で承認済みの行はない**（§2 のとおり所有者承認は未取得で、`Device` も承認要のまま）。`Device` を含め、所有者承認の URL・日付と、必要な spec 改訂の受理を確認できるまで実装へ進まない。それ以外は設計を「追加のみ」へ直す（既定実装・bits 保持・別名の入口）。直せないなら §3.4 の表に足して所有者へ戻し、承認までは実装せず保留ガードを維持する
4. 結論は設計記録の「互換への影響」に「追加のみ／破壊的で承認要／該当なし」と理由で書く。破壊的なら移行ノート（§5）・追跡表・実装 PR の `type(scope)!:` と `BREAKING CHANGE:` フッタ（`.claude/rules/conventional-commits.md`）を揃える

## 9. 保守・運用チェックリスト

| 節 | 結果 |
|---|---|
| 公開形 | 該当なし（公開面を変えない） |
| rustdoc／doctest | 該当なし（コード変更なし） |
| エラー型 | 該当なし |
| テスト計画 | 該当なし。G1／G2 の反転方式は §3.2 に記録 |
| 数値契約 | tolerance・baseline・FMA は不変 |
| 互換への影響 | 本記録そのもの |
| 適用記録・索引・追跡表 | `docs/README.md` の索引を更新。`compat-api-scope.md` の適用記録は該当なし。追跡表は更新しない（承認列は #3718 待ち） |
| ベンチ・性能 | 該当なし |
| 依存 | 追加なし。cargo-semver-checks は導入しない。新規 `unsafe` なし |

## 10. セキュリティ観点

docs のみの変更で、OWASP A03 の実装面は非該当。A08 の観点では「承認済み変更を自己修復の判定迂回に使わない」を §7 に反映した。秘密情報・実ホスト名は書いていない。

## 11. スコープ外（所有者承認後に起票。無断で起票しない）

- ガードレールの検出強化（§7）
- cargo-semver-checks の CI 化（§6）
- spec 改訂提案（#3715）
- 移行ノート雛形（#3780）
- 追跡表の自動集計（#3781）

## 12. Phase 2 設計記録の横断突き合わせ（イシュー #3764）

イシュー #3764（親 #3712 系。Phase 2 の最後の集約）。Phase 2 で追跡表 `docs/compat-matrix-ari-tracker.md` の 29 項目の設計記録がそろったので、Phase 3 の承認（#3766〜#3771）へ進む前に、次の 4 点を 1 か所に集める。

1. 各設計記録の「互換への影響」を §3.1・§4・§8 に照らした集約表（12.1）
2. 記録のずれ・訂正事項（12.2）
3. 移行ノート `docs/migration/0.14.0.md`（雛形は #3780）に載せる変更（12.3）
4. Phase 3 で問う論点の領域別一覧（12.4）と、依存・`unsafe` の件数（12.5）

### 12.0 位置づけ・境界

- 基準面: `origin/main` の `52ca9bf3a`（2026-11-01 確認）。コード・依存・tolerance・baseline・spec・CI は変更しない
- 本節は集約と突き合わせだけを行う。§1〜§11 の本文は書き換えない（食い違いは 12.2 に訂正事項として書く。§1.4 と同じやり方）。各設計記録の本文も書き換えない
- 本節は承認をしない。#3718 には所有者の回答の記録コメント（<https://github.com/Fandhe-AI/fandhe-ai/issues/3718#issuecomment-6543681116>、2026-10-31）があり、中身は「`Device` だけ 0.14.0 で・版数 0.14.0・spec 改訂提案は 1 件だけ出す・追跡表と雛形を承認」である。これは事実として引くだけで、§2 の承認欄・追跡表の承認列への転記は行わない（同コメントのとおり Phase 4 以降の最初の PR が行う）
- `Device::Rocm` の追加そのものの承認は #3771 の回答待ちである。spec 改訂提案 Fandhe-AI/fandhe-ai-spec#87 は 2026-11-01 時点で OPEN で、受理は未確認
- 推奨（実装／対象外）は各設計記録が書いたもので、いずれも所有者の承認前である。「承認済み」と言えるのは `backend-gpu-linalg-decision.md` §11 の公開形（#3140 のコメント 6227907703）だけである

### 12.1 集約表（追跡表の 29 項目）

1 行 1 項目で、キーは追跡表の通し番号 # である。設計記録の節は各 doc の見出しで解決することを確認した（見出しの書式は `## N.` と `## §N` の 2 種がある）。

区分欄の読み方（12.2 (c) の統一規則）:

- **追加のみ**: 公開名（型・関数・メソッド・variant・クレート）を足す。既存のシグネチャ・derive・variant は不変
- **該当なし**: 公開名も型も変わらず、変わるのは挙動だけ（`Err` が `Ok` になる受理範囲の拡大など）。挙動は 12.3 の「挙動のみの変更」欄に載せる
- **破壊的で承認要**: §8 手順 2 の破壊的条件に当たる。承認は #3718・#3771 の条件を満たすまで実装しない

Phase 3 の issue 本文の「項目 N」は、#3766・#3767 では追跡表の # と同じ（項目 1〜5）だが、#3768 以降は追跡表の # + 1 になる（例: #3768 の項目 8〜16 は追跡表 # 7〜15、#3769 の項目 17〜24 は # 16〜23、#3770 の項目 26〜28・30 は # 25〜27・29、#3771 の項目 25・29 は # 24・28）。同じ注記が #3750 §0 にもある。追跡表 # 6 に当たる Phase 3 の issue は無い。

| # | 項目 | Phase 3 | 設計記録（issue と節） | 推奨 | 触れる公開型 | 区分 | 備考・§1.2 の見立てとの差 |
|---|---|---|---|---|---|---|---|
| 1 | 低精度演算の `Var` 以外の配置 | #3766 | #3721（`autodiff-low-precision-op-extension-decision.md` §8） | 対象外（Q1 の推奨）。実装するなら案 C | なし（実装なら新モジュール・新関数） | 該当なし（実装なら追加のみ） | 対象外として数えるには Q2 の個別承認が要る（D-0 未掲載のため） |
| 2 | `TapeF64` の `jvp`・`jacfwd`・`jvp_multi`・`jvp_create_graph` | #3766 | #3722（`autodiff-functional-transforms-design.md` §35）、#3723（同 §36） | 実装 | `TapeF64`（メソッド追加）、新型 `CreateGraphResultF64`・`JvpCreateGraphResultF64` | 追加のみ | `VarF64`・`GradientsF64`・f32 の `Tape` は不変 |
| 3 | `create_graph` の C 区分 24 Op | #3767 | #3724（`autodiff-higher-order-grad-decision.md` §34）、#3725（同 §33）、#3726（同 §35）、#3727（同 §36）、#3728（同 §37）、#3739（同 §48） | 実装（全族。RNN セルの採否は #3728 に従い個別確認） | なし（`Op` は `pub(crate)`） | 該当なし | `Err` が `Ok` になる挙動。記録の表記は §34・§48 が「該当なし」、§33・§35〜§37 が「追加のみ」と割れている（12.2 (c)） |
| 4 | `create_graph` の B 区分の未実装 Op | #3767 | #3729（同 §38）、#3730（同 §39）、#3731（同 §40）、#3732（同 §41）、#3733（同 §42）、#3734（同 §43）、#3735（同 §44）、#3739（同 §48） | 実装。一部対象外: `CtcLoss` は対象外（§40）。`Pinv`・`Lstsq` は打ち切りなしの入力だけ受理（§43） | なし（`Op` は `pub(crate)`、オプション型・`Reduction` は不変） | 該当なし | 記録の表記は「追加のみ」。`Err` が `Ok` になる挙動 |
| 5 | 二階微分の拒否範囲と意味論 | #3767 | #3736（同 §45）、#3737（同 §46）、#3738（同 §47）、#3739（同 §48） | 実装。一部対象外: §45 は S0（案 A）で `LayerNorm`・`RmsNorm`・`BatchNorm` の拒否を狭める。§46 は `Det` だけ受理へ広げ、`Inv`・`Solve`・`SlogdetLogAbsDet` は対象外。§47 は B1（PyTorch 準拠） | なし | 該当なし | §45 の S0 は issue 題の S2 とは違う。§47 は**既に `Ok` を返している入力の値が変わる**（`padding_idx: Some(p)` の `hessian`・`hvp`・`jvp_create_graph` の列 `p`） |
| 6 | GPU 線形代数の opt-in 関数 | なし | 既存 `backend-gpu-linalg-decision.md` §11（Phase 2 の設計 issue なし） | 実装（公開形は #3140 のコメント 6227907703 で承認済み。実装は #3405） | facade の関数 2 件の追加 | 追加のみ（§11.2 の 9） | Phase 3 で問う論点なし。facade 経由の実機 parity の測定は #3866 |
| 7 | `add_upsample` の `scale_factor` 指定 | #3768 | #3740（`autodiff-spatial-layers-decision.md` §9）、#3750（`compat-new-layers-io-export-decision.md` §10〜§12） | 実装（案 A。別名メソッド `add_upsample_scale_factor`） | `Sequential`（inherent メソッド 1 件） | 追加のみ（案 B なら破壊的で承認要） | `add_upsample`・`UpsampleSize` は不変 |
| 8 | 独自 `Module` から組み込み層を呼ぶ経路 | #3768 | #3741（`facade-nn-module-exposure-decision.md` §23） | 実装（案 A。`compat::Sequential` に `nn::Module` の impl を足す） | `compat::Sequential`（trait impl の追加 1 件） | 追加のみ | §1.2 は「trait にメソッドを足す」と見込んだが、実際は trait impl の追加で `Module` のメソッドは増えない |
| 9 | C1: Hardshrink・Softshrink | #3768 | #3742（`nn-shrink-bilinear-channel-shuffle-decision.md` §1）、#3750（`compat-new-layers-io-export-decision.md` §10〜§12） | 実装（既存 Op の合成） | `Var`・`compat::Sequential`（inherent メソッド各 2 件） | 追加のみ | 差なし |
| 10 | C2: Bilinear | #3768 | #3743（同 `nn-shrink-...` §2） | 実装（案 A。既存 Op の合成） | `Var`（inherent メソッド 1 件） | 追加のみ | `Sequential::add_*` は作らない。§1.2 の # 10 行（`add_*` を見込む）とずれており、同記録の Q7 で #3768 が確かめ #3818 が直す |
| 11 | C3: BatchNorm3d | #3768 | #3744（`autodiff-batch-norm-3d-decision.md` §0〜§12）、#3750 | 実装（案 A） | `Var::batch_norm` 系（シグネチャ不変）、`Sequential::add_batch_norm3d` | 追加のみ | rank 5 が拒否から受理に変わる。rank 6 以上の `RankMismatch.expected` の値が 4 から 5 に変わる（挙動のみ） |
| 12 | C5: CosineSimilarity | #3768 | #3745（`autodiff-similarity-distance-decision.md` §0〜§12） | 実装（案 A） | `Var`（inherent メソッド） | 追加のみ | 差なし。# 13 と同じ記録 |
| 13 | C6: PairwiseDistance | #3768 | #3745（同上） | 実装（案 A） | `Var`（inherent メソッド） | 追加のみ | 差なし。# 12 と同じ記録 |
| 14 | C7: ChannelShuffle | #3768 | #3746（同 `nn-shrink-...` §3）、#3750 | 実装（案 A） | `Var`・`compat::Sequential`（inherent メソッド各 1 件） | 追加のみ | 差なし（§3.6 が # 14 行との一致を確認済み） |
| 15 | C9: 3 次元適応プーリング・LPPool・FractionalMaxPool | #3768 | #3747（`autodiff-pool3d-ops-decision.md` §18）、#3748（同 §19）、#3749（同 §20）、#3750 | 実装 | `Var`・`compat::Sequential`（inherent メソッド）。`BackendOps` は内部クレートで既定実装つきメソッド追加 | 追加のみ | FractionalMaxPool の乱数契約は #3768 で個別に確認する |
| 16 | SWA の `FitConfig` 形 | #3769 | #3751（`autodiff-swa-decision.md` §20） | 対象外（案 A。`Callback::Swa` だけを正とする） | なし（案 B なら `FitConfig` に bits で保持するフィールド） | 該当なし（案 B なら追加のみ） | `f32` を直接持つ形は `Eq` を失い破壊的になるため採らない |
| 17 | A1: EMA と `compile_with_amp` の併用 | #3769 | #3752（`autodiff-ema-decision.md` §17） | 実装（案 A。skip に追従） | なし（シグネチャ・variant・derive は不変） | 該当なし | `Err` が `Ok` になる挙動。記録の表記は「追加のみ」（12.2 (c)） |
| 18 | A2: EMA の減衰率ウォームアップ | #3769 | #3753（同 §18） | 実装（案 A。既定 OFF） | `EmaCallback`・`ExponentialMovingAverage`（非公開フィールドと inherent メソッド 2 件） | 追加のみ | 既存 API の意味論は既定 OFF で不変 |
| 19 | A3-a: param groups と AMP の併用 | #3769 | #3752（`autodiff-param-groups-decision.md` §14。skip 規則は `autodiff-ema-decision.md` §17） | 実装（(a) 新しい inherent メソッド `compile_with_amp_and_param_groups`） | `Sequential`（inherent メソッド） | 追加のみ | §1.2 の「`ParamGroupStep` へのメソッド追加」には触れない（`AmpConfig`・`ParamGroup`・`ParamGroupStep` は不変） |
| 20 | A3-b: param groups と `LrSchedule` の併用 | #3769 | #3754（同 §15） | 実装（案 A） | なし（groups があるときの `Err` を `Ok` にするだけ） | 該当なし | 記録の表記は「追加のみ」。groups 空の既定経路は bit 不変 |
| 21 | A3-c: param groups の `state_dict` 保存 | #3769 | #3755（同 §18） | 実装（案 A） | なし（`save_model`／`load_model`・`ModelIoError` は不変） | 該当なし（保存形式の追加） | 記録の表記は「追加のみ」。新リーダーは旧ファイルを読めるが、旧リーダーは新ファイルを fail-closed で拒否する（前方互換なし） |
| 22 | A3-d: param groups と `DeviceParamStore` 常駐更新 | #3769 | #3756（同 §19） | 実装（案 A） | `Sequential`（inherent メソッド）、`DeviceParamStore`（非公開フィールド）。内部の `BackendOps` に既定実装つきメソッド | 追加のみ | 新しい FFI `unsafe` が要る見込み（12.5）。実機 parity は #3862・#3863 |
| 23 | A3-g: `Optimizer` に Adadelta・Adamax・NAdam・RAdam | #3769 | #3757（`compat-optimizer-variants-0.14.0-decision.md` §0〜§13） | 実装 | `Optimizer`（既に `#[non_exhaustive]` のため variant 追加は非破壊。§1.4） | 追加のみ | model_io の kind を足す（`format_version` 2 は据え置き）。旧リーダーは未知の kind を fail-closed で拒否する |
| 24 | ROCm の facade 公開（`Device::Rocm`） | #3771 | #3758（`backend-rocm-device-exposure-decision.md` §10） | 実装（案 A。cfg なしで `Device::Rocm(usize)`） | `Device`（`Rocm(usize)` の追加と `#[non_exhaustive]` の後付け） | **破壊的で承認要**（`enum_variant_added`・`enum_marked_non_exhaustive`） | `_` 腕が要る箇所が §3.2 の「`resolve_ops` だけ」より広い（12.2 (b)）。`BackendError` は追加のみ。`available_devices`・`tape_for` などの戻り値に値が増える |
| 25 | B1: ONNX モデルの学習の facade 公開 | #3770 | #3759（`onnx-autograd-decision.md` §8） | 実装（案 A。`OnnxModel` への薄い委譲） | `OnnxModel`（メソッド追加）、`OnnxError`（`#[non_exhaustive]` への variant 追加）、新 3 型 | 追加のみ | `OnnxValue` は不変（§1.5 の懸念は解消）。6 op のエラーが内部で `UnsupportedOp` から `UnsupportedInAutograd` に変わる（facade 外の挙動差） |
| 26 | B3: ONNX export の `Sigmoid` | #3770 | #3760（`facade-onnx-export-exposure-decision.md` §19） | 実装（(α)。既存の統一複合判定で検証） | facade の公開項目は変わらない。autodiff の `Module::as_sigmoid`（既定実装つき） | 追加のみ（autodiff のフック）。facade の型は該当なし | これまで `Err` だった `Sigmoid` を含むモデルが `Ok` になる。単体の publish dry-run は registry 上の autodiff を参照して失敗しうる（一括 dry-run は成立） |
| 27 | B5: safetensors の f32 以外の型 | #3770 | #3761（`facade-safetensors-exposure-decision.md` §13） | 実装（案 C。別名の入口） | `LoadError`（`#[non_exhaustive]` への variant 追加）、新 2 型、別名の関数 | 追加のみ | 既存 7 アイテムのシグネチャは不変。§1.2 の見立てはジェネリック化の余地を残していた |
| 28 | HF Hub 連携の 2 段目（`hf-hub` の crates.io 公開と数え方） | #3771 | #3762（`hf-hub-integration-design.md` §10） | 実装（A。0.14.0 で公開） | 新クレートの全公開型（初回公開で基準版なし）。facade の型は該当なし | 追加のみ（新クレート） | 公開後は `EXPECTED_LIB_PUBLIC` が semver 面になる。`download` の公開とリダイレクト先ホストの allowlist は別承認 |
| 29 | `caches` で表せない内部状態を持つモデルの speculative decoding・連続バッチング | #3770 | #3763（`facade-speculative-decoding-batching-design.md` §29） | 実装（B2。別 trait＋専用の入口） | 新 trait `StatefulAutoregressiveModel`、新しい自由関数、`step_stateful`（inherent メソッド） | 追加のみ | 条件: `BatchScheduler` の `Send`／`Sync` を保つ（型消去の箱に `Send + Sync` を付け、正のプローブで固定） |

**結論**: 破壊的なのは # 24 `Device` の 1 件だけで、§3.1 (a)・§3.2 と #3718 の回答の記録（`Device` だけ 0.14.0）に一致する。ほかの 28 項目は「追加のみ」か「該当なし」で、設計を「追加のみ」に寄せる条件（既定実装・bits 保持・別名の入口・`OnnxValue` に variant を足さない）を各記録が満たしている。`ParamGroupStep` と `Module` には必須メソッドを足さない設計になっている（§4）。

### 12.2 訂正事項・記録のずれ（§1〜§4 は書き換えず、ここへ記録する）

- **(a) §1.2 の見立てと実際の設計の差**: # 7（別名メソッド）、# 8（メソッドでなく trait impl）、# 10（`add_*` を作らない）、# 19（`ParamGroupStep` でなく新しい inherent メソッド）、# 25（`OnnxValue` は不変）、# 27（ジェネリック化でなく別名の入口）。いずれも設計が見立てより「追加のみ」の側に寄っており、判定は変わらない。# 10 は記録側の Q7 で #3768 が確かめ、#3818 が §1.2 を直す
- **(b) §3.2 の `_` 腕の在庫**: §3.2 は「`_` 腕が必須になるのは `resolve_ops` だけ」と書くが、`backend-rocm-device-exposure-decision.md` §10.6 は doctest・保留ガード G1・テスト 4 本にも `_` 腕が要るとする。判定（破壊的で承認要）は変わらない。`Device` の承認が出た後に #3853 が在庫を数え直す
- **(c) 「Err→Ok で公開型は不変」のラベルの割れ**: 同じ種類の変更に、記録によって「追加のみ」（higher-order の §33・§35〜§44、EMA §17、param groups §15・§18 など）と「該当なし」（higher-order の §34・§45〜§48）が付いている。§8 手順 4 の読み方を本節で統一した（12.1 の区分欄の定義）。**型の区分は公開名を足すかで決め、挙動の変化は区分と別に 12.3 へ載せる**。記録の本文は書き換えず、どちらのラベルでも §8 の結論（破壊的でない）は同じである
- **(d) 追跡表 # 6 の承認欄**: 「前提の実機 parity は未実測のため公開の条件は未充足」は、`backend-gpu-linalg-decision.md` §11.1 の G2 充足の記述（2026-10-28。CUDA は `docs/perf/logs/cuda-linalg-parity-3138/README.md`、Metal は `docs/perf/logs/metal-linalg-lu-chol-qr-2676/README.md`）と食い違っていた。G2 は充足済みで、未実測なのは facade 経由の実測（#3866）である。§11.1 を読み直して確かめたうえで、追跡表の該当セルの後半だけを事実に直した。承認の値（公開形は承認済み）は変えない
- **(e) `docs/README.md` の索引**: 本書の注釈にあった「承認は #3718 待ち」は、#3718 に回答の記録コメントがあることと合わない。本 PR で「#3718 に回答の記録あり（転記は Phase 4）」へ直した。本書 §2 の承認欄自体は転記まで「未承認」のまま残る

### 12.3 移行ノートに載せる変更（雛形 #3780 の欄に合わせる）

| 欄 | 載せる変更 |
|---|---|
| 破壊的変更 | `Device::Rocm` の追加と `#[non_exhaustive]` の付与だけ（# 24）。#3771 Q1 の承認と spec#87 の受理が条件。見送りになった場合は §5 の「空欄の理由」欄を使う |
| 追加した API | # 2・# 6〜# 15・# 18・# 19・# 22・# 23・# 25〜# 29 の公開名。新クレート `fandhe-ai-hf-hub`（# 28）。承認された場合は `fandhe-ai-backend-rocm`（# 24） |
| 挙動のみの変更（`Err` が `Ok` になる） | # 3・# 4・# 5（§45・§46）、# 11（rank 5 の受理）、# 17、# 20、# 21（groups の保存）、# 26（`Sigmoid`） |
| 挙動のみの変更（**既に `Ok` の値が変わる**。独立した行にする） | # 5 の §47（`padding_idx: Some(p)` の二階。案 A なら値は変わらず注記だけ） |
| 挙動のみの変更（その他） | # 11 の rank 6 以上の `RankMismatch.expected`（4 から 5）。# 24 の `available_devices`・`tape_for`・`release_cached_memory`・`memory_pool_stats` の戻り値に値が増えること。# 25 の内部エラー種別の変化（facade 外） |
| 保存形式の前方互換なし（旧リーダーが fail-closed で拒否する） | #3750 の新しい kind（# 9〜# 15）、# 21 の groups を含む新ファイル、# 23 の `Optimizer` の kind |
| 対象外になれば載せないもの | # 1・# 16（推奨は対象外）、`CtcLoss`、`Inv`・`Solve`・`SlogdetLogAbsDet` |

### 12.4 Phase 3 で問う論点の領域別一覧

各記録の「所有者へ戻す論点」の節から、論点を領域別に引く。**(新)** 列は、承認 issue の本文が問っておらず記録だけが挙げている論点で、承認 issue の回答時に取りこぼしやすい。

| 領域 | 承認 issue | 引く節 | 論点（承認 issue 本文が問うもの） | (新) 記録だけが挙げる論点 |
|---|---|---|---|---|
| dtype・関数変換 | #3766 | `autodiff-low-precision-op-extension-decision.md` §8.11、`autodiff-functional-transforms-design.md` §35.13・§36.12 | 低精度の別配置を実装するか（推奨は対象外）、`TapeF64` の `jvp` 系を実装するか、f64 の数値契約 | 低精度の Q2（対象外として数えるための個別承認）と Q3（対象外時の保留ガードの維持と文言更新） |
| 二階微分 | #3767 | `autodiff-higher-order-grad-decision.md` §33.14〜§48.11 | C 区分 24 Op の採否、B 区分の採否、拒否範囲、`padding_idx` の意味論、GPU parity の判定方式 | §40 の `CtcLoss`、§43・§46 の Op ごとの採否、§47.16 の「既に `Ok` の値が変わる」点、§48.11 の 10（ROCm の parity は # 24 の承認後） |
| NN | #3768 | `autodiff-spatial-layers-decision.md` の「所有者へ戻す論点」、`compat-new-layers-io-export-decision.md` §13、`facade-nn-module-exposure-decision.md` §23.10、`nn-shrink-...` §1.11・§2.11・§3.11、`autodiff-batch-norm-3d-decision.md` §11、`autodiff-similarity-distance-decision.md` §11、`autodiff-pool3d-ops-decision.md` §18.12・§19.12・§20.12 | `add_upsample` の公開形、独自 Module 経路、各層の公開形の一括承認、保存形式と ONNX export | Bilinear の Q7（§1.2 の記録のずれの是正）、#3750 の移行ノートへの記載、FractionalMaxPool の乱数契約の個別確認 |
| 最適化 | #3769 | `autodiff-swa-decision.md` §20.10、`autodiff-ema-decision.md` §17.10・§18.10、`autodiff-param-groups-decision.md` §14.10・§15.10・§18.10・§19.10、`compat-optimizer-variants-0.14.0-decision.md` §12 | SWA の公開形、AMP の skip 追従、保存形式、残りの一括承認 | SWA の論点 4（§16.5 の理由の置き換え）、# 22 の新しい FFI `unsafe` の監査条件、# 23 の旧リーダーが kind を拒否すること |
| 相互運用・推論 | #3770 | `onnx-autograd-decision.md` §8.10、`facade-onnx-export-exposure-decision.md` §19.10、`facade-safetensors-exposure-decision.md` §13.10、`facade-speculative-decoding-batching-design.md` §29.10 | ONNX 学習の公開形、`Sigmoid` の数値契約、safetensors の入口、内部状態モデルの採否 | speculative の `Send`／`Sync` の条件 |
| バックエンド・公開クレート | #3771 | `backend-rocm-device-exposure-decision.md` §10.10、`hf-hub-integration-design.md` §10.10、本節 | `Device::Rocm`、`fandhe-ai-backend-rocm` の公開、`hf-hub` の公開、規則 3 の読み方、`download` の公開 | ROCm の gfx target の照会と新規 `unsafe`（12.5）、`download` 公開時のリダイレクト先ホストの allowlist |

追跡表 # 6 は「Phase 3 で問う論点なし（#3140 で承認済み・実装は #3405）」である。

### 12.5 依存の追加と新規 `unsafe`

- **依存の追加が要る設計は 0 件**。# 1〜# 29 の記録の依存節はすべて「追加なし」で、`docs/license-matrix.md` の更新も要らない見込みである（# 28 の `ureq` と # 24 の `libloading` は承認済みの区分の再利用で、新しい allow 識別子が要るかは #3773・#3774 が `cargo deny` で実測する）
- 0 件のそばにある事項（どれも新しいクレートの追加ではない）:
  - # 22（A3-d）: CUDA・Metal の grouped カーネル起動で既存と同じ形の FFI `unsafe` が要る見込み。SAFETY コメントと security-auditor の監査を #3837 の条件にしている
  - # 24（ROCm）: 新規 `unsafe` が要るかは論点 3 の選択で変わる（gfx target の照会を HIP のシンボル追加で行う場合は `docs/backend-rocm-unsafe-audit.md` への追記と監査が要る）
  - # 24・# 28: 承認済みの `libloading =0.9.0`（第 12 区分）と `ureq =3.4.2`（第 11 区分）が、公開クレートの直接依存として crates.io に出る（版・feature は不変。公開準備は #3774・#3773）
  - # 26: `fandhe-ai-onnx-interop` から autodiff の新しいフック `Module::as_sigmoid` への依存で、publish の順序に注意が要る（`docs/crates-io-publishing-order.md` §8.1）

### 12.6 保守・運用チェックリスト

| 節 | 結果 |
|---|---|
| 公開形 | 該当なし（docs のみ。公開面を変えない） |
| rustdoc／doctest | 該当なし（コード変更なし） |
| エラー型 | 該当なし |
| テスト計画 | 該当なし（docs のみ）。検証は参照の解決・表の形・網羅の機械確認で行った |
| 数値契約 | tolerance・baseline・FMA は不変 |
| 互換への影響 | 本節そのもの（12.1）。型の区分は 12.1 の定義に統一した |
| 適用記録・索引・追跡表 | `docs/README.md` の索引を更新。追跡表は注記 1 行と、事実として誤っているセルの訂正だけ（列・ヘッダ・承認の値・0.14.0 の区分は不変） |
| ベンチ・性能 | 該当なし |
| 依存 | 追加なし（12.5）。新規 `unsafe` なし |

### 12.7 セキュリティ観点

docs のみの変更で、OWASP A03 の実装面は非該当。A08 の観点は 2 点: 承認を代わりに記述しない（#3718 は URL つきの事実として引くだけで、承認欄へは転記しない）、本節の集約を自己修復の判定を迂回する根拠に使わない。秘密情報・実機のホスト名は書いていない。

### 12.8 スコープ外

- 移行ノートの雛形（#3780）、承認の転記（Phase 4 以降の最初の PR）、spec#87 の受理、追跡表の自動集計（#3781）
- 12.2 で見つけた記録のずれの是正（各実装 PR。例: # 10 の #3818）
- 新しい issue は起票しない
