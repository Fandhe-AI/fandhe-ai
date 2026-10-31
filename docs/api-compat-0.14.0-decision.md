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
