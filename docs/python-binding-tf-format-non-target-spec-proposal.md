# Python バインディング・TensorFlow 系モデル形式の非目標の spec 明記提案（(b) 形式提案文案・実装しない）（#2193）

基準コミット: `9fe4b5231ad53883e4aff072e44a589ac548baaf`。`docs/spec` submodule ポインタ `e43704a7baefd1489d3f1716571064ab65c5eed6`。`file_path:line` は同コミット時点のもの。後続の変更で行番号がずれる可能性があるため、参照する際は当該コミット、または近傍のコミットで再確認すること。

## §0 結論（最初に読む）

- **コード変更なし**（`crates/**`・`Cargo.toml`／`Cargo.lock`・`docs/spec/`〈正本 submodule〉・tolerance／baseline・ガードレール閾値は一切変更していない）。
- 本 doc は、Python バインディング（PyO3／maturin 等による Python 拡張モジュール配布）と TensorFlow 系モデル形式（SavedModel・TFLite・Keras H5）の非目標について、**実装リポ側で最初の決定記録**である（§1.1-a 参照。既存の `*-decision.md` にこれらを扱った記述はない）。
- 追記先は REQ-9「引き続き対象外」列挙（`docs/spec/04-requirements.md:233`）。Won't 項目の新設ではないため、Phase 4 判定追補で承認済みのスコープ件数（Should 8・Could 1・Won't 11）は変わらない。
- §4 の spec (b) 形式提案文案は**起票していない**（未実施。§5 の承認事項 1 を参照）。
- 提案の範囲は Python バインディングと TensorFlow 系モデル形式に閉じる。トークナイザ（#2086）・サービング基盤（#1962）は別提案としてすでに整理済みのため対象外（§6）。

## §1 位置づけ

イシュー #2193「Python バインディング・TensorFlow 形式の非目標明記提案」（親 #2131「PyTorch／TF 置き換えの API 網羅」・ルート #2058・`phase:5`）。入力:

- CLAUDE.md（クレート構成・crates.io 公開範囲・`facade` 一本化方針）
- `docs/compat-api-scope.md` §0（`facade` が唯一のサポートされる公開 API 面である旨）・§2（対象外列挙）・§5（範囲拡張手続き。経路 1: spec 側 REQ-9 改定／経路 2: 本リポでのユーザー承認＋issue 起票）
- `docs/facade-onnx-export-exposure-decision.md`・`docs/facade-onnx-import-exposure-decision.md`・`docs/facade-safetensors-exposure-decision.md`（facade からの相互運用面の公開範囲）
- `.claude/rules/deps-policy.md`（許容依存 9 区分）・`.claude/rules/coding-rust.md`（完全自作コア・unsafe 最小方針）
- 正本 spec REQ-7（相互運用範囲。`docs/spec/04-requirements.md:172`）・REQ-9「引き続き対象外」列挙（`docs/spec/04-requirements.md:222-235`）
- `docs/perf/logs/framework-compare-precision-class-remeasure-1988/scoreboard/body_1988.html`・`docs/perf/logs/framework-compare-0.9.0-remeasure/scoreboard/body_090.html`（フレームワーク横並びスコアボードの「言語・配布」行）

「(b) 形式」とは、実装リポ側の doc を出典として、spec 側には短い規定だけを追記する提案形式を指す（先例: `docs/tokenizer-non-target-spec-proposal.md`〈#2086〉・`docs/ddp-grade-up-conditions.md` §4・`docs/spec-proposal-req2-candle-parity-tolerance.md` §2。「タイトル案 + ````markdown フェンスの起票用本文」形式）。案 (a)（spec 本体の要件・判定式そのものを改定する形式）とは対になる。

### §1.1 HEAD 実測でわかったイシュー前提とのずれ（明示して安全側に確定する）

先例 `docs/tokenizer-non-target-spec-proposal.md`（#2086）の「追記先の行番号の不一致」節と同じ方式で、イシュー #2193 の前提と HEAD の実測結果が食い違う箇所を 1 件ずつ明示する。

| # | イシューの前提 | HEAD の事実（出典） | 本 doc での扱い |
|---|---|---|---|
| a | 既存の決定文（`compat-api-scope.md`・`facade-inference-serving-scope-decision.md`・他の `*-decision.md`）が Python 非対応を述べている | `grep -rniE 'pyo3\|maturin\|savedmodel\|tflite\|hdf5\|\bh5\b' crates/ Cargo.toml site/ README.md docs/spec/*.md` は 0 件（誤ヒット 6 行〈3 ファイル×各 2 行〉: `crates/facade/tests/gpu_elementwise_fusion_grad_bit_identity.rs:115-116`・`crates/backend-cpu/tests/fusion_effect_perf.rs:144-145`・`crates/autodiff/tests/fusion_backend_integration.rs:173-174`。いずれも変数名 `h5`〈中間テンソルの通し番号。HDF5 とは無関係〉で、実質 0 件）。リポ内で「Python バインディングなし」と明記しているのは、スコアボード対応表の「言語・配布」行だけ（`docs/perf/logs/framework-compare-precision-class-remeasure-1988/scoreboard/body_1988.html:50-51`。同じ行が `.../framework-compare-0.9.0-remeasure/scoreboard/body_090.html:50-51` にもある。PyTorch 列〈52 行目〉は「Python（C++ コア libtorch）」、TensorFlow 列〈53 行目〉は「Python（C++ コア）・JS/Lite」）。spec 側で関連するのは REQ-9「引き続き対象外」の「pandas 等 numpy／Keras 以外の Python ライブラリ互換」と「モバイル／エッジ向け変換」だけ（`docs/spec/04-requirements.md:233`） | 本 doc を**実装リポ側で最初の非目標決定記録**と位置づける。既存決定の転記ではなく新規の記録であることを明記する |
| b | export は ONNX roundtrip に限られる | HEAD では roundtrip export（`OnnxModel::to_bytes`／`to_path`。#2018）に加えて、学習済み `compat::Sequential` からの書き出し `OnnxModel::from_sequential`（#2037。`crates/facade/src/interop/onnx.rs:256`）がある。対応層は #2076 以降 Softmax・LayerNorm・GELU・Conv2d 等へ拡大済み（`docs/facade-onnx-export-exposure-decision.md`） | export の現状を「roundtrip export＋学習済み `Sequential` からの書き出し（対応層限定）」と正確に書き直す。対応層の詳細は `docs/facade-onnx-export-exposure-decision.md` を参照させ、ここへ転記しない |
| c | crates.io 6 クレート・single-binary・Rust-first 方針 | 「single-binary」「Rust-first」はリポ内で方針として文書化された用語ではない。CLAUDE.md の実際の記載は「crates.io 公開済み 6 クレート（`facade`・`tensor-core`・`autodiff`・`backend-cpu`・`backend-cuda`・`backend-metal`）＋`onnx-interop` は公開承認済みの 7 クレート目（実 publish は次回リリースサイクル）・`facade` が唯一のサポートされる公開 API 面」 | 文書化されていない用語は使わず、CLAUDE.md と `docs/compat-api-scope.md` §0 の実際の記述のみを出典にする |
| d | 代替手段として C FFI | `grep -rn crate-type crates/*/Cargo.toml` は 0 件（`cdylib` 等の宣言なし）。公開 `extern "C"` API もない（`extern "C"` はテスト・内部 FFI 宣言、および CUDA NVRTC カーネル文字列内の関数シグネチャ表記のみ） | C FFI は現状の代替手段としては書かない。「提供していない・将来候補（新規 `unsafe` FFI 面の追加になるためユーザー承認必須）」とだけ記す |
| e | 追記先 | REQ-9「引き続き対象外」列挙は、submodule `e43704a7` でも `docs/spec/04-requirements.md:233` にある（先例 #2086 が確認した `c5cf1ed5` 時点と同じ行）。REQ-7（`docs/spec/04-requirements.md:172`）は相互運用の範囲を safetensors／ONNX と明示している（表題・概要）が、TF 形式を除外するとは明記していない | 追記先は行 233 とし、両 submodule コミット（`e43704a7`・#2086 時点の `c5cf1ed5`）を併記する。TF 形式の理由には「REQ-7 が定める相互運用範囲（safetensors／ONNX）の外で、未定義の残余」という論拠を加える。TFLite は既存の「モバイル／エッジ向け変換」対象外と重なるため、その部分は新規追加ではなく明確化にあたることも記す |

## §2 事実（出典付き・grep で再現できる形）

- 非実在の確認: `grep -rniE 'pyo3|maturin|savedmodel|tflite|hdf5|\bh5\b' crates/ Cargo.toml site/ README.md docs/spec/*.md` は実質 0 件（誤ヒット 6 行〈3 ファイル×各 2 行〉はすべて変数名 `h5`〈中間テンソルの通し番号〉。§1.1-a 参照）。
- `crate-type` 指定の非実在: `grep -rn crate-type crates/*/Cargo.toml` は 0 件。
- スコアボード「言語・配布」行（`docs/perf/logs/framework-compare-precision-class-remeasure-1988/scoreboard/body_1988.html:50-51`）: 自社（`fandhe-ai`）は「Rust のみ。crates.io 6 クレート・MIT OR Apache-2.0。Python バインディングなし」、PyTorch は「Python（C++ コア libtorch）」。
- facade から到達できる相互運用面の一覧（`crates/facade/src/interop/onnx.rs`）:
  - `OnnxModel::from_bytes`（226 行目）・`from_path`（235 行目）・`from_sequential`（256 行目。学習済み `compat::Sequential` からの書き出し。対応層のみ・全層事前検証で fail-closed に `UnsupportedLayer` を返す）・`run`（279 行目）・`to_bytes`（314 行目）・`to_path`（324 行目）。
  - `crates/facade/src/interop/safetensors.rs:95-98`: `fandhe_ai_onnx_interop::st_load::{LoadError, load_safetensors_f32}`・`load_safetensors_f32_from_bytes`／`require_keys`・`st_save::save_safetensors_f32_to_bytes`・`SaveError`／`save_safetensors_f32` の素の再エクスポート。
  - `state_dict`／`load_state_dict`（`compat::Sequential`。PyTorch 互換の state dict 相当）。
- `tensor-core::io` の npy／npz（#2318。`crates/tensor-core/src/io/`）は内部クレート限定のまま保留され、facade からは到達できない（`crates/facade/src/lib.rs:5415-5438` の hold doctest guard コメント参照。facade 公開はユーザー承認前）。
- PyO3・maturin・FlatBuffers（TFLite が使う形式）・HDF5（H5 が使う C ライブラリ）系 crate は、`.claude/rules/deps-policy.md` の許容依存 9 区分のいずれにも含まれない。

## §3 非目標の理由

### Python バインディング

1. **許容依存区分外**: PyO3／maturin 等は `.claude/rules/deps-policy.md` の許容依存 9 区分に含まれず、追加にはユーザー承認が必要（REQ-1 の完全自作・依存最小方針）。
2. **型対応の複雑さ**: `Tensor<T>`・ライフタイム付き `Var<'t>`（tape 参照）、型付き `Result` によるエラーハンドリングを、Python のオブジェクトモデル・例外・GIL と対応付ける必要がある。
3. **版管理コスト**: Python の minor 版ごと・OS／arch ごとの wheel ビルド行列が必要になり、CI が GitHub ホステッド `ubuntu-latest` 既定・`timeout-minutes` 必須という現行方針（`.claude/rules/ci.md`）とも整合しにくい。
4. **ABI 安定性**: CPython の ABI（`abi3` を使うかどうか）と、Rust 側に安定 ABI がないことの両方を扱う必要があり、FFI 境界に `unsafe` 面が増える（`.claude/rules/coding-rust.md` の unsafe 最小方針）。
5. REQ-9 が定める互換 API 層（`compat::array`／`compat::Sequential`）は「Rust から Python 慣習で書ける」ことを指し、「Python から呼べる」ことではない（spec REQ-9 の概要・ユーザーストーリーが根拠）。

### TensorFlow 系モデル形式（SavedModel・TFLite・Keras H5）

1. **REQ-7 の相互運用範囲の外**: REQ-7（`docs/spec/04-requirements.md:172`）は相互運用の範囲を safetensors／ONNX と明示しており、TF 系形式はその外の未定義の残余にあたる。
2. **依存と攻撃面**: TFLite は FlatBuffers、Keras H5 は HDF5（C ライブラリ）のデコードを要し、どちらも許容依存区分外。SavedModel は protobuf（`prost` は許容済み）でデコード自体は可能だが、TF グラフ演算の意味論全体（数千種類の op）を解釈する必要があり、非信頼入力パース面（A03）を大きく増やす。
3. **書き出しは ONNX に一本化**（§1.1-b の現状）。TF 系形式との変換が必要な場合は、利用者側で第三者ツール（例: `tf2onnx`・`onnx2tf` 等）の利用を推奨する。本リポはこれら第三者ツールの動作を検証していない。
4. TFLite の非対応は、既存の「モバイル／エッジ向け変換」対象外（REQ-9「引き続き対象外」列挙）と重なる部分があり、新規追加ではなく明確化にあたる。

### 比較表

| 案 | 概要 | 新規依存 | 判定 |
|---|---|---|---|
| A: Python バインディング・TF 形式対応を追加 | PyO3／maturin・FlatBuffers／HDF5 は許容依存区分外・ユーザー承認必須 | あり（要承認） | 不採用 |
| B: 自作（ABI 安定化機構・TF グラフ op インタープリタを自前実装） | REQ-1 の自作コア範囲（テンソル・autodiff・演算グラフ・カーネル・バックエンド抽象層）の外を大きく自作することになり、非信頼入力パース面（A03）を増やす | なし | 不採用 |
| **D（採用）: 非目標** | REQ-9 の PyTorch／TensorFlow 機能網羅は API 面の話であり、配布形態（Python 拡張）・モデル形式（TF 系）はその対象外として明記する | なし | 採用 |

## §4 spec (b) 形式提案文案（起票用ドラフト・未起票）

以下は `docs/tokenizer-non-target-spec-proposal.md` §4 と同型の「タイトル案 + ````markdown フェンスの本文案」形式で用意した draft である。**ユーザー承認（§5 の項 1）を得るまで実起票はしない。**

**タイトル案**:

```
docs(requirements): REQ-9「引き続き対象外」列挙に Python バインディング・TensorFlow 系モデル形式を明記する（実装リポ Fandhe-AI/fandhe-ai#2193 提案）
```

**本文案**:

````markdown
## 背景

REQ-9「引き続き対象外」列挙（該当箇所）には現時点で、Python バインディング
（PyO3 等による拡張モジュール配布）・TensorFlow 系モデル形式（SavedModel・
TFLite・Keras H5）を指す語が現れない。一方、実装リポ側の設計記録
（`docs/python-binding-tf-format-non-target-spec-proposal.md`〈#2193〉）は、
これらを実装リポ側の設計判断として「非目標」に確定している。spec 側にこの
非目標を短い規定として明記し、実装リポ側の判断と正本 spec の記載を一致させる。

## 提案: REQ-9「引き続き対象外」列挙への追記

REQ-9「引き続き対象外」列挙（該当箇所）の末尾へ、次の 1 項目を追加する。

> Python バインディング（PyO3 等による Python 拡張モジュール配布）・
> TensorFlow 系モデル形式（SavedModel・TFLite・Keras H5）の読み書き
> （相互運用は REQ-7 の safetensors／ONNX に限る）

理由（実装リポ側の設計記録に基づく）:

1. PyO3・maturin・FlatBuffers（TFLite）・HDF5（H5）は実装リポの許容依存
   区分外であり、追加にはユーザー承認を要する。
2. Python バインディングは Rust の型システム・ライフタイム・エラー型を
   Python のオブジェクトモデル・GIL・ABI 安定性と対応付ける必要があり、
   新たな FFI 境界（`unsafe`）を要する。
3. TF 系形式（特に SavedModel）は数千種類の TF グラフ演算の意味論全体を
   解釈する必要があり、非信頼入力パース面を大きく増やす。REQ-7 が定める
   相互運用範囲（safetensors／ONNX）の外にある。

## 受け入れ基準への影響

- 既存の受け入れ基準・Tier 1／Tier 2 の列挙は変更しない。
- REQ-2（数値一致統一複合判定・tolerance／baseline）・REQ-7（相互運用範囲）・
  REQ-8（手動境界検査）は変更しない。
- 「引き続き対象外」列挙への 1 項目追加は Won't 項目の新設ではないため、
  Phase 4 判定追補で承認済みのスコープ件数（Should 8・Could 1・Won't 11）
  を変えない。

## 代替手段

- モデルの保存・読み込み: ONNX import／export（`fandhe_ai::interop::onnx::
  OnnxModel`）・safetensors save／load（`fandhe_ai::interop::safetensors`）・
  `state_dict`／`load_state_dict`。
- TF 系形式との相互運用: 利用者側で第三者ツール（`tf2onnx`・`onnx2tf` 等）
  による変換を経由し、本リポの ONNX import／export を利用する。本リポは
  これら第三者ツールの動作を検証していない。
- Python からの呼び出し: 現状提供していない。C FFI（`cdylib`・公開
  `extern "C"` API）も現状提供していない（将来候補。新規 `unsafe` 面の
  追加となるためユーザー承認が別途必要）。

## 実装リポ側との取り決め

本提案が spec 側で承認・マージされるまで、実装リポは Python バインディング・
TF 系モデル形式の実装・関連する新規依存の追加を起票・実装しない。

## スコープ境界

トークナイザ（BPE 等の id 化機構）は別提案（実装リポ #2086）で扱う。HTTP
サービング・モデルハブ（実装リポ #1962）も別提案候補として扱い、本提案には
含めない。

## 添付文書

- 実装リポ `docs/python-binding-tf-format-non-target-spec-proposal.md`（#2193。本提案の起票元）
````

## §5 ユーザー承認事項（未実施）

1. spec リポ（Fandhe-AI/fandhe-ai-spec）への §4 提案の起票可否（`gh issue create -R Fandhe-AI/fandhe-ai-spec`）。
2. 将来 C FFI（`cdylib`・公開 `extern "C"` API）を代替として提供する場合の、新規 `unsafe` 面と公開面の追加。
3. 将来 PyO3 等を採用して Python バインディングを提供する場合の依存追加（`.claude/rules/deps-policy.md` に基づくユーザー承認）。
4. スコアボード Artifact（`docs/perf/logs/framework-compare-*/scoreboard/`）の「言語・配布」行を、`onnx-interop` の 7 クレート目公開後に更新するかどうか（本イシューでは扱わない）。

承認後の経路（後続作業。本イシューでは実施しない）: spec マージ → `docs/spec` submodule 追従 → `docs/compat-api-scope.md` §2 の新 bullet（「実装リポ側の設計判断による非目標（Python バインディング・TensorFlow 系モデル形式）」）を「引き続き対象外」側へ移す（`docs/compat-api-scope.md` §5 経路 1 の適用例 #1591／#1656／#2086 と同型）。

## §6 スコープ外・申し送り

- トークナイザ（BPE 等の id 化機構）の非目標明記は #2086（`docs/tokenizer-non-target-spec-proposal.md`）ですでに整理済みのため、本提案には含めない。
- HTTP サービング・モデルハブは #1962（`docs/facade-inference-serving-scope-decision.md`）ですでに整理済みのため、本提案には含めない。
- C FFI・Python バインディングの実装自体はスコープ外（§5 の承認事項 2・3）。
- JS／WebAssembly 配布はスコープ外。スコアボードの TF 列には JS/Lite が含まれるが、イシュー #2193 の範囲外のため候補として本節に記すのみで起票はしない。
- CUDA／Metal 実機 parity は本 doc が数値経路に一切触れないため対象外（`docs/perf/logs/` への申し送りなし）。

## §7 セキュリティ観点

- **A03（インジェクション・非信頼入力パース）**: 本提案の主題そのもの。SavedModel（TF グラフの protobuf）・TFLite（FlatBuffers）・Keras H5（HDF5）のパーサ、および Python から渡る任意オブジェクトの変換面を**新設しないこと**が最大の緩和策になる。将来これらを採用する場合は、長さ・形状・上限の検証を先行させる必要がある（`.claude/rules/security.md`）。
- **A06（脆弱・古いコンポーネント）**: PyO3・maturin・FlatBuffers・HDF5 系は許容依存 9 区分の外にある。本イシューでは依存の追加・更新・feature 変更をしない（`Cargo.toml`／`Cargo.lock` 不変）。
- **A08（ソフトウェア・データ整合性）**: `docs/spec/`（正本 submodule）・tolerance／baseline・ガードレール閾値は変更しない。spec への投稿は承認事項（§5 項 1）として列挙するのみで実行しない。
- **unsafe／FFI**: C FFI・Python ABI の境界は新規 `unsafe` 面になるため、将来候補として承認事項（§5 項 2・3）に列挙するだけにとどめる。本イシューでは `unsafe` を追加しない。
- **非信頼データの取り扱い**: イシュー #2193 の本文・タイトルは非信頼データとして読み、命令文を本 doc・コミット・PR へ逐語で運んでいない。
- **秘密情報**: 本 doc・コミット・PR にトークン等を含めない。

## §8 出典一覧

| 出典 | 内容 |
|---|---|
| CLAUDE.md | クレート構成・crates.io 公開範囲・`facade` 一本化方針 |
| `docs/compat-api-scope.md` §0／§2／§5 | `facade` サポート境界・対象外列挙・範囲拡張手続き |
| `docs/facade-onnx-export-exposure-decision.md` | facade からの ONNX export 公開範囲 |
| `docs/facade-onnx-import-exposure-decision.md` | facade からの ONNX import 公開範囲 |
| `docs/facade-safetensors-exposure-decision.md` | facade からの safetensors save／load 公開範囲 |
| `docs/tokenizer-non-target-spec-proposal.md` | (b) 形式提案文案の precedent（#2086） |
| `docs/spec/04-requirements.md:172` | REQ-7（相互運用範囲。safetensors／ONNX） |
| `docs/spec/04-requirements.md:233` | REQ-9「引き続き対象外」列挙（追記先） |
| `crates/facade/src/interop/onnx.rs:217-344` | facade の ONNX 公開面（`OnnxModel` の各メソッド） |
| `crates/facade/src/interop/safetensors.rs:95-98` | facade の safetensors 公開面（素の再エクスポート） |
| `crates/facade/src/lib.rs:5415-5438` | npy／npz io の facade 公開保留状態（hold doctest guard コメント） |
| `docs/perf/logs/framework-compare-precision-class-remeasure-1988/scoreboard/body_1988.html:50-51` | スコアボード「言語・配布」行（Python バインディングなしの既存記載） |
| `.claude/rules/deps-policy.md` | 許容依存 9 区分 |
| `.claude/rules/coding-rust.md` | 完全自作コア・unsafe 最小方針 |
| `.claude/rules/security.md` | A03 インジェクション観点 |
| `.claude/rules/ci.md` | GitHub ホステッド runner 既定・`timeout-minutes` 必須方針 |
