# Python バインディング・TensorFlow 系モデル形式の非目標の spec 明記提案（(b) 形式提案文案・実装しない）（#2193・#2623 最新化）

基準コミット: `f3caacd71b002cc37a42f81fe9c1dc88aa34d8a4`（#2623 最新化時点の main）。`docs/spec` submodule ポインタ `2e998dd77117814f4af8ed160394ad1d6a8f888a`。`file_path:line` は同コミット時点のもの。後続の変更で行番号がずれる可能性があるため、参照する際は当該コミット、または近傍のコミットで再確認すること。初版（#2193）の基準は `9fe4b5231ad53883e4aff072e44a589ac548baaf`・submodule `e43704a7baefd1489d3f1716571064ab65c5eed6`。

## §0 結論（最初に読む）

- **コード変更なし**（`crates/**`・`Cargo.toml`／`Cargo.lock`・`deny.toml`・`docs/spec/`〈正本 submodule〉・tolerance／baseline・ガードレール閾値は一切変更していない）。
- **#2193 の提案はすでに spec へ反映済み**。`docs/spec/04-requirements.md:234` に「2026-09-29 追記・#2193」として REQ-9「引き続き対象外」列挙へ Python バインディング・TF 系モデル形式が追記され、変更履歴 `:435` に「PR #2321、2026-09-29 ユーザー承認」とある。したがって §4 の文案は履歴であり、起票待ちではない。
- **#2623 の推奨: 非目標を維持する（案 D）**。対象内化した場合の依存・`unsafe`・攻撃面の見積りは §3b に並べた。到達目標「PyTorch／TF を Rust で置き換える」は Rust から使えることで満たされ、Python から呼べることを必要条件としない。TF 系形式は ONNX（`tf2onnx`／`onnx2tf` 経由）で代替できる。
- 実依存の追加・`cargo tree` 実測・C FFI／PyO3 の実装は行わない。未決の判断は §5 に承認事項として残し、#2623 のコメントで承認を依頼する。
- 提案の範囲は Python バインディングと TensorFlow 系モデル形式に閉じる。トークナイザ（#2086・最新化 #2618）・サービング（#2624）・モデルハブは別 doc／別 issue で扱う（§6）。

### #2623 最新化の差分要約

| 旧記述（#2193 時点） | HEAD の事実 | 対応 |
|---|---|---|
| §4 の提案は「未起票」 | spec :234・:435 に反映済み | §0・§4・§5 を「反映済み」へ更新 |
| 許容依存 9 区分 | 10 区分（第 10 区分 `libc`。本体の直接依存は第 1〜8・第 10 の 9 区分） | 区分数の表記を更新 |
| crates.io 6 クレート＋`onnx-interop` は公開承認済み | 7 クレート公開済み（0.10.0 で `onnx-interop` 初公開。2026-10-03） | §1.1-c を更新 |
| スコアボード出典 `body_1988.html`・`body_090.html` | 正は 0.10.0 の `body_0100.html`（言語・配布 :51、相互運用 :106-109、推論 :123-125） | 出典を差し替え |
| facade の ONNX 公開面の行番号 | `from_path_with_limits`（:298）が増え、行番号がずれた | §2 を更新 |
| 非公開の内部 FFI の言及なし | `libc` を使う `external_data.rs` が加わった。公開 C API・`crate-type` は 0 件のまま | §1.1-d・§2 を更新 |
| §6 のモデルハブ参照「#1962 で整理済み」 | 誤り（`hf-hub-integration-design.md` §6.2 の指摘どおり） | 正しい参照先へ修正 |
| 見積り表なし | §3b を新設 | 受け入れ条件の中心 |

## §1 位置づけ

イシュー #2193「Python バインディング・TensorFlow 形式の非目標明記提案」（親 #2131「PyTorch／TF 置き換えの API 網羅」・ルート #2058・`phase:5`）。#2623 はその最新化と承認依頼（親 #2606・ルート #2499・`phase:5`）。入力:

- CLAUDE.md（クレート構成・crates.io 公開範囲・`facade` 一本化方針）
- `docs/compat-api-scope.md` §0（`facade` が唯一のサポートされる公開 API 面である旨）・§2（対象外列挙）・§5（範囲拡張手続き。経路 1: spec 側 REQ-9 改定／経路 2: 本リポでのユーザー承認＋issue 起票）
- `docs/facade-onnx-export-exposure-decision.md`・`docs/facade-onnx-import-exposure-decision.md`・`docs/facade-safetensors-exposure-decision.md`（facade からの相互運用面の公開範囲）
- `.claude/rules/deps-policy.md`（許容依存 10 区分）・`.claude/rules/coding-rust.md`（完全自作コア・unsafe 最小方針）
- 正本 spec REQ-7（相互運用範囲。`docs/spec/04-requirements.md:172`）・REQ-9「引き続き対象外」列挙と追記（`docs/spec/04-requirements.md:233-235`。:234 が本提案の反映、:235 が #2194 の反映）
- `docs/perf/logs/framework-compare-0.10.0-remeasure/scoreboard/body_0100.html`（フレームワーク横並びスコアボードの「言語・配布」「相互運用」「推論」行）

「(b) 形式」とは、実装リポ側の doc を出典として、spec 側には短い規定だけを追記する提案形式を指す（先例: `docs/tokenizer-non-target-spec-proposal.md`〈#2086〉・`docs/ddp-grade-up-conditions.md` §4・`docs/spec-proposal-req2-candle-parity-tolerance.md` §2）。案 (a)（spec 本体の要件・判定式そのものを改定する形式）とは対になる。

### §1.1 HEAD 実測でわかったイシュー前提とのずれ（明示して安全側に確定する）

| # | 前提 | HEAD の事実（出典） | 本 doc での扱い |
|---|---|---|---|
| a | 既存の決定文が Python 非対応を述べている | リポ内で「Python バインディングなし」と明記しているのはスコアボード対応表の「言語・配布」行（`body_0100.html:51`。PyTorch 列は :52「Python（C++ コア libtorch）」、TensorFlow 列は :53「Python（C++ コア）・JS/Lite」）と、spec :234 の反映だけ | 実装リポ側で最初の非目標決定記録と位置づける（初版から不変） |
| b | export は ONNX roundtrip に限られる | roundtrip export（`to_bytes`／`to_path`）に加え学習済み `compat::Sequential` からの書き出し `OnnxModel::from_sequential`（`crates/facade/src/interop/onnx.rs:333`）がある。対応層は `docs/facade-onnx-export-exposure-decision.md` を正とする（スコアボード :107 は「Sequential 6 層」） | 層名は転記せず参照させる |
| c | crates.io 6 クレート＋`onnx-interop` は次回公開 | 7 クレート公開済み（0.10.0 で `onnx-interop` 初公開。2026-10-03。CLAUDE.md・`body_0100.html:51`） | 現状に合わせる。「single-binary」「Rust-first」は文書化された用語ではないため使わない |
| d | 代替手段として C FFI | `grep -rn crate-type crates/*/Cargo.toml` は 0 件。公開 `extern "C"` API もない。内部（非公開）の `unsafe extern "C"` はバックエンド・`crates/docs-site/src/build.rs:279`・`crates/self-repair/src/fd_walk.rs:169` 等にあり、`libc` を使う `crates/onnx-interop/src/onnx/external_data.rs` も加わったが、いずれも公開 API ではない | C FFI は「提供していない・将来候補（新規 `unsafe` 面。ユーザー承認必須）」とだけ記す |
| e | 追記先は REQ-9 の行 233 | 本提案は spec :234 の独立した追記として反映済み（変更履歴 :435） | 追記先は反映済み。`compat-api-scope.md` §2 の移設は未了（§5 の承認事項 b） |
| f | 提案は未起票 | spec :234・:435 に反映済み（PR #2321・2026-09-29） | §0・§4・§5 を更新 |

## §2 事実（出典付き・grep で再現できる形）

- 非実在の確認: `grep -rniE 'pyo3|maturin|savedmodel|tflite|hdf5|\bh5\b' crates/ Cargo.toml site/ README.md docs/spec/*.md`（#2623 時点）のヒットは次の 8 行だけで、いずれもコード・依存の存在を示さない。
  - `crates/facade/tests/cpu_predict_resident_fixedcost_diag.rs:420`: コメント中の仮説記号「H5」。
  - `crates/facade/tests/gpu_elementwise_fusion_grad_bit_identity.rs:115-116`・`crates/backend-cpu/tests/fusion_effect_perf.rs:144-145`・`crates/autodiff/tests/fusion_backend_integration.rs:173-174`: 変数名 `h5`（中間テンソルの通し番号）。
  - `docs/spec/04-requirements.md:234`・`:435`: 本提案の spec 反映の記述そのもの。
- `crate-type` 指定の非実在: `grep -rn crate-type crates/*/Cargo.toml` は 0 件。
- スコアボード（`body_0100.html`）: 「言語・配布」:51（自社は「Rust のみ（Python バインディングなし）。crates.io 7 クレート…」）、「相互運用」:106-109（自社:107「ONNX import〈36 op・推論専用〉・export〈import 済みモデルの往復と Sequential 6 層〉・safetensors 保存／読込〈F32〉・save_model／load_model。npy／npz は内部クレートのみで未公開」、TF:109「SavedModel・TFLite・ONNX（tf2onnx）」）、「推論」:123-125。
- facade から到達できる相互運用面（`crates/facade/src/interop/onnx.rs`）: `OnnxModel::from_bytes`（237 行目）・`from_path`（282）・`from_path_with_limits`（298。#2623 時点で新規）・`from_sequential`（333。対応層のみ・全層事前検証で fail-closed に `UnsupportedLayer`）・`run`（363）・`to_bytes`（402）・`to_path`（412）。
- `crates/facade/src/interop/safetensors.rs:95-98`: `fandhe_ai_onnx_interop::st_load::{LoadError, load_safetensors_f32}`・`load_safetensors_f32_from_bytes`／`require_keys`・`st_save::save_safetensors_f32_to_bytes`・`SaveError`／`save_safetensors_f32` の素の再エクスポート（変更なし）。
- `state_dict`／`load_state_dict`（`compat::Sequential`）。
- `tensor-core::io` の npy／npz は内部クレート限定のまま保留（`crates/facade/src/lib.rs:5206-5278` 付近の hold doctest guard。`save_model`／`load_model` の保留ガードは :5313 以降）。facade 公開はユーザー承認前。
- PyO3・maturin・FlatBuffers（TFLite が使う形式）・HDF5（H5 が使う C ライブラリ）系 crate は、`.claude/rules/deps-policy.md` の許容依存 10 区分（本体の直接依存は第 1〜8・第 10 の 9 区分）のいずれにも含まれない。

## §3 非目標の理由

### Python バインディング

1. **許容依存区分外**: PyO3／maturin 等は `.claude/rules/deps-policy.md` の許容依存 10 区分に含まれず、追加にはユーザー承認が必要（REQ-1 の完全自作・依存最小方針）。
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

## §3b 対象内化した場合の見積りと推奨（#2623）

「依存」「`unsafe`」「攻撃面」を、対象内化した場合と非目標維持で並べる。**依存の追加はできないため `cargo tree` は実測していない**。ライセンスは crates.io の公開メタデータ（2026-10-03 取得。`/api/v1/crates/<name>/versions` の `license` 欄）だけを出典とし、推移的依存・feature 別の allow 適合・脆弱性履歴は「未実測」とする（`docs/license-matrix.md` §1・`docs/model-download-design.md` §3 と同方針）。`deny.toml` の allow は `MIT`・`Apache-2.0`・`Apache-2.0 WITH LLVM-exception`・`ISC`・`Zlib`・`Unicode-3.0`・`Unlicense`・`BSD-2-Clause`（`deny.toml` の `[licenses].allow` で再確認すること）。

### 対象内化した場合

| 項目 | 新規依存（deps-policy 区分外） | 新規 `unsafe`／FFI 面 | 攻撃面（A03 非信頼入力） | CI・配布への影響 | 未実測事項と確認方法 |
|---|---|---|---|---|---|
| Python バインディング | `pyo3`（0.29.3。MIT OR Apache-2.0）が区分外。`maturin` は Cargo 依存ではなくビルドツール（配布工程の追加） | 大。`Var<'t>` の借用ライフタイムは Python へ渡せないため、所有型ハンドルへの公開 API 再設計が要り、facade のサポート面が 2 系統になる（`compat-api-scope.md` §0 との整合）。numpy ゼロコピー・バッファプロトコルは `unsafe` を要し、GIL 解放時は `Send` 制約が付く | 中。Python 側から任意オブジェクト・バッファが渡る変換面が新設される | CPython の版 × OS × arch の wheel 行列。GitHub ホステッド `ubuntu-latest` 既定・`timeout-minutes` 必須の方針（`ci.md`）との整合が論点。`abi3` を使うかで行列の大きさが変わる | 推移的依存数・allow 適合は未実測。承認後に feature 組合せ別の `cargo tree` 実測が必要 |
| SavedModel | `prost` は許容済み。ただし `saved_model`・`graph`・`op_def` 等の proto の手書き derive が要る（`prost-build` は不可） | 小（デコード自体は安全コードで可能） | 大。TF グラフ op（数千種）の意味論の解釈、variables（checkpoint 形式）の読み取りが要り、非信頼入力パース面が最も大きい | 影響小 | 必要な proto 範囲・対応 op 数は未実測。ONNX import の 36 op（スコアボード :107）との差を見積る必要がある |
| TFLite | `flatbuffers`（25.12.19。Apache-2.0。直接ライセンスは allow 内だが区分外）を使うか、オフセット検証付きの自前パーサを書く | 中。自前パーサなら境界検査の自作が要る。`flatbuffers` crate 側の `unsafe` 使用範囲は未確認 | 大。FlatBuffers のオフセット・長さの検証漏れが A03 に直結する | 影響小 | 推移的依存・`unsafe` の使用範囲は未実測。モバイル／エッジ向け変換は既存の対象外（REQ-9）と重なる |
| Keras H5 | HDF5 系 crate（例: `hdf5-metno` 0.15.0。MIT OR Apache-2.0）が区分外。HDF5 の C ライブラリ（libhdf5）へのリンクが要る見込み（未実測） | 大。C ライブラリへの FFI | 大。HDF5 の C パーサを非信頼入力へ露出する。C ライブラリの脆弱性履歴は RustSec／NVD で確認していないため「未確認」 | C ツールチェーンや外部ライブラリが要る場合、「CUDA toolkit 非搭載でもビルドが成立する」方針（cfg ベース・動的ロード）と衝突しうる。ビルド行列が増える | libhdf5 のライセンス・静的／動的リンクの可否・`deny.toml` への適合は未実測 |
| （参考）C FFI（`cdylib`・公開 `extern "C"`） | 追加依存なし | 大。公開 ABI を新設する | 中。呼び出し側の誤用を前提にした入力検証が要る | 配布物（`.so`／`.dll`／`.dylib`）の追加 | 提供範囲が未定義。承認事項 c |

### 非目標を維持する場合

- 新規依存 0・新規 `unsafe` 0・攻撃面は増えない。本体の許容依存 10 区分と `facade` 一本化方針（`compat-api-scope.md` §0）を保てる。
- 代替経路:
  - TF 系 → 利用者側で第三者ツール（`tf2onnx`／`onnx2tf` 等）により ONNX へ変換 → facade の ONNX import。ONNX import は 36 op・推論専用という制約がある（`body_0100.html:107`）。第三者ツールの動作は本リポでは検証していない。
  - safetensors 保存／読込・`save_model`／`load_model`・`state_dict`／`load_state_dict`。
  - Python からの利用は提供しない。

### 推奨: 非目標を維持する（案 D）

1. すでに spec :234 に反映済みで、承認済みの方針と一致する。
2. 到達目標「PyTorch／TF を Rust で置き換える」とは独立している。置き換え先の利用者は Rust から使う前提で、Python から呼べることは必要条件ではない。
3. いずれの対象内化も許容依存区分外の追加を伴い、非信頼入力パース面（A03）か FFI の `unsafe` 面が増える。

### 再評価の発火条件（日付ではなく条件で書く）

- 許容依存への該当 crate の追加について、ユーザー承認が得られたとき。
- ONNX 経路（`tf2onnx` 等）で賄えない移行需要が、具体的なモデル・形式で示されたとき。
- facade が所有型ハンドルを前提とする公開 API の再設計（Python 以外の理由を含む）に着手するとき。

## §4 spec (b) 形式提案文案（2026-09-29 spec 反映済み・履歴として保持）

以下は `docs/tokenizer-non-target-spec-proposal.md` §4 と同型の「タイトル案 + ````markdown フェンスの本文案」形式で用意した draft である。**#2623 最新化時点ですでに spec 側へ反映済み**（`docs/spec/04-requirements.md:234`・変更履歴 `:435`。PR #2321・2026-09-29 ユーザー承認）。本節の文案は履歴として保持し、実際の追記文言は spec 本文を正とする（spec 側は §4 の草案より詳細で、REQ-1／REQ-7 への言及・Phase 4 判定追補との関係を含む）。

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

## §5 ユーザー承認事項（#2623 で更新）

完了・解消済み（取り消し）:

- ~~1. spec リポへの §4 提案の起票可否~~ → 反映済み（spec :234・:435。PR #2321・2026-09-29 ユーザー承認）。
- ~~4. スコアボードの「言語・配布」行を `onnx-interop` の 7 クレート目公開後に更新するか~~ → 更新済み（`body_0100.html:51` が 7 クレート表記）。

未決（#2623 のコメントで承認を依頼する）:

- (a) 0.10.0 以降も Python バインディング・TF 系形式を非目標として維持する方針の確認（§3b の推奨）。
- (b) `docs/compat-api-scope.md:347-360` の bullet が古い（「許容依存 9 区分」「spec 提案候補（未起票）」のまま）。これを「引き続き対象外」側へ移して現状に合わせる後続作業（§5 経路 1。編集は本 issue では行わない）を起票するかどうか。起票にもユーザー承認が要る（`out-of-scope-tracking.md`）。
- (c) 将来 C FFI（`cdylib`・公開 `extern "C"` API）を提供する場合の、新規 `unsafe` 面と公開面の追加。
- (d) 将来 PyO3 等を採用して Python バインディングを提供する場合の依存追加（`.claude/rules/deps-policy.md` に基づくユーザー承認）。§3b の未実測項目の実測は承認後に行う。

承認後の経路（後続作業。本イシューでは実施しない）: (b) が承認された場合は `docs/compat-api-scope.md` §2 の bullet 修正（§5 経路 1 の適用例 #1591／#1656／#2086 と同型）。spec の承認・マージ・submodule 追従は完了している。

## §6 スコープ外・申し送り

- トークナイザ（BPE 等の id 化機構）の非目標明記は #2086（`docs/tokenizer-non-target-spec-proposal.md`）で整理済み。最新化は兄弟 issue #2618 で扱う。
- サービング基盤・汎用 forward-mode AD・モデルハブは `docs/functorch-serving-hub-non-target-spec-proposal.md`（#2194。spec :235 に反映済み）と `docs/model-distribution-design.md` で整理済み。サービングの最新化は #2624 で扱う。初版は「モデルハブは #1962 で整理済み」と書いていたが誤りで（`docs/hf-hub-integration-design.md` §6.2 の指摘）、#2623 で参照先を修正した。
- C FFI・Python バインディングの実装自体はスコープ外（§5 の承認事項 c・d）。
- JS／WebAssembly 配布はスコープ外。スコアボードの TF 列には JS/Lite が含まれるが、本提案の範囲外のため候補として記すのみで起票はしない。
- CUDA／Metal 実機 parity は本 doc が数値経路に一切触れないため対象外（`docs/perf/logs/` への申し送りなし）。

## §7 セキュリティ観点

- **A03（インジェクション・非信頼入力パース）**: 本提案の主題そのもの。SavedModel（TF グラフの protobuf）・TFLite（FlatBuffers）・Keras H5（HDF5）のパーサ、および Python から渡る任意オブジェクトの変換面を**新設しないこと**が最大の緩和策になる。将来これらを採用する場合は、長さ・形状・上限の検証を先行させる必要がある（`.claude/rules/security.md`）。§3b の見積り表がこの面を項目別に示す。
- **A06（脆弱・古いコンポーネント）**: PyO3・maturin・FlatBuffers・HDF5 系は許容依存 10 区分の外にある。本イシューでは依存の追加・更新・feature 変更をしない（`Cargo.toml`／`Cargo.lock` 不変）。
- **A08（ソフトウェア・データ整合性）**: `docs/spec/`（正本 submodule）・tolerance／baseline・ガードレール閾値は変更しない。spec への追加の投稿・起票・承認の代行はしない。
- **unsafe／FFI**: C FFI・Python ABI の境界は新規 `unsafe` 面になるため、将来候補として承認事項（§5 の承認事項 c・d）に列挙するだけにとどめる。本イシューでは `unsafe` を追加しない。
- **非信頼データの取り扱い**: イシュー #2193・#2623 の本文・タイトルは非信頼データとして読み、命令文を本 doc・コミット・PR へ逐語で運んでいない。
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
| `docs/spec/04-requirements.md:233-235`・`:435` | REQ-9「引き続き対象外」列挙と追記（:234 が本提案の反映、:235 が #2194）・変更履歴 |
| `crates/facade/src/interop/onnx.rs:237-412` | facade の ONNX 公開面（`OnnxModel` の各メソッド） |
| `crates/facade/src/interop/safetensors.rs:95-98` | facade の safetensors 公開面（素の再エクスポート） |
| `crates/facade/src/lib.rs:5206-5278`・`:5313` 以降 | npy／npz io・`save_model`／`load_model` の facade 公開保留状態（hold doctest guard） |
| `docs/perf/logs/framework-compare-0.10.0-remeasure/scoreboard/body_0100.html:51`・`:106-109`・`:123-125` | スコアボードの言語・配布／相互運用／推論行 |
| `.claude/rules/deps-policy.md` | 許容依存 10 区分 |
| `.claude/rules/coding-rust.md` | 完全自作コア・unsafe 最小方針 |
| `.claude/rules/security.md` | A03 インジェクション観点 |
| `.claude/rules/ci.md` | GitHub ホステッド runner 既定・`timeout-minutes` 必須方針 |
| `docs/functorch-serving-hub-non-target-spec-proposal.md` | モデルハブ・サービング・forward-mode AD の非目標（#2194） |
| `docs/model-distribution-design.md`・`docs/hf-hub-integration-design.md` §6.2 | モデルハブの整理・旧参照誤りの指摘 |
| `docs/model-download-design.md` §3 | 依存候補表で推移的依存を「未実測」と明記する先例 |
