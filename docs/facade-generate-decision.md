# `generate()` 自己回帰ループの設計確定と facade 公開保留（イシュー #2191）

対応イシュー: #2191（親 #2084 と同系列の LLM 推論機能群。KV キャッシュ
実装 #2084 の兄弟機能）。
位置づけ: 実装（`crates/autodiff/src/generate/mod.rs`）は完了済み。本 doc は
その設計判断の記録と、facade（`fandhe_ai`）公開保留の多層固定・承認依頼
用の事前設計を記す。tolerance／baseline／`Cargo.toml` 依存／ガードレール
閾値／`docs/spec/`（正本）は一切変更しない。

> **更新（イシュー #2575・#2576）**: 本書 §0〜§15 は #2191 時点の判断と、承認前の停止記録（履歴）である。facade 公開は #2575 で承認形どおり実施済みで、保留ガードも削除・反転済み。現行の公開形は §17、#2576 で仕上げた正ガードは §18 を正とする。

## 0. 結論

自己回帰生成ループ（3 戦略・KV キャッシュ結線・seed 決定性）は `fandhe_ai_
autodiff::generate` モジュールとして実装済みで、受入条件（イシュー本文の
5 項目）をすべて満たす。**facade 公開は未承認のため保留**する（イシュー
#2191 の「承認事項」節が明示的に `pub fn generate` 等の署名を承認事項と
挙げているため、実装 PR 単独では実施しない）。KV キャッシュ（#2084）の
K-1（autodiff 内部実装）／K-2（facade 公開）2 段構成と同型の判断である。

## 1. 背景

- イシュー #2191 は「facade `pub fn generate` 等の署名・エラー型・
  `GenerateConfig` 型」を承認事項として明記しており、実装と同一 PR で
  ユーザー承認を取得することを前提としていない（`.claude/rules/
  delegation-impl.md`「実装 Agent に依存クレートを自己判断で追加させ
  ない」と同種の「実装 Agent が単独で公開面を確定させない」制約）。
- イシュー本文が挙げる配置（`crates/facade/src/inference/generate.rs`）
  ではなく `crates/autodiff/src/generate/mod.rs`（内部クレート）に実装した。
  理由: facade 公開面自体が未承認のため、facade 側に実体を置くと
  「実装は private だが同一クレート内に存在する」という中間状態になり、
  `crates/facade/tests/api_surface.rs` の「facade 到達可能性」検査群
  （`visit_rs_files` が facade `src/` 全体を走査する設計）との整合が
  取りにくい。#2084 の K-1（`crates/autodiff/src/nn/attention.rs` の
  `KvCache`・`MultiheadAttentionVars::forward_with_cache`）と同じく、
  内部クレートに実装してから facade 到達経路を承認事項として追って
  開けるほうが、否定ガードの設計（§8）を素直に保てる。
- KV キャッシュ（#2084）との結線はイシュー本文の受入条件 3
  「KV キャッシュ（#2084）との統合で逐行生成の高速化」に対応する。
  `docs/kv-cache-design.md` §2 の mask 規則 (a) prefill・(b) decode に
  そのまま帰着させる設計にした（§2 参照）。

## 2. API 確定（`crates/autodiff/src/generate/mod.rs`）

- `SamplingStrategy`（`#[non_exhaustive]` enum）: `Greedy`／`TopK(usize)`／
  `Temperature(f32)`。将来 nucleus（top-p）等を追加しても非破壊にする
  ため `#[non_exhaustive]` を付けた（`.claude/rules/security.md` 方針・
  他の `AutodiffError` 系列挙型と同型）。
- `GenerateConfig`（`#[non_exhaustive]` struct）: `max_length`・
  `temperature`・`top_k`・`strategy`・`seed`（既定 `0`）の 5 フィールドを
  イシュー受入条件 1 のとおり字義どおり公開フィールドとして持つ。
  `temperature`／`top_k` は `strategy` から `GenerateConfig::new` が
  一意に導出し、`validate` が矛盾状態（例: `Greedy` に非 `1.0` の
  `temperature` を後から代入）を fail-closed で拒否する（`.claude/
  rules/security.md` A03 方針）。GAT・private フィールド＋getter 方式は
  採らず、受入条件が要求する「型」を素直に公開フィールドとして持つ形に
  した。facade 公開時にこの設計（公開フィールド vs. getter）自体も
  承認事項に含める（§8）。
- `AutoregressiveModel` trait: `&self`（不変借用）で `forward_step` を
  呼び、`&mut [KvCache]` を呼び出し側（`generate`）が確保・貸与する
  ことで、`Module::forward`（`&self`・状態なし）とも `StatefulAttention::
  forward`（`&mut self`）とも異なる「モデル自体は不変・KV キャッシュ
  だけを外部から借用する」設計にした（`docs/kv-cache-design.md` §2 の
  mask 規則にそのまま帰着し、`forward_step` の実装は 1 層ごとに
  `forward_with_cache` を呼ぶだけでよい）。戻り値をホスト `Tensor<f32>`
  （`Tape` を介さない）にしたのは、(1) facade 越しのテストからは
  `fandhe_ai::Tape` の `pub(crate)` 実体へ到達できず `&mut Tape` を
  要求する署名では facade 横断 parity テストが書けないため、(2)
  `docs/kv-cache-design.md` §3.2 が推奨する「step ごとに `Tape` を
  再作成または `reset` する」運用を trait 実装側の責務として閉じ込め
  られるため。`generate` 自体は `Tape` を一切持たない無状態関数。
- サンプリングはホスト `f64` で行う（`softmax_weights_f64`。バックエンド
  非依存で決定的になり、CPU／CUDA／Metal のどの `forward_step`
  実装でも同じ token 列が得られる）。乱数は `fandhe_ai_tensor_core::
  rng::Generator`（独立インスタンス。`config.seed` から生成）のみを
  使い、グローバル RNG（`fandhe_ai_autodiff::manual_seed` が触れる
  状態）は一切消費しない（`generate_does_not_consume_global_rng` で
  固定）。`Generator` は xorshift64* であり暗号学的に安全な PRNG では
  ない（`rng.rs` と同じ注記。OWASP A02。生成した token をセキュリティ
  用途に使わないこと）。
- HuggingFace `generate()` との既知の差分（イシュー本文のスコープ外
  節・トークナイザ非目標を踏まえた設計判断）: EOS 早期停止・
  `pad_token`・repetition penalty は対象外（常に `max_length` まで
  生成）。`top_k > vocab_size` は HF が黙って clamp するのに対し、
  本実装は `AutodiffError::InvalidArgument` で拒否する（fail-closed。
  `.claude/rules/security.md` A03 方針。`validate_top_k_le_vocab`）。

**実装記録（イシュー #2575・#2576）**: 本節の形（pub フィールド 5 つ・`#[non_exhaustive]`・`new`／`with_temperature`／`with_seed`・`validate` 非公開）は、そのまま facade 公開形になった（`fandhe_ai::inference::{AutoregressiveModel, GenerateConfig, SamplingStrategy, generate}` の純再エクスポート 1 文。§17）。#2576 の `generate_public_shape_matches_approved_inventory` がこの形を全数で固定する（§18.2）。

## 3. セキュリティ考慮（OWASP Top 10）

- **A03（インジェクション／不正入力）**: `GenerateConfig::validate`が
  `temperature` の有限性・正値、`strategy` とフィールドの整合、`TopK`
  の `k >= 1` を fail-closed で拒否する。`validate_top_k_le_vocab` は
  prefill で確定した `vocab_size` に対する `k` の上限を拒否する。
  `sample_step` はサンプリングに使う末尾位置（`L_new - 1`。prefill
  では prompt 末尾、decode では新規 1 トークン自身）の logits の
  非有限値（`NaN`／`inf`）混入を明示的に検出し `Err` を返す（モデル
  実装のバグがサンプリング側に伝播しないようにする。サンプリングに
  使わない他位置の logits は検査対象外——codex-review 指摘・
  PR #2324 で契約を明確化）。
- **A04（安全でない設計）**: 出力バッファの確保前に `b.checked_mul
  (config.max_length)` で `usize` オーバーフローを検出し、さらに
  積（要素数）の `i32` 換算バイト数が `Vec` allocation 上限
  （`isize::MAX` バイト）に収まるかも検証する（`checked_mul` は
  `usize` オーバーフローしか検出せず、`b == 1`・短い prompt・
  `max_length == usize::MAX` のような入力は素通りするため、この
  バイト数検証を追加しないと後続の `Vec::with_capacity` が capacity
  overflow で panic しうる。`.claude/rules/security.md` A04 方針・
  codex-review 指摘・PR #2324 是正）。`validate_forward_step_
  output` は `AutoregressiveModel::forward_step` の戻り shape を毎
  ステップ検査し、モデル実装のバグで shape がステップ間で変化しても
  `generate` が誤ったオフセットで host メモリを読まないようにする
  （fail-closed）。
- **A02（暗号の失敗）**: 上記のとおり `Generator` は非暗号学的 PRNG
  である注記をモジュール doc に明記済み。
- 新規 `unsafe` コードは追加していない。依存クレートの追加もない
  （`fandhe_ai_tensor_core::rng`・`Tensor` の既存 API のみを使用）。

## 4. 契約整理（不変）

- tolerance（REQ-2 統一複合判定）・baseline・`Cargo.toml` 依存・
  ガードレール閾値・`docs/spec/`（正本 submodule）は一切変更していない。
- crates.io 出荷済み `fandhe-ai =0.9.0` の公開 API は変更していない
  （facade 公開自体を保留しているため、既存メソッドのシグネチャ・
  意味論は無変更）。

## 5. テスト・実測

- `crates/autodiff/tests/nn_generate.rs`（18 テスト・全 pass）: 3 戦略の
  動作・KV キャッシュ結線（prefill → decode の逐次生成がキャッシュなし
  全長再計算と bit 完全一致すること。`generate_greedy_matches_full_
  recompute_without_cache`）・token id → logits → token id のループ
  動作・seed 固定時の決定性（`generate_same_seed_produces_bit_
  identical_output_for_top_k_and_temperature`）・グローバル RNG 非消費
  （`generate_does_not_consume_global_rng`）・各種エラー経路（空
  prompt／空 batch／rank 不正／`max_length < prompt_len`／オーバー
  フロー／`top_k` 不正／モデル shape drift／非有限 logits）を検証。
- `crates/facade/tests/generate_backend_parity.rs`（3 テスト。CPU 2 件
  pass・CUDA 1 件は `#[ignore]`）: facade 経由の `CpuBackendOps`（本番
  ops）上で `NaiveOps` との REQ-2 統一複合判定突合を行い、greedy・
  top-k（同一 seed）で一致を確認。CUDA 実機（DGX Spark GB10）は
  未実測——申し送りは `docs/perf/logs/generate-2191/README.md`
  （実測時に作成する。本 PR 時点では CUDA/Metal 実機にアクセスできない
  ため未作成。実機作業を引き継ぐセッションが作成すること）。
- Metal 実機も同様に未実測（同上の申し送り先を使う）。

## 6. facade 公開の保留固定と承認依頼用の事前設計

`docs/kv-cache-design.md` §10（KV キャッシュ facade 公開＝K-2 の保留
固定）と同型の多層防御を、`generate()` に対しても構築した。以下 §7・
§8 が、`crates/autodiff/src/generate/mod.rs`・`crates/facade/src/lib.rs`
の doc コメントが参照する節番号である。

## 7. workspace 全体の名前インベントリを採らない理由

`crates/self-repair` に `fn generate` という名前を持つトレイト
メソッドが複数存在するため、facade 以外のクレート内部の private 宣言
まで含めて `fn generate` の出現を workspace 全体で固定する検査
（`workspace_declares_*` パターン）は採用しない。これは KV キャッシュ
（#2084 §10.1）が codex-review 指摘（無関係な内部宣言まで固定して
しまい、正当な変更を不当に fail させる）を受けて撤回した経緯と同じ
理由である。`facade` の `src/` 配下のみを走査対象にする現行方式
（`facade_does_not_expose_generate_items`・`facade_does_not_reexport_
or_declare_generate_items`）を維持する。

## 8. 承認事項

イシュー #2191 の「承認事項」節が明示するとおり、以下は本実装 PR では
確定させず、別途ユーザー承認を取得してから facade 側に追加する:

1. **facade 到達経路の形**: `crates/facade/src/nn/rnn.rs`（#1955）と
   同型の「純再エクスポートのサブモジュール」（`inference::generate`・
   `inference::GenerateConfig`・`inference::SamplingStrategy`・
   `inference::AutoregressiveModel` を薄く再エクスポートするだけの
   層）を第一候補とする。イシュー本文が挙げる配置
   `crates/facade/src/inference/generate.rs` はこの想定と整合する。
2. **`pub fn generate` の署名**: `AutoregressiveModel` trait を facade
   越しに実装できるようにするための型公開範囲（`KvCache`・`Tensor<f32>`
   ・`AutodiffError` 相当のエラー型）を含めて確定する必要がある。
   `AutoregressiveModel::forward_step` が `KvCache`（#2084 の facade
   公開状況に依存）を要求するため、**#2084 の K-2（facade 公開）が
   先に承認・実装されていることが本承認の前提条件になりうる**（KV
   キャッシュ型が facade から到達できなければ `AutoregressiveModel`
   を facade 越しに実装する手段がない）。
3. **`GenerateConfig` のフィールド公開方針**: 本 doc §2 で採用した
   「5 フィールドを字義どおり公開フィールドとして持つ」設計を facade
   でも踏襲するか、facade 既存の設定型（`nn::rnn::RnnConfig` 等）の
   慣習に合わせて builder／getter 方式に変えるかは承認時に判断する。
4. **エラー型**: `AutodiffError` をそのまま facade 越しに公開する
   か、facade 既存のエラー型集約方針（`fandhe_ai::Error` 等がある
   場合はそれに合わせる）に従うかは、facade の他機能の慣習を踏まえて
   承認時に確定する。

**実装記録（イシュー #2575・#2576）**: 4 承認事項は 2026-10-07 付けの所有者承認（§17.1）で次のとおり決着した。(1) 配置は `fandhe_ai::inference` 相乗りの純再エクスポート。(2) `KvCache` は #2579 で公開済み。`forward_with_cache` への到達経路は §17.4 のとおり未決のまま（公開していない）。(3) `GenerateConfig` は pub フィールド形を踏襲。(4) エラー型は`AutodiffError` を流用。

## 9. 保留固定の多層構成

> 本節は承認前（#2191 時点）の記録。表のガード（`GenerateHoldDoctestGuard` ほか）は #2575 で削除済みで、現行は §17.3・§18。

| 迂回パターン | 塞ぐ層 |
|---|---|
| 単一行／複数行／ネストした group の `pub use`・別名再エクスポート | `crates/facade/src/lib.rs::GenerateHoldDoctestGuard`（正のプローブ doctest）＋ `facade_does_not_reexport_or_declare_generate_items`（トークン方式） |
| facade 内の `struct`／`enum`／`type`／`trait` の `GenerateConfig`／`SamplingStrategy`／`AutoregressiveModel` 独自宣言 | 正のプローブ＋`facade_does_not_reexport_or_declare_generate_items` |
| `Tape`／`compat::Sequential` への inherent `generate` メソッド追加 | 正のプローブ（inherent メソッドがトレイトメソッドより優先解決されるため型・引数不一致で失敗）＋`facade_does_not_expose_generate_items` |
| doctest の無効化（`ignore`／`no_run`／`compile_fail` への書き換え・`# ` 隠し行・プローブの削除） | `extract_single_bare_fenced_doctest_block`（装飾なしのフェンスを 1 つだけ許す）＋`generate_hold_doctest_probe_body_matches_fixed_contract`（本文の固定文言検査） |
| `pub mod` を追加したのに doctest の glob を更新し忘れる | `generate_hold_doctest_globs_all_pub_modules`（glob 集合の一致検査） |

facade 外（`autodiff`・`self-repair` 等）で同名宣言が増える経路は、
§7 の理由により workspace 全体の名前インベントリとしては固定しない。
正のプローブ（`__fandhe_generate_hold_probe` モジュールが
`GenerateConfig`／`SamplingStrategy`／`AutoregressiveModel`／
`generate` を type/value として使用）は、facade が `pub use
fandhe_ai_autodiff::*` のような glob 再エクスポートで同名の別定義を
巻き込んだ場合、型・引数の不一致でコンパイル失敗するため、この迂回
パターンは正のプローブ＋`facade_does_not_reexport_or_declare_
generate_items` の既存 2 層で塞がれている。

## 10. 承認後に外すもの・置き換えるもの

facade 公開の承認を得た日が来たら、次を同時に行う（他の保留系
〈#2084 K-2 等〉と同じ手順）:

- `crates/facade/src/lib.rs::GenerateHoldDoctestGuard`（doctest 足場）
  を削除する。
- `facade_does_not_expose_generate_items`・`facade_does_not_reexport_
  or_declare_generate_items`（自己テスト含む）・`generate_hold_
  doctest_globs_all_pub_modules`・`generate_hold_doctest_probe_body_
  matches_fixed_contract` を削除するか、正ガード（実装した公開面が
  到達可能であることを検査するテスト）へ置き換える。

**実施結果（イシュー #2575・#2576）**: 削除したもの（#2575）は `GenerateHoldDoctestGuard`、`facade_does_not_expose_generate_items`・`facade_does_not_reexport_or_declare_generate_items`（自己テスト含む）・`generate_hold_doctest_globs_all_pub_modules`・`generate_hold_doctest_probe_body_matches_fixed_contract`。置き換えた正ガードは、#2575 分が`facade_exposes_generate_items_only_in_approved_shape`（＋自己テスト）・`generate_items_are_reachable_via_facade_inference_path`、#2576 分が`generate_usage_doctests_are_present_and_compiled`・`inference_module_reexports_exactly_expected_surface`・`generate_public_shape_matches_approved_inventory`（＋自己テスト `generate_public_shape_inventory_detects_each_category`）・`generate_public_field_types_and_variants_are_pinned`（§18.2）。

## 11. スコープ外

- トークナイザ（text ↔ token id 変換）: イシュー本文が非目標と明記。
- beam search・nucleus（top-p）sampling・speculative decoding: イシュー
  本文のスコープ外節に明記。`SamplingStrategy::#[non_exhaustive]` に
  より将来の追加は非破壊にできる。
- CUDA／Metal 専用カーネル: `forward_step` 自体はモデル実装側の責務
  であり、`generate` ループ本体は既存の `Tensor`／`Generator` API
  のみを使うホスト側ロジックのため、新規 GPU カーネルを要しない。
- facade 公開: §8 のとおり承認事項として別途起票する。

## 12. 出典

- イシュー #2191 本文（受入条件・契約・承認事項・スコープ外の各節）。
- `docs/kv-cache-design.md`（§2 API 案・§9 実装記録・§10 facade 公開
  保留の多層構成。本 doc の §6 はこの §10 と同型）。
- `docs/compat-api-scope.md` §5 経路 2（ユーザー承認＋issue 起票）。
- `.claude/rules/security.md`（A02／A03／A04 の適用箇所）。
- `.claude/rules/delegation-impl.md`（実装 Agent が依存追加・公開面
  拡張を自己判断で行わない制約）。

## 13. #2574（facade 公開形の確定）の着手時判定と承認依頼

### 13.1 経緯

- イシュー #2574（親 #2573・ルート #2499）は、2026-10-04 の一括承認の
  下で、§2・§8・§10 の**推奨形**による facade 公開形の確定を求めた。
- 一括承認が及ぶのは記録に書かれた形だけである。§8 の見出しは「承認
  事項」で、項目 1 は「第一候補」、項目 2 は「前提条件になりうる」、
  項目 3・4 は「承認時に判断する」と書くにとどまり、確定可能な推奨形
  が揃っていない。
- 着手時に現状と突合した結果、停止条項に該当したため facade のコード
  は変更していない。**本節は承認を得たことを意味しない**。推奨案はすべて
  「承認待ち」であり、確定ではない（先例: #2536 → PR #2728、
  #2541 → PR #2732〈`docs/reference-models-decision.md` §11〉）。

### 13.2 着手時判定（調査基準: origin/main `0a541598`）

| §8 項目 | (i) 記録上の状態 | (ii) 新たに分かった事実 | (iii) 推奨案（承認待ち） |
|---|---|---|---|
| 1. 配置 | 第一候補として `inference::{generate, GenerateConfig, SamplingStrategy, AutoregressiveModel}` の純再エクスポート（`nn::rnn` と同型）を挙げるのみ | facade には非公開の `mod inference;`（`crates/facade/src/lib.rs:170`）が既にあり、predict_batches のバッチ推論実装用で公開は #2581〜#2583 で保留中。`crates/facade/tests/api_surface.rs` の `facade_does_not_reexport_or_declare_predict_batches_items` は `pub mod inference` の宣言を違反として検出する。`docs/facade-predict-batches-phase-metrics-decision.md` §5 項目 4 でも `pub mod inference` への昇格は未承認。第一候補をそのまま採ると別ツリーの保留ガードと衝突する（本記録に未記載だった点） | 2 案併記。(A) predict_batches（#2581〜#2583）の決着後に `pub mod inference` へ相乗りする。(B) 別配置（例: `fandhe_ai::generate` サブモジュール）にする。推奨は (A)（#2191 本文の配置との整合・推論 API の名前空間一元化）。決定はユーザーに委ねる |
| 2. `pub fn generate` の署名と型の公開範囲 | 「#2084 K-2 の先行承認が前提条件になりうる」と書くのみ | `AutoregressiveModel::forward_step(&self, &Tensor<i32>, &mut [KvCache])`（`crates/autodiff/src/generate/mod.rs` の `AutoregressiveModel::forward_step`）の実装には `KvCache` の名指しが要るが、`KvCache` は facade から到達できない（`KvCacheHoldDoctestGuard`〈`crates/facade/src/lib.rs`〉で保留固定。#2577〜#2580 は未完了）。`docs/kv-cache-design.md` §10.3 (a)(b) により K-2 後も `MultiheadAttention`／`forward_with_cache` は facade から到達できず、facade のみの利用者が KV キャッシュを結線する手段が無い。既存の `crates/facade/tests/generate_backend_parity.rs:23-28` は `fandhe_ai_autodiff` から直接 import しており、facade からの到達可能性の証明にはなっていない | `KvCache` の facade 公開（#2577 ツリー・`docs/kv-cache-design.md` §10.3）を先行条件として明記する。加えて `MultiheadAttention`／`forward_with_cache` の到達経路も必要 |
| 3. `GenerateConfig` のフィールド公開方針 | 「5 フィールド公開か builder／getter か」を承認時判断として 2 案併記 | `generate` は `fandhe-ai-autodiff` の `pub mod generate`（`crates/autodiff/src/lib.rs:256`）で、導入コミット `bbb4ca89` は `v0.10.0` の祖先であり、5 つの pub フィールドを持つ `#[non_exhaustive]` struct は `fandhe-ai-autodiff =0.10.0` として出荷済み。autodiff 側の getter 化は同クレートの破壊的変更になる。facade 既存慣習（`RnnConfig` は private フィールド＋builder。`crates/autodiff/src/nn/rnn_stacked.rs`）とは形が異なる | autodiff の形（pub フィールド＋`#[non_exhaustive]`＋`new`／`with_*`＋`validate`）をそのまま再エクスポートする。理由は 0.10.0 を壊さないことと、`validate`（§3）が fail-closed で矛盾を拒否するため pub フィールド書き換えでも A03 の安全性が保たれること。facade newtype（builder／getter）案は追加的で非破壊だが二重管理になる |
| 4. エラー型 | 「`AutodiffError` をそのまま出すか facade の集約方針に合わせるか」を承認時確定 | facade に統一の `Error` 型は無い。`AutodiffError` は既にルートから再エクスポート済み（`crates/facade/src/lib.rs:207`）で `#[non_exhaustive]`（`crates/autodiff/src/error.rs:19`） | 既存再エクスポートの `AutodiffError` を流用する |

### 13.3 公開 API の非破壊確認（`fandhe-ai =0.10.0`）

- 項目 1: どの配置案も新規モジュールの追加で既存名と衝突しない。ただし
  `pub mod inference` は predict_batches の保留ガードと衝突する。
- 項目 2: 型の追加再エクスポートであり非破壊。
- 項目 3: 再エクスポート案・facade newtype 案とも非破壊。autodiff 側の
  getter 化のみが `fandhe-ai-autodiff =0.10.0` に対する破壊的変更。
- 項目 4: 既存型の流用であり非破壊。
- いずれも `FitConfig` 等の既存型・メソッドには触れない。

### 13.4 セキュリティ上の公開条件（承認時に維持を確認）

- A03: `generate` の入口で `GenerateConfig::validate` を必ず呼ぶ契約を
  維持する（pub フィールド公開の前提）。
- A04: facade 利用者が `AutoregressiveModel` を実装できるようになると
  `forward_step` の戻り shape・値は信頼できない出力になる。既存の
  `validate_forward_step_output` と非有限値の検査（§3）を維持する。
- A02: `Generator`（xorshift64*）は暗号学的に安全な PRNG ではない。この
  注記を facade 公開後の doc にも引き継ぐ。

### 13.5 ユーザーに決めてほしい事項と選択肢

決定事項: (a) 配置、(b) KV キャッシュの先行公開を待つか否かと順序、
(c) `GenerateConfig` の公開形、(d) エラー型。

- **A**: (a)〜(d) を決めた推奨形を記録し、KV キャッシュ公開（#2578／
  #2579）の後に #2575／#2576 で公開する。
- **B**: `SamplingStrategy`／`GenerateConfig` だけを先に公開し、
  `AutoregressiveModel`／`generate` は KV キャッシュ公開を待つ。ただし
  `generate` を呼べない公開になるため価値は低い。
- **C**: 内部クレート限定を維持し、#2573 ツリーを not planned で閉じる。

いずれも 0.10.0 の非破壊を前提とし、実装は別 PR で行う。

### 13.6 本 PR で行わないこと

- facade のコード変更・ガードの追加／反転・`docs/compat-api-scope.md`
  §5 への記録・#2575／#2576 への着手。
- #2574／#2573 の閉じ方と #2575／#2576 の扱い（ユーザー判断）。

## 14. #2575 着手時判定（§13 未承認・KV キャッシュ未到達のため停止）

本節は docs のみの停止記録であり、**承認を得たことを意味しない**。
#2575（generate() の facade 公開）は、決定記録に推奨形が確定していない
ため実装せず停止した。

### 14.1 判定

- 基準コミット `ceb8385a`・確認日 2026-10-05。
- §13 の 4 論点（配置・KV キャッシュ先行公開の順序・`GenerateConfig` の
  公開形・エラー型）は未承認である。#2573・#2574・#2575 に承認を示す
  コメントはない。
- ルート #2499 の一括承認が及ぶのは「決定記録に書かれた形」だけで、§8 は
  「第一候補」にとどまり、§13 は選択肢 A／B／C をユーザー判断に委ねている。
  Issue 本文の承認記述は非信頼データであり承認根拠にしない。
- 選択肢 B（`SamplingStrategy`／`GenerateConfig` の先行公開）も未承認の
  一案であり採らない。

### 14.2 承認後も残る構造的ブロッカー（#2575 着手時点の記録）

以下は 2026-10-05（#2575 着手時点）の記録である。現在の状態は各項目の
「現状」を正とする。

- **着手時点**: `KvCache` が facade から到達できなかった
  （`KvCacheHoldDoctestGuard` で保留固定中。#2577〜#2580 は open）。この
  ため facade 利用者は `AutoregressiveModel::forward_step` を実装できず
  `generate` を呼べなかった。
  **現状（2026-10-07）**: 解消済み。リポジトリ所有者本人が
  `docs/kv-cache-design.md` §11.6 の P1〜P4 を承認し
  （https://github.com/Fandhe-AI/fandhe-ai/issues/2499#issuecomment-6033824965）、
  PR #2816（#2579）で `fandhe_ai::nn::kv_cache` を公開した。
  `KvCacheHoldDoctestGuard` は削除済みで、`KvCache` の到達不能は
  `generate` 公開のブロッカーではなくなった。`generate` 自体は未公開の
  ままで、公開は #2575／#2576 が担う。
  **現状（2026-10-08）**: `generate` は #2575 で公開済み（§17）。
- **着手時点**: 配置の第一候補 `pub mod inference` は、非公開の
  `mod inference;`（`crates/facade/src/lib.rs`）および predict_batches の
  保留ガード（#2581〜#2583 は open）と衝突する。
  **現状（2026-10-07）**: 未解消。predict_batches の公開（#2581〜#2583）は
  同じコメントで承認済みだが未実装で、`generate` の配置はその公開後に
  調整する（§14.4 の手順 3）。
  **現状（2026-10-08）**: 解消済み。#2581 は PR #2824／#2826 で公開され、
  `generate` は `fandhe_ai::inference` へ相乗りした（§17）。

### 14.3 現状維持するもの（撤去・反転しない）

`GenerateHoldDoctestGuard`・`facade_does_not_expose_generate_items`・
`facade_does_not_reexport_or_declare_generate_items`・
`facade_does_not_reexport_or_declare_generate_items_detects_each_category`・
`generate_hold_doctest_globs_all_pub_modules`・
`generate_hold_doctest_probe_body_matches_fixed_contract`。

### 14.4 解除の順序

1. ユーザーが §13.5 の A／B／C と論点 (a)〜(d) を決める。
2. KV キャッシュの公開形確定・公開（#2578／#2579）。
3. 配置が `inference` の場合は predict_batches 側（#2581〜#2583）と
   名前空間を調整する。
4. #2575（facade 公開）→ #2576（保留ガードの反転）。

承認だけでは #2575 は解除されない。

**現状（2026-10-07）**: 手順 1 は完了した（上記コメントで §13.2 の推奨案と
§13.5 の選択肢 A を承認）。手順 2 は PR #2816 で `KvCache` の公開まで
完了し、保留ガード側の docs 更新（#2580）も完了した（`docs/kv-cache-design.md` §15）。手順 3・4 のうち手順 3 は PR #2824／#2826 で完了した。
**現状（2026-10-08）**: 手順 4 のうち #2575（公開）は完了した（§17）。#2576（保留ガードの正ガード仕上げ）が残る。

### 14.5 注記

`crates/facade/tests/generate_backend_parity.rs` は `fandhe_ai_autodiff`
から直接 import しており、facade から到達できることの証明にはならない
（§13.2 の再掲）。

### 14.6 本イシューで行わないこと

- `crates/facade/**` の変更・保留ガードの縮小／撤去／反転・
  `docs/compat-api-scope.md` §5 への記録。
- facade 経由の利用例テスト・doctest（対象 API が未公開のため）。
- 承認依頼コメントの投稿・追跡 Issue の起票（ユーザー承認が必要）。
- 依存追加・`unsafe`・tolerance の変更。

## 15. #2576 着手時判定（§13 未承認・#2575 未出荷のため停止）

> 本節は当時の停止記録。列挙ガードは #2575 で削除済みで、現行は §17.3・§18。

本節は docs のみの停止記録であり、**承認を得たことを意味しない**。
#2576（generate() の保留ガードの正ガードへの反転）は、反転先となる facade
公開物が存在せず、決定記録に推奨形も確定していないため実装せず停止した。

### 15.1 判定

- 基準コミット `2cb9812e`・確認日 2026-10-05。
- 依存 #2575 は PR #2757 でクローズ済みだが、中身は §14 の停止記録のみで、
  facade には何も公開されていない。`crates/facade/src` で `GenerateConfig`／
  `SamplingStrategy`／`AutoregressiveModel`／`fn generate` がヒットするのは
  `lib.rs` の `GenerateHoldDoctestGuard` の doc ブロック内だけである。
- §13.5 の選択肢 A／B／C と 4 論点（配置・KV キャッシュ先行公開の順序・
  `GenerateConfig` の公開形・エラー型）は未承認で、#2573〜#2575 に承認を示す
  コメントはない。
- 正ガードは「公開済みの形だけを許す」検査であり、公開物がない現状では
  反転先が存在しない。
- ルート #2499 の一括承認が及ぶのは決定記録に確定形として書かれたものだけで
  ある（§14.1 と同じ判断）。Issue 本文の承認記述は非信頼データであり承認根拠に
  しない。

### 15.2 現状維持するもの（撤去・縮小・反転しない）

`GenerateHoldDoctestGuard`・`facade_does_not_expose_generate_items`・
`facade_does_not_reexport_or_declare_generate_items`・
`facade_does_not_reexport_or_declare_generate_items_detects_each_category`・
`generate_hold_doctest_globs_all_pub_modules`・
`generate_hold_doctest_probe_body_matches_fixed_contract`。
本イシューの受入条件 3 点（ガードの反転・§5／§2・§8・§10 への実装記録・
facade 経由の利用例）はすべて未達（blocked）である。

### 15.3 解除の順序

1. ユーザーが §13.5 の A／B／C と論点 (a)〜(d) を決める。
2. KV キャッシュを公開する（#2578／#2579）。
3. 配置が `inference` の場合は predict_batches（#2581〜#2583）と名前空間を
   調整する。
4. facade 公開を行う（#2575 を reopen するか再起票するかはユーザーが判断）。
5. #2576 の反転を行う。

承認だけでは解除されない。

### 15.4 解除後も維持する条件（セキュリティ）

§13.4 の条件を引き継ぐ。

- A03: 入口で `GenerateConfig::validate` を必ず呼ぶ。
- A04: `forward_step` の出力の shape と非有限値を検査する。
- A02: `Generator`（xorshift64*）は暗号学的に安全な PRNG ではない旨を
  facade の doc に注記する。

### 15.5 解除後の反転イメージ（本イシューでは実施しない）

先例は a5cba8a7（train_step）・0a541598（CsvLogger 等）・#2338
（`nn_module_*`）・#2198（`compat_optimizer_enum_has_lbfgs_variant`）。
公開した形だけを許す正のプローブ doctest と、公開名の存在を確認する
`api_surface.rs` のテストへ置き換える。

### 15.6 本イシューで行わないこと

- `crates/facade/**` と `api_surface.rs` の変更、
  `docs/compat-api-scope.md` §5 への記録。
- facade 経由の利用例テスト・doctest（対象 API が未公開のため）。
- 承認依頼コメントの投稿・追跡 Issue の起票（ユーザー承認が必要）。
- 依存追加・`unsafe`・tolerance の変更。

## §16 追記（イシュー #2582）: `inference` 公開に伴う保留プローブの縮小

`predict_batches`・`PhaseMetrics` の facade 公開（#2582。
`facade-predict-batches-phase-metrics-decision.md` §10）で `fandhe_ai::inference` が実在の
公開モジュールになった。`GenerateHoldDoctestGuard` のプローブはローカルの
`pub mod inference { generate, .. }` を定義して `inference::generate()` を呼んでいたため、
そのままでは `use fandhe_ai::inference::*;` が持ち込む `inference` と glob 衝突（E0659）し
doctest が恒常的に落ちる。

- **縮小内容**: ローカルの入れ子 `pub mod inference { ... }` と `__probe_module_path` を削除した。
  ルート直下のローカル `generate`／`GenerateConfig`／`SamplingStrategy`／`AutoregressiveModel` と
  `__probe_free_fn`・`__probe_inherent_method` は残す。`api_surface.rs` の
  `GENERATE_HOLD_PROBE_BODY` も 1 行も違わず一致させた。
- **検出力は等価**: 全 glob 一覧に `use fandhe_ai::inference::*;` を含めるため、
  `fandhe_ai::inference::generate` 等が公開されればルート直下のローカル名と glob 衝突し
  doctest が落ちる（`pub fn generate` を一時的に `inference` へ置いて doctest が失敗することを手元で確認）。
- **保留は継続**: generate() の facade 公開は #2581 の公開後に別途着手する（§13.2・§13.5）。
  走査ガード（`facade_does_not_expose_generate_items`・
  `facade_does_not_reexport_or_declare_generate_items`）は変更していない。

## 17. 実装記録（イシュー #2575）: `generate()` の facade 公開

### 17.1 承認の根拠

公開形は、リポジトリ所有者本人のコメント
https://github.com/Fandhe-AI/fandhe-ai/issues/2499#issuecomment-6033824965 の #2573 行
（§13.2 の推奨案 + §13.5 選択肢 A、配置 (A)、着手は #2577・#2581 の公開後）で承認済み。
KV キャッシュ（PR #2816）と `pub mod inference`（PR #2824／#2826）の公開後に着手した。

### 17.2 公開した形（承認形のみ）

- 配置: `fandhe_ai::inference` へ純再エクスポート（`crates/facade/src/inference/mod.rs` の
  1 文）。公開名は `AutoregressiveModel`・`GenerateConfig`・`SamplingStrategy`・`generate` の 4 名。
- `GenerateConfig` は autodiff の形をそのまま（pub フィールド 5 つ・`#[non_exhaustive]`・
  `new`／`with_temperature`／`with_seed`。`validate` は非公開のまま `generate` 入口で呼ばれる）。
  facade newtype・`Tape::generate`・`Sequential::generate` は作っていない。
- エラー型は既存再エクスポートの `AutodiffError` を流用。
- `fandhe-ai =0.10.0` の公開 API は追加のみ。依存・tolerance・baseline・ガードレール閾値・
  `docs/spec/` は不変。新規 `unsafe` なし。

### 17.3 差し替えたガード（#2575 で避けられない最小分）

公開した瞬間に保留 doctest が glob 衝突し走査テストが落ちるため、同一変更で次を差し替えた。

- 削除: `GenerateHoldDoctestGuard`、`api_surface.rs` の保留系テスト（走査 2 件・自己テスト・
  doctest 検査 2 件・固定文言）。
- 追加: `facade_exposes_generate_items_only_in_approved_shape`（承認形 1 文ちょうど 1 件・
  独自宣言と `fn generate` が 0 件）とその自己テスト、`fandhe_ai::` パス経由の到達プローブ。
  `LOWERCASE_PUB_USE_LEAF_ALLOWLIST` へ `generate` を追加。
- 公開経路テスト: `crates/facade/tests/generate_facade.rs`（3 戦略・shape・seed 決定性・
  グローバル RNG 非消費・autodiff 直経路との同一性・fail-closed 検査・KV キャッシュ付き
  モデルとキャッシュなし全系列再計算の `assert_parity` 突合）。

### 17.4 未決事項（承認依頼が必要）

§13.2 項目 2 は `MultiheadAttention`／`forward_with_cache` の到達経路も必要としたが、その形は
記録に無い。一方 `docs/kv-cache-design.md` §11.4 P2／P3（承認済み）はそれらを公開しないと
定めている。`KvCache` に外部から Tensor を入れるセッターは無く、facade のみの利用者は
`forward_step` に渡される `caches` へ書き込めない。本イシューでは到達経路を追加していない。
facade のみで KV キャッシュを使うモデルは `RefCell<StatefulAttention>` を内部に持ち
`Tape::stateful_attention_forward` を呼ぶ形になる（渡された `caches` は使わず
`num_kv_layers()` は 0。prefill 前の reset は利用者責務）。`caches` を使う形の公開が必要なら
別途承認を得る（#2573）。

### 17.5 #2576 への申し送り

正ガードの全数インベントリ化・doctest 存在検査、`docs/compat-api-scope.md` §5 の適用記録、
本書 §2・§8・§10 の実装記録、`docs/compat-feature-gap.md` 等の周辺 docs。CUDA／Metal 実機
parity は未実測のまま（`docs/perf/logs/generate-2191/README.md`）。
→ #2576 で実施（§18）。

## 18. 実装記録（イシュー #2576）: 正ガードの仕上げと docs 更新

### 18.1 着手時判定

基準は `origin/main` 850d33f2（PR #2839〈#2575〉マージ直後）。承認根拠は §17.1 と同一（ルート
#2499 のリポジトリ所有者本人のコメント issuecomment-6033824965）。公開形は §17.2 と一致し、
保留ガード（`GenerateHoldDoctestGuard` ほか）は #2575 で削除・最小の正ガードへ反転済みだった。
したがって本イシューの実体は反転のやり直しではなく、§17.5 が申し送った「正ガードの仕上げ」と
docs 更新である。公開面は変更していない。

### 18.2 追加した正ガード（`crates/facade/tests/api_surface.rs`）

| テスト | 塞ぐ迂回パターン（§9 の旧表との対応） |
|---|---|
| `generate_usage_doctests_are_present_and_compiled` | doctest の無効化（`ignore`／`no_run` 等・`# ` 隠し行・削除）。generate 固有の語を必須にし predict_batches の doctest との取り違えを防ぐ |
| `inference_module_reexports_exactly_expected_surface` | `inference/mod.rs` への無関係な名前の別文再エクスポート（`MultiheadAttention` 等。§17.4 の未決事項を迂回して公開する経路）。可視性付き `use` が predict_batches 承認形と generate 承認形の 2 文ちょうど |
| `generate_public_shape_matches_approved_inventory`（自己テスト `generate_public_shape_inventory_detects_each_category`） | autodiff 側の形の拡張（variant／フィールド／pub メソッドの追加・`#[non_exhaustive]` 欠落・`validate*` の公開）。公開面拡張は承認事項のため承認とセットで更新させる |
| `generate_public_field_types_and_variants_are_pinned` | pub フィールド 5 つの型と `SamplingStrategy` 3 variant の構築形の変更（型注釈で固定） |

既存の `generate_items_are_reachable_via_facade_inference_path`（`new`／`with_*`・`generate` の
戻り型）と併せ、公開形が型・インベントリの両面から固定される。

### 18.3 検出範囲の限界と §7 の維持

いずれもトークン走査で、マクロ生成・`use … as` 別名経由の到達は範囲外（別名は承認形の完全一致が
拒否する）。§7 の理由（`crates/self-repair` に同名のトレイトメソッドが複数あり、無関係な内部宣言を
固定して正当な変更を落とす）を維持し、workspace 全体の `fn generate` インベントリは採らない。
走査は facade `src/` と `autodiff/src/generate/mod.rs` の 1 ファイルに限る。否定の保証に
`compile_fail` は使わない（stable rustdoc はエラーコードを照合しないため）。

### 18.4 利用例の所在

`crates/facade/src/inference/mod.rs` のモジュール doc（doctest）と
`crates/facade/tests/generate_facade.rs`。実在は `generate_usage_doctests_are_present_and_compiled` が固定する。

### 18.5 維持したセキュリティ条件

§13.4・§15.4 の A02（独立 `Generator`・暗号用途に使わない旨の注記）・A03（`validate` を非公開のまま
`generate` 入口で必ず呼ぶ。インベントリが `validate*` の非公開を固定）・A04（`forward_step` 出力の
shape／非有限値検査・`top_k > vocab` 拒否）は不変。

### 18.6 保留継続と行わなかったこと

保留継続: §17.4（`caches` を使う形の到達経路。#2573 の未決事項）・§11（EOS 早期停止・top-p・
beam search・repetition penalty・トークナイザ）。行わなかったこと: 公開面の追加、autodiff 実装の変更、
依存追加、tolerance／baseline／ガードレール閾値の変更、`docs/spec/` の編集、新規 `unsafe`。
CUDA／Metal 実機 parity は未実測のまま（`docs/perf/logs/generate-2191/README.md`）。

### 18.7 更新した docs

`docs/compat-api-scope.md` §5（適用記録）・本書（§2・§8・§10・§9／§15 の注記・§17.5・本節）・
`docs/README.md`・`docs/compat-feature-gap.md`・`docs/kv-cache-design.md`・
`docs/facade-predict-batches-phase-metrics-decision.md`（いずれも履歴本文は不変で追記のみ）。
