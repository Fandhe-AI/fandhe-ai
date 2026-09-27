# `generate()` 自己回帰ループの設計確定と facade 公開保留（イシュー #2191）

対応イシュー: #2191（親 #2084 と同系列の LLM 推論機能群。KV キャッシュ
実装 #2084 の兄弟機能）。
位置づけ: 実装（`crates/autodiff/src/generate.rs`）は完了済み。本 doc は
その設計判断の記録と、facade（`fandhe_ai`）公開保留の多層固定・承認依頼
用の事前設計を記す。tolerance／baseline／`Cargo.toml` 依存／ガードレール
閾値／`docs/spec/`（正本）は一切変更しない。

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
  ではなく `crates/autodiff/src/generate.rs`（内部クレート）に実装した。
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

## 2. API 確定（`crates/autodiff/src/generate.rs`）

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

## 3. セキュリティ考慮（OWASP Top 10）

- **A03（インジェクション／不正入力）**: `GenerateConfig::validate`が
  `temperature` の有限性・正値、`strategy` とフィールドの整合、`TopK`
  の `k >= 1` を fail-closed で拒否する。`validate_top_k_le_vocab` は
  prefill で確定した `vocab_size` に対する `k` の上限を拒否する。
  `sample_step` は logits の非有限値（`NaN`／`inf`）混入を明示的に
  検出し `Err` を返す（モデル実装のバグがサンプリング側に伝播しない
  ようにする）。
- **A04（安全でない設計）**: 出力バッファの確保前に `b.checked_mul
  (config.max_length)` で `usize` オーバーフローを検出する
  （`.claude/rules/security.md` A04 方針）。`validate_forward_step_
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
§8 が、`crates/autodiff/src/generate.rs`・`crates/facade/src/lib.rs`
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

## 9. 保留固定の多層構成

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
