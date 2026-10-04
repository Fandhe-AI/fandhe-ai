# Dropout2d・AlphaDropout・EmbeddingBag の設計判断記録

イシュー #2161（親 #2131）。`docs/autodiff-spatial-layers-decision.md`
（#2159）と同型の記録。

## §0 結論

PyTorch の `nn.Dropout2d`・`nn.AlphaDropout`・`nn.EmbeddingBag` に
相当する 3 層を、`fandhe_ai_autodiff` の `nn` モジュール
（`crates/autodiff/src/nn/{dropout,embedding_bag}.rs`）へ追加した。
いずれも既存 `Var` 演算（`Var::dropout_with_mask`・`Var::add`・
`Var::embedding`・`Var::sum`／`mean`／`max`／`narrow`／`cat`）の薄い
合成であり、新規 `Op`・`BackendOps` メソッド・VJP・GPU 専用カーネル
は追加していない。`Var` に inherent の `pub fn` は追加していない
（`Var` は facade から再エクスポートされるため。#2143・#2144・
#2146・#2159 の先例）。facade 公開（`compat::Sequential::
add_dropout2d`／`add_alpha_dropout`／`add_embedding_bag`・`Var::
dropout2d`／`alpha_dropout`／`embedding_bag` の委譲メソッド）は
#2161 時点では承認待ちのまま対象外とし、`crates/facade/src/lib.rs::
DropoutEmbeddingBagHoldDoctestGuard`（正のプローブ doctest）と
`crates/facade/tests/api_surface.rs` のソース走査（2 テスト＋自己
テスト）で多層固定していた。**その後イシュー #2528（親 #2520・ルート
#2499 の一括承認）で承認・実装済み**（§8 参照。層型の再エクスポートと
自由関数での公開は未承認のまま保留ガードで固定を継続）。

## §1 背景

イシュー #2161・親 #2131 の `gh api .../comments` はいずれも 0 件
（2026-09-25 時点で着手前に確認済み）。親 #2131 はこのツリーでの
facade 公開面の拡張を「設計判断記録 → 承認 → 実装」の 2 段階と
定めているため、本実装は内部クレート限定に倒す（同じツリーの先例
#2159 §1・#2158 と同じ判断）。イシュー本文は facade への `add_*`
3 件を「承認事項（経路 2）」として列挙しているのみで、承認そのもの
はまだない。

## §2 設計判断

### §2.1 Dropout2d

`crates/autodiff/src/nn/dropout.rs` に既存 `Dropout` と並べて追加
した。rank 4（`[N, C, H, W]`）限定（それ以外の rank は
`AutodiffError::Shape(RankMismatch)`）。マスク生成は新設
`crate::grad::feature_dropout_mask`（`[N, C]` を `dropout_mask` で
抽選してから `[N, C, H, W]` へホスト側展開）で、forward は
`Var::dropout_with_mask`（`Dropout` と共有）へそのまま委譲する。
[`Op::Dropout`] の VJP は「mask と input が同 shape」を前提にする
ため、展開済みの contiguous なマスクを渡す設計にした。

### §2.2 AlphaDropout

同ファイルに追加。数式は ATen `_dropout_impl` の alpha 分岐を再現
する: `alpha = 1.7580993408473766`（SELU 定数）・
`a = 1/sqrt((alpha^2*p+1)*(1-p))` を `f64` で計算してから `a`／
`alpha*a`／`alpha*a*p` を**それぞれ 1 回だけ** `f32` へ narrow し
（新設 `crate::grad::alpha_dropout_mask_and_bias`）、
`out = x*noise + b`（`noise` は keep 位置で `a`・drop 位置で
`0.0`。`b` は keep 位置で `alpha*a*p`・drop 位置で
`-(alpha*a)+alpha*a*p`）を `Var::dropout_with_mask`（`x*noise`）→
`Var::add`（`+b`。`b` は `Tape::var_no_grad` で勾配を持たない定数）
の 2 段構成で計算する。`p == 1.0` は `a` の分母がゼロになり
`inf * 0 = NaN` が生じるため、専用の全ゼロマスク経路（`b` は加え
ない）に分岐する（`at::native::_dropout_impl` の `p == 1` 特例と
同じ）。`forward_host`（tape 不要経路）は同じ関数列を
`crate::grad::dropout_with_fallback`→新設
`crate::grad::alpha_dropout_bias_add_with_fallback` で再現し、
`Module::forward`／`forward_host` の bit 完全一致を維持する。

### §2.3 EmbeddingBag

新規 `crates/autodiff/src/nn/embedding_bag.rs`。`EmbeddingBagMode`
（`#[non_exhaustive] enum { Sum, Mean, Max }`。既定 `Mean`）を追加。
`EmbeddingBag`（パラメータ本体）・`EmbeddingBagVars`（`bind` 後の
tape 登録済み型。`pub weight: Var<'t>`）は `Embedding`／
`EmbeddingVars` と同型の分離パターンを踏襲する。

- `forward(ids: &Tensor<i32> /* [B, L] */)`: `padding_idx == None`
  かつ `L > 0` の高速経路は `weight.embedding(ids, None)` で
  `[B, L, D]` を得てから `dim=1` を縮約する（`Var::embedding` 呼び
  出しが 1 回のみ）。それ以外（`padding_idx` あり、または `L == 0`）
  は `offsets = [0, L, 2L, .., B*L]`（`include_last_offset = true`）
  を組み立てて `forward_with_offsets` へ委譲する
- `forward_with_offsets(ids: &Tensor<i32> /* [N] */, offsets, include_last_offset)`:
  可変長 bag を扱う一般経路。offsets の検査（空でない・先頭 0・
  単調非減少・全て `<= N`・`include_last_offset` の整合・bag 数 >
  0）をすべて tape を操作する前に終える。**id 範囲検査は単一の
  `Var::embedding` 呼び出しへ一本化する**（`padding_idx` と一致する
  id を除外しつつ全 bag ぶんの id を host 側で 1 本へ連結してから
  1 回だけ `weight.embedding` を呼ぶ）。`Var::embedding` はバック
  エンド呼び出しより前に id 範囲を検査してから `push_eager` する
  契約のため、この 1 回の呼び出しは「全 bag ぶんの id が妥当なら
  成功しノードを 1 つ積む／範囲外 id が 1 件でもあれば何もノードを
  積まず即座に `Err`」という原子的な単位になる。bag ごとに
  `embedding` を呼ぶ設計だと、後方の bag で範囲外 id が見つかった
  時点で前方の bag が既にノードを積んでいる（失敗時に tape へ
  孤児ノードが残る）ため、この設計で回避した（実装計画レビューで
  発見した設計上の矛盾の是正）。各 bag は `narrow`（0 要素なら
  `tape.var_no_grad` でゼロ行）→ `sum`／`mean`／`max` → `reshape`
  → 最後に `Var::cat` で `[B, D]` へ組み立てる
- `Module::forward`（f32 `Var` 契約）は `EmbeddingVars::
  forward_from_var` と同型の橋渡し（`materialize_fallible`〈層 1・
  fail-closed〉→ `ids_from_f32`〈`embedding.rs` から `pub(crate)`
  化〉→ `forward`）。`supports_forward_host = false`（`Embedding`
  と同じ）

## §3 契約（変更禁止・維持）

- tolerance・baseline・`Cargo.toml` の依存・ガードレール閾値・
  `docs/spec/` は変更していない
- 新規 `unsafe` なし・依存追加なし
- crates.io 公開済みの `fandhe-ai =0.9.0`・`fandhe-ai-autodiff` の
  公開 API は非破壊（新しい pub 型の追加と `Module` trait への
  既定実装付きメソッドの追加のみ）

## §4 テスト

- `crates/autodiff/src/grad.rs`：`feature_dropout_mask`・
  `alpha_dropout_mask_and_bias`・`alpha_dropout_bias_add_with_
  fallback` の追加（unit test は呼び出し元の統合テストで間接検証）
- `crates/autodiff/src/nn/dropout.rs`・`embedding_bag.rs` の各
  unit test（構築検査・数値境界値・rank 検査・`padding_idx` 除外・
  offsets 検査・孤児ノード非発生・`set_parameter` 等）
- `crates/autodiff/tests/nn_dropout_variants.rs`：Dropout2d の
  チャネル単位一致・`p=1`・rank 検査・`forward_host` bit 一致、
  AlphaDropout の解析解一致（手計算値との突合）・`p=1` 特例・勾配
  `= noise`・`forward_host` bit 一致
- `crates/autodiff/tests/nn_embedding_bag.rs`：公開 API 経由の
  forward・`Module::forward` との一致・named_parameters・backward
  の勾配加算
- `crates/facade/tests/dropout_variants_backend_parity.rs`：CPU
  （`CpuBackendOps`）対 `NaiveOps` の parity（Dropout2d・
  AlphaDropout とも forward／backward bit 完全一致）と、`cuda_*`／
  `metal_*` の `#[ignore]` parity（実機未実測。`docs/perf/logs/
  dropout-embedding-bag-2161/README.md` 参照）
- `crates/facade/tests/embedding_bag_backend_parity.rs`：CPU 対
  NaiveOps の parity（Sum／Mean は REQ-2 複合判定、Max は bit 完全
  一致、backward は REQ-2 複合判定）と `cuda_*`／`metal_*` の
  `#[ignore]` parity（実機未実測）
- `crates/facade/src/lib.rs::DropoutEmbeddingBagHoldDoctestGuard`
  （正のプローブ doctest）・`crates/facade/tests/api_surface.rs` の
  `dropout_embedding_bag_hold_doctest_globs_all_pub_modules`・
  `dropout_embedding_bag_hold_doctest_probe_body_matches_fixed_
  contract`・`compat_sequential_does_not_expose_dropout_embedding_
  bag_add_methods`（＋自己テスト）

## §5 スコープ外（`out-of-scope-tracking.md` に従う。Issue 起票は
ユーザー承認後）

- facade `compat::Sequential::add_dropout2d`／`add_alpha_dropout`／
  `add_embedding_bag`、および `Var::dropout2d`／`alpha_dropout`／
  `embedding_bag` の facade 公開（#2161 時点の記述。#2528 で承認・実装済み。§8 参照）
- GPU 専用カーネル（既存の `mul`／`add`／`gather`／`sum`／`max` の
  経路と、ホストフォールバックだけで到達させる）
- Dropout2d の 3D 入力（PyTorch でバージョンにより意味が変わる）・
  `Dropout1d`／`Dropout3d`・`FeatureAlphaDropout`
- EmbeddingBag の `per_sample_weights`・`max_norm`・
  `scale_grad_by_freq`・`sparse`・`from_pretrained(freeze)`、多数の
  bag を一括処理する経路の性能最適化（一般経路は bag ごとにノードが
  増える）
- ONNX export（`onnx-interop::export_nn`）での 3 層への対応
- CUDA（GB10）・Metal 実機での `#[ignore]` parity の実測（申し送る）

## §6 承認事項（#2161 時点の列挙。#2528 で承認・実装済み。§8 参照）

1. facade `compat::Sequential` への `add_dropout2d`／
   `add_alpha_dropout`／`add_embedding_bag` の追加（経路 2）と、
   学習経路（`bind`／`trainable_parameters`／`apply_parameters`／
   `trainable_vars`／`trainable_grads`／
   `contains_resident_unsupported_layer`）への結線
2. 上記に伴う `Var::dropout2d`／`Var::alpha_dropout`／
   `Var::embedding_bag` の追加（経路 1）と保留ガード
   （`DropoutEmbeddingBagHoldDoctestGuard`・`api_surface.rs` の
   対応する否定ガード）の撤去

## §7 実機実測の申し送り

CUDA（DGX Spark GB10）・Metal 実機は本エージェント実行環境に無いため
`#[ignore]` テストを未実行のまま出荷する。実行コマンド・記入欄は
`docs/perf/logs/dropout-embedding-bag-2161/README.md` を参照。

## §8 実装記録（イシュー #2528・親 #2520・ルート #2499 本文「承認範囲」節の一括承認）

§6 の承認事項 1・2 を、ルート #2499（2026-10-04 一括承認: Phase 1〜3 は既存決定記録の推奨形で
facade 公開を実装してよい）に基づき実装した。§6 は名前だけを挙げ具体シグネチャを定めていなかったため、
既存の `Var::dropout`・`Var::embedding`・`add_dropout`・`add_embedding` の規約から機械的に導出した
（#2527 の `docs/autodiff-adaptive-max-global-pool-decision.md` §8 と同じ判断）。
`fandhe-ai =0.10.0` の公開 API は非破壊（追加のみ）。`Cargo.toml`／`Cargo.lock`・tolerance／baseline・
`docs/spec/` は不変。新規 `Op`／`BackendOps`／VJP／GPU カーネルの追加なし。

### 8.1 公開した名前

| 公開面 | シグネチャ |
|---|---|
| `Var::dropout2d` | `fn(&self, p: f32, training: bool) -> Result<Var<'t>, AutodiffError>`（`nn::dropout2d_forward` への 1 行委譲） |
| `Var::alpha_dropout` | `fn(&self, p: f32, training: bool) -> Result<Var<'t>, AutodiffError>`（`nn::alpha_dropout_forward` への 1 行委譲） |
| `Var::embedding_bag` | `fn(&self, ids: &Tensor<i32>, mode: EmbeddingBagMode, padding_idx: Option<usize>) -> Result<Var<'t>, AutodiffError>`（`self` は weight `[num_embeddings, D]`・`ids` は `[B, L]`・戻り値は `[B, D]`。`nn::embedding_bag_forward` への 1 行委譲） |
| `compat::Sequential::add_dropout2d` | `fn(self, p: f32) -> Result<Self, AutodiffError>` |
| `compat::Sequential::add_alpha_dropout` | `fn(self, p: f32) -> Result<Self, AutodiffError>` |
| `compat::Sequential::add_embedding_bag` | `fn(self, num_embeddings: usize, embedding_dim: usize, mode: EmbeddingBagMode, padding_idx: Option<usize>, seed: u64) -> Result<Self, AutodiffError>` |
| `fandhe_ai::EmbeddingBagMode` | ルートへ `pub use fandhe_ai_autodiff::nn::EmbeddingBagMode;` 1 行 |

判断:

- 3 つの `Var` メソッドの本体は、層（`Dropout2d`／`AlphaDropout`／`EmbeddingBag`）の `forward` と共有する
  `pub(crate)` の forward 関数（`nn/dropout.rs`・`nn/embedding_bag.rs`）へ 1 行で委譲する。`Var` 経由は層の `new()` 検査を通らない
  ため、`p` の検査（有限かつ `[0, 1]`）・rank 検査・`padding_idx` の範囲検査を共有 forward 側へ移し、tape 操作より前に終える
  （`Err` で孤児ノードを残さない。autodiff の unit test で固定）。層の挙動・数値は変えない。
- `EmbeddingBagMode` は `add_embedding_bag`／`Var::embedding_bag` の引数型で、公開しないと呼び出せないため、承認した名前から
  必然的に要る最小限の型公開としてルートへ再エクスポートした（先例: #2527 の `GlobalPoolMode`・#1757 の `InterpolateMode`）。
  `Dropout2d`／`AlphaDropout`／`EmbeddingBag`／`EmbeddingBagVars` の層型は再エクスポートしない（未承認のまま）。
- `Var` には可変長 bag（offsets）用のメソッドを公開しない（§6 に無い。`forward_with_offsets` は層型側のみ）。

### 8.2 学習・常駐・保存経路

- `Dropout2d`／`AlphaDropout` は無状態層で、`bind`／`trainable_*`／`apply_parameters` は既存走査のまま通過する。
  `set_training`／`eval()` は `Dropout` と同じく層へ伝播する（`Module::training` を保持）。常駐経路は `Dropout` と同じく
  `Module::forward` で処理されるため通過する（CPU の `predict_resident` が eval で `predict` と bit 一致することをテストで固定）。
- `EmbeddingBag` は `weight` 1 件を学習パラメータとして追跡する（`Sequential::bind` が `EmbeddingBagVars` を層順に収集し、
  `SequentialVars::forward` は bind 済み vars で `forward_from_var` を呼ぶ〈`Module::forward` は葉を作り直して勾配が失われるため使わない〉。
  `trainable_vars`／`trainable_grads` は weight を 1 件返し、`first_untracked_parametric_layer` の対象から外す）。入力は
  `add_embedding` と同じく id を f32 で詰めた `Tensor<f32>`（厳格に整数 id へ変換）。`contains_resident_unsupported_layer` に
  加え、常駐経路（`init_device_param_store`／`forward_resident`／`predict_resident`）は `Unsupported` で fail-closed に拒否する
  （テストで 3 入口とも固定）。
- `save_model`／`load_model`: kind `dropout2d`（`p`）・`alpha_dropout`（`p`）・`embedding_bag`（`num_embeddings`・`embedding_dim`・
  `mode`: `"sum"`／`"mean"`／`"max"`・`padding_idx`）を追加（allowlist 42 → 45・`format_version` 不変）。`EmbeddingBagMode` は
  `#[non_exhaustive]` のため、未知 variant は保存前に `UnsupportedModel` で拒否し dir に副作用を残さない。dropout 2 種は
  `Dropout` と同じく層とモデルの training 食い違いを保存時に拒否する。
- `Module` の `as_*` フックは追加していない（`as_embedding_bag` は #2161 で既存。`save_model`／`load_model` は `LayerSpec` だけを根拠にし、
  crates.io 公開済み `fandhe-ai-autodiff` の trait 面を広げないため。#2522・#2526・#2527 と同じ）。ONNX export は `UnsupportedLayer`
  で fail-closed（テストで固定）。

### 8.3 ガード反転

- `DropoutEmbeddingBagHoldDoctestGuard` から `Sequential`／`Var` のプローブと `EmbeddingBagMode` の型プローブを撤去し、
  層型 4 種と自由関数 3 個の衝突プローブだけを残した。
- `api_surface.rs` の否定テスト `compat_sequential_does_not_expose_dropout_embedding_bag_add_methods`（と自己テスト）を削除し、
  正ガードを新設した: `workspace_declares_dropout_embedding_bag_facade_fn_names_only_in_approved_locations`（定義元集合）・
  `var_dropout_embedding_bag_methods_are_thin_delegations`・`var_dropout_embedding_bag_methods_are_reachable_via_facade_only`・
  `compat_sequential_dropout_embedding_bag_add_methods_have_approved_signatures`（＋自己テスト）・
  `facade_reexports_embedding_bag_mode_only_in_approved_shape`（＋自己テスト。層型が再エクスポートされないことも検査）。
  固定文言 `DROPOUT_EMBEDDING_BAG_HOLD_PROBE_BODY` は縮小後の doctest に合わせて更新した。

### 8.4 保留継続

層型（`Dropout2d`／`AlphaDropout`／`EmbeddingBag`／`EmbeddingBagVars`）の facade 再エクスポート・自由関数での公開・可変長 bag
（offsets）の `Var`／`Sequential` 経路・`per_sample_weights`／`max_norm`・Dropout2d の 3D 入力・GPU 専用カーネル・ONNX export の
対象層の拡大。CUDA／Metal 実機 parity は
`docs/perf/logs/compat-sequential-dropout-embedding-bag-2528/README.md` へ申し送り（未実測）。
