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
承認待ちのまま対象外とし、`crates/facade/src/lib.rs::
DropoutEmbeddingBagHoldDoctestGuard`（正のプローブ doctest）と
`crates/facade/tests/api_surface.rs` のソース走査（2 テスト＋自己
テスト）で多層固定している。

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
  `embedding_bag` の facade 公開（承認待ち）
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

## §6 承認事項（未承認として列挙）

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
