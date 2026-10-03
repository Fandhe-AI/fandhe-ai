# bool 出力の比較・logical 演算・masked_select の設計判断記録

イシュー #2141（親 #2131）。`docs/unique-facade-exposure-decision.md`・
`docs/autodiff-custom-function-decision.md` と同型の記録。

## §0 結論

bool を返す比較 6 種（`gt_bool`／`ge_bool`／`lt_bool`／`le_bool`／
`eq_bool`／`ne_bool`）・logical 3 種（`logical_and`／`logical_or`／
`logical_not`）・`masked_select` を、**`fandhe_ai_autodiff` のうち facade
が再エクスポートしない自由関数モジュール `bool_ops`**（`crates/autodiff/
src/bool_ops.rs`）として実装した（案 C。§3 参照）。`Var` に inherent の
`pub fn` は追加していない（#2141 時点の記述。比較 6 種・`masked_select` の
7 件は #2510 で `Var` の委譲メソッドとして追加済み。logical 3 件は
未公開のまま #2594 で扱う。§6.1 参照）。新規 `Op`・`BackendOps` メソッド・VJP・tape
ノードは追加していない。facade 公開（`Var` への委譲メソッド追加）は
承認待ちのまま対象外とし、`crates/facade/src/lib.rs::
VarBoolOpsHoldDoctestGuard`（正のプローブ doctest）と
`crates/facade/tests/api_surface.rs` のソース走査・workspace インベント
リ（4 テスト）で多層固定している。

## §1 背景

既存の比較 6 種（`Var::gt`〜`ne`。`crates/autodiff/src/var.rs`）は f32 の
`0.0`／`1.0` マスクを `Var`（勾配ゼロの tape ノード）として返す。
`Var::where_cond`（`var.rs:4031`）・`Var::masked_fill`（`var.rs:4093`）は
既に条件として `&Tensor<bool>` を受け取るため、bool 出力比較の結果を
そのまま接続できる。

## §2 契約の出典

- f32→bool は `v != 0.0`（NaN→true・−0.0→false）。GPU は u8 で転送し、
  ホストで `v != 0` により実体化する（`docs/tensor-core-cast-design.md`）
- 非微分・動的 shape の出力は tape に記録せず detached な `Tensor` を
  返す（`docs/unique-facade-exposure-decision.md` §3 案 A）

**#2061 に関する注記**: イシュー本文は「#2061 で bool 出力型・非微分
契約が確定（前提）」としているが、#2061 の成果物
`docs/autodiff-var-dtype-multiplexing-design.md` は `Var<T>` の dtype
多重化（段階 0）の記録であり、bool 出力型や非微分契約そのものは定めて
いない。本実装が実際に拠る契約は上記 2 点である。

`Var::gt` の doc にあった「bool dtype 出力は #1613 の対象」注記は、
`fandhe_ai_autodiff::bool_ops::gt_bool` 等（本イシュー）への参照へ更新
した。挙動・シグネチャは変更していない。

## §3 API 配置の案比較

- **案 A（`Var` の inherent メソッド）**: facade へ即座に到達する。
  リポジトリ慣行（`docs/unique-facade-exposure-decision.md` §4）では
  既存の `Var` 再エクスポートを通るメソッド追加は `docs/compat-api-
  scope.md` §5（範囲拡張手続き）の対象外だが、本イシュー本文が facade
  公開面を承認事項として明示列挙し、親 #2131 がこのツリーに限り
  「設計判断記録 → 承認 → 実装」の 2 段階を定めている。#2141 に承認
  コメントは無い（2026-09-24 時点）ため、本イシューでは採用しない
- **案 B（拡張 trait）**: メソッド構文は使えるが、trait 名の分だけ
  公開面が増える。承認後に `Var` の inherent メソッドへ移すと入口が
  二重に残るため不採用
- **案 C（自由関数）**: 採用。承認後の撤去・委譲が単純
- **案 D（`Tensor<bool>` の inherent メソッド）**: `fandhe_ai::Tensor`
  も facade が再エクスポートする型（`crates/facade/src/lib.rs:194`）
  のため、案 A と同じ扱い

## §4 数値契約

- 比較 6 種は IEEE 754 準拠（`NaN` を含む比較は `eq` を含め常に偽・
  `ne` のみ真。`-0.0 == +0.0` は真）。既存の `scalar_binary_with_
  fallback`（比較カーネル）→ `cast_from_f32_with_fallback::<bool>`
  （cast カーネル）の合成で実装しているため、丸めは一切入らない
- logical 3 種はホスト常駐データのみで計算する（GPU 専用カーネルは
  対象外）
- `masked_select` は値をコピーするだけで算術を含まないため、NaN の
  payload も含めて bit 単位で保たれる

## §5 PyTorch との差異

- **`masked_select` は非微分**: PyTorch の `torch.masked_select` は
  微分可能だが、本実装は非微分（`Var::unique`／`argmax` と同型の設計
  判断——出力 shape が実行時にしか分からない動的 shape のため、既存の
  tape ノード表現に乗らない）
- logical 3 種は bool 入力のみを受け取る（`Var`〈f32〉入力の logical
  版——非ゼロと NaN を真とみなす PyTorch の float 入力版相当——は
  対象外。利用側が `Var::cast::<bool>()` で bool 化してから呼ぶ）

## §6 承認事項（未承認として列挙）

1. facade 公開: `Var` への委譲メソッド 7 件（比較 6 種・`masked_select`）
   および logical 3 件の公開形（`Var` の関連関数にするか facade 直下の
   関数にするか）。**7 件は #2510 で適用済み（ルート #2499 一括承認）。
   logical 3 件の公開形は未決のまま #2594**
2. 微分可能な `masked_select`
3. `Var` 入力の logical 版
4. GPU 専用カーネル（ホストでの bool 化・logical・select の GPU 化）

## §7 スコープ外

- `unique`・索引を返す系（`argmax`／`argmin` 等）・in-place 代入は本
  イシューの対象外（既存実装のまま）
- 兄弟イシュー #2147（`any`／`all`）とは関数名・モジュールを共有しない

## §8 実装記録

- `crates/autodiff/src/bool_ops.rs`（新規）: 比較 6 種・logical 3 種・
  `masked_select`・単体テスト 16 件（`Tape::new()`／NaiveOps 経由）
- `crates/autodiff/src/lib.rs`: `pub mod bool_ops;` を追加
- `crates/autodiff/src/var.rs`: `Var::gt` doc の #1613 注記を更新
  （挙動・シグネチャは不変）
- `crates/facade/src/lib.rs`: `VarBoolOpsHoldDoctestGuard`（正のプローブ
  doctest。`VarCustomHoldDoctestGuard`／`NnModuleHoldDoctestGuard` と同型）
- `crates/facade/tests/api_surface.rs`: `bool_ops_hold_doctest_globs_all_
  pub_modules`・`bool_ops_hold_doctest_probe_body_matches_fixed_contract`・
  `facade_does_not_reexport_or_declare_bool_ops`・
  `workspace_declares_bool_ops_fn_names_only_in_autodiff_bool_ops`
  （4 テスト。手動検証: `pub use fandhe_ai_autodiff::bool_ops;` の仮追加・
  `Var` への仮 inherent `gt_bool` 追加のいずれでも doctest／ソース走査が
  fail-closed に検出することを確認済み）
- `crates/facade/tests/bool_ops_backend_parity.rs`（新規）: CPU と
  NaiveOps の bit 一致（属性なし 3 件）＋CUDA／Metal の `#[ignore]`
  （未実測。`docs/perf/logs/bool-ops-2141/README.md` 参照）
- `crates/facade/tests/bool_ops_torch_behavior.rs`（新規）: PyTorch
  文書化済み意味論からの behavior parity example 3 件（`torch.where`・
  `torch.logical_and` + `masked_select`・`torch.logical_not`/`or` の
  真理値表と NaN）

承認取得後の追随: `Var::gt_bool` 等の薄い委譲メソッド 7 件は #2510 で実施済み
（§6.1）。logical 3 件分の保留ガード（`VarBoolOpsHoldDoctestGuard`・対応する
ソース走査）は #2594 まで残置し、その時点で撤去する。

### §6.1 実装記録（#2510・ルート #2499 一括承認）

- 公開した名前: `Var::{gt_bool, ge_bool, lt_bool, le_bool, eq_bool, ne_bool}
  (&self, &Var<'t>) -> Result<Tensor<bool>, AutodiffError>`・
  `Var::masked_select(&self, &Tensor<bool>) -> Result<Tensor<f32>,
  AutodiffError>`（`crates/autodiff/src/var.rs`。本体は
  `crate::bool_ops::<name>` への 1 式委譲。非微分・tape 非記録・意味論は
  自由関数と同一）。新規の型・`pub use`・`Op`・`BackendOps`・VJP なし
- ガード反転: `VarBoolOpsHoldDoctestGuard` の `__probe_var` を logical 3 件
  のプローブへ差し替え（モジュールプローブ・Tensor／Tape プローブは維持）。
  `api_surface.rs` は定数を承認 7 件／保留 3 件へ分割し、workspace
  インベントリを `workspace_declares_bool_ops_fn_names_in_approved_places_only`
  へ改名・正ガード化（7 名は `bool_ops.rs`＋`var.rs` に各 1 件・logical は
  `bool_ops.rs` のみ・`var.rs` の 7 委譲本体のトークン一致を固定）
- 手動確認（fail-closed）: `Var` への仮 `logical_and` 追加で doctest・
  インベントリが失敗することを確認し元に戻した
- 利用テスト: `crates/facade/tests/bool_ops_var_methods.rs`（6 件・CPU）
- CUDA／Metal: 委譲のみでカーネルを新設しないため新規の `#[ignore]` テスト・
  実測申し送りは無い（既存の `bool_ops_backend_parity.rs`・
  `docs/perf/logs/bool-ops-2141/README.md` のまま）
