# facade への nn 層型名の公開方針（決定記録）

> **コード変更を伴わない決定記録。** `crates/**`・`docs/spec`・`Cargo.toml`／`Cargo.lock`・tolerance／baseline・ガードレール閾値は不変。
> 基準コミット: main `8d5c8560`（2026-10-12）。件数は §2 のコマンドで再実測でき、基準コミットが進むと変動する。

- イシュー: #3022（親 #3018）。D-0 から切り離された論点「`fandhe_ai::nn::*` で全層型を出すか」を 1 本で決める
- 承認: 2026-10-12 の所有者コメント https://github.com/Fandhe-AI/fandhe-ai/issues/3022#issuecomment-6111750000 （§7）

## 0. 結論

**推奨は (i)+(iv)。層型名の一括公開はしない。例外は §3 (iv) の 3 条件を満たす場合に個別承認で認める。**

- 層型名（`Linear`・`Conv2d` 等）は引き続き `fandhe_ai::nn` に出さない。機能は `compat::Sequential::add_*`・`Var` メソッドで到達する
- 既存の `Rnn` 系・`loss`・`kv_cache`・`Transformer` 系は「カテゴリ別の個別承認」の先例として維持する（変更なし）
- 保留ガード `SpatialLayersHoldDoctestGuard` が見張る 5 型（`ConvTranspose1d`・`Upsample`・`ZeroPad2d`・`Identity`・`Unflatten`）は **公開対象外として確定**し、ガードは **現状のまま維持**する（変更 0）
- 推奨は (ii) ではないため、実装 issue の起票提案は不要

## 1. 背景と要件

- `fandhe_ai::nn` が出す層型は `Transformer`・`TransformerConfig`・`TransformerDecoderLayer` の 3 つのみで、`Linear`・`Conv2d` は出していない（`crates/facade/src/nn/mod.rs:9-12,50`）
- 5 型だけを出すと、他の層との方針が不整合になる。このため全層型を対象に方針を一度に決める
- 受け入れ条件: 選択肢 2 案以上と各案の公開名数・`api_surface.rs` への影響・推奨／所有者承認コメントの引用／5 型の扱いの一意確定／コード・`docs/spec` 差分なし

## 2. 現状事実

再実測コマンド（基準コミットで実行）:

- `grep -rhE '^pub struct' crates/autodiff/src/nn | wc -l`（178）。うち `nn/optim` が 50（`.../nn/optim` で同様に実測）。層・損失等は **128**。`pub enum` は 12 でうち optim が 5（非 optim は 7）
- `grep -rhE '^impl.* Module for' crates/autodiff/src/nn | wc -l`（**73**）
- `grep -cE 'pub fn add_' crates/facade/src/compat/sequential.rs`（**65**）

| 事実 | 出典 |
|---|---|
| 「層型は一切出さない」は厳密には正しくない。facade は次をカテゴリ別の個別承認で既に出している: `nn::loss` 24 名（損失 14 構造体＋引数型 10）、`nn::rnn` の `Rnn`／`Lstm`／`Gru`／`Stacked*`／`PackedSequence` 等（型と関数で約 27 名）、`nn::init` 13 名、`nn::kv_cache` 3 名、`Transformer` 系 3 名、`ForwardHookCtx` | `crates/facade/src/nn/{mod,rnn,init,loss,kv_cache}.rs` |
| **決定的な構造制約**: 内部の層型は `autodiff::nn::Module`（`forward` が生の `autodiff::Tape` を取る）を実装し、`bind` 等も生 `Tape` を取る。facade の `Tape` は newtype（内部フィールド `pub(crate)`）で生 `Tape` は再エクスポートしない。よって層型名だけを再エクスポートしても、利用者は `forward`／`bind` を呼べない。`Rnn` 系が「型再エクスポート＋`Tape::rnn_forward_seq` 等の薄い委譲メソッド」の組で公開されているのはこのため | `docs/facade-nn-module-exposure-decision.md` §1.3、`crates/facade/src/nn/rnn.rs` 冒頭 doc |
| facade の `nn::Module::forward` は `TapeRef` と `Var` だけを取り、署名に層型名を含まない。独自 `Module` を書くのに組み込み層の型名は要らない。独自層は `Sequential::add_module` で積める | `crates/facade/src/nn/module.rs`、`crates/facade/src/compat/sequential.rs:1879` |
| 5 型に対応する機能は公開済み: `add_conv_transpose1d`・`add_upsample`・`add_zero_pad2d`・`add_identity`・`add_unflatten` と `Var::conv_transpose1d`／`unflatten`（#2521・#2522） | `docs/autodiff-spatial-layers-decision.md` §8 |
| `add_upsample` は `size` 指定のみで `scale_factor` 指定は未公開。これは型名ではなく機能の未公開 | `crates/facade/src/compat/sequential.rs:1774`、同 doc §8 残課題 |
| `nn/mod.rs` の `pub use` は `api_surface.rs` が行単位の完全一致で固定している（`nn_mod_declares_only_approved_submodules`〈:5203〉、`facade_reexports_transformer_decoder_items_only_in_approved_shape`）。`api_surface.rs` は 39,398 行、`*HoldDoctestGuard` は `lib.rs` に 111 箇所の言及がある | `crates/facade/tests/api_surface.rs`、`crates/facade/src/lib.rs:3208` |

## 3. 選択肢と影響の比較

| 案 | 内容 | 公開名の追加数 | `api_surface.rs` への影響 | 5 型・保留ガード | `Module` 実装者の体験 | 非破壊契約の対象増 |
|---|---|---|---|---|---|---|
| (i) 現状維持 | 層型名は出さず `add_*`／`FunctionalBuilder` 経由に限る | 0 | 変更なし | 5 型は対象外。ガード変更 0 | 変わらない（`Var` 演算で書く） | 0 |
| (ii) 全層型を `nn::*` で出す | `impl Module` 73 型のうち未公開の約 60 型、設定・出力型を含め 80 前後（概算）を再エクスポート | 約 80。ただし型名だけでは `forward`／`bind` が呼べず、実用にするなら各型へ `Tape` 委譲か `Module` 適合も要り、さらに増える | `nn/mod.rs` 再エクスポート固定テストの全面改修と、各層の型名プローブ・正ガードの大量追加 | 5 型を公開し、ガードを全面反転 | 型を名指しできるが生 `Tape` 制約は残る | 約 80 名が恒久的に増える |
| (iii) 公開署名に現れる型だけを出す | facade の公開署名に現れる未公開の層型だけ | 実測でほぼ 0（`nn::Module::forward` も `TapeRef`／`Var` のみ） | ほぼなし | 5 型は対象外 | 変わらない | ほぼ 0。実質 (i) と同じ |
| (iv) カテゴリ別・個別承認（現行運用の明文化） | 次の 3 条件を全て満たす場合に限り、個別に承認して出す。(a) その型を名指ししないと使えない操作がある、(b) facade だけで呼べる経路（`Tape` 委譲メソッド等）が同時に承認される、(c) 公開名と `api_surface.rs` の正ガードを同一 PR で追加できる | 承認ごとに必要最小限 | 承認ごとに正ガードを同時追加 | 5 型は (a) を満たさず対象外 | 変わらない | 承認した名前だけ |

## 4. 推奨

**(i) を基本とし、(iv) の 3 条件を「例外を認める基準」として明文化する案（(i)+(iv)）を推奨する。**

- (ア) 型名だけを出しても生 `Tape` 制約で使えず、公開面の増加に見合う利用価値がない
- (イ) 独自 `Module` は `Var` 演算で書け、署名に層型名が要らない
- (ウ) 機能は `add_*`／`Var` メソッドで既に到達できる
- (エ) 公開名 1 つごとに非破壊契約の対象が恒久的に増える
- (オ) `Rnn`・`loss`・`kv_cache`・`Transformer` 系の既存の個別承認運用と整合する
- 公開面の無断拡大は `docs/compat-api-scope.md` §5 が禁じている。(iv) の条件 (c)（正ガードの同時追加）がその歯止めになる

## 5. 5 型の扱い（一意確定）

| 型 | 機能の到達経路 | 型名で追加できること |
|---|---|---|
| `ConvTranspose1d` | `Sequential::add_conv_transpose1d`・`Var::conv_transpose1d` | なし |
| `Upsample` | `Sequential::add_upsample`（`size` 指定のみ） | なし（`scale_factor` は §6 の機能課題） |
| `ZeroPad2d` | `Sequential::add_zero_pad2d` | なし |
| `Identity` | `Sequential::add_identity` | なし |
| `Unflatten` | `Sequential::add_unflatten`・`Var::unflatten` | なし |

- **5 型は公開対象外として確定する。** 3 条件のうち (a) を満たさない
- `SpatialLayersHoldDoctestGuard`（`crates/facade/src/lib.rs:3208`）は **現状のまま維持**し、変更は 0 とする。型名プローブ 5 件が引き続き「未公開」を機械固定する

## 6. 残課題（型名の問題とは切り離す）

- `add_upsample` の `scale_factor` 指定（`UpsampleSize` の再エクスポートまたは別名メソッド）。型名ではなく機能の未公開で、`docs/autodiff-spatial-layers-decision.md` §8 の残課題に既出
- 独自 `Module` から組み込み層を呼ぶ経路。`Sequential` への `nn::Module` 実装などの新しい公開面を要し、`docs/facade-nn-module-exposure-decision.md` §22.6 の起票候補と同根
- いずれも新規の公開面を要するため別承認が必要。自動運転のためユーザー承認なしに起票せず、実装担当の報告（outOfScope）に起票提案として残す（`.claude/rules/out-of-scope-tracking.md`）

## 7. 承認の記録

- 所有者コメント https://github.com/Fandhe-AI/fandhe-ai/issues/3022#issuecomment-6111750000 （2026-10-12）の要旨は「全て推奨で」。決定記録が示す推奨形をそのまま採用してよく、記録に形が書けない論点だけ所有者へ戻す。推奨が (ii) になった場合は実装 issue の起票提案までに留める
- この承認は **推奨形への包括承認** であり、(i)+(iv) という特定案への個別承認ではない。所有者が別案を選ぶ余地は残る
- 本記録の推奨は (ii) ではないため、実装 issue の起票提案は不要。コード・`docs/spec` は変更していない
- 実際に得ていない承認は記載しない

## 8. 出典

- `crates/facade/src/nn/mod.rs`・`rnn.rs`・`init.rs`・`loss.rs`・`kv_cache.rs`
- `crates/facade/src/compat/sequential.rs`、`crates/facade/src/lib.rs:3208`、`crates/facade/tests/api_surface.rs:5203`
- `docs/facade-nn-module-exposure-decision.md`（§1.3・§22.6）、`docs/autodiff-spatial-layers-decision.md`（§8）、`docs/facade-nn-loss-structs-exposure-decision.md`、`docs/compat-api-scope.md`（§5）
