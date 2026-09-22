# `nn::Module` trait と `ModuleList` の facade 公開設計判断記録（#2132）

イシュー #2132「`nn::Module` trait と `ModuleList` の facade 公開設計判断記録」。親 #2131（Phase 5「PyTorch／TF 置き換えの API 網羅」5-A 基盤）。後続 #2133（実装）の前提となる。

本ドキュメントは **コード変更を伴わない設計記録**を成果物とする。`crates/**`・`CLAUDE.md`・`docs/spec/`（正本 submodule）・`Cargo.toml`／`Cargo.lock`・tolerance／baseline・ガードレール閾値・CI／hooks はいずれも変更しない。

基準コミット: `ae3e0fe0`（本ブランチ作成時点の `origin/main`。2026-09-22）。

## 0. 結論・段階

本イシューは docs のみ・**段階 0**（facade 公開面は不変）。

`nn::Module` trait は `dyn Module` として object safe である（§3。`ModuleList` が既に `Vec<Box<dyn Module>>` を保持してコンパイル・テスト済みのため実証済み）。しかし **#2133 が想定する「素の再エクスポート」（`pub use fandhe_ai_autodiff::nn::{Module, ModuleList}`）は、REQ-12 以前に「ユーザー定義層を facade だけで書けるようにする」という目的自体を満たさない**（§1.3）。理由は `Module::forward` が生の `fandhe_ai_autodiff::Tape` を引数に取り、facade 利用者は `fandhe-ai` のみの依存では `impl Module for MyLayer` を書けないためである。

推奨案は **案 B**（facade 独自の薄い `Module` trait と `ModuleList`／`Sequential` コンテナを新設し、内部型・`BackendOps`・生 `Tape` を一切露出しない構成。§5・§6）。ただし採否・実装形は承認事項（§10）であり、本 doc では決定しない。**#2133 の実装形は本 doc の承認結果に従って再確定する必要がある**（本イシューでは #2133 を編集しない。申し送りは PR 本文と summary に記す）。

## 1. 背景

### 1.1 要件（受入基準の構造化要約）

PyTorch の `nn.Module` サブクラス相当（ユーザー定義層）を facade から扱えるようにするための設計判断記録を作る。成果物は次を満たす:

1. `nn::Module` trait の object safety（`dyn Module` が成立するか）を確認して記録する
2. sealed vs open の判断基準（trait 拡張・required method 追加の後方互換リスク）を明記する
3. `ModuleList`／内部 `Sequential` を facade へ再エクスポートするか・別 API へ統合するかの方針を確定（推奨案と承認事項として提示）する
4. crates.io 出荷済み `fandhe-ai =0.9.0` の既存公開メソッド（`set_parameter`／`named_parameters` 等の defaulted メソッドを含む）との互換性を表で示す
5. 「背景・設計案・承認事項」構成で書く

契約（イシュー共通）: 0.9.0 公開 API 非破壊／依存追加・新規 `unsafe`・facade 公開面拡張・spec 提案はユーザー承認事項として列挙し承認前に実施しない／`nn::Module` 実装本体の変更と required method の新規追加はスコープ外。

### 1.2 解決する課題

現状、facade（`fandhe_ai`）利用者がモデルを組む手段は `compat::Sequential::add_*`（Linear・活性化・Conv・Norm・Pooling 等の閉集合ビルダー）のみで、`Var` 演算を自由に組み合わせた独自層を定義し、コンテナに積む経路が存在しない。内部クレート `fandhe_ai_autodiff::nn` には `Module` trait・`ModuleList`／`Sequential`（#1759）が実装済みだが、facade は意図的に非公開としてきた（`crates/autodiff/src/nn/container.rs` モジュール doc「facade への非公開」・`crates/facade/src/nn/rnn.rs` doc「`Module` trait: … `BackendOps` を露出させずには到達できない（REQ-12）」）。本 doc はその非公開判断を再検討し、公開形を確定するための材料と推奨を記録する。

### 1.3 調査で判明した決定的な構造制約

- `Module::forward<'t>(&self, tape: &'t Tape, input: &Var<'t>)`（`crates/autodiff/src/nn/module.rs:81`）の `Tape` は**生の `fandhe_ai_autodiff::Tape`** である
- facade の `Tape` は newtype `pub struct Tape(pub(crate) fandhe_ai_autodiff::Tape)`（`crates/facade/src/lib.rs`）で、生 `Tape` は再エクスポートしない（REQ-12。`crates/facade/tests/api_surface.rs::facade_does_not_reexport_tape_or_backend_ops`〈:70〉・`compat_public_functions_do_not_accept_raw_autodiff_tape_argument`〈:164〉が機械固定）。`Var<'t>` のフィールド `tape: &'t Tape` は private でアクセサなし（`crates/autodiff/src/var.rs:109-110`）
- Rust の `impl Trait for T` はメソッド引数型を明示記述しなければならないため、**`fandhe-ai` のみに依存する利用者は、素の `pub use fandhe_ai_autodiff::nn::Module` では `impl Module for MyLayer` を書けない**（生 `Tape` を名指しできない）。`fandhe-ai-autodiff` を直接依存に足せば書けるが、それは `docs/compat-api-scope.md` §0 のサポート境界外
- したがって #2133 本文が想定する「`pub use autodiff::nn::{Module, ModuleList}` の素の再エクスポート」は、REQ-12 以前に「ユーザー定義層」という目的自体を満たさない。これが本 doc の可否判断の核であり、#2133 の実装形は本 doc の承認結果に従って再確定する必要がある

## 2. 現状のコード事実（基準コミット `ae3e0fe0`）

| 事実 | 出典 |
|---|---|
| `pub trait Module` は supertrait なし・sealed なし。required method は `forward` の 1 件のみ。他はすべて defaulted | `crates/autodiff/src/nn/module.rs:79-595` |
| defaulted メソッド: `forward_host(&dyn BackendOps, &Tensor<f32>)`〈:107〉・`supports_forward_host`〈:145〉・`as_linear`〜`as_transformer_encoder_layer`（`_mut` 込み。戻り値は `Option<&Linear>` 等の内部型）〈:159-317〉・`as_relu`〈:330〉・`is_pooling`〈:343〉・`set_training`〈:368〉・`training`〈:374〉・`named_parameters`〈:393〉・`set_parameter`〈:429〉・`state_dict`〈:440〉・`load_state_dict`〈:510〉 | 同上 |
| `forward_host` の doc は「`fandhe-ai-autodiff` は crates.io 公開クレートであり、本メソッドは非破壊拡張（デフォルトメソッド追加。外部実装者の既存 `impl Module` を壊さない）」と明記 → 内部 trait は既に **open + defaulted 限定拡張**の運用 | `module.rs:93-100` |
| `ModuleList { modules: Vec<Box<dyn Module>>, training: bool }` が存在しコンパイル済み → **object safety は実証済み**。`Sequential { inner: ModuleList }`・`Sequential::add<M: Module + 'static>`〈container.rs:250〉 | `crates/autodiff/src/nn/container.rs:59,218,250` |
| `dyn Module` に `Send`／`Sync` 境界なし（`Box<dyn Module>` は `Send` でない） | 同上 |
| `container.rs` doc「facade への非公開」: 「`Module` trait 自体が非公開のため使途がなく、`api_surface.rs` の走査対象を増やさない」 | `container.rs:25-29` |
| `nn/rnn.rs` doc: 「`Module` trait: … `forward_host`（`&dyn BackendOps` 引数を取る）は facade へ `BackendOps` を露出させずには到達できない（REQ-12）」として再エクスポート対象から除外（#1955） | `crates/facade/src/nn/rnn.rs` |
| facade `nn/mod.rs` の公開宣言は `pub mod rnn;` の 1 件のみ。`api_surface.rs::nn_mod_declares_only_rnn_submodule`〈:3323〉が完全一致で固定（`nn/mod.rs` へ何か足すと fail） | `crates/facade/src/nn/mod.rs`・`api_surface.rs:3320-3335` |
| `api_surface.rs` の ONNX 走査 `FORBIDDEN_INTERNAL_TYPE_SUBSTRINGS` に `"nn::Module"` が含まれる（走査対象は `src/interop/onnx.rs` のみ） | `api_surface.rs:1634,2111` |
| facade が再エクスポートする autodiff 型は `AutodiffError`／`Gradients`／`Var`／`LinearVars`／`VarHostView`／`QrVars`／`SvdVars` 等。`Linear`・`Conv2d`・`LayerNorm` 等の層型は非再エクスポート | `crates/facade/src/lib.rs:161-176` |
| `compat::Sequential::forward(&self, tape: &crate::Tape, …)` は facade `Tape` を取り、内部で `tape.0` を `nn::Sequential::forward` へ渡す。`layers()` は `pub(crate)` | `crates/facade/src/compat/sequential.rs:612,1044` |
| sealed trait の先例: `CastElement`（利用者が実装してはならない型境界。`f32`／`f64`／`i32`／`i64`／`bool` の 5 型に閉じる） | `docs/compat-api-scope.md:232` |
| 内部限定実装＋facade 公開は未承認のまま否定ガード固定の先例: custom autograd Function は §12.5 (b) 未承認（`Tape::custom` は内部クレート限定・facade `Tape` へ転送メソッドを追加しない限り到達不能） | `docs/autodiff-custom-function-decision.md` §12.5 |
| 純再エクスポート＋`Tape` 委譲メソッドの先例: `fandhe_ai::nn::rnn`（#1955。`Tape::rnn_forward_seq` 等は `&self.0` を渡すだけの薄い委譲） | `crates/facade/src/nn/rnn.rs`・`lib.rs:471-504` |
| `docs/compat-api-scope.md` §1.2／§1.3 に「ユーザー定義 Module」行は存在しない（Phase 5 #2131 は Tier 表とは別系統）。§5 に「#NNNN の設計記録は `docs/...` として完了した」形式の docs-only 完了記録の先例あり（#1652・#1775・#1962 等） | `docs/compat-api-scope.md` §1.2・§5 |
| 兄弟 issue: #2134（`named_modules`／`parameter_count`／`ModuleDict`／`summary` を autodiff trait へ defaulted 追加し facade `nn/mod.rs` で再エクスポート予定）・#2137（`freeze`／`set_requires_grad` 同型）・#2140（`nn::init` 再エクスポート）・#2138/#2139（hooks） | `gh issue view` |

## 3. object safety 確認

判定基準（Rust の object safety 規則）: (a) 型パラメータを持つメソッドがない（ライフタイムパラメータのみは可）、(b) `Self` を値で返す／受けるメソッドがない、(c) `where Self: Sized` が付いた required method がない、(d) 関連定数・関連型がない、(e) supertrait に `Sized` がない。

| メソッド | (a) | (b) | (c) | 判定 |
|---|---|---|---|---|
| `forward<'t>(&self, tape: &'t Tape, input: &Var<'t>) -> Result<Var<'t>, AutodiffError>` | ライフタイムのみ・可 | 値渡しなし | なし | OK |
| `forward_host(&self, ops: &dyn BackendOps, input: &Tensor<f32>) -> Result<Tensor<f32>, AutodiffError>` | なし | なし | なし | OK |
| `as_linear(&self) -> Option<&Linear>` 等（`_mut` 込み 8 種） | なし | 参照のみ | なし | OK |
| `set_training(&mut self, training: bool)`／`training(&self) -> bool` | なし | なし | なし | OK |
| `named_parameters`／`set_parameter`／`state_dict`／`load_state_dict` | なし | なし | なし | OK |

`Module` trait 自体に supertrait はなく `Sized` 境界も持たない。よって `dyn Module` は成立する。

実証根拠: `ModuleList { modules: Vec<Box<dyn Module>>, .. }`（`container.rs:59-60`）が既にコンパイル・テスト済み（`cargo test -p fandhe-ai-autodiff --lib container` を本判断記録作成時に実行し 13 件 pass を確認済み）。

既知事項:
- `dyn Module` は `Send`／`Sync` 境界を持たない。`Box<dyn Module>` はマルチスレッド共有を前提としない（DDP 等は別イシュー #1628 の対象）
- 既定で `'static`（`Sequential::add<M: Module + 'static>`。`container.rs:250`）

## 4. sealed vs open の判断基準

(a) **利用者が実装する trait は sealed にできない**。sealed trait は「利用者がこの型境界を実装してはならない」契約に使う（先例: `CastElement`。`docs/compat-api-scope.md:232`）。`nn::Module` は逆に「利用者が独自層として実装する」ことが目的であるため、sealed 化は目的と矛盾する。よって **`Module` は open trait**とする。

(b) **open trait の後方互換規則**（内部 `autodiff::nn::Module` は既にこの運用。`module.rs:93-100`）:
- **defaulted メソッド追加＝非破壊**（既存 `impl Module` を壊さない。`forward_host` 追加時〈#1028／#1760〉に確立済み）
- **required メソッド追加・既存シグネチャ変更・supertrait 追加＝破壊的変更**（semver major。crates.io 公開済み `fandhe-ai-autodiff =0.9.0` では不可）
- 内部型を返す defaulted フック（`as_linear` 等）は、`fandhe-ai-autodiff` の `Module` trait 自体としては open trait のままであり、外部実装者が型システム上オーバーライドすることを妨げない（型システムで強制される禁止契約ではない）。ただし戻り値の内部型（`Linear` 等）は外部クレートから構築できないため、外部実装者が意味のある値を返すオーバーライドを書くことは実用上できない。本 doc がこれらのフックを facade trait 側へ持ち込まない根拠は、autodiff 側へ新たな禁止契約を課すことではなく、あくまで REQ-12（`BackendOps`・生 `Tape`・内部型の facade 非露出）適合のみである（§5 案 B）

(c) `fandhe-ai-autodiff 0.9.0` 自体も crates.io 公開済みのため、autodiff 側 `Module` trait のシグネチャ変更（例: `forward` の tape 引数差し替え）は選択肢から除外する。

## 5. 公開形の案比較

判定軸: (1) 利用者が facade 依存のみで独自層を実装できるか、(2) REQ-12（`BackendOps`・生 `Tape` 非露出）適合、(3) 内部型露出の有無、(4) 0.9.0 非破壊、(5) 薄いラッパー原則（REQ-9）適合、(6) `api_surface.rs` への影響、(7) 新規 `unsafe` 要否。

| 案 | 内容 | 判定 |
|---|---|---|
| **A: 素の再エクスポート** | `pub use fandhe_ai_autodiff::nn::{Module, ModuleList, Sequential}` | (1) **不成立**（§1.3。生 `Tape` 引数のため利用者は `impl Module` を書けない）。`forward_host(&dyn BackendOps)`・`as_linear() -> Option<&Linear>` 等の内部型が facade 到達 trait の署名に露出し（2)(3) にも抵触。#1955（rnn）が `Module` を再エクスポート対象から明示除外した判断（§2 表）とも矛盾する → **不採用** |
| **B: facade 側 trait ＋ facade 側コンテナ**（推奨） | `fandhe_ai::nn::Module { fn forward<'t>(&self, tape: &'t fandhe_ai::Tape, input: &Var<'t>) -> Result<Var<'t>, AutodiffError>; }` を required とし、`named_parameters`／`set_parameter`／`state_dict`／`load_state_dict`／`set_training`／`training` を autodiff 側と同一意味論・同一命名の defaulted メソッドとして持つ薄い trait。`forward_host`・`as_*`・`as_relu`・`is_pooling` は載せない。`fandhe_ai::nn::{ModuleList, Sequential}` は `Box<dyn fandhe_ai::nn::Module>` を保持する facade 側の薄いコンテナ | (1) 成立（facade `Tape::var` と `Var` 演算だけで独自層を書ける）。(2)(3) 適合（内部型非露出）。(4) 適合（facade への追加のみ）。(5)(6)(7) は §6 で詳述 → **推奨候補** |
| **C: 案 A ＋ 生 `Tape` の再エクスポート／型エイリアス** | `pub type Tape = fandhe_ai_autodiff::Tape;` 等 | `Tape::new_with_ops` 等が到達可能になり REQ-12 違反・`facade_does_not_reexport_tape_or_backend_ops` 否定ガードと衝突 → **不採用** |
| **D: 段階 0 継続** | 非公開のまま | 親 #2131 の目的（ユーザー定義層）が未達のまま。比較基準線として記載 |

## 6. 推奨案（案 B）の詳細

配置は `crates/facade/src/nn/{module.rs, container.rs}` 相当（既存 `crates/facade/src/nn/rnn.rs` と並列）。名前衝突なし（`compat::Sequential`〈`crate::compat::sequential`〉と `nn::Sequential`〈`crate::nn::container`〉はパスが異なる。内部クレート `fandhe_ai_autodiff::nn::Sequential` と `fandhe_ai_facade::compat::Sequential` が既にパスで区別されているのと同型。`container.rs:18-23`）。

defaulted メソッド（`named_parameters`／`set_parameter`／`state_dict`／`load_state_dict`／`set_training`／`training`）は autodiff 側と同じ意味論・同じ fail-closed 検証（`set_parameter` は未知キー・shape 不一致で型付き `Err`、`load_state_dict` は strict two-pass ＋ ベストエフォート・ロールバック方式――パス 1 で全キーを検証してから適用し、パス 2 の途中失敗時は既に適用済みのキーを逆順で元の値へ戻す。ロールバック自体が失敗した場合（外部実装が状態を持つ・非決定的挙動を返す等）は完全な原子性を構造的には保証しない。`module.rs:452-510` のドキュメンテーションコメントが正）を鏡写しにする方針とする。

`api_surface.rs` への波及（#2133 で必要になる見込み）:
- `nn_mod_declares_only_rnn_submodule`〈:3323〉の期待集合を `["rnn"]` → `["rnn", "module", "container"]`（または同等の構成）へ更新
- `nn_rnn_module_is_pure_reexport` は rnn 専用の純再エクスポート検査であり、facade 独自 trait を持つ `nn::module`／`nn::container` には適用できない。別途「facade trait の defaulted メソッド集合が承認済み集合と一致する」検査が必要
- ONNX 走査の `FORBIDDEN_INTERNAL_TYPE_SUBSTRINGS` の `"nn::Module"`〈:1634,2111〉は `src/interop/onnx.rs` 限定の走査のため当面非衝突だが、`fandhe_ai::nn::Module` という新規公開名との混同を避ける命名レビューが必要

**橋渡しの非対称性**（決めていない論点。§10 承認事項 4）: 組み込み層（autodiff `Module` 実装済み型）→ facade trait は facade 内で `tape.0` を渡せるため安全に実装可能。一方、facade 利用者の独自層 → autodiff コンテナ（`compat::Sequential` へ積む等）は `&fandhe_ai_autodiff::Tape → &fandhe_ai::Tape` 方向の変換が必要で、次のいずれかが要る（`.claude/rules/coding-rust.md` の unsafe 統制方針〈FFI 境界・CPU SIMD intrinsics 等の必要最小限に限定〉に従い、安全な代替が既にある `#[repr(transparent)]` ＋ unsafe 参照キャスト案は設計候補から除外する）:
1. `var` 系メソッドのみを持つ借用ハンドル型（例: `TapeRef<'t>(pub(crate) &'t fandhe_ai_autodiff::Tape)`）を新設（安全だが第 2 のハンドル型が増える）
2. 第 1 段では橋渡し自体を対象外とし、facade 側 `nn::Module` 実装層は facade 側 `nn::{ModuleList, Sequential}` の中でのみ完結させる（`compat::Sequential` との相互運用は行わない）

利用例（擬似コード。実装は行わない）:

```rust
struct MyBlock { linear: /* 何らかの facade 層 */ }

impl fandhe_ai::nn::Module for MyBlock {
    fn forward<'t>(&self, tape: &'t fandhe_ai::Tape, x: &fandhe_ai::Var<'t>)
        -> Result<fandhe_ai::Var<'t>, fandhe_ai::AutodiffError> {
        // tape・x のみで既存 Var 演算を合成する
        todo!()
    }
}
```

## 7. 0.9.0 互換性表

| 公開面 | 案 B での扱い |
|---|---|
| `compat::Sequential::{set_training, train, eval, training, named_parameters, state_dict, load_state_dict, add_*, forward, predict, bind, …}` | 不変 |
| `fandhe_ai::nn::rnn`（`Rnn`／`Lstm`／`Gru` 等） | 不変 |
| root 再エクスポート群（`AutodiffError`／`Gradients`／`Var`／`LinearVars`／`VarHostView`／`QrVars`／`SvdVars` 等） | 不変 |
| autodiff 側 `Module` の defaulted メソッド（`set_parameter`〈#1752〉・`named_parameters`／`set_training`／`training`〈#1758〉・`state_dict`／`load_state_dict`〈#1752〉・`forward_host`／`supports_forward_host`〈#1028／#1760〉） | 本判断で一切変更しない |
| facade 新規公開面（`fandhe_ai::nn::{Module, ModuleList, Sequential}` 等） | **追加のみ**（semver minor 相当。既存公開面の削除・シグネチャ変更なし） |

## 8. 兄弟 issue との整合

- **#2133（実装）**: 想定形（素の `pub use`）は§1.3 の構造制約により再確定が必要。本 doc の承認結果（案 B の採否・橋渡し方式）を前提として実装形を決め直す
- **#2134／#2137**: 「`named_modules`／`parameter_count`／`ModuleDict`／`summary`」「`freeze`／`set_requires_grad`」を autodiff trait へ defaulted 追加し facade `nn/mod.rs` で再エクスポートする計画は、案 B 採用時には「autodiff trait への defaulted 追加」＋「facade trait 側 defaulted メソッドへの鏡写し追加」の 2 段構成へ読み替えが必要
- **#2140**（`nn::init` 純再エクスポート）: `Module` trait に依存しない独立の再エクスポートのため非衝突
- **#2138／#2139**（hooks）: `Var`／`Tape` レベルで独立のため非衝突

## 9. スコープ外

- `nn::Module`（内部クレート `autodiff::nn::module`）実装本体の変更
- required method の新規追加
- `compat::Sequential` への `add_module` 追加
- 組み込み層型（`Linear` 等）の facade 公開（別途承認が必要な独立の事項）
- GPU 専用カーネルの追加・変更

## 10. 承認事項（列挙のみ・実施しない）

1. 案 B の採否（または案 A／D の指名）
2. facade trait の defaulted メソッド集合（`named_parameters`／`set_parameter`／`state_dict`／`load_state_dict`／`set_training`／`training` の 6 件）と命名契約の踏襲
3. facade 側 `ModuleList`／`Sequential` の新設可否
4. 独自層 → autodiff コンテナ橋渡しの方式（§6「橋渡しの非対称性」の 2 択: 借用ハンドル型／第 1 段では対象外。unsafe 参照キャスト案は unsafe 統制方針により設計候補から除外済み）
5. #2133 本文の実装形更新
6. #2134／#2137 への鏡写し要件（facade trait 側への defaulted メソッド追加を伴う場合）

## 11. 出典

- `docs/compat-api-scope.md` §0・§5
- `docs/public-api-design.md`
- `docs/facade-onnx-import-exposure-decision.md`
- `docs/facade-safetensors-exposure-decision.md`
- `docs/autodiff-custom-function-decision.md` §12.5
- `crates/autodiff/src/nn/{module.rs, container.rs}`
- `crates/facade/src/{lib.rs, nn/mod.rs, nn/rnn.rs, compat/sequential.rs}`
- `crates/facade/tests/api_surface.rs`
- spec REQ-9／REQ-12（`docs/spec/04-requirements.md`）
