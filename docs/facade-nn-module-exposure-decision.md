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
- **required メソッド追加・既存シグネチャ変更・supertrait 追加は semver 破壊的変更である**。ただし `docs/compat-api-scope.md` §0（「`facade` が唯一のサポートされる公開 API 面であり `tensor-core`／`autodiff`／`backend-*` は内部クレート、これらを `facade` を経由せず直接利用することはサポート対象外」）の境界の下では、内部クレートの破壊的変更が一律不可という前提自体が成り立たない。先例として `Tape::new()`／`impl Default for Tape`（引数なし版）の削除という内部クレートの破壊的変更が、同じ REQ-9 2026-08-08 追記（内部クレート＝非サポート宣言）を根拠に許容されている（`docs/public-api-design.md:585`「この破壊は REQ-9 の 2026-08-08 追記…を根拠に許容するが、`!`／`BREAKING CHANGE:` 告知は省略しない」・`docs/fusion-graph-design.md` §1）。したがって本 doc が `Module` の required method 追加・既存シグネチャ変更を選択肢から外すのは crates.io 公開済みという事実そのものではなく、次の実質的な理由による: (i) `Tape::new()` の破壊時点では facade `Tape` newtype 経由が唯一の到達経路であり内部クレート直接利用が事実上存在しない構成だったのに対し、`fandhe-ai-autodiff` は crates.io 公開クレートとして誰でも `Cargo.toml` に直接追加でき `impl Module for Foo` を書ける（サポート対象外と宣言されていても技術的な破壊の影響範囲がゼロとは言い切れない）。(ii) 本イシューの契約（§1.1）が「`nn::Module` 実装本体の変更と required method の新規追加はスコープ外」と明記しており、autodiff 側 trait 自体の改修は #2132 のスコープ外である。(iii) 案 B は autodiff 側 trait を一切変更せずに REQ-12 を満たせるため、autodiff 側改修の当否そのものを判断する必要がない——この「変更不要で目的を達成できる」という関係が本 doc の推奨の実質的な決め手であり、破壊的変更が「不可能」だからではない
- 内部型を返す defaulted フック（`as_linear` 等）は、`fandhe-ai-autodiff` の `Module` trait 自体としては open trait のままであり、外部実装者が型システム上オーバーライドすることを妨げない（型システムで強制される禁止契約ではない）。ただし戻り値の内部型（`Linear` 等）は外部クレートから構築できないため、外部実装者が意味のある値を返すオーバーライドを書くことは実用上できない。本 doc がこれらのフックを facade trait 側へ持ち込まない根拠は、autodiff 側へ新たな禁止契約を課すことではなく、あくまで REQ-12（`BackendOps`・生 `Tape`・内部型の facade 非露出）適合のみである（§5 案 B）

(c) autodiff 側 `Module` trait のシグネチャ変更（例: `forward` の tape 引数を facade 互換の型へ差し替える）は §5 の案 E として比較したうえで、(b) に述べた実質的な理由——本イシューのスコープ契約（§1.1）が autodiff 側 trait 本体の変更を対象外としていること、および案 B が同変更なしで目的を達成できること——により選択肢から除外する。「crates.io 公開済みだから一律不可」という理由づけはしない（内部クレートの破壊的変更自体は `docs/compat-api-scope.md` §0 の境界の下で先例があり禁止されていない）。autodiff 側 trait 改修の当否そのものは本 doc のスコープ外の判断であり、必要になれば改めて別イシューでユーザー承認を得て検討する。

## 5. 公開形の案比較

判定軸: (1) 利用者が facade 依存のみで独自層を実装できるか、(2) REQ-12（`BackendOps`・生 `Tape` 非露出）適合、(3) 内部型露出の有無、(4) 0.9.0 非破壊、(5) 薄いラッパー原則（REQ-9）適合、(6) `api_surface.rs` への影響、(7) 新規 `unsafe` 要否。

| 案 | 内容 | 判定 |
|---|---|---|
| **A: 素の再エクスポート** | `pub use fandhe_ai_autodiff::nn::{Module, ModuleList, Sequential}` | (1) **不成立**（§1.3。生 `Tape` 引数のため利用者は `impl Module` を書けない）。`forward_host(&dyn BackendOps)`・`as_linear() -> Option<&Linear>` 等の内部型が facade 到達 trait の署名に露出し（2)(3) にも抵触。#1955（rnn）が `Module` を再エクスポート対象から明示除外した判断（§2 表）とも矛盾する → **不採用** |
| **B: facade 側 trait ＋ facade 側コンテナ**（推奨） | `fandhe_ai::nn::Module { fn forward<'t>(&self, tape: &'t fandhe_ai::Tape, input: &Var<'t>) -> Result<Var<'t>, AutodiffError>; }` を required とし、`named_parameters`／`set_parameter`／`state_dict`／`load_state_dict`／`set_training`／`training` を autodiff 側と同一意味論・同一命名の defaulted メソッドとして持つ薄い trait。`forward_host`・`as_*`・`as_relu`・`is_pooling` は載せない。`fandhe_ai::nn::{ModuleList, Sequential}` は `Box<dyn fandhe_ai::nn::Module>` を保持する facade 側の薄いコンテナ | (1) 成立（facade `Tape::var` と `Var` 演算だけで独自層を書ける）。(2)(3) 適合（内部型非露出）。(4) 適合（facade への追加のみ）。(5)(6)(7) は §6 で詳述 → **推奨候補** |
| **C: 案 A ＋ 生 `Tape` の再エクスポート／型エイリアス** | `pub type Tape = fandhe_ai_autodiff::Tape;` 等 | `Tape::new_with_ops` 等が到達可能になり REQ-12 違反・`facade_does_not_reexport_tape_or_backend_ops` 否定ガードと衝突 → **不採用** |
| **D: 段階 0 継続** | 非公開のまま | 親 #2131 の目的（ユーザー定義層）が未達のまま。比較基準線として記載 |
| **E: autodiff 側 `Module::forward` のシグネチャを facade 互換型へ改修** | `fandhe_ai_autodiff::nn::Module::forward` の `tape: &'t Tape` 引数を facade からも構築できる型（例: 抽象化した trait 境界・facade 側 newtype を autodiff が受け取れる形への逆依存）へ差し替え、`pub use` する | (1) 成立しうる（facade はそのまま素の再エクスポートで目的を達成できる可能性がある）。(2)(3) は改修内容次第。(4) は required シグネチャ変更のため semver 破壊的変更——ただし `docs/compat-api-scope.md` §0 の内部クレート境界の下で `Tape::new()` 破壊の先例（`docs/public-api-design.md:585`）があり「crates.io 公開済みだから一律不可」ではない。(7) 不要。**不採用の実質的理由**: (i) 本イシューの契約（§1.1）が `nn::Module` 実装本体の変更をスコープ外と明記しており、本 doc の決定範囲を超える。(ii) `autodiff` は `facade` に依存できない（依存方向の逆転は crate 分割の前提に反する）ため、facade 側の型を autodiff 側 trait のシグネチャへ直接持ち込むことはできず、実現には autodiff 側に facade 非依存の抽象境界（trait／ハンドル型）を新設する追加設計が要る。(iii) 案 B はこの改修なしで同じ目的を達成できるため、コスト・スコープ・影響範囲（autodiff 直接利用者・他の内部クレート呼び出し箇所すべて）の観点で劣後する → **不採用（内部抽象改修コストが見合わないため。「不可能」ではなく「案 B より高コストでスコープ外」）** |

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

**実装形の確定（#2394・2026-09-29 承認事項 4）**: 案 1（借用ハンドル型・`unsafe` なし）を採用した。`fandhe_ai::TapeRef<'t>` を `crates/facade/src/lib.rs` に `pub struct` として直接定義し（`pub use` を通さないため `facade_does_not_reexport_tape_or_backend_ops` は不変）、公開メソッドは `var`／`var_from`／`var_no_grad` のみ、crate 外の入口は `From<&Tape>` のみ、crate 内の構築は `pub(crate) fn from_autodiff`（#2397 のアダプタ用）とした。#2395 の `forward` の第 1 引数は `&'t fandhe_ai::Tape` ではなく `TapeRef<'t>` になるため、§5・§6 の擬似コードとはこの点でずれる。固定するガードは `tests/api_surface.rs` の `tape_ref_public_surface_is_exactly_var_family`・`tape_ref_declared_once_with_crate_private_field`・`tape_ref_pub_fns_do_not_return_raw_tape`（自己テスト `collect_type_impls_detects_each_category`）。最終まとめは #2403。

利用例（擬似コード。実装は行わない）:

```rust
struct MyBlock { linear: /* 何らかの facade 層 */ }

impl fandhe_ai::nn::Module for MyBlock {
    fn forward<'t>(&self, tape: fandhe_ai::TapeRef<'t>, x: &fandhe_ai::Var<'t>)
        -> Result<fandhe_ai::Var<'t>, fandhe_ai::AutodiffError> {
        // tape・x のみで既存 Var 演算を合成する
        todo!()
    }
}
```

> 確定形（#2394・#2395）: 上記コード例の第 1 引数は確定形の `tape: fandhe_ai::TapeRef<'t>`
> （値渡しの借用ハンドル。§5・§6 の擬似コードの `&'t fandhe_ai::Tape` ではない）である。

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
- **#2134／#2137**: 「`named_modules`／`parameter_count`／`ModuleDict`／`summary`」「`freeze`／`set_requires_grad`」を autodiff trait へ defaulted 追加し facade `nn/mod.rs` で再エクスポートする計画は、案 B 採用時には「autodiff trait への defaulted 追加」＋「facade trait 側 defaulted メソッドへの鏡写し追加」の 2 段構成へ読み替えが必要。**#2137 は前段（autodiff trait への defaulted 追加。`Module::freeze`／`set_requires_grad`／`requires_grad`）のみ実装済み**（`crates/autodiff/src/nn/module.rs`。`docs/autodiff-nograd-leaf-dinput-skip-decision.md`「実装記録（#2137）」参照）。facade 側鏡写しは本 doc §10 の承認（項目 1・2・6）待ちのまま未実施
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

## 12. facade 公開の保留記録（イシュー #2133）

イシュー #2133「`nn::Module` trait・`ModuleList` の facade 公開実装」の実装着手時
（2026-09-23・main HEAD `ea838b71`）に、`gh issue view 2133 --comments`・
`gh issue view 2132 --comments`（本 doc の対応 issue）・`gh issue view 2131
--comments`（親 issue）を確認したところ、いずれもコメント 0 件だった。PR #2230
（#2132 の設計記録 PR）に付いているのは github-actions（codex-review）による
自動レビューコメントのみで、リポジトリ所有者による §10 承認事項 1〜6 のいずれ
に対する明示的な承認コメントも存在しない。issue が起票されていること自体は
承認事項の承認にはならない（前例: #2063「facade: 高階微分 API の公開面追加」・
#2064「facade: custom autograd Function の公開面追加」も同様に未承認のまま
保留し、それぞれ PR #2208（`ef415e6d`）・PR #2212（`ea838b71`）で facade／
autodiff src を一切変更せず否定ガード＋保留記録 doc のみをマージした。同 doc
`docs/autodiff-higher-order-grad-decision.md` §15・`docs/autodiff-custom-
function-decision.md` §15 と本節は同型）。

#2133 本文が想定する形（案 A: `pub use fandhe_ai_autodiff::nn::{Module,
ModuleList}` の素の再エクスポート）は、§1.3・§5 で不採用と判定済みである
（`Module::forward` が生の `fandhe_ai_autodiff::Tape` を引数に取るため、
`fandhe-ai` のみに依存する利用者は `impl Module for MyLayer` を書けない）。
推奨案 B（facade 独自の薄い trait とコンテナの新設）も §10 で未承認のまま
であるため、案 A・案 B のいずれも実施できない。

自動運転（承認待ち不可）かつ判断は安全側に倒す方針、`docs/compat-api-scope.md`
§5（範囲拡張は経路 1／2 の承認必須）、`.claude/rules/security.md`（自己修復に
よる無断拡大禁止）に基づき、本イシューでは `crates/facade/src/**`（`#[cfg(doctest)]`
限定の非公開足場 1 件を除く）・`crates/autodiff/src/**` を一切変更せず、次の
否定ガード群を追加・拡充して「facade 未公開」状態を機械固定した:

- `crates/facade/tests/api_surface.rs::
  facade_does_not_reexport_nn_module_or_containers`（新規）: facade src の
  全 `pub use` 文の葉（[`collect_pub_use_leaves`]。ソース側・rename 前）を
  走査し、葉が `Module`／`ModuleList` である行、または葉が `Sequential` かつ
  パスに `fandhe_ai_autodiff`／`nn` を含む行を違反とする。別名再エクスポート
  （`as Layer` 等）もソース側の葉で判定するため検出できる。`compat/mod.rs` の
  `pub use sequential::{Sequential, SequentialVars};`（パスに
  `fandhe_ai_autodiff`／`nn` を含まない）は正当な既存形として許容する
- `crates/facade/tests/api_surface.rs::facade_declares_no_nn_module_items`
  （新規）: facade src に facade 独自の `trait`／`struct`／`enum`／`type`
  版の `Module`／`ModuleList` 宣言、または `struct Sequential`
  （`src/compat/sequential.rs` 以外）が可視性を問わず存在しないことを固定
  する。非公開 `use fandhe_ai_autodiff::nn::{..., Module, ...}`・
  `Box<dyn Module>` の型参照・`compat/sequential.rs` 自身の
  `pub struct Sequential` は正当な既存形として許容する
- `crates/facade/tests/api_surface.rs::
  compat_sequential_does_not_expose_module_add_methods`（新規）:
  `src/compat` 配下に `add_module`／`add_boxed`／`push_module` の `pub fn`
  宣言が存在しないことを固定する（§9 で `add_module` はスコープ外と明記
  済み）
- `crates/facade/src/lib.rs::NnModuleHoldDoctestGuard`（新規。
  `VarCustomHoldDoctestGuard`〈#2064〉と同型の正のプローブ 1 ブロック
  方式）: facade の全 `pub mod` を glob import したスコープへ、本 doctest
  内でのみ定義したローカル `__fandhe_nn_hold_probe::{Module, ModuleList}`
  を導入し、`fn __probe(_: &dyn Module, _: ModuleList, _: &Sequential) {}`
  を実際に書く。facade がどの経路（別名再エクスポート・trait 定義・
  `pub type` 別名・`nn::Sequential` の新設等）で `Module`／`ModuleList`／
  （`compat::Sequential` 以外の）`Sequential` という名前を公開しても、
  ローカル定義との glob 衝突により名前解決が曖昧になり（E0659 等）、
  エラーコードに依存せずコンパイルが失敗する。ドリフト検査（glob 対象
  集合の一致は `nn_module_hold_doctest_globs_all_pub_modules`、本文の固定
  文言 `NN_MODULE_HOLD_PROBE_BODY` との完全一致は `nn_module_hold_doctest_
  probe_body_matches_fixed_contract`）は `extract_var_custom_hold_doctest_
  guard_doc` を struct 名引数版 `extract_hold_doctest_guard_doc(content,
  struct_name)` へ最小限リファクタしたうえで共用する。既存の呼び出し側は
  `"VarCustomHoldDoctestGuard"` を渡すだけで挙動は不変であることを
  `custom_function_hold_doctest_*` の 2 テストが無変更のまま合格し続ける
  ことで確認済み

上記 3 件のソース走査ガードと doctest 1 件が実際に漏れを検出することは、
一時的な合成入力（コミットしない）で個別に確認済み: `src/nn/mod.rs` へ
`pub use fandhe_ai_autodiff::nn::{Module, ModuleList};` を追加すると
`facade_does_not_reexport_nn_module_or_containers` と `NnModuleHoldDoctestGuard`
doctest（`ModuleList` の glob 衝突・E0659）の両方が fail する。`src/lib.rs`
へ `pub use fandhe_ai_autodiff::nn::Module as Layer;`（別名再エクスポート）
を追加すると `facade_does_not_reexport_nn_module_or_containers` が fail する。
`src/nn/rnn.rs` へ `pub trait Module {}` を追加すると
`facade_declares_no_nn_module_items` が fail する。

既存 3 件のガード（`nn_mod_declares_only_rnn_submodule`・`nn_rnn_module_
reexports_exactly_expected_surface`・`facade_pub_use_leaves_are_not_
modules`）も #2133 の保留を部分的に担っていることを doc comment へ追記した
（`nn_mod_declares_only_rnn_submodule` は `nn::module`／`nn::container` の
無断新設を、`nn_rnn_module_reexports_exactly_expected_surface` は
`nn/rnn.rs` への `Module` 追加を、`facade_pub_use_leaves_are_not_modules`
は `nn` の別名モジュール再エクスポート一般形を、それぞれ fail-closed に
拒否する）。ロジック自体は変更していない。

承認取得後（経路 B）に実施する変更範囲（事前提示）:

- 案 B を採用する場合: `crates/facade/src/nn/{module.rs, container.rs}` を
  新設し、facade 独自の `nn::Module`（required は `forward(&self, &fandhe_
  ai::Tape, &Var)`。§10 承認事項 2 の defaulted 6 件を踏襲）と
  `ModuleList`／`Sequential` を置く
- `nn_mod_declares_only_rnn_submodule` の期待集合を `["rnn"]` から
  `["rnn", "module", "container"]`（または同等の構成）へ更新する
- defaulted メソッド集合（`named_parameters`／`set_parameter`／
  `state_dict`／`load_state_dict`／`set_training`／`training`）の一致検査を
  新設する
- 本 PR の否定ガード 3 件（ソース走査）と `NnModuleHoldDoctestGuard` を
  外すか、正ガードへ転換する
- facade だけに依存するユーザー定義層の最小 unit test を追加する
- 橋渡し方式（借用ハンドル型か、第 1 段では対象外か）は §10 承認事項 4 の
  承認結果に従う
- #2134／#2137 の鏡写し要件（§10 承認事項 6）は引き続き承認待ちのまま

本 PR のマージで #2133 は COMPLETED となる（前例 #2063・#2064 と同じ）。
承認取得後は本節の事前提示に基づき、新規イシュー（または #2133 の
reopen）で経路 B（公開実施）を別 PR で行う。

## 13. facade 公開の再追跡記録（イシュー #2338）

確認日時 2026-09-28・main HEAD `d1b2dd47`（`git checkout -B` 時点の
`origin/main`）。イシュー #2338「`nn::Module`・`ModuleList` の facade
公開が未実施のまま #2133 がクローズされた件を追跡し直す」の実装着手
時に、承認状況を再確認した。

### 13.1 確認結果（承認は存在しない）

`gh issue view` で #2131・#2132・#2133・#2338 のコメントを確認した:

| issue | コメント数 | 内容 |
|---|---|---|
| #2131（親） | 0 | ― |
| #2132（本 doc の対応 issue） | 0 | ― |
| #2133（実装。クローズ済み） | 1 | 所有者が付けた「未実施分は #2338 で追跡します」の定型コメント 1 件のみ。§10 承認事項への言及はなし |
| #2338（本イシュー） | 0 | ― |

PR #2230（#2132 の設計記録 PR）のコメント・レビューも確認した。
コメント 0 件・レビュー 5 件はすべて `github-actions`（codex-review）
による自動レビューで、所有者による明示的な承認コメントは存在しない。

§10 承認事項 1〜6 の状況は次のとおり、いずれも未承認のまま変わって
いない。

| # | 承認事項 | 状況 |
|---|---|---|
| 1 | 案 B の採否（または案 A／D の指名） | 未承認 |
| 2 | facade trait の defaulted メソッド集合（6 件）と命名契約の踏襲 | 未承認 |
| 3 | facade 側 `ModuleList`／`Sequential` の新設可否 | 未承認 |
| 4 | 独自層 → autodiff コンテナ橋渡しの方式 | 未承認 |
| 5 | #2133 本文の実装形更新 | 未承認 |
| 6 | #2134／#2137 への鏡写し要件 | 未承認 |

### 13.2 #2133 クローズの経緯（是正）

#2133 は「誤ってクローズされた」のではなく、PR #2233 によって
**意図的に** COMPLETED としてクローズされた。これは前例 #2063
（PR #2208）・#2064（PR #2212）と同じ運用（保留固定を機械的なガード
で完了させ、承認待ち事項は doc へ事前提示として記録したうえで issue
自体は閉じる）であり、本 doc §12 末尾にもその方針が明記されている。

実際の欠陥は運用そのものではなく、**この運用の結果として「facade
公開面拡張がユーザー承認待ちである」ことを追跡する open な issue が
残らなかったこと**である。#2338 はその追跡先として機能する。今後
同様の保留 PR で issue をクローズする際は、承認待ち事項を追う別の
open issue（今回でいう #2338）を残すか、親 issue（#2131）側に承認
待ち一覧を集約する運用が望ましい。

### 13.3 #2338 の AC3 と本 doc §1.3・§5 の矛盾

#2338 の受入条件 3（AC3）は「`Module`・`ModuleList` を facade へ
再エクスポートする」（案 A）という文言のままである。しかし本 doc
§1.3・§5 は案 A を **不採用**と判定済みである
（`Module::forward` が生の `fandhe_ai_autodiff::Tape` を引数に取る
ため、`fandhe-ai` のみに依存する利用者は `impl Module for MyLayer`
を書けない構造制約による）。

したがって、承認事項 1（§10）が承認された場合の実装形は AC3 の文言
どおりの素の再エクスポートではなく、**案 B**（facade 独自の薄い
trait とコンテナの新設。§5・§6）になる見込みである。本節はこの
矛盾を記録するのみとし、AC3 の文言を書き換えて「解消」する対応は
行わない（イシュー本文の編集は本 doc の対象外）。

### 13.4 保留固定の検証結果（HEAD `d1b2dd47` で合格）

§12 の否定ガード群が HEAD でも有効であることを、本イシューの実装
着手時に再実行して確認した（全件 pass）:

- `cargo test -p fandhe-ai --test api_surface`（`nn_module`／
  `facade_declares_no_nn_module`／`compat_sequential_does_not_expose_
  module_add`／`nn_mod_declares_only_rnn_submodule` を含む対象テスト
  群）: `facade_does_not_reexport_nn_module_or_containers`（+
  `_detects_each_category`）・`facade_declares_no_nn_module_items`
  （+ `_detects_each_category`）・`compat_sequential_does_not_expose_
  module_add_methods`（+ `_detects_offense`）・
  `nn_module_hold_doctest_globs_all_pub_modules`・
  `nn_module_hold_doctest_probe_body_matches_fixed_contract`・
  `nn_mod_declares_only_rnn_submodule` の計 11 件すべて pass
- `cargo test -p fandhe-ai --doc`: `NnModuleHoldDoctestGuard` を含む
  46 件の doctest すべて pass

1 件も fail していないため、ガードの退行是正は不要だった。

### 13.5 承認取得後に実施する変更範囲

§12 末尾の「承認取得後（経路 B）に実施する変更範囲」をそのまま参照
する（本節では重複して書かない）。本イシュー（#2338）でもコード
変更（`crates/facade/src/**`・`crates/autodiff/src/**`）は一切行わず、
`crates/facade/tests/api_surface.rs` の対象 3 テストの doc コメントへ
本節への参照を 1 行追記したのみである。

### 13.6 本 PR での #2338 の扱い

本 PR は #2338 を **close しない**。§13.2 で記録したとおり、
「承認待ち事項を追跡する open な issue が消える」という #2338 発生
の原因そのものを再発させないための判断であり、前例（#2063・#2064
→ #2133）の運用から意図的に外れる。PR 本文にもこの判断理由を明記
する。

## 14. 実装記録（イシュー #2395）

`fandhe_ai::nn::Module`（required `forward` 1 件 + defaulted 6 件）を
`crates/facade/src/nn/module.rs` に新設した（#2338 承認事項 1・2・4）。

- **公開形**: `nn/mod.rs` に非公開 `mod module;` と `pub use module::Module;`
  を置く。`pub mod module;` にしないのは、`collect_public_module_paths` の集合
  （約 43 件の hold doctest の glob 一覧）と `nn_mod_declares_only_rnn_submodule`
  の期待値 `["pub mod rnn;"]` を不変に保つためである。`pub use` 経路の漏れは
  縮小した `facade_does_not_reexport_nn_module_or_containers` が固定する。
  #2396 のコンテナも同型（`mod container;` + `pub use`）で揃える。
- **`load_state_dict`**: crate 内非公開の借用ブリッジ `ParamBridge` が autodiff 側
  `Module` を実装し、autodiff の既定実装（two-pass・キー昇順適用・逆順ロールバック・
  ロールバック失敗時の部分適用エラー）をそのまま再利用する。意味論の一致が構造的に
  保証されコピーによるドリフトがない。`set_parameter` 既定のメッセージは
  autodiff 側と同一文字列で、in-crate テストが一致を固定する。
- **ガードの縮小**: `facade_does_not_reexport_nn_module_or_containers` は
  `src/nn/mod.rs` の `pub use module::Module;` 1 件だけを許容し（件数 1 のインベントリ
  つき）、`ModuleList`／nn 系 `Sequential` は従来どおり違反とする。
  `facade_declares_no_nn_module_items` は `src/nn/module.rs` の `trait Module` のみ許容する。
  `NnModuleHoldDoctestGuard` の probe から `Module` を外した（glob 衝突回避。`ModuleList`
  と `Sequential` は #2396 まで保留）。`compat_sequential_does_not_expose_module_add_methods`
  は不変（`add_module` は #2398）。
- **正ガード**: `facade_nn_module_trait_methods_match_approved_set`（required =
  `forward`・defaulted 6 件の完全一致）と `facade_nn_module_trait_signatures_hide_internal_types`
  （シグネチャに `fandhe_ai_autodiff`／`BackendOps`／裸の `Tape` が現れず `forward` の第 1 引数が
  `TapeRef`）と自己テスト。#2400／#2401 で鏡写しメソッドを足すときは集合を更新する。
- **申し送り**: #2397 のアダプタは `ParamBridge` を一般化・置換してよい。#2403 で台帳
  （`docs/compat-api-scope.md`）を最終まとめする。

## 15. 実装記録（イシュー #2397）

facade `nn::Module` を autodiff `Module` として扱う crate 内アダプタを実装した（#2338 承認事項 4・§6・§10・§14 の申し送りに沿う）。

- **一般化**: 非公開の `ParamBridge<'a, M>` を `pub(crate) struct FacadeModuleAdapter<P>(pub(crate) P)`（`P: DerefMut, P::Target: Module`）へ置き換えた。`P = &mut M` は facade `load_state_dict` 既定の内部ブリッジ、`P = Box<dyn Module>` は autodiff コンテナ（`'static` 必須）へ積む実体になる。`unsafe` は使わず `TapeRef::from_autodiff` の安全な借用変換で橋渡しする
- **委譲**: `forward`・`named_parameters`・`set_parameter`・`set_training`・`training`。`state_dict`／`load_state_dict` は autodiff 既定のまま（委譲済みの `named_parameters`／`set_parameter` の上で動く）。facade 層が独自に `load_state_dict` を override していても本アダプタ経由では迂回される（`ModuleList` が子を扱うのと同じ意味論）
- **`supports_forward_host` は `false` へ override**: イシュー本文は「既定の `false` のまま」と記すが、autodiff trait の実際の既定は `true`。既定のままだと `compat::Sequential::predict` の事前判定を通過し、途中層の `Err` で手前層の副作用（Dropout の RNG 消費・BatchNorm の running stats 更新）が tape 経路再実行と二重化するため、`Embedding`／`MultiheadAttention` の前例に倣い明示的に `false` を返す
- **`forward_host` は `AutodiffError::InvalidArgument`（fail-closed）**: `BackendError::Unsupported` は「フォールバックの合図」で `predict_recorded` が捕捉して再実行するため使わない（前例 `ModuleList::forward_host`）
- **検証の読み替え**: #2396（facade `nn::Sequential`）は未マージのため、「facade `nn::Sequential` と bit 一致」は同じ層を手動連鎖させた参照経路（別 tape）との bit 一致（値・入力勾配・葉勾配・ノード数。`[Linear, Adapter]`／`[Adapter, Linear]` の 2 並び）で代替した
- **申し送り**: #2398 で `nn/mod.rs` に `pub(crate) use module::FacadeModuleAdapter;` を追加し `compat::Sequential` の公開入口を作る。`set_requires_grad`／`requires_grad`／`children`／`type_name` の委譲は #2400／#2401（それまで既定の fail-closed のため、パラメータ持ちアダプタを含む `Sequential::freeze()` は `Err`）
