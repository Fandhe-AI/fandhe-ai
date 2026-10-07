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
#2596 で crate 直下の委譲関数として公開済み。§6.1・§6.2・§6.3 参照）。新規 `Op`・`BackendOps` メソッド・VJP・tape
ノードは追加していない。facade 公開（`Var` への委譲メソッド追加）は
#2141 時点では承認待ちのまま対象外とした（履歴。現状は比較 6 種・
`masked_select` が #2510、logical 3 件が #2596 で公開済み）。
承認形外の配置（`Var`／`Tensor`／`Tape` 上・`bool_ops` モジュール再
エクスポート）は `crates/facade/src/lib.rs::VarBoolOpsHoldDoctestGuard`
（正のプローブ doctest）と `crates/facade/tests/api_surface.rs` の
ソース走査・workspace インベントリ（4 テスト）で引き続き多層固定している。

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
   logical 3 件は #2594 の §6.2 推奨案（B-1）がルート #2499 コメントで承認され、
   #2596 で適用済み（§6.3）**
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
（§6.1）。logical 3 件分は #2596 で実施済み（§6.3）。`VarBoolOpsHoldDoctestGuard` は
承認形外を拒むガードとして残置し、ソース走査のみ正ガード化した。

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

### §6.2 logical 3 件の facade 公開形（#2595・承認済み）

**本節は #2595 時点の推奨案の記録で、案 B-1 は #2596 の着手前にルート #2499 のコメントで承認された（§6.3）。以下の「承認待ち」「未取得」等の表記は #2595 時点の履歴である。**
親 #2594・祖 #2542、実装は兄弟 #2596（承認後にのみ着手）。§6 項目 1 は
logical 3 件の公開形を「`Var` の関連関数か facade 直下の関数か」の 2 案
併記のまま推奨形を持たず、ルート #2499 の一括承認（記録済みの推奨形にのみ
及ぶ）の対象外である。

#### 着手時判定（調査基準: `origin/main` 699847bf・2026-10-05）

- 3 関数のシグネチャは `(&Tensor<bool>, &Tensor<bool>) -> Result<Tensor<bool>,
  AutodiffError>`（`logical_not` は単項。`crates/autodiff/src/bool_ops.rs:197`・
  `:205`・`:213`）。`Var`・`Tape` を取らずホスト常駐データのみで計算し、確保前に
  要素数を検査する（`checked_bytes_for::<bool>`）。返しうるエラーは
  `AutodiffError::Shape(..)`（ブロードキャスト不可・要素数 overflow）のみ
- `AutodiffError`・`ShapeError` は `#[non_exhaustive]` で facade 直下に再
  エクスポート済み
- facade 直下には `Tensor` を受け取り返す委譲 `pub fn` の前例がある
  （`eye`／`zeros_like`／`ones_like`。`crates/facade/src/lib.rs:1103-1121`）
- facade の `pub fn`／`pub use` に `logical_*` は無い。`crates/facade/src/` での
  出現は `VarBoolOpsHoldDoctestGuard` の doc・プローブのみ
- `Tensor<bool>` は facade から生成（`Tensor::new`・`Var::cast::<bool>()`・
  `Var::*_bool`）・消費（`where_cond`・`masked_fill`・`masked_select`）できる
- 保留ガードは doctest `VarBoolOpsHoldDoctestGuard` と `api_surface.rs` の 4
  テスト。ソース走査 `facade_does_not_reexport_or_declare_bool_ops`
  （`api_surface.rs:8445`）は `bool_ops` を含む `pub use` 行と 3 名の `fn`
  宣言を拒否する
- `tensor-core` に `Tensor<bool>` 専用の inherent impl・`BitAnd`／`BitOr`／`Not`
  の impl は無い

#### 候補比較

比較軸: 呼び出し形／変更クレート／0.10.0 非破壊性／名前衝突／既存保留群・既存
公開との整合／将来拡張（§6 項目 3「`Var` 入力の logical 版」・項目 4「GPU 専用
カーネル」）への影響。

| 案 | 形 | 評価の要点 |
|---|---|---|
| A: `Var` 委譲（関連関数） | `Var::logical_and(&Tensor<bool>, &Tensor<bool>)` を `var.rs` に追加 | 追加のみで非破壊。公開済み 7 件と同じ委譲・同じ正ガードを流用できる。一方 3 関数は `Var`・`Tape` に触れず `Var::` 名前空間が実体と合わない。**`Var::logical_and` の名前を占有し、項目 3 の `Var` 入力版（自然な形は `x.logical_and(&y)`）を将来同名で追加できなくなる**。内部クレート `autodiff` の変更を伴う |
| B-1: facade 直下の委譲関数 | `fandhe_ai::logical_and(&a, &b)`。`facade/src/lib.rs` に `pub fn` 3 件、本体は `fandhe_ai_autodiff::bool_ops::<name>(..)` の 1 式 | 追加のみで非破壊。直下に同名なし。`zeros_like`／`ones_like`／`eye` と同じ既存パターン。`torch.logical_and` の自由関数形に対応。`Var`／`Tensor` のメソッド名を占有せず項目 3 を妨げない。変更は facade に閉じる。`bool_ops` を含む `pub use` の全面禁止（既存ソース走査規則）を維持できる |
| B-2: 3 関数のみ選択再エクスポート | `pub use fandhe_ai_autodiff::bool_ops::{logical_and, logical_or, logical_not};` | 利用側から見た形は B-1 と同じで本体も無い。ただし `bool_ops` を含む `pub use` を 1 行許す例外をガードに設ける必要があり、rustdoc に内部向け文面（イシュー番号を含む）がそのまま出る |
| C: モジュール再エクスポート | `pub use fandhe_ai_autodiff::bool_ops;` | 10 関数すべてが公開され、公開済み 7 件が `Var` メソッドと自由関数の二重入口になる（§3 案 B を退けた理由と同じ）。内部モジュール名が公開名として固定される。現行ガードが明示的に拒否する配置 |
| D: `Tensor<bool>` の inherent メソッド／演算子 | `a.logical_and(&b)`・`&`／`\|`／`!` | facade からは外部型に inherent impl を足せず `tensor-core` 側の実装が必要。`tensor-core` は `AutodiffError` を名指しできず、実装移設とエラー型変更を伴う。演算子トレイトはブロードキャスト失敗を `Result` で返しにくい。#2594 の対象範囲（facade 配下）を超える。B の上に将来追加することは妨げない |
| E: 独自型 | `BoolMask` newtype・拡張 trait | 新しい公開型／trait が増える。既存公開 API はすべて `Tensor<bool>` を直接やり取りするため相互変換が要る。trait は sealing の論点が加わり、後で inherent へ移すと入口が二重に残る（§3 案 B と同じ） |

名前衝突について:

- **facade 内**: 直下の `pub fn`／`pub use` と `pub mod` 配下に 3 名は無い（着手時
  に grep で確認。#2596 着手時に再実行する）
- **既存の保留群**: 他の `*HoldDoctestGuard` は `use fandhe_ai::*;` したスコープへ
  ローカル定義を置く方式で、`crates/facade/src/` で `logical_` が現れるのは
  `VarBoolOpsHoldDoctestGuard` のみ。同ガードのプローブ本体での 3 名の参照は
  経路付き・メソッド形・関連関数形で、裸の識別子としては使っていないため、
  直下への 3 関数追加と衝突しない見込み（#2596 で実機検証する）。新しい
  `pub mod` を設ける配置は、各ガードの glob import 集合を検査するテスト群
  （`*_hold_doctest_globs_all_pub_modules`）の更新を要する
- **下流利用者**: `use fandhe_ai::*;` と別クレートの glob import が同名を持ち、
  かつその名前を実際に使う場合に限り曖昧性エラーになる。下流のローカル定義・
  明示 import は glob より優先される。公開項目の追加は通常 minor 変更に分類
  される範囲だが、本記録は互換性の保証を断定しない

#### 推奨案（承認待ち）

**B-1**。

```rust
pub fn logical_and(a: &Tensor<bool>, b: &Tensor<bool>) -> Result<Tensor<bool>, AutodiffError>
pub fn logical_or(a: &Tensor<bool>, b: &Tensor<bool>) -> Result<Tensor<bool>, AutodiffError>
pub fn logical_not(a: &Tensor<bool>) -> Result<Tensor<bool>, AutodiffError>
```

根拠:

1. 3 関数は `Var`・`Tape` に触れないため、§6 項目 1 が挙げる「facade 直下の
   関数」が実体に合う
2. 既存の直下委譲関数と同パターンで、新しい型・trait・`pub mod`・`Op`・
   `BackendOps`・VJP・`unsafe`・依存が不要
3. `Var`／`Tensor` のメソッド名を空けておける
4. 変更が facade に閉じる
5. 意味論は内部自由関数と同一（bool 入力のみ・NumPy 互換ブロードキャスト・
   確保前の要素数検査を迂回しない 1 式委譲）

0.10.0 非破壊チェック: 既存項目のシグネチャ・意味論の変更なし／追加は直下の関数
3 件のみ／`FitConfig` 不変／tolerance・baseline・依存・閾値・`docs/spec` 不変。

#### ユーザーに決めてほしい事項

- (a) 公開形: A／B-1／B-2／C／D／E のどれか。推奨は B-1
- (b) 配置: facade 直下か新しい `pub mod` 配下か。推奨は直下
- (c) エラー型: `AutodiffError` のまま（`Var::*_bool`・`masked_select` と同じ。将来
  GPU 経路でバックエンド由来の失敗が加わっても型を変えずに済む）か、
  `ShapeError` へ狭める（`zeros_like` と同じ）か。推奨は `AutodiffError`
- (d) 保留ガード: 現行の撤去条件どおり doctest を削除するか、「承認形以外
  （`Var`／`Tensor`／`Tape` 上の配置・`bool_ops` モジュール公開）を拒む」ガード
  として残しソース走査だけを正ガード化するか。推奨は後者（#2510 の部分反転と
  同じ扱い）
- (e) §6 項目 2〜4 は引き続き対象外でよいか
- (f) #2595 の閉じ方と #2596 の着手条件（承認コメントが付いてから着手）

#### 承認後の実装スケッチ（#2596。本記録では実施しない）

- `crates/facade/src/lib.rs`: `pub fn` 3 件と facade のみを import した doctest。
  `VarBoolOpsHoldDoctestGuard` の doc を (d) の決定に合わせて更新
- `crates/facade/tests/api_surface.rs`: `BOOL_OPS_HELD_FN_NAMES` を承認済み集合へ
  移し、`facade_does_not_reexport_or_declare_bool_ops` を正ガードへ書き換え
  （`bool_ops` を含む `pub use` は 0 件／logical 3 件は `lib.rs` に各 1 件で本体が
  1 式委譲とトークン一致）。`workspace_declares_bool_ops_fn_names_in_approved_places_only`
  の期待集合へ `facade/src/lib.rs::<name>` を追加。直下の名前解決の正のプローブ
- `crates/autodiff/src/bool_ops.rs`・`lib.rs` の「未決・保留」doc 文言を更新
  （挙動不変）
- テスト: facade のみを import した利用テスト（真理値表・ブロードキャスト・
  ブロードキャスト不可の `Shape` エラー・巨大 broadcast view の確保前拒否）
- docs: 本記録への実装記録、`docs/compat-api-scope.md` §5 の保留記録更新、
  `docs/README.md`
- 新規 `Op`・カーネル・`unsafe`・依存なし。ホストのみの計算のため CUDA／Metal の
  新規 `#[ignore]` テスト・実測申し送りは不要

#### 本節の位置づけ（#2595 時点の履歴。承認取得後の実装は §6.3）

承認は未取得。保留ガードと 4 テストは維持する。本節の追記は `crates/`・
`Cargo.*`・tolerance・`docs/spec` を変更していない。

### §6.3 実装記録（#2596・親 #2594。§6.2 案 B-1 の適用）

- 承認の根拠: ルート #2499 のコメント
  https://github.com/Fandhe-AI/fandhe-ai/issues/2499#issuecomment-6033824965
  （2026-10-07T08:12:50Z）の表が「#2594: 本記録 §6.2 の推奨案」を承認している。
  範囲は §6.2 に書かれた形（(a)〜(f) すべて推奨どおり）に限る。#2594・#2596 の
  補足コメントは同コメントを根拠に指すだけで、承認の根拠は上記 URL とする。
- 公開した名前（`crates/facade/src/lib.rs`。`fandhe_ai` 直下。新しい `pub mod` なし）:
  `logical_and(a: &Tensor<bool>, b: &Tensor<bool>)`・`logical_or(同)`・
  `logical_not(a: &Tensor<bool>)`。戻り値は `Result<Tensor<bool>, AutodiffError>`。
  本体は `fandhe_ai_autodiff::bool_ops::<name>(..)` の 1 式委譲。doctest 付き。
- ガード反転（`crates/facade/tests/api_surface.rs`）:
  `BOOL_OPS_HELD_FN_NAMES` を `BOOL_OPS_APPROVED_ROOT_FN_NAMES` へ改名し、本体固定の
  `BOOL_OPS_ROOT_EXPECTED_BODIES` を追加。`facade_does_not_reexport_or_declare_bool_ops`
  を `facade_declares_logical_fns_only_as_approved_root_delegations` へ改名・正ガード化
  （`bool_ops` を含む `pub use` 0 件／`Var` 委譲 7 名の fn 宣言 0 件／logical 3 件は
  `lib.rs` に各 1 件・他に 0 件／本体の 1 式委譲をトークン一致で固定）。
  `workspace_declares_bool_ops_fn_names_in_approved_places_only` の期待集合へ
  `facade/src/lib.rs::<name>` を追加。到達性の正のプローブ
  `logical_fns_are_reachable_via_facade_root` を追加。
- 保留ガード doctest（`VarBoolOpsHoldDoctestGuard`）は (d) の決定どおり削除せず、
  プローブ本体（`BOOL_OPS_HOLD_PROBE_BODY`）も不変。doc 散文のみを承認形外
  （`Var`／`Tensor`／`Tape` 上の配置・`bool_ops` 再エクスポート）を拒むガードへ更新した。
  直下の同名自由関数はプローブと衝突しないことを `cargo test -p fandhe-ai --doc` で確認。
- 手動確認（fail-closed）: `lib.rs` への仮 `pub use fandhe_ai_autodiff::bool_ops;` で
  doctest（E0659）とソース走査が失敗、`logical_and` 本体の引数入れ替えでソース走査が
  失敗することを確認し元に戻した。
- 利用テスト: `crates/facade/tests/bool_ops_logical_fns.rs`（6 件・CPU。真理値表・
  ブロードキャスト・非 contiguous・`Shape` エラー・巨大 broadcast view の確保前拒否・
  `Var` 比較との合成）。
- CUDA／Metal: ホストのみの計算でカーネルを新設しないため新規 `#[ignore]` テスト・
  実測申し送りは無い。
- `fandhe-ai =0.10.0` 非破壊: 追加は直下の `pub fn` 3 件のみ。依存・tolerance・baseline・
  ガードレール閾値・`docs/spec` は不変。
- スコープ外のまま: 微分可能な `masked_select`・`Var` 入力の logical 版・GPU 専用
  カーネル・`Tensor<bool>` の inherent メソッド／演算子（§6 項目 2〜4・案 D）。
