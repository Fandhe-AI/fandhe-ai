# `Var` 演算子オーバーロード（`+`・`*`・`-`）の設計判断記録（#2135）

イシュー #2135「`Var` 演算子オーバーロード（`+`・`*`・`-`）の設計判断記録」。親 #2131（Phase 5「PyTorch／TF 置き換えの API 網羅」5-A 基盤）。後続 #2136（実装）がこの設計の承認を前提にする。兄弟イシューは #2141（比較・logical 演算）・#2145（`pow_scalar` 系）。

本ドキュメントは **コード変更を伴わない設計記録**を成果物とする。`crates/**`・`CLAUDE.md`・`docs/spec/`（正本 submodule）・`Cargo.toml`／`Cargo.lock`・tolerance／baseline・ガードレール閾値・CI／hooks はいずれも変更しない。

基準コミット: `ea838b71`（本ブランチ作成時点の `origin/main`。2026-09-23）。

## 0. 結論・段階

本イシューは docs のみ・**段階 0**（facade 公開面は不変）。

推奨は **案 A**（`Output = Result<Var<'t>, AutodiffError>`）＋ **4 通りの borrow/consumed 組合せを `Add`／`Mul`／`Sub` の 3 トレイトすべてに実装し、`Neg` は `Var`／`&Var` の 2 通り**。本体は既存 inherent メソッド（`Var::add`／`mul`／`sub`／`neg`）への 1 行委譲のみとし、新しい `Op`・新しい評価経路を追加しない（§5）。スカラー混合（`&Var + 2.0` 等）・`Div`（`/`）・案 B〜D はいずれも承認事項（§12）として列挙するのみで、本 doc では決定しない。

## 1. 背景

### 1.1 課題

`Var` の四則演算は現状メソッド形（`a.add(&b)?`）でしか書けない。PyTorch 風の `loss + reg` のような式を直接書けないため、Phase 5-A が目指す「PyTorch／TF 置き換え水準の API 網羅」に届かない。

### 1.2 本 issue で確定すること（受入基準の構造化要約。カッコ内は対応する本 doc の節）

1. 演算子トレイトのシグネチャ（借用形か consumed 形か、`Output` 型は何か）（§3・§4）
2. スカラー混合（`&Var + 2.0` 等）を入れるかどうか（§8）
3. 既存メソッド形との bit 同一を仕組みとして保証する方法（#2136 で入れるコメント文面を含む）（§5）
4. 単項 `-` と `Var::neg()` の bit 完全一致（§6。テストとしての固定は§9。演算子が未実装のため本 issue では確認自体はできず、#2136 が実装するテストの仕様として固定する読み替えを採る）
5. 複合式（`a + b * c` 等）の結合性テストの仕様（§9。同じ理由で本 issue ではテスト仕様の固定に留め、実装・実行は #2136 の責務とする）
6. 本 doc に背景・借用形の選択理由・bit 同一契約を書く（本 doc 全体。背景は §1、借用形の選択理由は §3〜§4・§7、bit 同一契約は §5〜§6）

### 1.3 契約（イシュー共通）

0.9.0 crates.io 公開 API を壊さない（既存メソッドのシグネチャ・意味論は変えない）。ユーザー承認が要る事項は列挙のみで実施しない。

## 2. 現状のコード事実（基準コミット `ea838b71`）

| 事実 | 出典 |
|---|---|
| `Var<'t>` は `#[derive(Debug, Clone, Copy)]`。中身は `tape: &'t Tape`・`id: NodeId` のみ | `crates/autodiff/src/var.rs:108-112` |
| `add`／`mul` は `&self, &Var<'t> -> Result<Var<'t>, AutodiffError>`。クロステープ検査（`TapeMismatch`）→ broadcast 検査（`Shape`）→ `pre_materialize_for_binary_merge` → `push_lazy`（遅延融合。連鎖長上限到達時はその場実体化）の順に進む。**lazy** | `crates/autodiff/src/var.rs:662-678`（`add`）・`685-701`（`mul`） |
| `sub`／`div` は `scalar_binary(other, ScalarBinaryOp::Sub/Div)` へ薄く委譲する。**eager**（`push_eager` 経由） | `crates/autodiff/src/var.rs:873-875`（`sub`）・`886-888`（`div`） |
| `neg` は `scalar_unary(ScalarUnaryOp::Neg)` へ委譲する。ホスト定義は `-x`（IEEE の符号ビット反転） | `crates/autodiff/src/var.rs:809-811`・`crates/tensor-core/src/scalar_op.rs:183`（`Self::Neg => -x`） |
| **`Var::add_scalar` は存在しない**。`ScalarUnaryOp` に `AddScalar`／`MulScalar` 相当の variant もない（`PowScalar` はあるが対象外） | `crates/autodiff/src/var.rs`・`crates/tensor-core/src/scalar_op.rs` を grep（存在しないことを確認） |
| CPU の融合 elementwise は `+`／`*` をそのまま使い、FMA（`mul_add`）は使わない。FMA 契約は GEMM 専用 | `crates/backend-cpu/src/fused_elementwise.rs:30-38`（モジュール doc「数値契約」） |
| facade は `pub use fandhe_ai_autodiff::{AutodiffError, Gradients, Var, nn::LinearVars};` で `Var` を再エクスポートしている。**`Var` にトレイト impl を足すと、facade 側のコードを変えなくても facade の公開面が広がる** | `crates/facade/src/lib.rs:184` |
| 演算子オーバーロードは `docs/compat-api-scope.md` の Tier 1／Tier 2 いずれにも列挙されていない。facade 公開には §5（範囲拡張の手続き。経路 2）の承認が要る | `docs/compat-api-scope.md` §1・§5 |
| 葉プレフィックス長（`retained_leaf_len`）は演算が 1 件も記録されていない間は未固定で、`leaf_count()` は現在の全ノード数を返す。定数葉（`Tape::var_no_grad`。`push_leaf` 経由）をこの状態で追加すると、後で `reset()` しても葉プレフィックスに含まれて残り続ける | `crates/autodiff/src/tape.rs:2329-2336`（`freeze_leaf_prefix`）・`2342-2350`（`leaf_count`）・`2395-2398`（`reset` doc「一度も演算を記録していない `Tape` の reset は全ノードを保持する no-op」） |
| Rust の言語仕様上の実測（orphan 規則・E0117・inherent／トレイトメソッド解決順序）。rustc 1.98.1 の scratch クレートで確認し、確認後にクレートは削除済み | 下記 §4・§7 |

## 3. `Output` 型の案比較（中心論点）

対象の 4 メソッド（`add`／`mul`／`sub`／`neg`）はすべて `Result` を返し、実際に失敗しうる（`TapeMismatch`・`Shape`）。`std::ops` の `Output` の中で `?` は使えない（トレイトメソッドの中身は実装者が書くが、`Output` 型自体は演算子の戻り値であり式の外側で `?` を使うかどうかは呼び出し側の選択になる）。

- **案 A（推奨）**: `type Output = Result<Var<'t>, AutodiffError>;`。panic せず、エラーを型で返す。使い方は `let y = (&a + &(&b * &c)?)?;`。式の途中に `?` が要る点は受け入れる。
- **案 B**: `Output = Var<'t>` とし、エラー時は `panic!`。`.claude/rules/coding-rust.md`「本番経路で `unwrap()`／`expect()` を使わない」に反するため不採用。承認事項に入れ、単独では選ばない。
- **案 C**: 連鎖を楽にするため、片側が `Result` の組合せの impl を追加で足す。`impl Add<Result<Var,E>> for &Var` と `impl Add<&Var> for Result<Var,E>` はどちらも `E` をジェネリックのままにしても orphan 規則（RFC 2451）上合法である。同規則は「Self・トレイト引数 `T0..Tn` のうち少なくとも 1 つがローカル型で、最初のローカル型 `Ti` より左（`T0..Ti-1`）に非被覆〈uncovered〉の型引数がないこと」を要求する。両方向とも `&Var`（参照は fundamental 型のため被参照型 `Var`〈ローカル〉を通じてローカル扱い）が `Ti` になり、その左に来うる `Result<Var,E>` は `E` が `Result` の型引数として現れる〈= 被覆される〉ため非被覆に当たらない。rustc 1.98.1 の scratch クレートで両方向とも `E` をジェネリックのまま実測・コンパイル成功を確認済み（イシュー #2135 PR #2237 レビューで「`impl Add<&Var> for Result<Var,E>` は `E` が非被覆で E0210 になる」との指摘があったが、上記の実測により誤りと判断し本 doc の結論は維持する。なお案 C 自体は下記の非対称性を理由に不採用のため、この論点は #2136 の実装可否には影響しない）。ただし `Result + Result`（`impl Add<Result<Var,E>> for Result<Var,E>`）は **E0117（孤児規則違反）で不可**（`Add`・`Result` いずれも外部クレート由来で、`Result<Var,E>` のジェネリック引数にローカル型が現れても impl 全体としては認められない。実測で確認済み）。そのため「`a*b + c*d` のように両辺が `Result` を返す式」だけが書けず、どの式が通るかが非対称になる。驚きを招くため不採用。承認事項に記録する。
- **案 D（将来候補）**: ローカルの newtype（例: `VarExpr<'t>`）を `Output` にし、全組合せで連鎖できるようにする。新しい公開型が増え、stable では `?`（`Try` trait）も使えないため、別途承認が要る将来案として記録するだけにする。

## 4. 借用形・4 組合せ

`Var<'t>` は `Copy` なので、consumed 形でも実質何も消費しない。そこで **4 通りの組合せ（`Var ⊕ Var`・`Var ⊕ &Var`・`&Var ⊕ Var`・`&Var ⊕ &Var`）を `Add`／`Mul`／`Sub` の 3 トレイトすべてに実装し、`Neg` は `Var`／`&Var` の 2 通り**を推奨する。実装はマクロで一括生成し、本体は既存メソッドへの 1 行委譲だけにする（§5）。

Issue 本文にあった「`Add<&Var> for &Var` のみを採るか」という立て方は、`Var` が `Copy` であることに気づく前の設計であり、本 doc で整理し直す（4 組合せが必須である理由は §7 の互換性実測を参照）。

## 5. bit 同一契約（仕組みとしての保証）

演算子の本体は `Var::add(&self_ref, &rhs_ref)` のように**既存の inherent メソッドを 1 回呼ぶだけ**にする。新しい `Op`・新しい評価経路・前後処理は持たない。これにより記録されるテープのノード列（`Op::Add`／`Op::Mul`／`Op::ScalarBinary{Sub}`／`Op::ScalarUnary{Neg}`）がメソッド形と完全に同じになり、forward 値・勾配・遅延／eager の区分・連鎖長上限の挙動が全バックエンド（CPU／CUDA／Metal）でメソッド形と一致する。

**禁止事項**（bit 同一を壊す実装形）:

- `Neg` を `0 - x` で実装すること（`x = +0.0` のとき `+0.0` になり `-0.0` にならない。`Neg` は IEEE の符号ビット反転が定義。§2 出典参照）
- `x * -1` で実装すること（`NaN` の payload が異なる別経路になり、`Op::Mul` と `Op::ScalarUnary{Neg}` で記録されるノードが変わる）
- `Sub` を `a + (-b)` で実装すること（`Op::Add` + `Op::ScalarUnary{Neg}` という別のノード列になり、`add` は lazy・`neg` は eager という §2 の非対称性から、勾配経路・連鎖長カウントが `Op::ScalarBinary{Sub}` の単独呼び出しと一致しなくなる）

**#2136 で `impl` ブロックに入れるコメント文面（要旨。日本語）**:

> 本 impl は `Var::xxx` への純粋な委譲であり、新しい `Op` や評価経路を追加しないことで、メソッド形と bit 同一であることを構造的に保証する（#2135 設計記録 §5）。

FMA 契約とは無関係であることも明記する（elementwise は `mul_add` を使わない。`crates/backend-cpu/src/fused_elementwise.rs:37`）。同じ式の中で `+`／`*`（lazy）と `-`（eager）が混ざっても、メソッド形でも同じノード列になるため数値は変わらない。

## 6. `Neg` と `neg()` の bit 完全一致

`-&a` ／ `-a`（演算子形）は `Var::neg()`（メソッド形）へ 1 行委譲するだけなので、`ScalarUnaryOp::Neg` の同一実行経路（`Self::Neg => -x`。`crates/tensor-core/src/scalar_op.rs:183`）を通り、forward 値・勾配ともに bit 完全一致になる。`+0.0`／`-0.0`／`inf`／`-inf`／非正規化数／`NaN`（payload はハードウェア依存のためクラス一致で比較）を含む fixture でのテストは #2136 の責務とする（§9）。

## 7. 0.9.0 互換性（重要。実測で確認済み）

トレイト impl の追加は semver 上 minor であり、既存メソッドのシグネチャは変わらない。

**inherent メソッドが隠れる危険（実測済み）**: 利用者のスコープに `use std::ops::Add;` があり、`a: Var` に対して `a.add(&b)` を呼ぶ場合を考える。Rust のメソッド解決は同一探索段で「by-value レシーバのトレイトメソッド」を「autoref レシーバの inherent メソッド」より先に試す。`impl Add<Var> for Var` **だけ**を実装すると、この呼び出しが **E0308（型不一致）でコンパイルできなくなる**（scratch クレートで確認済み。`Add::add` の第 2 引数型が `Var` なのに `&b`〈`&Var`〉を渡すことになるため）。`impl Add<&Var> for Var` も併せて実装すれば、たとえトレイトメソッドが選ばれても `Output` の型・中身（inherent への委譲）が同じであるため、ビルドも意味論も保たれる（同じく確認済み）。

- このため **4 通りの組合せを全部実装することは必須要件**にする。部分実装は 0.9.0 利用者のコードを壊しうる。
- `Mul`／`Sub` も同じ構造なので同じ要件をかける。`Neg` は単項のため `Var`／`&Var` の 2 通りで足りる。
- `(&a).add(&b)` の形は同じ探索段で inherent が優先されるため影響を受けない。

型推論が壊れる可能性（トレイト impl 追加一般の注意）は残るが稀である。#2136 の検証項目に、「`use std::ops::{Add, Mul, Sub, Neg};` をスコープに入れた状態で、既存のメソッド形呼び出しがコンパイルでき、結果が bit 同一になること」のテストを追加するよう指定する（§9）。

## 8. スカラー混合（`&Var + 2.0` 等）の判断

**判断: 段階 0（#2136 では入れない）**。理由:

1. 委譲先の `Var::add_scalar` が存在しない（§2）。
2. 新しい `ScalarUnaryOp::{AddScalar, MulScalar}` を追加する経路は、新規 `Op`・CUDA／Metal の既定フォールバック・NVRTC キャッシュキーの拡張を伴う。本 doc・#2136 は「既存演算への委譲のみ・新規 `Op` なし」という範囲であり、その外になる。
3. `var_no_grad` の定数葉と broadcast add で実現する経路は、式を評価するたびに隠れた葉ノードを追加する。とくに演算がまだ 1 件も記録されていない時点で評価すると、その定数が `reset()` 後も残る葉プレフィックスに入り、`leaf_count()`／`leaf(i)` の見え方が変わる（§2 の `tape.rs` 出典）。この副作用を利用者に説明なく持ち込むのは受け入れがたい。

2 経路の得失、および f32 リテラルの推論（f32 版 impl のみなら `2.0` は f32 と推論される）は上記のとおり整理済み。段階 0 続行か新 `ScalarUnaryOp` の別 issue 起票かは承認事項（§12）とし、将来 issue 候補として #2145（`pow_scalar` 系）との整合も含めて検討する。#2136 の受入条件「スカラー版のサポート確認」は、この判断に従って「非対応を確認する（将来 issue へ申し送る）」と読み替える。

## 9. #2136 向けテスト仕様（本 issue では実装しない）

演算子がまだ存在しないため、本 issue では演算子を対象にしたテストは書けない。受入条件の「結合性テスト 1 件以上」「`neg` の bit 完全一致確認」は、**#2136 が `crates/autodiff/tests/operator_overload.rs`（新規）に実装するテストの仕様**として以下に固定する。

- **結合性**: 案 A（§3）の下では `Output = Result<Var<'t>, AutodiffError>` のため、複合式は中間結果を都度 `?` で展開しない限りコンパイルできない（§3 の `let y = (&a + &(&b * &c)?)?;` と同じ形）。#2136 が書くテストは、Rust の演算子優先順位（`*` が `+`／`-` より先に評価される）に対応する下記の `?` 展開済みの演算子形を、対応する既存メソッド形と比較し、forward 値・`backward` 後の全勾配が `to_bits()` で完全一致することを確認する（2 件以上）。この比較が確認するのは「`?` 展開済みの演算子形」と「メソッド形」の bit 同一であり、パーサーの演算子優先順位そのものを検証するものではない（優先順位に対応する `?` の挿入位置は実装者〈#2136〉が手で組み立てる）：
  - `a + b * c` → 演算子形 `(&a + &(&b * &c)?)?` ／ メソッド形 `a.add(&b.mul(&c)?)?`
  - `a + b * c - d` → 演算子形 `((&a + &(&b * &c)?)? - &d)?` ／ メソッド形 `a.add(&b.mul(&c)?)?.sub(&d)?`
  - `(a + b) * c` → 演算子形 `(&(&a + &b)? * &c)?` ／ メソッド形 `a.add(&b)?.mul(&c)?`

  いずれも `?` を使うため、テスト関数のシグネチャは `Result<(), AutodiffError>`（または同等のエラー型）を返す形にする。
- **`neg`**: `-&a`／`-a` と `a.neg()` を、`+0.0`／`-0.0`／`inf`／`-inf`／非正規化数／`NaN`（クラス一致）を含む fixture で `to_bits()` 比較する。
- **互換性**: §7 の `use std::ops::*` スコープ下で、メソッド形の呼び出しがコンパイルでき bit 同一になることを確認する。
- **エラー伝播**: 別テープ同士の `&a + &b` が `Err(TapeMismatch)` を返し、broadcast 不能な shape なら `Err(Shape)` を返すこと（panic しない）を確認する。
- CUDA／Metal の 3 バックエンド bit 同一は `#[ignore]` の実機テストに分け、未実測なら `docs/perf/logs/<slug>-2136/` へ申し送る。
- 文字列走査型の否定ガード（「`Var` が `Add` を実装していないこと」の固定）は**本 issue では追加しない**（#2212 の経緯で heuristics だけでは収束しなかったため。`docs/README.md` の #2212 該当記録を参照）。

## 10. 兄弟 issue との整合

- **#2136（実装）**: 本 doc の§3〜§9 を前提として実装形を確定する。
- **#2141（比較・logical・bitwise 演算）**: スコープ外（§11）。本 doc の設計判断（委譲のみ・bit 同一契約の仕組み）は流用可能な前例として参照されうるが、独立 issue で扱う。
- **#2145（`pow_scalar` 系）**: §8 のスカラー混合判断（新 `ScalarUnaryOp` 追加の是非）と整合を取る必要がある将来検討事項として記録した。

## 11. スコープ外

- 複合代入（`+=` 等）
- 比較・logical・bitwise の演算子（#2141 ほか）
- `Div`（`/`）は issue タイトルにもスコープ外にも明記されていない。**勝手に決めず、承認事項の候補として記録する**（入れる場合も同じ委譲・4 組合せの要件に従う）
- スカラー混合（§8）と案 C／D（§3）

## 12. 承認事項（列挙のみ・実施しない）

1. 本設計（案 A・4 組合せ・委譲のみ）の承認。#2136 の前提になる
2. facade 公開面の拡張の承認（`Var` 再エクスポート経由で自動的に公開されるため、`docs/compat-api-scope.md` §5 経路 2 の適用）
3. スカラー混合の扱い（段階 0 を続けるか、新 `ScalarUnaryOp` の別 issue を起票するか）
4. `Div`（`/`）を #2136 に含めるか
5. 案 B／C／D を不採用とすることの確認

## 13. 出典

- `crates/autodiff/src/var.rs:108-112, 662-701, 809-811, 873-888`
- `crates/tensor-core/src/scalar_op.rs:74, 151, 183`
- `crates/autodiff/src/tape.rs:2329-2350, 2395-2398`
- `crates/backend-cpu/src/fused_elementwise.rs:25-38`
- `crates/facade/src/lib.rs:184`
- `docs/compat-api-scope.md` §0・§1・§5
- `docs/facade-nn-module-exposure-decision.md`（同型の先例。#2132）
- rustc 1.98.1 の scratch クレートでの実測（orphan 規則・E0117・inherent／トレイトメソッド解決順序。§3・§7）
- spec REQ-9（`docs/spec/04-requirements.md`）
