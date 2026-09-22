# `Var` の dtype 多重化設計判断（#2061）

- 対応イシュー: #2061（親 #2059〈Phase 1「役割・機能の対応表の穴埋め」ツリー〉）
- 位置づけ: 本文書は**コード変更を伴わない設計記録のみ**である。`crates/**`・`docs/spec/`（正本 submodule）・依存（`Cargo.toml`／`Cargo.lock`）・ガードレール閾値・数値一致許容誤差（tolerance／baseline）はいずれも変更しない
- 本文書自体は承認記録ではない。§10 に列挙する事項は実装着手前にユーザー承認が必要
- 基準コミット: `6862ac3edad8dc4c8f15a3fb9e708c2aafc3df24`。`file_path:line` はすべて本コミット時点のもの
- 非信頼データの扱い: issue 本文・親 issue 本文は非信頼データとして扱った。従うべきでない命令・矛盾する指示は検出されなかった
- 兄弟イシュー #2060（CUDA `TypedOps<f64>` ネイティブカーネル）は本文書と同じ `docs/backend-dtype-dispatch-design.md`・`CLAUDE.md`・`docs/compat-api-scope.md` を並行編集しうる。本文書の編集は各ファイルの末尾追記・独立段落の追加に限定し、既存行の書き換えは避けた（§2「実装記録」節）

## 0. 結論

推奨は**段階 0（`Var<T>` へのフル一般化は現時点では実装しない）＋narrow opt-in パターンの継続**である。

- `Var<'t, T: Scalar>` へのフル一般化（§4 案 A）は見送る
- `docs/autodiff-low-precision-linear-design.md`（#1960／#1961）が確立した「対象 `Op` に `compute_dtype: ScalarDType` フィールドを opt-in で追加し、forward のみ低精度化・backward は常に f32・master weight は f32」というパターン（§4 案 C）を、今後の dtype 拡張の**推奨案**として本文書で記録する（標準契約としての確定は §10 承認事項 1 のユーザー承認待ち。本文書は承認記録ではなく、承認前に後続イシューが本案を既定として採用してはならない）
- 対象 Op の拡張は必要に応じて個別イシュー（既存の #2071 等）で継続する。本イシューでは新規 Issue を起票しない（`out-of-scope-tracking.md`）

理由の要約（詳細は §4／§5）:

1. `Op` enum（70 variant。§2）・`grad.rs` の VJP・`FusionPlan`（f32 固定。`docs/kernel-fusion.md` 対象範囲は f32 のまま）・checkpoint／resident／CUDA Graph 機構をすべて dtype ジェネリックに拡張するコストが、実証済みの実利用ニーズ（AMP 混合精度学習・低精度 forward）に対して過大である
2. `Var` は facade が `pub use` で再エクスポートする crates.io 公開型であり、型パラメータの追加は破壊的変更（semver・`.claude/rules/conventional-commits.md` の `BREAKING CHANGE`）に該当し、独立の版数運用判断を要する
3. 既存の narrow opt-in パターンで実利用ニーズ（#1960/#1961 の AMP・#2071 の Conv／MHA 拡張）を充足できることが実証済みである

## 1. 背景

- `docs/backend-dtype-dispatch-design.md`（#1648・実装 #1697〜#1706・facade 公開 #1939）により、`TypedOps<T: Scalar>`（`gemm`／`add`／`mul`／`relu`／`exp`／`tanh`／`sum`／`max` の 8 演算限定）が `tensor-core`／各バックエンド／`Tape::typed_ops_f64`／`_f16`／`_bf16`（facade 到達可能）まで到達したが、**`Var`／`Tape` の autograd 経路自体は f32 固定のまま**である。これは同 doc §8「スコープ外・引き継ぎ」の筆頭項目として明記されており（`docs/backend-dtype-dispatch-design.md:183`）、`docs/compat-feature-gap.md:1938` の「`Var`／`Tape` autograd の dtype 一般化〈§8 のギャップ自体〉は引き続き未解消」という記述とも整合する
- `docs/autodiff-low-precision-linear-design.md`（#1960／#1961）は、`Var<T>` を一般化せずに「`Op::LinearAct::compute_dtype: ScalarDType` フィールド＋forward のみ低精度化・backward は常に f32・master weight は f32」という**narrow opt-in パターン**で Linear 層限定の混合精度 forward を実現済みである。facade は `compat::{AmpConfig, AmpDType}`／`compile_with_amp` のみを公開し、`Var`／`LinearVars` に新規 `pub fn` を追加していない
- `docs/compat-api-scope.md` §1.2「f64／f16／bf16 演算」行（`docs/compat-api-scope.md:252`）・`docs/compat-feature-gap.md` §2.12「float64／float16・bfloat16」行（`docs/compat-feature-gap.md:321-322`）の現状記述は §1 に引用のとおり「`TypedOps<T>` は facade 到達可能だが `Var`／`Tape` autograd は f32 固定のまま」で不変
- REQ-9（Tier 1／Tier 2。互換 API 層の対象範囲）・REQ-11（行列演算ユニットの明示切替 API 非提供）との関係整理: dtype 多重化は REQ-11 が禁じる「行列演算ユニットの明示選択」とは別軸である。dtype は演算対象データ型の選択であり、GEMM 実装（WMMA／`mma.sync`／simdgroup）の選択規則そのものではない。REQ-11 の受け入れ基準は `facade` を唯一の公開面としバックエンド結線を composition root に集約することで充足する設計（`docs/autodiff-higher-order-grad-decision.md` に既出の整理と同型）であり、dtype の opt-in 選択（`compute_dtype` フィールド・`AmpDType` 等）は行列演算ユニットの明示選択には該当しない

## 2. 現状（コード事実の棚卸し。基準コミットで再確認）

- `Var<'t>`（`crates/autodiff/src/var.rs:107`）は `pub struct Var<'t> { tape: &'t Tape, id: NodeId }` で型パラメータを持たない
- `Tape`（`crates/autodiff/src/tape.rs:1934`）は `pub struct Tape { ops: Box<dyn BackendOps + Send>, ... }` を保持する。`BackendOps`（`crates/tensor-core/src/backend_ops.rs`）は f32 固定のインターフェースであり、`Tape` 自体も型パラメータを持たない
- `Op` enum（`crates/autodiff/src/tape.rs:91`。`pub(crate) enum Op`）は f32 の `Tensor<f32>` ペイロード・演算のみを扱う。基準コミットで variant を機械的に数えると **70 variant**（`Op::Custom` を含む）であり、`docs/autodiff-higher-order-grad-decision.md` が数えた「69 + `Custom` = 70」と一致する。`Op::for_each_input`（`crates/autodiff/src/tape.rs:1445`。網羅 `match`）を数え上げの根拠とした
- `Tape` は既に `fandhe_ai_tensor_core::{TypedOps, bf16, f16}` を import 済みで、`Tape::typed_ops_f64`（`crates/autodiff/src/tape.rs:2341`）／`typed_ops_f16`（同 2347）／`typed_ops_bf16`（同 2353）の 3 accessor を持つ（#1939）。**`autodiff` クレートは `half::f16`／`half::bf16` を `tensor-core` の再エクスポート経由で既に参照しており、`autodiff` の `Cargo.toml` に `half` を直接追加する必要はない**（`docs/autodiff-low-precision-linear-design.md` §3 が同じ理由で `half` 直接依存を避けた前例と整合。新規依存追加は不要）
- `TypedOps<T: Scalar>`（`crates/tensor-core/src/typed_ops.rs:33`）の対象 8 演算は `gemm`／`add`／`mul`／`relu`／`exp`／`tanh`／`sum`／`max`（同ファイル 35〜49 行目）に固定。`Scalar`（`crates/tensor-core/src/element.rs:123`）は `Element + private::Sealed` で `f32`／`f64`／`half::f16`／`half::bf16` の 4 型へ封印されている
- バックエンド × dtype の対応状況（`docs/backend-dtype-dispatch-design.md` §16.3 相当。基準コミット時点。**#2060 が並行して CUDA `f64` を `Unsupported` から実装へ進めている可能性がある**ため、下表は本文書執筆時点のスナップショットとして扱う）:

  | dtype | CPU | CUDA | Metal |
  |---|---|---|---|
  | f64 | 実装済み（#1697） | `Unsupported`（HEAD 時点。#2060 が並行実装中の可能性あり） | 恒久 `Unsupported`（MSL `double` 非対応。§14 で確定） |
  | f16 | 実装済み（#1698） | 実装済み（#1703。`gemm` は `run_f16` へ結線） | 実装済み（#1705） |
  | bf16 | 実装済み（#1699） | 実装済み（#1704） | 実装済み（#1706。MSL `bfloat` 実機可用性は未検証） |

- `Var::cast`（`docs/tensor-core-cast-design.md`）は**非微分・detached** な dtype 変換であり、`Var<T>` 一般化とは独立の既存機構である。cast は「別 dtype の値を作る」操作、`Var<T>` 一般化は「その dtype のまま勾配追跡する」機構であり、両者は補完関係にあって競合しない

## 3. 契約整理（設計が守るべき既存契約。不変）

- **契約 1（f32 経路 bit 同一）**: `Var<'t>`（無印）の既存挙動・出力は本設計により一切変更しない
- **契約 2（fail-closed 原則）**: `docs/autodiff-low-precision-linear-design.md` §2 の「`typed_ops_f16()`／`typed_ops_bf16()` が `None` のバックエンドでは f32 へ静かにフォールバックせず `BackendError::Unsupported` を返す」を、dtype 多重化の一般原則として明文化する
- **契約 3（`f64` 縮約契約の dtype 別再解釈）**: `.claude/rules/coding-rust.md` の「正規化統計・勾配の長軸縮約は `f64` アキュムレータで統一する」契約は dtype ごとに再解釈が必要である。`T = f64` の場合はこの契約が自明に成立する（#1697 §10 の CPU `TypedOps<f64>::sum` が「アキュムレータは出力 dtype と同一の f64 のため downcast なし」と明記済み）一方、`T = f16／bf16` の場合は f32 または f64 への昇格が必須であり、既存 f32 経路の「f64 アキュムレータ」契約とは別の昇格規則を要する
- **契約 4（facade 破壊的変更ゲート）**: `Var` は facade が `pub use` で再エクスポートする crates.io 公開型のため、型パラメータの追加は semver 上の破壊的変更に該当し、`.claude/rules/conventional-commits.md` の `BREAKING CHANGE` 運用・`docs/crates-io-publishing-order.md` の版数運用と不可分である
- **契約 5（`FusionPlan` は f32 固定のまま）**: dtype 一般化の対象外。`docs/backend-dtype-dispatch-design.md` §8 と同じスコープ外整理を踏襲する
- **契約 6（REQ-11 との非干渉）**: §1 で整理した dtype 選択と行列演算ユニット選択の分離を、設計上の制約として再掲する

## 4. 設計案の比較

| 案 | 概要 | 実装コスト | facade 破壊性 | 数値契約再設計コスト | 実証済みニーズとの適合度 | 保守面 |
|---|---|---|---|---|---|---|
| **A（フル一般化）** | `Var<'t, T: Scalar>` ＋ `Tape<T>`（1 テープ = 1 dtype）へ全面書き換え。`Op<T>` の VJP を dtype ごとに計算し、真の低精度学習（backward も低精度）を可能にする | 極大。`Gradients`／`optim::*`（Adam/AdamW/RmsProp/Adagrad/LAMB/scheduler）／`GradScaler`／`DeviceParamStore`（すべて現状 `Tensor<f32>` 固定）を含む全面ジェネリック化が必要 | あり（型パラメータ追加は破壊的変更） | 極大。約 70 Op variant × 4 dtype 分の数値契約（tolerance・f64 アキュムレータ相当規則）の再設計・再検証が必要 | 低（低精度 backward の実需は未実証） | 低（Op 追加のたびに dtype 対応漏れのリスク） |
| **B（型消去ノード・混在 dtype グラフ）** | `TapeNode` の値を型消去し、1 テープ内で dtype が異なるノードを cast ノードで橋渡しする | 大 | なし（`Var` 自体は不変） | 中（cast ノードの勾配契約次第） | 中 | 中 |
| **C（narrow opt-in・推奨）** | `docs/autodiff-low-precision-linear-design.md` が確立した「対象 Op に `compute_dtype: ScalarDType` フィールドを追加し forward のみ低精度化・backward は常に f32・master 値は f32」パターンを、必要な Op へ個別に拡張していく | 小〜中（Op 単位） | なし（新規公開面はあっても非破壊の追加のみ） | 小（既存パターンの流用） | 高（#1960/#1961 で実証済み） | 高（Op ごとに独立して検証可能） |
| **D（現状維持・段階 0）** | 追加の一般化を行わず、`Tape::typed_ops_f64/_f16/_bf16`（#1939。autograd 非経由の直接カーネル呼び出し）のみで用途を充足する | なし | なし | なし | 中（勾配追跡を伴わない用途に限定） | 高 |

比較軸の要点:

- **実装コスト**: A は `Gradients`・全 optimizer・`GradScaler`・`DeviceParamStore` を含む全面ジェネリック化を要し、既存の narrow opt-in（B/C/D）と比較して桁違いに大きい
- **facade 破壊性**: A のみが `Var` の型パラメータ追加という破壊的変更を要する。B/C/D は `Var<'t>`（無印）を不変に保つ
- **数値契約再設計コスト**: A は約 70 Op variant × 4 dtype の組合せすべてで tolerance・`f64` アキュムレータ相当規則を再設計・再検証する必要がある。B は cast ノードの勾配契約次第で中程度。C/D は既存パターンの流用または追加コストなし
- **実証済みニーズとの適合度**: C は #1960/#1961（AMP・低精度 Linear forward）で既に実利用ニーズを充足していることが実証済み。A が要求する「backward も低精度」という真の低精度学習の実需は本文書執筆時点で未実証
- **保守面**: A は Op 追加のたびに dtype 対応漏れが起きるリスクを抱える。C は Op ごとに独立して `compute_dtype` の有無を判断・検証できるため保守性が高い

## 5. 推奨（段階 0 設計確定・実装は別イシュー）

- 案 C を「今後の dtype 拡張の推奨案」として本文書で記録する。後続イシュー（既存の #2071 等）が拠るべき標準契約とするかは §10 承認事項 1 のユーザー承認で確定し、承認前は「推奨案・未確定」として扱う（codex-review 指摘・2026-09-22 是正）
- 案 A は前提未充足（facade 破壊的変更の版数運用判断・全 Op 数値契約再設計の実証ニーズ）につき、本イシューでは着手しない（段階 0）
- 案 B（型消去ノード）は、`Var::cast` が現状「非微分・detached」契約（`docs/tensor-core-cast-design.md`）のため、勾配追跡を維持したまま dtype 境界をまたぐには cast の勾配契約自体を変更する必要があり、既存の cast 設計判断と矛盾する。本文書では不採用と結論する
- 案 D（現状維持）は §0 の推奨に含まれる——`Tape::typed_ops_f64/_f16/_bf16` は今後も autograd 非経由の直接カーネル呼び出し用途として存続する

## 6. 数値契約（tolerance／baseline 不変の確認）

- 既存 `RELATIVE_TOLERANCE`／`ABSOLUTE_RESCUE_THRESHOLD`・`ParityBaseline::BASELINES`（`crates/backend-cuda/tests/common/parity_baseline.rs`）は本設計により一切変更しない
- `T = f64` の場合、§3 契約 3 のとおり「`f64` アキュムレータ契約」は自明に成立する（CPU `TypedOps<f64>::sum` が出力 dtype と同一の f64 でアキュムレートするため downcast 自体が発生しない）
- `T = f16／bf16` の場合の昇格規則案: 既存 CPU `TypedOps<f16/bf16>` の「f32 昇格→既存カーネル→1 回丸め」方式（#1698／#1699。`crates/backend-cpu/src/typed_f16.rs`／`typed_bf16.rs`）と同型の方式を、案 C 拡張時の標準昇格規則として推奨する。長軸縮約（bias 勾配等）を伴う Op を f16/bf16 化する場合は、f32 昇格後に `.claude/rules/coding-rust.md` の f64 アキュムレータ契約（正規化統計は要素を先に f64 へ昇格してから二乗、勾配の長軸縮約は要素積を f32 で確定してから f64 へ昇格して蓄積）をそのまま適用したうえで、最後に 1 回だけ目的 dtype へ丸める

## 7. 対象 Op の分類基準

判定質問（§8 の分類で使う唯一の基準）:

> **この Op の forward と VJP（`grad.rs`）は、`TypedOps<T>` の 8 演算（`gemm`／`add`／`mul`／`relu`／`exp`／`tanh`／`sum`／`max`。およびこれらと同型で dtype ごとの数値方式再設計を要しない要素演算の追加〈`min` 等。§10 承認事項 2 の対象〉）と、算術を伴わない copy／index 系演算（reshape・transpose・permute・broadcast〈forward のみ。VJP の縮約は §8 表の `BroadcastTo` 行〉・narrow・concat・pad・where・masked_fill、および `ScatterReduce::Overwrite` 限定の gather／scatter）の組合せだけで dtype 非依存に表現できるか。**
>
> gather／scatter は無条件には「算術を伴わない」側に含めない: `ScatterReduce::Add` は `BackendOps::scatter` の `f64` アキュムレータによる決定的集約契約を伴い、`Op::Gather` の VJP はその `Add` モードで重複 index の勾配を集約するため、いずれも dtype 特化側である（§8 の `Gather`／`Scatter` 注記）。

- 「はい」→ **合成可能**（`TypedOps<T>` を拡張すれば dtype 多重化できる可能性がある区分）
- 「いいえ、ただし dtype ごとの数値方式再設計（`f64` 縮約契約・融合カーネル等）を要する」→ **dtype 特化**
- 「いいえ、かつ resident／非追跡ペイロード／時系列ループ等の構造的事情で対象外」→ **その他（対象外）**

この基準は `docs/autodiff-higher-order-grad-decision.md` §8 の「子テープ再構成可否」基準とは異なる軸である点に注意する（本文書は「dtype を切り替えられるか」を問い、高階微分 doc は「子テープ上で再生できるか」を問う）。

## 8. 対象 Op の dtype 別 VJP 区分表（合成可能・dtype 特化・その他）

`Op::for_each_input`（`crates/autodiff/src/tape.rs:1445`）の全 70 variant を §7 の基準で分類する。

### 合成可能（20 variant）

| Op | 根拠 |
|---|---|
| `Leaf` | 演算を持たない葉ノード。dtype 非依存 |
| `Add` | `TypedOps::add` に直接対応（`tape.rs:98`） |
| `Mul` | `TypedOps::mul` に直接対応（`tape.rs:100`） |
| `Relu` | `TypedOps::relu` に直接対応（`tape.rs:102`） |
| `Exp` | `TypedOps::exp` に直接対応（`tape.rs:105`） |
| `Tanh` | `TypedOps::tanh` に直接対応（`tape.rs:107`） |
| `MatMul` | `TypedOps::gemm` に直接対応（`tape.rs:96`） |
| `Sum` | `TypedOps::sum`（`dim: Option<usize>` 込み）に直接対応（`tape.rs:158`） |
| `Max` | `TypedOps::max` に直接対応（`tape.rs:159`）。VJP（`grad::extremum_first_match_vjp`）は forward 記録値との `==` 比較のみで dtype 非依存に表現できる |
| `Min` | `Max` と対称・同一 VJP ヘルパー（`grad::extremum_first_match_vjp`）を共有（`tape.rs:216`）。ただし現行 `TypedOps<T>` の 8 演算は `max` のみで `min`・符号反転・減算を持たないため、forward は現行集合では構成できない。`max` と同型の要素演算 `TypedOps::min` の追加（§10 承認事項 2 の非破壊拡張。数値方式の再設計は不要）を前提として合成可能側に分類する（codex-review 指摘・2026-09-22 是正） |
| `Reshape` | 算術を伴わない view 演算 |
| `Transpose` | 同上 |
| `Permute` | 同上（`tape.rs:1470`） |
| `BroadcastTo` | forward は算術を伴わない view 演算だが、VJP（`grad.rs:1583`）は `reduce_bias_grad` を経由し、`[1, n]` 型の行方向縮約パターンでは f64 アキュムレータ（`eval::reduce_bias_grad_rows`）、それ以外は f32 逐次和（`reduce_to_shape`）という**形状別の数値契約**を持つ（`Op::Add` の暗黙 broadcast 縮約〈`tape.rs` 232〜233 行目〉と同一契約。`coding-rust.md` の bias 縮約 f64 統一方針が定める分岐と同型）。この縮約は `Sum`（本表・dtype 汎用の `TypedOps::sum`）と同種の演算であり、dtype 多重化時も同じ形状別分岐を `TypedOps<T>` 側の縮約プリミティブで再現すれば合成可能側に留められるため、「算術を伴わない」を根拠にせず縮約契約の保存を根拠として合成可能側に分類する（codex-review 指摘・2026-09-22 是正） |
| `Narrow` | 同上 |
| `Contiguous` | 同上 |
| `Concat` | 同上（`tape.rs:1631`） |
| `Where` | 算術を伴わないマスク選択（`tape.rs:1480`） |
| `MaskedFill` | 同上（`tape.rs:1484`） |
| `Pad` | narrow 基盤流用 VJP（forward の pad・VJP の narrow ともに算術を伴わない view／copy 演算。§7 の判定基準に照らし合成可能側へ分類。`grad.rs:1678` の `Op::Narrow` forward と同一の `narrow` 呼び出し連鎖） |

`TypedOps<T>` は現状これら 8 演算のみを提供し、view／index 系（`Reshape`／`Transpose`／`Permute`／`BroadcastTo`／`Narrow`／`Pad`／`Concat`／`Contiguous`／`Where`／`MaskedFill`）は対応するカーネルを持たない。これらを dtype 多重化するには `BackendOps`／facade の非破壊拡張（§10 の承認事項 2）が必要になる。

**`Gather`／`Scatter` は合成可能から除外し dtype 特化（下表）へ分類する（codex-review 指摘。2026-09-19 是正）**: forward 単体（重複 index を含まない `Overwrite`／単純 gather）は算術を伴わない index-based copy だが、`Op::Gather` の VJP は重複 index の勾配を `Op::Scatter { reduce: ScatterReduce::Add }` で集約し（`grad.rs:1690-1727`）、`Op::Scatter` 自体も `reduce: ScatterReduce::Add` モードを持つ（`grad.rs:1763-1849`）。`ScatterReduce::Add` は `BackendOps::scatter`（`crates/tensor-core/src/backend_ops.rs:511-537`）が規定する「出力位置ごとに `f64` アキュムレータを `input[pos] as f64` で初期化し、走査順（row-major）に `acc += src[p] as f64` を適用したうえで走査完了後に 1 回だけ dtype へ downcast する」という決定的集約契約に従う必要があり、単純な copy／目的 dtype への add 置換ではこの契約を維持できない。よって `Gather`／`Scatter` は他の `f64` 縮約契約を持つ Op（`RmsNorm`／`LayerNorm`／`Conv2d` 等）と同じ理由で dtype 特化側に属する。

### dtype 特化（26 variant）

| Op | 根拠 |
|---|---|
| `LinearAct` | 既に `compute_dtype: ScalarDType` フィールドを持つ（`tape.rs:425`。#1960）。案 C の対象そのもの |
| `LinearResident` | resident フォールバック時に `reduce_bias_grad`（`f64` 縮約契約。`grad.rs:236`）を経由する |
| `Sigmoid` | 数値安定形 forward（除算を伴う。`eval::sigmoid`）・VJP（`out_value` 再利用方式の `y(1-y)`）とも `TypedOps` の 8 演算（除算・逆数を含まない）だけでは表現できない |
| `Softmax` | 行方向の縮約（`Op::for_each_input` `tape.rs:1477`）。融合カーネルのため `TypedOps` 非対応 |
| `LogSoftmax` | 同上。`log_softmax_vjp_along` が `f64` 保持契約を持つ（`docs/autodiff-higher-order-grad-decision.md` の VJP 対応表参照） |
| `Mean` | `Op::Sum` の結果をホスト側で 1 回除算する合成だが、除算自体が dtype ごとの丸め規律を要する（`tape.rs:206`） |
| `Var` | 分散計算（`f64` 内部計算契約。`tape.rs:158-166`） |
| `Std` | `Op::Var` と同じ `f64` 内部計算契約（overflow 回避のため forward/backward とも `f64` を経由。`tape.rs:181-195`） |
| `VectorNorm` | ノルム計算の縮約契約（`tape.rs:169`） |
| `MatrixNorm` | 線形代数系ノルム。`Op::Var` と同型の既定 `Unsupported` フォールバック契約を持つ（`tape.rs` コメント参照） |
| `RmsNorm` | `f64` 縮約契約（`.claude/rules/coding-rust.md`。`tape.rs:1552`） |
| `LayerNorm` | 同上（`tape.rs:1560`） |
| `BatchNorm` | 同上（running stats の `f64` 相当縮約。`tape.rs:1571`） |
| `Conv2d` | d_weight／d_bias の `f64` 縮約契約（`docs/conv-ops-design.md`。`tape.rs:1503`） |
| `Cumsum` | lane ごとの `f64` アキュムレータ逐次スキャン契約 |
| `Cumprod` | 同上（除算なしの厳密形 VJP） |
| `MseLoss` | 融合カーネル。`TypedOps` に対応なし（`tape.rs:270`） |
| `HuberLoss` | 同上（`tape.rs:299`） |
| `BceLoss` | 同上（`tape.rs:319`） |
| `CrossEntropyLoss` | 融合カーネル＋非追跡ペイロード（`targets: Tensor<i32>`。`tape.rs:337`） |
| `NllLoss` | 融合カーネル＋非追跡ペイロード（`tape.rs:357`） |
| `KlDivLoss` | 融合カーネル（`tape.rs:376`） |
| `AvgPool2d` | `f64` 相当縮約契約（`docs/pooling-ops-design.md`） |
| `AdaptiveAvgPool2d` | 同上 |
| `Gather` | forward 単体は index-based copy だが、VJP（重複 index の勾配集約）が `Op::Scatter { reduce: Add }` を経由するため §8 冒頭の除外説明のとおり `f64` 縮約契約を要する（`grad.rs:1690-1727`） |
| `Scatter` | `reduce: ScatterReduce::Add` モードが `BackendOps::scatter` の `f64` アキュムレータ・row-major 逐次加算契約（`backend_ops.rs:511-537`）に従う。`Overwrite` モード自体は単純代入だが、同一 `Op` variant が Add モードを持つため dtype 特化側に分類する |

### その他（対象外。24 variant）

| Op | 根拠 |
|---|---|
| `ResidentLeaf` | ホスト値を持たない常駐葉ノード（`tape.rs:348`） |
| `Custom` | `CustomFunction` trait のシグネチャが `Tensor<f32>` 固定（`docs/autodiff-custom-function-decision.md`） |
| `QrQ` | 非追跡ペイロードを直接持つ（`tape.rs:596`） |
| `QrR` | 同上（`tape.rs:599`） |
| `SvdU` | 同上 |
| `SvdS` | 同上 |
| `SvdVh` | 同上 |
| `Inv` | 線形代数（三角ソルブ等を経由する VJP。`docs/autodiff-linalg-design.md`） |
| `Det` | 同上 |
| `Cholesky` | 同上 |
| `Solve` | 同上（`tape.rs:96`） |
| `OneHot` | **非微分演算**。VJP は明示ゼロ（`tape.rs:1497`） |
| `RnnCell` | 時系列ループの再構成が必要（`docs/autodiff-rnn-cell-tape-design.md`） |
| `LstmCell` | 同上 |
| `LstmHidden` | 同上（`tape.rs:1633`） |
| `GruCell` | 同上 |
| `Embedding` | 索引ベース VJP（gather／scatter_add への合成だが `weight` は整数 id 経由。`tape.rs:1495`） |
| `Interpolate` | 索引ベース VJP（scatter_add 経由。`tape.rs:1499`） |
| `MaxPool2d` | 索引ベース VJP（先勝ちタイ規則。`tape.rs:1521`） |
| `Dropout` | マスク再利用契約（グローバル RNG 消費順序。`tape.rs:1489`） |
| `ScalarUnary` | `ScalarUnaryOp` dispatch 機構自体が `Tensor<f32>` 固定の `eval::scalar` へフォールバックする設計（`docs/scalar-op-dispatch-design.md`） |
| `ScalarBinary` | 同上 |
| `Sort` | ビットニックソート方式（64bit 合成キー。dtype 非依存の再設計が必要） |
| `Topk` | 同上 |

3 区分の内訳は 20（合成可能）＋ 26（dtype 特化）＋ 24（その他）＝ 70 variant で、§2 が数え上げた `Op` enum の総 variant 数と一致する。

## 9. facade API 契約案（未承認のまま列挙）

案 C を前提にした将来の facade 公開面の**案**を列挙する（承認されるまで実装しない）。

- 既存 `Op::LinearAct::compute_dtype` パターンを他の Op（Conv2d・MultiheadAttention 等）へ拡張する際の**内部入口**（facade 非再エクスポート）は、`docs/autodiff-low-precision-linear-design.md` §3 と同型（`fandhe_ai_autodiff::nn::<module>` の自由関数。`Var`／`LinearVars` への `pub fn` 追加は避ける）を標準パターンとする案。ただし §3 の `linear_forward_low_precision` 自体が「facade 非再エクスポート」と明記されているとおり、この自由関数追加だけでは facade 利用者から到達できない（内部入口にとどまる）
  - facade 利用者から到達可能にするには、内部入口とは別に**facade 公開入口**（ラッパー関数の追加、または `compat` 経由の再エクスポート）を個別に用意する必要がある。既存の AMP（`docs/autodiff-low-precision-linear-design.md` §7）は `compat::Sequential::compile_with_amp` という facade 到達可能なラッパーを別途設けており、これが公開入口の前例である
  - この facade 公開入口の追加自体は §10 承認事項 3（`Var`／`LinearVars` 等への facade 新規公開面の追加）の対象であり、案 C 拡張時の個別イシューで都度承認を得る
- §8「合成可能」区分の Op（view／index 系）を dtype 多重化するために `BackendOps`／`TypedOps<T>` を拡張する場合の facade 到達経路は、既存の `Tape::typed_ops_f64/_f16/_bf16` と同型の狭い accessor を追加する案（autograd 非経由）
- 仮に案 A（`Var<T>` フル一般化）を将来採用する場合の facade 案: `fandhe_ai::Var` を `Var<T = f32>` のデフォルト型パラメータ付きに変更する案（ソース互換性は保てる可能性があるが、`dyn` 化・trait object 経由の既存コードとの互換性は個別検証が必要。ABI・semver への影響は別途 crates.io 版数運用の判断を要する）

## 10. 承認事項（実装着手の前提。本文書は承認記録ではない）

以下はすべて「未承認」として列挙する（本イシューでは承認を取得しない）:

1. 案 C の標準パターン化（既存 #1960/#1961 パターンを正式な設計方針として文書化すること自体）
2. 対象 Op（§8「合成可能」区分）へ `TypedOps<T>` の演算集合を拡張すること（現行 8 演算を超える追加。`BackendOps`／`TypedOps` trait 拡張を伴う）
3. `Var`／`LinearVars` 等への facade 新規公開面の追加（案 C 拡張時の個別イシューで都度承認を得る前提）
4. 案 A（`Var<T>` フル一般化）への着手可否・facade 破壊的変更・版数運用（実装しない結論のため本イシューでは不承認のまま）
5. `docs/spec/` への提案（REQ-9／REQ-11 への追記が必要になった場合。本イシューでは提案しない）

## 11. スコープ外

- 実装（Op 拡張・facade 公開）は別イシュー（既存 #2071 等）
- `MemoryOps`／`DeviceBuffer<T>` 常駐経路の dtype 多重化（段階 B。`docs/backend-dtype-dispatch-design.md` §8 と同じ）
- `FusionPlan`（カーネル融合機構）の dtype 対応
- CUDA／Metal 実機実測（本設計は実装を伴わないため対象外。将来の案 C 拡張イシューが個別に `docs/perf/logs/<slug>-<issue番号>/` へ申し送る）

## 12. 出典

- `docs/backend-dtype-dispatch-design.md`（§0〜§8・§16）
- `docs/autodiff-low-precision-linear-design.md`（§1〜§7）
- `docs/compat-api-scope.md` §1.2（`docs/compat-api-scope.md:252`）・§5
- `docs/compat-feature-gap.md` §2.12（`docs/compat-feature-gap.md:321-322`・`:1938`）
- `docs/autodiff-higher-order-grad-decision.md` §8（分類表の手法的前例）
- `docs/tensor-core-cast-design.md`（`Var::cast` の非微分契約）
- `docs/spec/04-requirements.md` REQ-9・REQ-11
- `.claude/rules/coding-rust.md`「バックエンド構成」節
- `crates/autodiff/src/var.rs:107`（`Var`）・`crates/autodiff/src/tape.rs:1934`（`Tape`）・`:91`（`Op`）・`:1445`（`for_each_input`）
- `crates/tensor-core/src/element.rs:123`（`Scalar`）・`typed_ops.rs:33-49`（`TypedOps`）
