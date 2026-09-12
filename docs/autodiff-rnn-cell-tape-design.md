# RNN／LSTM／GRU セル演算・時系列ループの tape 設計（イシュー #1646）

- 対応イシュー: #1646（親 #1619「RNN／LSTM／GRU を追加する」→ Phase 3 #1573〈Tier 2〉→ ルート #1570）
- 位置づけ: 本文書は**設計判断のみ**を記録する。`crates/` 配下・`docs/spec/`（正本 submodule）へのコード変更は行わない。実装は兄弟イシュー #1647（3 バックエンド実装）へ引き渡す
- 基準コミット: `origin/main` `50120e4c`（#1656 で spec submodule 追従・#1654 で `docs/compat-feature-gap.md` 取り込み済み時点）。行番号・事実はすべて本コミットで再確認した

## 1. 背景

- 機能ギャップ表 `docs/compat-feature-gap.md` §2.7 の RNN/LSTM/GRU 行は「なし・難度 XL」で、必要物として (a) ゲート演算（sigmoid／tanh は既存）、(b) 時系列ループを既存の動的テープへどう積むかの設計、(c) 3 バックエンド実装、が挙げられている
- spec 側は 2026-09-12 に REQ-9 を改定し（`docs/spec/04-requirements.md` REQ-9・Fandhe-AI/fandhe-ai-spec#66〈PR #69〉）、RNN／LSTM／GRU を **Tier 2**（長尾。PyTorch／TensorFlow 機能網羅の第 2 段階）に明記した。実装リポ側の `docs/compat-api-scope.md` §1／§2／§5 更新は別イシュー #1591 の管轄で、本イシュー時点では **OPEN**
- 目的: (a)(b) の設計を確定し、判断根拠とともに doc として記録する。tolerance／baseline（数値一致の許容誤差・非後退ベースライン）の変更は対象外

## 2. 現状のコード事実（`origin/main` `50120e4c`）

| 事実 | 出典 |
|---|---|
| `Op` enum（`pub(crate)`、`#[derive(Debug, Clone)]`。`#[non_exhaustive]` ではない）は `Leaf`／`MatMul`／`Add`／`Mul`／`Relu`／`Exp`／`Tanh`／`Sigmoid`／`Sum`／`Max`／`MseLoss`／`CrossEntropyLoss`／`ResidentLeaf`／`LinearResident`／`LinearAct`／`Reshape`／`Transpose` の 16 variant。`Var` メソッドとほぼ 1:1 対応する | `crates/autodiff/src/tape.rs:86-270` |
| 登録経路は 3 種: `push_eager`（実体化済み値を持つ。`tape.rs:702`）・`push_lazy`（elementwise 5 演算 `Add/Mul/Relu/Exp/Tanh` のみ。`MAX_FUSED_CHAIN_LEN = 6` 到達で実体化。`tape.rs:834`）・`push_view`（`Reshape/Transpose`。値を持たず backward 時に `resolve_view` で再導出。`tape.rs:759`） | `tape.rs:400`（`is_lazy_elementwise`）・`tape.rs:413`（`is_view`）・`tensor-core::MAX_FUSED_CHAIN_LEN` |
| `Sigmoid` は融合対象外（`push_eager`。`BackendOps` に対応メソッドなし。forward は `eval::sigmoid`〈ホスト scalar 参照実装〉、GPU 選択時も `BackendOps` を経由しない） | `tape.rs:105-110`・`crates/autodiff/src/eval.rs:355` |
| backward は Wengert list の逆走査。同一入力 `NodeId` へ複数経路から流入した勾配は `accumulate()`（`eval::add` で合算）が fan-in を自動蓄積する | `crates/autodiff/src/backward.rs:132-246,237`（`accumulate`） |
| `grad::vjp(op, out_value, upstream, nodes, ops, resident, tape_id, tape_epoch)` は当該ノードの `out_value`（forward 出力）・`upstream`（出力側勾配）を直接受け取る。**加えて `nodes: &[TapeNode]` 全体と `materialize_fallible(nodes, ops, node_id)` を持つため、自ノードの入力 `NodeId`（テープに既に登録済みのノード）を辿って値を再取得できる**。ただし自ノードの forward 内部でのみ算出され独立ノードとして登録されなかった中間値（ゲート値等）は、`Op` payload に保持しない限り VJP から参照できない | `crates/autodiff/src/grad.rs:52-84`・`crates/autodiff/src/tape.rs:1305`（`materialize_fallible`） |
| `matmul_out_shape` は**2 次元限定**（rank≥3 は `RankMismatch`）。`Tensor::narrow`（`tensor-core`）は存在するが `Var::narrow` は未接続（#1599。#1619 の宣言依存外） | `crates/tensor-core/src/ops_shape.rs:43`・`crates/tensor-core/src/tensor.rs:468` |
| `Var::add` は bias broadcast 対応（VJP は `reduce_to_shape`）。`Var::sub`／`Var::neg` は未実装（#1593。#1619 の宣言依存） | `crates/autodiff/src/var.rs:339` |
| `Tape::reset` は「最初の非葉ノード記録**前**に登録した葉」のみ保持（葉プレフィックス）。forward 中に登録した葉は reset で破棄される。`Gradients` は世代（`epoch`）で無効化される | `tape.rs:679`（`reset`）・`backward.rs:75-83`（`Gradients::get`） |
| reuse 経路の `Op::LinearResident` VJP は `fill_resident_weight_grad` が staging slot へ**上書き**で書き込み（`gemm_fp32_strict_into`）、成功時は `Gradients` へ d_weight を含めない。同一 weight を複数ノードが参照した場合の累積（β=1 相当の加算）契約は存在しない | `crates/autodiff/src/grad.rs:208-364`・`docs/device-resident-update-design.md` §3.3b |
| `Tape` は `Send` 必須（`tests/fusion_backend_integration.rs::tape_is_send`）。`DeviceBuffer` を `TapeNode` へ持たせられない | `crates/autodiff/tests/fusion_backend_integration.rs:390` |
| `Module` trait は `forward(tape, input)` と tape 不要の `forward_host(ops, input)`（既定 `Unsupported`。`predict` 経路） | `crates/autodiff/src/nn/module.rs:31` |
| `compat::Sequential` は `Linear`／`ReLU`／`Sigmoid`／`Tanh` の平坦な鎖（2 次元入力、`Vec<Box<dyn Module>>`）。`add_*` の拡張は #1618 の管轄 | `crates/facade/src/compat/sequential.rs:96-150` |
| facade 公開面は `crates/facade/tests/api_surface.rs` が機械検査する（`pub use` での `Tape`／`BackendOps`／`new_with_ops` の再エクスポート禁止・`pub fn` が `BackendOps` を直接引数に取ることを禁止） | `crates/facade/tests/api_surface.rs:1-145` |
| `Gradients::get` は未到達ノードで `Ok(None)`・別世代で `Err(TapeMismatch)` | `backward.rs:75-83` |
| 数値契約: REQ-2 統一複合判定（相対誤差 1e-3 未満 または 絶対誤差 1e-5 未満）・FMA 契約（CPU 参照実装は `f32::mul_add`）。正規化統計の f64 アキュムレータ規則（`.claude/rules/coding-rust.md`）はゲート演算の対象外（長軸縮約ではなく GEMM＋elementwise のため） | `.claude/rules/coding-rust.md` |
| REQ-9 は 2026-09-12 追記で RNN／LSTM／GRU を Tier 2 に明記。個別機能の設計は実装リポの各 issue（Phase 3 #1573）で決定するとしている | `docs/spec/04-requirements.md` REQ-9（2026-09-12 追記） |
| `docs/compat-api-scope.md` §5「範囲拡張の手続き」: facade 公開面の対象範囲拡張は (1) 正本 spec 側の REQ-9 改定、または (2) 本リポのユーザー承認を得た Issue 起票・本文書の更新のいずれかを必須とする | `docs/compat-api-scope.md:347-355` |
| `Activation` enum は `None`／`Relu` の 2 値（epilogue 融合専用。ゲート活性化とは無関係） | `crates/tensor-core/src/backend_ops.rs:106-111` |
| 設計文書の先例フォーマット: `docs/autodiff-nograd-leaf-dinput-skip-decision.md`（#1219）。ヘッダ 3 行 → 背景 → 現状のコード事実表 → 契約整理 → 案比較・採用案 → スコープ外 → 承認事項・起票草案 → 引き継ぎ → 出典 | `docs/autodiff-nograd-leaf-dinput-skip-decision.md` |

## 3. 契約整理（設計が守るべき既存契約）

1. `Op` の 1 variant は 1 つの `Tensor<f32>` 値を持つ（`TapeNode::value`）。複数の再帰出力（LSTM の `h_t`／`c_t`）は 1 ノードでは表現できない
2. `vjp()` は自ノードの入力 `NodeId`（テープに既に登録済み）を `materialize_fallible` で再取得できるが、**独立ノードとして登録されなかった中間値**（セル内部だけで算出されるゲート値）は `Op` payload に保持しない限り参照不能（§2 の `grad.rs:52-84` 事実）
3. `Tape` は `Send` を維持し、`TapeNode`／`Op` に `!Send` な GPU バッファを持たせられない（決定 7 の reuse スコープ判断の前提）
4. `Tape::reset` は葉プレフィックスのみ保持し、per-step で登録した葉（`x_t`・`h0`）は毎 forward で再登録が必要（決定 3）
5. facade 公開面の拡張（`fandhe_ai` 再エクスポート・`compat::Sequential::add_lstm` 等）は `docs/compat-api-scope.md` §5 の手続きが必須（決定 10）
6. 数値契約は REQ-2 複合判定・CPU 参照は `f32::mul_add`。tolerance／baseline は変更しない（決定 8）
7. `Op` の入力列挙（`effective_subtree_size`・`build_lazy_plan`・`grad::vjp` 等）はいずれも個別 `match` 実装で、汎用 `Op::inputs()` ヘルパは存在しない。新 variant 追加はこれら全箇所への追記が必要（決定 1b 付随事項）

## 4. 決定事項

### 決定 1: セル演算の表現（合成 vs 専用 Op）

**候補 A（合成）**: 既存 `Var` プリミティブ（`matmul`／`add`／`mul`／`sigmoid`／`tanh`）の合成のみで表現。追加 `Op` variant なし。

**候補 B（専用 Op）**: 専用 variant `Op::RnnCell`／`Op::LstmCell`／`Op::GruCell`。forward は `BackendOps::{rnn,lstm,gru}_cell`（既定 `Unsupported`）＋ `eval.rs` ホスト参照実装フォールバック、VJP は `BackendOps::*_cell_backward` ＋ホストフォールバックの二段構え（`docs/compat-feature-gap.md` §4 テンプレートと同型）。

判断材料:

- 候補 A は LSTM 1 step あたり約 20 ノード以上（× T）になり、`Sigmoid` が `push_eager` のため融合が細切れになる（`MAX_FUSED_CHAIN_LEN = 6` の恩恵を受けない）
- 候補 B は VJP でゲート値（i／f／g／o、または r／z／n）が必要になる。§2 で確認したとおり `vjp()` は `nodes`／`materialize_fallible` を通じて**自ノードの入力**（x_t・h_prev・weight・bias。いずれもテープに既存登録済み）を再取得できるため、**ゲート値そのものを payload に保存せず、VJP 内で入力から forward と同じ GEMM＋活性化を再計算する**ことも可能（再計算方式）。一方 payload 保存方式はメモリコストと引き換えに再計算 GEMM を避けられる
- 再計算方式は「forward の GEMM をもう一度 backward で実行する」ため、backward の演算量が実質 2 倍（GEMM 分）になる。payload 保存方式はメモリ O(T·B·G·H)（G = ゲート数: RNN=1, LSTM=4。GRU は決定 1c により再帰側アフィン値 `q` を追加保持するため実効 G=4）を要するが GEMM の再計算は不要
- `push_view`（`Reshape`／`Transpose`）が採用する「値を持たず再導出する」方式（`resolve_view`）は shape のみの軽量な再導出であり、GEMM を伴うセル演算とはコスト構造が異なる。将来の activation checkpointing（#1624）はこの再計算方式の一般化に相当する

**採用案**: **候補 B（専用セル Op）を v1 として採用し、ゲート値は `Op` payload に非追跡 `Tensor<f32>` として保持する（payload 保存方式）**。理由: (1) v1 は正しさとテストの単純さを優先し、GEMM 再計算による誤差蓄積・実装複雑度の増加を避ける、(2) `CrossEntropyLoss { targets: Tensor<i32> }` の前例（非追跡 payload）と整合する、(3) メモリコスト O(T·B·G·H) は T・B・H が小さい典型的な RNN 用途では許容範囲。将来 T・B・H が大きくメモリが問題になる場合は再計算方式（GEMM 再計算方式。backward コストは payload 保存方式より大きい）への切り替えを別イシューで検討する。

候補 A（合成）は**参照実装**として doc に式ごと残し（§4 決定 12「式の定義」）、#1647 のテストで「専用 Op と合成の複合判定一致（数値微分含む）」を要求する対照実装に位置づける。

### 決定 1b: LSTM の 2 出力（h_t・c_t）の tape 表現

**問題**: テープは 1 ノード 1 値。LSTM は再帰出力 `h_t`（次層・損失へ渡す）と `c_t`（次 step のセルへの微分可能な入力）を持つ。`c_t` を payload にのみ置くと `NodeId` を持たず、`dL/dc_t` が次セルの VJP からも `h_t = o ⊙ tanh(c_t)` 経由でも流入するにもかかわらず時間方向の勾配が欠落し BPTT が誤る。RNN／GRU は単一出力のため本問題は生じない。

**候補 (a)**: 1 ノードの値を `[B, 2H]`（`h ‖ c` 連結）とし、次セルは両半分を読む。損失・出力へ渡すには列方向の narrow が必要 → `Var::narrow`（#1599）が LSTM の必須依存になる。

**候補 (b)**: 2 ノードを順に push する。`Op::LstmCell { x, h_prev, c_prev, w_ih, w_hh, b_ih, b_hh, gates_ifg }` → 値 `c_t`。続けて `Op::LstmHidden { cell: NodeId(c_t), gate_o }` → 値 `h_t`。逆走査は `LstmHidden` を先に処理し、その VJP が `dc_t`（`o ⊙ (1 − tanh²(c_t))` 経路。§2 契約 2 のとおり `c_t` は `LstmHidden` の入力 `NodeId` なので `materialize_fallible` で取得可能）を `cell` ノードへ、`gate_o` 寄与を `x`／`h_prev`／`W`／`b` へ返す。次に `LstmCell` の VJP が `i`／`f`／`g` 寄与を扱う。重み `NodeId` の共有は `backward.rs::accumulate` の fan-in で合算される。narrow 不要・backward GEMM はゲートブロック単位に分かれるだけで総 FLOP は融合 backward とほぼ同じ。

**候補 (c)**: 同じ入力を読む独立 2 ノード（`h_t` と `c_t` を別々に forward）。正しいが backward GEMM が重複する。

**採用案**: **候補 (b)**。`gates_ifg`（i／f／g の 3 ゲート）は `LstmCell` payload、`gate_o`（o ゲート）は `LstmHidden` payload に保持する（`c_t` 自体は payload に持たず `LstmHidden` の入力 `NodeId` として参照する。決定 1 の payload 保存方式と整合）。**push 順序（cell → hidden）が VJP 逆走査順序の前提**であることを契約として明記する。

**付随事項（#1647 への引き渡し）**: セル variant は入力 `NodeId` を約 7 個（LSTM）持つ。§3 契約 7 のとおり汎用 `Op::inputs()` は存在しないため、入力を列挙する全 `match`（`Tape::effective_subtree_size`〈`tape.rs:872`〉・`build_lazy_plan`〈`tape.rs:943`〉・`grad::vjp`・`Op::is_lazy_elementwise`／`is_view` の否定分岐〈新 variant はいずれも `false` を返す＝eager・非 view〉）へ新 variant を追加する必要がある。

### 決定 1b 追記: `LstmHidden` が出力ゲートの入力 `NodeId` を取得する方法（Cursor Bugbot・codex-review 指摘。PR #1662）

**問題**: 決定 1b の採用案は「`LstmHidden` の VJP が `gate_o` 寄与を `x`／`h_prev`／`W`／`b` へ返す」と記すが、`Op::LstmHidden { cell: NodeId(c_t), gate_o }` は入力 `NodeId` を `cell` の 1 個しか保持しない。`gate_o`（o ゲート値）は非追跡 `Tensor<f32>` payload であり、o ゲートの逆伝播（`d(pre_o) = dh_t ⊙ tanh(c_t) ⊙ σ'(pre_o)` から `x`／`h_prev`／`w_ih`／`w_hh`／`b_ih`／`b_hh` への GEMM VJP）に必要な `x`・`h_prev`・重み・bias の `NodeId` がどこにも保持されておらず、このままでは実装不能である（Cursor Bugbot 指摘・High severity: 出力ゲート勾配が欠落し LSTM の BPTT が壊れる。codex-review 指摘: NodeId 取得方法が設計文書に未記載）。

**採用案**: `LstmHidden` に新規フィールドを追加せず、**`cell`（`NodeId(c_t)`）が指す先のノードが必ず `Op::LstmCell` である**という決定 1b の push 順序契約（cell → hidden。逆走査では hidden → cell の順で処理される）を利用する。`LstmHidden` の VJP 実装は `nodes[cell.0].op`（§2 事実のとおり `grad::vjp` は `nodes: &[TapeNode]` 全体を受け取るため、自ノードの直接入力でない `NodeId` の先もこの経路で到達できる）を参照し、そこに保持された `Op::LstmCell { x, h_prev, w_ih, w_hh, b_ih, b_hh, .. }` の各 `NodeId` を読み出す。読み出した `x`／`h_prev`／`w_ih`／`w_hh`／`b_ih`／`b_hh` へ、o ゲート由来の勾配（決定 5 の列ブロック配置に従い `w_ih`／`w_hh`／`b_ih`／`b_hh` の o 列ブロックのみ）を `accumulate()`（`backward.rs` の fan-in 蓄積。§2 事実）で足し込む。この参照は `cell` ノードが `Op::LstmCell` 以外であることを想定しない不変条件に依存するため、実装は `match` の `_` 分岐で型付きエラー（fail-closed。`unwrap`／`expect` は使わない。決定 11 (g) の方針と整合）を返し、パニックしない。

この設計により `Op::LstmHidden` の payload・フィールド構成は決定 1b の採用案（`cell`・`gate_o` のみ）から変更しない。§3 契約 2 の「自ノードの入力 `NodeId` を `materialize_fallible` で再取得できる」という既存契約を、`cell` を経由した間接参照（`cell` の入力 `NodeId` 群）へ 1 段拡張する形であり、新たな契約違反は生じない。

**#1647 への追加の引き継ぎ**: 上記の `nodes[cell.0].op` 参照は決定 11 (b) の数値微分突合が通れば正しさが構造的に担保されるが、参照ロジック自体が実際に機能していることを検証するため、`cell` 経由の `NodeId` 参照を意図的に無効化・破損させると数値微分突合が失敗することを確認する構造テスト（決定 11 (h) の GRU `q_t` 検証と同型）を受入基準へ追加する（決定 11 に項目 (j) として追記）。

### 決定 1c: GRU 専用ペイロードへの再帰側アフィン値 `q` の追加（GEMM 再計算不要契約の維持）

**問題（codex-review 指摘。PR #1662）**: 決定 5 の GRU 式 `n_t = tanh(W_in x + b_in + r_t ⊙ (h_{t-1}·W_hn + b_hn))` において、`q_t := h_{t-1}·W_hn + b_hn`（再帰側アフィン値）は `pre_n_t := W_in x + b_in + r_t ⊙ q_t` の `r_t` に対する偏微分そのもの（`∂pre_n_t/∂r_t = q_t`）である。決定 1 の payload 保存方式は「ゲート値を保存して GEMM 再計算を避ける」ことを前提とするが、`r_t`・`z_t`・`n_t` の 3 値のみでは `q_t` を復元できない（`n_t` から逆算するには `tanh⁻¹` と `r_t` による除算が必要で、`r_t` が 0 に近い場合に非可逆・数値不安定になり、そもそも tanh 逆関数を経由する復元は本設計が避けたい追加計算そのもの）。`q_t` を保持しない場合、`∂n_t/∂r_t` の算出には backward 内で `h_{t-1}·W_hn + b_hn` の GEMM を再計算する必要が生じ、**決定 1 の「payload 保存方式は GEMM 再計算が不要」という前提が GRU に限り崩れる**。

**採用案**: `Op::GruCell` の payload に `r_t`／`z_t`／`n_t` に加えて `q_t`（`[B, H]`。非追跡 `Tensor<f32>`）を保持する（例: `Op::GruCell { x, h_prev, w_ih, w_hh, b_ih, b_hh, gates_rzn: Tensor<f32>, q: Tensor<f32> }`）。VJP は `q_t` を直接読むだけで済み、GEMM 再計算を伴わない。これにより GRU の実効ゲート数は `G=4`（LSTM と同数。決定 1 のメモリコスト表を更新済み）となる。

RNN／LSTM は本問題を生じない: RNN は `tanh` の引数がそのまま単一の GEMM 出力（他ゲートとの要素積を挟まない）。LSTM の `i／f／g／o` はいずれも独立な GEMM 出力へそのまま活性化関数を適用するのみで、ゲート同士の要素積（GRU の `r_t ⊙ q_t` に相当する構造）を経由しないため、ゲート値自体が偏微分の再構成に十分である。

### 決定 2: 時系列ループの tape 構築（展開 vs 単一 Sequence Op）

**候補 A（展開・unrolled）**: セル Op を T 回、動的テープ上へ展開する。重み `NodeId` を T 回参照し、BPTT は `backward.rs::accumulate` の fan-in 蓄積で自動成立する（追加機構不要）。

**候補 B（単一 Sequence Op）**: `Op::RnnSequence` 1 ノードが内部で forward 全 step を実行し、VJP 内で逆順ループする。テープ長は O(1) になるが、VJP が巨大化し「1 ノード＝1 VJP 寄与」という既存モデルに対する特殊化が必要（融合境界・`materialize_fallible` 層とも整合を取り直す必要がある）。

**採用案**: **候補 A（unrolled）を v1 とする**。テープ成長 O(T)・保存活性化（ゲート payload 込み）メモリ O(T·B·G·H) を明記する。以下は v1 スコープ外とし §5 に記録する: truncated BPTT、`pack_padded_sequence` 相当の可変長系列、双方向（bidirectional）、**Sequence レベル API 自体**のスタック（**codex-review 指摘〈PR #1662〉を受け決定 4a で切り分け**: セル単位の per-step 交互適用〈決定 4a (i)〉は勾配連続のまま v1 で成立するが、Sequence レベル API 同士のスタック〈決定 4a (ii)〉は候補 A の `Tensor` レベル入力スライスにより前段層への逆伝播が切れるため v1 スコープ外。詳細は決定 4a）。

### 決定 3: `Tape::reset` との相互作用（reuse 学習ループ）

- 契約: RNN 重み（`W_ih`・`W_hh`・`b_ih`・`b_hh`）は**最初の演算より前**に `Tape::var` で登録しないと葉プレフィックスに残らない（§3 契約 4）。`x_t`・`h0` の per-step 葉は reset で破棄される（意図どおりの挙動）
- nn モジュールの `bind` 順序契約は `Linear::bind` と同じ「per-step 葉登録」方式を踏襲する（reset のたびに毎 step 再登録になる）
- `Gradients` は世代（`epoch`）検査を跨いだ持ち越しができない（`backward.rs:75-83`）。stateful RNN（step 間で `h_t` を保持する用途）は `h_t` を `Var::value()`／`to_tensor()` で `Tensor` へ detach してから次 step の葉として再登録する必要がある

### 決定 4: 入力のスライスと出力の積み上げ（依存ギャップ）

- 事実: `matmul` は 2 次元限定、`Var::narrow` 未実装（#1599。#1619 の宣言依存**外**）、`stack`（#1598。宣言依存）、`sub`／`neg`（#1593。宣言依存）
- **候補 A**: セル API は `x_t: &Var`（`[B, D_in]`）を受け取り、時系列スライスは呼び出し側責務（`[T,B,D]` の `Tensor` を `Tensor::narrow`〈`tensor-core`。既存〉でスライスして `Tape::var` 登録＝各 step 入力は葉）
- **候補 B**: Sequence API が `[T,B,D]` の `Var` を受け、`Var::narrow`（#1599）で内部スライス
- **採用案**: **v1 は候補 A**（セル単位 API＋Module 側で `Tensor` レベルのスライス。`Var::narrow` 未実装でも成立する）。候補 B は #1599 完了後の拡張として §6 に記録する。出力 `[T,B,H]` の組み立ては `stack`（#1598）依存。GRU の `1 − z` は `Op::GruCell` 内部で閉じる（専用 Op のため `sub` 不要。候補 A 合成参照実装では `sub`〈#1593〉に依存する）

### 決定 4a: 多層スタック時の勾配連続性（`Module` trait との整合。codex-review 指摘・PR #1662）

**問題**: 決定 2 が当初記していた「多層は Module 側で単層セルを逐次適用するだけで自然に表現できる」は曖昧であり、2 通りの読み方で結論が異なる。

- **(i) セル単位の per-step 交互適用**（例: 各 step で `h1_t = cell1(x_t, h1_{t-1})` → `h2_t = cell2(&h1_t, h2_{t-1})` と 2 層を交互に進める）: 決定 4 の事実のとおりセル API は `x_t: &Var` を受け取るため、`h1_t` は detach されない tape 上の `Var` のまま `cell2` へ渡る。`Var::narrow` は不要で、**勾配は層をまたいで連続する（v1 で成立する）**。
- **(ii) Sequence レベル API のスタック**（層 1 の全 step 分の隠れ状態 `[T,B,H]` を一度組み立ててから層 2 の Sequence 入力とする構成）: `Module` trait（§2 事実）は `forward<'t>(&self, tape: &'t Tape, input: &Var<'t>) -> Result<Var<'t>, AutodiffError>` を要求するが、決定 4 採用案（候補 A）は Sequence レベルで「時系列方向のスライスは呼び出し側責務として `[T,B,D]` の生 `Tensor` を `Tensor::narrow` でスライスしてから `Tape::var`（葉）として登録する」設計であり、`[T,B,H]` を一度 `Var` として組み立てる（`stack`〈#1598〉を要する）にせよ、Sequence レベル API 自体が標準 `Module` trait を実装し `input: &Var` を受け取ってしまうと、その `Var` を `.value()`／`to_tensor()` で `Tensor` へ detach してからスライス・葉再登録することになり、**層 1 へ向かう逆伝播経路をサイレントに断ち切る**（§3 契約 2「独立ノードとして登録されなかった中間値は追跡されない」の系）。

決定 2 の記述は (i) を指すなら正確、(ii)（Sequence レベルのスタック）を指すなら不正確であるため、以下のとおり切り分けて訂正する。

**採用案**:

1. **(i) セル単位の交互適用は v1 で成立する**（勾配連続。追加設計不要）。決定 2 の記述はこの意味で維持する。
2. **(ii) Sequence レベル API 自体のスタック**（層 1 の Sequence 出力をそのまま層 2 の Sequence 入力とする構成）は v1 スコープ外とする。`Var::narrow`（#1599。決定 4 候補 B）により各 step のスライスを勾配追跡可能な経路（`push_view` 相当）で行える設計に置き換わるまで実装しない。
3. Sequence レベル API（`Rnn`／`Lstm`／`Gru`）は標準 `Module` trait を**実装する**（決定 9 の `forward_host` はデフォルト実装が trait メソッドであり `impl Module` を前提とするため、trait を実装しないという選択肢は決定 9 と矛盾する）。ただし `forward(tape, input: &Var)` は `input` を `Tensor` へ detach する経路を**取らない**: 候補 A（`Tensor` レベルスライス）を要求する Sequence 入力用には、`Var` を受ける `forward` ではなく `x: &Tensor<f32>` を直接受け取る専用メソッド `forward_seq(&self, tape: &Tape, x: &Tensor<f32>, h0: Option<&Var>) -> Result<Var, AutodiffError>` を追加する。`forward(tape, input: &Var)` 自体は `AutodiffError`（型付きエラー。決定 11 (g) の fail-closed 方針と整合）を返し、`forward_seq` の利用を促す（(ii) のサイレントな detach を防ぐ）。`forward_host`（決定 9）は既存の tape 不要・ホスト常駐 `Tensor` 経路のまま変更しない。
4. §5 のスコープ外一覧を本決定に合わせて修正する（対象は (ii) Sequence レベルのスタックのみ。(i) セル単位の交互適用は対象外＝v1 で可能）。

### 決定 5: ゲート配置・式・重み形状（parity の参照定義）

PyTorch 準拠で固定する:

- ゲート順: LSTM `i, f, g, o`。GRU `r, z, n`
- GRU の n ゲート式: `n = tanh(W_in x + b_in + r ⊙ (W_hn h + b_hn))`（再帰 bias を r 乗算の**内側**に置く配置）。**PyTorch と tf.keras `GRU`（既定 `reset_after=True`）はこの式で一致**し、Cho 原式（Keras `reset_after=False`）`n = tanh(W_in x + (r ⊙ h) W_hn + b)` とは異なる。#1647 は使用する公式ドキュメントの式で再確認すること
- 重み形状: LSTM `W_ih: [D, 4H]`／`W_hh: [H, 4H]`・`b_ih`／`b_hh` 各 `[4H]`。GRU は `[D, 3H]`／`[H, 3H]`・`[3H]`。RNN は `[D, H]`／`[H, H]`・`[H]`（本リポの `Linear` は `x·W` 規約〈`[in, out]`〉であり、PyTorch の `[4H, D]`〈`W·x` 規約〉を転置した規約であることに注意）
- ゲート GEMM: **(i) 融合 GEMM 1 本 `[B,D]×[D,4H]` ＋ 列ブロック分割**を専用 Op（決定 1 候補 B）で採用する（専用 Op 内部で列ブロックを直接読むため `narrow`／`split` の VJP を必要としない）。候補 A（合成参照実装）は **(ii) ゲートごとに 4 本の GEMM** で表現する
- 初期化・パラメータ順序（`trainable_parameters` の並び）は将来の state_dict 互換（#1616）を見据え PyTorch の並び（`weight_ih, weight_hh, bias_ih, bias_hh`）で固定する

### 決定 6: 初期状態 `h0`／`c0` の意味論

- 省略時はゼロ（`Tensor::zeros`）を Module 側で葉登録する。明示指定時は `&Var` を受け取る
- `h0`／`c0` の勾配は `Gradients::get` の既存契約（到達時 `Ok(Some)`・未到達 `Ok(None)`）に従う。特別扱いはしない

### 決定 7: reuse（デバイス常駐）経路のスコープ

- 事実: `fill_resident_weight_grad` は staging slot への 1 回書き込み（上書き）が前提。RNN の重みは T step すべてで共有されるため VJP が T 回呼ばれ、累積（β=1 相当の加算）が必要になるが、その契約は現状存在しない（§3 契約 3・`docs/device-resident-update-design.md` §3.3b）
- **採用案**: **v1 は fresh／ホスト経路（`Tape::var` の通常葉＋素の `Tape::backward`）限定**とする。reuse（デバイス常駐）化は「staging への累積契約の新設」を要件とする別イシュー起票案として §6 に切り出す（`docs/device-resident-update-design.md` への追補が必要）

### 決定 8: 数値契約

- 3 バックエンド parity は REQ-2 複合判定（相対誤差 1e-3 未満 または 絶対誤差 1e-5 未満）。ホスト参照は `f32::mul_add`（FMA 契約）
- ゲート内の縮約は GEMM＋elementwise のみであり、正規化統計の長軸縮約に適用される f64 アキュムレータ規則（`.claude/rules/coding-rust.md`）は該当しない
- tolerance／baseline の変更なし。run-to-run で決定的（bit 同一）であること

### 決定 9: 推論経路

- `nn::{Rnn, Lstm, Gru}` は `Module::forward_host`（tape 不要）を実装し `predict` 経路に乗せる。決定 4a 項目 3 のとおり `Module` trait 自体は実装するため、`forward_host` のデフォルト実装（trait メソッド）をオーバーライドすることに矛盾はない
- `predict_resident`／`linear_forward_device` との連携は v1 対象外（決定 7 のスコープと整合）

### 決定 10: 公開面（承認事項として分離）

- 内部クレート `fandhe_ai_autodiff::nn::{RnnCell, LstmCell, GruCell, Rnn, Lstm, Gru}`（本文書の設計対象）と、facade 公開面（`fandhe_ai` 再エクスポート・`compat::Sequential::add_lstm` 等）は分ける
- facade 公開面拡張は `docs/compat-api-scope.md` §5 の手続き（正本 spec 側の REQ-9 改定〈完了済み: Tier 2 明記〉＋実装リポ側 `docs/compat-api-scope.md` §1／§2／§5 の更新〈#1591。**OPEN**〉＋ユーザー承認）が完了するまで**実装不可**とする
- `compat::Sequential` への統合（2 次元入力前提の平坦鎖からの拡張）は #1618（`add_*` 拡張）の管轄で扱うことを推奨する
- 実装は `crates/facade/tests/api_surface.rs` の機械検査（`BackendOps`／生 `Tape` を再エクスポート・直接引数化しない）を通過する構成であることを条件とする

### 決定 11: #1647 への引き渡し（受入基準・テスト一覧）

受入基準案:

- (a) 専用 Op（決定 1 候補 B）と合成参照実装（決定 1 候補 A）の複合判定一致
- (b) 多入力への数値微分突合（`grad.rs` の既存数値微分パターンをセル演算の全入力〈x, h_prev,（LSTM は c_prev,）W_ih, W_hh, b_ih, b_hh〉へ拡張）
- (c) T=1／T>1 の BPTT（重み勾配が T 個の寄与和になっていること。LSTM は `c_t` 経路〈決定 1b〉を含めて数値微分と一致すること）
- (d) 3 バックエンド parity（CUDA／Metal は `#[ignore]` 分離）
- (e) `Tape::reset` 後の葉保持（決定 3 の per-step 再登録契約）
- (f) `cargo fmt --all -- --check`・`cargo clippy --workspace --all-targets --all-features -- -D warnings`
- (g) 未知 `Op` variant への fail-closed（型付きエラー。本番経路で `unwrap`／`expect` を使わない）
- (h) GRU の `r_t` に対する勾配が payload 保存済み `q_t`（決定 1c）から算出され、backward 内で `h_{t-1}·W_hn+b_hn` の GEMM 再計算を伴わないこと（数値微分突合〈項目 (b)〉が通れば正しさは担保されるため、本項目は「`q_t` を意図的に欠落・破損させると (b) の数値微分突合が失敗する」ことを確認するテストとして実装し、payload 保存方式が実際に機能していることを構造的に検証する）
- (i) Sequence レベル API（`Rnn`／`Lstm`／`Gru`）が `Module` trait を実装しつつ `forward(tape, input: &Var)` は型付きエラーを返し、`x: &Tensor<f32>` を受ける専用メソッド `forward_seq` が学習経路であること（決定 4a 項目 3）の型検査。セル単位の per-step 交互適用（決定 4a (i)）は勾配連続であることを数値微分突合で確認する。Sequence レベル API 同士のスタック（決定 4a (ii)）は v1 スコープ外のため受入基準に含めない
- (j) `LstmHidden` の VJP が `cell` ノード（`Op::LstmCell`）参照経由で `x`／`h_prev`／`w_ih`／`w_hh`／`b_ih`／`b_hh` の `NodeId` を取得し、o ゲート勾配を正しく `accumulate` すること（決定 1b 追記）。項目 (b)（数値微分突合）が通れば正しさは担保されるため、本項目は「`cell` ノード参照を意図的に無効化・破損させると (b) の数値微分突合が失敗する」ことを確認する構造テストとして実装する（項目 (h) と同型）

付随更新: `docs/public-api-design.md` §3.2 への追記、`Op` doc の「`Var` とほぼ 1:1」注記の更新、`docs/compat-feature-gap.md` §2.7 行の更新、`docs/kernel-fusion.md`（融合境界。専用セル Op は非 elementwise につき融合対象外である旨）への注記。

### 決定 12: 式の定義（候補 A 合成参照実装・全文）

以下は候補 A（合成、決定 1 の対照実装）の解析形。`σ` はシグモイド、`⊙` は要素積。

**RNN（tanh 版。PyTorch `nn.RNN` 既定）**:

```
h_t = tanh(x_t · W_ih + b_ih + h_{t-1} · W_hh + b_hh)
```

**LSTM**:

```
i_t = σ(x_t · W_ii + b_ii + h_{t-1} · W_hi + b_hi)
f_t = σ(x_t · W_if + b_if + h_{t-1} · W_hf + b_hf)
g_t = tanh(x_t · W_ig + b_ig + h_{t-1} · W_hg + b_hg)
o_t = σ(x_t · W_io + b_io + h_{t-1} · W_ho + b_ho)
c_t = f_t ⊙ c_{t-1} + i_t ⊙ g_t
h_t = o_t ⊙ tanh(c_t)
```

**GRU（`reset_after=True` 規約。決定 5 参照）**:

```
r_t = σ(x_t · W_ir + b_ir + h_{t-1} · W_hr + b_hr)
z_t = σ(x_t · W_iz + b_iz + h_{t-1} · W_hz + b_hz)
n_t = tanh(x_t · W_in + b_in + r_t ⊙ (h_{t-1} · W_hn + b_hn))
h_t = (1 − z_t) ⊙ n_t + z_t ⊙ h_{t-1}
```

候補 A ではこれらを `Var::matmul`／`add`／`mul`／`sigmoid`／`tanh` の合成で組み立てる（GRU の `1 − z_t` は `Var::sub`〈#1593〉が必要。専用 Op〈候補 B〉は内部で `1 − z_t` を閉じるため依存しない）。

## 5. スコープ外（v1 対象外）

- truncated BPTT
- `pack_padded_sequence` 相当の可変長系列サポート
- 双方向（bidirectional）RNN／LSTM／GRU
- Sequence レベル API 自体のスタック（決定 4a (ii): 候補 A の `Tensor` レベル入力スライスでは前段層への逆伝播が切れるため `Var::narrow`〈#1599〉による候補 B 化が前提。セル単位の per-step 交互適用〈決定 4a (i)〉は勾配連続のまま v1 で成立するためスコープ外ではない）
- reuse（デバイス常駐）経路（決定 7）
- `predict_resident`／`linear_forward_device` 連携（決定 9）
- facade 公開面（`fandhe_ai`／`compat`）への統合（決定 10。§6 の承認事項）
- ゲート値の再計算方式（決定 1 で不採用とした代替。将来の activation checkpointing〈#1624〉との統合時に再検討）
- `Var::narrow` を用いた候補 B（決定 4）でのテープ内スライス

## 6. 承認事項（#1647 着手前の前提）

1. **facade 公開面の拡張は実装不可**（決定 10）: `docs/compat-api-scope.md` §5 の範囲拡張手続きのうち、実装リポ側の §1／§2／§5 更新（#1591）が完了し、かつユーザー承認記録が残るまで、`fandhe_ai` 再エクスポートおよび `compat::Sequential::add_lstm` 等は実装しない。正本 spec 側（Fandhe-AI/fandhe-ai-spec#66）は既にマージ済みだが、これは §5 の手続き (1) のみを満たすものであり、実装リポ側の記録更新は別途必要
2. **依存追加の起票案（起票自体は本文書の管轄外）**: #1619 の宣言依存に `#1599`（`Var::narrow`。決定 1b・決定 4 候補 B・決定 4a (ii)〈Sequence レベル API 自体のスタックの前提〉）を追加することを提案する。本エージェントは自動運転・承認不可の制約により Issue 起票・コメント投稿を行わない。ユーザー承認後に `out-of-scope-tracking.md` の手続きで起票することを推奨する
3. **reuse 経路の累積契約の新設提案（決定 7）**: RNN 重み共有時の staging 累積（β=1 相当）契約を `docs/device-resident-update-design.md` へ追補する別イシューを提案する。本エージェントは起票を行わない
4. 上記 1〜3 のいずれも、#1647（3 バックエンド実装）が着手する前に解消しておくべき前提として記録するに留め、本イシュー自体はここまでで完了とする

## 7. 出典

- `docs/compat-feature-gap.md` §2.7・§4
- `docs/spec/04-requirements.md` REQ-9（2026-09-12 追記・Tier 2）
- `docs/compat-api-scope.md` §5
- `docs/autodiff-nograd-leaf-dinput-skip-decision.md`（設計文書フォーマットの先例）
- `docs/device-resident-update-design.md` §3.3b
- `crates/autodiff/src/tape.rs`・`crates/autodiff/src/grad.rs`・`crates/autodiff/src/backward.rs`・`crates/autodiff/src/var.rs`・`crates/autodiff/src/nn/module.rs`
- `crates/tensor-core/src/ops_shape.rs`・`crates/tensor-core/src/tensor.rs`・`crates/tensor-core/src/backend_ops.rs`
- `crates/facade/src/compat/sequential.rs`・`crates/facade/tests/api_surface.rs`
- `.claude/rules/coding-rust.md`
