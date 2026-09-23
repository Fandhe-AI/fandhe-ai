# 非学習葉への d_input 伝播スキップの設計判断（イシュー #1219）

- 対応イシュー: #1219（親 #1151 → #1135）
- 位置づけ: 本文書は**設計判断のみ**を記録する。実装は行わない（§9 の起票草案を実装イシューへ引き継ぐ。起票自体はユーザー承認後）
- 基準コミット: `origin/main` `eb421f2`（#1230 まで反映済み。ローカル `main` は #1219 着手時点で stale だったため、行番号・実測値はすべて本コミット時点で再確認した）

## 1. 背景

- 親 #1135（Phase 5 横並び再計測サイクルの残存未達）→ #1151（支配項確定・起票案 I。`docs/perf/train-step-phase-breakdown.md` §15.3・§15.6）
- 学習 1 step の backward で、`Op::LinearAct`（fresh 経路）／`Op::LinearResident`（reuse 経路）／`Op::MatMul` の VJP は入力側勾配 `d_input = g·Wᵀ` を無条件に計算する。層 1 の入力 `x`（`Tape::var` で登録した葉）は学習対象ではなく、その勾配は学習ループ（`SequentialVars::trainable_grads`・`DeviceParamStore::step`・framework-compare の `bench-fandhe` train ループ）から一切読まれない。reuse 経路ではこの d_input 計算がデバイス GEMM（`gemm_resident_lhs`）+ 転置 2 回 + D2H（GPU では同期点）を伴う
- 一方 `Gradients::get` は「同一テープの任意の `Var` の勾配を `Ok(Some)`／未到達なら `Ok(None)`」を返す公開契約であり、入力葉の勾配を読む利用者・テスト・example が多数存在する（§3.1）。多層構成では層 `i` の d_input が層 `i-1` の upstream（伝播の中継点）になるため、「葉だから」という理由だけで無条件にスキップすることはできない
- 目的: 上記 2 つの契約（`Gradients::get` の利用者との互換性・多層伝播の正しさ）と両立する「非学習葉への d_input 伝播スキップ」の設計を確定し、実装イシューへ引き渡せる受入基準・テスト一覧まで文書化する

## 2. 現状のコード事実

すべて `origin/main`（`eb421f2`）時点。

| 事実 | 出典 |
|---|---|
| 勾配追跡の ON/OFF は型分離（`Tensor<f32>` = 非追跡・`Var` = 追跡）で表現する。`requires_grad` 相当のフラグは `Op`／`TapeNode`／`Var` のいずれにも存在しない。`no_grad` 相当は「`Tensor` のまま演算する」ことで表現する既存設計 | `docs/public-api-design.md:352`・`crates/autodiff/src/tape.rs:579`（`Tape::var` が `push_eager(Op::Leaf, ..)` する） |
| `Tape::backward_impl` は逆走査で `grads[id] == None`（loss から未到達）のノードのみ VJP を飛ばす。これは**出力側**（loss からの到達性）の判定であり、**入力側**（葉が学習対象かどうか）の判定はない | `crates/autodiff/src/backward.rs:180`（`let Some(upstream) = grads[id].clone() else { continue; };`） |
| `Op` enum は `#[derive(Debug, Clone)]`（`#[non_exhaustive]` ではない。ワークスペース内クレートのみが構築するため） | `crates/autodiff/src/tape.rs:86-87` |
| `grad::vjp` は `Op::MatMul`（`crates/autodiff/src/grad.rs:84`）・`Op::LinearAct`（同 `:365`）で `matmul_vjp(ops, a, b, g)`（同 `:505`）により d_input・d_weight を**両方**計算する（`ops.gemm_fp32_strict` 経由） | `crates/autodiff/src/grad.rs:84,365,505` |
| `Op::LinearResident`（同 `:208`）は d_input を `ops.gemm_resident_lhs(w_dev, &g_t)`（`:286`）→ `transpose2d` で計算し、d_weight は `resident.fill_resident_weight_grad(..)`（`:338`）成功時に `Gradients` へは含めない（既知の縮小点。§2 後半の先行事例を参照） | `crates/autodiff/src/grad.rs:208,282-286,319,337-338` |
| `Gradients::get` 契約: 別テープ／別世代（`Tape::reset` 後）→ `Err(TapeMismatch)`、未到達または backward 後に追加されたノード → `Ok(None)`、到達 → `Ok(Some)` | `crates/autodiff/src/backward.rs:75-83` |
| 入力の列挙（「あるノードの入力 `NodeId` を列挙する」処理）は `Op` の `match` を必要とする箇所ごとに個別実装されている。汎用の `Op::inputs() -> impl Iterator<Item = NodeId>` のような共通ヘルパは存在しない（`Tape::effective_subtree_size`・`build_lazy_plan`・`grad::vjp` はいずれも独自に `match op { .. }` している） | `crates/autodiff/src/tape.rs:872-887`（`effective_subtree_size`）・`crates/autodiff/src/tape.rs:943`（`build_lazy_plan`）・`crates/autodiff/src/grad.rs`（`vjp` 本体） |

**先行事例（#1212／#1224）**: 常駐 weight の勾配を `Gradients` に含めない縮小実装を既に採用しており、`get()` が到達済みノードでも `Ok(None)` を返しうる点を「既知の縮小点。内部一貫性としては型付きエラー化が望ましい」と記録済み（`docs/device-resident-update-design.md` 追補 #1212 §2 末尾・`docs/perf/train-resident-grad-device-update.md` §4）。本設計判断はこの前例と整合させる（§5・§10）。

### 2.1 `Gradients::get` で入力葉の勾配を読む既定経路の依拠者

| 依拠箇所 | 用途 |
|---|---|
| `crates/facade/examples/backend_switching.rs` | `grads.get(&input)?.ok_or(..)` で入力勾配を取得する example |
| `crates/self-repair/tests/revalidation_bug_fix.rs`・`crates/self-repair/tests/fixtures/feature-addition-leaky-relu/**/bench_workload.rs` | `grad.get(&x)` で入力勾配を検証 |
| `crates/autodiff/tests/backward.rs`・`crates/autodiff/tests/nn_linear.rs`（`backward_grad_x_matches_numeric`）・`crates/autodiff/tests/nn_cross_entropy.rs`（「input に勾配が到達する」）・`crates/autodiff/tests/tape_reset.rs`・`crates/autodiff/tests/fusion_backend_integration.rs`・`crates/autodiff/tests/fusion_chain_limit.rs`・`crates/autodiff/tests/mse_loss_fusion.rs` | 入力側勾配の数値検証・数値微分との突合 |
| `crates/facade/src/compat/sequential.rs`・`crates/autodiff/src/compat/sequential.rs` のテスト `sequential_forward_on_external_tape_supports_backward` | 外部テープ上での forward/backward 結合検証 |
| `crates/facade/tests/tape_construction.rs`・`crates/facade/tests/compat_sequential.rs` | facade 層からの入力勾配取得検証 |

これらはいずれも `Tape::var` で登録した既定の葉に対して `get()` を呼んでおり、**既定経路の挙動を変えると全て壊れる**。したがって既定経路（`Tape::var`）は無変更のまま維持し、スキップは opt-in（利用者が明示的に選んだ葉に限る）とする（§3.1 の結論）。

### 2.2 学習ループ側は入力葉の勾配を読まない

- `SequentialVars::trainable_grads`（`crates/facade/src/compat/sequential.rs:758-778`）は `vars.weight`／`vars.bias`（学習対象のみ）の勾配だけを `grads.get(..)` で取得し、未到達なら `AutodiffError::InvalidArgument` で fail-closed に拒否する。入力 `x` の勾配は取得しない
- `DeviceParamStore::step`（`crates/autodiff/src/optim/device_store.rs`）・framework-compare の `bench-fandhe`（`scripts/bench/framework-compare/bench-fandhe/src/main.rs`）の train ループも同様に weight/bias のみを読む

### 2.3 多層時の d_input 依存

`Sequential::forward`／`forward_from_flat_leaves`（`crates/facade/src/compat/sequential.rs`）は `current`（前層の出力）を次層の `input` として渡す。層 `i`（`i ≥ 2`）の `input` は**非葉ノード**（前層の出力）であり、その VJP が計算する d_input は前層の VJP が使う upstream になる。したがって、スキップの禁止条件は「葉かどうか」ではなく**「`requires_grad == true` の入力ノードへの伝播をスキップしないこと」**に統一する（§5 の採用案 B は `requires_grad` を非葉ノードにも前方伝播させるため、学習対象の祖先を一つも持たない非葉ノード〈中継点〉への d_input も条件を満たせば省略可能であり、逆に葉であっても `requires_grad == true`（既定）のままなら省略不可。「非葉ノードは絶対にスキップ不可」ではない）。

### 2.4 view ノードを通過する入力

`Op::Reshape`／`Op::Transpose`（イシュー #1047。`crates/autodiff/src/tape.rs` `Op` doc）は葉を包む非葉ノードになりうる（例: `x.reshape(..)` を層の入力に渡す）。単純な「葉判定のみ」では、opt-in 葉が view を経由して層に入力される経路を見逃す。この点は§4 の案比較・§5 の採用案で扱う。

### 2.5 性能前提の再ベースライン

起票元（#1151）が根拠とした「fresh backward の約半分」という見積もりは #1223（`eval::matmul` scalar 実装 → `BackendOps::gemm` 切替）**前**の FLOP 算術に基づく。#1223 後の CPU／M4 Max 実測（`docs/perf/train-backward-gemm-wiring.md` §4）では fresh backward 中央値 1.321 ms・step_total 中央値 1.835 ms である。層 1 の d_input GEMM（`[64,256]×[256,784]` ≈ 25.7 MFLOP）は、backward 内 GEMM FLOP 総量（層 1 d_weight 25.7 MFLOP + d_input 25.7 MFLOP + 層 2 の d_weight／d_input／d_bias 各 ≈0.33 MFLOP ≈ 合計 52 MFLOP）の**約半分**という FLOP 比自体は変わらないが、**backward 壁時間に対する比率は #1223 後は未実測**である（GEMM 呼び出し固定費・rayon 分割・packing の相対比重が変わりうるため、FLOP 比をそのまま壁時間比に読み替えられない）。CUDA は #1223 後の実測値が存在しない（`docs/perf/train-backward-gemm-wiring.md` §6）。Metal は #1223（本切替）**単体**の寄与を切り分けた実測は存在しないが、後続 #1215（Metal NT/TN strided 結線）の train phases フル A/B は #1223 適用済みの `origin/main` を before として計測済みであり、切替後の壁時間自体（backward: fresh 1.649×・reuse 1.200×、step_total: fresh 1.323×・reuse 1.109×。#1215 の増分込み）は既存実測がある（`docs/perf/train-backward-gemm-wiring.md` §6・#1215）。本設計判断はこの前提の曖昧さを解消せず、**実装イシュー側で before/after の壁時間実測を必須**とする（§8・§9）。

## 3. 契約整理

### 3.1 `Gradients::get` の葉勾配契約と結論

`Gradients::get`（`crates/autodiff/src/backward.rs:75`）は `Err(TapeMismatch)`／`Ok(None)`／`Ok(Some)` の 3 値契約であり、§2.1 の依拠者はいずれも既定の葉（`Tape::var`）に対してこの契約に依拠している。

**結論**: 既定の `Tape::var` で登録した葉の挙動（d_input を計算し `Gradients` に含める）は変更しない。d_input 伝播のスキップは、利用者が明示的に選んだ葉に限る **opt-in** とする。これにより §2.1 の依拠者・公開 API 非破壊のガードレール条件（`.claude/rules/security.md`）の双方を満たす。

### 3.2 多層時の d_input 依存と拾い漏れ

- スキップの禁止条件は「`requires_grad == true` の入力ノードへの伝播をスキップしないこと」に統一する（§2.3）。前層 VJP の upstream になる非葉ノードは、学習対象の祖先を持つ限り `requires_grad == true` のままなのでスキップ不可だが、学習対象の祖先を一つも持たない非葉ノード（例: opt-in 葉のみから構成された中継点）は省略可能である
- 葉であっても、既定（`Tape::var`）のままなら `requires_grad == true` を維持し他の消費者（`Gradients::get` を直接呼ぶ利用者）が読む可能性があるため、**利用者が明示的に opt-in して `requires_grad == false` にした葉に限りスキップ可**とする
- 葉を包む view／elementwise ノード（例: 入力正規化を挟んでから層に渡す構成）は、葉判定だけでは opt-in の効果を追跡できない（§2.4）。採用案（§5）はこれを「`requires_grad` の前方伝播」で解決する

## 4. 設計案の比較

| 案 | 概要 | 正しさ | 既存テストへの影響 | 変更範囲 | view 対応 | reuse 経路（デバイス GEMM＋D2H 同期点）の除去 | 公開 API 非破壊 |
|---|---|---|---|---|---|---|---|
| A: 葉フラグのみ | `Tape::var_no_grad` 等で登録した葉に flag を持たせ、GEMM 系 VJP が `nodes[input.0]` の flag を直接見て d_input を省略する | 葉に直接連なる GEMM のみ正しい | 影響なし（既定不変） | 小 | ✕（view／elementwise 越しの入力を拾えない） | 部分的（葉が直接 GEMM の入力である場合のみ） | 満たす |
| B（推奨）: `requires_grad` の前方伝播 | PyTorch 型。`TapeNode` に `requires_grad: bool` を追加。葉は既定 `true`・opt-in 葉は `false`。非葉ノードは「入力のいずれかが `true`」で `push_eager`／`push_lazy`／`push_view` 時に確定する。逆走査は非葉ノードの `requires_grad == false` を意味的に保証しつつ、GEMM 系 VJP（`MatMul`／`LinearAct`／`LinearResident`）は d_input 計算前に入力ノードの flag を見て GEMM 自体を省略する | 全経路（view／elementwise 越し含む）で正しい | 影響なし（既定不変） | 中 | ○ | ○（`Op::LinearResident` の `gemm_resident_lhs` 呼び出し自体を省略できる） | 満たす |
| C: 見送り | 本イシューでは設計のみに留め実装しない | — | — | なし | — | — | — |

**比較軸の判断**: 案 A は view／elementwise を挟む入力（§2.4）を拾えず、正しさの保証範囲が狭い。案 B は `requires_grad` を前方伝播させることで view／elementwise を含む任意の経路を正しく扱え、reuse 経路のデバイス GEMM＋転置＋D2H（GPU 同期点）も条件が成立すれば呼び出し自体を省略できる。変更範囲は `TapeNode` へのフィールド追加とノード生成時・VJP 時の分岐追加に留まる。**案 B を採用する**。

**実装イシューへの調査指示**: `tape.rs` には「ノードの入力 `NodeId` を汎用的に列挙するヘルパ」（例: `Op::inputs() -> impl Iterator<Item = NodeId>`）は存在しない（§2 表）。`effective_subtree_size`（`tape.rs:872`）・`build_lazy_plan`（`tape.rs:943`）はいずれも `Op` を個別に `match` している。`requires_grad` の前方伝播も同様に、ノード生成箇所（`push_eager`／`push_lazy`／`push_view`）ごとに `Op` を `match` して入力ノードの `requires_grad` を集約する実装になる見込みであり、既存の汎用列挙ヘルパを流用する余地はない（新設する場合は `effective_subtree_size` 等との重複整理を実装イシュー側で検討する）。

## 5. 採用案の詳細（案 B）

### 5.1 フラグの置き場所

`TapeNode`（`tape.rs:460`）へ `requires_grad: bool` フィールドを追加する。`Op::Leaf` を `Op::Leaf { requires_grad: bool }` のような struct variant に変える案は、`matches!(node.op, Op::Leaf)` を用いる箇所（`Tape::leaf()` 等）・融合プランナの `match` パターンを全て変更する必要があるため不採用とする。`TapeNode` フィールドとして持たせることで、`Op` 自体の形は変えずに済む。

### 5.2 登録 API

- `autodiff::Tape` に内部メソッド `Tape::var_no_grad(&Tensor<f32>) -> Var<'_>` を追加する（名称は実装イシューで `var_no_grad` と `input`（勾配追跡なし入力、の意）のいずれかに確定する）
- facade newtype `fandhe_ai::Tape`（`crates/facade/src/lib.rs:146` の `var` の隣）に同名メソッドを追加する
- `crates/facade/tests/api_surface.rs` は `BackendOps` の非露出を検査するテストであり、`Tape` への新規メソッド追加は当該テストの検査対象外（影響なし）

### 5.3 `Gradients::get` の `requires_grad == false` ノードに対する挙動

`Ok(None)`（未到達と区別不能）ではなく、**型付きエラー**とする（#1212 の「既知の縮小点。内部一貫性としては型付きエラー化が望ましい」との整合。§2 先行事例）。実装イシューでの選択肢を優先順位付きで示す:

1. （推奨）`AutodiffError`（`#[non_exhaustive]`。`crates/autodiff/src/error.rs:19`）へ新 variant（例: `GradientTrackingDisabled`）を追加する。「未到達」（`Ok(None)`）と「そもそも追跡していない」（`Err`）を型で区別でき、fail-closed の意図が明確になる
2. 既存の `AutodiffError::InvalidArgument(String)` を流用する。variant 追加を避けられるが、他の `InvalidArgument` 用途（引数検証エラー全般）と意味が混ざる

型付きエラーを推奨する理由: `Ok(None)` のまま返すと「loss へ未到達」（正常な設計）と「勾配追跡していない葉」（利用者の opt-in による意図的な縮小）を呼び出し側が区別できず、誤って前者と解釈すると学習ループのバグ検知が遅れる（fail-closed の欠如）。

### 5.4 学習対象の祖先を持たない loss の `backward()`

`Tape::backward` の起点となる `loss` 自身が `requires_grad == false`（＝学習対象への経路が一つもない）場合は `Err(AutodiffError::Backward(..))` で fail-closed に拒否する（PyTorch の `loss.requires_grad == False` での `backward()` 呼び出しエラーと同様。誤用の早期検知）。

### 5.5 `Tape::reset`／`leaf`／`leaf_count` との整合

葉プレフィックス（`Tape::reset()`。#1048）は保持されるため、`requires_grad` フラグも葉プレフィックスとともに保持する。`reset()` 後の `Tape::leaf(index)`（`tape.rs:635`）は同じ `requires_grad` を持つ葉を再取得する。

### 5.6 `Op::ResidentLeaf` との関係

`Op::ResidentLeaf`（イシュー #1022。デバイス常駐パラメータの葉）は本設計の対象外とする。既に独立した縮小実装（§2 先行事例）を持つ別経路であり、本設計判断が扱う「非学習な**入力**（活性化データ x）」とは異なる（`Op::ResidentLeaf` は重みという学習対象そのものであり、`requires_grad` は常に `true` を維持する。省略されるのは d_weight の**計算**ではなく、常駐経路〈`ResidentResolver::fill_resident_weight_grad`〉が成功した場合の**ホスト側 `Gradients` への格納**のみである。d_weight は必ず 2 択のいずれかの経路で計算される〈`crates/autodiff/src/grad.rs:338-345` 付近の `Op::LinearResident` 実装〉: まず `resident.fill_resident_weight_grad(..)` を試み、成功時（`filled_resident == true`。バックエンドが `gemm_fp32_strict_into`／`MemoryOps` を実装する場合のみ）は同メソッド内部で `DeviceParamStore` が `gemm_fp32_strict_into` を使い常駐 grad staging へ直接書き込む（`contributions` へは含めない）。失敗時（`Ok(false)`。現時点で CUDA／Metal はここに該当する）はホスト経路 `ops.gemm_fp32_strict(&x_t, g)` へフォールバックし、通常どおり `contributions` へ含める。いずれの経路でも `filled_resident` が `true` の場合のみ `Gradients::get` からは「未到達」と区別できなくなる〉。これは opt-in ではなく無条件の縮小実装である点も本設計判断の opt-in 方式とは異なる）。

## 6. 数値一致・既存テストとの整合

- d_weight／d_bias の計算式・カーネル・累積順序は変更しない。opt-in の有無で weight／bias 勾配は **bit 同一**であることを設計上の契約とし、実装イシューのテストで固定する
- 既定経路（`Tape::var`）は無変更のため、§2.1 に列挙した既存の入力勾配テスト・example・self-repair fixture は非影響
- tolerance・baseline（`docs/perf/cuda-parity-baseline.md` 等）は一切変更しない（ユーザー承認事項に触れない。`.claude/rules/coding-rust.md`）

### 6.1 実装イシューで追加するテスト一覧

1. opt-in 葉に対する `Gradients::get` が型付きエラーを返す
2. 同一入力で opt-in／既定（非 opt-in）の weight・bias 勾配が bit 一致する（CPU 本番 ops・2 層 MLP 相当の構成）
3. 多層構成で層 `≥2` の d_input が引き続き伝播し、層 1 の weight 勾配が既定経路と一致する
4. opt-in 葉を `reshape`／`transpose`／`relu` 経由で層へ流しても d_input 計算がスキップされる（`requires_grad` の前方伝播の検証）
5. 全ての葉が opt-in（学習対象の祖先が一つもない）状態での `backward()` が fail-closed エラーを返す
6. `Tape::reset()` 後も `requires_grad` フラグが保持される
7. `Op::LinearResident` 経路（`DeviceParamStore`）で opt-in 時に `gemm_resident_lhs` が呼ばれないことを、テスト用 `BackendOps` 実装のカウンタで確認する
8. 既存の全テスト（`cargo test --workspace --all-features`）が pass すること

## 7. 公開 API・spec 整合

- 変更は追加のみ（`Tape::var_no_grad` の新設・`AutodiffError` への新 variant 追加）であり、`BREAKING CHANGE` は不要（公開 API 非破壊のガードレール条件を満たす）
- `docs/public-api-design.md:352` の「`no_grad` 相当は `Tensor` のまま演算することで表現する」という既存記述との関係: 本設計判断が扱うのは「テープ上の forward には参加するが、逆伝播で勾配を受け取らない葉」であり、`Tensor<T>` のまま演算する既存の `no_grad` 相当とは別概念である（テープに乗る点が異なる）。`docs/public-api-design.md` §3.1 への追補は実装イシュー側の作業として明記する
- `docs/spec/04-requirements.md`（正本 submodule。編集しない）に勾配契約の記述はなく、spec 変更提案は不要（REQ-8 の性能下限・REQ-2 の数値一致複合判定にも影響しない）
- `docs/compat-api-scope.md` の再エクスポート方針（`Gradients`／`AutodiffError` を値型として再エクスポートする方針）と整合する（新 variant も同じ再エクスポート経路に自然に乗る）

## 8. 性能見込み（再ベースライン）

- FLOP 算術（§2.5）: 層 1 の d_input GEMM（`[64,256]×[256,784]` ≈ 25.7 MFLOP）は backward 内 GEMM FLOP 総量（≈52 MFLOP）の約半分
- #1223 後の実測（`docs/perf/train-backward-gemm-wiring.md` §4）: CPU／M4 Max で fresh backward 中央値 1.321 ms・step_total 中央値 1.835 ms。**backward 壁時間に対する d_input GEMM の比率は未実測**
- reuse 経路では、上記 FLOP 削減に加えてデバイス GEMM（`gemm_resident_lhs`）+ 転置 2 回 + D2H（GPU では同期点）の除去が加わるため、fresh 経路より相対的な改善余地が大きい可能性がある（未実測）
- CUDA は #1223 後の実測値がない。Metal は #1223 単体の寄与を切り分けた実測はないが、切替後の壁時間自体（#1215 込み）は既存実測がある（`docs/perf/train-backward-gemm-wiring.md` §6・#1215）
- 計測方法: #1142／#1147 の参考系列手順（framework-compare の非コミット一時 path 差し替え）またはリポ内 bench（`bench-harness`）を用い、5 回計測の中央値を採用する（`.claude/rules/coding-rust.md` ベンチ規約）。実装イシューでは before/after の壁時間実測を `docs/perf/` 配下に必須で記録する

## 9. 実装イシューへの引き継ぎ（起票草案）

**起票はユーザー承認後に行う（本 PR では起票しない）。** `.claude/rules/out-of-scope-tracking.md` に従い、以下を起票草案として記録する。

- タイトル案: `perf(autodiff): 非学習葉への d_input 伝播を opt-in でスキップする（requires_grad 前方伝播）`
- 親: #1135・ラベル: `phase:5`
- 対象ファイル: `crates/autodiff/src/{tape.rs,backward.rs,grad.rs,error.rs}`・`crates/facade/src/lib.rs`・`docs/public-api-design.md` §3.1 追補
- テスト一覧: 本文書 §6.1 の 8 項目
- 受入基準:
  - weight／bias 勾配が opt-in の有無によらず bit 一致すること
  - 既存全テストが pass すること
  - before/after の壁時間実測（5 回中央値）を `docs/perf/` に記録すること
  - tolerance・baseline・依存関係を一切変更しないこと（ユーザー承認事項に触れない）
  - 実機ホスト名・個人情報をドキュメントに書かないこと

## 10. スコープ外

- `Op::ResidentLeaf` に対する `Gradients::get` の型付きエラー化（#1212 の既知の縮小点。本設計判断とは別軸のため別件として扱う）
- 入力勾配が必要な用途（敵対的サンプル生成・saliency map 等）は、既定経路（`Tape::var`）を使う限り従来どおり利用可能（opt-in しない限り影響なし）
- bias 勾配のデバイス常駐化（`docs/perf/train-resident-grad-device-update.md` §4。別件）
- `docs/public-api-design.md` §3.1 への no-grad 葉の追補自体（実装イシュー側の作業とする）

## 11. 出典一覧

- `docs/perf/train-step-phase-breakdown.md` §15.3・§15.6（起票案 I・支配項確定）
- `docs/perf/train-backward-gemm-wiring.md` §2・§4・§6（#1223 後の実測・数値一致確認）
- `docs/device-resident-update-design.md` 追補 #1212 §2（先行事例）
- `docs/perf/train-resident-grad-device-update.md` §4（先行事例の実測記録）
- `docs/public-api-design.md` §3.1（型分離方式・`no_grad` 相当の既存設計）
- `docs/compat-api-scope.md`（再エクスポート方針）
- `.claude/rules/coding-rust.md`（tolerance・ベンチ規約）
- `.claude/rules/security.md`（A08・公開 API 非破壊）
- `.claude/rules/out-of-scope-tracking.md`（起票フロー）
- `crates/autodiff/src/{tape.rs,backward.rs,grad.rs,error.rs}`・`crates/facade/src/lib.rs`（コード事実。§2）

## 実装記録（#1748）

本 issue（#1748）で §5 案 B の「登録 API＋契約」部分を実装した。§9（perf 目的の GEMM 系 VJP 内部での d_input 省略）は対象外のまま未実装。

- `TapeNode::requires_grad: bool` を新設し、葉ノードは明示値（`Tape::var` → `true`・`Tape::var_no_grad`／`Var::detach` → `false`・`Op::ResidentLeaf` → 常に `true`）、非葉ノード（`push_eager`／`push_lazy`／`push_view`）は `Op::for_each_input` で列挙した全入力の論理和で確定する（`crates/autodiff/src/tape.rs`）。
- 葉専用の `Tape::push_leaf(value, requires_grad)` を新設し、`Op::Leaf` は `push_eager` を経由しない契約へ変更した（`push_eager` は `debug_assert!(!matches!(op, Op::Leaf))` で機構的に固定）。
- `Var::detach()`（`crates/autodiff/src/var.rs`）: `materialize_fallible` で値を取得し `push_leaf(value, false)` で新しい葉へ登録する。専用 `Op::Detach` variant は追加していない（既存の葉ノード機構の再利用で過不足なく表現できるため。§2.7 対象外整理のとおり）。
- `Tape::backward_impl`（`crates/autodiff/src/backward.rs`）: 起点 `loss` が `requires_grad=false` なら `materialize_fallible` より前に `Err(AutodiffError::Backward)` を返す。逆走査の寄与蓄積は対象ノードの `requires_grad` を検査し `false` なら `accumulate` へ渡さず破棄する（`grads[target]` は `None` のまま。主ループの未到達分岐が checkpoint 帳簿を処理する）。
- `Gradients::get`（同ファイル）: 対象ノードが `requires_grad=false` なら新設 `AutodiffError::GradientTrackingDisabled`（`#[non_exhaustive]` への非破壊 variant 追加）を返す。「loss から未到達」の `Ok(None)` とは型で区別する。
- facade（`crates/facade/src/lib.rs`）: `Tape::var_no_grad` を薄いラッパーとして追加。`Var::detach` は既存 `Var` 再エクスポート経由で新規公開面なし（`docs/compat-api-scope.md` §5 手続き・親 #1612 承認範囲内）。
- 数値契約: 本機構は算術を一切伴わない（テープのメタ情報のみ）ため、CPU 本番 ops（`fandhe_ai::tape()`）と naive 参照実装（`fandhe_ai_autodiff::Tape::new()`）の勾配が bit 完全一致することをテストで確認した（`crates/facade/tests/no_grad_detach_backend_parity.rs`）。CUDA／Metal 実機での facade parity テストは未実測のまま Mac／GB10 セッションへ申し送り（同テストの `#[ignore]` 群）。
- テストは `crates/autodiff/tests/no_grad_detach.rs`（naive ops 経由の契約テスト。detach の定数扱い・葉プレフィックス保持・checkpoint poison 時の `Err`・reset 越境の `requires_grad` 保持等）と上記 facade parity テストの 2 ファイル。
- 対象外（未実装のまま）: `MatMul`／`LinearAct`／`LinearResident` の VJP 内部での d_input GEMM 省略（§9 の perf 起票草案・未起票）・`retain_graph`（複数回 backward の勾配蓄積契約。兄弟 #1749）・PyTorch `torch.no_grad()` コンテキスト相当（既存の型分離方式が担う。§9・`docs/public-api-design.md` §3.1 追補）・~~`requires_grad_()` によるフラグの事後切替~~（#2137 で実装。下記「実装記録（#2137）」参照）・`Op::ResidentLeaf` の `Gradients::get` 型付きエラー化（既知の縮小点。上記どおり変更なし）。

## 実装記録（#2137）

本 issue（#2137）で、上記「対象外」に残っていた「`requires_grad_()` によるフラグの事後切替」を、層別 `requires_grad` 凍結（PyTorch `module.requires_grad_(bool)`／`nn.Module.requires_grad_` 相当）として `fandhe_ai_autodiff::nn::Module` へ実装した。

- **配置は `autodiff` 内部クレートのみ**。`fandhe_ai::nn::Module` 自体がまだ存在しない（#2133 OPEN・`docs/facade-nn-module-exposure-decision.md` §10 承認待ち）ため、facade（`crates/facade/src/nn/mod.rs`）への鏡写しは行わない。`crates/facade/tests/api_surface.rs::nn_mod_declares_only_rnn_submodule` は無変更のまま green。
- `Module` trait（`crates/autodiff/src/nn/module.rs`）へ 3 つの defaulted メソッドを追加した（非破壊拡張。`fandhe-ai-autodiff` は crates.io 公開クレート）:
  - `set_requires_grad(&mut self, bool) -> Result<(), AutodiffError>`: 既定は fail-closed。`named_parameters()` が空なら `Ok(())`、非空なのにオーバーライドしていなければ `AutodiffError::InvalidArgument`（A08。「凍結したつもりで学習が続く」静かな事故を防ぐ。`set_training` の no-op 既定とは意図的に異なる）。
  - `freeze(&mut self) -> Result<(), AutodiffError>`: `set_requires_grad(false)` の別名。
  - `requires_grad(&self) -> bool`: 既定 `true`（`training()` と同じ契約）。
- `Tape::var_with_requires_grad(tensor, requires_grad)`（`pub(crate)`。`crates/autodiff/src/tape.rs`）を新設し、`Tape::var`／`Tape::var_no_grad` はこの共通ヘルパーを固定値で呼ぶ薄いラッパーへ整理した（挙動・bit 一致は不変）。
- パラメータを持つ全層（`Linear`・`Conv1d`／`Conv2d`／`ConvTranspose2d`・`RmsNorm`／`LayerNorm`・`BatchNormCore`〈`BatchNorm1d`／`BatchNorm2d` が委譲〉・`Embedding`・`RnnCell`／`LstmCell`／`GruCell`〈`Rnn`／`Lstm`／`Gru` が `cell` へ委譲〉）に private フィールド `requires_grad: bool`（既定 `true`）を追加し、`bind` が `Tape::var` の代わりに `Tape::var_with_requires_grad(.., self.requires_grad)` で葉を登録するよう変更した。`MultiheadAttention`（4 子 `Linear`）・`TransformerEncoderLayer`（5 子層）は全子への伝播として実装した（子はいずれも private フィールドのため `requires_grad()` の読み出しは代表 1 子の値を返す）。
- `ModuleList`（`crates/autodiff/src/nn/container.rs`）は `Module::load_state_dict`（`module.rs`）と同型の**ベストエフォート・ロールバック**を実装した。`Sequential` は `inner`（`ModuleList`）へそのまま委譲する。
  - **`requires_grad()` の複合層集約**: 子が 1 つでも `true` を返せば `true`（`any`）。子が 1 つも無い空 `ModuleList` は、`Module::requires_grad` 既定実装の「パラメータを持たない層は既定 `true` のまま」契約に合わせ `true` を返す（`any` の空列の自然な既定 `false` ではなく `true` を明示的に採用）。
  - **ロールバック方式**: 適用前に各子を再帰的にスナップショットし（`RequiresGradSnapshot`。子が入れ子コンテナ〈`ModuleList`／`Sequential`。`Module::as_module_list` が `Some` を返す〉なら孫まで再帰、末端層なら集約 `bool` 1 つ）、途中の子が `Err` を返したら適用済みの子を逆順に同じ構造をたどって復元する。入れ子コンテナが混在状態（一部凍結・一部解凍）のままロールバックされても、孫ごとの個別値が保たれる。
  - **是正記録（PR #2234 レビュー指摘・P1 x2）**: 初回実装（本 issue の当初コミット）は集約を `all`（全子が `true` の場合のみ `true`）としていたが、上記の公開契約「子が 1 つでも `true` なら `true`」と逆であり、一部凍結・一部解凍のコンテナが誤って `false` を返す不具合だった。また初回実装のロールバックは子ごとの**集約 `bool` 1 つ**のみをスナップショットしていたため、入れ子コンテナが混在状態のときにロールバックで孫全員へ単一値を強制し混在状態を破壊していた（fail-closed 契約違反）。`any` 集約への修正と、`Module::as_module_list`／`as_module_list_mut`（`as_linear` 等と同型の downcast フック）を新設した再帰的スナップショット方式への変更で是正した。
- **反映タイミング**: フラグは層側に保持する状態であり、次の `bind`（葉ノードを新規登録する呼び出し）から反映される。登録済みの `Var`（`TapeNode::requires_grad` は葉登録時に確定）・`Tape::reset` 後に保持される葉プレフィックスの葉は登録時のフラグを保持する（#1748 §5.5 の「反映タイミング」契約をそのまま踏襲）。
- **`training`／`state_dict` とは独立の軸**: `freeze()` は `Module::training`／`set_training` を変更しない（`BatchNorm` の running stats は training モードなら引き続き更新される）。フラグは `state_dict`／`load_state_dict` の対象外（テンソル値のみを扱う）。
- **数値契約**: `requires_grad` は `backward.rs::accumulate` 呼び出し前のゲート判定にのみ使われる純粋なメタデータであり、融合プラン選択・forward 演算列には一切影響しない（`tape.rs::push_eager`／`push_leaf` 参照）。したがって同一バックエンド内で「凍結あり」と「凍結なし」の出力・非凍結パラメータの勾配は bit 完全に一致する。この前提は `crates/autodiff/tests/nn_module_freeze.rs`（naive ops 経由の契約テスト・転移学習 example）と `crates/facade/tests/nn_module_freeze_backend_parity.rs`（CPU 本番 ops〈CI で実行〉・CUDA／Metal〈`#[ignore]`〉の同一バックエンド内 parity）で hard assert 済み。CUDA／Metal 実機は未実測のまま Mac／GB10 セッションへ申し送り。
- 対象外（承認事項として列挙。実施しない）: facade 新規公開面（`Module::freeze`／`set_requires_grad`／`requires_grad` の facade 鏡写し。#2133 の実装・`docs/facade-nn-module-exposure-decision.md` 案 B 承認が前提）・`compat::Sequential` 側の凍結 API（既存 fit／SGD の位置対応契約・resident 経路との整合を要する再設計）・per-parameter 粒度の凍結。

### 追加是正記録（PR #2234 レビュー指摘・第 2 ラウンド）

上記「是正記録（PR #2234 レビュー指摘・P1 x2）」の是正後、同 PR への追加レビュー（codex-review P1 x2・cursor[bot] Medium x1）で次の不備が指摘され、いずれも是正した。

- **`ModuleDict` に凍結操作が未実装だった（P1・review thread `PRRT_kwDOTuUCJc6lE8Gg`／cursor[bot] `PRRT_kwDOTuUCJc6lE80z`）**: `ModuleList`／`Sequential` には `set_requires_grad`／`requires_grad` を実装したが、同じ公開 `Module` である `ModuleDict`（イシュー #2134）は既定実装のままのため `ModuleDict::freeze()` が `InvalidArgument` を返していた。`ModuleList` と同じ伝播・集約（パラメータを持つ子の `any`）・ロールバック処理を `ModuleDict` にも実装し、混在状態（一部凍結・一部解凍）の単体テストを追加した。
- **`ModuleList::set_requires_grad` ロールバック範囲の抜け（P1・review thread `PRRT_kwDOTuUCJc6lE8Gn`）**: 失敗時のロールバック範囲が `0..index` で、エラーを返した子自身（`index`）が復元対象から漏れていた。`Module` は外部実装可能でエラー時状態不変の契約も持たないため、複数内部パラメータを順に変更するカスタム複合層では、エラーを返した子自身の内部で部分適用状態が残りうる（fail-closed／全体ロールバック契約違反）。ロールバック範囲を事前スナップショットを使った `0..=index`（逆順）へ修正し、変更後に失敗する子を用いた回帰テスト（`module_list_set_requires_grad_rollback_restores_failed_child_itself`／`module_dict_set_requires_grad_rollback_restores_failed_child_itself`）を追加した。
  - この修正に伴い、`restore_requires_grad`（`nn/container.rs`）の末端（`Leaf`）復元ロジックも調整した: 復元対象の子は「もともと失敗した `set_requires_grad` 呼び出しの当事者」でありうるため、(a) 状態を一切変更せず常に `Err` を返す「半端実装」（`named_parameters` は非空だが `set_requires_grad` 未オーバーライドの既定実装）や、(b) `Err` を返しつつも内部状態への書き換え自体は完了している複合層、のいずれでも復元が正しく完了したと判定できるよう、「呼び出しが `Err` を返したか」ではなく「呼び出し後の観測可能な状態（`requires_grad()`）がスナップショット値と一致するか」でロールバック成否を判定するよう変更した（fail-closed 判定自体は維持: 一致しなければ引き続き `Err`）。
- **ネスト時の破損（cursor[bot] Medium・`PRRT_kwDOTuUCJc6lE80z` 補足指摘）**: `ModuleList::as_module_list`（`restore_requires_grad`／`snapshot_requires_grad` が入れ子コンテナかどうかの判定に使う downcast フック）は `ModuleDict` を末端層として扱ってしまい、`ModuleList` に入れ子の `ModuleDict`（またはその逆）が混在状態を持つ場合にロールバックで混在状態が破壊されうる不備があった。`Module::as_module_dict`／`as_module_dict_mut`（`as_module_list` と同型の明示 downcast フック）を新設し、`RequiresGradSnapshot` へ `NestedDict` variant を追加、`snapshot_requires_grad`／`restore_requires_grad` の双方が `ModuleList` と `ModuleDict` のどちらのネストも再帰的に認識するよう一般化した。双方向（`ModuleList` に入れ子 `ModuleDict`／`ModuleDict` に入れ子 `ModuleList`）の混在状態保持を回帰テストで確認した。
