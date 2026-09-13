# activation checkpointing（任意サブグラフ再計算）の設計・実装記録（イシュー #1624）

- 対応イシュー: #1624（`feat(autodiff): activation checkpointing（任意サブグラフ再計算）を追加する`）
- 位置づけ: 設計判断・実装記録・実測記録を 1 ファイルにまとめる（`docs/autodiff-view-recompute-decision.md` と同じ構成方針）
- 基準コミット: `origin/main`（本 PR のベース）。`docs/autodiff-view-recompute-decision.md`（view 系ノードの再計算方式化。イシュー #1047）が確立した骨格を、任意の eager ノードへ一般化する

## 1. 背景・目的

- `docs/compat-feature-gap.md` §2.11「activation checkpointing」行は「部分（view 系ノードのみ #1080 で再計算方式化済み。任意サブグラフのチェックポイントではない）・難度 L」であり、spec REQ-9 2026-09-12 追記の Tier 2 一覧に列挙済み（`docs/compat-api-scope.md` §1.3）
- 対比対象: PyTorch `torch.utils.checkpoint`・TensorFlow `tf.recompute_grad`。forward で記録した中間値を保持せず、backward で必要になった時点で再計算して勾配を得ることでピークメモリを削減する
- 目的: forward で記録したサブグラフの中間値（`TapeNode::value`）を明示的に解放し、backward が必要とする時点で**同じ演算を同じバックエンドカーネルで再計算**して勾配を得る機構を追加する

## 2. 現状のコード事実（設計の根拠）

| 事実 | 出典 |
|---|---|
| `TapeNode { op, shape, value: OnceCell<Tensor<f32>>, lazy_chain_size }`。登録経路は `push_eager`（値あり）・`push_lazy`（elementwise 5 演算・値なし）・`push_view`（値なし）・`push_resident_leaf` | `crates/autodiff/src/tape.rs` |
| 実体化は層 1 `materialize_fallible`（`Result`）／層 2 `materialize_non_fallible`（infallible）。いずれも冒頭 `value.get()` → view／recompute 分岐 → `build_lazy_plan` → `run_fused` → `fallback_per_op` →（層 2 のみ）`eval_fallback` | `crates/autodiff/src/tape.rs` |
| `resolve_view`（旧）は `ops` を受け取らない infallible 関数で、非 view ノードが未実体化なら `debug_assert!(false)` + ゼロ埋めの契約違反分岐だった | `crates/autodiff/src/tape.rs`（#1624 で `recompute_value`／`recompute_fallible`／`recompute_infallible` へ一般化・置換済み） |
| `backward_impl` は逆走査ループ全体を**単一の不変借用**で完結させていた | `crates/autodiff/src/backward.rs`（#1624 で反復ごと借用へ再構成済み。§3.1 点 4・§5） |
| `Gradients` は全ノード分の `Vec<Option<Tensor<f32>>>` を保持する（中間ノードの勾配も残る） | `crates/autodiff/src/backward.rs` |
| eager 系 `Var::*` の forward は各メソッド内に個別実装（`matmul`／`sigmoid`／`sum`／`max` は `ops` を直接呼ぶだけで `eval::*` フォールバックを持たない。`sum`／`max` は Metal で `Unsupported`） | `crates/autodiff/src/var.rs` |
| `Tape: Send` の静的アサーション | `crates/autodiff/tests/fusion_backend_integration.rs` |
| facade `Tape(pub(crate) fandhe_ai_autodiff::Tape)` newtype。`Var`／`Gradients` は素で再エクスポート。`api_surface.rs` が `Tape`／`BackendOps` の再エクスポート禁止を機械検査 | `crates/facade/src/lib.rs`・`crates/facade/tests/api_surface.rs` |

### 2.1 依存 #1612（no_grad／detach／retain_graph）との関係

- #1612 は本実装着手時点で未完了（OPEN）。本設計は「テープに記録済みの Op から値を再導出する」方式のため **#1612 の API を消費しない**（PyTorch の checkpoint が forward を no_grad で走らせる方式とは異なる）。よって #1612 完了を待たずに実装した
- #1612 が先に main へ入った場合の相互作用（将来の検証テスト追加候補）: 区間内の `detach` は新規葉＝再計算境界になる・`no_grad` でテープに載らなかった演算は区間に含まれ得ない・複数回 `backward()` は再計算値のキャッシュ／再解放で自然に成立する（本実装は既に §6 (5) で検証済み）

## 3. 設計

### 3.1 モデル: 「記録は残し、値だけ捨てる。必要時に再計算し、backward が区間を離れたら再び捨てる」

1. **区間の登録**: チェックポイント区間 = ノード ID の閉区間 `[lo, output]`（`output` を除く）。`Tape` に区間レジストリ `checkpoints: RefCell<Vec<CheckpointRegion { lo, output }>>` を追加
2. **解放（release）**: 区間内で「再計算可能（§3.3）かつ `output` 以外」のノードの `value` を `OnceCell::take()` で空にし、`TapeNode.recompute = true` を立てる
3. **再計算**: `materialize_fallible`／`materialize_non_fallible` の冒頭で `is_view() || recompute` を判定し、`recompute_fallible`／`recompute_infallible`（§3.4）へ分岐する。これらは入力側を再帰的に辿り、必要な値を都度計算する
4. **backward 中の再解放**: `backward_impl` の逆走査で `id == region.lo` を処理し終えた時点で、その区間を再び解放する（`Tape::release_checkpoints_ending_at`）。これにより forward 直後の解放で空けたメモリを、backward の再計算後にもう一度空けられる
5. **複数回 `backward()`**: 2 回目も再計算→再解放で同じ結果（bit 同一）。`Tape::reset()` は区間レジストリを全消去する

### 3.2 公開 API（内部クレート `fandhe_ai_autodiff`）

- `Tape::checkpoint<'t, F>(&'t self, f: F) -> Result<Var<'t>, AutodiffError>` where `F: FnOnce() -> Result<Var<'t>, AutodiffError>`
  `lo = nodes.len()` を記録 → `f()` 実行 → 戻り `Var` の `tape_id` 検査（不一致は `TapeMismatch`）→ `output.id < lo`（区間が空）なら no-op で `Ok` → 区間登録＋解放。`f()` が `Err` の場合はそのまま伝播し**何も解放しない**
- `Var::checkpoint_from(&self, inputs: &[&Var<'t>]) -> Result<Var<'t>, AutodiffError>`（`self` をそのまま返す）
  `lo = inputs の最大 NodeId + 1`（空ならテープ先頭〈0〉）、`output = self`。同一テープ検査後 `Tape::register_checkpoint` を呼ぶ薄いラッパー
- 解放は `nodes.try_borrow_mut()` を使い、`Ref` 保持中に呼ばれた場合は panic ではなく `AutodiffError::InvalidArgument` で fail-closed
- **facade `Tape` newtype への `checkpoint` passthrough は追加していない**（承認事項。§9 参照）。facade からは既存の `Var` 再エクスポート経由（`Var::checkpoint_from`）でのみ到達可能

### 3.3 解放適格性（`Op::is_checkpoint_eligible()`。ワイルドカードなしの網羅 match）

| 分類 | Op | 扱い |
|---|---|---|
| 再計算可能（解放対象） | `MatMul`・`Sigmoid`・`Sum`・`Max` | `value` を `take()`・`recompute = true` |
| view（解放対象） | `Reshape`・`Transpose`・`Permute`（#1597）・`BroadcastTo`（#1597）・`Narrow`（#1598） | 既存の再計算方式化（イシュー #1047・#1597・#1598）と統合。`Op::is_checkpoint_eligible()` は網羅 match（ワイルドカードなし）のため、新規 view variant 追加時はコンパイルエラーで判断が強制される（PR #1681 codex-review 指摘: merge 時に `Narrow` の分岐漏れが一時発生していたのもこの網羅性チェックにより検出された） |
| 遅延 elementwise（**解放しない**） | `Add`・`Mul`・`Relu`・`Exp`・`Tanh` | `MAX_FUSED_CHAIN_LEN` 上限維持のための自己実体化であり、解放すると `build_lazy_plan` の不変条件を壊す |
| 非適格（値を保持・エラーにしない） | `Leaf`・`ResidentLeaf`・`LinearResident`・`LinearAct`・`MseLoss`・`CrossEntropyLoss`・`RnnCell`・`LstmCell`・`LstmHidden`・`GruCell`・`Inv`・`Solve`・`Det`・`Cholesky`・`QrQ`／`QrR`・`SvdU`／`SvdS`／`SvdVh`・`MatrixNorm`・`Softmax`・`LogSoftmax` | 正しさ優先・メモリ削減はベストエフォート（§8 スコープ外） |

**実装スコープの縮小（計画からの意図的な変更）**: 当初計画では `MseLoss`／`CrossEntropyLoss`／`LinearAct`／`Softmax`／`LogSoftmax`／線形代数系も含めた広い eligible 集合と、`Var::*` の forward 計算ロジックを `recompute.rs` へ独立抽出する「forward_eager 共有化」リファクタを想定していたが、実装時に以下の理由で eligible 集合を `MatMul`／`Sigmoid`／`Sum`／`Max`（+ 既存 view）へ絞った:

1. **`Var::matmul`／`sigmoid`／`sum`／`max` の forward 計算は `let value = self.tape.ops().X(...)?;` という 1 行の呼び出しのみ**（`Var::sigmoid` は `eval::sigmoid`）で、他の演算（`mse_loss`／`cross_entropy_loss`／線形代数系）のような shape 検査・`Unsupported` フォールバック分岐を持たない。このため `recompute_value`（`tape.rs`）内でこれら 4 演算の呼び出しを直接複製しても、var.rs 側のロジックとの重複は実質 1 行×4 個に留まり、独立ファイルへの抽出という大きな追加リファクタなしに bit 同一性を機械的に保証できる
2. `MseLoss`／`CrossEntropyLoss`・線形代数系（`Inv`／`Solve`／`Det`／`Cholesky`／`QrQ`／`QrR`／`SvdU`／`SvdS`／`SvdVh`／`MatrixNorm`）・`Softmax`／`LogSoftmax` は shape 検査・複雑な forward ロジック・（線形代数系は）多出力 payload を伴い、再計算時に完全に同一の経路を再現するには `recompute.rs` への本格的な forward_eager 抽出（var.rs 側の書き換えを伴う独立コミット）が必要で、本イシューの実装予算内でのリスク（既存 12 個以上の `Var::*` メソッドへの手入れによる回帰）に見合わないと判断した
3. `LinearAct`／`LinearResident` は `Activation` epilogue・resident デバイスバッファという追加の複雑性を持つ

この縮小判断はユーザー承認が前提の「out-of-scope-tracking.md」に基づく対象外事項として §8 に記録し、拡張余地は残す（`Op::is_checkpoint_eligible()` の網羅 match により、将来これらの Op を追加する際はコンパイルエラーで判断が強制される）

### 3.4 bit 同一を「構造」で保証する: `recompute_value` が forward と同じ呼び出しを再現する

`crates/autodiff/src/tape.rs` に以下を実装した:

```
fn recompute_value(
    nodes: &[TapeNode],
    ops: &dyn BackendOps,
    id: NodeId,
    memo: &mut std::collections::HashMap<usize, Tensor<f32>>,
) -> Result<Tensor<f32>, AutodiffError>
```

- `Op::Reshape`／`Op::Transpose`／`Op::Permute`（#1597）／`Op::BroadcastTo`（#1597）／`Op::Narrow`（#1598）: 既存 `resolve_view`（旧名）と同じ `Tensor::reshape`／`transpose`／`permute`／`broadcast_to`／`narrow`（zero-copy view 合成）
- `Op::MatMul(a, b)`: `ops.gemm(&a_val, &b_val)`（`Var::matmul` と同一呼び出し）
- `Op::Sigmoid(a)`: `eval::sigmoid(&a_val)`（`Var::sigmoid` と同一呼び出し）
- `Op::Sum { input, dim }`: `ops.sum(&input_val, dim)`（`Var::sum` と同一呼び出し）
- `Op::Max { input, dim }`: `ops.max(&input_val, dim)`（`Var::max` と同一呼び出し）
- それ以外（非適格・非 view ノードが未実体化のまま到達）: `Err`（契約違反として `recompute_infallible` 側で吸収）

`recompute_fallible`（層 1 用。`id` 自身の `OnceCell` へキャッシュする）と `recompute_infallible`（層 2 用。キャッシュしない）の 2 段構成にした理由は §4「reentrant init 問題と解決」参照。

**決定性の前提**: CPU BLIS・CUDA・Metal の GEMM は run-to-run bit 同一（既存の parity テスト・candle 比較で担保済み）。再計算は forward と同一の `ops` メソッド・同一入力を用いるため出力 bit 同一になる。

**反復的な post-order 走査（PR #1681 codex-review 是正・イシュー #1624）**: 当初実装の `recompute_value` は Rust の呼び出しスタックを直接使う再帰関数だった。`memo`（共有祖先の重複計算防止。§4 参照）は再帰**深さ**自体は抑制しないため、checkpoint 区間内に十分深い祖先鎖（例: 小さいテンソルへの `sigmoid` を数万段連鎖させた区間）を forward で構築し checkpoint 解放後に backward で再計算すると、Rust のスタックサイズを超えてプロセスが abort していた（本番経路 panic 禁止方針にも反し、`Result` にすら変換できない）。`RecomputeFrame::{Visit, Process}` の 2 段フレームによる明示的な作業スタック（`Vec`。ヒープ確保）を使う反復的な post-order 走査へ書き換え、`recompute_value` 自身の再帰深さを定数（この関数の呼び出しフレーム自体は 1 段）に固定した。共有祖先の重複計算防止（`memo` によるメモ化。`h = h.matmul(&h)` 型の fan-in）は `Visit` フレームの冒頭で `memo` を検査してスキップする形で従来どおり成立する。回帰テストは `crates/autodiff/tests/checkpoint_review_1624.rs::deep_checkpoint_chain_recompute_does_not_overflow_stack`（`sigmoid` を 50,000 段連鎖させた区間の単一の再計算呼び出しが abort しないことを確認）。

### 3.5 poison 契約: checkpoint 解放済みノードの再計算失敗を検出・伝播する（PR #1681 codex-review・Cursor Bugbot 指摘。イシュー #1624）

再計算（`recompute_value`）はバックエンド実行を伴うため、`push_view` の再導出前提（「入力は必ず実体化済み・再導出可能」という**構造的**契約）とは異なり、shape 不整合・カーネル起動失敗等の**実際に起こりうるバックエンド実行失敗**を持ちうる。この失敗を「あり得ない契約違反」として `debug_assert!` で握り潰すと、release ビルドではゼロテンソルへ静かに変換され、以後の計算・勾配が破損したまま「成功」として伝わってしまう（fail-open）。

- **`TapeNode::recompute_failed: Cell<bool>`**（poison フラグ）を追加した。層 2 専用ヘルパー `recompute_infallible`（`Result` を返せない契約のため、失敗時はゼロテンソルで応答せざるを得ない）が、対象ノード自身の再計算に失敗した場合にこのフラグを立てる。
- **`poisoned_err(node)`**: `recompute_failed` が立っているノードのキャッシュ済み値を「正しい実体化結果」として信頼せず `Err` を返すゲート。層 1 の全ての値読み出し経路（`recompute_value` 自身の cached-value 早期リターン・`materialize_fallible`・`lazy_leaf_value_fallible`・`value_of`〈fallible 版〉）がキャッシュ済み値を読む前に必ず呼ぶ契約とする。これにより `Tape::backward`（層 1 のみを経由する）は、checkpoint 解放済みノード自身が過去に再計算失敗を起こしていれば確実に `Err` を返す。
- **lazy な子孫への伝播（Bugbot 指摘の追加是正）**: `out = m.relu()` のように、checkpoint 解放済みノード `m` を入力に取る**未実体化 elementwise ノード `out`**（`m` 自身は非適格〈`Op::Relu` は §3.3 の「遅延 elementwise」〉のため `out` 自身は checkpoint に解放されない）を層 2（`to_tensor()`）から直接実体化すると、`materialize_non_fallible` は `build_lazy_plan` → `fallback_per_op`（いずれも `lazy_leaf_value_fallible` 経由で poison 検査つき）→ **`eval_fallback`**（最終手段。旧 `lazy_leaf_value`〈poison 検査なし〉経由）の順にフォールバックする。`build_lazy_plan`／`fallback_per_op` が `m` の poison を検出して返す `Err` は `.ok()` で握り潰され、最終的に `eval_fallback` がゼロ値で `out` の値を確定してしまう——`m` 自身は `recompute_infallible` 経由で poison されるが、`out` 自身の `recompute_failed` は**立たない**ため、`out` のキャッシュ済みゼロ値がそのまま「正常値」として以後の層 1 に読まれてしまう（層 2 は infallible な契約上、値そのものを返さない選択肢がないため、返す値自体は変えられない）。この伝播漏れを埋めるため、`materialize_non_fallible` はクロージャ内で値を計算した**後**に `elementwise_leaves_poisoned(nodes, id)`（`id` を根とする未実体化 elementwise 連結成分を `build_lazy_plan`／`eval_fallback` と同じ走査で辿り、到達した葉ノードのいずれかで `recompute_failed` が立っているかを検査する補助関数）を呼び、真なら `id` 自身の `recompute_failed` も立てる。以後 `materialize_fallible`（層 1）がこの `id` を読む際、`poisoned_err` で確実に `Err` へ変換できる。
- 回帰テストは `crates/autodiff/tests/checkpoint_review_1624.rs`:
  - `layer2_poisoned_recompute_is_detected_by_layer1_backward` — checkpoint 解放済みノード自身（`m`）が層 2 で先に poison された場合の検出（既存の `lazy_leaf_value_fallible` の poison 検査で成立）
  - `layer2_poison_propagates_to_lazy_descendant` — `m` を一度も単独で読み出さず、lazy な子孫 `out = m.relu()` を層 2 から直接実体化した場合の伝播漏れの検出（本節の追加是正が必要な回帰）

### 3.5.1 eager 演算（`push_eager`）への poison 伝播（PR #1681 codex-review 追加指摘。イシュー #1624）

上記 §3.5 の伝播は「未実体化 lazy elementwise 連鎖」を対象にしたものであり、`Var::sigmoid` 等の **eager 演算**（`Tape::push_eager` 経由で登録される非 elementwise・常実体化のノード。`Op::MatMul`／`Sum`／`Max`／`Sigmoid`／`MseLoss`／`LinearResident` 等）は対象外だった。`Var::sigmoid` は非 fallible な契約を保つため入力を `materialize_fallible`（層 1）ではなく `self.value()`（層 2）で読む——`m = a.matmul(&b)` を checkpoint 区間の内部ノードとして解放した後、`m.sigmoid()` の再計算失敗が `m` に poison を立てても、`push_eager` は新規登録するノードを常に `recompute_failed: false` で登録していたため、`sigmoid` の出力が「0.5 の一様値」を正常な計算結果として以後の `sum(None)`（fallible）・`Tape::backward` へ渡してしまっていた（fail-closed 契約違反。`.claude/rules/security.md` A08）。

- **`Op::for_each_input`**（`is_checkpoint_eligible` と同じくワイルドカードなしの網羅 match）を追加した。`self` が参照する全入力 `NodeId` を列挙する（`Op::Concat` のみ複数回・`Option<NodeId>` フィールドは `Some` の場合のみ）。新しい `Op` variant を追加した際に「どのフィールドが入力ノードか」の判断漏れをコンパイルエラーで検出する。
- **`Tape::push_eager`** は新規ノード登録の直前に `op.for_each_input` で全入力を走査し、`elementwise_leaves_poisoned(&nodes, input_id)`（§3.5 と同じ補助関数の再利用。入力自身が直接 poison されている場合・入力が未実体化の lazy elementwise 連鎖でその葉が poison されている場合の両方を検出する）がいずれかの入力で真なら、登録するノード自身も `recompute_failed: true` で登録する。
- **本チェックが実際に効く経路は限定的**: `push_eager` の呼び出し元のうち `matmul`／`sum`／`max`／`rnn_cell`／線形代数系（`inv`／`solve`／`det`／`cholesky`／`qr`／`svd`）等は入力を `materialize_fallible`（層 1）で読み `?` で poison を即座に `Err` 化するため、これらは元々 poison した入力で `push_eager` へ到達しない（`elementwise_leaves_poisoned` は常に false）。実際に効くのは `Var::sigmoid`（`.value()`／層 2 経由）のような infallible 読み出しを行う経路のみであり、将来同種の経路が追加された場合の安全網としても機能する。
- 回帰テストは `crates/autodiff/tests/checkpoint_review_1624.rs::eager_sigmoid_propagates_poison_from_checkpoint_freed_input`（`m.sigmoid()` の出力が `m` の poison を継承し、以後の `sum(None)`・`Tape::backward` がいずれも `Err` を返すことを確認する）。

## 4. reentrant init 問題と解決（実装中に発見した設計上の落とし穴）

当初、`materialize_non_fallible`（`OnceCell::get_or_init` を使う層 2）の `get_or_init` クロージャ内で、対象ノード自身の値を再計算し `OnceCell::set()` する実装にしたところ、`thread '...' panicked ... reentrant init` が発生した。原因は `OnceCell::get_or_init` が自身のクロージャ実行中に同じセルへの書き込み（`set`）を検出すると reentrant として panic する契約のため。

解決: `recompute_value`（キャッシュしない純粋な再帰計算。祖先ノードの再帰呼び出しは `recompute_fallible` を使うため祖先はキャッシュされる）と、それを包む 2 つの薄いラッパー（`recompute_fallible` は `id` 自身をキャッシュ、`recompute_infallible` はキャッシュしない）に分離した。`materialize_non_fallible` の `get_or_init` クロージャは `recompute_infallible`（キャッシュしない）を呼ぶことで、`get_or_init` 自身が返り値をキャッシュする既存契約と衝突しない。

## 5. backward_impl の借用構造変更

`backward_impl`（`crates/autodiff/src/backward.rs`）は従来「逆走査ループ全体を単一の `let nodes = self.nodes.borrow();` で完結させる」設計だったが、checkpoint の再解放（`Tape::release_checkpoints_ending_at`。`self.nodes.borrow_mut()` を要する）を逆走査の途中で呼ぶ必要があるため、「反復ごとに `Ref` を取得し、その反復内（VJP 呼び出しまで）で drop してから解放を試みる」方式へ再構成した。既存テスト（`tests/backward.rs`・`tests/tape_reset.rs`・`fusion_backend_integration.rs::tape_is_send` 含む）は全て非後退を確認済み。`docs/autodiff-higher-order-grad-decision.md` §2 に本変更の影響を追記した。

## 6. 対象ファイル・変更箇所

| パス | 変更 |
|---|---|
| `crates/autodiff/src/tape.rs` | `TapeNode.recompute: bool` 追加・`Op::is_checkpoint_eligible()`（網羅 match）・`CheckpointRegion`／`Tape.checkpoints` フィールド・`Tape::checkpoint`／`register_checkpoint`／`release_checkpoints_ending_at`・`release_checkpoint_region`（自由関数）・`recompute_value`／`recompute_fallible`／`recompute_infallible`（旧 `resolve_view` を置換・一般化）・`build_lazy_plan`／`lazy_leaf_value`／`fallback_per_op`／`eval_fallback` へ `ops` パラメータを追加してスレッド・`Tape::reset` でレジストリ消去 |
| `crates/autodiff/src/var.rs` | `Var::checkpoint_from` 追加 |
| `crates/autodiff/src/backward.rs` | `backward_impl` を反復ごと借用へ再構成し、区間離脱時に `release_checkpoints_ending_at` を呼ぶ |
| `crates/autodiff/src/lib.rs` | クレート doc に checkpoint の要約・本 doc への参照を追記 |
| `crates/autodiff/Cargo.toml` | `[[test]] name = "checkpoint_peak_memory" harness = false` 追加（`view_zero_alloc` と同型） |
| `crates/autodiff/tests/checkpoint.rs`（新規） | 統合テスト 14 件（§7 (b)） |
| `crates/autodiff/tests/checkpoint_peak_memory.rs`（新規） | ピークメモリ実測（§7 (c)） |
| `crates/facade/tests/checkpoint_backend_bit_identity.rs`（新規） | Metal 実機 `#[ignore]` テスト（§7 (d)） |
| `docs/autodiff-checkpoint-design.md`（本ファイル） | 設計判断・契約整理・実装記録・実測記録・承認事項・スコープ外 |
| `docs/compat-api-scope.md` §1.3 | 実装済み化への更新 |
| `docs/compat-feature-gap.md` §2.11 | fandhe-ai 列を実装済みへ更新 |
| `docs/autodiff-higher-order-grad-decision.md` §2 | `backward_impl` 借用構造変更の追記 |
| `CLAUDE.md` | 本ファイルへの参照追記 |

`docs/spec/` は編集していない。`BackendOps` trait・tolerance 定数・`BASELINES`・`Cargo.lock` は不変。依存追加なし・`unsafe` 追加なし。

## 7. 検証結果

- **ビルド・lint**: `cargo fmt --all --check`・`cargo clippy -p fandhe-ai-autodiff --lib -- -D warnings` green。`cargo doc -p fandhe-ai-autodiff --no-deps` warning なし
- **既存テスト非後退**: `cargo test -p fandhe-ai-autodiff` 全 green（275+3+15+... 全ファイル 0 failed。`tests/view_zero_alloc.rs`・`tests/fusion_backend_integration.rs::tape_is_send` 含む）
- (a) `tape.rs` 単体テストは今回は追加していない（`crates/autodiff/tests/checkpoint.rs` の統合テストで挙動を機械検証。private アクセス単体テストは §8 のフォローアップ候補）
- (b) `crates/autodiff/tests/checkpoint.rs`（14 件、全 green）:
  1. `checkpoint_grads_are_bit_identical_to_no_checkpoint` — MLP 連鎖で checkpoint 有無の全勾配が bit 同一
  2. `checkpoint_grad_matches_numeric_grad` — 中央差分との突合
  3. `multiple_sequential_checkpoint_regions_match_no_checkpoint` — 3 区間連鎖
  4. `nested_checkpoint_regions_match_no_checkpoint` — 入れ子区間
  5. `backward_called_twice_after_checkpoint_yields_same_gradient` — `backward()` 2 回で同一
  6. `failing_closure_does_not_release_and_tape_remains_usable` — 閉包が `Err` を返した場合何も解放されない
  7. `checkpoint_returning_var_from_another_tape_is_rejected` — `TapeMismatch`
  8. `empty_region_returning_existing_var_is_noop` — 区間が空の no-op
  9. `checkpoint_registry_is_cleared_after_reset` — `reset()` 後の新 epoch でも機能する
  10. `checkpoint_region_with_long_elementwise_chain_matches_no_checkpoint` — 融合連鎖を含む区間
  11. `escaped_intermediate_var_value_matches_original_after_release` — 逃げ出した中間 `Var` の `value()`（層 2）が再計算値を返す
  12. `checkpoint_region_with_view_ops_matches_no_checkpoint` — 二重 transpose（view チェーン）
  13. `checkpoint_from_matches_closure_based_checkpoint` — `Var::checkpoint_from` と `Tape::checkpoint` の結果一致
  14. `checkpoint_region_containing_ineligible_op_does_not_error` — 非適格 Op（`qr`）を含む区間でもエラーにならない
- (c) `crates/autodiff/tests/checkpoint_peak_memory.rs`（`harness = false`・`TrackingAllocator::measure`）: `K=6` 区間（各区間 `matmul → sigmoid → matmul → sigmoid`。256×256 f32 = 256 KiB/活性化）を連鎖させ、forward+backward の純増分ピークを比較した。**このマシン（M4 Max、CPU バックエンド）での実測値**: `peak_plain=20212792` bytes・`peak_ckpt=16279680` bytes（削減 3933112 bytes。理論値 `3*K*activation_bytes=4718592` bytes の 83%）。`peak_ckpt < peak_plain` かつ削減幅が理論値の半分を上回ることを機械的に確認した（`Gradients` が全ノード分の勾配を保持するため理論値 100% には届かない。§8 参照）
- (d) facade `#[ignore]` 実機テスト（`crates/facade/tests/checkpoint_backend_bit_identity.rs`）: **このマシン（Apple M4 Max・Metal）で実行・pass 確認済み**（`cargo test -p fandhe-ai --test checkpoint_backend_bit_identity -- --ignored --nocapture`）。`matmul → relu → matmul → sigmoid → mse_loss`（`mse_loss` を使う理由: `MetalBackendOps::sum`／`max` が `Unsupported` のため）の 2 層チェーンで checkpoint 有無の勾配が Metal 実機上で bit 同一であることを確認した。**CUDA は本エージェント実行環境に実機到達手段がないため未実測**（テストファイル自体は Metal 専用〈`#![cfg(target_os = "macos")]`〉のため CUDA 版は別途追加が必要。§8 参照）

## 8. スコープ外（`out-of-scope-tracking.md` に従い記載。起票はユーザー承認後）

- `Op::is_checkpoint_eligible()` の対象拡大（`MseLoss`／`CrossEntropyLoss`／`LinearAct`／`Softmax`／`LogSoftmax`・線形代数系）と、それに伴う `Var::*` の forward_eager 共有化リファクタ（§3.3「実装スコープの縮小」参照）
- facade `Tape::checkpoint` passthrough の追加（承認事項。§9）
- `Gradients` からの非葉勾配の早期解放（ピーク削減の残り半分。`Gradients::get` の契約変更を伴う）
- 遅延 elementwise ノードの解放（`MAX_FUSED_CHAIN_LEN` 契約の再設計が前提）
- reuse／resident 経路（`LinearResident`・`ResidentLeaf`・`DeviceParamStore`）向けのデバイス側解放・CUDA Graph capture との併用
- CUDA 実機での bit 同一実測（`#[ignore]` テストの CUDA 版自体が未作成。GB10 実機セッションへ申し送り）
- `nn::Sequential`／compat 層へのチェックポイント指定 API
- 祖先の再計算結果を `recompute_value` 自身がキャッシュしない設計（§4）による性能面のトレードオフ（同一 checkpoint 区間内で複数ノードが同じ祖先を必要とする場合、その祖先を都度再計算する）の最適化
- `tape.rs` 内の `#[cfg(test)]` 単体テスト（private フィールドへの直接アクセスによる `recompute` フラグ・レジストリの検証）の追加

## 9. 承認事項

- **facade `Tape::checkpoint`（閉包版）の passthrough 追加**: 現時点では追加していない。`Var::checkpoint_from`（低儀式版）が既存の `Var` 再エクスポート経由で facade からも到達可能なため、facade ユーザーは checkpoint 機能自体は利用できる。閉包版が必要な場合は別途ユーザー承認のうえ追加する
