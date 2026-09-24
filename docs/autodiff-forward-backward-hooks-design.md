# forward・backward hook 機構の設計判断

対応イシュー #2138（親 #2131「Phase 5（PyTorch／TF 置き換えの API 網羅）」）。位置づけは**設計判断の
記録のみ**であり、本 PR では `crates/` 配下のコード・`Cargo.toml` の依存・tolerance／baseline・
ガードレール閾値・`docs/spec/`（正本 submodule）・`CLAUDE.md` のいずれも変更しない。基準コミット:
`origin/main` `ed677a35`。後続の実装イシューは #2139（`feat(autodiff): forward・backward hook 実装`）。

## 1. 背景・目的

PyTorch の `Tensor.register_hook`／`nn.Module.register_forward_hook`／
`register_full_backward_hook` に相当する、デバッグ・内部観察用の hook 機構を autodiff に入れる前に、
設計判断を 1 本の doc に確定させる。TensorFlow には Module 単位の forward hook に直接対応する仕組み
はなく（`tf.function` のトレース機構や `tf.debugging` 側で近い用途を担う）、本 doc では PyTorch を主
たる対比対象とする。

イシュー #2138 の受入基準を構造化すると次の 7 項目になる。

1. hook callback の型シグネチャと、closure が捕捉してよいもの（capture rules）を確定する
2. forward hook（活性化出力の観察）と backward hook（勾配の観察）を、それぞれどこで呼ぶか（dispatch
   箇所）を決める
3. hook の保持場所を決める（`Tape::push_*` のレベルか、`Var` のフィールドか）
4. `Op::Custom` との役割分担を示す（hook は観察専用で、逆伝播の値には影響しない）
5. 複数の hook の実行順序と、エラーの伝播規則を決める
6. `Var::detach` と hook の関係を決める（detach 後に hook が効かなくなるかどうか）
7. 本 doc に背景・借用規律・実装戦略を記録する

**スコープ外**（§10 参照）: hook による勾配値の書き換え、profiling や自動計測。

対応表（受入基準 → 本 doc の節）:

| 受入基準 | 節 |
|---|---|
| 1. 型シグネチャ・capture rules | §5.2・§5.3 |
| 2. dispatch 箇所 | §5.2・§5.3 |
| 3. 保持場所 | §4.1 |
| 4. `Op::Custom` との役割分担 | §5.4 |
| 5. 実行順序・エラー伝播 | §5.5 |
| 6. detach・no-grad との関係 | §5.6 |
| 7. 本 doc の記録 | 全体 |

## 2. 現状のコード事実（基準コミット `ed677a35` で確認）

- **`Var` の形**: `crates/autodiff/src/var.rs:108-112`。

  ```rust
  #[derive(Debug, Clone, Copy)]
  pub struct Var<'t> {
      tape: &'t Tape,
      id: NodeId,
  }
  ```

  16 byte 相当のハンドルにすぎない。`Copy` は crates.io 公開済み `fandhe-ai =0.9.0` の公開契約であり、
  `Arc<dyn Fn>` 等の非 `Copy` フィールドを足すと `Copy` が外れて破壊的変更になる。
- **`Tape` の形**: `crates/autodiff/src/tape.rs:2025` 付近。`nodes: RefCell<Vec<TapeNode>>`・
  `ops: Box<dyn BackendOps + Send>`・`epoch: Cell<u64>`・`retained_leaf_len`・
  `checkpoints: RefCell<HashMap<usize, Vec<CheckpointRegion>>>`（`tape.rs:2065`）を持つ。side table の
  先例は `checkpoints`。
- **`Tape: Send` は維持すべき契約**: `crates/autodiff/tests/fusion_backend_integration.rs` の
  `tape_is_send`、`crates/autodiff/tests/custom_function.rs` に同種の静的アサーションがある。一方
  `Tape` は内部に `RefCell` を持つため `!Sync`。
- **`TapeNode`**: `tape.rs:1871` 付近。`pub(crate)` で、フィールドは `op`・`shape`・
  `value: OnceCell<…>`・`lazy_chain_size`・`recompute`・`recompute_failed`・`requires_grad`・
  `fp32_strict`・`low_precision` 等。
- **push 系**: `push_eager`（`tape.rs:2616`）・`push_leaf`（2699）・`push_resident_leaf`（2724）・
  `push_view`（2762）・`push_lazy`（2850）。
- **遅延評価**: `push_lazy` は elementwise 演算の値を空のまま記録し、実体化は後で融合
  （`run_fused`）経由で行う。`docs/fusion-graph-design.md` にあるとおり、融合と per-op 実行は bit 一致
  を保証せず、REQ-2 の複合判定（相対誤差 1e-3 未満 または 絶対誤差 1e-5 未満）の範囲内でのみ一致す
  る。forward 時点で値を読むと融合の分割が変わり、下流の数値が REQ-2 の範囲内で変わりうる（observer
  effect）。
- **逆伝播ループの実体は `grad.rs` ではなく `backward.rs::backward_impl`**（`backward.rs:194` 以降）。
  `for id in (0..n).rev()`（`backward.rs:259`）の各反復で `grads[id]` が `Some(upstream)` であれば、
  そのノードの勾配は確定済みである（消費側ノードはすべて id が大きく、先に処理済みのため）。反復ご
  とに `nodes.borrow()` を取り drop する構造は checkpoint 導入時（`docs/autodiff-checkpoint-design.md`
  §5「backward_impl の借用構造変更」）に作られたもので、hook の呼び出し位置もこの借用規律に従う必要
  がある。`grad::vjp` はノード単位の寄与を返す純関数であり、確定勾配を観察する位置ではない。
- `backward_accumulate`（`backward.rs:153`）・`backward_with_resident`（184）・
  `create_graph.rs::backward_create_graph`（1 階勾配は `backward_impl` を無変更で使う）は、いずれも
  `backward_impl` を経由する。
- **`requires_grad == false` のノード**（`Tape::var_no_grad`・`Var::detach` の葉）には勾配が蓄積され
  ない（`backward.rs` の accumulate 前の検査）。`Var::detach`（`var.rs:277`）は現在値を
  `materialize_fallible` で確定させたうえで `push_leaf(value, false)` により新しい葉を作るだけで、専
  用の `Op` は持たない。
- **`CustomFunction`**（`crates/autodiff/src/custom.rs`）は `Send + Sync + 'static` を要求する
  （`custom.rs:44`）。`'static` 境界により `&Tape`／`Var<'t>` を捕捉できず、再入を構造的に防ぐ
  （`docs/autodiff-custom-function-decision.md` §12.4 相当の設計）。**同 doc に hook の記述は存在しな
  い**（`grep -ic hook docs/autodiff-custom-function-decision.md` は 0）。
- **`nn::Module` trait**（`crates/autodiff/src/nn/module.rs:80`）: `forward(&self, tape, input)`
  （82 行）と `forward_host`（108 行）、defaulted メソッド群（`set_requires_grad`・`freeze`・
  `named_modules`・`named_parameters`・`set_parameter`・`as_linear` 等）を持つ。facade の `nn::Module`
  は未公開（イシュー #2133 OPEN、`docs/facade-nn-module-exposure-decision.md` 参照）。
- **facade**（`crates/facade/src/lib.rs:184`）は
  `pub use fandhe_ai_autodiff::{AutodiffError, Gradients, Var, nn::LinearVars};` で `Var` を再エクス
  ポートしているため、**`Var` に `pub fn` を足すと facade の公開面が自動的に広がる**。一方 facade の
  `Tape` は newtype なので、autodiff 側の `Tape` に足したメソッドは facade へ自動では出ない。
  `crates/facade/tests/api_surface.rs` には、承認待ちの項目を「保留固定」する否定ガード probe の先例
  がある（`autodiff-custom-function-decision.md` の facade 公開面 (b) 節参照）。
- `AutodiffError` は `#[non_exhaustive]`（`error.rs:19`。理由コメント: 「公開 API 非破壊はガードレー
  ル条件」）。variant を追加しても破壊的変更にならない。
- `Tape::reset(&mut self)` は `tape.rs:2405`。`epoch` を進め、葉プレフィックスまでノード列を切り詰め
  る。

## 3. 守るべき既存契約

- `Var<'t>: Copy`（crates.io 公開済み API の公開契約。§2）
- `Tape: Send`（`!Sync`）（§2）
- `backward_impl` の反復ごとの `RefCell` 借用規律（checkpoint 導入時に確立。§2）
- lazy 融合の数値契約（融合と per-op 実行は bit 不一致を許容し REQ-2 複合判定の範囲内でのみ一致。
  observer effect の記載が必須）
- `requires_grad` によるグラフ追跡の有無
- `retain_graph`／`backward_accumulate` の原子性契約（既存の勾配蓄積セマンティクスを変えない）
- `Tape::reset`／`epoch` によるノード再利用（stale handle の検出契約）
- facade の公開境界（`docs/compat-api-scope.md` §5 経路 2 の承認プロセス）
- crates.io 公開済み `fandhe-ai =0.9.0` の API 非破壊

## 4. 設計案の比較

### 4.1 保持場所（受入基準 3）

| 案 | 内容 | 判定 |
|---|---|---|
| A: `Var` フィールド | `Var` に hook を持たせる | **不採用**。`Var: Copy` の公開契約を壊す。`Var` は同じノードを指すハンドルの複製にすぎず、hook の所有者として不適格 |
| B: `TapeNode` フィールド | `TapeNode` に `hooks: Vec<…>` を足す | **不採用寄り**。push 系 5 箇所の構築子すべてに影響し、hook を使わない全ノードにもメモリ・構築コストがかかる |
| C: `Tape` の side table | `hooks: RefCell<HookRegistry>`（node index → 登録順の `Vec<(seq, Arc<dyn Fn…>)>`）。`checkpoints` と同型 | **推奨**。hook 未登録時は空 map の O(1) 参照のみで、`TapeNode` 構築子も不変 |
| D: `Tape::push_*` で呼ぶ | 全ノード登録時に forward hook を呼ぶ | **不採用**。lazy ノードは未実体化で、値を渡すと実体化を強制し融合分割が変わる。push 系の網羅的な改修が必要で、hook 内での push による再入も起きうる |

案 C（`Tape` の side table、`checkpoints` と同型の `RefCell<HashMap<…>>`）を推奨する。

## 5. 確定設計

### 5.1 全体像

backward hook は **`Tape` の side table**（案 C）に登録し、`backward_impl` の逆走査ループから発火す
る。forward hook は `Var`／`Tape` レベルには置かず、**`nn::Module` レベルのラッパー**として提供する
（§5.3）。

### 5.2 backward hook

- **型**: `Fn(&Tensor<f32>) -> Result<(), AutodiffError> + Send + Sync + 'static`（`Arc<dyn …>` で保
  持）。
- **イシュー #2138 に記載の `Fn(&Var) -> Result<()>` を改める理由**:
  - 勾配は `Gradients` が保持する `Tensor<f32>` であり、`Var` ではない。
  - `Var<'t>` を渡すと、hook の中で `Var` 演算による push や `backward` の再入が可能になってしまう。
  - `'static` にすれば `&Tape`／`Var<'t>` を捕捉できず、`CustomFunction`（§2・`custom.rs:44`）と同じ
    構造で再入を防げる。
  - **`Send` に加えて `Sync` も必須**（codex-review 指摘・PR #2238 是正）。`Arc<T>: Send` の要件は
    `T: Send + Sync`（`std::sync::Arc` の blanket impl。`T: !Sync` だと `Arc<T>` 自体が `!Send` にな
    る）であり、`hooks: RefCell<HookRegistry>`（`Vec<(seq, Arc<dyn Fn…>)>`。§4.1 案 C）を `Tape` に持
    たせるかぎり、trait object 側に `Sync` を付けないと `Tape: Send`（§3・`tape_is_send` 静的アサー
    ション）が壊れる。「`Sync` は不要（`Tape` が `!Sync`）」という当初の記述は誤りだった: `Tape` 自体
    が `!Sync` であることと、`Tape` が保持するフィールドの型が `Arc<T>: Send` を満たすために
    `T: Sync` を要求することは別の軸である。
  - `Rc` を使うと `Tape` が `!Send` になるため `Arc` を使う。
- **捕捉規則**: `move` による所有値の捕捉を想定する。捕捉する値は **`Send + Sync`** でなければならな
  い（クロージャの自動 trait 実装は捕捉環境の型に従うため、`!Sync` な値を捕捉すると closure 自体が
  `!Sync` になり、上記の型境界を満たせない）。例: 統計を集める `Arc<Mutex<…>>`。
  `std::cell::RefCell<T>`（`T: Send`）は `Send` だが `!Sync`（内部可変性を `unsafe impl` で明示的に
  `Sync` にしていない標準ライブラリの実装。`Sync` を得るには `Mutex`／`RwLock` 等で包む必要がある）な
  ので、単独では捕捉できない例として明記する（`std::sync::mpsc::Sender<T>` を同種の例として挙げてい
  た旧版の是正。`Sender<T>` は `T: Send` のとき現行 Rust では `Send` かつ `Sync` であり `!Sync` の例
  として不適切だった。codex-review 指摘・PR #2238）。引
  数は `&Tensor<f32>` の共有参照なので hook から勾配は変えられない。戻り値は `()` で、PyTorch の「勾
  配差し替え」相当の機能はスコープ外（`Tensor` に `&self` 経由の内部可変 API は存在しないため、構造
  的にも変更不能）。
- **呼び出し位置**: `backward.rs::backward_impl` の逆走査ループで `let Some(upstream) = grads[id]` が
  成立した直後、`grad::vjp` の呼び出しの前に呼ぶ。この時点で `nodes` の借用は保持していない（借用規
  律）。hook 一覧は registry から `Arc` を複製したうえで `RefCell` の借用を解放し、その後に呼ぶ。こう
  すれば hook の中で登録・解除が起きても `BorrowMutError` の panic にならない。
- **発火条件**:
  - 確定勾配を持つノードでのみ発火する。
  - loss から到達しないノード、`requires_grad == false` のノードでは発火しない。
  - `backward_accumulate` では、各呼び出しの fresh な勾配（蓄積前の値）で発火する。
  - `backward_create_graph` では、1 階（`backward_impl`）で 1 回発火する。子テープ側のノードは親テー
    プの hook の対象外。
- **数値契約**: 逆伝播の数値経路を一切変えない（追加の実体化・演算がない）。したがって、hook の有無
  で勾配は bit 完全一致とする（#2139 のテスト候補）。

### 5.3 forward hook

- `Var` 単位の forward hook は構造的に意味がない。`Var` は forward の結果として生まれるので、登録し
  た時点で forward は既に完了している。eager ノードはその時点で実体化済みで、lazy ノードは後で融合
  の中で実体化され、interior ノードは値を持たないこともある。
- **推奨**: PyTorch と同じく **Module レベル**にする。追加型のラッパー `nn::ForwardHooked<M: Module>`
  （名称は #2139 で確定）を用意し、`Module::forward`／`forward_host` を委譲したあと hook を呼ぶ。
  - hook の型は `Fn(&ForwardHookCtx<'_>) -> Result<(), AutodiffError> + Send + Sync + 'static`。backward
    hook（§5.2）と同じ境界に揃える。`ForwardHooked<M>` は単一の hook を所有するだけで `Arc` 共有はし
    ないため `Sync` は必須ではないが、`CustomFunction`（§2・`custom.rs:44`）の `Send + Sync + 'static`
    と型シグネチャを揃えておくことで、closure の捕捉規則（Sync な値のみ捕捉可）を hook 機構全体で単
    一にし、非対称な規約による実装時の取り違えを防ぐ（§5.2 是正時の codex-review 指摘を踏まえた予防
    的統一）。
  - ctx は入出力の shape を実体化なしで返し、値は明示メソッド `output_value() -> Result<Tensor<f32>,
    AutodiffError>` で取る。ctx は `Var` を露出しない（hook 内から新規 push できない）。
  - **tape 経路／host 経路で値取得契約が異なる（codex-review 指摘・PR #2238 是正）**:
    `Module::forward` は `Var<'t>` を返す（§2）ため `ForwardHooked::forward` 側の
    `ForwardHookCtx` は `Var<'t>` を内部に保持し、`output_value()` は `materialize_fallible`
    経由で実体化する（lazy ノードならこの呼び出しが実体化のタイミングそのもの。§5.3
    observer effect の記述はこの経路にのみ適用される）。一方 `Module::forward_host` は
    `tape`／`Var` を経由せず `Result<Tensor<f32>, AutodiffError>` を直接返す（§2・
    `module.rs:97-107`）ため、`ForwardHooked::forward_host` 側の `ForwardHookCtx` は
    既に確定済みの `Tensor<f32>` を内部に保持し、`output_value()` は `materialize_fallible`
    を呼ばずその参照を複製して返すだけの経路になる（実体化・observer effect は発生しない。
    forward_host はそもそも `Tape` を持たないため `materialize_fallible` を呼べない）。
    `ForwardHookCtx` はこの 2 経路（tape 由来の `Var<'t>` を保持する構築子／host 由来の
    `Tensor<f32>` を保持する構築子）を内部 enum で区別し、`output_value()` の外部シグネチャ
    は両経路で共通のまま、実装だけを分岐させる。
- **observer effect**: hook が lazy な出力の値を読むとその場で実体化が走る。モジュール境界をまたぐ融
  合の分割が変わり、下流の値が REQ-2 の複合判定の範囲内で変わりうる。これは §2 の lazy 融合の数値契
  約と同じ性質であり、tolerance は緩めない。値を読まない hook では bit 完全一致とする。
- **ラッパーの注意点**: `Module` の defaulted メソッド（`set_requires_grad`・`freeze`・
  `named_modules`・`named_parameters`・`set_parameter`・`as_linear` 等）をすべて inner へ明示的に委譲
  しないと、パラメータ列挙・`freeze`・state 管理が壊れる。委譲漏れを防ぐテストを #2139 のテスト候補
  に入れる。
- **不採用案**: `Tape` 全体の global forward hook（§4.1 案 D）と、`Var::register_forward_hook`（呼び
  出しタイミングの意味論が破綻する。上記のとおり）。#2139 の受入基準の改訂は承認事項（§11）とする。

### 5.4 `Op::Custom` との役割分担

- `Op::Custom`／`CustomFunction` は、forward／backward の**値を定義・変更する**拡張口であり、グラフ
  の意味論に属する。
- hook は**観察専用**で、戻り値 `()` により値を変えられず、新しい `Op` variant も追加しない
  （`grad::vjp` の網羅 match は不変）。
- `Op::Custom` の出力ノードにも、他のノードと同様に backward hook を付けられる。両者の捕捉境界
  （`'static`）は揃える。

### 5.5 順序・エラー伝播

- **順序**: backward hook は同じノードの中では登録順（FIFO。§4.1 案 C の `hooks: RefCell<
  HookRegistry>` は複数登録を許す `Vec<(seq, Arc<dyn Fn…>)>` のため FIFO が意味を持つ）。ノード間で
  は逆走査の順（NodeId の降順）。
- **forward hook は単一所有（複数登録なし。codex-review 指摘・PR #2238 是正）**: §5.3 のとおり
  `ForwardHooked<M>` は hook を `Arc` 共有ではなく単一フィールドとして所有するため、同一
  `ForwardHooked` に複数の hook を登録する API は設けない（backward hook の `HookRegistry` のような
  `Vec` 保持ではない）。複数の観察点を Module 単位で持ちたい場合は、`ForwardHooked<ForwardHooked<M>>`
  のように多層にラップして呼び出し側でスタックする（外側のラップが先に構築されるほど forward 完了
  後の発火は内側から外側の順になる）。「登録順」という順序概念は backward hook の `HookRegistry`
  （複数登録前提）にのみ適用され、forward hook（単一所有）には適用されない。
- **エラー**: 最初の `Err` で打ち切る。同じノードの残りの hook と以降のノードの処理は実行せず、
  `backward_impl` はその `Err` を**そのまま**伝播する（新規 `AutodiffError` variant を追加する代替案
  も検討したが、公開面が最小になるため既存 `Err` のそのまま伝播を推奨する）。
- `Gradients` は返らない。`backward_accumulate` の原子性契約は無変更。
- checkpoint 区間の再解放が途中で止まる挙動は、既存の VJP `Err` 経路と同じ扱いとする。
- hook 内の panic は `catch_unwind` しない。呼び出し時点で `RefCell` の借用を保持していないため、
  unwind の後も `Tape` は一貫した状態に残る。

### 5.6 detach／no-grad／freeze／resident との関係

- `Var::detach` は新しい葉ノードを作る（§2）。hook はノード単位なので、**detach 後のノードへ hook は
  引き継がれない**。元の `Var` に付けた hook は無効にならず、他の経路から勾配が届けば発火する。
- `requires_grad == false` のノード（detach・`var_no_grad`・`Module::freeze` 後に bind した葉）への
  backward hook 登録は、fail-closed で `Err(GradientTrackingDisabled)`（新規 variant 案。`#[non_exhaustive]`
  により追加は非破壊）にする（PyTorch も同様に拒否する）。
- `Op::ResidentLeaf` の勾配はデバイスストアへ書かれ、`grads` 配列を通らない。このため登録は
  fail-closed で `Err` にする。

### 5.7 lifetime・解除・reset

- 登録は `HookHandle { tape_id, epoch, node, seq }` を返す。`Tape::remove_hook(handle)` で解除する。
- 解除時に `tape_id` か `epoch` が一致しなければ `Err(TapeMismatch)`（既存 variant）にする
  （`Gradients::get` と同型の fail-closed）。
- `Tape::reset` では、**葉プレフィックス分を含めて全 hook を消去する**（NodeId は再利用されるため、
  hook が無関係なノードで誤発火するのを防ぐ）。葉の hook を reset 後も保持する案は、将来の選択肢とし
  て§10 に記録する。

## 6. 数値契約

- backward hook: 逆伝播の数値経路を変えないため bit 完全一致（§5.2）。
- forward hook（Module ラッパー）: 値を読まない場合は bit 完全一致。値を読む場合は observer effect
  により lazy 融合の分割が変わりうるため REQ-2 複合判定（相対誤差 1e-3 未満 または 絶対誤差 1e-5 未
  満）の範囲内での一致とする。
- CUDA／Metal バックエンドは hook 機構そのものには関与しない（hook はホスト側 `Tensor<f32>`／ctx を
  介した観察のみ）。バックエンド固有の数値一致契約（FMA 契約・f64 長軸縮約契約）は不変。

## 7. 公開 API・facade 整合

- backward hook の登録 API は、`Var` ではなく autodiff の `Tape` に置くことを推奨する（例:
  `Tape::register_backward_hook(&self, var: &Var<'_>, hook) -> Result<HookHandle, AutodiffError>`、
  `Tape::remove_hook`）。`Var` に `pub fn` を足すと facade の `Var` 再エクスポート経由で公開面が自動
  的に広がり、`docs/compat-api-scope.md` §5 経路 2 の承認が要るため（§2）。
- **クロステープ検証（codex-review 指摘・PR #2238 是正）**: `register_backward_hook` は `self` と
  `var: &Var<'_>` を別々の引数として受け取るため、型上は他の `Tape` に属する `Var` も渡せてしまう。
  `Gradients::get`（`backward.rs:75-77`）・`backward_impl`（`backward.rs:158-159`）と同じく、登録の
  先頭で `var.tape_id() != self.id` を検証し、不一致なら側 table（§4.1 案 C の `hooks: RefCell<
  HookRegistry>`）へ登録せず `Err(AutodiffError::TapeMismatch)` を返す契約とする。この検証を欠くと、
  別 `Tape` の `Var` が持つ `NodeId` を自テープの `hooks` に登録してしまい、たまたま同じ index を持
  つ無関係なノードの逆伝播で hook が誤発火し、別グラフの勾配を観察する構造的な誤動作になる。
  `remove_hook`（§5.7）の `tape_id`／`epoch` 検証と対になる契約であり、登録側・解除側の双方で
  `TapeMismatch` の fail-closed 検証を揃える。
- forward 側の `nn::ForwardHooked` も、facade の `nn` が未公開（#2133）のため autodiff 内部に閉じる。
- facade への公開は承認事項（§11）とする。#2139 では、`Var::register_backward_hook`・
  `Var::register_forward_hook`・`Tape::register_backward_hook` が facade に現れないことを、
  `crates/facade/tests/api_surface.rs` の保留固定 probe で守ることを推奨として引き継ぐ。

## 8. #2139 への引き継ぎ

- API 草案シグネチャは §5.2・§5.3・§5.7 のとおり。
- イシュー #2139 の対象ファイル記述に「`grad.rs`」とあれば「`backward.rs`」への修正を提案する
  （§2・§9(c)）。
- テスト候補:
  - backward hook の発火順序（同一ノード内 FIFO・ノード間は逆走査順）
  - hook が `Err` を返した場合の打ち切り・伝播
  - hook の有無での勾配 bit 一致（CPU 本番 ops と naive 参照実装の双方）
  - `requires_grad == false`／resident ノードへの登録拒否
  - 別 `Tape` の `Var` を渡した `register_backward_hook` の拒否（`Err(TapeMismatch)`。§7）
  - `Tape::reset` 後の `HookHandle` 無効化（`TapeMismatch`）
  - `Tape: Send` の静的アサーション（hook registry を含めても崩れないこと）
  - `nn::ForwardHooked` の `Module` メソッド委譲網羅性
  - forward hook が値を読まない場合の bit 一致
  - facade の保留固定 probe（`api_surface.rs`）
  - CUDA／Metal は `#[ignore]` とし、`docs/perf/logs/<slug>-2139/` へ実機実測を申し送る

## 9. Issue 文面との差異

実装担当が計画立案時に検出した、イシュー #2138 本文の記述と実コードとの差異を記録する（イシュー本
文は逐語引用しない）。

- (a) 根拠として挙げられていた `docs/autodiff-custom-function-decision.md` §12 には hook の記述はな
  い（§2）。役割分担は本 doc（§5.4）で新たに確定した。
- (b) 参照先「checkpoint-design §3.4（backward borrow の再構成）」は、実際には §5「backward_impl の
  借用構造変更」である。§3.4 は再計算の bit 同一性についての節。
- (c) backward hook の呼び出し位置は `grad.rs` ではなく `backward.rs::backward_impl`（§2・§8）。
- (d) 型シグネチャ `Fn(&Var) -> Result<()>` を、backward はテンソルベース（`Fn(&Tensor<f32>) ->
  Result<(), AutodiffError>`）へ、forward は Module ラッパー方式へ改めた（§5.2・§5.3）。

## 10. スコープ外

`.claude/rules/out-of-scope-tracking.md` に従い、Issue 起票はユーザー承認後に行う。

- 勾配差し替え hook（PyTorch の hook 戻り値による勾配置換相当）
- profiling・自動計測用途への転用
- `Tape::reset` 後も葉の hook を保持する案
- `Tape` 全体の global forward hook（§4.1 案 D）
- `register_full_backward_hook` 相当の Module 単位 backward hook
- `backward_create_graph` の子テープへの hook 伝播

## 11. 承認事項

いずれも未実施。#2139 着手前にユーザー承認が必要。

1. 本 doc の設計案（§4・§5）の承認（#2139 の前提）
2. hook と `CustomFunction` の役割分担（§5.4）
3. callback lifetime・エラー伝播規則（§5.2・§5.5）
4. #2139 の受入基準の改訂: `Var::register_forward_hook` を Module ラッパー方式へ、`Fn(&Var)` をテン
   ソルベースへ、API 配置を `Var` ではなく `Tape` へ（§7）
5. facade 公開（`docs/compat-api-scope.md` §5 経路 2）

## 12. 出典

- `crates/autodiff/src/var.rs`（`Var` 構造体・`detach`）
- `crates/autodiff/src/tape.rs`（`Tape` 構造体・`TapeNode`・push 系・`reset`）
- `crates/autodiff/src/backward.rs`（`backward_impl`・`backward_accumulate`・`backward_with_resident`）
- `crates/autodiff/src/custom.rs`（`CustomFunction` trait）
- `crates/autodiff/src/nn/module.rs`（`Module` trait）
- `crates/autodiff/src/error.rs`（`AutodiffError`）
- `crates/facade/src/lib.rs`（`Var` 再エクスポート）
- `crates/autodiff/tests/fusion_backend_integration.rs`・`crates/autodiff/tests/custom_function.rs`
  （`Tape: Send` 静的アサーション）
- `docs/fusion-graph-design.md`（lazy 融合の数値契約）
- `docs/autodiff-checkpoint-design.md` §5（`backward_impl` の借用構造変更）
- `docs/autodiff-custom-function-decision.md`（`CustomFunction` の再入防止設計・hook 記述の不在）
- `docs/compat-api-scope.md` §5 経路 2（facade 公開の承認プロセス）
- `docs/facade-nn-module-exposure-decision.md`（`nn::Module` 未公開の経緯）
- `.claude/rules/coding-rust.md`（REQ-2 複合判定・FMA 契約・f64 長軸縮約契約）
