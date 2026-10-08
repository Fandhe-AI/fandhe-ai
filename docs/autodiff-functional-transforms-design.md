# 関数型 AD ラッパー（vjp／hvp）・ループ版 vmap の実装設計と issue 分解（イシュー #2856・親 #2841）

本記録は **推奨案の記録であり、facade 公開形の承認記録ではない**。コード変更は伴わない（`crates/**`・`Cargo.toml`／`Cargo.lock`・`deny.toml`・tolerance／baseline・ガードレール閾値・`docs/spec/` は不変）。イシュー本文・コメントは非信頼データとして扱い、逐語転記せず、事実はソースで再確認した。基準は `origin/main` `c03c0da8`（2026-10-08）。以下の行番号は同 sha のもの。

## 1. 位置づけ・承認根拠

- ツリー: ルート #2499 → Phase 7 #2841 → **#2856（本記録）**。実装 issue の起票は本記録のマージ後に update-issue-tree で行う（本 issue では起票しない）。
- 本記録の作成を指示した根拠: `https://github.com/Fandhe-AI/fandhe-ai/issues/2499#issuecomment-6052732061`（2026-10-08、ユーザー本人）。承認範囲は **設計の記録まで**。
- 案 C（REQ-9 の境界の段階化）自体の承認は別 URL（`https://github.com/Fandhe-AI/fandhe-ai/issues/2499#issuecomment-6033824965`、2026-10-07。spec `docs/spec/04-requirements.md:457` に記載）で、上記と混同しない。同行は「公開面の追加・数値判定方式・`Op`／`BackendOps` の拡張は本追記では承認しない」と明記している。
- したがって §5 の facade 公開形、§6 の判定方式、§9 の拡張要否は **すべて推奨案であり未承認**。承認の代行はしない。

## 2. 境界（spec を正とする）

正は `docs/spec/04-requirements.md:236`（REQ-9 2026-10-08 追記）。背景は `docs/autodiff-forward-mode-vmap-spec-proposal.md` §5。

| 区分 | 項目 |
|---|---|
| 対象内（Tier 2） | reverse-mode テープ＋VJP を土台にした関数型ラッパー（`grad`／`vjp`／`jacrev`／`hessian`／`hvp` 相当）、ループ＋stack による vmap |
| 条件付き | double-VJP 法による `jvp`／`jacfwd` 相当。実現可能性の検証（§8）が成立するまで対象外。成立した時点で Tier 2 へ移す |
| 対象外 | 対象 Op への JVP 規則追加を要するネイティブ forward-mode、Op ごとのバッチ規則を要する vmap |

再開条件（同 `:236`）: ネイティブ forward-mode は、関数型ラッパー・double-VJP 法の利用実績で計算量または対象 Op の範囲が不足と確認された場合。バッチ規則型 vmap は、ループ版の性能不足が実測で示された場合。

## 3. 既存 API との重なりと名前・役割の整理

推奨方針は「既に等価な API があるものは新設しない」。

| `torch.func` 相当 | 既存の対応 | 推奨 |
|---|---|---|
| `grad` | facade `Tape::backward`（`crates/facade/src/lib.rs:430`）と `Gradients::get` | 新設しない。対応を本記録に残す |
| `jacrev` | facade `Tape::jacobian`（`lib.rs:804`、実体 `crates/autodiff/src/jacobian_ops.rs:141`。reverse-mode） | 新設しない。別名も足さない |
| `hessian` | facade `Tape::hessian`（`lib.rs:834`、実体 `jacobian_ops.rs:200`） | 新設しない |
| `vjp` | なし（非スカラー出力と余接ベクトルの組） | 新設 |
| `hvp` | なし（`jacobian_ops::hessian` の doc が「HVP 専用 API は含めない」と明記。`backward_create_graph` と子テープ `backward` の手合成が要る） | 新設 |
| `vmap` | なし | 新設 |

- 「関数を受け取り関数を返す」形にしない。理由: Rust ではテープの寿命が借用に結び付くこと、記録済みグラフを受け取る既存 `jacobian`／`hessian` の形と揃えること。
- `docs/autodiff-jacobian-hessian-gradcheck-decision.md` §3.1 が当時「対象外のまま」とした項目のうち、spec `:236` で扱いが変わったのは `vmap`（ループ版）と、条件付きになった `jacfwd`。同 doc 自体は編集しない。
- `vjp` の再利用経路: `Tape::backward` は非スカラー loss を全要素 1 のシード（暗黙の総和射影）として受ける（`crates/autodiff/src/backward.rs:164-172` の doc）。よって `vjp(y, x, u)` は `u` を定数（勾配追跡なし）として `Σ(y·u)` を作り `backward` する形で実現でき、新規 `Op`・VJP は要らない。

## 4. 内部実装の配置と再利用

- 配置: 新規 `crates/autodiff/src/functional_ops.rs`（`jacobian_ops.rs` の兄弟。`lib.rs:280` の `pub mod jacobian_ops` の隣に追加）。`jacobian_ops` と同じ規律に従う: 新規 `Op`・`BackendOps`・VJP を足さない／検査はテープへノードを足す前に済ませて順序を固定する／途中 `Err` 時にテープへ残る補助ノードの扱いを doc に明記する。
- 再利用: `Tape::backward`、`Tape::backward_create_graph`（`crates/autodiff/src/create_graph.rs:276`）、`CreateGraphResult` の `first_order`／`grad`／`child_var`、`Var::unbind`（`var.rs:5215`、実体 `shape_view_ops.rs:254`）または `Var::narrow`（`var.rs:3237`）、`Var::stack`（`var.rs:3187`）、`Var::contiguous`（`var.rs` 内。**`pub(crate)`**。crate 内部からは呼べるが公開面ではない）。
- `jacobian_ops.rs` の private ヘルパー（`checked_numel`:38・`check_on_tape`:61・`FlatElements`:73）の共有方法（`pub(crate)` 化か共通モジュールへの移動）は、実装 issue 1 の判断事項とする。
- `hvp(loss, x, v)` の形: `cg = backward_create_graph(loss, child)` → `g = cg.grad(x)` → 子テープ上で `Σ(g·v)` を作り `child.backward` → `cg.child_var(x)` の勾配。「勾配が `input` に届かない」「`g` が定数」は `hessian` と同じくゼロ扱いとする。子テープは呼び出し後に記録が残る（`hessian` doc と同じ契約）。
- 初期スコープは f32 の `Tape`／`Var`。`VarF64`（`f64_autograd.rs`）は対象外。`Var` は dtype を持たない（dtype 多重化は `docs/autodiff-var-dtype-multiplexing-design.md` の提案段階）ため、vmap の dtype 不一致検査は f32 `Var` では発生しない条件として扱う。

## 5. facade 公開形（推奨 1 案・未承認）

`jacobian`／`hessian`／`gradcheck` と同じ案（facade `Tape` の薄い委譲メソッド）を推奨する。

- `Tape::vjp(&self, output: &Var<'_>, input: &Var<'_>, cotangent: &Tensor<f32>) -> Result<Tensor<f32>, AutodiffError>`
- `Tape::hvp(&self, loss: &Var<'_>, input: &Var<'_>, vector: &Tensor<f32>, child: &Tape) -> Result<Tensor<f32>, AutodiffError>`
- `Tape::vmap<'t, F>(&'t self, input: &Var<'t>, in_dim: usize, f: F) -> Result<Var<'t>, AutodiffError> where F: FnMut(&Var<'t>) -> Result<Var<'t>, AutodiffError>`

選定理由（vmap）: `gradcheck` のように評価ごとに新テープを作る形にせず、**呼び出し側のテープ上のスライスを受けて同テープの `Var` を返す**。stack 後の結果が同一テープ上で微分可能に残り、`hvp`（vmap の出力を損失の一部にする形）や `vjp` と合成できる。**vmap のクロージャ内で勾配を取る合成（`vmap(grad)`＝per-sample gradient）は初期スコープ外**とする。理由: §3 の `grad` 相当（`backward`／`Gradients::get`）は `Tensor` を返し、`Tensor` を葉として再登録すると微分の接続が切れる。`backward_create_graph` の `grad` は子テープ上の `Var` で、`input` と同一テープを要求する vmap のクロージャ契約・§7 の `TapeMismatch` 検査に合わない。微分可能な per-sample gradient には、子テープ上で動くクロージャ型など別の公開形が要り、これは §11 の論点 6 として承認依頼に戻す。値だけが要る per-sample gradient は、呼び出し側が各スライスに対して `backward` を回す明示ループで得られる。複数入力・`out_dim` の有無・`Fn` か `FnMut` かの細部は §11 の論点とし、公開形の承認依頼（§10 の issue 6）で確定する。

- エラー型: 新しい型・variant を足さず、既存 `AutodiffError` の variant（`TapeMismatch`／`GradientTrackingDisabled`／`InvalidArgument`／`Shape(..)`／`Backward`／`DeviceMismatch`）を再利用する。`AutodiffError` は `#[non_exhaustive]`（`crates/autodiff/src/error.rs:19`）で追加は非破壊だが、推奨は再利用。
- `fandhe-ai =0.10.0` に対して追加のみ。既存シグネチャ・意味論・`FitConfig` は不変。
- 公開の手続き: 公開面の追加は `docs/compat-api-scope.md` §5 の手続き（ユーザー承認）が要る。承認までは保留ガード（名前案 `FunctionalTransformsHoldDoctestGuard`。`crates/facade/tests/api_surface.rs` の否定ガードと対）で固定し、承認後に正ガードへ反転する（`Tape::gradcheck` の #2846 → #2847 → #2863 と同じ流れ）。

## 6. 数値一致の判定方式

- 新しい tolerance・baseline を作らない。`vjp`／`hvp`／`vmap` は REQ-2 の統一複合判定（相対誤差 1e-3 未満 または 絶対誤差 1e-5 未満）をそのまま使う。
- 子テープ経路（`hvp`）は 1 階 VJP と bit 同一を主張しない（`create_graph` の既存契約）。`vmap` は同じ Op の合成なので、バッチなし実行との一致を統一複合判定で見る。bit 一致を契約にするかは §11 の論点。
- parity 層の事前登録（`docs/autodiff-jacobian-hessian-gradcheck-decision.md` §3.6 と同型）:

| 層 | 内容 | 実行場所 |
|---|---|---|
| 閉形式 | 小さな既知関数の `vjp`／`hvp`／`vmap` | CI（CPU） |
| 既存 API との突合 | `vjp(y, x, u)` と `Tape::jacobian(y, x)` の転置積、`hvp(loss, x, v)` と `Tape::hessian(loss, x)·v` | CI（CPU） |
| 第三者比較 | PyTorch 2.14.0 `torch.func.vjp`／`hvp`／`vmap` の fixture（Python はオフライン生成で CI 非依存。既存 `crates/autodiff/tests/fixtures/*-pytorch-reference/` と同じ運用で生成条件と sha256 を README に記す）。外れた場合は fandhe-ai の REQ-2 違反ではなく「判定不能」（`.claude/rules/coding-rust.md` 第三者比較対象の項） | CI（CPU） |
| fail-closed | §7 の各条件、非対象 Op、形状不一致 | CI（CPU） |
| 実機 | CUDA／Metal 対 CPU。`#[ignore]`。未実測は `docs/perf/logs/functional-transforms-<実装 issue 番号>/README.md` へ測定コマンドと記入欄を申し送る。実測値は推測で書かない | 実機 |

## 7. ループ版 vmap の fail-closed 条件

根拠は `Var::stack` の現行挙動（`var.rs:3162-3225`）。各検査はテープへノードを足す前に行い、順序を固定する。

| 条件 | 結果 | 備考 |
|---|---|---|
| `input` が別テープ | `Err(TapeMismatch)` | 最初に検査 |
| `in_dim >= rank` | `Err(Shape(AxisOutOfRange))` | |
| 空バッチ（`shape[in_dim] == 0`） | `Err(InvalidArgument)` | `unbind` より前。空の結果形状を推定しない。`Var::stack` も空リストを同 variant で拒否 |
| クロージャが別テープの `Var` を返す | `Err(TapeMismatch)` | `stack` の `check_same_tape` と同じ |
| スライス出力の形状不一致 | `Err(Shape(ShapeMismatch))` | `stack` 前に全件検査 |
| 非 contiguous 出力 | **推奨: contiguous 化して扱う** | `Var::contiguous`（`var.rs`、`pub(crate)`）は既に contiguous なら新規ノードを積まない。`Op::Contiguous` は `supports_create_graph` の対象（`tape.rs:2432` 以降）で、vmap 出力の上で `hvp` が通る。`Var::stack` 自体は `NonContiguousReshape` で拒否するため、vmap 側で `stack` の前に contiguous 化する。公開面は増えない（crate 内部呼び出し） |
| dtype 不一致 | f32 `Var` では発生しない | §4 参照 |

- クロージャが途中で `Err` を返した場合、それまでにクロージャが足したテープ上のノードは残る（値は変えない）。`jacobian_ops` の規律と同じく doc に明記する。
- 前提: クロージャは副作用のない関数。意味論等価は前提を満たす範囲のみ。性能は保証しない（再開条件は §2）。

## 8. double-VJP 法（`jvp`／`jacfwd`）の実現可能性検証計画

- 方法: `y = f(x)` を記録した親テープ上に、勾配追跡ありの葉 `u` を置き、`s = Σ(y·u)` も親テープに記録する（`u` と `y` を同一テープに置く。子テープに `u` を置くと親テープの `y` との積が `TapeMismatch` になり、子テープも非空になって `backward_create_graph` の入口条件〈子テープが空〉を満たせない）。次に空の子テープ `child` に対して `backward_create_graph(s, child)` を呼び、`g = cg.grad(&x)`（`= Jᵀu` を子テープ上の `Var` として得たもの）を作る。`t = Σ(g·v)` を子テープ上で作り、`child.backward(t)` で `cg.child_var(&u)` の勾配を取る。`g` は `u` について線形なので `∂t/∂u = J v` が成り立ち、`f` の二階微分に依らない。
- 対象 Op の範囲: `Op::supports_create_graph() == true` の集合（`tape.rs:2432`。`Leaf`／`Add`／`Mul`／`Relu`／`Exp`／`Tanh`／`Sigmoid`／`Sum`／`Mean`／`Reshape`／`BroadcastTo`／rank 2 `MatMul`／`Transpose`／`Permute`／`Narrow`／`Concat`／`Contiguous`／`Where`、再生可能な `ScalarUnary`／`ScalarBinary`）。
- 非対象 Op の拒否: `backward_create_graph` の既存検査（`create_graph.rs` の `validate_ancestors`:397 が resident／fused・未対応 Op・rank 3 以上の `MatMul` 等を `Err(Backward)` で拒否）をそのまま使い、新しい拒否ロジックを足さない。`Op::Custom` も対象外（`docs/autodiff-custom-function-decision.md` §14）。
- 成立の判定基準:
  1. 代数的根拠: 上記の線形性。
  2. 実測: 結果を `Tape::jacobian(y, x)·v` と統一複合判定で突合する。区分線形の Op（`relu`）を含め、二階微分に依存しないことを確かめる。対象 Op の全区分を網羅する。
  3. 機構面: `cg.child_var(&u)` が `Some`、`cg.grad(&x)` が勾配追跡ありであること。失敗しうるのは数式ではなくミラーの作り方なので判定項目に入れる。
  4. 非対象 Op で型付きエラーになり、子テープが無変更であること。
- 不成立の定義: 上のいずれかが満たせない、または既存契約（`create_graph` の対象 Op・`Op`／`BackendOps`）の変更が要る場合。その場合は実装せず承認依頼に戻す。
- ゲート: 「成立した時点で Tier 2 へ移す」（spec `:236`）。`jvp`／`jacfwd` の実装 issue（§10 の 10）は、検証 issue（同 9）の成立と、spec 側の追記（ユーザー側の作業。実装 Agent は `docs/spec/` を編集しない）の両方に依存させ、自動では進めない。

## 9. 依存追加・新規 `unsafe`・`Op`／`BackendOps` 拡張の要否

見込みはいずれも **不要**。`vjp`／`hvp`／`vmap` は既存 Op（`mul`／`sum`／`unbind`／`stack`／`contiguous`）と既存の `backward`／`backward_create_graph` の合成のみで、CUDA／Metal も既存 Op で到達するため `Unsupported` フォールバックの追加対象もない。Cargo 依存は増えない。

個別の承認事項として別扱いにするもの:

- `supports_create_graph` の対象 Op を広げること（`hvp`／double-VJP の適用範囲拡大）。
- 新しい判定契約（§6 の統一複合判定で足りない場合）。

## 10. 2 時間粒度の実装 issue 分解案

各 issue は 1 関心事・単独 PR。1〜3 は `functional_ops.rs` を共有するため直列化する（またはファイルを分ける）。

| # | タイトル案 | 依存 | 受け入れ条件の要点 |
|---|---|---|---|
| 1 | `feat(autodiff): vjp の内部実装` | なし | `functional_ops.rs` 新設・入口検査・閉形式テスト・保留ガード固定。ヘルパー共有方法の確定 |
| 2 | `feat(autodiff): hvp の内部実装` | 1 | `Tape::hessian·v` と統一複合判定で一致。非対象 Op の型付き拒否、失敗時に子テープ無変更 |
| 3 | `feat(autodiff): ループ版 vmap の内部実装` | 1 | §7 の fail-closed 全件。バッチなし実行との一致。途中 `Err` 時のノードの扱いを doc に明記 |
| 4 | `test(autodiff): vjp／hvp／vmap の PyTorch fixture と parity テスト` | 1〜3 | §6 の層。生成条件と sha256 を README に記す。実 `CpuBackendOps` が要るものは `crates/facade/tests/` |
| 5 | `test(autodiff): vmap と vjp／hvp の合成の検証` | 2, 3 | vmap 出力を損失の一部にした `hvp`／`vjp`。同一テープ上での微分可能性。`vmap(grad)`（per-sample gradient）は §5 のとおり初期スコープ外 |
| 6 | `docs(facade): vjp／hvp／vmap の公開形の承認依頼` | 1〜3 | `docs/compat-api-scope.md` §5.1 への行追加と §11 の論点の確定依頼。**承認は実装 Agent が代行しない** |
| 7 | `feat(facade): vjp／hvp を記録の形で公開` | 6 の承認 | 保留ガードを正ガードへ反転。薄い委譲の固定 |
| 8 | `feat(facade): vmap を記録の形で公開` | 6 の承認 | 同上 |
| 9 | `test(autodiff): double-VJP 法の実現可能性の検証` | 2 | §8 の判定基準。結果を本記録へ追記 |
| 10 | `feat(autodiff): jvp／jacfwd`（条件付き） | 9 の成立と spec 追記 | 不成立なら起票しない |
| 11 | `test(backend): CUDA／Metal 実機 parity の #[ignore] テストと perf/logs 申し送り` | 4 | 実測値は推測で書かない |

件数の差: 提案文書 §3.1 の概算（(a) 6〜10、(d-1) 3〜5、(b) 検証 2〜3）より減るのは、`grad`／`jacrev`／`hessian` を新設しない（§3）ため。

## 11. 承認依頼に戻す論点（実装せず止める）

1. facade シグネチャの細部（複数入力、`out_dim`、`Fn` か `FnMut` か、`hvp` の `child` 引数の要否）。
2. `vmap` 結果とバッチなし実行の bit 一致を契約にするか（推奨: しない。統一複合判定）。
3. double-VJP の検証で差が統一複合判定を外れた場合の扱い（新しい判定契約は別途承認）。
4. f16 等の低精度 forward を対象にするか（推奨: 初期スコープ外）。
5. `supports_create_graph` の対象 Op 拡張。
6. 微分可能な per-sample gradient（`vmap(grad)`）の公開形。子テープ上の `Var` を扱うクロージャ型など、`input` と同一テープを要求する §5 の契約とは別形が要る。

## 12. スコープ外・申し送り

- スコープ外: ネイティブ forward-mode、バッチ規則型 vmap、`vmap(grad)`（per-sample gradient。§5・§11-6）、`VarF64`、GPU 専用カーネル、実装そのもの、実装 issue の起票、`docs/compat-api-scope.md` §5.1 の行追加（issue 6 の仕事）。
- 要対応事項: facade 公開形の承認（ユーザー）、`jvp`／`jacfwd` を Tier 2 へ移す spec 追記（ユーザー側）。

## 13. セキュリティ観点

- 設計は fail-closed に寄せる: 入口検査をノード追加より前に行う、空バッチ・形状不一致・非対象 Op を型付きエラーで拒否する、結果領域の要素数は検査付き乗算で求める。
- 新規 `unsafe`・依存追加は不要。秘密情報は含まない。承認は §1 の範囲を超えて記述しない。

## 14. 出典

- spec: `docs/spec/04-requirements.md:236`・`:457`（読むのみ）
- `docs/autodiff-forward-mode-vmap-spec-proposal.md`、`docs/autodiff-jacobian-hessian-gradcheck-decision.md`（§3.6・§11）、`docs/autodiff-higher-order-grad-decision.md`（§8・§17）、`docs/compat-api-scope.md` §5、`docs/autodiff-var-dtype-multiplexing-design.md`、`docs/autodiff-custom-function-decision.md` §14
- ソース: `crates/autodiff/src/{backward.rs,create_graph.rs,jacobian_ops.rs,tape.rs,var.rs,shape_view_ops.rs,error.rs}`、`crates/facade/src/lib.rs`
