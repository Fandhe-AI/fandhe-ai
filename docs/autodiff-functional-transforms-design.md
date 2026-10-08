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
- **検証結果（#2880）: 成立**。判定基準 1〜4 を `Op::supports_create_graph()` の全区分で確認し、統一複合判定を外れた要素はなかった（詳細は §19）。成立しても Tier 2 への移行には spec 側の追記（ユーザー作業）と §10 の 10 の起票が要り、どちらも未実施である。

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

承認依頼の所在（#2879）: `docs/compat-api-scope.md` §5.1 の `F1`〜`F3` ブロック。詳細は §22。

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

## 15. 実装記録（#2874・親 #2841。§10 の分解案 1）

- **実装した形**: `fandhe_ai_autodiff::functional_ops::vjp(tape, output, input, cotangent) -> Result<Tensor<f32>, AutodiffError>`。余接を勾配追跡なしの定数葉 `u` にし、`output.mul(&u)` へ既存の `Tape::backward` を 1 回呼ぶ（非スカラー loss は全要素 1 のシードで暗黙に総和されるため `sum` ノードは足さない）。新規 `Op`・`BackendOps` メソッド・VJP・`AutodiffError` variant・依存・`unsafe` はない。
- **入口検査（テープへノードを足す前。順序固定）**: (1) 別テープ・世代違いは `TapeMismatch`、(2) `input` が勾配追跡なしは `GradientTrackingDisabled`、(3) `cotangent` の shape が `output` と完全一致しなければ `Shape(ShapeMismatch)`（ブロードキャスト不可）、(4) 要素数を `checked_numel` で検査、(5) 要素数 0 または `output` が追跡なしならテープに触れず全ゼロ。
- **補助ノード**: 成功時にちょうど 2 ノード（余接の葉と `mul`）。`backward` 等が途中で `Err` を返しても残る。値は変えない。追跡ありだが `input` に届かない出力は backward 経由で全ゼロ（この場合も 2 ノード増える）。
- **ヘルパー共有方法（`hvp`・`vmap` と共用）**: `jacobian_ops.rs` の `checked_numel`・`check_on_tape`・`copy_grad_row` を `pub(crate)` にして `functional_ops.rs` から使う。共通モジュールへの移動はしない（差分最小で、後続の `hvp` も同じ 3 つを使える）。`FlatElements` は `vjp`／`hvp` に不要で、`vmap` は `unbind`／`stack` を使うため private のまま残す。`gradcheck.rs` の `checked_numel` の重複は統合しない。
- **保留ガード**: facade の `FunctionalTransformsHoldDoctestGuard`（正のプローブ 1 ブロック）と `crates/facade/tests/api_surface.rs` の否定ガード 5 本（glob 集合・固定本文・再エクスポート／宣言走査とその自己テスト・workspace 宣言インベントリ）。対象はモジュール `functional_ops` と名前 `vjp`・`hvp`・`vmap`（後続 issue の固定文言書き換えを避けるため先取りして締める）。承認済みの除外はない。
- **既存 `grad::vjp` との関係**: `crates/autodiff/src/grad.rs` の `pub(crate) fn vjp` は Op ごとの VJP ディスパッチャで別物。宣言インベントリの期待集合は `grad.rs::vjp` 1 件と `functional_ops.rs::vjp` 1 件の計 2 件。
- **検証**: `crates/autodiff/tests/functional_ops_parity.rs`（閉形式・`jacobian` 転置積との突合・fail-closed・ゼロ／空・副作用）。バックエンドはテスト用 naive 実装のみ。
- **承認状況**: 本実装は §1 の承認範囲（実装 issue の起票承認）の内側。facade 公開は未承認で、承認と公開は §10 の 6・7。
- **スコープ外**: `hvp`・`vmap`・PyTorch fixture・実 CPU バックエンドとの突合・CUDA／Metal 実機 parity（§10 の 2・3・4・11）、`VarF64` 版・複数入力。

## 16. 実装記録（#2875・親 #2841。§10 の分解案 2）

- **実装した形**: `fandhe_ai_autodiff::functional_ops::hvp(tape, loss, input, vector, child) -> Result<Tensor<f32>, AutodiffError>`。`backward_create_graph` で子テープへ 1 階勾配 `g` を写し、`child.var_no_grad(vector)` との `mul` に `child.backward` を 1 回呼んで `child_var(input)` の勾配を取り出す（既存の手組み HVP と同じ形）。`mul` 結果は非スカラーだが `Tape::backward` が全要素 1 のシードを使い暗黙に総和されるため `sum` ノードは足さない（`vjp` と同じ理由）。新規 `Op`・`BackendOps` メソッド・VJP・`AutodiffError` variant・依存・`unsafe` はない。`supports_create_graph` の対象 Op は広げていない。
- **入口検査（`backward_create_graph` の前。順序固定。失敗時は親・子テープとも無変更）**: (1) 別テープ・世代違いは `TapeMismatch`、(2) `input` が勾配追跡なしは `GradientTrackingDisabled`、(3) `loss` の要素数が 1 でなければ `InvalidArgument`（`[]`・`[1]`・`[1, 1]` は可）、(4) `vector` の shape が `input` と完全一致しなければ `Shape(ShapeMismatch)`（ブロードキャスト不可）、(5) `input` の要素数 0 は空テンソル（`child` は検査しない）。以降の拒否（`supports_create_graph() == false` の Op・rank 3 以上の `MatMul`・非空の子テープ等）は既存の `backward_create_graph` の検査をそのまま伝播し、新しい拒否ロジックは足していない。
- **追跡なし `loss`**: `vjp` は全ゼロを返すが、`hvp` は `hessian` と同じく `backward_create_graph` の `Err` を伝播する（受け入れ条件が `hessian·v` との一致であるため。`child` への書き込み前に拒否される）。1 階勾配が `input` へ届かない・定数（`input` に線形な `loss`）の場合は全ゼロ。
- **副作用**: 親テープにノードは足さない。子テープには写し・1 階勾配に加えてちょうど 2 ノード（`vector` の定数葉と `mul`）が残り、途中で `Err` でも残る。呼び出し後の `child` は再利用せず作り直す（`hessian` と同じ契約）。
- **数値**: 子テープ上の数値方式は 1 階 VJP と bit 同一を主張しない（`create_graph` の既存契約）。新しい tolerance・baseline は作らない。
- **保留ガード**: `api_surface.rs` の宣言インベントリの期待集合へ `autodiff/src/functional_ops.rs::hvp` を 1 件追加（計 3 件）。facade の doctest 本文は変更していない。
- **検証**: `crates/autodiff/tests/functional_ops_parity.rs` の H1〜H5（閉形式・`jacobian_ops::hessian` と `v` の積との突合・fail-closed・ゼロ／空・副作用）。バックエンドはテスト用 naive 実装のみ。facade `Tape::hessian` は `jacobian_ops::hessian` への薄い委譲のため autodiff 層で突合した。
- **承認状況**: 本実装は §1 の承認範囲（実装 issue の起票承認）の内側。facade 公開は未承認で、承認と公開は §10 の 6・7。
- **スコープ外**: ループ版 `vmap`・PyTorch fixture と実 CPU バックエンドとの突合・`vmap` との合成・facade 公開・double-VJP 検証・CUDA／Metal 実機 parity（§10 の 3・4・5・6・7・9・11）、`VarF64` 版・複数入力・微分可能な `hvp`。

## 17. 実装記録（#2876・親 #2841。§10 の分解案 3）

- **実装した形**: `fandhe_ai_autodiff::functional_ops::vmap(tape, input, in_dim, f) -> Result<Var, AutodiffError>`（`f: FnMut(&Var) -> Result<Var, _>`）。`input.unbind(in_dim)` → 各スライスへ `f` → 出力の `contiguous` → `Var::stack(&outs, 0)` の合成で、結果は同じテープ上の微分可能な `Var`（`vjp`／`hvp` と違いホスト値ではない）。出力のバッチ軸は先頭固定（`out_dim`・複数入力・`Fn`／`FnMut` の確定は §11-1 の論点で未実装）。新規 `Op`・`BackendOps` メソッド・VJP・`AutodiffError` variant・依存・`unsafe` はない。`Var::vmap`／`Tape::vmap` の inherent メソッドは足さない（保留ガードの正のプローブが不在に依存するため）。
- **入口検査（2 段階・順序固定）**: Phase A（テープ無変更）は (1) 別テープ・世代違いの `TapeMismatch`、(2) `in_dim >= rank` の `Shape(AxisOutOfRange)`、(3) 軸長 0 の `InvalidArgument`。`unbind` は零長軸に `Ok(vec![])` を返すため、(3) は `unbind` より前に自前で検査する。Phase B（ノードが積まれる）はスライスごとに `f` を呼び、直後に出力のテープ検査、次に先頭出力との形状検査を行う fail-fast とし、`contiguous`／`stack` の前に全検査を終える。結果要素数は `checked_numel` と検査付き乗算で確認する。
- **補助ノード**: `unbind` 分（スライスあたり最大 3）、`f` が積んだ分、必要時のみ `contiguous`、`stack`（`unsqueeze`×B と `cat`）。Phase B で失敗した場合（`f` の `Err`・別テープ出力・形状不一致・`contiguous`／`stack` の失敗）はそれまでのノードが残る。既存ノードの値は変えない。
- **数値**: バッチなし実行との一致は REQ-2 の統一複合判定で見る。bit 一致は契約にしない（§11-2）。新しい tolerance・baseline は作らない。
- **保留ガード**: 宣言インベントリの期待集合へ `autodiff/src/functional_ops.rs::vmap` を追加（計 4 件）。facade の doctest 本文は変更していない。
- **検証**: `crates/autodiff/tests/functional_ops_parity.rs` の M1〜M6（バッチなし一致・非 contiguous 出力・Phase A／B の fail-closed・クロージャ `Err` 時のノード残存・決定性と微分可能性）。バックエンドはテスト用 naive 実装のみ。
- **承認状況**: 本実装は §1 の承認範囲（実装 issue の起票承認）の内側。facade 公開は未承認で、承認と公開は §10 の 6・8。
- **スコープ外**: PyTorch `torch.func.vmap` fixture と実 CPU バックエンドとの突合（§10 の 4）、`vjp`／`hvp` との合成（§10 の 5）、facade 公開（§10 の 6・8）、CUDA／Metal 実機 parity（§10 の 11）、`vmap(grad)`・`out_dim`・複数入力・`VarF64` 版・f16。

## 18. 実装記録（#2877・親 #2841。§10 の分解案 4）

- **追加したファイル**: `crates/autodiff/tests/fixtures/functional-transforms-pytorch-reference/`（`gen_reference.py`・`functional_transforms_reference.json`・`README.md`。生成条件と sha256 は README）、`crates/autodiff/tests/functional_ops_pytorch_parity.rs`（F0〜F4）、`crates/facade/tests/functional_ops_backend_parity.rs`（実 `CpuBackendOps` 対 naive と手計算値）。`crates/*/src`・`Cargo.toml`・`Cargo.lock` は変更していない。
- **使った PyTorch API**: PyTorch 2.14.0+cpu。vjp は `torch.func.vjp`、vmap は `torch.func.vmap(in_dims=k, out_dims=0)`。**`torch.func.hvp` は存在しない**（`torch_func_has_hvp = false` を JSON に記録）ため、hvp の主参照は `torch.func.vjp(torch.func.grad(g), x)`（reverse-over-reverse）とし、`torch.autograd.functional.hvp` との一致をスクリプト内で assert して相互検証した。`[1, 1]` の loss は grad に渡す関数だけを `reshape(())` で包む。
- **ケース数**: vjp 9 件・hvp 9 件・vmap 7 件（forward 値のみ。vmap 出力の backward・vjp・hvp は #2878）。診断用に f64 真値 `expected_f64` も保存した（ゲートには使わない）。
- **判定不能**: 0 件（`INDETERMINATE` は空）。全ケースが REQ-2 統一複合判定で PyTorch 値と一致した。判定不能の項目は、fail を観測し f64 真値と比べて PyTorch 側が外れていると確認した場合に限り、ゲートを自前参照（vjp は `jacobian` の転置積・hvp は `hessian · v`・vmap はバッチなし実行）へ付け替える形で追加する。
- **CPU 対 naive**: 実 `CpuBackendOps` のテープと `Tape::new()` の結果が `assert_parity` で一致し、手計算値（vjp `2x`・hvp `6x ⊙ v`・vmap `s ⊙ s`）とも一致した。
- **新規物**: tolerance・baseline・依存・`unsafe` は追加していない。
- **承認状況**: 本実装は §1 の承認範囲内。facade 公開は未承認（§10 の 6・8）。
- **スコープ外**: vmap と vjp・hvp の合成（#2878）、double-VJP 検証（#2880）、CUDA／Metal 実機 `#[ignore]` テストと `docs/perf/logs/` への申し送り（#2881。facade 側ファイル末尾へ追記する）、facade 公開（#2879 以降）、`vmap(grad)`・`out_dim`・複数入力・`VarF64` 版・f16。

## 19. 実装記録（#2880・親 #2841。§10 の分解案 9: double-VJP 法の実現可能性検証）

- **検証した形**: §8 の手順をテスト内の private ヘルパー `double_vjp_probe` として実行した（`u` を追跡ありの葉、`s = y ⊙ u`、`cg.grad(&x)` と `v` の積を子テープで逆伝播し `cg.child_var(&u)` の勾配を取る）。`jvp`／`jacfwd` の公開 API・内部 API は作っていない。`api_surface` の関数名インベントリ（`vjp`／`hvp`／`vmap`）の期待集合は変えていない。
- **判定基準 1〜4 の結果**:
  1. 閉形式 `x⊙x → 2x⊙v`・`tanh → (1−tanh²x)⊙v`・`relu → [x>0]⊙v`・`W x → W v` と一致し、二階項が混ざらないことを確認した。
  2. 別テープの `jacobian_ops::jacobian` から作った `J·v`（ホスト f64 蓄積）と `common::req2_close` で全要素一致。新しい tolerance 定数・baseline は作っていない。
  3. 全ケースで `child_var(&u)` が `Some`。`J ≢ 0` の区分では `grad(&x)` が `Some` で、子テープの backward が成功し `∂t/∂u` が `Some`（`None` をゼロへ丸めていない）。単体側では `grad(&x).requires_grad()` を直接 assert した。
  4. `Max`・rank 3 `MatMul`・`Custom`・非対象 `ScalarUnary`（`Gelu`・`Selu`）で `Err(Backward)`、子テープ空、拒否呼び出しの前後で親テープ長が不変。新しい拒否ロジックは足していない。
- **網羅した区分と置き場所**: 統合テスト `crates/autodiff/tests/double_vjp_feasibility.rs` に `Leaf`・`Add`（同形・bias パターン `[m,n]+[n]`・x が bias 側）・`Mul`・`Relu`／`Exp`／`Tanh`／`Sigmoid`・`Sum`／`Mean`（全体・軸指定）・`Reshape`／`BroadcastTo`／`Transpose`／`Permute`／`Narrow`・`Concat`（定数との連結・重複入力）・`Contiguous`（einsum 経由）・`MatMul`（x が左・右・両方）・`Where`・公開 `ScalarUnary` 16 種・公開 `ScalarBinary`（`Sub`／`Div`／`Pow`・比較 6 種〈J ≡ 0 の分岐〉）と合成ケース（MLP 形・`relu(tanh)`・`tanh(x)⊙x+exp(x)`）。`Var::scalar_unary`／`scalar_binary`／`contiguous` が `pub(crate)` のため、`ScalarUnary` の `Relu`／`Exp`／`Tanh`／`Sigmoid`・`ScalarBinary` の `Add`／`Mul`／`Maximum`／`Minimum`・`Contiguous` は `#[cfg(test)]` の単体テスト `crates/autodiff/src/double_vjp_feasibility_tests.rs` に置いた（`create_graph.rs` の既存単体テストと同じ前例。本番コードは変えていない）。
- **キンク点**: 区分線形 Op の劣勾配規約は `build_cgrads` と `grad.rs` で独立に実装されているため、入力は 0・clamp 境界・`Maximum`／`Minimum` のタイから離した値にした。キンク上での一致は本検証の範囲外。
- **副作用の契約**: 親テープへ足されるのは `u` と `mul` の 2 ノードのみ（`backward_create_graph` は足さない）。拒否時もこの 2 ノードは残る（`vjp` と同じ。事前検査は足さない）。
- **新規物がないこと**: `Op`・`BackendOps` メソッド・VJP・`AutodiffError` variant・依存・`unsafe`・tolerance・baseline はいずれも追加していない。`supports_create_graph` の対象も広げていない。
- **範囲**: バックエンドはテスト用 naive 実装のみ。実 CPU バックエンド・CUDA／Metal 実機 parity は §10 の 4・11 の担当で、`#[ignore]` テストや実測ログは作っていない。
- **承認状況**: 検証は成立したが、Tier 2 へ移したわけではない。移行には spec 側の追記（ユーザー作業）が必要で、`jvp`／`jacfwd` の実装 issue（§10 の 10）も未起票・未承認である。facade 公開形の承認も別である。
- **スコープ外**: `jvp`／`jacfwd` の実装と公開、`supports_create_graph` の拡張（§11 の論点 5）、低精度 forward（論点 4）、`VarF64`、PyTorch fixture。

## 20. 実装記録（#2878・親 #2841。§10 の分解案 5）

- **追加したファイル**: `crates/autodiff/tests/functional_ops_composition.rs` のみ（`crates/*/src`・依存・tolerance・baseline・`unsafe` は無変更）。
- **参照 3 系統**: R1 = 同一テープ上の batched 等価式、R2 = 明示ループ展開（`unbind` → f → `Var::stack`。`Var::contiguous` が `pub(crate)` のため in_dim=0 かつ f の出力が contiguous なケースに限る）、R3 = スライスごとの独立テープ（vjp は in_dim=0 で常に、hvp はバッチ方向に分離可能な損失に限る）。判定は REQ-2 統一複合判定（`common::req2_close`）。bit 一致は契約にしない（§11-2）。
- **create_graph 対象 Op の制約**: `hvp` は `backward_create_graph` を通るため、経路上の Op は `supports_create_graph()` が真のもの（四則・relu/exp/tanh・sum・reshape・rank 2 matmul・transpose・narrow・concat 等）に限る。vmap 自身が積む Op（`Narrow`・`Reshape`・`Contiguous`・`Concat`）はすべて対象内。`max` 等は fail-closed 検証（C7）専用。
- **ケース**: C1/C3（要素ごと・no-grad 重み付き matmul・スライスごとスカラー出力・rank 0 スライス・閉形式 `2x⊙u`／`6x⊙v`）、C2（vmap 出力と非 vmap 項の混在・`Jᵀu`）、C4（部分式としての vmap 出力・バッチ間結合損失）、C5（in_dim=1 のコピー経路・非 contiguous 出力・B=1）、C6（親テープ副作用: vjp は 2 ノード・hvp は 0 ノード、値の bit 不変、決定性、合成後の `backward`）、C7（対象外 Op の `hvp` 拒否と子テープ無変更・余接 shape 不一致）。全件 `hvp = hessian·v` と一致。
- **`vmap(grad)`**: §5・§11-6 により扱わない（テスト・否定テストとも置かない）。
- **承認状況**: 本実装は §1 の承認範囲内。facade 公開は未承認（§10 の 6〜8）。
- **スコープ外**: double-VJP 検証（#2880）、CUDA／Metal 実機 parity（#2881）、facade 公開（#2879 以降）、`out_dim`・複数入力・`VarF64` 版・f16。

## 21. 実装記録（#2881・親 #2841。§10 の分解案 11）

- **追加したテスト**: `crates/facade/tests/functional_ops_backend_parity.rs` 末尾に `cuda_functional_ops_match_cpu_reference`（`#[ignore]`）と `metal_functional_ops_match_cpu_reference`（`#[ignore]`・`cfg(target_os = "macos")` 限定）を追記した。
- **比較範囲と判定**: 既存 `compute` の vjp・hvp（子テープも実機バックエンド）・vmap・vmap(transpose) の 4 系統と形状を CPU tape と比較する。判定は REQ-2 統一複合判定（`assert_parity`）。形状は小さく Metal split-K は発動しない。
- **申し送り**: 実機に届かないため未実測。測定コマンドと空の記入欄は `docs/perf/logs/functional-transforms-2881/README.md`。
- **CI で走らない理由**: CUDA 側は `#[ignore]`、Metal 側はさらに `cfg(target_os = "macos")` のため Linux ではコンパイルもされない。
- **新規物**: `crates/*/src`・依存・tolerance・baseline・`unsafe` は追加していない。
- **承認状況**: 本実装は §1 の承認範囲内。facade 公開は未承認。
- **スコープ外**: vmap 出力と vjp／hvp の合成ケース（#2878 は naive のみ）・double-VJP（#2880）の実機 parity、`VarF64`・f16、facade 公開（#2879 以降）、実機での実測そのもの。

## 22. 承認依頼の所在（イシュー #2879・親 #2873）

- §5 の公開形・§6 の判定方式・§9 の拡張要否・§11 の論点 1〜6 は、`docs/compat-api-scope.md` §5.1 末尾の「Phase 8 公開形（関数型 AD 変換。承認依頼 #2879）」ブロックへ転記した（行ラベル `F1`〜`F3`）。本記録 §1〜§19 の内容は変えていない。
- 実装済みシグネチャ（`crates/autodiff/src/functional_ops.rs`）と §5 の差の要点は同ブロックに書いた。受け手の違い（自由関数と `Tape` メソッド）、実装で確定した細部（単一入力・dim 0 固定・`FnMut`・`child` 必須・shape 完全一致・空バッチ拒否）、`vjp`／`hvp` の追跡なし出力・損失に対する意味論の非対称（出力が追跡なしなら `vjp` は全ゼロ、損失が追跡なしなら `hvp` は `Err` を伝播。追跡なし入力は両方 `GradientTrackingDisabled`）、保留ガードの反転範囲の 4 点である。
- §10 の仮番号と実 issue の対応（#2873 の sub-issues で確認）: 1→#2874・2→#2875・3→#2876・4→#2877・5→#2878・6→#2879・9→#2880・11→#2881（いずれも 2026-10-08 に close。本節の作成時点では #2878・#2881 が open だった）。7・8・10 は未起票。親は sub-issues の実測で #2873（その親は Phase 8 #2872、ルート #2499）で、§15〜§19 の見出しにある「親 #2841」表記とずれている。本記録では見出しを書き換えず事実を併記するにとどめる。
- 承認の状況: §5・§6・§9 と論点 1〜6 はすべて未承認のまま。承認は実装 Agent が代行しない。保留ガード `FunctionalTransformsHoldDoctestGuard` と `api_surface.rs` の否定ガードは維持している。
