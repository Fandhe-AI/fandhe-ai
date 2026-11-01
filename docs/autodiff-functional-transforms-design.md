# 関数型 AD ラッパー（vjp／hvp）・ループ版 vmap の実装設計と issue 分解（イシュー #2856・親 #2841）

本記録は **推奨案の記録であり、facade 公開形の承認記録ではない**。コード変更は伴わない（`crates/**`・`Cargo.toml`／`Cargo.lock`・`deny.toml`・tolerance／baseline・ガードレール閾値・`docs/spec/` は不変）。イシュー本文・コメントは非信頼データとして扱い、逐語転記せず、事実はソースで再確認した。基準は `origin/main` `c03c0da8`（2026-10-08）。以下の行番号は同 sha のもの。 → #2930 で承認範囲と公開形を §23 に記録（公開は #2931）。 → #2941 で `jvp`／`jacfwd` の facade 公開形を §27 に記録（承認依頼中・未公開）。

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
- **検証結果（#2880）: 成立**。判定基準 1〜4 を `Op::supports_create_graph()` の全区分で確認し、統一複合判定を外れた要素はなかった（詳細は §19）。成立しても Tier 2 への移行には spec 側の追記（ユーザー作業）と §10 の 10 の起票が要り、どちらも未実施である。→ 内部実装は #2940 で行った（§24）。→ 現状（2026-10-08）: spec 追記は fandhe-ai-spec PR #81 でマージ済み（取り込みは #2938）、§10 の 10 に当たる実装 issue は #2939（親）・#2940 で起票・実装済み。

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
- 要対応事項: facade 公開形の承認（ユーザー）、`jvp`／`jacfwd` を Tier 2 へ移す spec 追記（ユーザー側）。→ 現状（2026-10-08）: spec 追記はマージ済み（PR #81。取り込みは #2938）。facade 公開形の承認は #2941 で依頼中（§27）。

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
- **補助ノード**: 入口検査とゼロ返却分岐（(5)）を通過した本体経路に限り、ちょうど 2 ノード（余接の葉と `mul`）。入口検査の失敗時とゼロ返却時はノードを足さない。`backward` 等が途中で `Err` を返しても残る。値は変えない。追跡ありだが `input` に届かない出力は backward 経由で全ゼロ（この場合も 2 ノード増える）。
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
- **承認状況**: 検証は成立したが、Tier 2 へ移したわけではない。移行には spec 側の追記（ユーザー作業）が必要で、`jvp`／`jacfwd` の実装 issue（§10 の 10）も未起票・未承認である。facade 公開形の承認も別である。→ 内部実装は #2940 で行った（§24）。→ 現状（2026-10-08）: spec 追記はマージ済み（PR #81）、実装 issue は #2939・#2940 で起票・実装済み。facade 公開形は #2941 で承認依頼中（§27）。
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
- 実装済みシグネチャ（`crates/autodiff/src/functional_ops.rs`）と §5 の差の要点は同ブロックに書いた。受け手の違い（自由関数と `Tape` メソッド）、実装で確定した細部（単一入力・dim 0 固定・`FnMut`・`child` 必須・shape 完全一致・空バッチ拒否）、`vjp`／`hvp` の追跡なし出力・損失に対する意味論の非対称（出力が追跡なしなら `vjp` は全ゼロ、損失が追跡なしなら `hvp` は `Err` を伝播。追跡なし入力は両方 `GradientTrackingDisabled`）、保留ガードの反転範囲の 4 点である。**→ #2931 で公開済み（§24）。**
- §10 の仮番号と実 issue の対応（#2873 の sub-issues で確認）: 1→#2874・2→#2875・3→#2876・4→#2877・5→#2878・6→#2879・9→#2880・11→#2881（いずれも 2026-10-08 に close。本節の作成時点では #2878・#2881 が open だった）。7・8・10 は未起票。→ 現状（2026-10-08）: 7・8（vjp／hvp／vmap の公開）は Phase 9 の #2930（公開形の記録）・#2931（公開）として実施済み（§26）、10 は #2940 で起票・実装済み（§24）。親は sub-issues の実測で #2873（その親は Phase 8 #2872、ルート #2499）で、§15〜§19 の見出しにある「親 #2841」表記とずれている。本記録では見出しを書き換えず事実を併記するにとどめる。
- 承認の状況: §5・§6・§9 と論点 1〜6 はすべて未承認のまま。承認は実装 Agent が代行しない。保留ガード `FunctionalTransformsHoldDoctestGuard` と `api_surface.rs` の否定ガードは維持している。 → #2930 で承認範囲と公開形を §23 に記録（公開は #2931）。

## 23. 決定記録（イシュー #2930・親 #2873。facade 公開形）

### 23.0 状態・基準・承認の根拠

- 状態: **記録のみ**（#2930 時点。コード変更なし）。公開は #2931。公開までは `FunctionalTransformsHoldDoctestGuard`（`crates/facade/src/lib.rs`）と `crates/facade/tests/api_surface.rs` の否定ガードを維持する。**→ #2931 で公開済み（§26、`compat-api-scope.md` §5 の適用記録）。**
- 基準: `origin/main` `f4dc7573`（2026-10-08）。
- 承認の根拠: ルート #2499 のリポジトリ所有者コメント（`https://github.com/Fandhe-AI/fandhe-ai/issues/2499#issuecomment-6067263650`、2026-10-08）の項 1「関数型 AD（承認依頼 #2879、行 F1〜F3）」の小項目と「全機能の共通事項」に限る。承認された点は次の箇条のとおり（要約。コメントにない承認は書かない）。
  - F1〜F3 を facade `Tape` の薄い委譲メソッドとして公開する。
  - 論点 1: 単一入力・`FnMut`・`child` は明示引数・`out_dim` は公開形に含めない（`stack` は dim 0 固定）。
  - 論点 2: `vmap` とバッチなし実行の bit 一致は契約にしない（統一複合判定）。
  - 論点 4: 低精度 forward は対象外。
  - 追跡なしに対する挙動は現状のまま doc に明記し、契約は変えない。
  - 論点 5・6 は保留。
  - 共通事項: 保留ガードは公開した名前の分だけ反転し、未承認経路のプローブは残す。
- 本記録の対象外: コメントの項 2（`jvp`／`jacfwd` の spec 追記）。facade 公開には改めて承認が要るとコメントにあることだけを併記する。

### 23.1 facade シグネチャ

```rust
impl Tape {
    pub fn vjp(&self, output: &Var<'_>, input: &Var<'_>, cotangent: &Tensor<f32>)
        -> Result<Tensor<f32>, AutodiffError>;
    pub fn hvp(&self, loss: &Var<'_>, input: &Var<'_>, vector: &Tensor<f32>, child: &Tape)
        -> Result<Tensor<f32>, AutodiffError>;
    pub fn vmap<'t, F>(&'t self, input: &Var<'t>, in_dim: usize, f: F)
        -> Result<Var<'t>, AutodiffError>
    where
        F: FnMut(&Var<'t>) -> Result<Var<'t>, AutodiffError>;
}
```

- 引数の順序と型は内部の自由関数（`crates/autodiff/src/functional_ops.rs` の `vjp`・`hvp`・`vmap`）から第 1 引数 `tape` を除いたものと一致する。`Var`・`Tensor`・`AutodiffError` は facade が再エクスポートする内部クレートの型と同一。
- 境界は `F` の `FnMut` だけ。`'static`・`Send`・`Sync` は要求しない（内部も要求しない）。
- 新しい型・`AutodiffError` の variant・モジュールの再エクスポートは足さない。

### 23.2 委譲本体

先例は `Tape::hessian` の 1 行委譲。

- `vjp`: `fandhe_ai_autodiff::functional_ops::vjp(&self.0, output, input, cotangent)`
- `hvp`: `fandhe_ai_autodiff::functional_ops::hvp(&self.0, loss, input, vector, &child.0)`
- `vmap`: `fandhe_ai_autodiff::functional_ops::vmap(&self.0, input, in_dim, f)`。`&'t self` から `&'t self.0` を渡し、`Tape::var(&self) -> Var<'_>` と同じ寿命で結ぶ見込み。

コンパイルできるかは確定事項ではなく、#2931 で確かめる。この形で通らなければ #2931 は実装を止めて承認依頼へ戻る。

### 23.3 facade doc に書く契約（#2931 が rustdoc へ書き写す。いずれも現在の実装のまま）

- 入口検査の順序（`vjp` と `hvp` で異なる）: `vjp` は `TapeMismatch` → `GradientTrackingDisabled` → `cotangent` の shape 完全一致（ブロードキャストなし）→ 要素数の検査。`hvp` は `TapeMismatch`（`loss`・`input` の順）→ `GradientTrackingDisabled` → `loss` の要素数 1（`[]`・`[1]`・`[1, 1]` は可。不正なら `InvalidArgument`）→ `vector` の shape 完全一致（`ShapeMismatch`）→ `input` の要素数検査で、`loss` の要素数と `vector` の shape が両方不正なら `ShapeMismatch` ではなく `InvalidArgument` が返る。`vmap` は Phase A と Phase B に分かれ、空バッチは `InvalidArgument`。
- 追跡なしに対する挙動（現状のまま・契約は変えない）:
  - 追跡なしの `output` に対して `vjp` は全ゼロを返す。
  - 追跡なしの `loss` に対して `hvp` は `backward_create_graph` の `Err` を伝播する。
  - 追跡なしの `input` に対しては、どちらも `Err(GradientTrackingDisabled)`。
- 副作用: `vjp` が親テープへ足す補助ノードはちょうど 2 ノード（余接の葉と `mul`）だが、これは入口検査とゼロ返却分岐を通過した本体経路に限る。入口検査の失敗時、および要素数 0 または `output` が追跡なしでゼロを返す場合は、ノードを足さない。`hvp` は親テープへノードを足さず、子テープにはノードが残るので呼び出し後の子テープは作り直す。`vmap` は Phase B で失敗すると、それまでのノードが残る。
- `child` は `tape()`／`tape_for()` で作った空の別 `Tape`（`hessian` と同じ契約）。
- 数値: REQ-2 の統一複合判定。`vmap` とバッチなし実行の bit 一致、`hvp` と 1 階 VJP の bit 同一は契約にしない。
- 適用範囲: `hvp` の経路上の Op は `supports_create_graph()` が真のものに限る（論点 5 は保留なので拡張しない）。低精度 forward は対象外で、f32 の `Tape` のみ（`VarF64` は対象外）。`vmap(grad)` は契約外（論点 6 は保留）で、値だけが要る場合は呼び出し側が明示ループで `backward` を回す。 → #3087 追記: 論点 5 は D-A（§29・#3020）で実装へ進み、#3077〜#3086 で対象 Op を拡張した。現在の対象は `supports_create_graph` の網羅 match と `autodiff-higher-order-grad-decision.md` §21.2 を正とする。facade での確認は §34。
- 利用例の doctest は facade 単独で完結する形にする（例: `vmap` は `|s| s.mul(s)`、`vjp` は `x⊙x` から `2x⊙u`、`hvp` は `x³` から `6x⊙v`）。書けなければ #2931 は止まる。

### 23.4 保留ガードの反転範囲

- 公開する名前: facade `Tape` の inherent メソッド `vjp`・`hvp`・`vmap` の 3 つだけ。
- `FunctionalTransformsHoldDoctestGuard` の正のプローブから外すもの: `__probe_methods` 内の `fandhe_ai::Tape::vjp(tape)`／`::hvp(tape)`／`::vmap(tape)` の 3 行、`impl __FandheFunctionalTransformsHoldProbe for fandhe_ai::Tape`（inherent メソッドと名前が衝突して UFCS の解決が変わるため Tape への impl ブロックごと外す）、引数 `tape: &fandhe_ai::Tape`。
- 残すもの（未承認経路のプローブ）: モジュール `functional_ops`（`functional_ops::__mark()`）、クレートルートの裸の自由関数 `vjp()`・`hvp()`・`vmap()`、`Var<'t>`・`Tensor<f32>` への probe trait の impl とそれぞれの UFCS 呼び出し 6 行。
- `api_surface.rs` の追随（#2931 への指定）:
  - `FUNCTIONAL_TRANSFORMS_HOLD_PROBE_BODY` を上記のプローブ本体へ更新する。
  - `facade_does_not_reexport_or_declare_functional_transforms`: 例外は「`crates/facade/src/lib.rs` の `impl Tape` 内の `pub fn vjp`／`hvp`／`vmap` の 3 件ちょうど」に限る。`pub use` での `functional_ops`／`vjp`／`hvp`／`vmap` の再エクスポートは引き続き違反とする。自己テスト（`..._detects_each_category`）も、例外が正しく狭いことを確かめる形へ更新する。
  - `workspace_declares_functional_transforms_fn_names_only_in_allowed_locations`: 期待集合を 4 件（`grad.rs::vjp`・`functional_ops.rs::{vjp, hvp, vmap}`）から 7 件（加えて `facade/src/lib.rs::{vjp, hvp, vmap}`）にする。
  - 正ガードを新設する: 承認形の 3 メソッドのシグネチャと 1 行委譲本体をトークン列で固定する（`facade_tape_backward_create_graph_is_thin_delegation` と同じ系統）。
- 検出範囲は既存ガードと同じく「列挙した名前・型・受け手に限る」。それ以上の保証は書かない。

### 23.5 記録と実装の差の解消（`compat-api-scope.md` §5.1 の差 1〜4）

| 差 | 内容 | 解消の根拠 |
|---|---|---|
| 1 | 受け手（自由関数か `Tape` メソッドか） | 「`Tape` の薄い委譲メソッドとして公開」 |
| 2 | 実装で確定した細部（単一入力・dim 0 固定・`FnMut`・`child` 必須） | 論点 1 の各項目 |
| 2 | 同上（shape の完全一致・`loss` の要素数 1・空バッチ拒否） | コメントが個別に承認した項目ではない。内部シグネチャをそのまま当てることに伴う「実装のまま」の挙動として §23.3 に書く |
| 3 | `vjp`／`hvp` の非対称 | 「現状のまま doc に明記・契約は変えない」 |
| 4 | 保留ガードの反転範囲 | 共通事項「公開した名前の分だけ反転・未承認経路のプローブは残す」 |

- 用語の差: コメントは「追跡なし入力」と書くが、実装上の非対称の対象は追跡なしの `output`／`loss` である（追跡なしの `input` は両方とも `GradientTrackingDisabled`）。指示は「現状のまま」なので、実装の挙動をそのまま書く。この読み方は解釈として明示する。
- 結論: 上の差はいずれも承認の範囲で決まり、停止事由はない。
- 未決として残すもの（推奨は作らない）:
  - 将来 `out_dim`・複数入力を足すときの拡張方法（新メソッドにするか引数化するか。引数化は破壊的変更になりうる）。
  - 論点 5（`supports_create_graph` の対象拡張）・論点 6（`vmap(grad)` の公開形）は保留。
  - 論点 3 はコメントで言及がなく、承認とも保留とも分類しない。#2880 では発動しておらず（§19）、`jvp`／`jacfwd` は項 2 の spec 経路で別に扱う。

### 23.6 #2931 への申し送り

- 23.1〜23.4 の実装、保留ガードと `api_surface.rs` の追随、正ガードの新設。
- `functional_ops.rs` のモジュール doc と保留ガードの doc comment にある「未承認」記述の更新。
- `compat-api-scope.md` §5 への適用記録の追加と、F1〜F3 の「公開先」列の更新。
- CUDA／Metal の実機 parity: 既存の `#[ignore]` テスト（#2881）と `docs/perf/logs/functional-transforms-2881/README.md` で足りるかを判断する。facade 経由の `#[ignore]` テストを足す場合は `docs/perf/logs/<slug>-2931/README.md` へ申し送る（実測値は推測で書かない）。
- 停止条件: `vmap` の寿命の結び方がコンパイルできない、または doctest が facade 単独で書けない場合は、実装せず承認依頼へ戻る。
- 本節はコード・依存・tolerance・baseline・ガードレール閾値・`docs/spec` を変更しない。
- **→ 上記は #2931 で実施した（§26）。**

## 24. 実装記録（#2940・親 #2939。§10 の分解案 10: `jvp`／`jacfwd` の内部実装）

- **実装した形**: `crates/autodiff/src/functional_ops.rs` に `pub(crate) fn jvp(tape, output, input, tangent, child)`（結果 shape = `output.shape()`）と `pub(crate) fn jacfwd(tape, output, input, child)`（結果 shape = `output.shape ++ input.shape`。`jacobian` と同じ並び）を追加した。facade へは出していない。`child` は呼び出し側が用意する空の子テープで、呼び出し後は再利用しない（`backward_create_graph` と同じ契約）。
- **`dead_code` の扱い**: 本番経路に呼び出し元がないため、両関数と private ヘルパーに `#[cfg_attr(not(test), expect(dead_code, reason = "#2940: …"))]` を付けた。`allow` ではなく `expect` なので、将来 facade 公開などで呼び出し元ができると unfulfilled 警告になり属性の撤去を強制できる。
- **アルゴリズム**（§8 のとおり）: 親テープに追跡ありの葉 `u`（**全要素 1**。`g` は `u` について線形なので値は結果に影響しない）と `s = output ⊙ u` を足す（`sum` は足さない。`Tape::backward` のシードが暗黙の総和になる）。`backward_create_graph(&s, child)` で `g = Jᵀu` を子テープへ写す。`jvp` は `child.backward(g ⊙ v)` の `∂/∂u`、`jacfwd` は `FlatElements::new(&g)` の要素 `g_k` ごとに `child.backward(g_k)` を回して `∂g_k/∂u = J[:, k]` を列として書き込む（第 2 段の VJP を列ごとに適用）。one-hot の接ベクトルで `jvp` を `n` 回呼ぶ方式は数学的に同じだが、子テープが `n` 本要り親テープにもノードが `2n` 増えるため採らない。そのため `jacobian_ops::FlatElements`（と `new`／`element`）を `pub(crate)` にした（§4 で実装 issue の判断事項としていた点の確定）。
- **入口検査（ノードを足す前。順序固定）**: (1) `TapeMismatch`、(2) `input.requires_grad() == false` → `GradientTrackingDisabled`、(3)（`jvp` のみ）`tangent.shape() != input.shape()` → `Shape(ShapeMismatch)`（ブロードキャスト不可。`vjp` が output 側と照合するのとは逆向き）、(4) `checked_numel`（`jacfwd` は `m × n` も検査付き乗算）、(5) 要素数 0 または `output.requires_grad() == false` → テープに触れず全ゼロ。
- **追跡なし output の意味論**: `jacobian`／`vjp` と同じ全ゼロ。追跡なし loss を `Err` にする `hvp` とは非対称（本記録で明記）。
- **fail-closed**: `child_var(u)` が `None` になるのは構造上の不変条件違反で、ゼロへ丸めず `Err(Backward)` にする。非対象 Op（`supports_create_graph() == false`）・rank 3 以上の `MatMul`・非空の子テープ・同一テープ・デバイス不一致・checkpoint 登録済みは、既存の `backward_create_graph`（`validate_ancestors` 等）が返す型付き `Err(Backward)` をそのまま伝播する。新しい拒否ロジック・variant は足していない。
- **副作用**: 前段以降の失敗でも親テープにはちょうど 2 ノード（`u` と `mul`）が残る（`validate_ancestors` より前に積まれる。§19 と同じ）。既存ノードの値は不変。拒否時の子テープの扱いは、空の別テープなら前段の事前検査による拒否では空のまま。非空の子テープは消去されず既存状態が維持され（拒否のみ）、親自身を子に渡した（同一テープの）場合は `double_vjp_stage` が追加した 2 ノードも残り、親への追加が子にも見える。成功時の子テープには写し・1 階勾配・第 2 段の補助ノードが残る。子テープ上の数値は 1 階 VJP や `jacobian` と bit 同一を主張せず、統一複合判定で見る。
- **計算量**: `jvp` は親 backward 1 回＋子 backward 1 回、`jacfwd` は子 backward `n` 回。
- **検証**: `crates/autodiff/src/functional_ops_jvp_tests.rs`（`#[cfg(test)]`。`pub(crate)` のため統合テストからは呼べない）。比較先は `jacobian_ops::jacobian`（facade の `Tape::jacobian` はこれへ 1 行委譲で、autodiff から facade へは依存できない）。J1 閉形式（`x⊙x`・`tanh`・`relu`・`Wx`・`sum`）、J2/J3 `jvp` 対 `jacobian·v`・`jacfwd` 対 `jacobian`（区分線形の `relu`／`abs`／`clamp`／`leaky_relu`／`Where`／`Maximum`／`Minimum` を含む、rank 0・非正方形・比較系 J ≡ 0）、J4 入口検査で親子テープ無変更、J5 非対象 Op（`Max`・rank 3 `MatMul`・`Custom`・`gelu`・`selu`・非空／同一テープの子）の拒否と子テープ無変更・親 +2 ノード、J6 決定性・既存ノード不変・`jacfwd` の列 k が `jvp(e_k)` と一致。統一複合判定を外れた要素はなかった。
- **共通ヘルパー**: #2880 のテスト 2 ファイルと重複していた `seq`／`pos`、ホスト f64 の `J·v`、全要素の REQ-2 判定ループを `tests/common/mod.rs`（`det_seq`／`det_pos`／`jacobian_times_vector_f64`／`assert_all_req2_close`）へ寄せた。閾値は `REQ2_*` 定数を経由し直書きしていない。`double_vjp_feasibility_tests` の `common` モジュールは同一ファイルの二重 `#[path]` 取り込み（`clippy::duplicate_mod`）を避けるため `pub(crate)` にして本テストと共用する。#2880 の検証内容（ケース一覧・基準 3 の機構検査）は変えていない。
- **保留ガード**: `FunctionalTransformsHoldDoctestGuard` の正のプローブ（free 関数・3 つの受け手 × 2 名）と固定文言、`api_surface.rs` の `FUNCTIONAL_TRANSFORMS_FN_NAMES`／`IDENTS`・宣言インベントリ（`functional_ops.rs::{jvp, jacfwd}` 各 1 件）・自己テストへ `jvp`／`jacfwd` を追加し、facade 非公開を機械固定した。
- **新規物**: 新規 `Op`・`BackendOps` メソッド・VJP・`AutodiffError` variant・依存・`unsafe`・tolerance・baseline はない。`supports_create_graph` の対象 Op も広げていない（§11 の論点 5 は保留のまま）。
- **承認状況**: ルート #2499 の所有者コメント（https://github.com/Fandhe-AI/fandhe-ai/issues/2499#issuecomment-6067263650 ）項 2 が示す、Tier 2 への移行（spec リポ Fandhe-AI/fandhe-ai-spec#81・実装リポ取り込み #2938。本記録の作成時点で #2938 は未マージ → のちにマージ済み〈`f13e7e4a`〉）と、その後の内部実装・公開形の記録までを根拠とする。facade 公開は承認範囲外で、公開形の記録と承認依頼は #2941（§27）。
- **スコープ外**: facade 公開（#2941）、CUDA／Metal 実機 parity と `docs/perf/logs`（#2942）、`supports_create_graph` の拡張（論点 5）、`VarF64`・f16・複数入力・微分可能な `jvp`。

## 25. 実装記録（#2942・親 #2939。jvp／jacfwd の CUDA／Metal 実機 parity 申し送り）

- **追加したテスト**: `crates/facade/tests/functional_ops_jvp_backend_parity.rs`（新設）。属性なし 3 本（`cpu_double_vjp_matches_naive_reference`・`cpu_double_vjp_matches_jacobian`・`cpu_double_vjp_matches_hand_computed_values`）と、`cuda_double_vjp_jvp_jacfwd_match_cpu_reference`（`#[ignore]`）・`metal_double_vjp_jvp_jacfwd_match_cpu_reference`（`#[ignore]`・`cfg(target_os = "macos")` 限定）。
- **`pub(crate)` のためミラー**: `jvp`／`jacfwd` は facade 非公開（可視性は #2940 の決定どおり変えない）で統合テストから呼べない。#2880 の `double_vjp_probe` と同様に、公開 API（`Tape::var`／`backward_create_graph`／`CreateGraph::grad`／`child_var`・子テープの `backward`・`Var::reshape`／`narrow`）だけで #2940 の手順を逐語ミラーした（`u` は全要素 1、`sum` ノードは足さない）。`jacfwd` ミラーは `g.reshape(&[n])` ＋ `narrow(0, k, 1)` で同一子テープ上の列ループを再現した（one-hot フォールバックは不要だった）。ヘルパー名は保留ガードに合わせ `fn jvp`／`fn jacfwd` を避けた（`double_vjp_jv`／`double_vjp_jac`）。#2941 等で公開が承認されたら本物の関数呼び出しへ置き換える。
- **比較範囲と判定**: `supports_create_graph()` が真の Op のみ（matmul・broadcast add・tanh・sigmoid・exp・mul・transpose・narrow）の 2 フィクスチャで、ミラー jvp／jacfwd と同一バックエンドの `jacobian_ops::jacobian`（および f64 蓄積の `J·v`）を CPU tape と実機で比較する。判定は REQ-2 統一複合判定（`assert_parity`）。区分線形 Op は使っていない（0 近傍の符号割れによる偽の失敗を避ける）。形状は小さく Metal split-K は発動しない。
- **申し送り**: 実機に届かないため未実測。測定コマンドと空の記入欄は `docs/perf/logs/functional-transforms-jvp-2942/README.md`。
- **CI で走らない理由**: CUDA 側は `#[ignore]`、Metal 側はさらに `cfg(target_os = "macos")` のため Linux ではコンパイルもされない。
- **新規物**: `crates/*/src`・依存・tolerance・baseline・`unsafe` は追加していない。
- **承認状況**: facade 公開は承認範囲外で未承認（#2941）。
- **スコープ外**: facade 公開・本物の関数への置換（#2941）、`supports_create_graph` の拡張（論点 5）、`VarF64`・f16、実機での実測そのもの。

## 26. 実装記録（イシュー #2931・親 #2873。facade 公開）

- 状態: **公開済み**。§23.1〜§23.4 の記録どおり、facade `Tape` の inherent メソッド `vjp`・`hvp`・`vmap` を内部 `fandhe_ai_autodiff::functional_ops` への 1 行委譲として公開した。シグネチャは §23.1 と完全一致する。§23.2 の停止条件（`vmap` の `&'t self` から `&'t self.0` を渡す寿命の結び方、facade 単独で書く doctest）は、いずれも問題なく通った（`Tape::vmap` は `Tape::custom` と同型の寿命）。
- rustdoc: §23.3 の契約（入口検査の順序・追跡なしの非対称・副作用・`child` の契約・数値・適用範囲）を書き写し、3 メソッドそれぞれに facade 単独で完結する doctest（`vjp`: `x⊙x` から `2x⊙u`、`hvp`: `x³` から `6x⊙v`、`vmap`: `|s| s.mul(s)`）を付けた。内部関数は intra-doc link にせずバッククォートで書いた。
- 保留ガード: `FunctionalTransformsHoldDoctestGuard` は §23.4 のとおり `Tape` 受け手の分（`impl ... for fandhe_ai::Tape`・UFCS 3 行・引数 `tape`）だけ外した。`api_surface.rs` は固定文言の更新、否定ガード走査の例外（`src/lib.rs` の `impl Tape` 直下・承認形シグネチャと完全一致・名前ごとに 1 件。`gradcheck` の先例と同型）、宣言インベントリの 7 件化、正ガード `facade_tape_functional_transforms_are_approved_thin_delegations`（と自己テスト）、到達性テスト `tape_functional_transforms_are_reachable_via_facade` を追加した。§23.4 の記述からの差はない。
- 結合テスト: `crates/facade/tests/functional_transforms_facade.rs`。CPU では内部実装との一致（REQ-2 統一複合判定）・手計算値・エラー伝播・追跡なしの非対称・`vmap` 結果の微分可能性を固定し、CUDA／Metal は `#[ignore]` で分離した。実機は未実測で、申し送り先は `docs/perf/logs/functional-transforms-facade-2931/README.md`（内部 API 層は #2881 の README が担当）。
- 維持した保留: §23.5 末尾の未決（`out_dim`・複数入力の拡張方法）、論点 5・6、モジュール `functional_ops`・裸の自由関数・`Var`／`Tensor<f32>` 上の同名メソッドの拒否。`jvp`／`jacfwd` の facade 公開は改めて承認が要る。
- 依存・`unsafe`・tolerance・baseline・ガードレール閾値・`docs/spec` は変更していない。

## 27. 決定記録（イシュー #2941・親 #2939。`jvp`／`jacfwd` の facade 公開形・承認依頼）

### 27.0 状態・基準・承認の根拠

- 状態: **記録のみ・未承認・未公開**（コード変更なし）。→ **2026-10-09 に承認され、#2956 で公開済み（§28）。** 以下は承認依頼時点の記録として残す。`FunctionalTransformsHoldDoctestGuard`（`crates/facade/src/lib.rs`）と `crates/facade/tests/api_surface.rs` の否定ガードは維持する。`crates/**`・`Cargo.toml`／`Cargo.lock`・tolerance／baseline・ガードレール閾値・`docs/spec/` は不変で、`fandhe-ai =0.10.0` の公開 API も変わらない。
- 基準: `origin/main` `a18afc40`（2026-10-08）。行番号と内部シグネチャはこの sha のもの。
- 承認の根拠: ルート #2499 のリポジトリ所有者コメント（`https://github.com/Fandhe-AI/fandhe-ai/issues/2499#issuecomment-6067263650`）の**項 2 の最終段落のみ**。内容は、spec 側の Tier 2 移行後に実装 issue を起票し、内部実装と公開形の記録まで進めてよい、facade への公開は記録を書いたあとで改めて承認依頼に戻す、というものである（要約。コメントにない承認は書かない）。項 1 の承認（F1〜F3 と論点 1・2・4）は `vjp`／`hvp`／`vmap` に限られ、`jvp`／`jacfwd` には**及ばない**。承認は実装 Agent が代行しない。

### 27.1 推奨 1 案: facade シグネチャ

`Tape::vjp`／`hessian` と同じ、facade `Tape` の薄い委譲メソッド。引数順・型は内部関数（`crates/autodiff/src/functional_ops.rs` の `pub(crate) fn jvp`／`jacfwd`）から第 1 引数 `tape` を除いたものと一致する。

```rust
impl Tape {
    pub fn jvp(&self, output: &Var<'_>, input: &Var<'_>, tangent: &Tensor<f32>, child: &Tape)
        -> Result<Tensor<f32>, AutodiffError>;   // 結果 shape = output.shape()
    pub fn jacfwd(&self, output: &Var<'_>, input: &Var<'_>, child: &Tape)
        -> Result<Tensor<f32>, AutodiffError>;   // 結果 shape = output.shape ++ input.shape
}
```

新しい型・`AutodiffError` variant・モジュール再エクスポート・裸の自由関数は足さない。`fandhe-ai =0.10.0` に対して追加のみ。

### 27.2 委譲本体と公開時に必要な内部側の変更

- 委譲本体（先例 `Tape::hessian` の `child: &Tape` → `&child.0`）: `jvp` は `fandhe_ai_autodiff::functional_ops::jvp(&self.0, output, input, tangent, &child.0)`、`jacfwd` は `…::jacfwd(&self.0, output, input, &child.0)`。
- 公開時に必要になる内部側の変更（本 issue では行わない。公開 issue への申し送り）: (1) `pub(crate)` → `pub`（facade は他クレートの `pub(crate)` を呼べない。兄弟の `vjp`／`hvp`／`vmap` は既に `pub`）、(2) 両関数と private ヘルパーの `#[cfg_attr(not(test), expect(dead_code, …))]` の撤去（呼び出し元ができると unfulfilled 警告で `-D warnings` に落ちる設計）、(3) `functional_ops.rs` のモジュール doc にある「facade へ出さない」記述の更新。

### 27.3 facade doc に書く契約（すべて現在の実装のまま）

- 入口検査の順: `TapeMismatch` → `GradientTrackingDisabled`（`input` が追跡なし）→（`jvp` のみ）`tangent.shape() != input.shape()` で `Shape(ShapeMismatch)` → `checked_numel`（`jacfwd` は `m × n` も検査付き乗算）→ 要素数 0 または追跡なし `output` は全ゼロ（テープに触れない）。
- `tangent` は **input** の shape と完全一致（ブロードキャスト不可）。`vjp` が output 側と照合するのとは逆向き。
- 追跡なし `output` は全ゼロ（`jacobian`／`vjp` と同じ。追跡なし loss を `Err` にする `hvp` とは非対称）。追跡なし `input` は `GradientTrackingDisabled`。
- 副作用: 入口検査通過後は、`backward_create_graph` 側の拒否（非対象 Op・rank 3 以上の `MatMul`・非空／同一の子テープ・デバイス不一致・checkpoint）でも親テープに 2 ノード（`u` と `mul`）が残る。`vjp` の「入口失敗時・ゼロ返却時はノードなし」とは異なる。拒否時の子テープは、空の別テープなら前段の事前検査による拒否では空のまま。非空の子テープは消去されず既存ノードが維持され、同一テープ（親自身）を渡した場合は `double_vjp_stage` が追加した 2 ノードも残り、親への追加が子にも見える。成功時の子テープにはノードが残るため再利用せず作り直す。`child` は `tape()`／`tape_for()` で作った空の別 `Tape`。
- 適用範囲: 経路上の Op は `supports_create_graph()` が真のものに限る（非対象は既存 `validate_ancestors` の型付き `Err(Backward)`）。f32 `Tape` のみ（`VarF64`・f16 は対象外）、単一入力、結果は非微分のホスト値。 → #3087 追記: 論点 5 は D-A（§29・#3020）で実装へ進み、#3077〜#3086 で対象 Op を拡張した。現在の対象は `supports_create_graph` の網羅 match と `autodiff-higher-order-grad-decision.md` §21.2 を正とする。facade での確認は §34。
- 数値: REQ-2 統一複合判定（相対誤差 1e-3 未満 または 絶対誤差 1e-5 未満）。`jacobian` や 1 階 VJP との bit 同一は主張しない。計算量は `jvp` が親 backward 1 回＋子 backward 1 回、`jacfwd` が子 backward を入力要素数 `n` 回（`jacobian` は出力要素数 `m` 回）。
- 実機 parity（CUDA／Metal）は #2942 の担当で未実測。値は書かない。
- 利用例 doctest 案（公開 issue で facade 単独で完結するか確認）: `y = x⊙x` に対し `jvp` が `2x⊙v`、`jacfwd` が `diag(2x)`。書けなければ公開 issue は実装せず承認依頼へ戻る。

### 27.4 保留ガードの反転範囲（推奨。承認時に確定）

- 公開する名前: facade `Tape` の inherent メソッド `jvp`・`jacfwd` の 2 つだけ。
- 外すもの: `__probe_methods` 内の `fandhe_ai::Tape::jvp(tape)`／`::jacfwd(tape)` の 2 行と、`Tape` への probe trait impl 内の同 2 メソッド。
- 残すもの: モジュール `functional_ops`、裸の自由関数 `jvp()`・`jacfwd()`、`Var<'t>`・`Tensor<f32>` 受け手のプローブ。
- `api_surface.rs`: 固定文言の更新、例外を「`crates/facade/src/lib.rs` の `impl Tape` 内の `pub fn jvp`／`jacfwd`」に限定、宣言インベントリへ `facade/src/lib.rs::{jvp, jacfwd}` を追加（件数は #2931 の前後で変わるため数字で固定せず名前で書く）、正ガード（シグネチャと 1 行委譲をトークン列で固定）を新設する。
- **#2931 との相互作用（申し送り）**: §23.4 は #2931 に「`Tape` への probe trait impl ブロックごと外す」と指定したが、このブロックには #2940 で `jvp`／`jacfwd` のプローブも入っていた。**→ #2931 は選択肢 (a) の形で解決済み**: `Tape` 向けは専用 trait `__FandheFunctionalTransformsHoldTapeProbe`（`jvp`／`jacfwd` の 2 メソッドのみ）へ分離して維持し、UFCS プローブ（`fandhe_ai::Tape::jvp`／`jacfwd`）も維持している（`crates/facade/src/lib.rs` の `FunctionalTransformsHoldDoctestGuard` doc）。多層防御は減っていない。§23.4 本文は書き換えない。

### 27.5 承認依頼の論点（推奨と選択肢。承認は代行しない）

| 論点 | 内容 | 選択肢 | 推奨 |
|---|---|---|---|
| J-a | `jacfwd` を公開するか | 公開する／`jvp` のみ公開し `jacfwd` は内部に留める（§3 の「等価 API があれば新設しない」原則。結果は `Tape::jacobian` と同じ J） | 公開する。計算量が入力要素数に比例し `n < m` で有利という差があり §3 の「等価」に当たらない。ただし対象 Op が `supports_create_graph` の範囲に狭まる点を doc に明記する |
| J-b | `child` の扱い | 明示引数／facade 内部で生成 | 明示引数（`hessian`・承認済み `hvp`・内部実装と同形。子テープの副作用を呼び出し側に見せる） |
| J-c | 名前 | `jvp`／`jacfwd`（`torch.func` 名・保留ガードで先取り済み）／`jacobian_forward` 等の別名 | `jvp`／`jacfwd` |
| J-d | `tangent` の shape 検査（input 側・完全一致・ブロードキャストなし） | 実装のまま契約にする／ブロードキャストを許す | 実装のまま |
| J-e | 追跡なし `output` は全ゼロ（`hvp` と非対称） | 実装のまま doc に明記／`Err` にそろえる（意味論変更） | 実装のまま（`jacobian`／`vjp` と同じ）。項 1 の `vjp`／`hvp` 向け承認とは別に改めて確認を求める |
| J-f | 失敗時にも親テープへ 2 ノード残る副作用 | 実装のまま doc に明記／入口で `validate_ancestors` 相当を先に走らせノードを残さない（実装変更） | 実装のまま doc に明記 |
| J-g | 内部側の可視性 | `pub fn`（兄弟と同じ）／`#[doc(hidden)] pub` | `pub fn` |

- 保留継続（推奨を作らない）: 論点 5（`supports_create_graph` の対象 Op 拡張）、複数入力、`VarF64`・f16、微分可能な `jvp`（結果を `Var` で返す形）。

### 27.6 公開 issue への申し送り（未起票。承認後に起票）

- 27.1〜27.4 の実装、27.2 の内部側変更、`docs/compat-api-scope.md` §5 の適用記録と `F4`・`F5` の公開先列の更新。
- facade 経由の `#[ignore]` テストを足す場合は `docs/perf/logs/<slug>-<issue>/README.md` へ申し送る（#2942 と重複しないか確認。実測値は推測で書かない）。
- 本節はコード・依存・tolerance・baseline・ガードレール閾値・`docs/spec` を変更しない。

## 28. 実装記録（イシュー #2956・親 #2928。`jvp`／`jacfwd` の facade 公開）

- 状態: **公開済み**。承認の根拠はルート #2499 のリポジトリ所有者コメント（`https://github.com/Fandhe-AI/fandhe-ai/issues/2499#issuecomment-6079384681`）の項 1 である。コメントに書かれた範囲（`Tape::jvp`／`Tape::jacfwd` の 2 メソッド公開、論点 J-a〜J-g を推奨どおり、保留ガードの部分反転、追加のみ・依存／tolerance／baseline／閾値／`unsafe` なし）だけを実装した。保留継続は論点 5（`supports_create_graph` の対象拡張）・複数入力・`VarF64`／f16・微分可能な `jvp`。v0.11.0 のリリースは別の手順で、本変更では版数を上げていない。
- 公開した形: §27.1 と完全一致する。`Tape::jvp(&self, output, input, tangent: &Tensor<f32>, child: &Tape) -> Result<Tensor<f32>, AutodiffError>` と `Tape::jacfwd(&self, output, input, child: &Tape) -> Result<Tensor<f32>, AutodiffError>`。本体は `fandhe_ai_autodiff::functional_ops::{jvp, jacfwd}` への 1 行委譲（§27.2）で、記録と内部実装（#2940）に差はなかったため停止条件には当たらなかった。
- 内部側の変更: `functional_ops.rs` の `jvp`／`jacfwd` を `pub(crate)` から `pub fn` にし（J-g）、`double_vjp_stage`・`write_column`・`jvp`・`jacfwd` の 4 か所の `expect(dead_code)` を撤去した。モジュール doc の公開状況を更新した。モジュール自体の再エクスポートは行わない。
- rustdoc: §27.3 の契約（入口検査の順序・追跡なし `output` は全ゼロで `hvp` は `Err` の非対称・失敗時も親テープに 2 ノード残る・`child` は空の子テープ・`tangent` は `input` と shape 完全一致・`supports_create_graph()` が真の Op に限る・REQ-2 統一複合判定・計算量）を書いた。doctest は 2 本あり、`jvp` は `x⊙x` から `2x⊙v` を `Tape::jacobian` とのホスト積と照合し、`jacfwd` は `diag(2x)` を `Tape::jacobian` と照合する（facade だけで書けた）。
- 保留ガード: `FunctionalTransformsHoldDoctestGuard` から `Tape` 受け手のプローブ用トレイトと impl・UFCS 2 行を外し、引数を `_tape` にした（`GradcheckAnomalyHoldDoctestGuard` と同じ先例）。`functional_ops` モジュール・裸の自由関数 5 名・`Var`／`Tensor<f32>` 上の 5 メソッドのプローブは残した。`api_surface.rs` は固定文言、承認シグネチャと本体を 3 件から 5 件へ、宣言インベントリを 11 件へ更新し、到達性テストと自己テスト（別シグネチャ・`impl Var` 内・裸の関数・本体のすり替えの検出）を足した。
- 結合テスト: `crates/facade/tests/functional_transforms_jvp_facade.rs`（CPU のみ・`#[ignore]` なし）。`Tape::jvp` を `Tape::jacobian` と接ベクトルの積に、`Tape::jacfwd` を `Tape::jacobian` に突き合わせた（mul・非正方形 matmul→tanh・sum・ブロードキャスト add・rank 0。REQ-2 統一複合判定）。エラー伝播、非対象 Op（`max`）と空でない子テープの拒否、追跡なし `output` の全ゼロ（`hvp` は `Err`）を固定した。失敗時に親テープへ 2 ノード残る点は、facade からノード数を観測できる公開 API がないため #2940 の内部テストに任せた。
- ミラーテストを維持した理由: §25 には「公開されたら本物の関数呼び出しへ置き換える」とあるが、`functional_ops_jvp_backend_parity.rs` のミラーは #2942／#2954 の GB10 実測記録の対象であり、置き換えると記録の意味が変わる。このためロジックは変えずモジュール doc だけを更新した（意図的に見送り）。置き換えるかどうかはユーザーの確認を経て別 Issue で扱う。
- 実機: 新しいカーネルやデバイス挙動の変更はなく、デバイス上の確認は #2942 の既存 `#[ignore]` テストが担当する。実測記録（`docs/perf/logs/functional-transforms-jvp-2942/README.md`）では CUDA（GB10）は pass、Metal（M4 Max）は未実測である。facade 経由の `#[ignore]` テストは足していない。
- 依存・`unsafe`・tolerance・baseline・ガードレール閾値・`docs/spec` は変更していない。

## 29. 2026-10-11 所有者判断の反映（#3020）

- §27.5 末尾の保留継続（複数入力の `jvp`・微分可能な `jvp`〈結果を `Var` で返す形〉・論点 5 の対象 Op 拡張）は D-A で実装へ進む（#3062・#3065・#3069〈`vmap(grad)`〉・#3075）。
- `VarF64`・f16 の `jvp` は #3021 項 3 の確認待ちで対象外のまま。TapeF64 の `jvp` は #3074 で決める。 → #3074 で §33 に記録（推奨 (b)・所有者確認待ち）
- 基づく判断: 2026-10-11 の所有者判断 D-A（推奨形で一括承認。#3018 ツリー起票の根拠。`compat-api-scope.md` §5 末尾の適用記録を参照）。本節は記録の反映であり新規承認ではない。
- 各 issue で本記録に公開形を書き、その形のまま実装まで進む。記録に形が無い点だけ承認依頼へ戻す。
- 複数入力の `jvp` は #3063 で §30 に記録・実装した。
- 依存・`unsafe`・tolerance・baseline・`docs/spec` は変更していない。既存本文は書き換えていない。

## 30. 決定記録・実装記録（イシュー #3063・親 #3062。複数入力 `jvp` の公開形）

- 状態: **公開済み（内部実装・facade 委譲まで。facade 結合テストと実機 parity は #3064）**。承認の根拠は 2026-10-11 の所有者判断 D-A（推奨形で一括承認。§29・`compat-api-scope.md` §5 末尾の D-A 表の「`jvp` の拡張（複数入力・微分可能な `jvp`）→ #3062・#3065」）。本節は記録の追加であり、これ以上の承認があったとは記さない。基準は `origin/main` `7eee9f80`。
- 変えないもの: tolerance・baseline・依存・`unsafe`・新規 `Op`／`BackendOps`／VJP／`AutodiffError` variant・既存 `Tape::jvp`／`Tape::jacfwd` のシグネチャと挙動・`docs/spec`。

### 30.1 公開形

```rust
impl Tape {
    pub fn jvp_multi(
        &self,
        output: &Var<'_>,
        inputs: &[&Var<'_>],
        tangents: &[&Tensor<f32>],
        child: &Tape,
    ) -> Result<Tensor<f32>, AutodiffError>;   // 結果 shape = output.shape()
}
```

- 委譲本体（1 行）: `fandhe_ai_autodiff::functional_ops::jvp_multi(&self.0, output, inputs, tangents, &child.0)`。内部は同じ引数の `pub fn jvp_multi(tape, output, inputs, tangents, child)`（兄弟関数と同様に `pub fn`。モジュールの再エクスポートはしない）。
- 追加のみの非破壊変更。名前は `jvp` との関係が分かる `jvp_multi`（`torch.func` に別名の関数はない）。新しい型・variant・裸の自由関数は足さない。
- 採らなかった案: (i) 既存 `jvp` の総称化（破壊的変更）、(ii) `inputs: &[Var<'_>]`（`tangents` 側の `&[&Tensor<f32>]` と対称にならない。`&[&Tensor<f32>]` の先例は `CustomFunction::forward`）、(iii) `child` を facade 内部で生成（J-b の決定と同形にそろえるため明示引数のまま）。

### 30.2 契約

- 入口検査（ノードを足す前。順序固定。違反が複数なら最初のものを返す）: ① 長さ不一致 → `InvalidArgument`、② `inputs` 空 → `InvalidArgument`、③ `output`・`inputs` の現世代検査（添字順）→ `TapeMismatch`、④ 追跡なしの入力 → `GradientTrackingDisabled`、⑤ 同一 `Var` の重複指定（`node_id` 一致）→ `InvalidArgument`（二重計上の誤用を防ぐ fail-closed。緩和は Err→Ok の非破壊変更で可能）、⑥ `tangents[i]` の shape 不一致（ブロードキャスト不可）→ `Shape(ShapeMismatch { lhs: input_shape, rhs: tangent_shape })`、⑦ 要素数を `checked_numel` で検査 → `Shape(ElementCountOverflow)`、⑧ 出力の要素数 0・追跡なし `output`・全入力の要素数 0 → テープに触れず全ゼロ（`child` も検査しない）。入力 1 つのときは単一 `jvp` とエラーの種類・順序が一致する。要素数 0 の入力が一部だけなら、その寄与を 0 として外す。
- 算法: 前段は単一版と同じ（親テープへ `u`・`mul` の 2 ノード、`backward_create_graph` を 1 回だけ）。同じ結果から `g_i = cg.grad(inputs[i])` を入力ごとに取り、子テープ上で `p_i = Σ(g_i ⊙ v_i)`（スカラー）を `add` で畳み込んだ合計を `child.backward` 1 回に渡す。`∂/∂u = Σ J_i·v_i`。計算量は入力数に依らず親 1 回＋子 1 回。寄与する入力がなければ全ゼロ。退避案（入力ごとに `child.backward` を呼びホスト側 f64 で加算）は子の backward が k 回になるため採らず、記録のみ。
- 副作用: 入口検査での失敗とゼロ返却ではどちらのテープにもノードを足さない。検査後の失敗（非対象 Op・rank 3 以上の `MatMul`・空でない／同一の子テープ・デバイス不一致・checkpoint 登録済み）でも親テープにちょうど 2 ノードが残る（k に依らない）。成功時の子テープには写し・1 階勾配・第 2 段の補助ノードが残るので `child` は作り直す。
- 数値: REQ-2 統一複合判定で `Σ J_i·v_i`（`jacobian` 由来）と一致する。k = 1 の結果も単一 `jvp` と REQ-2 判定で一致とだけ主張し、bit 同一は主張しない（`sum` ノード分だけ経路が違う）。
- 適用範囲: f32 の `Tape` のみ（`VarF64`・f16 は対象外）。結果は非微分のホスト値（微分可能な版は #3065）。経路上の Op は `supports_create_graph()` が真のものに限る。非有限値は検査せず伝播する。

### 30.3 保留ガードの扱い

- 公開名は facade `Tape` の inherent メソッド `jvp_multi` だけ。`FunctionalTransformsHoldDoctestGuard` は `jvp_multi` を先取りしていなかったため反転するものはなく、未承認の受け手のプローブ（裸の自由関数・`Var`／`Tensor<f32>` 上のメソッド・UFCS）を新たに足した。`Tape` 受け手には足さない（承認形と衝突）。
- `api_surface.rs`: 名前配列 5→6・識別子配列 6→7、承認シグネチャ／本体 5→6、宣言インベントリに内部 1 件と facade 1 件を追加（計 13 件）、固定文言と到達性テストと自己テストを更新した。

### 30.4 実装記録・#3064 への申し送り

- 内部: `double_vjp_stage` の前半を `double_vjp_front` に切り出し（単一版の操作列は不変。既存 J1〜J6 が回帰網）、`jvp_multi` はこれを共用する。テストは `functional_ops_jvp_tests.rs` の M1〜M6（mul・matmul→tanh・ブロードキャスト add・3 入力・rank 0、k = 1 等価、`J_i ≡ 0`・空入力、入口検査と順序、ゼロ返却、非対象 Op・不正な子、決定性）。
- facade: 1 行委譲・rustdoc・doctest（`x1 ⊙ x2`。手計算値と `Tape::jacobian` 由来の参照の両方で照合）。
- #3064 へ: facade 結合テスト（参照は入力ごとの `jacobian` と接ベクトルの積の f64 合計）、`J_i ≡ 0` のケース、k = 1 と単一 `jvp` の等価性、CUDA／Metal 実機 `#[ignore]` parity は本節の対象外で未実装。

### 30.5 実装記録（#3064）

- 追加: `crates/facade/tests/functional_transforms_jvp_facade.rs` に複数入力の CPU 結合テスト（`multi_input_*`。`jacobian` 由来 `Σ J_i·v_i` を f64 一括合計した参照との REQ-2 判定、`J_i ≡ 0`、k = 1 と単一 `jvp`、入口検査エラーと順序、非対象 Op・空でない子テープ〈`Backward`〉、追跡なし `output` の全ゼロ）。実機は新規 `functional_transforms_jvp_multi_backend_parity.rs`（CUDA／Metal の `#[ignore]`）。
- ミラー `functional_ops_jvp_backend_parity.rs` は維持。親テープに残る 2 ノードの副作用は facade から観測できないため内部 M5 に任せた。
- 実機は未実測（`docs/perf/logs/functional-transforms-jvp-multi-3064/README.md`。実測は #3171・#3172）。
- 依存・tolerance・baseline・`unsafe`・`docs/spec` は変更していない。

## 31. 決定記録（イシュー #3066・親 #3065。微分可能な `jvp` の公開形）

### 31.0 状態・根拠

- 状態: **決定記録のみ（コード変更なし。内部実装は #3067、facade 委譲・テスト・`compat-api-scope.md` は #3068）**。基準は `origin/main` `43464810`。承認の根拠は 2026-10-11 の所有者判断 D-A（§29・`compat-api-scope.md` §5 末尾の適用記録。「`jvp` の拡張（複数入力・微分可能な `jvp`）→ #3062・#3065」）。本節は記録の追加であり、これ以上の承認があったとは記さない。
- 変えないもの: tolerance・baseline・依存・`unsafe`・新規 `Op`／`BackendOps`／VJP／`AutodiffError` variant・既存 `Tape::jvp`／`jacfwd`／`jvp_multi` のシグネチャと挙動・`docs/spec`。既存の節の本文は書き換えない（§27.5・§29 の保留記述は本節で形が決まったことで解消する）。

### 31.1 公開形

```rust
// autodiff（functional_ops.rs）。crate ルートで JvpCreateGraphResult を再エクスポートする
pub fn jvp_create_graph<'g>(
    tape: &Tape, output: &Var<'_>, input: &Var<'_>, tangent: &Tensor<f32>,
    child: &Tape, grandchild: &'g Tape,
) -> Result<JvpCreateGraphResult<'g>, AutodiffError>;

pub struct JvpCreateGraphResult<'g> { /* フィールドは非公開 */ }
impl<'g> JvpCreateGraphResult<'g> {
    pub fn output(&self) -> Var<'g>;   // J·v（shape = output.shape()）。grandchild 上の Var
    pub fn child_var(&self, parent_var: &Var<'_>) -> Result<Option<Var<'g>>, AutodiffError>;
}

// facade（impl Tape）。1 行委譲。crate ルートに pub use fandhe_ai_autodiff::JvpCreateGraphResult; を足す
//（CreateGraphResult の再エクスポートと同じ先例）
pub fn jvp_create_graph<'g>(&self, output: &Var<'_>, input: &Var<'_>, tangent: &Tensor<f32>,
    child: &Tape, grandchild: &'g Tape) -> Result<JvpCreateGraphResult<'g>, AutodiffError>;
```

- 名前: `backward_create_graph`／`CreateGraphResult` との対応が分かる `jvp_create_graph`／`JvpCreateGraphResult` を採る。Issue が挙げた候補 `jvp_var` は、結果がどのテープ上の `Var` かを名前が示さず、PyTorch の `create_graph=True` に由来する `_create_graph` 接尾辞の慣例とも揃わないため採らない。名前は `origin/main` の `crates/`・`docs/` と `fandhe-ai-0.11.0`・`fandhe-ai-autodiff-0.11.0` のいずれにも出現しない（検索で 0 件を確認済み）。
- 単一入力のみ（既存 `jvp` と同形）。複数入力版は `double_vjp_front` を共用すれば追加のみで足せる。本ツリーの対象外。
- 採らなかった案: (i) 戻り値を `Var<'g>` だけにする: 入力側の写しを引けず `∂r/∂x` に届かない。(ii) `(Var<'g>, Option<Var<'g>>)`: 写せるのが `input` だけで、重みなど他の親の葉への微分ができない。(iii) 中間テープを内部で生成: `Tape` は `ops: Box<dyn BackendOps + Send>` を所有し複製する API がなく、作るには `BackendOps` の拡張が要る（本ツリーでは禁止）。(iv) `CreateGraphResult` を返す: `grad()` が 1 階勾配の意味を持つため誤用を招く。(v) `first_order()` 相当の bit 同一なホスト値を公開: 公開面を最小に保つため出さない（bit 同一は内部テストで固定）。

### 31.2 孫テープと微分の繋がり方

- `child.backward(..)` はホスト値の `Gradients` しか返さない。2 段目の VJP を `Var` 演算として記録する手段は既存の `Tape::backward_create_graph` を子テープ自身に適用すること（`child.backward_create_graph(&prod, grandchild)`）だけで、結果は孫テープ上の `Var<'g>` になる。中間の `child` と結果を置く `grandchild` はどちらも呼び出し側が空で渡す（J-b の「明示引数」と同形）。
- 結果 `r` は孫テープ上の写しに依存する。元の入力 `x` についての微分は、2 段の写し（親→子→孫）を通して得る。この写しを引く手段が `JvpCreateGraphResult::child_var` で、`CreateGraphResult::child_var` と同形の契約を持つ。
  - 親テープの ID・世代が一致しない → `TapeMismatch`。
  - `requires_grad == false` → `GradientTrackingDisabled`。
  - 1 段目・2 段目のどちらかで祖先に入らない → `Ok(None)`。結果がその変数に依存しない（勾配が恒等的にゼロ）ことを意味する（例: 線形 `y = W x` では `x` の写しが `None`）。
- 親テープの `x` そのものとは微分で繋がらない。境界は `backward_create_graph` の写しであり、`Var::to_tape`（#1614）と同じく非微分。
- 内部実装（#3067）は、成功時に 1 段目の写し（`CreateGraphResult::child_var`）と 2 段目の写しを合成し、親の `NodeId` を添字にした `Vec<Option<Var<'g>>>` を `JvpCreateGraphResult` に持たせる。そのため `CreateGraphResult` に `pub(crate)` の写し参照アクセサを足す（公開面は増えない）。公開型には `'g` だけを出し、1 段目のライフタイム `'c` は出さない。
- 3 階微分には当たらない: `create_graph.rs` の `reduce_bias_grad_var` の doc は「設計上 3 階微分は対象外」とするが、本方式はテープが 3 段になるだけで、利用者が孫テープ上で `backward` して得るのは `x` についての 2 階の量（スカラー出力 `L` なら `H·v`）である。facade の公開 API だけで手で組める合成でもある。
- 2 段目で写せる Op（#3067 のテストで確かめる推測を含む）: `build_cgrads` が子テープに記録する演算は `mul`／`add`／`sub`／`div`／`pow`／`neg`／`log`／`sin`／`cos`／`exp`／`sigmoid`／`PowScalar`／`matmul_fp32_strict`（rank 2）／`transpose`／`permute`／`narrow`／`reshape`／`contiguous`／`broadcast_to`／`Var::cat`／`Var::where_cond`／`sum`／`mean`／定数の `var_no_grad`、および `reduce_bias_grad_var` の `push_eager(Op::Sum { dim: Some(0) })`。いずれも `supports_create_graph()` が真の Op／再生可能な variant に収まる見込みで、漏れがあれば既存の `validate_ancestors` が型付き `Err(Backward)` で fail-closed にする（本記録では新しい拒否ロジックを足さない）。

### 31.3 入口検査・算法・数値

入口検査（ノードを足す前。順序固定）:

1. `output`／`input` が `tape` の現世代に属さない → `TapeMismatch`。
2. `input` が追跡なし → `GradientTrackingDisabled`。
3. `tangent.shape() != input.shape()` → `Shape(ShapeMismatch)`（ブロードキャスト不可）。
4. `checked_numel` → `Shape(ElementCountOverflow)`。
5. テープの事前検査（既存 `jvp` にはない新設）: `child` が `tape` と同一 → `Backward`／`grandchild` が `tape` または `child` と同一 → `Backward`／`child`・`grandchild` のデバイスが `tape` と異なる → `DeviceMismatch`／`child`・`grandchild` が空でない → `Backward`。誤用で親テープに 2 ノードが残る事態を減らすために置く。`backward_create_graph` 側の同じ検査は多層防御として残る。新しい variant は足さない。
6. 出力の要素数 0・入力の要素数 0・`output` が追跡なし → `output()` は `grandchild` 上の全ゼロの追跡なし葉（`var_no_grad`）1 個、写しはすべて `None`。親テープと `child` は無変更。

1〜4 は既存 `jvp` と同じ種類・同じ順序。

算法:

1. `double_vjp_front` と同じ前段（親テープへ `u`・`mul` の 2 ノード、1 段目の `backward_create_graph`。`double_vjp_stage` が `J ≡ 0` で `None` を返す扱いも共用）。
2. `J ≡ 0`（`input` へ勾配が届かない・`g` が定数）なら 6 と同じ全ゼロ葉を返す。
3. 子テープ上で `prod = g ⊙ child.var_no_grad(tangent)` を作る（既存 `jvp` と同じ手順）。
4. `child.backward_create_graph(&prod, grandchild)`。
5. `r = cg2.grad(&cu)` を返す。`None` は構造上の不変条件違反なので、ゼロへ丸めず `Err(Backward)` にする。

- 計算量: 親 backward 1 回・子 backward 1 回（2 段目の `first_order`）・孫テープへの写しと VJP 記録 1 回。
- 数値: 2 段目の `backward_create_graph` は内部で `child.backward(&prod)` を素のまま呼ぶため、`cg2.first_order()` の `cu` 勾配は、同じ手順で `prod` を作る限り既存 `jvp` の結果と bit 同一になる。一方、結果の `Var` の値は孫テープ上で `Var` 演算として計算し直すので、REQ-2 統一複合判定の範囲でのみ一致を主張する（`reduce_bias_grad_var` の縮約は host の f64 縮約と bit 同一にならない）。
- 結果 `r` が追跡なしになりうる（追跡ありの変数に依存しないとき。例: `y = A x` で `A` が `var_no_grad`）。このとき `r` を損失にした `backward` は既存どおり `Err` になる。

### 31.4 既存 `jvp`（§27）・`jvp_multi`（§30）との比較

| 観点 | `Tape::jvp` | `Tape::jvp_multi` | `Tape::jvp_create_graph` |
|---|---|---|---|
| 戻り値 | 非微分ホスト値 `Tensor<f32>` | 同左 | 孫テープ上の `Var<'g>`（`output()`）＋写し（`child_var`） |
| 追跡 | なし | なし | あり（孫テープ上で微分可能） |
| 引数のテープ | `child` 1 枚 | `child` 1 枚 | `child`・`grandchild` 2 枚（呼び出し側が空で用意） |
| 入口エラー | 1〜4（§27.3） | §30.2 の 1〜7 | 31.3 の 1〜4（`jvp` と同じ）＋ 5（テープの事前検査。新設） |
| 追跡なし `output`・要素数 0 | テープに触れず全ゼロのホスト値 | 同左 | `grandchild` に全ゼロ葉 1 個（`Var` を返すため避けられない）。親・`child` は無変更 |
| `J ≡ 0` | 全ゼロのホスト値 | 寄与なしなら全ゼロ | `grandchild` に全ゼロ葉 1 個 |
| 数値 | `jacobian` 由来と REQ-2 | 同左（k = 1 でも bit 同一は不主張） | 値は REQ-2。2 段目の `first_order` は `jvp` と bit 同一 |
| 計算量 | 親 1 回＋子 1 回 | 同左（入力数に依らない） | 同左＋孫テープへの記録 1 回 |

テープに残るノード（親／`child`／`grandchild`）:

| 状況 | 親 | `child` | `grandchild` |
|---|---|---|---|
| 入口検査 1〜5 で失敗 | 無変更 | 無変更 | 無変更 |
| 6 のゼロ返却 | 無変更 | 無変更 | +1（全ゼロ葉） |
| `J ≡ 0` | +2 | 1 段目の写し（非空） | +1（全ゼロ葉） |
| 1 段目で拒否（非対象 Op・rank 3 以上の matmul・checkpoint 等） | +2 | 空のまま | 空のまま |
| 2 段目で拒否 | +2 | 非空（写し・1 階勾配・tangent 葉・`prod`） | 空のまま（`validate_ancestors` が書き込みより前に走る） |
| 成功 | +2 | 非空（再利用不可） | 非空（結果グラフ。再利用不可） |

`J ≡ 0` の行は `double_vjp_stage` の実装（前段の後で `None` を判定）から読み取った値。#3067 のテストで確定させる。

### 31.5 `fandhe-ai =0.11.0` に対する非破壊確認

実測した結果（推測ではない）:

| 確認 | 結果 |
|---|---|
| `jvp_create_graph`・`JvpCreateGraphResult`・`jvp_var` の検索（`fandhe-ai-0.11.0/src`・`fandhe-ai-autodiff-0.11.0/src`） | 0 件（新規の名前） |
| 同検索（`origin/main` の `crates`・`docs`） | 0 件（衝突なし） |
| `git diff v0.11.0 origin/main` の `crates/autodiff/src/create_graph.rs` | `pub fn` の増減 0 件（`backward_create_graph`・`CreateGraphResult` のシグネチャ不変） |
| 同 `crates/facade/src/lib.rs` の `pub fn`／`pub use` 差分 | `jvp_multi` の追加 1 件のみ（`jvp`・`jacfwd`・`backward_create_graph` は不変。0.11.0 では `jvp` 1020 行・`jacfwd` 1057 行・`backward_create_graph` 737 行） |
| 使うエラー variant | `TapeMismatch`・`GradientTrackingDisabled`・`DeviceMismatch`・`Backward`・`Shape` はすべて 0.11.0 に存在。新しい variant なし |

- 追加は新しい型 `JvpCreateGraphResult` 1 つとメソッド 1 つのみ。再エクスポートの追加は minor 扱いで、glob import の衝突は受け入れる（既存の再エクスポート追加と同じ扱い）。`NodeId`／`TapeId`／`BackendOps` は外に出ない。
- `jvp_multi` は 0.11.0 の公開後に足したもので未リリースのため、非破壊の比較対象外。

### 31.6 #3067（autodiff 内部実装）への手順

1. `functional_ops.rs` に `JvpCreateGraphResult<'g>` と `pub fn jvp_create_graph` を足す。既存 `jvp`・`jvp_multi`・`double_vjp_stage`／`double_vjp_front` の挙動は変えない（前段は共用）。
2. `create_graph.rs` に `pub(crate)` の写し参照アクセサを足す（公開面は不変）。
3. `crates/autodiff/src/lib.rs` で `pub use functional_ops::JvpCreateGraphResult;` を足す。
4. `functional_ops_jvp_tests.rs` に単体テストを足す: `y = x ⊙ x` で `output() = 2x ⊙ v` と `sum(output())` の `x` への勾配が `2v`／`tanh` を含む合成で有限差分と一致（REQ-2）／スカラー出力 `L` で写し `x` への勾配が `hvp(L, x, v)` と一致／値が `jvp` と REQ-2 で一致し 2 段目の `first_order` が `jvp` と bit 同一／線形 `W x` で写し `x` が `None`・写し `W` が `Some`／入口検査の順序（5 を含む）と失敗時の各テープ無変更／ゼロ返却の 3 経路／非対象 Op（`max`・rank 3 matmul）の拒否と 31.4 のノード残り／2 段目で写せることを Op の族ごとに確認（bias パターンの add〈`push_eager` 経路〉・matmul・`Where`・区分線形・`ScalarUnary`／`ScalarBinary`）。
5. 2 段目で写せない Op が見つかった場合は、`Err(Backward)` の fail-closed をテストで固定し本節へ追記する。新しい拒否ロジックと `supports_create_graph` の拡張は足さない（#3075 の範囲）。

### 31.7 #3068（facade）への手順

1. `impl Tape` に 1 行委譲 `fandhe_ai_autodiff::functional_ops::jvp_create_graph(&self.0, output, input, tangent, &child.0, &grandchild.0)` を足し、crate ルートに `pub use fandhe_ai_autodiff::JvpCreateGraphResult;` を足す。
2. rustdoc に 31.3 の契約を書く。doctest は facade だけで完結させる（`y = x ⊙ x`、`output()` が `2x ⊙ v`、写し `x` への勾配が `2v`）。
3. `FunctionalTransformsHoldDoctestGuard` に反転するものはない。§30.3 と同様に、未承認の受け手のプローブ（裸の自由関数・`Var`／`Tensor<f32>` 上のメソッド・UFCS）を `jvp_create_graph` について足す。
4. `api_surface.rs` を更新する: 名前配列・識別子配列／承認シグネチャと 1 行委譲本体の正ガード `facade_tape_functional_transforms_are_approved_thin_delegations`／宣言インベントリ（内部 1 件・facade 1 件）／再エクスポート型の正ガード（`CreateGraphResult` の先例）／到達性テスト・自己テスト。既存の `jvp`／`jacfwd`／`hvp`／`jvp_multi` のガードは変えない。
5. `crates/facade/tests/functional_transforms_jvp_facade.rs` に CPU テストを足す（`Tape::hessian`／`Tape::hvp` との整合・有限差分・エラー伝播を含む）。
6. `docs/compat-api-scope.md` の F ブロックに F7 行（F6 と同形）を足し、本記録に実装記録の節を足す。
7. 実機 parity は #3168・#3169 の範囲で、本ツリーでは足さない。

### 31.8 所有者へ戻す論点・スコープ外

- 所有者へ戻す論点: 形は 31.1 で一意に決まっており、承認を要する論点はない。参考として 1 件だけ列挙する。2 枚のテープを明示で渡させる使い勝手について、補助ヘルパー（`tape.child()` 等）を足すかどうかは本記録では足さない（J-b の明示引数と同じ形を保つため）。
- スコープ外: 複数入力の微分可能版、`VarF64`・f16・`TapeF64` の微分可能な `jvp`（#3074・#3021 項 3）、`supports_create_graph` の対象 Op 拡張（#3075）、実機 parity（#3168・#3169）。
- 依存・`unsafe`・tolerance・baseline・ガードレール閾値・`docs/spec` は変更していない。

### 31.9 実装記録（#3067）

- **追加した API**: `fandhe_ai_autodiff::functional_ops::{JvpCreateGraphResult, jvp_create_graph}`（`functional_ops.rs`）。crate ルートの再エクスポートは `JvpCreateGraphResult` のみ。`create_graph.rs` に `pub(crate) fn CreateGraphResult::mirror()` を足した（公開面は不変）。新しい `Op`・`BackendOps`・VJP・`AutodiffError` variant・依存・`unsafe` はない。facade 委譲は #3068。
- **処理順**: §31.3 のとおり。入口検査 1〜4 は `jvp` と同じで、5（テープ関係の事前検査）をゼロ返却より先に行う。前段は `double_vjp_front` を直接呼び、`cg1.grad(input)` の判定（`J ≡ 0` なら全ゼロの追跡なし葉を返す）を本関数内に置いた。`jvp`／`jacfwd`／`jvp_multi` の操作列は変えていない。
- **写しの合成**: `cg1.mirror()` の各 `Some(cv)` のうち `cv.requires_grad()` のものだけ `cg2.child_var(&cv)` で孫テープへ写す。追跡なしの親ノードの子テープ上の写しは `var_no_grad` の葉で、`child_var` が `Err(GradientTrackingDisabled)` を返すため、この確認を省くと正しい呼び出しが失敗する。
- **テスト**: `functional_ops_jvp_tests.rs` の `c1`〜`c9`。C1 閉形式、C2 `tanh(Wx)`・`tanh(x)⊙x` の f64 中心差分、C3 `hvp` との一致、C4 値が `jvp` と REQ-2 一致・2 段目余接勾配が `jvp` と bit 同一（検査用に `#[cfg(test)]` の `stage2_cotangent_grad_host`。公開面には出さない）、C5 写しの `None`／`Some`・`child_var` の誤用、C6 入口検査の順序とテープ無変更、C7 ゼロ経路・ノード残り（§31.4 の表。`J ≡ 0` 行は別の追跡葉に依存する出力で親 +2・child 非空・grand +1・全ゼロを実測確認）、C8 Op の族ごとの 2 段目、C9 決定性。
- **2 段目で写せた Op**: bias パターンの add・matmul・`where_cond`・relu・clamp・abs・`log`・`sin`・`exp`・`sigmoid`・`pow_scalar`・sub・div・pow・reshape・transpose・narrow・cat。2 段目だけで拒否される Op は、今回構成した例では見つからなかった（現在の対象 Op では 2 段目の拒否は到達しない）。
- **未検証**: `DeviceMismatch` は `test_support` に別デバイスを返す `BackendOps` がないため単体テストで検証していない（実装は `backward_create_graph` と同じ検査）。
- **#3068 への申し送り**: facade `api_surface.rs` の `CreateGraphResult` 再エクスポート検査は `pub use` 行に `"CreateGraphResult"` が部分文字列として含まれるかで判定しており、`pub use fandhe_ai_autodiff::JvpCreateGraphResult;` も引っかかる。完全一致化するか承認形を別に足す必要がある。`jvp_create_graph` を宣言インベントリへ足すと `autodiff/src/functional_ops.rs` の 1 件が数えられる（autodiff 側のテストに同名の `fn` は置いていない）。

### 31.10 実装記録（#3068）

- **公開した形**: §31.1 と一致する（差はない）。`Tape::jvp_create_graph<'g>(&self, output, input, tangent, child: &Tape, grandchild: &'g Tape) -> Result<JvpCreateGraphResult<'g>, AutodiffError>` を `impl Tape` の 1 行委譲（`fandhe_ai_autodiff::functional_ops::jvp_create_graph(&self.0, ..., &child.0, &grandchild.0)`）として足し、crate ルートに `pub use fandhe_ai_autodiff::JvpCreateGraphResult;` を足した。承認根拠は 2026-10-11 の所有者判断 D-A（一括承認）で、これを超える承認はない。
- **rustdoc と doctest**: §31.3 の契約（テープの役割と再利用不可・微分の繋がり方・入口検査の順序・ゼロ経路・ノードの残り方・数値と適用範囲）を書いた。doctest は facade だけで完結し、`y = x ⊙ x` で `output() = 2x ⊙ v = [2, 2, 12]`、`grandchild.backward` 後に `child_var(&x)` の勾配が `2v = [2, 1, 4]` になることを確かめる。
- **保留ガード**: 反転対象はない。`FunctionalTransformsHoldDoctestGuard` に、裸の自由関数・`Var<'t>`・`Tensor<f32>` 上の `jvp_create_graph`（トレイトメソッド・UFCS）のプローブを足した。`Tape` 受け手のプローブは承認形と衝突するため足さない。
- **`api_surface.rs`**: 名前配列 7 名・識別子配列 8 件・承認シグネチャ（`<'g>`・`grandchild: &'g Tape` を含む）と 1 行委譲本体を足し、宣言インベントリを 15 件にした。`CreateGraphResult` の再エクスポート検査は `pub use` 行への部分文字列一致から識別子完全一致（`scan_pub_use_decls_containing_idents`）へ変えた（§31.9 の申し送り。検出力を保ったまま `JvpCreateGraphResult` 等の別名の型への誤一致だけを除く、締める方向の変更）。新設の正ガード `facade_reexports_jvp_create_graph_result_in_approved_shape` は `lib.rs` の承認形 1 行だけを許し、自己テスト `facade_reexports_create_graph_result_guards_detect_each_category` が各違反類型と両ガードの識別子分離を固定する。到達性テストは関数型を型で固定し CPU で最小呼び出しを行う。
- **結合テスト**（`functional_transforms_jvp_facade.rs`、CPU）: `create_graph_value_matches_jvp_and_closed_form`・`create_graph_second_order_is_two_v_for_square`・`create_graph_matches_hvp_and_hessian_for_scalar_loss`・`create_graph_child_var_semantics`・`create_graph_input_validation_errors_propagate`・`create_graph_zero_paths`・`create_graph_unsupported_op_is_rejected`。数値比較は REQ-2 統一複合判定（`assert_parity`）を使い、tolerance は追加・変更していない。
- **未検証**: `DeviceMismatch`（CPU だけでは作れない。§31.9 と同じ扱い）、facade からの親・子・孫テープのノード残り観測（公開 API がないため内部テスト C6／C7 に任せる）。
- **申し送り**: 実機（CUDA／Metal）parity は #3168・#3169。`supports_create_graph` の対象 Op 拡張は #3075。
- **変えていないもの**: 依存・`unsafe`・tolerance・baseline・ガードレール閾値・`docs/spec`・既存の公開シグネチャ。

## 32. 決定記録（イシュー #3070・親 #3069。`vmap(grad)` の公開形）

### 32.0 状態・根拠

- 状態: **決定記録のみ（コード変更なし。autodiff 内部実装は #3071、facade 委譲は #3072、facade テストと実機 `#[ignore]` は #3073）**。基準は `origin/main` `52861cd3`。承認の根拠は 2026-10-11 の所有者判断 D-A（§29・`compat-api-scope.md` §5 末尾の D-A 表「`vmap(grad)` → #3069」）。本節は記録の追加であり、これ以上の承認があったとは記さない。
- 経緯: 既存 `Tape::vmap`（facade `crates/facade/src/lib.rs:1270`、内部 `crates/autodiff/src/functional_ops.rs:203`）は入力 `Var` と同じテープ上でスライスをクロージャへ渡し `Var::stack` する。クロージャ内で勾配を取る合成は契約外だった（§5・§11 論点 6・§23.3）。理由は、`backward` の勾配がホスト `Tensor` で微分の接続が切れること、`backward_create_graph` の勾配が子テープ上の `Var` で同一テープを要求する `vmap` の契約と合わないことの 2 点。本節で形が決まったので §11 論点 6 の保留は解消する（§31.0 と同じ扱い）。
- 変えないもの: tolerance・baseline・依存・`unsafe`・新規 `Op`／`BackendOps`／VJP／`AutodiffError` variant・既存 `Tape::vmap` ほかのシグネチャと挙動・`docs/spec`。既存の節の本文は書き換えない。バッチ規則型 `vmap` は spec 上対象外のままで、本節は広げない。

### 32.1 公開形

```rust
// autodiff（functional_ops.rs）。兄弟関数と同じ pub fn（モジュールの再エクスポートはしない）
pub fn vmap_grad<F>(input: &Tensor<f32>, in_dim: usize, child: &mut Tape, f: F)
    -> Result<Tensor<f32>, AutodiffError>
where F: for<'c> FnMut(&'c Tape, &Var<'c>) -> Result<Var<'c>, AutodiffError>;

// facade（impl Tape の関連関数。self なし）。1 式の委譲
pub fn vmap_grad<F>(input: &Tensor<f32>, in_dim: usize, child: &mut Tape, f: F)
    -> Result<Tensor<f32>, AutodiffError>
where F: for<'c> FnMut(TapeRef<'c>, &Var<'c>) -> Result<Var<'c>, AutodiffError>;
// 本体: fandhe_ai_autodiff::functional_ops::vmap_grad(input, in_dim, &mut child.0,
//           |t, s| f(TapeRef::from_autodiff(t), s))
```

- 戻り値は、サンプルごとの勾配（各スライス葉についての `∂loss_i/∂slice_i`）を dim 0 に積んだ非微分のホスト値 `Tensor<f32>`。shape は `[B, …slice_shape]`。`slice_shape` は入力 shape から `in_dim` を除いたもの（`unbind` と同じ）。入力 shape と一致するのは `in_dim = 0` のときだけ（`vmap` の出力軸 dim 0 固定と同じ扱い）。
- 名前は `torch.func` の `vmap(grad(f))` に対応させた `vmap_grad`。実測: `origin/main` の `crates`・`docs`・`site`、および `fandhe-ai-0.11.0`・`fandhe-ai-autodiff-0.11.0`（`~/.cargo/registry`）で `vmap_grad`・`per_sample_grad` を検索し、0 件（2026-10-15）。
- facade は関連関数とする。親テープが無いので `&self` を取らない。裸の自由関数 `fandhe_ai::vmap_grad` は既存の保留方針（裸の自由関数は未承認経路）に合わせて公開しない。
- クロージャへ tape ハンドルを渡す理由: `Var::tape()` は `pub(crate)`（`var.rs:223`）で、スライス `Var` だけでは子テープ上に重み・ターゲット等の定数を登録できない。`matmul` を含む損失が書けなくなる。facade は `var` 系だけを持つ借用ハンドル `TapeRef<'c>` を渡す。先例は `nn::Module::forward(TapeRef<'t>, &Var<'t>)`（`crates/facade/src/nn/container.rs:396`）と #2397 の `TapeRef::from_autodiff`。`backward`／`reset` は公開面に出ない（REQ-12）。
- この形の代償: HRTB のため、クロージャはサンプルをまたいで `Var` を保持できない。定数はサンプルごとに登録し直す。これは明示ループ参照と同じで、性能は契約にしない。
- 採らなかった案:
  - (a) サンプル数ぶんのテープを受ける: `BackendOps` を B 個作らせ B 個のグラフを保持する。tape ハンドルの問題も解決しない。
  - (b) 1 枚のテープに `reset` せず全サンプルを記録し、サンプルごとに `backward`: `backward` は `nodes.len()` 全体を走査するため（`backward.rs` の `grads = vec![None; n]`・`(0..n).rev()`）、サンプル i が O(i·G) になり全体で O(B²·G) 時間・O(B·G) メモリ。明示ループより悪い。
  - (c) 子テープを値で受けて消費する: 学習ステップごとに `tape_for` し直す必要があり、`reset` 再利用の利点を失う。
  - (d) 戻り値を `Var` にする: 32.5 のとおり含めない。
- 停止条件（§23.2 と同型）: HRTB クロージャ、または facade のアダプタクロージャが newtype 越しにコンパイルできない場合、#3071／#3072 は実装を止めて本記録へ戻る。

### 32.2 子テープの扱い（葉プレフィックス契約との照合）

- 採用: 1 枚の `&mut Tape` をサンプルごとに `reset` して使い回す。
- 入口: `child.is_empty()` でなければ `Err(Backward)`（`create_graph.rs:260`・§31.3 の 5 と同じ variant）。空でない子テープへの `reset` は呼び出し側のノードを壊すため fail-closed。入口で `child.rollback_point()`（`tape.rs:3104`・`pub(crate)`）を保存する。
- サンプル i（0..B）ごとの処理:
  1. ホストで `input.narrow(in_dim, i, 1)` → `contiguous()` → `reshape(slice_shape)`。
  2. `child.var(&slice)` で追跡ありの葉 `s` を作る。
  3. `f(&*child, &s)` を呼び、損失を検査する（32.3）。
  4. `child.backward(&loss)` → `grads.get(&s)` を結果バッファの i 行目へコピーする。
  5. `child.reset()`。
- サンプルごとの本体は所有権のある `Tensor` を返してから `reset` する。`Var<'c>` が `&mut` の借用をまたがない構成にする（#3071 への注意）。
- 出口: 成功時は手順 5 で最後のサンプルの `reset` が済んでいるため追加の `reset` は呼ばず、`child.rollback_to(start)` だけで入口の状態へ戻す。Phase B の `Err` 時は、失敗したサンプルのグラフが残っているので `reset` を 1 回だけ呼んでから `child.rollback_to(start)` する（手順 5 を通らないサンプルに対する後片付けで、手順 5 と合わせても 1 サンプルあたり `reset` は 1 回）。いずれの場合も入口の状態へ戻す。これが無いと、サンプル 0 のスライス葉（とクロージャが最初の演算より前に登録した葉）が葉プレフィックスに固定され（`tape.rs:3141` の `freeze_leaf_prefix`・`reset`（`tape.rs:3210`）の doc「保持される葉の範囲」）、すべての `reset` を越えて残り、次の呼び出しの空チェックで失敗する。`rollback_to` の doc は「失敗時専用」なので、成功時の使用も含む形に内部 doc だけ広げる（公開面は不変）。
- 結果の契約: 子テープは空かつ入口の葉プレフィックス状態で戻る。`Tape::reset` は呼ぶたび世代（epoch）を 1 進めるため、`reset` の呼び出し回数は成功時 B 回（サンプルごとに 1 回）で、世代は B だけ進む（処理したサンプル数と一致）。`Err` 時は失敗サンプルの後片付けを含め、開始済みのサンプル数（成功済み＋失敗 1）だけ進む。hook・checkpoint 区間は `reset` が消す。ピークメモリはサンプル 1 個分のグラフ。Phase A の失敗では子テープは無変更。

### 32.3 入口検査・エラー順序（新しい variant なし。`AutodiffError` を再利用）

- Phase A（子テープ無変更）:
  1. `in_dim >= rank`（rank 0 を含む）→ `Shape(AxisOutOfRange)`
  2. `shape[in_dim] == 0`（空バッチ）→ `InvalidArgument`（`vmap` と同じ。結果形状を推定しない）
  3. 結果要素数を `checked_numel` で検査 → `Shape(ElementCountOverflow)`
  4. `child` が空でない → `Backward`
- Phase B（サンプル順。最初の違反で止まる）:
  5. クロージャの `Err` はそのまま伝播する。
  6. 戻り値が `child` の現世代に属さない → `TapeMismatch`（`check_on_tape`）。HRTB があっても、変性により leak した `'static` テープの `Var` が通りうるため実行時検査として残す。
  7. 戻り値の要素数が 1 でない → `InvalidArgument`（`hvp` と同じく `[]`・`[1]`・`[1, 1]` は可。素の `backward` は非スカラーを全要素 1 で暗黙に総和するため先に拒否する）。
  8. 戻り値が追跡なし → その行は全ゼロ（`vjp` の追跡なし `output` の先例・§23.3 に合わせる）。追跡ありでもスライスへ勾配が届かない（`grads.get(&s) == None`）→ 全ゼロ。
  9. `backward` やバックエンドの `Err`（`DeviceMismatch` 等）は伝播する。
- 入力は `Tensor` なので追跡なし入力という概念は無い。非有限値は検査せず伝播する。経路上の Op は 1 階の `backward` が対応する範囲で、`create_graph` の対象 Op には限らない。

### 32.4 数値

- 参照は `vmap_grad` を経由しない明示ループ（各スライスで `backward`。§23.3 の「値だけが要る場合」）とし、REQ-2 統一複合判定（相対誤差 1e-3 未満 または 絶対誤差 1e-5 未満）で比較する。同じ Op 列なので bit 一致の見込みは高いが契約にしない。tolerance・baseline は新設も変更もしない。

### 32.5 微分可能な結果は今回含めない

- 理由 1: 微分可能にするには、サンプルごとに `backward_create_graph` で勾配を孫テープ上の `Var` にする必要がある。各サンプルの勾配は別々の（`reset` で消える）テープに置かれ、`Var::stack` はテープをまたげない（`Var::to_tape` は非微分）。
- 理由 2: 経路上の Op が `supports_create_graph()` の対象に狭まる。
- 理由 3: 入力が `Tensor` なので、親グラフとの接続がそもそも無い。
- 将来の追加: §31.1 の命名パターンに従い別名の関数（例 `vmap_grad_create_graph`）として足せば、`vmap_grad` のシグネチャは不変で追加のみの非破壊変更になる（`AutodiffError` は `#[non_exhaustive]`）。

### 32.6 既存 `Tape::vmap` との比較と非破壊確認

| 観点 | `Tape::vmap`（§17・§23） | `Tape::vmap_grad` |
|---|---|---|
| 入力 | `Var`（親テープ上） | `Tensor<f32>`（ホスト値） |
| テープ | 入力と同一 | 空の子テープ `&mut`（サンプルごとに `reset`） |
| 戻り値 | 微分可能な `Var`（`Var::stack`） | 非微分の `Tensor<f32>` |
| クロージャ内の勾配 | 契約外 | 損失 `Var` の `backward` を内部で実行（クロージャは勾配を取らない） |
| Phase B 失敗時 | 記録済みノードが残る | 子テープは入口状態へ戻る |
| 計算量・メモリ | B 回ぶんのグラフを保持 | 時間は B 回ぶん、ピークメモリはサンプル 1 個分 |
| 数値判定 | REQ-2 | REQ-2（ループ参照。bit 一致は契約外） |

- 非破壊: `Tape::vmap` のシグネチャと挙動は不変。公開面の変更は facade `Tape` の関連関数 1 つの追加のみ。0.11.0 に同名なし（32.1 の実測）。新しい型・variant・再エクスポートは無い。`Tape::vmap` の rustdoc の「`vmap(grad)` は契約外」は #3072 で `vmap_grad` への案内に更新する（本 issue ではコードに触れない）。

### 32.7 #3071・#3072・#3073 への手順

- **#3071（autodiff）**:
  - `functional_ops.rs` に `pub fn vmap_grad` を追加する（既存関数の挙動は不変）。最初に HRTB クロージャのコンパイル可否を確かめる（32.1 の停止条件）。
  - モジュール doc の共通契約「`vmap` のみ同じテープ上の微分可能な `Var`」と `vmap` doc の「契約外」を更新する。`tape.rs` の `rollback_to` doc を成功時の使用も含む形に広げる（`pub(crate)` のまま）。
  - 単体テスト（`functional_ops_composition.rs` の隣）: `sum(x⊙x)`・`tanh`・`matmul` の損失と明示ループ参照の REQ-2 一致（バッチ 1・複数・`in_dim ≠ 0`）。32.3 の全エラーと順序。途中サンプルの `Err` で子テープが空・入口状態へ戻ること。直後の 2 回目の呼び出しが成功すること。追跡なし損失・未到達のゼロ行。
  - 新しい依存・`unsafe` は無い。
- **#3072（facade）**:
  - 関連関数の 1 式委譲と rustdoc、facade 単独の doctest（例 `|t, s| s.mul(s)?.sum(None)` で勾配 `2x`）。`Tape::vmap` rustdoc の更新。
  - `FunctionalTransformsHoldDoctestGuard` に、未承認の受け手のプローブ（裸の自由関数・`Var`／`Tensor<f32>` 上の `vmap_grad`）を足す。反転対象は無い見込み。
  - `api_surface.rs` の承認シグネチャと委譲本体の正ガード（アダプタクロージャを含む本体の形）。名前・識別子配列と宣言インベントリは名前で固定する（§27.4 方針。現行の 7／8／15 件から 8／9／17 件になる見込み）。到達性テスト・自己テスト。
  - `docs/compat-api-scope.md` §5.1 F ブロックに F8 行を足し、本設計記録に実装記録を足す。
- **#3073（テスト）**:
  - `functional_transforms_facade.rs` の CPU テスト（明示ループ参照・`in_dim` 0／1・バッチ 1・エラー経路）。CUDA／Metal の `#[ignore]` テスト。
  - `docs/perf/logs/functional-transforms-vmap-grad-3073/README.md`（未実測欄）と `docs/README.md` 索引。

### 32.8 所有者へ戻す論点・スコープ外

- 所有者へ戻す論点（形を本記録に書けないもの）:
  1. 共有パラメータについてのサンプルごとの勾配（一般的な ML の意味での per-sample gradient。`torch.func` の `in_dims=(None, 0)`）は本 API の戻り値に含まれない。返す形（複数出力・パラメータ指定の方法）は本記録では決めない。
  2. 定数のサンプルごとの再登録コスト（CUDA／Metal ではサンプルごとの転送）を実測で問題視した場合の、事前登録葉を共有する形の要否。
- スコープ外: バッチ規則型 `vmap`、微分可能版（32.5）、`out_dim`・複数入力、`VarF64`・f16・`TapeF64`、実機実測（#3171・#3172）、`supports_create_graph` の拡張（#3075）、`docs/spec`。
- 依存・`unsafe`・tolerance・baseline・ガードレール閾値・`docs/spec` は変更していない。

### 32.9 実装記録（#3071）

- **追加した API**: `fandhe_ai_autodiff::functional_ops::vmap_grad`（`crates/autodiff/src/functional_ops.rs`。モジュール・crate ルートへの再エクスポートなし）。シグネチャは §32.1 のとおり（HRTB `for<'c> FnMut(&'c Tape, &Var<'c>) -> Result<Var<'c>, AutodiffError>` はコンパイルできた。停止条件には該当しない）。サンプル 1 個分は非公開ヘルパー `vmap_grad_sample` が処理する。
- **処理順**: §32.3 の Phase A（1 軸範囲 → 2 空バッチ → 3 要素数オーバーフロー → 4 子テープ非空）→ Phase B（クロージャ → `check_on_tape` → 損失の要素数 1 検査 → 追跡なしはゼロ行 → `backward` → `grads.get` の行コピー）。新しい variant・Op・依存・`unsafe` は足していない。
- **後片付けの選択（§32.2 の契約内での内部の選択）**: サンプルごとに成否に関わらず `reset()` の直後に `rollback_to(start)` を呼ぶ（§32.2 の記述〈サンプルごとに `reset`、出口で `rollback_to`〉より厳しい形）。サンプル 0 の葉が葉プレフィックスに固定されて残ることを防ぎ、演算を記録しないクロージャで葉が積み上がることも防ぐ。`rollback_to` は世代・hook を触らないので必ず `reset` と併用する。戻り時の子テープは空で入口の葉プレフィックス状態と同じ。
- **テスト**: `crates/autodiff/tests/functional_ops_vmap_grad.rs` 9 件（ループ参照との REQ-2 一致〈sq・tanh・matmul × バッチ 1・複数・`in_dim` 0/1〉、閉形式 `2x`、rank 0 スライスと損失 shape `[]`／`[1]`／`[1,1]`、Phase A 各エラーと順序・子テープ無変更、Phase B の各エラーと失敗後の再利用、途中サンプル失敗後の後片付けと呼び出し回数、葉プレフィックスの復元、ゼロ行、要素数 0 スライス）。
- **未検証**: `DeviceMismatch`（CPU だけでは作れない）、結果要素数オーバーフロー（実在の `Tensor` では作れない。コード上の `checked_numel`／`checked_mul` 検査のみ）、世代の進み方（`Tape::epoch` が `pub(crate)` のため結合テストから観測していない）。
- **#3072 への申し送り**: 宣言インベントリには `autodiff/src/functional_ops.rs::vmap_grad` が 1 件数えられる（ヘルパー `vmap_grad_sample` は別名）。

### 32.10 実装記録（#3072）

- **公開した形**: §32.1 と一致する。差は実引数 `f` が `mut f: F` になった点だけ（`F: FnMut` をアダプタクロージャ内から呼ぶために必要。公開される型には影響しない）。`Tape::vmap_grad<F>(input: &Tensor<f32>, in_dim: usize, child: &mut Tape, mut f: F) -> Result<Tensor<f32>, AutodiffError> where F: for<'c> FnMut(TapeRef<'c>, &Var<'c>) -> Result<Var<'c>, AutodiffError>` を `impl Tape` に関連関数（`&self` なし）として足し、本体は `fandhe_ai_autodiff::functional_ops::vmap_grad(input, in_dim, &mut child.0, |t, s| { f(TapeRef::from_autodiff(t), s) })` の 1 式の委譲。HRTB クロージャと newtype 越しのアダプタはコンパイルできた（停止条件には該当しない。先例は `Tape::gradcheck`）。承認根拠は 2026-10-11 の所有者判断 D-A（一括承認）で、これを超える承認はない。
- **rustdoc と doctest**: 子テープの扱い（空を渡す・サンプルごとに内部で `reset`・成功時も失敗時も戻り時は空で再利用できる・panic 時は契約外）、クロージャ、エラー順序（§32.3）、数値と適用範囲（§32.4・§32.5・§32.8-1）を書いた。doctest は facade だけで完結し、`in_dim = 0` で行ごとの勾配 `2x` を、続けて同じ子テープで `in_dim = 1` を呼べること（shape `[3, 2]`）を確かめる。
- **`Tape::vmap` の rustdoc**: 「`vmap(grad)` は契約外」の文を、クロージャ内で勾配を取る合成は契約外で、サンプルごとの勾配は `Tape::vmap_grad` を使う旨に更新した。シグネチャ・本体・doctest は不変。
- **保留ガード**: 反転対象はない。`FunctionalTransformsHoldDoctestGuard` に、裸の自由関数・`Var<'t>`・`Tensor<f32>` 上の `vmap_grad`（トレイトメソッド・UFCS）のプローブを足した。`Tape` 受け手のプローブは承認形と衝突するため足さない。方式は正のプローブのまま。
- **`api_surface.rs`**: 名前配列 8 名・識別子配列 9 件・承認シグネチャ（`mut f: F`・`where` 節を含む）と委譲本体を足し、宣言インベントリに `autodiff/src/functional_ops.rs::vmap_grad` と `facade/src/lib.rs::vmap_grad` の 2 件を足した（件数ではなく名前で固定。§27.4）。自己テストに本体のすり替え（`&mut child.0` の脱落・アダプタの脱落・`input`／`in_dim` の入れ替え）、シグネチャ改変（`&self` の追加・`TapeRef` → `&Tape`・`for<'c>` の脱落）、裸の自由関数・再エクスポート・`impl Var` 内の承認形を足した。到達性テストは関数型を型で固定し、CPU で最小呼び出しを行う。
- **未検証**: `DeviceMismatch`（CPU だけでは作れない）、facade からの子テープの世代の進み方の観測（公開 API がない）。
- **申し送り**: facade の CPU 結合テスト（ループ参照・`in_dim` 0／1・バッチ 1・エラー経路）と実機 `#[ignore]` テスト・perf ログは #3073（実機は #3171・#3172）。所有者へ戻す論点は §32.8 のとおり。
- **変えていないもの**: 依存・`unsafe`・tolerance・baseline・ガードレール閾値・`docs/spec`・`crates/autodiff`・既存の公開シグネチャ。

## 33. 決定記録（イシュー #3074。`TapeF64` の `jvp`: 規模見積もりと実装／契約固定の比較）

### 33.0 状態・根拠・基準

- 状態: **記録のみ。推奨は (b)「`jvp` は f32 `Tape` 限定」を契約として固定、所有者確認待ち**。コード・公開 API・tolerance・baseline・依存は変えない。
- 根拠: 2026-10-11 の所有者判断 D-A（§29）。f16 は #3021 の所有者回答（`issuecomment-6111746880`・2026-10-12・項 3）で対象外が確定済みのため、本節は f64（`TapeF64`／`VarF64`／`GradientsF64`）だけを扱う。本節は承認を代行しない。
- 基準: `origin/main` `9ea9353f`。以下の行番号・規模はこの sha 上で再現できる。

### 33.1 現状の確認（コード事実）

- double-VJP 法の `jvp`／`jacfwd`／`jvp_multi`／`jvp_create_graph` は `Tape::backward_create_graph`（子テープ上で微分できる `Var` 勾配）を前提にする（§19・§27・§31）。
- `crates/autodiff/src/f64_autograd.rs` の `TapeF64` の公開メソッドは `new`／`var`／`var_no_grad`／`backward`（275 行目）の 4 つ。`VarF64` は `value`／`shape` と 8 演算（`add`／`mul`／`div`／`pow`／`matmul`／`sum`／`mean`／`max`）だけ。`GradientsF64::get`（421 行目）は `Option<&Tensor<f64>>` を返し、ホスト値しか持たない。
- 同ファイルに `create_graph`・子テープ・`rollback_to`／`reset`・`reshape`／`broadcast_to`／`transpose`・`log`／`where` はない（grep で確認）。
- `TapeF64::backward` は `&self` で、ノード列を読むだけで書き換えない。同じグラフに対して繰り返し呼べる。
- facade は newtype で公開している（`crates/facade/src/lib.rs` の `TapeF64` 1749 行・`VarF64` 1780 行・`GradientsF64` 1835 行）。
- f32 側の機構は `crates/autodiff/src/create_graph.rs`（1621 行。祖先収集・検査・ミラー構築・`replay_op`・`build_cgrads`・`CreateGraphResult<'c>`）。f32 でも `Max` は `supports_create_graph()` の対象外（#3075 の領域）。
- `TypedOps<f64>` は 8 演算で固定。`div`／`pow`／`mean` は既にホスト経路で実装済み。

### 33.2 演算ごとの作業表

判別の軸: double-VJP の `jvp` では `g = Jᵀu` が `u` について線形なので、各 VJP の局所係数（`a`・`b`・`y` だけに依存する項。`binary_partials_f64`〈1101 行〉の係数など）は、子テープ上で `var_no_grad` の定数として扱える。問うのは、追跡対象の `g` に対する VJP を既存 8 演算だけで書けるかである。

| 演算 | 子テープ上の VJP 表現 | 既存 `VarF64` 演算だけで足りるか | 足りない構造演算 | 備考 |
|---|---|---|---|---|
| `add`（同 shape） | `g` をそのまま渡す | 足りる | なし | 自明 |
| `mul`／`div`（同 shape） | `g` に定数係数を `mul` | 足りる | なし | 係数は `binary_partials_f64` 由来のホスト値 |
| `pow`（同 shape） | `g` に定数係数を `mul` | 足りる | なし | `ln a` も定数係数に入るため `log` 演算は不要。f32 版 `build_cgrads`（`create_graph.rs` 1063 行以降）の a==0 の NaN 安全化は、線形化した定数係数側で扱える見込み（推測） |
| ブロードキャストを伴う `add`／`mul`／`div`／`pow` | 入力 shape への縮約が要る（`reduce_to_shape_f64` 相当） | 足りない | `reshape`（size-1 軸を残す形）または縮約専用の追跡演算 | `sum(Some(axis))` は軸を落とすだけで size-1 軸を残せない |
| `matmul` | `g.matmul(bᵀ の定数)`・`aᵀ の定数.matmul(g)` | 足りる | なし（`jvp` のみの場合） | 転置した定数はホスト側 `Tensor::transpose` で作れる（`matmul_vjp_f64` 1268 行に前例）。`hvp` 系の完全な create_graph では追跡演算の `transpose` が要る（推測） |
| `sum(None)` | `g` に ones 定数を `mul`（ブロードキャスト） | 足りる | なし | スカラー `g` に限る |
| `mean(None)` | `g` に「`1/n` を要素に持つ定数」（`n` は縮約要素数）を `mul`、または `g` を `n` で `div` してからブロードキャスト（既存 `mean_vjp_f64` と同じ除算後ブロードキャスト。2 要素なら `[g/2, g/2]`） | 足りる（`mul`／`div` は既存 8 演算内） | なし | スカラー `g` に限る。`1/n` 係数を落とすと `sum` と同じ `[g, g]` になり誤り |
| `sum(Some(axis))`／`mean(Some(axis))` | 縮約した軸を `g` に挿し戻す（`unreduce_broadcast_f64` 1291 行相当）。`mean` は挿し戻しの前後いずれかで縮約軸長 `n` による `1/n` 係数（`div` または `mul`）を掛ける | | 足りない | `reshape` と `broadcast_to`（または `Unreduce`／`ReduceTo` の対） | 右詰めブロードキャストで偶然通るのは先頭軸だけ |
| `max` | first-match の one-hot マスク（`max_first_match_vjp_f64` 1356 行）を定数にして挿し戻す | 足りない | 上と同じ | f32 側が対象外なので揃えて対象外にするのが妥当（推測） |

結論: `reshape`／`broadcast_to`（または `ReduceTo`／`Unreduce` の対）を新しい `OpF64` variant として足し、その VJP を `TapeF64::backward` に実装しないと、縮約系とブロードキャストを伴う演算は扱えない。同 shape の要素演算・`matmul`・スカラー縮約だけなら既存 8 演算で閉じる。

### 33.3 構造面の作業表

| # | 作業 | 内容 | 備考 |
|---|---|---|---|
| 1 | 子テープ相当の f64 構造 | f32 `create_graph.rs` の祖先収集・検査・ミラー・cgrads の f64 版。入口検査（同一・空・デバイス）の追加 | f32 側は 1621 行 |
| 2 | 結果型 | `CreateGraphResult<'c>` の f64 版 | ライフタイムが `'g, 't` の 2 つになる |
| 3 | `GradientsF64` の高階化 | 既存型は変えられない（破壊的変更）ため新しい型を足す | |
| 4 | `TypedOps<f64>` 拡張 | 不要 | ホスト経路で閉じる。`div`／`pow` のホスト前例が根拠。multiplexing 設計 §10 の項 2 は消費しない |
| 5 | f64 版 `jvp` 層 | 子テープ backward を 2 段呼ぶ | `functional_ops` の f32 版に対応 |
| 6 | facade 公開 | `TapeF64` の新メソッドと結果型。`compat-api-scope.md` §5 の手続き・保留ガード・`api_surface.rs` | 承認事項が増える |
| 7 | テスト | 閉形式・`jacobian` 相当の参照・拒否経路。CUDA／Metal の `#[ignore]` テストと `docs/perf/logs` README | Metal は `typed_ops_f64()` が `None` でホスト経路 |

### 33.4 規模の類推（過去実績）

数字は `git show --stat --format= <sha> | tail -1` で再現できる。

| 類推元 | コミット | 変更規模 |
|---|---|---|
| f32 create_graph 初版（elementwise／sum 系） | `43f8a8f0`（#1998） | 8 files, +1632 |
| f32 create_graph の matmul／Linear 拡張 | `f19f54e3`（#2003） | 10 files, +1339 |
| create_graph の上に載せた jvp／jacfwd 内部実装 | `a18afc40`（#2945） | 11 files, +861 |
| jvp／jacfwd の facade 公開 | `1c1e76f4`（#2957） | 9 files, +424 |
| `TapeF64` 初版（葉・elementwise 4 演算） | `de472912`（#2253） | 6 files, +1526 |
| `TapeF64` の matmul・縮約 4 演算 | `2f17c547`（#2255） | 6 files, +2124 |

推測: f64 版は create_graph 機構の構築、新しい構造演算、jvp 層、facade 公開の合計になり、上の f32 側の同種作業（4 PR 相当）を下回らない。新しい行数・時間の見積もりは、根拠がないため書かない。

### 33.5 検討から外した案

- per-Op の JVP 規則による直接の接ベクトル伝播: spec REQ-9（2026-10-08 追記・#2617・#2880）が「対象 Op への JVP 規則の追加を要するネイティブ forward-mode」を対象外としている。
- `Var::cast::<f64>` との往復や f32 `Tape::jvp` への委譲: cast は勾配が切れており、f64 精度の意味が失われる。
- `GradientsF64` 自体を高階化する案: 既存の公開面が変わるため不可。

### 33.6 (a)／(b) の比較

| 観点 | (a) 実装する | (b) f32 `Tape` 限定を契約として固定 |
|---|---|---|
| 規模 | 33.4 の推測どおり大きい（新 `OpF64` variant・create_graph の f64 版・facade 公開） | docs の更新のみ |
| 破壊的変更 | 追加のみで可能 | なし |
| 新しい依存・`unsafe`・`TypedOps` 拡張 | いずれも不要 | いずれも不要 |
| 承認事項 | facade の新公開面 2〜3 個・新 `OpF64` variant・`Max` の扱い・実機 parity 未実測での公開可否 | 契約固定の確認のみ |
| 利用者への影響 | f64 で `J·v` を直接取れる | `TapeF64::backward` のループで代替できる（出力要素数 m 回の backward） |
| 保守コスト | f32 と f64 で create_graph 機構が二重になる | 増えない |

### 33.7 推奨と所有者確認事項

推奨は (b)。根拠: 縮約系に新しい構造演算が要り、create_graph 機構を丸ごと f64 で作り直す規模になる（33.2〜33.4）。f64 で `J·v` が要るという利用実績も記録にない。D-A の「規模の小さい追加のみ」という条件にも当てはまらないため、(a) へは自動では進まない。

| # | 何を | 選択肢 | 推奨 |
|---|---|---|---|
| 1 | `TapeF64` の `jvp` の扱い | (a) 実装する／(b) f32 `Tape` 限定を契約として固定する | (b) |
| 2 | 契約の範囲 | `jvp` だけ／double-VJP 系全体（`jvp`・`jacfwd`・`jvp_multi`・`jvp_create_graph`）／それに加えて `vjp`・`hvp`・`vmap`・`vmap_grad` の `VarF64` 対象外も同時に契約化 | double-VJP 系全体（同じ create_graph 前提のため） |
| 3 | 再開条件 | 再開条件なしで固定／「f64 精度の `J·v` の利用実績で不足が確認された場合」を再開条件とする | 後者 |
| 4 | 代替手順の掲載先 | 設計記録だけ／それに加えて facade `TapeF64` の rustdoc（後続 issue で doctest 化） | 後者 |

所有者の確認コメントで確定する。それまでは記録のみで、コード・rustdoc・`compat-api-scope.md` は変更しない。

### 33.8 (b) の代替手順（`TapeF64` 利用者向け）

擬似コード・コンパイル未確認（後続作業で doctest 化する）。メソッド名は facade の `TapeF64`／`VarF64`／`GradientsF64` の宣言に合わせた。

```text
// y: 出力（要素数 m）、x: 入力（tf.var で登録）、v: x と同 shape のホスト値
for k in 0..m {
    let e_k = tf.var_no_grad(&one_hot(k, y.shape()));   // y と同 shape の定数
    let s_k = y.mul(&e_k)?;                              // 非スカラー loss のシードは全要素 1 の総和射影なので sum 不要
    let grads = tf.backward(&s_k)?;
    match grads.get(&x)? {
        Some(row) => jv[k] = dot(row, v),               // row = Jᵀe_k（J の k 行目）
        None => jv[k] = 0.0,                            // 到達しなかったのでゼロ行
    }
}
```

- 1 回ごとに `TapeF64` へ 2 ノードが増える。backward は m 回。
- `x` が `var_no_grad` なら `GradientTrackingDisabled`。
- `backward` は非破壊（33.1）なので、同じグラフに繰り返し呼べる。

### 33.9 後続作業（(b) 採用時。未起票。起票は所有者確認の後）

- facade rustdoc を「対象外」から「契約として対象外（#3074）」へ格上げする。対象は `crates/facade/src/lib.rs` の 903・955・1011・1255 行付近。`TapeF64` の rustdoc に代替手順の doctest を加える。
- `docs/compat-api-scope.md` §5.1 の F4〜F7 行と適用記録（2484・2664・2669 行付近）の「保留継続: `VarF64`」を「契約として対象外（#3074・所有者確認 URL）」へ更新する。
- 本記録の §23.3／§27.3 に「→ #3074 で契約化」の前方参照を足す。
- いずれも本 issue では実施しない（起票はユーザー承認後。`out-of-scope-tracking.md`）。

### 33.10 変更していないもの

`crates/**`・公開 API・`TapeF64` の公開面・tolerance・baseline・依存・`unsafe`・ガードレール閾値・`docs/spec`。


## 34. 実装記録（イシュー #3087・親 #3075。facade の高階変換で拡張 Op を確認）

#3077〜#3086 で `create_graph` 対象へ広げた Op 族の代表が、facade の公開入口 `Tape::backward_create_graph`・`hvp`・`jvp`・`jacfwd`・`jvp_multi` でも通ることをテストで固定した。公開 API・1 階 `backward` の値・tolerance・baseline・依存・`unsafe`・`docs/spec`・`crates/autodiff/src` は変えていない（承認範囲は #3020 の D-A の一括承認の中）。

### 34.1 追加したテスト

| ファイル | 内容 |
|---|---|
| `crates/facade/tests/create_graph_facade.rs` | `backward_create_graph` を 17 Op で確認: layer_norm・rms_norm・batch_norm・conv2d・gelu・gelu_tanh・huber_loss・cross_entropy_loss・gather・cumsum・pad・masked_fill・max_pool2d・avg_pool2d・inv・det・rfft。GPU `#[ignore]` は LayerNorm・Conv2d・MaxPool2d（CUDA／Metal）。Softmax・LogSoftmax・MseLoss・NllLoss は既存 |
| `crates/facade/tests/functional_transforms_facade.rs` | `hvp` の有限差分突合 6 Op（layer_norm・conv2d・gelu・gather・max_pool2d・inv）と、`Tape::custom` による拒否 1 件 |
| `crates/facade/tests/functional_transforms_jvp_facade.rs` | `jvp`／`jacfwd` を `Tape::jacobian` と突合 9 Op（softmax・layer_norm・conv2d・gelu_tanh・cumsum・pad・avg_pool2d・rfft・solve）と、`jvp_multi` の conv2d（入力と重みの両方を追跡）1 件 |

### 34.2 拒否の基準を `Tape::custom` に替えた理由

従来の拒否テストは `LogSumExp`・`Max` を基準にしていた。この 2 つは決定記録 §21.2 の C 区分（今回のラウンドで fail-closed に固定したが、将来の拡張で対象になりうる）で、恒久対象外ではない。恒久対象外（A 区分）の `Op::Custom` に替え、将来の拡張で拒否テストが黙って意味を失うのを防ぐ。`.is_err()` は `Err(AutodiffError::Backward(_))` の一致に強めた。

### 34.3 意味のないテストを防ぐ方針

- x について線形な Op は `sum(op(x))` だと Hessian が恒等的にゼロになり、二階の再生が壊れても通る。すべてのケースに非線形な包み（`tanh`・要素積・固定重み）を入れた
- LayerNorm・BatchNorm・CrossEntropy は 1 階勾配の総和が恒等的に一定になる（Hessian 列和がゼロ）ため、一様でない固定方向 `v` の `H v` で検証し、`H v` がゼロでないことも assert する
- `max_pool2d` は値がすべて異なり差が刻み（`eps = 1e-2`）より十分大きい入力、Huber は `|tanh(x) - target|` が `delta` の折れ目をまたがない入力にした

### 34.4 数値の比較方法

- `first_order()` は素の `Tape::backward` と bit 一致（全 Op で成立）
- 子テープ上の `cg.grad` は §9 (i) のとおり REQ-2 が契約。実測で bit 一致した Op は bit 一致で固定し、1 ulp 差が出た gelu・gelu_tanh・inv・det（および損失の huber・cross_entropy）は REQ-2 で比べる
- 二階は `backward_create_graph` を通らない独立経路（素の `backward` の 1 階勾配の中央差分）と REQ-2 で比べる。tolerance は新設・緩和していない

### 34.5 GPU `#[ignore]` と申し送り

CUDA は `create_graph_extended_ops_cuda_matches_cpu`、Metal は `create_graph_extended_ops_metal_matches_cpu`。実行手順と記入欄は `docs/perf/logs/create-graph-extended-ops-3087/README.md`。実測は #3168・#3169 へ申し送る。線形代数は forward が GPU からホスト f64 にフォールバックするため代表にしていない（`autodiff-higher-order-grad-decision.md` §31.6 の申し送り 5）。

### 34.6 変えていないもの

公開 API・1 階 `backward` の値・tolerance・baseline・FMA 契約・依存・`unsafe`・`docs/spec`・`crates/autodiff/src`。C 区分・`Conv3d` 系・損失の残り variant を対象に加えていない。rustdoc は対象 Op の正本（`supports_create_graph` の網羅 match と決定記録 §21.2・§22〜§31）を示す一文だけを足した。

## 35. 決定記録（イシュー #3722・親 #3719。f64 テープの create_graph 機構）

### 35.0 位置づけ・承認根拠・境界・推奨

- 状態: **設計記録のみ。推奨は「実装する（案 A）」で、承認は #3766（論点 2「`TapeF64` の `jvp` 系」・論点 3「f64 の数値契約」）の回答待ち**。本節は承認を代行しない。
  - → #3785 追記: 承認は https://github.com/Fandhe-AI/fandhe-ai/issues/3766#issuecomment-6554563674（2026-11-01）で得られた。第 1 段の実装記録は §37。上の「本節は承認を代行しない」と、承認前の方針（回答が出るまでコード・rustdoc・公開 API・`docs/compat-api-scope.md` を変えない）は、この承認で役目を終えた。
- 基準: `origin/main` `69881d9da`。以下の行番号・規模はこの sha 上で再現できる。
- 根拠: §33（#3074 の規模見積もり）と、ルート #3711 の起票経緯コメント（<https://github.com/Fandhe-AI/fandhe-ai/issues/3711#issuecomment-6542000986>・2026-10-31。27 項目は「実装」を既定として設計する）。後者は設計方針の根拠であり、承認ではない。
- 推奨: **実装する（案 A）**。§33.7 の (b)「f32 `Tape` 限定で契約固定」は本節で改める。§33 の規模の議論（縮約系に構造演算が要る）は否定しない。そのうえで次を理由にする。
  - 写しを f64 専用に複製し、対象 Op を 8 つに限るので、規模に上限がある（35.10・35.11）。
  - 2 段に分けて実装できる（35.11）。
  - 所有者方針が実装既定である。
- 訂正の注記: §33.2 の表の `sum(Some(axis))`／`mean(Some(axis))` 行は列が 1 つずれている。訂正は本節で行い、§33 の本文は書き換えない。
- 境界: 本節が扱うのは create_graph 機構だけ。4 関数（`jvp`／`jacfwd`／`jvp_multi`／`jvp_create_graph`）の公開形・シグネチャ・`FunctionalTransformsHoldDoctestGuard` の扱いは #3723 が扱う。

### 35.1 公開形

内部シグネチャ案（autodiff。名前・形は #3766 の回答まで案）:

```rust
impl<'t> TapeF64<'t> {
    pub fn backward_create_graph<'c>(
        &self,
        loss: &VarF64<'_, 't>,
        child: &'c TapeF64<'t>,
    ) -> Result<CreateGraphResultF64<'c, 't>, AutodiffError>;
}
```

- 結果型 `CreateGraphResultF64<'c, 't>` は、`grad(&VarF64) -> Result<Option<VarF64<'c, 't>>, _>` と `child_var(..)` を持つ（f32 の `CreateGraphResult` と同じ意味論）。jvp の合成のため、内部向けの `mirror()` アクセサも持つ。
- 子テープの `'t` は親と共有する。`TapeF64<'t>` は `'t` について共変なので、同じ f32 `Tape` を借用した 2 本の `TapeF64` を親子にできる。別の `'tc` にする案は 35.13 の論点に載せる。
- 置き場: `crates/autodiff/src/f64_autograd/create_graph.rs` を、`f64_autograd.rs` の子モジュールとして新設する。`f64_autograd.rs` のパスは変えない。子モジュールは祖先の private 項目（`NodeF64`・`OpF64`・`binary_partials_f64` 等）に触れられる。結果型は `f64_autograd` から `pub use` し、`fandhe_ai_autodiff::f64_autograd::CreateGraphResultF64` の完全修飾パスを保つ。
- facade 配置案: D-2 形の newtype（`fandhe_ai::CreateGraphResultF64<'c, 't>`。フィールドは `pub(crate)`）とし、`TapeF64::backward_create_graph` へ 1 式で委譲する。`f64_autograd_not_reexported_from_facade` が `f64_autograd` 経由の `pub use` を禁じているため、f32 の `CreateGraphResult` のような再エクスポートにはできない。
- `first_order` の制約: newtype 越しに `&fandhe_ai::GradientsF64` を返すことは `unsafe` なしではできない。案は 3 つ。
  - (a) facade の結果型が facade の `GradientsF64` を所有する（内部に分解用メソッドを置く）。
  - (b) `first_order_get(&VarF64)` で値を素通しする。
  - (c) 省いて `TapeF64::backward` の再呼び出しで代替する。
  - 推奨は (a)。`repr(transparent)` キャストは使わない。
- 内部の新 `OpF64` variant はすべて `pub(crate)`。`VarF64` の公開メソッド（`F64_AUTOGRAD_SURFACE` の 15 件）は増やさない。
- 正ガードの反転・更新の対象（すべて `crates/facade/tests/api_surface.rs`。実装 PR で行う）:

| ガード | 更新内容 |
|---|---|
| `F64_AUTOGRAD_SURFACE` | `TapeF64::backward_create_graph` と新しい型を追加する |
| `f64_autograd_declared_once_in_lib_with_crate_private_field`・`f64_autograd_expected_decl` | 4 型目を追加する |
| `f64_autograd_public_surface_is_exactly_approved_delegations` | 承認済み委譲に追加する |
| `f64_autograd_not_reexported_from_facade`・`f64_autograd_reexport_violations` | 監視対象に新しい型名を足す |
| `f64_autograd_struct_declarations_workspace_inventory` | 新ファイルの宣言を足す |
| `f64_autograd_signature_probe` | 正のプローブを足す |
| `f64_autograd_guards_detect_each_violation_class`（自己テスト） | すり替えの検出例を足す |
| `workspace_declares_backward_create_graph_only_in_approved_locations` | 期待マップを 2 件から 4 件にする（または別名にする） |
| `facade_tape_backward_create_graph_is_thin_delegation` | `determinism_fn_body` は lib.rs に同名の `pub fn` が 2 件あると `None` を返す。同名を採るなら抽出を `impl Tape` スコープに限る |

- 否定のまま残すもの: `VarF64` 上の jvp 系の名前（#3723 の範囲）、`f64_autograd` モジュールの再エクスポート、`TapeF64` の `hvp`／`vjp`／`vmap`。
- #3723 への申し送り: `scan_functional_transforms_reexports_and_declarations` は `impl Tape` 直下だけを例外にしているので、`impl TapeF64` への拡張が要る。

### 35.2 rustdoc と doctest の例

- `TapeF64::backward_create_graph` と結果型に、役割の要約と CPU だけで通る doctest を置く。例: `f(x) = sum(x*x*x)` の二階 `6x` を `child.backward` で読む。
- GPU が要る例は置かない。置く場合は `no_run`／`#[ignore]` にする。
- 失敗系は `compile_fail` に頼らない。正のプローブ・インベントリ・拒否テストで組む。
- §33.8 の代替手順（f32 `Tape` へ載せ替える案内）は、案 A が採られれば不要になる。その場合は rustdoc 化しない。

### 35.3 エラー型

- 新しい variant は足さない。`TapeMismatch`・`DeviceMismatch`・`Backward`・`GradientTrackingDisabled`・`Shape`・`InvalidArgument` を再利用する。本番経路で `unwrap`／`expect` は使わない。
- 入口検査は次の順序で固定し、子テープへ書く前・確保の前に行う。
  1. `loss` が `self` に属さない: `TapeMismatch`
  2. 子テープが親と同じ: `Backward`（`RefCell` の二重借用による panic も防ぐ）
  3. f32 側 `Tape` の device が親子で違う: `DeviceMismatch`
  4. 子テープが空でない: `Backward`（内部に `is_empty` を足す）
  5. 祖先の収集と検査。範囲外 Op は `Backward`
  6. 無変更の `self.backward(loss)` で 1 階勾配を得る
  7. 写しと勾配グラフを構築する。途中で失敗したら内部の `rollback_to(0)` で子テープを空に戻す
- checkpoint 検査は該当なし（`TapeF64` に checkpoint がない）。
- 検査対象は「`requires_grad` かつ葉でない祖先」だけにする（§21.6 と同じ）。

### 35.4 テスト計画

実機の列はすべて「未実測」。

| 区分 | 置き場 | 内容 |
|---|---|---|
| CI | `crates/autodiff/tests/f64_create_graph.rs`（新設） | `first_order()` が `TapeF64::backward` と bit 同一 |
| CI | 同上 | `cg.grad` と 1 階勾配の一致（bit 一致を実測できた Op は bit で固定し、ほかは REQ-2。§34.4 の先例） |
| CI | 同上 | 閉形式の二階（`x*x*x`・`x@x`〈matmul の両オペランドを追跡〉・`div`・`pow`） |
| CI | 同上 | 1 階勾配の中心差分との REQ-2 突合 |
| CI | 同上 | 閉包性: 子テープからさらに孫テープへ create_graph する |
| CI | 同上 | 入口検査 4 種と範囲外 Op（`Max`）の拒否。拒否後に `child.is_empty()` であること |
| CI | 既存 f64 テスト | 無変更で通ること |
| CI | `crates/facade/tests/f64_create_graph_facade.rs`（新設） | 到達性とシグネチャ |
| `#[ignore]` | 同ファイル | `f64_create_graph_cuda_matches_cpu`（CUDA は `TypedOps<f64>` のネイティブ経路）。未実測 |
| `#[ignore]` | 同ファイル | `f64_create_graph_metal_matches_cpu`（Metal は `typed_ops_f64()` が `None` でホスト経路のみ）。未実測 |

- 実機の記入欄は `docs/perf/logs/f64-create-graph-3786/README.md`（実装 PR で新設）。実測は Phase 10 で行う。
- 線形の Op だけでは Hessian が 0 になりテストが意味を失うので、非線形な包みを入れる（§34.3 の方針）。

### 35.5 数値契約（#3766 論点 3 への回答案）

- REQ-2 の統一複合判定（相対誤差 1e-3 未満 または 絶対誤差 1e-5 未満）をそのまま使う。比較は既存の `fandhe_ai_backend_cpu::assert_parity_f64`／`compare_f64`（`crates/backend-cpu/src/parity.rs`）で行う。autodiff 側のテストは `tensor-core` の tolerance 定数を使い、新しい定数は作らない。
- 層:
  - (N1) `first_order` は構造上 bit 同一。
  - (N2) `cg.grad` 対 1 階勾配は REQ-2 が契約。CPU では逆数形（`g*(1/b)`）・同じホスト関数（`reduce_to_shape_f64`・`unreduce_broadcast_f64`）・同じ gemm dispatch を使うので bit 一致が見込めるが、これは推測であり契約にしない。
  - (N3) 二階の値対独立参照（閉形式・f64 中心差分）は REQ-2。
  - (N4) バックエンド間は REQ-2。`#[ignore]` で行う。
- FMA 契約・f64 長軸縮約契約への影響はない（f64 のままで、`host_gemm_f64` の `mul_add` を流用するため）。
- baseline は追加しない。単一の連続ループの経路で、split-K 等の対象外。tolerance・ガードレール閾値も変えない。
- f64 専用のより厳しい判定は採らない。新しい判定契約になり別承認が要るため、35.13 の論点に載せる。

### 35.6 互換への影響

`docs/api-compat-0.14.0-decision.md` §8 の手順と §1.2 の行 2 に照らして判定する。

| 対象 | 変更の種類 |
|---|---|
| `TapeF64` | メソッド追加（追加のみ） |
| `CreateGraphResultF64` | 新しい型（追加のみ） |
| `VarF64`・`GradientsF64` | 変更なし |
| `AutodiffError` | variant を足さないので該当なし |
| 内部の `OpF64`（`pub(crate)`） | 該当なし |

結論は「追加のみ」。移行ノート（#3780）は「移行作業なし」と書く。

### 35.7 適用記録・索引・追跡表

- `docs/compat-api-scope.md` へ書く適用記録の案（**承認後の PR で追記する。本 PR では書かない**）:
  - 見出し: f64 テープの create_graph 機構（`TapeF64::backward_create_graph`・`CreateGraphResultF64`）
  - 公開状態: 未公開
  - 根拠: 本節（§35）・#3766 の回答
  - チェックボックス: 未チェック（承認後の PR で追記）
- 本 PR で行うこと: `docs/README.md` の索引と、`docs/compat-matrix-ari-tracker.md` の # 2 行の「設計記録」セルの更新。承認・正ガード・0.14.0 の区分のセルは変えない（追跡表 §4 規則 1・6）。

### 35.8 ベンチと性能への影響

- 既定経路（f32 `Tape` 全体、既存 8 Op の `TapeF64::backward`）は変えない。`TapeF64::backward` に足すのは、内部の新 variant 向けの match 腕だけで、利用者が組むグラフには現れない。
- 前後比較は計画しない。実装で既存腕に手を入れる場合に限り、5 回計測の中央値で前後比較し、規則は計測前に固定する。

### 35.9 依存の追加の有無

依存の追加はない。許容 12 区分・禁止リストとの照合で変化はない。新しい `unsafe` はない（35.1 で `repr(transparent)` キャストを採らないのはこのため）。

### 35.10 機構の写し方（決定事項 1）

- 方式の写し:
  - V1（追跡される写しを係数にした `VarF64` 演算の合成）を基本にする。
  - V2（親の記録値から作るホストマスク）は `Pow` の `a==0`／`b==0` の安全置換に使う。f32 の `create_graph.rs` の `where_cond`＋`log` と同じ方式。
  - V3 は不要。f32 で V3 が要ったのは、1 階が f64 ホストで子が f32 という精度差のためで、f64 ではこの差がない。縮約は、1 階と同じホスト関数を forward にする内部 variant で揃える。
- 写しは full mirror にする。係数を `var_no_grad` の定数にする §33.2 の方式は `jvp`／`jacfwd`／`jvp_multi` には足りるが、`jvp_create_graph` が子テープを孫テープへ create_graph するとき、入力の写しについて微分できなくなる。#3723 の 4 関数すべてを満たすため full mirror を採る。
- 内部の新 variant（すべて `pub(crate)`）:

| variant | forward | VJP |
|---|---|---|
| `Transpose`（rank 2） | 転置 | `Transpose` |
| `Unreduce{dim}` | `unreduce_broadcast_f64` | 既存の `Sum{dim}` |
| `BroadcastTo` | ブロードキャスト | `ReduceTo` |
| `ReduceTo` | `reduce_to_shape_f64` | `BroadcastTo` |
| `Log` | 対数 | `Div` |
| `Where` | 定数の bool マスク | `Where` とゼロ定数 |

- 閉包性: 子テープへ積む Op はすべて f64 create_graph の対象であり、各 variant の VJP は集合の中で書ける（§21.1 (4) の f64 版）。
- fail-closed: `supports_create_graph_f64` をワイルドカードなしの網羅 match にし、新 variant を追加したら分類を強制する。
- 共通化か複製か:

| 観点 | 複製（推奨） | ジェネリック化 |
|---|---|---|
| 変更範囲 | f64 専用の小さな写し。f32 の機構は無変更 | f32 の `Op`（123 variant）・`push_eager`・`Var` に結びついた機構を `Tape`／`TapeF64` で統一する |
| f32 への影響 | なし（bit 同一契約を損なわない） | bit 同一契約を損なうリスクがある |
| 位置づけ | `autodiff-var-dtype-multiplexing-design.md` §5 の方針と矛盾しない | 同 §5 で着手しないとされた案 A そのもの |
| 保守コスト | f64 の Op 集合は 8 で固定。f32 の C 区分・B 区分の拡張は f64 に波及しない。二重化するのは骨格だけ | 統一の継続コストが f32 側の拡張ごとに発生する |

- 共有するのは形（入口検査の順序・結果型の API・テスト fixture の作り方）だけ。行数や時間の見積もりは、根拠がないため書かない（§33.4 の方針を継ぐ）。

### 35.11 対象 Op と段分け（決定事項 2・4）

| Op | 方式 | 必要な内部 variant | 段 |
|---|---|---|---|
| `add`／`mul`／`div`（同 shape とブロードキャスト） | V1 | `BroadcastTo`／`ReduceTo` | 第 1 段 |
| `pow` | V1＋V2 | `Log`／`Where` | 第 1 段 |
| `matmul`（rank 2） | V1 | `Transpose` | 第 2 段 |
| `sum`／`mean`（`None` と `Some(axis)`。`1/n` 係数） | V1 | `Unreduce` | 第 2 段 |
| `max` | fail-closed | なし | 対象外 |

- 段分けは起票済みの実装 issue の題名（#3785「基盤と add・mul・div・pow」、#3786「matmul・sum・mean・max と閉包性」）に揃えた。両 issue の受け入れ条件は本記録どおりの実装なので、題名と中身がずれたときの正は本節とする。
- 第 1 段の variant 集合（`BroadcastTo`／`ReduceTo`／`Log`／`Where`）で閉じる。`Log` の VJP は `Div`、`Where` の VJP は `Where`、`BroadcastTo` と `ReduceTo` は互いの VJP になる。
- 第 1 段のテストは `sum` を使わない。非スカラー loss は全要素 1 のシードで逆伝播するため（`f64_autograd.rs` の `Tensor::full(&loss_shape, 1.0f64)`）、`sum` なしで組める。
- `max` は f32 の `Max` が C 区分で、#3724 で設計中のため揃えて fail-closed にする。#3786 の題名にある `max` は「拒否の固定」と読む。実装するかは #3724 の結論と 35.13 の論点 5 に従う。
- rank 3 以上の `matmul` は forward が拒否する。
- 第 1 段（#3785）: autodiff 内部の基盤と要素演算族。
  - 入口検査・祖先の収集と検査・写し・勾配グラフ・`rollback_to`・結果型。
  - `add`／`mul`／`div`／`pow` と `BroadcastTo`／`ReduceTo`／`Log`／`Where`。
  - 第 2 段の Op（`matmul`・`sum`・`mean`）と `max` は fail-closed。
  - CPU の CI テスト。
  - facade の公開面は変えない。既存の f64 正ガード群が無変更で通ること、1 階 `backward` が bit 同一であること。
- 第 2 段（#3786）:
  - `Transpose`／`Unreduce` と `matmul`・`sum`・`mean`。
  - `Max` の拒否テスト。
  - 全 variant の閉包性テスト（孫テープでの再生）。
  - facade newtype と 35.1 のガード反転。
  - `#[ignore]` の GPU テストとログ README。
  - 承認を前提に、`compat-api-scope.md` の適用記録。
- #3787・#3788 はこの機構の上で 4 関数を作る（#3723 の範囲）。

### 35.12 セキュリティ観点・スコープ外・Closes／Refs

- OWASP:
  - A03: shape と要素数は、確保の前に `checked_bytes_for_f64`／`checked_product` 相当で検査する。
  - A04／A08: 子テープへ書く前に fail-closed で検査し、迂回経路を作らない。
  - DoS: 写しは祖先の値を複製するのでメモリが増える。rustdoc に明記する。
  - 秘密情報・ホスト名は書かない。`unsafe` はない。
- スコープ外（起票は所有者承認後。無断で起票しない）:
  - 項目 2 の実機 parity を測る Phase 10 issue（追跡表は「該当 issue なし」のまま）
  - f64 の `Max` の create_graph（#3724 の結論待ち）
  - `TapeF64` の `hvp`／`vjp`／`vmap`
- 本 PR は `Closes #3722`。docs のみの設計 issue で実測を要しないため。`Refs #3766 #3723 #3785 #3786` を併記する。

### 35.13 所有者へ戻す論点（#3766 で問う）

| # | 論点 | 推奨 | 代替 | 確認がないときの扱い |
|---|---|---|---|---|
| 1 | 項目 2 の機構 | 実装する（full mirror） | 対象外（§33 (b)） | 実装へ進めない |
| 2 | facade に `TapeF64::backward_create_graph` と `CreateGraphResultF64` を公開するか | 公開する（f32 と対称） | 内部限定にし、jvp 4 関数だけを公開する | 実装へ進めない |
| 3 | 名前 | 同名（ガード 2 件を impl スコープ化する） | 別名 | 実装へ進めない |
| 4 | `first_order` の形 | (a) 所有 | (b) 素通し／(c) 省略 | 実装へ進めない |
| 5 | `Max` | f32（#3724）に揃えて fail-closed | f64 だけ先に V2 で実装 | fail-closed |
| 6 | `Pow` の `a==0`／`b==0` | `Where`＋`Log` の安全置換 | `a>0` に限り、ほかは `Err` | 実装へ進めない |
| 7 | 数値契約 | REQ-2 のまま | f64 専用の判定（別承認） | REQ-2 のまま |
| 8 | 子テープのライフタイム | `'t` を共有 | 独立の `'tc` | 実装へ進めない |
| 9 | Phase 10 の実機 parity issue | 起票する（起票は承認後） | 起票しない | 起票しない |

確認がないときの扱いは、実装へ進めず、保留のまま正ガード（否定ガード）を維持する。

### 35.14 変えていないもの

`crates/**`・公開 API・tolerance・baseline・依存・`unsafe`・ガードレール閾値・`docs/spec`・`docs/compat-api-scope.md`。

## 36. 決定記録（イシュー #3723・親 #3719。`TapeF64` の `jvp`／`jacfwd`／`jvp_multi`／`jvp_create_graph` の公開形）

### 36.0 位置づけ・承認根拠・境界・推奨

- 状態: **設計記録のみ。推奨は「実装する」**。承認は #3766 論点 2・3 の回答待ちで、本書は承認を代行しない。回答までコード・rustdoc・公開 API・`docs/compat-api-scope.md` は変えない
- 基準 sha: `origin/main` `919f44aef`
- 根拠: §33（#3074）、§35（#3722）、ルート #3711 の起票経緯コメント（実装を既定とする設計方針。承認ではない。§35.0 と同じ引き方）
- 境界: §35 の機構（`TapeF64::backward_create_graph`・`CreateGraphResultF64`・内部 variant・段分け）は再決定しない。§36 は §35.13 の論点 1・2・3・4・8 の回答に依存する（論点 1 が「対象外」なら §36 も成立しない）
- 推奨と理由: §35 で機構を実装する前提が立った。その上の 4 関数は f32 `functional_ops` の合成を写すだけで、新しい `OpF64` variant は要らない（36.10）。対応表の項目 2 を「ある」にするには 4 関数の公開が要る

§33.7 の論点への回答:

| §33.7 の論点 | 回答 |
|---|---|
| 1. 扱い | (a) 実装する。(b) f32 限定の契約固定は選択肢として残す |
| 2. 契約の範囲 | double-VJP 系の 4 関数をすべて実装する。`TapeF64` の `vjp`／`hvp`／`vmap`／`vmap_grad` は 29 項目の外なので「対象外のまま（契約化しない）」とし、否定ガードを維持する |
| 3. 再開条件 | (a) を採るなら該当なし。(b) が採られた場合に限り §33.7 の推奨（f64 の `J·v` の利用実績で不足が確認されたら再開）を使う |
| 4. 代替手順の掲載先 | (a) を採るなら §33.8 の代替手順は rustdoc 化しない（§35.2 と同じ）。(b) の場合は `TapeF64` の rustdoc に doctest として載せる |

### 36.1 公開形

シグネチャ案（f32 版と同形。名前と形は #3766 の回答まで案）:

```rust
impl<'t> TapeF64<'t> {
    pub fn jvp(&self, output: &VarF64<'_, 't>, input: &VarF64<'_, 't>,
               tangent: &Tensor<f64>, child: &TapeF64<'t>) -> Result<Tensor<f64>, AutodiffError>;
    pub fn jacfwd(&self, output: &VarF64<'_, 't>, input: &VarF64<'_, 't>,
                  child: &TapeF64<'t>) -> Result<Tensor<f64>, AutodiffError>;
    pub fn jvp_multi(&self, output: &VarF64<'_, 't>, inputs: &[&VarF64<'_, 't>],
                     tangents: &[&Tensor<f64>], child: &TapeF64<'t>) -> Result<Tensor<f64>, AutodiffError>;
    pub fn jvp_create_graph<'g>(&self, output: &VarF64<'_, 't>, input: &VarF64<'_, 't>,
                                tangent: &Tensor<f64>, child: &TapeF64<'t>, grandchild: &'g TapeF64<'t>)
        -> Result<JvpCreateGraphResultF64<'g, 't>, AutodiffError>;
}
// JvpCreateGraphResultF64<'g, 't>:
//   output(&self) -> VarF64<'g, 't>
//   child_var(&self, &VarF64<'_, 't>) -> Result<Option<VarF64<'g, 't>>, AutodiffError>
```

- f32 との対応: 引数の順序・戻り値の構造・`child`／`grandchild` を明示引数にする点は f32 の `Tape::jvp` 等（`crates/facade/src/lib.rs`）と同じ。`'t` の共有は §35.13 論点 8 と連動し、独立の `'tc` を採る場合はシグネチャが変わる
- 内部の置き場: `crates/autodiff/src/f64_autograd/functional_ops.rs` を新設し、内部 `TapeF64` の inherent メソッドとして実装する。結果型は `f64_autograd` から `pub use` する。自由関数にしない理由は、facade の本体を `self.0.jvp(..)` の 1 式にでき、`f64_autograd` トークンを完全修飾以外で facade に書かずに済むため
- facade の配置: 4 メソッドは `crates/facade/src/lib.rs` の既存の唯一の `impl<'t> TapeF64<'t>` ブロックへ足す（ガードが固有 impl をちょうど 1 件と固定している）。結果型は D-2 形の newtype `pub struct JvpCreateGraphResultF64<'g, 't>(pub(crate) ...)` とし `pub use` にしない（`f64_autograd_not_reexported_from_facade` が再エクスポートを禁じているため。§35.1 の `CreateGraphResultF64` と同じ）。`jvp_multi` の本体はスライスの写し（`inputs.iter().map(|v| &v.0).collect::<Vec<_>>()`）を含む 1 式にして本体表へ登録する
- 実装の分担（題名とずれたときは本節を正とする）: #3787 は autodiff 内部の `jvp`／`jacfwd`・CI テスト・f32 との突き合わせ（facade は変えない）。#3788 は内部の `jvp_multi`／`jvp_create_graph`・facade の 4 メソッドと newtype・下のガード更新・`#[ignore]` の GPU テスト・ログ README・`compat-api-scope.md` の適用記録

**ガードの反転・更新計画**（すべて `crates/facade/tests/api_surface.rs`。#3788 で行い、#3787 は k の autodiff 分だけ）:

| # | ガード（テスト名・定数） | 現状で壊れる理由 | 更新内容 |
|---|---|---|---|
| a | `F64_AUTOGRAD_SURFACE`・`f64_autograd_public_surface_is_exactly_approved_delegations` | 公開メソッドの完全一致と 1 式本体の固定 | `TapeF64` に 4 メソッド、`JvpCreateGraphResultF64` の項（`output`・`child_var`）を追加 |
| b | `f64_autograd_reexport_violations`・`f64_autograd_not_reexported_from_facade` | `ok_suffix` が `TapeF64｜VarF64｜GradientsF64` のみ受理 | 受理集合と監視名に `JvpCreateGraphResultF64` を足す（§35 の `CreateGraphResultF64` と合わせる） |
| c | `f64_autograd_declared_once_in_lib_with_crate_private_field`・`f64_autograd_expected_decl` | 新しい struct が対象外 | `pub(crate)` フィールド・derive なしの宣言を足す |
| d | `f64_autograd_struct_declarations_workspace_inventory` | 期待パスが `autodiff/src/f64_autograd.rs::<型>` に固定 | 型ごとの期待パスにする（新しい型は `f64_autograd/functional_ops.rs`） |
| e | `f64_autograd_signature_probe` | 新メソッドが無い | 4 メソッドと結果型 2 メソッドの関数ポインタを使う正のプローブを足す |
| f | `f64_autograd_guards_detect_each_violation_class` | 自己テスト | 新しい型・メソッドのすり替え（本体変更・2 件目の impl・trait impl）の検出例を足す |
| g | `scan_functional_transforms_reexports_and_declarations`（`facade_does_not_reexport_or_declare_functional_transforms`） | `impl Tape` 直下だけを例外にしており、同名 `fn jvp` が 2 件で違反になる（§35.1 の申し送り） | 承認表を（受け手, 名前）で引く形にし、`impl TapeF64` 直下で承認シグネチャと一致する 4 名を受け手ごとに最大 1 件まで差し引く |
| h | `FUNCTIONAL_TRANSFORMS_APPROVED_SIGNATURES`／`_BODIES`・`check_facade_tape_functional_transforms_shape`（`facade_tape_functional_transforms_are_approved_thin_delegations`） | `fn_signatures_by_name(..).len() == 1` を要求し、同名 2 件では本体抽出が `None` になる | 抽出を `impl Tape` の範囲に限る。f64 側は a・e で固定する |
| i | `..._detects_each_category`（g・h の自己テスト） | 既存の負例を維持する必要がある | `impl Var`／`impl Tensor`／裸の関数への同名追加は違反のまま検出。`impl TapeF64` の別シグネチャ・2 件目も違反として足す |
| j | `facade_tape_backward_create_graph_is_thin_delegation`・`workspace_declares_backward_create_graph_only_in_approved_locations` | §35 の担当 | §36 では変えない（§35.1 の計画に従う） |
| k | `workspace_declares_functional_transforms_fn_names_only_in_allowed_locations` | 期待マップが固定 | `facade/src/lib.rs::{4 名}` を 1 から 2 へ。`autodiff/src/f64_autograd/functional_ops.rs::{4 名}` を各 1 で足す（#3787 で `jvp`／`jacfwd`、#3788 で残り）。`vjp`／`hvp`／`vmap`／`vmap_grad` は不変 |
| l | `FunctionalTransformsHoldDoctestGuard`・`functional_transforms_hold_doctest_probe_body_matches_fixed_contract`・`FUNCTIONAL_TRANSFORMS_HOLD_PROBE_BODY` | `TapeF64`／`VarF64` のプローブはもともと無い | **反転対象なし**（§31・§32 と同じ書き方）。未承認経路の否定プローブ（`VarF64` 受け手の 8 名、`TapeF64` 受け手の `vjp`／`hvp`／`vmap`／`vmap_grad`）を足し、固定本文定数を更新する |
| m | `facade_reexports_jvp_create_graph_result_in_approved_shape` | 識別子の完全一致で判定するため `JvpCreateGraphResultF64` と衝突しない | 変えない（「F64 名は検出しない」負例の追加は任意） |

否定のまま残すもの: モジュール `functional_ops`・`f64_autograd` の再エクスポート、裸の自由関数、`VarF64`／`GradientsF64`／`Tensor<f64>` 上の同名メソッド、`TapeF64` の `vjp`／`hvp`／`vmap`／`vmap_grad`。

### 36.2 rustdoc と doctest

- 4 メソッドと結果型に、役割の要約・入口検査の順序・副作用（親へ 2 ノード）・ゼロ経路・数値（REQ-2。bit 同一は契約にしない）・計算量・メモリ（写しによる増加）を書く。構成は f32 版 rustdoc と同じ
- facade 単独で通る CPU の例: `y = x ⊙ x`・`v = [1, 0.5, 2]` で `jvp` が `[2, 2, 12]`、`jacfwd` が `diag(2x)`、`jvp_multi` で `x1 ⊙ x2`、`jvp_create_graph` で `grandchild.backward(&res.output().sum(None)?)` が `2v`。比較は絶対誤差 1e-5 以内（REQ-2 の救済項に揃え、f64 専用の厳しい値を暗に契約しない）。`expect` の代わりに `ok_or(..)?` を使う
- GPU の例は置かない（置くなら `no_run`）。失敗系は `compile_fail` に頼らず、正のプローブ・インベントリ・拒否テストで組む
- f32 側 rustdoc の「f32 の `Tape` のみ（`VarF64`・f16 は対象外）」は #3788 で「f64 版は `TapeF64::jvp` 等」への案内に直す（f32 のシグネチャは変えない）

### 36.3 エラー型

- 新しい variant は足さない（`AutodiffError` は `#[non_exhaustive]`）。`TapeMismatch`・`GradientTrackingDisabled`・`Shape`（`ShapeMismatch`／`ElementCountOverflow`）・`InvalidArgument`・`Backward`・`DeviceMismatch` を再利用する。本番経路で `unwrap`／`expect` を使わない
- 入口検査の順序は f32 版の rustdoc を写して固定する
  - `jvp`: (1) 別テープ → `TapeMismatch` (2) 追跡なしの入力 → `GradientTrackingDisabled` (3) `tangent` の shape 不一致 → `ShapeMismatch` (4) 要素数の検査付き乗算 → `ElementCountOverflow` (5) 要素数 0 または追跡なしの `output` はテープに触れず全ゼロ (6) 以降は §35.3 の `backward_create_graph` の検査（同一の子 → `Backward`、デバイス不一致 → `DeviceMismatch`、空でない子 → `Backward`、範囲外 Op〈`Max` 等〉 → `Backward`）を伝播
  - `jacfwd`: `tangent` の検査を除き同じ。`m×n` は検査付き乗算
  - `jvp_multi`: 長さ不一致・空 → `InvalidArgument`、別テープ、追跡なし、重複 → `InvalidArgument`、shape、要素数の順
  - `jvp_create_graph`: (1)〜(4) のあとテープの事前検査（同一 → `Backward`、デバイス → `DeviceMismatch`、空でない → `Backward`）。ここまでの失敗では 3 枚のテープとも無変更
- 副作用: 入口検査を通った後は、失敗しても親へちょうど 2 ノード（全要素 1 の追跡ありの葉 `u` と `mul`）が残る。f32 と同じ
- 確保の前に検査する（A03）。結果のバイト数・`m×n` は `checked_*` で求める

### 36.4 テスト計画

| 受け入れ基準 | テスト | 区分 | 担当 | `<cuda-node>` | `<metal-node>` | `<rocm-node>` |
|---|---|---|---|---|---|---|
| f32 と f64 の突き合わせ | `crates/autodiff/tests/f64_functional_transforms.rs`（新設）。同じグラフを `Tape` と `TapeF64` に組み、入力は f32 で厳密に表せる値にする。f32 の結果を f64 へ広げ `compare_f64`／`assert_parity_f64`（REQ-2）で比較。対象は §35.11 の範囲（同 shape・ブロードキャストの `add`／`mul`／`div`／`pow`〈a>0〉、`matmul` rank 2、`sum`／`mean`） | CI | #3787（jvp・jacfwd）／#3788（残り） | 該当なし | 該当なし | 該当なし |
| 独立参照との一致 | 同上。§33.8 の「backward の繰り返し」で作る f64 ヤコビアンとの `J·v` の一致、閉形式との一致 | CI | 同上 | — | — | — |
| 関数間の整合 | `jacfwd` と one-hot 接ベクトルの `jvp` の列の一致、入力 1 つの `jvp_multi` と `jvp` の一致（bit 同一は主張しない）、`jvp_create_graph().output()` と `jvp` の一致、孫での二階と閉形式の一致 | CI | 同上 | — | — | — |
| 拒否と順序 | 入口検査の各種と順序。`Max` の拒否で f32・f64 とも `Backward`。失敗時に親へ 2 ノード残り、子・孫は無変更（`is_empty`） | CI | 同上 | — | — | — |
| 既定経路が bit 同一 | 既存の f64・f32 テストが無変更で通る | CI | 両方 | — | — | — |
| 到達性・シグネチャ | `crates/facade/tests/f64_functional_transforms_facade.rs`（新設） | CI | #3788 | — | — | — |
| GPU | 同ファイル `f64_jvp_family_cuda_matches_cpu`（`TypedOps<f64>` のネイティブ経路） | `#[ignore]` | #3788 | 未実測 | — | — |
| GPU | 同ファイル `f64_jvp_family_metal_matches_cpu`（Metal はホスト経路のみ） | `#[ignore]` | #3788 | — | 未実測 | — |
| ROCm | 該当なし（ROCm は facade 未公開。項目 24） | — | — | — | — | 該当なし |

- 実機の記入欄は `docs/perf/logs/f64-jvp-facade-3788/README.md`（#3788 で新設）。実測は Phase 10。追跡表の行 2 の実機 parity は「該当 issue なし」なので、起票は承認後の論点にする（§35.13 論点 9 と統合）
- 線形の Op だけでは二階が 0 になりテストの意味がなくなるので、非線形な包みを入れる（§34.3）

### 36.5 数値契約

- REQ-2 の統一複合判定（相対誤差 1e-3 未満 または 絶対誤差 1e-5 未満）をそのまま使う。比較関数は `fandhe_ai_backend_cpu::{compare_f64, assert_parity_f64}`。tolerance 定数は `tensor-core` のものを使い、新しい定数は作らない
- f32 と f64 の突き合わせも REQ-2 で判定する（f32 側の丸めを含む比較なので f64 専用の厳しい判定は採らない。#3766 論点 3 の推奨）
- FMA 契約・f64 長軸縮約の契約への影響はない（f64 ホスト・`host_gemm_f64` のまま）
- baseline は追加しない（単一の連続ループの経路で split-K 等の対象外）。ガードレール閾値も変えない

### 36.6 互換への影響（`docs/api-compat-0.14.0-decision.md` §8 の手順と §1.2 の行 2 で判定）

| 対象 | 種類 |
|---|---|
| `TapeF64` | inherent メソッド 4 つの追加（追加のみ） |
| `JvpCreateGraphResultF64` | 新しい型（追加のみ。フィールドは `pub(crate)` で構築不可） |
| `VarF64`・`GradientsF64` | 変更なし |
| f32 の `Tape`・`JvpCreateGraphResult` | 変更なし（rustdoc の案内文だけ） |
| `AutodiffError` | variant を足さないので該当なし |

結論は「追加のみ」。移行ノート（#3780）には「移行作業なし」と書く。

### 36.7 適用記録・索引・追跡表

- `compat-api-scope.md` の適用記録の案（**承認後の #3788 で追記する。本 PR では書かない**）: §5.1 の F ブロックに `F9`〜`F12`（`TapeF64::jvp`／`jacfwd`／`jvp_multi`／`jvp_create_graph`）と結果型の行を足す。公開状態は「未公開」、根拠は §36 と #3766 の回答 URL。F4〜F7 の「保留継続: `VarF64`」は本文を書き換えず、追記で「→ #3788 で公開」と前方参照する
- 本 PR で行うこと: `docs/README.md` の索引の追記と、`docs/compat-matrix-ari-tracker.md` の行 2 の設計記録セルの更新のみ

### 36.8 ベンチと性能への影響

- 既定経路（f32 `Tape` 全体と `TapeF64::backward`）は変えない。4 関数は新しい入口で既存の呼び出しに影響しないため、前後比較は計画しない
- 計算量: `jvp`／`jvp_multi` は親 backward 1 回＋子 backward 1 回。`jacfwd` は子 backward を n 回。`jvp_create_graph` はそれに孫への記録 1 回を足す
- メモリ: full mirror（§35.10）で祖先の値が複製される。rustdoc に明記する

### 36.9 依存の追加の有無

なし。許容 12 区分・禁止リストとの照合で変化はない。新しい `unsafe` もない（newtype の `repr(transparent)` キャストは採らない）。

### 36.10 算法の写し方（f64 固有の決定）

- `jvp`: 親に `u`（全要素 1）と `s = output ⊙ u` を足し、`backward_create_graph(s, child)` から `g = cg.grad(input)` と `cu = cg.child_var(u)` を得る。`child.backward(g ⊙ v_const)` の `cu` 勾配を返す。既存の `VarF64` 演算だけで閉じる
- `jacfwd` の列の取り出し: f32 の `FlatElements`（`contiguous`／`reshape`／`narrow`）は `VarF64` に無く、§35.10 の variant 集合にも無い
  - **推奨 J1**: 列 k ごとに `child.var_no_grad(e_k)`（入力 shape の one-hot 定数）を作り `g ⊙ e_k` から `child.backward` する。非スカラー loss のシードは全要素 1 なので `sum` は不要（§33.8 と同じ）。各列の後、§35.3 で #3785 が導入する内部 `rollback_to` で `g` 構築時点の印まで子テープを戻し、メモリを O(n) に抑える
  - J2: 戻さない。子テープに 2n ノードと n² 要素の定数が積もる
  - J3: 内部に `Narrow`／`Reshape` の variant を足す。#3722 の variant 集合の変更になるため所有者へ戻す
- `jvp_multi`: 入力ごとの shape が異なると `add` できないため、各 `g_i ⊙ v_i` を `sum(None)` でスカラーにしてから `add` で畳み、子 backward を 1 回で済ませる。子側は 1 階の `TapeF64::backward` だけなので create_graph の対象 Op は増えない
- `jvp_create_graph`: 2 段目は `child.backward_create_graph(g ⊙ v, grandchild)`（§35.10 の閉包性に依存）。写しの合成（`cg1.mirror()` を `cg2.child_var` で引き継ぐ）は f32 の `functional_ops::jvp_create_graph` と同じ。第 2 段（#3786）が前提
- 依存順は #3785 → #3786 → #3787 → #3788（issue の依存と一致）

### 36.11 セキュリティ観点・スコープ外・Closes／Refs

- OWASP
  - A03: shape・要素数・`m×n`・バイト数を確保の前に検査する
  - A04／A08: 子・孫テープへ書く前に fail-closed で検査し、迂回経路を作らない
  - DoS: 写しと `jacfwd` のメモリを rustdoc に明記する
  - 秘密情報・ホスト名は書かない（`<cuda-node>` 等で表す）。`unsafe` はない
- スコープ外（起票は所有者承認後。無断で起票しない）: `TapeF64` の `vjp`／`hvp`／`vmap`／`vmap_grad`、f64 の `Max` の create_graph（#3724 の結論待ち）、項目 2 の Phase 10 実機 parity issue
- 本 PR は `Closes #3723`（docs のみの設計 issue で実測を要しないため。#3884 が #3722 を閉じた先例と同じ）。`Refs #3766 #3722 #3787 #3788` を併記する

### 36.12 所有者へ戻す論点（#3766 論点 2 で問う）

| # | 論点 | 選択肢 | 推奨 | 確認がないときの扱い |
|---|---|---|---|---|
| 1 | 4 関数を実装・公開するか | (a) 実装・公開／(b) f32 限定の契約で固定（§33.7） | (a) | 実装へ進めない（保留ガードを維持） |
| 2 | 公開の範囲 | 4 関数と `TapeF64::backward_create_graph` を公開／4 関数だけを公開（機構は内部。後者でも `jvp_create_graph` の結果型は公開が要る） | §35.13 論点 2 に従う | 実装へ進めない |
| 3 | 名前 | 同名（ガード g・h・k を受け手の範囲で引く）／接尾辞（`jvp_f64` 等） | 同名 | 実装へ進めない |
| 4 | 子・孫テープの型とライフタイム | `&TapeF64<'t>` で `'t` を共有／独立の `'tc` | 共有（§35.13 論点 8 と連動） | 実装へ進めない |
| 5 | `jvp_create_graph` の結果型 | facade newtype `JvpCreateGraphResultF64<'g,'t>`／`jvp_create_graph` を見送る | newtype | 実装へ進めない |
| 6 | `jacfwd` の列の取り出し | J1／J2／J3（新 variant。#3722 の範囲変更） | J1 | 実装へ進めない |
| 7 | 追跡なしの `output`・失敗時に親へ残る 2 ノード | f32 と同じ（全ゼロ・2 ノード）／変える | f32 と同じ | 設計は f32 と同じで保つが、実装へは進めない |
| 8 | `TapeF64` の `vjp`／`hvp`／`vmap`／`vmap_grad` | 対象外のまま（否定プローブを追加）／同時に設計 | 対象外のまま | 対象外のまま |
| 9 | f32 と f64 の突き合わせの判定 | REQ-2／f64 専用の判定（別承認） | REQ-2（#3766 論点 3） | REQ-2 |
| 10 | Phase 10 の実機 parity issue | 起票（承認後）／起票しない | 起票（§35.13 論点 9 と統合） | 起票しない |

### 36.13 変えていないもの

`crates/**`・公開 API・tolerance・baseline・依存・`unsafe`・ガードレール閾値・`docs/spec`・`docs/compat-api-scope.md`。


## 37. 実装記録（イシュー #3785・親 #3719。f64 create_graph 第 1 段）

### 37.0 位置づけ

- 承認: §35.0 の承認待ちは、#3766 の所有者の回答コメント（2026-11-01、https://github.com/Fandhe-AI/fandhe-ai/issues/3766#issuecomment-6554563674）で論点 2 (a)（full mirror・対象 8 Op・2 段）と論点 3（REQ-2 の判定式をそのまま使う）が承認された。§35.13 の各論点は推奨どおり。本節は §35.11 の第 1 段の実装記録である。
- 基準: `origin/main` `4e19fdff2`。第 2 段（`matmul`／`sum`／`mean`、`Transpose`／`Unreduce`）と facade の newtype、実機の `#[ignore]` テストは #3786。

### 37.1 追加したもの

- `crates/autodiff/src/f64_autograd/create_graph.rs`（新設）: `TapeF64::backward_create_graph`（入口検査は §35.3 の順序）、`CreateGraphResultF64`（`first_order`／`grad`／`child_var`）、祖先の収集と検査、写し（full mirror）、勾配グラフ、失敗時の `rollback_to`。
- `crates/autodiff/src/f64_autograd.rs`: 内部 variant `BroadcastTo`／`ReduceTo`／`Log`／`Where`（`pub(crate)`）、それらの 1 階 VJP 腕、crate 内部の `is_empty`／`rollback_point`／`rollback_to`、内部 forward（`cg_broadcast_to`／`cg_reduce_to`／`cg_log`／`cg_where`。公開メソッドは増やさない）。既存の腕は変えていない。
- 範囲外の Op（`matmul`／`sum`／`mean`／`max`）は追跡対象の祖先にあれば `Backward` で拒否する（子テープへ書く前）。`max` は fail-closed のまま。定数側の部分木は検査しない。
- テスト: `crates/autodiff/tests/f64_create_graph.rs`（15 件。二階の閉形式・中心差分・三階の閉包性・`Pow` の `a==0`／`b==0`・正の底でゼロ指数の混合二階微分（`pow_positive_base_zero_exponent_keeps_mixed_partials`）・入口検査・範囲外 Op・再試行）、`create_graph.rs` の単体 3 件。facade `api_surface.rs` は宣言インベントリを承認形 3 件へ更新し、検出力の自己テストを足した。

### 37.2 §35 から外れた点（所有者への報告事項）

1. **マスクの格納**: §35.10 の「定数の bool マスク」は `Tensor<bool>` を `OpF64` に持たせず、0.0／1.0 の f64 定数葉（`requires_grad == false`）を指す `Where{cond, on_true, on_false}` にした。`Tensor<bool>` を持たせると `OpF64` の `Copy` 束縛が崩れ、既存の `backward` の腕を書き換えることになるため。写しの段では定数として写るので閉包性は保たれる。
2. **`mirror()` は見送り**: 第 1 段では使う箇所がなく、dead_code で clippy を落とすため作っていない（`#[allow]` で黙らせない）。#3786／#3788 が必要になった時点で足す。
3. **`rollback_point`／`rollback_to` は一般形**: 子テープの先頭に限らない切り戻しにした（#3787 の `jacfwd` J1 案が再利用できる）。前提は、`point` より後のノードを指す `VarF64` を呼び出し側が持たないこと。
4. **判定ヘルパー**: §35.5 の `compare_f64` ではなく、autodiff の `tests/common::req2_close` を使った（autodiff は backend-cpu に依存しない不変条件。判定式は同じ）。`compare_f64` は #3786 の facade テストで使う。
5. **`Pow` の安全置換の範囲**: §35.10・§35.13 論点 6 の「`Where`＋`Log` の安全置換」という方式は変えず、da 側でマスクする位置を「`b == 0` かつ底が有限の正の値でない」に絞った。`b == 0` 全体を定数マスクで消すと、底が正の位置で混合二階微分が 0 になり、正しい値 `1/a` を失うため（PR #3940 の Codex レビュー指摘）。1 階の `binary_partials_f64` は変えていない。

### 37.3 測った bit 一致の範囲

- `first_order()` は素の `backward` と全要素で `to_bits` 一致（テスト `first_order_is_bit_identical_to_plain_backward`）。構造上同一。
- `grad()`（子テープ上の勾配）は 1 階勾配と REQ-2 の範囲で一致する。式の組み立てが異なる（例: `Div` は先に `1/b` を作る）ため bit 一致は契約にしていない。
- NaiveOps（ホスト経路）のみで測った。CUDA／Metal の実測は未実施で、#3786 が `docs/perf/logs/f64-create-graph-3786/README.md` に手順と記入欄を置く。ここでは代替の値で埋めない。

### 37.4 変えていないもの

`crates/facade/src/**`、依存（`Cargo.toml`／`Cargo.lock`）、tolerance 定数、parity baseline、ガードレール閾値、`unsafe`、`docs/spec`、f32 の `create_graph.rs`、§36 の本文。なお f32 側の `ScalarBinaryOp::Pow`（`mask_b0`）にも同じ構造があり、混合二階微分が 0 になりうる。このイシューでは変えていない（対象外の候補として申し送る）。

### 37.5 #3786・#3787・#3788 への申し送り

- #3786: 第 2 段の Op、facade newtype（`first_order` は §35.13 論点どおり (a) の所有）、§35.1 の表の残りのガード反転（`F64_AUTOGRAD_SURFACE`・`f64_autograd_*` 群・期待マップへ facade の f64 宣言を足して 4 件にする・`facade_tape_backward_create_graph_is_thin_delegation` を impl の範囲で抽出する）、実機テスト、公開の適用記録。
- #3787・#3788: `rollback_to` の再利用、`mirror()` の追加、移行ノートの公開名。
