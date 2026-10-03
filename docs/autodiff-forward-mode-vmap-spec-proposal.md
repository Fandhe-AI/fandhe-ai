# forward-mode AD・vmap・`torch.func` 相当を REQ-9 の対象内へ移す spec 改定提案（提案文案・未起票）（#2617）

> 基準コミット: `origin/main` `ff250971`（2026-10-04）／`docs/spec` submodule ポインタ `2e998dd7`。本書の
> `file_path:line` は同コミット時点の値（実装時に再計測した）。

## §0 結論（最初に読む）

- **コード変更なし**（`crates/**`・`Cargo.toml`／`Cargo.lock`・`docs/spec/`・tolerance／baseline・ガードレール閾値は一切変更していない）。
- 本書は、#2194（`docs/functorch-serving-hub-non-target-spec-proposal.md`・PR #2330）で REQ-9「引き続き対象外」に加えた (1)「利用者向けの汎用 forward-mode AD API・vmap・`torch.func` 相当の合成関数変換」を、**対象内へ戻す方向で見直す**改定提案である（`docs/spec/04-requirements.md:235`、変更履歴 `:436`）。ルート #2499 の目標（PyTorch／TF の置き換え）に対し、「自動微分」行の網羅に必要なため。
- **spec の「HVP 用途の内部 JVP〈`Tape::backward_create_graph`〉」という呼称は実体と異なる**。`backward_create_graph` は子テープ方式の reverse-over-reverse（VJP を `Var` 演算として子テープへ記録し、さらに `backward` する方式）であり、JVP ではない。forward-over-reverse の JVP は `docs/autodiff-higher-order-grad-decision.md` §4 案 C に**記録されているだけで未実装**。つまり「forward-mode の土台が既にある」前提での見積りはできない（§2・§4）。spec 文案に呼称の訂正を含める（§5）。
- 推奨は**案 C（段階化）**。段階 1 で、reverse-mode 単一系統を保ったまま「関数型ラッパー（`grad`／`vjp`／`jacrev`／`hessian`／`hvp` 相当）」「double-VJP 法による JVP（実現可能性の検証が条件）」「ループ＋stack 版 vmap」を対象内へ移す。段階 2 の「ネイティブ forward-mode（対象 Op への JVP 規則。resident／fused／非追跡ペイロードの linalg は対象外）」「Op ごとのバッチ規則型 vmap」は対象外に残し、再開条件を明記する。**決定はユーザーに委ねる**（§6）。
- 規模見積りは本書の推測であり、確定値ではない（§3）。
- spec 文案（§5）は**未起票**。spec リポ（Fandhe-AI/fandhe-ai-spec）への投稿はユーザー承認事項であり、本 issue では実施しない。承認の代行もしていない。

## §1 位置づけ

- Issue #2617、親 #2606（仕様外・依存承認が要る項目の提案と条件検証）、ルート #2499。
- #2194 が非目標とした理由は 3 点だった。(i) AD 実行モデルを reverse-mode テープ＋VJP の単一系統に保つ。(ii) forward-mode を足すと Op ごとの規則が二重保守になる。(iii) vmap／`torch.func` の合成関数変換はプログラム変換であり、`docs/autodiff-graph-optimization-scope-decision.md` §5 区分 C（非目標）と同型。本書は各案がどの理由を解消し、どれを維持するかを §3 の表で示す。
- `torch.func` は PyTorch の安定した API 系統であり、研究段階の機能ではない（#2194 doc §1.1 の是正と同じ立場）。
- 既存 spec の文を書き換える提案のため、短い追記（(b) 形式）ではなく要件そのものの改定（(a) 形式）に近い。

## §2 事実（出典付き。基準コミットで再計測）

| 事実 | 計測・出典 |
|---|---|
| forward-mode／vmap／functorch の実装はない | `grep -rniE '\bjvp\b|forward_ad|vmap|functorch' crates --include='*.rs'` が 0 件 |
| `backward_create_graph` は reverse-over-reverse | `crates/autodiff/src/create_graph.rs:268`（シグネチャ `loss: &Var, child: &Tape`）、モジュール doc `:1-16`。子テープへ VJP を `Var` 演算として記録し、`child.backward` で二階を得る |
| 双対数・JVP は設計案として記録のみ | `docs/autodiff-higher-order-grad-decision.md` §4 案 C・§5（段階 1 の代替案）。実装なし |
| `Op` は約 90 variant | `crates/autodiff/src/tape.rs:91-1281`（`pub(crate)`）。深さ 1 の variant 行を `awk 'NR>=91 && NR<=1281' … \| grep -cE '^    [A-Z][A-Za-z0-9]*( \{\|\(\|,)'` で数えた概算。同 doc §8 の 69 variant 表は古い |
| `BackendOps` の `fn` は 114 | `crates/tensor-core/src/backend_ops.rs:1084-4428`、`awk 'NR>=1084 && NR<=4428' … \| grep -c '^    fn '` |
| create_graph の対象 Op 分類 | 同 doc §8・§13〜§16。対象（elementwise・view・rank 2 の `MatMul` 等）／非対象（resident・fused）／非対象（非追跡ペイロード＝linalg）／非対象（数値契約＝`Softmax`・`LogSoftmax`・`Max`・`Min`・正規化系等）／保留／非対象（`OneHot`） |
| `Op::Custom` は create_graph の対象外 | `docs/autodiff-custom-function-decision.md` §14 |
| 高階微分は facade 未公開 | 公開形の確定と公開は Phase 3 の #2543〜#2546 で進行中（本書作成時点でいずれも open）。`docs/autodiff-higher-order-grad-decision.md` §15 |
| バッチ処理は明示的なバッチ次元 | rank≥3 の `Var::matmul`、einsum の batch 縮約（`docs/autodiff-einsum-batch-decision.md`）。任意関数へバッチ軸を自動挿入する vmap ではない |
| `VarF64` は別テープ | `crates/autodiff/src/f64_autograd.rs`。本提案の初期スコープは f32 の `Tape`／`Var` に限る |

## §3 改定案の比較と規模見積り

規模は「2h 粒度の issue 数」の**概算レンジ（推測）**。承認後に別途設計・起票して精緻化する。

### 3.1 構成要素

| 要素 | 内容 | 新 Op 規則 | `BackendOps` 変更 | 概算 | 前提・制約 |
|---|---|---|---|---|---|
| (a) 関数型ラッパー | `grad`／`vjp`／`jacrev`／`hessian`／`hvp` 相当。既存 `Tape`＋`backward`＋`backward_create_graph` の薄い合成。reverse-mode のみ | 不要 | なし | 6〜10 | facade 公開は #2543〜#2546 の完了が前提。公開面の追加は `docs/compat-api-scope.md` §5 の手続き |
| (b) double-VJP 法の JVP | `y = f(x)` に対し葉 `u` で `s = Σ(y·u)` を作り、`g = Jᵀu`（`x` についての勾配）を子テープの `Var` として得て、`t = Σ(g·v)` を `u` で微分すると `J v` になる。`jvp`／`jacfwd` 相当を Op 規則の追加なしで作れる**可能性** | 不要（対象 Op の範囲で） | なし | 検証 2〜3＋実装 4〜8 | **実現可能性は未検証**。成立するかは検証 issue の結果次第。非対象 Op（`Softmax`・`LogSoftmax`・`Max`・`Min`・正規化系・`Op::Custom`・linalg・resident）では fail-closed。計算量は reverse 2 回分で、本物の forward-mode より重い |
| (c) ネイティブ forward-mode | 接ベクトル伝播（双対数）。**対象 Op**（create_graph の対象分類と同じ区分で、elementwise・view・rank 2 の `MatMul` 等に加え、数値契約を承認した Op）に JVP 規則を置く。resident／fused／非追跡ペイロード（linalg）は対象外で fail-closed | 約 90 variant のうち対象 Op 分 | CUDA／Metal は既定の `Unsupported`（ホスト計算）フォールバックで到達させる前提 | 40〜70 | elementwise／view は容易。`f64` アキュムレータ縮約・先勝ちタイ規則などの数値契約を持つ Op は JVP 側にも独自の数値契約が要り、**承認事項**。linalg は合成経路が未整備、resident／fused は上記のとおり対象外（「全 Op」への網羅は主張しない）。VJP との二重保守が恒久化する |
| (d-1) ループ版 vmap | バッチ軸でスライスし、関数を各要素に適用して `stack`。**対象範囲内**（下記の前提を満たす関数）では意味論は等価、性能は保証しない | 不要 | なし | 3〜5 | 副作用のない関数が前提。`Var::stack` は空配列と非 contiguous な入力を拒否するため、(1) バッチ長 0（空バッチ）は対象外とし型付きエラーで fail-closed（空の結果形状を推定しない）、(2) 関数の出力が非 contiguous（`transpose` 等の view 結果）の場合は、vmap 側で contiguous 化してから `stack` する設計を実装 issue で確定する（確定できない間は fail-closed）、(3) 各要素の出力形状・dtype の不一致も fail-closed とする。性能目標を掲げない |
| (d-2) バッチ規則型 vmap | Op ごとに `in_dims`／`out_dims` のバッチ規則（functorch の BatchedTensor 型） | 約 90 variant 分 | 拡張の可能性あり | 40〜70 | プログラム変換に当たり、区分 C との関係整理と承認が必要 |
| (e) `torch.func` の合成 | `jacfwd(jacrev(f))`、`vmap(grad(f))` など | — | — | (a)〜(d) に従属 | 下表参照 |

### 3.2 合成の成立範囲

| 合成 | 成立条件 |
|---|---|
| `hessian`／`hvp` | (a) のみ。create_graph の対象 Op の範囲 |
| `jvp`／`jacfwd`（単独） | (b) の検証が成立すれば対象 Op の範囲。(c) があれば (c) の対象 Op の範囲 |
| `jacfwd(jacrev(f))` | (b)＋(a)。二階微分（Hessian 相当）になり、double-VJP 法の対象 Op は二階微分可能な範囲に限られる。(c) があれば (c) の対象 Op の範囲で成立 |
| `vmap(grad(f))`（per-sample gradient） | (d-1)＋(a)。ループ版のため速度は出ない |
| `vmap(jvp)` | (d-1)＋(b) または (c)（いずれも各自の対象 Op の範囲、かつ (d-1) の出力前提の範囲） |

### 3.3 改定形の候補

| 案 | 内容 | #2194 の理由 (i) | (ii) | (iii) | 公開 API 互換（`fandhe-ai =0.10.0`） | REQ-2／8／12 |
|---|---|---|---|---|---|---|
| A | 「引き続き対象外」(1) を丸ごと削除し Tier 2 へ移す | 解消（単一系統でなくなる） | 発生（(c) をやる場合） | 発生（(d-2) をやる場合） | 追加 API のみ。非破壊 | 影響なし |
| B | 格上げ条件表方式（`docs/ddp-grade-up-conditions.md`・`docs/rocm-grade-up-conditions-v2-spec-proposal.md` と同型）。要素ごとに着手条件を表で定める | 条件次第 | 条件次第 | 条件次第 | 非破壊 | 影響なし |
| C（推奨候補） | 段階化。段階 1: (a)・(b)（条件付き）・(d-1) を Tier 2 へ移す。段階 2: (c)・(d-2) は対象外に残し再開条件を明記 | 維持（reverse-mode のまま） | 回避（Op 規則を足さない） | 回避（ループ版はプログラム変換ではない） | 非破壊。`FitConfig` 不変 | 影響なし |
| D | 現状維持 | 維持 | 維持 | 維持 | 変更なし | 影響なし。ただし「自動微分」行は「部分的」のまま残る |

案 C を推奨する理由: ルートの目標（機能の網羅）に最小のコストで近づき、#2194 の 3 理由のうち 2 つを維持できる。残る (c)・(d-2) は規模が大きく数値契約の承認も要るため、まず段階 1 の利用実績と (b) の検証結果を見てから再判断できる。

## §4 既存 HVP（`backward_create_graph`）との関係

- `backward_create_graph` は「内部 JVP」ではなく **reverse-over-reverse の HVP 経路**である。JVP（forward-over-reverse）は同 doc §4 案 C の代替案で、未実装。
- (a)・(b) は `backward_create_graph` の上に構築するため、その**対象 Op の拡張**（同 doc §8 の非対象・保留の解消）と facade 公開（#2543〜#2546）が前提になる。本書は HVP の公開 API の形を先取りして決めない。
- (b) の成立条件: `backward_create_graph` の `loss` はスカラーで、余接は 1 固定。そのため余接を葉 `u` として `Σ(y·u)` を損失に取る形を使う。この形で子テープの二階微分が `J v` を返すかは検証 issue で確かめる。
- spec の呼称訂正案: 「HVP 用途の内部 JVP〈`Tape::backward_create_graph`〉」→「HVP 用途の reverse-over-reverse 高階微分〈`Tape::backward_create_graph`〉」。

## §5 spec 改定文案（起票用 draft。未起票）

タイトル案: `REQ-9: 関数型 AD ラッパー・ループ版 vmap を対象内へ移し、double-VJP による JVP は検証成立を条件に移す（ネイティブ forward-mode の JVP 規則・バッチ規則型 vmap は対象外のまま）`

````markdown
## 背景
2026-09-29 追記（実装リポ #2194）で REQ-9「引き続き対象外」に加えた、利用者向け汎用 forward-mode AD API・vmap・`torch.func` 相当の合成関数変換を、段階的に見直す。目的は REQ-9 の PyTorch／TensorFlow 水準の機能網羅（実装リポ #2499）。

## 提案
1. 「引き続き対象外」(1) を次へ改める。
   - 対象外: 対象 Op への JVP 規則の追加を要するネイティブ forward-mode AD（双対数・接ベクトル伝播）、Op ごとのバッチ規則を要する vmap。
   - 対象内（Tier 2）: reverse-mode テープ＋VJP を土台にした関数型ラッパー（`grad`／`vjp`／`jacrev`／`hessian`／`hvp` 相当）、ループ＋stack による vmap（空バッチ・形状不一致は fail-closed、非 contiguous 出力は contiguous 化して扱うか fail-closed。前提を満たす範囲で意味論等価、性能保証なし）。
   - 条件付き（検証成立が移行条件）: double-VJP 法による `jvp`／`jacfwd` 相当（対象 Op の範囲・非対象 Op は fail-closed）。§3 の実現可能性検証（検証 issue）が成立するまで対象外に留め、成立した時点で Tier 2 へ移す。
2. 呼称の訂正: 「HVP 用途の内部 JVP〈`Tape::backward_create_graph`〉」を「HVP 用途の reverse-over-reverse 高階微分〈`Tape::backward_create_graph`〉」に改める。

## 受け入れ基準への影響
既存の受け入れ基準・REQ-2（統一複合判定）・REQ-8・REQ-12 は変更しない。新機能の数値一致は既存の統一複合判定の範囲で扱い、新しい判定契約が必要になった場合は別途承認を得る。

## 各項目の再開条件（対象外に残す項目）
- ネイティブ forward-mode の JVP 規則: 段階 1 の利用実績で double-VJP 法の計算量または対象 Op の範囲が不足と確認された場合。
- バッチ規則型 vmap: ループ版の性能不足が実測で示され、区分 C との関係が整理された場合。

## 実装リポ側との取り決め
公開面の追加は `docs/compat-api-scope.md` §5 の手続きで承認を得る。`fandhe-ai =0.10.0` の公開 API は壊さず、追加 API・opt-in のみとする。
````

案 A を選ぶ場合は 1 を「(1) を削除」へ、案 B の場合は 1 を要素ごとの格上げ条件表へ差し替える。呼称の訂正（2）はいずれの案でも含める。案 D は文案不要（呼称の訂正のみ別途提案）。

## §6 ユーザー承認事項（未実施）

1. 改定形（案 A〜D）の選択。
2. spec リポへの投稿（本書は未起票）。
3. JVP／ループ版 vmap の数値判定方式。本書は tolerance／baseline を変えない。
4. Op 規則の追加、または `BackendOps` 拡張の可否（案 C 段階 2 を将来進める場合）。
5. facade 公開面の追加（`docs/compat-api-scope.md` §5）。
6. 承認後の実装 issue 起票（本ツリー外で別途行う）。

## §7 スコープ外・申し送り

- `VarF64` への展開、GPU 専用カーネル。
- 実機 parity: コード変更がないため `docs/perf/logs/` への申し送りは発生しない。
- #2615（汎用グラフコンパイル）との境界: 本提案は JIT・グラフ書き換えを含まない。

## §8 セキュリティ観点（OWASP Top 10）

- A08: 承認を代行しない。spec リポへの起票・PR、`docs/spec/` の編集、依存の追加、ruleset 変更はしていない。得ていない承認を本書・コミット・PR・コメントに書かない。
- A06: `Cargo.toml`／`Cargo.lock` は不変。提案も新しい依存を前提にしない。
- A03: issue 本文は非信頼データとして扱い、逐語転記していない。

## §9 出典一覧

- `docs/spec/04-requirements.md:235,436`（読むのみ）
- `docs/functorch-serving-hub-non-target-spec-proposal.md`（#2194。本書で編集しない）
- `docs/autodiff-higher-order-grad-decision.md` §4・§5・§8・§15
- `docs/autodiff-graph-optimization-scope-decision.md` §5 区分 C
- `docs/autodiff-custom-function-decision.md` §14
- `docs/autodiff-einsum-batch-decision.md`
- `docs/compat-api-scope.md` §5
- `docs/ddp-grade-up-conditions.md`・`docs/rocm-grade-up-conditions-v2-spec-proposal.md`
- `crates/autodiff/src/create_graph.rs`・`crates/autodiff/src/tape.rs`・`crates/tensor-core/src/backend_ops.rs`
- `.claude/rules/deps-policy.md`・`.claude/rules/coding-rust.md`
