# MultiMargin・MultiLabelMargin・MultiLabelSoftMargin・sigmoid focal loss の CPU 実装記録（イシュー #2653）

- 対象: PyTorch の `F.multi_margin_loss`／`F.multilabel_margin_loss`／`F.multilabel_soft_margin_loss` と、`torchvision.ops.sigmoid_focal_loss` の式（`torch.nn.functional` に focal loss は存在しない）
- 親: #2651（兄弟 #2652 は `docs/autodiff-elementwise-loss-ops-decision.md`）
- 実装: `crates/autodiff/src/margin_focal_loss_ops.rs`（自由関数 4 本・オプション型 3 つ）・`crates/autodiff/src/eval/margin_focal_loss.rs`（forward／VJP カーネル）
- 範囲: **内部クレート限定**。facade 公開面の拡張は本イシューの範囲外（承認依頼 #2677・承認後の #2678）

## 0. 結論

- 新規 `Op` 4 種（`MultiMarginLoss`・`MultiLabelMarginLoss`・`MultiLabelSoftMarginLoss`・`SigmoidFocalLoss`）とホスト参照実装で実装した（`Op::PoissonNllLoss` と同型）。`BackendOps` は拡張せず、CPU／CUDA／Metal のどの tape からも同じホスト経路で到達する。
- 追跡対象は 4 損失とも `input` のみ。`target`（添字・ラベル）と `weight` は非追跡データとして `Op` に埋め込む。
- PyTorch 2.14.0 の実行値 fixture（`cases` 164 件・`edge_cases` 23 件）と REQ-2 統一複合判定で突合し、全件一致した。tolerance・baseline は変更していない。差分は `focal_saturated_g0.5` の 1 件のみ（§5・`diverges` 付き）。
- facade 非公開を保留ガード（`MarginFocalLossOpsHoldDoctestGuard` と `api_surface.rs` の否定ガード）で機械固定した。
- CUDA／Metal 実機の `#[ignore]` テスト 4 件は未実測（§10）。

## 1. 着手時の判定

事実のみを記す。

- 着手時点（`origin/main` = `44e3fb5b`）で #2653 にコメントはなく、#2677 は open。公開面の拡張は未承認として扱う。
- 同名の `fn` 宣言（4 名）・型名（3 つ）・モジュール名は `crates/*/src` に 0 件。
- 兄弟 #2652（PR #2795）が先にマージ済みのため、同じ様式（新規モジュール・新規 `Op`・保留ガード・fixture）を鏡写しにした。`loss_ops.rs`／`elementwise_loss_ops.rs` には追記せず、再利用は `elementwise_loss_ops::materialize_single` の可視性（`pub(crate)`）と `eval::elementwise_loss` の `finalize`／`softplus_neg`／`sigmoid` の可視性（`pub(super)`）のみ。ロジックは不変。
- #2679 の概要にも「損失」への言及がある事実は、割り振りを #2677 の表に委ねる。

## 2. 実装方式

| 項目 | 内容 |
|---|---|
| 配置 | 新規 `margin_focal_loss_ops.rs`（公開入口）と `eval/margin_focal_loss.rs`（数値カーネル。`*_forward`／`*_vjp` 命名で workspace inventory の同名 `fn` 検査を避ける） |
| Op | 4 variant。`targets`（検証済み添字）・`weight`・`target`（ラベル）は非追跡 payload |
| 融合・checkpoint・create_graph | いずれも非対応側（`is_checkpoint_eligible`・`supports_create_graph` で fail-closed） |
| focal の解釈 | sigmoid 版（`torchvision.ops.sigmoid_focal_loss` の式）1 本のみ。参照値は素の torch 2.14.0 の autograd で同式を評価して作り（torchvision 非導入）、出自を fixture README に明記した |

### 却下した案

| 案 | 却下理由 |
|---|---|
| 既存 Op の合成 | マージン系の `z > 0` 厳密判定（`z == 0`・NaN 非寄与）、`multilabel_margin` の集合除外ループ、focal の桁落ちしない `q = 1 − p_t` を合成では正確に制御しにくく、`f64` アキュムレータ契約も直接満たせない |
| `loss_ops.rs`／`elementwise_loss_ops.rs` への追記 | 承認済み・保留中の既存 inventory に未承認分が混ざる |
| target も `Var`（追跡対象）にする | 添字 target は微分不能。ラベルへの勾配はスコープ外（既存 `bce_with_logits_loss` は target も `Var` である点が異なる） |
| nn 構造体の追加・`Var` メソッドの先行追加 | 公開面の拡張は承認依頼 #2677 の範囲 |

## 3. 数値契約

共通: 要素を先に `f64` へ昇格して要素式を評価し、index 順に `f64` で蓄積して最後に 1 回だけ `f32` へ downcast する。`s` は VJP の scale（`Mean` で `g/n`、`Sum` で `g`）。

| 損失 | 要素式 | 勾配 | `Mean` の分母 `n` |
|---|---|---|---|
| MultiMargin | `z_j = margin − x_y + x_j`（`j ≠ y`）。`z_j > 0` のとき `h = z`（p=1）／`z²`（p=2）。行 `L = w[y]·Σh/C`（**`Sum` でも `/C`**） | `z_j > 0` の `j` に `s·w[y]·(1 または 2z)/C`、`x_y` から同量を引く | 行数 `N`（rank 1 は 1） |
| MultiLabelMargin | 行ごとに最初の負値の手前までを target 列とし、`L = Σ_{t∈列} Σ_{d∉集合} max(0, 1−x_t+x_d)/C` | `z > 0` の組で `dx_t −= s/C`・`dx_d += s/C` | 行数 `N` |
| MultiLabelSoftMargin | `l = t·softplus(−x) + (1−t)·softplus(x)`。行 `L = Σ_c w_c·l/C` | `dx = s·w_c·(σ(x)−t)/C` | 行数 `N` |
| sigmoid focal | `ce = (1−t)x + softplus(−x)`・`q = t·σ(−x)+(1−t)σ(x)`（`= 1−p_t` の桁落ちしない形）・`l = α_t·ce·q^γ` | `dx = s·α_t·[(p−t)q^γ + γ·ce·q^(γ−1)(1−2t)p(1−p)]`（`γ == 0` は第 2 項 0） | `numel` |

**最も踏みやすい誤り**: `elementwise_loss_scale(upstream, reduction, n)` の `n` は multi 系 3 種が行数 `N`、focal が `numel`。`grad.rs::vjp` は multi 系で `rows_cols` の `N` を渡す（fixture の Mean／Sum・rank 1／2 の全ケースで検出される）。

空・退化形: `N == 0`（focal は `numel == 0`）は損失 `0.0`・勾配は空（既存 `mse_loss` 規約。PyTorch の Mean は NaN）。multi 系の `C == 0` かつ `N > 0` は `InvalidArgument`（`/C` が未定義）。

### 実測で確定した点（fixture が正）

計画段階の見立てを fixture で検証した。いずれも見立てどおりだった。

- **`z == 0` の勾配は 0**（multi_margin p=1・p=2、multilabel_margin）。判定は `z > 0` の厳密不等号。
- **NaN は損失にも勾配にも寄与しない**（`z > 0` が偽）。`z = +inf` は損失 `+inf`。
- **multilabel_margin**: 重複 target 添字は重複分だけ加算・先頭 `-1` の行は損失 0・全クラス target（終端なし）の行は損失 0・`-1` 以降の値は集合に入らない。
- **PyTorch は終端以降も含む全 target 要素を `-1 <= t < C` で検査する**（`[1,-1,2,7,0]` は out of range）。本実装も同じ範囲検査を行う。
- **multilabel_soft_margin の `x = ±inf`**: `0·inf` の NaN も PyTorch と同じ値クラス。
- **focal の `γ = 0` と `alpha = None`** は既存 `bce_with_logits_loss` と REQ-2 一致（契約整合テスト）。

## 4. 境界検査（REQ-8・A03）

検査順序: ①rank（multi 系は 1・2。focal は任意）・shape（`require_same_shape`・target の rank／要素数）→ ②確保前バイト数上限（`checked_bytes_for::<f32>`）→ ③スカラー引数（`p ∈ {1,2}`・`margin`・`alpha ∈ [0,1]`・`gamma ≥ 0` の有限性）→ ④非追跡テンソルの値（weight は shape `[C]`・有限・非負、添字範囲、ラベルは有限かつ `[0,1]`）→ ⑤実体化 → ⑥forward → ⑦`push_eager`。エラー時に tape へ孤児ノードを残さない（単体テストで `tape.len()` 不変を固定）。`i32 → usize` 変換は範囲検査後のみ。`multilabel_margin` は行あたり `O(C²)` 時間だが確保量は入力 numel と長さ `C` のマスクのみ。本番経路に `unwrap()`／`expect()`／未検査添字アクセスはない。新規 `unsafe` なし。

## 5. PyTorch 2.14.0 との差分

| 項目 | PyTorch | 本実装 |
|---|---|---|
| `reduction='none'` | あり | 非対応（`Reduction` は Mean／Sum のみ） |
| `multi_margin_loss` の `p` | `{1, 2}` | 同じ（それ以外は `InvalidArgument`） |
| weight | 任意値 | 有限かつ非負・shape 厳密に `[C]` |
| `multilabel_soft_margin` の rank | 任意 | rank 1・2 のみ。target は有限かつ `[0,1]` |
| focal の入手元 | `torch.nn.functional` に無い | sigmoid 版のみ。`alpha < 0` で無効は `alpha = None` で表す。softmax 版は非対応 |
| 空テンソルの Mean | NaN | `0.0`（既存 `mse_loss` 規約） |
| **focal の飽和域（`γ < 1`）** | f32 で `1 − p_t` が 0 に潰れ、`pow(·, γ)` の逆伝播（`∞·0`）で勾配 NaN | `q = 1 − p_t` を `t·σ(−x)+(1−t)σ(x)` の桁落ちしない形で `f64` 評価し、真の勾配（0 に収束）に近い有限値。`q == 0` かつ `γ < 1` の第 2 項は極限値 0。fixture `focal_saturated_g0.5`（`diverges` 付き）で本実装の値（有限・`|g| < 1e-5`・非飽和の 2 要素は PyTorch と一致）を明示 assert |

`diverges` を付けたのはこの 1 件のみ（数学的理由を記録できる差分に限定）。`x = ±inf` の単一要素（`focal_inf_*`）と `γ ∈ {0, 1, 2}` の飽和は PyTorch と値クラス一致で、tolerance は変更していない。

## 6. テスト構成

- `crates/autodiff/src/margin_focal_loss_ops.rs` 単体テスト: 手計算値、中心差分と VJP の一致、空テンソル、エラー経路と `tape.len()` 不変、`weight = ones` と weight なしの bit 一致、上流勾配スケール。
- `crates/autodiff/tests/margin_focal_loss_parity.rs`: fixture 全件の forward と勾配（REQ-2）、`edge_cases` の値クラス、契約整合（focal〈alpha なし・γ=0〉・multilabel_soft_margin〈weight なし・Mean〉は既存 `bce_with_logits_loss(Mean)` と REQ-2 一致）。
- `crates/facade/tests/margin_focal_loss_ops_backend_parity.rs`: 4 損失（multi_margin は 2 構成）× {forward, backward} の CPU 対 NaiveOps（属性なし・bit 一致）、CUDA／Metal は `#[ignore]`。
- fixture の出自: `crates/autodiff/tests/fixtures/margin-focal-loss-pytorch-reference/README.md`（生成スクリプト・sha256 つき）。

## 7. facade 公開形の推奨案（推奨案の記録であり承認記録ではない）

- 推奨は `Var` の 1 行委譲メソッド 4 本（`multi_margin_loss`・`multilabel_margin_loss`・`multilabel_soft_margin_loss`・`sigmoid_focal_loss`）。既存 7 損失（#2538〜#2540）・兄弟 #2652 の推奨形と同形で、追加のみ・非破壊。
- 推奨しない: モジュール再エクスポート、`Tensor`／`Tape` への配置、`Sequential::add_*`（損失は層ではない）、`compat::Loss` への variant 追加。
- オプション型 3 つ（`MultiMarginOptions`・`MultiLabelSoftMarginOptions`・`SigmoidFocalLossOptions`）と `Reduction` の名指しは、既知ギャップ（`docs/autodiff-loss-ops-decision.md` の #2538 追記）と同じ論点で、#2600 ツリーの記録側で扱う。
- 承認依頼は #2677、公開は承認後の #2678（`var.rs` が対象ファイル）。本イシューでは `var.rs` にメソッドを足さない。

## 8. スコープ外

- facade 公開（`Var` 委譲 4 本、保留ガードの正ガード反転）
- nn 構造体（`MultiMarginLoss` 等）、`compat::Loss` への variant 追加、`model_io`／ONNX 対応
- softmax（多クラス）版 focal、`reduction='none'`、target への勾配、soft label の `[0,1]` 外、負の weight、`multi_margin` の `p ∉ {1,2}`、`multilabel_soft_margin` の rank 3 以上と weight の broadcast 形
- GPU 専用カーネルと CUDA／Metal 実機実測、`create_graph`（高階微分）、activation checkpoint、f64／f16／bf16 経路

## 9. 多層防御

| 層 | 内容 |
|---|---|
| 1 | `MarginFocalLossOpsHoldDoctestGuard`（`crates/facade/src/lib.rs`。全 `pub mod` を glob import した上でローカルのモジュール・型・同名メソッドをプローブ。同名の公開が現れると名前解決の曖昧性（E0659）で失敗する） |
| 2 | glob 集合の一致（`margin_focal_loss_ops_hold_doctest_globs_all_pub_modules`） |
| 3 | プローブ本文の固定文言一致（`..._probe_body_matches_fixed_contract`） |
| 4 | ソース走査（`facade_does_not_reexport_or_declare_margin_focal_loss_ops` と自己テスト）と workspace の `fn` 宣言 inventory（`workspace_declares_margin_focal_loss_ops_fn_names_only_in_allowed_locations`） |

検出範囲は列挙した名前・型に限り、マクロ生成や別名経由のメソッドまでは保証しない。

**反証確認**: `crates/facade/src/lib.rs` に `pub use fandhe_ai_autodiff::margin_focal_loss_ops;` を仮に足し、doctest が `E0659`（`margin_focal_loss_ops` is ambiguous）で失敗し、ソース走査テスト `facade_does_not_reexport_or_declare_margin_focal_loss_ops` が失敗することを確認して元に戻した。

## 10. 実機申し送り

CUDA／Metal の `#[ignore]` テスト（計 4 件）は本実行環境から実機に到達できないため未実測。手順と記入欄は `docs/perf/logs/margin-focal-loss-ops-2653/README.md`。新規 GPU カーネルはなく、ホスト経路の bit 一致確認である。

## 11. 出典

- `docs/autodiff-elementwise-loss-ops-decision.md`（兄弟 #2652。様式・保留ガード）
- `docs/autodiff-loss-ops-decision.md`（既存 7 損失の設計・数値契約）
- `.claude/rules/coding-rust.md`（`f64` アキュムレータ契約・境界検査）
- PyTorch 2.14.0 の実行値（`crates/autodiff/tests/fixtures/margin-focal-loss-pytorch-reference/`）
