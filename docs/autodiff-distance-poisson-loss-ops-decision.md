# CosineEmbedding・MarginRanking・TripletMargin・PoissonNLL 損失 設計判断記録

イシュー #2167（親 #2131「PyTorch／TF 置き換えの API 網羅」）。
`docs/autodiff-loss-ops-decision.md`（#2166）と同型の記録・同じ判断
枠組みを踏襲する。

## §0 結論

PyTorch `nn.CosineEmbeddingLoss`・`nn.MarginRankingLoss`・
`nn.TripletMarginLoss`・`nn.PoissonNLLLoss` 相当の 4 API を、#2166 と
同じく **`fandhe_ai_autodiff::loss_ops`（facade 非公開の自由関数
モジュール）** へ追加した。`Var` に inherent の `pub fn` は追加して
いない。

- `cosine_embedding_loss(x1, x2, y, margin, reduction)`: 新規
  `Op::CosineEmbeddingLoss`
- `margin_ranking_loss(x1, x2, y, margin, reduction)`: 新規
  `Op::MarginRankingLoss`
- `triplet_margin_loss(anchor, positive, negative, options, reduction)`:
  新規 `Op::TripletMarginLoss`（`options: &TripletMarginOptions`）
- `poisson_nll_loss(input, target, options, reduction)`: 新規
  `Op::PoissonNllLoss`（`options: &PoissonNllOptions`）

4 つとも `BackendOps` に対応メソッドを持たず、常にホスト参照実装
（`crate::eval`）を経由し `push_eager`（実体化済み）でテープへ記録
する（#2166 の `Op::L1Loss` と同型のパターンを採用。§2.1 の判断
理由参照）。`nn::loss::CosineEmbeddingLoss`・`MarginRankingLoss`・
`TripletMarginLoss`・`PoissonNllLoss`（新規構造体）を薄いラッパーと
して提供する。`nn::loss` は facade が再エクスポートしないため、
これらも facade 公開面には出ない。

## §1 背景

イシュー #2167・親 #2131 のどちらにも所有者の承認コメントはない。
関連イシュー #2169（`compile()` の `Loss` enum への統合）は別 PR の
担当で、本 PR は `crates/facade/src/compat/training.rs` を変更しない。
親 #2131 が定める「設計判断記録 → 承認 → 実装」の 2 段階方針に従い、
本実装は内部クレート限定に倒す（facade 公開面の拡張は §5「承認事項」
に整理する）。

## §2 各演算の設計

### §2.1 `BackendOps` メソッドを新設しない判断

#2166 の実装計画は「defaulted な `BackendOps` メソッドを新設し
`Unsupported` へフォールバックする」型（`Op::PNorm` 等）と、「
`BackendOps` メソッドを持たずホスト計算のみで完結する」型
（`Op::L1Loss`）の 2 通りを比較したうえで後者を採用した。本イシューも
同じ損失群（#2131 の 5-C 節）の延長であり、同じ判断を踏襲する。GPU
専用カーネルは §6「スコープ外」で別イシューへ引き継ぐ。

### §2.2 CosineEmbedding 損失（`Op::CosineEmbeddingLoss`）

**shape 契約**: `x1`・`x2` は同一 shape で rank 1（`[D]`。この場合
`y` は 0 次元 `[]`）または rank 2（`[N, D]`。この場合 `y` は
`[N]`）のいずれか。

**数式**（サンプル `n`。`.claude/rules/coding-rust.md` の勾配長軸
縮約契約に従い要素を先に `f64` へ昇格してから二乗・積算する）:
`m1 = Σ_d x1[n,d]² + EPSILON`・`m2 = Σ_d x2[n,d]² + EPSILON`
（`EPSILON = 1e-12`。PyTorch `aten/src/ATen/native/Loss.cpp` の
`cosine_embedding_loss` 実装と同じ値。ATen ソースへの直接到達手段が
本リポの実行環境になかったため、PyTorch 公式ドキュメント・広く知られた
実装値を根拠とする——不確実性の明示）・`dot = Σ_d x1[n,d]·x2[n,d]`・
`cos = dot / sqrt(m1·m2)`。`y[n] == 1.0` なら `L_n = 1 − cos`、
`y[n] == -1.0` なら `L_n = max(0, cos − margin)`。`Mean` は
`(Σ_n L_n) / N`、`Sum` は `Σ_n L_n`（`N == 0` はいずれも損失 `0.0`）。

**勾配**: `y=1` は符号反転（`dx1 = −s·d(cos)/d(x1)`）、`y=-1` は
hinge が有効（`cos − margin >= 0`）なときのみ流す。境界規約
（`cos − margin` がちょうど 0 のときの扱い）は、PyTorch の
`clamp_min` の VJP 契約（`self >= min` で勾配を通す）に倣い、
**境界ちょうど 0 でも勾配を通す**と決定した（`tools/autograd/
derivatives.yaml` への直接到達手段が本リポの実行環境になかったため、
`clamp_min` の一般的な劣勾配規約を根拠とする）。
`d(cos)/d(x1)[d] = x2[n,d]/sqrt(m1·m2) − cos·x1[n,d]/m1`（`x2` 側は
対称）。

### §2.3 MarginRanking 損失（`Op::MarginRankingLoss`）

**数式**: `L_i = max(0, −y_i·(x1_i − x2_i) + margin)`。`Mean` は
`numel` で除算、`Sum` はそのまま加算（`numel == 0` はいずれも損失
`0.0`）。

**勾配**: hinge が有効（`raw >= 0`。§2.2 と同じ境界規約）なときのみ
`dx1_i = −s·y_i`・`dx2_i = +s·y_i` を流す。

### §2.4 TripletMargin 損失（`Op::TripletMarginLoss`）

**オプション**（[`TripletMarginOptions`]。builder 構築・非公開
フィールド・`#[non_exhaustive]`。#2166 の `CrossEntropyOptions` と
同型）: `margin`（既定 `1.0`）・`p`（既定 `2.0`）・`eps`（既定
`1e-6`）・`swap`（既定 `false`）。PyTorch `nn.TripletMarginLoss()` の
既定と一致する。

**shape 契約**: `anchor`・`positive`・`negative` は同一 shape で
rank 1（`[D]`）または rank 2（`[N, D]`）のいずれか。

**距離の定義**: `d(u, v) = ‖u − v + eps‖_p`（`eps` を**差へ先に
加えてから** `p` ノルムを取る。PyTorch `aten/src/ATen/native/
Distance.cpp::pairwise_distance` の実装に準拠する一般的な理解に
基づく——ATen ソースへの直接確認は本リポの実行環境の制約により
できていない）。`d_ap = d(anchor, positive)`・`d_an = d(anchor,
negative)`。`swap = true` のとき `d_pn = d(positive, negative)`・
`d_neg = min(d_an, d_pn)`、それ以外は `d_neg = d_an`。
`L_n = max(0, d_ap − d_neg + margin)`。`Mean` は `N` で除算
（rank 1 は `N=1`）、`Sum` はそのまま加算（`N == 0` は損失 `0.0`）。

**`p` ノルムの実装**（`crate::eval::p_norm_f64`）: 当初は
`reduce_ops::norm_p`（#2147）の overflow-safe なスケール形を採らず、
素直な `(Σ|v_i|^p)^(1/p)` を `f64` で計算していた（`triplet_margin_
loss` の距離差はネットワーク出力の差分スケールが想定され、
`rmsnorm`／`reduce_ops::norm_p` のような大規模縮約軸ほどの overflow
リスクはない、という判断だった）。しかし PR #2286（codex-review 指摘）
で、有効な `p`（`p >= 1.0` 制約を満たす `p=1024` 等）と現実的な差分値
（例 `|v_i|=2`）の組合せでも `|v_i|^p` 自体が `f64` の範囲を超えて
overflow し、`d_ap`／`d_an` が `inf` になって以降の hinge 計算が
`inf − inf = NaN` を生む欠陥が実際に指摘されたため、上記の「意図的な
スコープ限定」は撤回する。`reduce_ops::norm_p` フォールバック
（`eval::vector_norm_p_along`）と同じ overflow-safe なスケール形
（`mx = max|v_i|` を括り出してから `mx · (Σ (|v_i|/mx)^p)^(1/p)` を
計算する）へ揃えた。`p_norm_grad_f64`（勾配）も同様に、
`|v_i|^(p-1)`／`norm^(p-1)` を別々に計算せず `(|v_i| / norm)^(p-1)`
（`|v_i| <= norm` より必ず `[0, 1]` に収まり overflow しない）へ
修正した。

**CosineEmbeddingLoss の分母（`(m1 * m2).sqrt()`）**（PR #2286
codex-review 指摘）: forward（`eval::cosine_embedding_loss_forward`）・
VJP（`grad::cosine_embedding_loss_vjp`）とも、各ノルムを先に `sqrt`
してから乗じる `m1.sqrt() * m2.sqrt()` へ修正した（中間積 `m1 * m2`
の指数が `m1`／`m2` の 2 倍になるのを避ける標準的な安定化。数学的に
同値）。

**NaN の距離・ランキング値の hinge 適用**（PR #2286 codex-review
指摘）: `CosineEmbeddingLoss`（`y == -1` の hinge 分岐）・
`MarginRankingLoss`・`TripletMarginLoss` の forward で使っていた
`raw.max(0.0)`（Rust の `f64::max` は左辺が `NaN` のとき右辺を返す
ため `NaN` が黙って `0.0` に潰れる）を、`NaN` を伝播する
`eval::nan_propagating_max_f64` へ置き換えた。VJP 側の hinge 判定
（`raw >= 0.0` 等。`NaN` との比較は `false` を返すため、hinge が
無効な扱いとなり勾配 0）は既存の PyTorch `clamp` backward マスクと
同じ挙動のため変更していない。

**勾配**: hinge が有効（`d_ap − d_neg + margin >= 0`。§2.2 と同じ
境界規約）なときのみ、`d_ap` の寄与 `+1`・`d_neg` の寄与 `−1` を
各距離のノルム勾配（`d(‖v‖_p)/d(v_i) = sign(v_i)·|v_i|^(p−1) /
‖v‖_p^(p−1)`。`‖v‖_p == 0` の要素は勾配 `0`——PyTorch
`norm_backward` の `masked_fill` に倣う一般的な理解）経由で
`anchor`・`positive`・`negative` へ配分する。`swap` で `d_an ==
d_pn`（同値）のときは、PyTorch `min.other` の VJP の一般的な理解
（同値時は等分）に倣い `1/2` ずつに等分する。

### §2.5 PoissonNLL 損失（`Op::PoissonNllLoss`）

**オプション**（[`PoissonNllOptions`]）: `log_input`（既定
`true`）・`full`（既定 `false`）・`eps`（既定 `1e-8`）。PyTorch
`nn.PoissonNLLLoss()` の既定と一致する。

**数式**: `log_input = true` のとき `L = exp(x) − t·x`、`false` の
とき `L = x − t·log(x + eps)`。`full = true` のとき、`t > 1` の
要素にのみ Stirling 近似項 `t·log(t) − t + 0.5·log(2π·t)` を加える
（`t <= 1` は寄与 `0` のままスキップし、`t·log(t)` を計算してから
マスクする経路は取らない——`t = 0` で `NaN` を生まないための実装上の
選択）。`Mean` は `numel` で除算、`Sum` はそのまま加算（`numel == 0`
は損失 `0.0`）。

**`target` に勾配を定義する判断**（R5 の一部）: `target` を `&Var`
（追跡対象）として受け取り、`full = true` の Stirling 項の微分を
含めて `dTarget` を必ず実装する。`&Var` を受け取りながら勾配が
黙って欠落する API は作らない（`crate::loss_ops` モジュール全体の
方針）。

**勾配**: `log_input = true` なら `dx = exp(x) − t`・`dt = −x`、
`false` なら `dx = 1 − t/(x+eps)`・`dt = −log(x+eps)`。`full = true`
かつ `t > 1` の要素は `dt` に `log(t) + 0.5/t` を加える。

## §3 PyTorch との差分

- `y`（CosineEmbedding・MarginRanking）は厳密に `±1.0` の要素のみ
  許容する（PyTorch の `MarginRankingLoss` は任意の `y` を受け付ける
  が、`y` を符号として使う設計意図に反する任意値を fail-closed に
  拒否する安全側の判断）
- `TripletMarginLoss` の `p` は `>= 1` のみ許容する（PyTorch は
  `p < 1` の擬似ノルムも受け付けうるが、`p < 1` は三角不等式が
  成立しない領域であり、勾配式の意味論も PyTorch の一般的な理解と
  ずれるリスクがあるため fail-closed に拒否する）
- 空入力（`N == 0`／`numel == 0`）は 4 損失とも損失 `0.0`・勾配 `0`
  を返す（PyTorch は `NaN` を返すケースがあるが、`Op::MseLoss` の
  空バッチ規約 `n == 0 → 0.0` を踏襲する安全側の判断）
- `TripletMarginWithDistanceLoss`（任意の距離関数を渡す版）は未実装
  （§6「スコープ外」）
- **境界規約（hinge・`min` 同値・p ノルム 0）は ATen ソース
  （`derivatives.yaml`・`Loss.cpp`・`Distance.cpp`）への直接確認では
  なく、PyTorch の公開ドキュメント・広く知られた実装パターンからの
  類推で決定した**（本リポの実行環境に GitHub 上の PyTorch ソースへ
  到達する手段がなかったため）。数値微分突合テスト
  （`crates/autodiff/tests/loss_ops.rs`）は kink（境界点）を意図的に
  避けた点のみで検証しており、境界ちょうど 0 の劣勾配選択自体は
  PyTorch 実機との突合では検証していない。実機（PyTorch）との突合が
  可能なセッションでの再確認を推奨する

## §4 数値一致・parity

4 つの `Op` はいずれもホスト計算のみで `BackendOps` を経由しないため、
3 バックエンド（CPU／CUDA／Metal）で **bit 完全一致**することが
期待される（`Op::L1Loss` と同型の性質）。CPU（`CpuBackendOps`）と
`NaiveOps` の bit 一致は `crates/facade/tests/loss_ops_backend_
parity.rs` の属性なしテストで固定済み。CUDA（DGX Spark GB10）・
Metal（Apple Silicon）実機は本リポの実行環境に到達手段がないため
未実測（`docs/perf/logs/loss-ops-2167/README.md` へ申し送る）。
受け入れ判定は REQ-2 の統一複合判定を正とする。

## §5 承認事項（facade 公開面の拡張）

以下は未承認のため保留する（`crates/facade/src/lib.rs::
LossOpsHoldDoctestGuard`（#2166 で導入・本イシューで対象を 6 関数へ
拡張）・`crates/facade/tests/api_surface.rs` の対応する 4 テストで
機械的に固定）:

- `Var::cosine_embedding_loss(&self, x2, y, margin, reduction)` の
  facade 到達
- `Var::margin_ranking_loss(&self, x2, y, margin, reduction)` の
  facade 到達
- `Var::triplet_margin_loss(&self, positive, negative, options,
  reduction)` の facade 到達
- `Var::poisson_nll_loss(&self, target, options, reduction)` の
  facade 到達

承認が得られたら、`Var` への薄い委譲メソッド（`loss_ops::*` を呼ぶ
だけ）を追加し、`LossOpsHoldDoctestGuard`・対応する api_surface の
テストのうち該当名を撤去する（#2166 の 2 関数分は引き続き保留対象の
まま残る場合、ガードは該当関数名のみ縮小する）。

## §6 スコープ外（out-of-scope-tracking.md）

- `compat::Loss`（`compile()`）への 4 損失の追加は #2169 の担当。
  本 PR では `crates/facade/src/compat/training.rs` を変更しない
- GPU 専用カーネル（`BackendOps` の融合メソッド）は別イシュー。本 PR
  では起票しない（承認なしに Issue を作らない規約に従い、必要な
  場合は PR 上でユーザーに提案する）
- CUDA（GB10）／Metal（M4 Max）の実機 parity は未実測。§4・
  `docs/perf/logs/loss-ops-2167/README.md` へ申し送る
- `reduction='none'`（`Reduction` enum の拡張）は既存公開 enum に
  関わるため本イシューの対象外
- `TripletMarginWithDistanceLoss`（任意の距離関数を渡す版）
- §3 に記した境界規約（hinge・`min` 同値・p ノルム 0 の劣勾配選択）の
  PyTorch 実機突合による再確認
