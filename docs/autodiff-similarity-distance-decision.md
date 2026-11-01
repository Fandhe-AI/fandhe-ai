# CosineSimilarity（C5）・PairwiseDistance（C6）の `Var` 演算の公開形の設計判断記録

## §0 位置づけ・承認根拠・境界・推奨

- 本書は追跡表 `docs/compat-matrix-ari-tracker.md` の **行 # 12（C5 CosineSimilarity）・# 13（C6 PairwiseDistance）** の設計記録である（イシュー #3745）。イシュー本文は「項目 13・14」と書くが、追跡表の通し番号とは 1 ずれている（`docs/nn-shrink-bilinear-channel-shuffle-decision.md` §1.0 と同じずれ方）。更新対象は # 12・# 13 行である。
- 流れ: 設計 #3745 → 承認 #3768 → 実装 #3821 → 実機 parity #3860（`<cuda-node>`）・#3861（`<metal-node>`）。
- **承認根拠: なし。未承認（#3768 待ち）。** 本書は何も承認しない。
- 書式は `docs/decision-record-template.md`（#3716）の必須 9 節に従う（§1〜§9 が対応）。
- **設計のみ**で、コード・公開面・tolerance・baseline・ガードレール閾値・依存・`docs/spec` は変えない。
- 行番号は origin/main の `6537d0925` を基準にする。実装 PR の前に引き直すこと。実機は `<cuda-node>`・`<metal-node>`・`<rocm-node>` で表す。

### 現状

- 棚卸し（`docs/compat-unlisted-holds-inventory.md:75`・`:76`）は C5・C6 を「未公開・起票する」と分類した。理由は 3 点。
  1. `crates` 全体に `cosine_similarity` は 0 件。近いものは `Var::cosine_embedding_loss`（`crates/autodiff/src/var.rs:4733`）だけ。
  2. pairwise distance の計算は `triplet_margin_loss` の内部にしか無い。
  3. 2 入力の演算で `Sequential::add_*` には載らない。
- `docs/api-compat-0.14.0-decision.md:53-54` は C5・C6 をすでに「`Var` 演算・追加のみ」と分類している。

### 調査で確定した事実（設計の核）

1. **内部の距離計算**（イシューが挙げる `loss_ops.rs:212`・`:692` はどちらも ATen 参照の doc コメントで、実体は `crates/autodiff/src/eval.rs`）。
   - `p_norm_f64`（`eval.rs:4612`、モジュール内 private）: overflow しないスケール形の p ノルム。NaN は伝播し、`mx==0` なら 0、`±inf` なら inf。
   - `p_norm_grad_f64`（`eval.rs:4641`、`pub(crate)`）: ノルムの勾配。`‖v‖==0` なら勾配 0、`±inf` 要素には均等に分配。
   - `triplet_distance_stats`（`eval.rs:4834`、`pub(crate)`）: 差に `eps` を**先に**足してからノルムを取る。VJP 側 `grad.rs` の `triplet_margin_loss_vjp` も同じヘルパーを使う。
2. **PyTorch 参照版**（v2.14.0、`aten/src/ATen/native/Distance.cpp`。リポの fixture と同じ版）。ソース読解のみで、**実行値は未突合**。
   - `pairwise_distance`（`:57-64`）: `at::norm(x1 - x2 + eps, p, 最後の軸, keepdim)`。broadcast を許す。`eps`・`p` の範囲検査は無い。
   - `cosine_similarity`（`:273-338`）: `eps >= 0` を検査し、broadcast し、各入力の L2 ノルムを `NoGradGuard` の下で `clamp_min_(eps)` して `((x1/n1c)*(x2/n2c)).sum(dim)` を返す。つまり **eps は各ノルムへ個別にかかるクランプ**で、積 `‖x1‖·‖x2‖` にはかからず、**クランプは勾配に記録されない**（クランプ前のノルムの勾配がそのまま流れる straight-through）。
   - 既存の `cosine_embedding_loss_forward`（`eval.rs:4677`）は二乗和へ `1e-12` を足す別規約。**新しい演算はこの経路を再利用しない**。
3. **合成方式の制約**。
   - 合成に使う `sub`／`norm_p`（`reduce_ops.rs:454` → `Op::PNorm`）／`norm_l2`（`Op::VectorNorm`）等の先例は `tensor_product_ops::cdist`（`tensor_product_ops.rs:254`）。
   - `Op::PNorm`・`Op::VectorNorm` は create_graph の C 区分（fail-closed。`docs/autodiff-higher-order-grad-decision.md` §33、実装は #3791）。合成でも今は二階微分が通らない。
   - `Var::clamp` は境界の外側の勾配を 0 にするため、straight-through を合成では再現できない。`Var` にスカラー加算は無く、`+eps` には定数テンソルの broadcast が要る。
4. **ガードの現状**。`crates/facade/src/lib.rs`・`crates/facade/tests/api_surface.rs` に `cosine_similarity`／`pairwise_distance` の否定ガードも保留 doctest ガードも **0 件**。反転するものは無く、#3821 が正ガードを新設する。`LossOpsHoldDoctestGuard`（`lib.rs:4245` 付近）と `LOSS_OPS_FN_NAMES` 系の固定集合プローブ（`api_surface.rs:17083`・`:17111`・`:17241`・`:17290`・`:17401`）は、`loss_ops.rs` に関数を足すと影響を受ける。
5. `Op` は `pub(crate)`（`tape.rs:102`）。variant を足しても公開面は変わらない。

### 推奨

**推奨: 実装する（案 A）。**

| 案 | 内容 | 利点 | 欠点・影響 |
|---|---|---|---|
| A（推奨） | 新モジュール `crates/autodiff/src/similarity_ops.rs` に自由関数 2 件。新 `Op::CosineSimilarity { x1, x2, dim, eps }`・`Op::PairwiseDistance { x1, x2, p, eps, keepdim }`（`pub(crate)`）を足し、forward・VJP を host の f64 で計算して `p_norm_f64`／`p_norm_grad_f64` を再利用。`Var` へ 1 行委譲 | triplet と同じ eps 規約と f64 縮約契約。host 計算のみで 3 バックエンド bit 一致が期待できる（先例 `docs/autodiff-distance-poisson-loss-ops-decision.md` §4）。PyTorch の straight-through 勾配を正確に書ける。`BackendOps`・カーネル・`unsafe`・依存は 0 件 | GPU tape で host 同期が 1 回入る。`Op` の網羅 match に 2 variant が増える。create_graph は false で始まる |
| B | 既存 Op の合成（`cdist` 型） | 新 `Op` が不要。GPU 常駐 | create_graph は A と同じく通らない。straight-through を再現できない。`+eps` に定数テンソルが要る。f32 中間値のため bit 一致でなく REQ-2 判定 |
| C | 専用 GPU カーネル | 速くなりうる | `BackendOps` 追加・`unsafe`・REQ-8 境界検査・新 parity 契約が要る。性能要求は無い |
| D | 対象外 | 公開面不変 | 対応表の行が「ある」にならない |

- **どの案でも今の時点で create_graph は得られない**。
- 戻り値はペアごとのテンソル（例 `[N]`）で損失スカラーではないため、`Reduction` は取らない。
- triplet の既存経路（`triplet_distance_stats`・`triplet_margin_loss_forward`／`_vjp`）は**変更しない**（bit 不変）。

## §1 公開形とガード反転計画

**シグネチャ案**

| 対象 | シグネチャ |
|---|---|
| `Var::cosine_similarity` | `fn cosine_similarity(&self, other: &Var<'t>, dim: usize, eps: f32) -> Result<Var<'t>, AutodiffError>` |
| `Var::pairwise_distance` | `fn pairwise_distance(&self, other: &Var<'t>, p: f32, eps: f32, keepdim: bool) -> Result<Var<'t>, AutodiffError>` |

- PyTorch の既定値（`dim=1`・`eps=1e-8`、`p=2.0`・`eps=1e-6`・`keepdim=false`）は Rust に既定引数が無いので持たず、rustdoc にのみ書く。`dim` は `usize`（負の dim は扱わない。既存 `Var` 演算と同じ）。
- 出力 shape: cosine は入力から `dim` を除いた shape（rank 1 なら `[]`）。pairwise は最後の軸を除いた shape（`keepdim` 真なら長さ 1 で残す）。
- 配置: 自由関数は `crates/autodiff/src/similarity_ops.rs`、`Var` 側は `var.rs` の 1 行委譲。facade の `fandhe_ai::Var` は再エクスポートなので追加配置は不要。`loss_ops.rs` に置かない理由は、`LOSS_OPS_FN_NAMES` の固定集合プローブと `var_loss_ops_methods_are_thin_delegations` の件数（7 件）を動かさないため。
- `nn::` に層構造体は置かない。パラメータを持たない 2 入力の演算で `Module` にも `Sequential` にも載らないため。`add_*` も置かない。

**ガード**

- 反転する既存ガードは**該当なし**（0 件。上記 4）。
- #3821 が新設する正ガード（既存命名にならう）: `var_similarity_ops_methods_are_thin_delegations`、`var_similarity_ops_are_reachable_via_facade_var`、`workspace_declares_similarity_ops_fn_names_only_in_allowed_locations`（`similarity_ops.rs` の自由関数と `var.rs` の委譲を各 1 件だけ許すインベントリ）、`facade_does_not_reexport_or_declare_similarity_ops`、`var_similarity_ops_methods_have_approved_signatures`（自己テスト `..._detects_offense` 付き）。
- 否定プローブとして残す未承認経路: モジュール再エクスポート、`Tensor<f32>`／`Tape` 上の同名メソッド。
- `LossOpsHoldDoctestGuard` とそのドリフト検査は変更しない。

## §2 rustdoc と doctest

- 2 API それぞれに役割要約を書く。内容は PyTorch 相当名、式、eps の位置、ゼロノルムの勾配、NaN と inf の扱い。
- `fandhe_ai::tape()` の CPU だけで通る doctest を置く（GPU 不要）。
- 失敗系は `compile_fail` を使わず、`is_err` を確かめる通常の doctest とテストで検証する（`docs/facade-speculative-decoding-batching-design.md` §18.3 の方針）。

## §3 エラー型

- 新しい variant は足さない。引数違反は `AutodiffError::InvalidArgument`、軸の範囲外は `AutodiffError::Shape(ShapeError::AxisOutOfRange)`、rank 0 は `RankMismatch`（`cdist`／`cross` と同型）、テープ不一致は `check_same_tape`。
- 検証順: ① `check_same_tape` → ② shape 一致（`require_same_shape`、broadcast しない）→ ③ rank >= 1 かつ `dim < rank` → ④ 確保前のバイト数検査（`checked_bytes_for::<f32>`、REQ-8）→ ⑤ スカラー検査 → ⑥ 実体化 → ⑦ host 計算 → ⑧ tape へ積む。
  - ⑤: cosine は `eps` が有限かつ `>= 0`。pairwise は `p` が有限かつ `>= 1`、`eps` が有限かつ `>= 0`（triplet と同じ）。
  - 途中でエラーになっても孤児ノードを残さない。本番経路で `unwrap`／`expect` を使わない。
- 空の縮約軸（`shape[dim]==0`、または最後の軸が 0）は値 0 を返し、勾配は空。cosine は PyTorch と同じ（ノルム 0 を eps にクランプした空和 0）。pairwise は `cdist` の `M==0` と同じ扱い。
- **PyTorch との既知の差分**: broadcast を受け付けない。pairwise の `p` は `0`・`inf`・`(0,1)` を拒否する。pairwise の負の `eps` を拒否する（PyTorch は無検査）。

## §4 テスト計画（#3821）

| 受入基準 | テスト | 区分 | `<cuda-node>` | `<metal-node>` | `<rocm-node>` |
|---|---|---|---|---|---|
| 手計算の forward、ゼロノルム（`eps` クランプ）、`eps=0` のゼロベクトルで NaN、NaN と inf、空の軸、`keepdim`、任意 rank と `dim` | `similarity_ops.rs` 単体テスト | CI | - | - | - |
| 引数拒否（tape 不一致・shape 不一致・rank 0・dim 範囲外・`p<1`・非有限・負の `eps`）と孤児ノードが残らないこと | 同上 | CI | - | - | - |
| 中央差分による勾配突合（クランプ境界と kink は避ける） | `crates/autodiff/tests/similarity_ops.rs` | CI | - | - | - |
| ゼロノルムの straight-through 勾配の手計算 | 同上 | CI | - | - | - |
| triplet の内部距離との bit 一致（各値 = `triplet_distance_stats(..).d_ap as f32`） | `eval`／`grad` の crate 内単体テスト | CI | - | - | - |
| create_graph は型付きエラーで拒否（fail-closed） | `crates/autodiff/tests/create_graph.rs` に節を追加 | CI | - | - | - |
| facade 経由の到達と doctest | `crates/facade/tests/similarity_ops_facade.rs`・rustdoc | CI | - | - | - |
| CPU と NaiveOps の bit 一致 | `crates/facade/tests/similarity_ops_backend_parity.rs::cpu_bit_exact_{forward,backward}_matches_naive_reference` | CI | - | - | - |
| CUDA と CPU の bit 一致 | 同ファイル `cuda_bit_exact_{forward,backward}_matches_cpu_reference` | `#[ignore]` | 未実測（#3860） | - | 該当なし |
| Metal と CPU の bit 一致 | 同ファイル `metal_bit_exact_*`（`cfg(target_os = "macos")`） | `#[ignore]` | - | 未実測（#3861） | 該当なし |
| PyTorch 2.14.0 実行値 fixture との REQ-2 判定（straight-through 勾配を含む） | 実装時に用意 | CI | - | - | - |
| 正ガード 5 件 | `crates/facade/tests/api_surface.rs` | CI | - | - | - |

- 実機の手順と記録先は `docs/perf/logs/similarity-ops-3821/README.md`（#3821 で新設）。
- ROCm は facade 非公開（追跡表 # 24 は保留）なので「該当なし」。
- cfg により 0 本しか実行されなかった結果は成功の根拠にしない。

## §5 数値契約

- 判定は REQ-2 の統一複合判定（相対誤差 1e-3 未満 または 絶対誤差 1e-5 未満）。host 計算のみなので、テストはそれより強い bit 完全一致で書く。
- `RELATIVE_TOLERANCE`／`ABSOLUTE_RESCUE_THRESHOLD`・baseline・ガードレール閾値は変えず、新しい baseline も作らない。
- 長軸縮約は f64 アキュムレータ。要素を先に f64 へ昇格してから二乗・積算し、最後に 1 回だけ `f32` へ下げる（正規化統計と同じ既存規約を守る側の話で、変更ではない）。FMA 契約への影響は無い（matmul 系を経由しない）。
- 数式:
  - cosine: `n_i = sqrt(Σ x_i²)`、`c_i = max(n_i, eps)`、`s = Σ (x1/c1)(x2/c2)`。
  - 勾配: `∂s/∂x1_k = x2_k/(c1·c2) − s·x1_k/(c1·n1)`。`n1 == 0` なら第 2 項は 0（クランプ中も第 2 項を流す PyTorch の straight-through に合わせる）。ゼロベクトル側の勾配は `x2/(eps·c2)`（大きな値になる）、相手側は 0。
  - pairwise: `d = p_norm_f64(x1 − x2 + eps)`（差も f64 で計算）、勾配は `±g·p_norm_grad_f64`。
- PyTorch の f32 計算（差を f32 で丸める、正規化してから積を取る）との差は丸め水準にとどまる見込みだが、**未実測**。実装時に fixture で確認する。

## §6 互換への影響

- 判定（`docs/api-compat-0.14.0-decision.md` §8 の手順）は **追加のみ**。`:53-54` の分類と一致する。
  - `Var`: inherent メソッドが 2 件増える。
  - `Op`: `pub(crate)` なので公開型ではない。
  - `AutodiffError`: variant を足さない。新しい公開型も作らない。
- 利用者が `Var` へ同名のトレイトメソッドを定義していても inherent が優先して解決される。通常のメソッド追加と同じ扱い。

## §7 適用記録・索引・追跡表

- `docs/compat-api-scope.md` への適用記録は承認前なので本 PR では追記しない。承認後に #3821 が追記する文案: 「**適用記録（イシュー #3821・親 <Phase 親>。承認: <#3768 の回答 URL>）**: `Var::cosine_similarity`／`Var::pairwise_distance` の 2 件を追加のみで公開。新規 `Op` 2 件は `pub(crate)`・依存なし。」
- 本 PR で更新するのは `docs/README.md` の索引 1 行と、追跡表 # 12・# 13 の「設計記録」セルだけ。

## §8 ベンチと性能

- 既定経路への影響は無い。新 API の追加のみで triplet 経路は不変。
- GPU tape では host 同期が 1 回入る（先例の loss_ops と同じ）。新経路なので前後比較は不要。専用カーネルはスコープ外。
- 実装時に triplet の共有ヘルパーを書き換える場合だけ、既存の `loss_ops_backend_parity.rs` で bit が変わらないことの確認を必須にする。推奨は書き換えないこと。

## §9 依存

- 依存の追加は無い。許容 12 区分・禁止リストとの照合は該当なし。新規 `unsafe` は無い。

## §10 VJP・create_graph・内部再利用

- forward は `eval.rs` の triplet ヘルパーの近くに `cosine_similarity_forward`／`pairwise_distance_forward` を置く。こうすれば `p_norm_f64` の可視性を変えずに済む（別モジュールでも `pub(crate)` 昇格だけで公開面は変わらない）。VJP は `grad.rs`。
- `tape.rs` の `supports_create_graph`（`:2526`）の網羅 match では、新 2 variant を false 側に置く（ワイルドカードは足さない）。`docs/autodiff-higher-order-grad-decision.md` の分類表への追記要否は #3821 で確かめる。
- create_graph への対応は別の設計 issue の対象とする（所有者承認事項。無断では起票しない）。

## §11 所有者へ戻す論点（#3768 へ渡す）

「確認がないときの扱い」はすべて、確認があるまで実装へ進めない（公開面なしのまま fail-closed）。

| 点 | 選択肢 | 推奨 |
|---|---|---|
| Q1 実装方式 | (a) 新 host Op（案 A） (b) 既存 Op の合成（案 B） | (a) |
| Q2 cosine の eps とゼロノルムの勾配 | (a) PyTorch と同じ（各ノルムへ個別クランプ・straight-through） (b) クランプ式の真の微分（クランプ中は第 2 項 0） (c) `cosine_embedding_loss` と同じく二乗和へ 1e-12 | (a) |
| Q3 broadcast | (a) 同一 shape のみ（`broadcast_to` は利用者側） (b) `cdist` と同じ NumPy 互換 | (a) |
| Q4 pairwise の `p` の範囲 | (a) 有限かつ `>= 1`（triplet と同じ） (b) 有限かつ `> 0`（`norm_p`／`cdist` と同じ） (c) PyTorch と同じく `0`・`inf` も受理 | (a)。`p<1` では差の要素が 0 のとき `p_norm_grad_f64` が `0·inf = NaN` になるため |
| Q5 `eps` の範囲 | (a) 両演算とも有限かつ `>= 0` (b) pairwise だけ PyTorch に合わせ無検査 | (a) |
| Q6 `keepdim` | (a) pairwise に持たせる (b) 持たせず利用者が `unsqueeze` | (a) |
| Q7 層構造体 | (a) 置かない（`Var` 演算のみ） (b) `nn::` に内部構造体（型名公開は #3022 の別判断） | (a) |
| Q8 未承認経路の固定方法 | (a) 正ガードと否定プローブだけ (b) `SimilarityOpsHoldDoctestGuard` も新設 | (a) |
| Q9 create_graph | (a) false で始め別 issue で設計 (b) #3821 で同時に設計・実装 | (a) |

## §12 セキュリティ観点・スコープ外

- A03: `dim`・`p`・`eps`・shape は、tape 操作と確保（`checked_bytes_for`）より前に検証し fail-closed にする（REQ-8）。
- A08: 判定の迂回経路を作らない。正ガードで 1 行委譲を固定する。
- 秘密情報とホスト名は書かない。PyTorch ソースは参照データとして版とパスだけを記録した。

**スコープ外**（新規 issue は所有者の承認なしに起票しない）

- create_graph への対応
- broadcast への対応
- `p=inf`／`0`
- 専用 GPU カーネル
- `Tensor<f32>`／`Tape` 上のメソッド
- 層型名の公開（#3022）
- ONNX（該当する op が無い）
- `Sequential` の保存（2 入力なので #3750 の対象外）
