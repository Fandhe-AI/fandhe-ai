# Hardshrink・Softshrink・Bilinear・ChannelShuffle の公開形の設計判断記録

## §0 位置づけ・共通事項

- 本書は追跡表 `docs/compat-matrix-ari-tracker.md` の C1・C2・C7 の設計記録をまとめる 1 本である。節の分担は次のとおり。
  - **§1 C1 Hardshrink・Softshrink**: イシュー #3742（本 PR が記述）
  - **§2 C2 Bilinear**: イシュー #3743（本 PR が記述）
  - **§3 C7 ChannelShuffle**: イシュー #3746（本 PR が記述）
- 書式は `docs/decision-record-template.md`（#3716）の「保守・運用チェックリスト（必須 9 節）」に従う。先例は `docs/autodiff-spatial-layers-decision.md` §9（#3740）。
- **設計のみ**で、コード・公開面・tolerance・baseline・ガードレール閾値・依存・`docs/spec` は変えない。
- 行番号は origin/main の `9f92a2d22` を基準にする。実機は `<cuda-node>`・`<metal-node>`・`<rocm-node>` で表す。

## §1 C1 Hardshrink・Softshrink（イシュー #3742）

### §1.0 位置づけ・推奨

- 追跡表の **行 # 9**（C1）。イシュー本文は「項目 10」と書くが、追跡表の通し番号とは 1 ずれている（承認 #3768・実機 #3860 も同じずれ方）。更新対象は # 9 行である。
- 流れ: 設計 #3742 → 承認 #3768 → 実装 #3817 → 実機 parity #3860（`<cuda-node>`）・#3861（`<metal-node>`）。保存・export の横断方針は #3750（設計中）。
- **承認根拠: なし。未承認（#3768 待ち）。**
- 現状: `crates/` に `hardshrink`／`softshrink` は 0 件。近い名前は `tanhshrink` のみ（`crates/autodiff/src/var.rs:5690`、`crates/facade/src/compat/sequential.rs:1132`）。棚卸し `docs/compat-unlisted-holds-inventory.md:71` が「未公開・起票する（`Var` 演算 2 件＋`add_*`）」と分類した。

**推奨: 実装する。** 理由は 3 点。

1. 既存 Op の合成だけで作れ、新規 `Op`・`BackendOps`・カーネル・`unsafe`・依存が要らない。
2. `tanhshrink`／`threshold`（#2650・#2678・#2679）と同じ型の追加で、ガード・テスト・保存の先例を流用できる。
3. 互換区分は「追加のみ」（`docs/api-compat-0.14.0-decision.md` §1.2 の # 9 行と一致）。

保存・読込（model_io）と ONNX export は #3750 の横断方針に委ね、本書では決めない。

### §1.1 公開形とガード反転計画

**シグネチャ案**

| 対象 | シグネチャ |
|---|---|
| `Var::hardshrink` | `fn hardshrink(&self, lambd: f32) -> Result<Var<'t>, AutodiffError>` |
| `Var::softshrink` | `fn softshrink(&self, lambd: f32) -> Result<Var<'t>, AutodiffError>` |
| `compat::Sequential::add_hardshrink` | `fn add_hardshrink(mut self, lambd: f32) -> Result<Self, AutodiffError>` |
| `compat::Sequential::add_softshrink` | `fn add_softshrink(mut self, lambd: f32) -> Result<Self, AutodiffError>` |

- `Var` の 2 件は `crates/autodiff/src/var.rs` の 1 行委譲にする（`tanhshrink` と同じ）。facade の `fandhe_ai::Var` は再エクスポートなので配置の追加は要らない。
- `add_*` は構築時に検査するので `Result` を返す（先例 `add_celu`〈`sequential.rs:1087`〉・`add_rrelu`〈`:1155`〉）。
- 引数名は PyTorch に合わせ `lambd`。既定値 0.5 は Rust に既定引数が無いので持たず、rustdoc に PyTorch の既定値を書くだけにする。

**実体の置き場所（推奨）**

- 自由関数は新モジュール `crates/autodiff/src/shrink_ops.rs`（`hardshrink`・`softshrink`・`lambd` 検査関数）。
- 層は新モジュール `crates/autodiff/src/nn/shrink.rs`（`Hardshrink`・`Softshrink`。`Module` 実装は `nn::softmin_threshold::Tanhshrink` と同型）。
- `softmin_threshold_ops.rs` へ足さない理由: その固定契約プローブ（`softmin_threshold_ops_hold_doctest_probe_body_matches_fixed_contract`・`workspace_declares_softmin_threshold_ops_fn_names_only_in_allowed_locations`）の固定集合を動かさずに済む。
- `shrink_ops` モジュールと層型は facade から再エクスポートしない（層型名の公開は #3022 の個別承認 3 条件に従う別判断）。
- `LayerSpec`（`pub(super)`、非公開）に `Hardshrink { lambd: f32 }`・`Softshrink { lambd: f32 }` を足す。

**ガードの扱い**

- `hardshrink`／`softshrink` を固定する既存の否定ガード・保留 doctest ガードは**無い**（`crates/facade/src/lib.rs`・`crates/facade/tests/api_surface.rs` に 0 件）。反転するガードは該当なし。
- `VarActivationOpsHoldDoctestGuard`（`crates/facade/src/lib.rs:3617`）と `SoftminThresholdOpsHoldDoctestGuard`（同 `:7554`）、およびそれらの固定文言ドリフト検査は**変更しない**。
- #3817 が新設する正ガード（既存の命名にならう）:
  - `var_shrink_ops_methods_are_thin_delegations`: `Var::{hardshrink, softshrink}` が `crate::shrink_ops::*` への 1 行委譲であること。
  - `compat_sequential_shrink_layers_add_methods_have_approved_signatures`: `ADD_HARDSHRINK_PARAMS`／`ADD_SOFTSHRINK_PARAMS` 定数との一致と `pub fn` がちょうど 1 件であること。`..._detects_offense` の自己テストを付ける。
  - `compat_sequential_exposes_shrink_add_methods_issue_3817_counts_declarations`
  - `workspace_declares_shrink_ops_fn_names_only_in_allowed_locations`: `shrink_ops.rs` の自由関数・`var.rs` の委譲・`sequential.rs` の `add_*` の各 1 件だけを許すインベントリ。
  - `facade_does_not_reexport_or_declare_shrink_ops`: facade の src が `shrink_ops` や層型を再エクスポートしていないこと。
- 未承認経路（モジュール再エクスポート・`Tape`／`Tensor<f32>` 上のメソッド）は否定プローブとして残す。保留 doctest ガードを新設するかは論点 Q4。

### §1.2 rustdoc と doctest

- 4 つの公開 API それぞれに、役割要約（PyTorch 相当名・式・`lambd` の条件・NaN の扱い）と利用例 doctest を置く。例は CPU の `fandhe_ai::tape()` だけで動く形にする（facade 単独で通る。GPU 不要）。
- 失敗系（`lambd` が負・NaN・inf）は `compile_fail` を使わず、`Result::is_err` を確かめる通常の doctest／テストで検証する（`docs/facade-speculative-decoding-batching-design.md` §18.3 の方針）。

### §1.3 エラー型

- 新しい variant は足さない。既存の `AutodiffError::InvalidArgument(String)`（`crates/autodiff/src/error.rs:58`、`#[non_exhaustive]`）を使う。
- 検査: `lambd` は有限かつ `lambd >= 0`（`-0.0` は受け入れる）。両演算で同じ条件。
- 順序: `lambd` の検査 → mask の構築 → tape への積み込み。エラー時に孤児ノードを残さない。
- `add_*` は構築時に同じ関数で検査し `Err` を返す。本番経路で `unwrap`／`expect` を使わない。範囲外は fail-closed。
- PyTorch との差（論点 Q1）: Softshrink は PyTorch でも負の `lambd` を拒否する。Hardshrink は PyTorch では拒否されず負の `lambd` は恒等写像になる（ソースの定義に基づく記述で**実測はしていない**）。本推奨は Hardshrink も負を拒否するため、既知の差分として記録する。

### §1.4 テスト計画

| 受入基準 | テスト | 区分 | `<cuda-node>` | `<metal-node>` | `<rocm-node>` |
|---|---|---|---|---|---|
| 手計算の forward・backward、境界 `\|x\|==λ`、NaN、`λ=0`、空テンソル | `crates/autodiff/src/shrink_ops.rs` 単体テスト | CI | - | - | - |
| 中央差分による勾配突合 | 同上 `*_matches_central_difference` | CI | - | - | - |
| `lambd` の拒否（負・NaN・±inf）と孤児ノード無し | 同上 | CI | - | - | - |
| create_graph の二階有限差分 | `crates/autodiff/tests/create_graph.rs` に節を追加 | CI | - | - | - |
| facade 経由の到達と doctest | `crates/facade/tests/shrink_ops_facade.rs`・rustdoc | CI | - | - | - |
| `add_*` の forward・`Err`・fit | `crates/facade/tests/compat_sequential_shrink_layers.rs` | CI | - | - | - |
| CPU と NaiveOps の bit 一致 | `crates/facade/tests/shrink_ops_backend_parity.rs::cpu_bit_exact_{forward,backward}_matches_naive_reference` | CI | - | - | - |
| CUDA と CPU の bit 一致 | 同ファイル `cuda_bit_exact_{forward,backward}_matches_cpu_reference` | `#[ignore]` | 未実測（#3860） | - | 該当なし |
| Metal と CPU の bit 一致 | 同ファイル `metal_bit_exact_*`（`cfg(target_os = "macos")`） | `#[ignore]` | - | 未実測（#3861） | 該当なし |
| 層経路のバックエンド parity | `crates/facade/tests/compat_sequential_shrink_layers_backend_parity.rs`（`cuda_*`／`metal_*`） | CI＋`#[ignore]` | 未実測（#3860） | 未実測（#3861） | 該当なし |
| 正ガード 5 件（§1.1） | `crates/facade/tests/api_surface.rs` | CI | - | - | - |

- 実機の手順と記録先は `docs/perf/logs/shrink-ops-3817/README.md`（#3817 で新設）。cfg で 0 本しか実行されなかった結果は pass の根拠にしない。
- ROCm は facade 非公開（追跡表 # 24 保留）のため「該当なし」。

### §1.5 数値契約

- 判定は REQ-2 の統一複合判定（相対誤差 1e-3 未満 または 絶対誤差 1e-5 未満）。演算は選択と IEEE 減算 1 回だけなので、テストはそれより強い **bit 完全一致**で書く（`threshold` と同じ扱い）。
- `RELATIVE_TOLERANCE`／`ABSOLUTE_RESCUE_THRESHOLD`・parity baseline・ガードレール閾値は変えない。新しい baseline は作らない。
- FMA 契約・f64 長軸縮約の契約への影響: なし（縮約も積和も無い）。

### §1.6 互換への影響

- 判定（`docs/api-compat-0.14.0-decision.md` §8 の手順）: **追加のみ**。
  - `Var`（autodiff。facade が再エクスポート）: inherent メソッド 2 件の追加。
  - `compat::Sequential`（フィールド非公開、同 §1.3）: inherent メソッド 2 件の追加。
  - `AutodiffError`: variant を足さない。`LayerSpec`: `pub(super)` で公開型ではない。
- 利用者が `Var` に同名のトレイトメソッドを生やしている場合は inherent メソッドが優先して解決される。Rust の通常の追加と同じ扱いで、破壊的には数えない。

### §1.7 適用記録・索引・追跡表

- `docs/compat-api-scope.md` への適用記録は承認前なので本 PR では追記しない。承認後の #3817 が次の文案で追記する（公開状態は未公開から変更）:
  `**適用記録（イシュー #3817・親 <Phase 親>。承認: <#3768 の回答 URL>）**: `Var::hardshrink`／`Var::softshrink` と `compat::Sequential::add_hardshrink`／`add_softshrink` の 4 件を追加のみで公開。新規 `Op`・依存なし。`
- 本 PR の更新は `docs/README.md` の索引 1 行と、追跡表 # 9 行の「設計記録」セルのみ。「承認」「0.14.0 の区分」「正ガード」列は変えない（正ガードは #3817 で `path:line` を書く）。

### §1.8 ベンチと性能

- 既定経路への影響: なし（新 API の追加のみ。既存の演算・カーネルに触れない）。
- 補足: mask 構築で `host_view` を通るため GPU tape では同期が 1 回入る（`threshold` と同じ）。新経路なので前後比較は不要。専用カーネルによる最適化はスコープ外。

### §1.9 依存

- 依存の追加: なし。許容 12 区分・禁止リストとの照合は該当なし。新規 `unsafe`: なし。

### §1.10 VJP・create_graph・合成方式

- 新規 `Op` は作らない。既存 Op の合成で書く。独自 VJP は書かず、合成元から自動で決まる。
  - `hardshrink(x, λ) = masked_fill(x, mask(-λ <= v <= λ), 0.0)`。mask は `crate::activation_ops::build_value_mask`（`activation_ops.rs:83`）で作る。NaN は比較が偽になり素通し（勾配 1）。
  - `softshrink(x, λ) = x.sub(&x.clamp(-λ, λ))`（`Var::clamp` は `var.rs:934`）。範囲内は勾配 0、境界 `|x| == λ` も 0（clamp は境界を含めて係数 1 なので 1 − 1）、範囲外は 1。NaN 入力は forward が NaN、勾配は 1 − 0 = 1。
  - 最終形は実装時に決めてよい（`where_cond` 形も可）。ただし「新規 `Op` ゼロ」「§1.5 の数値契約」「下の既知の差分」の 3 点は変えない。
- **create_graph は対象にする（追加作業なし）**。`MaskedFill` は #3084（`docs/autodiff-higher-order-grad-decision.md` §29）、`ScalarUnary`（`Clamp` を含む）と減算系は同 §16 で対象のため、合成すれば二階微分がそのまま通る。#3817 のテストに CPU の二階有限差分突合を各 1 本入れる。clamp／mask の境界は不連続なので、境界から離れた入力で検証する。
- 既知の差分（いずれも**実測していない**。実装時に参照版のソースで確認する）:
  - Softshrink の NaN 入力: PyTorch のカーネルには版によって 0 を返すものがある。本合成は NaN を伝播する。
  - Softshrink の範囲内の出力: 本合成は `x - x = +0`。PyTorch は負の `x` で `-0` になりうるので符号付きゼロが異なりうる。
  - Hardshrink の負の `lambd`: §1.3 のとおり拒否する。

### §1.11 所有者へ戻す論点（#3768 へ渡す）

「確認がないときの扱い」はすべて、確認があるまで実装へ進めない（現状の公開面なしを維持する fail-closed）。

| 点 | 選択肢 | 推奨 |
|---|---|---|
| Q1 `lambd` の検証 | (a) 両方で有限かつ `>= 0` を要求 (b) PyTorch に合わせ Softshrink だけ `>= 0`、Hardshrink は無検査 (c) 有限だけ要求 | (a) |
| Q2 NaN・符号付きゼロの差 | (a) 合成のまま NaN を伝播し既知の差分として記録 (b) PyTorch の参照版に合わせ mask で 0 に置換 | (a)（参照版の挙動は実装時に確認） |
| Q3 実体の置き場所 | (a) 新モジュール `shrink_ops`／`nn::shrink` (b) `softmin_threshold_ops` に追記 | (a) |
| Q4 未承認経路の固定方法 | (a) ソース走査の正ガードと否定ガードだけ (b) `ShrinkOpsHoldDoctestGuard` も新設 | (a) |
| Q5 保存・export | #3750 の横断方針に従う。それまでは `LayerSpec` に実 variant を置き、保存を `UnsupportedModel` で fail-closed にするか、kind `hardshrink`／`softshrink`（`params`: `lambd`）まで実装するかを #3750 が決める | #3750 に委ねる |

- Q5 補足: `crates/facade/src/compat/model_io.rs:5023` の単体テストが kind allowlist を 70 種に固定している。kind を足す場合は #3750 と数を合わせる必要がある。

### §1.12 セキュリティ観点・スコープ外

- A03: `lambd` は保存ファイル経由も含む外部入力として、tape 操作・確保より前に検証して fail-closed にする。A08: 判定の迂回経路を作らない（正ガードで 1 行委譲を固定）。秘密情報・ホスト名は書かない。
- スコープ外: ONNX export（ONNX の `Shrink` op。#3750）、専用 GPU カーネル、層型名の facade 公開（#3022）、`Tensor<f32>`／`Tape` 上のメソッド。新しいイシューは起票しない（所有者の承認事項）。
- #3817 は実機 parity が済むまで `Refs` 運用とする。

## §2 C2 Bilinear（イシュー #3743）

行番号は origin/main の `b52c981c4` を基準にする。

### §2.0 位置づけ・推奨

- 追跡表の **行 # 10**（C2）。イシュー本文は「項目 11」と書くが、§1.0 と同じく 1 ずれている。更新対象は # 10 行である。
- 流れ: 設計 #3743 → 承認 #3768（論点 3）→ 実装 #3818 → 実機 parity #3860（`<cuda-node>`）・#3861（`<metal-node>`）。保存・export の横断方針は #3750（設計中）。
- **承認根拠: なし。未承認（#3768 待ち）。**
- 現状: `crates/` に `nn.Bilinear` 相当は 0 件（`bilinear` を含むのは補間の `InterpolateMode::Bilinear` と `eval.rs` の `bilinear_*` だけ）。棚卸し `docs/compat-unlisted-holds-inventory.md:72` が「未公開・起票する」と分類した。facade に Bilinear 層の否定ガード・保留 doctest ガードは無い。したがって「反転」は正ガードを新しく作ることを指す。

| 案 | 内容 | 利点 | 欠点・影響 |
|---|---|---|---|
| **A（推奨）** | `Var::bilinear` を足す（自由関数を新モジュールに置き 1 行で委譲）。autodiff 内部に層 `nn::Bilinear`／`BilinearVars` を置くが facade からは再エクスポートしない。`Sequential::add_*` は設けない | 新規 `Op`・カーネル・`unsafe`・依存が 0 件。公開名の追加は inherent メソッド 1 件。PyTorch の `F.bilinear` と同じ形 | パラメータ管理と学習ループは利用者が書く |
| B | A に加え `FunctionalBuilder` にパラメータ付きの 2 入力ノードを足す | `FunctionalModel` の `fit`・`state_dict` で学習できる | 結合ノードは層を持たない前提（`functional.rs` 冒頭 doc）を崩す。キー体系・bind・保存（#3750）の設計が要り、公開面も大きい |
| C | 専用 GPU カーネル | 中間テンソルを作らず速くなりうる | `BackendOps` 追加・`unsafe`・境界検査（REQ-8）・新しい parity 契約が要る。性能要求が無い |
| D | 対象外にする | 公開面は変わらない | 対応表の行が「ある」にならない |

**推奨: 実装する（案 A）。** 理由は 4 点。

1. 既存 Op の合成だけで作れ、新規 `Op`・`BackendOps`・カーネル・`unsafe`・依存が要らない。
2. `kron`／`tensordot`／`cdist`（`crates/autodiff/src/tensor_product_ops.rs`・#2640・#2678）と同じ型の追加で、ガードとテストの先例を使える。
3. 互換区分は「追加のみ」。
4. PyTorch でも `nn.Bilinear` は `forward(input1, input2)` を取り `nn.Sequential` に載らない。入力が 2 つの層を `Sequential` に載せないのは同じ扱いである。

2 入力の経路について、#3741 の独自 Module 経路（`nn::Module::forward(&self, TapeRef<'t>, &Var<'t>)`・`crates/facade/src/nn/module.rs:45`）も入力は 1 つなので、それだけでは解決しない。利用者は自分の構造体に 2 入力の inherent メソッドを書き、中で `Var::bilinear` を呼ぶ（重みは `TapeRef::var`〈`crates/facade/src/lib.rs:2011`〉で葉にする）。案 B は論点 Q3 で所有者に問う。

### §2.1 公開形とガード反転計画

**シグネチャ案**

| 対象 | シグネチャ |
|---|---|
| `Var::bilinear` | `fn bilinear(&self, x2: &Var<'t>, weight: &Var<'t>, bias: Option<&Var<'t>>) -> Result<Var<'t>, AutodiffError>`（`self` が `x1`。PyTorch `F.bilinear(input1, input2, weight, bias)` と同じ並び） |

- 形状: `x1 = [*, in1]`、`x2 = [*, in2]`（先頭の次元は完全一致）、`weight = [out, in1, in2]`（PyTorch と同じ並び。Q1）、`bias = [out]`、出力 `[*, out]`。先頭の次元は `[N, in]` に平らにして計算し、最後に `[*, out]` へ戻す。受け入れる rank は Q2（推奨: rank 2 以上。rank 1 は拒否し、PyTorch との差として記録する。PyTorch の rank 1 の挙動は**未確認**で、実装時に参照版で確かめる）。
- 実体の置き場所: 自由関数は新モジュール `crates/autodiff/src/bilinear_ops.rs`。`var.rs` には `crate::bilinear_ops::bilinear(self, x2, weight, bias)` への 1 行委譲だけを置く（`Var::cdist`〈`var.rs:5413`〉と同じ）。`tensor_product_ops.rs` に足さない理由は、その固定契約プローブ（`tensor_product_ops_hold_doctest_probe_body_matches_fixed_contract`・`workspace_declares_tensor_product_ops_fn_names_only_in_allowed_locations`）の固定集合を動かさずに済むため（§1.1 で別モジュールにしたのと同じ理由）。
- 層: autodiff 内部に `crates/autodiff/src/nn/bilinear.rs` を新設し、`Bilinear`（`new(in1, in2, out, bias, seed)`／`from_parameters`／`bind`）と `BilinearVars<'t> { pub weight, pub bias }`（`forward(&self, x1, x2)`）を置く。`Linear`／`LinearVars`（`crates/autodiff/src/nn/linear.rs:53,565`）と同じ構成。初期化は `U(-1/√in1, 1/√in1)`（PyTorch のソースの定義に基づく記述で**未実測**）。`in1 == 0` は bound が inf になるので引数として弾く（`Linear::new` の `in_features == 0` と同じ。`linear.rs:81-91`）。
- facade: `fandhe_ai::Var` は autodiff の再エクスポートなので `Var::bilinear` は追加の配置なしで届く。`bilinear_ops`・`Bilinear`・`BilinearVars` は再エクスポートしない。#3022 の (iv) 条件 (a)「その型を名指ししないと使えない操作がある」を満たさないため。`LinearVars`（`lib.rs:281`）の先例に合わせるかは Q4。
- `Sequential::add_bilinear` は**作らない**（`Sequential::forward` は入力 1 つ。`sequential.rs:2540`）。`LayerSpec` も増やさない。
- 反転する既存ガード: **該当なし**。`TensorProductOpsHoldDoctestGuard`（`lib.rs:6377`）とその固定文言検査は変えない。

**#3818 が新しく作る正ガード（既存の命名にならった案）**

- `var_bilinear_ops_method_is_thin_delegation`: 本体が `crate :: bilinear_ops :: bilinear ( self , x2 , weight , bias )` と一致すること（`var_activation_ops_methods_are_thin_delegations`〈`api_surface.rs:12228`〉と同じ形）。
- `var_bilinear_is_reachable_via_facade_only`: `fandhe_ai::{Var, Tensor, AutodiffError}` だけで fn ポインタのシグネチャが一致し、手計算できる期待値が出ること（スタブでは通らない）。
- `workspace_declares_bilinear_ops_fn_names_only_in_allowed_locations`: `bilinear_ops.rs` の自由関数と `var.rs` の委譲がそれぞれ 1 件だけであることのインベントリ。
- `facade_does_not_reexport_or_declare_bilinear_ops`: facade の src が `bilinear_ops`・`Bilinear`・`BilinearVars` を再エクスポートも宣言もしていないこと。違反の種類ごとに検出できるかを合成入力で確かめる自己テスト `..._detects_each_category` を付ける。
- `compat_sequential_has_no_bilinear_add_method`: `scan_type_impl_surface(content, "Sequential")` に `add_bilinear` が無いこと（2 入力の層を載せない決定を固定する）。

否定プローブとして残す範囲はモジュールの再エクスポート、`Tape`／`Tensor<f32>` 上のメソッド、層型の再エクスポート。保留 doctest ガード（`BilinearOpsHoldDoctestGuard`）を作るかは Q5（推奨: 作らない）。

### §2.2 rustdoc と doctest

- `Var::bilinear` に役割要約（PyTorch 相当名・式・形状・`weight` の並び・bias の扱い）と、CPU の `fandhe_ai::tape()` だけで動く doctest を置く。重みを `var` で葉にし、手計算した値と比べる。GPU は要らない。
- 2 入力の書き方の例（独自構造体の inherent メソッドから `Var::bilinear` を呼ぶ。重みは `nn::init::uniform`〈`crates/autodiff/src/nn/init.rs:687`〉で作る）を rustdoc に 1 本置く。
- 失敗系（形状不一致・rank 不足・別 tape）は `compile_fail` を使わず、`Result::is_err` と variant を確かめる通常の doctest／統合テストで検証する（`docs/facade-speculative-decoding-batching-design.md` §18.3 の方針）。

### §2.3 エラー型

- 新しい variant は足さない（`AutodiffError` は `#[non_exhaustive]`。`crates/autodiff/src/error.rs:19-21`）。
- 別 tape → `TapeMismatch`（`check_same_tape`）。rank・形状の不一致（`x1` の最後の次元 ≠ `weight[1]`、`x2` の最後の次元 ≠ `weight[2]`、先頭の次元の不一致、bias ≠ `[out]`、`weight` の rank ≠ 3）→ `Shape(ShapeError)`（使えない意味論のエラーは `InvalidArgument`）。層の構築時の `in1 == 0` → `InvalidArgument`。
- 検査の順: tape の一致 → rank → 各次元 → tape への積み込み（reshape・matmul）。エラーのときに孤児ノードを残さない。本番経路で `unwrap`／`expect` を使わない。範囲外は fail-closed。
- 大きさ 0 の軸: `N == 0` と `out == 0` は空テンソルとして通す。`in1 == 0`／`in2 == 0` は縮約が空なので bias だけが出る形になる（`matmul_zero_size_axis` の先例。挙動は #3818 のテストで固定する）。

### §2.4 テスト計画

| 受け入れ基準 | テスト | 区分 | `<cuda-node>` | `<metal-node>` | `<rocm-node>` |
|---|---|---|---|---|---|
| 手計算の forward・backward（4 勾配）、bias なし、先頭の次元が多い場合、大きさ 0 の軸 | `crates/autodiff/src/bilinear_ops.rs` の単体テスト | CI | - | - | - |
| 中央差分による勾配の突合 | 同上 `*_matches_central_difference` | CI | - | - | - |
| 形状・rank・tape の拒否と孤児ノードが無いこと | 同上 | CI | - | - | - |
| create_graph の二階有限差分 | `crates/autodiff/tests/create_graph.rs` に節を足す | CI | - | - | - |
| 層 `nn::Bilinear` の初期化範囲・`from_parameters` の検査・`bind`→`forward` | `crates/autodiff/src/nn/bilinear.rs` の単体テスト | CI | - | - | - |
| facade 経由で届くことと doctest | `crates/facade/tests/bilinear_ops_facade.rs`・rustdoc | CI | - | - | - |
| CPU と f64 ホスト参照の突合（REQ-2 の複合判定）、CPU と NaiveOps の合成の一致 | `crates/facade/tests/bilinear_ops_backend_parity.rs::cpu_*` | CI | - | - | - |
| CUDA と CPU の parity（forward・backward） | 同ファイル `cuda_bilinear_{forward,backward}_matches_cpu_reference` | `#[ignore]` | 未実測（#3860） | - | 該当なし |
| Metal と CPU の parity | 同ファイル `metal_bilinear_*`（`cfg(target_os = "macos")`） | `#[ignore]` | - | 未実測（#3861） | 該当なし |
| 正ガード 5 件と自己テスト（§2.1） | `crates/facade/tests/api_surface.rs` | CI | - | - | - |

- 実機の手順と記録先は `docs/perf/logs/bilinear-ops-3818/README.md`（#3818 で新設）。cfg で 0 本しか動かなかった結果は pass の根拠にしない。Linux や M4 Max 開発機の代替値で pass にしない。
- ROCm は facade 非公開（追跡表 # 24 は保留）なので「該当なし」。

### §2.5 数値契約

- 判定は REQ-2 の統一複合判定（相対誤差 1e-3 未満 または 絶対誤差 1e-5 未満）。
- §1 と違い、**バックエンド間の bit 一致は主張しない**。経路が `matmul`（FMA 契約。CUDA の TF32 opt-in も効く）と `sum`（CPU は f64 チャンクの 2 段）を通るため。CPU と NaiveOps の比較は同じ合成どうしで行う。
- 縮約は 2 段（in1 は matmul の連続した K ループ、in2 は `sum`）で、どちらも既存 Op の既存の契約のまま。新しい結合順序のカーネルは作らないので、baseline 方式（coding-rust.md の Tensor Core／split-K 節）の対象ではない。実機で複合判定を外れたら tolerance を緩めず所有者へ戻す。
- `RELATIVE_TOLERANCE`／`ABSOLUTE_RESCUE_THRESHOLD`・parity baseline・ガードレール閾値は変えない。新しい baseline は作らない。FMA 契約・f64 長軸縮約の契約への影響はない。

### §2.6 互換への影響

判定（`docs/api-compat-0.14.0-decision.md` §8 の手順）: **追加のみ**。

- `Var`: inherent メソッド 1 件を足す。`AutodiffError`: variant を足さない。`compat::Sequential`: 変えない。
- `Bilinear`／`BilinearVars`: autodiff 内部だけ（Q4 で再エクスポートが選ばれても新しい型の追加で「追加のみ」）。
- 利用者が `Var` に同名のトレイトメソッドを持つ場合は inherent が先に解決される。通常の追加と同じ扱いで破壊的には数えない。
- 記録のずれ: `api-compat-0.14.0-decision.md:51`（§1.2 の # 10 行）は「層型・`Var` 演算・`add_*`」と見込むが、本設計は `add_*` を作らない。この doc は本 PR では直さず、Q7 として #3768 で確かめ #3818 で直す。

### §2.7 適用記録・索引・追跡表

- `docs/compat-api-scope.md` の適用記録は承認前なので本 PR では書かない。承認後に #3818 が追記する文案: `**適用記録（イシュー #3818・親 <Phase 親>。承認: <#3768 の回答 URL>）**: Var::bilinear を追加のみで公開。新規 Op・依存なし。Sequential::add_* は設けない（2 入力）。公開状態: 公開済み`。
- 本 PR の更新は `docs/README.md` の索引 1 行と、追跡表 # 10 行の「設計記録」セルだけ。「承認」「0.14.0 の区分」「正ガード」列は変えない（正ガードの `path:line` は #3818 が書く）。

### §2.8 ベンチと性能

- 既定経路への影響はない（新しい API を足すだけで既存の演算・カーネルに触れない）。
- 中間テンソル `[N, out, in2]` と forward ごとの weight の permute／reshape の分だけメモリと時間がかかる。新しい経路なので前後比較は要らない（未実測）。専用カーネル（案 C）はスコープ外。

### §2.9 依存

- 依存の追加: なし。許容 12 区分・禁止リストとの照合: 該当なし。新規 `unsafe`: なし。

### §2.10 VJP・create_graph・合成方式

新規 `Op` は作らない。独自の VJP は書かず、合成元の Op から自動で決まる。

- 合成: `W = [out, in1, in2]` を `permute([1, 0, 2])`（`var.rs:2631`）→ `reshape([in1, out·in2])`（`:2531`）→ `x1f.matmul(W′)`（rank 2。`:394`）→ `reshape([N, out, in2])` → `mul(x2f.unsqueeze(1))`（`:689`・`:2976`。broadcast）→ `sum(Some(2))`（`:1166`）→ bias があれば `add(bias)`（`:666`）→ `reshape([*, out])`。permute した view への reshape に `contiguous` が要るかは実装時に確かめる。
- 選ばない形: `einsum` は 1〜2 オペランドに限られる（`var.rs:2732` の doc）ので 3 オペランドの一括縮約には使えない。2 段目を rank 3 の `matmul`（`gemm_batched`）にすると create_graph が `Backward` で拒否される（同 doc・`docs/autodiff-higher-order-grad-decision.md` §14）。`x1 ⊗ x2` を作る 1 回の GEMM は中間テンソルが `[N, in1·in2]` と大きい。
- **create_graph は対象にする（追加の作業なし）**。使うのは rank 2 の `MatMul`・`Permute`・`Reshape`・`Mul`・`BroadcastTo`・単一軸の `Sum`・`Add` で、いずれも `docs/autodiff-higher-order-grad-decision.md:119`（§8 の対象）と §14 に入っている。#3818 のテストに CPU の二階有限差分の突合を入れる。
- PyTorch との差（rank 1 の入力の扱い、weight の並び）は PyTorch のソースの定義に基づく記述で**未実測**。実装時に参照版で確かめる。

### §2.11 所有者へ戻す論点（#3768 の論点 3 へ渡す）

確認がないときの扱いはどれも、確認があるまで実装へ進めない（公開面が無い今の状態を保つ fail-closed）。

| 点 | 選択肢 | 推奨 |
|---|---|---|
| Q1 weight の並び | (a) PyTorch と同じ `[out, in1, in2]` (b) 内部で扱いやすい `[in1, out, in2]` | (a)（state_dict の相互運用） |
| Q2 受け入れる rank | (a) rank 2 以上・先頭の次元は任意で完全一致 (b) rank 2 だけ（`LinearVars::forward` と同じ） | (a) |
| Q3 2 入力の学習経路 | (a) `Var::bilinear` だけ (b) `FunctionalBuilder` にパラメータ付き 2 入力ノードも足す | (a)（(b) は別の承認） |
| Q4 `BilinearVars` の再エクスポート | (a) 出さない (b) `LinearVars` の先例に合わせ root に出す | (a) |
| Q5 未承認経路の固め方 | (a) 正ガードと否定ガードだけ (b) `BilinearOpsHoldDoctestGuard` も作る | (a) |
| Q6 保存・export | #3750 の横断方針に従う（`Sequential` に載せないので現状の model_io・ONNX export に影響しない） | #3750 に委ねる |
| Q7 記録のずれ | `api-compat-0.14.0-decision.md:51` の「`add_*`」を #3818 で「`Var` 演算（`add_*` なし）」に直す | 直す（#3818） |

### §2.12 セキュリティ観点・スコープ外

- A03: 形状は tape 操作や確保より前に検査して fail-closed にする（外部由来の重みを `from_parameters` で受ける場合も含む）。A04／A08: 判定の迂回経路を作らない（1 行委譲を正ガードで固定）。REQ-12: 生の `Tape`・内部の層型を公開シグネチャに出さない。秘密情報・ホスト名は書かない。
- スコープ外: `FunctionalBuilder` のノード（Q3 (b)）、専用 GPU カーネル（案 C）、層型名の facade 公開（#3022）、ONNX export・model_io（#3750）、`Tensor<f32>`／`Tape` 上のメソッド。新しいイシューは起票しない（所有者の承認事項）。
- #3818 は実機 parity が済むまで `Refs` 運用とする。

## §3 C7 ChannelShuffle（イシュー #3746）

行番号は origin/main の `6537d0925` を基準にする。PyTorch についての記述はすべてソースの定義に基づく記述で、**未実測**である（実装時に参照版で確かめる）。

### §3.0 位置づけ・推奨

- 追跡表の **行 # 14**（C7）。イシュー本文は「項目 15」と書くが、§1.0・§2.0 と同じく 1 ずれている。更新対象は # 14 行である。
- 流れ: 設計 #3746 → 承認 #3768（論点 3・4）→ 実装 #3822 → 実機 parity #3860（`<cuda-node>`）・#3861（`<metal-node>`）。保存・export の横断方針は #3750（設計中）。
- **承認根拠: なし。未承認（#3768 待ち）。**
- 現状: `crates/` に `channel_shuffle`／`ChannelShuffle` は 0 件。棚卸し `docs/compat-unlisted-holds-inventory.md:77` が「未公開・起票する（優先度は低い。`add_pixel_shuffle` と同型。`reshape`／`permute` で合成できる）」と分類した。先例は PixelShuffle（`crates/autodiff/src/nn/pixel_shuffle.rs`・`docs/autodiff-pixel-shuffle-decision.md`）。

| 案 | 内容 | 利点 | 欠点・影響 |
|---|---|---|---|
| **A（推奨）** | `Var::channel_shuffle`＋`Sequential::add_channel_shuffle` を既存 Op（`contiguous`・`reshape`・`permute`）の合成で作る。層型 `nn::ChannelShuffle` は autodiff 内部に置き facade からは再エクスポートしない | 新規 `Op`・カーネル・`unsafe`・依存が 0 件。PixelShuffle と同型でガード・テスト・保存の先例を使える | GPU tape では `Contiguous` がホストで eager 実行される（§3.8） |
| B | `add_channel_shuffle` だけを作る | 公開名が 1 件 | `Var` から直接呼べず、独自 Module（#3741）の利用者が困る |
| C | 専用 `Op`／カーネル | 中間コピーを減らせる | `BackendOps` 追加・REQ-8 の境界検査・新しい parity 契約が要る。性能要求が無い |
| D | 対象外にする | 公開面は変わらない | 対応表の行が「ある」にならない |

**推奨: 実装する（案 A）。** 理由は 4 点。

1. 新規 `Op`／`BackendOps`／カーネル／`unsafe`／依存が要らない。
2. PixelShuffle と同型で、ガード・テスト・保存の先例をそのまま使える。
3. 互換区分は「追加のみ」。
4. PyTorch の `nn.ChannelShuffle(groups)` は 1 入力で `nn.Sequential` に載る。

### §3.1 公開形とガード反転計画

**シグネチャ案**

| 対象 | シグネチャ |
|---|---|
| `Var::channel_shuffle` | `fn channel_shuffle(&self, groups: usize) -> Result<Var<'t>, AutodiffError>` |
| `compat::Sequential::add_channel_shuffle` | `fn add_channel_shuffle(mut self, groups: usize) -> Result<Self, AutodiffError>`（構築時に検査するので `Result`。先例 `add_pixel_shuffle`〈`crates/facade/src/compat/sequential.rs:2475`〉） |

- 軸の意味（Q1）: チャネル軸は **dim 1** に固定し、入力は `(N, C, *)` の rank 3 以上とする。並べ替えは `out[:, j*g+i] = in[:, i*(C/g)+j]`（`g = groups`）。PyTorch の native 実装は `reshape({b, g, c/g, -1}).permute({0,2,1,3})` に当たる定義で、これに合わせる。`pixel_shuffle` は `rank-3` 軸を使うが、rank 5 では C が dim 1 にならないためその方式は取らない。
- 実体の置き場所（Q3）: 新モジュール `crates/autodiff/src/nn/channel_shuffle.rs` に次を置く。
  - `channel_shuffle_out_shape`（`pub(crate)`。forward と forward_host が共有する判定）
  - `channel_shuffle_forward`（`pub(crate)`。`Var` と層が共有する本体）
  - `pub struct ChannelShuffle { groups }`（`new`／`groups`／`forward`／`forward_host`）
- `nn/mod.rs` に `mod`／`pub use`／`pub(crate) use` を足し、`nn/module.rs` に `impl Module for ChannelShuffle`（`:1957` の PixelShuffle と同型。`named_parameters` は既定の空）を足す。`var.rs` には `crate::nn::channel_shuffle_forward(self, groups)` への 1 行委譲だけを置く（`Var::pixel_shuffle`〈`var.rs:3073`〉と同じ）。
- `pixel_shuffle.rs` に足さない理由は、その固定契約（`PIXEL_SHUFFLE_FACADE_FN_NAMES`・`PixelShuffleHoldDoctestGuard`）の集合を動かさずに済むため（§1.1・§2.1 と同じ）。
- `LayerSpec`（`pub(super)`）に `ChannelShuffle { groups: usize }` を足す（`sequential.rs:511` の隣）。
- facade: `fandhe_ai::Var` は再エクスポートなので追加の配置は要らない。`ChannelShuffle` 型は #3022 の個別承認 3 条件を満たさないので再エクスポートしない。
- 反転する既存の否定ガード・保留 doctest ガード: **該当なし**。「反転」は正ガードを新しく作ることを指す。`PixelShuffleHoldDoctestGuard` と `PIXEL_SHUFFLE_HOLD_PROBE_BODY` は変えない。`Sequential` の `add_*` 総数を固定するガードは `api_surface.rs` に無い。

**#3822 が新しく作る正ガード（`api_surface.rs` の pixel_shuffle 群の命名にならった案）**

- `workspace_declares_channel_shuffle_facade_fn_names_only_in_approved_locations`: `channel_shuffle` は `autodiff/src/var.rs` に、`add_channel_shuffle` は `facade/src/compat/sequential.rs` に、各 1 件だけ。
- `var_channel_shuffle_method_is_thin_delegation`: 本体が `crate : : nn : : channel_shuffle_forward ( self , groups )` と一致する。
- `var_channel_shuffle_method_is_reachable_via_facade_only`: fn ポインタの型が一致し、手計算できる値（`[1, 4, 1, 1]`・g=2 で並びが `[0, 2, 1, 3]`）が出る。`groups=0`・非整除で `Err`。
- `compat_sequential_channel_shuffle_add_method_has_approved_signature` と `..._detects_offense`: 引数部が `mut self, groups: usize) -> Result<Self, AutodiffError>` と一致する。
- `facade_does_not_reexport_or_declare_channel_shuffle_layer`: facade の src が `ChannelShuffle` 型・`channel_shuffle_forward` を再エクスポートも宣言もしない。自己テストを付ける。

保留 doctest ガード（`ChannelShuffleHoldDoctestGuard`）を作るかは Q4（推奨: 作らない）。

### §3.2 rustdoc と doctest

- `Var::channel_shuffle` と `add_channel_shuffle` の両方に、役割要約（PyTorch 相当名・並べ替えの式・groups の条件）と、CPU の `fandhe_ai::tape()`／`Sequential::predict` だけで動く doctest を付ける（`add_pixel_shuffle` の doctest〈`sequential.rs:2463-2474`〉と同型。GPU は要らない）。
- 失敗系は `compile_fail` を使わず、`Result::is_err` を確かめる通常のテストで検証する（`docs/facade-speculative-decoding-batching-design.md` §18.3 の方針）。

### §3.3 エラー型

- 新しい variant は足さない（`AutodiffError` は `#[non_exhaustive]`）。
- `groups == 0` → `InvalidArgument`。`ChannelShuffle::new` と `channel_shuffle_forward` の入口の両方で検査する。
- rank < 3 → `Shape(ShapeError::RankMismatch)`。`C % groups != 0` → `Shape(ShapeError::ShapeMismatch)`。空間軸の積は `checked_mul` で計算し、溢れたら `ElementCountOverflow`。
- 検査の順は groups → rank → 整除 → tape への積み込み。`Err` のときに孤児ノードを残さない。
- `add_*` は構築時に groups だけを検査し、shape は forward 時に遅延検査する（`PixelShuffle::new` と同じ契約）。
- 本番経路で `unwrap`／`expect` を使わない。`groups == 1` と `groups == C` は有効（結果は恒等）。大きさ 0 の軸は空テンソルとして通す（PyTorch の挙動は未実測）。

### §3.4 テスト計画

| 受け入れ基準 | テスト | 区分 | `<cuda-node>` | `<metal-node>` | `<rocm-node>` |
|---|---|---|---|---|---|
| 手計算の forward・backward、g=1／g=C の恒等、大きさ 0 の軸、rank 3／4／5 | `crates/autodiff/src/nn/channel_shuffle.rs` の単体テスト・`crates/autodiff/tests/nn_channel_shuffle.rs` | CI | - | - | - |
| 拒否（g=0・非整除・rank<3）と孤児ノードが無いこと | 同上 | CI | - | - | - |
| forward と forward_host の bit 一致、中央差分による勾配の突合 | 同上 | CI | - | - | - |
| create_graph の二階有限差分 | `crates/autodiff/tests/create_graph.rs` に節を足す | CI | - | - | - |
| facade 経由で届くことと doctest | `crates/facade/tests/channel_shuffle_facade.rs`・rustdoc | CI | - | - | - |
| `add_*` の forward・`Err`・fit・常駐経路 | `crates/facade/tests/compat_sequential_channel_shuffle.rs` | CI | - | - | - |
| CPU と NaiveOps の bit 一致 | `crates/facade/tests/channel_shuffle_backend_parity.rs::cpu_channel_shuffle_{forward,backward}_bit_exact_vs_naive` | CI | - | - | - |
| CUDA と CPU の bit 一致 | 同ファイル `cuda_channel_shuffle_*` | `#[ignore]` | 未実測（#3860） | - | 該当なし |
| Metal と CPU の bit 一致 | 同ファイル `metal_channel_shuffle_*`（`cfg(target_os = "macos")`） | `#[ignore]` | - | 未実測（#3861） | 該当なし |
| 層経路の parity | `crates/facade/tests/compat_sequential_channel_shuffle_backend_parity.rs::{cpu_reference_runs_without_device, cuda_channel_shuffle_layers_bit_exact, metal_channel_shuffle_layers_bit_exact}` | CI と `#[ignore]` | 未実測（#3860） | 未実測（#3861） | 該当なし |
| model_io の往復（Q5 が (a) の場合） | `compat_sequential_model_io_layers.rs`・`model_io.rs` の単体テスト | CI | - | - | - |
| ONNX export の fail-closed | `OnnxModel::from_sequential` が `UnsupportedLayer` を返す | CI | - | - | - |
| 正ガード 5 件と自己テスト（§3.1） | `crates/facade/tests/api_surface.rs` | CI | - | - | - |

- 実機の手順と記録先は `docs/perf/logs/channel-shuffle-3822/README.md`（#3822 で新設）。cfg で 0 本しか実行されなかった結果、Linux や M4 Max 開発機の代替値は pass の根拠にしない。
- ROCm は facade 非公開（追跡表 # 24 は保留）なので「該当なし」。

### §3.5 数値契約

- 判定は REQ-2 の統一複合判定（相対誤差 1e-3 未満 または 絶対誤差 1e-5 未満）。演算は純粋な並べ替えのコピーなので、テストはそれより強い **bit 完全一致**で書く（`pixel_shuffle_backend_parity.rs` と同じ扱い）。
- `RELATIVE_TOLERANCE`／`ABSOLUTE_RESCUE_THRESHOLD`・parity baseline・ガードレール閾値は変えない。新しい baseline は作らない。
- FMA 契約・f64 長軸縮約の契約への影響: なし（積和も縮約も無い）。

### §3.6 互換への影響

判定（`docs/api-compat-0.14.0-decision.md` §8 の手順）は**追加のみ**。

- `Var`・`compat::Sequential`: inherent メソッドを各 1 件足す。
- `AutodiffError`: variant を足さない。`LayerSpec`: `pub(super)` で公開型ではない。`ChannelShuffle`: autodiff 内部だけ。
- 利用者が `Var` に同名のトレイトメソッドを持つ場合は inherent メソッドが優先して解決される。通常の追加と同じ扱いで、破壊的には数えない。
- `docs/api-compat-0.14.0-decision.md:55`（# 14 行）の記述と一致しており、記録のずれは無い。

### §3.7 適用記録・索引・追跡表

- `docs/compat-api-scope.md` の適用記録は承認前なので本 PR では書かない。#3822 が追記する文案: 「**適用記録（イシュー #3822。承認: <#3768 の回答 URL>）**: `Var::channel_shuffle` と `compat::Sequential::add_channel_shuffle` を追加のみで公開。新規 Op・依存なし。公開状態: 公開済み」。
- 本 PR の更新は `docs/README.md` の索引 1 行と、追跡表 # 14 行の「設計記録」セルだけ。正ガードの `path:line` と 0.14.0 移行ノート（#3780）の該当欄は #3822 が書く。

### §3.8 ベンチと性能

- 既定経路への影響: なし（新しい API を足すだけで、既存の演算・カーネルに触れない）。
- GPU tape では `Op::Contiguous` がホストで eager 実行されるため往復が入る（pixel_shuffle と同じ。`docs/autodiff-pixel-shuffle-decision.md` §2.1）。新しい経路なので前後比較は要らない（未実測）。専用カーネルはスコープ外。

### §3.9 依存

- 依存の追加: なし。許容 12 区分・禁止リストとの照合は該当なし。新規 `unsafe`: なし。

### §3.10 VJP・create_graph・合成方式と保存・export

- 合成（`pixel_shuffle_forward`〈`pixel_shuffle.rs:130`〉と同じ手順）: ① `contiguous` ② `reshape([N, g, C/g, S])`（S は空間軸の積）③ `permute([0, 2, 1, 3])` ④ `contiguous` ⑤ `reshape(in_shape)`。新規 `Op` も独自の VJP も作らない。
- **create_graph は対象にする（追加の作業なし）。** `Reshape`・`Permute`・`Contiguous` は `docs/autodiff-higher-order-grad-decision.md:119` の初期スコープに入っている。CPU の二階有限差分の突合を 1 本入れる。
- model_io（Q5）: pixel_shuffle と同型の kind `channel_shuffle`（`params`: `groups`）を推奨する。最終決定は #3750 の横断方針（タグの追加規則・後方互換・版数）に従う。#3822 が触る箇所は `crates/facade/src/compat/model_io.rs` の `spec_kind`（`:397`）・パラメータ無しの arm（`:801`）・`param_fields`（`:1514`）・`build_model`・`spec_from_kind`・`all_kind_specs`・kind 数のアサート・非信頼入力の表・不正値の表（`groups: 0`）・モジュール doc の種別数と、統合テスト（`compat_sequential_model_io.rs`・`compat_sequential_model_io_layers.rs`）。kind 数は兄弟の実装と合算した数に合わせる。load 時の `groups = 0` は `add_channel_shuffle` の `Err` を経由して型付きエラーで拒否する。
- ONNX export（Q6）: ONNX に ChannelShuffle の単体 op は無く、`from_sequential` の対応層は 6 種で PixelShuffle も export していない。推奨は今回 export しないこと。`layer_kind_name`（`crates/onnx-interop/src/onnx/export_nn.rs:197`）が `"unknown"` を返し、`UnsupportedLayer` で fail-closed のままにする（`as_channel_shuffle` フックは足さない）。`Reshape → Transpose → Reshape` への分解 export は #3750 の論点とする（PyTorch exporter の分解方法は未実測）。
- 既知の差分（未実測）: PyTorch の rank 条件・大きさ 0 の挙動・軸の定義。

### §3.11 所有者へ戻す論点（#3768 の論点 3・4 へ渡す）

確認がないときの扱いは、どの論点も「確認があるまで実装へ進めない（公開面が無い今の状態を保つ fail-closed）」とする。

| 点 | 選択肢 | 推奨 |
|---|---|---|
| Q1 チャネル軸 | (a) dim 1 固定・rank 3 以上（PyTorch の native 実装）／(b) `rank-3` 軸（pixel_shuffle と同じ） | (a) |
| Q2 groups の検証 | (a) 0 だけ構築時に拒否し、整除は forward 時に検査／(b) `add_*` に C も渡して構築時に整除まで検査 | (a)（PyTorch と同じ引数形） |
| Q3 実体の置き場所 | (a) 新モジュール `nn/channel_shuffle.rs`／(b) `pixel_shuffle.rs` に追記 | (a) |
| Q4 未承認経路の固定方法 | (a) 正ガードと否定ガードだけ／(b) `ChannelShuffleHoldDoctestGuard` も作る | (a) |
| Q5 model_io | (a) kind `channel_shuffle`（`groups`）を #3750 の規則で足す／(b) #3750 が決まるまで保存を `UnsupportedModel` で拒否 | (a)（#3750 に従う） |
| Q6 ONNX export | (a) 今回含めない（`UnsupportedLayer`）／(b) Reshape・Transpose・Reshape へ分解して export する | (a)（(b) は #3750 で別に判断） |

### §3.12 セキュリティ観点・スコープ外

- A03: `groups` は保存ファイル経由の値も含めて外部入力として扱い、tape 操作や確保より前に検査する。空間軸の積は `checked_mul` で計算する。
- A08: forward と forward_host で同じ `channel_shuffle_out_shape` を使い、判定の迂回経路を作らない。1 行委譲は正ガードで固定する。
- REQ-12: 内部の層型を公開シグネチャに出さない。秘密情報・ホスト名は書かない。
- スコープ外: 専用 GPU カーネル、層型名の facade 公開（#3022）、ONNX export（#3750）、`Tensor<f32>`／`Tape` 上のメソッド。新しいイシューは起票しない（所有者の承認事項）。
- #3822 は実機 parity が済むまで `Refs` 運用とする。
