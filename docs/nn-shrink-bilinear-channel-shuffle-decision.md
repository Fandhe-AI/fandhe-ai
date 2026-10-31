# Hardshrink・Softshrink・Bilinear・ChannelShuffle の公開形の設計判断記録

## §0 位置づけ・共通事項

- 本書は追跡表 `docs/compat-matrix-ari-tracker.md` の C1・C2・C7 の設計記録をまとめる 1 本である。節の分担は次のとおり。
  - **§1 C1 Hardshrink・Softshrink**: イシュー #3742（本 PR が記述）
  - §2 C2 Bilinear: イシュー #3743 が追記する
  - §3 C7 ChannelShuffle: イシュー #3746 が追記する
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

## §2 C2 Bilinear（イシュー #3743 が追記）

## §3 C7 ChannelShuffle（イシュー #3746 が追記）
