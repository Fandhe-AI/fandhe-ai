# BatchNorm3d（C3）と `Var::batch_norm` の rank 5 拡張の設計判断記録

## §0 位置づけ・承認根拠・境界・推奨

- 本書は追跡表 `docs/compat-matrix-ari-tracker.md` の **行 # 11**（C3 BatchNorm3d）の設計記録である（イシュー #3744）。イシュー本文は「項目 12」と書くが、追跡表の通し番号とは 1 ずれている（# 12 は C5。`docs/nn-shrink-bilinear-channel-shuffle-decision.md` §1.0 の # 9 と同じずれ方）。更新対象は # 11 行である。
- 流れ: 設計 #3744 → 承認 #3768 → 実装 #3819（`Var::batch_norm` の rank 5）・#3820（`BatchNorm3d` 層と `add_batch_norm3d`）→ 実機 parity #3860（`<cuda-node>`）・#3861（`<metal-node>`）。保存・export の横断方針は #3750（未記述）。
- **承認根拠: なし。未承認（#3768 待ち）。** 本書は何も承認しない。
- 書式は `docs/decision-record-template.md`（#3716）の「保守・運用チェックリスト（必須 9 節）」に従う（§1〜§9 が対応）。
- **設計のみ**で、コード・公開面・tolerance・baseline・ガードレール閾値・依存・`docs/spec` は変えない。
- 行番号は origin/main の `86b3d279f` を基準にする。実装 PR の前に引き直すこと。実機は `<cuda-node>`・`<metal-node>`・`<rocm-node>` で表す。

### 現状

- `Var::batch_norm`／`batch_norm_with_batch_stats`／`batch_norm_infer` は rank 2〜4 だけを受理し、rank 5（`[N, C, D, H, W]`）は対象外と明記されている（`crates/autodiff/src/nn/batch_norm.rs:45-48`、レイアウト関数 `crates/tensor-core/src/ops_shape.rs:797` 以降）。
- このため `BatchNorm3d`（PyTorch `nn.BatchNorm3d`）と `Sequential::add_batch_norm3d` は無く、棚卸し `docs/compat-unlisted-holds-inventory.md:73` は「未公開」と分類している。

### 調査で確定した事実（設計の核）

1. **rank を見る分岐はレイアウト関数 1 か所だけ**。`batch_norm_layout`（`ops_shape.rs:797-826`）以外の消費側は、この関数が返す `(n, c, spatial)` だけを受け取り rank を見ない。
   - `Var::batch_norm_with_batch_stats`／`batch_norm_infer`（`crates/autodiff/src/var.rs:1902`・`:1986`）
   - VJP `Op::BatchNorm`（`crates/autodiff/src/grad.rs:1963`）
   - create_graph（`create_graph_norm.rs:980`・`:1038`、`create_graph.rs:1621`）
   - `batch_norm_forward_host`（`crates/autodiff/src/nn/module.rs:3026`）
   - 4 バックエンドの `batch_norm_train`／`batch_norm_infer`（CPU `backend-cpu/src/ops.rs:2954`・`:2996`、CUDA `backend-cuda/src/ops.rs:7318`・`:7366`、Metal `backend-metal/src/ops.rs:5688`・`:5734`、ROCm `backend-rocm/src/ops.rs:1129`・`:1162`）
   - したがって rank 5 対応は `batch_norm_layout` に `5 =>` の腕を足すだけで済む。中身は rank 4 の腕と同型（どれかの軸が 0 なら `spatial = 0`、それ以外は `checked_numel(&shape[2..])` で `D*H*W`）。
2. **縮約軸**: チャネル軸は dim 1 のまま（NCDHW 固定）。各チャネルで `M = n*D*H*W` 要素を縮約する。添字写像 `batch*(c*spatial) + ch*spatial + sp` と縮約順序（32 レーン + butterfly。`docs/batch-norm-ops-design.md` §3.1）は変えない。
3. **bit 等価性**: 同一バックエンド上で、rank 5 入力 `x` の結果は `x.reshape([N, C, D*H*W])`（rank 3）の結果と bit 単位で一致する。カーネルへ渡る `(n, c, spatial)` と dense データが同一だからである。これを CI と実機テストの柱にする。
4. **GPU カーネルの新規実装は不要**。CUDA の `i32` 上限超過（`crates/backend-cuda/src/batch_norm.rs:127-131`）と Metal の `u32` 上限超過（`BatchNormSizeLimitExceeded`）は既存写像で `BackendError::Unsupported` となりホスト実装へフォールバックする。rank 5 で `spatial` が大きくなっても新しい上限ロジックは要らない。
5. **層の rank 制限は別の層で保たれる**。`BatchNormVars::forward` は `accepted_ranks` を検査し（`nn/batch_norm.rs:518-522`）、`BatchNorm1d`／`BatchNorm2d` の `forward_host` も `BATCH_NORM_{1D,2D}_RANKS` を先に検査する。レイアウトを rank 5 へ広げても `BatchNorm2d` が rank 5 を受理することはない。ただし回帰テストで固定する（§4）。
6. **ONNX**: import 側（`crates/onnx-interop/src/ops/batch_norm.rs:21-24`）は `batch_norm_layout` を使わず rank 2 以上を自前で処理済みで変更不要。facade の export は BatchNorm を扱わない（`UnsupportedLayer`）。1d／2d と同じ扱いで該当なし。
7. **facade 側に BatchNorm を守るガードは無い**。`crates/facade/tests/api_surface.rs` と `crates/facade/src/lib.rs` で `BatchNorm`／`batch_norm` は 0 件。反転する否定ガードも保留 doctest ガードも無い。
8. **model_io**: kind は文字列 allowlist で照合し（`"batch_norm1d"`／`"batch_norm2d"`。`crates/facade/src/compat/model_io.rs:369-370`・`:3720`）、未知 kind は `UnsupportedModel` で拒否する。`FORMAT_VERSION = 2`（`:122`）は BN の running stats を `buffer_keys` に書く版である。

### 推奨

**推奨: 実装する（案 A）。** 理由は 4 点。

1. 変更点はレイアウト関数 1 か所で、消費側はすべて rank 非依存（上記 1）。
2. 新規 `Op`・`BackendOps`・カーネル・`unsafe`・依存が要らない。
3. 既存の数値契約（f64 アキュムレータ・縮約順序・Metal soft-f64）をそのまま継承する。
4. 互換区分は「挙動拡張＋追加のみ」で、`docs/api-compat-0.14.0-decision.md` §1.2 の # 11 行と一致する。

| 案 | 内容 | 利点 | 欠点・影響 |
|----|------|------|-----------|
| A（推奨） | `batch_norm_layout` を rank 2〜5 にし、`BatchNorm3d`・`add_batch_norm3d` を足す。rank 6 以上は拒否のまま | 変更最小。fail-closed を保つ | rank 6 以上の `RankMismatch.expected` が 4 から 5 に変わる |
| B | rank 2 以上の任意 rank を受理（PyTorch の `F.batch_norm` 相当） | PyTorch に近い | 公開面の受理範囲が大きく広がり fail-closed が弱まる |
| C | 対象外にする | 変更なし | C3 が「未公開」のまま残り、0.14.0 の目標を満たさない |

**訂正注記（実装者向け）**: イシューおよび #3819 の本文は Metal の数値方式を「scale/ssq」と書くが、実装は **soft-f64**（IEEE 754 binary64 の整数エミュレーション）で、scale/ssq は不採用と記録済みである（`crates/backend-metal/src/shaders/batch_norm.metal` 冒頭、`docs/batch-norm-ops-design.md` §9.1）。#3819／#3861 の実装・測定は実装どおりの契約に従い、カーネルを scale/ssq へ「直さない」こと。

## §1 公開形とガード反転計画

**シグネチャ案**

| 対象 | シグネチャ |
|---|---|
| `Var::batch_norm`／`batch_norm_with_batch_stats`／`batch_norm_infer` | シグネチャ不変。受理 rank が 2〜5 になる |
| `compat::Sequential::add_batch_norm3d` | `fn add_batch_norm3d(mut self, num_features: usize, eps: f32, momentum: f32) -> Result<Self, AutodiffError>`（`add_batch_norm2d`〈`crates/facade/src/compat/sequential.rs:1689`〉と同型） |
| `add_batch_norm3d_restored`（`pub(super)`） | 保存済みモデルの復元用。2d 版（`sequential.rs:1738` 付近）と同型 |

- facade の `Var` は再エクスポートなので配置の追加は要らない。`add_batch_norm3d` は `fandhe_ai::compat::Sequential` の inherent メソッド。
- 内部（autodiff 側。facade からは再エクスポートしない）:
  - `BATCH_NORM_3D_RANKS: &[usize] = &[5]`
  - `nn::BatchNorm3d`（`BatchNorm2d` と同型。`new`／`without_affine`／`from_parameters`／`bind`／アクセサ）と `impl Module for BatchNorm3d`
  - `Module::as_batch_norm3d`／`as_batch_norm3d_mut`（既定 `None`）
  - `nn/mod.rs:292` の再エクスポートへの追加
- `LayerSpec::BatchNorm3d { num_features, eps, momentum }`（`pub(super)`、非公開）。
- 層型名 `BatchNorm3d` は facade へ再エクスポートしない（1d／2d も出していない。層型名の公開は #3022 の個別承認 3 条件に従う別判断）。

**ガードの扱い**

- 反転する否定ガード・保留 doctest ガードは**無い**（上記 7）。
- #3820 が新設する正ガード（`api_surface.rs:14424-14528` の既存命名にならう）:
  - `ADD_BATCH_NORM3D_PARAMS` 定数
  - `compat_sequential_batch_norm3d_add_method_has_approved_signature` と自己テスト `..._detects_offense`（引数順・型・戻り型のすり替えを検出）
  - `compat_sequential_exposes_batch_norm3d_add_method_issue_3820`（`src/compat` 配下の `pub fn add_batch_norm3d` がちょうど 1 件）と自己テスト `..._counts_declarations`
- #3819 は公開名を増やさないので、新しい api_surface ガードは不要。
- 否定プローブを残す範囲: 層型 `BatchNorm3d` が facade から再エクスポートされていないこと。既存の層型非公開の固定方式に合わせ、#3820 で追加の要否を確かめる。

## §2 rustdoc と doctest

- `Var::batch_norm` 系の doc を「BatchNorm1d／2d／3d・rank 2〜5」へ直す（#3819）。
- `add_batch_norm3d` には役割要約・受理 rank が 5 だけであること・モードの注意（1d の doc と同じ）を書き、facade 単独で通る doctest を置く。CPU の `Sequential::new().add_batch_norm3d(2, 1e-5, 0.1)?` で `[2, 2, 2, 2, 2]` を `predict` し shape が保たれることを確かめる。GPU 不使用なので `#[ignore]`／`no_run` は不要。
- 失敗系は `compile_fail` に頼らず、正のプローブ（rank 4 を渡すと `Err`）とインベントリ（§1 の件数ガード）で組む。
- 古くなる記述（実装 PR で直す。本 PR では列挙だけ）: `crates/autodiff/src/nn/batch_norm.rs` の冒頭 doc と `:45-48`、`crates/tensor-core/src/ops_shape.rs` の `batch_norm_layout` の doc、`crates/autodiff/tests/nn_module_mode.rs:314`、`crates/onnx-interop/src/ops/batch_norm.rs:23`、`docs/batch-norm-ops-design.md:32,231`（既存本文は書き換えず追記で訂正する）。

## §3 エラー型

新しい variant は無い。既存の型付きエラーを使う。

| 入力 | エラー |
|---|---|
| rank 0・1 | `ShapeError::RankMismatch { expected: 2 }`（不変） |
| rank 6 以上 | `RankMismatch { expected: 5 }`（現在 rank 5 以上は `expected: 4`。値だけが変わる） |
| `spatial` の積の overflow | `ElementCountOverflow` |
| train の `M <= 1` | `AutodiffError::InvalidArgument` |
| `BatchNorm3d` に rank 5 以外 | `RankMismatch { expected: 5 }`（`BatchNormVars::forward` と `forward_host` の両方） |
| `eps`／`momentum`／`num_features == 0` | 既存の `validate_*` |

- 検証順: shape 検査と `checked_numel` を確保（`Tensor::full` 等）より前に行う。どれかの軸が 0 なら積を取らない（`[0, 1, usize::MAX, 2, 2]` のような空テンソルを overflow と誤判定しないため。rank 4 の腕と同じ理由）。本番経路で `unwrap`／`expect` を使わない。
- model_io の非信頼 manifest（A03）: `batch_norm3d` の `params` は固定スキーマ（キー集合の完全一致）で読む。v1（`LEGACY_FORMAT_VERSION`）の manifest に `batch_norm3d` が現れたら拒否する（v1 の時点でこの kind は存在しないため）。fail-closed を推奨とし §11 に入れる。

## §4 テスト計画

実機列はすべて「未実測」。cfg により 0 本しか走らなかった結果を pass の根拠にしない。

| 受け入れ基準 | テスト | 区分 | `<cuda-node>` | `<metal-node>` | `<rocm-node>` |
|---|---|---|---|---|---|
| レイアウト関数が rank 5 を受理 | `ops_shape.rs` の `rank5_is_rejected`（`:865`）を反転し `rank5_spatial_is_dhw`（`[2,3,4,5,6]` → `(2,3,120)`）へ。`rank6_is_rejected`（`expected: 5`）・`rank5_zero_axis_yields_spatial_zero`・`rank5_spatial_overflow_is_element_count_overflow` を追加（#3819） | CI | 該当なし | 該当なし | 該当なし |
| rank 5 と rank 3 reshape の bit 一致 | autodiff の `Var::batch_norm`（train／infer）で出力・`batch_mean`／`batch_var`・1 階勾配（dx／dw／db）を比較。naive f64 参照との REQ-2 突合も行う（#3819） | CI | 該当なし | 該当なし | 該当なし |
| `BatchNorm2d` が rank 5 を拒否 | tape 経路と `forward_host` の両方（#3819） | CI | 該当なし | 該当なし | 該当なし |
| create_graph が rank 5 で成立 | `create_graph_norm.rs::channel_rows_round_trip_for_rank2_rank3_rank4` に rank 5 を追加。`create_graph.rs`・`create_graph_batch_norm.rs` の掃引に rank 5 を追加し rank 3 reshape 版と一致を確認（#3819） | CI | 該当なし | 該当なし | 該当なし |
| CPU の bit 一致 | `backend-cpu/tests/batch_norm_parity.rs` に rank 5 版を追加（#3819） | CI | 該当なし | 該当なし | 該当なし |
| `BatchNorm3d` の train／eval・running stats | unbiased `M/(M-1)`（`M = n*D*H*W`）の更新、`compat_sequential_layers.rs` に `add_batch_norm3d`（#3820） | CI | 該当なし | 該当なし | 該当なし |
| 保存・復元 | model_io の往復（running stats を bit で保つ・`training` フラグ復元・`batch_norm3d` kind の非信頼入力〈キー欠落・未知キー・型違い・v1 での出現〉）、safetensors 経路（`crates/facade/src/interop/safetensors.rs:84` の層数記述の更新を含む）（#3820） | CI | 該当なし | 該当なし | 該当なし |
| 公開形のガード | §1 の正ガードと自己テスト（#3820） | CI | 該当なし | 該当なし | 該当なし |
| facade の doctest | §2 の `add_batch_norm3d` doctest | CI | 該当なし | 該当なし | 該当なし |
| GPU と CPU の REQ-2 一致 | `crates/facade/tests/compat_sequential_layers_backend_parity.rs::cuda_batch_norm3d_matches_cpu`／`metal_batch_norm3d_matches_cpu`（`run_batch_norm2d_parity` と同型） | `#[ignore]` | 未実測 | 未実測 | 該当なし |
| バックエンド単体の rank 5 | `backend-cuda/tests/batch_norm_parity.rs`・`backend-metal/tests/batch_norm_parity.rs` に新設（例: `batch_norm_rank5_train_matches_naive`、`batch_norm_rank5_matches_rank3_reshape_bit_identical`）。既存の `*_across_shapes*` に混ぜて既存の実測記録の意味を変えない | `#[ignore]` | 未実測 | 未実測 | 該当なし |
| facade の GPU 経路 | `crates/facade/tests/batch_norm_backend_parity.rs` に rank 5 版 | `#[ignore]` | 未実測 | 未実測 | 該当なし |

- `<rocm-node>`: ROCm は facade で非公開のため該当なし（任意で `backend-rocm` 単体の rank 5 確認）。
- create_graph の実機 parity は §48（#3739）の測り方に従う。

## §5 数値契約

- REQ-2 の統一複合判定（相対誤差 1e-3 未満 または 絶対誤差 1e-5 未満）を適用する。`RELATIVE_TOLERANCE`／`ABSOLUTE_RESCUE_THRESHOLD`・parity baseline・ガードレール閾値は変えない。BatchNorm は単一チャネル縮約の統一判定で baseline 方式の対象外であり、新しい baseline は無い。
- 正規化統計は f64 アキュムレータ（CPU・CUDA は `f64`／`double`、Metal は **soft-f64**。scale/ssq は不採用。`docs/batch-norm-ops-design.md` §9.1）。縮約順序は §3.1 の butterfly、FMA 契約は affine の `mul_add`。いずれも不変。dw／db の長軸縮約の規律（要素積を f32 で確定してから f64 へ昇格）も不変。
- GPU と CPU の判定契約は 1d／2d と同じで、CUDA と Metal は REQ-2 で判定し bit 一致は主張しない。同一バックエンド内での rank 5 と rank 3 reshape の bit 一致は追加で主張する（上記 3）。
- baseline の追加・更新: なし。

## §6 互換への影響

判定手順は `docs/api-compat-0.14.0-decision.md` §8。**区分: 追加のみ**（`!`／`BREAKING CHANGE:` は不要）。

| 対象 | 判定 |
|---|---|
| `Var::batch_norm` 系 | シグネチャ不変。rank 5 が拒否から受理に変わる挙動拡張。rank 6 以上の `RankMismatch.expected` は 4 から 5 に変わるが型は不変で値だけが変わる |
| `Sequential::add_batch_norm3d` | inherent メソッドの追加のみ |
| `LayerSpec` | `pub(super)` のため該当なし |
| autodiff の `Module` trait | 内部クレートで、既定実装つきメソッドの追加のため該当なし |
| `ModelIoError` | variant の追加なし。該当なし |
| model_io の保存形式 | kind の追加のみで `FORMAT_VERSION` は据え置き。旧版の読み手は新ファイルを `UnsupportedModel` で拒否（fail-closed）。古いファイルは引き続き読める |

## §7 適用記録・索引・追跡表

- 適用記録の案（実装 PR #3820 が `docs/compat-api-scope.md` へ書く）: `**適用記録（イシュー #3820・親 …。承認: <URL>）**: <公開形・公開状態>`。現時点の公開状態は「未公開」。
- [ ] `docs/compat-api-scope.md` に適用記録を追記した（実装 PR #3820 で行う）
- [x] `docs/README.md` の索引を更新した（本 PR）
- [x] 追跡表 `docs/compat-matrix-ari-tracker.md` の # 11 行の設計記録列を更新した（本 PR。正ガード列は「未特定」のまま。予定名は §1）
- 0.14.0 移行ノート（#3780）への反映は実装後に行う。

## §8 ベンチと性能

- 既定経路（rank 2〜4）への影響は無い。`match` の腕が 1 本増えるだけでカーネルは変わらないため、前後比較は不要。
- 注意: CUDA／Metal の train カーネルは 1 チャネルを 1 warp（simdgroup）で処理する構成のため、3D 入力で `M` が大きいと遅くなりうる。これは新経路の性能の問題で既定経路の後退ではない。最適化はスコープ外（§12）。実測は未実測。

## §9 依存

- 依存追加: なし。許容 12 区分・禁止リストとの照合は追加が無いため不要。
- 新規 `unsafe`: なし。`docs/license-matrix.md` の更新も不要。

## §10 VJP・create_graph（§28）への影響

- VJP（`batch_norm_vjp_channels`）は `(n, c, spatial)` だけを受け取るので変更不要。
- §28 の S1 構造は `[c, m]` のチャネル行で shape に依存しない。入口検査（`validate_norm_s1`）・空入力の拒否（`shape.contains(&0)`）・`checked_mul` も rank に依存しない。影響はテストの拡張だけで、拒否範囲・閾値・方式の割り当て（V3）は不変。
- §45（#3736。S0 前スケーリング案。#3767 待ち）が承認された場合、rank 5 もレイアウト経由で同じ扱いを受ける。両者は独立に進められる。
- running stats が二重更新されないこと（§28.4）は rank 5 でも統合テストで固定する。

## §11 所有者へ戻す論点（#3768 へ渡す）

| 点 | 選択肢 | 推奨 | 確認がないときの扱い |
|----|--------|------|----------------------|
| 1. 受理する rank の範囲 | 2〜5／任意の rank 2 以上 | 2〜5（fail-closed） | 確認があるまで実装へ進めない |
| 2. rank 6 以上の `RankMismatch.expected` を 4 から 5 へ変えてよいか | 変える／変えない | 変える（最も近い許容 rank という既存規則どおり） | 確認があるまで実装へ進めない |
| 3. model_io の kind `"batch_norm3d"`・buffer キーは 1d／2d と同じ・版は据え置き | 据え置き／版を上げる | 据え置き。#3750 の横断規則が後から決まればそれに従う | 確認があるまで実装へ進めない |
| 4. v1 manifest に現れた `batch_norm3d` | 拒否／受理 | 拒否 | 確認があるまで実装へ進めない |
| 5. 層型 `BatchNorm3d` を facade へ再エクスポートするか | する／しない | しない（#3022 の別判断） | 確認があるまで実装へ進めない |
| 6. 実機 parity テストの置き方 | 新設／既存の形状網羅へ混ぜる | 新設 | 確認があるまで実装へ進めない |

## §12 セキュリティ観点・スコープ外

- **A03**: 形状・`eps`・`momentum` を確保より前に検証する。model_io の非信頼 manifest は固定スキーマで読み、未知 kind は拒否する。積の overflow は型付きエラーにする（panic にしない）。
- **A04／A08**: ホストフォールバックに入るのは `Unsupported` のときだけ。判定を迂回する経路を作らない。
- 秘密情報・ホスト名・個人パスは書かない。
- スコープ外（無断で起票せず、所有者の承認を得てから起票する）: channels-last、`momentum=None`（累積移動平均）、`track_running_stats=false`、3D 入力向け GPU カーネル最適化（`M` が大きい場合の並列化）、ONNX export への BatchNorm 追加（#3750 の範囲）。
- `Closes`／`Refs`: 本 PR は設計記録のみで実測を伴わない。運用は兄弟 PR #3904／#3905 に合わせる。#3819／#3820 は `Refs` のみ。
