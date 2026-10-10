# compat API 層の対象範囲（TASK-9.2b）

イシュー #96（親: #94 = TASK-9.2、ルート: Phase 4 #71）の成果物。
REQ-9「Python 慣習寄りの互換 API 層」（`docs/spec/04-requirements.md:222-236`）の
受け入れ基準のうち「互換 API 層の対象範囲は本フェーズでは初期方針（活性化関数・
基本レイヤーの薄いラッパー）に限定し、全方位の API 網羅は対象外とすること」
（改定前基準。`04-requirements.md:227`。履歴として保持）を明文化する。TASK-9.2
の成果物として `docs/compat-api-scope.md` が明示されている
（`docs/spec/05-tasks.md:307-311`）。

**2026-09-12 改定（実装リポ ルート #1570・spec イシュー #66・spec PR #69
〈`c5cf1ed`〉）**: 上記の改定前基準（受け入れ基準 3「全方位の API 網羅は対象外」）
は、対象範囲を PyTorch／TensorFlow（Keras）の機能網羅へ Tier 1（MLP → CNN →
Transformer が組める最小集合）・Tier 2（長尾）の 2 段階で拡張する基準へ
置き換えられた（`04-requirements.md:228-235`）。本書 §1／§2／§5 は
イシュー #1591 でこの改定へ追従する。改定前の初期方針（配列生成関数・
Sequential ビルダー・基本レイヤー／活性化の薄いラッパー）は 1.1 節「実装済み
公開面」として引き続き記録する。

`docs/public-api-design.md` は「compat 層は自作コアの素の公開 API とは別
レイヤーであること」という境界のみを 1 章で明記し（同ファイル 6・13・556 行目）、
詳細範囲は REQ-9 系後続タスクへ委譲している。本文書はその受け皿である。

## 0. サポート境界（TASK-9.4・REQ-9 の 2026-08-08 追記・イシュー #411・
REQ-9 の 2026-08-29 追記・イシュー #986）

`facade` クレートが**唯一のサポートされる公開 API 面**である。`tensor-core`／
`autodiff`／`backend-cpu`／`backend-cuda`／`backend-metal` は内部クレートで
あり、これらを `facade` を経由せず直接利用することはサポート対象外とする
（出典: `docs/spec/04-requirements.md:237-238` の 2026-08-08 追記・
`docs/spec/04-requirements.md:239-241` の 2026-08-29 追記・
`docs/spec/05-tasks.md:322` TASK-9.4）。

- **本文書が定める対象範囲（1〜2 節）は `fandhe_ai::compat` として提供される
  公開面を指す**（4.2 節参照。旧 `fandhe_ai_autodiff::compat` は TASK-9.4 で
  `fandhe_ai::compat` へ移設済み。**移行期間中は `fandhe_ai_autodiff::compat` に
  非推奨シム〈`#[deprecated]`〉を残し、既存コードのソース互換性を保つ**。
  4.3 節参照。codex-review PR #424 P1 是正）
- `autodiff` の `Tape::new_with_ops`／`nn::Module` 実装等、compat 層が内部で
  依拠する API は Rust の可視性としては `pub`（`autodiff` クレートの
  ドキュメント上は到達可能）だが、**サポート境界上は内部 API** であり、
  REQ-12「利用者向け融合制御 API」・REQ-9「互換 API 層」のいずれにも
  該当しない。技術的に `pub` であることと、利用者向けにサポートされる
  公開面であることは区別する
- **確定入口は次の 5 つ**である（正本 spec 側の記述〈REQ-9 の 2026-08-08
  追記・2026-08-29 追記〉と整合済み）:
  1. `fandhe_ai::tape()`／`fandhe_ai::tape_for(Device)`（composition root。
     `crates/facade/src/lib.rs`）
  2. `fandhe_ai::compat::{array, Sequential}`（本文書が定める compat 公開面。
     1〜2 節。1 節の対象範囲は本追記で変更しない）
  3. **`fandhe_ai::optim`**（`crates/facade/src/optim.rs`。イシュー #961・
     親 #960・PR #972）: `Sgd`／`SgdConfig`／`AdamW`／`AdamWConfig`／
     `clip_grad_norm`／`global_grad_norm`／`ClipGradResult`／`LrScheduler`／
     `ConstantLr`／`StepLr`／`CosineAnnealingLr`／`ExponentialLr`／
     `LinearWarmupLr`〈#1745〉／`OneCycleLr`／`OneCycleLrConfig`／
     `OneCycleAnneal`〈#1747〉の素の再エクスポート（`docs/facade-optimizer-
     promotion-decision.md` §4 案 A）。値型・純関数のみで構成され
     `BackendOps` 系の型・注入経路を含まないため REQ-12（利用者向け融合
     制御 API を設けない・任意 `BackendOps` 実装を注入できる公開 API を
     設けない）と矛盾しない（出典: `docs/spec/04-requirements.md:240`。
     `clip_grad_value`〈#1753・親 #1631〉が実装済みで追加された
     identifier だが、同 spec 出典自体はまだ追従しておらず古いまま
     （`docs/spec/` は編集しない。§1.3 の #1631 行に実装記録を追記済み）
  4. **デバイス常駐更新経路 `fandhe_ai::DeviceParamStore`／
     `Tape::step_device_param_store`**（`Tape::sync_device_param_store_to_host`・
     `Tape::backward_device_param_store`〈#1022 で追加。`Op::LinearResident`
     を含むグラフの backward 入口。素の `Tape::backward` はこのグラフに
     対し型付きエラーを返すため必須〉を含む。#954。クレート root からの
     再エクスポート。`crates/facade/src/lib.rs`）: 学習ループのパラメータ
     （および momentum 等の optimizer 状態）をデバイス上に常駐させ、SGD
     更新をデバイス上で完結させることでステップごとのホスト⇔デバイス
     往復を削減する経路。#1022 で forward 用のパラメータ download を
     排除した新経路 `register_resident_params`／`snapshot_resident_params`
     を追加し、`DeviceParamStore::linear_forward`（`BackendOps::
     gemm_resident_rhs` 経由）がデバイス常駐のまま forward する（旧
     `register_resident_leaves`／`snapshot_resident_leaves`〈download を
     伴う・`Vec<Var<'t>>` を返す〉は crates.io 0.4.0 公開済み API との
     SemVer 互換のため `#[deprecated]` として維持する。codex-review PR
     #1059 P1 是正）。
     `facade::Tape` newtype からの薄い委譲として提供し、`BackendOps`／
     `MemoryOps` は利用者向け公開面へ露出しない。数値一致は REQ-2 の
     バックエンド間統一複合判定（相対誤差 1e-3 未満 または絶対誤差
     1e-5 未満）・FMA 契約に従う（出典: `docs/spec/
     04-requirements.md:241`・設計 `docs/device-resident-update-design.md`
     〈#951・#1022 追補〉・#955〈parity テスト・ベンチ非後退確認〉）。
     イシュー #1479（`docs/device-resident-update-design.md` 追補：
     #1479）で `Tape::resident_grads_to_host`／`Tape::
     param_grads_to_host` を同じ確定入口へ追加した: resident 経由
     （`GradStaging`）で新鮮に充填済みの重み勾配を、`step()` を呼ばずに
     読み出し専用でホストへ取得する API。CUDA／Metal（resident 未対応。
     `gemm_fp32_strict_into` 未実装）では `resident_grads_to_host` が
     `BackendError::Unsupported` を返す一方、`param_grads_to_host`
     （unified 版。未充填 slot は `grads.get(...)` へフォールバック）は
     3 バックエンド共通で `Ok` を返す
  5. **デバイスメモリプール解放 API `fandhe_ai::release_cached_memory(Device)`／
     `fandhe_ai::memory_pool_stats(Device)`**（イシュー #1020・REQ-14 14-3。
     クレート root からの再エクスポート。`crates/facade/src/lib.rs`）:
     `resolve_ops(device)?.release_cached_device_memory()` /
     `.device_memory_pool_stats()`（`BackendOps` の非破壊拡張〈デフォルト
     メソッド追加〉）への薄い委譲。値は unit／`Option<PoolStats>`（POD。
     `fandhe_ai::PoolStats` として root 再エクスポート）のみで、
     `DeviceAllocator`／`BufferHandle`／`SizeClassPool` 等のプール実装型は
     一切露出しない（`crates/facade/tests/api_surface.rs::
     facade_does_not_expose_pool_implementation_types` が機械的に固定）。
     `BackendOps`／`MemoryOps` は他の確定入口と同じく利用者向け公開面へ
     露出しない。設計・採用判断は `docs/device-memory-pool-design.md`・
     `docs/backend-cuda-pool-allocator-decision.md` を参照

  5. **デバイスメモリプールの明示解放・統計 API `fandhe_ai::
     release_cached_memory(Device)`／`fandhe_ai::memory_pool_stats(Device)`**
     （`crates/facade/src/lib.rs`。イシュー #1018 ツリー・#1019 設計
     〈`docs/device-memory-pool-design.md`〉・#1021 Metal 実装）: REQ-14
     の明示解放 API。`device` に対応するバックエンドのデバイスメモリ
     プール（サイズクラス別フリーリスト。`fandhe_ai_tensor_core::
     backend_ops::BackendOps::release_cached_device_memory`／
     `device_memory_pool_stats` を薄く再公開）がアイドル保持している
     バッファを全て解放する、または統計スナップショット（POD
     `fandhe_ai::PoolStats`。内部ハンドル表現を一切含まない）を返す。
     `Device` は識別子 enum（外部型）であり facade は inherent メソッドを
     追加できない（orphan rule。`docs/facade-device-handle-design.md`
     「案 B のみ採用」と同じ理由）ため、`tape`／`tape_for` と同型の
     自由関数として提供する。プールを持たないバックエンド（CPU 等）は
     解放を no-op（`Ok(())`）・統計を `None` とする。

  `SgdConfig` はクレート root（4 の経路）と `crate::optim::SgdConfig`
  （3 の経路）の 2 経路から再エクスポートされる同一型〈`lib.rs`
  コメント〉であり、`DeviceParamStore`／`Tape::step_device_param_store` は
  デバイス常駐更新という別経路のため `optim` モジュールには含めず root
  公開のままである（`optim.rs` モジュール doc「デバイス常駐更新との
  違い」と整合）。

  **確定に至る経緯（履歴）**: 3・4 は当初、正本 spec 側の改定が未了のため
  「移行予定の入口」として区別していた（イシュー #962。codex-review
  PR #974 P1 是正）。5 節手続きのうち経路 2（本リポジトリのユーザー承認を
  得たうえでの Issue 起票・本文書の更新。親 #960 ツリー・#961〜#963）で
  正本 spec 改定の要否「要」を確定させたうえで、経路 1（正本 spec
  リポジトリ側での REQ-9 受け入れ基準の改定）を実施した（実装リポ
  イシュー #984・spec リポ `Fandhe-AI/fandhe-ai-spec` PR #59・
  2026-08-29 マージ）。submodule ポインタ更新（実装リポ #985・PR #988）
  完了を受け、本イシュー #986 で確定入口の列挙へ統合した。
- サポート境界の変更（内部クレートの直接利用をサポート対象に含める等）は
  正本 spec リポジトリ側での REQ-9／REQ-12 受け入れ基準の改定を要する
  （5 節「範囲拡張の手続き」と同じ手続き。本節の `optim`・デバイス常駐
  更新経路の追加は同手続き〈経路 1〉を経て確定した適用例である）
- **内部クレートの `pub` enum への `#[non_exhaustive]` 付与は本節の適用例**
  （codex-review PR #648 P1 是正）: `fandhe_ai_tensor_core::fusion::FusedOpKind`
  （`crates/tensor-core/src/fusion/plan.rs`）は `facade` から再エクスポート
  されず（`crates/facade/src/lib.rs` の `pub use fandhe_ai_tensor_core::{..}` に
  `FusedOpKind` は含まれない）、`tensor-core` 自体も `publish = false`
  （ワークスペース `Cargo.toml`）のため、本節が定める意味での「サポート
  される公開面の利用者」は存在しない。既に安定した公開 enum への
  `#[non_exhaustive]` 遡及付与は一般に破壊的変更たりうる（外部の
  非ワイルドカード exhaustive match を壊すため）が、本 enum の唯一の
  参照元はワークスペース内クレート（`backend-cpu`・`autodiff`）に限られ、
  いずれも `_` 分岐を持つ形で参照を更新済み（`backend-cpu::
  fused_elementwise::eval_one` 等）であるため、この一般論はここには
  適用されない

## 1. 対象範囲（in scope）

REQ-9 の受け入れ基準（`docs/spec/04-requirements.md:222-236`）・
TASK-9.1／TASK-9.2（`docs/spec/05-tasks.md:299-311`）に基づき、以下の
3 層で定義する。1.1 節は改定前の初期方針に基づく実装済み集合、
1.2／1.3 節は 2026-09-12 改定（イシュー #1591）で追加された対象範囲で
あり、未実装のまま Phase 2／Phase 3 の各 issue へ実装を委ねる。「対象
範囲」への列挙は実装状況を意味しない（実装状況の正は
`docs/compat-feature-gap.md` と各 issue とする）。

### 1.1 実装済み公開面（改定前の初期方針の範囲）

- **`compat::array()`**: numpy 慣習のテンソル生成関数（`np.array()` 相当）。
  自作テンソル（`tensor-core`）の上に構築する（TASK-9.2a・#95）
- **`compat::Sequential`**: Keras 慣習のレイヤー積み上げビルダー
  （`.add_linear()`／`.add_relu()` 等のメソッドチェーン）。自作 NN モジュール
  （`fandhe_ai_autodiff::nn`）の上に構築する（TASK-9.2a・#95）
- **基本レイヤー・基本活性化関数の薄いラッパー**（TASK-9.1）:
  - Linear 層（TASK-9.1a・#91）
  - ReLU・Sigmoid・Tanh の 3 活性化関数（TASK-9.1b・#92 で実装済み。
    `crates/autodiff/src/nn/activation.rs`）。イシュー #1714 で
    Silu／Hardswish／LeakyRelu／Elu を追加（1.2 節該当行参照）
- **`Var::host_view`／`Tensor::host_slice`（借用ビュー読み出し API）**:
  5 節手続き 2（ユーザー承認済みイシュー #1335 起票）により対象範囲へ
  追加。compat 層固有のラッパーではなく `fandhe_ai_autodiff`／
  `fandhe_ai_tensor_core` の値型に直接生えた読み出し専用 API（facade
  では `VarHostView` の 1 文 1 行 `pub use` として再エクスポート）だが、
  0 節が定める「`facade` が唯一のサポートされる公開 API 面」の一部と
  して本文書に記録する。詳細（寿命契約・同期契約）は
  `docs/public-api-design.md` §2.2／§3.1／§4.2 を正とする。
- **`Sequential` の学習パラメータ取得 API**（#294）:
  `Sequential::bind(&tape)` が返す `SequentialVars`（`crates/facade/src/
  compat/sequential.rs`。TASK-9.4・#411 で `fandhe_ai_autodiff::compat` から移設。
  4.2 節参照）経由で学習可能パラメータ（`Linear` 層の `weight`/`bias`）・
  勾配へアクセスできる。`fandhe_ai_autodiff::optim::Sgd`・
  `fandhe_ai_autodiff::nn::optim::AdamW` を直接呼び出すことは 0 節の
  定義どおり `pub` であってもサポート境界上は `facade` を経由しない
  内部 API であり、確定入口として案内しない（内部 API 直接利用を確定
  入口として扱わない。codex-review PR #974 P1 是正）。**`fandhe_ai::optim`**
  （0 節参照）は 0 節の 2026-08-29 追記で確定入口となったため、
  `Sequential` との接続手順の正は `crates/facade/src/compat/sequential.rs`
  のモジュール doc（doctest 付き。#963）とする。`fit()`/`compile()` 等の
  Keras 風高水準学習ループ API は 1.2 節（Tier 1・#1618）へ移行し、
  対象範囲となった

### 1.2 Tier 1（MLP → CNN → Transformer が組める最小集合。対象範囲・未実装）

REQ-9 2026-09-12 追記（`04-requirements.md:231`）の列挙を、
`docs/compat-feature-gap.md` §2 の大分類順に転記し、実装リポ Phase 2
（親 #1572）の各 issue へ対応付ける。個別機能の設計（dispatch 方式・
数値契約等）は各 issue で決定する。

| Tier 1 機能 | 実装 issue |
|---|---|
| 要素演算（sub／div／pow／sqrt／log／三角関数／比較） | #1592・#1593（#1710 で算術系〈`Var::sub`／`div`／`pow`／`sqrt`〉実装済み。#1711 で `log`／`log2`／`log10`／`sin`／`cos`／`tan`／`abs`／`neg` 実装済み。#1712 で `clamp`／比較演算 6 種〈`Var::clamp`／`gt`／`ge`／`lt`／`le`／`eq`／`ne`〉実装済み。いずれも `ScalarUnaryOp`／`ScalarBinaryOp`〈#1634〉への薄い委譲・facade 到達経路は既存 `Var` 再エクスポート経由・新規 `pub use`／`pub fn` は facade へ追加していない。比較演算の出力は f32 の 0/1 マスク・`Tensor<bool>` 出力は #1613 の対象で本イシュー範囲外。CUDA／Metal 実機での facade parity は未実測のまま Mac／GB10 セッションへ申し送り → Metal〈M4 Max〉は 2026-09-16 に実測済み（pass。`docs/perf/logs/metal-realdevice-phase2-2026-09-16/README.md`）・CUDA〈GB10〉は引き続き未実測 → CUDA〈GB10〉も 2026-09-16 に実測済み（pass。`docs/perf/logs/cuda-realdevice-phase2-2026-09-16/README.md`））。#2145 で `floor`／`ceil`／`round`／`sign`／`reciprocal`／`rsqrt`／`erf`／`pow_scalar` を内部クレート限定（`fandhe_ai_autodiff::scalar_unary_ops`。既存 `Op::ScalarUnary`〈`tensor_core::ScalarUnaryOp` へ新 7 variant 追加〉への薄い委譲のみ・新規 `Op` なし）で実装。#2512 で `Var::floor`／`ceil`／`round`／`sign`／`reciprocal`／`rsqrt`／`erf`／`pow_scalar` の委譲メソッドとして facade 公開済み（`docs/autodiff-scalar-unary-ops-decision.md`） |
| softmax／log_softmax | #1594（実装済み。`Var::softmax`／`log_softmax`・`nn::activation::Softmax`／`LogSoftmax`。facade 到達経路は既存 `Var` 再エクスポート経由〈新規 `pub use`／`pub fn` は facade へ追加しない〉）。#1952 で `log_softmax` backward の Metal 専用カーネル実装済み（`BackendOps::log_softmax_backward`〈#1949 と共有の trait 拡張〉を `crate::log_softmax_backward::MetalLogSoftmaxBackward` へ結線。`exp(y)` の丸め差を除き REQ-2 統一複合判定・`y=0`／`y=-inf` 行は bit 完全一致契約。facade 新規公開面なし。M4 Max 実機実測は未実施のまま Mac セッションへ申し送り〈`docs/perf/logs/metal-log-softmax-backward-1952/README.md`〉）。#1949 で CUDA `log_softmax` backward 専用カーネル実装済み（`BackendOps::log_softmax_backward`〈非破壊拡張・既定 `Unsupported`〉・`crates/backend-cuda/src/{kernels_log_softmax_backward.rs, log_softmax_backward.rs}`。1 warp = 1 行・`Σ_dim(g)` を lane 0 のみが `i` 昇順で逐次和する方式（`double` アキュムレータ）で縮約し、ホスト参照実装〈`grad::log_softmax_vjp_along` の添字昇順逐次和〉と完全同順にする（当初採用していた `__shfl_xor_sync` butterfly reduction は加算順序がホストと異なり、大きさの近い符号違いの値が相殺する入力で桁落ちの位置がずれ REQ-2 統一複合判定を外れる具体例が判明したため撤回した。codex-review 指摘・PR #1994）。`expf` の丸め差異は残るため bit 完全一致は主張せず REQ-2 統一複合判定で検証・tolerance／baseline 不変。CPU は既定のまま〈ホストフォールバック維持〉・forward は無変更。facade 新規公開面なし・GB10 実機実測は未実施のまま `docs/perf/logs/cuda-log-softmax-backward-1949/` へ申し送り → CUDA〈GB10〉は 2026-09-18 に実測済み（`log_softmax_backward_parity` 4/4・facade `softmax_backend_parity` 3/3 pass。`docs/perf/logs/cuda-log-softmax-backward-1949/README.md`）。#2065 で `compat::Sequential::add_softmax`／`add_log_softmax` 実装済み（facade 新規公開面は `add_softmax`／`add_log_softmax` の 2 `pub fn` のみ。§5「適用記録（経路2。イシュー #2065）」参照） |
| GELU／SiLU 等の活性化 | #1595（#1713 で GELU〈誤差関数版・tanh 近似版〉・Softplus 実装済み。`Var::gelu`／`gelu_tanh`／`softplus`・`nn::activation::Gelu`／`GeluTanh`／`Softplus`。facade 到達経路は既存 `Var` 再エクスポート経由〈新規 `pub use`／`pub fn` は facade へ追加していない〉。#1714 で `Silu`／`Hardswish`／`LeakyRelu`／`Elu` 実装済み。`Var::silu`／`hardswish`／`leaky_relu`／`elu`・`nn::activation::{Silu, Hardswish, LeakyRelu, Elu}`・`compat::Sequential::add_silu`／`add_hardswish`／`add_leaky_relu`／`add_elu`。CUDA／Metal 専用カーネル実装済み〈`LeakyRelu`／`Hardswish` は選択・算術のみで bit 同一想定・`Silu`／`Elu` は超越関数を含むため REQ-2 統一複合判定のみ。Metal `Elu` は `expm1` 相当が MSL に無いため `exp`／`log` から桁落ちなく再構成する自作ヘルパー `fai_expm1_f32` を使う〈単純な `exp(x) - 1.0f` はゼロ近傍・大 `alpha` 入力で桁落ちし REQ-2 統一複合判定を満たさなかったため不採用。PR #1825 codex-review P1 是正〉——`scalar_op_source.rs` モジュール doc「`Elu` の `expm1` 非対応」参照〉・facade `compat::Sequential::add_*` 4 件の新規公開面追加は親 #1595 コメント〈2026-09-12 ユーザー承認〉に基づく §5 経路 2 の適用。CUDA／Metal 実機での facade parity は両実装とも未実測のまま Mac／GB10 セッションへ申し送り → Metal〈M4 Max〉は 2026-09-16 に実測済み（pass。`docs/perf/logs/metal-realdevice-phase2-2026-09-16/README.md`）・CUDA〈GB10〉は引き続き未実測 → CUDA〈GB10〉も 2026-09-16 に実測済み（pass。`docs/perf/logs/cuda-realdevice-phase2-2026-09-16/README.md`））。#2065 で `compat::Sequential::add_gelu`／`add_gelu_tanh`／`add_softplus` 実装済み（facade 新規公開面は 3 `pub fn` のみ。`nn::activation::Gelu`／`GeluTanh`／`Softplus` 自体は #1713 実装済みの再利用。§5「適用記録（経路2。イシュー #2065）」参照）。イシュー #2146（親 #2131）で `mish`／`hardtanh`／`relu6`／`prelu`／`glu` を内部クレート限定（`fandhe_ai_autodiff::activation_ops`。facade 非公開の自由関数モジュール）で実装し、`nn::activation::{Mish, Hardtanh, Relu6, PRelu, Glu}` を追加した。`Var::mish`／`hardtanh`／`relu6`／`prelu`／`glu` の委譲メソッドは #2516 で公開済み・`compat::Sequential::add_*` 5 種は #2529 で公開済み（`docs/autodiff-activation-ops-decision.md` §6・§10）） |
| LayerNorm／RMSNorm／BatchNorm | #1596（実装済み。`fandhe_ai_autodiff::Var::rms_norm`／`layer_norm`・`nn::RmsNorm`／`LayerNorm`。facade 到達経路は既存 `Var` 再エクスポート経由——新規 `pub use`／`pub fn` は facade へ追加しない。`docs/norm-ops-design.md`）・#1608 → #1732 で CPU 実装済み（`Var::batch_norm`／`batch_norm_with_batch_stats`／`batch_norm_infer`・`nn::BatchNorm1d`／`BatchNorm2d`〈train／eval・running stats。本クレート初のモード依存層〉。facade 新規公開面なし。#1735 で CUDA 実装済み（`CudaBackendOps::batch_norm_train`／`batch_norm_infer`。facade 新規公開面なし。GB10 実機 parity は未実測のまま申し送り → CUDA〈GB10〉は 2026-09-16 に実測済み（pass。`docs/perf/logs/cuda-realdevice-phase2-2026-09-16/README.md`））・#1736 で Metal 実装済み（facade 新規公開面なし。M4 Max 実機実測は未実施のまま `docs/perf/logs/metal-batch-norm-1736/` へ申し送り → Metal〈M4 Max〉は 2026-09-16 に実測済み（pass。`docs/perf/logs/metal-realdevice-phase2-2026-09-16/README.md`）・CUDA〈GB10〉は引き続き未実測 → CUDA〈GB10〉も 2026-09-16 に実測済み（pass。`docs/perf/logs/cuda-realdevice-phase2-2026-09-16/README.md`））。`docs/batch-norm-ops-design.md`）。**#1760 で `compat::Sequential::add_layer_norm`／`add_rms_norm`／`add_batch_norm1d`／`add_batch_norm2d` 実装済み**（facade 新規 `pub fn` 4 件。`nn::Module` に `as_layer_norm`／`as_rms_norm`／`as_batch_norm1d`／`as_batch_norm2d`〈`_mut` 込み〉フックを追加し `compat::Sequential` の `bind`／`trainable_parameters`／`apply_parameters`〈`Module::set_parameter` による in-place 更新で BatchNorm の running stats／`num_batches_tracked`／`training` を保持〉に結線。新規 `Op`／`BackendOps`／VJP なし・デバイス常駐経路は fail-closed 拒否のまま。CUDA／Metal 実機での facade parity は未実測のまま Mac／GB10 セッションへ申し送り → Metal〈M4 Max〉は 2026-09-16 に実測済み（pass。`docs/perf/logs/metal-realdevice-phase2-2026-09-16/README.md`）・CUDA〈GB10〉は引き続き未実測 → CUDA〈GB10〉も 2026-09-16 に実測済み（pass。`docs/perf/logs/cuda-realdevice-phase2-2026-09-16/README.md`））**#1950 で CUDA backward（`BackendOps::rmsnorm_backward`／`layer_norm_backward`。`crate::norm_backward::CudaNormBackward`）実装済み**（forward・既存 backward API は無変更・facade 新規公開面なし。GB10 実機実測は未実施のまま `docs/perf/logs/cuda-norm-backward-1950/` へ申し送り → CUDA〈GB10〉は 2026-09-18 に実測済み（`norm_backward_parity` 5/5・facade `norm_backend_parity` `cuda_` 4/4・非後退確認 9/9 pass。`docs/perf/logs/cuda-norm-backward-1950/README.md`）。Metal backward は #1953 のまま対象外）。 |
| 形状操作（permute／squeeze／expand／cat／stack／split） | #1597（実装済み: permute／squeeze／unsqueeze／expand〈broadcast_to〉／flatten。facade 到達経路は既存 `Var` 再エクスポート経由〈新規 `pub use`／`pub fn` なし〉）・#1598（実装済み: `Var::cat`／`stack`／`narrow`／`split`／`split_with_sizes`／`chunk`。§5 は Tier 1 列挙済み機能につき再適用不要と判断し facade へ新規 `pub use`／`pub fn` を追加していない。narrow は本 issue で `#1599` 側の対象から解消済み）。#2065 で `compat::Sequential::add_flatten` 実装済み（`nn::Flatten`。`Var::flatten`〈#1597〉の薄いラッパー。facade 新規公開面は `add_flatten` の 1 `pub fn` のみ。§5「適用記録（経路2。イシュー #2065）」参照）。#2143 で `repeat`／`tile`／`flip`／`roll` を内部クレート限定（`fandhe_ai_autodiff::rearrange_ops`。既存 `Var::index_select`／`broadcast_to`／`reshape`〈`repeat` の空テンソル最終化のみ〉の合成のみ・新規 `Op` なし）で実装。facade 公開は #2511 で `Var::flip`／`roll`／`repeat`／`tile` の委譲メソッドとして公開済み（`docs/autodiff-rearrange-ops-decision.md`）。#2144 で `tril`／`triu`／`diag`／`trace`／`outer`／`dot` を内部クレート限定（`fandhe_ai_autodiff::matrix_ops`。既存 `Var::masked_fill`／`gather`／`narrow`／`broadcast_to`／`transpose`／`pad`／`squeeze`／`mul`／`sum` の合成のみ・新規 `Op` なし）で実装。facade 公開は #2513 で `Var::tril`／`triu`／`diag`／`trace`／`outer`／`dot` の委譲メソッドとして公開済み（`docs/autodiff-matrix-ops-decision.md`）） |
| index 系（narrow／where／gather／scatter） | #1599（narrow は #1598 で実装済み・where／masked_fill は #1637 で実装済み〈`Var::where_cond`／`masked_fill`。facade 到達経路は既存 `Var` 再エクスポート経由・新規 `pub use`／`pub fn` は facade へ追加していない〉のため対象外。gather／scatter／scatter_add／index_select は #1638（→ #1776 で Op 定義・CPU 参照実装・VJP 実装済み。`Var::gather`／`index_select`／`scatter`／`scatter_add`。facade 到達経路は既存 `Var` 再エクスポート経由で同様に新規公開面なし。CUDA は #1777・Metal は #1778 でそれぞれ実装済み〈`CudaBackendOps`／`MetalBackendOps` の `gather`／`scatter`。facade 新規公開面なし〉）のため対象外。#2148 で `advanced_indexing`／`index_put`／`index_put_` を内部クレート限定（`fandhe_ai_autodiff::indexing_ops`。既存 `Var::index_select`／`scatter`／`scatter_add`／`reshape`／`broadcast_to` の合成のみ・新規 `Op` なし）で実装。#2518 で `Var::advanced_indexing`／`index_put`／`index_put_` の委譲メソッドとして facade 公開済み（§5「適用記録（経路 2。イシュー #2518）」・`docs/autodiff-indexing-inplace-design.md` §6 参照）） |
| バッチ行列積 | #1600（#1715 で実装済み: `Var::matmul` の rank≥3 受理・バッチ次元 NumPy 互換ブロードキャスト・`BackendOps::gemm_batched`／`gemm_batched_fp32_strict`〈既定は per-batch `gemm`/`gemm_fp32_strict` への合成〉・`CpuBackendOps::gemm_batched` オーバーライド〈bit 同一〉。§5 は Tier 1 列挙済み機能につき再適用不要と判断し facade へ新規 `pub use`／`pub fn` を追加していない。#1716 で CUDA 実装済み: `CudaBackendOps::gemm_batched`／`gemm_batched_fp32_strict` 専用オーバーライド（デバイス常駐バッチループ経路 `CudaGemm::run_tiled_f32_batched`。A・B を 1 回ずつ H2D・出力を 1 本だけ確保・バッチごとに既存 NN tiled f32 カーネルを起動・D2H 1 回。`gemm_fp32_strict_impl` 単体入口と bit 同一のカーネル選択。TF32 opt-in 時は `Tf32`／`Tf32x3` とも新設 `fandhe_ai_tensor_core::gemm_batched_via_per_batch_gemm` 経由で per-batch `gemm` 合成へフォールバックしカウンタ・fail-closed 挙動は #1042／#1355 のまま不変）。facade 新規公開面なし・GB10 実機実測は未実施のまま申し送り → CUDA〈GB10〉は 2026-09-16 に実測済み（pass。`docs/perf/logs/cuda-realdevice-phase2-2026-09-16/README.md`）。#1717 で `MetalBackendOps::gemm_batched`／`gemm_batched_fp32_strict` 専用オーバーライド実装済み〈正規化済みオペランドを 1 回ずつ upload・バッチごとの GEMM を encode-only で 1 コマンドバッチへ積み `download` 1 回のみ同期する「バッチループ方式」。`gemm.rs`／shader 自体は無変更。per-batch `gemm`〈`dispatch_auto`／split-K〉とは classic strided カーネル経由のため bit 同一は主張せず REQ-2 統一複合判定。split-K 実行時トグル非依存。facade 新規公開面なし。M4 Max 実機実測は未実施のまま Mac セッションへ申し送り → Metal〈M4 Max〉は 2026-09-16 に実測済み（pass。`docs/perf/logs/metal-realdevice-phase2-2026-09-16/README.md`）・CUDA〈GB10〉は引き続き未実測 → CUDA〈GB10〉も 2026-09-16 に実測済み（pass。`docs/perf/logs/cuda-realdevice-phase2-2026-09-16/README.md`）〉） |
| 縮約（mean／min／argmax／var／std／複数軸） | #1601（`amax`／`max` の勾配分配方式はこの issue で設計判断を記録して確定。`04-requirements.md:234`。2 節参照）。#1719 で mean・複数軸・keepdim 部分を実装済み: `Var::mean`〈単一軸／全軸。`tape::Op::Mean`。`BackendOps` は非拡張で `sum` の結果をホスト側で 1 回だけ除算〉・`sum_dims`／`max_dims`／`mean_dims`〈複数軸・`keepdim`。`crate::reduce_dims` が permute／contiguous／reshape で単一軸縮約へ併合。単一軸は `sum(Some(d))` と bit 同一に直接委譲〉。facade 到達経路は既存 `Var` 再エクスポート経由〈新規 `pub use`／`pub fn` なし〉。CUDA 実機 parity は未実測のまま GB10 セッションへ申し送り。#1720 で `min`／`argmax`／`argmin` 実装済み（`Var::min`／`argmax`／`argmin`。facade 到達経路は既存 `Var` 再エクスポート経由・新規 `pub use`／`pub fn` は facade へ追加していない。`min` の VJP は #1718〈amax／amin 勾配分配方式の確定。本追記時点で OPEN〉未確定につき現行 `Var::max` の先勝ち決定的方式と共有するヘルパー `grad::extremum_first_match_vjp`〈旧 `max_vjp` を改称・`max_vjp` 自体は薄いラッパーとして維持〉を適用。CPU は `min`／`argmax`／`argmin` すべて実装済み。CUDA は `min`／`argmax`／`argmin` すべてカーネル実装済み（`argmax`／`argmin` はイシュー #1948 で `arg_reduce::CudaArgReduce` として実装。走査契約〈タイは最初の添字・NaN 無視〉は CPU 参照実装と添字完全一致。facade 新規公開面なし。GB10 実機実測は未実施のまま `docs/perf/logs/cuda-argmax-argmin-1948/` へ申し送り → CUDA〈GB10〉は 2026-09-18 に実測済み（`argmax_and_argmin_*` 5/5 pass・添字完全一致。`docs/perf/logs/cuda-argmax-argmin-1948/README.md`））。Metal は `argmax`／`argmin` をイシュー #1951 で `crate::reduce::MetalReduce` の argext カーネルへ結線済み化（CPU 参照実装と添字完全一致。facade 新規公開面なし。M4 Max 実機実測は未実施のまま `docs/perf/logs/metal-argext-1951/README.md` へ申し送り）。`min` は CUDA・Metal ともカーネル実装済み（Metal は #1951 以前から実装済み）。var／std は #1723 で実装済み: `Var::var`／`std`〈既存 `sum`／`max` と同じ `dim: Option<usize>` シグネチャ＋`correction`。`Op::Var` 専用 Op・`BackendOps::var` 既定 `Unsupported` のホストフォールバック契約。§5「Tier 1 列挙済み機能は再適用不要」の対象。facade 到達経路は既存 `Var` 再エクスポート経由・新規 `pub use`／`pub fn` なし。多軸・`keepdim` は本 issue 対象外のまま〉。`docs/compat-feature-gap.md` §2.5 追補参照。`amax`／`max` の勾配分配方式〈先勝ち決定的 対 均等分配〉は #1718 で確定（既存 `Var::max`／`min`／`max_dims`〈`max(dim)`／`min(dim)` 族の意味論〉は先勝ち決定的を維持・PyTorch `torch.amax`／`amin` 相当の均等分配は別 `Op`／別ヘルパーとして後続 issue で実装する方針。`docs/autodiff-amax-grad-distribution-decision.md` 参照）のため `sum_dims`／`max_dims` とも既存の先勝ち決定的規約を無変更で維持）。`prod`／`logsumexp`／`any`／`all`／`norm_p`（p-ノルム）は #2147 で内部クレート限定の自由関数として実装済み: `fandhe_ai_autodiff::reduce_ops::{prod, logsumexp, any, all, norm_p}`〈`crates/autodiff/src/reduce_ops.rs`〉。`prod`（`cumprod` → `narrow` → `contiguous` → `squeeze` の合成）・`any`／`all`（`ne` → `max`／`min` の合成。出力は厳密 0.0／1.0 で bit 完全一致・勾配ゼロ）は新規 `Op` なし。`logsumexp`／`norm_p` は専用 `Op::LogSumExp`／`Op::PNorm`（`BackendOps::logsumexp`／`vector_norm_p`〈defaulted・既定 `Unsupported`〉のホストフォールバック契約。`Var::var`／`vector_norm` と同型）。**facade 公開（経路 2）は #2514 で `Var::prod`／`logsumexp`／`any`／`all`／`norm_p` の委譲メソッドとして公開済み（以下の保留ガードに関する記述は #2147 時点のもの）**——`crates/facade/src/lib.rs::VarReduceOpsHoldDoctestGuard`（正のプローブ doctest）＋`api_surface.rs` の 4 テストで到達不能を多層固定している点が、既に facade 公開済みの本行の他エントリ（`mean`／`min`／`argmax`／`var`／`std` 等）と異なる。設計判断・数値契約・PyTorch との差分は `docs/autodiff-reduce-ops-decision.md` を正とする。CUDA／Metal 実機実測は未実施のまま `docs/perf/logs/reduce-ops-2147/README.md` へ申し送り） |
| 乱数生成と RNG 契約 | #1602（#1724 実装済み: グローバル RNG 契約〈manual_seed〉。#1725 実装済み: 乱数テンソル生成本体〈randn／rand／randint〉。#1726 実装済み: 決定的生成系〈arange／linspace／eye／zeros_like／ones_like〉。`fandhe_ai::{manual_seed, randn, rand, randint, arange, linspace, eye, zeros_like, ones_like}`。facade 到達経路は新規 `pub fn`（`RngError`／`CreationError` は 1 行の `pub use`）。`docs/rng-global-contract-design.md`）。**#2156 実装済み（内部クレート限定）**: 確率分布サンプラー〈`bernoulli`／`multinomial`／`normal`〉とグローバル状態と独立の `Generator`（`fandhe_ai_autodiff::{bernoulli, multinomial, normal, Generator}`。実体は `tensor-core::rng`）。**#2593 で facade 公開済み**（`fandhe_ai::{bernoulli, multinomial, normal, Generator}`。crate ルートの委譲 `pub fn` 3 件＋`pub use` 1 行。ガードは承認形だけを許す正ガードへ反転。`docs/rng-distributions-generator-decision.md` §5.2）。#2157 で決定性モード（`set_deterministic`／`is_deterministic`）を `fandhe_ai_autodiff::determinism`〈`Var` を経由しない自由関数〉として実装。CPU バックエンドの棚卸し結果、本番経路に非決定的な縮約は見つからず no-op 契約として確定。**#2507 で facade 公開済み**（`fandhe_ai::{set_deterministic, is_deterministic}`。crate ルートの委譲 `pub fn`。`docs/autodiff-determinism-mode-design.md`）） |
| Dropout | #1603（実装済み。`Var::dropout(p, training)`〈`torch.nn.functional.dropout` 相当。inverted dropout〉・`tape::Op::Dropout`・`nn::Dropout`〈`p`／`training` を保持し `Module::set_training`／`training` を実際にオーバーライドする本クレート内実装で唯一の層。既定 `training=true`〉・`compat::Sequential::add_dropout`（facade 新規 `pub fn` 1 件）。マスク生成は `crate::grad::dropout_mask` が `fandhe_ai_tensor_core::rng::rand`（#1602 のホスト側グローバル RNG 契約）を経由し `BackendOps` を経由しない——forward／backward とも既存必須メソッド `BackendOps::mul` への単一乗算に帰着するため `BackendOps` 非拡張（Embedding／SDPA／einsum と同型の合成方針）で 3 バックエンド bit 同一が構造的に成立する。`training=false` または `p=0.0` は早期リターン（`Op` 非記録・RNG 非消費）。`predict`／`forward_host` とモードの整合は「PyTorch `model(x)` と同じくコンテナの `training` フラグを尊重する」方式で確定（Keras の「常に推論モード」は不採用）——`compat::Sequential::predict` は `training=true` のままだと `Dropout` を適用するため、決定的な推論には呼び出し側が先に `eval()` を呼ぶ必要がある（`nn::Dropout` モジュール doc「train／eval と `predict`／`forward_host` の整合」参照）。CUDA／Metal 実機での facade parity テストは未実測のまま Mac／GB10 セッションへ申し送り → Metal〈M4 Max〉は 2026-09-16 に実測済み（pass。`docs/perf/logs/metal-realdevice-phase2-2026-09-16/README.md`）・CUDA〈GB10〉は引き続き未実測 → CUDA〈GB10〉も 2026-09-16 に実測済み（pass。`docs/perf/logs/cuda-realdevice-phase2-2026-09-16/README.md`））。**#2161（親 #2131）で `Dropout2d`（チャネル単位 dropout）・`AlphaDropout`（SELU 向けアフィン補正）を内部クレート限定で追加**（`nn::Dropout2d`／`nn::AlphaDropout`。いずれも `Var::dropout_with_mask` の再利用・新規 `Op` なし）。facade 公開（`compat::Sequential::add_dropout2d`／`add_alpha_dropout`）は未承認のまま保留（`docs/autodiff-dropout-embedding-bag-decision.md` §6）。CUDA／Metal 実機は未実測のまま申し送り（`docs/perf/logs/dropout-embedding-bag-2161/README.md`） |
| Embedding | #1604（実装済み。`Var::embedding`〈`tape::Op::Embedding`。gather を forward・`scatter_add`〈`ScatterReduce::Add` の決定的集約契約〉を backward に使う合成。`BackendOps` 非拡張〉・`nn::Embedding`／`EmbeddingVars`〈`padding_idx` 対応。forward は当該行を素通し・backward のみゼロ上書き〉。`Module` trait は非実装（id 入力が f32 `Var` 契約と不一致・`compat::Sequential` の学習可能パラメータ収集は `as_linear` フック限定のため、実装すると黙って学習されない罠になる。詳細は `crates/autodiff/src/nn/embedding.rs` モジュール doc）。facade 到達経路は既存 `Var`／`nn` 再エクスポート経由・新規 `pub use`／`pub fn` は facade へ追加していない。CUDA／Metal 実機での facade parity は未実測のまま Mac／GB10 セッションへ申し送り → Metal〈M4 Max〉は 2026-09-16 に実測済み（pass。`docs/perf/logs/metal-realdevice-phase2-2026-09-16/README.md`）・CUDA〈GB10〉は引き続き未実測 → CUDA〈GB10〉も 2026-09-16 に実測済み（pass。`docs/perf/logs/cuda-realdevice-phase2-2026-09-16/README.md`））。**#1760 で `compat::Sequential::add_embedding` 実装済み**（facade 新規 `pub fn` 1 件。`Embedding` へ `impl Module` を新設〈`nn/embedding.rs` モジュール doc「`Module` trait は実装しない（確定判断）」を解消——`EmbeddingVars::forward_from_var` が f32 `Var` 入力を `Var::to_tensor()` で実体化し厳格な f32→i32 変換〈非有限・非整数・負・`i32::MAX` 超過を拒否。黙示の飽和・切り捨て変換はしない〉を挟んで橋渡しする〉・`as_embedding`／`_mut` フックで `compat::Sequential` の学習経路へ結線。CUDA／Metal 実機での facade parity は未実測のまま Mac／GB10 セッションへ申し送り → Metal〈M4 Max〉は 2026-09-16 に実測済み（pass。`docs/perf/logs/metal-realdevice-phase2-2026-09-16/README.md`）・CUDA〈GB10〉は引き続き未実測 → CUDA〈GB10〉も 2026-09-16 に実測済み（pass。`docs/perf/logs/cuda-realdevice-phase2-2026-09-16/README.md`））。**#2161（親 #2131）で `EmbeddingBag`（bag 単位 sum／mean／max 縮約）を内部クレート限定で追加**（`nn::EmbeddingBag`／`EmbeddingBagVars`／`EmbeddingBagMode`。`Var::embedding` と縮約演算の合成・新規 `Op` なし）。facade 公開（`compat::Sequential::add_embedding_bag`）は未承認のまま保留（`docs/autodiff-dropout-embedding-bag-decision.md` §6）。CUDA／Metal 実機は未実測のまま申し送り（`docs/perf/logs/dropout-embedding-bag-2161/README.md`） |
| MultiheadAttention | #1605（sub-issue (a): #1639 で実装済み。`Var::scaled_dot_product_attention`——既存の `matmul`〈rank≥2〉／`transpose`／`mul`／`masked_fill`／`softmax` への分解のみで実装し `Op`／`BackendOps` を新規拡張しない（`crate::einsum` と同型）。causal（top-left aligned）／明示 `attn_mask`〈PyTorch bool 規約〉・`scale` 既定値〈`1/sqrt(E)`〉に対応。facade 到達経路は既存 `Var` 再エクスポート経由・新規 `pub use`／`pub fn` は facade へ追加していない。`dropout_p`／`enable_gqa`／attention weights 返却／f16・bf16 は対象外。CUDA／Metal 実機での facade parity は未実測のまま Mac／GB10 セッションへ申し送り → Metal〈M4 Max〉は 2026-09-16 に実測済み（forward pass・backward は Metal tape 上の loss 縮約 `Var::sum` が `MetalBackendOps::sum` 未実装の `Unsupported` を返すため判定不能＝FAIL 記録。演算自体の不一致は未観測。同 README §3.1）・CUDA〈GB10〉は引き続き未実測 → CUDA〈GB10〉も 2026-09-16 に実測済み（forward pass・backward は CUDA reduction カーネル〈`kernels_reduce.rs`〉が GB10 の NVRTC で `identifier "INFINITY" is undefined` のコンパイルエラーとなり `Var::sum` が失敗するため判定不能＝FAIL 記録。演算自体の不一致は未観測。同 README §3.1）。sub-issue (b) `MultiheadAttention` Module〈in/out projection・head 分割〉は #1640 で実装済み: `nn::MultiheadAttention`／`MultiheadAttentionVars`（**#1760 で `compat::Sequential::add_multihead_attention` 実装済み**。facade 新規 `pub fn` 1 件。`as_multihead_attention`／`_mut` フックを追加し `Module::forward` の self-attention 契約〈`q=k=v=input`・mask なし・非 causal〉のまま `compat::Sequential` の学習経路へ結線。CUDA／Metal 実機での facade parity は未実測のまま Mac／GB10 セッションへ申し送り → Metal〈M4 Max〉は 2026-09-16 に実測済み（pass。`docs/perf/logs/metal-realdevice-phase2-2026-09-16/README.md`）・CUDA〈GB10〉は引き続き未実測 → CUDA〈GB10〉も 2026-09-16 に実測済み（pass。`docs/perf/logs/cuda-realdevice-phase2-2026-09-16/README.md`））。q/k/v/out の 4 `nn::Linear` 合成・`Var::matmul`〈rank≥3 バッチ〉／`transpose`／`masked_fill`／`softmax` の既存演算合成のみで新規 `Op`／`BackendOps` メソッドは追加していない。実装着手時点で #1639 が未マージだったため、attention 本体（scale・mask／causal・softmax）は `Var::scaled_dot_product_attention` を呼ばず #1639 と同一の数式・mask 極性〈`true`=attend〉・causal 規約〈top-left aligned `j<=i`〉を private ヘルパーとして複製している（`nn/attention.rs::sdpa_compose`。#1639 マージ後の置き換えは本 issue のスコープ外として未実施のまま残る）。入出力は rank-3・batch_first 固定（`[B,L,E]`／`[B,S,E]`）。facade 新規公開面なし（既存 `Var`／`nn` 再エクスポート経由）。CUDA／Metal 実機での facade parity テストは未実測のまま Mac／GB10 セッションへ申し送り → Metal〈M4 Max〉は 2026-09-16 に実測済み（pass。`docs/perf/logs/metal-realdevice-phase2-2026-09-16/README.md`）・CUDA〈GB10〉は引き続き未実測 → CUDA〈GB10〉も 2026-09-16 に実測済み（pass。`docs/perf/logs/cuda-realdevice-phase2-2026-09-16/README.md`）） （**#2163 でオプション〈`batch_first`・`kdim`/`vdim`・`key_padding_mask`〉を追加済み**: `nn::MultiheadAttentionConfig`・`MultiheadAttentionVars::forward_with_key_padding_mask`。既存 `Var` 演算の合成のみで新規 `Op`／`BackendOps`／カーネル追加なし。既定経路（`batch_first=true`・`key_padding_mask=None`）は bit 同一。facade 公開は #2530 で公開済み（`compat::MultiheadAttentionConfig` 再エクスポート・`compat::Sequential::add_multihead_attention_with_config`。保留ガードは正ガードへ反転）。CUDA／Metal 実機 parity は未実測のまま申し送り。詳細は `docs/autodiff-mha-options-decision.md`） |
| TransformerEncoderLayer | #2068・親 #2059（実装済み。`nn::TransformerEncoderLayer`／`TransformerEncoderLayerVars`。既存の `nn::MultiheadAttention`／`nn::Linear`／`nn::LayerNorm` の合成のみで新規 `Op`／`BackendOps`／VJP を追加しない。self-attention（`q=k=v=current`・mask なし・非 causal）→ residual → LayerNorm → FFN（`relu` 活性化固定）→ residual → LayerNorm の post-norm 合成（PyTorch `nn.TransformerEncoderLayer` の既定 `norm_first=False`・`activation="relu"` と揃える）。Dropout は結線しない（`nn/transformer_encoder_layer.rs` モジュール doc「対象外」参照）。`Module::supports_forward_host` は `false`（`nn::attention`・`nn::embedding` と同型）。**#2068 で `compat::Sequential::add_transformer_encoder` 実装済み**（facade 新規 `pub fn` 1 件。`as_transformer_encoder_layer`／`_mut` フックを追加し `compat::Sequential` の学習経路〈`bind`／`trainable_parameters`／`apply_parameters`〈`Module::set_parameter` による in-place 更新〉〉へ結線。デバイス常駐経路は fail-closed 拒否のまま。§5「適用記録（イシュー #2068）」参照）。CUDA／Metal 実機での facade parity は未実測のまま Mac／GB10 セッションへ申し送り（`docs/perf/logs/compat-sequential-transformer-encoder-2068/`）。**#2165（親 #2131。#2068 の対）で `TransformerDecoderLayer`（self-attention→cross-attention→FFN の合成。`nn::MultiheadAttention` 2 個〈`self_attn`／`multihead_attn`。両方とも非既定 config を fail-closed 拒否〉・`nn::Linear` 2 層・`nn::LayerNorm` 3 個）と `Transformer`（encoder スタック＋`encoder_norm`・decoder スタック＋`decoder_norm`。`TransformerConfig` ビルダー）を内部クレート限定で追加**（新規 `Op`／`BackendOps`／VJP なし。`Module::as_transformer_decoder_layer`／`as_transformer`〈各 `_mut` 込み〉フックを追加。**facade 公開は当初保留だったが、#2532 で decoder 1 層、#2533 で `Transformer` を公開済み**（`compat::Sequential::add_transformer_decoder_layer`／`add_transformer`・`Transformer`／`TransformerDecoderLayer`／`TransformerConfig` の再エクスポート。保留ガード `TransformerDecoderHoldDoctestGuard` は #2533 で撤去し `api_surface.rs` の正ガードへ反転。§5 適用記録参照）。CUDA／Metal 実機は未実測のまま申し送り。詳細は `docs/autodiff-transformer-decoder-decision.md`） |
| Conv1d／Conv2d | #1606（設計記録 #1641。`docs/conv-ops-design.md`。#1764 で CPU 実装済み〈`Var::conv2d`。facade 新規公開面なし〉。#1765 で `Var::conv1d`〈`Var::conv2d` の reshape 併合。新規 `Op`／`BackendOps`／カーネルなし・facade 新規公開面なし〉も実装済み。nn 層は #1645・CUDA／Metal 専用 im2col／col2im カーネルは #1643／#1644 が対象。#1766 で CUDA `im2col`／`col2im` 実装済み〈facade 新規公開面なし・GB10 実機 parity は未実測 → CUDA〈GB10〉は 2026-09-16 に実測済み（forward pass・backward は CUDA reduction カーネル〈`kernels_reduce.rs`〉が GB10 の NVRTC で `identifier "INFINITY" is undefined` のコンパイルエラーとなり `Var::sum` が失敗するため判定不能＝FAIL 記録。演算自体の不一致は未観測。`docs/perf/logs/conv-realdevice-1771/cuda/`）〉。#1767 で CUDA Conv1d 経路を検証済み〈`Var::conv1d` は #1766 の CUDA `im2col`／`col2im` へ reshape 併合のみで自動到達済み・新規カーネルなし・facade 新規公開面なし・1d 形状の parity・bit 同一テストを追加・GB10 実機実測は未実施のまま `docs/perf/logs/cuda-conv1d-1767/` へ申し送り → CUDA〈GB10〉は 2026-09-16 に実測済み（forward pass・backward および `cuda_conv1d_matches_manual_reshape_conv2d_bit_exact`〈conv1d↔conv2d bit 一致〉は同じ `Var::sum` の NVRTC `INFINITY` 未定義エラーにより判定不能＝FAIL 記録。演算自体の不一致は未観測。`docs/perf/logs/conv-realdevice-1771/cuda/`）〉。#1768 で Metal `im2col`／`col2im` 実装済み〈facade 新規公開面なし・M4 Max 実機 parity は未実測 → Metal〈M4 Max〉は 2026-09-16 に実測済み（forward pass・backward は Metal tape 上の loss 縮約 `Var::sum` が `MetalBackendOps::sum` 未実装の `Unsupported` を返すため判定不能＝FAIL 記録。演算自体の不一致は未観測。同 README §3.1）・CUDA〈GB10〉は引き続き未実測 → CUDA〈GB10〉も 2026-09-16 に実測済み（forward pass・backward は CUDA reduction カーネルの NVRTC `INFINITY` 未定義エラーにより判定不能＝FAIL 記録。`docs/perf/logs/conv-realdevice-1771/cuda/`）〉。#1769 で Metal Conv1d 経路を検証済み〈新規カーネルなし・facade 新規公開面なし・1d model／ops／facade bit 同一テスト追加・M4 Max 実機実測は未実施のまま `docs/perf/logs/metal-conv1d-1769/` へ申し送り → Metal〈M4 Max〉は 2026-09-16 に実測済み（forward pass・backward は Metal tape 上の loss 縮約 `Var::sum` が `MetalBackendOps::sum` 未実装の `Unsupported` を返すため判定不能＝FAIL 記録。演算自体の不一致は未観測。同 README §3.1）・CUDA〈GB10〉は引き続き未実測 → CUDA〈GB10〉も 2026-09-16 に実測済み（forward pass・backward および conv1d↔conv2d bit 一致は CUDA reduction カーネルの NVRTC `INFINITY` 未定義エラーにより判定不能＝FAIL 記録。`docs/perf/logs/conv-realdevice-1771/cuda/`）〉。#1770 で nn 層（`nn::Conv2d`／`nn::Conv1d`・`Module::as_conv2d`／`as_conv1d`〈各 `_mut` 込み〉フック・`compat::Sequential::add_conv2d`／`add_conv1d`〈facade 新規 `pub fn` 2 件のみ〉）を実装済み化。`trainable_parameters`／`bind`／`SequentialVars::forward`／`trainable_vars`／`trainable_grads`／`apply_parameters` を Conv 層対応へ拡張し optimizer 学習経路へ接続・デバイス常駐経路（`init_device_param_store`／`forward_resident`／`predict_resident`）は Conv 層を含む `Sequential` を `BackendError::Unsupported` で fail-closed 拒否（対象外のまま）。新規 `Op`／`BackendOps`／VJP なし・CUDA／Metal 実機での facade parity テストは未実測のまま #1771 へ申し送り）。#1771 で実機ランブック（`docs/perf/logs/conv-realdevice-1771/`）を新設し #1766〜#1770 の 4 イシューの実行手順・判定規則を統合するとともに、nn 層（`compat::Sequential`）の backward・Conv1d・「特化」契約・学習ループ〈record-only〉を対象とする `#[ignore]` テスト 12 件（CUDA／Metal 各 6 件）を `crates/facade/tests/nn_conv_backend_parity.rs` へ追加。facade 新規公開面なし・CUDA／Metal 実機実測は本エージェント実行環境に実機なしのため未実施のまま親 #1645 を受け皿として GB10／Mac セッションへ申し送り → Metal〈M4 Max〉は 2026-09-16 に実測済み（forward pass・backward は Metal tape 上の loss 縮約 `Var::sum` が `MetalBackendOps::sum` 未実装の `Unsupported` を返すため判定不能＝FAIL 記録。演算自体の不一致は未観測。同 README §3.1）・CUDA〈GB10〉は引き続き未実測 → CUDA〈GB10〉も 2026-09-16 に実測済み（`nn_conv_backend_parity` 6 件中 5 passed・1 failed〈`cuda_sequential_conv1d_matches_manual_reshape_conv2d_bit_exact`。CUDA reduction カーネルの NVRTC `INFINITY` 未定義エラーにより `Var::sum` が失敗し判定不能＝FAIL 記録〉。`docs/perf/logs/conv-realdevice-1771/cuda/`）。**#2158 で Conv3d（im2col の空間 3 軸一般化。CPU 参照実装）を実装済み**（新規型 `Conv3dParams`・新規 `BackendOps` メソッド `im2col3d`／`col2im3d`／`conv3d`〈既定 `Unsupported`。CPU のみ `im2col3d`／`col2im3d` を override〉・新規 `Op::Conv3d`・`nn::Conv3d`／`Conv3dVars`・`Module::as_conv3d`／`_mut`。入口は自由関数 `fandhe_ai_autodiff::conv3d_ops::conv3d`〈**facade 新規公開面なし**——`Var::conv3d`／`compat::Sequential::add_conv3d` は未承認のため保留し `crates/facade/src/lib.rs::VarConv3dHoldDoctestGuard` で機械的に固定。窓口は #2158／親 #2131〉。CUDA／Metal は override なし・実機 parity は `docs/perf/logs/conv3d-2158/` へ申し送り。詳細設計は `docs/conv-ops-design.md` §16）|
| Pooling | #1607（設計記録 #1727。`docs/pooling-ops-design.md`。#1728 で CPU 実装済み: `Var::max_pool2d`／`max_pool1d`／`avg_pool2d`／`avg_pool1d`／`adaptive_avg_pool2d`／`adaptive_avg_pool1d`・`nn::{MaxPool1d, MaxPool2d, AvgPool1d, AvgPool2d, AdaptiveAvgPool1d, AdaptiveAvgPool2d}`・`BackendOps::max_pool2d`／`avg_pool2d`／`adaptive_avg_pool2d`〈既定 `Unsupported`〉。MaxPool は forward（値・索引）・backward とも CPU 参照実装とホストフォールバック（`eval::*`）で bit 完全一致・AvgPool／AdaptiveAvgPool も `f64` アキュムレータ縮約契約が両実装で同一のため同じく bit 完全一致。facade 到達経路は既存 `Var` 再エクスポート経由（新規公開面なし）。CUDA／Metal 専用カーネルは #1729／#1730 が対象のまま（既定 `Unsupported` → ホストフォールバックのため機能的には到達可能）。#1729 で CUDA forward カーネル実装済み〈`crates/backend-cuda/src/{pooling,kernels_pooling}.rs`。MaxPool／AvgPool／AdaptiveAvgPool の 2d カーネル 3 種・値/索引 bit 完全一致契約。実装時点で #1728〈共有基盤 `Pool2dParams`・`BackendOps` 3 メソッド〉が未マージだったため `ops.rs::CudaBackendOps` への override 配線は含まない〈#1728 マージ後の追従 PR へ引き継ぐ〉・facade 新規公開面なし・GB10 実機実測は未実施のまま `docs/perf/logs/cuda-pooling-1729/` へ申し送り → CUDA〈GB10〉は 2026-09-16 に実測済み（pass。`docs/perf/logs/cuda-realdevice-phase2-2026-09-16/README.md`）〉）。#1730 で Metal forward カーネル実装済み〈`crates/backend-metal/src/{pooling.rs, pooling_model.rs, shaders/pooling.metal}`。Max は bit 完全一致・Avg／Adaptive は soft-f64 で CPU `f64` 参照実装と bit 完全一致。実装時点で #1728 が未マージだったため `ops.rs::MetalBackendOps` への override 配線は同様に含まなかった〈#1728 マージ後の追従 PR へ引き継ぐ〉・facade 新規公開面なし・M4 Max 実機実測は未実施のまま `docs/perf/logs/metal-pooling-1730/README.md` へ申し送り → Metal〈M4 Max〉は 2026-09-16 に実測済み（pass。`docs/perf/logs/metal-realdevice-phase2-2026-09-16/README.md`）・CUDA〈GB10〉は引き続き未実測 → CUDA〈GB10〉も 2026-09-16 に実測済み（pass。`docs/perf/logs/cuda-realdevice-phase2-2026-09-16/README.md`）〉。**#1729／#1730 マージ後の追従 PR（#1607 ツリー）で `ops.rs::CudaBackendOps`／`MetalBackendOps` への override 配線を完了**（`context_cache::cached_pooling`〈両バックエンド〉・`map_pooling_error`〈`PoolingSizeLimitExceeded` → `Unsupported`・`InvalidPoolingShape` → `ShapeMismatch`〉を追加し、既定 `Unsupported` フォールバックに落ちていた本番経路をカーネル実装へ接続。Linux で実行可能な shape 再検査の早期リターン回帰テストを両バックエンドへ追加〈CUDA は driver 非接触・Metal はファイル自体が `cfg(target_os = "macos")` 限定〉。CUDA／Metal 実機でのカーネル数値実測は #1729／#1730 と同様に未実施のまま申し送り → Metal〈M4 Max〉は 2026-09-16 に実測済み（pass。facade 配線経由の `facade_pooling_backend_parity.log` 3 pass。`docs/perf/logs/metal-realdevice-phase2-2026-09-16/README.md`）・CUDA〈GB10〉は引き続き未実測 → CUDA〈GB10〉も 2026-09-16 に実測済み（pass。facade 配線経由の `facade_pooling_backend_parity.log` 3 pass。`docs/perf/logs/cuda-realdevice-phase2-2026-09-16/README.md`））。**#1957 で `compat::Sequential::add_max_pool2d`／`add_max_pool1d`／`add_avg_pool2d`／`add_avg_pool1d`／`add_adaptive_avg_pool2d`／`add_adaptive_avg_pool1d` 実装済み**（2026-09-17 ユーザー承認〈選択肢 A・6 メソッド一括追加〉。`nn::Module` に `is_pooling(&self) -> bool`〈`as_relu` と同型の bool フック・既定 `false`〉を追加し Pooling 6 型でオーバーライド、`compat::Sequential::contains_resident_unsupported_layer` の判定へ組み込み常駐経路を fail-closed 拒否。facade 新規 `pub fn` 6 件のみ・新規 `Op`／`BackendOps`／VJP なし。受入条件（`Sequential` 経由の出力が直接呼び出しと bit 完全一致）は `crates/facade/tests/compat_sequential_pooling.rs` で検証済み。CUDA／Metal 実機での facade parity は未実測のまま Mac／GB10 セッションへ申し送り（`crates/facade/tests/compat_sequential_pooling_backend_parity.rs`））。**#2160 で `nn::{AdaptiveMaxPool2d, AdaptiveMaxPool1d, GlobalPool}`（`GlobalPoolMode::{Avg,Max}`）実装済み**（MaxUnpool は引き続きスコープ外。設計判断・承認事項は `docs/autodiff-adaptive-max-global-pool-decision.md` を正とする。新規 `Op` なし〈既存 `Op::MaxPool2d` を forward 記録に再利用〉・`GlobalPool(Avg)` は既存 `adaptive_avg_pool2d`／`adaptive_avg_pool1d` へ委譲・CPU 参照実装とホストフォールバックで bit 完全一致。**#2160 時点では内部クレート限定だったが、イシュー #2527 で `Var::adaptive_max_pool2d`／`adaptive_max_pool1d`・`compat::Sequential::add_adaptive_max_pool2d`／`add_adaptive_max_pool1d`／`add_global_pool`・ルート `GlobalPoolMode` を公開済み**（§5「適用記録（経路 2。イシュー #2527）」。層型の再エクスポート・`Tensor`／`Tape` メソッドは未承認のまま `AdaptiveMaxGlobalPoolHoldDoctestGuard`〈`crates/facade/src/lib.rs`〉で保留固定）。CUDA／Metal 専用カーネルは未実装〈既定 `Unsupported` → ホストフォールバック〉のため CPU と bit 完全一致する契約のまま実機実測は `docs/perf/logs/adaptive-max-global-pool-2160/README.md` へ申し送り） |
| 損失（BCE／NLL／Huber／KLDiv） | #1609（#1737 で BCE／BCEWithLogits 実装済み: `Var::bce_loss`／`bce_with_logits_loss`・`nn::loss::BceLoss`／`BceWithLogitsLoss`・3 バックエンド融合カーネル。facade 到達経路は既存 `Var` 再エクスポート経由〈新規 `pub use`／`pub fn` なし〉。CUDA／Metal 実機 parity は未実測のまま Mac／GB10 セッションへ申し送り → Metal〈M4 Max〉は 2026-09-16 に実測済み（pass。`docs/perf/logs/metal-realdevice-phase2-2026-09-16/README.md`）・CUDA〈GB10〉は引き続き未実測 → CUDA〈GB10〉も 2026-09-16 に実測済み（pass。`docs/perf/logs/cuda-realdevice-phase2-2026-09-16/README.md`）。#1739 で Huber／SmoothL1 実装済み: `Var::huber_loss`／`smooth_l1_loss`・`nn::loss::HuberLoss`／`SmoothL1Loss`。`BackendOps::huber_loss`／`huber_loss_backward`〈`HuberKind::{Huber, SmoothL1}`。既定 `Unsupported`〉・CPU／CUDA／Metal 3 バックエンド専用融合カーネル〈`MseLoss` と同型の 2 段 reduction・`dTarget = −dPred` 契約〉・VJP は `grad::huber_loss_vjp`〈`Unsupported` のときのみ〉。#1738 で NLLLoss／KLDivLoss 実装済み: `Var::nll_loss`〈`tape::Op::NllLoss`。`targets` は `Op::CrossEntropyLoss` と同型の非追跡 `Tensor<i32>`〉・`Var::kl_div_loss`／`kl_div_loss_with_log_target`〈`tape::Op::KlDivLoss`。`input`／`target` とも追跡対象・`tensor-core::KlDivTarget` で `Probabilities`／`LogProbabilities` を分岐〉・`BackendOps::nll_loss`／`nll_loss_backward`・`kl_div_loss`／`kl_div_loss_backward`〈`MseReduction` を共用・既定 `Unsupported`〉・CPU／CUDA／Metal 3 バックエンド融合カーネル実装済み・`nn::loss::NllLoss`／`KlDivLoss`（薄いラッパー）。`log_softmax(x).nll_loss(t) ≡ cross_entropy_loss(x, t)` の forward／backward 一致を統合テストで確認済み。facade 到達経路はいずれも既存 `Var`／`nn` 再エクスポート経由・新規 `pub use`／`pub fn` は facade へ追加していない。CUDA／Metal 実機での facade parity は未実測のまま Mac／GB10 セッションへ申し送り → Metal〈M4 Max〉は 2026-09-16 に実測済み（pass。`docs/perf/logs/metal-realdevice-phase2-2026-09-16/README.md`）・CUDA〈GB10〉は引き続き未実測 → CUDA〈GB10〉も 2026-09-16 に実測済み（pass。Huber は #1739 分も含め `docs/perf/logs/cuda-realdevice-phase2-2026-09-16/README.md`）。#2166 で L1 損失（`nn.L1Loss` 相当）・CrossEntropy の label_smoothing／ignore_index／class_weight（`nn.CrossEntropyLoss` の同名オプション相当）を内部クレート限定で実装済み: `fandhe_ai_autodiff::loss_ops::l1_loss`／`cross_entropy_loss_with`（自由関数。**#2538 で `Var::l1_loss`／`cross_entropy_loss_with` を委譲メソッドとして facade 公開済み**〈`LossOpsHoldDoctestGuard` は部分反転・`api_surface.rs` に正ガード追加〉。`loss_ops` モジュール・`CrossEntropyOptions` の再エクスポートは未承認）・`nn::loss::L1Loss`（新規）／`CrossEntropyLoss::forward_with`（追加メソッド。facade は `nn::loss` を再エクスポートしないため到達しない）。`cross_entropy_loss_with` は既定オプション時に既存 `Var::cross_entropy_loss` へ丸ごと委譲するため既存経路は不変（R3）。CUDA／Metal 実機は未実測のまま `docs/perf/logs/loss-ops-2166/README.md` へ申し送り。詳細は `docs/autodiff-loss-ops-decision.md`。#2167 で CosineEmbedding／MarginRanking／TripletMargin／PoissonNLL を同じ内部クレート限定パターンで実装済み: `fandhe_ai_autodiff::loss_ops::{cosine_embedding_loss, margin_ranking_loss, triplet_margin_loss, poisson_nll_loss}`（自由関数。`Var` への委譲メソッドは #2539 で facade 公開済み——`LossOpsHoldDoctestGuard` は 8 行削除の部分反転）・`nn::loss::{CosineEmbeddingLoss, MarginRankingLoss, TripletMarginLoss, PoissonNllLoss}`（新規。facade は `nn::loss` を再エクスポートしないため到達しない）。境界規約（hinge・`min` 同値・p ノルム 0 の劣勾配）は ATen ソース未確認のまま類推で決定（実機突合セッションへ申し送り）。CUDA／Metal 実機は未実測のまま `docs/perf/logs/loss-ops-2167/README.md` へ申し送り。詳細は `docs/autodiff-distance-poisson-loss-ops-decision.md`。#2168 で CTC 損失を同じ内部クレート限定パターンで実装済み: `fandhe_ai_autodiff::loss_ops::ctc_loss`（自由関数。`CtcLossOptions`: blank／zero_infinity。`Var` への委譲メソッド・facade 到達は未承認のため保留——`LossOpsHoldDoctestGuard`〈7 関数へ拡張〉＋`api_surface.rs` の 4 テストで機械的に固定）・`nn::loss::CtcLoss`（新規。facade は `nn::loss` を再エクスポートしないため到達しない）。CPU 上の `f64` 対数空間 DP（forward: 前向き α のみ・VJP: 後ろ向き β を現フレーム非含有規約で計算）のみで、CUDA／Metal からはホスト計算フォールバックで到達し実機未実測（`docs/perf/logs/ctc-loss-2168/README.md` へ申し送り）。詳細は `docs/autodiff-ctc-design.md`。**イシュー #2169（親 #2131）で `compat::Sequential::compile()` の `Loss` enum へ BCE／BCEWithLogits／NLL／KLDiv／Huber／SmoothL1／L1 の 7 variant を追加する設計を検討したが、イシュー本文の承認事項節が facade 公開面の拡張（§5 経路 2）を明記しているため未承認のまま保留した**（コード変更は `CompileLossVariantsHoldDoctestGuard`＋`api_surface.rs` の否定ガードのみ。`Loss` enum・`FitTarget::loss_for` 本体は不変。詳細は `docs/facade-compile-loss-variants-decision.md`） |
| optimizer（Adam／RMSprop／Adagrad／LAMB） | #1610（#1742 で Adam〈coupled L2 weight decay〉実装済み。`fandhe_ai_autodiff::nn::optim::adam` モジュール〈`AdamW` を鏡写しにした別実装・decay を勾配へ加算する分岐のみが差分〉。facade は `pub use fandhe_ai_autodiff::nn::optim::{Adam, AdamConfig};` の 1 行追加のみ〈`optim.rs`〉。`weight_decay==0` で `AdamW` と bit 完全一致・`weight_decay>0` は `torch.optim.Adam` の `_single_tensor_adam` 定義に基づく恒等式で検証（`crates/autodiff/tests/nn_optim_adam.rs`）。新規 `Op`／`BackendOps`／`Var` メソッド／VJP は追加していない。`DeviceParamStore` への結線は #1959 で完了済み（`DeviceParamStore::step_adam`／`step_adamw`・`BackendOps::adam_step_device(_tracked)`・`facade::Tape::step_device_param_store_adam`／`_adamw`。CPU 実装のみ・CUDA／Metal は `Unsupported` のまま。CPU 実装はホスト参照実装〈`Adam::step`／`AdamW::step`〉と bit 完全一致・`crates/backend-cpu/tests/adam_device_parity.rs`・`crates/facade/tests/device_param_store_adam_train.rs`。`docs/device-resident-update-design.md`「Adam／AdamW の常駐 step 結線」節）。RMSprop／Adagrad は #1743 で実装済み: `RmsProp`／`RmsPropConfig`・`Adagrad`／`AdagradConfig`〈`crates/autodiff/src/nn/optim/{rmsprop,adagrad}.rs`。`AdamW`〈#194〉を鏡写しにした別実装〉。`Tape`／`Var`／`BackendOps` に一切依存しない値型・純関数のため新規 `Op`／`BackendOps` メソッド／VJP なし。正しさは実 PyTorch 2.14.0+cpu 実行値 fixture との統一複合判定〈`.claude/rules/coding-rust.md` 既存 tolerance〉・閉形式（t=1）一致・決定性（bit 完全一致）で検証（`tests/nn_optim_{rmsprop,adagrad}.rs`）。facade は `fandhe_ai::optim::{RmsProp, RmsPropConfig, Adagrad, AdagradConfig}` の素の再エクスポートのみ（`docs/facade-optimizer-promotion-decision.md` §4 案 A）。**`crate::DeviceParamStore` はイシュー #2175 で結線済み**（`DeviceParamStore::step_rmsprop`／`step_adagrad`・`BackendOps::rmsprop_step_device`／`adagrad_step_device`（`_tracked` 版込み）・`facade::Tape::step_device_param_store_rmsprop`／`_adagrad`。CPU 実装のみ・CUDA／Metal は `Unsupported` のまま。CPU 実装はホスト参照実装と bit 完全一致・`crates/backend-cpu/tests/{rmsprop,adagrad}_device_parity.rs`・`crates/facade/tests/device_param_store_{rmsprop,adagrad}_train.rs`。`docs/device-resident-update-design.md`「RmsProp／Adagrad／LAMB の常駐 step 結線（#2175）」節）。#1744 で LAMB（layer-wise trust ratio。You et al., 2019）実装済み。`fandhe_ai_autodiff::nn::optim::lamb` モジュール（`AdamW` と同一の moment 更新演算列を使う独立実装。weight decay は paper 定義どおり更新方向 `u` へ coupled で織り込み、`AdamW` の decoupled 乗算減衰とは構造が異なる）。facade は `pub use fandhe_ai_autodiff::nn::optim::{Lamb, LambConfig};` の 1 行追加のみ。独立 f64 参照実装（paper Algorithm 2 そのまま）との統一複合判定突合・解析的恒等式 3 件（zero-grad の wd 非依存性・`wd=0` での `AdamW` 換算・2 の冪スケール不変性は bit 完全一致）で検証（`crates/autodiff/tests/nn_optim_lamb.rs`）。新規 `Op`／`Var` メソッド／VJP は追加していない・**`DeviceParamStore` はイシュー #2175 で結線済み**（`DeviceParamStore::step_lamb`・`BackendOps::lamb_step_device(_tracked)`・`facade::Tape::step_device_param_store_lamb`。layer-wise trust ratio は連結バッファの `segment_numels` で表現。CPU 実装のみ・CUDA／Metal は `Unsupported` のまま。非有限検出時は no-op 失敗・no-poison。`crates/backend-cpu/tests/lamb_device_parity.rs`・`crates/facade/tests/device_param_store_lamb_train.rs`）。**#2170 で `compat::Optimizer` enum（Keras 風 `compile()`／`fit()` 経路。1.2 節）へ `RmsProp`／`Adagrad`／`Lamb` variant を追加**（既存の再エクスポート済み `*Config` 型を保持する `#[non_exhaustive]` enum への variant 追加のみで facade 新規公開面は 3 variant に限る。`compat::training::OptimizerState`〈非公開〉が `new`／`lr`／`step` を各 `RmsProp`／`Adagrad`／`Lamb::new`／`config().lr`／`step` へ結線。新規 `Op`／`BackendOps`／VJP／カーネルなし。正しさは手動ループとの bit 完全一致で検証〈統合テスト `crates/facade/tests/compat_sequential_fit_optimizers.rs`〉）。**LR スケジューラ非対応**: 3 者とも `set_lr` を持たない値型のため、`Callback::LrSchedule` と組み合わせた `fit_with_callbacks` 呼び出しは `InvalidArgument`（fail-closed）。`set_lr` 追加（facade 公開面拡張）はユーザー承認事項として本イシューでは実施していない（`docs/compat-callbacks-design.md` §2 参照）。イシュー #2171（親 #2131）で Adadelta（Zeiler, 2012）・Adamax（Kingma & Ba, 2015 §7.1）・NAdam（Dozat, 2016）・RAdam（Liu et al., 2019）を実装済み: `Adadelta`／`AdadeltaConfig`・`Adamax`／`AdamaxConfig`・`NAdam`／`NAdamConfig`・`RAdam`／`RAdamConfig`（`crates/autodiff/src/nn/optim/{adadelta,adamax,nadam,radam}.rs`。`AdamW`〈#194〉を鏡写しにした別実装）。`Tape`／`Var`／`BackendOps` に一切依存しない値型・純関数のため新規 `Op`／`BackendOps` メソッド／VJP なし。正しさは実 PyTorch 2.14.0+cpu 実行値 fixture との統一複合判定・閉形式（t=1）一致・決定性（bit 完全一致）で検証（`tests/nn_optim_{adadelta,adamax,nadam,radam}.rs`）。**facade（`fandhe_ai::optim`）へはイシュー #2501 で素の再エクスポートとして公開済み**（`crates/facade/src/optim.rs`。保留ガード `OptimizerExtHoldDoctestGuard` は削除し、`api_surface.rs` のソース走査を承認形のみ許す正ガードへ反転。承認形は `docs/autodiff-optimizer-adadelta-adamax-nadam-radam-decision.md` §8・§8.1 参照）。`crate::DeviceParamStore` は非対応。**#2173 で param groups（層別学習率・weight decay。PyTorch `torch.optim.Optimizer.param_groups` 相当）を実装済み**（`fandhe_ai_autodiff::nn::optim::{ParamGroup, ParamGroupStep}`〈`crates/autodiff/src/nn/optim/param_group.rs`〉。上記 6 optimizer〈Adam／AdamW／RmsProp／Adagrad／Lamb／`crate::optim::Sgd`〉全てへ `ParamGroupStep` を実装し、既存 `step()` はスロット単位の `lr`／`weight_decay` を受け取る `pub(crate) step_with_slot_hparams` への薄い委譲へ変更〈演算式の形は不変・`groups=&[]` で既存 `step()` と bit 完全一致〉。**#2298 で `Adadelta`／`Adamax`／`NAdam`／`RAdam`〈#2171〉へも横展開し、対象は計 10 optimizer になった**（`NAdam`／`RAdam` の `decoupled_weight_decay` はグループ上書き対象外のまま。`docs/autodiff-param-groups-decision.md` §8）。**facade へはイシュー #2553（ルート #2499 のコメントによる #2551 の承認〈設計判断記録 §9 の推奨案〉に基づく）で公開済み**——`fandhe_ai::optim::{ParamGroup, ParamGroupStep}` の素の再エクスポートと `compat::Sequential::compile_with_param_groups(Optimizer, Loss, &[ParamGroup])`（既存 `compile()` は不変）。保留ガード `ParamGroupsHoldDoctestGuard` は削除し、`api_surface.rs` を承認形のみを許す正ガード（`facade_param_groups_public_surface_matches_approved_contract`。#2553 時点の名は `..._approved_form`）へ反転した。`Lbfgs` との併用・`LrSchedule` との併用・groups 非空の `save_model` は fail-closed 拒否。実装記録は `docs/autodiff-param-groups-decision.md` §12、本 doc §5 の適用記録を参照。新規 `Op`／`BackendOps`／VJP／カーネルは追加していない・`DeviceParamStore` 常駐経路の group 対応は対象外。詳細は `docs/autodiff-param-groups-decision.md`。**#2197 で L-BFGS（closure・strong Wolfe line search。`torch.optim.LBFGS` 相当）を実装済み**（`fandhe_ai_autodiff::nn::optim::{Lbfgs, LbfgsConfig, LbfgsLineSearch}`〈`crates/autodiff/src/nn/optim/lbfgs.rs`〉。呼び出し元の closure を optimizer が内部で複数回評価する形のため既存 6 optimizer とは API 形状が異なる。**2026-09-27 所有者承認（#2172 コメント）で `compat::Optimizer::Lbfgs(LbfgsConfig)` variant・`LbfgsConfig` のみの facade 再エクスポート・`compile()`/`fit()` 統合を実装済み**（`compat::training::OptimizerState::Lbfgs`〈非公開〉が `new`／`lr`／`set_lr`／`step`〈closure 経由のため到達しない防御的 `InvalidArgument`〉へ結線し、`Sequential::run_fit` は専用ヘルパー `lbfgs_batch_step` へ分岐する。`compile_with_amp`・`accumulate_steps > 1`・カスタム学習 step フックとの併用は fail-closed 拒否。学習曲線の受け入れ検証は `crates/facade/tests/compat_sequential_fit_lbfgs.rs`）。**`Lbfgs`（optimizer 本体）・`LbfgsLineSearch`（line search 方式選択）は #2502（ルート #2499 本文「承認範囲」節の一括承認）で facade 公開済み**（`fandhe_ai::optim::{Lbfgs, LbfgsConfig, LbfgsLineSearch}`。保留ガード `LbfgsHoldDoctestGuard` は撤去し `api_surface.rs` の期待集合で承認形のみを許す正ガードへ反転。詳細は `docs/autodiff-lbfgs-decision.md` §7・§9、本 doc §5 参照）。**#2179 で EMA（指数移動平均。`torch.optim.swa_utils.AveragedModel`／Keras 3 `EMAOverlay` 相当）を実装済み**（`fandhe_ai_autodiff::nn::ExponentialMovingAverage`〈`crates/autodiff/src/nn/ema.rs`〉。**内部クレート限定**——facade（`fandhe_ai::optim` 再エクスポート・`compat::FitConfig` の `use_ema`／`ema_decay` 相当追加・`fit()` 統合）は未承認のため保留（`crates/facade/src/lib.rs::EmaHoldDoctestGuard`＋`api_surface.rs` の否定ガード 3 種で機械固定。詳細は `docs/autodiff-ema-decision.md` §4・§5） |
| scheduler（Cosine／Exponential／Plateau／OneCycle） | #1611（#1745 で式ベース 3 種を実装済み: `CosineAnnealingLr`／`ExponentialLr`／`LinearWarmupLr`。いずれも `LrScheduler::lr_at(step) -> f32` のみを持つ stateless 純関数で `Op`／`BackendOps`／`Var` を新規拡張しない。`CosineAnnealingLr` は PyTorch `CosineAnnealingLR._get_closed_form_lr` 準拠の周期的閉形式〈`step>t_max` で clamp しない〉、`LinearWarmupLr` は PyTorch に同名クラスがないため `LinearLR` の `end_factor=1.0` 固定形として定義。facade は `crates/facade/src/optim.rs` への `pub use` 1 行追加のみ。Plateau 分は #1746 で実装済み: `ReduceLrOnPlateau`／`ReduceLrOnPlateauConfig`／`PlateauMode`／`ThresholdMode`〈`fandhe_ai_autodiff::nn::optim::reduce_lr_on_plateau`。PyTorch `torch.optim.lr_scheduler.ReduceLROnPlateau` 準拠〉。既存 `ConstantLr`／`StepLr`〈stateless 純関数〉とは異なり検証指標に応じて内部状態〈patience／best／cooldown〉を進める唯一の例外であり、`LrScheduler::lr_at` は状態を進めず現在値を返すだけで、状態を進める入口は `ReduceLrOnPlateau::step(metric)` に限定〈`nn::optim::reduce_lr_on_plateau` モジュール doc〉。`Tape`／`Var`／`BackendOps` に一切依存しない値型・純関数のため新規 `Op`／`BackendOps`／`Var` メソッド／VJP は追加していない。facade は `fandhe_ai::optim::{ReduceLrOnPlateau, ReduceLrOnPlateauConfig, PlateauMode, ThresholdMode}` の素の再エクスポートのみ〈`optim.rs`〉。#1747 で `OneCycleLr`／`OneCycleLrConfig`／`OneCycleAnneal`〈PyTorch `OneCycleLR` 相当〉を実装済み化: `new` 構築時にフェーズ境界（2 フェーズ／3 フェーズ）を事前計算して保持することで `lr_at` 自体は参照のみの stateless 純関数として実装〈内部可変状態を持たない〉。momentum cycling（`cycle_momentum` 等）は対象外。`step >= total_steps` は PyTorch の `ValueError` と異なり `total_steps - 1` へ clamp する（`lr_at` が `Result` を返せない契約のため）。`Op`／`BackendOps`／`Var` 新規拡張なし・facade は `pub use` 1 行追加のみ。これにより状態保持型（Plateau／OneCycle）も式ベース型（Cosine／Exponential／LinearWarmup）も出揃い、本行の対象外事項はなくなった。#2176 で `MultiStepLr`／`CosineAnnealingWarmRestarts`／`CyclicLr`／`LambdaLr`／`SequentialLr` の 5 種を追加実装済み〈`fandhe_ai_autodiff::nn::optim::lr_scheduler`。いずれも式ベース・stateless 純関数で `Op`／`BackendOps`／`Var` を新規拡張しない〉。`CosineAnnealingWarmRestarts` の周期位置決定は整数演算（`checked_add`／`checked_mul`）で行い PyTorch の浮動小数 `log` 由来の境界誤判定を再現しない意図的な逸脱を含む。**facade（`fandhe_ai::optim`）へは #2503 で 5 種とも純再エクスポート済み**（ルート #2499 の一括承認。保留ガードは承認形のみを許す正ガード `facade_reexports_lr_scheduler_ext_items_only_in_approved_shape` へ反転。詳細は `docs/autodiff-lr-scheduler-ext-decision.md` §8.1） |
| autograd 制御（no_grad／detach／retain_graph） | #1612（#1748 で no_grad／detach 実装済み: `Tape::var_no_grad`〈追跡なし葉。`requires_grad=false` の `Op::Leaf`。facade は `Tape::var_no_grad` ラッパーを 1 メソッド追加〉・`Var::detach`〈既存の `requires_grad=false` 葉ノード機構を再利用する設計・専用 `Op` variant は追加していない・facade は既存 `Var` 再エクスポート経由で新規公開面なし〉。`TapeNode::requires_grad`〈bool〉を前方伝播〈`Op::for_each_input` で入力の論理和〉し、`Tape::backward` は起点 loss が `requires_grad=false` なら `Err(AutodiffError::Backward)`・逆走査中は `requires_grad=false` ノードへの寄与を `accumulate` へ渡さず破棄する。`Gradients::get` は対象ノードが `requires_grad=false` なら新設 `AutodiffError::GradientTrackingDisabled`〈`#[non_exhaustive]` への非破壊 variant 追加〉を返し「loss から未到達」の `Ok(None)` と型で区別する。算術を伴わない機構のため CPU 本番 ops と naive 参照実装の勾配が bit 完全一致することをテストで確認済み。retain_graph（複数回 backward の勾配蓄積契約）は #1749 で実装済み（retain_graph 自体は追加 API なしで常時成立する契約〈`Tape` は `reset`／drop までグラフを保持し続けるため複数回 `backward` が無条件で成功する〉と確定・`Tape::backward_accumulate`〈facade 到達経路も追加。`grad::vjp_elementwise_add` への委譲のみで新規 `Op`／`BackendOps` なし〉を新設。設計は `docs/autodiff-retain-graph-accumulate-decision.md`）。設計は `docs/autodiff-nograd-leaf-dinput-skip-decision.md` §5「案 B」） |
| cast | #1613（#1750 で dtype 変換基盤と CPU cast を実装済み: `CastDType`／`CastElement`〈sealed trait・`f32`／`f64`／`i32`／`i64`／`bool` の 5 型〉・`BackendOps::cast_ops` capability accessor・`CastOps`〈8 方向・既定 `Unsupported`〉・ホスト参照実装〈`tensor_core::cast::{cast_from_f32, cast_to_f32}`〉・CPU 実装〈`backend-cpu::cast`。ホスト参照実装への委譲〉・`Var::cast`／`Var::to_f32`／`Tape::var_from`。出力が非 f32 の cast は非微分・detached〈`Var::argmax`／`unique` と同型〉・facade は `CastDType`／`CastElement` の再エクスポートのみ〈`CastOps` は非公開のまま〉。設計・数値契約は `docs/tensor-core-cast-design.md`。#1751 で CUDA〈8 方向〉・Metal〈f64 2 方向を除く 6 方向。MSL `double` 非対応〉のネイティブカーネルを実装済み・facade 新規公開面なし・CUDA／Metal 実機実測は未実施のまま Mac／GB10 セッションへ申し送り → Metal〈M4 Max〉は 2026-09-16 に実測済み（pass。`docs/perf/logs/metal-realdevice-phase2-2026-09-16/README.md`）・CUDA〈GB10〉は引き続き未実測 → CUDA〈GB10〉も 2026-09-16 に実測済み（pass。`docs/perf/logs/cuda-realdevice-phase2-2026-09-16/README.md`）） |
| デバイス転送と列挙 | #1614（実装済み: `fandhe_ai::available_devices()`〈3 バックエンドの `DeviceProvider` を束ねた列挙入口。`Device::Cpu` → `Device::Cuda(0..n)` 昇順 → `Device::Metal`〈macOS のみ〉の決定的順序。`DeviceProvider`／`DeviceInfo`／`enumerate_all` は再エクスポートせず `Device` 識別子のみを返す〉・`Var::device`／`Var::to`〈同一デバイスなら恒等・不一致なら新設 `AutodiffError::DeviceMismatch { requested, actual }`〉・`Var::to_tape`〈別 `Tape` への値転送。同一 tape は恒等・それ以外は `materialize_fallible` 経由で実体化した値〈bit 完全一致・算術なし〉を新しい葉として登録し `requires_grad` を引き継ぐ・勾配はテープをまたがない・checkpoint 解放済み poison ノードは `Err` で fail-closed〉・facade `Tape::device`／`Tape::transfer`〈`Var::to_tape` の唯一の facade 到達経路〉。`Tensor` 側は独立の転送 API を設けず `tape_for(device)?.var(&t)` と等価という設計上の非対応として確定。新規 `Op`／`BackendOps`／VJP は追加していない。設計・実装記録は `docs/facade-device-transfer-enumeration-design.md`。CUDA／Metal 実機での facade parity テストは未実測のまま Mac／GB10 セッションへ申し送り → Metal〈M4 Max〉は 2026-09-16 に実測済み（pass。`docs/perf/logs/metal-realdevice-phase2-2026-09-16/README.md`）・CUDA〈GB10〉は引き続き未実測 → CUDA〈GB10〉も 2026-09-16 に実測済み（pass。`docs/perf/logs/cuda-realdevice-phase2-2026-09-16/README.md`）） |
| Dataset／DataLoader | #1615（実装済み: `data::Dataset`／`data::TensorDataset<T>`〈map-style。異種 dtype／複数列は 2/3 要素タプル impl で表現し同一シャッフル順を保証〉・`data::DataLoader`／`data::DataLoaderConfig`〈batch_size／shuffle／drop_last〉・`data::Batches`〈`ExactSizeIterator`〉・`data::DataError`。`crates/tensor-core/src/data.rs`（`rng`／`creation` と同じホスト側完結レイヤー。`Op`／`BackendOps`／`Var`／VJP を一切経由しない）。シャッフルは Fisher–Yates＋`rng::with_global_rng` の rejection sampling（`manual_seed` 契約下で決定的）。facade は `fandhe_ai::data::{Batches, DataError, DataLoader, DataLoaderConfig, Dataset, TensorDataset}` の純再エクスポートのみ（`crates/facade/src/data.rs`）。設計・受入基準テンプレート読み替えは `docs/dataset-dataloader-design.md`）。**#2182（親 #2131）で `Sampler`（`SequentialSampler`／`RandomSampler`／`WeightedRandomSampler`）・`SamplerDataLoader<D>`・`HookedDataLoader<T>`〈サンプル単位 `TransformFn`・collate 単位 `CollateFn`〉を実装済み**（既存 `DataLoader`／`DataLoaderConfig`・facade 6 型再エクスポートは不変のまま追加のみ。**当初は内部クレート限定だったが、#2505 で 11 名の純再エクスポートとして facade（`fandhe_ai::data`）へ公開済み**〈`DataHooksHoldDoctestGuard` は削除、`api_surface.rs` は承認形の正ガードへ反転。`DataLoader` への統合はしない〉。数値経路（`Op`／`BackendOps`／VJP）は一切追加していないため CUDA／Metal 実機 parity は対象外。詳細は `docs/tensor-core-data-sampler-hooks-decision.md`）。**#2183（親 #2131）で `PrefetchConfig`・`PrefetchDataLoader<D>`・`PrefetchBatches<D>`（マルチワーカー prefetch。`std::thread`／`std::sync::mpsc` ベース。`rayon` 追加は承認事項のため未実施）を実装済み**（既存の `DataLoader`／`Sampler` 系・facade 再エクスポートは不変のまま追加のみ。**当初は内部クレート限定だったが #2506 で 3 名を `fandhe_ai::data` から純再エクスポートとして facade 公開済み**（`api_surface.rs` は承認形の正ガード＋インベントリ。`Sequential::fit` への結線は保留）。出力は同一 `Sampler` を使う `SamplerDataLoader` と bit 完全一致〈決定性契約〉。数値経路は一切追加していないため CUDA／Metal 実機 parity は対象外。詳細は `docs/tensor-core-data-prefetch-decision.md`） |
| state_dict／safetensors | #1616（#1752 で state_dict 部分を実装済み: `nn::Module` trait に `set_parameter`〈defaulted・既定 `Err`。`Linear`／`RmsNorm`／`LayerNorm`／`MultiheadAttention`／`Rnn`／`Lstm`／`Gru`／`ModuleList`／`Sequential` で実装〉・`state_dict`／`load_state_dict`〈defaulted。`HashMap<String, Tensor<f32>>`。strict・two-pass アトミック〈パス 1 でキー集合完全一致・shape 完全一致を検証してから、全通過後のパス 2 で書き戻す。`compat::Sequential::apply_parameters` の #294／#426 と同型の不変条件〉〉を追加。`compat::Sequential::state_dict`／`load_state_dict` へ 1 行委譲する facade 新規公開面 2 件〈`docs/compat-api-scope.md` §5 経路 2。親 #1616 の 2026-09-12 ユーザー承認済み〉。新規 `Op`／`BackendOps`／VJP はいずれも該当なし〈数値経路を変更しない機構のため bit 完全一致契約。CUDA／Metal 実機 parity は数値経路非依存のため対象外〉。#1754 で `Tensor` の `Debug`／`Display`〈打ち切り付きの値プレビュー。`docs/public-api-design.md` §7 の `Tape: Debug` 公開契約越しの DoS 耐性を含む。軸ごとの打ち切りのみでは高階小軸長形状〈例 `shape=[2;20]`〉を打ち切れない穴・極端な rank でのスタックオーバーフローをコードレビュー指摘で発見し、総出力要素数のグローバル予算・rank 上限ガードを追加是正済み〈`tensor-core::tensor_fmt` モジュール doc 参照〉〉実装済み〈`tensor-core::tensor_fmt`。facade 新規公開面なし〉。**#2019 で safetensors save／load を実装済みへ更新**（案 A・素の再エクスポート。`fandhe_ai::interop::safetensors::{LoadError, SaveError, load_safetensors_f32, load_safetensors_f32_from_bytes, require_keys, save_safetensors_f32, save_safetensors_f32_to_bytes}`。`fandhe_ai_onnx_interop::st_load`／`st_save` からの純再エクスポート・ロジック複製なし・facade への `safetensors` 直接依存追加なし。`compat::Sequential::state_dict`／`load_state_dict` との bit 完全一致往復・不足キー／dtype 不一致／形状不一致の fail-closed 拒否を統合テストで検証済み。`ModelCheckpoint` からのファイル保存結線は引き続き対象外。詳細は `docs/facade-safetensors-exposure-decision.md` §11）。**#2087 でローカルモデルレジストリ `fandhe_ai::model::ModelRegistry` を実装済み**（詳細は `docs/facade-model-registry-decision.md`） |
| Module の train／eval | #1617（#1758 で `nn::Module` trait 契約〈`set_training`／`training`。既定 no-op／`true`。無状態モジュールはオーバーライドせず、モードの正はコンテナ〈`compat::Sequential`〉が保持するフラグとする契約〉と `named_parameters`〈`Vec<(String, &Tensor<f32>)>`。struct フィールド名／accessor 名ベースの命名契約・`Linear`／`RmsNorm`／`LayerNorm`／`MultiheadAttention`／`Rnn`／`Lstm`／`Gru` で実装〉を実装済み。`compat::Sequential` に `set_training`／`train`／`eval`／`training`／`named_parameters`〈index 接頭辞契約。`trainable_parameters()` と同一順序〉を追加（facade 新規 `pub fn` 5 件）。新規 `Op`／`BackendOps`／VJP／GPU カーネルはいずれも該当なし〈数値経路 bit 完全一致。CUDA／Metal 実機 parity は数値経路非依存のため対象外〉。`predict`／`forward_host` とモード〈Dropout 等の `training=False` 意味論〉の整合は #1603 で確定済み（コンテナの `training` フラグを尊重する方式。§1.2「Dropout」行参照）。コンテナ再構成は #1759 で実装済み: `fandhe_ai_autodiff::nn::container::{ModuleList, Sequential}`（PyTorch `nn.ModuleList`／`nn.Sequential` 相当。`Module` trait を非公開のため facade からは再エクスポートしない）を新設し、`compat::Sequential` の層保持・Linear→ReLU 融合先読み走査（forward）・`set_training`／`training`／`named_parameters` の本体を移設。`compat::Sequential` は `inner: nn::Sequential` を持つ薄いラッパーへ再構成（`forward` は `self.inner.forward(&tape.0, input)` へ 1 行委譲・`set_training`／`training`／`named_parameters` も `self.inner` へ委譲）。公開シグネチャ・数値挙動は不変（既存テスト 30 件 bit 完全一致確認済み）・facade 新規公開面なし。ネストしたコンテナ内 `Linear` は compat の学習契約〈`bind`／`trainable_parameters`／`apply_parameters`・デバイス常駐経路〉に到達しない制限が残るが、facade は `ModuleList`／`nn::Sequential` を構築する経路自体を公開していないため到達不能（`container.rs` モジュール doc「ネストの限界」参照）） |
| Keras 風 `Sequential` の層追加と `compile()`／`fit()`／`evaluate()`／callbacks の最小版 | #1618（#1761 で `compile()`／`fit()`／`evaluate()` 最小版実装済み。`compat::{Loss, Optimizer, FitConfig, History, FitTarget}`・`Sequential::{compile, is_compiled, fit, evaluate}`。既存公開 API〈`Sequential::bind`／`fandhe_ai::optim`／`fandhe_ai::data::DataLoader`／`Var::mse_loss`／`cross_entropy_loss`〉の合成のみで新規 `Op`／`BackendOps`／VJP なし・CPU `tape()` 固定。正しさは手動学習ループとのパラメータ・loss 系列 bit 完全一致で検証（統合テスト `crates/facade/tests/compat_sequential_fit.rs`）。#1763 で callbacks（`EarlyStopping`／`ModelCheckpoint`）・`validation_data`・LR スケジューラ連携を実装済み（`compat::{Callback, EarlyStopping, ModelCheckpoint, LrSchedule, Monitor, MonitorMode}`・`Sequential::fit_with_callbacks`・`fandhe_ai::optim::{Sgd, AdamW, Adam}::set_lr` 新設。新規 `Op`／`BackendOps`／VJP なし・正しさは手動ループとの bit 完全一致で検証（統合テスト `crates/facade/tests/compat_sequential_callbacks.rs`）。設計判断は `docs/compat-callbacks-design.md`。`DataLoader` 直接入力は引き続き対象外）。**#2073 で `ModelCheckpoint::to_file`（safetensors ファイル保存の薄い結線）を実装済み**（facade 新規 `pub fn` 1 件。§5「適用記録（経路 2。イシュー #2073）」参照）。**#1760 で `Sequential::add_*` の対象レイヤーを Conv2d／Conv1d〈#1770〉に加え LayerNorm／RmsNorm／BatchNorm1d／BatchNorm2d／Embedding／MultiheadAttention へ拡張**（詳細は本表の各該当行。§5「適用記録」参照）。**#2072 で分類 metrics（accuracy・precision・recall・F1・confusion matrix）を実装済み**（`compat::{Metrics, MetricsResult}`・`Sequential::fit_with_metrics`・`Monitor::ValMetric`。設計判断は `docs/compat-metrics-design.md`。§5「適用記録（経路 2。イシュー #2072）」参照）。**#2170 で `compile()` の `Optimizer` enum へ `RmsProp`／`Adagrad`／`Lamb` variant を追加**（詳細は本表「optimizer（Adam／RMSprop／Adagrad／LAMB）」行参照。LR スケジューラ〈`Callback::LrSchedule`〉は新 3 variant 非対応で fail-closed 拒否）。**イシュー #2169（親 #2131）で `compile()` の `Loss` enum へ 7 variant（BCE／BCEWithLogits／NLL／KLDiv／Huber／SmoothL1／L1）を追加する設計を検討したが、承認事項として明記されているため未承認のまま保留した**（詳細は本表「損失（BCE／NLL／Huber／KLDiv）」行参照。CTC は `ctc_loss` が長さテンソル 2 本を追加要求し `FitTarget::loss_for` の 2 引数契約に収まらないため対象外）。**イシュー #2178（親 #2131）で `Callback` へ `CsvLogger`／`JsonLogger`／`LambdaCallback` の 3 variant を追加する設計を検討し、#2178 時点では未承認のまま保留した（`CallbacksLoggersHoldDoctestGuard`＋`api_surface.rs` の 4 テストで固定）が、ルート #2499 の一括承認により #2571 で公開済み**（保留ガードは削除し承認形の正ガードへ反転済み。§5 適用記録〈イシュー #2571・#2572〉参照。詳細は `docs/compat-callbacks-loggers-decision.md`） |

**GroupNorm／InstanceNorm（#2066・親 #2058）**: 上記「LayerNorm／
RMSNorm／BatchNorm」行と異なり、GroupNorm／InstanceNorm は本 Tier 1
表（1 節）にも Tier 2（1.3 節）にも個別の行を持たない。それでも
`nn::GroupNorm`／`nn::InstanceNorm`（`crates/autodiff/src/nn/
normalization.rs`）を実装済みである——既存の最終軸限定 `Var::
layer_norm`（affine なし）を「軸削減 reshape → `layer_norm` → 逆
reshape」で呼ぶ合成のみで新規 `Op`／`BackendOps` を追加しないため、
本文書 §5 の範囲拡張手続き（新規カーネル・新規演算の追加）の対象外
と判断した。**PyTorch 既定と異なり affine（学習可能な per-channel
`weight`／`bias`）を持たない**（`docs/norm-ops-design.md` §11「affine
非対応」参照。`.claude/rules/coding-rust.md` の勾配長軸縮約契約との
抵触を避けるため別イシューへ見送り）。`Module` trait への統合
（`as_group_norm`／`as_instance_norm`）を行い、**facade 公開面
（`compat::Sequential::add_group_norm`／`add_instance_norm`）は
イシュー #2525 で公開済み**（§5「適用記録（経路 2。イシュー #2525）」）。
詳細は `docs/norm-ops-design.md` §11・§11.x。

**PixelShuffle／PixelUnshuffle（#2162・親 #2131）**: 上記「GroupNorm／
InstanceNorm」行と同じ理由で、本 Tier 1 表（1 節）にも Tier 2（1.3
節）にも個別の行を持たない。それでも `nn::PixelShuffle`／
`nn::PixelUnshuffle`（`crates/autodiff/src/nn/pixel_shuffle.rs`）を
実装済みである——既存の `Var::reshape`／`Var::permute`／
`Var::contiguous` の合成のみで構成し新規 `Op`／`BackendOps` を追加
しないため、本文書 §5 の範囲拡張手続き（新規カーネル・新規演算の
追加）の対象外と判断した。`Module` trait への統合（`forward`／
`forward_host`）は行ったが、**facade 公開面拡張
（`compat::Sequential::add_pixel_shuffle`／`add_pixel_unshuffle`・
`Var::pixel_shuffle`／`Var::pixel_unshuffle`）はイシュー #2526 で
公開済み**（§5「適用記録（経路 2。イシュー #2526）」。型の再エクスポート
は未承認のまま）。詳細は `docs/autodiff-pixel-shuffle-decision.md`。

### 1.3 Tier 2（長尾。対象範囲・未実装）

REQ-9 2026-09-12 追記（`04-requirements.md:232`）の列挙を、実装リポ
Phase 3（親 #1573）の各 issue へ対応付ける。

| Tier 2 機能 | 実装 issue |
|---|---|
| RNN／LSTM／GRU | #1619（#1647 で内部クレート `fandhe_ai_autodiff::nn` として実装済み。`docs/autodiff-rnn-cell-tape-design.md`）。**#1955 で facade 公開済み**（2026-09-17 ユーザー承認「選択肢 C」。`fandhe_ai::nn::rnn`〈`Rnn`／`Lstm`／`Gru`・`RnnSeqOutput`／`LstmSeqOutput`・`RnnCellVars`／`LstmCellVars`／`GruCellVars` の 8 型純再エクスポート〉＋`Tape::rnn_forward_seq`／`lstm_forward_seq`／`gru_forward_seq`〈生 `Tape` を引数に取る `forward_seq` への薄い委譲入口。`step_device_param_store` と同型の回避不能な橋渡し〉。`compat::Sequential::add_rnn`／`add_lstm`／`add_gru` は追加しない（多層の `add_stacked_rnn`／`add_stacked_lstm`／`add_stacked_gru` も同様に追加しない〈#2534 案 A・2026-10-07 承認。否定ガードで固定〉。`Var → Var` 平坦鎖と `forward_seq` の構造的不整合。`nn::rnn` モジュール doc「`Sequential::add_*` を設けない理由」参照）。`RnnCell`／`LstmCell`／`GruCell`・`Module` trait は facade 非到達のまま（意図した制限）。正しさは facade 経由と内部クレート直接呼び出しの CPU forward・backward bit 完全一致で検証済み（`crates/facade/tests/nn_rnn_facade_bit_identity.rs`）。CUDA／Metal 実機 facade parity は未実測のまま申し送り。§5 適用記録参照）。**#2164 で多層・双方向・層間 dropout（`RnnConfig`）を内部クレートに追加実装済み**（`nn::rnn_stacked::{RnnConfig, StackedRnn, StackedLstm, StackedGru}`。既存 8 型・facade 公開面には触れず新しい型で提供）。**#2535 で facade 公開済み**（ルート #2499 の一括承認。`RnnConfig`・`Stacked*` 6 型を `nn::rnn` へ再エクスポート＋`Tape::stacked_{rnn,lstm,gru}_forward_seq`。`Rnn`／`Lstm`／`Gru::with_config` は引き続き非公開。詳細は `docs/autodiff-rnn-stacked-config-decision.md` §9） |
| einsum | #1620（実装済み。`Var::einsum`。既存の `matmul`／`sum`／`permute`／`reshape`／`mul` への分解のみで新規カーネルは追加していないため `BackendOps` は非拡張。facade 到達経路は既存 `Var` 再エクスポート経由〈新規 `pub use`／`pub fn` なし〉。batch 添字を伴う縮約〈例 `"bij,bjk->bik"`〉は #2149（親 #2131）で rank≥3 `matmul`〈#1600〉への分解として内部クレート限定（`fandhe_ai_autodiff::einsum_batch::einsum_batched`。既存の `matmul`／`sum`／`permute`／`reshape` の合成のみ・新規 `Op` なし）で実装済み。**#2517 で `Var::einsum` 自体が batch 添字付き縮約を受理するよう拡張され facade 公開済み**（新しい公開名なし・非破壊。`docs/autodiff-einsum-batch-decision.md` §11・§5 適用記録参照）。`docs/compat-feature-gap.md` §2.6 追補参照） |
| 線形代数（inv／solve／det／qr／cholesky／svd） | #1621（実装済み。`Var::inv`／`solve`／`det`／`cholesky`／`qr`／`svd`／`matrix_norm`。rank-2 限定・CPU 実装先行・GPU は `Unsupported` フォールバック。`docs/autodiff-linalg-design.md`）。**#2150 で `eigh`／`slogdet`／`pinv`／`matrix_rank`／`lstsq` を内部クレート限定（`fandhe_ai_autodiff::linalg_ops`）で実装済み**（#2515 で `Var::eigh`／`slogdet`／`pinv`／`matrix_rank`／`lstsq` の委譲メソッドとして facade 公開済み。`docs/autodiff-linalg-ops-decision.md`） |
| 高階微分 | #1622（設計記録。`docs/autodiff-higher-order-grad-decision.md`）。#1941 で前提 issue（#1593／#1597／#1599／#1601／#1612）完了後の HEAD へ設計記録を更新し、主案 A-2（子テープ方式）の `create_graph` API 契約案・対象 Op 分類（`Op` enum 69 variant の対象／非対象／保留区分）を確定。#1942 で elementwise／sum 系 Op（`Leaf`・`Add`・`Mul`・`Relu`・`Exp`・`Tanh`・`Sigmoid`・`Sum`・`Mean`・`Reshape`・`BroadcastTo` の 11 variant）を対象に `Tape::backward_create_graph`（内部クレート `fandhe_ai_autodiff` 限定。`Op::supports_create_graph()` 網羅 match で対象 Op を判定）を実装し、有限差分突合・閉形式突合で正しさを検証済み（`crates/autodiff/tests/create_graph.rs`）。facade 公開（同 doc §10 承認事項 5）は未承認のまま不実施。#1943 で `MatMul`（**rank 2 × rank 2 限定**。rank≥3 は `validate_ancestors` が入口で型付き `Err` 事前拒否）を対象へ追加し（累計 12 variant）、`nn::Linear`（`MatMul`→`Add` bias）経由の 2 階微分・小型 MLP（`Linear`→`tanh`→`Linear`）の HVP（ヘッセ・ベクトル積。有限差分の方向微分と突合）を検証済み。resident グラフ（`DeviceParamStore`）は素の `Tape::backward` より前に構造的検査で型付き `Err(Backward)` を返す（誤誘導的な `InvalidArgument` を避ける）よう順序を調整。`ScalarUnary`／`ScalarBinary`・`Transpose` 等の残る対象 Op は後続イシューへ引き継ぎ。`Op::Custom`（#1946）は `create_graph` の対象外——`Op::supports_create_graph()` が `false` を返し `Tape::backward_create_graph` は型付き `Err(Backward)` で fail-closed に拒否する（`docs/autodiff-higher-order-grad-decision.md` §8／`docs/autodiff-custom-function-decision.md` §14 参照）。CUDA／Metal 実機実測は未実施のまま Mac／GB10 セッションへ申し送り。#2063: 承認事項 5 未承認のため facade 未公開のまま。`api_surface.rs` の否定ガード 2 件（`facade_does_not_reexport_create_graph_result`・`facade_tape_does_not_expose_backward_create_graph_method`）で機械固定。**#2545 で §17.2 の確定形どおり facade 公開済み**（`fandhe_ai::CreateGraphResult`・facade `Tape::backward_create_graph`。否定ガードは承認形の正ガードへ反転。記録は `docs/autodiff-higher-order-grad-decision.md` §17〜§19） |
| custom autograd Function | #1623（設計記録。`docs/autodiff-custom-function-decision.md`）。#1945 で trait 境界（`CustomFunction`。案 B）を HEAD 基準で確定済み（同 doc §12）。**#1946 で内部クレート限定（§12.5 (a)）実装済み**（`fandhe_ai_autodiff::{CustomFunction, Tape::custom}`。`Op::Custom` variant・`grad.rs` の VJP 腕 1 つのみ追加・既存 backward は 1 行も変更なし。自作 `CustomRelu` と組み込み `Var::relu` の勾配 bit 完全一致で検証済み〈同 doc §13〉。facade 公開面（(b)。`fandhe_ai::CustomFunction` 再エクスポート・facade `Tape::custom` 転送）は未承認のまま対象外——`crates/facade/tests/api_surface.rs` の否定ガード 2 件で機械固定。§14 に `create_graph` との関係〈`Op::Custom` は `supports_create_graph()=false` で非対応〉を追記）。**#2064 の実装着手時（2026-09-22）にも本イシュー・親 #2059 のいずれにもリポジトリ所有者の明示的な承認コメントが確認できなかったため未承認のまま保留**——`facade/src/**`・`autodiff/src/**` は変更せず、既存否定ガード 2 件に加え AC-4 否定ガード 4 件（`facade_public_functions_do_not_take_custom_function`・`compat_sequential_does_not_expose_custom_add_method`・`custom_function_trait_signatures_are_host_tensor_only`・`autodiff_src_does_not_declare_pub_fn_custom_on_var`）を追加して機械固定した。承認取得後の実施範囲は同 doc §15 参照。**#2549 で §16.1 の確定形どおり facade 公開済み**（`fandhe_ai::CustomFunction`・facade `Tape::custom`。否定ガードは承認形の正ガードへ反転。記録は同 doc §16・§17） |
| activation checkpointing | #1624（実装済み。`Var::checkpoint_from`／内部クレート `Tape::checkpoint`。対象 Op は `MatMul`／`Sigmoid`／`Sum`／`Max`・view 系〈`Reshape`／`Transpose`〉限定〈`Op::is_checkpoint_eligible()`〉。facade `Tape` passthrough は承認未取得のため未追加——facade からは既存の `Var` 再エクスポート経由〈`Var::checkpoint_from`〉で到達可能。`docs/autodiff-checkpoint-design.md`） |
| AMP | #1625（#1721 でコア関数を実装・#1722 で facade 公開済み。`fandhe_ai::optim::{GradScaler, GradScalerConfig, UnscaleResult, scale_loss, scale_grads, unscale_grads, has_non_finite}` の純再エクスポート〈案 A〉。ホスト `Tensor<f32>` 勾配限定・デバイス常駐更新経路〈`DeviceParamStore`〉への unscale／非有限検出は未結線。承認記録は #1625 コメント〈2026-09-12〉。**#2181 でデバイス常駐更新経路への結線を実装済み**（`fandhe_ai_autodiff::optim::device_store::amp`〈`DeviceParamStore::step_amp`／`step_adam_amp`／`step_adamw_amp`〉＋ facade `Tape::step_device_param_store_amp`／`_adam_amp`／`_adamw_amp` の薄い委譲 3 件。SGD／Adam／AdamW 限定。既存 `Tape::param_grads_to_host`〈#1479〉でスケール済み勾配をホストへ実体化してから unscale・非有限検出する〈CUDA／Metal 専用のデバイス側 unscale カーネルは持たないホスト計算フォールバック〉。新規 `BackendOps`／`MemoryOps` trait メソッドは追加していない。§5 適用記録参照）。#1961 で `compat::Sequential::compile_with_amp`〈Linear 限定低精度 forward〈#1960〉＋`GradScaler` を `fit` から opt-in〉を実装し、真の混合精度〈f16／bf16 forward・f32 master weight・backward は常に f32〉を統合済み——`docs/backend-dtype-dispatch-design.md` §8 の「`Var`／`Tape` dtype 一般化」自体は依然として対象外だが、Linear 層限定の低精度 forward という部分的な適用は本 issue で解消した。facade 新規公開面は `compat::{AmpConfig, AmpDType}`・`Sequential::compile_with_amp`／`amp_loss_scale`（`ScalarDType` 自体は非公開のまま・`AmpDType` が facade ローカルの薄い写像）。MNIST 規模〈784→256→10・batch 64・20 step〉での f32 比 REQ-2 統一複合判定は F16／Bf16 とも `fail_count=0` で達成。`docs/autodiff-low-precision-linear-design.md` §7・`docs/compat-fit-evaluate-design.md` §4 参照）。**#2071 で Conv2d・MultiheadAttention へ拡張**（同じ narrow opt-in パターン。facade 新規公開面なし——`compile_with_amp` の対象層が Linear／Conv2d／MHA の 3 種になっただけで facade 公開 API 自体は不変。`docs/autodiff-low-precision-linear-design.md` §8 参照） |
| f64／f16／bf16 演算 | #1626（設計記録。`docs/backend-dtype-dispatch-design.md`）。#1939 で `TypedOps<f64>`／`<f16>`／`<bf16>`（8 演算限定〈`gemm`／`add`／`mul`／`relu`／`exp`／`tanh`／`sum`／`max`〉）の facade 公開面を実装済み（`fandhe_ai::{Scalar, ScalarDType, TypedOps}` の純再エクスポート＋`Tape::typed_ops_f64`／`_f16`／`_bf16`。autograd 非経由〈勾配なし〉・`half` 自体は facade が再エクスポートしない〈利用者は `fandhe_ai_tensor_core` 経由で直接名指し〉。`Var`／`Tape` の dtype 一般化・8 演算以外は引き続き対象外。同 doc §16 参照）。f64 の独立自動微分グラフ（`TapeF64`／`VarF64`／`GradientsF64`）は #2599 で facade newtype として公開済み（§5 の適用記録 #2599） |
| **量子化** | #1627（除外事項「分散学習・量子化の網羅対応」〈Won't・条件付き〉に従属。実装着手は同除外事項の格上げ条件充足と Phase 4 要件見直しでの新 REQ 追加のユーザー承認まで不可。5 節参照） |
| **複数 GPU／DDP** | #1628（同上に従属。設計判断の記録〈docs のみ〉に留め、実装・通信層の依存追加は行わない。5 節参照） |
| ONNX import 公開／export | #1629（#1652 で import 側の設計判断を記録・#1775 で export 側の設計判断を記録。案 B〈薄いラッパー型〉採用。#1963 で publish 前提〈`onnx-interop` の crates.io 公開〉のユーザー承認を取得し、**#2017 で import 側**・**#2018 で export 側**（いずれも `fandhe_ai::interop::onnx::{OnnxModel, OnnxValue, OnnxError, OnnxExportOptions}`。`OnnxModel::{from_bytes, from_path, run, to_bytes, to_path}`）を実装済み〈`docs/facade-onnx-import-exposure-decision.md` §12・`docs/facade-onnx-export-exposure-decision.md` §14〉。export は import 済みモデルの roundtrip 限定・facade 新規公開面は `OnnxExportOptions`・`to_bytes`・`to_path` の 3 件のみ・新規 `Op`／`BackendOps`／VJP なし。**#2037 で `OnnxModel::from_sequential(&Sequential)` を実装済み**（Linear／ReLU 限定。`docs/facade-onnx-export-exposure-decision.md` §17）） |
| topk／sort／cumsum | #1733 で sort／argsort／topk 実装済み（`Var::sort`／`argsort`／`topk`。CPU 参照実装〈`backend-cpu::sort_topk`〉・scatter ベース VJP〈`Op::Sort`／`Op::Topk`〉・facade 新規公開面なし〈既存 `Var` 再エクスポート経由〉）。#1741 で CUDA／Metal カーネル実装済み（64bit 合成キー〈`hi`＝正規化済み値キー・`lo`＝ライン内元添字〉によるビットニックソート方式。GPU 非安定ソートでも `values`／`index` が CPU 参照実装と bit 完全一致する契約〈`sort_model.rs` の鍵設計。CUDA・Metal で意図的複製・ホストモデルによる Linux 実行可能なアルゴリズム検証を実施済み〉。facade 新規公開面なし。CUDA・Metal とも実機実測は未実施のまま GB10／Mac セッションへ申し送り → Metal〈M4 Max〉は 2026-09-16 に実測済み（pass。`docs/perf/logs/metal-realdevice-phase2-2026-09-16/README.md`）・CUDA〈GB10〉は引き続き未実測 → CUDA〈GB10〉も 2026-09-16 に実測済み（pass。`docs/perf/logs/cuda-realdevice-phase2-2026-09-16/README.md`））。#1731 で cumsum／cumprod 実装済み（`Var::cumsum`／`cumprod`・`Op::Cumsum`／`Op::Cumprod`。CPU 参照実装先行・VJP はホスト側のみ〈厳密形・除算なし〉・facade 到達経路は既存 `Var` 再エクスポート経由〈新規 `pub use`／`pub fn` なし〉）。#1740 で CUDA／Metal 専用カーネル実装済み（`backend-cuda::scan::CudaScan`・`backend-metal::scan::MetalScan`。lane ごとの `f64`／binary64 ソフトウェアエミュレーションアキュムレータ逐次計算で CPU 参照実装と bit 完全一致契約〈Metal は `crate::soft_f64` 方式の逐語移植〉。サイズ上限・エラー写像: 要素数積の `usize` オーバーフローは両バックエンドとも `ShapeMismatch` で拒否〈フォールバックなし〉、カーネル引数の上限超過〈CUDA `i32`: `ScanSizeLimitExceeded`／Metal `u32`: `scan_model::plan_scan` の `SizeLimitExceeded`〉のみ `Unsupported` へ写像しホストフォールバックへ委譲、内部契約違反・起動失敗は覆い隠さない。GB10／M4 Max 実機は本エージェント実行環境に到達不能のため未実測のまま申し送り → Metal〈M4 Max〉は 2026-09-16 に実測済み（pass。`docs/perf/logs/metal-realdevice-phase2-2026-09-16/README.md`）・CUDA〈GB10〉は引き続き未実測 → CUDA〈GB10〉も 2026-09-16 に実測済み（pass。`docs/perf/logs/cuda-realdevice-phase2-2026-09-16/README.md`））。unique（`torch.unique` の values のみ）は #1734 で実装済み（`Var::unique`。非微分演算・detached な `Tensor<f32>` を返し `Op` を tape に記録しない。3 バックエンド〈CPU 参照実装・CUDA／Metal ビットニックソート方式〉とも bit 完全一致契約。facade 新規公開面なし〈既存 `Var` 再エクスポート経由〉。`return_inverse`／`return_counts`／`dim` 指定・`sorted=false`・`unique_consecutive` は対象外。`docs/unique-facade-exposure-decision.md`）。**#2153 追補**: `topk` の `sorted=false`・負 `dim`、`unique` の `dim` 指定・`return_inverse`／`return_counts`・`unique_consecutive` は内部クレート限定モジュール `fandhe_ai_autodiff::topk_unique_ops` として実装済み（facade 公開は承認待ち。`docs/autodiff-topk-unique-ops-decision.md`）。`docs/compat-feature-gap.md` §2.2 追補参照 |
| `nn.functional` の残り（pad／interpolate／one_hot 等） | #1631（#1755 で one_hot 実装済み。`Var::one_hot`〈**非微分演算**。VJP は明示ゼロ〉・`Op::OneHot`・CPU／CUDA／Metal 3 バックエンドカーネル・facade 新規公開面なし〈既存 `Var` 再エクスポート経由〉。#1756 で pad 実装済み（`Var::pad`・`BackendOps::pad`〈既定 `Unsupported`〉・narrow 基盤流用 VJP〈zero-copy view 連鎖〉・CPU／CUDA／Metal 専用カーネル実装済み〈算術を含まない純粋なコピー演算のため 3 バックエンド bit 完全一致契約〉・facade 新規公開面なし〈既存 `Var` 再エクスポート経由〉。CUDA／Metal 実機実測は未実施のまま申し送り → Metal〈M4 Max〉は 2026-09-16 に実測済み（pass。`docs/perf/logs/metal-realdevice-phase2-2026-09-16/README.md`）・CUDA〈GB10〉は引き続き未実測 → CUDA〈GB10〉も 2026-09-16 に実測済み（pass。`docs/perf/logs/cuda-realdevice-phase2-2026-09-16/README.md`））。イシュー #1757 で interpolate（nearest）実装済み〈`Var::interpolate`・`Op::Interpolate`・`InterpolateMode`〈`tensor-core::backend_ops`。`#[non_exhaustive]`〉。3 バックエンド専用カーネル・scatter_add ベース VJP・facade へ `InterpolateMode` を再エクスポート〈承認記録は #1631 コメント 2026-09-12〉。CUDA／Metal 実機は未実測のまま申し送り → Metal〈M4 Max〉は 2026-09-16 に実測済み（pass。`docs/perf/logs/metal-realdevice-phase2-2026-09-16/README.md`）・CUDA〈GB10〉は引き続き未実測 → CUDA〈GB10〉も 2026-09-16 に実測済み（pass。`docs/perf/logs/cuda-realdevice-phase2-2026-09-16/README.md`）〉。#1753 で clip_grad_value 実装済み（`fandhe_ai_autodiff::nn::optim::clip::clip_grad_value`。`clip_grad_norm` と同型の純関数・`Gradients`／`Var` 非依存・`Op`／`BackendOps`／VJP 非拡張。facade 到達経路は `crates/facade/src/optim.rs` の `pub use`〈`fandhe_ai::optim::clip_grad_value`〉）。イシュー #1762 で interpolate（bilinear）実装済み〈`InterpolateMode::Bilinear { align_corners: bool }`〈`#[non_exhaustive]` variant 追加〉・座標・重みの単一情報源 `tensor-core::interpolate`〈`bilinear_scale`／`bilinear_src_coord`／`bilinear_blend`〉・3 バックエンド専用カーネル（CUDA `fmaf`／Metal `fma`／CPU `f32::mul_add` による明示 FMA。受入契約は REQ-2 統一複合判定であり `Nearest` の bit 完全一致契約とは異なる）・VJP は 4 近傍重み付き scatter_add への委譲〈`grad::bilinear_src_index_and_weight_map`〉。facade 新規公開面なし（`InterpolateMode` 再エクスポートのみ）。CUDA／Metal 実機は未実測のまま申し送り → Metal〈M4 Max〉は 2026-09-16 に実測済み（pass。`docs/perf/logs/metal-realdevice-phase2-2026-09-16/README.md`）・CUDA〈GB10〉は引き続き未実測 → CUDA〈GB10〉も 2026-09-16 に実測済み（pass。`docs/perf/logs/cuda-realdevice-phase2-2026-09-16/README.md`）〉。イシュー #2152 で interpolate の残り 5 モード（`NearestExact`・`Area`・`Linear`・`Trilinear`・`Bicubic`。いずれも `#[non_exhaustive]` variant 追加）と `scale_factor`（`tensor-core::interpolate_size_from_scale_factor`。facade 新規公開面なし・`Var` へのメソッド追加なし）を実装済み〈CPU ネイティブ・ホスト参照とも同じ `tensor-core::interpolate` 単一情報源を呼ぶため forward／backward とも bit 完全一致（`NearestExact`／`Area` は算術面でも既存 `Nearest`／`adaptive_avg_pool2d` と同型）。CUDA／Metal は本 issue でカーネル実装せず `Unsupported` によるホストフォールバックのまま・実機 parity 未実測（`docs/perf/logs/interpolate-modes-2152/README.md`）。設計判断は `docs/autodiff-interpolate-modes-decision.md` を正とする〉。`docs/compat-feature-gap.md` 追補参照）。イシュー #2159 で `nn.ConvTranspose1d`（`nn::ConvTranspose1d`。`[*,*,1,*]` reshape 併合で `Var::conv_transpose2d`〈#2067〉へ委譲）・`nn.Upsample`（`nn::Upsample`／`UpsampleSize`。`size`／`scale_factor` 二択を型で表現し `Var::interpolate` へ委譲）・`nn.ZeroPad2d`（`nn::ZeroPad2d`。`Var::pad` へ委譲）・`nn.Identity`（`nn::Identity`）・`nn.Unflatten`（`nn::Unflatten`。`Var::reshape` へ委譲）の 5 層を内部クレート限定（`fandhe_ai_autodiff::nn`）で実装済み。いずれも新規 `Op`／`BackendOps`／VJP なし。facade 公開は 5 層とも完了済み（`add_conv_transpose1d`／`add_unflatten`・`Var::conv_transpose1d`／`unflatten` は #2521、`add_upsample`／`add_zero_pad2d`／`add_identity` は #2522）。型の再エクスポート・自由関数での公開は未承認のまま対象外とし、`crates/facade/src/lib.rs::SpatialLayersHoldDoctestGuard`（型名・自由関数の衝突プローブ doctest）＋`api_surface.rs` のソース走査で固定。設計判断は `docs/autodiff-spatial-layers-decision.md` を正とする |

1.2／1.3 共通の注記: 各機能の追加は薄いラッパー原則（3 節）・完全自作
コア（REQ-1）を維持し、バックエンド間数値一致は REQ-2 統一複合判定・
FMA 契約・正規化統計の f64 アキュムレータ契約に従う（**tolerance／
baseline の変更は本改定に含めない**）。新規カーネルは REQ-8 の手動
境界チェックを維持する（`04-requirements.md:234`）。

## 2. 対象外（out of scope）

**2026-09-12 改定（イシュー #1591）**: 改定前は「全方位の Python API 網羅」を
一括で対象外としていたが、REQ-9 2026-09-12 追記により Tier 1／Tier 2（1 節）
へ機能単位で対象範囲へ組み入れられた。Keras の全レイヤー種別（Conv 系・
RNN 系・Embedding 等）・callbacks・`fit()`／`compile()`・Softmax・GELU 等の
活性化関数は、いずれも 1.2／1.3 節の Tier 列挙に含まれるため**対象外の
記述から撤去した**（1.2 節の対応 issue へ実装を委ねる）。

- **引き続き対象外**（`04-requirements.md:233`）:
  - pandas 等、numpy／Keras 以外の Python ライブラリとの互換
  - **任意 `BackendOps` 実装を注入できる推論入口**（旧 `Sequential::
    predict_with_ops`）。TASK-9.4（#411）で公開面から撤去した
    （破壊的変更。REQ-12「任意 `BackendOps` 実装を注入できる公開 API を
    設けない」・`crates/facade/tests/api_surface.rs` の機械検査と整合
    させるため。0 節・4.2 節参照）。`Sequential::predict` は既定バック
    エンド（`fandhe_ai::tape()`。TASK-2.5 ユーザー承認済み）へ透過的に
    結線される
  - sparse／complex テンソル（非対応の明文化は #1633 で完了。決定記録
    `docs/tensor-core-sparse-complex-decision.md`。除外事項には従属しない
    対象外項目で、再開は 5 節手続きを要する）
  - `torch.fx`／TorchScript／`torch.jit`・分散 RPC・モバイル／エッジ
    向け変換
  - 汎用グラフ JIT（`torch.compile`／`tf.function` 相当）: 範囲整理は
    `docs/autodiff-graph-optimization-scope-decision.md`（#1632）で確定
    済み（実装済み／延長候補／非目標の 3 区分）。汎用 JIT 自体は引き続き
    対象外・延長候補（区分 B）は承認事項付きの別 issue へ引き継ぐ
- **未定義（Tier 列挙にも「引き続き対象外」にも該当しない残余。5 節手続き
  の対象）**: numpy の ufunc 長尾・ファンシーインデックス・ブロード
  キャスト以外の高度な配列操作のうち、1.2 節の要素演算・index 系（#1592・
  #1593・#1599）でカバーされない範囲。独自に in／out を確定せず、必要に
  なった時点で 5 節の手続きに従い判断する
- **実装リポ側の設計判断による非目標（トークナイザ・サービング基盤）**:
  （注記: 語彙 lookup 型のテキスト変換〈単語／文字レベル〉は REQ-9 の 2026-10-08 追記で対象内となり、#2937 で `fandhe_ai::text` へ公開済み。サブワードトークナイザ・Unicode 正規化・語彙ファイルは引き続き対象外。）
  `04-requirements.md:233` の「引き続き対象外」列挙には現れないが、#1962
  の設計記録（`docs/facade-inference-serving-scope-decision.md`）により
  実装リポ側の判断として非目標を確定した項目。PyTorch／TensorFlow 本体
  もトークナイザを同梱しない（Hugging Face `tokenizers`／`tf.text`／
  `keras_nlp` は別パッケージ）ため REQ-9 の網羅対象そのものに当たらない
  こと、外部トークナイザ crate は許容依存 9 区分外であること、自作は
  REQ-1 の自作コア範囲外の非信頼入力パース面（BPE／Unicode 正規化／
  語彙ファイルパース）を新設することが根拠。paged attention・連続
  バッチング・speculative decoding・量子化 KV・HTTP サーバ／スケジューラ
  （注記: REQ-9 2026-10-08 追記で Tier 2 になり、連続バッチング第 1 段階と
  greedy 版 speculative decoding は #2934 で `fandhe_ai::inference` へ公開済み。
  他は引き続き対象外。設計記録 §17・§18）
  等のサービング基盤も同 doc §6 で非目標として整理済み。spec の
  「引き続き対象外」列挙への追記は spec 提案候補（未起票）として同 doc
  §9 に記録し、トークナイザ分の (b) 形式文案は
  `docs/tokenizer-non-target-spec-proposal.md`（#2086）で確定した
  （未起票。起票・spec 反映後に本 bullet を「引き続き対象外」側へ移す
  作業は §5 経路 1 の後続）
- **実装リポ側の設計判断による非目標（Python バインディング・TensorFlow
  系モデル形式）**: `04-requirements.md:233` の「引き続き対象外」列挙には
  現れないが、`docs/python-binding-tf-format-non-target-spec-proposal.md`
  （#2193）により実装リポ側の判断として非目標を確定した項目。Python
  バインディング（PyO3／maturin 等による Python 拡張モジュール配布）は
  許容依存 9 区分外・型対応の複雑さ（`Tensor<T>`／`Var<'t>` と Python
  オブジェクトモデル・GIL の対応付け）・版管理コスト・ABI 安定性への
  懸念が根拠。TensorFlow 系モデル形式（SavedModel・TFLite・Keras H5）は
  REQ-7 が定める相互運用範囲（safetensors／ONNX）の外にあり、TFLite の
  FlatBuffers・H5 の HDF5 が許容依存区分外であること、SavedModel は
  TF グラフ演算の意味論全体を解釈する必要があり非信頼入力パース面
  （A03）を増やすことが根拠。書き出しは ONNX への一本化を維持し、TF 系
  形式との変換は利用者側の第三者ツール利用を推奨する。spec の
  「引き続き対象外」列挙への追記は spec 提案候補（未起票）として同 doc
  §4（提案文案）・§5（ユーザー承認事項）に記録した（未起票。起票・spec
  反映後に本 bullet を「引き続き対象外」側へ移す作業は §5 経路 1 の後続）
- **KV キャッシュ（自己回帰デコード用）**: 上記トークナイザとは対照的に
  「未定義」残余のうち §5 経路 2（ユーザー承認＋issue 起票）の起票案が
  ある項目として `docs/facade-inference-serving-scope-decision.md` §9 に
  記録した（K-1／K-2。既存 `Var` 演算の合成のみで新規 `Op`／`BackendOps`／
  依存を要しない設計。K-1 実装着手（§5 経路 2）は 2026-09-24 に
  ユーザー承認済み。K-2 はユーザー承認前のため Tier 1／Tier 2 表への
  行追加は行わない）。K-1 の設計自体は `docs/kv-cache-design.md`
  （#2083）として確定した（コード変更なし。K-1 実装着手自体は上記の
  とおり承認済み）。**K-1 は内部クレート `fandhe_ai_autodiff::nn`
  （`KvCache`／`MultiheadAttentionVars::forward_with_cache`／
  `StatefulAttention`）として実装済み（#2084。同じツリー内の前例
  〈#2085・#2137・#2134〉に倣った内部クレート限定の非破壊追加であり、
  かつ 2026-09-24 にユーザー承認済み。`docs/kv-cache-design.md` §6
  承認事項 1）。**facade 公開（K-2。`add_stateful_attention`／
  `StatefulAttention` 相当）は（#2084 時点では）未承認のため保留し、
  `crates/facade/tests/api_surface.rs` の否定ガード（`facade_does_not_
  expose_kv_cache_stateful_attention`・`KvCacheHoldDoctestGuard` の
  正のプローブ・`facade_does_not_reexport_or_declare_kv_cache_items` の
  多層構成。`docs/kv-cache-design.md` §10）で固定した
  （Tier 1／Tier 2 表への行追加は引き続き行わない）。**その後 2026-10-07 にリポジトリ所有者本人が K-2 の公開形（`docs/kv-cache-design.md` §11.6 の P1〜P4 を推奨どおり）を承認し
  （https://github.com/Fandhe-AI/fandhe-ai/issues/2499#issuecomment-6033824965）、PR #2816（#2579）で `fandhe_ai::nn::kv_cache`・`Tape::stateful_attention_forward` として公開した。保留ガードは正ガードへ置換済み（同 doc §14）。本 bullet の「未承認・保留」は #2084 時点の記録であり、§5 の適用記録を参照（Tier 1／Tier 2 表への行追加は行わない）**
- **`amax`/`max` 縮約 API**（PyTorch `torch.amax` 相当）: 縮約 API 自体は
  Tier 1（1.2 節・#1601）で対象範囲となった。`crates/autodiff/src/grad.rs`
  の `max_vjp` は同値タイ発生時「最初に現れる最大要素 1 箇所のみ」へ
  勾配を伝播する先勝ち決定的挙動を採用しており（PoC-v2-2 のビット一致
  決定性方針との整合を優先した設計判断。#224 で再確認済み）、PyTorch
  `amax` の均等分配とは異なる。この勾配分配方式（先勝ち決定的 対 均等
  分配）は #1718 で**確定済み**（既存 `Var::max`／`min`／`max_dims`
  〈`max(dim)`／`min(dim)` 族の意味論〉は先勝ち決定的を維持し、PyTorch
  `torch.amax`／`amin` 相当の均等分配は別 `Op`／別ヘルパーとして後続
  issue で実装する方針。`docs/autodiff-amax-grad-distribution-decision.md`
  参照。`04-requirements.md:234`）。**均等分配版は #2154 で
  `fandhe_ai_autodiff::extremum_ops::{amax, amin}`〈`crates/autodiff/
  src/extremum_ops.rs`。`Op::Amax`／`Op::Amin`・`grad::
  extremum_even_split_vjp`〉として内部クレート限定の自由関数で実装
  済み（`reduce_ops`〈#2147〉と同型の facade 非公開保留。**facade 公開
  〈`Var::amax`／`amin` の委譲メソッド追加〉は #2514 で公開済み（以下の保留ガードに関する記述は #2154 時点のもの）**——
  `crates/facade/src/lib.rs::VarExtremumOpsHoldDoctestGuard`＋
  `api_surface.rs` の 4 テストで到達不能を多層固定。既存
  `Var::max`／`min`／`max_dims` の先勝ち挙動は無変更。CUDA／Metal 実機
  実測は未実施のまま `docs/perf/logs/amax-amin-2154/README.md` へ
  申し送り。詳細は `docs/autodiff-amax-grad-distribution-decision.md`
  §9 を正とする）
- 対象外要望が生じた場合の受け皿は 2 通り。
  - 実装リポ側で追跡が完結する事項: `.claude/rules/out-of-scope-tracking.md`
    の規約に沿って Issue で追跡する
  - 受け入れ基準・REQ-9 自体の改定が必要な事項: 正本 spec リポジトリ
    （`Fandhe-AI/fandhe-ai-spec`）側での対応をユーザーに提案する
    （`docs/spec/` は本リポでは編集しない）

## 3. 設計原則

- **薄いラッパーに徹する**: compat 層は数値計算ロジックを自ら持たず、
  自作コア（`tensor-core`／`autodiff`）の API への委譲のみを行う
  （REQ-9・`.claude/rules/coding-rust.md`「互換 API 層は自作コアの上の
  薄いラッパーに徹する」）。`crates/autodiff/src/nn/activation.rs` の
  `Relu`／`Sigmoid`／`Tanh` は各 `forward` が対応する `Var` メソッドを
  呼ぶだけの実装であり、この原則の実例である
- **v1 試作は参考実績にとどめる**: v1 の `compat::array()`／
  `compat::Sequential`（PoC-1）・`add_leaky_relu`（PoC-2）は Burn 前提の
  試作であり、「薄いラッパー層が構造的に成立すること」の参考実績として
  残すが、v2（完全自作コア）の受け入れ基準達成の直接的な実測根拠には
  用いない（`04-requirements.md:222-236` 概要。改定前基準は `:227`）。自作コア確定後の再実装・再検証を
  要する
- **PoC-v2-6 を v2 実例として参照する**: `Mlp::from_safetensors`
  （`docs/spec/03-poc/poc-v2-6-interop/code/rust/src/mlp.rs`）は自作テンソル
  上に構築された薄い互換層の v2 実例である。numpy／Keras 慣習の本格的な
  互換 API 層そのものではないが、自作コア上でも薄いラッパーが成立し
  得ることを示す傍証として位置づける（`04-requirements.md:222-236`）

## 4. 実装配置

### 4.1 TASK-9.2a（#95）時点の確定（履歴）

- `compat::array()`／`compat::Sequential` は `fandhe_ai_autodiff::compat` モジュール
  （`crates/autodiff/src/compat/`）として実装した。当時の 9 クレート構成
  （`tensor-core`・`autodiff`・`backend-cpu`・`backend-cuda`・`backend-metal`・
  `onnx-interop`・`guardrail`・`self-repair`・`bench-harness`）に compat 専用
  クレートは追加しなかった。`Sequential` が `nn::Linear`/`nn::Module` に依存し、
  `tensor-core` は `autodiff` に依存できない（下位クレートが上位クレートへ
  依存すると循環する）ため、`autodiff` 配下以外に置く選択肢はなかった
- Linear 層（TASK-9.1a・#91）は `crates/autodiff/src/nn/linear.rs` として
  マージ済み
- 活性化関数（ReLU・Sigmoid・Tanh）は TASK-9.1b（#92）で実装済み
  （`crates/autodiff/src/nn/activation.rs`）。共通 `Module` trait は
  TASK-9.2a（#95）で `crates/autodiff/src/nn/module.rs` に確定し、`Linear`・
  `Relu`・`Sigmoid`・`Tanh` の 4 実装を持つ（`nn/mod.rs` から
  `pub use module::Module;`）。`compat::Sequential` はこの trait を介して
  `Vec<Box<dyn Module>>` で層を均一に扱う
- `Sequential` 経由の学習（勾配取得・パラメータ更新）は #294
  （`crates/autodiff/src/compat/sequential.rs` の `SequentialVars`・
  `crates/autodiff/src/nn/module.rs` の `Module::as_linear`/
  `as_linear_mut`）で対応済み。当初（#95）は `add_linear` が内部で保持
  する `Linear` の `LinearVars`（勾配取得の入口）を外部へ公開せず
  「`Sequential` からパラメータ・勾配へアクセスする手段が構造的にない」
  としていたが、`Module` trait への `as_linear`/`as_linear_mut` 追加
  （既定実装 `None`。活性化層は非オーバーライドのため非破壊）により
  解消した。`fit()`/`compile()` 等の高水準学習ループ API は 1 節注記の
  とおり引き続き対象外

### 4.2 TASK-9.4（#411）での移設確定（現行）

- 10 クレート化（イシュー #52・`facade` クレート新設。TASK-9.3・#410 で
  composition root の実装が先行完了）を受け、compat 公開面（`compat::array`／
  `compat::Sequential`）を `fandhe_ai_autodiff::compat` から **`fandhe_ai::compat`
  （`crates/facade/src/compat/`）へ移設した**。4.1 節が前提としていた
  「9 クレート構成に compat 専用クレートは存在しない」という制約が
  `facade` 新設により解消したための再配置である
- `predict_with_ops`（任意 `BackendOps` 実装を注入できる推論入口）は本移設
  で公開面から撤去した（破壊的変更）。`fandhe_ai::compat::Sequential::predict`
  は `fandhe_ai::tape()`（composition root・既定 CPU・`CpuBackendOps`・融合
  有効）へ結線済みであり、ops を明示指定する経路（旧 `predict_with_ops`）は
  REQ-12「任意 `BackendOps` 実装を注入できる公開 API を設けない」・
  `crates/facade/tests/api_surface.rs` の機械検査と矛盾するため維持しない
- `autodiff` は compat 層が依拠する `Tape`／`Var`／`nn`（`Module`・
  `Linear`・`activation` 等）を `pub` API として提供し続けるが、これは
  「サポート境界」節（0 節）が定めるとおり内部クレートとしての公開であり、
  compat 層を経由しない直接利用はサポート対象外である
- `fandhe_ai::compat::Sequential` の `forward`／`bind` は `fandhe_ai_autodiff::Tape`
  （内部クレートの生の型）を直接引数に取らず、`facade` 所有の newtype
  `fandhe_ai::Tape`（`crates/facade/src/lib.rs`）を取る。`Var`・
  `Gradients`・`AutodiffError`・`LinearVars`（`autodiff` 由来）・
  `Tensor`（`tensor-core` 由来）は迂回経路を持たない値型・エラー型のため
  `facade` の正式な公開契約として再エクスポートし、`compat` の公開
  シグネチャはこの再エクスポートパスを使う（codex-review PR #424 P1
  是正・`crates/facade/tests/api_surface.rs` の機械検査と整合）

### 4.3 移行期間中のソース互換シム（codex-review PR #424 P1 是正）

`compat` 公開面の唯一のサポート対象実装は 4.2 節のとおり `fandhe_ai::compat`
だが、TASK-9.4（#411）が `fandhe_ai_autodiff::compat` モジュール自体を互換 shim
なしで削除したことで、`fandhe_ai_autodiff::compat::{array, Sequential,
SequentialVars}` を利用する既存コードが破壊されるという P1 指摘
（codex-review PR #424・ベース側レビュー基準「公開 API の破壊的変更は
P1」）を受けた是正である。

- `crates/autodiff/src/compat/`（`mod.rs`／`array.rs`／`sequential.rs`）に
  移設前の実装を複製して残す（`facade` は `autodiff` に依存する構造の
  ため、`fandhe_ai_autodiff::compat` から `fandhe_ai::compat` へ委譲することはできない
  ―― 依存方向が逆になり循環する。したがって委譲ではなく実装の複製に
  よってのみソース互換を保てる）
- 復元対象は codex-review が指摘した 3 つの公開項目（`array`・
  `Sequential`・`SequentialVars`）。`Sequential::predict` は移設前と
  同じ挙動（`default_ops::naive_ops()` による naive CPU 参照実装）を
  維持する
- `array`／`Sequential`／`SequentialVars` は `#[deprecated]` を付与し、
  `fandhe_ai::compat` への移行を促す（`crates/autodiff/src/compat/mod.rs`
  モジュール doc 参照）
- 撤去予定: `fandhe_ai::compat` への移行が完了し利用実績が確認でき次第、
  別イシューで本シムごと削除する（`.claude/rules/out-of-scope-tracking.md`
  対象）

### 4.4 `predict_with_ops` の再復元（codex-review PR #424 P1 是正・2 巡目）

4.3 節の初回是正では旧 `predict_with_ops`（任意 `BackendOps` 実装を注入
できる推論入口）を「REQ-12 違反のため復元しない」としていたが、これは
誤りだった。REQ-12「任意 `BackendOps` 実装を注入できる公開 API を設けない」
は 0 節が定める**サポート対象公開 API 面（= `facade`）**を対象とする制約
であり、`fandhe_ai_autodiff::compat` の非推奨シムは移行期間中のソース互換シム
（サポート対象公開面ではない）であるため REQ-12 の対象外である。codex-review
はこの区別を踏まえ「`predict_with_ops` を `#[deprecated]` 付きで維持する」
ことを P1 として指摘し、本節でこれに従い復元した。

- `crates/autodiff/src/compat/sequential.rs` に `predict_with_ops`
  （`Box<dyn BackendOps + Send>` を受け取る版）を `#[deprecated]` 付きで
  復元し、`predict`（無引数版）はこれへ委譲する（移設前の実装と同一。
  4.2 節「PR #403 の P1 是正で `predict`/`predict_with_ops` に分離」の
  形へ戻す）
- **`fandhe_ai::compat::Sequential` 側には追加しない**（4.2 節の判断は維持。
  `facade` は唯一のサポート対象公開面であり、ops 注入経路を設けない
  という REQ-12 の制約はここでこそ効く）
- 1 節「対象外（out of scope）」の「任意 `BackendOps` 実装を注入できる
  推論入口は対象外」との記述は、`fandhe_ai::compat`（サポート対象公開面）
  の対象範囲についての記述であり、本節の `fandhe_ai_autodiff::compat` 非推奨
  シムでの復元と矛盾しない（0 節「技術的に `pub` であることと、利用者
  向けにサポートされる公開面であることは区別する」を参照）

## 5. 範囲拡張の手続き

本文書が定める対象範囲（1 節）の拡張は、以下いずれかの手続きを経ることを
必須とする。AI 自律メンテナンス（self-repair ループ）による無断拡大は
行わない（REQ-5／`.claude/rules/security.md` の自己修復ガードレールと整合）。

1. 正本 spec リポジトリ（`Fandhe-AI/fandhe-ai-spec`）側での REQ-9
   受け入れ基準の改定
2. 本リポジトリのユーザー承認を得たうえでの Issue 起票・本文書の更新

**適用記録（イシュー #2934・親 #2932。承認: `https://github.com/Fandhe-AI/fandhe-ai/issues/2499#issuecomment-6067263650` の項 1）**: 第 1 段階の 3 名（`generate_speculative`〈greedy・B = 1〉・`SpeculativeConfig`・`BatchScheduler`）と署名に現れる `RequestId`・`SchedulerLimits` を、`fandhe_ai::inference` へ別の `pub use` 2 文で純再エクスポートした（facade 独自の型・メソッド・別名なし。`fandhe-ai =0.10.0` に対しては追加のみ）。`KvCache` の公開メソッド・`forward_with_cache` の可視性・依存・tolerance・baseline は不変。内部状態保持型が対象外であることを rustdoc の契約に明記した。保留ガードは公開した名前の分だけ正ガードへ反転し、未承認経路（サンプリング版・`KvCache` の書き換え経路・`kv_rewind`・モジュール自体の再エクスポート）の否定プローブを追加した。論点 1・2・3・5・7・8 は保留のまま。詳細は設計記録 `docs/facade-speculative-decoding-batching-design.md` §18。

### 5.1 Phase 4 公開形一覧（承認依頼 #2677。#2678・#2679 担当の行と行 19・20 の損失 3 本・オプション型を取る 5 本と 5 型〈#2854 で公開済み〉・行 30 の gradcheck〈#2847 で公開済み〉は承認・公開済み。行 12〜15 の `Var` 委譲は #2850・#2851 で公開済み〈層化は保留継続〉）

**本節は、承認依頼（#2677）として集約した公開形一覧である。** 親 #2625・ルート #2499 の Phase 4
（内部実装＋保留ガードまで先行・公開は承認後）で内部クレートに実装した機能の公開形について、各決定記録に
散らばっていた推奨案を 1 表に集約した（#2677）。表の公開形は各決定記録の推奨をそのまま転記したもので、本節で
新しい推奨を作っていない。

- **承認の状況**: 2026-10-07 のユーザー承認コメント（ルート #2499・issuecomment-6033824965）が、行 1〜11・16〜18・
  21〜30 を各決定記録の推奨形で承認した（行 19・20 は `hinge_embedding_loss`・`soft_margin_loss`・
  `multilabel_margin_loss` の 3 本だけ）。行 12〜15 の全体とオプション型を引数に取る損失 5 本・オプション型 5 つは
  保留で、保留ガードは維持する。記録に形が書かれていない点は実装せずに承認依頼へ戻す条件つきである。
- **#2679 担当の公開は完了した**: 行 16・21〜28 と、行 17・18 の `Sequential::add_*` は #2679 で公開済み
  （下表の「公開先」欄に「公開済み」と書いた行。保留ガードは承認形の正ガードへ反転または縮小済み。反転後のガード名と
  公開した識別子は §5 の適用記録〈#2679〉に記す）。行 1〜11・17・18 の `Var` 委譲、行 29、行 30 の `Tape::backward_detect_anomaly` は
  #2678（演算・自動微分）で公開済み（公開した識別子と保留は §5 の適用記録〈#2678〉に記す）。**#2678 でも保留を維持したもの**: 行 19・20 の 3 本
  （`Reduction` を `fandhe_ai::nn::loss::Reduction` で名指しできる前提が #2602 未マージで満たされなかった。**#2602〈PR #2835〉のマージで前提が満たされ、
  本節の #2677 の適用記録〈損失 3 本〉で公開した**）、行 30 の `Tape::gradcheck` と
  `GradcheckOptions`／`GradcheckReport`（決定記録に facade シグネチャが書かれていなかった。→ #2846 で決定記録 §11 に記録済み・公開は #2847）、行 12〜15 の全体（承認の対象外。→ 公開形は #2849 で各決定記録 §12 に記録済み。**行 12・13 の `Var` 委譲は #2850、行 14・15 の `Var` 委譲（と `SpectralNormState` の再エクスポート）は #2851 で公開済み〈本節末尾の各適用記録〉**）。
  **2026-10-08 追記（#2853）**: ルート #2499 のコメント（issuecomment-6052732061）で、オプション型 5 つは `fandhe_ai::nn::loss` へ再エクスポートし、損失 5 本は `Var` の 1 行委譲にする方向が承認され、その形（パス・完全なシグネチャ・構築確認）を `facade-nn-loss-structs-exposure-decision.md` §11 に記録した。公開は #2854 で、それまでは未公開のまま保留ガードを維持する。
  **2026-10-08 追記（#2854）**: 上の 5 本とオプション型 5 つを、記録の形（§11）のまま公開した（本節末尾の #2854 の適用記録）。
  現在も保留なのは、`gradcheck` 系のうちモジュール名・裸の自由関数（`Tape::gradcheck` と型 2 つは #2847 で公開済み）、行 12〜15 の層化（`nn::*` 層型・`Sequential::add_*`・行 15 の結線方式。`Var` 委譲は #2850・#2851 で公開済み）、およびモジュール `elementwise_loss_ops`／`margin_focal_loss_ops` の再エクスポートと `Tape`／`Tensor` 上の同名メソッド。`norm_except_dim`・`fold_ops` 等のモジュール再エクスポートも非公開のまま。
- 公開時は保留ガードと `api_surface.rs` の否定ガードを承認形の正ガードへ反転する（#2679 で実施した型は §5 の適用記録）。
- 行 12〜15 のように層化（`nn::MaxPool3d` 等と `Sequential::add_*`）が未実装の機能は、その旨を行に書いた。
- 公開形の類型は 3 つ（`Var` の委譲メソッド／モジュール再エクスポート／`Sequential::add_*`）。どれにも当てはまらない推奨（facade `Tape` の委譲メソッド・facade 独自の薄いラッパー）は、決定記録の表現のまま書いた。
- 「非破壊性」の † は、決定記録に非破壊の明記がない行（推奨形が追加のみであることは決定記録の推奨節から読み取れるが、公開時に `api_surface.rs` で確認する）。
- 行とガードは、起票時点で `crates/facade/src/lib.rs` の `*HoldDoctestGuard` 30 個と 1 対 1 に対応していた（Phase 4 以前に承認・公開済みの保留ガードは本表に含めない）。#2679 で行 21・22・24・25・26 のガード 5 個は、残すべき未承認経路がないため削除した（宣言場所インベントリは維持）。

| # | 機能（由来） | 公開形 | 非破壊性 | 保留ガード | 公開先 | 決定記録・承認事項の所在 |
|---|---|---|---|---|---|---|
| 1 | FFT `rfft`／`irfft`／`fft`／`ifft`／`stft`／`istft`（#2631〜#2633） | `Var` 委譲 6 本＋`FftNorm`・`StftOptions`・`IstftOptions`・`StftPadMode` のクレートルート再エクスポート（`fft_ops` モジュールは再エクスポートしない） | 追加のみ† | `FftOpsHoldDoctestGuard` | #2678 で公開済み | `autodiff-fft-ops-decision.md` §7（承認事項 3 件）・§12・§13 |
| 2 | 低精度 forward（#2628） | `Var::{matmul,add,mul,relu,exp,tanh}_low_precision(.., dtype: ScalarDType)` の 1 行委譲（`Result` を返す。`compile_with_amp` の対象層は変えない） | 追加のみ（新メソッドのみ） | `VarLowPrecisionOpsHoldDoctestGuard` | #2678 で公開済み | `autodiff-low-precision-op-extension-decision.md` §3.6・§4（承認事項 (a)〜(j)）・§6.5 |
| 3 | 逆三角・双曲線関数（#2634） | `Var::atan`／`asin`／`acos`／`sinh`／`cosh`／`asinh`／`acosh`／`atanh`／`atan2` の委譲 9 本 | 追加のみ | `TrigOpsHoldDoctestGuard` | #2678 で公開済み | `autodiff-trig-ops-decision.md` §7 |
| 4 | 非有限値の判定・置換（#2635） | `Var::isnan`／`isinf`／`isfinite`／`nan_to_num`（`Option<f32>` 3 つ）の委譲 4 本 | 追加のみ | `NonfiniteOpsHoldDoctestGuard` | #2678 で公開済み | `autodiff-nonfinite-ops-decision.md` §7 |
| 5 | 累積演算（#2636） | `Var::cummax`／`cummin`／`logcumsumexp` の委譲 3 本 | 追加のみ | `CumulativeOpsHoldDoctestGuard` | #2678 で公開済み | `autodiff-cumulative-ops-decision.md` §7 |
| 6 | 順序統計・NaN 無視縮約（#2637） | `Var::median`／`median_with_indices`／`kthvalue`／`quantile`／`nanmean`／`nansum` の委譲 6 本＋`QuantileInterpolation` のルート再エクスポート | 追加のみ | `StatReduceOpsHoldDoctestGuard` | #2678 で公開済み | `autodiff-stat-reduce-ops-decision.md` §7 |
| 7 | ヒストグラム・二分探索（#2638。非微分） | `Var::histc`／`searchsorted`／`bucketize` の委譲 3 本＋`Tape::bincount`／`bincount_weighted`（整数入力に `Var` の受け手がないため `Tape` に置く） | 追加のみ | `BinningOpsHoldDoctestGuard` | #2678 で公開済み（層を含まないため #2679 は担当なし） | `autodiff-binning-ops-decision.md` §7 |
| 8 | 形状演算（#2639） | `Var::unbind`／`tensor_split`／`tensor_split_indices`／`movedim`／`swapaxes`／`rot90`／`meshgrid` の委譲 7 本＋`MeshgridIndexing` のルート再エクスポート | 追加のみ | `ShapeViewOpsHoldDoctestGuard` | #2678 で公開済み | `autodiff-shape-view-ops-decision.md` §7 |
| 9 | 索引付き更新（#2641） | `Var::scatter_reduce`／`index_add`／`index_copy`／`masked_scatter` の委譲 4 本＋`ScatterReduceMode` のルート再エクスポート | 追加のみ | `IndexedUpdateOpsHoldDoctestGuard` | #2678 で公開済み（同上） | `autodiff-indexed-update-ops-decision.md` §7（`index_add` の `alpha` を公開形に含めるかも承認事項） |
| 10 | テンソル積・距離・外積（#2640） | `Var::kron`／`tensordot`／`tensordot_axes`／`cdist`／`cross` の委譲 5 本 | 追加のみ | `TensorProductOpsHoldDoctestGuard` | #2678 で公開済み | `autodiff-tensor-product-ops-decision.md` §7 |
| 11 | pad の非定数モード（#2642） | `Var::pad_with_mode(pads, mode)`＋`PadMode` のルート再エクスポート（既存 `Var::pad` は不変。`ReflectionPad2d` 等の層は推奨に含めない） | 追加のみ | `PadModesHoldDoctestGuard` | #2678 で公開済み（同上） | `autodiff-pad-modes-decision.md` §7 |
| 12 | 3D プーリング（#2643） | `Var::max_pool3d`（`(Var, Tensor<i32>)` を返す）／`avg_pool3d` の委譲 2 本。**層化（`nn::MaxPool3d`／`AvgPool3d`・`Sequential::add_*`）は未実装** | 追加のみ | `Pool3dOpsHoldDoctestGuard` | #2850 で `Var` 委譲を公開済み（公開形は #2849 で確定。保留ガードは公開した名前の分だけ正ガードへ反転。層化は保留継続） | `autodiff-pool3d-ops-decision.md` §7・§12 |
| 13 | ConvTranspose3d・MaxUnpool（#2644） | `Var::conv_transpose3d`／`max_unpool1d`／`max_unpool2d`／`max_unpool3d` の委譲 4 本。**層化（`nn::ConvTranspose3d`／`nn::MaxUnpool*`・`Sequential::add_*`）は未実装** | 追加のみ | `ConvTranspose3dMaxUnpoolHoldDoctestGuard` | #2850 で `Var` 委譲を公開済み（公開形と設計判断 3 件は #2849 で確定。保留ガードは公開した名前の分だけ正ガードへ反転。層化は保留継続） | `autodiff-conv-transpose3d-max-unpool-decision.md` §7・§12 |
| 14 | Fold・Unfold（#2645） | `Var::unfold`／`fold` の委譲 2 本。**層化（`nn::Fold`／`nn::Unfold`・`Sequential::add_*`）は未実装** | 追加のみ | `FoldUnfoldHoldDoctestGuard`（`Var` 受け手は反転済み。残りは未承認経路の保留） | #2851 で `Var` 委譲を公開済み（名前と引数順は #2849 で確定。層化は保留継続） | `autodiff-fold-unfold-decision.md` §7・§12 |
| 15 | LRN・重み再パラメータ化（#2646） | `Var::local_response_norm`／`weight_norm`／`spectral_norm` の委譲 3 本。`SpectralNormState` はクレートルート・`norm_except_dim` は公開しない（#2849 で確定）。**層化（`nn::LocalResponseNorm`・`Linear`／`Conv` への parametrization 結線）は未実装** | 追加のみ† | `LrnWeightReparamHoldDoctestGuard`（`Var` 受け手 3 件と `SpectralNormState` は反転済み。残りは未承認経路の保留） | #2851 で `Var` 委譲と `SpectralNormState` 再エクスポートを公開済み（公開形は #2849 で確定。層化と結線方式は保留継続） | `autodiff-lrn-weight-reparam-decision.md` §7・§12 |
| 16 | 可変長系列（#2647） | モジュール再エクスポート（`fandhe_ai::nn::rnn` へ `PackedSequence`・出力型 4 種・自由関数 8 本を `pub use`）。`Var` 委譲・`Sequential::add_*` は不採用 | 追加のみ† | `PackedSequenceHoldDoctestGuard` | #2679 で公開済み（`nn::rnn`。ガードは未承認経路〈`Var`／`Tape` 委譲等〉のプローブへ縮小） | `autodiff-packed-sequence-decision.md` §7（再エクスポート対象一覧・`PackedSequence::new` の扱い等が承認事項） |
| 17 | 活性化 5 種（#2649） | `Var::selu`／`celu`／`softsign`／`hardsigmoid`／`log_sigmoid` の委譲 5 本（#2678）＋`Sequential::add_selu`／`add_celu`／`add_softsign`／`add_hardsigmoid`／`add_log_sigmoid`（#2679） | 追加のみ | `ActivationScalarOpsHoldDoctestGuard` | #2678・#2679 で公開済み（`Var` 委譲は #2678、`Sequential::add_*` は #2679） | `autodiff-activation-scalar-ops-decision.md` §7 |
| 18 | Softmin・Tanhshrink・Threshold・RReLU（#2650） | `Var::softmin`／`tanhshrink`／`threshold`／`rrelu` の委譲 4 本（#2678）＋`Sequential::add_softmin`／`add_tanhshrink`／`add_threshold`／`add_rrelu`（#2679）。`rrelu_with_noise` は公開しない | 追加のみ | `SoftminThresholdOpsHoldDoctestGuard` | #2678・#2679 で公開済み（`Var` 委譲は #2678、`Sequential::add_*` は #2679。`rrelu_with_noise` は公開しない） | `autodiff-softmin-threshold-ops-decision.md` §7 |
| 19 | 要素ごと損失 4 種（#2652） | `Var::bce_with_logits_loss_with`／`hinge_embedding_loss`／`soft_margin_loss`／`gaussian_nll_loss` の委譲 4 本。オプション型 2 つと `Reduction` の名指しは別論点（#2600 ツリー側の記録） | 追加のみ | `ElementwiseLossOpsHoldDoctestGuard` | オプション型 2 つの公開パスは `fandhe_ai::nn::loss`・シグネチャは `facade-nn-loss-structs-exposure-decision.md` §11 に記録（#2853）。`hinge_embedding_loss`・`soft_margin_loss` は #2677、`bce_with_logits_loss_with`・`gaussian_nll_loss` とオプション型 2 つは #2854 の適用記録で公開済み（`Var` 委譲 4 本。`Reduction` は `fandhe_ai::nn::loss::Reduction`）。ガードはモジュール `elementwise_loss_ops` の再エクスポートと `Tape`／`Tensor<f32>` 上の同名メソッドの拒否だけへ縮小。この 2 点は保留 | `autodiff-elementwise-loss-ops-decision.md` §7・§14、`facade-nn-loss-structs-exposure-decision.md` §11 |
| 20 | マージン・focal 損失 4 種（#2653） | `Var::multi_margin_loss`／`multilabel_margin_loss`／`multilabel_soft_margin_loss`／`sigmoid_focal_loss` の委譲 4 本。オプション型 3 つと `Reduction` の名指しは別論点（同上） | 追加のみ | `MarginFocalLossOpsHoldDoctestGuard` | オプション型 3 つの公開パスは `fandhe_ai::nn::loss`・シグネチャは `facade-nn-loss-structs-exposure-decision.md` §11 に記録（#2853）。`multilabel_margin_loss` は #2677、`multi_margin_loss`・`multilabel_soft_margin_loss`・`sigmoid_focal_loss` とオプション型 3 つは #2854 の適用記録で公開済み（`Var` 委譲 4 本。同上）。ガードはモジュール `margin_focal_loss_ops` の再エクスポートと `Tape`／`Tensor<f32>` 上の同名メソッドの拒否だけへ縮小。この 2 点は保留 | `autodiff-margin-focal-loss-ops-decision.md` §7・§14、`facade-nn-loss-structs-exposure-decision.md` §11 |
| 21 | Rprop・ASGD（#2655） | `fandhe_ai::optim`（`crates/facade/src/optim.rs`）へ `Asgd`・`AsgdConfig`・`Rprop`・`RpropConfig` の素の再エクスポート | 追加のみ | `OptimizerRpropAsgdHoldDoctestGuard`（#2679 で削除） | #2679 で公開済み | `autodiff-optimizer-rprop-asgd-decision.md` §8 |
| 22 | Adafactor・Lion（#2656） | `fandhe_ai::optim` へ `Adafactor`・`AdafactorConfig`・`Lion`・`LionConfig` の素の再エクスポート | 追加のみ | `OptimizerAdafactorLionHoldDoctestGuard`（#2679 で削除） | #2679 で公開済み | `autodiff-optimizer-adafactor-lion-decision.md` §8 |
| 23 | SWA（#2658） | モジュール公開（`fandhe_ai::optim`）。`SwaLr`・`SwaAnneal` は純再エクスポート、`AveragedModel` は facade 独自の薄いラッパー（内部 `nn::Module` を露出しないため）。fit への結線は含めない | 追加のみ† | `SwaHoldDoctestGuard` | #2679 で公開済み（`optim`。ガードは `fit` 結線〈`FitConfig`／`Sequential` のメソッド〉のプローブへ縮小） | `autodiff-swa-decision.md` §7（承認事項 5 件） |
| 24 | `PolynomialLr`・`ChainedScheduler`（#2659） | `fandhe_ai::optim` へ `ChainedScheduler`・`PolynomialLr` の純再エクスポート（1 行追加。newtype・別名なし） | 追加のみ† | `LrSchedulerPolyChainedHoldDoctestGuard`（#2679 で削除） | #2679 で公開済み | `autodiff-lr-scheduler-poly-chained-decision.md` §8（承認事項: 型名・`total_iters == 0` と `power < 0` の拒否・`ChainedScheduler` の対応メンバー集合） |
| 25 | データセット合成（#2661） | `fandhe_ai::data` へ `ConcatBatch`・`ConcatDataset`・`Subset`・`random_split`・`random_split_fractions` の純再エクスポート | 追加のみ | `DatasetComposeHoldDoctestGuard`（#2679 で削除） | #2679 で公開済み | `tensor-core-dataset-compose-decision.md` §5 |
| 26 | iterable データセット・バッチサンプラー（#2662） | `fandhe_ai::data` へ `BatchSampler`・`IterableBatches`・`IterableDataLoader`・`IterableDataset`・`StackSamples` の純再エクスポート | 追加のみ | `IterableBatchSamplerHoldDoctestGuard`（#2679 で削除） | #2679 で公開済み | `tensor-core-iterable-dataset-batch-sampler-decision.md` §5 |
| 27 | Functional API（#2665・#2667） | モジュール再エクスポート（`compat/mod.rs` の `pub use`。仮称 `FunctionalBuilder`・`FunctionalModel`・`Node`・`save_functional_model`・`load_functional_model`）。学習用ハンドル `FunctionalVars` は公開せず内部のまま。`Var` 委譲・`Sequential::add_*` は不採用 | 追加のみ（同 §9） | `FunctionalApiHoldDoctestGuard` | #2679 で公開済み（`compat`。ガードは `FunctionalVars`・モジュール公開・`Sequential::apply`／`call` のプローブへ縮小） | `facade-functional-api-decision.md` §10・§13（承認事項 11 項目。1 項目目が本書 §5 による対象範囲への組み入れ）・§18 |
| 28 | Functional の結合層（#2666） | `FunctionalBuilder::{concatenate, add, multiply, average}` を #2679 で型と同時に公開（`merge_ops` の自由関数は再エクスポートしない） | 追加のみ（同 §9） | `MergeOpsHoldDoctestGuard` | #2679 で公開済み（`FunctionalBuilder` のメソッド。ガードは `merge_ops` の自由関数・`Var` 等への結合メソッドのプローブとして維持） | `facade-functional-api-decision.md` §13・§17 |
| 29 | jacobian・hessian（#2670） | facade `Tape` の委譲メソッド `Tape::jacobian`／`Tape::hessian`（`Var` 委譲・モジュール再エクスポートではない） | 追加のみ | `JacobianHessianHoldDoctestGuard` | #2678 で公開済み | `autodiff-jacobian-hessian-gradcheck-decision.md` §3.7・§4（承認事項 (a)〜(k)）・§8 |
| 30 | gradcheck・anomaly detection（#2671） | facade `Tape` の委譲メソッド `Tape::gradcheck`／`Tape::backward_detect_anomaly`＋`GradcheckOptions`／`GradcheckReport` のルート再エクスポート | 追加のみ | `GradcheckAnomalyHoldDoctestGuard` | `Tape::backward_detect_anomaly` は #2678 で公開済み。`Tape::gradcheck`・`GradcheckOptions`・`GradcheckReport` は #2847 で公開済み（シグネチャは #2846 で決定記録 §11 に記録。保留ガードは公開した名前の分だけ正ガードへ反転） | 同 §3.7・§4・§9・§11 |

**承認時に決めてほしい事項（決定記録上、推奨が 1 案に定まっていない点）**

- 行 15: `SpectralNormState` の公開位置と、`norm_except_dim` の置き場所（`Tensor` 上のメソッド／自由関数の再エクスポート／公開しない）。→ #2849 で確定（`autodiff-lrn-weight-reparam-decision.md` §12）。
- 行 14: `Var::unfold` の名前（`torch.Tensor.unfold` と別演算になる）と引数順（crate 内の `conv2d` 系に揃えるか PyTorch に揃えるか）。→ #2849 で確定（`autodiff-fold-unfold-decision.md` §12）。
- 行 19・20: オプション型（`BceWithLogitsOptions`・`GaussianNllOptions`・`MultiMarginOptions` 等）と `Reduction` を公開面で名指しするか。（#2678 の結果: 承認は 3 本のみで、`Reduction` の公開経路〈#2602〉が未整備のため 3 本とも保留。**#2853 で決定・#2854 で公開**: オプション型は `fandhe_ai::nn::loss` の 1 経路で名指しする。形は `facade-nn-loss-structs-exposure-decision.md` §11。）
- 行 12〜15 の層化と、行 15 の parametrization の結線方式（重みの置換か層ラッパーか）。

**公開先の要確認（§5 の適用記録と、機能の性質が食い違う点。記録は書き換えていない）**

- 行 23（SWA）: 決定記録の推奨は `fandhe_ai::optim` への公開（学習系）で、#2679 で公開した（§5 の適用記録は #2678 と #2679 の両方を挙げていた点は、公開した結果に合わせて #2679 が担当と整理した）。
- 行 7・9・11（#2638・#2641・#2642）: 推奨形は `Var`／`Tape` の委譲メソッドと型の再エクスポート（演算系）だが、§5 の適用記録は #2678 と #2679 の両方を挙げている（層は推奨に含まれない）。#2677 の承認の記録どおり #2678 が公開した。
- 行 16（可変長系列）は `nn::rnn` へのモジュール再エクスポートで、#2678・#2679 の両方を挙げている。

**適用記録（経路 2。イシュー #2679・親 #2625。Phase 4 の層・学習系の公開）**: ルート #2499 の 2026-10-07 ユーザー承認コメント
（issuecomment-6033824965。「Phase 4（#2625）」節で行 16〜18・21〜30 を各記録の推奨形で承認）に従い、`fandhe-ai =0.10.0` の
公開 API を壊さない追加のみで次を公開した。依存・tolerance・baseline・ガードレール閾値・`docs/spec` は変更していない。

- `fandhe_ai::nn::rnn`（行 16）: `PackedSequence`・`PackedRnnSeqOutput`・`PackedLstmSeqOutput`・`StackedPackedRnnSeqOutput`・`StackedPackedLstmSeqOutput` と
  自由関数 8 本（`pack_padded_sequence`・`pad_packed_sequence`・`{rnn,gru,lstm}_forward_packed`・`stacked_{rnn,gru,lstm}_forward_packed`）の純再エクスポート。
- `compat::Sequential::add_*`（行 17・18）: `add_selu`・`add_celu`・`add_softsign`・`add_hardsigmoid`・`add_log_sigmoid`・`add_softmin`・
  `add_tanhshrink`・`add_threshold`・`add_rrelu`（`save_model` の kind は決定記録に形がないため追加せず `UnsupportedModel` で拒否）。
- `fandhe_ai::optim`（行 21〜24）: `Rprop`・`Asgd`・`Adafactor`・`Lion`（各 `*Config` 付き）・`PolynomialLr`・`ChainedScheduler`・`SwaLr`・`SwaAnneal` の
  素の再エクスポートと、facade 独自の薄いラッパー `AveragedModel`（EMA と同型。`fit` への結線なし）。
- `fandhe_ai::data`（行 25・26）: `Subset`・`ConcatDataset`・`ConcatBatch`・`random_split`・`random_split_fractions` と
  `BatchSampler`・`IterableBatches`・`IterableDataLoader`・`IterableDataset`・`StackSamples` の純再エクスポート。
- `fandhe_ai::compat`（行 27・28）: `FunctionalBuilder`（結合 4 種のメソッドを含む）・`FunctionalModel`・`Node`・`save_functional_model`・`load_functional_model`。
  `FunctionalVars` は非公開のまま。

保留ガードの反転・縮小は `crates/facade/tests/api_surface.rs` に承認形だけを許す正ガード（`facade_exposes_phase4_training_data_only_in_approved_shape`・
`facade_exposes_packed_sequence_only_in_approved_shape`・`facade_exposes_swa_only_in_approved_shape`・`facade_exposes_functional_api_only_in_approved_shape`・
`compat_sequential_phase4_activation_layers_add_methods_have_approved_signatures`・各 `*_are_reachable_via_facade_only`・
`*_usage_doctests_are_present_and_compiled`）として置いた。**実機（CUDA／Metal）の parity 実測は未実施**で、
`docs/perf/logs/compat-sequential-activation-scalar-layers-2679/README.md` に実行コマンドと記入欄を置いた。

**適用記録（経路 2。イシュー #2678・親 #2625。Phase 4 の演算・自動微分の公開）**: ルート #2499 の 2026-10-07 ユーザー承認コメント
（issuecomment-6033824965。「Phase 4（#2625）」節で行 1〜11・17・18・29・30 を各記録の推奨形で承認。行 19・20 は 3 本だけ）に従い、`fandhe-ai =0.10.0` の
公開 API を壊さない追加のみで次を公開した。依存・tolerance・baseline・ガードレール閾値・`docs/spec` は変更していない。

- `Var` の委譲メソッド（1 行委譲。新規 `Op`・`BackendOps` メソッド・`AutodiffError` variant・`unsafe` なし）: FFT 6 本（行 1）・低精度 forward 6 本（行 2。
  `matmul_low_precision` は既存の本体付き `pub(crate)` メソッドを `pub` にしただけで、委譲の向きが記録 §6.5 と逆）・逆三角／双曲線 9 本（行 3）・非有限値 4 本（行 4）・
  累積 3 本（行 5）・順序統計／NaN 無視縮約 6 本（行 6）・`histc`／`searchsorted`／`bucketize`（行 7）・形状 7 本（`meshgrid` は関連関数。行 8）・
  索引付き更新 4 本（`index_add` は `alpha` なし。行 9）・テンソル積 5 本（行 10）・`pad_with_mode`（行 11）・活性化 5 本（行 17）・`softmin`／`tanhshrink`／`threshold`／`rrelu`（行 18。`rrelu_with_noise` は公開しない）。
- facade `Tape` のメソッド（`&self.0` を渡すだけの 1 行委譲。`hessian` は `&child.0`）: `bincount`・`bincount_weighted`（行 7）・`jacobian`・`hessian`（行 29）・`backward_detect_anomaly`（行 30）。
- クレートルートの型の再エクスポート（内部クレートのルート経由・別名なし・1 文 1 行）: `FftNorm`・`StftPadMode`・`StftOptions`・`IstftOptions`・`QuantileInterpolation`・
  `ScatterReduceMode`・`PadMode`・`MeshgridIndexing`。`fft_ops` 等のモジュールは再エクスポートしない。
- **保留を維持した点（記録に形が書かれていない・前提が満たされない）**: ① 行 19・20 の 3 本（`hinge_embedding_loss`・`soft_margin_loss`・`multilabel_margin_loss`）は `Reduction` を
  `fandhe_ai::nn::loss::Reduction` で名指しできる前提だが、#2602 が未マージで `nn::loss` がないため公開せず、`Reduction` の公開経路も本イシューでは作らない
  （損失系 2 つの保留ガードと否定ガードは無変更）。**→ #2602〈PR #2835〉のマージ後、下の #2677 の適用記録で公開した。**② `Tape::gradcheck` と `GradcheckOptions`／`GradcheckReport` は、`autodiff-jacobian-hessian-gradcheck-decision.md` §3.4・§3.7・§9.4 に
  facade メソッドのレシーバとテープ生成の受け方が書かれていないため保留（#2677 へコメント済み）。**→ シグネチャは #2846 で同記録 §11 に記録済み。#2847 で公開した（本節末尾の #2847 の適用記録）。**③ 行 12〜15 は承認の対象外で一切触れていない。

保留ガードの反転・縮小は、`FftOpsHoldDoctestGuard` ほか 14 個の `*HoldDoctestGuard` から公開した受け手の `impl`・UFCS 行と型のローカル定義を外す部分反転（先例 #2516・#2519）で、
`GradcheckAnomalyHoldDoctestGuard` は `Tape::backward_detect_anomaly` のプローブだけを外した。`crates/facade/tests/api_surface.rs` には承認形だけを許す正ガード
（`var_phase4_ops_methods_are_thin_delegations`・`phase4_matmul_low_precision_free_fn_delegates_to_method`・`facade_tape_phase4_methods_are_thin_delegations`・
`facade_reexports_phase4_ops_types_only_in_approved_shape`・`phase4_held_items_remain_unexposed`）を置き、各グループの宣言場所インベントリへ `var.rs`（と `facade/src/lib.rs`）の各 1 件を足した。
利用テストは `crates/facade/tests/phase4_ops_facade.rs`。**実機（CUDA／Metal）の parity 実測は未実施**で、`docs/perf/logs/phase4-ops-autodiff-exposure-2678/README.md` に実行コマンドと記入欄を置いた。

**適用記録（経路 2。イシュー #2677・親 #2625。Phase 4 の損失 3 本の公開）**: ルート #2499 の 2026-10-07 ユーザー承認コメント
（https://github.com/Fandhe-AI/fandhe-ai/issues/2499#issuecomment-6033824965。「Phase 4（#2625）」節で行 19・20 は `hinge_embedding_loss`・`soft_margin_loss`・
`multilabel_margin_loss` の 3 本だけを承認）に従い、#2678（PR #2828）で `Reduction` の公開経路が未整備のため見送られた 3 本を、#2602（PR #2835）で
`fandhe_ai::nn::loss::Reduction` が公開された後に、`fandhe-ai =0.10.0` の公開 API を壊さない追加のみで公開した。依存・tolerance・baseline・ガードレール閾値・`docs/spec` は変更していない。

- 公開した `Var` の 1 行委譲メソッド 3 本（`crates/autodiff/src/var.rs`。新規 `Op`・`BackendOps` メソッド・`AutodiffError` variant・`unsafe` なし）:
  `Var::hinge_embedding_loss(&self, y: &Tensor<f32>, margin: f32, reduction: Reduction) -> Result<Var<'t>, AutodiffError>`・
  `Var::soft_margin_loss(&self, y: &Tensor<f32>, reduction: Reduction) -> Result<Var<'t>, AutodiffError>`・
  `Var::multilabel_margin_loss(&self, target: &Tensor<i32>, reduction: Reduction) -> Result<Var<'t>, AutodiffError>`。
  シグネチャは決定記録 §7 の「既存 7 損失と同形の `Var` 1 行委譲」と、自由関数
  （`elementwise_loss_ops`／`margin_focal_loss_ops`）の引数列から一意に定まる（レシーバ = `input`）。`Reduction` は
  `fandhe_ai::nn::loss::Reduction` の 1 経路のみで、crate ルートには出していない。モジュール `elementwise_loss_ops`／`margin_focal_loss_ops` は再エクスポートしない。
- **保留のまま残したもの**: `bce_with_logits_loss_with`・`gaussian_nll_loss`・`multi_margin_loss`・`multilabel_soft_margin_loss`・`sigmoid_focal_loss` の 5 本と、
  オプション型 `BceWithLogitsOptions`・`GaussianNllOptions`・`MultiMarginOptions`・`MultiLabelSoftMarginOptions`・`SigmoidFocalLossOptions` の 5 つ（承認の対象外）。
- 保留ガードの縮小: `ElementwiseLossOpsHoldDoctestGuard`・`MarginFocalLossOpsHoldDoctestGuard` から、受け手 `Var` の trait メソッド・UFCS 行のうち公開した 3 本を外した
  （残すと公開した inherent メソッドと衝突してコンパイルできない）。承認していない受け手 `Tape`／`Tensor<f32>` への同名メソッドの配置は、専用トレイトで引き続き拒否する。
  `api_surface.rs` では、公開した 3 本の本体を `var_phase4_ops_methods_are_thin_delegations` へ追加し、2 系統の宣言場所インベントリに `var.rs` の各 1 件を足し、
  `phase4_held_items_remain_unexposed` を保留の 5 本・5 型の否定（`var.rs` に `fn` なし、走査ガードが検出）に差し替えた。
- テスト: `crates/facade/tests/loss_var_delegates.rs`（`fandhe_ai` の import だけで 3 本の到達・シグネチャ・forward／backward の自由関数との bit 一致・閉形式値・不正引数の型付きエラー）、
  `crates/facade/src/nn/loss.rs` のモジュール doc に facade 単独で完結する doctest。**実機（CUDA／Metal）の parity は既存の `*_loss_ops_backend_parity.rs`（`#[ignore]`）のまま未実測**（新規 GPU カーネルなし）。

**適用記録（経路 2。イシュー #2854・親 #2852。オプション型を取る損失 5 本の公開）**: ルート #2499 の 2026-10-08 所有者コメント
（https://github.com/Fandhe-AI/fandhe-ai/issues/2499#issuecomment-6052732061。「オプション型は `fandhe_ai::nn::loss` へ再エクスポートし、損失は `Var` の 1 行委譲にする」）と、その形を書いた
`facade-nn-loss-structs-exposure-decision.md` §11（#2853）のとおりに公開した。承認の範囲は記録の形に限り、それ以上は主張しない。
`fandhe-ai =0.10.0` の公開 API を壊さない追加のみで、依存・tolerance・baseline・ガードレール閾値・`docs/spec` は変更していない。

- 公開したオプション型 5 つ（`fandhe_ai::nn::loss::<型名>` の 1 経路のみ。別名・glob・crate ルート・`nn` 直下には出していない）:
  `BceWithLogitsOptions`・`GaussianNllOptions`・`MultiMarginOptions`・`MultiLabelSoftMarginOptions`・`SigmoidFocalLossOptions`。
  内部は `crates/autodiff/src/nn/loss.rs` の `pub use` 2 文で既存 4 型と同じ経路へ集約し、facade は `fandhe_ai_autodiff::nn::loss::` 接頭辞だけから再エクスポートした（`fandhe_ai::nn::loss` は 19 名から 24 名）。
- 公開した `Var` の 1 行委譲メソッド 5 本（`crates/autodiff/src/var.rs`。新規 `Op`・カーネル・`unsafe` なし）:
  `bce_with_logits_loss_with(&self, target: &Var<'t>, reduction: Reduction, options: &BceWithLogitsOptions)`（**この 1 本だけ `reduction` → `options` の順**）・
  `gaussian_nll_loss(&self, target: &Var<'t>, var: &Var<'t>, options: &GaussianNllOptions, reduction: Reduction)`・
  `multi_margin_loss(&self, target: &Tensor<i32>, options: &MultiMarginOptions, reduction: Reduction)`・
  `multilabel_soft_margin_loss(&self, target: &Tensor<f32>, options: &MultiLabelSoftMarginOptions, reduction: Reduction)`・
  `sigmoid_focal_loss(&self, target: &Tensor<f32>, options: &SigmoidFocalLossOptions, reduction: Reduction)`。戻り値はすべて `Result<Var<'t>, AutodiffError>`。
- 保留を維持したもの（決定記録 §11.6）: モジュール `elementwise_loss_ops`／`margin_focal_loss_ops` の再エクスポート、`Tape`／`Tensor` 上の同名メソッド、裸の自由関数、nn 構造体、`compat::Loss` の variant。
- ガード: `ElementwiseLossOpsHoldDoctestGuard`・`MarginFocalLossOpsHoldDoctestGuard` は構造体を残したまま、公開済みの型・`Var` メソッドのプローブを外し、モジュール名と `Tape`／`Tensor<f32>` 上の同名メソッドの拒否だけへ縮小した。
  `api_surface.rs` では `NN_LOSS_NAMES` を 24 名へ、`PHASE4_VAR_EXPECTED_BODIES` を 81 件へ（5 本の本体を固定）、2 系統の宣言場所インベントリの `var.rs` を 4 名各 1 件へ、固定文言 2 つを新しい doctest 本文へ更新し、
  `*_IDENTS` からオプション型名を外した（独自宣言の禁止〈`*_TYPE_NAMES`〉とモジュール経由の迂回拒否は維持し、自己テストに新 5 型の違反例・非違反例を追加）。
- テスト: `crates/facade/tests/loss_var_delegates.rs`（`fandhe_ai` の import だけで 5 本の到達・シグネチャ・forward／backward の自由関数との bit 一致〈オプション既定値と非既定値・`Mean`／`Sum`〉・閉形式値・不正オプションの型付きエラー）、
  `nn_loss_items_are_reachable_via_facade_only`、`crates/facade/src/nn/loss.rs` の facade 単独 doctest。glob 衝突は `cargo test -p fandhe-ai --doc` で確認した。
  **実機（CUDA／Metal）の parity は既存の `*_loss_ops_backend_parity.rs`（`#[ignore]`）のまま**（新規 GPU カーネルなし）。

**適用記録（経路 1 の適用例。イシュー #1591）**: 本改定（1／2 節の
Tier 1／Tier 2 再編）は、実装リポ ルート #1570 のユーザー指示 → spec 提案
（spec イシュー #66）→ spec PR #69（`c5cf1ed`）でのマージ → 実装リポ
`docs/spec` submodule 追従（#1656）→ 本イシュー #1591 での本文書更新、
という経路 1 の手続きに沿って行った。

**Tier 1／Tier 2 に列挙済みの機能の実装は本節の再適用を要しない**（1 節の
各 issue の承認事項に従う）。1 節の Tier 列挙にも 2 節の「引き続き対象外」
にも含まれない機能の追加は、従来どおり本節の手続きを要する。

**Tier 2 の量子化・複数 GPU／DDP のゲート**: 両者は正本 spec の除外事項
「分散学習・量子化の網羅対応」（Won't・条件付き〈量子化 GEMM〉。
`docs/spec/04-requirements.md:356` の 2026-09-12 注記）に従属する。
REQ-9 の 2026-09-12 追記はこの除外事項自体を変更していない。
**#1627（量子化）は除外事項の格上げ条件（a〜e）の充足と、Phase 4
要件見直しでの新 REQ 追加のユーザー承認を得るまで実装着手不可**とし、
**#1628（DDP）は設計判断の記録（docs のみ）に留め、実装・通信層の
依存追加は行わない**。implement-issue-tree で Phase 3（親 #1573）を
消化する際は、#1627 を skip／blocked 扱いとする。

**#1627（量子化）の設計記録は `docs/backend-int8-quantization-decision.md` として完了した。** 除外事項の格上げ条件（a〜e）充足と Phase 4 新 REQ 承認まで段階 0・blocked のまま close しない。コード変更なし（`crates/**`・依存・tolerance／baseline は不変）。issue 上の承認コメント（`unsafe asm!`〈SME〉・`BackendOps` trait 拡張・facade 公開面拡張の技術的許可）は実装着手前の先取り記録であり、正本 spec の除外事項ゲート自体を解除するものではないと整理した（同 doc §0.1）。

**#1628 の設計記録は `docs/facade-multi-gpu-ddp-decision.md` として完了した。** #2074 で格上げ条件表の (b) 形式提案文案と nccl リンク契約の実測を `docs/ddp-grade-up-conditions.md` に記録。ゲート（設計記録のみ・依存追加なし）は不変。

**#1633（sparse／complex テンソルの非対応の明文化）の設計記録は `docs/tensor-core-sparse-complex-decision.md` として完了した。** 量子化／DDP と異なり除外事項「分散学習・量子化の網羅対応」には従属しない（sparse／complex は REQ-9 の「引き続き対象外」列挙にのみ現れ、格上げ条件表を持つ Won't 項目ではない）。コード変更なし。再開には本節の範囲拡張手続き（経路 1 または経路 2）を要する（同 doc §3・§9）。

**#1962（推論・サービング〈KV キャッシュ・トークナイザ・グラフ最適化区分 B〉のスコープ・段階）の設計記録は `docs/facade-inference-serving-scope-decision.md` として完了した。** コード変更なし。トークナイザ・サービング基盤（paged attention・連続バッチング・speculative decoding・量子化 KV・HTTP サーバ／スケジューラ）は実装リポ側の設計判断による非目標（2 節に独立 bullet として記録・spec 提案は未起票）。KV キャッシュ（自己回帰デコード用）は「未定義」残余のうち本節経路 2 の起票案あり（同 doc §9 の K-1／K-2。K-1 実装着手は 2026-09-24 にユーザー承認済み・K-2〈facade 公開〉はユーザー承認前。**その後 2026-10-07 承認・#2579 で公開。§5 適用記録〈イシュー #2579・#2580〉参照**）。グラフ最適化区分 B（`docs/autodiff-graph-optimization-scope-decision.md`）は段階 0 を維持しつつ HEAD 時点のゲート状況を更新し、B-1（GPU `run_fused` elementwise allowlist）のみ起票案（同 doc §9 の G-1）として記録した。#2086 でトークナイザ非目標の spec (b) 形式提案文案を `docs/tokenizer-non-target-spec-proposal.md` に記録（未起票）。#2193 で Python バインディング（PyO3／maturin 等）・TensorFlow 系モデル形式（SavedModel・TFLite・Keras H5）の非目標を実装リポ側で初めて決定記録化し、spec (b) 形式提案文案を `docs/python-binding-tf-format-non-target-spec-proposal.md` に記録（未起票）。

**適用記録（経路 2。イシュー #1955）**: RNN／LSTM／GRU（内部クレート実装は #1647 で完了済み）の facade 公開可否・公開形（`compat::Sequential::add_*` を設けるか／独立モジュールとして純再エクスポートするか）を issue コメントで 2026-09-17 にユーザー承認（「選択肢 C」: `Sequential::add_rnn`／`add_lstm`／`add_gru` は追加しない・`fandhe_ai::nn::rnn` として素の再エクスポートで公開する）。新規公開面は `pub mod nn`（`nn::rnn` の 8 型純再エクスポート）と `Tape` の委譲メソッド 3 件（`rnn_forward_seq`／`lstm_forward_seq`／`gru_forward_seq`）のみ。**委譲メソッドを追加した回避不能な理由**: `Rnn::forward_seq` 等は第 1 引数に生の `fandhe_ai_autodiff::Tape` を取るが、facade の `Tape` newtype（内部フィールドは `pub(crate)`）はこれを取り出す手段を持たないため、`&self.0` を渡すだけの薄い委譲を追加しない限り「facade のみの import で `forward_seq` に到達できる」という受入基準自体が構造的に満たせない（`Tape::step_device_param_store`〈#935〉・`Tape::backward_device_param_store`〈#1022〉と同型・同じ理由の前例）。新規 `Op`／`BackendOps`／VJP は追加していない（内部クレート `fandhe_ai_autodiff::nn::rnn` の実装は #1647 のまま不変）。`crates/facade/tests/api_surface.rs` に固定ガード 5 件（再エクスポート識別子の完全一致・純再エクスポート検査・`nn/mod.rs` の宣言限定・facade 経由到達の実行時固定・`compat::Sequential` への `add_rnn`／`add_lstm`／`add_gru` 非存在の否定ガード）を追加した。

**#1652（ONNX import 公開可否）の設計記録は `docs/facade-onnx-import-exposure-decision.md` として完了した。** DDP／量子化と異なり本項目は正本 spec の除外事項（上記）に従属しない——facade へ公開する方針自体は案 B（薄いラッパー型）として推奨されるが、facade は crates.io 公開クレートであり非公開クレートへの通常依存を持てないため、「facade から公開する」は `onnx-interop` 自体を crates.io へ公開することと構造的に等価になる。この publish 承認（命名確定・`RELEASE_CRATES` 変更を含む）は 2026-09-12 の facade 公開面拡張の承認範囲には含まれない別個の事項であり、承認が得られるまでは非公開のまま段階 0（現状維持）とする。#1775（ONNX export の facade 公開）・#1754（safetensors save／load の facade 再公開）はいずれも同じ publish 前提を共有するため blocked のまま close しない（同 doc §6.2）。

**適用記録（経路 2。イシュー #1758）**: `compat::Sequential` への `set_training`／`train`／`eval`／`training`／`named_parameters` の追加は、親 #1617 のユーザー承認コメント（2026-09-12「今承認するので進めてください」）に基づく §5 経路 2 の適用（`add_silu` 等〈#1714〉と同型）。facade 新規 `pub fn` は上記 5 件のみ（目視確認）。`crates/facade/tests/api_surface.rs` は `pub use` での `Tape`／`BackendOps` 再エクスポート禁止・`pub fn` が `BackendOps` を引数として直接取らないことを機械検査するが、本 5 件を名指しでは検証していない（戻り値経由の露出は対象外）。

**#1775（ONNX export の facade 公開）の設計記録は `docs/facade-onnx-export-exposure-decision.md` として完了した。** #1652 と同じ publish 前提（上記段落）を共有するため段階 0・blocked のまま close しない。本 issue では facade（`crates/facade/src/**`・`Cargo.toml`）へのコード追加は一切行わず、代わりに「facade（crates.io 公開クレート）が非公開クレート `onnx-interop` へ通常依存しない」ことを固定する負の guard テスト（`crates/facade/tests/api_surface.rs::facade_does_not_depend_on_unpublished_onnx_interop`／`facade_sources_do_not_reference_onnx_interop`）を追加した——CI が `cargo publish --dry-run` を実行しないため、公開クレートの `Cargo.toml` へ非公開クレートへの path 依存を誤って追加しても通常の `cargo build`／`cargo test` は成功してしまい、次回リリース（`release-all.yml`）まで壊れに気づけないという盲点を機械的に前倒しする。`onnx-interop` の crates.io 公開承認取得後は、本テストを削除ではなく「承認済み依存形状の検査」へ差し替える（同 doc §6）。

**追補（#1963・2026-09-17）**: 上記 #1652・#1775・#1754 が共有する publish
前提（`onnx-interop` 自体の crates.io 公開というユーザー承認）は 2026-09-17
にイシュー #1963 で取得済み（`docs/crates-io-publishing-order.md` §13）。
`onnx-interop` は `fandhe-ai-onnx-interop` として 7 クレート目の公開準備が
完了し、次回リリースサイクルで実 publish される。ただし facade ラッパー
API 形状・配置の承認は別事項であり未取得のまま残るため、facade 公開面
（段階 0・上記 3 issue の guard テスト）は変更なく継続する。

**適用記録（経路 2。イシュー #2017・2026-09-17 ユーザー承認）**: 上記
追補で残っていた「facade ラッパー API 形状・配置の承認」を #2017 の issue
本文・承認コメントで取得し、ONNX import のみを `fandhe_ai::interop::onnx`
として実装・公開した。公開面は `OnnxModel`（`from_bytes`／`from_path`／
`run` の 3 メソッド）・`OnnxValue`（`F32`／`I64`／`Bool`／`F16` の 4
variant）・`OnnxError`（`#[non_exhaustive]`・8 variant。内部クレートの
`GraphError`／`InterpError` はペイロードに含めず自己完結型として定義。
`docs/facade-onnx-import-exposure-decision.md` §4.2 参照）のみ。
`crates/facade/src/interop/{mod.rs, onnx.rs}` を新設し
`crates/facade/Cargo.toml` へ `fandhe-ai-onnx-interop` の通常依存を追加
（外部依存の新規追加ではなく workspace 内 path 依存の結線）。ONNX
export は **#2018 で公開済み**（下記追記参照）。safetensors save／load は
#2019 で facade 公開済み（`fandhe_ai::interop::safetensors`。§1.2
「state_dict／safetensors」行・`docs/facade-safetensors-exposure-
decision.md` §11 参照）。
`crates/facade/tests/api_surface.rs`
の旧負ガード 2 件（`facade_does_not_depend_on_unpublished_onnx_interop`／
`facade_sources_do_not_reference_onnx_interop`）は予告どおり「承認済み
依存形状の検査」（`facade_depends_on_onnx_interop_only_in_approved_shape`／
`facade_sources_reference_onnx_interop_only_in_interop_module`）へ差し替え、
公開面自体を固定する `interop_module_exposes_only_approved_onnx_surface`／
`onnx_import_types_are_reachable_via_facade` を新設した。詳細な実装記録
（設計判断・テスト構成・§6.3 充足状況）は `docs/facade-onnx-import-
exposure-decision.md` §12 を参照。

**適用記録（経路 2。イシュー #2018・2026-09-17 ユーザー承認）**: §10
承認事項 2（facade ラッパー API 形状）を issue 本文・承認コメントで取得
し、ONNX export（import 済みモデルの roundtrip export に限定。
`compat::Sequential`／`nn` -> `ExportNode` 橋渡しは対象外のまま）を
`fandhe_ai::interop::onnx` へ実装・公開した。新規公開面は
`OnnxExportOptions`（`#[non_exhaustive]`・`ir_version`／`opset_version`
の 2 フィールドのみ）・`OnnxModel::to_bytes`／`to_path` の 3 件のみ
（`OnnxError` に新規 variant は追加しない。`ExportError::UnsupportedOp`
は既存 `OnnxError::UnsupportedOp` へ domain 修飾形で写像）。新規
`Op`／`BackendOps`／VJP は追加していない（`build_model_proto` →
`proto::encode_model` への薄い委譲のみ）。`crates/facade/tests/
api_surface.rs` の承認範囲を 6 件 → 9 件へ拡張し、正例テスト・負例
テストの回帰固定・`FORBIDDEN_INTERNAL_TYPE_SUBSTRINGS` への
`ExportError`／`onnx::export::` 追加を行った。正しさは facade 単独の
roundtrip bit 完全一致（`crates/facade/tests/interop_onnx_export.rs`）
と内部クレート直接呼び出しとのバイト完全一致（`crates/facade/tests/
interop_onnx_internal_parity.rs`）で検証済み。詳細な実装記録は
`docs/facade-onnx-export-exposure-decision.md` §14 を参照。

**適用記録（経路 2。イシュー #2037・2026-09-18 ユーザー承認〈親
#2034〉）**: `docs/facade-onnx-export-exposure-decision.md` §15.7 承認事項
の項 1〜4（項 5〈Sigmoid 数値契約〉は保留のまま対象外）に基づき、
`onnx::export_nn::graph_from_layers`（#2036。Linear／ReLU 限定の橋渡し）
を 1 段委譲する `OnnxModel::from_sequential(&Sequential) -> Result<Self,
OnnxError>` を facade へ実装した。新規公開面はこの 1 メソッドのみ（承認
事項 3）・`OnnxError::UnsupportedLayer { index, layer_kind }` の 1
variant 新設（承認事項 4）。`crates/facade/tests/api_surface.rs` の
承認範囲を 9 件 → 10 件へ拡張し、正例・負例（承認外追加 `pub fn`・内部
型混入の両方の検出）・到達性テストを追加。正しさは facade 単独の
roundtrip bit 完全一致（`crates/facade/tests/
interop_onnx_export_sequential.rs`。学習済みモデルが `Sequential::
predict` と bit 完全一致）と内部クレート直接呼び出しとのバイト完全
一致（`crates/facade/tests/interop_onnx_internal_parity.rs` 群 C）で
検証済み。詳細な実装記録は `docs/facade-onnx-export-exposure-decision.md`
§17 を参照。

**適用記録（イシュー #2360・2026-09-29）**: `OnnxModel::from_path_with_limits`
と `OnnxExternalDataLimits`（external data 読み込み予算。合計バイト上限・
ファイル数上限）を facade へ追加した（新規公開面 2 件・追加のみ・
`OnnxError` 不変・既定値不変）。`api_surface.rs` の承認範囲を 10 → 12 件、
内部型名禁止リストを 10 → 12 件へ更新。詳細は
`docs/facade-onnx-import-exposure-decision.md` §16。

**適用記録（経路 2。イシュー #2065・親 #2059）**: `compat::Sequential`
への `add_softmax`／`add_log_softmax`／`add_gelu`／`add_gelu_tanh`／
`add_softplus`／`add_flatten` の 6 メソッド追加は、イシュー #2065 本文
の受け入れ条件自体（6 メソッドの実装・facade 公開を直接指示しており、
承認待ち事項として issue コメントへ列挙していない）、およびユーザーが
本ツリー（ルート #2058）を `autoMerge=true` で起動した実行指示の 2 点
を根拠として §5 経路 2 が既に充足されていると扱う（過去の `add_*` 追加
〈#1714／#1758／#1760 等〉が親 issue コメントでの明示承認を引用する
のとは異なる根拠づけであることを明記する）。`Softmax`／`LogSoftmax`／
`Gelu`／`GeluTanh`／`Softplus`（`nn::activation`）は #1594／#1713 で
実装済みの再利用のみで新規カーネルなし。`Flatten`（`nn::Flatten`）は
本 issue で新設した `Var::flatten`〈#1597〉の薄いラッパーで、新規
`Op`／`BackendOps`／VJP は追加していない。facade 新規公開面は上記 6
`pub fn` のみ（内部クレート `fandhe_ai_autodiff` の新規 `pub` 型
`nn::Flatten` 1 件は crates.io 公開クレートへの非破壊追加）。正しさは
`predict`（tape 不要経路）と `forward`（`Tape` 経由）の bit 完全一致・
常駐経路（`init_device_param_store`／`predict_resident`）の実地成功
確認で検証済み（`crates/facade/tests/compat_sequential_activation_
shape.rs`）。CUDA／Metal 実機 facade parity は未実測のまま Mac／GB10
セッションへ申し送り（`docs/perf/logs/compat-sequential-activation-
shape-2065/`）。

**適用記録（経路 2。イシュー #2068・親 #2059）**: `compat::Sequential`
への `add_transformer_encoder` 追加は、イシュー #2068 本文の受け入れ
条件自体（`facade add_transformer_encoder 公開・api_surface.rs 承認`
を明示のチェック項目として列挙しており、承認待ち事項として issue
コメントへ別掲していない）、および #2065 と同じくユーザーが本ツリー
（ルート #2058）を `autoMerge=true` で起動した実行指示の 2 点を根拠
として §5 経路 2 が既に充足されていると扱う（#2065 の適用記録と同じ
根拠づけ）。`TransformerEncoderLayer`（`nn::TransformerEncoderLayer`／
`TransformerEncoderLayerVars`）は本 issue で新設したが、既存の
`nn::MultiheadAttention`／`nn::Linear`／`nn::LayerNorm` の合成のみで
新規 `Op`／`BackendOps`／VJP は追加していない。`nn::Module` trait への
`as_transformer_encoder_layer`／`_mut`（defaulted・既定 `None`）の
追加は非破壊拡張のため crates.io 公開済み `fandhe-ai-autodiff` の外部
`impl Module` を壊さない。facade 新規 `pub fn` は `add_transformer_
encoder` の 1 件のみ（目視確認）。新規公開型は追加していない
（`TransformerEncoderLayer`／`TransformerEncoderLayerVars`／
`FeedForwardActivation` は内部クレート `fandhe_ai_autodiff` の新規
`pub` 型で、facade へは再エクスポートしていない——`add_multihead_
attention`〈#1760〉と同様、`compat::Sequential::add_*` 経由のみで
到達する構成）。**コードレビューで、`SequentialVars::trainable_vars`／
`trainable_grads`（`bind` 後の学習経路。`trainable_parameters`／
`apply_parameters` とは別メソッド）が `self.encoders` を収集しておらず
`TransformerEncoderLayer` を含むモデルが標準的な学習ループ
（`bind → forward → tape.backward → trainable_grads → optim.step`）で
一切学習されない穴が指摘され、他層と同じ順序契約
（`named_parameters` と同じ self_attn.q/k/v/out → linear1 → linear2 →
norm1 → norm2 の weight→bias 順）で 16 `Var`／勾配を収集する分岐を
両メソッドへ追加して是正済み**（`crates/facade/tests/compat_
sequential_layers.rs::transformer_encoder_bind_trainable_vars_and_
grads_count_matches_trainable_parameters`）。`apply_parameters` は 6 層種別〈#1760〉と同じ
`Module::set_parameter` による in-place 更新方式（`Linear`／`Conv2d`／
`Conv1d` の `Rebuilt` 方式とは異なる）。デバイス常駐経路
（`init_device_param_store`／`forward_resident`／`predict_resident`）
は `contains_resident_unsupported_layer` を拡張し、
`TransformerEncoderLayer` を含むモデルを fail-closed に拒否する
（`MultiheadAttention` 等と同じ理由）。CUDA／Metal 実機での facade
parity は未実測のまま Mac／GB10 セッションへ申し送り
（`crates/facade/tests/compat_sequential_layers_backend_parity.rs`）。

**#1754（safetensors save／load の facade 再公開・`Tensor` の Debug／Display）は 2 部構成として完了した。** (A)「`Tensor<T>` の `Debug`／`Display`」は本 issue でコード実装済み（`crates/tensor-core/src/tensor_fmt.rs`。打ち切り付きの値プレビュー・`Tape: Debug` の既存公開契約〈`docs/public-api-design.md` §7〉越しの DoS 耐性〈`.claude/rules/security.md` A04〉を含む。facade 新規公開面なし・既存 `pub use fandhe_ai_tensor_core::Tensor` 経由でそのまま到達）。**コードレビューで、初版の軸ごとの打ち切り（`len > 2 * FMT_EDGE_ITEMS` の軸のみ省略）だけでは、全軸長が閾値以下の高階テンソル（例 `shape=[2;20]`。numel は 100 万超）で 1 軸も打ち切られず出力が無制限に増大する穴、および `shape` の rank 自体に上限がないため再帰深さが rank に比例しスタックオーバーフローしうる穴が指摘され、総出力要素数のグローバル予算（`FMT_MAX_ELEMS`）と rank 上限ガード（`FMT_MAX_RENDER_RANK`）を追加して是正済み**（`tensor_fmt` モジュール doc・追加テスト参照）。(B)「safetensors 再公開」は当初 #1652・#1775 と同じ publish 前提を共有し段階 0・blocked としていたが、**#2019 で案 A（`fandhe_ai_onnx_interop::st_load`／`st_save` の素の再エクスポート）が確定し facade 公開を完了した**（`fandhe_ai::interop::safetensors`。`docs/facade-safetensors-exposure-decision.md` §11）。

**適用記録（`DeviceParamStore::predict_device_chain`。イシュー #1688）**:
GPU 推論チェーン単一同期化（`docs/inference-chain-single-sync-design.md`。
#1579 設計・#1688 実装）に伴い、`fandhe_ai_autodiff::optim::
DeviceParamStore` へ新規 `pub fn predict_device_chain` を追加した
（決定 1(b) 採用。展開先はすべて既存の公開型のタプルのため新規 `pub`
型は追加しない）。`crate::DeviceParamStore` は `fandhe_ai_autodiff::
optim::DeviceParamStore` の直接再エクスポート（facade `lib.rs`）のため、
本メソッドはそのまま `fandhe_ai` の公開面拡張になる。**回避不能な理由**:
`Tape::ops()` が `pub(crate)`（`crates/autodiff/src/tape.rs`）であり、
facade 側から `BackendOps` へ直接到達する手段がない。既存の
`linear_forward`／`linear_forward_with_activation`（同型の設計）と同じ
理由で `DeviceParamStore` 側に新規メソッドを追加する必要がある。
`fandhe_ai_facade::compat::sequential::Sequential::predict_resident` の
シグネチャ自体は不変（内部実装のみ chain 経路を優先するよう差し替え）。

**適用記録（経路 2。イシュー #1603）**: `compat::Sequential::add_dropout` の
追加は、親 issue コメント（2026-09-12 ユーザー承認記録。「facade 公開面
〈`fandhe_ai`／`compat`〉の §5 手続きに基づく範囲拡張」「`BackendOps`
trait 拡張」まで承認済み）に基づく §5 経路 2 の適用（`add_silu` 等
〈#1714〉・`add_dropout` は `Dropout::new` の検査を構築時点で早期化する
ため `Result` を返す点のみ異なる）。`BackendOps` trait 自体は承認範囲
内だったが、forward／backward とも既存必須メソッド `BackendOps::mul`
への単一乗算に帰着することが判明したため実際には拡張していない
（1 節「Dropout」行参照）。facade 新規 `pub fn` は `add_dropout` の 1 件
のみ（目視確認・`api_surface.rs` の既存機械検査で非破壊を確認）。

**適用記録（経路 2。イシュー #1752）**: `compat::Sequential::state_dict`／
`load_state_dict` の追加は、親 #1616 のユーザー承認コメント（2026-09-12
「§5 経路 2 の範囲拡張承認」）に基づく §5 経路 2 の適用（`add_silu` 等
〈#1714〉・`named_parameters`〈#1758〉と同型）。`nn::Module` trait への
`set_parameter`（defaulted・既定 `Err`）・`state_dict`／`load_state_dict`
（defaulted・`named_parameters`／`set_parameter` の上に組む合成のみ）の
追加は非破壊拡張のため crates.io 公開済み `fandhe-ai-autodiff` の外部
`impl Module` を壊さない。facade 新規 `pub fn` は `state_dict`／
`load_state_dict` の 2 件のみ（目視確認・`api_surface.rs` の既存機械
検査で非破壊を確認）。新規公開型は追加していない（`HashMap`／
`Tensor`／`AutodiffError` はいずれも既存）。

**適用記録（経路 2。イシュー #1770）**: `compat::Sequential::add_conv2d`／
`add_conv1d` の追加は、親 #1645 のコメント（2026-09-12「今承認するので
進めてください」）に基づく §5 経路 2 の適用（`add_silu` 等〈#1714〉と
同型）。`nn::Module` trait への `as_conv2d`／`as_conv2d_mut`／
`as_conv1d`／`as_conv1d_mut`（defaulted・既定 `None`）の追加は非破壊拡張
のため crates.io 公開済み `fandhe-ai-autodiff` の外部 `impl Module` を
壊さない。facade 新規 `pub fn` は `add_conv2d`／`add_conv1d` の 2 件のみ
（目視確認・`crates/facade/tests/api_surface.rs` の既存機械検査で非破壊
を確認）。新規公開型は追加していない。`trainable_parameters`／`bind`／
`SequentialVars::forward`／`trainable_vars`／`trainable_grads`／
`apply_parameters` の内部実装を Conv 層対応へ拡張したが、いずれも既存
公開シグネチャは変更していない（戻り値の型・順序契約は不変）。

**適用記録（経路 2。イシュー #1760）**: `compat::Sequential::
add_layer_norm`／`add_rms_norm`／`add_batch_norm1d`／`add_batch_norm2d`／
`add_embedding`／`add_multihead_attention` の追加は、親 #1618 のコメント
（2026-09-12 ユーザー承認記録「facade 公開面〈`fandhe_ai`／`compat`〉の
§5 手続きに基づく範囲拡張」「`BackendOps` trait 拡張」まで承認済み）に
基づく §5 経路 2 の適用（`add_conv2d`／`add_conv1d`〈#1770〉と同型）。
`nn::Module` trait への `as_layer_norm`／`as_rms_norm`／
`as_batch_norm1d`／`as_batch_norm2d`／`as_embedding`／
`as_multihead_attention`（`_mut` 込み。defaulted・既定 `None`）の追加は
非破壊拡張のため crates.io 公開済み `fandhe-ai-autodiff` の外部
`impl Module` を壊さない。`Embedding` への `impl Module` 新設（`nn/
embedding.rs` の「`Module` trait は実装しない（確定判断）」節を本
イシューで解消）は crate 内の新規実装であり外部実装者の型を壊さない。
`BackendOps` trait 自体は承認範囲内だったが、6 層いずれも既存 `Var`
演算（`layer_norm`／`rms_norm`／`batch_norm_*`／`embedding`／既存
`matmul`／`transpose`／`masked_fill`／`softmax`）の合成のみで成立する
ため実際には拡張していない（本表の各該当行参照）。facade 新規 `pub fn`
は `add_layer_norm`／`add_rms_norm`／`add_batch_norm1d`／
`add_batch_norm2d`／`add_embedding`／`add_multihead_attention` の 6 件
のみ（目視確認・`crates/facade/tests/api_surface.rs` の既存機械検査で
非破壊を確認）。新規公開型は追加していない。`trainable_parameters`／
`bind`／`SequentialVars::forward`／`trainable_vars`／`trainable_grads`／
`apply_parameters` の内部実装を 6 層種別対応へ拡張したが、いずれも
既存公開シグネチャは変更していない（戻り値の型・順序契約は不変）。
`apply_parameters` は `Linear`／`Conv2d`／`Conv1d`（層丸ごと再構築する
`Rebuilt` 方式）と異なり、6 層種別は `Module::set_parameter` による
in-place 更新方式とした——`Linear::from_parameters` 相当の「層を
作り直す」方式では `BatchNorm` の running stats／`num_batches_tracked`／
`training` を失う罠になるため（回帰ガードは `crates/facade/tests/
compat_sequential_layers.rs::apply_parameters_preserves_batch_norm_
running_stats_and_mode`）。デバイス常駐経路（`init_device_param_store`／
`forward_resident`／`predict_resident`）は旧 `contains_conv_layer` を
`contains_resident_unsupported_layer` へ拡張し、新規 6 層種別を含む
モデルを fail-closed に拒否する（`Conv2d`／`Conv1d` と同じ理由）。
CUDA／Metal 実機での facade parity は未実測のまま Mac／GB10 セッション
へ申し送り → Metal〈M4 Max〉は 2026-09-16 に実測済み（pass。
`docs/perf/logs/metal-realdevice-phase2-2026-09-16/README.md`）・
CUDA〈GB10〉は引き続き未実測（`crates/facade/tests/compat_sequential_layers_backend_
parity.rs`）。

**適用記録（イシュー #1961・親 #1958）**: `compat::{AmpConfig, AmpDType}`・
`Sequential::compile_with_amp`／`amp_loss_scale` の追加は、AMP（#1625）・
`compile()`／`fit()`（#1618）がいずれも本節 2 段落目「Tier 1／Tier 2 に
列挙済みの機能の実装は本節の再適用を要しない（1 節の各 issue の承認
事項に従う）」に該当するため §5 の再適用は不要——#1625・#1618 いずれも
2026-09-12 のユーザー承認コメントで「facade 公開面（`fandhe_ai`／
`compat`）の §5 手続きに基づく範囲拡張」を承認済み（範囲外は
tolerance／baseline 変更・依存追加・unsafe 監査省略のみ）。#1961 自体・
兄弟 #1959（PR #2002）にはこの根拠に基づく個別の承認コメントは無いが、
#1961 の概要が「`compat::Sequential::fit` からの opt-in」を成果物として
明示していることを根拠とした（同一ランで承認前提と判断された他イシュー
と異なり親から切り離されていない）。facade 新規公開面は `AmpConfig`／
`AmpDType`（`#[non_exhaustive]`）・`Sequential::compile_with_amp`／
`amp_loss_scale` の 4 件のみ（目視確認）。`ScalarDType` 自体は facade へ
再エクスポートしていない（`AmpDType::to_scalar_dtype` が内部でのみ
`ScalarDType` へ写像する。`docs/autodiff-low-precision-linear-design.md`
§6 の非公開方針〈#1939 未承認〉は維持）。実装記録は同 doc §7・
`docs/compat-fit-evaluate-design.md` §4 を参照。

**適用記録（経路 2。イシュー #1957）**: `compat::Sequential::add_max_pool2d`／
`add_max_pool1d`／`add_avg_pool2d`／`add_avg_pool1d`／
`add_adaptive_avg_pool2d`／`add_adaptive_avg_pool1d` の追加は、issue
コメント（2026-09-17 ユーザー承認記録「選択肢 A（6 メソッド一括追加）」）
に基づく §5 経路 2 の適用（`add_conv2d` 等〈#1770〉・`add_layer_norm` 等
〈#1760〉と同型）。引数は承認依頼時点の案から実シグネチャへ合わせた
（Avg 系は `dilation` を持たず `count_include_pad: bool` が必須。
`nn::pooling::{AvgPool2d, AvgPool1d}::new` 参照）。`nn::Module` trait へ
`is_pooling(&self) -> bool`（`as_relu` と同型の bool フック・既定
`false`）を追加し Pooling 6 型でオーバーライドする非破壊拡張
（crates.io 公開済み `fandhe-ai-autodiff` の外部 `impl Module` を壊さない）。
`compat::Sequential::contains_resident_unsupported_layer` へ組み込み、
Pooling を含む Sequential のデバイス常駐経路（`init_device_param_store`／
`forward_resident`／`predict_resident`）を fail-closed 拒否する
（承認事項どおり）。facade 新規 `pub fn` は上記 6 件のみ（目視確認・
`api_surface.rs` の既存機械検査で非破壊を確認）。新規公開型は追加して
いない。実装記録は `docs/pooling-ops-design.md` §16 を参照。

**適用記録（経路 2。イシュー #2072・親 #2059）**: `compat::Sequential`
への `fit_with_metrics`・`compat::{Metrics, MetricsResult}`・
`Monitor::ValMetric(Metrics)` variant の追加は、イシュー #2072 本文の
受入基準自体が上記公開面を明示のチェック項目として列挙していること、
および #2065／#2068 と同じくユーザーが本ツリー（ルート #2058）を
`autoMerge=true` で起動した実行指示の 2 点を根拠として §5 経路 2 が
既に充足されていると扱う（#2065／#2068 の適用記録と同じ根拠づけ）。
metrics 算術（accuracy・macro precision／recall／F1・confusion matrix）
はホスト側の整数カウント＋`f64` 導出のみの新規合成で、新規 `Op`／
`BackendOps`／VJP／カーネルは追加していない（`Var::argmax` 等の既存
演算の合成のみ）。配置は Issue 記載の `nn::Metrics` ではなく
`compat::Metrics`（`crates/facade/src/compat/metrics.rs`）とした——
`nn` モジュールは rnn 限定という既存契約（`nn/mod.rs` モジュール doc）
があり、`Loss`／`History`／`Monitor` 等の関連公開面が既に `compat` に
配置されているため。facade 新規公開面は `Metrics`（enum）・
`MetricsResult`（`#[non_exhaustive]` struct。想定構築経路は
`MetricsResult::compute` のみ）・`Sequential::fit_with_metrics`・
`Monitor::ValMetric` variant の 4 件のみ（目視確認・
`crates/facade/tests/api_surface.rs::metrics_types_are_reachable_via_
facade_only` で機械固定）。`evaluate_with_metrics`・第 3 の公開型・
`MonitorMode` の自動推定は受入基準に列挙されていないため追加していない
（対象外のまま）。実装記録は `docs/compat-metrics-design.md` を参照。

**#2061（`Var` の dtype 多重化）の設計記録は
`docs/autodiff-var-dtype-multiplexing-design.md` として完了した。**
コード変更なし。facade 公開面拡張は本イシューでは不承認のまま段階 0 を
維持し、再開には本節の範囲拡張手続き（経路 1 または経路 2）を要する。

**適用記録（イシュー #2076・親 #2034）**: `nn::Module` trait へ
`as_gelu`（`bool`）・`as_softmax`（`Option<&Softmax>`）の
2 フックを追加した（`as_relu`／`as_linear` と同型の閉集合ダウンキャスト
方式。§1 の閉集合方針の維持）。`Sigmoid` は §15.7 項 5（数値契約）が
承認保留のため `as_sigmoid` フックは追加していない（代替 (γ)。承認が
得られ次第、別 PR で結線する）。**本 2 フックは `compat::Sequential`
（`bind`／`trainable_parameters`／`apply_parameters` 等の学習経路）から
は使われない**——用途は `onnx-interop::onnx::export_nn`
（非公開クレート内部限定。`docs/onnx-export-op-mapping.md` §7）が
ONNX export 対応層を判別するためのみであり、facade 新規公開面はない
（`Softmax::dim()` を `pub(crate)` から `pub` へ変更したが、facade は
`nn::activation::Softmax` を再エクスポートしないため facade の公開面
拡張には該当しない）。詳細・数値契約は
`docs/facade-onnx-export-exposure-decision.md` §18 を参照。

**適用記録（経路 2。イシュー #2073・親 #2059）**: `compat::callbacks::
ModelCheckpoint` へのファイル保存機構（safetensors）の結線を、親
#2059（`phase:1`）の受入基準に明記された当該メソッドとして実装した。
新規公開面は `ModelCheckpoint::to_file(path: impl AsRef<Path>) -> Self`
（infallible ビルダー。FS には触れず、実際の I/O は
`ModelCheckpoint::observe`〈`pub(super)`〉呼び出し時のみ発生する）の
1 件のみ。safetensors save 自体は #2019 で facade 公開済み
（`fandhe_ai::interop::safetensors::save_safetensors_f32`）であり、
本イシューはその薄い結線（親ディレクトリ作成 + 委譲）のみを追加する
（REQ-9）。エラー型は内部クレート `autodiff::AutodiffError` へ
`InvalidArgument` として写像し（`DataError → InvalidArgument` の
既存前例に倣う。`AutodiffError` 自体に I/O variant は追加しない）、
`SaveError`（safetensors 側の型）は compat の公開シグネチャには
一切現れない。`ModelCheckpoint::restore_best_weights` 相当のビルダー・
safetensors metadata（best 値・epoch）の埋め込みは「facade 新規 `pub
fn` 1 件」の制約と両立しないため本イシューのスコープ外のまま切り出し
候補として記録する（`docs/compat-callbacks-design.md` §8）。復元
フローは既存 `EarlyStopping::restore_best_weights` との併用、または
`load_safetensors_f32` → `Sequential::load_state_dict` の組み合わせで
行う。`crates/facade/tests/api_surface.rs::
fit_types_are_reachable_via_facade_only` のビルダー連鎖へ `.to_file(..)`
を追加して固定した。詳細は `docs/compat-callbacks-design.md` §4.3・§9
を参照。

**保留記録（イシュー #2178・親 #2131。#2571 で解消済み。履歴として残す）**:
`compat::Callback` への `CsvLogger`／`JsonLogger`／`LambdaCallback` の
3 variant 追加は、#2178 時点では未承認のまま保留した（保留ガード
`CallbacksLoggersHoldDoctestGuard` ＋ `api_surface.rs` の 4 テスト。#2571 で削除済み）。
その後ルート #2499 本文「承認範囲」節の一括承認により #2571 で公開した
（本節末尾の「適用記録（経路 2。イシュー #2571 …）」参照）。

**公開済み（イシュー #2560・親 #2558。詳細は `docs/autodiff-ema-decision.md` §13。適用記録は本書 §5 の「適用記録（経路 2。イシュー #2560・#2561 …）」を参照）**: 以下の保留記録は #2179 時点のもの。

**保留記録（イシュー #2179・親 #2131）**: EMA（指数移動平均。PyTorch
`torch.optim.swa_utils.AveragedModel`／Keras 3 `EMAOverlay` 相当）は
内部クレート限定で `fandhe_ai_autodiff::nn::ExponentialMovingAverage`
（`crates/autodiff/src/nn/ema.rs`）として実装済み。`compat::FitConfig`
への `use_ema`／`ema_decay` 相当のオプション追加・`fandhe_ai::optim`
への再エクスポート・`fit()` の各 step 後自動更新（本節経路 2）は、
兄弟イシュー #2177（`FitConfig` 3 フィールド追加。保留固定・PR
#2306 で出荷中）・#2180（`FitConfig::accumulate_steps` 追加）と同種の
変更であり、親 #2131 の「設計判断記録 → 承認 → 実装の 2 段」規則
（先例 #2171・#2173・#2176・#2198）に従い未承認のまま保留した。
#2170（variant 追加は承認事項に該当しないと本文が明記）とは異なり、
#2179 にはその種の個別論証も承認コメントもない。コード変更は
`crates/facade/src/lib.rs::EmaHoldDoctestGuard`（型名 glob 衝突＋
`FitConfig`／`Sequential` inherent メソッド衝突の 2 系統を 1 ブロックで
兼ねる正のプローブ doctest）＋`crates/facade/tests/api_surface.rs` の
3 テスト（doctest ドリフト検査 2 件・facade 再エクスポート／独自
宣言／inherent メソッド追加の否定ガード。実数は自己テストを含め 4 件で、現行のガードは適用記録を参照）のみで、
`compat::{training, sequential, mod}.rs` 本体・`Cargo.toml`／
`Cargo.lock`・tolerance／baseline・`docs/spec/` は不変。承認後の完全な
公開 API 案は `docs/autodiff-ema-decision.md` §5 を参照。

**保留記録（イシュー #2180・親 #2131）**: 勾配累積（`accumulate_steps`
回の backward ごとに 1 回だけ `optimizer.step` するミニバッチ相当技法。
PyTorch の gradient accumulation パターン相当）は `compat::FitConfig`
（累積ロジック本体・`Sequential::run_fit` の窓処理）まで実装済み。
公開ビルダー `FitConfig::accumulate_steps(n: u32)`（本節経路 2）は、
親 #2131 の「設計判断記録 → 承認 → 実装の 2 段」規則（先例 #2171・
#2173・#2176・#2178・#2179・#2198）に従い未承認のまま保留した。
`accumulate_steps` フィールド自体・`#[cfg(test)]` 限定のテスト専用
セッター（`FitConfig::with_accumulate_steps_for_test`）は既に存在し、
`accumulate_steps == 1`（既定）は既存 `fit`／`fit_with_callbacks`／
`fit_with_metrics` と bit 完全一致する（R3）。コード変更の内訳は
2 系統: (1) 累積ロジック本体（`crates/facade/src/compat/training.rs`
の `FitConfig::accumulate_steps` 非公開フィールド・`Sequential::
run_fit` の窓処理・テスト専用セッター・`#[cfg(test)] mod
accumulate_tests`）は本イシューの実装スコープとして通常どおり
変更・追加している。(2) 未承認のまま保留するのは**公開ビルダー**
`FitConfig::accumulate_steps(n: u32)` のみで、その不在は
`crates/facade/src/lib.rs::GradAccumulationHoldDoctestGuard`（正の
プローブ doctest）＋`crates/facade/tests/api_surface.rs` の 3 テスト
（doctest ドリフト検査 2 件・facade の `fn accumulate_steps` 宣言 0 件
の否定ガード）で機械的に固定する。`Cargo.toml`／`Cargo.lock`・
tolerance／baseline・`docs/spec/` は不変。承認後の完全な公開 API 案・
数値契約は `docs/compat-grad-accumulation-decision.md` §5 を参照。
→ イシュー #2508 で公開済み（本節末尾の適用記録を参照）。

**保留記録（イシュー #2184・親 #2131）**: 学習 step カスタムフック
（Keras `Model.train_step()` 相当。`fit()` の既定バッチ処理を丸ごと
差し替えられる機構）は `crates/facade/src/compat/training.rs::
CustomStepHook`（非公開 `type` エイリアス）・`Sequential::run_fit` へ
の配線まで実装済み。facade 公開面 3 件（`TrainStepFn`／
`TrainStepOptimizer`／`TrainStepOutput`・`Sequential::
fit_with_train_step`。本節経路 2）は、親 #2131 の「設計判断記録 →
承認 → 実装の 2 段」規則（先例 #2171・#2173・#2176・#2178・#2179・
#2180・#2198）に従い未承認のまま保留した。`custom_step` 引数自体・
`#[cfg(test)]` 限定のテスト専用入口（`Sequential::
fit_custom_step_for_test`）は既に存在し、`custom_step = None`（既存 3
入口）は既存 `fit`／`fit_with_callbacks`／`fit_with_metrics` と bit
完全一致する（R3）。コード変更の内訳は 2 系統: (1) フック本体
（`crates/facade/src/compat/training.rs` の `CustomStepHook` 型・
`fit_with_callbacks_named`／`run_fit` の `custom_step` 引数・AMP／
勾配累積併用拒否の引数検査・テスト専用入口・`#[cfg(test)] mod
train_step_tests`）は本イシューの実装スコープとして通常どおり
変更・追加している。(2) 未承認のまま保留するのは**facade 公開面 3
件**のみで、その不在は `crates/facade/src/lib.rs::
TrainStepHoldDoctestGuard`（正のプローブ doctest）＋
`crates/facade/tests/api_surface.rs` の 4 テスト（doctest ドリフト
検査 2 件・facade の再エクスポート／独自宣言の否定ガードとその自己
テスト）で機械的に固定する。`Cargo.toml`／`Cargo.lock`・tolerance／
baseline・`docs/spec/` は不変。承認後の完全な公開 API 案・数値契約は
`docs/compat-train-step-hook-decision.md` §5 を参照。
→ イシュー #2568 で公開、#2569 で正ガードへ反転済み（本節末尾の適用記録を参照）。

**実装記録（イシュー #2362〈#2369〜#2377〉。#2188 の保留記録を更新）**: 2026-09-29 に親 #2362 で
承認を受け、`compat::Sequential` の層構成シリアライズの主案（`fandhe_ai::compat::{save_model, load_model}`・
`ModelIoError`。本節経路 2 の承認済み・実装済み扱い）を公開済み。対応範囲は `add_*` 全 30 層
（`add_module` の利用者定義層を除く）・BatchNorm の running stats・compile 状態（6 optimizer×AMP の有無と
Lbfgs。#2370〜#2373）で、テストは #2374〜#2376（bit 一致行列・改竄 manifest・ファイル I/O 脅威）。
旧世代ファイルの手動掃除・並行 save／load の非サポート・fsync 非保証は `save_model` の API doc に明記した（#2377）。
**代替案の inherent メソッド `Sequential::save`／`load` は未承認のまま保留し、保留ガードを維持する。**
設計・承認の記録は `docs/compat-model-io-decision.md`。以下は #2188 時点の保留記録（経緯として残す）。

**保留記録（イシュー #2188・親 #2131。#2188 時点の記述）**: `compat::Sequential` の層構成
シリアライズ（Keras `model.save()`／`load_model()` 相当。層構成＋
パラメータ＋`compile()` 状態の一括保存）は、本イシューに本番の
呼び出し元が存在しないため、`accumulate_steps`（#2180）・
`train_step_fn`（#2184）とは異なり**内部ロジックすら実装していない**
（先行実装すると `clippy -D warnings` の `dead_code` に抵触するため。
`docs/compat-model-io-decision.md` §0）。facade 公開面 2 件
（`fandhe_ai::compat::{save_model, load_model}`）・型 1 件
（`ModelIoError`。本節経路 2）は、親 #2131 の「設計判断記録 → 承認 →
実装の 2 段」規則（先例 #2171・#2173・#2176・#2178・#2179・#2180・
#2184・#2198）に従い未承認のまま保留した。コード変更は
`crates/facade/src/lib.rs::ModelIoHoldDoctestGuard`（正のプローブ
doctest。モジュール `model_io`・自由関数 `save_model`／`load_model`・
型 `ModelIoError`・`Sequential` への inherent メソッドの 4 経路を同時に
検出する）と `crates/facade/tests/api_surface.rs` の 4 テスト（doctest
ドリフト検査 2 件・facade の再エクスポート／独自宣言の否定ガードとその
自己テスト・workspace 全体の `save_model`／`load_model` 定義元
インベントリ〈期待集合は空〉）のみで、本番コード（`compat/
{sequential,training,mod}.rs`・`interop/safetensors.rs` のモジュール
doc 更新を除く）は不変。公開 API のみで組める手動 roundtrip テスト
（`crates/facade/tests/compat_sequential_model_io_manual.rs`）で
`state_dict`／`load_state_dict`／`interop::safetensors` 経由の bit
完全一致を先行検証した。承認後の完全な公開 API 案・ファイル形式・
意味論・検証計画は `docs/compat-model-io-decision.md` §2・§4〜§6 を
参照。
**（#2188 時点の記述。#2369・#2370・#2372〜#2376 で更新）** 公開は #2369 で最小構成〈Linear＋活性化 7 種〉として
実施済みで、#2370 で対象を `add_*` 全 30 層（`add_module` の利用者定義層を除く）へ広げた
（BatchNorm の running stats も保存・復元する〈`num_batches_tracked` は非復元。#2371〉。compile 状態は #2372・#2373〈Lbfgs〉、検証は #2374〜#2376。`docs/compat-model-io-decision.md` §4・§5）。

**#2083 の設計記録は `docs/kv-cache-design.md` として完了した。**
コード変更なし。KV キャッシュ（K-1）は既存 `Var` 演算（`cat`／
`narrow`／`detach`・`nn/attention.rs` の `project`／`split_heads`／
`sdpa_compose`）の合成のみで実装可能と確定し、キャッシュはホスト
`Tensor<f32>` 保持（`TapeNode::value` がホスト `Tensor<f32>` である
現行構造のため）、デバイス常駐化は K-3（段階 0）へ切り分けた。
K-1 実装着手（本節 §5 経路 2）は 2026-09-24 にユーザー承認済み・
#2084 で実装済み。facade 公開面拡張（K-2）は引き続き未承認のまま
（同 doc §6）。#2084 では保留固定を多層防御へ強化し、承認依頼用の
K-2 事前設計（`MultiheadAttention` 自体が facade 未到達である点・
`compat::Sequential` へのメソッド追加は §9 の「`Module` 非実装」判断
と矛盾する点を含む）を記録した（同 doc §10）。

> **更新（イシュー #2579・#2580）**: 上の「K-2 は引き続き未承認」は #2083〜#2084 時点の記録である。
> 現行は直後の適用記録（2026-10-07 承認・#2579 で公開）が正。

**適用記録（経路 2。イシュー #2579・#2580・親 #2577・ルート #2499 のコメント〈決定記録 `docs/kv-cache-design.md` §11.6 の P1〜P4 を推奨どおりとする承認。`issuecomment-6033824965`〉に基づく）**:
KV キャッシュ（K-2）を決定記録 §11.4（P1〜P4）の確定形どおり公開した（実施は #2579・PR #2816）。
公開名は `fandhe_ai::nn::kv_cache::{KvCache, MultiheadAttentionConfig, StatefulAttention}`（純再エクスポート 1 文）と
`Tape::stateful_attention_forward`（`StatefulAttention::forward` への 1 行委譲）。内部クレート側では
`StatefulAttention::from_config` を追加した（`MultiheadAttention` 型を facade が公開しないための入口）。
`fandhe-ai =0.10.0` に対して追加のみ。
正ガード: #2579 で `facade_reexports_kv_cache_items_only_in_approved_shape`（＋自己テスト
`..._detects_each_category`）・`tape_stateful_attention_forward_is_thin_delegation`・
`workspace_declares_stateful_attention_forward_only_in_facade_lib`・`kv_cache_is_reachable_via_facade_only`
を導入し、旧保留ガード（`KvCacheHoldDoctestGuard`・走査ガード 3 件・固定文言）を削除した。期待集合
（`nn_mod_declares_only_approved_submodules` 等）も承認済みの公開面へ更新している。#2580 で
`nn_kv_cache_module_is_pure_reexport`・`nn_kv_cache_module_reexports_exactly_expected_surface`
（別文での `MultiheadAttentionVars` 追加という既存ガードの穴を塞ぐ）・`kv_cache_usage_doctests_are_present_and_compiled`・
`stateful_attention_from_config_signature_is_pinned` を追加した。
利用例は `nn/kv_cache.rs` のモジュール doc（prefill → decode の doctest）と `crates/facade/tests/kv_cache_facade.rs`。
Tier 1／Tier 2 表（§1.2・§1.3）への行追加は行っていない（#2822・#2826 の先例と同じ）。
保留継続: `MultiheadAttention`・`MultiheadAttentionVars`・`forward_with_cache` の公開、
`compat::Sequential::add_stateful_attention`、`StatefulAttention::new(mha)`／`mha()` の扱い（§11.4 P2 の残課題）、
K-3（デバイス常駐キャッシュ）、`sdpa_compose` 置換、padding mask、`generate()` の公開（#2575／#2576 → #2575 で公開済み。§5 の generate 適用記録を参照）。
`Cargo.toml`／`Cargo.lock`・tolerance／baseline・`docs/spec` は不変。CUDA／Metal 実機 parity は未実測で、
申し送り先は `docs/perf/logs/kv-cache-2084/README.md`（新規カーネルなし）。
詳細は `docs/kv-cache-design.md` §14・§15。

**#2132（`nn::Module`／`ModuleList` の facade 公開可否）の設計記録は
`docs/facade-nn-module-exposure-decision.md` として完了した。** コード
変更なし。素の再エクスポート（`pub use fandhe_ai_autodiff::nn::{Module,
ModuleList}`）は `Module::forward` が生の `fandhe_ai_autodiff::Tape` を
引数に取るため、facade のみに依存する利用者は `impl Module` を書けず
「ユーザー定義層」という目的自体を満たさないと確認した（同 doc §1.3）。
`dyn Module` の object safety は `ModuleList { modules: Vec<Box<dyn
Module>>, .. }` の実装で既に実証済み（同 doc §3）。sealed 化は利用者
実装という目的と矛盾するため `Module` は open trait のまま・defaulted
メソッド追加のみ非破壊という既存運用を確認（同 doc §4）。facade 側の
薄い `Module` trait と `ModuleList`／`Sequential` コンテナを新設する
案 B を推奨候補として記録したが、facade 公開面の拡張自体は本節経路 2
の承認待ち（段階 0 継続）。#2133（実装）が想定していた素の再エクスポート
形は本判断により再確定が必要（同 doc §8）。

**#2133（`nn::Module`／`ModuleList` の facade 公開実装）は経路 2 未適用のまま
承認待ちで保留した。** コード変更なし（`#[cfg(doctest)]` 限定の非公開足場
1 件を除く）。承認事項（§10）が未承認のため否定ガード＋保留記録 doc のみを
追加した。詳細は `docs/facade-nn-module-exposure-decision.md` §12。

**#2338（#2133 が承認待ちのままクローズされた件の追跡し直し）も経路 2 未適用
のまま承認待ちを継続している。** コード変更なし。#2131・#2132・#2133・#2338・
PR #2230 のコメント・レビューを再確認したが、所有者による §10 承認事項 1〜6
への明示的な承認は依然として見つからない（PR #2230 のレビューは
`github-actions`〈codex-review〉の自動レビューのみ）。#2133 のクローズは
前例 #2063／#2064 と同じ意図的な運用だったが、その結果として承認待ちを
追う open issue が残らなかった点が #2338 の実質的な発生理由であり、本 PR
は #2338 を close せずに追跡先として維持する。詳細は
`docs/facade-nn-module-exposure-decision.md` §13。

**2026-09-29 追記: 承認済み・実施済み**（#2132／#2133／#2338 の上記 3 段落は当時の記録として残す。下記の適用記録を参照）。

**適用記録（経路 2。イシュー #2338・2026-09-29 ユーザー承認）**: 所有者が
`docs/facade-nn-module-exposure-decision.md` §10 承認事項 1〜6 を推奨案
（案 B）で承認した
（https://github.com/Fandhe-AI/fandhe-ai/issues/2338#issuecomment-5881439884。
子 issue の分解と確定事項は #issuecomment-5882029568、追加承認 2 件は
#issuecomment-5888987363・#issuecomment-5890443852）。子 issue #2394〜#2402
（PR #2407・#2412・#2417・#2419・#2425・#2432・#2426・#2427・#2429）で
実施済み。公開面は追加のみ（semver minor 相当）で、`fandhe-ai =0.9.0` の
既存公開面は変えていない。

- **新しい公開面**:
  - `fandhe_ai::nn::Module`（open trait）。required は `forward<'t>(&self,
    tape: TapeRef<'t>, input: &Var<'t>)` の 1 件、defaulted は合計 14 件
    （基本 6 件: `named_parameters`・`set_parameter`・`state_dict`・
    `load_state_dict`・`set_training`・`training`／凍結 3 件:
    `set_requires_grad`・`freeze`・`requires_grad`／イントロスペクション
    4 件: `children`・`named_modules`・`parameter_count`・`type_name`／
    `children_mut`）
  - `fandhe_ai::nn::{ModuleList, Sequential, ModuleDict}`（facade 側で子を
    直接保持するコンテナ）と `fandhe_ai::nn::summary`
  - `fandhe_ai::TapeRef<'t>`（`var`／`var_from`／`var_no_grad` と
    `From<&Tape>`。`lib.rs` で直接定義し、生 `Tape` へ戻る経路はない）
  - `compat::Sequential::add_module<M: nn::Module + 'static>(self, m: M) -> Self`
    （唯一の入口。§9 の除外を上書き）
- **公開していないもの**: autodiff の `nn::Module`・`ModuleList`・`Sequential`・
  `ModuleDict` の再エクスポート、生 `Tape`・`BackendOps`、
  `FacadeModuleAdapter`（`pub(crate)`）、組み込み層型（`Linear` 等）
- **ガードの切り替え**（`crates/facade/tests/api_surface.rs`）: 否定ガード
  `facade_does_not_reexport_nn_module_or_containers`・
  `facade_declares_no_nn_module_items`・
  `facade_does_not_reexport_module_dict_or_summary`・
  `compat_sequential_does_not_expose_module_add_methods`（`add_module` 1 件のみ
  許容）を承認済みの形だけを許す正ガードへ切り替えた。`NnModuleHoldDoctestGuard`
  と関連テスト・定数は削除した（#2396）。`nn_mod_declares_only_rnn_submodule` は
  非公開 `mod` と `pub use` の方式のため変更なし。新設した正ガードは
  `tape_ref_public_surface_is_exactly_var_family`・
  `tape_ref_declared_once_with_crate_private_field`・
  `tape_ref_pub_fns_do_not_return_raw_tape`・
  `facade_nn_module_trait_methods_match_approved_set`・
  `facade_nn_module_trait_signatures_hide_internal_types`・
  `compat_sequential_add_module_is_sole_approved_entry`・
  `nn_module_types_are_reachable_via_facade_only`・
  `nn_{mod,module_rs,container_rs}_public_items_match_expected_set`・
  `nn_containers_inherent_and_trait_impls_match_expected_set`・
  `onnx_forbidden_nn_module_substring_does_not_collide_with_facade_nn_module`
  （それぞれ自己テストを含む）
- **autodiff（内部クレート）**: 追加承認 1・2 に基づき `#[doc(hidden)]` の
  `requires_grad_snapshot`／`restore_requires_grad_snapshot`・
  `RequiresGradSnapshot` を追加した。facade の公開面には現れない
- **既知の制限**: `compat::Sequential` に積んだパラメータを持つ facade 層では
  `fit`／`bind` 系が fail-closed になる（決定記録 §17）。forward 内で登録した
  パラメータ・勾配を公開経由で集める経路はない（同 §20）。autodiff の
  `named_modules` はアダプタで包んだ facade コンテナの内側へ降りない（同 §18）
- **詳細**: `docs/facade-nn-module-exposure-decision.md` §14〜§22。
  examples の `ReferenceModule` の移行評価は
  `docs/reference-models-decision.md` §10.8 (d)

**（#2509 で公開済み。以下は #2169 時点の保留記録）#2169（`compat::Sequential::compile()` の `Loss` enum への BCE／
BCEWithLogits／NLL／KLDiv／Huber／SmoothL1／L1 追加）は経路 2 未適用の
まま承認待ちで保留した。** コード変更なし（`#[cfg(doctest)]` 限定の
非公開足場 1 件を除く）。イシュー本文の承認事項節が facade 公開面の
拡張（本節経路 2）を明記しているため未承認のまま対象外とし、
`CompileLossVariantsHoldDoctestGuard`（正のプローブ doctest）＋
`api_surface.rs` の 4 テストで機械固定した。CTC は受け入れ条件に
含まれない不整合があるため対象外と確定（`ctc_loss` が長さテンソル
2 本を追加要求し `FitTarget::loss_for` の 2 引数契約に収まらない
構造的理由による）。詳細は `docs/facade-compile-loss-variants-decision.md`
§4〜§5。

**#2135（`Var` 演算子オーバーロード〈`+`・`*`・`-`〉の facade 公開可否）
の設計記録は `docs/autodiff-var-operator-overload-design.md` として
完了した。** コード変更なし。`Var` へのトレイト impl（`Add`／`Mul`／
`Sub`／`Neg`）は facade が `Var` を再エクスポートしている
（`crates/facade/src/lib.rs:184`）ため、facade 側のコードを変更せず
とも facade の公開面を自動的に広げる。本節経路 2 の承認が得られるまで
#2136（実装）は着手不可（同 doc §12 承認事項 2）。推奨案（`Output =
Result<Var, AutodiffError>`・borrow／consumed 4 組合せ・既存 inherent
メソッドへの委譲のみで bit 同一を保証）・スカラー混合の段階 0 判断・
`Div` の扱いはいずれも承認事項として列挙のみ（同 doc §8・§11・§12）。

**#2140（`nn::init` 初期化関数群の facade 公開）は経路 2 未適用のまま承認待ち
で保留した。** イシュー本文が前提とした `Initializer` trait・各層の
`with_init` コンストラクタは実際には存在せず（前提の食い違い。`docs/
facade-nn-init-exposure-decision.md` §1）、本イシューの実体は「autodiff 側
への新規実装」＋「facade 公開（承認事項）」の 2 段だった。**autodiff 側
（`fandhe_ai_autodiff::nn::init` を `pub mod` 化し `uniform`／`normal`／
`constant`／`xavier_uniform`／`xavier_normal`／`kaiming_uniform`／
`kaiming_normal`／`orthogonal`／`trunc_normal` の 9 関数を実装）は本イシュー
で完了済み**（内部クレートへの非破壊追加のため §5 手続き対象外・`fandhe_ai_tensor_core::rng::with_global_rng` へ従属し既存の個別シード API
〈`Linear::new(.., seed)` 等〉とは独立）。facade 公開面拡張（`crates/
facade/src/nn/init.rs` の新設）は §5 経路 2 の承認取得まで実施していない。
詳細は `docs/facade-nn-init-exposure-decision.md`。
**→ イシュー #2504 で公開済み（本節末尾の適用記録参照）。**

**#2136（`Var` 演算子オーバーロード実装。`Add`／`Mul`／`Sub`）は経路 2 未適用の
まま承認待ちで保留した。** `crates/autodiff/src/**`・`crates/facade/src/**`
は変更しない（コード変更なし）。#2136・#2135・親 #2131・設計記録 PR #2237 の
いずれにも所有者の明示承認コメントが確認できなかった（bot の自動レビューの
み）。代わりに `crates/facade/tests/api_surface.rs` へ型レベルの正のプローブ
（`var_does_not_implement_arithmetic_operator_traits_while_2136_on_hold`。
`static_assertions::assert_not_impl_any!` と同型の曖昧性トリックを手書き）を
1 件追加し、`Var`／`&Var` が算術演算子トレイトを実装していないことを
fail-closed に固定した。詳細は
`docs/autodiff-var-operator-overload-design.md` §14。

**#2138（forward・backward hooks の facade 公開可否）の設計記録は
`docs/autodiff-forward-backward-hooks-design.md` として完了した。**
コード変更なし。hook 機構は 1 節の Tier 1／Tier 2 いずれの列挙にも
含まれない（Tier 外の新規事項）。backward hook の登録 API を
`Var::register_backward_hook` の形にすると、facade が `Var` を
再エクスポートしている（`crates/facade/src/lib.rs:184`）ため facade
側のコードを変更せずとも公開面が自動的に広がる。同 doc の推奨（§5.1・
§7）はこれを避け、登録の入口を autodiff の `Tape`（例:
`Tape::register_backward_hook`）に置く案であり、facade の `Tape` は
2 メソッドだけを出す newtype のため、この形であれば facade 公開面は
不変のまま保てる。forward hook は `Var` 単位では意味論が破綻するため
`nn::ForwardHooked<M: Module>` という Module レベルのラッパーとして
設計した（同 doc §5.3）が、facade の `nn` は #2133 と同じ理由で未公開
（`docs/facade-nn-module-exposure-decision.md`）のため、facade へ公開
するには `nn::Module` 公開（#2133）と本節経路 2 の双方の承認が要る。
（2026-09-29 追記: facade `nn::Module` は #2338 で公開済み。上記適用記録を参照。#2139 自体の承認状況は変わらない。）
本節経路 2 の承認（同 doc §11 承認事項 5）は #2139 着手可否を左右
する条件の一つに過ぎない。同 doc §11 は設計案（§4・§5）の承認・
hook と `CustomFunction` の役割分担・callback lifetime・エラー伝播
規則・受入基準の改訂・facade 公開の 5 項目すべてを「いずれも未実施。
#2139 着手前にユーザー承認が必要」と明記しており、facade 公開を
伴わない内部クレート `autodiff` 側の実装であっても、これら 5 項目
の承認が揃うまで #2139（実装）は着手不可である。型シグネチャ
（backward hook は `Fn(&Tensor<f32>) -> Result<(),
AutodiffError> + Send + Sync + 'static`）・保持形（`Arc`）・エラー
伝播（fail-fast）・`AutodiffError` への variant 追加要否はいずれも
承認事項として列挙のみ（同 doc §11）。

**#2139（forward・backward hooks 実装）は経路 2 未適用のまま承認待ちで
保留した。** 上記のとおり同 doc §11 の承認事項 5 項目（facade 公開の経路 2
承認を含む）がそろうまで着手不可のため、本体実装（`crates/autodiff/
src/**`）・facade 公開面（`crates/facade/src/**` 本番コード）のいずれも
変更していない（`#[cfg(doctest)]` 限定の非公開足場 1 件を除く）。代わりに
`crates/facade/src/lib.rs::VarHooksHoldDoctestGuard`（正のプローブ
doctest）＋`crates/facade/tests/api_surface.rs` の 4 層構成（doctest
ドリフト検査 2 件・workspace 全体の定義元インベントリ 1 件〈
`workspace_declares_hook_registration_fns_only_on_autodiff_tape_and_facade_delegation`〈#2587 で改名〉〉・`crates/autodiff/src/`
限定の `register_hook` allowlist 化ガード 1 件〈
`autodiff_declares_no_register_hook_fn`。doctest 正のプローブが facade
経由の到達可能性しか見ないため autodiff 側の本体実装を検出できない穴を
塞ぐ、workspace 全体インベントリの `register_hook` 意図的除外に対応する
補完ガード〉）で「facade 未公開・本体未実装」状態を fail-closed に固定
した。詳細な承認状態の確認結果・保留の根拠・ガードの構成・承認取得後の
実装範囲は `docs/autodiff-forward-backward-hooks-design.md` §13（4 層構成
の内訳は同 §13.3）。

**事実訂正（イシュー #2586）**: 上記「本体未実装」「workspace 全体で 0 件」は #2586
以前の記述である。ルート #2499 の承認（2026-10-07）後、#2586 で autodiff 内部に
`Tape::register_backward_hook`・`remove_hook`・`HookHandle`・`nn::ForwardHooked`・
`nn::ForwardHookCtx` を実装した。facade は未公開のまま（`crates/facade/src/**` 不変）で、
workspace 全体の定義元インベントリ（`workspace_declares_hook_registration_fns_only_on_
autodiff_tape`。#2587 で `..._and_facade_delegation` へ改名）は `autodiff/src/tape.rs` のみ許す形へ縮小した（保留 doctest・
`autodiff_declares_no_register_hook_fn` は維持）。経路 2 の適用記録は facade 公開を行う
#2587 で追記する。実装記録は同 doc §17。

**適用記録（経路 2。イシュー #2587・親 #2584・ルート #2499）**: forward／backward hooks の
facade 公開を、ルート #2499 の 2026-10-07 付コメント
（`https://github.com/Fandhe-AI/fandhe-ai/issues/2499#issuecomment-6033824965`。リポジトリ所有者名義）が
承認した `docs/autodiff-forward-backward-hooks-design.md` §14 の推奨案の形に限って実施した
（承認は §14 に書かれた形のみ。P5′ はコンパイル上成立したため差し戻しなし）。公開した名前:
`Tape::register_backward_hook`・`Tape::remove_hook`（facade `impl Tape` の autodiff `Tape` への 1 行委譲 2 件）、
`fandhe_ai::HookHandle`（crate ルートの純再エクスポート）、`fandhe_ai::nn::ForwardHooked<M: nn::Module>`
（facade 独自ラッパー。内部は `FacadeModuleAdapter<Box<M>>` で autodiff 側 `nn::ForwardHooked` を再利用）、
`fandhe_ai::nn::ForwardHookCtx`（純再エクスポート）。`Var`・`compat::Sequential`・`TapeRef` には足しておらず、
`pub mod hooks` も作っていない。`register_forward_hook`・`register_hook`・`remove_backward_hook` はどの型にも
無い。保留ガード `VarHooksHoldDoctestGuard` は承認形外（`hooks` モジュール・`Var`／`compat::Sequential` への
5 メソッド・`Tape` の承認外 3 メソッド）を拒否する形へ部分反転し、`api_surface.rs` に承認形の正ガード
（薄い委譲・再エクスポート形・struct 宣言所在・公開 item／固有メソッド集合・単一実装・facade のみでの到達）を
新設、定義元インベントリは `autodiff/src/tape.rs` と `facade/src/lib.rs` の委譲 2 件を許す形に更新した。
利用例は `crates/facade/tests/hooks.rs` と doctest。公開 API は追加のみ（`fandhe-ai =0.10.0` 非破壊）で、
実機 parity の申し送りは発生しない。実装記録は同 doc §18。

**適用記録（経路 2。イシュー #2198・親 #2172・ルート #2131）**: L-BFGS
（`fandhe_ai_autodiff::nn::optim::Lbfgs`。#2197 で内部クレート限定実装
済み）の facade（`fandhe_ai::optim`）公開・`compat::Optimizer` enum
（Keras 風 `compile()` 経路）への `Lbfgs(LbfgsConfig)` variant 追加は、
着手時点（2026-09-26）で #2198・親 #2172・ルート #2131 のいずれにも
所有者の明示承認コメントが確認できず、#2136（PR #2247）・#2084
（PR #2252）と同型の判断で保留固定のみを行っていた。**2026-09-27 に
所有者が facade 公開面拡張を承認**（`compat::Optimizer::Lbfgs
(LbfgsConfig)` variant の追加・`LbfgsConfig` のみの facade
再エクスポート・`compile()`/`fit()` 統合の 3 点。#2172 コメント）し、
`crates/facade/src/optim.rs`（`LbfgsConfig` の単一識別子 `pub use`）・
`crates/facade/src/compat/training.rs`（`OptimizerState::Lbfgs`・
`Sequential::run_fit` の `lbfgs_batch_step` 分岐）を実装した。学習曲線
の受け入れ検証は `crates/facade/tests/compat_sequential_fit_lbfgs.rs`。

`Lbfgs`（optimizer 本体）・`LbfgsLineSearch`（line search 方式選択）は
承認範囲外のまま保留を継続し、多層防御を次の縮小形へ更新した:
正のプローブ doctest（`crates/facade/src/lib.rs::LbfgsHoldDoctestGuard`。
プローブを `{Lbfgs, LbfgsLineSearch}` の 2 型のみへ縮小し、`compat::
Optimizer` enum への variant 追加を検出していた `__fandhe_lbfgs_
variant_probe` は variant 追加自体が承認済みとなったため削除）・
ソース走査（`crates/facade/tests/api_surface.rs::
facade_does_not_reexport_or_declare_lbfgs_items`。`NAMES` から
`LbfgsConfig` を除外）・enum variant 走査は正のガードへ反転
（`crates/facade/tests/api_surface.rs::
compat_optimizer_enum_has_lbfgs_variant`。variant がちょうど 1 個
存在することを固定）。詳細な保留の根拠・承認前の実装設計（事前提示）は
`docs/autodiff-lbfgs-decision.md` §7・§8、承認後の実装記録は §9。

**適用記録（経路 2。イシュー #2502・親 #2500・ルート #2499）**: 上記で
保留を継続していた `Lbfgs`・`LbfgsLineSearch` を、ルート
#2499 一括承認（`docs/autodiff-lbfgs-decision.md` §8 の推奨形）に基づき
`crates/facade/src/optim.rs` の波括弧形 `pub use` 1 行で公開した。
保留ガード（`LbfgsHoldDoctestGuard` と `api_surface.rs` のソース走査 6 項目）
を撤去し、`optim_module_reexports_exactly_expected_surface` の期待集合と
`lbfgs_types_are_reachable_via_facade_only` を正ガードとした。帰結として
`Lbfgs` の inherent `state_dict`／`load_state_dict`／`history_len` も facade から
到達可能になる。`OptimizerStateDict` trait は #2556 で公開済み（`Lbfgs` は対象外のまま。適用記録は上記「適用記録（経路 2。イシュー #2556・#2557）」）。

**#2177（`fit()` の class_weight・sample_weight・validation_split
対応）は経路 2 未適用のまま承認待ちで保留した。** コード変更なし
（`#[cfg(doctest)]` 限定の非公開足場 1 件を除く）。イシュー本文の
承認事項節が `FitConfig` への直接フィールド追加を明記しているが、
公開済み `FitConfig`（`Copy + Eq` derive・`crates/facade/tests/
api_surface.rs` の `PartialEq` 固定テストあり）にそのまま適用すると
0.9.0 公開 API 非破壊契約に反するため、非破壊な代替設計
（`FitConfig::validation_split`・別型 `FitWeights<'a>`・新入口
`fit_with_weights`）を承認依頼用に記録した。`crates/facade/src/
lib.rs::FitWeightingHoldDoctestGuard`（正のプローブ doctest）＋
`crates/facade/tests/api_surface.rs` のテスト（doctest ドリフト検査
2 件・facade 再エクスポート／独自宣言の否定ガード・`FitConfig` の
`Copy + Eq` 維持固定）で機械固定した。詳細な承認事項・意味論・
実装スケッチは `docs/compat-fit-sample-weighting-decision.md` §2〜§7。
→ イシュー #2564 で公開、#2565 で正ガードへ反転済み（本節末尾側の適用記録「イシュー #2564・#2565」を参照）。

**適用記録（#2181）**: AMP（`GradScaler`）の `DeviceParamStore` 常駐更新
への結線は、1 節「Tier 2 に列挙済みの機能」（AMP 行）の実装に該当するため
本節（範囲拡張の手続き）の再適用を要しない。追加した facade 新規公開面
（`Tape::step_device_param_store_amp`／`_adam_amp`／`_adamw_amp` の
委譲メソッド 3 件）は、#1959（`step_device_param_store_adam`／`_adamw`）・
#2175（`step_device_param_store_rmsprop`／`_adagrad`／`_lamb`）と同型の
「生の `fandhe_ai_autodiff::Tape` を取る `DeviceParamStore` の状態機械
メソッドへの回避不能な薄い委譲」であり、AMP（Tier 2・#1625 で 2026-09-12
ユーザー承認済み）とデバイス常駐更新（fit とは独立の Tier 1 相当の既存
経路）の組み合わせにすぎない。新規公開型は追加していない（戻り値は
`bool`）。実装記録は `docs/device-resident-update-design.md` 追補・
`crates/autodiff/src/optim/device_store/amp.rs` モジュール doc を参照。

**#2201（参照モデル定義 API。`Mlp`／`LeNet`）は経路 2 未適用のまま
承認待ちで保留した。** イシュー本文は `Mlp`／`LeNet` 型を facade から
`pub use` する公開面拡張を前提としているが、#2201・親 #2190 のいずれにも
所有者の明示承認コメントが確認できなかったため、`crates/facade/src/` は
一切変更していない。他の保留エントリ（#2198・#2177 等）と異なり、本
issue では公開面へ到達しかねないコード自体を `crates/facade/src/` へ
書いていない（`Mlp`／`LeNet` は `crates/facade/examples/models/`
配下の**利用者コード**として実装し、`compat::Sequential::add_*` の
組み合わせのみで構成した）ため、`HoldDoctestGuard` 方式の否定ガードは
追加していない——ガードで守るべき「facade 側の保留対象コード」自体が
存在しないため（ソース走査の否定ガードのみを追加する設計は #2212 の
レビューで受け入れられなかった前例と同じ判断軸）。承認取得後の移行手順
（`src/models/` への移設・`pub mod models`／`pub use`・本物の doctest
への切り替え）は `docs/reference-models-decision.md` §3.1 を参照。**（2026-10-10 追記: `Mlp`／`LeNet` は #2974 で `fandhe_ai::models` として公開済み。本段落の「未適用」は承認前の記録として残す。下記「適用記録（経路 2。イシュー #2974）」参照）**

**#2202（参照モデル実装。`ResNetBlock`／`ResNet`／`Transformer`）は
経路 2 未適用のまま承認待ちで保留した。** イシュー本文の承認事項節は
3 型の facade `pub use` を前提としているが、#2202・親 #2190 のいずれ
にも所有者の明示承認コメントが確認できなかったため、`crates/facade/
src/` は一切変更していない。対象 item は `ResNetBlock`／`ResNet`／
`Transformer` の 3 型に加え、examples 限定の trait `ReferenceModule`／
`Trainable`（`crates/facade/examples/models/reference_module.rs`。
いずれも保留中の facade `nn::Module`〈#2133〉の代わりに利用者コード側
で用意した代替）。（2026-09-29 追記: facade `nn::Module` は #2338 で公開済み。上記適用記録を参照。#2202 自体の承認状況は変わらない。）#2201（`Mlp`／`LeNet`）と同じ理由により、本 issue
でも facade 公開面へ到達しかねないコード自体を `crates/facade/src/`
へ書いていない（`crates/facade/examples/models/` 配下の**利用者
コード**として実装し `#[path]` で個別に取り込む）ため、`HoldDoctest
Guard` 方式の否定ガードは追加していない——ガードで守るべき「facade 側
の保留対象コード」自体が存在しないため（#2212 のレビューで受け入れ
られなかった前例と同じ判断軸）。doctest の代わりは統合テスト 2 本
（`crates/facade/tests/example_resnet_cifar10.rs`・
`example_transformer_cifar10.rs`）と runnable example（`cargo run -p
fandhe-ai --example main`）。承認取得後の移行手順（`src/models/` への
移設・`pub mod models`／`pub use`・本物の doctest への切り替え・
`api_surface.rs` の期待値更新）は #2201 と同じ（`docs/reference-
models-decision.md` §3.1）で、詳細は同 doc §10.1・§10.2 を参照。

**保留記録（イシュー #2141・親 #2131。#2510 で 7 件を公開済みのため logical 3 件に縮小。logical 3 件も #2596 で公開済み〈下記適用記録〉。以下は履歴）**:
logical 3 種（`logical_and`／`logical_or`／`logical_not`）の facade 公開形（`Var` の
関連関数か facade 直下の関数か）は未決のまま保留（#2594）。実装自体は
`fandhe_ai_autodiff::bool_ops`（内部クレート限定の自由関数モジュール）。
`crates/facade/src/lib.rs::VarBoolOpsHoldDoctestGuard` と `api_surface.rs` の
ソース走査・workspace インベントリで logical 3 件の配置・`Tensor`／`Tape` 上への配置・
モジュール再エクスポートを固定している。詳細は
`docs/autodiff-bool-ops-exposure-decision.md` §0・§6・§6.1 を参照。

**適用記録（経路 2。イシュー #2510・親 #2499・ルート #2499 本文「承認範囲」節の一括承認〈Phase 1〜3 の facade 公開を設計判断記録の推奨形で実装してよい〉に基づく）**:
bool 比較 6 種（`gt_bool`／`ge_bool`／`lt_bool`／`le_bool`／`eq_bool`／`ne_bool`）と
`masked_select` を、設計判断記録（`docs/autodiff-bool-ops-exposure-decision.md` §6）の
推奨形どおり `Var` の委譲メソッド 7 件として公開した（`bool_ops` 自由関数への 1 式委譲・
非微分・tape 非記録）。新規の型・`pub use`・`Op`／`BackendOps`・VJP は追加していない。
保留ガードは承認形のみを許す正ガードへ部分反転した（`workspace_declares_bool_ops_fn_names_in_approved_places_only`
が委譲本体を固定。doctest は logical 3 件の保留を維持）。スコープ外: logical 3 件（#2594）・
微分可能な `masked_select`・GPU 専用カーネル。（#2596 で logical 3 件は下記のとおり公開済み。）

**適用記録（経路 2。イシュー #2596・親 #2594。承認の根拠はルート #2499 のコメント https://github.com/Fandhe-AI/fandhe-ai/issues/2499#issuecomment-6033824965 〈2026-10-07〉の「#2594: `docs/autodiff-bool-ops-exposure-decision.md` §6.2 の推奨案」〉）**:
`logical_and`／`logical_or`／`logical_not` を §6.2 案 B-1 の形で `fandhe_ai` 直下の委譲 `pub fn`
3 件として公開した（`fandhe_ai_autodiff::bool_ops::<name>` への 1 式委譲・エラー型は
`AutodiffError`・新しい `pub mod` なし）。`VarBoolOpsHoldDoctestGuard` は承認形外
（`Var`／`Tensor`／`Tape` 上の配置・`bool_ops` 再エクスポート）を拒むガードとして残し、
ソース走査を `facade_declares_logical_fns_only_as_approved_root_delegations` へ正ガード化した。
`fandhe-ai =0.10.0` の公開 API は追加のみ。実装記録は同 doc §6.3。スコープ外: 微分可能な
`masked_select`・`Var` 入力の logical 版・GPU 専用カーネル（§6 項目 2〜4）。

**適用記録（経路 2。イシュー #2599・親 #2597。承認の根拠はルート #2499 のコメント https://github.com/Fandhe-AI/fandhe-ai/issues/2499#issuecomment-6033824965 〈2026-10-07〉の「#2597: `docs/autodiff-var-dtype-multiplexing-design.md` §4.1・§10.1 の推奨案」〉）**:
f64 専用の独立自動微分グラフを案 D-2 の形で公開した。`fandhe_ai` 直下に facade 所有の newtype `TapeF64<'t>`
（`new`／`var`／`var_no_grad`／`backward`）・`VarF64<'g, 't>`（`Clone + Copy`。`value`／`shape`／`add`／`mul`／`div`／`pow`／`matmul`／
`sum`／`mean`／`max`）・`GradientsF64`（`get`）を `lib.rs` に直接宣言し、公開メソッド 15 件は内部
`fandhe_ai_autodiff::f64_autograd` への 1 式委譲とした（新 `pub mod`・`pub use`・trait impl なし）。
承認形だけを許す正ガードは `api_surface.rs` の `f64_autograd_*` 群（宣言 1 件・メソッド集合と本体の 1 式委譲・再エクスポート不在・
workspace インベントリ・シグネチャプローブ・自己テスト）。着手時点で f64 専用の保留ガードは無く、汎用 2 ガードは変更せず残した。
`fandhe-ai =0.10.0` の公開 API は追加のみ。依存・tolerance・baseline・ガードレール閾値・`docs/spec` は変更していない。
実装記録は `docs/autodiff-var-dtype-multiplexing-design.md` §4.1・§10.1。CUDA／Metal 実機 parity は未実測で
`docs/perf/logs/f64-autograd-facade-2599/README.md` に手順と記入欄を置いた。スコープ外: バッチ matmul・`keepdim`・
多軸縮約・`min`・GPU ネイティブ f64 カーネル・`TypedOps<f64>` 拡張・§10 の項目 1・2・4・5。

> **更新（#2556）**: 下記の保留のうち `OptimizerStateDict` の再エクスポートは #2556 で公開済み
> （`docs/autodiff-optimizer-state-dict-decision.md` §11）。保留ガードは正ガードへ反転した。
> 適用記録は下記「適用記録（経路 2。イシュー #2556・#2557）」。以下の保留記録は履歴として残し、
> 現在も保留なのは `compat::Sequential` の optimizer 状態 API と complete checkpoint のみ。

**適用記録（経路 2。イシュー #2556・#2557・親 #2555・ルート #2499。承認の根拠はルート #2499 の
コメント〈issuecomment-6033824965〉の記録のみで、それ以上の承認はない）**:
`fandhe_ai::optim::OptimizerStateDict` 1 つを、`src/optim.rs` の素の再エクスポート 1 行
（`pub use fandhe_ai_autodiff::nn::optim::OptimizerStateDict;`）で公開した。到達する impl は
`AdamW`・`Adam`・`RmsProp`・`Adagrad`・`Lamb`・`Adadelta`・`Adamax`・`NAdam`・`RAdam`・`Sgd`
の 10 型で、`Lbfgs` は対象外（inherent のまま）。trait は sealing せず、代わりにメソッドを
追加しない契約とした。保留ガード（`OptimizerStateDictHoldDoctestGuard` と 2 テスト・否定ガード）
は #2556 で撤去・反転した。#2557 で次の正ガードを追加した:
`optimizer_state_dict_trait_matches_approved_shape`（trait 形状固定）・
`optimizer_state_dict_impls_are_exactly_approved_ten`（impl 集合固定）・
`optimizer_state_dict_usage_doctest_is_present_and_compiled`（利用例 doctest プローブ）・
`optimizer_state_dict_shape_guards_detect_each_category`（自己検証）。#2556 分は
`facade_reexports_optimizer_state_dict_only_in_approved_form`・
`optimizer_state_dict_is_reachable_via_facade_only`。利用例は `optim.rs` の doctest と
`crates/facade/tests/optim_state_dict_facade.rs`（バイト列経由・ファイルパス経由）。記録は
`docs/autodiff-optimizer-state-dict-decision.md` §11・§12。スコープ外: `compat::Sequential`
の optimizer 状態 API・complete checkpoint・sealing・外部由来 safetensors の入力サイズ上限。

**保留記録（イシュー #2174・親 #2131）**: `fandhe_ai::optim::
OptimizerStateDict` の再エクスポートまたは `AdamW`／`Adam`／
`RmsProp`／`Adagrad`／`Lamb` への inherent `state_dict`／
`load_state_dict` 追加・`compat::Sequential` の optimizer state
取得／復元 API・model と optimizer の complete checkpoint（未起票）は
未承認のまま保留した。`crates/facade/src/lib.rs::
OptimizerStateDictHoldDoctestGuard`（正のプローブ doctest）と
`crates/facade/tests/api_surface.rs` の 4 テストで多層固定している。
詳細は `docs/autodiff-optimizer-state-dict-decision.md` §0・§5 を参照。

**保留記録（イシュー #2189・親 #2131）**: `Tensor<f32>` の NumPy 互換
`.npy`／`.npz` 読み書き（`load_npy`／`save_npy`／`load_npz`／
`save_npz`）・`NpyError` の再エクスポートは未承認のまま保留した。
`crates/facade/src/lib.rs::NpyIoHoldDoctestGuard`（正のプローブ
doctest）と `crates/facade/tests/api_surface.rs` の 4 テストで多層
固定している。詳細は `docs/tensor-core-npy-npz-io-decision.md` §7
を参照。**→ #2590 で下記の適用記録へ置き換わった（承認済み・公開済み）。**

**適用記録（イシュー #2590・親 #2588・ルート #2499 Phase 3。npy・npz の読み書き）**:
ルート #2499 の 2026-10-07 のリポジトリ所有者の承認コメント（`docs/tensor-core-npy-npz-io-decision.md` §10 の推奨案・バイト列版は含めない）に基づき、
`fandhe_ai::interop::npy`（新ファイル `crates/facade/src/interop/npy.rs`）として `NpyError`・`load_npy`・`save_npy`・`load_npz`・`save_npz` の 5 名を
`tensor-core::io` からの純再エクスポートで公開した（facade に型・関数・`impl` を定義しない。別名なし）。
非公開のまま残すもの: バイト列版 4 関数（`read_npy_bytes` 等）、クレートルート直下への名前追加、`Tensor` への inherent メソッド（`Tensor::<f32>::load_npy` 形）。
書き出しの非原子性（`std::fs::write`）と読み込みのパス扱い（symlink を辿る・上限 1 GiB）は現状のままで、モジュール doc に明記した。
ガードは「承認形だけを許す」正ガードへ反転した（`facade_reexports_npy_io_only_from_interop_npy`・`interop_npy_reexports_exactly_expected_surface` 等）。
`NpyIoHoldDoctestGuard` は `Tensor<f32>` への関連関数追加を検出するプローブだけを残す部分反転。
`Cargo.toml`／`Cargo.lock`・依存・`unsafe`・tolerance／baseline・`docs/spec/` は不変。詳細は `docs/tensor-core-npy-npz-io-decision.md` §12。

**適用記録（イシュー #2593・親 #2591・ルート #2499 Phase 3。乱数分布と Generator）**:
ルート #2499 の 2026-10-07 のリポジトリ所有者の承認コメント（`docs/rng-distributions-generator-decision.md` §5.1.3 の推奨案・案 B-1）に基づき、
`fandhe_ai::bernoulli`・`fandhe_ai::multinomial`・`fandhe_ai::normal`（crate ルートの委譲 `pub fn`。本体は `fandhe_ai_autodiff::<name>` への 1 式委譲）と
`fandhe_ai::Generator`（`pub use fandhe_ai_tensor_core::Generator;` 1 行）の 4 名を公開した。新しい `pub mod` は追加していない。
`normal` は `nn::init::normal` と同名のまま共存する（引数順・`std == 0` の乱数消費・エラー型が異なる点は doc に明記）。
非公開のまま残すもの: `Var`／`Tensor` へのメソッド追加、`randn`／`rand`／`randint` の `Generator` 版、`Generator` の状態 get/set。
ガードは「承認形だけを許す」正ガードへ反転した（`facade_declares_rng_distributions_only_as_approved_root_delegations`）。`RngDistributionsHoldDoctestGuard` は `Var`／`Tensor<f32>` への同名メソッド追加を検出するプローブだけを残す部分反転。
`Cargo.toml`／`Cargo.lock`・依存・`unsafe`・tolerance／baseline・`docs/spec/` は不変。詳細は `docs/rng-distributions-generator-decision.md` §5.2。

> **更新（イシュー #2575・#2576）**: 下記の保留は #2575 で承認形どおり公開済みとなり、保留ガード（`GenerateHoldDoctestGuard` と `api_surface.rs` の保留系テスト）は削除・正ガードへ反転済み。下記は着手時点（#2191）の履歴であり、現行の公開形・ガードは直後の適用記録を正とする。

**保留記録（イシュー #2191・親 #2084）**: 自己回帰生成ループ（3 戦略・
KV キャッシュ結線・seed 決定性）の facade 公開（`pub fn generate` 等
の署名・エラー型・`GenerateConfig` 型）は未承認のまま保留した。実装
自体は `fandhe_ai_autodiff::generate`（内部クレート）として完了済み。
`crates/facade/src/lib.rs::GenerateHoldDoctestGuard`（正のプローブ
doctest）とトークン方式の否定ガード（`facade_does_not_reexport_or_
declare_generate_items` 等）で多層固定している。詳細は
`docs/facade-generate-decision.md` §0・§1 を参照。

**適用記録（経路 2。イシュー #2575・#2576・親 #2573・ルート #2499 のコメント〈決定記録 §13.2 推奨案・§13.5 選択肢 A の承認。`issuecomment-6033824965`〉に基づく）**:
自己回帰生成ループを決定記録（`docs/facade-generate-decision.md` §13.2・§17）の確定形どおり公開した（実施は #2575・PR #2839）。公開名は `fandhe_ai::inference::{AutoregressiveModel, GenerateConfig, SamplingStrategy, generate}` の 4 件で、`inference/mod.rs` の純再エクスポート 1 文。エラー型は既存の `AutodiffError` を流用し、facade newtype・`Tape::generate`・`Sequential::generate` は作っていない。新しい `pub mod` はなく、`fandhe-ai =0.10.0` に対して追加のみ。
ガード: #2575 で `GenerateHoldDoctestGuard` と保留系テスト群（走査 2 件・自己テスト・doctest 検査 2 件・固定文言）を削除し、`facade_exposes_generate_items_only_in_approved_shape`（＋自己テスト）・`generate_items_are_reachable_via_facade_inference_path` を導入。#2576 で `generate_usage_doctests_are_present_and_compiled`・`inference_module_reexports_exactly_expected_surface`・`generate_public_shape_matches_approved_inventory`（＋自己テスト `generate_public_shape_inventory_detects_each_category`）・`generate_public_field_types_and_variants_are_pinned` を追加した（全数インベントリ・doctest 実在検査・`inference/mod.rs` の `pub use` 全数固定。検出範囲の限界と workspace 全体インベントリを採らない理由は決定記録 §18.3）。
利用例は `inference/mod.rs` のモジュール doc の doctest と `crates/facade/tests/generate_facade.rs`。Tier 1／Tier 2 表（§1.2・§1.3）への行追加は行っていない（#2822・#2826・#2580 の先例と同じ）。
保留継続: `caches` を使う形の到達経路（決定記録 §17.4。#2573 の未決事項）、EOS 早期停止・top-p・beam search・repetition penalty・トークナイザ（決定記録 §11）。`Cargo.toml`／`Cargo.lock`・tolerance／baseline・`docs/spec`・`unsafe` は不変。CUDA／Metal 実機 parity は未実測で、申し送り先は `docs/perf/logs/generate-2191/README.md`。詳細は決定記録 §17・§18。

> **更新（イシュー #2582・#2583）**: 下記の保留は #2582 で承認形どおり公開済みとなり、保留ガード
> （`PredictBatchesHoldDoctestGuard` と `api_surface.rs` の 4 テスト）は正ガードへ反転済み。
> 下記は着手時点（#2192）の履歴であり、現行の公開形・ガードは直後の適用記録を正とする。

**保留記録（イシュー #2192・親 #2131）**: `Sequential::
predict_batches`・`PhaseMetrics`・`get_phase_metrics`／
`current_phase_metrics`／`reset_phase_metrics`・非公開 `mod inference`
の `pub mod` 昇格は未承認のまま保留した。`crates/facade/src/lib.rs::
PredictBatchesHoldDoctestGuard`（正のプローブ doctest）と
`crates/facade/tests/api_surface.rs` の 4 テストで多層固定している。
詳細は `docs/facade-predict-batches-phase-metrics-decision.md` §0・
§5 を参照。

**適用記録（経路 2。イシュー #2582・#2583・親 #2581・ルート #2499 のコメント〈決定記録 §8 の推奨案での承認。`issuecomment-6033824965`〉に基づく）**:
`Sequential::predict_batches`（#2192 で内部実装済みの DataLoader 反復推論）と推論フェーズ計測を、
決定記録（`docs/facade-predict-batches-phase-metrics-decision.md` §8.4・§10・§11）の確定形どおり公開した。
公開名は 7 件: `fandhe_ai::compat::Sequential::predict_batches` と
`fandhe_ai::inference::{PhaseMetrics, PhaseStat, InferencePhase, PredictBatchInput, get_phase_metrics,
reset_phase_metrics}`（`pub mod inference` を新設し、実体の `batch` は非公開のままフラットに再エクスポート）。
`fandhe-ai =0.10.0` に対して追加のみ（`PhaseMetrics`・`InferencePhase` は `#[non_exhaustive]`、
`PredictBatchInput` は sealed）。集計単位はスレッド単位で、`current_phase_metrics`・プロセス全体集計・
open トレイト・`dyn` 境界は作らない。
正ガード: #2582 で `facade_exposes_predict_batches_items_only_in_approved_shape`（＋自己テスト）・
`workspace_declares_predict_batches_fn_names_only_in_approved_locations`・
`inference_is_public_and_batch_stays_private`・
`predict_batches_public_surface_is_reachable_with_pinned_signatures`、#2583 で全数インベントリ
`inference_phase_variants_are_exactly_approved_four`・`predict_batch_input_impls_are_exactly_approved_three`・
`phase_metrics_and_phase_stat_pub_methods_are_exactly_approved`・`inference_internal_items_stay_crate_private`
（自己テスト `predict_batches_inventory_guards_detect_each_category`）と doctest 存在検査
`predict_batches_usage_doctests_are_present_and_compiled`。利用例は `inference/mod.rs` のモジュール doc・
`Sequential::predict_batches` の doc・`PhaseMetrics` の doc（doctest）と
`crates/facade/tests/inference_predict_batches.rs`。保留を継続する項目: 他ローダー
（`SamplerDataLoader`／`PrefetchDataLoader`／`HookedDataLoader`）向け別名メソッド・プロセス全体集計・
`phase_metrics()` 別名・`DeviceTransfer` の実計測。`Cargo.toml`／`Cargo.lock`・tolerance／baseline・
`docs/spec` は不変、実機申し送りは不要。詳細は決定記録 §10・§11。

| 出典 | 内容 |
|------|------|
| `docs/spec/04-requirements.md:222-236` | REQ-9 概要・受け入れ基準（2026-09-12 改定後）・関連 PoC |
| `docs/spec/04-requirements.md:227` | REQ-9 改定前基準「全方位の API 網羅は対象外」（履歴として保持） |
| `docs/spec/04-requirements.md:228-235` | REQ-9 2026-09-12 追記（Tier 1／Tier 2／引き続き対象外・`amax` 勾配分配の扱い） |
| `docs/spec/04-requirements.md:356` | 除外事項「分散学習・量子化の網羅対応」の 2026-09-12 注記（Tier 2 量子化・DDP の従属関係） |
| `docs/spec/04-requirements.md:432` | REQ-9 2026-09-12 追記に対応する改定履歴表エントリ（イシュー #66） |
| `docs/spec/05-tasks.md:309` | TASK-9.2 の 2026-09-12 追記（対象範囲拡張は実装リポ #1591 以降で扱う旨） |
| `docs/spec/06-roadmap.md:99` | M3 完了条件の 2026-09-12 追記（対象範囲限定記述への参照注記） |
| spec リポ `Fandhe-AI/fandhe-ai-spec` PR #69（`c5cf1ed`） | REQ-9 の 2026-09-12 追記（マージ済み） |
| spec イシュー #66 | REQ-9 改定提案（実装リポ ルート #1570 起票） |
| 実装リポ #1570（ルート） | REQ-9 改定・Tier 1／Tier 2 issue ツリーのルート |
| 実装リポ #1572（Phase 2・Tier 1 親） | 1.2 節の各 issue の親 |
| 実装リポ #1573（Phase 3・Tier 2 親） | 1.3 節の各 issue の親 |
| 実装リポ #1656 | `docs/spec` submodule ポインタ更新（spec PR #69 反映） |
| 実装リポ #1591（本イシュー） | 本文書 §1／§2／§5 の更新 |
| `docs/compat-feature-gap.md` | fandhe-ai 公開面の実装状況スナップショット（対象 HEAD 固定。§3 の「compat-api-scope.md の位置づけ」列は本改定前の状態を記述したまま） |
| 実装リポ #1627 | Tier 2 量子化。除外事項「分散学習・量子化の網羅対応」に従属し実装着手不可。設計記録は `docs/backend-int8-quantization-decision.md`（段階 0・blocked のまま close しない） |
| 実装リポ #1628 | Tier 2 複数 GPU／DDP。同上に従属し設計記録のみ |
| 実装リポ #1633 | sparse／complex テンソルの非対応の明文化。除外事項に従属しない対象外項目。設計記録は `docs/tensor-core-sparse-complex-decision.md` |
| `docs/spec/04-requirements.md:233` | REQ-9「引き続き対象外」列挙（sparse／complex テンソルを含む） |
| `docs/spec/05-tasks.md:299-311` | TASK-9.1（基本 NN モジュール）・TASK-9.2（compat 再実装・対象範囲明文化） |
| `docs/spec/03-poc/poc-v2-6-interop/code/rust/src/mlp.rs` | `Mlp::from_safetensors`（自作コア上の薄い互換層の v2 実例） |
| `docs/public-api-design.md:6,13,556` | compat 層と自作コア素の公開 API の境界記述 |
| `crates/autodiff/src/nn/activation.rs` | TASK-9.1b（#92）実装済みの ReLU／Sigmoid／Tanh |
| `crates/autodiff/src/nn/mod.rs` | compat 層が積む「レイヤー」モジュール群の入口コメント |
| `.claude/rules/coding-rust.md` | 「互換 API 層は自作コアの上の薄いラッパーに徹する（REQ-9）」 |
| `.claude/rules/out-of-scope-tracking.md` | 対象外事項の Issue 追跡規約 |
| イシュー #91（TASK-9.1a・Linear 層） | クローズ済み |
| イシュー #92（TASK-9.1b・基本活性化関数群） | クローズ済み |
| イシュー #94（TASK-9.2・親） | クローズ済み |
| イシュー #95（TASK-9.2a・compat::array／Sequential 実装） | クローズ済み |
| イシュー #96（TASK-9.2b・本文書） | クローズ済み |
| イシュー #410（TASK-9.3・`facade` クレート新設・composition root） | クローズ済み |
| イシュー #411（TASK-9.4・compat 層の facade への移設・サポート境界明文化） | クローズ済み |
| `crates/autodiff/src/compat/`（削除済み） | TASK-9.2a（#95）実装・TASK-9.4（#411）で `crates/facade/src/compat/` へ移設 |
| `crates/facade/src/compat/` | TASK-9.4（#411）移設先の `array`／`Sequential`（現行の compat 公開面） |
| `crates/autodiff/src/nn/module.rs` | TASK-9.2a（#95）実装済みの共通 `Module` trait（現行も `autodiff` 側に残置） |
| `docs/spec/04-requirements.md:237-238` | REQ-9 の 2026-08-08 追記（サポート境界の明文化） |
| `docs/spec/04-requirements.md:239-241` | REQ-9 の 2026-08-29 追記（`optim`・デバイス常駐更新経路を確定入口へ追加） |
| `docs/spec/04-requirements.md:426` | REQ-9 の 2026-08-29 追記に対応する改定履歴表エントリ |
| `docs/spec/05-tasks.md:322` | TASK-9.4 |
| `crates/facade/src/optim.rs` | `fandhe_ai::optim` 公開面（#961・PR #972。4 行の `pub use`） |
| `crates/facade/tests/api_surface.rs` | optim 固有検査（純再エクスポート・昇格元公開面との 1 対 1） |
| `docs/facade-optimizer-promotion-decision.md` §4・§6・§7・§8-3・§9 | 昇格の設計判断・整合確認・spec 改定要否・提案元・改定完了記録 |
| `docs/device-resident-update-design.md` | デバイス常駐更新経路の設計（更新経路・所有権・数値一致契約。#951） |
| イシュー #932（設計判断） | クローズ済み |
| イシュー #954（デバイス常駐更新の実装） | クローズ済み |
| イシュー #955（parity テスト・ベンチ非後退確認） | クローズ済み |
| イシュー #960（親。§8 提案のユーザー承認を受けた起票） | 参照 |
| イシュー #961（`fandhe_ai::optim` 実装） | クローズ済み |
| イシュー #962（本文書 §0 入口列挙の暫定更新〈移行予定扱い〉） | クローズ済み |
| イシュー #963（`Sequential` doc 差し替え） | クローズ済み |
| PR #972 | イシュー #961 の実装 PR |
| イシュー #984（spec 改定提案・正本 spec PR #59 の起票元） | クローズ済み |
| イシュー #985（`docs/spec` submodule ポインタ更新） | クローズ済み |
| イシュー #986（本文書 §0 の暫定注記削除・確定入口統合） | 本イシュー |
| spec リポ `Fandhe-AI/fandhe-ai-spec` PR #59 | REQ-9 の 2026-08-29 追記（マージ済み。merge commit `64364b4bf7e46f91f07d779b2d1c4d14adbd4e48`） |
| PR #988 | `docs/spec` submodule ポインタ更新（イシュー #985 の実装 PR） |

**#2347（ONNX external data のパス入力 import 入口 `onnx::external_data::
build_graph_with_external_data` の facade 公開）は 2026-09-28 にユーザー
承認を得て実施済み。** 新規メソッドの追加ではなく、既存 API
`crates/facade/src/interop/onnx.rs::OnnxModel::from_path` を external
data 対応へ拡張する形（`std::fs::read` → protobuf デコード →
`build_graph_with_external_data`。基点ディレクトリはモデルファイルの
親ディレクトリ、オプションは `ExternalDataOptions::default()`）を採用
した。`OnnxModel::from_bytes` は変更しておらず、external data
（`.onnx.data` companion ファイル）を持つモデルは従来どおり
`OnnxError::InvalidModel` で拒否される（`crates/facade/tests/
interop_onnx_internal_parity.rs::
facade_from_bytes_rejects_external_data_model_from_path_attempts_
resolution` で固定）。`from_path` が実際に external data を解決できる
ことの正例は `crates/facade/tests/interop_onnx_external_data.rs::
from_path_resolves_external_data_and_matches_manifest_reference` を
参照（詳細は `docs/onnx-external-data-decision.md`・`docs/
facade-onnx-import-exposure-decision.md` §15 を参照）。

**適用記録（経路 2。イシュー #2501・親 #2499・ルート #2499 本文「承認範囲」節の一括承認〈Phase 1〜3 の facade 公開を設計判断記録の推奨形で実装してよい〉に基づく）**:
`Adadelta`／`AdadeltaConfig`・`Adamax`／`AdamaxConfig`・`NAdam`／`NAdamConfig`・
`RAdam`／`RAdamConfig`（#2171 で内部クレート限定実装済み）の
`fandhe_ai::optim` からの素の再エクスポートを、設計判断記録
（`docs/autodiff-optimizer-adadelta-adamax-nadam-radam-decision.md` §8）の
推奨形どおりルート #2499 の一括承認に基づき実装した（`crates/facade/src/
optim.rs` の 4 行追加のみ）。保留ガード（`OptimizerExtHoldDoctestGuard` と
対応する `api_surface.rs` の否定テスト）は承認形のみを許す正ガード
（`facade_reexports_optimizer_ext_items_only_in_approved_shape`）へ反転した。
`compat::Optimizer` variant・`DeviceParamStore` 結線・`OptimizerStateDict`／
`ParamGroupStep` の facade 公開は含まない。実装記録は同 decision doc §8.1。

**適用記録（経路 2。イシュー #2553・親 #2551・ルート #2499 のコメント〈設計判断記録 §9 の推奨案での承認〉に基づく）**:
`fandhe_ai::optim::{ParamGroup, ParamGroupStep}`（#2173・#2298 で内部クレート限定実装済み）の
素の再エクスポートと、`compat::Sequential::compile_with_param_groups(&mut self,
Optimizer, Loss, &[ParamGroup])` の追加（計 3 件）を、設計判断記録
（`docs/autodiff-param-groups-decision.md` §9・§12）の推奨形どおり実装した。
`SlotHparams`・`resolve_slot_hparams`・`step_with_slot_hparams` は `pub(crate)` のまま
非公開。保留ガード `ParamGroupsHoldDoctestGuard` は削除し、`api_surface.rs` の
否定テストは承認形のみを許す正ガードへ反転した。AMP 併用 API・`LrSchedule` 併用・
groups の保存形式・`DeviceParamStore` 常駐経路は含まない。実装記録は同 decision doc §12。

**適用記録（イシュー #2554・親 #2551。#2553 の正ガードの仕上げ）**: 公開面は増やしていない
（`crates/facade/src` の差分はコメントのみ）。`api_surface.rs` を `train_step` 節（#2753）と同じ
構成へ仕上げた。正ガードは `facade_param_groups_public_surface_matches_approved_contract`
（鍵 3 種がちょうど 1 件・`param_groups_inventory_violations`）・`param_groups_types_match_approved_shape`
（`ParamGroup`／`ParamGroupStep` の形状）・`param_groups_types_are_reachable_via_facade_only`
（公開済み 10 optimizer の `step_with_groups` と `compile_with_param_groups` のシグネチャ）・
`param_groups_usage_doctests_are_present_and_compiled`（正の doctest プローブ）・
`facade_param_groups_public_surface_guards_detect_each_category`（自己テスト）・
`workspace_declares_param_group_fn_names_only_in_allowed_locations`（不変）。旧名との対応は
同 decision doc §13.3。利用例は `optim.rs` のモジュール doc・`compile_with_param_groups` の doc
（doctest）と `crates/facade/tests/compat_sequential_param_groups.rs`（15 件。#2554 で
`named_parameters_position_is_slot_index_for_groups` を追加）。保留は継続する項目: AMP 併用 API・
`LrSchedule` 併用（案 A）・groups の保存形式・`DeviceParamStore` 常駐経路・`Lbfgs` 併用・
スロット添字の公開ヘルパー。`Cargo.toml`／`Cargo.lock`・tolerance／baseline・`docs/spec` は不変、
実機申し送りは不要。記録は同 decision doc §13。

**適用記録（経路 2。イシュー #2560・#2561・親 #2558・ルート #2499 のコメント〈決定記録 §10 の推奨案での承認。`issuecomment-6033824965`〉に基づく）**:
EMA（#2179 で内部クレート限定実装済み）を、決定記録（`docs/autodiff-ema-decision.md` §10・§13・§14）の
推奨形どおり公開した。公開名は 3 件: `fandhe_ai::optim::ExponentialMovingAverage`（facade 独自の
薄いラッパー。`crates/facade/src/optim_ema.rs`）・`fandhe_ai::compat::EmaCallback`・
`compat::Callback::Ema(EmaCallback)`（`fit_with_callbacks` への結線）。`fandhe-ai =0.10.0` に対して
追加のみ（`Callback` は `#[non_exhaustive]`、`FitConfig`〈`Copy + Eq`〉・`Sequential` は不変）。
正ガード（#2561）: `facade_exposes_ema_only_in_approved_shape`（`EmaCallback` を含む承認形のインベントリ）・
`facade_exposes_ema_only_in_approved_shape_detects_each_category`（自己テスト）・
`ema_types_are_reachable_via_facade_only`・`ema_usage_doctests_are_present_and_compiled`。
`EmaHoldDoctestGuard` は `FitConfig`／`Sequential` への `use_ema`／`ema_decay` 追加という禁止経路専用として
名前を維持（ドリフト検査 2 件は不変。理由は決定記録 §14.3）。利用例は `optim.rs`・`optim_ema.rs`・
`compat/callbacks.rs` の doctest と `crates/facade/tests/compat_sequential_fit_ema.rs`・
`compat_sequential_ema_manual.rs`。保留を継続する項目: `compile_with_amp` 併用（fit 開始前に
`InvalidArgument` で拒否）・decay ウォームアップ・`BatchNorm` running buffer・`DeviceParamStore` 常駐経路・
`FitConfig`／`Sequential` への接続。`Monitor::Loss` の `ModelCheckpoint`／`EarlyStopping` 併用は、
#2844 で決定記録 §15 の形（判定値は生の重みの訓練損失・保存／復元重みは EMA 重み）により受理した
（根拠: ルート #2499 のコメント `https://github.com/Fandhe-AI/fandhe-ai/issues/2499#issuecomment-6052732061`。§16）。
`Cargo.toml`／`Cargo.lock`・tolerance／baseline・`docs/spec` は不変、実機申し送りは不要。
詳細は決定記録 §13・§14。

**適用記録（経路 2。イシュー #2505・親 #2500・ルート #2499 本文「承認範囲」節の一括承認〈Phase 1〜3 の facade 公開を設計判断記録の推奨形で実装してよい〉に基づく）**:
`Sampler`・`SequentialSampler`／`RandomSampler`／`WeightedRandomSampler`・
`SamplerDataLoader`／`SamplerBatches`・`HookedDataLoader`／`HookedBatches`・
`TransformFn`／`CollateFn`・`default_collate`（計 11 名。#2182 で内部クレート
限定実装済み）の `fandhe_ai::data` からの素の再エクスポートを、設計判断記録
（`docs/tensor-core-data-sampler-hooks-decision.md` §5）の推奨形どおり実装
した（`crates/facade/src/data.rs` の `pub use` 4 行追加のみ。別名・facade
独自の型／関数は持たない）。保留ガード（`DataHooksHoldDoctestGuard` と
doctest ドリフト検査 2 件）は削除し、否定ガード
`facade_does_not_reexport_or_declare_data_hooks` は承認形のみを許す正ガード
（`facade_reexports_data_hooks_items_only_in_approved_shape`）へ反転した。
定義元インベントリは維持する。`DataLoader`／`DataLoaderConfig` への統合・
`DistributedSampler`・prefetch 系（#2506 で別途公開）の facade 公開は含まない。
実装記録は同 decision doc §8。

**適用記録（経路 2。イシュー #2506・親 #2500・ルート #2499 本文「承認範囲」節の一括承認に基づく）**:
`PrefetchConfig`・`PrefetchDataLoader`・`PrefetchBatches`（#2183 で内部クレート
限定実装済み）の `fandhe_ai::data` からの素の再エクスポートを、設計判断記録
（`docs/tensor-core-data-prefetch-decision.md` §4・§8）のとおり実装した
（`crates/facade/src/data.rs` の `pub use` 1 行追加のみ）。否定ガードは承認形のみを
許す正ガード `facade_reexports_prefetch_items_only_in_approved_shape` へ反転し、
定義元インベントリは維持する。`Sequential::fit` への結線（#2603。→ 下記 #2605 で実施）・rayon 化・
`PREFETCH_MAX_*` 定数の公開は含まない。実装記録は同 decision doc §4。

**適用記録（経路 2。イシュー #2605・親 #2603・ルート #2499 のコメント〈2026-10-07、`issuecomment-6033824965`〉に基づく）**:
`PrefetchDataLoader` の fit 結線として `compat::Sequential::fit_with_prefetch`
（inherent メソッド 1 件。`fit_with_metrics` の引数 + `prefetch: PrefetchConfig`）を
`crates/facade/src/compat/training.rs` に公開した（`docs/tensor-core-data-prefetch-decision.md`
§4 案 B・「#2605 実装記録」）。`fit_with_metrics` と bit 一致し、適用は学習ローダーのみ。
案 A（`FitConfig` ビルダー）・案 C/D/E・rayon 化・`PREFETCH_MAX_*` の公開は含まない。保留ガードは
存在しなかったため、正ガード `facade_exposes_fit_with_prefetch_only_in_approved_shape`（自己テスト・
`workspace_declares_fit_with_prefetch_only_in_facade_training`・
`fit_with_prefetch_is_reachable_via_facade_only` を含む）を新設した。

**適用記録（経路 2。イシュー #2503・親 #2499・ルート #2499 本文「承認範囲」節の一括承認〈Phase 1〜3 の facade 公開を設計判断記録の推奨形で実装してよい〉に基づく）**:
`MultiStepLr`・`CosineAnnealingWarmRestarts`・`CyclicLr`・`LambdaLr`・
`SequentialLr`（#2176 で内部クレート限定実装済み）の `fandhe_ai::optim` からの
素の再エクスポートを、設計判断記録（`docs/autodiff-lr-scheduler-ext-decision.md`
§8）の推奨形どおりルート #2499 の一括承認に基づき実装した（`crates/facade/src/
optim.rs` の `pub use` 2 行追加のみ。rustfmt の折り返しを避けるための分割）。
保留ガード（`LrSchedulerExtHoldDoctestGuard` と対応する `api_surface.rs` の
否定テスト）は承認形のみを許す正ガード
（`facade_reexports_lr_scheduler_ext_items_only_in_approved_shape`）へ反転した。
`CyclicLr` の triangular2／exp_range・momentum cycling、`LambdaLr` の param group
ごとの lambda は含まない。実装記録は同 decision doc §8.1。

**適用記録（経路 2。イシュー #2507・親 #2499・ルート #2499 本文「承認範囲」節の一括承認〈Phase 1〜3 の facade 公開を設計判断記録の推奨形で実装してよい〉に基づく）**:
決定論モード（#2157 で内部クレート限定実装済み）を、設計判断記録
（`docs/autodiff-determinism-mode-design.md` §6・§7）の推奨形どおり crate ルートの
委譲 `pub fn` 2 件（`fandhe_ai::set_deterministic`／`fandhe_ai::is_deterministic`）
として公開した。新規の型・`pub use`・`Op`／`BackendOps` は追加していない。
保留ガード（`DeterminismHoldDoctestGuard` と対応する `api_surface.rs` の否定テスト）は
承認形のみを許す正ガード
（`facade_declares_determinism_fns_only_as_approved_root_delegations`）へ反転した。
no-op 契約・CPU 限定の保証範囲は不変（GPU の決定性は未検証）。

**適用記録（経路 2。イシュー #2508・親 #2500・ルート #2499 本文「承認範囲」節の一括承認〈Phase 1〜3 の facade 公開を設計判断記録の推奨形で実装してよい〉に基づく）**:
勾配累積（#2180 で内部実装済み）を、設計判断記録
（`docs/compat-grad-accumulation-decision.md` §5）の推奨形どおり
`compat::FitConfig::accumulate_steps(mut self, n: u32) -> Self`（`shuffle`／`drop_last` と
同型のビルダー）として公開した。新規公開面はこの `pub fn` 1 件のみで、フィールドは非公開・
`Copy + Eq` も維持している（追加 API のみ。`fandhe-ai =0.10.0` の公開 API は非破壊）。
保留ガード（`GradAccumulationHoldDoctestGuard` と `api_surface.rs` の否定テスト 3 件）は
承認形のみを許す正ガード（`facade_declares_fit_config_accumulate_steps_exactly_once`・
`facade_does_not_declare_with_accumulate_steps_for_test`・
`fit_config_accumulate_steps_is_reachable_via_facade_only`）へ反転し、テスト専用セッター
`with_accumulate_steps_for_test` は削除した。`Cargo.toml`／`Cargo.lock`・tolerance／baseline・
`docs/spec/` は不変。GPU 経路には触れない（ホスト側 f32 加算のみ）。

**適用記録（経路 2。イシュー #2504・親 #2500・ルート #2499 本文「承認範囲」節の一括承認〈Phase 1〜3 の facade 公開を設計判断記録の推奨形で実装してよい〉に基づく）**:
`nn::init` 初期化関数群（#2140 で内部クレート限定実装済み）を、設計判断記録
（`docs/facade-nn-init-exposure-decision.md` §2.1・§4）の推奨形どおり
`fandhe_ai::nn::init` の純再エクスポートモジュールとして公開した（初期化関数 9 個＋
補助 4 個〈`FanMode`／`Nonlinearity`／`calculate_gain`／`calculate_fan_in_and_fan_out`〉の
計 13 名。明示列挙・glob／別名なし）。新規の `Op`／`BackendOps`／VJP は追加していない
（追加 API のみ。`fandhe-ai =0.10.0` の公開 API は非破壊）。
保留ガード（`api_surface.rs::facade_does_not_reexport_nn_init`）は承認形のみを許す正ガード
（`facade_reexports_nn_init_items_only_in_approved_shape`）へ反転した。
`tensor_core::rng::normal`（#2591 の保留）とは別機能の同名のため、保留は弱めず経路限定で共存させた
（`docs/rng-distributions-generator-decision.md` §5）。`Cargo.toml`／`Cargo.lock`・tolerance／
baseline・`docs/spec/` は不変。GPU 経路には触れない（ホスト側生成のみ）。

**適用記録（経路 2。イシュー #2509・親 #2500・ルート #2499 本文「承認範囲」節の一括承認〈Phase 1〜3 の facade 公開を設計判断記録の推奨形で実装してよい〉に基づく）**:
`compat::Loss` へ L1・Bce・BceWithLogits・Nll・KlDiv・Huber・SmoothL1 の 7 unit variant を、
設計判断記録（`docs/facade-compile-loss-variants-decision.md` §2・§4）の推奨形どおり追加した。
新規公開面はこの 7 variant のみで、新規 `pub fn`／`pub use` はない（`Copy + Eq` 維持・
`fandhe-ai =0.10.0` の公開 API は非破壊）。`L1` は内部クレートの `loss_ops::l1_loss` を
非 `pub` の `use` で結線しただけで、`loss_ops` の公開保留は維持している。
保留ガード（`CompileLossVariantsHoldDoctestGuard` と `api_surface.rs` の否定テスト）は
承認形 9 種を固定する正ガード（`compat_loss_enum_variants_are_exactly_approved_set`）へ反転した。
`model_io` の `compiled.loss` 文字列 allowlist も 9 種へ拡張した（`format_version` は据え置き。
決定記録 §2.5）。fit／evaluate は CPU 固定 tape のため GPU 経路には触れない。
`Cargo.toml`／`Cargo.lock`・tolerance／baseline・`docs/spec/` は不変。

**適用記録（経路 2。イシュー #2511・親 #2500・ルート #2499 本文「承認範囲」節の一括承認〈Phase 1〜3 の facade 公開を設計判断記録の推奨形で実装してよい〉に基づく）**:
`repeat`／`tile`／`flip`／`roll`（#2143 で内部クレート限定実装済み）を、設計判断記録
（`docs/autodiff-rearrange-ops-decision.md` §2.1 案 A・§6）の推奨形どおり
`Var::flip`／`roll`／`repeat`／`tile` の薄い委譲メソッド 4 件として公開した（本体は
`rearrange_ops` 自由関数への 1 行委譲。`rearrange_ops` モジュール自体は facade から再エクスポート
しない。追加 API のみ。`fandhe-ai =0.10.0` の公開 API は非破壊）。新規の `Op`／`BackendOps`／VJP は
追加していない。
保留ガード（`VarRearrangeOpsHoldDoctestGuard` と `api_surface.rs` の否定テスト 4 件）は承認形のみを
許す正ガード（`facade_does_not_reexport_or_declare_rearrange_ops`・
`workspace_declares_rearrange_ops_fn_names_only_in_approved_locations`・
`var_rearrange_ops_methods_are_thin_delegations`・`var_rearrange_ops_are_reachable_via_facade_only`）へ
反転した。`roll` の `dims=None`・GPU 専用カーネルは対象外のまま。`Cargo.toml`／`Cargo.lock`・
tolerance／baseline・`docs/spec/` は不変。実機 parity は #2143 の申し送り
（`docs/perf/logs/shape-repeat-tile-flip-roll-2143/README.md`）が有効。

**適用記録（経路 2。イシュー #2521・親 #2520・ルート #2499 本文「承認範囲」節の一括承認に基づく）**:
`ConvTranspose1d`／`Unflatten`（#2159 で内部クレート限定実装済み）を、設計判断記録
（`docs/autodiff-spatial-layers-decision.md` §6）の推奨形どおり
`compat::Sequential::add_conv_transpose1d`／`add_unflatten` と `Var::conv_transpose1d`／`Var::unflatten`
（薄い委譲メソッド）の 4 名として公開した（forward 本体は `autodiff::nn` 内 `pub(crate)` の共有関数へ集約。
型の再エクスポート・自由関数での公開は行わない。追加 API のみで `fandhe-ai =0.10.0` の公開 API は非破壊）。
新規の `Op`／`BackendOps`／VJP は追加していない。学習経路（`bind`／`trainable_*`／`apply_parameters`）へ
結線し、`ConvTranspose1d` の常駐経路は `Unsupported`、`save_model` は両層とも `UnsupportedModel` で
fail-closed（manifest 未対応。決定記録 §6 実装記録）。保留ガードは該当名のみ縮小し、承認形だけを許す正ガード
4 件を新設した。`Cargo.toml`／`Cargo.lock`・tolerance／baseline・`docs/spec/` は不変。
残り 3 層は #2522 の担当。実機 parity は #2159 の申し送りが有効。

**適用記録（経路 2。イシュー #2515・親 #2500・ルート #2499 本文「承認範囲」節の一括承認〈Phase 1〜3 の facade 公開を設計判断記録の推奨形で実装してよい〉に基づく）**:
`eigh`／`slogdet`／`pinv`／`matrix_rank`／`lstsq`（#2150 で内部クレート限定実装済み）を、設計判断記録
（`docs/autodiff-linalg-ops-decision.md` §0・§6）の推奨形どおり
`Var::eigh`／`slogdet`／`pinv`／`matrix_rank`／`lstsq` の薄い委譲メソッド 5 件として公開した（本体は
`linalg_ops` 自由関数への 1 行委譲。`linalg_ops` モジュール自体は
facade から再エクスポートしない（`EighVars`／`SlogdetVars` 型は 2026-10-07 承認で再エクスポート）。追加 API のみ。`fandhe-ai =0.10.0` の公開 API は非破壊）。
新規の `Op`／`BackendOps`／VJP は追加していない。
保留ガード（`VarLinalgOpsHoldDoctestGuard` と `api_surface.rs` の否定テスト 4 件）は承認形のみを
許す正ガード（`facade_does_not_reexport_or_declare_linalg_ops`・
`workspace_declares_linalg_ops_fn_names_only_in_approved_locations`・
`var_linalg_ops_methods_are_thin_delegations`・`var_linalg_ops_are_reachable_via_facade_only`）へ
反転した。型 `EighVars`／`SlogdetVars` の再エクスポート（`QrVars` と揃える形）は 2026-10-07 承認（issue #2515 コメント）で
追加済み（決定記録 §6）。
GPU 専用カーネル（#2672）も対象外。`Cargo.toml`／`Cargo.lock`・tolerance／baseline・`docs/spec/` は不変。
実機 parity は #2150 の申し送り（`docs/perf/logs/linalg-ops-2150/README.md`）が有効。

**適用記録（経路 2。イシュー #2516・親 #2500・ルート #2499 本文「承認範囲」節の一括承認〈Phase 1〜3 の facade 公開を設計判断記録の推奨形で実装してよい〉に基づく）**:
`mish`／`hardtanh`／`relu6`／`prelu`／`glu`（#2146 で内部クレート限定実装済み）を、設計判断記録
（`docs/autodiff-activation-ops-decision.md` §2.1・§6）の推奨形どおり
`Var::mish`／`hardtanh(&self, min, max)`／`relu6`／`prelu(&self, weight)`／`glu(&self, dim)` の
薄い委譲メソッド 5 件として公開した（本体は `activation_ops` 自由関数への 1 行委譲。
`activation_ops` モジュール自体は facade から再エクスポートしない。追加 API のみ。
`fandhe-ai =0.10.0` の公開 API は非破壊）。新規の `Op`／`BackendOps`／VJP は追加していない。
保留ガードは部分反転（`VarActivationOpsHoldDoctestGuard` は `Var` プローブのみ撤去して維持し、
`Tensor<f32>`／`Tape` 上の配置・モジュール再エクスポートの拒否を継続。`compat::Sequential::add_*` 5 種は
#2529 で公開済み）。`api_surface.rs` は正ガード
（`var_activation_ops_methods_are_thin_delegations`・`var_activation_ops_are_reachable_via_facade_only`・
`workspace_declares_activation_ops_fn_names_only_in_approved_locations`）を追加・改名し、
`facade_does_not_reexport_or_declare_activation_ops` を維持した。GPU 専用カーネルは対象外のまま。
`Cargo.toml`／`Cargo.lock`・tolerance／baseline・`docs/spec/` は不変。実機 parity は #2146 の申し送り
（`docs/perf/logs/activation-ops-2146/README.md`）が有効。

**適用記録（経路 2。イシュー #2513・親 #2500・ルート #2499 本文「承認範囲」節の一括承認〈Phase 1〜3 の facade 公開を設計判断記録の推奨形で実装してよい〉に基づく）**:
`tril`／`triu`／`diag`／`trace`／`outer`／`dot`（#2144 で内部クレート限定実装済み）を、設計判断記録
（`docs/autodiff-matrix-ops-decision.md` §2.1 案 A・§6）の推奨形どおり
`Var::tril`／`triu`／`diag`／`trace`／`outer`／`dot` の薄い委譲メソッド 6 件として公開した（本体は
`matrix_ops` 自由関数への 1 行委譲。`matrix_ops` モジュール自体は facade から再エクスポート
しない。追加 API のみ。`fandhe-ai =0.10.0` の公開 API は非破壊）。新規の `Op`／`BackendOps`／VJP は
追加していない。
保留ガード（`VarMatrixOpsHoldDoctestGuard` と `api_surface.rs` の否定テスト 4 件）は承認形のみを
許す正ガード（`facade_does_not_reexport_or_declare_matrix_ops`・
`workspace_declares_matrix_ops_fn_names_only_in_approved_locations`・
`var_matrix_ops_methods_are_thin_delegations`・`var_matrix_ops_are_reachable_via_facade_only`）へ
反転した。rank 3 以上の `trace`／`diag`・GPU 専用カーネルは対象外のまま。`Cargo.toml`／`Cargo.lock`・
tolerance／baseline・`docs/spec/` は不変。実機 parity は #2144 の申し送り
（`docs/perf/logs/shape-matrix-ops-2144/README.md`）が有効。

**適用記録（経路 2。イシュー #2512・親 #2500・ルート #2499 本文「承認範囲」節の一括承認〈Phase 1〜3 の facade 公開を設計判断記録の推奨形で実装してよい〉に基づく）**:
`floor`／`ceil`／`round`／`sign`／`reciprocal`／`rsqrt`／`erf`／`pow_scalar`（#2145 で内部クレート限定実装済み）を、
設計判断記録（`docs/autodiff-scalar-unary-ops-decision.md` §2 案 C・§9）の推奨形どおり
`Var::floor`／`ceil`／`round`／`sign`／`reciprocal`／`rsqrt`／`erf`／`pow_scalar(&self, exponent: f32)` の
薄い委譲メソッド 8 件として公開した（本体は `scalar_unary_ops` 自由関数への 1 行委譲。
`scalar_unary_ops` モジュール自体は facade から再エクスポートしない。追加 API のみ。
`fandhe-ai =0.10.0` の公開 API は非破壊）。新規の `Op`／`BackendOps`／VJP は追加していない。
保留ガード（`VarScalarUnaryOpsHoldDoctestGuard` と `api_surface.rs` の否定テスト 4 件）は承認形のみを
許す正ガード（`facade_does_not_reexport_or_declare_scalar_unary_ops`・
`workspace_declares_scalar_unary_ops_fn_names_only_in_approved_locations`・
`var_scalar_unary_ops_methods_are_thin_delegations`・`var_scalar_unary_ops_are_reachable_via_facade_only`）へ
反転した。GPU 専用カーネル・`create_graph`（#2543）は対象外のまま。`Cargo.toml`／`Cargo.lock`・
tolerance／baseline・`docs/spec/` は不変。実機 parity は #2145 の申し送り
（`docs/perf/logs/scalar-unary-ops-2145/README.md`）が有効。

**適用記録（経路 2。イシュー #2514・親 #2500・ルート #2499 本文「承認範囲」節の一括承認〈Phase 1〜3 の facade 公開を設計判断記録の推奨形で実装してよい〉に基づく）**:
`prod`／`logsumexp`／`any`／`all`／`norm_p`（#2147）と `amax`／`amin`（#2154）を、設計判断記録
（`docs/autodiff-reduce-ops-decision.md` §0・§8、`docs/autodiff-amax-grad-distribution-decision.md` §9.2）の推奨形どおり
`Var::prod`／`logsumexp`／`any`／`all`（いずれも `(&self, dim: Option<usize>)`）・`Var::norm_p(&self, p: f32, dim: Option<usize>)`・
`Var::amax`／`amin(&self, dim: Option<usize>)` の薄い委譲メソッド 7 件として公開した（本体は `reduce_ops`／`extremum_ops`
自由関数への 1 行委譲。両モジュール自体は facade から再エクスポートしない。追加 API のみ。`fandhe-ai =0.10.0` の公開 API は非破壊）。
新規の `Op`／`BackendOps`／VJP は追加していない。保留ガード（`VarReduceOpsHoldDoctestGuard`・`VarExtremumOpsHoldDoctestGuard` と
`api_surface.rs` の否定テスト各 4 件）は承認形のみを許す正ガード 8 件（`facade_does_not_reexport_or_declare_reduce_ops`・
`facade_does_not_reexport_or_declare_extremum_ops`・`workspace_declares_reduce_ops_fn_names_only_in_approved_locations`・
`workspace_declares_extremum_ops_fn_names_only_in_approved_locations`・`var_reduce_ops_methods_are_thin_delegations`・
`var_reduce_ops_are_reachable_via_facade_only`・`var_extremum_ops_methods_are_thin_delegations`・
`var_extremum_ops_are_reachable_via_facade_only`）へ反転した。対象外: bool 出力版 `any`／`all`（#2141）・`p ∈ {0, ±inf, 負}` のノルム・
GPU 専用カーネル・checkpoint／`create_graph` 対応・`amax_dims`／keepdim 版。`Cargo.toml`／`Cargo.lock`・tolerance／baseline・
`docs/spec/` は不変。実機 parity は #2147・#2154 の申し送り（`docs/perf/logs/reduce-ops-2147/README.md`・
`docs/perf/logs/amax-amin-2154/README.md`）が有効。

**適用記録（経路 2。イシュー #2517・親 #2500・ルート #2499 本文「承認範囲」節の一括承認〈Phase 1〜3 の facade 公開を設計判断記録の推奨形で実装してよい〉に基づく）**: `docs/autodiff-einsum-batch-decision.md` §3 の推奨形（案 A）に従い、`Var::einsum` の受理範囲を batch 添字付き 2 項縮約（例 `"bij,bjk->bik"`。rank≥3 `Var::matmul`〈`gemm_batched`〉への分解）へ拡張した。新しい公開名はなく、従来 `Err` だった入力が `Ok` になるだけの非破壊拡張（シグネチャ不変・既受理入力は bit 同一。`fandhe-ai =0.10.0` の公開 API は非破壊）。`einsum_batch::einsum_batched` は公開済み 0.10.0 互換のため維持（`Var::einsum` と同一挙動の 1 行委譲）、内部の `BatchContraction` モードは撤去した。新規の `Op`／`BackendOps`／VJP／GPU カーネルは追加していない。保留ガード（`VarEinsumBatchHoldDoctestGuard` と `api_surface.rs` の否定テスト 2 件）は削除し、承認形のみを許す正ガード 4 件（`facade_does_not_reexport_or_declare_einsum_batch`・`workspace_declares_einsum_batched_fn_only_in_autodiff_einsum_batch`・`var_einsum_and_einsum_batched_are_thin_delegations`・`var_einsum_batch_contraction_is_reachable_via_facade_only`）へ反転した。対象外: GPU 専用カーネル・ellipsis・3 オペランド以上・size-1 broadcast・複数 batch 添字の bit 同一契約・create_graph 下の rank≥3 `MatMul`。`Cargo.toml`／`Cargo.lock`・tolerance／baseline・`docs/spec/` は不変。実機 parity は #2149 の申し送り（`docs/perf/logs/einsum-batch-2149/README.md`）が有効。

**適用記録（経路 2。イシュー #2518・親 #2500・ルート #2499 本文「承認範囲」節の一括承認〈Phase 1〜3 の facade 公開を設計判断記録の推奨形で実装してよい〉に基づく）**:
`advanced_indexing`・`index_put`・`index_put_`（#2148）を、設計判断記録（`docs/autodiff-indexing-inplace-design.md` §0・§5・§6）の形どおり
`Var::advanced_indexing(&self, indices: &[Tensor<i32>])`・`Var::index_put(&self, indices, values: &Var<'t>, accumulate: bool)`・
`Var::index_put_(&mut self, indices, values: &Var<'t>, accumulate: bool)` の薄い委譲メソッド 3 件として公開した（本体は `indexing_ops`
自由関数への 1 行委譲。同モジュール自体は facade から再エクスポートしない。追加 API のみ。`fandhe-ai =0.10.0` の公開 API は非破壊）。
`index_put_` は `Var` が `Copy` なハンドルであるため再束縛の糖衣であり、同じノードを指す他のコピーは古い値のまま残る。
新規の `Op`／`BackendOps`／VJP／GPU カーネルは追加していない。保留ガード（`VarIndexingOpsHoldDoctestGuard` と `api_surface.rs` の否定テスト 2 件・
固定文言）は削除し、承認形のみを許す正ガード（`facade_does_not_reexport_or_declare_indexing_ops`・
`workspace_declares_indexing_ops_fn_names_only_in_approved_locations`・`var_indexing_ops_methods_are_thin_delegations`・
`var_indexing_ops_are_reachable_via_facade_only`）へ反転した。対象外: 負の添字の wrap-around（設計判断記録 §6 承認事項 2）・
`BackendOps` の拡張と GPU 専用カーネル（同 承認事項 3）・`Tensor`／`Tape` への同名メソッド。`Cargo.toml`／`Cargo.lock`・tolerance／baseline・
`docs/spec/` は不変。実機 parity は #2148 の申し送り（`docs/perf/logs/indexing-inplace-2148/README.md`）が有効。

**適用記録（経路 2。イシュー #2519・親 #2500・ルート #2499 本文「承認範囲」節の一括承認〈Phase 1〜3 の facade 公開を設計判断記録の推奨形で実装してよい〉に基づく）**:
`topk_with_options`・`unique_with_options`・`unique_consecutive`（#2153）を、設計判断記録（`docs/autodiff-topk-unique-ops-decision.md` §0・§6）の形どおり
`Var::topk_with_options(&self, k: usize, opts: TopkOptions)`・`Var::unique_with_options(&self, opts: UniqueOptions)`・
`Var::unique_consecutive(&self, opts: UniqueOptions)` の薄い委譲メソッド 3 件として公開した（本体は `topk_unique_ops` 自由関数への 1 行委譲。
同モジュール自体は facade から再エクスポートしない。追加 API のみ。`fandhe-ai =0.10.0` の公開 API は非破壊）。入出力型 `TopkOptions`・`UniqueOptions`・
`UniqueOutput` は autodiff ルート経由で facade ルートへ 1 行 `pub use` した（§6 の「再エクスポート」を既存規約で具体化したもの）。
新規の `Op`／`BackendOps`／VJP／GPU カーネルは追加していない。保留ガード（`VarTopkUniqueOpsHoldDoctestGuard` と `api_surface.rs` の否定テスト 2 件・固定文言）は削除し、
承認形のみを許す正ガード（`facade_does_not_reexport_or_declare_topk_unique_ops`・`workspace_declares_topk_unique_ops_fn_names_only_in_approved_locations`・
`var_topk_unique_ops_methods_are_thin_delegations`・`var_topk_unique_ops_are_reachable_via_facade_only`・`facade_reexports_topk_unique_types_only_in_approved_shape`）へ反転した。
対象外: `sorted=false` 専用カーネル・CUDA／Metal の `unique_ext` 専用カーネル・`Tensor`／`Tape` への同名メソッド。`Cargo.toml`／`Cargo.lock`・tolerance／baseline・
`docs/spec/` は不変。実機 parity は #2153 の申し送り（`docs/perf/logs/topk-unique-2153/README.md`）が有効。

**適用記録（経路 2。イシュー #2522・親 #2500・ルート #2499 本文「承認範囲」節の一括承認に基づく）**:
`Upsample`・`ZeroPad2d`・`Identity`（#2159）を、設計判断記録（`docs/autodiff-spatial-layers-decision.md` §6・§8）の形どおり
`compat::Sequential::add_upsample(size, mode)`（size 指定のみ）・`add_zero_pad2d([left, right, top, bottom])`・`add_identity()` の `pub fn` 3 件として公開した
（追加 API のみ。`fandhe-ai =0.10.0` の公開 API は非破壊）。新規公開型・`nn::Module` への `as_*` フック追加はない。
3 層は無状態層で学習経路を素通しし、常駐経路は `Flatten` 先例どおり対応扱い、`save_model`／`load_model` は kind 3 種を追加して対応した（`format_version` 不変）。
保留ガード `SpatialLayersHoldDoctestGuard` はメソッドプローブのみ撤去（型名・`add_conv_transpose1d`／`add_unflatten`・`Var` プローブは維持）し、
`api_surface.rs` に正ガード `compat_sequential_exposes_spatial_layer_add_methods_issue_2522` を追加した。
保留継続: `scale_factor` 指定・型の再エクスポート・ONNX export 対応・GPU 専用カーネル。`Cargo.toml`／`Cargo.lock`・tolerance／baseline・`docs/spec/` は不変。
実機 parity は `docs/perf/logs/compat-sequential-spatial-2522/README.md` へ申し送り。

**適用記録（経路 2。イシュー #2523・親 #2520・ルート #2499 本文「承認範囲」節の一括承認に基づく）**:
`ConvTranspose2d`（#2067）を `compat::Sequential::add_conv_transpose2d(in_channels, out_channels, kernel_size, stride, padding, output_padding, dilation, groups, seed)`
（`[usize; 2]` 引数・bias あり固定）の `pub fn` 1 件として公開した（追加 API のみ。`fandhe-ai =0.10.0` の公開 API は非破壊）。
具体シグネチャは `docs/conv-ops-design.md` に記載がなかったため `add_conv2d` から機械的に導出した（§15「#2523 実装記録」）。
学習経路（`bind`／`trainable_*`／`apply_parameters`）へ結線し、常駐経路は `BackendError::Unsupported` で fail-closed、
`save_model`／`load_model` は kind `conv_transpose2d` を追加して対応した（`format_version` 不変）。
保留ガードは存在せず反転対象なし。`api_surface.rs` に正ガードを新設した。
保留継続: `nn::ConvTranspose2d` 型の再エクスポート・bias なし構成・AMP 低精度 forward・`output_padding >= stride`・ONNX export 対応・GPU 専用カーネル。
`Cargo.toml`／`Cargo.lock`・tolerance／baseline・`docs/spec/` は不変。実機 parity は `docs/perf/logs/compat-sequential-conv-transpose2d-2523/README.md` へ申し送り。

**適用記録（経路 2。イシュー #2524・親 #2520・ルート #2499 本文「承認範囲」節の一括承認に基づく）**:
`Conv3d`（#2158）を `Var::conv3d(weight, bias, stride, padding, dilation, groups)`（`[usize; 3]` 引数。`conv3d_ops::conv3d` への 1 式委譲）と
`compat::Sequential::add_conv3d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, seed)`（bias あり固定）の
`pub fn` 各 1 件として公開した（追加 API のみ。`fandhe-ai =0.10.0` の公開 API は非破壊）。
具体シグネチャは `docs/conv-ops-design.md` に記載がなかったため `Var::conv2d`／`add_conv2d` から機械的に導出した（§16.7）。
学習経路へ結線し、常駐経路は `BackendError::Unsupported` で fail-closed、`save_model`／`load_model` は kind `conv3d` を追加して対応した（`format_version` 不変）。
`VarConv3dHoldDoctestGuard` は承認した 2 形のプローブだけを外して縮小し、承認外の形（`conv3d_ops` 再エクスポート・`Tensor`／`Tape` の `conv3d`）の衝突プローブは残した。
保留継続: `conv3d_ops`／`nn::Conv3d` 型の再エクスポート・bias なし構成・AMP 低精度 conv3d・ONNX export 対応・GPU 専用カーネル。
`Cargo.toml`／`Cargo.lock`・tolerance／baseline・`docs/spec/` は不変。実機 parity は `docs/perf/logs/compat-sequential-conv3d-2524/README.md` へ申し送り。

**適用記録（経路 2。イシュー #2525・親 #2520・ルート #2499 本文「承認範囲」節の一括承認に基づく）**:
`GroupNorm`／`InstanceNorm`（#2066）を `compat::Sequential::add_group_norm(groups, eps)`・`add_instance_norm(eps)` の
`pub fn` 各 1 件として公開した（追加 API のみ・新規公開型なし。`fandhe-ai =0.10.0` の公開 API は非破壊）。
具体シグネチャは `docs/norm-ops-design.md` に記載がなかったため内部コンストラクタから機械的に導出した（§11.x）。
affine・`num_channels` 引数は持たない。学習経路へ結線し、常駐経路は `BackendError::Unsupported` で fail-closed、
`save_model`／`load_model` は kind `group_norm`／`instance_norm` を追加して対応した（`format_version` 不変）。ONNX export は `UnsupportedLayer`。
保留ガードは存在しなかったため、`api_surface.rs` に承認形だけを許す正ガード 4 件を新設した。
保留継続: affine 付き構成・型／既定 eps 定数の再エクスポート・常駐経路対応・ONNX export 対応・GPU 専用カーネル。
`Cargo.toml`／`Cargo.lock`・tolerance／baseline・`docs/spec/` は不変。実機 parity は `docs/perf/logs/compat-sequential-group-instance-norm-2525/README.md` へ申し送り。

**適用記録（経路 2。イシュー #2526・親 #2520・ルート #2499 本文「承認範囲」節の一括承認に基づく）**:
`PixelShuffle`／`PixelUnshuffle`（#2162）を `compat::Sequential::add_pixel_shuffle(upscale_factor)`・
`add_pixel_unshuffle(downscale_factor)` と `Var::pixel_shuffle`／`Var::pixel_unshuffle` の `pub fn` 各 1 件として公開した
（追加 API のみ・新規公開型なし。`fandhe-ai =0.10.0` の公開 API は非破壊）。具体シグネチャは
`docs/autodiff-pixel-shuffle-decision.md` §6 に記載がなかったため内部コンストラクタと `Var::unflatten` から機械的に導出した。
`Var` メソッドと層は共有 forward を通し、倍率 `0` の `InvalidArgument` 判定を一本化した。学習経路・常駐経路は無状態層として通過し
（未対応層との混在は従来どおり `Unsupported`）、`save_model`／`load_model` は kind `pixel_shuffle`／`pixel_unshuffle` を追加した
（`format_version` 不変）。ONNX export は `UnsupportedLayer`。`Module` の `as_*` フックは追加していない。
保留ガードは型名・自由関数のプローブだけに縮小し、承認形は `api_surface.rs` の正ガードで固定した。
保留継続: 型の再エクスポート・自由関数公開・ONNX `DepthToSpace`／`SpaceToDepth`・GPU 専用カーネル。
`Cargo.toml`／`Cargo.lock`・tolerance／baseline・`docs/spec/` は不変。実機 parity は
`docs/perf/logs/compat-sequential-pixel-shuffle-2526/README.md` へ申し送り。

**適用記録（経路 2。イシュー #2527・親 #2520・ルート #2499 本文「承認範囲」節の一括承認に基づく）**:
`AdaptiveMaxPool2d`／`AdaptiveMaxPool1d`／`GlobalPool`（#2160）を `compat::Sequential::add_adaptive_max_pool2d(output_size)`・
`add_adaptive_max_pool1d(output_size)`・`add_global_pool(mode, keepdims)` と `Var::adaptive_max_pool2d`／`Var::adaptive_max_pool1d`
（戻り値は `(values, index)`）の `pub fn` として公開し、`add_global_pool` の引数型 `GlobalPoolMode` をルートへ再エクスポートした
（追加 API のみ。`fandhe-ai =0.10.0` の公開 API は非破壊）。具体シグネチャは `docs/autodiff-adaptive-max-global-pool-decision.md`
§6 が名前だけを挙げていたため既存規約から機械的に導出した（同 §8）。学習経路は無状態層として通過し、常駐経路は Pooling 層として
`Unsupported`（fail-closed）、`save_model`／`load_model` は kind `adaptive_max_pool2d`／`adaptive_max_pool1d`／`global_pool` を追加した
（`format_version` 不変）。ONNX export は `UnsupportedLayer`。`Module` の `as_*` フックは追加していない。
保留ガードは `Tensor`／`Tape` のプローブだけに縮小し、承認形は `api_surface.rs` の正ガードで固定した。
保留継続: 層型の再エクスポート・自由関数公開・`Tensor`／`Tape` メソッド・GPU 専用カーネル・ONNX `GlobalMaxPool`／`GlobalAveragePool`。
`Cargo.toml`／`Cargo.lock`・tolerance／baseline・`docs/spec/` は不変。実機 parity は
`docs/perf/logs/compat-sequential-adaptive-max-global-pool-2527/README.md` へ申し送り。

**適用記録（経路 2。イシュー #2528・親 #2520・ルート #2499 本文「承認範囲」節の一括承認に基づく）**:
`Dropout2d`／`AlphaDropout`／`EmbeddingBag`（#2161）を `compat::Sequential::add_dropout2d(p)`・`add_alpha_dropout(p)`・
`add_embedding_bag(num_embeddings, embedding_dim, mode, padding_idx, seed)` と `Var::dropout2d(p, training)`・
`Var::alpha_dropout(p, training)`・`Var::embedding_bag(ids, mode, padding_idx)` の `pub fn` として公開し、`add_embedding_bag`／
`Var::embedding_bag` の引数型 `EmbeddingBagMode` をルートへ再エクスポートした（追加 API のみ。`fandhe-ai =0.10.0` の公開 API は非破壊）。
具体シグネチャは `docs/autodiff-dropout-embedding-bag-decision.md` §6 が名前だけを挙げていたため既存規約から機械的に導出した（同 §8）。
Dropout2d／AlphaDropout は無状態層として学習経路・常駐経路を通過し（`set_training` は伝播する）、EmbeddingBag は `weight` 1 件を
学習パラメータとして追跡し常駐経路では `Unsupported`（fail-closed）とした。`save_model`／`load_model` は kind `dropout2d`／`alpha_dropout`／
`embedding_bag` を追加した（`format_version` 不変）。ONNX export は `UnsupportedLayer`。`Module` の `as_*` フックは追加していない。
保留ガードは層型・自由関数のプローブだけに縮小し、承認形は `api_surface.rs` の正ガードで固定した。
保留継続: 層型の再エクスポート・自由関数公開・可変長 bag（offsets）・GPU 専用カーネル・ONNX export の対象層の拡大。
`Cargo.toml`／`Cargo.lock`・tolerance／baseline・`docs/spec/` は不変。実機 parity は
`docs/perf/logs/compat-sequential-dropout-embedding-bag-2528/README.md` へ申し送り。

**適用記録（経路 2。イシュー #2529・親 #2520・ルート #2499 本文「承認範囲」節の一括承認に基づく）**:
`Mish`／`Hardtanh`／`Relu6`／`Glu`／`PRelu`（#2146）を `compat::Sequential::add_mish()`・`add_hardtanh(min_val, max_val)`・
`add_relu6()`・`add_glu(dim)`・`add_prelu(num_parameters, init)` として公開した（追加 API のみ。`fandhe-ai =0.10.0` の公開 API は非破壊。
具体シグネチャは `docs/autodiff-activation-ops-decision.md` §2.1 が名前だけを挙げていたため §2.4 の層コンストラクタから機械的に導出した。同 §10）。
無状態 4 層は学習経路・常駐経路を通過し、`PRelu` は `weight` 1 件を学習パラメータとして追跡し（`nn::Module` に既定 `None` の
`as_prelu`／`as_prelu_mut` を追加）常駐経路では `Unsupported`（fail-closed）とした。`save_model`／`load_model` は kind `mish`／`hardtanh`／
`relu6`／`glu`／`prelu` を追加した（45 → 50 種。`format_version` 不変。`prelu` の `init` は記録しない）。ONNX export は `UnsupportedLayer`。
保留ガードは `VarActivationOpsHoldDoctestGuard` の (b)（`add_*` 衝突プローブ）だけを撤去し、承認形は `api_surface.rs` の正ガードで固定した。
保留継続: `activation_ops` の再エクスポート・`Tensor<f32>`／`Tape` 上の配置・層型の再エクスポート・`PRelu::from_parameters` の公開・
GPU 専用カーネル・ONNX export の対象層の拡大。`Cargo.toml`／`Cargo.lock`・tolerance／baseline・`docs/spec/` は不変。実機 parity は
`docs/perf/logs/compat-sequential-activation-layers-2529/README.md` へ申し送り。

**適用記録（経路 2。イシュー #2530・ルート #2499 本文「承認範囲」節の一括承認に基づく）**:
`MultiheadAttentionConfig`（#2163）を `fandhe_ai::compat::MultiheadAttentionConfig`（`compat/mod.rs` の再エクスポート 1 行）と
`compat::Sequential::add_multihead_attention_with_config(config, seed)` として公開した（追加 API のみ。`fandhe-ai =0.10.0` の公開 API は非破壊。
再エクスポートの配置は決定記録に定めがなく、`nn/mod.rs` の完全一致集合を避け `compat` に置く判断を `docs/autodiff-mha-options-decision.md` に記録した）。
Sequential は self-attention 固定のため `kdim`/`vdim != embed_dim` は追加時に `InvalidArgument` で拒否し、選べるのは `bias`・`batch_first`。
学習・推論経路は既存の `as_multihead_attention` フックで結線済み（`nn::Module` へのメソッド追加なし）。resident は `Unsupported`、AMP は非既定 config を
`InvalidArgument` で拒否。`save_model`／`load_model` は kind `multihead_attention_config` を追加した（50 → 51 種。`format_version` 不変。
既存 kind `multihead_attention` は不変）。保留ガード（`MhaOptionsHoldDoctestGuard`・否定ガード）は承認形の正ガードへ反転した。
保留継続: `key_padding_mask` の Sequential 経由指定・cross-attention・ONNX export の対象層拡大。`Cargo.toml`／`Cargo.lock`・tolerance／baseline・`docs/spec/` は不変。
実機 parity は `docs/perf/logs/mha-config-sequential-2530/README.md` へ申し送り。

**適用記録（経路 2。イシュー #2532・親 #2531・ルート #2499 本文「承認範囲」節の一括承認に基づく）**:
`TransformerDecoderLayer`・`TransformerConfig`（#2165）を `fandhe_ai::nn::TransformerDecoderLayer`・`fandhe_ai::nn::TransformerConfig`
（`nn/mod.rs` の 1 文の再エクスポート）と `compat::Sequential::add_transformer_decoder_layer(d_model, num_heads, dim_feedforward, seed)` として公開した
（追加 API のみ。`fandhe-ai =0.10.0` の公開 API は非破壊。`docs/autodiff-transformer-decoder-decision.md` §承認事項の承認形のうち decoder 1 層の分）。
`Sequential` は単一入力列のため `tgt = memory = 直前層の出力`・mask なし・非 causal 固定で、`predict` と `bind().forward` が bit 一致する。
`bind`／`SequentialVars::forward`／`trainable_vars`／`trainable_grads`／`contains_resident_unsupported_layer` へ結線済み（既存の `as_transformer_decoder_layer`
フックを使用。`nn::Module` へのメソッド追加なし）。1 層 26 パラメータ。resident 経路は `Unsupported`（fail-closed）、AMP 低精度経路は decoder を f32 のまま通す。
`save_model`／`load_model` は kind `transformer_decoder_layer` を追加した（51 → 52 種。`format_version` 不変。manifest 上限の閾値は変更しない）。
ONNX export は `UnsupportedLayer`。保留ガード（`TransformerDecoderHoldDoctestGuard`・否定ガード）は decoder 1 層の分だけ縮め、承認形は `api_surface.rs` の
正ガード（`facade_reexports_transformer_decoder_items_only_in_approved_shape`・`compat_sequential_declares_add_transformer_decoder_layer_exactly_once`）で固定した。
保留継続: `Transformer`・`add_transformer`（#2533）・`FeedForwardActivation` の再エクスポートや `Tape` 委譲メソッド（単体構築・単体 forward。承認形の範囲外のため未実施）・
`tgt`／`memory` を別々に与える 2 入力 API・mask／causal 指定・Dropout 結線・GPU 専用カーネル。`Cargo.toml`／`Cargo.lock`・tolerance／baseline・`docs/spec/` は不変。
実機 parity は `docs/perf/logs/transformer-decoder-sequential-2532/README.md` へ申し送り。

**適用記録（経路 2。イシュー #2535・親 #2534・ルート #2499 本文「承認範囲」節の一括承認に基づく）**:
`RnnConfig`・`StackedRnn`／`StackedLstm`／`StackedGru`・`StackedRnnSeqOutput`／`StackedLstmSeqOutput`（#2164）を `fandhe_ai::nn::rnn` へ純再エクスポートし（既存 8 型と合わせ 14 型）、`Tape::stacked_rnn_forward_seq`／`stacked_lstm_forward_seq`／`stacked_gru_forward_seq`（`&self.0` を渡すだけの薄い委譲。#1955 の `rnn_forward_seq` 等と同じ回避不能な理由）を追加した（追加 API のみ。`fandhe-ai =0.10.0` の公開 API は非破壊）。`Rnn`／`Lstm`／`Gru::with_config`・`compat::Sequential::add_rnn` 等・`Stacked*` の facade `nn::Module` 実装は追加しない（`Sequential::add_stacked_*` も #2534 案 A〈2026-10-07 承認〉で追加しないと確定し、否定ガード `compat_sequential_does_not_expose_rnn_add_methods` が `add_stacked_rnn`／`add_stacked_lstm`／`add_stacked_gru` を固定する。保存・復元 API〈案 C〉は今回採らず後続の検討事項）。facade からは eval モード（`set_training(false)`）へ到達できないため `dropout > 0` の forward は常に学習モード（推論は `dropout = 0.0`。既知の制限）。`RnnConfigHoldDoctestGuard` を `with_config` 禁止のみへ縮小し、`api_surface.rs` に正ガード（`nn_rnn_module_reexports_exactly_expected_surface` の期待集合 14 型化・`nn_rnn_stacked_types_are_reachable_via_facade_only`・`tape_stacked_rnn_methods_are_thin_delegations`・`workspace_declares_stacked_rnn_tape_fn_names_only_in_facade_lib`・`workspace_declares_no_rnn_with_config_fn`）を追加した。正しさは `crates/facade/tests/nn_rnn_stacked_facade_bit_identity.rs`（CPU bit 一致）で検証。CUDA／Metal 実機 parity は `docs/perf/logs/rnn-stacked-2164/README.md` の申し送りが有効。

**適用記録（経路 2。イシュー #2533・親 #2531・ルート #2499 本文「承認範囲」節の一括承認に基づく）**:
`Transformer`（#2165。encoder スタック＋`encoder_norm`・decoder スタック＋`decoder_norm`）を `fandhe_ai::nn::Transformer`（`nn/mod.rs` の 1 文の再エクスポートに追加。
`TransformerConfig`・`TransformerDecoderLayer` と合わせ 3 名形）と `compat::Sequential::add_transformer(config: TransformerConfig, seed: u64)` として公開した
（追加 API のみ。`fandhe-ai =0.10.0` の公開 API は非破壊）。シグネチャは決定記録の「`Transformer` の構築は `TransformerConfig` 経由のみ」に合わせ config 方式とした。
`Sequential` は単一入力列のため `src = tgt = 直前層の出力`・mask なし・非 causal 固定で、`predict` と `bind().forward` が bit 一致する（`Module::forward` と同一呼び出し）。
活性化は `relu` のみ対応で、`TransformerConfig::with_activation` による他の活性化は `InvalidArgument` で fail-closed 拒否する（`FeedForwardActivation` は facade 未公開で、保存 manifest に活性化を持たせないため）。
`eps` は `with_eps` で変えられるため保存・復元する。パラメータ数は `16 * N_enc + 2 + 26 * N_dec + 2`（`named_parameters` 順 = `encoder.layers.{i}` → `encoder.norm` → `decoder.layers.{i}` → `decoder.norm`）。
`bind`／`SequentialVars::forward`／`trainable_vars`／`trainable_grads`／`apply_parameters`（汎用経路）／`first_untracked_parametric_layer`／`contains_resident_unsupported_layer` へ結線済み
（既存の `as_transformer` フックを使用。`nn::Module` へのメソッド追加なし）。`trainable_vars`／`trainable_grads` の encoder／decoder 分岐は Transformer 分岐と共有するためヘルパー関数へ抽出した（挙動不変）。
resident 経路は `Unsupported`（fail-closed）、AMP 低精度経路は Transformer を f32 のまま通す。ONNX export は `UnsupportedLayer`。
`save_model`／`load_model` は kind `transformer`（params: `d_model`・`num_heads`・`num_encoder_layers`・`num_decoder_layers`・`dim_feedforward`・`eps`）を追加した（52 → 53 種。`format_version` 不変。
manifest 上限の閾値は変更しない）。load 側は改竄 manifest の層数が巨大でも期待キー列を確保しないよう、キー数を checked 算術で実キー数と照合してから期待キー列を作る。
保留ガード `TransformerDecoderHoldDoctestGuard` と `api_surface.rs` の否定ガード 3 件を撤去し、承認形は `api_surface.rs` の正ガード
（`facade_reexports_transformer_decoder_items_only_in_approved_shape`・`compat_sequential_declares_add_transformer_exactly_once`）で固定した。
保留継続: `FeedForwardActivation` の再エクスポートや `Tape` 委譲メソッド（単体構築・単体 forward）・`src`／`tgt` を別々に与える 2 入力 API・mask／causal 指定・pre-norm・Dropout 結線・GPU 専用カーネル・ONNX export の対象拡大。
`Cargo.toml`／`Cargo.lock`・tolerance／baseline・`docs/spec/` は不変。実機 parity は `docs/perf/logs/transformer-sequential-2533/README.md` へ申し送り。

**適用記録（経路 2。イシュー #2538・親 #2537・ルート #2499 本文「承認範囲」節の一括承認に基づく）**:
`l1_loss`・`cross_entropy_loss_with`（#2166 で内部クレート限定実装済み）を、設計判断記録（`docs/autodiff-loss-ops-decision.md` §5）の推奨形どおり
`Var::l1_loss(&self, target: &Var, reduction: Reduction)`・`Var::cross_entropy_loss_with(&self, targets, class_dim, reduction, options: &CrossEntropyOptions)` の
`crate::loss_ops` への 1 行委譲メソッドとして公開した（`Var` は facade が再エクスポートするため `fandhe_ai::Var` 経由で到達。追加 API のみ。`fandhe-ai =0.10.0` の公開 API は非破壊）。
`LossOpsHoldDoctestGuard` は `__probe_var` から該当 4 行のみ削除する部分反転とし（`Tensor`／`Tape`・`loss_ops` モジュール・残り 5 名のプローブは維持）、
承認形は `api_surface.rs` の正ガード（`var_loss_ops_methods_are_thin_delegations`・`var_loss_ops_are_reachable_via_facade_var`）と、インベントリ
（`workspace_declares_loss_ops_fn_names_only_in_approved_locations`。`var.rs` 各 1 件を追加）で固定した。
保留継続: `loss_ops` モジュール・`CrossEntropyOptions`・`Reduction` の facade 再エクスポート（決定記録に推奨形の記載がなく一括承認の範囲外。facade 単独では非既定オプションを構築できない既知ギャップ）・
残り 5 関数の `Var` 委譲（#2539・#2540）・`nn::loss` 構造体の公開・GPU 専用カーネル。`Cargo.toml`／`Cargo.lock`・tolerance／baseline・`docs/spec/` は不変。
実機 parity は `docs/perf/logs/loss-ops-2166/README.md` の申し送りが有効（ホスト計算経路は不変）。

**適用記録（経路 2。イシュー #2539・親 #2537・ルート #2499 本文「承認範囲」節の一括承認に基づく）**:
`cosine_embedding_loss`・`margin_ranking_loss`・`triplet_margin_loss`・`poisson_nll_loss`（#2167 で内部クレート限定実装済み）を、設計判断記録
（`docs/autodiff-distance-poisson-loss-ops-decision.md` §5）の推奨形どおり `Var` の `crate::loss_ops` への 1 行委譲メソッドとして公開した
（`fandhe_ai::Var` 経由で到達。追加 API のみ。`fandhe-ai =0.10.0` の公開 API は非破壊）。
`LossOpsHoldDoctestGuard` は `__probe_var` から該当 8 行のみ削除する部分反転とし（`ctc_loss` の `Var` プローブ・`Tensor`／`Tape`・`loss_ops` モジュールのプローブは維持）、
承認形は `api_surface.rs` の正ガード（`var_loss_ops_methods_are_thin_delegations`〈6 件化〉・`var_distance_poisson_loss_ops_are_reachable_via_facade_var`）と、
インベントリ（`workspace_declares_loss_ops_fn_names_only_in_approved_locations`。`var.rs` 4 件を追加）で固定した。
保留継続: `TripletMarginOptions`・`PoissonNllOptions`・`Reduction`・`loss_ops` モジュールの facade 再エクスポート（推奨形の記載がなく一括承認の範囲外）・
`ctc_loss`（#2540）・`nn::loss` 構造体の公開（#2600）・GPU 専用カーネル。`Cargo.toml`／`Cargo.lock`・tolerance／baseline・`docs/spec/` は不変。
実機 parity は `docs/perf/logs/loss-ops-2167/README.md` の申し送りが有効（ホスト計算経路は不変）。

**適用記録（経路 2。イシュー #2540・親 #2537・ルート #2499 本文「承認範囲」節の一括承認に基づく）**:
`docs/autodiff-ctc-design.md` §5 の推奨形（単一案）どおり、`Var::ctc_loss(&self, targets, input_lengths, target_lengths, options, reduction)`
を `crate::loss_ops::ctc_loss` への 1 行委譲メソッドとして公開した（`fandhe-ai =0.10.0` の公開 API に追加のみ・非破壊）。
`LossOpsHoldDoctestGuard` から `Var` 側プローブ（`__probe_var`・`Var` impl）を撤去し、承認形は `api_surface.rs` の正ガード
（`var_loss_ops_methods_are_thin_delegations`〈7 件化〉・`var_ctc_loss_is_reachable_via_facade_var`）とインベントリ（`var.rs::ctc_loss` 追加）で固定した。
保留継続: `CtcLossOptions`・`Reduction`・`loss_ops` モジュールの facade 再エクスポート（推奨形外）・`nn::loss::CtcLoss` の公開（#2600）・GPU 専用カーネル。
`Cargo.toml`／`Cargo.lock`・tolerance／baseline・`docs/spec/` は不変。実機 parity は `docs/perf/logs/ctc-loss-2168/README.md` の申し送りが有効（ホスト計算経路は不変）。

**適用記録（経路 2。イシュー #2602・親 #2600・ルート #2499。承認コメント https://github.com/Fandhe-AI/fandhe-ai/issues/2499#issuecomment-6033824965）**:
`docs/facade-nn-loss-structs-exposure-decision.md` §4 の承認形どおり、`fandhe_ai::nn::loss` を新設し `fandhe_ai_autodiff::nn::loss` の 19 名
（損失構造体 14・`Reduction`・`CrossEntropyOptions`・`TripletMarginOptions`・`PoissonNllOptions`・`CtcLossOptions`）を明示列挙で純再エクスポートした
（`fandhe-ai =0.10.0` の公開 API に追加のみ・非破壊）。`Reduction`・オプション型は `nn::loss` 経由のみで、crate root・`nn` 直下・`loss_ops` 経由には出さない。
承認形は `api_surface.rs` の正ガード（`facade_reexports_nn_loss_items_only_in_approved_shape` ほか 5 件）と `nn_mod_*` の期待値拡張で固定した。
保留継続: `loss_ops` モジュールの再エクスポート・`Tensor`／`Tape` 上の同名メソッド・Phase 4 保留中の損失 3 本・`CrossEntropyLoss` への `new`／`Default` 追加。
`Cargo.toml`／`Cargo.lock`・tolerance／baseline・`docs/spec/` は不変。新規カーネルがないため実機 parity の申し送りは不要。

**適用記録（経路 2。イシュー #2550・#2549・親 #2542・ルート #2499 本文「承認範囲」節の一括承認に基づく）**:
`docs/autodiff-custom-function-decision.md` §16.1 の確定形（単一案）どおり、`fandhe_ai::CustomFunction`（crate ルートの `pub use`）と
facade `Tape::custom`（`self.0.custom(func, inputs)` の 1 行委譲）を公開した（`fandhe-ai =0.10.0` の公開 API に追加のみ・非破壊）。
保留ガードは承認形だけを許す正ガードへ反転した（`facade_reexports_custom_function_only_at_crate_root`・`facade_tape_custom_is_approved_thin_delegation`・
`workspace_declares_custom_fn_only_on_tape_and_facade_delegation`。`VarCustomHoldDoctestGuard` は `Tape::custom` プローブのみ削除する部分反転。対応表は同 doc §17.2）。
保留継続: `Var::custom`／`add_custom`・`Tape::add_custom`・`Sequential::custom`／`add_custom`・`TapeRef::custom`・`nn::Module`／`add_module`・二階微分向けの trait 拡張。
`Cargo.toml`／`Cargo.lock`・tolerance／baseline・`docs/spec/` は不変。host 実行のため REQ-2 の対象外で、申し送りは `docs/perf/logs/facade-custom-function-2549/README.md`。

**適用記録（経路 2。イシュー #2546・#2545・親 #2542・ルート #2499 本文「承認範囲」節の一括承認に基づく）**:
`docs/autodiff-higher-order-grad-decision.md` §17.2 の確定形（単一案）どおり、`fandhe_ai::CreateGraphResult`（crate ルートの `pub use`）と
facade `Tape::backward_create_graph`（`self.0.backward_create_graph(loss, &child.0)` の 1 行委譲）を公開した（`fandhe-ai =0.10.0` の公開 API に追加のみ・非破壊）。
保留ガードは承認形だけを許す正ガードへ反転した（`facade_reexports_create_graph_result_in_approved_shape`・`facade_tape_backward_create_graph_is_thin_delegation`・
`workspace_declares_backward_create_graph_only_in_approved_locations`・`tape_backward_create_graph_is_reachable_via_facade`。実施は #2735、対応表は同 doc §19.2）。
利用例テストとして `child_built_with_tape_for_parent_device_works`・`rejects_parent_with_registered_checkpoint_via_facade` を `crates/facade/tests/create_graph_facade.rs` に追加した。
保留継続: `tape.child()` 等の子テープ構築ヘルパー・`TapeRef::backward_create_graph`・`new_with_ops` への到達経路・HVP 専用 API（案 C）・`backward_accumulate`／checkpoint 併用・残る非対象 Op の拡張。
`Cargo.toml`／`Cargo.lock`・tolerance／baseline・`docs/spec/` は不変。GPU 実機の申し送りは `docs/perf/logs/create-graph-facade-2545/README.md`。

**適用記録（経路 2。イシュー #2568・#2569・親 #2566・ルート #2499 本文「承認範囲」節の一括承認に基づく）**:
`docs/compat-train-step-hook-decision.md` §8.1 の確定形（単一案）どおり、`fandhe_ai::compat::{TrainStepFn, TrainStepOptimizer, TrainStepOutput}`
（`compat/mod.rs` の `pub use` 葉 3 件）と inherent `Sequential::fit_with_train_step`（`compat/training.rs`）の 7 要素を公開した（実施は #2751）。
`fandhe-ai =0.10.0` の公開 API に追加のみ・非破壊（`FitConfig` は `Copy + Eq` のまま不変。inherent メソッドの解決先が変わりうる注記は同 doc §8.3）。
保留ガードは承認形だけを許す正ガードへ反転した（`facade_train_step_public_surface_matches_approved_contract`・
`facade_train_step_optimizer_and_output_shapes_match_approved_contract`・`workspace_declares_train_step_fn_names_only_in_approved_location`・
`train_step_types_are_reachable_via_facade_only`。実施は #2569、対応表は同 doc §9）。`TrainStepHoldDoctestGuard` は禁止経路
（`FitConfig::train_step_fn`・`FitConfig::fit_with_train_step`・`Sequential::train_step_fn`）専用の doctest として維持した。
利用例テストは `crates/facade/tests/compat_sequential_train_step.rs`（`f32` target の既存テストに加え、#2569 で `i32` クラス添字 target の
`fit_with_train_step_class_index_target_cross_entropy` を追加）。
保留継続: `test_step_fn`／`predict_step_fn`・AMP／勾配累積／L-BFGS とフックの併用・`TrainStepOptimizer::set_lr`・`DeviceParamStore` 常駐経路でのフック（同 doc §6）、
`Reduction` の facade 再エクスポート（#2538 の保留）。
`Cargo.toml`／`Cargo.lock`・tolerance／baseline・`docs/spec/` は不変。新規演算は無く CPU のみで検証できるため実機 parity の申し送りは不要。

**適用記録（経路 2。イシュー #2564・#2565・親 #2562・ルート #2499 のコメント〈`issuecomment-6033824965`・2026-10-07〉の「#2562: `docs/compat-fit-sample-weighting-decision.md` §11.1 の確定形と §11.3 の推奨案〈式が未定義の組み合わせは fail-closed〉」に基づく）**:
§11.1 の確定形どおり、`fandhe_ai::compat::FitWeights`（`compat/mod.rs` の `pub use` 葉 1 件。`FitWeights::{new, class_weight, sample_weight}`）・inherent `FitConfig::validation_split`・
inherent `Sequential::fit_with_weights` の 6 要素を公開した（実施は #2564・PR #2823）。`fandhe-ai =0.10.0` の公開 API に追加のみ・非破壊（`FitConfig` は `Copy + Eq` のまま不変）。
保留ガードは承認形だけを許す正ガードへ反転した（`facade_fit_weighting_public_surface_matches_approved_contract`・`facade_fit_weights_shape_matches_approved_contract`・
`workspace_declares_fit_weighting_fn_names_only_in_approved_location`・`fit_weighting_items_are_reachable_via_facade_only`。実施は #2565、対応表は同 doc §15）。
`FitWeightingHoldDoctestGuard` は禁止経路（`FitConfig::class_weight`／`sample_weight`／`fit_with_weights`／`fit_weighted`・`Sequential::validation_split`／`class_weight`／`sample_weight`／`fit_weighted`）専用の doctest として維持した。
利用例は `crates/facade/tests/compat_sequential_fit_weights.rs`（#2564）と各公開項目の doctest、facade 単独 import の到達性テスト（#2565）。
保留継続: 式が未定義の組み合わせ（`CrossEntropy`／`Mse` 以外の `sample_weight`・`CrossEntropy` 以外の `class_weight`・非既定の重み × `Lbfgs`）の `InvalidArgument` 拒否（fail-closed のまま）、`fit_weighted` 等の別名、`FitConfig` へ重みを持たせる形。
`Cargo.toml`／`Cargo.lock`・tolerance／baseline・`docs/spec/` は不変。新規演算は無く CPU のみで検証できるため実機 parity の申し送りは不要。

**適用記録（経路 2。イシュー #2571・親 #2570・ルート #2499 本文「承認範囲」節の一括承認に基づく）**:
`docs/compat-callbacks-loggers-decision.md` §3 の推奨形（単一案）どおり、`compat::{CsvLogger, JsonLogger, LambdaCallback}` と
`compat::Callback::{CsvLogger, JsonLogger, Lambda}`（3 variant 追加。`#[non_exhaustive]`）を公開した（`fandhe-ai =0.10.0` の公開 API に追加のみ・非破壊。`FitConfig` は不変）。
保留ガード（`CallbacksLoggersHoldDoctestGuard` と `api_surface.rs` の 4 テスト）は削除し、承認形だけを許す正ガード
（`facade_declares_callback_loggers_only_in_approved_shape`・`compat_callback_enum_variants_are_exactly_approved_set`・`fit_types_are_reachable_via_facade_only` の拡張）へ反転した。
JSON は手書き（`serde_json` 非追加）、append 時の既存ファイルは上限付き読み込み＋再帰しないパーサで検証する。実装時に固定した点（列順・公開アクセサなし・読み込み上限）は同 doc §9。
保留継続: 公開アクセサの追加・`&mut Sequential` を渡す callback・外部ロギング基盤・パスのシンボリックリンク検査。
`Cargo.toml`／`Cargo.lock`・tolerance／baseline・`docs/spec/` は不変。ホスト側のファイル I/O のみでカーネルを持たないため CUDA／Metal 実機申し送りは不要。

**適用記録（イシュー #2572・親 #2570。#2571 の実装記録の確認と利用例・補強ガード）**:
ガード反転は #2571 で前倒し済みのため、#2572 では docs（`docs/compat-callbacks-loggers-decision.md` §3・§7・§8 を実装済みへ更新し §10 を追記）と、
公開アクセサ・trait impl の承認形を固定する補強ガード（`callback_logger_types_expose_only_approved_methods_and_traits`）、
ロガー × `LrSchedule` の利用例テスト（`compat_sequential_callbacks.rs` 19 節）だけを追加した。公開面・`Cargo.toml`／`Cargo.lock`・tolerance／baseline・`docs/spec/` は不変。

**適用記録（イシュー #2631・親 #2630・ルート #2499 Phase 4。FFT 第 1 弾）**:
`rfft`／`irfft` を内部クレート限定（`fandhe_ai_autodiff::fft_ops`・`fandhe_ai_tensor_core::{FftNorm, fft}`）で実装した。
facade 公開面は追加していない（保留ガード `FftOpsHoldDoctestGuard` と `api_surface.rs` の否定ガードが固定）。
公開形（`Var::rfft`／`Var::irfft` と `FftNorm` の再エクスポート）は未承認で、承認依頼は #2677（公開自体は承認後の #2678）。
本書 1 節の対象範囲表・`docs/compat-feature-gap.md` の判定列は変更していない。
内部実装はルート #2499 の Phase 4 方針（内部実装＋保留ガードまで先行・公開は承認後）に基づく。詳細は `docs/autodiff-fft-ops-decision.md`。

**適用記録（イシュー #2628・親 #2626・ルート #2499 Phase 4。低精度 forward の Op 拡張）**:
MatMul と elementwise 5 演算（Add／Mul／Relu／Exp／Tanh）の opt-in 低精度 forward を内部クレート限定（`fandhe_ai_autodiff::low_precision_ops`・`fandhe_ai_tensor_core::*_low_precision`）で CPU 実装した。
facade 公開面は追加していない（保留ガード `VarLowPrecisionOpsHoldDoctestGuard` と `api_surface.rs` の否定ガードが固定）。
公開形（`Var::{matmul,add,mul,relu,exp,tanh}_low_precision(.., dtype)`）は未承認で、承認依頼は #2677（公開自体は承認後の #2678）。
本書 1 節の対象範囲表・`docs/compat-feature-gap.md` の判定列は変更していない。
内部実装はルート #2499 の Phase 4 方針（内部実装＋保留ガードまで先行・公開は承認後）に基づく。詳細は `docs/autodiff-low-precision-op-extension-decision.md` 6 節。

**適用記録（イシュー #2632・親 #2630・ルート #2499 Phase 4。FFT 第 2 弾）**:
`fft`／`ifft`（実部・虚部の実テンソル対による c2c）を `rfft`／`irfft` と同じ内部クレート限定の方式で実装した。
facade 公開面は追加していない（保留ガードのプローブと `api_surface.rs` のインベントリへ `fft`／`ifft` を追加）。
公開形（`Var::fft`／`Var::ifft`・`FftNorm` の再エクスポート）は未承認で、承認依頼は #2677（公開自体は承認後の #2678）。
本書 1 節の対象範囲表・`docs/compat-feature-gap.md` の判定列は変更していない。詳細は `docs/autodiff-fft-ops-decision.md` §12。

**適用記録（イシュー #2633・親 #2630・ルート #2499 Phase 4。FFT 第 3 弾）**:
`stft`／`istft`（窓付き短時間フーリエ変換とその逆変換）を `rfft`／`irfft`／`fft`／`ifft` と同じ内部クレート限定の方式で実装した。
facade 公開面は追加していない（保留ガードのプローブと `api_surface.rs` のインベントリへ `stft`／`istft`・`StftOptions`／`IstftOptions`／`StftPadMode` を追加）。
公開形（`Var::stft`／`Var::istft`・`StftOptions`／`IstftOptions`／`StftPadMode` の再エクスポート）は未承認で、承認依頼は #2677（公開自体は承認後の #2678）。
本書 1 節の対象範囲表・`docs/compat-feature-gap.md` の判定列は変更していない。詳細は `docs/autodiff-fft-ops-decision.md` §13。

**適用記録（イシュー #2634・親 #2625・ルート #2499 Phase 4。逆三角関数・双曲線関数）**:
`atan`／`asin`／`acos`／`atan2`／`sinh`／`cosh`／`asinh`／`acosh`／`atanh` の 9 演算を内部クレート限定（`fandhe_ai_autodiff::trig_ops`・`fandhe_ai_tensor_core::{ScalarUnaryOp, ScalarBinaryOp}` の追加 variant）で CPU 実装した。
facade 公開面は追加していない（保留ガード `TrigOpsHoldDoctestGuard` と `api_surface.rs` の否定ガードが固定）。
公開形（`Var::atan` 等の委譲メソッド）は未承認で、承認依頼は #2677（公開自体は承認後の #2678）。
本書 1 節の対象範囲表・`docs/compat-feature-gap.md` の判定列は変更していない。詳細は `docs/autodiff-trig-ops-decision.md`。

**適用記録（イシュー #2635・親 #2625・ルート #2499 Phase 4。非有限値の判定・置換）**:
`isnan`／`isinf`／`isfinite`／`nan_to_num` の 4 演算を内部クレート限定（`fandhe_ai_autodiff::nonfinite_ops`・`fandhe_ai_tensor_core::ScalarUnaryOp` の追加 variant）で CPU 実装した。
facade 公開面は追加していない（保留ガード `NonfiniteOpsHoldDoctestGuard` と `api_surface.rs` の否定ガードが固定）。
公開形（`Var::isnan` 等の委譲メソッド）は未承認で、承認依頼は #2677（公開自体は承認後の #2678）。
本書 1 節の対象範囲表・`docs/compat-feature-gap.md` の判定列は変更していない。詳細は `docs/autodiff-nonfinite-ops-decision.md`。

**適用記録（イシュー #2636・親 #2625・ルート #2499 Phase 4。累積演算）**:
`cummax`／`cummin`／`logcumsumexp` の 3 演算を内部クレート限定（`fandhe_ai_autodiff::cumulative_ops`・`fandhe_ai_tensor_core::cumulative`・`BackendOps::scan_*`）で CPU 実装した。
facade 公開面は追加していない（保留ガード `CumulativeOpsHoldDoctestGuard` と `api_surface.rs` の否定ガードが固定）。
公開形（`Var::cummax` 等の委譲メソッド）は未承認で、承認依頼は #2677（公開自体は承認後の #2678）。
本書 1 節の対象範囲表・`docs/compat-feature-gap.md` の判定列は変更していない。詳細は `docs/autodiff-cumulative-ops-decision.md`。

**適用記録（イシュー #2637・親 #2625・ルート #2499 Phase 4。順序統計・NaN 無視縮約）**:
`median`／`kthvalue`／`quantile`／`nanmean`／`nansum` の 5 演算を内部クレート限定（`fandhe_ai_autodiff::stat_reduce_ops`・`fandhe_ai_tensor_core::stat_reduce`・`BackendOps::stat_*`）で CPU 実装した。
facade 公開面は追加していない（保留ガード `StatReduceOpsHoldDoctestGuard` と `api_surface.rs` の否定ガードが固定）。
公開形（`Var::median` 等の委譲メソッドと `QuantileInterpolation` の再エクスポート）は未承認で、承認依頼は #2677（公開自体は承認後の #2678）。
本書 1 節の対象範囲表・`docs/compat-feature-gap.md` の判定列は変更していない。詳細は `docs/autodiff-stat-reduce-ops-decision.md`。

**適用記録（イシュー #2643・親 #2625・ルート #2499 Phase 4。3D プーリング）**:
`max_pool3d`／`avg_pool3d` の 2 演算を内部クレート限定（`fandhe_ai_autodiff::pool3d_ops`・`fandhe_ai_tensor_core::pool3d`・`BackendOps::pool3d_max`／`pool3d_avg`）で CPU 実装した。
facade 公開面は追加していない（保留ガード `Pool3dOpsHoldDoctestGuard` と `api_surface.rs` の否定ガードが固定）。
公開形（`Var::max_pool3d` 等の委譲メソッド）は未承認で、承認依頼は #2677（公開自体は承認後の #2678）。層化（`nn::MaxPool3d`／`AvgPool3d`・`Sequential::add_*`）は #2679 の対象で本イシューでは実装していない。
本書 1 節の対象範囲表・`docs/compat-feature-gap.md` の判定列は変更していない。詳細は `docs/autodiff-pool3d-ops-decision.md`。

**適用記録（イシュー #2644・親 #2625・ルート #2499 Phase 4。ConvTranspose3d・MaxUnpool）**:
`conv_transpose3d`／`max_unpool1d`／`max_unpool2d`／`max_unpool3d` の 4 演算を内部クレート限定（`fandhe_ai_autodiff::conv_transpose3d_ops`・`max_unpool_ops`・`fandhe_ai_tensor_core::conv_transpose3d`・`max_unpool`）で CPU 実装した（新規 `BackendOps` メソッドなし。既存の `gemm_batched`・`col2im3d`・`im2col3d`・`scatter`・`gather` フックの合成）。
facade 公開面は追加していない（保留ガード `ConvTranspose3dMaxUnpoolHoldDoctestGuard` と `api_surface.rs` の否定ガードが固定）。
公開形（`Var::conv_transpose3d`／`Var::max_unpool1d/2d/3d` の委譲メソッド）は未承認で、承認依頼は #2677（公開自体は承認後の #2678）。層化（`nn::ConvTranspose3d`／`nn::MaxUnpool*`・`Sequential::add_*`）は #2679 の対象で本イシューでは実装していない。
本書 1 節の対象範囲表・`docs/compat-feature-gap.md` の判定列は変更していない。詳細は `docs/autodiff-conv-transpose3d-max-unpool-decision.md`。

**適用記録（イシュー #2645・親 #2625・ルート #2499 Phase 4。Fold・Unfold）**:
`fold`／`unfold`（`F.fold`／`F.unfold` 相当）の 2 演算を内部クレート限定（`fandhe_ai_autodiff::fold_ops`・`fandhe_ai_tensor_core::fold`）で CPU 実装した（新規 `BackendOps` メソッドなし。既存の `im2col`／`col2im` フックの再利用）。
facade 公開面は追加していない（保留ガード `FoldUnfoldHoldDoctestGuard` と `api_surface.rs` の否定ガードが固定）。
公開形（`Var::unfold`／`Var::fold` の委譲メソッド）は未承認で、承認依頼は #2677（公開自体は承認後の #2678）。層化（`nn::Fold`／`nn::Unfold`・`Sequential::add_*`）は #2679 の対象で本イシューでは実装していない。
本書 1 節の対象範囲表・`docs/compat-feature-gap.md` の判定列は変更していない。詳細は `docs/autodiff-fold-unfold-decision.md`。

**適用記録（イシュー #2646・親 #2625・ルート #2499 Phase 4。LRN・重み再パラメータ化）**:
`local_response_norm`／`weight_norm`／`norm_except_dim`／`spectral_norm`（`F.local_response_norm`・`torch._weight_norm`・`parametrizations.spectral_norm` 相当）を内部クレート限定（`fandhe_ai_autodiff::{lrn_ops, weight_reparam_ops}`・`fandhe_ai_tensor_core::{lrn, weight_reparam}`）で CPU 実装した（新規 `BackendOps` フック 3 件・既定 `Unsupported`。CPU は共有ホストカーネルを呼ぶだけ）。
facade 公開面は追加していない（保留ガード `LrnWeightReparamHoldDoctestGuard` と `api_surface.rs` の否定ガードが固定）。
公開形（`Var::local_response_norm`／`Var::weight_norm`／`Var::spectral_norm` の委譲メソッドと `SpectralNormState` の公開位置）は未承認で、承認依頼は #2677（公開自体は承認後の #2678）。層化（`nn::LocalResponseNorm`・`Linear`／`Conv` への parametrization 結線・`Sequential::add_*`）は #2679 の対象で本イシューでは実装していない。
本書 1 節の対象範囲表・`docs/compat-feature-gap.md` の判定列は変更していない。詳細は `docs/autodiff-lrn-weight-reparam-decision.md`。

**適用記録（イシュー #2638・親 #2625・ルート #2499 Phase 4。ヒストグラム・二分探索系）**:
`histc`／`bincount`／`searchsorted`／`bucketize` の 4 演算（非微分）を内部クレート限定（`fandhe_ai_autodiff::binning_ops`・`fandhe_ai_tensor_core::binning`・`BackendOps::binning_*`）で CPU 実装した。
facade 公開面は追加していない（保留ガード `BinningOpsHoldDoctestGuard` と `api_surface.rs` の否定ガードが固定）。
公開形（`Var::histc` 等の委譲メソッド）は未承認で、承認依頼は #2677（公開自体は承認後の #2678・#2679）。
本書 1 節の対象範囲表・`docs/compat-feature-gap.md` の判定列は変更していない。詳細は `docs/autodiff-binning-ops-decision.md`。

**適用記録（イシュー #2639・親 #2625・ルート #2499 Phase 4。形状演算）**:
`unbind`／`movedim`／`swapaxes`／`tensor_split`（分割数形・境界添字列形）／`meshgrid`／`rot90` を内部クレート限定（`fandhe_ai_autodiff::shape_view_ops`。新規 `Op`・`BackendOps` メソッドなしの既存 `Op` 合成）で CPU 実装した。
facade 公開面は追加していない（保留ガード `ShapeViewOpsHoldDoctestGuard` と `api_surface.rs` の否定ガードが固定）。
公開形（`Var::unbind` 等の委譲メソッドと `MeshgridIndexing` の再エクスポート）は未承認で、承認依頼は #2677（公開自体は承認後の #2678）。
本書 1 節の対象範囲表・`docs/compat-feature-gap.md` の判定列は変更していない。詳細は `docs/autodiff-shape-view-ops-decision.md`。

**適用記録（イシュー #2641・親 #2625・ルート #2499 Phase 4。索引付き更新）**:
`scatter_reduce`／`index_add`／`index_copy`／`masked_scatter` を内部クレート限定（`fandhe_ai_autodiff::indexed_update_ops`・`fandhe_ai_tensor_core::indexed_update`・`BackendOps::indexed_scatter_reduce`）で CPU 実装した（`scatter_reduce` は専用 `Op`＋共有ホストカーネル、残り 3 演算は既存 `Op::Scatter` の合成）。
facade 公開面は追加していない（保留ガード `IndexedUpdateOpsHoldDoctestGuard` と `api_surface.rs` の否定ガードが固定）。
公開形（`Var::scatter_reduce` 等の委譲メソッドと `ScatterReduceMode` の再エクスポート）は未承認で、承認依頼は #2677（公開自体は承認後の #2678・#2679）。
本書 1 節の対象範囲表・`docs/compat-feature-gap.md` の判定列は変更していない。詳細は `docs/autodiff-indexed-update-ops-decision.md`。

**適用記録（イシュー #2640・親 #2625・ルート #2499 Phase 4。テンソル積・距離・外積）**:
`kron`／`tensordot`（`tensordot_axes`）／`cdist`／`cross` を内部クレート限定（`fandhe_ai_autodiff::tensor_product_ops`。新規 `Op`・`BackendOps` メソッドなしの既存 `Op` 合成）で CPU 実装した。
facade 公開面は追加していない（保留ガード `TensorProductOpsHoldDoctestGuard` と `api_surface.rs` の否定ガードが固定）。
公開形（`Var::kron` 等の委譲メソッド）は未承認で、承認依頼は #2677（公開自体は承認後の #2678）。
本書 1 節の対象範囲表・`docs/compat-feature-gap.md` の判定列は変更していない。詳細は `docs/autodiff-tensor-product-ops-decision.md`。

**適用記録（イシュー #2642・親 #2625・ルート #2499 Phase 4。pad の非定数モード）**:
`pad` の reflect／replicate／circular モードを内部クレート限定（`fandhe_ai_autodiff::pad_ops`・`fandhe_ai_tensor_core::pad_modes`・`BackendOps::pad_modes_forward`）で CPU 実装した（専用 `Op::PadMode`＋共有ホストカーネル。既存 `Var::pad`〈定数埋め〉は不変）。
facade 公開面は追加していない（保留ガード `PadModesHoldDoctestGuard` と `api_surface.rs` の否定ガードが固定）。
公開形（`Var::pad_with_mode` の委譲メソッドと `PadMode` の再エクスポート）は未承認で、承認依頼は #2677（公開自体は承認後の #2678・#2679）。
本書 1 節の対象範囲表・`docs/compat-feature-gap.md` の判定列は変更していない。詳細は `docs/autodiff-pad-modes-decision.md`。

**適用記録（イシュー #2647・親 #2625・ルート #2499 Phase 4。可変長系列）**:
`pack_padded_sequence`／`pad_packed_sequence`／`PackedSequence` と、`Rnn`／`Lstm`／`Gru`・`StackedRnn`／`StackedLstm`／`StackedGru` の packed 実行（自由関数 8 本）を内部クレート限定（`fandhe_ai_autodiff::nn::packed_sequence`。新規 `Op`・`BackendOps` メソッドなしの既存 `Op` 合成）で CPU 実装した。
facade 公開面は追加していない（保留ガード `PackedSequenceHoldDoctestGuard` と `api_surface.rs` の否定ガードが固定）。
公開形（`fandhe_ai::nn::rnn` へのモジュール再エクスポート）は未承認で、承認依頼は #2677（公開自体は承認後の #2678・#2679）。
本書 1 節の対象範囲表・`docs/compat-feature-gap.md` の判定列は変更していない。詳細は `docs/autodiff-packed-sequence-decision.md`。

**適用記録（イシュー #2649・親 #2648。活性化 5 種）**:
`selu`／`celu`／`softsign`／`hardsigmoid`／`log_sigmoid` を内部クレート限定（`fandhe_ai_autodiff::activation_scalar_ops`・`nn::activation` の層 5 型・`fandhe_ai_tensor_core::ScalarUnaryOp` の追加 variant）で CPU 実装した。
facade 公開面は追加していない（保留ガード `ActivationScalarOpsHoldDoctestGuard` と `api_surface.rs` の否定ガードが固定）。
公開形（`Var` 委譲メソッド・`compat::Sequential::add_*`）は未承認で、承認依頼は #2677（公開自体は承認後の #2678・#2679）。
本書 1 節の対象範囲表・`docs/compat-feature-gap.md` の判定列は変更していない。詳細は `docs/autodiff-activation-scalar-ops-decision.md`。

**適用記録（イシュー #2650・親 #2648・ルート #2499 Phase 4。Softmin・Tanhshrink・RReLU・Threshold）**:
`softmin`／`tanhshrink`／`threshold`／`rrelu`／`rrelu_with_noise` と nn 層 `Softmin`／`Tanhshrink`／`Threshold`／`RRelu` を内部クレート限定（`fandhe_ai_autodiff::softmin_threshold_ops`・`fandhe_ai_autodiff::nn::softmin_threshold`。新規 `Op`・`BackendOps` メソッドなしの既存 `Op` 合成）で CPU 実装した。
facade 公開面は追加していない（保留ガード `SoftminThresholdOpsHoldDoctestGuard` と `api_surface.rs` の否定ガードが固定）。
公開形（`Var::softmin` 等の委譲メソッドと `compat::Sequential::add_softmin` 等）は未承認で、承認依頼は #2677（公開自体は承認後の #2678・#2679）。
本書 1 節の対象範囲表・`docs/compat-feature-gap.md` の判定列は変更していない。詳細は `docs/autodiff-softmin-threshold-ops-decision.md`。

**適用記録（イシュー #2652・親 #2651・ルート #2499 Phase 4。pos_weight 付き BCEWithLogits・HingeEmbedding・SoftMargin・GaussianNLL）**:
`bce_with_logits_loss_with`／`hinge_embedding_loss`／`soft_margin_loss`／`gaussian_nll_loss` を内部クレート限定（`fandhe_ai_autodiff::elementwise_loss_ops`。新規 `Op` 4 種とホスト参照実装）で CPU 実装した。
facade 公開面は追加していない（保留ガード `ElementwiseLossOpsHoldDoctestGuard` と `api_surface.rs` の否定ガードが固定）。
公開形（`Var` の委譲メソッド 4 本）は未承認で、承認依頼は #2677（公開自体は承認後の #2678）。
本書 1 節の対象範囲表・`docs/compat-feature-gap.md` の判定列は変更していない。詳細は `docs/autodiff-elementwise-loss-ops-decision.md`。

**適用記録（イシュー #2653・親 #2651・ルート #2499 Phase 4。MultiMargin・MultiLabelMargin・MultiLabelSoftMargin・sigmoid focal loss）**:
`multi_margin_loss`／`multilabel_margin_loss`／`multilabel_soft_margin_loss`／`sigmoid_focal_loss` を内部クレート限定（`fandhe_ai_autodiff::margin_focal_loss_ops`。新規 `Op` 4 種とホスト参照実装）で CPU 実装した。
facade 公開面は追加していない（保留ガード `MarginFocalLossOpsHoldDoctestGuard` と `api_surface.rs` の否定ガードが固定）。
公開形（`Var` の委譲メソッド 4 本）は未承認で、承認依頼は #2677（公開自体は承認後の #2678）。
本書 1 節の対象範囲表・`docs/compat-feature-gap.md` の判定列は変更していない。詳細は `docs/autodiff-margin-focal-loss-ops-decision.md`。

**適用記録（イシュー #2655・親 #2654・ルート #2499 Phase 4。Rprop・ASGD）**:
`Rprop`／`RpropConfig`／`Asgd`／`AsgdConfig` を内部クレート限定（`fandhe_ai_autodiff::nn::optim`。`Tape`／`Var`／`BackendOps` 非依存のホスト値型で、新規 `Op`・`BackendOps` メソッド・カーネルなし）で CPU 実装した。
facade 公開面は追加していない（保留ガード `OptimizerRpropAsgdHoldDoctestGuard` と `api_surface.rs` の否定ガードが固定）。
公開形（`fandhe_ai::optim` への 4 名の素の再エクスポート）は未承認で、承認依頼は #2677（公開自体は承認後の #2679）。
本書 1 節の対象範囲表・`docs/compat-feature-gap.md` の判定列は変更していない。詳細は `docs/autodiff-optimizer-rprop-asgd-decision.md`。

**適用記録（イシュー #2656・親 #2654・ルート #2499 Phase 4。Adafactor・Lion）**:
`Adafactor`／`AdafactorConfig`／`Lion`／`LionConfig` を内部クレート限定（`fandhe_ai_autodiff::nn::optim`。`Tape`／`Var`／`BackendOps` 非依存のホスト値型で、新規 `Op`・`BackendOps` メソッド・カーネルなし）で CPU 実装した。
Lion の参照値は `torch.optim` 2.14.0 に Lion が無いため、公式参照実装の更新則を torch 2.14.0 のテンソル演算で実行した値である。
facade 公開面は追加していない（保留ガード `OptimizerAdafactorLionHoldDoctestGuard` と `api_surface.rs` の否定ガードが固定）。
公開形（`fandhe_ai::optim` への 4 名の素の再エクスポート）は未承認で、承認依頼は #2677（公開自体は承認後の #2679）。
本書 1 節の対象範囲表・`docs/compat-feature-gap.md` の判定列は変更していない。詳細は `docs/autodiff-optimizer-adafactor-lion-decision.md`。

**適用記録（イシュー #2658。内部クレート限定）**: SWA 相当の `AveragedModel`（等重み平均）・`SwaLr`／`SwaAnneal`（`SWALR` 相当）を内部クレート限定（`fandhe_ai_autodiff::nn`／`nn::optim`。ホスト値型・`f32` 純関数で、新規 `Op`・`BackendOps` メソッド・VJP・カーネルなし）で CPU 実装した。
facade 公開面は追加していない（保留ガード `SwaHoldDoctestGuard` と `api_surface.rs` の否定ガードが固定）。公開形は未承認で、承認依頼は #2677（公開自体は承認後の #2678・#2679）。
本書 1 節の対象範囲表・`docs/compat-feature-gap.md` の判定列は変更していない。詳細は `docs/autodiff-swa-decision.md`。

**適用記録（イシュー #2659。内部クレート限定）**: `PolynomialLr`（PyTorch `PolynomialLR` 相当）・`ChainedScheduler`（同 `ChainedScheduler` 相当）を内部クレート限定（`fandhe_ai_autodiff::nn::optim`。ホスト `f32` 純関数で、新規 `Op`・`BackendOps` メソッド・VJP・カーネルなし。`LrScheduler` trait は不変）で CPU 実装した。
facade 公開面は追加していない（保留ガード `LrSchedulerPolyChainedHoldDoctestGuard` と `api_surface.rs` の否定ガードが固定）。公開形は未承認で、承認依頼は #2677（公開自体は承認後の #2679）。
本書 1 節の対象範囲表・`docs/compat-feature-gap.md` の判定列は変更していない。詳細は `docs/autodiff-lr-scheduler-poly-chained-decision.md`。

**適用記録（イシュー #2661・親 #2660。内部クレート限定）**: `Subset`・`ConcatDataset`（と先頭軸連結用の `ConcatBatch`）・`random_split`／`random_split_fractions`（`torch.utils.data` 相当）を内部クレート限定（`fandhe_ai_tensor_core::data`。ホスト側ユーティリティで、新規 `Op`・`BackendOps` メソッド・VJP・カーネルなし。既存の公開型・trait へは何も足していない）で実装した。
facade 公開面は追加していない（保留ガード `DatasetComposeHoldDoctestGuard` と `api_surface.rs` の否定ガードが固定）。公開形（`fandhe_ai::data` への純再エクスポート）は未承認で、承認依頼は #2677（公開自体は承認後の #2679）。
本書 1 節の対象範囲表・`docs/compat-feature-gap.md` の判定列は変更していない。詳細は `docs/tensor-core-dataset-compose-decision.md`。

**適用記録（イシュー #2662・親 #2660。内部クレート限定）**: `IterableDataset`・`IterableDataLoader`（と積み上げ用 `StackSamples`）・`BatchSampler`（`torch.utils.data` 相当）を内部クレート限定（`fandhe_ai_tensor_core::data`。ホスト側ユーティリティで、新規 `Op`・`BackendOps` メソッド・VJP・カーネルなし。既存の公開型・trait へは何も足していない）で実装した。
facade 公開面は追加していない（保留ガード `IterableBatchSamplerHoldDoctestGuard` と `api_surface.rs` の否定ガードが固定）。公開形（`fandhe_ai::data` への純再エクスポート）は未承認で、承認依頼は #2677（公開自体は承認後の #2679）。
本書 1 節の対象範囲表・`docs/compat-feature-gap.md` の判定列は変更していない。詳細は `docs/tensor-core-iterable-dataset-batch-sampler-decision.md`。

**適用記録（イシュー #2670・親 #2668。内部クレート限定）**: `jacobian`・`hessian` を内部クレート限定（`fandhe_ai_autodiff::jacobian_ops`。既存の `Tape::backward`／`backward_create_graph` の要素ごとの繰り返しのみで、新規 `Op`・`BackendOps` メソッド・VJP・カーネルなし。`Var`・`Tape` へ inherent メソッドは足していない）で CPU 実装した。
facade 公開面は追加していない（保留ガード `JacobianHessianHoldDoctestGuard` と `api_surface.rs` の否定ガードが固定）。公開形（`Tape::jacobian`／`Tape::hessian` の委譲メソッド）は未承認で、承認依頼は #2677（公開自体は承認後の #2678）。
本書 1 節の対象範囲表・`docs/compat-feature-gap.md` の判定列は変更していない。詳細は `docs/autodiff-jacobian-hessian-gradcheck-decision.md` 「実装記録（#2670）」。

**適用記録（イシュー #2671・親 #2668。内部クレート限定）**: `gradcheck`・`backward_detect_anomaly` を内部クレート限定（`fandhe_ai_autodiff::gradcheck`・`fandhe_ai_autodiff::anomaly`。既存の `jacobian`／`Tape::backward` の合成のみで、新規 `Op`・`BackendOps` メソッド・VJP・`AutodiffError` variant・カーネルなし。`Var`・`Tape` へ inherent メソッドは足していない）で CPU 実装した。
facade 公開面は追加していない（保留ガード `GradcheckAnomalyHoldDoctestGuard` と `api_surface.rs` の否定ガードが固定）。公開形（`Tape::gradcheck`／`Tape::backward_detect_anomaly` の委譲メソッドと `GradcheckOptions`／`GradcheckReport` の再エクスポート）は未承認で、承認依頼は #2677（公開自体は承認後の #2678）。
本書 1 節の対象範囲表・`docs/compat-feature-gap.md` の判定列は変更していない。詳細は `docs/autodiff-jacobian-hessian-gradcheck-decision.md` 「実装記録（#2671）」。**→ `Tape::gradcheck` と型 2 つは #2847 で公開した（本節末尾の #2847 の適用記録）。**

**適用記録（イシュー #2847・親 #2499 Phase 4。承認: https://github.com/Fandhe-AI/fandhe-ai/issues/2499#issuecomment-6052732061）**: 決定記録 `autodiff-jacobian-hessian-gradcheck-decision.md` §11 の形で、facade `Tape::gradcheck(device, f, inputs, options)`（関連関数・`TapeRef` アダプタ）と `GradcheckOptions`／`GradcheckReport` のクレートルート再エクスポート（別名なし・1 文 1 行）を公開した（追加のみ・`fandhe-ai =0.10.0` の既存 API は不変）。内部 `gradcheck` の `make_tape` 境界は `Fn() -> Result<Tape, AutodiffError>` へ変更（出荷済み API 外）。保留ガード `GradcheckAnomalyHoldDoctestGuard` は公開した名前の分だけ反転し、正ガード `facade_tape_gradcheck_matches_approved_shape`・`facade_reexports_gradcheck_types_only_in_approved_shape` を新設した。維持した保留: モジュール名 `gradcheck`／`anomaly`・裸の自由関数・`Var`／`Tensor<f32>` 上の同名メソッド。`TapeRef` の公開面は不変（`custom` を持たないため `gradcheck` のクロージャ内で `CustomFunction` は使えない。追加は承認範囲外）。本書 1 節の対象範囲表・`compat-feature-gap.md` の判定列は変更していない。CUDA／Metal 実機は未実測（`docs/perf/logs/tape-gradcheck-facade-2847/README.md`）。詳細は決定記録 §12。

**適用記録（イシュー #2850・親 #2499 Phase 4。承認: https://github.com/Fandhe-AI/fandhe-ai/issues/2499#issuecomment-6052732061）**: 行 12・13 の公開形（決定記録 `autodiff-pool3d-ops-decision.md` §12.1・`autodiff-conv-transpose3d-max-unpool-decision.md` §12.1）のとおり、`Var::max_pool3d`（`(Var, Tensor<i32>)` を返す）・`avg_pool3d`・`conv_transpose3d`・`max_unpool1d`・`max_unpool2d`・`max_unpool3d` の 6 本を `Var` の 1 行委譲として公開した（追加のみ・`fandhe-ai =0.10.0` の既存 API は不変。委譲本体は `var_phase4_ops_methods_are_thin_delegations` が固定）。保留ガード `Pool3dOpsHoldDoctestGuard`・`ConvTranspose3dMaxUnpoolHoldDoctestGuard` は公開した名前（`Var` 受け手）の分だけ反転した。維持した保留: 層化（`nn::*` 層型・`compat::Sequential::add_*`）、モジュール名・内部型、`Tape`／`Tensor<f32>` 上の同名メソッド。承認コメントが明示したのは「行 12〜15 は `Var` 委譲に限って公開」「層化は保留継続」の 2 点で、シグネチャ確定は各決定記録 §7 の 1 案から導いた事項である。本書 1 節の対象範囲表・`compat-feature-gap.md` の判定列は変更していない。CUDA／Metal 実機は未実測（`docs/perf/logs/pool3d-ops-2643/README.md`・`docs/perf/logs/conv-transpose3d-max-unpool-2644/README.md`）。詳細は各決定記録 §13。

**適用記録（イシュー #2851・親 #2499 Phase 4。承認: https://github.com/Fandhe-AI/fandhe-ai/issues/2499#issuecomment-6052732061）**: 行 14・15 の公開形（決定記録 `autodiff-fold-unfold-decision.md` §12.1・`autodiff-lrn-weight-reparam-decision.md` §12.1・§12.2）のとおり、`Var::unfold`・`fold`・`local_response_norm`・`weight_norm`・`spectral_norm` の 5 本を `Var` の 1 行委譲として公開し、`SpectralNormState` をクレートルートへ別名なし 1 行で再エクスポートした（追加のみ・`fandhe-ai =0.10.0` の既存 API は不変。行 15 の † は新規 inherent メソッド 3 本とルート再エクスポート 1 型のみで既存シグネチャ・意味論の変更なしと確認。委譲本体は `var_phase4_ops_methods_are_thin_delegations`、再エクスポートの形は `facade_reexports_spectral_norm_state_only_in_approved_shape` が固定）。保留ガード `FoldUnfoldHoldDoctestGuard`・`LrnWeightReparamHoldDoctestGuard` は公開した名前（`Var` 受け手・`SpectralNormState`）の分だけ反転した。維持した保留: 層化（`nn::*` 層型・`compat::Sequential::add_*`）、重み再パラメータ化の結線方式、`norm_except_dim`・`SPECTRAL_NORM_INIT_POWER_ITERATIONS`、モジュール名・内部型、`Tape`／`Tensor<f32>` 上の同名メソッド。承認コメントが明示したのは「行 12〜15 は `Var` 委譲に限って公開」「`Var::unfold` は `nn.functional.unfold` 対応名」「引数順は crate 内 `conv2d` 系」「層化・結線方式は保留継続」で、`SpectralNormState` の位置と `norm_except_dim` 非公開は決定記録 §12 が導出と明記した形である。本書 1 節の対象範囲表・`compat-feature-gap.md` の判定列は変更していない。CUDA／Metal 実機は未実測（`docs/perf/logs/fold-unfold-2645/README.md`・`docs/perf/logs/lrn-weight-reparam-2646/README.md`）。詳細は各決定記録 §13。

**適用記録（イシュー #2931・親 #2873。承認: https://github.com/Fandhe-AI/fandhe-ai/issues/2499#issuecomment-6067263650 の項 1）**: 決定記録 `autodiff-functional-transforms-design.md` §23 の形で、facade `Tape::vjp`・`Tape::hvp`・`Tape::vmap` の 3 メソッド（`F1`〜`F3`）を内部 `fandhe_ai_autodiff::functional_ops` への 1 行委譲として公開した（追加のみ・`fandhe-ai =0.10.0` の既存 API は不変。新規 `Op`・`BackendOps` メソッド・VJP・`AutodiffError` variant・依存・`unsafe` なし）。承認範囲は F1〜F3 の公開と論点 1（単一入力・`FnMut`・`child` は明示引数・`out_dim` なし）・論点 2（`vmap` の bit 一致は契約にしない）・論点 4（低精度 forward は対象外）で、追跡なしに対する挙動は現状のまま doc に明記した（追跡なし `output` の `vjp` は全ゼロ、追跡なし `loss` の `hvp` は `Err` を伝播、追跡なし `input` は両方 `GradientTrackingDisabled`）。論点 5・6 は保留、論点 3 はコメントに言及がない。保留ガード `FunctionalTransformsHoldDoctestGuard` は `Tape` 受け手の 3 名分だけ反転し（`impl ... for fandhe_ai::Tape` と UFCS 3 行を除去）、正ガード `facade_tape_functional_transforms_are_approved_thin_delegations`（シグネチャと 1 行委譲本体をトークン列で固定）と到達性テスト `tape_functional_transforms_are_reachable_via_facade` を新設、宣言インベントリは 4 件から 7 件へ更新した。維持した保留: モジュール `functional_ops` の再エクスポート・裸の自由関数・`Var`／`Tensor<f32>` 上の同名メソッド・`vmap(grad)`（論点 6）・`out_dim`・複数入力・`VarF64`・`supports_create_graph` の対象拡張（論点 5）。本書 1 節の対象範囲表・5.1 節の Phase 4 表・`compat-feature-gap.md` の判定列は変更していない。CUDA／Metal 実機は未実測（`docs/perf/logs/functional-transforms-facade-2931/README.md`）。詳細は設計記録 §25。

**適用記録（イシュー #2874・親 #2841。内部クレート限定）**: `vjp` を内部クレート限定（`fandhe_ai_autodiff::functional_ops`。既存の `Tape::backward` と `mul` の合成のみで、新規 `Op`・`BackendOps` メソッド・VJP・`AutodiffError` variant・カーネルなし。`Var`・`Tape` へ inherent メソッドは足していない）で実装した。 → #2930 で承認範囲と公開形を設計記録 §23 に記録（公開は #2931）。
facade 公開面は追加していない（保留ガード `FunctionalTransformsHoldDoctestGuard` と `api_surface.rs` の否定ガードが固定。`hvp`・`vmap` の名前も先に締めてある）。公開形は未承認。本書 5.1 節の表・1 節の対象範囲表・`docs/compat-feature-gap.md` の判定列は変更していない。詳細は `docs/autodiff-functional-transforms-design.md` 「15. 実装記録（#2874）」。 → 公開形の承認依頼は #2879 で §5.1 末尾の `F1`〜`F3`（Phase 8 公開形・関数型 AD 変換）へ追加した。

**適用記録（イシュー #2875・親 #2841。内部クレート限定）**: `hvp` を内部クレート限定（`fandhe_ai_autodiff::functional_ops::hvp`。既存の `backward_create_graph` と子テープ上の `mul`／`backward` の合成のみで、新規 `Op`・`BackendOps` メソッド・VJP・`AutodiffError` variant・カーネルなし）で実装した。facade 公開面は追加していない（保留ガードの宣言インベントリへ `functional_ops.rs::hvp` を登録して固定）。公開形は未承認。本書 5.1 節の表・1 節の対象範囲表は変更していない。詳細は `docs/autodiff-functional-transforms-design.md` 「16. 実装記録（#2875）」。 → 公開形の承認依頼は #2879 で §5.1 末尾の `F1`〜`F3`（Phase 8 公開形・関数型 AD 変換）へ追加した。 → #2930 で承認範囲と公開形を設計記録 §23 に記録（公開は #2931）。

**適用記録（イシュー #2876・親 #2841。内部クレート限定）**: ループ版 `vmap` を内部クレート限定（`fandhe_ai_autodiff::functional_ops::vmap`。既存の `unbind`・`contiguous`・`stack` の合成のみで、新規 `Op`・`BackendOps` メソッド・VJP・`AutodiffError` variant・カーネルなし）で実装した。facade 公開面は追加していない（保留ガードの宣言インベントリへ `functional_ops.rs::vmap` を登録して固定）。公開形は未承認。本書 5.1 節の表・1 節の対象範囲表は変更していない。詳細は `docs/autodiff-functional-transforms-design.md` 「17. 実装記録（#2876）」。 → 公開形の承認依頼は #2879 で §5.1 末尾の `F1`〜`F3`（Phase 8 公開形・関数型 AD 変換）へ追加した。 → #2930 で承認範囲と公開形を設計記録 §23 に記録（公開は #2931）。

**適用記録（イシュー #2940・親 #2939。内部クレート限定）**: double-VJP 法の `jvp`／`jacfwd` を内部クレート限定（`fandhe_ai_autodiff::functional_ops` の `pub(crate)`。既存の `backward_create_graph` と子テープ上の `mul`／`backward` の合成のみで、新規 `Op`・`BackendOps` メソッド・VJP・`AutodiffError` variant・依存・`unsafe`・tolerance・baseline なし）で実装した。facade 公開面は追加していない（保留ガードの正のプローブ・固定文言・宣言インベントリへ `jvp`／`jacfwd` を登録して固定）。公開形は未承認。本書 5.1 節の表・1 節の対象範囲表は変更していない。詳細は `docs/autodiff-functional-transforms-design.md` 「24. 実装記録（#2940）」。→ 公開形の承認依頼は #2941 で 5.1 節末尾の `F4`〜`F5` へ追加した（設計記録 §27）。

**Phase 8 公開形（承認依頼 #2883・親 #2882・Phase 8 #2872）**: speculative decoding・連続バッチングの公開形と判定方式の承認依頼。上の Phase 4 表（行 1〜30）とは別系統のため、行ラベルは `S1`〜`S3` とし Phase 4 の番号空間と混ぜない（Phase 4 表の「行と保留ガードは `*HoldDoctestGuard` と 1 対 1」という前提も、保留ガード未設置のこの 3 行には当てはまらない）。以下は設計記録 `docs/facade-speculative-decoding-batching-design.md` §5・§6・§9・§10 の**転記**であり、本節で新しい推奨・tolerance・baseline は作っていない。すべて**未承認**で、承認は実装 Agent が代行しない。ルート #2499 の 2026-10-08 コメント（`https://github.com/Fandhe-AI/fandhe-ai/issues/2499#issuecomment-6052732061`）の承認範囲は設計の記録までで、案 C の境界の承認（`https://github.com/Fandhe-AI/fandhe-ai/issues/2499#issuecomment-6033824965`、2026-10-07）は「公開面の追加・数値判定方式・`Op`／`BackendOps` の拡張」を承認していない。公開（コード・`pub use`・保留ガードの反転）は承認後に別 issue（設計記録 §11 の仮番号 9）で行う。 → 承認（`issuecomment-6067263650` 項 1）と確定した公開形は本ブロック末尾の「承認と公開形の記録（#2933）」と設計記録 §17。

| # | 機能（由来） | 公開形（設計記録 §5 の転記・未承認） | 非破壊性 | 保留ガード | 公開先 | 決定記録・承認事項の所在 |
|---|---|---|---|---|---|---|
| S1 | `generate_speculative`（speculative decoding・greedy） | `fandhe_ai::inference` への追加。`GENERATE_APPROVED_REEXPORT` の 1 文は変えず別の `pub use` 文で足す。シグネチャは target・draft・`input_ids`・`&GenerateConfig`・`&SpeculativeConfig` を受け `Result<Tensor<i32>, AutodiffError>` を返す形に「相当」（正確な型パラメータ境界は未確定）。受理統計は返さない。B = 1 限定 | 追加のみ（`fandhe-ai =0.10.0` の既存シグネチャ・意味論・`FitConfig` は不変） | 未設置（設計記録に名前なし・設置担当 issue も未確定） | 未公開（承認待ち） | 設計記録 §5・§6.1・§7・§10 |
| S2 | `SpeculativeConfig` | `#[non_exhaustive]`。フィールドは draft の先読み長 `k` のみ。`k == 0` は `Err(InvalidArgument)`、残り長を超える `k` は残り長に丸める | 同上 | 同上 | 同上 | 設計記録 §5・§7 |
| S3 | `BatchScheduler`（連続バッチング第 1 段階。名前は案） | 素の `struct`。`new(limits)`（上限は必須）・`submit(input_ids, &GenerateConfig) -> Result<RequestId, AutodiffError>`・`step(&mut self, model) -> Result<usize, AutodiffError>`・`take_finished() -> Vec<(RequestId, Tensor<i32>)>`。要求ごとに `GenerateConfig` と独立シードを持つ。内部状態保持型モデルは対象外であることを doc の契約にする | 同上 | 同上 | 同上 | 設計記録 §5・§6.3・§7・§8・§10 |

エラー型は既存の `AutodiffError` を流用し、新しい型・variant は足さない（設計記録 §5）。

判定方式（設計記録 §6。未承認）:

- greedy で状態を持たないテスト用モデルは token 列の完全一致。KV キャッシュ付きの実モデルでは token 列一致を事前登録の仮説とし、契約上の判定は位置ごとの logits に対する既存の統一複合判定（相対誤差 1e-3 未満 または 絶対誤差 1e-5 未満）に帰着させる。bit 一致は確定事実として書かない。
- 連続バッチング第 1 段階は、各要求を単独で `generate` した結果との token 列完全一致。
- サンプリング版は提案なし（論点 1）。新しい tolerance・baseline は作らない。

拡張要否（設計記録 §9。推奨・未承認）: 依存追加・新規 `unsafe`・新規 `Op`／`BackendOps`／VJP はいずれも不要の見込み。

承認依頼する論点（設計記録 §10 の転記。推奨は設計記録にあるものだけ）:

| 論点 | 内容 | 選択肢（設計記録にあるもの） | 推奨 |
|---|---|---|---|
| 1 | サンプリング版の分布一致の判定契約（統計検定の標本数・有意水準） | 設計記録に列挙なし（§6.2 は tolerance・標本数を提案しないと明記） | 推奨なし。承認までサンプリング版（§11 の 6）はブロック |
| 2 | greedy の token 列一致の仮説が実モデルで破れた場合の扱い | 設計記録に列挙なし（§6.1 は margin 付き入力で避け、破れたら承認依頼に戻すとする） | 推奨なし |
| 3 | 内部状態保持型モデル（`facade-generate-decision.md` §17.4）への到達経路 | `caches` を使う形の公開／`AutoregressiveModel` の既定メソッドとしての巻き戻しフック（既存実装は壊れないが trait 拡張で公開面が変わる）／別 trait（既存に影響しないが型パラメータが増える） | 推奨なし（設計記録上「いずれも未決」） |
| 4 | `KvCache` の公開メソッド追加（`truncate` 等）と「単一の書き手」不変条件の緩和 | §4.1 の (i) clone 保存・復元（公開面は増えない）／(ii) `pub(crate)` の切り詰め／(iii) `KvCache::truncate` の `pub` 追加 | 設計記録 §4.1 の推奨は (i)、次点は (ii)。(iii) は公開面の追加で承認事項 |
| 5 | テンソル単位バッチ化に要る padding mask・可変長キャッシュ | 同じ `S_cached` の要求のグルーピング／padding mask・行単位 gather（§8.2） | 推奨なし（§8.1 の第 1 段階はこれを必要としない形） |
| 6 | facade 公開面の追加そのもの | 上表 S1〜S3 の 1 案 | 設計記録 §5 の 1 案（未承認） |
| 7 | paged attention | 条件付き（K-3 のデバイス常駐 KV と CPU 参照実装が成立するまで） | 推奨なし |
| 8 | 連続バッチングで内部状態保持型モデルを扱うための要求ごとのモデル状態の分離方式 | 要求ごとに別インスタンスを渡す／モデル側の状態の退避・復元フック（trait 拡張か別 trait かは論点 3 と同じ） | 推奨なし（決まるまで第 1 段階は対象外） |

設計記録に形が書かれていない点（承認時に決めてほしい事項。本節は推奨を作らない）:

- 保留ガードの名前と、どの issue で設置するか（設計記録 §5 は「承認までは保留ガードで固定」と方針のみ）。
- `RequestId` の型の定義。
- `BatchScheduler::new(limits)` の `limits` の型・フィールド構成（同時要求数・キュー長・要求ごとの `max_length` の上限は必須、とだけ記録されている）。
- `step` の `model` 引数の型（ジェネリック境界・`?Sized` の有無）。
- 要求の途中で `forward_step` が失敗したときの失敗の表現型（設計記録 §7 は実装 issue で決めるとしている）。
- `generate_speculative` の正確な型パラメータ境界。

本書 1 節の対象範囲表・`docs/compat-feature-gap.md` の判定列は変更していない。facade 公開面は追加していない。

**承認と公開形の記録（#2933・親 #2932）**: 上の `S1`〜`S3` は承認依頼時点の転記として残す。ルート #2499 のリポジトリ所有者コメント（`https://github.com/Fandhe-AI/fandhe-ai/issues/2499#issuecomment-6067263650`、2026-10-08）の項 1 で、第 1 段階の 3 名（`generate_speculative`〈greedy・B = 1〉・`SpeculativeConfig`・`BatchScheduler`）の公開が、設計記録 §5 の推奨形に main の内部実装のシグネチャをそのまま当てた形で承認された。KV の巻き戻しは案 (i) のまま（`KvCache` の公開メソッドは増やさない）、内部状態保持型が対象外であることは doc の契約として明記する。確定した公開形（完全シグネチャ・`pub use` 文・ガード反転範囲）は設計記録 `docs/facade-speculative-decoding-batching-design.md` §17 を正とする。本記録は docs のみで、公開（コード・`pub use`・ガード反転）は #2934 が行うため、現時点の公開状態は未公開である。 → 公開は #2934 で実施済み（設計記録 §18）。

| # | 公開名 | 確定シグネチャ（main の内部実装） | 公開パス | 保留ガード | 公開状態 |
|---|---|---|---|---|---|
| S1 | `generate_speculative` | `<T, D>(target: &T, draft: &D, input_ids: &Tensor<i32>, config: &GenerateConfig, spec: &SpeculativeConfig) -> Result<Tensor<i32>, AutodiffError>`（`T`・`D` は `AutoregressiveModel + ?Sized`。Greedy 以外と `num_kv_layers() == 0` は `Err`） | `fandhe_ai::inference` | S 専用は未設置。#2934 で正ガード `facade_exposes_speculative_scheduler_items_only_in_approved_shape` を新設 | 公開済み（#2934） |
| S2 | `SpeculativeConfig` | `#[non_exhaustive]`・`Debug, Clone, PartialEq`・`pub k: usize`・`new(k: usize) -> SpeculativeConfig` | 同上 | 同上 | 同上 |
| S3 | `BatchScheduler`（署名に現れる `RequestId`・`SchedulerLimits` を含む） | `new(SchedulerLimits)`・`queued_len`・`active_len`・`submit(&mut self, &Tensor<i32>, &GenerateConfig) -> Result<RequestId, AutodiffError>`・`step<M: AutoregressiveModel + ?Sized>(&mut self, &M) -> Result<usize, AutodiffError>`・`take_finished() -> Vec<(RequestId, Tensor<i32>)>`・`take_failed() -> Vec<(RequestId, AutodiffError)>`。`RequestId` は不透明 `u64` newtype、`SchedulerLimits::new(max_active, max_queued, max_length) -> Result<SchedulerLimits, AutodiffError>`（`#[non_exhaustive]`・0 は拒否） | 同上 | 同上 | 同上 |

「承認時に決めてほしい事項」の決着: 保留ガードの名前と設置担当は #2934（正ガードの新設。承認形の定数は `GENERATE_APPROVED_REEXPORT` と別の `pub use` 文）／`RequestId` は不透明 `u64` newtype／`limits` は `SchedulerLimits`（3 上限必須）／`step` の `model` は `&M`（`M: AutoregressiveModel + ?Sized`）／失敗の表現型は `take_failed() -> Vec<(RequestId, AutodiffError)>`／`generate_speculative` の型パラメータ境界は `T`・`D` とも `AutoregressiveModel + ?Sized`。

論点表の決着: 論点 4 は案 (i) で確定（`KvCache` の公開メソッドは増やさない）／論点 6 は第 1 段階の 3 名に限って承認／論点 1・2・3・5・7・8 は保留継続（サンプリング版・内部状態保持型・padding mask・paged attention 等は公開しない）。保留ガードは公開した名前の分だけ正ガードへ反転し、未承認経路のプローブは残す。本書 1 節の対象範囲表と `docs/compat-feature-gap.md` の判定列は #2934 の適用記録で扱い、本記録では変更していない。

**Phase 8 公開形（テキスト変換。承認依頼 #2896・親 #2895・Phase 8 #2872）**: 語彙 lookup 型テキスト変換（Keras `TextVectorization` 相当）の配置・公開形・判定方式の承認依頼。上の Phase 4 表（行 1〜30）・直前の `S1`〜`S3` とは別系統のため、行ラベルは `T1`〜`T5` とし他の番号空間と混ぜない（Phase 4 表の「行と保留ガードは `*HoldDoctestGuard` と 1 対 1」という前提も、保留ガード未設置のこの 5 行には当てはまらない）。以下は設計記録 `docs/facade-text-vectorization-design.md` §5・§6・§7・§8・§9・§10 の**転記**であり、本節で新しい推奨・tolerance・baseline は作っていない。すべて**未承認**で、承認は実装 Agent が代行しない。承認根拠は範囲の異なる 2 つを別々に引く。ルート #2499 の 2026-10-07 コメント（`https://github.com/Fandhe-AI/fandhe-ai/issues/2499#issuecomment-6033824965`）は #2618 折衷案の**範囲**（語彙 lookup 型の単語／文字レベル変換を対象内、サブワードトークナイザと Unicode 正規化表の自作を対象外）だけを承認しており、spec 変更履歴（`docs/spec/04-requirements.md:459`）は「配置・公開 API の追加は承認しない」と明記している。2026-10-08 コメント（`https://github.com/Fandhe-AI/fandhe-ai/issues/2499#issuecomment-6052732061`）の承認範囲は設計と issue 分解の記録までである。公開（コード・`pub use`・保留ガードの反転）は承認後に別 issue（設計記録 §11 の仮番号 9）で行う。 → 承認（`issuecomment-6067263650` 項 1）と確定した公開形は本ブロック末尾の「承認と公開形の記録（#2936）」と設計記録 §16。

| # | 機能（由来） | 公開形（設計記録 §5・§6 の転記・未承認） | 非破壊性 | 保留ガード | 公開先 | 決定記録・承認事項の所在 |
|---|---|---|---|---|---|---|
| T1 | `text`（モジュール） | facade 内の新モジュール `fandhe_ai::text`。実体は `crates/facade/src/text/`。採らない案は 3 つ（コア外の新クレート・`tensor-core`・`autodiff` 配下）。`Sequential` の層にはしない（独立 struct） | 追加のみ（`fandhe-ai =0.10.0` の既存シグネチャ・意味論・`FitConfig` は不変） | 未設置（設計記録に名前なし・設置担当 issue も未確定。方針は設計記録 §6 末尾の `*HoldDoctestGuard` の正のプローブ doctest と `crates/facade/tests/api_surface.rs` の否定ガード） | 未公開（承認待ち） | 設計記録 §5・§6 |
| T2 | `TextVectorization` | 非公開フィールドの struct（`Clone`・`Debug`。`Debug` は語彙の件数だけを出す）。メソッドは `from_vocabulary`・`adapt`・`transform`（`[B, L]` の `Tensor<i32>`）・`vocabulary`（index 0 = `""`、1 = `"[UNK]"`）・`vocabulary_size`・`config` の 6 本。メソッド名 `transform` と型名はそれ自体が承認論点 | 同上 | 同上 | 同上 | 設計記録 §6 |
| T3 | `TextVectorizationConfig`・`Standardize`・`Split` | `TextVectorizationConfig` は `#[non_exhaustive]`・`Debug, Clone, PartialEq, Eq`。フィールドは `max_tokens`・`standardize`・`split`・`ngrams`・`output_sequence_length`・`limits`。`Default` と `with_*` ビルダ。`Standardize` は 4 値（`None`・`Lower`・`StripPunctuation`・`LowerAndStripPunctuation`、既定は `LowerAndStripPunctuation`）、`Split` は 3 値（`None`・`Whitespace`・`Character`、既定は `Whitespace`）。いずれも `#[non_exhaustive]` | 同上 | 同上 | 同上 | 設計記録 §3・§6 |
| T4 | `TextLimits` | `#[non_exhaustive]`・`Debug, Clone, Copy, PartialEq, Eq`。フィールドは設計記録 §8 の各上限、`Default` と `with_*`。設定値が絶対上限を超えたら `Err`。API の入力上限でありガードレール閾値ではない | 同上 | 同上 | 同上 | 設計記録 §6・§8 |
| T5 | `TextError` | facade 独自の自己完結型。`#[non_exhaustive]`・`Debug`。`AutodiffError` は使わない。variant 案は 14 個: `BatchTooLarge`・`InputTooLong`・`CorpusTooLarge`・`VocabularyTooLarge`・`VocabularyTokenTooLong`・`EmptyVocabularyToken`・`ReservedVocabularyToken`・`DuplicateVocabularyToken`・`InvalidMaxTokens`・`InvalidNgrams`・`OutputSequenceLengthTooLarge`・`TooManyDistinctTokens`・`OutputTooLarge`・`Shape(ShapeError)`。エラーには入力文字列の中身を入れない（index と長さのみ） | 同上 | 同上 | 同上 | 設計記録 §6・§13 |

判定方式（設計記録 §7。未承認）:

- 出力はすべて `i32` の id で浮動小数点演算・GPU カーネル・`BackendOps` を通らないため、判定は整数の完全一致（手書きの期待 id 列との `==`）とする。統一複合判定（相対誤差 1e-3 未満 または 絶対誤差 1e-5 未満）は、id が大きいと 1 ずれでも相対誤差が 1e-3 未満になり通ってしまうため適用しない。
- tolerance は新設せず、既存の tolerance も変更しない。FMA 契約と f64 アキュムレータ契約は浮動小数点の縮約がないので対象外。
- CUDA／Metal の実機 parity は対象がない（ホスト側のみの処理）。`#[ignore]` テストも `docs/perf/logs/` への申し送りも作らない。
- `Var::embedding(.., Some(0))` との結線テストでは、形状が `[B, L, D]` になることとパディング位置の勾配が 0 であることを確かめる。

拡張要否（設計記録 §9。推奨・未承認）: 依存追加・新規 `unsafe`・新規 `Op`／`BackendOps`／VJP はいずれも不要。別承認で必要になりうるもの: Unicode の大小文字変換、Unicode White_Space 分割、書記素クラスタ、callable、`multi_hot` 系の出力モード、`Sequential` の層化。入力上限（バッチ要素数・文字列バイト数・語彙数・n-gram 数・出力要素数等）の既定値・絶対上限は設計記録 §8 の表を正とし、本節では数値を二重管理しない。

承認依頼する論点（設計記録 §10 の転記。推奨は設計記録にあるものだけ）:

| 論点 | 内容 | 選択肢（設計記録にあるもの） | 推奨 |
|---|---|---|---|
| 1 | facade 公開面の追加（本書 §5 経路 2）と名前（`text`・`TextVectorization`・`transform`） | 上表 T1〜T5 の 1 案。別名の候補は設計記録にない | 設計記録 §6 の 1 案（未承認） |
| 2 | 配置の確定 | facade 内部モジュール／コア外の新クレート／`tensor-core`／`autodiff` 配下（設計記録 §5） | facade 内の `fandhe_ai::text`（設計記録 §5。採らない案の理由は同節を参照） |
| 3 | 小文字化と空白の定義 | ASCII（`to_ascii_lowercase`・`split_ascii_whitespace`）／Unicode の大小文字変換・Unicode White_Space（条件付き・別承認）（設計記録 §3） | 設計記録 §3 は ASCII を対象内、Unicode を条件付きとしている。Keras／TF の既定挙動との差は要出典確認 |
| 4 | Keras との bit 互換を契約にするか | 契約にする（Python／TF の golden fixture と生成ツールの持ち込みが要る）／契約にしない（設計記録 §7） | 契約にしない。本ライブラリの仕様を doc で定義し、テストは手書きの期待値で行う（設計記録 §7） |
| 5 | `TextLimits` の既定値・絶対上限 | 設計記録 §8 の表の値。`with_*` で変えられるが絶対上限は超えられない | 設計記録 §8 の表の既定値（推奨）列。数値は同節を参照先とし転記しない |
| 6 | 出力モード `multi_hot`／`count`／`tf_idf`・`StringLookup` 相当を後で入れるか（一次資料の確認を含む） | 設計記録 §3 は第 1 段階の対象外（条件付き）とし、#2618 §9.5 にならい一次資料の確認後に別途範囲を決める、とだけ書く | 推奨なし |
| 7 | `Sequential` の層化（`Tensor<f32>` 前提とのずれ） | 独立 struct で層化を保留／`add_text_vectorization` 等で層化（設計記録 §5） | 第 1 段階は独立 struct とし層化は保留（設計記録 §5。`Sequential` は `Tensor<f32>` 入力前提のため） |
| 8 | callable の標準化・分割 | `Box<dyn Fn>` を公開面に加える／入れない（設計記録 §3） | 第 1 段階では入れない（設計記録 §3） |

設計記録の Keras に関する記述（`standardize` の 4 値・`max_tokens` の数え方・n-gram の順・既定挙動）は一次資料で未確認の「要出典確認」であり、本節でも事実として断定しない。

設計記録に形が書かれていない点（承認時に決めてほしい事項。本節は推奨を作らない）:

- 保留ガードの名前と、どの issue で設置するか（設計記録 §6 は方針のみ。#2897〜#2903 はいずれも `pub(crate)` の内部実装で、ガードを置く issue は未確定）。
- `TextLimits` の各フィールド名（設計記録 §6 は「§8 の各上限」とだけ記録）。
- 各 `with_*` ビルダのメソッド名。
- `Standardize`・`Split` の derive 一式（設計記録 §6 は `Default` だけを記録）。
- `TextError` の各 variant のフィールドの型。
- `Standardize` などを `fandhe_ai::text` 配下だけに置くか、クレートルートにも置くか。
- `transform` という名前の確定（論点 1 に含まれる）。

本書 1 節の対象範囲表・`docs/compat-feature-gap.md` の判定列は変更していない。facade 公開面は追加していない。

**承認と公開形の記録（#2936・親 #2935）**: 上の `T1`〜`T5` は承認依頼時点の転記として残す。ルート #2499 のリポジトリ所有者コメント（`https://github.com/Fandhe-AI/fandhe-ai/issues/2499#issuecomment-6067263650`、2026-10-08）の項 1 で、テキスト変換の facade 公開が、設計記録 §6 の推奨形に main の内部実装のシグネチャを当てた形で承認された。論点 1・2・4・5・7・8 は推奨どおり、論点 3 は ASCII のみ（Unicode の小文字化・空白は保留）、論点 6 は保留である。確定した公開形（完全シグネチャ・`TextError` の 17 variant・記録と実装の差分の処置・ガード反転範囲）は設計記録 `docs/facade-text-vectorization-design.md` §16 を正とする。本記録は docs のみで、公開（コード・`pub mod text`・ガード反転）は #2937 が行うため、現時点の公開状態は未公開である。 → 公開は #2937 で実施済み（設計記録 §17。下の「適用記録（イシュー #2937）」参照）。

| # | 公開名 | 確定シグネチャ（要約。正は設計記録 §16.3） | 公開パス | 保留ガード | 公開状態 |
|---|---|---|---|---|---|
| T1 | `text` | `pub mod text`。6 名をフラットに `pub use`（サブモジュールは非公開・クレートルートへ再エクスポートしない） | `fandhe_ai::text` | 専用の `*HoldDoctestGuard` はない。#2937 で `text_module_hygiene.rs::text_module_is_exposed_only_in_approved_shape`（旧 `text_module_is_not_exposed_from_facade` を反転）と `api_surface.rs::facade_exposes_text_items_only_in_approved_shape`・`text_is_public_and_submodules_stay_private`・`text_public_shape_is_pinned`・`text_unapproved_paths_are_absent` の正ガード・否定プローブへ置換 | 公開済み（#2937） |
| T2 | `TextVectorization` | `from_vocabulary`・`adapt`（`Result<Self, TextError>`）・`transform`（`Result<Tensor<i32>, TextError>`、`[B, L]`）・`vocabulary`・`vocabulary_size`・`config` の 6 本。`Clone`・手書き `Debug` | `fandhe_ai::text::TextVectorization` | 同上 | 公開済み（#2937） |
| T3 | `TextVectorizationConfig`・`Standardize`・`Split` | config は `#[non_exhaustive]`・`Debug, Clone, PartialEq, Eq, Default` で 6 つの `pub` フィールドと `with_*` 6 本（`-> Self`）。`Standardize`・`Split` は `#[non_exhaustive]`・`Debug, Clone, Copy, PartialEq, Eq, Default` | `fandhe_ai::text::{TextVectorizationConfig, Standardize, Split}` | 同上 | 公開済み（#2937） |
| T4 | `TextLimits` | `#[non_exhaustive]`・`Debug, Clone, Copy, PartialEq, Eq`・`Default`。`with_<フィールド名>(self, usize) -> Result<Self, TextError>` が 9 本（`max_batch`・`max_input_bytes`・`max_corpus_bytes`・`max_vocabulary_size`・`max_vocabulary_token_bytes`・`max_distinct_tokens`・`max_ngrams`・`max_output_sequence_length`・`max_output_elements`）。アクセサは公開しない | `fandhe_ai::text::TextLimits` | 同上 | 公開済み（#2937） |
| T5 | `TextError` | `#[non_exhaustive]`・`Debug`・`Display`・`std::error::Error`。variant は 17 個（§6 の 14 個に `LimitAboveAbsoluteMaximum { limit: &'static str, value, max }`・`NgramCountOverflow { tokens, n }`・`FrequencyOverflow` を追加）。フィールド型は `usize`、`Shape(ShapeError)` | `fandhe_ai::text::TextError` | 同上 | 公開済み（#2937） |

「承認時に決めてほしい事項」の決着: 保留ガードの名前と設置担当は上表のとおり（T 専用ガードは作らず #2937 が既存否定ガードを反転）／`TextLimits` のフィールド名 9 個と `with_*` 名は T4／`Standardize`・`Split` の derive は T3／`TextError` の variant のフィールド型は T5／`fandhe_ai::text` 配下のみでクレートルートへは置かない／`transform` の名前は確定。

論点の決着: 1・2・4・5・7・8 は推奨どおり確定。3 は ASCII 部分のみ確定で **Unicode の大小文字変換・Unicode White_Space 分割は保留のまま**。**6（出力モード・`StringLookup` 相当）は保留のまま**。数値（既定値・絶対上限）は設計記録 §8 を正とし二重管理しない。本書 1 節の対象範囲表・`docs/compat-feature-gap.md` の判定列は #2937 の適用記録で扱い、本記録では変えない。

**適用記録（イシュー #2937・親 #2935。承認: `https://github.com/Fandhe-AI/fandhe-ai/issues/2499#issuecomment-6067263650` の項 1）**: 語彙 lookup 型のテキスト変換を設計記録 §16 の確定形どおり `fandhe_ai::text::{TextVectorization, TextVectorizationConfig, TextLimits, Standardize, Split, TextError}` として公開した（`lib.rs` の `pub mod text;`、`text/mod.rs` の `pub use` 5 文。サブモジュールは非公開・クレートルートへ再エクスポートしない）。小文字化と空白分割は ASCII に限ることを公開 doc に明記し、利用例 doctest（`adapt` → `transform` → `Var::embedding(.., Some(0))`）を付けた。保留ガードは公開した名前の分だけ正ガードへ反転し、未承認経路（Unicode 系 variant・callable・出力モード・`StringLookup` 相当・`Sequential` への層化・内部項目・クレートルート再エクスポート）の否定プローブは残した。論点 3 の Unicode 部分と論点 6 は保留のまま。本書 1 節の対象範囲表には、#2934 の先例に倣って行を足さない（公開面は新しい互換 API の行ではなく追加の公開モジュールのため）。`Cargo.toml`／`Cargo.lock`・tolerance／baseline・ガードレール閾値・`docs/spec`・`unsafe` は不変。CUDA／Metal の実機 parity はホスト側処理のみのため対象なし。詳細は設計記録 §17。

**Phase 8 公開形（関数型 AD 変換。承認依頼 #2879・親 #2873・Phase 8 #2872）**: `vjp`・`hvp`・ループ版 `vmap` の facade 公開形と論点の承認依頼。上の Phase 4 表（行 1〜30）・`S1`〜`S3`・`T1`〜`T5` とは別系統のため、行ラベルは `F1`〜`F3` とし他の番号空間と混ぜない。保留ガードの対応は Phase 4 表の「行と 1 対 1」でも S／T ブロックの「未設置」でもなく、**`FunctionalTransformsHoldDoctestGuard` 1 個が `F1`〜`F3` の 3 行すべてを覆う**。以下は設計記録 `docs/autodiff-functional-transforms-design.md` §5・§6・§9・§11 の**転記**であり、本節で新しい推奨・tolerance・baseline は作っていない。すべて**未承認**で、承認は実装 Agent が代行しない。承認根拠は範囲の異なる 2 つを別々に引く。ルート #2499 の 2026-10-07 コメント（`https://github.com/Fandhe-AI/fandhe-ai/issues/2499#issuecomment-6033824965`）は案 C の境界だけを承認しており、公開面の追加・数値判定方式・`Op`／`BackendOps` の拡張は承認していない。2026-10-08 コメント（`https://github.com/Fandhe-AI/fandhe-ai/issues/2499#issuecomment-6052732061`）の承認範囲は設計と issue 分解の記録、および実装 issue の起票までである。公開（`pub fn` の追加・保留ガードの反転）は承認後に設計記録 §10 の 7（`vjp`／`hvp`）・8（`vmap`）として別 issue で行う（どちらも未起票）。

| # | 機能（由来） | 公開形（設計記録 §5 の転記・未承認） | 非破壊性 | 保留ガード | 公開先 | 決定記録・承認事項の所在 |
|---|---|---|---|---|---|---|
| F1 | `vjp`（#2874） | facade `Tape` の委譲メソッド `Tape::vjp(&self, output: &Var<'_>, input: &Var<'_>, cotangent: &Tensor<f32>) -> Result<Tensor<f32>, AutodiffError>`。`Var` 委譲・モジュール再エクスポート・裸の自由関数の形ではない | 追加のみ（`fandhe-ai =0.10.0` の既存シグネチャ・意味論・`FitConfig` は不変） | `FunctionalTransformsHoldDoctestGuard` ＋ `crates/facade/tests/api_surface.rs` の否定ガードと宣言インベントリ（`grad.rs::vjp`・`functional_ops.rs::{vjp, hvp, vmap}` に加え、#2931 で `facade/src/lib.rs::{vjp, hvp, vmap}` を足した 7 件）。#2931 で `Tape` 受け手の 3 名分だけ部分反転し、正ガード `facade_tape_functional_transforms_are_approved_thin_delegations` を新設 | 公開済み（#2931。承認は §23） | 設計記録 §3・§5・§6・§11・§15 |
| F2 | `hvp`（#2875） | `Tape::hvp(&self, loss: &Var<'_>, input: &Var<'_>, vector: &Tensor<f32>, child: &Tape) -> Result<Tensor<f32>, AutodiffError>`。`child` は呼び出し側が用意する空の子テープ（`Tape::hessian` と同じ契約） | 同上 | 同上（`F1`〜`F3` で 1 個を共有） | 公開済み（#2931。§10 の 7） | 設計記録 §4・§5・§6・§11・§16 |
| F3 | ループ版 `vmap`（#2876） | `Tape::vmap<'t, F>(&'t self, input: &Var<'t>, in_dim: usize, f: F) -> Result<Var<'t>, AutodiffError> where F: FnMut(&Var<'t>) -> Result<Var<'t>, AutodiffError>`。結果は同じテープ上の微分可能な `Var` | 同上 | 同上 | 公開済み（#2931。承認は §23。§10 の 8） | 設計記録 §5・§6・§7・§11・§17 |

- エラー型: 新しい型・variant を足さず、既存 `AutodiffError` を再利用する（設計記録 §5）。
- 判定方式（設計記録 §6。未承認）: REQ-2 の統一複合判定（相対誤差 1e-3 未満 または 絶対誤差 1e-5 未満）をそのまま使う。`hvp` は 1 階 VJP と bit 同一を主張しない。`vmap` はバッチなし実行との一致を見る。新しい tolerance・baseline は作らない。
- 拡張要否（設計記録 §9。推奨・未承認）: 依存追加・新規 `unsafe`・新規 `Op`／`BackendOps`／VJP はいずれも不要の見込み。
- 検証の状態: `vmap` と `vjp`／`hvp` の合成検証（#2878）は `crates/autodiff/tests/functional_ops_composition.rs` として実装済みで、`hvp = hessian·v` との一致を確認済み（設計記録 §20）。#2878 は 2026-10-08 に close した（本節の作成時点では open だった）。本承認依頼はこの実装済み範囲のみを根拠にする。CUDA/Metal 実機 parity（#2881。2026-10-08 に close）は `#[ignore]` テストと `docs/perf/logs/functional-transforms-2881/README.md` の申し送りまでで（設計記録 §21）、実機では未実測のため成立したとは書かない。

実装済みの内部シグネチャ（`crates/autodiff/src/functional_ops.rs` の `vjp`・`hvp`・`vmap`）と設計記録 §5 の公開形の差（承認者が確認する事項）:

1. **受け手の違い（シグネチャ差ではない）**: §5 は facade `Tape` のメソッド、内部は第 1 引数に `tape: &fandhe_ai_autodiff::Tape` を取る自由関数。`&self.0`（`hvp` の `child` は `&child.0`）を渡す 1 行委譲で吸収できる見込みで、`Tape::hessian` が先例。`vmap` は `&'t self` から `&'t self.0` を渡す形になり、`Tape::var(&self) -> Var<'_>` と同じ寿命の結び方になる見込み。コンパイル可否は確定事項ではなく、公開 issue で確かめる。
2. **実装で確定した細部**（§5・§11 の論点 1 では未確定だった点。新しい推奨ではなく「実装済みの形」）: 単一入力のみ／`vmap` の出力バッチ軸は先頭（dim 0）固定で `out_dim` は無い／クロージャ境界は `FnMut`／`hvp` は `child` 引数が必須／`cotangent`・`vector` は shape の完全一致を要求しブロードキャストしない／`hvp` の `loss` は要素数 1 に限る（`[]`・`[1]`・`[1, 1]` は可）／`vmap` は空バッチを `InvalidArgument` で拒否する。
3. **意味論の非対称**: 追跡なしの `output` に対して `vjp` は全ゼロを返す（設計記録 §15）。追跡なしの `loss` に対して `hvp` は `backward_create_graph` の `Err` を伝播する（§16。`hessian` に合わせたため）。公開時にこの非対称を契約として固定するかは承認者の判断を要する。
4. **保留ガードの反転範囲**: §5 は「facade `Tape` の薄い委譲」とだけ書き、モジュール `functional_ops` の再エクスポート・裸の自由関数・`Var`／`Tensor<f32>` 上の同名メソッドの扱いを明記していない。先例（行 29 の `jacobian`／`hessian`、行 30 の `gradcheck`）に倣うなら反転するのは `Tape` のプローブだけで他は拒否を続けるが、記録に明記がないため承認時に決めてほしい。

| 論点 | 内容 | 選択肢（設計記録にあるもの） | 推奨 |
|---|---|---|---|
| 1 | facade シグネチャの細部 | 複数入力（単一／複数）・`out_dim`（固定 0／引数化）・`Fn` か `FnMut`・`hvp` の `child`（明示引数／内部生成） | 設計記録 §5 の 1 案（単一入力・`FnMut`・`child` を明示）。`out_dim` に推奨は記録にない（実装は dim 0 固定）。上の 2 の実装済みの形を併記する |
| 2 | `vmap` とバッチなし実行の bit 一致を契約にするか | する／しない | しない（統一複合判定。設計記録 §11・§17） |
| 3 | double-VJP の差が統一複合判定を外れた場合の扱い | 新しい判定契約を別途承認する／承認依頼へ戻す | 新しい判定契約は別承認（§9）。#2880 は判定基準 1〜4 で成立し外れた要素は 0 件だった（§19）ため現時点では発動していない。`jvp`／`jacfwd` の Tier 2 への移行（spec 追記）と §10 の 10 の起票は未実施 → 現状（2026-10-08）: spec 追記は fandhe-ai-spec PR #81 でマージ済み（実装リポへの取り込みは #2938）、実装 issue は #2939（親）・#2940（内部実装）で起票・実装済み（`docs/autodiff-functional-transforms-design.md` §24） |
| 4 | f16 等の低精度 forward を対象にするか | 対象にする／初期スコープ外 | 初期スコープ外（設計記録 §11） |
| 5 | `supports_create_graph` の対象 Op を広げるか | 広げる／現行のまま | 推奨なし。記録上は本公開と切り離した別承認事項（§9）で、#2880 でも広げていない |
| 6 | 微分可能な per-sample gradient（`vmap(grad)`）の公開形 | 初期スコープ外のまま（値だけなら呼び出し側の明示ループ）／子テープ上で動くクロージャ型など別の公開形 | 推奨なし。§5 は初期スコープ外とし、別形が要るとだけ書いている |

設計記録に形が書かれていない点（承認時に決めてほしい事項。本節は推奨を作らない）:

- 保留ガードの反転範囲（上の 4）。
- `vjp`／`hvp` の追跡なし入力に対する非対称の扱い（上の 3）。
- `vmap` の `out_dim`・複数入力を将来足すときの拡張方法（新メソッドにするか引数化するか。後者は破壊的変更になりうる）。
- `Tape::vmap` の利用例 doctest が facade 単独で完結する形で書けるか（公開 issue で確認する）。


→ 2026-10-08 の所有者コメント（`https://github.com/Fandhe-AI/fandhe-ai/issues/2499#issuecomment-6067263650`）項 1 で、F1〜F3 の公開と論点 1・2・4 が承認された（論点 5・6 は保留。論点 3 は言及なし）。承認範囲・確定した公開形・保留ガードの反転範囲（上の差 4）は設計記録 `docs/autodiff-functional-transforms-design.md` §23（#2930）に記録した。差 1〜3 も同 §23.5 で解消済み。`out_dim`・複数入力の拡張方法は未決、doctest の可否は #2931 で確かめる。**→ 公開は #2931 で行った（本節末尾の #2931 の適用記録）。** 上の「すべて未承認」の転記文は履歴として残す。

本書 1 節の対象範囲表・`docs/compat-feature-gap.md` の判定列は変更していない。facade 公開面は追加していない。

**Phase 9 公開形（`jvp`／`jacfwd`。承認依頼 #2941・親 #2939・Phase 9 #2928）**: double-VJP 法の `jvp`／`jacfwd` の facade 公開形と論点の承認依頼。行ラベルは F 系を継続して `F4`・`F5` とする（`FunctionalTransformsHoldDoctestGuard` 1 個が覆うという F ブロックの前提がそのまま当てはまるため）。ただし `F1`〜`F3` は承認済み、`F4`・`F5` は依頼時点では**未承認**だった（→ 2026-10-09 の所有者コメント `issuecomment-6079384681` 項 1 で承認、#2956 で公開済み。以下の「未承認」の転記文は履歴として残す）。以下は設計記録 `docs/autodiff-functional-transforms-design.md` §27 の**転記**であり、新しい推奨・tolerance・baseline は作っていない。承認根拠は 2026-10-08 の所有者コメント（`https://github.com/Fandhe-AI/fandhe-ai/issues/2499#issuecomment-6067263650`）項 2 の最終段落のみ（内部実装と公開形の記録まで。facade 公開は記録後に改めて承認依頼）で、項 1 の承認（F1〜F3）は `jvp`／`jacfwd` に及ばない。承認は実装 Agent が代行しない。

| # | 機能（由来） | 公開形（設計記録 §27.1 の転記・未承認） | 非破壊性 | 保留ガード | 公開先 | 決定記録・承認事項の所在 |
|---|---|---|---|---|---|---|
| F4 | `jvp`（#2940） | facade `Tape` の委譲メソッド `Tape::jvp(&self, output: &Var<'_>, input: &Var<'_>, tangent: &Tensor<f32>, child: &Tape) -> Result<Tensor<f32>, AutodiffError>`（結果 shape = `output.shape()`）。`child` は呼び出し側が用意する空の子テープ | 追加のみ（`fandhe-ai =0.10.0` の既存シグネチャ・意味論は不変） | `FunctionalTransformsHoldDoctestGuard` ＋ `crates/facade/tests/api_surface.rs` の否定ガードと宣言インベントリ。#2956 で `Tape` 受け手の `jvp`／`jacfwd` の分だけ部分反転し、正ガード `facade_tape_functional_transforms_are_approved_thin_delegations` が承認形 5 名を固定（インベントリは 11 件） | 公開済み（#2956。承認は issuecomment-6079384681 項 1） | 設計記録 §8・§19・§24・§27 |
| F5 | `jacfwd`（#2940） | `Tape::jacfwd(&self, output: &Var<'_>, input: &Var<'_>, child: &Tape) -> Result<Tensor<f32>, AutodiffError>`（結果 shape = `output.shape ++ input.shape`） | 同上 | 同上（`F4`・`F5` で F ブロックの 1 個を共有） | 公開済み（#2956。承認は issuecomment-6079384681 項 1） | 同上 |

- 判定方式（設計記録 §27.3。未承認）: REQ-2 の統一複合判定。`jacobian` や 1 階 VJP との bit 同一は主張しない。新しい tolerance・baseline は作らない。
- 拡張要否: 依存・`unsafe`・新規 `Op`／`BackendOps`／VJP はなし。内部可視性の `pub` 化と `expect(dead_code)` の撤去のみが公開時に必要（§27.2）。
- 実装と公開形の差: 受け手の違い（内部は第 1 引数に `tape` を取る自由関数、公開は `&self.0` を渡す 1 行委譲）、内部が `pub(crate)` であること、`expect(dead_code)` が付いていること。
- 論点（設計記録 §27.5）: J-a `jacfwd` を公開するか（推奨: 公開）／J-b `child` は明示引数（推奨: 明示）／J-c 名前（推奨: `jvp`／`jacfwd`）／J-d `tangent` は input 側 shape と完全一致（推奨: 実装のまま）／J-e 追跡なし `output` は全ゼロで `hvp` と非対称（推奨: 実装のまま。改めて確認を求める）／J-f 失敗時にも親テープへ 2 ノード残る（推奨: 実装のまま doc に明記）／J-g 内部は `pub fn`（推奨）か `#[doc(hidden)] pub` か。保留継続（推奨なし）: 論点 5、複数入力、`VarF64`・f16、微分可能な `jvp`。
- 設計記録に形が書かれていない点だった #2931 との相互作用は解決済み: `Tape` 向けプローブは専用 trait `__FandheFunctionalTransformsHoldTapeProbe`（`jvp`／`jacfwd` の 2 メソッドのみ）へ分離して維持し、UFCS プローブも維持している（設計記録 §27.4）。

本書 1 節の対象範囲表・`docs/compat-feature-gap.md` の判定列は変更していない。facade 公開面は追加していない。

**適用記録（イシュー #2956・親 #2928。承認: https://github.com/Fandhe-AI/fandhe-ai/issues/2499#issuecomment-6079384681 の項 1）**: 設計記録 `autodiff-functional-transforms-design.md` §27.1 の形で、facade `Tape::jvp`・`Tape::jacfwd` の 2 メソッド（`F4`・`F5`）を内部 `fandhe_ai_autodiff::functional_ops` の `jvp`／`jacfwd`（#2956 で `pub(crate)` から `pub fn` 化）への 1 行委譲として公開した（追加のみ・`fandhe-ai =0.10.0` の既存 API は不変。新規 `Op`・`BackendOps` メソッド・VJP・`AutodiffError` variant・依存・`unsafe`・tolerance・baseline なし）。承認範囲は F4・F5 の公開と論点 J-a〜J-g（推奨どおり）で、保留ガード `FunctionalTransformsHoldDoctestGuard` は `Tape` 受け手の分だけ反転し（プローブ用トレイトと impl・UFCS 2 行を除去）、正ガードは承認形 5 名・宣言インベントリは 11 件になった。維持した保留: モジュール `functional_ops` の再エクスポート・裸の自由関数・`Var`／`Tensor<f32>` 上の同名メソッド・`supports_create_graph` の対象拡張（論点 5）・複数入力・`VarF64`／f16・微分可能な `jvp`。実機は #2942 の既存 `#[ignore]` テストが担当する（CUDA は pass、Metal は未実測）。本書 1 節の対象範囲表・`docs/compat-feature-gap.md` の判定列は変更していない。詳細は設計記録 §28。

**適用記録（経路 2。イシュー #2974・親 #2541・Phase 11-1。承認: `https://github.com/Fandhe-AI/fandhe-ai/issues/2499#issuecomment-6097478475`）**: 参照モデル `Mlp`／`LeNet`（examples の利用者コード。#2201）を `fandhe_ai::models::{Mlp, LeNet}` として公開した（`lib.rs` の `pub mod models;`、`models/mod.rs` の `pub use` 2 文。サブモジュールは非公開・クレートルートへ再エクスポートしない）。公開メソッドは両モデルの `new`・`forward`・`predict`・`sequential`・`sequential_mut` と、`Mlp::with_seed`・`Mlp::dropout`・`LeNet::num_classes`。PyTorch 対応表（`pytorch_param_map`・`MlpParamMap`・`LeNetParamMap`）と `ReferenceModule`／`Trainable` の trait 化は公開せず examples に残す。保存は `sequential()` 経由で既存の `save_model`／`load_model` を使い、新しい保存経路は足していない。`crates/facade/tests/api_surface.rs` に正ガード 6 件（`models_is_public_and_submodules_stay_private`・`facade_exposes_models_items_only_in_approved_shape`・同 `_detects_each_category`・`models_unapproved_paths_are_absent`・`models_items_are_reachable_via_facade_models_path`・`models_usage_doctest_is_present_and_compiled`）を追加した。不変事項: 依存追加なし・tolerance／baseline／ガードレール閾値の変更なし・`docs/spec/` 不変・新規 `unsafe` なし。`ResNet`／`TransformerClassifier` は後続の Phase 11-2 で扱う（#2541 は 2 段完了後に閉じる）。詳細は `docs/reference-models-decision.md` §11.6〜§11.7。
