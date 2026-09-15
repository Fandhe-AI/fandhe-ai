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
| 要素演算（sub／div／pow／sqrt／log／三角関数／比較） | #1592・#1593（#1710 で算術系〈`Var::sub`／`div`／`pow`／`sqrt`〉実装済み。#1711 で `log`／`log2`／`log10`／`sin`／`cos`／`tan`／`abs`／`neg` 実装済み。#1712 で `clamp`／比較演算 6 種〈`Var::clamp`／`gt`／`ge`／`lt`／`le`／`eq`／`ne`〉実装済み。いずれも `ScalarUnaryOp`／`ScalarBinaryOp`〈#1634〉への薄い委譲・facade 到達経路は既存 `Var` 再エクスポート経由・新規 `pub use`／`pub fn` は facade へ追加していない。比較演算の出力は f32 の 0/1 マスク・`Tensor<bool>` 出力は #1613 の対象で本イシュー範囲外。CUDA／Metal 実機での facade parity は未実測のまま Mac／GB10 セッションへ申し送り） |
| softmax／log_softmax | #1594（実装済み。`Var::softmax`／`log_softmax`・`nn::activation::Softmax`／`LogSoftmax`。facade 到達経路は既存 `Var` 再エクスポート経由〈新規 `pub use`／`pub fn` は facade へ追加しない〉） |
| GELU／SiLU 等の活性化 | #1595（#1713 で GELU〈誤差関数版・tanh 近似版〉・Softplus 実装済み。`Var::gelu`／`gelu_tanh`／`softplus`・`nn::activation::Gelu`／`GeluTanh`／`Softplus`。facade 到達経路は既存 `Var` 再エクスポート経由〈新規 `pub use`／`pub fn` は facade へ追加していない〉。#1714 で `Silu`／`Hardswish`／`LeakyRelu`／`Elu` 実装済み。`Var::silu`／`hardswish`／`leaky_relu`／`elu`・`nn::activation::{Silu, Hardswish, LeakyRelu, Elu}`・`compat::Sequential::add_silu`／`add_hardswish`／`add_leaky_relu`／`add_elu`。CUDA／Metal 専用カーネル実装済み〈`LeakyRelu`／`Hardswish` は選択・算術のみで bit 同一想定・`Silu`／`Elu` は超越関数を含むため REQ-2 統一複合判定のみ。Metal `Elu` は `expm1` 相当が MSL に無いため `exp`／`log` から桁落ちなく再構成する自作ヘルパー `fai_expm1_f32` を使う〈単純な `exp(x) - 1.0f` はゼロ近傍・大 `alpha` 入力で桁落ちし REQ-2 統一複合判定を満たさなかったため不採用。PR #1825 codex-review P1 是正〉——`scalar_op_source.rs` モジュール doc「`Elu` の `expm1` 非対応」参照〉・facade `compat::Sequential::add_*` 4 件の新規公開面追加は親 #1595 コメント〈2026-09-12 ユーザー承認〉に基づく §5 経路 2 の適用。CUDA／Metal 実機での facade parity は両実装とも未実測のまま Mac／GB10 セッションへ申し送り） |
| LayerNorm／RMSNorm／BatchNorm | #1596（実装済み。`fandhe_ai_autodiff::Var::rms_norm`／`layer_norm`・`nn::RmsNorm`／`LayerNorm`。facade 到達経路は既存 `Var` 再エクスポート経由——新規 `pub use`／`pub fn` は facade へ追加しない。`docs/norm-ops-design.md`）・#1608 → #1732 で CPU 実装済み（`Var::batch_norm`／`batch_norm_with_batch_stats`／`batch_norm_infer`・`nn::BatchNorm1d`／`BatchNorm2d`〈train／eval・running stats。本クレート初のモード依存層〉。facade 新規公開面なし。#1735 で CUDA 実装済み（`CudaBackendOps::batch_norm_train`／`batch_norm_infer`。facade 新規公開面なし。GB10 実機 parity は未実測のまま申し送り）・Metal は #1736 が残対象。`docs/batch-norm-ops-design.md`） |
| 形状操作（permute／squeeze／expand／cat／stack／split） | #1597（実装済み: permute／squeeze／unsqueeze／expand〈broadcast_to〉／flatten。facade 到達経路は既存 `Var` 再エクスポート経由〈新規 `pub use`／`pub fn` なし〉）・#1598（実装済み: `Var::cat`／`stack`／`narrow`／`split`／`split_with_sizes`／`chunk`。§5 は Tier 1 列挙済み機能につき再適用不要と判断し facade へ新規 `pub use`／`pub fn` を追加していない。narrow は本 issue で `#1599` 側の対象から解消済み） |
| index 系（narrow／where／gather／scatter） | #1599（narrow は #1598 で実装済み・where／masked_fill は #1637 で実装済み〈`Var::where_cond`／`masked_fill`。facade 到達経路は既存 `Var` 再エクスポート経由・新規 `pub use`／`pub fn` は facade へ追加していない〉のため対象外。gather／scatter／scatter_add／index_select は #1638（→ #1776 で Op 定義・CPU 参照実装・VJP 実装済み。`Var::gather`／`index_select`／`scatter`／`scatter_add`。facade 到達経路は既存 `Var` 再エクスポート経由で同様に新規公開面なし。CUDA は #1777・Metal は #1778 でそれぞれ実装済み〈`CudaBackendOps`／`MetalBackendOps` の `gather`／`scatter`。facade 新規公開面なし〉）のため対象外） |
| バッチ行列積 | #1600（#1715 で実装済み: `Var::matmul` の rank≥3 受理・バッチ次元 NumPy 互換ブロードキャスト・`BackendOps::gemm_batched`／`gemm_batched_fp32_strict`〈既定は per-batch `gemm`/`gemm_fp32_strict` への合成〉・`CpuBackendOps::gemm_batched` オーバーライド〈bit 同一〉。§5 は Tier 1 列挙済み機能につき再適用不要と判断し facade へ新規 `pub use`／`pub fn` を追加していない。#1716 で CUDA 実装済み: `CudaBackendOps::gemm_batched`／`gemm_batched_fp32_strict` 専用オーバーライド（デバイス常駐バッチループ経路 `CudaGemm::run_tiled_f32_batched`。A・B を 1 回ずつ H2D・出力を 1 本だけ確保・バッチごとに既存 NN tiled f32 カーネルを起動・D2H 1 回。`gemm_fp32_strict_impl` 単体入口と bit 同一のカーネル選択。TF32 opt-in 時は `Tf32`／`Tf32x3` とも新設 `fandhe_ai_tensor_core::gemm_batched_via_per_batch_gemm` 経由で per-batch `gemm` 合成へフォールバックしカウンタ・fail-closed 挙動は #1042／#1355 のまま不変）。facade 新規公開面なし・GB10 実機実測は未実施のまま申し送り。#1717 で `MetalBackendOps::gemm_batched`／`gemm_batched_fp32_strict` 専用オーバーライド実装済み〈正規化済みオペランドを 1 回ずつ upload・バッチごとの GEMM を encode-only で 1 コマンドバッチへ積み `download` 1 回のみ同期する「バッチループ方式」。`gemm.rs`／shader 自体は無変更。per-batch `gemm`〈`dispatch_auto`／split-K〉とは classic strided カーネル経由のため bit 同一は主張せず REQ-2 統一複合判定。split-K 実行時トグル非依存。facade 新規公開面なし。M4 Max 実機実測は未実施のまま Mac セッションへ申し送り〉） |
| 縮約（mean／min／argmax／var／std／複数軸） | #1601（`amax`／`max` の勾配分配方式はこの issue で設計判断を記録して確定。`04-requirements.md:234`。2 節参照）。#1719 で mean・複数軸・keepdim 部分を実装済み: `Var::mean`〈単一軸／全軸。`tape::Op::Mean`。`BackendOps` は非拡張で `sum` の結果をホスト側で 1 回だけ除算〉・`sum_dims`／`max_dims`／`mean_dims`〈複数軸・`keepdim`。`crate::reduce_dims` が permute／contiguous／reshape で単一軸縮約へ併合。単一軸は `sum(Some(d))` と bit 同一に直接委譲〉。facade 到達経路は既存 `Var` 再エクスポート経由〈新規 `pub use`／`pub fn` なし〉。CUDA 実機 parity は未実測のまま GB10 セッションへ申し送り。#1720 で `min`／`argmax`／`argmin` 実装済み（`Var::min`／`argmax`／`argmin`。facade 到達経路は既存 `Var` 再エクスポート経由・新規 `pub use`／`pub fn` は facade へ追加していない。`min` の VJP は #1718〈amax／amin 勾配分配方式の確定。本追記時点で OPEN〉未確定につき現行 `Var::max` の先勝ち決定的方式と共有するヘルパー `grad::extremum_first_match_vjp`〈旧 `max_vjp` を改称・`max_vjp` 自体は薄いラッパーとして維持〉を適用。CPU は `min`／`argmax`／`argmin` すべて実装済み。CUDA は `min` カーネル実装済み・`argmax`／`argmin` は明示 `Unsupported`（ホストフォールバック）。Metal は 3 演算とも明示 `Unsupported`（ホストフォールバック）。GB10／M4 Max 実機 parity は未実測のまま申し送り）。var／std は #1723 で実装済み: `Var::var`／`std`〈既存 `sum`／`max` と同じ `dim: Option<usize>` シグネチャ＋`correction`。`Op::Var` 専用 Op・`BackendOps::var` 既定 `Unsupported` のホストフォールバック契約。§5「Tier 1 列挙済み機能は再適用不要」の対象。facade 到達経路は既存 `Var` 再エクスポート経由・新規 `pub use`／`pub fn` なし。多軸・`keepdim` は本 issue 対象外のまま〉。`docs/compat-feature-gap.md` §2.5 追補参照。`amax`／`max` の勾配分配方式〈先勝ち決定的 対 均等分配〉は #1718 で確定（既存 `Var::max`／`min`／`max_dims`〈`max(dim)`／`min(dim)` 族の意味論〉は先勝ち決定的を維持・PyTorch `torch.amax`／`amin` 相当の均等分配は別 `Op`／別ヘルパーとして後続 issue で実装する方針。`docs/autodiff-amax-grad-distribution-decision.md` 参照）のため `sum_dims`／`max_dims` とも既存の先勝ち決定的規約を無変更で維持） |
| 乱数生成と RNG 契約 | #1602（#1724 実装済み: グローバル RNG 契約〈manual_seed〉。#1725 実装済み: 乱数テンソル生成本体〈randn／rand／randint〉。#1726 実装済み: 決定的生成系〈arange／linspace／eye／zeros_like／ones_like〉。`fandhe_ai::{manual_seed, randn, rand, randint, arange, linspace, eye, zeros_like, ones_like}`。facade 到達経路は新規 `pub fn`（`RngError`／`CreationError` は 1 行の `pub use`）。`docs/rng-global-contract-design.md`） |
| Dropout | #1603（実装済み。`Var::dropout(p, training)`〈`torch.nn.functional.dropout` 相当。inverted dropout〉・`tape::Op::Dropout`・`nn::Dropout`〈`p`／`training` を保持し `Module::set_training`／`training` を実際にオーバーライドする本クレート内実装で唯一の層。既定 `training=true`〉・`compat::Sequential::add_dropout`（facade 新規 `pub fn` 1 件）。マスク生成は `crate::grad::dropout_mask` が `fandhe_ai_tensor_core::rng::rand`（#1602 のホスト側グローバル RNG 契約）を経由し `BackendOps` を経由しない——forward／backward とも既存必須メソッド `BackendOps::mul` への単一乗算に帰着するため `BackendOps` 非拡張（Embedding／SDPA／einsum と同型の合成方針）で 3 バックエンド bit 同一が構造的に成立する。`training=false` または `p=0.0` は早期リターン（`Op` 非記録・RNG 非消費）。`predict`／`forward_host` とモードの整合は「PyTorch `model(x)` と同じくコンテナの `training` フラグを尊重する」方式で確定（Keras の「常に推論モード」は不採用）——`compat::Sequential::predict` は `training=true` のままだと `Dropout` を適用するため、決定的な推論には呼び出し側が先に `eval()` を呼ぶ必要がある（`nn::Dropout` モジュール doc「train／eval と `predict`／`forward_host` の整合」参照）。CUDA／Metal 実機での facade parity テストは未実測のまま Mac／GB10 セッションへ申し送り） |
| Embedding | #1604（実装済み。`Var::embedding`〈`tape::Op::Embedding`。gather を forward・`scatter_add`〈`ScatterReduce::Add` の決定的集約契約〉を backward に使う合成。`BackendOps` 非拡張〉・`nn::Embedding`／`EmbeddingVars`〈`padding_idx` 対応。forward は当該行を素通し・backward のみゼロ上書き〉。`Module` trait は非実装（id 入力が f32 `Var` 契約と不一致・`compat::Sequential` の学習可能パラメータ収集は `as_linear` フック限定のため、実装すると黙って学習されない罠になる。詳細は `crates/autodiff/src/nn/embedding.rs` モジュール doc）。facade 到達経路は既存 `Var`／`nn` 再エクスポート経由・新規 `pub use`／`pub fn` は facade へ追加していない。CUDA／Metal 実機での facade parity は未実測のまま Mac／GB10 セッションへ申し送り） |
| MultiheadAttention | #1605（sub-issue (a): #1639 で実装済み。`Var::scaled_dot_product_attention`——既存の `matmul`〈rank≥2〉／`transpose`／`mul`／`masked_fill`／`softmax` への分解のみで実装し `Op`／`BackendOps` を新規拡張しない（`crate::einsum` と同型）。causal（top-left aligned）／明示 `attn_mask`〈PyTorch bool 規約〉・`scale` 既定値〈`1/sqrt(E)`〉に対応。facade 到達経路は既存 `Var` 再エクスポート経由・新規 `pub use`／`pub fn` は facade へ追加していない。`dropout_p`／`enable_gqa`／attention weights 返却／f16・bf16 は対象外。CUDA／Metal 実機での facade parity は未実測のまま Mac／GB10 セッションへ申し送り。sub-issue (b) `MultiheadAttention` Module〈in/out projection・head 分割〉は #1640 で実装済み: `nn::MultiheadAttention`／`MultiheadAttentionVars`。q/k/v/out の 4 `nn::Linear` 合成・`Var::matmul`〈rank≥3 バッチ〉／`transpose`／`masked_fill`／`softmax` の既存演算合成のみで新規 `Op`／`BackendOps` メソッドは追加していない。実装着手時点で #1639 が未マージだったため、attention 本体（scale・mask／causal・softmax）は `Var::scaled_dot_product_attention` を呼ばず #1639 と同一の数式・mask 極性〈`true`=attend〉・causal 規約〈top-left aligned `j<=i`〉を private ヘルパーとして複製している（`nn/attention.rs::sdpa_compose`。#1639 マージ後の置き換えは本 issue のスコープ外として未実施のまま残る）。入出力は rank-3・batch_first 固定（`[B,L,E]`／`[B,S,E]`）。facade 新規公開面なし（既存 `Var`／`nn` 再エクスポート経由）。CUDA／Metal 実機での facade parity テストは未実測のまま Mac／GB10 セッションへ申し送り） |
| Conv1d／Conv2d | #1606（設計記録 #1641。`docs/conv-ops-design.md`。#1764 で CPU 実装済み〈`Var::conv2d`。facade 新規公開面なし〉。#1765 で `Var::conv1d`〈`Var::conv2d` の reshape 併合。新規 `Op`／`BackendOps`／カーネルなし・facade 新規公開面なし〉も実装済み。nn 層は #1645・CUDA／Metal 専用 im2col／col2im カーネルは #1643／#1644 が対象。#1766 で CUDA `im2col`／`col2im` 実装済み〈facade 新規公開面なし・GB10 実機 parity は未実測〉。#1767 で CUDA Conv1d 経路を検証済み〈`Var::conv1d` は #1766 の CUDA `im2col`／`col2im` へ reshape 併合のみで自動到達済み・新規カーネルなし・facade 新規公開面なし・1d 形状の parity・bit 同一テストを追加・GB10 実機実測は未実施のまま `docs/perf/logs/cuda-conv1d-1767/` へ申し送り〉。#1768 で Metal `im2col`／`col2im` 実装済み〈facade 新規公開面なし・M4 Max 実機 parity は未実測〉。#1769 で Metal Conv1d 経路を検証済み〈新規カーネルなし・facade 新規公開面なし・1d model／ops／facade bit 同一テスト追加・M4 Max 実機実測は未実施のまま `docs/perf/logs/metal-conv1d-1769/` へ申し送り〉。#1770 で nn 層（`nn::Conv2d`／`nn::Conv1d`・`Module::as_conv2d`／`as_conv1d`〈各 `_mut` 込み〉フック・`compat::Sequential::add_conv2d`／`add_conv1d`〈facade 新規 `pub fn` 2 件のみ〉）を実装済み化。`trainable_parameters`／`bind`／`SequentialVars::forward`／`trainable_vars`／`trainable_grads`／`apply_parameters` を Conv 層対応へ拡張し optimizer 学習経路へ接続・デバイス常駐経路（`init_device_param_store`／`forward_resident`／`predict_resident`）は Conv 層を含む `Sequential` を `BackendError::Unsupported` で fail-closed 拒否（対象外のまま）。新規 `Op`／`BackendOps`／VJP なし・CUDA／Metal 実機での facade parity テストは未実測のまま #1771 へ申し送り）。#1771 で実機ランブック（`docs/perf/logs/conv-realdevice-1771/`）を新設し #1766〜#1770 の 4 イシューの実行手順・判定規則を統合するとともに、nn 層（`compat::Sequential`）の backward・Conv1d・「特化」契約・学習ループ〈record-only〉を対象とする `#[ignore]` テスト 12 件（CUDA／Metal 各 6 件）を `crates/facade/tests/nn_conv_backend_parity.rs` へ追加。facade 新規公開面なし・CUDA／Metal 実機実測は本エージェント実行環境に実機なしのため未実施のまま親 #1645 を受け皿として GB10／Mac セッションへ申し送り）|
| Pooling | #1607（設計記録 #1727。`docs/pooling-ops-design.md`。実装は #1728〈CPU〉・#1729〈CUDA〉・#1730〈Metal〉） |
| 損失（BCE／NLL／Huber／KLDiv） | #1609（#1737 で BCE／BCEWithLogits 実装済み: `Var::bce_loss`／`bce_with_logits_loss`・`nn::loss::BceLoss`／`BceWithLogitsLoss`・3 バックエンド融合カーネル。facade 到達経路は既存 `Var` 再エクスポート経由〈新規 `pub use`／`pub fn` なし〉。CUDA／Metal 実機 parity は未実測のまま Mac／GB10 セッションへ申し送り。#1739 で Huber／SmoothL1 実装済み: `Var::huber_loss`／`smooth_l1_loss`・`nn::loss::HuberLoss`／`SmoothL1Loss`。`BackendOps::huber_loss`／`huber_loss_backward`〈`HuberKind::{Huber, SmoothL1}`。既定 `Unsupported`〉・CPU／CUDA／Metal 3 バックエンド専用融合カーネル〈`MseLoss` と同型の 2 段 reduction・`dTarget = −dPred` 契約〉・VJP は `grad::huber_loss_vjp`〈`Unsupported` のときのみ〉。#1738 で NLLLoss／KLDivLoss 実装済み: `Var::nll_loss`〈`tape::Op::NllLoss`。`targets` は `Op::CrossEntropyLoss` と同型の非追跡 `Tensor<i32>`〉・`Var::kl_div_loss`／`kl_div_loss_with_log_target`〈`tape::Op::KlDivLoss`。`input`／`target` とも追跡対象・`tensor-core::KlDivTarget` で `Probabilities`／`LogProbabilities` を分岐〉・`BackendOps::nll_loss`／`nll_loss_backward`・`kl_div_loss`／`kl_div_loss_backward`〈`MseReduction` を共用・既定 `Unsupported`〉・CPU／CUDA／Metal 3 バックエンド融合カーネル実装済み・`nn::loss::NllLoss`／`KlDivLoss`（薄いラッパー）。`log_softmax(x).nll_loss(t) ≡ cross_entropy_loss(x, t)` の forward／backward 一致を統合テストで確認済み。facade 到達経路はいずれも既存 `Var`／`nn` 再エクスポート経由・新規 `pub use`／`pub fn` は facade へ追加していない。CUDA／Metal 実機での facade parity は未実測のまま Mac／GB10 セッションへ申し送り） |
| optimizer（Adam／RMSprop／Adagrad／LAMB） | #1610（#1742 で Adam〈coupled L2 weight decay〉実装済み。`fandhe_ai_autodiff::nn::optim::adam` モジュール〈`AdamW` を鏡写しにした別実装・decay を勾配へ加算する分岐のみが差分〉。facade は `pub use fandhe_ai_autodiff::nn::optim::{Adam, AdamConfig};` の 1 行追加のみ〈`optim.rs`〉。`weight_decay==0` で `AdamW` と bit 完全一致・`weight_decay>0` は `torch.optim.Adam` の `_single_tensor_adam` 定義に基づく恒等式で検証（`crates/autodiff/tests/nn_optim_adam.rs`）。新規 `Op`／`BackendOps`／`Var` メソッド／VJP は追加していない・`DeviceParamStore` 未結線（`Sgd` 専用のまま）。RMSprop／Adagrad は #1743 で実装済み: `RmsProp`／`RmsPropConfig`・`Adagrad`／`AdagradConfig`〈`crates/autodiff/src/nn/optim/{rmsprop,adagrad}.rs`。`AdamW`〈#194〉を鏡写しにした別実装〉。`Tape`／`Var`／`BackendOps` に一切依存しない値型・純関数のため新規 `Op`／`BackendOps` メソッド／VJP なし。正しさは実 PyTorch 2.14.0+cpu 実行値 fixture との統一複合判定〈`.claude/rules/coding-rust.md` 既存 tolerance〉・閉形式（t=1）一致・決定性（bit 完全一致）で検証（`tests/nn_optim_{rmsprop,adagrad}.rs`）。facade は `fandhe_ai::optim::{RmsProp, RmsPropConfig, Adagrad, AdagradConfig}` の素の再エクスポートのみ（`docs/facade-optimizer-promotion-decision.md` §4 案 A）。`crate::DeviceParamStore` は非対応（対応する `BackendOps` メソッド未追加）。#1744 で LAMB（layer-wise trust ratio。You et al., 2019）実装済み。`fandhe_ai_autodiff::nn::optim::lamb` モジュール（`AdamW` と同一の moment 更新演算列を使う独立実装。weight decay は paper 定義どおり更新方向 `u` へ coupled で織り込み、`AdamW` の decoupled 乗算減衰とは構造が異なる）。facade は `pub use fandhe_ai_autodiff::nn::optim::{Lamb, LambConfig};` の 1 行追加のみ。独立 f64 参照実装（paper Algorithm 2 そのまま）との統一複合判定突合・解析的恒等式 3 件（zero-grad の wd 非依存性・`wd=0` での `AdamW` 換算・2 の冪スケール不変性は bit 完全一致）で検証（`crates/autodiff/tests/nn_optim_lamb.rs`）。新規 `Op`／`BackendOps`／`Var` メソッド／VJP は追加していない・`DeviceParamStore` 未結線（パラメータテンソルごとの L2 norm reduction カーネルが未実装のため） |
| scheduler（Cosine／Exponential／Plateau／OneCycle） | #1611（#1745 で式ベース 3 種を実装済み: `CosineAnnealingLr`／`ExponentialLr`／`LinearWarmupLr`。いずれも `LrScheduler::lr_at(step) -> f32` のみを持つ stateless 純関数で `Op`／`BackendOps`／`Var` を新規拡張しない。`CosineAnnealingLr` は PyTorch `CosineAnnealingLR._get_closed_form_lr` 準拠の周期的閉形式〈`step>t_max` で clamp しない〉、`LinearWarmupLr` は PyTorch に同名クラスがないため `LinearLR` の `end_factor=1.0` 固定形として定義。facade は `crates/facade/src/optim.rs` への `pub use` 1 行追加のみ。Plateau 分は #1746 で実装済み: `ReduceLrOnPlateau`／`ReduceLrOnPlateauConfig`／`PlateauMode`／`ThresholdMode`〈`fandhe_ai_autodiff::nn::optim::reduce_lr_on_plateau`。PyTorch `torch.optim.lr_scheduler.ReduceLROnPlateau` 準拠〉。既存 `ConstantLr`／`StepLr`〈stateless 純関数〉とは異なり検証指標に応じて内部状態〈patience／best／cooldown〉を進める唯一の例外であり、`LrScheduler::lr_at` は状態を進めず現在値を返すだけで、状態を進める入口は `ReduceLrOnPlateau::step(metric)` に限定〈`nn::optim::reduce_lr_on_plateau` モジュール doc〉。`Tape`／`Var`／`BackendOps` に一切依存しない値型・純関数のため新規 `Op`／`BackendOps`／`Var` メソッド／VJP は追加していない。facade は `fandhe_ai::optim::{ReduceLrOnPlateau, ReduceLrOnPlateauConfig, PlateauMode, ThresholdMode}` の素の再エクスポートのみ〈`optim.rs`〉。#1747 で `OneCycleLr`／`OneCycleLrConfig`／`OneCycleAnneal`〈PyTorch `OneCycleLR` 相当〉を実装済み化: `new` 構築時にフェーズ境界（2 フェーズ／3 フェーズ）を事前計算して保持することで `lr_at` 自体は参照のみの stateless 純関数として実装〈内部可変状態を持たない〉。momentum cycling（`cycle_momentum` 等）は対象外。`step >= total_steps` は PyTorch の `ValueError` と異なり `total_steps - 1` へ clamp する（`lr_at` が `Result` を返せない契約のため）。`Op`／`BackendOps`／`Var` 新規拡張なし・facade は `pub use` 1 行追加のみ。これにより状態保持型（Plateau／OneCycle）も式ベース型（Cosine／Exponential／LinearWarmup）も出揃い、本行の対象外事項はなくなった） |
| autograd 制御（no_grad／detach／retain_graph） | #1612（#1748 で no_grad／detach 実装済み: `Tape::var_no_grad`〈追跡なし葉。`requires_grad=false` の `Op::Leaf`。facade は `Tape::var_no_grad` ラッパーを 1 メソッド追加〉・`Var::detach`〈既存の `requires_grad=false` 葉ノード機構を再利用する設計・専用 `Op` variant は追加していない・facade は既存 `Var` 再エクスポート経由で新規公開面なし〉。`TapeNode::requires_grad`〈bool〉を前方伝播〈`Op::for_each_input` で入力の論理和〉し、`Tape::backward` は起点 loss が `requires_grad=false` なら `Err(AutodiffError::Backward)`・逆走査中は `requires_grad=false` ノードへの寄与を `accumulate` へ渡さず破棄する。`Gradients::get` は対象ノードが `requires_grad=false` なら新設 `AutodiffError::GradientTrackingDisabled`〈`#[non_exhaustive]` への非破壊 variant 追加〉を返し「loss から未到達」の `Ok(None)` と型で区別する。算術を伴わない機構のため CPU 本番 ops と naive 参照実装の勾配が bit 完全一致することをテストで確認済み。retain_graph（複数回 backward の勾配蓄積契約）は #1749 で実装済み（retain_graph 自体は追加 API なしで常時成立する契約〈`Tape` は `reset`／drop までグラフを保持し続けるため複数回 `backward` が無条件で成功する〉と確定・`Tape::backward_accumulate`〈facade 到達経路も追加。`grad::vjp_elementwise_add` への委譲のみで新規 `Op`／`BackendOps` なし〉を新設。設計は `docs/autodiff-retain-graph-accumulate-decision.md`）。設計は `docs/autodiff-nograd-leaf-dinput-skip-decision.md` §5「案 B」） |
| cast | #1613（#1750 で dtype 変換基盤と CPU cast を実装済み: `CastDType`／`CastElement`〈sealed trait・`f32`／`f64`／`i32`／`i64`／`bool` の 5 型〉・`BackendOps::cast_ops` capability accessor・`CastOps`〈8 方向・既定 `Unsupported`〉・ホスト参照実装〈`tensor_core::cast::{cast_from_f32, cast_to_f32}`〉・CPU 実装〈`backend-cpu::cast`。ホスト参照実装への委譲〉・`Var::cast`／`Var::to_f32`／`Tape::var_from`。出力が非 f32 の cast は非微分・detached〈`Var::argmax`／`unique` と同型〉・facade は `CastDType`／`CastElement` の再エクスポートのみ〈`CastOps` は非公開のまま〉。設計・数値契約は `docs/tensor-core-cast-design.md`。#1751 で CUDA〈8 方向〉・Metal〈f64 2 方向を除く 6 方向。MSL `double` 非対応〉のネイティブカーネルを実装済み・facade 新規公開面なし・CUDA／Metal 実機実測は未実施のまま Mac／GB10 セッションへ申し送り） |
| デバイス転送と列挙 | #1614（実装済み: `fandhe_ai::available_devices()`〈3 バックエンドの `DeviceProvider` を束ねた列挙入口。`Device::Cpu` → `Device::Cuda(0..n)` 昇順 → `Device::Metal`〈macOS のみ〉の決定的順序。`DeviceProvider`／`DeviceInfo`／`enumerate_all` は再エクスポートせず `Device` 識別子のみを返す〉・`Var::device`／`Var::to`〈同一デバイスなら恒等・不一致なら新設 `AutodiffError::DeviceMismatch { requested, actual }`〉・`Var::to_tape`〈別 `Tape` への値転送。同一 tape は恒等・それ以外は `materialize_fallible` 経由で実体化した値〈bit 完全一致・算術なし〉を新しい葉として登録し `requires_grad` を引き継ぐ・勾配はテープをまたがない・checkpoint 解放済み poison ノードは `Err` で fail-closed〉・facade `Tape::device`／`Tape::transfer`〈`Var::to_tape` の唯一の facade 到達経路〉。`Tensor` 側は独立の転送 API を設けず `tape_for(device)?.var(&t)` と等価という設計上の非対応として確定。新規 `Op`／`BackendOps`／VJP は追加していない。設計・実装記録は `docs/facade-device-transfer-enumeration-design.md`。CUDA／Metal 実機での facade parity テストは未実測のまま Mac／GB10 セッションへ申し送り） |
| Dataset／DataLoader | #1615（実装済み: `data::Dataset`／`data::TensorDataset<T>`〈map-style。異種 dtype／複数列は 2/3 要素タプル impl で表現し同一シャッフル順を保証〉・`data::DataLoader`／`data::DataLoaderConfig`〈batch_size／shuffle／drop_last〉・`data::Batches`〈`ExactSizeIterator`〉・`data::DataError`。`crates/tensor-core/src/data.rs`（`rng`／`creation` と同じホスト側完結レイヤー。`Op`／`BackendOps`／`Var`／VJP を一切経由しない）。シャッフルは Fisher–Yates＋`rng::with_global_rng` の rejection sampling（`manual_seed` 契約下で決定的）。facade は `fandhe_ai::data::{Batches, DataError, DataLoader, DataLoaderConfig, Dataset, TensorDataset}` の純再エクスポートのみ（`crates/facade/src/data.rs`）。設計・受入基準テンプレート読み替えは `docs/dataset-dataloader-design.md`） |
| state_dict／safetensors | #1616（#1752 で state_dict 部分を実装済み: `nn::Module` trait に `set_parameter`〈defaulted・既定 `Err`。`Linear`／`RmsNorm`／`LayerNorm`／`MultiheadAttention`／`Rnn`／`Lstm`／`Gru`／`ModuleList`／`Sequential` で実装〉・`state_dict`／`load_state_dict`〈defaulted。`HashMap<String, Tensor<f32>>`。strict・two-pass アトミック〈パス 1 でキー集合完全一致・shape 完全一致を検証してから、全通過後のパス 2 で書き戻す。`compat::Sequential::apply_parameters` の #294／#426 と同型の不変条件〉〉を追加。`compat::Sequential::state_dict`／`load_state_dict` へ 1 行委譲する facade 新規公開面 2 件〈`docs/compat-api-scope.md` §5 経路 2。親 #1616 の 2026-09-12 ユーザー承認済み〉。新規 `Op`／`BackendOps`／VJP はいずれも該当なし〈数値経路を変更しない機構のため bit 完全一致契約。CUDA／Metal 実機 parity は数値経路非依存のため対象外〉。#1754 で `Tensor` の `Debug`／`Display`〈打ち切り付きの値プレビュー。`docs/public-api-design.md` §7 の `Tape: Debug` 公開契約越しの DoS 耐性を含む。軸ごとの打ち切りのみでは高階小軸長形状〈例 `shape=[2;20]`〉を打ち切れない穴・極端な rank でのスタックオーバーフローをコードレビュー指摘で発見し、総出力要素数のグローバル予算・rank 上限ガードを追加是正済み〈`tensor-core::tensor_fmt` モジュール doc 参照〉〉実装済み〈`tensor-core::tensor_fmt`。facade 新規公開面なし〉。safetensors save／load の facade 再公開自体は `onnx-interop` の publish 承認前提〈#1652 §6.2 と同一〉が未充足のため段階 0 のまま blocked。案比較・再開条件は `docs/facade-safetensors-exposure-decision.md`） |
| Module の train／eval | #1617（#1758 で `nn::Module` trait 契約〈`set_training`／`training`。既定 no-op／`true`。無状態モジュールはオーバーライドせず、モードの正はコンテナ〈`compat::Sequential`〉が保持するフラグとする契約〉と `named_parameters`〈`Vec<(String, &Tensor<f32>)>`。struct フィールド名／accessor 名ベースの命名契約・`Linear`／`RmsNorm`／`LayerNorm`／`MultiheadAttention`／`Rnn`／`Lstm`／`Gru` で実装〉を実装済み。`compat::Sequential` に `set_training`／`train`／`eval`／`training`／`named_parameters`〈index 接頭辞契約。`trainable_parameters()` と同一順序〉を追加（facade 新規 `pub fn` 5 件）。新規 `Op`／`BackendOps`／VJP／GPU カーネルはいずれも該当なし〈数値経路 bit 完全一致。CUDA／Metal 実機 parity は数値経路非依存のため対象外〉。`predict`／`forward_host` とモード〈Dropout 等の `training=False` 意味論〉の整合は #1603 で確定済み（コンテナの `training` フラグを尊重する方式。§1.2「Dropout」行参照）。コンテナ再構成は #1759 で実装済み: `fandhe_ai_autodiff::nn::container::{ModuleList, Sequential}`（PyTorch `nn.ModuleList`／`nn.Sequential` 相当。`Module` trait を非公開のため facade からは再エクスポートしない）を新設し、`compat::Sequential` の層保持・Linear→ReLU 融合先読み走査（forward）・`set_training`／`training`／`named_parameters` の本体を移設。`compat::Sequential` は `inner: nn::Sequential` を持つ薄いラッパーへ再構成（`forward` は `self.inner.forward(&tape.0, input)` へ 1 行委譲・`set_training`／`training`／`named_parameters` も `self.inner` へ委譲）。公開シグネチャ・数値挙動は不変（既存テスト 30 件 bit 完全一致確認済み）・facade 新規公開面なし。ネストしたコンテナ内 `Linear` は compat の学習契約〈`bind`／`trainable_parameters`／`apply_parameters`・デバイス常駐経路〉に到達しない制限が残るが、facade は `ModuleList`／`nn::Sequential` を構築する経路自体を公開していないため到達不能（`container.rs` モジュール doc「ネストの限界」参照）） |
| Keras 風 `Sequential` の層追加と `compile()`／`fit()`／`evaluate()`／callbacks の最小版 | #1618（#1761 で `compile()`／`fit()`／`evaluate()` 最小版実装済み。`compat::{Loss, Optimizer, FitConfig, History, FitTarget}`・`Sequential::{compile, is_compiled, fit, evaluate}`。既存公開 API〈`Sequential::bind`／`fandhe_ai::optim`／`fandhe_ai::data::DataLoader`／`Var::mse_loss`／`cross_entropy_loss`〉の合成のみで新規 `Op`／`BackendOps`／VJP なし・CPU `tape()` 固定。正しさは手動学習ループとのパラメータ・loss 系列 bit 完全一致で検証（統合テスト `crates/facade/tests/compat_sequential_fit.rs`）。callbacks／`validation_data`／metrics／LR スケジューラ連携・`DataLoader` 直接入力は #1763 へ引き継ぎ） |

### 1.3 Tier 2（長尾。対象範囲・未実装）

REQ-9 2026-09-12 追記（`04-requirements.md:232`）の列挙を、実装リポ
Phase 3（親 #1573）の各 issue へ対応付ける。

| Tier 2 機能 | 実装 issue |
|---|---|
| RNN／LSTM／GRU | #1619 |
| einsum | #1620（実装済み。`Var::einsum`。既存の `matmul`／`sum`／`permute`／`reshape`／`mul` への分解のみで新規カーネルは追加していないため `BackendOps` は非拡張。facade 到達経路は既存 `Var` 再エクスポート経由〈新規 `pub use`／`pub fn` なし〉。batch 添字を伴う縮約〈例 `"bij,bjk->bik"`〉は rank≥3 `matmul`〈#1600〉未実装のため対象外——ガード撤去のみでは対応できず #1600 実装後に分解経路の再設計が必要。`docs/compat-feature-gap.md` §2.6 追補参照） |
| 線形代数（inv／solve／det／qr／cholesky／svd） | #1621（実装済み。`Var::inv`／`solve`／`det`／`cholesky`／`qr`／`svd`／`matrix_norm`。rank-2 限定・CPU 実装先行・GPU は `Unsupported` フォールバック。`docs/autodiff-linalg-design.md`） |
| 高階微分 | #1622（設計記録。`docs/autodiff-higher-order-grad-decision.md`） |
| custom autograd Function | #1623（設計記録。`docs/autodiff-custom-function-decision.md`） |
| activation checkpointing | #1624（実装済み。`Var::checkpoint_from`／内部クレート `Tape::checkpoint`。対象 Op は `MatMul`／`Sigmoid`／`Sum`／`Max`・view 系〈`Reshape`／`Transpose`〉限定〈`Op::is_checkpoint_eligible()`〉。facade `Tape` passthrough は承認未取得のため未追加——facade からは既存の `Var` 再エクスポート経由〈`Var::checkpoint_from`〉で到達可能。`docs/autodiff-checkpoint-design.md`） |
| AMP | #1625（#1721 でコア関数を実装・#1722 で facade 公開済み。`fandhe_ai::optim::{GradScaler, GradScalerConfig, UnscaleResult, scale_loss, scale_grads, unscale_grads, has_non_finite}` の純再エクスポート〈案 A〉。ホスト `Tensor<f32>` 勾配限定・真の混合精度〈f16 forward／f32 master weight〉は `docs/backend-dtype-dispatch-design.md` §8 のとおり対象外・デバイス常駐更新経路〈`DeviceParamStore`〉への unscale／非有限検出は未結線。承認記録は #1625 コメント〈2026-09-12〉） |
| f64／f16／bf16 演算 | #1626 |
| **量子化** | #1627（除外事項「分散学習・量子化の網羅対応」〈Won't・条件付き〉に従属。実装着手は同除外事項の格上げ条件充足と Phase 4 要件見直しでの新 REQ 追加のユーザー承認まで不可。5 節参照） |
| **複数 GPU／DDP** | #1628（同上に従属。設計判断の記録〈docs のみ〉に留め、実装・通信層の依存追加は行わない。5 節参照） |
| ONNX import 公開／export | #1629（#1652 で import 側の設計判断を記録・#1775 で export 側の設計判断を記録。案 B〈薄いラッパー型〉を方針として推奨するが、facade は crates.io 公開クレートのため公開には `onnx-interop` 自体の crates.io 公開という別個のユーザー承認が必要——2026-09-12 の facade 公開面拡張の承認範囲には含まれない。現状は import・export とも非公開のまま段階 0。`docs/facade-onnx-import-exposure-decision.md`・`docs/facade-onnx-export-exposure-decision.md`） |
| topk／sort／cumsum | #1733 で sort／argsort／topk 実装済み（`Var::sort`／`argsort`／`topk`。CPU 参照実装〈`backend-cpu::sort_topk`〉・scatter ベース VJP〈`Op::Sort`／`Op::Topk`〉・facade 新規公開面なし〈既存 `Var` 再エクスポート経由〉）。#1741 で CUDA／Metal カーネル実装済み（64bit 合成キー〈`hi`＝正規化済み値キー・`lo`＝ライン内元添字〉によるビットニックソート方式。GPU 非安定ソートでも `values`／`index` が CPU 参照実装と bit 完全一致する契約〈`sort_model.rs` の鍵設計。CUDA・Metal で意図的複製・ホストモデルによる Linux 実行可能なアルゴリズム検証を実施済み〉。facade 新規公開面なし。CUDA・Metal とも実機実測は未実施のまま GB10／Mac セッションへ申し送り）。#1731 で cumsum／cumprod 実装済み（`Var::cumsum`／`cumprod`・`Op::Cumsum`／`Op::Cumprod`。CPU 参照実装先行・VJP はホスト側のみ〈厳密形・除算なし〉・facade 到達経路は既存 `Var` 再エクスポート経由〈新規 `pub use`／`pub fn` なし〉）。#1740 で CUDA／Metal 専用カーネル実装済み（`backend-cuda::scan::CudaScan`・`backend-metal::scan::MetalScan`。lane ごとの `f64`／binary64 ソフトウェアエミュレーションアキュムレータ逐次計算で CPU 参照実装と bit 完全一致契約〈Metal は `crate::soft_f64` 方式の逐語移植〉。サイズ上限・エラー写像: 要素数積の `usize` オーバーフローは両バックエンドとも `ShapeMismatch` で拒否〈フォールバックなし〉、カーネル引数の上限超過〈CUDA `i32`: `ScanSizeLimitExceeded`／Metal `u32`: `scan_model::plan_scan` の `SizeLimitExceeded`〉のみ `Unsupported` へ写像しホストフォールバックへ委譲、内部契約違反・起動失敗は覆い隠さない。GB10／M4 Max 実機は本エージェント実行環境に到達不能のため未実測のまま申し送り）。unique（`torch.unique` の values のみ）は #1734 で実装済み（`Var::unique`。非微分演算・detached な `Tensor<f32>` を返し `Op` を tape に記録しない。3 バックエンド〈CPU 参照実装・CUDA／Metal ビットニックソート方式〉とも bit 完全一致契約。facade 新規公開面なし〈既存 `Var` 再エクスポート経由〉。`return_inverse`／`return_counts`／`dim` 指定・`sorted=false`・`unique_consecutive` は対象外。`docs/unique-facade-exposure-decision.md`）。`docs/compat-feature-gap.md` §2.2 追補参照 |
| `nn.functional` の残り（pad／interpolate／one_hot 等） | #1631（#1755 で one_hot 実装済み。`Var::one_hot`〈**非微分演算**。VJP は明示ゼロ〉・`Op::OneHot`・CPU／CUDA／Metal 3 バックエンドカーネル・facade 新規公開面なし〈既存 `Var` 再エクスポート経由〉。#1756 で pad 実装済み（`Var::pad`・`BackendOps::pad`〈既定 `Unsupported`〉・narrow 基盤流用 VJP〈zero-copy view 連鎖〉・CPU／CUDA／Metal 専用カーネル実装済み〈算術を含まない純粋なコピー演算のため 3 バックエンド bit 完全一致契約〉・facade 新規公開面なし〈既存 `Var` 再エクスポート経由〉。CUDA／Metal 実機実測は未実施のまま申し送り）。イシュー #1757 で interpolate（nearest）実装済み〈`Var::interpolate`・`Op::Interpolate`・`InterpolateMode`〈`tensor-core::backend_ops`。`#[non_exhaustive]`〉。3 バックエンド専用カーネル・scatter_add ベース VJP・facade へ `InterpolateMode` を再エクスポート〈承認記録は #1631 コメント 2026-09-12〉。CUDA／Metal 実機は未実測のまま申し送り〉。#1753 で clip_grad_value 実装済み（`fandhe_ai_autodiff::nn::optim::clip::clip_grad_value`。`clip_grad_norm` と同型の純関数・`Gradients`／`Var` 非依存・`Op`／`BackendOps`／VJP 非拡張。facade 到達経路は `crates/facade/src/optim.rs` の `pub use`〈`fandhe_ai::optim::clip_grad_value`〉）。イシュー #1762 で interpolate（bilinear）実装済み〈`InterpolateMode::Bilinear { align_corners: bool }`〈`#[non_exhaustive]` variant 追加〉・座標・重みの単一情報源 `tensor-core::interpolate`〈`bilinear_scale`／`bilinear_src_coord`／`bilinear_blend`〉・3 バックエンド専用カーネル（CUDA `fmaf`／Metal `fma`／CPU `f32::mul_add` による明示 FMA。受入契約は REQ-2 統一複合判定であり `Nearest` の bit 完全一致契約とは異なる）・VJP は 4 近傍重み付き scatter_add への委譲〈`grad::bilinear_src_index_and_weight_map`〉。facade 新規公開面なし（`InterpolateMode` 再エクスポートのみ）。CUDA／Metal 実機は未実測のまま申し送り〉。`docs/compat-feature-gap.md` 追補参照） |

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
  参照。`04-requirements.md:234`）
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

**#1628 の設計記録は `docs/facade-multi-gpu-ddp-decision.md` として完了した。**

**#1633（sparse／complex テンソルの非対応の明文化）の設計記録は `docs/tensor-core-sparse-complex-decision.md` として完了した。** 量子化／DDP と異なり除外事項「分散学習・量子化の網羅対応」には従属しない（sparse／complex は REQ-9 の「引き続き対象外」列挙にのみ現れ、格上げ条件表を持つ Won't 項目ではない）。コード変更なし。再開には本節の範囲拡張手続き（経路 1 または経路 2）を要する（同 doc §3・§9）。

**#1652（ONNX import 公開可否）の設計記録は `docs/facade-onnx-import-exposure-decision.md` として完了した。** DDP／量子化と異なり本項目は正本 spec の除外事項（上記）に従属しない——facade へ公開する方針自体は案 B（薄いラッパー型）として推奨されるが、facade は crates.io 公開クレートであり非公開クレートへの通常依存を持てないため、「facade から公開する」は `onnx-interop` 自体を crates.io へ公開することと構造的に等価になる。この publish 承認（命名確定・`RELEASE_CRATES` 変更を含む）は 2026-09-12 の facade 公開面拡張の承認範囲には含まれない別個の事項であり、承認が得られるまでは非公開のまま段階 0（現状維持）とする。#1775（ONNX export の facade 公開）・#1754（safetensors save／load の facade 再公開）はいずれも同じ publish 前提を共有するため blocked のまま close しない（同 doc §6.2）。

**適用記録（経路 2。イシュー #1758）**: `compat::Sequential` への `set_training`／`train`／`eval`／`training`／`named_parameters` の追加は、親 #1617 のユーザー承認コメント（2026-09-12「今承認するので進めてください」）に基づく §5 経路 2 の適用（`add_silu` 等〈#1714〉と同型）。facade 新規 `pub fn` は上記 5 件のみ（目視確認）。`crates/facade/tests/api_surface.rs` は `pub use` での `Tape`／`BackendOps` 再エクスポート禁止・`pub fn` が `BackendOps` を引数として直接取らないことを機械検査するが、本 5 件を名指しでは検証していない（戻り値経由の露出は対象外）。

**#1775（ONNX export の facade 公開）の設計記録は `docs/facade-onnx-export-exposure-decision.md` として完了した。** #1652 と同じ publish 前提（上記段落）を共有するため段階 0・blocked のまま close しない。本 issue では facade（`crates/facade/src/**`・`Cargo.toml`）へのコード追加は一切行わず、代わりに「facade（crates.io 公開クレート）が非公開クレート `onnx-interop` へ通常依存しない」ことを固定する負の guard テスト（`crates/facade/tests/api_surface.rs::facade_does_not_depend_on_unpublished_onnx_interop`／`facade_sources_do_not_reference_onnx_interop`）を追加した——CI が `cargo publish --dry-run` を実行しないため、公開クレートの `Cargo.toml` へ非公開クレートへの path 依存を誤って追加しても通常の `cargo build`／`cargo test` は成功してしまい、次回リリース（`release-all.yml`）まで壊れに気づけないという盲点を機械的に前倒しする。`onnx-interop` の crates.io 公開承認取得後は、本テストを削除ではなく「承認済み依存形状の検査」へ差し替える（同 doc §6）。

**#1754（safetensors save／load の facade 再公開・`Tensor` の Debug／Display）は 2 部構成として完了した。** (A)「`Tensor<T>` の `Debug`／`Display`」は本 issue でコード実装済み（`crates/tensor-core/src/tensor_fmt.rs`。打ち切り付きの値プレビュー・`Tape: Debug` の既存公開契約〈`docs/public-api-design.md` §7〉越しの DoS 耐性〈`.claude/rules/security.md` A04〉を含む。facade 新規公開面なし・既存 `pub use fandhe_ai_tensor_core::Tensor` 経由でそのまま到達）。**コードレビューで、初版の軸ごとの打ち切り（`len > 2 * FMT_EDGE_ITEMS` の軸のみ省略）だけでは、全軸長が閾値以下の高階テンソル（例 `shape=[2;20]`。numel は 100 万超）で 1 軸も打ち切られず出力が無制限に増大する穴、および `shape` の rank 自体に上限がないため再帰深さが rank に比例しスタックオーバーフローしうる穴が指摘され、総出力要素数のグローバル予算（`FMT_MAX_ELEMS`）と rank 上限ガード（`FMT_MAX_RENDER_RANK`）を追加して是正済み**（`tensor_fmt` モジュール doc・追加テスト参照）。(B)「safetensors 再公開」は #1652・#1775 と同じ publish 前提を共有するため段階 0・blocked のまま close しない。設計記録は `docs/facade-safetensors-exposure-decision.md` として完了した（案比較・推奨〈facade 直接 `safetensors` 依存の独立実装。承認未取得〉・再開条件を記録）。

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

## 6. 出典一覧

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
