# compat API 層の対象範囲（TASK-9.2b）

イシュー #96（親: #94 = TASK-9.2、ルート: Phase 4 #71）の成果物。
REQ-9「Python 慣習寄りの互換 API 層」（`docs/spec/04-requirements.md:200-209`）の
受け入れ基準のうち「互換 API 層の対象範囲は本フェーズでは初期方針（活性化関数・
基本レイヤーの薄いラッパー）に限定し、全方位の API 網羅は対象外とすること」
（`04-requirements.md:206`）を明文化する。TASK-9.2 の成果物として
`docs/compat-api-scope.md` が明示されている（`docs/spec/05-tasks.md:307-311`）。

`docs/public-api-design.md` は「compat 層は自作コアの素の公開 API とは別
レイヤーであること」という境界のみを 1 章で明記し（同ファイル 6・13・556 行目）、
詳細範囲は REQ-9 系後続タスクへ委譲している。本文書はその受け皿である。

## 0. サポート境界（TASK-9.4・REQ-9 の 2026-08-08 追記・イシュー #411・
REQ-9 の 2026-08-29 追記・イシュー #986）

`facade` クレートが**唯一のサポートされる公開 API 面**である。`tensor-core`／
`autodiff`／`backend-cpu`／`backend-cuda`／`backend-metal` は内部クレートで
あり、これらを `facade` を経由せず直接利用することはサポート対象外とする
（出典: `docs/spec/04-requirements.md:209-210` の 2026-08-08 追記・
`docs/spec/04-requirements.md:211-213` の 2026-08-29 追記・
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
- **確定入口は次の 4 つ**である（正本 spec 側の記述〈REQ-9 の 2026-08-08
  追記・2026-08-29 追記〉と整合済み）:
  1. `fandhe_ai::tape()`／`fandhe_ai::tape_for(Device)`（composition root。
     `crates/facade/src/lib.rs`）
  2. `fandhe_ai::compat::{array, Sequential}`（本文書が定める compat 公開面。
     1〜2 節。1 節の対象範囲は本追記で変更しない）
  3. **`fandhe_ai::optim`**（`crates/facade/src/optim.rs`。イシュー #961・
     親 #960・PR #972）: `Sgd`／`SgdConfig`／`AdamW`／`AdamWConfig`／
     `clip_grad_norm`／`global_grad_norm`／`ClipGradResult`／`LrScheduler`／
     `ConstantLr`／`StepLr` の素の再エクスポート（`docs/facade-optimizer-
     promotion-decision.md` §4 案 A）。値型・純関数のみで構成され
     `BackendOps` 系の型・注入経路を含まないため REQ-12（利用者向け融合
     制御 API を設けない・任意 `BackendOps` 実装を注入できる公開 API を
     設けない）と矛盾しない（出典: `docs/spec/04-requirements.md:212`）
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
     04-requirements.md:213`・設計 `docs/device-resident-update-design.md`
     〈#951・#1022 追補〉・#955〈parity テスト・ベンチ非後退確認〉）
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

REQ-9 の受け入れ基準（`docs/spec/04-requirements.md:203-206`）・
TASK-9.1／TASK-9.2（`docs/spec/05-tasks.md:299-311`）に基づき、以下に限定する。

- **`compat::array()`**: numpy 慣習のテンソル生成関数（`np.array()` 相当）。
  自作テンソル（`tensor-core`）の上に構築する（TASK-9.2a・#95）
- **`compat::Sequential`**: Keras 慣習のレイヤー積み上げビルダー
  （`.add_linear()`／`.add_relu()` 等のメソッドチェーン、
  `docs/spec/04-requirements.md:205`）。自作 NN モジュール（`fandhe_ai_autodiff::nn`）の
  上に構築する（TASK-9.2a・#95）
- **基本レイヤー・基本活性化関数の薄いラッパー**（TASK-9.1）:
  - Linear 層（TASK-9.1a・#91）
  - ReLU・Sigmoid・Tanh の 3 活性化関数（TASK-9.1b・#92 で実装済み。
    `crates/autodiff/src/nn/activation.rs`）

対象範囲はこの 3 種（配列生成関数・Sequential ビルダー・基本レイヤー／活性化の
薄いラッパー）に限定する。

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
  高水準学習ループ API は 2 節のとおり引き続き対象外

## 2. 対象外（out of scope）

- 全方位の Python API 網羅を明示的に対象外とする（`04-requirements.md:206`・
  `05-tasks.md:309`）。具体的には以下を含むが、これらに限らない。
  - numpy の ufunc 群・ファンシーインデックス・ブロードキャスト以外の
    高度な配列操作の網羅
  - Keras の全レイヤー種別（Conv 系・RNN 系・Embedding 等）・callbacks・
    `fit()`／`compile()` 等の学習ループ API の網羅
  - Softmax（損失関数 CrossEntropy・#189 と密結合のため対象外。
    `crates/autodiff/src/nn/activation.rs` 冒頭コメント）・GELU 等の
    追加活性化関数（必要になった時点で後続イシューに切り出す）
  - pandas 等、numpy／Keras 以外の Python ライブラリとの互換
  - `amax`/`max` 縮約 API（PyTorch `torch.amax` 相当）は対象外。
    `crates/autodiff/src/grad.rs` の `max_vjp` は同値タイ発生時「最初
    に現れる最大要素 1 箇所のみ」へ勾配を伝播する先勝ち決定的挙動を
    採用しており、PyTorch `amax` の均等分配とは異なる（PoC-v2-2 の
    ビット一致決定性方針との整合を優先した設計判断。#224 で再確認
    済み・変更なし）。compat／facade（REQ-9 追記・#52）の公開面に
    `amax` 相当の縮約 API を追加する段階になった場合にのみ、PyTorch
    互換（均等分配）の要否を改めて判断する
  - **任意 `BackendOps` 実装を注入できる推論入口**（旧 `Sequential::
    predict_with_ops`）は対象外。TASK-9.4（#411）で公開面から撤去した
    （破壊的変更。REQ-12「任意 `BackendOps` 実装を注入できる公開 API を
    設けない」・`crates/facade/tests/api_surface.rs` の機械検査と整合
    させるため。0 節・4.2 節参照）。`Sequential::predict` は既定バック
    エンド（`fandhe_ai::tape()`。TASK-2.5 ユーザー承認済み）へ透過的に
    結線される
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
  用いない（`04-requirements.md:207`）。自作コア確定後の再実装・再検証を
  要する
- **PoC-v2-6 を v2 実例として参照する**: `Mlp::from_safetensors`
  （`docs/spec/03-poc/poc-v2-6-interop/code/rust/src/mlp.rs`）は自作テンソル
  上に構築された薄い互換層の v2 実例である。numpy／Keras 慣習の本格的な
  互換 API 層そのものではないが、自作コア上でも薄いラッパーが成立し
  得ることを示す傍証として位置づける（`04-requirements.md:200-209`）

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

## 6. 出典一覧

| 出典 | 内容 |
|------|------|
| `docs/spec/04-requirements.md:200-215` | REQ-9 概要・受け入れ基準・関連 PoC |
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
| `docs/spec/04-requirements.md:209-210` | REQ-9 の 2026-08-08 追記（サポート境界の明文化） |
| `docs/spec/04-requirements.md:211-213` | REQ-9 の 2026-08-29 追記（`optim`・デバイス常駐更新経路を確定入口へ追加） |
| `docs/spec/04-requirements.md:390` | REQ-9 の 2026-08-29 追記に対応する改定履歴表エントリ |
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
