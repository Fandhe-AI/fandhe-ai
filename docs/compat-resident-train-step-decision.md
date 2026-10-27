# 常駐入力・ターゲットのハンドルと学習ステップ入口の公開形 決定記録

イシュー #3249（親 #3248「学習 1 step の同期を 1 回へ」）。
`docs/compat-train-step-hook-decision.md` §0 が定める「公開面の追加は
設計判断記録 → 実装の 2 段」の 1 段目にあたる。本記録は**公開形（名前・型・
エラー）の確定のみ**を扱い、コード（`.rs`）・`Cargo.toml`・`docs/spec`・
tolerance・baseline は変更しない。

## §0 結論

| 項目 | 確定内容 |
|---|---|
| 方式 | **案 A**（専用の学習ステップ・チェーン。facade の公開 API の追加）。案 B（Tape のデバイス値化）は不採用（§3） |
| 変更の種類 | **追加のみ**。既存の公開シグネチャ・挙動は変えない（`fandhe-ai =0.11.0` に対する追加。§7） |
| 初期の対応範囲 | 損失は MSE（mean）のみ、最適化は SGD のみ、層は `Linear`＋`ReLU` 系の常駐対応構成のみ（§5） |
| 公開状態 | 未公開（本記録は公開形の確定のみ。実装は #3250・#3251〜#3253・公開は #3254）。**追記（#3254）: P1〜P7 を facade に公開済み（§14。公開形は本記録の表のとおりで変更なし）** |
| `unsafe`・依存 | 新規 `unsafe`・依存追加なし。`unsafe impl Send` は作らない（§4） |
| REQ-12 | 整合する。所有者へ戻す項目なし（§6） |

## §1 位置づけ・出典

- 課題: 学習 1 step の同期回数が比較相手（candle・burn・PyTorch は 1 回）より多い。
  実測は Mac mini Metal で wait 7 回・GB10 CUDA で 8 回
  （`docs/perf/train-step-phase-breakdown.md` §18.4・§19.6）。
- 依存: #3217（`docs/train-step-resident-parts-design.md`。内部の encode-only 部品）は
  CLOSED・記録済み。後続: #3250（常駐入力の推論・行列積）・#3251（Metal チェーン）・
  #3252（CUDA チェーン）・#3253（CPU 経路）・#3254（facade 公開の仕上げ）・
  #3258（CUDA Graph の step 全体 capture）。後続は「名前・形は #3249 の記録に従う」
  としているため、本記録は具体的なシグネチャまで確定する。
- 関連 doc: `docs/inference-chain-single-sync-design.md`（推論チェーン・決定 7 の
  フォールバック前例・`_tracked` 契約）、`docs/backend-cuda-graph-step-capture-design.md`
  （§0・F2: step 全体 capture が構造的に不可能だった理由）、
  `docs/compat-api-scope.md` §0・§5。

## §2 所有者回答の転記

出典: イシュー #3249 のコメント
<https://github.com/Fandhe-AI/fandhe-ai/issues/3249#issuecomment-6111772331>
（投稿者 `aLiz-Nancy`。本文見出し「所有者回答（2026-10-12）」・createdAt
2026-10-11T17:35:34Z。「全て推奨で」との回答）。

| 論点 | 確定内容（回答文の転記） |
|---|---|
| 常駐の方式 | 案 A（専用の学習ステップ・チェーン。公開 API の追加） |
| 公開 API の形 | 不透明な常駐入力ハンドル・学習ステップ入口・loss ハンドル・ホスト転送を伴わない完了待ちの追加のみ（既存シグネチャは変えない）。名前は決定記録で決める |
| 対応範囲 | 初期は MSE と SGD。範囲外の構成は型付きエラーか従来経路へ落とす |

付帯条件: REQ-12 との整合を決定記録で確認し、整合しない形しか書けない場合だけ
所有者へ戻す（確認結果は §6）。個々の `unsafe` は security-auditor の監査が必須
（本記録は新規 `unsafe` を導入しない。必要になった時点で停止して所有者へ戻す）。
承認の範囲は上記回答文のとおりで、本記録が選ぶ具体名・シグネチャは回答が
「決定記録で決める」と委ねた事項である。

## §3 案 A／案 B の比較と案 B の不採用理由

| 観点 | 案 A: 専用チェーン（採用） | 案 B: Tape をデバイス値化 |
|---|---|---|
| 呼び出し形 | 新入口を追加（既存 `fit`・Tape 経路は不変） | 既存の `Tape`／`Var` API の裏でデバイス値を保持 |
| 波及範囲 | facade ＋ autodiff の内部チェーン | `TapeNode.value`・`backward_impl`・全 VJP |
| bit 一致の保ちやすさ | 固定順序のチェーンで Tape 経路と突合しやすい | VJP ごとに経路が分岐し検証面が広い |
| `Tape: Send` | 不変（デバイス値は Tape に入らない） | 崩れる |
| CUDA Graph の step 全体 capture | 可能（固定構造・確保ゼロ。#3258） | 不可（毎 step 作り直す動的グラフ） |
| 公開 API の追加量 | 7 項目（§4） | 少ないが内部の型が大きく変わる |

案 B の不採用理由（出典つき）:

1. `TapeNode.value` はホスト専用の `OnceCell<Tensor<f32>>`
   （`crates/autodiff/src/tape.rs:2730`）で、デバイス値へ変えると `backward_impl` と
   全 VJP（`crates/autodiff/src/grad.rs`）の入出力型に波及する。
2. 勾配は `Vec<Option<Tensor<f32>>>`（`crates/autodiff/src/backward.rs:49`・`:283`）で
   ホスト専用。
3. `Tape: Send` は静的アサーションで固定された契約
   （`crates/autodiff/tests/fusion_backend_integration.rs:388-392`）。`DeviceBuffer` は
   `Box<dyn BufferHandle>` を持ち、`BufferHandle: fmt::Debug + 'static` に `Send` が
   無い（`crates/tensor-core/src/buffer.rs:91`・`:136`）。Tape がデバイス値を抱えると
   `Tape: Send` が崩れ、Metal では `MetalBuffer` への新規 `unsafe impl Send` 等が要る
   （`docs/public-api-design.md` 項目 10・#1335 の整理）。
4. Tape は毎 step 作り直す動的グラフのため capture キーが安定せず、#3258 の
   step 全体 capture に向かない。

案 A の利点: `predict_device_chain`（`crates/autodiff/src/optim/device_store.rs:1896`）／
`build_device_chain_steps`（`crates/facade/src/compat/sequential.rs:3797`）と同型の固定
チェーン（forward → MSE → backward〈dW・bias 勾配・ReLU マスク・L2 の d_input。
L1 の d_input は計算しない = #3245〉→ SGD）を encode-only で積み、同期は loss 読み出し
の 1 回になる。

## §4 公開形（確定名。後続 issue はこの表に従い、変える場合は同じ PR で本記録を更新する）

### 設計の制約

- **ハンドルの寿命**: 常駐ハンドルは毎 step の `Tape` より長生きするため `'t`
  （Tape 寿命）を持たない。所有者の識別は `DeviceParamStore` の `store_id`＋`device`
  （＋CUDA の世代 `generation`）で行う（`ResidentLeaf` の `store_id` 検査の前例）。
  #3250 の「別の Tape・store のハンドルは型付きエラー」はこの検査で実現する。
- **束縛先**: ハンドル作成入口は 1 つに絞り（P2）、行列積（P4）も同じハンドル型を
  受ける。束縛先の検査は device＋世代を必須とし、ストアを取る API（P3・P5）では加えて
  `store_id` を検査する。理由: 入口を増やすと公開面と検査経路が二重化し、#3254 の
  正ガードも増えるため。
- **配置**: `DeviceParamStore` は
  `pub use fandhe_ai_autodiff::optim::{DeviceParamStore, ResidentLeaf, SgdConfig};`
  （`crates/facade/src/lib.rs:273`）で型ごと再エクスポートされており、autodiff 側で
  `DeviceParamStore` に `pub fn` を足すとその時点で facade の公開面になる。公開時期を
  #3254 で制御するため、公開入口は facade 側（`Sequential`・facade `Tape` の inherent
  メソッド、facade が定義する newtype）に置き、autodiff 側の内部実装は `pub(crate)`
  相当か facade から再エクスポートしない形にする。
- **Send**: 新しいハンドル型は `DeviceBuffer` を内包するため `!Send`（`DeviceParamStore`
  と同じ）。`Send` を約束しない・`unsafe impl Send` を足さない・#3254 で
  `assert_send::<…>()` を書かない。デバイス値は `TapeNode` に入らないため `Tape: Send`
  は不変。これが「`Tape: Send` と `MTLBuffer` の衝突を不透明ハンドル側で閉じる」答えである。

### 公開形の表

| # | 項目 | 確定形（facade） | 実装 issue |
|---|---|---|---|
| P1 | 不透明な常駐テンソルハンドル | `pub struct ResidentTensor(…)`（フィールド非公開・`!Send`・`Debug`。`shape()`・`device()` の読み取りのみ公開。入力とターゲットで共用。勾配は持たない） | #3250 |
| P2 | ハンドル作成入口 | `Tape::upload_resident(&self, store: &DeviceParamStore, x: &Tensor<f32>) -> Result<ResidentTensor, AutodiffError>` | #3250 |
| P3 | `predict_resident` の常駐入力版 | `Sequential::predict_resident_input(&self, store: &DeviceParamStore, input: &ResidentTensor) -> Result<Tensor<f32>, AutodiffError>`（出力はホスト `Tensor`。D2H と同期 1 回は残る） | #3250 |
| P4 | 行列積の常駐入力版 | `Tape::matmul_resident(&self, a: &ResidentTensor, b: &ResidentTensor) -> Result<Tensor<f32>, AutodiffError>`（rank 2・勾配なしの推論用。返り値はホスト `Tensor`） | #3250 |
| P5 | 学習ステップ入口 | `Sequential::train_step_resident(&self, store: &mut DeviceParamStore, x: &ResidentTensor, y: &ResidentTensor, config: &SgdConfig) -> Result<ResidentLoss, AutodiffError>`（損失は MSE・mean 固定を doc に明記。将来の損失・最適化は別入口の追加で扱い本シグネチャは変えない） | #3251〜#3253（内部）・#3254（公開） |
| P6 | loss ハンドル | `pub struct ResidentLoss(…)`（`!Send`）。`ResidentLoss::read(&self, store: &mut DeviceParamStore) -> Result<f32, AutodiffError>` が唯一の D2H＋同期点。ハンドルは store を所有・借用しないため、loss バッファ・step 世代・失敗トークン・poison 状態へは引数の store 経由で到達する（`store_id` 不一致は `InvalidArgument`）。読み出し後に失敗トークンを再検査し、セット済みなら store を poison して `StorePoisoned`。store を drop した後はハンドルを渡す先がなく read できない（バッファ寿命は store が保証し、共有所有〈Arc 等〉は採らない） | #3254 |
| P7 | ホスト転送を伴わない完了待ち | `Tape::wait_device_param_store(&self, store: &DeviceParamStore) -> Result<(), AutodiffError>`（既存 `sync_device_param_store_to_host`〈`crates/facade/src/lib.rs:676`〉の命名に揃える。D2H なしでデバイス完了を待ち、失敗トークンを検査） | #3253／#3254 |

### 付随決定

- **エラー**: 新しいエラー型・variant は足さず、既存の `BackendError`
  （`#[non_exhaustive]`。`crates/tensor-core/src/device.rs:202`）を `AutodiffError::Backend`
  で包んで返す。

  | 条件 | 返す variant |
  |---|---|
  | 範囲外の構成・損失・最適化（§5 の第 1 段） | `Unsupported`（または `InvalidArgument`） |
  | 不正な `SgdConfig`・古い `ResidentLoss`・rank 不正 | `InvalidArgument` |
  | x と y の形状不一致・層の入出力形状不一致 | `ShapeMismatch` |
  | 別デバイスで作ったハンドル | `DeviceMismatch` |
  | CUDA の世代が古いハンドル | `StaleDeviceGeneration` |
  | 別ストアのハンドル | `InvalidArgument`（`store_id` 不一致） |
  | poison 済みストア・読み出し後の失敗トークン検知 | `StorePoisoned` |
  | `forward_resident` 後の未消費 pending が残るストア | `PendingForwardUnconsumed` |
  | 別 Tape の登録を渡された場合 | `TapeMismatch` |

- **loss バッファ**: store が持つ永続スカラー（ワークスペース）に書き、`ResidentLoss` は
  軽量ハンドル（`store_id`＋step 世代カウンタ。バッファ・失敗状態は持たず、`read` が受け取る
  store 経由で参照する。所有権は store に一本化し共有所有は採らない）とする。次の `train_step_resident` で
  上書きされた後の古いハンドルの `read` は fail-closed（`InvalidArgument`）。理由:
  定常状態の確保ゼロが #3258 の capture 前提であるため。
- **完了待ちの内部部品**: `BackendOps`／`MemoryOps` に D2H なしの同期メソッドは存在
  しない（`crates/tensor-core/src` に `synchronize`／`wait_idle` なし）。既定実装つきの
  内部メソッド追加（#3217 記録 §3 と同じ非破壊追加の型）が必要で、#3253（CPU は即時完了）・
  #3251／#3252 へ割り当てる。需要元は
  `scripts/bench/framework-compare/bench-fandhe/src/main.rs` の `run_train_reuse`
  （1138 行付近）のコメント。
- **store の状態機械との共存**（`crates/autodiff/src/optim/device_store.rs` モジュール doc
  「状態機械」〈55 行付近〉・「状態種別ガード」）: チェーンは `register_resident_params`
  の pending を作らずにパラメータを読む（`snapshot_resident_params` と同様）。`forward_resident`
  後に未消費の pending があるストアは `PendingForwardUnconsumed` で拒否する。状態種別
  ガード（SGD と RmsProp／Adagrad／LAMB の混在拒否）・momentum 構成の一貫性検査・
  `check_not_poisoned` は既存 `DeviceParamStore::step`（同 `:2190`）と同じものを通す。
- **SGD の対応フィールド**: `SgdConfig`（`crates/autodiff/src/optim/sgd.rs:34`）の `lr`・
  `momentum`・`dampening`・`weight_decay`・`nesterov` のうち、既存 `DeviceParamStore::step`
  が `config.validate()`（有限値・非負・nesterov 条件）の後に受け付ける範囲と**同じ範囲**を
  受け付け、範囲外は既存 `step` と同じ型付きエラー（`InvalidArgument`）にする。
  MSE は mean のみで、他の `MseReduction` は `Unsupported`。

## §5 範囲外の扱い（2 段を区別する）

1. **モデル構成・損失・最適化が範囲外**（`Linear`＋`ReLU` 以外の層、`Sigmoid`／`Tanh` 混在、
   未 track のパラメトリック層、空 `Sequential`、MSE mean 以外、SGD 以外）: facade 入口で
   store を変更する前に `Unsupported`／`InvalidArgument` を返す
   （`contains_resident_unsupported_layer`〈`crates/facade/src/compat/sequential.rs:3184`〉・
   `reject_untracked_parametric_layer_resident`〈同 `:3163`〉と同じ fail-closed）。
   黙って別の計算をしない。
2. **バックエンドがチェーン部品を持たない**（ROCm・未 override で部品が `Unsupported`）:
   store に何も書いていない段階に限り、step 全体を従来の Tape 経路
   （`forward_resident` → MSE → `backward_device_param_store` → `step_device_param_store`）へ
   フォールバックする（推論チェーンの決定 7 と同型）。これは #3251〜#3253 が Tape 経路との
   bit 一致を受け入れ条件にしているから許される。`BackendOps` 内部でのホストへの黙示
   フォールバックは引き続き禁止（#3217 記録 §3・A08）。部分的に積んだ後の失敗はフォール
   バックせずエラー（必要なら poison）。

## §6 REQ-12 との整合

確認箇所: `docs/spec/04-requirements.md` の REQ-12 受け入れ基準「ライブラリ利用者が明示的に
融合を制御する API は提供しないこと」と 2026-08-08 注記（`facade` を唯一の公開面とし利用者向け
公開面を `Device` 識別子に限る）。確認コマンド: `grep -n "利用者が明示的に融合" docs/spec/04-requirements.md`。
あわせて REQ-9 の対象外（汎用グラフ JIT・利用者向け `compile` 入口。
`docs/graph-compile-scope-spec-proposal.md` §4）を確認した。

結果: **整合する。所有者へ戻す項目なし。**

- P1〜P7 はどれも融合の有無・カーネル選択・同期方式を利用者に選ばせない（チェーン化は
  内部で自動。`fit`／`predict_resident` と同じ「学習・推論の手続き API」）。
- `BackendOps`／`MemoryOps`／`DeviceBuffer`／`DispatchFailureCell` をシグネチャに出さない
  （`Sequential::init_device_param_store` 付近の「`mem`／`ops` を表面に出さない」原則に従う）。
- 任意の `BackendOps` を注入する経路を作らない。トレース・グラフ書き換え・`compile` 入口では
  ない。

## §7 不変条件

- 既存の公開シグネチャ・挙動は変えない（追加のみ）。
- `pub use` は 1 文 1 行を維持する（`crates/facade/src/lib.rs` の注記・
  `crates/facade/tests/api_surface.rs` の `facade_does_not_reexport_tape_or_backend_ops`）。
- tolerance・baseline・FMA 契約・f64 アキュムレータ契約（bias 勾配 bit 一致）・REQ-8 の
  カーネル境界検査は不変。
- 数値契約: 学習ステップ入口の loss・更新後パラメータは同一バックエンドの Tape 経路と
  bit 一致（NaN はクラス一致）。`predict_resident_input`／`matmul_resident` はホスト入力版と
  bit 一致。
- 新規 `unsafe`・依存追加なし。

## §8 後続 issue への引き継ぎ

- **#3250**: P1〜P4 の実装。別ストア・別デバイス・古い世代の型付きエラー。ホスト版との
  bit 一致テスト（CPU は CI、CUDA・Metal は `#[ignore]`）。H2D 0 回の診断カウンタ確認。
  **実装済み（§10 に記録）。実機のうち GB10（CUDA）は未実測、Metal は開発機で実測済み。**
- **#3251／#3252／#3253**: P5 の内部チェーン（`_tracked` で store の failure_token へ登録、
  L1 の d_input は計算しない）と完了待ちの内部メソッド。
  **#3251 は実装済み（§11 に記録。ループはバックエンド非依存で #3252・#3253 が再利用する。
  Metal は Mac mini で wait 1 回・Tape 経路と bit 一致を実測済み）。完了待ちの内部メソッド（P7 の実体）は
  #3251 では足さず #3254 へ寄せた（§11 範囲外）。#3253（CPU）は実装済み（§13）。**
- **#3254**: facade 公開の仕上げ。`crates/facade/tests/api_surface.rs` に正ガード（追加した名前の
  `pub use` 行・facade の inherent `pub fn` 集合・関数ポインタ型による署名固定プローブ・
  利用例 doctest の実在検査）を足す。内部実装が先に入る期間に facade へ早期に漏れないよう、
  必要なら `*HoldDoctestGuard`（`crates/facade/src/lib.rs` の既存パターン）で保留を固定し、
  #3254 で正ガードへ反転する。`!Send` を前提とし `assert_send` を足さない。
  `docs/compat-api-scope.md` §5 の適用記録・`docs/public-api-design.md`・`docs/README.md` を更新する。
- **#3255**: 公開 API だけでの Tape 経路との bit 一致テスト。**実装済み（§15。CPU は CI、CUDA・Metal は `#[ignore]` で未実測）。** 実機の実行・実測は §16（#3256）。
- **#3258**: P5 の固定構造（永続ワークスペース・永続 loss スカラー）を capture 区間の前提として
  参照する。
  **#3254 は実装済み（§14 に記録）。**

## §9 OWASP Top 10 観点

- **A01／A04**: 別ストア・別デバイス・古い世代のハンドルを `store_id`・`device`・`generation` で
  拒否し、範囲外構成は状態変更前に型付きエラー。黙示の経路変更はしない（フォールバックは
  bit 一致が保証された Tape 経路へ、状態未変更時のみ）。
- **A03**: shape・rank・numel・offset の `checked_mul`／`checked_add` 検査を driver／encode
  呼び出しより前に行う（#3217 記録 §3）。カーネル側の手動境界チェックは省略しない（REQ-8）。
- **A05／A06**: 依存追加なし。新規 `unsafe` なし。必要になったら停止して security-auditor の
  監査と所有者確認へ戻す。
- **A08**: encode-only は `_tracked` 版で store の failure_token へ登録し、loss 読み出しと
  完了待ちの後に失敗トークンを再検査して poison する。Tape 経路との bit 一致を受け入れ条件に
  固定し、tolerance・baseline は変えない。
- **A09**: 同期・H2D 回数の診断カウンタによる実機確認を後続 issue の受け入れ条件に残す。

## §10 #3250 実装記録（P1〜P4）

### 実装した名前と配置

| 層 | 名前 |
|---|---|
| tensor-core | `BackendOps::gemm_device(&self, a, b) -> Result<DeviceBuffer<f32>, BackendError>`（既定は `Unsupported`。非破壊） |
| autodiff（内部名） | `optim::resident_input::ResidentInput`（P1 の実体）、`optim::resident_input::matmul_resident(tape, a, b)`、`DeviceParamStore::upload_resident_input(tape, x)`、`DeviceParamStore::predict_device_chain_resident(tape, input, steps)` |
| facade（確定名） | `ResidentTensor`（P1。`shape()`・`device()`・`Debug` のみ）、`Tape::upload_resident`（P2）、`Sequential::predict_resident_input`（P3）、`Tape::matmul_resident`（P4） |

- facade の `pub use fandhe_ai_autodiff::optim::{..}` には `ResidentInput` を足さない（`facade::Tape.0` が
  `pub(crate)` で生の autodiff `Tape` を facade から渡せないため、前例の `predict_device_chain` と同じく
  autodiff 側は `&Tape` を取る `pub fn`、facade が薄く委譲する）。
- 公開の仕上げ（`api_surface.rs` の正ガード・doctest・`compat-api-scope.md` §5）は #3254。本 PR は
  doc コメントに ```rust の例を書かない。`!Send`（`DeviceBuffer` が `Send` でない）で `assert_send` なし。

### 挙動の決定

- **P3 はホスト経路へ落とさない**。`predict_resident`（ホスト入力版）は `build_device_chain_steps` が
  `None`（Sigmoid／Tanh 混在・層の過不足・空 `Sequential`）やバックエンドの `Unsupported` のとき
  ホストの Var 経路へ全体フォールバックするが、`predict_resident_input` はフォールバックすると本件が
  取り除く H2D が戻る（入力を download する）ため、`Unsupported` を返して止める。チェーンは store を
  書き換えないので状態は変わらない。`predict_resident` の挙動は不変。
- **チェーン本体を共有**: `predict_device_chain_once` が入力を `ChainInput`（ホスト〈upload して所有し第 1 層の
  発行直後に解放〉／常駐〈借用のみ・解放しない〉）で受け、層ループ・scratch 再利用・download 後の
  failure_token 再検査と poison・capture 衝突の待機再試行を両版で共有する（bit 一致を構造で保つ）。
- 常駐入力のチェーンは Graph capture 経路（`linear_chain_forward_captured`。ホスト入力を capture 内で
  upload する API）を通らない。**常駐入力の capture 対応は #3258 への申し送り**。
- 追加検査はチェーンを積む前: 別 store → `InvalidArgument`、別デバイス → `DeviceMismatch`、rank != 2 →
  `InvalidArgument`、列数と第 1 層 weight の行数の不一致 → `ShapeMismatch`、世代切れ（CUDA）→
  バックエンドの `StaleDeviceGeneration`。poison 済み store → `StorePoisoned`。
- **P4 のカーネルは `gemm` と同じ出どころ**: ホスト版は `Var::matmul` → `ops.gemm`。既存のデバイス入力
  カーネルは別の起動経路なので bit 一致を当てにせず、`gemm_device` を足した。
  - CPU: `gemm` の汎用分岐と同じ `gemm_blis_parallel` をスライスのまま呼ぶ。
  - CUDA: `Fp32Strict`（既定）のみ。`gemm` が contiguous 入力で通る NN 経路（`select_tiled_f32_kernel(0, ..)`）と
    同じ `launch_tiled_f32_resident`（`a_offset = 0`）を持つ `gemm_resident_lhs_device_impl`（`w @ b`・RowMajor）へ
    委譲する。**`Tf32`／`Tf32x3` はデバイスポインタを受ける核がないため `Unsupported`**（別の精度で黙って計算しない）。
  - Metal: `gemm`（`dispatch_auto`）と同じ経路選択（split-K の 3 段ゲート → 経路 → classic）を、upload 済み
    バッファに対し `MetalGemm::dispatch_auto_resident` で行う。`dispatch_auto` は 8 の倍数へホストでパディング
    してから起動するため、**`m`／`n`／`k` が 8 の倍数のときだけ**同じ dispatch・bit 一致になる。倍数でない形状は
    常駐バッファをホストへ戻さずにパディングできないため `Unsupported`。まず `linear_forward_device` の核で
    試したところ、split-K 形状（64×784×256）で 1 ULP ずれたため採らなかった。
  - ROCm: 既定の `Unsupported` を継承。
- 新規 `unsafe` なし・依存追加なし・tolerance／baseline 変更なし。

### 検証

- CI で動く: tensor-core（既定）・CPU の `tests/gemm_device_parity.rs`（`gemm` と bit 一致・不正 shape）、autodiff の
  モック（常駐版 Δupload = 0・Δdownload = 呼び出し数、対照のホスト版 Δupload = 1、別 store・rank・列数・
  poison）、facade の `tests/resident_input_inference.rs`（CPU の bit 一致・型付きエラー・同じハンドルを a と b に
  渡す・拒否後も store が使える）。
- 実機 `#[ignore]`: facade の同ファイル（CUDA・Metal）、`backend-cuda/src/resident_input_diag_tests_3250.rs`、
  `backend-metal/src/resident_input_diag_tests_3250.rs`（`gemm_device` と `gemm` の bit 一致、窓の中の入力 H2D = 0
  〈CUDA は `mem_uploads`〈`MemoryOps::upload` の新カウンタ〉と `h2d_calls`、Metal は `new_with_data` の
  `cfg(test)` カウンタ〉、D2H は読み出し 1 回以上）。
- **実測状況**: Metal は開発機（Apple M4 Max・debug ビルド）で `#[ignore]` テスト全件 pass を確認した
  （bit 一致・窓の中の H2D = 0）。**GB10（CUDA）は未実測**（ビルドのみ確認）。性能（N=256〜4096 の gemm・infer の
  H2D 除去量）は未計測で、本 PR は性能の合格ラインを主張しない。

## §11 #3251 実装記録（学習ステップ全体の encode-only チェーン。P5・P6 の内部実装）

### 実装した名前と配置

| 層 | 名前 |
|---|---|
| autodiff（内部名） | `DeviceParamStore::train_step_device_chain(tape, x, y, steps, config) -> TrainStepLoss`、`DeviceParamStore::read_train_step_loss(tape, &TrainStepLoss) -> f32`、`optim::resident_input::TrainStepLoss`（P6 の内部側。`store_id` と世代だけを持つ不透明型） |
| autodiff（配置） | `crates/autodiff/src/optim/device_store/train_chain.rs`（`amp.rs`・`distributed.rs` と同じ子モジュール）。`grad.rs::mse_loss_scale` を `pub(crate)` にした（Tape 経路と同じ式を共有） |
| facade | 本番コードの追加なし。`Sequential`・`Tape` に `pub fn` を足さない（`api_surface.rs` が impl 面を走査しているため）。テスト補助だけを `#[cfg(test)]` の `compat/sequential/train_step_chain_tests_3251.rs` に置いた |

- **チェーンのループはバックエンド非依存**: ループが呼ぶのは `BackendOps::*_tracked` の 6 部品と
  `sgd_step_device_tracked` だけ。Metal 固有なのは実機テストと記録だけで、#3252（CUDA）・#3253（CPU）は
  部品を実装すれば同じループを再利用する（チェーンが 2 本にならない）。
- 手順（層 `i = 1..N`）: forward（`linear_forward_device_tracked` を連鎖。層出力は全部保持）→ `mse_loss_device_tracked`
  （mean）→ `mse_loss_backward_device_tracked`（`scale` は Tape 経路と同じ `mse_loss_scale(1.0, m*n, Mean)`）→
  `i = N..1` の降順で dW（`gemm_fp32_strict_into_device_tracked`。入力は転置、勾配は RowMajor、bias 縮約を融合し、
  勾配連結バッファの `layout` のオフセットへ直接書く）、`i >= 2` のときだけ d_input（`gemm_resident_lhs_device_tracked`）
  → ReLU マスク（`relu_backward_mask_device_tracked`）。**L1 の d_input は計算しない**（#3245）→ SGD 1 回。
- 勾配連結バッファ（`train_chain_grad`）・loss スカラー・中間バッファは store が次の step まで保持する。
  `grad_staging` とは別にして、その鮮度契約（`filled`／`bias_host_routed`）を汚さない。

### 対応範囲と事前検証（encode より前。違反は型付きエラーで store 不変）

- `steps` の最後の層以外は `Activation::Relu`、最後の層は `Activation::None`（`Unsupported`）。
  ReLU のない中間 Linear は勾配が転置格納のまま dW に渡り、Metal の dW デバイス版が `Unsupported` になる
  （Tape 経路もホスト経路へ落ちて bit 一致を保証できない）ため。最終層の ReLU は MSE の出力側マスクが要るため範囲外。
- `steps` が store の全パラメータをちょうど 1 回ずつ覆うこと（取りこぼし〈勾配バッファに古い値が残る〉は
  `Unsupported`、weight tying は `InvalidArgument`）。層の `in/out` の連鎖、`x`・`y` の rank／形状、空バッチ（`Unsupported`）。
- `x`・`y` は同じ store・同じデバイスの `ResidentInput`。`pending`（forward 登録が未消費）があれば `PendingForwardUnconsumed`。
- SGD の前提条件は `step()` と同じ（`config.validate()`、RmsProp／Adagrad／LAMB の履歴の拒否、momentum 構成の途中変更の拒否、
  `is_first_step = !sgd_used`、velocity の遅延確保）。

### 失敗の境界（§5 の 2 段目）

- 最初の dW を書く前に部品が `Unsupported` を返したら、そのまま返して poison しない。params・velocity・`step_count`・
  `sgd_used` は不変。
- それ以外の `Err`（dW 以降・SGD・`bias_filled == false` の契約違反）は `step()` と同じく poison。
  `DeviceContextCaptureInProgress`（SGD 入口。params 未変更）だけは poison しない。
- 成功時は encode 成功の時点で `step_count`・`sgd_used`・世代を確定する。loss の読み出しは確定に影響しない。
- `read_train_step_loss`: 別 store・古い世代のハンドルは `InvalidArgument`。download の失敗は poison。
  download の後に `failure_token` を再検査し、セット済みなら poison して `StorePoisoned`（A08）。
- Tape 経路への**全体フォールバックは作らない**（黙って別の計算をしない）。公開入口（#3254）が判断する。

### 検証

- CI（GitHub ホステッド）: autodiff の記録型モック 8 件（部品の呼び出し回数〈L1 の d_input 無し〉・チェーン区間の
  upload／download 0・全部品が store の `failure_token` を受けること・`is_first_step` の列・失敗時の poison 境界・
  事前検証・古い／別 store のハンドル・pending）と、facade の CPU テスト 1 件（CPU は部品が未実装〈#3253〉で `Unsupported`、
  store 不変。#3253 が入って `Ok` になったら loss の有限性を見る）。
- 実機 `#[ignore]`（facade の in-crate）: Tape 経路（`forward_resident` → `mse_loss` → `backward_device_param_store` →
  `step_device_param_store`）との bit 一致（784-256-10・784-128-64-10、batch 64、momentum 0／0.9／dampening／weight_decay／
  nesterov、各 3〜4 step、loss と全パラメータ）、1 step の wait。
- **実測（Mac mini・5 run すべて pass）**: bit 一致 A1・A2 は全 run で Tape 経路と一致。A3 はチェーン区間 wait = 0、
  loss 読み出しを含めて wait = 1、encode = 10（机上導出と一致。MSE forward が partial + finalize の 2 段のため 9 部品呼び出しで
  10 encode）、command_buffers = 1。受け入れ条件 1（学習 1 step の wait が 1 回）を満たす。
  記録は `docs/perf/logs/train-step-chain-metal-3251/`（RULE.txt を事前登録）。
- 性能（0.65〜0.9 ms の目安）は本 issue では計測していない（`m = 64` タイルの見直し〈#3207〉との合算評価が別途要る）。

### 範囲外・申し送り

- **#3254**: 公開入口 `Sequential::train_step_resident`・`ResidentLoss::read`、P7 `Tape::wait_device_param_store`、
  `BackendOps` の D2H なし同期メソッド（P7 の内部実体。#3251〜#3253 に割り当てていたが本 issue の作業項目・受け入れ条件に
  なく、並行する #3252・#3253 との trait の衝突点になるため #3254 へ寄せる）、Tape 経路への全体フォールバック（§5 の 2 段目）。
  facade のテスト補助（`chain_train_step`）の形を本番コードへ移す。
- **確保ゼロ化は範囲外**: Metal の各部品は新しいバッファを返す（pool 再利用で同期は起きない。#3219・#3224 の実測で部品連結は
  wait 0）。`_into` 版の部品は新しい設計項目で #3258 と同系。
- **#3252（CUDA）・#3253（CPU）**: 部品が実装済みなら同じ `train_step_device_chain` がそのまま動く。CUDA の Graph capture
  （#3258）は固定構造（永続ワークスペース・永続 loss スカラー）を前提にする。

## §12 #3252 実装記録（CUDA の学習チェーン。区間内の同期・H2D・確保を 0 にする）

### 実装した名前と配置

| 層 | 名前 |
|---|---|
| tensor-core（内部 trait の非破壊拡張。既定は `Unsupported`） | `BackendOps::mse_loss_device_into_tracked`・`mse_loss_backward_device_into_tracked`・`relu_backward_mask_device_into_tracked`・`gemm_resident_lhs_device_into_tracked`（いずれも `_tracked` 版だけ。末尾の `out` が出力先、その前が既存部品と同じ引数、最後が `token`）。#3205 の `linear_forward_device_into_tracked` と合わせて、チェーンの出力を作る 5 部品が `_into` 版を持つ。dW は #3223 から既に勾配連結バッファへ書く形のため追加なし |
| backend-cuda | `ops.rs` の 4 本の `*_impl` を `*_core(.., out: Option<&mut DeviceBuffer<f32>>) -> Result<Option<DeviceBuffer<f32>>, _>` へ一般化し、確保版は `*_core(.., None)`、`_into` 版は `*_core(.., Some(out))` を呼ぶ（検査・起動ヘルパ・カーネルは共有）。`token` は使わない明示 override。診断は `gemm.rs` の `GemmLaunchDiagCounters::mem_allocs`・`diag_count_mem_alloc`（`CudaMemory::alloc_zeroed_inner` の成功、MSE partial・GEMM device workspace の grow で数える） |
| autodiff | `device_store/train_chain.rs`: `TrainChainBufs`（層出力・dPred・d_input・mask 後勾配。鍵は `(m, 各層の出力幅)`）、`TrainChainCaps`（部品ごとの `_into` 対応ラッチ）、`produce_part`。`predict_device_chain` の `chain_scratch` とは別フィールド（鮮度契約を混ぜない）。公開入口は増やしていない（`api_surface.rs` 不変） |

- **チェーンのループは #3251 のまま 1 本**。各部品はまず再利用バッファへ `_into_tracked` を試し、`Unsupported` なら
  その部品だけ確保版 `_tracked` へ進み、`caps` へラッチして以後は `_into` を試さない。`Unsupported` は部品単位で
  消費し、チェーン全体の `Unsupported` としては表に出さない（Metal・CPU の挙動と §11 の失敗境界は不変）。
  世代切れ（`StaleDeviceGeneration`）は再利用バッファを捨てて確保し直す。初回（再利用バッファなし）は確保した出力へ
  `_into` を 1 回だけ試して能力を確定する（同じ結果の上書き。`predict_device_chain` と同じ方式）。
- **失敗経路のバッファは再利用せず `train_chain_hold` へ退避**する（実行中の GPU 処理と競合させない）。成功した step の
  バッファだけを次 step の出力先として store に戻す。
- **古い loss ハンドル**: loss バッファを再利用する step の開始時に世代を進める。書き込み後に失敗しても、古いハンドルが
  新 step の途中値を読むことはない（世代検査が拒否する）。

### bit 一致の前提（コードで確認した事実）

再利用バッファは確保版のような memset 済みではない。bit が確保版と一致するのは、出力カーネルが各要素へ**無条件に
代入**し既存値を読まないため。

- ReLU マスク: `ew_relu_mask_select_f32` は `out[idx] = ..`（`kernels_elementwise.rs`）
- MSE forward: `mse_finalize_f32` の lane 0 が `out[0] = block_sum * factor`、backward は `dpred[idx] = scale * (..)`
  （`kernels_mse.rs`）
- d_input・forward の GEMM: 選ばれうる tiled f32 variant はすべて `c[..] = acc`（`kernels.rs`・
  `kernels_tiled_pipeline.rs`・`kernels_tiled_pipeline_128x64.rs`。`+=` で `c` を読む variant は無い）
- 代入でない variant が将来入った場合は、その `_into` 版で再利用バッファへストリーム順序の `memset_zeros` を先に積んで
  確保版と bit をそろえる（memset は確保でも同期でもない）。黙って bit ずれを受け入れない

### 窓内の隠れた確保・同期の監査

- 確保: チェーンから到達する GEMM 経路（`launch_gemm_dw_device`・`launch_gemm_dinput_device`・`launch_tiled_f32_resident`・
  `launch_tiled_bias_act_f32_resident`）に呼び出しごとの `alloc_zeros` は無い。`gemm.rs` の `alloc_zeros`（tile_counter・
  unit_counter・partials）は `compile_*` 系（構築時）にあり、呼び出しごとの経路ではない。永続ワークスペース
  （`CudaGemm::device_workspace`・`CudaMse::partial_workspace`）は初回だけ grow し、grow も `mem_allocs` で数える
  （初回 step で 11 件、定常状態で 0 件を実測）。bias 縮約（`launch_sum_axis_generic_into`）・SGD 起動も確保しない
- 同期: チェーンの部品から到達する `synchronize()` は無い（`memory.rs` の `readback` 系・`host_readback`・`host_view_managed`
  は `download`／`with_host_view` 経路で、`read_train_step_loss` の 1 回だけが通る。`pool.rs` の同期は破棄経路）。
  計数不能なのは cudarc `CudaSlice::Drop` が `has_async_alloc == false` の環境でだけ行う同期で、GB10 は `true`
  （#3223 §6 と同じ前提）。定常状態では `_into` による再利用で `CudaSlice` の drop 自体が起きないため、非 async-alloc
  環境でも窓内には出ない
- 診断カウンタ（`cfg(any(test, feature = "internal-diagnostics"))`）の範囲外はこの監査で補う。既定ビルドの公開面・性能に影響しない

### 検証

- CI（GitHub ホステッド）: tensor-core の既定動作（4 部品とも `Unsupported`・`token` 非接触）、backend-cuda の
  `out` のデバイス／shape 不一致を driver より前に拒否する単体テスト、autodiff の記録型モック 5 件（`_into` 対応時に
  2 step 目以降の確保版部品呼び出し・`alloc_zeroed`・upload・download がすべて 0／`_into` 未対応時に従来経路で同じ結果で
  `Unsupported` を表に出さない／鍵が変わったら作り直す／世代切れで確保し直す／失敗 step のバッファは再利用せず hold に残る）。
  既存モック 12 件（L1 の d_input 無し・poison 境界・失敗トークンほか）は変更なしで通る
- 実機 `#[ignore]`（GB10・5 run 各すべて pass。記録は `docs/perf/logs/train-step-chain-cuda-3252/`。RULE.txt を事前登録）:
  - B1（`train_step_chain_diag_tests_3252`）: 784-256-10・batch 64・momentum 0.9・ウォームアップ 3 step の後の窓で
    `stream_syncs`・`d2h_calls`・`h2d_calls`・`mem_uploads`・`mem_allocs` がすべて 0、loss 読み出しで `stream_syncs` = 1・
    `d2h_calls` = 1。対照として初回 step の `mem_allocs` = 11、同期 0
  - B2・B3（facade の in-crate）: loss と更新後の全パラメータが Tape 経路と bit 一致（784-256-10・784-128-64-10・
    momentum 0／0.9／dampening／weight_decay／nesterov・各 3〜4 step。2 step 目以降が再利用経路）
- Metal の後退なし: 開発機（M4 Max）で `train_step_chain_tests_3251` の Metal `#[ignore]` 3 件が通る（Metal は `_into` 未対応のため
  経路は不変）。Mac mini での再計測は未実施
- `unsafe {` の数（変更前後）: backend-cuda の `ops.rs` 0→0・`gemm.rs` 28→28・`memory.rs` 5→5・`mse.rs` 3→3・
  `elementwise.rs` 7→7・`reduce.rs` 12→12。**新規 `unsafe` なし**（起動ヘルパは既に `&mut CudaArgMut` を受けるため）
- 性能（目安 0.18〜0.25 ms）は本 issue の受け入れ条件に含まれないため計測していない（A/B は別途）

### 範囲外・申し送り

- **#3254**: 公開入口、`ResidentInput` を永続バッファへ `upload_into` で更新する入口（窓の外。公開形を伴う）、P7 の D2H なし完了待ち、
  Tape 経路への全体フォールバック
- **#3258**: step 全体の CUDA Graph capture。永続スクラッチ・永続 loss は固定構造として揃った（capture 中は確保・grow を
  拒否するため、capture 前にウォームアップで定常状態にする必要がある）
- **Metal の `_into` 版部品（確保ゼロ化）**: 未着手。Metal は従来どおり step ごとに確保する（pool 再利用で同期は起きない）
- **CPU（#3253）**: 実装済み（§13）。同じループを再利用する

## §13 #3253 実装記録（CPU の学習チェーン部品）

所有者回答（§2。案 A・既存シグネチャは変えず追加のみ・初期対応は MSE と SGD）の範囲で、CPU の
`BackendOps` に足りなかった部品を override した。**CPU 専用のチェーンは作らず、§11 のループをそのまま
再利用する**（チェーンを 2 本にしない）。facade の本番コード・公開面（`api_surface.rs`）は変えていない
（公開入口 P5〜P7 は #3254）。

### 実装した部品（`crates/backend-cpu/src/ops_train_chain_parts.rs`・`ops.rs` の `impl BackendOps`）

| 部品 | 確保版 | `_into_tracked` | 比較先のホスト経路・核 |
|---|---|---|---|
| forward | 既存 `linear_forward_device` | 新規 | 同じ `gemm_blis_bias_act_parallel_prepacked`（共有本体 `linear_forward_device_core`） |
| MSE forward | 新規 | 新規 | `mse::mse_sum_sq_f32`＋最終式（`mse_reduce_value` をホスト `mse_loss` と共有） |
| MSE backward | 新規 | 新規 | `mse::mse_loss_backward_f32` |
| ReLU マスク | 新規 | 新規 | 選択（`mask > 0 ? g : +0.0`）。`elementwise_mul_mask` と同式 |
| dW＋bias 縮約 | 新規 | （trait に `_into` 版は無い。確保版が out と offset を直接受ける） | `(Transposed, RowMajor)`→`gemm_blis_parallel_tn`・`(RowMajor, Transposed)`→`_nt`・`(RowMajor, RowMajor)`→`gemm_blis_parallel` |
| d_input | 新規 | 新規 | `Transposed`→`gemm_blis_parallel_nt`・`RowMajor`→`gemm_blis_parallel` |

各部品は共有本体 `*_core(.., out: Option<&mut DeviceBuffer<f32>>)` を 1 本持ち、確保版は `None`、`_into`
は `Some(out)` を渡す（検査とカーネル呼び出しを二重に持たない。CUDA §12 と同じ構成）。`_tracked`
（非 into）は override せず既定の委譲に任せる。CPU は同期実行のため `DispatchFailureCell` は使わず、
エラーは同期的に `Err` で返す。

### 設計判断

- **0 埋めが必須**: BLIS 核は C タイルを load して書き戻すため、再利用する出力先は書き込み前に必ず
  `fill(0.0)` する（ホスト `gemm_fp32_strict_into` と同じ。無いと 2 step 目以降で bit が崩れる）
- **bias 縮約の融合**: チェーンは dW の呼び出しで `bias_filled == false` を契約違反として止める。CPU は
  ホストで `reduce_bias_grad_rows`（`f64` 逐次和・最後に 1 回 `f32`・`k == 1` は複製して `-0.0` 保持）を
  通るため、デバイス版も同一アルゴリズムで `out[bias_offset..+n]` へ書き `true` を返す
  （`.claude/rules/coding-rust.md` の `f64` アキュムレータ契約。ホストの CPU 経路は変更していない）
- **`(Transposed, Transposed)` は `Unsupported`**: ホストは `contiguous()` 実体化の別経路で 1 対 1 の核が
  無い。別の核で黙って計算しない（チェーンはこの形を使わない）
- **退化形状**: `_into` 版は空・`q == 0` を契約どおり `Unsupported`（確保版へ任せる）。確保版は空／全 0 を返す
- 境界検査（REQ-8・A03）: shape・offset は `checked_*`、スライス取得は `get(range)`。検査はすべて書き込み前で、
  `Err` のとき `out` は変わらない

### 検証（すべて CI で実行。`#[ignore]` なし）

- `crates/backend-cpu/tests/mse_device_parity.rs`・`relu_mask_device_parity.rs`・`gemm_device_input_parity.rs`:
  ホスト版との `to_bits()` 一致（NaN はクラス一致）、事前に汚した `out` への上書き、view offset、
  `k == 1` の `-0.0`、`out_offset`・bias 範囲のオーバーフロー／重なり拒否、型付きエラー時の `out` 不変
- facade `train_step_chain_tests_3251.rs`: CPU のチェーンが Tape 経路と loss 列・全パラメータで bit 一致
  （`[784,256,10]` plain 4 step・`[96,64,10]` momentum／nesterov・3 層・端数形状・batch 1。step 2 以降は
  `_into` 再利用経路）。範囲外（Sigmoid／Tanh・最終層 ReLU・空バッチ・別 store の入力）は型付きエラーで
  store 不変。CI の debug ビルドの負荷のため 784 幅は 1 構成に絞った
- 新規 `unsafe` なし・依存追加なし・tolerance／baseline 不変

### 範囲外・申し送り

- 公開入口・`ResidentLoss::read`・P7 の D2H なし完了待ち・Tape 経路への全体フォールバック: #3254
- GEMM 内部のパック用ヒープ確保の削減: #3197（job 統合）の範囲
- Mac mini・GB10 の CPU セルでの A/B 計測: 受け入れ条件外（CPU は同期がなく効果は小さい見込み）

## §14 #3254 実装記録（facade への公開。P1〜P7 の仕上げ）

イシュー #3254（親 #3248）。公開形は §4 の表のとおりで**署名は変えていない**（追加のみ）。
新しい `pub use` 行・依存・`unsafe`・エラー型・variant は足していない。

### 実装した名前と配置

| 名前 | 配置 |
|---|---|
| P1〜P4（実装は #3250） | `crates/facade/src/lib.rs`（`ResidentTensor`・`Tape::upload_resident`・`Tape::matmul_resident`）・`compat/sequential.rs`（`predict_resident_input`）。本 PR で正ガード・doctest を追加 |
| P5 `Sequential::train_step_resident` | `crates/facade/src/compat/sequential.rs`（`predict_resident_input` の直後） |
| P6 `ResidentLoss`・`ResidentLoss::read` | `crates/facade/src/lib.rs`（`ResidentTensor` の直後。`facade` 内で定義する型） |
| P7 `Tape::wait_device_param_store` | `crates/facade/src/lib.rs`（`impl Tape`。1 行の委譲） |
| P7 の部品 `BackendOps::synchronize_device` | `crates/tensor-core/src/backend_ops.rs`（既定は `Unsupported`）。CPU は `Ok(())`、Metal は共有コンテキストの `synchronize`、CUDA は `with_sync_point_call` 内で default stream を `synchronize`（`unsafe` 追加なし）。ROCm は既定のまま |
| 内部入口（autodiff。生の `&Tape` を取るため利用者から到達不可） | `crates/autodiff/src/optim/device_store/train_chain.rs`: `validate_train_step_device_chain`・`download_resident_input`・`record_host_train_step_loss`・`wait_device` |

### 設計判断

- **tape を内部で作る**: P5・P6 は `Tape` を引数に取らない（§4 の署名どおり）。`predict_resident_input` の
  前例に倣い `tape_for(store.device())` を内部で作る。作り直しても順序が保たれる根拠: CUDA は
  `CudaBackendOps::new(ordinal)` が `context_cache::cached_device(ordinal)`（プロセス内キャッシュ）経由で
  同じ context・default stream を共有し、Metal は `MetalBackendOps` が ZST で積んだ仕事を
  `context_cache::cached_context()` のプロセス全体のバッチに持つ。`ResidentInput`・`TrainStepLoss` は
  `store_id`（＋世代）に束縛され tape には束縛されない（§4「ハンドルの寿命」）。
  固定費として、Metal の `tape_for` は呼び出しごとに存在確認（#2114）の費用がかかる（同期回数とは無関係）。
- **範囲外の 2 段の判別（§5）**: 第 1 段（モデル・損失・最適化が範囲外）は**フォールバックせず**、
  `store` を変更する前に型付きエラーを返す。判別を 1 か所にまとめるため、チェーンの encode 前検査を
  autodiff の private `precheck_train_chain` に切り出し、`train_step_device_chain` と
  副作用のない `validate_train_step_device_chain` が共有する（検査を二重に書かない）。第 2 段
  （バックエンドが部品を持たない）は、**事前検証を通った後**にチェーンが `Unsupported` を返した場合だけ
  step 全体を Tape 経路（`x`・`y` を download → `forward_resident` → `mse_loss` →
  `backward_device_param_store` → `step_device_param_store`）で実行する。§11 の失敗境界により、最初の dW の前の
  `Unsupported` は store を poison せず状態を変えない。`Unsupported` 以外のエラー（書き込み後の失敗・poison・
  capture 衝突）はフォールバックせずそのまま返す。この経路は D2H を伴う（§5 が許す）。
- **フォールバックの loss ハンドルも 1 型**: `ResidentLoss`・`read` は 1 本のまま保つため、フォールバックの
  loss（ホスト `f32`）は store の `train_chain_host_loss` に置き、世代を進めて新しい `TrainStepLoss` を返す。
  `read_train_step_loss` は host loss が `Some` なら同期なしで返す（poison・失敗トークンは従来どおり検査）。
  チェーンが成功した step の確定時に host loss を `None` へ戻す。古いハンドルは両経路の後で `InvalidArgument`。
- **`synchronize_device`**: `BackendOps` に既定実装つき（`Unsupported`）で足す非破壊追加（#3217 記録 §3 と同型）。
  `wait_device_capture_idle`（capture 専用）とは別物。`DeviceParamStore::wait_device` は poison・デバイス検査 →
  待機 → 失敗トークン再検査（セット済みなら poison して `StorePoisoned`）。待機の `Unsupported` は store を変えず
  返し、実行時エラーは poison する。
- **`!Send`**: `ResidentLoss` は `PhantomData<*const ()>` を持つ newtype で `!Send` を固定する（`unsafe` なし。
  内部の `TrainStepLoss` は整数だけで自動では `Send` になるため）。`assert_send` は書かず、`ResidentTensor`・
  `ResidentLoss` の `compile_fail` doctest で固定する。
- **`ResidentLoss::read` が `&mut DeviceParamStore` を取る**のは §4 の署名どおり（内部の読み出しは `&self`）。

### 追加したガード

- `crates/facade/tests/api_surface.rs`: `facade_resident_train_step_surface_matches_approved_shape`
  （P2〜P7 のシグネチャ完全一致・置き場所〈`impl Tape`／`impl Sequential`／`impl ResidentLoss` 直下に各 1 件〉・他ファイルに同名 `fn` が無いこと、
  `ResidentTensor`＝`{device, shape}`・`ResidentLoss`＝`{read}`・タプルフィールド非公開・属性なし・trait impl は `Debug` のみ、
  `ResidentInput`／`TrainStepLoss`／`resident_input` の `pub use` が無いこと、利用例 doctest と `!Send` の `compile_fail` doctest の実在）、
  `facade_resident_train_step_signatures_are_pinned_by_fn_pointer_probes`（関数ポインタ型での署名固定）、
  自己テスト `facade_resident_train_step_surface_detects_each_category`。保留ガード（`*HoldDoctestGuard`）は
  常駐学習ステップ用のものが存在しなかったため反転対象なし（新設もしない）。
- doctest: `ResidentLoss`（P1・P2・P5・P6・P7 の利用例）・`Tape::matmul_resident`（P3・P4）・
  `Sequential::train_step_resident`・`compile_fail` 2 件。

### 検証

- CI で実行（すべて CPU）: `cargo test -p fandhe-ai-tensor-core`（`synchronize_device` の既定は `Unsupported`）・
  `-p fandhe-ai-backend-cpu`・`-p fandhe-ai-autodiff train_chain`（24 件。既存 17 件は変更なし）・
  `-p fandhe-ai --lib train_step_chain`（付け替えた公開入口越しの bit 一致〈`[784,256,10]`・momentum／nesterov・3 層・端数形状・batch 1〉、
  第 1 段の拒否で `Unsupported` 注入のチェーンが呼ばれないこと、第 2 段フォールバックが Tape 経路と bit 一致すること、
  `Unsupported` 以外はフォールバックしないこと）・`--test resident_train_step`（公開 API だけの結合テスト）・
  `--test api_surface`・`--doc`。
- 実機: Metal は**開発機（M4 Max）で `#[ignore]` テストを通過確認**（公開入口の bit 一致・1 step の wait 1 回・
  `wait_device_param_store` の wait 1 回）。**Mac mini での受け入れ記録の再取得は未実施**。
  **GB10（CUDA）は未実測**（`#[ignore]` テストは追加済み。窓内の `stream_syncs`／`d2h_calls` の数値は未取得）。 → §16 で GB10 を実測済み。
  性能（ms）は受け入れ条件外のため計測していない。

### 範囲外・申し送り

- `scripts/bench/framework-compare/bench-fandhe` の `run_train_reuse` の P7 化: crates.io の `fandhe-ai` を新しい版へ
  上げた後（ピン更新は所有者承認）。
- step 全体の CUDA Graph capture・常駐入力の capture 対応: #3258。
- Metal の `_into` 版部品（確保ゼロ化）: §12 の申し送りのまま。
- `ResidentInput` を永続バッファへ `upload_into` で更新する入口（§12 で #3254 へ申し送り）: 新しい公開形を伴い §4 の
  P1〜P7 に無いため本 PR では足さない。**公開形の追加は決定記録 → 所有者確認の 2 段が必要**。

## §15 #3255 テスト記録（公開 API だけでの Tape 経路との bit 一致）

`crates/facade/tests/resident_train_step.rs` に追記した（結合テストのバイナリを増やさず、常駐学習ステップの
公開 API テストを 1 か所にまとめる）。本番コード・公開 API・tolerance・baseline は変更していない。比較は bit 一致
（期待値が NaN のときだけ NaN クラス一致）で、tolerance 定数は参照しない。

### 追加したテストと受け入れ基準

| テスト | 内容 | 基準 |
|---|---|---|
| `public_api_chain_matches_tape_path_bit_exact_{two_layers,three_layers,odd_shapes_and_batch_one}`（CPU） | 決定的シード（`Xorshift64Star`）で複数 step の loss 列と全パラメータが Tape 経路と bit 一致（plain・momentum・nesterov・端数形状・batch 1） | 1 |
| `public_api_chain_unread_steps_then_final_params_match_tape_path`（CPU） | 途中の loss を読まず積み、最後のハンドルだけ読んで bit 一致。途中の古いハンドルは `InvalidArgument` | 1 |
| `public_api_out_of_scope_models_are_unsupported_without_fallback` ほか `public_api_*_rejected*`・`shape_mismatch` | Tanh・最終層 ReLU・出力幅 0・空バッチは `Unsupported`（フォールバックせず params 不変）。不正 `SgdConfig`・momentum 途中変更・RMSProp 使用後・未消費 forward・形状不一致は型付きエラーで params 不変 | 2 |
| `resident_tensor_from_another_tape_instance_is_accepted_and_bit_identical`・`only_y_of_another_store_is_invalid_argument_and_keeps_store` | ハンドルは store_id＋device に束縛され tape インスタンスには束縛されない（§4）。片方だけ別 store なら `InvalidArgument` | 4 |
| `{metal,cuda}_public_api_*`・`cuda_tf32_mode_chain_matches_tape_path_bit_exact`（`#[ignore]`） | 実機での同じ bit 一致比較（784 幅・batch 64） | 3 |
| `{metal,cuda}_upload_resident_with_cpu_tape_is_device_mismatch`（`#[ignore]`） | デバイス store を CPU tape で扱うと `DeviceMismatch`、デバイス入力を CPU store へ渡すと `InvalidArgument` | 4 |

### 分担と制約

- 注入による第 2 段フォールバックの bit 一致は非公開の `train_step_resident_with` が要るためクレート内テスト
  （`train_step_chain_tests_3251.rs`）が担う。公開面の CPU テストは「第 1 段の拒否はフォールバックしない」ことだけを固定する。
- CPU の tape はすべて同一デバイスのため、公開面で `TapeMismatch` は作れない（leaf は `train_step_resident` 内部の tape で作られる）。
  別デバイスの tape は `#[ignore]` の実機テストで扱う。
- Conv を含むモデルは `init_device_param_store` の段階で拒否されるため `train_step_resident` まで届かず、本テストの対象外。
- CUDA の TF32 テストは、モードでも Tape 経路と bit 一致することだけを固定し、第 2 段フォールバックに入ることは主張しない
  （未確認のため。`backend-cuda` の部品が TF32 で `Unsupported` を返すかは #3256 の実機実行時に確認する）。

### 検証

CPU テストは CI で全件 pass。Metal の `#[ignore]` 4 件は開発機（M4 Max）で参考として pass したが、受け入れ記録（Mac mini）
ではない。GB10（CUDA）・Mac mini（Metal）での実行・実測は #3256。

## §16 #3256 実機記録（常駐チェーンの parity と学習 1 step の同期回数）

実測・ログ・事前登録規則は `docs/perf/logs/resident-chain-parity-3256/`（`RULE.txt`・`README.md`・`macmini/`・`gb10/`）。
本番コード・公開 API・tolerance・baseline・既存 assert は変更していない。追加したのは backend-cuda の `#[ignore]` テスト 2 件
（`train_step_chain_diag_tests_3252.rs`）だけ。

### 結果（各行 5 run・全 pass・invalid 0）

| 項目 | 結果 |
|---|---|
| Metal（Mac mini・M4）公開 API の Tape 経路との bit 一致（#3255 の 4 件） | 5/5 pass |
| CUDA（GB10）公開 API の Tape 経路との bit 一致（#3255 の 4 件＋TF32 の 1 件） | 5/5 pass |
| Metal 公開入口の同期 | encode 10・チェーン区間 wait 0・読み出しで 1（従来 Tape 経路は 7。§14 の「Mac mini での受け入れ記録は未取得」を解消） |
| CUDA 公開入口と同じ呼び出し列の同期 | 窓の `stream_syncs`／`d2h_calls`／`h2d_calls`／`mem_uploads`／`mem_allocs` が 0、読み出しで `stream_syncs` 1・`d2h_calls` 1（§14 の「GB10 未実測」を解消） |
| CUDA TF32 モードのチェーン | `Ok`（第 2 段フォールバックに入らない）。#3255 の申し送り（§15）を閉じる |

### 測り方と監査範囲

- CUDA の診断カウンタは facade から届かない（`internal-diagnostics` を facade の dev-dependency にすると CI の
  `cargo test --workspace` の対象が増える）。そのため facade の `resolve_ops`・`tape_for`・`train_step_resident_with`・
  `ResidentLoss::read` と同じ呼び出し列を backend-cuda 内で再現して数えた。facade 固有のホスト処理は純計算でデバイス呼び出しを
  持たないため監査で補った。facade から直接数えた値ではない。
- P7 の CUDA 同期（`synchronize_device`）は `stream_syncs` の対象外。機能確認のみで回数は未記録。
- 残る同期点は無い（窓 0・読み出し 1）。TF32 でチェーンが FP32 厳密のまま走る理由の掘り下げは本記録の対象外。

## §17 #3257 計測基盤と実測の状態（参考系列）

- 目的: 0.11.0 時点の差（Mac mini の train Metal 2.133 ms 対 PyTorch MPS 0.689 ms ほか）に対し、main HEAD の常駐チェーン（`train_step_resident`＋`ResidentLoss::read`、`upload_resident`＋`predict_resident_input`）がどこまで縮めるかを、現行経路・candle・PyTorch と同じ round で並べて測る。§14 のとおり bench の P7 化のコミットはピン更新後（#3273）なので、`bench-resident-chain.patch` を runner ツリーにだけ当てる計測専用方式にした。
- 収録済み: 事前登録規則・計測専用パッチ・ランナー `run_ab_resident_chain.sh`・判定 `judge_resident_chain_ab.py`（単体テストは CI）。置き場は `docs/perf/logs/resident-chain-ab-3257/`。
- ローカル煙試験（M4 Max・CPU。記録に使わない）: train の loss checksum と PARAM_DIGEST、infer の checksum が reuse と chain で一致した。
- 実機計測（Mac mini・GB10）は 2026-10-24 に実施済み。結果は §17.1〜§17.4、生ログと判定出力は `docs/perf/logs/resident-chain-ab-3257/{macmini,gb10}/`・`judge-3257*.md`。事前表明した仮説は RULE.txt にある。
- 限界: 公開面からはチェーンが Tape 経路へフォールバックしたかを観測できない（根拠は §16）。infer の after でも出力の D2H と同期 1 回は残る。
- 正式系列は #3273。

### 17.1 実測結果（5 round 中央値・判定は RULE 適用）

- 計測: 2026-10-24（UTC 04:29〜04:36）。runner は `786d18b50`（#3584。以降の main の差分は skills 更新のみで `crates/`・`scripts/bench/framework-compare/` に変更なし）、`.rev-stamp` `dirty=0`、計測専用パッチ適用。機体ロックは `#3257` が取得（owner 時刻 04:28:52Z）。torch は両機体とも 2.14.0（GB10 は 2.14.0+cu130）。size=64、TF32 OFF、`--graph` なし。
- 判定: `judge-3257.md`（`--require-both`）の終了コードは 0（全 8 セル判定可能・INVALID／REJECT なし）。機体別の `judge-3257-macmini.md`・`judge-3257-gb10.md` も終了コード 0。
- checksum: 全 run・両腕（before=`--mode reuse`・after=`--mode chain`）で一致（train は loss 0.080541 と PARAM_DIGEST、infer は 13.976574。judge が fail-closed に検査し通過）。tolerance・baseline は変更していない。
- ratio_ab = after÷before（小さいほど改善）。ratio_r は 5 round 個別の比。

| 機体 | task | device | before ms | after ms | ratio_ab | 判定 |
|---|---|---|---:|---:|---:|---|
| Mac mini | train | cpu | 0.4991 | 0.5009 | 1.004 | REGRESS |
| Mac mini | infer | cpu | 0.2050 | 0.2010 | 0.980 | NEUTRAL |
| Mac mini | train | metal | 1.5679 | 0.5047 | 0.322 | IMPROVE |
| Mac mini | infer | metal | 0.3331 | 0.3279 | 0.984 | NEUTRAL |
| GB10 | train | cpu | 0.7231 | 0.7659 | 1.059 | REGRESS |
| GB10 | infer | cpu | 0.1911 | 0.1620 | 0.848 | IMPROVE |
| GB10 | train | cuda | 0.2765 | 0.1343 | 0.486 | IMPROVE |
| GB10 | infer | cuda | 0.0899 | 0.0776 | 0.863 | IMPROVE |

- REGRESS（CPU train 2 セル）は RULE の厳格条件（ratio>1.00）による。Mac mini の +0.4% は ratio_r が 0.948〜1.018 に散り、誤差範囲に近い。GB10 の +5.9% は ratio_r が 0.893〜1.102 に散る。いずれも事後に規則を緩めず、そのまま記録する。CPU はデバイス同期が無いので、チェーン化は RULE の仮説どおり「中立に近い」見込みだった。実測は GB10 infer CPU の 0.848（IMPROVE）が仮説外で、原因は未調査。
- Mac mini の infer Metal は ratio_r が 0.776〜1.075 と散る（before の q1〜q3 幅も大きい）。NEUTRAL は「改善とは言えない」の意味で、後退でもない。

### 17.2 相手との比（5 round 中央値。参考。判定外）

- 相手最速÷x は judge の列そのもの。1 超は fandhe-ai が相手最速より速い。「1 超で 1 位・0.90 以上で僅差」は参考の目安で、判定には使わない。candle・torch の列は各 JSONL の 5 round 中央値（`results-{candle,torch}-3257-*`）。

| 機体 | task | device | before | after | candle | torch | 相手最速÷before | 相手最速÷after |
|---|---|---|---:|---:|---:|---:|---:|---:|
| Mac mini | train | cpu | 0.4991 | 0.5009 | 0.6248 | 0.1368 | 0.27 | 0.27 |
| Mac mini | infer | cpu | 0.2050 | 0.2010 | 0.1595 | 0.0317 | 0.15 | 0.16 |
| Mac mini | train | metal | 1.5679 | 0.5047 | 1.1195 | 0.4904 | 0.31 | 0.97 |
| Mac mini | infer | metal | 0.3331 | 0.3279 | 0.3258 | 0.3210 | 0.96 | 0.98 |
| GB10 | train | cpu | 0.7231 | 0.7659 | 2.3249 | 0.3734 | 0.52 | 0.49 |
| GB10 | infer | cpu | 0.1911 | 0.1620 | 0.2488 | 0.1382 | 0.72 | 0.85 |
| GB10 | train | cuda | 0.2765 | 0.1343 | 0.2710 | 1.0266 | 0.98 | 2.02 |
| GB10 | infer | cuda | 0.0899 | 0.0776 | 0.0414 | 0.0396 | 0.44 | 0.51 |

- 単位は ms。参考として 0.11.0 正式系列の値は Mac mini の train Metal 2.133 ms 対 PyTorch 0.689 ms（`docs/perf/logs/framework-compare-0.11.0-macmini-remeasure/`）、GB10 の train CUDA 0.331 ms 対 candle 0.272 ms（`docs/perf/logs/framework-compare-0.11.0-remeasure/gb10/`）。本計測の before は HEAD の reuse 経路で、版も runner も異なるため 0.11.0 の値とは直接比べない。
- 相手とは初期値・入力が非対称で checksum は一致しない（candle・torch の checksum は値が異なる）。本節の比は時間の比で、数値一致の主張ではない。入力の対称化は #3263・#3266 の範囲。
- 推定との比較: RULE の事前仮説（推定）は train CUDA 0.18〜0.25 ms・train Metal 0.65〜0.9 ms。実測は 0.134 ms・0.505 ms で、いずれも仮説の範囲より速かった。仮説は保守的すぎた。Metal は PyTorch MPS と同程度で明確には勝てない、という仮説は実測と整合する（0.97 倍）。infer の改善は小さい、という仮説は Metal（−1.6%）で整合し、CUDA（−13.7%）はやや大きかった。

### 17.3 届かないセルの残りの差の分解

対象は after が相手最速に届かない 6 セル。枠組みは `docs/perf/loss-attribution-matrix.md` §10（ハードの上限／共通の計測窓／自分たちの制約）。**実測**は本計測で直接得た値（Δ・中央値）だけ。内訳の各項は 0.11.0 の registry 版で別途実測した固定費（`docs/perf/logs/cuda-train-infer-gb10-3189/gb10/aggregate.md`）からの**推定**であり、main HEAD の値ではない。run の追加は RULE で禁止されているため、内訳そのものの実測は追加していない。

| セル | Δ = after − 相手最速（実測） | 内訳 | 分類 |
|---|---:|---|---|
| Mac mini train metal | +0.0143 ms（+2.9%、vs torch） | 分解不能。Mac mini の train／infer の内訳は未実測。M4 Max の旧内訳（`train-step-phase-breakdown.md`）は機体が違うため流用しない | 未説明残差（Mac mini の内訳は未実測） |
| Mac mini infer metal | +0.0069 ms（+2.2%、vs torch） | 同上。ratio_r が 0.776〜1.075 に散り、Δ は round 間の揺れと同程度 | 未説明残差。差は誤差範囲に近い |
| GB10 infer cuda | +0.0380 ms（vs torch） | 推定: launch の発行 14.1 µs（L1 7.5＋L2 6.6。相手も launch するので全額が Δ ではなく上限）、同期の待ち固定費 1 回 約 9.7 µs（kernel_wait。相手も同期 1 回を負うので Δ に入らない見込み）、checksum 窓 0.6〜0.8 µs（両者共通で Δ へほぼ寄与しない）。残りの約 23.9 µs（Δ から launch 発行を引いた値）はカーネル時間差と launch 数差を含み、分離できない | 計測窓（同期・checksum は共通の窓）＋自分たちの制約（launch 回数、推定）。未説明残差 約 23.9 µs（推定） |
| GB10 train cpu | +0.3925 ms（vs torch） | 同期が無いので wait 固定費はゼロ。CPU の GEMM・autodiff の演算時間の差で、本計測に内訳なし | 未説明残差。内訳は未実測（チェーン化の対象外） |
| GB10 infer cpu | +0.0238 ms（vs torch） | 同上 | 未説明残差。内訳は未実測 |
| Mac mini train／infer cpu | +0.3641 ms／+0.1693 ms（vs torch） | 同上 | 未説明残差。内訳は未実測（チェーン化の対象外） |

- GB10 train cuda は after が相手最速（candle）より 2.02 倍速いので分解の対象外。
- 数字で振り分けられたのは GB10 infer cuda の推定分だけで、それ以外は「内訳が未実測」として残した。ハードの上限／計測窓／自分たちの制約への割り当ては、内訳の実測（Mac mini の phases 腕、GB10 の HEAD での再分解）が揃うまで確定しない。制約の緩和は提案しない。

### 17.4 限界・未実測

- 公開面からはチェーンが Tape 経路へフォールバックしたかを観測できない（§16）。infer の after でも出力の D2H と同期 1 回は残る。checksum 一致は bit 同一の証明ではなく、train は PARAM_DIGEST（chain 腕と reuse 腕で一致）で補っている。
- 機体ロック取得前後の GPU 占有（`nvidia-smi`）と計測中の他プロセスの記録は取れていない（計測時のログに無い）。Mac mini の load average は計測中 2.6〜3.3 で、ランナーの負荷ゲートは通過している。
- Mac mini の train／infer の内訳は §17.6（#3612。計測部品は登録済み・実機は未実測）、GB10 の HEAD での再分解は未実測（起票は所有者承認待ち）。
- 本計測は参考系列で、正式な 6 フレームワーク同席計測とピン更新（0.12.0）は #3273。

### 17.5 #3611 CPU train の REGRESS の切り分け（手順の登録。状態: 未実測）

§17.1 の CPU train の REGRESS（Mac mini ratio_ab 1.004・GB10 1.059）が、ノイズ幅・配置・チェーン化の実際の後退のどれかを、計測より前に固定した手順で分類する。

- 手順の正本: `docs/perf/logs/resident-chain-cpu-train-diag-3611/DIAG.txt`（計測前に固定。#3257 の RULE と判定は変えない。分類であり判定ではない）。ランナーは `scripts/bench/framework-compare/run_diag_resident_chain_cpu.sh`、集計は `summarize_diag_resident_chain_cpu.py`。
- 腕は 7 つ（A/A 2・B1 の chain・配置をずらした B2 の reuse／chain・#3257 時点のコード B0 の reuse／chain）。#3257 の before／after は同一バイナリだったため、配置の対照は「配置をずらした再ビルドでも差が同じ向きで残るか」で見る。
- ツリーは HEAD の main ではなく #3622 の `a8e91f0c9`（版 0.11.0 のまま `crates/` が最新の最後のコミット）を使う。#3623 の 0.12.0 版上げ以降は bench-fandhe の `fandhe-ai =0.11.0` ピンが path patch に解決されず、#3273 のピン更新まで HEAD は計測できないため。`.rs` は main と同一。
- 結果（分類・数値・ログのパス）と所有者判断事項は計測後にここへ追記する。両機体とも現時点では未実測。

### 17.6 #3612 Mac mini Metal 常駐チェーンの内訳（手順の登録。状態: 未実測）

§17.3 の Mac mini train metal（+14.3 us）・infer metal（+6.9 us）の未説明残差を、HEAD の常駐チェーン経路の区間別時間で `docs/perf/loss-attribution-matrix.md` §10 の枠組みへ振り分ける。

- 手順の正本: `docs/perf/logs/resident-chain-metal-breakdown-3612/RULE.txt`（計測前に固定）。診断は `crates/backend-metal/src/resident_chain_breakdown_diag_tests_3612.rs`（`cfg(test)`・`#[ignore]`）、ランナーは `scripts/bench/framework-compare/run_breakdown_resident_chain_metal.sh`、集計は `summarize_breakdown_resident_chain_metal.py`。
- 腕: B0（本番の同期位置。定常で wait が 1 step あたり 1 であることを assert し、Tape 経路へのフォールバックの検出を兼ねる）と B1（encode 直後に GPU タイムスタンプ付きで同期し、`gpu_busy`・`host_fixed`・`readout` を分ける）。全体時間の正は A0（#3257 と同じ窓の bench-fandhe）。
- 限界: 常駐チェーンは 1 コマンドバッファのため、HEAD の経路では forward／backward／SGD の GPU 時間は分けられない（B2 は未実装）。torch 側の内訳は未測定なので、|Δ| がノイズ幅を超えても 1 つのタグへは確定させない。
- 結果と所有者判断事項は計測後にここへ追記する。§17.3 の 2 行は実測まで書き換えない。
