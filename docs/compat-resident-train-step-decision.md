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
| 公開状態 | 未公開（本記録は公開形の確定のみ。実装は #3250・#3251〜#3253・公開は #3254） |
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
- **#3251／#3252／#3253**: P5 の内部チェーン（`_tracked` で store の failure_token へ登録、
  L1 の d_input は計算しない）と完了待ちの内部メソッド。
- **#3254**: facade 公開の仕上げ。`crates/facade/tests/api_surface.rs` に正ガード（追加した名前の
  `pub use` 行・facade の inherent `pub fn` 集合・関数ポインタ型による署名固定プローブ・
  利用例 doctest の実在検査）を足す。内部実装が先に入る期間に facade へ早期に漏れないよう、
  必要なら `*HoldDoctestGuard`（`crates/facade/src/lib.rs` の既存パターン）で保留を固定し、
  #3254 で正ガードへ反転する。`!Send` を前提とし `assert_send` を足さない。
  `docs/compat-api-scope.md` §5 の適用記録・`docs/public-api-design.md`・`docs/README.md` を更新する。
- **#3258**: P5 の固定構造（永続ワークスペース・永続 loss スカラー）を capture 区間の前提として
  参照する。

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
