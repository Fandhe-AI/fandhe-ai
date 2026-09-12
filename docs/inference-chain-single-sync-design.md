# GPU 推論チェーン単一同期化（`linear_forward_device` の facade 結線）の設計（イシュー #1579）

- 対応イシュー: #1579（親 #1571「低レイヤー診断で見つかった性能課題を解消する」Phase 1 → ルート #1570）
- 位置づけ: 本文書は**設計判断のみ**を記録する。`crates/` 配下・`docs/spec/`（正本 submodule）へのコード変更は行わない。実装は兄弟イシュー #1580（Metal）・#1581（CUDA）へ引き渡す
- 基準コミット: `origin/main` `f713113f`。行番号・事実はすべて本コミットで再確認した

## 1. 背景

- GB10（`fandhe-ai =0.8.0` ピン系列）実測で infer cuda は candle 比 3.4 倍遅く（0.141 ms 対 0.043 ms）、train cuda は 1.71 倍遅い（`docs/perf/lowlayer-diagnosis-2026-09-12.md` §2）。推論 forward は 2 層で 132 µs（内訳は未分解。「層境界ごとのホスト往復が支配的」はコード読解による帰属であり実測分解ではない）。M4 Max も `infer/metal/64` 695 µs 対 candle 216 µs（#1580 本文）
- 原因（コード事実）: `Sequential::predict_resident` が `BackendOps::gemm_resident_rhs_act`（`a` をホスト `Tensor` で受け取り `y` をホスト `Tensor` で返す）を層ごとに呼ぶため、層境界ごとに H2D（`a` upload）＋ D2H（`y` download＝同期点）が発生する
- 既存資産: #1216 で `BackendOps::linear_forward_device`（`a`／`w`／`bias`／戻り値すべて `DeviceBuffer` 常駐）を CPU／CUDA／Metal に実装済み。**facade／autodiff への結線（Phase 2）のみ未実施**（`docs/inference-forward-fixed-cost-design.md` §3.2・§4、`docs/perf/linear-forward-device-gpu.md` §5、`docs/perf/infer-reuse-phase-breakdown.md` §6）
- 目的: 層境界のホスト `Tensor` 実体化を排し、チェーン末尾で 1 回だけ同期する構造を CUDA／Metal 共通の設計として確定し、(a) Metal の encode-only 化と failure_token（`*_tracked`）契約、(b) CUDA のストリーム順序契約との整合、(c) bit 同一契約、(d) `Sequential::predict_resident` との関係を記録する。数値一致は run-to-run bit 同一を契約とする。tolerance・baseline は変更しない

## 2. 現状のコード事実

| 事実 | 出典 |
|---|---|
| `Tape::ops()` は `pub(crate)`。`DeviceParamStore::checked_resident_buffer(store_id, slot) -> DeviceBufferView`（private）も同様。facade から `ops`／weight の `DeviceBufferView` に直接は到達できない | `crates/autodiff/src/tape.rs:755`・`crates/autodiff/src/optim/device_store.rs:1531` |
| `TapeNode::value` は `OnceCell<Tensor<f32>>`（ホスト値）。`Tape` は `Send` 必須（`tests/fusion_backend_integration.rs::tape_is_send`）で `DeviceBuffer` を `TapeNode` へ持たせられない → デバイス常駐活性化を `Var` として表現できない | `crates/autodiff/src/tape.rs:524-527`・`crates/autodiff/tests/fusion_backend_integration.rs:390` |
| `DeviceParamStore` は `device`／`store_id`／`layout: Vec<ParamLayout>`／`params: DeviceBuffer<f32>`（全パラメータ連結）／`poisoned: AtomicBool`／`failure_token: DispatchFailureCell` を保持。`ops` は保持せず各メソッドが `&Tape` 経由で `tape.ops()` を使う。`check_not_poisoned()` は `poisoned` を見た後 `failure_token.is_set()` を検査し、set 済みなら `poisoned` へ自己遷移してから拒否する | `device_store.rs:309-360`・`1148-1163` |
| `predict_resident` は `crate::tape_for(store.device())` で `Tape` を構築 → `snapshot_resident_params` → `tape.var(input)` → `forward_from_flat_leaves` → `Var::to_tensor()`。`forward_from_flat_leaves` は `Linear` 層で次層が `ReLU` なら `Activation::Relu` に融合し `i += 2`、それ以外（`Sigmoid`／`Tanh`）は `layer.forward(tape, ..)`。leaves 過少・過剰は fail-closed で拒否する | `crates/facade/src/compat/sequential.rs:543-640` |
| `Sequential::predict`（CPU 固定）は `predict_tape_free` → `Unsupported` なら `predict_via_tape` へ全体フォールバック（部分実行なし） | `sequential.rs:221-300`・`docs/inference-forward-fixed-cost-design.md` §3.1 |
| `BackendOps::linear_forward_device` の既定は `Unsupported`（fail-safe）。呼び出し元は `Unsupported` 検出時に層構成全体を per-op 経路へフォールバックする契約とドキュメントに明記済み | `crates/tensor-core/src/backend_ops.rs:1030-1075`（`# デフォルト実装` 節） |
| **Metal `linear_forward_device` は `encode_strided_bias_act_prepared`（`encode_strided_bias_act_prepared_impl(.., token: None, ..)`。`ctx.synchronize()` を呼ばない encode-only）を呼ぶ**。`_tracked` 版は存在しない | `crates/backend-metal/src/ops.rs:1349-1508`・`crates/backend-metal/src/gemm.rs:3209-3232`（呼び出しが `None` を渡す） |
| Metal `encode_strided_bias_act_prepared_with_c_offset`（`pub(crate)`）は `token: Option<&DispatchFailureCell>` を受け取り `MetalContext::encode` と同一ロック区間でバッチへ登録する（#1555／PR #1556 で `gemm_fp32_strict_into_tracked` が利用） | `gemm.rs:3262-3300`・`context.rs:425-470` |
| Metal `synchronize_observed` は `committed` バッチを `take` して `waitUntilCompleted` → `status == Error` なら登録済みトークン全てへ `propagate_failure` し `first_error` を返す。**バッチはこの直後に drop される**ため、別スレッドが先に `synchronize()` してエラーを回収した場合、自スレッドの後続 `download`（＝`synchronize`）は `Ok` を返しうる（自スレッドが待った時点でバッチが既に take 済みなら `committed` は空）。トークン未登録の encode-only 経路は、他スレッド回収時にも失敗が伝わらない fail-closed 契約違反になる（memory `metal-encode-only-failure-token`・codex P0 指摘 PR #1556） | `context.rs:571-640` |
| 既存の `*_tracked` パターン: `sgd_step_device_tracked`（`backend_ops.rs:365`）・`gemm_fp32_strict_into_tracked`（`:611`）・`gemm_fp32_strict_into_with_bias_reduce_tracked`（`:692`）。既定実装は `token` を無視して非 tracked 版へ委譲。CPU／CUDA は上書きせず Metal のみ上書きする（`crates/backend-metal/src/ops.rs:663,847,894`） | `backend_ops.rs:322-372` |
| **既存の「事後再検査」先例**: `DeviceParamStore::sync_to_host` は `download_split`（実データ転送・同期点）の**直後にもう一度** `check_not_poisoned()` を呼ぶ（コメント「`register_resident_params`／`snapshot_resident_params` と同じレース〈Cursor Bugbot 指摘・PR #1057〉への対処」）。決定 4 の「`download` 復帰後に `failure_token` を再検査」はこの既存パターンの踏襲である | `device_store.rs:2274-2288` |
| CUDA `linear_forward_device` は `begin_driver_call(ordinal, &resident_generations)`（`a`・`w`・`bias` の世代検査）→ `launch_tiled_bias_act_f32_resident` → **`download` しない**。コメント「同期点は呼び出し元の `download`（`readback` の `synchronize`）へ集約される。同一ストリーム FIFO により次層カーネルは前層の出力完了後に実行される」 | `crates/backend-cuda/src/ops.rs:1955-1959` |
| CUDA `MemoryOps::download` = `memcpy_dtoh`（`cuMemcpyDtoHAsync`）→ `stream.synchronize()`（唯一のホストブロック同期点）。`upload`（`clone_htod`）は同期なし | `crates/backend-cuda/src/memory.rs:677-731,923-975`・`docs/backend-cuda-async-execution-design.md` §4 表 |
| Metal `upload_inner` は同期なし。**`upload_into` は `self.context.synchronize()` を挟む**（`memory.rs:481`。呼び分けを誤ると不要な同期点が増える）。`download_inner`／`with_host_view` は `synchronize()`。`alloc_zeroed` はプール経路で `zero_fill_logical`（ホスト書き込み）のみ・同期なし（#1099） | `crates/backend-metal/src/memory.rs:307-346,370-380,449-481,516-524`・`pool.rs:340-400` |
| CPU: `gemm_resident_rhs_act` → `gemm_resident_rhs_impl` は融合 `gemm_blis_bias_act_parallel`。`CpuBackendOps::linear_forward_device` は非融合合成（`gemm_blis_parallel` → bias 行方向加算 → `max(x,0)`）。両者の bit 一致は `crates/backend-cpu/tests/gemm_epilogue_parity.rs`（融合＝非融合）＋ `linear_forward_device_parity.rs`（非融合合成＝`linear_forward_device`）から推移的に成立 | `crates/backend-cpu/src/ops.rs:368-430,900-1007` |
| #1216 実機テスト: Metal `linear_forward_device_parity.rs`（`gemm_resident_rhs_act` との bit 一致・2 層チェーン）は M4 Max で pass 済み。**CUDA `linear_forward_device_real_device.rs` は未実測**のまま記入欄が残っている | `docs/perf/linear-forward-device-gpu.md` §3・§4 |
| `ResidentLeaf<'t>`（`pub`。フィールドは全 private: `node_id`／`store_id`／`slot`／`shape`／`tape_id`／`PhantomData`）は `store.snapshot_resident_params(&tape)` から発行され、`linear_forward` 系は `tape_id` を検査して異なる `Tape` 由来の葉混入を fail-closed に拒否する | `device_store.rs:272-284,1397` |
| `Tape::push_resident_leaf`（`pub(crate)`）が常駐葉ノードを tape へ登録する。値を持たないノードのため `predict_resident` の既存呼び出しでも既に N 個（leaves 数分）登録済み | `tape.rs:794` |
| facade 公開面契約: `crates/facade/tests/api_surface.rs` が `pub fn` の `BackendOps` 直接引数・`Tape`／`new_with_ops` の再エクスポートを禁止する。`DeviceParamStore`／`ResidentLeaf` は `fandhe_ai` から再エクスポートされる（`crates/facade/src/lib.rs:116` 付近）ため、`DeviceParamStore` への `pub fn` 追加は公開面の拡張になる | `crates/facade/src/lib.rs:101-126`・`api_surface.rs` |
| `Activation` は `#[non_exhaustive]` の `None`／`Relu` のみ。`Sigmoid`／`Tanh` はデバイス常駐チェーン非対応（段階 B の既知制約） | `backend_ops.rs:106-111`・`inference-forward-fixed-cost-design.md` §4 |
| 兄弟 issue: #1580（Metal 実装・M4 Max A/B）・#1581（CUDA 実装・GB10 A/B）・#1582（MSE backward の encode-only 化。同期の二重主張禁止）。いずれも本 #1579 に依存し、受入規則は「5 run 中央値・checksum 完全一致・非後退 `ratio<=1.00`・事後緩和禁止」 | `gh issue view 1580/1581/1582` |

## 3. 契約整理（設計が守るべき既存契約）

1. `MemoryOps::download`／`with_host_view` は「復帰時点でホストデータ確定」という全バックエンド共通の同期点（`crates/tensor-core/src/buffer.rs` の `MemoryOps` 契約）
2. CUDA: 単一ストリーム FIFO・launch は非同期・遅延エラーは次の同期点で `TransferFailed` 等として表面化し帰属は失われる・ordinal poison 状態機械（`docs/backend-cuda-async-execution-design.md` §3〜§5・§12）。CUDA Graph capture（#1349）は update 区間限定であり本チェーンは対象外
3. Metal: encode → flush → synchronize の 3 段・バッチ内エラーは登録トークン全てへ `set`・`DeviceParamStore` は自トークンを検査して自己 poison する（`docs/backend-metal-command-batching-design.md` §3.5〜§3.8）。encode-only 経路の新設は `*_tracked` 版で failure_token 登録が必須（PR #1556 codex P0 指摘）
4. `linear_forward_device` の `Unsupported` は層構成全体のフォールバック契約（部分実行なし）
5. `Sequential::forward_from_flat_leaves` の leaves 件数契約（層順 weight → bias・過少／過剰は fail-closed）
6. facade 公開面（`api_surface.rs`・`docs/compat-api-scope.md` §5）・REQ-12（`mem`／`ops` を表面に出さない）
7. 数値: REQ-2 複合判定・FMA 契約・tolerance／baseline は不変

## 4. 決定事項

### 決定 1: チェーンの置き場所

`autodiff` 側 `DeviceParamStore` に tape 不要の推論チェーンメソッドを新設し、facade `Sequential::predict_resident` は**公開シグネチャ不変**で内部を差し替える（#1028 の `predict` 差し替えと同型）。

根拠: `Tape::ops()` が `pub(crate)`・`checked_resident_buffer` が private で facade からは到達不能（§2）。`forward_resident`（学習 forward）は backward に tape ノードが必要なためスコープ外。

API 形は候補比較のうえ採用する:

- **(a) 新型 `LinearChainStep { weight_slot, bias_slot: Option<_>, act }` を `pub` で公開**し `predict_device_chain(&self, tape, input, plan: &[LinearChainStep])` で受ける案。tape ノード登録ゼロだが、生 slot 番号を持つ新規 `pub` 型が `fandhe_ai` 再エクスポート経由で公開面に載る（承認範囲が広い）
- **(b) 既存の不透明型 `ResidentLeaf` を組で受ける**案。`snapshot_resident_params(&tape)` から受け取った `&ResidentLeaf` の組（`(&ResidentLeaf, Option<&ResidentLeaf>, Activation)` 列。具体的な引数形は実装 issue で確定）を `predict_device_chain(&self, tape: &Tape, input: &Tensor<f32>, steps: &[..])` に渡す。新規 `pub` 型ゼロ・追加は `pub fn` 1 つ。代償は常駐葉ノード N 個（値なし・`push_resident_leaf`）が tape に登録されること（現行 `predict_resident` も同数登録しており増分ゼロ）

**採用: (b)**。公開面の増分が最小で、既存の `store_id`／`slot`／`tape_id` 検証をそのまま再利用でき、`forward_from_flat_leaves` の leaves 走査コードと同型に書ける。(a) は「tape ノード完全ゼロ」を優先する場合の代替として記録する。

facade 側: `Sequential` が `layers` と `snapshot_resident_params` の leaves から steps を構築する private ヘルパー（`Linear` → leaves 消費は `trainable_parameters()` と同一の順序契約・次層 `ReLU` を `Activation::Relu` へ融合し `i += 2`・過少／過剰 leaves は fail-closed）を持つ。`Sigmoid`／`Tanh` を含む層構成、または steps 構築失敗時は steps を作らず現行経路（`forward_from_flat_leaves`）へ全体フォールバックする。

### 決定 2: 活性化を `Var`（tape ノード）にしない

`TapeNode::value` はホスト `OnceCell<Tensor>`（`Send` 制約）のため常駐活性化は `Var` にできない（§2）。推論は入力・中間活性化・出力を `Var` にせず `DeviceBuffer` 列で連鎖する。これにより `tape.var(input)` の clone・層ごとの `push_eager`・`to_tensor()` の固定費が消える。決定 1 (b) 採用時に残るのは値なしの常駐葉ノード N 個のみ（現行と同数。`tape` は `check_device`・`ops`／`memory_ops` の運搬体として使う）。

### 決定 3: 同期点は `upload` 1 回 + `download` 1 回

チェーン内の同期点を実装箇所ごとに列挙し、他に同期点が無いことを示す:

- 入力: `MemoryOps::upload`（CUDA `clone_htod`・Metal `upload_inner` とも同期なし。**Metal `upload_into` は synchronize するため使わない**）
- 各層: `linear_forward_device`（`_tracked` 版。決定 4）。出力 `alloc_zeroed` は CUDA stream 非同期／Metal プール経路（`zero_fill_logical`）とも同期なし（#1099）。カーネルは launch／encode のみ
- 最終: `download`（CUDA `memcpy_dtoh`→`synchronize`／Metal `synchronize`→`contents` コピー）

中間出力 `DeviceBuffer` の `Drop` も同期点にならない。Metal はプール返却が `defer_pool_return` で in-flight 中は待たずに退避し、CUDA `CudaSlice::drop` はデバイス側 `stream.wait` のみで `has_async_alloc=true` は GB10 実測済み（`lowlayer-diagnosis` §2）。`has_async_alloc=false` 環境ではホスト同期へフォールバックしうるが、正しさ（数値契約）は不変と注記する。

Metal は `should_auto_flush` によりコマンドバッファが分割されうるため、不変条件は **「`wait_until_completed` が predict 1 回あたり 1 増える」** と定義する（`command_buffers == 1` は 2 層規模〈784→256→ReLU→10〉での期待値に留め、不変条件本体には含めない）。

### 決定 4: Metal failure_token 契約

`BackendOps::linear_forward_device_tracked(&self, a, w, bias, act, token: &DispatchFailureCell)` を、既定委譲（`token` 無視 → `linear_forward_device`）の非破壊拡張として追加し、Metal のみ上書きする。配線は既存の `pub(crate) encode_strided_bias_act_prepared_with_c_offset(.., c_offset = 0, .., token: Some(..))` へ流すだけで `gemm.rs` の新規入口は不要（`gemm_fp32_strict_into_tracked` と同型）。チェーンは `self.failure_token` を毎層渡す。

**加えて最終 `download` 復帰後に `failure_token.is_set()` を再検査し、set 済みなら `poisoned` へ遷移して `StorePoisoned` を返す**（`download` の結果を返さない）。根拠: `synchronize_observed` はバッチを take→drop するため、別スレッドの `synchronize` が先にエラーを回収すると自スレッドの `download` は `Ok` で不正データを返しうる（§2「Metal `synchronize_observed`」行）。トークン登録（他スレッド回収時にも set される）＋事後検査の組で fail-closed を成立させる。この「同期後にもう一度 poison 検査する」形は新規発明ではなく、既存の `DeviceParamStore::sync_to_host`（§2「既存の『事後再検査』先例」行）と同じパターンを踏襲する。入口では従来どおり `check_not_poisoned()`（既存 4 エントリと同じ）。

エラー時の状態遷移を 3 ケースで明記する:

1. `linear_forward_device(_tracked)` 自身が `Err`（起動時エラー・shape 検査）→ そのまま伝播し poison しない。パラメータは読み取りのみで書き換えていないため（`Unsupported` は決定 7 の全体フォールバックへ回る）
2. 最終 `download` が `Err`（自スレッドの `synchronize` がバッチエラーを直接返した／CUDA の `TransferFailed`）→ `sync_to_host` と同じ扱いに揃え `Err` をそのまま伝播する。Metal ではバッチ内の dispatch が同居していれば同一 `synchronize` が `self.failure_token` を set しているため、次回入口（および本チェーンの事後検査）で自己 poison する
3. `download` は `Ok` だが `failure_token.is_set()` → `poisoned` へ遷移し `StorePoisoned` を返す（上記）

`step()`（PR #1017 系）のように「`Err` を見た瞬間に即 poison」としない理由: 推論はパラメータを書き換えず、`sync_to_host` と同じ読み取り専用エントリであるため。

### 決定 5: CUDA はストリーム順序契約に従い `_tracked` を上書きしない

`sgd_step_device_tracked` と同じ判断。根拠:

- 同一ストリーム FIFO で前層出力→次層入力の順序が保証される（`ops.rs:1957` コメント・`docs/backend-cuda-async-execution-design.md` §3）
- `begin_driver_call` の世代検査に `a` を含む（中間出力は新規確保のため常に最新世代）
- 最終 `download` も `with_driver_call` 経由で ordinal poison に乗り、遅延エラーが `Err` として返る（§2「CUDA `MemoryOps::download`」行・`docs/backend-cuda-async-execution-design.md` §5 表「遅延（非同期）」行）

CUDA Graph capture（#1349）は update 区間限定のため本チェーンは対象外（契約整理 2）。

### 決定 6: 数値契約（3 層）

1. **run-to-run bit 同一**（Issue 契約）。決定的カーネル・固定順序 epilogue のため成立する
2. **新チェーン出力 ≡ 現行 `predict_resident` 出力の bit 同一**。同一の融合カーネルを再利用するため: Metal は #1216 (b) で M4 Max 実測済み、CUDA は同 (b) が**未実測**のため #1581 のゲートとして引き渡す。CPU は融合／非融合の推移的一致を根拠にしつつ**直接テストで確認**する
3. **CPU `predict` との比較は従来どおり REQ-2 複合判定**。tolerance 定数・`ParityBaseline`・`docs/spec/` は不変

### 決定 7: フォールバック fail-closed

plan 構築不能（`Sigmoid`／`Tanh` 混在・leaves 件数不一致）、または最初の `linear_forward_device(_tracked)` が `Unsupported` を返した場合は、チェーン全体を現行経路へ戻す（部分結果を使わない・`Unsupported` 以外のエラーは吸収せず伝播する）。shape 検査（rank・`w[0]==k`・bias `[n]` 厳密一致）は各バックエンド実装がカーネル前に行う既存契約に依存し、chain 側は slot 範囲・`store_id`・device 一致・leaves 件数を事前検査する。

### 決定 8: `Sequential::predict_resident` との関係

公開 API・戻り値・エラー variant は不変。内部が「tape 経由 per-op 経路」から「デバイス常駐チェーン→フォールバック」へ変わる。`predict`（CPU 固定・tape 不要経路）とは独立。#1582（MSE backward）とは同期区間を二重主張しない（推論 forward の同期のみを対象とする）。

### 決定 9: 計測・受入規則の事前登録（#1580／#1581 へ）

5 run 中央値・checksum 完全一致・非後退 `ratio<=1.00`・事後緩和禁止・内部ホスト名非記録。計器: `bench-fandhe --task infer --mode reuse --phases`（`predict_resident` 区間）・`crates/facade/tests/infer_fixed_cost_bench.rs` 相当の実機版・Metal `diagnostic_batch_counters()`。効果測定には crates.io ピン更新が別途必要（`docs/perf/infer-reuse-phase-breakdown.md` §6）で、それまでは path patch 系列で A/B を取る。

## 5. スコープ外

- `forward_resident`（学習 forward）のデバイス常駐化・backward への拡張
- `Sigmoid`／`Tanh` のデバイス常駐対応（`Activation` 拡張）・多層融合カーネル・CUDA Graph によるチェーン全体 capture
- `Sequential::predict`（CPU 固定経路）の変更
- MSE backward の encode-only 化（#1582）
- crates.io 公開・framework-compare 承認ピン更新

## 6. 承認事項（#1580／#1581 着手前の前提）

1. 公開クレート `fandhe-ai-tensor-core` の `BackendOps` trait への default メソッド追加（`linear_forward_device_tracked`）。非破壊（SemVer 互換）だが #1570 ツリー運用では trait 拡張は個別承認対象。先例: `gemm_fp32_strict_into_tracked`（PR #1556）・`gemm_fp32_strict_into_with_bias_reduce_tracked`（#1566）は同型の非破壊追加を codex-review 指摘対応として個別承認なしで行っている
2. `DeviceParamStore` の新 `pub fn`（決定 1 (b) 採用時は `pub fn` 1 つ・新規 `pub` 型なし。(a) へ変更する場合は `LinearChainStep` 型が加わる）は `fandhe_ai` 再エクスポート経由で公開面の拡張になる（`docs/compat-api-scope.md` §5 の範囲拡張手続き。`predict_resident` 自体のシグネチャは不変）。回避不能な理由（決定 1: `Tape::ops()` が `pub(crate)`）を添える
3. tolerance・baseline は変更なし（対象外の明記）。Metal 実装の bit 同一が実機で崩れた場合は判定方式を緩めず REJECT として記録する
4. 上記はいずれも起票・承認をユーザーが行う。本 Issue は Issue 起票・コメント投稿を行わない

## 7. #1580／#1581 への引き渡し（テスト一覧）

- Linux 実行可能: `linear_forward_device_tracked` 既定委譲テスト（`sgd_step_device_tracked_default_delegates_to_sgd_step_device` と同型）・download `Err` 伝播（poison しない）／`is_set()` 事後検査（poison する）の 2 分岐・plan 構築（`Linear`+`ReLU` 融合／`Sigmoid` 混在でフォールバック／leaves 過少・過剰拒否）・モック `BackendOps` で `Unsupported` 全体フォールバック・トークン set 済みで `download` 後に `StorePoisoned`・CPU バックエンドでの新旧 `predict_resident` bit 同一
- 実機 `#[ignore]`: Metal／CUDA で新旧 `predict_resident` bit 同一・run-to-run 同一・Metal `wait_until_completed` 増分 1／predict・CUDA は `linear_forward_device_real_device.rs`（#1216 未実測分）を先に green にする
- A/B: 決定 9 の規則。#1580 は M4 Max、#1581 は GB10

## 8. 出典

`docs/inference-forward-fixed-cost-design.md`・`docs/perf/linear-forward-device-gpu.md`・`docs/perf/infer-reuse-phase-breakdown.md`・`docs/backend-cuda-async-execution-design.md`・`docs/backend-metal-command-batching-design.md`・`docs/perf/lowlayer-diagnosis-2026-09-12.md` §2・§7・`docs/device-resident-update-design.md`・Issue #1579／#1580／#1581／#1582
