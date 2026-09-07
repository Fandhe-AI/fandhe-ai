# 学習 step の CUDA Graph capture／instantiate／launch 経路（イシュー #1349）

## 0. 位置づけ・スコープ

親 #1348（CUDA Graph による launch 固定費削減の試作）・ルート #1341 → #1269 配下。学習 1 step は小形状カーネルの多数 launch で固定費が支配的である（`docs/perf/train-step-phase-breakdown.md`）ことを踏まえ、backward が確定した 2 step 目以降を CUDA Graph へ stream capture し、以降は graph launch のみで再利用する opt-in 経路を実装した。

**当初想定との乖離**: 起票時は「学習 step の capture」を想定していたが、実装開始時の調査（本文書 §1）により、現行データパス（forward・backward がホスト `Tensor` の授受を含み、pageable H2D・複数の同期点を跨ぐ）では **step 全体を 1 個の graph として capture することは構造的に不可能**であることが判明した。CUDA Graph capture は「同一ストリーム上の driver 呼び出しのみで完結する区間」を要求するが、forward の `gemm_resident_rhs_act`（ホスト `Tensor` を返す＝`readback` で同期）・`mse_loss`（ホスト）・backward の `gemm_resident_lhs`（ホスト `Tensor` を返す）はいずれも同期点を含む。そのため本実装は **学習 step のうち update 区間（`BackendOps::sgd_step_device_tracked`）のみ**を capture 対象とする。

## 1. 背景・実測根拠

- `docs/perf/train-step-phase-breakdown.md`: backward が学習 1 step の 75〜97% を占める支配項であり、update（`device_update`）は 0.5〜0.7% に留まる
- 本実装が capture できる区間（update）は step 全体の 1% に満たないため、**性能効果は中立が期待値**であり、本 PR での結線判断は「機構として正しく動作すること」を優先する。性能の正式計測・採否判断は兄弟イシュー #1350 へ引き継ぐ

## 2. 調査で確定した前提・制約

| # | 事実 | 影響 |
|---|---|---|
| F1 | `CudaDevice::new` は `ctx.default_stream()`（legacy NULL stream）を保持する。legacy stream は `cuStreamBeginCapture` の対象にできない。`ctx.new_stream()` は cudarc の「multi-stream mode」を有効化し、以降の全 `device_ptr` 呼び出しに `cuStreamWaitEvent` を自動挿入する | opt-in 有効時のみ `new_stream()` を使う（`crates/backend-cuda/src/device.rs::StreamKind`）。OFF 時は現行どおり `default_stream()` |
| F2 | forward・backward はホスト境界（`readback` 同期）・pageable H2D を含む | capture 可能なのは update 区間のみ |
| F3 | cudarc 0.19.8 の安全 API は `CudaStream::begin_capture`／`end_capture`／`CudaGraph::launch`／`upload` のみ。`cuGraphExecUpdate_v2` は `sys` の生 FFI（`unsafe`）のみ | exec update は本イシューで実装しない（`unsafe` 導入はユーザー承認事項） |
| F4 | `CudaGraph` は `Send`／`Sync` 非実装（driver の規定でも非スレッド安全） | プロセスワイド static ではなく `thread_local!` キャッシュを採用 |
| F5 | `CUgraphInstantiate_flags` に「ゼロ」相当の variant がない | `CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH` を使う（mem alloc ノードを含まない本 graph では不活性だが、`end_capture` が必須引数として要求する） |
| F6 | `context_cache::classify_cuda_result` は capture 系 `CUresult`（900〜908）を従来 wildcard で `Sticky` に分類していた | 明示 arm を追加し意図をコード化（`context_cache.rs`） |
| F7 | opt-in の既存パターン（`crate::precision`。`AtomicBool` + facade setter/getter） | 同型で `crate::graph::set_step_graph_enabled` を追加 |
| F8 | framework-compare の `bench-fandhe` は crates.io ピン版のため新 API を呼べない | 環境変数 `FANDHE_AI_CUDA_GRAPH_STEP` を API setter と併設 |
| F9 | CUDA の非常駐勾配経路（`any_resident == false`）は毎 step `mem.upload(&grad_tensor)` で新規バッファを確保する | graph 経路では永続 staging バッファ（`upload_into`）へ切り替える（新規実装。§4） |

## 3. スコープ

### 3.1 実装した

1. 共有ストリームの opt-in 時限定 `new_stream()` 化（`device.rs::StreamKind`）
2. capture 状態機械（`context_cache.rs`）: `CaptureSession`／`CaptureGuard`・`begin_capture_session`／`begin_sync_point_call`／`is_capturing_on_current_thread`・capture 系 `CUresult` の明示 `Sticky` 化
3. `backend-cuda::graph` モジュール: opt-in フラグ（3 状態: `Off`／`StreamOnly`／`On`）・thread-local graph キャッシュ・`run_captured_sgd_step_segment`（capture → instantiate → launch の手順本体。§4.3 追記: 当初の任意クロージャ版 `run_captured_segment`／`CapturedSegmentBody` は codex-review P0 指摘対応で撤廃し、SGD 更新区間が触れる全リソースを直接引数として受け取る固定形へ変更）
4. `tensor-core::BackendOps` の非破壊拡張: `SegmentResource`／`SegmentKey`／`SegmentRun`・`captured_segment_key`／`run_captured_sgd_step_segment`（既定 `Ok(None)`／`Unsupported`）
5. `backend-cuda::ops::CudaBackendOps` での実装（`captured_segment_key`／`run_captured_sgd_step_segment`）
6. `autodiff::DeviceParamStore::step` の update 区間結線（`!any_resident` 分岐に graph-capture 経路を追加。既存の非 capture 経路〈`else`〉は無変更）
7. `CudaMemory::upload_into`（新規実装。既定 `Unsupported` から `MemoryOps` の非破壊拡張として CUDA 実装を追加。graph 経路の永続 staging バッファ書き込みに必須）
8. facade 公開 API: `set_cuda_graph_step_enabled`／`cuda_graph_step_enabled`
9. テスト: GPU 不要のホストモデルテスト（`context_cache.rs::poison_state_tests`・`graph.rs::tests`・`tensor-core::backend_ops::tests`・`autodiff::optim::device_store::tests::graph_capture_wiring_replays_stable_key_and_matches_non_graph_path`）・`#[ignore]` 実機テスト（`crates/backend-cuda/tests/graph_capture_real_device.rs`〈opt-in ON 側〉・`crates/backend-cuda/tests/graph_capture_real_device_optin_off.rs`〈opt-in OFF 側。codex-review P2／Cursor Bugbot Low 指摘対応・PR #1390 再修正で ON 側から別ファイル＝別プロセスへ分離。`context_cache::cached_device` のプロセス内キャッシュ共有によるテスト順序依存を断つため〉・`crates/facade/tests/cuda_graph_step_bit_identity.rs`）

### 3.2 スコープ外（PR 本文へ記録・起票はユーザー承認後）

- **exec update**（`cuGraphExecUpdate_v2`）: `unsafe` 導入が必要（F3）
- **forward／backward の capture**: 前提となる「forward の facade 常駐チェーン結線（#1216 Phase 2 未完了）」「backward の d_input／d_weight デバイス直接計算（CUDA の `gemm_fp32_strict_into` 未実装）」「loss のデバイス常駐化」が未整備（F2）
- **tape レベルの構造検出**（tape 長・形状・dtype ハッシュ）: step 全体 capture と同時に導入する対象であり、本イシューのキー（`SegmentKey`）は「パラメータ layout + SGD 設定 + 世代 + バッファ同一性」に限定
- **`bench-fandhe --graph` フラグ・GB10 実測**: #1350
- **同期点ガードの適用範囲**: `memory.rs`（`download`／`upload`／`upload_into`／`alloc_zeroed`）・`ops.rs::release_cached_device_memory` に限定した。`gemm.rs::synchronize()`（診断専用・本番ディスパッチから呼ばれない）・`pool.rs` 内部の低レベル確保（`alloc_zeroed`／`alloc_uninit`。`CudaMemory::alloc_zeroed` 経由で既に上位ガード済み）には個別のガードを追加していない。update 区間の body（`sgd_step_device_tracked`）はこれらを呼ばないため受け入れ条件には影響しないが、将来 capture 対象を拡張する場合は再検討が必要
- **cudarc の自動 event 管理の回避策**: 当初は「実機プローブ（§5）で必要性が判明した場合のみユーザー承認を得て実装する・本 PR では実装しない」としていたが、PR #1390 の codex-review P1 指摘（`new_stream()` 有効化時、`launch_builder` が capture 外で記録された read/write イベントへ通常フラグで待機し capture 中は禁止操作としてエラー→ordinal poison につながる）を受け、**案 A（`CudaContext::disable_event_tracking()`〈`unsafe fn`〉）を採用し実装済み**（`device.rs::CudaDevice::new`。§4.1 追記）。本クレートはこの `CudaContext` につき常に単一ストリームのみを保持し複数ストリームを実際には使わないため、cudarc のイベントベース cross-stream 同期機構はそもそも不要という前提に基づく。案 B（生ポインタ引数の launch 変種）は不要となったため対象外のまま

## 4. 設計の要点

### 4.1 ストリーム切替

`CudaDevice` に `StreamKind`（`Legacy`／`Created`）を追加。`crate::graph::step_graph_mode()` が `StreamOnly` 以上（opt-in ON）を返す場合のみ `ctx.new_stream()` を保持し、`Off`（既定）では現行どおり `ctx.default_stream()` を保持する。`CudaDevice` は ordinal ごとに 1 回だけ構築されプロセス生存期間中キャッシュされるため、**opt-in は最初の CUDA デバイス初期化より前に設定する必要がある**。後から ON にした場合、`captured_segment_key` は `BackendError::Unsupported` を返し fail-closed に拒否する（§4.7）。

**イベント追跡の無効化（codex-review P1 指摘対応。PR #1390 追記）**: `ctx.new_stream()` は cudarc を「multi-stream mode」へ切り替え、以降に確保する全 `CudaSlice` へ read/write 用 `CudaEvent` を付与する（既定 `is_event_tracking() == true`）。この状態では `launch_builder`（`sgd.rs::CudaSgd::run` 等）が引数バッファの直近 read/write イベントへ `CU_EVENT_WAIT_DEFAULT` で自動 `cuStreamWaitEvent` するが、その待機対象イベントは通常 capture 開始前（forward／backward 等）に記録されたものであり、capture 外で記録済みのイベントへの `CU_EVENT_WAIT_DEFAULT` 待機は stream capture 中は禁止操作（driver がエラーを返し ordinal を poison する）。本クレートはこの `CudaContext` につき常に単一ストリームのみを保持し複数ストリームを実際には使わないため、イベントベース cross-stream 同期機構はそもそも不要と判断し、`new_stream()` 直後・他のどの `CudaSlice` 確保よりも前に `ctx.disable_event_tracking()`（`unsafe fn`）を呼び、以降確保する `CudaSlice` にイベントを付与しないようにした（`device.rs::CudaDevice::new`）。

**「単一ストリームしか作らない」不変条件の可視性による担保（codex-review P0 再指摘対応。PR #1390 再修正）**: 上記 `disable_event_tracking()` の unsafe 根拠は「この `CudaDevice` はこの `ctx` 上で唯一のストリームしか作らない」という本クレート内部の運用契約だが、`CudaDevice::context()`/`stream()` が無条件 `pub`（crates.io 公開クレート `fandhe-ai-backend-cuda` から `pub use` 再公開済み）のままだと、クレート外の利用者が安全な公開 API の組み合わせ（`context().clone()` → `.new_stream()`）だけで第 2 のストリームを作れてしまい、上記の不変条件を型システムが保証できなかった（`AGENTS.md` の unsafe 不変条件保証の要件違反）。そこで `context()`/`stream()` の可視性を `internal-diagnostics` feature（既存の `Cargo.toml` feature。診断専用ツールの可視性制御）でゲートし、既定ビルド（同 feature 無効。crates.io の通常利用者はこれを有効化しない）では `pub(crate)` に絞った。この feature を要求する本クレート自身の実機診断テスト・ベンチ（`tests/large_buffer_percall_alloc_ab_1149.rs`・`tests/setmaxnreg_common/mod.rs` 経由の 4 ファイル・`examples/device_attributes_dump.rs` 等）は `Cargo.toml` の `[[test]]`/`[[example]]` `required-features` で個別にゲートし、CI の `cargo test --workspace --all-features`（rust-ci test ジョブ・`make test`）では引き続きビルド・実行される（`device.rs::CudaDevice::context` doc コメント参照）。

### 4.2 capture 状態機械

`context_cache::OrdinalState` に `capture: Option<ThreadId>` を追加。`begin_capture_session`（同一 ordinal への再入を拒否）・`is_capturing_on_current_thread`（同期点ガードの判定）・`begin_sync_point_call`（`begin_driver_call` と同じ排他区間で、capture 中なら driver に触れる前に `Unsupported` で拒否）を実装した。capture モードは `CU_STREAM_CAPTURE_MODE_THREAD_LOCAL`。

**cross-thread 排他（codex-review P0 指摘対応。追記）**: `CU_STREAM_CAPTURE_MODE_THREAD_LOCAL` は「capture を乱しうる driver API 呼び出し」の判定を capture 開始スレッドに限定する目的のモードであり、**別スレッドが同じ共有ストリームへ直接カーネル起動すること自体を driver 側が防いでくれる保証ではない**。そのため `begin_driver_call`（`captured_segment_key`／`sgd_step_device_tracked` 等、通常の driver 呼び出し全般が通る唯一の入口）自身が `state.capture` を検査し、capture 中は**capture を開始したスレッド以外**からの呼び出しを `Unsupported` で一律拒否するよう変更した（capture を開始したスレッド自身は通す——`body` 実行がこの入口を再度通るため）。

**body の panic 対応（codex-review P1・Cursor Bugbot 指摘対応。追記）**: `body` が panic（unwind）した場合でも `stream.end_capture` を必ず呼んでから panic を再送出するよう `run_captured_segment` を変更した（`std::panic::catch_unwind` で包み、`end_capture` 実行後に `resume_unwind`）。以前は `body()` の panic が `end_capture` 呼び出しをスキップし、driver 側の capture 状態が終了されないまま残る整合性違反があった。

**バッファ解放の排他（codex-review P0 再指摘・Cursor Bugbot High 指摘対応。PR #1390 再修正）**: `cudarc::driver::CudaSlice::drop`（`memory.rs::CudaBufferHandle::Drop` から実行される実際の `cuMemFreeAsync`/`cuMemFree` 発行）は `begin_driver_call`／`begin_capture_session` の排他機構を一切経由しないため、別スレッドが capture 中に無関係なバッファを drop すると、その解放が capture 中の共有ストリームへ意図せず記録されうる。この排他を `context_cache::begin_buffer_release`（`BufferReleaseToken` を返す RAII 関数）で実装した。旧稿（`wait_until_not_capturing`）は「駐機して戻るだけ」で、戻った直後に別の capture セッションが実際に開始してしまう競合窓が残っていた（P0 再指摘）。本関数は駐機解除後に `state.in_flight` へ登録してから返し、`memory.rs` は返した `BufferReleaseToken` を実際の `CudaSlice` drop が完了するまで保持することで、その競合窓を閉じる。

さらに、単純に「`state.capture` が `Some`（`begin_capture_session` の in_flight ドレイン待機中を含む）の間は常に駐機する」設計は、以下のデッドロックを生む（Cursor Bugbot High 指摘）: スレッド B が既に `CallToken` を保持したまま（`in_flight` に計上済みのまま）一時バッファを drop すると、その `Drop` が別スレッド A の capture 完了を待って駐機する一方、A は `begin_capture_session` の drain で B の `in_flight` が 0 になるのを待ち続け、両者が永久に待ち合う。これを避けるため `OrdinalState` に **`capturing_active: bool`**（`capture` が `Some` の区間のうち、in_flight ドレインが完了した後だけ `true` になるサブフラグ）を追加し、`begin_capture_session` がドレイン完了と**同一ロック区間内**でこれを立てる（ロックを一旦手放してから別関数で立てる設計は同種の競合窓を生むため不採用）。`begin_buffer_release` は `state.capture.is_some()` ではなく `state.capturing_active` を駐機条件にする——ドレイン中（driver 側はまだ capture していない）の解放は素通しし、実際に driver capture が進行中の区間だけを対象にすることで、上記デッドロックを構造的に回避する。`CaptureGuard::drop` で `capturing_active` も必ず `false` へ戻す。

### 4.3 graph モジュール

**任意クロージャの撤廃（codex-review P0 指摘対応。追記）**: 当初は `crate::graph::run_captured_segment(ordinal, stream, key, resources, body)`（`resources: &mut [&mut DeviceBuffer<f32>]`・`body: &mut CapturedSegmentBody<'_>` という任意クロージャ）という設計だったが、`body` が `resources` に含まれない外部 `DeviceBuffer<f32>` をクロージャキャプチャ経由で直接触れる抜け道があった。現在は `crate::graph::run_captured_sgd_step_segment(ordinal, stream, key, ops, param, grad, velocity, config, token)` へ変更し、SGD 更新区間が触れる全リソース（`param`／`grad`／`velocity`）を直接引数として受け取り、区間本体（capture 対象のカーネル起動）も本関数が固定的に `CudaBackendOps::sgd_step_device_tracked` を呼ぶことで行う（呼び出し元は任意コードを注入できない）。`CapturedSegmentBody` 型は撤廃済み。

**capture 開始前の in_flight ドレイン順序（codex-review P0 指摘対応。追記）**: `begin_capture_session`（他スレッドの `in_flight` をドレインしてから返る）を、呼び出しスレッド自身が当該 ordinal のトークンを 1 つも保持していない状態で先に呼び、その後に初めて `begin_driver_call` を呼ぶ順序へ変更した（逆順だと、他スレッドが capture 開始の直前に `begin_driver_call` を通過済み〈`in_flight` に計上済みだが実際の driver 呼び出しはまだ〉だった場合、その呼び出しが capture 開始後に共有ストリームへカーネル起動を発行し意図せず graph へ混入しうる窓があった）。

**SGD カーネルウォームアップの位置（Cursor Bugbot Medium 指摘対応。PR #1390 再修正）**: `sgd_step_device_tracked` は内部で `context_cache::cached_sgd`（`ordinal` キーの NVRTC コンパイル済みカーネルの singleflight キャッシュ）を参照するが、プロセス内でこの ordinal の SGD が一度も呼ばれていない場合、キャッシュミスにより NVRTC コンパイル＋`cuModuleLoadDataEx`（driver へのモジュールロード）が初回発生する。旧稿はこのウォームアップを `begin_capture_session` の**後**（`state.capture` 設定済み・in_flight ドレイン完了後）で行っていたため、NVRTC コンパイルという遅い操作の間ずっと他スレッドの `begin_driver_call` が `Unsupported` で拒否され続けていた（capture 意図の登録自体はコンパイルと無関係なため、この長時間ブロックは不要）。現在は `begin_capture_session` より**前**に、独立した `begin_driver_call` 境界（poison／世代検査を保ったまま。ウォームアップ専用トークンは使用後すぐ drop）でウォームアップを完了させる。

手順（PR #1390 再修正でウォームアップの位置を変更）:

1. thread-local キャッシュ（`STEP_GRAPHS`。上限 8・世代不一致 evict）から `key` に一致する graph を take → ヒットなら `begin_driver_call` で poison／世代検査してから `launch()` のみ実行し `SegmentRun::Replayed`
2. ミスなら、まず独立した `begin_driver_call` 境界で SGD カーネルをウォームアップ（上記）→ `begin_capture_session`（drain 完了と同一ロック区間で `capturing_active` を設定済み。§4.2 追記）→ `begin_driver_call` → `stream.begin_capture` → SGD 更新 1 回（`sgd_step_device_tracked`。panic しても直後で必ず `end_capture` する。§4.2 追記）→ `stream.end_capture` → 成功時は `graph.upload()` → 初回 `launch()` → キャッシュへ格納 → `SegmentRun::Captured`
3. SGD 更新の `Err`・`end_capture` の `Err`・空 graph（`Ok(None)`）はいずれも fail-closed なエラーとして呼び出し元へ返す（graph はキャッシュに残さない）
4. **`graph.upload()` の失敗も fail-closed で伝播する**（codex-review P0 指摘対応・追記。以前は poison 化のみで握りつぶし、直後の `launch()` が別途成功すると全体が `Ok(Captured)` 扱いになっていた——「最初に失敗した driver エラーを伝播する」という本クレート全体の契約に反する後退だった）

### 4.4 `BackendOps` の非破壊拡張

`SegmentKey { generation, config_key, resources: Vec<SegmentResource> }`（`SegmentResource { addr, numel }`）。`CudaBackendOps::captured_segment_key` は opt-in OFF または非 capturable stream なら `Ok(None)`（後者は opt-in ON だが legacy stream のままの設定ミスを示すため実際には `Err(Unsupported)`。§4.7）、それ以外は各バッファの `device_ptr` から `SegmentKey` を構築する。**driver 呼び出し境界（codex-review P0 指摘対応。追記）**: `resources` の世代収集（host-only・driver 非接触）→ `begin_driver_call`（poison／世代検査）→ `device_handle_raw` → `segment_resources_for` の順に固定し、poison・Retiring・別スレッド capture 中のいずれでも driver に触れる前に拒否されるようにした。

**replay 直前の再検証（codex-review P0 指摘対応。追記）**: `SegmentKey` 自身はバッファの所有権・借用を保持しない値型のため、`param`／`grad`／`velocity`（呼び出しの間ライフタイムが保証される借用）から現在のアドレス集合を再計算し `key.resources` と完全一致することを確認してから初めて `crate::graph::run_captured_sgd_step_segment` へ委譲する（不一致なら driver に一切触れず `InvalidArgument` で拒否。解放済み・別バッファへ再利用済みのアドレスを参照する古い graph を安全確認なしに再生する事態を防ぐ）。

**デバイス一致検査（codex-review P0 再指摘対応。PR #1390 再修正）**: `captured_segment_key`／`run_captured_sgd_step_segment` のいずれにも、通常経路の `sgd_step_device` が持つ `Device::Cuda(self.ordinal)` 一致検査（driver に触れる前・host-only）が欠けていた。この検査がないと、同一スレッドで GPU 0 の graph をキャッシュした `key` と GPU 1 の `param`/`grad`/`velocity`（`Device::Cuda(1)`）を GPU 1 の `CudaBackendOps`（`self.ordinal == 1`）へ渡した際、世代番号がたまたま一致すれば上記「replay 直前の再検証」（アドレス一致検査）も通過してしまい、検査・エラー観測が GPU 1 に対して行われる一方でキャッシュ済み graph は GPU 0 上で再生される——GPU 0 の capture 排他・poison 機構を迂回する。両関数の冒頭で `resources`／`param`/`grad`/`velocity` 全ての `.device()` が `Device::Cuda(self.ordinal)` と一致することを検査し、不一致は `BackendError::DeviceMismatch` で拒否するよう追加した。

### 4.5 `DeviceParamStore::step` の結線

`!any_resident` 分岐（CUDA が現状該当する唯一の経路）に graph-capture 判定を追加した:

1. `ops.captured_segment_key(&[], config_key)` で安価に capability を問い合わせる（`resources` は空でよい——`Some`/`None` は opt-in・ストリーム種別のみで決まる契約）
2. `total_numel == 0` は常に対象外（空 graph の回避）
3. capability あり: 永続 `GradStaging`（`total_numel` 分。初回のみ確保）を用意し、`mem.upload_into` で grad を書き込んでから、実際のバッファ集合で `captured_segment_key` を再計算して `key` を確定し、`ops.run_captured_sgd_step_segment(key, param, grad, velocity, config, token)` で update を実行する
4. capability なし: 既存の「毎回新規 grad バッファへ upload」経路（本イシュー導入前と無変更）

`config_key` は `fold_sgd_step_config_key` で `SgdStepConfig` の全フィールド（`is_first_step` を含む）から算出する。ハイパーパラメータ変更や `is_first_step` の遷移（1 step 目 → 2 step 目）で `config_key` が変わり、新規 capture が発生する。

**衝突耐性（codex-review P1 指摘対応。追記）**: 当初は単語単位の XOR→乗算という弱い畳み込み（手書き FNV-1a 風）を使っていたが、`nesterov`／`is_first_step` の 1 ビット値が上位ビットへほとんど拡散せず、実際に**異なる設定同士が衝突する具体例**が確認された。これは「衝突しても実害は再利用が起きない方向にのみ倒れる」という当初の想定を裏切る——`SegmentKey` は `generation`／`resources`（バッファアドレス）が同一パラメータ列を使い続ける限り不変のため、`config_key` の衝突は「古い設定〈例: 変更前の `lr`〉で capture 済みの graph を新しい設定のまま気づかず再生する」という契約違反（設定変更の無視）に直結する。現在は `std::collections::hash_map::DefaultHasher`（SipHash 系。バイト単位で逐次混合するため単語単位の弱い混合より衝突耐性が大幅に高い）へ切り替えた。

### 4.6 facade 公開 API

`fandhe_ai::set_cuda_graph_step_enabled`／`cuda_graph_step_enabled`（`crate::precision` と同型の委譲）。加えて環境変数 `FANDHE_AI_CUDA_GRAPH_STEP`（`1`／`true`＝ON・`stream-only`＝ストリーム種別のみ変更する診断用中間状態）を提供し、crates.io ピン版を呼ぶ `bench-fandhe`（#1350）からも opt-in を切り替えられるようにした（F8）。

### 4.7 エラー・poison 契約

| 事象 | 結果 |
|---|---|
| body 内で同期点（download／upload／alloc／release）を呼ぶ | `Unsupported`（driver に触れる前）。ordinal は poison しない |
| capture 中に capture 開始スレッド以外から driver 呼び出し | `Unsupported`（driver に触れる前。§4.2 追記） |
| capture 中の driver エラー（begin/end capture・launch・upload） | ordinal を `Poisoned{false}` へ（明示 `Sticky` 分類）。`upload` 失敗も fail-closed で伝播（§4.3 追記） |
| `end_capture` が空 graph | fail-closed エラー |
| `body` が panic | `end_capture` を実行してから panic を再送出（§4.2 追記） |
| `run_captured_sgd_step_segment` の `param`／`grad`／`velocity` が `key.resources` と不一致 | `InvalidArgument`（driver に触れる前。§4.4 追記） |
| 世代不一致 | `StaleDeviceGeneration` |
| opt-in ON だが legacy stream で初期化済み | `Unsupported`（設定順序の誤りを早期に顕在化） |

## 5. 実機実測

本エージェント実行環境に CUDA 実機が存在しないため、以下は **未実測**のまま記入欄を残す:

- §4.1 の実機プローブ（cudarc の自動 event 管理と capture の互換性。「リスクと安全側の判断」参照）
- `crates/backend-cuda/tests/graph_capture_real_device.rs`（opt-in ON 側。`#[ignore]`）・`crates/backend-cuda/tests/graph_capture_real_device_optin_off.rs`（opt-in OFF 側。`#[ignore]`）の実行結果
- `crates/facade/tests/cuda_graph_step_bit_identity.rs`（`#[ignore]`）の実行結果（受け入れ条件 (a)）
- `make test-ignored-cuda` による非後退確認（ストリーム切替が opt-in OFF 時の既存動作を壊していないことの実機確認）

実機セッションで最初に実行すべき手順:

```sh
# 1. 既存回帰の非後退確認
make test-ignored-cuda

# 2. capture 可否プローブ・bit 同一・fail-closed 検証（本イシュー新規。
#    opt-in ON 側と OFF 側は別ファイル＝別プロセス。PR #1390 再修正）
cargo test -p fandhe-ai-backend-cuda --release --test graph_capture_real_device \
  -- --ignored --nocapture --test-threads=1
cargo test -p fandhe-ai-backend-cuda --release --test graph_capture_real_device_optin_off \
  -- --ignored --nocapture

# 3. facade 経由の 10 step bit 同一検証（2 プロセス比較。ファイル冒頭コメント参照）
cargo test -p fandhe-ai --release --test cuda_graph_step_bit_identity \
  -- --ignored --nocapture eager_baseline
FANDHE_AI_CUDA_GRAPH_STEP=1 \
cargo test -p fandhe-ai --release --test cuda_graph_step_bit_identity \
  -- --ignored --nocapture graph_capture
```

**§4.1 の実機プローブが失敗した場合**（`CUDA_ERROR_STREAM_CAPTURE_ISOLATION` 等。「リスクと安全側の判断」参照）: 回避策（案 A `disable_event_tracking()`〈`unsafe fn`〉・案 B 生ポインタ launch 変種）はユーザー承認事項のため、実測エビデンスとともに承認依頼を起票し、承認が得られるまで本機構は「機構としては実装済みだが実機で capture が成立しない」状態のまま残す。

## 6. #1350 への申し送り

- 計測は `--phases` の `device_update` 区間に効果が閉じることを前提に、`FANDHE_AI_CUDA_GRAPH_STEP` の 3 状態（未設定＝OFF／`stream-only`／`1`）で比較し、「created stream の event 管理コスト」と「capture の効果」を分離して記録する
- `step_total` は中立見込み（update が step の 0.5〜0.7% のため）。後退が観測されても opt-in 既定 OFF のため結線撤回は不要
- launch 回数の比較は `SegmentRun::Captured`／`Replayed` の計数、または既存の launch カウンタ基盤を流用する

## 7. リスクと安全側の判断

- ストリーム切替（§4.1）が最大のリスク。opt-in 時限定にすることで OFF 時の挙動・性能を現行と同一に保つ
- cudarc の自動 event 管理（ON 時のみ有効化される `cuStreamWaitEvent`）は capture と非互換であることが PR #1390 の codex-review 指摘（§3.2・§4.1 追記）で確定した。回避策（`disable_event_tracking()`）は unsafe 面の判断を伴うが、本クレートがこの `CudaContext` につき単一ストリームしか使わない前提が安全性根拠として成立するため実装・適用済み（§4.1）
- 別スレッドが同じ ordinal の `DeviceBuffer` を drop すると `CudaSlice::drop` が capture 中の共有ストリームへ非同期解放を発行しうる問題（codex-review P1 指摘。PR #1390）に対し、`context_cache::begin_buffer_release` を `CudaBufferHandle::Drop` から呼び、返した `BufferReleaseToken` を実際の解放が完了するまで保持することで対応した（`memory.rs`。§4.2「バッファ解放の排他」参照。P0 再指摘・Bugbot High デッドロック指摘を経て `capturing_active` サブフラグによる再修正済み）
- 「step 全体」を capture できない事実は起票時の想定との乖離であり、隠さず本文書 §0 に明記する
