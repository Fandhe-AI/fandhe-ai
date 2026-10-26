# 学習 step の CUDA Graph capture／instantiate／launch 経路（イシュー #1349）

## 0. 位置づけ・スコープ

親 #1348（CUDA Graph による launch 固定費削減の試作）・ルート #1341 → #1269 配下。学習 1 step は小形状カーネルの多数 launch で固定費が支配的である（`docs/perf/train-step-phase-breakdown.md`）ことを踏まえ、backward が確定した 2 step 目以降を CUDA Graph へ stream capture し、以降は graph launch のみで再利用する opt-in 経路を実装した。

**当初想定との乖離**: 起票時は「学習 step の capture」を想定していたが、実装開始時の調査（本文書 §1）により、現行データパス（forward・backward がホスト `Tensor` の授受を含み、pageable H2D・複数の同期点を跨ぐ）では **step 全体を 1 個の graph として capture することは構造的に不可能**であることが判明した。CUDA Graph capture は「同一ストリーム上の driver 呼び出しのみで完結する区間」を要求するが、forward の `gemm_resident_rhs_act`（ホスト `Tensor` を返す＝`readback` で同期）・`mse_loss`（ホスト）・backward の `gemm_resident_lhs`（ホスト `Tensor` を返す）はいずれも同期点を含む。そのため本実装は **学習 step のうち update 区間（`BackendOps::sgd_step_device_tracked`）のみ**を capture 対象とする。

**#3259 追補**: その後 #3252 で前提が揃い、step 全体の capture 設計は §8 に記録した（本節の記述は #1349 当時の事実として残す）。

## 1. 背景・実測根拠

- `docs/perf/train-step-phase-breakdown.md`: backward が学習 1 step の 75〜97% を占める支配項であり、update（`device_update`）は 0.5〜0.7% に留まる
- 本実装が capture できる区間（update）は step 全体の 1% に満たないため、**性能効果は中立が期待値**であり、本 PR での結線判断は「機構として正しく動作すること」を優先する。性能の正式計測・採否判断は兄弟イシュー #1350 へ引き継ぐ
- **step 全体 capture の前提（#3252）**: 学習チェーン（`DeviceParamStore::train_step_device_chain`）は、CUDA で層出力・dPred・d_input・mask 後勾配・loss を永続バッファへ `_into_tracked` で書き、定常状態の区間内にホスト同期・H2D・確保を持たない（GB10 実測は `docs/perf/logs/train-step-chain-cuda-3252/`、設計は `docs/compat-resident-train-step-decision.md` §12）。#3258 の step 全体 capture が要る固定構造（永続スクラッチ・永続 loss）はこれで揃った。capture 中の確保・ワークスペース grow は拒否されるため、capture の前にウォームアップで定常状態にする

## 2. 調査で確定した前提・制約

| # | 事実 | 影響 |
|---|---|---|
| F1 | `CudaDevice::new` は `ctx.default_stream()`（legacy NULL stream）を保持する。legacy stream は `cuStreamBeginCapture` の対象にできない。`ctx.new_stream()` は cudarc の「multi-stream mode」を有効化し、以降の全 `device_ptr` 呼び出しに `cuStreamWaitEvent` を自動挿入する | opt-in 有効時のみ `new_stream()` を使う（`crates/backend-cuda/src/device.rs::StreamKind`）。OFF 時は現行どおり `default_stream()` |
| F2 | forward・backward はホスト境界（`readback` 同期）・pageable H2D を含む。CUDA の `gemm_fp32_strict_into` は #1559 で実装済み（`backend-cuda::ops` の `gemm_fp32_strict_into`）だが、ホスト入力の関数内同期（`stream.synchronize()`・設計判断 A）を伴い、capture 中は `Unsupported` を返す（`docs/train-step-resident-parts-design.md` §6・§7） | capture 可能なのは update 区間のみ（結論不変。理由は forward・backward の capture の前提が依然未整備なこと） |
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
9. テスト: GPU 不要のホストモデルテスト（`context_cache.rs::poison_state_tests`・`graph.rs::tests`・`tensor-core::backend_ops::tests`・`autodiff::optim::device_store::tests::graph_capture_wiring_replays_stable_key_and_matches_non_graph_path`）・`#[ignore]` 実機テスト（`crates/backend-cuda/tests/graph_capture_real_device.rs`〈opt-in ON 側〉・`crates/backend-cuda/tests/graph_capture_real_device_optin_off.rs`〈opt-in OFF 側。codex-review P2／Cursor Bugbot Low 指摘対応・PR #1390 再修正で ON 側から別ファイル＝別プロセスへ分離。`context_cache::cached_device` のプロセス内キャッシュ共有によるテスト順序依存を断つため〉・`crates/facade/tests/cuda_graph_step_bit_identity.rs`〈`eager_baseline`／`graph_capture`／`graph_capture_completes_training_loop_without_error`。各 `--exact` 単独実行必須〉・`crates/facade/tests/cuda_graph_step_two_gpu_bit_identity.rs`〈2 GPU 機械比較テスト。codex-review P2 再指摘対応・PR #1390 マージ時是正で `cuda_graph_step_bit_identity.rs` から分離。3 関数名がいずれも部分文字列 `graph_capture` を含み `cargo test graph_capture` のような部分一致フィルタで同一プロセス・並行スレッドに選ばれると「opt-in OFF で開始する」前提等が崩れるため。共有ヘルパーは `crates/facade/tests/cuda_graph_step_common/mod.rs`〉。イシュー #1480 で `cuda_graph_step_common::train_on_cuda` へ各 step の重み勾配比較（`Tape::param_grads_to_host` の戻り値。`step[{step}].grad[{p}][{j}].bits` ラベル）を追加し、上記 2 テストファイルの比較対象を loss・入力勾配・パラメータに加えて重み勾配へ拡張済み〈§5「#1480 追補」参照〉）

### 3.2 スコープ外（PR 本文へ記録・起票はユーザー承認後）

- **推論チェーンの capture は別機構（#2115）**: 推論 forward チェーン（`predict_device_chain`）の capture は本書の update 区間 capture とは別の opt-in（`FANDHE_AI_CUDA_GRAPH_INFER`）として `crates/backend-cuda/src/graph.rs` に追加した（`docs/perf/infer-chain-graphcapture-cuda-ab.md`）。created stream の選択（§4.1）は両 opt-in の OR で決まる

- **exec update**（`cuGraphExecUpdate_v2`）: `unsafe` 導入が必要（F3）
- **forward／backward の capture**: 前提となる「forward の facade 常駐チェーン結線（#1216 Phase 2 未完了）」「backward の d_input／d_weight デバイス直接計算（CUDA の `gemm_fp32_strict_into` は #1559 で実装済みだが、ホスト入力の関数内同期を伴い capture 中は `Unsupported`）」「loss のデバイス常駐化」が未整備（F2）（#3252 で前提が揃い、設計は §8）
- **tape レベルの構造検出**（tape 長・形状・dtype ハッシュ）: step 全体 capture と同時に導入する対象であり、本イシューのキー（`SegmentKey`）は「パラメータ layout + SGD 設定 + 世代 + バッファ同一性」に限定（step 全体のキー設計は §8.3）
- **`bench-fandhe --graph` フラグ・GB10 実測**: #1350
- **同期点ガードの適用範囲**: `memory.rs`（`download`／`upload`／`upload_into`／`alloc_zeroed`）・`ops.rs::release_cached_device_memory` に限定した。`gemm.rs::synchronize()`（診断専用・本番ディスパッチから呼ばれない）・`pool.rs` 内部の低レベル確保（`alloc_zeroed`／`alloc_uninit`。`CudaMemory::alloc_zeroed` 経由で既に上位ガード済み）には個別のガードを追加していない。update 区間の body（`sgd_step_device_tracked`）はこれらを呼ばないため受け入れ条件には影響しないが、将来 capture 対象を拡張する場合は再検討が必要（**#3260 で step 全体へ拡張した際に再監査済み**。結果は §8.8.3。`mse.rs` の永続ワークスペース初回確保にガードが無かったため是正した）
- **cudarc の自動 event 管理の回避策**: 当初は「実機プローブ（§5）で必要性が判明した場合のみユーザー承認を得て実装する・本 PR では実装しない」としていたが、PR #1390 の codex-review P1 指摘（`new_stream()` 有効化時、`launch_builder` が capture 外で記録された read/write イベントへ通常フラグで待機し capture 中は禁止操作としてエラー→ordinal poison につながる）を受け、**案 A（`CudaContext::disable_event_tracking()`〈`unsafe fn`〉）を採用し実装済み**（`device.rs::CudaDevice::new`。§4.1 追記）。本クレートはこの `CudaContext` につき常に単一ストリームのみを保持し複数ストリームを実際には使わないため、cudarc のイベントベース cross-stream 同期機構はそもそも不要という前提に基づく。案 B（生ポインタ引数の launch 変種）は不要となったため対象外のまま

## 4. 設計の要点

### 4.1 ストリーム切替

`CudaDevice` に `StreamKind`（`Legacy`／`Created`）を追加。`crate::graph::step_graph_mode()` が `StreamOnly` 以上（opt-in ON）を返す場合のみ `ctx.new_stream()` を保持し、`Off`（既定）では現行どおり `ctx.default_stream()` を保持する。`CudaDevice` は ordinal ごとに 1 回だけ構築されプロセス生存期間中キャッシュされるため、**opt-in は最初の CUDA デバイス初期化より前に設定する必要がある**。後から ON にした場合、`captured_segment_key` は `BackendError::Unsupported` を返し fail-closed に拒否する（§4.7）。

**イベント追跡の無効化（codex-review P1 指摘対応。PR #1390 追記）**: `ctx.new_stream()` は cudarc を「multi-stream mode」へ切り替え、以降に確保する全 `CudaSlice` へ read/write 用 `CudaEvent` を付与する（既定 `is_event_tracking() == true`）。この状態では `launch_builder`（`sgd.rs::CudaSgd::run` 等）が引数バッファの直近 read/write イベントへ `CU_EVENT_WAIT_DEFAULT` で自動 `cuStreamWaitEvent` するが、その待機対象イベントは通常 capture 開始前（forward／backward 等）に記録されたものであり、capture 外で記録済みのイベントへの `CU_EVENT_WAIT_DEFAULT` 待機は stream capture 中は禁止操作（driver がエラーを返し ordinal を poison する）。本クレートはこの `CudaContext` につき常に単一ストリームのみを保持し複数ストリームを実際には使わないため、イベントベース cross-stream 同期機構はそもそも不要と判断し、`new_stream()` 直後・他のどの `CudaSlice` 確保よりも前に `ctx.disable_event_tracking()`（`unsafe fn`）を呼び、以降確保する `CudaSlice` にイベントを付与しないようにした（`device.rs::CudaDevice::new`）。

**「単一ストリームしか作らない」不変条件の可視性による担保（codex-review P0 再指摘対応。PR #1390 再修正）**: 上記 `disable_event_tracking()` の unsafe 根拠は「この `CudaDevice` はこの `ctx` 上で唯一のストリームしか作らない」という本クレート内部の運用契約だが、`CudaDevice::context()`/`stream()` が無条件 `pub`（crates.io 公開クレート `fandhe-ai-backend-cuda` から `pub use` 再公開済み）のままだと、クレート外の利用者が安全な公開 API の組み合わせ（`context().clone()` → `.new_stream()`）だけで第 2 のストリームを作れてしまい、上記の不変条件を型システムが保証できなかった（`AGENTS.md` の unsafe 不変条件保証の要件違反）。そこで `context()`/`stream()` の可視性を `internal-diagnostics` feature（既存の `Cargo.toml` feature。診断専用ツールの可視性制御）でゲートし、既定ビルド（同 feature 無効。crates.io の通常利用者はこれを有効化しない）では `pub(crate)` に絞った。この feature を要求する本クレート自身の実機診断テスト・ベンチ（`tests/large_buffer_percall_alloc_ab_1149.rs`・`tests/setmaxnreg_common/mod.rs` 経由の 4 ファイル・`examples/device_attributes_dump.rs` 等）は `Cargo.toml` の `[[test]]`/`[[example]]` `required-features` で個別にゲートし、CI の `cargo test --workspace --all-features`（rust-ci test ジョブ・`make test`）では引き続きビルド・実行される（`device.rs::CudaDevice::context` doc コメント参照）。

### 4.2 capture 状態機械

`context_cache::OrdinalState` に `capture: Option<ThreadId>` を追加。`begin_capture_session`（同一 ordinal への再入を拒否）・`is_capturing_on_current_thread`（同期点ガードの判定）・`begin_sync_point_call`（`begin_driver_call` と同じ排他区間で、capture 中なら driver に触れる前に `Unsupported` で拒否）を実装した。capture モードは `CU_STREAM_CAPTURE_MODE_THREAD_LOCAL`。

**cross-thread 排他（codex-review P0 指摘対応。追記）**: `CU_STREAM_CAPTURE_MODE_THREAD_LOCAL` は「capture を乱しうる driver API 呼び出し」の判定を capture 開始スレッドに限定する目的のモードであり、**別スレッドが同じ共有ストリームへ直接カーネル起動すること自体を driver 側が防いでくれる保証ではない**。そのため `begin_driver_call`（`captured_segment_key`／`sgd_step_device_tracked` 等、通常の driver 呼び出し全般が通る唯一の入口）自身が `state.capture` を検査し、capture 中は**capture を開始したスレッド以外**からの呼び出しを `BackendError::DeviceContextCaptureInProgress`（一過性の競合を表す専用 variant。恒久的な未実装・設定誤りを表す `Unsupported` とは意味が異なる——呼び出し元がリトライすべきか否かの判断材料になる）で一律拒否するよう変更した（capture を開始したスレッド自身は通す——`body` 実行がこの入口を再度通るため）。

**body の panic 対応（codex-review P1・Cursor Bugbot 指摘対応。追記。PR #1390 再修正で catch_unwind 経路を変更）**: `body` が panic（unwind）した場合でも `stream.end_capture` を必ず呼んでから、panic を `BackendError::KernelLaunchFailed`（panic ペイロードから抽出した可読メッセージを含む）へ変換して呼び出し元の `Result` 経由の失敗処理へ合流させるよう `run_captured_sgd_step_segment` を変更した（`std::panic::catch_unwind` で包み、`end_capture` 実行後に `body_result` を `Err(BackendError::KernelLaunchFailed(..))` にする）。旧稿は `end_capture` 実行後に `std::panic::resume_unwind` で panic をライブラリ境界外へ再送出していたが、これは `.claude/rules/coding-rust.md`「本番経路で panic しない」規約に反し、呼び出し元（`fandhe_ai_autodiff::optim::device_store::DeviceParamStore::step`）の `Result` ベースの poison・pending 復元処理を丸ごと迂回してしまうため、PR #1390 の codex-review 再指摘を受けて型付きエラーへの変換方式へ改めた。以前は `body()` の panic が `end_capture` 呼び出しをスキップし、driver 側の capture 状態が終了されないまま残る整合性違反があった。

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
| capture 中に capture 開始スレッド以外から driver 呼び出し | `DeviceContextCaptureInProgress`（一過性の競合。driver に触れる前。恒久的な `Unsupported` とは異なる。§4.2 追記） |
| capture 中の driver エラー（begin/end capture・launch・upload） | ordinal を `Poisoned{false}` へ（明示 `Sticky` 分類）。`upload` 失敗も fail-closed で伝播（§4.3 追記） |
| `end_capture` が空 graph | fail-closed エラー |
| `body` が panic | `end_capture` を実行してから panic を `BackendError::KernelLaunchFailed` へ変換して返す（§4.2 追記。PR #1390 再修正） |
| `run_captured_sgd_step_segment` の `param`／`grad`／`velocity` が `key.resources` と不一致 | `InvalidArgument`（driver に触れる前。§4.4 追記） |
| 世代不一致 | `StaleDeviceGeneration` |
| opt-in ON だが legacy stream で初期化済み | `Unsupported`（設定順序の誤りを早期に顕在化） |

## 5. 実機実測

**#1350（GB10 実機実測。2026-09-08）で実測完了**。§4.1 の実機プローブ
（cudarc の自動 event 管理と capture の互換性）は失敗せず、
`disable_event_tracking()`（案 A）採用済みの現行実装のまま capture が
成立することを確認した——「§4.1 の実機プローブが失敗した場合」の回避策
検討は不要だった。

- `make test-ignored-cuda` 1 段目（`--all-features`。graph 系 2 テストを
  `--skip`）: 34 passed・2 failed。2 件の失敗（`init_cost_diag_gemm_
  new_lru_cold_vs_warm`・`cuda_gemm_new_second_construction_reuses_
  module_cache`）はいずれも `#[ignore]` テスト並行実行時のモジュール
  キャッシュ LRU 相互汚染という失敗メッセージ自体が明記する既知の
  非決定性（イシュー #1107）で、本イシューの変更（launch カウンタ
  追加）とは無関係——ストリーム切替（§4.1）が opt-in OFF 時の既存動作
  を壊していないことの非後退確認としては成立している
- `cargo test -p fandhe-ai-backend-cuda --release --test
  graph_capture_real_device -- --ignored --nocapture --test-threads=1`
  （opt-in ON 側）: **2 passed・0 failed**
  （`sgd_update_segment_captures_then_replays_bit_identically`・
  `different_config_key_produces_a_different_segment_key`）。#1350 が
  追加した launch カウンタ assert（capture で `captured`+1・
  `graph_launches`+1・`replayed` 不変／replay で `replayed`+1・
  `graph_launches`+1・`captured` 不変）を含めて自己検証済み
- `scripts/verify-cuda-graph-step-bit-identity.sh`
  （`crates/facade/tests/cuda_graph_step_bit_identity.rs` の eager_
  baseline／graph_capture 2 プロセス比較の自動化版。受け入れ条件 (a)）:
  **OK: eager_baseline（opt-in OFF）と graph_capture（opt-in ON）は
  2662 行すべて bit 同一**
- `graph_capture_real_device_optin_off.rs`・`cuda_graph_step_two_gpu_
  bit_identity.rs`（2 GPU 機のみ）は #1350 の実測範囲外のまま（GB10
  は単一 GPU のため後者は本来スコープ外）

実測ログは `docs/perf/logs/cuda-graph-step-ab-1350/`
（`test-ignored-cuda-first-invocation.log`・
`graph_capture_real_device_optin_off.log`・
`verify-cuda-graph-step-bit-identity.log`）。性能実測（A/B）・採否
判定は `docs/perf/train-step-phase-breakdown.md` §16 を参照。

以下は上記実測を実行する際に用いた手順（再現用に維持する）:

```sh
# 1. 既存回帰の非後退確認
make test-ignored-cuda

# 2. capture 可否プローブ・bit 同一・fail-closed 検証（本イシュー新規。
#    opt-in ON 側と OFF 側は別ファイル＝別プロセス。PR #1390 再修正）
cargo test -p fandhe-ai-backend-cuda --release --test graph_capture_real_device \
  -- --ignored --nocapture --test-threads=1
cargo test -p fandhe-ai-backend-cuda --release --test graph_capture_real_device_optin_off \
  -- --ignored --nocapture

# 3. facade 経由の 10 step bit 同一検証（2 プロセス比較。--exact 必須の
#    理由はファイル冒頭コメント参照。PR #1390 マージ時是正）
cargo test -p fandhe-ai --release --test cuda_graph_step_bit_identity \
  -- --ignored --nocapture --exact eager_baseline
FANDHE_AI_CUDA_GRAPH_STEP=1 \
cargo test -p fandhe-ai --release --test cuda_graph_step_bit_identity \
  -- --ignored --nocapture --exact graph_capture

# 3b. 2 GPU 搭載機のみ: 機械比較版（別ファイル・別バイナリ）
cargo test -p fandhe-ai --release --test cuda_graph_step_two_gpu_bit_identity \
  -- --ignored --nocapture --exact graph_capture_matches_eager_baseline_bit_identical_across_two_gpus
```

**§4.1 の実機プローブが失敗した場合**（`CUDA_ERROR_STREAM_CAPTURE_ISOLATION` 等。「リスクと安全側の判断」参照）: 回避策（案 A `disable_event_tracking()`〈`unsafe fn`〉・案 B 生ポインタ launch 変種）はユーザー承認事項のため、実測エビデンスとともに承認依頼を起票し、承認が得られるまで本機構は「機構としては実装済みだが実機で capture が成立しない」状態のまま残す。

### 5.1 #1480 追補: 各 step の重み勾配比較の追加・GB10 実機実測

PR #1390 の codex-review P2 指摘（本文書 §3.1 項目 9 が引用する `crates/facade/tests/cuda_graph_step_common/mod.rs` の doc コメント「重み自体の勾配は引き続き含まない」節）が「対象外（切り出し先 Issue: TBD）」として引き継いだ事項をイシュー #1480 で解消した。

- **背景**: 当時は `Op::LinearResident.weight`（`ResidentLeaf`）の勾配をホストへ読み出す公開 API が存在せず、`Gradients::get()` を呼べる `Var` を外部から構築する手段もなかった（`ResidentLeaf` が意図的に `node_id` を非公開にしているため）ため、per-step パラメータ比較（更新後の重み）で間接的に代替していた
- **依存イシュー #1479（実装 PR #1492）**が `DeviceParamStore::resident_grads_to_host`（strict 版。resident staging 限定）／`param_grads_to_host`（統合版。resident 未充填 slot は `grads.get(...)` フォールバック）と `facade::Tape` の薄い委譲を追加し、この制約を解消した
- **本イシューの変更**: `cuda_graph_step_common::train_on_cuda` の backward 直後・`step_device_param_store` 前という呼び出し窓内で `Tape::param_grads_to_host(&store, &grads)` を呼び、`per_step_grads: Vec<Vec<Tensor<f32>>>` として `TrainOnCudaResult` へ追加した（5 要素タプルへ拡張）。`print_bit_identity_report` は `step[{step}].grad[{p}][{j}].bits` ラベルで出力し、`cuda_graph_step_bit_identity.rs`（`eager_baseline`／`graph_capture`）・`cuda_graph_step_two_gpu_bit_identity.rs`（`assert_eq!` による機械比較。重み勾配ブロックを新設）双方を追従させた
- **（記録時点の状態。#1559 以降は CUDA も resident staging に到達し、`assert_grad_readout_contract` の期待も `Some` へ更新済み。`docs/train-step-resident-parts-design.md` §7）CUDA では統合版 `param_grads_to_host` を使う**: CUDA は `gemm_fp32_strict_into` 未実装のため resident staging 経路（`GradStaging`）に到達せず、`Op::LinearResident` の重み勾配はホスト経路で `Gradients` へ書き込まれる。strict 版 `resident_grads_to_host` は CUDA で必ず `BackendError::Unsupported` を返す設計（`crates/facade/tests/device_param_store_backend_parity.rs::assert_grad_readout_contract` が別途検証）のため、統合版のみが全 slot を `Ok` で返す
- **本比較が検証する内容**: CUDA は resident 経路に到達しないため、本比較は「backward が計算した重み勾配そのもの（ホスト経路で `Gradients` へ書き込まれた値）が capture 経路・非 capture 経路で bit 同一」であることの直接検証であり、`GradStaging` の D2H 検証ではない（CPU のみ resident 経路が成立。`docs/device-resident-update-design.md` 追補 #1479 参照）。既存の `dinput`（d(loss)/d(x)）比較が backward の入力側端点を見るのに対し、本比較は backward のパラメータ側端点（SGD 更新の入力そのもの）を見る点で相補的である
- **GB10 実機実測（2026-09-09）**: `eager_baseline`（opt-in OFF）・`graph_capture`（opt-in ON）を 2 プロセスとして実行し、`step[...].loss.bits`／`step[...].dinput[...].bits`／`step[...].grad[...][...].bits`（新規）／`step[...].param[...][...].bits`／`final.param[...][...].bits` を突合した結果、**4782 行すべて bit 同一**（loss 10 + dinput 320 + grad 2120〈イシュー #1480 新規〉+ param 2120 + final 212。#1350 の 2662 行 + grad 2120 行）。`graph_capture_completes_training_loop_without_error`・`graph_capture_real_device`（既存回帰）も非後退で pass。#1479／#1492 が本イシューへ引き継いだ `device_param_store_backend_parity.rs::grad_readout_contract_on_cuda` も GB10 実機で pass 済み（strict 版が `Unsupported` を返すこと・統合版の勾配値がホスト参照実装〈`eval::matmul` ベース〉と一致することを検証）。実測ログは `docs/perf/logs/cuda-graph-step-grad-1480/`（`env_info.txt`・`eager_baseline_raw.log`／`graph_capture_raw.log`・`eager_bits.txt`／`graph_bits.txt`・`verify-cuda-graph-step-bit-identity.log`・`graph_capture_completes_training_loop_without_error.log`・`grad_readout_contract_on_cuda.log`・`graph_capture_real_device.log`）
- **本イシューは性能結線を含まない**（テスト・doc の追加のみ。update 区間の capture 機構自体は #1349／#1350 で完結済みで既定 OFF のまま不変）ため、`docs/perf` の before/after 5 回計測は対象外

## 6. #1350 への申し送り（完了。実測結果は `docs/perf/train-step-phase-breakdown.md` §16）

- 計測は `--phases` の `device_update` 区間に効果が閉じることを前提に、`FANDHE_AI_CUDA_GRAPH_STEP` の 3 状態（未設定＝OFF／`stream-only`／`1`）で比較し、「created stream の event 管理コスト」と「capture の効果」を分離して記録する
- `step_total` は中立見込み（update が step の 0.5〜0.7% のため）。後退が観測されても opt-in 既定 OFF のため結線撤回は不要
- launch 回数の比較は `SegmentRun::Captured`／`Replayed` の計数、または既存の launch カウンタ基盤を流用する
- **結果（GB10 実機実測・2026-09-08）**: `step_total` ON/OFF 中央値比 0.9552〜0.9644（事前宣言の ADOPT 基準 ≤0.95 未達）・`device_update` ON/OFF 中央値比は中央値 0.9719 と改善方向だが 5 run 中 1 run で符号反転（5/5 一貫の基準未達）・checksum は全セル完全一致。事前宣言基準により**既定 OFF 維持（中立）と確定**。`stream-only` 単独でも on/off 相当の改善が見られたため、`device_update` 改善の一部は capture 自体でなく created stream 化（F9 の (ii)）由来の可能性を示唆する。launch 回数自体（update 区間はカーネル 1 個のみ）は減らないという構造上の事前宣言も実測で確認した（詳細は同 §16）

## 7. リスクと安全側の判断

- ストリーム切替（§4.1）が最大のリスク。opt-in 時限定にすることで OFF 時の挙動・性能を現行と同一に保つ
- cudarc の自動 event 管理（ON 時のみ有効化される `cuStreamWaitEvent`）は capture と非互換であることが PR #1390 の codex-review 指摘（§3.2・§4.1 追記）で確定した。回避策（`disable_event_tracking()`）は unsafe 面の判断を伴うが、本クレートがこの `CudaContext` につき単一ストリームしか使わない前提が安全性根拠として成立するため実装・適用済み（§4.1）
- 別スレッドが同じ ordinal の `DeviceBuffer` を drop すると `CudaSlice::drop` が capture 中の共有ストリームへ非同期解放を発行しうる問題（codex-review P1 指摘。PR #1390）に対し、`context_cache::begin_buffer_release` を `CudaBufferHandle::Drop` から呼び、返した `BufferReleaseToken` を実際の解放が完了するまで保持することで対応した（`memory.rs`。§4.2「バッファ解放の排他」参照。P0 再指摘・Bugbot High デッドロック指摘を経て `capturing_active` サブフラグによる再修正済み）
- 「step 全体」を capture できない事実は起票時の想定との乖離であり、隠さず本文書 §0 に明記する

## 8. #3259 追補: 学習ステップ全体の capture 設計（決定記録）と所有者確認

親 #3258。#1350 の判定（§6。update 区間 capture は中立・既定 OFF）は**覆さない**。本節は「capture の範囲を step 全体へ広げる」別の候補を、実測前に設計として固定する記録である。`.rs`・`docs/spec`・tolerance・baseline・FMA 契約は変えない。

### 8.0 結論

- **(A) で確定**（所有者回答。§8.6）。step 全体の capture は**再 capture だけ**で実装し、新規 `unsafe` を足さない。opt-in・既定 OFF
- 既定の反転は #3261 の事前登録規則で ADOPT が出た後の別 PR とする
- §0 の「step 全体の capture は構造的に不可能」は #1349 当時の事実である。前提は #3252（`DeviceParamStore::train_step_device_chain` の区間内で同期・H2D・確保が 0。定常状態 0 件・初回 11 件を GB10 で実測。`docs/compat-resident-train-step-decision.md` §12）で揃った

### 8.1 判断材料

- **#1350（update 区間だけ）**: `step_total` ON/OFF 中央値比 0.9552／0.9644（ADOPT 基準 ≤ 0.95 未達）。`device_update` は中央値 0.9719 だが 5 run 中 1 run で符号反転。checksum は全セル一致。判定は中立・既定 OFF（`docs/perf/train-step-phase-breakdown.md` §16.7・§16.8）
- **推論チェーン capture（#2115。`FANDHE_AI_CUDA_GRAPH_INFER`。#3190 で実測）**: 2026-10-20 に GB10 で**実測済み・判定は undetermined**（`docs/perf/infer-chain-graphcapture-cuda-ab.md` §3）。
  - reuse の after/before 比 0.8610（batch 64）・0.9270（1024）・0.9761（4096）、fresh は 0.9243・0.9763・0.9955。checksum は全セル一致
  - R1(b) で `DeviceContextCaptureInProgress` が出て ADOPT 条件 5 を満たさなかった。#3205 の再判定（2026-10-23）も undetermined（`docs/perf/infer-chain-fixed-cost-reduction-3205.md` §10）
- **示唆（仮説）**: 固定費削減の効果はカーネル数が多い区間ほど大きい（reuse の batch 64 が最大）。step 全体は update 区間（カーネル 1 個）より launch がはるかに多い。ただし効くかどうかは実測で決める
- **推定値の扱い**: #3258 起票時の到達目安「約 0.10〜0.15 ms」は**推定**で根拠は薄い。**合格ラインには使わない**。起票時の現状値（train CUDA 0.331 ms・candle 0.272 ms）も起票時の値であり、本書では docs 出典を確認していないため判断に使わない。合格ラインは #3261 の RULE.txt が実測前に決める

### 8.2 capture の範囲と前提

- 範囲は `train_step_device_chain` が 1 区間に積む forward・MSE（mean）・backward（dW・bias 勾配・ReLU マスク・d_input）・SGD の全カーネル
- 区間の外: 入力の H2D、loss の `download`（`read_train_step_loss`。同期はこの 1 回だけ）、store のホスト側状態の更新（`step_count`・`sgd_used`・`train_chain_generation`）
- ウォームアップで定常状態にしてから capture する。条件は次の 4 つ
  - `TrainChainCaps` の全部品が probed
  - `TrainChainBufs` が確定
  - 永続ワークスペース（`CudaGemm::device_workspace`・`CudaMse::partial_workspace`）の grow が済み
  - NVRTC のコンパイルが済み
- capture 中の確保・grow・同期は §4.2 の同期点ガードで拒否される。§3.2 の「同期点ガードの適用範囲は update 区間 body が呼ばないものに限る」は、step 全体へ広げる際に再監査が要る。#3252 §12「窓内の隠れた確保・同期の監査」を #3260 へ申し送る
- 初回の momentum step は `is_first_step` がカーネル引数のため 2 step 目以降と graph が分かれる。初回はウォームアップ（capture しない）に回す

### 8.3 構造キー

- 型は `SegmentKey` と推論側 `run_captured_linear_chain` のキーにそろえる。構成は「デバイス世代 + `config_key` + 触れる全バッファのアドレスと長さ」
- 対象: params・勾配連結バッファ（`train_chain_grad`）・velocity、`TrainChainBufs` の層出力・dPred・d_input・mask 後勾配、loss スカラー、入力 x・目標 y、層構成（各層の `(k, n, act, bias 有無)`）
- **世代の取り違えに注意する**。世代は 2 種ある。
  - `context_cache::current_generation`（デバイス世代。invalidate からの回復で進む）
  - store の `train_chain_generation`（古い loss ハンドルを拒否するため**毎 step 進む**）

  キーに入れるのは**デバイス世代だけ**。`train_chain_generation` を入れると毎 step キャッシュミスになる
- **毎 step 新規 Tape**: facade の `train_step_resident` は `tape_for` を毎回作る（`compat-resident-train-step-decision.md` §14）。キーは Tape の同一性ではなく store の永続バッファで作る。Tape は `cached_device(ordinal)` 経由で同じ context・stream を共有するため順序は保たれる
- **x／y の再アップロードが前提になる**: 現行 `DeviceParamStore::upload_resident_input` は呼ぶたびに新しいバッファを確保する。step ごとに upload し直す使い方では x／y のアドレスが変わり、毎 step 再 capture になる。永続バッファへ `upload_into` で上書きする入口は無い。対処案は 2 つで、どちらも **#3260 の実装事項**として申し送る
  - (i) 永続の入力 staging を graph キャッシュ側に持ち、launch 前に stream 順序の D2D コピーで詰める（ホスト同期なし）
  - (ii) store 内部に `upload_into` で上書きする入口を足す

  公開形の追加・変更が要る場合は `compat-resident-train-step-decision.md` §4 に従い別承認とする。(i) は cudarc の安全 API だけで書けるかを #3260 で確かめ、新規 `unsafe` が要るなら止めて所有者へ戻す
- キャッシュは thread-local（`CudaGraph` は `!Send`。F4）。ordinal で分け、上限件数を持ち、世代が合わないものは evict する（推論側と同形）

### 8.4 lr などのスカラー（更新方式の比較）

SGD の `lr`・`momentum`・`dampening`・`weight_decay`・`nesterov`・`is_first_step` と MSE の scale はカーネル引数として graph に焼き込まれる。設定の変化は `fold_sgd_step_config_key` の `config_key` で区別する。

| 案 | 内容 | 新規 `unsafe` | 長所 | 短所 |
|---|---|---|---|---|
| (A) 再 capture だけ（**推奨・確定**） | `config_key`／構造キーが変わったら capture と instantiate をやり直す | なし | 既存の `begin_capture`／`end_capture`／`launch` の安全 API だけで書ける。#1349 と同型 | lr を毎 step 変えるスケジューラでは毎 step 再 capture になり逆効果になりうる（§8.5 の抑止で防ぐ） |
| (B) `cuGraphExecUpdate_v2` も使う | 引数だけ違う graph を exec update で差し替える | **要る**（cudarc では `sys` の生 FFI のみ。F3・spec 提案 H5／B-4） | lr スケジュールでも再 instantiate を避けられる | security-auditor の監査と所有者承認が要る。**未承認** |
| (C) 見送る | step 全体の capture を実装しない | なし | 変更なし | #3258 の候補が消える |

(A) では lr スケジュール下で効果が出ないことを明記する。(B) が必要になったら、実測を添えて所有者へ戻す。

### 8.5 opt-in・ストリーム・fail-safe

- **opt-in**: 新しい環境変数（例 `FANDHE_AI_CUDA_GRAPH_TRAIN`。名前の確定は #3260）と単一の const ゲート（既定 OFF）を設ける。`FANDHE_AI_CUDA_GRAPH_STEP`（#1349／#1350）は #1350 の判定対象と混ぜないため再利用しない。framework-compare の `bench-fandhe` は crates.io のピン版なので届くのは環境変数のみ（F8）
- **created stream とイベント追跡**: capture は legacy stream では不可（F1）。新しい opt-in を `device.rs` が created stream を選ぶ OR 条件（`step_graph_mode`・`infer_graph_enabled`）に加える。最初のデバイス初期化より前に設定が要り、ordinal ごとに固定される。`disable_event_tracking()`（§4.1。既存の `unsafe` で新規ではない）は適用済み。**既定を反転すると全 CUDA 利用者の既定ストリームとイベント追跡が変わる**（`infer-chain-graphcapture-cuda-ab.md` §1）ため、反転は ADOPT 後の別 PR で影響評価を含める
- **fail-safe**
  - キーを作れない・ウォームアップ未了・capture 不可の場合は `Ok(None)` 相当で、**capture しない既存のチェーン**へ進む（Tape 経路へは落とさず `Unsupported` を表に出さない）
  - capture 途中の失敗は `end_capture` で破棄する。エラーの意味は §4.7 と `train_chain.rs` のモジュール doc「失敗の境界」にそろえる。書き込み前の失敗は poison しない。replay の `graph.launch()` 失敗は書き込み後の失敗として poison する
  - 別スレッドの capture との衝突は #3205 の `wait_device_capture_idle`（待ち上限つき）と同じ扱いにする。推論側 R1(b) の欠陥を繰り返さないよう、並列テストを #3260 の受入に含める
  - 再 capture が続く場合に備え、キーが連続して変わったら capture をやめて既存チェーンへ戻す抑止を案とする。回数は #3260 で決める（ガードレール閾値には当たらない）
  - 失敗トークン（A08）は `_tracked` の登録と loss 読み出し後の再検査をそのまま使う
- **数値**: 同じカーネルを同じ順に再生するだけで FMA 契約と結合順序は変わらない。graph 経路と非 graph 経路は **bit 一致**を受入にする（#1350／#1480 と同じ 2 プロセス比較）。tolerance・baseline は変えない

### 8.6 所有者確認の転記

| 項目 | 内容 | 状態 |
|---|---|---|
| (A) 再 capture だけ・新規 `unsafe` なし・既定 OFF | 推奨 | **確定** |
| (B) `cuGraphExecUpdate` を使う | 必要になったら実測を添えて所有者へ戻す | **未承認** |
| (C) 見送り | 採らない | - |
| 既定の反転 | 事前登録規則の ADOPT が前提 | #3594 で一度反転を試みたがチェーン外の capture 衝突が未解決のため見送り（既定 OFF 維持。§8.12） |
| 公開形の追加・新規 `unsafe` | 承認範囲外 | **未承認** |

- 出所: イシュー #3259 の所有者コメント <https://github.com/Fandhe-AI/fandhe-ai/issues/3259#issuecomment-6111775631>（ID `IC_kwDOTuUCJc8AAAABbEpLjw`）。`createdAt` は 2026-10-11T17:35:55Z（UTC。JST では 2026-10-12 02:35。コメント本文の記載日は 2026-10-12）。判断待ちの issue について「全て推奨で」と回答し、本 issue は (A) で確定
- tolerance・baseline・FMA 契約は不変

### 8.7 申し送り

- **#3260（既定 OFF で実装）**: x／y の上書き入口（§8.3）、同期点ガードの再監査（§8.2）、並列 capture の衝突テスト、新 opt-in と created stream の OR、`crates/backend-cuda/src/graph.rs` 冒頭 `//!`（「update 区間のみ」）の更新（本 issue は `.rs` を変えないためここでは直さない）。`unsafe` を足す場合は security-auditor 監査必須（足さないのが前提）
- **#3261（GB10 A/B）**: 実測前に RULE.txt を固定する。合格ラインはそこで決め、推定値は使わない。#1350 の基準（中央値比 ≤ 0.95・5/5 符号一貫・checksum 一致）は前例として引くのみ。bit ダンプの diff を含める


## 8.8 #3260 実装記録: 学習ステップ全体の capture（既定 OFF）

親 #3258。§8 の (A) 再 capture だけ・新規 `unsafe` なし・既定 OFF で実装した。**性能の評価・既定の反転は #3261 の範囲**で、本節は機構の実装と、bit 一致・fail-safe の実機確認だけを記録する。tolerance・baseline・FMA 契約・`docs/spec` は変えていない。

### 8.8.1 構成

| 項目 | 実装 |
|---|---|
| opt-in | 環境変数 `FANDHE_AI_CUDA_GRAPH_TRAIN`（`1`／`true` の完全一致だけ ON。それ以外は fail-closed で OFF。値はエコーしない）。const `TRAIN_GRAPH_DEFAULT_ENABLED = false`。**API の setter は持たない**（§8.6: 公開形の追加は未承認）。`facade` は変えず、`api_surface.rs` が再公開 0 件を固定する |
| stream | `device.rs` の created stream 選択に `train_graph_stream_requested()` を OR で追加。ordinal ごとに固定され、最初の CUDA 初期化より前の設定が要る。`internal-diagnostics` ビルドでは stream が Legacy に固定され capture に届かない（実機テストは既定 feature で実行する） |
| trait | `tensor-core::BackendOps::train_chain_captured(TrainChainCapturePlan)`（既定 `Ok(None)`。非破壊拡張）。`plan` は触れる全バッファを引数で受ける固定形で、任意クロージャは使わない（§4.3 の codex P0 の前例） |
| CUDA 本体 | `backend-cuda/src/graph.rs::launch_train_chain`。`autodiff` の `encode_train_chain` ＋ `sgd_step_device_tracked` と**同じ順・同じ引数**で `*_into_tracked` 部品を呼ぶ（カーネルも引数も同じなので bit 一致する）。順序を変えるときは両側を直す旨を双方のコメントに書いた |
| 結線 | `autodiff` の `DeviceParamStore::try_captured_train_step`。部品ごとの encode の前に 1 回試す |
| キー | `SegmentKey { generation（デバイス世代のみ）, config_key, resources }`。`config_key` は x／y の形状・各層の `(w_offset, k, n, bias_offset, relu)`・MSE の係数・SGD の全設定（f32 は `to_bits`）・velocity の有無を `DefaultHasher` で畳み込む。`resources` は params・grad・velocity・loss・層出力・dPred・d_input・mask 後勾配・x・y のアドレスと長さ。毎回、その呼び出しで借りている live なバッファから計算する |
| 診断 | `TrainGraphStats { captured, replayed, graph_launches, fallbacks }`（推論側 `InferGraphStats` と別型）と `train_graph_stats()` |

### 8.8.2 判断の記録

1. **x／y の扱い（§8.3 の申し送りへの回答）**: x／y のアドレスをキーに入れ、graph は呼び出し元のバッファを直接読む。step ごとに upload し直す使い方ではキーが外れて再 capture になるが、下の抑止で既存チェーンへ戻り、結果は bit 一致のまま保たれる。理由は 3 つ。(1) 計測ランナー（`docs/perf/logs/resident-chain-ab-3257/bench-resident-chain.patch`）は x／y を窓の外で 1 回だけ upload する。(2) 案 (i)（graph が持つ staging へ D2D コピー）は step ごとに `memcpy_dtod` が 2 回増え、固定費削減を食う。(3) 案 (ii)（公開形の追加）は承認範囲外。**案 (i) は `memory.rs::CudaStorage::copy_from_dtod`（cudarc の安全 API `memcpy_dtod` を包むもの）で新規 `unsafe` なしに書ける**ことを確認した。#3261 の結果しだいで後続の候補にする
2. **capture 前の実 launch はしない**: 推論側 `capture_chain` は capture 前に本体を 1 回実 launch するが、SGD は冪等ではない（2 回更新になる）ため写さない。capture は記録だけで実行しないので「capture ＋初回 launch」でちょうど 1 step になる。モジュールロード・永続ワークスペースの grow は、適用条件（下記）でウォームアップ済みの 2 step 目以降に限ることで済ませる
3. **適用条件（autodiff 側。§8.2 のウォームアップ）**: 成功 step を 1 回以上経ている（`is_first_step == false`）・前 step の永続バッファ一式を鍵つきで再利用できる・勾配バッファと velocity を新規確保していない・使う部品の `_into` 対応が確定済み（単層は forward・MSE・MSE backward、多層は mask・d_input も）・配列の本数が層数とそろっている。満たさなければ従来のチェーンがそのまま走る
4. **`Unsupported` を外へ返さない**: autodiff／facade は書き込み前の `Unsupported` を Tape 経路への全体フォールバックと解釈するため、同期点ガード・ワークスペース grow の拒否〈`CaptureExclusionRejected` 由来〉を含め、launch 前の失敗はすべて `Ok(None)` に変える。その際 ordinal が poison 済みなら型付きエラーで返す
5. **失敗の分類**: launch 前の失敗（本体のエラー・panic・`end_capture` 失敗・空 graph・`upload` 失敗）は graph を捨て `Ok(None)`（`fallbacks` と抑止の失敗数を増やす）。`DeviceContextCaptureInProgress` は launch 前の衝突として表へ返し、`train_step_device_chain` が待って step 全体をやり直す。初回 launch の失敗と replay の失敗は書き込み後の失敗として表へ返し、store を poison する（replay の失敗時は既存 graph を戻す）
6. **別スレッドの capture との衝突（R6）**: `train_step_device_chain` に待ち上限つきの再試行ループを置いた（`predict_device_chain` と同型。上限は 1000 ms で `PREDICT_CAPTURE_WAIT_TIMEOUT` と同値。テストでは 30 ms）。再試行が安全なのは、書き込み前の衝突と SGD 段の衝突がどちらも params を変えず（既存の「poison しない」判定の範囲）、勾配バッファと loss は次の試行が全要素を上書きするため。capture が 1 つも無ければこのエラーは起きず、ループは 1 回で抜ける
7. **再 capture の抑止（§8.5 の「回数は #3260 で決める」への回答）**: スレッド・ordinal ごとの状態機械 `TrainCaptureThrottle`。連続の新規 capture（間に replay が無い）が **3 回**で、replay が起きるまで新規 capture を止める。capture 準備・記録・終了の失敗が **2 回**に達したら、そのスレッド・ordinal では新規 capture を止める。止まっても既存 graph の replay は続ける。性能の抑止パラメータでありガードレール閾値・tolerance ではない。lr を毎 step 変えるスケジューラでは 3 step で既存チェーンへ戻る（(A) の既知の短所）
8. **GEMM 精度**: `Fp32Strict` 以外では `Ok(None)`（replay だけ FP32 で走り非 graph 経路と食い違うのを防ぐ）

### 8.8.3 同期点ガードの再監査（§8.2・§8.7）

| 対象 | 結果 |
|---|---|
| `linear_forward_device_into_tracked`・`gemm_resident_lhs_device_core`・`relu_backward_mask_device_core`・`mse_loss_device_core`・`mse_loss_backward_device_core`・`sgd_step_device_tracked` | alloc・upload・download・同期を呼ばない。`with_sync_point_call` を通らない（`ops.rs` の使用箇所は host 入力の `gemm_fp32_strict_into` NN/TN 経路と `synchronize_device` だけ） |
| `gemm_fp32_strict_into_device_impl`（dW・bias 勾配） | `with_driver_call` ＋ `cached_gemm`／`cached_reduce` ＋ `launch_gemm_dw_device` だけ。ホスト入力版の関数内同期は無い。ワークスペースの grow は `workspace_slot` が capture 中に `CaptureExclusionRejected` で拒否する |
| `mse.rs::launch_mse_loss_device_f32` の `partial_workspace` 初回確保 | **ガードが無かった**。`is_capturing_on_current_thread` なら `CaptureExclusionRejected` で拒否するよう是正した（`gemm.rs::workspace_slot` と同型。定常状態は確保済みで経路は変わらない） |
| `cached_*` の getter | キャッシュがヒットすれば driver のモジュールロードは起きない。capture の前に独立した `begin_driver_call` 境界で確認する（ウォームアップ済みの適用条件で保証） |

実機の確認: 全 6 組み合わせ・10 step で `fallbacks == 0`（capture 中に拒否された操作が無かった）。replay 中の step の窓で `stream_syncs`・`d2h_calls`・`h2d_calls`・`mem_uploads`・`mem_allocs` がすべて 0。

### 8.8.4 実機確認（GB10・`spark-dbd9`・既定 feature・`--release`）

ログは `docs/perf/logs/train-step-graph-3260/`（`env_info.txt`・`.rev-stamp`）。`.rev-stamp` は転送元 HEAD ＋ 未コミット件数で、実装コミット前の作業ツリーを転送して実行した。

| 検証 | 結果 |
|---|---|
| `train_step_graph_diag_tests_3260`（7 件。`FANDHE_AI_CUDA_GRAPH_TRAIN=1`・`--test-threads=1`） | 7 passed（`diag_all.log`） |
| 同一プロセスで capture あり／なしを 10 step 比較（784-256-10・batch 64・momentum。loss と全パラメータ bit 一致）＋ stats | captured 1・replayed 8・graph_launches 9・fallbacks 0 |
| facade `train_step_resident` 経由・6 組み合わせ（2 層／3 層 × plain／momentum／nesterov）・各 10 step の bit 一致 | passed。組み合わせごとに capture 1・replay 8（`facade_in_process.log`） |
| 2 プロセス比較（環境変数なしの Legacy stream と `=1`）の `BITS` 90 行 | diff 空（`bits_off.txt`／`bits_on.txt`）。ON 側 stats は captured 6・replayed 48・fallbacks 0 |
| fail-safe: capture 失敗の注入（1 回・上限到達） | 1 回: fallbacks 1・次の step で capture・bit 一致。上限（2 回）到達後は capture を試さず既存チェーンで bit 一致 |
| lr を 1 回変更 | 再 capture（captured 2・replayed 5）・bit 一致。毎 step 変更は captured 3 で止まり replay 0・bit 一致 |
| x／y を毎 step upload し直す | bit 一致（プールのアドレス再利用により captured 4・replayed 4。キーが外れた分は再 capture）。結果は不変 |
| 並列: 2 スレッドが別 store で capture ON のまま 20 step（1 step 目後に柵） | エラー 0・単独実行と bit 一致。30 回繰り返して 30/30 passed（`two_threads_x30.log`） |
| replay 中の窓 | 同期・D2H・H2D・確保がすべて 0。loss の読み出しは同期 1・D2H 1 |
| 既存の回帰（opt-in なし／各 opt-in） | `train_step_chain_diag_tests_3252`・facade `resident_train_step` の CUDA 5 件・`graph_capture_real_device`・`graph_capture_real_device_optin_off`・`infer_graph_capture_real_device` がすべて pass（`regression.log`） |

**未検証・制約**
- facade の `Sequential` は bias なしの Linear を作れないため、bias なしの経路は公開面からは到達できず、実機では確認していない（キーへの反映はホストの単体テストで確認）
- 性能（launch 固定費の削減の有無）は未測定。#3261 が RULE.txt を実測前に固定して測る
- **既知の制約**: 通常チェーン（capture しない step）の途中で他スレッドの capture が始まり、最初の dW の書き込み後に `DeviceContextCaptureInProgress` が出ると、既存の契約（`train_chain.rs`「失敗の境界」）どおり store が poison される。並列テストは各スレッドの 1 step 目の後に柵を置いてこの窓を避けている。この契約の見直し（SGD 前の衝突は params 未変更のため再試行できる可能性）は #3260 の範囲外

### 8.8.5 申し送り（#3261）

- A/B の「after」は `FANDHE_AI_CUDA_GRAPH_TRAIN=1`、「before」は未設定。x／y を窓の外で 1 回だけ upload する計測ランナーでは replay が続く。step ごとに upload し直すランナーでは再 capture が走り、3 回で既存チェーンへ戻る
- 実測前に RULE.txt を固定する（§8.7）。bit ダンプの diff は `cuda_graph_train_step_bit_identity.rs::dump_bits_for_two_process_diff` を使える

## 8.9 #3261 GB10 A/B の結果（学習ステップ全体の capture）

事前登録規則 `docs/perf/logs/train-step-graph-ab-3261/RULE.txt`（sha256 `d0ec1c0a…`）に従い、`FANDHE_AI_CUDA_GRAPH_TRAIN` なし（off）／`=1`（on）を GB10 の train CUDA（size=64・TF32 OFF）で 5 round 計測した。生ログ・JSONL・前提ゲートのログは `docs/perf/logs/train-step-graph-ab-3261/gb10/`（ホスト名・ホームパスをマスク済み）、判定出力は同ディレクトリの `judge-3261.md`／`judge-3261.exit` にある。

### 8.9.1 実施条件

| 項目 | 内容 |
|------|------|
| 実施日・ノード | 2026-10-25（UTC。実機ロック取得 22:55Z・解除 23:07Z）・GB10 ノード `spark-939e`（GPU 使用率 0%・load1 < 1.0。実ホスト名は `.local.md` のみ） |
| runner | `6a97087ac7e29381eb932584783dc2495bd889e5`（`.rev-stamp` 2 行目 `dirty=0`）に `bench-resident-chain.patch`（sha256 `92d0a8d4…`）だけを適用。on／off は同一バイナリ（`bin_sha256` 1 つ） |
| torch | 2.14.0+cu130（専用 venv。`gb10/torch-version.txt`）。candle は同ハーネスの `candle-core =0.11.0` |
| 専有ゲート | 5 round すべて `gate=pass`（load1 0.09〜0.26）。`skipped` ログは空 |

### 8.9.2 判定: **ADOPT**（`judge_train_graph_ab.py` の終了コード 0）

| セル | off 中央値 ms | on 中央値 ms | ratio | ratio_r（5 round） | 相手最速÷off | 相手最速÷on |
|---|---|---|---|---|---|---|
| chain（判定） | 0.1347 | 0.1126 | 0.836 | 0.839, 0.839, 0.836, 0.833, 0.833 | 2.03x | 2.42x |
| reuse（対照） | 0.2769 | 0.2625 | 0.948 | 0.953, 0.958, 0.947, 0.959, 0.942 | 0.99x | 1.04x |
| fresh（対照） | 0.5058 | 0.4915 | 0.972 | 0.972, 0.922, 0.972, 0.936, 1.045 | 0.54x | 0.56x |

- ADOPT 条件 1〜5 をすべて満たす: 専有ゲート全通過・R0〜R2 合格（R0: 3＋2＋1 passed、R1: facade 1 passed・backend-cuda 8 passed、R2: 2 プロセスの BITS 差分が空。全ログ `rc=0`）・hard 条件（chain・reuse・fresh とも checksum 0.080541 が全 run で一致、chain・reuse は PARAM_DIGEST も一致〈chain は `fnv1a64=0xfb061246845ce46f`〉）・chain の ratio 0.836 ≦ 0.95 かつ 5/5 round で ratio_r < 1.00・reuse／fresh の ratio ≦ 1.00。
- 相手比は同一 round の candle・torch の最速（本計測では全 round で candle）を fandhe の 5 round 中央値で割った参考値で、判定には使っていない。candle の中央値は 0.2703〜0.2746 ms（torch は 0.998〜1.601 ms）。
- #3257 の基準（chain 0.1343 ms・candle 比 2.02x）に対し、off の chain は 0.1347 ms・2.03x で再現した。on で 0.1126 ms・2.42x。
- 事前表明した到達目安 0.10〜0.15 ms は根拠の薄い**推定**であり合格ラインではない。on の 0.1126 ms はこの範囲に入ったが、判定には使っていない。
- 帰属: chain の改善が capture の効果。reuse・fresh は bench の経路が `train_step_device_chain` に届かないため、効果は created stream 化とイベント追跡の無効化（stream 変更）だけによる。fresh の ratio_r には 1.045 の round が 1 つあり（5/5 符号一貫は fresh には課していない）、中央値では中立〜わずかな改善である。

### 8.9.3 既定は反転していない

- ADOPT でも `TRAIN_GRAPH_DEFAULT_ENABLED` は `false` のまま。既定の反転は §8.6 で所有者判断（未承認）とされ、全 CUDA 利用者の stream・イベント追跡の変更と、lr スケジューラ下で 3 step 後に既存チェーンへ戻る点の影響評価を伴う別 PR を要する。
- #3261 の本文は ADOPT 時に反転 PR を求めていたが、後に出た所有者の指示と本節の決定記録（§8.6）を優先し、本件では反転しない。
- tolerance・baseline・FMA 契約・依存ピンは変更していない（コード変更なし）。
- → 影響 2 点の評価と、反転を見送った理由は §8.12 に記録した。

### 8.9.4 限界

- bench からは capture が replay されたかを観測できない。R1（facade の bit 一致テストと backend-cuda の 8 テスト）で結線を補っているが、R1 の形状は bench の形状と同一ではない。
- checksum の一致は bit 同一性の証明ではない。PARAM_DIGEST と R2（2 プロセスの bit 比較）で補っている。
- 単一機体（GB10）・train CUDA・size=64 の結果で、infer・CPU・Metal・他形状には及ばない。正式な 6 フレームワーク同席計測は #3273。

## 8.10 #3594 既定 ON への反転の保留（同一 ordinal の並行学習との競合）

- 既定 ON への反転は、同じ ordinal で別スレッドが capture を開始すると、非 capture の学習チェーンが最初の dW 書き込み後に `DeviceContextCaptureInProgress` を受け、`train_chain.rs` の encoded エラー処理（`written=true`）が store を poison して復旧できなくなる競合（§8.8.4 の既知の制約）を、既定の利用者へ広げる機能回帰になる（PR #3595 の codex-review P1 指摘）。
- 解消案は、チェーン全体の実行と capture を ordinal 単位で排他すること（non-capture チェーンが capture 中のコンテキストに入らない、または capture 開始側がチェーン完走を待つ）。GB10 実機での検証が必要なため本 PR の範囲に収めない。
- よって `TRAIN_GRAPH_DEFAULT_ENABLED` は `false` を維持する。反転は、ordinal 排他の実装と「同一 ordinal で 2 スレッド同時学習しても store が poison されない」回帰テストを付けた後続 PR で行う。
- tolerance・baseline・FMA 契約・依存・`unsafe` は変更していない（コード変更なし）。
- → §8.11 の排他だけでは足りず、反転は §8.12 の未解決事項のため見送った。

## 8.11 #3596 学習チェーンと Graph capture の ordinal 単位の排他

§8.8.4 の既知の制約と §8.10 の保留（#3594）を解消するための実装記録。`TRAIN_GRAPH_DEFAULT_ENABLED` は `false` のまま（反転は #3594）。

### 8.11.1 競合経路の洗い出し（R1）

| 経路 | 複数回の driver 呼び出し | 書き込み後に `DeviceContextCaptureInProgress` を受けたとき | 対処 |
|---|---|---|---|
| `train_step_device_chain` の encode（`encode_train_chain`） | はい | store を poison する（不具合。§8.8.4） | **本件のスコープで排他し、書き込み後には起きなくする** |
| 同経路の SGD 段（`sgd_step_device_tracked`） | 1 回 | poison しない（既存の分岐） | スコープ内では起きない。防御として分岐は残す |
| `try_captured_train_step`・replay | — | launch 前に返る（何も書かない） | 変更なし |
| `predict_device_chain(_resident)` | はい（読み取りのみ） | store を変えず、待って再試行する | 変更なし |
| `step()`・`step_adam_impl` 等の更新経路（`device_store.rs`） | upload → update | 保留分を戻して poison せずに返す（再試行は呼び出し元） | 変更なし。呼び出し元の再試行が前提 |
| reuse 経路・fresh 経路・Tape の通常演算 | 演算ごと | 型付きエラーで返る。**勾配の書き込み後に拒否されたときの store・Tape の扱いは今回の範囲で確認していない** | 未確認。必要なら別 Issue の候補（起票はユーザー承認後）。追跡は #3608・§8.12.5.5 |
| graph の replay・初回 launch と非 capture 処理が ordinal 共有の永続ワークスペース（GEMM transposed／staged・MSE partial）を取り合う経路（データ競合。`DeviceContextCaptureInProgress` ではない） | — | 他 step の値を読む（loss の取り違え） | **#3596 再オープン後に対処**（§8.11.6） |
| `begin_buffer_release`・`invalidate_with` | — | — | §8.11.3 のとおりデッドロックしない |

capture モードは 3 経路（step・推論・学習）とも `CU_STREAM_CAPTURE_MODE_THREAD_LOCAL`。このモードは他スレッドから共有 stream へ起動することを防がないため、排他はホスト側（`begin_driver_call`。§4.2 参照）で担保する。本件はその担保をチェーン全体へ広げるもので、モードの選択は変えない。

### 8.11.2 設計

- 新しいロックや状態フィールドは足さない。非 capture チェーン（encode ＋ SGD）の間、**外側の `CallToken` を 1 つ持ち続ける**。`BackendOps::begin_device_chain_scope`（tensor-core。既定は中身のないスコープで CPU・Metal は不変）が `DeviceChainScope`（`!Send`・RAII）を返し、CUDA 実装（`ops.rs`）が `begin_driver_call` のトークンを包む。
- チェーン内の各部品は「このスレッドが in-flight を持つ」通過条件で通る。別スレッドの `begin_capture_session` は既存の `in_flight` ドレインでチェーンの完走を待つ。capture 開始後に来た新しいチェーンはスコープ取得（書き込み前）で一過性エラーを受け、`train_step_device_chain` の待機・再試行ループが処理する。
- スコープは `try_captured_train_step` が `Ok(false)` を返した**後**に取る（スコープを持つスレッドは capture を開始できないため）。取得失敗は書き込み前の失敗として扱い、`Unsupported`・一過性衝突は poison せず velocity の新規確保分を戻す。それ以外は従来どおり poison する。
- legacy stream（既定 OFF。`is_capturable_stream() == false`）の ordinal では capture 経路が成立しない（3 経路とも capturable stream を前提）。Legacy と確定済みの ordinal はスコープ取得の時点で driver に触れず素通りにする（§8.11.8。r3 まではトークンを取ってから手放していた）。デバイス同期は足さない。新しい `unsafe` はない。
- `DeviceChainScope` は内部クレート用で facade には出さない。`mem::forget` してはならない（in-flight が戻らず capture・invalidate が待ち続ける）。

### 8.11.3 デッドロック分析

| 状況 | 結果 |
|---|---|
| capture 自身がチェーンを実行する | `launch_train_chain` は部品を直接呼びスコープを取らない。autodiff はスコープの外で capture を試す。万一スコープを持ったまま capture を始めても `begin_capture_session` が `InvalidArgument` で即座に拒否する（CI テストで固定） |
| 再 capture（同じスレッド・step ごと） | 前 step のスコープは drop 済みで待たない |
| スコープを持ったまま待機 | `wait_foreign_capture_idle` は即座にエラー（CI テストで固定）。再試行ループはスコープの外 |
| ドレイン中のチェーン側の buffer 解放 | ドレイン中は capture が有効でないため `begin_buffer_release` は止まらない（既存） |
| ドレイン中のチェーン内の入れ子 `begin_driver_call` | 通過条件で通る（既存テスト `begin_driver_call_allows_nested_call_...`） |
| `invalidate_with` | スコープを持つスレッド自身から `invalidate` へ届く経路はない（呼び出しは context_cache 外では公開 API のみ）。他スレッドの `invalidate` はチェーン完走後に進む |
| capture 側の飢え | capture 開始後に新しいチェーンは入れない。待つのは既に走っている 1 step 分だけ |
| エラー・panic | RAII。`?` の早期 return と `catch_unwind` を CI テストで確認 |

### 8.11.4 テストと実測

- CI（実行済み）: `context_cache.rs` の `chain_scope_*` 4 件、`train_chain.rs` の mock テスト 4 件、`backend_ops.rs` の既定実装テスト。
- GB10 `#[ignore]`: `unfenced_chain_and_capture_threads_do_not_poison_and_stay_bit_exact`（`train_step_graph_diag_tests_3260.rs`。柵なし・30 繰り返し）を追加。
- **未実測（本記録時点）**: GB10 での上記テストの pass、backend-cuda・facade の `#[ignore]` 全件の main との突き合わせ、1 スレッド chain セルの排他あり・なし比較（既定 OFF・opt-in ON）。実機を使えない環境で実装したため、別の機体の値では埋めない。#3594 の前提（これらの実測）が揃うまで `Closes` は付けない。

### 8.11.5 GB10 実測結果（2026-10-26。#3596 再オープン後）

- 出典: `docs/perf/logs/train-chain-exclusion-3596/`（RULE.txt・judge-3596.md・gb10/）。機体は GB10（driver 580.173.02）、before `f7fe6b4df`・after `087e7f6d4`（`crates/` は `d3f38f71b` と同一）。§8.11.4 は記録時点の履歴として残す。
- **G1 不合格**: `unfenced_chain_and_capture_threads_do_not_poison_and_stay_bit_exact` は 8 回中 7 回失敗した。poison は起きていないが、非 capture スレッド（thread 1）の loss 列が単独実行の参照と一致せず、別 step の loss らしい値が読める（例: `loss idx 18: 1.2480911 vs 0.6393652`）。§8.11.3 の分析で捕捉できていない経路が残っていると推定されるが、原因は未特定（本 Issue の範囲外）。
- **`#[ignore]` 突き合わせ**: after にだけ出た fail は G1 のみ（補足実行のフレーク 1 件を除く）。共通の失敗（TF32／f16 の parity・stress・probe 系など）は排他と無関係の既存事項。詳細は `gb10/ignored-compare-3596.md`。
- **chain セル非後退: 未計測**。ランナーの前提ゲート g2（G1 を含む）が失敗し、規則どおり停止した。g3・g4（2 プロセス bit 一致）は合格。判定ツールの結果は REGRESSION（終了コード 3）。
- 限界: 1 機体・1 日の結果。G1 の確率的な失敗は件数が少なく、頻度の推定は粗い。コード・閾値・規則は変更していない。
- 結論: 排他（#3597）だけでは並行時の結果取り違えを防げていないため、#3594（既定 ON 反転）の前提はさらに遠のく。所有者判断: 原因調査と修正方針、別名 RULE での再実測。

### 8.11.6 原因の切り分け（G1 の loss 取り違え。2026-10-26）

**状態: コード上の原因特定と修正は済み。GB10 での確認は §8.11.7 のとおり実施済みで、修正の効果は確認できたが、下記の機構の推定は実機の `DIAG3596` と一部食い違った**（下記の本文は静的な読解に基づく推定として残す）。

- 実機での確認（2026-10-26。RULE-r2 [原因切り分け]。ログは `docs/perf/logs/train-chain-exclusion-3596/gb10-r2/diag/`）: 診断テストは #3597 以前の `f7fe6b4df`（テスト専用パッチ付き）でも、修正前 main の `c1204755e`（同）でも落ちた（各 2/2 回）ので、#3597 以前からある欠陥という推定は支持された。パラメータは一致し（`params_equal=true`）、ずれるのは非 capture スレッドの loss だけという点も一致した。一方で、**ずれた loss は参照列のどの index とも bit 一致しなかった**（`equals-reference-idx []`。例: 期待 1.1649379 に対し 0.22982085）。「相手 step の loss を読む」という推定のとおりなら別 index と一致するはずなので、機構の記述は未確定として扱う（RULE-r2 に従い、修正の機構は確定させない。所有者判断事項）。修正後（`616be4ff7`）は 5/5 回 pass。

- 機構（推定）: `CudaMse` は ordinal ごとに 1 つで、デバイス上の永続 `partial` ワークスペースを持つ（`mse.rs`）。非 capture の forward は、このロックを partial と finalize の 2 起動にまたがって保持することで安全になっている。一方、学習 graph（`launch_train_chain` の capture 結果）は同じ `partial` のアドレスを焼き込むが、replay（`run_captured_train_chain` ①）も初回 launch も GEMM の `device_workspace` mutex しか取らず、MSE のロックを取っていなかった。2 スレッドは同じ ordinal の同じ stream に積むため、「非 capture の partial → graph（自分の partial と finalize）→ 非 capture の finalize」の順になりうる。その場合、非 capture 側の finalize は graph が書いた partial を読み、相手スレッドの step の loss を自分の loss として書く。
- ログの症状との一致: 失敗は非 capture スレッドだけ（graph は stream 上で一体に実行されるので割り込まれない）。値は参照列の別 index の値になる（相手は同じ初期値・同じ入力）。poison は起きない。パラメータは影響を受けない見込み（MSE backward は pred／target から再計算し、dW のワークスペースは GEMM mutex で守られる）。
- #3597 との関係: #3597 が防ぐのは「capture の開始」とチェーンの衝突であり、「replay」とチェーンの衝突ではない。したがって #3260 からある既存の欠陥を、柵なしの G1 が初めて露出させたと推定する。確定には `train_graph_shared_workspace_tests_3596`（capture の開始を重ねず replay と非 capture チェーンだけを衝突させる診断テスト）を #3597 以前のツリー（`f7fe6b4df` に診断テストのみ足したもの）で実行する（RULE-r2 の [原因切り分け]。未実測）。
- テスト側の原因の否定材料: 参照は `capture=false` の `run_scenario` で作り、初期値・入力は固定パターンで乱数がない。capture 側スレッドは 30 回とも参照と一致している。loss はスレッドごとの store から step 直後に同スレッド内で読んでいる。確定は新しい `DIAG3596` 行（ずれた loss が参照列のどの index と一致するか、パラメータが一致するか）で行う。
- 永続ワークスペースの監査: `Mutex<Option<CudaSlice` 系の ordinal 共有領域は GEMM `device_workspace` と MSE `partial_workspace` の 2 つ。推論 graph・step graph（#1349 の SGD）が同様の共有領域に触れるかは #3609 で確認した（触れない。下記「永続ワークスペースの監査の追補」）。
- 修正: `graph.rs::with_train_graph_workspaces_locked` が、replay と初回 launch を GEMM → MSE の順で両方のロック下で行う（`CudaMse::with_partial_workspace_locked`）。**ロック順は GEMM → MSE に固定**。非 capture の経路はこの 2 つを入れ子で取らず、graph の launch は積むだけで他スレッドを待たないため、待ちの輪はできない。世代不一致・mutex 汚染・launch 失敗の扱いは従来のまま。capture する部品に新しい永続ワークスペースを足したら、このヘルパーにロックを足す。1 スレッドでの追加費用は競合のない mutex 1 回で、同期・確保・driver 呼び出しは増えない。
- 却下した代替案: store ごとの partial バッファ（tensor-core の trait・autodiff・メモリ量に波及する）。チェーン全体と replay を ordinal ロックで直列化する案（必要以上に待つ）。`cuGraphExecUpdate`（未承認・新しい `unsafe`）。
- 新しい `unsafe` なし。tolerance・baseline・依存・FMA 契約は変更なし。

#### 永続ワークスペースの監査の追補（#3609。推論 graph・step graph）

**結論: 推論 graph・step graph は ordinal 共有の永続ワークスペースを焼き込まない。#3601 と同じ種類の取り違えは、この 2 経路では起こらない。** 静的な読解のみで、コードは変えていない（基準 `c52a9042e`。行番号は `crates/backend-cuda/src/` 配下）。GB10 での実測はしていない（下記「限界」）。

**capture の入口は 3 つだけ**（`begin_capture(` の grep）: step graph `graph.rs:549`・推論 graph `graph.rs:1013`・学習 graph `graph.rs:1593`。学習 graph は #3601 の修正済みで本節の対象外。

**焼き込むデバイスアドレスと所有の単位**

| graph | 焼き込むアドレス | 所有者 | 分類 | 根拠 |
|---|---|---|---|---|
| step（#1349） | `param`・`grad`・`velocity`（`None` のときは `grad` を再 push） | 呼び出し元の `DeviceParamStore` | インスタンス | `sgd.rs:117` の `run`。`push` 順は同関数の launch ブロック。`SegmentKey` の再検証は `ops.rs:3822`・`ops.rs:1255` |
| step | SGD のスカラー（lr 等） | カーネル引数の値 | 該当なし（アドレスではない） | 同上 |
| 推論（#2115） | `input_staging`・`acts[i]` | thread-local の `INFER_GRAPHS` が持つ `CachedInferChain` | thread-local | `graph.rs:807〜823`（構造体・`thread_local!`） |
| 推論 | 各層の `w`・`bias`（`bias` が無いときは `a` を再利用） | 呼び出し元 | インスタンス | `graph.rs:885`〜（`launch_chain_layers`）→ `gemm.rs:6408`（`launch_tiled_bias_act_f32_resident`）。key の生成は `ops.rs:3927〜3986` |

どちらの capture 本体も、確保・Mutex・ワークスペースを持たない。step は `STEP_GRAPHS`（`graph.rs:318〜339`）が thread-local。`CudaSgd` のフィールドは `stream` とコンパイル済み関数ハンドルだけ（`sgd.rs:54`）。推論の launch は引数（`a`・`w`・`bias`・`c`）とスカラーだけで起動する（`gemm.rs:6408` 以降）。`kernels.rs`・`kernels_sgd.rs` にモジュールスコープの `__device__`／`__constant__` 変数は無い（grep）。

**ordinal 共有の領域と入口**（#3601 で確定した 2 つ）

- GEMM `device_workspace`（`gemm.rs:519`）: ロックを取る入口は `lock_device_workspace`（`gemm.rs:5750`）経由の `with_workspace_epoch`（`gemm.rs:5743`）・`launch_gemm_dw_device`（`gemm.rs:5818`）・`launch_gemm_dinput_device`（`gemm.rs:5910`）の 3 つだけ。
- MSE `partial_workspace`（`mse.rs:184`）: 入口は `launch_mse_loss_device_f32`（`mse.rs:425`、ロックは `mse.rs:435`）と `with_partial_workspace_locked`（`mse.rs:235`）。`launch_mse_loss_device_f32` の呼び出し元は `ops.rs:3195`（`mse_loss_device` の実装）だけ。
- 推論・step の capture 本体はこれらの入口を呼ばない。推論は `launch_tiled_bias_act_f32_resident` だけ（`dw`・`dinput` 系や MSE を通らない）、step は `CudaSgd::run` だけ。学習チェーンの部品（`graph.rs:1462`・`1478`・`1500` の `mse_loss_device_into_tracked`・`gemm_fp32_strict_into_device_tracked`・`gemm_resident_lhs_device_into_tracked`）は `launch_train_chain` の中でだけ使われる。

**§8.11.6 で「未精読」だった 3 点の分類**

- `mse.rs:180` の `target_cache`: 使うのは `with_target_dev`（`mse.rs:252`）で、その呼び出しはホスト版 `run_mse_loss_f32`（`mse.rs:498`）・`run_mse_backward_f32`（`mse.rs:557`）だけ。デバイス版・graph 経路は通らず、どの graph にも焼き込まれない。
- `h2d_staging`（`memory.rs:621`・`gemm.rs:509`）: pinned のホストメモリ（#1585・opt-in）で、デバイスアドレスではない。推論 replay の `upload_into`（`memory.rs:1702`）は capture の外（replay の前）で動く。
- `launch_tiled_bias_act_f32_resident` の全体: 上記のとおり引数だけで launch し、ワークスペースを持たない。

**ロックの突き合わせ**

- step の replay（`graph.rs:473〜489`）・推論の replay（`execute_chain`、`graph.rs:1087`〜）・両者の初回 launch が取るのは、`begin_driver_call` による driver 呼び出しの区間（世代検査・poison 検査）だけ。これで足りるのは、graph が焼き込むものがすべて thread-local またはインスタンス所有で、他スレッドの非 capture チェーンと共有する可変領域が無いため。
- 対比: 学習 graph の replay は `with_train_graph_workspaces_locked`（`graph.rs:1706`）で GEMM → MSE の順に両方のロックを取る。ordinal 共有の領域を焼き込むからである。
- 非 capture 経路（`mse_loss_device`・`gemm_*_device`）は共有領域を使うが、推論・step の graph はそれらを読み書きしないので、stream 上で順序が入れ替わっても取り違える対象が無い。

**デバイスメモリの再配布の窓（プールの確認）**

- `input_staging`・`acts` は `alloc_zeroed_inner`（`memory.rs:1261`）の device 分岐で `stream.alloc_zeros` により確保する。`gemm.rs` の `PooledCudaHandle`（drop 時にストリーム順序を持たないプール。`gemm.rs:510〜518` のコメント）は経由しない。
- cudarc 0.19.8 は非同期確保が使えるとき `malloc_async` で確保し、drop は `free_async` でストリーム順序に従う（cudarc `driver/safe/core.rs:809〜811`・`1535〜1536`）。`CudaBufferHandle::drop`（`memory.rs:156`）は `begin_buffer_release` の後にそれを呼ぶ。
- 推論 graph の evict・TLS 破棄は同一スレッド内で起き、`execute_chain` は毎回 `download` で同期して終わる（`graph.rs:1087〜1114`）ので、最後の launch は完了している。ほかのスレッドが同じアドレスを受け取る窓は、静的な読解では見つからなかった。
- step graph の `param`・`grad`・`velocity` の解放は呼び出し元（`DeviceParamStore`）の drop で、`SegmentKey` が live 借用から作り直され replay 直前に再検証される（`ops.rs:3822`・#1349 §4.4）。これは #3601 より前からある契約で、本節の範囲では変えない。

**限界**

- 静的な監査のみで、GB10 の実測はしていない。「無い」という結論のため、再現テストの案は書かない（根拠は上記の呼び出し連鎖）。
- cudarc 内部の `CudaSlice` のイベント追跡と graph の関係、同じストアを複数スレッドが同時に更新・推論する利用者側のデータ競合は対象外。必要なら別 Issue の候補（起票はユーザー承認後）。

**将来への注意**: 推論・step の capture 本体に ordinal 共有の永続ワークスペースを足すときは、`with_train_graph_workspaces_locked` と同じく、replay と初回 launch に該当のロックを足す。

### 8.11.7 修正後の GB10 実測結果（RULE-r2。`3596-r2-gb10`）

**GB10 で実施済み・判定 REGRESSION（終了コード 3）**（2026-10-26）。規則は `docs/perf/logs/train-chain-exclusion-3596/RULE-r2.txt`（計測前に登録）。出典: `judge-3596-r2.md`・`gb10-r2/`。機体は GB10（driver 580.173.02・rustc 1.97.1）、before `f7fe6b4df`・after `616be4ff7`（`crates/` は基準 `63f85eff2` と同一）。

- **G1 は合格**: 前提ゲート g1（`--exact`）・g2（3260 診断 9 件）・g3・g4（facade と 2 プロセス bit 一致、BITS 90 行・差分空）がすべて合格。初回（§8.11.5）で不合格だった loss の取り違えは再現しなかった。診断テストも after で 5/5 pass。
- **`#[ignore]` 全件**: 排他が原因の新規 fail は 0 件。after にだけ出た 2 件（`small_shape_matrix_unit_has_no_floor_tflops_record`・`wmma_tf32_staged_exceeds_tiled_f32_tflops_at_4096`）は一過性・既存の不安定で、単独実行は両ツリーで同じ傾向だった。詳細は `gb10-r2/ignored-compare-3596-r2.md`。
- **chain セル（1 スレッド・size=64・5 round）**: hard 条件（checksum 0.080541・PARAM_DIGEST）は全 run・全腕で一致。ratio（排他あり / なし）は graph off が **1.002**（ratio_r 1.001〜1.004 で 5/5 round が 1.00 超。0.1346 → 0.1349 ms）、graph on が 1.000（0.1126 ms）。RULE-r2 の判定は「いずれかのモードで ratio > 1.00 かつ 5/5 round で ratio_r > 1.00」を REGRESSION とするため、off で REGRESSION になった。
- 限界: 差は約 0.3 µs。2 腕は別ビルドで #3597・#3601 以外の変更を含みうる。1 機体・単一構成の結果で、並行時の性能は対象外。コード・閾値・規則は変更していない。
- 所有者判断事項: off の 0.2% の差を後退として扱うか（別名 RULE-r3 の事前登録による再実測が要るか）、§8.11.6 の機構の食い違いの追加調査。#3594 の前提は引き続き満たされない（§8.12.2）。

### 8.11.8 既定 OFF の ordinal は排他スコープを素通りにする（r3。2026-10-26）

RULE-r2 の判定は REGRESSION（既定 OFF の chain セル ratio 1.002。§8.11.7）。所有者判断（2026-10-26）に基づき、`begin_device_chain_scope`（`ops.rs`）を次のとおり変えた。`TRAIN_GRAPH_DEFAULT_ENABLED` は `false` のまま。GEMM→MSE のロック順（§8.11.6）、tolerance、baseline、FMA 契約、依存、`unsafe` は変えない。

- **分岐**: `device::resolved_stream_kind(ordinal)` が `Some(Legacy)` なら、`begin_driver_call`・デバイスキャッシュ・`observe_cuda_result` に触れず `DeviceChainScope::none()` を返す。`Some(Created)` と `None` は従来の経路（トークンを取り、capturable stream なら保持）。
- **`None`（デバイス未構築）は fail-closed**: 後から Created のデバイスが作られうるので、素通りにしない。2 step 目以降はデバイスがキャッシュされ判定が確定する。
- **判定の材料**: `device.rs` の `STREAM_KIND_CACHE`（ordinal ごとに最初の `CudaDevice::new` で決まる）。読み出しは mutex 1 回で driver 非接触。lock-free のヒント表は、mutex 1 回（数十 ns）が r2 の差（約 0.3 µs）より十分小さく、状態を増やさずに済むため採らなかった。
- **判定が古くならない根拠**:
  1. 単調性: `STREAM_KIND_CACHE` への書き込みは insert のみ（remove・clear・retain はない）。一度 Legacy と記録された ordinal は変わらない。invalidate 後の再構築も同じ関数を通る。
  2. capture の入口（step の `captured_segment_key`、推論チェーン、`train_chain_captured`）はいずれも `is_capturable_stream()` を確かめてから `begin_capture_session` へ進む。Legacy の ordinal では capture session が始まらず、排他の相手がいない。
  3. `internal-diagnostics` では Created 分岐がコンパイルされないため常に Legacy。
  4. `None` は現行経路（fail-closed）。
  5. poison・Retiring の検査はスコープ取得から encode の最初の部品の `begin_driver_call` へ移るだけ。`train_chain.rs` の encode エラー処理で、スコープ取得の失敗時と同じく poison される。
- **採らなかった案**: `begin_capture_session` で Legacy の ordinal を拒否する防御。`internal-diagnostics` では accessor が常に Legacy を返し、偽 ordinal を使う既存の CI テストを壊すため。根拠 1・2 で足りる。
- **テスト（GPU 不要）**: Legacy 登録済みの ordinal は別スレッドの capture 中でも `Ok`・非保持・in-flight 0（素通りの証拠）。未登録は `DeviceContextCaptureInProgress`（現行経路へ入った証拠）。未登録・capture なしは `Err` でトークンを残さない。

### 8.11.9 r3 の GB10 実測結果（RULE-r3。`3596-r3-gb10`）

**GB10 で実施済み・判定 UNDETERMINED（終了コード 4）**（2026-10-26）。規則は `docs/perf/logs/train-chain-exclusion-3596/RULE-r3.txt`（計測前に main へ登録。マージ SHA `46a2a43d9`）。出典: `judge-3596-r3.md`・`gb10-r3/`。機体は GB10（driver 580.173.02・rustc 1.97.1）、before `f7fe6b4df`・after `46a2a43d991f7099eaa4dfb97e07878e106b5c61`（`crates/` は基準 `2794cfba6` と同一）。

- **G1 は合格**: 前提ゲート g1（`--exact`）・g2（3260 診断 9 件）・g3・g4（facade と 2 プロセス bit 一致、BITS 90 行・差分空）がすべて合格。
- **`#[ignore]` 全件**: 排他が原因の新規 fail は 0 件。after にだけ出た 1 件（`gemm_device_matches_host_gemm_bitwise`）は単独 2/2 pass の一過性。`train_step_graph_diag_tests_3260` は after で 9 件全件 pass。詳細は `gb10-r3/ignored-compare-3596-r3.md`。
- **chain セル（1 スレッド・size=64・5 round）**: hard 条件（checksum 0.080541・PARAM_DIGEST）は全 run・全腕で一致。専有ゲートは全 round 通過。ratio（排他あり / なし）は graph off が **1.007**（ratio_r 0.995, 1.006, 1.006, 1.007, 1.009。0.1344 → 0.1353 ms）、graph on が 1.000（0.1126 ms）。off は ratio > 1.00 だが round 1 が 0.995 で 5/5 の符号一貫がないため、RULE-r3 の判定は UNDETERMINED（REGRESSION でも NON-REGRESSION でもない）。
- 事前表明の仮説（off の ratio 0.99〜1.00）は外れた。off の差は r2（1.002）より大きい。原因の切り分けは未実施で、本節では推定しない。
- 限界: 2 腕は別ビルド。1 機体・単一構成の結果で、並行時の性能は対象外。コード・閾値・規則は変更していない。
- 所有者判断事項: off の差を追加計測（round 数を増やす別名 RULE の事前登録等）で切り分けるか、判定基準を見直すか。完了条件（`judge-3596-r3.exit` = 0）が未達のため #3596 は閉じない。#3594 の前提は引き続き満たされない（§8.12.2）。

### 8.11.10 r3 の差の切り分け診断（DIAG-r4。手順の登録）

**状態: GB10 で実測済み。分類は場合 (i)「差は A/A と同程度。コードに原因は見つからない」（集計ツールの終了コード 0）。コード（`crates/`）は変えていない**（2026-10-27。出典: `docs/perf/logs/train-chain-exclusion-3596/gb10-diag-r4/summary-3596-diag-r4.md`・`.exit`）。

- 目的: RULE-r3 の UNDETERMINED で残った既定 OFF の chain セルの約 0.9 µs の差の出どころを切り分ける（所有者判断 2026-10-26: driver 呼び出しは原因ではなかった）。判定ではなく分類であり、RULE.txt・RULE-r2.txt・RULE-r3.txt と判定記録は変えない。
- 手順の正本: `docs/perf/logs/train-chain-exclusion-3596/DIAG-r4.txt`（腕 7 つ・5 round 巡回・専有ゲートは RULE-r3 と同一・A/A ノイズ幅 N による分類規則と結論の対応表を計測前に固定）。
- 既定 OFF 経路に乗る before→after の差分は 2 つだけ: `train_chain.rs` の `begin_device_chain_scope` 呼び出し（毎 step・dyn dispatch）と、その実体が読む `device::resolved_stream_kind`（`Mutex` 1 回）。`graph.rs`／`mse.rs` の #3601 のロックは既定 OFF では `train_chain_captured` が即 `Ok(None)` で届かないため経路外で、対照腕（`rv-graphlock`）として置く。
- 腕: `aa0`／`aa1`（before の同一バイナリ = A/A）、`excl0-rebuild`（before の再ビルド）、`excl1`（after）、`rv-scopecall`／`rv-kindlookup`／`rv-graphlock`（after に計測専用パッチ 1 本。`diag-r4/*.patch`。`crates/` へはコミットしない）。
- 道具: `scripts/bench/framework-compare/run_diag_train_chain_exclusion_cuda.sh`（ランナー）、`summarize_diag_train_chain_exclusion.py`（集計。テストは CI で実行）。
- 差が A/A と同程度、または配置・ビルド起因なら記録して止め、所有者へ報告する。コードに原因が見つかった場合に限り、最小の修正を入れ、別名の RULE-r4.txt を計測前に登録して再実測する（#3596 は NON-REGRESSION が出るまで閉じない）。

#### 実測結果（`gb10-diag-r4/`。ラベル `3596-diag-r4-gb10`）

- 条件: GB10（aarch64）・before = `f7fe6b4dfaabbdbc51a900709fe6648c36120c58`、after = `9bff0b00d452128b6628575bf307ab968b32586a`（`git diff 46a2a43d9 9bff0b00d -- crates` は空で、DIAG-r4 の `after_crates_identical_to` を満たす）。7 腕 × 5 round を DIAG-r4 の巡回順で実施。`precheck=assert ok`、ランナー終了コード 0、専有ゲートは全 round 通過、skipped ログは空。`env_info` の `diag_sha256` は main の `DIAG-r4.txt` の sha256 と一致し、`bin_sha256_aa0` と `bin_sha256_aa1` も一致。ホスト名・`$HOME` はマスク済み（`<home>`）。
- hard 条件（checksum・PARAM_DIGEST の全 run・全腕での一致）は集計ツールが検査して合格（終了コード 0）。

| 腕 | 代表値 median_s (s) | rho | rho_r (r1..r5) |
|---|---|---|---|
| aa0 | 0.000134382 | 1.0000 | 1.0000 1.0000 1.0000 1.0000 1.0000 |
| aa1 | 0.000134422 | 1.0003 | 1.0003 1.0010 0.9947 1.0028 0.9979 |
| excl0-rebuild | 0.000134608 | 1.0017 | 1.0033 1.0075 0.9935 1.0032 1.0031 |
| excl1 | 0.000135047 | 1.0049 | 1.0004 1.0120 0.9980 1.0068 1.0026 |
| rv-scopecall | 0.000134479 | 1.0007 | 1.0094 1.0058 0.9926 1.0018 0.9984 |
| rv-kindlookup | 0.000134462 | 1.0006 | 1.0025 1.0012 0.9946 1.0021 0.9961 |
| rv-graphlock | 0.000135240 | 1.0064 | 1.0081 1.0101 1.0021 1.0067 1.0033 |

- N（A/A ノイズ幅）= 0.0053、Delta_after = rho(excl1) - 1 = +0.0049。|Delta_after| <= N のため場合 (i)。revert 腕の原因候補判定は場合 (ii) のときだけ行う規則なので適用されない（参考として rho は rv-scopecall 1.0007・rv-kindlookup 1.0006・rv-graphlock 1.0064）。
- 併記（ビルド・配置の影響あり）: rv-graphlock（既定 OFF の経路外の対照腕。|rho - 1| = 0.0064 > N）。場合 (i) のため分類は (i) のままで、この動きの原因は推定で埋めない。
- 読み方: RULE-r3 で観測した約 0.9 µs（ratio 1.007）の差は、本診断では excl1 で +0.66 µs（rho 1.0049）に縮み、A/A（aa1）の round 内ばらつき（rho_r 0.9947〜1.0028）の幅 N に収まった。差の出どころとしてコード上の原因（スコープ呼び出し・`resolved_stream_kind` の mutex）は見つからなかった。
- 限界（DIAG-r4 のとおり）: 単一機体・train CUDA・size=64・1 スレッド・別ビルド比較・5 round で検出力は限られる。診断腕のパッチは計測専用で、排他の正しさ（G1 ほか）を検証しない。
- 所有者判断事項: 場合 (i) のため、DIAG-r4 の規則どおり停止して所有者へ報告する。選択肢は (a) 約 0.5% の差を測定ノイズ水準として許容し、RULE の判定基準（閾値 1.00・符号一貫の要件）を見直すか、(b) round 数を増やす別名 RULE の事前登録で再実測するか、のいずれか。閾値・round 数・判定方法は変えていない。`TRAIN_GRAPH_DEFAULT_ENABLED` は `false` のまま。#3596 は NON-REGRESSION が出ていないため閉じない。
- **所有者判断（2026-10-27）**: 差は A/A のノイズ幅と同じ程度で、ビルドごとの配置の違い（対照腕 rv-graphlock の +0.64%）でも同じ程度に動く。round を増やしてもこの偏りは消えないため、再計測はしない。性能は RULE-r3 の判定 UNDETERMINED のまま記録し、NON-REGRESSION（pass）としては扱わない。閾値・判定方法・round 数は変えない。正しさの受入基準（G1〜G4・`#[ignore]` 全件の突き合わせで、排他が原因の新規 fail は 0 件。r3）を満たしたので #3596 を閉じる。`TRAIN_GRAPH_DEFAULT_ENABLED` は `false` のまま。

## 8.12 #3594 既定 ON への反転は見送り（既定 OFF 維持）

### 8.12.1 結論
- `TRAIN_GRAPH_DEFAULT_ENABLED` は `false` のまま。コード変更なし（反転の差分は PR レビュー指摘 P1 を受けて取り下げた）。tolerance・baseline・FMA 契約・依存・`unsafe` も変更なし。

### 8.12.2 見送りの理由（未解決事項）
- #3596 の `begin_device_chain_scope` が保護するのは encode と SGD の区間だけである。チェーン外の通常操作（`ResidentLoss::read` → `read_train_step_loss` → `mem.download`、Tape の通常演算など）は、待機も再試行もしない。
- 既定 ON では、環境変数なしで同じ ordinal を複数スレッドが使うと、一方の学習 capture 中に他方の通常操作が `DeviceContextCaptureInProgress` で失敗し、公開 API（`step` → `loss.read()?`）からそのまま返る。従来は設定不要で動いていた並行学習が、別 store の capture タイミングだけで中断する。
- さらに、graph の replay と非 capture チェーンが共有の永続ワークスペースを取り合い loss を取り違える欠陥があった（§8.11.6。修正済み。GB10 で G1・診断テストの pass は確認したが、既定 OFF の chain セルが RULE-r2 の閾値を超えた〈§8.11.7〉。r3 の再実測は UNDETERMINED〈§8.11.9〉）。反転の前提に関わる。
- 反転の前提条件: チェーン外の通常操作にも、capture 中は安全に待機して再試行する仕組みを入れ、その並行回帰テスト（GB10 実機）を通すこと。

### 8.12.3 反転 PR に再度必要な確認（未実測）
- 腕 A（既定 ON）・B（`=0`）・C（main）の backend-cuda・facade `#[ignore]` 全件の突き合わせ、`unfenced_chain_and_capture_threads_do_not_poison_and_stay_bit_exact`（30 回）、`train_step_graph_diag_tests_3260` の 7 件、2 プロセス bit 一致、1 スレッド chain セルの排他あり・なし比較。
- 影響評価（再開時の参考）: 既定 stream が created stream になり全 CUDA 利用者に波及する・lr スケジューラ下では 3 回の無駄な capture 後に従来チェーンへ戻る（有界・bit 一致）・`GemmPrecision` が `Fp32Strict` 以外なら capture しない。

### 8.12.4 申し送り
- チェーン外の通常操作の待機・再試行は別 Issue で追跡する（起票はユーザー承認後）。
  - 追跡先は #3608。設計案は §8.12.5（所有者確認待ち）。

### 8.12.5 #3608 チェーン外の通常操作の待機・再試行（設計案・所有者確認待ち）

§8.12.2 の反転前提条件（チェーン外の通常操作にも capture 中の安全な待機・再試行を入れる）を満たすための設計案。**本節は設計の記録だけで、コードは変更していない**。入口の全体に及ぶ変更のため、所有者の確認が記録されるまで実装しない（#3608 作業 1）。確認の記録は本節末尾に追記する。既定 OFF（`TRAIN_GRAPH_DEFAULT_ENABLED = false`）・tolerance・baseline・依存・`unsafe` は変えない。§8.11.1 の表で「未確認」とした行（reuse 経路・fresh 経路・Tape の通常演算）への答えは 8.12.5.5 に分けて書く。

#### 8.12.5.1 待機・再試行を入れる場所

現状（`origin/main` 1009825c9 で確認）。

- `context_cache::begin_driver_call` は、別スレッドが capture 中で、かつ呼び出しスレッドが当該 ordinal の `CallToken` を持たないとき、待たずに `DeviceContextCaptureInProgress` を返す。この拒否条件は「待っても自己デッドロックしない条件」と一致する。
- `wait_foreign_capture_idle` は上限付きの Condvar 待ちとして既にある（トークン保持時は即エラー）。`begin_buffer_release` は capture 終了を待つ型の前例である。
- 通常操作の入口は 1 か所ではない。`begin_driver_call` の直接呼び出しが ops.rs 24 か所・memory.rs 9 か所（うちテスト分を含む）・graph.rs 10 か所あり、さらに 63 か所が `context_cache::with_driver_call` を経由する（gemm 系・transpose・mse・norm・softmax など約 30 のカーネルモジュールが同名のヘルパーを持ち、実体はそこへ委譲する）。**当初の想定「`ops.rs`／`memory.rs` の 4 ラッパーだけで覆える」は不足**で、カーネル側の入口も対象になる。

| 案 | 内容 | 判断 |
|---|---|---|
| (A) `begin_driver_call` 自体を待ち型にする | 全入口が一度に変わる | **採らない**。doc コメントの契約（待たない）を壊す。graph.rs は即時エラーで graph を戻し呼び出し元の再試行へ委ねる設計、`step()` の `captured_segment_key` は即時エラーで直接実行へ落とす設計であり、どちらも即時拒否が前提。既存テストと autodiff の mock（`CAP_BUSY_*`）も同じ前提 |
| (B) 待ち型の入口を 1 関数で新設し、通常操作の入口だけ切り替える | `context_cache::begin_driver_call_waiting(ordinal, gens, timeout)`（と sync point 版）を足す。手順は「試す → `DeviceContextCaptureInProgress` のときだけ `wait_foreign_capture_idle` で待つ → 再試行 → 期限まで繰り返す」。他のエラー（Poisoned／Unrecoverable／Retiring／Stale／Unsupported）は待たずに返す | **推奨** |

(B) の切り替え対象（推奨）。

1. `ops.rs::CudaBackendOps::with_driver_call`／`with_sync_point_call`、`memory.rs::CudaMemory::with_driver_call`／`with_sync_point_call`。
2. `context_cache::with_driver_call`（63 か所が経由するカーネル側の入口）。これを切り替えると gemm・norm 等の通常演算が覆われる。一方でベンチ・診断用の低レベル API も影響を受けるため、**切り替えるか否かを所有者に確認する**。切り替えない場合、Tape の通常演算のうちカーネル側入口を通る演算は覆われず、受入基準を満たせない可能性がある。推奨は「切り替える」。
3. ops.rs・memory.rs の直接呼び出し（resident 系・epilogue 融合系など）は、1 件ずつ「待つ／待たない」を段階 B の着手前に表にする。

待たせない入口（現行の即時版のまま残す）。

| 入口 | 理由 |
|---|---|
| `begin_device_chain_scope`（ops.rs:4099） | `train_step_device_chain` の再試行ループの内側にあり、ここで待つと二重に待つ |
| `captured_segment_key`（ops.rs:3763 付近の直接呼び出し） | 即時エラーで `step()` が直接実行へ落ちる契約 |
| graph.rs の全 10 か所 | 即時エラーで graph を戻し呼び出し元が再試行する契約。replay はトークン → ワークスペースロックの順 |
| `begin_capture_session`・`begin_buffer_release`・`invalidate_with` | 既存の排他・待機の責務で、順序を変えない |

tensor-core の trait は変えないため、CPU・Metal には影響しない。

#### 8.12.5.2 待ち方・タイムアウト・エラー型

- 待ち方は `wait_foreign_capture_idle`（Condvar。`CaptureGuard::drop` の `notify_all` で起きる）。ポーリングや sleep は使わない。
- 上限は backend-cuda に定数を 1 つ新設する（例 `DRIVER_CALL_CAPTURE_WAIT_TIMEOUT`）。本番は 1000 ms（`PREDICT_CAPTURE_WAIT_TIMEOUT`・`TRAIN_CAPTURE_WAIT_TIMEOUT` と同じ値）、`cfg(test)` は 30 ms。**値は所有者に確認する**。
- 期限切れは既存の `BackendError::DeviceContextCaptureInProgress { ordinal }` を返す。新しい variant は足さない（公開 API 非破壊。`crates/facade/tests/api_surface.rs` の固定内容は不変）。
- 待機は driver に触れる前の入口だけで行い、演算の途中で再試行して二重に書き込むことはしない（演算単位で再実行しない）。

#### 8.12.5.3 デッドロック分析

| 状況 | 結果・根拠 |
|---|---|
| capture を持つスレッド自身の呼び出し | `begin_driver_call` は通る。sync point は待つ前に `Unsupported` を返す（現行の判定順を変えない） |
| 当該 ordinal の `CallToken`／`DeviceChainScope` を持つスレッド | `begin_driver_call` が通過条件で通るので待ちに入らない。`wait_foreign_capture_idle` もトークン保持時は即エラー。順序は必ず「試す → 拒否されたときだけ待つ」 |
| 別の ordinal のトークンを持ったまま待つ（ordinal をまたぐ輪） | 期限付きなので有界。安全側として「どの ordinal のトークンも持たないときだけ待つ」条件を推奨する。**所有者に確認する** |
| phase が Active 以外 | 待たずに `wait_foreign_capture_idle` が `Ok` を返し、再試行した `begin_driver_call` が型付きエラーを返す |
| capture 側の `in_flight` ドレイン | 待つスレッドはトークンを持たないのでドレインを妨げない |
| ordinal 共有 Mutex を保持したまま入口へ入る経路 | **未確認**。`.lock()` を持つ backend-cuda のファイルは context_cache.rs・host_staging.rs・memory.rs・readback_prefetch.rs・readback_policy.rs・ops.rs・mse.rs・device.rs・module_cache.rs・graph.rs・gemm.rs ほか。「ロック → 入口」の順の経路が 1 つでもあれば、その経路は待たない版を使うかロック順を直すかを決める。段階 B の着手前に全箇所を確かめて本表へ追記する（確認できるまで実装しない） |
| 飢え | capture はキャッシュミスのときだけ起き、replay は capture を保持しない。待つのは 1 回の capture の区間だけ |

#### 8.12.5.4 テスト計画（実装は確認後）

- CI（GPU 不要。`internal-diagnostics` と偽 ordinal で §8.11.4 の既存テストと同型）: 別スレッドの capture が期限内に終わる → `Ok` / capture が続く → 期限後に `DeviceContextCaptureInProgress` / トークン保持スレッドは待たない / 非 Active は待たずに型付きエラー / capture スレッド自身の sync point は即 `Unsupported` / 待たせない入口は従来どおり即時拒否。既存の `begin_driver_call_rejects_other_thread_while_capturing` 等は変えない。
- GB10（`#[ignore]`）: `train_step_graph_diag_tests_3260.rs` の G1 ハーネスを再利用し、スレッド A が opt-in ON で capture する学習 step を繰り返し、スレッド B が同じ ordinal で非 capture の step・`loss.read()`・Tape 演算を繰り返す。期待は B に `CaptureInProgress` が出ない・B の loss 列が単独実行と bit 一致・poison なし。
- 既存の G1〜G4 相当と `#[ignore]` 全件の非後退を GB10 で確かめる。計測規則は計測前に新しい名前の RULE として登録する（既存 RULE は変えない。規則を後から緩めない）。

#### 8.12.5.5 書き込み後に拒否されたときの store・Tape の扱い（§8.11.1 の未確認行）

(B) では待機を各 driver 呼び出しの前に行うため、演算の途中で拒否が起きるのは期限切れのときだけになる。期限切れ時の扱いは次のとおり。**いずれも本段階ではコードを追っておらず「未確認」**。段階 B の前に経路ごとに確かめ、表を埋める。推測では埋めない。

| 経路 | 確認する点 | 状態 |
|---|---|---|
| reuse 経路・fresh 経路（`device_store.rs` の更新経路） | 拒否が書き込みの前か後か・保留分の戻し・poison への遷移・params／velocity の状態・再試行で回復できるか | 未確認 |
| Tape の forward | 出力は新規バッファなので既存状態は変わらない見込み | 未確認 |
| Tape の backward（勾配の in-place 累積） | 途中失敗で Tape と勾配が部分更新のまま残るか。残る場合の利用者向け契約（例: その Tape を捨てる） | 未確認 |

確認の結果コード修正が必要と分かった場合、その採否は所有者に確認する事項とし、本 Issue の中で勝手に直さない。

#### 8.12.5.6 性能の予想

既定 OFF（Legacy stream）では capture が始まらないため待機の経路に届かない。競合のない高速経路で増えるのは、拒否エラーを見る分岐だけと予想する。これは予想であり**未実測**。既定 OFF の chain セルに RULE 計測が必要かは所有者に確認する。「後退なし」とは書かない。

#### 8.12.5.7 所有者に確認する事項

1. 方式 (B)（待ち型の入口を新設し通常操作の入口だけ切り替える）の採否。
2. `context_cache::with_driver_call`（カーネル側 63 か所）を待ち型にするか。
3. ordinal をまたぐ待機の条件（「どの ordinal のトークンも持たないときだけ待つ」）。
4. 待機の上限値 1000 ms。
5. 既定 OFF の chain セルに対する RULE 計測の要否。
6. 8.12.5.5 で修正が必要と分かった場合の扱い。

§8.12.3 の反転 PR 用の再確認項目は引き続き**未実測**である。本節で反転の前提条件が満たされたことにはならない。

所有者の確認: **未記録**（確認後にここへ日付と内容を追記する）。
