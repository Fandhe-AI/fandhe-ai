# CUDA 非同期実行モデルの同期契約設計（イシュー #1012・#1011 ツリー第 1 段）

## 0. 位置づけ・スコープ

本文書は #1011 ツリー（「演算ごとの `stream.synchronize()` 廃止」）の第 1 段（設計のみ）であり、後続の #1013（実装）・#1014（回帰テスト）・#1016（Metal 側設計）が着手する際の契約の正本とする。本文書自体はコード・`docs/spec/`（正本 submodule）・CI・依存関係を一切変更しない。設計判断は「現状の実装を正しく契約化する」ことを優先し、新規機構（多ストリーム化・イベント DAG・CUDA Graph 等）の導入は不採用または保留として §10 に記録する。

## 1. 背景・実測根拠

フレームワーク横並びベンチ（`scripts/bench/framework-compare/results/summary.md`）で MLP 学習 1 step の実測値は本ライブラリ（CUDA バックエンド）が 12.4 ms、candle が 0.28〜0.81 ms であり、1 桁以上の差がある。親イシュー #1008 はこの主因を「演算ごとの同期・毎 step の D2H/H2D・都度割当」と診断している。

比較対象の同期モデルの要点（ソースの写経はしない）:

- candle: 演算ごとの明示的な同期呼び出しを持たない。ストリームへの投入は非同期のまま連鎖し、ホストが結果を読む操作でのみブロックする
- burn / cubecl: カーネル起動 (`launch`) 自体は非同期であり、同期はホスト側の `read`/`sync` 相当の操作に限定される

本ライブラリの `backend-cuda` は現状、演算ラッパーのほぼ全てが「起動直後に `stream.synchronize()`」というパターンを踏襲しており、この構造上の差が上記の性能差の一因になっていると考えられる。ただし削除の前に「どの操作が同期点として必須か」「非同期化してもデータ競合・エラー帰属の喪失が起きないか」を契約として確定する必要があり、これが本イシューの目的である。

## 2. 現状整理

### 2.1 ストリーム構成

`CudaDevice::new`（`crates/backend-cuda/src/device.rs:98`）は `ctx.default_stream()` を 1 本だけ保持する。`crate::context_cache`（`ordinal` をキーとするプロセス内キャッシュ。`crates/backend-cuda/src/device.rs:283` 付近から参照）経由で、`CudaGemm`・`CudaElementwise`・`CudaRmsNorm`・`CudaSoftmax`・`CudaSgd`・`CudaMemory` はいずれも同一 `Arc<CudaStream>` を共有する。つまり **ordinal ごとに単一の非同期ストリームがすでに成立している**構成であり、本設計はこれを契約として明文化する（新規のストリーム分割は行わない）。

### 2.2 `stream.synchronize()` の棚卸し

`grep -rn synchronize crates/backend-cuda/src` による本番経路の棚卸し（診断・ベンチ専用ファイルは対象外）。

| ファイル | 行 | 分類（#1013 での扱い） |
|---|---|---|
| `gemm.rs` | 1355, 1639, 1693, 1747, 1825, 1880, 1958, 2053 | 本番ラッパー（launch 直後）→ 除去候補 |
| `gemm_wmma.rs` | 287, 367 | 同上 |
| `gemm_mma.rs` | 713, 882 | 同上 |
| `gemm_mma_tf32.rs` | 218, 252 | 同上 |
| `kernels_mma.rs` | 1254, 2918 | カーネル起動ヘルパ内 → 除去候補（呼び出し側の readback 境界へ移す） |
| `kernels_mma_tf32.rs` | 1335 | 同上 |
| `kernels_wmma_opt.rs` | 528, 631, 1571, 2268 | 同上 |
| `elementwise.rs` | 180, 208 | 本番ラッパー → 除去候補 |
| `softmax.rs` | 385 | 同上 |
| `rmsnorm.rs` | 830, 1102, 1105 | 同上（1102/1105 は分岐内二重同期。§9 で単一化） |
| `transpose.rs` | 393, 418, 450, 599 | 同上 |
| `sgd.rs` | 174 | デバイス常駐経路（`sgd_step_device`）→ **最優先の除去対象**（D2H を伴わない唯一の経路） |
| `memory.rs` | 276 | `download` の D2H 境界 → **維持**（契約上の同期点） |
| `fresh_overhead_diag_tests.rs`・`jit_cache_bench_tests.rs` | — | 診断・ベンチ専用 → 対象外 |

### 2.3 API 形状の制約

`BackendOps::gemm`／`add`／`mul`／…（`crates/tensor-core/src/backend_ops.rs`）はホスト `Tensor` を受け取りホスト `Tensor` を返す。この経路は **戻り値の D2H が構造的な同期点**であり、非同期化による恩恵は `DeviceBuffer` を扱う常駐経路（`sgd_step_device`・`DeviceParamStore` の常駐 forward 経路）に集中する。「都度同期の除去」とは「launch 直後の明示 `synchronize()` を readback 境界 1 箇所へ集約する」ことであって、ホスト `Tensor` API を使う限り同期回数自体は D2H の回数分だけ残ることに注意する。

### 2.4 cudarc 0.19.8 の意味論

（`cudarc = { version = "=0.19.8", ... }`、`Cargo.toml:95`。参照パスは `cudarc-0.19.8/src/driver/safe/` 配下）

- `LaunchArgs::launch` は `cuLaunchKernel` を発行して即時復帰する（`launch.rs`）。起動時エラー（無効な launch config・引数不整合等）は同期的に `DriverError` として返る
- `device_ptr()` は event tracking が既定で有効であり、read/write イベントを記録する（`core.rs`）。`CudaSlice::drop` はその記録イベントに対し **デバイス側で `stream.wait`** してから `cuMemFreeAsync`（`has_async_alloc` が真の環境）で解放する。`has_async_alloc` が偽の環境では Drop がホスト側 `stream.synchronize()` へフォールバックする（正しさは保たれ性能のみ劣化する）
- `clone_dtoh` は `cuMemcpyDtoHAsync` をペイジャブルな `Vec` へ発行する。cudarc 側では同期を挟まない（`SyncOnDrop::Sync(None)`）。CUDA の仕様上ペイジャブルなホストメモリへの DtoH は完了までホストを実際にはブロックする実装がほとんどだが、これは実装依存の挙動であり契約として依存してはならない。pinned メモリを将来導入した場合は非同期化しうるため、**契約は既存どおり明示 `synchronize()` に依存させる**（`memory.rs:276` の方針を維持）
- Drop 中に生じたエラーは `ctx` 側に蓄積され（`record_err` 相当）、次にドライバ API を呼ぶ操作（`attribute` 等）で表面化する
- `CudaStream::record_event`／`wait`／`fork`／`join`・`CudaEvent::synchronize` は利用可能だが、本設計では多ストリーム化を採用しないため使用しない（§10）

### 2.5 Metal 側の現状（#1016 の前提）

`MetalContext::dispatch_sync`（`crates/backend-metal/src/context.rs`）は dispatch ごとに `commit` + `waitUntilCompleted` + status 検査を行う。`download`（`crates/backend-metal/src/memory.rs`）は unified memory の `contents()` 読みであり `synchronize` 相当の待ち合わせを持たない。#1016 は「コマンドバッファを共有し `waitUntilCompleted` はホスト実体化時のみ発行する」構成へ変更する前提であり、§7 で CUDA との語彙対応を示す。

### 2.6 既存の関連文書

- `crates/tensor-core/src/buffer.rs`（「download の同期契約」節）: `download` は復帰時点でホストデータが確定していることを全バックエンド共通で保証する契約を既に定めている。本文書はこれを上書きせず前提として踏襲する
- `crates/bench-harness/src/sync.rs`（`SyncPoint`）: REQ-8 計測用の「ホスト転送を伴わない完了待ち」ユーティリティ。§4・§7 で参照する

## 3. 実行モデル（design decision 1）

**採用**: ordinal ごとに **単一の非同期ストリーム**（2.1 の現行構成をそのまま契約化する）。全カーネル起動・H2D・D2H・デバイスメモリ解放はこのストリームへ投入順に並び、**投入 API はホストをブロックしない**ことを原則とする（例外は §4 の同期点のみ）。

以下を不変条件とし、#1013 の実装合格条件・#1014 のテスト対象とする。

- **I1（ストリーム順序）**: 同一 ordinal 上のすべての演算は投入順に実行される。依存関係を満たすためだけの明示同期は不要
- **I2（同期点での完了保証）**: ホストがデバイス結果を読む API（§4）の復帰時点で、その API 呼び出しに先行して投入された全作業が完了している
- **I3（ホスト側一時バッファの解放）**: ホスト側入力 (`Vec` 等) は `clone_htod` 復帰後に解放してよい（ペイジャブル H2D は driver 側でステージングされる。pinned メモリを導入する場合はこの前提を再確認する必要がある。§10）
- **I4（デバイス一時バッファの解放）**: `CudaSlice::drop` はイベント待ち + `cuMemFreeAsync` により、明示同期なしで use-after-free を起こさない。ただし `has_async_alloc` が偽の環境では Drop がホスト同期にフォールバックする。**GB10 実機で `CudaContext::has_async_alloc()` が真であることは本文書執筆時点で未実測**であり、§8 T4 で確認する
- **I5（数値意味論の不変）**: カーネル側の手動境界チェック（`.claude/rules/coding-rust.md`）・FMA 契約・バックエンド間数値一致の複合判定（相対誤差 1e-3 未満 または 絶対誤差 1e-5 未満）は非同期化によって変化しない

多ストリーム化・イベント DAG・CUDA Graph は本設計では採用しない（§10 に理由を記す）。

## 4. 同期点一覧（design decision 2）

| API | 同期種別 | 現状 | #1013 後 | 根拠 |
|---|---|---|---|---|
| `MemoryOps::download` | ホストブロック（契約上の同期点） | `clone_dtoh` → `synchronize` | 変更なし | `crates/tensor-core/src/buffer.rs`「download の同期契約」 |
| ホスト `Tensor` を返す `BackendOps` 演算（`gemm`／`add`／`mul`／`relu`／`exp`／`tanh`／`sum`／`max` 等） | ホストブロック | 演算ごとに `launch` 直後 `synchronize()` | readback ヘルパ（§9）呼び出し 1 回に集約 | 戻り値が `Tensor` である以上 D2H は構造的な同期点（2.3） |
| `DeviceParamStore::sync_to_host`／`register_resident_leaves`／`snapshot_resident_leaves` | ホストブロック | `download` 経由 | 変更なし | `download` の契約に従属 |
| `bench_harness::SyncPoint::wait_idle` | ホストブロック（計測専用・ホスト転送なし） | 変更なし | 変更なし | REQ-8 計測境界（`crates/bench-harness/src/sync.rs`） |
| `CudaDevice` 破棄時 | ホストブロック（Drop 経路依存） | `has_async_alloc` 偽時にホスト同期へフォールバック | 変更なし | 2.4 |
| `CudaSlice::drop`（デバイス一時バッファ解放） | デバイス側 wait のみ（ホスト非ブロック） | 変更なし | 変更なし | I4 |
| `launch`（カーネル起動） | 同期なし（起動時エラーのみ同期的） | 変更なし | 変更なし | 2.4 |
| `sgd_step_device` | 同期なし | 現状 `synchronize()` あり（`sgd.rs:174`） | **除去**（最優先） | 2.2・D2H を伴わない唯一の常駐経路 |
| `MemoryOps::upload` | 同期なし | 変更なし | 変更なし | ペイジャブル H2D の復帰＝ステージング完了であり完了待ちではない |

同期点を増やしてよい条件は、診断・ベンチ・エラー検出目的の任意同期に限り、`internal-diagnostics` 相当のモジュールまたはテストコード内のみで許容する（本番経路の演算ラッパーには追加しない）。

## 5. エラー伝播（design decision 3）

| 分類 | 発生源 | 検出タイミング | 型付きエラー | 帰属 |
|---|---|---|---|---|
| 起動時（同期） | 無効な launch config・引数不整合・NVRTC コンパイル失敗 | 投入 API の復帰時 | `CudaError::Driver`／`CudaError::Compile` → `BackendError::KernelLaunchFailed` | 当該演算に帰属する |
| 遅延（非同期） | カーネル実行時の fault 等 | **次の同期点**（`download` 等、§4） | `synchronize()`／`clone_dtoh` から返る `DriverError` → 既存の `BackendError::TransferFailed`（メッセージに「先行して投入されたカーネルの遅延エラーを含みうる」旨を付す） | 帰属は失われる（当該同期点までに投入された任意の演算に由来しうる） |
| Drop 時 | `cuMemFreeAsync`／イベント wait の失敗 | 次にドライバ API を呼ぶ操作（`check_err` 相当）経由 | 蓄積後、次の `CudaError::Driver` として表面化 | 帰属なし |

方針:

- 遅延エラー・Drop 時エラーの発生後は、同一 `CudaContext` での処理継続を禁じる（fail-closed）。ただし継続禁止を実効化するには「誰が・どの状態を・どこで検査して拒否するか」を契約化する必要があり、2.6 の `context_cache` の現状（「エントリはプロセスの生存期間中 evict されない」・`context_cache.rs` 冒頭コメント「所有モデル・生存期間」節）はエラー検出後の**自動的な**回復手段を持たない。したがって本設計は以下の点を #1013 の実装対象として明文化する（レビュー指摘 PR #1053・本ファイル該当行への回答。当初案〈`poisoned` フラグのみのクリア／readback・Drop 限定の `mark_poisoned` 呼び出し〉は下記 codex-review P0 2 件・Bugbot 指摘により世代管理・呼び出し経路の一般化が必要と判明したための改訂）:
  1. **context 世代（generation）による poison 状態管理**: `context_cache` のキャッシュエントリ（`cached_device`／`cached_gemm`／`cached_elementwise`／`cached_rmsnorm`／`cached_softmax` の内側 `Arc<Mutex<Option<Arc<T>>>>` と同じ `ordinal` キー空間）に、`ordinal` ごとに独立した `poisoned: AtomicBool` に加え **`generation: AtomicU64`（初期値 0）** を追加する。`poisoned` は「現在の `generation` に対して」真偽が定義される値として扱う（後述の `invalidate` が世代を進めた時点で、その世代においては非 poisoned から再出発する）。`CudaGemm`・`CudaElementwise`・`CudaRmsNorm`・`CudaSoftmax`・`CudaSgd`・`CudaMemory` が同一 `Arc<CudaStream>`（延いては同一 `CudaContext`）を共有する現行構成（2.1）を踏まえ、いずれか 1 つの経路が遅延エラー・Drop 時エラーを検出したら同一 `ordinal` の全スイートへ波及させる
  2. **以後の API 呼び出しを拒否する経路**: `crate::context_cache` に `mark_poisoned(ordinal)`（エラー検出側が呼ぶ）と `is_poisoned(ordinal) -> bool`（各スイートの演算メソッド入口が呼ぶ。内部的には現在の `generation` に対する `poisoned` を読む）を追加する。`ops::CudaBackendOps` の各演算メソッド（`gemm`／`add`／`mul`／…）・`sgd_step_device`・`MemoryOps::download`／`upload` は、実処理に入る前に `is_poisoned` を検査し、真であれば新規 variant（下記）で即座に拒否する。これにより「poison 検出後も同一 `CudaContext` 上で新たな演算を投入できてしまう」抜け道を塞ぐ
  3. **明示的な再生成手段（世代を進める）**: `context_cache` に `invalidate(ordinal)` を追加する。当初案（`cached_device`／各 `cached_*` エントリを `None` へ戻し `poisoned` フラグをクリアするのみ）は、**すでに `Arc<CudaContext>`／`Arc<CudaStream>`／`DeviceBuffer` を clone 済みの呼び出し元がそのハンドルを保持し続けた場合、`is_poisoned(ordinal)` が false に戻った後もそれらの旧ハンドルが古い poisoned `CudaContext` を指したまま使われ得る（fail-closed 契約の迂回。codex-review P0 指摘 1）」という欠陥を持つため採用しない。改訂案として、`invalidate(ordinal)` は次を単一のアトミック操作（`ordinal` ごとの `Mutex` 区間内）として行う: (a) `generation` を 1 増やす、(b) 新しい世代に対して `poisoned` を false へ初期化する、(c) `cached_device`／各 `cached_*` エントリを `None` へ戻す。**世代を跨いだハンドルは意味的に別物として扱う契約**とし、`is_poisoned(ordinal)` は常に「現在の `generation` における」状態だけを答える（旧世代のハンドルを保持し続けている呼び出し元がいても、それらは新たに `is_poisoned` を問い合わせる経路（各演算メソッド入口）を経由する限り、次の呼び出しで新世代のキャッシュから取り直した非 poisoned なハンドルに置き換わる。すでに手元にある旧ハンドルを直接使い続けるコードパスの排除は、呼び出し元がハンドルを長期保持しない設計〈2.3・各演算メソッド入口で `context_cache::cached_device` 等から都度取得する現行実装〉に依存する契約として明記する）。加えて `context_cache` は `current_generation(ordinal) -> u64` を公開する。これは次項 4 のとおり `context_cache` 以外の `ordinal` キー付きプロセス内キャッシュが自分のエントリの鮮度を検証するために必要な参照点である。呼び出し元（学習ループ・facade 側）は下記 variant のエラーを受け取ったら `invalidate(ordinal)` を呼んでから次の `tape_for(Device::Cuda(ordinal))` を行うことで、新しい `CudaContext` を明示的に再構築できる。これが「回復手段は再生成のみ」という記述が前提とする API であり、`context_cache` 側に該当 API を新設するのは本改訂が初めて（実装なしの記述不整合を解消する変更点）
  4. **世代契約の伝播範囲（`context_cache` 外のプロセス内キャッシュへの適用。Bugbot 指摘への回答）**: `ordinal` をキーとするプロセス内キャッシュは `context_cache` の `cached_*` 系列だけではない。`crates/backend-cuda/src/ops.rs` の `static_cuda_memory`（`Box::leak` で `'static` 化した `CudaMemory` を `ordinal` キーの `HashMap` で保持する、`context_cache` とは別系統のプロセス内シングルトンキャッシュ。イシュー #935）は、内部に `Arc<CudaStream>`（延いては旧 `CudaContext`）を保持する `CudaMemory` を返すため、`context_cache::invalidate(ordinal)` を呼んでもこのキャッシュは更新されず、`memory_ops()` 経由で取得される `MemoryOps`（`download`／`upload`）が旧 poisoned context を指したまま使われ続ける（codex-review 指摘 1・Bugbot 指摘「Invalidate skips leaked memory cache」はこの経路を指している）。これを解消するため、`static_cuda_memory` のキャッシュ値を `(bound_generation: u64, mem: &'static CudaMemory)` へ拡張する契約とする: 取得時に `context_cache::current_generation(ordinal)` を読み、キャッシュ済みエントリの `bound_generation` と不一致であれば「新しい世代の `CudaDevice`（`context_cache::cached_device` から取り直したもの）上に新しい `CudaMemory` を `Box::leak` で構築し、`(現在の generation, 新エントリ)` で `HashMap` を上書きする」。旧エントリは（`context_cache` の他の `cached_*` と同じ「プロセスの生存期間中 evict しない」設計方針を踏襲し）解放せずリークしたままにするが、以後どの呼び出し経路からも参照されなくなる（`HashMap` の値が新エントリへ置き換わるため）。同種の「`ordinal` キーの `Box::leak` 系プロセス内キャッシュ」を将来追加する場合も、同じ「`current_generation(ordinal)` との突合・不一致時の再構築」契約に従うことを本設計の一般契約として明記する
  5. **遅延エラー・Drop 時エラーの検出経路の一般化（readback・Drop 限定にしない。codex-review P0 指摘 2 への回答）**: `CudaSlice::drop` は cudarc 内部実装でありこのリポジトリから直接フックできない。Drop 中に生じたエラーは `ctx` 側に蓄積され、**readback（`clone_dtoh`）に限らず次に driver API を呼んだ任意の箇所**（`upload`／`clone_htod`・カーネル `launch`・`synchronize`・デバイスメモリ確保等）で表面化しうる。したがって「readback ヘルパ・Drop 実装からのみ `mark_poisoned` を呼ぶ」という当初案（旧版 §9 item 7）は、Drop エラーが readback 以外の driver API 呼び出しで表面化した場合に poison が設定されないまま処理が継続してしまう抜け道を残す。個別の呼び出し箇所ごとに `mark_poisoned` 呼び出しを手書きする方式も見落としのリスクが高いため採用せず、代わりに次の共通経路を設ける: `context_cache` に `observe_driver_result<T>(ordinal, result: Result<T, cudarc::driver::result::DriverError>) -> Result<T, cudarc::driver::result::DriverError>` を追加する。この関数は `Err` の場合に副作用として `mark_poisoned(ordinal)` を呼んでから同じ `Err` をそのまま返す（型を変えない透過的なラッパー）。`crates/backend-cuda/src/{gemm,gemm_wmma,gemm_mma,gemm_mma_tf32,kernels_mma,kernels_mma_tf32,kernels_wmma_opt,elementwise,softmax,rmsnorm,transpose,sgd,memory}.rs` の cudarc `Result<_, DriverError>` を返す driver API 呼び出し箇所（`launch`・`clone_dtoh`・`clone_htod`・`synchronize`・`alloc*` 等）を、この `observe_driver_result` 経由に統一する（#1013 の実装時に機械的に適用できるよう §9 の変更順序へ組み込む）。これにより、Drop エラーがどの driver API 呼び出しで表面化しても、その呼び出し自体が本設計の共通経路を通る限り poison が確実に設定される
  - 学習ループ側の `DeviceParamStore` poisoned 遷移（`BackendError::StorePoisoned`、`crates/tensor-core/src/device.rs:219-230`）は「`step()` 自体の実行時エラー」のみを検出対象とする既存契約であり、`DeviceParamStore` を経由しない素の `BackendOps::gemm` 呼び出し等、任意の `download` で初めて表面化しうる context-wide な遅延エラーはこの経路を通らない。上記 1〜5 は `DeviceParamStore` の外側・`context_cache` 層で完結する別契約として設ける（`StorePoisoned` を置き換えず、両者は独立に有効なまま併存する）
- `BackendError` への新規 variant 追加を**本設計で採用する**（旧版の「行わない」判断を、上記の実装可能性検証の結果として訂正する）。`BackendError::DeviceContextPoisoned(String)`（`ordinal` とトリガー元エラーの要約を保持する文字列）を `#[non_exhaustive]` 列挙体への非破壊追加として `crates/tensor-core/src/device.rs` に追加し、`StorePoisoned` と同様「帰属情報を持たないが継続を拒否する」ことを示す。メッセージには `invalidate(ordinal)` 経由での再生成が必要である旨を含める
- 「同期漏れによって静かに間違った値が返る」失敗はエラーとして検出できない。この種の失敗は §8 のテスト方針で構造的に防ぐ（境界チェックを緩めることでは対処しない。`.claude/rules/coding-rust.md`）

## 6. 複数デバイス・複数 tape・スレッド間の順序保証（design decision 4）

- **同一 ordinal**: 複数の `Tape` は同一ストリームを共有する。**ホスト側の投入順（happens-before）がそのままデバイス側の実行順になる**。異なるスレッドから投入された場合も driver 側で直列化されるが、スレッド間の相互順序自体はホスト側の同期プリミティブ（`Mutex`・チャネル等、呼び出し側の責務）で定める契約とする。`DeviceBuffer` を生産者から消費者へ渡す場合は、同一スレッド上で行うか、ホスト側で happens-before 関係を確立していることを前提とする（I1 の適用条件）
- **異なる ordinal**: ストリームを共有しない。デバイス間でのデータ受け渡しは `download` → `upload` のホスト経由のみを契約とし、それぞれが §4 の同期点となる。P2P・`memcpy_dtod` をまたぐデバイス間直接転送は対象外とし、既存の `BackendError::DeviceMismatch` による拒否契約を維持する
- `cudarc` の `bind_to_thread` は呼び出しごとに行われるため、スレッド間移動自体は許容される（`context_cache` の `Send + Sync` 静的検査と整合する）

## 7. Metal 側（#1016）との契約整合

| 語彙 | CUDA | Metal（#1016 で採用する対応物） |
|---|---|---|
| 投入単位 | ストリームへのカーネル起動 | コマンドバッファ内の encoder dispatch |
| 順序保証 | 単一ストリームの投入順 | 単一 `MTLCommandQueue` + コマンドバッファの commit 順 |
| 同期点 | `download`／ホスト `Tensor` を返す演算 | `download`（`contents()` を読む前に `waitUntilCompleted` が必須。unified memory であっても完了前の読み出しは未定義動作） |
| エラー検出 | 次の同期点で `DriverError` として表面化 | 同期点での `MTLCommandBufferStatus::Error` 検査 |
| 一時バッファの寿命 | `CudaSlice::drop` のイベント待ち | コマンドバッファ完了までの `Retained` 保持 |

`bench_harness::SyncPoint` は両バックエンド共通の「ホスト転送を伴わない完了待ち」ユーティリティとして維持する。#1016 はこの表を出発点として Metal 側の同期点一覧・エラー伝播分類を独自に確定させる。

## 8. テスト方針（#1014 への引き渡し）

- **配置**: `crates/backend-cuda/tests/async_ordering_real_device.rs`（新規・実機依存のため `#[ignore]`）に加え、既存の数値一致回帰テスト群の非後退確認
- **T1（順序依存）**: 同一ストリームで forward → backward → `sgd_step_device` を明示同期なしに連鎖投入し、各段で同期する参照実行と統一複合判定（相対誤差 1e-3 未満 または 絶対誤差 1e-5 未満。緩和しない）で一致することを確認する
- **T2（早期 D2H の検出）**: 実行時間の長いカーネル（大形状 GEMM）投入直後に `download` し、期待値と一致することを確認する。負の対照として、`internal-diagnostics` 限定で「明示同期なしの生 `memcpy_dtoh` を pinned バッファへ発行し不一致を観測する」試験は任意とする（ペイジャブルメモリでは CUDA の実装依存の待ち合わせにより不一致を再現できない場合がある点を明記する）
- **T3（エラー伝播・context poison 遷移）**: 起動時エラー（無効な grid 次元等）が投入 API 呼び出し時点で即座に `KernelLaunchFailed` として返ることを確認する。遅延エラー（カーネル実行時 fault）はテスト側で安全に誘発する手段がない（境界チェックを外すことは規約違反）ため直接再現しないが、§5 で新設する context 単位の poison 状態遷移そのものは以下の 2 段でテスト可能にする（既存の `DeviceParamStore` poisoned モックテストは `step()` 失敗経路のみを検証するものであり、これとは別に新設が必要）:
  - **T3a（`context_cache` 単体）**: `crate::context_cache` のテスト内で `mark_poisoned(ordinal)` → `is_poisoned(ordinal)` が真になる → `invalidate(ordinal)` 後に `is_poisoned(ordinal)` が偽へ戻り `current_generation(ordinal)` が 1 増え `cached_device` 等が再構築されることを、実 CUDA デバイスなしのモック／フェイク実装（`CudaDevice` 生成を要さない範囲）で検証する（CI・GitHub ホステッドで実行可能）
  - **T3c（`static_cuda_memory` の世代追従・§5 item 4）**: `ops.rs` の `static_cuda_memory` キャッシュが `context_cache::current_generation(ordinal)` の変化を検知して新しい `CudaMemory` を構築し直すこと（`bound_generation` 不一致時の差し替え）を、実 CUDA デバイスなしで検証可能な範囲（`current_generation` の値のみを差し替えて分岐を確認する等）でユニットテスト化する。実 `CudaDevice` を要する完全経路の確認は `#[ignore]` の実機依存テストへ委ねる
  - **T3b（呼び出し側の拒否経路・実機依存）**: `#[ignore]` テストとして、実 CUDA デバイス上で `mark_poisoned` を直接呼んでから `ops::CudaBackendOps` の演算メソッド（`gemm` 等）・`sgd_step_device`・`download`／`upload` を呼び出し、いずれも実処理に入らず `BackendError::DeviceContextPoisoned` を即座に返すことを確認する。実際のカーネル fault 注入は行わず（境界チェックを外すことは規約違反）、poison フラグを直接セットする経路のみを検証対象とする（driver 抽象のモック化によるフォールト注入は #1013 実装時に技術的な到達可否を判断し、不可の場合は本 T3b の直接セット方式を正とする）
- **T4（前提確認）**: `CudaContext::has_async_alloc()` が実機で真であることを記録するプローブテスト（`#[ignore]`）。GB10 実機セッションで実測し、本文書の I4 の注記を確定値へ更新する
- CI（GitHub ホステッド・CUDA 非搭載環境）で実行されるのは型検査・モックベースのテストのみであり、実機実行は `make test-ignored-cuda`（または相当の手順）に委ねる

## 9. #1013（実装）への引き渡し事項

変更順序（この順で進めることで、各段階でビルド・既存テストが green のまま維持できる）:

1. `memory.rs` に readback ヘルパ（`clone_dtoh` → `synchronize` の順を一本化する内部関数）を追加する
2. `sgd.rs:174` の `synchronize()` を除去する（最優先。D2H を伴わない唯一の常駐経路であり、除去効果が最も直接的に測れる）
3. ホスト `Tensor` を返すラッパー群（`gemm.rs`・`gemm_wmma.rs`・`gemm_mma.rs`・`gemm_mma_tf32.rs`・`kernels_mma.rs`・`kernels_mma_tf32.rs`・`kernels_wmma_opt.rs`・`elementwise.rs`・`softmax.rs`・`rmsnorm.rs`・`transpose.rs`）の「launch 直後 `synchronize()`」を、readback ヘルパ 1 回への置換に統一する（2.2 の棚卸し表の順で機械的に適用できる）
4. `rmsnorm.rs:1102`／`rmsnorm.rs:1105` のような分岐内二重同期を単一化する
5. `context_cache.rs` に §5 の poison 状態管理を追加する: `ordinal` ごとの `generation: AtomicU64`・`poisoned: AtomicBool`（現在世代に対する状態）・`mark_poisoned`／`is_poisoned`／`invalidate`（世代を進めるアトミック操作。§5 item 3）／`current_generation(ordinal) -> u64`（他キャッシュが鮮度検証に使う公開 API。§5 item 4）・`observe_driver_result`（§5 item 5）を追加する
6. `crates/tensor-core/src/device.rs` の `BackendError` に `DeviceContextPoisoned(String)`（`#[non_exhaustive]` 非破壊追加）を追加する
7. `ops::CudaBackendOps` の各演算メソッド・`sgd_step_device`・`MemoryOps::download`／`upload` の入口に `is_poisoned` 検査を追加し、真であれば `DeviceContextPoisoned` を返す（実処理へ進まない）
8. `crates/backend-cuda/src/{gemm,gemm_wmma,gemm_mma,gemm_mma_tf32,kernels_mma,kernels_mma_tf32,kernels_wmma_opt,elementwise,softmax,rmsnorm,transpose,sgd,memory}.rs` の cudarc driver API 呼び出し箇所を `context_cache::observe_driver_result` 経由へ統一し、遅延エラー・Drop 時エラーがどの呼び出しで表面化しても `mark_poisoned` が漏れなく呼ばれるようにする（§5 item 5。readback ヘルパ・Drop 実装への個別手書き呼び出しはしない）
9. `ops.rs` の `static_cuda_memory` を `(bound_generation: u64, mem: &'static CudaMemory)` キャッシュへ拡張し、取得時に `context_cache::current_generation(ordinal)` と突合して不一致なら新世代の `CudaDevice` 上に新しい `CudaMemory` を構築・差し替える（§5 item 4。`context_cache::invalidate` だけでは波及しない `ordinal` キー付き別系統キャッシュへの世代伝播）

**触らないもの**: カーネル本体・手動境界チェック・数値一致の許容誤差・`memory.rs:276`（`download` の同期。維持対象）・診断/ベンチ専用テストの同期呼び出し。

**性能記録テンプレート**: `bench-fandhe train cuda` の 1 step 実行時間を before/after で記録する（5 回計測の中央値。`.claude/rules/coding-rust.md`）。本文書の執筆環境（Linux x86_64・RTX 3060、DGX Spark 非到達）では計測を実施していないため、実測は DGX Spark GB10 セッションで行い #1013 の PR に記録する。

## 10. 代替案と採否

| 代替案 | 採否 | 理由 |
|---|---|---|
| 多ストリーム化 + イベント DAG | 不採用 | 現行 API はホスト `Tensor` 中心であり並列度を活かせない。順序契約が複雑化し I1〜I5 の検証コストが増す |
| CUDA Graph | 保留 | #1024（起動キャッシュの結線）完了後に再検討する |
| pinned host memory | 保留 | 別イシューで扱う。導入時は I3・`download` の同期契約（2.4）を再確認する必要がある |
| `CudaContext::set_blocking_synchronize` | 不採用 | ホスト側のスピン待ちに切り替える効果がベンチ条件と無関係であり、影響が不明である |

## 11. スコープ外・後続イシュー候補

以下はユーザー承認なしに起票しない（`.claude/rules/out-of-scope-tracking.md`）。

- pinned host memory の導入
- CUDA Graph の導入
- ホスト `Tensor` API の `DeviceBuffer` 版への拡張（#1022 と重なる可能性がある）

## 12. 出典

- `scripts/bench/framework-compare/results/summary.md`（MLP 学習 1 step 実測値）
- `crates/backend-cuda/src/device.rs`（ストリーム構成・`context_cache` 連携）
- `crates/backend-cuda/src/ops.rs`（`static_cuda_memory` の `Box::leak` プロセス内キャッシュ・`memory_ops` 経由の生成契約。§5 item 4・イシュー #935）
- `crates/backend-cuda/src/{gemm,gemm_wmma,gemm_mma,gemm_mma_tf32,kernels_mma,kernels_mma_tf32,kernels_wmma_opt,elementwise,softmax,rmsnorm,transpose,sgd,memory}.rs`（`synchronize()` 棚卸し）
- `crates/tensor-core/src/backend_ops.rs`（`BackendOps` トレイト定義）
- `crates/tensor-core/src/buffer.rs`（「download の同期契約」節）
- `crates/tensor-core/src/device.rs`（`BackendError` 定義・`StorePoisoned` 等の拡張 variant）
- `crates/bench-harness/src/sync.rs`（`SyncPoint`）
- `crates/backend-metal/src/context.rs`・`crates/backend-metal/src/memory.rs`（Metal 側現状）
- cudarc 0.19.8（`Cargo.toml:95`）の `src/driver/safe/{launch,core}.rs` 相当（`clone_dtoh`・event tracking・`has_async_alloc`・Drop 時エラー蓄積の意味論）
- `docs/spec/04-requirements.md` REQ-2（バックエンド間同期方式・FMA 契約）・REQ-8（性能下限・`bench-harness` の計測境界）
- `docs/device-resident-update-design.md`（#934・同型の設計文書構成・`DeviceParamStore` poisoned 契約）
