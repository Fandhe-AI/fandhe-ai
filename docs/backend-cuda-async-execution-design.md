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

- 遅延エラー・Drop 時エラーの発生後は、同一 `CudaContext` での処理継続を禁じる（fail-closed）。ただし継続禁止を実効化するには「誰が・どの状態を・どこで検査して拒否するか」を契約化する必要があり、2.6 の `context_cache` の現状（「エントリはプロセスの生存期間中 evict されない」・`context_cache.rs` 冒頭コメント「所有モデル・生存期間」節）はエラー検出後の**自動的な**回復手段を持たない。したがって本設計は以下の点を #1013 の実装対象として明文化する（レビュー指摘 PR #1053・本ファイル該当行への回答。世代管理までを導入した前回改訂〈`generation: AtomicU64` + `poisoned: AtomicBool` の別々の atomics・compare-and-mark を `invalidate` と同一 `Mutex` 区間に閉じる方式〉は、下記 codex-review P0 2 件〈TOCTOU による fail-closed 迂回・キャッシュ再構築のみでの回復確認不足〉により、状態を単一の `Mutex<OrdinalState>` へ統合し「in-flight 呼び出しの世代別ゲート・drain」と「回復の事後検証」を追加する必要があると判明したための再改訂）:
  1. **単一 `Mutex` に統合した ordinal 単位の状態機械（in-flight ゲートを内包する。codex-review P0 指摘 2 件への回答の土台）**: `context_cache` の `ordinal` キー空間に、これまでの `generation: AtomicU64`／`poisoned: AtomicBool` という**独立した 2 つの atomics** を廃し、単一の `Mutex<OrdinalState>` へ統合する。

     ```text
     struct OrdinalState {
         generation: u64,        // 初期値 0。invalidate が回復を確認できた場合のみ 1 増える
         phase: Phase,           // Active | Retiring | Poisoned { unrecoverable: bool }
         in_flight: u64,         // 現在の generation に対して begin_driver_call 済み・未解放の CallToken 数
     }
     enum Phase { Active, Retiring, Poisoned { unrecoverable: bool } }

     /// begin_driver_call が返す RAII ハンドル。in_flight の解放（-1・Condvar 通知）は
     /// Drop 実装が担う（PR #1053 codex-review P1 指摘「CallToken に Drop/RAII がなく
     /// begin 後に `?`・引数準備失敗・panic 等で破棄されると in_flight が永久に減算され
     /// ず invalidate の drain が停止しうる」への回答。§5 item 3 参照）。
     struct CallToken {
         ordinal: usize,
         generation: u64,
     }
     ```

     `poisoned` を単独の bool として読める旧 API（`is_poisoned`）は `phase` からの導出値（`matches!(phase, Phase::Poisoned { .. })`）として維持し、呼び出し側（演算メソッド入口）の契約は変えない。`in_flight` を同じ `Mutex` に同居させることで、次項 2〜3 の「状態検査から driver 呼び出し登録まで」と「invalidate の世代遷移判定」を**単一の排他区間**で行えるようにする（これが P0 指摘 1〈TOCTOU〉を解消する鍵であり、旧版のように「compare-and-mark だけを排他化し、検査から呼び出し登録までは排他区間の外」という構成を取らない）。`CudaGemm`・`CudaElementwise`・`CudaRmsNorm`・`CudaSoftmax`・`CudaSgd`・`CudaMemory` が同一 `Arc<CudaStream>`（延いては同一 `CudaContext`）を共有する現行構成（2.1）を踏まえ、状態は `ordinal` ごとに 1 つで全スイートへ波及させる
  2. **driver 呼び出しの登録・解放（in-flight ゲート ＋ 世代検査を単一区間へ統合。codex-review P0 指摘 1〈TOCTOU〉への回答）**: `context_cache` に `begin_driver_call(ordinal: usize, resource_generations: &[u64]) -> Result<CallToken, BackendError>` を追加する。**世代検査（旧版では呼び出し元入口の独立した事前検査ステップだったもの。旧項 5）はこの関数の引数として渡し、`Mutex<OrdinalState>` を取得した同一区間内で phase 検査・in-flight 登録とアトミックに行う**契約へ改める。旧版は「① 引数の `DeviceBuffer` 等の `generation` を `current_generation(ordinal)` と比較 → （ロックを跨ぐ窓）→ ② `begin_driver_call(ordinal)` で phase 検査・in-flight 登録」という 2 段構成であり、①と②の間に別スレッドの `invalidate` が retire・drain・世代インクリメントまで完了できてしまうと、①を通過済みの呼び出しが②で新世代の `CallToken` を発行されて in-flight 登録される（世代検査は旧世代に対して行われたのに、実際に driver へ渡る呼び出しは新世代の状態の下で進む）という TOCTOU が残っていた。本改訂はこの窓を閉じるため、世代検査を `begin_driver_call` 自身の責務として統合する。`ops::CudaBackendOps` の各演算メソッド（`gemm`／`add`／`mul`／…）・`sgd_step_device`・`MemoryOps::download`／`upload` は、実際の driver 呼び出しに入る**直前**に、引数として渡された各リソース（`DeviceBuffer` 等）の `generation` フィールドをまとめて `resource_generations` として渡してこれを呼ぶ。実装は `Mutex<OrdinalState>` を取得したうえで、以下の順で検査する:
     - `phase` が `Poisoned { .. }` なら、実処理に入らず `BackendError::DeviceContextPoisoned`（`unrecoverable` なら後述の `BackendError::DeviceContextUnrecoverable`）で即座に拒否する（旧 `is_poisoned` 検査と同じ役割）
     - `phase` が `Retiring`（後述の `invalidate` が同じ generation を drain 中）なら、新規呼び出しをその generation に登録させず `BackendError::DeviceContextRetiring { ordinal }`（`#[non_exhaustive]` 非破壊追加。一時的で短時間のうちに解消する契約であり、呼び出し元は短い再試行またはエラーとしての伝播のいずれでもよい）で拒否する
     - `phase` が `Active` かつ `resource_generations` のいずれかが `state.generation` と不一致なら、`in_flight` を増やさずに `BackendError::StaleDeviceGeneration { ordinal, resource_generation, current_generation: state.generation }`（後述）で拒否する（**この分岐が P0 指摘 1 の解消そのものである**: 世代の確認と in-flight 登録可否の決定を同一ロック内・同一呼び出しで行うため、「確認が通った直後に世代が進む」窓が構造的に存在しない）
     - `phase` が `Active` かつ全ての `resource_generations` が `state.generation` と一致するなら `in_flight += 1` し、`CallToken { generation: state.generation }` を返す

     これにより「検査（`is_poisoned`／世代一致）と driver 呼び出しの実際の投入登録」が**同じ排他区間内でアトミックに**行われ、検査通過直後に別スレッドの `invalidate` が世代を進めてしまう窓（旧版の欠陥）が閉じる。`CallToken` の生成自体が「この呼び出しは現在の generation に対して、かつ渡された全リソースが現在の generation と一致することを確認したうえで、in-flight としてカウント済み」であることの証跡になる。driver 呼び出し本体（`launch`・`clone_dtoh` 等）は `Mutex` の外で従来どおり非同期に実行してよく、並行度は損なわれない

     **`begin_driver_call` は「演算 1 回」につき 1 回だけ呼ぶ（PR #1053 codex-review P1 指摘「1 演算内で複数 driver API 呼び出し〈`clone_htod`・`alloc*`・`launch`・`synchronize`・`clone_dtoh`〉を発行するため 1 個の `CallToken` では全結果を監視できない」への回答）**: `gemm`・`sgd_step_device`・`download` 等 1 回の演算呼び出しは内部で複数の driver API を順に発行しうるが、`begin_driver_call` は演算入口で 1 回だけ呼び、返る `CallToken` を演算の最後まで**所有したまま**保持する。in-flight という概念自体が「1 個の driver 呼び出し」ではなく「1 個の演算（=1 個の generation 一貫性検査を経た作業単位）」を数える契約であり、演算内部の個々の driver 呼び出しの結果はそれぞれ次項 3 の `observe_driver_result(ordinal, &token, result)` に**トークンを借用で**渡して分類する（1 個の `CallToken` に対し `observe_driver_result` を複数回呼んでよい）。`in_flight` の減算は個々の `observe_driver_result` 呼び出しではなく `CallToken` の Drop（次項 3）で 1 回だけ行われるため、演算の最後の driver 呼び出しが終わるまで in_flight は減らず、`invalidate` の drain は「演算全体の完了」を正しく待つ
  3. **`observe_driver_result` は `CallToken` を借用しエラー分類のみを行う。in-flight の解放は `CallToken` の Drop に一本化する（PR #1053 codex-review P1 指摘 2 件〈CallToken が複数 driver 呼び出しを監視できない／Drop/RAII 欠如による drain 永久停止〉への回答）**: `context_cache::observe_driver_result<T>(ordinal: usize, token: &CallToken, result: Result<T, cudarc::driver::result::DriverError>) -> Result<T, cudarc::driver::result::DriverError>` は `token` を**消費せず借用**で受け取り、`Mutex<OrdinalState>` を取得してエラー分類・sticky 判定時の poison 化（後述 item 5 の a〜c）のみを同一区間内で行う（`in_flight` は変更しない）。同一演算内の複数 driver 呼び出しそれぞれについてこの関数を呼べる（item 2 の追記）。`in_flight -= 1` と `phase == Retiring && in_flight == 0` になった場合の `invalidate` 側 `Condvar` 通知は、`CallToken` の **`Drop` 実装**が担う: 演算が正常終了してトークンがスコープを抜けるときはもちろん、演算の途中で `?` によるアーリーリターン・引数準備失敗・panic（`Drop` は unwind 時にも走る）が起きても Drop は必ず実行されるため、in_flight の解放漏れが構造的に起こらない（旧版は解放を呼び出し元の明示的な消費に依存しており、上記のいずれかの経路でトークンが破棄されると `in_flight` が減らず `invalidate` の drain が永久に完了しない欠陥があった）。**`begin_driver_call` で得た generation と現在の `state.generation` が不一致になることは、この設計では原理的に起きない**（`invalidate` は同じ generation の `in_flight` が 0 になるまで `Retiring` のまま世代を進めないため）。したがって旧版が抱えていた「compare-and-mark の対象 generation が呼び出し開始時点から変わっている」という不整合そのものが構造的に排除される（比較のための世代引数の受け渡しは、生成された `CallToken` が既に運ぶため独立には行わない）
  4. **明示的な再生成手段（`invalidate`）: retire → drain（in-flight 解放待ち）→ ストリーム完了同期〈失敗時はここで `Poisoned { unrecoverable: true }` に確定し独立検証へは進まない〉→ 独立検証（実処理プローブ。sync 成功時のみ到達）→ 公開、の順で世代を進める（codex-review P0 指摘 1・2 双方、および PR #1053 二巡目 codex-review P0・Cursor Bugbot High 指摘への回答）**: `context_cache` に `invalidate(ordinal) -> Result<(), BackendError>` を追加する。当初案（`cached_device`／各 `cached_*` エントリを `None` へ戻し `poisoned` フラグを直ちにクリアするのみ）は、(i) 旧世代のハンドルを保持し続ける呼び出し元・in-flight 中の driver 呼び出しが新世代の解放後も実行され続け得る（TOCTOU。codex-review P0 指摘 1）、(ii) `poisoned = false` への復帰がキャッシュ再構築という Rust 側の操作でしかなく、CUDA 側の sticky error が実際に解消したかを確認していない（fail-open。codex-review P0 指摘 2）という 2 つの欠陥を持つため採用しない。改訂案の `invalidate` は次の 5 段階を踏む:

     a. **retire（複数スレッドからの並行 `invalidate` 呼び出しに対する所有権の一本化、および回復可能な `Poisoned` からの retire 起動。PR #1053 codex-review P1 指摘「並行 invalidate が同一世代を重複公開し generation を二重に進めうる」・別件 P1 指摘「`invalidate` の retire 分岐が `Poisoned { unrecoverable: true }`／`Retiring`／`Active` の 3 つしか定義しておらず、`observe_driver_result`（item 7c）が sticky エラーで遷移させる `Poisoned { unrecoverable: false }` からは b〜e の回復処理を開始できず、item 5 が明記する公開回復手順（`DeviceContextPoisoned` を受け取った呼び出し元が `invalidate(ordinal)` を呼んで再生成する）が実行不能だった」への回答）**: `Mutex<OrdinalState>` を取得し、`phase` に応じて 3 分岐する。① `phase` が `Poisoned { unrecoverable: true }` であれば、それ以上何もせず同じエラー（`BackendError::DeviceContextUnrecoverable`）を返して終了する（回復不能な ordinal に対する `invalidate` は恒久的に no-op）。② `phase` が既に `Retiring` である場合（別スレッドが同じ generation に対する `invalidate` を実行中）は、**この呼び出しはその実行の「所有者」にならず**、同じ `Mutex` 上の `Condvar` で `phase` が `Retiring` でなくなるまで待機し（`Mutex` は待機中解放される）、`Retiring` を抜けた時点の最新の `phase`（`Poisoned`〈b'／c で失敗と判明した〉か `Active`〈e で公開済み〉のいずれか）を読み取り、`Poisoned { unrecoverable: true }` なら `DeviceContextUnrecoverable` を、`Poisoned { unrecoverable: false }` なら `DeviceContextPoisoned` を、`Active` なら `Ok(())` を、それぞれ**先行呼び出しが確定させた結果として**そのまま返して終了する（この呼び出し自身は b〜e を再実行しない。③へ遷移して retire の所有権を新たに得ることはなく、公開済み世代の二重 retire・`generation` の二重インクリメントは構造的に起こらない）。③ `phase` が `Active` または `Poisoned { unrecoverable: false }`（**回復可能な poison。呼び出し元が item 5 の公開回復手順に従い `DeviceContextPoisoned` を受けて `invalidate` を呼ぶ経路はここに入る**）である場合、この呼び出しが retire の「所有者」となり、対象 generation の `phase` を `Retiring` に設定して b へ進む（`Active` からの遷移では、この時点から item 2 の `begin_driver_call` は新規呼び出しをこの generation に対して受け付けなくなる。すでに `Active` として発行済みの `CallToken` を持つ in-flight 呼び出しはそのまま実行を継続してよい。`Poisoned { unrecoverable: false }` からの遷移では、item 2 の phase 検査〈`Poisoned` なら即座に拒否〉により**新規呼び出しの登録は既に止まっている**ため `in_flight` はこの時点から単調非増加であり、続く b の drain は既発行の `CallToken` が `Drop` されるのを待つだけでよい（残存が無ければ即座に完了する））。この一本化により、生成をまたぐ b〜e（drain・ストリーム同期・独立検証・公開）を実行できるのは常に高々 1 スレッドであり、`generation` の二重インクリメント・「健全な新世代を別呼び出しがさらに退役させる」という不整合は構造的に起こらない。以降 b〜e の記述で `phase` の遷移が起きるたびに（`Retiring` から抜けるたびに）この `Condvar` を `notify_all` する契約とする（b の drain 通知用 `Condvar` と同一のものを再利用してよい）

     b. **drain（in-flight 解放待ち）**: 同じ `Mutex` 上の `Condvar` で `in_flight == 0` になるまで待機する（`Mutex` は待機中解放される。標準の `Condvar::wait_while` 相当）。これにより「旧世代の in-flight 呼び出しがなくなるまで新世代を有効化しない」というライフサイクル契約が成立し、drain 完了時点で「この generation に対する新規 driver 呼び出しは今後一切発生しない」ことが保証される（新規呼び出しは a で `Retiring` として拒否され、既存呼び出しは全て `observe_driver_result` を経て `in_flight` から抜けきっている）。**この保証が codex-review P0 指摘 1 の「旧世代の in-flight 呼び出しと invalidate の競合」を解消する**

     b'. **ストリーム完了同期（codex-review P0 指摘 2〈drain が非同期 CUDA 作業の完了を待たずに旧世代を退役させる〉、および PR #1053 二巡目 codex-review P0・Cursor Bugbot High 指摘「sync 失敗を軽量プローブの成功で覆す fail-open」への回答）**: `in_flight == 0` は `begin_driver_call` に対応する driver API 呼び出し（`launch`・`clone_dtoh` 等）が**ホストへ復帰したこと**しか意味せず、ストリーム上でカーネル・転送が**実際に完了したこと**は保証しない（`launch` は `cuLaunchKernel` の非同期発行であり、復帰時点でカーネルは実行中でありうる。2.4）。旧世代の作業が実行され続けたまま b で新世代への遷移が進むと、旧世代由来の遅延 fault が新世代の呼び出しへ表面化しうる（P0 指摘 2 が指すシナリオ）。そこで b の `in_flight == 0` 確認直後、`Mutex` を解放したうえで当該 ordinal の `Arc<CudaStream>`（2.1 の「ordinal ごとに単一ストリーム」構成）に対し明示的な `stream.synchronize()`（ブロッキング）を 1 回発行し、旧世代として投入された全ての作業（カーネル起動・H2D/D2H 転送）がデバイス側で実際に完了するまで待つ。単一ストリーム構成のため、投入順に実行される全作業はこの 1 回の `synchronize()` で漏れなくカバーされ、呼び出しごとの CUDA event 記録は不要である。**この `synchronize()` が失敗した場合、`Mutex<OrdinalState>` を取得し直して `generation` は進めず `phase` を直ちに `Poisoned { unrecoverable: true }` に確定させ、`BackendError::DeviceContextUnrecoverable { ordinal, probe_error: <この synchronize が返したエラー> }` を返して `invalidate` を終了する（c へは進まない）**。旧案（初版・PR #1053 一巡目レビューで修正）は synchronize のエラーを検出した時点で独立検証（c）へ進まず即座に「回復可能な poison」として終了する fail-closed 側の欠陥を持っていたが、その修正として導入した「synchronize の結果に関わらず必ず c へ進み、c の軽量プローブだけを唯一の判定根拠とする」構成（PR #1053 一巡目で採用）は、二巡目レビューで別の欠陥が判明したため本改訂で撤回する: 当該 `Arc<CudaStream>` は同一 ordinal の primary context 上で動作しており、c で構築する「新しい」ハンドルも同じ primary context を再 retain する（c 参照）。したがって synchronize がこの primary context 上で実際に観測した失敗を「旧世代限りの残存エラーであり新世代の健全性を否定しない」として退けることはできない（同じ context を指している以上、退役させる世代の失敗と公開しようとする世代の健全性は不可分である）。加えて Bugbot 指摘のとおり `cuCtxGetApiVersion` 等の軽量プローブは成功か `CUDA_ERROR_INVALID_VALUE`／`CUDA_ERROR_INVALID_CONTEXT` 相当しか返しえず、sticky な実行時 fault（`CUDA_ERROR_ILLEGAL_ADDRESS` 等）が実際に発生していても健全に見えてしまう。したがって synchronize の失敗は、それ自体を「回復不能」の十分条件として扱う（**synchronize は c より前段の、それだけで確定的な fail-closed ゲートである**）。synchronize が成功した場合に限り c（独立検証。実処理を伴う強化プローブへ改訂）へ進む。この改訂により、真に破壊された primary context は synchronize の失敗をもって有限回（1 回）で確実に `DeviceContextUnrecoverable`（d）へ到達し、軽量プローブの成功による無限の「回復可能」誤判定・再試行ループが構造的に起こらない（codex-review 二巡目 P0・Bugbot High 指摘の両方を解消する）

     c. **独立検証（実処理を伴う強化プローブ。PR #1053 二巡目 Cursor Bugbot High 指摘「`cuCtxGetApiVersion` は成功か invalid-value／invalid-context 相当しか返さず sticky fault を健全と誤判定しうる」への回答）**: b' のストリーム完了同期が成功した場合にのみ到達する（b' が失敗した場合は上記のとおり c を経由せず d へ確定する）。`Mutex` は解放してよく、以降 CUDA 呼び出し自体はブロッキングな同期呼び出しでよく、他の ordinal の処理と競合しない。新しい `CudaContext` を構築する。ここでの「新しい」は Rust 側でハンドルを作り直すことではなく、**CUDA 側で実際に計算・転送パイプラインが機能する健全な context であることを、実処理を伴うプローブで確認できたもの**を指す契約とする: 具体的には (i) 小さな固定サイズのデバイスメモリを 1 個 `alloc`、(ii) 既知の入力パターンを `clone_htod` で H2D 転送、(iii) 恒等（入力をそのまま出力へコピーする、または `+0` 等の副作用のない）カーネルを 1 回 `launch`、(iv) `clone_dtoh` で D2H readback、(v) 読み戻した値が (ii) で送った入力パターンと一致することを確認、の 5 段階を新ハンドル上で順に実行し、全段階が成功しかつ (v) の値照合が一致する場合にのみ「回復した」とみなす。`cuCtxGetApiVersion` のような副作用のない同期問い合わせのみを行うプローブは、launch・転送のいずれも実行しないため sticky な実行時 fault を検出できず（Bugbot 指摘のとおり成功か invalid-value／invalid-context 相当しか返らない）、本設計では回復確認の根拠として採用しない。(i)〜(v) のいずれかの段階が失敗する、または (v) の値照合が不一致の場合は d へ、全段階が成功し値照合も一致した場合は e へ進む（同一 ordinal の primary context を再 retain する cudarc の実装では、Rust 側のハンドルを作り直すだけでは CUDA 側の sticky error 状態は消えないため、この実処理確認を経て初めて「回復した」とみなす契約とする）

     d. **回復不能の確定（fail-closed を維持。codex-review P0 指摘 2、および二巡目 codex-review P0・Bugbot High 指摘への回答）**: 到達経路は 2 通りある: (i) b' のストリーム完了同期そのものが失敗した場合（c を経由せず直接ここへ確定する）、(ii) b' が成功したうえで c の実処理プローブ（5 段階のいずれか、または値照合）が失敗した場合。いずれの経路でも `generation` は進めず `phase` を `Poisoned { unrecoverable: true }` に設定し、`BackendError::DeviceContextUnrecoverable { ordinal, probe_error: String }`（`#[non_exhaustive]` 非破壊追加。`probe_error` には (i) なら synchronize が返したエラー、(ii) なら c の失敗した段階のエラーまたは値照合の不一致内容を格納する。メッセージに「同一プロセス内での回復手段はなく、当該デバイスを使うプロセスの再起動が必要である」旨を含める）を返す。以後この ordinal に対するあらゆる `begin_driver_call`／`invalidate` は a の分岐によって同じエラーを返し続ける（**「回復不能時はプロセス再起動を要求する型付きエラーとして fail-closed に維持する」契約そのもの**）
     e. **公開**: プローブが成功した場合に限り、`generation` を 1 増やし、`phase` を `Active` に戻し、`in_flight` を 0 のまま維持し、`cached_device`／各 `cached_*` エントリを検証済みの新しい `CudaContext` から再構築する。この時点で初めて新世代が「非 poisoned」として他スレッドから観測可能になる

     旧版が前提としていた「旧世代のハンドルを保持し続けている呼び出し元がいても、次の呼び出しで新世代のキャッシュから取り直した非 poisoned なハンドルに置き換わる」という運用上の期待（**呼び出し元がハンドルを長期保持しない**という検証されない前提）は、上記の drain（旧世代の新規 driver 呼び出し自体を発生させない）・ストリーム完了同期（b'。旧世代の投入済み作業の実完了を保証する）・世代検査（下記 item 5）の組み合わせで型検査による保証へ置き換える。CUDA 経由で確保される各リソース（`DeviceBuffer`〈`crates/tensor-core/src/buffer.rs`〉、および `CudaGemm`／`CudaElementwise`／`CudaRmsNorm`／`CudaSoftmax`／`CudaSgd`／`CudaMemory` が内部的に保持する `Arc<CudaContext>`／`Arc<CudaStream>`）に、生成時点で `context_cache::current_generation(ordinal)` を読み取った `generation: u64` フィールドを持たせる（CPU バックエンド等 ordinal を持たない `DeviceBuffer` では検査自体を no-op とし、CUDA 経由で確保されたものにのみ有効値を持たせる）
  5. **世代跨ぎハンドルの拒否（検査本体は item 2 の `begin_driver_call` に統合済み。本項は付随する契約を記す）**: `DeviceBuffer` 等の `generation` が `context_cache::current_generation(ordinal)` と一致するかの検査は、item 2 の `begin_driver_call(ordinal, resource_generations)` が phase 検査・in-flight 登録と**同一のロック区間で**行う。呼び出し元入口で別途「事前検査 → 別途 `begin_driver_call`」という 2 段構成は取らない（2 段構成では確認から登録までの間に世代が進む窓が残り、P0 指摘 1 を再発させる。詳細は item 2）。不一致時は `BackendError::StaleDeviceGeneration { ordinal, resource_generation: u64, current_generation: u64 }`（後述）で拒否され、`is_poisoned` の値に関わらず機械的に拒否される。これにより、`invalidate` 後に呼び出し元が偶然（あるいは誤って）保持し続けていた旧世代の `DeviceBuffer` を新世代の演算へ渡した場合も確実に拒否される。**世代を跨いだハンドルは意味的に別物として扱う契約**であり、`is_poisoned(ordinal)` は常に「現在の `generation` における」状態だけを答える。加えて `context_cache` は `current_generation(ordinal) -> u64` を公開する（次項 6 のとおり `context_cache` 以外の `ordinal` キー付きプロセス内キャッシュが自分のエントリの鮮度を検証するため、および呼び出し元が `DeviceBuffer` 生成時に埋め込む `generation` フィールドの参照点として必要）。**呼び出し元の再試行契約はエラー種別で分岐する（PR #1053 codex-review P1 指摘「`StaleDeviceGeneration`〈現行世代自体は健全〉に対しても `DeviceContextPoisoned` と同様に `invalidate(ordinal)` を呼ぶ指示になっており、健全な現行世代を不要に退役させ再構築済みの他ハンドルまで stale 化する」への回答）**: `StaleDeviceGeneration` は「現行世代は健全（非 poisoned）だが、呼び出し元が渡したハンドルが古い世代のものだった」ことしか意味しない（item 5 冒頭の定義）。したがって呼び出し元は `invalidate(ordinal)` を呼んではならず、単に手元の古いハンドル（`DeviceBuffer` 等）を破棄し、`context_cache`（`cached_device` 等）または自身のキャッシュ層から `current_generation(ordinal)` に一致する**最新のハンドルを取り直す**（例: `tape_for(Device::Cuda(ordinal))` を再度呼ぶ）だけでよい。ここで誤って `invalidate(ordinal)` を呼ぶと、健全な現行世代を不要に `Retiring` 化し、その時点で他の呼び出し元がすでに取得・使用中の現行世代ハンドルまで次の世代境界で stale 化してしまう（本 P1 指摘が指す不整合）。一方 `DeviceContextPoisoned`／`DeviceContextUnrecoverable` を受け取った場合（context 自体が汚染されている場合）に限り、呼び出し元は `invalidate(ordinal)` を呼んでから次の `tape_for(Device::Cuda(ordinal))` を行うことで、新しい（プローブ検証済みの）`CudaContext` を明示的に再構築できる。`invalidate` が `DeviceContextUnrecoverable` を返した場合は呼び出し元に伝播し、当該デバイスでの処理を諦めてプロセス再起動を促す
  6. **世代契約の伝播範囲（`context_cache` 外のプロセス内キャッシュへの適用。Bugbot 指摘への回答）**: `ordinal` をキーとするプロセス内キャッシュは `context_cache` の `cached_*` 系列だけではない。`crates/backend-cuda/src/ops.rs` の `static_cuda_memory`（`Box::leak` で `'static` 化した `CudaMemory` を `ordinal` キーの `HashMap` で保持する、`context_cache` とは別系統のプロセス内シングルトンキャッシュ。イシュー #935）は、内部に `Arc<CudaStream>`（延いては旧 `CudaContext`）を保持する `CudaMemory` を返すため、`context_cache::invalidate(ordinal)` を呼んでもこのキャッシュは更新されず、`memory_ops()` 経由で取得される `MemoryOps`（`download`／`upload`）が旧 poisoned context を指したまま使われ続ける（codex-review 指摘・Bugbot 指摘「Invalidate skips leaked memory cache」はこの経路を指している）。これを解消するため、`static_cuda_memory` のキャッシュ値を `(bound_generation: u64, mem: &'static CudaMemory)` へ拡張する契約とする: 取得時に `context_cache::current_generation(ordinal)` を読み、キャッシュ済みエントリの `bound_generation` と不一致であれば「新しい世代の `CudaDevice`（`context_cache::cached_device` から取り直したもの）上に新しい `CudaMemory` を `Box::leak` で構築し、`(現在の generation, 新エントリ)` で `HashMap` を上書きする」。旧エントリは（`context_cache` の他の `cached_*` と同じ「プロセスの生存期間中 evict しない」設計方針を踏襲し）解放せずリークしたままにするが、以後どの呼び出し経路からも参照されなくなる。同種の「`ordinal` キーの `Box::leak` 系プロセス内キャッシュ」を将来追加する場合も、同じ「`current_generation(ordinal)` との突合・不一致時の再構築」契約に従うことを本設計の一般契約として明記する
  7. **遅延エラー・Drop 時エラーの検出経路の一般化とエラー分類（readback・Drop 限定にしない、回復可能エラーで fail-close しない）**: `CudaSlice::drop` は cudarc 内部実装でありこのリポジトリから直接フックできない。Drop 中に生じたエラーは `ctx` 側に蓄積され、**readback（`clone_dtoh`）に限らず次に driver API を呼んだ任意の箇所**（`upload`／`clone_htod`・カーネル `launch`・`synchronize`・デバイスメモリ確保等）で表面化しうる。したがって「readback ヘルパ・Drop 実装からのみ `mark_poisoned` を呼ぶ」という当初案は、Drop エラーが readback 以外の driver API 呼び出しで表面化した場合に poison が設定されないまま処理が継続してしまう抜け道を残す。個別の呼び出し箇所ごとに手書きする方式も見落としのリスクが高いため採用せず、item 2・3 の `begin_driver_call`／`observe_driver_result` という共通経路（全 driver 呼び出し箇所に機械的に適用する。§9 参照）へ一本化する。`observe_driver_result` 内でのエラー分類・poison 化は次のとおり:

     a. **エラー分類**（Bugbot High 指摘「同期 `launch`／`alloc*` 失敗を含む全 `DriverError` で ordinal を poison するのは過剰」への回答）: `e` が保持する raw `CUresult` を、CUDA の「sticky（一度発生するとコンテキスト全体が以後の全 API 呼び出しでエラーを返し続ける致命的エラー）」区分と「operation-local（当該呼び出し単体の入力に起因し、他の投入済み作業の正しさに影響しない回復可能エラー）」区分に分類する。sticky 側の代表例は `CUDA_ERROR_ILLEGAL_ADDRESS`・実行時 fault による `CUDA_ERROR_LAUNCH_FAILED`・`CUDA_ERROR_ECC_UNCORRECTABLE`・`CUDA_ERROR_HARDWARE_STACK_ERROR`・`CUDA_ERROR_MISALIGNED_ADDRESS`・`CUDA_ERROR_INVALID_ADDRESS_SPACE`・`CUDA_ERROR_ASSERT`（実装時に確定リストを `context_cache.rs` 冒頭コメントへ明記する）。operation-local 側の代表例は `CUDA_ERROR_INVALID_VALUE`・`CUDA_ERROR_INVALID_HANDLE`・`CUDA_ERROR_OUT_OF_MEMORY`・`CUDA_ERROR_INVALID_IMAGE`。**分類が未知の `CUresult` に対しては安全側（sticky 扱い・poison する）に倒す**
     b. **operation-local と判定した場合**: poison せず `Err(e)` をそのまま返す（呼び出し元で `BackendError::KernelLaunchFailed`／`BackendError::TransferFailed` 等の既存の型付きエラーへ変換する）
     c. **sticky と判定した場合**: `Mutex<OrdinalState>`（item 3 のエラー分類と同一区間）を取得し、**現在の `phase` に応じて分岐する**（PR #1053 codex-review P1 指摘「sticky error が Retiring を上書きし drain 通知条件〈`phase == Retiring && in_flight == 0`〉を満たせず invalidate が永久待機しうる」への回答）: `phase` が `Active` なら `Poisoned { unrecoverable: false }` に設定する（従来どおり）。`phase` が既に `Retiring`（`invalidate` がこの generation を drain 中）である場合は **`phase` を上書きしない**（`Retiring` のまま維持する）。ここで `Poisoned` へ書き換えてしまうと item 4 b の drain 完了条件（`phase == Retiring && in_flight == 0`）が「`phase == Retiring`」を満たせなくなり、`invalidate` が `Condvar` の通知を待ったまま永久に復帰しない（T3e が回帰防止対象とする不整合）。`phase` を `Retiring` のまま維持しても安全性は失われない: この sticky を明示的に記録し直さなくても、item 4 b' のストリーム完了同期（drain 後に必ず実行される）が同じ contamination を独立に検出する。実際に fault が sticky であれば、この b' の `synchronize()` 自体が失敗し、b' の定義（item 4 b'）どおりただちに `Poisoned { unrecoverable: true }`（d）へ確定する。したがってこの分岐は「今すぐ `Poisoned` にする責務を invalidate 側の b' へ委譲する」ことを意味し、poison 化そのものを省略するわけではない。この設計では `CallToken` が発行された時点で generation は確定しており、かつ `invalidate` は同じ generation の `in_flight` が 0 になるまで進行しない（item 4 の drain）ため、**「死につつある旧世代のエラーで既に切り替わった新世代を汚染する」という不整合は構造的に起こり得ない**（旧版の compare-and-mark が担っていた役割を、世代遷移そのものの直列化で代替する）

     `crates/backend-cuda/src/{gemm,gemm_wmma,gemm_mma,gemm_mma_tf32,kernels_mma,kernels_mma_tf32,kernels_wmma_opt,elementwise,softmax,rmsnorm,transpose,sgd,memory}.rs` の cudarc `Result<_, DriverError>` を返す driver API 呼び出し箇所（`launch`・`clone_dtoh`・`clone_htod`・`synchronize`・`alloc*` 等）を、`begin_driver_call` で得た `CallToken` を伴わせてこの `observe_driver_result` 経由に統一する（#1013 の実装時に機械的に適用できるよう §9 の変更順序へ組み込む）
  - 学習ループ側の `DeviceParamStore` poisoned 遷移（`BackendError::StorePoisoned`、`crates/tensor-core/src/device.rs:219-230`）は「`step()` 自体の実行時エラー」のみを検出対象とする既存契約であり、`DeviceParamStore` を経由しない素の `BackendOps::gemm` 呼び出し等、任意の `download` で初めて表面化しうる context-wide な遅延エラーはこの経路を通らない。上記 1〜7 は `DeviceParamStore` の外側・`context_cache` 層で完結する別契約として設ける（`StorePoisoned` を置き換えず、両者は独立に有効なまま併存する）
- `BackendError` への新規 variant 追加を**本設計で採用する**。`BackendError::DeviceContextPoisoned(String)`（`ordinal` とトリガー元エラーの要約を保持する文字列）を `#[non_exhaustive]` 列挙体への非破壊追加として `crates/tensor-core/src/device.rs` に追加し、`StorePoisoned` と同様「帰属情報を持たないが継続を拒否する」ことを示す。メッセージには `invalidate(ordinal)` 経由での再生成が必要である旨を含める
- 加えて `BackendError::StaleDeviceGeneration { ordinal: usize, resource_generation: u64, current_generation: u64 }`（同じく `#[non_exhaustive]` 非破壊追加）を追加する。これは item 5 の「引数として渡された `DeviceBuffer` 等の世代が現行世代と不一致」を示す専用 variant であり、`DeviceContextPoisoned`（context 全体が poison 状態）とは区別する: `StaleDeviceGeneration` は現行世代自体は正常（非 poisoned）でも成立しうる（例: `invalidate` 後、旧世代のハンドルを渡し続けた場合）。メッセージには当該ハンドルを新世代のキャッシュ（`context_cache::cached_device` 等）から取り直す必要がある旨を含める
- 加えて `BackendError::DeviceContextRetiring { ordinal: usize }`（`#[non_exhaustive]` 非破壊追加）を追加する。これは item 2 の「`invalidate` が drain 中の generation に対する新規呼び出し」を示す一時的なエラーであり、`DeviceContextPoisoned`（恒久的または `invalidate` 待ちの拒否）とは異なり短時間のうちに解消する契約であることをメッセージに明記する
- 加えて `BackendError::DeviceContextUnrecoverable { ordinal: usize, probe_error: String }`（`#[non_exhaustive]` 非破壊追加）を追加する。これは item 4-d の「`invalidate` がストリーム完了同期（b'）または独立検証プローブ（c）のいずれかに失敗し、当該 ordinal を恒久的に poison した」ことを示す専用 variant であり、メッセージに「同一プロセス内での回復手段はなく、プロセス再起動が必要である」旨を明記する。この variant を受け取った以後、当該 ordinal への `invalidate` 呼び出しは常に同じエラーを返す（fail-closed の恒久化）
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
  - **T3a（`context_cache` 単体・poison と回復。§5 item 1・4。二巡目レビューで b'／c の分離が強化されたことに伴い改訂）**: `crate::context_cache` のテスト内で sticky 相当のフェイクエラーにより `phase` が `Poisoned { unrecoverable: false }` へ遷移し `is_poisoned(ordinal)` が真になることを確認する。`invalidate(ordinal)` は、ストリーム完了同期（b'）と独立検証プローブ（c。5 段階の実処理プローブ）の双方をモック（成功／失敗を個別に差し替え可能にする）した状態で 3 系統検証する: (i) b' 成功・c 成功時に `is_poisoned(ordinal)` が偽へ戻り `current_generation(ordinal)` が 1 増え `cached_device` 等が再構築されること、(ii) b' 成功・c 失敗（5 段階のいずれか、または値照合の不一致）時に `generation` は増えず `phase` が `Poisoned { unrecoverable: true }` になり `BackendError::DeviceContextUnrecoverable` が返ること、(iii) b' 自体が失敗した場合は c のモックが一度も呼び出されないまま `generation` は増えず `phase` が `Poisoned { unrecoverable: true }` になり `BackendError::DeviceContextUnrecoverable` が返ること、かつ (ii)・(iii) いずれの場合も以後の `invalidate` 呼び出しが同じエラーを返し続けること。実 CUDA デバイスなしのモック／フェイク実装（`CudaDevice` 生成を要さない範囲）で検証する（CI・GitHub ホステッドで実行可能）
  - **T3g（in-flight drain・§5 item 2〜4。codex-review P0 指摘 1〈TOCTOU〉への回帰防止）**: `crate::context_cache` のテスト内で `begin_driver_call(ordinal, &[current_generation])`（現行世代と一致する generation を渡す）により `CallToken` を取得したまま（in-flight を維持したまま）別スレッドから `invalidate(ordinal)` を呼び、`invalidate` がその `CallToken` が **`Drop` される**（§5 item 3。`observe_driver_result` 呼び出し自体は `in_flight` を変更しないため、テストは `token` をスコープから明示的に外す、または `drop(token)` を呼ぶことでこれを模す）まで完了・復帰しないこと（drain によるブロック）を確認する。加えて、`Retiring` 期間中に新規 `begin_driver_call(ordinal, &[current_generation])` を呼ぶと `BackendError::DeviceContextRetiring` で即座に拒否されることを確認する。実 CUDA デバイスなしのモック／フェイク実装で検証する（CI・GitHub ホステッドで実行可能）
  - **T3j（`CallToken` の RAII 解放・§5 item 1・3。PR #1053 codex-review P1 指摘「CallToken に Drop/RAII がなく begin 後に破棄されると in_flight が減算されず drain が永久停止しうる」への回帰防止）**: `begin_driver_call` で `CallToken` を取得した後、`observe_driver_result` を一度も呼ばずに（`?` によるアーリーリターンや panic を模して）スコープを抜けさせても（`std::panic::catch_unwind` で panic 経路も含めて検証する）、`Drop` により `in_flight` が確実に 1 減ること、および別スレッドで待機中の `invalidate` の drain がそのタイミングで解放されることを、実 CUDA デバイスなしのモック／フェイク実装で確認する
  - **T3k（1 個の `CallToken` が複数 driver 呼び出しを監視する・§5 item 2・3。PR #1053 codex-review P1 指摘「1 個の CallToken では演算内の複数 driver 呼び出し〈clone_htod・alloc・launch・synchronize・clone_dtoh〉全結果を監視できない」への回帰防止）**: `begin_driver_call` で取得した 1 個の `CallToken` を借用（`&token`）で `observe_driver_result` に複数回（sticky・operation-local を混在させて）渡し、各呼び出しがそれぞれ独立に分類・処理される一方 `in_flight` はトークン 1 個分（1）のまま変化しないこと、およびトークンが最終的に 1 回だけ `Drop` されたときにのみ `in_flight` が 1 減ることを、実 CUDA デバイスなしのモック／フェイク実装で確認する
  - **T3l（並行 `invalidate` の所有権一本化・§5 item 4a。PR #1053 codex-review P1 指摘「並行 invalidate が同一世代を重複公開し generation を二重に進めうる」への回帰防止。二巡目レビューで b' が c より前段の独立した fail-closed ゲートになったことに伴い前提を明確化）**: b' のストリーム完了同期が成功するようモックした前提で、複数スレッドから同一 ordinal に対し同時に `invalidate(ordinal)` を呼んだ場合、独立検証プローブ（c）の呼び出し回数がちょうど 1 回であること、`current_generation(ordinal)` がちょうど 1 だけ増えること、かつ全スレッドが同一の結果（成功なら `Ok(())`、失敗なら同一の `BackendError`）を受け取ることを、実 CUDA デバイスなしのモック／フェイク実装で確認する（b' が失敗するケースでの所有権一本化・c 未呼び出しは T3n が検証する）
  - **T3m（`StaleDeviceGeneration` は現行世代を退役させない・§5 item 5。PR #1053 codex-review P1 指摘「StaleDeviceGeneration〈現行世代自体は健全〉に対しても invalidate(ordinal) を呼ぶ指示になっている」への回帰防止。呼び出し元契約の明文化）**: `StaleDeviceGeneration` を受け取った呼び出し元の契約として `invalidate(ordinal)` を呼ばないことをドキュメントテスト／統合テストレベルで明示する: 世代不一致の `resource_generations` を渡して `StaleDeviceGeneration` を得た後、`is_poisoned(ordinal)` が偽のまま・`current_generation(ordinal)` が不変のままであること（＝現行世代は健全なままであり `invalidate` を呼ぶ必要がないこと）を確認する
  - **T3n（sync 失敗は c を経由せず即座に Unrecoverable へ確定する・§5 item 4 b'。PR #1053 二巡目 codex-review P0・Cursor Bugbot High 指摘「sync 失敗を軽量プローブの成功で覆す fail-open」への回帰防止。旧 T3n の期待値を反転する）**: `invalidate` のストリーム完了同期（b'）をモックし常にエラーを返すよう固定した場合、`invalidate` が独立検証プローブ（c）を**一度も呼び出さずに** `BackendError::DeviceContextUnrecoverable { ordinal, probe_error }` に到達し `phase` が `Poisoned { unrecoverable: true }` になり、`generation` が変化しないことを確認する（c 側のモック呼び出し回数がちょうど 0 であることをアサートし、プローブ成功による復帰の再発を直接検知する）。あわせて、sync が成功するケースでは c の実処理プローブ（5 段階）が呼び出されること、c の 5 段階のいずれかまたは値照合が失敗した場合も同じく `Poisoned { unrecoverable: true }`・`BackendError::DeviceContextUnrecoverable` に到達することを、実 CUDA デバイスなしのモック／フェイク実装で確認する
  - **T3h（世代検査と in-flight 登録の単一区間性・§5 item 2。codex-review P0 指摘 1〈TOCTOU〉への直接的な回帰防止）**: `begin_driver_call(ordinal, resource_generations)` に、現行世代と**一致しない** generation を含む `resource_generations` を渡すと、`in_flight` を増やさずに `BackendError::StaleDeviceGeneration` を返すことを確認する。加えて、複数スレッドから同時に `begin_driver_call` を発行しても（一部は一致・一部は不一致の generation を渡す）、`in_flight` カウントには一致した呼び出しの分だけが加算され、不一致の呼び出しがカウントに紛れ込まないことをスレッドセーフティの観点で確認する（旧版の「事前検査 → 別呼び出しでの登録」という 2 段構成であれば再現しえた、検査と登録の間の窓を突くレース条件が、統合後の単一呼び出しでは再現しないことの直接的な回帰テスト）。実 CUDA デバイスなしのモック／フェイク実装で検証する（CI・GitHub ホステッドで実行可能）
  - **T3i（drain 時のストリーム完了同期・§5 item 4 b'。codex-review P0 指摘 2〈drain が非同期 CUDA 作業の完了を待たない〉への回帰防止。実機依存）**: `#[ignore]` テストとして、実 CUDA デバイス上で長時間実行される複数のカーネルを投入した直後（`launch` 復帰済みだが実行完了前）に別スレッドから `invalidate(ordinal)` を呼び、`invalidate` の復帰が「`launch` 呼び出し群の host 復帰」ではなく「投入した全カーネルのデバイス側完了」まで実際にブロックされることを、カーネル内で書き込む完了マーカー（デバイスメモリ上のフラグ）を `invalidate` 復帰直後に読み出して検証する。あわせて、意図的に不正な参照（境界外アクセス）を injects するのではなく、正常なカーネルの完了待ちのみを確認対象とする（境界チェックを緩める形の fault 注入は `.claude/rules/coding-rust.md` 違反のため行わない）
  - **T3c（`static_cuda_memory` の世代追従・§5 item 6）**: `ops.rs` の `static_cuda_memory` キャッシュが `context_cache::current_generation(ordinal)` の変化を検知して新しい `CudaMemory` を構築し直すこと（`bound_generation` 不一致時の差し替え）を、実 CUDA デバイスなしで検証可能な範囲（`current_generation` の値のみを差し替えて分岐を確認する等）でユニットテスト化する。実 `CudaDevice` を要する完全経路の確認は `#[ignore]` の実機依存テストへ委ねる
  - **T3b（呼び出し側の拒否経路・実機依存）**: `#[ignore]` テストとして、実 CUDA デバイス上で `context_cache` を直接 `Poisoned { unrecoverable: false }` 状態へ遷移させてから `ops::CudaBackendOps` の演算メソッド（`gemm` 等）・`sgd_step_device`・`download`／`upload` を呼び出し、いずれも実処理に入らず `BackendError::DeviceContextPoisoned` を即座に返すことを確認する。実際のカーネル fault 注入は行わず（境界チェックを外すことは規約違反）、poison 状態を直接セットする経路のみを検証対象とする（driver 抽象のモック化によるフォールト注入は #1013 実装時に技術的な到達可否を判断し、不可の場合は本 T3b の直接セット方式を正とする）
  - **T3d（世代跨ぎハンドルの拒否・§5 item 2・item 5。codex-review P0 指摘への回帰防止）**: `crate::context_cache` のテスト内で `invalidate(ordinal)` を呼んで世代を進めた後、`invalidate` 前に取得しておいた（旧世代の `generation` を保持する）`DeviceBuffer`／ハンドル相当のフェイク値の `generation` を `resource_generations` として `begin_driver_call(ordinal, resource_generations)` へ渡し、`is_poisoned(ordinal)` が偽（poison 状態ではない）であっても `BackendError::StaleDeviceGeneration` で拒否され、かつ `in_flight` が増えていないことを、実 CUDA デバイスなしのモック／フェイク実装で検証する（CI・GitHub ホステッドで実行可能）
  - **T3e（poison 化と世代の非汚染・§5 item 3・7c。codex-review P1・Bugbot Medium 指摘への回帰防止）**: `context_cache::observe_driver_result` のテスト内で、`begin_driver_call(ordinal, &[current_generation])` で取得した `CallToken` に対する sticky エラーが、`phase` が `Active` の場合は `Poisoned { unrecoverable: false }` へ遷移させ `is_poisoned` が真になることを確認する。**加えて（PR #1053 codex-review P1 指摘「sticky error が Retiring を上書きし drain 通知条件を満たせず invalidate が永久待機しうる」への回帰防止）**: `phase` が既に `Retiring` の状態（別スレッドの `invalidate` が同じ generation を drain 中）で同じ sticky エラーを観測させても `phase` が `Poisoned` へ上書きされず `Retiring` のまま維持されること、かつその後 `in_flight` が 0 になった時点で `invalidate` 側の drain 待機が正しく解除されること（`phase` が上書きされていれば `phase == Retiring && in_flight == 0` を満たせず永久待機する）を確認する。さらに、T3g と同様に in-flight（`CallToken` 未 `Drop`）の状態で `invalidate` を並行実行しても、`invalidate` が drain で待機し続けるため「死につつある旧世代のエラーが `invalidate` 完了後の新世代を汚染する」という不整合が構造的に起こらないことを、実 CUDA デバイスなしのフェイク `DriverError` とモック drain で検証する
  - **T3f（operation-local エラーの非 poison・Bugbot High 指摘への回帰防止）**: `observe_driver_result` のテスト内で、operation-local に分類される `CUresult`（例: `CUDA_ERROR_INVALID_VALUE`・`CUDA_ERROR_OUT_OF_MEMORY`）を渡した場合に `phase` が `Poisoned` へ遷移せず `is_poisoned(ordinal)` が偽のまま保たれ、かつ `Err` がそのまま呼び出し元へ伝播することを、実 CUDA デバイスなしのフェイク `DriverError` で検証する
- **T4（前提確認）**: `CudaContext::has_async_alloc()` が実機で真であることを記録するプローブテスト（`#[ignore]`）。GB10 実機セッションで実測し、本文書の I4 の注記を確定値へ更新する
- CI（GitHub ホステッド・CUDA 非搭載環境）で実行されるのは型検査・モックベースのテストのみであり、実機実行は `make test-ignored-cuda`（または相当の手順）に委ねる

## 9. #1013（実装）への引き渡し事項

変更順序（この順で進めることで、各段階でビルド・既存テストが green のまま維持できる）:

1. `memory.rs` に readback ヘルパ（`clone_dtoh` → `synchronize` の順を一本化する内部関数）を追加する
2. `sgd.rs:174` の `synchronize()` を除去する（最優先。D2H を伴わない唯一の常駐経路であり、除去効果が最も直接的に測れる）
3. ホスト `Tensor` を返すラッパー群（`gemm.rs`・`gemm_wmma.rs`・`gemm_mma.rs`・`gemm_mma_tf32.rs`・`kernels_mma.rs`・`kernels_mma_tf32.rs`・`kernels_wmma_opt.rs`・`elementwise.rs`・`softmax.rs`・`rmsnorm.rs`・`transpose.rs`）の「launch 直後 `synchronize()`」を、readback ヘルパ 1 回への置換に統一する（2.2 の棚卸し表の順で機械的に適用できる）
4. `rmsnorm.rs:1102`／`rmsnorm.rs:1105` のような分岐内二重同期を単一化する
5. `context_cache.rs` に §5 の状態機械を追加する: `ordinal` ごとの `Mutex<OrdinalState>`（`generation: u64`・`phase: Phase`〈`Active`／`Retiring`／`Poisoned { unrecoverable: bool }`〉・`in_flight: u64`。§5 item 1）＋ `Retiring` の出入り（drain 通知・retire 所有権の一本化。§5 item 4a）双方に使う `Condvar`・`CallToken`（`Drop` 実装で `in_flight -= 1` と通知を行う RAII ハンドル。§5 item 1・item 3）・`begin_driver_call(ordinal, resource_generations: &[u64]) -> Result<CallToken, BackendError>`（phase 検査・世代一致検査・in-flight 登録を単一ロック区間でアトミックに行う。演算入口で 1 回だけ呼ぶ。§5 item 2・item 5）・`observe_driver_result(ordinal, token: &CallToken, result)`（`CallToken` を借用しエラー分類・poison 化のみを同一区間で行う〈`in_flight` は変更しない〉。1 個の `CallToken` に対し演算内の driver 呼び出しの数だけ呼んでよい。§5 item 3・item 7）・`invalidate(ordinal) -> Result<(), BackendError>`（retire〈他スレッドが実行中なら所有権を取らず結果を待つ〉→ drain → ストリーム完了同期〈失敗時はここで `Poisoned { unrecoverable: true }` を確定し `DeviceContextUnrecoverable` を返して終了する。c へは進まない〉→ 独立検証プローブ〈alloc → H2D → 恒等カーネル起動 → D2H → 値照合の実処理プローブ。sync 成功時のみ到達する〉→ 公開、の段階。§5 item 4）・`is_poisoned(ordinal) -> bool`（`phase` からの導出。既存呼び出し契約を維持）・`current_generation(ordinal) -> u64`（他キャッシュ・`DeviceBuffer` 世代検査が鮮度検証に使う公開 API。§5 item 5・item 6）を追加する
6. `crates/tensor-core/src/device.rs` の `BackendError` に `DeviceContextPoisoned(String)`・`StaleDeviceGeneration { ordinal, resource_generation, current_generation }`・`DeviceContextRetiring { ordinal }`・`DeviceContextUnrecoverable { ordinal, probe_error }`（いずれも `#[non_exhaustive]` 非破壊追加。§5 で定義した 4 variant 全てをここで追加する。item 5・8 が要求する `DeviceContextRetiring`／`DeviceContextUnrecoverable` の返却経路が本項での追加漏れにより成立しなくなることを防ぐ。PR #1053 codex-review P1 指摘への回答）を追加する
7. `DeviceBuffer`（`crates/tensor-core/src/buffer.rs`）および `CudaGemm`／`CudaElementwise`／`CudaRmsNorm`／`CudaSoftmax`／`CudaSgd`／`CudaMemory` が内部的に保持するハンドルに、生成時点の `context_cache::current_generation(ordinal)` を記録する `generation: u64` フィールドを追加する（§5 item 4・item 5。CUDA 経由で確保されたリソースにのみ有効値を持たせる）
8. `ops::CudaBackendOps` の各演算メソッド・`sgd_step_device`・`MemoryOps::download`／`upload` の入口に、実際の driver 呼び出し直前で `begin_driver_call(ordinal, resource_generations)` を 1 回呼ぶ呼び出しを追加する（引数には呼び出しに関わる各リソースの `generation` をまとめて渡す）。`Poisoned` なら `DeviceContextPoisoned`／`DeviceContextUnrecoverable`、`Retiring` なら `DeviceContextRetiring`、世代不一致なら `StaleDeviceGeneration` を返し、いずれの場合も実処理へ進まない。**世代一致検査を `begin_driver_call` とは別の事前ステップとして実装しない**（同一ロック区間でのアトミックな検査が P0 指摘 1 の解消条件そのものであるため。§5 item 2・item 5）
9. `crates/backend-cuda/src/{gemm,gemm_wmma,gemm_mma,gemm_mma_tf32,kernels_mma,kernels_mma_tf32,kernels_wmma_opt,elementwise,softmax,rmsnorm,transpose,sgd,memory}.rs` の cudarc driver API 呼び出し箇所を、item 8 で取得した `CallToken` を**借用**（`&CallToken`。1 演算内の複数 driver 呼び出しそれぞれに同じトークンを渡す。§5 item 2）で伴わせて `context_cache::observe_driver_result` 経由へ統一する。`CallToken` 自体は演算の最後（正常終了・アーリーリターン・panic のいずれでも）にスコープを抜けて `Drop` され、そこで初めて `in_flight` が解放される（§5 item 3。個々の `observe_driver_result` 呼び出しでは `in_flight` を変更しない）。これにより (i) sticky／operation-local のエラー分類で回復可能なエラーを無用に poison せず、(ii) in-flight ゲート・drain（§5 item 2・4）により死につつある旧世代のエラーが `invalidate` 完了後の新世代を汚染しないことの両方を満たしたうえで、遅延エラー・Drop 時エラーがどの呼び出しで表面化しても sticky エラーは漏れなく poison 化される（§5 item 7。readback ヘルパ・Drop 実装への個別手書き呼び出しはしない）
10. `ops.rs` の `static_cuda_memory` を `(bound_generation: u64, mem: &'static CudaMemory)` キャッシュへ拡張し、取得時に `context_cache::current_generation(ordinal)` と突合して不一致なら新世代の `CudaDevice` 上に新しい `CudaMemory` を構築・差し替える（§5 item 6。`context_cache::invalidate` だけでは波及しない `ordinal` キー付き別系統キャッシュへの世代伝播）

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
- `invalidate`（§5 item 4）の根本的な回復手段の強化: 本設計の `invalidate` は同一 ordinal の primary context を再 retain する前提（c 参照）であり、sticky error が実際に primary context を汚染した場合はプロセス内で解消する手段を持たない（`Poisoned { unrecoverable: true }` へ確定しプロセス再起動を要求する契約に留める）。`cuDevicePrimaryCtxReset` 等による同一プロセス内での primary context の完全破棄・再生成を伴う根本的な回復は本設計のスコープ外とし、採否の判断は #1013（実装）のタイミングへ委ねる

## 12. 出典

- `scripts/bench/framework-compare/results/summary.md`（MLP 学習 1 step 実測値）
- `crates/backend-cuda/src/device.rs`（ストリーム構成・`context_cache` 連携）
- `crates/backend-cuda/src/ops.rs`（`static_cuda_memory` の `Box::leak` プロセス内キャッシュ・`memory_ops` 経由の生成契約。§5 item 6・イシュー #935）
- `crates/backend-cuda/src/{gemm,gemm_wmma,gemm_mma,gemm_mma_tf32,kernels_mma,kernels_mma_tf32,kernels_wmma_opt,elementwise,softmax,rmsnorm,transpose,sgd,memory}.rs`（`synchronize()` 棚卸し）
- `crates/tensor-core/src/backend_ops.rs`（`BackendOps` トレイト定義）
- `crates/tensor-core/src/buffer.rs`（「download の同期契約」節）
- `crates/tensor-core/src/device.rs`（`BackendError` 定義・`StorePoisoned` 等の拡張 variant）
- `crates/bench-harness/src/sync.rs`（`SyncPoint`）
- `crates/backend-metal/src/context.rs`・`crates/backend-metal/src/memory.rs`（Metal 側現状）
- cudarc 0.19.8（`Cargo.toml:95`）の `src/driver/safe/{launch,core}.rs` 相当（`clone_dtoh`・event tracking・`has_async_alloc`・Drop 時エラー蓄積の意味論）
- `docs/spec/04-requirements.md` REQ-2（バックエンド間同期方式・FMA 契約）・REQ-8（性能下限・`bench-harness` の計測境界）
- `docs/device-resident-update-design.md`（#934・同型の設計文書構成・`DeviceParamStore` poisoned 契約）
