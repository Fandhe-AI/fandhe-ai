# CUDA `with_host_view` ホストステージング（イシュー #1336）

## 0. 要約

親イシュー #1334（「デバイス結果をホストへ 1 回だけ移し借用で返す」読み出し
API）の #1335（PR #1404）で `tensor-core::MemoryOps::with_host_view` を
新設し、CPU（ホストバッファ直接借用）・Metal（`synchronize` 後
`contents()` 直接借用）を実装した。CUDA は #1335 のスコープ外として
既定実装（`download` 経由の毎回新規 `Vec<f32>` 確保 + D2H）のまま残されて
いた。

本イシュー（#1336）は CUDA 側を実装する: 形状（要素数）ごとに再利用する
ホストステージングバッファへ `memcpy_dtoh` 1 回で D2H し、そのスライスを
呼び出し元クロージャへ借用として渡す。実装は完了し GPU 非依存の単体
テスト・受け入れ条件を検証する `#[ignore]` 実機テストを整備し、
GB10 実機（DGX Spark GB10・sm_121）で D2H＋読み出し時間の before/after
5 回計測中央値（R3）を実行した（§5・§6）。当初の計測はホスト読み出しを
`black_box` で保護する是正（§5.3。codex-review P1・Cursor Bugbot 指摘）
より前のバイナリで取得したもので参考値に留めていたが、**イシュー #1438
で是正後のバイナリによる再計測を完了し、以下の数値は確定した実測結果
である**（§5.3・「採否」節）。glibc mmap 閾値（32 MiB）を超える N=4096
（64 MiB）で約 7.9 倍改善に加え、閾値未満の N=1024/2048 も約 1.4〜1.6
倍の明確な改善を示した（是正前の参考値は N=1024/2048 が「差なし」寄り
だったが、is-optimized-away の懸念解消後は全 N で改善が確認できた）。
**イシュー #1478（2026-09-09 ユーザー承認）で `HOST_STAGING_KIND` の
既定を `Pinned` へ切り替え済み**（§8。unsafe 経路の既定化について
実装フェーズで機械的自己監査を実施済み〈unsafe 箇所数 1・`SAFETY`
根拠不変・確保上限／fail-closed 契約不変。`security-audit.md`〉。
正式な security-auditor 承認は PR レビューで行う）。

## 1. 背景（実測根拠）

`docs/perf/cuda-large-buffer-percall-alloc-transfer-threshold.md`
（§4.2・§6）が実測した「大容量バッファ per-call アロケーション＋転送」の
フェーズ分解によれば、D2H の宛先を**毎回新規確保**する構成（P4）のみが
31→32 MiB で 24〜33 倍の段差（glibc `M_MMAP_THRESHOLD` 動的上限）を示す
一方、**事前タッチ済み再利用 `Vec`（P5）は線形・段差なし**である。現状の
`MemoryOps::with_host_view` の CUDA 既定実装（`download` 経由）はまさに
P4 相当（`CudaStream::clone_dtoh` が毎回新規 `Vec<f32>` を確保）であり、
本イシューはこれを P5 相当（事前タッチ済み再利用バッファ）へ置き換える。

## 2. 設計

### 2.1 モジュール構成

- `crates/backend-cuda/src/host_staging.rs`（新規・非公開 `mod`）:
  `HostStagingCache`（形状ごとに再利用するホストバッファのキャッシュ。
  take/put 方式）・`HostStaging`（`Pinned`／`Pageable` の 2 種）・
  `HostStagingKind`・`HostStagingStats`（診断用スナップショット）。
- `crates/backend-cuda/src/memory.rs`: `CudaMemory` に
  `host_staging: Arc<Mutex<HostStagingCache>>` フィールドを追加し、
  `MemoryOps::with_host_view` を上書き実装。`take_or_alloc_staging`／
  `return_staging`（`host_staging` へのロック境界を限定するヘルパー）・
  `host_staging_stats`（`internal-diagnostics` feature 限定の診断
  アクセサ）・`release_host_staging`（REQ-14 型の明示解放 API。常時
  公開）を追加。

### 2.2 種別（`HostStagingKind`）

| 種別 | 実装 | unsafe | 特性 |
|------|------|--------|------|
| `Pageable`（切替前既定・#1336〜#1438） | 事前タッチ済み `Vec<f32>` | なし | 確保コストは通常の `Vec` 確保と同じ。ホスト読み出しは通常速度。`new_with_host_staging_kind` 経由で A/B 対照腕として明示選択できる |
| `Pinned`（**既定**。イシュー #1478） | cudarc `CudaContext::alloc_pinned`（`CU_MEMHOSTALLOC_WRITECOMBINED` 固定・page-locked） | **1 箇所**（`HostStaging::alloc` 内の `ctx.alloc_pinned::<f32>(numel)` 呼び出し） | GPU 側の DMA 転送は高速だが、WRITECOMBINED メモリは CPU 側の読み出しが著しく遅いことが知られている（cudarc-0.19.8 `core.rs:1406-1427` のドキュメンテーションコメント）。§5.2 の確定実測では「D2H＋ホスト読み出し」の合計でも全 N で `Pageable` を上回った（§8 で本番既定化） |

`docs/backend-cuda-managed-placement-decision.md` が既存の設計判断で
「pinned host memory は対象外」としていた領域に本イシューで対応した
（`crates/backend-cuda/src/host_staging.rs` モジュール冒頭コメント参照）。

`cudarc-0.19.8` にはキャッシュ可能な pinned（`CU_MEMHOSTALLOC` フラグ
0）を得る手段として `driver::result::malloc_host` があるが、これは
`HostSlice` トレイトの実装・RAII 型の自作（`malloc_host`／`free_host`／
`from_raw_parts` を扱う unsafe 面の拡大）を要するため、本イシューでは
実装せず、実測で `Pinned`（WRITECOMBINED）が有望であった場合の後続候補
としてのみ記録する（ユーザー承認が必要な範囲）。

### 2.3 ロック方針（take/put）

`HostStagingCache` の `Mutex` は「データ構造操作のみ」の間だけ保持し、
`memcpy_dtoh`・`synchronize`・呼び出し元クロージャ `f` の実行中は保持
しない。これにより `f` の内部で別バッファの `with_host_view` を再入
しても deadlock しない（同一 numel の再入は新規確保で対応する）。

新規取得側（`take_or_alloc_staging`）は poison を fail-closed に拒否する
一方、返却側（`return_staging` → `host_staging::put_back`）は poison 後も
`PoisonError::into_inner` で panic せず返却を試みる（以降のアクセスは
`take_or_alloc_staging` 側の poison 検査が引き続き拒否する）。

### 2.4 世代検査

`context_cache::invalidate`（ordinal の poison 回復）はストリームの世代を
進めるが `CudaMemory` 自体は再生成されないため、ステージングエントリは
確保時点の ordinal 世代を刻印し、`take` 時に世代不一致なら破棄して
miss 扱いにする（fail-closed。イシュー #1013 設計文書 §9 item 7 と同じ
方針）。

### 2.5 分岐（`with_host_view` 本体）

1. `handle.storage == None`（空テンソル）: FFI を呼ばず `f(&[])`。
2. `CudaStorage::Managed`: `host_view_managed`（新設ヘルパー）へ委譲し、
   `UnifiedSlice::as_slice()` の借用をコピーなしでそのまま `f` へ渡す
   （`download` が `to_vec()` するのと異なり、managed 配置本来のゼロ
   コピー特性を保つ）。`host_staging` キャッシュは経由しない。
3. `CudaStorage::Device`: `host_staging` から確保・`memcpy_dtoh` で
   D2H・`synchronize` の後、`f` へ借用を渡す。使用後は成功・失敗
   いずれの経路でも `return_staging` でキャッシュへ返却する。

同期契約は `download`（`download_inner`）と同一で、`with_driver_call`
（poison／世代状態機械）を唯一の driver 呼び出し境界とする。

## 3. unsafe（1 箇所・security-auditor レビュー対象）

`crates/backend-cuda/src/host_staging.rs::HostStaging::alloc` の
`HostStagingKind::Pinned` 分岐のみ:

```rust
let pinned = unsafe { ctx.alloc_pinned::<f32>(numel)? };
```

- **理由**: `cudarc::driver::CudaContext::alloc_pinned` が unsafe な唯一の
  理由（cudarc-0.19.8 `core.rs:1410-1411`）は、返す `PinnedHostSlice` の
  内容が確保直後は不定であること。
- **安全性根拠**: 本関数はキャッシュ miss 時にのみ呼ばれ、呼び出し元
  （`memory.rs::CudaMemory::with_host_view`）は返ったバッファを呼び出し元
  クロージャ `f` へ渡す前に必ず `CudaStream::memcpy_dtoh` で要素数ぶん
  全域を D2H 上書きしてから `as_slice()` で読み出す（未初期化内容が外部へ
  露出する経路は存在しない）。要素型は `f32` であり全ビットパターンが
  有効な浮動小数点表現になるため（`cudarc::driver::ValidAsZeroBits` が
  `f32` に実装済み）、未初期化ビット列を無効値として解釈する余地もない。
  `memory.rs::alloc_zeroed_inner` の managed 確保分岐（`alloc_unified` +
  `memset_zeros`）・`pool.rs::CudaAllocator::alloc_uninit` と同一クラスの
  安全性根拠である。
- **既定で通る**（イシュー #1478。旧: `HOST_STAGING_KIND = HostStagingKind::
  Pageable` で当時は通らなかった）: 現在は `HOST_STAGING_KIND =
  HostStagingKind::Pinned` のため、本番経路（`CudaMemory::new`）は
  キャッシュ miss 時にこの unsafe ブロックへ到達する。確保失敗
  （`cuMemHostAlloc`）は `CudaError` として fail-closed に呼び出し元へ
  伝播し、`Pageable` へのサイレントフォールバックはしない（§8）。
  `Pageable` は `new_with_host_staging_kind` 経由で明示選択した場合の
  対照腕としてのみ使う。

## 4. テスト

### 4.1 GPU 非依存単体テスト（`host_staging.rs::tests`・`memory.rs::tests`）

- `HostStagingCache::take`／`put` の hit／miss・世代不一致破棄・長さ
  不一致破棄・cap 超過破棄・`release_all`。
- `HostStaging::Pageable` の直接構築が bit 完全一致で読み出せること。
- Mutex poison からの復旧（`put_back` は panic しない／`take` 直接呼び出し
  経路は poison を観測できる）。
- `CudaMemory::with_host_view` のハンドル型不一致・device ordinal 不一致
  → `DeviceMismatch`（呼び出し元クロージャは呼ばれない）。
- 空バッファ（`numel == 0`）→ FFI を呼ばず空スライスを渡す。
- `CudaMemory: Send + Sync` の静的検査。

いずれも `cargo test -p fandhe-ai-backend-cuda --lib --all-features` で
green（本エージェント実行環境で確認済み）。

### 4.2 `#[ignore]` 実機テスト（`tests/host_view_real_device.rs`）

- `with_host_view` が `download().as_slice()`／`upload` の元データと
  NaN／inf／`MIN_POSITIVE` を含めて bit 完全一致すること。
- 同一形状の複数回呼び出しで `host_staging_stats().hits` が増加すること
  （キャッシュ再利用の確認）。
- `release_host_staging` がキャッシュを空にし、以降の呼び出しが miss から
  再開すること。
- `PooledMemory<CudaMemory>` 経由の `with_host_view` が透過的に動作し
  `download` と bit 同一であること。
- managed 配置（`crate::placement::set_managed_placement_enabled(true)`）
  での `with_host_view` が `download` と bit 同一であること（デバイスが
  managed memory 非対応の場合はスキップ）。
- `with_host_view_using_kind`（`internal-diagnostics` feature 限定の
  診断専用入口）が `Pageable`／`Pinned` いずれも `download` と bit 完全
  一致すること。
- **（実機実測フェーズ追加）** `CudaMemory::new_with_host_staging_kind`
  （同じく `internal-diagnostics` feature 限定・§5 の A/B 計測が使う
  診断専用コンストラクタ）でキャッシュ種別を `Pinned` に固定した場合も、
  本番と同じ `with_host_view`（キャッシュ経由）経路で `download` と
  bit 完全一致し、2 回目以降は `host_staging_stats().hits` が増加する
  こと（`with_host_view_using_kind` は毎回新規確保のため公平な A/B の
  前提にならず、本コンストラクタで補った。§5.1 参照）。

実行コマンド:

```sh
cargo test -p fandhe-ai-backend-cuda --release --all-features \
    --test host_view_real_device -- --ignored --nocapture --test-threads=1
```

## 5. 実機実測（R3・完了）

GB10 実機（DGX Spark GB10・sm_121・CUDA 13.0・rustc 1.97.0）で 2026-09-08
に実施した。実行ログ・env_info・集計スクリプトは
`docs/perf/logs/cuda-host-view-staging-1336/`（README.md に再現手順あり。
内部ホスト名は含めない）。

### 5.1 実行手順（実施済み）

1. `docs/real-hardware-verification-env.md` §3・§4・§6 に従い DGX Spark
   へ rsync 転送した（イシュー専用ディレクトリ
   `~/work/rust-ai-library-run-1336/`）。転送前後で秘密ファイル
   （`*.local.md`・`.env*`）が含まれないことを確認済み。
2. 計測前後に `nvidia-smi`（utilization／compute-apps）・`uptime`・
   `rustc -V`・`nvcc --version` を `env_info.txt` へ記録した（内部
   ホスト名は含めない）。常駐サービス（ComfyUI・Kokoro）以外に GPU を
   使うプロセスがないことを毎回確認してから計測した。
3. `#[ignore]` 実機テスト（4.2 節。新規追加分含め全 7 件）を実行し
   bit 同一を確認した（ゲート A）。
4. D2H＋読み出し時間の A/B（`before` = `download()`＋読み出し／
   `after_pageable` = 本番既定 `with_host_view`／`after_pinned` =
   `new_with_host_staging_kind(Pinned)` 経由のキャッシュ経由 `with_host_
   view`）を N=1024/2048/4096（4/16/64 MiB）で計測した。各 run = 20
   warmup + 20 計測の中央値、独立 5 プロセス起動の中央値を採用
   （`host_view_staging_readout_ab_1336.rs`）。

**事前宣言した判定基準**（実測前に本節へ記載・計測後に変更していない）:

- ゲート A（必須）: `#[ignore]` 全件 pass。
- ゲート B（本番 `Pageable` の非後退）: 全 N で `before` 比 median +5%
  以内、または改善。改善が期待できるのは glibc mmap 閾値（32 MiB）を
  超える N=4096 のみで、N=1024/2048 は差なしでも失敗と読まない。
- ゲート C（情報のみ）: キャッシュ経由 `Pinned` vs `Pageable`。結果に
  関わらず `HOST_STAGING_KIND` は本ラン単独では切り替えない。

### 5.2 実測値（5 プロセス起動・中央値。詳細は `aggregate.md`。**§5.3 の計測保護是正後の再計測値〈2026-09-08・イシュー #1438〉。確定**）

| N | bytes(MiB) | before（`download`＋読み出し・ms） | after `Pageable`（ms） | after `Pinned`（ms） | pageable/before | pinned/before |
|---|---|---|---|---|---|---|
| 1024 | 4 | 0.4252 | 0.2680 | 0.2108 | **0.630x**（約 1.6 倍高速） | 0.496x（約 2.0 倍高速） |
| 2048 | 16 | 1.3366 | 0.9338 | 0.7565 | **0.699x**（約 1.4 倍高速） | 0.566x（約 1.8 倍高速） |
| 4096 | 64 | 26.5788 | 3.3413 | 3.1270 | **0.126x**（約 7.9 倍高速） | 0.118x（約 8.5 倍高速） |

- **ゲート A**: `host_view_real_device` の `#[ignore]` 7/7 件 pass
  （`docs/perf/logs/cuda-host-view-staging-1336/ignored-host_view_real_device.log`）。
- **ゲート B**: 全 N で満たす（3 サイズとも `before` 比 +5% を大幅に下回り改善。
  事前予測では N=1024/2048〈glibc mmap 閾値 32 MiB 未満〉は「差なしでも失敗と
  読まない」としていたが、是正後の再計測では両サイズとも明確な改善
  〈0.630x・0.699x〉を示した——§5.3 のとおり是正前の数値は before 側も含めて
  計測消失の影響を受けていた可能性が高く、是正後の値をもって初めて確定する）。
- **ゲート C（情報のみ）**: キャッシュ経由 `Pinned` は全 N で `Pageable`
  よりさらに速い（N=1024 で約 21%・N=2048 で約 19%・N=4096 で約 6%）。
  WRITECOMBINED メモリの CPU 読み出し劣化（2.2 節の懸念）は本ワーク
  ロード（線形逐次読み出し＋XOR 畳み込み）では顕在化しなかった。ただし
  `HOST_STAGING_KIND` は本ラン単独では切り替えない（6 節）。

env_info: GB10・utilization.gpu 0%（計測前後とも）・load average
0.1〜1.7 台（常駐 ComfyUI・Kokoro のみ、計測対象の GPU 使用プロセス
なし）・rustc 1.97.0。5 run とも他プロセスの介入なしを確認済み
（`docs/perf/logs/cuda-host-view-staging-1336/env_info.txt`）。

### 5.3 計測保護の是正（codex-review P1・Cursor Bugbot 指摘。再計測完了・イシュー #1438）

`host_view_staging_readout_ab_1336.rs::measure_with_cold` の計測区間
（`bench_harness::run` へ渡すクロージャ）が `workload()` の戻り値
（`fold_bits` による XOR 畳み込み値。`u32`）を `let _ = workload();` で
破棄していた。`fold_bits` は副作用を持たない純粋計算のため、
`bench_harness::run` 自体が呼び出しを `black_box` で包む契約（`bench-
harness::protocol::run` のドキュメンテーションコメント）だけでは
クロージャ内部の計算過程（D2H＋要素走査）までは保護されず、release
最適化でホスト読み出し全体が除去されうる状態だった（codex-review・
Cursor Bugbot が同一箇所を独立に指摘。一致度が高い）。

`std::hint::black_box(workload())` へ修正し戻り値を消費した（後続コミットで
反映済み）。**イシュー #1438 で GB10 実機再計測を完了し、5.2 節を
是正後の値で確定した。** 是正前後の比較（旧値は git 履歴の本節・§5.2 を
参照）: 是正前は N=1024/2048 の pageable/before が 0.993x／1.018x と
「差なし」寄りだったのに対し、是正後は 0.630x／0.699x と明確な改善を
示す。これは是正前の `before` 側計測（`download()`＋読み出し）も
コンパイラ最適化により実際の D2H＋走査コストの一部が消失していた
可能性を示唆する（是正後は before／after 双方が保護されるため、両者の
相対比較としての信頼性が上がった）。N=4096 の改善幅も 0.097x→0.126x
（約 10.3 倍→約 7.9 倍）とやや縮小しているが、大幅改善という結論自体は
不変。総じて「is-optimized-away の可能性」という §5.3 の当初の懸念は
解消され、5.2 節の数値は確定した実測結果として扱ってよい。

再計測手順は README.md の「再現手順」節のとおり
（`ab-run1.log`〜`ab-run5.log`・`aggregate.md`・`env_info.txt` は
2026-09-08 実測値で上書き済み）。

## 6. 採否

- **本番既定は `HostStagingKind::Pinned` へ切り替えた**（イシュー
  #1478・2026-09-09 ユーザー承認。切替前は `Pageable`〈unsafe 経路を
  通さない安全側〉を維持していたが、§5.2 の確定値〈イシュー #1438 の
  is-optimized-away 懸念是正後〉で `Pinned` が全 N で `Pageable` を
  一貫して上回ることを確認したうえで、unsafe 経路の既定化についての
  ユーザー承認を得て切替を実施した。実装フェーズの機械的自己監査は
  `security-audit.md`（正式な security-auditor 承認は PR レビューで
  行う）。詳細・GB10 実機再計測は §8）。
- 実装（`HostStagingCache`・`with_host_view` の 3 分岐・GPU 非依存
  テスト・`#[ignore]` 実機テスト）は完了し（ゲート A は bit 同一等の
  受け入れ条件検査のため §5.3 の計測保護是正の影響を受けず確定
  済み）、`Pageable` 種の下で #1146 が示した「事前タッチ済み再利用
  `Vec`」の段差回避効果は §5.2 の確定値で**全 N で**確認できた
  （N=4096・64 MiB で `before` 比約 7.9 倍改善に加え、32 MiB 未満の
  N=1024/2048 も想定に反し約 1.4〜1.6 倍の明確な改善を示した。§5.3）。

## 7. 引き継ぎ（対象外事項。ユーザー承認・別イシュー起票が必要）

1. **facade からの到達経路が現状存在しない**: `autodiff`／`facade` は
   `MemoryOps::with_host_view` を呼んでいない。`Var::matmul` の出力は
   `gemm` 内部の `readback`（`gemm.rs::run_f32_kernel`）で既にホスト
   常駐 `Tensor` になっており、`Var::host_view` は D2H を伴わない
   （`docs/public-api-design.md` §6 項目 10）。よって本イシューの CUDA
   上書きの直接の受益者は `DeviceBuffer<f32>` を保持する経路
   （`autodiff/src/optim/device_store.rs` の `mem.download(..)` 呼び出し・
   resident 系）であり、framework-compare の gemm reuse 計測（イシュー
   #1337 等）では本経路を通らない。
2. **`gemm` 内 `readback` のステージング化**: `gemm.rs::run_f32_kernel` の
   `readback` ヘルパー自体を `host_staging` 相当のキャッシュへ結線する
   ことは自然な後続候補だが、本イシューのスコープ外（自動運転のため
   Issue 起票はしない。将来 issue 化を検討）。**追記（#1436）**:
   `host-view-readout` feature 有効時の CUDA reuse N=1024/2048 後退は、
   `readback` 宛先の事前タッチ未実施が有力な原因仮説である（事前タッチ
   済み宛先を使う `PretouchedReusedDest` 腕が全 N で d2h 最速という点は
   交絡なく確定しているが、「on 腕固有の free 欠如が原因」という機構
   自体は腕単体・プロセス分離計測が未実施のため仮説にとどまる。
   `docs/perf/cuda-host-view-readout-small-shape-regression.md`
   §0・§8・§10 候補 A・§11。PR #1442 codex-review 指摘）。**追記（#1437）**:
   `readback` 自体（`memory.rs`。`gemm.rs` 等 30 箇所超が共有する唯一の
   D2H 同期点）を `ReadbackDest::PretouchedFresh`（反復ごとに事前タッチ
   済み宛先を**新規確保**する方式。`HostStagingCache` の再利用〈候補 A〉
   ではない）へ切り替え、Gate 1（全 N で `on@after / off@base ≤ 1.00`）
   を達成した。`HostStagingCache` 自体は変更していない（引き続き本経路
   〈`readback`〉には到達しない。`docs/perf/cuda-host-view-readout-
   small-shape-regression.md` §13）。
3. **キャッシュ可能 pinned（フラグ 0）**: `driver::result::malloc_host` +
   自作 `HostSlice` 実装は unsafe 面が広がるため本イシューでは実装しない
   （2.2 節）。`Pinned`（WRITECOMBINED）の確定実測（§5.2）は本番既定化
   （イシュー #1478）の根拠となったが、より広い読み出しパターン
   （ストライドアクセス・複数スレッド同時読み出し等）での追加実測は
   未実施のまま引き継ぐ（本番既定化そのものはこれらの追加実測を前提
   条件とせずユーザー承認済み）。
4. **`HOST_STAGING_KIND` の `Pinned` への切替**: §5.2〜§5.3 の確定実測
   （is-optimized-away 懸念是正後）に基づき、イシュー #1478・
   2026-09-09 ユーザー承認により切替を実施済み（§8）。GB10 実機での
   ゲート A／B 再計測を完了し、実装フェーズの機械的自己監査を実施した
   （正式な security-auditor 承認は PR レビューで行う）。

## 8. `Pinned` 既定化の実測（イシュー #1478）

### 8.1 事前宣言ゲート（実装計画段階で事前宣言し、計測後に緩和・変更
していない。本節への転記自体は計測後）

- ゲート A（必須）: `host_view_real_device` の `#[ignore]` 全件 pass。
  本イシューで新規追加した `default_cuda_memory_uses_pinned_staging_
  and_matches_pageable_bit_exact`（`CudaMemory::new` が実際に `Pinned`
  へ解決され、かつ `Pageable` 対照腕・`download()` と bit 完全一致する
  ことを検証）を含む。
- ゲート B（本番切替の非後退）: 全 N（1024/2048/4096）で
  `after_pinned/after_pageable`（5 プロセス起動中央値の比）が
  **≤ 1.05**、かつ `default_kind,Pinned` 行が 5 run すべてに存在する
  こと（`CudaMemory::new` が実際に `Pinned` へ解決されていることの
  自己証明。イシュー #1336 の `after_pageable` 系列は暗黙の
  `CudaMemory::new` に依存していたため、本番既定切替後は
  `new_with_host_staging_kind(Pageable)` で明示構築するよう是正した
  うえで計測している）。
- ゲート C（framework-compare gemm cuda reuse 非後退ガード）: F2（下記）
  の構造的非到達により想定結果は「差なし」。

### 8.2 F1〜F4（実装時に判明した事実）

- **F1**: `host_view_staging_readout_ab_1336.rs` の対照腕（旧称
  `after_pageable`）は暗黙の `CudaMemory::new`（本番既定コンストラクタ）
  で構築していたため、既定を `Pinned` へ切り替えると系列名と実体が
  乖離する。`new_with_host_staging_kind(Pageable)` による明示構築へ
  是正し、加えて `mem_default = CudaMemory::new(&device)` を計測に
  使わず構築して `host_staging_kind()`（`internal-diagnostics` feature
  限定の診断アクセサ。本イシューで新設）を `default_kind,<kind>` として
  出力することで、本番既定コンストラクタが実際にどの種別へ解決されて
  いるかを自己証明する。
- **F2**: `autodiff`／`facade`／`bench-fandhe` のいずれからも
  `MemoryOps::with_host_view` の呼び出しは 0 件（`grep` 確認済み）。
  `Var::matmul` の出力は `gemm` 内部の `readback`（イシュー #1437 で
  `ReadbackDest::PretouchedFresh` へ切替済み。`host_staging` とは別
  経路）で既にホスト常駐化されるため、framework-compare の `gemm cuda
  reuse` 計測は本イシューの変更に到達しない。ゲート C は「本変更が
  本番経路を壊していないこと」の非後退ガードであり、効果測定ではない
  （§7 項目 1 と同じ構造）。
- **F3**: 現時点で `with_host_view` の本番呼び出し元は存在せず、
  `Pinned` 既定化の受益は将来の消費者（`DeviceBuffer<f32>` を保持する
  経路。§7 項目 1・2）に対する先行整備という位置づけである
  （prospective）。
- **F4**: `Pinned` 既定化後は `cuMemHostAlloc`（`alloc_pinned`）が失敗
  した場合、`CudaError` として呼び出し元へ fail-closed に伝播し、
  `Pageable` へのサイレントフォールバックは行わない（意図的な契約。
  `host_staging.rs` モジュール冒頭コメント参照）。

### 8.3 GB10 実機実測（2026-09-09）

実行ログ・env_info・集計スクリプトは
`docs/perf/logs/cuda-host-staging-pinned-default-1478/`（README.md に
再現手順あり。内部ホスト名は含めない）。転送は `git archive` 経由
（本リポジトリの隔離 worktree 上で並走する別イシュー〈#1479〉の
未コミット WIP による汚染を避けるため。作業ツリー rsync は使わなかった）。

**ゲート A**: `host_view_real_device` の `#[ignore]` 8/8 件 pass（新規
追加分含む。`ignored-host_view_real_device.log`）。

**ゲート B**（5 プロセス起動中央値。詳細は `aggregate.md`）:

| N | before_med_ms | pageable_med_ms | pinned_med_ms | pinned/pageable | 判定 |
|---|---|---|---|---|---|
| 1024 | 0.4014 | 0.2606 | 0.2069 | 0.7939 | PASS |
| 2048 | 1.2494 | 0.9170 | 0.7838 | 0.8547 | PASS |
| 4096 | 30.1436 | 3.1983 | 3.0325 | 0.9482 | PASS |

全 N で判定基準（≤1.05）を満たし、いずれも改善方向（約 6〜21%）。
`default_kind` は全 5 run とも `Pinned`。§5.2・§5.3（イシュー #1438
確定値）の傾向を本番既定切替後の系列構成（F1 是正後）でも再現した。

**ゲート C（縮小スコープ）**: 時間制約により正式な `run_gemm_gate_cuda.sh`
（5 回計測中央値・candle 併走）は実施せず、`bench-fandhe --task gemm
--device cuda --size <N> --mode reuse` を before（コミット `e8cd3a2`）／
after（本イシューの test コミットまで反映した HEAD）で各 1 回実行して
比較した（詳細・数値は `docs/perf/logs/cuda-host-staging-pinned-default-
1478/gate-c-sanity.md`）。全 N で `checksum` が完全一致し
（F2 の構造的非到達を実測でも裏付け）、`parity_fail_count` 0（両方）・
timing は誤差範囲内で非後退（比 0.96〜0.99）。5 回計測中央値による
正式なゲート再計測は本イシューのスコープ外として引き継ぐ（§9）。

### 8.4 採否

**ADOPT**（既に §6 で `HOST_STAGING_KIND = HostStagingKind::Pinned` へ
切替済み）。ADOPT の根拠は**ゲート A・B の事前宣言基準達成**である
（両方とも必須ゲート・達成済み）。ゲート C は縮小スコープ（1 回計測・
`bench-fandhe` バイナリ直接比較・candle 併走なし・manifest 検証なし）
であり、事前宣言した 5 回計測中央値の判定基準（§8.1）を満たす形では
実施していない。ゲート C の結果（checksum 完全一致・timing 非後退）は
F2（構造的非到達）を裏付ける**参考値**として扱い、ADOPT の必須根拠とは
しない。正式な 5 回計測中央値ゲート C は §9 へ引き継ぐ。unsafe 経路の
既定化についての実装フェーズ機械的自己監査は `security-audit.md`
（正式な security-auditor 承認は PR レビューで行う）。

## 9. 引き継ぎ（イシュー #1478 スコープ外事項）

- ゲート C の正式な 5 回計測中央値による `run_gemm_gate_cuda.sh` 実行
  （candle 併走・fail-closed manifest 検証込み）は時間制約により本
  イシューでは未実施（§8.3）。F2 の構造的非到達により影響は想定され
  ないが、必要であれば別イシューで実施できる。
- キャッシュ可能 pinned（フラグ 0）・より広い読み出しパターンでの追加
  実測は §7 項目 3 のとおり引き続き対象外。
- `with_host_view` の本番呼び出し元の新設（F3。`gemm` 内 `readback` の
  ステージング化・resident `GradStaging` の重み勾配ホスト読み出し API
  等）は §7 項目 1・2 のとおり対象外（兄弟イシュー #1479 が
  `GradStaging` 読み出し API を別途扱う）。
- **H2D 側（ホスト→デバイス転送）の対称な pinned staging** は本ドキュメント
  が扱う D2H 側（`with_host_view`）とは別イシュー（#1585）で追加した
  （`crate::host_staging::{set_pinned_h2d_enabled, pinned_h2d_enabled}`・
  `H2dStagingCache`。既存 `unsafe` を再利用し新規追加なし・既定 OFF）。
  設計・事前登録判定規則・実測記入欄は `docs/perf/cuda-h2d-pinned-staging.md`
  を参照。
