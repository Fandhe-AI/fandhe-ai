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
5 回計測中央値（R3）を実行した（§5・§6）。**この計測はホスト読み出しを
`black_box` で保護する是正（§5.3。codex-review P1・Cursor Bugbot 指摘）
より前のバイナリで取得したものであり、release 最適化で計測対象が消え
ていた可能性を排除できないため、以下の数値は是正後の再計測が済むまで
参考値として扱う**（§5.3・「採否」節）。是正前の参考値では glibc mmap
閾値（32 MiB）を超える N=4096（64 MiB）で約 10.3 倍改善・閾値未満の
N=1024/2048 は想定どおり差なし相当だった。既定の `HOST_STAGING_KIND` は
参考値の傾向に関わらず、unsafe 経路（`Pinned`）を通さない安全側
（`Pageable`）のまま維持する（§6。`Pinned` への切替は再計測後にユーザー
承認事項として引き継ぐ）。

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
| `Pageable`（既定） | 事前タッチ済み `Vec<f32>` | なし | 確保コストは通常の `Vec` 確保と同じ。ホスト読み出しは通常速度 |
| `Pinned` | cudarc `CudaContext::alloc_pinned`（`CU_MEMHOSTALLOC_WRITECOMBINED` 固定・page-locked） | **1 箇所**（`HostStaging::alloc` 内の `ctx.alloc_pinned::<f32>(numel)` 呼び出し） | GPU 側の DMA 転送は高速だが、WRITECOMBINED メモリは CPU 側の読み出しが著しく遅いことが知られている（cudarc-0.19.8 `core.rs:1406-1427` のドキュメンテーションコメント）。D2H 単体では有利でも「D2H＋ホスト読み出し」の合計で `Pageable` に劣る可能性があり、決め打ちしない |

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
- **既定では通らない**: `HOST_STAGING_KIND = HostStagingKind::Pageable`
  （既定）のため、本番経路は通常この unsafe ブロックへ到達しない。
  `#[ignore]` 実機テスト（`tests/host_view_real_device.rs`）でのみ
  `Pinned` 種の確保が実行される。

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

### 5.2 実測値（5 プロセス起動・中央値。詳細は `aggregate.md`。**§5.3 の計測保護是正前の参考値・再計測待ち**）

| N | bytes(MiB) | before（`download`＋読み出し・ms） | after `Pageable`（ms） | after `Pinned`（ms） | pageable/before | pinned/before |
|---|---|---|---|---|---|---|
| 1024 | 4 | 0.2616 | 0.2598 | 0.2076 | 0.993x | 0.794x |
| 2048 | 16 | 0.9016 | 0.9180 | 0.7780 | 1.018x | 0.863x |
| 4096 | 64 | 33.5578 | 3.2441 | 3.0391 | **0.097x**（約 10.3 倍高速） | 0.091x（約 11.0 倍高速） |

- **ゲート A**: `host_view_real_device` の `#[ignore]` 7/7 件 pass
  （`docs/perf/logs/cuda-host-view-staging-1336/ignored-host_view_real_device.log`）。
- **ゲート B**: 全 N で満たす（N=1024: 0.993x・N=2048: 1.018x はいずれも
  +5% 以内、N=4096 は大幅改善）。
- **ゲート C（情報のみ）**: キャッシュ経由 `Pinned` は全 N で `Pageable`
  よりさらに速い（N=1024 で約 20%・N=2048 で約 15%・N=4096 で約 6%）。
  WRITECOMBINED メモリの CPU 読み出し劣化（2.2 節の懸念）は本ワーク
  ロード（線形逐次読み出し＋XOR 畳み込み）では顕在化しなかった。ただし
  `HOST_STAGING_KIND` は本ラン単独では切り替えない（6 節）。

env_info: GB10・utilization.gpu 0%（計測前後とも）・load average
1.0〜1.5 台（常駐 ComfyUI・Kokoro のみ、計測対象の GPU 使用プロセス
なし）・rustc 1.97.0・nvcc 13.0.88。5 run とも他プロセスの介入なしを
確認済み（`docs/perf/logs/cuda-host-view-staging-1336/env_info.txt`）。

**注意（再掲。§5.3 参照）**: 上記ゲート A〜C・表の数値は `workload()` の
戻り値を `black_box` で保護する是正より前のバイナリで取得した参考値で
あり、is-optimized-away の可能性を排除できていない。ゲート B／C の
「非後退」「上回る」という判定文言は是正後の再計測が済むまで確定した
結論として扱わない。

### 5.3 計測保護の是正（codex-review P1・Cursor Bugbot 指摘。未再実測）

`host_view_staging_readout_ab_1336.rs::measure_with_cold` の計測区間
（`bench_harness::run` へ渡すクロージャ）が `workload()` の戻り値
（`fold_bits` による XOR 畳み込み値。`u32`）を `let _ = workload();` で
破棄していた。`fold_bits` は副作用を持たない純粋計算のため、
`bench_harness::run` 自体が呼び出しを `black_box` で包む契約（`bench-
harness::protocol::run` のドキュメンテーションコメント）だけでは
クロージャ内部の計算過程（D2H＋要素走査）までは保護されず、release
最適化でホスト読み出し全体が除去されうる状態だった（codex-review・
Cursor Bugbot が同一箇所を独立に指摘。一致度が高い）。

`std::hint::black_box(workload())` へ修正し戻り値を消費するよう是正
した（本コミット）。**この是正は 5.2 節の実測値を得た計測より後に
行っており、本エージェント実行環境に CUDA 実機（GB10）が無いため
是正後の再計測は未実施のまま記入欄を残す**。5.2 節の数値は是正前の
バイナリでの実測であり、除去が実際に発生していたかは不明（是正後の
再計測で before/after 比が大きく変わらなければ除去は起きていなかった
と判断できる。別セッション・別イシューで GB10 実機に接続できる
エージェントが再実行し本節を更新すること）。

再計測手順は README.md の「再現手順」節をそのまま使える
（`ab-run1.log`〜`ab-run5.log` を上書きし `aggregate.py` を再実行、
5.2 節の表と本節の記述を更新する）。

## 6. 採否

- **本番既定は `HostStagingKind::Pageable`（unsafe 経路を通さない安全側）
  のまま維持する**（実測完了後も変更しない）。5 節の参考値では `Pinned`
  が全 N で `Pageable` を上回る傾向が見られたが（ゲート C）、これは
  §5.3 の計測保護是正前の値であり確定した結論ではない。unsafe 経路の
  既定化は「性能で押し切らず必要最小限に留める」方針
  （`.claude/rules/security.md`）に基づきユーザー承認事項として残す。
  **`Pinned` への切替の判断材料は、§5.3 の是正後の再計測が完了するまで
  揃っていない**（is-optimized-away の可能性を排除できていない値を
  根拠に別イシューへ進めない）。再計測完了後、ユーザーが承認すれば
  `HOST_STAGING_KIND` の切替は本節を更新のうえ別イシューで実施できる。
- 実装（`HostStagingCache`・`with_host_view` の 3 分岐・GPU 非依存
  テスト・`#[ignore]` 実機テスト）は完了し（ゲート A は bit 同一等の
  受け入れ条件検査のため §5.3 の計測保護是正の影響を受けず確定
  済み）、既定 `Pageable` 種の下で #1146 が示した「事前タッチ済み
  再利用 `Vec`」の段差回避効果は**是正前の参考値では**確認できた
  （N=4096・64 MiB で `before` 比約 10.3 倍改善。32 MiB 未満の
  N=1024/2048 は想定どおり差なし相当）。是正後の再計測でこの傾向が
  維持されるかは未確認のまま引き継ぐ（§5.3）。

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
   Issue 起票はしない。将来 issue 化を検討）。
3. **キャッシュ可能 pinned（フラグ 0）**: `driver::result::malloc_host` +
   自作 `HostSlice` 実装は unsafe 面が広がるため本イシューでは実装しない
   （2.2 節）。5 節の**是正前の参考値**では `Pinned`（WRITECOMBINED）が
   全 N で `Pageable` を上回る傾向だったが、§5.3 の計測保護是正後の
   再計測が未実施のため確定していない。再計測に加え、より広い読み出し
   パターン（ストライドアクセス・複数スレッド同時読み出し等）での追加
   実測を経てユーザー承認を得れば検討候補になる。
4. **`HOST_STAGING_KIND` の `Pinned` への切替可否**: 5 節（ゲート C）の
   **是正前の参考値**では `Pinned` が全 N で `Pageable` を一貫して
   上回る傾向だった（N=1024 約 20%・N=2048 約 15%・N=4096 約 6% 高速）。
   ただし §5.3 の計測保護是正（`black_box` 適用）後の再計測が未実施の
   ため、この傾向自体が最適化除去の産物でないかは未確認。再計測で傾向
   が維持されることを確認し、かつ unsafe 経路の既定化についてユーザー
   承認が得られれば（6 節）、`HOST_STAGING_KIND` の切替は別イシューで
   実施できる。
