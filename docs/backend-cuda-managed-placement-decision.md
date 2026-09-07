# CUDA managed 配置（DeviceBuffer opt-in）の設計判断（#1352）

イシュー #1352「managed／host-registered 割当を `DeviceBuffer` の opt-in 配置
として実装し既存経路との出力 bit 同一と poison/invalidate 契約の維持を確認
する」に対応する。親 #1351（GB10 物理統合メモリ向けゼロコピー割当の試作・
実測）。兄弟 #1353（GB10 5 回中央値の前後比較・既定化可否判断）。

前提承認: #1338 で「GB10 managed／host-registered メモリ既定化」は方向性として
採用済み（既定化の可否は #1353 の実測結果で最終判断する）。本イシューは
**既定を変えない opt-in 実装**までがスコープ。

## 背景・目的

DGX Spark GB10 はホストと GPU が物理統合メモリのため、`cuMemAlloc` +
`cuMemcpyHtoD`/`cuMemcpyDtoH` の往復（H2D/D2H）が本来不要になりうる。
framework-compare の fresh モード・学習ループ（`DeviceParamStore` の
per-step `upload(grad)`／`download`）で転送固定費が残っている
（`docs/perf/cuda-gemm-reuse-phase-breakdown.md`・`train-step-phase-
breakdown.md`）。本イシューの成果物は cudarc 0.19.8 の既存 API 範囲で
managed 割当（`cuMemAllocManaged`）を `DeviceBuffer` の配置オプションとして
追加し、(a) 既定 OFF、(b) 既存経路と出力 bit 同一、(c) 同期契約・
poison／invalidate 契約（`docs/backend-cuda-async-execution-design.md`
§4〜§5）を維持すること。性能実測・既定化判断は #1353 が担う。

## 採用しなかった方式

- **host-registered メモリ（`cuMemHostRegister`）**: cudarc 0.19.8 に safe
  wrapper が存在しない（`unified_memory.rs` は `alloc_unified`〈managed〉
  のみを提供し、`cuMemHostRegister` 系 API は driver `result` 層にも
  wrapper がない）。追加の FFI 直呼びは依存境界を広げるため不採用。
  タイトルの「host-registered」は方式候補として不採用と確定する。
- **`cuMemAdvise`（`cudaMemAdviseSetAccessedBy` 相当）**: cudarc 0.19.8 では
  `result::mem_advise`（`unsafe`）のみで safe wrapper がない。`unsafe`
  ブロックを増やすため不採用。
- **`UnifiedSlice::prefetch`（`cuMemPrefetchAsync`）**: safe API として存在
  するが、単一ストリーム・`CU_MEM_ATTACH_GLOBAL` 構成下での効果が実測
  されておらず、本イシューでは未使用。#1353 の実測後に導入候補として
  再検討する。

## API 形状（確定）

- `backend-cuda::placement` モジュール（`crates/backend-cuda/src/
  placement.rs`）にプロセスワイドの opt-in フラグ（`static AtomicBool`、
  既定 `false`）と setter/getter（`set_managed_placement_enabled` /
  `managed_placement_enabled`）を追加した。`crate::precision`（TF32
  opt-in）と同型の構成。
- `facade` から自由関数として再公開する（`set_cuda_tf32_gemm_enabled` と
  同型の composition root 直委譲）。
  - 採用名: `fandhe_ai::set_cuda_managed_memory_enabled(enabled: bool)` /
    `fandhe_ai::cuda_managed_memory_enabled() -> bool`。
- **既定は OFF（device-only 配置）**。フラグ OFF 時の経路・出力は本イシュー
  導入前と bit-exact に不変（`memory.rs::CudaMemory::alloc_zeroed_inner`／
  `upload_inner` の managed 分岐は `if placement::managed_placement_
  enabled() { .. } else { 従来どおり }` の `else` 節で従来コードをそのまま
  保持）。

## 実装（確定）

- `crates/backend-cuda/src/memory.rs`:
  - `CudaBufferHandle.slice: Option<CudaSlice<f32>>` を `storage:
    Option<CudaStorage>` へ改名し、`CudaStorage { Device(CudaSlice<f32>),
    Managed(UnifiedSlice<f32>) }` を導入した（配置の実体）。
  - 配置非依存の起動引数抽象 `CudaArg`（読み取り専用: `Slice`／`View`／
    `Unified`／`UnifiedView`）・`CudaArgMut`（書き込み可能:
    `SliceMut`／`UnifiedMut`。可変部分ビュー variant は現状の呼び出し元が
    必要としないため持たない）を追加し、`.push(&mut LaunchArgs)` が
    cudarc 側の対応する `PushKernelArg` 実装へ委譲する。カーネル本体・
    起動 config は配置に依らず完全に共有するため、出力は配置に依らず
    bit 同一となる（本イシューの核心契約）。
  - `alloc_zeroed_inner`／`upload_inner` は opt-in 時 `CudaContext::
    alloc_unified::<f32>(numel, true)`（`attach_global = true`。単一
    ストリーム構成のため `CU_MEM_ATTACH_GLOBAL` で足りる）で確保し、
    `memset_zeros`（alloc）または `as_mut_slice().copy_from_slice`
    （upload。新規確保直後のためホスト memcpy のみで `cuMemcpyHtoD` は
    発行しない）で埋める。
  - `download_inner` の managed 分岐は専用の `host_readback`（`stream.
    synchronize()` → `UnifiedSlice::as_slice()`）を使う。`readback`
    （`cuMemcpyDtoHAsync` 経由）を流用すると managed 配置の目的
    （ゼロコピー）を損なうため別関数とした。
- `crates/backend-cuda/src/device.rs`: `CudaDevice::new` で
  `CU_DEVICE_ATTRIBUTE_MANAGED_MEMORY`／`CU_DEVICE_ATTRIBUTE_
  CONCURRENT_MANAGED_ACCESS` を 1 回だけ照会し `managed_memory_supported:
  bool` としてキャッシュする（`cached_device` 経由で ordinal ごとに
  再利用されるため per-op コストなし）。
- `crates/backend-cuda/src/error.rs`: `CudaError::ManagedMemoryUnsupported`
  を追加。**意図的に非 `Driver` variant**とする理由: `cudarc::driver::
  safe::unified_memory::CudaContext::alloc_unified` 自身も
  `MANAGED_MEMORY` 属性を検査し非対応なら `DriverError(
  CUDA_ERROR_NOT_PERMITTED)` を返すが、このエラーコードは
  `context_cache::classify_cuda_result` の operation-local 一覧に
  含まれないため sticky（ordinal を poison する）扱いになってしまう。
  `memory.rs` は driver 呼び出しの**前**に `CudaMemory::
  check_managed_placement_supported`（`device.managed_memory_supported()`
  を検査）で fail-closed に拒否し、この誤 poison を構造的に防ぐ。
- `crates/backend-cuda/src/ops.rs`: `sgd_step_device`／`gemm_resident_rhs`／
  `linear_forward_device`／`gemm_resident_lhs`（NT 転置分岐を除く）の
  4 箇所を `CudaArg`／`CudaArgMut` 経由へ更新した。
- `crates/backend-cuda/src/gemm.rs`／`sgd.rs`: `launch_tiled_bias_act_f32_
  resident`／`launch_tiled_f32_resident`／`CudaSgd::run` の可視性を `pub`
  から `pub(crate)` へ引き下げたうえでシグネチャを `CudaArg`／`CudaArgMut`
  へ変更した（`gemm`／`sgd` モジュール自体は private・`CudaGemm` は
  `pub use` で再公開されるが、CLAUDE.md「`facade` が唯一のサポートされる
  公開 API 面であり `backend-*` は内部クレート」の定義により
  `backend-cuda` の公開 API 非破壊契約の対象外と判断した）。

## スコープ外（対象化しなかった経路）

- **fresh モードの素の `CudaBackendOps::gemm`**（`run_tiled_f32` 系。
  `CudaMemory` を経由せず `clone_htod`／`alloc_zeros`／`clone_dtoh` を
  直接呼ぶ）: `CudaStorage` を経由しないため managed 化していない。
  fresh モードは opt-in 時も従来どおり device-only で動作する。
- **`gemm_resident_lhs` の NT 転置分岐**（`b` が dense 転置と判定される
  場合。`ops.rs` の `dense_transposed_view` 判定）: この分岐は
  `device.stream().clone_htod`／`alloc_zeros` を `CudaMemory` を経由せず
  直接呼ぶ独立経路のため、`launch_tiled_f32_resident_nt` の `a_dev`
  （`w` の部分ビュー）のみ `CudaArg` 化し、`bt_dev`／`c_dev` は
  device-only（`CudaSlice`）のまま据え置いた。opt-in 時もこの分岐は
  常に device-only で動作する。
- `upload_into`（CUDA 未実装）・managed 対応 `SizeClassPool`（プール経由の
  managed 再利用）・pinned host memory・multi-stream 化は対象外。

## unsafe（1 箇所・security-auditor レビュー対象）

`memory.rs` の `unsafe { self.stream.context().alloc_unified::<f32>(..) }`
（`alloc_zeroed_inner`／`upload_inner` の 2 呼び出し）が本イシューで追加する
唯一の `unsafe` ブロック。cudarc の `alloc_unified` が `unsafe fn` である
理由は「T が任意ビットパターンで有効か cudarc 側で保証しない」ことのみ
（cudarc-0.19.8 `unified_memory.rs:88-93`）。`f32` は全ビットパターンが
有効な浮動小数点表現（NaN／inf を含む）のため無効ビットパターンは存在
しない。加えて確保直後の内容は `memset_zeros`（alloc）または全域
`copy_from_slice`（upload）で露出前に確定させる。`pool.rs::
CudaAllocator::alloc_uninit` の `unsafe { stream.alloc }` と同一クラスの
安全性根拠（FFI 境界・f32 に無効ビットパターンなし・露出前に全上書き）。

## 同期契約の差分（`docs/backend-cuda-async-execution-design.md` への追補）

- **download（managed）**: `host_readback` は `stream.synchronize()` →
  `UnifiedSlice::as_slice()` の順で行う。`UnifiedSlice::as_slice` 自身が
  行う内部 `event.synchronize()` は、単一ストリーム構成
  （`LaunchArgs::launch` が `is_managing_stream_synchronization()`
  で判定する multi-stream モードでのみ event を記録する。cudarc-0.19.8
  `launch.rs:100-135`）では何も記録されないため、呼び出し元による明示
  `stream.synchronize()` が唯一の同期点である。
- **drop（managed）**: `CudaSlice::drop` は該当ストリーム上に
  `cuMemFreeAsync`（デバイス側完了待ちのみの非同期解放）を発行するが、
  `UnifiedSlice::drop`（cudarc-0.19.8 `unified_memory.rs:46-53`）は
  `event.synchronize()` の後に**同期** `cuMemFree` を呼ぶ。学習ループの
  per-step `upload(grad)` → drop がステップごとの暗黙同期になりうる
  **性能リスク**であり、#1353 の実測で顕在化した場合は managed 対応
  プールアロケータ（`SizeClassPool` の managed 拡張）を別イシュー案として
  検討する（本イシューでは実装しない）。

## poison／invalidate 契約の維持

managed 経路の全 driver 呼び出し（`alloc_unified`・`memset_zeros`・
`synchronize`）は既存の `CudaMemory::with_driver_call`（`context_cache::
begin_driver_call` → `observe_cuda_result`）のクロージャ内で行う。
`check_managed_placement_supported` による事前拒否（非対応デバイス）も
同じクロージャ内・`alloc_unified` 呼び出しより**前**で行うため、実際の
driver 呼び出しには到達しない。`observe_cuda_result` は `CudaError::
Driver(_)` のみを分類・poison 化対象とし、`ManagedMemoryUnsupported`
（非 `Driver` variant）は無変更で通過する（`context_cache::
observe_cuda_result` の `match` の `other => other` 節）ため、この
事前拒否自体が誤って ordinal を poison することもない。分類表
（`context_cache::classify_cuda_result`）・許容誤差は変更していない。

## #1353 向け計測テンプレート

- 比較軸: fresh／reuse／train（`bench-fandhe --task {gemm,train}`）×
  `set_cuda_managed_memory_enabled({true,false})`。
- 各条件 5 回計測の中央値（`.claude/rules/coding-rust.md`）。
- `env_info` に `uptime`（load average）・GPU 名・driver 版を記録する
  （内部ホスト名は含めない）。
- `bench-fandhe` は crates.io 公開版 `fandhe-ai =0.7.0` に完全固定されて
  おり（`scripts/bench/framework-compare/bench-fandhe/src/main.rs`・
  deps-policy 第 9 区分）、本イシューの変更（未公開）は次回リリース
  （0.8.0）公開後でないと framework-compare 経由では計測できない
  （`docs/cuda-tf32-optin-api-decision.md`「framework-compare の制約」と
  同型の制約）。#1353 は crate 内 example／実機ベンチハーネス（`bench-
  harness`）による直接計測、または 0.8.0 公開後の framework-compare 計測
  のいずれかを選ぶ。

## 実機実測

本エージェント実行環境に CUDA 実機（DGX Spark GB10）が存在しないため、
`crates/backend-cuda/tests/managed_placement_real_device.rs`（`#[ignore]`）
は実装済みだが未実行のまま引き継ぐ。実行手順:

```sh
export PATH=$HOME/.cargo/bin:/usr/local/cuda/bin:$PATH
uptime
cargo test -p fandhe-ai-backend-cuda --release --all-features \
    --test managed_placement_real_device -- --ignored --nocapture
cargo test -p fandhe-ai-backend-cuda --release --all-features \
    --test memory_real_device --test async_ordering_real_device \
    --test gemm_resident_real_device --test linear_forward_device_real_device \
    -- --ignored --nocapture   # 既存契約テストの非後退確認
```

合格条件: 新規テスト全 pass（`to_bits()` 完全一致）・既存 `#[ignore]`
テスト非後退・リークなし。実機実行結果は本ドキュメントへ追記する
（未実測欄）。

<!-- 実測記入欄（#1353 または実機アクセス可能なセッションが追記） -->

## フォローアップ・出典

- 性能比較（5 回中央値）・既定化可否判断は #1353。
- fresh モード managed 化・NT 転置分岐 managed 化・managed 対応プール・
  pinned host memory は必要になった時点で別イシューとして起票する
  （`.claude/rules/out-of-scope-tracking.md`）。
- 設計判断の出典: `docs/backend-cuda-async-execution-design.md`
  （非同期実行モデル・poison/invalidate 状態機械）・`docs/cuda-tf32-
  optin-api-decision.md`（同型の opt-in API 設計の先例）・
  `docs/device-memory-pool-design.md`（プールアロケータ設計。managed
  拡張の将来検討先）。
