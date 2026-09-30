# GB10 unified memory・Grace CPU の最適化考察（イシュー #2123）

## §0 位置づけ

- 親 #2121（Phase 4「チップ情報に基づく最適化の考察」）。兄弟: #2122（sm_121 の命令・デバイス属性プローブ）・#2124（M4 Max 考察）。
- 依拠する記録: `docs/backend-cuda-managed-placement-decision.md`（#1352）・`docs/perf/cuda-managed-placement-ab.md`（#1353）・`docs/perf/cuda-gemm-readback-reuse-2108.md`（#2108）・`docs/backend-cpu-gb10-affinity-design.md`（#1576）・`docs/perf/logs/cpu-gb10-affinity-ab-2117/`（#2117）・`docs/cpu-gemm-sme-unsafe-audit.md`（SME `asm!` の先例）。
- **分業**: 本考察の実行ホストは x86_64 で GB10 に到達できない。本 PR は「コード読解で確定できる結論」「プローブと計測基盤」「実測前に固定した判定規則（`docs/perf/logs/gb10-unified-memory-grace-2123/RULE.txt`）」「GB10 セッションへの申し送り」までとし、未計測の値は推定で埋めない。
- 本 PR は本番経路（`src/`）・依存・既定値・tolerance を変更しない。

## §1 前提の訂正

Issue は「#1353 の REJECT は host-registered 方式の性能不足が原因」としているが、原因は異なる。

- host-registered（`cuMemHostRegister`）は cudarc 0.19.8 に wrapper がなく、そもそも実装していない（`backend-cuda-managed-placement-decision.md`「採用しなかった方式」）。
- REJECT の実因は `cuMemAllocManaged`（`UnifiedSlice`）配置での後退で、`UnifiedSlice::drop` の同期 `cuMemFree` が per-step の暗黙同期点になる。train reuse 1.71 倍・`device_update` 単独 2.82 倍の後退（`cuda-managed-placement-ab.md` §4・§5・§8）。
- Issue の「phase 3 再実測」は資料上の節を特定できないため、本書では「train reuse セルの HEAD 上での再実測」と「§6 の帯域再計測」と解釈する。

## §2 D2H 省略の可否（受入 1）

managed opt-in 時の `memcpy_dtoh` の扱い（`crates/backend-cuda/src/memory.rs`）。

| 経路 | 省略可否 | 根拠 |
|---|---|---|
| `MemoryOps::download`（`DeviceBuffer` 常駐。`DeviceParamStore` 系） | **実装済み（#1352）**。D2H DMA はなく `stream.synchronize()` 後に `UnifiedSlice::as_slice().to_vec()`。`Vec` への memcpy は 1 回残る | `host_readback`・`download_inner` |
| `with_host_view`（常駐バッファの借用読み出し） | 真のゼロコピー | `CudaStorage::Managed` 分岐 |
| host 返却型の各演算（`readback()`／`readback_with()`。batch_norm・bce・elementwise・gemm_mma・norm_backward 等） | **省略不可**。`Tensor<f32>` が `Arc<Storage{Vec}>` を持ち宛先が必ず pageable な `Vec` になる | `memory.rs` の readback 群・#2108 doc §2(d) |
| fresh `CudaBackendOps::gemm` | 省略不可（`CudaStorage` を経由しない） | managed-placement-decision「スコープ外」 |
| `gemm_resident_lhs` の NT 転置分岐 | 省略不可（device-only のまま） | 同上 |

結論: 「条件付き省略」は **`DeviceBuffer` 常駐経路でのみ成立し、既に実装済み**。host 返却型の演算へ広げるには tensor-core のストレージ抽象の変更と新規 `unsafe` が必要で、ユーザー承認事項（#2108 §2(d) と同判断）。pinned staging による宛先再利用は #2108 が別途扱う。

## §3 prefetch／advise 契約

cudarc 0.19.8 `unified_memory.rs` の事実:

- `UnifiedSlice::prefetch` は safe API。ATTACH_GLOBAL／SINGLE では device 宛先で、`CONCURRENT_MANAGED_ACCESS=0` なら `CUDA_ERROR_NOT_PERMITTED`。ATTACH_HOST は `HOST_NUMA_CURRENT` 宛先。
- `result::mem_advise` は `unsafe` のみ（safe wrapper なし）。
- `CUDA_ERROR_NOT_PERMITTED` は `classify_cuda_result` の operation-local 一覧に無く sticky poison 扱いになる。導入する場合は #1352 と同様に driver 呼び出し前に属性で事前拒否する。

契約案（実装は本イシューのスコープ外）: upload 直後に device へ prefetch・`host_readback` 前に host へ prefetch（単一ストリーム順序に乗せる）／`CONCURRENT_MANAGED_ACCESS=0` は事前拒否／`mem_advise` は unsafe のため不採用。

GB10 の属性値による推奨の分岐は `RULE.txt` の R-UM-prefetch に事前登録した。値は `unified_memory_probe`（`crates/backend-cuda/examples/`）で取得する。`PAGEABLE_MEMORY_ACCESS_USES_HOST_PAGE_TABLES=1` の場合の「pageable な `Vec` を GPU が直接読む」方式は別候補だが、ストレージ抽象と `unsafe` を伴うため承認事項。

## §4 training loop への影響（受入 4）

- 見立て: `UnifiedSlice::drop` の同期解放が残る限り REJECT は再現する見込み。既定化の再評価の前提条件は managed 対応 `SizeClassPool`（同期解放の除去）で、別イシュー候補。
- 再実測は GB10 で `run_ab_managed_cuda.sh`（v0.9.0 ピン + `AB_PATCH_FACADE_PATH`）を HEAD で 5 回中央値実行し、`RULE.txt` の R-UM-train で判定する。帯域（`managed_placement_bandwidth_real_device`）は R-UM-bw で記録のみ（既定化判定に使わない。#1353 §8 と同位置づけ）。
- 結果: **未計測**。

## §5 Grace CPU（受入 2・3）

- 大コア親和性は #1576／#2117 で設計・A/B 基盤済み。本書では参照のみとする。
- **SVE2 検出（受入 2）**: `/proc/cpuinfo` 経路は既存 GB10 ログで確認済み（`docs/perf/logs/cpu-gemm-b-laneq-vec-ab-1318/lscpu-dgx.txt` のフラグ行に `sve sve2`、`docs/perf/logs/cpu-gemm-sme-fmopa-1587/gb10/sme_report.txt` に `cpuinfo_tokens: sve sve2`）。`std::arch::is_aarch64_feature_detected!("sve2")` は stable でコンパイルでき（`--target aarch64-unknown-linux-gnu` で確認）、std の Linux 実装は getauxval（AT_HWCAP／AT_HWCAP2）を読むため、新規 `unsafe`・依存なしで getauxval 系経路を使える。`libc` は deps-policy 第 10 区分で onnx-interop 用途限定のため直接 FFI は行わない。補助に `/proc/self/auxv` の safe 読み取り（AT_HWCAP=16 bit22・AT_HWCAP2=26 bit1）と `/proc/sys/abi/sve_default_vector_length`。3 経路の実機一致は `grace_sve2_probe_report`（`crates/backend-cpu/tests/`）で GB10 にて確定する（R-SVE2-detect）。**現時点の結論: 検出手段は確定、実機一致は未計測**。
- **SVE2 GEMM マイクロカーネル（受入 3）**: stable Rust に SVE intrinsics はなく `asm!` のみ（C コンパイラ・`cc` は許容依存外）。SME `fmopa` の先例（v0–v31／p0–p15／ffr の clobber 全列挙・`unsafe` 監査）があり、非 streaming SVE2 は ZA／SMSTART を伴わないため ABI 上の論点は SME より少ない。よって「ABI 互換性の範囲内で作成可能」と判定する。ただし価値は VL と帯域で決まる。GB10 の CPU GEMM は DRAM 帯域律速の根拠があり（`docs/cpu-gemm-prefetch-decision.md` 2026-09-08 追補）、比較対象すべてに優位（`docs/perf/cpu-gemm-sme-fmopa-microkernel.md` §1）。実行時 VL が 128 bit（16 byte）なら幅の利得がなく Non-Goal（R-SVE2-kernel）。`sve_default_vector_length` は新規プロセスの既定値で実行スレッドの VL と一致する保証がないため判定には使わず、実行時 VL は未取得（取得は別イシュー・unsafe 承認事項）のため判定不能とする。

## §6 受入基準ごとの結論

| 受入 | 結論 | 状態 |
|---|---|---|
| 1 D2H 省略の検証 | 常駐経路は実装済み・host 返却型は承認事項（§2） | 確定（コード読解）。GB10 裏取りは未 |
| 2 SVE2 検出 | std のみの 3 経路で検出可能な設計（§5） | cpuinfo は確定・3 経路一致は GB10 待ち（規則固定済み） |
| 3 SVE2 マイクロカーネル | `asm!` で ABI 上は可能。VL 16 byte なら Non-Goal | 実行時 VL 未取得のため判定不能（規則固定済み） |
| 4 unified memory 既定化の影響 | REJECT 維持の見込み。前提は managed 対応 `SizeClassPool` | 再実測は GB10 待ち（規則固定済み） |
| 5 本番結線かスコープ外化 | **本番結線しない（本イシューでは不採用）**。GB10 実測で規則を満たした項目のみ別イシューで再評価 | 確定 |

## §7 GB10 申し送り

実行手順と記入欄は `docs/perf/logs/gb10-unified-memory-grace-2123/README.md`。未計測欄は推定で埋めない。

## §8 スコープ外と起票候補（ユーザー承認前のため未起票）

- managed 対応 `SizeClassPool`（同期解放の除去）
- pageable 直接アクセス（tensor-core ストレージ抽象の変更と新規 `unsafe`）
- SVE2 カーネル（VL が 256 bit 以上の機体が現れた場合のみ・`asm!` の個別承認前提）
- NUMA スケジューリング（Issue 側でスコープ外）
