# candle 0.11.0 CUDA 経路解析（cuBLAS dispatch 層・D2H readback）

## §1 位置づけ

イシュー #2091（親 #2089「他ライブラリのコード取得・詳細解析」）の**読み取り解析**記録。
`candle-core` 0.11.0 の CUDA 経路（stream・cuBLAS dispatch・出力確保・D2H readback・
同期）を、本リポジトリ `crates/backend-cuda` の非同期実行モデルと対比し、
`docs/perf/cuda-gemm-reuse-phase-breakdown.md` §12.6・§12.8（#1973）で確認された
「N=1024／2048 で自作 `matmul` 区間単体でも candle fresh 全体より遅い」の
原因帰属に使う**差分候補**を列挙する。**実測・A/B・性能に関する断定は行わない**
（Phase 3 の範囲）。他ライブラリのコード・シェーダ・派生物はリポジトリへ持ち込まず、
結論と出典のみを本 doc に置く。

## §2 出典

| 項目 | 内容 |
|------|------|
| crate | `candle-core` |
| 版 | `0.11.0` |
| sha256 | `5ecb245093b0f791b89d3420c3df9c6d49c60ab63ba54db896bf8a3baf486706`（`scripts/bench/framework-compare/Cargo.lock` の checksum と、`static.crates.io` から取得した `.crate` の実測 sha256 が完全一致することを確認済み） |
| 上流 URL | `https://github.com/huggingface/candle/tree/0.11.0/candle-core/src/cuda_backend/` |
| ライセンス | `MIT OR Apache-2.0`（`candle-core-0.11.0/Cargo.toml` の `license` フィールド・同梱 `LICENSE` を確認。`docs/license-matrix.md` 8b 節と同じ許容ライセンス系） |
| cudarc | `0.19.8`（`scripts/bench/framework-compare/Cargo.lock` の `candle-core 0.11.0` 依存エントリで確認。本リポジトリの `cudarc = "=0.19.8"`〈`Cargo.toml:126`〉と**同一バージョン**のため、driver 層の意味論は共通で差分は「使い方」の側にある） |

`candle-core` 0.11.0 の `src/cuda_backend/` は `mod.rs`・`device.rs`・`error.rs`・
`utils.rs`・`cudnn.rs` の 5 ファイル構成（0.10.2 に存在した `ops.rs` という
ファイル名は 0.11.0 には**実在しない**。ファイル名・行番号はすべて 0.11.0 の
実物で採り直した）。

## §3 stream・cuBLAS handle

candle の CUDA デバイス生成には 2 経路があり、ハーネス（`scripts/bench/framework-compare/bench-candle/src/main.rs:37`）が使う `Device::new_cuda(0)` は
`BackendDevice::new`（`cuda_backend/device.rs:418` 付近）を経由する。

```rust
// candle-core-0.11.0/src/cuda_backend/device.rs（BackendDevice::new）
let context = cudarc::driver::CudaContext::new(ordinal).w()?;
let stream = context.per_thread_stream();
Self::from_context_and_stream(context, stream)
```

**計画時点の想定（legacy NULL stream = `ctx.default_stream()`）との相違点**:
実際には `context.per_thread_stream()`（cudarc `CudaContext::per_thread_stream`。
`cudarc-0.19.8/src/driver/safe/core.rs:662`）が使われている。これは CUDA の
「per-thread デフォルトストリーム」（`cu_stream` ハンドルとして `0x2` の
sentinel 値を持つ特殊ストリーム。[NVIDIA doc](https://docs.nvidia.com/cuda/cuda-runtime-api/stream-sync-behavior.html#stream-sync-behavior)）であり、
`ctx.default_stream()`（NULL ポインタ・全スレッドで単一の legacy stream。
本リポジトリの既定 `StreamKind::Legacy`〈`crates/backend-cuda/src/device.rs:162-163`〉
と同じ実体）とも、`ctx.new_stream()`（`CU_STREAM_NON_BLOCKING` の明示的な
非 legacy ストリーム。本リポジトリの `StreamKind::Created`）とも異なる**第三の
選択肢**である。`candle-core` 自体は別途 `new_with_stream`（`device.rs:387`）
で `ctx.new_stream()` を使う経路も持つが、ハーネスの `Device::new_cuda(0)` は
そちらを通らない。

per-thread stream は legacy NULL stream と異なり「他スレッドの legacy 操作と
暗黙に直列化されない」点で `new_stream()` に近い非 legacy 系ストリームだが、
CUDA graph capture 可否・stream priority 等の細部は `new_stream()` と同一では
ない。本リポジトリの `StreamKind::{Legacy, Created}` の二分法（`device.rs:107-118`
の `ResolvedStreamKind`）に第三の分類が必要かどうかは、本 doc では判断せず
**差分候補**として §9 へ引き継ぐ。

cuBLAS handle は `from_context_and_stream`（`device.rs:392-410`）内で
`CudaBlas::new(stream.clone())` により **1 回だけ**生成し `Arc` で共有する。
本リポジトリの `STREAM_KIND_CACHE`（`crates/backend-cuda/src/device.rs:104`
`resolve_stream_kind_for`）は `StreamKind::{Legacy, Created}` で挙動が分かれ、
両者を一括りに「単一の `(ctx, stream)` ペアをキャッシュ」と呼ぶのは不正確
である。`Created` の場合のみ 1 回目に生成した `Arc<CudaContext>`・
`Arc<CudaStream>` の組をキャッシュへ保持し、以後の呼び出しへそのまま共有
する（candle の cuBLAS handle 共有と同型なのはこちらの分岐のみ）。既定の
`Legacy` の場合はストリーム**種別**の決定（この ordinal は Legacy である
という事実）だけをキャッシュし、`ctx`・`stream` 本体は毎回の呼び出しで
`CudaContext::new(ordinal)` → `ctx.default_stream()` により新規に取得し
直す（`ctx.default_stream()` はどの `CudaContext` インスタンスから呼んでも
プロセス内で単一の NULL stream を指すため、`ctx` が呼び出しごとに別
インスタンスでも問題にならない。`crates/backend-cuda/src/device.rs` の
`resolve_stream_kind_for` 内コメント参照）。

## §4 cuBLAS dispatch 層（到達限界）

`Tensor::matmul` → `CudaStorage::matmul`（`cuda_backend/mod.rs:2234`）が
dtype ごとに `gemm_config`（`mod.rs:1379`。転置・leading dimension・行優先→
列優先変換のための引数入れ替えを決定）を呼び、続けて dtype 専用の
`gemm_strided_batched_{bf16,f16,f32}`（`mod.rs:2562,2612,2671` 付近）から
`cudarc::cublas::result::gemm_strided_batched_ex` を呼ぶ 1 段構成である。

f32 の compute type 選択（`mod.rs:2571-2575`。`gemm_strided_batched_f32`）は
`gemm_reduced_precision_f32()` フラグの真偽で `CUBLAS_COMPUTE_32F_FAST_TF32`
（true）／`CUBLAS_COMPUTE_32F`（false・既定）を切り替える 2 分岐であり、
algo は固定で `CUBLAS_GEMM_DEFAULT_TENSOR_OP`（`mod.rs:2608`）。TF32 の
性能・精度比較は #2093 の範囲であり、本 doc では dispatch 境界の事実のみを
記録する。

**到達限界**: 本解析が到達できるのは `gemm_strided_batched_ex` の呼び出し
行までであり、cuBLAS 内部の GEMM 実装（closed source）はスコープ外。

## §5 出力確保

`matmul`（`mod.rs:2263-2265` 付近、f32 分岐）は `unsafe { dev.alloc::<f32>(elem_count)? }`
で出力バッファを確保する。`alloc`（未初期化の async device alloc。`alloc_zeros`
ではない）を毎回呼び、プールやキャッシュは介さない。

## §6 readback・同期

`to_cpu_storage`（`mod.rs:1731`）の f32 分岐:

```rust
// candle-core-0.11.0/src/cuda_backend/mod.rs（CudaStorage::to_cpu_storage）
CudaStorageSlice::F32(slice) => {
    let cpu_storage = slice.stream().clone_dtoh(slice).w()?;
    Ok(CpuStorage::F32(cpu_storage))
}
```

`clone_dtoh`（cudarc `CudaStream::clone_dtoh`。`cudarc-0.19.8/src/driver/safe/core.rs:1630-1638`）は

```rust
// cudarc-0.19.8/src/driver/safe/core.rs（CudaStream::clone_dtoh）
let mut dst = Vec::with_capacity(src.len());
unsafe { dst.set_len(src.len()) };
self.memcpy_dtoh(src, &mut dst)?;
Ok(dst)
```

という実装で、**未タッチのページアブル `Vec<T>`**（`with_capacity` + `set_len`。
ゼロクリアも事前タッチもしない）に対して `memcpy_dtoh_async`（driver API の
非同期メモリコピー）を発行する。`to_cpu_storage`・`clone_dtoh`・`memcpy_dtoh`
のいずれにも**明示的な `synchronize()` 呼び出しはない**。完了保証は、
ページアブルホストメモリへの DtoH コピーが CUDA driver 実装上ステージング
バッファ経由で実質的にホストをブロックする（NVIDIA ドキュメントに明記された
一般的挙動）ことに暗黙に依存している。

**本リポジトリの readback との対比**: `crates/backend-cuda/src/memory.rs:753-785`
の `readback`／`readback_with` は既定で `ReadbackDest::PretouchedFresh`
（`pretouched_host_vec` で宛先を非ゼロで事前に埋めてから `memcpy_dtoh` し、
**明示的に `stream.synchronize()` する**）を使う。`internal-diagnostics`
feature 限定の `ReadbackDest::Fresh`（`memory.rs` 内。事前タッチなし＋
`clone_dtoh`＋明示 `synchronize()`）が candle の readback に構造上最も近いが、
candle は**その `Fresh` よりもさらに一段軽く、明示的な `synchronize()` すら
呼ばない**。

## §7 H2D

`clone_htod`（`device.rs:145-155` 付近）はページアブルなホストソースを
device へアップロードする経路で、本リポジトリの既定経路（pinned staging
は opt-in・既定 OFF）と同型のページアブル transfer である。

## §8 自作との対比表

| 軸 | candle 0.11.0（`Device::new_cuda(0)` 経路） | 本リポジトリ（既定・`internal-diagnostics` 無効） |
|----|----|----|
| stream | `context.per_thread_stream()`（per-thread デフォルトストリーム。legacy でも明示 `new_stream()` でもない第三の種別） | `ctx.default_stream()`（`StreamKind::Legacy`。ordinal ごとに `STREAM_KIND_CACHE` で決定を固定） |
| cuBLAS handle | デバイス生成時に 1 回 `CudaBlas::new` し `Arc` 共有 | （GEMM は cuBLAS を使わず自作 NVRTC JIT カーネル経由。§ dispatch 層参照） |
| dispatch 層 | `matmul` → `gemm_config` → `gemm_strided_batched_{dtype}` → `cudarc::cublas::result::gemm_strided_batched_ex` の 1 段 | `BackendOps::gemm`（`crates/backend-cuda/src/ops.rs:2395`）→ 精度モード分岐 → `CudaGemmAuto`（`gemm_auto.rs`）→ `fandhe_ai_tensor_core::dispatch::select_gemm_kernel`（規則選択）→ `gemm_variant_selection.rs`／`gemm_variant.rs`（実装選択）→ NVRTC JIT + `module_cache.rs` の多段構成 |
| 出力確保 | 毎回 `unsafe { dev.alloc::<f32>(n) }`（未初期化 async alloc・プールなし） | プール経由の `alloc_c`（実測 0.0002〜0.0006 ms。`docs/perf/cuda-gemm-reuse-phase-breakdown.md` §12.8） |
| readback 宛先 | 未タッチのページアブル `Vec<T>`（`with_capacity`＋`set_len`） | 既定 `PretouchedFresh`（`pretouched_host_vec` で事前タッチ済み宛先） |
| 同期 | 明示 `synchronize()` なし（ページアブル DtoH の暗黙ブロッキングに依存） | 明示 `stream.synchronize()`（readback の唯一の同期点。`docs/backend-cuda-async-execution-design.md` §2.3・§2.4・§3・§4） |
| H2D | `clone_htod`（ページアブル） | 既定はページアブル（pinned staging は opt-in・既定 OFF） |

## §9 差分候補（Phase 3 への引き継ぎ・未検証）

いずれも**未検証の仮説**であり、実測・A/B は Phase 3（#1972 配下の後続 sub）の範囲。

1. **readback 宛先の事前タッチの有無**: `docs/perf/cuda-gemm-reuse-phase-breakdown.md`
   §12.8 は `iter_total` の約 46〜56%（N=1024／2048／4096 で 1.235／4.254／17.98 ms）
   を readback 関連の未説明分としている。事前タッチしていない candle の
   `Vec<T>` が、なぜ本リポジトリが問題視する #1436 型のページフォールト
   負担を（少なくとも計測上は）負わないように見えるのかは未検証の問い。
   考えられる仮説（優劣を判断せず並べるのみ）:
   - 反復ごとに `Vec` を確保・解放することで glibc 側の mmap 閾値適応が
     定常化し、既タッチページが再利用される
   - CUDA driver 側のページロッキング・ステージングバッファの効果
   - 別プロセス fresh 計測であることによる計測境界の非対称（`docs/perf/cuda-gemm-reuse-phase-breakdown.md` §12.6 に既記載の留意点と同根）
2. **明示的な `synchronize()` の有無という契約の強さの差**: 本リポジトリは
   pinned staging への将来移行に備えて readback に明示同期を維持する契約
   （`docs/backend-cuda-async-execution-design.md`）だが、candle はページアブル
   DtoH の暗黙ブロッキングのみに依存している。この契約の強さの差が
   固定費として現れているかは未検証
3. **dispatch 層の段数**: candle は cuBLAS 呼び出しまで 1 段、本リポジトリは
   規則選択 → 実装選択 → variant 選択 → NVRTC JIT の多段。ホスト側固定費
   （`launch_issue` 等）になりうる候補として記録するが、
   `docs/perf/cuda-gemm-reuse-phase-breakdown.md` §12.8 では `launch_issue`
   自体は 0.005〜0.008 ms と小さく「候補にしない」区分に分類済みのため、
   優先度は低い
4. **出力確保**: candle の未初期化 async alloc（プールなし）と本リポジトリの
   プール経由 `alloc_c` の差。§12.8 では `alloc_c` は `iter_total` の 0.03%
   未満のため「候補にしない」区分だが、candle 側の未初期化 alloc の
   実測コストは本 doc の範囲外で測っていない
5. **#1692 との関係の明確化**: #1692 は「本リポジトリの mse 経路が readback
   1 箇所の同期契約に既に準拠している」ことを確認した記録にとどまる。
   本 doc の「未説明分」は `docs/perf/cuda-gemm-reuse-phase-breakdown.md`
   §12.5／§12.8 に属する事実であり、**#1692 の記録へ誤って帰属させない**

## §10 スコープ外

- cuBLAS 内部の GEMM 実装（closed source。§4 の到達限界）
- CUDA Graph・TF32 opt-in の性能・精度比較（#2093）
- 性能実測・A/B（Phase 3.8〜3.10。本 doc は読み取り解析のみ）
- candle の CPU／Metal バックエンド経路（本イシューは CUDA 限定）

## §11 関連ドキュメント

- `docs/perf/cuda-gemm-reuse-phase-breakdown.md` §12.5〜§12.8（本 doc の差分候補の実測根拠）
- `docs/backend-cuda-async-execution-design.md` §2.3・§2.4・§3・§4（本リポジトリの同期契約）
- `docs/license-matrix.md` 8b 節（framework-compare ハーネスのライセンス方針）
- `.claude/rules/deps-policy.md`（framework-compare ハーネスの依存区分・`candle-core =0.11.0` ピン）
