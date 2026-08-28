# CUDA fresh モード GEMM N=2048 固有の約 166 ms 残存オーバーヘッド診断（#956）

イシュー #956「fresh モードの GEMM N=2048 のみに約 166 ms の再現性ある残存オーバーヘッド」
の診断記録。受け入れ条件（1. 内訳の特定・記録、2. 解消可能ならキャッシュ等で解消）に
対応する。計測コード自体は本イシューで整備済み
（`crates/backend-cuda/src/fresh_overhead_diag_tests.rs`・`crates/facade/tests/
gemm_fresh_overhead_diag.rs`。いずれも `#[ignore]` 実機専用テスト）。

## 状態: 実機未実測（実装セッションからは DGX Spark GB10 へ到達不能）

本ドキュメントは静的経路分析・仮説群の整理・計測方法の確立までを完了し、実測結果（§6）は
「未実測・実機セッションへ引き継ぐ」プレースホルダのまま残す。`docs/perf/
cuda-tape-init-cost-diagnosis.md`（#926）・`docs/perf/cuda-jit-cache-benchmark.md`（#534）
が同じ理由（実機セッションでの実測未実施）で「計測コード＋ドキュメント骨子を先行整備し
実機セッションへ引き継ぐ」構成を確立しており、本ドキュメントも同型の位置づけを取る。

受け入れ条件 2（解消可能なら修正する）は本ドキュメントの時点では **条件付き未達**
（実測待ち）: 主因が実機診断で確定し、かつ候補が§8 の決定規則を満たすことを実測で
確認できた場合のみ修正を適用する方針（§8）。実測なしの推測修正は行わない。

## 1. 背景

イシュー #929（`context_cache` プロセス内キャッシュ。PR #946 反映）以降、フレームワーク
横並びベンチ（`scripts/bench/framework-compare/run_all_cuda.sh`）を DGX Spark GB10
（GB10・driver 580.173.02・CUDA 13.0）で再計測した結果、次の再現性ある事象が観測された。

| 計測 | 中央値 |
|------|--------|
| fresh N=1024 | 約 1.9 ms |
| **fresh N=2048** | **約 182.8 ms**（Q1〜Q3 が 181〜185 ms に収束。毎回発生する固定コスト） |
| fresh N=4096 | 約 140.3 ms（reuse の約 135.3 ms とほぼ一致） |
| reuse N=2048 | 約 16.2 ms（`init_s` 約 443 ms は別記録。#926 の主題） |

checksum は fresh/reuse で一致（`-6016.774008`）しており、正しさの問題ではない。
差分 約 166 ms（182.8 − 16.2）は **N=2048 の fresh モードにのみ**現れ、N=1024・N=4096 では
fresh ≒ reuse である。

## 2. 帰属の明確化

`scripts/bench/framework-compare/bench-fandhe/src/main.rs` の `run_gemm`（fresh）と
`run_gemm_reuse`（reuse）を比較すると、計測区間内で **fresh にのみ**含まれる処理は次の
3 つに限られる（`main.rs:79-134`／`main.rs:134-190`）。

| # | fresh のみの処理 | 経路 | サイズ依存 |
|---|----------------|------|-----------|
| (f1) | `fandhe_ai::tape_for(Device::Cuda(0))` | `crates/facade/src/lib.rs::tape_for` → `resolve_ops` → `CudaDeviceProvider::select`（`device_count()` + `probe` → `context_cache::cached_device`〈#946 でキャッシュ済み〉+ `ctx.total_mem()` + `ctx.attribute(MULTIPROCESSOR_COUNT)`）→ `Tape::new_with_ops` | なし |
| (f2) | `tape.var(&a_data)`・`tape.var(&b_data)` | `crates/autodiff/src/tape.rs::var` → `tensor.clone()`（`Arc<Storage>` 共有。`crates/tensor-core/src/tensor.rs:51-53`）。深いコピーは発生しない | なし（Arc clone） |
| (f3) | イテレーション末尾の `Tape` drop | `nodes: Vec<TapeNode>` の drop。A・B は Arc 参照カウント減のみ。**結果テンソル C（N=2048 で 16 MiB の `Vec<f32>`。`clone_dtoh` が確保）はここで解放される**（`TapeNode.value: OnceCell<Tensor<f32>>`。`crates/autodiff/src/tape.rs:189-194`）。reuse では C ノードが tape 上に蓄積され解放されない | **あり** |

両モード共通の処理は `Var::matmul`（`crates/autodiff/src/var.rs:137-151`）→
`CudaBackendOps::gemm`（`crates/backend-cuda/src/ops.rs:471-495`）→ `CudaGemm::run_tiled_f32`
→ `run_f32_kernel`（`crates/backend-cuda/src/gemm.rs:1653-1720`: `clone_htod`×2 →
`alloc_zeros` → launch → `synchronize` → `clone_dtoh`）と、`checksum_var`（`to_tensor()` +
`contiguous().as_slice().to_vec()` の 16 MiB コピー）である。

### 絞り込みの論理

- fresh N=1024 の合計が約 1.9 ms であるため、**サイズ非依存の fresh 限定コスト (f1)(f2)
  の合計は高々 1.9 ms** に収まる。したがって約 166 ms は「サイズ依存かつ fresh 限定」の
  要因に帰属する
- サイズ依存かつ fresh 限定の事象は **(f3) 出力ホストバッファ C（16 MiB）の毎イテレーション
  解放**とその後続影響（次イテレーションの `clone_dtoh` 宛先／`to_vec` 宛先の確保が
  「直前に解放された同サイズ領域」を再利用する点）のみ
- N=4096（64 MiB）で fresh ≒ reuse になる点は、ホストアロケータ（glibc の動的 mmap
  しきい値は 64-bit で上限 32 MiB。16 MiB は上限未満で解放時にしきい値が引き上がり以降
  brk ヒープ側へ移る一方、64 MiB は常に mmap）や driver のページャブル転送経路の
  サイズ分岐と整合しうる

## 3. 静的経路分析と仮説

| 仮説 | 内容 | 判別方法 |
|------|------|---------|
| H1 | `clone_dtoh` の宛先 `Vec`（`Vec::with_capacity` + `set_len` で未タッチページ。`cudarc-0.19.8/src/driver/safe/core.rs:1630-1641`）への `cuMemcpyDtoHAsync` が、直前に解放・再確保された領域で著しく遅い（統合メモリ GB10 のページャブル転送経路でのページフォールト処理） | 宛先を「事前タッチ済み」「保持して再利用」「pinned」に切り替えて D2H 単体時間を比較 |
| H2 | glibc の動的 mmap しきい値遷移（16 MiB 解放 → brk ヒープ化 → trim/再フォールト）の反復コスト | `MALLOC_MMAP_THRESHOLD_`／`MALLOC_TRIM_THRESHOLD_`／`MALLOC_TOP_PAD_` を実行プロセスの環境変数で固定して再計測。`strace -c -e trace=mmap,munmap,brk,madvise` で fresh/reuse の syscall 回数比較 |
| H3 | 解放（munmap／brk 縮小）自体が driver 側（MMU notifier 経由の GPU ページテーブル無効化等）で高コスト | tape drop を計測区間外へ出した変種と、C の `Tensor` を計測ループ外の `Vec` に保持して解放を抑止する変種の比較 |
| H4 | (f1) `tape_for` の毎回プローブ（`total_mem`／`attribute` 取得）や H2D 側 | N=1024 の合計 1.9 ms で上界が決まるため主因ではない。診断では参考値として同時計測し除外を確定する |

## 4. 実機到達性

計画立案・実装セッションいずれも `docs/real-hardware-verification-env.local.md`
（`.gitignore` 対象・実ホスト名）が存在せず、実機（DGX Spark GB10）へ到達できなかった。
#926/#945・#534 と同じ「診断コード + ドキュメント骨子を先行整備し実機セッションへ引き継ぐ」
構成で本 PR を完了する。

## 5. 計測方法

### 5.1 backend-cuda フェーズ分解（`crates/backend-cuda/src/fresh_overhead_diag_tests.rs`）

`context_cache::cached_device(0)`／`cached_gemm(0, &device)` で本番と同じキャッシュ済み
ハンドルを取得し、`CudaGemm` の公開ヘルパー（`upload_f32` 相当の個別 `clone_htod`・
`alloc_output_f32`・`launch_tiled_f32`）と `stream.clone_dtoh`／`memcpy_dtoh` を組み合わせて、
1 回の GEMM を次のフェーズへ分解計測する（N=1024/2048/4096、warmup 3 + 計測 10 trial の
中央値・Q1/Q3）。

- (a) H2D A
- (b) H2D B
- (c) C（デバイス側）確保
- (d) launch + synchronize
- (e) D2H
- (f) ホスト出力バッファ解放（V0/V1）または事前タッチ+解放（V2）

D2H 宛先の変種（H1 判別）:

- **V0**（`fresh_overhead_diag_v0_fresh_drop_each_trial`）: 毎試行 `clone_dtoh` して結果を
  即 drop（fresh 相当）
- **V1**（`fresh_overhead_diag_v1_keep_alive`）: 毎試行 `clone_dtoh` するが結果を
  `Vec<Vec<f32>>` に保持し試行間で drop しない（reuse 相当。H3 判別）
- **V2**（`fresh_overhead_diag_v2_pre_touched`）: 確保後に全要素へ明示書き込みでページを
  事前タッチしてから `memcpy_dtoh`（未タッチページ由来のフォルトコストを転送区間から除く）

(g) `context_cache::cached_device(0)` 単体時間（`fresh_overhead_diag_g_cached_device_select_reference`。
H4 の除外確定用参考値）も併記する。

実行コマンド（`--test-threads=1` 必須。同一 GPU 上での複数テストスレッド競合を避ける）:

```bash
cargo test -p fandhe-ai-backend-cuda --release --lib -- \
  --ignored --nocapture --test-threads=1 fresh_overhead_diag
```

### 5.2 facade 経路トグル比較（`crates/facade/tests/gemm_fresh_overhead_diag.rs`）

framework-compare の fresh／reuse プロトコルを facade 公開 API で再現し、変種比較を行う
（N=1024/2048/4096、warmup 20 → 計測 20、中央値・Q1/Q3・checksum を記録）。

- **P0 (fresh)**: `bench-fandhe::run_gemm` と同一の呼び出し列
- **P1 (reuse)**: `bench-fandhe::run_gemm_reuse` と同一の呼び出し列（P0 との checksum
  一致を確認する唯一の hard assert。相対誤差 1e-3 未満）
- **P2 (fresh + keep C alive)**: fresh だが C の `Tensor<f32>`（`Arc<Storage>` の安価な
  clone）をループ外 `Vec` に保持し `Tape` drop による解放を抑止する（H3 判別）
- **P3 (fresh + tape drop 計測区間外)**: fresh だが `Tape` drop を `elapsed()` の後に
  行い、本体（matmul + 実体化）と drop のコストを分離する
- **P4 (fresh, checksum を as_slice 直読み)**: fresh だが checksum の `to_vec()`（16 MiB
  の追加ホストコピー）を省き `as_slice()` を直接集計する（`to_vec` の寄与分離）

実行コマンド:

```bash
cargo test -p fandhe-ai --release --test gemm_fresh_overhead_diag -- \
  --ignored --nocapture --test-threads=1
```

### 5.3 環境記録項目

`docs/real-hardware-verification-env.md` §6.1 準拠（GPU・driver・CUDA・rustc・リビジョン・
実施日・GPU 占有状況。実ホスト名は書かない）。

### 5.4 H2 判別（環境変数注入。プロセス外から）

```bash
MALLOC_MMAP_THRESHOLD_=4194304 cargo test -p fandhe-ai-backend-cuda --release --lib -- \
  --ignored --nocapture --test-threads=1 fresh_overhead_diag
strace -f -c -e trace=mmap,munmap,brk,madvise \
  cargo test -p fandhe-ai-backend-cuda --release --lib -- \
  --ignored --nocapture --test-threads=1 fresh_overhead_diag_v0_fresh_drop_each_trial
```

## 6. 実測結果（未実測。実機セッションで転記する）

_(空欄。§5 のコマンドを実機で実行し、フェーズ別中央値・Q1/Q3 の表をここへ転記する)_

## 7. 支配的要因の判定（未確定）

_(未実測のため未確定。実測後、H1〜H4 のうちどれが支配的かをここへ記録する)_

## 8. 対応方針

採否の決定規則: 以下 3 条件をすべて満たす候補のみ採用する。

1. fresh N=2048 の中央値が「reuse 中央値 + 数 ms」へ近づく
2. N=256/512/1024/4096 の fresh・reuse いずれも後退しない（5 回計測中央値・Q1/Q3 の
   ノイズ内）
3. checksum・既存 parity テスト（`tests/cpu_cuda_parity.rs` 等。tolerance 変更なし）が pass

- **F-A（H1 が主因の場合。第一候補・safe Rust のみ）**: `gemm.rs::run_f32_kernel`／
  `run_f16_kernel` と `memory.rs::download_inner` の D2H 宛先を、`clone_dtoh`（未タッチ
  ページ）から「確保後に全ページを確定させた `Vec`」への `memcpy_dtoh` + `synchronize` へ
  変更する共通ヘルパーを `memory.rs` に追加し、両経路から呼ぶ
- **F-B（H1 が主因だが F-A で不十分な場合）**: pinned ステージングバッファ（`malloc_host`/
  `free_host`。FFI 境界の `unsafe`）を `context_cache` と同型の single-flight キャッシュで
  保持する。**本候補は別 PR とする**（変更範囲・監査の分離）
- **H2/H3 が主因の場合**: ライブラリ側でホストアロケータ挙動を変える手段（`mallopt`
  呼び出し・独自アロケータ）は公開ライブラリの責務を超えるため採用しない。`tensor-core`
  側の出力バッファプールは設計変更を伴うためスコープ外として記録し、対応は後続イシュー
  提案に留める

修正を適用した場合は §5 の全計測を再実行し、本節に「修正前／修正後」の 2 系列で記録する。
`scripts/bench/framework-compare/results/` は承認済み一次データのため本 PR では更新しない。

## 9. スコープ外

- train／infer タスクでの同種事象（毎ステップ新規 tape・複数演算）は本イシューの対象外
  （GEMM に限定）
- `tensor-core` の出力バッファプール・ホストアロケータ制御・candle/Burn 側プロトコル
  変更は対象外
- F-B（pinned ステージングバッファ）採用は本 PR に含めない（別 PR）
