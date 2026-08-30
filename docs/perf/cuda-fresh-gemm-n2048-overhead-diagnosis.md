# CUDA fresh モード GEMM N=2048 固有の約 166 ms 残存オーバーヘッド診断（#956）

イシュー #956「fresh モードの GEMM N=2048 のみに約 166 ms の再現性ある残存オーバーヘッド」
の診断記録。受け入れ条件（1. 内訳の特定・記録、2. 解消可能ならキャッシュ等で解消）に
対応する。計測コード自体は本イシューで整備済み
（`crates/backend-cuda/src/fresh_overhead_diag_tests.rs`・`crates/facade/tests/
gemm_fresh_overhead_diag.rs`。いずれも `#[ignore]` 実機専用テスト）。

## 状態: 実機実測完了（2026-08-31・DGX Spark GB10・イシュー #1025）— 非再現を確認

**結論を先に書く**: イシュー #1025 着手時点の HEAD（コミット `d6bd4ff`）で §5.1〜§5.3 の
受け入れ条件判定に必要な計測を実機実行した結果（§5.4 の H2 判別用環境変数注入・`strace`
は §6.4・§9 のとおり未実施）、#956 が特定した「fresh N=2048 のみ約 166〜184 ms の残存オーバーヘッド」は
**再現しなかった**。fresh N=2048 は reuse N=2048 とほぼ同水準（後述 §6.3 の facade P0/P1 5 回計測は
1 回が fresh < reuse、残り 4 回は fresh が reuse をわずかに上回るが、いずれの回も
**fresh ≤ reuse × 1.10** の範囲内）であり、イシュー #1025 の受け入れ条件
「fresh N=2048 を reuse の +10% 以内へ」は **コード変更なしに既に満たされている**。

これを受け、本 PR は「主因を特定してコードを修正する」のではなく「実機実測で非再現を確認し、
記録として確定する」形で受け入れ条件を満たす（§7・§8 に判断根拠を記す）。#926/#945・#534 が
確立した「診断コード＋ドキュメント骨子を先行整備し実機セッションへ引き継ぐ」構成を、本
ドキュメントが実際に実機実測で完結させた最初の記録になる。

`docs/perf/cuda-tape-init-cost-diagnosis.md`（#926）・`docs/perf/cuda-jit-cache-benchmark.md`
（#534）は引き続き実機未実測のプレースホルダのままであり、本ドキュメントの完了はそれらの
状態に影響しない。

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

イシュー #1025 の実装セッションでは `docs/real-hardware-verification-env.local.md`
（`.gitignore` 対象・実ホスト名）が存在し、DGX Spark GB10 実機（`CUDA_NODE` 変数。
`docs/real-hardware-verification-env.md` §2.4）へ到達できた。§5 の全計測を実機で実行し
§6 に記録する。#926/#945・#534 で確立した「診断コード + ドキュメント骨子を先行整備し
実機セッションへ引き継ぐ」構成が、本イシューで実際に引き継がれ完結した形になる。

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

## 6. 実測結果（2026-08-31・DGX Spark GB10・イシュー #1025）

### 6.1 環境

| 項目 | 値 |
|---|---|
| GPU | NVIDIA GB10（DGX Spark） |
| driver | 580.173.02 |
| CUDA | 13.0（`nvcc` V13.0.88） |
| rustc | 1.97.0（2026-07-07） |
| 検証対象コミット | `d6bd4ff`（イシュー #1025 実装セッション着手時点の `main` HEAD） |
| GPU 占有状況 | `nvidia-smi --query-gpu=utilization.gpu` 0%（実行前後で確認） |
| 実施日 | 2026-08-31 |

### 6.2 backend-cuda フェーズ分解（V0〜V3・(g)。1 回目計測）

```bash
cargo test -p fandhe-ai-backend-cuda --release --lib -- \
  --ignored --nocapture --test-threads=1 fresh_overhead_diag
```

(g) `cached_device(0)`（warm）: median 0.000 ms（H4 除外を再確認）

| variant | N | (a) H2D A | (b) H2D B | (c) C確保 | (d) launch+sync | (e) D2H | (f) 解放/事前タッチ | 合計 |
|---|---|---|---|---|---|---|---|---|
| V0-fresh | 1024 | 0.077 | 0.075 | 0.003 | 0.002 | 0.398 | 0.001 | **0.554** |
| V0-fresh | 2048 | 0.282 | 0.291 | 0.003 | 0.002 | 2.614 | 0.000 | **3.254** |
| V0-fresh | 4096 | 1.145 | 1.137 | 0.003 | 0.002 | 539.075 ※ | 9.900 | **551.218** ※ |
| V1-keep_alive | 1024 | 0.082 | 0.078 | 0.003 | 0.002 | 32.350 ※ | 0.000 | **32.516** ※ |
| V1-keep_alive | 2048 | 0.294 | 0.290 | 0.003 | 0.002 | 129.131 ※ | 0.000 | **129.720** ※ |
| V1-keep_alive | 4096 | 1.143 | 1.138 | 0.003 | 0.002 | 539.503 ※ | 0.000 | **541.790** ※ |
| V2-pre_touched | 1024 | 0.076 | 0.076 | 0.003 | 0.002 | 0.075 | 0.612 | **0.844** |
| V2-pre_touched | 2048 | 0.296 | 0.290 | 0.006 | 0.004 | 0.299 | 7.895 | **8.809** |
| V2-pre_touched | 4096 | 1.151 | 1.140 | 0.010 | 0.005 | 1.154 | 46.097 | **49.566** |

※ 印を付けたセルは定常値ではない疑いがある（プロセス内ウォームアップ汚染。直後の注記参照）。

（単位はいずれも ms・中央値。Q1/Q3 は生ログ `/tmp/trip1_backend_cuda.log`〈本セッション限り・
コミットしない〉参照。V3 の値は §10 へ集約する）

**注意（V0/V1 の (e) D2H は定常値ではない。プロセス内ウォームアップ汚染の疑い）**: 上表の
V0-fresh N=4096（(e) 539.075 ms）・V1-keep_alive 全サイズ（(e) 32.350／129.131／539.503 ms）
は、単体で見ると物理的に説明がつかない（GB10 の D2H で 4 MiB を 32 ms・16 MiB を 129 ms 要する
理由がない。同一データを同じ手順で転送する V2-pre_touched は同サイズでそれぞれ 0.075／0.299／
1.154 ms — 400 倍以上速い）。5 テストはアルファベット順に `g → V0 → V1 → V2 → V3` の順で
同一プロセス内を連続実行されており、**プロセス内で最初に大きい `Vec` を確保・解放する
テスト（V0 の N=4096・V1 の全サイズ）が、その一回限りのコスト（`clone_dtoh` の未タッチ
ページに対する初回フォールト・アロケータの初回大口径拡張等）を計測区間へ巻き込んでいる
可能性が高い**。最後に実行される V2（既にウォームアップ済みの状態で計測）が全サイズで
安定して速い点も、この説明と整合する。したがって V0/V1 の (e) 列は**定常状態のフェーズ
コストとして扱わない**（V0/V1 単体の値から「166 ms 相当」ないし「逆方向の関係」を主張
しない。本ドキュメントの結論は §6.3・§7 のとおり P0/P1 の facade 計測に基づく）。

### 6.3 facade P0〜P4 トグル比較（本イシューの受け入れ条件判定の主根拠。5 回計測。checksum 一致を確認済み）

```bash
cargo test -p fandhe-ai --release --test gemm_fresh_overhead_diag -- \
  --ignored --nocapture --test-threads=1
```

`coding-rust.md`「ベンチは 5 回計測の中央値を採用」に従い、本テスト（1 プロセス起動あたり
N=1024/2048/4096 それぞれ warmup 20 → 計測 20 の中央値を 1 サンプルとして出す）を **5 回
プロセスごと再起動して実行**した。N=2048（本イシューの対象サイズ）の P0/P1 5 サンプル:

| 回 | P0 fresh（median） | P0 Q1 | P0 Q3 | P1 reuse（median） | P0/P1 比（median） |
|---|---|---|---|---|---|
| 1 回目 | 5.907 ms | 5.902 ms | 6.039 ms | 11.954 ms | 49% |
| 2 回目 | 13.157 ms | 13.102 ms | 13.211 ms | 12.973 ms | 101% |
| 3 回目 | 13.234 ms | 13.157 ms | **138.827 ms** | 13.040 ms | 102% |
| 4 回目 | 12.493 ms | 11.847 ms | **140.083 ms** | 11.758 ms | 106% |
| 5 回目 | 13.214 ms | 13.043 ms | 13.496 ms | 12.844 ms | 103% |
| **中央値** | **13.157 ms** | — | — | **12.973 ms** | **101%** |

5 回すべてで **中央値ベースで fresh ≤ reuse × 1.10** を満たす（1 回目は fresh がむしろ reuse
よりかなり速い外れ値だが、受け入れ条件「fresh が reuse +10% 以内」への違反ではない）。
5 サンプルの中央値同士の比も 101% であり、#956/#1025 が報告した「fresh が reuse の約 11 倍
（184 ms 対 16 ms。#956 実測は中央値 184.0 ms・Q1 178.9 ms・Q3 186.8 ms — 20 反復の
ほぼ全てが遅かった）」という関係は明確に再現しない。**本イシューの受け入れ条件の判定は
この P0/P1 中央値比を主根拠とする**（P0/P1 は framework-compare `bench-fandhe::run_gemm`／
`run_gemm_reuse` と同一の呼び出し列。§5.2）。

**残存テール（正直な記録）**: 5 回中 2 回（3・4 回目）で P0 の Q3 が約 139〜140 ms に達している
（強調セル）。1 プロセスあたり 20 反復中、中央値・Q1 は他の回と同水準のまま Q3 のみ跳ね上がる
形であり、「20 反復のほぼ全てが遅い」という #956 当時の分布（中央値 184.0・Q1 178.9・Q3 186.8）
とは異なる。ただし通常の四分位定義では Q3 は「20 反復中、値の大きい方から上位 25%（約 5
反復）がその水準以上」であることを意味するため、この上位 25% を「まれ」と呼ぶのは正確では
ない。生の反復別データ（`/tmp/trip1_facade_gemm.log` 等・本セッション限りでコミットしない）
を保存していないため、実際に旧事象相当のコストを踏んだ反復数が上位 25%（約 5 反復）ちょうど
なのか、それより少ないのかは本記録からは確定できない。中央値ベースの受け入れ条件
（fresh ≤ reuse+10%）は 5 回とも満たすため「解消」と判定するが、「完全消滅」ではなく
「5 プロセス起動中 2 回で、各回内の上位 25%（Q3 が示す約 5 反復以上）が旧事象相当のコストを
踏んでいる」という表現がより正確である。原因の特定（§6.2 で挙げたウォームアップ汚染機構が
反対に一部反復でのみ再発している可能性等）は §9 のスコープ外として記録する。

参考までに全サイズの 2 回目計測値を記す（P2〜P4 の役割は §5.2 参照。この行は代表値であり
5 回計測ではない）。

| N | P0 fresh | P1 reuse | P2 keep-alive | P3 body（drop 区間外） | P3 drop単独 | P4 as_slice直読み |
|---|---|---|---|---|---|---|
| 1024 | 1.271 ms | 2.515 ms | 2.447 ms | 1.262 ms | 0.000 ms | 0.554 ms |
| 2048 | 13.157 ms | 12.973 ms | 13.089 ms | 6.008 ms | 0.001 ms | 3.142 ms |
| 4096 | 78.837 ms | 76.645 ms | 79.087 ms | 77.153 ms | 9.259 ms | 38.798〜540.390 ms（後述 §6.4） |

### 6.4 補記: N=4096 P4（checksum as_slice 直読み）のばらつき

N=4096 の P4（`to_vec()` を経ずに `as_slice()` を直接集計するチェックサム変種）のみ、2 回の
計測間で 540.390 ms → 38.798 ms と大きくばらついた（P0/P1/P2/P3 は 2 回ともノイズ内で
安定）。V0/V1 の (e) D2H が N=4096 で 539 ms 前後になる回があった点（§6.2）とも整合する
挙動で、N=4096 大サイズ・特定アクセスパターン固有の別事象の可能性がある。**本イシュー
（#1025）は N=2048 に限定**であり、N=4096 のこの挙動は原因未特定のまま §9 のスコープ外へ
追加する（H2 環境変数判別・strace 判別は、本イシューの主題である N=2048 が非再現と確定した
時点で実施を見送った。実施すれば N=4096 側の材料にはなるが #1025 の受け入れ条件には
直接関与しない）。

## 7. 支配的要因の判定

**判定: 実機実測時点（2026-08-31・HEAD `d6bd4ff`）で N=2048 固有オーバーヘッドは再現せず、
H1〜H4 のいずれが「今も」支配的かを判別する対象事象自体が存在しない。**

**判定の主根拠は §6.3 の facade P0/P1（5 回計測）**である。#956 が 2026-08-29（crates.io
0.4.0 相当）に観測した「fresh N=2048 のみ約 166〜184 ms」（reuse の約 11 倍）は、本イシュー
着手時点の HEAD では発生していないことを、production 経路と同一の呼び出し列（P0/P1）で
5 回のプロセス再起動計測により確認した（中央値比 101%。5 回すべて fresh ≤ reuse×1.10）。
§6.2 の backend-cuda フェーズ分解は (c) C 確保・(g) `cached_device` が小さいままであること
（H4 除外の再確認）の裏付けとして使うに留め、V0/V1 の (e) D2H 列はプロセス内ウォームアップ
汚染の疑いがあるため定常値としては採用しない（§6.2 注記参照）。

**帰属先の推定（判断根拠であり再修正の対象ではない）**: #956 実測（2026-08-29）から本実測
（2026-08-31）までの間に着地した以下の PR が、(f3) 帰属分析が指していたホスト側確保・解放・
tape 経路を変更している。いずれも該当日付が #956 の計測日より後であり、時系列上の候補として
記録する（個々の PR を切り分けて再検証する追加実験は行っていない。§9 参照）。

| PR | 内容 | 変更範囲との関係 |
|---|---|---|
| #1061（`perf(backend-cuda): CUDA プールアロケータの実装とテスト`） | (c) C 確保をサイズクラス別プール経由へ | #956 の帰属分析が (c) を直接の主因から除外していたが、プール導入がアロケータ全体の呼び出しパターン・ホスト側フォールバック経路にも波及した可能性 |
| #1077（`perf(autodiff): matmul VJP の転置をゼロコピー化`） | `eval::matmul` の stride 対応 | fresh/reuse 双方の matmul 経路の中間確保が変化 |
| #1079（`perf(autodiff): Linear の epilogue 融合を学習経路へ結線`） | epilogue 融合 | GEMM 単体経路には直接関与しないが tape 構造に影響 |
| #1080（`perf(autodiff): view 系ノードを再計算方式に`） | reshape/transpose の中間バッファ非確保 | tape 上のノード数・確保パターンが変化 |
| #1081（`perf(autodiff): 学習ループで tape を再利用可能に`） | tape ノードクリア API（`Tape::reset` 等）の追加 | tape 構造そのものに新しいコードパスを追加した点で時系列上の候補に挙げるが、下記のとおり fresh 経路への直接の関与は確認できていない |

**#1081 の帰属推定は撤回する**: 診断対象の P0 fresh（`scripts/bench/framework-compare/
bench-fandhe/src/main.rs` の `run_gemm`）は反復ごとに `make_tape` で新規 `Tape` を生成し
測定後に drop するのみで、`Tape::reset`（ノードクリア API）を一切呼ばない
（同ファイルに `reset` の呼び出しは存在しない）。したがって #1081 が (f3)（`Tape` drop に
よるホスト側 `Vec<f32>` 解放）の前提を変えたという推定はコード経路から裏付けられず、
本ドキュメントの帰属推定からは除外する。#956 の非再現の実際の原因は、上表の他の PR
（#1061・#1077・#1079・#1080）を含め特定できておらず、個々の PR の寄与を分離する追加実験
（各 PR 直前へのビルド戻し等）は、本イシューの受け入れ条件（fresh N=2048 が reuse +10%
以内）が既に満たされているため費用対効果が低いと判断し実施しなかった（§9 スコープ外）。

## 8. 対応方針

**コード修正は適用しない。** §8 冒頭で定めていた採否の決定規則（1. fresh N=2048 が reuse +
数 ms 以内へ近づく、2. 他サイズが後退しない、3. checksum・parity が pass）のうち条件 1 は
**実機実測時点で既に満たされている**（§6.3。fresh ≤ reuse×1.10、コード変更なし）ため、F-A・F-B
（pinned ステージング）のいずれも適用対象がない。

- **F-A（`memory.rs::readback` の D2H 宛先を事前タッチ済み `Vec` へ変更）**: 未適用。適用の
  前提となる「fresh のみに 166 ms 規模のホスト側コストが乗る」事象が実測で再現しなかった
  ため、適用すれば根拠のない変更になる（§9 で後続イシュー化を検討する余地として記録するに
  留める）
- **F-B（pinned ステージングバッファ）**: 未適用。理由は F-A と同じ。加えて F-B は `unsafe`
  境界の追加・pinned メモリ資源枯渇のリスクを伴うため、根拠なく導入しない
- **H2/H3（ホストアロケータ・driver 側解放コスト）向けの `mallopt`・独自アロケータ**: 従来
  どおり不採用（公開ライブラリの責務外。§8 原文の判断を維持）

修正前後の 2 系列比較は行っていない（「修正前」= §6 の実測値、「修正後」は存在しない）。
`scripts/bench/framework-compare/results/` は承認済み一次データのため本 PR では更新しない
（実装計画 Step 5 のとおり、`fandhe-ai` は crates.io `=0.4.0` にピン留めされており、本 PR の
記録内容はローカル HEAD の facade 直接呼び出し実測であって framework-compare 自体の再計測
ではない）。

## 9. スコープ外

- train／infer タスクでの同種事象（毎ステップ新規 tape・複数演算）は本イシューの対象外
  （GEMM に限定）
- `tensor-core` の出力バッファプール・ホストアロケータ制御・candle/Burn 側プロトコル
  変更は対象外
- F-A・F-B（§8）はいずれも適用対象なしのため未実施。将来 N=2048 前後で同種の事象が再発した
  場合の予防的措置として F-A（`readback` 宛先の事前タッチ）を検討する余地はあるが、根拠と
  なる実測がない状態での先回り実装は行わない
- §7 で挙げた PR（#1061・#1077・#1079・#1080・#1081）のうちどれが実際に #956 の事象を
  解消したかを個別に切り分ける追加実験（ビルドを各 PR 直前へ戻しての再計測）は実施していない
- §6.3「残存テール」: P0 N=2048 で 5 回中 2 回、Q3 が約 139〜140 ms に達する残存テールが
  観測された。通常の四分位定義では Q3 は上位 25%（20 反復中 約 5 反復）がその水準以上である
  ことを意味し、「まれ」とは言えない規模である。中央値は reuse 同水準で受け入れ条件
  （fresh ≤ reuse+10%、中央値ベース）は満たすが、テールの原因・実際に踏んだ反復数の正確な
  内訳は未特定（#956 の事象が完全消滅ではなく、発生頻度が「20 反復のほぼ全て」から
  「5 プロセス起動中 2 回・各回の上位 25% 程度」へ低下した可能性がある）
- §6.4 の N=4096 P4（`as_slice` 直読み）のばらつき（540 ms ↔ 39 ms）は原因未特定。H2/H3 判別
  用の環境変数注入・`strace` 判別（§5.4）も本イシューでは実施していない（N=2048 が非再現と
  確定した時点で N=4096 固有の別事象として切り出し、本イシューの計測手順を流用できる旨を
  記録するに留める）
- 実機回帰確認（`cargo test -p fandhe-ai-backend-cuda --release -- --ignored --test-threads=1`）
  で観測した以下 6 件の失敗は、本イシューでコード変更を行っていないため非後退の対象では
  なく、**HEAD 時点で既に存在していた事象**として記録するに留める（原因調査・修正は本
  イシューのスコープ外）:
  `gemm::tests::wmma_tf32_basic_kernel_parity_does_not_regress`・
  `gemm::tests::wmma_tf32_opt_kernel_k4096_stress`・
  `gemm::tests::wmma_tf32_opt_kernel_matches_reference_across_shapes`（いずれも
  `baseline_provenance_unconfirmed == true` による意図的な fail-closed。
  `docs/perf/cuda-parity-baseline.md`「ベースライン更新規約」参照）・
  `module_cache_wiring_tests::cuda_gemm_new_stores_disk_cache_entry_for_every_kernel`・
  `nvrtc::jit_cache_bench_tests::jit_cache_bench_cold_compile_vs_warm_load_latency`・
  `nvrtc::jit_cache_bench_tests::jit_cache_bench_module_load_and_throughput_parity`（後者 3 件は
  共有実機ノードの `/tmp` 配下キャッシュルート pin に対する `Invalid argument (os error 22)`
  というノード環境要因で、GEMM D2H 経路とは無関係）

## 10. プール導入後の再計測（#1020／PR #1061）: V3 変種・実測完了

イシュー #1020（`crate::pool::CudaAllocator`。サイズクラス別プール。実装は PR #1061）が
(c) C 確保フェーズをプール経由へ差し替えた。`fresh_overhead_diag_tests.rs::
fresh_overhead_diag_v3_pooled_output`（`#[ignore]`）の実機実測結果を以下に記す（実測記入欄は
`docs/backend-cuda-pool-allocator-decision.md` §7.1 に転記済み）。

| N | `run_tiled_f32`（H2D+launch+D2H+(c) alloc 込みの GEMM 全体。1 回目＝プールミス・warmup） | (c) `alloc_zeroed_f32` 単体（プールヒット・中央値） |
|---|---|---|
| 1024 | 2.616 ms | 0.002 ms |
| 2048 | 8.948 ms | 0.003 ms |
| 4096 | 54.981 ms | 0.003 ms |

**注意（測定区間が異なる。両列を単純比較しない）**: 左列は `run_tiled_f32`（H2D・launch・
synchronize・D2H・(c) alloc を含む GEMM 全体、かつプールミス時の 1 回限りの warmup）、右列は
`alloc_zeroed_f32`（(c) フェーズのみ、プールヒット時）であり、測定対象の範囲が異なる
（`fresh_overhead_diag_tests.rs::fresh_overhead_diag_v3_pooled_output` の実装参照）。左列から
右列への短縮を主張することはできない。(c) フェーズ単体でのプールミス時コストは本テストでは
計測していない。

`release_cached`: `freed_bytes=88080384`（3 サイズ分のキャッシュを正しく解放）。

**正直な記録（更新版）**: 上表からは「V3 が GEMM 全体を短縮した」とは主張できない（測定区間が
異なるため。上記注意参照）。§7 の帰属分析どおり本プールが直接効くのは (c) デバイス側確保
フェーズのみであり、プールヒット時の `alloc_zeroed_f32` 単体コスト（0.002〜0.003 ms）は
§6.2 の V0-fresh (c) 列（0.003 ms 前後）と大差ない値で整合する。ただし V0-fresh は検証対象
コミット `d6bd4ff`（PR #1061 のプール実装コミット `a0c1394` を祖先に持つ HEAD）上で実行されて
おり、計測コードも `CudaGemm::alloc_output_f32` を経由するため、この (c) 列自体が「プール
導入前」の値だと解釈する根拠はない（V0 はプール導入後の HEAD で実行した変種であり、プール
導入前の実測値ではない）。プール導入単独が N=2048 固有の 166 ms を解消した主因であるとは
断定できず（V0-fresh・V3-pooled いずれも (c) は小さい値で大差がない）、§7 のとおり非再現の
実際の主因は特定できていない（#1081 のノードクリア API はコード経路上 fresh 測定に関与しない
ため帰属推定から除外済み。§7 参照）。
