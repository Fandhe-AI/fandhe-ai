# CUDA tape 初期化コスト（約 440〜460 ms）の内訳診断（#926）

イシュー #926「CUDA tape_for 初期化コスト（約 440〜460 ms）の内訳診断」の診断記録。
受け入れ条件（1. 内訳の定量記録、2. 支配的要因の特定、3. Phase 2 対応方針への示唆）に
対応する。計測コード自体は本イシューで整備済み
（`crates/backend-cuda/src/init_cost_diag_tests.rs`。`#[ignore]` 実機専用テスト）。

## 状態: 実機未実測（実装セッションからは DGX Spark GB10 へ到達不能）

本ドキュメントは静的経路分析・既存実測（#391）との突合・計測方法の確立までを完了し、
実測結果（§6）は「未実測・実機セッションへ引き継ぐ」プレースホルダのまま残す。
`docs/perf/cuda-jit-cache-benchmark.md`（#534）が同じ理由（実機セッションでの実測未実施）で
「計測コード＋ドキュメント骨子を先行整備し実機セッションへ引き継ぐ」構成を確立しており、
本ドキュメントも同型の位置づけを取る。

## 1. 背景

フレームワーク横並びベンチ（PR #915・`scripts/bench/framework-compare/results/summary.md`、
計測 2026-08-28）の DGX Spark GB10 実測で、fandhe-ai の CUDA GEMM 中央値が行列サイズに
ほぼ非依存の 440〜460 ms 帯（N=256: 440 ms 〜 N=2048: 458 ms、N=4096 でも 594 ms）に
張り付いた。candle / Burn は同条件で µs〜ms オーダーであり、summary はこの差を
「計測ごとに新規 tape を作る」プロトコル下で fandhe-ai のみ毎回初期化コストを計測区間に
含むことに起因すると推定している。学習 2.418 s/step・推論 1.822 s/回も同じ機構
（ステップごとの新規 tape × 複数演算で `CudaElementwise::new` 等の都度コンパイルが
多重発生）で説明できる見込みだが、詳細分解は本ドキュメントでは GEMM に限定する。

## 2. 帰属の明確化

ベンチ（`scripts/bench/framework-compare/bench-fandhe/src/main.rs::run_gemm`）は計測
イテレーションごとに `facade::tape_for` → `tape.var` を行うが、`Instant::now()` は
**`matmul` 呼び出し直前**に取る。したがって `tape_for` 自体（`crates/facade/src/lib.rs::
tape_for` → `resolve_ops`。デバイス存在プローブ + `CudaBackendOps::new`〈`ordinal` 保持の
みで driver 非接触〉）は計測区間の**外**であり、計測区間に入るのは `matmul` 呼び出し時に
発生する**遅延初期化**である。

`crates/backend-cuda/src/ops.rs::CudaBackendOps` は「`CudaDevice`／`CudaGemm` は各メソッド
呼び出し時に都度構築する」設計（`ops.rs:36-40`。TASK-1.9b のハンドル常駐が未着地である旨が
struct コメントに明記済み）。1 回の `gemm`（= `matmul`）呼び出しごとに以下が発生する
（`ops.rs:299-323`）。

1. `device_handle()`（`ops.rs:58-61`）→ `CudaDevice::new`（`device.rs:88-115`。
   `CudaContext::new(ordinal)` + `default_stream()` + `name()`/`compute_capability()`）
2. `CudaGemm::new`（`gemm.rs:614`）: **NVRTC コンパイル 8 本 + `load_module`/
   `load_function` 8 回**（naive f32/f16・tiled f32/f16・tiled_bias_act_f32 の 5 本は `?`
   合流〈`gemm.rs:617-647`〉、WMMA TF32 基本/opt/staged の 3 本は失敗退避方式
   〈`gemm.rs:674` 以降〉）
3. `run_tiled_f32`（`gemm.rs:1040`）: デバイスバッファ確保 + H2D 転送 + カーネル起動 +
   同期 + D2H 転送

イシュータイトルの「`tape_for` 初期化コスト」は、この `matmul` 起点の遅延初期化を指すと
解釈する。`tape_for` という関数呼び出し自体に初期化コストが乗っているわけではない。

## 3. 静的経路分析

### 3.1 f32 GEMM 経路へのキャッシュ未結線

プロセス内 LRU モジュールキャッシュ（`crates/backend-cuda/src/module_cache.rs`）と NVRTC
ディスクキャッシュ（`nvrtc.rs` の `store_cache_entry`/`load_cache_entry`）の結線は
mma_f16 経路（`kernels_mma.rs::RenderedMmaKernel::compile`。#511・`lib.rs:226-243`
ドキュメンテーションコメント参照）のみである。`CudaBackendOps::gemm`（f32 GEMM の本番経路）
が呼ぶ `CudaGemm::new` はこのキャッシュ機構を一切経由せず、呼び出しのたびに NVRTC
コンパイル・driver ロードを実行する。

加えて `module_cache` のキーは `Arc<CudaContext>` ポインタ識別を含む
（`module_cache.rs` ドキュメンテーションコメント参照）。`ops.rs::device_handle` は毎呼び出し
ごとに新規 `CudaContext`（新しい `Arc` ポインタ）を作るため、仮に `CudaGemm::new` がこの
キャッシュを経由する構成へ変わったとしても、現行の「呼び出しごとに context を作り直す」
構造ではキーが毎回変わり原理的にヒットし得ない。

### 3.2 内訳候補（本ドキュメントの作業仮説）

| 候補 | 該当コード | 想定寄与 |
|------|-----------|---------|
| (p1) デバイス・コンテキスト生成 | `device.rs::CudaDevice::new`（`CudaContext::new` + `default_stream()`） | #391 実測（後述 §4）で約 195〜205 ms |
| (p2) NVRTC source→PTX コンパイル ×8 | `gemm.rs::CudaGemm::new` 内 `compile_ptx` 呼び出し 8 箇所 | 未分離。§4 の first_kernel 累積差分から間接推定するのみ |
| (p3) driver `load_module`/`load_function`（PTX→SASS JIT）×8 | 同上 | 未分離。§4「コールド／ウォームの検証」の cold/warm 差（約 200 ms）が主にこの層に起因する可能性が高い（#391 の解釈） |
| (p4) 実 GEMM 実行（確保 + H2D + 起動 + 同期 + D2H） | `gemm.rs::run_tiled_f32` | N に依存する項（framework-compare の N=256〜4096 でほぼ一定という実測は、この項が (p1)〜(p3) の合計に対し小さいことを示唆） |

作業仮説（実測前の暫定順位。§7 で実測後に確定する）: **NVRTC 8 本の毎回コンパイルが第 1
候補**（8 カーネル分のソース→PTX コンパイルは #391 が未分離のまま「first_kernel_secs」に
含めていた最大の未知項であり、cold/warm 差では説明されない「warm でも約 310〜320 ms
かかる」部分の主因と推定）、**デバイス・コンテキスト生成（p1）が第 2 候補**（#391 実測で
単独 195〜205 ms・全体の 4〜5 割を占める既知の定量値のため）。

## 4. 既存実測との突合

`docs/perf/startup-cost-measurement.md`（#391・GB10 実機実測・2026-08-10）の CUDA 節
（同ドキュメント 185〜286 行）は次を実測済み。

| 指標 | cold（中央値, ms） | warm（中央値, ms） |
|------|---------------------|----------------------|
| `device_init_secs`（`CudaDevice::new`） | 196.4〜204.3 | 192.5〜198.3 |
| `first_kernel_secs`（累積。`CudaGemm::new` 〜初回カーネル完了） | 510.8〜530.8 | 308.1〜322.0 |
| `wall_secs`（プロセス全体） | 621.3〜641.4 | 419.3〜432.3 |

**warm の `wall_secs`（約 419〜432 ms）は framework-compare の 440〜460 ms 帯とオーダーが
整合する**（同一デバイス初期化＋カーネルコンパイル＋実行という測定対象が本質的に同じ
であるため。#391 はプロセスレベルの 1 回計測、framework-compare は計測ループ内で
`CudaBackendOps::gemm` を毎回フレッシュハンドルで呼ぶ点が異なるが、いずれも「都度
`CudaDevice::new` + `CudaGemm::new`」を経由する点は共通）。

#391 は「NVRTC の source→PTX コンパイルはキャッシュ状態に関係なく毎プロセス発生し、
`CUDA_CACHE_PATH` が効くのはドライバ側 PTX→SASS JIT キャッシュのみ」という契約
（`crates/backend-cuda/src/nvrtc.rs` 参照）を明記したうえで、**NVRTC source→PTX と driver
PTX→SASS の寄与の分離計測は「本ハーネスの計測粒度では分離できず、スコープ外」と明記して
いた**（`startup-cost-measurement.md:267-268`）。**本イシューはこの分離を埋める**ことを
目的とし、`init_cost_diag_tests.rs::init_cost_diag_phase_breakdown` がカーネル単位で
`compile_ptx`（NVRTC）と `load_module`/`load_function`（driver）を個別に `Instant` 計測する
構成にした。

## 5. 計測方法

### 5.1 計測コード

`crates/backend-cuda/src/init_cost_diag_tests.rs`（`#[ignore]`・crate ルート直下の兄弟
モジュールとして `lib.rs` に登録。`kernels`／`kernels_wmma_opt` という非公開 `mod` の内部
定数・関数へ到達するため integration test ではなくこの配置を取る。ファイル冒頭コメント
参照）に 3 つの診断テストを実装した。

| テスト関数 | 計測内容 |
|-----------|---------|
| `init_cost_diag_phase_breakdown` | (p1) `CudaDevice::new`・(p2) 8 カーネル個別の `compile_ptx`・(p3) 8 カーネル個別の `load_module`+`load_function`・本番と同一呼び出し列（`CudaGemm::new` 単体）での (p2)+(p3) 合計・(p4) `run_tiled_f32`(N=1024)。ウォームアップ 3 trial + 計測 10 trial の中央値・Q1/Q3 |
| `init_cost_diag_e2e_matches_framework_compare_shape` | 本番 API 経路（`CudaBackendOps::gemm`）を N=256/1024/4096 でフレッシュハンドル計測し、framework-compare が観測した「N にほぼ非依存」の帯を再現するかを記録 |
| `init_cost_diag_reused_handle_steady_state_reference` | 同一 `CudaGemm` ハンドルを 1 度だけ構築し `run_tiled_f32` を反復した場合の 1 回あたり時間（初期化を除いた下限の参照値。内訳帰属の検算専用） |

8 カーネルの内訳（`gemm::CudaGemm::new` と同一のソース・関数名の組）:
naive f32/f16・tiled f32/f16・tiled_bias_act_f32（5 本）・WMMA TF32 基本/opt/staged（3 本）。

すべてのテストは **実行が成功すること**（NVRTC コンパイル・driver ロード・カーネル起動が
例外なく完了すること）のみを検証条件とし、フェーズ間の大小関係・絶対値への `assert!` は
行わない（GPU クロック挙動・他プロセス競合等の環境揺らぎを hard assert に持ち込むと実機
ランナー上で flaky 化するため。`jit_cache_bench_tests.rs`〈#534〉と同じ判断）。数値は
`println!` に残し、本ドキュメント §6 へ転記する一次情報とする。

### 5.2 スコープ縮小: ドライバ側 JIT キャッシュ（`CUDA_CACHE_PATH`）の cold/warm 分離を
行わない

`bench-harness::startup`（#170）はプロセスレベルの cold/warm 比較を子プロセスの環境変数
（`Command::env("CUDA_CACHE_PATH", …)`）で実現する。edition 2024 で `std::env::set_var` が
`unsafe` になり、テストバイナリ全体で共有されるグローバル状態を書き換えると決定的な再現が
できなくなるため（`nvrtc.rs::resolve_cache_root_impl` ドキュメンテーションコメントが同じ
理由を明記）、本イシューの軽量な in-process フェーズ計測では子プロセス分離までは行わず、
**プロセスのアンビエント `CUDA_CACHE_PATH`（実機ランナーの `~/.nv/ComputeCache`。実質
「ウォーム」相当）1 条件のみ**を計測する。ドライバ側 JIT キャッシュの cold 条件を厳密に
分離した計測は本イシューのスコープ外とする（実装完了報告の `outOfScope` に記録）。

### 5.3 実行コマンド（実機）

```bash
cargo test -p fandhe-ai-backend-cuda --release --lib -- --ignored --nocapture --test-threads=1 init_cost_diag
```

`--test-threads=1` は必須（Review #945 指摘）。本ファイルの 3 テストはいずれも
device 0 を使うため、既定の並行テストハーネスのまま起動すると 3 テストの
`Instant` 計測区間が同一 GPU 上で競合し、フェーズ計測（(p1)〜(p4)・e2e gemm・
再利用ハンドル）にカーネル起動待ち・SM 占有の競合が混入して値が歪みうる
（`init_cost_diag_tests.rs` 冒頭コメント「実行時は必ず `--test-threads=1`」参照）。

環境記録項目（GPU・driver・NVRTC 版・rustc・リビジョン）・GPU 占有確認手順は
`docs/real-hardware-verification-env.md` §6.1 に従う。実ホスト名は書かない
（`<cuda-node>` 表記。`docs/real-hardware-verification-env.local.md` 参照方式を踏襲）。

## 6. 実測結果

**未実測**（実装セッションに `docs/real-hardware-verification-env.local.md` が存在せず
DGX Spark GB10 へ到達不能）。実機セッションへ引き継ぐ。

以下は §5.3 のコマンド出力を転記する枠（実測後に埋める）。

### 6.1 実行環境

| 項目 | 値 |
|------|-----|
| GPU | （実測後に記入。`nvidia-smi` 実測） |
| OS | （実測後に記入） |
| CUDA (nvcc) | （実測後に記入） |
| toolchain | stable（`rust-toolchain.toml` 準拠） |
| rustc | （実測後に記入） |
| 計測リビジョン | （実測後に記入。`git rev-parse HEAD`） |
| ビルドプロファイル | `--release` |
| 実施日 | （実測後に記入） |
| GPU 占有状況 | （実測後に記入。`docs/real-hardware-verification-env.md` §6.1 手順） |

### 6.2 フェーズ別実測（中央値 / Q1 / Q3, ms）

| フェーズ | 中央値 | Q1 | Q3 |
|---------|--------|----|----|
| (p1) `CudaDevice::new` | — | — | — |
| (p2) compile_ptx[naive_f32] | — | — | — |
| (p3) load[naive_f32] | — | — | — |
| (p2) compile_ptx[naive_f16] | — | — | — |
| (p3) load[naive_f16] | — | — | — |
| (p2) compile_ptx[tiled_f32] | — | — | — |
| (p3) load[tiled_f32] | — | — | — |
| (p2) compile_ptx[tiled_f16] | — | — | — |
| (p3) load[tiled_f16] | — | — | — |
| (p2) compile_ptx[tiled_bias_act_f32] | — | — | — |
| (p3) load[tiled_bias_act_f32] | — | — | — |
| (p2) compile_ptx[wmma_tf32] | — | — | — |
| (p3) load[wmma_tf32] | — | — | — |
| (p2) compile_ptx[wmma_tf32_opt] | — | — | — |
| (p3) load[wmma_tf32_opt] | — | — | — |
| (p2) compile_ptx[wmma_tf32_staged] | — | — | — |
| (p3) load[wmma_tf32_staged] | — | — | — |
| (p2+p3 本番同一呼び出し) `CudaGemm::new` | — | — | — |
| (p4) `run_tiled_f32`(N=1024) | — | — | — |
| (p1+p2+p3+p4 再構成合計) | — | — | — |

### 6.3 e2e 整合確認（`CudaBackendOps::gemm`、N 別中央値, ms）

| N | 中央値 |
|---|--------|
| 256 | — |
| 1024 | — |
| 4096 | — |

### 6.4 再利用ハンドルの定常状態参照値

| 指標 | 中央値, ms |
|------|-----------|
| reused-handle `run_tiled_f32`(N=1024) per-call | — |

## 7. 支配的要因の判定

**未確定**（§6 の実測記入後に確定する）。§3.2 の作業仮説（NVRTC 8 本の毎回コンパイルが
第 1 候補、デバイス・コンテキスト生成が第 2 候補）を実測値で検証し、本節を更新する。

## 8. Phase 2 への示唆

実測確定前の優先順位案（§3.2 の作業仮説に基づく暫定順位。実測後に確定する）:

1. **(a) デバイスハンドル・コンパイル済みカーネルの常駐化**: `ops.rs::CudaBackendOps` は
   `ordinal: usize` のみを保持し、`CudaDevice`／`CudaGemm` を都度構築する設計
   （`ops.rs:36-40`。TASK-1.9b 系）。呼び出し間でハンドルを再利用する構成へ変更すれば
   (p1)〜(p3) を初回 1 回のみに削減できる。#925（bench-harness の再利用モード追加）が
   このハンドル常駐化を計測面から支える前提となる。
2. **(b) f32 GEMM 経路への `module_cache`／NVRTC ディスクキャッシュ結線**: 現状 mma_f16
   経路（#511）にのみ結線されているプロセス内 LRU・ディスクキャッシュを、`CudaGemm::new`
   （naive/tiled/WMMA-TF32 の 8 カーネル）へも横展開する。ただし §3.1 のとおり、(a) の
   context 常駐化を伴わない限り `module_cache` のキー（`Arc<CudaContext>` ポインタ識別）は
   毎回変わりヒットしないため、**(a) と (b) は独立ではなく (a) が (b) の前提**である
   点に注意する。
3. **(c) context をプロセス内で共有しキャッシュキーのポインタ識別を実効化する設計**:
   (a)(b) を実現する具体的な機構（`ordinal` ごとの `CudaDevice`／`CudaGemm` シングルトン
   キャッシュ等）の設計自体。並行性（複数スレッドからの共有）・エラー時の再構築方針を
   含めた設計判断が必要であり、本イシューのスコープ外（Phase 2 の実装イシューで設計する）。

## 9. スコープ外

- デバイスハンドル・カーネルの常駐化などの**改善実装**（Phase 2。本ドキュメントは示唆のみ）
- bench-harness への再利用モード追加（#925）
- 初期化除外後のカーネル単体性能の対 candle/Burn ベースライン計測（#928）
- Metal 側の固定オーバーヘッド診断（#927）
- `scripts/bench/framework-compare/` の変更（承認済みピン構成のため不変更）
- ドライバ側 JIT キャッシュ（`CUDA_CACHE_PATH`）の cold/warm 厳密分離計測（§5.2 参照。
  子プロセス分離を要するため本イシューの軽量 in-process 計測では扱わない）
