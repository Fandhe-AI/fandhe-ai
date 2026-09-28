# candle 0.11.0 CPU GEMM 解析（gemm crate vs 自作 BLIS）

イシュー #2092。親 #2089（Phase 2「他ライブラリのコード取得・詳細解析」2.3）・
祖父 #2088 系列の並列兄弟は #2090〜#2098（#2098 が集約）。

## 1. 位置づけ・目的

framework-compare のスコアボード（2026-09-19 版。イシュー本文の記載値）では、
Apple M4 Max の CPU gemm が candle 0.11.0 に対し N=256/1024/2048 で 0.66〜0.87 倍と
記録されている。**この 0.66〜0.87 という値、および題目に記載された「0.84〜0.97×」
という値は、本リポジトリ内の実測記録として逐語の出典を確認できなかった**（§7 で
リポジトリ内に実在する系列を代わりに整理する。数値の再構成・裏取りは行わない）。

candle 0.11.0 の CPU GEMM の実体は自作カーネルではなく [`gemm`](https://github.com/sarah-ek/gemm/)
crate 0.19.0（sarah-ek/gemm。faer 系）である。これまで対 gemm crate の差分は
施策単位の A/B（KC 再スイープ・B laneq ベクトル転置・prefetch・2D 動的分配・SME 等）
で個別に検証されてきたが、gemm crate 側の実装（マイクロカーネル・packing・スレッド
分配・ブロックサイズ決定）を体系的に読んで自作との構造差を 1 か所にまとめた記録は
まだない。

本 doc の目的は、**読み取り解析のみ**で構造差を整理し、M4 Max での劣位を**仮説として**
帰属させ、Phase 3（実測・A/B）と #2098（原因帰属の対照表）の入力にすることである。
性能実測・A/B・数値契約の変更は行わない（スコープ外）。

## 2. 解析対象と版

| 対象 | 版 | 取得元・確認方法 | ライセンス |
|------|----|--------------------|-----------|
| `gemm` | 0.19.0 | ローカル cargo registry cache（`~/.cargo/registry/src/index.crates.io-*/gemm-0.19.0/`）。`.cargo_vcs_info.json` に vcs sha `86102c5b712737978371ac9ef7a11982f686d7bc`（[github.com/sarah-ek/gemm](https://github.com/sarah-ek/gemm/)） | MIT（`Cargo.toml` 実測） |
| `gemm-common` | 0.19.0 | 同上 | MIT |
| `gemm-f32` | 0.19.0 | 同上 | MIT |
| `gemm-f64`／`gemm-f16`／`gemm-c32`／`gemm-c64` | 0.19.0 | 同上（型別実装。本 doc の対象は f32 のみで詳細は読んでいない） | MIT |
| `candle-core` | 0.11.0 | `curl -fsSL https://static.crates.io/crates/candle-core/candle-core-0.11.0.crate` を取得し、`Cargo.lock`（`scripts/bench/framework-compare/Cargo.lock` L899-901）の checksum `5ecb245093b0f791b89d3420c3df9c6d49c60ab63ba54db896bf8a3baf486706` と一致を確認してから scratchpad へ展開（ビルド・実行はしていない）。解析後に削除済み | MIT（candle-core 本体。本 doc では未再確認・framework-compare 側の既存整理に委ねる） |

自作側は `crates/backend-cpu/src/gemm_blis/`（`mod.rs`・`cache_params.rs`・`pack.rs`・
`partition.rs`・`microkernel.rs`・`microkernel/neon.rs`）と
`crates/backend-cpu/src/thread_limit.rs`・`small_shape_thread_cap.rs` を対象とする。
イシュー本文の記載パス（`gemm_blis_parallel.rs`・`partition.rs` 直下）は実在しないため、
上記の実パスへ読み替えている。

## 3. マイクロカーネル（R1）

### 3.1 gemm crate（aarch64 f32・neon 経路）

型ディスパッチは `gemm@0.19.0:src/gemm.rs` の `gemm_dispatch`（L58-101）が
`TypeId` で分岐し、f32 は `gemm_f32::gemm::f32::get_gemm_fn()`（L80-101）を呼ぶ。
`gemm-f32@0.19.0:src/gemm.rs`（3 行のみ）は `gemm_common::gemm_def!(f32, 2)` を
展開する。`gemm-common@0.19.0:src/gemm.rs` の `gemm_def!` マクロ（L986-1093）が
実行時 CPU 機能検出（`feature_detected!("neon")`）で `neon::gemm_basic` を選び
（`experimental-apple-amx` feature は本 crate では未有効化のため amx 経路は不使用。
§3.3）、`__inject_mod!(neon, f32, 2 * 2, Scalar, false)`（L1084-1085）で
レーン幅 `N = 4`（f32×4 = 128bit NEON レジスタ 1 本）を確定する。

具体的なマイクロカーネル本体は `gemm-f32@0.19.0:src/microkernel.rs` の
`neon::f32` サブモジュール（L357-452）にあり、`microkernel!` マクロ
（`gemm-common@0.19.0:src/microkernel.rs` L398 以降）で `x1x1`〜`x4x4` の
16 変種を展開したうえで、`microkernel_fn_array!` マクロ
（`gemm-common@0.19.0:src/microkernel.rs` L99-111）が 4×4 の関数ポインタ表
`UKR`（型 `[[MicroKernelFn<T>; NR]; MR_DIV_N]`）を構築し、`MR_DIV_N = 4`・
`NR = 4` を確定する。`__inject_mod!` 側で `MR = { MR_DIV_N * N }`（L885）と
展開されるため、**aarch64 f32 の本番マイクロカーネルタイルは MR=16・NR=4**
（アキュムレータは `MR_DIV_N(4) × NR(4) = 16` 本の `float32x4_t`、A ロードに
`vld1q_f32_x4`〈4 本〉、B に 1 本で計 21 本。v0〜v31 の 32 本の NEON レジスタに
十分収まる）。FMA は `x4x4` 変種（`microkernel!(["neon"], 4, x4x4, 4, 4, 1, 4)`。
`gemm-f32@0.19.0:src/microkernel.rs` L446）がレーン指定 FMA
（`mul_add_lane::<LANE>` → `vfmaq_laneq_f32::<LANE>`。`gemm-f32@0.19.0:src/microkernel.rs`
L392-399）を使う設計で、B 側の 1 本のベクタレジスタを 4 レーンへブロードキャストせず
レーンごとに `vfmaq_laneq_f32` を発行する（`gemm-common@0.19.0:src/microkernel.rs`
L468-494 の `execute_neon` 分岐）。K ループの unroll 数は 4（`microkernel!` 第 2 引数）。

`x4x4` 以外の 15 変種（`x1x1`〜`x3x4` 等）は端数（M・N が MR・NR の倍数でない
残り）処理用で、`UKR[mr_idx][nr_idx]` 表から実行時に選ばれる（実装は確認したが
選択ロジック自体〈`gemm_basic_generic` 内の呼び出し箇所〉までは読み切っていない。
本 doc の R1 の範囲では「主カーネルは MR=16/NR=4 固定・端数は表引きで縮小変種」
までを確定事実とする）。

`amx`（`experimental-apple-amx` feature 限定。本 crate の `Cargo.toml`〈candle 経由〉
では未有効化。§3.3・§4.2）も同モジュールに存在する（`gemm-f32@0.19.0:src/microkernel.rs`
L461-486）が、feature ゲートにより到達しない。

### 3.2 自作（`gemm_blis`・本番既定 neon）

`SME_PRODUCTION_ENABLED = false`（`crates/backend-cpu/src/gemm_blis/mod.rs:3085`）
のため M4 Max の本番マイクロカーネルは neon 固定カーネル
（`crates/backend-cpu/src/gemm_blis/microkernel/neon.rs`）。`MR = 8`・`NR = 12`
（同ファイル L154・L157。BLIS armv8a 型・NR は f32x4 レジスタ 3 本ぶん）。
firestorm 型 A/B 対抗の 12×8 変種（`MR_12X8 = 12`・`NR_12X8 = 8`。同ファイル
L797-799）も存在するが本番未選択（同ファイル冒頭コメント「両実機での採否判断は
#1318 へ引き継ぎ」）。FMA は `vfmaq_laneq_f32`（同ファイル L150 のインポート）を
使っており、gemm crate と同じくレーン指定 FMA 方式である。

### 3.3 差分要点

- gemm crate は MR=16/NR=4（アキュムレータ 16 本）、自作は MR=8/NR=12
  （アキュムレータ `8/4 × 12 = 24` 本相当）。M 方向を広く・N 方向を狭く取る
  gemm crate と、N 方向を広く取る自作という形状の向きが逆
- 両者とも aarch64 では `vfmaq_laneq_f32` によるレーン指定 FMA を採用しており、
  FMA 命令の種別自体は一致する（丸め方針上の構造差ではない）
- `amx` 経路は gemm crate 側に存在するが、**candle 0.11.0 の `Cargo.toml`
  が `experimental-apple-amx` feature を有効化していない**（§4.2）ため到達しない。
  加えて `gemm-common@0.19.0:src/cache.rs` の `has_amx_impl()`（L52-70）は
  `machdep.cpu.brand_string` が `"Apple M1"|"M2"|"M3"` のいずれかである場合のみ
  `true` を返す実装になっており、**そもそも Apple M4 系では feature を有効化しても
  amx 経路には入らない**（brand 文字列のパターンマッチに `"M4"` が含まれていない）。
  この点は M4 Max の劣位要因の候補から外せる

## 4. packing（R2）

### 4.1 gemm crate

`gemm_basic_generic`（`gemm-common@0.19.0:src/gemm.rs` L174 以降）の要点:

- **RHS（B）の pack 要否**は aarch64 では
  `do_pack_rhs = _requires_row_major_rhs || m > get_rhs_packing_threshold() * MR`
  （同ファイル L411-412）で、`DEFAULT_RHS_PACKING_THRESHOLD = 2`（aarch64 限定。
  同ファイル L112-113。コメント「we REALLY want to pack the rhs on aarch64 since
  we can use mul_add_lane」）。x86 系は `128` で aarch64 の方が積極的に pack する
- **LHS（A）の事前 pack 要否**は `do_prepack_lhs = m <= 2 * mc && ((m % N != 0) || lhs_rs != 1)`
  （同ファイル L418）で、単一・複数スレッドで別の閾値
  （`DEFAULT_LHS_PACKING_THRESHOLD_SINGLE_THREAD = 8`・`_MULTI_THREAD = 16`。
  同ファイル L117-118）を使う
- **pack バッファは `dyn_stack::MemBuffer` で 1 回だけ確保**し（同ファイル L420-440。
  `do_pack_rhs || do_prepack_lhs` のときのみ）、K の外側ループ（`depth_outer`）ごとに
  その中身を pack し直す（同ファイル L500 以降のループ構造）。**RHS の pack は
  n チャンク・k チャンクの単位で 1 回だけ行い（同ファイル L530-624）、その後の
  m 方向ジョブ分配（§5）は全スレッドが同じ `packed_rhs` バッファを読む共有方式**
  （`n_threads <= 1` は単一スレッドで pack、`n_threads > 1` は `par_for_each` で
  n 方向を `n_tasks.msrv_div_ceil(NR)` 個のタスクに `base/rem` 均等分配して並列 pack。
  同ファイル L560-617）
- LHS の事前 pack（`do_prepack_lhs` が真の場合）も同様に 1 回だけ pack し
  （同ファイル L625-636）、以降のジョブから共有読み出しする
- `do_prepack_lhs` が偽の場合（大半の M サイズ）は、LHS はジョブ実行時に
  各スレッドがオンデマンドで pack する（`did_pack_lhs` 済みフラグ配列で二重 pack
  を避ける。同ファイル L490・L652-659 付近）。この場合の LHS pack バッファは
  スレッドごとの一時領域（`did_pack_lhs_storage`。tid=0 のみ外側の共有配列を使う）

### 4.2 candle 0.11.0 の呼び出し層

取得した `candle-core-0.11.0/Cargo.toml`（L188-190）の `[dependencies.gemm]` は
`features = ["wasm-simd128-enable"]` のみで、**`experimental-apple-amx` は
0.11.0 でも未有効化**（0.10.2〈ローカル registry cache〉と同一。§3.3 の amx 未到達
判断はそのまま成立する）。stride・転置の渡し方（`cpu_backend/mod.rs` の matmul
呼び出し箇所の cs/rs 引数構成）までは本 doc では読み切っていない（§9 参照）。

### 4.3 自作（`gemm_blis::pack`）

`crates/backend-cpu/src/gemm_blis/pack.rs`（モジュール doc）は BLIS 5-loop の
pc/ic 層で使う連続バッファ生成であり、panel バッファへの直接書き込み方式
（#554。同ファイル L20-26 付近）を採る。`docs/cpu-gemm-b-packing-sharing-decision.md`
（#565）の実測ベース整理（§A）によれば、**当時の構造は「全タスクが同一の
jc×pc 範囲（B の列・K のブロック）に対して各自 pack しており、タスク間（行パネル間）
では共有されない」**（同 doc §A・L44 付近）。同 doc はこの重複コストを解析モデルで
見積り（§B。NC 拡大候補では 1 本あたり数十〜百 MiB 級）、BLIS 方式（共有 pack ＋
ic 並列）を推奨案としたが、**2026-09-28 時点で結線されているのは 2D 動的分配
（`TWO_D_DYNAMIC_PRODUCTION_ENABLED = true`。#1313）であり、#565 の記述がこの
2D 動的分配後の実装に完全に当てはまるかは本 doc では確認していない**（#565 は
2D 動的分配採用〈#1313〉より前の記録の可能性がある。§7・§9 で仮説として扱う）。

### 4.4 差分要点

| 項目 | gemm crate（aarch64） | 自作（本番既定） |
|------|----------------------|-------------------|
| RHS pack 要否 | `m > 2 * MR` またはロウメジャー強制（aarch64 は積極的） | 未確認（§9） |
| RHS pack の共有範囲 | 1 回 pack → 全スレッド共有読み出し | #565 時点は「タスク間で共有されない」（2D 動的後の実態は未確認） |
| LHS pack | 条件付き事前 pack（`m <= 2*mc` かつ非連続）。それ以外はジョブ実行時オンデマンド＋重複防止フラグ | 未確認（§9） |
| pack バッファ確保 | `dyn_stack::MemBuffer` を呼び出しあたり 1 回確保 | panel バッファへの直接書き込み方式（#554） |

## 5. スレッド分配（R3）

### 5.1 gemm crate

- **並列化しきい値**: `DEFAULT_THREADING_THRESHOLD = 48 * 48 * 256`（`gemm-common@0.19.0:src/gemm.rs`
  L109）。実行時に `set_threading_threshold` で変更可能な `AtomicUsize`
  （同ファイル L120・L132）。`total_work = m * n_chunk * k_chunk` がこの値未満なら
  `n_threads = 1`（同ファイル L511-522）。複素数型は `/4`（c32）・`/16`（c64）補正
  （同ファイル L397-409。f32/f64 は補正なし）
- **スレッド数**: `Parallelism::Rayon(n_threads)` の `n_threads == 0` なら
  `rayon::current_num_threads()`、それ以外は指定値をそのまま使う（同ファイル L385-395）。
  candle 側は `Parallelism::Rayon(get_num_threads())`（§5.2）で呼ぶため、
  `n_threads == 0` にはならず candle 側のスレッド数がそのまま伝わる
- **ジョブ空間**: n 方向を `NR` 単位・m 方向を `MR` 単位に分割した 2 次元ミニチャンク
  （`n_col_mini_chunks × n_row_mini_chunks`）を 1 次元の job id 空間へ平坦化し
  （同ファイル L638-650）、`min_jobs_per_thread = n_jobs / n_threads`・
  `rem = n_jobs - n_threads * min_jobs_per_thread` で**静的な連続区間分配**
  （各スレッドは `min_jobs_per_thread` または `+1` 個の連続 job id を担当。
  同ファイル L661-672）。**動的な work-stealing による粒度調整はこの層には無い**
  （rayon の `into_par_iter().for_each` は `0..n_threads` という小さいイテレータに
  対して呼ばれるだけで、ジョブそのものの再分配は行われない。§5.3 参照）
- RHS pack の並列化も同じ `base/rem` 静的分配パターンを使う（§4.1）

### 5.2 candle 0.11.0 の呼び出し層

`candle-core-0.11.0/src/utils.rs`（scratchpad 展開物）を読むと:

- `get_num_threads()`（L343-346）は rayon と同じ環境変数 `RAYON_NUM_THREADS`
  を見る `rayon_num_threads()` へ委譲
- `rayon_num_threads()`（L327-333）は `RAYON_NUM_THREADS` が未設定・不正なら
  `default_num_threads()` にフォールバック
- `default_num_threads()`（L313-325）は **macOS では
  `perf_core_count()`（`hw.perflevel0.logicalcpu` を sysctl で取得。P コアの
  論理コア数）を最優先し、取得できない場合のみ `num_cpus::get_physical()`
  （全物理コア）にフォールバック**。macOS 以外は常に `num_cpus::get_physical()`
- さらに candle は独自の永続 `rayon::ThreadPool`（`POOL: OnceLock<rayon::ThreadPool>`。
  L367）を持ち、`num_threads(get_num_threads())` に加えて
  **`start_handler` で各ワーカースレッドの QoS を `QOS_CLASS_USER_INTERACTIVE`
  へ引き上げる（`set_thread_affinity`。L379-386）**。これは macOS スケジューラに
  「P コアで実行してほしい」ヒントを与える機構で、スレッド数の P コア限定
  （`perf_core_count`）と合わせて **2 重に P コア優先化**している
- `gemm::Parallelism::None` になる条件（1 要素 gemv 等の特殊系列を除く一般の
  matmul 呼び出しでの分岐）までは本 doc では読み切っていない（§9）

### 5.3 自作（`gemm_blis`。本番既定 `TwoDDynamic`）

- `TWO_D_DYNAMIC_PRODUCTION_ENABLED = true`（#1313。`mod.rs:3072`）・
  `TWO_D_JOBS_PER_WORKER = 2`（`mod.rs:3051`）で `dispatch_two_d_dynamic` を
  `crate::small_shape_thread_cap::run_capped`（小形状の仕事量ベース並列度上限。
  #1575）と `crate::gb10_affinity::with_gb10_affinity_if_applicable`（GB10 の
  大コア affinity ルーティング。#1576）でラップして呼ぶ（`mod.rs:594・784` 付近）。
  「`TwoDDynamic`」という名前・`TWO_D_JOBS_PER_WORKER` という定数名から、
  gemm crate の静的 `base/rem` 分配とは異なる**動的**なジョブ取得（ワーカーあたり
  複数ジョブを持たせて動的に消化する設計）が想定されるが、`dispatch_two_d_dynamic`
  自体の内部実装（`partition.rs::job_grid`・`JobGrid` 構造体。`partition.rs:151・366`）
  までは本 doc では読み切っていない（§9）。詳細設計は
  `docs/cpu-gemm-2d-dynamic-partition-design.md`・実測 A/B は
  `docs/perf/cpu-gemm-2d-dynamic-partition-ab.md` を参照
- **スレッド数**: `TwoDDynamic` 経路は `rayon::current_num_threads()` をそのまま
  使う想定（`TWO_D_DYNAMIC_PRODUCTION_ENABLED` 分岐では `effective_num_threads`
  の呼び出しが見当たらない。`else` 側〈行パネル分割・非本番〉のみが
  `crate::thread_limit::effective_num_threads(rayon::current_num_threads())`
  を呼ぶ。`mod.rs:805` 付近）。すなわち **`TwoDDynamic` は P/E コアを区別せず
  rayon の既定スレッド数（通常は全論理コア数）をそのまま使う**
- `thread_limit::BIG_CORE_LIMIT_ENABLED`（P コア限定への単一 const ゲート）は
  **`false`（`thread_limit.rs:112`）で無効**。この機構自体は candle の
  `perf_core_count()` P コア限定と構造的に同型だが、イシュー #1364 の実機実測
  （`docs/perf/cpu-gemm-default-thread-limit.md` §6）で REJECT が確定している
  （§7 で詳述）

### 5.4 差分要点

| 項目 | gemm crate（candle 経由） | 自作本番（`TwoDDynamic`） |
|------|---------------------------|---------------------------|
| 並列化しきい値 | `m*n_chunk*k_chunk < 48*48*256` で直列化 | 未確認（`should_serialize`。`mod.rs:269`。詳細は §9） |
| ジョブ空間 | m×n の 2D ミニチャンク→1D job id | 「2D」を明示する命名（`TwoDDynamic`）だが内部詳細未確認 |
| 分配方式 | 静的 `base/rem` 連続区間（動的取得なし） | 命名上は動的（`TWO_D_JOBS_PER_WORKER` によるワーカーあたり複数ジョブ） |
| スレッド数 | P コア数（macOS）or 全物理コア（非 macOS）。RAYON_NUM_THREADS で上書き可 | rayon 既定（通常は全論理コア。P/E 区別なし。`BIG_CORE_LIMIT_ENABLED=false`） |
| P コア優先化 | スレッド数限定 + QoS 引き上げの 2 重 | 機構はあるが無効化済み（#1364 REJECT） |

## 6. 自作との差分表まとめ（R4）

| 項目 | gemm 0.19.0（candle 経由。aarch64） | 自作本番（`TwoDDynamic`・neon） | 出典 |
|------|--------------------------------------|-----------------------------------|------|
| カーネル形状 MR×NR | 16×4（`MR_DIV_N=4, N=4`） | 8×12 | §3.1・§3.2 |
| FMA 種別 | `vfmaq_laneq_f32`（レーン指定） | `vfmaq_laneq_f32`（レーン指定・同種） | §3.1・§3.2 |
| ブロックサイズ決定 | `kernel_params()` による実行時キャッシュ連想度ベース導出（`m,n<=64` は簡略式）。sysctl 実測に依存 | `default_blocks()` 固定値（MC=128/KC=256/NC=512。実機スイープで確定・#749/#1315） | §4.1（`gemm-common@0.19.0:src/cache.rs` L514-589）・`gemm_blis/mod.rs:145-167` |
| ランタイムキャッシュ検出の本番結線 | 常時有効（`kernel_params` が毎呼び出し `CACHE_INFO`〈OnceCell〉を参照） | `cache_params.rs` に実装はあるが本番 3 公開関数は未結線（`default_blocks()` 固定。`docs/perf/cpu-gemm-runtime-cache-detect.md`） | `gemm_blis/mod.rs:143-159` |
| RHS pack 共有 | 1 回 pack → 全スレッド共有読み出し | #565 時点は非共有（2D 動的後の実態は §9 へ） | §4.1・§4.3 |
| pack バッファ確保 | `dyn_stack::MemBuffer` を呼び出しあたり 1 回 | panel バッファ直接書き込み（#554） | §4.1・§4.3 |
| ジョブ分配 | 静的 `base/rem` 連続区間 | 命名上は動的（`TwoDDynamic`。詳細未確認） | §5.1・§5.3 |
| スレッド数方針 | macOS で P コア数限定 + QoS 引き上げ（既定 ON） | rayon 既定（P/E 区別なし。P コア限定機構は #1364 で REJECT 済み） | §5.2・§5.3 |
| 並列化しきい値 | `m*n_chunk*k_chunk < 48*48*256` | `should_serialize`（詳細未確認） | §5.1・`mod.rs:269` |
| amx（Apple 専用命令）経路 | crate には存在するが feature 未有効化・M4 では brand 判定的にも到達不能 | 該当機構なし（NEON のみ） | §3.3 |
| C 書き込み（staging） | 未確認（本 doc 範囲外） | 未確認（本 doc 範囲外） | §9 |
| 転置入力の扱い | `gemm@0.19.0:src/gemm.rs` の `do_transpose`（dst 列優先/行優先で lhs/rhs を入れ替え）（L179-223） | `GemmTranspose::Nn` 固定引数を持つ API（`dispatch_two_d_dynamic` 呼び出し引数） | §3.1（`gemm.rs`）・`mod.rs:797` |

## 7. M4 Max での劣位の構造的仮説（未実測・R5）

**前置き**: 以下はすべて仮説であり、実測（Phase 3）で検証されていない。GB10 では
自作が優位という非後退の実測記録（§7.4）との整合性を判断材料に含めている。

### 7.1 仮説 A: P コア限定スレッド方針の差

candle は macOS で既定 P コア数（M4 Max: 12）のみを使い、ワーカースレッドの QoS も
引き上げる（§5.2）。自作 `TwoDDynamic` は rayon 既定（M4 Max: 16 論理コア＝P12+E4）
をそのまま使い、E コアも並列ワーカーに含める。E コアがボトルネックスレッドになる
「ストラグラー」効果で全体のジョブ完了が遅延する可能性がある。

- **既存実測との整合**: イシュー #1364 の on/off 比較（`docs/perf/cpu-gemm-default-thread-limit.md`
  §6.2）では、**Apple M4 Max 単体では P コア限定が 0.80〜0.91 倍（改善）**だった
  （全 6 セルで一貫）。これは本仮説を支持する方向の実測結果である
- **ただし総合判定は REJECT**: 同じ機構を DGX Spark GB10 で有効化すると、
  Grace CPU の非一様な `cpu_capacity` 分布により大コア検出が `Some(1)` に縮退し
  実質シングルスレッド化、1.19〜4.34 倍の重大な性能後退を起こした（同 doc §6.2）。
  決定規則（両実機で非後退を要求）に照らして不採用が確定している
- **GB10 との非対称性との整合**: GB10 では自作が gemm crate（候補・候補 candle
  いずれも）と拮抗〜優位（§7.4）であり、P コア限定を持たない自作の現在の挙動が
  GB10 側では有利に働いている可能性と整合する。すなわち「P コア限定」は
  M4 Max だけを見れば効きうる差分だが、両実機の非対称性込みで見ると
  **プラットフォーム別の実装（機種判定で分岐する設計）でない限り単純に真似できる
  差分ではない**、という仮説になる

### 7.2 仮説 B: ブロックサイズの導出方式（固定 vs 実行時キャッシュ連想度ベース）

gemm crate は `kernel_params()` で L1/L2/L3 の連想度・サイズから毎呼び出し
KC/MC/NC を導出する（§4.1・§6）。自作は M4 Max 実機スイープで固定した
MC=128/KC=256/NC=512 を使う（`docs/perf/cpu-gemm-blocking-sweep.md` §7）。
gemm crate の `kernel_params` は `auto_mc = min(auto_mc, 8*mr)`
（`gemm-common@0.19.0:src/cache.rs` L589）という上限を持ち、mr=8（aarch64 neon
の `MR_DIV_N=4, N=4→MR=16`ではなく式中の `mr` 引数はマイクロカーネルの `MR` その
ものを指すため実際は `min(auto_mc, 8*16)=128` 相当）で、値としては自作の
MC=128 と近い可能性がある一方、KC/NC は実機の実測キャッシュサイズに応じて
動的に変わる。固定値が特定の M4 Max 個体でチューニングされている（#749・#1315
は KC の細粒度再スイープでも KC=256 を上回れないと結論）ため、ブロックサイズ
自体が劣位要因である可能性は低いと見るが、**キャッシュ連想度を考慮した動的導出
という「方式」そのものの効果（NC の n 依存拡大が有効化されていない #753 の
未着手事項とも関連）**は未検証のまま残っている

### 7.3 仮説 C: RHS packing の共有範囲

gemm crate は RHS を 1 回 pack して全スレッドが共有読み出しする（§4.1）のに対し、
自作は #565 時点で「タスク間で共有されない」（§4.3）。2D 動的分配後の実態が
同じ非共有構造のままなら、スレッド数が多いほど B 側の重複 pack コストと
帯域消費が線形に増える。M4 Max（P12+E4=16 論理コア）は GB10（P/E 非対称だが
`lscpu` 上は 10+10=20 論理コア）よりスレッド数が少ないため、この仮説が正しければ
「スレッド数が多いほど不利」となり GB10 の方が影響が大きいはずで、**GB10 で
自作が優位（§7.4）という実測と方向が逆になる**。したがって本仮説は
「pack 共有の欠如が主要因」とは考えにくいという消極的な判断材料になる
（ただし 2D 動的分配後の実装を確認していないため確証はない。§9）

### 7.4 GB10 との非対称性（参考実測）

- 対 gemm crate 直接比較（`oss-gemm-compare`。`Parallelism::Rayon(0)`）:
  M4 Max は RowPanel（旧本番既定）比で 0.838〜0.889（`docs/perf/cpu-gemm-candle-cpu-retune.md`
  §8 実測表。N=1024/2048/4096）
- 対 candle（framework-compare。candle 側のスレッド数方針経由。`TwoDDynamic`
  採用後・`fandhe-ai =0.9.0` 正式系列）: M4 Max 系列 B は N=512 で 1.152（達成）・
  N=1024 で 0.921（未達）・N=2048 で 0.899（未達）
  （`docs/perf/cpu-gemm-candle-gate-remeasurement.md` §25.2）
- 同じ系列で **DGX Spark GB10 は N=1024 で 1.013・N=2048 で 1.267・N=4096 で 1.785**
  （すべて達成。同 doc §25.2）と自作が明確に優位

M4 Max のみ N>=1024 で劣位・GB10 は全域で優位という非対称性は、§7.1 の P コア
限定仮説（M4 Max では効くが機種判定なしには適用できない）と最も整合する。

## 8. 差分候補と既存判定のタグ（契約: 重複実験なし。R6 対応の一部）

| 候補 | 既存判定 | 出典 |
|------|---------|------|
| KC 再スイープ（128〜512 グリッド） | **REJECT**（現行 KC=256 を上回る候補なし） | `docs/perf/cpu-gemm-candle-cpu-retune.md` §8.1 |
| B laneq ベクトル転置 | **REJECT** | `docs/perf/cpu-gemm-candle-cpu-retune.md` §8.2 |
| prefetch intrinsics（`asm!` PRFM） | 到達可能だが **ユーザー承認事項として保留**（採否未確定。`unsafe` 新規導入との整合） | `docs/cpu-gemm-prefetch-decision.md` |
| 2D 動的分配（`TwoDDynamic`） | **ADOPT 済み**（本番結線）。#1313 | `mod.rs:3072`・`docs/cpu-gemm-2d-dynamic-partition-design.md` |
| SME `fmopa` マイクロカーネル | 実装済みだが**本番ゲート OFF**（`SME_PRODUCTION_ENABLED=false`） | `docs/perf/cpu-gemm-sme-fmopa-microkernel.md`・#1587 |
| P コア限定スレッド数（`BIG_CORE_LIMIT_ENABLED`） | **REJECT**（GB10 の非一様 capacity 検出により重大後退。M4 Max 単体では改善方向） | `docs/perf/cpu-gemm-default-thread-limit.md` §6 |
| B パネル packing のスレッド間共有化（BLIS 案 B） | **推奨案どまり**（#565 で設計検討・未実装。2D 動的分配後の再評価は未実施） | `docs/cpu-gemm-b-packing-sharing-decision.md` |
| ランタイムキャッシュ検出（`cache_params.rs`）の本番結線 | **実装済み・未結線**（機種識別子未確定のため #753 へ保留） | `docs/perf/cpu-gemm-runtime-cache-detect.md`・`mod.rs:143-159` |
| gemm crate 型のキャッシュ連想度ベース動的ブロックサイズ導出方式そのものの採用可否 | **新規（未検討）**。Phase 3 送り | 本 doc §7.2 |
| `dispatch_two_d_dynamic`／`JobGrid` の内部実装（静的か動的か・pack 共有の有無） | **新規（本 doc で未読了）**。Phase 3 または後続の explorer 調査送り | 本 doc §5.3・§9 |

## 9. 本 doc で読み切れなかった点（Phase 3 への申し送り）

- `dispatch_two_d_dynamic`・`partition.rs::job_grid`／`JobGrid` の内部実装
  （静的均等割りか動的ワークスティーリングか、`TWO_D_JOBS_PER_WORKER=2` の意味）
- `gemm_blis::pack` の 2D 動的分配後の RHS/LHS pack 共有範囲（#565 の記述が
  現行構造にそのまま当てはまるかどうかを含む）
- `should_serialize`（`mod.rs:269`）の直列化しきい値の具体的な式（gemm crate の
  `48*48*256` との比較）
- candle `cpu_backend/mod.rs` の matmul 呼び出し箇所での stride・転置フラグの
  渡し方、および `Parallelism::None` になる具体的条件
- gemm crate の `UKR` 表からの端数カーネル選択ロジック（`gemm_basic_generic` 内
  の呼び出し箇所）

## 10. 出典・ライセンス

- `gemm`・`gemm-common`・`gemm-f32`・`gemm-f64`・`gemm-f16`・`gemm-c32`・`gemm-c64`
  （いずれも 0.19.0）: MIT（各 `Cargo.toml` 実測。§2）。上流
  [github.com/sarah-ek/gemm](https://github.com/sarah-ek/gemm/)（vcs sha
  `86102c5b712737978371ac9ef7a11982f686d7bc`）
- `matrixmultiply`・`gemm` crate の許容依存第 9 区分としての承認・実測記録は
  `docs/license-matrix.md` §8a-1・`docs/oss-comparison-harness-decision.md` を正とする
  （本 doc は依存を追加・変更しないため `Cargo.toml`／`Cargo.lock`／`deny.toml` は不変）
- `candle-core` 0.11.0: crates.io 配布物を checksum 照合のうえ scratchpad へ
  一時展開して読解し、解析後に削除した（ビルド・実行はしていない）

## 11. 解析契約の遵守宣言

- 他ライブラリ（gemm crate・candle-core）のソースコード片（関数本体・マクロ定義等）
  はコードブロックへ転記していない。実装の詳細は日本語の散文・表・数値パラメータの
  記述に `crate@版:path:line` 形式の出典を添える形で記した
  （§3〜§6 の各記述を参照）
- 依存追加・更新は行っていない（`Cargo.toml`／`Cargo.lock`／`deny.toml` 不変）
- 性能実測・tolerance・baseline・ガードレール閾値の変更は行っていない
- 既存の REJECT／保留判定（KC 再スイープ・B laneq・prefetch・P コア限定等）は
  再実験せず、§8 で判定を引用するにとどめた
- 絶対パス・内部ホスト名は本文に含めていない。gemm crate 群の出典は
  `~/.cargo/registry/src/index.crates.io-*/`（ユーザーホーム相対・ワイルドカード
  表記。crates.io 公開パッケージの標準展開先でホスト固有情報を含まない）以降を
  `crate@版:path:line` 形式で記した
