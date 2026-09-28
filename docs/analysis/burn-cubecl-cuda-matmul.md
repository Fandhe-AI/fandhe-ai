# burn 0.21／cubecl CUDA matmul の読み取り解析（stage・mma・TF32）

イシュー #2093（親 #2089「他ライブラリのコード取得・詳細解析（負けセルの原因
帰属）」Phase 2）に対応する。framework-compare スコアボードで GB10 の CUDA
GEMM N=2048 において burn 0.21（cubecl 経由）が fandhe-ai reuse の約 2 倍
速いという観測に対し、burn／cubecl／cubek のソースを**読み取りのみ**で解析
し、速度差の構造的な要因候補を列挙する。

## 1. 位置づけ・解析契約

- **読み取りのみ**: 上流のコード・シェーダ・派生物は本リポへ持ち込まない。
  引用は「結論＋出典（crate 名・版・パス・行番号・上流 URL）＋一行要約」の
  形式のみとし、上流コードブロックの逐語貼り付けは行わない
- **閉源部分の扱い**: 解析はディスパッチ層（feature 解決・`Strategy`
  選択・`adjust_dtypes` の精度降格判定）まで。cubecl IR のコード生成・
  PTX/SASS の詳細生成ロジックは本 doc のスコープ外
- **既存記録との重複回避**: 既存の REJECT 記録（Stream-K・persistent・
  split-K・3×TF32）・undetermined 事項は再実験しない。本 doc は「burn 側に
  同種の構造があるか」を突合するのみ
- **対象版**（`scripts/bench/framework-compare/Cargo.lock` の registry 解決
  で確認。2026-09-28 時点）: `burn`／`burn-cuda`／`burn-cubecl`／
  `burn-autodiff`／`burn-backend`／`burn-std` = `0.21.0`、`cubecl`／
  `cubecl-core`／`cubecl-cuda`／`cubecl-cpp`／`cubecl-common`／
  `cubecl-runtime`／`cubecl-ir` = `0.10.0`、`cubek`／`cubek-matmul`／
  `cubek-std` = `0.2.0`
- **スコープ外**（別 doc・別イシューへ）: cubecl IR 仕様の深掘り、CUDA
  Graph／persistent kernel の burn 側状況、本解析で列挙した要因候補の実測
  （Phase 3）

## 2. 結論サマリ（前提の訂正を先頭に置く）

イシュー本文は「double buffering・stage 構成が burn の速さの要因」という
仮説を挙げているが、**framework-compare の計測構成（`bench-burn`）は
autotune・fusion のいずれも無効**であり、double buffering を含む autotune
専用ルーチン群は計測経路に現れない。

再確認コマンド（本 doc 記載どおり再現可能。`--locked --offline` のためネット
ワーク取得・lock 更新を起こさない）:

```
cd scripts/bench/framework-compare
cargo tree --locked --offline -p bench-burn --no-default-features --features cuda -e features
```

出力全文（1647 行）に `autotune`・`fusion` の文字列は 1 件も現れない
（`burn-cubecl-fusion` も依存木に不在）。`bench-burn/Cargo.toml` は
`burn = { version = "=0.21.0", default-features = false, features = ["std",
"ndarray", "autodiff"] }` + `cuda = ["burn/cuda"]` であり、`autotune`
feature を有効化していない。

この結果、計測経路は次の 1 本に固定される:

```
Strategy::Auto（既定）
  → auto()（cubek-matmul-0.2.0/src/launch/strategy.rs:549-574）
    → Strategy::SimpleCyclicCmma を試行
      → 成功（GB10 は Cmma 利用可）→ SimpleCyclicCmma を採用
      → Unavailable のときのみ SimpleUnit へフォールバック
```

`SimpleAlgorithm`（`cubek-matmul-0.2.0/src/routines/simple.rs:39-40`）は
doc comment で「Plane accelerated **single stage** matmul」と明記されており、
既定ローダは `SyncFullCyclicLoading`（同期・フルロード・cyclic 順）。つまり
計測経路は **single stage・同期ロードの CMMA（WMMA フラグメント API）
＋ TF32 stage** であり、**double buffering ではない**。

要因候補は「バッファリング方式」より先に **(a) 精度クラス差**（burn は
TF32 Tensor Core、fandhe-ai の既定計測は FP32 SIMT）と **(b) 計測境界差**
（fresh／reuse の定義・readout の位置）を筆頭に置くのが、ソース根拠に基づく
誠実な帰属になる（詳細は §8）。

## 3. burn 側 dispatch フロー（受入 2）

1. **feature 解決**: `bench-burn` は `burn/cuda` feature のみを有効化する
   （`bench-burn/Cargo.toml`）。`cargo tree ... -e features` の実行結果
   （§2）で `burn-cubecl-fusion` が依存木に現れないことを確認済み
2. **型エイリアス**: `burn-cuda-0.21.0/src/lib.rs:9-13` 付近で
   `Cuda<F = f32, I = i32> = CubeBackend<CudaRuntime, F, I, u8>`
   （非 Fusion の `CubeBackend`。Fusion ラッパーを経由しない）
3. **matmul 戦略の取得**: `burn-cubecl-0.21.0/src/kernel/matmul/base.rs`
   の `MatmulStrategy::default()` は `Cube`（`Simple`/`Auto` 相当）を返し、
   `launch_matmul(&Default::default(), …)` を呼ぶ。autotune feature が
   無効なため、`burn-cubecl-0.21.0/src/kernel/matmul/tune/base.rs` の
   autotune 集合（`DoubleCyclic*`・`OrderedDouble*`・`Specialized*`・
   `SimpleTma*`・`SpecializedTma*` 等、`double_buffering_priority` を持つ
   tunable 群）は **到達しない**（feature ゲートの外側）
4. **cubek 側への委譲**: `launch_matmul` は最終的に
   `cubek_matmul::launch::strategy::Strategy::Auto`（`#[default]`。
   `cubek-matmul-0.2.0/src/launch/strategy.rs:184` 付近の `enum Strategy`
   定義）へ到達し、`auto()`（同ファイル `:549-574`）が実行される

### autotune 有効時の分岐（計測構成では未到達・参考記録）

`autotune` feature を有効化した場合にのみ、`burn-cubecl-0.21.0/src/kernel/
matmul/tune/base.rs` の tunable 群（cyclic／tilewise／hybrid／async
cyclic・strided／TMA の double buffering 系、specialized 系）が候補に入り、
形状条件に応じて `double_buffering_priority`／`should_tune_double_buffering`
相当の判定でどれを autotune するかが決まる（判定式の細部は本 doc のスコープ
外）。**本イシューの計測条件（bench-burn のビルド）ではこの経路自体に到達
しないため、double buffering は N=2048 の速さの要因候補から除外する**（§8
(e)）。

## 4. cubek／cubecl の kernel 構成（受入 1）

### 4.1 `Strategy::Auto` の実体（`cubek-matmul-0.2.0/src/launch/strategy.rs`）

`fn auto<R: Runtime>(...)`（`:549-574`）:

```
Strategy::SimpleCyclicCmma(Default::default()) を launch_ref
  Err(MatmulSetupError::Unavailable(_)) の場合のみ
    → Strategy::SimpleUnit(Default::default()) へフォールバック（.unwrap()）
  それ以外の Err は panic!
```

GB10（sm_121・arch_major=12）は Cmma（Tensor Core WMMA）を利用できるため
（§5 のアーキ条件参照）、通常は `SimpleCyclicCmma` がそのまま成功し
`SimpleUnit` へは落ちない。

### 4.2 `SimpleAlgorithm` の構成（`cubek-matmul-0.2.0/src/routines/simple.rs`）

- 型定義（`:39-48`）: doc comment「Plane accelerated single stage matmul
  with configurable readers (default to cyclic)」。デフォルト型引数は
  `LL = RL = AL = SyncFullCyclicLoading<...>`（同期・cyclic・フルロード）
- `SimpleArgs`（`:59-65`）は `tile_matmul: TileMatmulKind` と
  `multi_rows: bool` のみを持ち、`Strategy::SimpleCyclicCmma` は
  `tile_matmul = TileMatmulKind::Cmma`（`stamp_kind` 経由で `set_simple` が
  設定。`strategy.rs` 冒頭のヘルパー群）
- **stage 数**: 型が `single_stage::simple::SimpleMatmulFamily`
  （`simple.rs` の import）を使うことから、名前どおり **1 stage**
  （single-buffered）である。double buffering 用の型
  （`components::global::double_buffered` 系）は `routines/double_buffering.rs`
  にのみ現れ、`SimpleAlgorithm` からは参照されない
- **CMMA/MMA の軸**: `TileMatmulKind::Cmma`（WMMA フラグメント API。
  `cubecl-cpp-0.10.0/src/cuda/mma/cuda_compiler.rs`。`nvcuda::wmma`
  相当）と `TileMatmulKind::Mma`（インライン PTX `mma.sync` を直接発行する
  経路。`cubecl-cpp-0.10.0/src/cuda/mma/ptx_wmma_compiler.rs` 等）の 2 系統
  があり、`Strategy` enum は `*Cmma`／`*Mma` の対で両方を用意している。
  計測経路（`Strategy::Auto` → `auto()`）は常に `Cmma` 側（`SimpleCyclicCmma`）
  を試す実装になっており、`Mma`（インライン PTX）側は `auto()` からは選択
  されない

### 4.3 autotune 専用ルーチンの構造（参考・計測経路では未到達）

- `routines/double_buffering.rs`: `CyclicDoubleBufferingAlgorithm`・
  `TilewiseDoubleBufferingAlgorithm`・`HybridDoubleBufferingAlgorithm`・
  `AsyncCyclicDoubleBufferingAlgorithm`・`AsyncStridedDoubleBufferingAlgorithm`・
  `TmaDoubleBufferingAlgorithm` 等、ロード方式×タイル順の組み合わせごとに
  型が分かれる（`launch/strategy.rs` の `Strategy` enum 列挙〈`:150-165`
  付近〉に対応）
- `routines/ordered_double_buffering.rs`・`routines/specialized.rs`:
  producer/consumer 分離（specialized）・順序制御付き double buffering の
  variant
- TMA 系（`AsyncFullTmaLoading`・`SimpleTmaAlgorithm`・
  `TmaDoubleBufferingAlgorithm`）は `arch_version >= 90`（Hopper 以降）
  相当の `Tma` feature 登録が前提（`cubecl-cuda-0.10.0/src/runtime.rs` の
  `arch_version >= 90` 節。§5 参照）であり、GB10（Blackwell 系・
  arch_major=12）でも feature 自体は登録されうるが、**autotune 経由でしか
  選択されない**ため計測経路には現れない
- いずれも `Strategy::Auto` の `auto()` 本体（§4.1）からは直接参照されて
  おらず、autotune tunable として burn 側（`tune/base.rs`）からのみ登録
  される

## 5. TF32 降格の実装と自作との対比（受入 3）

### 5.1 `adjust_dtypes` の降格条件

`cubek-matmul-0.2.0/src/definition/blueprint.rs:84-110`
（`pub fn adjust_dtypes<R: Runtime>(client, dtypes, requires_accelerator)`）:

- `requires_accelerator` が真、かつ `lhs_global == rhs_global == f32` かつ
  `client.properties().supports_type(tf32_dtype)` が真のとき、
  `lhs_stage`／`rhs_stage`／`lhs_register`／`rhs_register` を全て TF32 へ
  書き換える（f32 → f16 の flex32 経路は別条件で排他）
- `requires_accelerator` は `TileMatmulKind::requires_accelerator()`
  （`cubek-matmul-0.2.0/src/components/tile.rs:102-109`）が返す値で、
  `Cmma`・`Mma` は `true`、`Register`・`PlaneVec`・`Interleaved` は
  `false`。`SimpleCyclicCmma` は `Cmma` を使うため常に `true`
- 呼び出し元は `routines/simple.rs:203`・`routines/specialized.rs:180`・
  `routines/selector/plane.rs:50`・`routines/interleaved.rs:236` の 4 箇所
  （いずれも `tile_matmul.requires_accelerator()` をそのまま渡す）

### 5.2 `supports_type(tf32)` のアーキ条件

TF32 型自体は `cubecl-cuda-0.10.0/src/runtime.rs:176`
（`device_props.register_type_usage(...TF32..., TypeUsage::Conversion)`）で
**アーキ条件なしに常時登録**される。しかし TF32 を使った MMA 演算の
可否は別軸で、`cubecl-cpp-0.10.0/src/cuda/mma/cuda_compiler.rs:127-136`
（WMMA フラグメント API の `supported_wmma_combinations`）が
`if arch.get_version() >= 80 { result.push(MmaConfig { a_type: TF32,
b_type: TF32, cd_type: F32, m: 16, n: 16, k: 8 }); }` という条件を持つ
（**sm_80（Ampere）以上限定**）。GB10 は arch_major=12（sm_121 系。
`cubecl-cuda-0.10.0/src/runtime.rs:79-83` の `arch_version` 算出ロジック）
のため `>= 80` を満たし、TF32 の Cmma 経路が利用可能になる。`Strategy::Auto`
が `SimpleCyclicCmma`（Cmma・`requires_accelerator() == true`）を選ぶ限り、
入力が f32 であれば常にこの経路を通って TF32 へ降格される。**ユーザー側
から TF32 降格を無効化する公開スイッチは `cubek-matmul`／`burn-cubecl` の
読み取り範囲では確認できなかった**（§5.1 のとおり `TileMatmulKind::Mma`
も `requires_accelerator() == true` であり `Strategy::SimpleCyclicMma`
（Mma・PTX 直接発行）へ切り替えても降格条件は変わらず TF32 降格を避け
られない。`TileMatmulKind::Register`（非 Tensor Core・
`requires_accelerator() == false`）へ明示的に切り替えれば TF32 降格を
避けられるが、`Strategy::Auto`／`MatmulStrategy::default()` の既定経路
にはその選択肢がない）

### 5.3 自作 `CudaGemmPrecision` との対比

| 観点 | burn／cubek（計測経路） | fandhe-ai（`crates/backend-cuda/src/precision.rs`） |
|---|---|---|
| 既定モード | TF32 降格（`Strategy::Auto` が常に Cmma を試す） | `Fp32Strict`（FP32 厳密。既定不変） |
| 降格の可視性 | 呼び出し元から見えない暗黙降格（`bench-burn` は `precision_class(device)` で `cuda` なら `PrecisionClass::Tf32` を **参照値側** に反映しているだけで、burn 側の挙動そのものは変更していない） | `fandhe_ai::set_cuda_gemm_precision` 経由の明示 opt-in（3 値: `Fp32Strict`／`Tf32`／`Tf32x3`） |
| 失敗時の挙動 | Cmma が `Unavailable` の場合のみ `SimpleUnit`（非 Tensor Core）へフォールバック（f32 精度は保たれるが速度特性が変わる） | fail-closed（型付きエラーをそのまま伝播。黙示フォールバックなし。`precision.rs` 冒頭コメント） |
| 数値一致契約 | 記述なし（cubek/burn 側に REQ-2 相当の複合判定は無い） | REQ-2 統一複合判定（相対誤差 1e-3 未満 または 絶対誤差 1e-5 未満）を実機実測で確認済み（`docs/perf/cuda-tensor-core-tolerance-*.md`） |

`bench-burn/src/main.rs:82-87`（`fn precision_class(device: &str)
-> PrecisionClass`）は `device == "cuda"` なら無条件で
`PrecisionClass::Tf32` を返す。これは burn 側の実行を変える呼び出しでは
なく、**参照実装（`GemmReference`）の許容誤差計算に供給する精度クラスの
ラベル**であり、「burn の CUDA 経路は常に TF32 で走る」という事実（§5.1・
§5.2 のソース根拠）をベンチハーネス側が追認している記述と読める。

## 6. 自作 CUDA GEMM 経路との algorithm 差分

イシュー本文が挙げる `crates/backend-cuda/src/gemm.rs::dispatch_auto` は
**CUDA 側のソースに存在しない**（`dispatch_auto` という命名は Metal 側
〈`crates/backend-metal/`〉の命名であり、混同と判断する）。CUDA 側の実際の
対応物は次のとおり:

- `crates/backend-cuda/src/ops.rs:2395` `CudaBackendOps::gemm`:
  `crate::precision::gemm_precision()` の `match` で分岐（`:2396-2469`
  付近）
  - `Fp32Strict`（既定）: 早期 return し `gemm_fp32_strict_impl`
    （`:180-184` 周辺のコメントが「常に FP32 厳密で計算する」と明記）
    → `gemm.rs:3145` `CudaGemm::run_tiled_f32` → `gemm.rs:2988`
    `select_tiled_f32_kernel`（classic／cp.async pipeline 3 stage／
    128×64 タイルの選択ロジック）
  - `Tf32`: `gemm.rs:4550` `CudaGemm::run_wmma_tf32`
    （staged→opt→basic の 3 段選択。`:4675` `run_wmma_tf32_opt_kernel`・
    `:4731` `run_wmma_tf32_staged_kernel`）
  - `Tf32x3`: `crate::gemm_mma_tf32x3::CudaMmaTf32x3Gemm::run_tf32x3`
    （3×TF32・split-single 法。`docs/cuda-tf32x3-split-single-decision.md`）

burn 計測経路との構造差分（表）:

| 観点 | burn（`SimpleCyclicCmma`） | fandhe-ai `Fp32Strict`（既定計測条件） | fandhe-ai `Tf32`（opt-in） |
|---|---|---|---|
| 精度クラス | TF32（Tensor Core） | FP32（SIMT） | TF32（Tensor Core） |
| stage 数 | 1（single stage） | 3（cp.async pipeline。`select_tiled_f32_kernel`） | 3 段選択（staged→opt→basic） |
| タイル | plane 単位・cyclic ロード | 128×64（cp.async pipeline 経路） | WMMA staged 構成 |
| ロード方式 | 同期・フルロード cyclic | cp.async（非同期） | 実装依存（3 経路） |

FP32 SIMT（cp.async pipeline 3 stage）と TF32 Tensor Core（Cmma・single
stage）は演算器自体が異なるため、stage 数の単純比較では速度差を説明でき
ない。§8 で精度クラス差を筆頭候補とする根拠はここにある。

## 7. 既存 REJECT との差別化（受入 4）

`cubek-matmul-0.2.0/src`・`cubecl-cuda-0.10.0/src`・
`burn-cubecl-0.21.0/src` に対する grep 結果（2026-09-28 実行）:

| パターン | ヒット | 判定 |
|---|---|---|
| `stream.?k`（大小無視） | `cubecl-cuda-0.10.0/src/compute/stream.rs`・`server.rs` の `StreamKind::NonBlocking`（CUDA stream の種別。Stream-K GEMM アルゴリズムとは無関係） | **Stream-K 実装なし**（誤検出のみ） |
| `persistent` | `burn-cubecl-0.21.0/src/backend.rs` の `memory_persistent_allocations`／`memory_persistent_allocation`（メモリ確保の永続化 API。persistent kernel スケジューリングとは無関係） | **persistent kernel 実装なし**（誤検出のみ） |
| `split.?k` | ヒットなし | **split-K 実装なし** |
| `3.?x.?tf32` | ヒットなし | **3×TF32 実装なし** |
| `flex32` | `cubek-matmul-0.2.0/src/{components/tile.rs, definition/{blueprint.rs,spec.rs}}`・`cubecl-cuda-0.10.0/src/compute/{communication.rs,server.rs}`・`burn-cubecl-0.21.0/src/{element.rs, kernel/cast/base.rs}` | cubecl 独自の縮小精度型 `flex32`（f16 相当。`adjust_dtypes` の f16 フォールバック分岐 §5.1）であり、fandhe-ai の 3×TF32 とは無関係の別概念 |
| `tma` | 37 ファイル | `AsyncFullTmaLoading`・`SimpleTmaAlgorithm`・`TmaDoubleBufferingAlgorithm` 等、TMA ロードストラテジーが**実装として存在**するが、§4.3 のとおり autotune 経由でのみ選択され `Strategy::Auto` の既定経路には現れない |

既存記録との突合:

- **Stream-K**（`docs/cuda-streamk-decision.md`・
  `docs/perf/cuda-gemm-tiled-pipeline-streamk.md`）: fandhe-ai 側も REJECT
  済み。burn／cubek 側にも実装がなく、両者とも「未実装」で条件は揃っている
- **persistent kernel**（`docs/perf/cuda-gemm-tiled-pipeline-persistent.md`）:
  同上。両者とも未実装
- **TMA**（`docs/backend-cuda-tma-gemm-load-design.md`・
  `docs/perf/logs/cuda-tma-stage1-1975/`）: fandhe-ai 側は Stage 1 相当を
  調査済み（詳細は当該 doc）。cubek 側は TMA ロード自体を実装済みだが
  autotune 経由限定であり、**計測条件（burn 既定・autotune 無効）では
  fandhe-ai 側と同様に「未使用」で条件が揃う**。「実装しない理由」ではなく
  「autotune 集合の一部だが本比較の計測条件では未到達」という位置づけの
  違いに注意（fandhe-ai は実装判断そのものが REJECT／調査中、burn 側は
  実装は存在するが計測条件が選択しない）
- **3×TF32**（`docs/cuda-tf32x3-split-single-decision.md`・
  `docs/perf/cuda-tensor-core-tolerance-tf32x3-gb10.md`）: fandhe-ai は
  `Tf32x3` として実装・opt-in 提供済み（§6）。burn／cubek 側には同種の
  hi/lo 分割・複数 `mma.sync` 累積による f32 相当精度の Tensor Core 近似は
  見つからなかった（grep 結果「ヒットなし」は不在の直接証拠ではなく本 grep
  パターンでの不検出に留まる点に注意。「実装しない理由」は推定である）

## 8. N=2048 で速い要因の構造的候補（受入 5）

優先度付きで列挙する。実測による確定は Phase 3 が引き継ぐ。

### (a) 精度クラス差（優先度最高）

burn の計測経路は §5 のとおり常に TF32 Tensor Core（`SimpleCyclicCmma`）を
使う。一方 `docs/perf/logs/framework-compare-precision-class-remeasure-1988/
aggregate.md` のスコアボードでは、fandhe-ai 側の GEMM ベンチは reuse
（`bench-fandhe` の `--tf32` 既定 `false` = FP32 厳密。§6 引用箇所）で計測
されている（burn 側だけが強制 TF32、fandhe-ai 側は明示 opt-in しない限り
FP32、という非対称な比較条件）。TF32 Tensor Core と FP32 SIMT は演算器が
異なるため、この非対称性自体が速度差の主要因になりうる。fandhe-ai の TF32
opt-in 経路（`run_wmma_tf32`）は既に実装・実機実測済み
（`docs/cuda-tf32-optin-api-decision.md`・
`docs/perf/cuda-tensor-core-tolerance-*.md`）であり、Phase 3 では同条件
（fandhe-ai `Tf32` opt-in 有効）での再計測が要因切り分けの第一候補になる

### (b) 計測境界差

`bench-burn/src/main.rs:117-125` 付近の計測クロージャは
`Instant::now()` の直後に `a.clone().matmul(b.clone())` と `readout(c)`
（host 同期・要素読み出し）を計測窓内に含める「fresh」計測である
（clone・matmul・readout を毎回計測窓内で行う）。fandhe-ai 側は reuse
方式（詳細な計測境界の定義は `docs/perf/cuda-gemm-candle-gate-
remeasurement.md` §4.3／§16、host_copy・checksum 診断コストの扱いは
#1182 を参照）であり、計測窓の定義そのものが異なる可能性がある。
`docs/perf/logs/framework-compare-precision-class-remeasure-1988/
aggregate.md` の「同一 run 内比」表（burn fresh median_s ÷ fandhe-ai reuse
median_s）は N=2048 で中央値 0.4811（burn が高速）であり、これは fresh
（burn）と reuse（fandhe-ai）という異なる計測境界同士の比較値であることに
注意する

### (c) kernel 構成差

§6 のとおり、TF32 Tensor Core・single stage・plane 単位・同期 cyclic ロード
（burn）と、FP32 SIMT・cp.async pipeline 3 stage・128×64 タイル
（fandhe-ai 既定）は演算器・タイル形状・ロード方式のいずれも異なる。同一
精度クラス（TF32 同士）で揃えたうえでの構成差の寄与は、(a) を切り分けた
後でなければ評価できない

### (d) burn ≈ candle という同等性が示す示唆

同スコアボードの「同一 run 内比」表では N=2048 で candle 中央値 0.5075、
burn 中央値 0.4811 と近接している。candle CUDA GEMM は cuBLAS 経由
（別解析対象）であり、burn（cubek 自前カーネル）と cuBLAS がほぼ同水準に
達しているという事実は、「Tensor Core（TF32）を使えばこの水準に到達
できる」という解釈を支持する一方、cubek 独自の kernel 構成（single stage
等）が cuBLAS 並みの効率を持つという主張までは裏付けない（cuBLAS は
Tensor Core 利用の参照点であって、cubek の実装効率そのものの証拠ではない）

### (e) autotune 専用ルーチン（double buffering 等）は要因候補から除外

§2・§3・§4.3 の grep 実測（`autotune`／`fusion` 文字列が feature tree に
不在）により、double buffering を含む autotune 集合は計測経路に到達しない
ことをソースレベルで確認済みである。よってイシュー原文の仮説（stage・
double buffering が速さの要因）は、**現行の計測条件においては前提が成立
しない**として要因候補から除外する

## 9. ライセンス記録（受入 6）

registry 解決済み crate の `Cargo.toml` `license` フィールドと同梱
`LICENSE-*` の有無（`~/.cargo/registry/src/index.crates.io-*/` を実測。
ローカル絶対パスは伏せ、crate 名・版のみ記載）:

| crate | 版 | `license` | `LICENSE-*` 同梱 |
|---|---|---|---|
| `burn` | 0.21.0 | MIT OR Apache-2.0 | あり |
| `burn-cuda` | 0.21.0 | MIT OR Apache-2.0 | なし |
| `burn-cubecl` | 0.21.0 | MIT OR Apache-2.0 | あり |
| `burn-autodiff` | 0.21.0 | MIT OR Apache-2.0 | あり |
| `burn-backend` | 0.21.0 | MIT OR Apache-2.0 | なし |
| `burn-std` | 0.21.0 | MIT OR Apache-2.0 | あり |
| `cubecl` | 0.10.0 | MIT OR Apache-2.0 | あり |
| `cubecl-core` | 0.10.0 | MIT OR Apache-2.0 | あり |
| `cubecl-cuda` | 0.10.0 | MIT OR Apache-2.0 | あり |
| `cubecl-cpp` | 0.10.0 | MIT OR Apache-2.0 | なし |
| `cubecl-common` | 0.10.0 | MIT OR Apache-2.0 | あり |
| `cubecl-runtime` | 0.10.0 | MIT OR Apache-2.0 | あり |
| `cubecl-ir` | 0.10.0 | MIT OR Apache-2.0 | あり |
| `cubek` | 0.2.0 | MIT OR Apache-2.0 | なし |
| `cubek-matmul` | 0.2.0 | MIT OR Apache-2.0 | なし |
| `cubek-std` | 0.2.0 | MIT OR Apache-2.0 | なし |

上流ルートの LICENSE ファイル存在確認（`gh api repos/tracel-ai/<repo>/
contents` 読み取り。2026-09-28 実行）: `tracel-ai/burn`・`tracel-ai/cubecl`・
`tracel-ai/cubek` のいずれもリポジトリルートに `LICENSE-APACHE`・
`LICENSE-MIT` が存在することを確認した（一部 crate は個別ディレクトリに
`LICENSE-*` を同梱していないが、上流リポジトリルートのライセンスが適用
される MIT OR Apache-2.0 のデュアルライセンス構成である）。参照タグの存在
確認（`gh api repos/tracel-ai/<repo>/git/refs/tags`）: `v0.21.0`
（burn）・`v0.10.0`（cubecl）・`v0.2.0`（cubek）のいずれも存在を確認した。

**NOTICE 義務**: 本 doc・本 PR は上流コードの引用・持ち込みを行わず、
結論と出典（crate 名・版・パス・行番号・上流 URL）のみを記録するため、
著作権表示・ライセンス全文の同梱義務は発生しない。`docs/license-matrix.md`
8b 節に burn 0.21.0 は既に記録済み（framework-compare 第 9 区分の適用範囲
拡張）であり、本イシューは新規の依存追加を伴わないため `docs/license-
matrix.md` 自体の変更は行わない。

## 10. 出典一覧・スコープ外・Phase 3 への引き継ぎ

### 出典

- スコアボード: `docs/perf/logs/framework-compare-precision-class-
  remeasure-1988/`（`README.md`・`aggregate.md`・JSONL 群）
- `docs/candle-parity-precision-class-decision.md`
- `docs/cuda-tf32-optin-api-decision.md`
- `docs/cuda-tf32x3-split-single-decision.md`
- `docs/cuda-streamk-decision.md`
- `docs/perf/cuda-gemm-tiled-pipeline-streamk.md`
- `docs/perf/cuda-gemm-tiled-pipeline-persistent.md`
- `docs/backend-cuda-tma-gemm-load-design.md`・
  `docs/perf/logs/cuda-tma-stage1-1975/`
- `docs/perf/cuda-tensor-core-tolerance-tf32x3-gb10.md`
- `docs/perf/cuda-gemm-candle-gate-remeasurement.md`
- リポジトリ内ソース: `crates/backend-cuda/src/{ops.rs, gemm.rs,
  precision.rs, gemm_mma_tf32x3.rs}`、`scripts/bench/framework-compare/
  bench-burn/src/main.rs`、`scripts/bench/framework-compare/bench-fandhe/
  src/main.rs`
- registry ソース（版は §1 記載のとおり）: `burn-cuda`／`burn-cubecl`／
  `cubek-matmul`／`cubecl-cuda`／`cubecl-cpp` の各該当ファイル（本文中に
  パス・行番号を明記）
- 上流リポジトリ: `github.com/tracel-ai/burn`（タグ `v0.21.0`）・
  `github.com/tracel-ai/cubecl`（タグ `v0.10.0`）・
  `github.com/tracel-ai/cubek`（タグ `v0.2.0`）

### スコープ外（本 doc では扱わない）

- cubecl IR（中間表現）のコード生成仕様の深掘り
- CUDA Graph／persistent kernel の burn 側状況（burn-cubecl 全体の対応可否
  調査は別 doc の対象）
- 性能実測そのもの（本 doc は静的解析のみ。実測は Phase 3）

### 受入条件との対応

| # | 受入条件 | 対応節 |
|---|---|---|
| 1 | cubecl／cubek 側 kernel 構成のロジック抽出 | §4 |
| 2 | burn 側 dispatch フロー | §3 |
| 3 | TF32 降格の実装と自作との対比 | §5 |
| 4 | 既存 REJECT との差別化 | §7 |
| 5 | N=2048 で速い要因の構造的候補 | §8 |
| 6 | ライセンス記録 | §9 |

### Phase 3 への引き継ぎ

- (a) fandhe-ai `Tf32` opt-in 有効時の N=2048 再計測（精度クラスを揃えた
  比較）
- (b) fresh／reuse の計測境界を揃えた比較（burn 側 readout 位置に合わせた
  fandhe-ai 側の計測構成、または逆）
- (c) TF32 精度クラスを揃えたうえでの kernel 構成差（stage 数・タイル・
  ロード方式）の寄与評価
