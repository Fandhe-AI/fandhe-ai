# Metal GEMM half フラグメント／f32 累算候補（`gemm_simdgroup_tiled_hfrag`）

イシュー #1369（親 #1368・E9）の実装記録。`crates/backend-metal/src/shaders/gemm.metal`
へ half 入力フラグメント（`simdgroup_half8x8`）＋ float アキュムレータ
（`simdgroup_float8x8`）構成の候補カーネル `gemm_simdgroup_tiled_hfrag` を
追加し、REQ-2 の形状別判定方式で全形状 × 転置 4 パターンの parity を確認
した記録。

## §0 目的・範囲

- 目的: E9（親 #1368）の狙い「half MMA による ALU スループット向上」を、
  f16 専用ストレージへ移行せず f32 API 形状（`device const float*`/
  `device float*` 入出力）のまま検証するための候補カーネルを用意し、
  正しさ（parity）を確認する。
- 範囲: カーネル追加・Rust 側入口（候補評価専用・非結線）・parity 記録に
  限る。純カーネル時間の比較・採否判断は兄弟イシュー #1370 のスコープ。
- 本番結線は行わない（§2「非結線の理由」参照）。

## §1 現行構成の確認結果

`gemm_simdgroup_tiled_f16`（本番 f16 経路。イシュー #796/#797）は既に
`typedef simdgroup_half8x8 MM_T;`（A/B フラグメント）・
`typedef simdgroup_float8x8 ACC_T;`（アキュムレータ）で構成されている
（イシュー #380 の実機 spike で half 統一から f32 累算へ変更済み。
`gemm.metal` 冒頭「累算精度契約」コメント参照）。この事実は
`tests/shader_source_evidence.rs::gemm_simdgroup_tiled_f16_source_uses_half_fragments_and_f32_accumulator`
（本イシューで追加）が機械的に固定する。

したがって親イシュー #1368 の前提「f16 経路を half 入力＋float 累積へ
切り替える」は**本番 f16 経路で既に実現済み**であり、f16 経路
（`dispatch_f16_auto_unverified` → `gemm_simdgroup_tiled_f16`）に「切り替え
候補」を新設する余地はない。f16 経路は `TransposePattern::Nn` 固定
（転置入口なし）でもあり、「全形状 × 転置」の parity 確認は f16 経路では
成立しない。

本イシューはこの事実を踏まえ、**f32 ストレージ（`device const float*`
入出力）の GEMM に対して、threadgroup タイルへの協調ロード時に f32→half
変換し、フラグメントを `simdgroup_half8x8`・アキュムレータを
`simdgroup_float8x8` とする候補カーネル**（`gemm_simdgroup_tiled_hfrag`。
"half fragment" の意）を新設する解釈で実装した。

## §2 候補設計

### §2.1 MSL（`crates/backend-metal/src/shaders/gemm.metal`）

- `gemm_simdgroup_tiled_f16` の直後（ファイル末尾）に
  `kernel void gemm_simdgroup_tiled_hfrag(` を追加した。
- 引数・バッファ束縛は `gemm_simdgroup_tiled`（f32 版）と同一
  （`a`/`b`/`c` は `device const float*`/`device float*`。buffer 0〜2）に、
  `GemmStrides`（buffer 4）・`TileClassRegion`（buffer 5。シグネチャ対称性
  のためだけに受け取り、本体では読まない）を加える。
- function constant は既存の index を再利用し新規追加していない
  （`BM`/`BN`/`BK`/`WM`/`WN`〈0〜4〉・`USE_TGP_STAGING`〈5。ただし本カーネル
  は参照しない〉・`TGP_PAD`〈6〉・`SWIZZLE_ENABLED`〈7〉・`TRANS_A`/`TRANS_B`
  〈9/10〉）。16 個の function constant 総数は不変（`tests/shader_source_evidence.rs::gemm_simdgroup_tiled_hfrag_introduces_no_new_function_constants`
  が固定）。
- **スコープ境界**: staged 経路（協調ロード）のみを実装する。direct-load
  （`USE_TGP_STAGING=false`）は実装せず、`MetalGemm::pipeline_for_tile_hfrag`
  が `crate::tile::fallback_chain` 巡回中に `staged=false` の候補
  （`TileConfig::SINGLE_SIMDGROUP_8X8` を含む）を fail-closed に拒否する
  ことが唯一の防御（本カーネル自体は `USE_TGP_STAGING` を一切参照しない）。
  `TILE_CLASS`/`FRAG_LOAD_DEVICE_HOISTED`/`FRAG_LOAD_KSTEPS`/
  `COOP_LOAD_LAYOUT`/`UNROLL_ACC_ENABLED` も一切参照しない（no-op 契約。
  `MetalGemm::pipeline_for_tile_hfrag` が常に既定値／`0`／`false` を渡す）。
- **協調ロード**: device 側から f32 の `float4`（128bit・4 要素）を
  ベクトルロードし（アラインメント成立根拠は f32 版 `gemm_simdgroup_tiled`
  と同一。`TileConfig::validate` の 8 整除検査が `VEC_WIDTH=4` 整除を間接
  包含）、`half4(float4)`（要素ごと round-to-nearest-even 変換）で half4 へ
  変換して threadgroup へ 1 回で store する（f16 版のような 2 分割 store
  は不要 — f16 版は half データをビットコピーするため 8 要素・2 分割
  だったが、hfrag は f32→half の値変換を伴う 4 要素・単一 half4 store）。
  境界グループは要素単位のスカラー読み出し＋0 埋めへフォールバックする
  （REQ-8）。
- **K ループ**: `MM_T`（`simdgroup_half8x8`）フラグメント配列
  （`MAX_ACC=8`）を `simdgroup_load`（転置時は `transpose_matrix=true`
  相当の第 4 引数 `true`）でロードし、
  `simdgroup_multiply_accumulate(ACC_T&, MM_T, MM_T, ACC_T)` で f32 累算
  する。`FRAG_LOAD_KSTEPS` の 16 幅一括ロード方式は実装しない（既定の
  8 幅刻み単一フラグメント処理のみ）。
- **エピローグ**: 出力が `device float*` のため `gemm_simdgroup_tiled_f16`
  のような f32 staging 領域は不要。`simdgroup_store` を f32 版
  `gemm_simdgroup_tiled` と同じ境界チェック付きで直接 `c` へ書く。
- **手動境界チェック（REQ-8）**: ブロック原点の早期 return・協調ロードの
  group/elem in-bounds ＋ 0 埋め・エピローグの要素単位境界チェックを
  f32/f16 版と同じ設計で維持する（省略しない）。

### §2.2 Rust（`crates/backend-metal/src/tile.rs`）

- `TileConfig::shared_mem_bytes_hfrag_for(pattern) -> u32`
  （`staged=false` は常に `0`。staged はタイル要素数 × 2 バイト〈half〉）
  を追加した。内部実装は `shared_mem_bytes_for`（f32 版）と要素数計算式を
  共有する `tiled_elem_bytes_for_pad(..., elem_bytes)` へリファクタし、
  `shared_mem_bytes_for` を `elem_bytes=4`・`shared_mem_bytes_hfrag_for` を
  `elem_bytes=2` で呼ぶ形にした（`tiled_bytes_for_pad` は薄いラッパーとして
  維持し既存呼び出し元は無変更）。
- 単体テスト: 全 `CANDIDATES`（staged）× 4 パターンで
  `shared_mem_bytes_hfrag_for == shared_mem_bytes_for / 2` を固定・
  `staged=false` で 0・オーバーフロー時 `u32::MAX` 飽和を確認する。

### §2.3 Rust（`crates/backend-metal/src/gemm.rs`）

- `MetalGemm::tiled_hfrag_cache: Mutex<HashMap<(TileConfig, TransposePattern), Retained<MtlPipeline>>>`
  を追加（f32 版・f16 版とは独立したキャッシュ）。
- `fn pipeline_for_tile_hfrag(&self, ctx, cfg, pattern)`: `pipeline_for_tile_f16`
  と同型のフォールバック戦略だが、`candidate.staged == false` を明示的に
  スキップする（`SINGLE_SIMDGROUP_8X8` へサイレントフォールバックしない。
  fail-closed）。デバイス上限検査は `validate`（f32 単位）に加えて
  `shared_mem_bytes_hfrag_for(pattern) <= max_shared_mem_bytes` でも行う。
- `fn encode_dispatch_tiled_hfrag(...)`: `encode_dispatch_tiled_f16` と同じ
  buffer 結線パターンに `GemmStrides`（buffer 4）・
  `TileClassRegion::full_grid`（buffer 5。常に dispatch grid 全体）を追加。
  `setThreadgroupMemoryLength` は `shared_mem_bytes_hfrag_for(pattern).max(16)`。
- `pub fn dispatch_hfrag_tiled_unverified(&self, ctx, a, b, m, n, k, pattern, cfg) -> Result<(Vec<f32>, TileConfig), MetalError>`
  （`#[doc(hidden)]`。候補評価専用・本番非結線を明示）: `pad8`・
  `GemmStrides` 構築（`lda`/`ldb` は NN: `k_eff`/`n_eff`・NT: `k_eff`/`k_eff`・
  TN: `m_eff`/`n_eff`・TT: `m_eff`/`k_eff`）・パディング・ディスパッチ・
  readback を一括で行う。`a`/`b` はパターンに応じた行優先ストレージ
  （転置時は `[k,m]`/`[n,k]`）を渡す契約。
- `#[cfg(test)] pub(crate) fn diag_encode_tiled_hfrag_nn(...)`: `diag_encode_tiled_nn`
  と同型の 1 バッチ・1 ラベル診断入口（#1370 が純カーネル時間計測に使う
  想定）。

### §2.4 非結線の理由

`gemm_simdgroup_tiled_hfrag` は入力を half（RTE）へ丸めるため f32 参照との
厳密ゼロ fail は一般に成立しない（CUDA TF32 経路と同じ性質。§4/§5 の実測
参照）。REQ-2 は「非 Tensor Core 経路（f32 FMA）は統一複合判定をそのまま
適用（変更なし）」と規定し、精度を落とす候補を既定経路（`tile::select`／
`dispatch_auto`／`MetalBackendOps::gemm`）へ結線することは本イシューの
承認範囲外。将来の本番利用は CUDA `set_cuda_tf32_gemm_enabled`
（`docs/cuda-tf32-optin-api-decision.md`。既定 OFF・fail-closed）と同型の
opt-in 公開 API 設計＋ユーザー承認が前提（§8 スコープ外）。

## §3 自己検証（実装済みテスト）

- `crates/backend-metal/tests/shader_source_evidence.rs`: 45 テスト green
  （f16 の half/float 構成固定・hfrag の half フラグメント／f32 累算／
  f32 入出力／行列演算ユニット命令・REQ-8 境界チェック・TRANS_A/TRANS_B
  分岐・実験的ゲート非参照・function constant 総数不変の 7 テストを新規
  追加）。
- `crates/backend-metal/src/tile.rs`（Linux 実行可）: `shared_mem_bytes_hfrag_for`
  系 3 テスト green。
- `crates/backend-metal/src/gemm.rs`（実機 `#[ignore]`）:
  `dispatch_hfrag_tiled_unverified_nn_matches_cpu_reference_small_shape`・
  `diag_encode_tiled_hfrag_nn_matches_cpu_reference_small_shape`・
  `all_staged_candidates_match_hfrag_cpu_reference_512_nn`（全 staged
  `CANDIDATES` を 512³ NN で総当たり）の 3 テストが M4 Max 実機で green。
- `crates/backend-metal/tests/gemm_hfrag_parity.rs`（新規・実機 `#[ignore]`）:
  §4 参照。

## §4 parity 結果（M4 Max 実機実測。2026-09-07）

**正しさゲート（丸め済み入力参照との統一複合判定）**: 全ケース
（正方 64/128/512/1024・端数 3 形状・縦長／横長／K 末尾・全 staged
CANDIDATES × 512³ NN・全 4 転置パターン）で **PASS**（`assert_parity` が
panic せず完走。生ログ: `docs/perf/logs/metal-gemm-hfrag-parity-1369/`）。

**REQ-2 形状別判定用の実測記録（丸めなし f32 入力参照。ゲートしない）**:
全ケースで `strict_exact=false`（`fail_count > 0`）。half 丸めを経る本
カーネルの設計上、丸めなし参照に対する厳密ゼロ fail は成立しない
（§2.4 参照）。代表値（`mean_abs_diff`／`max_abs_diff`／`max_rel_err`。
`fail_count/total`）:

| 形状 | pattern | fail_count/total | mean_abs_diff | max_abs_diff | max_rel_err |
|------|---------|-------------------|----------------|---------------|--------------|
| 64³ | Nn | 643/4096 | 5.518e-4 | 3.674e-3 | 1.744e-1 |
| 128³ | Nn | 2730/16384 | 7.877e-4 | 4.051e-3 | 1.822e0 |
| 512³ | Nn | 42566/262144 | 1.567e-3 | 9.197e-3 | 1.867e0 |
| 1024³ | Nn | 170251/1048576 | 2.220e-3 | 1.353e-2 | 1.948e0 |
| 60×68×36（ragged） | Nn | 682/4080 | 4.140e-4 | 2.109e-3 | 4.771e-1 |
| 2048×256×512（tall） | Nn | 85433/524288 | 1.571e-3 | 9.789e-3 | 1.999e0 |
| 256×2048×512（wide） | Nn | 85179/524288 | 1.572e-3 | 9.963e-3 | 1.973e0 |
| 96×96×40（k_tail） | Nn | 1515/9216 | 4.438e-4 | 2.697e-3 | 4.640e-1 |

全転置パターン（Nt/Tn/Tt）は同一形状の Nn と同オーダーの値（生ログ参照。
`max_rel_err` はゼロに近い真値要素での相対誤差の性質上 1 を超えることが
あるが、正しさゲート自体は丸め済み入力参照に対して統一複合判定
〈相対誤差 1e-3 未満または絶対誤差 1e-5 未満〉で PASS している）。

4096³ は本イシューでは実施していない（スカラー CPU 参照の実行時間が
長大なため。実施の要否は #1370 または後続イシューの判断に委ねる）。

**厳密判定成立形状**: なし（丸めなし参照に対しては全形状で `fail_count > 0`。
§2.4 の設計上の帰結）。

## §5 承認待ちベースライン候補

REQ-2 の非後退ベースライン行（`crates/backend-metal/tests/common/parity_baseline.rs`
相当の新設・CUDA 側 `ParityBaseline` と同型の fail-closed 非後退検査化）は
**本イシューでは行わない**。実測値のみを§4 に記録し、ゲート化は人間承認
必須（`.claude/rules/coding-rust.md`「TF32/f16 Tensor Core 経路の parity
テスト判定方式」）。承認が得られた場合、baseline 行の追加候補は§4 の表
（`fail_count`/`total`・`mean_abs_diff`/`max_abs_diff`/`max_rel_err`）を
そのまま初期 ceiling 値とすることを想定する。

## §6 env_info

`docs/perf/logs/metal-gemm-hfrag-parity-1369/env_info.txt` 参照（macOS
26.6.2・rustc 1.96.0・Apple M4 Max・実行前後の `uptime`）。内部ホスト名は
含めない。

## §7 #1370 への引き継ぎ

- 純カーネル時間の 5 回中央値比較・採否判断は #1370 のスコープ。
- 入口は `MetalGemm::diag_encode_tiled_hfrag_nn`（`#[cfg(test)] pub(crate)`。
  `diag_encode_tiled_nn` と同型の 1 バッチ・1 ラベル計測境界）。
- 比較対象は本番選択構成（`tile::select` が選ぶ `TileConfig`）系列を
  想定（`docs/perf/cuda-gemm-reuse-phase-breakdown.md` 等の既存フェーズ
  分解ハーネスと同じ設計判断）。
- **実施済み → §9**（イシュー #1370）。

## §8 スコープ外

- **純カーネル時間の 5 回中央値比較・採否判断**: #1370。
- **本番結線**: §2.4 参照。opt-in 公開 API の設計とユーザー承認が前提。
- **REQ-2 非後退ベースライン行の追加・ゲート化**: §5 参照。人間承認必須。
- **direct-load（非 staged）経路・`SINGLE_SIMDGROUP_8X8` フォールバック**:
  候補は staged 構成限定（§2.1）。
- **f16 経路（`gemm_simdgroup_tiled_f16`）の追加最適化**: E9 前提は既に
  満たされているため対象外（§1）。
- **4096³ の全パターン parity**: スカラー参照の実行時間の都合で未実施
  （§4）。

## §9 純カーネル時間比較・採否判断（イシュー #1370）

### §9.0 目的・範囲

hfrag 候補（§2）の純カーネル専有時間（GPU タイムスタンプ。
`kernel_gpu`。イシュー #1276）を M4 Max 実機で 5 プロセス起動・5 回
中央値（20 warmup + 20 測定）で比較し、**opt-in 候補として前進させる
価値があるか**を判定する。範囲は診断テスト追加・実測・判定記録に限り、
`tile.rs`／`gemm.rs`／`shaders/gemm.metal` の本番経路は無変更のまま
（§2.4 のとおり本番結線自体は opt-in 公開 API 設計＋ユーザー承認が
前提のため、本イシューはその設計・承認を行わない）。

### §9.1 環境・プロトコル

- 実機: M4 Max（macOS 26.6.2・rustc 1.96.0）。詳細は
  `docs/perf/logs/metal-gemm-hfrag-kernel-gpu-ab-1370/env_info.txt`。
- プロトコル: `--release`・`--test-threads=1`・20 warmup + 20 測定・
  5 プロセス起動の中央値（`docs/perf/metal-gemm-reuse-phase-breakdown.md`
  §11.5 と同一）。
- 実行時 load average は約 6〜8（20 ユーザー共有環境）で、N=512/1024 の
  一部構成に run 間 最大 約 4.5 倍の二峰性（同一構成・同一 N でも run
  ごとに束が変わる）が観測された。契約検証（`resolved_cfg == cfg`・
  複合判定 pass）はすべて成立しており、機構上の欠陥ではなく計測環境の
  ノイズと判断する（`docs/perf/logs/metal-gemm-hfrag-kernel-gpu-ab-1370/
  aggregate.md`「他セッション負荷変動」参照）。
- **発見事項（ステップ 5 スモーク結果）**: N=512 のスモークテスト
  （丸め済み入力・複合判定）は 5/5 run とも pass した（`fail-closed
  assert` の切替は不要だった）。

### §9.2 S 系列（候補スイープ。N=1024/2048/4096）

全 staged `tile::CANDIDATES`（`CANDIDATES[7]`＝`SINGLE_SIMDGROUP_8X8`は
`staged=false`のため対象外。§2.1「スコープ境界」）を hfrag 単独で
計測し、N 別の最良タイルを確定した（詳細表は `aggregate.md`「S 系列」）。

| N | 最良候補 index | tile (bm,bn,bk,wm,wn) | 中央値 (ms) |
|---|---|---|---|
| 1024 | 6 | (64,32,8,4,1) | 0.2312 |
| 2048 | 0 | (64,64,16,2,2) | 1.5443 |
| 4096 | 9 | (64,64,32,2,2) | 12.2568 |

`cand4`／`cand10`（いずれも `wm=1`）は N=2048/4096 で本番選択構成比
約 10〜12 倍という極端な劣化を示した（f32 版では同程度の劣化はない）。
half フラグメントロードと `wm=1` の相互作用が疑われるが、原因切り分けは
本イシューのスコープ外とする（§9.7）。

確定値は `crates/backend-metal/src/gemm_hfrag_diag_tests.rs::
HFRAG_BEST_BY_N` へ反映済み（N=512 は S 系列対象外のため本番選択構成
`CANDIDATES[5]` をそのまま使う）。

### §9.3 A 系列（同一タイル比較。N=512/1024/2048/4096）

本番選択構成（`tile::select_for_device`。f32/hfrag 共通）で base（f32）
/head（hfrag）を比較した「タイル形状を揃えた純粋な half MMA 効果」。

| N | tile | base(f32) 中央値 (ms) | head(hfrag) 中央値 (ms) | 5 run 比 | 符号一貫性 |
|---|---|---|---|---|---|
| 512 | (64,32,32,2,2) | 0.2004 | 0.2147 | 1.066〜1.073 | 5/5（hfrag遅い） |
| 1024 | (64,32,8,4,1) | 1.0193 | 1.0432 | 0.974〜1.026 | **不一致** |
| 2048 | (64,32,16,2,2) | 1.6140 | 1.7880 | 1.103〜1.161 | 5/5（hfrag遅い） |
| 4096 | (32,64,16,2,2) | 13.7396 | 16.7565 | 1.150〜1.223 | 5/5（hfrag遅い） |

**同一タイルでは hfrag は f32 より一貫して遅い**（判定可能な全 N で
約 7〜22% 遅い）。half フラグメントへの変換コスト（協調ロード時の
f32→half 変換）が、同一タイル形状での half MMA 自体のスループット
優位を相殺している。

### §9.4 B 系列（結線判断の主根拠。N=512/1024/2048/4096）

head を S 系列確定の N 別最良タイル、base を本番選択構成（f32）とした
比較。

| N | base tile | head tile | base 中央値 (ms) | head 中央値 (ms) | 5 run 比 | 符号一貫性 | 判定 |
|---|---|---|---|---|---|---|---|
| 512 | (64,32,32,2,2) | 同左 | 0.0462* | 0.0495* | 1.066〜1.075 | 5/5（hfrag遅い） | REJECT |
| 1024 | (64,32,8,4,1) | 同左 | 0.2245* | 0.2317* | 0.894〜1.046 | 不一致 | undetermined |
| 2048 | (64,64,16,2,2) | 同左 | 1.6102 | 1.5551 | 0.957〜1.149 | 不一致 | undetermined |
| **4096** | **(32,64,16,2,2)** | **(64,64,32,2,2)** | 13.6670 | 12.2688 | **0.897〜0.899** | **5/5（hfrag速い）** | **ADOPT-as-opt-in-candidate** |

（*N=512/1024 の絶対値は上記「二峰性」の影響で run 間分散が大きい
（詳細は `aggregate.md`）。ただし比自体は 5 run とも安定した符号を示す
N=512 では信頼できる。N=1024 は絶対値・比とも不安定で undetermined。）

### §9.5 妥当性帯チェック（#1275 分母との突合。副次）

`docs/perf/metal-gemm-reuse-phase-breakdown.md` §11.5 の分母（同一
プロトコル）: N=1024 1.0267 ms・N=2048 3.1849 ms・N=4096 13.7051 ms。
本イシューの A/B base（f32）: N=1024 約 0.22〜1.02 ms（二峰性）・
N=2048 約 1.61 ms・N=4096 約 13.67〜13.74 ms。N=4096 は分母と概ね整合。
N=1024/2048 の乖離は `docs/perf/metal-gemm-n4096-kernel-gap.md` §16.4 が
記録する既知のプロトコル間乖離・run 間変動の範囲内と考えられ、本イシュー
の判定はこの突合を単独の判定根拠にしない（同一 run 内 `head_over_base_
kernel_gpu` を主判定とする設計。§9.1）。

### §9.6 採否判断

- **N=4096 のみ ADOPT-as-opt-in-candidate**: hfrag（cand9・bk=32）が
  本番選択構成（cand2・bk=16）に対し 5/5 run 符号一貫・比の分散も
  0.897〜0.899 と極小で約 10.2〜11.5% 高速。
- **ただしこの速さは「half MMA 自体の演算スループット向上」ではなく
  「hfrag の SMEM 使用量が f32 のちょうど半分（`shared_mem_bytes_
  hfrag_for`）であるため、f32 では選ばれない `bk=32`・64×64 構成が
  hfrag では viable かつ最速になる」という間接効果である**（§9.3 の
  同一タイル比較が、同一タイルでは hfrag が f32 より遅いことを示す
  ため）。E9（親 #1368）の狙い「half MMA による ALU スループット向上」
  はこの実測では確認されなかった。
- **N=512 は REJECT**（5/5 run 符号一貫で hfrag が遅い）。
- **N=1024/2048 は undetermined**（run 間で符号が反転。§9.1 の共有負荷
  ノイズが主因と考えられる）。
- **総合判断**: hfrag は N=4096 に限れば「SMEM 節約による viable な
  タイル拡大」という間接効果で本番選択構成を上回るが、half MMA 自体の
  優位性は同一タイル比較で否定された。opt-in 候補としての前進価値は
  N=4096 限定かつ「hfrag 固有の効果ではなく SMEM 予算差の産物」という
  留保付きであり、**全形状を通した無条件の opt-in 候補前進は推奨しない**。
  N=4096 限定の効果を追求する場合、hfrag（精度低下を伴う）ではなく
  f32 のまま `bk=32` 構成（`CANDIDATES[9]`）の SMEM 使用量そのものを
  削減する設計（例: エピローグ領域の再利用）を先に検討する方が精度を
  落とさず同じ間接効果を得られる可能性があり、より筋の良い方向と考える
  （後続イシューでの検討を推奨。§9.7）。

### §9.7 スコープ外・引き継ぎ

- hfrag の本番結線（opt-in 公開 API 設計＋ユーザー承認が前提。§9.6 の
  留保付き判断のため、無条件の起票は推奨しない）。
- `cand4`／`cand10`（`wm=1`）が hfrag で極端に劣化する原因の切り分け
  （§9.2）。
- N=4096 の「SMEM 節約による viable なタイル拡大」効果を、精度を落とす
  hfrag 経由ではなく f32 のまま実現できないかの検討（§9.6 の代替案）。
- N=1024/2048 の undetermined 判定の再検証（低負荷環境での再実行）。
- REQ-2 非後退ベースライン行の追加・ゲート化（§5・§8。人間承認必須）。
- NT/TN/TT・端あり形状の hfrag 純カーネル時間（診断入口が NN 限定）。
- 4096³ の丸めなし参照 parity（§4・§8 を継承）。
- M4 Max 以外の機種・DGX Spark（本イシューは Metal 限定）。

### §9.8 関連ログ

`docs/perf/logs/metal-gemm-hfrag-kernel-gpu-ab-1370/`（S/A/B 系列の
5 プロセス起動生ログ・`aggregate.md`〈抽出コマンド・詳細表・妥当性帯
突合〉・`env_info.txt`・`uptime_before_*.txt`・`pmset_therm_after.txt`。
内部ホスト名は含めない）。

