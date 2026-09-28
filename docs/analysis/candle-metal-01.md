# candle 0.11.0 Metal GEMM 解析（MLX steel vs 自作タイル）

イシュー #2090「candle 0.11.0 Metal GEMM 解析（MLX steel vs 自作タイル）」に対応する。
親 #2089（Phase 2: 他ライブラリ解析）。

## §0 目的・範囲・解析の契約

framework-compare の M4 Max Metal GEMM で candle 0.11.0 に負けているセル（2026-09-19 版
スコアボードで 0.59〜0.74×）の原因を特定するため、candle の Metal GEMM（MLX steel から
派生）のタイル構成・スウィズル・同期方式を自作の `gemm_simdgroup_tiled`／`tile::CANDIDATES`／
`dispatch_auto` と対比し、既存 REJECT／判定不能な実験（E1〜E9・NAX・MPP・hfrag・split-K・
thread_elements・smem swizzle 等）とどこが違うのか、「まだ試していない差分」だけを列挙する。

**契約**: 他ライブラリのコード・シェーダ・派生物をリポへ持ち込まない（結論と出典のみ）。
一行の式も逐語では引用せず数式表記＋`file:line` に置き換える。閉源部分（cuBLAS・MPS 内部
実装）はディスパッチ層までとし、本 doc は candle 側のソースを実際に読める範囲に限定する。
実測・`#[ignore]` テスト・`docs/perf/logs/` は本 doc の範囲外（docs 専用・コード変更ゼロ）。
tolerance・baseline・`Cargo.toml`・ガードレール閾値・`docs/spec/` は変更しない。

## §1 出典・受入基準記載パスとの差分

出典クレート（いずれも crates.io からダウンロードし `scripts/bench/framework-compare/Cargo.lock`
記載の checksum で照合済み）:

| crate | version | checksum（sha256） | VCS sha1 |
|---|---|---|---|
| `candle-metal-kernels` | 0.11.0 | `242e83c6acf639bb273c929d73c67a882bb4dd08a140f121096e19ba2f213d3e` | `31f35b147389700ed2a178ee66a91c3cc25cc80d` |
| `candle-core` | 0.11.0 | `5ecb245093b0f791b89d3420c3df9c6d49c60ab63ba54db896bf8a3baf486706` | `31f35b147389700ed2a178ee66a91c3cc25cc80d` |

上流 URL: `https://github.com/huggingface/candle/tree/<sha>/candle-metal-kernels`（`candle-core`
側も同一 sha1 の `.cargo_vcs_info.json`）。

イシュー #2090 本文に記載された解析対象パス（`kernels.metal`・`matmul_params.metal`・
`candle-core/src/metal_backend/ops.rs`）は **0.11.0 には存在しない**。実際の対象は次の通り:

- `candle-metal-kernels-0.11.0/src/metal_src/mlx_gemm.metal`（GEMM 本体。1483 行）
- `candle-metal-kernels-0.11.0/src/metal_src/gemv.metal`（M=1／N=1 経路）・`src/metal_src/utils.metal`
- `candle-metal-kernels-0.11.0/src/kernels/mlx_gemm.rs`（`select_tile_config`・`check_batch_collapse`・
  `call_mlx_gemm`・`call_mlx_gemv`）
- `candle-metal-kernels-0.11.0/src/metal/device.rs`（`architecture_name`／`device_type`）
- `candle-core-0.11.0/src/metal_backend/mod.rs` の `fn matmul`（1730 行目）

`mlx_gemm.rs` には `should_use_split_k` が定義されているが `#[allow(dead_code)]` で未使用
（candle は split-K を使わない）。

## §2 ディスパッチ層

`candle-core-0.11.0/src/metal_backend/mod.rs:1730` の `fn matmul` は、`self.dtype` が
`F32`／`F16`／`BF16` のとき常に `candle_metal_kernels::call_mlx_gemm` を呼ぶ
（`candle-core-0.11.0/src/metal_backend/mod.rs:1753`）。それ以外の dtype は
未対応 dtype を報告するエラーを返して拒否する（`candle-core-0.11.0/src/metal_backend/mod.rs`
の `fn matmul` 内、非対応 dtype 分岐）。
**cuBLAS／MPS／MPP へのフォールバック判定は存在しない**（該当コードなし。確認完了）。

`call_mlx_gemm`（`candle-metal-kernels-0.11.0/src/kernels/mlx_gemm.rs:453`）は
`m == 1 || n == 1` のとき `call_mlx_gemv`（`mlx_gemm.rs:520-521`）に分岐する（M=1／N=1 の
gemv 特化経路）。自作 `gemm.rs` には gemv 特化経路は存在しない（grep で該当識別子なし）。

batch collapse（`check_batch_collapse`・`mlx_gemm.rs:180` 付近）: `batch_size > 1` かつ A が
転置されておらず、A の batch ストライドが `M * K`（連続）、B の batch ストライドが `0`
（2D へブロードキャスト）のとき、`[batch, M, K] @ [K, N]` を `[batch*M, K] @ [K, N]` へ畳み込む。

転置判定はストライドから行う（`lhs_m1`／`lhs_m2` と `m`／`k` の一致パターンで `a_trans`
を、`rhs_m1`／`rhs_m2` と `k`／`n` の一致パターンで `b_trans` を決定。`mlx_gemm.rs:494-509`）。
パイプラインは「カーネル名＋function constant（`has_batch`・`align_M`／`align_N`／`align_K` 等）」
の組ごとにキャッシュされる。

## §3 タイル構成

### §3.1 タイル定数

`candle-metal-kernels-0.11.0/src/kernels/mlx_gemm.rs:41-45` に 5 つの `TileConfig`
（`bm, bn, bk, wm, wn`）定数がある:

| 定数名 | `(bm, bn, bk, wm, wn)` |
|---|---|
| `TILE_32_32_16_2_2` | `(32, 32, 16, 2, 2)` |
| `TILE_64_64_16_2_2` | `(64, 64, 16, 2, 2)` |
| `TILE_64_64_16_1_2` | `(64, 64, 16, 1, 2)` |
| `TILE_64_32_32_2_2` | `(64, 32, 32, 2, 2)` |
| `TILE_32_64_16_1_2` | `(32, 64, 16, 1, 2)` |

自作 `crates/backend-metal/src/tile.rs:852` 以降の `CANDIDATES`（全 11 構成、index 0〜10）
との対応:

| candle 定数 | 自作 `CANDIDATES` index |
|---|---|
| `TILE_64_64_16_2_2` | `[0]` |
| `TILE_64_32_16_2_2` 相当なし（自作のみ） | `[1]`（自作専用） |
| `(32,64,16,2,2)` 相当なし（自作のみ） | `[2]`（自作専用） |
| `TILE_32_32_16_2_2` | `[3]` |
| `TILE_64_64_16_1_2` | `[4]` |
| `TILE_64_32_32_2_2` | `[5]` |
| `(64,32,8,4,1)`（MLX classic のみ・candle には無い） | `[6]` |
| — | `[7]`（`SINGLE_SIMDGROUP_8X8`。自作専用） |
| `TILE_32_64_16_1_2` | `[8]`（#1143 で追加） |
| — | `[9]`（E7・`(64,64,32,2,2)`。自作専用） |
| — | `[10]`（E8・`(128,64,16,2,2)`。自作専用） |

candle 5 構成は自作 `CANDIDATES[0, 3, 4, 5, 8]` の 5 index に **完全一致で収録済み**。
MLX classic 経路の 6 構成（candle 5 構成 + `(64,32,8,4,1)`）も自作 `CANDIDATES[0, 3, 4, 5, 6, 8]`
の 6 index に完全一致で収録済み（`index 8` は #1143 で追加）。
自作だけにある構成: `[1]`（`64,32,16,2,2`）・`[2]`（`32,64,16,2,2`）・`[7]`（単一 simdgroup
8x8）・`[9]`（E7）・`[10]`（E8）。

### §3.2 選択ロジック（`select_tile_config`）

`mlx_gemm.rs:59-176`。`m < 16` なら常に `TILE_32_32_16_2_2`（M が極小のときの thread
under-utilization 回避）。それ以外は `total_output = batch_size * m * n`（**K は含まない**）
と `2^20`（1,048,576）の比較で `is_large_matmul` を決め、デバイス種別（`MetalDeviceType::
Phone`／`BasePro`／`Ultra`／`Max`／`Medium`）ごとに分岐する。

Phone／BasePro（`mlx_gemm.rs:88-100`）は `is_large_matmul` を**参照しない**: nt（`!a_trans
&& b_trans`）なら常に `TILE_64_32_32_2_2`、それ以外（F32 の nn を含む）は常に
`TILE_64_64_16_2_2`。Max／Medium・Ultra は `is_large_matmul` で分岐し、F32 の nn（非 nt）
は小さい方（`is_large_matmul == false`）で `TILE_64_32_32_2_2`、大きい方
（`is_large_matmul == true`）で `TILE_64_64_16_2_2` になる（`mlx_gemm.rs:145-176`）。

square GEMM（`m == n == N`）では `total_output = N * N` であり、`N = 1024` でちょうど
`2^20` の境界（`is_large_matmul` は `>=` 判定なので `N = 1024` から真）、`N = 2048`／`4096`
は大きく超える。`N = 512` は `512 * 512 = 262144 < 2^20` で `is_large_matmul == false`。

- **`N >= 1024` の正方 f32 NN は、デバイス種別に関わらず全て `TILE_64_64_16_2_2` = 自作
  `CANDIDATES[0]` を使う**（Phone/BasePro は無条件、Max/Medium/Ultra は `is_large_matmul`
  成立のため）。
- `N = 512` の正方 f32 NN は Max／Medium／Ultra では `TILE_64_32_32_2_2`
  （`CANDIDATES[5]`）、Phone／BasePro では `TILE_64_64_16_2_2`（`CANDIDATES[0]`）になり、
  デバイス種別に依存する。

**M4 Max の `MTLDevice.architecture.name` が `MetalDeviceType` のどの区分（`Max`／
`Medium`／それ以外）に解決されるかは本 doc の解析範囲では未確認**（`candle-metal-kernels-
0.11.0/src/metal/device.rs` の `architecture_name`／`device_type` 判定ロジックは読めるが、
実機の `architecture.name` 文字列自体は Mac 実機でしか取得できない）。ただし M4 Max は上記
どの区分に分類されても `N >= 1024` 正方 f32 NN の結論（`TILE_64_64_16_2_2`）は変わらない。
`N = 512` の結果だけがデバイス区分に依存するため未確定のまま残す。

**見出し結論**: `docs/perf/metal-gemm-n4096-kernel-gap.md:12` に記録された cand0
（`CANDIDATES[0]`＝候補表インデックス 0。candle `TILE_64_64_16_2_2` と同一タイル形状）は
N=4096 で 1.22 TFLOPS（`CANDIDATES[2]` の約 12%）へ崩壊する。candle は N=1024/2048/4096 の
正方 f32 NN で常にこの同じタイル形状（`CANDIDATES[0]` 相当）を使っている。**つまり同じ
タイル形状で自作と candle の間に大差があり、差の原因はタイル選択ではなくカーネル本体
（loader／MMA・アドレッシング・同期）にある**。

自作側の選択（`tile::select_for_device`／`select_with_occupancy_for_device`。`tile.rs:1317-
1318` の実測テーブル）は正方 GEMM で N=512→`CANDIDATES[5]`、N=1024→`CANDIDATES[6]`、
N=2048→`CANDIDATES[1]`、N=4096→`CANDIDATES[2]` を返す（M4 Max 40 コア限定の厳密
`(m,n,k)` タプル一致。`tile.rs:1310-1326`）。N=1024/2048/4096 では自作と candle は**異なる
タイル**を選んでいる点に注意（candle は常に `CANDIDATES[0]` 系、自作は N ごとに
`[6]`／`[1]`／`[2]` を実測選択）。§3 の「同一タイル」比較は「candle が実際に使うタイル形状
（`CANDIDATES[0]`）を自作カーネルへ適用した場合の純カーネル時間」（cand0 実験）についての
ものであり、本番選択パス同士の比較ではない。

## §4 スウィズル

### §4.1 threadgroup ID スウィズル

candle のホスト側 `call_mlx_gemm`（`mlx_gemm.rs:566`）は `swizzle_log` を
**リテラル `0` で固定**しており、`mlx_gemm.rs:567` の `tile_swizzle = 1 << swizzle_log = 1`
となるため実質無効化されている。カーネル側（`mlx_gemm.metal`）には swizzle 再マップ式が
2 か所ある: `mlx_gemm.metal:250-253`（`swizzle` 関数。`tid_x' = tid.x >> log`、
`tid_y' = (tid.y << log) + (tid.x & (2^log - 1))`）と `mlx_gemm.metal:741-743`／
`1065-1067`（fused カーネル内の同型の再マップ）。式は構造こそ持つが、ホストが常に
`log = 0` を渡すため実行時には恒等写像になる。

自作は `crates/backend-metal/src/shaders/gemm.metal:660` の `SWIZZLE_ENABLED`
（function constant index 7。既定 `false`）と `tile::swizzled_grid` により opt-in で
同種のスウィズルを持つが、既定 OFF のまま（`docs/perf/metal-gemm-tgid-swizzle-ab.md`・
E5・#1279/#1280。判定不可のため OFF 維持）。**candle も実効的にスウィズル無効**であるため、
candle の優位（あれば）はスウィズルに由来しない。

### §4.2 lane レベル（simdgroup fragment）

candle の BlockMMA（`mlx_gemm.metal:125-178` 付近、`load_unsafe`／`load_safe` を持つ
ローダ型）は simdgroup 内の lane 配置を用いて fragment オフセットを計算する
（`STEEL_PRAGMA_UNROLL` によるループ展開を伴う）。threadgroup メモリ（smem）格納位置に
対する XOR swizzle は候補実装内に見当たらず（`swizzle` 識別子はホスト側 `swizzle_log`
関連と threadgroup ID 再マップの 2 か所のみ）、`tgp_padding`（§5）によるバンク衝突回避
のみを用いている。

自作は E3（`docs/perf/metal-gemm-n4096-kernel-gap.md` §10・`tgp-k1` フラグメントロード
方式）と smem XOR swizzle（`COOP_SMEM_SWIZZLE`。`docs/perf/metal-gemm-coop-load-candidates.md`・
`docs/perf/metal-gemm-tile-class-split.md`・`docs/perf/metal-gemm-n4096-kernel-gap.md` に
実装記録あり）を持つ。candle 側には smem 格納位置の XOR swizzle 相当が無い。

## §5 同期・メモリ

candle の K ループ（`mlx_gemm.metal:685-800` 付近）は threadgroup メモリを単一バッファ
（ダブルバッファなし）で使い、1 反復あたり `threadgroup_barrier(mem_flags::mem_threadgroup)`
をロード前・ロード後で計 2 回呼ぶ（`mlx_gemm.metal:685, 699, 710, 720` 等、複数の反復
バリアント〈align 済み／端数〉に分かれて出現）。BlockMMA 内の fragment ロードと MMA の間
には `simdgroup_barrier(mem_flags::mem_none)`（`mlx_gemm.metal:343, 354, 365`）。ループ前
にも `threadgroup_barrier(mem_flags::mem_none)`（`mlx_gemm.metal:749, 789, 1163, 1250`）
がある。

自作の `FINE_BARRIER_ENABLED`（`gemm.metal:678`。function constant index 8・既定 `false`。
E5・`docs/perf/metal-gemm-fine-barrier-ab.md`）は同種のバリア粒度変更を扱う opt-in フラグ
だが既定 OFF のまま。

padding: candle は `tgp_padding_a = tgp_padding_b = 16 / sizeof(T)`
（`mlx_gemm.metal:625-626`。f32 なら 4 要素）。自作 `TileConfig::TGP_PAD_ELEMS = 4`
（`crates/backend-metal/src/tile.rs:284`・#538）と**同値**（`docs/perf/metal-gemm-tgp-padding.md`
参照）。

境界処理: candle は function constant `align_M`／`align_N`／`align_K`
（`mlx_gemm.metal:1005-1007`）が真のとき境界検査なしのロード（`load_unsafe`。
`mlx_gemm.metal:688, 694, 776-777`）を使い、端数タイルだけ `load_safe`
（`mlx_gemm.metal:690, 696, 717-718, 797-798, 1212-1213`）を使う。
**REQ-8（`.claude/rules/coding-rust.md`「カーネル実装の境界検査」）により、手動境界チェックを
外す形はそのままでは自作へ採用できない**（`docs/backend-metal-aligned-load-decision.md`
参照。自作は align 済みタイルでも手動境界チェックを維持する方針）。

unroll: candle は `STEEL_PRAGMA_UNROLL`（`mlx_gemm.metal:10` で定義されるマクロで、
full unroll を指示する clang の loop pragma に展開される）をロード・MMA の内側
ループに多用する（`mlx_gemm.metal:125-178` 等）。自作側の対応する実験は E1
（`docs/perf/metal-gemm-n4096-kernel-gap.md` §7・#1188/#1282/#1284）。

epilogue・`use_out_source`・axpby 相当のフラグは通常の matmul（`Op::MatMul`／bias 無し）
では使われない経路のため、本 doc の対比範囲外とする。

## §6 既存 REJECT との差分

| 既存実験 | REJECT／判定不能の理由（出典） | candle の選択 | 未試行の差分候補か |
|---|---|---|---|
| E1 loop unroll pragma（#1188/#1282/#1284） | 条件付き gating を実装したが本番結線は撤回（`docs/perf/metal-gemm-n4096-kernel-gap.md` §7.6a／§7.7a／§7.10.6：`UNROLL_ACC_ENABLED = false` 維持、判定不可） | `STEEL_PRAGMA_UNROLL`（full unroll）を常時使用（§5） | 自作は opt-in のみ・既定 OFF。candle は常時 ON という差分は残る |
| E2 ソーステキスト特殊化（#1288/#1289） | 採否判断は `docs/perf/metal-gemm-n4096-kernel-gap.md` §9.4（詳細は同 doc） | — | 既存記録参照 |
| E3 フラグメントロード方式（#1295） | `docs/perf/metal-gemm-n4096-kernel-gap.md` §10.4 採否判断（`tgp-k1` 系） | lane 配置ベースの BlockMMA fragment ロード（§4.2） | smem XOR swizzle との組合せ差分は未試行 |
| E4 協調ロードレイアウト（#1300） | `docs/perf/metal-gemm-n4096-kernel-gap.md` §11.4 採否判断 | — | 既存記録参照 |
| E5 fine barrier（`docs/perf/metal-gemm-fine-barrier-ab.md`） | 判定不可・既定 OFF 維持 | 単一バッファ・2 回 barrier/反復（§5） | candle の barrier 粒度（ロード前後の 2 回）は自作 opt-in 版と厳密一致していない可能性があり未検証 |
| E5 tgid swizzle（`docs/perf/metal-gemm-tgid-swizzle-ab.md`） | 判定不可・既定 OFF 維持 | `swizzle_log = 0` 固定で実質無効（§4.1） | candle も無効なので優位要因ではない（確認済み） |
| E6 タイルクラス分割（`docs/perf/metal-gemm-n4096-kernel-gap.md` §3／§12） | §12.4 採否判断 | — | 既存記録参照 |
| E7 `(64,64,32,2,2)` 候補（#1329/#1330） | `docs/perf/metal-gemm-n4096-kernel-gap.md` §13.5／§14.5 採否判断 | candle には `bk=32` の構成自体はある（`TILE_64_32_32_2_2`＝`(64,32,32,2,2)`。§3.1）が、E7 の `(64,64,32,2,2)` と bm／bn まで完全一致する構成はない | — |
| E8 `(128,64,16,2,2)` 候補（#1325/#1331/#1332） | `docs/perf/metal-gemm-n4096-kernel-gap.md` §15.6／§16.5 採否判断 | candle には `bm=128` の構成なし（§3.1） | — |
| E9 hfrag（`docs/perf/metal-gemm-hfrag-candidate.md`・`docs/perf/metal-gemm-n4096-kernel-gap.md` §17） | 既存記録参照 | — | — |
| split-K（`docs/backend-metal-splitk-decision.md`） | 既存記録参照 | candle は split-K 未使用（`should_use_split_k` が `#[allow(dead_code)]`。§1） | candle も split-K を使わないため優位要因ではない |
| thread_elements（`docs/perf/metal-gemm-thread-elements-candidate.md`） | 既存記録参照（本番未結線・#1694 で REJECT 確定） | — | — |
| NAX 経路（`docs/backend-metal-mlx-classic-nax-decision.md`） | NAX（`MetalPerformancePrimitives`）不採用 | candle も NAX を使わず `mlx_gemm.metal` の手書きカーネルを使う（cuBLAS/MPS/MPP フォールバックが無いことと整合。§2） | — |
| MPP tensor ops（`docs/backend-metal-mpp-tensor-decision.md`） | 既存記録参照 | 同上（NAX と同じ理由で不使用） | — |
| smem XOR swizzle（`COOP_SMEM_SWIZZLE`） | `docs/perf/metal-gemm-coop-load-candidates.md` 等 | candle には無い（§4.2） | candle に無い自作固有の最適化のため差分候補としては除外するが、劣後要因かどうかは candle との比較からは判断できず未検証（§6 未試行の差分候補 3.） |
| async copy（`docs/backend-metal-async-copy-decision.md`） | 既存記録参照 | candle のロードは `load_unsafe`／`load_safe`（同期的な vector load。§5）で async copy 相当は確認できず | 未試行の差分候補 |
| serpentine（`docs/perf/metal-gemm-serpentine-ab.md`） | 既存記録参照 | — | — |
| morton mapping（`docs/backend-metal-morton-mapping-decision.md`） | 既存記録参照 | — | — |
| register accumulator（`docs/perf/metal-gemm-register-accumulator-ab.md`） | 既存記録参照 | — | — |
| float4 staged load（`docs/perf/metal-gemm-float4-staged-load.md`） | 既存記録参照 | — | — |
| tgp padding（`docs/perf/metal-gemm-tgp-padding.md`） | 既存記録（採用済み・`TGP_PAD_ELEMS=4`） | 同値（§5） | 一致のため差分候補ではない |
| aligned load 境界検査省略（`docs/backend-metal-aligned-load-decision.md`） | REQ-8 により手動境界検査を外す最適化は不採用 | candle は `align_M/N/K` で境界検査なしロードを使用（§5） | **REQ-8 の制約下でどこまで近づけられるか（境界検査を保ったまま unroll・レジスタ配置を candle 相当に近づける）が未試行の中核候補** |

**未試行の差分候補（優先度付けは #2098 に委ねる）**:

1. `STEEL_PRAGMA_UNROLL` 相当の unroll を既定 ON にした場合の純カーネル時間（E1 は
   条件付き gating を実装済みだが常時 ON の効果は未測定。§6 表）。
2. candle のロード方式（`load_unsafe`／`load_safe`。REQ-8 の制約下で境界検査を保った
   まま vector load の形をどこまで近づけられるか）。
3. smem XOR swizzle と E3 フラグメントロードの組合せ（candle は両方とも持たない。この
   組合せ自体は未試行であり、candle が両者を持たないという事実だけからは自作のこの
   組合せが劣後要因かどうか・優位要因になっているかのいずれも判断できない。A/B 実測後に
   判断する）。
4. async copy（candle 側の確認された不在と、自作の既存決定記録との突合の深掘り）。

## §7 既存 doc との不整合の記録

`docs/backend-metal-mlx-classic-nax-decision.md` §1 の対比表（行 20-30）は、`(32,64,16,1,2)`
に「本実装に完全一致で存在しない（最近傍 index 2）」と記す。これは #1143（`CANDIDATES[8]`
追加）より前の内容であり、現在は `CANDIDATES[8]` が完全一致で存在する（§3.1）。また同表の
`CANDIDATES` 行番号（`tile.rs:270-339` 付近）も #1143／E7／E8 追加後の現在の行番号と
一致しない。**本 doc は既存 doc の内容を訂正するものではなく、不整合の存在のみを記録する**
（訂正・追記が必要かどうかの判断・Issue 起票はユーザー判断とし、本 doc の範囲外とする）。

## §8 スコープ外・Phase 3 への申し送り

- M4 Max 実機の `MTLDevice.architecture.name`（末尾文字を含む正確な値）の確認。Mac 実機で
  `architecture.name` を 1 行出力するだけの手順で足りる（§3.2）。判明すれば N=512 の
  candle タイル選択が確定する。
- §6 の未試行差分候補の A/B 実測は Phase 3（#2098 以降）のスコープ。
- cuBLAS／MPS／MPP の詳細調査（本 doc では「候補として存在しない」ことの確認に留めた）。
- CPU 側の同種解析は #2092 のスコープ。

## §9 ライセンス記録

`candle-metal-kernels` 0.11.0・`candle-core` 0.11.0 はいずれも `Cargo.toml` の
`license = "MIT OR Apache-2.0"`、`repository = https://github.com/huggingface/candle`
（§1）。`candle-core-0.11.0` の展開物には `LICENSE` ファイルが**存在する**が、
`candle-metal-kernels-0.11.0` の展開物には LICENSE ファイルが**存在しない**（Cargo.toml
の license 式と上流リポジトリの `LICENSE-MIT`／`LICENSE-APACHE` を出典とする）。

`mlx_gemm.metal` 冒頭コメント（ファイル先頭部）には MLX（`ml-explore/mlx` の
`steel/gemm` 実装）からの抽出である旨と `Copyright © 2024 Apple Inc.`、MLX コミット
`02efb310cac667bc547d1b96f21596c221f84fe7` への参照がある。MLX は MIT ライセンス。
**candle 由来（MIT OR Apache-2.0）と MLX 由来（MIT・Apple Inc.）の二重の出典**を持つ
点を記録する。

crates.io checksum（`scripts/bench/framework-compare/Cargo.lock` 記載値。§1 の表と同一）:
`candle-metal-kernels` = `242e83c6acf639bb273c929d73c67a882bb4dd08a140f121096e19ba2f213d3e`、
`candle-core` = `5ecb245093b0f791b89d3420c3df9c6d49c60ab63ba54db896bf8a3baf486706`。

依存は追加・更新していないため `docs/license-matrix.md` の更新対象ではない
（同 doc 8b 節に既存の `candle-core` 0.11.0 実測記録がある）。
