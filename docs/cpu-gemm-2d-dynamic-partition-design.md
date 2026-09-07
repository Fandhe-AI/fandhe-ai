# CPU GEMM `gemm_blis_parallel` の (mc, nc) 2D タイル動的分配設計・bit 完全一致条件（イシュー #1307）

## 状態: 設計記録のみ（`crates/` 配下のコード変更なし）。実装は #1311、実機 A/B 計測は #1312、本番結線は #1313 が担当する。

## 位置づけ

起票元 #1307・親 #1303・ルート #1283・前提 #1305（`docs/perf/cpu-gemm-candle-gate-remeasurement.md`
§17）・後続 #1310 系（#1311・#1312）・#1313。

`.claude/rules/coding-rust.md`「カーネル実装の境界検査（REQ-8）」・「バックエンド構成（REQ-2）」の
数値一致契約・`.claude/rules/security.md`（`unsafe` 理由コメント＋レビュー必須）に従う。

### #1305 の結論の引き継ぎ

`docs/perf/cpu-gemm-candle-gate-remeasurement.md` §17 が専有環境での
`RAYON_NUM_THREADS` スイープを両実機で 5 回計測中央値で再実測し、以下を確定した:

- **DGX Spark GB10（Cortex-X925 ×10 + Cortex-A725 ×10）**: N=1024・T=10 の非単調性
  （対 T=8 比 0.5094・5 run 中 4 run が 1.0 未満。§17.3）は **H1（異種コア由来）と確定**
  （§17.7）。taskset 大コア pin で T=10 が T=8 の **1.32 倍**へ回復・小コア pin で **0.52**
  へ劣化（§17.4）。分割粒度（行パネル行数）は 3 条件で完全に同一のためこの結果は
  「分割粒度」（H3）では説明できない。整列形状 N=1920 でも T=20（全コア）で **一貫した劣化**
  （比 0.81・5 run 全て 1.0 未満。§17.5）が観測され、H1 と整合する追加根拠となっている。
  N=2048・N=4096 では非単調性は観測されなかった（§17.3・§17.7）
- **Apple M4 Max**: 専有ゲート不通過のため **undetermined**（§17.6・§17.7）。数値自体は
  非再現方向（大コア数 T=12 でもむしろ改善する傾向）を示すが確定させない
- #1307 への前提（§17.8 相当の整理）: 本設計の主対象は **N≈1024・T=10〜20（全コア使用時の
  緩和を含む）**。`IcDynamic`（行パネル単位の動的配布のみ・#1367 で REJECT）が異種コア吸収に
  足りなかった構造（pc ごとの同期点＋列全幅 B pack。§2.1）を、2D 分配で解消できるかが本設計の
  判断点である。M4 Max の効果は undetermined につき過大評価しない。#1364（大コア数限定・
  `BIG_CORE_LIMIT_ENABLED`。REJECT 確定）は既定スレッド数を絞る施策であり、本設計（既定
  スレッド数のまま分配粒度を変える施策）の前提を否定しない

### #1338 承認事項

2026-09-06、C 列ブロックの raw pointer 分割による `unsafe` 導入が、理由コメント＋
security-auditor レビュー必須を条件にユーザー承認済み（イシュー #1338 コメント）。本設計は
これを**前提としてよいが必須ではない**（§4 で安全側の主案を先に示し、raw pointer 方式は
代替として位置づける）。

## §1 本イシューの成果物

`gemm_blis_parallel` を (mc, nc) 2D タイル job の動的分配へ変更する設計記録として本ドキュメントを
新設し、以下を確定する:

1. 現行 `RowPanel` と 2D 動的分配の対比・タイル粒度・rayon 分配方式（§2・§5・§6）
2. bit 完全一致契約（REQ-2）を満たす分割条件と回帰テストでの検証方法（§3・§4・§10）
3. `GemmDriverVariant` への追加方針（既存 A/B ハーネスへの統合）と #1041 と同一の採用ゲート（§9）
4. #1305 の結果分岐（背景負荷由来と結論された場合の縮小・中止条件）（§11）

後続 #1311 がコード変更なしで着手できる粒度（関数シグネチャ・擬似コード・テスト一覧）まで書く。

## §2 現行 `RowPanel` と 2D 動的分配の対比

### §2.1 既存の並列ドライバ候補と結果

| variant（[`GemmDriverVariant`](../crates/backend-cpu/src/gemm_blis/mod.rs)） | 分配 | packing | 実機結果 |
|---|---|---|---|
| `RowPanel`（本番既定） | 静的行パネル `m.div_ceil(T)`（`mod.rs:531`・`mod.rs:571-577`） | B はタスクごと個別（Q 倍重複）・A は (jc,pc,ic) ごと | 基準 |
| `SharedB`（#750） | 静的行パネル | B を (jc,pc) ごと共有・同期点あり | REJECT（#1144） |
| `SharedBPcOuter`（#1041） | 静的行パネル・pc 最外 | B 共有・A は pc ごと 1 回 | REJECT（#1144） |
| `IcDynamic`（#1366） | 行パネルを `AtomicUsize` で動的配布・pc 最外 | B を pc ごと列全幅で共有 pack・**pc ごとに同期点** | REJECT（#1367。DGX N=1024 で 0.6263・N=2048 で 0.8492） |
| `gemm_blis_parallel_2d_with_blocks`（#753。`mod.rs:1765`） | MC タイル数均等化した**行範囲のみ**（`partition::row_ranges_for_workers`） | `RowPanel` と同じ | 本番未結線（列方向は未分配。`docs/perf/cpu-gemm-runtime-cache-detect.md` §4「unsafe を使わない設計判断」） |

`IcDynamic` の DGX 後退は「pc ごとの同期点＋列全幅 B pack」の構造に帰属推定されている
（`docs/cpu-gemm-b-packing-sharing-decision.md` §F「追記（イシュー #1367）」）。2D 設計は
この構造を**繰り返さない**（job 内 K 全域を単一 worker が同期なしで処理する。§2.2）。

### §2.2 2D 動的分配の定義

対比表（分配単位／分配方式／packing 重複／同期点／`unsafe`／異種コア吸収／既存 variant との関係）:

| 観点 | `RowPanel`（本番既定） | 2D 動的分配（本設計） |
|---|---|---|
| 分配単位 | 行パネル `T` 個（静的・行数のみ均等） | (行帯 × 列帯) タイル job `J` 個（`J > T`。§5） |
| 分配方式 | `par_chunks_mut` によるチャンク分割（job 数 ≈ T。rayon の work-stealing スケジューラは 2D と同一で、違いは job 数のみ） | rayon `into_par_iter`（job 数 J > T。§6） |
| packing 重複 | B: タスク（≈T）ごと個別 pack（Q 倍重複） | B: job（≈J）ごと個別 pack。§5 の解析では `RowPanel` 以下（表参照） |
| 同期点 | パネル完了後の join のみ | job 完了後の join のみ（**pc ごとのバリアなし**。`IcDynamic` との違い） |
| `unsafe` | なし | 主案 S: なし／代替 U: raw pointer（#1338 承認済み。§4） |
| 異種コア吸収 | job 数 ≈ T（worker 数程度）のため吸収余地が乏しい（rayon スケジューラ自体は work stealing だが分割が粗く再配分の機会が少ない） | job 数を worker 数より増やし work stealing による吸収余地を広げる（§8） |
| 既存 variant との関係 | — | #753 の `gemm_blis_parallel_2d_with_blocks`（行範囲のみ分配）を**包含**する（cb=1 の特殊形。削除・置換しない） |

- **job 空間**: [`partition::tile_grid`](../crates/backend-cpu/src/gemm_blis/partition.rs)（`partition.rs:67`）
  が返す `Tile { row: Range<usize>, col: Range<usize> }`（`partition.rs:40`）の集合（行帯 × 列帯）を
  新設の `job_grid`（§5）が (mc_job, nc_job) 刻みで求める
- **job 本体**: その C 部分ブロックに対し **K 全域**（pc = 0, kc, 2·kc, … 昇順）を単一 worker が
  逐次処理する。構造は既存 [`dispatch_region`](../crates/backend-cpu/src/gemm_blis/mod.rs)
  （`mod.rs:871`）を「列範囲 `cols: Range<usize>`」で一般化したもの（jc は
  `cols.start..cols.end` を `blocks.nc` 刻み → pc → ic〈`blocks.mc` 刻み〉→ jr → ir）。job 間の
  同期は rayon の join のみ
- **packing**: job 内 private（B は (jc,pc) ごと・A は (jc,pc,ic) ごと。job ごと確保、または
  worker ローカルで再利用。§6）
- **分配**: rayon work stealing（§6）

## §3 bit 完全一致契約（REQ-2）を満たす分割条件

以下を**必要十分条件**として列挙する。`.claude/rules/coding-rust.md`「バックエンド構成（REQ-2）」
のうち matmul 系 FMA 契約（`f32::mul_add` 基準・split-single 例外は本設計と無関係）と、
「カーネル実装の境界検査（REQ-8）」の手動境界チェック維持要件を前提とする。

1. **K 非分割**: job は (行帯, 列帯) の 2 軸のみで定義し、K 方向（pc）を job 間で分割しない。
   部分和・後加算・split-k を禁止する。既存カーネルは「C 要素ごとに acc へ C をロードしてから
   p 昇順の FMA 連鎖を行い、レーン間縮約を行わない」契約（
   [`microkernel.rs`](../crates/backend-cpu/src/gemm_blis/microkernel.rs) の `run`／`run_with_ldc`
   契約コメント〈`microkernel.rs:320-417`〉・
   [`microkernel/neon.rs`](../crates/backend-cpu/src/gemm_blis/microkernel/neon.rs) の `compute`
   実装〈NEON `vfmaq_laneq_f32` 連鎖〉）を持ち、本設計はこれを変更しない
2. **pc 昇順・同一 kc 境界**: 全 job が pc=0 から `blocks.kc` 刻み昇順で K を回し、pc ごとに
   C を load→FMA→store する（既存 `dispatch_region` と同一。`blocks.kc` は呼び出しあたり 1 回
   確定して全 job へ同一値を渡す。既存の [`default_blocks`](../crates/backend-cpu/src/gemm_blis/mod.rs)
   と同じく MC=128／KC=256／NC=512 の既定を job 粒度計算の入力として使う）
3. **1 要素 1 job**: 各 C 要素はちょうど 1 つの job に属し、その job の単一 worker が K 全域を
   処理する。`tile_grid` の被覆完全・互いに素は `partition::tests::tile_grid_covers_every_point_exactly_once`
   （`partition.rs:166`）で既に検証済みであり、`job_grid`（§5）が生成する `tiles` は
   `tile_grid(m, n, mc_job, nc_job)` と同一集合であることを契約とする（§5・§10）
4. **ISA 単一選択**: ISA トークン（`try_new`。カーネル型の選択）は呼び出しあたり 1 回、分配前に
   確定し、全 job が同じ `K: Microkernel` でモノモーフィック化された job 本体を実行する（job
   ごとの再検出はしない。既存 `dispatch_region` と同一方針）
5. **カーネル本体不変**: C の acc ロード→p 昇順 FMA 連鎖→store の演算列は変更しない。C の
   アドレス解決（§4）のみを抽象化し、**FMA 本体は単一ソース**に保つ（CUDA persistent タイル
   キュー版が `TP_TILE_CORE` を非 persistent 版と完全共有して出力 bit 同一を機構として保証する
   構造〈`docs/perf/cuda-gemm-tiled-pipeline-persistent.md`〉と同型の保証方式）
6. **タイル整列・nc_job・mc_job は性能パラメータであり正しさ条件ではない**: 端タイルと
   完全タイルは bit 同一（#557・`docs/perf/cpu-gemm-full-tile-direct-c.md`）、パネル境界の
   変化も bit 同一（`tests/gemm_blis_parity.rs::gemm_blis_parallel_matches_naive_bit_exact_across_thread_pools`）、
   `nc=n` の `IcDynamic` も bit 同一（`docs/perf/cpu-gemm-ic-dynamic-variant.md` §3）。よって
   `mc_job`／`nc_job`／`jobs_per_worker`（§5）の値を変えても正しさは変わらない
7. **手動境界チェック維持**: 新カーネル入口（§4 の `run_rows`／raw pointer 版）も
   `check_panel_lengths`（`microkernel.rs:209`）や C 行長検査を fail-closed（
   [`TileBoundsError`](../crates/backend-cpu/src/gemm_blis/microkernel.rs)〈`microkernel.rs:99`〉）
   で行う。端タイルの copy-in/out は安全な slice 操作のみとする
   （`.claude/rules/coding-rust.md`「カーネル実装の境界検査」）
8. **FMA 契約不変**: `f32::mul_add`／`vfmaq_laneq_f32` 等の丸め方針は変更しない
   （`.claude/rules/coding-rust.md`「バックエンド構成（REQ-2）」）

## §4 C 列分割の借用問題と実装方式（本設計の中核判断）

### §4.1 問題の所在

[`Microkernel::run_with_ldc`](../crates/backend-cpu/src/gemm_blis/microkernel.rs)
（`microkernel.rs:417` にトレイトのデフォルト実装、各 ISA が上書き）は完全タイルで
`c[row0 .. row0 + (mr-1)*ldc + nr]` 相当の `&mut [f32]` を受け取り、`ldc = n` のためこの
スライスは**同一行の他 job の列を含む**。2D 分配で同じ行帯の別列 job が並行するとき、
重なり合う `&mut [f32]` を同時に生成すること自体が UB（実アクセスが互いに素でも `&mut` の
一意性違反）であり、既存入口をそのまま流用できない。

`docs/perf/cpu-gemm-runtime-cache-detect.md` §4 が同種の問題（gemm crate 方式の完全な 2D
分配）について「常時 raw pointer 経路の `unsafe` が必要」と判断し #753 では不採用としたのは、
その時点で列方向分配のユーザー承認がなかったため（§4「この判断の限界」）。#1338 で
raw pointer 分割の `unsafe` 導入がユーザー承認された今、本設計はこの制約を主案（unsafe 非導入）
と代替案（unsafe・承認済み）の 2 案として扱う。

### §4.2 主案 S: 行セグメント slice 方式（`unsafe` 非導入）

job は `rows: Vec<&mut [f32]>`（担当行帯の各行について、その行の担当列範囲
`c[r*n + cols.start .. r*n + cols.end]` の `&mut`）を所有する。構築は次の手順による:

```text
fn split_into_jobs<'c>(c: &'c mut [f32], n: usize, tiles: &[Tile]) -> Vec<JobCTile<'c>> {
    // 1. c.chunks_mut(n) で行スライスへ分割（各行は他行と disjoint。
    //    コンパイル時借用検査で保証される）。
    // 2. 行帯でグループ化した行スライス配列を作る。
    // 3. 同一行帯内の列帯ごとに split_at_mut 連鎖で列範囲へ分割し、
    //    (行帯, 列帯) = タイルごとに Vec<&mut [f32]> を集約する。
    //    列帯は tile_grid が返す境界と一致するため、同一行の異なる
    //    列帯の &mut は重ならない（split_at_mut がコンパイル時に保証）。
}
```

job 間で `&mut` が重ならないことをコンパイル時借用検査で保証する（`IcDynamic` の
`Mutex<Option<&mut [f32]>>` スロットも不要。job は所有 `Vec` で `Send` になる）。

[`Microkernel`](../crates/backend-cpu/src/gemm_blis/microkernel.rs) trait に**行スライス入口**
をデフォルト実装付きで追加する:

```text
/// 行帯 × 列帯タイル向けの C 書き込み入口。`rows` は担当行の行ごとに
/// 独立した &mut [f32]（他 job の列を含まない）で、`col_off` は各行
/// スライス内でのタイル先頭列オフセット（0 が通例。呼び出し側の
/// 集約方法によっては非 0 もありうるため引数として残す）。
/// FMA 本体は `run_with_ldc` と同一ソースを共有し（REQ-2 契約 §3-5）、
/// C アクセスのみ `rows[i][col_off + j]` に置き換える。
fn run_rows(
    &self,
    ap: &[f32],
    bp: &[f32],
    rows: &mut [&mut [f32]],
    col_off: usize,
    kc_len: usize,
) -> Result<(), TileBoundsError> {
    // 既定実装: rows.len() == Self::MR・各 rows[i].len() >= col_off + Self::NR
    // を検査（TileBoundsError::PanelLengthMismatch 相当）したうえで、
    // 行アクセサ経由で run_with_ldc 相当のロジックへ委譲する。
}
```

ISA 実装（neon/avx2/avx512/scalar）は `compute` の C アクセス `c[i*ldc + off]` を
`rows[i][col_off + off]` に置き換えるだけで、FMA 本体は共有する（§3 条件 5）。端タイルは
既存どおり `MAX_TILE` スタックバッファ経由（`rows[ic+ir+i][col..col+nr_eff]` の copy-in/out。
安全な slice 操作のみ）。

オーバーヘッド見積り: タイル×pc ごとに MR 個の行ポインタ取得・長さ検査（NEON: 1 タイル
あたり `MR*NR*kc` ≈ 8×12×256 ≈ 24576 回の FMA 相当命令に対し、行ポインタ取得は高々 MR=8 回
→ 1% 未満と推定。実測は #1311／#1312 が確認する）。

課題: 4 ISA ＋ scalar への入口追加（#1311 の実装量）。`run_rows` と `run_with_ldc` の bit 同一を
ISA ごとの単体テストで固定する（§10）。

### §4.3 代替 U: raw pointer 方式（#1338 承認済み）

```text
/// C の部分ブロックへの排他アクセスを表す軽量ハンドル。tile_grid が
/// 生成する互いに素なタイルへの 1:1 対応を呼び出し側が保証する前提で
/// Send/Sync を unsafe impl する（各スレッドは自分の CTileView にのみ
/// 触れ、ptr が指す領域外へは決してアクセスしない）。
struct CTileView {
    ptr: *mut f32,
    ldc: usize,
    rows: Range<usize>,
    cols: Range<usize>,
}
// SAFETY: 複数 CTileView が同時に存在しうるが、それぞれが指す
// (rows, cols) 矩形は tile_grid の被覆完全・互いに素性
// （partition::tests::tile_grid_covers_every_point_exactly_once）
// により重複しない。カーネルは ldc 契約（i<MR, j<NR の範囲のみに
// c[i*ldc+j] でアクセスする。microkernel.rs の run_with_ldc 契約）を
// 守るため、実際にアクセスする範囲も (rows, cols) 内に収まる。
unsafe impl Send for CTileView {}
unsafe impl Sync for CTileView {}
```

カーネルに `*mut f32 + ldc` の raw 入口を追加する（既存 `run_with_ldc` は内部で slice→pointer
化しているため薄い wrapper）。

**SAFETY 不変条件（実装時に `// SAFETY:` コメントとして必須）**:

1. job は `tile_grid` の互いに素なタイルに 1:1 対応する（被覆完全・重複なしを
   `partition::tests` で機械検証済み。§10 の `job_grid` テストが同じ性質を保証する）
2. カーネルは `c[i*ldc+j]`（`i < MR`, `j < NR`）以外に触れない契約（`microkernel.rs` の
   `ldc` 契約・`check_panel_lengths` による事前検査）を守る
3. 端タイルの `&mut` 実体化は自 job の列範囲内の行セグメントに限定する（S 案と同じ
   `chunks_mut`／`split_at_mut` 経由で構築し、raw pointer からの `&mut` 生成は最小化する）
4. 重なる `&mut [f32]` を一切生成しない（生ポインタから `&mut` を作るのは条件 3 の行
   セグメントのみ。任意の raw pointer 演算から直接 `&mut` を作らない）

採用条件: S の実装で許容できない性能低下（`run_rows` 経由で `RowPanel` 比 5% 超の後退が
cb=1 構成で観測される等）または実装上の障害が #1311 で判明した場合に U へ切り替える。
`unsafe` 導入時は理由コメント＋security-auditor レビュー必須（`.claude/rules/security.md`）。

### §4.4 比較表

| 観点 | 主案 S（行セグメント slice） | 代替 U（raw pointer） |
|---|---|---|
| `unsafe` 追加 | なし | あり（#1338 承認済み・要 security-auditor レビュー） |
| カーネル変更量 | 4 ISA ＋ scalar へ `run_rows` 入口追加 | 4 ISA ＋ scalar へ raw ポインタ入口追加（`run_with_ldc` の薄い wrapper） |
| 実行時オーバーヘッド | 行ポインタ取得・長さ検査（推定 1% 未満） | ポインタ演算のみ（S よりわずかに軽いと推定。実測未検証） |
| レビュー体制 | 通常レビュー | security-auditor 必須（SAFETY 不変条件の記述込み） |
| bit 同一の保証方法 | コンパイル時借用検査＋ §10 の bit 一致テスト | §10 の bit 一致テスト＋ SAFETY 不変条件の静的検証（`partition::tests`） |

**結論**: 主案 S を採用する。U は#1311 で S が性能・実装上の障害に直面した場合に再承認なしで
切替可能な代替として、本節の SAFETY 不変条件込みで確定しておく。

## §5 job 粒度（純関数 `job_grid` と解析 pack 表）

### §5.1 純関数契約

`partition.rs`（#1311 で実装）へ追加する純関数:

```text
fn job_grid(
    m: usize,
    n: usize,
    mr: usize,
    nr: usize,
    blocks: &BlockingParams,
    num_threads: usize,
    jobs_per_worker: usize,
) -> JobGrid {
    // JobGrid { mc_job: usize, nc_job: usize, tiles: Vec<Tile>,
    //           row_bands: usize, col_bands: usize }
}
```

契約:

- `mc_job` は `mr` の倍数・`nc_job` は `nr` の倍数（端は `tile_grid` の端数として許容）
- `tiles` は `tile_grid(m, n, mc_job, nc_job)` と同一（被覆完全・互いに素。§3 条件 3）
- `row_bands * col_bands >= min(jobs_per_worker * num_threads, 到達可能最大 job 数)`。
  `到達可能最大 job 数 = ceil(m/mr) * ceil(n/nr)`（各行帯・列帯を `mr`／`nr` 単位まで
  細分した場合の上限）。この `row_bands`／`col_bands` は `align_up` 後に `tile_grid` が
  実際に生成する帯数（`row_bands = ceil(m/mc_job)`・`col_bands = ceil(n/nc_job)`）を指す。
  `mc_job`／`nc_job` は `mr`／`nr` の倍数へ切り上げられるため、選択規則が意図した分割数
  （後述の `rb`／`cb`）より少ない帯数しか実際には生成されない場合がある（`align_up` による
  丸め上げの結果、意図した帯より 1 帯あたりの担当範囲が広がり、帯数自体が減るため。
  codex-review #1431／Cursor Bugbot 指摘）。§5.2 の選択規則はこの実帯数の積が
  **`bound` 以上になる** (rb, cb) のみをコスト比較の候補とする契約とする — 意図した
  `rb * cb` ではなく `align_up` 後の実帯数の積で下限を再検査し、下限を満たさない候補が
  誤って選ばれないことを保証する
- 寸法計算は全て `checked_mul`／`saturating_add` を用い、オーバーフロー時は型付きエラー
  （`GemmError::DimProductOverflow` 相当）を返す fail-closed 設計とする
  （`partition::bands`〈`partition.rs:52`〉の既存方針を踏襲）
- `m==0`／`n==0`／`num_threads==1` で全域的に正しく動作する。**`num_threads==1` は下限式・
  選択規則の双方で明示的に job 1 個（`rb=cb=1`・`mc_job=align_up(m, mr)`・
  `nc_job=align_up(n, nr)`）の特例とし、§5.2 のコスト最小化探索を経由しない**（直列実行と
  同一経路になる。並列コスト比較を 1 スレッドに対して行う意味がないため）

### §5.2 選択規則

**`num_threads == 1` の特例**: 下限式（§5.1）どおり job 1 個に固定する
（`rb = cb = 1`・`mc_job = align_up(m, mr)`・`nc_job = align_up(n, nr)`）。
以下のコスト最小化探索は経由しない（1 スレッドに対して並列コスト比較を行う意味がないため）。

`num_threads >= 2` のとき、目標 job 数 `J = jobs_per_worker × num_threads`・
`cap = ceil(m/mr)`（行帯方向に到達可能な最大帯数）・
`到達可能最大 job 数 = cap * ceil(n/nr)`・
`bound = min(J, 到達可能最大 job 数)`（§5.1 の下限。**`align_up` 後の実帯数の積で判定する**）
を求める。

**候補ごとに `align_up` 後の実帯数を先に求める**（意図した `cb`／`rb` そのものではなく、
これらの実帯数を下限判定・コスト計算の両方に使う。§5.1 の契約）:

- 列側: `cb ∈ 1..=ceil(n/nr)` の各候補について `nc_job(cb) = align_up(ceil(n/cb), nr)`・
  `real_cb(cb) = ceil(n / nc_job(cb))` を求める。`align_up` により `nc_job(cb)` が
  意図したより大きく切り上げられると `real_cb(cb) <= cb` になり得る（実際に生成される
  列帯数が意図した `cb` を下回る = **alignment collapse**）
- 行側: `rb ∈ 1..=cap` の各候補について `mc_job(rb) = align_up(ceil(m/rb), mr)`・
  `real_rb(rb) = ceil(m / mc_job(rb))` を求める（同様に `real_rb(rb) <= rb` になり得る）

`cb` の候補ごとに次の手順で `rb` を決め、**実帯数の積で下限を再検査してから**候補として残す:

- `nc_job = align_up(ceil(n/cb), nr)`・`real_cb = ceil(n/nc_job)`
- `rb = min(cap, max(ceil(J/real_cb), ceil(bound/real_cb)))`（意図した `cb` ではなく
  `real_cb` を分母に使う。`real_cb` が `cb` より小さい場合に必要な `rb` が大きくなる方向へ
  自動的に補正される）
- `mc_job = align_up(ceil(m/rb), mr)`・`real_rb = ceil(m/mc_job)`
- **`real_rb * real_cb >= bound` を満たさない候補は棄却する**（`rb` を `cap` まで
  大きくしても `real_cb` 自体が `align_up` で頭打ちのため解消しない場合がある。これが
  codex-review #1431／Cursor Bugbot が指摘した欠陥そのもの — 意図した `cb` に基づく
  `cap * cb >= bound` の検査だけでは、この頭打ちを検出できない）

残った候補について pack 総量モデル

```text
cost(rb, cb) = real_cb_eff * m * k + real_rb * k * n
```

（`real_cb_eff = real_cb * ceil(nc_job / NC)`。A は列帯ごと・B は行帯ごとに重複するため、
コスト計算にも意図した `cb`／`rb` ではなく実帯数 `real_cb`／`real_rb` を使う）を最小化する
(rb, cb) を決める。同コストなら `nc_job` の大きい方（A 再利用回数が多く端タイルが少ない）を
選ぶ。全候補が棄却された場合（`bound` 自体が `到達可能最大 job 数` を超えないため理論上
発生しないが、fail-closed の防御として）は `rb = cap`・`cb = ceil(n/nr)`（最大分割）を採用する。

**具体例 1（cb 絞り込みのみでは不十分なケース。cb=1 除外だけでは足りない反例）**:
`m=n=24`・`mr=8`・`nr=12`・`NC=512`・`num_threads=8`・`jobs_per_worker=2` のとき
`J=16`・`cap=ceil(24/8)=3`・`到達可能最大 job 数=3*ceil(24/12)=3*2=6`・`bound=min(16,6)=6`。
`cb=1`: `nc_job=align_up(24,12)=24`・`real_cb=ceil(24/24)=1`。`rb=min(3, max(ceil(16/1)=16,
ceil(6/1)=6))=3`。`mc_job=align_up(ceil(24/3)=8,8)=8`・`real_rb=ceil(24/8)=3`。
`real_rb*real_cb=3*1=3 < bound=6` のため棄却。
`cb=2`: `nc_job=align_up(ceil(24/2)=12,12)=12`・`real_cb=ceil(24/12)=2`（この形状では
alignment collapse は起きず `real_cb=cb`）。`rb=min(3, max(ceil(16/2)=8, ceil(6/2)=3))=3`。
`mc_job=align_up(ceil(24/3)=8,8)=8`・`real_rb=3`。`real_rb*real_cb=3*2=6=bound` を満たし採用。

**具体例 2（alignment collapse が実際に発生し `cap * cb >= bound` の検査だけでは
防げないケース。codex-review #1431／Cursor Bugbot 指摘の反例）**:
`m=16`・`n=108`・`mr=8`・`nr=12`・`NC=512`・`num_threads=8`・`jobs_per_worker=2` のとき
`J=16`・`cap=ceil(16/8)=2`・`到達可能最大 job 数=2*ceil(108/12)=2*9=18`・
`bound=min(16,18)=16`。
`cb=8`（`cap*cb=16>=bound=16` を満たすため、意図した `cb` だけを見る検査では通過してしまう）:
`nc_job=align_up(ceil(108/8)=14,12)=24`・`real_cb=ceil(108/24)=5`（`align_up` が 14 を
24 へ切り上げたため、意図した 8 帯ではなく実際には 5 帯しか生成されない）。
`rb=min(2, max(ceil(16/5)=4, ceil(16/5)=4))=2`。`mc_job=align_up(ceil(16/2)=8,8)=8`・
`real_rb=ceil(16/8)=2`。`real_rb*real_cb=2*5=10 < bound=16` のため棄却（意図した `rb*cb=16`
は下限を満たすように見えるが、実際に生成される job は 10 個のみ）。
`cb=9`（`n/nr` の上限）: `nc_job=align_up(ceil(108/9)=12,12)=12`・`real_cb=ceil(108/12)=9`
（この `cb` では `align_up` の切り上げが `ceil(n/nr)` と一致し collapse しない）。
`rb=min(2, max(ceil(16/9)=2, ceil(16/9)=2))=2`。`mc_job=8`・`real_rb=2`。
`real_rb*real_cb=2*9=18>=bound=16` を満たし採用（`cb=8` では `bound` を満たせず、`cb=9` まで
上げて初めて満たすことが `real_cb` 経由の検査で判明する）。

`jobs_per_worker` は const（既定 2。§5.3 の表で `RowPanel` の pack 総量を全形状で下回る側）。
`#[cfg(test)]` のパラメータ化入口 `gemm_blis_parallel_2d_dynamic_with_params(..., jobs_per_worker)`
を設け、#1312 が `{2, 4}`（必要なら 8）をスイープする。

### §5.3 解析 pack 表（試算値。実測ではない）

NEON MR=8/NR=12・NC=512・`m=n=k=N` 前提。`RowPanel` は Q=T（タスクごと個別 B pack）前提。
本表は §5.2 の算出式による解析値であり、#1311 の `job_grid` 実装後は同関数の出力で
再現・検証する（値が食い違えば本表を更新する。実測値ではないため捏造リスクはない）。

| N | T | `RowPanel` A/B 重複（総計） | 2D `jobs_per_worker=2`: job 数 (rb×cb)・`mc_job`×`nc_job`・A/B 重複（総計） | 2D `jobs_per_worker=4` |
|---|---|---|---|---|
| 1024 | 10 | 2 / 10（12） | 20 (5×4)・208×264・4 / 5（**9**） | 40 (8×5)・128×216・5 / 8（13） |
| 1024 | 20 | 2 / 20（22） | 40 (8×5)・128×216・5 / 8（**13**） | 80 (10×8)・104×132・8 / 10（18） |
| 2048 | 10 | 4 / 10（14） | 20 (4×5)・512×420・5 / 4（**9**） | 40 (8×5)・256×420・5 / 8（13） |
| 2048 | 20 | 4 / 20（24） | 40 (8×5)・256×420・5 / 8（**13**） | 80 (10×8)・208×264・8 / 10（18） |
| 4096 | 8 | 8 / 8（16） | 18 (2×9)・2048×456・9 / 2（**11**） | 36 (4×9)・1024×456・9 / 4（13） |
| 4096 | 20 | 8 / 20（28） | 45 (5×9)・824×456・9 / 5（**14**） | 81 (9×9)・456×456・9 / 9（18） |

結論: 近正方 job 格子は T が大きいほど `RowPanel`（B が Q 倍）より pack 総量が小さく、
`jobs_per_worker=2` では全掲載形状で `RowPanel` 以下（太字が採用値）。`jobs_per_worker=4` は
N=1024・T≤10 で僅かに上回る（バランス改善とのトレードオフ。#1312 で実測判定）。A 重複は
`RowPanel` の `n/NC` から `cb_eff` へ増える点（N=1024 で 2→4〜5）は明示しておく。

x86_64（AVX2 6×16・AVX-512 8×32）は同じ規則で `mr`/`nr` が変わるだけであり、本表は
再計算しない（実機実測対象は両 aarch64: DGX Spark GB10・Apple M4 Max）。

## §6 rayon での分配方式

**主案**: `jobs.into_par_iter().with_min_len(1).try_for_each(|job| ...)`（`Vec<Job>` を消費。
rayon の適応分割＝work stealing。カスタム同期なし・`Mutex` なし）。job 配列は
**column-band-major**（同一列帯の行帯を連続）に並べる（性能上の推奨。正しさ非依存）。
これは、隣接 job を同時処理する worker 群が同じ B 列を共有キャッシュ上で参照しやすくする
ための順序であり、job 分割の正しさ（§3）には影響しない。

代替案（不採用）:

- `AtomicUsize` カウンタ claim（`IcDynamic` 型）: 所有 `Job` を配るには
  `Mutex<Option<Job>>` スロットまたは `unsafe` が要り、主案に対する優位性が明確でないため
  採らない（後退した場合の切替候補として記載のみ残す）
- `rayon::join` 再帰分割（#565 案 C）: job 粒度が固定サイズであるため再帰分割の利点
  （不均一な部分問題の適応分割）が薄く、`into_par_iter` の work stealing で足りる
- `broadcast`: 全 worker が同一クロージャを実行する API のため、job キューの消費という
  本設計の要件に合わない

worker ローカル `PanelBuffers` の再利用（`map_init` または job ごと確保）は #1311 の実装裁量
とし、bit 一致に影響しない（§3 条件 6 の「タイル整列・粒度は性能パラメータ」に準ずる）。

## §7 packing の扱い（採らないものを明示）

採用: job private packing（§2.2）。理由: `IcDynamic`／`SharedB*` の同期点構造を繰り返さない・
pack 総量は §5 の表のとおり `RowPanel` 以下。

**採らない**:

- job 間の B 共有 pack（#750/#1041/#1144 で REJECT 済みの構造を踏襲しない）
- pc 最外＋バリア（#1367 で REJECT 済みの構造を踏襲しない）
- worker ローカル B キャッシュ（複雑さに見合う根拠がない。将来拡張の候補としてのみ記録）
- 列全幅 B pack（footprint 増。`IcDynamic` と同じ懸念）
- split-K（§3 条件 1 に違反する）

## §8 対象形状・期待効果

- 主対象: N≈1024・T=10〜20（DGX H1 確定。§0）。`docs/perf/cpu-gemm-candle-gate-remeasurement.md`
  §17.4 の大コア pin 値（対 T=8 比 1.32）を**上限の目安**とし、目標は「pin なし T=10 で
  T=8 以上（比 ≥ 1.0）」「T=20 で T=8 以上」とする。上限を超える予測はしない
- 副次: N=2048/4096 は非後退（≥0.95）が要件。pack 総量は §5 のとおり減る方向だが、job 粒度が
  大きくキャッシュ挙動が変わるため実測に委ねる（#1312）
- M4 Max: undetermined（§0）のため予測しない。12P+4E の異種構成への理論的転用可能性のみ
  将来検討事項として記録する（本設計のスコープには含めない）

## §9 `GemmDriverVariant` への追加方針・A/B 統合・採用ゲート

- 新 variant 名: `GemmDriverVariant::TwoDDynamic`（`#[cfg(test)]`。既存 4 候補
  （`RowPanel`／`SharedB`／`SharedBPcOuter`／`IcDynamic`）と同じく
  `gemm_blis_parallel_variant` の `match` へ追加する。`dispatch_two_d_dynamic` の arch 別 3 版
  （neon/avx2-avx512/scalar 相当）を `dispatch_ic_dynamic` と同型で用意する
- A/B 統合: `gemm_blis_variant_ab_1024_2048`／`_4096`（`mod.rs` の A/B ハーネス群）の
  `variants` 配列へ追加する。出力形式 `variant=… size=… median_gflops=…` は不変とし、
  既存の `run_variants_interleaved` の巡回計測をそのまま利用する。
  `docs/perf/logs/cpu-gemm-ic-dynamic-ab-1367/aggregate.py` を流用可能にする
- 採用ゲート（#1312 が計測前に固定する）:
  - **主判定は対 `RowPanel`**: N=1024/2048 で比 ≥1.00・N=4096 で比 ≥0.95・両実機一致
    （#1367 の採用ゲートと同一基準。`docs/cpu-gemm-b-packing-sharing-decision.md` §F
    「追記（イシュー #1367）」参照）
  - **参考判定は対 gemm crate**（#1041 系「1024/2048 で gemm crate 以上かつ 4096 非劣化」）。
    `oss-gemm-compare` は本番経路のみ計測可で variant 選択に非対応のため、結線前は
    `RowPanel` の対 gemm crate 比 × 対 `RowPanel` 比の連鎖で推定し、確定は #1313 結線後の
    `oss-gemm-compare` 再計測で行う
- 前提: `gemm_blis_two_d_dynamic_matches_row_panel_bit_exact_large`（`#[ignore]`・release）が
  両実機で pass すること（§10）
- 計測条件: `RAYON_NUM_THREADS` 未設定（既定全コア）に加え、DGX は **T=10・T=20 を明示**
  （#1305 で非単調性が観測された値）。5 回独立プロセス中央値・load average を env_info へ記録

`oss-gemm-compare` への variant 選択オプション追加（#1367 が未実施のまま残した事項）は、
#1312 の任意前提として提案するに留める（本設計のスコープ外。§13）。

## §10 回帰テスト計画（#1311 が実装する一覧）

| テスト | 目的 |
|---|---|
| `partition::tests::job_grid_*` | 被覆完全・互いに素・`mr`/`nr` 整列・job 数下限（`ceil(m/mc_job) * ceil(n/nc_job)` という **`align_up` 後の実帯数の積**で判定。意図した `rb`／`cb` 自体では判定しない）・`num_threads=1`／`m<mr`／`n<nr`／`m==0`／`n==0`／オーバーフロー近傍の全域性・`jobs_per_worker` 単調性 |
| `partition::tests::job_grid_lower_bound_survives_alignment_collapse` | §5.2 具体例 2（`m=16,n=108,mr=8,nr=12,num_threads=8,jobs_per_worker=2`）を固定入力として再現し、`ceil(m/mc_job) * ceil(n/nc_job) >= bound` が実際に成立することを検証する回帰テスト（codex-review #1431／Cursor Bugbot 指摘の反例固定化） |
| `microkernel::tests::run_rows_matches_run_with_ldc_bit_exact_*`（ISA ごと） | 新入口と既存入口の bit 同一（乱数タイル・`ldc=n`／`ldc=NR` 双方・境界エラーの fail-closed） |
| `gemm_blis_parallel_variant_all_candidates_match_naive_bit_exact` 拡張 | `TwoDDynamic` を配列へ追加（5 形状 × スレッド数 1/2/3/16） |
| `gemm_blis_two_d_dynamic_matches_row_panel_bit_exact_across_shapes_and_threads` | C 初期値非ゼロ・端あり形状（`m`/`n` が MR/NR 非倍数・`m<mr`・`n<nr`・`k=0`・`k<kc`・非正方・512³）× スレッド数 1/2/3/16 × `jobs_per_worker` {1,2,4,8} |
| `gemm_blis_two_d_dynamic_multi_pc_matches_serial_bit_exact` | 小 `kc`（複数 pc）・小 `mc`/`nc` で job 内 jc/ic ループを複数回通す |
| `gemm_blis_two_d_dynamic_is_deterministic_across_runs` | 同一入力 2 回で bit 同一（分配順序非依存の直接検証） |
| `gemm_blis_two_d_dynamic_matches_row_panel_bit_exact_large`（`#[ignore]`） | 1024/2048/4096・release・既定スレッド数（実機前提条件） |
| 転置（`Nt`/`Tn`）の bit 一致 | job 本体が `GemmTranspose` を受けるため、`tests/gemm_transposed_parity.rs` 相当を variant 経由で追加（#1313 結線対象の事前保証） |

## §11 #1305 結果に基づく分岐（縮小・中止条件）

実際の結果: DGX で H1 確定（§0）。**設計は予定どおり全体を進める**（本章の分岐は判断過程の
記録として残す）。

仮に両実機で「背景負荷由来」（H2）と結論されていた場合の扱い: 2D 設計は「pack 総量削減」
（§5）の観点のみで縮小継続し、`jobs_per_worker=2` 固定・T 明示計測なしの単純 A/B へ縮小する。
それでも `RowPanel` 比 1.00 未満なら中止する。

#1312 での中止条件:

- DGX N=1024・T=10 で `jobs_per_worker ∈ {2, 4}` のいずれも対 `RowPanel` 比 < 1.00 →
  REJECT・#1313 は「不採用」記録でクローズする
- 片実機のみ達成 → REJECT（両実機一致要件。§9 と同一基準）
- 形状・スレッド数条件付き採用は、条件付き部分集合で両実機一致要件を満たす場合に限り
  #1313 で結線条件として明記する

M4 Max が計測中に専有できない場合の扱い: `docs/perf/cpu-gemm-candle-gate-remeasurement.md`
§17 と同じ「共有負荷下として記録・undetermined」規則を踏襲し、DGX の結果だけで ADOPT しない。

## §12 #1313（本番結線）時の考慮事項

- job 本体は `GemmTranspose` を受け（`dispatch_region` の一般化のため自然に対応する）、
  `gemm_blis_parallel_with_transpose`（Nn/Nt/Tn）を一括で置換できる
- `gemm_blis_bias_act_parallel`（`mod.rs:652-733` 付近）: `apply_epilogue` を **job ごとに
  K 全域完了後 1 回**（各要素はちょうど 1 job に属するため多重適用が構造的に起きない。
  `docs/cpu-gemm-b-packing-sharing-decision.md` §C「案 B」の epilogue 懸念は本設計では
  生じない）。または join 後に全体へ 1 回でも良い（#1313 の実装裁量）
- `partition` モジュールと新入口の `#[cfg(test)]` 解除、`mod.rs` 冒頭コメント（並列化方式の
  説明。`mod.rs:15` 付近）の更新
- `m==1`／`n==0` の専用経路は不変。小形状で job 数 1 の場合は直列と同一結果になる
- `oss-gemm-compare`・framework-compare gemm cpu の before/after を #1313 の受入根拠とする

## §13 スコープ外（本ドキュメントで扱わない事項）

- `crates/` の実装（#1311）・実機 A/B（#1312）・結線（#1313）
- worker ローカル B キャッシュ・NUMA 配慮・x86_64 実機実測
- `oss-gemm-compare` の variant 選択オプション（採否は #1312 で判断）
- `docs/spec/`（正本 submodule）の変更（本設計に伴う仕様変更は不要）

## 出典

イシュー #1307/#1305/#1303/#1283/#1310〜#1313/#1338/#1365〜#1367/#753/#565/#750/#1041/#1144・
`crates/backend-cpu/src/gemm_blis/{mod.rs,partition.rs,microkernel.rs,microkernel/neon.rs}`
（origin/main `0e3b6f0` 時点）・`docs/perf/cpu-gemm-candle-gate-remeasurement.md` §8.2/§14/§17・
`docs/perf/cpu-gemm-ic-dynamic-variant.md`・`docs/cpu-gemm-b-packing-sharing-decision.md`・
`docs/perf/cpu-gemm-runtime-cache-detect.md` §4・`docs/perf/cpu-gemm-full-tile-direct-c.md`・
`docs/perf/cpu-gemm-blocking-sweep.md` §3.2・
`.claude/rules/{coding-rust,security,deps-policy,out-of-scope-tracking}.md`
