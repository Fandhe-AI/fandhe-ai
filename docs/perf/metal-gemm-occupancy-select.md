# Metal GEMM occupancy 判定の `tile.rs::select()` 組み込み（#542）

イシュー #542「perf(backend-metal): occupancy 判定を tile.rs::select() のタイル選択へ組み込み」の記録。
親 #480（GEMM 最適化）D-7b（D-7 の分割: `docs/perf/metal-gemm-bottleneck-diagnosis.md` 冒頭リスト参照）。
#541（D-7a・`docs/perf/metal-gemm-occupancy-target.md`）で実装した occupancy 目標算出の基盤
（`tile::actual_groups`・`tile::OccupancyParams`・`tile::is_underoccupied`）を、`tile::select()` の
タイル選択へ実際に組み込む。

## 状態: 判定式の実装・単体テスト（Linux worktree）は完了。**実機計測（受け入れ基準 2）は未実施（Mac 実機実行待ち）**

本実装環境は Linux のため、`crate::context::MetalContext::new` が実行時に取得する GPU コア数・
threadgroup memory 上限の実測値、および `examples/gemm_bench.rs` の旧/新比較セクションの実測は
Mac 実機セッション（実機ツリー #408 系）に委ねる（#541・#532 と同じ前例）。

## 1. 判定式（`crate::tile::select_with_occupancy`）

2 段階判定:

1. **形状判定**（`tile::select` と同一ロジック）: `(m, n, k)` から形状優先の `TileConfig` を決める
   （`SMALL=64` 未満は `SINGLE_SIMDGROUP_8X8`、`LARGE=512` 以上かつ正方は `CANDIDATES[0]`（64×64）、
   縦長〈`m >= 2n`〉は `CANDIDATES[1]`（64×32）、横長〈`n >= 2m`〉は `CANDIDATES[2]`（32×64）、
   それ以外の中形状は `CANDIDATES[3]`（32×32））。
2. **occupancy 縮退**: 段 1 の結果が大タイル系（`CANDIDATES[0..=2]`）かつ `params: Option<OccupancyParams>`
   が `Some` のとき、`actual_groups(m, n, cfg)` と `params.ideal_groups(IDEAL_GROUPS_MULTIPLIER_F32, cfg)`
   を比較し `is_underoccupied` なら `CANDIDATES[3]`（32×32 中形状）へ縮退する。

```
select_with_occupancy(m, n, k, params) =
    shape_cfg = select(m, n, k)                                     // 段 1
    if shape_cfg ∉ {CANDIDATES[0], CANDIDATES[1], CANDIDATES[2]}:
        return shape_cfg                                            // 縮退対象外
    if params is None: return shape_cfg                             // fail-safe
    actual = actual_groups(m, n, shape_cfg)
    if actual is None: return shape_cfg                             // fail-safe
    ideal = params.ideal_groups(IDEAL_GROUPS_MULTIPLIER_F32, shape_cfg)
    if ideal is None: return shape_cfg                              // fail-safe
    return CANDIDATES[3] if is_underoccupied(actual, ideal) else shape_cfg
```

`select(m, n, k) = select_with_occupancy(m, n, k, None)` として後方互換入口を維持する（既存呼び出し元・
既存テストは変更不要）。

## 2. fail-safe フォールバック方針（#541 doc §5 の残課題を確定）

以下はいずれも occupancy 判定を無効化し段 1（形状判定）の結果をそのまま返す（`select()` と完全一致。
panic 経路なし）:

- `params` が `None`（`MetalContext::occupancy_params()` が GPU コア数取得不能で `None` を返した場合、
  または呼び出し側が意図的に無効化したい場合）
- `actual_groups` が `None`（`cfg.bm`／`cfg.bn` が 0。`CANDIDATES` 内の構成では実質発生しない）
- `OccupancyParams::ideal_groups` が `None`（`gpu_core_count == 0`・`multiplier == 0`・SMEM 予算超過に
  よる `smem_groups_per_core == 0`・オーバーフロー）

## 3. 事前検証値（M4 Max 想定値。机上計算・実機実測ではない）

`OccupancyParams { gpu_core_count: 40, max_threadgroup_memory_bytes: 32768 }`（`docs/perf/
metal-gemm-occupancy-target.md` §3.3 の期待値。実機実測は未完了）での机上計算:

| 候補 | shared_mem_bytes | smem_groups_per_core | ideal_groups（係数 6） |
|------|-------------------|------------------------|---------------------------|
| `CANDIDATES[0]`（64×64×16） | 9472 | 3 | 120 |
| `CANDIDATES[1]`（64×32×16） | 7424 | 4 | 160 |
| `CANDIDATES[2]`（32×64×16） | 6912 | 4 | 160 |
| `CANDIDATES[3]`（32×32×16） | 4864 | 6 | 240 |

正方 size と `CANDIDATES[0]` の `actual_groups`（`ceil(size/64)^2`）:

| size | actual_groups | ideal_groups | 判定 | 選択構成 |
|------|----------------|---------------|------|-----------|
| 512 | 64 | 120 | under-occupied（64 ≤ 120） | `CANDIDATES[3]`（32×32）へ縮退 |
| 1024 | 256 | 120 | 非 under-occupied | `CANDIDATES[0]`（64×64）維持 |
| 2048 | 1024 | 120 | 非 under-occupied | `CANDIDATES[0]`（64×64）維持 |
| 4096 | 4096 | 120 | 非 under-occupied | `CANDIDATES[0]`（64×64）維持 |

上記の期待値どおり `crates/backend-metal/src/tile.rs` の `select_with_occupancy_shrinks_512_square_...`・
`select_with_occupancy_keeps_large_squares_from_1024_...` テストで固定済み（Linux で実行可能な純粋関数
テスト）。**期待どおりであれば受け入れ基準 2（512/1024/2048/4096 で劣化なし）の実測リスクは size=512 の
縮退判定に局所化される**（1024 以上は選択構成が現行 `select()` と同一のため）。

## 4. 実機計測手順（Mac 実機セッションで実施）

```sh
cargo run -p backend-metal --example gemm_bench --release
```

`examples/gemm_bench.rs` の `--- occupancy 判定組み込み比較 ---` セクション（size ∈ {512, 1024, 2048,
4096}）が、旧（`tile::select` を明示 `SimdgroupTiled` で使う対照群）と新（`dispatch_auto`。
`ctx.occupancy_params()` 経由で `select_with_occupancy` を使う）を並べて TFLOPS・選択された
`TileConfig`（`bm`/`bn`）・`occupancy_params` の実測値を出力する（5 回以上計測の中央値。
`.claude/rules/coding-rust.md`「ベンチは 5 回計測の中央値を採用」）。

`cargo test -p backend-metal -- --ignored --nocapture` で `MetalOccupancyInfo::probe` 系テストの実測値
（GPU コア数・SMEM 上限）も確認できる（`docs/perf/metal-gemm-occupancy-target.md` §3.2 と同じ手順）。

## 5. 記録テンプレート（要実機実測記入）

| size | old_tile（`select`） | old_tflops | new_tile（`select_with_occupancy`） | new_tflops | new_over_old | 判定 |
|------|------------------------|------------|-----------------------------------------|------------|----------------|------|
| 512 | **要実測記入** | **要実測記入** | **要実測記入** | **要実測記入** | **要実測記入** | **要実測記入**（劣化ありなら閾値・係数を見直す） |
| 1024 | **要実測記入** | **要実測記入** | **要実測記入** | **要実測記入** | **要実測記入** | **要実測記入** |
| 2048 | **要実測記入** | **要実測記入** | **要実測記入** | **要実測記入** | **要実測記入** | **要実測記入** |
| 4096 | **要実測記入** | **要実測記入** | **要実測記入** | **要実測記入** | **要実測記入** | **要実測記入** |

劣化が確認された場合は `IDEAL_GROUPS_MULTIPLIER_F32`（現状 6）・`SMEM_GROUPS_PER_CORE_CAP`（現状 16）の
係数見直しを Mac 実機セッションで行う（ユーザー承認済みの `.claude/rules/coding-rust.md`
「テスト許容誤差の変更はユーザー承認必須」の対象外〈性能係数の調整であり数値一致 tolerance ではない〉
だが、値変更自体は実機実測根拠を伴わせる）。

## 6. スコープ外（`out-of-scope-tracking.md` に従い本 PR に混入させない）

- 実機実測値の記入・係数（6/9・`SMEM_GROUPS_PER_CORE_CAP`）の M4 Max 向け確定 → Mac 実機セッション
  （実機ツリー #408 系）
- `examples/gemm_diagnosis.rs` のローカル算出式のクレート内 API への置換（#541 doc §4） → 判定挙動へ
  影響しないため本 PR では見送り、既存の追跡記述を維持
- バッチ次元（MFA の `batchDimension`）導入・f16 系係数（`IDEAL_GROUPS_MULTIPLIER_ALL_16BIT`）の運用
- バックエンド抽象層の経路選択（#67/#68 レイヤ）への occupancy 反映
