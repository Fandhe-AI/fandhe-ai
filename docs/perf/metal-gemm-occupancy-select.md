# Metal GEMM occupancy 判定の `tile.rs::select()` 組み込み（#542）

> **段 1（形状判定）の正方大形状閾値は #744 で是正済み（実測範囲に限定。PR #760 レビュー対応で本注記を
> 実装へ整合）**: 本ファイルの「1. 判定式」「3. 事前検証値」節が前提とする「正方形状は `LARGE=512` 以上で
> `CANDIDATES[0]`（64×64）を選ぶ」という段 1 の挙動は、イシュー #744・2026-08-19 M4 Max 実機実測
> （size=2048 で `CANDIDATES[3]`〈32×32 staged〉が `CANDIDATES[0]` 比 2.8 倍）により、**`m == n == k`
> （真の正方立方形状）かつ `m <= 4096`（実測範囲。512/1024/2048/4096 の 4 点）の帯域に限って**撤去され、
> この範囲では `CANDIDATES[3]` を返すようになった。**`LARGE=512` 定数自体は実装に残存する**: `m != n`
> の準正方長方形、`m == n` でも `k != m`（K 未実測の正方出力）、および `m == n == k` でも 4096 超（実測
> 対象外）の場合は、引き続き `LARGE` 境界前後で `CANDIDATES[0]`／`CANDIDATES[3]` を切り替える（#744
> 是正前と同一挙動）。本ファイルの本文・実測値・記録テンプレートは記録当時のまま変更していない。段 2
> （occupancy 縮退）自体のロジック（縮退対象を大タイル系 `CANDIDATES[0..=2]` に限る旨）は変更されて
> いないが、#744 是正後は段 1 が実測範囲内の正方立方形状に対し `CANDIDATES[0]` を返すことがなくなった
> ため、その帯域での縮退対象は実質発生しない（実測対象外の帯域では縮退判定は引き続き生きている）。
> 是正の判断根拠・実測値は `docs/perf/metal-tile-select-correction.md`
> を参照。

イシュー #542「perf(backend-metal): occupancy 判定を tile.rs::select() のタイル選択へ組み込み」の記録。
親 #480（GEMM 最適化）D-7b（D-7 の分割: `docs/perf/metal-gemm-bottleneck-diagnosis.md` 冒頭リスト参照）。
#541（D-7a・`docs/perf/metal-gemm-occupancy-target.md`）で実装した occupancy 目標算出の基盤
（`tile::actual_groups`・`tile::OccupancyParams`・`tile::is_underoccupied`）を、`tile::select()` の
タイル選択へ実際に組み込む。

## 状態: 判定式の実装・単体テストは完了。**`dispatch_auto` への組み込みはイシュー #747 で不採用確定（#744 是正で吸収）**

`crate::gemm::MetalGemm::dispatch_auto` は `tile::select`（形状のみ）を呼び続ける構成で確定した。
`tile::select_with_occupancy` は #744 是正後、実測帯域〈512/1024/2048/4096〉では `select()` と
常に同一結果になることを確認したため、本番ディスパッチへは**組み込まない**（詳細判断は本ファイル
「6. #747 判断」節）。`select_with_occupancy` 自体は削除せず、実測外の帯域（縦長・横長・準正方大
形状長方形・K 未実測正方出力・4096 超正方立方）向けに実装・単体テスト済みのまま残し、
`examples/gemm_bench.rs` の比較セクションから直接呼び出し可能。

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
cargo run -p fandhe-ai-backend-metal --example gemm_bench --release
```

`examples/gemm_bench.rs` の `--- occupancy 判定組み込み比較 ---` セクション（size ∈ {512, 1024, 2048,
4096}）が、旧（`tile::select` を明示 `SimdgroupTiled` で使う対照群。`dispatch_auto` の現行本番挙動と
同一）と新（`tile::select_with_occupancy`〈`ctx.occupancy_params()` 経由〉が選ぶ構成を同じく明示
`SimdgroupTiled` でディスパッチ）を並べて TFLOPS・選択された `TileConfig`（`bm`/`bn`）・
`occupancy_params` の実測値を出力する（5 回以上計測の中央値。`.claude/rules/coding-rust.md`
「ベンチは 5 回計測の中央値を採用」）。**`dispatch_auto` 自体は本ドキュメント記載の非劣化確認が
完了するまで `select_with_occupancy` を呼ばない**（`crate::gemm` モジュールドキュメンテーション
コメント参照。codex-review P1・PR #684）。

`cargo test -p fandhe-ai-backend-metal -- --ignored --nocapture` で `MetalOccupancyInfo::probe` 系テストの実測値
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

## 6. #747 判断（occupancy 選定式のサイズ帯条件分岐。**不採用〈#744 是正で吸収〉確定**）

イシュー #747「perf(backend-metal): occupancy 選定式のサイズ帯条件分岐（512 で +45.6%・2048 で
−5.6% を実測）」の判断記録。

### 実機比較結果（2026-08-19・M4 Max・`gemm_bench` occupancy 比較セクション・5 回中央値）

| size | new_over_old（新/旧） | 判定 |
|------|--------------------------|------|
| 512  | +45.6%（32×32 縮退により大幅改善） | 改善 |
| 1024 | ±1% 未満 | 非劣化 |
| 2048 | −5.6%（劣化） | 要判断 |
| 4096 | ±1% 未満 | 非劣化 |

この結果だけを見ると「512 のみ occupancy 縮退を有効化し、2048 は無効化する」サイズ帯条件分岐が
必要に見えるが、以下の理由で **`dispatch_auto` への組み込みは不採用**と判断する。

### 判断: #744 是正による吸収

イシュー #744（PR #760）は `tile::select` の段 1（形状判定）を是正し、`m == n == k` かつ
`m <= 4096`（真の正方立方・実測範囲内）の場合に **occupancy 縮退を経ずに** `CANDIDATES[3]`
（32×32/bk16/staged）を直接返すようにした。この是正により:

- 512/1024/2048/4096 の正方立方帯域では、段 1 が既に `CANDIDATES[3]` を返すため
  `select_with_occupancy` の段 2（occupancy 縮退。縮退対象は大タイル系 `CANDIDATES[0..=2]`
  のみ）は適用対象から外れる。すなわち **本 issue の実測データが示していた「512 で 32×32 を
  選ぶと +45.6%」という利得は、#744 是正後は occupancy 縮退を経由せず段 1 自体が実現している**
- 2048 の −5.6% は、#744 是正後の判定式では旧（`select`）・新（`select_with_occupancy`）とも
  同一構成（32×32 staged）を選ぶ帯域の比較になるため、**同一構成同士の比較に現れた計測ノイズ**
  と解釈する（イシュー #746 の実測で `size=2048` のノイズは 2〜4% 程度であり、−5.6% はこの床を
  やや超えるが実装差のない比較のため測定誤差の範囲と判断する）
- `crates/backend-metal/src/tile.rs::select_with_occupancy_747_confirms_absorption_by_744_correction`
  （Linux で実行可能な純粋関数テスト）で、512/1024/2048/4096 の全帯域において
  `select_with_occupancy(size, size, size, Some(params))` と `select(size, size, size)` が常に
  同一構成を返すことを固定した

よって「小サイズ帯のみ occupancy 有効化」というサイズ帯条件分岐そのものが #744 是正により
不要になったと判断し、`dispatch_auto` への `tile::select_with_occupancy` 組み込みは
**不採用（#744 是正で吸収）で確定**する。`select_with_occupancy` 自体は削除せず、実測外の帯域
（縦長・横長・準正方大形状長方形・K 未実測正方出力・4096 超正方立方）では引き続き occupancy
縮退判定が生きたまま残る（`crates/backend-metal/src/tile.rs` コメント参照）。これらの帯域で
occupancy 縮退が有利という実測根拠は存在しないため、本判断では未適用のまま据え置く。

### 実機セッションへの委譲事項（Mac 実機セッション。実機ツリー #408 系）

本 PR（コメント・テスト・docs のみ・本番挙動不変）に対する確認は以下のとおり:

- **数値一致 ignored テスト全 pass の確認**（`cargo test -p fandhe-ai-backend-metal --release -- --ignored`）:
  本変更は `dispatch_auto`・カーネル・選択ロジックを一切変更していないため理論上非影響。形式確認
  として記入する。結果: （未計測）
- **512 帯改善の floor bench 反映確認**（`cargo run -p fandhe-ai-backend-metal --release --example
  gemm_f32_prepared_bench`）: #744 是正（既マージ）による改善が floor bench に反映されている
  ことの確認。結果: （未計測。`docs/perf/metal-tile-select-correction.md`「floor bench 参考記録」
  節と同一の記入対象）
- **準正方長方形帯域（縦横比 2 未満・`m != n`）の候補比較実測**: `examples/gemm_bench.rs` の
  候補比較セクションで `(1536,1024,1024)`・`(3072,2048,2048)` 等を計測し、occupancy 縮退が
  この帯域で有利かを判断する材料とする（`docs/perf/metal-tile-select-correction.md`
  「準正方長方形帯（未計測）」節と同一の追跡対象。本判断とは独立の後続実測）。結果: （未計測）

## 7. スコープ外（`out-of-scope-tracking.md` に従い本 PR に混入させない）

- 実機実測値の記入・係数（6/9・`SMEM_GROUPS_PER_CORE_CAP`）の M4 Max 向け確定 → Mac 実機セッション
  （実機ツリー #408 系）
- `examples/gemm_diagnosis.rs` のローカル算出式のクレート内 API への置換（#541 doc §4） → 判定挙動へ
  影響しないため本 PR では見送り、既存の追跡記述を維持
- バッチ次元（MFA の `batchDimension`）導入・f16 系係数（`IDEAL_GROUPS_MULTIPLIER_ALL_16BIT`）の運用
- バックエンド抽象層の経路選択（#67/#68 レイヤ）への occupancy 反映
