# Metal GEMM occupancy 目標算出（GPU コア数 × threadgroup memory 予算）（#541）

イシュー #541「perf(backend-metal): GPU コア数と threadgroup memory 予算から occupancy 目標を算出する
仕組みを追加」の記録。親 #480「GEMM 最適化の計測前提確定・実機プローブ・ボトルネック診断」の
D-7a（D-7 の分割: `docs/perf/metal-gemm-bottleneck-diagnosis.md` 冒頭リスト参照）として、
`tile::select()` が扱うタイル候補ごとに「コア数 × 係数」（MFA〈metal-flash-attention〉型の経験式）と
「threadgroup memory 予算による同時常駐数上限」を組み合わせた occupancy 目標（`idealGroups`）を
算出する機構を追加する。**`select()` への組み込み・タイル 2 段階切替の閾値運用は後続イシュー #542 の
スコープであり、本イシューは算出機構のみを実装する。**

## 状態: 算出機構・純粋関数の実装とテストは完了（Linux worktree）。**GPU コア数・SMEM 上限の実機取得値は未記入（Mac 実機実行待ち）**

本実装環境は Linux のため、macOS 実機依存の値取得（`crate::device::probe_gpu_core_count`・
`MTLDevice::maxThreadgroupMemoryLength`）は `#[ignore]` テストの雛形整備までに留める（#487・#532 と
同じ前例）。§3「実機記録テンプレート」は Mac 実機セッション（実機ツリー #408 系）で記入する。

## 1. 算出式

### 1.1 `actual_groups`（`crate::tile::actual_groups`）

実際に起動される threadgroup 数:

```
actual_groups(m, n, cfg) = ceil(m / cfg.bm) * ceil(n / cfg.bn)
```

`examples/gemm_diagnosis.rs::analytics::analyze` の `actual_groups` 算出（#487）と同一式（本イシューで
クレート内 API として一本化し、算式のドリフトを防ぐ。§4「`gemm_diagnosis.rs` との重複解消」参照）。

### 1.2 `smem_groups_per_core`（`crate::tile::OccupancyParams::smem_groups_per_core`）

1 コアに同時常駐できる threadgroup 数を threadgroup memory 予算から求める（TileKernels 型: MLX の
`SM` あたり同時実行ブロック数のヒューリスティックと同種）:

```
smem_groups_per_core(cfg) =
    SMEM_GROUPS_PER_CORE_CAP                                    if !cfg.staged
    min(max_threadgroup_memory_bytes / cfg.shared_mem_bytes(),
        SMEM_GROUPS_PER_CORE_CAP)                                if cfg.staged
```

`staged=false`（device メモリから直接 `simdgroup_load` する経路）は threadgroup memory を消費しない
ため SMEM 制約が存在せず、上限キャップ（[`SMEM_GROUPS_PER_CORE_CAP`] = 16。参照実装〈TileKernels〉
由来の経験的上限）をそのまま返す。

### 1.3 `ideal_groups`（`crate::tile::OccupancyParams::ideal_groups`）

コア飽和目標 threadgroup 数（MFA 経験式 `idealGroups = gpu_core_count * multiplier` に、
threadgroup memory 予算による上限を組み合わせた一般化）:

```
ideal_groups(multiplier, cfg) =
    gpu_core_count * min(multiplier, smem_groups_per_core(cfg))
```

`gpu_core_count == 0`・`multiplier == 0`・積のオーバーフローは `None`（fail-closed。
`examples/gemm_diagnosis.rs::parse_device_profile_override` の CLI 検証と同方針）。

### 1.4 `is_underoccupied`（`crate::tile::is_underoccupied`）

```
is_underoccupied(actual, ideal) = actual <= ideal
```

境界一致（`actual == ideal`）は under-occupied 側に倒す（fail-safe）。`select()` の閾値運用への組み込み
は #542 のスコープ。

## 2. 既定係数の出典と注意（**MFA 数値の直採用禁止**）

| 定数 | 値 | 出典 |
|------|----|------|
| `IDEAL_GROUPS_MULTIPLIER_F32` | 6 | MFA の FP32 系経験式。`examples/gemm_diagnosis.rs::analytics::DeviceProfile::M4_MAX`（#487）と同一出典 |
| `IDEAL_GROUPS_MULTIPLIER_ALL_16BIT` | 9 | MFA の全 16bit 系経験式（`backend-metal` は現状 f32 GEMM が主経路のため将来 f16 系カーネル向けに先行定義） |
| `SMEM_GROUPS_PER_CORE_CAP` | 16 | TileKernels 型の同時常駐数上限（経験的キャップ） |

**注意（イシュー #541 本文明記）**: MFA の具体数値（48×48×24 等のタイル・係数 6/9）は Apple7/8/9 の
実測値であり **M4 Max へそのまま採用しない**。上記はいずれも「実機実測で確定させる既定パラメータ」の
初期値であり、Mac 実機セッションでの `examples/gemm_bench.rs`／`examples/gemm_diagnosis.rs` 実測
（#542 のスコープ）で見直しうる。

## 3. GPU コア数・SMEM 上限の取得手順・実機記録テンプレート

### 3.1 取得手段

`MTLDevice` に GPU コア数を取得する公開 API は存在しないため（#487 codex-review P1 で「機種識別子から
の対応表推定」は却下済み）、`crate::device::probe_gpu_core_count`（イシュー #541）は IOKit IORegistry
の `AGXAccelerator` サービスが公開する `gpu-core-count` プロパティを読む（MFA／`applegpuinfo` と同方式。
機種識別子からの推定ではなく実機からの実測読み取りのため #487 の却下理由には該当しない）。

threadgroup memory 上限は既存の safe API（`MTLDevice::maxThreadgroupMemoryLength()`）をそのまま使う
（`crate::device::MetalOccupancyInfo::probe`）。

### 3.2 実行コマンド（Mac 実機）

```sh
cargo test -p backend-metal -- --ignored
```

`device::tests::probe_gpu_core_count_returns_positive_value_on_apple_silicon`・
`device::tests::metal_occupancy_info_reports_smem_upper_bound_at_least_32kib` が実測値を `println!` で
出力する（`cargo test ... -- --ignored --nocapture` で標準出力を確認する）。

### 3.3 記録表（要実測記入）

| 機種 | `sysctl -n hw.model` | `probe_gpu_core_count()` | `maxThreadgroupMemoryLength` | 記録日 |
|------|----------------------|---------------------------|-------------------------------|--------|
| M4 Max（実機検証環境。`docs/real-hardware-verification-env.md` §1） | `Mac16,6`（`docs/perf/metal-gemm-dynamic-tile.md:53` 実測記録） | **要実測記入**（期待値: 40。`docs/perf/metal-gemm-bottleneck-diagnosis.md` §3 実測前提値と同一出典） | **要実測記入**（期待値: 32768 以上） | — |

## 4. `gemm_diagnosis.rs` との重複解消（挙動不変の確認）

`examples/gemm_diagnosis.rs::analytics::analyze` の `actual_groups`／`ideal_groups` 算出は本イシュー
時点でローカル実装のまま温存した（クレート内 API〈`tile::actual_groups`・
`OccupancyParams::ideal_groups`〉への置換は、出力値が変わらないことの突き合わせ確認〈本イシュー計画
§3.3〉を安全側に倒して見送った）。両実装の算出式は本ドキュメント §1.1・§1.3 の式と数学的に同一であり
（`ideal_groups` は `gemm_diagnosis.rs` 側が SMEM 制約を掛けない素の `gpu_core_count * multiplier` のみ
を算出する点が本モジュールとの差分）、算式ドリフトの実害はない。置換自体は #542 以降、SMEM 制約込みの
occupancy 判定を `gemm_diagnosis.rs` にも反映するタイミングで行う想定。

## 5. スコープ外（#542 ほかへ委ねる事項）

- ~~`select()` への occupancy 判定の組み込み・タイル 2 段階切替・閾値の実測確定~~ →
  イシュー #542（`crate::tile::select_with_occupancy`）で実装完了。詳細は
  `docs/perf/metal-gemm-occupancy-select.md` を参照
- ~~GPU コア数が取得不能（`None`）な場合の `select()` 側フォールバック方針の確定~~ →
  #542 で確定（`params: None`／`actual_groups`／`ideal_groups` のいずれかが `None` なら形状のみの
  判定へ fail-safe フォールバック。`docs/perf/metal-gemm-occupancy-select.md` §2）
- 実機での実測値記入（§3.3）・係数（6/9・`SMEM_GROUPS_PER_CORE_CAP`）の M4 Max 向け確定 →
  引き続き Mac 実機セッション（実機ツリー #408 系）へ委ねる（#542 でも未完了。
  `docs/perf/metal-gemm-occupancy-select.md` §5 記録テンプレート）
- バッチ次元（MFA の `batchDimension`）の導入
- `examples/gemm_diagnosis.rs` の算式をクレート内 API へ実際に置換すること（§4）（#542 でも見送り。
  `docs/perf/metal-gemm-occupancy-select.md` §6）
