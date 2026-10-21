# GB10 小形状 GEMM 大コア affinity（#1576）A/B 実測記入欄

設計は `docs/backend-cpu-gb10-affinity-design.md`（イシュー #1576）。既定 `GB10_AFFINITY_ENABLED = false`（`crates/backend-cpu/src/gb10_affinity.rs`）。

## 実測状況

2026-10-21 に GB10 実機で A/B を実施した（イシュー #3187。ラベル `2117-gb10-3187`・正式系列）。結果と verdict は末尾の記入欄を参照。
実装（検出ロジック・unsafe FFI・専用プール・ルーティング・単体テスト）は既定 OFF のまま維持している（verdict が REJECT のため）。

## 事前登録判定規則（実測前に固定・事後緩和しない）

- 5 run 中央値・checksum 完全一致・非後退 `ratio(on/off) <= 1.00`
- **対象セル**（診断で改善が確認された形状。§1 参照）: `bench-fandhe --task train/infer --device cpu --size 64 --mode fresh/reuse`（4 セル）
- **ガードセル**（`TWO_D_DYNAMIC_PRODUCTION_ENABLED` の大形状本番経路から性能を奪っていないことの確認）: `bench-fandhe --task gemm --device cpu --size 256/512/1024/2048/4096 --mode fresh/reuse`（10 セル）。1 セルでも `ratio > 1.00` なら**全体を REJECT** とし、閾値の事後調整（ガードセルだけ都合よく除外する等）は行わない
- 二値構成（`GB10_AFFINITY_ENABLED = false/true` の 2 バイナリビルド。`on-arm.patch` として保存。#1301/#1481 と同型手順）
- 内部ホスト名は含めない。専有ゲート or record_only を明記する

## 実行基盤（イシュー #2117）

判定規則の固定版（14 セル・専有ゲート load1<1.0・機構発火の前提）は `docs/perf/logs/cpu-gb10-affinity-ab-2117/RULE.txt`、実行は `scripts/bench/framework-compare/run_ab_gb10_affinity_cpu.sh`（`compare_gemm_ab.py --sizes affinity`）。上記の判定規則は緩めていない。実測は 2026-10-21 に実施済み（末尾の記入欄）。

注記: `GB10_AFFINITY_MAX_WORK`（32*1024*1024）に対し gemm N=256（`256^3 = 16.7M`）はルーティング対象（処置セル）で、非ルーティングの真のガードは N=512 以上。判定規則は 14 セルすべてに同一適用する。

## §1 対象形状の根拠

`docs/perf/lowlayer-diagnosis-2026-09-12.md` §3（実機実測。出典 `docs/perf/logs/lowlayer-diagnosis-2026-09-12/dgx/rayon-sweep.jsonl`／`rayon-sweep-pinned.jsonl`）。

| 条件 | train reuse（size=64・3 起動中央値） |
|---|---|
| 無 pin T20（全コア） | 1.029 s |
| 無 pin T10（スレッド数のみ大コア数に一致） | 2.359 s |
| 大コア pin T10（`taskset -c 5-9,15-19`） | 0.857 s |

## 実測記入欄（GB10・2026-10-21・イシュー #3187）

条件: 正式系列（`series=official`・`precheck=assert`）。5 round すべて専有ゲート通過（`gate=pass gate_ok=1`・load1 0.72〜0.98。
`logs/cpu-gb10-affinity-ab-2117/gb10/load_gate-2117-cpu-2117-gb10-3187.log`）。機構発火の前提は成立
（before: `enabled=false`・`pool_active=false`／after: `enabled=true`・`env_override=false`・大コア 10 個
`[5,6,7,8,9,15,16,17,18,19]`・`pool_active=true`）。計測コミット `d56c64ba8`（after は `on-arm.patch` のみ適用）。
各セルは 5 round 中央値（us／ms）。ratio = after/before。checksum は 14 セルすべて完全一致。
出典: `logs/cpu-gb10-affinity-ab-2117/gb10/compare-{gemm,train,infer}-2117-cpu-2117-gb10-3187.md`。

| セル | before（off） | after（on） | ratio | checksum | 判定 |
|---|---|---|---|---|---|
| train fresh size=64 | 1.124 ms | 1.198 ms | 1.0662 | 完全一致 | 後退 |
| train reuse size=64 | 851.3 us | 946.1 us | 1.1115 | 完全一致 | 後退（5 round すべて > 1.00） |
| infer fresh size=64 | 180.5 us | 134.8 us | 0.7470 | 完全一致 | 非後退 |
| infer reuse size=64 | 186.6 us | 143.7 us | 0.7704 | 完全一致 | 非後退 |
| gemm fresh N=256（処置セル） | 186.6 us | 155.9 us | 0.8352 | 完全一致 | 非後退 |
| gemm reuse N=256（処置セル） | 392.0 us | 391.2 us | 0.9979 | 完全一致 | 非後退 |
| gemm fresh N=512 | 924.8 us | 1.004 ms | 1.0857 | 完全一致 | 後退 |
| gemm reuse N=512 | 1.698 ms | 1.756 ms | 1.0341 | 完全一致 | 後退 |
| gemm fresh N=1024 | 3.743 ms | 3.570 ms | 0.9539 | 完全一致 | 非後退 |
| gemm reuse N=1024 | 5.320 ms | 5.343 ms | 1.0043 | 完全一致 | 後退 |
| gemm fresh N=2048 | 15.843 ms | 16.554 ms | 1.0449 | 完全一致 | 後退 |
| gemm reuse N=2048 | 26.754 ms | 26.734 ms | 0.9993 | 完全一致 | 非後退 |
| gemm fresh N=4096 | 122.821 ms | 121.774 ms | 0.9915 | 完全一致 | 非後退 |
| gemm reuse N=4096 | 123.204 ms | 121.867 ms | 0.9892 | 完全一致 | 非後退 |

verdict: **REJECT**（RULE.txt「正式系列で 1 セルでも中央値 ratio > 1.00」。該当 6 セル）。

- train size=64（#1576 の対象セルのうち）が fresh／reuse とも悪化した。train reuse は run 内比が 5 round すべて 1.00 超
  （1.0838・1.1374・1.1527・1.1484・1.0932）
- 処置セル N=256 と infer は改善または同等だが、RULE は 14 セルに同一規則を適用する（ガードセルだけの除外や閾値の事後調整はしない）
- `GB10_AFFINITY_ENABLED = false` を維持する。既定反転（#3227）の前提は成立していない。
  `GB10_AFFINITY_MAX_WORK` の再チューニング等は RULE の対象外（別イシュー）
- 注意: 同セッションの reuse/fresh 帰属（`docs/perf/cpu-gemm-reuse-fresh-inversion.md` §6）とは別の計測。
  train の悪化要因は本計測では切り分けていない
