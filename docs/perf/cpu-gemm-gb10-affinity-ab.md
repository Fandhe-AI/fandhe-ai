# GB10 小形状 GEMM 大コア affinity（#1576）A/B 実測記入欄

設計は `docs/backend-cpu-gb10-affinity-design.md`（イシュー #1576）。既定 `GB10_AFFINITY_ENABLED = false`（`crates/backend-cpu/src/gb10_affinity.rs`）。

## 実測状況

本エージェント実行環境に DGX Spark GB10 実機への到達手段が無いため、性能 A/B は**未実施**。実装（検出ロジック・unsafe FFI・専用プール・ルーティング・単体テスト）は完了し、既定 OFF のまま GB10 実機実測を伴う後続セッションへ引き継ぐ。

## 事前登録判定規則（実測前に固定・事後緩和しない）

- 5 run 中央値・checksum 完全一致・非後退 `ratio(on/off) <= 1.00`
- **対象セル**（診断で改善が確認された形状。§1 参照）: `bench-fandhe --task train/infer --device cpu --size 64 --mode fresh/reuse`（4 セル）
- **ガードセル**（`TWO_D_DYNAMIC_PRODUCTION_ENABLED` の大形状本番経路から性能を奪っていないことの確認）: `bench-fandhe --task gemm --device cpu --size 256/512/1024/2048/4096 --mode fresh/reuse`（10 セル）。1 セルでも `ratio > 1.00` なら**全体を REJECT** とし、閾値の事後調整（ガードセルだけ都合よく除外する等）は行わない
- 二値構成（`GB10_AFFINITY_ENABLED = false/true` の 2 バイナリビルド。`on-arm.patch` として保存。#1301/#1481 と同型手順）
- 内部ホスト名は含めない。専有ゲート or record_only を明記する

## §1 対象形状の根拠

`docs/perf/lowlayer-diagnosis-2026-09-12.md` §3（実機実測。出典 `docs/perf/logs/lowlayer-diagnosis-2026-09-12/dgx/rayon-sweep.jsonl`／`rayon-sweep-pinned.jsonl`）。

| 条件 | train reuse（size=64・3 起動中央値） |
|---|---|
| 無 pin T20（全コア） | 1.029 s |
| 無 pin T10（スレッド数のみ大コア数に一致） | 2.359 s |
| 大コア pin T10（`taskset -c 5-9,15-19`） | 0.857 s |

## 実測記入欄（未実測）

| セル | before（off） | after（on） | ratio | checksum |
|---|---|---|---|---|
| train fresh size=64 | 未実測 | 未実測 | — | — |
| train reuse size=64 | 未実測 | 未実測 | — | — |
| infer fresh size=64 | 未実測 | 未実測 | — | — |
| infer reuse size=64 | 未実測 | 未実測 | — | — |
| gemm fresh/reuse N=256/512/1024/2048/4096（ガードセル 10 個） | 未実測 | 未実測 | — | — |

verdict: **未確定（undetermined。実測未実施）**
