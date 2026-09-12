# CPU GEMM 小形状スレッドプール上限（イシュー #1575）

## 背景

低レイヤー診断（#1574。`docs/perf/lowlayer-diagnosis-2026-09-12.md` §3・
§7 A-1a）で、Apple M4 Max の batch 64 学習・推論（小形状 GEMM 主体）は
プロセス全体の `RAYON_NUM_THREADS` をグローバルプール既定値（16）から
`4` へ絞ると train 約 1.47 倍・infer 約 1.30 倍速いことが観測された。

既存機構（`crate::thread_limit`〈コア種別ベース。#1363/#1364 実測により
`BIG_CORE_LIMIT_ENABLED=false`〉・`GEMM_THREADING_THRESHOLD`/
`should_serialize`〈#811/#1027・`#[cfg(test)]` 限定・本番未結線〉）は
いずれも本目的に使えない（判定軸・粒度が異なる。設計理由の詳細は
`crates/backend-cpu/src/small_shape_thread_cap.rs` モジュール冒頭コメント）。

本イシューは **M×N×K の仕事量に基づき、M4 Max（自機判定）でのみ小形状
GEMM を小さい専用 rayon プールで実行する**新機構
（`crates/backend-cpu/src/small_shape_thread_cap.rs`）を、既存 2 機構を
置き換えずに新設する。

## 設計

`docs/spec/` 変更なし・依存追加なし・`unsafe` 追加なし。

- `SMALL_SHAPE_CAP_ENABLED`（単一 const ゲート。#1313
  `TWO_D_DYNAMIC_PRODUCTION_ENABLED` と同型）
- `should_cap(m, n, k)`: `m * n.max(N_CLAMP) * k < SMALL_SHAPE_CAP_MAX_WORK`
  の純関数判定
- `eligible_from_probe`: P/E 非対称構成（`hw.perflevel0.logicalcpu <
  hw.logicalcpu`）かつ `machdep.cpu.brand_string` が allowlist
  （`"Apple M4 Max"` のみ）に完全一致する場合のみ有効
- `run_capped`: 条件を満たす場合のみ専用 `rayon::ThreadPool`
  （`SMALL_SHAPE_CAP_THREADS` スレッド）へ `install` する。`RAYON_NUM_THREADS`
  明示設定時は上書きしない（`crate::thread_limit` と同じ契約）
- 結線箇所: `gemm_blis_parallel_with_transpose`／`gemm_blis_bias_act_parallel`
  の `dispatch_two_d_dynamic` 呼び出しのみ（forward・VJP の d_input/d_weight
  を含む本番全入口を網羅。`linear_forward_device`／`gemm_resident_rhs` 等は
  いずれかを内部で呼ぶため自動的にカバーされる）

bit 完全一致契約: 本機構は並列度のみを変え GEMM カーネル本体を変えない
ため、専用プール実行はグローバルプール実行と常に bit 完全一致する
（`small_shape_thread_cap::tests::dedicated_pool_execution_matches_global_pool_bit_exact`
で検証。既存の `gemm_blis_parallel_*_matches_naive_bit_exact_across_thread_pools`
系テストが検証する不変条件と同種）。

## 事前登録規則

実装着手前に固定した判定規則の全文はイシュー #1575 のコメントを正とする:
<https://github.com/Fandhe-AI/fandhe-ai/issues/1575#issuecomment-5644510967>

## Phase 0（M4 Max・定数確定スイープ）

**実施内容**: `examples/small_shape_cap_sweep.rs`
（`cargo run --release -p fandhe-ai-backend-cpu --example
small_shape_cap_sweep -- <global|dedicated:N>`）を用い、学習 5 形状
（64×256×784・64×10×256・784×256×64 相当・64×256×10 相当・
256×10×64 相当。実際の VJP 転置入口〈`gemm_blis_parallel_nt`/`_tn`〉では
なく同一 `(m,n,k)` の `gemm_blis_parallel`〈NN〉で計測している点に注意
——`should_cap` の判定は転置パターンに依存せず `(m,n,k)` の積のみで
決まるため、専用プール切替の性能効果を近似する上で妥当な代理計測とした）
＋交差確認用正方 128/256/512 を、腕 {`global`（既定 16 スレッド）・
`dedicated:{2,4,6,8}`・プロセス全体 `RAYON_NUM_THREADS=4`（参照）} ×
各 5 プロセス起動で計測した（record_only。専有ゲートなし。共有負荷下
`load average` 6.4〜9.1 → 計測後 4.7〜7.4。`docs/perf/logs/
cpu-gemm-small-shape-thread-cap-1575/`）。

**checksum**: 全腕・全形状で `f64` 逐次和のビット表現が完全一致
（`aggregate_phase0.md` 冒頭。bit 完全一致契約を計測レベルでも確認）。

**選択規則適用**: 学習 5 形状すべてで off 比 `ratio<=1.00` を満たしたのは
`dedicated:6` のみ（`dedicated:2`／`4` は `train_64x256x784_nn` で
`ratio>1.00`〈1.52／1.056〉・`dedicated:8` は同形状で `ratio=1.0272` に
より不成立）。成立候補が 1 つのみのため幾何平均最小の選択は自動的に
`dedicated:6`（幾何平均 ratio ≈0.62）に確定した。

→ **`SMALL_SHAPE_CAP_THREADS = 6`**

`dedicated:6` の交差確認形状 ratio は 正方 128: 0.7995・正方 256: 0.7902・
**正方 512: 1.0980（後退）**。事前登録の tie-break（学習最大仕事量
12,845,056 を含み・後退開始仕事量 134,217,728 を含まない 2 のべき乗の
うち最小＝最も保守的なもの）に従い `1 << 24`（16,777,216）を採用した
（`1 << 25` も条件を満たすが tie-break により不採用。正方 256
〈16,777,216〉はこの境界とちょうど等しく cap 対象外〈グローバルプール
のまま〉になるが、同形状は元々改善方向だったため後退はしない）。

→ **`SMALL_SHAPE_CAP_MAX_WORK = 1 << 24`（16,777,216）**

実測全文は `docs/perf/logs/cpu-gemm-small-shape-thread-cap-1575/
aggregate_phase0.md`・生ログ（`phase0_*.log`）・`env_info.txt` を参照。

Phase 0 完了後に上記 2 定数を確定・コミットし、Phase 1 の結果を見て
再調整しない（事前登録規則）。

## Phase 1（framework-compare・同一バイナリ on/off）

事前登録手順: `SMALL_SHAPE_CAP_ENABLED=true` へ切り替えたバイナリを
`env -u RAYON_NUM_THREADS`（after・cap 有効）／`RAYON_NUM_THREADS=16`
（before・明示設定により cap 無効）で interleave 計測する。

**実施内容**: `scripts/bench/framework-compare/` で
`SMALL_SHAPE_CAP_ENABLED=true` にした本リポの `crates/facade` を
`[patch.crates-io.fandhe-ai]`（path patch）で `bench-fandhe` へ差し込んで
ビルドし（`cargo metadata` で `fandhe-ai` の解決元が path 依存
〈`source: None`〉であることを確認済み）、判定対象 4 セル（train／infer
cpu fresh／reuse）＋参考 6 セル（gemm cpu 512/1024/2048 fresh／reuse。
本機構は `dispatch_two_d_dynamic` 呼び出し経路のため gemm タスクにも
到達しうる。参考記録）を before/after 各 5 run（run 単位で実行順反転）
interleave 計測した（record_only。計測中 load average 22〜23 の共有負荷
下。`docs/perf/logs/cpu-gemm-small-shape-thread-cap-1575/phase1/`・
`env_info_phase1.txt`）。

**結果**（`aggregate_phase1.md`。ratio = after 5-run 中央値 / before
5-run 中央値）:

| セル | before (s) | after (s) | ratio | 判定対象 |
|---|---|---|---|---|
| train:fresh | 0.000781459 | 0.000805437 | **1.0307** | ○ |
| train:reuse | 0.000940772 | 0.000869563 | 0.9243 | ○ |
| infer:fresh | 0.000201937 | 0.000155605 | 0.7706 | ○ |
| infer:reuse | 0.000190146 | 0.000162292 | 0.8535 | ○ |
| gemm:fresh:512 | 0.000671271 | 0.000660709 | 0.9843 | 参考 |
| gemm:reuse:512 | 0.000638167 | 0.000643062 | 1.0077 | 参考 |
| gemm:fresh:1024 | 0.002987167 | 0.002976042 | 0.9963 | 参考 |
| gemm:reuse:1024 | 0.004357084 | 0.003889521 | 0.8927 | 参考 |
| gemm:fresh:2048 | 0.019987500 | 0.023424041 | 1.1719 | 参考 |
| gemm:reuse:2048 | 0.023663562 | 0.025603458 | 1.0820 | 参考 |

checksum は全 10 セルで before/after 完全一致（bit 完全一致契約を
本計測でも確認）。

**判定**: `train:fresh` の `ratio=1.0307`（>1.00）が事前登録規則
「判定対象 4 セルすべてで `ratio<=1.00`」に抵触した。
`train:reuse`／`infer:fresh`／`infer:reuse` の 3 セルは基準を満たしたが、
規則は 4 セル全一致を要求するため **REJECT** と確定する
（`SMALL_SHAPE_CAP_ENABLED = false` 維持。機構自体は削除せず維持）。

共有負荷下（record_only）での単発計測であり `train:fresh` の後退幅
（約 +3%）はノイズ帯の可能性もあるが、事前登録規則は結果を見て緩和・
再解釈しない方針のため、本判定を正式結果として記録する（再測定・
再判定は本 PR のスコープ外。必要であれば別イシューで低負荷環境
再計測を提案する）。

→ **verdict = REJECT**（`SMALL_SHAPE_CAP_ENABLED = false` 確定）

## GB10（DGX Spark）

`cfg(target_os = "macos")` により構造的に非到達。Linux での
`eligible() == false` は `small_shape_thread_cap::tests::
small_shape_cap_report_bounds`（`#[cfg(not(target_os = "macos"))]` 分岐）
が構造的根拠として担保する。

**実機健全性確認（2026-09-12）**: GB10 実機（`local.fandhe.spark-dbd9`。
rsync 転送・`PATH=$HOME/.cargo/bin:/usr/local/cuda/bin:$PATH`）へ到達し
`cargo test -p fandhe-ai-backend-cpu --lib small_shape` を実行、14 件
全 pass を確認した（`eligible=false` を含む全契約テストが Linux 実機上
でも成立）。機構が `SMALL_SHAPE_CAP_ENABLED=false`（Phase 1 REJECT 確定
済み）のため本番挙動への影響は元々なく、`cfg(target_os = "macos")` の
構造的非到達を実機上で直接確認する健全性記録に留める（framework-compare
train/infer/gemm cpu の before/after 比較は、機構が既定 OFF かつ GB10 が
構造的非到達のため意味を持たず実施しない）。
