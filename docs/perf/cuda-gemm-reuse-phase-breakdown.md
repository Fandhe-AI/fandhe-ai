# CUDA GEMM reuse 計測境界のフェーズ分解（H2D／カーネル／D2H／同期。イシュー #1182）

## §1 位置づけ

`#1142`（`docs/perf/cuda-gemm-candle-gate-remeasurement.md` §4.3・§8）は、N=1024/2048/4096
の `gemm --mode reuse` が candle 比未達（#1031 未達）のまま終わった原因について、
「reuse の計測境界に残る H2D／D2H／同期の固定費が candle 比を押し下げている」と
**推定**したまま確定していなかった。一方 `#1137`（`docs/perf/cuda-wmma-f16-perf-triage.md`
隣接。`cuda_floor_bench`）はカーネル単体（launch-only）計測で candle を上回る TFLOPS を
記録している。本ドキュメントはこの 2 つの計測の間を、`train --phases`（#1009・#1010）と
同じ方法論（公開 API 呼び出し境界でのフェーズ分解 + 非公開 API での内部分解）で埋め、
#1142 §4.3 の推定を実測で検証・精緻化する。

**結論を先に述べる**: #1142 §4.3 の推定は**部分的に不正確**だった。`matmul` 区間
（H2D A/B・カーネル実行・D2H・ストリーム同期を含む）単体は candle の fresh 全体より
**高速**（§7）であり、reuse 総計が candle 比未達になる主因は H2D／D2H そのものではなく、
**ベンチハーネス自身が追加する `host_copy`（二重ホストコピー）と `checksum`（全要素和）
の計算コストが `iter_total` の 40〜58% を占めること**である（§6）。

## §2 計測環境・プロトコル

- 実機: DGX Spark GB10（sm_121）。`docs/perf/logs/cuda-gemm-reuse-phase-1182/env_info.txt`
  参照（実ホスト名は含めない）。計測前後で `nvidia-smi --query-gpu=utilization.gpu` 0% を
  確認（GPU 競合なし）
- driver 580.173.02・CUDA 13.0（nvcc V13.0.88）・rustc/cargo 1.97.0
- 転送元: main コミット `b5a2cb681b9a27d4506c71c612c00d0b2fb96e47` + 本 PR の未コミット
  変更ツリー（`docs/real-hardware-verification-env.md` §3〜4 の rsync 方式。単一コミットは
  本実測の後に作成される。#1025/#1139 と同じパターン）
- 2 層構成:
  - **Layer A**（公開 API 境界。`bench-fandhe gemm --mode reuse --phases`。framework-compare
    ピン `fandhe-ai =0.6.0`〈crates.io 公開版〉固定）: `matmul`／`to_tensor`／`host_copy`／
    `checksum`／`iter_total` の 5 区間。N=1024/2048/4096 × 5 回計測（各 20 warmup + 20 測定
    の中央値）
  - **Layer B**（`crates/backend-cuda` 非公開 API。`gemm_reuse_phase_diag_tests`。HEAD ツリー）:
    `h2d_a`／`h2d_b`／`alloc_c`（プール経由）／`launch_issue`（投入のみ）／`kernel_wait`
    （明示 `stream.synchronize()`）／`d2h`（`clone_dtoh` + `synchronize`）／`host_copy`
    （`to_vec()`）の 7 区間。N × 変種（`Select`＝本番同一の形状条件付き自動選択、
    `Classic`＝常に classic 固定）の 6 組合せ × 5 回計測（各 20 warmup + 20 測定の中央値）
- 参考系列（HEAD path patch によるビルド）は時間の都合で本ラウンドでは未実施
  （§9 AC-3 に記載。0.6.0 と HEAD の `kernels.rs`〈TILED_F32〉に差分があるため
  `Classic` 変種を近似比較用として用意した — §5）
- ログ: `docs/perf/logs/cuda-gemm-reuse-phase-1182/`（`layerA-phases-N{1024,2048,4096}.log`・
  `layerA-ac2.log`・`layerB-run{1..5}.log`・`env_info.txt`）

## §3 Layer A 実測（正式系列 `fandhe-ai =0.6.0`）

N ごとの phase 中央値（5 回計測の中央値。単位 ms、括弧内は 5 回の生値）:

| N | matmul | to_tensor | host_copy | checksum | iter_total |
| --- | --- | --- | --- | --- | --- |
| 1024 | 0.582（0.571, 0.588, 0.581, 0.582, 0.581） | ~0 | 1.272（1.437, 1.272, 1.232, 1.355, 1.102） | 0.535（0.533, 0.540, 0.533, 0.540, 0.535） | 2.402（2.537, 2.402, 2.346, 2.475, 2.235） |
| 2048 | 3.214（3.214, 3.230, 3.206, 3.209, 3.216） | ~0 | 4.148（4.061, 4.033, 4.211, 4.253, 4.148） | 2.131（2.131, 2.134, 2.130, 2.131, 2.129） | 9.514（9.514, 9.457, 9.620, 9.620, 9.506） |
| 4096 | 38.227（32.662, 38.227, 36.881, 39.862, 41.056） | ~0 | 18.545（14.387, 18.545, 16.793, 19.841, 18.566） | 8.635（8.628, 8.605, 8.635, 8.635, 8.649） | 65.088（56.189, 65.088, 62.307, 68.135, 69.254） |

`init_s`（デバイス/tape 初期化。1 回のみ）: N=1024 約 0.40〜0.54 s・N=2048 約 0.41 s・
N=4096 約 0.44〜0.47 s（既存 `run_gemm_reuse` の `init_s` と同オーダー）。

**AC-2（挙動不変）**: `--phases` なしの `gemm --mode reuse` を同一セッションで実行し
（`results-dgx-gemm-nonphases-ac2.jsonl`）、checksum が完全一致することを確認した
（N=1024: `-1855.597736`、N=2048: `-6016.774008`、N=4096: `-25768.747284`。いずれも
phases 版 1 ラン目と bit 単位で一致）。JSONL のキー集合（`gflops` の有無・`phase`/
`phase_index` の有無）も既存スキーマのまま変わらない。`run_gemm_reuse` 関数本体は本
イシューで一切変更していない（`git diff` で確認済み）。要素単位検証（`parity_*`）は
全 N・全反復で `fail_count=0`（厳密ゼロ）。

`summarize.py --strict` を新規節（`(a'')`）付きで実行し exit 0 を確認した
（`docs/perf/logs/cuda-gemm-reuse-phase-1182/` にはログのみを残し、`summary.md` §後述
に生成表を転記）。

## §4 Layer B 実測（`crates/backend-cuda` 内部分解。5 回計測の中央値、単位 ms）

| N | kernel | h2d_a | h2d_b | alloc_c | launch_issue | kernel_wait | d2h | host_copy |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1024 | Select | 0.0845 | ≈h2d_a | ~0 | ~0.003 | 0.1908 | 0.0752 | 1.4081 |
| 1024 | Classic | 0.0846 | ≈h2d_a | ~0 | ~0.003 | 0.3191 | 0.0755 | 1.3709 |
| 2048 | Select | 0.3043 | ≈h2d_a | ~0 | ~0.007 | 1.3132 | 0.2823 | 5.7229 |
| 2048 | Classic | 0.3032 | ≈h2d_a | ~0 | ~0.007 | 2.2644 | 0.2826 | 5.5644 |
| 4096 | Select | 1.1634 | ≈h2d_a | ~0 | ~0.008 | **13.6009**（12.24〜16.81 のばらつき） | **495.07**（26.0〜647 の二峰性） | 20.6074 |
| 4096 | Classic | 1.1623 | ≈h2d_a | ~0 | ~0.008 | **20.2845** | **528.77**（24.9〜655 の二峰性） | 20.3596 |

N=4096 の `d2h` は 5 回中 2〜3 回が 500〜650 ms、残り 2〜3 回が 25〜27 ms という**顕著な
二峰性**を示した（各回の生値は `docs/perf/logs/cuda-gemm-reuse-phase-1182/layerB-run{1..5}.log`
参照）。これは既知の「大容量バッファ per-call アロケーション＋転送の閾値サイズと二峰性」
（`docs/perf/cuda-large-buffer-percall-alloc-transfer-threshold.md`。イシュー #1169）と
整合する挙動であり、本ファイルの `d2h` 区間が反復ごとに**新規 `Vec<f32>`（未タッチページ）
を `clone_dtoh` の宛先にする**設計（本番 `readback` と同型）に起因すると考えられる。
N=1024/2048 では顕在化しない（64 MiB 未満のため閾値を下回る）。

## §5 突合

`kernel_wait`（Layer B）と #1137 `cuda_floor_bench`／#1136 classic baseline の
TFLOPS 換算値との比較（2·N³ / TFLOPS）:

| N | 変種 | kernel_wait 実測 | TFLOPS 換算（出典） | 差 |
| --- | --- | --- | --- | --- |
| 1024 | Select | 0.191 ms | 0.191 ms（#1137: 11.23 TFLOPS） | ほぼ一致 |
| 2048 | Select | 1.313 ms | 1.322 ms（#1137: 13.00 TFLOPS） | 1% 未満 |
| 4096 | Select | 13.601 ms | 13.45 ms（#1137: 10.22 TFLOPS） | 1% 程度 |
| 1024 | Classic | 0.319 ms | 0.318 ms（#1136: 6.75 TFLOPS） | ほぼ一致 |
| 2048 | Classic | 2.264 ms | 2.297 ms（#1136: 7.48 TFLOPS） | 1.5% 程度 |
| 4096 | Classic | 20.285 ms | 20.36 ms（#1136: 6.75 TFLOPS） | 1% 未満 |

`kernel_wait` は独立計測（#1136/#1137）と極めて良く一致しており、本診断テストの計時方式
（`launch_issue` の非同期投入直後ではなく明示 `stream.synchronize()` を挟んだ区間を
`kernel_wait` とする設計。ファイル冒頭コメント参照）の妥当性を裏付ける。

`h2d_a + h2d_b + alloc_c + launch_issue + kernel_wait + d2h`（Layer B。`host_copy` を除く
= Layer A `matmul` に対応する区間）と Layer A `matmul` の突合（N=1024/2048。N=4096 は
§4 の d2h 二峰性のため突合に使えない）:

| N | Σ Layer B（matmul 相当） | Layer A `matmul` | 差 |
| --- | --- | --- | --- |
| 1024（Select） | 0.566 ms | 0.582 ms | 3% 程度 |
| 2048（Select） | 3.163 ms | 3.214 ms | 2% 程度 |

N=1024/2048 では 2 層の独立計測が ±3% で一致しており、Layer B の分解が Layer A の
`matmul` 区間の内訳として妥当であることを確認した。N=4096 は Layer B 側の `d2h` 二峰性
（§4）により Σ Layer B（500〜570 ms オーダー）が Layer A `matmul`（38.2 ms）と大きく
乖離する。これは production 経路（`run_f32_kernel`。Layer A が経由する）と診断テスト
（Layer B）とで D2H の挙動が異なることを示唆しており、**N=4096 の D2H 内訳は本ラウンドの
計測方法では確定できない**（§8 スコープ外・§9 ユーザー判断事項）。

`Select` と `Classic` は 0.6.0/HEAD の `kernels.rs`（TILED_F32 ソース）差分（イシュー
#1137 の cp.async パイプライン分岐追加）を反映し、`kernel_wait` が N=2048/4096 で
`Select`（pipeline 経由）の方が `Classic` より 1.5〜1.7 倍速い。Layer A（0.6.0 固定）の
`matmul` は 0.6.0 の `select_tiled_f32_kernel`（pipeline 分岐なし。事実上 `Classic` 相当）
を経由するため、`Classic` 系列との突合がより正確な近似となる。

## §6 固定費の帰属（#1142 §4.3 推定の当否）

Layer A の `iter_total` に対する各区間の比率（N 別。中央値ベース）:

| N | matmul | host_copy | checksum |
| --- | --- | --- | --- |
| 1024 | 24.2% | 53.0% | 22.3% |
| 2048 | 33.8% | 42.7% | 22.4% |
| 4096 | 58.1%〜38.2%（回次でばらつき） | 25.6%〜28.5% | 13.3%〜15.4% |

**#1142 §4.3 の推定「H2D／D2H／同期の固定費が候補比を押し下げている」は部分的に不正確**
である。`matmul`（H2D A/B・カーネル実行・D2H・同期を全て含む区間）単体は §7 のとおり
candle の fresh 全体より高速であり、H2D／D2H／同期自体は candle 比未達の主因ではない。
`iter_total` を押し上げているのは、`matmul` の**外側**でベンチハーネス自身が行う
`host_copy`（`readout_var` の `contiguous().as_slice().to_vec()`。`clone_dtoh` が既に
返した `Vec<f32>` に対する**二重目**のホストコピー）と `checksum`（全要素和。イシュー
#965 の縮退検出のための診断コスト）であり、両者を合計すると `iter_total` の
**約 66〜75%**（N=1024: 75.3%・N=2048: 65.1%・N=4096 は d2h 二峰性の影響で回次により
39〜44%）を占める。これらは fandhe-ai の GEMM 実行そのものではなく**計測ハーネスの
診断コスト**である点が、#1142 §4.3 の推定を精緻化する本ドキュメントの主要な訂正点。

## §7 カーネル専有時間ベースの candle 比（参考値）

分母は候補系列の同一セッション再計測ではなく `#1142`（`docs/perf/
cuda-gemm-candle-gate-remeasurement.md` §表。正式系列・GB10 実機・candle `gemm cuda <N>
fresh` の 5 回計測中央値）を参照する（N=2048 は candle 無効データのため #1142 で判定不能
のまま。本ラウンドでは candle 再計測を実施していない — §9）。

| N | fandhe `matmul` 中央値（本ラウンド） | candle fresh 中央値（#1142 正式系列） | 比（candle/fandhe） |
| --- | --- | --- | --- |
| 1024 | 0.582 ms | 0.9236 ms | **1.59 倍**（fandhe 優位） |
| 2048 | 3.214 ms | 判定不能（candle 無効データ。#1142） | - |
| 4096 | 38.227 ms | 56.324 ms | **1.47 倍**（fandhe 優位） |

参考として `kernel_wait`（Select。転送を一切含まない純カーネル時間）ベースでは:

| N | fandhe `kernel_wait` 中央値 | candle fresh 中央値 | 比（candle/fandhe） |
| --- | --- | --- | --- |
| 1024 | 0.1908 ms | 0.9236 ms | 4.84 倍 |
| 4096 | 13.601 ms | 56.324 ms | 4.14 倍 |

**この参考値は分子分母が非対称**（fandhe 側は転送を除外〈`kernel_wait`〉または同一
セッション内〈`matmul`〉、candle 側は別セッション・転送込みの fresh 全体）であり、
fandhe に有利な方向に偏っている点に注意（§9 の「candle 側 kernel-only 計測」課題参照）。
既存の正式ゲート判定（`compare_gemm_gate.py`。`iter_total` 相当の `gemm --mode reuse`
非 phases 版の median_s を使用）は本ドキュメントでは変更しない（AC-2）。#1031 の
判定結果（N=1024/4096 未達・N=2048 判定不能。#1142 確定）はそのまま維持される。

## §8 スコープ外

- `Tensor<f32>` のデバイス常駐化・公開 API へのホスト転送なし同期 API 追加
- candle `gemm-transfer-split`（#1103 が metal 限定と決定済み）の cuda 拡張
- #1031 判定境界の再定義・spec（REQ-8／REQ-2）変更・tolerance／baseline 追加
- framework-compare ピン `=0.7.0` 更新・`run_all*.sh` への `gemm --mode reuse --phases`
  組み込み（診断専用のため標準スイープには含めない）
- N=4096 の D2H 二峰性の根本原因の特定（#1169 と同一現象の可能性が高いが、本ラウンドでは
  確定判断まで至っていない）
- 参考系列（HEAD path patch によるビルド）の実測

## §9 ユーザー判断事項（AC-3。事実／選択肢／推奨／影響範囲）

### 事実

1. `matmul`（H2D+カーネル+D2H+同期）単体は N=1024/4096 で candle fresh 全体より
   1.47〜1.59 倍高速（§7）。`kernel_wait`（純カーネル）はさらに 4.1〜4.8 倍高速
2. 既存の正式ゲート判定（`iter_total` 相当）が candle 比未達となる主因は `host_copy`
   （二重ホストコピー）＋ `checksum`（診断用全要素和）であり、`iter_total` の
   約 66〜75%（N=4096 は D2H 二峰性次第で変動）を占める（§6）
3. N=4096 の D2H は診断テスト内で顕著な二峰性（25〜27 ms vs 500〜650 ms）を示すが、
   Layer A（production 経路）の `matmul` はこの二峰性の影響を受けていないように見える
   （§5）。原因は本ラウンドでは未確定

### 選択肢

- (i) #1031 の判定境界を `iter_total`（現行。ハーネスの診断コストを含む）から `matmul`
  （production 経路の実測。H2D+カーネル+D2H+同期）へ再定義する。spec（REQ-8／
  `docs/spec/04-requirements.md`）の変更が必要な場合は spec リポジトリ側への提案が必要
  （`.claude/rules/out-of-scope-tracking.md`）
- (ii) `host_copy`／`checksum` をベンチハーネス側で削減する（例: checksum を全反復ではなく
  末尾反復のみで検証する・要素単位検証〈parity〉のみに統一する）。ただしイシュー #965/#970
  の縮退検出契約（全反復検証）を弱めることになるため、判定ロジックの安全性とのトレード
  オフの検討が必要
- (iii) N=4096 の D2H 二峰性の根本原因を追加調査する（#1169 と同一現象か切り分ける）
- (iv) candle 側の kernel-only 計測（`gemm-transfer-split` の cuda 拡張。#1103 の metal
  限定決定を覆すため要相談）を追加し、§7 の非対称性を解消する
- (v) 現状維持（既存ゲート判定・#1031 未達判定をそのまま確定とする）

### 推奨

即断は避け、まず (iii) の追加診断（D2H 二峰性の原因特定）を先行させることを推奨する
（(i) の判定境界再定義は #1031 の意味を変える重い決定であり、二峰性の原因が
「診断テスト固有のアーティファクト」なのか「production 経路にも潜在する」のかで
判断が変わりうるため）。(ii) はガードレール変更に準じる慎重な検討が要る
（`.claude/rules/security.md` A08・出力の安全性）。

### 影響範囲

- (i) を採用する場合: `compare_gemm_gate.py`・`docs/spec/04-requirements.md`（REQ-8）・
  `docs/performance-floor-decision.md`・`docs/gemm-optimization-baseline.md` の判定基準
  文言に影響
- (ii) を採用する場合: `scripts/bench/framework-compare/bench-fandhe/src/main.rs`
  （`run_gemm`／`run_gemm_reuse` 双方）・イシュー #965/#970 の契約文書に影響
- 上記いずれもユーザー承認必須（依存追加ではないが判定ロジック・許容誤差に準じる変更の
  ため。`.claude/rules/delegation-impl.md`「禁止事項」参照）

## §10 関連ドキュメント

- `docs/perf/cuda-gemm-candle-gate-remeasurement.md`（#1142。#1031 ゲート判定の確定記録。
  §4.3・§8 に本ドキュメントへの参照を追記済み）
- `docs/perf/cuda-gemm-tiled-pipeline.md`（#1137。cp.async パイプラインの GB10 実測・
  本番結線判断）
- `docs/perf/train-step-phase-breakdown.md`（#1010。同じ方法論の train 版）
- `docs/perf/cuda-large-buffer-percall-alloc-transfer-threshold.md`（#1169。N=4096 D2H
  二峰性と同一現象の可能性がある既知事象）
- `scripts/bench/framework-compare/README.md`「`gemm --mode reuse --phases`」節
- `docs/performance-targets.md` §8.5
