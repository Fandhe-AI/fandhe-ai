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

## §11 借用ビュー readout の実装（イシュー #1337）

§9 の選択肢 (ii)「`host_copy`／`checksum` をハーネス側で削減する」のうち、
**#965/#970 の縮退検出契約（checksum 全反復計算・要素単位 parity 検証）を
一切弱めずに `host_copy`（`.to_vec()` の memcpy）のみを消す**形を
`scripts/bench/framework-compare/bench-fandhe`（`readout_var`）へ実装した
（cargo feature `host-view-readout`。既定 OFF・`fandhe_ai::VarHostView`／
`Tensor::host_slice`〈#1335・#1336〉利用・crates.io 公開版 `fandhe-ai
=0.7.0` には該当 API が未収録のため `managed-placement` と同じ path patch
分離方式）。詳細な設計・区間再定義・A/B 手順は
`scripts/bench/framework-compare/README.md`「gemm --mode reuse --phases」
節「借用ビュー readout（イシュー #1337）」小節を正とする。

`readout_var` の借用ビュー切替は checksum／parity 契約・legacy 経路との
bit 同一性を `main.rs` の単体テスト（`readout_var_matches_legacy_to_vec_
bit_exact`・`host_view_readout_keeps_tape_usable`）で自己検証済み（feature
有効・無効いずれのビルドでも green）。CUDA/Metal/CPU 3 バックエンド ×
対象形状での切替前後・candle 比の実機実測記録（本 issue の受入条件）は
以下へ記録した:

- CUDA: イシュー #1360・`docs/perf/cuda-gemm-candle-gate-remeasurement.md`
  §12（Phase 4／5 反映後の初回実測）・§13（#1337 独自実行による再現性確認・
  公正性の論点。DGX Spark GB10）
- Metal: イシュー #1337・`docs/perf/metal-gemm-candle-gate-remeasurement.md`
  §12（Apple M4 Max）
- CPU: イシュー #1337・`docs/perf/cpu-gemm-candle-gate-remeasurement.md`
  §15（DGX Spark GB10 Grace CPU・Apple M4 Max 両実機）

`#1336`（CUDA pinned host staging）は本経路に到達しない点（`Var::matmul`
の出力は `gemm` バックエンド内部の readback で既にホスト常駐 `Tensor` に
なっているため）は README 側に明記済み（誤帰属防止）。上記いずれの節でも
`#1336` には帰属しないことを確認しているが、**`#1337`（readout 単独）への
厳密な帰属は off/on 間でライブラリソース（registry ピン対 HEAD path）を
揃えて計測できた比較に限られる**。同一ソースで比較できたのは CUDA
§12（`docs/perf/cuda-gemm-candle-gate-remeasurement.md`）のみであり、
Metal §12・CPU §15 は off 腕が registry・on 腕が HEAD path という
ソース差を含む参考比較のため、readout 単独への帰属は保留のまま各節に
明記している（詳細は各節を参照）。

## §12 `fandhe-ai =0.9.0` ピンでの再計測（イシュー #1973・**2026-09-18 GB10 実測済み**）

### §12.1 位置づけ

§3〜§9 の実測は crates.io 公開版 `fandhe-ai =0.6.0` 時点（2026-09-04
計測）のものであり、その後 §11（#1337 借用ビュー readout の既定経路化・
#1438）・readback 宛先の `PretouchedFresh` 既定化（#1437）・128×64
cp.async pipeline 結線（#1344）等、reuse 経路に影響しうる変更が複数
マージされている（pinned host staging 既定化〈#1478〉は §11 のとおり
本経路に到達しないため対象外）。イシュー #1973（親 #1972）
は、framework-compare の承認ピンが `fandhe-ai =0.9.0`（2026-09-17 公開。
`.claude/rules/deps-policy.md` 参照）へ更新された現時点で §3〜§9 と同じ
2 層方法論（Layer A: `--phases` 公開 API 境界・Layer B: `crates/backend-cuda`
非公開 API 内部分解）を再実行し、candle 比の約 0.5 倍という乖離（出典:
issue #1973 本文のスコアボード参照）がどのフェーズに帰属するかを確定する
目的で起票された。

対象形状は §3 の N=1024／2048／4096 を継承する（issue #1973 は**題名**が
「N=512〜2048」、**受け入れ条件本文**が「N=1024／2048／4096」と表記が揺れて
いる。`gemm_reuse_phase_diag_tests.rs::SIZES` 定数〈`[1024, 2048, 4096]`〉・
#1182 実測範囲・受け入れ条件本文と整合する**スキャフォールド側の N=1024／
2048／4096 を正とした**。N=512 は本ラウンドでは未計測）。判定規則は
`docs/perf/logs/cuda-gemm-reuse-phase-1973/README.md` の事前登録どおり
（緩和なし）。生ログ・集計スクリプト（`aggregate.py`・`aggregate_layer_b.py`）
・`env_info.txt` は同ディレクトリ。

### §12.2 計測環境・系列

- 実機: DGX Spark GB10（driver 580.173.02・CUDA 13.0・Ubuntu 24.04.4
  aarch64・rustc/cargo 1.97.0）。計測前 load average 約 0.2〜0.6・GPU 利用率
  0%・専有環境。計測 UTC 2026-09-18T01:39:45Z〜01:44:29Z（`orchestrate.sh`
  一括実行）。転送元 main `536c56a8`
- **Layer A**: registry ピン `fandhe-ai =0.9.0`（`bench-fandhe gemm --device
  cuda --mode reuse --phases`。5 run × 各 20 warmup + 20 trial の中央値）
- **Layer B**: HEAD ツリー（`536c56a8`）の `gemm_reuse_phase_diag_select`／
  `_classic`。`git diff v0.9.0..HEAD -- crates/backend-cuda/src` は
  arg_reduce／norm_backward／log_softmax_backward の新規ファイルと
  `ops.rs`／`reduce.rs`／`error.rs`／`context_cache.rs`／`lib.rs` のみで、
  GEMM reuse 経路（`gemm.rs`・`memory.rs`・`kernels_*`）に差分はない。
  したがって #1182（§5）と異なり **Layer A と Layer B は同一の CUDA GEMM
  コード**を計測しており、`Classic` を近似比較に使う必要はなく `Select`
  同士で直接突合できる
- candle 参照（診断用・非判定）: `bench-candle gemm --device cuda --mode
  fresh` 5 run 中央値（同一セッション・同一環境）

### §12.3 Layer A 実測（`fandhe-ai =0.9.0`。5 run 中央値、単位 ms）

| N | matmul | to_tensor | host_copy | checksum | iter_total | matmul 比 | checksum 比 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1024 | 1.6601（1.6432, 1.6601, 1.6847, 1.8868, 1.6321） | 0.0001 | 0.0001 | 0.5280 | 2.1882（2.1836, 2.1882, 2.2114, 2.4206, 2.1662） | 75.9% | 24.1% |
| 2048 | 6.3041（6.3478, 6.1069, 6.3304, 6.3041, 5.9233） | 0.0003 | 0.0002 | 2.1393 | 8.4470（8.4879, 8.2578, 8.4696, 8.4470, 8.0670） | 74.6% | 25.3% |
| 4096 | 30.1819（30.1071, 30.1819, 30.2909, 30.2537, 30.0538） | 0.0005 | 0.0002 | 8.6272 | 38.8332（38.7364, 38.8332, 38.8996, 38.8833, 38.6751） | 77.7% | 22.2% |

`to_tensor`／`host_copy` が ≈0 なのは §11 の借用ビュー readout が既定経路
（#1438）として効いていることの直接確認である（#1182 §3 では `host_copy`
が 1.272／4.148／18.545 ms）。要素単位検証（`parity_*`）は全 N・全 run で
`fail_count=0`。

**AC-2（挙動不変）**: `--phases` なしの `gemm --mode reuse`（`layerA-ac2.log`）
の checksum は N=1024 `-1855.597736`・N=2048 `-6016.774008`・N=4096
`-25768.747284` で phases 版と bit 単位で一致（#1182 の値とも同一）。非
phases の `median_s` は 1.908／8.070／39.080 ms で、phases 版 `iter_total`
（2.188／8.447／38.833 ms）との差 +0.28／+0.38／−0.25 ms は区間計時の
オーバーヘッドとノイズの範囲であり、両者を混同しない。

### §12.4 Layer B 実測（HEAD `536c56a8`。5 run 中央値、単位 ms。生値は `aggregate_layer_b.md`）

| N | kernel | h2d_a | h2d_b | alloc_c | launch_issue | kernel_wait | d2h | host_copy | Σ(matmul 相当) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1024 | Select | 0.0842 | 0.0784 | 0.0002 | 0.0050 | 0.1823 | 0.0750 | 1.3231 | 0.4251 |
| 1024 | Classic | 0.0844 | 0.0791 | 0.0002 | 0.0049 | 0.3189 | 0.0753 | 1.3980 | 0.5628 |
| 2048 | Select | 0.3040 | 0.2910 | 0.0005 | 0.0078 | 1.1652 | 0.2819 | 5.2765 | 2.0504 |
| 2048 | Classic | 0.3036 | 0.2908 | 0.0005 | 0.0079 | 2.2636 | 0.2827 | 6.3346 | 3.1491 |
| 4096 | Select | 1.1618 | 1.1393 | 0.0006 | 0.0083 | 9.8940 | **二峰性**（25.66, 624.07, 618.79, 607.74, 620.13） | 20.4549 | （d2h 除く 12.204） |
| 4096 | Classic | 1.1610 | 1.1422 | 0.0006 | 0.0083 | 20.9149 | **二峰性**（625.32, 635.01, 27.15, 24.41, 24.76） | 20.0126 | （d2h 除く 23.227） |

`Σ(matmul 相当)` は `host_copy` を除く 6 区間の和（§5 と同じ定義）。N=4096
の `d2h` は run 内中央値そのものが約 25 ms／約 620 ms の二峰性を示し、
Select 5 run 中 4 run・Classic 5 run 中 2 run が高モードに落ちた。これは
**標本の偏りであって Select／Classic の差ではない**（#1182 §4 と同じ
#1169 系の挙動。診断テストが反復ごとに未タッチの新規 `Vec<f32>` を
`clone_dtoh` 宛先にする設計に起因すると考えられる）。一方 Layer A の
N=4096 `matmul` は 5 run とも 30.05〜30.29 ms（q1/q3 29.9〜30.4）で二峰性を
示さず、本番 `readback`（#1437 `PretouchedFresh` 既定）には同現象が現れて
いないことを事実として記録する。

`kernel_wait` の TFLOPS 換算（2·N³/t）と既存独立計測との突合:

| N | 変種 | kernel_wait（本ラウンド） | #1182 §4 | 換算 TFLOPS | 所見 |
| --- | --- | --- | --- | --- | --- |
| 1024 | Select | 0.1823 ms | 0.1908 ms | 11.78 | #1137（11.23 TFLOPS）と同水準 |
| 2048 | Select | 1.1652 ms | 1.3132 ms | 14.74 | #1182 比 0.887 倍 |
| 4096 | Select | 9.8940 ms | 13.6009 ms | 13.89 | #1182 比 0.727 倍 |
| 1024 | Classic | 0.3189 ms | 0.3191 ms | 6.73 | 一致 |
| 2048 | Classic | 2.2636 ms | 2.2644 ms | 7.59 | 一致 |
| 4096 | Classic | 20.9149 ms | 20.2845 ms | 6.57 | 3% 程度 |

`Classic`（固定カーネル）が #1182 と一致していることから計測系・GPU 状態は
安定しており、`Select` の N=2048／4096 の短縮はコード側の変更に帰属する
候補として記録する（候補: #1344 の 128×64 cp.async pipeline 結線。同 issue
の純カーネル時間実測 N=2048 1.12 倍・N=4096 1.35 倍と本ラウンドの 1.127／
1.375 倍が整合するが、本ラウンドは切替前後の同一セッション比較ではない
ため帰属は未検証）。

### §12.5 突合（Layer A `matmul` 対 Σ Layer B）

| N | Layer A `matmul` | Σ Layer B（Select・matmul 相当） | 差（未説明分） | 差 / iter_total |
| --- | --- | --- | --- | --- |
| 1024 | 1.6601 ms | 0.4251 ms | **1.235 ms** | 56.4% |
| 2048 | 6.3041 ms | 2.0504 ms | **4.254 ms** | 50.4% |
| 4096 | 30.1819 ms | 12.204 ms（d2h 除く） | **17.98 ms**（D2H 転送そのものを含む上界。1024／2048 行とは非可比） | 46.3%（上界） |

#1182 §5 では両層が ±3% で一致していたのに対し、本ラウンドは Layer A
`matmul` が Σ Layer B の 3〜4 倍（N=1024: 3.9 倍・N=2048: 3.1 倍）であり、
**約 1.2／4.3／18 ms の未説明分が本番経路の `matmul` 区間内に存在する**。
Layer A・Layer B は同一 GEMM コード（§12.2）であり、`h2d_a`／`h2d_b`／
`kernel_wait`／`launch_issue`／`alloc_c` はいずれも Layer A `matmul` に
含まれる要素で総和 0.35／2.05／12.2 ms に過ぎないため、未説明分は Layer B
が模していない **本番 readback（D2H 宛先の確保・事前タッチ・読み戻し）**
に帰属する候補が最有力である。

数値上の傍証（帰属は**未検証**）: 未説明分（1.235／4.254／17.98 ms）は
Layer B `host_copy`（未タッチの新規 `Vec<f32>` へ同サイズを memcpy する
コスト: 1.323／5.277／20.45 ms）と同オーダーで一致する。本番 `readback`
は #1437 以降、反復ごとに新規 `Vec` を確保して非ゼロ sentinel で事前
タッチ（`ReadbackDest::PretouchedFresh`。`crates/backend-cuda/src/memory.rs`）
してから `memcpy_dtoh` する設計であり、この事前タッチ 1 回分が Layer B
`host_copy` と同じ「4／16／64 MiB の未タッチページへの書き込み」に相当
する。#1182 時点（v0.6.0）では同じページタッチ費用がハーネス側 `host_copy`
（`.to_vec()`）区間で 1.272／4.148／18.545 ms として観測されており、#1438
で `host_copy` が ≈0 になった代わりに #1437 の事前タッチが `matmul` 区間
の内側へ移った、という**区間帰属の移動**として整合する。したがって
Layer A `matmul` 0.582 → 1.660 ms（N=1024）は GEMM 実行の後退ではなく、
`iter_total` は 2.402 → 2.188（N=1024）・9.514 → 8.447（N=2048）・65.088 →
38.833（N=4096）と全 N で短縮している。ただし本ラウンドは `PretouchedFresh`
と `Fresh` の切替比較を含まないため、上記帰属は #1437 の Layer B 分離計測
（`docs/perf/cuda-host-view-readout-small-shape-regression.md` §13）を
本番 `matmul` 区間で再確認する後続計測を要する（未検証）。

### §12.6 candle 参照（診断用・非判定。同一セッション 5 run 中央値）

| N | candle fresh（ms） | fandhe `iter_total` | fandhe/candle | fandhe `matmul` | matmul/candle | fandhe `kernel_wait`（Select） | kernel_wait/candle |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1024 | 0.9247（0.9229, 0.9267, 0.9272, 0.9247, 0.9240） | 2.1882 | 2.366 | 1.6601 | 1.795 | 0.1823 | 0.197 |
| 2048 | 4.2145（4.1882, 4.2069, 4.2220, 4.2145, 4.3896） | 8.4470 | 2.004 | 6.3041 | 1.496 | 1.1652 | 0.276 |
| 4096 | 58.2561（57.5679, 58.2561, 58.2787, 58.2247, 58.6186） | 38.8332 | 0.667 | 30.1819 | 0.518 | 9.8940 | 0.170 |

比は fandhe/candle（1 未満で fandhe 優位）。非 phases `median_s` ベースでは
candle/fandhe = 0.485／0.522／1.49 倍であり、issue #1973 本文の「約 0.5 倍」
（N=1024／2048）を同一セッションで再現した。N=1024／2048 は **`matmul`
区間単体でも candle fresh 全体より遅い**（#1182 §7 では 1.59 倍 fandhe
優位だった向きが反転）が、§12.5 のとおり `matmul` 区間には本番 readback
の宛先事前タッチ（推定 1.2／4.3 ms）が含まれる。純カーネル時間
（`kernel_wait`）は candle fresh 全体の 0.17〜0.28 倍。candle 側は所有
`Vec`（`.to_vec2()`）読み出し・fandhe 側は借用ビュー読み出しという非対称
（§11・README）と、candle 側が別プロセス fresh・転送込みである点は #1182
§7 と同じく留意する。正式ゲート判定（`compare_gemm_gate.py`・#1031）は
本節では更新しない。

### §12.7 #1182（v0.6.0）との差分所見

1. `host_copy` 1.272／4.148／18.545 ms → ≈0（#1438 借用ビュー readout）。
   `iter_total` は全 N で短縮（2.402→2.188・9.514→8.447・65.088→38.833 ms）
2. `matmul` 0.582／3.214／38.227 → 1.660／6.304／30.182 ms。N=1024／2048 の
   増加分（+1.08／+3.09 ms）は §12.5 の未説明分と同オーダーで、readback
   宛先の事前タッチが `matmul` 区間へ移動した区間帰属の変化として整合
   （未検証）。N=4096 は #1182 側の `matmul` が d2h 二峰性の影響で回次
   ばらつきが大きく（32.7〜41.1 ms）、直接比較は保留
3. `kernel_wait`（Select）は N=2048／4096 で 0.887／0.727 倍へ短縮・Classic
   は不変（§12.4。候補 #1344・未検証）
4. Layer B の `alloc_c`（0.0002〜0.0006 ms）・`launch_issue`（0.005〜0.008
   ms）・`d2h`（N=1024／2048）・`h2d_*` は #1182 と同水準
5. N=4096 Layer B `d2h` の二峰性は残存（診断テスト側の設計に起因。本番
   Layer A には現れない。§12.4）

### §12.8 削減候補の優先順位（issue 受け入れ条件。根拠は上記の数値のみ）

| 順位 | 候補 | 根拠（N=1024／2048／4096） | 区分 |
| --- | --- | --- | --- |
| 1 | **readback（D2H 宛先の確保・事前タッチ）** | Layer A `matmul` と Σ Layer B の未説明分 1.235／4.254／17.98 ms = `iter_total` の 56.4／50.4／46.3%（N=4096 は D2H 転送そのものを含む上界。§12.5）。N=1024 でこれを除くと `iter_total` 0.95 ms ≈ candle fresh 0.92 ms | ライブラリ側（`crates/backend-cuda::memory::readback`）。帰属は未検証（§12.5） |
| 2 | **checksum（ハーネス診断コスト）** | `iter_total` の 24.1／25.3／22.2%（0.528／2.139／8.627 ms）。#965 縮退検出契約のため削減は候補 3 とは別枠・#1339（デバイス側 checksum）の GPU 実装が対応先 | ハーネス側（ライブラリ側の削減対象外。§6 と同じ結論） |
| 3 | **H2D（A／B の毎反復アップロード）** | `h2d_a+h2d_b` 0.163／0.595／2.301 ms = `iter_total` の 7.4／7.0／5.9%。reuse モードでも葉 `Var` はホスト `Tensor` のため `Var::matmul` ごとに転送される（Layer B が同型で模している事実） | ライブラリ側（デバイス常駐 `Tensor` API は §8 スコープ外のまま） |
| — | 同期（`launch_issue`＋`kernel_wait` 待ち） | `launch_issue` 0.005〜0.008 ms・`kernel_wait` は TFLOPS 換算で独立計測と一致（§12.4）。同期由来の固定費は検出されない | 候補にしない |
| — | アロケータ（`alloc_c`） | 0.0002〜0.0006 ms（プール経由）。`iter_total` の 0.03% 未満 | 候補にしない |

結論: issue 題名の 3 候補（アロケータ・同期・readback）のうち、本ラウンドの
数値で候補として残るのは **readback のみ**であり、アロケータ・同期は
削減対象から外れる。readback の未説明分は `PretouchedFresh` の事前タッチに
帰属する候補が数値上最有力だが未検証であり、後続 sub（#1972 配下）では
(a) 本番 `matmul` 区間での `ReadbackDest` 切替比較、(b) 事前タッチ済み
宛先の反復間再利用（#1336 `HostStagingCache` と同型の設計。§11 のとおり
現状は本経路に到達しない）の A/B を、事前登録規則付きで行うことを入力
として引き継ぐ。tolerance／baseline／本番コードは本イシューで変更しない。
