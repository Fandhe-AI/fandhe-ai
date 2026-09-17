# イシュー #1973 実測スキャフォールド（CUDA GEMM reuse フェーズ分解・`fandhe-ai =0.9.0` 再計測）

## 位置づけ（本 PR 時点では未実測）

本 PR の実行環境（Linux。GPU 利用不能）には DGX Spark GB10 実機への
到達手段がないため、**実測値は一切含まれていない**。本ディレクトリは
スキャフォールド（実行スクリプト・事前登録判定規則・記入欄）のみを
提供し、実測は GB10 実機を持つセッションへ申し送る（
`docs/perf/logs/train-resident-grad-cuda-1560/`・
`docs/perf/logs/cuda-mse-backward-1692/` と同型の運用）。

## 目的

イシュー #1973「#1182 の分解は v0.6.0 時点。`--phases` 相当の
Layer A/B を 0.9.0 で GB10 再実測し、candle との差 0.5 倍分がどの
フェーズに乗っているかを確定する。削減優先順位を後続 sub の入力に
する」の実測記録先。`docs/perf/cuda-gemm-reuse-phase-breakdown.md`
（#1182・v0.6.0 時点）の方法論をそのまま踏襲し、`framework-compare`
ピンが `fandhe-ai =0.9.0`（2026-09-17 公開）へ更新された現時点で
再計測する。

## 対象形状

N=1024／2048／4096（#1182 の対象形状を継承）。issue #1973 本文の受け
入れ条件は「N=512／1024／2048」と記載されているが、#1182 の実測範囲・
`gemm_reuse_phase_diag_tests.rs::SIZES` 定数（`[1024, 2048, 4096]`）
との整合を優先し、本スキャフォールドは 1024／2048／4096 を対象とする
（表記揺れの解消要否は issue 側で別途確認する）。

## 事前登録判定規則（計測前に固定・結果を見て変更しない）

- **内訳表**: N=1024／2048／4096 それぞれについて、Layer A
  （`matmul`／`to_tensor`／`host_copy`／`checksum`／`iter_total`）の
  5 run 中央値・`iter_total` に対する比率を記録する（`aggregate.py`
  が機械算出）
- **Layer B**: `h2d_a`／`h2d_b`／`alloc_c`／`launch_issue`／
  `kernel_wait`／`d2h`／`host_copy` の 5 run 中央値を N × 変種
  （Select／Classic）ごとに記録する（`gemm_reuse_phase_diag_tests.rs`
  の標準出力形式をそのまま転記する）
- **AC-2（挙動不変）**: `--phases` なしの `gemm --mode reuse` の
  checksum が phases 版と bit 単位で一致することを確認する（#1182
  §3 と同じ確認）。`run_gemm_reuse` 本体を本スキャフォールドでは
  一切変更しない
- **candle 参照（診断用・非判定）**: `bench-candle --task gemm
  --device cuda --mode fresh` の 5 run 中央値を記録する。#1031 の
  正式ゲート判定（`run_gemm_gate_cuda.sh`・`compare_gemm_gate.py`）を
  代替しない（本スキャフォールドは診断専用であり正式ゲート判定は
  別イシューのまま変更しない）
- **削減候補の優先順位**: Layer A／B の区間比率から根拠付きで
  記録する（#1182 §6 の帰属手法を踏襲。「削減候補（アロケータ・
  同期・readback）の優先順位を根拠付きで記録する」という issue の
  受け入れ条件に対応）
- **判定に使う 5 run 中央値・比率は事後に緩和しない**（結果が FAIL
  相当でも是正せず記録のみとする。issue #1973「共通ルール」節）

## ディレクトリ構成

| パス | 内容 |
|------|------|
| `orchestrate.sh` | Layer A（`--phases`）・AC-2・candle 参照・Layer B の一括実行スクリプト（`--dry-run` あり） |
| `aggregate.py` | `layerA-phases-N*.log` から phase 別 5 run 中央値・`iter_total` 比を算出し Markdown 表を出力する（`--self-test` あり。python3 標準ライブラリのみ） |
| `env_info.txt` | 実行環境・driver／CUDA バージョン・計測前後の `nvidia-smi` 出力の記入欄（未実測のため未記入。orchestrate.sh が末尾へ uptime／nvidia-smi を追記する） |
| `layerA-phases-N{1024,2048,4096}.log` | Layer A（`--phases`）5 run 分の JSONL 出力（未生成） |
| `layerA-ac2.log` | AC-2（非 phases）1 回ずつの出力（未生成） |
| `candle-fresh-N{1024,2048,4096}.log` | candle 参照（診断用）5 run 分の出力（未生成） |
| `layerB-run{1..5}.log` | Layer B（`gemm_reuse_phase_diag_select`／`_classic`）各 1 回分の出力（未生成） |

## GB10 実機での実行手順

1. `docs/real-hardware-verification-env.md` の手順で GB10 実機へ
   接続する
2. 本ディレクトリで `./orchestrate.sh` を実行する（ビルド〜Layer A〜
   AC-2〜candle 参照〜Layer B を一括実行する。約 20〜30 分想定）
3. `python3 aggregate.py` で Layer A の集計 Markdown を生成し、
   `docs/perf/cuda-gemm-reuse-phase-breakdown.md` の #1973 節へ転記
   する
4. Layer B（`layerB-run*.log`）の標準出力から N × 変種ごとの中央値
   （`gemm_reuse_phase_diag_tests.rs` の `print_quartiles_ms` 出力）を
   手動集計し、同じく転記する
5. 削減候補の優先順位を根拠付きで記録し、`docs/perf/cuda-gemm-
   reuse-phase-breakdown.md` の #1973 節・issue #1973 のチェックリスト
   を更新する
6. 生成物一式（`layerA-*.log`・`layerB-*.log`・`candle-fresh-*.log`・
   `env_info.txt`）を本ディレクトリへ回収し、内部ホスト名・ユーザー名・
   絶対パスが含まれないことを確認してマスクする

## 注意

- 内部ホスト名・ユーザー名・絶対パスを成果物（ログ・env_info・
  README）に書かない
- 実測値の捏造・事前登録規則の事後緩和は禁止（security.md A08）
- 本番コード（`crates/*/src`）は本イシューでは変更しない
- `--no-verify`・`#[allow(clippy::…)]`・tolerance／baseline／依存の
  変更は禁止
