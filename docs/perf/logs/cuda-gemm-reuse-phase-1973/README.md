# イシュー #1973 実測スキャフォールド（CUDA GEMM reuse フェーズ分解・`fandhe-ai =0.9.0` 再計測）

## 位置づけ（**2026-09-18 DGX Spark GB10 実測済み**）

本ディレクトリは当初、実行環境（Linux。GPU 利用不能）に GB10 実機への
到達手段がないためスキャフォールド（実行スクリプト・事前登録判定規則・
記入欄）のみを収録していたが、2026-09-18（UTC 01:39〜01:44）に GB10 実機
セッションで `orchestrate.sh` を一括実行し、生成物一式（`layerA-*.log`・
`layerA-ac2.log`・`candle-fresh-*.log`・`layerB-run{1..5}.log`・
`env_info.txt`）を回収済みである。判定規則は下記「事前登録判定規則」
のとおりで事後緩和はしていない。集計は `aggregate.md`（Layer A。
`python3 aggregate.py`）・`aggregate_layer_b.md`（Layer B。`python3
aggregate_layer_b.py`）に機械生成し、転記先は
`docs/perf/cuda-gemm-reuse-phase-breakdown.md` §12。

### 結果要約（詳細・根拠は同 doc §12）

- Layer A（`fandhe-ai =0.9.0`）: `iter_total` 2.188／8.447／38.833 ms
  （N=1024／2048／4096）。`matmul` が 75〜78%・`checksum` が 22〜25%・
  `to_tensor`／`host_copy` ≈0（借用ビュー readout が既定経路で有効）。
  AC-2 checksum は phases 版と bit 一致・parity 全 run `fail_count=0`
- Layer B（HEAD `536c56a8`。GEMM 経路は v0.9.0 と同一コード）: `alloc_c`
  0.0002〜0.0006 ms・`launch_issue` 0.005〜0.008 ms・`kernel_wait`（Select）
  0.182／1.165／9.894 ms。N=4096 `d2h` は run 内中央値が約 25／620 ms の
  二峰性（診断テスト側の設計に起因・本番 Layer A には現れない）
- 突合: Layer A `matmul` − Σ Layer B（host_copy 除く）= 1.235／4.254／
  約 18 ms（`iter_total` の 56／50／46%。N=4096 は D2H 転送込みの上界）が本番 readback（D2H 宛先の
  確保・事前タッチ）に帰属する候補として最有力（未検証）
- candle 参照（診断用）: candle/fandhe（非 phases median_s）= 0.485／
  0.522／1.49 倍で issue 本文の「約 0.5 倍」を同一セッションで再現
- 削減候補の優先順位: 1. readback（宛先の確保・事前タッチ）・2. checksum
  （ハーネス側・ライブラリ削減対象外）・3. H2D（A／B 毎反復転送）。
  同期・アロケータは候補にしない

### 形状の表記揺れ

issue #1973 は**題名**が「N=512〜2048」、**受け入れ条件本文**が「N=1024／
2048／4096」と表記が揺れている。本スキャフォールドは「N=1024／2048／4096」
（#1182 の実測範囲・`gemm_reuse_phase_diag_tests.rs::SIZES` 定数・受け入れ
条件本文と一致）。**スキャフォールド側を正とし N=1024／2048／4096 で実測した**
（N=512 は未計測）。

## 目的

イシュー #1973「#1182 の分解は v0.6.0 時点。`--phases` 相当の
Layer A/B を 0.9.0 で GB10 再実測し、candle との差 0.5 倍分がどの
フェーズに乗っているかを確定する。削減優先順位を後続 sub の入力に
する」の実測記録先。`docs/perf/cuda-gemm-reuse-phase-breakdown.md`
（#1182・v0.6.0 時点）の方法論をそのまま踏襲し、`framework-compare`
ピンが `fandhe-ai =0.9.0`（2026-09-17 公開）へ更新された現時点で
再計測する。

## 対象形状

N=1024／2048／4096（#1182 の対象形状を継承）。issue #1973 の題名は
「N=512〜2048」だが受け入れ条件本文は「N=1024／2048／4096」であり、#1182
の実測範囲・`gemm_reuse_phase_diag_tests.rs::SIZES` 定数（`[1024, 2048,
4096]`）とも整合するため、本スキャフォールドは 1024／2048／4096 を対象と
する（上記「形状の表記揺れ」参照）。

## 事前登録判定規則（計測前に固定・結果を見て変更しない）

- **内訳表**: N=1024／2048／4096 それぞれについて、Layer A
  （`matmul`／`to_tensor`／`host_copy`／`checksum`／`iter_total`）の
  5 run 中央値・`iter_total` に対する比率を記録する（`aggregate.py`
  が機械算出）
- **完全性検査（fail-closed）**: `aggregate.py` は各対象サイズで
  5 run が揃い、各 run に全 phase が重複なく含まれることを検証する。
  壊れた JSON 行・run 数不足・phase 欠落／重複を検出した場合は
  当該サイズを黙って除外せず、正式な集計値（Markdown 表）を一切
  出力せず非ゼロ終了でエラーにする（`LogIntegrityError`。詳細は
  `aggregate.py` の doc comment を正とする）
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
| `env_info.txt` | 実行環境・driver／CUDA バージョン・計測前後の `nvidia-smi` 出力（記入済み。orchestrate.sh が末尾へ uptime／nvidia-smi を追記済み。ホスト名は masked） |
| `layerA-phases-N{1024,2048,4096}.log` | Layer A（`--phases`）5 run 分の JSONL 出力（実測済み） |
| `layerA-ac2.log` | AC-2（非 phases）1 回ずつの出力（実測済み） |
| `candle-fresh-N{1024,2048,4096}.log` | candle 参照（診断用）5 run 分の出力（実測済み） |
| `layerB-run{1..5}.log` | Layer B（`gemm_reuse_phase_diag_select`／`_classic`）各 1 回分の出力（実測済み。絶対パスは `<home>` へマスク済み） |
| `aggregate.md` | `aggregate.py` の生成物（Layer A 5 run 中央値表） |
| `aggregate_layer_b.py`／`aggregate_layer_b.md` | Layer B の 5 run 中央値・生値・Σ(matmul 相当) を機械集計するスクリプトと生成物（python3 標準ライブラリのみ） |

## GB10 実機での実行手順

1. `docs/real-hardware-verification-env.md` の手順で GB10 実機へ
   接続する
2. 本ディレクトリで `./orchestrate.sh` を実行する（ビルド〜Layer A〜
   AC-2〜candle 参照〜Layer B を一括実行する。約 20〜30 分想定）
3. `python3 aggregate.py` で Layer A の集計 Markdown を生成し、
   `docs/perf/cuda-gemm-reuse-phase-breakdown.md` の #1973 節へ転記
   する
4. `python3 aggregate_layer_b.py > aggregate_layer_b.md` で Layer B
   （`layerB-run*.log`）の N × 変種ごとの 5 run 中央値・生値を機械集計し、
   同じく転記する
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
