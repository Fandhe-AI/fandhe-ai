# Metal GEMM 計測窓の内訳（Mac mini・イシュー #3182）

数値の正式記録は `docs/perf/metal-gemm-reuse-phase-breakdown.md` §12。本ディレクトリはその生ログ一式。
**診断のみ**（計測窓の構成・判定式・tolerance・baseline・`scripts/bench/framework-compare/**`・本番コードは不変）。
規則の正は `RULE.txt`（計測前コミットに含む。固定 2026-10-20T04:20:00Z）。

## 構成

| パス | 内容 |
|---|---|
| `RULE.txt` | 事前登録規則（式 F1〜F7・腕・負荷ゲート・経路の前提） |
| `orchestrate_macmini.sh` | Mac mini 側の計測本体（ビルド → 5 run × 固定順。実行手順は `docs/real-hardware-verification-env.md` §7） |
| `aggregate.py` | 集計（標準ライブラリのみ。`python3 -I aggregate.py --self-test`／`python3 -I aggregate.py . > aggregate.md`）。欠損・非数値・run／サイズの欠け・重複は拒否 |
| `aggregate.md` | 機械集計の出力（§12 の数表の出典） |
| `layerA-run{1..5}.jsonl` | `bench-fandhe gemm reuse --phases`（task=`gemm_phases`。N=256〜4096） |
| `window-reuse-run{i}.jsonl`・`fresh-run{i}.jsonl` | `bench-fandhe gemm`（phases なし）の reuse／fresh 窓合計 |
| `candle-run{i}.jsonl` | `bench-candle gemm`（fresh）と `gemm-transfer-split`（`gemm_transfer_incl`／`excl`） |
| `layerB-run{i}.log` | 診断テスト出力（`gemm_window_route_precheck`・`gemm_window_phase_diag_small_sizes`・`gemm_reuse_phase_diag_production_batch`。`<home>` はホームのマスク） |
| `run{i}.err` | 各 run の stderr（5 本とも空） |
| `load_gate.log`・`env_info.txt`・`rev-stamp.txt`・`tree.txt` | 負荷ゲート（5/5 `pass`）・機体情報・計測ツリーの同定（フル SHA）・registry ピン 1 行 |

## 腕と条件

- ツリー: 計測前コミット `3114cdda2fe6c073184ad704537a5ed1076f6a10` を `git archive` し `.rev-stamp` を置いて rsync で転送。
  ツリー指紋は送信側・Mac mini 側とも `063846beb…ad7909` で一致。
- 腕 A: `bench-fandhe`（registry `fandhe-ai =0.11.0`）・`bench-candle`（`candle-core =0.11.0`）。腕 B: main workspace（HEAD ソース）の診断テスト。
  **A と B のソース版が違う**ため、A の列と B の列を 1 つの窓として足さない。
- target は main workspace が `target-3182`、framework-compare が `target-3182-fc`（既存の共有 target は触っていない）。
- 起動時に他のビルド・ベンチが走っていないことを確認し、計測中に並走させていない。ただし同じ Mac mini は共有機で、各 run の終了時 load1 は自分の計測で 2.5〜3.6 まで上がる。

## 注意と限界

- `aggregate.py` は計測後に 1 点だけ直した（`checksum` は負値がありうるため、符号を許す検証へ）。計測値・式には影響しない。
- `gemm --mode fresh --phases` は `bench-fandhe` が拒否する仕様のため、fresh は phases なしの窓合計のみ。
- 系列 B（`framework-compare-0.11.0-macmini-remeasure`）との N=4096 ドリフトは fandhe −26.7%・candle −22.3%（10% 超。記録のみで判定は不変。原因は未切り分け）。
- N=1024 の run1 は外れ値（Layer B の kernel_gpu 1.85 ms、他 4 run は 0.86 ms 付近）。原因は未特定。
- ホスト名・ユーザー名はコミット物に含めない（`layerB-run*.log` の絶対パスは `<home>` へ置換済み）。
- checksum が一致しても bit 同一の証拠ではない。
