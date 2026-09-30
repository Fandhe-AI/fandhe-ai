# CUDA gemm N=256 起動固定費の診断ログ（イシュー #2109）

親 #2099（Phase 3 負けセル対処）の sub。スコアボード 2026-09-19 版の
G-CUDA-G256（fandhe-ai 2 位・対 candle 0.83×）の起動固定費（launch 回数・
同期・確保・解放・host dispatch）の内訳を診断する。設計・仮説・再実行しない
実験の表は [`docs/perf/cuda-gemm-small-launch-cost.md`](../../cuda-gemm-small-launch-cost.md)、
判定規則は本ディレクトリの [`RULE.txt`](./RULE.txt)（実測前に固定）を正とする。

## 位置づけ

- 本 PR は計装・診断テスト・事前登録規則・計測スクリプトまで。**GB10 での
  5 run 実測（受入基準 2・3 の数値記入）は実機セッションへ申し送る**
  （開発機は CUDA toolkit〈NVRTC〉非搭載のため実機テストを実行できない）。
- 修正実装（readback 宛先再利用 #2108・Graph capture #2115 等）は対象外。

## 構成

| ファイル | 役割 |
|---|---|
| `RULE.txt` | 事前登録判定規則（実測前に固定。事後に緩和しない） |
| `orchestrate.sh` | Layer A（registry）／HEAD path-patch Layer A（H5 の突合相手）／AC-2／candle 参照／Layer B（5 プロセス）／任意の nsys を一括実行。`--dry-run` 付き。収録時にホスト名・`$HOME` をマスク |
| `aggregate.py` | 集計（python3 標準ライブラリのみ）。`--self-test` あり。checksum・件数・run 数・未マスクパス（nsys・env_info の任意入力を含む全入力）に加え、全レコード種別（Layer B の出力一致 `verify` 行を含む）の重複とキー集合の過不足（RULE.txt 1）を fail-closed で検査 |
| `env_info.txt` | 環境情報の記入欄（実測時に記入） |
| `layerA-*.log` `candle-fresh-N256.log` `layerB-run{1..5}.log` `counts-exact.log` `nsys-cuda-api.log` `load_gate.log` | 実測時に生成される生ログ（未生成） |

## 実行手順（GB10 実機。別セッション）

```
./docs/perf/logs/cuda-gemm-small-launch-cost-2109/orchestrate.sh          # 任意 nsys は RUN_NSYS=1
python3 docs/perf/logs/cuda-gemm-small-launch-cost-2109/aggregate.py > docs/perf/logs/cuda-gemm-small-launch-cost-2109/aggregate.md
```

`orchestrate.sh` は出力先（既定は本ディレクトリ）に既存の計測ログがあると開始前に
`exit 1` で停止する（RULE.txt 1: 上書き禁止）。再計測は空の別ディレクトリを
`LOG_DIR=<dir>` で指定し、集計は `aggregate.py --log-dir <dir>` で行う。

`aggregate.py` が非ゼロ終了した系列は無効とし、原因を記録して系列全体を
やり直す（一部 run の差し替えはしない）。

## 結果欄

未記入（GB10 実測待ち）。
