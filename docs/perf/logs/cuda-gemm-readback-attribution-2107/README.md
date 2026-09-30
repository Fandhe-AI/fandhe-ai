# #2107 readback 宛先確保の帰属検証（実測基盤）

`docs/perf/cuda-gemm-reuse-phase-breakdown.md` §13 の実測基盤。判定規則は
`RULE.txt`（実測前に固定。事後に緩めない）が正。

## 状態

計装・規則・オーケストレータ・集計器のみ実装済み。**GB10 での 5 run 実測と
帰属判定は未実施（実機セッションへ申し送り）**。開発ホスト（x86／RTX 3060）は
NVRTC（CUDA toolkit）が無く `CudaGemm` 構築に失敗するため smoke も未実施。

## 構成

| ファイル | 内容 |
|---|---|
| `RULE.txt` | 事前登録判定規則 |
| `orchestrate.sh` | `<gb10|x86> [--dry-run] [--out <dir>]`。テストバイナリを 1 回ビルドし (N, 腕) ごとに独立プロセスで 5 run＋同一セッション Layer A。既存 run の上書きは拒否。出力は host 名・`$HOME` をマスク |
| `aggregate.py` | `python3 aggregate.py <dir>` で `aggregate.md` を生成。`--self-test`（GPU 不要・合成 fixture） |
| `env_info.txt` | 実測時に `orchestrate.sh` が `<dir>/` へ採取（driver・CUDA・OS・rustc・commit・load・GPU 利用率・`layerA_same_code`）。本ファイルは実測前の空テンプレート |

出力ディレクトリ（`<dir>` = 既定 `gb10/`）: `run{1..5}/n{N}_{arm}.jsonl`・
`layerA-phases-N{N}.log`・`load_gate.log`・`load_gate_status.txt`・`env_info.txt`。

## 実行手順（GB10・別セッション）

```
./orchestrate.sh gb10
python3 aggregate.py gb10
```

判定結果を §13.4 へ転記する。`--dry-run` で実行コマンド列のみ確認できる。
