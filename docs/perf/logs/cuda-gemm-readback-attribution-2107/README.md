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
| `aggregate.py` | `python3 aggregate.py <dir>` で `aggregate.md` を生成。`--self-test`（GPU 不要・合成 fixture）。帰属判定は本番順序の 2 腕（`prod_order_fresh`／`prod_order_reused`。起動後に同期しない）の readback 壁時計差（exposed cost）のみを使い、直列腕（起動直後に同期する 5 腕）は補助診断・記録のみ |
| `check_layer_a_path_identity.py` | Layer A（fandhe-ai =0.9.0）と HEAD の計測経路項目（tile 選択・カーネルソース生成・readback・生成／起動経路。ファイル全体の diff ではない）の同一性検査（コメント・診断 feature ゲート項目を除いた正規化比較。生成／起動経路は人手レビュー済み sha256 一致のみ許容）。`orchestrate.sh` が呼び出し `layerA_same_code` を決める |
| `env_info.txt` | 実測時に `orchestrate.sh` が `<dir>/` へ採取（driver・CUDA・OS・rustc・commit・load・GPU 利用率・`layerA_same_code`）。本ファイルは実測前の空テンプレート |

出力ディレクトリ（`<dir>` = 既定 `gb10/`）: `run{1..5}/n{N}_{arm}.jsonl`・
`layerA-phases-N{N}.log`・`load_gate.log`・`load_gate_status.txt`・`env_info.txt`。

## 腕の 2 群と判定の範囲

- 直列腕: 区間内訳（`dest_alloc`／`pretouch_fill`／`d2h_*`）を見る補助診断。起動直後の同期で
  CPU 側の宛先準備と GPU カーネル実行の重なりを除くため、本番順序の値ではない。
- 本番順序腕: `prod_order_fresh`（本番 `memory::readback`）と `prod_order_reused`（確保済み・
  タッチ済み宛先。exposed cost を出す計測対照であり施策の再評価ではない）。差が
  exposed cost。判定が示すのは「本番順序で表に出た費用が residual に占める割合」のみで、
  重なりに隠れた費用・Layer B が再現しない Layer A 側の費用は検出対象外（重なりの大きさは
  直列腕との差として記録のみ）。

## 実行手順（GB10・別セッション）

```
./orchestrate.sh gb10
python3 aggregate.py gb10
```

判定結果を §13.4 へ転記する。`--dry-run` で実行コマンド列のみ確認できる。
