# TMA Stage 2（候補 C1）GB10 A/B 実測記録（イシュー #3215）

#3214 で追加した候補 C1（TMA 2 段 128x64。腕 `tma128_none`・`tma128_strided`）が、本番の現行カーネルより速いかを
DGX Spark GB10 で事前登録規則どおりに判定した記録。実測日は 2026-10-22。規則の正は `RULE.txt`（計測前にコミットして固定）。
集計と判定語は `aggregate.py` が収録ログから機械生成した `aggregate.md` を正とし、本 README は要約と運用記録だけを書く。
設計は `docs/backend-cuda-tma-gemm-load-design.md` §11.6・`docs/cuda-sm121-gemm-candidates-design.md` §4・§7、
届かないセルの 3 分類は `docs/perf/loss-attribution-matrix.md` §10.3.3・§10.5.3・§10.8。

## 結果（要約。数値の正は `aggregate.md`）

- ゲート A: 128x64 の 3 件と前提の 1 件が全て ok。BITIDENT（全 N・5 run）は全腕で一致。ゲート B: 6 バイナリで FAILED 0 件。
- ゲート C（結線範囲は N=1024・2048・4096。比 = cur ÷ 候補、5 run 中央値、>1 が改善）:

| 腕 | N=1024 | N=2048 | N=4096 | 判定 |
|---|---|---|---|---|
| `tma128_none` | 1.1452 | 1.1241 | 1.0354 | ADOPT（結線候補。ゲート D は結線 PR で実施） |
| `tma128_strided` | 1.1750 | 1.1327 | 1.0376 | ADOPT（結線候補。ゲート D は結線 PR で実施） |

  全 N で 5/5 run 一貫（反対側の run は 0）。N=256 は cur（64x64）比 0.68〜0.76、N=512 は 1.07。この 2 点は結線範囲外で判定に使っていない。
- 純カーネル（event）の TFLOPS（cur／`tma128_strided`）: N=1024 11.84／13.91、N=2048 14.49／16.42、N=4096 13.29／13.79。
  cuBLAS FP32 は 15.69／17.36／17.24。`tma128_strided` ÷ cuBLAS FP32 は 1.128／1.057／1.250（cur は 1.325／1.198／1.297）。
- gemm CUDA セル（現行のみ）の対最速相手の比（reuse）は 0.832／0.427／0.405／0.478／1.010（N=256〜4096）。0.11.0 の 0.85／0.42／0.40／0.48／1.01 と同水準で、
  版差による動きはない。ADOPT 腕を結線した場合のセルは `aggregate.md` に**推定**として併記した（測定値ではない）。
- 本 issue は結線しない。結線 PR（形状分岐・tensor map キャッシュ・ゲート D〈同一 HEAD の base／after を framework-compare gemm cuda で比較〉）の起票は所有者判断とし、
  本 PR では起票していない。N=256 の後退（0.68〜0.76）があるため、結線は N の下限ゲートを要する（N=512 は 1.07 で、結線範囲に入れるかは結線 PR のゲート D で決める）。

## 規則からの逸脱と運用記録（隠さない）

1. **`cur` 腕の起動 API**: 計画書の `launch_tiled_f32_pooled(Select)` は `cfg(test)` 限定で example から到達できないため、`CudaGemm::launch_tiled_f32`
   （`select_tiled_f32_kernel(0, m, n, k)` を同一に呼ぶ）を使った。出力確保は計測区間外で差は生じない。RULE.txt 項 2 に計測前から記載。
2. **計測基盤の形**: 診断は crate ルートの `#[cfg(test)]` モジュールではなく `examples/gemm_candidate_event_ab_3215.rs`（`internal-diagnostics` 限定・`test = true`）にした。
   必要な API（`launch_tiled_pipeline_*`・`prepare_tiled_pipeline_tma_maps`・`device.stream()`）が全て公開（feature 限定）で、`lib.rs` を触らずに済むため。
3. **セル計測スクリプトの修正（2 回失敗した後に成功）**: 計測ツリーは `31cd381f0`（`rev-stamp.txt`。RULE.txt を含むコミット）。ゲート A・B・純カーネル・cuBLAS は
   このツリーで実行した（`tree-fingerprint-node.txt` と `tree-fingerprint-mac.txt`〈同 SHA の `git archive`〉が一致）。セル計測の初回は `run_cell_gemm_cuda_3215.sh` が
   `LC_ALL=C` 下の `cargo tree` 罫線を grep できず依存元の取得で fail-closed に停止した（`cell-attempt1-failed/`・`stage_rc.txt` の `cell rc=1`）。`--charset utf8` を足した
   1 回目の修正でも括弧式が C ロケールでバイト解釈されて同じ停止（`cell-attempt2-failed/`・`cell-attempt2-failed.stdout.log`）。`LC_ALL=C` を外した最終版
   （コミット `977e060d1`。`rev-stamp-cell.txt`）で 5 round 完走（`cell/`・`cell.stdout.log`）。修正はスクリプトのみで `crates/` は無変更のため、
   セルの計測対象ソースは `31cd381f0` と同一。計測値は 2 回の失敗から一切流用していない（失敗時の `results.jsonl` は 0 行）。
   初回の失敗ログ（`cell.stdout.log` の旧内容）は再実行時に上書きされたため残っていない。`stage_rc.txt` の末尾 `cell_rerun rc=0` は最終版の終了コードを手で追記した行。
4. **ノード側の Cargo.lock**: 診断用に手で実行した `cargo tree` がノード上の `framework-compare/Cargo.lock` を書き換えたため、計測前に Mac 側の `git archive` の内容へ戻した
   （md5 一致を確認。成果物には残さない。`Cargo.lock` はツリー指紋の除外対象）。
5. **ツリーの転送**: `git archive <SHA>` のパイプ（追跡ファイルのみ）。`docs/spec`（submodule）は含まれない。
6. 専有ゲートは全て通過（ゲート A・B・純カーネル・cuBLAS の 22 回は `load_gate.log`、セル 5 round は `cell/load_gate.log`）。環境は driver 580.173.02・CUDA 13.0・compute_cap 12.1・rustc 1.97.0・torch 2.14.0+cu130（`env_info.txt`）。ホスト名・ユーザー名・ホームは `<node>`・`<home>` に伏せた。

## 保存物

`RULE.txt`・`orchestrate_gb10.sh`・`tree_fingerprint.py`・`aggregate.py`（`--self-test` 付き）・`aggregate.md`・`env_info.txt`・`load_gate.log`・`stage_rc.txt`・`build.log`・
`gateA_*.log`・`gateB_*.log`・`kern_run{1..5}.log`・`cublas/run{1..5}/{fp32,tf32}.jsonl`・`cell/`・`rev-stamp*.txt`・`tree-fingerprint-{node,mac}.txt`・`uptime_*.txt`。

## 再現

```sh
# Mac: 計測 SHA の追跡ファイルだけを GB10 へ送り、.rev-stamp を置く（秘密情報・.env*・*.local.md を含めない）
git archive <SHA> | ssh <GB10> 'tar -x -C ~/work/stage2-3215-tree'
# GB10: 一式を切り離して実行し、固有マーカー ORCHESTRATE-DONE で完了を待つ
REPO_DIR=$HOME/work/stage2-3215-tree OUT=$HOME/work/stage2-3215-out VENV_PY=$HOME/work/.venv-bench/bin/python \
  setsid nohup bash docs/perf/logs/cuda-tma-stage2-ab-gb10-3215/orchestrate_gb10.sh > $HOME/work/stage2-3215-orch.log 2>&1 &
# 集計（収録ログだけで決定的に再生成できる）
python3 -I aggregate.py --dir . --out aggregate.md
python3 -I aggregate.py --self-test
```
