# GB10 の相手（PyTorch／cuBLAS）純カーネル時間と Python 側 checksum の診断

イシュー #3185。規則の正は `docs/perf/logs/opp-kernel-gb10-3185/RULE.txt`、診断本体は `scripts/bench/framework-compare/opp_kernel_cuda_diag.py`。

## 状態

**実測済み（2026-10-21 UTC。GB10。`.rev-stamp` = `c22b8c26e4ebd171cebd5c65346dc03435430e4c`）。** 5 run × 3 腕（fp32・tf32・cpu）・`cpu-blas-probe`・fandhe テスト 5 run × 2 種がすべて完走し、`aggregate` の fail-closed 照合（RULE 項 11。fp32／cpu 腕の checksum を 0.11.0 系列と照合）を通過した。`gate.log` は 26 件すべて GATE-PASS（load1 0.56〜0.99・GPU 利用率 0%）。生ログは `docs/perf/logs/opp-kernel-gb10-3185/gb10/`（集計は `aggregate.md`／`aggregate.json`）。

- 環境: torch 2.14.0+cu130（CUDA 13.0）・Python 3.12.3・numpy 2.5.4。`~/work/.venv-bench` は所有者承認（2026-10-21、イシューコメント）に基づき本作業で新規導入した（`gb10/venv-install.txt`）
- 逸脱なし。ただし系列外のスモークで `measure --run 0` が「run が不正」で拒否された（run は 1 以上。`gb10/smoke.txt`）。`--run 1` のラベルで再実行して通り、正式系列の `run1` とは出力先が別（`~/work/smoke-3185/`）で系列には数えていない
- `bench_py.py` の sha256 は RULE 項 3 と一致（`env.txt`）。ノード上の既存 `~/work/bench_py.py` は無かった

## 目的

相手（PyTorch・candle。どちらも cuBLAS SGEMM と読める）の計測窓には matmul・`.cpu().numpy()`・Python `checksum()` が混ざり、「相手のカーネルが速いのか、取り出しが速いのか」を判定できていない。次を数字で切り分ける。

- (a) cuBLAS FP32 の純カーネル時間に対する fandhe FP32 の位置（fandhe 単体は 0.9.0 の Layer B で N=2048 が 1.165 ms。`docs/perf/cuda-gemm-reuse-phase-breakdown.md` §12.4）
- (b) TF32（`allow_tf32` ON）を使わないと超えられない壁があるか
- Python checksum の ns／要素（Issue 本文の推定 約 2.7、範囲 2.4〜3.0。Rust 側は実測 0.51）の実測

burn（cubecl）の純カーネルは対象外。判定・閾値は設けない（record_only）。

## 方法

| 項目 | 内容 |
|---|---|
| Python 側 | `opp_kernel_cuda_diag.py measure`。腕 fp32／tf32／cpu、N=256〜4096、区間 `mm_pure`（wall）・`mm_event`（CUDA event）・`readout`・`checksum`・`window` ほか。`allow_tf32` は読み戻しを検査し全行に記録する |
| fandhe 側 | 既存 `#[ignore]` テストのログを `parse-fandhe` で読む（Rust は変更しない）。N>=1024 は Layer B（`gemm_reuse_phase_diag_select`、wall、20/20）、N<=512 は `gemm_small_launch_cost_diag`（L2S は wall、D 層は event、50/200） |
| 比 | 時計ごとに列を分ける。wall 対 wall、event 対 event。N>=1024 の fandhe には event 時計の値がないため「該当計測なし」 |
| 照合 | fp32／cpu 腕の checksum を 0.11.0 系列（`docs/perf/logs/framework-compare-0.11.0-remeasure/gb10/results-dgx-py-0.11.0-median5.jsonl`）と fail-closed で照合。tf32 腕は記録のみ |
| CPU BLAS 経路 | `cpu-blas-probe`（`DNNL_VERBOSE`／`ONEDNN_VERBOSE`・`torch.__config__.show()`・`/proc/self/maps` のライブラリ名） |
| 入力 | bench_py の SEED_A／SEED_B（Python 側）。fandhe 側テストは別シード（FP32 SIMT の時間はデータにほぼ依存しない前提） |

変えないもの: `bench_py.py`・相手側ベンチ・判定規則（`checksum_contract.py`・`compare_gemm_gate.py`）・#3179 のツールと規則。

## 結果

値は 5 run の中央値（`gb10/aggregate.md`。run 別の値と全区間はそちらを参照）。fandhe と torch では warmup／iters（fandhe 50/200 または Layer B 20/20、torch 20/20）と入力シードが異なる（FP32 SIMT の時間はデータにほぼ依存しない前提）。比は時計ごとに分けている。

### cuBLAS（PyTorch）の純カーネル時間と fandhe

| N | cuBLAS fp32 mm_pure [ms] | fp32 mm_event [ms] | cuBLAS tf32 mm_pure [ms] | tf32 mm_event [ms] | fp32 TFLOPS（対 31） | tf32 TFLOPS（対 53.3） | tf32/fp32（wall） |
|---|---|---|---|---|---|---|---|
| 256 | 0.0180 | 0.0170 | 0.0118 | 0.0103 | 1.86（0.06） | 2.84（0.05） | 0.656 |
| 512 | 0.0348 | 0.0337 | 0.0180 | 0.0164 | 7.71（0.25） | 14.89（0.28） | 0.518 |
| 1024 | 0.1372 | 0.1367 | 0.0676 | 0.0659 | 15.65（0.50） | 31.77（0.60） | 0.493 |
| 2048 | 0.9739 | 0.9846 | 0.4215 | 0.4253 | 17.64（0.57） | 40.76（0.76） | 0.433 |
| 4096 | 7.9627 | 7.9606 | 3.2933 | 3.3167 | 17.26（0.56） | 41.73（0.78） | 0.414 |

### fandhe FP32 カーネルとの比（時計ごと）

| N | fandhe wall [ms] | wall 比（fandhe/cuBLAS fp32 mm_pure） | fandhe event [ms]（D 層 dev_kernel_seg） | event 比（対 fp32 mm_event） | 備考 |
|---|---|---|---|---|---|
| 256 | 該当計測なし | 該当計測なし | 0.0146 | 0.858 | b2b 0.0123（比 0.722）。L2S kernel_wait（wall）は 0.0145 だが投入を含む同境界でないため wall 比に使わない |
| 512 | 該当計測なし | 該当計測なし | 0.0372 | 1.106 | b2b 0.0349（比 1.037）。L2S kernel_wait 0.0368 |
| 1024 | 0.1864 | 1.358 | 該当計測なし | 該当計測なし | Layer B（launch_issue+kernel_wait） |
| 2048 | 1.1693 | 1.201 | 該当計測なし | 該当計測なし | 同上（0.9.0 の 1.165 と整合） |
| 4096 | 10.0066 | 1.257 | 該当計測なし | 該当計測なし | 同上。run 別 9.79〜10.38 |

- 読み取り（判定なし）: N=256〜512 の event 時計では fandhe の純カーネル時間は cuBLAS fp32 の 0.86〜1.11 倍。N>=1024 の wall 時計では fandhe は 1.20〜1.36 倍。N>=1024 の fandhe event 値はこの診断では取れない（Rust の計装追加が要る）ため、wall 比に含まれる投入・同期のオーバーヘッド分を切り分けられない
- cuBLAS fp32 の到達率は N=2048〜4096 で公称 31 TFLOPS の 0.56〜0.57。fandhe の N=4096 は 10.0 ms（約 13.7 TFLOPS 相当、対 31 で 0.44）

### TF32 の壁

- tf32/fp32 の時間比は N=1024 で 0.49、N=2048 で 0.43、N=4096 で 0.41（wall・event とも同程度）。cuBLAS の TF32 は N=4096 で 41.7 TFLOPS（対 53.3 で 0.78）
- FP32 の cuBLAS（17.3 TFLOPS）に対し、TF32 は N>=1024 で 2.0〜2.4 倍速い。FP32 SIMT のまま TF32 の時間に届く壁はあるか、の材料: cuBLAS FP32 自体が公称 31 の 0.57 で止まっているため、FP32 側の到達上限は TF32 側の値より低い。TF32 の checksum は FP32 と全 N で bit 不一致（TF32 が効いた事実と整合。差は `aggregate.md`）
- `allow_tf32` の読み戻しは全行で腕と一致している（食い違えば停止する設計）

### Python checksum の ns／要素（GB10・fp32 腕）

| N | checksum [ns/要素] | 推定 2.7 との差 [ns] | checksum_hot（参考） |
|---|---|---|---|
| 256 | 2.571 | -0.129 | 2.458 |
| 512 | 2.556 | -0.144 | 2.464 |
| 1024 | 2.592 | -0.108 | 2.565 |
| 2048 | 3.245 | +0.545 | 3.237 |
| 4096 | 2.944 | +0.244 | 2.932 |

- 実測は 2.56〜3.25 ns／要素で、推定 2.7（範囲 2.4〜3.0）と N=2048 を除き範囲内。N=2048 は範囲上限 3.0 を 0.25 上回る。tf32 腕の checksum は 2.55〜3.25 で fp32 と同水準
- cpu 腕は N=512 で 5.89、N=1024 で 4.91 とばらつきが大きく、run 別でも 1.28〜1.66 ms のように揺れた（CPU 側の短時間計測は他負荷の影響を受けうる。参考値）

### 検算（window との整合）

fp32 腕で `mm_pure + readout + checksum` と `window` の比は N=256〜2048 で 0.994〜1.015（`aggregate.md` の Σ区間/window）。N=4096 は fp32 0.865・tf32 0.840 で、区間の和が window より小さい（窓内の checksum が単独計測より重い、または readout の初回転送の差が考えられるが、本診断では原因を特定していない）。食い違いは隠さず記録する。

### CPU 側 BLAS 経路

**確認できなかった。** `DNNL_VERBOSE=1`／`ONEDNN_VERBOSE=1` を付けた N=1024 の cpu `torch.mm` は stderr が空（0 byte。`gb10/cpu-blas/cpu_blas_stderr.txt`）で、根拠となる行がなかった。`/proc/self/maps` には `libnvpl_blas_*`・`libscipy_openblas64_*`・`libarm_compute*` が載るが、載っていることは `mm` がそれを呼んだ証拠にならないため経路は推測しない。`torch.__config__.show()` の Build settings には `BLAS_INFO=nvpl` とあるが、これも実行時経路の確認ではない。

## 受入基準との対応

| 基準 | 状態 |
|---|---|
| 1. cuBLAS 純カーネル時間（N × FP32・TF32、5 run 生ログ） | 達成（`gb10/run{1..5}/{fp32,tf32}.jsonl`） |
| 2. Python checksum の ns／要素と推定値との差 | 達成（上表。N=2048 のみ範囲外） |
| 3. fandhe FP32 単体と cuBLAS FP32 の比（時計ごと） | 達成。N<=512 は event 比、N>=1024 は wall 比。互いに欠けるセルは「該当計測なし」 |
| 4. CPU 側 BLAS 経路の確認可否 | 達成（確認できなかったと明記） |
| 5. 相手側ベンチと判定規則を変えない | 達成（該当ファイルは変更なし） |

## #3214・#3262 への引き継ぎ

両イシューへのコメントは投稿していない（RULE）。判定・閾値は置かず、材料だけを記す。

- #3214（FP32 カーネル候補の目標値）: 同一機体・同一時計での cuBLAS fp32 純カーネル時間は N=1024 で 0.137 ms、2048 で 0.974 ms、4096 で 7.96 ms（mm_pure。event もほぼ同値）。fandhe の現状（wall）は 0.186／1.169／10.007 ms で、cuBLAS 比 1.36／1.20／1.26 倍
- #3262（既定精度の判断材料）: cuBLAS は FP32 のままだと公称 31 TFLOPS の 0.50〜0.57 で頭打ち（N>=1024）。TF32 にすると同 N で 2.0〜2.4 倍速くなり（41.7 TFLOPS、対 53.3 で 0.78）、FP32 SIMT のままでは TF32 の時間に届かない。結果は bit 不一致のため REQ-2 の複合判定との関係は別途確認が要る
- 相手側ベンチの窓には、matmul に加え readout（N=4096 で約 1.1 ms）と Python checksum（約 49 ms）が含まれる。N=4096 の GPU 純カーネル 8.0 ms に対し window は約 67.6 ms で、窓時間の大半は matmul 以外である

## 残課題

- N>=1024 の fandhe event 時計による計測は Rust の計装追加が要るため本イシューの対象外（必要なら別イシュー化を提案する）
- N<=512 の fandhe wall（投入を含む同境界）の計測は既存テストに無い（同上）
- burn（cubecl）の純カーネルは対象外
- ノード上の作業ディレクトリ（`~/work/*-3185`・`~/work/smoke-3185`・`~/work/opp-3185`・`~/work/.venv-bench`）は残置している。後片付けは所有者判断
