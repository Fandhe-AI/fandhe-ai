# <cuda-node> の PyTorch CPU・CUDA の MLP 窓の区間内訳（#4149）

## 状態

**PR-1 時点: 実測なし。** 本ディレクトリは測る前に固定する部品（RULE 雛形・集計スクリプト・1 round の起動キット・観測専用 shim）だけを置く。
`RULE.txt`（PR-2・単独コミット）、`round{i}/`・`env/`・`aggregate.md`・`aggregate.tsv`（PR-3・<cuda-node> での実測）は後続で追加する。
実測値・内訳・分類・確度は、実測するまで書かない（推定値で埋めない）。#4149 は実測と記録の PR-3 で閉じる（部品の PR は `Refs #4149`）。
PR-1 は GB10 に到達できないため `--self-test`・単体テスト・構文検査までしか行っていない（GB10 での `env`／1 round の dry run は未検証）。

## 目的

GB10 の 3 セル（G-CPU-TRN・G-CPU-INF・G-CUDA-INF）の相手（PyTorch）の窓を、判定ではなく内訳として記録する（record_only。閾値なし）。

1. CPU: PyTorch が使う BLAS 実体（NVPL か oneDNN か別か）と、MLP train／infer の窓の区間別時間
   （純 mm〔dispatch 込み〕／非 GEMM の要素演算／残差〔dispatch・autograd・Python 固定費〕）。NVPL `cblas_sgemm` の直呼びとの差も出す。
2. CUDA: cuBLAS 呼び出しの種類と回数、CUDA Graph の有無、同期の固定費、infer 窓の区間別時間
   （issue〔enqueue〕／readout_sync〔D2H＋同期〕／checksum／GPU 側の幅〔event〕）。
3. 各値に証拠タグ（実測／推定／未測定）を付ける。残差は「推定」。到達不能な量は unavailable のまま空欄にする。

## 読み方の注意

- 時間は **計時腕**（`measure`）の値だけ。`profile` 腕（`torch.profiler`）は計測を摂動するため op 数・kernel 名と回数・CUDA API 回数（`cudaGraphLaunch` の有無）だけを使い、時間値は混ぜない。
  GB10 で CUPTI が使えない場合は unavailable＋理由として残す。
- cuBLAS／`sgemm_` の呼び出し回数と転送先ライブラリは、計時外の観測専用 shim（`probe/`）で数える。shim は計時腕では使わない（`run_round.sh` が `LD_PRELOAD` 未設定を検査する）。
- CUDA の窓は X が device 上にあり **H2D を含まない**（`bench_py.py` の infer と同一）。fandhe 側の窓は H2D を含むため、区間を並べるときは非対称に注意する。帰属の比較は #4150 の担当でここでは行わない。
- CUDA の `residual`（推定）は、enqueue と並行して進む GPU 実行待ちを含む。`issue_minus_gpu_span` の符号が launch 律速か GPU 律速かの目安になる（判定ではなく数値の記録）。
- `graph_replay_ref` は「相手が Graph を使った場合の床」を示す参考腕で、窓は eager のため窓の値としては使わない。
- CPU の純カーネル相当は `torch.mm(out=)` を 10 回まとめた 1 回あたりで、dispatch 費用を含む（残差は下限側に読む）。直呼び（NVPL `cblas_sgemm`）との差には ctypes の引数変換費が含まれ、torch ラッパー費としては帰属させない（負値もありうる）。
- BLAS 実体が NVPL 推定と食い違っても宣言は動かさず、実測時に `env/`（`torch.__config__.show()`・shim の出力）の事実をここへ注記する。直呼びは torch が NVPL を使っていなければ unavailable になる。
- 既知の事前情報（#3646）: GB10 の torch は `BLAS_INFO=nvpl`・`USE_MKLDNN=1`・SVE128 でビルドされている。実際にどちらを通るかは shim で確かめる。
- ノイズ帯 NB-RANGE5 は `docs/perf/noise-band-decision.md`（案・所有者承認待ち。#4142）。承認後に定義が変われば `RULE.v2.txt` で反映する。
- 同じ GB10 を #4147 の実測も使う。実機ロックで直列化し、並走させない。

## ファイル

| ファイル | 役割 |
|---|---|
| `RULE.template.txt` | 測る前に固定する規則の雛形（`aggregate_4149.py --check-template` で検査） |
| `aggregate_4149.py` | 厳格パースの集計（`--dir`・`--self-test`・`--check-template`・`--check-rule`・`--fingerprint`）。sha256 を RULE に固定 |
| `run_round.sh` | 1 round の起動（ゲート → env → CPU／CUDA measure → profile） |
| `probe/probe_shim_4149.c` | 観測専用 shim（sgemm_／cblas_sgemm／cuBLAS GEMM 入口の回数と転送先）。sha256 を RULE に固定 |
| `probe/probe_driver_4149.py`・`probe/run_probe.sh` | shim 越しに MLP 1 step を実行して回数を出す計時外の経路確認 |
| `scripts/bench/framework-compare/torch_gb10_mlp_diag.py` | 計測本体（`env`／`measure`／`profile`・`--self-test`）。sha256 を RULE に固定。#4148 の `torch_cpu_mlp_diag.py`・#3179 の `opp_kernel_cuda_diag.py` を importlib で再利用（この 2 つの sha256 も RULE に固定し env.json で照合） |
| `scripts/bench/framework-compare/torch_gb10_mlp_diag_test.py` | torch 不要の単体テスト（CI 未接続・手動実行） |

## 手順

1. （PR-2）PR-1 のマージ後の main SHA の git archive で `tree_sha`・`tree_fingerprint`・各 sha256 を埋め、
   `python3 -I aggregate_4149.py --check-rule RULE.txt` でプレースホルダ 0・sha 一致を確認して `RULE.txt` を**単独コミット**する。
2. （PR-3・<cuda-node>）実機ロックを取り、ツリー指紋を RULE と照合する。
   `ROOT=… PY=… BP=… EXPECT_SHA=… HW_LOCK_OWNER_FILE=… HW_LOCK_HELD=1 bash run_round.sh <i> <記録先>/round<i>` を 1〜5 で回す。run の差し替え・除外・上書きは禁止。
3. 全 round の後に `PY=… bash probe/run_probe.sh <記録先>/env/probe.txt` を 1 回実行する（計時外）。
4. 回収時にパス上のユーザー名・ホスト名をマスクし（`<home>`・`<cuda-node>`）、`env/` にツリー指紋・`.rev-stamp`・`torch.__config__.show()`・`nvidia-smi` 概要を置く。
5. `python3 -I aggregate_4149.py --dir <記録先>` で `aggregate.md`・`aggregate.tsv` を生成し、下の結果欄と
   `docs/perf/loss-attribution-matrix.md` の G-CPU-TRN・G-CPU-INF・G-CUDA-INF の相手欄を実測値＋タグで更新する（未測定は未測定のまま）。

## 結果

（実測後に追記。PR-1 時点は空。）
