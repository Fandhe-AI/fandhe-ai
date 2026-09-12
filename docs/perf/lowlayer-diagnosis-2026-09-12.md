# 低レイヤー診断（GB10／M4 Max 実測。2026-09-12）

## 1. 目的・系列

低レイヤー診断 artifact（`tensor-core`／`autodiff`・`backend-cpu`／`cuda`／`metal`・
`facade` の仕組みを読んで性能余地・便利機能の追加余地を整理したもの。
https://claude.ai/code/artifact/4e107064-a190-4861-897d-3dce44d05428
）§1〜§6 の設計読解で残った不確実点（§6）を、DGX Spark GB10 と Apple M4 Max の
実測で埋めた記録が同 artifact §7 であり、本ドキュメントはその実測部分を
**数値を変えずに**転記する。対象コードは origin/main HEAD `097bff19`
（#1556 Metal resident grad staging を含む）。

**系列はすべて「診断・方向づけ」用であり、ADOPT／REJECT の正式判定ではない**
（framework-compare は単一起動・ハーネス内 20 反復、スレッド sweep は 3 起動、
backward 内訳は 5 起動。詳細は §6）。

| 系列 | 機体 | コード | 負荷 | 用途 |
|------|------|--------|------|------|
| DGX 0.8.0 | DGX Spark GB10（専有・sm_121） | ピン `fandhe-ai =0.8.0`（crates.io 公開版） | load average 0.03〜4.45（5 点実測。注 1） | framework-compare 全 task・diag テスト・RAYON sweep |
| Mac HEAD | Apple M4 Max（共有） | origin/main HEAD `097bff19` path patch＋backward 内訳計装（`f91cafa3` からの差分は Metal split-K トグル・#1556 のみ。CPU 計測経路は同一） | load ≈ 4 | backward 内訳・train phases・readout interleave |

注 1: DGX 0.8.0 の負荷は単一値ではなく計測ステージごとに変動する。実測は
`docs/perf/logs/lowlayer-diagnosis-2026-09-12/dgx/uptime_{before,after}_*.txt`
（1 分平均）: `uptime_before_all.txt` 1.05・`uptime_after_all.txt` 1.76・
`uptime_after_sweep.txt` 1.56・`uptime_before_run2.txt` 0.03・
`uptime_after_run2.txt` 4.45。専有環境ではあるが他ログイン（`users` 8〜9）が
残存しており、run2（大コア pin sweep・追加 CPU gemm 計測）の前後で
最大 4.45 まで上昇している。全系列を一律 load ≈ 0.05 として扱った当初の
記載は誤りであり、当該区間の framework-compare／診断テスト結果は
完全な専有条件下の値ではない点に留意する。

実行コマンド・オーケストレーションは
`docs/perf/logs/lowlayer-diagnosis-2026-09-12/scripts/`
（`dgx-prebuild.sh`／`dgx-run.sh`／`dgx-run2.sh`／`dgx-venv.sh`／`dgx-py.sh`／
`mac-diag.zsh`）を参照。診断専用計装は
`docs/perf/logs/lowlayer-diagnosis-2026-09-12/diag-instrumentation.patch`
（本番実装ではない）。

## 2. GB10 の CUDA 実測（§1「CUDA も同じ根因」の解消）

出典: `docs/perf/logs/lowlayer-diagnosis-2026-09-12/dgx/results-dgx-0.8.0.jsonl`・
`results-dgx-py-0.8.0.jsonl`・`run_all_cuda.log`・`async_ordering_real_device.log`・
`tma_probe_real_device.log`・`gemm_transposed_parity.log`・
`gemm_transposed_perf.log`。

| task | fandhe fresh | fandhe reuse | candle | PyTorch cuda | fandhe reuse 時間 ÷ 最良時間 |
|------|--------------|--------------|--------|--------------|------------------------------|
| train cuda（ms/step） | 0.546 | 0.469 | 0.275 | 1.272 | 1.71（candle 比） |
| infer cuda（ms） | 0.152 | 0.141 | 0.043 | 0.041 | 3.4 |
| gemm cuda reuse 判定比（candle 時間 ÷ fandhe 時間。1 以上で達成） | — | — | — | — | N=1024 0.431・N=2048 ≈ 0.51・N=4096 ≈ 1.01 |

- **train cuda phases（µs）**: forward 177／151（fresh／reuse）・backward
  220／231・param_readout 75・host_sgd 66・device_update 91。推論 forward は
  2 層で 132 µs（内訳未分解。「層境界ごとのホスト往復が支配的」はコード読解に
  よる帰属で、実測は合計のみ）。Metal 限定と見ていた候補（推論チェーン単一
  同期化）は両 GPU 共通の候補になる。
- `has_async_alloc()` = `true`（`docs/backend-cuda-async-execution-design.md`
  I4 の未実測を解消）。TMA は cluster／cta 両 variant が `compute_121`／
  `121a`／`121f` でコンパイル・実行 bit 一致。#1214 NT／TN 入口は parity
  5/5・N=1024/2048 で 2.8〜5.9 倍（train A/B は未実施。§7 参照）。
  async ordering 3/3 pass。
- GB10 CPU gemm（fresh）は candle・burn・PyTorch 2.14・TF 2.21・SciPy の
  全てに勝つ（N=256〜4096）。reuse 判定比は 256 0.697・512 0.850・
  1024 1.040・2048 1.033・4096 1.210。「CPU は AMX に負ける」は Mac 限定の
  話で、SME カーネル候補は引き続き Mac 側の答えのまま。

## 3. 小形状スレッド方針は機体で逆向き

出典: `docs/perf/logs/lowlayer-diagnosis-2026-09-12/dgx/rayon-sweep.jsonl`・
`rayon-sweep-pinned.jsonl`。

| GB10（ms・3 起動中央値） | T4 | T8 | T10 | T20 |
|--------------------------|----|----|----|----|
| train reuse 無 pin | 2.651 | 2.533 | 2.359 | 1.029 |
| train reuse 大コア pin（`taskset -c 5-9,15-19`） | 1.900 | 1.710 | 0.857 | — |
| infer fresh 無 pin／pin | 0.299／— | —／0.129 | — | 0.182／— |

M4 Max は 4 スレッド上限が 12 より 1.3〜1.5 倍速いが、GB10 は上限を
下げるほど遅く、効くのは大コア親和性（pin T10 が無 pin T20 より
1.2〜1.3 倍速い）。一律の「形状別スレッド上限」は GB10 A/B で不成立。
方針は「M4＝上限」「GB10＝親和性」の 2 系統に分けて、それぞれ自機のみで
採否判定する（`cpu_capacity` 誤検出で REJECT 済みの #1364 とは別経路が
必要。GB10 の `cpu_capacity` 実値は 718／731／1017／1024／997 の 5 段階）。

## 4. backward の非 GEMM 残差

出典: `docs/perf/logs/lowlayer-diagnosis-2026-09-12/mac/train-phases.jsonl`・
`diag-{cpu,metal}-{fresh,reuse}-run{1..5}.err`（`FANDHE_DIAG_BACKWARD=1`
計装出力）。

| M4 Max・5 起動中央値 | backward 合計 µs | gemm % | loss µs | mask µs | 非 GEMM % |
|----------------------|-------------------|--------|---------|---------|-----------|
| cpu fresh | 505 | 76.4 | 71 | 8.7 | 24 |
| cpu reuse | 576 | 77.7 | 67 | 45 | 22 |
| metal fresh | 839 | 82.3 | 124 | 9 | 18 |
| metal reuse | 425 | 65.8 | 95 | 44 | 34 |

非 GEMM は 18〜34% で、MSE backward と reuse 時のマスクが 2 大項目。
機構はマイクロ計測で確定した。

- **MSE backward ≈ 70 µs は rayon の fork-join 固定費**。
  `mse_loss_backward_f32` は `par_iter_mut().zip().zip()` で 640 要素を
  並列化しており、逐次ループなら 50 ns 未満・`RAYON_NUM_THREADS=1` でも
  3.9 µs・16 スレッドで 75 µs（確保・`Tensor` 構築は 165 ns 未満。
  共有負荷 load ≈ 4 下の値で、専有環境では絶対値は縮む見込み。しきい値で
  逐次へ落とす方向自体は不変）。`CHUNK=4096` の `par_chunks` は forward の
  二乗和側だけで、backward には当てはまらない。対策は要素数しきい値で
  逐次へ落とす、または GPU では encode-only 化（Metal は
  `mse_loss_backward` が `dispatch_sync`）。
- **reuse のマスク 5 倍は非連続 transpose view が原因**。下流層の VJP が
  返す `d_input = transpose2d(&tmp)`（`grad.rs:346`）はゼロコピー view で、
  単一寄与なら accumulate がコピーせずそのまま上流へ渡す。
  `elementwise_mul_mask` の `dense_vec` は非連続入力を要素ごと
  `get(&index)` で走査するため、連続比 49 倍（667 ns → 32.6 µs／16384
  要素）。実配線でも `upstream.strides()=[1,4]`・`out_value` は連続を確認。
  対策はマスク側で stride 対応（またはマスク適用を GEMM 入口の転置と
  融合）で、bit 同一。
- **train phases（µs）**: cpu fresh step 854／backward 533・reuse
  999／606・device_update 121。metal fresh 1717／868（forward 771）・
  reuse 1169／459（forward_resident 559）。CPU reuse 逆転（GB10 でも
  0.892 → 1.175 ms）は上記マスク（非連続 view 走査）と device_update で
  説明する。診断行の `gemm_calls`（fresh 2・reuse 4）は計装単位の差であり
  GEMM 呼び出し回数の増加ではない: fresh は `matmul_vjp` 全体を 1 回と
  数える（内部で d_input・d_weight の GEMM を 2 回呼ぶ）のに対し、reuse は
  `Op::LinearResident` の個々の GEMM 相当処理を数えるため、実際の GEMM
  回数は両モードとも同じ 4 回である。

## 5. Metal readout legacy の局所化（#1520 の続き）

出典: `docs/perf/logs/lowlayer-diagnosis-2026-09-12/mac/readout-phases.jsonl`。

N=1024 Metal reuse を legacy／borrowed で 5 回 interleave: matmul 区間
1.694 → 2.161 ms、host_copy 0.233 → 0.000 ms、checksum ≈ 0.53 で同じ、
iter_total 2.470 → 2.677 ms。後退は GPU 待ち＋ダウンロードを含む matmul
区間に閉じており、host_copy 削減では埋まらない。原因は未特定（CUDA
#1436 と同型の「borrowed＋ダミー確保・解放」腕が次の切り分け候補）。

## 6. 証拠等級と限界

- 本記録は**診断・方向づけ用**であり、ADOPT／REJECT の正式判定を含まない。
  正式判定は各後続 issue（§7 参照）で、専有ゲートまたは record_only 明記の
  実測プロトコルに従って別途行う。
- framework-compare の全数値（§2・§3）は**共有負荷下**（Mac HEAD は
  load ≈ 4）または**専有だが単一起動**（DGX 0.8.0 は §1 注 1 のとおり
  load average 0.03〜4.45 で変動しており厳密な専有条件下ではない・
  ハーネス既定の内部反復のみで複数プロセス起動の中央値ではない）。
  スレッド sweep（§3）は 3 プロセス起動中央値、backward 内訳（§4）は
  5 プロセス起動中央値。
- 推論 forward のホスト往復支配の帰属（§2「層境界ごとのホスト往復が
  支配的」）はコード読解による帰属であり、実測はフェーズ合計値のみで
  内訳分解は行っていない。
- Metal readout legacy の後退機構（§5）は matmul 区間への局所化までで、
  根本原因は未特定のまま次の診断へ引き継ぐ。

## 7. 確定した方針（issue ツリー草案の改訂）

| 草案 | 改訂後 | 根拠 | 区分 | issue |
|------|--------|------|------|-------|
| A-1 形状別スレッド上限 | A-1a M4: 小形状スレッド上限／A-1b GB10: 大コア親和性 | §3 | 中立・自機判定 | 1575／1576 |
| A-2 CPU reuse 逆転の調査 | A-2 マスクの stride 対応（bit 同一）＋backward gemm 回数の整理 | §4 | 中立 | 1577 |
| A-3 Metal 推論 1 コマンドバッファ化 | A-3 GPU 推論チェーンの単一同期化（CUDA／Metal 共通）＋MSE backward encode-only | §2・§4 | 中立（failure_token） | 1579／1580／1581／1582 |
| A-4 thread_elements | 不変 | — | REQ-2 判定 | 1586 |
| A-5 SME | 不変（Mac 限定の答え） | §2 | 承認要（`unsafe asm!`） | 1587 |
| A-6 GB10 診断 | 完了 → TMA Phase B を設計 issue として起票（実装は含めない） | §2 | 診断済み | 1589 |
| A-7〜A-10・B-3 | 不変（A-9 は「マスク先行」を A-2 へ統合） | — | 各 issue のとおり | 1584／1583／1585 |
| 新 A-11 MSE backward の逐次しきい値 | CPU: 要素数しきい値で rayon を回避（bit 同一） | §4 | 中立 | 1578 |
| 新 D-1 Metal readout matmul 区間の切り分け | borrowed＋ダミー確保・解放腕 | §5 | 診断 | 1588 |

本節の数値は方向づけ用。`docs/perf/` への正式記録（5 起動・専有ゲートまたは
record_only 明記・内部ホスト名なし）は起票後の各 issue で行う。

## 8. 出典

- 低レイヤー診断 artifact §7:
  https://claude.ai/code/artifact/4e107064-a190-4861-897d-3dce44d05428
- 機能ギャップ表: `docs/compat-feature-gap.md`（同 artifact §2 由来）
- 生ログ・生データ・オーケストレーションスクリプト:
  `docs/perf/logs/lowlayer-diagnosis-2026-09-12/`（`README.md` にファイル
  構成を記載）
- 差し戻し記録: `scripts/bench/framework-compare/results/raw/results.jsonl`
  （追跡ファイル）への Mac HEAD path patch 計測時の誤追記 63 行は
  `git checkout --` で差し戻し済み。当該 63 行のバックアップは
  `docs/perf/logs/lowlayer-diagnosis-2026-09-12/mac-head-097bff19-path-patch.jsonl`
