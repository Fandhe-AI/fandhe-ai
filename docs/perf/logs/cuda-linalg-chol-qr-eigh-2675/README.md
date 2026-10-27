# CUDA 分解系線形代数（cholesky・QR・eigh。#2675）GB10 実機 parity 実測記録・Stage A／B の履歴

> **実測結果（#2675 reopen・2026-10-27）**: DGX Spark GB10（`<cuda-node>`・sm_121）で分解系の parity を実測した。**5 run すべて pass**（backend 層 `running 6 tests` が 6 passed; 0 failed、facade 層（ゲート ON）`running 3 tests` が 3 passed; 0 failed）。事前登録規則は実測前にコミット済みの `RULE.txt`、生ログは `run1.log`〜`run5.log`、環境は `env.txt`、対象コミットは `rev-stamp.txt`（`f175ea28f56180c6034f85804658ed5d120a32fe`・`dirty=0`）。tolerance・カーネル・入力集合・判定方式は不変。LU 系（#2674）と #3138 のほかのコマンド（`cuda_eigh`／`cuda_pinv` のゲート OFF 経路を含む）は範囲外。
>
> **ログの注記**: 各 run ログの `rc=` 行は、シェル記述の誤りにより cargo の終了コードではなく直前の `echo` の終了コードを記録している（常に 0）。したがって `rc=` は判定根拠にしない。判定は各ログの `running N tests` と `test result: ok. N passed; 0 failed` の行（5 run × 2 コマンドの全 10 件で期待件数と一致）による。ログは改変していない。

> **更新（#3136）**: Stage B（起動の実装）は実装済み。D-C 承認（親 #3133）と `docs/gpu-linalg-kernel-launch-unsafe-audit.md`（CU-3〜CU-5）に基づく。実機（GB10）での parity 実測は **#3138 へ申し送り・未実施**（本実行環境は GPU なし）。以下は Stage A 時点の記録で、「承認待ち」の記述は履歴として残す。

> **#3138**: 実機 parity の事前登録規則・手順を `../cuda-linalg-parity-3138/` に用意した（未実測。GB10 での実施待ち）。

`docs/backend-gpu-linalg-decision.md` §3〜§6・§10 参照。本実装エージェント実行環境は CUDA 実機（DGX Spark GB10）に
到達できず、また **カーネル起動に必要な新規 `unsafe` のユーザー承認がイシュー #2675 に確認できなかった**ため、
実装は #2674（LU 系）と同じ「Stage A」（`unsafe` なし）までで止めている。

## 実装済み（Stage A）

- `crates/backend-cuda/src/kernels_linalg_factor.rs`: Cholesky・reduced Householder QR・巡回 Jacobi eigh の `double`
  カーネルソース（1 行列 = 1 block・`__d*_rn` intrinsic による非縮約 FMA 契約・手動境界チェック維持・縮約と収束判定は
  逐次で CPU の算術順序と同一）。ソース走査テスト付き。
- `crates/backend-cuda/src/linalg.rs`: サイズ検証 `validate_square_factor_dims`／`validate_qr_dims`、ホスト後処理
  `normalize_max_abs_sign`／`eigh_sort_and_normalize`／`qr_finalize`（`backend-cpu` の同名処理の意図的複製）、
  NVRTC コンパイル・ロード `CudaLinalgFactor::new`。ゲートは #2674 の `set_gpu_linalg_enabled` を共有（既定 OFF・facade 非公開）。
- `ops.rs`: `linalg_cholesky`／`linalg_qr` を置換し、`linalg_eigh` のオーバーライドを新設（CUDA ではこれまで trait 既定に
  落ちていた）。ゲート OFF は従来どおり `Unsupported`。ゲート ON は rank・正方性・サイズ検証（デバイス非接触）→
  空入力はホストで CPU と同じ形 → それ以外は fail-closed の `Unsupported`（起動は承認待ち）。`Var` はホスト `f64` へフォールバックする。

## 承認依頼（Stage B: 新規 `unsafe` 3 ブロック）

cudarc の `launch_builder(..).launch(cfg)` は `unsafe fn`。次の 3 ブロックのみを追加する承認を求める。

- ブロック 1: `linalg_cholesky_f64` の `launch`（grid `(1,1,1)`・block `FACTOR_BLOCK_DIM`）。
- ブロック 2: `linalg_qr_f64` の `launch`（同上）。
- ブロック 3: `linalg_eigh_f64` の `launch`（同上）。
- SAFETY 根拠: 各バッファ長は検証済みの次元から確保、`i32` 範囲は検証済みで切り詰めなし、カーネル内に手動境界チェック
  （REQ-8）、単一 block のため block 間同期は不要、`status` は必ず読み戻して判定する。

承認後の作業: 3 ブロックと `run_*` 関数・`context_cache` への `cached_linalg_factor` 追加、`factor_launch_pending` を
実起動へ置換（H2D は f32→f64 変換、D2H 後に `linalg.rs` のホスト後処理と 1 回の `as f32`）、`status[0] != 0` を
cholesky は「対称正定値でない」・eigh は「巡回 Jacobi 非収束」の `InvalidArgument` へ写像。

## 測定コマンド（実測で使用した形。`RULE.txt` と同一）

```sh
CARGO_TARGET_DIR=$HOME/work/target-fandhe-ai cargo test -p fandhe-ai-backend-cuda --release --test linalg_factor_parity -- --ignored --nocapture --test-threads=1
CARGO_TARGET_DIR=$HOME/work/target-fandhe-ai cargo test -p fandhe-ai --release --test linalg_gpu_optin_cuda_parity -- --ignored --exact --nocapture --test-threads=1 facade_cholesky_gate_on_matches_cpu facade_qr_gate_on_matches_cpu facade_eigh_gate_on_matches_cpu
```

## 実測記入欄

| 日付 | 実機 | コマンド | 結果 | 備考 |
|---|---|---|---|---|
| 2026-10-27 | DGX Spark GB10（`<cuda-node>`・sm_121） | 上記 2 コマンド × 独立 5 run | **5 run すべて pass**。backend 層 6 件（NVRTC ロード・cholesky／QR／eigh の CPU 一致・非 SPD と eigh 非収束の `InvalidArgument` 写像）と facade 層 3 件（cholesky／qr／eigh のゲート ON）が毎 run 全 pass（`6 passed; 0 failed`／`3 passed; 0 failed`）。出力に NaN／inf は無し（ログ全文を検索。0 件） | 対象 `f175ea28f56180c6034f85804658ed5d120a32fe`（`dirty=0`）。ドライバ 580.173.02・CUDA 13.0（nvcc 13.0.88）・rustc 1.97.0。hw-lock 保持 2026-10-27T11:16:08Z〜11:16:34Z（UTC）。計測前後とも他プロセスの常駐（LLM サービス・画像生成の GPU メモリ使用）はあったが GPU 使用率は 0%。tolerance・カーネル・入力集合は不変。`rc=` 行は上記注記のとおり判定に使わない。下の既存 2 行の「未実測」のうち分解系の分は本行が引き継ぐ（LU 系と #3138 のほかのコマンドは対象外） |
| 未実測 | DGX Spark GB10 | 上記 | 未実測 | NVRTC コンパイル確認のみ。parity は Stage B（unsafe 承認）後 |
| 未実測 | DGX Spark GB10 | `../cuda-linalg-parity-3138/README.md` のコマンド | 未実測（#3138 スキャフォールド PR） | 規則は `../cuda-linalg-parity-3138/RULE.txt` |
