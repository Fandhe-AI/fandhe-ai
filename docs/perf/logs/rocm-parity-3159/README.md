# ROCm（MI300X・gfx942）対 CPU の f32 GEMM parity（イシュー #3159）

**状態: 未実測**（所有者のインスタンス起動待ち）。本ディレクトリの現状は段 A（準備のみ）で、下の記入欄はすべて空欄。得ていない結果は書かない。

## 対象範囲

- 測るのは `RocmGemmF32::run`（f32 GEMM・`GfxTarget::Gfx942`）と `is_available()`（`probe_real_device.rs`）のみ。
- reduce・softmax・norm など `docs/backend-rocm-ops-coverage-plan.md` の R0〜R13（linalg〈R13〉の kernel 実行器は #3444 で実装済み。下の「linalg」節。`BackendOps` 結線は R0 待ち）は未実装（`BackendOps` 未実装・起票は所有者承認待ち）のため対象外。各 R イシューが同じ `#[ignore]` 方式でテストを足す。
- 判定は REQ-2 統一複合判定（相対 1e-3 未満 または 絶対 1e-5 未満）のみ。tolerance・baseline・FMA 契約は変えない。規則は `RULE.txt`（事前登録）。

## 前提

- MI300X（gfx942・CDNA3・wave64）、Ubuntu 24.04 系、実施時点で最新の安定版 ROCm、rustup の stable。
- 実施前に単価を再確認する。約 1 万円を超えるなら着手せず所有者へ戻す（`docs/rocm-cloud-amd-procurement-decision.md` §2・§7）。実機 CI は置かない。

## 手順（クラウド固有の接続情報は書かない）

1. 転送元で `git rev-parse HEAD > .rev-stamp` を実行する（`RULE.txt` のコミットより後であること）。
2. `docs/real-hardware-verification-env.md` §3 と同じ形の rsync で転送する。`--filter=':- .gitignore'` を付け、`.env*`・`*.local.md`・`.claude/settings.local.json` を明示除外する。接続先は Git 管理外のローカルファイルとシェル変数（例 `"$ROCM_NODE"`。クォート必須）で渡す。
3. ノード上で `bash docs/perf/logs/rocm-parity-3159/run_rocm_parity.sh <tree-abs-path> <out-abs-path>` を実行する（ビルド後に run1・run2 を順に実行。ホスト名・ホームパス・ユーザー名はマスクし、残れば fail-closed で停止）。
4. 出力（`.rev-stamp`・`env_info.txt`・`run*/ignored.log`・`run*/exit_code`）をこのディレクトリへ戻し、下の記入欄を埋める。ログの `ROCM_PARITY` 行・`ROCM_DETERMINISM` 行を転記する。

## 形状（`RULE.txt` と同一）

(1,1,1) (1,1,0) (64,64,16) (63,65,17) (130,70,33) (5,129,200) (128,128,64) (257,255,513) (128,128,32) (65,64,16) (64,63,16) (64,64,15) (64,64,17) (1,257,33) (257,1,33) (193,129,48) (64,64,4096) (33,17,4097) (128,96,8192) (0,5,3) (5,0,3)

## 記入欄（段 B で記入）

| 項目 | 値 |
|---|---|
| 実施日 | |
| .rev-stamp | |
| ROCm 版 | |
| gfx ターゲット | |
| wave 幅 | |
| rustc | |
| probe（`is_available()`） | |
| 判定（pass／fail） | |

形状ごと（run1／run2）:

| 形状 m×n×k | total | fail | max_abs | max_rel | bit_mismatch |
|---|---|---|---|---|---|
| | | | | | |

決定性（`ROCM_DETERMINISM`。記録のみ）: （未記入）

unsafe 監査 §6（`docs/backend-rocm-unsafe-audit.md`）のうちテストの完走で間接的に確認できた項目: （未記入）。grid 上限の実照会・`hipFree` のデバイス非依存性・スレッド安全性など完走だけでは確かめられない項目は「未確認」と書いて残す。

## linalg（#3444）

**状態: 未実測**。`inv`・`solve`・`det`・`cholesky`・`qr`・`eigh`（`RocmLinalgLu`・`RocmLinalgFactor`。`double` カーネル・出力のみ `f32`）の実機 parity。`tests/linalg_real_device.rs`（`#[ignore]`）が `run_rocm_parity.sh` の `--ignored` 全件実行に自動で含まれる（スクリプトは変更しない）。**本テストの失敗は run の終了コードに影響する。**

- 判定は REQ-2 統一複合判定（相対 1e-3 未満 または 絶対 1e-5 未満）のみ。`RULE.txt`（事前登録・編集しない）の GEMM 用規則とは別に、linalg の追加規則は設けない。tolerance・baseline・FMA 契約は変えない。CPU との bit 不一致数は観測値で、assert しない。
- 入力集合（`docs/backend-gpu-linalg-decision.md` §5 準拠）: n = 1・2・3・17・64（solve・inv・cholesky は `1e20`・`1e-20` 倍を含む）、対角優位・κ ≤ 1e3 の SPD・対称行列、QR は 1×1・5×3・3×5・17×17・64×64・65×17・17×65、特異（solve・inv は `Singular`、det は `0.0`）、非正定値・eigh の NaN 入力（`LinalgInvalidArgument`）。
- ログ行は `ROCM_PARITY_LINALG\t<名前>\ttotal=…\tfail=…\tmax_abs=…\tmax_rel=…\tmean_abs=…\tp99_abs=…\tbit_mismatch=…`。
- 補助診断（記録のみ）: オフラインの `clang -x hip --offload-arch=gfx942 --offload-device-only -O3 -S` に `render_linalg_lu_source`／`render_linalg_factor_source` と同じソースを通し、`v_fma_f64`・`v_fmac_f64` の有無を数える。HIPRTC と同一ではない。あわせて `fabs`・`copysign`・`isfinite`・`atomicExch(int*, int)` が暗黙ヘッダで解決されること、`#pragma clang fp contract(off)` がファイルスコープで受理されることを、実 HIPRTC のコンパイル成否で確認する。

| 項目 | 値 |
|---|---|
| 実施日 | |
| 判定（pass／fail） | |
| HIPRTC コンパイル（LU 系・分解系） | |
| `v_fma_f64` の有無（補助診断） | |

演算ごと（`ROCM_PARITY_LINALG` 行の転記）:

| 名前 | total | fail | max_abs | max_rel | bit_mismatch |
|---|---|---|---|---|---|
| | | | | | |

## softmax 系（#3435）

**状態: 未実測**。`softmax`・`log_softmax`（`RocmSoftmax`）と `log_softmax_backward`（`RocmLogSoftmaxBackward`）の実機 parity。`tests/softmax_real_device.rs`（`#[ignore]`）が `run_rocm_parity.sh` の `--ignored` 全件実行に自動で含まれる（スクリプトは変更しない）。**本テストの失敗は run の終了コードに影響する。**

- 判定は REQ-2 統一複合判定（相対 1e-3 未満 または 絶対 1e-5 未満）のみ。`exp2f`／`expf`／`logf` の丸めと wave64 の結合順が CPU と異なるため bit 一致は assert せず、`bit_mismatch` は観測値。tolerance・baseline・FMA 契約は変えない。
- 入力集合: 形状 (rows, cols) = (1,1)・(3,4)・(7,63)・(5,64)・(5,65)・(4,128)・(3,1000)・(2,4096)・(70000,3)（grid-stride 発生形状）、振幅 1・30、`f32::MAX` 混在行・全要素 `-f32::MAX` 行（有限であること）、backward の相殺入力 `g=[1e20,-1e20,1,0]` と大きな上流勾配（`2e38`）。
- ログ行は `ROCM_PARITY_SOFTMAX\t<名前>\ttotal=…\tfail=…\tmax_abs=…\tmax_rel=…\tmean_abs=…\tp99_abs=…\tbit_mismatch=…`。
- 補助確認（記録のみ）: 実 HIPRTC で `__shfl_xor`（`WARP_SHFL_XOR`）・`float4`・`exp2f`・`logf`・`#pragma clang fp contract(off)` がコンパイルできること、wave64 で 1 block = 64 スレッドの起動が通ること、`__launch_bounds__(WARP_SIZE)` が受理されること。

| 項目 | 値 |
|---|---|
| 実施日 | |
| 判定（pass／fail） | |
| HIPRTC コンパイル（softmax 系・backward） | |

演算ごと（`ROCM_PARITY_SOFTMAX` 行の転記）:

| 名前 | total | fail | max_abs | max_rel | bit_mismatch |
|---|---|---|---|---|---|
| | | | | | |

## fail 時の切り分け（推測で結論を書かない）

1. fail の形状・要素数・最大誤差を上の表へ記録する。
2. bit 不一致と fail の関係を見る（K が大きいほど増えるか、端数形状に偏るか）。
3. 補助診断（記録のみ・判定に使わない）: `render_gemm_f32_source` と同じソースをオフラインの `clang -x hip --offload-arch=gfx942 --offload-device-only -O3 -S` に通し、`v_fma_f32`・`v_fmac_f32`・`v_pk_fma_f32` の有無を数える。HIPRTC と同一ではない。
4. 原因候補を書き、`.claude/rules/out-of-scope-tracking.md` に従い後続 Issue として提案する（起票はユーザー承認後）。
- tolerance の緩和・baseline の追加・`-ffp-contract` の追加は行わない。

## 秘密情報

インスタンス識別子・認証情報・内部ホスト名・個人のパス・GPU の UUID／BDF は収録しない。マスク検査を通ったログだけを置く。
