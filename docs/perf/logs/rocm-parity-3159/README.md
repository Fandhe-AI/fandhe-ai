# ROCm（MI300X・gfx942）対 CPU の f32 GEMM parity（イシュー #3159）

**状態: 未実測**（所有者のインスタンス起動待ち）。本ディレクトリの現状は段 A（準備のみ）で、下の記入欄はすべて空欄。得ていない結果は書かない。

## 対象範囲

- `--ignored` 全件実行が被覆する範囲（`tests/*_real_device.rs` の `#[ignore]` 16 本）: `RocmGemmF32::run`（f32 GEMM・`GfxTarget::Gfx942`）と `is_available()`（probe）、および `RocmBackendOps` 経由の linalg〈R13〉・softmax 系〈R4〉・reduce〈R3〉・loss〈R5〉・norm〈R6〉・paged KV〈R12〉。各 Op の記入欄は下の節に分けてある。
- `docs/backend-rocm-ops-coverage-plan.md` §3 で実機が不要とされた R0〜R2・R7〜R11（elementwise・optim・rnn_cell・gather/scatter・gemm_derived 等）は実機テストを置かないため対象外。
- 判定は REQ-2 統一複合判定（相対 1e-3 未満 または 絶対 1e-5 未満）のみ。tolerance・baseline・FMA 契約は変えない。規則は `RULE.txt`（事前登録）。

## 前提

- MI300X（gfx942・CDNA3・wave64）、Ubuntu 24.04 系、実施時点で最新の安定版 ROCm、rustup の stable。
- 実施前に単価を再確認する。約 1 万円を超えるなら着手せず所有者へ戻す（`docs/rocm-cloud-amd-procurement-decision.md` §2・§7）。実機 CI は置かない。

## 手順（クラウド固有の接続情報は書かない）

詳細手順（起動前の準備・費用順の実行と停止・回収・マスク漏れ検査・記入欄との対応表）は `docs/real-hardware-verification-env.md` §9。

1. 転送元で `git rev-parse HEAD > .rev-stamp` を実行する（`RULE.txt` のコミットより後であること）。
2. `docs/real-hardware-verification-env.md` §3 と同じ形の rsync で転送する。`--filter=':- .gitignore'` を付け、`.env*`・`*.local.md`・`.claude/settings.local.json` を明示除外する。接続先は Git 管理外のローカルファイルとシェル変数（例 `"$ROCM_NODE"`。クォート必須）で渡す。
3. ノード上で `bash docs/perf/logs/rocm-parity-3159/run_rocm_parity.sh <tree-abs-path> <out-abs-path>` を実行する（ビルド後に run1・run2 を順に実行。ホスト名・ホームパス・ユーザー名はマスクし、残れば fail-closed で停止）。
4. 出力（`.rev-stamp`・`env_info.txt`・`run*/ignored.log`・`run*/exit_code`）をこのディレクトリへ戻し、下の記入欄を埋める。ログの `ROCM_PARITY` 行・`ROCM_DETERMINISM` 行を転記する。
5. 実行本数の照合（スクリプトは終了コードしか見ないため手動で行う）: 実行前に `cargo test -p fandhe-ai-backend-rocm --release -- --ignored --list` で ignored テスト数（現状 16 本）を取り、記入欄へ転記する。実行後、各 `run*/ignored.log` の `test result:` の passed 合計がこの本数と一致し、`0 passed` のまま終わった実機テストバイナリが無いことを確かめる。一致しない結果・0 本実行の結果は pass の根拠にしない。

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
| ignored 本数（`--list`） | |
| passed 合計（run1／run2） | |
| exit_code（run1／run2） | |
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

## reduce（#3434）

**状態: 未実測（#3159）**。`sum`・`max`・`min`（`RocmReduce`。wave 幅は `WarpGeometry` で注入）と `argmax`・`argmin`（`RocmArgReduce`）の実機 parity。`tests/reduce_real_device.rs`（`#[ignore]`）が `run_rocm_parity.sh` の `--ignored` 全件実行に自動で含まれる（スクリプトは変更しない）。**本テストの失敗は run の終了コードに影響する。**

- 判定は REQ-2 統一複合判定（相対 1e-3 未満 または 絶対 1e-5 未満）。tolerance・baseline・FMA 契約は変えない。`max`・`min`・`argmax`・`argmin` と軸指定 `sum` は CPU と**完全一致**を assert する。全軸 `sum` は block/lane の結合順序が CPU（4096 要素チャンクの `f64` partial）と異なるため複合判定のみ assert し、bit 不一致数は観測値（`bit_mismatch`）として出力する。
- 入力集合: 全軸 numel = 1・255・256・257・5000・`256*1024+1`（ブロック上限 1024 を超え grid-stride が複数周する）、軸指定の形状 `[3,4]`・`[2,3,5]`・`[1,300]`・`[300,1]`・`[257,3]`・`[64,784]` の全軸、NaN・±inf 混在（`max`・`min`）、タイ・NaN・全 NaN・チャンク境界（argmax・argmin。numel 4095・4096・4097・12295・`256*1024+1`）、`[1e8, 1, -1e8]` の総和が 1（`f64` アキュムレータ）。
- ログ行は `ROCM_PARITY_REDUCE\t<名前>\ttotal=…\tfail=…\tmax_abs=…\tmax_rel=…\tmean_abs=…\tp99_abs=…\tbit_mismatch=…`（argmax・argmin は `exact=true`）。
- 補助診断（記録のみ）: HIPRTC で wave64（gfx942）の `__shfl_xor`（`double` オーバーロードを含む。マクロ `WARP_SHFL_XOR`）・`__uint_as_float`・`__float_as_uint`・`fmaxf`／`fminf`・`#pragma clang fp contract(off)` がコンパイルできること。`WARPS_PER_BLOCK = 4`（256 / 64）で `warp_sums` が未初期化読みにならないこと（ホスト側モデルは `src/reduce_model.rs` が wave32・wave64 の両方で確認済みで、実機の結果ではない）。

| 項目 | 値 |
|---|---|
| 実施日 | |
| 判定（pass／fail） | |
| HIPRTC コンパイル（reduce・arg_reduce） | |
| 全軸 `sum` の `bit_mismatch` 合計（観測値） | |

演算ごと（`ROCM_PARITY_REDUCE` 行の転記）:

| 名前 | total | fail | max_abs | max_rel | bit_mismatch |
|---|---|---|---|---|---|
| | | | | | |

## paged KV（#3443）

**状態: 未実測**。`paged_kv_append`・`paged_attention_forward`（`RocmPagedKv`）の実機 parity。`tests/paged_kv_real_device.rs`（`#[ignore]`）が `run_rocm_parity.sh` の `--ignored` 全件実行に自動で含まれる（スクリプトは変更しない）。**本テストの失敗は run の終了コードに影響する。**

- 判定は REQ-2 統一複合判定（相対 1e-3 未満 または 絶対 1e-5 未満）。append は純コピーなので K／V pool 全体の bit 一致を assert する。attention は GPU の `expf` の丸めが CPU と異なりうるため複合判定のみで、`bit_mismatch` は観測値。tolerance・baseline・FMA 契約は変えない。
- 入力集合: block_size 1/4/16、batch 1/2、`(E, H)` = (8,4)・(16,4)・(64,4)・(128,2)（head_dim 2〜64）、prefill → decode の複数ステップ（L_new = 3/5/15/200 から 1 への遷移を含む）、非連続ブロック ID と番兵の未使用枠。
- ログ行は `ROCM_PARITY_PAGED_KV\t<名前>\ttotal=…\tfail=…\tmax_abs=…\tmax_rel=…\tmean_abs=…\tp99_abs=…\tbit_mismatch=…`。
- 補助確認（記録のみ）: 実 HIPRTC で `__builtin_fmaf`・`expf`・`__uint_as_float`・`#pragma clang fp contract(off)`・`__launch_bounds__(128)` がコンパイルできること、wave64 で 1 block = 128 スレッド（2 wave）の起動と `__syncthreads()` 越しの共有フラグ・作業バッファの可視性が通ること。

| 項目 | 値 |
|---|---|
| 実施日 | |
| 判定（pass／fail） | |
| HIPRTC コンパイル（paged KV） | |

ケースごと（`ROCM_PARITY_PAGED_KV` 行の転記）:

| 名前 | total | fail | max_abs | max_rel | bit_mismatch |
|---|---|---|---|---|---|
| | | | | | |

## loss（#3436）

**状態: 未実測**。`mse`・`huber`（Huber／SmoothL1）・`bce`（Probabilities／Logits）・`nll`・`kl_div` の forward／backward の実機 parity。`tests/loss_real_device.rs`（`#[ignore]`）が `run_rocm_parity.sh` の `--ignored` 全件実行に自動で含まれる（スクリプトは変更しない）。**本テストの失敗は run の終了コードに影響する。** 外れたケースがあっても全ケースを走らせて fail 要素数と最大誤差を記録し、最後に落とす。

- 判定は REQ-2 統一複合判定のみ（`RULE.txt` の GEMM 用規則とは別の追加規則は設けない）。同一入力 2 回実行の bit 一致（決定性）も assert する。tolerance・baseline・FMA 契約は変えない。
- 入力集合: テスト内の `SIZES`（1・7・255・256・257・4097・256×1024+3）× reduction（Mean／Sum）、nll は shape 5 種（最後は grid-stride が 2 周以上する形状）。
- ログ行は `ROCM_PARITY_LOSS\t<名前>\ttotal=…\tfail=…\tmax_abs=…\tmax_rel=…\tmean_abs=…\tp99_abs=…\tbit_mismatch=…`。
- 補助確認（記録のみ）: 実 HIPRTC で `logf`・`log1pf`・`expf` が解決できること、`__builtin_fmaf`・`#pragma clang fp contract(off)`・`__launch_bounds__` が受理されること、`WARPS_PER_BLOCK = 4` の wave64 での縮約順序。

| 項目 | 値 |
|---|---|
| 実施日 | |
| 判定（pass／fail） | |
| HIPRTC コンパイル（loss） | |

ケースごと（`ROCM_PARITY_LOSS` 行の転記）:

| 名前 | total | fail | max_abs | max_rel | bit_mismatch |
|---|---|---|---|---|---|
| | | | | | |

## 正規化（#3437）

**状態: 未実測**。`rmsnorm`・`layer_norm`・`batch_norm_train`・`batch_norm_infer` の forward（CPU 参照比較）と `rmsnorm_backward`・`layer_norm_backward`（テスト内 f64 参照比較）の実機 parity。`tests/norm_real_device.rs`（`#[ignore]`）が `run_rocm_parity.sh` の `--ignored` 全件実行に自動で含まれる（スクリプトは変更しない）。**本テストの失敗は run の終了コードに影響する。** 外れたケースがあっても全ケースを走らせて記録し、最後に落とす。

- 判定は REQ-2 統一複合判定のみ（GEMM 用規則とは別の追加規則は設けない）。`bit_mismatch` は観測値。tolerance・baseline・FMA 契約は変えない。
- 入力集合: テスト内の形状（forward は (1,1)・(3,31)・(5,64)・(4,65)・(2,3,257)、backward は rows×hidden 5 種、batch_norm は (4,3)・(2,3,17)・(2,5,4,33)）× weight／bias の有無。
- ログ行は `ROCM_PARITY_NORM\t<名前>#<連番>\ttotal=…\tfail=…\tmax_abs=…\tmax_rel=…\tmean_abs=…\tp99_abs=…\tbit_mismatch=…`。
- 補助確認（記録のみ）: 実 HIPRTC で `__shfl_xor` の double 版・`__builtin_fma` の f64 命令化がコンパイルできること、wave64 の縮約順序。

| 項目 | 値 |
|---|---|
| 実施日 | |
| 判定（pass／fail） | |
| HIPRTC コンパイル（norm） | |

ケースごと（`ROCM_PARITY_NORM` 行の転記）:

| 名前 | total | fail | max_abs | max_rel | bit_mismatch |
|---|---|---|---|---|---|
| | | | | | |

## 秘密情報

インスタンス識別子・認証情報・内部ホスト名・個人のパス・GPU の UUID／BDF は収録しない。マスク検査を通ったログだけを置く。
