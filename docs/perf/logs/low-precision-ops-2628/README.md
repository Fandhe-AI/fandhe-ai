# MatMul・elementwise 5 演算の低精度 forward（#2628）CUDA／Metal 実機 parity の申し送りと実測記録（#2629）

**#2629 で実測済み（2026-10-11・コミット `56b9d6ec`）。結果は末尾「実測記入欄」を参照。以下の申し送り記述は記入時点（#2628）のまま残す。**

`docs/autodiff-low-precision-op-extension-decision.md` 6 節参照。**記入時点の本書は実測値を含まない**。本実装エージェント実行環境は CUDA／Metal 実機に到達できないため、`crates/facade/tests/low_precision_ops_backend_parity.rs` のうち実機を対象とする 2 テストは `#[ignore]` のまま未実測である。実測は #2629（実機ツリー #2683）が担う。

- `cuda_low_precision_ops_match_cpu_reference`（`Device::Cuda(0)`）
- `metal_low_precision_ops_match_cpu_reference`（`Device::Metal`。`cfg(target_os = "macos")` 限定）

各テストは 6 Op（matmul／add／mul／relu／exp／tanh）× {F16, Bf16} の forward を、CPU tape と同一 dtype・有限出力の入力で `fandhe_ai_backend_cpu::parity::assert_parity` により突合する。

## 測定コマンド案

```sh
# CUDA（DGX Spark GB10 等の実機上で）
cargo test -p fandhe-ai --release --test low_precision_ops_backend_parity -- --ignored --nocapture --test-threads=1 cuda_

# Metal（Apple Silicon 実機上で）
cargo test -p fandhe-ai --release --test low_precision_ops_backend_parity -- --ignored --nocapture --test-threads=1 metal_
```

CPU 版（P1・非有限出力）は同テストファイルの属性なしテストで既に検証済み（green）。

## 事前登録判定規則

- 判定は REQ-2 統一複合判定（相対誤差 1e-3 未満 または 絶対誤差 1e-5 未満）。tolerance・baseline は変更しない。
- **P6（CUDA／Metal vs CPU。自前バックエンド間）の不一致は「判定不能」にせず、通常の parity 失敗として扱う**。カーネルまたは丸め方針の不具合として修正する（PyTorch 比較〈P4・P5〉の「第三者比較対象の判定不能」とは別）。
- CUDA／Metal は `TypedOps<f16>`／`TypedOps<bf16>` の accessor（`typed_ops_f16`／`typed_ops_bf16`）経由で到達する。accessor が `None` のバックエンドは `Unsupported` で失敗し、それ自体が実測結果として記録される（ホスト計算・f32 へのフォールバックは持たない）。
- **Metal の bf16 の実機可用性は記入時点では未検証**（#2629 で「可」と確定。末尾参照）。
- #2629 の計画段階（実測前）で固定した方針（結果判明後に README へ転記。判定規則は実測前から不変）: fail が出た場合は生ログと組合せ別数値をそのまま記録し verdict=`fail` とする。本 PR ではカーネル修正・tolerance 変更・baseline 新設を行わない。`Unsupported` は fail として記録する。到達不能は `undetermined`。

## 実測記入欄

実測（#2629）。対象コミット `56b9d6ec`（診断テスト追加後の HEAD。CUDA は `.rev-stamp` で一致を確認）。ログ中の絶対パスは `<worktree>`／`~` へ、ホスト名は `masked` へ置換済み。

| 日付（UTC） | 実機 | コマンド | 結果 | 備考 |
|---|---|---|---|---|
| 2026-10-11 | DGX Spark GB10（driver 580.173.02・CUDA 13.0） | 上記 CUDA（`cuda_low_precision_ops_match_cpu_reference`）。exit 0 | `1 passed; 0 failed`（verdict=pass） | `cuda/ignored.log`・`cuda/env_info.txt`。診断 `cuda_low_precision_ops_parity_report` も exit 0・`cuda/report.log`。計測前後で GPU 使用率 0%（常駐 2 プロセスのみ・計測中の新規出現なし） |
| 2026-10-11 | Apple Silicon（M4 Max・macOS 27.0.1） | 上記 Metal（`metal_low_precision_ops_match_cpu_reference`）。exit 0 | `1 passed; 0 failed`（verdict=pass） | `metal/ignored.log`・`metal/env_info.txt`。診断 `metal_low_precision_ops_parity_report` も exit 0・`metal/report.log` |

### 組合せ別結果（`report.log`。CUDA・Metal とも全 12 組合せ同一）

| Op | F16 | Bf16 |
|---|---|---|
| matmul | bit 一致 4/4・fail 0 | bit 一致 4/4・fail 0 |
| add／mul／relu／exp／tanh | 各 bit 一致 6/6・fail 0 | 各 bit 一致 6/6・fail 0 |

全 12 組合せ × 2 実機で `max_abs_diff = 0`・`max_rel_err = 0`（CPU 参照と bit 一致）。tolerance・baseline は不変。

### Metal の bf16 可用性

**可**。`typed_ops_bf16` accessor は `Some` を返し、bf16 全 6 Op が `Unsupported` にならず CPU と bit 一致した（M4 Max）。

### 検証範囲の注意

入力は `inputs()` 固定の小形状（最大 6 要素）のみ。大形状での Tensor Core 経路の結合順序差は本実測の範囲外。

## #2994 大形状（Tensor Core 経路）

状態: **実測済み（#3170・2026-10-19 UTC・コミット `b640875f`）**。f16 は 8 形状中 5 形状（64³・128³・256³・128×128×100・65×63×33）で fail=0、3 形状（512³・1024³・256×256×1024）で fail≠0。bf16（f32 strict 経路）は 8 形状すべて fail=0 かつ CPU と bit 一致。判定は下記の事前登録規則どおり（結果判明後に規則は変更していない）。なお #3009 時点では GB10 へ到達できず undetermined だった。

### 対象と経路

診断テスト `cuda_low_precision_matmul_large_shape_report`（`crates/facade/tests/low_precision_ops_backend_parity.rs`）。CPU と CUDA の `matmul_low_precision` を比較する。入力は `Xorshift64Star`（lhs=`seed`・rhs=`seed+1`、[-1,1)）。既存 `ParityPath::MmaF16` baseline とは入力生成経路が異なり、数値は直接比較できない。

| 形状 m×n×k | seed | f16 の予測経路 |
|---|---|---|
| 64×64×64 / 128×128×128 / 256×256×256 / 512×512×512 / 1024×1024×1024 | 11〜15 | Mma（`mma.sync`） |
| 256×256×1024 | 16 | Mma |
| 128×128×100（k%8≠0） | 17 | Wmma フォールバック |
| 65×63×33（n%8≠0） | 18 | Wmma フォールバック |

- **bf16 は Tensor Core 経路を通らない**。`typed_bf16.rs` は f32 へ昇格して `gemm_fp32_strict`（単一の連続 K ループ）へ委譲する。bf16 は「f32 strict 経路」として同形状を測る。
- ログの `route=` は**予測**（`mma_available` と整列条件からの再現）であり観測ではない。観測は `backend-cuda` の `F16MatrixUnitImpl` 診断（feature `internal-diagnostics`）が必要で、本 issue の対象外とした。

### 測定コマンド

```bash
cargo test -p fandhe-ai --release --test low_precision_ops_backend_parity -- --ignored --exact --nocapture --test-threads=1 cuda_low_precision_matmul_large_shape_report
```

出力: `ROUTE` 行（cc・mma_available）、形状 × dtype ごとの `REPORT` 行（total・bit_eq・fail・mean_abs・max_abs・max_rel・ulp0/ulp1/ulp_gt1）、`SUMMARY` 行。ログは `cuda/large_shape_report.log`・`cuda/env_info_large.txt` に、ホスト名を `masked`・絶対パスを `~` に置換して保存する。

### 事前登録判定規則（実測前に固定）

1. 合否基準は REQ-2 統一複合判定のまま。tolerance・baseline・`compare`／`passes` は変更しない。テストが assert するのは shape 一致・`compare` が `Ok`・全要素有限・`total==m*n` のみで、fail 数は記録専用（緩和ではない）。
2. f16（Mma／Wmma）で fail=0 → 「厳密ゼロ fail 成立形状」として記録。fail≠0 → 形状二分方式（`docs/cuda-tensor-core-parity-judgment-decision.md`）の baseline 非後退方式の候補として**提案のみ**書き、`ParityBaseline` 行の候補値（fail_count・mean/max_abs・max_rel の実測値）を添える。baseline 追加は人間承認事項で別 issue・別 PR とする。
3. bf16（f32 strict）で fail が出た場合: 出力 dtype 量子化（bf16 の 1 ulp は相対 約 3.9e-3〜7.8e-3 で `RELATIVE_TOLERANCE` 1e-3 より大きい）による丸め反転の可能性がある。`ulp_gt1=0` なら全差分が 1 ulp 以内であり量子化起因の可能性を示し、`ulp_gt1>0` ならカーネル側の差を示す。判定方式は決定記録・spec の論点として提案のみ。tolerance 変更は提案しない。
4. `Unsupported`・`Err` は fail として記録。実機に届かなければ `undetermined`。

### 実測記入欄（#2994）

| 日付 | 実機 | コミット（`.rev-stamp`） | exit | 備考 |
|---|---|---|---|---|
| 2026-10-19 UTC | DGX Spark GB10（driver 580.173.02・CUDA 13.0・cc 12.1） | `b640875f86399515b2dab4dfc38089caaa5e5c08`（`cuda/rev-stamp-large.txt`） | 0（`test result: ok. 1 passed`） | 計測は単独ビルド後に実施（ビルド並走なし）。計測前後の compute-apps・utilization 0% に変化なし（常駐 3 プロセスのみ・`cuda/env_info_large.txt`）。`ROUTE cuda cc=(12, 1) mma_available=true reason=None`。ログ `cuda/large_shape_report.log`・ビルドログ `cuda/build_large.log` |

### 結果表（`cuda/large_shape_report.log` の `REPORT` 行を転記。16 行 = 8 形状 × {F16, Bf16}）

`route=` は予測であり観測ではない。

| dtype | 形状 m×n×k | seed | route（予測） | total | bit_eq | fail | mean_abs | max_abs | max_rel | ulp0 | ulp1 | ulp_gt1 | strict_pass |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| F16 | 64x64x64 | 11 | f16:Mma | 4096 | 4096 | 0 | 0e0 | 0e0 | 0e0 | 4096 | 0 | 0 | true |
| F16 | 128x128x128 | 12 | f16:Mma | 16384 | 16364 | 0 | 3.711174940690398e-7 | 1.953125e-3 | 9.433962264150943e-4 | 16364 | 20 | 0 | true |
| F16 | 256x256x256 | 13 | f16:Mma | 65536 | 65401 | 0 | 1.9330200302647427e-6 | 7.8125e-3 | 1.2142857142857143e-2 | 65401 | 127 | 8 | true |
| F16 | 512x512x512 | 14 | f16:Mma | 262144 | 261313 | 5 | 4.145593038629158e-6 | 1.5625e-2 | 3.8848920863309355e-1 | 261313 | 764 | 67 | false |
| F16 | 1024x1024x1024 | 15 | f16:Mma | 1048576 | 1043013 | 192 | 1.0508246703011537e-5 | 3.125e-2 | 4.803921568627451e-1 | 1043013 | 5114 | 449 | false |
| F16 | 256x256x1024 | 16 | f16:Mma | 65536 | 65205 | 16 | 8.295384759549052e-6 | 1.5625e-2 | 2.5336500395882817e-2 | 65205 | 293 | 38 | false |
| F16 | 128x128x100 | 17 | f16:Wmma-or-Tiled | 16384 | 16366 | 0 | 6.130503606982529e-7 | 3.90625e-3 | 8.19000819000819e-4 | 16366 | 18 | 0 | true |
| F16 | 65x63x33 | 18 | f16:Wmma-or-Tiled | 4095 | 4094 | 0 | 2.91109376192384e-11 | 1.1920928955078125e-7 | 8.143322475570033e-4 | 4094 | 1 | 0 | true |
| Bf16 | 64x64x64 | 11 | fp32_strict(non-TC) | 4096 | 4096 | 0 | 0e0 | 0e0 | 0e0 | 4096 | 0 | 0 | true |
| Bf16 | 128x128x128 | 12 | fp32_strict(non-TC) | 16384 | 16384 | 0 | 0e0 | 0e0 | 0e0 | 16384 | 0 | 0 | true |
| Bf16 | 256x256x256 | 13 | fp32_strict(non-TC) | 65536 | 65536 | 0 | 0e0 | 0e0 | 0e0 | 65536 | 0 | 0 | true |
| Bf16 | 512x512x512 | 14 | fp32_strict(non-TC) | 262144 | 262144 | 0 | 0e0 | 0e0 | 0e0 | 262144 | 0 | 0 | true |
| Bf16 | 1024x1024x1024 | 15 | fp32_strict(non-TC) | 1048576 | 1048576 | 0 | 0e0 | 0e0 | 0e0 | 1048576 | 0 | 0 | true |
| Bf16 | 256x256x1024 | 16 | fp32_strict(non-TC) | 65536 | 65536 | 0 | 0e0 | 0e0 | 0e0 | 65536 | 0 | 0 | true |
| Bf16 | 128x128x100 | 17 | fp32_strict(non-TC) | 16384 | 16384 | 0 | 0e0 | 0e0 | 0e0 | 16384 | 0 | 0 | true |
| Bf16 | 65x63x33 | 18 | fp32_strict(non-TC) | 4095 | 4095 | 0 | 0e0 | 0e0 | 0e0 | 4095 | 0 | 0 | true |

`SUMMARY`: strict_zero_fail は F16 の 5 形状と Bf16 の 8 形状、nonzero_fail は `F16@512x512x512`・`F16@1024x1024x1024`・`F16@256x256x1024`。

### 判定（事前登録規則 1〜4 を逐条適用）

- 規則 1: tolerance・baseline・`compare`／`passes` は未変更。テストは assert を満たし exit=0。
- 規則 2（f16）: **厳密ゼロ fail 成立形状** は 64³・128³・256³・128×128×100（Wmma-or-Tiled）・65×63×33（Wmma-or-Tiled）。fail≠0 の 3 形状は形状二分方式（`docs/cuda-tensor-core-parity-judgment-decision.md`）の baseline 非後退方式の候補として**提案のみ**とする。`ParityBaseline` 行の候補値（実測値。既存 `ParityPath::MmaF16` とは入力生成経路が異なり直接比較できない）:

| 形状 | fail_count | total | mean_abs | max_abs | max_rel |
|---|---|---|---|---|---|
| 512×512×512 | 5 | 262144 | 4.145593038629158e-6 | 1.5625e-2 | 3.8848920863309355e-1 |
| 1024×1024×1024 | 192 | 1048576 | 1.0508246703011537e-5 | 3.125e-2 | 4.803921568627451e-1 |
| 256×256×1024 | 16 | 65536 | 8.295384759549052e-6 | 1.5625e-2 | 2.5336500395882817e-2 |

  fail は K が大きい Mma 形状に限られ、ulp 分布は 512³ が ulp0=261313・ulp1=764・ulp_gt1=67、1024³ が ulp0=1043013・ulp1=5114・ulp_gt1=449、256×256×1024 が ulp0=65205・ulp1=293・ulp_gt1=38。baseline 追加は人間承認事項で別 issue・別 PR とする。
- 規則 3（bf16）: fail=0（全 8 形状で bit 一致）のため適用対象なし。f32 strict 経路は CPU と bit 一致した。

### 所有者判断待ち（提案のみ）

- f16: fail≠0 の 3 形状（fail 件数 5／192／16・上記 ulp 分布）の扱い。baseline 非後退方式（形状二分方式）への追加を提案する。tolerance 変更は提案しない。実施する場合は別 issue・別 PR で人間承認を経る。
- bf16: fail なし。判定方式の変更は不要。

### スコープ外・申し送り

- baseline 行の追加・更新（実測後、人間承認を経て別 issue）
- f16 の fail≠0 形状の baseline 化（上記「所有者判断待ち」。別 issue 要）
- Metal の大形状 parity、`F16MatrixUnitImpl` の facade 経由の実観測（公開面の拡張を伴う）

## #3617 HEAD 再計測（baseline 候補値の確認）

状態: HEAD 再計測は 2026-10-27 に GB10 で実施済み（下記「実測記入欄」）。baseline 行は
`ParityPath::FacadeLowPrecisionMatmulF16` として追加したが、**値の確定は所有者の確認待ち**
（形・値の出典・ceiling の型は #3617 コメント 6432022446 で承認済み。決定記録 §7）。

事前登録規則（計測前に固定）:

1. 測るのは既存診断 `cuda_low_precision_matmul_large_shape_report` のみ（コード・tolerance 不変）
2. 同一ビルドで 2 回実行し、3 形状の total・fail・mean_abs・max_abs・max_rel が完全一致なら「安定」、不一致なら「不安定」とし baseline 化は所有者判断
3. 安定かつ上記 `b640875f` の値と完全一致なら「候補値を HEAD で再確認」。不一致は両値を記録し所有者が選ぶ。規則は緩めない
4. 厳密ゼロ fail 5 形状で fail≠0 が出たら後退として記録し報告する（直さない）
5. `Unsupported`・`Err`・非有限値は fail として記録。実機未到達は undetermined

事前登録規則（ゲート実行用。ゲートを実行する前に固定）:

6. ゲート実行で測るのは `cuda_low_precision_matmul_f16_nonzero_fail_shapes_do_not_regress` 1 本のみ。tolerance・baseline は変えない
7. exit=0（`1 passed`）なら pass。fail・panic・mma 利用不可は fail として記録し、baseline も規則も緩めない
8. 同一ビルドで診断を 1 回流し、規則 2・3 の測定値と完全一致することを確かめる。不一致なら記録して所有者判断

実測記入欄（規則 1〜5）:

| 項目 | 値 |
|------|-----|
| 日付 | 2026-10-27 |
| 機体 | GB10（DGX Spark・sm_121・CUDA 13.0）。計測前後 GPU util 0%（他用途の常駐プロセスあり） |
| `.rev-stamp` | `a053ea7ed0f49d3b5ff01051d5ee8ff36d8661e9`（`cuda/rev-stamp-3617-measure.txt`） |
| exit | run1=0・run2=0（`cuda/large_shape_report_3617_run1.log`・`_run2.log`） |
| 規則 2（安定性） | REPORT・ROUTE 行が 2 回で完全一致 = 安定 |
| 規則 3（`b640875f` との一致） | 3 形状の total・fail・mean_abs・max_abs・max_rel が完全一致 = 候補値を HEAD で再確認 |
| 規則 4（厳密ゼロ fail 5 形状） | 全て fail=0（後退なし） |
| 規則 5 | `Unsupported`・`Err`・非有限値なし。`mma_available=true` |

baseline の ceiling 換算（表示桁最終桁 +1）:

| 形状 | 項目 | 実測 | ceiling |
|------|------|------|---------|
| 512³ | mean_abs | 4.145593038629158e-6 | 4.145593038629159e-6 |
| 512³ | max_abs | 1.5625e-2 | 1.5626e-2 |
| 512³ | max_rel | 3.8848920863309355e-1 | 3.8848920863309356e-1 |
| 1024³ | mean_abs | 1.0508246703011537e-5 | 1.0508246703011538e-5 |
| 1024³ | max_abs | 3.125e-2 | 3.126e-2 |
| 1024³ | max_rel | 4.803921568627451e-1 | 4.803921568627452e-1 |
| 256×256×1024 | mean_abs | 8.295384759549052e-6 | 8.295384759549053e-6 |
| 256×256×1024 | max_abs | 1.5625e-2 | 1.5626e-2 |
| 256×256×1024 | max_rel | 2.5336500395882817e-2 | 2.5336500395882818e-2 |

ゲート実行（規則 6〜8）の結果: 2026-10-27・GB10・`.rev-stamp` `045a18a231ac8dfa255b275e9f30a33b0879e897`
（`cuda/rev-stamp-3617-gate.txt`）。ゲート `cuda_low_precision_matmul_f16_nonzero_fail_shapes_do_not_regress`
は 1 passed・exit 0（`cuda/gate_3617.log`）= pass（規則 7）。規則 8（同一ビルドの診断値の一致）は、
ゲート時のビルド（`045a18a2`）の診断ログを保存していないため記録上は未確認（保存済みの診断ログは
`a053ea7ed` の run1・run2 のみ。`045a18a2` との `crates/` の差分は `tests/` 配下のテストファイルのみで、計測対象のライブラリコードは同一）。baseline の値は所有者の確認待ち。
