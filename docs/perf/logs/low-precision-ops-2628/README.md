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
