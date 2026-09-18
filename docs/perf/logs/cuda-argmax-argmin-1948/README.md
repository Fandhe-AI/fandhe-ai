# CUDA argmax／argmin 実機実測 申し送り（イシュー #1948）

> **2026-09-18 DGX Spark GB10 で実測済み**（5/5 pass。下記「実測結果（記入欄）」参照）。以下の申し送り文は PR 時点の記録としてそのまま残す。

本エージェント実行環境に DGX Spark GB10 等の CUDA 実機への到達手段がないため、
`crates/backend-cuda/tests/reduce_parity.rs` に追加した `#[ignore]` テスト群
（`argmax_and_argmin_*`）は未実行のまま本 PR をマージする。実機セッションで
以下を実行し、本ファイルへ結果を追記すること。

## 実行コマンド

```sh
# Linux 実行可能な単体テスト（GPU 不要。ホストモデル vs CPU 参照実装の突合）
cargo test -p fandhe-ai-backend-cuda --lib arg_reduce
cargo test -p fandhe-ai-backend-cuda --lib argmax

# 実機必須テスト（コンパイル確認のみは --no-run で可能）
cargo test -p fandhe-ai-backend-cuda --release --test reduce_parity -- --ignored --nocapture argm

# 一括実行（他の #[ignore] テストと同時実行する場合）
make test-ignored-cuda  # 対象リポジトリの Makefile に定義があれば
```

## 保存すべきファイル

- 上記コマンドの標準出力（各テスト名・pass/fail）
- `env_info.txt`（内部ホスト名は含めない。GPU 型番・CUDA/driver バージョン・
  `nvidia-smi` 出力の要点のみ）

## 事前登録判定規則

- `argmax_and_argmin_match_cpu_reference_on_real_device`／
  `argmax_and_argmin_nan_and_infinity_semantics_match_cpu_on_real_device`／
  `argmax_and_argmin_match_cpu_reference_for_transposed_view_on_real_device`:
  添字が CPU 参照実装（`fandhe_ai_backend_cpu::reduction::argmax`／`argmin`）
  と**全要素完全一致**すること（tolerance は使わない。整数添字の比較のため）。
- `argmax_and_argmin_empty_reduction_semantics_match_cpu_on_real_device`:
  `BackendError::KernelLaunchFailed` の文言が CPU 側と同一
  （`"empty reduction for op \"argmax\""` / `"...\"argmin\""`）であること。
- `argmax_and_argmin_are_run_to_run_deterministic_on_real_device`: 同一入力を
  3 回実行し添字ベクトルが bit 単位で完全一致すること。
- FAIL が生じた場合は**是正せず記録のみ**とする（本 PR の受け入れ条件は
  Linux 実行可能テストの green のみであり、実機実測は申し送り事項）。

## 実測結果（記入欄）

**2026-09-18（UTC 01:47Z）DGX Spark GB10 で実測済み**（転送元 origin/main `536c56a8`。
環境は `env_info.txt`〈`hostname: masked`〉。ログは `reduce_parity-argm.log`〈ユーザー名・
絶対パスは `<home>` へ置換済み〉）。

```sh
cargo test -p fandhe-ai-backend-cuda --release --test reduce_parity -- --ignored --nocapture argm
```

| テスト | 結果 |
|---|---|
| `argmax_and_argmin_match_cpu_reference_on_real_device` | ok |
| `argmax_and_argmin_nan_and_infinity_semantics_match_cpu_on_real_device` | ok |
| `argmax_and_argmin_match_cpu_reference_for_transposed_view_on_real_device` | ok |
| `argmax_and_argmin_empty_reduction_semantics_match_cpu_on_real_device` | ok |
| `argmax_and_argmin_are_run_to_run_deterministic_on_real_device` | ok |

`test result: ok. 5 passed; 0 failed`（0.41 s）。事前登録判定規則はすべて充足
（添字完全一致・空縮約エラー文言一致・3 回実行の bit 同一）。同日の全 `#[ignore]`
群実行（`docs/perf/logs/cuda-realdevice-phase4-2026-09-18/README.md`）でも本 5 件は
並列実行のまま ok（by-name 新規 ok として記録）。是正・閾値変更なし。
