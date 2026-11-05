# GPU 線形代数の opt-in 関数の facade 経由の実機 parity（Metal。#3866）

対象は #3405（PR #4014）で facade に公開した `fandhe_ai::set_metal_gpu_linalg_enabled`／`metal_gpu_linalg_enabled`（macOS 限定）。
`crates/facade/tests/linalg_gpu_optin_metal_parity.rs` の `#[ignore]` テスト 5 本（inv・solve・det・cholesky・qr。eigh は Metal の範囲外）を、
`<metal-node>`（Mac mini・Apple M4。`docs/real-hardware-verification-env.md` §7.1）で 1 本ずつ実行し、ゲート ON の Metal tape と CPU tape を突き合わせた。
GB10 側は同 PR の [`../gpu-linalg-optin-cuda-parity-3866/`](../gpu-linalg-optin-cuda-parity-3866/README.md)。

- 事前登録: [`RULE.txt`](./RULE.txt)（宣言 2026-11-04T15:01Z。実機実行より前にコミット 49a470f4b で push した。実行後に変えていない）
- 実行スクリプト: [`run_metal_parity.sh`](./run_metal_parity.sh)（#3865 版 `run_metal_parity.sh`〈PR #4030〉の複製。
  変えたのは対象一覧と、非 0 終了の判定分岐〈tape_for の失敗は undetermined・それ以外は fail〉だけで、差分はスクリプト冒頭に列挙した。
  Mac mini 上で手動実行。CI からは呼ばない。`--self-test` 付き）
- tolerance 定数（`RELATIVE_TOLERANCE`／`ABSOLUTE_RESCUE_THRESHOLD`）・parity baseline・判定規則・入力集合・テストコードは不変

## 位置づけ

このテストファイルは #3405 で新設され、今回が初めての実機実行。カーネル parity（G2）は backend 層で充足済み
（[`../metal-linalg-lu-chol-qr-2676/README.md`](../metal-linalg-lu-chol-qr-2676/README.md) の 2026-10-28 再実測。Mac mini・5/5 run 全件 pass）。
入力は backend-metal の正本 `crates/backend-metal/tests/linalg_parity_inputs.rs` を `#[path]` で共有する。本測定は facade の 1 行委譲を
通しても同じ結果になることを確かめる。

## 状態

**実測済み: 5 本すべて pass、fail 0、undetermined 0**（2026-11-04・`<metal-node>`）。

- 実行コミット（`rev-stamp.txt`）: `49a470f4b4196f736bbe9631ce4d556c825f568e`・`dirty=0`。事前登録と実行スクリプトを含むコミットそのもので、
  `crates/`・`Cargo.toml`・`Cargo.lock`・`rust-toolchain.toml` は対象コミット `2b00e494b` と同一（送信前に `git diff --quiet` で確認）。
  転送は rsync だけで、ノード側で `.rev-stamp` とスクリプトの SHA-256 を読んで送信側と照合した。
- スクリプト対応: `env_info.txt` の `script_sha256`（`6cd05747…5ca48`）は、上記コミットの `run_metal_parity.sh` の SHA-256 と一致する。
- 実機ロック: `env_info.txt` の実行前・実行後とも `hw_lock: held owner=#3866 2026-11-04T15:03:57Z`。終了後にロックが外れていることを確かめた。
  ほかの利用者のプロセスは止めていない。開始前と各テスト前に、並走する cargo／rustc が無いことを確かめた（無し）。
- 負荷: `env_info.txt` の load average は実行前 8.30・実行後 10.33（1 分値）、メモリ空き率は前後とも 72%。時間の計測ではないため判定には使わない。
- 回収物は手で編集していない。`SHA256SUMS` はこのディレクトリで `shasum -a 256 -c SHA256SUMS` が通る。
  ホスト名・`/Users/`・ユーザー名の残存は 0 件（`<metal-node>`／`<user>`／`<home>` へマスク済み）。
- デバイス経路: 各ケースで先に `MetalBackendOps::new()` の `linalg_*` を直接呼び、`Unsupported`（静かなホストフォールバック）でないことを
  確かめてから `Var` で比較する（テストコードのとおり）。腕 A は `fandhe_ai::tape_for(Device::Metal)` で、CPU への無言のフォールバックは無い。
- 数値: 対象テストは各ケースの誤差を出力せず、`assert_parity`（REQ-2 統一複合判定・厳密ゼロ fail）と有限値の検査が通ったことだけを
  pass で示す。したがって本記録に最大誤差の数値は無い。pass は「全ケースで fail_count=0 かつ出力が有限」を意味する。

## 結果表（`summary.tsv` の写し）

| # | テスト（`--exact`） | ケース数 | exit | `test result` | 判定 |
|---|---|---|---|---|---|
| 1 | `facade_inv_gate_on_matches_cpu` | 16 | 0 | ok. 1 passed; 0 failed（0.42s） | pass |
| 2 | `facade_solve_gate_on_matches_cpu` | 16 | 0 | ok. 1 passed; 0 failed（0.23s） | pass |
| 3 | `facade_det_gate_on_matches_cpu` | 4 | 0 | ok. 1 passed; 0 failed（0.22s） | pass |
| 4 | `facade_cholesky_gate_on_matches_cpu` | 16 | 0 | ok. 1 passed; 0 failed（0.22s） | pass |
| 5 | `facade_qr_gate_on_matches_cpu` | 9（Q・R） | 0 | ok. 1 passed; 0 failed（0.21s） | pass |

入力は `SIZES=[1, 2, 8, 17, 64]` × `SCALES=[1.0, 1e20, 1e-20]` の対角優位行列・SPD 行列と `1e20 × I₂`。det は等倍・`n ≤ 17`（n = 1, 2, 8, 17）のみ、
qr は `QR_SHAPES=[(8,3), (3,8), (17,17)]` × `SCALES`（テストコードのとおり）。

対象の実装 issue #3405 は既に CLOSED。

## 回収物

| ファイル | 内容 |
|---|---|
| `rev-stamp.txt` | 展開ツリーの `.rev-stamp`（2 行形式） |
| `env_info.txt` | 実行前後の環境（macOS・CPU・メモリ・rustc・負荷・ロック保持） |
| `build.log` | 事前ビルド（`--no-run`）のログ |
| `summary.tsv` | 5 行の判定表 |
| `logs/gpu-linalg-3405/` | 各テストの全出力（`--nocapture`）と終了コード |
| `SHA256SUMS` | 上記の SHA-256 |

## #4044 追記（共有 target の成果物の再利用の棚卸し）

- 分類: **B（再利用あり、ただし前任とライブラリのコードが同一）**
- 根拠: build.log の `Compiling` は backend-cpu・backend-rocm・bench-harness・`fandhe-ai` のみで、backend-metal・tensor-core・autodiff 等は再利用。前任（3865）とのライブラリのコードの差分は 0 件。
- 判定基準・全 16 計測の表・限界は [`docs/real-hardware-shared-target-reuse-decision.md`](../../../real-hardware-shared-target-reuse-decision.md) §4。本追記は結果を変えない（既存の本文・回収ログは無改変）。
