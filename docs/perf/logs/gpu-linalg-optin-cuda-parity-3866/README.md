# GPU 線形代数の opt-in 関数の facade 経由の実機 parity（CUDA。#3866）

対象は #3405（PR #4014）で facade に公開した `fandhe_ai::set_cuda_gpu_linalg_enabled`／`cuda_gpu_linalg_enabled`。
`crates/facade/tests/linalg_gpu_optin_cuda_parity.rs` の `#[ignore]` テスト 6 本（inv・solve・det・cholesky・qr・eigh）を、
`<cuda-node>`（DGX Spark GB10。`docs/real-hardware-verification-env.md`）で 1 本ずつ実行し、ゲート ON の CUDA tape と CPU tape を突き合わせた。
Mac mini 側は同 PR の [`../gpu-linalg-optin-metal-parity-3866/`](../gpu-linalg-optin-metal-parity-3866/README.md)。

- 事前登録: [`RULE.txt`](./RULE.txt)（宣言 2026-11-04T15:01Z。実機実行より前にコミット 49a470f4b で push した。実行後に変えていない）
- 実行スクリプト: [`run_cuda_parity.sh`](./run_cuda_parity.sh)（#3864 版 `../onnx-train-cuda-parity-3864/run_cuda_parity.sh` の複製。
  変えたのは対象一覧と、非 0 終了の判定分岐〈tape_for の失敗は undetermined・それ以外は fail〉だけで、差分はスクリプト冒頭に列挙した。
  GB10 上で手動実行。CI からは呼ばない。`--self-test` 付き）
- tolerance 定数（`RELATIVE_TOLERANCE`／`ABSOLUTE_RESCUE_THRESHOLD`）・parity baseline・判定規則・入力集合・テストコードは不変

## 位置づけ

同じ 6 本は 2026-10-27 に GB10 で 5 run すべて pass している（[`../cuda-linalg-parity-3138/README.md`](../cuda-linalg-parity-3138/README.md) の cmd3。#3138）。
当時の `GateOn` は backend の `fandhe_ai_backend_cuda::linalg::set_gpu_linalg_enabled` を直接呼んでいた。#3405 で `GateOn` を
facade の公開関数経由に変えたので、本測定はその 1 行委譲を通しても同じ結果になることを確かめる。カーネル parity（G2）は #3138 で充足済み。

## 状態

**実測済み: 6 本すべて pass、fail 0、undetermined 0**（2026-11-04・`<cuda-node>`）。

- 実行コミット（`rev-stamp.txt`）: `49a470f4b4196f736bbe9631ce4d556c825f568e`・`dirty=0`。事前登録と実行スクリプトを含むコミットそのもので、
  `crates/`・`Cargo.toml`・`Cargo.lock`・`rust-toolchain.toml` は対象コミット `2b00e494b` と同一（送信前に `git diff --quiet` で確認）。
  転送は rsync だけで、ノード側で `.rev-stamp` とスクリプトの SHA-256 を読んで送信側と照合した。
- スクリプト対応: `env_info.txt` の `script_sha256`（`99a453fd…d0faf`）は、上記コミットの `run_cuda_parity.sh` の SHA-256 と一致する。
- 実機ロック: `env_info.txt` の実行前・実行後とも `hw_lock: held owner=#3866 2026-11-04T15:03:53Z`。終了後にロックが外れていることを確かめた。
  ほかのサービスのプロセスは止めていない（常駐 compute プロセスは実行前後とも 3 件。GPU 使用率は前後とも 0%）。並走する cargo／rustc は無し。
- 回収物は手で編集していない。`SHA256SUMS` はこのディレクトリで `shasum -a 256 -c SHA256SUMS`（Linux は `sha256sum -c`）が通る。
  ホスト名・`/home/`・ユーザー名の残存は 0 件（`<cuda-node>`／`<user>`／`<home>` へマスク済み）。
- デバイス経路: 各ケースで先に `CudaBackendOps::new(0)` の `linalg_*` を直接呼び、`Unsupported`（静かなホストフォールバック）でないことを
  確かめてから `Var` で比較する（テストコードのとおり）。腕 A は `fandhe_ai::tape_for(Device::Cuda(0))` で、CPU への無言のフォールバックは無い。
- 数値: 対象テストは各ケースの誤差を出力せず、`assert_parity`（REQ-2 統一複合判定・厳密ゼロ fail）と有限値の検査が通ったことだけを
  pass で示す。したがって本記録に最大誤差の数値は無い。pass は「全ケースで fail_count=0 かつ出力が有限」を意味する。

## 結果表（`summary.tsv` の写し）

| # | テスト（`--exact`） | ケース数 | exit | `test result` | 判定 |
|---|---|---|---|---|---|
| 1 | `facade_inv_gate_on_matches_cpu` | 16 | 0 | ok. 1 passed; 0 failed（0.32s） | pass |
| 2 | `facade_solve_gate_on_matches_cpu` | 15 | 0 | ok. 1 passed; 0 failed（0.25s） | pass |
| 3 | `facade_det_gate_on_matches_cpu` | 16 | 0 | ok. 1 passed; 0 failed（0.24s） | pass |
| 4 | `facade_cholesky_gate_on_matches_cpu` | 16 | 0 | ok. 1 passed; 0 failed（0.41s） | pass |
| 5 | `facade_qr_gate_on_matches_cpu` | 16（Q・R） | 0 | ok. 1 passed; 0 failed（0.27s） | pass |
| 6 | `facade_eigh_gate_on_matches_cpu` | 16（固有値・固有ベクトル） | 0 | ok. 1 passed; 0 failed（0.70s） | pass |

入力は `SIZES=[2, 3, 8, 17, 64]` × `SCALES=[1.0, 1e20, 1e-20]` の対角優位行列・SPD 行列（κ ≤ 1e3）と `1e20 × I₂`（solve は `1e20 × I₂` なし）。
det は CPU の期待値が非有限（例: `1e20 × I₂` → 1e40）のとき bit 一致で判定する（テストコードのとおり）。

対象の実装 issue #3405 は既に CLOSED。

## 回収物

| ファイル | 内容 |
|---|---|
| `rev-stamp.txt` | 展開ツリーの `.rev-stamp`（2 行形式） |
| `env_info.txt` | 実行前後の環境（GPU・ドライバ・nvcc・rustc・負荷・常駐件数・ロック保持） |
| `build.log` | 事前ビルド（`--no-run`）のログ |
| `summary.tsv` | 6 行の判定表 |
| `logs/gpu-linalg-3405/` | 各テストの全出力（`--nocapture`）と終了コード |
| `SHA256SUMS` | 上記の SHA-256 |
