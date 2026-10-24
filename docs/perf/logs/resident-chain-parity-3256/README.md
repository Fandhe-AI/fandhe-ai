# 常駐チェーンの実機 parity と学習 1 step の同期回数（イシュー #3256）

親 #3248（学習 1 step の同期を 1 回へ）の仕上げ。公開入口 `Sequential::train_step_resident` →
`ResidentLoss::read`（#3254）と、公開 API だけで書いた parity テスト（#3255）を Mac mini（Metal・Apple M4）と
DGX Spark GB10（CUDA）で実行した記録。事前登録規則は `RULE.txt`（測定前に固定。実測値を含まない）。
工程の記録は `docs/compat-resident-train-step-decision.md` §16。

## 結論

- **Metal・CUDA とも、常駐チェーンの loss 列と更新後パラメータは Tape 経路と bit 一致した**（M1〜M4・C1〜C5 が各 5 run すべて pass）。
- **学習 1 step の同期は 1 回**（loss を読む 1 回だけ）。
  - Metal（公開入口 `train_step_resident` を通す M5）: encode 10・command buffer 1・チェーン区間の `waitUntilCompleted` 0・
    `ResidentLoss::read` を含めて 1。従来の Tape 経路は 7 回（`docs/perf/train-step-phase-breakdown.md` §18.3）。
  - CUDA（公開入口と同じ呼び出し列を backend-cuda 内で再現した C7）: 窓（存在検証・tape 作成・snapshot・validate・チェーン）の
    `stream_syncs`・`d2h_calls`・`h2d_calls`・`mem_uploads`・`mem_allocs` がすべて 0、読み出し区間（新 tape 作成を含む）で
    `stream_syncs` 1・`d2h_calls` 1（4 byte）。従来の Tape 経路は約 8〜9 回（同 §19.6。机上の読解で、実測ではない）。
  - 残る同期点: なし（窓内 0、読み出し 1 のみ）。
- CUDA TF32 モード（C8・記録専用）: チェーンは **`Ok` で成功し、第 2 段（Tape 経路）へフォールバックしない**（窓の `d2h_calls` 0・
  `stream_syncs` 0）。C5（TF32 モードの公開入口の bit 一致）も 5 run すべて pass。
- 不一致・invalid run: なし。

## 結果（各行 5 run・独立プロセス。判定は `RULE.txt`）

| 行 | ノード | 内容 | 5 run の結果 |
|---|---|---|---|
| M1 | Mac mini | `metal_public_api_chain_matches_tape_path_bit_exact_two_layers` | 5/5 pass |
| M2 | Mac mini | `metal_public_api_chain_matches_tape_path_bit_exact_three_layers` | 5/5 pass |
| M3 | Mac mini | `metal_public_api_unread_steps_then_final_params_match_tape_path` | 5/5 pass |
| M4 | Mac mini | `metal_upload_resident_with_cpu_tape_is_device_mismatch` | 5/5 pass |
| M5 | Mac mini | 公開入口の encode 10・チェーン区間 wait 0・読み出しで wait 1 | 5/5 pass（毎回 `encode=10 command_buffers=1 wait=0 / read: wait=1`） |
| M6 | Mac mini | `DeviceParamStore` の wait が 1 回 | 5/5 pass |
| C1 | GB10 | `cuda_public_api_chain_matches_tape_path_bit_exact_two_layers` | 5/5 pass |
| C2 | GB10 | `cuda_public_api_chain_matches_tape_path_bit_exact_three_layers` | 5/5 pass |
| C3 | GB10 | `cuda_public_api_unread_steps_then_final_params_match_tape_path` | 5/5 pass |
| C4 | GB10 | `cuda_upload_resident_with_cpu_tape_is_device_mismatch` | 5/5 pass |
| C5 | GB10 | `cuda_tf32_mode_chain_matches_tape_path_bit_exact` | 5/5 pass |
| C6 | GB10 | #3252 の B1（内部チェーンの窓 0・読み出し 1） | 5/5 pass |
| C7 | GB10 | 公開入口と同じ呼び出し列の窓 0・読み出し 同期 1／D2H 1・初回 `mem_allocs > 0` | 5/5 pass（毎回 window 全 0・read `stream_syncs: 1, d2h_calls: 1`） |
| C8 | GB10 | TF32 モードでのチェーンの結果（記録専用） | 5/5 `Ok`（フォールバックなし）。初回 step の `mem_allocs` 11・同期 0・D2H 0 |
| C9 | GB10 | P7 の wait と、内部 tape を捨てた後の read の機能確認 | 5/5 pass |

全 75 ログが `rc=0`・`running 1 test`・`1 passed` を満たす（0 件実行の素通りなし）。invalid（foreign の cargo／rustc 検出）は 0 件。

## 監査範囲（カウンタの範囲外）

- facade 固有のホスト処理（`contains_resident_unsupported_layer`・`reject_untracked_parametric_layer_resident`・
  `build_device_chain_steps`）はデバイス呼び出しを持たない純計算のためコード監査で補った。CUDA の公開入口は
  facade から診断カウンタへ届かない（`internal-diagnostics` を facade の dev-dependency にすると `cargo test --workspace` の
  対象が増えるため採らない）ので、**C7 は facade から直接数えたものではなく、同じ呼び出し列を backend-cuda 内で再現して数えた値**。
- P7 の `wait_device_param_store`（CUDA は `synchronize_device`）は `stream_syncs`（`readback_with` 内の同期だけを数える）の
  対象外。CUDA は機能確認（C9）のみで回数は記録していない。
- Metal の M5 は公開入口を通した値（`__diagnostic_batch_counters_snapshot`）。

## 測定ツリー

- 測定コミット `d6295634e5850942cedd1206bbbb87fd9b737e78`（作業中コミット。squash で消えるため `crates/` の tree hash を記録する）。
  測定ツリーの `crates/` の tree hash: **`e72669c2bff31d10f713bb4450ae6501855e1160`**。
- `macmini/.rev-stamp`・`gb10/.rev-stamp`（`stage-tree.sh` が書いた展開ツリーのもの）。ノード側で `.tree-manifest`（12406 件）を
  `shasum -a 256 -c`／`sha256sum -c` で全件照合し、非 OK は両ノードとも 0 件（`tree-manifest-check.txt`）。
- ビルドは測定前に 1 回（`build.log`・判定外）。測定中はビルド・別プロセスを並走させていない。実機ロック
  （`~/work/.fandhe-hw-lock`・owner `#3256 <UTC>`）は両ノードで取得し、回収後に解除した。
- GB10 は他テナントの常駐サービス（vLLM・ComfyUI）が compute プロセスとして居る状態（`env_info.txt`）。計数・bit 一致の判定は
  時間を測らないため GPU 負荷に依存しない（`RULE.txt`）。

## 同梱物

`RULE.txt`・`run_rows.sh`（ノード側スクリプト）・`macmini/`・`gb10/`（`.rev-stamp`・`tree-manifest-check.txt`・`env_info.txt`・
`build.log`・`<行>.run{1..5}.log`〈末尾に `rc=`〉・`DONE`）。ホスト名・ユーザー名・絶対パスはマスク済み。

## 対象外

- facade から直接 CUDA の同期を数える仕組み（feature 統合の影響があるため別途判断）。
- Tape 経路の CUDA 同期回数の 5 run 実測（§19.6 は机上の読解のまま）。
- CUDA TF32 でチェーンが FP32 厳密のまま走る理由の掘り下げ（本記録は挙動の観測のみ）。
