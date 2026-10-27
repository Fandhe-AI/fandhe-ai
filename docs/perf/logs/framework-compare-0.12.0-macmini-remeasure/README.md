# 0.12.0 の 6 フレームワーク同席計測 — Mac mini（Apple M4）（イシュー #3273）

**状態: 実測済み（2026-10-27。系列 A・B とも 5/5 完走。系列 B を正式採用）。** 目的・判定規則・受け入れ条件・結果の表（新条件・旧条件・0.11.0 の 3 条件）・1.0 以下のセルの理由は `../framework-compare-0.12.0-remeasure/README.md` を正とする（二重管理しない）。本ディレクトリは Mac mini の生ログ・中央値・実機テストのログを収める。GB10 の値とは混ぜていない。

## 計測の要点

- 機体: Mac mini（`<metal-node>`・Apple M4・10 コア・16 GB）。run ごとに `machdep.cpu.brand_string == "Apple M4"` を検査した（M4 Max の値は使っていない）。
- 腕: 新条件 `bdaca14ccff7a01a4582eb84c6655f025a23208f`・旧条件 `0ae5670359c0f9d7a82f8bbd71791fb516713627`。全 10 run の `tree.txt`・`tree-old.txt` が `fandhe-ai v0.12.0`、`rev-stamp*.txt` が RULE の SHA と一致（`../framework-compare-0.12.0-remeasure/aggregate/check_identity.py` が系列 A・B とも rc=0）。
- Python 3 FW: uv の Python 3.12.15。`venv-torch` = torch 2.14.0・scipy 1.18.1・numpy 2.5.4、`venv-tf` = tensorflow 2.16.2・tensorflow-metal 1.2.0・numpy 1.26.4・keras 3.15.1（`env/pip-freeze-venv-*.txt`）。0.11.0 README の記録と同じ版で、TensorFlow は Metal の GPU を 1 台認識した。
- 事前登録規則: `RULE.txt`（固定 2026-10-27T07:17:00Z。系列 A の開始 07:17:47Z より前）。
- 日時: 系列 A 07:17:47Z〜07:26:03Z、系列 B 07:26:03Z〜07:37:16Z（`loop-macmini.log`）。
- 負荷: 開始時 load1 は系列 A 1.86／3.08／2.97／4.51／3.07、系列 B 2.01／1.87／2.98／2.52／2.08。系列 B の `gate.log` は 8 行で、run1・run4・run5 の前に 3.47・4.88・3.63 で 1 分ずつ待った。`GATE-INVALID`・`GATE-TIMEOUT` なし。
- 完走: 全 10 run の `run.log` 最終行が `done. rows=48 pyrows=32 oldrows=26 skipped=4`。skipped は burn Metal GEMM N=512〜4096 fresh の MEASURE_ERROR（結果全ゼロ・upstream 既知）の 4 件だけ。
- 実機ロック `~/work/.fandhe-hw-lock`（owner `#3273 2026-10-27T07:03:07Z`）を venv 構築の前に取り、系列・実機テスト・回収の後 07:56:09Z に外した。計測中にビルドは並走させていない（ビルドは系列 A の run1 の前のスモークと run1 の冒頭だけ）。

## 結果（Mac mini 系列 B の要点。表は GB10 側 README）

| セル | 新条件 | 旧条件 | 0.11.0 |
|---|---|---|---|
| gemm Metal N=256／512 | 1 位 1.16／1.06 | 0.93／0.84 | 0.84／0.80 |
| gemm Metal N=1024／2048／4096 | 0.99／0.94／0.85 | 0.86／0.81／0.75 | 0.69／0.73／0.58 |
| gemm CPU N=256〜2048 | 0.70／0.77／0.67／0.53 | 0.68／0.78／0.68／0.53 | 1.04／1.04／0.89／0.69 |
| train Metal／infer Metal | 1 位 1.01／1 位 1.02 | 0.32／0.94 | 0.32／1.13 |
| train CPU／infer CPU | 0.25／0.15 | 0.26／0.15 | 0.20／0.25 |

- 系列 A（対照）と系列 B で区分が反転したセルは 0 件。
- gemm CPU の窓は、全フレームワークで 0.11.0 の約 0.5〜0.8 倍だった（同じ Python FW の版）。N=256・512 の 0.11.0 の 1 位は、PyTorch の窓が 0.320／1.403 ms だったセッションの値で、本計測では 0.167／0.731 ms。

## 実機テスト（#3256 の M1〜M6、#3272 の X1）

時間系列の後、同じ実機ロックの中で新腕の全体ツリーを使って実行した（`tests-metal/`）。判定は `../resident-chain-parity-3256/RULE.txt`。

| 行 | テスト | 5 run の結果 |
|---|---|---|
| M1 | `metal_public_api_chain_matches_tape_path_bit_exact_two_layers` | 5/5 pass |
| M2 | `metal_public_api_chain_matches_tape_path_bit_exact_three_layers` | 5/5 pass |
| M3 | `metal_public_api_unread_steps_then_final_params_match_tape_path` | 5/5 pass |
| M4 | `metal_upload_resident_with_cpu_tape_is_device_mismatch` | 5/5 pass |
| M5 | 公開入口の encode 10・チェーン区間 wait 0・読み出しで wait 1 | 5/5 pass（毎回 `encode=10 command_buffers=1 wait=0 / read: wait=1`） |
| M6 | `DeviceParamStore` の wait が 1 回 | 5/5 pass |
| X1 | `bench-fandhe` `tests::resident_paths_match_tape_paths_metal`（#3272） | 5/5 pass |

35 ログすべてが `running 1 test`・`1 passed`・末尾 `rc=0` を満たす。invalid（foreign の cargo／rustc）は 0 件。

## 収録物

| パス | 内容 |
|---|---|
| `RULE.txt` | Mac mini の事前登録規則（固定 2026-10-27T07:17:00Z） |
| `scripts/macmini-run-0120.sh`・`macmini-0120a-loop.sh`・`macmini-0120b-loop.sh` | 1 round・系列 A・系列 B（本計測で使った版。変更なし） |
| `macmini-series-a/`・`macmini-series-b/` | `run{1..5}/`（`results-macmini-0.12.0.jsonl` 48 行・`-py` 32 行・`-old` 26 行・旧腕置換 `results-macmini-oldarm-swapped-0.12.0.jsonl`・`tree*.txt`・`rev-stamp*.txt`・`run.log`・`skipped-macmini-0.12.0.log`・`uptime_*.txt`）と `*-median5.jsonl`（新腕 48・Python 32・旧腕 26・旧腕置換 48）。系列 B は `gate.log` も持つ |
| `loop-macmini.log` | 系列 A → 系列 B の連続実行ログ |
| `smoke/` | RULE 固定前の行数スモーク（判定に使わない。パスはマスク済み） |
| `tests-metal/` | 実機テストのログ（`3256/`・`3272/`） |
| `env/` | `venv-torch`・`venv-tf` の `uv pip freeze` |
| `aggregate/median_rounds.py` | 0.11.0 版のコピー |
