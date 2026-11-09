# ROCm を facade 経由（`Device::Rocm`）で MI300X で測る（イシュー #3867）

**状態: 未実測**。下の記入欄は実測後に埋める。得ていない結果は書かない。

#4171 で `RULE.txt` の target_commit を登録し直した（旧 22f6c0a2f。crates/ などに 50 ファイルの差分が出たため）。実測前の登録し直しで、判定・腕・テスト・run 数は変えていない。#4044 の §4.11（`cargo clean -p` と成果物の時刻検査）を `run_rocm_facade.sh` に足した。

## 対象範囲

- `crates/facade/tests/rocm_facade_real_device.rs` の `#[ignore]` 4 本（`docs/backend-rocm-device-exposure-decision.md` §10.4）。
- backend 単体の parity（GEMM・linalg・softmax 系・reduce・loss・norm・paged KV）は #3159（`../rocm-parity-3159/`）が担う。本件は facade の `available_devices`・`tape_for(Device::Rocm(0))` の結線と、facade 経由の matmul・MLP 1 ステップの CPU parity を確かめる。
- 判定は REQ-2 統一複合判定（`fandhe_ai_backend_cpu::compare`。相対 1e-3 未満 または 絶対 1e-5 未満）のみ。tolerance・baseline は変えない。規則は `RULE.txt`（事前登録・編集しない）。

## 手順

`docs/real-hardware-verification-env.md` §9.5。#3159 と同じインスタンス・同じ展開ツリーで、#3159 の run1・run2 の後に
`bash docs/perf/logs/rocm-facade-3867/run_rocm_facade.sh <tree-abs-path> <out-abs-path>` を実行する（4 本を 1 本ずつ別プロセス・各 1 回）。
回収するのは `.rev-stamp`・`env_info.txt`・`build_artifacts.txt`・各テストの `test.log`・`exit_code`・`list_count` だけ。

## 記入欄

| 項目 | 値 |
|---|---|
| 実施日 | |
| .rev-stamp | |
| ROCm 版 | |
| rustc | |
| 判定（pass／fail／未実測） | |

テストごと:

| テスト | list_count | exit_code | 判定 | 比較ごとの fail_count・max_abs_diff・max_rel_err |
|---|---|---|---|---|
| `rocm_available_devices_lists_rocm0_and_tape_for_ok` | | | | （比較なし） |
| `rocm_facade_matmul_parity_vs_cpu` | | | | |
| `rocm_facade_mlp_train_step_parity_vs_cpu` | | | | |
| `rocm_tape_for_out_of_range_ordinal_is_err` | | | | （比較なし） |

## 秘密情報

インスタンス識別子・認証情報・内部ホスト名・個人のパス・GPU の UUID／BDF は収録しない。マスク検査（§9.4 手順 4）を通ったログだけを置く。
