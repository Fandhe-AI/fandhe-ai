# ROCm（MI300X・gfx942）対 CPU の f32 GEMM parity（イシュー #3159）

**状態: 未実測**（所有者のインスタンス起動待ち）。本ディレクトリの現状は段 A（準備のみ）で、下の記入欄はすべて空欄。得ていない結果は書かない。

## 対象範囲

- 測るのは `RocmGemmF32::run`（f32 GEMM・`GfxTarget::Gfx942`）と `is_available()`（`probe_real_device.rs`）のみ。
- reduce・softmax・norm など `docs/backend-rocm-ops-coverage-plan.md` の R0〜R13 は未実装（`BackendOps` 未実装・起票は所有者承認待ち）のため対象外。各 R イシューが同じ `#[ignore]` 方式でテストを足す。
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

## fail 時の切り分け（推測で結論を書かない）

1. fail の形状・要素数・最大誤差を上の表へ記録する。
2. bit 不一致と fail の関係を見る（K が大きいほど増えるか、端数形状に偏るか）。
3. 補助診断（記録のみ・判定に使わない）: `render_gemm_f32_source` と同じソースをオフラインの `clang -x hip --offload-arch=gfx942 --offload-device-only -O3 -S` に通し、`v_fma_f32`・`v_fmac_f32`・`v_pk_fma_f32` の有無を数える。HIPRTC と同一ではない。
4. 原因候補を書き、`.claude/rules/out-of-scope-tracking.md` に従い後続 Issue として提案する（起票はユーザー承認後）。
- tolerance の緩和・baseline の追加・`-ffp-contract` の追加は行わない。

## 秘密情報

インスタンス識別子・認証情報・内部ホスト名・個人のパス・GPU の UUID／BDF は収録しない。マスク検査を通ったログだけを置く。
