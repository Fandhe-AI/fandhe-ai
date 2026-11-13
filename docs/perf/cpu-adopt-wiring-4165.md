# CPU 系の ADOPT 結線記録（イシュー #4165・親 #4164・ルート #4140）

Phase 4 の CPU 系の担当。Phase 3 の A/B で **ADOPT と判定された候補だけ**を既定経路へ結線し、前後比較を残す。REJECT は結線しない。

## 1. 状態

- 基準: origin/main `28435bb15`（2026-11-09）。
- **追記（2026-11-12）: 推論限定 jpw1（#4161）は #4266 で ADOPT（記録 PR #4269）。所有者判断で aarch64 Linux だけ既定 ON に結線した。以下の「0 件」は 2026-11-09 時点の記述で、jpw1 以外の 4 候補は引き続き未結線。**
- **ADOPT 0 件・REJECT 0 件・判定なし（未実測）5 件。結線はしていない。**
- 依存イシュー（#4156・#4157・#4158・#4159・#4161・#4162）はすべて CLOSED だが、閉じた PR はどれも道具立て（計測部品・監査の自己点検）だけで、実機の A/B 結果も独立監査も出ていない。

## 2. 候補表

| 候補 | 閉じた PR | ゲート定数（現在値） | 判定の状態 | 止まっている理由 | 出典 |
|---|---|---|---|---|---|
| SME 既定 ON（#4156） | #4194 | `crates/backend-cpu/src/gemm_blis/mod.rs:3558` `SME_PRODUCTION_ENABLED = false` | 未実測 | #4142（NB-RANGE5／NB-SEP）・#4143（GB10 の数え方）の所有者回答が未了で、RULE §9 が空。実機セッション未実施 | `perf/cpu-gemm-sme-fmopa-microkernel.md` §5.17 |
| SME unsafe 独立監査（#4157。前提条件） | #4200 | — | 自己点検のみ。独立 security-auditor は未実施 | §14.5 が空 | `cpu-gemm-sme-unsafe-audit.md` §14.1・§14.5 |
| eager release サイズゲート（GB10 #4158） | #4195 | `crates/autodiff/src/eager_release.rs:48` `EAGER_MATMUL_RELEASE_DEFAULT_ENABLED = false` | 未実測 | 実機セッション未実施 | `perf/cpu-gemm-reuse-fresh-inversion.md` §18 |
| eager release の Mac mini 非後退（#4159） | #4198 | 同上 | 未実測 | 同上 | 同 §19 |
| 推論限定 jpw1（#4161） | #4196 | `crates/backend-cpu/src/gemm_blis/infer_scope.rs:24` `INFER_SCOPE_JPW1_ENABLED = cfg!(aarch64 Linux)` | ADOPT（#4266・記録 PR #4269）。aarch64 Linux だけ既定 ON に結線（#4165） | 結線済み（macOS・x86 は OFF のまま） | `cpu-gemm-small-m-job-merge-prepack-decision.md` §14 |
| N=256 の SME スケジュール（#4162） | #4201 | `crates/backend-cpu/src/gemm_blis/sme_small_sched.rs:37` `SME_SMALL_SCHED_ENABLED = false` | 未実測 | RULE 未固定（テンプレートのみ）。#4156 が ADOPT でない限り本番に効かない | `perf/cpu-gemm-sme-fmopa-microkernel.md` §5.18 |

## 3. 0 件の意味

0 件は「判定が出ていない」ためで、**REJECT と判定されたからではない**。#4168 の分類の付け直しでは「結線なし・未判定」として扱い、REJECT と混同しない。

## 4. 結線しなかったことの確認

- 上の 4 つの定数は false のまま。ドリフト検出テスト（`mod.rs` の `sme_production_disabled_after_gb10_reject`）も反転していない。
- コード・定数・テスト・依存・tolerance・baseline は変更していない。
- 結線がないため前後比較は取得していない（取得対象がない）。

## 5. 参考: ADOPT が出た場合の結線手順（本 PR では実行しない）

- SME 既定 ON: **所有者承認と #4157 の独立監査（§14.5 の記入）の両方**が条件。満たしたら `SME_PRODUCTION_ENABLED` を true にし、`sme_production_disabled_after_gb10_reject` を反転する。
- eager release: #4158 が ADOPT かつ #4159 が非後退のときだけ `EAGER_MATMUL_RELEASE_DEFAULT_ENABLED` を反転する。
- 推論限定 jpw1: GB10 だけで ADOPT になった場合は、機体を限って有効にするか、Mac mini の非後退を別途確かめる（#3177 で L2 形状が 1.06〜1.14 に悪化した前例がある）。
- N=256 SME スケジュール: #4156 の ADOPT が前提。
- 共通: 結線ごとに前後比較（5 ラウンドの中央値）を記録する。後続の結線は新しいイシューで扱う。

## 6. 申し送り（所有者の確認事項）

- 依存 5 イシューは道具立ての PR の `Closes` で実測前に閉じている。#4165 を 0 件で閉じると、Phase 4・Phase 6（0.15.0）は CPU 系の改善なしで進む。再オープンして実測するか、0.15.0 では 0 件を受け入れるかは所有者の判断事項。
- #4142・#4143 の所有者回答（RULE §9 への転記）、#4157 の独立 security-auditor 監査、各 A/B の実機セッションが未了。
- `cpu-gemm-sme-unsafe-audit.md` §13.6 の P2 2 件（thin のゲートが `SME_PRODUCTION_ENABLED` に従属していない・thin 側に lazy-ZA の回帰テストがない）の是正イシュー起票は承認後に行う。
- Metal／CUDA 系は #4166・#4167、分類の付け直しは #4168。
