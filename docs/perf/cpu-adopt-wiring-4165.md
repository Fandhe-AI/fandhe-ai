# CPU 系の ADOPT 結線記録（イシュー #4165・親 #4164・ルート #4140）

Phase 4 の CPU 系の担当。Phase 3 の A/B で **ADOPT と判定された候補だけ**を既定経路へ結線し、前後比較を残す。REJECT は結線しない。

## 1. 状態

- 基準: origin/main `28435bb15`（2026-11-09）に始まり、2026-11-12・2026-11-13 に追記した。
- **追記（2026-11-12）: 推論限定 jpw1（#4161）は #4266 で ADOPT（記録 PR #4269）。所有者判断で aarch64 Linux だけ既定 ON に結線した。**
- **追記（2026-11-13）: 所有者が #4165 の結線を承認した（チャット。原文「承認します」）。次の 2 つを結線した。**
  - (1) SME 既定 ON（#4156 が ADOPT）: `crates/backend-cpu/src/gemm_blis/mod.rs` の `SME_PRODUCTION_ENABLED` を `true` にした（`docs/perf/logs/sme-default-on-ab-4156/on-arm.patch` と同じ 1 行）。
  - (2) N=256 の SME スケジュール t4-jpw1（#4162 が ADOPT(t4-jpw1)）: `crates/backend-cpu/src/gemm_blis/sme_small_sched.rs` の `SME_SMALL_SCHED_ENABLED=true`・`SME_SMALL_SCHED_THREADS=4`・`SME_SMALL_SCHED_JPW=1` にした（`docs/perf/logs/sme-n256-sched-ab-4162/arm-t4-jpw1.patch` と同じ値。`MAX_DIM=256` は据え置き）。
- **状態: ADOPT 3 件を結線済み（推論限定 jpw1・SME 既定 ON・t4-jpw1）。** 結線なしの記録として次の 2 件を残す。
  - eager release サイズゲート（#4158）: **REJECT**。結線しない。
  - eager release の Mac mini 非後退（#4159）: #4158 が REJECT のため**測っていない**（close）。eager release は OFF のまま。
- 本節以降の 2026-11-09 時点の記述（「ADOPT 0 件」「未実測」など）は、§2 の表と §4・§5・§6 で現状に直した。

## 2. 候補表（2026-11-13 時点）

| 候補 | 閉じた PR | ゲート定数（現在値） | 判定 | 結線 | 出典 |
|---|---|---|---|---|---|
| SME 既定 ON（#4156） | #4194（道具立て） | `crates/backend-cpu/src/gemm_blis/mod.rs` `SME_PRODUCTION_ENABLED = true` | **ADOPT**（Mac mini。到達 10 セル 5/5・ON/OFF 中央値比 gemm 0.463〜0.760・train 0.880〜0.913・ガード後退なし。GB10 は SME 非搭載でガードのみ pass・P-BIT 6726847 行 sha256 一致） | **結線済み**（2026-11-13・所有者承認） | `perf/cpu-gemm-sme-fmopa-microkernel.md` §5.17・§5.19 |
| SME unsafe 独立監査（#4157。前提条件） | #4200（自己点検）・#4277（独立監査の記録） | — | **済み**（独立 security-auditor 監査で P0・P1 なし） | 結線の前提を満たした | `cpu-gemm-sme-unsafe-audit.md` §14.5・§14.11・§14.12 |
| eager release サイズゲート（GB10 #4158） | #4195（道具立て）・#4228（判定記録） | `crates/autodiff/src/eager_release.rs:48` `EAGER_MATMUL_RELEASE_DEFAULT_ENABLED = false` | **REJECT**（2026-11-10・GB10。処置 4 セルは 5/5 で改善したが、ガードの既定経路 gemm reuse N=4096 が 5/5 round で 1.00 超） | 結線しない（OFF のまま） | `perf/cpu-gemm-reuse-fresh-inversion.md` §18.4 |
| eager release の Mac mini 非後退（#4159） | #4198（道具立て） | 同上 | **測っていない**（#4158 が REJECT のため。2026-11-13 close） | 結線しない（OFF のまま） | 同 §19（記述は「部品のみ・未実測」） |
| 推論限定 jpw1（#4161） | #4196 | `crates/backend-cpu/src/gemm_blis/infer_scope.rs:24` `INFER_SCOPE_JPW1_ENABLED = cfg!(aarch64 Linux)` | ADOPT（#4266・記録 PR #4269） | aarch64 Linux だけ既定 ON に結線（2026-11-12。macOS・x86 は OFF のまま） | `cpu-gemm-small-m-job-merge-prepack-decision.md` §14 |
| N=256 の SME スケジュール（#4162） | #4201（道具立て）。記録 PR #4283 | `crates/backend-cpu/src/gemm_blis/sme_small_sched.rs` `ENABLED = true`・`THREADS = 4`・`JPW = 1` | **ADOPT(t4-jpw1)**（Mac mini。gemm 256 fresh 0.677・reuse 0.764・ガード 4 セルの中央値比 0.996〜1.008・後退なし） | **結線済み**（2026-11-13・所有者承認） | `perf/cpu-gemm-sme-fmopa-microkernel.md` §5.18・§5.19 |

補足: thin（`SME_THIN_M64_ENABLED`）は判定 undetermined のため OFF のまま。

## 3. REJECT・未測定の扱い

REJECT（#4158）と「測っていない」（#4159）は別物として扱う。どちらも結線しないが、#4159 は REJECT ではなく、測っていない。#4168 の分類の付け直しでも混同しない。

## 4. 結線したこと

- 定数 4 つ（上記 (1)(2)）を、ADOPT した腕の patch と同じ値にした。腕は patch で定数を書き換えて測っており、本番で同じ値にすれば測った挙動と一致する。
- ドリフト検出テストを新しい値に合わせて反転・書き換えた。`mod.rs` の `sme_production_disabled_after_gb10_reject` は `sme_production_enabled_is_true_with_neon_fallback` へ戻した。`sme_small_sched.rs` の `sme_small_sched_disabled_by_default` は `sme_small_sched_enabled_with_t4_jpw1`、`..._plan_is_inert` は `sme_small_sched_enabled_plan_fires_in_band_only` へ書き換え、ON の値と帯内の plan・帯外の素通りを固定する。
- **挙動が変わる範囲**は、aarch64 で OS が SME と f32f32 を報告し、SVL が 64B の機体だけ。
  - m≥256・n≥256・k≥64 は NEON から SME になる。
  - そのうち m=n=256・k∈[64,256] は、専用プール 4 スレッドと jpw=1 になる。
  - x86、および SME の無い aarch64（GB10。検出が false）は不変。
- **数値は bit 一致**。#4156 の bitdump（ON/OFF の sha256 一致）と #4162 の P-BIT で確認済み。FMA 契約・unsafe・tolerance・baseline・依存・`docs/spec/` は変えない。
- 別 PR で扱う事項（本結線に含めない）: #4278（sme・avx2・avx512 の `kernel_unchecked{,_with_ldc}` 6 関数を `pub(crate)` にする。所有者判断 (a)）、#4280（thin のゲートを `SME_PRODUCTION_ENABLED` に従属させる）、#4279（P3 の `checked_mul`）。
- thin（`SME_THIN_M64_ENABLED`）と eager release は OFF のまま。

## 5. 前後比較の扱い（所有者判断）

- 所有者判断で、結線のための追加の性能計測はしない。既存の 2 件の合成で代替する。
  - #4156: OFF 対 ON（`perf/cpu-gemm-sme-fmopa-microkernel.md` §5.17）。
  - #4162: SME ON を base とした t4-jpw1 対 base（同 §5.18）。
- **現 main（OFF）と結線後（ON + t4-jpw1）を直接並べた計測は無い**。合成は、2 つの効果が独立に掛かる前提の推定であり、直接の実測ではない。
- マージ前に Mac mini で PR ブランチを検証する（結果は PR に記録する。本書に値は書かない）。
- 5 ラウンド中央値の前後比較を、本結線の直接計測として後から取る場合は新しいイシューで扱う。

## 6. 申し送り

解消したもの:

- #4156 の ADOPT（§5.17）・#4157 の独立監査（`cpu-gemm-sme-unsafe-audit.md` §14.5・§14.11）・#4162 の ADOPT（§5.18）が揃い、2026-11-13 に所有者が結線を承認した。
- 2026-11-09 時点で「実機の A/B 結果も独立監査も出ていない」としていた状態は解消した。

残るもの:

- **#4278**: sme・avx2・avx512 の `kernel_unchecked{,_with_ldc}` 6 関数の `pub(crate)` 化（所有者判断 (a)）。別 PR。
- **#4279**: P3 の `checked_mul`。別 PR。
- **#4280**: thin のゲートを `SME_PRODUCTION_ENABLED` に従属させる。別 PR。`cpu-gemm-sme-unsafe-audit.md` §13.6 の P2（thin のゲート従属・thin 側の lazy-ZA の回帰テスト）の系列。thin を有効にする前に必須。
- **M4 Max の非後退（未計測）**: §5.4.1 で境界 (256,64) の中央値が 1.161、2/5 run が 1.0 未満で 5/5 を満たさなかった。SME を報告する Linux aarch64 機も未計測。所有者判断で受け入れて記録する。計測できる機会に後続で確かめる。
- **ロールバック**: `SME_PRODUCTION_ENABLED` を `false` に戻す 1 行と、ドリフト検出テスト `sme_production_enabled_is_true_with_neon_fallback` の反転（t4-jpw1 は SME OFF なら効かない）。t4-jpw1 だけを外す場合は、`sme_small_sched.rs` の 3 定数と対応するテスト。
- **過去の A/B の道具の再現**: #4156・#4162 の腕の patch（`on-arm.patch`・`arm-t4-jpw1.patch` など）と、その A/B スクリプト（`scripts/bench/framework-compare/run_ab_sme_default_on_4156_cpu.sh` など）は、結線前の main を前提にしている。結線後の main には当たらない（2026-11-13 に `git apply --check` で確認）。再現は結線前のコミット（`00db8720e`）で行う。
- eager release（#4158 REJECT・#4159 未測）の既定反転は #3227 の別判断。現時点では結線しない。
- Metal／CUDA 系は #4166・#4167、分類の付け直しは #4168。
