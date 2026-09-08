# CPU GEMM (mc, nc) 2D 動的分配 `TwoDDynamic` variant の実装記録

イシュー #1311（親: #1310。設計: `docs/cpu-gemm-2d-dynamic-partition-design.md`。
兄弟: #1312「両実機 A/B・採否判定」・#1313「本番結線」）。

**状態: 実装完了・本番未結線**。両実機 A/B・採否判定は #1312 で実施済み
（判定は **undetermined**。M4 Max 専有ゲート未通過のため ADOPT を確定できず、
DGX 単独では採用しない規則〈設計 §11〉に従い #1313 へ「結線せず記録のみ」で
引き継ぐ。詳細は `docs/perf/cpu-gemm-2d-dynamic-partition-ab.md`）。本ファイルが
記録する実装自体（`crates/backend-cpu/src/gemm_blis/{partition.rs,mod.rs}` の
`#[cfg(test)]` 限定コード）は変更しない。本番公開入口
（`gemm_blis_parallel`／`gemm_blis_bias_act_parallel`）は引き続き未変更。

## 1. 目的・位置づけ

`RowPanel`（本番既定）は C を `m.div_ceil(num_threads)` 行の静的行パネルへ
等分割する。DGX Spark GB10（Cortex-X925 ×10 + Cortex-A725 ×10 の異種コア）
実機での N=1024・T=10 の非単調性が異種コア由来（H1）と確定し
（`docs/perf/cpu-gemm-candle-gate-remeasurement.md` §17）、行パネル単位の
動的配布のみの `IcDynamic`（#1366）は「pc ごとの同期点＋列全幅 B pack」の
構造が原因と推定される後退（DGX N=1024/2048 で対 RowPanel 比 0.63／0.85）
により REJECT 確定（#1367）した。

本イシューは、job（(行帯 × 列帯) の 2D タイル）を worker 数より多く生成し
rayon work stealing で動的分配する方式（設計 `docs/cpu-gemm-2d-dynamic-partition-design.md`）
を `GemmDriverVariant::TwoDDynamic` として `#[cfg(test)]` 限定で実装し、
既存 A/B ハーネス（`gemm_blis_variant_ab_1024_2048`／`_4096`）と bit 完全
一致回帰へ統合する。`IcDynamic` と異なり **job 内で K 全域を単一 worker が
同期なしで処理する**（pc ごとのバリアなし・split-K 禁止）ため、異種コア環境
でも job 単位の work stealing による再配分余地を広げつつ、`IcDynamic` の
後退要因と推定される構造を持ち込まない設計。

## 2. データフロー

```
gemm_blis_parallel_variant(TwoDDynamic, a, b, c, m, n, k, blocks)
  └ dispatch_two_d_dynamic(a, b, c, n, k, 0..m, blocks, Nn, TWO_D_JOBS_PER_WORKER)
      // arch 別 ISA トークン確定（x86_64: AVX-512→AVX2→Scalar／aarch64: NEON／他: Scalar）
      └ gemm_blis_two_d_dynamic_region<K>(kernel, a, b, c, n, k_dim, rows, blocks, transpose, jobs_per_worker)
          num_threads = effective_num_threads(rayon::current_num_threads())
          grid = partition::job_grid(mc_total, n, K::MR, K::NR, &blocks, num_threads, jobs_per_worker)
          jobs = split_c_into_jobs(c_region, n, &grid)   // column-band-major
          jobs.par_iter_mut().try_for_each(|job| run_two_d_job(kernel, a, b, job, n, k_dim, blocks, transpose, row_start))
              └ run_two_d_job: job-local C staging（copy-in）
                  → 既存・無改変の gemm_blis_ic_loop（jc→pc→ic→jr→ir）を job 幅で反復
                  → job-local C staging（copy-out）
```

パラメータ化入口 `gemm_blis_parallel_two_d_dynamic_with_params(a, b, c, m, n, k, blocks,
jobs_per_worker, transpose)` を通じて `jobs_per_worker`／`transpose`（Nn/Nt/Tn）を
注入できる（#1312 が `{2, 4}` をスイープする想定）。

## 3. `job_grid`（純関数。`partition.rs`）

設計 §5.1〜§5.2 の判定順序（`m==0`／`n==0` → `num_threads==1` → コスト最小化探索）
をそのまま実装した（`job_grid_with_trace` が内部の `reached_fallback` フラグ込みの
本体、公開 `job_grid` は薄いラッパー）。探索は `real_rb(rb)` の単調非減少性を
使った二分探索で `real_rb(rb) >= t` を満たす最小の `rb` を求め、コスト
`cost = real_cb_eff * m + real_rb * n`（`k` は共通因子のため省略）を最小化する
`(rb, cb)` を選ぶ。同コストは `nc_job` が大きい方を採用する（設計 §5.2）。

### 3.1 設計 §5.3 表との突合

`job_grid_reproduces_design_pack_table_rows`（`partition.rs`）が、設計 §5.3
「解析 pack 表」の 6 行（NEON MR=8/NR=12/NC=512 前提。`jobs_per_worker=2`）を
実装出力と突合し、**全 6 行が一致**することを確認済み（値の食い違いはなし。
設計記録 §5.3 の表を更新する必要はなかった）。

| N | T | row_bands | col_bands | mc_job | nc_job |
|---|---|---|---|---|---|
| 1024 | 10 | 5 | 4 | 208 | 264 |
| 1024 | 20 | 8 | 5 | 128 | 216 |
| 2048 | 10 | 4 | 5 | 512 | 420 |
| 2048 | 20 | 8 | 5 | 256 | 420 |
| 4096 | 8 | 2 | 9 | 2048 | 456 |
| 4096 | 20 | 5 | 9 | 824 | 456 |

### 3.2 その他の契約テスト（`partition.rs::tests`）

- `job_grid_tiles_equal_tile_grid_and_cover_exactly_once`: `tiles` が
  `tile_grid(m, n, mc_job, nc_job)` と同一集合（被覆完全・互いに素）
- `job_grid_aligns_mc_to_mr_and_nc_to_nr`: `mc_job % mr == 0`・`nc_job % nr == 0`
- `job_grid_meets_lower_bound_with_real_band_product`: ランダム形状 × T ×
  `jobs_per_worker` で `row_bands * col_bands >= bound` を検証（200 ケース）
- `job_grid_handles_degenerate_inputs`: `m==0`／`n==0`／`m<mr`／`n<nr`／
  `num_threads==1`／`usize::MAX` 近傍（`DimProductOverflow`）
- `job_grid_real_band_counts_are_monotone_non_decreasing`: `real_band` の
  単調非減少性
- `job_grid_lower_bound_survives_alignment_collapse`: 設計 §5.2「具体例 2」
  （alignment collapse 反例）を固定入力として再現し `(rb,cb)=(2,9)` を確認
- `job_grid_all_candidates_rejected_is_unreachable`: 300 ランダムケースで
  フォールバック分岐（`reached_fallback`）に到達しないことを実行時検査
- `job_grid_jobs_per_worker_product_non_decreasing`: 設計 §5.2「具体例 3」
  （`jobs_per_worker` 5→6 で job 数が逆転していた旧版の反例）が
  `(rb,cb)=(6,2)`・12 job で両方一致することを確認 + ランダム形状での
  経験的非減少性検査（100 ケース）

全 9 テスト green（`cargo test -p fandhe-ai-backend-cpu --lib gemm_blis::partition::tests::job_grid`）。

## 4. C 列分割: 実装形（S′。job-local C staging）

設計 §4.2 の主案 S（`Microkernel::run_rows` 新入口。4 ISA + scalar への
カーネル変更が必要）ではなく、以下の **S′**（job-local C staging）を採用した。
理由: `unsafe` 非導入という設計目的をより強く満たしつつ、既存・無改変の
`gemm_blis_ic_loop` を再利用できるため実装量・レビュー負荷が小さい。

- 各 job は自分の `mc_len × nc_len_job` の C 部分ブロックを job-local な
  連続バッファ `c_local`（行ストライド `ldc = nc_len_job`）へ 1 回だけ
  copy-in し、K 全域（jc→pc→ic→jr→ir）を**既存・無改変の `gemm_blis_ic_loop`**
  （`IcLoopContext.n` を C 行ストライドとしてのみ使う既存契約）で処理して
  から 1 回だけ copy-out する（`run_two_d_job`）
- `split_c_into_jobs` は `c.chunks_mut(n)`（行分割）→ 列帯境界での
  `split_at_mut` 連鎖により job 間の `&mut` 非重複をコンパイル時借用検査
  で保証する（`unsafe` を一切導入しない）
- job 配列は column-band-major（同一列帯の行帯を連続）に並べる（設計 §6。
  性能上の推奨で正しさには影響しない）

bit 完全一致（設計 §3 条件 1〜8）は、カーネル本体・`gemm_blis_ic_loop` を
一切変更せず、C アドレス解決のみを job-local バッファへ委譲することで
成立する（copy-in/out は安全な `copy_from_slice` のみ）。

### フォールバック連鎖（本イシューでは発生させない）

本イシューは性能 A/B を行わないため、S′ のまま完了する（性能上の障害が
#1312 で判明した場合のみ S → U〈raw pointer。#1338 承認済み〉への切替を
検討する。設計 §4.3）。

### #1312 への引き継ぎ: staging コピーの単離診断

`RAYON_NUM_THREADS=1` では `job_grid` が job 1 個を返し、`TwoDDynamic` は
「直列 `gemm_blis_ic_loop` ＋ C 全体の copy-in/out 1 回」に一致する。した
がって **T=1 の `TwoDDynamic`／`RowPanel` 比が staging コピーのオーバー
ヘッドを単離**する（コピー総量は job 数に依らず `2·m·n` 要素 = 演算量
`2·m·n·k` の `1/k`）。#1312 が REJECT となった場合に「2D 分配自体」と
「staging コピー」のどちらに帰属するかを判別する際、この手順を使うこと。

## 5. bit 完全一致回帰テスト（`mod.rs::tests`）

| テスト | 内容 |
|---|---|
| `gemm_blis_parallel_variant_all_candidates_match_naive_bit_exact` | `all_gemm_driver_variants()` 経由で `TwoDDynamic` を自動的に含む（既存拡張） |
| `gemm_blis_two_d_dynamic_matches_row_panel_bit_exact_across_shapes_and_threads` | C 初期値非ゼロ・端あり形状 11 種 × スレッド数 {1,2,3,16} × `jobs_per_worker` {1,2,4,8} |
| `gemm_blis_two_d_dynamic_multi_pc_matches_serial_bit_exact` | 小ブロック（`mc=16,kc=8,nc=24`）で job 内 jc/pc/ic ループを複数回通す |
| `gemm_blis_two_d_dynamic_is_deterministic_across_runs` | 同一入力・同一プール（T=3,16）で 2 回実行し bit 同一 |
| `gemm_blis_two_d_dynamic_transposed_matches_row_panel_bit_exact` | `Nt`／`Tn` が本番 `RowPanel` 経路（`gemm_blis_parallel_nt`／`_tn`）と bit 完全一致（#1313 結線対象の事前保証） |
| `gemm_blis_two_d_dynamic_matches_row_panel_bit_exact_large`（`#[ignore]`） | 1024/2048/4096 正方・release・プール既定スレッド数 |
| `split_c_into_jobs_rows_are_disjoint_and_cover_c` | 番兵値の書き込みで C の被覆完全性を直接検証 |

すべて green（`cargo test -p fandhe-ai-backend-cpu --lib gemm_blis::`。
131 passed・12 ignored・fail 0）。`#[ignore]` の大形状テストは実機
（Apple M4 Max。本エージェント実行環境がそのまま実機。2026-09-08）で
`--release -- --ignored gemm_blis_two_d_dynamic_matches_row_panel_bit_exact_large --nocapture`
を実行し pass を確認済み。DGX Spark GB10 側は #1312 の実測前に同コマンド
での再確認を引き継ぐ。

A/B ハーネス（`gemm_blis_variant_ab_1024_2048`／`_4096`。`#[ignore]`）は
`all_gemm_driver_variants()` に `TwoDDynamic` が追加されたことで自動的に
候補へ含まれる（sanity 実行のみ・値は採否根拠にしない。#1312 が本格計測する）。

## 6. スコープ外（#1313 が引き継いだ）

- **本番結線・`gemm_blis_bias_act_parallel` の epilogue 統合・`partition` の
  `#[cfg(test)]` 解除は #1313 で実施済み**（M4 Max 専有ゲート付き Phase 0
  再計測で ADOPT 確定・`TWO_D_DYNAMIC_PRODUCTION_ENABLED = true` で結線。
  §7 追記・`docs/perf/cpu-gemm-2d-dynamic-partition-ab.md`「#1313 追記」節
  参照）
- `oss-gemm-compare` への variant 選択オプション追加（#1312 では未実施のまま
  対象外と判断済み。#1313 でも同様に対象外のまま）
- 設計 §4.2 主案 S（`run_rows` 入口）の実装（S′ が障害に直面した場合の
  フォールバックとしてのみ着手する。#1313 でも S′ のまま問題なく結線でき
  たため未着手）

## 7. 実機実測（#1312）

両実機（DGX Spark GB10・Apple M4 Max）5 回独立プロセス中央値 A/B 計測を実施
した。DGX（専有ゲート通過）は `jobs_per_worker ∈ {2, 4}` とも全形状（N=1024/
2048/4096）で `RowPanel` を 1.09〜1.80 倍上回った。一方 Apple M4 Max は計測時
の 1 分 load average が概ね 30〜60（他セッション並走の共有負荷）で推移し、
事前宣言した専有ゲート（1 分 load average < 6 を 2 回連続）を通過しなかった。
採用ゲート条件 7（「ADOPT／条件付き ADOPT は M4 Max のゲート通過が必須」）に
より、M4 Max 側の数値が良好に見えても本イシューでは確証としない設計であり、
最終判定は **undetermined**（REJECT でも ADOPT でもない）と確定した。
DGX・N=1024 の T=10/T=8 非単調性（#1305）は `TwoDDynamic` で「残存」判定
（比 0.86 前後・閾値 0.90 未満）だが `RowPanel` の 0.48 から大幅に改善した。
詳細な実測表・判定根拠・#1313 への引き継ぎ内容は
`docs/perf/cpu-gemm-2d-dynamic-partition-ab.md` を参照。

**追記（#1313）**: M4 Max 専有ゲート付き Phase 0 再計測（1 回・有界）で
attempt=8（load average 3.40）にゲート通過し、jpw=2・jpw=4 とも Tier 1 条件を
両実機で満たしたため ADOPT が確定し本番結線した（既定 `TWO_D_JOBS_PER_WORKER=2`
は変更なし）。framework-compare gemm cpu N=512/1024/2048 × fresh/reuse の
before/after は両実機・全 12 セルで非後退（ratio 0.60〜0.96・改善方向）・
checksum 完全一致を確認した。詳細は
`docs/perf/cpu-gemm-2d-dynamic-partition-ab.md`「#1313 追記」節を参照。

## 出典

イシュー #1311・#1312・親 #1310・設計 `docs/cpu-gemm-2d-dynamic-partition-design.md`・
`docs/perf/cpu-gemm-ic-dynamic-variant.md`・
`docs/perf/cpu-gemm-2d-dynamic-partition-ab.md`・
`crates/backend-cpu/src/gemm_blis/{partition.rs,mod.rs}`。
