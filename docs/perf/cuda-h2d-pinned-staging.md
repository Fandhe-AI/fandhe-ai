# CUDA H2D pinned staging（イシュー #1585）

## 1. 背景・目的

D2H 側（`MemoryOps::with_host_view`。イシュー #1336・#1478）には形状ごとに再利用する
ホストステージングバッファ（`crate::host_staging::HostStagingCache`。`Pinned`／
`Pageable` 2 種・既定 `Pinned`）が既に導入済みだが、H2D 側（`MemoryOps::upload`／
`upload_into`・`gemm.rs` の各 `run_*`／`launch_*` 系・`ops.rs` の転置 NT 分岐が発行する
`clone_htod`／`memcpy_htod`）には対称な機構がなかった。cudarc-0.19.8 の `clone_htod`／
`memcpy_htod` は内部で `cuMemcpyHtoDAsync` を発行するが、pageable ソースでは driver が
一時 pinned バッファへ同期的にステージングするため、呼び出し元が明示的に pinned
メモリを用意すればこの暗黙ステージングを避けられる可能性がある。

本イシューは、この H2D 用 pinned staging を **既定 OFF・明示 opt-in** で追加する
（低レイヤー診断 `docs/perf/lowlayer-diagnosis-2026-09-12.md` §7 表の B-3 行で起票
された候補）。既存の H2D 実測値（`docs/perf/cuda-gemm-reuse-phase-breakdown.md` §4:
`h2d_a` = 0.085／0.304／1.163 ms @ N=1024/2048/4096。同一反復の `host_copy` は
1.4／5.7／20.6 ms）から、H2D は framework-compare 単位の反復時間に対し数 %〜1 割
程度の寄与に留まると見積もられる。このため Layer A（framework-compare 実践規模）
は**非後退ゲート**、Layer B（H2D 単体マイクロ A/B）を**改善根拠**とする役割分担を
事前に定める。

既定化（`HOST_STAGING_KIND` 切替〈#1478〉と同型のユーザー承認＋security-auditor
到達）は本イシューのスコープ外。判定結果が ADOPT でも既定は OFF のまま出荷し、
既定化は別イシューへ引き継ぐ。

## 2. 設計概要

- opt-in フラグ: `crate::host_staging::{set_pinned_h2d_enabled, pinned_h2d_enabled}`
  （`crate::placement::{set_managed_placement_enabled, managed_placement_enabled}`
  と同型のプロセスワイド `AtomicBool`・`Ordering::SeqCst`）。facade 委譲は
  `fandhe_ai::{set_cuda_pinned_h2d_enabled, cuda_pinned_h2d_enabled}`。
- H2D 専用キャッシュ `crate::host_staging::H2dStagingCache`: D2H 側
  `HostStagingCache` とは別型で、**同一 numel に対して複数エントリ**
  （`HashMap<usize, Vec<H2dStagingEntry>>`。LIFO）を保持する。正方 GEMM の A・B
  が同一 numel になる場合に 2 個目が毎回 miss して `cuMemHostAlloc` を都度発行する
  （機構と無関係な性能後退）ことを避けるため。世代検査・`HOST_STAGING_CAP_BYTES`
  （256 MiB）共有・`release_all` は D2H 側と同じ契約。
- 新規確保は既存の `HostStaging::alloc(HostStagingKind::Pinned, ..)`
  （D2H 側と共有する唯一の `unsafe` ブロック）を再利用する。**新規 `unsafe` は
  追加していない**。
- ヘルパ関数 `crate::host_staging::{upload_new, upload_into}`:
  - フラグ OFF または `data` が空の場合は、それぞれ
    `stream.clone_htod(data)`／`stream.memcpy_htod(data, dst)` をそのまま呼ぶ
    （導入前と経路・出力とも bit 同一）。
  - ON の場合はキャッシュから pinned バッファを取得（miss なら新規確保）し、
    `data` を `as_mut_slice()`（`PinnedHostSlice` 側は内部 `event.synchronize()`
    で前回発行分の完了を待つ）で同期コピーしてから、**`PinnedHostSlice` 自身**を
    `clone_htod`／`memcpy_htod` へ渡す（`as_slice()` で得た `&[f32]` を渡すと
    `[T]` 側の `HostSlice` 実装が event を記録せず非同期 DMA と次回のホスト
    書き込みが競合しうるため、必ずステージング型そのものを渡す契約）。
- 結線箇所（f32 経路限定）:
  - `crates/backend-cuda/src/memory.rs::CudaMemory::{upload_inner, upload_into}`
    の `Device` 配置分岐（`h2d_staging: Arc<Mutex<H2dStagingCache>>` フィールド）。
  - `crates/backend-cuda/src/gemm.rs::CudaGemm`: `upload_h2d_new`（薄い委譲。
    `pub(crate)`）を新設し、本番 f32 経路の `clone_htod` 呼び出し
    （`run_f32_kernel`／`run_tiled_bias_act_f32`／`run_tiled_f32_nt`／`_tn`／
    `run_tiled_f32_resident_lhs_nt` 系／`upload_f32` 等）を置換。**TF32
    Tensor Core 経路**（`run_wmma_tf32` が到達する `run_wmma_f32_kernel`／
    `run_wmma_tf32_opt_kernel`／`run_wmma_tf32_staged_kernel`）も入力が
    `&[f32]` のため同じ `upload_h2d_new` を経由し、有効化時は pinned
    staging の対象に含まれる（f32 系カーネルという構造上の帰結であり、
    個別に選別結線したものではない）。
  - `crates/backend-cuda/src/ops.rs`: `gemm_resident_lhs` の NT 転置分岐
    （`gemm.upload_h2d_new(bt)`）。
  - **対象外**（変更なし・従来どおり `stream.clone_htod` 直呼び）: f16
    Tensor Core 経路（`crates/backend-cuda/src/gemm_mma.rs` の `run_f16`
    系・`gemm.rs::run_f16_kernel`。f16 データは `upload_h2d_new` の型
    （`&[f32]`）と一致しないため構造的に非到達）・elementwise／rmsnorm／
    softmax／transpose／mse。
- I3（ホスト側一時バッファの解放）の再確認は
  `docs/backend-cuda-async-execution-design.md` §3 に追補済み。

## 3. 事前登録規則（実測前に固定。計測後に変更しない）

対象機体: DGX Spark GB10（sm_121）。内部ホスト名は記録しない。負荷ゲートは
`record_only`（uptime を計測前後に記録し判定条件にはしない）。

- **ゲート A（必須・マージ条件）**: `#[ignore]` 実機テスト全件 pass。on/off の
  出力が bit 同一（REQ-2 複合判定は bit 同一の十分条件として自動充足）。
- **Layer A（非後退ゲート）**: framework-compare（gemm cuda N∈{1024,2048,4096}
  ×{fresh,reuse}・train cuda 64×{fresh,reuse}・副次 infer 64 reuse）で off→on
  交互 5 run、中央値比 `ratio = on/off <= 1.00` かつ checksum 完全一致。1 セルでも
  `ratio > 1.00` なら当該セルを「後退」として記録する。
- **Layer B（改善根拠）**: H2D 単体（発行＋`synchronize`）で N ごとに 5 プロセス
  起動中央値の `pinned_staged/pageable`。`< 1.00` を改善、`>= 1.00` を非改善として
  記録する。
- **判定**: ADOPT-as-opt-in ＝ ゲート A pass かつ Layer A 全セル非後退かつ
  Layer B 全 N 改善。REJECT ＝ ゲート A pass だが Layer A に後退セルあり、
  または Layer B 全 N 非改善。undetermined ＝ 5 run 揃わない／実機未到達／
  checksum 不一致（不一致はバグとして先に修正）。
- **判定が規定するもの**: 記録上の verdict のみ。既定値は結果に依らず OFF
  （opt-in 契約）。ADOPT でも既定化は行わず別イシューへ引き継ぐ。REJECT でも
  opt-in 実装は削除せず維持する（対照腕・再計測用）。

## 4. 実測結果

**本イシューの実装環境に DGX Spark GB10 実機への到達手段がないため、実測は
未実施のまま記入欄を残す**（`docs/real-hardware-verification-env.local.md` が
本 worktree に存在しない）。ゲート A（`#[ignore]` テスト）・Layer A／B の
ハーネス自体は §5 のとおり整備済みで、実機到達可能なセッションでの実行手順は
`docs/real-hardware-verification-env.md`・メモリ「DGX Spark 実機作業手順」を
参照する。
**（2026-09-16 追記）** ゲート A を DGX Spark GB10 実機で実測済み（§4.1）。
**（2026-09-16 同日追記）** Layer A／B のハーネスを同日に実装し（§5・§6）、同じ GB10 で
実測を完了した（§4.2）。事前登録規則（§3）により **verdict は REJECT**（ゲート A pass・
Layer A 後退セルあり・Layer B 全 N 非改善）。既定 OFF は不変・opt-in 実装は維持。

| 項目 | 結果 |
|---|---|
| ゲート A（`#[ignore]` 実機テスト） | 2026-09-16 実測済み → **pass（3 passed / 0 failed・rc=0）**。DGX Spark GB10（sm_121）・driver 580.173.02・CUDA 13.0・rustc 1.97.0・ツリー 3e43bbd0（crates/・scripts/ は origin/main 565300e4 と同一）。`pinned_h2d_sequential_uploads_match_plain_upload_bit_exact`・`pinned_h2d_upload_into_partial_update_matches_plain_bit_exact`・`pinned_h2d_upload_after_release_staging_is_bit_exact` の 3 件。ログ: `docs/perf/logs/cuda-h2d-pinned-staging-1585/gateA.log` |
| Layer A（framework-compare 非後退） | 2026-09-16 実測済み（§4.2）→ **後退セルあり（判定 8 セル全て `on/off > 1.00`）**。gemm cuda fresh/reuse: N=1024 2.0758／1.4136・N=2048 1.5288／1.1961・N=4096 1.2166／1.2175。train cuda 64 fresh/reuse: 1.0595／1.0539。checksum は全セル完全一致。副次（非判定）の infer 64 reuse は 0.9584（改善方向）。専有ゲート（load1 < 1.0 かつ GPU 0 % の 30 秒 3 連続）は初回で通過 |
| Layer B（H2D 単体 A/B） | 2026-09-16 実測済み（§4.2）→ **全 N 非改善**。`pinned_staged/pageable`（5 プロセス起動中央値）: N=1024 2.400・N=2048 2.992・N=4096 3.297（pageable 0.0813／0.2942／1.1449 ms・pinned_staged 0.1951／0.8802／3.7747 ms）。両腕の読み戻しは bit 同一・pinned 腕のキャッシュ hit を確認 |
| **verdict** | **REJECT**（§3 の規則: ゲート A pass だが Layer A に後退セルあり、かつ Layer B 全 N 非改善）。**既定 OFF（`PINNED_H2D_ENABLED=false`）維持・opt-in 実装は削除せず維持**（対照腕・再計測用。§3「判定が規定するもの」） |

### 4.1 2026-09-16 GB10 実測記録（ゲート A のみ）

- 実行コマンド: `cargo test -p fandhe-ai-backend-cuda --release --all-features
  --test pinned_h2d_real_device -- --ignored --nocapture --test-threads=1`
- 結果: `test result: ok. 3 passed; 0 failed; 0 ignored`（finished in 0.87s・rc=0）。
  §3 ゲート A の「`#[ignore]` 実機テスト全件 pass・on/off 出力 bit 同一」を充足。
- 環境: NVIDIA GB10（sm_121）・driver 580.173.02・CUDA 13.0（NVRTC V13.0.88）・
  rustc 1.97.0・Linux 6.17.0-1031-nvidia。負荷ゲートは `record_only` 相当
  （1 分 load average 1.53・GPU 利用率 2 %・他利用なし）。内部ホスト名は
  記録しない（`docs/perf/logs/cuda-h2d-pinned-staging-1585/env_info.txt`）。
- Layer A／Layer B は §3 で事前登録したものの、この時点では対応するハーネスが
  本リポジトリに未実装だった。同日中にハーネスを実装して実測した記録は §4.2。

### 4.2 2026-09-16 GB10 実測記録（Layer A／Layer B。ハーネス実装後）

- ツリー: 本ブランチ `8f598e2b`（Layer A／B ハーネス実装コミットまで。`crates/`
  の本番コードは origin/main `565300e4` と同一・追加はテスト／ベンチ／スクリプトのみ）。
  環境は §4.1 と同一（GB10・driver 580.173.02・CUDA 13.0・rustc 1.97.0）。
- **Layer B**（`docs/perf/logs/cuda-h2d-pinned-staging-1585/run_layer_b.sh`。
  `pinned_h2d_upload_ab_1585` を 5 プロセス起動・各 run 20 warmup＋20 計測・
  `--test-threads=1`・record_only〈実行前 load average 0.15・実行後 0.35〉）:

  | N | MiB | pageable median（5 run 中央値） | pinned_staged median | pinned_staged/pageable | 判定 |
  |---|---|---|---|---|---|
  | 1024 | 4 | 0.0813 ms | 0.1951 ms | 2.400 | 非改善 |
  | 2048 | 16 | 0.2942 ms | 0.8802 ms | 2.992 | 非改善 |
  | 4096 | 64 | 1.1449 ms | 3.7747 ms | 3.297 | 非改善 |

  pinned 腕の cold（初回 miss・`cuMemHostAlloc`）は 0.89／2.84／11.23 ms。両腕とも
  読み戻し bit 同一（`fold_bits`）・pinned 腕は計測区間の全呼び出しが
  `H2dStagingCache` hit（`h2d_staging_stats`）。集計: 同ディレクトリ `aggregate.md`・
  生ログ `layer_b_run{1..5}.log`。
- **Layer A**（`scripts/bench/framework-compare/run_ab_pinned_h2d_cuda.sh
  pinned-h2d-1585`。`--features pinned-h2d-toggle` ＋ `AB_PATCH_FACADE_PATH`〈同一
  ツリーの `crates/facade`〉・単一バイナリ〈sha256 一致〉・off→on 交互 5 run・
  専有ゲート既定 ON で初回通過〈load1 0.60／0.36／0.42・GPU 0 %・常駐
  compute_apps=2 は記録のみ〉）:

  | セル | off median | on median | on/off | checksum | 判定 |
  |---|---|---|---|---|---|
  | gemm/cuda/1024/fresh | 953.4 us | 1.979 ms | 2.0758 | 完全一致 | 後退 |
  | gemm/cuda/1024/reuse | 2.078 ms | 2.938 ms | 1.4136 | 完全一致 | 後退 |
  | gemm/cuda/2048/fresh | 4.177 ms | 6.386 ms | 1.5288 | 完全一致 | 後退 |
  | gemm/cuda/2048/reuse | 8.343 ms | 9.980 ms | 1.1961 | 完全一致 | 後退 |
  | gemm/cuda/4096/fresh | 38.080 ms | 46.327 ms | 1.2166 | 完全一致 | 後退 |
  | gemm/cuda/4096/reuse | 37.737 ms | 45.946 ms | 1.2175 | 完全一致 | 後退 |
  | train/cuda/64/fresh | 530.0 us | 561.6 us | 1.0595 | 完全一致 | 後退 |
  | train/cuda/64/reuse | 313.7 us | 330.6 us | 1.0539 | 完全一致 | 後退 |
  | infer/cuda/64/reuse（副次・非判定） | 99.1 us | 95.0 us | 0.9584 | 完全一致 | （参考） |

  生ログ・JSONL・ゲートログ・`compare-pinned-h2d-*.md`・DGX 側ランナーは
  `docs/perf/logs/cuda-h2d-pinned-staging-1585/layer_a/`（内部ホスト名・絶対パスは
  マスク済み）。skipped ログは空（失敗セルなし）。
- **判定**: §3 の事前登録規則により **REJECT**（ゲート A pass・Layer A に後退セル
  8／8・Layer B 全 N 非改善）。規則の事後緩和は行わない。既定 OFF は不変・opt-in
  実装は維持する。
- **原因の推定（判定とは分けて記録）**: GB10 は unified memory 構成のため pageable
  H2D 自体がすでにメモリコピー相当の帯域（4 MiB を 0.08 ms ≈ 50 GB/s）で完了して
  おり、pinned staging が追加するホスト側の 1 回の `memcpy`（ページャブル→pinned）が
  純増分として現れる。ディスクリート GPU（PCIe 経由の DMA）で pinned が有効になる
  前提が GB10 では成立していない可能性が高い。gemm fresh セルの後退幅が reuse より
  大きいのは fresh が毎反復 H2D を含むためで、Layer B の比と整合する。
  この推定の検証（ディスクリート GPU での再計測・`memcpy` 段の分離計測）は本
  issue の対象外。

機械的自己監査（GPU 非依存で確認可能な事項）は以下のとおり済み:

- `crates/backend-cuda/src/host_staging.rs` の実コード `unsafe` は
  `HostStaging::alloc` の `Pinned` 分岐 1 箇所のまま（新規 `unsafe` 追加なし）。
- `PINNED_H2D_ENABLED` の既定値は `false`（drift ガード
  `default_is_disabled_when_no_prior_test_left_it_enabled` で機械検証）。
- フラグ OFF 時の `upload_new`／`upload_into` は `stream.clone_htod`／
  `memcpy_htod` を直接呼ぶだけであり、導入前と経路・出力は構造的に bit 同一
  （`H2dStagingCache` へ一切触れない）。
- `cargo clippy --workspace --all-targets --all-features -- -D warnings`・
  `cargo fmt --all -- --check`・`cargo test --workspace --all-features`
  （GPU 非依存分。785 件超・pre-existing failure なし）は本リポジトリの
  Linux/macOS ホスト（CUDA 実機なし）で確認済み。

## 5. GPU 非依存で整備済みのテスト・ハーネス

- `crates/backend-cuda/src/host_staging.rs::h2d_staging_tests`: フラグ既定値・
  往復・`H2dStagingCache` の take/put（同一 numel 複数エントリ・世代不一致
  破棄・cap 超過破棄・poison 回復）を GPU 非依存で検証（10 件・全 pass）。
- 実機テスト（`#[ignore]`。本イシュー実行環境では未実行）は §6「引き継ぎ」参照。

## 6. 引き継ぎ・スコープ外

- 実機（DGX Spark GB10）でのゲート A／Layer A／Layer B 実測（§3・§4）。
  → ゲート A は 2026-09-16 に GB10 で実測済み（pass。§4.1）。Layer A／B は
  ハーネス未実装のため未実施のまま（下記）。
- ~~Layer A／B ハーネスの実装（別イシューへ引き継ぎ）~~ → 2026-09-16 に本ブランチで
  実装済み（Layer A: `scripts/bench/framework-compare/run_ab_pinned_h2d_cuda.sh`・
  `bench-fandhe --pinned-h2d`〈feature `pinned-h2d-toggle`〉・`compare_pinned_h2d_ab.py`。
  Layer B: `crates/backend-cuda/tests/pinned_h2d_upload_ab_1585.rs`・
  `docs/perf/logs/cuda-h2d-pinned-staging-1585/{run_layer_b.sh,aggregate.py}`）。
  GB10 実測・verdict（REJECT）は §4.2。
- REJECT の原因推定（§4.2「原因の推定」）の検証（ディスクリート GPU での再計測・
  ホスト側 `memcpy` 段の分離計測）。GB10 単独では pinned staging が有効になる前提
  （PCIe DMA）が成立しない可能性が高い。
- pinned H2D staging の既定化（`HOST_STAGING_KIND`〈#1478〉と同型のユーザー
  承認・security-auditor 到達が前提）。
- f16 Tensor Core 経路（`gemm_mma.rs` の `run_f16` 系・`run_f16_kernel`。
  TF32 Tensor Core 経路〈`run_wmma_tf32`〉は f32 系カーネルのため §2 の
  とおり既に対象内）・elementwise／rmsnorm／softmax／transpose／mse への
  拡張。
- キャッシュのプロセス／ordinal 共有化（`CudaMemory`／`CudaGemm` それぞれが
  インスタンス単位でキャッシュを持つため、プロセス全体の pinned 常駐量は
  理論上 `cap_bytes` の複数倍になりうる）。
- 実機 `#[ignore]` テスト自体は `tests/pinned_h2d_real_device.rs`（PR #1678
  の codex-review 指摘対応で追加。`PinnedHostSlice` を実際に経由する
  `upload_new`／`upload_into` の連続 upload・部分更新・解放後の再確保を
  bit 単位で検証）まで実装済み。ゲート A は §4.1・Layer A／B は §4.2 で
  GB10 実測済み。
