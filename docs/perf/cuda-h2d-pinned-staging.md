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

| 項目 | 結果 |
|---|---|
| ゲート A（`#[ignore]` 実機テスト） | 未実施（記入欄） |
| Layer A（framework-compare 非後退） | 未実施（記入欄） |
| Layer B（H2D 単体 A/B） | 未実施（記入欄） |
| **verdict** | **undetermined**（実機未到達） |

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
  bit 単位で検証）まで実装済みだが、本 PR 実行環境に DGX Spark GB10 実機
  への到達手段がなく実行自体は未実施のまま §4 の記入欄を残す。Layer B
  マイクロベンチハーネス・framework-compare `--pinned-h2d` の実装は未着手
  のままで、実機到達可能なセッションでの計測と合わせて行う。
