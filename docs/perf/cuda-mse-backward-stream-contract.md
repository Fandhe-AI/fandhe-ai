# CUDA `mse_loss_backward` ストリーム順序契約の確認・GB10 A/B（イシュー #1692）

親 #1582（MSE backward を Metal encode-only／CUDA ストリーム化）の分解
sub-issue。「CUDA 側 `mse_loss_backward` が本設計文書の非同期投入契約に
既に準拠していることを確認し、必要な場合のみ軽微な調整を行う」を目的とする。

## 1. コード読解で確定した事実（実測ではない）

- `crates/backend-cuda/src/ops.rs::CudaBackendOps::mse_loss_backward` は
  明示的な `stream.synchronize()` を持たず、
  `crates/backend-cuda/src/mse.rs::CudaMse::run_mse_backward_f32` へ
  委譲するのみ。
- `run_mse_backward_f32`（forward の `run_mse_loss_f32` も同様）は
  `clone_htod`（非同期 H2D）→ カーネル起動（非同期）→
  `memory::readback`（`clone_dtoh` + `stream.synchronize()` を 1 箇所へ
  集約）という構成で、`grep -rn synchronize
  crates/backend-cuda/src/{mse,kernels_mse}.rs` は 0 件（`docs/backend-
  cuda-async-execution-design.md` §2.2 棚卸し表に追記済み）。
- `docs/backend-cuda-async-execution-design.md` §2.3 が定める「ホスト
  `Tensor` を返す `BackendOps` 演算は戻り値の D2H が構造的な同期点で
  あり、readback 境界 1 箇所へ集約する」契約に**既に準拠している**。
- 唯一の潜在リスクだった I4（`CudaSlice::drop` が `has_async_alloc()`
  偽の環境ではホスト側 `stream.synchronize()` へフォールバックする）は、
  `docs/perf/lowlayer-diagnosis-2026-09-12.md` §2（出典
  `docs/perf/logs/lowlayer-diagnosis-2026-09-12/dgx/
  async_ordering_real_device.log`）が GB10 実機 `has_async_alloc() =
  true` を実測済みであることで解消済みと確認した。

**結論**: `crates/backend-cuda/src/{ops,mse}.rs` への機能変更は不要。
本 PR の変更は `docs/backend-cuda-async-execution-design.md`（§2.2 棚卸し
表・§16 実装記録の追記）・`crates/backend-cuda/src/mse.rs`（doc comment
追記のみ）・`crates/facade/tests/mse_backward_bench.rs`（新設マイクロ
ベンチ）・本ドキュメントに限る。

## 2. 事前登録判定規則（実装着手前に固定。issue コメント原文）

イシュー #1692 のコメント
<https://github.com/Fandhe-AI/fandhe-ai/issues/1692#issuecomment-5654656690>
に固定済み。要約:

- **対象**: `crates/facade/tests/mse_backward_bench.rs`（新設・`#[ignore]`）
  の `mse_backward_cases`。デバイス `FANDHE_BENCH_DEVICE=cuda`。
- **形状**: 主 `train_shape`（`[64, 10]`。訓練ハーネス実形状・640 要素・
  固定費支配）、副 `general_shape`（`SIZES = [16384, 65536, 1048576]`。
  一般形状スイープ）
- **計測区間**: forward は `.to_tensor()` で計測前に実体化、計測区間は
  `tape.backward(&loss)` のみ（`mse_loss` の VJP 呼び出しに限定）
- **手順**: 5 round・run 単位で before/after の起動順を反転・warmup 5 →
  計測 20
- **判定**: `median_s` の ratio(after/before) <= 1.00 を非後退の目安と
  する。ただし本 issue は `crates/*/src` への機能変更を伴わない見込み
  のため、**`crates/*/src` の差分がコメントのみ（または皆無）の場合、
  before/after は同一ソースの再測定であり ADOPT／REJECT の判定対象では
  なく、ノイズ床・再現性の記録として扱う**（この解釈自体を規則として
  明記し、事後に持ち出さない）
- **checksum**: `grad[...].fold_bits` が before/after で完全一致する
  こと（数値契約は不変のため必須）
- **正しさ（REQ-2）**: `crates/backend-cuda/tests/mse_parity.rs::
  mse_matches_cpu_across_shapes`（`#[ignore]`）を GB10 で実行し複合判定
  （相対誤差 1e-3 未満 または 絶対誤差 1e-5 未満）pass を確認する
- **秘密情報**: 内部ホスト名を出力・報告に含めない（`docs/real-
  hardware-verification-env.md` の規約どおり `$CUDA_NODE` 変数越しに
  扱う）

## 3. 実測記録

本エージェント実行環境（Linux コンテナ／worktree）に DGX Spark GB10
実機への到達手段（`docs/real-hardware-verification-env.local.md`・
`CUDA_NODE` 環境変数・`~/.ssh/config` のいずれも確認できず）がないため、
**実測値は一切含まれていない**。本ドキュメントはスキャフォールド
（実行スクリプト・事前登録判定規則・記入欄）のみを提供し、実測は GB10
実機を持つセッションへ申し送る（`docs/perf/logs/train-resident-grad-
cuda-1560/` と同型の運用）。

実行スクリプト・記入欄は
`docs/perf/logs/cuda-mse-backward-1692/`（`README.md`・
`orchestrate.sh`・`env_info.txt`）を参照。

**2026-09-16 実測済み → DGX Spark GB10 実機（driver 580.173.02・CUDA 13.0・
NVRTC V13.0.88・rustc 1.97.0・Linux 6.17.0-1031-nvidia）で `orchestrate.sh`
を 5 round（起動順反転・別プロセス・warmup 5 → 計測 20）実行し、下表を
埋めた。** 生ログ・集計は `docs/perf/logs/cuda-mse-backward-1692/`
（`runs-1692/`・`mse_parity.log`・`orchestrate_stdout.log`・`env_info.txt`・
`aggregate.py`／`aggregate.md`）を参照。実測条件: 専有ゲートなし・実行
区間の load average 1.41〜1.53・GPU 利用率 0〜4%（他プロセスの GPU メモリ
常駐あり〈python 170 MiB・Xorg 18 MiB〉だが演算負荷なし）。

- **before/after の差分範囲（要注記）**: before 腕は `c9505fb5`（PR #1780
  の第 1 親）・after 腕は `3e43bbd0`（`crates/`・`scripts/` が origin/main
  `565300e4` と同一）で、両者の間には 108 コミット（#1780〜#1888）が入る。
  計測区間 `tape.backward(&loss)` が通る MSE 経路——
  `crates/backend-cuda/src/ops.rs::CudaBackendOps::mse_loss_backward`・
  `crates/backend-cuda/src/mse.rs::CudaMse::run_mse_backward_f32`（本
  issue の追記は doc comment 7 行のみ）・`crates/autodiff/src/grad.rs` の
  `Op::MseLoss` 分岐と `mse_loss_vjp`——は before/after で byte 同一である
  ことを確認した。一方、tape の backward driver（`crates/autodiff/src/
  backward.rs`）は #1859（no_grad／detach）・#1861（retain_graph／
  `backward_accumulate`）の差分を含むため、本 A/B は「MSE 経路は同一ソース
  の再測定」だが「計測区間全体が完全に同一ソース」ではない。§2 の規則
  どおり ADOPT／REJECT の判定対象とはせず、ノイズ床・再現性の記録として
  扱う。
- **ノイズ床**: 5 round 中央値の比（after/before）は全 4 case で
  0.9687〜1.0069。round 別の比は 0.8348〜1.1423 で、`general_shape[1048576]`
  が最も広い（10 run 中 3 run〈r1 before・r2 after・r5 after〉が約
  0.0072〜0.0074 s、残り 7 run が約 0.0082〜0.0087 s の bimodal。低値群は
  腕にも起動順にも固定されず、`docs/perf/cuda-gemm-tiled-naive-speedup-
  4096-triage.md` §6 と同様の確率的 slow／fast モードと整合するが機構は
  本記録では未特定）。`train_shape`（主対象・640 要素）は before
  0.000017024〜0.000017968 s・after 0.000017504〜0.000018080 s で、5 round
  中央値の比 0.9955・round 別 0.9742〜1.0620。
- **再現性**: `grad[...].fold_bits` は 4 case すべてで before 5 run・after
  5 run の 10 run 完全一致（腕内・腕間とも）。数値契約不変。
- **正しさ（REQ-2）**: `mse_parity.rs::mse_matches_cpu_across_shapes` は
  GB10 で pass（1 passed・0 failed・0.35s。`mse_parity.log`）。

### 記入欄（GB10 実機実測後に埋める。2026-09-16 実測済み）

| 項目 | 値 |
|------|-----|
| 実測日 | 2026-09-16（GB10 実機。UTC 01:41:25Z〜01:41:42Z） |
| before_sha | `c9505fb5`（PR #1780 の第 1 親。MSE 経路は after と byte 同一） |
| after_sha | `3e43bbd0`（`crates/`・`scripts/` は origin/main `565300e4` と同一） |
| `mse_parity.rs::mse_matches_cpu_across_shapes` | pass（1 passed・0 failed・0.35s） |
| `train_shape` median_s ratio（after/before・5 round 中央値） | 0.9955（before 0.000017680 s → after 0.000017600 s） |
| `general_shape[16384/65536/1048576]` median_s ratio | 1.0069／1.0046／0.9687（before 0.000032464／0.000087664／0.008438272 s → after 0.000032688／0.000088064／0.008174207 s） |
| `grad[...].fold_bits` 完全一致 | 一致（4 case × 10 run。`0x7fb3edad4355b8f2`／`0x4812f01cab0b2ac3`／`0xf7ea5b2a5f6e9680`／`0x134c26f7f42eb94a`） |
| 結論（ADOPT／REJECT／ノイズ床記録のみ） | ノイズ床記録のみ（§2 規則どおり判定なし。比 0.9687〜1.0069・fold_bits 完全一致・parity pass） |

## 4. スコープ外事項

- MSE VJP の `dPred` はホストへ readback された後、`Op::LinearResident`
  等の次段 VJP へ再度 H2D される可能性がある（backward チェーン全体で
  のホスト往復の残存）。これは `BackendOps::linear_forward_device` 系の
  デバイス常駐チェーン拡張（#1216／#1673 の設計）と関係する別範囲で
  あり、本 issue のスコープ外とする（`.claude/rules/out-of-scope-
  tracking.md` に従いユーザー承認なしに issue 化はしない）。

## 5. 出典

- `crates/backend-cuda/src/{ops,mse}.rs`
- `docs/backend-cuda-async-execution-design.md` §2.2・§2.3・§16
- `docs/perf/lowlayer-diagnosis-2026-09-12.md` §2
- `crates/facade/tests/elementwise_vjp_bench.rs`（イシュー #1583。本ベンチの参照実装）
- `crates/backend-cuda/tests/mse_parity.rs`
