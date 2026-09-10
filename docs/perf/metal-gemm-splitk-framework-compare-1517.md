# Metal split-K 結線前後 framework-compare A/B（イシュー #1517）

## §0 状態

**未実測**（2026-09-10 時点）。本ラン（Linux x86_64・CI 環境）には Apple
M4 Max 実機への到達経路がないため、本ドキュメントは計測スクリプト・
帰属表（機械生成済み）・事前登録判定規則までを整備し、実測は Mac セッ
ションへ引き継ぐ（ルート #1509 の運用方針。メモリ
`issue-1509-linux-side-policy`）。実測値は捏造しない——以下 §5/§6 の表は
空欄のまま残す。

- verdict: **未実測**
- 結線維持可否: **未確定**（実測後に §6 へ記録する）

## §1 前提

- 「結線前後」の実体は `crates/backend-metal/src/tile.rs` の
  `SPLIT_K_DISPATCH_AUTO_PRODUCTION_ENABLED` 定数の `false`（結線前・
  現行既定）/`true`（結線後）である（イシュー #1516・PR #1530 で
  `MetalGemm::dispatch_auto` → `tile::select_route_for_device` →
  `GemmRoute::SplitK` 分岐を定数ゲート付きで追加済み）。
- `docs/backend-metal-splitk-decision.md` §5「ゲート既定値・切替条件」に
  記載の切替手順: ①#1515 ADOPT 確定 → ②定数 `true` → ③実機 `#[ignore]`
  群 pass → **④本イシュー（#1517）の A/B 非後退・checksum 一致** →
  ⑤後退時 `false` へ差し戻し。
- `SPLIT_K_NUMERIC_CONTRACT_APPROVED` は承認済み（#1513 で `true`）。
- 既存 `run_ab_gemm_metal.sh`（イシュー #1306。before=crates.io 承認ピン
  registry／after=HEAD path patch）は本イシューには使えない
  （0.8.0 ↔ HEAD はゲート以外の差分も含むため、字義通りの「結線前後」
  にならない。`docs/perf/metal-gemm-n4096-kernel-gap.md` §19.1 と同型の
  教訓）。本イシュー専用の `run_ab_splitk_metal.sh`（両腕とも
  `crates/facade` への path patch）を用いる。

## §2 帰属表（framework-compare の各 GEMM 形状が split-K に到達するか）

### §2.1 `gemm` タスク（N=512/1024/2048/4096・正方・NN・contiguous）

`MetalBackendOps::gemm` は `dispatch_auto`（split-K 判定を行う唯一の
本番入口）へ進む。`tile::should_split_k` は正方形状・N≥512 では
並列度条件（`actual_groups >= 40`。`tile.rs::should_split_k_with` 手順 2）
により **`None`**（split-K 非対象）と判定される——8 セルはいずれも
split-K 非到達。本 A/B の `gemm` セルで検出できるのは
`select_route_for_device` の判定コスト自体（純関数 1 回・classic 経路と
同じ結果を返すまでの分岐オーバーヘッド）のみである。

### §2.2 `train` タスク（`bench-fandhe`: `BATCH=64・D_IN=784・
D_HIDDEN=256・D_OUT=10`）

`crates/backend-metal/tests/splitk_train_shape_attribution.rs`
（`cargo test -p fandhe-ai-backend-metal --test
splitk_train_shape_attribution -- --nocapture`。Linux で実行・実装計画
§3.3 手順 6 の定数出典確認込み）の実測出力:

| shape | (m,n,k) | should_split_k | plan |
|---|---|---|---|
| forward L1 (64,784)x(784,256) | (64,256,784) | Some | partitions=4 k_per_partition=192 bm=32 bn=32 bk=16 |
| forward L2 (64,256)x(256,10) | (64,10,256) | Some | partitions=8 k_per_partition=32 bm=32 bn=16 bk=16 |
| backward L1 d_input | (64,784,256) | None | - |
| backward L1 d_weight | (784,256,64) | None | - |
| backward L2 d_input | (64,256,10) | None | - |
| backward L2 d_weight | (256,10,64) | None | - |

**形状条件（`should_split_k`）だけでは forward 2 本は split-K 対象**だが、
**入口条件**（コード読みで裏取り。実装計画 §3.3 手順 1〜4）により
到達可否は fresh/reuse・層ごとに分かれる（訂正: PR #1531 レビュー
指摘〈Cursor Bugbot・codex-review、根拠
`crates/backend-metal/tests/splitk_train_shape_attribution.rs#L51-L55`〉
を受け、下記のとおり「forward はすべて非到達」という当初の記述を
訂正する）:

- **forward・reuse（`Sequential::forward_from_flat_leaves`。
  `crates/facade/src/compat/sequential.rs:606-613`）**: L1・L2 とも
  `store.linear_forward_with_activation` → `BackendOps::
  gemm_resident_rhs_act` へ委譲する。次層が `ReLU` でない L2 も
  `Activation::None` で同じ融合 strided prepared 専用入口を通るため
  `dispatch_auto` 非経由（`gemm_bias_act_route` が `Fused` を返す条件は
  `crates/backend-metal/src/ops.rs::gemm_bias_act_route` 参照）。
- **forward・fresh（`Sequential::forward`。`crates/facade/src/
  compat/sequential.rs:152-190`）**: 次層が `ReLU` の場合（L1）のみ
  `LinearVars::forward_with_activation`（epilogue 融合
  `gemm_bias_act`。`crates/autodiff/src/nn/linear.rs`）へ結線し
  `dispatch_auto` を経由しない。**次層が `ReLU` でない L2（出力層・
  fresh train ではここに該当）は非融合の `LinearVars::forward`
  （`matmul` → `add`。`crates/facade/src/compat/sequential.rs:185`・
  `crates/autodiff/src/nn/linear.rs::LinearVars::forward`）を使う**。
  この `matmul` は `MetalBackendOps::gemm`（`ops.rs:554`）→
  `layout::classify_2d` が NN・contiguous と判定 → `dispatch_auto`
  （split-K 判定を行う唯一の本番入口）へ到達する。形状
  `(BATCH,D_OUT,D_HIDDEN)=(64,10,256)` は `should_split_k` が `Some`
  （上表参照）のため、**結線後（ゲート `true`）は fresh train の L2
  forward が split-K 経路へ到達しうる**。
- **backward**: `matmul_vjp`（`crates/autodiff/src/grad.rs`）は
  `da = gemm(g, b^T)`・`db = gemm(a^T, g)` という形で必ず転置オペランド
  を含む GEMM を発行する。`MetalBackendOps::gemm`（`ops.rs:554`）は
  `layout::classify_2d` が転置 view を検出すると
  `gemm_strided_nt_tn`（`dispatch_strided_bias_act_prepared` 経由）へ
  分岐し、`dispatch_auto` を一切呼ばない（`ops.rs:561-567` で確認
  済み）。reuse 経路（`gemm_resident_lhs`）も同型の strided 専用入口。
  したがって backward は形状条件でも非到達（`should_split_k` が
  `None`）に加え、入口条件でも非到達という二重の理由で split-K に
  到達しない。

**結論（実測前の構造分析。訂正版）**: `gemm`（8 セル）は形状条件で
split-K に到達しない。`train` の reuse・backward（全形状）は入口条件
（backward は形状条件でも）で到達しないが、**`train` の fresh は L2
forward（非融合 `matmul`）が `dispatch_auto` を経由するため、結線後
（ゲート `true`）は split-K 経路へ到達しうる**。したがって本 A/B は
「gemm 8 セル＋train reuse 1 セル」については結線による本番既定経路の
非後退ガード（split-K 非到達のまま）であり、「train fresh 1 セル」は
split-K が実際に効く経路を含む非後退ガードとなる（=このセルに限り
split-K の性能効果自体も間接的に観測されうる。性能面の主根拠は
引き続き #1515・`docs/perf/metal-gemm-splitk-ab.md` を参照）。

補足（実装計画 §3.3 手順 5）: 上表の「backward L1 d_input」は
`docs/autodiff-nograd-leaf-dinput-skip-decision.md`（非学習葉への
d_input 伝播スキップ）の対象になりうる形状（`x` は学習対象でない葉）。
実際に計算がスキップされる場合、L1 d_input の GEMM 自体が発行されない
可能性があるが、いずれにせよ非到達という結論は変わらない。

## §3 事前登録判定規則（計測後に変更しない）

- **対象セル**: gemm `(N ∈ {512,1024,2048,4096}) × (fresh, reuse)` 8
  セル＋train `(64) × (fresh, reuse)` 2 セル。各セル before/after
  ちょうど 5 件（`compare_gemm_ab.py` の fail-closed 規則を継承。6 件
  以上・欠落は判定不能）。
- **指標**: `ratio = median(after median_s) / median(before median_s)`
  （実行時間比。小さいほど良い。TFLOPS 比と混同しない）。
- **非後退**: 全 10 セルで `ratio ≤ 1.00`（`--threshold 1.00` を明示。
  既定 1.05 は使わない）かつ checksum **bit 完全一致**（複合判定 pass
  のみは不可）かつ gemm は `parity_fail_count == 0`。
- **後退セルの扱い（結線維持可否の判定規則）**:
  - (a) checksum 不一致、または `ratio > 1.05` のセルが 1 つでもある →
    **結線維持不可**。`docs/backend-metal-splitk-decision.md` §5
    手順⑤（ゲート `false` へ差し戻し）を実施し理由を記録する。
  - (b) `1.00 < ratio ≤ 1.05` のセル → 「後退（共有負荷ノイズ帯・結線
    維持可）」として記録する。§2 の帰属表（split-K 非到達）と負荷推移
    （`monitor_<label>.log`）を原因欄に記す。**ただし**
    `compare_gemm_ab.py --per-run` が出力する run 単位ペア（append
    順 k 番目＝run k。5 run とも同一セルへ 1 行ずつ追記する計測スクリプト
    の契約に基づく）の run 内比（`after_k/before_k`）が **5/5 run すべて
    `> 1.00`**（符号一貫。`--per-run` 出力の「符号一貫」列が「はい」）
    である場合は、ノイズ帯ではなく一貫した後退とみなし (a) と同じ
    **結線維持不可**とする（機械出力で判定し、目視の「ノイズだろう」で
    (b) へ丸めない）。

- **記録する verdict は 3 値**: 「非後退（全 10 セル ≤ 1.00・checksum
  完全一致）」／「後退（ノイズ帯・結線維持可）」／「結線維持不可」。
  §0・§6 にこの 3 値で記入する（(b) を「非後退」へ丸めたり誤って差し
  戻したりしない）。
- **undetermined**: 件数不足・warmup/iters/version 不一致・スクリプト
  fail-closed 停止（差分ガード・sha256 変化）のみ。負荷が高いこと自体
  は undetermined の理由にしない（ルート #1509 の運用方針）。
- **`--phases` 表**は診断用（各腕 1 回・別ファイル）で判定に用いない。
- run の差し替え禁止。中断は `env_info.txt` に記録し、再実行は新しい
  label で行う。

## §4 実施手順（Mac）

`docs/perf/logs/metal-gemm-splitk-framework-compare-1517/README.md` を
参照。

## §5 記入欄（実測結果。未実測）

### §5.1 gemm 8 セル

| size/mode | before median | after median | after/before | checksum | 判定 |
|---|---|---|---|---|---|
| 512/fresh | - | - | - | - | 未実測 |
| 512/reuse | - | - | - | - | 未実測 |
| 1024/fresh | - | - | - | - | 未実測 |
| 1024/reuse | - | - | - | - | 未実測 |
| 2048/fresh | - | - | - | - | 未実測 |
| 2048/reuse | - | - | - | - | 未実測 |
| 4096/fresh | - | - | - | - | 未実測 |
| 4096/reuse | - | - | - | - | 未実測 |

### §5.2 train 2 セル

| size/mode | before median | after median | after/before | checksum | 判定 |
|---|---|---|---|---|---|
| 64/fresh | - | - | - | - | 未実測 |
| 64/reuse | - | - | - | - | 未実測 |

### §5.3 `--phases` 診断表（train・参考値。判定に用いない）

未実測。

### §5.4 負荷推移・env_info

`docs/perf/logs/metal-gemm-splitk-framework-compare-1517/env_info.txt`
を参照（未実測のためテンプレートのまま）。

## §6 結線維持可否の判定（実測後に記入）

- verdict: **未確定**
- 理由: -
- 対応（結線維持 or `docs/backend-metal-splitk-decision.md` §5 手順⑤
  への差し戻し）: -

## §7 スコープ外（out-of-scope-tracking.md に従い記録のみ・起票はユーザー
承認後）

- framework-compare へ K 支配的形状（例 `(64,64,4096)`）の gemm セルを
  追加して split-K の到達を実践規模で検出すること。
- facade への Metal split-K runtime トグル公開 API。
- 「未結線」等の既存 docs 記述の横断整合（#1518）。
- `SPLIT_K_DISPATCH_AUTO_PRODUCTION_ENABLED` の `true` 切替（Mac セッ
  ション。#1515 ADOPT 確定後・ドリフトテスト更新込み）。

## §8 参照

- `docs/backend-metal-splitk-decision.md`（§5 切替手順）
- `docs/perf/metal-gemm-splitk-ab.md`（#1515。split-K 単体の性能 A/B）
- `docs/perf/metal-gemm-splitk-shapes.md`（対象形状の劣化定量化）
- `scripts/bench/framework-compare/run_ab_splitk_metal.sh`
- `scripts/bench/framework-compare/compare_gemm_ab.py`（`--task
  {gemm,train}`）
- `crates/backend-metal/tests/splitk_train_shape_attribution.rs`
