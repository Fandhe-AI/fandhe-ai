# Metal split-K 結線前後 framework-compare A/B（イシュー #1517）

## §0 状態

**実測完了**（2026-09-11・M4 Max・共有負荷下・record_only）。

- verdict: **結線維持不可（false へ差し戻し）**
- 結線維持可否: **ゲート `SPLIT_K_DISPATCH_AUTO_PRODUCTION_ENABLED = false` を維持**（理由は §6 参照）
- 追記（2026-09-11・#1516 マージ）: 上記 verdict は事前登録規則どおりの確定記録として不変。その後ユーザーが保守性を主眼とする別根拠でゲート `true` への結線を決定した（`docs/backend-metal-splitk-decision.md` §5「ユーザー判断による本番結線」）。本 doc の判定規則・記録は書き換えない

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
- **非後退**: 全 10 セルで `ratio ≤ 1.00`（`--threshold 1.00` を明示）
  かつ checksum **bit 完全一致**（複合判定 pass のみは不可）かつ gemm
  は `parity_fail_count == 0`。
- **後退セルの扱い（結線維持可否の判定規則）**: `docs/backend-metal-
  splitk-decision.md` §5「ゲート既定値・切替条件」の手順④⑤（非後退・
  checksum 一致を確認できた場合のみ結線維持、後退時は `false` へ差し
  戻す）と同一の判定条件を用いる。誤差帯・符号一致数による救済は設け
  ない——checksum 不一致、または `ratio > 1.00` のセルが 1 つでもあれ
  ば**結線維持不可**とし、`docs/backend-metal-splitk-decision.md` §5
  手順⑤（ゲート `false` へ差し戻し）を実施して理由を記録する。§2 の
  帰属表（split-K 非到達）と負荷推移（`monitor_<label>.log`）は原因の
  参考情報として記す（判定基準そのものを緩めない）。

- **記録する verdict は 2 値**: 「非後退（全 10 セル ≤ 1.00・checksum
  完全一致・結線維持）」／「結線維持不可（`false` へ差し戻し）」。
  §0・§6 にこの 2 値で記入する。
- **undetermined**: 件数不足・warmup/iters/version 不一致・スクリプト
  fail-closed 停止（差分ガード・sha256 変化）のみ。負荷が高いこと自体
  は undetermined の理由にしない（ルート #1509 の運用方針）。
- **`--phases` 表**は診断用（各腕 1 回・別ファイル）で判定に用いない。
- run の差し替え禁止。中断は `env_info.txt` に記録し、再実行は新しい
  label で行う。

## §4 実施手順（Mac）

`docs/perf/logs/metal-gemm-splitk-framework-compare-1517/README.md` を
参照。

## §5 記入欄（実測結果）

### §5.1 gemm 8 セル

| size/mode | before median | after median | after/before | checksum | 判定 |
|---|---|---|---|---|---|
| 512/fresh | 656.8 us | 646.3 us | 0.9840 | 完全一致 | 非後退 |
| 512/reuse | 722.3 us | 721.7 us | 0.9992 | 完全一致 | 非後退 |
| 1024/fresh | 2.222 ms | 2.138 ms | 0.9622 | 完全一致 | 非後退 |
| 1024/reuse | 2.415 ms | 2.597 ms | 1.0752 | 完全一致 | 後退 |
| 2048/fresh | 8.504 ms | 8.712 ms | 1.0245 | 完全一致 | 後退 |
| 2048/reuse | 10.721 ms | 9.686 ms | 0.9035 | 完全一致 | 非後退 |
| 4096/fresh | 34.755 ms | 35.049 ms | 1.0085 | 完全一致 | 後退 |
| 4096/reuse | 40.047 ms | 39.501 ms | 0.9864 | 完全一致 | 非後退 |

### §5.2 train 2 セル

| size/mode | before median | after median | after/before | checksum | 判定 |
|---|---|---|---|---|---|
| 64/fresh | 1.519 ms | 1.511 ms | 0.9947 | 完全一致 | 非後退 |
| 64/reuse | 1.427 ms | 1.456 ms | 1.0207 | 完全一致 | 後退 |

### §5.3 `--phases` 診断表（train・参考値。判定に用いない）

`compare-train-1517-run1.md` に全文記録。要約:
- fresh: step_total 0.966 倍（forward 0.962・backward 0.989）
- reuse: step_total 0.990 倍（forward_resident 0.992・backward 0.994）

### §5.4 負荷推移・env_info

`docs/perf/logs/metal-gemm-splitk-framework-compare-1517/env_info.txt`
を参照。M4 Max・macOS 26.6.2・共有負荷下・開始前 uptime load averages
6.44 7.25 6.00・計測中 load1 min 6.32 / median 7.66 / max 11.56・終了時
11.56 8.33 6.55。watchlist 並走プロセス（python/torch/mlx/cargo/gemm_/
bench）件数は全て 0。中断・再試行なし。

## §6 結線維持可否の判定

- verdict: **結線維持不可（false へ差し戻し）**

- 理由: gemm 3 セル（1024/reuse・2048/fresh・4096/fresh）・train 1 セル
  （64/reuse）で `ratio > 1.00`。checksum は全 10 セル完全一致・gemm parity
  0 fail（§3 事前登録規則「`ratio ≤ 1.00` かつ checksum 完全一致 かつ
  gemm parity_fail_count == 0」の全条件不成立）。

- 帰属分析（参考情報・判定基準は緩めない）: 後退セルはいずれも §2
  帰属表上 split-K に構造的に非到達（gemm NN 正方は `should_split_k` が
  `None`・train reuse は入口条件で非到達）のため、before/after は同一
  カーネル経路の比較。5 run 内の比値は全後退セルで符号一貫性なし（run
  間で改善・後退が反転）。共有負荷下の計測ノイズと整合するが、規則（§3）
  では誤差帯・符号一致による救済を明示的に禁じているため判定は変えない。

- 対応: `docs/backend-metal-splitk-decision.md` §5 手順⑤（ゲート
  `false` へ差し戻し）を実施。理由を同ファイル「切替判断の記録」節へ
  記録（2026-09-11・ユーザー判断）。

- **規則改定なし・再計測条件**: 本 A/B の判定規則（§3）は変更しない。
  同一規則で新たに実測する場合も verdict 判定フロー（後退セル有→結線
  維持不可）は変わらない。規則自体を改定する（例: 到達セルのみ ratio
  判定）場合は、計測前に issue でユーザー承認が必要。

## §6a #1548 事後監視（実行時トグル・同一バイナリ A/B。2026-09-12）

#1544 のユーザー判断（既定 ON）に対する事後監視。#1546 の実行時トグル
（`bench-fandhe --metal-split-k on|off`・feature `metal-split-k-toggle`）を
使い、単一バイナリで before=`off`／after=`on` を run 単位 interleave 計測
（`scripts/bench/framework-compare/run_ab_splitk_metal.sh` commit a87aac69）。
**結果は記録のみ**。#1517 の判定・#1544 の決定は変更しない。

### 計測結果（run2・低負荷・正式値）

**gemm 8 セル（N=512/1024/2048/4096 × fresh/reuse）**

| size/mode | before median | after median | after/before | checksum | 判定 |
|---|---|---|---|---|---|
| 512/fresh | 648.3 us | 654.5 us | 1.0096 | 完全一致 | 後退 |
| 512/reuse | 728.5 us | 728.6 us | 1.0002 | 完全一致 | 後退 |
| 1024/fresh | 2.385 ms | 2.429 ms | 1.0184 | 完全一致 | 後退 |
| 1024/reuse | 2.884 ms | 2.900 ms | 1.0055 | 完全一致 | 後退 |
| 2048/fresh | 8.426 ms | 7.955 ms | 0.9442 | 完全一致 | 非後退 |
| 2048/reuse | 9.353 ms | 10.122 ms | 1.0822 | 完全一致 | 後退 |
| 4096/fresh | 34.444 ms | 33.845 ms | 0.9826 | 完全一致 | 非後退 |
| 4096/reuse | 39.915 ms | 47.695 ms | 1.1949 | 完全一致 | 後退 |

**train 2 セル（fresh/reuse）**

| size/mode | before median | after median | after/before | checksum | 判定 |
|---|---|---|---|---|---|
| 64/fresh | 1.646 ms | 1.562 ms | 0.9493 | 完全一致 | 非後退 |
| 64/reuse | 1.426 ms | 1.433 ms | 1.0049 | 完全一致 | 後退 |

**`--phases` 診断表（train・参考値。非判定）**

| phase | fresh | reuse |
|---|---|---|
| step_total 比 | 1.053 | 1.282 |

### 所見

**規則判定**（§3 事前登録規則に従う）:

- 規則 1 超過セル: gemm 6 セル（512/fresh・512/reuse・1024/fresh・1024/reuse・
  2048/reuse・4096/reuse）+ train 1 セル（64/reuse）= **7 セル**
- 規則 1 符号一貫セル（全 5 run が `ratio > 1.00`）: **0 件**
- 規則 2（checksum）: 全 10 セル完全一致

**評価**:

上記 7 セルの後退は、いずれも §2 帰属表上 split-K に**構造的に非到達**
（gemm NN 正方は `should_split_k` が `None`・train reuse は入口条件で非到達）
のため、before/after は同一カーネル経路の比較・符号一貫性なし・共有負荷下
（run1 時点は load1 3.58→17.56→18.12→16.98）から run2 で低負荷に低下
（load1 5.02→4.03→3.75→3.69）した計測ノイズと整合する。ノイズ帯であり、
結線の有無で差が生じない。checksum は全一致・gemm は split-K 非到達のため
parity ゼロ fail（期待値）。

**判定: 記録のみ・決定不変。** #1544 のユーザー判断（既定 ON）を変更する
入力にはしない。

### ログ参照先

- `docs/perf/logs/metal-gemm-splitk-framework-compare-1548/`

## §7 スコープ外（out-of-scope-tracking.md に従い記録のみ・起票はユーザー
承認後）

- framework-compare へ K 支配的形状（例 `(64,64,4096)`）の gemm セルを
  追加して split-K の到達を実践規模で検出すること。
- facade への Metal split-K runtime トグル公開 API。
- 「未結線」等の既存 docs 記述の横断整合（#1518。**完了**）。
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
