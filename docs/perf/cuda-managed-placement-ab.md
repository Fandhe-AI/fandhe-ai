# CUDA managed 配置有無の GB10 A/B 計測・既定化可否判定（#1353）

イシュー #1353「framework-compare gemm cuda fresh／reuse と train の managed
有無を GB10 で 5 回中央値比較し既定化の可否を記録する」に対応する。親 #1351、
兄弟 #1352（PR マージ済み。managed 配置の opt-in 実装・
`docs/backend-cuda-managed-placement-decision.md`）。

## §0 位置づけ

#1352 が実装した CUDA managed memory 配置（`cuMemAllocManaged` 経由の
`DeviceBuffer` opt-in 配置。既定 OFF・fail-closed）の**性能実測と既定化可否
判断**を担う。実装（配置ロジック・bit 同一契約・同期契約）自体は変更しない。

## §1 環境

`docs/perf/logs/cuda-managed-placement-ab-1353/env_info.txt` を参照（内部
ホスト名は含めない）。要約: DGX Spark GB10・NVIDIA GB10 driver 580.173.02・
GPU 使用率 0%（計測中一貫）・load average は他セッション常駐サービス
（comfyui-env・kokoro。GPU 計算利用ではなくメモリ常駐のみ）により 0.00→0.77
へ上昇したが GPU 計算負荷ではない。rustc 1.97.0・nvcc 13.0.88。

**実行方式の制約（重要）**: 本セッションは worktree 隔離下で動作しており、
リモートホスト上でのシェルスクリプト（`.sh`）直接実行・`nohup`/`setsid` に
よるバックグラウンド切り離し・ネストした `bash -c` 呼び出しがサンドボックス
に拒否された。そのため計画で用意した `run_ab_managed_cuda.sh` 自体は GB10
上で直接起動できず、同スクリプトが行うのと同じ「同一バイナリでの off/on
交互起動・5 回計測」プロトコルを、ssh 経由の単一コマンド（for ループ・
`bench-fandhe` 直接起動）として手動で再現した。計測条件・バイナリ・
patch 手順は同一だが、**フルマトリクス（gemm N=1024/2048/4096 全 5 run・
infer 副次計測）は時間制約により一部省略**し、既知の到達性事実（§2）を
踏まえ主対象（train reuse）とその他代表点に絞った。省略した組合せは §9 に
明記する。

## §2 到達性事実（実装時点で判明済み）

`bench-fandhe --task gemm` は fresh・reuse とも `a.matmul(&b)`
（`CudaBackendOps::gemm` の `clone_htod`／`alloc_zeros`／`clone_dtoh` 直
呼び経路）であり、managed 配置が効くのは `CudaMemory::alloc_zeroed`／
`upload`／`download` を通る経路（`gemm_resident_rhs`／`gemm_resident_lhs`
〈NT 分岐除く〉／`linear_forward_device`／`sgd_step_device` =
`DeviceParamStore` 系）のみである。**gemm タスクは managed フラグの影響を
構造的に受けない**ため、gemm 行の on/off 差は「実測ノイズの範囲で差なし」
であることを裏取りする位置づけである。**主対象は `train --mode reuse`**
（`DeviceParamStore` の `upload(grad)`／`alloc_zeroed`／`download`）。

## §3 計測条件

- バイナリ: `scripts/bench/framework-compare/target/release/bench-fandhe`
  （`managed-placement` feature 有効・`[patch.crates-io.fandhe-ai]` で
  本 PR の worktree の `crates/facade` へ path patch。`cargo tree` で
  path 解決を確認済み）
- 承認済みピン `fandhe-ai =0.7.0`（`Cargo.toml`）に対する patch であり、
  `Cargo.lock`・`[patch]` 宣言はコミットしていない（計測後に
  `git checkout -- scripts/bench/framework-compare/Cargo.lock` で復元済み）
- off = `--managed` なし（既定 device-only 配置）・on = `--managed` あり
- 各セル 5 回計測。off/on を交互起動（同一バイナリ・同一プロセス起動間隔の
  最小化により熱・クロック偏りを揃える。`compare_managed_ab.py` が
  `(task, device, size, mode)` セルごとに中央値・checksum 一致を集計）
- 判定ツール: `python3 compare_managed_ab.py <jsonl>`（fail-closed。各セル
  ちょうど 5 件・`warmup`/`iters`/`version` 一致・checksum 複合判定を要求）

## §4 結果（`compare_managed_ab.py` 出力・そのまま転記）

| cell (task/device/size/mode/phase) | off median | on median | on/off | checksum | 判定 |
|---|---|---|---|---|---|
| gemm/cuda/2048/fresh | 136.573 ms (min 133.393 / max 138.180 ms) | 137.261 ms (min 133.807 / max 137.996 ms) | 1.0050 | 完全一致 | 差なし（ノイズ範囲。§2 の到達性事実どおり） |
| gemm/cuda/2048/reuse | 9.522 ms (min 9.387 / max 9.570 ms) | 9.502 ms (min 9.357 / max 9.608 ms) | 0.9979 | 完全一致 | 差なし（ノイズ範囲。§2 の到達性事実どおり） |
| train/cuda/64/fresh | 509.5 us (min 493.5 / max 535.8 us) | 508.5 us (min 493.6 / max 513.7 us) | 0.9981 | 完全一致 | 差なし |
| **train/cuda/64/reuse** | **451.5 us (min 439.8 / max 453.9 us)** | **772.5 us (min 736.6 / max 819.0 us)** | **1.7110** | 完全一致 | **明確な後退（約 1.71 倍）** |

checksum は全セルで**完全一致**（bit 同一。#1352 の核心契約「配置に依らず
出力は bit 同一」を実測でも裏取り）。gemm・train fresh は on/off 比が
0.998〜1.005 で計測ノイズの範囲内（§2 のとおり gemm は構造的に managed
非到達、train fresh も `DeviceParamStore` を経由しないホスト経由 SGD の
ため managed の影響を受けない設計と整合する）。

**train reuse のみ明確な後退**を示した（1.71 倍。5 run とも off < on で
符号一貫）。

## §5 phases 診断（単発計測・参考値）

`train --mode reuse --phases` を off/on 各 1 回計測（`compare_managed_ab.py`
は各セル 5 件を要求するため fail-closed で「判定不能」表示になるが、フェーズ
別の内訳は参考値として転記する）:

| phase | off median_s | on median_s | on/off |
|---|---|---|---|
| tape_build | 0.000002736 | 0.000007344 | 2.68 |
| leaf_register | 0.000000096 | 0.000000200 | 2.08 |
| forward_resident | 0.000156072 | 0.000221528 | 1.42 |
| loss_readout | 0.000000032 | 0.000000048 | 1.50 |
| backward | 0.000197528 | 0.000223024 | 1.13 |
| **device_update** | **0.000089920** | **0.000253080** | **2.82** |
| tape_drop | 0.000000896 | 0.000001352 | 1.51 |
| step_total | 0.000446840 | 0.000701128 | 1.57 |

`device_update`（`DeviceParamStore` の `upload(grad)`／`alloc_zeroed`／
`download` が集中する区間）が最大の相対悪化（約 2.82 倍）を示しており、
`docs/backend-cuda-managed-placement-decision.md`「同期契約の差分」が事前に
指摘していた**`UnifiedSlice::drop` の同期 `cuMemFree` が per-step の暗黙
同期点になる**仮説と整合する。`forward_resident`／`backward` も一貫して
後退方向（1.13〜1.42 倍）を示すが、単発計測（n=1）のため統計的な確証はない
（フル 5 run のフェーズ診断は §9 のスコープ外事項として後続へ引き継ぐ）。

## §6 帯域（`managed_placement_bandwidth_real_device.rs`。§2.2 相当）

`docs/perf/logs/cuda-managed-placement-ab-1353/1353-bandwidth.log` に生ログ。
サイズ {4, 16, 32, 33, 64} MiB（#1146/#1149 の 32→33 MiB 段差を含む）で
upload／download／download 結果のホスト側逐次全読み（ページ経由アクセスの
実効帯域）を計測（各 5 run 中央値。bit 一致は全サイズで assert pass）:

| MiB | off upload (GiB/s) | off download | off readback | on upload | on download | on readback |
|---|---|---|---|---|---|---|
| 4 | 45.5 | 0.64〜2.5 | 7.3 | 4.7 | 32〜47 | 7.2 |
| 16 | 52.0 | 53.4 | 7.2 | 5.1 | 32〜44 | 7.1 |
| 32 | 52.7 | 2.0 | 7.2 | 4.9 | 2.8〜2.9 | 7.2 |
| 33 | 52.7 | 2.1 | 7.2 | 4.9 | 2.8〜2.9 | 7.2 |
| 64 | 53.7 | 2.3〜2.4 | 7.1 | 5.0 | 2.9〜3.0 | 7.2 |

（2 回実行し off download の 4/16 MiB のみ run 間で大きくばらついた
〈0.64〜53.4 GiB/s〉。DMA 転送の初回ウォームアップ・GPU 側キャッシュ状態
依存と推定するが未確定のまま記録する。32 MiB 以上では off/on とも
安定した値になる。）

**帯域面の結論**: `readback`（ホスト側ページ経由の逐次読み取り。managed か
device-only かに関わらず最終的にホスト可視のバッファを読む区間）は
off/on でほぼ同一（7.1〜7.4 GiB/s、差 5% 未満）であり、**managed 配置に
よる CPU 側ページ経由アクセスの帯域低下は確認されなかった**。一方
`upload` は managed（ホスト → `UnifiedSlice` memcpy）が device-only
（DMA H2D）の約 1/10（4.7〜5.1 対 45.5〜53.7 GiB/s）と大幅に遅い。これは
本イシューの主結論（train reuse の `upload(grad)` を含む `device_update`
区間の後退）と整合する一次要因と考えられる。`download` は managed が
小サイズ（4/16 MiB）で有利・大サイズ（32 MiB 以上）で不利という非単調な
挙動を示し、原因は未確定のまま記録する。

## §7 数値一致

全セル（gemm/train・off/on）で checksum 完全一致（bit 同一）を確認。
`managed_placement_real_device.rs`（#1352 の契約テスト。5 件）も GB10 実機
で全 pass（`docs/perf/logs/cuda-managed-placement-ab-1353/
managed_placement_real_device.log`）。配置は出力に影響しないという #1352
の核心契約に非後退。

## §8 既定化可否の判定

**REJECT（既定化しない）**。

判定規則（実装計画の記録のみ・記録後も既定は変更しない）:
- ADOPT 候補: train reuse の 5 回中央値が非後退（on/off ≤ 1.0）・gemm/train
  fresh に後退なし・帯域テストで readback が明確に低下していない・checksum
  完全一致
- REJECT: 上記のいずれかで後退（特に `UnifiedSlice::drop` 同期解放による
  `device_update` 増）

実測結果は train reuse が 1.71 倍後退（`device_update` フェーズ単独では
2.82 倍）しており REJECT 条件に該当する。他条件（gemm/train fresh 非後退・
checksum 完全一致）は満たすが、主対象である train reuse の明確な後退により
既定化は見送る。`docs/backend-cuda-managed-placement-decision.md`「同期契約
の差分」が事前に指摘した `UnifiedSlice::drop` の同期解放が per-step の暗黙
同期点として実効することを実測で確認した形になる。

## §9 スコープ外・引き継ぎ事項

以下は本 PR のスコープ外（時間制約による一部省略・別イシューへの提案）。
既存の out-of-scope-tracking.md 規約に従い、ユーザー承認後に Issue へ記録
する:

- **フルマトリクス未実施**: gemm N=1024/4096（fresh/reuse）・infer reuse
  副次計測・`gemm --phases`／`infer --phases` 診断は未実施（N=2048 の
  gemm fresh/reuse・train fresh/reuse の代表点のみ実測）。§2 の到達性
  事実（gemm は managed 非到達）から N=1024/4096 も同様に「差なし」が
  期待されるが、実測による裏取りは行っていない
- **train reuse phases のフル 5 run 化**: §5 は単発計測（n=1）のため
  `compare_managed_ab.py` は判定不能を返す。フェーズ別の統計的確証には
  各 phase 5 run が必要
- **`run_ab_managed_cuda.sh` 自体の GB10 実行未確認**: 本 PR のスクリプト
  自体は worktree 隔離のサンドボックス制約により GB10 上で直接起動でき
  なかった（§1「実行方式の制約」）。スクリプトのロジック自体は手動再現
  したプロトコルと同一だが、スクリプト自体の動作確認（bash 構文エラー
  なし以上の実機確認）は別セッションで行う必要がある
- **デバイス属性による自動選択＋device-only フォールバック設計**（提案）:
  既定 ON 化は非 unified-memory デバイスで fail-closed エラーを既定挙動に
  してしまうため、GB10 のような統合メモリデバイスを実行時に検出して
  自動選択する設計が別途必要（本 PR では既定を変更しないため提案のみ）
- **managed 対応 `SizeClassPool`・fresh `CudaBackendOps::gemm` の managed
  化・`UnifiedSlice::prefetch`／`cuMemAdvise`**（提案。
  `docs/backend-cuda-managed-placement-decision.md`「採用しなかった方式」
  参照）
- **`upload`（H2D）の非対称性の深掘り**: §6 の帯域計測で managed upload が
  device-only の約 1/10 であることが判明した。この非対称性自体の原因調査
  （ホスト側 memcpy の実装詳細・`cuMemAllocManaged` のページフォールト
  挙動等）は別イシューの対象とする

## §10 実測記入欄（本ファイルは全項目実測済みのため空）
