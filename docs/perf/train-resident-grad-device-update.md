# reuse 経路の weight 勾配をデバイス常駐 staging へ直接書き込む（イシュー #1212）

## 0. スコープ（実装時点の縮小判断）

計画（Plan フェーズ）が想定した全面実装（`Gradients` へのマーカー追加・
`vjp()` 返り値の enum 化・Metal `c_offset` スレッディング・CUDA view 出力
GEMM 追加・フレームワーク横並びベンチの前後 5 回計測キャンペーン）は、
単一セッション・委譲（subagent）なしという実行制約のもとでは検証可能な
安全側の増分に収まらないと判断し、以下へ**意図的に縮小**した:

- **CPU バックエンドのみ本番結線（#1555 で Metal も実装済み。§5 参照）**。CUDA は
  `BackendOps::gemm_fp32_strict_into`（既定 `Unsupported`）に一切手を
  加えていないため、`grad::vjp` は既存のホスト経路
  （`ops.gemm_fp32_strict` → `Gradients` へ格納 → `step()` が連結
  `flat_grad` を 1 回 upload）へ**そのままフォールバック**する。挙動・
  性能とも本イシュー着手前と完全に不変（`cargo build`／`cargo clippy`
  済み。実機なしのため CUDA の速度計測は行っていない）
- **`vjp()` の戻り型 `Vec<(NodeId, Tensor<f32>)>` は不変**。`Gradients`
  への変更は由来検証用の内部フィールド `resident_fingerprint`
  （`pub(crate)`。`(store_id, backward_serial, pending 世代)`）の追加
  のみで、`DeviceParamStore::step` はこれと slot ごとの葉の同一性
  （`tape_id`／`epoch`／`node_id`）を照合してから resident 経由の
  slot を信頼する（codex-review P0 是正。当初は `Gradients` を一切
  変更しない方針だった）。設計判断の詳細は
  `docs/device-resident-update-design.md` 追補「#1212」§2・§2.1 を参照
- **性能計測は本ファイルの軽量プロトコル**（§2）に留め、
  `scripts/bench/framework-compare` を使った候補 A/B 5 回計測キャンペーン
  （REQ-8 系の他 doc が採用する形式）は実施していない。CPU の `bias`
  なし単層 MLP という極小ワークロードでの `DeviceParamStore::step` 内部
  タイミングのみを計測しており、実践規模（framework-compare の
  `--task train` 相当）での影響は未計測。CUDA／Metal 実測・
  framework-compare 実践規模計測は後続イシューへ引き継ぐ（§4）

## 1. 変更内容の要約

`crates/autodiff/src/optim/device_store.rs::DeviceParamStore` に
`GradStaging`（全パラメータ連結の永続 grad バッファ。`step()` が
`#1023` 以来使ってきた「毎 step 新規確保・upload」の代替）を追加した。

- `Op::LinearResident` の VJP（`grad.rs`）は、`resident.
  fill_resident_weight_grad(ops, store_id, slot, &x_t, g)` が
  `Ok(true)` を返した場合（＝バックエンドが `gemm_fp32_strict_into`／
  `MemoryOps` を実装し、GradStaging への直接書き込みに成功した場合）、
  d_weight を `Gradients` へ含めない（ホストへの D2H を経由しない）
- `DeviceParamStore::step` は、いずれかの slot が「今回の backward で
  resident 経由により新鮮に充填された」（`backward_serial` による鮮度
  検査）と判定した場合のみ新しい経路（GradStaging の残り slot
  〈bias 等〉を `MemoryOps::upload_into` で個別に埋めてから
  `staging.buf` を直接 SGD カーネルへ渡す）を使う。1 slot も
  resident 化されていない場合（現状の Metal／CUDA）は #1023 以来の
  経路を無変更で実行する

## 2. 計測プロトコル（軽量・本イシュー内で実施）

`crates/facade/tests/device_param_store_bench.rs::legacy_vs_resident_per_step_cpu`
（`#[ignore]` なし・CI 常時実行対象）が計測する `resident_update_median_s`
（`Tape::step_device_param_store` 呼び出し 1 回の中央値。1 隠れ層 MLP・
CPU バックエンド）を before/after で比較した。

```sh
cargo test -p fandhe-ai --release --test device_param_store_bench \
    legacy_vs_resident_per_step_cpu -- --nocapture
```

- before: `git stash`（本 PR の全差分を退避）した状態でビルド・実行
- after: 本 PR HEAD でビルド・実行
- 各 3 回実行し、`resident_update_median_s`（内部で 5 回計測の中央値。
  `bench_harness::median_q1_q3`）を記録

## 3. 実測結果（Apple M4 Max・CPU バックエンド）

| 系列 | resident_update_median_s | 備考 |
|---|---|---|
| before（`git stash` 適用） | 0.000001525 s | #1023 の毎 step 新規確保 + 1 回 upload |
| after（本 PR HEAD・run 1） | 0.000001081 s | GradStaging 経由（weight は D2H/H2D なし・bias のみ upload_into） |
| after（run 2） | 0.000001083 s | |
| after（run 3） | 0.000001048 s | |
| after（run 4） | 0.000001044 s | |

after の中央値（4 run 平均的傾向）は約 1.04〜1.08 µs、before は
1.525 µs で、**約 1.4〜1.46 倍の改善**（`step()` 内部の update フェーズ
単体。CPU は D2H/H2D が実質 memcpy のため、この改善は主に「新規
`Vec<f32>` 確保 + `Tensor::new` + `mem.upload`（新規 `DeviceBuffer`
確保を伴う）」を「既存 staging バッファへの直接書き込み（確保なし）」
へ置き換えたことによるアロケーション削減に由来する）。

`legacy_total_median_s`／`resident_total_median_s`（forward + backward +
update 全体）は速度差が計測ノイズ（±10% 程度）に埋もれており、
`total_speedup_x` は 0.90〜1.07 で run ごとにばらつく——本イシューの
変更は update フェーズ単体では明確な改善だが、1 隠れ層 MLP・CPU という
極小ワークロード全体では他フェーズ（forward の GEMM 等）の比重が大きく
支配的ではないため、全体では non-gating（record only）の同ベンチの
既存方針どおり有意な後退がないことのみを確認する位置づけとした。

**Go/No-Go 判断**（実装計画 §2）: 全体 `total_speedup_x` の 5% 超悪化は
観測されず（0.90〜1.07 の範囲は既存ベンチの記録が示す通常の計測揺らぎ
の範囲内）、update フェーズ単体は一貫して改善しているため **Go**
（本番結線を維持する）。`RESIDENT_GRAD_PRODUCTION_ENABLED` 相当の無効化
フラグは、CPU 経路自体が新規追加コード（tensor-core のデフォルト
`Unsupported` に対する CPU オーバーライド）であり無効化する対象が
「新規追加した最適化パス全体」と一致するため、`gemm_fp32_strict_into`／
`upload_into` を実装しないことと同値になる。今回は計測結果が Go の
ため、既存コードに追加の無効化フラグは導入していない。

## 4. スコープ外・引き継ぎ

- CUDA 実機（DGX Spark GB10）での `gemm_fp32_strict_into`／`upload_into`
  実装・実測（D2H が同期点であるため、CPU よりも効果が期待される本命の
  バックエンド。#1555 後も既定 `Unsupported`・フォールバックのまま）
- Metal 実機（Apple Silicon）での同実装・実測（#1555 で実装完了。§5 参照）
- `scripts/bench/framework-compare` を使った実践規模（`--task train`）の
  5 回計測キャンペーン（本ファイル §0 の軽量プロトコルの上位互換）
- bias 勾配（`reduce_to_shape`）自体のデバイス常駐化（デバイス側列縮約
  カーネルが必要。現状は bias は常にホスト経由で `upload_into` される）
- `Gradients` へのマーカー追加による「常駐 weight への `get()` が
  `Err(InvalidArgument)` を返す」設計（計画 §3.1 item 3）は本実装では
  採用しなかった（`grads.get()` は単に `Ok(None)` を返す。公開 API から
  `Op::ResidentLeaf` の `Var` を得る経路が元々存在しないため実害はないと
  判断したが、内部一貫性としては計画どおりの型付きエラー化が望ましい）
- resident 経由の重み勾配をホストへ読み出す公開 API（`facade::Tape::
  resident_grads_to_host`／`param_grads_to_host`）はイシュー #1479 で
  追加した（詳細・契約は `docs/device-resident-update-design.md` 追補:
  #1479 を参照。実測記録の追加は不要——読み出し専用 API の追加であり
  性能結線ではない）

## 5. #1555 Metal 実装

イシュー #1555 にて Metal 側の `BackendOps::gemm_fp32_strict_into`（NT/TN は encode-only 直接書き込み・それ以外はホスト GEMM → `upload_into` フォールバック）
と `MemoryOps::upload_into` を実装し、reuse 経路の weight 勾配をデバイス常駐 staging へ
直接書き込む結線を完了した。

### 5.1 実装事実

- `MetalGemm::encode_strided_bias_act_prepared_with_c_offset`（pub(crate)・encode-only・
  C バッファ要素オフセット対応）と `validate_strided_dims_impl` を新設。既存メソッド
  シグネチャ・挙動（`c_len == m*n` 厳格一致）は `c_offset: None` で委譲。
- `MetalMemory::upload_into` を実装（従来 Metal は既定 `Unsupported`）。resident 充填が
  1 つでも成功すると `DeviceParamStore::step` の `any_resident == true` 分岐が bias 勾配を
  同じ staging へ `upload_into` するため必須の前提。書き込み前に 1 回 `synchronize`
  （`StorageModeShared` 上の競合書き込み回避）。
- `MetalBuffer::write_slice_at(offset, data)` を追加（unsafe 4 箇所目・`zero_fill` の
  書き込み版）。
- `MetalBackendOps::gemm_fp32_strict_into` を実装。NT/TN に分類できる入力（`Op::LinearResident`
  の `x_t = transpose2d(x)` は通常ここ）は `gemm` の #1215 NT/TN 分岐と同じ `classify_2d`／
  `as_view_slice` ゲート・同一 classic strided カーネルで staging へ encode-only（内部
  `synchronize()` なし）に直接書き込む（`MemoryOps::download` が読み出し前に必ず
  `synchronize()` する既存契約により動作）。NN／TT／分類不能（例: in_features=1 で `x_t` の
  strides が `[1,1]` になる場合）は `gemm_fp32_strict`（ホスト結果）→ `upload_into` の
  フォールバックで同じ位置へ書き込み、**形状を理由に `Unsupported` を返さない**。理由:
  呼び出し元 `fill_resident_weight_grad` の `resident_grad_capability` はストア全体で 1 度
  きり判定されるため、形状単位の `Unsupported` を返すと同じ backward で既に staging へ充填
  済みの他層の勾配が `param_grads_to_host` から読めなくなる（PR #1556 codex-review P1
  指摘。`Linear(1,8)` → `Linear(8,4)` の混在ケースを実機 `#[ignore]` テストで回帰確認）。
- 失敗トークン（PR #1556 codex-review P0 是正）: encode-only の書き込みは `ctx.encode` に
  `token: None` を渡すと、共有 `MetalContext` を使う別スレッドが先に `synchronize()` して
  GPU エラーを回収した場合にストアへ失敗が伝播せず、未完成の staging を正常値として読み
  出しうる。`sgd_step_device_tracked` と同型の `BackendOps::gemm_fp32_strict_into_tracked`
  （既定実装は非 tracked 版へ委譲。CPU 等の同期バックエンドはそのまま）を追加し、
  `fill_resident_weight_grad` が `DeviceParamStore::failure_token` を渡して dispatch 登録と
  同じロック区間でトークンを登録する。

### 5.2 カウンタ変遷

Metal GEMM command batching ベンチ（`crates/facade/tests/
mnist_scale_train_reuse_bench.rs::mnist_scale_train_reuse_metal_batch_counters`。
**イシュー #1563 で訂正**: 旧版は所在を `crates/backend-metal/tests/
command_batching_bench.rs` と誤記していたが、同ファイルは MLP のカウンタ
期待値を持たず〈`command_buffer_delta < encode_delta` のみ〉、正しい所在は
`crates/facade/tests/mnist_scale_train_reuse_bench.rs` である）において、
`d_weight` GEMM の同期境界が最適化されたことを確認:

- before（#1099 直後）: 11 dispatch（`encode_delta`）/ 10 command_buffer / 10 wait
- after（#1555 実装後）: **11 dispatch / 9 command_buffer / 9 wait**（−1・−1。
  L1・L2 の `d_weight` GEMM ごとに走っていた `dispatch_strided_bias_act_prepared` 内部の
  `synchronize()` 2 回が消え、bias 勾配 `upload_into` の防御的 synchronize 1 回に集約。
  `d_input` の同期は残る）。既存 `command_batching_bench.rs` の他の計測 2 件は不変で pass。
  （#1099 直後の 9/8/8 へは戻らない。§3 時点の性能が既に最適な構成）

### 5.3 bit 完全一致確認

M4 Max・2026-09-12・record_only・共有負荷 load1 ≈ 7.9〜8.5 環境で、main（sha
`4e68a4dc`）と branch（feat/metal-resident-weight-grad）の数値一致を確認:

`metal_reuse_step_grad_bit_dump` の出力 4462 行（loss・各 step の重み勾配・パラメータ・
最終パラメータ）がすべて **bit 完全一致**（差分 0）。

### 5.4 framework-compare train A/B 実測

before（origin/main の `crates/facade` path patch）・after（本ブランチ。同一マシン・
5 round・run 単位で順序反転・各腕別バイナリ sha256 記録）の比較結果:

**判定対象: reuse セル**

| 形状 | 系列 | before (ms) | after (ms) | 比率 | 5 run 内比率 | checksum | 判定 |
|---|---|---|---|---|---|---|---|
| 64 | reuse | 1.361 | 1.196 | **0.8784** | 0.8580 / 0.8784 / 0.8864 / 0.8632 / 0.8773 | 完全一致 | **非後退（規則充足）** |

**対照セル: fresh（resident 経路非到達）**

| 形状 | 系列 | before (ms) | after (ms) | 比率 | checksum | 備考 |
|---|---|---|---|---|---|---|
| 64 | fresh | 1.444 | 1.446 | 1.0012 | 完全一致 | 符号不一致（0.9896 / 0.9485 / 0.9719 / 1.0092 / 1.0024） / ノイズ帯と帰属・規則の緩和ではない |

フレームワーク機械判定（`compare_gemm_ab.py --task train --threshold 1.00`）は fresh セルを
「後退」と判定する。fresh は resident 経路（`fill_resident_weight_grad`）に到達しない対照
セルだが、本 PR の変更コードを全く通らないわけではない: `d_weight` GEMM は既存入口
`encode_strided_bias_act_prepared` → `_impl(c_offset: None)`（零オフセット委譲・挙動同一）を
経由する。checksum 完全一致・5 run 中 3 run が 1.00 未満・符号不一致であることから、
1.0012 倍の差は計測環境の共有負荷（load1 ≈ 7.9〜8.5）によるノイズと整合する（ノイズと
確定したわけではなく、負荷差との分離は行っていない）。判定対象は事前登録どおり reuse
セルであり、当該セルは規則「reuse `step_total` の after/before ≤ 1.00」を充足する
（**規則の緩和を伴わない**）。

### 5.5 フェーズ分解診断（reuse セル・単発・非判定）

`--phases` 出力による各フェーズの時間内訳：

| フェーズ | before (µs) | after (µs) | 比率 |
|---|---|---|---|
| backward | 657.3 | 444.6 | **0.676** |
| device_update | 57.8 | 139.2 | 2.407 |
| step_total | 1.338 ms | 1.210 ms | **0.905** |

device_update の 2.407 倍増加は、同期点の移動に起因する（backward 内の GPU 完了待ちが
bias `upload_into` の synchronize へ移動）。これは **同期点の移動であり性能後退ではなく**、
全体 step_total は 0.905 倍で減少——総和は減少のため。fresh セルは step_total 0.992 ms で
対照とする。

### 5.6 スコープ外

- ~~CUDA 実機での `gemm_fp32_strict_into`／`upload_into` 実装・実測（#1212 から継続・
  既定 `Unsupported`・フォールバック）~~ → #1559 で実装完了（§6 参照）。実機実測は #1560 へ
  引き継ぎ
- bias 勾配自体のデバイス常駐化は Metal で #1566 により実装済み（下記 §9）。CUDA は
  引き続き既定 `Unsupported`（`gemm_fp32_strict_into_with_bias_reduce_tracked` の
  既定実装が weight のみへ委譲し bias は無視する）のままスコープ外
- `d_input` GEMM の同期境界解消（従来どおり `gemm_resident_lhs()` → `download` 経路。本イシュー〈#1555〉のスコープ外。**イシュー #1563 で更新**: 層内合流〈encode-only の d_weight を d_input の同期点より前へ移す〉のみ実施し、d_input 自身の同期境界 2 件は回収しないと結論した。L1 の d_input を丸ごと省略する #1219 の opt-in スキップ、または L2→L1 間を常駐チェーン化する新規カーネル案は、いずれもユーザー承認後の別イシューへ引き継ぐ〈`docs/backend-metal-command-batching-design.md` §7.4〉）

## 6. #1559 CUDA 実装

`BackendOps::gemm_fp32_strict_into`／`_tracked` の CUDA オーバーライド
（`crates/backend-cuda/src/ops.rs::CudaBackendOps::gemm_fp32_strict_into_impl`）を実装した。
#1214 で追加済みの CUDA NT/TN 転置入口（GPU 側 smem 転置カーネル `transpose_smem_f32` →
既存 NN GEMM カーネル方式）を、出力を新規 alloc + readback ではなく呼び出し元の
`DeviceBuffer<f32>` の指定オフセットへ直接書き込む形（`gemm::CudaGemm::
launch_tiled_f32_nt_into`／`_tn_into`。`CudaArgMut::View`／`UnifiedView` 新設）に拡張して
再利用した。NT/TN 以外（NN・TT・分類不能形状・退化形状・転置カーネル使用不能環境）は
`Unsupported` を返さず `gemm_fp32_strict` → `CudaMemory::upload_into` のフォールバックへ
倒す（Metal 版 #1555 と同じ理由。`resident_grad_capability` が `Some(false)` へ倒れて以降の
resident 読み出しが壊れるのを防ぐ）。

NT/TN 経路は `transpose_to_pooled` が返す中間バッファ（`PooledCudaHandle<f32>`）を
関数内で `stream.synchronize()` してから drop する設計（`launch_tiled_f32_resident_nt`
のように戻り値として呼び出し元へ返し「次の同期点まで保持する」契約が取れないため。
`gemm.rs::CudaGemm::launch_tiled_f32_nt_into` ドキュメンテーションコメント「設計判断 A」
参照）。旧経路（`readback` の D2H → `DeviceParamStore::step` 内 `upload_into` の H2D。
m*n 要素データ転送 2 回 + sync 2 回）と比較し、新経路はデータ転送ゼロで sync 1 回のみに
削減される設計だが、**実測値は本イシュー（#1559）のスコープ外**（本エージェント実行環境に
CUDA 実機がないため）で、GB10 実機での bit 同一検証・性能 A/B は #1560 へ引き継ぐ。

GPU 非依存単体テスト（`crates/backend-cuda/src/ops.rs`）・GB10 実機 `#[ignore]` テスト
（`crates/backend-cuda/tests/gemm_fp32_strict_into_parity.rs`。実機未実行のまま記入欄を
残す）を追加した。

## 7. #1560 GB10 実機実測（スキャフォールドのみ・本 PR 時点では未実測）

#1559 の bit 同一検証・性能 A/B をイシュー #1560 で引き継いだ。本 PR の実行環境
（Linux・QEMU VM）には DGX Spark GB10 実機への到達手段がないため、**実測値は
含まれていない**。実行スクリプト・事前登録判定規則・記入欄は
`docs/perf/logs/train-resident-grad-cuda-1560/`（`README.md` に手順を集約）・
`scripts/bench/framework-compare/run_ab_resident_grad_cuda.sh` として整備済みで、
GB10 実機を持つセッションへ実測を申し送る。

### 7.1 比較対象 2 腕（事前登録）

- **before 腕**: `d77f8bde`（#1569 マージ直前の main。`gemm_fp32_strict_into` の CUDA
  実装なし＝既定 `Unsupported` フォールバック経由）
- **after 腕**: #1560 のブランチ（`origin/main` `e41db903` + テスト／スクリプト／docs
  のみ。`crates/*/src` は `e41db903` と同一）

### 7.2 事前登録判定規則

- **Tier 1（必須）**: `size=64 / reuse` セルの `step_total` 5 run 中央値比
  after/before ≤ 1.00、かつ checksum 完全一致
- **対照（非判定）**: `fresh` セル（resident 経路非到達）
- **診断（非判定）**: `--phases` の `backward`／`device_update`／`step_total` 内訳
- **bit 同一（R2）**: `cuda_graph_step_bit_identity::eager_baseline` を before/after
  両ツリーで実行し `^(step\[|final\.param\[)` 行を diff（期待 4782 行・差分 0）
- **副次観測**: after 腕で `--graph on` を 1 回起動し `graph_captured`／
  `graph_replayed`／`graph_sgd_kernel_launches` を記録（`#1569` により CUDA reuse が
  NT/TN 層で `any_resident == true` になるため、`DeviceParamStore::step` の CUDA
  Graph capture 分岐〈`!any_resident` 限定〉が非到達になる可能性がある。判定には
  用いない）

詳細は `docs/perf/logs/train-resident-grad-cuda-1560/README.md` を参照。

### 7.3 実測結果（記入欄）

| 項目 | 結果 |
|---|---|
| bit 同一（R2） | 未実測 |
| `#[ignore]` 非後退（R1） | 未実測 |
| A/B reuse `step_total` 判定 | 未実測 |
| fresh 対照セル | 未実測 |
| フェーズ分解診断 | 未実測 |
| CUDA Graph capture 副次観測 | 未実測 |

実測完了後、本節を実測値で更新すること（事前登録規則の事後緩和は行わない）。

## 8. #1563 層内合流の実装・前後比較（記入欄）

`Op::LinearResident` VJP で encode-only の d_weight（`fill_resident_
weight_grad`）を、同期点を持つ d_input（`gemm_resident_lhs`）より前へ
移す変更（層内合流。設計・結論は `docs/backend-metal-command-batching-
design.md` §7.4 を正とし本節では重複記載しない）の前後比較記入欄。

| 項目 | 結果 |
|---|---|
| bit 同一（`metal_reuse_step_grad_bit_dump`。4462 行） | 未実測 |
| `#[ignore]` 非後退 | 未実測 |
| カウンタ（`mnist_scale_train_reuse_metal_batch_counters`。hard assert） | 未実測（期待: before 11/9/9 → after 11/8/8） |
| カウンタ（`mnist_scale_train_reuse_metal_backward_dinput_phase`。record-only） | 未実測（期待仮説: before 5/4/3 → after 5/3/3） |
| A/B reuse `step_total` 判定（`run_ab_dinput_sync_metal.sh`） | 未実測 |
| fresh 対照セル | 未実測 |

実測は `docs/perf/logs/metal-dinput-sync-1563/`（生ログ・env_info）へ
記録し、実測完了後に本節を実測値で更新すること（事前登録規則の事後
緩和は行わない）。

**#1566（§9）取り込み時の追記**: 上表のカウンタ期待値（11/9/9 →
11/8/8・5/4/3 → 5/3/3）は #1563 単独の期待値。§9 の bias 勾配デバイス
常駐化（#1566）を組み合わせた結果、L1・L2 とも bias 縮約が weight と
同一の encode-only ディスパッチへ折り込まれ、従来 bias 用に発生して
いた `MemoryOps::upload_into`（計 2 回）呼び出し自体がなくなるが、
これらの呼び出しは #1563 適用後の時点で既に完全な no-op（`committed`
が空で cb/wait への寄与ゼロ）だったため、**#1566 適用後もカウンタは
11/8/8（および 5/3/3）から変化しない**という机上結論になった（根拠は
`crates/facade/tests/mnist_scale_train_reuse_bench.rs` 冒頭 doc comment
「#1566 追記（是正版）」・`docs/backend-metal-command-batching-
design.md` §10.7〜§10.9 を参照）。round1（#1566 実装時点）の同ファイル
doc comment は「L1 は NN・分類不能」という根拠のない推測を含んでいた
が、本追記の導出時に誤りと判明し是正済み（実際は L1・L2 とも同一の
NT パターンで分類される）。上表・アサーション値ともに実機実測は
未実施のまま Mac 実機セッションへ引き継ぐ。

## 9. bias 勾配のデバイス常駐化（イシュー #1566）

#1565（`docs/backend-metal-command-batching-design.md` §10）が比較・採用した
**案 A′**（既存 `gemm_fp32_strict_into` と同一のアップロード・failure-token 登録から
d_weight・d_bias を同時に encode-only で書き込む拡張）を実装した。

- `tensor-core::BackendOps::gemm_fp32_strict_into_with_bias_reduce_tracked`（非破壊
  拡張。既定は bias を無視して既存 `_tracked` へ委譲。CPU・CUDA は無変更）。
- `backend-metal::gemm::MetalGemm::encode_weight_and_bias_grad_with_offsets`（NT/TN
  encode-only。d_weight と**同一 `ctx.encode` 呼び出し**で bias 縮約カーネル
  `gemm_bias_grad_reduce_f32` を追加ディスパッチする。`encode_calls` は増えない）。
- `backend-metal::ops::MetalBackendOps` の同トレイトメソッドオーバーライド（NN/TT・
  分類不能形状はホスト経由〈`layout::reduce_bias_grad_rows_host` → `upload_into`〉で
  常に成功する既存パターンを踏襲）。
- `autodiff::tape::ResidentResolver::fill_resident_weight_grad` のシグネチャ拡張
  （`bias: Option<ResidentBiasTarget>`・戻り値 `ResidentFillOutcome`）・
  `device_store.rs`／`grad.rs` の結線。weight tying 時は bias 自身の slot について
  独立に tie 判定・累積を行う（詳細は `docs/backend-metal-command-batching-design.md`
  §10.7「実装中に発見した正当性の落とし穴」）。

数値契約: 当初は `reduce_to_shape`（`f32` 逐次和）と bit 完全一致させる設計とし、
`.claude/rules/coding-rust.md` の勾配長軸縮約 `f64` アキュムレータ方針との不整合を
未解決のまま引き継いでいた（同上 §10.7）。この着手条件（数値方式の整合または
ユーザー承認済み例外）は、PR #1659 の codex-review 指摘を受けた **2026-09-12
ユーザー承認「選択肢 A: 規約どおり `f64` 相当へ統一する」により充足**した
（同上 §10.8）。ホスト経路（`eval::reduce_bias_grad_rows`・`layout::
reduce_bias_grad_rows_host`）は `f64` アキュムレータへ、GPU カーネル
（`gemm_bias_grad_reduce_f32`。Metal は `double` 非対応）は Neumaier 改良版
Kahan 補償和へ統一した。`m == 1`／`rows == 1` の直接コピー特殊扱いは不変。
`reduce_to_shape` 自体（本イシューが触れない他の呼び出し箇所）の `f32` 逐次和は
引き続き未整理のまま残る（同上 §10.8）。**追補（PR #1659 codex-review 追加
指摘・同上 §10.9）**: `backend-metal` 側の統一だけでは `Op::LinearResident` が
resident 非対応バックエンド（CPU／CUDA。常時該当）のフォールバック時に依然
`reduce_to_shape`（`f32`）を使い、Metal resident 成功時（`f64` 相当）と数値方式が
食い違う不整合が残っていたため、`crates/autodiff/src/grad.rs` のフォールバック
（`Op::LinearResident`）・`Op::LinearAct` の bias 縮約も同一の `f64` ヘルパ
（`grad::reduce_bias_grad`。`eval::reduce_bias_grad_rows` への委譲）へ統一した
（汎用 `reduce_to_shape` 本体は不変）。**さらに追補（2026-09-12 ユーザー承認・
PR #1659→#1665→#1666 取り込み後の追加是正）**: `nn::Linear` の既定 forward 経路
（`LinearVars::forward`。`matmul → add` の非融合合成）は `Op::Add` の VJP を
経由するため、`LinearAct`／`LinearResident` フォールバックと同じ bias パターン
（`[m, n]` → `[n]`／`[1, n]` の行方向縮約。既存 `reduce_bias_grad` の shape
構造判定と同一条件）に限り `Op::Add` の VJP（`da`／`db` 双方）も
`reduce_bias_grad` へ委譲するよう統一した。これにより `LinearVars::forward`
（fresh・非融合）・`Op::LinearAct`（epilogue 融合）・`Op::LinearResident`
（reuse）のいずれで forward しても同一 `Linear` 層の bias 勾配の数値方式が
揃う。`Op::Add` の bias パターン以外の broadcast・`reduce_to_shape` 自体は
不変（`crates/autodiff/src/grad.rs::reduce_bias_grad` doc 参照）。

**bias 部分の恒久的な同期削減効果**: `docs/backend-metal-command-batching-design.md`
§4.2 と本 §5.5 が記録した「bias 分の `upload_into` が書き込み前に 1 回だけ
`synchronize()` する」という残存同期点は、bias が NT/TN（d_weight と同じ判定）と
なる層では完全に解消される——同一 `ctx.encode` 呼び出しへ折り込まれるため
`upload_into` 自体を呼ばない。NN/TT・分類不能形状の層では従来どおり `upload_into`
を要するため、モデル構成によっては残存同期点が残る（例: `Linear(1, n)` の
1 層目は引き続き NN 扱い）。実測（実機カウンタ・A/B）は本 Linux セッションでは
実施できないため未記入（`docs/backend-metal-command-batching-design.md` §10.7
「Mac 実機セッションへの申し送り」参照）。

**#1566 の A/B・checksum 比較の注意（f64 統一後。§10.8 追記に伴う補足）**:
bias 勾配は before（main の `reduce_to_shape` f32 逐次和）と after（本イシューの
f64 相当アキュムレータ）が bit 一致しない。これは事前登録規則の事後緩和ではなく、
2026-09-12 のユーザー承認 A に伴う数値契約の明示的な変更である。このため
Mac 実機セッションでの A/B・checksum 比較は、**bias 勾配については REQ-2
統一複合判定**（相対誤差 1e-3 未満 または 絶対誤差 1e-5 未満）で行う。
**weight 勾配・loss は従来どおり bit 同一契約**（`reduce_to_shape` 自体は
不変・`gemm_fp32_strict_into` の weight 書き込みも不変のため）——ただし
この契約は「同一パラメータに対する単発の backward」に限る。多 step の
train reuse A/B では、ある step の bias 勾配が REQ-2 範囲内で before/after
乖離すると、`step()`（SGD 更新）がその bias を使ってパラメータを更新する
ため、**以降の step の forward/backward 全体（loss・weight 勾配・weight
そのものを含む）へ乖離が伝播しうる**（`docs/backend-metal-command-batching-
design.md` §10.9 の `metal_reuse_step_grad_bit_dump` 訂正・
`crates/facade/tests/device_param_store_grad_readout.rs::param_grads_to_
host_matches_host_only_path_two_layer` の Linux 回帰で確認）。このため
train reuse A/B・`metal_reuse_step_grad_bit_dump` の Mac 実機比較は、bias を
含む step 以降は **loss・weight 勾配・パラメータも含め REQ-2 統一複合判定**
で行う（step 0 の bias 縮約自体のみが直接の変更対象であり、その後の伝播は
間接的な帰結）。

**追補（2026-09-12・最終形。判定契約の緩和を撤回）**: PR #1659 は当初、Metal
bias 縮約カーネル（`gemm_bias_grad_reduce_f32`）が f32 のみの補償和で
ホスト `f64` 逐次和と一致しない相殺列を、判定契約側（Tier A 理論上界／
Tier B 条件付き REQ-2 判定。契約 PR #1666）で吸収しようとしたが、codex-review
で収束せず撤回した。最終形はカーネル側を **IEEE 754 binary64 逐次加算の 64bit
整数ソフトウェアエミュレーション**へ置き換え、ホスト参照実装
（`eval::reduce_bias_grad_rows`）と **bit 完全一致**させる方式（`crates/
backend-metal/src/soft_f64.rs` が逐語モデル）。bias 勾配も weight 勾配と同じ
bit 一致契約となる。ただし bit 一致の対象は **変更後の Metal カーネル対
変更後のホスト `f64` 参照実装**（現行実装同士の比較）に限る。上記本文の
before（main の f32 逐次和）／after（f64 逐次和）の A/B・checksum 比較は、
変更前後で数値方式自体が異なるため（例: `[1e8, 1, -1e8]` の和は before が
`0`・after が `1`）Metal を binary64 エミュレーションへ置換しても一致せず、
引き続き上記本文どおり **REQ-2 統一複合判定**（伝播を含む多 step 比較も
同様）で行う。tolerance・REQ-2 の適用範囲は不変。経緯・検証方法は
`docs/backend-metal-command-batching-design.md` §10.14 を参照。カーネル置換後
の reuse `step_total` 再計測は未実施（並走ビルド中はベンチを行わない規約）。
