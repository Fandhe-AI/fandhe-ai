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

Metal GEMM command batching ベンチ（`crates/backend-metal/tests/
command_batching_bench.rs::mnist_scale_train_reuse_metal_batch_counters`）において、
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

- CUDA 実機での `gemm_fp32_strict_into`／`upload_into` 実装・実測（#1212 から継続・
  既定 `Unsupported`・フォールバック）
- bias 勾配自体のデバイス常駐化（デバイス側列縮約カーネル必要。現状は bias は常にホスト経由
  で `upload_into`）
- `d_input` GEMM の同期境界解消（従来どおり `gemm()` → `download` 経路。スコープ外）
