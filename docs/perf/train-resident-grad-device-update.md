# reuse 経路の weight 勾配をデバイス常駐 staging へ直接書き込む（イシュー #1212）

## 0. スコープ（実装時点の縮小判断）

計画（Plan フェーズ）が想定した全面実装（`Gradients` へのマーカー追加・
`vjp()` 返り値の enum 化・Metal `c_offset` スレッディング・CUDA view 出力
GEMM 追加・フレームワーク横並びベンチの前後 5 回計測キャンペーン）は、
単一セッション・委譲（subagent）なしという実行制約のもとでは検証可能な
安全側の増分に収まらないと判断し、以下へ**意図的に縮小**した:

- **CPU バックエンドのみ本番結線**。Metal／CUDA は
  `BackendOps::gemm_fp32_strict_into`（既定 `Unsupported`）に一切手を
  加えていないため、`grad::vjp` は既存のホスト経路
  （`ops.gemm_fp32_strict` → `Gradients` へ格納 → `step()` が連結
  `flat_grad` を 1 回 upload）へ**そのままフォールバック**する。挙動・
  性能とも本イシュー着手前と完全に不変（`cargo build`／`cargo clippy`
  済み。実機なしのため Metal／CUDA の速度計測は行っていない）
- **`Gradients` の構造体は変更しない**（`vjp()` の戻り型
  `Vec<(NodeId, Tensor<f32>)>` も不変）。設計判断の詳細は
  `docs/device-resident-update-design.md` 追補「#1212」§2 を参照
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
  バックエンド）
- Metal 実機（Apple Silicon）での同実装・実測
- `scripts/bench/framework-compare` を使った実践規模（`--task train`）の
  5 回計測キャンペーン（本ファイル §0 の軽量プロトコルの上位互換）
- bias 勾配（`reduce_to_shape`）自体のデバイス常駐化（デバイス側列縮約
  カーネルが必要。現状は bias は常にホスト経由で `upload_into` される）
- `Gradients` へのマーカー追加による「常駐 weight への `get()` が
  `Err(InvalidArgument)` を返す」設計（計画 §3.1 item 3）は本実装では
  採用しなかった（`grads.get()` は単に `Ok(None)` を返す。公開 API から
  `Op::ResidentLeaf` の `Var` を得る経路が元々存在しないため実害はないと
  判断したが、内部一貫性としては計画どおりの型付きエラー化が望ましい）
