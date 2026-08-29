# CUDA 都度同期除去（イシュー #1013）RTX 3060 実測記録

## 0. 位置づけ

本文書はイシュー #1013（GEMM／elementwise／reduction／SGD カーネルの都度
`stream.synchronize()` 除去。設計文書 `docs/backend-cuda-async-execution-design.md`
の実装）における、本ホスト（RTX 3060・CC 8.6）での before/after 実測記録である。
DGX Spark GB10・Metal M4 Max での再計測は未実施のまま残す（§4「未実施事項」）。

## 1. 計測環境

- Linux x86_64・NVIDIA RTX 3060（CC 8.6）・driver 595.71.05
- NVRTC はセッション一時提供（pip パッケージ由来。`nvidia-cuda-nvrtc==13.0.88`
  等を scratchpad へ展開し `LD_LIBRARY_PATH`／`CUDA_INCLUDE_PATH` で指定）。
  資格情報は含まない一時的な実行時ライブラリ提供であり、本体・
  `scripts/bench/framework-compare/` の依存構成は変更していない
- `cargo test -p fandhe-ai --release --test device_param_store_bench -- --ignored --nocapture`
  （5 回計測）

## 2. 計測対象と計測境界の見直し（before/after で同一手順を使う理由）

`crates/facade/tests/device_param_store_bench.rs::legacy_vs_resident_per_step_cuda`
（#936 の非後退確認ベンチ）を流用した。ただし、このベンチのループ本体には
#1013 適用後は明示的な完了待ちが一切残らない（forward/backward/update いずれも
非同期投入のみ）ため、素の `Instant` 計測はホスト側のディスパッチ時間
（enqueue コスト）しか捉えず、GPU 実行完了を反映しない見かけ上の高速化に
化ける（レビュー時に事前検討した既知のリスク）。

これを避けるため、本計測用に `run_resident_path` の 1 step 末尾へ
`tape.sync_device_param_store_to_host(&store)`（readback 境界。
`DeviceParamStore::sync_to_host` への薄い委譲）を追加し、旧経路が暗黙に
持っていた「launch 直後の同期」と同等の完了保証を計測境界へ復元した
（`crates/facade/tests/device_param_store_bench.rs` 差分参照）。**before・
after いずれもこの追加を含む同一のベンチファイルで計測する**（比較可能性
を保つため。before は #1013 のソース変更を適用する前の HEAD に対して、この
ベンチ差分のみを追加した状態で計測した）。

## 3. 実測結果（5 回計測の中央値。q1/q3 はベンチ出力のまま）

### 3.1 素の baseline（ベンチ差分なし。#1013 適用前 HEAD）

同期境界の追加なしで計測した参考値（旧経路は launch 直後に内部
`synchronize()` するため、この状態でも total_secs は GPU 完了を捕捉できる）。

| 指標 | 中央値 (5 回) |
|---|---|
| resident_total_median_s | 89.9〜92.1 µs |
| resident_update_median_s | 7.1〜7.4 µs |

### 3.2 before/after（§2 のベンチ差分を含む同一手順。比較可能性を保った計測）

| 指標 | before（#1013 適用前） | after（#1013 適用後） |
|---|---|---|
| resident_total_median_s | 95.8〜100.2 µs | 108.5〜111.7 µs |
| resident_update_median_s | 7.0〜7.5 µs | 4.0〜4.3 µs |
| legacy_total_median_s（参考。旧経路は無変更） | 107.7〜111.6 µs | 119.9〜126.0 µs |

## 4. 結果の解釈（誠実な報告。速報的な高速化を主張しない）

- **update フェーズ単体**（`step_device_param_store` 呼び出しの enqueue から
  return までの区間）は before 比で約 1.7 倍高速化した（7.0〜7.5 µs →
  4.0〜4.3 µs）。これは `sgd.rs::CudaSgd::run` の都度 `synchronize()` 除去
  （#1013 の最優先項目）が意図どおり dispatch 時間を短縮したことを直接示す
- 一方 **1 step 全体**（forward + backward + update + 明示 readback）は
  before 比でむしろ増加した（95.8〜100.2 µs → 108.5〜111.7 µs）。本ベンチの
  モデルは `BATCH=4, D_IN=8, D_HIDDEN=16, D_OUT=4` という極めて小さいトイ
  モデル（#936 の非後退確認用に選定された形状）であり、GPU 実行時間より
  ホスト側のカーネル起動オーバーヘッド（`cuLaunchKernel` 呼び出し自体の
  レイテンシ）が支配的な領域にある。非同期化がもたらす「複数カーネルの
  完了待ちを 1 回にまとめる」利点は、まとめて隠すべき実 GPU 実行時間が
  ほぼ存在しないこの規模では顕在化せず、むしろ本計測用に追加した明示
  readback（`sync_to_host`。全パラメータの D2H）1 回分のコストが正味の
  増加として観測されたと考えられる
- したがって **本計測は「#1013 が全体を高速化した」ことを主張する根拠には
  ならない**。親イシュー #1008 が診断した実際の性能差（MLP 学習 1 step で
  candle 比 1 桁以上）は、本ベンチより大幅に大きい実践的なモデル形状
  （`scripts/bench/framework-compare/` の比較対象）で生じており、その規模
  でこそ非同期化による「都度同期の除去」が効果を持つと考えられる。
  この規模での再計測は、`scripts/bench/framework-compare/bench-fandhe` が
  crates.io の `fandhe-ai =0.4.0` にピン固定されているため本 PR のローカル
  変更では実行できない（`.claude/rules/deps-policy.md` 第 9 区分の契約。
  次回 crates.io 公開後、または承認済み path 差し替えでの再計測が必要）
- 受入基準の「`bench-fandhe train cuda` の 1 step 改善」は、上記の理由により
  **本 PR 時点では未検証のまま残す**（スコープ外。§5 参照）。update フェーズ
  単体の直接計測（本文書 §3.2）は、除去した都度同期が意図どおり dispatch
  コストを下げたことの直接証拠として記録する

## 5. 未実施事項（スコープ外。ユーザー承認後に Issue 追跡）

- DGX Spark GB10・Metal M4 Max での本ベンチ再計測
- `scripts/bench/framework-compare/bench-fandhe train cuda` によるフレーム
  ワーク横並び 1 step 改善の実測（crates.io 公開版ピンのため次回公開後）
- より大きい実践的なモデル形状（本ベンチのトイモデルではなく、
  `docs/backend-cuda-async-execution-design.md` §1 が言及する規模）での
  workspace 内 A/B 計測の追加（非同期化の効果が顕在化する規模の特定）
- 設計文書 §9 item 7・9〜11（`ops.rs` 各演算入口への `begin_driver_call`／
  `observe_driver_result` 結線。Phase C）の実装。本 PR は状態機械本体
  （Phase B: `context_cache` の poison 状態機械・`BackendError` 4 variant・
  `DeviceBuffer::generation`）のみを実装し、advisor 助言（部分結線は
  fail-open になりかねない）に基づき結線自体は次イシューへ引き渡す
- イシュー #1014（実機回帰テスト。設計文書 §8 の T1・T2・T3b・T3i・T4）
