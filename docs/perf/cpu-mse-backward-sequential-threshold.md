# mse_loss_backward の要素数しきい値による逐次フォールバック（イシュー #1578）

## 1. 背景・機構

低レイヤー診断（`docs/perf/lowlayer-diagnosis-2026-09-12.md` §4・§7
「A-11」。一次情報源イシュー #1574）が実測したとおり、
`crates/backend-cpu/src/mse.rs::mse_loss_backward_f32` は
`par_iter_mut().zip().zip()` で並列化しているが、
framework-compare の `train`（`BATCH=64 × D_OUT=10 = 640` 要素。
`scripts/bench/framework-compare/bench-fandhe/src/main.rs`）規模では
rayon の fork-join 固定費が支配的になる。backward は要素独立
（アキュムレータなし）の map 演算のため、逐次・並列のどちらを通っても
**構造上 bit 同一**になる（`scale * (p - t)` を並列分岐と同一の式で
計算し、結合則の影響を受ける演算の跨りがない）。

forward の二乗和 `mse_sum_sq_f32`（`par_chunks(CHUNK=4096)` の決定的
reduction）は別経路であり、本イシューの対象外・変更なし。

## 2. 設計

`crates/backend-cpu/src/mse.rs`:

- `MSE_BACKWARD_PARALLEL_MIN_ELEMS`（`pub(crate)`）: この要素数
  **未満**は逐次ループへフォールバックするしきい値。
  `crate::elementwise::PARALLEL_THRESHOLD`（`1 << 15`。「ベンチ根拠が
  出るまでの保守的な固定値」と自己申告）と同型のパターンだが、
  MSE backward 固有の実測で個別に決定するため別定数として持つ。
- `mse_loss_backward_f32_with_threshold(pred, target, scale, dpred, min_elems)`
  （`pub(crate)`）: しきい値を明示的に受け取る本体。`pred.len() <
  min_elems` なら逐次 `for` ループ、それ以外は従来の `par_iter_mut`
  経路。
- `mse_loss_backward_f32`（既存シグネチャ・`ops.rs` 側は無変更）:
  `MSE_BACKWARD_PARALLEL_MIN_ELEMS` を既定値として `_with_threshold`
  を呼ぶ薄いラッパー。
- `unsafe` なし・新規 `pub` API なし（`fandhe-ai-backend-cpu` は
  crates.io 公開クレートのため公開面を増やさない）。

## 3. bit 同一契約の検証

- `mse.rs` 単体テスト `mse_loss_backward_threshold_bit_exact`:
  forced-seq（`min_elems = usize::MAX`）／forced-par（`min_elems = 0`）／
  素朴 `for` ループの 3 者を `n ∈ {0,1,2,639,640,641,32767,32768,32769}`
  ×`scale ∈ {1.0,-2.5,0.0}` で `to_bits()` 完全一致（`NaN`・`-0.0`・
  subnormal・`±inf` を含む）。
- `mse_loss_backward_default_threshold_is_expected`: 既定ラッパーが
  `_with_threshold(…, MSE_BACKWARD_PARALLEL_MIN_ELEMS)` と bit 一致
  （ラッパー結線のドリフト検出）。
- 統合テスト `tests/mse_parity.rs::mse_loss_backward_bit_matches_naive`:
  公開 API（`CpuBackendOps::mse_loss_backward`）と素朴参照実装を
  `n ∈ {0,1,2,100,640,4095,4096,4097,8193,32767,32768,32769,65535,65536,65537}`
  で `to_bits()` 完全一致。
- 長さ不一致は逐次・並列いずれの分岐でも `BackendError::ShapeMismatch`
  として検出（`mse_loss_backward_with_threshold_length_mismatch_both_arms`）。

全テスト green（`cargo test -p fandhe-ai-backend-cpu`）。

## 4. 事前登録した規則（イシュー #1578 コメント。実装着手前に固定）

イシューコメント: https://github.com/Fandhe-AI/fandhe-ai/issues/1578#issuecomment-5645816781

### Phase 0（しきい値決定・マイクロベンチ）

- 計測: `cargo test -p fandhe-ai-backend-cpu --release --lib
  mse::tests::mse_backward_threshold_sweep -- --ignored --nocapture`
  （`RAYON_NUM_THREADS` 未設定・本番同一）。
- サイズ `n ∈ {640, 2560, 4096, 8192, 16384, 32768, 65536, 131072,
  262144}`。腕は forced-seq／forced-par。各 (n, 腕) 1000 反復の
  プロセス内中央値（ns）を計測、プロセス起動 5 回の中央値を代表値。
  `r(n) = seq_median / par_median`。
- 決定: 機体ごとに `n*` = 最初に `r(n) > 1.00` となる n（なければ
  `1 << 18`）。`T = min(n*_M4Max, n*_GB10)`（上限 `1 << 18`）。
  `T <= 640` なら REJECT。

### Phase 1（framework-compare train A/B・record_only）

- 腕: before = `origin/main` の `crates/facade`、after = 本ブランチの
  `crates/facade`（`[patch.crates-io.fandhe-ai]` path patch）。
- 対象セル: `bench-fandhe --task train --size 64 --device cpu --mode
  {fresh,reuse}`（判定は reuse のみ・fresh は対照/参考）。
- 各セル 5 run・起動順反転。`ms/step` 5 run 中央値で
  `ratio = after/before`。
- 判定: 全対象セル `ratio <= 1.00` かつ checksum 完全一致 → ADOPT。
  いずれか `ratio > 1.00` → REJECT。事後緩和・セル除外・run 追加に
  よる再判定は行わない。

## 5. Phase 0 実測結果

M4 Max・GB10（DGX Spark）各 5 プロセス起動。生ログ:
`docs/perf/logs/cpu-mse-backward-threshold-1578/{m4max,gb10}/run{1..5}.log`。

両機体・全 checksum が seq/par で完全一致（bit 同一の直接確認）。

### M4 Max（16 threads・共有負荷下。load average 6〜25 と変動大）

| n | seq median (ns) | par median (ns) | r(n)=seq/par |
|---|---|---|---|
| 640 | 42 | 60875 | 0.0007 |
| 2560 | 125 | 77958 | 0.0016 |
| 4096 | 208 | 81750 | 0.0025 |
| 8192 | 416 | 94000 | 0.0044 |
| 16384 | 958 | 103750 | 0.0092 |
| 32768 | 3250 | 103250 | 0.0315 |
| 65536 | 7291 | 113916 | 0.0640 |
| 131072 | 13666 | 128500 | 0.1064 |
| 262144 | 27375 | 140583 | 0.1947 |

`r(n)` はスイープ上限（262144）まで一度も 1.00 を超えなかった
→ `n*_M4Max = 1 << 18`（事前登録規則「見つからなければ `1 << 18`」）。

### GB10（DGX Spark。20 threads・ほぼアイドル。load average 0.01〜1.66）

| n | seq median (ns) | par median (ns) | r(n)=seq/par |
|---|---|---|---|
| 640 | 48 | 32976 | 0.0015 |
| 2560 | 128 | 43088 | 0.0030 |
| 4096 | 192 | 45440 | 0.0042 |
| 8192 | 368 | 51024 | 0.0072 |
| 16384 | 928 | 57728 | 0.0161 |
| 32768 | 1920 | 64512 | 0.0298 |
| 65536 | 4000 | 70256 | 0.0569 |
| 131072 | 8624 | 78768 | 0.1095 |
| 262144 | 42416 | 90992 | 0.4662 |

GB10 もアイドル環境で一度も 1.00 を超えなかった
→ `n*_GB10 = 1 << 18`。

### Phase 0 決定

`T = min(n*_M4Max, n*_GB10) = 1 << 18 = 262144`（`T > 640` のため
Phase 0 単独では REJECT にならず、候補 `T` として Phase 1 へ進む）。

640 要素規模では並列は逐次の**2 桁以上遅い**（`r(640)` が
0.0007〜0.0015）ことが両機体で一貫して確認できた。一方 262144 要素
（rayon が本来効くはずの規模）でも `r(n)` が 1.0 に届かず、fork-join
固定費が本カーネルの演算強度（要素あたり 1 減算 + 1 乗算のみ・
メモリバウンド）に対して常に支配的であることを示す。これは
`elementwise::PARALLEL_THRESHOLD = 1 << 15` が「未チューニングの保守的
固定値」と自己申告していた値より実測しきい値が大きく上回ることを
示唆する（§8 スコープ外）。

## 6. Phase 1 実測結果（framework-compare train A/B）

`scripts/bench/framework-compare/run_ab_1578.sh 1578`
（`AB_DEVICE=cpu`・5 round・起動順反転）。生ログ・判定表:
`docs/perf/logs/cpu-mse-backward-threshold-1578/{m4max,gb10}/
compare-train-1578-cpu{,-fresh-reference}.md`。

### M4 Max（`T = 1 << 18` を適用した after 腕）

| セル | before median | after median | ratio | checksum |
|---|---|---|---|---|
| 64/reuse（判定対象） | 933.2 us | 855.2 us | **0.9163** | 完全一致 |
| 64/fresh（参考） | 836.1 us | 736.2 us | 0.8805 | 完全一致 |

reuse フェーズ分解（診断）: backward 508.3us → 468.9us（0.922）・
step_total 909.5us → 847.4us（0.932）。非後退（`ratio <= 1.00`）。

### GB10（同一 after 腕）

| セル | before median | after median | ratio | checksum |
|---|---|---|---|---|
| 64/reuse（判定対象） | 980.0 us | 996.4 us | **1.0167** | 完全一致 |
| 64/fresh（参考） | 932.7 us | 969.8 us | 1.0398 | 完全一致 |

reuse フェーズ分解（診断）: backward 543.2us → 449.5us（**0.827**。
mse backward 自体は明確に改善）だが device_update（SGD 更新。本イシュー
の変更対象外のフェーズ）247.3us → 280.6us（1.135）が step_total 全体を
押し上げ、`ratio = 1.0167 > 1.00` となった。run 内比は
`1.0578, 1.2644, 1.0117, 0.9734, 0.8833` とばらつきが大きく（record_only・
負荷ゲートなしのため）、5 run 中 2 run は `ratio < 1.00`（符号一貫では
ない）。

## 7. 判定（verdict）・出荷状態

事前登録規則: 「対象セルのいずれかで `ratio > 1.00` → REJECT」。
GB10 の reuse セルが `ratio = 1.0167 > 1.00` であるため、
**Phase 1 は REJECT と確定する**（M4 Max 単独では非後退だが、事前登録
規則は事後緩和を許さない契約のため、GB10 1 件の超過で全体を REJECT
とする）。

- mse backward 自体（フェーズ分解の `backward` 列）は両機体で改善
  方向（M4 Max 0.922・GB10 0.827）であり、しきい値フォールバックの
  機構自体が意図どおり機能していることは確認できる。
- GB10 の step_total 後退は `device_update`（SGD 更新。本変更の対象
  外コード）の変動に起因しており、mse backward の変更に直接帰属
  できる後退ではない可能性が高いが、事前登録規則はフェーズ単位の
  帰属で判定を分けない契約のため、原因分析として記録するに留め
  判定は変えない。
- **出荷状態**: `MSE_BACKWARD_PARALLEL_MIN_ELEMS = 0`（常に並列。
  変更前と bit 同一の挙動）で確定する。機構（しきい値分岐・
  `_with_threshold` API・テスト）は残す。将来、GB10 の
  `device_update` フェーズ変動が別途解消され再計測できる場合は、
  `T = 1 << 18` を再適用候補として本 doc の Phase 0 の値を再利用
  できる。

ガードセル（Metal／CUDA デバイス。CPU mse 非到達）は本イシューの
実行環境・時間制約により未計測（verdict が既に REJECT に確定して
いるため判定には影響しない）。

## 8. スコープ外

- GPU 側 MSE backward の同期除去（既存 #1582）。
- `ops.rs::mse_loss_backward` の `vec![0.0f32; n]` 確保（低レイヤー
  診断で 165 ns 未満と確認済み・非支配的）。
- `elementwise::PARALLEL_THRESHOLD`（`1 << 15`・未チューニング）の
  実測チューニング。本イシューの Phase 0 結果（両機体とも 262144 要素
  でも `r(n)` が 1.0 未満）は、この定数が要素独立 elementwise 演算に
  対して過小評価である可能性を示唆する参考値になる。
- GB10 `device_update`（SGD 更新）フェーズの変動要因の切り分け
  （本イシューの変更対象コードではない）。
- Metal／CUDA ガードセルの実測（verdict に影響しないため本イシューでは
  未実施）。
