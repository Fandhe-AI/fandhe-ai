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
mse backward 自体は明確に改善）で、同時に device_update（SGD 更新。
本イシューの変更対象外のフェーズ）247.3us → 280.6us（1.135）の増加も
観測した。ただしこのフェーズ値は非計装の 5 run 中央値比
`ratio = 1.0167 > 1.00` とは別に単発計測したものであり、run 内比は
`1.0578, 1.2644, 1.0117, 0.9734, 0.8833` とばらつきが大きく
（record_only・負荷ゲートなしのため）5 run 中 2 run は `ratio < 1.00`
（符号一貫ではない）。そのため device_update の増加が step_total の
`ratio > 1.00` を引き起こしたと断定することはできない（原因は未特定。
§7 参照）。

## 7. 判定（verdict）・出荷状態

事前登録規則: 「対象セルのいずれかで `ratio > 1.00` → REJECT」。
GB10 の reuse セルが `ratio = 1.0167 > 1.00` であるため、
**Phase 1 は REJECT と確定する**（M4 Max 単独では非後退だが、事前登録
規則は事後緩和を許さない契約のため、GB10 1 件の超過で全体を REJECT
とする）。

- mse backward 自体（フェーズ分解の `backward` 列）は両機体で改善
  方向（M4 Max 0.922・GB10 0.827）であり、しきい値フォールバックの
  機構自体が意図どおり機能していることは確認できる。
- GB10 の step_total 後退時に `device_update`（SGD 更新。本変更の
  対象外コード）フェーズの時間が併せて増加している（247.3us →
  280.6us）ことを観測したが、run 間ばらつきが大きく（record_only・
  負荷ゲートなし）5 run 中 2 run は `ratio < 1.00` と符号一貫でも
  ないため、後退の原因を mse backward の変更や device_update の
  変動のどちらかに断定することはできない。原因は**未特定**のまま
  記録するに留め、事前登録規則（フェーズ単位の帰属で判定を分けない
  契約）のとおり判定（REJECT）は変えない。
- **出荷状態**: `MSE_BACKWARD_PARALLEL_MIN_ELEMS = 0`（常に並列。
  変更前と bit 同一の挙動）で確定する。機構（しきい値分岐・
  `_with_threshold` API・テスト）は残す。将来、GB10 で本イシューの
  対象外要因（device_update フェーズの変動を含むが原因未特定）が
  解消され再計測できる場合は、`T = 1 << 18` を再適用候補として本 doc
  の Phase 0 の値を再利用できる。

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

## 9. #3195 再測定（T = 4096）

§1〜§8 は変更しない（#1578 の REJECT 判定は不変）。本節は別候補・別規則による再測定の記録である。

- 規則: `docs/perf/logs/cpu-mse-backward-threshold-3195/RULE.txt`（計測前に固定）。候補は T = 4096 の単一。
- 変える条件: (1) 候補 T = 4096、(2) GB10 `device_update` 固定費の低下（#3193・#3194。279 -> 169 µs）、
  (3) round ごとの負荷ゲート・クールダウン、(4) ADOPT／REJECT／undetermined の 3 値判定（ADOPT 条件は #1578 より緩めない）。
- train セル（n = 640）では 640 < T のどの T も同じ逐次分岐を通るため、after 腕の train 実行コードパスは
  #1578 の T = 1<<18 と同一。候補変更が効くのは train 以外の 640 < n < 1<<18 の入力。
- 状態: **ADOPT**（2026-10-23。RULE.txt §6 を機械適用）。`MSE_BACKWARD_PARALLEL_MIN_ELEMS` を 0 から 4096 へ変更した。
- 前提: 3 者 bit 一致テスト（`mse_loss_backward_threshold_bit_exact`）pass（両機体）。Stage 0 前提ゲート (a) checksum 全一致・
  (b) 両機体 n in {640, 2560} で 5/5 プロセス r(n) < 1.00 とも成立（r(640) 中央値: Mac mini 0.0018・GB10 0.0014、
  r(2560): 0.0041・0.0030、r(4096): 0.0055・0.0042）。
- Stage 1（framework-compare train 64・cpu。before = main `fd1ca01a9`〈定数 0〉、after = 同 sha + 定数 1 行〈4096〉。5 round）:

| 機体 | mode | before 中央値 | after 中央値 | after/before | round ごとの ratio | checksum | gate_ok |
|---|---|---|---|---|---|---|---|
| Mac mini | fresh | 721.8 µs | 692.6 µs | 0.9595 | 0.9904, 0.9667, 0.9549, 0.9559, 0.9962 | 完全一致 | 5/5 |
| Mac mini | reuse | 698.5 µs | 670.5 µs | 0.9599 | 0.9598, 0.9650, 0.9474, 0.9669, 0.9608 | 完全一致 | 5/5 |
| GB10 | fresh | 885.4 µs | 850.2 µs | 0.9602 | 0.9482, 0.9875, 0.9570, 0.9103, 0.9774 | 完全一致 | 5/5 |
| GB10 | reuse | 862.7 µs | 827.3 µs | 0.9589 | 0.9367, 0.9624, 0.9644, 1.0493, 0.9556 | 完全一致 | 5/5 |

- GB10 reuse の round 4 のみ ratio > 1.00（1.0493）だが、全 round 一貫の後退ではなく REJECT 条件に該当しない。
- 注記: train（n = 640）の after 腕の実行コードパスは #1578 の T = 1<<18 と同一で、#1578 で GB10 reuse が後退した（1.0167）のに対し
  今回の差が出た要因は RULE §1 条件 2〜4（固定費低下・負荷ゲート）である。系列間の差の因果確定は本節の範囲外。
- 記録: `docs/perf/logs/cpu-mse-backward-threshold-3195/`（README・`macmini/`・`gb10/`・`stage0-aggregate.md`）。
- ハーネス: `scripts/bench/framework-compare/run_ab_mse_backward_threshold_cpu.sh`。
