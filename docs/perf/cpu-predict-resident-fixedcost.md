# CPU `predict_resident` 固定費の切り分け（診断基盤。GB10 実測は #3188 で実施済み・M4 Max は未実施）

イシュー #2105。`docs/perf/infer-reuse-phase-breakdown.md` §10.6.2〜§10.6.4 が
報告した「DGX Spark GB10 の CPU で reuse（`Sequential::predict_resident`）が
fresh（`Sequential::predict`）より遅い」逆転の原因を、フェーズ分解で切り分ける
ための診断基盤。

## 1. 状態

| 項目 | 状態 |
|---|---|
| フェーズ分解テスト（`crates/facade/tests/cpu_predict_resident_fixedcost_diag.rs`） | 実装済み |
| 事前登録の判定規則（`docs/perf/logs/cpu-predict-resident-fixedcost-2105/RULE.txt`） | 実測前に固定済み |
| 計測・集計スクリプト（`orchestrate.sh`／`aggregate.py`） | 実装済み（x86 ホストで流れの smoke のみ） |
| DGX Spark GB10 での 5 run 実測と仮説判定 | **実施済み（#3188。pre／head の 2 系列。§6）** |
| Apple M4 Max での 5 run 実測と仮説判定 | **未実施（実機セッションへ申し送り）** |
| 対処（修正）の実装 | 本イシューの対象外。§7 に起票案のみ（起票はしていない） |

本 doc に実測値は一切載せていない。x86_64（RTX 3060 ホスト）で流れを確認した
smoke 出力は記録に使わない。

## 2. 現象（既存記録の要約）

registry `fandhe-ai =0.9.0`・batch 64・784→256→ReLU→10（`infer-reuse-phase-breakdown.md`
§10.6.2〜§10.6.4 の実測）。

- GB10 CPU: reuse の iter_total 196.0 µs、fresh 177.1 µs（reuse が +18.9 µs 遅い）
- M4 Max CPU: reuse 176.0 µs、fresh 185.5 µs（逆転しない）
- bench-fandhe の `--phases` では `predict_resident` が単一区間で内訳が取れない

## 3. コード事実（読解。本 PR は本番コードを変更しない）

`Sequential::predict_resident`（`crates/facade/src/compat/sequential.rs`）は次の順で動く。

1. ガード 2 つ（`contains_resident_unsupported_layer`・`reject_untracked_parametric_layer_resident`）
2. `crate::tape_for(device)`: `Tape::new_with_ops(Box<CpuBackendOps>)`
3. `snapshot_resident_params`: 葉ノード 4 個を push
4. `build_device_chain_steps`（`Vec` 構築）→ `DeviceParamStore::predict_device_chain`
   - `MemoryOps::upload`（入力約 200 KB の `to_vec` と `TrackedAllocation` 確保）
   - 層ごとに `linear_forward_device_tracked` → `CpuBackendOps::linear_forward_device`
   - `MemoryOps::download`（出力の clone）

`CpuBackendOps::linear_forward_device`（`crates/backend-cpu/src/ops.rs`）は
bit-exact 契約を理由に意図的に非融合で、`gemm_blis_parallel` → bias の行方向ループ
→ relu ループの 3 段、出力は毎回 `vec!` 確保→`wrap_vec` する。

fresh の `predict` は tape を使わない経路で `&Tensor` を直接渡す（入力コピーなし）。
L1 は融合 `Linear::forward_host_with_activation`（`gemm_bias_act`→
`gemm_blis_bias_act_parallel`）、L2 は `gemm`→`add`（`cpu-infer-predict-profile.md` §4.2）。

補足（記述の食い違い。是正は本 PR の対象外。**#3196 で解消済み**: bit 一致テストで確認し融合経路へ切替、`cpu-linear-forward-device-fusion.md` 参照）: `cpu-infer-predict-profile.md` §2・§3 と
`gemm_blis_bias_act_parallel` の doc は epilogue が K 全体の蓄積後に適用されるため融合版が
非融合合成と bit 一致すると述べる。一方 `linear_forward_device` の doc は「融合カーネルは
加算順序が変わりうる」と述べる。両者の整合は後続の対処 issue で確認する材料である。

両 GEMM 入口は GB10 affinity（#1576）と small_shape_thread_cap（#1575）を同じ仕組みで
共有する。これらは既に決定記録があるため本イシューではスイープしない。

## 4. 事前登録仮説

| 仮説 | 内容 | 主な見る区間 |
|---|---|---|
| H1 カーネル差 | reuse の L1 は非融合、fresh の L1 は融合 | `l1_linear_forward_device` 対 `l1_linear_relu_fused`（ablation に非融合ホスト版） |
| H2 コピー・確保 | reuse だけに `upload`／`download`／`wrap_vec` がある | `upload`＋`readout`（ablation に入力コピー単独） |
| H3 アロケータ／ページフォルト | 約 200 KB の呼び出しごと確保が Linux の mmap 閾値付近で minor fault を生む | minflt 差分（補助情報のみ） |
| H4 tape 固定費 | `Tape::new_with_ops`・葉 push・steps 構築・ガード | `tape_build`＋`iter_total` 残差（H6 を除く） |
| H6 chain 検証・tracked 差 | 分解が省略する `predict_device_chain` の resident buffer 検証・`linear_forward_device_tracked` | `chain_public` − `forward_resident` − `readout` |
| H5 順序効果・二峰性 | reuse の min が fresh の min 付近まで下がる（§10.6.2） | arm 順のラウンド反転・q1/q3/min/max |

再実行しない既存実験: #1575 thread cap・#1576 GB10 affinity・B packing 共有・SME ゲート・
2D dynamic・`RAYON_NUM_THREADS` スイープ（決定記録あり）。

## 5. 計測プロトコル

判定規則の正は `docs/perf/logs/cpu-predict-resident-fixedcost-2105/RULE.txt`
（実測前に固定。事後に緩めない）。要点:

- 1 run = 1 プロセスで独立 5 run。各セルは run ごとの median の中央値と min–max
- checksum は arm 内で run 間完全一致、`reuse_decomposed` ≡ `reuse`（fail-closed）
- `ratio = reuse.iter_total / fresh.iter_total`。`<= 1.00` を非後退基準として記録
- 帰属: 1 区間の差分が iter_total 差の 50% 以上なら対応仮説を「支持」、なければ「未確定」
- GB10 は `load1 < 1.0` の専有ゲート（30 秒間隔・最大 30 分）、M4 Max は record_only

テストは 2 本。

- `cpu_predict_resident_decomposition_matches_public_api_bit_exact`（CI 実行）: 公開
  `predict_resident`・autodiff レベルの写し・backend レベルの写しが bit 一致すること
  （分解の帰属の前提）
- `cpu_predict_resident_fixedcost_phases`（`#[ignore]`・record-only）: fresh／reuse／
  reuse_decomposed／ablation を 20 反復 × 4 ラウンド（arm 順を反転）で計測し JSON 行を出力

minflt の採取は `/proc` 読み出しが計測区間を歪めるため既定 OFF。`FIXEDCOST_MINFLT=1`
の別 run（別 `--out`）でのみ有効化する。

## 6. 結果（実機実測後に記入。現在は未実測）

### M4 Max

未実測。`bash docs/perf/logs/cpu-predict-resident-fixedcost-2105/orchestrate.sh m4max` の後
`python3 .../aggregate.py <dir>` の出力を転記する。

### GB10（#3188）

継承 RULE を無変更で適用し、2 系列を 5 run ずつ実測した（追加規則は `logs/cpu-train-fixedcost-gb10-3188/RULE.txt`、計測前に固定）。pre = `17c7767b5`（#3196 の直前）、head = `4f6716f63`（#3188 着手時の origin/main）。`.rev-stamp`・ツリー指紋（計測前後・Mac 側アーカイブ展開と差分 0 件）・負荷ゲート全 run pass・checksum 一致・aarch64 bit 一致テスト pass を確認済み。生ログと集計は `logs/cpu-train-fixedcost-gb10-3188/{pre,head}/predict-resident-2105/`。

| 区間（中央値 µs） | pre | head |
|---|---|---|
| fresh iter_total | 193.7 | 190.9 |
| reuse iter_total | 197.2 | 193.3 |
| ratio（reuse / fresh） | 1.0179 | 1.0124 |
| diff（reuse − fresh） | 3.5 | 2.4 |
| upload | 6.1 | 6.5 |
| readout | 0.8 | 0.9 |
| L1 `linear_forward_device`（reuse） | 146.0 | 145.8 |
| L1 融合（fresh_layers） | 133.9 | 135.8 |
| L2 `linear_forward_device`（reuse） | 41.4 | 39.0 |
| L2 GEMM + add（fresh） | 50.3 | 48.8 |
| tape_build | 1.4 | 1.5 |
| chain_public | 196.0 | 197.5 |
| H1 カーネル差（L1） | +12.1 | +9.9 |
| H2 コピー／確保（upload + readout） | +7.0 | +7.4 |
| H4 tape 固定費 | -0.0 | -6.5 |
| H6 chain 検証・tracked 差 | -0.2 | +6.4 |

### 仮説判定

継承 RULE の帰属規則（50% 規則）に従う。ratio は両系列とも 1.00 超で「逆転が再現」。

- pre: **H1 カーネル差（L1）・H2 コピー／確保（upload + readout）** に帰属
- head: **H1 カーネル差（L1）・H2 コピー／確保・H6 chain 検証・tracked 差** に帰属

**reuse > fresh の差を説明できるか: 一部のみ**。逆転の向きは両系列で再現するが、差は 2.4〜3.5 µs（ratio 1.01〜1.02）で、`infer-reuse-phase-breakdown.md` §10.6 の記録（reuse 196.0 µs・fresh 177.1 µs、差 18.9 µs）の約 1/5〜1/8 にとどまる。run 内の iter_total は 130〜440 µs で振れ（q1/q3 中央値の幅は約 25 µs）、2〜3 µs の差は run 内ばらつきより小さい。H1 の正の寄与（+9.9 µs）と H2 の +7.4 µs は L2 差（-9.8 µs）で相殺され、差の符号は残るが小さい。

- **想定と異なった点**: 追加規則では head の H1 は構造上 0 付近と見込んでいたが、実測は pre から変化せず（reuse の L1 は 146.0 → 145.8 µs、fresh の融合 L1 は 133.9 → 135.8 µs）、head でも L1 の約 10 µs 差が残る。#3196 は GB10 のこのフェーズ分解では L1 を短縮していない。原因は本診断では確定していない
- upload（約 6 µs）＋ readout（約 1 µs）は定常的な固定費で、fresh 側の `input_to_vec_copy`（約 7 µs）と同規模
- bench と診断の値がずれる理由: 計測窓（bench は連続窓、診断は区間ごとの `Instant`）と版（bench 記録は 0.9.0〜0.11.0 の公開版）が異なる

## 7. 対処の起票案（列挙のみ。起票はユーザー承認待ち）

実測で支持された仮説に応じた候補。いずれも本イシューでは実装しない。

- （#3196 で実装済み）`linear_forward_device` の CPU 実装を融合 epilogue へ切り替える案と、§3 の doc の
  食い違いの是正（H1 支持時）
- `predict_device_chain` の入力 `upload` のコピー削減（`&Tensor` を借用する CPU 専用
  ゼロコピー経路など。H2 支持時）
- tape 構築・snapshot の CPU 推論での省略や再利用（H4 支持時）

## 8. 限界

- 本テストは HEAD（main）を計測する。#1981 の実測は registry 0.9.0 で、以降の変更
  （#2398 のガード追加等）が入っているため直接比較は差分を含む
- x86_64 では実測していない（申し送り）。M4 Max の 5 run は未実施。GB10 は #3188 で実施済み
- `FIXEDCOST_MINFLT=1` の補助 run（H3 のページフォルト）は未実施（H3 は判定に使わない補助情報）
- head 系列の H1 は構造上 0 付近と見込んだが、実測では約 10 µs 残った（§6）。原因は未確定
- 帰属は区間差分の 50% 規則による推定であり、因果の確定ではない
- 計測区間の `Instant` 呼び出し自体のオーバーヘッドが、細かい区間（数百 ns〜数 µs）に
  含まれる。`reuse_decomposed` の iter_total は `reuse` より大きくなりうる

出典: `docs/perf/infer-reuse-phase-breakdown.md` §10.6、`docs/perf/cpu-infer-predict-profile.md`。
