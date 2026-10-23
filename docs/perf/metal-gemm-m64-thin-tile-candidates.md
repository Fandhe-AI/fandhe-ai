# Metal GEMM: m=64 級形状向け細タイル候補（既定 OFF）

イシュー #3208。A/B 計測は兄弟 #3209（Mac mini 上）。本 doc は候補の定義・到達範囲・bit 一致確認・所有者への申し送りを記録する。性能は計測していない。

## 1. 背景と前提

- 対象は MLP（784→256→10・batch 64）の GEMM 6 形状。依存の #3181 は正方形状のみ、#3180 は phase 内訳のみを測っており、m=64 形状のカーネル実測は存在しない。候補は机上で決め、#3209 が測る。
- 候補は `tile::M64_THIN_CANDIDATES`（`CANDIDATES` とは別配列）、ゲートは `tile::M64_THIN_TILE_ENABLED`（**既定 false**）。シェーダは変更しない（REQ-8 の手動境界検査は既存カーネルのまま）。tolerance・baseline・FMA 契約は不変。

## 2. 実際に走る経路（コード読解。origin/main `d731babdf`）

| 形状（Metal 側の向き） | 用途 | 現行で実際に通るカーネル |
|---|---|---|
| (64, 256, 784) NN | L1 forward | reuse／infer: スカラー `gemm_tiled_bias_act`（16×16）。fresh: `dispatch_auto` → split-K（机上計算） |
| (64, 10, 256) NN | L2 forward | 同上 |
| (784, 64, 256) NT | L1 d_input | スカラー `gemm_tiled_bias_act`（strided） |
| (256, 64, 10) NT | L2 d_input | 同上 |
| (784, 256, 64) TN | L1 dW | 同上 |
| (256, 10, 64) TN | L2 dW | 同上 |

結論: train reuse と infer の Metal セルでは、m=64 の 6 形状は `tile::select`／`CANDIDATES` を通らない。ゲートが効くのは `dispatch_auto` の classic 分岐だけで、split-K が先に取る形状と常駐 strided 経路には届かない。

### イシュー前提の訂正

- 本番は `params: None`（occupancy 縮退なし）。(64,256,784) が classic に来た場合の現行選択は `CANDIDATES[2]`（32×64）で `[3]` ではない。
- 「32×32 で 16 TG」は split-K のタイル（`split_k_tile`）に当たる。
- 結合順序を変えない tile.rs 候補を足しても、既定経路の train reuse／infer セルには届かない。

## 3. 型 A 候補（結合順序を変えない。本イシューで実装）

| 名前 | bm×bn | bk | wm,wn | acc（行×列） | 狙い |
|---|---|---|---|---|---|
| T1 `[0]` | 16×32 | 16 | 2,2 | 1×2 | 横長・短辺 64。64×256 で 32 TG（現行 `[2]` は 8 TG） |
| T2 `[1]` | 16×64 | 16 | 2,2 | 1×4 | 横長。16 TG |
| T3 `[2]` | 32×16 | 16 | 2,2 | 2×1 | 縦長・短辺 64。784×64 で 100 TG（現行 `[1]` は 26 TG） |
| T4 `[3]` | 16×16 | 16 | 2,2 | 1×1 | 最大 TG 数 |

ゲート ON 時の写像（`select_m64_thin_with`）: m,n,k が全て 64 以上かつ min(m,n)==64・max(m,n)>=128 のとき、横長は T1、縦長は T3。T2・T4 は明示指定（`dispatch_variant`／`dispatch_strided_tiled_prepared`）で測る。

## 4. bit 一致の確認結果（実機）

- 機体: Apple M4 Max（開発機）、ベース `0ae3f00ff` + 本変更。
- コマンド: `cargo test -p fandhe-ai-backend-metal --release --lib gemm_m64_thin_tile_diag_tests -- --ignored --nocapture --test-threads=1`
- 比較: 非 NaN は `to_bits` 完全一致、NaN は NaN 同士。入力に ±inf・NaN・極端値を混入。
- 結果: NN 44 件（11 形状 × 4 候補。端数形状を含む）pass、NT/TN 48 件（6 形状 × 2 向き × 4 候補。戻り値の解決済み構成が要求した候補と一致することも検査）pass。全候補が型 A として bit 一致。
- 性能は未計測。計測は #3209（Mac mini）。

## 5. 型 B（結線せず所有者へ申し送り）

- B1: 常駐 strided 経路（スカラー `gemm_tiled_bias_act`。逐次 fma）を `gemm_simdgroup_tiled` へ回す。演算方式が変わり bit 一致は保証されず、simdgroup 側の strided 経路には bias／relu の epilogue もない。train／infer セルに効きうる唯一の候補。数値判定方式と結線可否は所有者判断。
- B2: split-K の適用範囲拡張（strided 経路への適用、m=64 向け partitions／タイル変更）。結合順序が変わるため `docs/backend-metal-splitk-parity-judgment-decision.md` の全形状一律 baseline の対象で、baseline 追加は実機実測値のみ・人間承認が必要。

## 6. #3209 への注記

- ゲートは `crates/backend-metal/src/tile.rs` の `pub(crate) const M64_THIN_TILE_ENABLED: bool = false;` の 1 行を書き換える（`TRAIN_FORWARD_ENCODE_ONLY_DEFAULT_ENABLED` と同書式）。
- 上記のとおり既定経路の train reuse／infer には届かないため、純カーネル時間は `MetalGemm::dispatch_variant(GemmVariant::SimdgroupTiled(T))` と `dispatch_strided_tiled_prepared(.., cfg = T)` で候補を明示指定して測る。strided は m,n,k が 8 の倍数・ld が 4 の倍数の形状のみ適格。

## 7. Mac mini A/B（#3209）

**状態: Mac mini 実測済み（正式系列。純カーネル 5 run 全 gate pass・セル A/B 5 round 全 gate pass）。**
機械判定は `docs/perf/logs/metal-m64-thin-tile-ab-macmini-3209/aggregate.md`（以下 `aggregate.md`）をそのまま採用した。
結論の要点は次の 3 つ。

1. 純カーネル（`kernel_gpu`）では、T1〜T4 は規則 11 の機械判定でいずれも ADOPT（ADOPT_CANDIDATE の形状を 1 つ以上持ち、
   bit 一致・parity 失敗なし）。ただし形状ごとの勝ち負けは分かれる（T4 は (784,64,256) と (784,256,64) で +23〜54% 悪化、T2 は 2 形状でのみ改善）。
2. **ゲート `M64_THIN_TILE_ENABLED` の写像判定は undetermined（規則の前提不成立）**。到達追跡で、本番の選択順（split-K → classic）により
   (64,256,784) は split-K が先に取るため、ゲートを ON にしても T1 には到達しない（route は SplitK のまま）。
   ゲート反転で実際に変わる形状は (784,64,256)→T3 の 1 件だけだった。机上の前提（2 件）が食い違ったため RULE 項 5 に従い undetermined とする。
3. train／infer セル A/B は `anomaly`（train の 64/fresh が 5/5 round で after が遅い向き）。ゲート判定は ADOPT にしない（RULE 項 13）。
   checksum は 4 セルとも完全一致。

### 7.1 条件

| 項目 | 値 |
|---|---|
| 機体 | Mac mini（Apple M4・GPU 10 コア〈`gpu_cores=10`・`verified_tile_table=M4Base`〉・P 4 + E 6 コア・16 GB・macOS 27.0.1・rustc 1.99.0）。出典 `env_info.txt`・`aggregate.md` |
| 計測対象リビジョン | `bd2ce9464783edfdc395a923a76d4b88f6529ab3`（`rev-stamp.txt`）。展開ツリー指紋は開発機と Mac mini で一致（`tree-fingerprint.txt`） |
| 規則 | `docs/perf/logs/metal-m64-thin-tile-ab-macmini-3209/RULE.txt`（sha256 `d5bfebcacf27b6b95da47297cc71ba66f2ce3e44a76d3e50c107e8cd26a26e40`。固定コミット `b5da0c2b0f4a080edf183ce7cb7fa6c7eb922d2b`。計測前に単独コミット。本文は変更していない） |
| 判定の根拠 | 純カーネル時間（`kernel_gpu`）のみ。train／infer セルは A/A の対照（record_only） |
| 計測基盤 | `gemm_m64_thin_tile_ab_diag_tests.rs`（到達追跡・CPU 参照 parity・純カーネル計測。全件 `#[ignore]`）・`MetalGemm::diag_encode_strided_tiled`（`#[cfg(test)]`。encode-only）・`orchestrate_macmini.sh`・`aggregate.py`・`run_ab_m64_thin_tile_metal.sh` |

### 7.2 開発機でのスモーク（判定に使わない）

M4 Max（`verified_tile_table=M4Max`）で診断 3 テストを 1 回ずつ実行し、行の書式・assert（1 バッチ・1 ラベル・
`kernel_gpu <= commit_wait`）・CPU 参照 parity（候補 × 8 セルすべて fallback なしで pass）・
`aggregate.py` の正規表現との一致を確認した。M4 Max は Mac mini と性能を直接比較できないため、値は収録しない。

### 7.3 実装上の確認事項

- NN の 8 非整列形状（n=10 または k=10 の 3 形状）は、本番 `dispatch_auto` と同じく 8 の倍数へゼロ詰めした実効次元
  （(64,16,256)・(256,64,16)・(256,16,64)）で計測する（`diag_encode_tiled_nn` が非整列を拒否するため）。
- 到達追跡は実効次元で行う。机上の前提（ゲート ON で変わるのは (64,256,784)→T1 と (784,64,256)→T3 の 2 件）が
  食い違う場合、ゲート判定は undetermined に落ちる（RULE 項 5・13）。**今回は (64,256,784) が split-K 先取りで食い違った**（7.4）。
- `prod` 腕は本番スカラー strided 経路（bias+relu 等）で、カーネルが異なるため判定対象外（record_only）。
  `docs/backend-metal-command-batching-design.md` §7.3.4 の L1 `kernel_gpu` は dinput 位相（別カーネル・別条件）の値であり、
  `prod` 腕の L1 forward とは同条件ではないため、同節への参照行は追加しない。

### 7.4 結果

出典はすべて `docs/perf/logs/metal-m64-thin-tile-ab-macmini-3209/`（以下 `logs/`）。`kernel_gpu` は run 内中央値（ms）の 5 run 中央値で、
run 別の値・副指標（encode／commit_wait／alloc_c）は `logs/aggregate.md` に全件ある（外れ値の除外はしていない）。

| 項目 | 結果 |
|---|---|
| 前提ゲート: 既存 bit 一致 | NN 44 件・NT/TN 48 件の計 92 件 pass（`logs/prereq_bitident.log`、`prereq_bitident_strict rc=0`） |
| 候補別 bit 一致（`BITIDENT`） | 32 件すべて `ok=true`（`logs/bitident_report.log`） |
| 前提ゲート: CPU 参照 parity（統一複合判定） | 40 件すべて `ok=true`・fallback なし（`logs/parity.log`） |
| 正式系列 | はい（`logs/load_gate.log` の全 gate pass・`series done`。`ab/load-gate-*.log` は `series=official` で 5 round 全 pass） |

到達表（`logs/trace_run{1..5}.log`。5 run 全一致）:

| 形状（実効） | ゲート OFF の route | ゲート ON の thin |
|---|---|---|
| 64x256x784 | SplitK（partitions 4） | T1（ただし split-K が先に選ばれるため本番では到達しない） |
| 64x10x256（64x16x256） | SplitK（partitions 8） | none |
| 784x64x256 | Classic C1 | T3 |
| 256x64x10（256x64x16） | Classic C7 | none |
| 784x256x64 | Classic C1 | none |
| 256x10x64（256x16x64） | Classic C7 | none |

`kernel_gpu` 5 run 中央値（ms。比は cur 比。**太字は ADOPT_CANDIDATE＝5/5 run 速い**）:

| 向き | 形状 | cur | T1 | T2 | T3 | T4 | prod（参考） |
|---|---|---|---|---|---|---|---|
| NN | 64x256x784 | 0.0623 | **0.0328（-47.4%）** | **0.0508（-18.5%）** | **0.0341（-45.3%）** | **0.0312（-49.9%）** | 0.0785 |
| NN | 64x10x256 | 0.0110 | 0.0133（+20.9%） | 0.0231（+110.0%） | 0.0125（+13.6%） | 0.0110（+0.0%） | 0.0126 |
| NN | 784x64x256 | 0.0252 | **0.0189（-25.0%）** | **0.0228（-9.5%）** | **0.0204（-19.0%）** | 0.0310（+23.0%） | 0.0694 |
| NN | 256x64x10 | 0.0040 | **0.0039（-2.5%）** | 0.0052（+30.0%） | 0.0042（+5.0%） | 0.0042（+5.0%） | 0.0045 |
| NN | 784x256x64 | 0.0181 | 0.0235（+29.8%） | 0.0239（+32.0%） | 0.0243（+34.3%） | 0.0278（+53.6%） | 0.0703 |
| NN | 256x10x64 | 0.0065 | **0.0062（-4.6%）** | 0.0090（+38.5%） | **0.0063（-3.1%）** | **0.0057（-12.3%）** | 0.0056 |
| NT（実向き strided） | 784x64x256 | 0.0255 | **0.0188（-26.3%）** | **0.0227（-11.0%）** | **0.0205（-19.6%）** | 0.0303（+18.8%） | 0.0700 |
| TN（実向き strided） | 784x256x64 | 0.0190 | 0.0243（+27.9%） | 0.0240（+26.3%） | 0.0259（+36.3%） | 0.0285（+50.0%） | 0.0711 |

判定（RULE 項 11・12。機械判定の採用。出典 `logs/aggregate.md`）:

| 対象 | 判定 |
|---|---|
| T1 | ADOPT |
| T2 | ADOPT |
| T3 | ADOPT |
| T4 | ADOPT |
| ゲート（`M64_THIN_TILE_ENABLED` の写像） | **undetermined**（到達前提不成立・セル A/B anomaly） |

注意: 候補の総合 ADOPT は「規則 11 の機械判定」であり、形状別に悪化する候補を全形状に適用してよいという意味ではない。
結線するなら形状ごとの写像（勝つ形状のみ）が必要で、それは別 PR・所有者判断になる（7.5）。

train／infer セル A/B（`logs/ab/compare-{train,infer}-3209-metal-3209.md`。before = main、after = ゲート ON の 1 行差。`--cell-ab anomaly`）:

| セル | before 中央値 | after 中央値 | after/before | checksum | 5 round の向き |
|---|---|---|---|---|---|
| train 64/fresh | 2.325 ms | 2.347 ms | 1.0093 | 完全一致 | 全 round で after が遅い（5/5）→ anomaly |
| train 64/reuse | 1.881 ms | 1.878 ms | 0.9984 | 完全一致 | 混在 |
| infer 64/fresh | 754.9 us | 762.5 us | 1.0100 | 完全一致 | 混在 |
| infer 64/reuse | 366.1 us | 374.4 us | 1.0229 | 完全一致 | 混在（0.73〜1.48 の大きなばらつき） |

before／after のバイナリ sha256 は異なり（`logs/ab/sha-3209-*.txt`）、`fandhe-ai` は各 checkout の `crates/facade` に解決している（`logs/ab/tree-3209-*.txt`）。
anomaly は RULE 項 13 の字義どおり（5/5 round 一貫）。差は 0.9% で、ゲート ON で本番が変えるのは (784,64,256) の 1 形状のみ
（かつ train reuse／infer の Metal セルは `tile::select` を通らずスカラー strided カーネルで動く。7.3・`docs/perf/metal-gemm-m64-thin-tile-candidates.md` §2）なので、
因果は未確認で、ノイズ帯として扱う。

### 7.5 所有者への申し送り

- ADOPT 候補（T1〜T4）の結線（`M64_THIN_TILE_ENABLED` の反転や選択表への追加）は、結合順序が変わらない候補に限って別 PR で行う。
  今回の実測では、split-K が先取りする (64,256,784) には写像が効かないこと、候補によっては形状別に +20〜54% 悪化することから、
  「ゲートを丸ごと ON」は推奨しない。勝つ形状だけを写す形（例: (784,64,256)→T1／T3）にするなら別 PR で設計する。
  いずれもゲートの写像は機体非依存（M4 Max にも効く）ため、結線の前に M4 Max での確認が要る。
- (64,256,784) は split-K（partitions 4）が選ばれるため、細タイルの効果を得るには split-K 側の拡張（型 B2）が要る。
- 型 B（B1: 常駐 strided のスカラー経路の simdgroup 化、B2: split-K の拡張）は結合順序・演算方式が変わるため、
  引き続き所有者判断（baseline の追加は実機実測値のみ・人間承認）。
- `docs/backend-metal-command-batching-design.md` §7.3.4 への反映は不要と判断した（7.3 参照）。
