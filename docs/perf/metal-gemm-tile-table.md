# Metal GEMM タイル選択テーブルの形状・転置別チューニング（イシュー #1039）

親 #1037（タイル構成のテーブル駆動選択）配下。兄弟 #1038（PR #1074）・#1040（PR #1076）の後続。

## 1. 計測環境

- 機種: MacBook Pro（Apple M4 Max・GPU 40 コア・統合メモリ 64GB）
- OS: macOS（Darwin 25.6.0・arm64）
- 実行コマンド: `cargo run -p fandhe-ai-backend-metal --example gemm_transpose_tile_sweep --release`
- 計測プロトコル: `bench_harness::protocol::run` + `MeasurementConfig::default()`（warmup 20 回・計測 20 回・中央値）・決定的シード `0xC0FFEE`（`docs/perf/metal-bench-noise-protocol.md` 準拠の一般規約。interleaved A/B・ラウンド交互までは適用していない。`gemm_tile_sweep.rs`〈#1036〉と同一水準のプロトコル）
- 2 回計測（別プロセス実行。生ログ: `docs/perf/logs/metal-gemm-tile-table-1039/sweep_run1.log`〈run1〉・`sweep_run2.log`〈run2〉）。プロセス間変動は `docs/perf/metal-bench-noise-protocol.md` が指摘する既知の系統誤差源のため、順位が 2 回で一致した点のみ「確定」として `tile.rs` へ反映する

## 2. 候補ラベル対応表

`crate::tile::CANDIDATES`（`pub(crate)`。`tile.rs` 定義順）:

| ラベル | bm×bn×bk | wm×wn | staged |
|---|---|---|---|
| cand0_64x64_wm2wn2 | 64×64×16 | 2×2 | true |
| cand1_64x32_wm2wn2_tall | 64×32×16 | 2×2 | true |
| cand2_32x64_wm2wn2_wide | 32×64×16 | 2×2 | true |
| cand3_32x32_wm2wn2 | 32×32×16 | 2×2 | true |
| cand4_64x64_wm1wn2 | 64×64×16 | 1×2 | true |
| cand5_64x32x32_wm2wn2 | 64×32×32 | 2×2 | true |
| cand6_64x32x8_wm4wn1 | 64×32×8 | 4×1 | true |
| cand7_single_simdgroup_8x8 | 8×8×8 | 1×1 | false |

## 3. NN 経路: 形状別 GFLOPS/s（TFLOPS。中央値）

`dispatch_tiled_prepared`（転送非計測）で `CANDIDATES` 全 8 候補を明示指定して比較。**太字**は各形状の最良候補（run1/run2 で順位一致）。

| shape (m,n,k) | cand0 | cand1 | cand2 | cand3 | cand4 | cand5 | cand6 | cand7 |
|---|---|---|---|---|---|---|---|---|
| (512,512,512) | 0.237/0.234 | 0.707/0.755 | 0.688/0.722 | 0.736/0.885 | 0.303/0.329 | **1.024/1.374** | 0.993/1.298 | 0.687/0.831 |
| (1024,1024,1024) | 0.888/0.940 | 4.631/4.655 | 5.013/4.517 | 5.648/4.637 | 0.837/0.794 | 5.092/4.901 | **5.770/5.199** | 2.138/2.078 |
| (2048,2048,2048) | 1.217/1.216 | **9.456/9.635** | 9.182/9.310 | 8.849/8.998 | 0.504/0.516 | 8.285/8.407 | 8.821/8.864 | 2.677/2.692 |
| (4096,4096,4096) | 1.219/1.217 | 8.902/8.842 | **9.905/9.758** | 8.357/8.237 | 0.491/0.492 | 7.819/7.718 | 7.775/7.564 | 2.774/2.614 |
| (2048,2048,64) | 0.665/0.699 | 2.271/1.990 | 2.037/1.999 | 2.051/1.980 | 0.421/0.421 | 2.057/1.988 | 2.039/**2.119** | 1.294/1.309 |
| (2048,2048,512) | 1.164/1.182 | **7.265**/7.074 | 6.615/6.216 | 6.539/6.224 | 0.512/0.513 | 6.935/6.388 | 7.508/6.514 | 2.443/2.261 |
| (1536,1024,1024) | 1.180/1.164 | **6.354/6.294** | 6.098/5.883 | 6.017/5.907 | 0.498/0.500 | 5.882/5.892 | 6.046/6.091 | 2.318/2.250 |
| (1024,1536,1536) | 1.207/1.189 | 6.815/6.734 | 6.680/**6.874** | 6.606/6.502 | 0.511/0.478 | 6.427/6.592 | 6.893/6.540 | 2.501/2.315 |
| (4096,1024,1024) | 1.173/1.185 | **8.219/7.955** | 7.420/7.167 | 7.105/6.728 | 0.510/0.511 | 7.511/7.299 | 7.644/7.141 | 2.425/2.273 |
| (1024,4096,1024) | 1.226/1.230 | 8.260/**7.900** | 7.837/7.289 | 7.466/6.968 | 0.509/0.513 | **8.265**/7.771 | 7.727/7.441 | 2.422/2.264 |

- 全形状 2 回計測（run1/run2。生ログ `sweep_run1.log`／`sweep_run2.log`）。**太字**は各形状・各 run で最良値だった候補。`(2048,2048,64)`・`(2048,2048,512)`・`(1024,1536,1536)` は run1 と run2 で最良候補（太字）が異なり、順位が入れ替わっている（プロセス間変動。§5「順位不安定のため反映しない」節参照）
- (1024,4096,1024)（横長）は run1 で cand5 が最良（8.265 対 cand2 7.837）だが run2 では cand1 が最良（7.900 対 cand5 7.771）で順位が入れ替わる。差が数% 台でプロセス間変動の範囲内と判断し、`tile.rs` 側の閾値は変更しない（現行 `CANDIDATES[2]` を維持）

## 4. NT/TN/TT 経路: strided classic tiled（`dispatch_strided_bias_act_prepared`。タイル variant なし）

| shape (m,n,k) | NT | TN | TT | 参考: NN 最良候補 |
|---|---|---|---|---|
| (512,512,512) | 0.562 | 0.530 | 0.560 | 1.374（cand5） |
| (1024,1024,1024) | 1.444 | 1.471 | 1.464 | 5.770（cand6） |
| (2048,2048,2048) | 1.694 | 1.681 | 1.677 | 9.635（cand1） |
| (4096,4096,4096) | 1.689 | 1.621 | 1.525 | 9.905（cand2） |
| (2048,2048,64) | 0.995 | 1.004 | 0.993 | 2.271（cand1） |
| (2048,2048,512) | 1.486 | 1.496 | 1.503 | 7.508（cand6） |
| (1536,1024,1024) | 1.440 | 1.435 | 1.417 | 6.354（cand1） |
| (1024,1536,1536) | 1.486 | 1.483 | 1.467 | 6.893（cand6） |
| (4096,1024,1024) | 1.484 | 1.535 | 1.471 | 8.219（cand1） |
| (1024,4096,1024) | 1.491 | 1.481 | 1.452 | 8.265（cand5） |

NT/TN/TT はいずれも NN 最良候補の概ね 15〜25%（大形状ほど低い比率。4096 立方では約 15%）に留まる。strided classic tiled 経路は `gemm_simdgroup_tiled` のタイル variant・staged 協調ロードを経由しないため、この差は転置ロードの最適化余地（simdgroup タイル化・staged 協調ロードの導入）を示唆する。

## 5. 判断: `tile::select_with_occupancy` への反映

**NN（正方立方・準正方長方形の実測点）**: 2 回計測いずれも最良候補の順位が一致した以下の厳密一致 `(m, n, k)` タプルに限り、`tile.rs` 側の `shape_cfg`（`select_with_occupancy` の段 1・形状判定の出力）を測定済み最良候補へ差し替えるよう更新した（`fn select_with_occupancy` 冒頭の厳密一致テーブル）。既存の occupancy 縮退判定（段 2。`params` が `Some` の場合）はこの差し替え後も迂回しない構造にした（P1・codex-review 指摘・PR #1108 レビュー対応。`select()` 経由〈`params: None`〉では従来どおり縮退せず `shape_cfg` を返すため本番ディスパッチの挙動は変わらない）。実測範囲外への無根拠拡張はしない（#744/PR #760 と同一判断軸）。

**機種ゲート（P1・codex-review 指摘・PR #1108 レビュー再指摘対応で最終形へ是正）**: 本テーブルは M4 Max（40 コア構成）実機実測のみが根拠であり、既存の公開 API（PR #1108 以前から存在する `select(m, n, k)`・`select_with_occupancy(m, n, k, params)`）はデバイス情報を受け取らないため、無対応のままでは M1〜M5 を含む全 Apple Silicon 機種へ無条件適用されてしまう（`AGENTS.md`「実機固有値をロジックへ直書きしない」規約への抵触）。**公開 API 互換性を保つため、既存の `select`／`select_with_occupancy` はシグネチャを変えず維持し**、デバイス情報を受け取る別名 `select_for_device(m, n, k, gpu_core_count)`／`select_with_occupancy_for_device(m, n, k, gpu_core_count, params)` を新設した。

初回是正では `gpu_core_count: Option<u32>` 引数を追加し `Some(40)`（GPU コア数一致のみ）で厳密一致テーブルを評価していたが、GPU コア数だけでは機種を一意に識別できない（M3 Max にも 40 コア構成が存在する）ため、codex-review の再指摘（PR #1108 レビュー）を受けて以下へ是正した:

- `tile::verify_m4_max(gpu_core_count: Option<u32>, soc_brand: Option<&str>) -> Option<VerifiedM4MaxGpuCoreCount>`: GPU コア数（`crate::device::probe_gpu_core_count` の IOKit 実測値）と SoC ブランド文字列（`crate::device::probe_soc_brand_string` の実測値。例: `"Apple M4 Max"`）の**両方**が一致した場合にのみ、検証済みであることを表す opaque 型 [`VerifiedM4MaxGpuCoreCount`] を返す
- `select_for_device`／`select_with_occupancy_for_device` の `gpu_core_count` 引数の型を生の `Option<u32>` から `Option<VerifiedM4MaxGpuCoreCount>` へ変更した。`VerifiedM4MaxGpuCoreCount` はフィールド非公開で `verify_m4_max` からのみ構築可能なため、未検証の GPU コア数（生の `u32`）を渡してブランド照合を迂回することはコンパイル時に不可能になっている
- 呼び出し元（`crate::gemm::MetalGemm::dispatch_auto` 等）は `MetalContext::verified_m4_max_gpu_core_count()`（`MetalContext::new` が 1 回だけキャッシュした GPU コア数・SoC ブランド文字列を `verify_m4_max` へ渡す薄いラッパー）の戻り値をそのまま渡す

コア数・ブランドのいずれかが不一致・取得不能（`None`）な機種は本テーブルを経由せず、既存の形状クラス判定（縦長・横長・正方立方・大形状フォールバック）のみへ従来どおり流れる。回帰テストは `tile::tests::select_exact_match_table_is_gated_by_m4_max_gpu_core_count`（機種ゲート自体の固定）・`tile::tests::verify_m4_max_requires_both_gpu_core_count_and_soc_brand_to_match`（GPU コア数・SoC ブランドの両方一致が必須であることの固定）。他機種（M1〜M5・M4 Max 以外のコア数構成）での候補の優劣は引き続き未実測のため、§6「スコープ外」の「M4 Max 以外の Apple Silicon 別テーブル」の解消を待たずにテーブルを機種非依存へ拡張することはしない。

| (m,n,k) | 旧選択（#744/PR #760 是正後） | 新選択（#1039） | 改善率（最良候補比） |
|---|---|---|---|
| (512,512,512) | CANDIDATES[3]（0.736/0.885） | CANDIDATES[5]（1.024/1.374） | 約 1.39〜1.55 倍 |
| (1024,1024,1024) | CANDIDATES[3]（5.648/4.637） | CANDIDATES[6]（5.770/5.199） | 約 1.02〜1.12 倍 |
| (2048,2048,2048) | CANDIDATES[3]（8.849/8.998） | CANDIDATES[1]（9.456/9.635） | 約 1.07 倍 |
| (4096,4096,4096) | CANDIDATES[3]（8.357/8.237） | CANDIDATES[2]（9.905/9.758） | 約 1.19 倍 |
| (1536,1024,1024) | CANDIDATES[0]（1.180/1.164） | CANDIDATES[1]（6.354/6.294） | 約 5.35〜5.41 倍 |

`m == n == k` の #744（2026-08-19）実測時点で確定させた「CANDIDATES[3] 一律選択」がもはや最良候補ではなくなっている（準正方長方形は #744/PR #760 時点で `CANDIDATES[0]` 安全側フォールバックのまま未実測だった）。逆転の原因は #1038〜#1040 の staged 経路変更（タイル variant 群の整理・転置 stride 対応境界確立等。詳細な原因切り分けは本イシューのスコープ外）と推定される。

**順位不安定のため反映しない（P1・codex-review 指摘・cursor[bot] 指摘・PR #1108 レビュー対応）**: `m == n` だが `k != m`（K 未実測の正方出力）の `(2048,2048,64)`・`(2048,2048,512)`、および準正方長方形 `(1024,1536,1536)` は、2 回計測で最良候補の順位が入れ替わった（プロセス間変動。§3 表参照）ため、「2 回一致した点のみ反映する」方針（§1）に従い厳密一致テーブルへは含めない。いずれも `CANDIDATES[0]`（旧選択・#744/PR #760 是正前の安全側フォールバック）のまま据え置く。

| (m,n,k) | CANDIDATES[0]（据え置き） | run1 最良 | run2 最良 |
|---|---|---|---|
| (2048,2048,64) | 0.665/0.699 | CANDIDATES[1]（2.271） | CANDIDATES[6]（2.119） |
| (2048,2048,512) | 1.164/1.182 | CANDIDATES[6]（7.508） | CANDIDATES[1]（7.074） |
| (1024,1536,1536) | 1.207/1.189 | CANDIDATES[6]（6.893） | CANDIDATES[2]（6.874） |

両 run とも `CANDIDATES[0]` 比では cand1/cand2/cand6 いずれも約 3〜6 倍高いスループットであり、`CANDIDATES[0]` 据え置きが最良候補ではないことは両 run で一致している。一方 cand1/cand2/cand6 間の順位（数% 台の差）はプロセス間変動で入れ替わるため、現時点でどれを厳密一致テーブルへ採用するかの根拠にはできない。再計測での順位確認は `out-of-scope-tracking.md` の規約に沿って別イシューで追跡する。

**縦長・横長の一般式（`m >= 2n` → CANDIDATES[1]、`n >= 2m` → CANDIDATES[2]）**: 実測点 (4096,1024,1024)（縦長）は現行選択 CANDIDATES[1] と一致（変更不要）。(1024,4096,1024)（横長）は run1/run2 で最良候補の順位が入れ替わる（cand5 と cand1 が僅差）ため、現行選択 CANDIDATES[2] からの変更根拠としては不十分と判断し変更しない。

**NT/TN/TT（strided classic tiled）**: タイル variant を持たないため `tile::select_with_occupancy` の選択対象外（#1040 確定構成のまま）。本イシューでは NN 最良値との差分定量化に留め、`gemm_simdgroup_tiled` への転置ロード拡張の要否判断材料として §4 の表を記録する（実装自体は親 #1037 系の別イシューへ引き継ぐ。§6「スコープ外」参照）。

**#1143 追記**: N=4096 カーネル純境界の candle 比ギャップ縮小調査（イシュー
#1143）で `CANDIDATES` へ index 8（`(32,64,16,1,2)`。MLX steel classic 経路の
未収録構成）を追加し、N=4096 NN で cand2（現行選択）と比較した。比較対象の
cand2 実測値は本 PR（#1143）自身の計測ではなく、本表 §3・§5（イシュー #1039・
上記「1. 計測環境」の 2 回計測。`sweep_run{1,2}.log`）に記録済みの 9.758〜
9.905 TFLOPS（本表 37/55 行目）を指す。#1143 自身の実測（`docs/perf/metal-gemm-n4096-kernel-gap.md`
§3）では cand2 は run1=9.6479／run2=7.6978 TFLOPS（約 2 TFLOPS の run 間ばら
つき）であり、両者は別々の計測キャンペーンの値であるため単純に一つの値域とし
て扱わない。cand8 は 0.96〜0.97 TFLOPS と大幅に劣後し、他 9 形状 × 4 パターン
でも一度も最良候補にならなかった（`docs/perf/metal-gemm-n4096-kernel-gap.md` §3・
生ログ `logs/metal-gemm-n4096-kernel-gap-1143/sweep_run{1,2}.log`）。cand8 劣後
という結論自体は両キャンペーンのいずれの値を基準にしても変わらない。
`select_with_occupancy_for_device` の選択（本表 §5 の判断）は変更しない。

**#1329 追記**: E7（親 #1324）の sub-issue として `CANDIDATES` へ index 9
（`(64,64,32,2,2)`。`CANDIDATES[0]` の bk=32 版）を追加した。本 issue は
候補追加と正確性（parity）確認のみを対象とし、性能値（純カーネル時間
の before/after・本表 §3/§5 相当の TFLOPS 計測）は未計測のまま
（`select_with_occupancy_for_device` の選択ロジック・本表 §5 の判断は
変更しない）。性能実測・`tile::select` への組み込み判断は後続イシュー
#1330 へ引き継ぐ（詳細は `docs/perf/metal-gemm-n4096-kernel-gap.md`
§13）。

**#1331 追記**: E8（親 #1325）の sub-issue として `CANDIDATES` へ index 10
（`(128,64,16,2,2)`。`CANDIDATES[0]` の bm=128 版）を追加した。本 issue も
候補追加と正確性（parity）確認のみを対象とし、性能値（純カーネル時間
の before/after・本表 §3/§5 相当の TFLOPS 計測）は未計測のまま
（`select_with_occupancy_for_device` の選択ロジック・本表 §5 の判断は
変更しない）。性能実測・`tile::select` への組み込み判断は後続イシュー
#1332 へ引き継ぐ（詳細は `docs/perf/metal-gemm-n4096-kernel-gap.md`
§15）。

## 6. スコープ外（計画 §7 を踏襲）

- `gemm_simdgroup_tiled` への転置ロード導入（NT/TN/TT へのタイル variant 適用）: §4 の定量化を判断材料として親 #1037 系の別イシューへ引き継ぐ
- 親 #1037 の受け入れ条件（N=4096 で candle Metal 5,040 GFLOPS 超え）の達成判定自体: 本イシュー（#1039。本表 §3・§5・37/55 行目、`sweep_run{1,2}.log`。#1143 自身の計測ではない）の実測値（NN 4096 立方で 9.76〜9.91 TFLOPS）は判断材料として記録するが、達成ゲート判定は親側で行う
- M4 Max 以外の Apple Silicon 別テーブル・DGX Spark 側作業
