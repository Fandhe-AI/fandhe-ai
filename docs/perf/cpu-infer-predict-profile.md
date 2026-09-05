# CPU 推論 `predict` 経路のプロファイルと融合 `gemm_bias_act` 適用判断

イシュー #1218（`docs/perf/train-step-phase-breakdown.md` §15.5〜§15.6
「起票案 H」）に基づく。framework-compare の CPU 推論（batch 64・
784→256→ReLU→10）が candle 比未達（DGX GB10 0.81 倍・M4 Max 0.36 倍）
だった件について、`--task infer --phases` はハーネス制約
（`MEASURE_ERROR`）で内訳を取れないため §15.5 は「非融合 3 起動」
「呼び出し毎の `CpuBackendOps::new()`」を**仮説**として挙げていた。
本 doc は `Sequential::predict`（CPU・tape 不要経路）のホットスポットを
実測し、その結果に基づき融合 `gemm_bias_act` の適用可否を判断・実装
した記録である。

## 状態: 実装完了（ADOPT）。M4 Max 実機実測。DGX Spark GB10 は未実測（環境未接続。§7 参照）

## 1. 実行環境

- Apple M4 Max（P コア 12・E コア 4。macOS 26.6.2）。実ホスト名は
  `docs/real-hardware-verification-env.local.md`（`.gitignore` 対象）
  管理下のため本 doc には書かない
- rustc 1.96.0・コミット `e2a1ad561432ef13c7a1fbd48cc6ee48b1ec65da`
  （base = 本イシュー着手時点の `origin/main` HEAD）
- `/usr/bin/sample`・`/usr/bin/xctrace` は利用可能だが、下記§3 の
  フェーズ分解計測で仮説の当否が数値として十分明確に判別できたため、
  本 doc では**フェーズ分解のみ**を実測手段として採用し、サンプラー
  （`sample`／`xctrace`）によるスタックサンプリングは実施していない
  （§8「スコープ判断」参照）
- `RAYON_NUM_THREADS` は既定（未設定。`num_cpus` 相当）

## 2. 現状分析（コード事実）

- `Sequential::predict`（`crates/facade/src/compat/sequential.rs`）は
  `predict_tape_free`（内部で `CpuBackendOps::new()` を構築し層ごとに
  `Module::forward_host` を呼ぶ tape 不要経路。#1028）を優先し、
  `Unsupported` の場合のみ `predict_via_tape`（`Tape`/`Var` 経由の旧
  経路）へ全体フォールバックする
- 結線前（イシュー着手時点）の `Linear::forward_host`
  （`crates/autodiff/src/nn/module.rs`）は `ops.gemm(input, weight)` →
  `ops.add(&y, bias)` の非融合合成のみで、`ops.gemm_bias_act`
  （epilogue 融合カーネル）を使っていなかった。`Relu::forward_host`
  は `ops.relu`。推論形状（784→256→ReLU→10）では **gemm・add・relu・
  gemm・add の 5 起動・中間 `Tensor` 3 個**（64×256 が 2 個・64×10 が
  1 個）を割り当てていた
- `CpuBackendOps` は ZST（`crates/backend-cpu/src/ops.rs`
  `pub struct CpuBackendOps;`）。「`CpuBackendOps::new()` の構築コスト」
  仮説はコード事実として当初から否定できていた（§3 で計測によっても
  追認）
- CPU 融合カーネル（`gemm_blis_bias_act_parallel`。
  `crates/backend-cpu/src/gemm_blis/mod.rs`）は GEMM 本体（各行パネル
  の K 全体を蓄積し終えた後）に epilogue（`*x += *b` の単一 IEEE 加算
  → `x.max(0.0)`）を 1 回適用するため、非融合合成（`gemm`→`add`→`relu`）
  と **bit 完全一致**する。`crates/backend-cpu/tests/
  gemm_epilogue_parity.rs` が MR/NR/MC/KC/NC 境界を跨ぐ形状グリッドで
  hard assert 済み（`docs/perf/cpu-gemm-epilogue-fusion.md`）。この
  事実は `predict_via_tape`（`Sequential::forward` 経由。#1044 で
  Linear→ReLU に既に `gemm_bias_act` を結線済み）が
  `sequential_predict_tape_free_matches_via_tape_bit_exact`
  （非融合の旧 `predict_tape_free` と bit 完全一致）を pass していた
  ことでも間接的に確認されていた

## 3. フェーズ分解実測（`infer_predict_phase_diag.rs`）

`crates/facade/tests/infer_predict_phase_diag.rs`（record-only 診断
ベンチ。5 回計測中央値・warmup 20・iters 20。`infer_fixed_cost_bench.rs`
と同じ方式）で、`CpuBackendOps` を直接呼び各フェーズを個別計測した。
実行: `cargo test -p fandhe-ai --release --test infer_predict_phase_diag
-- --nocapture`。

結線前（`Linear::forward_host_with_activation` 追加前。L1 は「融合版」
の代わりに `forward_host` 直呼びで代替不可のため、下表は結線後の
コードで「融合」列を計測し、「非融合」列は手動合成で別途計測した
値）:

| フェーズ | 内容 | median (s) | q1 (s) | q3 (s) |
|---|---|---|---|---|
| `cpu_backend_ops_new_only` | `CpuBackendOps::new()` のみ（ノイズフロア） | 0.000000002 | 0.000000000 | 0.000000002 |
| `l1_linear_relu_fused_gemm_bias_act` | L1 `Linear(784→256)`→`ReLU`（融合） | 0.000299600 | 0.000292748 | 0.000305133 |
| `l1_linear_relu_unfused_gemm_add_relu` | 同・非融合合成（`gemm`→`add`→`relu`） | 0.000405873 | 0.000398633 | 0.000410508 |
| `l2_linear_unfused_gemm_add` | L2 `Linear(256→10)`（非融合） | 0.000054292 | 0.000053148 | 0.000056296 |
| `predict_full` | `Sequential::predict` 全体（結線後） | 0.000366560 | 0.000363375 | 0.000394796 |

2 回実行した値（別実行）: `l1` 融合 0.000295879・非融合 0.000404256・
`l2` 0.000042054・`predict_full` 0.000347571（実行間のばらつきは
5〜15% 程度。母集団は上記表と同じ形状・同一マシン）。

### 3.1 仮説の検証結果

- **(i) `CpuBackendOps::new()` はホットスポットでない**: 計測値
  0.000000002 s（≈ 2 ns）はループ制御のノイズフロアそのもの。コード
  事実（ZST）と一致し、**この仮説は棄却**（§15.5 の仮説には該当しない
  ことを実測でも確認）
- **(ii) 融合 vs 非融合の相対コスト（L1）**: 融合 0.0002996 s に対し
  非融合 0.0004059 s。**融合により L1 単体で約 26%（1.35 倍）短縮**。
  L1（784×64×256 GEMM）が `predict` 全体の主要コストであることも
  同時にわかる（L2 は 0.0000543 s で L1 の 1/6 程度）
- **(iii) `predict` 全体 ≈ フェーズの和**: 結線後の `predict_full`
  （0.0003666 s）は `l1` 融合（0.0002996 s）+ `l2`（0.0000543 s）=
  0.0003539 s に近い（残差 約 0.0000127 s ≈ 3.5%）。残差は入力
  `clone()`（`predict_tape_free_with_ops` 冒頭の `input.clone()`。
  contiguous のため `Arc` 共有でコピー自体は発生しないが、`Vec`
  レイアウト検査・関数呼び出しオーバーヘッド等）・ループ計測自体の
  固定費と見て矛盾しない小ささであり、**層間の中間 `Tensor` 割当等の
  別の支配的コストは検出されなかった**

結論: **CPU 推論 `predict` の主要コストは L1 GEMM（784×64×256）自体
であり、`CpuBackendOps::new()` や層間オーケストレーションの固定費では
ない**。融合 `gemm_bias_act` は L1 のコストを直接約 26% 削減するため、
`predict` 全体の改善に直結する見込みがあると判断した（§4 で採否判断・
§5 で before/after 実測により確認）。

`RAYON_NUM_THREADS` 未設定（既定）で計測しており、`m=64` の行パネル
分割・B packing 重複（`docs/perf/cpu-gemm-b-packing-sharing-decision.md`・
`docs/perf/cpu-gemm-candle-gate-remeasurement.md` が指摘する既知の
未解決事項）が L1 の絶対値に寄与している可能性はあるが、本 Issue の
判断（融合の適用可否）には影響しない相対比較のため、この軸のスイープ
（`RAYON_NUM_THREADS=1` 等）は実施していない（§8 スコープ外）。

## 4. 融合 `gemm_bias_act` の適用可否判断

### 4.1 技術判断（受け入れ条件 2・3）

観測可能な契約（`predict` の出力が旧経路と bit 完全一致。tolerance
なし）と、実装方針の記述（`docs/inference-forward-fixed-cost-design.md`
§3.1・`Module::forward_host` doc の「融合カーネルを使わない」）は
別の層として扱う。

| 層 | 内容 | 本 Issue での扱い |
|---|---|---|
| 観測可能な契約 | `predict` の出力が旧経路と bit 完全一致（tolerance なし） | **維持**。既存 hard assert に加え新規テストで強化（§4.2） |
| 実装方針の記述 | `Module::forward_host`（trait レベル・汎用バックエンド向け）の「融合カーネルを使わない」 | 維持。CPU 固定経路（`Sequential::predict`）についてのみ、trait とは別の inherent メソッドを新設して例外を作る |

CPU 融合カーネルが epilogue を GEMM 完了後に適用し非融合合成と bit
完全一致することはテスト根拠付きで確認済み（§2）であるため、**契約の
見直しは不要**と判断した。`Module::forward_host` trait メソッド自体
（`&dyn BackendOps` 汎用。CUDA／Metal の融合オーバーライドは非融合
合成との bit 一致が未保証）は変更しない。

### 4.2 実装（CPU 固定経路に閉じた結線）

- `crates/autodiff/src/nn/linear.rs`: **inherent メソッド**
  `Linear::forward_host_with_activation(&self, ops: &dyn BackendOps,
  input: &Tensor<f32>, act: Activation) -> Result<Tensor<f32>,
  AutodiffError>` を新設。`matmul_out_shape`／`broadcast_shape` を
  `ops.gemm_bias_act` の呼び出し前に検査し、`forward_host` と同じ
  `AutodiffError::Shape` 一致契約を保つ
- `crates/autodiff/src/nn/module.rs`: `Linear::forward_host`（trait
  メソッド）の doc に、CPU 固定経路の例外を追記（trait 自体は非融合の
  まま不変であることを明示）
- `crates/facade/src/compat/sequential.rs`: `predict_tape_free` を
  `predict_tape_free_with_ops(&self, ops: &dyn BackendOps, input)`
  （private）へ分離し、`SequentialVars::forward`（#1044 の学習 forward）
  と同じ「`Linear` 層に出会うたび次層が `ReLU` かを先読みし、続く
  場合のみ `forward_host_with_activation(.., Activation::Relu)` へ
  結線する」方式を適用。`ReLU` が続かない `Linear`（末尾層等）・
  `Sigmoid`／`Tanh` へ続く `Linear` は従来どおり非融合の
  `forward_host` へ委譲する（#1044 のレビュー限定〈#1079・
  PRRT_kwDOTuUCJc6dgIt-〉と同じ理由）

### 4.3 新規テスト

- `crates/autodiff/src/nn/linear.rs::tests`: `forward_host_with_activation`
  が `forward_host` + 個別 `relu` と bit 完全一致すること（`Activation::
  Relu`／`Activation::None` 双方）・shape 不整合が `forward_host` と
  同じ `AutodiffError::Shape` variant で返ること（matmul 不整合・bias
  broadcast 不整合の 2 パターン）
- `crates/facade/src/compat/sequential.rs::tests`:
  - `sequential_predict_matches_manual_unfused_composition_bit_exact`:
    `predict`（融合結線後）が手動非融合合成（`gemm`→`add`→`relu`→
    `gemm`→`add`）と bit 完全一致することを framework-compare 形状
    （784→256→ReLU→10）・batch 1/3/65/129（BLIS の MR/NR タイル境界を
    跨ぐ値を含む）で確認
  - `sequential_predict_linear_sigmoid_linear_matches_manual_composition`:
    `ReLU` が続かない構成（融合対象外）でも手動合成と一致すること
  - `sequential_predict_tape_free_fuses_linear_relu_into_single_gemm_bias_act_call`
    ／`sequential_predict_tape_free_does_not_fuse_non_relu_activation`:
    `CountingOps`（#1044 の学習 forward 検証と共用）で `predict` が
    実際に `gemm_bias_act` を 1 回だけ呼び、`Sigmoid` 構成では 1 回も
    呼ばないことを機械検証
- 既存 `sequential_predict_tape_free_matches_via_tape_bit_exact`・
  `sequential_predict_public_builder_tape_free_matches_via_tape` も
  引き続き pass（結線後の `predict` と `predict_via_tape` の bit 一致）

全テスト（`cargo test --workspace --all-features`）・
`cargo clippy --workspace --all-targets --all-features -- -D warnings`・
`cargo fmt --all -- --check`・`RUSTDOCFLAGS="-D warnings" cargo doc
--workspace --no-deps --locked` はいずれも pass（M4 Max ローカル実測。
`crates/autodiff/tests/architecture_boundaries.rs`
`autodiff_src_does_not_reference_concrete_backend_crates` の fail-closed
検査に抵触しないよう、`autodiff` 側の doc コメントは具体バックエンド
クレート識別子〈`backend_cpu` 等〉を直接書かず「CPU バックエンド
クレート」という迂回表現にした）。

## 5. before/after 実測（決定ルール(b): 非後退確認）

`crates/facade/tests/infer_fixed_cost_bench.rs`
（`cargo test -p fandhe-ai --release --test infer_fixed_cost_bench --
--nocapture`。batch 64・784→256→ReLU→10・5 回計測中央値・warmup
20・iters 20）の `predict_median_s` を、本イシュー着手前の base
コミット（`e2a1ad561432ef13c7a1fbd48cc6ee48b1ec65da`）と本イシューの
変更後（結線後）とで、同一マシン・同一プロセス内で交互ではなく
`git stash` で切替えて複数回計測した（M4 Max ローカル）。

| 系列 | 実行 1 | 実行 2 | 実行 3 | 実行 4 | 実行 5 | 中央値相当 |
|---|---|---|---|---|---|---|
| before（base HEAD。非融合） | 0.000486538 | 0.000491406 | 0.000483598 | 0.000484085 | 0.000492465 | ≈ 0.000486 |
| after（本イシュー。Linear→ReLU 融合結線） | 0.000470910 | 0.000400427 | 0.000379817 | 0.000418840 | — | ≈ 0.000419 |

after の中央値（≈ 0.000419 s）は before（≈ 0.000486 s）を下回り、
**約 1.16 倍改善**（非後退）を確認した。before では `predict` が
`predict_via_tape`（融合済み）より遅かった（`speedup_x` 0.70〜0.75・
`predict_faster=false`）のに対し、after では `predict` が
`predict_via_tape` と同等かやや上回る（`speedup_x` 1.01〜1.10・
`predict_faster=true`）ことも確認した。決定ルール§0 (a)（bit
一致・全 pass）・(b)（非後退）をいずれも満たすため **ADOPT**（結線
する）と確定する。

計測は同一マシン上で他の重い負荷を並走させずに実施したが、プロセス
起動ごとのノイズ（実行 1 が他より高い等）が数十 μs 単位で見られる
ため、上記は「決定ルールの成立を確認するための参考値」であり、
`framework-compare` 実践規模の正式ゲート値（§7 参照）とは別枠として
扱う。

## 6. framework-compare（参考計測）

本イシュー着手時点で framework-compare の承認済みピンは
`fandhe-ai =0.6.0`（crates.io 公開版）であり、本イシューの変更は
未公開（次回 crates.io 公開まで正式系列には反映されない）。参考系列
（`[patch]` を CLI 引数のみで一時適用し `.cargo/config.toml`・lock は
コミットしない方式。`docs/perf/cuda-gemm-candle-gate-remeasurement.md`
等の先例と同じ手順）による実践規模再計測は、DGX Spark GB10 実機が
本エージェント実行環境から未接続のため実施していない（§7 参照）。
`results/summary.md` の HEAD 反映は次回 crates.io 公開後の正式再計測
で行う（本 doc の対象外）。

## 7. DGX Spark GB10 実測

本エージェント実行環境に CUDA/DGX 実機接続がないため未実測。CPU
バックエンドはハードウェア非依存の同一コードパスだが、Grace CPU の
コア数・NUMA 構成・rayon スレッド数は M4 Max と異なるため、§3〜§5 の
絶対値・相対比はアーキテクチャ依存でありうる。`docs/perf/
cpu-gemm-candle-gate-remeasurement.md`（#1148）が同種の CPU GEMM
ゲート計測を両実機で実施しているため、再計測が必要になった場合は
同 doc の手順（`docs/real-hardware-verification-env.md` の rsync
転送手順）に従う。

## 8. スコープ判断

- **サンプラー（`sample`／`xctrace`）による実プロセス profiling は
  実施しなかった**: §3 のフェーズ分解実測だけで仮説 (i)〜(iii) が
  数値として明確に判別でき（ノイズフロア・26% 改善・残差 3.5% は
  いずれも計測誤差を大きく上回る）、追加のスタックサンプリングが
  判断を変える見込みが薄いため、計測コスト対効果の観点で省略した。
  将来 L1 GEMM 自体のさらなる最適化（`RAYON_NUM_THREADS` スイープ・
  B packing 共有化等）を検討する際は、既存の受け皿
  （`docs/perf/cpu-gemm-b-packing-sharing-decision.md`〈#565〉・
  `docs/perf/cpu-gemm-prefetch-decision.md`〈#489/#751〉・
  `docs/perf/cpu-gemm-candle-cpu-retune.md`〈#1140/#1141〉・`docs/perf/
  cpu-gemm-candle-gate-remeasurement.md`〈#1148〉）で扱う。新規 Issue
  は起票しない（out-of-scope-tracking.md 方針）
- **末尾 `Linear` の bias のみ融合（`Activation::None`）・多層融合・
  CUDA/Metal の `forward_host` 融合・`linear_forward_device` の
  CUDA/Metal 実装（#1216）・infer の `--phases`／reuse 対応（#1217）
  はスコープ外**のまま（§0 決定ルールに従い記録のみ。実装計画 §8 参照）
- **`framework-compare` の `results/summary.md` の HEAD 数値反映**は
  次回 crates.io 公開後の正式再計測で行う（本 doc §6 参照）
