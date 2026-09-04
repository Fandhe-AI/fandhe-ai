# CUDA GEMM N=1024/2048/4096 reuse candle 比再計測と #1031 ゲート判定の確定（イシュー #1142）

## 状態: DGX Spark GB10 実機実測完了。#1031（reuse candle 超え）は正式系列・参考系列（#1164 結線後 HEAD）のいずれも未達成と判定した

## 1. 位置づけ

親 #1031「N=1024/2048/4096 reuse で candle 超え（各 5 回計測の中央値）」の受け入れ判定を、
#1136（classic baseline）・#1137（cp.async 多段パイプラインの本番結線）・#1139（スウィズル。
GB10 到達不能でブロック）の成果を踏まえた最新既定経路で再計測し確定する。本ドキュメントは
その一次記録（プロトコル・実測値・N=2048 candle 無効データの分析・#1031 突合・判定・
ユーザー判断事項）。tolerance・baseline・依存ピンは一切変更しない（`.claude/rules/coding-rust.md`
「テスト・ベンチ」節。本 PR は docs(perf) 区分）。

## 2. 計測環境・プロトコル

- 実機: DGX Spark GB10（詳細は `docs/perf/logs/cuda-gemm-candle-gate-1142/env_info.txt`。
  実ホスト名は記載しない）
- 転送元コミット: `7e3e4b663694e50607fd307afe516386c1e94762`（origin/main HEAD。#1136/#1137/#1139
  の成果物を含む）
- 集計ツール: `scripts/bench/framework-compare/run_gemm_gate_cuda.sh` /
  `compare_gemm_gate.py`（本 PR で新規作成。イシュー #1142。`README.md`「GEMM ゲート 5 回計測」
  節参照）
- N=1024/2048/4096 それぞれ fandhe-ai（`gemm cuda <N> reuse`）・candle（`gemm cuda <N> fresh`。
  reuse 非対応）を run 内で交互に 5 回起動し、run 間中央値で判定（coding-rust.md「ベンチは
  5 回計測の中央値」）
- **2 系列を独立に計測・記録する**（承認済みピンで再計測しても #1137 の反映前とほぼ同値になり
  「最新既定経路」の値を得られないため。詳細は §3）:
  - **正式系列**（`0.6.0`。#1031 の正式判定に用いる）: `bench-fandhe/Cargo.toml` の承認済みピン
    `fandhe-ai =0.6.0`（crates.io 公開版。2026-09-02 公開）のまま計測。コミット済み
    manifest・`Cargo.lock` は変更していない
  - **参考系列**（`head-7e3e4b6`。次回 crates.io 公開後の正式再計測で確定すべき見込み値）:
    ノード側のみで `cargo build --release -p bench-fandhe --config
    'patch.crates-io.fandhe-ai.path="<facade 絶対パス>"'` により `crates/facade`
    （転送元コミット #1164 結線後の HEAD）へ path 差し替えてビルド。`[patch]` セクション・
    `.cargo/config.toml` は一切コミットしていない（CLI 引数のみ）
- GPU 競合確認: 各系列の計測前後で `nvidia-smi --query-gpu=utilization.gpu` を記録し
  いずれも 0% を確認（`results/run_gemm_gate_cuda-dgx-*.log`）
- 生データ: `scripts/bench/framework-compare/results/raw/results-dgx-gemm-gate-0.6.0.jsonl`・
  `results-dgx-gemm-gate-head-7e3e4b6.jsonl`（各 30 行）、失敗記録は両系列とも空
  （`skipped-dgx-gemm-gate-*.log`）

## 3. なぜ 2 系列が必要か

- `fandhe-ai =0.6.0` の crates.io 公開（2026-09-02）は PR #1164（cp.async パイプラインの
  `CudaGemm::run_tiled_f32` 系 3 入口への本番結線。マージ 2026-09-02T21:45Z）より**前**であり、
  正式系列（承認済みピンのまま）は #1137 の性能改善を含まない。実際、正式系列の値は v0.6.0
  横並び再計測（`results/summary.md` 環境 10。計測日 2026-09-02）の単発計測値とほぼ同水準
  （後述 §4 表で確認）
- 「最新既定経路」（#1136/#1137/#1139 適用後）の値を得るには、次回 crates.io 公開（v0.7.0
  想定。ピン更新はユーザー承認必須）を待つか、参考系列のように HEAD へ path 差し替えて計測する
  必要がある
- 参考系列は**正式なゲート判定には用いない**（§6）。次回ピン更新後の正式再計測で確定する
  見込み値としての位置づけ

## 4. 実測結果

### 4.1 正式系列（`fandhe-ai =0.6.0`）

| N | fandhe-ai reuse 中央値（min–max, n=5） | candle fresh 中央値（n=5） | candle/fandhe | GFLOP/s（fandhe） | 判定 |
|---|---|---|---|---|---|
| 1024 | 2.482 ms（2.391–2.617 ms） | 923.6 µs | 0.372 | 865.2 | 未達 |
| 2048 | - | - | - | - | 判定不能（candle 無効データ。§5） |
| 4096 | 68.337 ms（68.318–69.104 ms） | 56.324 ms | 0.824 | 2011.2 | 未達 |

環境 10（v0.6.0 横並び再計測・単発計測。`results/summary.md`）の同一 (task,device,size,mode)
との比較: N=1024 0.35 倍 → 0.372 倍・N=4096 0.81 倍 → 0.824 倍（誤差範囲内でほぼ一致）。N=2048
は環境 10 でも同一の `fail=2/4194304, max_abs=3.624e-05, max_rel=2.811e-01` で無効。**5 回計測に
拡張しても環境 10 の単発計測から実質的な変化はない**（承認済みピンが #1137 を含まないため。
§3）。

### 4.2 参考系列（`head-7e3e4b6`。#1164 結線後）

| N | fandhe-ai reuse 中央値（min–max, n=5） | candle fresh 中央値（n=5） | candle/fandhe | GFLOP/s（fandhe） | 判定 |
|---|---|---|---|---|---|
| 1024 | 2.414 ms（2.364–2.517 ms） | 923.5 µs | 0.383 | 889.7 | 未達 |
| 2048 | - | - | - | - | 判定不能（candle 無効データ。§5） |
| 4096 | 62.600 ms（60.252–63.437 ms） | 56.216 ms | 0.898 | 2195.5 | 未達 |

正式系列比: N=1024 は 0.372→0.383 倍（ほぼ横ばい）、N=4096 は 0.824→0.898 倍（**改善したが
未達のまま**。fandhe-ai reuse 中央値は 68.337 ms → 62.600 ms で約 1.09 倍高速化）。

### 4.3 解釈: カーネル改善が reuse 比の達成に直結しない理由

#1137 の GB10 実測（`docs/perf/cuda-gemm-tiled-pipeline.md`）は launch-only（カーネル単体）の
比較で N=4096 の after/before が 1.514 倍と報告されている。一方 framework-compare の reuse
計測境界は H2D（未使用だが reuse でも毎 step のホスト入力生成）・カーネル・D2H（`Tensor<f32>`
の `loss.to_tensor()` 相当のホスト実体化）を含み、`Tensor<f32>` がホスト常駐設計であるため
reuse でも各回の結果実体化が単一 in-order ストリーム上の同期点として残る
（`docs/backend-cuda-async-execution-design.md`）。カーネル単体で 1.5 倍速くなっても、
この固定費（H2D/D2H・同期）を含む reuse 全体時間では希釈され、N=4096 で 0.824→0.898 倍
（約 9%改善）にとどまった。**この構造的な計測境界要因により、カーネル最適化のみでは
#1031 の「reuse で candle 超え（1.0 倍以上）」を達成できない可能性がある**（§8 スコープ外
事項参照）。

**実測確定（イシュー #1182 追補）**: 上記の「H2D/D2H を含む固定費が希釈要因」という推定は
`docs/perf/cuda-gemm-reuse-phase-breakdown.md`（#1182。GB10 実機フェーズ分解実測）により
**部分的に不正確**と判明した。H2D＋カーネル＋D2H＋同期（`matmul` 区間）単体は candle の
fresh 全体より高速（N=1024: 1.59 倍、N=4096: 1.47 倍）であり、reuse 総計を候補比未達へ
押し下げている主因はベンチハーネス自身が追加する `host_copy`（二重ホストコピー）と
`checksum`（診断用全要素和）であることが確定した（同ドキュメント §6・§7）。

## 5. N=2048 candle 無効データの原因・再現条件

### 5.1 再現条件（決定的）

- 形状: N=2048（正方行列。M=N=K=2048）
- 入力: xorshift64\* の同一シード・同一生成式（GEMM は全フレームワーク共通入力。
  `results/summary.md`「checksum 相互突合」節）
- 対象: `candle-core =0.11.0` の CUDA GEMM 経路（`bench-candle gemm cuda 2048 fresh`）
- 実測値: `parity_fail_count=2`・`parity_total=4194304`・`parity_max_abs_err=3.623962e-05`・
  `parity_max_rel_err=2.811288e-01`（本体の数値一致契約「相対誤差 1e-3 未満 または 絶対誤差
  1e-5 未満」を fail-closed に外れる。`scripts/bench/framework-compare/checksum_contract.py`）
- **完全に決定的**: 正式系列・参考系列それぞれ 5 run・計 10 run すべてで上記 4 値が
  一致（`compare_gemm_gate.py` の run 別内訳表・生 JSONL で確認）。かつ、これは v0.6.0
  横並び再計測（環境 10。計測日 2026-09-02・別セッション・別バイナリビルド）の値
  （`fail=2/4194304, max_abs=3.624e-05, max_rel=2.811e-01`）とも一致する。**入力・形状・
  candle バージョンが同じであれば、いつ・どのセッションで計測しても再現する**
- fandhe-ai 側（reuse）は N=2048 で `parity_fail_count=0`（10 run 全件）。無効の原因は
  fandhe-ai 側ではなく candle-core 側の GEMM カーネル出力にある

### 5.2 他形状・他環境との突合

| 環境 | フレームワーク/デバイス | N=2048 | 参考: N=1024 | 参考: N=4096 |
|---|---|---|---|---|
| DGX Spark GB10（本計測） | candle/cuda | **無効**（fail=2） | 有効（fail=0。要素ごとの複合判定〈rel<1e-3 または abs<1e-5〉はいずれの要素も通過。全要素中の max_rel=0.34・max_abs=1.8e-5 はそれぞれ異なる要素の値で、両者が同一要素で同時に閾値を外れてはいない） | 有効（fail=0） |
| 環境 10（`results/summary.md`） | candle/cuda | **無効**（fail=2。本計測と同値） | 有効 | 有効 |
| 環境 10 | candle/cpu | **無効**（fail=2, max_abs=3.815e-05, max_rel=3.944e-01） | - | - |
| 環境 10 | burn/cpu | **無効**（fail=5, max_abs=3.529e-05, max_rel=3.052e-01） | - | - |
| 環境 11（Apple M4 Max） | candle/cpu | 有効（同一 N=2048 で DGX とは異なり pass） | - | - |
| 環境 11 | burn/cpu | **無効**（同一の fail=5, max_abs=3.529e-05 と完全一致。決定的） | - | - |

- **burn/cpu の N=2048 無効は DGX・M4 Max で完全同一値**であり、参照実装（`GemmReference`。
  k 昇順 FMA 逐次）と burn 側カーネルの累積順序差が実行環境非依存で決定的に発生している
  ことを示す
- **candle/cpu の N=2048 無効は DGX でのみ発生し M4 Max では発生しない**（環境依存）。
  一方 candle/cuda の N=2048 無効は本計測・環境 10 いずれも DGX 上で発生し、CPU 側と
  CUDA 側で異なる要素数・誤差値（cpu: fail=2, max_abs=3.815e-05／cuda: fail=2,
  max_abs=3.623962e-05）が出ている。両者が同一の 2 要素なのか異なる要素なのかは
  本計測の範囲（要素インデックスの取得は §5.3 のとおり未実施）では特定できていない

**追記（イシュー #1184。GB10 実機での実値取得により確定）**: candle/cuda（`idx=13850`・
`idx=4130484`）と candle/cpu（`idx=1372466`・`idx=1633751`）の fail 2 要素は**異なる要素**
であることを確認した（同一 GB10 ノード・同一バイナリ・同一入力での 1 回起動。§5.3 追記参照）。
device 間でカーネルの累積順序が異なるため、たまたま複合判定を割る要素の位置も変わる、という
解釈と整合する

### 5.3 仮説と限界

**仮説**: N=2048 の一部要素（真値が 0 近傍と推定される要素）において、参照実装（k 昇順
FMA 逐次累積）と各フレームワークの実装（BLAS/cuBLAS/cuDNN 相当の異なる累積順序・ブロッキング）
との丸め誤差が、たまたま abs 側閾値（1e-5）・rel 側閾値（1e-3）の両方を同時に超える形状・
入力の組み合わせが N=2048 に存在する、というもの。N=1024・N=4096 で同一種の丸め誤差が
発生していない（またはより小さく複合判定内に収まっている）ことと整合する。burn/cpu の
決定性（§5.2）は、少なくとも burn 側についてはこの丸め誤差が実装・環境に依らず固定の
入力パターンで再現する構造的な性質であることを示唆する

**限界（本計測で確認していない事項）**: 実際に fail した要素の値（reference 値・実測値の
生データ）は取得していない。`bench-common::parity` に fail 要素の値を出力する診断計装
（`FRAMEWORK_COMPARE_PARITY_DUMP` 環境変数案）を追加する設計を検討したが、既存 JSONL
フィールド（`parity_fail_count`/`parity_total`/`parity_max_abs_err`/`parity_max_rel_err`）
と本計測（正式系列・参考系列で計 10 run の決定性確認・環境 10/11 との突合）だけで
「決定的・candle 側の丸め誤差・tolerance 契約上は判定不能のまま維持」という結論を出すには
十分と判断し、本 PR ではその計装を追加していない（R2「原因・再現条件を記録」は本節で
満たしていると判断）

**追記（イシュー #1183）**: 上記の診断計装は `FRAMEWORK_COMPARE_PARITY_DUMP` 環境変数として
追加済み（`scripts/bench/framework-compare/bench-common/src/parity.rs`。使い方は
`scripts/bench/framework-compare/README.md` §「fail 要素ダンプ」）。本イシューは計装の追加のみで、
GB10 実機での N=2048 fail 要素の実際の値取得・本節の仮説検証は別途実施する（本エージェント実行環境に
CUDA 実機なし）

**追記（イシュー #1184。GB10 実機実測完了・2026-09-03）**:

- **取得方法**: `FRAMEWORK_COMPARE_PARITY_DUMP=1` で `bench-candle --task gemm --device
  {cuda,cpu} --size 2048 --mode fresh` を GB10 実機（`~/work/rust-ai-library-run`。転送元
  `808b4be`）で 1 回ずつ起動し、stderr の `PARITY_DUMP` 行（warmup 20 + 計測 20 = 40 call 分）
  を取得した（`docs/perf/logs/cuda-gemm-candle-parity-1184/parity-dump-{cuda,cpu}-2048.txt`）。
  JSONL の 4 parity 値は candle/cuda が `fail=2/4194304, max_abs=3.623962e-05,
  max_rel=2.811288e-01`、candle/cpu が `fail=2/4194304, max_abs=3.814697e-05,
  max_rel=3.944416e-01` で、いずれも §5.1・§5.2 の既存記録・#1142 環境 10 の値と完全一致した
  （新規ビルド・新規セッションでの再現性を追加確認）
- **決定性**: candle/cuda・candle/cpu とも 40 call すべてで同一の 2 idx・同一の `ref_bits`/
  `actual_bits`（bit 完全一致）が出た（`truncated=false`。実行間の非決定性なし）
- **厳密真値との突合**: `scripts/bench/framework-compare/parity_dump_truth.py`（本イシューで
  新規作成。標準ライブラリのみ）で `Xorshift64Star`/`fill_vec` を Python 側で厳密再現し
  （2 進有理数として誤差ゼロ表現）、fail 要素ごとに (1) 有理数演算による厳密真値
  `exact = Σ_k A[row,k]·B[k,col]`、(2) f64 逐次和、(3) f32 FMA 逐次累積の厳密丸め再現、
  (4) 部分和の最大絶対値 `max|partial|` とキャンセレーション由来の期待誤差フロア
  `√K · ulp_f32(max|partial|)` を計算した（出力: `docs/perf/logs/cuda-gemm-candle-parity-1184/truth-2048.txt`）。
  **(3) の f32 FMA 逐次再現は 4 要素すべてで `ref_bits` と bit 完全一致**（`fma_bit_match=True`）
  し、これが RNG 再現の正しさと「参照実装が契約どおり k 昇順 FMA 逐次累積として動作している
  こと」の直接証拠になっている

  | idx | device | row,col | exact | ref | actual | \|ref−exact\| | \|actual−exact\| | max\|partial\| | √K·ulp(max\|partial\|) |
  |---|---|---|---|---|---|---|---|---|---|
  | 13850 | cuda | 6,1562 | 2.166853e-03 | 2.168937e-03 | 2.157688e-03 | 2.084e-06 | 9.165e-06 | 3.969 | 1.079e-05 |
  | 4130484 | cuda | 2016,1716 | 9.197426e-03 | 9.188101e-03 | 9.199142e-03 | 9.325e-06 | 1.717e-06 | 6.099 | 2.158e-05 |
  | 1372466 | cpu | 670,306 | 9.918718e-03 | 9.920587e-03 | 9.933233e-03 | 1.869e-06 | 1.452e-05 | 5.613 | 2.158e-05 |
  | 1633751 | cpu | 797,1495 | 5.382382e-03 | 5.374012e-03 | 5.385637e-03 | 8.370e-06 | 3.255e-06 | 4.748 | 2.158e-05 |

- **仮説の判定: 真（部分的に再校正）**。4 要素すべてで `|ref−exact|`・`|actual−exact|` は
  `√K·ulp(max|partial|)` と同水準（比 0.08〜0.85 倍。いずれも 1 倍未満で異常な誤差ではない）
  であり、参照実装（k 昇順 FMA 逐次）・candle 側カーネルのどちらも通常の累積丸め誤差の範囲に
  収まっている。**どちらか一方の実装が恒常的に他方より誤差が大きい、という片側優位の構造は
  ない**（`|actual−exact|` が `|ref−exact|` を上回る要素〈13850・1372466〉と下回る要素
  〈4130484・1633751〉が両方存在し、比は 0.08〜7.8 倍とばらつく）。したがって §5.1 の
  「candle 側の丸め誤差」という表現は不正確で、**参照実装・candle 側双方が持つ通常の
  累積丸め誤差が、たまたま同時に abs/rel 両閾値を超える形状・入力の組み合わせが N=2048 に
  存在する**、と訂正する
- **「0 近傍」の再評価**: §5.3 当初の仮説文言・plan 段階の見積り（`3.62e-5/0.281 ≈ 1.3e-4`）は
  誤りだった。**`0.281`（`parity_max_rel_err`）は fail 2 要素のいずれの値でもない**
  （fail 要素の実際の相対誤差は 1.2e-3〜5.2e-3。`compare_elementwise` は複合判定で pass した
  要素も含めて全要素中の `max_abs_err`/`max_rel_err` を独立に追跡するため、`parity_max_rel_err`
  は「abs 側で救済されたが rel が極端に大きい別の passing 要素」、`parity_max_abs_err` も
  同様に「rel 側で救済されたが abs が大きい別の passing 要素」に由来しうる——`fail_count`
  の対象要素とは限らない。これは本イシューで判明した、`summarize.py`/`compare_gemm_gate.py`
  の既存出力を読む上での注意点であり、判定ロジック自体の不具合ではない。fail 2 要素自体の
  `exact` は 2.2e-3〜9.2e-3 のオーダーで、機械イプシロン近傍という意味の「0 近傍」ではない。
  一方、各要素の**部分和の最大絶対値**（3.97〜6.10）に対し最終値（`exact`）はその
  400〜2600 分の 1 まで縮小しており、**累積過程で大きな桁のキャンセレーションが起きた
  結果、最終値が「相対的に」小さくなっている**（これが「0 近傍」の実体）。桁が縮小した
  分だけ、累積過程で生じた丸め誤差フロア（`√K·ulp(max|partial|)`）が最終値そのものと
  同程度の大きさになり、abs 救済閾値 1e-5・rel 閾値 1e-3 の両方を同時に割り込む
- **本イシューでのスコープ外**: burn/cpu（fail=5）の実値取得は別バイナリ（`bench-burn`）を
  要し、本イシューの対象外のまま据え置く（§8）

### 5.4 tolerance の扱い

**tolerance は緩めない**。本体の数値一致契約（相対誤差 1e-3 未満 または 絶対誤差 1e-5 未満。
`.claude/rules/coding-rust.md`）は本計測でも不変のまま適用し、N=2048 は「判定不能」の
まま記録する。判定方式の変更（別シードでの追加計測・参考比の併記・spec 側への追記等）は
§9「ユーザー判断事項」に列挙するのみで、本 PR では実施しない

**追記（イシュー #1184）**: `PARITY_REL_TOL`/`PARITY_ABS_TOL`・`compare_elementwise`・
`compare_gemm_gate.py`/`summarize.py` の判定ロジックは本イシューでも一切変更していない
（`bench-common`・両 Python スクリプトへの差分なし。`git diff` で確認可能）。N=2048 は
引き続き「判定不能」のまま記録する

## 6. #1031 受け入れ条件との突合

| # | #1031 の受け入れ条件 | 正式系列（0.6.0） | 参考系列（head-7e3e4b6） | 出典 |
|---|---|---|---|---|
| 1 | N=1024 reuse で candle 超え | 未達（0.372 倍） | 未達（0.383 倍） | §4.1・§4.2 |
| 2 | N=2048 reuse で candle 超え | 判定不能（candle 無効データ） | 判定不能（candle 無効データ） | §4.1・§4.2・§5 |
| 3 | N=4096 reuse で candle 超え | 未達（0.824 倍） | 未達（0.898 倍。改善したが未達） | §4.1・§4.2 |
| 4 | parity 0 fail（fandhe-ai 側） | 達成（10 run 全件 `parity_fail_count=0`） | 達成（同上。加えて #1163/#1164 の GB10 実測〈`gemm_tiled`/`cpu_cuda_parity` --ignored 全 PASS〉が既存の出典として補強） | §4.1・§4.2・`docs/perf/cuda-gemm-tiled-pipeline.md`「#1137 本番結線判断」節 |

**総合判定: #1031 は正式系列・参考系列のいずれにおいても未達成（未達 2 件・判定不能 1 件）**。
`crate::precision`（TF32）等の精度緩和経路は本計測の対象外（REQ-2 の FP32 SIMT 経路が対象）。

## 7. `results/summary.md`・`performance-targets.md` への反映

- `results/summary.md` 環境 12 節・「目標達成ゲート総括」への追補は本 PR に含む
  （`scripts/bench/framework-compare/results/summary.md` 参照）
- `docs/performance-targets.md` §8.2「#1142 追補」（§2 段階的下限表・§3 丸め規則は不変）
- `docs/perf/gemm-optimization-baseline.md` §6（candle 比ゲートは REQ-8 PyTorch 比の対象外
  である旨の参照節。表自体は変更しない）
- `docs/perf/cuda-gemm-tiled-pipeline.md`「#1137 本番結線判断」末尾に本ドキュメントへの
  参照 1 行を追記済み

## 8. スコープ外事項（本 PR では対応しない）

- **reuse 計測境界の H2D/D2H 固定費削減**: `Tensor<f32>` のホスト常駐設計に起因する
  reuse でも残る同期点（§4.3）。カーネル最適化（#1031 のスコープ）では解消できない
  構造要因であり、#1031 の未達が今後も残る可能性がある。対処には `Tensor<f32>`
  のデバイス常駐化等、別スコープの設計変更が必要（後続 issue 化の要否は §9 ユーザー判断）。
  **実測確定（#1182）**: 固定費の主因は H2D/D2H 自体ではなく `host_copy`／`checksum`
  であることが判明した。`docs/perf/cuda-gemm-reuse-phase-breakdown.md` §6・§9 参照
- **N=2048 の判定方式変更**（別シード追加計測・参考比の併記・spec 側 REQ-2 への追記）:
  tolerance 契約はユーザー承認必須のため本 PR では変更しない（§9）
- **crates.io v0.7.0 公開・framework-compare ピン `=0.7.0` 更新**: 正式系列で #1137 の
  改善を反映した判定を得るために必要（ユーザー承認事項。deps-policy.md 第 9 区分）
- **スウィズル（#1034）の結線判断**: #1139 で GB10 実機到達不能によりブロック中
  （`docs/perf/cuda-gemm-tiled-f32-swizzle-ab.md`）。本 PR の対象外
- `docs/perf/performance-floor-decision.md`（REQ-8 の PyTorch 比下限）は変更しない
  （candle 比とは別軸のため）

## 9. ユーザー判断事項

- **#1031 のクローズ可否**: 本計測により正式系列・参考系列いずれでも未達成が確定した。
  クローズせず残課題として維持するか、達成条件・スコープの見直し（例: reuse 計測境界の
  再定義、H2D/D2H を除いたカーネル専有時間での判定への変更）を検討するかはユーザー判断
- **後続 issue 化の要否**: reuse 固定費削減（§8）・N=2048 判定方式変更（§8）を追跡する
  新規 issue を起票するかはユーザー判断（`out-of-scope-tracking.md` に従い、本 PR では
  Issue 操作を行わない）
- **crates.io 次回公開のタイミング**: #1137 を含む正式ピン更新（v0.7.0 想定）の要否・時期
- **N=2048 判定方式の変更（イシュー #1184 で判明した事実を踏まえた整理。ユーザー承認事項）**:
  §5.3 追記のとおり、fail 2 要素の丸め誤差は `√K·ulp(max|partial|)` と同水準（K=2048・
  入力 U[-0.5,0.5) の累積丸め誤差フロアそのもの）であり、「たまたま」ではなく K が大きい
  正方 GEMM 形状で構造的に起こりうる。tolerance 自体（`PARITY_REL_TOL`/`PARITY_ABS_TOL`）の
  変更は提案しないが、判定方式の候補として以下をユーザー判断のため列挙する（本イシューでは
  いずれも実施しない）:
  - (a) 別シード（`SEED_A`/`SEED_B` 以外）での追加計測により、N=2048 で同種の fail が
    シード非依存に発生するか確認する
    - (b) N=2048 は「判定不能」のまま、参考情報として reuse/candle 比を注記付きで併記する
  - (c) spec（`docs/spec/04-requirements.md` REQ-2）へ「大規模 K での要素単位複合判定は、
    キャンセレーションで最終値が縮小した要素を対象外とする」等の例外規定を追記する
    （fandhe-ai-spec 側での対応が必要）

## 10. 関連ドキュメント

- `docs/perf/cuda-gemm-tiled-pipeline.md`（#1137 本番結線判断・カーネル単体 launch-only 計測）
- `docs/perf/cuda-gemm-tiled-f32-swizzle-ab.md`（#1034 スウィズルのブロック判断）
- `docs/backend-cuda-async-execution-design.md`（reuse でも残る同期点の設計根拠）
- `scripts/bench/framework-compare/README.md`「GEMM ゲート 5 回計測（#1142）」節
- `scripts/bench/framework-compare/results/summary.md` 環境 10/11/12 節
- `docs/performance-targets.md` §8/§8.1/§8.2
- `docs/perf/logs/cuda-gemm-candle-gate-1142/`（実行ログ・env_info）
- `docs/perf/logs/cuda-gemm-candle-parity-1184/`（イシュー #1184。fail 要素ダンプ生データ・
  厳密真値突合結果・env_info）
- `scripts/bench/framework-compare/parity_dump_truth.py`（イシュー #1184。`PARITY_DUMP` 行から
  厳密真値を計算する再現用スクリプト）
