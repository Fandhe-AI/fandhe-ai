# CUDA GEMM N=1024/2048/4096 reuse candle 比再計測と #1031 ゲート判定の確定（イシュー #1142）

## 状態: DGX Spark GB10 実機実測完了。#1031（reuse candle 超え）は正式系列・参考系列（#1164 結線後 HEAD）のいずれも未達成と判定した。#1185 で正式系列 `fandhe-ai =0.7.0` を 2026-09-06 に再計測し未達成を確定（§11）。#1360 で Phase 4／5（#1342 の 128×64 cp.async pipeline 本番結線・#1337 借用ビュー readout）反映後の GB10 再計測を実施し、正式系列（ピン未更新のため §11 と同値。再現性確認）は未達成が継続、参考系列（path 差し替え HEAD）は readout off で未達成継続・on で N=4096 のみ達成（N=1024/2048 は大幅後退）を記録した（§12。正式判定はピン更新後に確定）。#1337 の正式記録・公正性の論点は §13 参照（独自再現も同一符号で確認）。#1260 で承認済み tolerance 契約変更（#1241 承認・#1247/#1250 実装）下の N=2048 を再計測し、「判定不能」が解消して確定判定（未達・0.476 倍）へ遷移したことを確認した（§14。N=1024/4096 は引き続き未達）。#1438 で借用ビュー readout の既定経路化（旧 host-view-readout cargo feature 撤去）の before/after を GB10 実機実測し、全 3 形状非後退・checksum 完全一致を確認（ADOPT。正式判定は §14 のまま不変。§15）

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
  （2026-09-06 更新: v0.7.0 公開・ピン更新〈PR #1233〉により解消済み。§11）
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
- **2026-09-06 更新（イシュー #1185）**: 正式系列 `fandhe-ai =0.7.0` でも未達成が確定した
  （§11）ことを受け、ユーザー指示（2026-09-06）「未達の場合は後継ツリーを新規起票し現 issue は
  クローズ」に従い、上記の残課題（reuse 固定費削減・N=2048 判定方式・達成条件の見直し）は
  後継ツリー #1234（tolerance 契約変更トラッキング。N=2048 判定不能の解消） へ引き継ぎ、現 issue #1031 はクローズする

## 10. 関連ドキュメント

- `docs/perf/cuda-gemm-tiled-pipeline.md`（#1137 本番結線判断・カーネル単体 launch-only 計測）
- `docs/perf/cuda-gemm-tiled-f32-swizzle-ab.md`（#1034 スウィズルのブロック判断）
- `docs/backend-cuda-async-execution-design.md`（reuse でも残る同期点の設計根拠）
- `scripts/bench/framework-compare/README.md`「GEMM ゲート 5 回計測（#1142）」節
- `scripts/bench/framework-compare/results/summary.md` 環境 10/11/12 節
- `docs/performance-targets.md` §8/§8.1/§8.2
- `docs/perf/logs/cuda-gemm-candle-gate-1142/`（実行ログ・env_info）
- `docs/perf/logs/gemm-candle-gate-0.7.0-1185/`（イシュー #1185。=0.7.0 正式系列再計測の
  実行ログ・env_info。CUDA / Metal / CPU 共通）
- `docs/perf/logs/cuda-gemm-candle-parity-1184/`（イシュー #1184。fail 要素ダンプ生データ・
  厳密真値突合結果・env_info）
- `scripts/bench/framework-compare/parity_dump_truth.py`（イシュー #1184。`PARITY_DUMP` 行から
  厳密真値を計算する再現用スクリプト）
- `docs/perf/logs/cuda-gemm-candle-gate-1360/`（イシュー #1360。Phase 4／5 反映後の 3 系列
  実行ログ・env_info）
- `docs/perf/cuda-gemm-tiled-pipeline.md` §8（#1342 の 128×64 cp.async pipeline 本番結線）
- `scripts/bench/framework-compare/README.md`「借用ビュー readout（イシュー #1337）」節
- `docs/perf/candle-parity-tolerance-candidates.md`（イシュー #1237。#1184 ダンプ実値から
  スケール付き絶対誤差／ULP 判定候補の fail 数を机上計算した結果。tolerance 契約変更の
  ユーザー承認判断に使う定量根拠）
- `scripts/bench/framework-compare/parity_tolerance_candidates.py`（イシュー #1237。上記の
  計算スクリプト）
- `docs/perf/logs/candle-parity-tolerance-candidates-1237/`（イシュー #1237。上記の実行ログ・
  env_info）
- `docs/perf/candle-parity-tolerance-baseline-impact.md`（イシュー #1238。上記候補判定を
  fandhe-ai 本体側の parity 非後退契約〈`BASELINES`。45 行〉へ適用した場合の影響の机上確認。
  `BASELINES`・tolerance 定数は不変のまま）
- `docs/perf/logs/candle-parity-tolerance-baseline-impact-1238/`（イシュー #1238。上記の実行
  ログ・env_info）
- `docs/candle-parity-tolerance-contract-decision.md`（イシュー #1239。tolerance 契約変更の
  決定記録 draft。候補比較・推奨案〈未承認〉・ユーザー承認待ち事項〈#1241〉の整理。
  §9 で #1241 承認〈2026-09-08〉・Phase 2 実装範囲〈#1247／#1250 実装済み・#1260／#1262 が
  N=2048 再計測〉を記録）
- `docs/perf/logs/cuda-gemm-candle-gate-1260/`（イシュー #1260。承認済み契約下の N=2048
  再計測の実行ログ・判定表出力・env_info）

## 11. 2026-09-06 追補: 正式系列 `fandhe-ai =0.7.0` 再計測（イシュー #1185）

### 11.1 位置づけ・プロトコル

- v0.7.0 の crates.io 公開と framework-compare の承認ピン `fandhe-ai =0.7.0` 更新（PR #1233）
  を受け、**正式系列のみ**で N=1024/2048/4096 reuse の 5 回計測中央値を DGX Spark GB10 で
  再取得した（#1185 受け入れ条件 1 項目目）。参考系列（path 差し替えビルド）は計測していない
- プロトコルは §2 と同一（`run_gemm_gate_cuda.sh 0.7.0`・`compare_gemm_gate.py --device cuda`。
  manifest で `fandhe_ai_source=registry`・`candle_core_source=registry` を確認済み）
- 計測環境: GB10・driver 580.173.02・CUDA 13.0・rustc 1.97.0。計測直前の GPU 利用率 0 %・
  load average 0.04（実質アイドル。`docs/perf/logs/gemm-candle-gate-0.7.0-1185/env_info.txt`）
- 生データ: `scripts/bench/framework-compare/results/raw/results-dgx-gemm-gate-0.7.0.jsonl`
  （30 行）・`skipped-dgx-gemm-gate-0.7.0.log`（空）・`manifest-dgx-gemm-gate-0.7.0.json`。
  実行ログ: `docs/perf/logs/gemm-candle-gate-0.7.0-1185/run_gemm_gate_cuda-dgx-0.7.0.log`

### 11.2 実測結果（正式系列 `0.7.0`）

| N | fandhe-ai reuse 中央値（min–max, n=5） | candle fresh 中央値（n=5） | candle/fandhe | GFLOP/s（fandhe） | 判定 |
|---|---|---|---|---|---|
| 1024 | 2.424 ms（2.316–2.581 ms） | 923.7 µs | 0.381 | 886.1 | 未達 |
| 2048 | - | - | - | - | 判定不能（candle 無効データ。§5・§11.4） |
| 4096 | 62.301 ms（61.783–63.038 ms） | 56.296 ms | 0.904 | 2206.1 | 未達 |

fandhe-ai 側は全 15 run で `parity_fail_count=0`（N=1024/2048/4096 とも）。tolerance は
緩めていない。

### 11.3 参考系列との一致確認（#1185 受け入れ条件 2 項目目）

| N | 参考系列 `head-7e3e4b6`（#1142 §4.2） | 正式系列 `0.7.0`（§11.2） | 差 |
|---|---|---|---|
| 1024 | 0.383（fandhe 2.414 ms / candle 923.5 µs） | 0.381（fandhe 2.424 ms / candle 923.7 µs） | fandhe +0.4 %・candle +0.02 % |
| 2048 | 判定不能 | 判定不能 | 同一の candle 無効データ（§11.4） |
| 4096 | 0.898（fandhe 62.600 ms / candle 56.216 ms） | 0.904（fandhe 62.301 ms / candle 56.296 ms） | fandhe -0.5 %・candle +0.1 % |

**参考系列（0.383／判定不能／0.898 倍）と誤差範囲内で一致する**（fandhe-ai reuse 中央値の差は
N=1024 で +0.4 %・N=4096 で -0.5 %。いずれも 5 run の min–max 幅〈N=1024: 2.316–2.581 ms・
N=4096: 61.783–63.038 ms〉の内側）。#1137（cp.async 多段パイプライン。#1164 結線）が
正式系列に反映されたことで、0.6.0 正式系列（§4.1: 0.372／判定不能／0.824）からの改善
（N=4096: 68.337 ms → 62.301 ms・0.824 → 0.904 倍）が参考系列で観測した値のとおり再現した。

### 11.4 N=2048 判定不能の再現

candle 側 N=2048 fresh は 5 run すべてで `parity_fail_count=2, parity_total=4194304,
parity_max_abs_err=3.623962e-05, parity_max_rel_err=2.811288e-01`（§5.1・環境 10・#1142 の
両系列と完全に同一の決定的な値）。原因分析は §5・イシュー #1184（fail 2 要素の実値・厳密真値
突合。§5.3）を参照。fandhe-ai 側は 0 fail。判定方式の変更（§9 の (a)〜(c)）は本追補でも
実施しておらず、N=2048 は「判定不能」のまま据え置く（reuse/candle 比の参考併記も行わない）。

### 11.5 #1031 ゲート判定（確定）

| # | #1031 の受け入れ条件 | 正式系列（0.7.0） | 出典 |
|---|---|---|---|
| 1 | N=1024 reuse で candle 超え | 未達（0.381 倍） | §11.2 |
| 2 | N=2048 reuse で candle 超え | 判定不能（candle 無効データ） | §11.2・§11.4 |
| 3 | N=4096 reuse で candle 超え | 未達（0.904 倍） | §11.2 |
| 4 | parity 0 fail（fandhe-ai 側） | 達成（全 15 run `parity_fail_count=0`） | §11.2 |

**総合判定: #1031 は正式系列 `fandhe-ai =0.7.0` においても未達成（未達 2 件・判定不能 1 件）。
#1142 の 2 系列判定を正式系列単独で確定した（#1185 受け入れ条件 3 項目目）。** 達成条件の
見直し要否・後継ツリーへの引き継ぎは §9「2026-09-06 更新」を参照。

## 12. 2026-09-07 追補: Phase 4／5 反映後の再計測（イシュー #1360）

### 12.1 位置づけ・プロトコル

- §11（イシュー #1185）以降、Phase 4／5（#1341 配下）で CUDA f32 SIMT GEMM に以下が入った:
  - #1342（PR #1385・2026-09-06 マージ）: 128×64×16 cp.async pipeline カーネルを
    `N>=1024 && K>=1024` で本番結線（`TILED_PIPELINE_128X64_PRODUCTION_ENABLED = true`。
    `crates/backend-cuda/src/gemm.rs:979`）
  - #1345（PR #1387）: persistent タイルキューは REJECT（本番結線なし）
  - #1334/#1335/#1336/#1337（PR #1404/#1408/#1411）: `bench-fandhe` の `readout_var` を
    cargo feature `host-view-readout`（既定 OFF）で借用ビュー（`VarHostView`／
    `Tensor::host_slice`）readout 経路へ切替可能化。`run_gemm_gate.sh` に
    `GEMM_GATE_BENCH_FANDHE_FEATURES`（allowlist・`GEMM_GATE_PATCH_FACADE_PATH` 併用必須）を
    追加済み
- crates.io の承認ピンは §11 時点から更新されていない（`gh api repos/Fandhe-AI/fandhe-ai/tags
  --jq '.[0].name'` = `v0.7.0`。#1342／#1334 系はいずれも v0.7.0 公開後にマージされたコード
  のため registry ピンには未反映）。よって §11 と同じ「正式系列 + 参考系列」の 2 系列方式を
  踏襲し、**参考系列を 2 本**（readout off／on）に分けて計測した:

  | ラベル | ビルド | 位置づけ |
  |---|---|---|
  | `0.7.0-1360` | registry ピン（`fandhe_ai_source=registry`） | 正式系列。§11.2 の `0.7.0` との再現性確認（新規ファイル名。既存 `results-dgx-gemm-gate-0.7.0.jsonl` は上書きしていない） |
  | `head-3d5e833-readout-off` | `GEMM_GATE_PATCH_FACADE_PATH=<facade 絶対パス>`（feature なし） | 参考系列 A。#1342（128×64 結線）のみの効果 |
  | `head-3d5e833-readout-on` | 同上 + `GEMM_GATE_BENCH_FANDHE_FEATURES=host-view-readout` | 参考系列 B。#1337（借用ビュー readout）を加えた見込み値 |

  `3d5e833` は origin/main HEAD（3d5e8332365404b1ba0ceafee50d87801595d73d）。転送元・作業
  worktree・参考系列ラベルとも同一 sha で揃えている
- 実機: DGX Spark GB10（GPU: NVIDIA GB10・driver 580.173.02・CUDA 13.0.88・rustc 1.97.0。
  内部ホスト名は記載しない。詳細 `docs/perf/logs/cuda-gemm-candle-gate-1360/env_info.txt`）
- 集計ツール: `run_gemm_gate_cuda.sh`／`compare_gemm_gate.py --device cuda`（§2 と同一。
  ロジック変更なし）。readout 切替の効果分離には `compare_gemm_ab.py --device cuda --sizes
  gate --modes reuse`（`jq -c 'select(.framework == "fandhe-ai")'` で抽出した
  `*.fandhe-only.jsonl` を入力）を追加使用
- 計測前後の GPU アイドル確認・実行順序（3 系列の間に別の `cargo` コマンドを挟まない）は
  `env_info.txt` に記録。manifest の `fandhe_ai_source`／`bench_fandhe_features` は各系列の
  意図どおり（§12.1 の表）であることを確認済み
- 生データ: `scripts/bench/framework-compare/results/raw/{results,skipped,manifest}-dgx-gemm-
  gate-{0.7.0-1360,head-3d5e833-readout-off,head-3d5e833-readout-on}.{jsonl,log,json}`（各
  30 行・`skipped-*.log` 空）。実行ログ・env_info:
  `docs/perf/logs/cuda-gemm-candle-gate-1360/`

### 12.2 正式系列 `0.7.0-1360` の実測結果（§11.2 との再現性確認）

| N | fandhe-ai reuse 中央値（min–max, n=5） | candle fresh 中央値（n=5） | candle/fandhe | GFLOP/s（fandhe） | 判定 |
|---|---|---|---|---|---|
| 1024 | 2.445 ms（2.284–2.497 ms） | 925.9 µs | 0.379 | 878.4 | 未達 |
| 2048 | - | - | - | - | 判定不能（candle 無効データ。§5・§11.4 と同一の決定的な `fail_count=2`） |
| 4096 | 62.281 ms（61.961–62.860 ms） | 55.700 ms | 0.894 | 2206.8 | 未達 |

fandhe-ai 側は全 15 run で `parity_fail_count=0`。§11.2（0.381／判定不能／0.904）と誤差範囲内
で一致（fandhe reuse 中央値差: N=1024 +0.9 %・N=4096 -0.03 %）。**同一 registry ピンでの
再計測のため #1342／#1337 の効果はここには現れない**（想定どおり。ピン未更新のため）。

### 12.3 参考系列（path 差し替え・#1342 反映後 HEAD）の実測結果

| N | readout-off 中央値（min–max, n=5） | readout-on 中央値（min–max, n=5） | candle fresh 中央値（n=5） | off の candle 比 | on の candle 比 | off 判定 | on 判定 |
|---|---|---|---|---|---|---|---|
| 1024 | 2.382 ms（2.291–2.571 ms） | 35.832 ms（31.175–44.628 ms） | 917.7–956.3 µs（off 956.260 µs／on 917.707 µs） | 0.401 | 0.026 | 未達 | 未達（大幅後退） |
| 2048 | 9.542 ms（9.451–9.845 ms） | 11.418 ms（11.148–135.399 ms） | - | 判定不能 | 判定不能 | 判定不能 | 判定不能 |
| 4096 | 60.011 ms（59.866–60.993 ms） | 38.278 ms（37.075–40.431 ms） | 54.870–55.998 ms | 0.933 | **1.433** | 未達 | **達成** |

- **readout-off**（#1342 の 128×64 結線のみの効果）: N=1024 0.401・N=4096 0.933 と §12.2 の
  正式系列（0.379／0.894）よりやや改善しているが、いずれも未達のまま（fandhe-ai 側
  `parity_fail_count=0` は全 run で維持）
- **readout-on**（#1342 + #1337 の効果）: **N=4096 で 1.433 倍・達成**（GFLOP/s 3590.6。
  readout-off比で fandhe reuse 中央値が 60.011 ms → 38.278 ms・1.57 倍改善）。一方
  **N=1024 は readout-off比で 15.0 倍の大幅後退**（2.382 ms → 35.832 ms。min–max 幅も
  31.175–44.628 ms と大きくばらつく）。N=2048 は candle 側が判定不能のため fandhe/candle 比の
  達成判定はできないが、fandhe 側単体でも readout-off比で中央値 1.20 倍・max が 135.399 ms
  まで跳ねる後退が見られる（§12.4）
- readout-on の parity は全 run `parity_fail_count=0`（tolerance 契約は不変）

### 12.4 readout 切替効果（`compare_gemm_ab.py --device cuda --sizes gate --modes reuse`）

```
| size/mode | before(off) median | after(on) median | after/before | checksum | 判定 |
|---|---|---|---|---|---|
| 1024/reuse | 2.382 ms (min 2.291 ms / max 2.571 ms) | 35.832 ms (min 31.175 ms / max 44.628 ms) | 15.0419 | 完全一致 | 後退 |
| 2048/reuse | 9.542 ms (min 9.451 ms / max 9.845 ms) | 11.418 ms (min 11.148 ms / max 135.399 ms) | 1.1966 | 完全一致 | 後退 |
| 4096/reuse | 60.011 ms (min 59.866 ms / max 60.993 ms) | 38.278 ms (min 37.075 ms / max 40.431 ms) | 0.6378 | 完全一致 | 非後退 |
```
（`compare_gemm_ab.py` は非回帰判定〈閾値 1.05〉のため N=1024/2048 は「後退」表示。checksum
は off/on とも fandhe-ai・candle 双方で完全一致——数値精度への影響はなし、純粋に readout 経路の
コストの問題）。

- N=4096 の改善は `host_copy`（readout 前のホスト側全要素コピー）の消失に帰属すると考えられる
  （`docs/perf/cuda-gemm-reuse-phase-breakdown.md` §11 の借用ビュー readout 非到達整理を踏襲。
  #1336 の CUDA pinned host staging はこの readout 経路に到達しないため、N=4096 の改善は
  `#1337`〈borrowed-view readout そのもの〉の効果であり `#1336` の効果ではない）
- N=1024/2048 の後退の原因分析（借用ビュー readout の小規模 N での固定費増加の切り分け）は
  本イシューのスコープ外とし、§12.7 のユーザー判断事項へ引き継ぐ

### 12.5 N=2048 判定不能の再現

candle 側 N=2048 fresh は本追補の 3 系列いずれでも 5 run すべてで
`parity_fail_count=2, parity_total=4194304, parity_max_abs_err=3.623962e-05,
parity_max_rel_err=2.811288e-01`（§5.1・§11.4 と完全に同一の決定的な値）。fandhe-ai 側は
3 系列とも 0 fail。#1258（#1234 Phase 2 依存・ユーザー承認必須）が未解消のため、本追補でも
N=2048 は「判定不能」のまま据え置く。

### 12.6 ゲート判定表（#1360 時点）

| # | #1031 の受け入れ条件 | 正式系列 `0.7.0-1360` | 参考系列 readout-off | 参考系列 readout-on |
|---|---|---|---|---|
| 1 | N=1024 reuse で candle 超え | 未達（0.379 倍） | 未達（0.401 倍） | 未達（0.026 倍・大幅後退） |
| 2 | N=2048 reuse で candle 超え | 判定不能 | 判定不能 | 判定不能 |
| 3 | N=4096 reuse で candle 超え | 未達（0.894 倍） | 未達（0.933 倍） | **達成（1.433 倍）** |
| 4 | parity 0 fail（fandhe-ai 側） | 達成 | 達成 | 達成 |

**正式判定（registry ピン `fandhe-ai =0.7.0` に基づく）: #1031 は引き続き未達成**（§11 の
判定を再現・変更なし）。**参考系列（次回ピン更新後の見込み値）**: readout-off は 3 形状とも
未達のまま。readout-on は N=4096 のみ達成条件を満たすが N=1024/2048 で大幅後退するため、
「Phase 4／5 反映後に #1031 が達成される」とは言えない（形状依存の混在結果であり、
`host-view-readout` を既定化する場合は N=1024/2048 側の後退対策が前提になる）。

### 12.7 ユーザー判断事項

- **正式再計測の確定タイミング**: crates.io 次回公開（v0.8.0 相当。#1342／#1334 系を含む）と
  framework-compare ピン更新（deps-policy 第 9 区分。ユーザー承認必須）後に、正式系列のみで
  再計測し §12 の参考系列見込み値を正式判定へ格上げする
- **`host-view-readout` の既定化・feature ゲート撤去**（PR #1411 の申し送り）: §12.4 の
  N=1024/2048 後退が未解決のため、既定化するにはこの後退の原因切り分け・対策が前提になる
- **N=2048 判定不能の解消**: #1258／#1234 Phase 2（tolerance 契約変更。ユーザー承認必須）→
  §14（イシュー #1260）で実測・解消を確認済み
- **N=1024/2048 の readout-on 後退の原因調査**: 本イシューのスコープ外（§12.4）。追跡する
  新規 issue を起票するかはユーザー判断。**追記（#1436）**: `docs/perf/cuda-host-
  view-readout-small-shape-regression.md` でフェーズ分解診断済み。増分が `matmul`
  （`d2h`）区間に集中する事実は交絡なく確定しているが、機構（glibc 動的 mmap 閾値適応の
  有無）は有力仮説であり、on 腕固有の原因特定は腕単体・プロセス分離計測が未実施のため
  仮説にとどまる（同ドキュメント §0/§8/§11。PR #1442 codex-review 指摘）。是正候補は
  #1437 へ引き継ぎ

## 13. 2026-09-07 追補: #1337 の正式記録・独自再現・公正性の論点

### 13.1 位置づけ

- CUDA 側の readout off/on 効果は既に §12（イシュー #1360）で実測済み。**本節はその記録を
  #1337 の正式な受け入れ記録として整理し**、加えて #1337 独自の実行（両実機横断計測の一部と
  して DGX で off/on を再実行）による再現性を追記する。§12 が採用する判定・数値そのものは
  再導出しない（同じ機構の結果を重複して結論化しない）
- #1337 は Metal（§12〈metal doc〉）・CPU（§15〈cpu doc〉両実機）・CUDA（本節）の 3
  バックエンド横断で実測を揃えることが受け入れ条件であり、CUDA 単体の判定は既に §12（本
  ドキュメント）で完結している
- **13.2 の位置づけ（重要な限定）**: §12.3 の readout off/on は `head-3d5e833-readout-off`／
  `head-3d5e833-readout-on` のとおり**両腕とも同一 sha の path 差し替え HEAD**（
  `fandhe_ai_source=path:<facade 絶対パス>`）で揃えており、readout feature 単独の効果を
  分離できている。一方 13.2 の独自実行（`docs/perf/logs/gemm-candle-gate-readout-1337/`
  配下の manifest・実行ログで確認済み）は **off 腕が `fandhe_ai_source=registry`（crates.io
  `fandhe-ai =0.7.0`）・on 腕のみ `fandhe_ai_source=path:<facade 絶対パス>`〈HEAD
  `1c298ff...`〉**であり、off/on 間でライブラリのソース（バージョン・ビルド）自体が異なる。
  13.2 の off/on 差分には readout feature の効果に加え、v0.7.0 公開後にマージされた
  #1342（128×64 cp.async pipeline 本番結線）等のコード差分が混入しており、**13.2 単独の
  データからは readout（#1337）への効果を分離帰属できない**。13.2 は「§12.3 の on 腕
  （HEAD path・readout-on）との再現性確認」の範囲でのみ有効な参考値として扱い、13.2 の
  off 腕・off/on 比較（「符号が一致」を含む）は #1337 の効果としては撤回する（本欄が本来
  比較すべき対象は §12.3 の readout-off 腕であり、13.2 の off 腕〈registry〉ではない）。
  同一 HEAD source での 13.2 off 腕の再計測は本エージェント実行環境に CUDA 実機接続手段が
  ないため未実施のまま残す

### 13.2 #1337 独自実行による再現性確認（DGX Spark GB10）

- 転送元 sha: `1c298ff5641b948dae3c1c65699930054af8f747`（§12 の `3d5e833` より後、PR #1420
  まで含む HEAD。両 sha 間の `crates/backend-cuda/src` 差分は #1415/#1420 の checksum
  device 側 reduction・テスト追加のみで GEMM 本体カーネル・readout 経路には変更なし）
- ラベル: `head-1c298ff-readout-off`／`head-1c298ff-readout-on`。**off/on 間でソースが揃って
  いない**（13.1 参照）: manifest 実測（`docs/perf/logs/gemm-candle-gate-readout-1337/
  run_gemm_gate_cuda-dgx-readout-{off,on}.log` の `依存元検証 OK` 行）で off 腕は
  `fandhe_ai_source=registry`（`GEMM_GATE_PATCH_FACADE_PATH` 未使用。crates.io
  `fandhe-ai =0.7.0`）、on 腕のみ `fandhe_ai_source=path:<facade 絶対パス>`（
  `GEMM_GATE_PATCH_FACADE_PATH` 使用・HEAD `1c298ff...`）＋
  `GEMM_GATE_BENCH_FANDHE_FEATURES=host-view-readout` であることを確認した。§12 と同一
  プロトコルなのは on 腕（HEAD path + feature）のみで、off 腕は §12.3 の
  `head-3d5e833-readout-off`（HEAD path・feature なし）とは異なる
- 実機: 計測前後とも `nvidia-smi utilization.gpu` 0〜8%・load average 1 桁台前半（他ジョブ
  混入なし。`docs/perf/logs/gemm-candle-gate-readout-1337/env_info.txt`）
- 実測結果:

  | N | readout-off 中央値（min–max, n=5） | readout-on 中央値（min–max, n=5） | candle fresh 中央値（n=5） | off の candle 比 | on の candle 比 | off 判定 | on 判定 |
  |---|---|---|---|---|---|---|---|
  | 1024 | 2.190 ms（2.125–2.307 ms） | 36.033 ms（34.787–36.162 ms） | 923.9 / 923.4 µs | 0.422 | 0.026 | 未達 | 未達（大幅後退） |
  | 2048 | 8.744 ms（8.368–8.764 ms） | 11.850 ms（11.788–11.942 ms） | - | 判定不能 | 判定不能 | 判定不能 | 判定不能 |
  | 4096 | 54.438 ms（52.367–60.003 ms） | 40.957 ms（40.848–41.547 ms） | 53.537 / 56.228 ms | 0.983 | **1.373** | 未達 | **達成** |

  fandhe-ai 側は全 30 run で `parity_fail_count=0`。checksum は off/on 全セル完全一致
  （`docs/perf/logs/gemm-candle-gate-readout-1337/compare_gemm_ab-cuda.md`）。N=2048 は
  candle 側が両腕とも決定的 `parity_fail_count=2`（§5・§12.5 と同一の既知事象）

- **on 腕（readout-on。HEAD path + feature）の符号は §12.3/§12.4 と完全に一致**（N=1024
  大幅後退・N=4096 のみ達成・N=2048 判定不能）。絶対値は run 間ノイズで多少異なる（本節
  N=4096: 1.373 倍 vs §12.3: 1.433 倍。いずれも 5 run の min–max 幅の内側で説明できる差）。
  **独立した invocation・わずかに後の sha による on 腕の再現性が確認できた**ことで、§12 の
  on 腕（readout 適用後）の頑健性が補強される
- **off 腕（本節。registry v0.7.0）と on 腕（本節。HEAD path + feature）の比較・「off/on
  candle 比」列・「off 判定」列は #1337 の readout 効果としては使わない**（13.1 の限定
  参照）。off 腕は §12.2 の正式系列 `0.7.0-1360`（同じく registry。N=1024: 2.445 ms・
  N=4096: 62.281 ms）とおおむね近い値（本節 N=1024: 2.190 ms・N=4096: 54.438 ms）であり、
  §12.3 の readout-off 腕（HEAD path・feature なし。N=1024: 2.382 ms・N=4096: 60.011 ms）
  とも大きくは外れないが、ソースが異なる以上厳密な A/B としては扱わない。#1337
  （readout 単独）への正式な帰属は §12.3／§12.4（同一 sha 揃え）を正とする

### 13.3 公正性の論点（親 #1334 受け入れ条件）

- candle 側ハーネス（`bench-candle`）は #1337 で変更していない（`to_vec2` のまま）。読み出し
  経路は各ライブラリの公開 API の一部であり、fandhe-ai 側の `host-view-readout` feature
  切替はハーネスの偏向ではなく製品側実装（借用ビュー readout。#1335/#1336 が用意した
  `VarHostView`／`Tensor::host_slice`）の測定である。checksum 完全一致・parity 0 fail
  （fandhe-ai 側）を根拠として、切替が数値契約を変えていないことも確認済み（§12.3・本節
  13.2）
- 一方、candle には対応する借用 API が無い／使用していないため、`iter_total` 境界の比較が
  「読み出し方式そのものの差」を一部含むことは限界として明記する（§12.4 のとおり）
- **#1336 の非到達**: `Var::matmul` 出力は `gemm` 内部readback で既にホスト常駐 `Tensor`
  のため、CUDA pinned host staging（`MemoryOps::with_host_view`。#1336）はこの readout
  経路を通らない。本節・§12 で観測した効果は `#1337`（borrowed-view readout そのもの）に
  帰属し `#1336` には帰属しない（`docs/perf/cuda-host-view-staging-readout.md` §7 と整合）

### 13.4 スコープ・ユーザー判断事項

- `host-view-readout` 既定化の可否は本節では判断しない（N=1024/2048 の大幅後退が未解決。
  §12.7 のユーザー判断事項を参照。据え置き）
- N=1024/2048 readout-on 後退の原因調査は本イシューのスコープ外（§12.4・§12.7 と同じ）。
  **追記（#1436）**: `docs/perf/cuda-host-view-readout-small-shape-regression.md` で
  フェーズ分解診断済み。機構（glibc 動的 mmap 閾値適応の有無）は有力仮説で、on 腕固有の
  原因特定は分離計測未実施のため仮説にとどまる（§12.7 の追記と同じ整理）
- #1031 の正式判定（registry ピン）は §11 のまま不変（本節・§12 とも参考系列のみ）

## 14. 2026-09-08 追補: 承認済み tolerance 契約下での N=2048 再計測（イシュー #1260）

### 14.1 位置づけ・プロトコル

- イシュー #1241（2026-09-08 承認）で tolerance 契約変更（候補 A-1・係数 `c=0.5`。
  `docs/candle-parity-tolerance-contract-decision.md` §8）が承認され、Phase 2 実装として
  #1247（`bench-common::parity` へ第 3 救済項 `parity_scaled_abs_bound`／
  `parity_scaled_abs_rescued` を追加。`bench-candle` は救済ありの `GemmReference::verify`・
  `bench-fandhe` は救済なしの `verify_strict` を使用）・#1250（`compare_gemm_gate.py::
  _parity_check` の判定不能条件を新契約へ更新。candle 側 `rescued>0` は許容、fandhe-ai 側
  `rescued>0` は判定不能へ倒す fail-closed）がマージ済み（`origin/main`
  9f4af172a0586384a45328d29555c1bd08297ee6）。本追補は §5・§11.4・§12.5 で「判定不能」と
  記録し続けてきた N=2048 が、この契約下で実際に解消されるかを GB10 実機で確認する
- **コード変更なし**（tolerance 定数〈`PARITY_REL_TOL`/`PARITY_ABS_TOL`〉・
  `PARITY_SCALED_ABS_COEFF`／`F32_UNIT_ROUNDOFF`・判定式・`bench-common`・
  `compare_gemm_gate.py`・`crates/`・`docs/spec/` は一切変更していない。本追補は既存契約下の
  **実測確認のみ**）
- プロトコルは §2 と同一（`run_gemm_gate_cuda.sh 0.7.0-1260`・`compare_gemm_gate.py --device
  cuda`）。**正式系列のみ**を計測した（`GEMM_GATE_PATCH_FACADE_PATH`・
  `GEMM_GATE_BENCH_FANDHE_FEATURES` いずれも未指定。manifest で `fandhe_ai_source=registry`・
  `candle_core_source=registry`・`bench_fandhe_features=""` を確認済み）。参考系列（HEAD
  path 差し替え）は計測していない（本追補の目的は契約変更の効果確認であり、正式ピンが
  `fandhe-ai =0.7.0` のまま変わっていないため §11／§12 の正式系列と同じ位置づけで比較できる）
- 計測環境: DGX Spark GB10・driver 580.173.02・CUDA 13.0.88・rustc 1.97.0。計測直前の GPU
  利用率 0 %・load average 0.16/0.07/0.05（実質アイドル）。ただし本 run は `bench-candle`
  （CUDA feature）のフルビルド直後に開始しており、計測完了後の load average は 3.54（直前の
  ビルド残余負荷。計測対象プロセスの割り込みではない）。詳細は
  `docs/perf/logs/cuda-gemm-candle-gate-1260/env_info.txt`
- 生データ: `scripts/bench/framework-compare/results/raw/results-dgx-gemm-gate-0.7.0-1260.jsonl`
  （30 行）・`skipped-dgx-gemm-gate-0.7.0-1260.log`（空）・
  `manifest-dgx-gemm-gate-0.7.0-1260.json`。実行ログ・判定表出力:
  `docs/perf/logs/cuda-gemm-candle-gate-1260/`（`run_gemm_gate_cuda-dgx-0.7.0-1260.log`・
  `compare_gemm_gate-0.7.0-1260.md`）

### 14.2 実測結果（正式系列 `0.7.0-1260`）

| N | fandhe-ai reuse 中央値（min–max, n=5） | candle fresh 中央値（n=5） | candle/fandhe | GFLOP/s（fandhe） | 判定 |
|---|---|---|---|---|---|
| 1024 | 2.221 ms（2.056–2.456 ms） | 924.5 µs | 0.416 | 966.8 | 未達 |
| 2048 | 8.825 ms（8.509–9.907 ms） | 4.205 ms | 0.476 | 1946.7 | **未達（candle 救済 2 要素。判定不能ではなく確定）** |
| 4096 | 59.677 ms（48.201–62.048 ms） | 54.868 ms | 0.919 | 2303.1 | 未達 |

fandhe-ai 側は全 15 run（N=1024/2048/4096）で `parity_fail_count=0` **かつ**
`parity_scaled_abs_rescued=0`（`verify_strict` 経路。救済項に依存せず従来どおり厳密ゼロ fail）。
`compare_gemm_gate.py` の終了コードは 3（N=1024/4096 の「未達」判定が残るため。仕様どおりで
失敗ではない）。

**§11.2（0.381／判定不能／0.904）・§12.2（0.379／判定不能／0.894）と比べ N=1024（+9〜12%）・
N=4096（+1.7〜2.8%）がやや高い値を示しているが、これはコード変更によるものではない**
（正式ピン `fandhe-ai =0.7.0` は不変。tolerance 判定側のみが変わった）。N=4096 の fandhe-ai
reuse 中央値は 5 run で 48.201–62.048 ms と従来（§11.2: 61.783–63.038 ms・§12.2:
61.961–62.860 ms）より明らかに広い分散を示しており、直前に完了した `bench-candle`（CUDA
feature）のフルビルド残余負荷（env_info.txt の load average 3.54）が計測序盤の run に影響した
ためと考えられる（原因の厳密な切り分けは本追補のスコープ外。判定〈未達〉自体には影響しない）。

### 14.3 N=2048 の要素単位判定（中核）

candle 側 N=2048 fresh の 5 run すべてで以下が完全に決定的に一致した
（`docs/perf/logs/cuda-gemm-candle-gate-1260/compare_gemm_gate-0.7.0-1260.md`）:

```
parity_fail_count = 0        (旧契約〈救済なし〉では 2)
parity_total      = 4194304
parity_max_abs_err = 3.623962e-05   (§5.1・§11.4・§12.5 と完全同一)
parity_max_rel_err = 2.811288e-01   (同上)
parity_scaled_abs_bound   = 1.525878e-05
parity_scaled_abs_rescued = 2
```

`bound=1.525878e-05` は机上評価（`docs/perf/candle-parity-tolerance-candidates.md` §4:
`c=0.5, eps=2^-24` で `bound≈1.526e-05`）と実測が一致し、`rescued=2` は §5.3／イシュー #1184
で特定した fail 2 要素（`idx=13850`・`idx=4130484`。§5.3 の表: `|actual−exact|` が
9.165e-06／1.717e-06）が両方とも新設の第 3 項（`c・u・K・S_A・S_B ≈ 1.526e-05`）以下に収まり
救済されたことと整合する（両要素とも `|actual−exact|` は `bound` 未満）。

**重要な注意（誤読防止）**: `parity_max_abs_err`（3.623962e-05）は **pass 要素も含む全要素中の
最大値**であり `bound`（1.525878e-05）を上回ったまま変化していない。これは §5.3 で既に
確認したとおり `parity_max_abs_err`／`parity_max_rel_err` は「fail 対象要素」の値ではなく
「複合判定を独立に通過した別の要素」に由来しうるためであり（§5.3「『0 近傍』の再評価」節）、
`fail_count=0` かつ `max_abs_err > bound` が両立することは救済ロジックの不具合ではない
（`_parity_check` は `fail_count`／`rescued`／`bound` の整合〈`rescued>0` なら
`bound >= CHECKSUM_ABS_TOL`〉のみを検査し、`max_abs_err` 自体を判定に使わない設計。
`compare_gemm_gate.py:384-402`）。

### 14.4 fandhe-ai 側の不変確認

全 15 run（N=1024×5・N=2048×5・N=4096×5）で `parity_fail_count=0` **かつ**
`parity_scaled_abs_rescued=0`（`bound` は `verify_strict` 経路のため `0.000000e+00` 固定。
`bench-fandhe::run_gemm` 系は `ScaledAbsTolerance::NONE` を使うため構造的に救済に依存しない。
`scripts/bench/framework-compare/README.md`「承認済み契約の実装（イシュー #1247）」節）。
`compare_gemm_gate.py::_parity_check` の fail-closed 検査（`framework=="fandhe-ai"` かつ
`rescued>0` は判定不能へ倒す）が発火しないことを実測で確認した。

### 14.5 #1031 後継ゲート判定表

| # | #1031 の受け入れ条件 | 正式系列（`0.7.0-1260`） | 出典 |
|---|---|---|---|
| 1 | N=1024 reuse で candle 超え | 未達（0.416 倍） | §14.2 |
| 2 | N=2048 reuse で candle 超え | **未達（0.476 倍。判定不能ではなく確定判定）** | §14.2・§14.3 |
| 3 | N=4096 reuse で candle 超え | 未達（0.919 倍） | §14.2 |
| 4 | parity 0 fail（fandhe-ai 側） | 達成（全 15 run `parity_fail_count=0` かつ `rescued=0`） | §14.4 |

**総合判定: tolerance 契約変更（#1241 承認・#1247/#1250 実装）により N=2048 の「判定不能」は
解消し、3 形状すべてで確定判定（いずれも未達）が得られるようになった。** #1031 自体は
すでに #1185（イシュー #1185・2026-09-06）でユーザー指示によりクローズ済みで後継ツリー
#1234 へ引き継がれているため、本追補は #1234 配下のイシュー #1258（N=2048 判定不能の解消）の
実測完了を記録するものであり、#1031 の再判定を行うものではない。

**§11.2・§12.2 との再現性**: N=1024（0.381→0.379→0.416 倍）・N=4096（0.904→0.894→0.919 倍）は
§14.2 で述べたとおり測定序盤のビルド残余負荷による分散拡大が主因と考えられ、いずれも「未達」の
方向性自体は 3 回とも一致している。N=2048 は今回初めて candle/fandhe 比（0.476 倍）を確定値
として記録できた（§11.2・§12.2 では「-」表記のまま参考比も併記していなかった）。

### 14.6 スコープ外・ユーザー判断事項

- **tolerance 定数・判定式は不変**（`PARITY_REL_TOL`/`PARITY_ABS_TOL`/
  `PARITY_SCALED_ABS_COEFF`/`F32_UNIT_ROUNDOFF` を含め本追補で一切変更していない）
- **本体 `assert_parity`／`ParityBaseline`（`crates/backend-cuda/tests/common/
  parity_baseline.rs`）は対象外**（#1254／#1256 は承認スコープ外〈ハーネス限定〉として
  NOT_PLANNED クローズ済み。`docs/candle-parity-tolerance-contract-decision.md` §9）
- **N=1024/N=4096 の未達（0.416／0.919 倍）の扱い**: #1234 後継ツリーの既存スコープ
  （reuse 計測境界の再定義・カーネル改善の希釈問題。§4.3・§8）に従い、本追補では対応しない
- **N=4096 の分散拡大の原因調査**: 本追補のスコープ外。次回の正式再計測（crates.io 次回公開・
  ピン更新後）で、直前にフルビルドを挟まないプロトコル（既存バイナリキャッシュを使う・
  ビルドと計測の間にアイドル待機を挟む）を徹底することで再発を避けられるかはユーザー判断
- **CPU 側（イシュー #1262）は本追補の対象外**（並走イシュー。`docs/perf/cpu-gemm-candle-gate-
  remeasurement.md` 側で別途記録）

## 15. §15（イシュー #1438。借用ビュー readout の既定経路化後の GB10 再計測）

### 15.0 位置づけ

イシュー #1438 は、当初 `bench-fandhe` の cargo feature `host-view-readout`（既定 OFF・#1337）
として導入した借用ビュー readout を、#1436/#1437 の是正（`ReadbackDest::PretouchedFresh`。
CUDA D2H 宛先未タッチによる N=1024/2048 の後退の解消）を確認したうえで既定経路化し、feature
自体を撤去した。本節はその before/after 再計測を記録する。**正式判定（`fandhe-ai =0.7.0` 系列
の #1031 達成可否）は §11／§14 のまま不変**（ピン未更新のため §14 と同値の未達成が継続する）。
本節は「feature 撤去・既定化」というコード変更そのものが性能を後退させていないかの確認記録。

### 15.1 プロトコル

`docs/perf/logs/gemm-candle-gate-readout-default-1438/env_info.txt` を参照。要点:

- before（legacy）腕: base コミット `fddca17`（origin/main。#1437 マージ直後）の bench-fandhe
  ソース（旧 feature 既定 OFF＝legacy `to_tensor()+to_vec()` 経路）
- after（default）腕: 本 PR HEAD の bench-fandhe ソース（借用ビュー readout が既定・feature 撤去済み）
- 両腕とも本 PR HEAD の同一 `crates/facade` へ path patch（facade ソースを揃え readout 経路の
  差のみを分離。#1337/#1436 §13「§12」で指摘された facade ソース不一致の反省を踏まえる）
- GB10 実機実測 2026-09-08。GPU 空き判定は `nvidia-smi --query-compute-apps`（常駐 2 プロセスの
  み・utilization.gpu 0%）。低負荷ウィンドウで計測

### 15.2 実測結果（fandhe-ai reuse・5 回計測中央値・before/after 比較）

| N | before 中央値（min–max） | after 中央値（min–max） | after/before | checksum |
|---|---|---|---|---|
| 1024 | 2.311 ms（2.177–2.454 ms） | 2.161 ms（2.118–2.288 ms） | 0.935 | 完全一致 |
| 2048 | 9.651 ms（9.357–10.050 ms） | 8.535 ms（8.171–9.104 ms） | 0.884 | 完全一致 |
| 4096 | 57.177 ms（55.873–57.512 ms） | 37.503 ms（36.819–38.278 ms） | **0.656** | 完全一致 |

全 3 サイズで非後退（`after/before < 1.00`）。checksum は 3 サイズとも完全一致（要素単位
`parity_fail_count=0`）。N=4096 は 1.53 倍の顕著な改善を示す（借用ビュー readout の
`host_copy`〈追加コピー省略〉が主因と推定。#1182 の診断と整合）。
出典: `docs/perf/logs/gemm-candle-gate-readout-default-1438/compare_gemm_ab-cuda-dgx.md`

### 15.3 candle 比ゲート（#1031。参考系列としての記録。正式判定は §14 のまま不変）

| N | before candle 比 | before 判定 | after candle 比 | after 判定 |
|---|---|---|---|---|
| 1024 | 0.404 | 未達 | 0.428 | 未達 |
| 2048 | 0.438 | 未達（candle 救済 2 要素） | 0.494 | 未達（candle 救済 2 要素） |
| 4096 | 0.959 | 未達 | **1.489** | **達成** |

出典: `docs/perf/logs/gemm-candle-gate-readout-default-1438/gate_cuda-dgx-{legacy,default}.md`。
N=4096 が after 腕で #1031 の受け入れ条件（candle 超え）を満たす一方、N=1024/2048 は未達のまま
（形状依存の結果。#1360 §12 の参考系列 readout-on 実測と同符号）。**正式系列（`fandhe-ai =0.7.0`
ピン）は本 PR で feature 撤去のみ行いピン自体は更新していないため §14 の判定（3 形状とも未達）
は不変のまま**——本節の測定はピン更新後の正式系列とは異なる「facade を HEAD path patch で
揃えた参考系列」の before/after であることに注意。

### 15.4 公正性の論点

- `bench-candle` は本 PR で一切変更していない（`.to_vec2()` による所有 `Vec` 読み出しのまま）。
  fandhe-ai 側だけが借用ビュー・candle 側が所有コピーという非対称は本 PR 後も残る
  （`scripts/bench/framework-compare/README.md`「借用ビュー readout（既定経路）」節に明記）
- CUDA `#1336` の pinned host staging（`MemoryOps::with_host_view`）は本経路に到達しない
  （`docs/perf/cuda-host-view-staging-readout.md` §7。§13.4 と同じ注記）

### 15.5 採否

全 3 サイズ非後退・checksum 完全一致のため、CUDA については既定経路化を承認（ADOPT）。
判定木（§3.2 に相当する事前宣言）の条件 (a)「全 N で after/before ≤ 1.00」を満たすため、
runtime `Device` 分岐によるフォールバックは導入しない。

## 16. 2026-09-10 追補: 正式系列 `fandhe-ai =0.8.0` 再計測（イシュー #1489）

### 16.1 位置づけ・プロトコル

- v0.8.0 が 2026-09-09 に crates.io へ公開され（`release-all.yml` run 34417008617）、framework-compare
  の承認ピンが #1487（PR #1504）で `fandhe-ai =0.8.0` へ更新された。本追補は、この**正式系列**
  （registry 解決・path patch なし）で CUDA GEMM N=1024/2048/4096 reuse の candle 比ゲート
  （旧 #1031。後継ツリー #1234／#1468 Phase 5 #1473）を GB10 で再計測し確定判定を記録する。
  §15（#1438）は「HEAD path patch で facade を揃えた参考系列」の before/after であり、正式系列で
  #1438 以降の変更を計測するのは本追補が初めて
- **コード変更なし**（tolerance 定数・判定式・`bench-common`・`compare_gemm_gate.py`・`crates/`・
  `docs/spec/` は不変。本追補は既存契約下の実測記録のみ）
- **正式ピン `0.7.0` → `0.8.0` の差分のうち本計測経路（`bench-fandhe gemm cuda reuse`）に
  影響しうるもの**（`git log v0.7.0..v0.8.0`）:
  - 借用ビュー readout（`Tensor::host_slice`。#1404〈PR〉・#1337）と、その CUDA reuse
    N=1024/2048 後退を是正した `ReadbackDest::PretouchedFresh`（#1437・PR #1450）。bench-fandhe
    側は #1438 で既定経路化済み（`readout_method=borrowed-view-default-1438`）だが、0.7.0 ピンでは
    facade 側 API が未収録だった（#1487 以前は `bench_fandhe_pin_guard.sh` で早期停止）
  - f32 SIMT GEMM の 128×64×16 cp.async pipeline カーネルの形状条件付き本番結線
    （#1344・PR #1385。N≥1024 かつ K≥1024）
  - `HOST_STAGING_KIND` 既定の `Pinned` 化（#1478・PR #1494）は `MemoryOps::with_host_view`
    経路のみに作用し、本 gemm reuse 計測では非到達（`docs/perf/cuda-host-view-staging-readout.md`
    §7・§15.4 と同じ注記）
- プロトコルは §2／§14 と同一（`run_gemm_gate_cuda.sh 0.8.0-1489`・`compare_gemm_gate.py --device
  cuda`）。§14.6 の反省（直前フルビルドの残余負荷による分散拡大）を受け、**ビルドを計測から分離**
  した: `bench-fandhe`／`bench-candle`（CUDA feature）を prebuild したうえで事前宣言の専有ゲート
  （1 分 load average < 1.0 かつ utilization.gpu 0 % を 30 秒間隔・連続 3 サンプル。最大 20
  サンプルで不成立なら `verdict=undetermined` を 1 回記録して終了）を通過してから計測した
  （`docs/perf/logs/cuda-gemm-candle-gate-0.8.0-1489/orchestrate.sh`）。ゲートは sample 1〜3
  （load1 0.10／0.18／0.11）で通過
- 計測環境: DGX Spark GB10・driver 580.173.02・CUDA 13.0.88・rustc 1.97.0。詳細は
  `docs/perf/logs/cuda-gemm-candle-gate-0.8.0-1489/env_info.txt`
- 生データ: `scripts/bench/framework-compare/results/raw/results-dgx-gemm-gate-0.8.0-1489.jsonl`
  （30 行）・`skipped-dgx-gemm-gate-0.8.0-1489.log`（空）・`manifest-dgx-gemm-gate-0.8.0-1489.json`
  （`fandhe_ai_source=registry`・`candle_core_source=registry`・`bench_fandhe_features=""`・
  `readout_method=borrowed-view-default-1438`）。実行ログ・判定表出力・ゲートログ:
  `docs/perf/logs/cuda-gemm-candle-gate-0.8.0-1489/`

### 16.2 実測結果（正式系列 `0.8.0-1489`）

| N | fandhe-ai reuse 中央値（min–max, n=5） | candle fresh 中央値（n=5） | candle/fandhe | GFLOP/s（fandhe） | 判定 |
|---|---|---|---|---|---|
| 1024 | 2.156 ms（2.093–2.273 ms） | 924.7 µs | 0.429 | 996.0 | 未達 |
| 2048 | 8.473 ms（8.302–9.043 ms） | 4.200 ms | 0.496 | 2027.5 | 未達（candle 救済 2 要素） |
| 4096 | 38.398 ms（37.720–38.571 ms） | 56.738 ms | **1.478** | 3579.3 | **達成** |

出典: `docs/perf/logs/cuda-gemm-candle-gate-0.8.0-1489/compare_gemm_gate-0.8.0-1489.md`。
`compare_gemm_gate.py` の終了コードは 3（N=1024/2048 の「未達」が残るため。仕様どおり）。

- **§15.3 の参考系列 after 腕（0.428／0.494／1.489）と 3 形状とも誤差範囲内で一致**した。
  §15 が「facade を HEAD path patch で揃えた参考系列」として観測した readout 既定化の効果が、
  crates.io 公開版 `0.8.0` の正式系列でそのまま再現したことを意味する
- §14.2（`0.7.0`: 0.416／0.476／0.919）との差は N=4096 のみ顕著（59.677 ms → 38.398 ms・
  0.643 倍）で、§15.2 の after/before（0.656）と整合する。N=1024/2048 は §14.2 と同水準
  （それぞれ 2.221→2.156 ms・8.825→8.473 ms。約 3〜4 % 改善）で、§15.2 の 0.935／0.884 ほどの
  幅は出ていない（§14.2 側の計測がビルド残余負荷下だった点も含め、ノイズ範囲と判断）
- N=4096 の run 間分散は 37.720–38.571 ms（幅 2.3 %）で、§14.2（48.201–62.048 ms）より大幅に
  縮小した。ビルドを計測から分離したプロトコルが §14.6 の懸念を解消したことを示す
- candle 側は 3 形状とも §14.2 と同水準（924.5→924.7 µs・4.205→4.200 ms・54.868→56.738 ms）で、
  比較基準の変動ではなく fandhe-ai 側の改善が N=4096 達成の要因であることが確認できる

### 16.3 要素単位判定

- fandhe-ai 側: 全 15 run で `parity_fail_count=0` **かつ** `parity_scaled_abs_rescued=0`
  （`verify_strict` 経路。§14.4 と同じ）
- candle 側 N=2048: 5 run すべてで `parity_fail_count=0`・`parity_scaled_abs_rescued=2`・
  `parity_scaled_abs_bound=1.525878e-05`・`max_abs_err=3.623962e-05`・`max_rel_err=2.811288e-01`
  （§14.3 と完全同一の決定的再現。承認済み契約 A-1 の救済項で確定判定となる）
- candle 側 N=1024: `rescued=0`（`bound=7.629375e-06`）・N=4096: `fail_count=0`・`max_abs_err=0`
  （`bound=3.051757e-05`）

### 16.4 ゲート判定表（正式系列 `0.8.0`。確定）

| # | 旧 #1031 の受け入れ条件 | 正式系列（`0.8.0-1489`） | 出典 |
|---|---|---|---|
| 1 | N=1024 reuse で candle 超え | 未達（0.429 倍） | §16.2 |
| 2 | N=2048 reuse で candle 超え | 未達（0.496 倍。candle 救済 2 要素・確定判定） | §16.2・§16.3 |
| 3 | N=4096 reuse で candle 超え | **達成（1.478 倍）** | §16.2 |
| 4 | parity 0 fail（fandhe-ai 側） | 達成（全 15 run `parity_fail_count=0` かつ `rescued=0`） | §16.3 |

**総合判定: 正式系列 `fandhe-ai =0.8.0` で N=4096 が初めて candle 比ゲートを達成した
（1.478 倍）。N=1024／2048 は未達のまま（0.429／0.496 倍）で、3 形状すべての達成という
旧 #1031 の受け入れ条件は依然として満たしていない。** §11／§14 の「3 形状とも未達」判定は
本追補で「N=4096 達成・N=1024/2048 未達」へ更新される（`docs/performance-targets.md` §8.14）。

### 16.5 スコープ外・引き継ぎ

- N=1024/2048 の未達要因は §4.3／§8・#1182（reuse 計測境界の `host_copy`／`checksum` 診断コスト
  が小形状ほど支配的）で整理済みの構造であり、本追補では対応しない（後継ツリー #1234／#1468 の
  既存スコープ）
- 公正性の論点（`bench-candle` は所有 `Vec` 読み出しのまま・fandhe-ai 側のみ借用ビュー）は
  §15.4 のまま不変
- 参考系列（HEAD path patch）は計測していない（HEAD とピンが同一版 `0.8.0` のため。次に
  `crates/` へ本計測経路に影響する変更が入った時点で §12 と同型の 2 系列併記へ戻す）

