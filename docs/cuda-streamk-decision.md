# CUDA GEMM StreamK スケジューリング: 設計検討・採否判断（#812）

イシュー #812「perf(backend-cuda): クロスタイル先読み・XOR swizzle・StreamK の要否判断」の StreamK 節。
GEMM OSS 比較ギャップ改修ツリー #785 → Phase 5 親 #790 配下。`docs/backend-metal-splitk-decision.md`
（#810）・`docs/backend-metal-mlx-classic-nax-decision.md`（#549）と同型の決定記録として、機構要約・
本カーネル構成への定量当てはめ・採否判断を残す。**本イシューは設計検討（調査・机上分析・記録）であり、
本番カーネルの実装・定数は変更しない**（イシュー本文の明示。リポジトリ内に StreamK への言及は
本ドキュメント作成前は 0 件だった。`git grep -i streamk origin/main` で確認済み）。ただし
`crates/backend-cuda/src/kernels_mma.rs` の冒頭モジュールドキュメントコメント（`//!`。クロスタイル
先読み・XOR swizzle 節への本イシュー判断の参照追記）は本イシュー内で更新している。「本番カーネルの
実装・定数は変更しない」は本番カーネルの動作（本番経路のコード・定数）を指し、既存ソースの設計コメント
のみの更新はこれに含まれない。本番カーネル定数・`crates/backend-cuda/tests/`・`gemm*.rs`・
`swizzle.rs` は変更していない（`git diff origin/main -- crates/backend-cuda/src/kernels_mma.rs` は
コメント行のみの差分であることを確認済み）。

## 判断サマリ

**不採用（保留）。** 理由は 2 点:

1. **主効果（tail effect 解消）は形状依存が大きく、最大の実測対象形状 M=N=K=4096 では小さいが
   （§2 の wave 定量化。quantization loss ≈ 3.0%）、中間サイズの正方形状（2048・1024）では無視できない
   水準（それぞれ ≈ 11.1%・≈ 33.3%）に達する**。したがって「主要ワークロードでは効果が小さい」という
   主張は 4096 系列にのみ成立し、2048・1024 系列を含めた一般論としては成立しない（§2）。この点だけでは
   StreamK 不採用の根拠として不十分であり、決定は主として理由 2（fixup のアキュムレート順序変更）に
   基づく。
2. **fixup（partial 結果の加算還元）がアキュムレート順序を変えるため、非決定的な加算順序（atomic 加算
   等）を伴う実装では出力が実行のたびに変動しうるリスクがある。** `MmaF16` 経路は既に統一複合判定
   （「相対誤差 1e-3 未満 または 絶対誤差 1e-5 未満」）による非後退契約（fail_count・平均絶対誤差が
   記録済みベースラインを上回らないことを検査。bit 一致は合否条件ではない。
   `crates/backend-cuda/tests/parity_nonregression.rs`）で運用済みのため、アキュムレート順序が変わる
   こと自体は tolerance 定数・ベースライン fixture の変更を自動的には要求しない。**実測されていない
   のはリスクの有無**（fixup の加算順序が非決定的になり実行のたびに出力が変動する構成を取るか否か、
   および取った場合に既存 tolerance・fixture のまま非後退契約を通るか否か）であり、実機未到達
   （§1 冒頭）のため確認できていない。既存 fixture のまま非後退を通る場合は tolerance・fixture の
   変更は不要である一方、既存 baseline を悪化させる場合の受け入れ（fixture 再生成によるベースライン
   悪化の追認）はテスト弱体化に当たり、`.claude/rules/coding-rust.md`「バックエンド間数値一致テストの
   許容誤差（tolerance）を単独で緩和しない」・`.claude/rules/security.md`「ガードレール閾値・ポリシー
   除外リスト・テスト許容誤差の変更は必ず人間（ユーザー）の承認を経る」に抵触するユーザー承認必須事項
   となる（§4）。この理由は形状に依らず StreamK 全般に適用されるため、理由 1 の形状依存性とは独立に
   不採用（保留）の判断を支える。

**再評価条件**（§5）: 中間〜小サイズの正方形状（2048・1024）または非正方・K 大の実ワークロードで
tail effect が実測（ncu 等）で支配的と確認された場合、かつ StreamK 実装を実機で計測し既存 tolerance・
fixture のまま `assert_no_parity_regression` の非後退契約を通ることを確認した場合（通らない場合は
fixture 再生成についてユーザー承認を得た場合）に限り、再検討の対象とする。

## 1. StreamK の機構要約

出典: NVIDIA/cutlass（`Fandhe-AI` 外部リポジトリ調査。`docs/cuda-tensor-core-design.md` 参考文献節と
同じ tag `v4.7.0`・commit `dcf215af68a2d08d305076c152a06f201728cd53`。`include/cutlass/gemm/kernel/
threadblock_swizzle_streamk.h` 相当。BSD-3-Clause ライセンス、コード・コメントの転記は行わず機構の
事実関係のみを記載）。

標準の tile-based GEMM スケジューリング（本カーネル `kernels_mma.rs`・`kernels_wmma_opt.rs` を含む）は
出力 C を `MMA_BM x MMA_BN` タイル単位に分割し、1 threadblock が 1 タイル分の K 全反復を担当して
grid へ 1 回起動する（"data-parallel" 分割）。GPU の SM 数に対し grid のタイル数（wave 数）が割り切れ
ない場合、最終 wave は一部 SM のみが稼働し残りが遊休する（tail effect / wave quantization）。

StreamK は分割軸を「出力タイル」から「K 反復（と出力タイルの組）」へ変え、全 K 反復の総量を SM 数
（またはその倍数の "slot" 数）で均等分割する。1 つの出力タイルの K 反復が複数 threadblock（複数 SM）
にまたがって分担されうるため、各 threadblock は自分が担当した K 範囲分の部分和（partial sum）のみを
計算し、グローバルメモリ上の一時バッファへ書き出す。1 出力タイルの全部分和が出揃った後、別途 "fixup"
フェーズ（専用 threadblock、または該当タイルの最後の部分和を計算した threadblock 自身）がそれらを
加算して最終結果を確定する。これにより SM 稼働率を tile 境界に依存させず均等化できる（tail effect の
解消が主効果）。

## 2. 本カーネル構成への定量当てはめ（wave quantization の机上定量化）

- **実 SM 数**: 48（GB10・sm_121。`docs/perf/sm121-device-attributes.md` 実測。2026-08-19 出典イシュー
  #739・2026-08-20 再確認イシュー #777）
- **現行ブロックタイル**: `MMA_BM=64`・`MMA_BN=128`（`kernels_mma.rs::MMA_BM`/`MMA_BN`。#804 Step F
  フォールバックにより本番未変更のまま。`docs/cuda-tensor-core-design.md` §16）
- **1 SM あたりの常駐ブロック数**: 概算 2 blocks/SM（`docs/perf/cuda-gemm-mma-bank-conflict.md` §1
  「占有率への影響」節の cc 8.6 概算を踏襲。sm_121 実測は SMEM 容量のみ確認済みで常駐ブロック数の
  実機再確認は未了）→ **同時実行スロット数 ≈ 48 × 2 = 96**

主要ワークロード M=N=K=4096（`docs/perf/gemm-optimization-baseline.md` の対象形状の 1 つ・`gemm_mma_bench`
既定計測点）での grid サイズ:

```text
grid_x = ceil(N / MMA_BN) = ceil(4096 / 128) = 32
grid_y = ceil(M / MMA_BM) = ceil(4096 / 64)  = 64
grid   = grid_x * grid_y = 2048 blocks

waves      = grid / スロット数 = 2048 / 96 ≈ 21.33 waves（理想 waves。整数境界を無視した理論値）
実効 waves = ceil(waves) = 22 waves（実際に GPU が消化するラウンド数。端数 wave も 1 wave 分の
             レイテンシを要するため切り上げになる）
quantization loss ＝ (実効 waves − 理想 waves) / 実効 waves = (22 − 21.33) / 22 ≈ 3.0%
```

（端数 block 数 32 / 総 block 数 2048 ≈ 1.5% は「端数 wave に属する block の割合」であり、tail effect
による実際のコスト増（quantization loss）とは分母が異なる別の指標のため、上式とは一致しない）

M=N=K=4096 では quantization loss は 3.0% に留まる。しかし `docs/perf/gemm-optimization-baseline.md` の
対象形状に含まれる 2048・1024 系列は同じタイル・スロット数仮定でも grid サイズが大きく異なり、
quantization loss は無視できない水準になる（一般化せず個別に計算する）:

```text
# M=N=K=2048
grid_x = ceil(2048 / 128) = 16
grid_y = ceil(2048 / 64)  = 32
grid   = grid_x * grid_y = 512 blocks

waves      = 512 / 96 ≈ 5.33 waves（理想）
実効 waves = ceil(5.33) = 6 waves
quantization loss = (6 − 5.33) / 6 ≈ 11.1%

# M=N=K=1024
grid_x = ceil(1024 / 128) = 8
grid_y = ceil(1024 / 64)  = 16
grid   = grid_x * grid_y = 128 blocks

waves      = 128 / 96 ≈ 1.33 waves（理想）
実効 waves = ceil(1.33) = 2 waves
quantization loss = (2 − 1.33) / 2 ≈ 33.3%
```

grid サイズが小さくなるほど 1 wave あたりの block 数に対する端数の相対比率が増え、quantization loss は
急激に悪化する（4096: 3.0% → 2048: 11.1% → 1024: 33.3%）。したがって「主要系列では tail effect の
改善余地が小さい」という主張は M=N=K=4096 単体にのみ成立し、`gemm-optimization-baseline.md` の対象
形状全体（2048・1024 を含む）へ一般化することはできない。StreamK の主効果自体は 2048・1024 系列では
むしろ相応の改善余地があり得る。

なお、より小さい形状（例 M=N=512: `grid = ceil(512/128) * ceil(512/64) = 4 * 8 = 32 blocks < 96 スロット`）
はそもそも 1 wave 未満で全 SM を使い切れず端数 wave が支配的になる。**この領域についての先行する記述
「`gemm_auto.rs` のコストモデルが tiled CPU 経路との切替を担う」は実装と一致しないため訂正する**:
`crates/backend-cuda/src/gemm_auto.rs`（`CudaGemmAuto`）のコストモデルは `tensor_core::dispatch::
select_gemm_kernel` の決定規則に従い naive／tiled（`CudaGemm`）／WMMA f16（`CudaWmmaGemm`）という
**CUDA 内の複数カーネル候補間**の選択を行うものであり、CPU 経路への切替は行わない。加えて、本番
GEMM 経路（`crates/backend-cuda/src/lib.rs` の `ops::CudaBackendOps::gemm`）は現時点で
`CudaGemm::run_tiled_f32`（tiled 固定）に直結しており、`CudaGemmAuto` を介した自動選択へは未結線
（`lib.rs` 冒頭コメント「既定カーネル変種の選択は保守的に tiled 固定とし、`CudaGemmAuto` を介した
Tensor Core 経路の自動選択への切替は別スコープ」）。したがって小サイズ形状も含め、本番経路は現状すべて
同一の tiled CUDA カーネルを通り、サイズによる経路分岐（CPU/GPU 切替）は存在しない。

小サイズ領域での StreamK 導入（fixup バッファ確保・追加カーネル起動・K 分割ロジック）の要否は、実在
しない経路分岐を根拠にはできない。一方でこの領域はカーネル起動オーバーヘッド・レイテンシが相対的に
支配的になりやすく、fixup フェーズ（追加のグローバルメモリ書き出し・別カーネル起動）のコストが
StreamK の得る改善を相殺しうると見込まれるが、これは机上の見込みであり定量化した実測（起動レイテンシ・
fixup オーバーヘッドの ncu 計測等）は本イシュー時点で未実施である。小サイズ領域の採否は §5 の再評価
条件の対象とし、本ドキュメントでは結論を留保する。

## 3. fixup とアキュムレート順序変更のリスク

data-parallel 分割（現行）では、1 出力要素のアキュムレートは単一 threadblock 内の単一 warp が
「K タイル t 順 → kstep 順」の固定順序で `mma.sync` を発行して計算する（`kernels_mma.rs` 冒頭コメント
「B-3」「B-4」節）。この順序不変性を根拠に、B-3〜B-5 系の性能改修（`docs/perf/
cuda-gemm-mma-ldmatrix-double-buffer.md` §4・`docs/perf/cuda-gemm-mma-bank-conflict.md`「数値への影響
（bit 一致）」節）は「アキュムレート順序が変わらない → 出力は bit 一致 → tolerance 定数・ベースライン
fixture の変更検討自体が不要」という、実測を要さない最も強い形の論拠で数値後退リスクをゼロと確認して
きた。**これは `MmaF16` 経路の parity 判定方式そのものが bit 一致を要求しているという意味ではない**:
`crates/backend-cuda/tests/parity_nonregression.rs` 冒頭コメントのとおり `MmaF16` 系経路は既に REQ-2
統一複合判定（「相対誤差 1e-3 未満 または 絶対誤差 1e-5 未満」）による非後退契約
（`assert_no_parity_regression`。fail_count・平均絶対誤差が記録済みベースラインを上回らないことを
検査し、bit 一致は合否条件ではない）で運用されている。上記の bit 一致論拠は「アキュムレート順序を
変えない改修であれば実測なしで非後退契約を通過できる」という、より強い十分条件を使って実機実測の
手間を省いていたに過ぎない。

StreamK は複数 threadblock が同一出力要素の部分和を独立に計算し、fixup フェーズで**それらを加算する
順序**（どの threadblock の部分和から先に加算するか）が生じるため、上記の bit 一致論拠（実測不要の
十分条件）は使えなくなる。浮動小数点加算は結合則を満たさないため、fixup の加算順序が実行のたびに
変動しうる構成（実装によっては atomic 加算や non-deterministic な SM スケジューリング順に依存する）
では、出力が既存ベースラインと bit 一致しなくなる可能性が高い。**しかし bit 一致しないことは、
既存 tolerance・fixture のまま `assert_no_parity_regression` の非後退契約自体に落ちることを意味しない**
——順序変更由来の差分が既存の相対誤差 1e-3・絶対誤差 1e-5 の範囲に収まり、かつ fail_count・平均絶対
誤差が記録済みベースライン以下であれば、tolerance 定数・fixture を一切変更せずに非後退契約を通過
しうる。この判定は実機での `assert_no_parity_regression` 実行でしか確認できず、本イシューは実機未到達
（§1 冒頭）のため未実施である。

したがって規約抵触が生じるのは「順序変更を行えば」ではなく「実測の結果、既存 tolerance・fixture のまま
非後退契約に落ちた場合」に限られる。その場合に fixture を悪化させて再生成することは以下の規約に抵触
するユーザー承認必須事項である:

- `.claude/rules/coding-rust.md`「バックエンド間数値一致テストの許容誤差（tolerance）を単独で緩和しない
  （ポリシー除外リストのブラインドスポット対象）」
- `.claude/rules/security.md`「ガードレール閾値・ポリシー除外リスト・テスト許容誤差の変更は必ず人間
  （ユーザー）の承認を経る」

StreamK を採用するには、まず既存 tolerance・fixture を変更しない状態で実機実測して非後退を判定し、
非後退に落ちた場合に限り fixture 再生成についてユーザー承認を検討する、という順序を踏む必要がある。
本イシュー時点ではこの実測自体が実機未到達により未実施であるため、規約抵触の有無は確定していない
（「抵触する」ではなく「抵触するかどうかが実測なしには判定できない」がより正確な現状認識であり、
不採用（保留）の理由はこのリスク未確定性そのものにある）。

## 4. 結論

§2 で示したとおり主効果（tail effect 解消）の大きさは形状依存であり、M=N=K=4096 では小さいが
2048・1024 系列では無視できない水準に達するため、主効果の小ささのみを根拠に不採用とすることはできない。
また §3 で示したとおり、アキュムレート順序変更が直ちに tolerance・fixture の変更を要求するわけではなく、
既存の非後退契約（`assert_no_parity_regression`）のまま通過しうるかどうかは実機実測でしか確定できない。
この「実測なしにはリスクの有無を判定できない」状態そのものが、実機未到達（§1 冒頭）という制約下では
形状に依らず StreamK 全般に適用され、不採用（保留）の判断を支えるのに十分である。したがって本イシュー
時点では **不採用（保留）** と判断する。実装
（カーネル追加・fixup バッファ管理・K 分割ロジック）そのものは行わない。

## 5. 再評価条件

以下の両方を満たした場合にのみ再検討の対象とする:

1. 中間〜小サイズの正方形状（2048・1024。§2 の机上計算で quantization loss ≈ 11.1%・≈ 33.3%）または
   非正方・K 大の実ワークロードのいずれかが実ワークロードとして要求され、かつ tail effect（wave
   quantization によるスループット低下）が実機計測（ncu の SM 稼働率・
   `smsp__cycles_active.avg.pct_of_peak_sustained_elapsed` 系メトリクス、または `gemm_mma_bench` の
   形状別 TFLOPS 比較）で支配的要因と確認された場合。2048・1024 系列は §2 の机上計算で quantization
   loss が既に非小さいことを確認済みのため、実機計測は「効果があるか」ではなく「fixup オーバーヘッドを
   差し引いても正味の改善になるか」の確認に主眼を置く
2. §3 のアキュムレート順序変更（fixup）を含む StreamK 実装を実機で `assert_no_parity_regression`
   実行し、既存 tolerance・fixture のまま非後退契約を通ることを確認した場合。通らなかった場合は
   ベースライン fixture の再生成についてユーザー承認を得た場合（`.claude/rules/coding-rust.md`・
   `.claude/rules/security.md` により tolerance 緩和・fixture 再生成は人間承認必須のため）

## 参考文献

- NVIDIA/cutlass（tag `v4.7.0`、commit `dcf215af68a2d08d305076c152a06f201728cd53`。
  `include/cutlass/gemm/kernel/threadblock_swizzle_streamk.h` 相当。BSD-3-Clause ライセンス）
- `docs/perf/sm121-device-attributes.md`（GB10 実 SM 数 48 実測記録。#482・#739・#777）
- `docs/cuda-tensor-core-design.md` §16（#804 Step F フォールバック。本番タイル定数未変更の経緯）
- `crates/backend-cuda/tests/parity_nonregression.rs`（parity 非後退契約の機械検査）
- `.claude/rules/coding-rust.md`（バックエンド間数値一致の統一複合判定・tolerance 緩和の承認要件）
- `.claude/rules/security.md`（ガードレール閾値・テスト許容誤差の変更承認要件）
