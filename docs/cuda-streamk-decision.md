# CUDA GEMM StreamK スケジューリング: 設計検討・採否判断（#812）

イシュー #812「perf(backend-cuda): クロスタイル先読み・XOR swizzle・StreamK の要否判断」の StreamK 節。
GEMM OSS 比較ギャップ改修ツリー #785 → Phase 5 親 #790 配下。`docs/backend-metal-splitk-decision.md`
（#810）・`docs/backend-metal-mlx-classic-nax-decision.md`（#549）と同型の決定記録として、機構要約・
本カーネル構成への定量当てはめ・採否判断を残す。**本イシューは設計検討（調査・机上分析・記録）であり、
`crates/backend-cuda/src/` は一切変更しない**（イシュー本文の明示。リポジトリ内に StreamK への言及は
本ドキュメント作成前は 0 件だった。`git grep -i streamk origin/main` で確認済み）。

## 判断サマリ

**不採用（保留）。** 理由は 2 点:

1. **主効果（tail effect 解消）は形状依存が大きく、最大の実測対象形状 M=N=K=4096 では小さいが
   （§2 の wave 定量化。quantization loss ≈ 3.0%）、中間サイズの正方形状（2048・1024）では無視できない
   水準（それぞれ ≈ 11.1%・≈ 33.3%）に達する**。したがって「主要ワークロードでは効果が小さい」という
   主張は 4096 系列にのみ成立し、2048・1024 系列を含めた一般論としては成立しない（§2）。この点だけでは
   StreamK 不採用の根拠として不十分であり、決定は主として理由 2（fixup のアキュムレート順序変更）に
   基づく。
2. **fixup（partial 結果の加算還元）がアキュムレート順序を変えるため、本リポジトリ全体の parity 非後退
   契約（「bit 一致 → tolerance・fixture 変更不要」という論拠。`tests/parity_nonregression.rs`）が
   成立しなくなる。** 採用するにはベースライン fixture の再生成が必要になり、これは
   `.claude/rules/coding-rust.md`「バックエンド間数値一致テストの許容誤差（tolerance）を単独で緩和
   しない」・`.claude/rules/security.md`「ガードレール閾値・ポリシー除外リスト・テスト許容誤差の変更は
   必ず人間（ユーザー）の承認を経る」に抵触するユーザー承認必須事項である（§4）。この理由は形状に依らず
   StreamK 全般に適用されるため、理由 1 の形状依存性とは独立に不採用（保留）の判断を支える。

**再評価条件**（§5）: 中間〜小サイズの正方形状（2048・1024）または非正方・K 大の実ワークロードで
tail effect が実測（ncu 等）で支配的と確認された場合、かつ fixup 由来のアキュムレート順序変更を
許容する parity 判定方式（現行の bit 一致論拠に代わる統一複合判定「相対誤差 1e-3 未満 または
絶対誤差 1e-5 未満」への切替）についてユーザー承認を得た場合に限り、再検討の対象とする。

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
「B-3」「B-4」節）。この順序不変性が、本リポジトリ全体の parity 非後退契約の論拠（「アキュムレート
順序が変わらない → 出力は bit 一致 → tolerance 定数・ベースライン fixture は変更不要」）を支えている
（例: `docs/perf/cuda-gemm-mma-ldmatrix-double-buffer.md` §4、`docs/perf/cuda-gemm-mma-bank-conflict.md`
「数値への影響（bit 一致）」節）。

StreamK は複数 threadblock が同一出力要素の部分和を独立に計算し、fixup フェーズで**それらを加算する
順序**（どの threadblock の部分和から先に加算するか）が生じる。浮動小数点加算は結合則を満たさない
ため、fixup の加算順序が実行のたびに変動しうる構成（実装によっては atomic 加算や non-deterministic な
SM スケジューリング順に依存する）では、出力が既存ベースラインと bit 一致しなくなる可能性が高い。

これは以下の規約に抵触する:

- `.claude/rules/coding-rust.md`「バックエンド間数値一致テストの許容誤差（tolerance）を単独で緩和しない
  （ポリシー除外リストのブラインドスポット対象）」
- `.claude/rules/security.md`「ガードレール閾値・ポリシー除外リスト・テスト許容誤差の変更は必ず人間
  （ユーザー）の承認を経る」

したがって StreamK を採用するには、(a) 既存の「bit 一致」論拠に代わる複合判定（`.claude/rules/
coding-rust.md` に既に定義済みの統一複合判定「相対誤差 1e-3 未満 または 絶対誤差 1e-5 未満」を
`kernels_mma.rs` 経路にも適用する）への切替と、(b) それに伴うベースライン fixture の再生成が必要に
なり、いずれもユーザー承認必須事項である。

## 4. 結論

§2 で示したとおり主効果（tail effect 解消）の大きさは形状依存であり、M=N=K=4096 では小さいが
2048・1024 系列では無視できない水準に達するため、主効果の小ささのみを根拠に不採用とすることはできない。
しかし §3（承認必須のアキュムレート順序変更を要する）は形状に依らず適用され、これ単独で不採用（保留）
の判断を支えるのに十分である。したがって本イシュー時点では **不採用（保留）** と判断する。実装
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
2. §3 のアキュムレート順序変更（fixup）を許容する parity 判定方式への切替について、ユーザー承認を
   得た場合（tolerance 緩和・ベースライン fixture 再生成を伴うため）

## 参考文献

- NVIDIA/cutlass（tag `v4.7.0`、commit `dcf215af68a2d08d305076c152a06f201728cd53`。
  `include/cutlass/gemm/kernel/threadblock_swizzle_streamk.h` 相当。BSD-3-Clause ライセンス）
- `docs/perf/sm121-device-attributes.md`（GB10 実 SM 数 48 実測記録。#482・#739・#777）
- `docs/cuda-tensor-core-design.md` §16（#804 Step F フォールバック。本番タイル定数未変更の経緯）
- `crates/backend-cuda/tests/parity_nonregression.rs`（parity 非後退契約の機械検査）
- `.claude/rules/coding-rust.md`（バックエンド間数値一致の統一複合判定・tolerance 緩和の承認要件）
- `.claude/rules/security.md`（ガードレール閾値・テスト許容誤差の変更承認要件）
