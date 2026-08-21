# CUDA GEMM StreamK スケジューリング: 設計検討・採否判断（#812）

イシュー #812「perf(backend-cuda): クロスタイル先読み・XOR swizzle・StreamK の要否判断」の StreamK 節。
GEMM OSS 比較ギャップ改修ツリー #785 → Phase 5 親 #790 配下。`docs/backend-metal-splitk-decision.md`
（#810）・`docs/backend-metal-mlx-classic-nax-decision.md`（#549）と同型の決定記録として、機構要約・
本カーネル構成への定量当てはめ・採否判断を残す。**本イシューは設計検討（調査・机上分析・記録）であり、
`crates/backend-cuda/src/` は一切変更しない**（イシュー本文の明示。リポジトリ内に StreamK への言及は
本ドキュメント作成前は 0 件だった。`git grep -i streamk origin/main` で確認済み）。

## 判断サマリ

**不採用（保留）。** 理由は 2 点:

1. **主効果（tail effect 解消）が本カーネルの主要ワークロード（正方・大型 M=N=K=4096 系列）では小さい**
   （§2 の wave 定量化）。効果が見込める領域（小サイズ・非正方・K 支配的形状）は `gemm_auto.rs` の
   コストモデル選択・レイテンシ律速の領域であり、StreamK 導入の複雑度に見合わない
   （§2 末尾）。
2. **fixup（partial 結果の加算還元）がアキュムレート順序を変えるため、本リポジトリ全体の parity 非後退
   契約（「bit 一致 → tolerance・fixture 変更不要」という論拠。`tests/parity_nonregression.rs`）が
   成立しなくなる。** 採用するにはベースライン fixture の再生成が必要になり、これは
   `.claude/rules/coding-rust.md`「バックエンド間数値一致テストの許容誤差（tolerance）を単独で緩和
   しない」・`.claude/rules/security.md`「ガードレール閾値・ポリシー除外リスト・テスト許容誤差の変更は
   必ず人間（ユーザー）の承認を経る」に抵触するユーザー承認必須事項である（§4）。

**再評価条件**（§5）: 非正方・K 大の実ワークロードで tail effect が実測（ncu 等）で支配的と確認された
場合、かつ fixup 由来のアキュムレート順序変更を許容する parity 判定方式（現行の bit 一致論拠に代わる
統一複合判定「相対誤差 1e-3 未満 または 絶対誤差 1e-5 未満」への切替）についてユーザー承認を得た場合
に限り、再検討の対象とする。

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

waves  = grid / スロット数 = 2048 / 96 ≈ 21.33 waves
端数 wave の比率 ≈ 0.33 / 21.33 ≈ 1.5%
```

M=N=K=4096（および 2048・1024 系列も同様に grid が数百〜数千 blocks となり波数は 2 桁）では端数 wave
が支配する比率は数 % 未満に留まり、StreamK の主効果（tail effect 解消）による改善余地は小さい。

一方、小サイズ（例 M=N=512: `grid = ceil(512/128) * ceil(512/64) = 4 * 8 = 32 blocks < 96 スロット`）は
そもそも 1 wave 未満で全 SM を使い切れず端数 wave が支配的になるが、この領域は `gemm_auto.rs` の
コストモデルがカーネル選択（tiled CPU 経路との切替等）を担う帯域であり、レイテンシ律速（起動オーバー
ヘッド・カーネル起動レイテンシが支配的）でもあるため、StreamK 導入（fixup バッファ確保・追加カーネル
起動・K 分割ロジック）による複雑度増が見合わない。

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

上記 §2（主効果が小さい主要ワークロード）・§3（承認必須のアキュムレート順序変更を要する）の 2 点から、
本イシュー時点では **不採用（保留）** と判断する。実装（カーネル追加・fixup バッファ管理・K 分割
ロジック）そのものは行わない。

## 5. 再評価条件

以下の両方を満たした場合にのみ再検討の対象とする:

1. 非正方・K 大（例: M・N が小さく K が数千〜のワークロード）が実ワークロードとして要求され、かつ
   tail effect（wave quantization によるスループット低下）が実機計測（ncu の SM 稼働率・
   `smsp__cycles_active.avg.pct_of_peak_sustained_elapsed` 系メトリクス、または `gemm_mma_bench` の
   形状別 TFLOPS 比較）で支配的要因と確認された場合
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
