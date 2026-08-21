# CUDA GEMM mma.sync warp 内 ldmatrix 先読みダブルバッファ計測記録（#495・B-4）

イシュー #495「perf(backend-cuda): kernels_mma.rs に warp 内 ldmatrix 先読みダブルバッファを追加」の実測記録テンプレート。
GEMM 性能改善ツリー #479 → Phase 2 親 #490 の B-4。先行 B-3（#494・ブロックタイル拡大）とのペアで、SMEM→レジスタのロードレイテンシ隠蔽によるスループット改善を狙う。

## 状態: 未実測・実機実行待ち（#502 で再計測）

本実装セッションの実行環境は CUDA **driver**（`libcuda`。compute capability 8.6・RTX 3060 実機）は存在するが NVRTC（`libnvrtc`）が存在しないため（`crates/backend-cuda/src/kernels_mma.rs` 冒頭コメント「検証状態」参照）、本ファイルが記録する変更（kstep ループのソフトウェアパイプライン化）は **NVRTC による構文検証を一度も通過していない**。sm_86（この実機）・sm_121（DGX Spark GB10。設計上のターゲット）のいずれでも未検証。`docs/perf/cuda-gemm-mma-block-tile.md`（B-3 時点の同種記録）・`docs/perf/metal-gemm-dynamic-tile.md` の先例（実機での最初の実行が構文検証を兼ねる）と同じ位置づけ。

本実装セッションで検証済みの事項:

- `cargo build --workspace`（`const _: () = assert!(...)` によるコンパイル時境界検査。変更なし・§1 参照）
- `cargo fmt --all -- --check` / `cargo clippy --workspace --all-targets --all-features -- -D warnings`
- `cargo test -p backend-cuda`（`kernels_mma.rs` 内 `#[cfg(test)]` の `#define` 整合検査・REQ-8 needle・ダブルバッファ構造ロック〈新規〉・タイル定数 pin・`gemm_mma.rs` の launch config div_ceil 被覆テスト）
- `tests/parity_nonregression.rs` の通常 CI 実行分（tolerance 定数 pin・fixture 自己整合。無変更で green）
- `git diff origin/main -- crates/backend-cuda/tests/parity_nonregression.rs crates/backend-cuda/tests/common/` が無差分（§4 の bit 一致論拠の機械確認）

未検証・実機実行待ちの事項（#502「Phase B 完了時点の再計測」へ引き継ぐ）:

- NVRTC によるカーネルソースの構文検証そのもの
- B-3（#494）比の TFLOPS 改善判定（下記 §3 記録欄）
- レジスタ予算実測（§5 リスク参照。`--ptxas-options=-v` 相当の function 属性確認）
- sm_121（DGX Spark GB10）実機の SMEM/レジスタ属性（`docs/perf/sm121-device-attributes.md` は未実測のまま）

## 1. 背景

B-3（#494）はブロックタイル拡大（`MMA_BM=64`・`MMA_BN=128`）でブロックスレッド数を 512（B-1 時点相当）へ回復させたが、warp 内の kstep ループ（`kernels_mma.rs` の `for (int kstep = 0; kstep < BK / MMA_K; ++kstep)`）自体は各 kstep で A/B フラグメントの `ldmatrix` を発行してから直後に `mma.sync` を発行する構造のままで、SMEM→レジスタのロードレイテンシが Tensor Core 演算とオーバーラップしない。

本イシュー（B-4）は CUTLASS `mma_multistage.h` の `PipeState`（`warp_loaded_frag_A_[2]` 等）・`mac_loop_iter`（次段のフラグメントを先読みしてから `warp_mma_` を呼ぶ）と同型の warp 内ダブルバッファを導入し、ロードレイテンシを隠蔽する。

## 2. 実装内容

- A/B フラグメントを 2 面バッファ化: `unsigned a_frag[WARP_TILES_M][4]` → `unsigned a_frag[2][WARP_TILES_M][4]`（B も同様に `unsigned b_frag[2][WARP_TILES_N][2]`）
- `ldmatrix` 発行箇所を「1 フラグメント単位」のマクロ化（`LDSM_A_FRAG(buf, stage, kstep, mi)`/`LDSM_B_FRAG(buf, stage, kstep, nj)`）。発行箇所自体は A/B 各 1 箇所のまま（既存テスト `mma_f16_source_issues_mma_sync_from_single_loop_site` と同じ「ループ化・非コピペ」方針をロードにも適用）。呼び出し側（プロローグ・kstep ループ内の先読み）が `#pragma unroll` 付き `mi`/`nj` ループでマクロを呼ぶ
- warp プロローグ: K タイル t ごとに kstep=0 のフラグメントをバッファ 0 へロードしてから kstep ループへ入る
- kstep ループ自体に `#pragma unroll` を付与（`BK / MMA_K` はコンパイル時定数のためトリップ回数既知）。これにより `cur = kstep % 2`・`nxt = (kstep + 1) % 2` がコンパイル時定数へ畳み込まれる
- kstep ループ内: `kstep + 1 < BK / MMA_K` のときのみ次段（kstep+1）のフラグメントをバッファ `nxt` へ先読みしてから、バッファ `cur` で mi×nj の `mma.sync` 4 発行を行う

`MMA_K_STEPS_PER_STAGE = BK / MMA_K = 2`（現構成）のため、kstep=1 のロードが kstep=0 の mma とオーバーラップする。

### `#pragma unroll` が cosmetic ではなく必須である理由（レビュー指摘反映）

`a_frag[buf][mi][...]`/`b_frag[buf][nj][...]` はインラインアセンブリの出力オペランド（`"=r"(...)`）の添字である。`buf`（`cur`/`nxt`）・`mi`/`nj` が実行時変数のままだと、レジスタ割り当ては実行時に添字で選べないため local memory へ溢れ、「ロードレイテンシを隠すはずの先読み最適化」がむしろ SMEM/local memory トラフィックを増やす性能後退になりうる（CUTLASS が `warp_mma_k` ループへ `CUTLASS_PRAGMA_UNROLL` を付ける理由と同じ）。よって kstep ループの `#pragma unroll` は cosmetic な最適化ヒントではなく、この実装が正しく機能するための前提条件である。カーネルソースのコメント（`LDSM_A_FRAG`/`LDSM_B_FRAG` マクロ直前・kstep ループ直前）とテスト（`mma_f16_source_uses_ldmatrix_double_buffer_structure` の `#pragma unroll` 位置検査）の両方に明記・固定した。

### `_Pragma` 演算子を採用しなかった判断

初版実装ではマクロ内で `for` ループと `_Pragma("unroll")` を完結させる設計（`LDSM_A_FRAGS(buf, stage, kstep)` が `WARP_TILES_M` 回のループごとロードする形）を検討したが、`_Pragma` 演算子は本ファイル内に前例がなく、NVRTC 上での挙動を実機なし（本ファイル冒頭「状態」節参照）で確認できないため不採用とした。代わりに「1 フラグメント単位」のマクロ＋呼び出し側の `#pragma unroll`（既存の mi/nj 二重ループ＝プリプロセッサ後の実際の文出現位置に置く形。`kernels_mma.rs` の mma.sync 発行側・エピローグ guarded store 側に前例あり）へ構成を変更した。呼び出し箇所は増える（プロローグ 2 箇所・先読み 2 箇所）が、`ldmatrix` の発行箇所自体は A/B 各 1 箇所のまま変わらない。

### クロスタイル先読みの不採用（意図的な CUTLASS からの縮小）

CUTLASS `mac_loop_iter` は `(warp_mma_k+1) % K` の wrap-around で**次タイルの kstep=0 まで**先読みするが、本実装は**タイル内先読みに限定**する（タイル境界を跨ぐ先読みは行わない）。理由:

- 本カーネルのループ内 `cp.async.wait_group (STAGES-2)` はイテレーション t の時点で**タイル t のグループ完了までしか保証しない**（`kernels_mma.rs` の #492 コメント「正しさ」参照）
- タイル t+1 の SMEM 完了保証前に読み出すクロスタイル先読みは、wait/sync 配置の大規模再構成（CUTLASS の `warp_mma_k==K-2` での段送り相当）を要し、NVRTC 構文検証不能な本環境ではリスクが高い
- タイル内限定でも #495 受け入れ基準（「A/B フラグメントを 2 面バッファ化し `warp_mma_k` ループ内で次段を先読み」「`kWarpGemmIterations >= 2` 構成で動作」）は満たす

クロスタイル先読みは残余の最適化余地として、B-5（#496・cp.async issue interleaving）以降または #502 実測後の判断へ引き継ぐ。

### #812 追加判断（実機到達不能・机上定量化。判断: 保留）

イシュー #812「perf(backend-cuda): クロスタイル先読み・XOR swizzle・StreamK の要否判断」の実装セッションでも
`docs/real-hardware-verification-env.local.md`・`CUDA_NODE` が不在で実機へ到達できなかった（#804・#803 と
同じ制約）ため、タイル切替ストールの実測（ncu の `smsp__average_warps_issue_stalled_short_scoreboard`／
`_barrier` 系メトリクス）は実行できていない。以下は机上での露出レイテンシ定量化に基づく判断。

- 現行構成 `MMA_BK=32`・`MMA_K=16` → `K_STEPS = BK / MMA_K = 2`。タイル内先読み（#495・本節上記）は
  kstep=1 のフラグメントロードのみを kstep=0 の `mma.sync` 発行と重ねられる。**タイル境界（各 K タイル
  先頭・kstep=0）のフラグメントロードは先読みで隠せず、全 ksteps の半数（1/2 = 50%）が非先読みのまま
  露出する**。#495 実装時点の記述（本ファイル §5 リスク節）は「クロスタイル先読み不採用」の主眼を
  同期バグリスク回避に置いており、露出比率の大小そのものは判断根拠にしていなかったため、ここで明示的に
  定量化する: 50% という比率は「小さい残差」ではなく、無視できない割合である
- ただし、この 50% は「mma 演算とオーバーラップしない cp.async 待ち」ではなく、SMEM 上に既に到着済みの
  データに対する **SMEM→レジスタの `ldmatrix` レイテンシのみ**が対象である（グローバルメモリのレイテンシは
  `cp.async` 3 ステージパイプライン〈`MMA_STAGES=3`〉が別途隠蔽している。本ファイル冒頭コメント「検証状態」
  節・`kernels_mma.rs` 冒頭コメント「B-3」節参照）。`ldmatrix` は SMEM 直読みのため、その絶対レイテンシは
  グローバルメモリ往復に比べ小さいと見込まれる（sm_121 の SMEM/L1 実効帯域・レイテンシ実測は
  `docs/perf/sm121-device-attributes.md` を参照するが、命令レベルのレイテンシそのものは同 doc の対象外
  であり実測なし）。すなわち露出比率（50%）は大きいが、露出**量**（絶対時間）が Tensor Core 演算に対し
  無視できない規模かどうかは実測なしには確定できない
- **クロスタイル先読みを実装する場合の同期リスクは #495 時点の判断から変わらない**: `cp.async.wait_group
  (STAGES-2)` はイテレーション t の時点でタイル t のグループ完了までしか保証しないため、タイル t+1 の
  SMEM 完了保証前に読み出すクロスタイル先読みは wait/sync 配置の大規模再構成（CUTLASS の
  `warp_mma_k==K-2` での段送り相当）を要し、NVRTC 構文検証不能な本環境ではリスクが高いままである
- **より安価な代替案**: `MMA_BK` を拡大すれば `K_STEPS = BK / MMA_K` が増え、非先読み kstep（各タイル
  1 回のみ）が全体に占める比率（`1 / K_STEPS`）が縮小する（`K_STEPS=2` → 50% 露出だが `K_STEPS=4`
  なら 25% 露出）。これは wait/sync 配置の再構成を要さず、`docs/perf/cuda-gemm-mma-block-tile-stages.md`
  §3 の `BM`/`BN`/`STAGES` 拡大候補表の延長線上にある机上検討事項であり、クロスタイル先読みより低リスク
  にタイル境界露出を縮小できる可能性がある（`BK` 拡大は SMEM 使用量も増やすため §3 の候補評価と合わせて
  検討する。本イシューでは候補算出のみに留め実装しない）

**判断: 保留（不採用ではなく再評価条件付き保留）。** 根拠は露出比率の小ささではなく、wait/sync 再構成の
同期バグリスクが NVRTC 構文検証不能な環境では依然許容できないため（リスク起点の判断。#495 時点の判断枠組みを
維持）。

**再評価条件**: (1) 実機到達後、ncu のストール比率メトリクス（`short_scoreboard`／`barrier`）でタイル
境界の露出が Tensor Core 発行密度に対し有意（例: 総ストールサイクルの 2 桁 % 台）と確認された場合、かつ
(2) NVRTC による構文検証が可能な環境（実機 CUDA toolkit 到達）で wait/sync 再構成の正しさをテスト
（`--ignored` parity テスト含む）で確認できる場合。(1) を満たさない場合は §2「より安価な代替案」（`BK`
拡大）を優先候補として#804 の候補表拡張で扱う。

## 3. 段階的計測手順（実機・CUDA driver + NVRTC 搭載・compute capability 8.0 以上）

```sh
git fetch origin
git checkout perf/495-mma-ldmatrix-double-buffer   # 本イシューの実装ブランチ
cargo test -p backend-cuda -- --ignored --nocapture   # parity 非後退の全行検査（数値一致確認を性能計測より先に実施）
cargo run -p backend-cuda --example gemm_mma_bench --release   # TFLOPS 計測
```

### 記録欄（実機セッションで埋める）

| 対象 | M=N=K=2048 TFLOPS（5 回中央値） | M=N=K=4096 TFLOPS（5 回中央値） | B-3（#494）比 |
|------|-------------------------------|-------------------------------|---------------|
| B-4（ダブルバッファ適用後） | 未計測 | 未計測 | 未計測 |

判定基準（#495 実装計画・受け入れ基準）: 4096 TFLOPS が B-3 完了時点を下回らないこと。B-3 自身も未実測（#502 で再計測予定）であるため、B-3/B-4 の非後退比較は #502 の実機セッションへ引き継ぐ。

## 4. 数値 bit 一致の論拠（parity 非後退契約）

各出力要素のアキュムレート順序は「K タイル t 順 → kstep 順」の `mma.sync` 系列のみで決まり、**ldmatrix の発行タイミング（先読み）は mma の発行順序・オペランド値を一切変えない**。よって出力は B-3（#494）時点と bit 一致であり:

- tolerance 定数（`RELATIVE_TOLERANCE`/`ABSOLUTE_RESCUE_THRESHOLD`）・ベースライン fixture（TF32: 42493/262144・mean_abs_diff 1.574e-3 等）は**変更しない**
- `git diff origin/main -- crates/backend-cuda/tests/parity_nonregression.rs crates/backend-cuda/tests/common/` が無差分であることを本実装セッションで機械確認済み（§「状態」参照）
- FMA 契約（f16 入出力・f32 アキュムレート、`mma.sync` の固定契約）は不変

## 5. リスク・安全側判断の記録

- **レジスタ予算増によるリスク**: フラグメントレジスタが 12 本→24 本へ倍増（アキュムレータ 16 本と合わせ約 40 本 + 索引類/thread）。compute capability 8.6 の 2 ブロック/SM 常駐条件（64 本/thread 以下）に近づくため、占有率低下でスループットが悪化する可能性がある。緩和策: (1) カーネルソースの構造差分のみで即時ロールバック可能、(2) 実機実測（`--ptxas-options=-v` 相当の function 属性確認を含む）は #502 で行い判定する
- **NVRTC 未検証**: 変更は既存 `ldmatrix`/`mma.sync` の発行位置とレジスタ配列次元のみで、新規 PTX 命令は追加していない。B-1〜B-3 で検証済みの命令列を維持するため構文リスクは増加しない
- **クロスタイル先読みの同期バグ回避**: タイル内先読みに限定することで、wait/sync 配置の再構成を要するクロスタイル先読みの同期バグリスクを回避した（§2 参照）
- **数値後退リスク**: bit 一致論拠（§4）+ parity 関連ファイル無差分の機械確認により正しさリスクはない。実機セッションでは ignored parity テストを性能計測より先に実行する運用を維持する

## 6. §4 parity 非後退契約の機械確認

```sh
git diff origin/main -- crates/backend-cuda/tests/parity_nonregression.rs crates/backend-cuda/tests/common/
```

無差分であることを確認する（§4 の bit 一致論拠の裏付け。tolerance 定数・ベースライン fixture を変更していないことをコミット前に検査する。本実装セッションで実施済み）。
