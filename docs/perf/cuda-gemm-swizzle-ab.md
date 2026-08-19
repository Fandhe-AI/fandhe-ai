# CUDA GEMM タイル→SM 割り当てスウィズル A/B 計測記録（#499）

イシュー #499「perf(backend-cuda): L2 再利用のためのタイル→SM 割り当てスウィズルを実装」の設計根拠・
A/B 計測手順・記録テンプレート。`crates/backend-cuda/src/kernels_mma.rs` の `MMA_F16` カーネル（f16
`mma.sync`/`ldmatrix`/`cp.async` 経路）の C タイル→SM（ブロック）割り当て順を、DeepGEMM
`get_swizzled_block_idx` 同型のグルーピング方式（M 方向を動的選択したグルーピング幅で束ね、各グループ内で
N を先に全走査してから次の M グループへ移る順序へ並べ替える）で並べ替える案の効果を計測する。

## 1. 設計根拠（確信度差の明示）

- **DeepGEMM**: 本方式は DeepGEMM の `get_swizzled_block_idx` と同型のグルーピング幅動的選択の考え方を
  踏襲するが、本実装は **M 方向グルーピング**（グループ内で N を先に全走査する。本ファイル冒頭「A/B
  計測手順」節参照）のため、選択式は `usage = g * BLOCK_M + ceil_div(num_sms, g) * BLOCK_N`
  （`crates/backend-cuda/src/swizzle.rs::swizzle_group_usage` が単一の真実源）である。DeepGEMM 自体の
  N-grouping 式（`g * BLOCK_N + ceil_div(num_sms, g) * BLOCK_M`）とは軸が入れ替わる点に注意
  （Bugbot 指摘・PR #667 レビュー是正）。DeepGEMM ではこの動的選択の考え方自体が production 経路として
  有効化されている（確信度: 高）。
- **MLX**: MLX にも shift/mask 式の同種機構（swizzle）が存在するが、classic GEMM 経路では無効化された
  ままであり、有効時の効果は同リポジトリ内で実証されていない（確信度: 低・未実証）。
- 本イシューはこの確信度差を踏まえ、DeepGEMM 同型の方式のみを実装対象とする（MLX 側の shift/mask 方式は
  対象外）。

## 2. 状態: 実測完了（2026-08-19・GB10 実機・commit `cbc16e7`）。現行基準では不採用確定・本番結線は行わない（f16 `mma.sync` 経路・#499。TF32 opt-staged 経路は §7 を参照）

**2026-08-19、DGX Spark GB10 実機で A/B 計測を実施し、下記「6. 実測結果」の TFLOPS 比較を確定させた**
（出典: イシュー #739・#740。5 回計測中央値・`gemm_mma_swizzle_bench --features internal-diagnostics`）。
下記「4. 判断基準」の既存基準（`size ∈ {2048, 4096}` の両方で改善）に照らすと 2048 が未達（中立）のため
**不採用が確定**し、**本番結線（`kernels_mma.rs::MMA_F16`／`gemm_mma.rs::CudaMmaGemm::launch_f16` への
swizzle remap の結線）は行わない**。「未計測の間は『採用済み』として扱わない」原則のとおり、判断が確定
していない状態で暫定的に結線することもしない。4096 単独では大きな改善があるため、判断基準自体を変更
（例: 4096 限定基準の新設）すれば別の結論になりうるが、これは性能採用基準の変更にあたりユーザー承認が
必須であり本ドキュメント時点では未承認である。承認を得た上での基準改定検討・（承認された場合の）本番
結線は後続イシュー #740 に引き継ぐ（下記「4. 判断基準」参照）。

以下は結線前の設計・実装経緯（2026-08-15 以前の状態）:

`crates/backend-cuda/src/swizzle.rs`（ホスト側参照実装・グルーピング幅動的選択）・
`kernels_mma.rs::mma_f16_source_with_swizzle`（変種ソース生成）・
`gemm_mma.rs::CudaMmaGemm::new_with_swizzle`（変種カーネルのコンパイル・保持）を実装したが、**本番カーネル
（`kernels_mma::MMA_F16` 定数）・本番ディスパッチ経路（`ops.rs`／`gemm_auto.rs`）は 1 バイトも変更していない**。

理由は #497（蛇行走査順）・PR #657 codex-review 指摘（P1: 性能改善を実測せずに本番カーネルへ変更を導入して
いる）と同型の判断: 本実装セッションの実行環境（RTX 3060・compute capability 8.6・NVRTC 非搭載）では mma
経路の NVRTC コンパイル・実行・A/B 計測ができないため（`kernels_mma.rs` 冒頭「検証状態」参照）、コード実装
（opt-in 経路・GPU 非依存の単体テストで検証可能な部分）のみをこの PR で完了し、実機 A/B 計測・採否確定は
実機ツリー #408 側のセッションへ引き継ぐ。

`#497` との差分: 受け入れ基準 1 項（動的グルーピング幅選択・全単射性のある remap アルゴリズム自体）は GPU
非依存で完全に単体テスト可能なため、opt-in の実験経路（本番から到達不能。`CudaMmaGemm::new_with_swizzle`
を明示的に呼ばない限り NVRTC コンパイルされない）としてコード実装まで行った。#497 は蛇行走査コード自体を
revert したが、#499 は opt-in 経路として温存する点が異なる。

## 3. 計測手順（DGX Spark GB10・sm_121 実機）

base（`CudaMmaGemm::new`。本番カーネル）と head（`CudaMmaGemm::new_with_swizzle`。動的選択幅 + 参考として
固定候補 `{8, 16}`）それぞれについて、5 回計測の中央値 TFLOPS を比較する（`bench-harness::protocol::run`
が中央値計測を実装済み。`coding-rust.md` 準拠。接続・転送手順は `docs/real-hardware-verification-env.md`
に従う。実ホスト名はローカル管理外ファイル参照）。

```sh
git fetch origin
gh pr checkout 667   # 本イシューの実装 PR（perf/499-tile-sm-swizzle。opt-in 経路のみ・本番カーネル無変更）
# PR 番号（#667）で明示する: ブランチ名固定だとマージ後にブランチが削除・更新停止して
# "git checkout <ブランチ名>" が実行不能になる（PR #667 codex-review P2 是正）。`gh pr
# checkout` は PR がマージ・クローズ後もリモートの pull/667/head 参照から取得できるため
# ブランチ削除後も実行可能（`git fetch origin refs/pull/667/head && git checkout
# FETCH_HEAD` でも同等）。PR マージ後は squash commit として main に取り込まれるため、
# 単に `git checkout main` でも同じコードを指す。

# 数値一致確認（TFLOPS 比較より前に必須。swizzle はブロック割り当ての置換のみで
# 各出力要素のアキュムレート順序を変えないため、bit 一致で主張できる前提を検証する）。
# cpu_cuda_mma_parity・parity_nonregression（feature 非依存）はこの通常経路で
# 実行される。
cargo test -p backend-cuda --release -- --ignored --nocapture

# mma_f16_swizzle_variant_matches_base_bit_exact_output は internal-diagnostics
# feature（既定 off。CudaMmaGemm::new_with_swizzle 自体が同 feature でゲート
# されているため）配下の #[cfg] のみでコンパイルされる（gemm_mma.rs 同テスト
# doc コメント参照）。上記コマンドには --features がないため対象テストが
# コンパイル・実行されず green と誤認する（PR #667 codex-review P1 是正）。
cargo test -p backend-cuda --lib --release --features internal-diagnostics -- --ignored --nocapture mma_f16_swizzle_variant_matches_base_bit_exact_output

# A/B 計測（動的選択幅の表示 + base/head の TFLOPS 比較。internal-diagnostics feature 必須）
cargo run -p backend-cuda --example gemm_mma_swizzle_bench --release --features internal-diagnostics
```

`cpu_cuda_mma_parity`・`parity_nonregression`（tolerance pin テスト含む）・
`mma_f16_swizzle_variant_matches_base_bit_exact_output`（本イシューで追加した swizzle 変種 vs base の bit
一致テスト）等が green であること（tolerance 定数〈`RELATIVE_TOLERANCE`・`ABSOLUTE_RESCUE_THRESHOLD`〉・
`parity_baseline.rs` は変更しない）。

レジスタスピル確認（TFLOPS 比較の前に必須。remap は `blockIdx.y`/`blockIdx.x`/`gridDim.x`/`gridDim.y` から
導出する追加の整数演算であり、コンパイラが定数畳み込みしきれない場合はレジスタ使用量が変わりうる。スピルが
起きると効果測定が「改善なし」ではなく「性能後退」として現れるため、両者を切り分ける）:

```sh
# NVRTC の -Xptxas -v 相当（レジスタ使用量ログ）で base/head 間の register 数・
# local memory 使用量に差がないことを確認してから TFLOPS を比較する
```

**2026-08-19 実測での注記**: 上記のレジスタスピル確認・bit 一致テスト（`cpu_cuda_mma_parity`・
`mma_f16_swizzle_variant_matches_base_bit_exact_output` 等）の実施証跡はイシュー #739 本文に記載が無い
ため、本ドキュメントでは実施済みと断定しない。「6. 実測結果」の TFLOPS 比較は確定済みだが、これら前提
手順の実施は本番結線を担う #740 側で必須事項として引き継ぐ。

## 4. 判断基準

- base に対し head（動的選択幅）の中央値 TFLOPS が size ∈ {2048, 4096} の両方で改善していれば「採用」とし、
  `kernels_mma.rs::MMA_F16`／`gemm_mma.rs::CudaMmaGemm::launch_f16`（本番経路）へ swizzle remap を結線する
  PR を起票し、本ドキュメントへ実測結果を追記する
- 改善が確認できなければ**採用しない**と判断し、その判断と実測値を本ドキュメントへ記録して本イシュー
  （#499）をクローズする（opt-in コード自体は削除せず、`select_swizzle_group_width`/`swizzled_block_idx`
  の設計記録として残すかは採否判断時にユーザーと相談する）
- **未計測の間は「採用済み」として扱わない**。本番カーネルへの結線は、上記いずれかの判断が実機計測を
  もって確定してから行う（暫定導入は行わない）
- **2026-08-19 実測後の判定（本イシュー #739）**: 下記「6. 実測結果」のとおり 4096 は明確改善
  （×1.5957〜1.5902）だが、上記判断基準が要求する `size ∈ {2048, 4096}` の**両方**の改善に対し
  2048 は中立（×0.97〜1.00）で未達であり、512/1024 も中立にとどまる。よって**現行の判断基準に
  照らすと「採用」の条件を満たさない**。本イシューの結論は現行基準どおり**不採用**であり、本番カーネル
  （`kernels_mma.rs::MMA_F16`／`gemm_mma.rs::CudaMmaGemm::launch_f16`）への結線は行わない
- **4096 限定適用の扱い（未承認）**: 4096 単独では大きな改善が確認されているため、判断基準を
  「4096 単独改善で採用」等へ緩和すれば別の結論になりうるが、これは確定済み性能採用基準（上記 2 項）の
  変更にあたり、`.claude/rules/coding-rust.md`／`.claude/rules/security.md` の「ガードレール閾値・
  性能採用基準の変更はユーザー承認必須」に従いユーザー承認が必要である。**本ドキュメント時点ではその
  承認は得られていない**ため、4096 限定の本番結線は行わず、必要であれば後続イシュー #740 でユーザー
  承認を得たうえで判断基準の改定（本docへの明記込み）と本番結線を行う。#740 は「採否の先送り」では
  なく「（承認が得られた場合の）4096 限定基準の新設可否の検討」に限定して引き継ぐ

## 5. §1.2 parity 非後退契約の機械確認

```sh
git diff origin/main -- crates/backend-cuda/tests/parity_nonregression.rs crates/backend-cuda/tests/common/parity_baseline
```

無差分であることを確認する（本ファイル §2 の「本番カーネル無変更」論拠の裏付け。tolerance 定数・ベース
ライン fixture を変更していないことをコミット前に検査する。`cuda-gemm-mma-block-tile.md` §6・
`cuda-gemm-serpentine-ab.md` 「数値一致確認」節と同手順）。

## 6. 実測結果（2026-08-19・GB10 実機・commit `cbc16e7`）

計測条件: GB10 実機・commit `cbc16e7`・5 回計測中央値・`gemm_mma_swizzle_bench --features
internal-diagnostics`（出典: イシュー #739）。

| size | base TFLOPS | head（動的選択幅）TFLOPS | 倍率 |
|------|-------------|----------------------------|------|
| 512 | — | — | ×0.97〜1.00（中立） |
| 1024 | — | — | ×0.97〜1.00（中立） |
| 2048 | — | — | ×0.97〜1.00（中立） |
| 4096 | 34.3279 | 54.7754 | ×1.5957 |

固定候補幅 `g16` の 4096 での結果は ×1.5902（動的選択幅 `g8` 相当の ×1.5957 とほぼ同水準）。512／1024／
2048 は base 比 ×0.97〜1.00 の範囲でほぼ中立（個別 TFLOPS 値の内訳はイシュー本文に記載が無いため倍率
のみ記録）。

**判断（§4 判断基準に照らして）**: 4096 は明確改善、512/1024/2048 は中立のため、判断基準の「size ∈
{2048, 4096} の両方で改善」を満たさない（2048 は中立で明確な改善ではない）。よって現行基準では**不採用
確定**であり本番結線は行わない。4096 帯限定の新基準採用を検討する場合はユーザー承認を要し、承認後の
判断改定・本番結線は上記「2. 状態」節のとおり #740 に引き継ぐ。

## 7. TF32 opt-staged 経路への横展開（#741）

イシュー #741「feat(backend-cuda): wmma_tf32_staged への threadblock swizzle 実装 + 実機 A/B」の
記録。#1〜#6 節は f16 `mma.sync` 経路（#499）の記録であり、本節は TF32 opt-staged 経路
（`kernels_wmma_opt.rs::gemm_wmma_tf32_staged`）への同型対策の横展開を扱う。

### 7.1 設計根拠

- **ncu 実測**（2026-08-19・GB10）: M=N=K=4096 時に TF32 opt-staged 経路の L2 hit rate が
  96.77%→76.51% へ崩壊している（`docs/perf/cuda-gemm-bottleneck-diagnosis.md`）。#499 の効果仮説
  （タイル→SM 割り当て順を変えることで L2 上の A/B タイル再利用率を上げる）が TF32 staged 経路にも
  適用できるかを検証する。
- **f16 経路の実測**（#499・#740）: 同型対策で M=N=K=4096 において約 1.60 倍（34.33→54.78 TFLOPS・
  グループ幅 8）の改善が確認されている（#740 記載値）。ただし f16 経路とブロックタイル形状
  （f16: 64×128、TF32 staged: 64×64）・カーネル構造（cp.async 多段パイプライン段数等）が異なるため、
  同等の改善幅を保証するものではなく、本節の A/B 計測で個別に確認する。
- remap 式自体は `swizzle.rs::swizzled_block_idx`・`kernels_mma.rs::mma_f16_source_with_swizzle` と
  単一の設計を共有する（`kernels_wmma_opt.rs::wmma_tf32_f32_staged_source_with_swizzle` ドキュメンテー
  ションコメント参照）。

### 7.2 状態: opt-in 実装のみ完了。未計測のまま本番カーネルへ導入しない

`kernels_wmma_opt.rs::wmma_tf32_f32_staged_source_with_swizzle`（変種ソース生成）・
`gemm.rs::CudaGemm::new_with_tf32_staged_swizzle`（変種カーネルのコンパイル・保持。`internal-diagnostics`
feature ゲート）を実装したが、本 §2 節と同一の判断（本実装セッションは NVRTC 非搭載のため実機 A/B
計測ができない）により、**本番カーネル（`kernels_wmma_opt::wmma_tf32_f32_staged_source()`）・本番
ディスパッチ経路（`ops.rs`／`gemm_auto.rs`／`run_wmma_tf32` の 3 段選択）は 1 バイトも変更していない**。

並行イシュー（#740: f16 swizzle の本番結線、#743: TF32 staged の SMEM バンクコンフリクト対策）との
コンフリクトを避けるため、本イシューは `ops.rs`／`gemm_auto.rs`／`swizzle.rs` を変更せず、
`kernels_wmma_opt.rs` への変更もブロック原点アンカー置換の変種生成関数の追記のみに限定した
（既存カーネル本体・定数は無変更）。

### 7.3 計測手順（DGX Spark GB10・sm_121 実機）

base（`CudaGemm::new`。本番 TF32 opt-staged カーネル）と head
（`CudaGemm::new_with_tf32_staged_swizzle`。動的選択幅 + 参考として固定候補 `{8, 16}`）それぞれについて、
中央値 TFLOPS を比較する（`bench-harness::protocol::run`。§3 と同じ計測コア）。

```sh
git fetch origin
gh pr checkout <本イシューの PR 番号>   # perf/741-wmma-tf32-staged-swizzle（opt-in 経路のみ・本番カーネル無変更）

# 数値一致確認（TFLOPS 比較より前に必須）。既存 parity 系（feature 非依存）はこの通常経路で実行される。
cargo test -p backend-cuda --release -- --ignored --nocapture

# wmma_tf32_staged_swizzle_variant_matches_base_bit_exact_output は internal-diagnostics
# feature（既定 off）配下の #[cfg] のみでコンパイルされる（gemm.rs 同テスト doc コメント参照）。
# 上記コマンドには --features がないため対象テストがコンパイル・実行されず green と誤認する
# （#499 §3 と同じ注意点）。
cargo test -p backend-cuda --lib --release --features internal-diagnostics -- --ignored --nocapture wmma_tf32_staged_swizzle_variant_matches_base_bit_exact_output

# NVRTC ログでのレジスタ/スピル差分確認（§3 と同じ理由。remap は追加の整数演算のため
# コンパイラが定数畳み込みしきれない場合はレジスタ使用量が変わりうる）

# A/B 計測（動的選択幅の表示 + base/head の TFLOPS 比較。internal-diagnostics feature 必須）
cargo run -p backend-cuda --example gemm_wmma_tf32_swizzle_bench --release --features internal-diagnostics
```

### 7.4 判断基準

- base に対し head（動的選択幅）の中央値 TFLOPS が size ∈ {2048, 4096} の両方で改善していれば「採用」とし、
  `kernels_wmma_opt.rs::wmma_tf32_f32_staged_source()`／`gemm.rs`（本番経路。`run_wmma_tf32` の 3 段選択）
  へ swizzle remap を結線する後続 PR を起票する。結線後は parity 非後退テスト全 pass に加え、
  `examples/cuda_floor_bench.rs` で REQ-8 CudaF32 50% 下限（`docs/performance-targets.md`）への余裕改善を
  確認する
- 改善が確認できなければ「採用しない」と判断し、実測値を本節へ記録した上で opt-in 実装を温存するか
  revert するかをユーザーと相談する（§4 と同型の判断）
- **未計測の間は「採用済み」として扱わない**

### 7.5 §2 本番ディスパッチ経路無変更の機械確認

```sh
git diff origin/main -- crates/backend-cuda/src/ops.rs crates/backend-cuda/src/gemm_auto.rs crates/backend-cuda/src/swizzle.rs
```

無差分であることを確認する（本節「状態」の裏付け。並行イシュー #740／#743 とのコンフリクト回避の
裏取りを兼ねる）。

### 7.6 実測結果

（未計測。実機到達不可のためこのセッションでは計測していない。実機セッションで本節へ追記する。
実機到達可能時は §7.3 の手順で bit 一致テスト → NVRTC ログ確認 → A/B 計測の順に実行し、結果を
ここへ記録する）
