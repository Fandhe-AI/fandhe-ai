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

## 2. 状態: 実測完了（2026-08-19・GB10 実機・commit `cbc16e7`）。本番結線は #740 が担う

**2026-08-19、DGX Spark GB10 実機で A/B 計測を実施し、下記「6. 実測結果」の TFLOPS 比較を確定させた**
（出典: イシュー #739・#740。5 回計測中央値・`gemm_mma_swizzle_bench --features internal-diagnostics`）。
ただし**本番結線（`kernels_mma.rs::MMA_F16`／`gemm_mma.rs::CudaMmaGemm::launch_f16` への swizzle remap
の結線）は本イシュー（#739）のスコープでは行わない**。「未計測の間は『採用済み』として扱わない」原則の
後継として、判断基準（§4）に照らした採否確定・本番結線・小サイズでの非劣化確認は後続イシュー #740 が
担う（下記「4. 判断基準」参照）。

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
- **2026-08-19 実測後の運用（本イシュー #739 の判断）**: 下記「6. 実測結果」のとおり 4096 は明確改善
  （×1.5957〜1.5902）・512/1024/2048 は中立（×0.97〜1.00）が確定した。全サイズ適用か 4096 帯限定
  適用かの確定・本番カーネルへの結線・小サイズでの非劣化確認（レジスタスピル確認・bit 一致テストの
  再実行込み）は、実機計測が完了した本イシューではなく**後続イシュー #740 が実施する**（本ドキュメント
  はここでは判断を確定させない。「未計測の間は採用済みとして扱わない」原則の後継として「結線は #740 で
  実施」を明示する）

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
{2048, 4096} の両方で改善」を厳密には満たさない（2048 は中立で明確な改善ではない）。全サイズ適用か
4096 帯限定適用かの最終確定は、上記「2. 状態」節のとおり #740 が実機での小サイズ非劣化確認込みで
担う。
