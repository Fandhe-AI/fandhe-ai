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

## 2. 状態: 採用・本番結線済み（イシュー #740）

GB10 実機 A/B 計測（2026-08-19。イシュー #740 記載の実測値。§6 参照）で 4096: 34.3279 → 54.7754 TFLOPS
（×1.5957・group_width=8）を確認し、512〜2048 は 0.97〜1.00 倍とほぼ中立（劣化許容 5% 以内）だったため、
`crates/backend-cuda/src/gemm_mma.rs::CudaMmaGemm::new`（本番既定コンストラクタ）へ swizzle を**本番結線
済み**とした。`new` は `device.multiprocessor_count()` から `swizzle::select_swizzle_group_width` で
グルーピング幅を動的選択し、`kernels_mma::mma_f16_source_with_swizzle(group_width)` を常にコンパイルする
（サイズ条件分岐なしの全サイズ適用。§4 の判断基準・実装計画 §3.1 参照）。

`ops.rs`／`gemm_auto.rs` は mma_f16 経路自体を参照しないため（`CudaGemmAuto::run_f16` の MatrixUnit 分岐は
WMMA のみ）無変更のままであり、結線点は `CudaMmaGemm::new` に閉じる（イシュー #740 実装計画 §2「結線点の
特定」）。swizzle 無適用の base カーネルは診断専用の `CudaMmaGemm::new_without_swizzle`
（`internal-diagnostics` feature 限定）へ役割を移した（旧 #499 セッション時点では逆に swizzle 側がこの
位置で opt-in ゲートされていた）。実際に選択された `group_width` は `CudaMmaGemm::swizzle_group_width()`
アクセサ（feature 非依存）で可観測（`examples/cuda_floor_bench.rs` の起動時診断が出力する）。

**本セッション（実装エージェント。RTX 3060・NVRTC 非搭載環境）による再検証範囲**: 上記の実測値は本セッション
が新規に計測したものではなく、イシュー #740 に記載された実測値をそのまま記録・結線根拠として採用した
（`docs/perf/sm121-device-attributes.md` の GB10 実測 SM 数 28 に対する `select_swizzle_group_width(28, 64,
128) == 8` の CI 恒久検査は本セッションで追加し green を確認済み。§5 の機械確認・非実機テスト・
`--all-features` コンパイルも本セッションで実施し green）。§3「計測手順」・実機 `--ignored` テスト
（`mma_f16_swizzle_variant_matches_base_bit_exact_output`・`cargo test -p backend-cuda --lib --features
internal-diagnostics -- --ignored`）・`cuda_floor_bench` の実機再計測は、本セッションの実行環境が GB10 実機
に接続できないため未実施（fail-closed。§4「未計測の間は『採用済み』として扱わない」の裏返しとして、結線後の
再計測未実施を主張しない）。次回 GB10 実機セッションでこれらを実行し、本ドキュメントへ追記することを推奨する。

## 3. 計測手順（DGX Spark GB10・sm_121 実機。#499 時点の手順・イシュー #740 の採否判断根拠となった A/B）

**注意（イシュー #740 で `CudaMmaGemm::new`/`new_with_swizzle` の役割が反転）**: 下記手順は #499 当時
（swizzle が opt-in だった時点）の記録であり、`new` を base（swizzle 無適用）として記述している。#740 で
`new` 自身が swizzle 既定になったため、現在この役割の base は `CudaMmaGemm::new_without_swizzle`
（`internal-diagnostics` feature 限定）が担う（§2 参照）。本節は歴史的記録として当時の手順文言のまま残す。

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

## 4. 判断基準

- base に対し head（動的選択幅）の中央値 TFLOPS が size ∈ {2048, 4096} の両方で改善していれば「採用」とし、
  `kernels_mma.rs::MMA_F16`／`gemm_mma.rs::CudaMmaGemm::launch_f16`（本番経路）へ swizzle remap を結線する
  PR を起票し、本ドキュメントへ実測結果を追記する
- 改善が確認できなければ**採用しない**と判断し、その判断と実測値を本ドキュメントへ記録して本イシュー
  （#499）をクローズする（opt-in コード自体は削除せず、`select_swizzle_group_width`/`swizzled_block_idx`
  の設計記録として残すかは採否判断時にユーザーと相談する）
- **未計測の間は「採用済み」として扱わない**。本番カーネルへの結線は、上記いずれかの判断が実機計測を
  もって確定してから行う（暫定導入は行わない）

## 5. §1.2 parity 非後退契約の機械確認

```sh
git diff origin/main -- crates/backend-cuda/tests/parity_nonregression.rs crates/backend-cuda/tests/common/parity_baseline
```

無差分であることを確認する（本ファイル §2 の「本番カーネル無変更」論拠の裏付け。tolerance 定数・ベース
ライン fixture を変更していないことをコミット前に検査する。`cuda-gemm-mma-block-tile.md` §6・
`cuda-gemm-serpentine-ab.md` 「数値一致確認」節と同手順）。

## 6. 実測結果

**出典**: イシュー #740「feat(backend-cuda): mma_f16 threadblock swizzle の本番結線（4096 実測 1.60 倍）」
記載の GB10 実機 A/B 計測値（2026-08-19。DGX Spark GB10・sm_121）。本ドキュメント更新セッション
（実装エージェント。RTX 3060・NVRTC 非搭載環境）自身による再計測ではない（§2「本セッションによる再検証範囲」
参照）。

| size (M=N=K) | base TFLOPS | swizzle (group_width=8) TFLOPS | 比 |
|---|---|---|---|
| 4096 | 34.3279 | 54.7754 | ×1.5957 |
| 512〜2048 | — | — | ×0.97〜1.00（ほぼ中立。個別サイズ内訳は未転記） |

**判断**: §4 の判断基準（改善が確認できれば採用）に対し、4096 の大幅改善（4096 は §4 が要求する 2 サイズの
うち 1 つ）と 512〜2048 の中立（劣化許容 5% 以内。実装計画 §3.1「コンティンジェンシー」の閾値）を根拠に
「採用」と判断し、サイズ条件分岐なしの全サイズ適用（実装計画 §3.1）で `CudaMmaGemm::new` へ結線した
（§2 参照）。2048 単独の改善幅は本節に未転記のため、§4 の字義どおりの「2048/4096 両方の改善」ではなく
「4096 大幅改善 + 512〜2048 中立」を採用条件として扱った判断根拠は実装計画 §3.3 に記録する。

**結線後の再計測**: 未実施（§2 参照）。次回 GB10 実機セッションで下記を実行し、本節を追記・更新する。

- `cargo test -p backend-cuda --lib --features internal-diagnostics -- --ignored --nocapture
  mma_f16_swizzle_variant_matches_base_bit_exact_output`（`CudaMmaGemm::new`〈本番既定〉自身の bit 一致を
  追加検証する版。`gemm_mma.rs` 参照）
- `cargo test -p backend-cuda --test parity_nonregression -- --ignored`（結線後の parity 非後退確認）
- `cargo run -p backend-cuda --release --example cuda_floor_bench`（5 回中央値。起動時診断に
  `mma_f16 kernel: threadblock swizzle group_width=...` が出力されることを確認する）
