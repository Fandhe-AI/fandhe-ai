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

## 2. 状態: 未採用（差し戻し済み。PR #758 レビュー是正）

イシュー #740 で一度 `crates/backend-cuda/src/gemm_mma.rs::CudaMmaGemm::new`（本番既定コンストラクタ）へ
swizzle を本番結線したが、PR #758 の codex-review／Cursor Bugbot 指摘により差し戻した。理由:

- **§4 の採用基準を字義どおり満たしていない**: §4 は size ∈ {2048, 4096} の**両方**で中央値 TFLOPS 改善を
  要求するが、実測（§6）は 4096 のみ大幅改善（×1.5957）で 2048 は ×0.97〜1.00（未改善・中立）。「4096 大幅
  改善＋512〜2048 中立」を代替の採用基準として読み替えたが、これは人間承認を経ていない基準改訂に相当する
- **結線前必須の確認が未実施**: レジスタスピル確認（§3）・本番コンストラクタ自身での bit 一致再検証・
  parity 非後退再検証（§5）が結線当時に完了していなかった
- **CI 恒久検査の入力値の誤り**: `select_swizzle_group_width` の CI 恒久検査
  （`crates/backend-cuda/src/swizzle.rs`）が使う SM 数 `28` を「GB10 実測 SM 数」として扱っていたが、
  `docs/perf/sm121-device-attributes.md` L58 の `28` は RTX 3060（sm_121 ではない）の例示ダンプであり、
  同ドキュメント L79 のとおり GB10（sm_121）自体の SM 数は本リポでは未実測（Cursor Bugbot 指摘・PR #758）

`CudaMmaGemm::new` は base カーネル（`kernels_mma::mma_f16_source()`。swizzle 無適用）へ差し戻し済み。
swizzle 適用版は引き続き `CudaMmaGemm::new_with_swizzle`（`internal-diagnostics` feature 限定）から opt-in
で利用できる。§4 の採用基準を字義どおり満たすか、満たさない場合は人間承認を伴って採用基準を正式改訂した
うえで、上記 3 点（採用基準・事前確認・SM 数入力値）をすべて解消してから再結線を検討する。

`ops.rs`／`gemm_auto.rs` は mma_f16 経路自体を参照しないため（`CudaGemmAuto::run_f16` の MatrixUnit 分岐は
WMMA のみ）無変更のままであり、結線点は `CudaMmaGemm::new` に閉じる（イシュー #740 実装計画 §2「結線点の
特定」）。実際に選択された `group_width`（`new_with_swizzle` 経由時）は `CudaMmaGemm::swizzle_group_width()`
アクセサ（feature 非依存）で可観測（`examples/cuda_floor_bench.rs` の起動時診断が出力する）。

**本 PR（#758）時点の再検証範囲**: 本 PR は上記差し戻しのみを行い、§6 の実測値の新規再計測は行っていない
（実行環境が NVRTC 非搭載のため）。§6 は「未採用」判断に至った過去の実測記録として保持する。§3「計測手順」・
実機 `--ignored` テスト（`mma_f16_swizzle_variant_matches_base_bit_exact_output`・`cargo test -p
backend-cuda --lib --features internal-diagnostics -- --ignored`）・`cuda_floor_bench` の実機再計測は、
実行環境が GB10 実機に接続できないため未実施。次回 GB10 実機セッションで §4 の採用基準（2048/4096 両方の
改善）を満たすか再計測し、満たす場合のみ再結線・本ドキュメント更新を行うことを推奨する。

## 3. 計測手順（DGX Spark GB10・sm_121 実機。#499 時点の手順・イシュー #740 の採否判断根拠となった A/B）

**注意（イシュー #740 で一時反転した役割は PR #758 レビュー指摘により差し戻し済み）**: 下記手順は #499
当時（swizzle が opt-in だった時点）の記録であり、`new` を base（swizzle 無適用）として記述している。
#740 で `new` 自身が一時 swizzle 既定へ切り替わった際は、この役割の base を `CudaMmaGemm::
new_without_swizzle`（`internal-diagnostics` feature 限定・診断用入口）が代替で担っていたが、PR #758
レビュー指摘によりその結線自体を差し戻したため、現在は `new`（feature 非依存・本番既定コンストラクタ）と
`new_without_swizzle`（`internal-diagnostics` feature 限定）の**双方が base カーネルを返す**（§2 参照。
`new_without_swizzle` は `new` の差し戻し後は冗長だが、`new_with_swizzle` と対称の明示的な base 入口として
`gemm_mma.rs::CudaMmaGemm::new_without_swizzle` ドキュメンテーションコメントのとおり維持している）。本節は
#499 当時の歴史的記録として当時の手順文言のまま残す。

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

**旧判断（差し戻し済み。§2 参照）**: イシュー #740 時点では、4096 の大幅改善（4096 は §4 が要求する 2
サイズのうち 1 つ）と 512〜2048 の中立（劣化許容 5% 以内。実装計画 §3.1「コンティンジェンシー」の閾値）を
根拠に「採用」と判断し、サイズ条件分岐なしの全サイズ適用（実装計画 §3.1）で `CudaMmaGemm::new` へ結線した。
しかし 2048 単独の改善幅は本節に未転記のままであり、§4 の字義どおりの「2048/4096 両方の改善」を満たして
いないにもかかわらず「4096 大幅改善 + 512〜2048 中立」を採用条件として代替した判断は人間承認を経ていな
かった。PR #758 レビュー指摘（codex-review P1）によりこの判断は差し戻し、`CudaMmaGemm::new` は base
カーネルへ戻した（§2 参照）。§4 の採用基準を字義どおり満たす実測（2048 単独の中央値 TFLOPS を含む）が
得られ、かつ結線前必須の確認（レジスタスピル・bit 一致・parity）が完了してから再結線を判断する。

**結線後の再計測**: 未実施（§2 参照）。次回 GB10 実機セッションで下記を実行し、2048 単独の改善有無を含めて
本節を追記・更新する。再結線は §4 の採用基準を満たす場合、または人間承認を伴う採用基準の正式改訂を経た
場合に限る。

- `cargo test -p backend-cuda --lib --features internal-diagnostics -- --ignored --nocapture
  mma_f16_swizzle_variant_matches_base_bit_exact_output`（`CudaMmaGemm::new`〈本番既定〉自身の bit 一致を
  追加検証する版。`gemm_mma.rs` 参照）
- `cargo test -p backend-cuda --test parity_nonregression -- --ignored`（結線後の parity 非後退確認）
- `cargo run -p backend-cuda --release --example cuda_floor_bench`（5 回中央値。起動時診断に
  `mma_f16 kernel: threadblock swizzle group_width=...` が出力されることを確認する）
