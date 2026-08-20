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

**2026-08-19、DGX Spark GB10 実機で A/B 計測を実施し、下記「6. 実測結果」の TFLOPS 比較を確定させた**
（出典: イシュー #739・#740。5 回計測中央値・`gemm_mma_swizzle_bench --features internal-diagnostics`、
commit `cbc16e7`）。下記「4. 判断基準」の既存基準（`size ∈ {2048, 4096}` の両方で改善）に照らすと 2048 が
未達（中立）のため**不採用が確定**し、**本番結線（`kernels_mma.rs::MMA_F16`／
`gemm_mma.rs::CudaMmaGemm::launch_f16` への swizzle remap の結線）は行わない**。「未計測の間は『採用済み』
として扱わない」原則のとおり、判断が確定していない状態で暫定的に結線することもしない。4096 単独では
大きな改善があるため、判断基準自体を変更（例: 4096 限定基準の新設）すれば別の結論になりうるが、これは
性能採用基準の変更にあたりユーザー承認が必須であり本ドキュメント時点では未承認である。

以下は本 PR（#758）時点までの設計・実装経緯（イシュー #740 での一時結線・差し戻しを含む）:

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

**意思決定の記録（2026-08-19 ユーザー判断）**: PR #758 の残指摘対応にあたり、§4 の採用基準（size ∈
{2048, 4096} 両方の中央値 TFLOPS 改善）を「4096 限定」等へ改訂する案は採らず、**現行の採用基準を維持した
まま swizzle を不採用のままマージする**ことをユーザーが確定した。基準改訂自体は不承認ではなく単に未検討・
未提案のまま見送られており、必要になれば別途あらためて人間承認を経て提案する。

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

**出典**: イシュー #740「feat(backend-cuda): mma_f16 threadblock swizzle の本番結線（4096 実測 1.60 倍）」
記載の GB10 実機 A/B 計測値（2026-08-19。DGX Spark GB10・sm_121・commit `cbc16e7`・5 回計測中央値・
`gemm_mma_swizzle_bench --features internal-diagnostics`）。本ドキュメント更新セッション（実装エージェント。
RTX 3060・NVRTC 非搭載環境）自身による再計測ではない（§2「本セッションによる再検証範囲」参照）。

| size (M=N=K) | base TFLOPS | swizzle (group_width=8) TFLOPS | 比 |
|---|---|---|---|
| 512 | — | — | ×0.97〜1.00（中立） |
| 1024 | — | — | ×0.97〜1.00（中立） |
| 2048 | — | — | ×0.97〜1.00（中立） |
| 4096 | 34.3279 | 54.7754 | ×1.5957 |

固定候補幅 `g16` の 4096 での結果は ×1.5902（動的選択幅 `g8` 相当の ×1.5957 とほぼ同水準）。512／1024／
2048 は base 比 ×0.97〜1.00 の範囲でほぼ中立（個別 TFLOPS 値の内訳はイシュー本文に記載が無いため倍率
のみ記録）。

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

**計測境界（イシュー #776 で是正・#7.7 参照）**: `measure_wmma_tf32` は
H2D/D2H 転送・出力バッファ確保を計測ループ外へ出し、GPU 実行（カーネル起動＋同期）
のみを計測する（`cuda_floor_bench.rs::measure_wmma_tf32`・`gemm_mma_swizzle_bench.rs::
measure_mma_f16` と同じ方針。base／head 双方に同一方針を適用するため
apples-to-apples の比較になる）。#776 以前は高水準 API `run_wmma_tf32`（毎 iteration
H2D×2＋`alloc_zeros`＋D2H を含む）を直接計測しており、M=N=K=4096 で計測が極端に
不安定だった（§7.7 参照）。#776 以前の記録（本節 §7.6 も含む「未計測」時点の手順）と
#776 以後の記録は計測境界が異なるため絶対値を比較しない。

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

**注意（PR #758 時点）**: 上記 3 ファイルのうち `swizzle.rs` は、本チェックの起点である `origin/main` に
イシュー #740 側の変更（本番結線の差し戻し・doc コメント是正・`select_swizzle_group_width_pins_example_
sm_count_28_to_g8` テスト追加。§2 参照）が未取り込みの間は非空差分として現れる。これは #741 自身が
`swizzle.rs` を変更したことを意味しない（#741 のコミット自体は本ファイルに触れていない）。#740（PR #758）が
main へマージされた後にこのチェックを実行すれば、`origin/main` 自体に同変更が含まれるため差分は再び空になる。

### 7.6 実測結果

（未計測。実機到達不可のためこのセッションでは計測していない。イシュー #776 の
2026-08-20 再計測・原因調査・是正の記録は §7.7 を参照。#776 是正後の実機再計測結果も
判明次第 §7.7 へ追記する。実機到達可能時は §7.3 の手順で bit 一致テスト →
NVRTC ログ確認 → A/B 計測の順に実行する）

### 7.7 4096 計測異常の原因調査・是正（イシュー #776）

イシュー #776 報告値（2026-08-20 DGX GB10 実機再計測。イシュー本文記載）によると、
`gemm_wmma_tf32_swizzle_bench` の M=N=K=4096 計測が run 間で最大約 16 倍
（base 生値 0.21〜3.08 TFLOPS）と極端に不安定だった。同イシュー本文は、同一カーネル
（`wmma_tf32_staged`）の launch-only 計測である `cuda_floor_bench` の 4096 計測が
約 9.0 TFLOPS で安定していたことも報告している。本節は本 PR（#776）セッションでの
原因調査・是正の記録である。§7.4 の採用判定は是正後の実機再計測が揃うまで「未計測」の
ままとする。

#### 7.7.1 一次仮説（コード調査）

是正前の `measure_wmma_tf32`（`gemm_wmma_tf32_swizzle_bench.rs`）は高水準 API
`run_wmma_tf32`（`gemm.rs:1107` 以降）を計測クロージャへ直接渡していた。この API は
ホスト側スライスを受け取り、**毎 iteration** H2D 転送×2（`clone_htod`）・出力バッファ
確保（`alloc_zeros`。M=N=4096 で 64MB）・カーネル起動・同期・D2H 転送を行う。M=4096 では
毎 iteration 約 192MB のデバイス確保・解放が計測区間に混入する。

一方、`cuda_floor_bench.rs::measure_wmma_tf32`（`upload_f32`/`alloc_output_f32` を
ループ外・`launch_wmma_tf32`〈launch+sync のみ〉を計測）・姉妹ベンチ
`gemm_mma_swizzle_bench.rs::measure_mma_f16`（同型の低水準 API 方式）はいずれも
毎 iteration のデバイス確保・解放を計測区間に含めない設計であり、上記イシュー報告の
とおり 4096 でも安定している。

**一次仮説**: GB10 の unified memory 構成（nvidia-smi の memory 表示が [N/A]）上で、
毎 iteration の大容量デバイス確保・解放によるアロケータ・ページマッピングの状態依存
揺らぎが計測値を支配していた。カーネル自体（`wmma_tf32_staged`）の性能特性ではなく、
本ベンチの計測境界設計（H2D/D2H・確保/解放を計測区間に含めていたこと）が原因と判断した。
本 PR セッションでは実機接続情報がなく上記仮説の実機検証（§7.7.3）は実施できていない。

#### 7.7.2 是正内容

`measure_wmma_tf32`（`gemm_wmma_tf32_swizzle_bench.rs`）を `cuda_floor_bench.rs`・
`gemm_mma_swizzle_bench.rs` と同じ低水準 API 方式へ統一した:

1. `run_wmma_tf32` を 1 回 probe 実行して可用性を確認する
2. `upload_f32`/`alloc_output_f32` を計測ループ外で 1 回だけ実行する
3. 計測クロージャは `launch_wmma_tf32`（GPU 実行＋synchronize のみ）に限定する

swizzle 変種の計測は維持される: `new_with_tf32_staged_swizzle` は `wmma_tf32_staged`
スロットを差し替え、`launch_wmma_tf32` は同スロットを最優先する 3 段選択のため、
低水準 API 経由でも head 変種カーネルが起動される（A/A 誤認は起きない。本ベンチの
形状は全て正方かつ `n % 4 == 0 && k % 4 == 0` を満たし staged 整列条件を満たす）。

**本 PR（#776）で変更したのは `crates/backend-cuda/examples/gemm_wmma_tf32_swizzle_bench.rs`
と本ドキュメントのみであり、本番カーネル・本番ディスパッチ経路
（`kernels_wmma_opt.rs`・`gemm.rs` の 3 段選択・`ops.rs`／`gemm_auto.rs`／`swizzle.rs`）は
1 バイトも変更していない**（§7.5 と同型の確認: `git diff origin/main --
crates/backend-cuda/src/` が空であることをローカルで確認済み）。

計測境界が変わるため、本節・§7.6 以前の記録（是正前の高水準 API 計測。実測値は
本節へ記録されていないため直接比較対象なし）と #776 是正後の記録は絶対値を比較しない
（§7.3 追記を参照）。

#### 7.7.3 実機再計測

**実機再計測待ち**: 本実装セッション（2026-08-21）は DGX Spark GB10 実機
（`docs/real-hardware-verification-env.local.md`）への接続情報がローカルに存在せず、
実機切り分け（§2.2 の 5 run 生値記録・`nvidia-smi dmon` 併走）・§7.4 の採用判定が
実施できなかった。本セッションで完了したのはハーネス是正とローカル検証のみである:
`cargo build -p backend-cuda --example gemm_wmma_tf32_swizzle_bench --release
--features internal-diagnostics` でのビルド成立確認、実行して CUDA 非搭載環境での
DriverUnavailable スキップ経路が正常動作することの確認、`cargo fmt --all -- --check`・
`cargo clippy --workspace --all-targets --all-features -- -D warnings`・
`cargo test --workspace` の全 pass。実機到達可能なセッションで以下を実施し、
本節へ追記する:

1. 是正後ハーネスで 512/1024/2048/4096 の 5 run を実行し生値を記録する
2. 4096 の 5 run 生値が中央値の ±10% 程度以内に収まることを確認する（収まらない場合は
   追加要因〈クロックスロットリング・メモリ断片化等〉を切り分けて追記する）
3. 512〜2048 の非後退確認（新境界の絶対値が旧値を下回らないこと・swizzle 比が旧計測と
   矛盾しないことの両面）
4. §7.4 の採用判定（2048・4096 両方の中央値 TFLOPS 改善の有無）を実施し結果を記録する
