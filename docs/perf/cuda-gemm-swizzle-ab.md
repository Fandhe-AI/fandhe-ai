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

## 2. 状態: サイズ条件付き適用ロジックを本番既定コンストラクタへ結線済み（イシュー #782・2026-08-21 GB10 実機ゲート通過）

**2026-08-21 更新（イシュー #782）**: GB10 実機（main `0d80a50`。結線前・`new` は当時まだ base 専用）で
A/B 再計測（4096 ×1.592・512〜2048 劣化 5% 以内）・bit 一致テスト（`mma_f16_swizzle_variant_matches_
base_bit_exact_output`）を実施し、この結果と §4.1 の採用基準を根拠にユーザー承認のもと
`gemm_mma.rs::CudaMmaGemm::new`（本番既定コンストラクタ）へサイズ条件付き適用機構を結線した。
下記「(b) 結線前必須の確認」4 項目は、結線後の PR HEAD（`52d71ae`）そのものに対する 2026-08-21 の
GB10 実機マージ前検証（下記 §6.3）で **bit 一致（`new` 自身）・parity 非後退・結線後の
`cuda_floor_bench` 実測（4096 = 53.9033 TFLOPS・≥50 TFLOPS 確認）の 3 項目を解消済み**。
**レジスタスピル確認のみ、現行コードにレジスタ使用量ログの観測経路が存在しないため実行不能であり
未解消として残る**（実行不能の理由と間接根拠は §6.3 を参照。観測経路の新設は別イシューで追跡する）。
`CudaMmaGemm::new_with_size_conditional_swizzle`（`internal-diagnostics` feature 限定の実機検証専用
入口）は `new` へ統合されたため廃止した。以下は結線判断に至るまでの経緯（#775 時点までの歴史的記録）
として保持する。

### 2.0 以下は #775 時点までの経緯（結線待ち時の記録。歴史的記録として保持）

**出典（承認記録の一次確認）**: イシュー #775 は GitHub アカウント `aLiz-Nancy`（本リポジトリのユーザー
アカウント）が起票した（`gh issue view 775 --json author,body` で 2026-08-21 実装セッションが実測確認）。
Issue 本文に記載の A/B 実測表（DGX Spark GB10・sm_121・5 回計測中央値・`gemm_mma_swizzle_bench
--features internal-diagnostics`。2026-08-20）は以下のとおり（下記「サイズ条件付き適用の実装」節の閾値
2048 タイルはこの実測表から導出）:

| size (M=N=K) | base TFLOPS | swizzle(動的幅g8) TFLOPS | 比 |
|---|---|---|---|
| 512 | 14.8866 | 14.5762 | 0.979 |
| 1024 | 33.5042 | 33.1729 | 0.990 |
| 2048 | 48.3496 | 47.9694 | 0.992 |
| 4096 | 34.4089 | 54.3055 | 1.578 |

**採用基準の改訂（2026-08-20 実測・イシュー #775 起票時点でユーザー自身が明示した受け入れ条件が承認記録）**:
下記「4. 判断基準」の旧基準（size ∈ {2048, 4096} 両方の中央値 TFLOPS 改善）は 2048 が字義上未達のままだが、
イシュー #775 の受け入れ条件（4096 級で適用・512〜2048 は劣化 5% 以内の非後退ガードレール）自体を、
**サイズ条件付き適用という改訂後の採用基準**として扱う（`.claude/rules/coding-rust.md`／`.claude/
rules/security.md` の「性能採用基準の変更はユーザー承認必須」に対する承認記録に相当）。#758 差し戻し理由
（下記「以下は #758 時点までの経緯」参照）3 点のうち、(a) 採用基準の無承認読み替えはこの改訂を明記する
ことで、(c) `select_swizzle_group_width` の CI 恒久検査が依拠する SM 数入力は
`gemm_mma.rs::CudaMmaGemm::new_with_size_conditional_swizzle`（下記参照）が
`device.multiprocessor_count()` の実測値を動的に使う（ハードコード値へ依存しない）ことで、それぞれ解消した
（SM 数実測値 48。2026-08-19 実測〈イシュー #739〉・2026-08-20 ベンチ起動診断で再確認〈イシュー #777〉）。

**(b) 結線前必須の確認は本 PR 時点でも未実施（PR レビュー是正・#758 と同型の不備の再発防止）**: レジスタ
スピル確認・`CudaMmaGemm::new` 自身での bit 一致実機再検証・parity 非後退実機再検証・`cuda_floor_bench`
実機再計測は、本 PR の実装セッションが NVRTC・CUDA 実機非搭載の環境で作業したため**未実施**（下記「§3
計測手順」「§6.1 結線後の検証」参照）。ホスト側検証（`cargo fmt`/`clippy`/`test --workspace
--all-features`・`swizzle.rs` の境界値ユニットテスト・`git diff origin/main -- tests/
parity_nonregression.rs tests/common/parity_baseline` の無差分確認）は実施済みで全て pass しているが、
これは GB10 実機での bit 一致・parity・レジスタスピル・性能の実機検証を代替しない。

当初の実装（コミット 1e8235f・3255823）は (b) が未実施のまま `CudaMmaGemm::new`（本番既定コンストラクタ）
への本番結線を完了させており、本節冒頭の「マージ判断を行うこと」という前提とコードの状態（既に結線済み）
が矛盾していた（PR レビュー指摘・High）。この矛盾を解消するため、**`CudaMmaGemm::new` への結線は行わず**、
サイズ条件付き適用ロジック（`should_apply_swizzle`・`launch_f16` のディスパッチ）自体は温存したまま、
実機検証専用の opt-in コンストラクタ `CudaMmaGemm::new_with_size_conditional_swizzle`
（`internal-diagnostics` feature 限定）からのみ到達できるよう是正した。GB10 実機到達可能なセッションで
上記 4 項目（§6.1）を実施・記録した後続 PR で、`CudaMmaGemm::new` の既定をこの経路へ切り替えること。

**サイズ条件付き適用の実装（opt-in）**: `crates/backend-cuda/src/swizzle.rs::should_apply_swizzle`（総
ブロックタイル数 `num_m_blocks * num_n_blocks >= 2048`。4096 実測点〈2048 タイル〉以上のみ適用し、2048
実測点〈512 タイル〉未満へは外挿しない保守的閾値。**この閾値は正方形形状〈M=N=K〉の実測点のみに基づき、
非正方形形状〈例: M=32768, N=512〉への外挿は未検証**——`swizzle.rs::SWIZZLE_APPLY_TILE_COUNT_THRESHOLD`
ドキュメンテーションコメント参照。PR レビュー指摘・Medium）が、`gemm_mma.rs::CudaMmaGemm::launch_f16` が
呼び出し形状ごとに base／swizzle 変種のいずれを起動するかを判定する。
`new_with_size_conditional_swizzle`（opt-in・実機検証専用入口）は `device.multiprocessor_count()` の実測に
成功すれば動的選択幅で swizzle 変種を追加コンパイルし（失敗時は安全側で base のみを保持）、`launch_f16`
がこの閾値でディスパッチする。個別呼び出しでの適用有無は `CudaMmaGemm::swizzle_applies(m, n)` で観測できる。
**本番既定コンストラクタ `CudaMmaGemm::new` はこの経路に結線しておらず常に base のみを返す**（上記
「(b) 結線前必須の確認」参照）。

### 2.1 以下は #758 時点までの経緯（差し戻し時の記録。歴史的記録として保持）

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
  （当時の記述。その後 2026-08-19 実測〈イシュー #739〉・2026-08-20 ベンチ起動診断の再確認〈イシュー #777〉
  で GB10 の SM 数は 48 と実測済み。本番経路は `device.multiprocessor_count()` 実測値を動的に使うため
  本件は解消済み〈上記 §2 冒頭「採用基準の改訂」段落 (c) 参照〉）

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

### 4.1 採用基準の改訂（2026-08-20・イシュー #775・確定）

上記の旧基準（size ∈ {2048, 4096} 両方の中央値 TFLOPS 改善）は維持しつつ、**サイズ条件付き適用**を新たな
採用経路として追加する。イシュー #775 のユーザー起票の受け入れ条件（4096 級で適用・512〜2048 は劣化 5%
以内の非後退ガードレール）自体を、この改訂の承認記録とする（§2 参照）。

- **判定基準（サイズ条件付き適用）**: base に対し head（動的選択幅）の中央値 TFLOPS が、実測改善を確認した
  総ブロックタイル数（`num_m_blocks * num_n_blocks`）以上の形状でのみ改善していれば、その閾値以上の形状
  限定で「採用」とし、`crates/backend-cuda/src/swizzle.rs::should_apply_swizzle` の閾値としてハードコード
  する。閾値未満の形状は、劣化が 5% 以内（非後退ガードレール）であることを条件に base のまま維持する
  （劣化が 5% を超える場合は閾値の見直しをユーザーと相談する）
- **2026-08-20 実測（イシュー #775）に基づく確定閾値**: 4096（総タイル数 2048）が ×1.578 の改善、2048
  （総タイル数 512）が ×0.992（劣化 0.8%。5% 以内）のため、閾値を総ブロックタイル数 **2048** に設定する
  （下記「6. 実測結果」参照）。512/1024/2048 はいずれも劣化 5% 以内のためガードレールを満たす

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

### 6.1 2026-08-20 実測（イシュー #775・サイズ条件付き結線の根拠。歴史的記録——#782 で `new` へ結線済み。下記「結線後の検証」の到達手順は §6.2 を参照すること）

**出典**: イシュー #775 Issue 本文記載の GB10 実機 A/B 計測（2026-08-20。DGX Spark GB10・sm_121・5 回計測
中央値・`gemm_mma_swizzle_bench --features internal-diagnostics`。§2 の表と同一データ）。

| size (M=N=K) | base TFLOPS | swizzle (動的幅 g8) TFLOPS | 比 |
|---|---|---|---|
| 512 | 14.8866 | 14.5762 | ×0.979 |
| 1024 | 33.5042 | 33.1729 | ×0.990 |
| 2048 | 48.3496 | 47.9694 | ×0.992 |
| 4096 | 34.4089 | 54.3055 | ×1.578 |

4096（総ブロックタイル数 2048）の改善が §6（2026-08-19 実測）と同水準（×1.578 vs ×1.5957）で安定再現し、
512〜2048（総ブロックタイル数 512 以下）はいずれも劣化 5% 以内（×0.979〜0.992＝劣化 0.8〜2.1%）の非後退
ガードレールを満たす。この実測を根拠に §4.1 の採用基準改訂（サイズ条件付き適用・閾値 = 総ブロックタイル数
2048）を確定し、`swizzle.rs::should_apply_swizzle`／`launch_f16` のサイズ条件付きディスパッチ実装を確定
した。ただし下記「結線後の検証」が未完了のため、**`gemm_mma.rs::CudaMmaGemm::new`（本番既定コンストラクタ）
への結線は行わず**、実機検証専用の opt-in 入口 `CudaMmaGemm::new_with_size_conditional_swizzle`
（`internal-diagnostics` feature 限定）からのみディスパッチへ到達できるようにしている（§2 参照。PR
レビュー是正: 当初の実装は本節が求める検証未完了のまま `new` への結線を完了させており自己矛盾していた）。

**結線後の検証（#775 時点で未実施と記録していた 4 項目。#782 での解消状況は §6.2 参照）**: 当時（本
節が書かれた #775 実装セッション）は NVRTC・CUDA 実機非搭載の環境で作業したため、下記コマンドは
いずれも実行できていなかった。ホスト側検証（`cargo fmt`/`clippy`/`test --workspace --all-features`・
`swizzle.rs` の境界値ユニットテスト・`parity_nonregression.rs`／`tests/common/parity_baseline` の
無差分確認）は実施し全て pass していたが、これは実機検証を代替しない。

- `cargo test -p backend-cuda --lib --features internal-diagnostics -- --ignored --nocapture
  mma_f16_swizzle_variant_matches_base_bit_exact_output`（当時: `CudaMmaGemm::
  new_with_size_conditional_swizzle`〈opt-in・実機検証専用入口〉自身の bit 一致——base 選択時〈閾値未満
  形状〉・swizzle 選択時〈閾値以上形状。m=n=4096, k=32〉の両方——を検証する版。#782 で同関数は `new`
  自身を検証する版へ更新済み。`gemm_mma.rs` 参照）
- `cargo test -p backend-cuda --test parity_nonregression -- --ignored`（結線後の parity 非後退確認）
- レジスタスピル確認（§3「レジスタスピル確認」節の手順。base／swizzle 変種間でレジスタ数・local memory
  使用量に有意差がないことを確認する）
- 非正方形形状（縦長・横長。M≠N）での A/B 計測（PR レビュー指摘・Medium。上記実測はいずれも正方形形状
  のみで、`should_apply_swizzle` の閾値はアスペクト比を考慮しないため非正方形形状への外挿は未検証。
  `swizzle.rs::SWIZZLE_APPLY_TILE_COUNT_THRESHOLD` ドキュメンテーションコメント参照）
- `cargo run -p backend-cuda --release --example cuda_floor_bench`（5 回中央値。実機検証・`new` 既定切替
  後に、起動時診断が swizzle 適用状態を正しく報告することを確認する）

### 6.2 2026-08-21 実測（イシュー #782・本番結線ゲート。main `0d80a50`）

**出典**: イシュー #782 Issue 本文記載の GB10 実機受け入れ再計測（2026-08-21・DGX Spark GB10・sm_121・
main `0d80a50` 時点。`gemm_mma_swizzle_bench`・外側 5 回中央値）。

| size (M=N=K) | base TFLOPS | swizzle (動的幅 g8) TFLOPS | 比 |
|---|---|---|---|
| 512 | — | — | ×0.982 |
| 1024 | — | — | ×0.989 |
| 2048 | — | — | ×0.992 |
| 4096 | 33.952 | 54.040 | ×1.592 |

4096（総ブロックタイル数 2048）の改善が §6.1（2026-08-20 実測・×1.578）と同水準で再現し、512〜2048 は
いずれも劣化 5% 以内（×0.982〜0.992）の非後退ガードレールを満たす。起動時診断は `num_sms=48
dynamic_group_width=8`（SM=48 → 動的選択幅 g8 の再現）を出力した。

**§6.1「結線後の検証」4 項目の解消状況（イシュー #782 本文記載の範囲のみ。過大記載しない）**:

- **bit 一致**: `mma_f16_swizzle_variant_matches_base_bit_exact_output`（`internal-diagnostics` feature
  限定・実機 `#[ignore]` テスト）ok — swizzle 変種と base の数値完全一致を直接検証済み **（解消）**
- **`cuda_floor_bench` 非後退確認**: 上記 A/B 計測時点（main `0d80a50`。**この時点ではまだ `new` は
  base 専用**）で全系列 ±1.2% 以内の run-to-run 安定性を確認した。これは結線前 base 経路の安定性の
  裏付けであり、結線後（本 PR マージ後）の `CudaMmaGemm::new` 経由 4096 が実際に swizzle 変種を選択し
  ≥50 TFLOPS 級へ改善することの確認ではない。**この確認はイシュー #782 の受け入れ条件チェックリストで
  「マージ後確認可」と明記された未完了項目であり、本 PR マージ後に別途実施する（未解消）**
- **parity 非後退**: **本ゲートでは実施していない**。イシュー #782 の受け入れ条件チェックリストで
  「マージ後確認可」の未チェック項目として明記されており、本 PR マージ後に `cargo test -p backend-cuda
  --test parity_nonregression -- --ignored` を実行して確認する（未解消。なお `parity_nonregression.rs`
  の `mma_f16` ベースライン形状は 256×256×4096＝総ブロックタイル数 4×2=8 で `should_apply_swizzle` の
  閾値〈2048〉未満のため base カーネルを選択する契約であり、結線の有無によらず数値は変化しない見込み
  だが、これは机上確認であり実機実行による確認の代替にはならない）
- **レジスタスピル確認**: **本ゲートでは別途再計測していない**。イシュー #782 の本文・上記実測記録に
  レジスタ・local memory 使用量の実測値は含まれておらず、§3「レジスタスピル確認」節の NVRTC
  `-Xptxas -v` 手順は本ゲートの一部としては実施されなかった（未解消のまま記録する）

上記のうち bit 一致の解消と、§4.1 の採用基準（サイズ条件付き適用・閾値 = 総ブロックタイル数 2048）の
2026-08-21 A/B 再現を根拠に、ユーザー承認のもと `gemm_mma.rs::CudaMmaGemm::new`（本番既定コンストラクタ）
へサイズ条件付き適用機構を結線した（イシュー #782）。**本節（§6.2）時点で未解消と記録した 3 項目の
うち parity 非後退・`cuda_floor_bench` 結線後実測はその後のマージ前検証（§6.3）で解消済み**。
`CudaMmaGemm::new_with_size_conditional_swizzle`（`internal-diagnostics` feature 限定の実機検証専用
入口）は `new` と重複するため廃止した。

**未検証のまま残る事項（§2.0「非正方形形状」節参照）**: 非正方形形状（縦長・横長）での A/B 計測は
本ゲートでも実施していない。今後の性能調査で必要になった場合に別途対応する。

### 6.3 2026-08-21 マージ前検証（イシュー #782・結線後 PR HEAD `52d71ae` 自身に対する実機確認）

**出典**: PR #784 の codex-review P1 指摘（「§6.1 の 4 項目を実施・記録してから結線する契約に反し、
マージ後確認へ先送りしている」）への対応として、**結線済みコードそのもの**（PR HEAD `52d71ae`）を
DGX Spark GB10 へ転送して実施したマージ前検証。各ベンチ直前の `nvidia-smi utilization.gpu` はすべて
0 %（常駐テナントの干渉なし）。

- **bit 一致（`new` 自身・解消）**: `mma_f16_swizzle_variant_matches_base_bit_exact_output`（検証対象を
  `new` 自身へ変更済み・m=n=4096 で `swizzle_applies` の assert により実際に swizzle 変種へディスパッチ
  されることを含めて検証）ok、pin テスト
  `mma_f16_new_wires_size_conditional_swizzle_into_production_constructor` ok（`internal-diagnostics`
  あり: 16 件中 11 passed / 5 failed、feature なし: 13 件中 8 passed / 5 failed。failed はいずれも
  #186×2・basic 版 provenance fail-closed×1・JIT キャッシュ `/tmp` pin×2 の既知 5 件のみで新規 fail なし）
- **parity 非後退（解消）**: `cargo test -p backend-cuda --release --test parity_nonregression --
  --ignored` → `parity_baselines_do_not_regress` ok（1 passed / 0 failed）。ベースライン形状
  （256×256×4096 = 総ブロックタイル数 8）は閾値 2048 未満で base 経路を選ぶ契約のため、swizzle 分岐
  自体の数値網羅は上記 bit 一致テスト（m=n=4096）が担う（役割分担）
- **結線後 `cuda_floor_bench` 実測（解消）**: feature なしビルド・外側 5 回。起動時診断は「swizzle
  variant wired into the production constructor (CudaMmaGemm::new; issue #782)・`swizzle_group_width()
  = Some(8)`・`swizzle_applies` = 512/1024/2048 false・4096 true」を出力。mma_f16 中央値（TFLOPS）:
  512 = 15.2105、1024 = 33.2053、2048 = 47.9969、**4096 = 53.9033**（結線前 base 33.92 比 ×1.589・
  ≥50 TFLOPS 確認）。512〜2048 は結線前値比 +1.5%／+0.2%／−0.2% でいずれも劣化 5% 以内
- **レジスタスピル確認（実行不能・未解消）**: §3 の NVRTC `-Xptxas -v` 相当手順は現行コードでは到達
  不能であることを実機セッションで確認した。理由: (1) `nvrtc.rs::compile_ptx` はレジスタ使用量ログ
  出力オプションを渡しておらず、PTX→SASS のレジスタ割り当ては driver JIT 時に行われるためログが
  どこにも出力されない、(2) `kernels_mma.rs` のカーネルソース取得口が非公開でオフライン `ptxas -v`
  にも掛けられない、(3) JIT ディスクキャッシュにも成果物が残存せず `cuobjdump` 事後解析も不成立。
  間接根拠として、結線後 4096 実測（53.9033）が opt-in A/B の swizzle 側実測（54.04〜54.31）と同水準
  であり、病的なスピルが発生していればこの一致は説明できない。観測経路（ソース取得口・レジスタ使用量
  ログ要求オプション等）の新設と直接確認は別イシューで追跡する

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

**追記（イシュー #775 時点）**: イシュー #775 が `swizzle.rs`（`should_apply_swizzle` 追加等）を変更した
ため、#775 マージ後は本チェックが再び非空差分を示す。これも上記と同型の一時的な非対称であり、#741 自身が
`swizzle.rs` を変更したことを意味しない。

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

**一次仮説（未検証）**: GB10 の unified memory 構成（nvidia-smi の memory 表示が [N/A]）上で、
毎 iteration の大容量デバイス確保・解放によるアロケータ・ページマッピングの状態依存
揺らぎが計測値を支配していた、というのが有力な原因仮説である。カーネル自体
（`wmma_tf32_staged`）の性能特性ではなく、本ベンチの計測境界設計（H2D/D2H・確保/解放を
計測区間に含めていたこと）を是正対象と位置づける。ただし launch-only の姉妹ベンチが
安定していたという事実だけでは、転送・確保・解放のどの要素が揺らぎを支配していたか、
またクロック・温度等の交絡要因の寄与を切り分けられていない。本 PR セッションでは実機
接続情報がなく上記仮説の実機検証（§7.7.3）は実施できていないため、**確定原因としては
扱わず**、§7.7.3 の実機再計測での比較結果が得られた時点で確定原因へ更新する。

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
3. 512〜2048 の非後退確認。§7.7.2 のとおり計測境界（launch-only／H2D・D2H・確保込み）が
   異なるため旧値（是正前ハーネスによる記録）との絶対値比較はしない（§7.3 追記・本節冒頭
   と同型。比較不能な値同士の判定を避ける）。非後退条件は**是正後ハーネス内で完結する
   同一境界の比較のみ**とする: 各 size で head（動的選択幅）の中央値 TFLOPS が base を
   下回らないこと（base/head 比。両者とも同一セッション・同一 launch-only 境界で計測）
4. §7.4 の採用判定（2048・4096 両方の中央値 TFLOPS 改善の有無）を実施し結果を記録する
