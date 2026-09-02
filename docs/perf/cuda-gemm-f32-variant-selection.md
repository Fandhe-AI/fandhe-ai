# CUDA f32 GEMM の形状別カーネル選択（simple / double-buffer / split-K。#1035）

イシュー #1035「perf(backend-cuda): 形状別カーネル選択を simple / double-buffer / split-K のヒューリスティックへ拡張する」の実機比較記録テンプレート。
親ツリー #1029（GEMM カーネルの candle 超え）・#1007 の Phase 2。cuBLAS フル FP32（candle CUDA）を上回ることが目標で、本イシューは特に小サイズ（N=256〜512。candle 実測 423〜1,109 GFLOP/s @GB10、`docs/perf/cuda-gemm-kernel-vs-frameworks-baseline.md` §3.2）で SM が遊ぶ形状への対処を担う。

## 状態（追補: イシュー #1100。下記「1b」節）: SplitK の複合判定 FAIL と DoubleBuffer の性能逆転を修正した。SplitK は選択ヒューリスティックから撤退（K 支配的形状は Simple へ）し、DoubleBuffer は base（`TILED_F32`）と同一のバッファ管理（プール経由 `alloc_uninit_f32`）へ揃えた。**本 PR の実装セッションは CUDA 実機（GB10）接続を持たないため、DoubleBuffer 是正後の GB10 再実測は未実施**（下記「1b」節参照。#994 の先例に従い安全側〈選択条件は変更しない〉のまま「未実測」を明記する）

## 状態（#1035 当時の記録。下記そのまま維持）: DGX Spark GB10 実機実測完了（#1031 実機実測セッション。下記「1a. GB10 実機実測結果」節）。受け入れ条件 (a)「全 N で candle 以上」は**未達**、かつ `#[ignore]` 正当性テストが SplitK 経路で複合判定 FAIL を検出した。本ファイル冒頭の実装セッション記録（下記）は当時の状態としてそのまま残す

本実装セッションは実機接続情報（`docs/real-hardware-verification-env.local.md`）を持たないため、本イシューの受け入れ条件が要求する「(a) 全 N で candle 以上」の検証は実行できない（`docs/perf/cuda-gemm-cost-model-selection.md`・#527 が同じ理由で「未実測・要実機実行」のまま安全側クローズしている先例と同型）。受け入れ条件 (b)「選択ロジックのユニットテスト」は本ランで充足済み（§2）。

本実装セッションで検証済みの事項:

- `cargo build --workspace`
- `cargo fmt --all -- --check` / `cargo clippy --workspace --all-targets --all-features -- -D warnings`
- `cargo test -p fandhe-ai-backend-cuda`（`gemm_variant::tests`・`kernels_gemm_variants::tests` の GPU 不要ユニットテスト一式。§2 参照）
- `cargo test -p fandhe-ai-backend-cuda --features internal-diagnostics`（`gemm_f32_variants.rs` の環境適応スモーク 2 件。`#[ignore]` 2 件は未実行）
- `cargo test --workspace` / `cargo test --workspace --all-features`（回帰確認。全 green）
- `git diff origin/main -- crates/backend-cuda/tests/parity_nonregression.rs crates/backend-cuda/tests/common/parity_baseline crates/backend-cuda/src/kernels.rs crates/backend-cuda/src/kernels_mma.rs crates/backend-cuda/src/kernels_wmma.rs crates/backend-cuda/src/kernels_wmma_opt.rs` が無差分（既存カーネル・fixture・tolerance を一切変更していないことの機械確認）

未検証・実機実行待ちの事項:

- 下記 §1 の実機 A/B（N=256〜4096・K 支配的非正方形状 vs candle）
- `#[ignore]` テスト（`tests/gemm_f32_variants.rs`）による複合判定・決定性・境界形状の実機検証
- 暫定閾値（`SPLITK_MIN_K`・`SPLITK_MAX_SPLITS`・`SPLITK_PARTIAL_MAX_BYTES`・`DOUBLE_BUFFER_MIN_K`）の実測補正
- 本番既定経路（`gemm_auto.rs::CudaGemmAuto::run_f32`・`CudaGemm::new`・`run_tiled_f32`）への結線判断（ユーザー承認必須）

## 0. 安全側判断（opt-in 診断経路に留める理由）

- 本ランは NVRTC 実行不能環境（CUDA toolkit 非搭載）のため、DoubleBuffer／SplitK カーネルの数値検証・性能 A/B がすべて実機待ちになる。よって #1034（PR #740→#758 差し戻しの教訓）と同じ判断で、**未検証カーネルを本番既定経路へ結線しない**。選択ヒューリスティック（`gemm_variant.rs`）とカーネル（`kernels_gemm_variants.rs`）・実行経路（`gemm_variant_selection.rs`）はすべて `internal-diagnostics` feature（既定 off）限定の opt-in とし、本番既定コンストラクタ（`CudaGemm::new`）・`run_tiled_f32`・`kernels::TILED_F32` は一切変更していない
- 選択ヒューリスティックの閾値定数（`SPLITK_MIN_K`=1024・`SPLITK_MIN_K_PER_SPLIT`=32・`SPLITK_MAX_SPLITS`=32・`SPLITK_PARTIAL_MAX_BYTES`=256 MiB・`DOUBLE_BUFFER_MIN_K`=64）は実機実測前の**暫定値**であり、`cuda-gemm-cost-model-selection.md`（#527）と同じ方針で補正は 1 回限りとし、実測を追わない補正ループは行わない

## 1a. GB10 実機実測結果（#1031 実機実測セッション）

実機: DGX Spark GB10（compute capability (12, 1) = sm_121）・driver 580.173.02・
CUDA 13.0.88（`nvcc --version` 実測）・rustc 1.97.0。計測時 `nvidia-smi
--query-gpu=utilization.gpu --format=csv,noheader` で 0% を確認済み。commit
`10011cd4f8ef097351c0dc1244eb55c8a021040b`。

### 正当性検証（`#[ignore]` テスト）: SplitK 経路で複合判定 FAIL を検出

```sh
cargo test -p fandhe-ai-backend-cuda --release --features internal-diagnostics -- --ignored --nocapture
```

`gemm_f32_variants.rs` の `#[ignore]` テスト 2 件のうち:

- `split_k_execution_is_bit_deterministic_across_repeated_runs`: **PASS**
- `run_f32_matches_cpu_reference_across_variant_shapes`: **FAIL**（形状を順に検証する
  アサーションループの 3 件目で panic し以降の形状は未検証。4096³・2048³
  〈DoubleBuffer 想定〉は通過済みだったが、`m=128 n=128 k=8192`〈`ExpectedCategory::
  KDominant`。SplitK { num_splits: 8 } が選択される形状〉で以下の複合判定 FAIL を
  検出）:
  ```
  gemm_f32_variants m=128 n=128 k=8192 variant=SplitK { num_splits: 8 }:
  複合判定 FAIL（fail_count=8/16384, max_abs_diff=3.662e-4, max_rel_err=1.090e-2,
  mean_abs_diff=3.574e-5, mean_rel_err=6.131e-6, p50_abs_diff=2.575e-5,
  p99_abs_diff=1.688e-4, p999_abs_diff=2.594e-4）
  ```
  16384 要素中 8 要素が複合判定（相対誤差 1e-3 未満 または 絶対誤差 1e-5 未満）を
  満たさない。CPU 参照実装との数値不一致であり、fresh モードの計測揺らぎ等の
  非決定性ではない（`split_k_execution_is_bit_deterministic_across_repeated_runs`
  が PASS しているため SplitK 自体は決定的だが、その決定的な出力が CPU 参照と
  一致しない）。`m=n=128,k=16384`〈SplitK 対象のもう一方の想定形状〉・非整列形状
  （1000³・33×65×97・1×1×1）はテストが早期 panic したため**未検証**。

**本ファイルは実装 Agent の権限外（コード変更・許容誤差変更はユーザー承認必須。
`.claude/rules/security.md`「自己修復ループ固有のガードレール」・`coding-rust.md`
「バックエンド間数値一致テストの許容誤差を単独で緩和しない」）のため、この
SplitK 数値不一致に対する原因調査・修正は行っていない**。後続 Issue での追跡が
必要（`.claude/rules/out-of-scope-tracking.md` 準拠。本レポートで報告し
ユーザー判断を仰ぐ）。

### 性能計測（正当性が未確定な SplitK 経路を含む点に留意。参考データとして記録）

```sh
cargo run -p fandhe-ai-backend-cuda --example gemm_f32_variant_bench --release --features internal-diagnostics
```

`num_sms=Some(48) double_buffer_available=true split_k_partial_error=None split_k_reduce_error=None`

| 形状（M, N, K） | 選択された変種 | base_gflops（`TILED_F32`。GFLOP/s） | selected_gflops（GFLOP/s） | candle 実測（GFLOP/s） | candle 比 |
|-----------------|----------------|------------------------------|-----------------|------------------------|-----------|
| 256, 256, 256 | DoubleBuffer | 544.7 | 514.9（ratio 0.9452） | 423.0（`cuda-gemm-kernel-vs-frameworks-baseline.md` §3.2 N=256） | 約 1.22 倍 |
| 512, 512, 512 | DoubleBuffer | 1933.7 | 1303.4（ratio 0.6740） | 1109.3（同 N=512） | 約 1.18 倍 |
| 1024, 1024, 1024 | DoubleBuffer | 3903.5 | 1890.8（ratio 0.4844） | 2269.1（同 N=1024） | 約 0.83 倍（**暫定未達**・境界差あり） |
| 4096, 4096, 4096 | DoubleBuffer | 2826.1 | 226.6（ratio 0.0802） | 2265.1（同 N=4096） | 約 0.10 倍（**暫定未達**・境界差あり） |
| 128, 128, 8192 | SplitK { num_splits: 8 } | 388.2 | 873.8（ratio 2.2509） | 該当値なし（`cuda-gemm-kernel-vs-frameworks-baseline.md` は K 支配的非正方形状の candle 値を含まない。推定しない） | 判定不能（正当性も上記のとおり未確定） |
| 256, 256, 16384 | DoubleBuffer | 1119.6 | 1012.7（ratio 0.9045） | 該当値なし（同上） | 判定不能 |

（`base_gflops`／`selected_gflops` は例出力の TFLOPS を ×1000 して GFLOP/s へ換算。
ratio は例出力の `ratio` フィールドをそのまま転記）

**計測境界差（candle 比の解釈に必須）**: `selected_gflops`（`CudaGemmF32VariantSelection::
run_f32`。`gemm_f32_variant_bench.rs` 冒頭コメント「計測境界」節が明記するとおり
H2D→カーネル起動→D2H を一括計測する高水準 API）は、測定クロージャへ毎回ホスト側の
`Vec<f32>`（`a`/`b`）を渡すため**イテレーションごとに A・B の H2D 転送を含む**。一方
candle 側の比較値（`bench-candle/src/main.rs::run_gemm`）は `Tensor::from_vec` による
A・B のデバイスへの転送を計測ループの**外**（warmup 前に 1 回）で行い、計測窓は
`matmul` 呼び出し＋結果のホスト実体化（D2H）のみを含む。すなわち両者は同一の計測境界
ではなく、**fandhe-ai 側の `selected_gflops` は candle 側には含まれない A・B の
毎回の H2D 転送コストを余分に負っている**（fandhe-ai に有利な方向ではなく、
不利な方向のバイアスである）。したがって境界差を除いて同一境界へ揃えた場合、
selected/candle 比は**改善する方向**に動き、N=1024（約 0.83 倍）は「未達」判定が
覆る可能性を否定できない（N=4096 の約 0.10 倍は、`TILED_F32` 単体比でも 0.08 倍と
いう fandhe-ai 内部の同一境界比較が大幅悪化を示しているため、境界差の補正で
「達成」へ反転する見込みは薄いが、確定判断は同一境界での再計測を待つ）。
このため **N=1024・4096 は同一境界で再計測するまで「比較不能（暫定未達）」として
扱い、確定した採用判断には用いない**（Review 指摘・PR #1098）。N=256・512 の
「達成」（約 1.18〜1.22 倍）は同一境界へ揃えれば比率がさらに改善しうる側であり、
この境界差で過大評価される方向ではない（達成判定は維持）。

**採用判断（全 N で candle 以上で確定）**: **確定せず（暫定未達成）**。判定できた
4 形状のうち 256/512 は candle を上回る（約 1.18〜1.22 倍）が、**1024・4096 は本計測の
境界では candle を下回る**（約 0.83 倍・約 0.10 倍。上記のとおり同一境界での再計測まで
比較不能として扱い、確定判断には用いない）。ただし本番結線の可否は candle 比とは
独立に、次の fandhe-ai 内部比較だけで「結線しない」と判断できる:とくに N=4096 は選択された DoubleBuffer 変種が基準経路
（`TILED_F32`）比でも 0.08 倍まで低下しており、現在の選択ヒューリスティック
（`SPLITK_MIN_K`・`SPLITK_MAX_SPLITS`・`SPLITK_PARTIAL_MAX_BYTES`・
`DOUBLE_BUFFER_MIN_K` の暫定閾値）は N=1024・4096 の正方形状で DoubleBuffer 変種を
選択した結果、`TILED_F32` 単体よりも大幅に悪化させている。128×128×8192・
256×256×16384（K 支配的形状）は candle 参照値が本リポの既存ベースライン
ドキュメントに存在しないため判定不能とし、推定値は記入しない。128×128×8192 は
加えて正当性が未確定（上記）。

**本ファイル §1 手順 3 が指示する「未達の場合は暫定閾値の補正を 1 回だけ行い
再計測する」は本セッションでは実施していない**（実装 Agent の権限外。閾値変更＝
`gemm_variant.rs` のコード変更であり `docs/spec-proposal` 系のスコープ判断・
ユーザー承認が必要）。本番既定経路への結線判断（同手順 4）も同様に未実施。

## 1b. イシュー #1100 の修正内容（SplitK parity 失敗・DoubleBuffer 性能逆転）

「1a」節が検出した 2 件（SplitK の複合判定 FAIL・DoubleBuffer の `TILED_F32` 単体比劣化）に対する修正記録。**本 PR の実装セッションは CUDA 実機（GB10・Metal）への接続情報を持たないため、下記の是正は GPU 不要のホスト側検証（ユニットテスト・ホストモデルシミュレーション）のみで完結させ、GB10 実機での再実測は未実施のまま安全側で完了させる**（#994 の先例と同じ判断: 未検証の変更を選択条件の緩和方向へ倒さない）。

### SplitK: 撤退（parity FAIL の根本原因はカーネルバグではなく演算順序）

計画セッションでテストと同一の入力生成（`bench_harness::rng::Xorshift64Star`・seed=3・m=n=128・k=8192）・同一の CPU 参照（`matmul_reference_fma` の逐次 k 昇順 `mul_add` 連鎖）・SplitK と同一の数式モデル（split ごとの f32 FMA 連鎖 → s 昇順 f32 縮約・`k_per_split=1024`×8 分割）をホスト側 Rust で再現したところ（`crates/backend-cuda/tests/splitk_reorder_error_host_model.rs`）、GB10 実機の FAIL レポート（fail_count=8/16384・max_abs_diff=3.662e-4・max_rel_err=1.090e-2）と 3 指標が一致した（相対許容 5e-3 で機械検査済み）。さらに縮約・部分和の精度を f32→f64（ほぼ厳密値）へ引き上げても fail 数は減らないことも同テストで固定した（`higher_precision_reduction_does_not_eliminate_fail_count`）。

差分の支配項は CPU 参照実装（K=8192 の逐次 f32 `mul_add` 連鎖）自身の丸め誤差であり、真値ゼロ近傍（桁落ち）要素では絶対誤差救済 1e-5 を超えるため、**K 方向の累積順序を CPU 参照と一致させない限り、いかなる精度改善でも複合判定は通らない**（split-K という手法の本質と非両立）。tolerance（`RELATIVE_TOLERANCE`・`ABSOLUTE_RESCUE_THRESHOLD`）・参照実装（`matmul_reference_fma`）の変更はユーザー承認必須（`.claude/rules/security.md` A08）のため本イシューでは行わない。

よって `gemm_variant.rs::select_f32_gemm_variant` から SplitK 選択分岐を撤退した（K 支配的形状も `blocks < num_sms` のため DoubleBuffer の前提〈`blocks >= num_sms`〉を満たさず自然に `Simple` へ倒れる）。カーネル自体（`SPLITK_PARTIAL_F32`／`SPLITK_REDUCE_F32`。インデックス計算・境界チェック・決定性）は誤りではないため削除せず、`CudaGemmF32VariantSelection::run_split_k_forced`（診断専用。ヒューリスティックを経由しない明示起動）として保持する。spec 側での「split 順序を反映した parity 契約」の再検討要否はユーザー判断に委ねる（`out-of-scope-tracking.md` 準拠。本 PR では Issue 起票を行わない）。

### DoubleBuffer: バッファ管理を base と同一条件へ揃えた（性能逆転の是正）

`CudaGemm::run_tiled_f32`（base）は #1020 でプールアロケータ経由 `alloc_uninit_f32` へ移行済みだったが、`gemm_variant_selection.rs::run_double_buffer`／`run_split_k` は毎イテレーション raw `stream.alloc_zeros`（cuMemAlloc + memset + 都度解放）で C（および split-K の c_partial）を確保していた。GB10 実測で観測された「N が大きいほど悪化する」傾向（N=4096 で `TILED_F32` 単体比 0.08 倍）はこのバッファ管理差が有力仮説だったため、`CudaGemmF32VariantSelection` に `context_cache::cached_allocator` 経由のプールを導入し、DoubleBuffer・SplitK（診断専用）双方の出力バッファを base と同一の `alloc_uninit_f32` へ揃えた。

`alloc_uninit_f32` の安全性根拠（前利用データが D2H 前に必ず上書きされること）: `TILED_DB_F32` は `row < m && col < n` の全要素へ無条件書き込み（`num_tiles == 0` の早期 return パスも `c[row*n+col] = 0.0f` を無条件出力）、`SPLITK_PARTIAL_F32` も全 `(bz, row, col)`（`row<m && col<n`）へ無条件書き込み（末尾の空分割も `acc=0.0f` のまま出力）、`SPLITK_REDUCE_F32` も `idx < m*n` の全要素へ無条件書き込み。いずれも `docs/backend-cuda-pool-allocator-decision.md` §「`alloc_uninit` の適用」の確認済みパターンと同型。

**GB10 実機再実測は未実施**（本 PR の実装セッションに CUDA 実機接続なし）。バッファ管理差の是正がどの程度性能を改善するかは実機実測待ちであり、`gemm_variant.rs::DOUBLE_BUFFER_MIN_K` 等の選択閾値は本 PR では**変更していない**（安全側: 未検証のまま閾値を動かさない。実装計画 §5 手順 6 の「実機不達の場合の fallback」に従う）。次に実機接続を持つセッションで下記コマンドを実行し、結果を本節に追記すること:

```sh
cargo test -p fandhe-ai-backend-cuda --release --features internal-diagnostics -- --ignored --nocapture
cargo run -p fandhe-ai-backend-cuda --example gemm_f32_variant_bench --release --features internal-diagnostics
```

合格基準:

- `run_f32_matches_cpu_reference_across_variant_shapes`: 全形状で複合判定 green（128×128×8192 は撤退により `Simple` が選ばれ、`TILED_F32` と同一の数値契約になるため green のはず）
- `split_k_forced_execution_is_bit_deterministic_and_reproduces_gb10_fail`: 2 回実行が bit 一致し、複合判定は引き続き FAIL（PASS した場合は #1100 の撤退判断の前提が崩れているため要再検討）
- DoubleBuffer が選択される全形状（N=256/512/1024/4096・256×256×16384）で `TILED_F32` 単体比が 1.0 倍以上に改善しているか確認する。1.0 倍未満が残る場合は、実装計画 §5 手順 6 に従い（1 回限りの）閾値補正または DoubleBuffer 自体の選択除外を検討する（ユーザー承認のうえ実施）

### 記録欄（イシュー #1136。opt-in 診断テストのみ実行。補助・受入基準外）

2026-09-03 04:xx JST・DGX Spark GB10（sm_121）実機で以下 2 コマンドを実行（`docs/perf/cuda-gemm-simt-register-blocking.md` §7 の再実測セッションの一部として実施。commit `1a32082e4b521d7a0bed868db3a3b0a65e2bae9a`）:

```sh
cargo test -p fandhe-ai-backend-cuda --release --locked --features internal-diagnostics \
  --test gemm_f32_variants -- --ignored --nocapture --test-threads=1
```

結果: `run_f32_matches_cpu_reference_across_variant_shapes` ... ok・
`split_k_forced_execution_is_bit_deterministic_and_reproduces_gb10_fail` ... ok
（2 passed; 0 failed）。上記「合格基準」の 1 点目（`run_f32_matches_cpu_reference_across_variant_shapes`
全形状 green）は満たされたことを確認した。2 点目（`split_k_forced_…` が bit 決定的かつ
複合判定 FAIL を再現し続けること）は、Rust テスト関数自体がその再現を検証して `ok`
を返す設計であるため、テスト結果 `ok` は「FAIL の再現を確認できた」ことと整合する
（複合判定の内部統計は本記録では未転記）。

**未実施**: `cargo run -p fandhe-ai-backend-cuda --example gemm_f32_variant_bench --release
--features internal-diagnostics`（DoubleBuffer の `TILED_F32` 単体比の性能比較）は
イシュー #1136 の実装計画がこの記録欄への任意追記対象として明示していなかったため
未実行。DoubleBuffer 閾値補正の要否判断は引き続き未実施のまま（ユーザー承認事項）。

生ログ: `docs/perf/logs/cuda-simt-remeasurement-1136/parity_gemm_f32_variants.log`

### 本 PR で検証済みの事項（GPU 不要）

- `cargo build --workspace`
- `cargo fmt --all -- --check` / `cargo clippy --workspace --all-targets --all-features -- -D warnings`
- `cargo test -p fandhe-ai-backend-cuda --all-features`（`gemm_variant::tests`〈撤退後の新契約〉・`kernels_gemm_variants::tests`・`splitk_reorder_error_host_model.rs`〈GB10 レポート値の再現・精度モデル比較の 2 件〉・`gemm_f32_variants.rs`〈環境適応スモーク 2 件。`#[ignore]` 2 件は未実行〉を含む全 green）
- `cargo test --workspace` / `cargo test --workspace --all-features`（回帰確認）
- `git diff origin/main -- crates/backend-cuda/tests/parity_nonregression.rs crates/backend-cuda/tests/common/parity_baseline crates/backend-cuda/src/kernels.rs crates/backend-cuda/src/kernels_mma.rs crates/backend-cuda/src/kernels_wmma.rs crates/backend-cuda/src/kernels_wmma_opt.rs` が無差分（既存カーネル・fixture・tolerance を一切変更していないことの機械確認）

## 1. 実機手順

前提: CUDA driver + NVRTC 搭載実機（DGX Spark GB10 等。`docs/real-hardware-verification-env.md` の接続手順）。

```sh
git fetch origin
git checkout perf/1035-f32-variant-selection   # 本イシューの実装ブランチ
cargo test -p fandhe-ai-backend-cuda --features internal-diagnostics -- --ignored --nocapture
cargo run -p fandhe-ai-backend-cuda --example gemm_f32_variant_bench --release --features internal-diagnostics
```

1. **`#[ignore]` テストを実行**し、複合判定（相対誤差 1e-3 未満 または 絶対誤差 1e-5 未満）・SplitK 決定性（2 回実行 bit 一致）・境界形状（33/65/1000 等の非整列サイズ・K 支配的非正方形状）がすべて green であることを確認する
2. **A/B ベンチを実行**し、下記記録欄へ N=256〜4096（アラインメント済み正方）・K 支配的非正方形状（split-K 対象）ごとの `variant`（選択された変種）・`base_tflops`（`TILED_F32`）・`selected_tflops`（選択された変種）・candle 実測値（`docs/perf/cuda-gemm-kernel-vs-frameworks-baseline.md`）との比を記入する
3. **受け入れ条件 (a)「全 N で candle 以上」を判定する**。満たさない場合は暫定閾値の補正を **1 回だけ** 行い再計測する（補正はコミット・PR に根拠を明記する。補正ループ禁止）
4. **本番既定経路への結線判断**: 受け入れ条件を満たし、かつ数値検証（`#[ignore]` テスト）が全 green であることを確認したうえで、ユーザー承認を得てから後続 Issue として本番結線（`gemm_auto.rs::CudaGemmAuto::run_f32` からの呼び出し）を実施する。本 PR ではこの結線を行わない

### 記録欄

DGX Spark GB10 実機実測完了。データ本体・正当性検証結果・採用判断は
**上記「1a. GB10 実機実測結果」節**に記録済み（本節は重複を避けるため転記しない）。

## 2. GPU 不要ユニットテストの検証範囲

**イシュー #1100 で更新**: SplitK は `select_f32_gemm_variant` の選択候補から撤退した（下記「1b」節）。

`crates/backend-cuda/src/gemm_variant.rs::tests`（`select_f32_gemm_variant`・`derive_split_count`・`validate_split_k_launch`）:

- num_sms 判定不能（`None`）・境界条件（0 次元・`num_sms=0`）では常に `Simple`
- K 支配的な形状（grid が SM を埋められない）は SplitK ではなく `Simple` が選ばれる（撤退後の契約。`k_dominant_small_grid_shape_selects_simple_not_split_k`）
- DoubleBuffer: grid が SM を十分埋めるアラインメント済み大形状で選ばれる／非整列・候補利用不能・K 閾値未満では選ばれない
- `derive_split_count`（診断専用 `recommend_split_count` 経由で到達可能）が常に 2 冪（または 1）かつ `[1, SPLITK_MAX_SPLITS]` の範囲内であること・`num_blocks`/`num_sms` が 0 の場合は 1 を返すこと
- 選択の決定性（同一入力 2 回で同一結果）・巨大形状（`i32::MAX` 近傍）での非 panic（`u64` オーバーフロー安全性）
- `validate_split_k_launch`（診断専用 `run_split_k_forced` が起動前に呼ぶ）の範囲外 `num_splits` 拒否・cap 超過拒否・正常系受理

`crates/backend-cuda/tests/splitk_reorder_error_host_model.rs`（GPU 不要。イシュー #1100 で新規追加）:

- GB10 実機レポート（fail_count=8/16384・max_abs_diff=3.662e-4・max_rel_err=1.090e-2）と同一の形状・分割数・シードで split-K の数式モデル（f32 部分和 + f32 縮約）をホスト側 Rust で再現し、3 指標が一致することを固定する（`f32_partial_f32_reduce_matches_gb10_report`）
- 部分和・縮約の精度を f64（ほぼ厳密値）まで引き上げても複合判定 FAIL が解消しないことを固定する（`higher_precision_reduction_does_not_eliminate_fail_count`）

`crates/backend-cuda/src/kernels_gemm_variants.rs::tests`（カーネルソース構造検査。`kernels_rmsnorm.rs` の split-K テスト群と同型。SplitK カーネル自体は診断専用として保持しているため引き続き検証する）:

- split-K 部分和カーネルが `c_partial` へ無条件に 1 回だけ書くこと（末尾要素ブロックの扱い）
- split-K の 2 カーネルがいずれも atomics を使わないこと（決定的書き込み）
- 縮約カーネルが `c` へ 1 回だけ書き、`c_partial` へは書き戻さないこと（第 3 パスを作らない契約）
- 縮約の反復順序が `s` 昇順の固定順序であること（決定性の根拠）
- double-buffer カーネルの smem が 2 面であること・C への書き込み時の手動境界チェック（REQ-8）・タイルロードの三項ガードを維持していること
- split-K 部分和カーネルのタイルロードも三項ガードを維持していること

`crates/backend-cuda/tests/gemm_f32_variants.rs`（`internal-diagnostics` feature 限定。イシュー #1100 で `KDominant` の期待値・決定性テストを更新）:

- 環境適応スモーク（非 ignore）: `CudaGemmF32VariantSelection::new` が CUDA 非搭載環境で panic せず型付きエラーを返すこと・`selected_variant`（SplitK を返さない）が panic しないこと
- `#[ignore]`（実機必須。本ランは未実行）: CPU 参照実装との複合判定（アラインメント済み大形状・K 支配的非正方は撤退後の契約〈`Simple` または `DoubleBuffer`〉・非整列・境界サイズを網羅する `run_f32_matches_cpu_reference_across_variant_shapes`）・`run_split_k_forced` の bit 決定性と GB10 FAIL 再現（`split_k_forced_execution_is_bit_deterministic_and_reproduces_gb10_fail`）

## 3. スコープ外・追跡事項（`out-of-scope-tracking.md` 準拠）

- 本番既定経路（`CudaGemm::new`・`run_tiled_f32`・`CudaGemmAuto::run_f32`）への選択ヒューリスティック結線は、実機実測（A/B・複合判定・parity 非後退）とユーザー承認後の後続作業とする
- DoubleBuffer 是正後の GB10 実機再実測（本 PR〈#1100〉の実装セッションでは実行不能。上記「1b」節）
- 暫定閾値（`DOUBLE_BUFFER_MIN_K`・occupancy 係数・バッファ cap）の実機補正は上記結線判断とセットで実施する（補正は 1 回限り・補正ループ禁止）
- split 順序を反映した parity 契約（参照実装・判定方式）の spec 側検討要否（`docs/spec/` は編集しないため、必要なら Fandhe-AI/fandhe-ai-spec 側でユーザーが判断する。上記「1b」節）
- #1033（cp.async 多段パイプライン）の DoubleBuffer 候補への差し替え統合は #1033 マージ後の後続判断とする（`gemm_variant_selection.rs::CudaGemmF32VariantSelection` の候補は `Option` スロットで保持しており差し替え可能な構造）
- resident 経路（`launch_tiled_f32_resident`）への変種適用は本 PR のスコープ外とし、本番既定経路への結線判断と同時に検討する
