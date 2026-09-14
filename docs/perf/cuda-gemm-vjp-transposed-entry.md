# CUDA GEMM VJP 専用 NT/TN 転置入口（イシュー #1214）

## 0. 目的・スコープ

`docs/matmul-vjp-zero-copy-decision.md` §3.2 表 2 行目「CUDA 本番 GEMM
カーネルの lda／転置対応」を、CPU 版（#1213・`docs/perf/cpu-gemm-vjp-
transposed-entry.md`）と同じ **NT（`b` が転置格納）／TN（`a` が転置
格納）の 2 パターン限定**で解消する。方式は CPU の BLIS packing 側
吸収とは異なり、**GPU 側 smem 転置カーネル（`kernels_transpose::
transpose_smem_source_f32(false)`。#601 実装済み・`docs/perf/cuda-gemm-
transpose-ab.md` §2 の結線イシュー）→ 既存 NN GEMM カーネル
（`select_tiled_f32_kernel` が選ぶ classic／cp.async パイプライン）**を
採る（設計判断の詳細は `docs/matmul-vjp-zero-copy-decision.md` §4.3）。

対象外（本イシューのスコープ外）: TT（両方転置）・一般 stride
（`narrow` 後の転置等）・TF32 opt-in 経路（`run_wmma_tf32`）・
`gemm_bias_act` の融合経路・`gemm_resident_rhs`・`Op::LinearResident.
d_input` のデバイス側直接計算・#1212（reuse 経路の grad 常駐化）・
Metal NT/TN strided 結線（→ #1215 で完了。`docs/perf/metal-gemm-vjp-
transposed-entry.md`）・`dense_transposed_view` の `tensor-core` への
昇格（公開 API 変更を伴う）。

## 1. コード変更

| ファイル | 変更内容 |
|---------|---------|
| `crates/backend-cuda/src/transpose.rs` | `tiled_launch_config` を `pub(crate)` 化し `gemm.rs` から再利用。既存 `CudaTranspose` の挙動は無変更 |
| `crates/backend-cuda/src/error.rs` | `CudaError::TransposeEntryUnavailable`（VJP 専用 NT/TN 転置入口の smem 転置カーネルが `CudaGemm::new` 時点で使用不能な場合の型付きエラー）を追加 |
| `crates/backend-cuda/src/gemm.rs` | `CudaGemm` に `transpose_smem_f32: Option<CudaFunction>`／`transpose_smem_f32_error: Option<String>` フィールド・`compile_transpose_smem_f32`（`load_function_cached` 経由ロード。`new` の早期 return には合流させない fail-soft）・可観測点 `GEMM_TRANSPOSED_ENTRY_LAUNCH_COUNT`（thread_local）・内部ヘルパー `transpose_to_pooled`（`validate_transpose_dims`／`validate_transpose_output_len` → `alloc_uninit_f32` → smem 転置起動）・公開入口 `run_tiled_f32_nt`／`run_tiled_f32_tn`／`launch_tiled_f32_resident_nt`（いずれも `pub(crate)`）・可用性照会 `transpose_smem_f32_available` を追加 |
| `crates/backend-cuda/src/ops.rs` | `dense_transposed_view`（CPU 版 `backend-cpu::ops::dense_transposed_view` と同一ロジックの private 複製）・可観測点 `GEMM_HOST_REPACK_COUNT` を追加。`gemm_fp32_strict_impl` を NT/TN 判定で分岐（フォールバックは `gemm_fp32_strict_fallback` に共通化）。`gemm_resident_lhs` の `b` アップロードを NT 判定で分岐（`MemoryOps::upload` の代わりに `bt` の生 storage を直接 `clone_htod` して `launch_tiled_f32_resident_nt` へ渡す） |
| `crates/backend-cuda/tests/gemm_transposed_parity.rs`（新規・`#[ignore]`） | `CudaBackendOps::gemm_fp32_strict`／`gemm_resident_lhs` 経由の bit 完全一致（NT/TN/TT/一般 stride）＋ CPU 参照実装との REQ-2 複合判定 |
| `crates/backend-cuda/tests/gemm_transposed_perf.rs`（新規・`#[ignore]`） | 本ドキュメント §3 の補助 A/B 計測 |
| `crates/backend-cuda/src/ops.rs`（`#[cfg(test)]`） | `dense_transposed_view_tests`（GPU 不要の純ロジック）・`repack_count_tests`（env-adaptive。CUDA 非搭載環境では `BackendError::CudaUnavailable` で早期 return） |
| `docs/matmul-vjp-zero-copy-decision.md` | §4.3 追補 |
| `docs/perf/cuda-gemm-transpose-ab.md` | §2 追記（smem パディング変種の結線先明記） |

`crates/tensor-core`（`Tensor`・`BackendOps` trait の公開 API）・
`backend-cpu`・`backend-metal`・`Cargo.toml`／`Cargo.lock`（依存追加
なし）・tolerance 定数（`RELATIVE_TOLERANCE`／`ABSOLUTE_RESCUE_
THRESHOLD`）・`tests/common/parity_baseline.rs` の `BASELINES`・
`kernel_specs()` の長さ（8 のまま）は無変更。

## 2. 数値一致契約

GEMM カーネル（classic／cp.async パイプラインいずれも）に渡るデバイス
上のバイト列は「転置オペランドを `contiguous()` してから upload した
場合」と同一になる設計（転置カーネルは smem 経由の純データ移動のみで
丸めを一切追加しない）。GEMM 本体のカーネル選択（`select_tiled_f32_
kernel`）・累積順序・FMA 契約も NT/TN 経路と NN 経路で完全に同一の
呼び出し（同じ `m,n,k` から導出される同じ関数・同じ `LaunchConfig`）
のため、計算結果は **bit 完全一致**する契約
（`crates/backend-cuda/tests/gemm_transposed_parity.rs` で検証。CPU
参照実装との REQ-2 統一複合判定〈相対誤差 1e-3 未満 または 絶対誤差
1e-5 未満〉も `fandhe_ai_backend_cpu::assert_parity` で併せて確認する）。
tolerance の新設・変更は行っていない。

GPU 側 smem 転置カーネル（`kernels_transpose::transpose_smem_
source_f32(false)`）の epilogue ストアガード `if (out_row < n &&
out_col < m)` は出力グリッド全体（`rows*cols` 要素）を標準の行列転置
としてちょうど 1 回ずつ書き切る（重複書き込み・欠落のいずれも生じ
ない）ため、`transpose_to_pooled` の中間バッファ確保に `alloc_uninit_
f32`（前利用データが起動完了までに全要素上書きされ露出しない。
`docs/backend-cuda-pool-allocator-decision.md` §「`alloc_uninit` の
適用」の確認済みケースに準じる）を用いている。

## 3. 計測プロトコル・実測結果

**§3.1（実機テスト）・§3.2（補助 A/B）は低レイヤー診断（イシュー
#1574・2026-09-12・GB10 専有・tree `097bff19`）で実測済み**（診断系列。
`docs/perf/lowlayer-diagnosis-2026-09-12.md` §2）。**§3.3（train
fresh/reuse A/B）は本ドキュメント作成時点でも未実施**であり、イシュー
#1590 が実行スキャフォールドを整備した（実測本体は GB10 実機セッション
へ引き継ぎ。詳細は `docs/perf/logs/cuda-gemm-vjp-transposed-entry-1590/
README.md`）。

### 3.1 実機テスト（`#[ignore]`。実測済み・#1574）

```sh
cargo test -p fandhe-ai-backend-cuda --release --test gemm_transposed_parity -- --ignored --nocapture
cargo test -p fandhe-ai-backend-cuda --release --test gemm_transposed_perf -- --ignored --nocapture
```

`gemm_transposed_parity`（5 件）は 2026-09-12 に GB10 実機（専有・
`--features internal-diagnostics --test-threads=1`）で **5/5 pass**
（出典 `docs/perf/logs/lowlayer-diagnosis-2026-09-12/dgx/
gemm_transposed_parity.log`）。負荷は当該計測ステージ前後の `uptime`
実測で load average 1.05〜1.76 の範囲（`uptime_before_all.txt` 1.05・
`uptime_after_all.txt` 1.76・`uptime_after_sweep.txt` 1.56。`docs/perf/
lowlayer-diagnosis-2026-09-12.md` §1 注 1 の系列表記のとおり、専有環境
だが他ログインが残存し単一値では表せない）。

### 3.2 補助 A/B（診断系列・#1574。正式な 5 プロセス起動値は #1590 の
記入欄）

`crates/backend-cuda/tests/gemm_transposed_perf.rs`（`nt_transposed_
entry_vs_contiguous_across_shapes`／`tn_transposed_entry_vs_contiguous_
across_shapes`。各テスト内部で before/after を計測し `speedup` を出力）
を 2026-09-12 に GB10 実機で **1 プロセス起動**実行した値（出典
`docs/perf/logs/lowlayer-diagnosis-2026-09-12/dgx/
gemm_transposed_perf.log`）。**§3.3 と同じ事前登録規則における「正式な
補助 A/B」は 5 プロセス起動中央値**（`docs/perf/logs/cuda-gemm-vjp-
transposed-entry-1590/README.md`「対象テスト」節）であり、下表は診断
系列（1 起動値）として記録するに留め正式値へ昇格させない。

| パターン | m | k | n | before 中央値 (s) | after 中央値 (s) | 倍率（診断系列・1 起動） |
|----------|---|---|---|-------------------|-------------------|--------------------------|
| NT | 64 | 784 | 256 | 0.000439 | 0.000079 | 5.563x |
| NT | 64 | 256 | 10 | 0.000038 | 0.000036 | 1.057x |
| NT | 1024 | 1024 | 1024 | 0.002463 | 0.000431 | 5.709x |
| NT | 2048 | 2048 | 2048 | 0.013150 | 0.002219 | 5.926x |
| TN | 64 | 784 | 256 | 0.000160 | 0.000072 | 2.242x |
| TN | 64 | 256 | 10 | 0.000050 | 0.000024 | 2.096x |
| TN | 1024 | 1024 | 1024 | 0.002458 | 0.000461 | 5.337x |
| TN | 2048 | 2048 | 2048 | 0.022510 | 0.007973 | 2.823x |

8 形状すべてで倍率 ≥ 1.0（後退なし）。小形状 `m=64,k=256,n=10`（層 2
相当。旧 §4 が後退リスクとして名指ししていた形状）も NT 1.057x・
TN 2.096x と僅かではあるが改善方向であり、当該診断系列の範囲では §4
（旧）が懸念した後退は観測されなかった（1 起動のみのため確定判断には
用いない）。

ログ置き場: `docs/perf/logs/cuda-gemm-vjp-transposed-entry-1590/`（正式
5 起動値・train A/B の記入欄。診断系列そのものの生ログは `docs/perf/
logs/lowlayer-diagnosis-2026-09-12/dgx/gemm_transposed_{parity,perf}.log`
に不変のまま残す）。

### 3.3 train fresh/reuse A/B（未実施。イシュー #1590 でスキャフォールド
整備済み）

`docs/perf/train-backward-gemm-wiring.md` §3 と同一系統の参考系列方式
だが、本イシュー固有の理由（本ブランチ HEAD の `bench-fandhe` ハーネス
は #1214 マージ直前の facade に存在しない API を呼ぶためビルド不能）
により、単一チェックアウトへの facade path patch ではなく **before
（`82058501`。#1214 マージ直前）／after（`ab0b77d0`。#1214 マージ
コミット自身）をそれぞれ丸ごと展開したツリー**を使う（各ツリー自身の
`scripts/bench/framework-compare/` で、その場の `crates/facade` への
`[patch.crates-io.fandhe-ai]` path patch を適用してビルドするため、
ハーネスと facade のバージョンが常に一致する）。

実行スクリプト: `scripts/bench/framework-compare/
run_ab_vjp_transposed_cuda.sh <label>`（`AB_BEFORE_TREE`／
`AB_AFTER_TREE` に両ツリーの絶対パスを指定）。オーケストレーション・
事前登録判定規則・記入欄は `docs/perf/logs/cuda-gemm-vjp-transposed-
entry-1590/`（`orchestrate.sh`・`run_ignored_tests.sh`・
`aggregate_aux_ab.py`・`env_info.txt`）。`bench-fandhe --task train
--device cuda --size 64 --mode {fresh,reuse}` を各 5 round・起動順反転
で実行し、**fresh／reuse 両方**を Tier 1 必須判定とする（#1560／#1689
が reuse のみを判定対象としたのと異なり、fresh も `matmul_vjp`
〈`g @ bᵀ` NT・`aᵀ @ g` TN〉経由で NT／TN 入口へ到達するため）。

| モード | 指標 | before 中央値 | after 中央値 | 倍率 |
|--------|------|---------------|---------------|------|
| fresh | backward | （未実測） | （未実測） | — |
| fresh | step_total | （未実測） | （未実測） | — |
| reuse | backward | （未実測） | （未実測） | — |
| reuse | step_total | （未実測） | （未実測） | — |

本エージェント実行環境（Linux。`docs/real-hardware-verification-env.
local.md`・`CUDA_NODE` 未確認・ローカル GPU も driver/library version
mismatch で初期化不可）には DGX Spark GB10 実機への到達手段がないため、
上表は未実測のまま GB10 実機セッションへ申し送る。

## 4. 採否判断（保留・イシュー #1590 で更新）

§3.1（parity 5/5 pass）・§3.2（補助 A/B 8/8 形状 ≥1.0 倍。診断系列・
1 起動）は実測済みだが、§3.3（train A/B・正式判定対象）は未実測のため
ADOPT／REJEC の確定判断は保留する。ただし以下の理由により、コード上の
結線自体はメモリ `prod-wiring-preapproved`（本番結線は事前承認済み・
後退の可能性は前後比較を記録する運用）に従い実施済みである:

1. §2 のとおり計算結果は設計上 bit 完全一致（CPU 版 #1213 と同型の
   契約）であり、正しさへのリスクは実装レベルでは低い（§3.1 の
   parity 5/5 pass で実測確認済み）
2. 転置カーネル自体は #601 で実装・単体テスト済みであり、既存カーネル
   選択ロジック（`select_tiled_f32_kernel`・`kernel_specs()`）には
   一切手を入れていない
3. 小形状（層 2 相当 `m=64,k=256,n=10`）では「ホストの数 KB の strided
   copy」を「カーネル起動 + プール確保」に置き換えるため、後退の
   可能性が理論上は残る。§3.2 の診断系列（1 起動）では当該形状も
   NT 1.057x・TN 2.096x と改善方向だったが、1 起動のみのため §3.3
   の train A/B（size=64・fresh/reuse）が正式な確定判断の対象である
   ことに変わりはない

後続セッションが §3.3 を実測した時点で、その結果に基づき本節を更新し
ADOPT／REJECT を確定する（`docs/perf/train-backward-gemm-wiring.md`
§7.3 の判断規則と同じ運用: 補助 A/B 全形状・train backward/step_total
とも非後退なら ADOPT。小形状のみ後退で train 総和が非後退なら「記録
して受容」または「numel 閾値ゲート追加」を選び明記する。train 総和で
後退なら結線を無効化して入口は残す）。判定規則自体は `docs/perf/logs/
cuda-gemm-vjp-transposed-entry-1590/README.md`「事前登録判定規則」節に
計画確定済みであり事後に緩和しない。

## 5. 後続

- 本ドキュメント §3.3・§4 の GB10 実機実測・採否確定（イシュー #1590。
  スキャフォールドは整備済み。実測は GB10 実機セッションへ申し送り）
- #1212: reuse 経路の grad をデバイス常駐のまま `device_update` へ直結
- #1215: Metal GEMM の NT/TN strided 結線 → 完了
  （`docs/perf/metal-gemm-vjp-transposed-entry.md`）
- TT（両方転置）・一般 stride 化・TF32 opt-in 経路・`gemm_bias_act`
  融合経路・`gemm_resident_rhs` への適用: 本イシューでは対象外のまま
  （`docs/matmul-vjp-zero-copy-decision.md` §3.2・§4.3 の該当行は変更
  しない）
- `dense_transposed_view` の `tensor-core` への昇格（公開 API 変更を
  伴うため別途承認が必要）
- 小形状後退時の numel 閾値ゲート・train 総和後退時の結線無効化は
  本番コード変更を伴うため、§3.3 実測の結果次第で別イシューへ切り出す
  （out-of-scope-tracking.md）
