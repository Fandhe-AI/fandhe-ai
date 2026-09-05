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
#1215（Metal NT/TN strided 結線）・`dense_transposed_view` の
`tensor-core` への昇格（公開 API 変更を伴う）。

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

## 3. 計測プロトコル・実測結果（未実施）

**本実装セッションの実行環境には CUDA 実機（DGX Spark GB10 等）が
存在しないため、以下は未実施のまま記入欄を残す**（イシュー #1214 の
再分解記録。issue 本文「再分解（実装 Agent 追記）」節参照）。実装
（コード変更・GPU 非依存テスト・`#[ignore]` テストの型検査）自体は
本セッションで完了している。

### 3.1 実機テスト（`#[ignore]`。未実施）

```sh
cargo test -p fandhe-ai-backend-cuda --release --test gemm_transposed_parity -- --ignored --nocapture
cargo test -p fandhe-ai-backend-cuda --release --test gemm_transposed_perf -- --ignored --nocapture
```

### 3.2 補助 A/B（未実施）

`crates/backend-cuda/tests/gemm_transposed_perf.rs` を warmup 2 回・
measured 5 回中央値で実行する（CPU 版 §3.1 と同型のプロトコル）。
形状は `(64,784,256)`・`(64,256,10)`・`(1024,1024,1024)`・
`(2048,2048,2048)` × NT/TN。生ログの記録先: `docs/perf/logs/cuda-gemm-
vjp-transposed-entry-1214/{env_info.txt,aux_ab.txt}`（内部ホスト名は
含めない）。

| パターン | m | k | n | before 中央値 (s) | after 中央値 (s) | 倍率 |
|----------|---|---|---|-------------------|-------------------|------|
| NT | 64 | 784 | 256 | （未実測） | （未実測） | — |
| NT | 64 | 256 | 10 | （未実測） | （未実測） | — |
| NT | 1024 | 1024 | 1024 | （未実測） | （未実測） | — |
| NT | 2048 | 2048 | 2048 | （未実測） | （未実測） | — |
| TN | 64 | 784 | 256 | （未実測） | （未実測） | — |
| TN | 64 | 256 | 10 | （未実測） | （未実測） | — |
| TN | 1024 | 1024 | 1024 | （未実測） | （未実測） | — |
| TN | 2048 | 2048 | 2048 | （未実測） | （未実測） | — |

### 3.3 train fresh/reuse A/B（未実施）

`docs/perf/train-backward-gemm-wiring.md` §3 と同一の参考系列方式
（`scripts/bench/framework-compare` を `--config 'patch.crates-io.
fandhe-ai.path="<facade 絶対パス>"'` で before〈`origin/main`〉／
after〈本ブランチ HEAD〉別々にビルド）。`bench-fandhe --task train
--device cuda --size 64 --mode {fresh,reuse}` を各 5 run。

| モード | 指標 | before 中央値 | after 中央値 | 倍率 |
|--------|------|---------------|---------------|------|
| fresh | backward | （未実測） | （未実測） | — |
| fresh | step_total | （未実測） | （未実測） | — |
| reuse | backward | （未実測） | （未実測） | — |
| reuse | step_total | （未実測） | （未実測） | — |

## 4. 採否判断（保留）

実機実測が未実施のため、ADOPT／REJECT の確定判断は保留する。ただし
以下の理由により、コード上の結線自体はメモリ `prod-wiring-preapproved`
（本番結線は事前承認済み・後退の可能性は前後比較を記録する運用）に
従い実施済みである:

1. §2 のとおり計算結果は設計上 bit 完全一致（CPU 版 #1213 と同型の
   契約）であり、正しさへのリスクは実装レベルでは低い
2. 転置カーネル自体は #601 で実装・単体テスト済みであり、既存カーネル
   選択ロジック（`select_tiled_f32_kernel`・`kernel_specs()`）には
   一切手を入れていない
3. 小形状（層 2 相当 `m=64,k=256,n=10`）では「ホストの数 KB の strided
   copy」を「カーネル起動 + プール確保」に置き換えるため、実装計画
   §3.2 が指摘するとおり**後退の可能性が実在する**。§3.2 の実測完了
   まで、この後退可能性は解消されていない

後続セッションが §3 を実測した時点で、§3.2・§3.3 の結果に基づき本節
を更新し ADOPT／REJECT を確定する（`docs/perf/train-backward-gemm-
wiring.md` §7.3 の判断規則と同じ運用: 補助 A/B 全形状・train
backward/step_total とも非後退なら ADOPT。小形状のみ後退で train 総和
が非後退なら「記録して受容」または「numel 閾値ゲート追加」を選び
明記する。train 総和で後退なら結線を無効化して入口は残す）。

## 5. 後続

- 本ドキュメント §3〜§4 の GB10 実機実測・採否確定（イシュー #1214
  本文「再分解」節の (b)）
- #1212: reuse 経路の grad をデバイス常駐のまま `device_update` へ直結
- #1215: Metal GEMM の NT/TN strided 結線
- TT（両方転置）・一般 stride 化・TF32 opt-in 経路・`gemm_bias_act`
  融合経路・`gemm_resident_rhs` への適用: 本イシューでは対象外のまま
  （`docs/matmul-vjp-zero-copy-decision.md` §3.2・§4.3 の該当行は変更
  しない）
- `dense_transposed_view` の `tensor-core` への昇格（公開 API 変更を
  伴うため別途承認が必要）
