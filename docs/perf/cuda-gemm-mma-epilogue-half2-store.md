# CUDA mma_f16 エピローグ `__half2` ベクトル store 計測記録（#805）

イシュー #805「perf(backend-cuda): mma_f16 エピローグの `__half2` ベクトル化
store」の設計根拠・bit 一致論拠・実測記録。

## 変更内容

`crates/backend-cuda/src/kernels_mma.rs` の `MMA_F16_SOURCE` テンプレート
（`mma_f16_source()`・`render_mma_f16` の両経路が参照する単一テンプレート）の
エピローグを、mma アキュムレータ `d[mi][nj][0..3]` を C へ書き戻す際の
隣接列ペア（`c0`・`c1 = c0 + 1`）に対する `__float2half` スカラー store
4 回から、`__floats2half2_rn` による `__half2`（4 バイト）ベクトル store
2 回（行 `r0` 分・行 `r1` 分）へ置き換えた（CUTLASS
`predicated_tile_iterator.h` の AlignedArray 方式を簡易化した形）。
エピローグの guarded store 命令数を半減させる目的（受け入れ基準 1）。

## 設計根拠（bit 一致・境界検査非希釈の論拠）

- **整列**: ホスト側 `gemm_mma.rs::validate_mma_alignment` が起動前に
  `n % 8 == 0 && k % 8 == 0` を fail-closed で検査するため（`MMA_N = 8`）
  `DIM_N` は常に偶数。`c0 = col0_warp + nj * MMA_N + tid_in_group * 2`
  （`col0_warp` は 8 の倍数）も常に偶数。ゆえに要素添字
  `r * DIM_N + c0` は常に偶数 → バイトオフセットは 4 の倍数となり、C
  バッファ（cudarc デバイス確保・256B 整列）基点からの `__half2`
  （4B 整列要求）store は常に整列する。
- **ペアの全有効/全無効**: `DIM_N` 偶数・`c0` 偶数のため `c0 < DIM_N` ⇔
  `c1 < DIM_N`。列ペアは「両方有効」か「両方無効」の二値になり、ペア単位の
  境界判定（`c1 < DIM_N`）が REQ-8 の手動境界検査を弱めずに成立する
  （cp.async 側「8 要素チャンクが境界を跨がない」設計と同じ論法）。
- **丸めの同一性**: `__floats2half2_rn` は `__float2half` と同じ
  round-to-nearest-even のため、書き込まれる値は変わらない（bit 一致）。
  `tests/parity_nonregression.rs` の parity 非後退契約（fixture・
  tolerance）は変更不要。
- **defensive fallback**: 上記不変条件下では `c0 < DIM_N` かつ
  `c1 >= DIM_N` は到達不能だが、将来 `DIM_N` 偶数制約が緩んだ場合に列
  `c0` 側の書き落としが起きないよう、ペア判定が不成立のときは `c0`
  単独のスカラー store へフォールバックする実装を残した（fail-closed。
  `.claude/rules/coding-rust.md` 「境界検査を無効化する最適化はシェーダ
  側で手動境界チェックを維持したうえで行う」に準拠）。

## ホスト側テスト（本セッションで実施・green）

CUDA 実機非依存（GPU driver 不要）の範囲は本セッションのサンドボックス
（CUDA driver はあるが `libnvrtc` 非搭載。下記「現状」節参照）で完結して
実施した:

```bash
cargo fmt --all
cargo clippy --workspace --all-targets --all-features -- -D warnings
cargo test -p fandhe-ai-backend-cuda
cargo build --workspace --all-targets
```

いずれも green（`cargo test -p fandhe-ai-backend-cuda` はライブラリ内 needle・render
テスト（`kernels_mma::tests::*`）37 件 pass を含む。新規追加した
`mma_f16_source_epilogue_uses_half2_pair_store`（#805 受け入れ基準 1 の
機械検査: `__half2` ペア store の存在・旧 4 連スカラー store パターンの
不在をロック）・更新した `mma_f16_source_epilogue_store_is_inside_warp_tile_loop`
（store needle を新パターンへ更新）を含む）。

## 現状: 実機検証未完・ブロック中

本セッションの実行環境には CUDA **driver**（`nvidia-smi` で
`NVIDIA GeForce RTX 3060`・compute capability 8.6 を検出）は存在するが、
NVRTC（`libnvrtc`）は存在せず `nvrtc::compile_ptx` は
`CudaError::NvrtcUnavailable` を返す（本ファイル冒頭コメント「検証状態」
と同じ既知制約。`cargo test -p fandhe-ai-backend-cuda --release --test
cpu_cuda_mma_parity -- --ignored --nocapture` で実際に
`NvrtcUnavailable { detail: "libnvrtc dynamic library not found
(dlopen failed); ..." }` を確認済み）。したがって以下は**未実施**である
（実測値を捏造しない。`.claude/rules/coding-rust.md`「ベンチは 5 回計測の
中央値を採用」・security.md の実測原則に従う。#599 の
`cuda-gemm-epilogue-fusion.md` と同型の記録方式）:

- 数値一致: `cpu_cuda_mma_parity.rs` 全形状・K4096 ストレス・
  `specialized_mma_parity.rs`・`parity_nonregression.rs` の実機
  `--ignored` テスト
- 性能: `gemm_mma_bench` example の変更前後 5 回計測中央値比較
- PTX 確認（`examples/mma_ptx_dump.rs`）による store 命令が
  `st.global.u32`（ペア化）になったことの直接確認

DGX Spark GB10（sm_121）・本セッションで検出した RTX 3060（sm_86）の
いずれについても、NVRTC 経由のコンパイル自体が本セッションから実行不能
なため構文検証も未完了である。

## 実機検証時の再現コマンド（未実施・手順のみ記録）

```bash
# docs/real-hardware-verification-env.md の手順で対象 CUDA 実機へ
# ブランチを転送したうえで、実機上で実行する。

# 1. mma 系数値一致（最初の実行が NVRTC 構文検証を兼ねる）
cargo test -p fandhe-ai-backend-cuda --release --test cpu_cuda_mma_parity \
  -- --ignored --nocapture
cargo test -p fandhe-ai-backend-cuda --release --test specialized_mma_parity \
  -- --ignored --nocapture
cargo test -p fandhe-ai-backend-cuda --release --test parity_nonregression \
  -- --ignored --nocapture

# 2. 性能（変更前後で 5 回計測し中央値を比較）
cargo run -p fandhe-ai-backend-cuda --example gemm_mma_bench --release

# 3.（任意）PTX 上で store がペア化されたことの確認
cargo run -p fandhe-ai-backend-cuda --example mma_ptx_dump --release
```

## 実測すべき項目（実機到達後に本節を実測値で置き換える）

| 項目 | 状態 |
|------|------|
| mma_f16 の CPU-CUDA 数値一致（全形状・parity 非後退契約 bit 一致） | 未実施 |
| K=4096 ストレスケースの数値一致 | 未実施 |
| `gemm_mma_bench` の変更前後 5 回計測中央値比較（非劣化確認） | 未実施 |
| PTX store 命令のペア化確認（任意） | 未実施 |

実機へ到達できない状態で「性能非劣化」を主張しない（#771 の先例に従う）。
本 PR は上記が未実施であることを明示した状態でコミットし、実機検証は
別途（レビュー・マージ後の実機ゲート）で実施する。
