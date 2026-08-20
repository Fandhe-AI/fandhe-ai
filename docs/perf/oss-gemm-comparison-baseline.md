# OSS GEMM 直接比較: 再現手順・計測境界・ベースライン（イシュー #755）

GEMM 第 2 次最適化ツリー（#735。Phase 4 親 #754）の目標「既存ライブラリ（OSS）を
上回る」の進捗を、恒久化したハーネス（`scripts/bench/gemm_bench_mlx_f32.py`）で
再現可能に追跡する。設計判断（ハーネスの置き場所・依存の扱い）は
`docs/oss-comparison-harness-decision.md` を参照。

## 1. 再現手順

### 1.1 CPU（matrixmultiply・gemm crate）— 現状未導入

`matrixmultiply`・`gemm` crate は許容依存 8 区分外であり、ユーザー承認が未取得の
ため本リポジトリへ導入していない（`docs/oss-comparison-harness-decision.md`「経緯」
節）。§6 の 2026-08-19 集約値をベースラインとして記録するに留め、依存追加の
ユーザー承認が得られ次第、再現用ハーネスの設計・実装を別途行う。

### 1.2 Metal（MLX。macOS・Apple Silicon 実機限定）

```sh
python3 -m venv .venv-mlx-bench
source .venv-mlx-bench/bin/activate
pip install mlx numpy
python3 scripts/bench/gemm_bench_mlx_f32.py
```

### 1.3 Metal（PyTorch MPS。既存スクリプトを再利用）

```sh
python3 -m venv .venv-mps-bench
source .venv-mps-bench/bin/activate
pip install torch
python3 scripts/bench/gemm_bench_torch_mps_f32.py
```

venv はリポジトリ管理外（`.venv*/` は `.gitignore` 済み）。実測時の
`mlx`／`torch` バージョンは標準出力の 1 行目（`mlx=... device=...`／
`torch=... device=...`）に記録されるため、キャンペーン表（§7）へ転記する。

### 1.4 Metal（自作実装。既存 Rust example を再利用）

新規 Rust コードは追加しない。既存の `crates/backend-metal/examples/` を実機上で
そのまま実行する:

- デバイス内境界: `cargo run --release -p backend-metal --example gemm_f32_prepared_bench`
- 転送込み境界: `cargo run --release -p backend-metal --example gemm_bench`

## 2. 計測プロトコル

計測系統（Rust Metal・Python MLX/PyTorch。CPU 側は §1.1 参照）とも共通:

- warmup 20 回以上・計測 20 回以上・中央値採用（`bench_harness::protocol::run` の
  下限。Python 側は `time.perf_counter()` + `statistics.median` で同水準を踏襲）
- 決定的シード `0xC0FFEE`（`bench_harness::rng::Xorshift64Star` と同一値。
  Python 側は値域・分布形状を揃えた別実装で近似。`gemm_bench_mlx_f32.py`・
  `gemm_bench_torch_mps_f32.py` 冒頭コメント参照）
- 形状: 正方行列 M=N=K ∈ {512, 1024, 2048, 4096}・dtype f32

## 3. 計測境界の定義と比較可能ペア

CPU は転送区間がそもそも存在しない（ホストメモリ上で完結）ため単一境界
（「デバイス内」に相当）のみ。Metal 側は 2 境界を定義する。

| 境界 | 定義 |
|------|------|
| デバイス内（device-resident / prepared） | 事前に GPU 上へ実体化済みのバッファに対する matmul + 同期完了待ちのみを計測区間に含める。ホスト⇔デバイス転送は計測区間外 |
| 転送込み（transfer-included） | ホスト側データからのアップロード（H2D 相当）・GEMM・結果読み戻し（D2H 相当）を 1 回の計測区間に含める |

MLX はユニファイドメモリ特性により、Rust 側 Metal 実装ほど厳密な物理コピーの
有無では対応しない（近似の詳細は `gemm_bench_mlx_f32.py` 冒頭コメント参照）。

### 比較可能ペア表

| 分子（自作） | 分母（OSS） | 境界 | 直接比較可否 |
|---|---|---|---|
| CPU `gemm_blis_parallel` | `matrixmultiply` (0.3.11) | デバイス内相当（ホスト完結） | 参考値のみ（§6。再現用ハーネス未導入。§1.1 参照） |
| CPU `gemm_blis_parallel` | `gemm` crate (0.19.0, `Parallelism::Rayon(0)`) | デバイス内相当（ホスト完結） | 参考値のみ（§6。再現用ハーネス未導入。§1.1 参照） |
| Metal `gemm_f32_prepared_bench` | MLX デバイス内 | デバイス内 | 可 |
| Metal `gemm_bench`（`dispatch_auto`） | MLX 転送込み | 転送込み | 可（ただし `dispatch_auto` 自体が REQ-8 の同期方式契約〈ホスト転送を伴わない完了待ち〉を満たさない参考値。`docs/perf/gemm-optimization-baseline.md` §2） |
| Metal `gemm_bench`（`dispatch_auto`） | PyTorch MPS f32 | 転送込み | 可（同上・参考値。`gemm_bench_torch_mps_f32.py` 冒頭コメント） |

## 4. CPU 比較の公平性軸（参考記録・2026-08-19 実測時点）

| 実装 | スレッド構成 |
|------|-------------|
| 自作 `gemm_blis_parallel` | rayon 既定スレッド数（`rayon::current_num_threads()`） |
| `matrixmultiply` | 単一スレッド（`threading` feature 非有効化。crates.io 既定 feature 構成をそのまま使用） |
| `gemm` crate | `Parallelism::Rayon(0)`（既定スレッド数・rayon 共有スレッドプール） |

`matrixmultiply` は既定構成では単一スレッドであり、自作・`gemm` crate（いずれも
並列）との比較は「並列実装 vs 単一スレッド実装」であることを比率解釈時に踏まえる。
再現用ハーネス未導入（§1.1）のため、この構成は将来の再導入時の参考記録である。

## 5. 出力突合と既知の限界

`docs/oss-comparison-harness-decision.md`「出力突合とその限界」節を参照。
要点: 統一複合判定（相対誤差 1e-3 未満 または 絶対誤差 1e-5 未満。許容誤差の
値自体は `.claude/rules/coding-rust.md` に従い単独緩和しない）で自作実装を
基準に照合する運用とする。2026-08-19 の使い捨てハーネスによる smoke 実測では
K=1024 以降で複合判定をわずかに超える不一致が観測されたが、原因（縮約順序差
由来の丸め誤差蓄積か、比較対象 OSS crate への引数渡しの誤りか）はハーネス
コード削除により未検証のまま残る。**将来ハーネスを再導入する際は、出力不一致を
既定で fail-closed（非 0 終了）として扱う**（性能計測の継続を理由に正しさの
検証を弱めない。`docs/oss-comparison-harness-decision.md`「将来の再導入条件」）。

## 6. 2026-08-19 ベースライン（出典: #735・#752・#753 本文。scratchpad ハーネス消失により集約値のみ）

使い捨てハーネスによる初回比較（main コミット `cbc16e7` 時点）の確定済み集約値。
per-size 詳細は当時のハーネスがリポジトリに残らなかったため再現できない。
CPU 側（`matrixmultiply`・`gemm` crate）の再現用ハーネスは依存追加のユーザー
承認が未取得のため本リポジトリには存在しない（`docs/oss-comparison-harness-decision.md`
「経緯」節）。Metal 側（MLX・PyTorch MPS）は §7 の再計測キャンペーンで
per-size 詳細表を確定させる。

| 比較 | 集約比率 |
|------|---------|
| CPU 自作 vs `matrixmultiply` | 1.3〜2.6 倍（自作が優位） |
| CPU 自作 vs `gemm` crate | 0.87〜0.95 倍（自作がやや劣位） |
| Metal 自作 vs MLX（転送込み・4096） | 約 1/5（自作 1.57 TFLOPS vs MLX 9.95 TFLOPS） |
| Metal 自作 vs PyTorch MPS | MLX とほぼ同水準（MLX ≈ PyTorch MPS） |

**注記**: 上記 CPU 側 2 行は 2026-08-19 の scratchpad ハーネス（本 PR で削除した
恒久ハーネスとは別実装）による計測値であり、§5 で述べた「出力突合の不一致原因
未検証」の対象（削除済みの恒久ハーネス実装）とは異なる。この集約比率自体を
無効とするものではないが、再現・per-size 詳細化は §1.1 のとおりユーザー承認
取得後に行う。

## 7. 再計測キャンペーン表（Metal・MLX/PyTorch MPS のみ）

各キャンペーンの生 JSON・ログは `docs/perf/oss-comparison/<日付>/` へコミット
する運用とする。CPU 側（`matrixmultiply`・`gemm` crate）はハーネス未導入
（§1.1）のため対象外。実機（Apple Silicon）での第 0 回フル計測は #735 各
Phase 完了時の運用としてこの表へ追記していく。

### 7.1 以降

（次回実機計測時に追記。列: 日付・commit・per-size 比率・前回比差分）
