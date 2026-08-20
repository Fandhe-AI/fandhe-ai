# OSS GEMM 直接比較: 再現手順・計測境界・ベースライン（イシュー #755）

GEMM 第 2 次最適化ツリー（#735。Phase 4 親 #754）の目標「既存ライブラリ（OSS）を
上回る」の進捗を、恒久化したハーネス（`scripts/bench/oss-gemm-compare/`・
`scripts/bench/gemm_bench_mlx_f32.py`）で再現可能に追跡する。設計判断（ハーネスの
置き場所・依存の扱い）は `docs/oss-comparison-harness-decision.md` を参照。

## 1. 再現手順

### 1.1 CPU（matrixmultiply・gemm crate。Linux/macOS 共通）

```sh
cd scripts/bench/oss-gemm-compare
cargo build --release
./target/release/oss-gemm-compare > result.jsonl
```

引数なしで既定サイズ（512/1024/2048/4096）を計測する。`--sizes 512,1024` で
サイズを明示指定できる（正整数のカンマ区切りのみ受理）。標準出力の JSON Lines を
`docs/perf/oss-comparison/<日付>/` へ保存する運用とする（§4 参照）。

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
`torch=... device=...`）に記録されるため、キャンペーン表（§5）へ転記する。

### 1.4 Metal（自作実装。既存 Rust example を再利用）

新規 Rust コードは追加しない。既存の `crates/backend-metal/examples/` を実機上で
そのまま実行する:

- デバイス内境界: `cargo run --release -p backend-metal --example gemm_f32_prepared_bench`
- 転送込み境界: `cargo run --release -p backend-metal --example gemm_bench`

## 2. 計測プロトコル

3 系統（Rust CPU/Metal・Python MLX/PyTorch）とも共通:

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
| CPU `gemm_blis_parallel` | `matrixmultiply` (0.3.11) | デバイス内相当（ホスト完結） | 可（同一プロトコル・同一入力） |
| CPU `gemm_blis_parallel` | `gemm` crate (0.19.0, `Parallelism::Rayon(0)`) | デバイス内相当（ホスト完結） | 可（同一プロトコル・同一入力） |
| Metal `gemm_f32_prepared_bench` | MLX デバイス内 | デバイス内 | 参考値（出力確保コストの非対称性あり。§3.1 参照） |
| Metal `gemm_bench`（`dispatch_auto`） | MLX 転送込み | 転送込み | 可（ただし `dispatch_auto` 自体が REQ-8 の同期方式契約〈ホスト転送を伴わない完了待ち〉を満たさない参考値。`docs/perf/gemm-optimization-baseline.md` §2） |
| Metal `gemm_bench`（`dispatch_auto`） | PyTorch MPS f32 | 転送込み | 可（同上・参考値。`gemm_bench_torch_mps_f32.py` 冒頭コメント） |

### 3.1 デバイス内境界（Metal ↔ MLX）の出力確保コスト非対称性（レビュー指摘対応。イシュー #755）

`gemm_f32_prepared_bench.rs` は出力バッファ `c_buf` をループ外で 1 回だけ確保し、
計測ループ内では同一バッファへの書き込みを繰り返す（同ファイル 114 行・
123〜127 行）。一方 MLX の `mx.matmul(a, b)`（`gemm_bench_mlx_f32.py::
measure_device_resident`）は immutable な関数型配列を返す言語仕様上、
呼び出しのたびに新しい出力配列を確保する（`out=` 相当の書き込み先指定 API を
持たない）。この非対称性は MLX 側の言語仕様上の制約でありコード側で対称化
できないため、上表では「可」ではなく「参考値」とし、非対称の方向
（MLX 側にのみ出力確保コストが乗る＝MLX 側が不利になり、自作実装を
相対的に有利に見せる方向へ働く）を明記し、直接比較には使わない前提を
記録する。詳細は `gemm_bench_mlx_f32.py` 冒頭コメント
「デバイス内境界の残存非対称性」節を参照。

## 4. スレッド構成（CPU 比較の公平性軸）

| 実装 | スレッド構成 |
|------|-------------|
| 自作 `gemm_blis_parallel` | rayon 既定スレッド数（`rayon::current_num_threads()`） |
| `matrixmultiply` | 単一スレッド（`threading` feature 非有効化。crates.io 既定 feature 構成をそのまま使用） |
| `gemm` crate | `Parallelism::Rayon(0)`（既定スレッド数・rayon 共有スレッドプール） |

各実装が実際に使用したスレッド数は JSON Lines の `impl_threads` フィールドに
記録される。`matrixmultiply` は既定構成では単一スレッドであり、自作・
`gemm` crate（いずれも並列）との比較は「並列実装 vs 単一スレッド実装」である
ことを比率解釈時に踏まえる。

## 5. 出力突合と既知の限界

`docs/oss-comparison-harness-decision.md`「出力突合とその限界」節を参照。
要点: 統一複合判定（相対誤差 1e-3 未満 または 絶対誤差 1e-5 未満。許容誤差の
値自体は `.claude/rules/coding-rust.md` に従い単独緩和しない）で自作実装を
基準に照合する。K=1024 以降で複合判定をわずかに超える不一致が観測される
既知の実測結果があり、実装バグではなく縮約順序差由来の丸め誤差蓄積と
判断している。**既定で fail-closed**（レビュー指摘対応。イシュー #755）:
全サイズの JSON Lines 出力（`output_match`／`mismatch_detail` フィールドを
含む）を終えたうえで、突合 NG を 1 件でも検出していれば非 0 終了する。
既知の限界（K=1024 以降の丸め誤差蓄積）を理由に既定挙動を非 fatal へ戻す
ことはしない。

## 6. 2026-08-19 ベースライン（出典: #735・#752・#753 本文。scratchpad ハーネス消失により集約値のみ）

使い捨てハーネスによる初回比較（main コミット `cbc16e7` 時点）の確定済み集約値。
per-size 詳細は当時のハーネスがリポジトリに残らなかったため再現できない。
恒久ハーネスによる初回フル計測（第 0 回再計測。§7）で per-size 詳細表を確定させる。

| 比較 | 集約比率 |
|------|---------|
| CPU 自作 vs `matrixmultiply` | 1.3〜2.6 倍（自作が優位） |
| CPU 自作 vs `gemm` crate | 0.87〜0.95 倍（自作がやや劣位） |
| Metal 自作 vs MLX（転送込み・4096） | 約 1/5（自作 1.57 TFLOPS vs MLX 9.95 TFLOPS） |
| Metal 自作 vs PyTorch MPS | MLX とほぼ同水準（MLX ≈ PyTorch MPS） |

## 7. 再計測キャンペーン表

各キャンペーンの生 JSON は `docs/perf/oss-comparison/<日付>/` へコミットする
（本イシュー実装時点では実機フル計測を実施しておらず、Linux x86_64 環境での
ワイヤリング確認（smoke run）のみ実施したため、本節に生 JSON はまだ存在しない。
smoke run のログは本節 7.1 に要約として記録し、実機での第 0 回フル計測は
#735 各 Phase 完了時の運用としてこの表へ追記していく）。

### 7.1 実装時 smoke run（2026-08-20・Linux x86_64・非対象アーキテクチャ）

**目的**: ハーネスのビルド・実行・JSON Lines 出力・突合ロジックの動作確認のみ。
本体 CPU バックエンドの実プロダクション対象は aarch64 NEON
（`crates/backend-cpu/src/gemm_blis/` の NEON 系ファイル群参照）であり、
本 smoke run は x86_64 上での実行のため **TFLOPS 数値そのものはベースライン
としては採用しない**（環境: 12 論理コア・commit `5cdc471`）。

- サイズ 64/128/256: 3 実装（`self_gemm_blis_parallel`・`matrixmultiply`・
  `gemm`）とも出力突合 pass（統一複合判定を満たす）
- サイズ 512〜4096: ビルド・計測・JSON Lines 出力は正常動作。出力突合は
  `matrixmultiply`・`gemm` crate 双方で K≧1024 において複合判定をわずかに
  超える不一致を検出した（§5「出力突合と既知の限界」参照。ハーネス自体の
  不具合ではなく実測結果）。`output_match=false` を JSON Lines に記録した
  うえで、全サイズの出力完了後に非 0 終了する既定の fail-closed 挙動を
  確認した（レビュー指摘対応。イシュー #755）
- 結論: ハーネスの配線（ビルド・計測・JSON 出力・突合判定）は
  意図どおり機能することを確認した。実機（Apple Silicon・NEON 最適経路）
  での第 0 回フル計測は次回実機アクセス時に実施し、本表へ per-size 詳細
  （JSON パス・比率・前回比差分）を追記する

### 7.2 以降

（次回実機計測時に追記。列: 日付・commit・per-size 比率・前回比差分）
