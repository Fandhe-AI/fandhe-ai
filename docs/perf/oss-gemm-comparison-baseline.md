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

- デバイス内境界: `cargo run --release -p fandhe-ai-backend-metal --example gemm_f32_prepared_bench`
- 転送込み境界: `cargo run --release -p fandhe-ai-backend-metal --example gemm_bench`

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

各キャンペーンの生 JSON は `docs/perf/oss-comparison/<日付>/` へコミットする。
smoke run（#755 実装時・Linux x86_64）のログは本節 7.1 に要約として記録し、
実機での第 0 回フル計測は §7.2（2026-08-23・生データコミット済み）で実施した。
以後 #735 各 Phase 完了時の運用としてこの表へ追記していく。

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
  での第 0 回フル計測は §7.2（2026-08-23）で実施済み

### 7.2 第 0 回実機フル計測（2026-08-23・Apple M4 Max・commit `7cc5f3d`）

恒久ハーネスによる初の実機（Apple Silicon・aarch64 NEON 最適経路）フル計測。
生データは `docs/perf/oss-comparison/2026-08-23/` に保存
（`env.txt`・`cpu-oss-compare.jsonl`・Metal/MLX/PyTorch 各系統の標準出力ログ）。
なお同ディレクトリ直下の `metal-self-f16.txt`・`torch-mps-f16.txt` は本キャンペーン中の
**探索的な単発実行**（1 プロセス実行）であり、f16 の正式値としては引用しない。
f16 の正規結果（#799 プロトコル: parity 先行確認 + 5 回独立実行）はサブディレクトリ
`f16-799/` と §7.3 の #799 枠（正は `docs/perf/metal-f16-vs-mps-f16.md`）を正とする。

- 環境: Apple M4 Max（16 論理コア・GPU 40 コア・64 GiB）・macOS 26.6.2・
  rustc 1.96.0・MLX 0.32.1（device=gpu）・PyTorch 2.13.0（device=mps）
- プロトコル: §2 のとおり（warmup 20 回以上・計測 20 回以上・中央値・シード `0xC0FFEE`）

#### CPU（自作 `gemm_blis_parallel` vs `matrixmultiply` 0.3.11 vs `gemm` crate 0.19.0）

| size | 自作 TFLOPS | matrixmultiply | 自作/mm 比 | gemm | 自作/gemm 比 | output_match |
|---|---|---|---|---|---|---|
| 512 | 0.3990 | 0.1092 | 3.65 倍 | 0.4725 | 0.84 倍 | 全実装 pass |
| 1024 | 0.7875 | 0.1135 | 6.94 倍 | 0.8609 | 0.91 倍 | mm のみ NG |
| 2048 | 0.8941 | 0.1122 | 7.97 倍 | 1.0031 | 0.89 倍 | mm のみ NG |
| 4096 | 1.1025 | 0.1108 | 9.95 倍 | 1.0966 | 1.01 倍 | mm・gemm とも NG |

- スレッド構成は §4 のとおり（自作・gemm は 16 スレッド・matrixmultiply は単一スレッド）。
  対 matrixmultiply 比は「並列 vs 単一スレッド」の参考値
- 出力突合 NG は K≧1024 の既知の丸め誤差蓄積（§5）。最大 rel_diff 0.00465
  （matrixmultiply・1024）・gemm crate は 4096 のみ NG（rel_diff 0.00356）。
  既定どおり fail-closed の非 0 終了（`cpu-oss-compare.exit` に `exit=1` を記録）
- 前回比（§6 集約値との対比）: 対 `gemm` crate は 0.87〜0.95 倍 → 0.84〜1.01 倍
  （4096 でパリティ到達）。対 matrixmultiply は §6 当時（1.3〜2.6 倍）と比較して
  大幅に拡大したが、§6 は使い捨てハーネス・別スレッド構成の可能性があり
  直接比較不能（本キャンペーンを以後の基準とする）

#### Metal f32（自作 vs MLX vs PyTorch MPS）

| size | 自作 prepared | 自作 転送込み* | MLX device | MLX 転送込み | MPS f32 |
|---|---|---|---|---|---|
| 512 | 0.8362 | 0.6831 | 1.3184 | 1.1339 | 0.8822 |
| 1024 | 1.7916 | 1.8656 | 5.2721 | 3.8437 | 6.9894 |
| 2048 | 8.3602 | 3.2055 | 11.2772 | 7.8093 | 11.3685 |
| 4096 | 7.2331 | 4.1640 | 12.7778 | 9.6404 | 13.0534 |

\* 転送込みは `gemm_bench` の自動選択カーネル（dynamic_tile_auto）系列。

比較可能ペアの比率（自作 ÷ 参照。§3 の比較可否・参考値の位置づけに従う）:

| size | prepared ÷ MLX device（参考値・§3.1） | 転送込み ÷ MLX 転送込み | 転送込み ÷ MPS f32 |
|---|---|---|---|
| 512 | 0.63 | 0.60 | 0.77 |
| 1024 | 0.34 | 0.49 | 0.27 |
| 2048 | 0.74 | 0.41 | 0.28 |
| 4096 | 0.57 | 0.43 | 0.32 |

- 前回比（§6）: 対 MLX 転送込み・4096 は約 1/5（1.57 対 9.95 TFLOPS）→ 0.43 倍
  （4.16 対 9.64 TFLOPS）。自作側の絶対値が約 2.7 倍改善したが全サイズで劣後継続

#### Metal f16

対 PyTorch MPS f16 の per-size 詳細は §7.3 の #799 枠（正は
`docs/perf/metal-f16-vs-mps-f16.md`「タイル化後再計測」節）に記録する。

### 7.3 以降

#### Metal f16（対 PyTorch MPS f16）タイル化後再計測（イシュー #799・状態: 消化済み・2026-08-23）

GEMM OSS 比較ギャップ改修ツリー（#785）Phase 2 完了（#796〜#798。非タイル `gemm_simdgroup_f16` →
タイル化 `gemm_simdgroup_tiled_f16` への世代更新）を受けた、対 PyTorch MPS f16 比の追加キャンペーン枠。
本節が対象とする実測手順・per-size 結果表は `docs/perf/metal-f16-vs-mps-f16.md`「タイル化後再計測
（イシュー #799）」節が正であり、本節では二重管理せず状態のみを記録する。

- 直近の確定値（旧経路・#785 本文・2026-08-21 再計測）: size=4096 で **18.5%**（2.27 対 12.26 TFLOPS）
- 本イシュー（#799）着手時点: Linux dev-box（本イシュー実装セッション）から Metal 実機（M4 Max）への
  到達手段が `docs/real-hardware-verification-env.md`／`docs/real-hardware-verification-env.local.md`
  のいずれにも存在せず、**実機未到達のため per-size 詳細・比率・前回比差分は未記入**（同型の #795・#814・
  #818・#821 と同じ先例に従い、実測線・計測手順の整備のみ完了させ実測は Mac 実機セッションへ引き継ぐ。
  推定・外挿・捏造は行わない）
- 後続消化時の追記先: 本節（日付・commit・per-size 比率・前回比差分）と
  `docs/perf/metal-f16-vs-mps-f16.md`「タイル化後再計測」節（詳細結果表）の両方

**消化記録（2026-08-23・Apple M4 Max・commit `7cc5f3d`）**: 正規プロトコル
（parity 3 系統 PASS 確認・プロセス単位 5 回独立実行の中央値）で実測完了。
詳細（per-size 5 回生値・中央値/Q1/Q3・resolved tile 構成・計測衛生）は
`docs/perf/metal-f16-vs-mps-f16.md`「実測結果（イシュー #799・2026-08-23）」節・
生データは `docs/perf/oss-comparison/2026-08-23/f16-799/` を正とする。

| 日付 | commit | per-size 比率（新経路 tiled ÷ MPS f16） | 前回比差分 |
|------|--------|------|------|
| 2026-08-23 | `7cc5f3d` | 512: 130.91%・1024: 101.65%・2048: 80.00%・4096: **63.90%** | 旧経路 4096 比 18.5%（#785）→ 63.90%（+45.4 pt・約 3.6 倍）。同セッション実測の旧経路 17.51% は 18.5% と同水準（再現性確認） |

#### CUDA f32/f16（対 PyTorch CUDA）Phase 3/4 完了後再計測（イシュー #807・状態: 実機セッション待ち）

GEMM OSS 比較ギャップ改修ツリー（#785）Phase 4 完了（親 #789「CUDA タイル形状拡大」。依存 #804・#806 は
CLOSED だが本番カーネル定数は未変更）を受けた、対 PyTorch CUDA 比の確定計測枠。本節が対象とする実測
手順・per-size 結果表は `docs/perf/cuda-phase34-remeasurement.md`（#807）が正であり、本節では二重管理
せず状態のみを記録する。

- 直近の確定値（Phase B/C 適用後・#571・2026-08-18 実測）: 判定対象形状の対 PyTorch 比最小値
  f32=51.96%（4096）・f16=37.47%（4096）
- 本イシュー（#807）着手時点: Linux dev-box（本イシュー実装セッション）から CUDA 実機（DGX Spark GB10）
  への到達手段が `docs/real-hardware-verification-env.md`／`docs/real-hardware-verification-env.local.md`
  のいずれにも存在せず、**実機未到達のため per-size 詳細・比率・前回比差分は未記入**（同型の #502・
  #571・#572・#799・#803・#804・#806 と同じ先例に従い、実測線・計測手順の整備のみ完了させ実測は
  DGX Spark GB10 実機セッションへ引き継ぐ。推定・外挿・捏造は行わない）
- #804/#806（Phase 4 の依存イシュー）はいずれも「診断機構・机上候補表の整備」までで CLOSED しており、
  本番カーネル定数（ブロックタイル・ステージ数）は変更されていない。本イシューの実測対象は main HEAD
  の本番経路そのままである（`docs/perf/cuda-phase34-remeasurement.md` §1 参照）
- 後続消化時の追記先: 本節（日付・commit・per-size 比率・前回比差分）と
  `docs/perf/cuda-phase34-remeasurement.md`（詳細結果表）の両方

（次回実機計測時に追記。列: 日付・commit・per-size 比率・前回比差分）
