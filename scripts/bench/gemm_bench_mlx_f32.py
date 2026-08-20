"""MLX f32 GEMM 計測スクリプト（イシュー #755・OSS 直接比較ハーネスの恒久化）。

`crates/backend-metal/examples/gemm_f32_prepared_bench.rs`（「デバイス内」
= prepared 境界）・`crates/backend-metal/examples/gemm_bench.rs`
（`MetalGemm::dispatch_auto`。「転送込み」= H2D + GEMM + D2H 境界）と
同一実機・同一形状（M=N=K=512/1024/2048/4096）・同一決定的シード
（`0xC0FFEE`）で計測する。`scripts/bench/gemm_bench_torch_mps_f32.py`
（既存・PyTorch MPS 版）と同一プロトコル（warmup 20 回・計測 20 回・
`time.perf_counter()` 中央値）・同一出力形式に揃え、CPU vs matrixmultiply
/ gemm crate（`scripts/bench/oss-gemm-compare/`）と対になる Metal 側の
OSS 直接比較を担う（`docs/perf/oss-gemm-comparison-baseline.md` 参照）。

## 2 つの計測境界（MLX のユニファイドメモリ特性に関する注記）

Apple Silicon の MLX はユニファイドメモリ上で動作し、CUDA/discrete GPU の
ような明示的な H2D/D2H コピー API を持たない（`mx.array(...)` はホスト
バッファを指すハンドルを作るのみで、実際の計算配置・移動は遅延評価
グラフの評価時に処理系が決める）。そのため本スクリプトの 2 境界は、
Rust 側 Metal 実装（明示的なコマンドバッファ・バッファ管理を持つ）ほど
厳密な物理コピーの有無では対応しない。「同じ操作の反復回数」という
計測プロトコル上の対応で近似する:

- **デバイス内（prepared 相当）**: `mx.array` 変換をループ外で 1 回だけ
  行い、ループ内は生成済み配列に対する `mx.matmul` + `mx.eval`（完了待ち）
  のみを計測する。`gemm_f32_prepared_bench.rs` の「バッファ確保・A/B
  アップロード済みの状態から計測開始」という区切りに対応する。
- **転送込み（dispatch_auto 相当）**: ループ内で毎回 NumPy 配列から
  `mx.array(...)` を作り直し、結果を `np.array(c)` でホスト側へ読み戻す
  （MLX の遅延評価では `np.array(...)` 変換時に暗黙の同期が走る）まで
  を 1 回の計測区間に含める。`dispatch_auto` の「1 ディスパッチごとに
  A・B アップロードと C readback を含む」境界に対応する。

いずれの境界も MLX 内部でのメモリ移動（統合メモリゆえ物理コピーが発生
しない場合を含む）を含みうるため、Rust 側 Metal 実装との対比は
`docs/perf/oss-gemm-comparison-baseline.md` の計測境界節に記載する注記
付き参考値として扱う（`gemm_bench_torch_mps_f32.py` の f32 dispatch_auto
比較が「計測境界差の注記付き参考値」とされているのと同じ位置づけ）。

### デバイス内境界の残存非対称性（出力確保コスト。レビュー指摘対応。イシュー #755）

`measure_device_resident` の `mx.matmul(a, b)`（下記関数の計測ループ内）は
**呼び出しのたびに新しい出力配列を生成する**（MLX の `mx.array` は immutable
な関数型配列であり、`out=` 相当の「既存バッファへ書き込む」API を提供しない。
`mx.compile` の donation 機構で出力バッファの再利用を狙う経路はあるが実機
未検証のため採用しない）。一方 `crates/backend-metal/examples/
gemm_f32_prepared_bench.rs` は `c_buf` をループ外で 1 回だけ確保し、計測ループ
内では同一バッファへ書き込みを繰り返す（同ファイル 114 行・123〜127 行）。

この非対称性は MLX の言語仕様上の制約でありコード側で対称化できないため、
`docs/perf/oss-gemm-comparison-baseline.md` §3「比較可能ペア表」の当該行を
「直接比較可」ではなく出力確保コストの非対称性を明記した参考値として扱う
よう改めた。MLX 側の測定値には Metal 側に存在しない出力配列確保コストが
含まれるため、MLX 側が不利な非対称であり、自作実装を相対的に有利に見せる
（優位性を誇張しうる）方向に働く。したがって直接比較には使わず、
この非対称性を踏まえた参考値として扱う点を比較解釈時の前提として記録する。

本リポジトリの依存ポリシー（`.claude/rules/deps-policy.md`）は MLX を
許容依存 8 区分に含めない。本スクリプトはリポジトリの `Cargo.toml`／
`Cargo.lock` に影響しない実機側の一時的な計測手段であり、`workspace` の
依存追加ではない（`gemm_bench_torch_mps_f32.py` と同じ位置づけ）。

## 実行前準備（macOS・Apple Silicon 実機）

```sh
python3 -m venv .venv-mlx-bench
source .venv-mlx-bench/bin/activate
pip install mlx numpy
python3 scripts/bench/gemm_bench_mlx_f32.py
```

## 引数の扱い（OWASP A03 観点）

コマンドライン引数は受け取らない（形状・回数は本スクリプト内の定数で
固定し、シェル展開・eval を一切使わない。`.claude/rules/security.md`
「A03 インジェクション」対応）。
"""

import statistics
import time

import mlx.core as mx
import numpy as np

# `crates/backend-metal/examples/gemm_bench.rs::SEED` と同一値
# （PoC-v2 系・既存 bench・`gemm_bench_torch_mps_f32.py` と同じ入力分布に
# 揃える）。
SEED = 0xC0FFEE

# `bench-harness::protocol::run`（TASK-8.1）の下限と同一（warmup 20 回・
# 計測 20 回。`.claude/rules/coding-rust.md`「ベンチは 5 回計測の中央値を
# 採用」の下限を満たす）。
WARMUP_ITERS = 20
MEASURE_ITERS = 20

# `gemm_bench.rs` の正方形状計測対象（既存 f32/f16 スクリプトと同一系列）。
SIZES = [512, 1024, 2048, 4096]


def require_metal_device() -> None:
    """MLX の既定デバイスが GPU（Metal）であることを確認する
    （実機依存の明示化。CPU フォールバックを計測に混入させない）。"""
    device = mx.default_device()
    if device.type != mx.DeviceType.gpu:
        raise SystemExit(
            f"mx.default_device() == {device}: 本スクリプトは Apple Silicon "
            "実機（Metal/GPU backend）でのみ実行できる"
        )


def tflops(size: int, median_secs: float) -> float:
    flops = 2.0 * (size**3)
    return flops / median_secs / 1e12


def make_inputs(size: int, generator: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """`[-1.0, 1.0)` 一様分布の f32 入力を NumPy 側で生成する
    （`bench_harness::rng::Xorshift64Star::next_f32` と分布形状を揃える。
    厳密な PRNG アルゴリズム一致までは求めず、値域・分布形状の整合のみを
    条件とする。既存 `gemm_bench_torch_mps_f32.py` と同じ判断）。"""
    a = generator.random((size, size), dtype=np.float32) * 2.0 - 1.0
    b = generator.random((size, size), dtype=np.float32) * 2.0 - 1.0
    return a, b


def measure_device_resident(size: int, generator: np.random.Generator) -> float:
    """デバイス内境界: `mx.array` 変換をループ外で 1 回だけ行う。"""
    a_np, b_np = make_inputs(size, generator)
    a = mx.array(a_np)
    b = mx.array(b_np)
    mx.eval(a, b)

    for _ in range(WARMUP_ITERS):
        c = mx.matmul(a, b)
        mx.eval(c)

    secs: list[float] = []
    for _ in range(MEASURE_ITERS):
        start = time.perf_counter()
        c = mx.matmul(a, b)
        mx.eval(c)
        secs.append(time.perf_counter() - start)

    median_secs = statistics.median(secs)
    return tflops(size, median_secs)


def measure_transfer_included(size: int, generator: np.random.Generator) -> float:
    """転送込み境界: 毎回 NumPy → `mx.array` 変換と `np.array(...)` による
    ホスト読み戻しを計測区間に含める（`dispatch_auto` の「1 ディスパッチ
    ごとに A・B アップロードと C readback を含む」境界に対応。ファイル
    冒頭コメント参照）。"""
    a_np, b_np = make_inputs(size, generator)

    for _ in range(WARMUP_ITERS):
        a = mx.array(a_np)
        b = mx.array(b_np)
        c = mx.matmul(a, b)
        _ = np.array(c)

    secs: list[float] = []
    for _ in range(MEASURE_ITERS):
        start = time.perf_counter()
        a = mx.array(a_np)
        b = mx.array(b_np)
        c = mx.matmul(a, b)
        _ = np.array(c)
        secs.append(time.perf_counter() - start)

    median_secs = statistics.median(secs)
    return tflops(size, median_secs)


def main() -> None:
    require_metal_device()
    print(f"mlx={mx.__version__} device={mx.default_device()}")
    generator = np.random.default_rng(SEED)
    for size in SIZES:
        device_resident = measure_device_resident(size, generator)
        transfer_included = measure_transfer_included(size, generator)
        print(
            f"size={size} "
            f"mlx_f32_device_resident_tflops={device_resident:.4f} "
            f"mlx_f32_transfer_included_tflops={transfer_included:.4f}"
        )


if __name__ == "__main__":
    main()
