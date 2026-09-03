"""PyTorch CPU f32 GEMM 計測スクリプト（イシュー #1141・
`gemm_bench_torch_mps_f32.py` の CPU 版）。

CPU GEMM 再チューニング候補（`SharedBPcOuter` 等。イシュー #1041）の
M4 Max 実機実測（`docs/perf/cpu-gemm-candle-cpu-retune.md` §5 手順 4・
副次目標）向けに、対 PyTorch CPU 比の参考値を得るためのスクリプト。
プロトコル（warmup 20 回・計測 20 回・`time.perf_counter()` 中央値）は
MPS 版を踏襲するが、CPU 側は GPU のような明示的な同期呼び出し
（`torch.mps.synchronize()` 相当）が不要なため `torch.mm` 呼び出し自体が
計測区間の終端となる。

本リポジトリの依存ポリシー（`.claude/rules/deps-policy.md`）は PyTorch を
許容依存 8 区分に含めない。本スクリプトはリポジトリの `Cargo.toml`／
`Cargo.lock` に影響しない実機側の一時的な計測手段であり、`workspace` の
依存追加ではない（`gemm_bench_torch_mps_f32.py`・`gemm_bench_torch_mps_f16.py`
と同じ位置づけ）。

## 実行前準備（macOS・Apple Silicon 実機。venv はリポジトリ管理外・
`.gitignore` 対象）

```sh
python3 -m venv .venv-mps-bench
source .venv-mps-bench/bin/activate
pip install torch  # Accelerate/AMX backend を含む標準ビルド
python3 scripts/bench/gemm_bench_torch_cpu_f32.py
```

## 引数の扱い（OWASP A03 観点）

コマンドライン引数は受け取らない（形状・回数は本スクリプト内の定数で
固定し、シェル展開・eval を一切使わない。`.claude/rules/security.md`
「A03 インジェクション」対応。`gemm_bench_torch_mps_f32.py` と同方針）。
"""

import statistics
import time

import torch

# `gemm_bench_torch_mps_f32.py::SEED` と同一値（入力分布を揃える）。
SEED = 0xC0FFEE

# `bench-harness::protocol::run`（TASK-8.1）の下限と同一（warmup 20 回・
# 計測 20 回。`.claude/rules/coding-rust.md`「ベンチは 5 回計測の中央値を
# 採用」の下限を満たす）。
WARMUP_ITERS = 20
MEASURE_ITERS = 20

# `docs/perf/cpu-gemm-candle-cpu-retune.md` §5 記入表と同一形状
# （1024/2048/4096）に加え、既存 OSS 比較ハーネスの対象下限 512 も含める。
SIZES = [512, 1024, 2048, 4096]


def tflops(size: int, median_secs: float) -> float:
    flops = 2.0 * (size**3)
    return flops / median_secs / 1e12


def measure(device: torch.device, size: int) -> float:
    """`size x size x size` の f32 GEMM を CPU 上で計測し、中央値 TFLOPS
    を返す（`gemm_bench_torch_mps_f32.py::measure` と同一構成。CPU 経路は
    `torch.mm` 呼び出し自体が同期的に完了するため明示的な同期呼び出しは
    不要）。

    出力テンソル `c_buf` はループ外で事前確保し、`torch.mm(..., out=c_buf)`
    でループ内は書き込み先を使い回す（MPS 版と同一方針: 2 次元行列積に
    特化した `torch.mm` を使うことで `out=` カーネル登録の不確実性を
    避ける）。"""
    generator = torch.Generator().manual_seed(SEED)
    a = torch.rand((size, size), generator=generator, dtype=torch.float32).to(
        device=device
    )
    b = torch.rand((size, size), generator=generator, dtype=torch.float32).to(
        device=device
    )
    c_buf = torch.empty((size, size), device=device, dtype=torch.float32)

    for _ in range(WARMUP_ITERS):
        torch.mm(a, b, out=c_buf)

    secs: list[float] = []
    for _ in range(MEASURE_ITERS):
        start = time.perf_counter()
        torch.mm(a, b, out=c_buf)
        secs.append(time.perf_counter() - start)

    median_secs = statistics.median(secs)
    return tflops(size, median_secs)


def main() -> None:
    device = torch.device("cpu")
    print(f"torch={torch.__version__} device={device}")
    print(f"torch.get_num_threads()={torch.get_num_threads()}")
    # Accelerate/AMX 等の CPU 実行系（BLAS backend）の到達性根拠を残す
    # （`docs/perf/cpu-gemm-candle-cpu-retune.md` 記入時の参考情報）。
    print("torch.__config__.parallel_info():")
    print(torch.__config__.parallel_info())
    for size in SIZES:
        result = measure(device, size)
        print(f"size={size} pytorch_cpu_f32_tflops={result:.4f}")


if __name__ == "__main__":
    main()
