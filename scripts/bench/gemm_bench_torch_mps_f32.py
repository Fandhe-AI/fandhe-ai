"""PyTorch MPS f32 GEMM 計測スクリプト（#547・Phase D 完了時点再計測）。

`crates/backend-metal/examples/gemm_bench.rs`（`MetalGemm::dispatch_auto`。
1 ディスパッチごとに A・B アップロードと C readback を含む「転送込み」境界。
`docs/perf/gemm-optimization-baseline.md` §2 系列 (b)）と同一実機・同一形状
（M=N=K=512/1024/2048/4096）・同一決定的シード（`0xC0FFEE`）で計測し、
実測値は `docs/perf/metal-gemm-dynamic-tile.md`「Phase D 完了時点再計測
（#547）」節へ転記する運用とする。

`scripts/bench/gemm_bench_torch_mps_f16.py`（#156・TASK-8.3b）の f32 版。
プロトコル（warmup 20 回・計測 20 回・`time.perf_counter()` 中央値・
`torch.mps.synchronize()` 同期）はそのまま踏襲する。**f32 側は
`dispatch_auto` が転送込み境界のため、`docs/performance-targets.md` §4 の
同期方式契約（ホスト転送を伴わない完了待ち）を単独では満たさない。よって
本スクリプトによる対 MPS f32 比は計測境界差の注記付き参考値とし、REQ-8 の
分母・分子には使わない**（`docs/perf/gemm-optimization-baseline.md` §2
「基準系列の決定」参照。§4 準拠の f32 prepared 入口整備・確定計測は
Phase F の #572 のスコープ）。

本リポジトリの依存ポリシー（`.claude/rules/deps-policy.md`）は PyTorch を
許容依存 8 区分に含めない。本スクリプトはリポジトリの `Cargo.toml`／
`Cargo.lock` に影響しない実機側の一時的な計測手段であり、`workspace` の
依存追加ではない（`gemm_bench_torch_mps_f16.py` と同じ位置づけ）。

## 実行前準備（macOS・Apple Silicon 実機）

```sh
python3 -m venv .venv-mps-bench
source .venv-mps-bench/bin/activate
pip install torch  # MPS 対応版（公式ビルドは標準で MPS backend を含む）
python3 scripts/bench/gemm_bench_torch_mps_f32.py
```

## 引数の扱い（OWASP A03 観点）

コマンドライン引数は受け取らない（形状・回数は本スクリプト内の定数で
固定し、シェル展開・eval を一切使わない。`.claude/rules/security.md`
「A03 インジェクション」対応）。
"""

import statistics
import time

import torch

# `crates/backend-metal/examples/gemm_bench.rs::SEED` と同一値
# （PoC-v2 系・既存 bench・`gemm_bench_torch_mps_f16.py` と同じ入力分布に
# 揃える）。
SEED = 0xC0FFEE

# `bench-harness::protocol::run`（TASK-8.1）の下限と同一（warmup 20 回・
# 計測 20 回。`.claude/rules/coding-rust.md`「ベンチは 5 回計測の中央値を
# 採用」の下限を満たす）。
WARMUP_ITERS = 20
MEASURE_ITERS = 20

# `gemm_bench.rs` の正方形状計測対象（256 は除く。REQ-8 主指標は
# 2048/4096・512/1024 は参考値。PoC-v2-4 先例・`gemm_f16_bench.rs` と
# 同一形状に揃える）。
SIZES = [512, 1024, 2048, 4096]


def require_mps() -> torch.device:
    """MPS backend が利用可能でなければ即座に終了する（実機依存の明示化。
    フォールバックして CPU 計測を混入させない）。"""
    if not torch.backends.mps.is_available():
        raise SystemExit(
            "torch.backends.mps.is_available() == False: "
            "本スクリプトは Apple Silicon 実機（MPS backend）でのみ実行できる"
        )
    return torch.device("mps")


def tflops(size: int, median_secs: float) -> float:
    flops = 2.0 * (size**3)
    return flops / median_secs / 1e12


def measure(device: torch.device, size: int) -> float:
    """`size x size x size` の f32 GEMM を計測し、中央値 TFLOPS を返す
    （`bench-harness::protocol::run` と同じ「warmup → 計測 → 中央値」構成。
    同期境界はホスト転送を伴わない `torch.mps.synchronize()`。ファイル
    冒頭コメント参照）。

    出力テンソル `c_buf` はループ外で事前確保し、`torch.mm(..., out=c_buf)`
    でループ内は書き込み先を使い回す（`gemm_bench_torch_mps_f16.py::measure`
    と同一方針・同一理由: `torch.matmul` の `out=` は次元・ブロードキャスト
    の組み合わせによって MPS backend 側の `out=` カーネル登録が未整備な
    場合があるため、2 次元行列積に特化した `torch.mm` を使う）。"""
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
    torch.mps.synchronize()

    secs: list[float] = []
    for _ in range(MEASURE_ITERS):
        start = time.perf_counter()
        torch.mm(a, b, out=c_buf)
        torch.mps.synchronize()
        secs.append(time.perf_counter() - start)

    median_secs = statistics.median(secs)
    return tflops(size, median_secs)


def main() -> None:
    device = require_mps()
    print(f"torch={torch.__version__} device={device}")
    for size in SIZES:
        result = measure(device, size)
        print(f"size={size} pytorch_mps_f32_tflops={result:.4f}")


if __name__ == "__main__":
    main()
