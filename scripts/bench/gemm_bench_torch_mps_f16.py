"""PyTorch MPS f16 GEMM ベースライン計測スクリプト（TASK-8.3b・#156）。

REQ-8 性能下限表「Metal f16 対 PyTorch MPS f16」の分母（同一実機上で
PyTorch とのみ比較する方針。REQ-8 v2）を計測する。
`crates/backend-metal/examples/gemm_f16_bench.rs`（Rust 側 `gemm_simdgroup_f16`
実測）と同一実機・同一形状（M=N=K=512/1024/2048/4096）・同一決定的シード
（`0xC0FFEE`）・同一同期境界（ホスト転送を伴わない完了待ち。Rust 側は
コマンドバッファ完了待ち、本スクリプトは `torch.mps.synchronize()`）で
計測し、実測値は `docs/perf/metal-f16-vs-mps-f16.md` へ転記する運用とする。

本リポジトリの依存ポリシー（`.claude/rules/deps-policy.md`）は PyTorch を
許容依存 8 区分に含めない。本スクリプトはリポジトリの `Cargo.toml`／
`Cargo.lock` に影響しない実機側の一時的な計測手段であり、`workspace` の
依存追加ではない（PoC-v2-4 の `code/pytorch/gemm_bench_torch_mps.py` と
同じ位置づけ。実装計画 §3.6 の判断）。

## 実行前準備（macOS・Apple Silicon 実機）

```sh
python3 -m venv .venv-mps-bench
source .venv-mps-bench/bin/activate
pip install torch  # MPS 対応版（公式ビルドは標準で MPS backend を含む）
python3 scripts/bench/gemm_bench_torch_mps_f16.py
```

## 引数の扱い（OWASP A03 観点）

コマンドライン引数は受け取らない（形状・回数は本スクリプト内の定数で
固定し、シェル展開・eval を一切使わない。`.claude/rules/security.md`
「A03 インジェクション」対応）。
"""

import statistics
import time

import torch

# `crates/backend-metal/examples/gemm_f16_bench.rs::SEED` と同一値
# （PoC-v2 系・既存 bench と同じ入力分布に揃える）。
SEED = 0xC0FFEE

# `bench-harness::protocol::run`（TASK-8.1）の下限と同一（warmup 20 回・
# 計測 20 回。`.claude/rules/coding-rust.md`「ベンチは 5 回計測の中央値を
# 採用」の下限を満たす）。
WARMUP_ITERS = 20
MEASURE_ITERS = 20

# REQ-8 の主指標は 2048/4096（512 は起動オーバーヘッド支配のため参考値。
# PoC-v2-4 先例。`gemm_f16_bench.rs` と同一形状）。
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
    """`size x size x size` の f16 GEMM を計測し、中央値 TFLOPS を返す
    （`bench-harness::protocol::run` と同じ「warmup → 計測 → 中央値」構成。
    同期境界はホスト転送を伴わない `torch.mps.synchronize()`。ファイル
    冒頭コメント参照）。

    出力テンソル `c_buf` はループ外で事前確保し、`torch.mm(..., out=c_buf)`
    でループ内は書き込み先を使い回す。Rust 側（`gemm_f16_bench.rs`）は
    `c_buf` をループ外で確保しディスパッチと完了待ちのみを計測するため、
    PyTorch 側で `torch.matmul(a, b)` の戻り値生成（毎回の新規テンソル
    割り当て）を計測に含めると同一同期境界の契約が崩れ、Metal/PyTorch 比が
    不当に高く出て REQ-8 の下限決定を誤らせる（#346 codex-review 指摘）。
    `torch.matmul` ではなく 2 次元専用の `torch.mm` を使う理由: 本スクリプトの
    入力 `a`／`b` は常に正方 2 次元行列であり、`torch.matmul` の `out=` は
    次元・ブロードキャストの組み合わせによって MPS backend 側の `out=`
    カーネル登録が未整備な場合がある（PyTorch のバージョンに依存する既知の
    制約）のに対し、`torch.mm` は 2 次元行列積に特化し `out=` 対応が
    より安定している。"""
    generator = torch.Generator().manual_seed(SEED)
    a = torch.rand((size, size), generator=generator, dtype=torch.float32).to(
        device=device, dtype=torch.float16
    )
    b = torch.rand((size, size), generator=generator, dtype=torch.float32).to(
        device=device, dtype=torch.float16
    )
    c_buf = torch.empty((size, size), device=device, dtype=torch.float16)

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
        print(f"size={size} pytorch_mps_f16_tflops={result:.4f}")


if __name__ == "__main__":
    main()
