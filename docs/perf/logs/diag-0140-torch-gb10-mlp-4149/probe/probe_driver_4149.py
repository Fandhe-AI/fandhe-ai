#!/usr/bin/env python3
"""観測専用 shim（probe_shim_4149.c）の呼び出し回数を、MLP の 1 step ごとに決定的に読む駆動スクリプト（#4149）。

役割: LD_PRELOAD で shim を載せた状態で、MLP（batch 64・784→256→10）の train 1 step／infer 1 回を実行し、
step 前後の GEMM 入口の呼び出し回数の差・転送先ライブラリ・先頭の (m,n,k) を JSON 1 行で標準出力へ出す。
shim なしで起動した場合は checksum だけを出す（shim 有り／無しで結果が変わらないことの確認用）。
時間は測らない（計時腕とは別プロセス）。呼び出し元: probe/run_probe.sh。
形状と乱数は固定（torch.manual_seed）で、bench_py.py には依存しない（計測外の経路確認のため）。
shim のパスは run_probe.sh が mktemp 領域に作った固定名のものを引数で受け取り、ctypes で開くだけ（シェルに渡さない）。
"""
import argparse
import ctypes
import json
import sys

NAMES = ("sgemm_", "cblas_sgemm", "cublasSgemm_v2", "cublasGemmEx", "cublasLtMatmul")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", choices=("cpu", "cuda"), required=True)
    ap.add_argument("--op", choices=("train", "infer", "import-only"), required=True)
    ap.add_argument("--shim", default=None)
    args = ap.parse_args()
    import torch
    shim = None
    if args.shim:
        shim = ctypes.CDLL(args.shim)  # LD_PRELOAD 済みなら同じライブラリのハンドルが返る
        shim.fandhe_probe_count.restype = ctypes.c_ulong
        shim.fandhe_probe_count.argtypes = [ctypes.c_int]
        shim.fandhe_probe_next_lib.restype = ctypes.c_char_p
        shim.fandhe_probe_next_lib.argtypes = [ctypes.c_int]
        shim.fandhe_probe_first.restype = ctypes.c_int
        shim.fandhe_probe_first.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int]
        shim.fandhe_probe_reset_first.restype = None
        shim.fandhe_probe_reset_first.argtypes = []

    def counts():
        return [int(shim.fandhe_probe_count(i)) for i in range(len(NAMES))] if shim else None

    out = {"device": args.device, "op": args.op, "torch": torch.__version__}
    if args.op == "import-only":
        out["counts_after_import"] = counts()
        print(json.dumps(out))
        return 0
    dev = torch.device(args.device)
    g = torch.Generator().manual_seed(4149)
    x = torch.randn(64, 784, generator=g).to(dev)
    y = torch.randn(64, 10, generator=g).to(dev)
    w1 = (torch.randn(784, 256, generator=g) * 0.05).to(dev).requires_grad_(True)
    b1 = torch.zeros(256, device=dev, requires_grad=True)
    w2 = (torch.randn(256, 10, generator=g) * 0.05).to(dev).requires_grad_(True)
    b2 = torch.zeros(10, device=dev, requires_grad=True)

    def fwd():
        return torch.relu(x @ w1 + b1) @ w2 + b2

    def run():
        if args.op == "infer":
            with torch.no_grad():
                return float(fwd().sum().cpu())
        loss = ((fwd() - y) ** 2).mean()
        gs = torch.autograd.grad(loss, (w1, b1, w2, b2))
        return float(sum(gg.sum() for gg in gs).cpu())

    run()  # 初回の遅延初期化（cuBLAS ハンドル生成・ライブラリ解決）を計数の外へ出す
    if dev.type == "cuda":
        torch.cuda.synchronize()
    if shim:
        shim.fandhe_probe_reset_first()  # first_mnk を計測 step の GEMM だけにする
    before = counts()
    out["checksum"] = run()
    if dev.type == "cuda":
        torch.cuda.synchronize()
    after = counts()
    if shim:
        out["per_step_calls"] = {n: after[i] - before[i] for i, n in enumerate(NAMES)}
        out["next_lib"] = {n: shim.fandhe_probe_next_lib(i).decode() for i, n in enumerate(NAMES)}
        out["first_mnk"] = {n: [[shim.fandhe_probe_first(i, j, w) for w in (0, 1, 2)]
                                for j in range(4)] for i, n in enumerate(NAMES)}
    print(json.dumps(out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
