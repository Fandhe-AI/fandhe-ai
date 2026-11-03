#!/usr/bin/env python3
"""FractionalMaxPool2d／3d の PyTorch 参照値 fixture 生成器（イシュー #3826。設計は
docs/autodiff-pool3d-ops-decision.md §20・§24）。

`_random_samples` を明示して forward を実行し、samples・input・output・indices と
重み付き和の backward（入力勾配）を JSON に収める。xorshift64* と PyTorch の生成器は一致しない
ため、乱数そのものではなく「同じ標本に対する選択結果」を突合する。あわせて実行可能性の式と
範囲外標本の扱いを実測して `probes` に収める。

実行: python3 gen_reference.py > fractional_max_pool_reference.json （torch==2.14.0+cpu）
"""
import json
import sys

import torch
import torch.nn.functional as F

torch.manual_seed(3826)
F32_BELOW_ONE = 1.0 - 2.0 ** -24


def flat(t):
    return [float(v) for v in t.reshape(-1).tolist()]


def case(name, dim, in_shape, k, out, samples):
    x = torch.randn(*in_shape, dtype=torch.float32)
    s = torch.tensor(samples, dtype=torch.float32).reshape(in_shape[0], in_shape[1], dim)
    xg = x.clone().requires_grad_(True)
    fn = F.fractional_max_pool2d if dim == 2 else F.fractional_max_pool3d
    y, idx = fn(xg, k, output_size=out, return_indices=True, _random_samples=s)
    w = (torch.arange(y.numel()).reshape(y.shape) % 5 + 1).float()
    (y * w).sum().backward()
    return {
        "name": name,
        "dim": dim,
        "in_shape": list(in_shape),
        "kernel": list(k),
        "output": list(out),
        "samples": flat(s),
        "input": flat(x),
        "output_values": flat(y.detach()),
        "indices": [int(v) for v in idx.reshape(-1).tolist()],
        "weights": flat(w),
        "input_grad": flat(xg.grad),
    }


def probe(dim, in_shape, k, out, samples):
    x = torch.randn(*in_shape)
    fn = F.fractional_max_pool2d if dim == 2 else F.fractional_max_pool3d
    s = torch.tensor(samples, dtype=torch.float32).reshape(in_shape[0], in_shape[1], dim)
    base = {"dim": dim, "in_shape": list(in_shape), "kernel": list(k), "output": list(out),
            "samples": samples}
    try:
        y, idx = fn(x, k, output_size=out, return_indices=True, _random_samples=s)
        return {**base, "accepted": True, "indices": [int(v) for v in idx.reshape(-1).tolist()][:8]}
    except Exception as e:  # noqa: BLE001 - 実測のため全例外を記録する
        return {**base, "accepted": False, "error": type(e).__name__}


cases = [
    case("2d_batched", 2, (2, 2, 7, 9), (2, 3), (4, 5), [0.1, 0.7, 0.35, 0.9, 0.0, 0.5, 0.62, 0.21]),
    case("2d_overlap_equal_feasibility", 2, (1, 1, 5, 5), (2, 2), (4, 4), [0.0, F32_BELOW_ONE]),
    case("2d_single_window", 2, (1, 1, 6, 6), (3, 3), (1, 1), [0.5, 0.25]),
    case("3d_batched", 3, (1, 2, 5, 6, 7), (2, 2, 3), (3, 4, 4), [0.2, 0.6, 0.9, 0.0, 0.45, 0.77]),
    case("3d_small", 3, (1, 1, 4, 4, 4), (2, 2, 2), (2, 2, 2), [0.3, 0.8, 0.05]),
]

probes = [
    probe(2, (1, 1, 4, 4), (2, 2), (3, 3), [0.1, 0.2]),  # out+k-1 == in
    probe(2, (1, 1, 4, 4), (2, 2), (4, 3), [0.1, 0.2]),  # out+k-1 > in
    probe(3, (1, 1, 4, 4, 4), (2, 2, 2), (3, 3, 3), [0.1, 0.2, 0.3]),  # 3d: out+k-1 == in
    probe(3, (1, 1, 4, 4, 4), (2, 2, 2), (2, 2, 2), [0.1, 0.2, 0.3]),  # 3d: out+k-1 < in
    probe(2, (1, 1, 6, 6), (2, 2), (3, 3), [1.0, 0.2]),  # 標本 == 1
    probe(2, (1, 1, 6, 6), (2, 2), (3, 3), [-0.1, 0.2]),  # 標本 < 0
]

json.dump({"torch_version": torch.__version__, "cases": cases, "probes": probes}, sys.stdout)
