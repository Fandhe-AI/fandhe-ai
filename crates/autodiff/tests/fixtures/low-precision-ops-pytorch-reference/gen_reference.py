#!/usr/bin/env python3
"""MatMul と elementwise 5 演算の低精度 forward／f32 勾配の PyTorch 参照値を生成する。

イシュー #2628（親 #2626）の `crates/facade/tests/low_precision_ops_pytorch_parity.rs`
が参照する固定フィクスチャ `low_precision_ops_reference.json` の生成スクリプト。
CI は Python／PyTorch に依存せず、コミット済み JSON のみを読む（`README.md` 参照）。

- forward 参照: 各ケース × {float16, bfloat16} で `op(x.to(dtype), ...).float()`。
  実行できない組合せは例外クラスと先頭行を `supported: false` として記録する
  （握りつぶさない）。
- 勾配参照: float32 autograd で `(op(x, ...) * g).sum().backward()`（`g` は固定
  シードの実テンソル）。入力勾配を記録する（6 Op とも記録。ゲート対象かどうかは
  Rust 側が決める）。

入力・上流勾配・shape はすべて JSON に保存し、Rust 側で入力を再生成しない。
master dtype は float32。記録する入力・出力が有限であることを生成時に assert する。
"""

import json
import platform

import torch

EXPECTED_TORCH_PREFIX = "2.14.0"
assert torch.__version__.startswith(EXPECTED_TORCH_PREFIX), torch.__version__

SEED = 20261005
DTYPES = {"f16": torch.float16, "bf16": torch.bfloat16}


def rnd(shape, scale=1.0):
    return (torch.randn(shape, dtype=torch.float32) * scale).contiguous()


def exact(values, shape):
    """f16／bf16 の双方で厳密表現できる値（2 の冪の小さな倍数）のテンソル。"""
    return torch.tensor(values, dtype=torch.float32).reshape(shape)


def build_cases():
    torch.manual_seed(SEED)
    cases = []

    def add(name, op, inputs):
        cases.append({"name": name, "op": op, "inputs": inputs})

    # matmul（K を小さく保つ。rank 2 とバッチ付き）。
    add("matmul_2x3_3x2", "matmul", [rnd([2, 3]), rnd([3, 2])])
    add("matmul_4x5_5x3", "matmul", [rnd([4, 5]), rnd([5, 3])])
    add("matmul_batched", "matmul", [rnd([2, 2, 3]), rnd([2, 3, 2])])
    add(
        "matmul_exact",
        "matmul",
        [exact([1, 2, 0.5, -1, 0.25, 4], [2, 3]), exact([1, -2, 0.5, 2, 0.25, 1], [3, 2])],
    )

    # add／mul（同形・bias パターン・一般 broadcast）。
    for op in ("add", "mul"):
        add(f"{op}_same", op, [rnd([2, 3]), rnd([2, 3])])
        add(f"{op}_bias", op, [rnd([3, 4]), rnd([4])])
        add(f"{op}_broadcast", op, [rnd([2, 1, 3]), rnd([4, 1])])
        add(
            f"{op}_exact",
            op,
            [exact([1, 2, 0.5, -1, 0.25, 4], [2, 3]), exact([2, -0.5, 1, 0.25, 8, -2], [2, 3])],
        )

    # relu（負値・ゼロを含む）。
    add("relu_random", "relu", [rnd([3, 4], 2.0)])
    add("relu_zero_mix", "relu", [exact([-2, -0.5, 0, 0, 0.5, 2], [2, 3])])
    add("relu_exact", "relu", [exact([1, -1, 0.25, -0.25, 4, 0], [2, 3])])

    # exp（出力が f16 で有限になる入力域に限定。f16 は入力約 11.09 超で inf）。
    add("exp_random", "exp", [rnd([3, 4], 1.5)])
    add("exp_exact", "exp", [exact([0, 0, 0, 0, 0, 0], [2, 3])])

    # tanh。
    add("tanh_random", "tanh", [rnd([3, 4], 1.5)])
    add("tanh_exact", "tanh", [exact([0, 0, 0, 0, 0, 0], [2, 3])])
    return cases


def apply(op, xs):
    if op == "matmul":
        return torch.matmul(xs[0], xs[1])
    if op == "add":
        return xs[0] + xs[1]
    if op == "mul":
        return xs[0] * xs[1]
    if op == "relu":
        return torch.relu(xs[0])
    if op == "exp":
        return torch.exp(xs[0])
    if op == "tanh":
        return torch.tanh(xs[0])
    raise AssertionError(op)


def tensor_json(t):
    t = t.detach().to(torch.float32).contiguous()
    assert torch.isfinite(t).all(), "非有限値を記録しない"
    return {"shape": list(t.shape), "data": t.flatten().tolist()}


def main():
    out_cases = []
    for case in build_cases():
        op = case["op"]
        xs = case["inputs"]
        # f32 autograd の勾配参照。
        leaves = [x.clone().requires_grad_(True) for x in xs]
        y = apply(op, leaves)
        g = torch.randn(y.shape, dtype=torch.float32)
        (y * g).sum().backward()
        grads = [tensor_json(leaf.grad) for leaf in leaves]
        forward = {}
        for key, dtype in DTYPES.items():
            try:
                low = apply(op, [x.to(dtype) for x in xs]).float()
                forward[key] = {"supported": True, "out": tensor_json(low)}
            except Exception as e:  # noqa: BLE001 - 例外クラスと先頭行を記録する
                forward[key] = {
                    "supported": False,
                    "error": f"{type(e).__name__}: {str(e).splitlines()[0]}",
                }
        out_cases.append(
            {
                "name": case["name"],
                "op": op,
                "inputs": [tensor_json(x) for x in xs],
                "upstream": tensor_json(g),
                "forward": forward,
                "grad_inputs": grads,
            }
        )
    doc = {
        "torch_version": torch.__version__,
        "python_version": platform.python_version(),
        "seed": SEED,
        "cases": out_cases,
    }
    print(json.dumps(doc, separators=(",", ":"), allow_nan=False))


if __name__ == "__main__":
    main()
