#!/usr/bin/env python3
"""framework-compare と同一プロトコルの Python 版（PyTorch / TensorFlow / SciPy）。

bench-common（xorshift64*・固定シード・warmup 20 + 計測 20・線形補間分位点・f64 逐次 checksum）
と bench-candle（計測窓 = matmul ディスパッチ + ホスト実体化 + 全要素 checksum。parity は窓外）を
そのまま移植し、同じ JSONL スキーマで出力する。

  bench_py.py --framework pytorch|tensorflow|scipy --task gemm|train|infer --device cpu|metal --size N --out FILE
"""
import argparse, json, math, sys, time
import numpy as np

# ---- bench-common/src/lib.rs:147-197 ------------------------------------
SEED_A, SEED_B = 0xA11CE, 0xB0B
SEED_X, SEED_Y = 0xDA7A0001, 0xDA7A0002
SEED_L1, SEED_L2 = 0x1111_1111, 0x2222_2222
WARMUP, ITERS = 20, 20
BATCH, D_IN, D_HID, D_OUT = 64, 784, 256, 10
TRAIN_STEPS, TRAIN_WARMUP, LR = 100, 20, 0.01
M64 = (1 << 64) - 1

CACHE = __import__("pathlib").Path(__file__).with_name("cache")

def fill_vec(seed: int, n: int) -> np.ndarray:
    """xorshift64*（Vigna）。上位 24bit → [0,1) → -0.5 で [-0.5,0.5)。bench-common と同順。
    純 Python の逐次生成なので (seed, n) ごとに .npy へキャッシュする（値は決定的）。"""
    CACHE.mkdir(exist_ok=True)
    cp = CACHE / f"fill_{seed:x}_{n}.npy"
    if cp.exists():
        return np.load(cp)
    x = seed & M64
    if x == 0:
        x = 0x9E3779B97F4A7C15
    out = np.empty(n, dtype=np.float32)
    for i in range(n):
        x ^= x >> 12
        x ^= (x << 25) & M64
        x ^= x >> 27
        r = (x * 0x2545F4914F6CDD1D) & M64
        out[i] = np.float32((r >> 40) / 16777216.0) - np.float32(0.5)
    np.save(cp, out)
    return out

def quantile(secs, p):
    s = sorted(secs)
    idx = p * (len(s) - 1)
    lo, hi = int(math.floor(idx)), int(math.ceil(idx))
    frac = idx - lo
    return s[lo] * (1 - frac) + s[hi] * frac

def checksum(arr) -> float:
    a = np.asarray(arr, dtype=np.float32).reshape(-1)
    # f64 逐次和（row-major）。np.sum の pairwise と区別するため累積で計算
    return float(np.cumsum(a.astype(np.float64))[-1]) if a.size else 0.0

# ---- bench-common/src/parity.rs（FMA 参照 + 複合判定 + スケール付き絶対誤差救済）----
def reference_gemm(a, b, n):
    """c[i][j] = fma(a[i][k], b[k][j], c[i][j])、k 昇順。f32×f32 の積は f64 で正確なので
    f64 で加算して f32 へ 1 回丸めることで FMA を再現する（二重丸めの稀な境界ケースを除き bit 同一）。"""
    CACHE.mkdir(exist_ok=True)
    cp = CACHE / f"ref_{n}.npy"
    if cp.exists():
        return np.load(cp)
    A = a.reshape(n, n).astype(np.float64)
    B = b.reshape(n, n).astype(np.float64)
    c = np.zeros((n, n), dtype=np.float32)
    for k in range(n):
        c = (c.astype(np.float64) + np.outer(A[:, k], B[k, :])).astype(np.float32)
    np.save(cp, c)
    return c

def parity(actual, ref, n, a, b):
    act = np.asarray(actual, dtype=np.float32).reshape(-1).astype(np.float64)
    r = ref.reshape(-1).astype(np.float64)
    diff = np.abs(act - r)
    scale = np.maximum(np.maximum(np.abs(act), np.abs(r)), 1e-12)
    rel = diff / scale
    legacy = (rel < 1e-3) | (diff < 1e-5)
    sa, sb = float(np.max(np.abs(a))), float(np.max(np.abs(b)))
    bound = 0.5 * (2.0 ** -24) * n * sa * sb if (math.isfinite(sa) and math.isfinite(sb) and sa >= 0 and sb >= 0) else 0.0
    scaled = diff <= bound
    ok = legacy | scaled
    finite = np.isfinite(act)
    ok &= finite
    return dict(parity_total=int(act.size), parity_fail_count=int((~ok).sum()),
                parity_max_abs_err=float(diff.max()), parity_max_rel_err=float(rel.max()),
                parity_scaled_abs_bound=bound, parity_scaled_abs_rescued=int((scaled & ~legacy & finite).sum()))

# ---- フレームワーク別バックエンド ------------------------------------------
class Torch:
    name = "pytorch"
    def __init__(self, device):
        import torch
        self.t = torch
        self.dev = torch.device({"metal": "mps", "cuda": "cuda"}.get(device, "cpu"))
        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("MEASURE_ERROR: torch CUDA not available")
        self.version = torch.__version__
    def upload(self, x):
        return self.t.from_numpy(np.ascontiguousarray(x)).to(self.dev)
    def matmul_to_host(self, a, b):
        return (a @ b).cpu().numpy()
    def train_setup(self, w1, b1, w2, b2):
        self.p = [self.upload(v).requires_grad_(True) for v in (w1, b1, w2, b2)]
    def train_step(self, x, y):
        t = self.t
        w1, b1, w2, b2 = self.p
        h = t.relu(x @ w1 + b1)
        pred = h @ w2 + b2
        loss = ((pred - y) ** 2).mean()
        gs = t.autograd.grad(loss, self.p)
        with t.no_grad():
            for p, g in zip(self.p, gs):
                p.sub_(g * LR)
        return float(loss.item())
    def forward_to_host(self, x):
        t = self.t
        w1, b1, w2, b2 = self.p
        with t.no_grad():
            return (t.relu(x @ w1 + b1) @ w2 + b2).cpu().numpy()

class TF:
    name = "tensorflow"
    def __init__(self, device):
        import os
        os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
        import tensorflow as tf
        self.tf = tf
        self.version = tf.__version__
        gpus = tf.config.list_physical_devices("GPU")
        if device in ("metal", "cuda"):
            if not gpus:
                raise RuntimeError(f"MEASURE_ERROR: TensorFlow GPU ({device}) device not available")
            self.devname = "/GPU:0"
        else:
            tf.config.set_visible_devices([], "GPU")
            self.devname = "/CPU:0"
        self.ctx = tf.device(self.devname)
    def upload(self, x):
        with self.ctx:
            return self.tf.identity(self.tf.constant(np.ascontiguousarray(x)))
    def matmul_to_host(self, a, b):
        with self.ctx:
            return self.tf.matmul(a, b).numpy()
    def train_setup(self, w1, b1, w2, b2):
        with self.ctx:
            self.p = [self.tf.Variable(v) for v in (w1, b1, w2, b2)]
    def train_step(self, x, y):
        tf = self.tf
        with self.ctx:
            with tf.GradientTape() as tape:
                w1, b1, w2, b2 = self.p
                pred = tf.nn.relu(x @ w1 + b1) @ w2 + b2
                loss = tf.reduce_mean(tf.square(pred - y))
            gs = tape.gradient(loss, self.p)
            for p, g in zip(self.p, gs):
                p.assign_sub(g * LR)
            return float(loss.numpy())
    def forward_to_host(self, x):
        tf = self.tf
        with self.ctx:
            w1, b1, w2, b2 = self.p
            return (tf.nn.relu(x @ w1 + b1) @ w2 + b2).numpy()

class SciPy:
    """SciPy: GEMM は scipy.linalg.blas.sgemm（Accelerate/OpenBLAS 直呼び）、MLP は NumPy で手書き backprop。"""
    name = "scipy"
    def __init__(self, device):
        if device != "cpu":
            raise RuntimeError("MEASURE_ERROR: scipy is CPU only")
        import scipy, scipy.linalg.blas as blas
        self.blas = blas
        self.version = scipy.__version__
    def upload(self, x):
        return np.ascontiguousarray(x, dtype=np.float32)
    def mm(self, a, b):
        return self.blas.sgemm(1.0, a, b)
    def matmul_to_host(self, a, b):
        return self.mm(a, b)
    def train_setup(self, w1, b1, w2, b2):
        self.p = [np.array(v, dtype=np.float32) for v in (w1, b1, w2, b2)]
    def train_step(self, x, y):
        w1, b1, w2, b2 = self.p
        z1 = self.mm(x, w1) + b1
        h = np.maximum(z1, 0)
        pred = self.mm(h, w2) + b2
        d = pred - y
        loss = float(np.mean(d * d))
        n = d.size
        gp = (2.0 / n) * d
        gw2 = self.mm(h.T, gp); gb2 = gp.sum(0)
        gh = self.mm(gp, w2.T) * (z1 > 0)
        gw1 = self.mm(x.T, gh); gb1 = gh.sum(0)
        w1 -= LR * gw1; b1 -= LR * gb1; w2 -= LR * gw2; b2 -= LR * gb2
        return loss
    def forward_to_host(self, x):
        w1, b1, w2, b2 = self.p
        return self.mm(np.maximum(self.mm(x, w1) + b1, 0), w2) + b2

BACKENDS = {"pytorch": Torch, "tensorflow": TF, "scipy": SciPy}

# ---- タスク ----------------------------------------------------------------
def run_gemm(be, n):
    a = fill_vec(SEED_A, n * n); b = fill_vec(SEED_B, n * n)
    A = be.upload(a.reshape(n, n)); B = be.upload(b.reshape(n, n))
    secs, last = [], None
    for it in range(WARMUP + ITERS):
        t0 = time.perf_counter()
        c = be.matmul_to_host(A, B)
        cs = checksum(c)
        dt = time.perf_counter() - t0
        if it >= WARMUP:
            secs.append(dt)
        last = (c, cs)
    c, cs = last
    if cs == 0.0 or not math.isfinite(cs):
        raise RuntimeError("MEASURE_ERROR: gemm checksum is degenerate")
    ref = reference_gemm(a, b, n)
    rec = dict(task="gemm", size=n, checksum=cs, **parity(c, ref, n, a, b))
    return secs, rec, {"gflops": 2 * n ** 3 / quantile(secs, 0.5) / 1e9}

def mlp_init():
    w1 = fill_vec(SEED_L1, D_IN * D_HID).reshape(D_IN, D_HID); b1 = np.zeros(D_HID, np.float32)
    w2 = fill_vec(SEED_L2, D_HID * D_OUT).reshape(D_HID, D_OUT); b2 = np.zeros(D_OUT, np.float32)
    x = fill_vec(SEED_X, BATCH * D_IN).reshape(BATCH, D_IN); y = fill_vec(SEED_Y, BATCH * D_OUT).reshape(BATCH, D_OUT)
    return w1, b1, w2, b2, x, y

def run_train(be):
    w1, b1, w2, b2, x, y = mlp_init()
    be.train_setup(w1, b1, w2, b2)
    X, Y = be.upload(x), be.upload(y)
    secs, last = [], None
    for s in range(TRAIN_STEPS):
        t0 = time.perf_counter()
        last = be.train_step(X, Y)
        dt = time.perf_counter() - t0
        if s >= TRAIN_WARMUP:
            secs.append(dt)
    if not math.isfinite(last):
        raise RuntimeError("MEASURE_ERROR: non-finite loss")
    return secs, dict(task="train", size=BATCH, checksum=float(last)), {}

def run_infer(be):
    w1, b1, w2, b2, x, y = mlp_init()
    be.train_setup(w1, b1, w2, b2)
    X = be.upload(x)
    secs, last = [], None
    for it in range(WARMUP + ITERS):
        t0 = time.perf_counter()
        out = be.forward_to_host(X)
        cs = checksum(out)
        dt = time.perf_counter() - t0
        if it >= WARMUP:
            secs.append(dt)
        last = cs
    return secs, dict(task="infer", size=BATCH, checksum=last), {"throughput_per_s": 1.0 / quantile(secs, 0.5)}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--framework", required=True, choices=BACKENDS)
    ap.add_argument("--task", required=True, choices=["gemm", "train", "infer"])
    ap.add_argument("--device", required=True, choices=["cpu", "metal", "cuda"])
    ap.add_argument("--size", type=int, default=64)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    be = BACKENDS[a.framework](a.device)
    if a.task == "gemm":
        secs, rec, extra = run_gemm(be, a.size)
    elif a.task == "train":
        secs, rec, extra = run_train(be)
    else:
        secs, rec, extra = run_infer(be)
    warm = WARMUP if a.task != "train" else TRAIN_WARMUP
    iters = ITERS if a.task != "train" else TRAIN_STEPS - TRAIN_WARMUP
    row = dict(framework=a.framework, version=be.version, task=rec["task"], device=a.device, size=rec["size"],
               median_s=quantile(secs, 0.5), q1_s=quantile(secs, 0.25), q3_s=quantile(secs, 0.75), **extra,
               checksum=rec["checksum"], warmup=warm, iters=iters, mode="fresh")
    for k in ("parity_total", "parity_fail_count", "parity_max_abs_err", "parity_max_rel_err", "parity_scaled_abs_bound", "parity_scaled_abs_rescued"):
        if k in rec:
            row[k] = rec[k]
    with open(a.out, "a") as f:
        f.write(json.dumps(row) + "\n")
    print(json.dumps(row))

if __name__ == "__main__":
    main()
