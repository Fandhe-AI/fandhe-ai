#!/usr/bin/env python3
"""pack_padded_sequence／pad_packed_sequence／RNN 系 packed 実行の PyTorch 参照値を生成する。

イシュー #2647（親 #2625）の `tests/packed_sequence_parity.rs` が参照する固定
フィクスチャ `packed_sequence_reference.json` の生成スクリプト。CI は Python／
PyTorch に依存せず、コミット済み JSON のみを読む（`README.md` 参照）。

f32 はすべて u32 ビットパターン配列で保存する（JSON は NaN／inf を運べないため。
NaN の payload も保存する）。入力・上流勾配・出力・勾配をすべて保存し、Rust 側で
入力を再生成しない。dtype は float32、乱数は固定シード 2647。

重みレイアウト: 本リポは `x·W` 規約（`weight_ih: [D, G*H]`・`weight_hh: [H, G*H]`）、
PyTorch は `[G*H, D]`・`[G*H, H]`。この転置は **本スクリプト側で行い**、JSON には
本リポのレイアウトで保存する（重み勾配も同じ向き）。ゲート順は LSTM `i,f,g,o`・
GRU `r,z,n` で PyTorch と同じ。
"""

import json
import platform
import struct
import sys

import torch

EXPECTED_TORCH_PREFIX = "2.14.0"
assert torch.__version__.startswith(EXPECTED_TORCH_PREFIX), torch.__version__

SEED = 2647
SPECIAL_BITS = [0x7FC00001, 0x7F800000, 0xFF800000, 0x80000000, 0xFFC00002]
GEN = torch.Generator().manual_seed(SEED)


def bits_of(t):
    flat = t.detach().contiguous().reshape(-1)
    if flat.numel() == 0:
        return []
    return [v & 0xFFFFFFFF for v in flat.view(torch.int32).tolist()]


def from_bits(bits, shape):
    buf = bytearray(struct.pack("<%dI" % len(bits), *bits))
    return torch.frombuffer(buf, dtype=torch.float32).clone().reshape(shape)


def rnd(shape, scale=1.0):
    return torch.randn(tuple(shape), generator=GEN, dtype=torch.float32) * scale


def with_special(t):
    """先頭付近の複数要素へ NaN payload・±inf・-0.0 を差し込む。"""
    bits = bits_of(t)
    for i, b in enumerate(SPECIAL_BITS):
        if i < len(bits):
            bits[i] = b
    return from_bits(bits, list(t.shape))


def grad_bits(t):
    return bits_of(t.grad) if t.grad is not None else None


# ---------------------------------------------------------------- pack / unpack

PACK_SPECS = [
    # name, T, B, rest, lengths, batch_first, enforce_sorted, pre_transpose, nonfinite
    ("sorted_tm", 4, 3, [2], [4, 3, 1], False, True, False, False),
    ("sorted_bf", 4, 3, [2], [4, 3, 1], True, True, False, False),
    ("unsorted_tm", 5, 4, [3], [2, 5, 3, 5], False, False, False, False),
    ("unsorted_bf", 5, 4, [3], [2, 5, 3, 5], True, False, False, False),
    ("ties_tm", 4, 5, [2], [2, 4, 2, 4, 2], False, False, False, False),
    ("ties_bf", 4, 5, [2], [2, 4, 2, 4, 2], True, False, False, False),
    ("all_equal", 3, 3, [2], [3, 3, 3], False, True, False, False),
    ("len1_seq", 3, 3, [2], [3, 1, 1], False, False, False, False),
    ("max_lt_T", 6, 3, [2], [2, 4, 3], False, False, False, False),
    ("rank2_tm", 4, 3, [], [4, 2, 3], False, False, False, False),
    ("rank2_bf", 4, 3, [], [4, 2, 3], True, False, False, False),
    ("rest2dims", 3, 2, [2, 2], [3, 2], False, True, False, False),
    ("single_batch", 4, 1, [3], [3], False, True, False, False),
    ("pre_transpose_tm", 4, 3, [2], [4, 2, 3], False, False, True, False),
    ("nonfinite_tm", 4, 3, [2], [4, 3, 1], False, True, False, True),
    ("nonfinite_bf_unsorted", 4, 3, [2], [1, 4, 3], True, False, False, True),
]


def gen_pack_cases():
    cases = []
    for (name, T, B, rest, lengths, bf, es, pre_t, nonfinite) in PACK_SPECS:
        if pre_t:
            # 保存形状は [B,T,*]（Rust 側で transpose(0,1) の view を作る）。
            stored_shape = [B, T] + rest
            x_leaf = rnd(stored_shape).requires_grad_(True)
            x_in = x_leaf.transpose(0, 1)
        else:
            stored_shape = ([B, T] if bf else [T, B]) + rest
            x = rnd(stored_shape)
            if nonfinite:
                x = with_special(x)
            x_leaf = x.clone().requires_grad_(True)
            x_in = x_leaf
        ps = torch.nn.utils.rnn.pack_padded_sequence(
            x_in, lengths, batch_first=bf, enforce_sorted=es)
        g = rnd(list(ps.data.shape))
        (ps.data * g).sum().backward()
        cases.append({
            "name": name,
            "stored_shape": stored_shape,
            "pre_transpose": pre_t,
            "batch_first": bf,
            "enforce_sorted": es,
            "lengths": lengths,
            "x_bits": bits_of(x_leaf),
            "data_shape": list(ps.data.shape),
            "data_bits": bits_of(ps.data),
            "batch_sizes": ps.batch_sizes.tolist(),
            "sorted_indices": None if ps.sorted_indices is None else ps.sorted_indices.tolist(),
            "unsorted_indices": None if ps.unsorted_indices is None else ps.unsorted_indices.tolist(),
            "g_bits": bits_of(g),
            "x_grad_bits": grad_bits(x_leaf),
        })
    return cases


UNPACK_SPECS = [
    # name, T, B, rest, lengths, enforce_sorted, batch_first, padding_value, total_length
    ("pad0_tm", 4, 3, [2], [4, 3, 1], True, False, 0.0, None),
    ("pad0_bf", 4, 3, [2], [4, 3, 1], True, True, 0.0, None),
    ("pad_nonzero", 5, 4, [3], [2, 5, 3, 5], False, False, -3.5, None),
    ("pad_nonzero_bf", 5, 4, [3], [2, 5, 3, 5], False, True, 7.25, None),
    ("pad_nan", 4, 3, [2], [4, 2, 3], False, False, float("nan"), None),
    ("pad_inf_bf", 4, 3, [2], [4, 2, 3], False, True, float("inf"), None),
    ("max_lt_T_none", 6, 3, [2], [2, 4, 3], False, False, 0.0, None),
    ("total_length_eq_max", 6, 3, [2], [2, 4, 3], False, False, 1.5, 4),
    ("total_length_gt_tm", 4, 3, [2], [4, 2, 3], False, False, 1.5, 7),
    ("total_length_gt_bf", 4, 3, [2], [4, 2, 3], False, True, -2.0, 6),
    ("rank2_data", 4, 3, [], [4, 2, 3], False, False, 9.0, None),
    ("all_equal", 3, 3, [2], [3, 3, 3], True, False, 0.0, None),
]


def f32_py(v):
    return struct.unpack("<I", struct.pack("<f", v))[0]


def gen_unpack_cases():
    cases = []
    for (name, T, B, rest, lengths, es, bf, pv, tl) in UNPACK_SPECS:
        x = rnd([T, B] + rest)
        ps0 = torch.nn.utils.rnn.pack_padded_sequence(x, lengths, enforce_sorted=es)
        data = ps0.data.clone().requires_grad_(True)
        ps = torch.nn.utils.rnn.PackedSequence(
            data, ps0.batch_sizes, ps0.sorted_indices, ps0.unsorted_indices)
        out, lens = torch.nn.utils.rnn.pad_packed_sequence(
            ps, batch_first=bf, padding_value=pv, total_length=tl)
        g = rnd(list(out.shape))
        (out * g).sum().backward()
        cases.append({
            "name": name,
            "batch_first": bf,
            "padding_value_bits": f32_py(pv),
            "total_length": tl,
            "data_shape": list(data.shape),
            "data_bits": bits_of(data),
            "batch_sizes": ps0.batch_sizes.tolist(),
            "sorted_indices": None if ps0.sorted_indices is None else ps0.sorted_indices.tolist(),
            "out_shape": list(out.shape),
            "out_bits": bits_of(out),
            "lengths": lens.tolist(),
            "g_bits": bits_of(g),
            "data_grad_bits": grad_bits(data),
        })
    return cases


# ---------------------------------------------------------------- RNN

GATES = {"rnn": 1, "lstm": 4, "gru": 3}

RNN_SPECS = [
    # name, kind, D, H, layers, bidir, T, B, lengths, enforce_sorted, with_h0, bias, control
    ("rnn_1l_sorted", "rnn", 3, 4, 1, False, 5, 3, [5, 3, 2], True, False, True, False),
    ("rnn_1l_unsorted_h0", "rnn", 3, 4, 1, False, 5, 4, [2, 5, 3, 5], False, True, True, False),
    ("rnn_1l_nobias", "rnn", 2, 3, 1, False, 4, 3, [4, 1, 3], False, False, False, False),
    ("rnn_2l", "rnn", 3, 3, 2, False, 5, 3, [5, 4, 2], True, False, True, False),
    ("rnn_bi", "rnn", 3, 3, 1, True, 5, 3, [3, 5, 2], False, False, True, False),
    ("rnn_2l_bi_h0", "rnn", 2, 3, 2, True, 4, 4, [2, 4, 3, 1], False, True, True, False),
    ("lstm_1l_sorted", "lstm", 3, 4, 1, False, 5, 3, [5, 3, 2], True, False, True, False),
    ("lstm_1l_unsorted_h0c0", "lstm", 3, 4, 1, False, 5, 4, [2, 5, 3, 5], False, True, True, False),
    ("lstm_2l", "lstm", 3, 3, 2, False, 5, 3, [5, 4, 2], True, False, True, False),
    ("lstm_bi", "lstm", 3, 3, 1, True, 5, 3, [3, 5, 2], False, False, True, False),
    ("lstm_2l_bi_h0c0", "lstm", 2, 3, 2, True, 4, 4, [2, 4, 3, 1], False, True, True, False),
    ("gru_1l_sorted", "gru", 3, 4, 1, False, 5, 3, [5, 3, 2], True, False, True, False),
    ("gru_1l_unsorted_h0", "gru", 3, 4, 1, False, 5, 4, [2, 5, 3, 5], False, True, True, False),
    ("gru_2l", "gru", 3, 3, 2, False, 5, 3, [5, 4, 2], True, False, True, False),
    ("gru_bi", "gru", 3, 3, 1, True, 5, 3, [3, 5, 2], False, False, True, False),
    ("gru_2l_bi_h0", "gru", 2, 3, 2, True, 4, 4, [2, 4, 3, 1], False, True, True, False),
    # 対照: 全系列長 = T。PyTorch の非 packed 実行結果も保存する。
    ("rnn_control_full", "rnn", 3, 3, 1, False, 4, 3, [4, 4, 4], True, False, True, True),
    ("lstm_control_full_bi", "lstm", 3, 3, 1, True, 4, 3, [4, 4, 4], True, False, True, True),
    ("gru_control_full_2l", "gru", 3, 3, 2, False, 4, 3, [4, 4, 4], True, False, True, True),
]


def gen_rnn_cases():
    cases = []
    for (name, kind, D, H, L, bi, T, B, lengths, es, with_h0, bias, control) in RNN_SPECS:
        dirs = 2 if bi else 1
        cls = {"rnn": torch.nn.RNN, "lstm": torch.nn.LSTM, "gru": torch.nn.GRU}[kind]
        model = cls(D, H, num_layers=L, bidirectional=bi, bias=bias)
        for p in model.parameters():
            p.data = rnd(list(p.shape), 0.5)
        x_leaf = rnd([T, B, D]).requires_grad_(True)
        h0 = rnd([L * dirs, B, H], 0.5).requires_grad_(True) if with_h0 else None
        c0 = (rnd([L * dirs, B, H], 0.5).requires_grad_(True)
              if with_h0 and kind == "lstm" else None)
        state = None
        if kind == "lstm" and with_h0:
            state = (h0, c0)
        elif with_h0:
            state = h0

        ps = torch.nn.utils.rnn.pack_padded_sequence(x_leaf, lengths, enforce_sorted=es)
        out, hn = model(ps, state)
        if kind == "lstm":
            h_n, c_n = hn
        else:
            h_n, c_n = hn, None
        g1 = rnd(list(out.data.shape))
        g2 = rnd(list(h_n.shape))
        g3 = rnd(list(c_n.shape)) if c_n is not None else None
        loss = (out.data * g1).sum() + (h_n * g2).sum()
        if c_n is not None:
            loss = loss + (c_n * g3).sum()
        loss.backward()

        params = dict(model.named_parameters())
        weights = []
        for layer in range(L):
            for d in range(dirs):
                sfx = f"l{layer}" + ("_reverse" if d == 1 else "")
                w_ih = params[f"weight_ih_{sfx}"]
                w_hh = params[f"weight_hh_{sfx}"]
                entry = {
                    "weight_ih_shape": [w_ih.shape[1], w_ih.shape[0]],
                    "weight_ih_bits": bits_of(w_ih.t()),
                    "weight_hh_shape": [w_hh.shape[1], w_hh.shape[0]],
                    "weight_hh_bits": bits_of(w_hh.t()),
                    "weight_ih_grad_bits": bits_of(w_ih.grad.t()),
                    "weight_hh_grad_bits": bits_of(w_hh.grad.t()),
                }
                if bias:
                    entry["bias_ih_bits"] = bits_of(params[f"bias_ih_{sfx}"])
                    entry["bias_hh_bits"] = bits_of(params[f"bias_hh_{sfx}"])
                    entry["bias_ih_grad_bits"] = bits_of(params[f"bias_ih_{sfx}"].grad)
                    entry["bias_hh_grad_bits"] = bits_of(params[f"bias_hh_{sfx}"].grad)
                weights.append(entry)

        case = {
            "name": name, "kind": kind, "input_size": D, "hidden_size": H,
            "num_layers": L, "bidirectional": bi, "bias": bias,
            "x_shape": [T, B, D], "x_bits": bits_of(x_leaf),
            "lengths": lengths, "enforce_sorted": es,
            "weights": weights,
            "h0_bits": bits_of(h0) if h0 is not None else None,
            "c0_bits": bits_of(c0) if c0 is not None else None,
            "out_shape": list(out.data.shape), "out_bits": bits_of(out.data),
            "h_n_bits": bits_of(h_n),
            "c_n_bits": bits_of(c_n) if c_n is not None else None,
            "batch_sizes": out.batch_sizes.tolist(),
            "g1_bits": bits_of(g1), "g2_bits": bits_of(g2),
            "g3_bits": bits_of(g3) if g3 is not None else None,
            "x_grad_bits": grad_bits(x_leaf),
            "h0_grad_bits": grad_bits(h0) if h0 is not None else None,
            "c0_grad_bits": grad_bits(c0) if c0 is not None else None,
        }
        if control:
            with torch.no_grad():
                if state is None:
                    pad_out, pad_hn = model(x_leaf.detach())
                elif kind == "lstm":
                    pad_out, pad_hn = model(x_leaf.detach(), (h0.detach(), c0.detach()))
                else:
                    pad_out, pad_hn = model(x_leaf.detach(), h0.detach())
            case["padded_out_bits"] = bits_of(pad_out)
            case["padded_out_shape"] = list(pad_out.shape)
            ph = pad_hn[0] if kind == "lstm" else pad_hn
            case["padded_h_n_bits"] = bits_of(ph)
        cases.append(case)
    return cases


# ---------------------------------------------------------------- error cases

def try_pack(shape, lengths, bf, es):
    x = torch.zeros(tuple(shape), dtype=torch.float32)
    try:
        torch.nn.utils.rnn.pack_padded_sequence(x, lengths, batch_first=bf, enforce_sorted=es)
        return False
    except Exception:
        return True


def try_unpack(total_length):
    x = torch.zeros(4, 2, 2)
    ps = torch.nn.utils.rnn.pack_padded_sequence(x, [4, 2], enforce_sorted=True)
    try:
        torch.nn.utils.rnn.pad_packed_sequence(ps, total_length=total_length)
        return False
    except Exception:
        return True


def gen_error_cases():
    pack = [
        ("pack_len0", [3, 2, 2], [3, 0], False, False),
        ("pack_len_gt_T", [3, 2, 2], [4, 2], False, False),
        ("pack_len_count_mismatch", [3, 2, 2], [3], False, False),
        ("pack_unsorted_enforce", [3, 2, 2], [2, 3], False, True),
        ("pack_rank1", [3], [3], False, False),
        ("pack_batch_first_len_gt_T", [2, 3, 2], [4, 2], True, False),
    ]
    out = []
    for (name, shape, lengths, bf, es) in pack:
        out.append({
            "name": name, "kind": "pack", "shape": shape, "lengths": lengths,
            "batch_first": bf, "enforce_sorted": es,
            "torch_raises": try_pack(shape, lengths, bf, es),
        })
    out.append({"name": "unpack_total_length_lt_max", "kind": "unpack",
                "total_length": 3, "torch_raises": try_unpack(3)})
    out.append({"name": "unpack_total_length_eq_max", "kind": "unpack",
                "total_length": 4, "torch_raises": try_unpack(4)})
    return out


def main():
    doc = {
        "torch_version": torch.__version__,
        "python_version": platform.python_version(),
        "seed": SEED,
        "pack_cases": gen_pack_cases(),
        "unpack_cases": gen_unpack_cases(),
        "rnn_cases": gen_rnn_cases(),
        "error_cases": gen_error_cases(),
    }
    json.dump(doc, sys.stdout, separators=(",", ":"))
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
