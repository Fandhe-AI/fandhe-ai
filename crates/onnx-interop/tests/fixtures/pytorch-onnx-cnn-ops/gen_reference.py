"""PyTorch を実行して ONNX Conv・Pool・BN・Flatten の import fixture を生成する
（イシュー #2329・親 #2185）。

役割: `torch.onnx.export`（TorchScript exporter `dynamo=False` と既定の
dynamo exporter `dynamo=True` の両方）で各ケースの `.onnx` を実生成し、
initializer と PyTorch `state_dict` の bit 一致を生成時に自己検証したうえで、
PyTorch の実行結果（`model(x)`）を `reference.json` へ書き出す。

再生成手順（`README.md` にも記載）:
    python3 -m venv /path/to/venv
    /path/to/venv/bin/pip install --index-url https://download.pytorch.org/whl/cpu torch
    /path/to/venv/bin/pip install onnx onnxscript
    /path/to/venv/bin/python gen_reference.py   # このディレクトリで実行

出力はすべて `Path(__file__).parent` 相対（絶対パス・ホスト名・ユーザー名を
メタデータに含めない。`.claude/rules/security.md` A05）。値はすべて f32 の
u32 bit パターン（10 進テキストの丸めに依存しない）として保存する。

CI はコミット済み `reference.json`・`*.onnx` のみを読み、torch には依存しない
（`.claude/rules/ci.md`「グローバル状態を汚す処理を workflow に書かない」と
同じ方針。`crates/autodiff/tests/fixtures/lbfgs-pytorch-reference/README.md`
の先例を踏襲する）。
"""

from __future__ import annotations

import hashlib
import json
import platform
import struct
import sys
from pathlib import Path

import torch
import torch.nn as nn

HERE = Path(__file__).resolve().parent

torch.manual_seed(0)
torch.set_num_threads(1)


def f32_bits(t: torch.Tensor) -> list[int]:
    """f32 テンソルを row-major の u32 bit パターン列へ変換する。

    Rust 側（`f32::from_bits`）と 10 進テキストを経由せず bit 単位で
    突合するための表現（`.claude/rules/coding-rust.md` の bit 一致契約と
    同じ考え方）。
    """
    arr = t.detach().to(torch.float32).contiguous().flatten().tolist()
    out = []
    for v in arr:
        out.append(struct.unpack("<I", struct.pack("<f", v))[0])
    return out


def tensor_record(t: torch.Tensor) -> dict:
    return {"shape": list(t.shape), "bits": f32_bits(t)}


# ---------------------------------------------------------------------------
# モデル定義（ケースごとに 1 op を検証する。BN は Conv->BN 畳み込み〈eval 時
# peephole〉を避けるため単独モデルにする。計画 §2 ケース表）。
# ---------------------------------------------------------------------------


class Conv2dBasic(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 4, 3, padding=1, bias=True)

    def forward(self, x):
        return self.conv(x)


class Conv2dStrideDilGroup(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(4, 4, 3, stride=2, padding=2, dilation=2, groups=2)

    def forward(self, x):
        return self.conv(x)


class Conv2dNoBias(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 4, 3, padding=1, bias=False)

    def forward(self, x):
        return self.conv(x)


class Conv1dBasic(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv1d(3, 4, 3, padding=1, stride=2)

    def forward(self, x):
        return self.conv(x)


class MaxPool2dBasic(nn.Module):
    def __init__(self):
        super().__init__()
        self.pool = nn.MaxPool2d(2, 2)

    def forward(self, x):
        return self.pool(x)


class MaxPool2dPadDilCeil(nn.Module):
    def __init__(self):
        super().__init__()
        self.pool = nn.MaxPool2d(3, stride=2, padding=1, dilation=2, ceil_mode=True)

    def forward(self, x):
        return self.pool(x)


class MaxPool1dBasic(nn.Module):
    def __init__(self):
        super().__init__()
        self.pool = nn.MaxPool1d(3, stride=2, padding=1)

    def forward(self, x):
        return self.pool(x)


class AvgPool2dIncludePad(nn.Module):
    def __init__(self):
        super().__init__()
        self.pool = nn.AvgPool2d(3, 2, padding=1)  # count_include_pad=True (既定)

    def forward(self, x):
        return self.pool(x)


class AvgPool2dExcludePad(nn.Module):
    def __init__(self):
        super().__init__()
        self.pool = nn.AvgPool2d(3, 2, padding=1, count_include_pad=False)

    def forward(self, x):
        return self.pool(x)


class AvgPool2dCeilOverhangIncl(nn.Module):
    def __init__(self):
        super().__init__()
        self.pool = nn.AvgPool2d(3, 2, padding=1, ceil_mode=True, count_include_pad=True)

    def forward(self, x):
        return self.pool(x)


class AvgPool2dCeilOverhangExcl(nn.Module):
    def __init__(self):
        super().__init__()
        self.pool = nn.AvgPool2d(3, 2, padding=1, ceil_mode=True, count_include_pad=False)

    def forward(self, x):
        return self.pool(x)


class AvgPool1dBasic(nn.Module):
    def __init__(self):
        super().__init__()
        self.pool = nn.AvgPool1d(3, 2, padding=1)

    def forward(self, x):
        return self.pool(x)


class Gap2d(nn.Module):
    def __init__(self):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        return self.pool(x)


class Gap1d(nn.Module):
    def __init__(self):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):
        return self.pool(x)


class Bn2dEval(nn.Module):
    def __init__(self, eps=1e-5):
        super().__init__()
        self.bn = nn.BatchNorm2d(3, eps=eps)
        with torch.no_grad():
            self.bn.weight.copy_(torch.tensor([1.5, 0.5, 2.0]))
            self.bn.bias.copy_(torch.tensor([0.1, -0.2, 0.3]))
            self.bn.running_mean.copy_(torch.tensor([0.05, -0.1, 0.2]))
            self.bn.running_var.copy_(torch.tensor([1.2, 0.8, 2.5]))
        self.bn.eval()

    def forward(self, x):
        return self.bn(x)


class Bn2dEvalNonDefaultEps(Bn2dEval):
    def __init__(self):
        super().__init__(eps=1e-2)


class Bn1dEval(nn.Module):
    def __init__(self):
        super().__init__()
        self.bn = nn.BatchNorm1d(3)
        with torch.no_grad():
            self.bn.weight.copy_(torch.tensor([1.1, 0.9, 1.3]))
            self.bn.bias.copy_(torch.tensor([-0.05, 0.15, 0.0]))
            self.bn.running_mean.copy_(torch.tensor([0.02, -0.03, 0.01]))
            self.bn.running_var.copy_(torch.tensor([0.9, 1.1, 1.5]))
        self.bn.eval()

    def forward(self, x):
        return self.bn(x)


class FlattenDefault(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()

    def forward(self, x):
        return self.flatten(x)


class FlattenStart2(nn.Module):
    def forward(self, x):
        return torch.flatten(x, 2)


CASES = [
    ("conv2d_basic", Conv2dBasic, (1, 3, 8, 8)),
    ("conv2d_stride_dil_group", Conv2dStrideDilGroup, (1, 4, 9, 9)),
    ("conv2d_nobias", Conv2dNoBias, (1, 3, 8, 8)),
    ("conv1d_basic", Conv1dBasic, (1, 3, 9)),
    ("maxpool2d_basic", MaxPool2dBasic, (1, 3, 8, 8)),
    ("maxpool2d_pad_dil_ceil", MaxPool2dPadDilCeil, (1, 3, 9, 9)),
    ("maxpool1d_basic", MaxPool1dBasic, (1, 3, 9)),
    ("avgpool2d_include_pad", AvgPool2dIncludePad, (1, 3, 8, 8)),
    ("avgpool2d_exclude_pad", AvgPool2dExcludePad, (1, 3, 8, 8)),
    # kernel 3・stride 2・padding 1・ceil_mode=True の入力形状は 6x6 とする（7x7 では
    # ceil_mode が生む最終窓の右端・下端が padded 領域の終端と一致するのみで、
    # padded 領域からのはみ出し〈overhang〉を伴わない。6x6 は floor_mode に対し
    # ceil_mode が窓を 1 行・1 列増やし、その最終窓が padded 領域（6+2*1=8）を
    # 越えるため divisor クリップ規則を実際に踏む。イシュー #2329 PR #2343
    # codex-review 指摘・2026-09-28 実測で確認済み: divisor_override=9（クリップ
    # 無効化）との出力差が 7x7 では 0、6x6 では非ゼロ）
    ("avgpool2d_ceil_overhang_incl", AvgPool2dCeilOverhangIncl, (1, 3, 6, 6)),
    ("avgpool2d_ceil_overhang_excl", AvgPool2dCeilOverhangExcl, (1, 3, 6, 6)),
    ("avgpool1d_basic", AvgPool1dBasic, (1, 3, 9)),
    ("gap2d", Gap2d, (1, 3, 7, 9)),
    ("gap1d", Gap1d, (1, 3, 9)),
    ("bn2d_eval", Bn2dEval, (2, 3, 5, 5)),
    ("bn2d_eval_eps", Bn2dEvalNonDefaultEps, (2, 3, 5, 5)),
    ("bn1d_eval", Bn1dEval, (2, 3, 7)),
    ("flatten_default", FlattenDefault, (2, 3, 4, 5)),
    ("flatten_start2", FlattenStart2, (2, 3, 4, 5)),
]

EXPORTERS = ["ts", "dynamo"]


def state_dict_bits(model: nn.Module) -> dict:
    return {k: tensor_record(v) for k, v in model.state_dict().items()}


def export_one(name: str, model: nn.Module, x: torch.Tensor, exporter: str) -> dict | None:
    """1 ケース・1 exporter を export する。失敗した場合は None を返し、
    呼び出し元が `errors` へ記録する（生成できないケースの捏造をしない。
    実装計画 §8 リスク表）。
    """
    onnx_path = HERE / f"{name}_{exporter}.onnx"
    data_path = HERE / f"{onnx_path.name}.data"
    try:
        if exporter == "ts":
            torch.onnx.export(
                model,
                x,
                str(onnx_path),
                dynamo=False,
                opset_version=13,
                export_params=True,
                input_names=["x"],
                output_names=["y"],
            )
        else:
            torch.onnx.export(
                model,
                x,
                str(onnx_path),
                dynamo=True,
                export_params=True,
                input_names=["x"],
                output_names=["y"],
            )
    except Exception as e:  # noqa: BLE001 — 生成失敗を記録するため広く捕捉する
        return {"error": f"{type(e).__name__}: {e}"}

    import onnx

    # dynamo exporter は既定でテンソルサイズに応じ initializer を external
    # data（`.onnx.data` companion file・`data_location=EXTERNAL`）へ逃がす
    # （実装計画で判明した事実。本クレートの `onnx::graph::decode_tensor` は
    # external data を意図的に非対応——`raw_data`/`float_data` の完全一致
    # 検証のみを行い、`external_data`/`data_location` フィールドは未宣言
    # のまま無視する設計——であり、これは import の失敗要因である
    # ため R5 の対象として README に記録する。CNN op の意味論突合が本
    # fixture 群の主目的のため、生成後に external data をモデル本体へ
    # inline し直し、companion file を残さない自己完結 .onnx にする
    # （`onnx.load` は既定で external data を自動でメモリへ読み込むため、
    # `save_as_external_data=False` で再保存するだけで inline 化できる）。
    was_external = data_path.exists()
    mdl = onnx.load(str(onnx_path))  # load_external_data=True が既定
    if was_external:
        onnx.save_model(mdl, str(onnx_path), save_as_external_data=False)
        data_path.unlink(missing_ok=True)
        mdl = onnx.load(str(onnx_path))
    op_types = [n.op_type for n in mdl.graph.node]
    opset = [(o.domain, o.version) for o in mdl.opset_import]

    # initializer -> state_dict 対応の自己検証（生成時。R2 の事前保証）。
    # dynamo exporter は initializer 名を `state_dict` のキーと異なる形
    # （例: プレフィックス付与）に変える場合があるため、値（bit パターン・
    # shape）で対応付けを探す。1 対 1 で一致するものを `mapped`、
    # 対応が見つからないものを `unmapped` として記録する（隠さず記録する。
    # 実装計画 §2 の「隠さず unmapped として記録する」方針）。
    sd = {k: v for k, v in model.state_dict().items()}
    sd_bits = {k: (tuple(v.shape), f32_bits(v)) for k, v in sd.items()}
    used_sd_keys: set[str] = set()
    name_map: dict[str, str] = {}
    unmapped: list[str] = []
    for init in mdl.graph.initializer:
        arr = onnx.numpy_helper.to_array(init).astype("float32").flatten().tolist()
        bits = tuple(struct.unpack("<I", struct.pack("<f", v))[0] for v in arr)
        shape = tuple(init.dims)
        found = None
        for k, (sshape, sbits) in sd_bits.items():
            if k in used_sd_keys:
                continue
            if tuple(sshape) == shape and tuple(sbits) == bits:
                found = k
                break
        if found is not None:
            name_map[init.name] = found
            used_sd_keys.add(found)
        else:
            unmapped.append(init.name)

    return {
        "onnx_file": onnx_path.name,
        "op_types": op_types,
        "opset": opset,
        "ir_version": mdl.ir_version,
        "initializer_names": [i.name for i in mdl.graph.initializer],
        "name_map": name_map,
        "unmapped_initializers": unmapped,
        "was_external_data": was_external,
    }


def deterministic_seed(name: str) -> int:
    """ケース名から決定的なシードを導出する。

    Python の組込み `hash()` は文字列に対して既定でプロセスごとにランダムな
    salt を用いる（`PYTHONHASHSEED`）ため、再生成のたびに異なる乱数列に
    なってしまい、`README.md` に記録する実測値（bit 一致可否・
    `max_rel_err`）が再現しない。`hashlib.md5`（暗号強度は不要。決定的な
    軽量ハッシュとして使うのみ）で seed を導出することで、同じケース名は
    常に同じ入力・重みを生成する。
    """
    digest = hashlib.md5(name.encode("utf-8"), usedforsecurity=False).digest()
    return int.from_bytes(digest[:4], "little")


def main() -> None:
    out_cases = {}
    for name, cls, shape in CASES:
        torch.manual_seed(deterministic_seed(name))
        model = cls().eval()
        x = torch.randn(*shape)
        with torch.no_grad():
            y = model(x)
        assert torch.isfinite(y).all(), f"{name}: 出力に非有限値"

        exporters_out = {}
        for exp in EXPORTERS:
            exporters_out[exp] = export_one(name, model, x, exp)

        out_cases[name] = {
            "input": tensor_record(x),
            "output": tensor_record(y),
            "state_dict": state_dict_bits(model),
            "exporters": exporters_out,
        }

    meta = {
        "torch_version": torch.__version__,
        "python_version": platform.python_version(),
        "machine": platform.machine(),
        "system": platform.system(),
    }
    try:
        import onnx

        meta["onnx_version"] = onnx.__version__
    except Exception:  # noqa: BLE001
        meta["onnx_version"] = None

    doc = {"meta": meta, "cases": out_cases}
    (HERE / "reference.json").write_text(json.dumps(doc, indent=1, sort_keys=True))
    print("wrote", HERE / "reference.json", file=sys.stderr)
    for name, c in out_cases.items():
        for exp, e in c["exporters"].items():
            if "error" in e:
                print(f"{name} [{exp}]: ERROR {e['error']}", file=sys.stderr)
            else:
                print(
                    f"{name} [{exp}]: ops={e['op_types']} opset={e['opset']} "
                    f"unmapped={e['unmapped_initializers']}",
                    file=sys.stderr,
                )


if __name__ == "__main__":
    main()
