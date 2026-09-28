"""PyTorch dynamo exporter の生出力（external data を再 inline 化しない）を
生成する（イシュー #2347）。

`../pytorch-onnx-cnn-ops/gen_reference.py`（イシュー #2329）と同じ
`CASES`・`deterministic_seed` を import で再利用し、同じ決定的シードで
同じモデル・同じ入力を再構成する。#2329 との違いは exporter を dynamo
限定にし、生成後に external data を本体へ inline し直さない点のみ
（`export_one` の再 inline 化の工程を持たない dynamo 専用版）。

再生成手順（`README.md` にも記載）:
    python3 -m venv /path/to/venv
    /path/to/venv/bin/pip install --index-url https://download.pytorch.org/whl/cpu torch==2.14.0
    /path/to/venv/bin/pip install onnx==1.23.0 onnxscript==0.7.2
    /path/to/venv/bin/python gen_external.py   # このディレクトリで実行

出力はすべて `Path(__file__).parent` 相対（絶対パス・ホスト名・ユーザー名を
メタデータに含めない。`.claude/rules/security.md` A05）。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
CNN_OPS_DIR = HERE.parent / "pytorch-onnx-cnn-ops"
sys.path.insert(0, str(CNN_OPS_DIR))

from gen_reference import CASES, deterministic_seed, tensor_record  # noqa: E402

torch.set_num_threads(1)

# #2329 の 19 ケースのうち、dynamo exporter が実際に initializer を
# external data（`data_location=EXTERNAL`・非空の companion `.data`）へ
# 逃がすのは Conv 系の重み（432／288 バイト）のみだった（本スクリプトの
# 実測。README.md「実測結果」節参照）。他ケース（Pool・BN・Flatten・GAP）
# はいずれも companion `.data` ファイルは作られるが 0 バイト（external な
# initializer が 1 件も無い）であり external data 経路を検査する fixture
# として意味を持たないため、`CASES` から対象を絞り込む（INT64 の shape
# 定数〈`flatten_start2`／`gap1d` 等〉も 16〜24 バイトと小さく external に
# ならなかった。A1 要件の実測に基づく調整。計画 §4.5-4「INT64 が external
# にならなかった場合は、実測値に合わせて要件を調整」）。
EXTERNAL_CASE_NAMES = {"conv2d_basic", "conv2d_nobias", "conv2d_stride_dil_group"}
CASES = [c for c in CASES if c[0] in EXTERNAL_CASE_NAMES]


def export_external(name: str, model: torch.nn.Module, x: torch.Tensor) -> dict:
    """dynamo exporter で export し、external data をそのまま残す
    （companion `.onnx.data` を削除・再 inline しない）。"""
    onnx_path = HERE / f"{name}_dynamo.onnx"
    data_path = HERE / f"{onnx_path.name}.data"
    # 既存の生成物を削除してから export する（再生成の冪等性）。
    onnx_path.unlink(missing_ok=True)
    data_path.unlink(missing_ok=True)
    try:
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

    was_external = data_path.exists()
    # load_external_data=False: external_data/data_location フィールドを
    # inline 化せずそのまま読む（manifest 記録用）。
    mdl = onnx.load(str(onnx_path), load_external_data=False)

    externals = []
    for init in mdl.graph.initializer:
        if init.data_location == onnx.TensorProto.EXTERNAL:
            kv = {e.key: e.value for e in init.external_data}
            externals.append(
                {
                    "name": init.name,
                    "data_type": init.data_type,
                    "dims": list(init.dims),
                    "location": kv.get("location"),
                    "offset": kv.get("offset"),
                    "length": kv.get("length"),
                }
            )

    op_types = [n.op_type for n in mdl.graph.node]
    return {
        "onnx_file": onnx_path.name,
        "data_file": data_path.name if was_external else None,
        "was_external_data": was_external,
        "op_types": op_types,
        "external_initializers": externals,
    }


def main() -> None:
    manifest = {}
    for name, cls, shape in CASES:
        torch.manual_seed(deterministic_seed(name))
        model = cls().eval()
        x = torch.randn(*shape)
        with torch.no_grad():
            y = model(x)
        entry = export_external(name, model, x)
        # 参照値（入力・出力）はこのスクリプト自身が生成した `model`／`x`
        # から直接記録する（`../pytorch-onnx-cnn-ops/reference.json` を
        # 再利用しない）。同じ `deterministic_seed`／`CASES` を import して
        # いても、torch のマイナーバージョン内パッチ差・生成環境差により
        # 重み初期化の実際のビット列が再現しない場合がありうることを実測で
        # 確認した（本ファイル README.md「実測結果」節。#2347 実装時に
        # conv2d_basic で重みが一致しないことを検出）ため、本 fixture は
        # 常に「このスクリプトが今回生成した重みに対する、このスクリプトが
        # 今回計算した参照出力」という自己完結ペアで判定する。
        entry["input"] = tensor_record(x)
        entry["output"] = tensor_record(y)
        manifest[name] = entry

    (HERE / "manifest.json").write_text(json.dumps(manifest, indent=1, sort_keys=True))
    for name, m in manifest.items():
        if "error" in m:
            print(f"{name}: ERROR {m['error']}", file=sys.stderr)
        else:
            n_ext = len(m["external_initializers"])
            print(
                f"{name}: was_external={m['was_external_data']} n_external_initializers={n_ext} ops={m['op_types']}",
                file=sys.stderr,
            )


if __name__ == "__main__":
    main()
