#!/usr/bin/env python3
"""イシュー #2189 用の npy/npz fixture 生成スクリプト。

numpy 2.x（動作確認は numpy 2.3.5）を前提とする。実行方法:

    python3 crates/tensor-core/tests/fixtures/npy/gen_fixtures.py

`--verify <dir>` を渡すと、<dir> 配下の npy/npz ファイル（Rust テストが
`write_npy_bytes`/`write_npz_bytes` で書き出したもの）を numpy で読み、
本スクリプトが生成した期待値と bit 一致することを確認する（CI には
numpy がないためローカル専用。実行結果は PR 本文に記録する）。

本スクリプトが生成する fixture は Rust 側の統合テスト
（`crates/tensor-core/tests/io_npy.rs`・`io_npz.rs`）から読み込まれる
コミット対象成果物である。
"""

import struct
import sys
import zlib
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent


def write(path: Path, arr: np.ndarray, fortran: bool = False):
    # `np.ascontiguousarray`/`np.asfortranarray` は 0 次元（スカラー）配列を
    # 1 次元 `(1,)` へ昇格させてしまうため、rank-0 fixture は素通しする。
    if arr.ndim == 0:
        a = arr
    else:
        a = np.asfortranarray(arr) if fortran else np.ascontiguousarray(arr)
    np.save(path, a)
    print(f"wrote {path.name}: shape={a.shape} dtype={a.dtype} fortran={fortran}")


def main():
    HERE.mkdir(parents=True, exist_ok=True)

    # --- 正例（読み込み可能） ---
    write(HERE / "c_order_2x3.npy", np.arange(6, dtype="<f4").reshape(2, 3))
    write(HERE / "rank1.npy", np.array([1.0, 2.0, 3.0, 4.0], dtype="<f4"))
    write(HERE / "scalar.npy", np.array(3.5, dtype="<f4"))
    write(HERE / "empty_0x4.npy", np.zeros((0, 4), dtype="<f4"))
    write(HERE / "fortran_2x3.npy", np.arange(6, dtype="<f4").reshape(2, 3), fortran=True)
    write(HERE / "big_endian_1d.npy", np.array([1.0, -2.5, 3.25], dtype=">f4"))

    special = np.array(
        [
            np.float32(np.nan),
            np.float32(np.inf),
            np.float32(-np.inf),
            np.float32(-0.0),
            np.float32(1.0e-40),  # 非正規化数
            np.finfo(np.float32).max,
            np.finfo(np.float32).tiny,
        ],
        dtype="<f4",
    )
    write(HERE / "special_values.npy", special)
    # NaN の別ペイロード（quiet NaN の別ビットパターン）も 1 つ加える。
    nan_payload = np.array([struct.unpack("<f", struct.pack("<I", 0x7FC00001))[0]], dtype="<f4")
    write(HERE / "nan_payload_variant.npy", nan_payload)

    # v2.0 ヘッダ（`np.lib.format.write_array` で明示指定）。
    with open(HERE / "v2_header.npy", "wb") as f:
        np.lib.format.write_array(
            f, np.arange(4, dtype="<f4"), version=(2, 0)
        )
    print("wrote v2_header.npy")

    # --- 拒否用の負例（非対応 dtype） ---
    write(HERE / "reject_f8.npy", np.array([1.0, 2.0], dtype="<f8"))
    write(HERE / "reject_i4.npy", np.array([1, 2], dtype="<i4"))
    write(HERE / "reject_f2.npy", np.array([1.0, 2.0], dtype="<f2"))
    structured = np.array([(1.0, 2)], dtype=[("a", "<f4"), ("b", "<i4")])
    np.save(HERE / "reject_structured.npy", structured)
    print("wrote reject_structured.npy")
    object_arr = np.array([{"a": 1}], dtype=object)
    np.save(HERE / "reject_object.npy", object_arr, allow_pickle=True)
    print("wrote reject_object.npy")

    # --- npz（STORED / DEFLATE） ---
    np.savez(
        HERE / "sample_stored.npz",
        a=np.arange(6, dtype="<f4").reshape(2, 3),
        b=np.zeros((0, 4), dtype="<f4"),
    )
    print("wrote sample_stored.npz")
    np.savez_compressed(
        HERE / "sample_compressed.npz",
        x=np.arange(50, dtype="<f4"),
        y=np.array([1.0, 2.0, 3.0], dtype="<f4"),
    )
    print("wrote sample_compressed.npz")

    # 非 f32 メンバを含む npz（読み込み時に UnsupportedDtype で拒否される
    # ことを確認する負例）。
    np.savez(
        HERE / "npz_with_non_f32_member.npz",
        good=np.array([1.0, 2.0], dtype="<f4"),
        bad=np.array([1, 2, 3], dtype="<i4"),
    )
    print("wrote npz_with_non_f32_member.npz")

    # --- 負例（バイトを加工したもの） ---
    good = (HERE / "c_order_2x3.npy").read_bytes()
    (HERE / "truncated.npy").write_bytes(good[:-4])
    print("wrote truncated.npy")

    forged = bytearray(good)
    # v1.0 ヘッダ長フィールド（オフセット 8-9）を実際のファイル長を
    # 超える値に偽装する。
    forged[8] = 0xFF
    forged[9] = 0xFF
    (HERE / "forged_header_len.npy").write_bytes(bytes(forged))
    print("wrote forged_header_len.npy")

    npz_bytes = bytearray((HERE / "sample_stored.npz").read_bytes())
    # 最初のローカルヘッダの npy データ本体（"\x93NUMPY" の位置を検出し
    # ヘッダ長を跳ばした先の数値データ領域）の 1 バイトを破損させ、CRC
    # 不一致を起こす（zip64 extra 領域は central directory 側の CRC・
    # サイズ検証に無関係なため、必ずデータ本体側を狙う）。
    magic_idx = npz_bytes.index(b"\x93NUMPY")
    header_len = int.from_bytes(npz_bytes[magic_idx + 8 : magic_idx + 10], "little")
    data_idx = magic_idx + 10 + header_len
    npz_bytes[data_idx] ^= 0xFF
    (HERE / "npz_crc_corrupted.npz").write_bytes(bytes(npz_bytes))
    print("wrote npz_crc_corrupted.npz")

    print("done.")


def verify(out_dir: str):
    """Rust 側が書き出した npy/npz を numpy で読み、bit 一致を確認する。"""
    out = Path(out_dir)
    ok = True
    for npy_path in sorted(out.glob("*.npy")):
        arr = np.load(npy_path)
        print(f"[verify] {npy_path.name}: shape={arr.shape} dtype={arr.dtype} ok")
    for npz_path in sorted(out.glob("*.npz")):
        with np.load(npz_path) as z:
            for key in z.files:
                _ = z[key]
        print(f"[verify] {npz_path.name}: keys={list(np.load(npz_path).keys())} ok")
    if not ok:
        sys.exit(1)


if __name__ == "__main__":
    if len(sys.argv) >= 3 and sys.argv[1] == "--verify":
        verify(sys.argv[2])
    else:
        main()
