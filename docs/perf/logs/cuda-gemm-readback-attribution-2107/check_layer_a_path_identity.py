#!/usr/bin/env python3
"""イシュー #2107: Layer A（fandhe-ai =0.9.0）と Layer B（HEAD）の計測経路同一性検査。

`orchestrate.sh` から呼ばれ、`git show <ref>:<path>` で v0.9.0 と HEAD の
「計測経路上の項目」だけを取り出し、コメント行と
`#[cfg(feature = "internal-diagnostics")]` ゲート項目（本番既定ビルドに
含まれない）を除いた正規化テキストを比較する（python3 標準ライブラリのみ）。
`crates/backend-cuda` のファイル単位の diff は #2299 の feature gate・ドキュメント
変更・診断専用機能追加で常に差分が出るため、Layer A の計測経路
（GEMM 起動選択・tiled カーネルソース・D2H readback 宛先確保）に限定して
比較する（RULE.txt「同一コード確認」節。PR #2452 codex-review 指摘 P1）。
出力は `layerA_same_code: yes|no|unknown` の 1 行 + 項目別の `path_item:` 行。
項目が抽出できない場合は fail-closed で `unknown`（`yes` にしない）。
使い方: check_layer_a_path_identity.py [<base_ref=v0.9.0>] [<head_ref=HEAD>]
"""

from __future__ import annotations

import re
import subprocess
import sys

SRC = "crates/backend-cuda/src/"
# (ファイル, 種別, 名前)。Layer A 経路: bench-fandhe → 本番 matmul
# （run_f32_kernel → 選択 → tiled カーネル）と本番 readback。
ITEMS = [
    ("gemm.rs", "fn", "validate_gemm_dims"),
    ("gemm.rs", "fn", "validate_output_len"),
    ("gemm.rs", "fn", "validate_tiled_k_bound"),
    ("gemm.rs", "fn", "tiled_f32_kernel_kind"),
    # tile 選択（64x64／128x64）を決める本番既定の有効化フラグ・しきい値・選択関数
    # （値の差し替えで Layer A が通るタイルが変わる。PR #2452 Bugbot 指摘）。
    ("gemm.rs", "const", "TILED_PIPELINE_128X64_PRODUCTION_ENABLED"),
    ("gemm.rs", "const", "TILED_PIPELINE_128X64_MIN_N"),
    ("gemm.rs", "const", "TILED_PIPELINE_128X64_MIN_K"),
    ("gemm.rs", "fn", "tiled_pipeline_tile_kind"),
    ("gemm.rs", "fn", "tiled_pipeline_launch_config"),
    ("gemm.rs", "fn", "tiled_f32_launch_config"),
    ("gemm.rs", "fn", "select_tiled_f32_kernel"),
    ("gemm.rs", "fn", "select_tiled_pipeline_handle"),
    ("gemm.rs", "fn", "run_f32_kernel"),
    ("kernels.rs", "const", "TILED_F32"),
    ("kernels_tiled_pipeline.rs", "fn", "tiled_pipeline_f32_source"),
    ("kernels_tiled_pipeline.rs", "fn", "tiled_pipeline_f32_source_with_stages"),
    ("memory.rs", "fn", "pretouched_host_vec"),
    ("memory.rs", "fn", "readback"),
    ("memory.rs", "fn", "readback_with"),
    # 既定の宛先確保方式を決める定数（値の差し替えで経路が変わる）。
    ("memory.rs", "const", "READBACK_DEST"),
]
# tiled パイプラインのカーネルソース生成経路（64x64 と 128x64 の両方。gemm.rs の選択が
# どちらも起動しうる）。`*_source` 系は `render_source` を呼ぶだけの薄いラッパーのため、
# テンプレ断片・`#define` 群・ステージ数/タイル寸法定数・`render_source` 本体・
# LazyLock 静的変数まで比較しないとカーネル編集を検出できない（PR #2452 Bugbot 指摘）。
for _f, _p, _consts in (
    (
        "kernels_tiled_pipeline.rs",
        "TP",
        ["BM", "BN", "BK", "THREAD_M", "THREAD_N", "THREADS_X", "THREADS_Y", "BLOCK_THREADS",
         "DEFAULT_STAGES", "MIN_STAGES", "MAX_STAGES", "A_PAD", "B_PAD", "A_CHUNKS", "B_CHUNKS",
         "SMEM_BYTES_PER_STAGE"],
    ),
    (
        "kernels_tiled_pipeline_128x64.rs",
        "TP128",
        ["BM", "BN", "BK", "THREAD_M", "THREAD_N", "THREADS_X", "THREADS_Y", "BLOCK_THREADS",
         "DEFAULT_STAGES", "MIN_STAGES", "MAX_STAGES", "A_CHUNKS", "B_CHUNKS",
         "A_CHUNKS_PER_ROW", "SMEM_BYTES_PER_STAGE"],
    ),
):
    ITEMS += [(_f, "const", f"{_p}_{c}") for c in _consts]
    ITEMS += [(_f, "const", "MAX_WAIT_GROUP_IMMEDIATE")]
    ITEMS += [(_f, "const", f"{_p}_{c}") for c in
              ("CP_ASYNC_HELPER", "NON_PERSISTENT_PREFIX", "TILE_CORE", "KERNEL_SUFFIX")]
    ITEMS += [(_f, "fn", "render_defines"), (_f, "fn", "render_source")]
ITEMS += [
    ("kernels_tiled_pipeline.rs", "static", "TILED_PIPELINE_F32_SOURCE"),
    ("kernels_tiled_pipeline_128x64.rs", "static", "TILED_PIPELINE_128X64_F32_SOURCE"),
    ("kernels_tiled_pipeline_128x64.rs", "fn", "tiled_pipeline_128x64_f32_source"),
]
# カーネルの生成（NVRTC コンパイル・ロード）経路と本番起動経路。`CudaGemm::new`・
# `compile_tiled_pipeline*` は #2299 の feature gate（`context_ptr` の `let`・タプル→
# 構造体化）で v0.9.0 と文面が変わり、文面比較だけでは「同一」とも「実質差」とも
# 断定できない。差分が出た場合は `no` ではなく `unknown`（人手確認要）とし、
# `yes` にはしない（PR #2452 codex-review 指摘 P1。生成・起動経路が比較対象外のまま
# `layerA_same_code: yes` を返していた）。
CONSTRUCTION_ITEMS = [
    ("gemm.rs", "fn", "new"),
    ("gemm.rs", "fn", "kernel_specs"),
    ("gemm.rs", "fn", "tiled_pipeline_descriptor"),
    ("gemm.rs", "fn", "compile_tiled_pipeline"),
    ("gemm.rs", "fn", "compile_tiled_pipeline_128x64"),
    ("gemm.rs", "fn", "new_with_tiled_pipeline_128x64"),
    ("gemm.rs", "fn", "launch_tiled_f32"),
    ("module_cache.rs", "fn", "load_function_cached"),
]
GATE = '#[cfg(feature = "internal-diagnostics")]'


def show(ref: str, rel: str) -> str:
    return subprocess.run(
        ["git", "show", f"{ref}:{SRC}{rel}"],
        check=True, capture_output=True, text=True,
    ).stdout


def extract(text: str, kind: str, name: str) -> str | None:
    m = re.search(rf"^[ \t]*(?:pub(?:\([a-z]+\))?\s+)?{kind}\s+{re.escape(name)}\b", text, re.M)
    if not m:
        return None
    start = m.start()
    if kind in ("const", "static"):
        # 生文字列 `r#"..."#;` は本文中の `;` を含むため終端を `"#;` で探す。
        # それ以外（数値定数・`LazyLock` 静的変数）は最初の `;` まで。
        eq = text.find("=", m.end())
        if eq < 0:
            return None
        if text[eq + 1 :].lstrip().startswith('r#"'):
            end = text.find('"#;', eq)
            return None if end < 0 else text[start : end + 3]
        end = text.find(";", eq)
        return None if end < 0 else text[start : end + 1]
    depth, seen, i = 0, False, m.end()
    while i < len(text):
        c = text[i]
        if c == "{":
            depth += 1
            seen = True
        elif c == "}":
            depth -= 1
            if seen and depth == 0:
                return text[start : i + 1]
        i += 1
    return None


def normalize(body: str, kind: str) -> str:
    if kind == "const" and 'r#"' in body:
        return body  # カーネルソース本体はコメントも含め完全一致を要求する
    out: list[str] = []
    skipping, depth, seen = False, 0, False
    for raw in body.splitlines():
        s = raw.strip()
        if not skipping and s == GATE:
            skipping, depth, seen = True, 0, False
            continue
        # 本番既定 `READBACK_DEST` は常に `PretouchedFresh`。`Fresh` 腕は v0.9.0 では
        # 無条件・HEAD では feature gate 付きだが、いずれも本番経路では到達しない。
        if not skipping and s.startswith("ReadbackDest::Fresh =>"):
            skipping, depth, seen = True, 0, False
        if skipping:
            depth += s.count("{") - s.count("}")
            seen = seen or "{" in s
            if depth <= 0 and (s.endswith(",") or s.endswith(";") or (seen and s.endswith("}"))):
                skipping = False
            continue
        if not s or s.startswith("//"):
            continue
        out.append(re.sub(r"\s+", " ", s))
    return "\n".join(out)


def main() -> int:
    base = sys.argv[1] if len(sys.argv) > 1 else "v0.9.0"
    head = sys.argv[2] if len(sys.argv) > 2 else "HEAD"
    lines, verdict = [], "yes"
    try:
        for rel, kind, name in ITEMS:
            a = extract(show(base, rel), kind, name)
            b = extract(show(head, rel), kind, name)
            if a is None or b is None:
                lines.append(f"path_item: {rel}::{name} 抽出不能")
                if verdict == "yes":
                    verdict = "unknown"
            elif normalize(a, kind) == normalize(b, kind):
                lines.append(f"path_item: {rel}::{name} identical")
            else:
                lines.append(f"path_item: {rel}::{name} DIFFERS")
                verdict = "no"
        for rel, kind, name in CONSTRUCTION_ITEMS:
            a = extract(show(base, rel), kind, name)
            b = extract(show(head, rel), kind, name)
            if a is None or b is None:
                lines.append(f"path_item: {rel}::{name} 抽出不能（生成・起動経路）")
                if verdict == "yes":
                    verdict = "unknown"
            elif normalize(a, kind) == normalize(b, kind):
                lines.append(f"path_item: {rel}::{name} identical（生成・起動経路）")
            else:
                lines.append(f"path_item: {rel}::{name} DIFFERS（生成・起動経路。要人手確認）")
                if verdict == "yes":
                    verdict = "unknown"
    except (subprocess.CalledProcessError, OSError):
        print("layerA_same_code: unknown")
        return 0
    print(f"layerA_same_code: {verdict}")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())
