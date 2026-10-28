#!/usr/bin/env bash
# イシュー #3646 G-CPU-INF (iii) の補足確認（計測外・時間は測らない）。GB10 ノード上で実行する。
# 使い方: bash run_probe.sh <sgemm_probe.c のあるディレクトリ> <出力ファイル>
# PyTorch の mm（64x784 @ 784x256 と 64x256 @ 256x10。MLP infer の 2 層）を LD_PRELOAD shim 越しに 1 回ずつ実行し、
# sgemm_／cblas_sgemm が呼ばれるか・転送先ライブラリを記録する。値は RULE の固定形状だけを使う。
set -euo pipefail
src="$1"
out="$2"
work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT
gcc -O1 -shared -fPIC -o "$work/libprobe.so" "$src/sgemm_probe.c" -ldl
py="$HOME/work/.venv-bench/bin/python"
{
  echo "# gcc: $(gcc --version | head -n 1)"
  echo "# --- mm 2 回（l1_fwd 形・l2_fwd 形）。shim 無しの checksum と shim 有りの checksum を比較する"
  "$py" -c 'import torch; a=torch.ones(64,784); b=torch.ones(784,256); c=torch.mm(a,b); d=torch.mm(torch.ones(64,256), torch.ones(256,10)); print("plain", float(c.sum()), float(d.sum()))'
  LD_PRELOAD="$work/libprobe.so" "$py" -c 'import torch; a=torch.ones(64,784); b=torch.ones(784,256); c=torch.mm(a,b); d=torch.mm(torch.ones(64,256), torch.ones(256,10)); print("probed", float(c.sum()), float(d.sum()))'
  echo "# --- import torch のみ（mm なし）。mm 以外の sgemm_ 呼び出しの有無（基準）"
  LD_PRELOAD="$work/libprobe.so" "$py" -c 'import torch; print("import-only")'
} 2>&1 | sed -e "s#$HOME#<home>#g" >"$out"
