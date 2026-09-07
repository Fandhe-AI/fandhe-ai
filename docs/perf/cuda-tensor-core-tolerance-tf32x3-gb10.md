# 3×TF32（split-single 法）の誤差分布・スケールスイープ・純カーネル時間 GB10 実測（イシュー #1356）

## 0. 実測状況（重要）


**GB10 実機実測を完了した（2026-09-07。イシュー #1356 reopen 対応。CUDA_NODE は `docs/real-hardware-verification-env.local.md` から解決し本ドキュメント・ログには書いていない）**。§2〜§10 の記入欄を実測値で充填し、§11 で採否を確定した。

- **P1（`#[ignore]` テスト 2 件）は FAIL した**（`mma_tf32x3_matches_reference_across_shapes` は形状 512×512×512・seed 5004 で `assert_parity` 複合判定 FAIL。`mma_tf32x3_k4096_stress` も FAIL。§10 参照）。P1 不成立のため、本ドキュメントは §11 の 3 択語彙（「opt-in 維持・推奨」／「opt-in 維持・条件付き推奨」／「opt-in 維持・非推奨」）を選ばず、実測 `fail_count`／`max_abs_diff`／`max_rel_err` を「baseline 提案値（未承認）」として §11 に記録し、ユーザー判断へ回す（本ドキュメント §11「P1 が不成立の場合」の運用に従う）。
- P2（対 f64 精度。§7・§11 参照）は形状依存で部分的に不成立（K が支配的な形状〈256×256×1024・256×256×4096〉で `max_abs_diff` が f32 SIMT の 2 倍を超える）。
- P3（`s²` 比例・`max_rel_err` スケール不変性。§8）は 3 経路とも成立した。
- P4（純カーネル時間。§9）は 5 形状すべてで `mma_tf32x3 / f32_simt` < 1.0（0.721〜0.930 倍）であり、性能面のみで判断すれば「非推奨」に相当する信号だが、P1 不成立を受けて総合判定としては 3 択を選ばない。
- 誤差分布（`--routes mma`）は 2 回実行で決定性（`diff` 完全一致）を確認した。整列非対応 5 形状のスキップ（§5）も期待どおり確認した。

## 1. 位置づけ

- イシュー #1356「3×TF32 の誤差分布（対 CPU f32 FMA 参照・f64 参照）とスケールスイープ・純カーネル時間を GB10 で実測し f32 SIMT／TF32 と比較して採否を記録する」の実測記録。親ツリー #1354・承認元 #1338・依存イシュー #1355（PR #1400）。
- #1355（PR #1400）は 3×TF32（split-single 法。`CudaMmaTf32x3Gemm`）を precision の第 3 モード（既定 OFF・opt-in）として実装したが、GB10 実機実測は未実施のまま `docs/cuda-tf32x3-split-single-decision.md` §8 に記入欄のみを残していた。本ドキュメントはその実測記入欄を引き継ぐ独立ドキュメントであり、GB10 実機実測を完了した（2026-09-07。§0 参照）。`docs/cuda-tf32x3-split-single-decision.md` §8 は本ドキュメントへの参照リンク・結果要約へ差し替え済み。
- 手順は `docs/perf/cuda-tensor-core-tolerance-gb10-scale-sweep.md`（#995）と同一プロトコル（`wmma_tolerance_probe` の `--scales 0.1,1,10,100`・15 形状 × 5 シード・`fandhe_ai_backend_cpu::compare` 統計）を `--routes mma` へ拡張して踏襲する。
- **閾値定数（`RELATIVE_TOLERANCE`＝1e-3・`ABSOLUTE_RESCUE_THRESHOLD`＝1e-5）・判定式・テスト許容誤差は本ドキュメントでは一切変更していない**（`.claude/rules/coding-rust.md`・`.claude/rules/security.md` A08）。f64 参照行は診断行であり REQ-2 判定には使わない（`wmma_tolerance_probe.rs` ファイル冒頭コメント参照）。

## 2. 計測環境

| 項目 | 値 |
|---|---|
| GPU | NVIDIA GB10（compute_121 / sm_121） |
| driver | 580.173.02（`nvidia-smi` 表示） |
| compute capability | 12.1（`compute_121`。プローブ・ベンチ両方の `device compute capability:` 行で確認） |
| CUDA SDK | 13.0（`nvcc` release 13.0, V13.0.88, build cuda_13.0.r13.0/compiler.36424714_0） |
| rustc | 1.97.0 (2d8144b78 2026-07-07) |
| cargo | 1.97.0 (c980f4866 2026-06-30) |
| ビルド feature | `--release --features internal-diagnostics`（`--example wmma_tolerance_probe --example gemm_tf32x3_kernel_time_bench`） |
| 計測対象コミット | `9c3b286eef0c53df6acf16dde70413f698531219`（`.rev-stamp`。PR #1404 マージ後の origin/main HEAD + 本イシューの §3.2／§3.7 doc 更新〈未コミットのまま転送〉） |
| 計測日 | 2026-09-07 |
| GPU 空き確認（計測前） | `utilization.gpu` = 0%、`nvidia-smi --query-compute-apps` は常駐サービス 2 件のみ（ComfyUI 170 MiB・Kokoro 870 MiB。停止せず） |
| GPU 空き確認（計測後） | `utilization.gpu` = 0%、compute-apps 件数変化なし（他プロセス混入なし） |
| `uptime` / load average（計測前） | `up 11 days, 6:58, 5 users, load average: 0.16, 1.09, 1.02` |
| `uptime` / load average（計測後） | `up 11 days, 7:02, 5 users, load average: 0.54, 0.83, 0.93` |

環境詳細・生ログは `docs/perf/logs/cuda-gemm-tf32x3-1356/`（内部ホスト名・ユーザー名は含めない）を参照。

## 3. 再現手順

`docs/real-hardware-verification-env.md` §3（rsync 転送）・§4.4（`setsid nohup` 切り離し）・§6（GPU 空き確認）に準拠する。ノード実名は環境変数 `$CUDA_NODE`（`docs/real-hardware-verification-env.local.md`。Git 管理外）経由で渡し、本ドキュメント・ログには一切書かない。

### 3.1 転送

```sh
git rev-parse HEAD > .rev-stamp
rsync -a --delete --delete-excluded --filter=':- .gitignore' \
  --exclude '.git/' --exclude '.codex/' --exclude '.env*' \
  --exclude '.claude/settings.local.json' --exclude '.venv*/' \
  --exclude 'real-hardware-verification-env.local.md' \
  ./ "$CUDA_NODE":~/work/rust-ai-library-run/
rm .rev-stamp
```

### 3.2 GPU 空き確認・ビルド

```sh
ssh "$CUDA_NODE" 'nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader'
ssh "$CUDA_NODE" 'nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader'
ssh "$CUDA_NODE" 'uptime'

ssh "$CUDA_NODE" '
  cd ~/work/rust-ai-library-run && \
  env PATH=$HOME/.cargo/bin:/usr/local/cuda/bin:$PATH \
      CARGO_TARGET_DIR=$HOME/work/target-fandhe-ai \
      cargo build --release -p fandhe-ai-backend-cuda --features internal-diagnostics \
      --example wmma_tolerance_probe --example gemm_tf32x3_kernel_time_bench
'
```

`--features internal-diagnostics` は §4「実行カーネル種別の実測」の `kernel` 列（`f32 SIMT` 節の `Classic`／`Pipeline` 判別）を有効化するために付与する（`wmma_tolerance_probe.rs::f32_simt_kernel_kind` の設計。feature 非有効時は `n/a (internal-diagnostics)` 固定になる）。

### 3.3 誤差分布（`--routes mma`。2 回実行し決定性を確認）

```sh
BIN='$HOME/work/target-fandhe-ai/release/examples/wmma_tolerance_probe'
ssh "$CUDA_NODE" "env PATH=\$HOME/.cargo/bin:/usr/local/cuda/bin:\$PATH \"$BIN\" \
  --scales 0.1,1,10,100 --routes mma > ~/work/probe-mma-run1.md"
ssh "$CUDA_NODE" "env PATH=\$HOME/.cargo/bin:/usr/local/cuda/bin:\$PATH \"$BIN\" \
  --scales 0.1,1,10,100 --routes mma > ~/work/probe-mma-run2.md"
ssh "$CUDA_NODE" 'diff ~/work/probe-mma-run1.md ~/work/probe-mma-run2.md && echo IDENTICAL'
```

### 3.4 `#[ignore]` テスト（P1: REQ-2 判定行の直接判定）

```sh
ssh "$CUDA_NODE" "cd ~/work/rust-ai-library-run && \
  env PATH=\$HOME/.cargo/bin:/usr/local/cuda/bin:\$PATH \
  CARGO_TARGET_DIR=\$HOME/work/target-fandhe-ai \
  cargo test -p fandhe-ai-backend-cuda --release --all-features \
  --test gemm_mma_tf32x3 -- --ignored --nocapture > ~/work/ignored-gemm_mma_tf32x3.log"
```

### 3.5 ローカルでの事前確認（本エージェント実行環境。CUDA 非搭載のため exit 0 のスキップ経路のみ確認）

```sh
cargo build --release -p fandhe-ai-backend-cuda \
  --example wmma_tolerance_probe --example gemm_tf32x3_kernel_time_bench
./target/release/examples/wmma_tolerance_probe --routes mma   # => CUDA driver 非搭載のためスキップ（exit 0）
./target/release/examples/wmma_tolerance_probe                # => 変更前と byte 単位で同一（AC-2）
./target/release/examples/wmma_tolerance_probe --routes bogus # => usage 表示・exit 1（fail-closed）
cargo test -p fandhe-ai-backend-cuda --all-features \
  --example wmma_tolerance_probe --example gemm_tf32x3_kernel_time_bench
```

上記はいずれもローカル（CUDA 非搭載）で確認済み（50 件の純粋関数テスト green・AC-2 の `diff` 一致・`--routes bogus` の exit 1 を確認）。GB10 実機での実行結果は未取得。

### 3.6 純カーネル時間（5 回プロセス起動）

```sh
BIN='$HOME/work/target-fandhe-ai/release/examples/gemm_tf32x3_kernel_time_bench'
for i in 1 2 3 4 5; do
  ssh "$CUDA_NODE" "nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader"
  ssh "$CUDA_NODE" "env PATH=\$HOME/.cargo/bin:/usr/local/cuda/bin:\$PATH \"$BIN\" \
    > ~/work/bench-run${i}.csv"
done
```

### 3.7 集計スクリプト（Python3 標準ライブラリのみ）

`docs/perf/cuda-tensor-core-tolerance-gb10-scale-sweep.md` §3 の `parse_file`／`dedupe_256`／`agg_max` を、17 列表（`ref` 列付き）・`## f32 SIMT`／`## TF32 mma.sync`／`## 3xTF32` の 3 節へ拡張したもの。実測後、生ログから機械的に §6〜§9 の表を再現できる。

```python
import math

def parse_file(path):
    rows = []
    route = None
    with open(path) as f:
        for line in f:
            if line.startswith("## f32 SIMT"):
                route = "f32_simt"; continue
            if line.startswith("## TF32 mma.sync"):
                route = "mma_tf32"; continue
            if line.startswith("## 3xTF32"):
                route = "mma_tf32x3"; continue
            if not line.startswith("| ") or line.startswith("| scale |") or line.startswith("|---"):
                continue
            cols = [c.strip() for c in line.strip("|\n").split("|")]
            if len(cols) != 17:
                continue
            scale, shape, seed, ref_label = cols[0], cols[1], cols[2], cols[3]
            failtotal = cols[4]
            if failtotal.startswith("("):
                continue  # skipped / unexpected error 行は集計対象外
            fail, total = map(int, failtotal.split("/"))
            def pf(x):
                x = x.strip()
                return None if x in ("n/a", "NaN") else float(x)
            rows.append(dict(route=route, scale=scale, shape=shape, ref=ref_label,
                fail=fail, total=total, max_abs=pf(cols[5]),
                max_rel=pf(cols[7]), max_fail_abs=pf(cols[12]), kernel=cols[15]))
    return rows

def dedupe_256(rows):
    # `wmma_tolerance_probe.rs::SHAPES` は "256x256x256 (block tile x8)" と
    # "256x256x256 (K sweep base)" の 2 定義を意図的に持つ（`m` 由来のシード
    # 導出式により同一入力・同一結果になる重複。#995 §3 と同じマージ方針）。
    # 集計前に片方を破棄しないと total が二重計上される。
    return [r for r in rows if r["shape"] != "256x256x256 (block tile x8)"]

def agg_max(values):
    finite = [v for v in values if v is not None]
    if not finite:
        return None
    if any(math.isinf(v) for v in finite):
        return float("inf")
    return max(finite)
```

§9 の純カーネル時間表は `gemm_tf32x3_kernel_time_bench` の 5 run CSV（`route,m,n,k,kernel,median_ms,q1_ms,q3_ms,tflops`。`#` 始まりは route skip 行のため除外）を以下で集計する（`statistics.median` で 5 run の `median_ms`／`tflops` をさらに中央値化。`.claude/rules/coding-rust.md` の「5 回計測の中央値を採用」に従う）:

```python
import csv
import statistics
from collections import defaultdict

def load_bench_csv(path):
    rows = []
    with open(path, newline="") as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            if line.startswith("route,"):
                continue  # ヘッダ行
            cols = line.rstrip("\n").split(",")
            if len(cols) != 9:
                continue
            route, m, n, k, kernel, median_ms, q1_ms, q3_ms, tflops = cols
            rows.append(dict(route=route, shape=f"{m}x{n}x{k}", kernel=kernel,
                median_ms=float(median_ms), tflops=float(tflops)))
    return rows

def aggregate_bench_runs(paths):
    # (route, shape) ごとに 5 run 分の値を集める
    grouped = defaultdict(list)
    for p in paths:
        for r in load_bench_csv(p):
            grouped[(r["route"], r["shape"])].append(r)
    out = {}
    for key, entries in grouped.items():
        out[key] = dict(
            kernel=entries[0]["kernel"],
            median_ms=statistics.median(e["median_ms"] for e in entries),
            tflops=statistics.median(e["tflops"] for e in entries),
            n_runs=len(entries),
        )
    return out

# 使用例:
# agg = aggregate_bench_runs([f"bench-run{i}.csv" for i in range(1, 6)])
# shape ごとに agg[("f32_simt", shape)]["tflops"] 等を参照し、
# tf32x3/f32_simt・tf32x3/mma_tf32 の比を §9 表へ転記する。
```

## 4. 実行カーネル種別の実測

実測（`internal-diagnostics` feature 有効時）:

- `f32 SIMT` 節の `kernel` 列は全形状・全スケールで `Pipeline` 固定だった（`Classic` への降格は観測されなかった）。
- `TF32 mma.sync` 節の `kernel` 列は全形状で `mma_tf32` 固定。
- `3xTF32` 節の `kernel` 列は全形状で `mma_tf32x3` 固定。

想定どおり単一カーネルのため多段選択の分岐は観測されない（`wmma_tolerance_probe.rs::f32_simt_kernel_kind` の設計と一致）。

## 5. 整列非対応形状のスキップ確認

実測で確認済み。`SHAPES` のうち `n % 4 == 0 && k % 4 == 0` を満たさない 5 形状（1x1x1・17x23x19・17x19x23・33x31x65・130x70x90）は `TF32 mma.sync`／`3xTF32` の両節で全スケール・全シード `(skipped: alignment n%4/k%4)` 行になることを生ログ（`probe-mma-run1.md`）で確認した。`f32 SIMT` 節は整列制約がないため 15 形状すべてで実測値を持つ。

## 6. 誤差分布表（対 f32fma。REQ-2 判定行）

閾値（REQ-2 判定・変更対象外）: `RELATIVE_TOLERANCE = 1e-3`、`ABSOLUTE_RESCUE_THRESHOLD = 1e-5`。5 シード合計。整列非対応形状（mma 系 2 経路のみ）は `(skipped)` と表示する。集計は §3.7 のスクリプト（`aggregate.py`／`gen_tables.py` 相当。`docs/perf/logs/cuda-gemm-tf32x3-1356/` に保存済み）で生ログから再現可能。

**要点**: `f32_simt` は `f32fma` 参照との比較で全形状・全スケール fail=0（同一計算経路のため）。`mma_tf32`（単発 TF32）は形状が大きいほど fail 率・`max_abs_diff` が増大する（例: 256x256x4096 s=100 で fail 60229/327680・`max_abs_diff`=2.910e2）。`mma_tf32x3`（3×TF32）は同条件で fail 2876/327680・`max_abs_diff`=3.812e1 と大幅に改善するが、fail=0 にはならない（P1 で確認した `#[ignore]` テストの FAIL と整合する）。

### f32fma（REQ-2 判定行） 対 f32_simt

| 形状 | scale | fail/total（5 シード） | max_abs_diff | max_rel_err | max_fail_abs_diff |
|---|---|---|---|---|---|
| 1x1x1 (sub-K-tile, TF32 suite) | 0.1 | 0/5 | 0.000e+00 | 0.000e+00 | 0.000e+00 |
| 1x1x1 (sub-K-tile, TF32 suite) | 1 | 0/5 | 0.000e+00 | 0.000e+00 | 0.000e+00 |
| 1x1x1 (sub-K-tile, TF32 suite) | 10 | 0/5 | 0.000e+00 | 0.000e+00 | 0.000e+00 |
| 1x1x1 (sub-K-tile, TF32 suite) | 100 | 0/5 | 0.000e+00 | 0.000e+00 | 0.000e+00 |
| 32x32x32 (block tile) | 0.1 | 0/5120 | 0.000e+00 | 0.000e+00 | 0.000e+00 |
| 32x32x32 (block tile) | 1 | 0/5120 | 0.000e+00 | 0.000e+00 | 0.000e+00 |
| 32x32x32 (block tile) | 10 | 0/5120 | 0.000e+00 | 0.000e+00 | 0.000e+00 |
| 32x32x32 (block tile) | 100 | 0/5120 | 0.000e+00 | 0.000e+00 | 0.000e+00 |
| 64x64x64 (block tile x2) | 0.1 | 0/20480 | 0.000e+00 | 0.000e+00 | 0.000e+00 |
| 64x64x64 (block tile x2) | 1 | 0/20480 | 0.000e+00 | 0.000e+00 | 0.000e+00 |
| 64x64x64 (block tile x2) | 10 | 0/20480 | 0.000e+00 | 0.000e+00 | 0.000e+00 |
| 64x64x64 (block tile x2) | 100 | 0/20480 | 0.000e+00 | 0.000e+00 | 0.000e+00 |
| 128x128x128 (block tile x4) | 0.1 | 0/81920 | 0.000e+00 | 0.000e+00 | 0.000e+00 |
| 128x128x128 (block tile x4) | 1 | 0/81920 | 0.000e+00 | 0.000e+00 | 0.000e+00 |
| 128x128x128 (block tile x4) | 10 | 0/81920 | 0.000e+00 | 0.000e+00 | 0.000e+00 |
| 128x128x128 (block tile x4) | 100 | 0/81920 | 0.000e+00 | 0.000e+00 | 0.000e+00 |
| 256x256x256 (K sweep base) | 0.1 | 0/327680 | 0.000e+00 | 0.000e+00 | 0.000e+00 |
| 256x256x256 (K sweep base) | 1 | 0/327680 | 0.000e+00 | 0.000e+00 | 0.000e+00 |
| 256x256x256 (K sweep base) | 10 | 0/327680 | 0.000e+00 | 0.000e+00 | 0.000e+00 |
| 256x256x256 (K sweep base) | 100 | 0/327680 | 0.000e+00 | 0.000e+00 | 0.000e+00 |
| 512x512x512 (block tile x16) | 0.1 | 0/1310720 | 0.000e+00 | 0.000e+00 | 0.000e+00 |
| 512x512x512 (block tile x16) | 1 | 0/1310720 | 0.000e+00 | 0.000e+00 | 0.000e+00 |
| 512x512x512 (block tile x16) | 10 | 0/1310720 | 0.000e+00 | 0.000e+00 | 0.000e+00 |
| 512x512x512 (block tile x16) | 100 | 0/1310720 | 0.000e+00 | 0.000e+00 | 0.000e+00 |
| 17x23x19 (non-multiple edge, TF32 suite) | 0.1 | 0/1955 | 0.000e+00 | 0.000e+00 | 0.000e+00 |
| 17x23x19 (non-multiple edge, TF32 suite) | 1 | 0/1955 | 0.000e+00 | 0.000e+00 | 0.000e+00 |
| 17x23x19 (non-multiple edge, TF32 suite) | 10 | 0/1955 | 0.000e+00 | 0.000e+00 | 0.000e+00 |
| 17x23x19 (non-multiple edge, TF32 suite) | 100 | 0/1955 | 0.000e+00 | 0.000e+00 | 0.000e+00 |
| 17x19x23 (non-multiple edge, f16 suite) | 0.1 | 0/1615 | 0.000e+00 | 0.000e+00 | 0.000e+00 |
| 17x19x23 (non-multiple edge, f16 suite) | 1 | 0/1615 | 0.000e+00 | 0.000e+00 | 0.000e+00 |
| 17x19x23 (non-multiple edge, f16 suite) | 10 | 0/1615 | 0.000e+00 | 0.000e+00 | 0.000e+00 |
| 17x19x23 (non-multiple edge, f16 suite) | 100 | 0/1615 | 0.000e+00 | 0.000e+00 | 0.000e+00 |
| 33x31x65 (non-multiple edge) | 0.1 | 0/5115 | 0.000e+00 | 0.000e+00 | 0.000e+00 |
| 33x31x65 (non-multiple edge) | 1 | 0/5115 | 0.000e+00 | 0.000e+00 | 0.000e+00 |
| 33x31x65 (non-multiple edge) | 10 | 0/5115 | 0.000e+00 | 0.000e+00 | 0.000e+00 |
| 33x31x65 (non-multiple edge) | 100 | 0/5115 | 0.000e+00 | 0.000e+00 | 0.000e+00 |
| 100x100x100 (non-multiple edge) | 0.1 | 0/50000 | 0.000e+00 | 0.000e+00 | 0.000e+00 |
| 100x100x100 (non-multiple edge) | 1 | 0/50000 | 0.000e+00 | 0.000e+00 | 0.000e+00 |
| 100x100x100 (non-multiple edge) | 10 | 0/50000 | 0.000e+00 | 0.000e+00 | 0.000e+00 |
| 100x100x100 (non-multiple edge) | 100 | 0/50000 | 0.000e+00 | 0.000e+00 | 0.000e+00 |
| 130x70x90 (non-multiple edge) | 0.1 | 0/45500 | 0.000e+00 | 0.000e+00 | 0.000e+00 |
| 130x70x90 (non-multiple edge) | 1 | 0/45500 | 0.000e+00 | 0.000e+00 | 0.000e+00 |
| 130x70x90 (non-multiple edge) | 10 | 0/45500 | 0.000e+00 | 0.000e+00 | 0.000e+00 |
| 130x70x90 (non-multiple edge) | 100 | 0/45500 | 0.000e+00 | 0.000e+00 | 0.000e+00 |
| 64x96x128 (non-square) | 0.1 | 0/30720 | 0.000e+00 | 0.000e+00 | 0.000e+00 |
| 64x96x128 (non-square) | 1 | 0/30720 | 0.000e+00 | 0.000e+00 | 0.000e+00 |
| 64x96x128 (non-square) | 10 | 0/30720 | 0.000e+00 | 0.000e+00 | 0.000e+00 |
| 64x96x128 (non-square) | 100 | 0/30720 | 0.000e+00 | 0.000e+00 | 0.000e+00 |
| 256x256x512 (K sweep) | 0.1 | 0/327680 | 0.000e+00 | 0.000e+00 | 0.000e+00 |
| 256x256x512 (K sweep) | 1 | 0/327680 | 0.000e+00 | 0.000e+00 | 0.000e+00 |
| 256x256x512 (K sweep) | 10 | 0/327680 | 0.000e+00 | 0.000e+00 | 0.000e+00 |
| 256x256x512 (K sweep) | 100 | 0/327680 | 0.000e+00 | 0.000e+00 | 0.000e+00 |
| 256x256x1024 (K sweep) | 0.1 | 0/327680 | 0.000e+00 | 0.000e+00 | 0.000e+00 |
| 256x256x1024 (K sweep) | 1 | 0/327680 | 0.000e+00 | 0.000e+00 | 0.000e+00 |
| 256x256x1024 (K sweep) | 10 | 0/327680 | 0.000e+00 | 0.000e+00 | 0.000e+00 |
| 256x256x1024 (K sweep) | 100 | 0/327680 | 0.000e+00 | 0.000e+00 | 0.000e+00 |
| 256x256x4096 (K sweep, PoC-v2-5 stress) | 0.1 | 0/327680 | 0.000e+00 | 0.000e+00 | 0.000e+00 |
| 256x256x4096 (K sweep, PoC-v2-5 stress) | 1 | 0/327680 | 0.000e+00 | 0.000e+00 | 0.000e+00 |
| 256x256x4096 (K sweep, PoC-v2-5 stress) | 10 | 0/327680 | 0.000e+00 | 0.000e+00 | 0.000e+00 |
| 256x256x4096 (K sweep, PoC-v2-5 stress) | 100 | 0/327680 | 0.000e+00 | 0.000e+00 | 0.000e+00 |

### f32fma（REQ-2 判定行） 対 mma_tf32

| 形状 | scale | fail/total（5 シード） | max_abs_diff | max_rel_err | max_fail_abs_diff |
|---|---|---|---|---|---|
| 1x1x1 (sub-K-tile, TF32 suite) | 0.1 | (skipped: alignment n%4/k%4) | - | - | - |
| 1x1x1 (sub-K-tile, TF32 suite) | 1 | (skipped: alignment n%4/k%4) | - | - | - |
| 1x1x1 (sub-K-tile, TF32 suite) | 10 | (skipped: alignment n%4/k%4) | - | - | - |
| 1x1x1 (sub-K-tile, TF32 suite) | 100 | (skipped: alignment n%4/k%4) | - | - | - |
| 32x32x32 (block tile) | 0.1 | 176/5120 | 2.211e-05 | 5.585e-01 | 2.211e-05 |
| 32x32x32 (block tile) | 1 | 807/5120 | 1.857e-03 | 2.556e-01 | 1.857e-03 |
| 32x32x32 (block tile) | 10 | 936/5120 | 2.700e-01 | 7.124e-01 | 1.917e-01 |
| 32x32x32 (block tile) | 100 | 945/5120 | 2.031e+01 | 9.061e-01 | 2.031e+01 |
| 64x64x64 (block tile x2) | 0.1 | 1587/20480 | 3.366e-05 | 6.078e-01 | 3.366e-05 |
| 64x64x64 (block tile x2) | 1 | 3374/20480 | 3.377e-03 | 1.046e+00 | 3.377e-03 |
| 64x64x64 (block tile x2) | 10 | 3999/20480 | 3.503e-01 | 1.186e+00 | 3.181e-01 |
| 64x64x64 (block tile x2) | 100 | 3821/20480 | 3.554e+01 | 6.074e-01 | 3.554e+01 |
| 128x128x128 (block tile x4) | 0.1 | 9853/81920 | 5.199e-05 | 1.466e+00 | 5.034e-05 |
| 128x128x128 (block tile x4) | 1 | 13140/81920 | 4.336e-03 | 1.911e+00 | 4.225e-03 |
| 128x128x128 (block tile x4) | 10 | 15521/81920 | 5.544e-01 | 1.949e+00 | 5.061e-01 |
| 128x128x128 (block tile x4) | 100 | 15143/81920 | 4.986e+01 | 1.848e+00 | 4.986e+01 |
| 256x256x256 (K sweep base) | 0.1 | 48557/327680 | 7.097e-05 | 1.958e+00 | 7.097e-05 |
| 256x256x256 (K sweep base) | 1 | 53426/327680 | 6.828e-03 | 1.925e+00 | 6.828e-03 |
| 256x256x256 (K sweep base) | 10 | 62631/327680 | 8.137e-01 | 1.966e+00 | 8.137e-01 |
| 256x256x256 (K sweep base) | 100 | 60426/327680 | 8.010e+01 | 1.910e+00 | 8.010e+01 |
| 512x512x512 (block tile x16) | 0.1 | 214260/1310720 | 1.099e-04 | 1.972e+00 | 1.099e-04 |
| 512x512x512 (block tile x16) | 1 | 212553/1310720 | 9.985e-03 | 1.984e+00 | 9.985e-03 |
| 512x512x512 (block tile x16) | 10 | 249591/1310720 | 1.254e+00 | 1.991e+00 | 1.254e+00 |
| 512x512x512 (block tile x16) | 100 | 239348/1310720 | 1.120e+02 | 1.984e+00 | 1.120e+02 |
| 17x23x19 (non-multiple edge, TF32 suite) | 0.1 | (skipped: alignment n%4/k%4) | - | - | - |
| 17x23x19 (non-multiple edge, TF32 suite) | 1 | (skipped: alignment n%4/k%4) | - | - | - |
| 17x23x19 (non-multiple edge, TF32 suite) | 10 | (skipped: alignment n%4/k%4) | - | - | - |
| 17x23x19 (non-multiple edge, TF32 suite) | 100 | (skipped: alignment n%4/k%4) | - | - | - |
| 17x19x23 (non-multiple edge, f16 suite) | 0.1 | (skipped: alignment n%4/k%4) | - | - | - |
| 17x19x23 (non-multiple edge, f16 suite) | 1 | (skipped: alignment n%4/k%4) | - | - | - |
| 17x19x23 (non-multiple edge, f16 suite) | 10 | (skipped: alignment n%4/k%4) | - | - | - |
| 17x19x23 (non-multiple edge, f16 suite) | 100 | (skipped: alignment n%4/k%4) | - | - | - |
| 33x31x65 (non-multiple edge) | 0.1 | (skipped: alignment n%4/k%4) | - | - | - |
| 33x31x65 (non-multiple edge) | 1 | (skipped: alignment n%4/k%4) | - | - | - |
| 33x31x65 (non-multiple edge) | 10 | (skipped: alignment n%4/k%4) | - | - | - |
| 33x31x65 (non-multiple edge) | 100 | (skipped: alignment n%4/k%4) | - | - | - |
| 100x100x100 (non-multiple edge) | 0.1 | 5108/50000 | 4.027e-05 | 1.934e+00 | 4.027e-05 |
| 100x100x100 (non-multiple edge) | 1 | 7956/50000 | 3.625e-03 | 1.884e+00 | 3.625e-03 |
| 100x100x100 (non-multiple edge) | 10 | 9314/50000 | 4.395e-01 | 1.772e+00 | 4.395e-01 |
| 100x100x100 (non-multiple edge) | 100 | 9044/50000 | 4.146e+01 | 1.998e+00 | 4.146e+01 |
| 130x70x90 (non-multiple edge) | 0.1 | (skipped: alignment n%4/k%4) | - | - | - |
| 130x70x90 (non-multiple edge) | 1 | (skipped: alignment n%4/k%4) | - | - | - |
| 130x70x90 (non-multiple edge) | 10 | (skipped: alignment n%4/k%4) | - | - | - |
| 130x70x90 (non-multiple edge) | 100 | (skipped: alignment n%4/k%4) | - | - | - |
| 64x96x128 (non-square) | 0.1 | 3586/30720 | 4.596e-05 | 1.290e+00 | 4.596e-05 |
| 64x96x128 (non-square) | 1 | 5067/30720 | 4.179e-03 | 1.743e+00 | 4.178e-03 |
| 64x96x128 (non-square) | 10 | 5966/30720 | 5.252e-01 | 1.489e+00 | 4.692e-01 |
| 64x96x128 (non-square) | 100 | 5676/30720 | 4.828e+01 | 1.848e+00 | 4.828e+01 |
| 256x256x512 (K sweep) | 0.1 | 53660/327680 | 1.083e-04 | 1.911e+00 | 1.011e-04 |
| 256x256x512 (K sweep) | 1 | 52794/327680 | 9.237e-03 | 1.952e+00 | 9.237e-03 |
| 256x256x512 (K sweep) | 10 | 62101/327680 | 1.129e+00 | 1.924e+00 | 1.129e+00 |
| 256x256x512 (K sweep) | 100 | 60069/327680 | 1.085e+02 | 2.000e+00 | 1.085e+02 |
| 256x256x1024 (K sweep) | 0.1 | 56494/327680 | 1.510e-04 | 1.958e+00 | 1.510e-04 |
| 256x256x1024 (K sweep) | 1 | 53474/327680 | 1.296e-02 | 1.891e+00 | 1.296e-02 |
| 256x256x1024 (K sweep) | 10 | 63059/327680 | 1.542e+00 | 1.958e+00 | 1.542e+00 |
| 256x256x1024 (K sweep) | 100 | 60311/327680 | 1.499e+02 | 1.913e+00 | 1.499e+02 |
| 256x256x4096 (K sweep, PoC-v2-5 stress) | 0.1 | 59400/327680 | 3.074e-04 | 1.961e+00 | 3.074e-04 |
| 256x256x4096 (K sweep, PoC-v2-5 stress) | 1 | 53251/327680 | 2.532e-02 | 1.969e+00 | 2.532e-02 |
| 256x256x4096 (K sweep, PoC-v2-5 stress) | 10 | 62807/327680 | 3.103e+00 | 1.973e+00 | 3.103e+00 |
| 256x256x4096 (K sweep, PoC-v2-5 stress) | 100 | 60229/327680 | 2.910e+02 | 1.930e+00 | 2.910e+02 |

### f32fma（REQ-2 判定行） 対 mma_tf32x3

| 形状 | scale | fail/total（5 シード） | max_abs_diff | max_rel_err | max_fail_abs_diff |
|---|---|---|---|---|---|
| 1x1x1 (sub-K-tile, TF32 suite) | 0.1 | (skipped: alignment n%4/k%4) | - | - | - |
| 1x1x1 (sub-K-tile, TF32 suite) | 1 | (skipped: alignment n%4/k%4) | - | - | - |
| 1x1x1 (sub-K-tile, TF32 suite) | 10 | (skipped: alignment n%4/k%4) | - | - | - |
| 1x1x1 (sub-K-tile, TF32 suite) | 100 | (skipped: alignment n%4/k%4) | - | - | - |
| 32x32x32 (block tile) | 0.1 | 0/5120 | 2.608e-08 | 4.990e-04 | 0.000e+00 |
| 32x32x32 (block tile) | 1 | 0/5120 | 2.861e-06 | 5.407e-04 | 0.000e+00 |
| 32x32x32 (block tile) | 10 | 0/5120 | 3.052e-04 | 3.773e-04 | 0.000e+00 |
| 32x32x32 (block tile) | 100 | 0/5120 | 3.125e-02 | 4.888e-04 | 0.000e+00 |
| 64x64x64 (block tile x2) | 0.1 | 0/20480 | 8.196e-08 | 9.666e-03 | 0.000e+00 |
| 64x64x64 (block tile x2) | 1 | 0/20480 | 8.583e-06 | 2.061e-02 | 0.000e+00 |
| 64x64x64 (block tile x2) | 10 | 2/20480 | 7.935e-04 | 1.787e-02 | 1.335e-04 |
| 64x64x64 (block tile x2) | 100 | 2/20480 | 8.594e-02 | 1.613e-02 | 1.055e-02 |
| 128x128x128 (block tile x4) | 0.1 | 0/81920 | 2.086e-07 | 3.396e-02 | 0.000e+00 |
| 128x128x128 (block tile x4) | 1 | 0/81920 | 2.098e-05 | 3.658e-02 | 0.000e+00 |
| 128x128x128 (block tile x4) | 10 | 30/81920 | 2.197e-03 | 2.606e-02 | 5.428e-04 |
| 128x128x128 (block tile x4) | 100 | 25/81920 | 1.875e-01 | 2.110e-02 | 6.610e-02 |
| 256x256x256 (K sweep base) | 0.1 | 0/327680 | 6.109e-07 | 1.112e+00 | 0.000e+00 |
| 256x256x256 (K sweep base) | 1 | 29/327680 | 5.913e-05 | 1.170e+00 | 1.918e-05 |
| 256x256x256 (K sweep base) | 10 | 182/327680 | 6.104e-03 | 9.572e-01 | 1.675e-03 |
| 256x256x256 (K sweep base) | 100 | 163/327680 | 5.781e-01 | 8.429e-01 | 1.738e-01 |
| 512x512x512 (block tile x16) | 0.1 | 0/1310720 | 1.639e-06 | 4.628e-01 | 0.000e+00 |
| 512x512x512 (block tile x16) | 1 | 1123/1310720 | 1.850e-04 | 4.596e-01 | 5.864e-05 |
| 512x512x512 (block tile x16) | 10 | 1488/1310720 | 1.782e-02 | 6.208e-01 | 5.802e-03 |
| 512x512x512 (block tile x16) | 100 | 1503/1310720 | 1.719e+00 | 3.975e-01 | 6.164e-01 |
| 17x23x19 (non-multiple edge, TF32 suite) | 0.1 | (skipped: alignment n%4/k%4) | - | - | - |
| 17x23x19 (non-multiple edge, TF32 suite) | 1 | (skipped: alignment n%4/k%4) | - | - | - |
| 17x23x19 (non-multiple edge, TF32 suite) | 10 | (skipped: alignment n%4/k%4) | - | - | - |
| 17x23x19 (non-multiple edge, TF32 suite) | 100 | (skipped: alignment n%4/k%4) | - | - | - |
| 17x19x23 (non-multiple edge, f16 suite) | 0.1 | (skipped: alignment n%4/k%4) | - | - | - |
| 17x19x23 (non-multiple edge, f16 suite) | 1 | (skipped: alignment n%4/k%4) | - | - | - |
| 17x19x23 (non-multiple edge, f16 suite) | 10 | (skipped: alignment n%4/k%4) | - | - | - |
| 17x19x23 (non-multiple edge, f16 suite) | 100 | (skipped: alignment n%4/k%4) | - | - | - |
| 33x31x65 (non-multiple edge) | 0.1 | (skipped: alignment n%4/k%4) | - | - | - |
| 33x31x65 (non-multiple edge) | 1 | (skipped: alignment n%4/k%4) | - | - | - |
| 33x31x65 (non-multiple edge) | 10 | (skipped: alignment n%4/k%4) | - | - | - |
| 33x31x65 (non-multiple edge) | 100 | (skipped: alignment n%4/k%4) | - | - | - |
| 100x100x100 (non-multiple edge) | 0.1 | 0/50000 | 1.714e-07 | 6.221e-03 | 0.000e+00 |
| 100x100x100 (non-multiple edge) | 1 | 0/50000 | 1.717e-05 | 3.473e-03 | 0.000e+00 |
| 100x100x100 (non-multiple edge) | 10 | 11/50000 | 1.587e-03 | 3.961e-03 | 2.884e-04 |
| 100x100x100 (non-multiple edge) | 100 | 13/50000 | 1.562e-01 | 2.801e-03 | 3.311e-02 |
| 130x70x90 (non-multiple edge) | 0.1 | (skipped: alignment n%4/k%4) | - | - | - |
| 130x70x90 (non-multiple edge) | 1 | (skipped: alignment n%4/k%4) | - | - | - |
| 130x70x90 (non-multiple edge) | 10 | (skipped: alignment n%4/k%4) | - | - | - |
| 130x70x90 (non-multiple edge) | 100 | (skipped: alignment n%4/k%4) | - | - | - |
| 64x96x128 (non-square) | 0.1 | 0/30720 | 1.788e-07 | 6.972e-03 | 0.000e+00 |
| 64x96x128 (non-square) | 1 | 0/30720 | 2.003e-05 | 9.274e-03 | 0.000e+00 |
| 64x96x128 (non-square) | 10 | 8/30720 | 1.831e-03 | 1.296e-02 | 4.248e-04 |
| 64x96x128 (non-square) | 100 | 9/30720 | 2.031e-01 | 8.149e-03 | 3.948e-02 |
| 256x256x512 (K sweep) | 0.1 | 0/327680 | 1.550e-06 | 7.267e-01 | 0.000e+00 |
| 256x256x512 (K sweep) | 1 | 321/327680 | 1.488e-04 | 6.769e-01 | 6.615e-05 |
| 256x256x512 (K sweep) | 10 | 398/327680 | 1.562e-02 | 9.461e-01 | 5.726e-03 |
| 256x256x512 (K sweep) | 100 | 405/327680 | 1.531e+00 | 7.246e-01 | 5.420e-01 |
| 256x256x1024 (K sweep) | 0.1 | 0/327680 | 4.143e-06 | 2.764e-01 | 0.000e+00 |
| 256x256x1024 (K sweep) | 1 | 757/327680 | 3.815e-04 | 2.175e-01 | 1.619e-04 |
| 256x256x1024 (K sweep) | 10 | 787/327680 | 4.199e-02 | 2.162e-01 | 1.466e-02 |
| 256x256x1024 (K sweep) | 100 | 797/327680 | 4.062e+00 | 2.041e-01 | 1.602e+00 |
| 256x256x4096 (K sweep, PoC-v2-5 stress) | 0.1 | 27/327680 | 4.214e-05 | 1.710e+00 | 1.299e-05 |
| 256x256x4096 (K sweep, PoC-v2-5 stress) | 1 | 2881/327680 | 3.807e-03 | 1.487e+00 | 1.213e-03 |
| 256x256x4096 (K sweep, PoC-v2-5 stress) | 10 | 2902/327680 | 3.799e-01 | 1.549e+00 | 1.178e-01 |
| 256x256x4096 (K sweep, PoC-v2-5 stress) | 100 | 2876/327680 | 3.812e+01 | 1.423e+00 | 1.309e+01 |

## 7. 誤差分布表（対 f64。診断行）

`matmul_reference_fma`（REQ-2 判定に使う f32 FMA 参照）自体の丸め誤差規模を可視化する診断行。§6 と同じ列構成・集計方法。

### f64（診断行） 対 f32_simt

| 形状 | scale | fail/total（5 シード） | max_abs_diff | max_rel_err | max_fail_abs_diff |
|---|---|---|---|---|---|
| 1x1x1 (sub-K-tile, TF32 suite) | 0.1 | 0/5 | 1.092e-10 | 4.126e-08 | 0.000e+00 |
| 1x1x1 (sub-K-tile, TF32 suite) | 1 | 0/5 | 1.208e-08 | 2.725e-08 | 0.000e+00 |
| 1x1x1 (sub-K-tile, TF32 suite) | 10 | 0/5 | 1.623e-06 | 2.883e-08 | 0.000e+00 |
| 1x1x1 (sub-K-tile, TF32 suite) | 100 | 0/5 | 7.167e-05 | 4.807e-08 | 0.000e+00 |
| 32x32x32 (block tile) | 0.1 | 0/5120 | 1.261e-08 | 2.205e-04 | 0.000e+00 |
| 32x32x32 (block tile) | 1 | 0/5120 | 1.709e-06 | 1.366e-04 | 0.000e+00 |
| 32x32x32 (block tile) | 10 | 0/5120 | 1.252e-04 | 2.341e-04 | 0.000e+00 |
| 32x32x32 (block tile) | 100 | 0/5120 | 1.267e-02 | 2.023e-04 | 0.000e+00 |
| 64x64x64 (block tile x2) | 0.1 | 0/20480 | 3.560e-08 | 4.792e-03 | 0.000e+00 |
| 64x64x64 (block tile x2) | 1 | 0/20480 | 3.353e-06 | 2.049e-03 | 0.000e+00 |
| 64x64x64 (block tile x2) | 10 | 0/20480 | 2.848e-04 | 7.668e-04 | 0.000e+00 |
| 64x64x64 (block tile x2) | 100 | 0/20480 | 3.126e-02 | 7.294e-04 | 0.000e+00 |
| 128x128x128 (block tile x4) | 0.1 | 0/81920 | 6.522e-08 | 8.904e-03 | 0.000e+00 |
| 128x128x128 (block tile x4) | 1 | 0/81920 | 8.392e-06 | 7.264e-03 | 0.000e+00 |
| 128x128x128 (block tile x4) | 10 | 5/81920 | 7.136e-04 | 3.800e-03 | 1.672e-04 |
| 128x128x128 (block tile x4) | 100 | 4/81920 | 9.034e-02 | 4.274e-03 | 9.585e-03 |
| 256x256x256 (K sweep base) | 0.1 | 0/327680 | 1.565e-07 | 5.514e-01 | 0.000e+00 |
| 256x256x256 (K sweep base) | 1 | 0/327680 | 1.679e-05 | 5.293e-01 | 0.000e+00 |
| 256x256x256 (K sweep base) | 10 | 24/327680 | 1.376e-03 | 1.405e+00 | 3.093e-04 |
| 256x256x256 (K sweep base) | 100 | 26/327680 | 1.443e-01 | 1.892e+00 | 3.969e-02 |
| 512x512x512 (block tile x16) | 0.1 | 0/1310720 | 3.508e-07 | 1.223e-01 | 0.000e+00 |
| 512x512x512 (block tile x16) | 1 | 0/1310720 | 3.869e-05 | 2.040e-01 | 0.000e+00 |
| 512x512x512 (block tile x16) | 10 | 178/1310720 | 3.829e-03 | 2.373e-01 | 8.882e-04 |
| 512x512x512 (block tile x16) | 100 | 176/1310720 | 3.741e-01 | 6.085e-02 | 7.473e-02 |
| 17x23x19 (non-multiple edge, TF32 suite) | 0.1 | 0/1955 | 6.936e-09 | 4.627e-05 | 0.000e+00 |
| 17x23x19 (non-multiple edge, TF32 suite) | 1 | 0/1955 | 6.571e-07 | 8.708e-05 | 0.000e+00 |
| 17x23x19 (non-multiple edge, TF32 suite) | 10 | 0/1955 | 8.803e-05 | 3.553e-05 | 0.000e+00 |
| 17x23x19 (non-multiple edge, TF32 suite) | 100 | 0/1955 | 7.139e-03 | 1.796e-04 | 0.000e+00 |
| 17x19x23 (non-multiple edge, f16 suite) | 0.1 | 0/1615 | 7.809e-09 | 3.148e-04 | 0.000e+00 |
| 17x19x23 (non-multiple edge, f16 suite) | 1 | 0/1615 | 8.413e-07 | 1.323e-03 | 0.000e+00 |
| 17x19x23 (non-multiple edge, f16 suite) | 10 | 0/1615 | 8.600e-05 | 1.530e-04 | 0.000e+00 |
| 17x19x23 (non-multiple edge, f16 suite) | 100 | 0/1615 | 8.840e-03 | 9.275e-04 | 0.000e+00 |
| 33x31x65 (non-multiple edge) | 0.1 | 0/5115 | 3.588e-08 | 3.483e-04 | 0.000e+00 |
| 33x31x65 (non-multiple edge) | 1 | 0/5115 | 2.500e-06 | 3.227e-04 | 0.000e+00 |
| 33x31x65 (non-multiple edge) | 10 | 0/5115 | 2.737e-04 | 6.887e-04 | 0.000e+00 |
| 33x31x65 (non-multiple edge) | 100 | 0/5115 | 2.994e-02 | 2.805e-04 | 0.000e+00 |
| 100x100x100 (non-multiple edge) | 0.1 | 0/50000 | 5.310e-08 | 5.971e-04 | 0.000e+00 |
| 100x100x100 (non-multiple edge) | 1 | 0/50000 | 5.444e-06 | 6.990e-03 | 0.000e+00 |
| 100x100x100 (non-multiple edge) | 10 | 2/50000 | 5.589e-04 | 2.075e-03 | 1.209e-04 |
| 100x100x100 (non-multiple edge) | 100 | 3/50000 | 5.380e-02 | 1.372e-03 | 5.461e-03 |
| 130x70x90 (non-multiple edge) | 0.1 | 0/45500 | 4.696e-08 | 1.931e-02 | 0.000e+00 |
| 130x70x90 (non-multiple edge) | 1 | 0/45500 | 4.464e-06 | 2.252e-02 | 0.000e+00 |
| 130x70x90 (non-multiple edge) | 10 | 3/45500 | 4.901e-04 | 9.234e-02 | 4.256e-05 |
| 130x70x90 (non-multiple edge) | 100 | 5/45500 | 5.382e-02 | 6.841e-02 | 3.645e-03 |
| 64x96x128 (non-square) | 0.1 | 0/30720 | 6.402e-08 | 2.138e-03 | 0.000e+00 |
| 64x96x128 (non-square) | 1 | 0/30720 | 6.107e-06 | 3.104e-03 | 0.000e+00 |
| 64x96x128 (non-square) | 10 | 2/30720 | 5.896e-04 | 2.290e-03 | 9.045e-05 |
| 64x96x128 (non-square) | 100 | 2/30720 | 6.426e-02 | 1.434e-03 | 4.549e-03 |
| 256x256x512 (K sweep) | 0.1 | 0/327680 | 3.391e-07 | 1.064e-01 | 0.000e+00 |
| 256x256x512 (K sweep) | 1 | 0/327680 | 3.164e-05 | 2.769e-01 | 0.000e+00 |
| 256x256x512 (K sweep) | 10 | 37/327680 | 3.654e-03 | 7.928e-01 | 7.714e-04 |
| 256x256x512 (K sweep) | 100 | 38/327680 | 3.548e-01 | 8.898e-02 | 8.464e-02 |
| 256x256x1024 (K sweep) | 0.1 | 0/327680 | 9.328e-07 | 3.924e-02 | 0.000e+00 |
| 256x256x1024 (K sweep) | 1 | 4/327680 | 7.418e-05 | 3.293e-02 | 1.869e-05 |
| 256x256x1024 (K sweep) | 10 | 62/327680 | 6.614e-03 | 3.365e-02 | 1.501e-03 |
| 256x256x1024 (K sweep) | 100 | 52/327680 | 6.134e-01 | 5.058e-02 | 2.503e-01 |
| 256x256x4096 (K sweep, PoC-v2-5 stress) | 0.1 | 0/327680 | 3.021e-06 | 7.608e-02 | 0.000e+00 |
| 256x256x4096 (K sweep, PoC-v2-5 stress) | 1 | 71/327680 | 3.425e-04 | 7.938e-02 | 6.691e-05 |
| 256x256x4096 (K sweep, PoC-v2-5 stress) | 10 | 119/327680 | 2.657e-02 | 5.092e-02 | 7.633e-03 |
| 256x256x4096 (K sweep, PoC-v2-5 stress) | 100 | 123/327680 | 2.765e+00 | 5.346e-02 | 5.631e-01 |

### f64（診断行） 対 mma_tf32

| 形状 | scale | fail/total（5 シード） | max_abs_diff | max_rel_err | max_fail_abs_diff |
|---|---|---|---|---|---|
| 1x1x1 (sub-K-tile, TF32 suite) | 0.1 | (skipped: alignment n%4/k%4) | - | - | - |
| 1x1x1 (sub-K-tile, TF32 suite) | 1 | (skipped: alignment n%4/k%4) | - | - | - |
| 1x1x1 (sub-K-tile, TF32 suite) | 10 | (skipped: alignment n%4/k%4) | - | - | - |
| 1x1x1 (sub-K-tile, TF32 suite) | 100 | (skipped: alignment n%4/k%4) | - | - | - |
| 32x32x32 (block tile) | 0.1 | 176/5120 | 2.211e-05 | 5.585e-01 | 2.211e-05 |
| 32x32x32 (block tile) | 1 | 807/5120 | 1.857e-03 | 2.557e-01 | 1.857e-03 |
| 32x32x32 (block tile) | 10 | 936/5120 | 2.699e-01 | 7.124e-01 | 1.917e-01 |
| 32x32x32 (block tile) | 100 | 945/5120 | 2.032e+01 | 9.062e-01 | 2.032e+01 |
| 64x64x64 (block tile x2) | 0.1 | 1588/20480 | 3.366e-05 | 6.096e-01 | 3.366e-05 |
| 64x64x64 (block tile x2) | 1 | 3373/20480 | 3.377e-03 | 1.046e+00 | 3.377e-03 |
| 64x64x64 (block tile x2) | 10 | 3999/20480 | 3.504e-01 | 1.185e+00 | 3.181e-01 |
| 64x64x64 (block tile x2) | 100 | 3820/20480 | 3.554e+01 | 6.071e-01 | 3.554e+01 |
| 128x128x128 (block tile x4) | 0.1 | 9854/81920 | 5.197e-05 | 1.466e+00 | 5.034e-05 |
| 128x128x128 (block tile x4) | 1 | 13140/81920 | 4.336e-03 | 1.911e+00 | 4.225e-03 |
| 128x128x128 (block tile x4) | 10 | 15518/81920 | 5.542e-01 | 1.949e+00 | 5.060e-01 |
| 128x128x128 (block tile x4) | 100 | 15139/81920 | 4.986e+01 | 1.847e+00 | 4.986e+01 |
| 256x256x256 (K sweep base) | 0.1 | 48555/327680 | 7.097e-05 | 1.957e+00 | 7.097e-05 |
| 256x256x256 (K sweep base) | 1 | 53431/327680 | 6.828e-03 | 1.923e+00 | 6.828e-03 |
| 256x256x256 (K sweep base) | 10 | 62631/327680 | 8.136e-01 | 1.966e+00 | 8.136e-01 |
| 256x256x256 (K sweep base) | 100 | 60429/327680 | 8.010e+01 | 1.908e+00 | 8.010e+01 |
| 512x512x512 (block tile x16) | 0.1 | 214250/1310720 | 1.099e-04 | 1.971e+00 | 1.099e-04 |
| 512x512x512 (block tile x16) | 1 | 212552/1310720 | 9.986e-03 | 1.994e+00 | 9.986e-03 |
| 512x512x512 (block tile x16) | 10 | 249580/1310720 | 1.254e+00 | 1.993e+00 | 1.254e+00 |
| 512x512x512 (block tile x16) | 100 | 239354/1310720 | 1.121e+02 | 1.984e+00 | 1.121e+02 |
| 17x23x19 (non-multiple edge, TF32 suite) | 0.1 | (skipped: alignment n%4/k%4) | - | - | - |
| 17x23x19 (non-multiple edge, TF32 suite) | 1 | (skipped: alignment n%4/k%4) | - | - | - |
| 17x23x19 (non-multiple edge, TF32 suite) | 10 | (skipped: alignment n%4/k%4) | - | - | - |
| 17x23x19 (non-multiple edge, TF32 suite) | 100 | (skipped: alignment n%4/k%4) | - | - | - |
| 17x19x23 (non-multiple edge, f16 suite) | 0.1 | (skipped: alignment n%4/k%4) | - | - | - |
| 17x19x23 (non-multiple edge, f16 suite) | 1 | (skipped: alignment n%4/k%4) | - | - | - |
| 17x19x23 (non-multiple edge, f16 suite) | 10 | (skipped: alignment n%4/k%4) | - | - | - |
| 17x19x23 (non-multiple edge, f16 suite) | 100 | (skipped: alignment n%4/k%4) | - | - | - |
| 33x31x65 (non-multiple edge) | 0.1 | (skipped: alignment n%4/k%4) | - | - | - |
| 33x31x65 (non-multiple edge) | 1 | (skipped: alignment n%4/k%4) | - | - | - |
| 33x31x65 (non-multiple edge) | 10 | (skipped: alignment n%4/k%4) | - | - | - |
| 33x31x65 (non-multiple edge) | 100 | (skipped: alignment n%4/k%4) | - | - | - |
| 100x100x100 (non-multiple edge) | 0.1 | 5111/50000 | 4.027e-05 | 1.934e+00 | 4.027e-05 |
| 100x100x100 (non-multiple edge) | 1 | 7957/50000 | 3.625e-03 | 1.884e+00 | 3.625e-03 |
| 100x100x100 (non-multiple edge) | 10 | 9315/50000 | 4.394e-01 | 1.772e+00 | 4.394e-01 |
| 100x100x100 (non-multiple edge) | 100 | 9044/50000 | 4.146e+01 | 1.999e+00 | 4.146e+01 |
| 130x70x90 (non-multiple edge) | 0.1 | (skipped: alignment n%4/k%4) | - | - | - |
| 130x70x90 (non-multiple edge) | 1 | (skipped: alignment n%4/k%4) | - | - | - |
| 130x70x90 (non-multiple edge) | 10 | (skipped: alignment n%4/k%4) | - | - | - |
| 130x70x90 (non-multiple edge) | 100 | (skipped: alignment n%4/k%4) | - | - | - |
| 64x96x128 (non-square) | 0.1 | 3586/30720 | 4.595e-05 | 1.290e+00 | 4.595e-05 |
| 64x96x128 (non-square) | 1 | 5065/30720 | 4.179e-03 | 1.743e+00 | 4.177e-03 |
| 64x96x128 (non-square) | 10 | 5967/30720 | 5.252e-01 | 1.489e+00 | 4.691e-01 |
| 64x96x128 (non-square) | 100 | 5676/30720 | 4.828e+01 | 1.847e+00 | 4.828e+01 |
| 256x256x512 (K sweep) | 0.1 | 53641/327680 | 1.082e-04 | 1.912e+00 | 1.012e-04 |
| 256x256x512 (K sweep) | 1 | 52800/327680 | 9.237e-03 | 1.951e+00 | 9.237e-03 |
| 256x256x512 (K sweep) | 10 | 62099/327680 | 1.130e+00 | 1.923e+00 | 1.130e+00 |
| 256x256x512 (K sweep) | 100 | 60056/327680 | 1.085e+02 | 2.000e+00 | 1.085e+02 |
| 256x256x1024 (K sweep) | 0.1 | 56497/327680 | 1.510e-04 | 1.958e+00 | 1.510e-04 |
| 256x256x1024 (K sweep) | 1 | 53478/327680 | 1.295e-02 | 1.891e+00 | 1.295e-02 |
| 256x256x1024 (K sweep) | 10 | 63060/327680 | 1.543e+00 | 1.958e+00 | 1.543e+00 |
| 256x256x1024 (K sweep) | 100 | 60312/327680 | 1.499e+02 | 1.902e+00 | 1.499e+02 |
| 256x256x4096 (K sweep, PoC-v2-5 stress) | 0.1 | 59384/327680 | 3.075e-04 | 1.960e+00 | 3.075e-04 |
| 256x256x4096 (K sweep, PoC-v2-5 stress) | 1 | 53249/327680 | 2.533e-02 | 1.958e+00 | 2.533e-02 |
| 256x256x4096 (K sweep, PoC-v2-5 stress) | 10 | 62807/327680 | 3.101e+00 | 1.972e+00 | 3.101e+00 |
| 256x256x4096 (K sweep, PoC-v2-5 stress) | 100 | 60230/327680 | 2.908e+02 | 1.934e+00 | 2.908e+02 |

### f64（診断行） 対 mma_tf32x3

| 形状 | scale | fail/total（5 シード） | max_abs_diff | max_rel_err | max_fail_abs_diff |
|---|---|---|---|---|---|
| 1x1x1 (sub-K-tile, TF32 suite) | 0.1 | (skipped: alignment n%4/k%4) | - | - | - |
| 1x1x1 (sub-K-tile, TF32 suite) | 1 | (skipped: alignment n%4/k%4) | - | - | - |
| 1x1x1 (sub-K-tile, TF32 suite) | 10 | (skipped: alignment n%4/k%4) | - | - | - |
| 1x1x1 (sub-K-tile, TF32 suite) | 100 | (skipped: alignment n%4/k%4) | - | - | - |
| 32x32x32 (block tile) | 0.1 | 0/5120 | 1.998e-08 | 2.786e-04 | 0.000e+00 |
| 32x32x32 (block tile) | 1 | 0/5120 | 2.130e-06 | 4.428e-04 | 0.000e+00 |
| 32x32x32 (block tile) | 10 | 0/5120 | 2.273e-04 | 3.616e-04 | 0.000e+00 |
| 32x32x32 (block tile) | 100 | 0/5120 | 2.777e-02 | 6.910e-04 | 0.000e+00 |
| 64x64x64 (block tile x2) | 0.1 | 0/20480 | 7.454e-08 | 1.441e-02 | 0.000e+00 |
| 64x64x64 (block tile x2) | 1 | 0/20480 | 6.012e-06 | 1.860e-02 | 0.000e+00 |
| 64x64x64 (block tile x2) | 10 | 2/20480 | 6.354e-04 | 1.712e-02 | 1.108e-04 |
| 64x64x64 (block tile x2) | 100 | 2/20480 | 6.429e-02 | 1.685e-02 | 1.091e-02 |
| 128x128x128 (block tile x4) | 0.1 | 0/81920 | 1.858e-07 | 2.529e-02 | 0.000e+00 |
| 128x128x128 (block tile x4) | 1 | 0/81920 | 1.849e-05 | 2.953e-02 | 0.000e+00 |
| 128x128x128 (block tile x4) | 10 | 25/81920 | 1.944e-03 | 2.509e-02 | 3.967e-04 |
| 128x128x128 (block tile x4) | 100 | 26/81920 | 1.606e-01 | 2.528e-02 | 4.984e-02 |
| 256x256x256 (K sweep base) | 0.1 | 0/327680 | 5.433e-07 | 1.164e+00 | 0.000e+00 |
| 256x256x256 (K sweep base) | 1 | 21/327680 | 5.621e-05 | 1.164e+00 | 1.398e-05 |
| 256x256x256 (K sweep base) | 10 | 177/327680 | 5.255e-03 | 1.106e+00 | 1.754e-03 |
| 256x256x256 (K sweep base) | 100 | 174/327680 | 5.471e-01 | 1.140e+00 | 1.751e-01 |
| 512x512x512 (block tile x16) | 0.1 | 0/1310720 | 1.645e-06 | 5.285e-01 | 0.000e+00 |
| 512x512x512 (block tile x16) | 1 | 1120/1310720 | 1.576e-04 | 4.642e-01 | 5.523e-05 |
| 512x512x512 (block tile x16) | 10 | 1490/1310720 | 1.601e-02 | 5.028e-01 | 5.389e-03 |
| 512x512x512 (block tile x16) | 100 | 1499/1310720 | 1.579e+00 | 4.313e-01 | 5.457e-01 |
| 17x23x19 (non-multiple edge, TF32 suite) | 0.1 | (skipped: alignment n%4/k%4) | - | - | - |
| 17x23x19 (non-multiple edge, TF32 suite) | 1 | (skipped: alignment n%4/k%4) | - | - | - |
| 17x23x19 (non-multiple edge, TF32 suite) | 10 | (skipped: alignment n%4/k%4) | - | - | - |
| 17x23x19 (non-multiple edge, TF32 suite) | 100 | (skipped: alignment n%4/k%4) | - | - | - |
| 17x19x23 (non-multiple edge, f16 suite) | 0.1 | (skipped: alignment n%4/k%4) | - | - | - |
| 17x19x23 (non-multiple edge, f16 suite) | 1 | (skipped: alignment n%4/k%4) | - | - | - |
| 17x19x23 (non-multiple edge, f16 suite) | 10 | (skipped: alignment n%4/k%4) | - | - | - |
| 17x19x23 (non-multiple edge, f16 suite) | 100 | (skipped: alignment n%4/k%4) | - | - | - |
| 33x31x65 (non-multiple edge) | 0.1 | (skipped: alignment n%4/k%4) | - | - | - |
| 33x31x65 (non-multiple edge) | 1 | (skipped: alignment n%4/k%4) | - | - | - |
| 33x31x65 (non-multiple edge) | 10 | (skipped: alignment n%4/k%4) | - | - | - |
| 33x31x65 (non-multiple edge) | 100 | (skipped: alignment n%4/k%4) | - | - | - |
| 100x100x100 (non-multiple edge) | 0.1 | 0/50000 | 1.293e-07 | 6.814e-03 | 0.000e+00 |
| 100x100x100 (non-multiple edge) | 1 | 0/50000 | 1.549e-05 | 5.125e-03 | 0.000e+00 |
| 100x100x100 (non-multiple edge) | 10 | 10/50000 | 1.310e-03 | 3.916e-03 | 2.390e-04 |
| 100x100x100 (non-multiple edge) | 100 | 13/50000 | 1.326e-01 | 3.535e-03 | 2.493e-02 |
| 130x70x90 (non-multiple edge) | 0.1 | (skipped: alignment n%4/k%4) | - | - | - |
| 130x70x90 (non-multiple edge) | 1 | (skipped: alignment n%4/k%4) | - | - | - |
| 130x70x90 (non-multiple edge) | 10 | (skipped: alignment n%4/k%4) | - | - | - |
| 130x70x90 (non-multiple edge) | 100 | (skipped: alignment n%4/k%4) | - | - | - |
| 64x96x128 (non-square) | 0.1 | 0/30720 | 1.712e-07 | 7.667e-03 | 0.000e+00 |
| 64x96x128 (non-square) | 1 | 0/30720 | 1.580e-05 | 1.235e-02 | 0.000e+00 |
| 64x96x128 (non-square) | 10 | 8/30720 | 1.688e-03 | 1.248e-02 | 4.340e-04 |
| 64x96x128 (non-square) | 100 | 8/30720 | 1.855e-01 | 7.065e-03 | 3.984e-02 |
| 256x256x512 (K sweep) | 0.1 | 0/327680 | 1.447e-06 | 7.430e-01 | 0.000e+00 |
| 256x256x512 (K sweep) | 1 | 309/327680 | 1.438e-04 | 7.664e-01 | 5.583e-05 |
| 256x256x512 (K sweep) | 10 | 407/327680 | 1.410e-02 | 7.398e-01 | 5.808e-03 |
| 256x256x512 (K sweep) | 100 | 406/327680 | 1.430e+00 | 7.179e-01 | 4.978e-01 |
| 256x256x1024 (K sweep) | 0.1 | 0/327680 | 3.707e-06 | 2.757e-01 | 0.000e+00 |
| 256x256x1024 (K sweep) | 1 | 758/327680 | 3.849e-04 | 2.432e-01 | 1.461e-04 |
| 256x256x1024 (K sweep) | 10 | 787/327680 | 4.282e-02 | 2.412e-01 | 1.382e-02 |
| 256x256x1024 (K sweep) | 100 | 785/327680 | 4.136e+00 | 2.145e-01 | 1.486e+00 |
| 256x256x4096 (K sweep, PoC-v2-5 stress) | 0.1 | 24/327680 | 4.080e-05 | 1.723e+00 | 1.269e-05 |
| 256x256x4096 (K sweep, PoC-v2-5 stress) | 1 | 2888/327680 | 3.465e-03 | 1.448e+00 | 1.215e-03 |
| 256x256x4096 (K sweep, PoC-v2-5 stress) | 10 | 2910/327680 | 3.617e-01 | 1.525e+00 | 1.116e-01 |
| 256x256x4096 (K sweep, PoC-v2-5 stress) | 100 | 2879/327680 | 3.811e+01 | 1.404e+00 | 1.272e+01 |

## 8. s² 比例・rel_err 不変性（P3。#995 §8/§9 と同じ判定基準）

一様に f64（診断行）の `max_abs_diff` を `r(s)` の指標として使う（f32fma 側は fail=0 セルで `max_fail_abs_diff=0` になり比が定義不能な形状があるため。f64 診断行の `max_abs_diff` は丸め誤差そのものを表すため常に非ゼロ）。事前固定の判定基準: `|log10 r(s)| ≤ 0.3`（2 倍以内の乖離）を「成立」とする。

### s² 比例（`r(s) = max_abs_diff_f64(s) / (s² · max_abs_diff_f64(1))`）

| 経路 | セル数 | min | median | max | max｜log10 r｜ | 判定 |
|---|---|---|---|---|---|---|
| f32_simt | 45（15 形状 × 3 スケール） | 0.593 | 0.988 | 1.435 | 0.227 | 成立 |
| mma_tf32 | 30（10 形状 × 3 スケール） | 0.997 | 1.171 | 1.453 | 0.162 | 成立 |
| mma_tf32x3 | 30（10 形状 × 3 スケール） | 0.835 | 1.044 | 1.304 | 0.115 | 成立 |

3 経路とも `|log10 r| ≤ 0.3` を全セルで満たし、**`s²` 比例が成立する**（絶対誤差 = 相対誤差 × 出力スケール ≈ 相対誤差 × s² という理論予測と整合。#995 §8 の TF32 での結論と同型）。形状別の詳細（15 形状 × r(0.1)/r(10)/r(100)）は `docs/perf/logs/cuda-gemm-tf32x3-1356/s2_summary.txt` に保存済み。

### `max_rel_err` のスケール不変性（f64 診断行。`s=1` を基準とした比）

| 経路 | セル数 | min | median | max | 所見 |
|---|---|---|---|---|---|
| f32_simt | 45 | 0.085 | 0.869 | 4.100 | 100x100x100・130x70x90 でばらつきが大きい（f32_simt は f32fma と厳密一致し f64 側の誤差自体が極小のため、桁落ちキャンセレーションの影響を受けやすい） |
| mma_tf32 | 30 | 0.580 | 1.006 | 3.544 | 32x32x32（小形状）のみ比 2.18〜3.54 と外れる。他形状は 0.74〜1.06 に収まる |
| mma_tf32x3 | 30 | 0.572 | 0.950 | 1.561 | 全形状・全セルが 0.57〜1.56 に収まり 3 経路中もっともスケール不変に近い |

**判定: `max_rel_err` は 3 経路ともおおむねスケール不変**（#995 §9 と同様、要素数が少ない形状で単一セルの偶然による外れ値が生じうる点に留意）。`mma_tf32x3` が 3 経路中もっとも安定している点は §6/§7 の誤差分布改善（対 `mma_tf32`）と整合する。

**P3 総合判定: 成立**（3 経路とも `s²` 比例・`max_rel_err` スケール不変性のいずれも判定基準を満たす）。

## 9. 純カーネル時間（5 回計測中央値）


`gemm_tf32x3_kernel_time_bench` を 5 回プロセス起動（各起動直前に `nvidia-smi --query-gpu=utilization.gpu` で 0% を確認。他プロセス混入なし）。各 (route, shape) の `median_ms`／`tflops` をさらに 5 run で中央値化（`statistics.median`）。生 CSV は `docs/perf/logs/cuda-gemm-tf32x3-1356/bench-run{1..5}.csv`。

| 形状 | f32_simt (TFLOPS) | wmma_tf32 (TFLOPS) | mma_tf32 (TFLOPS) | mma_tf32x3 (TFLOPS) | tf32x3/f32_simt | tf32x3/mma_tf32 |
|---|---|---|---|---|---|---|
| 512×512×512 | 6.970 | 8.156 | 8.478 | 6.251 | 0.897 | 0.737 |
| 1024×1024×1024 | 11.801 | 12.631 | 16.194 | 10.977 | 0.930 | 0.678 |
| 2048×2048×2048 | 14.667 | 14.320 | 18.461 | 13.000 | 0.886 | 0.704 |
| 4096×4096×4096 | 13.068 | 14.042 | 10.748 | 9.424 | 0.721 | 0.877 |
| 256×256×4096 | 3.345 | 3.118 | 3.211 | 2.979 | 0.891 | 0.928 |

**P4 判定（計測前固定の閾値。§11 参照）**: 5 形状すべてで `mma_tf32x3 / f32_simt` < 1.0（0.721〜0.930 倍）。3 回 mma を実行する split-single 法の構造上、理論上限は 1/3 ≈ 0.333 倍だが、実測の `mma_tf32x3 / mma_tf32` 比（0.678〜0.928）はこれを大きく上回り、3 回のうち 1 回分は共有可能な中間データ（TF32 丸め）を再利用できている効率を示す（ただし採否判定には使わない指標として §11 の定義どおり参考記録に留める）。`mma_tf32x3` は 4096³ で唯一 `mma_tf32`（10.748 TFLOPS）を下回らない・むしろ上回る形状はなく、全形状で `f32_simt` にも劣後する。

## 10. `#[ignore]` テスト結果

`tests/gemm_mma_tf32x3.rs`（`--release --all-features`。生ログ: `docs/perf/logs/cuda-gemm-tf32x3-1356/ignored-gemm_mma_tf32x3.log`）。

| テスト | 結果 | 詳細 |
|---|---|---|
| `launch_tf32x3_zero_dim_shape_is_noop_or_zero_fills_without_launch` | PASS | — |
| `launch_tf32x3_zero_dim_shape_returns_empty_without_launch` | PASS | — |
| `mma_tf32x3_matches_reference_across_shapes`（16x8x8／64³／128³／512³／60x68x36・seed 5001〜5005） | **FAIL** | 形状 512×512×512（seed 5004）で `assert_parity` 複合判定 FAIL: `fail_count=212/262144, max_abs_diff=1.469e-4, max_rel_err=4.970e-1, mean_abs_diff=2.152e-5, mean_rel_err=1.517e-5, p50_abs_diff=1.786e-5, p99_abs_diff=7.343e-5, p999_abs_diff=1.011e-4`。テストは形状を順に走査し最初の FAIL で panic するため、16x8x8／64³／128³ は暗黙に pass 済み（先行 assert に到達していた）が、512³ で停止し 60x68x36（seed 5005）は未実行のまま |
| `mma_tf32x3_k4096_stress`（4096³・seed 9001） | **FAIL** | `fail_count=151916/16777216, max_abs_diff=3.731e-3, max_rel_err=1.985e0, mean_abs_diff=4.872e-4, mean_rel_err=1.191e-4, p50_abs_diff=4.101e-4, p99_abs_diff=1.606e-3, p999_abs_diff=1.991e-3` |

`test result: FAILED. 2 passed; 2 failed`（9.89s）。§6 の対 f32fma 誤差分布表と照合すると、512×512×512・s=1（`[-1, 1)` 入力相当）の集計行は fail 1123/1310720（0.086%）であり、`assert_parity`（単一シード・厳密ゼロ fail）が要求する fail=0 を満たさないことと整合する（形状は一致するがシード条件〈テストは seed 5004 固定・プローブは seed 1〜5 の 5 シード集計〉は完全には一致しない点に注意。§10 冒頭の既記載どおり）。**P1: 不成立**。

## 11. 採否（P1〜P4 総合判定。3 択語彙: 「opt-in 維持・推奨」／「opt-in 維持・条件付き推奨」／「opt-in 維持・非推奨」）

判定基準は計測前に固定済み（下記）で、実測後に事後変更していない:

- P1（REQ-2 判定・厳密ゼロ fail）: `#[ignore]` テスト 2 件が GB10 で pass するか
- P2（対 f64 精度）: `max_abs_diff`（対 f64）が f32 SIMT の同指標の 2 倍以内か・mma_tf32 に対し 1 桁以上改善しているか
- P3（スケール依存）: §8 の判定
- P4（性能。数値閾値は計測前に固定・事後変更しない）: 5 形状（512³／1024³／2048³／4096³／256×256×4096）それぞれの `mma_tf32x3 / f32_simt`（TFLOPS 比）を基準に 3 択を選ぶ
  - P4-推奨: 5 形状すべてで `mma_tf32x3 / f32_simt` ≥ 1.0
  - P4-条件付き推奨: 一部形状のみ ≥ 1.0（成立する形状を条件として本節に明記する）
  - P4-非推奨: 5 形状すべてで `mma_tf32x3 / f32_simt` < 1.0
  - `mma_tf32x3 / mma_tf32`（3 回 mma の理論上限 1/3 に対する実効効率の参考指標）は採否判定には使わない（参考記録に留める）

### 実測結果

| 判定項目 | 結果 |
|---|---|
| P1（`#[ignore]` テスト 2 件） | **不成立**（2 件とも FAIL。§10） |
| P2（対 f64 精度） | **形状依存で部分的に不成立**。40 セル（10 形状 × 4 スケール）中、`max_abs_diff(f64)` の `tf32x3/f32_simt` 比が 2 倍を超えるセルは 24/40（60%。とくに K 支配的形状 256x256x512／256x256x1024／256x256x4096 で顕著、最大 13.783 倍〈256x256x4096, s=100〉）。`mma_tf32` に対する「1 桁以上改善」（`mma_tf32/tf32x3 ≥ 10`）は 36/40（90%）で成立するが、256x256x4096 の 4 セルは 7.3〜8.6 倍にとどまり不成立。詳細は `docs/perf/logs/cuda-gemm-tf32x3-1356/gen_p2.py`（再実行可能）の出力を参照 |
| P3（スケール依存） | **成立**（3 経路とも `s²` 比例〈max｜log10 r｜: f32_simt 0.227・mma_tf32 0.162・mma_tf32x3 0.115〉・`max_rel_err` スケール不変性〈§8〉の判定基準を満たす） |
| P4（純カーネル時間） | **5 形状すべてで `mma_tf32x3 / f32_simt` < 1.0**（0.721〜0.930 倍。§9）。性能面のみで判断すれば「P4-非推奨」に相当する |

### 総合判定

**P1 が不成立のため、本ドキュメントは §11 の 3 択（「opt-in 維持・推奨」／「opt-in 維持・条件付き推奨」／「opt-in 維持・非推奨」）を選ばない**（計測前固定の運用どおり）。P1 不成立時の運用に従い、baseline 行・`ParityPath::MmaTf32x3` は追加せず、実測値を「baseline 提案値（未承認）」として以下に記録し、ユーザー判断へ回す。

**baseline 提案値（未承認）**:

- `mma_tf32x3_matches_reference_across_shapes`（512×512×512, seed 5004）: `fail_count=212/262144, mean_abs_diff=2.152e-5, max_abs_diff=1.469e-4, max_rel_err=4.970e-1`
- `mma_tf32x3_k4096_stress`（4096³, seed 9001）: `fail_count=151916/16777216, mean_abs_diff=4.872e-4, max_abs_diff=3.731e-3, max_rel_err=1.985e0`

**参考所見（P1 不成立を前提に、仮に P1 を満たしていた場合の傾向）**: P4（性能）は 5 形状すべてで基準未達（0.721〜0.930 倍）であり、P2 も K 支配的形状で 2 倍基準を超える。両者を踏まえると、たとえ P1 が pass していたとしても総合判定は「opt-in 維持・非推奨」寄りのシグナルが強い。ただし P1 自体が不成立（fail=0 を満たさない）ことは、3×TF32 が REQ-2 の厳密ゼロ fail 判定という受け入れ基準を現状満たしていないという、より根本的な問題であり、これは baseline 方式（`ParityBaseline` の非後退検査。`.claude/rules/coding-rust.md` 記載の「実機実測で成立が確認された形状に限り」厳密ゼロ fail 判定を適用する方針）による受け入れへの切り替えを検討するか、`CudaMmaTf32x3Gemm` 実装自体の見直しが必要かの判断をユーザーに委ねる。

## 12. 制約・スコープ外

- レジスタ圧・occupancy の定量化（`CudaFunction` が非公開のため `internal-diagnostics` 側の拡張が必要。`docs/cuda-tf32x3-split-single-decision.md` §6 参照）
- `gemm_bias_act`／`gemm_resident_*`／VJP への 3×TF32 適用
- framework-compare `--tf32` の 3×TF32 対応（ピン `fandhe-ai =0.7.0` は `set_cuda_gemm_precision` を持たない）
- baseline 行・`ParityPath::MmaTf32x3` の追加（P1 不成立時のみ提案値を記録。承認は別途）
- Metal 側の同種検討

## 13. 生ログ一覧

実測完了。`docs/perf/logs/cuda-gemm-tf32x3-1356/` に保存済み:

- `probe-mma-run1.md`／`probe-mma-run2.md`（誤差分布。2 回実行。`diff` 完全一致を確認済み）
- `ignored-gemm_mma_tf32x3.log`（`#[ignore]` テスト実行ログ。2 件 FAIL の詳細を含む）
- `bench-run1.csv` 〜 `bench-run5.csv`（純カーネル時間。5 回プロセス起動）
- `env_info.txt`（GPU・driver・compute capability・CUDA SDK・rustc・cargo・`.rev-stamp`・ビルド feature・GPU 空き確認前後）
- `uptime_before_run.txt`／`uptime_after_run.txt`
- `aggregate.py`／`gen_tables.py`／`gen_s2.py`／`gen_relerr.py`／`gen_p2.py`／`gen_bench.py`（生ログから §6〜§9・§11 の表を再現する集計スクリプト。Python3 標準ライブラリのみ）
- `agg_f32fma.json`／`agg_f64.json`／`s2_results.json`（中間集計結果。再現性検証用）
