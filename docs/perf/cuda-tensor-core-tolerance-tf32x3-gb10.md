# 3×TF32（split-single 法）の誤差分布・スケールスイープ・純カーネル時間 GB10 実測（イシュー #1356）

## 0. 実測状況（重要）

**本ドキュメントは GB10 実機実測記入欄のスケルトンであり、本エージェント実行環境に CUDA 実機が接続されていないため実測値は未取得のまま作成した**（`docs/real-hardware-verification-env.local.md` が本 worktree に存在せず、実機への rsync 転送・ビルド・実行手順（§3）を実行できなかった）。数値を推定・外挿・捏造せず「未実測」を明記する（`docs/cuda-tf32x3-split-single-decision.md` §8 の記入欄・#994 初版と同じ扱い）。§6〜§9 の表は列見出しのみのプレースホルダであり、後続セッションが §3 の手順をそのまま実行すれば埋められる。

- コード側の実装（`crates/backend-cuda/examples/wmma_tolerance_probe.rs` の `--routes mma`・`crates/backend-cuda/examples/gemm_tf32x3_kernel_time_bench.rs`）はローカル（CUDA 非搭載環境）でビルド・単体テストとも green を確認済み（§3.5「ローカルでの事前確認」）。
- §11「採否」は**未確定**とし、本ドキュメント §11 の 3 択語彙（「opt-in 維持・推奨」／「opt-in 維持・条件付き推奨」／「opt-in 維持・非推奨」）のいずれも選ばない。

## 1. 位置づけ

- イシュー #1356「3×TF32 の誤差分布（対 CPU f32 FMA 参照・f64 参照）とスケールスイープ・純カーネル時間を GB10 で実測し f32 SIMT／TF32 と比較して採否を記録する」の実測記録。親ツリー #1354・承認元 #1338・依存イシュー #1355（PR #1400）。
- #1355（PR #1400）は 3×TF32（split-single 法。`CudaMmaTf32x3Gemm`）を precision の第 3 モード（既定 OFF・opt-in）として実装したが、GB10 実機実測は未実施のまま `docs/cuda-tf32x3-split-single-decision.md` §8 に記入欄のみを残していた。本ドキュメントはその実測記入欄を引き継ぐ独立ドキュメントであり、実測完了後は同 decision doc §8 から本ドキュメントへの参照リンクへ差し替える想定（本 PR の時点では未実測のためリンクのみ追加し、値は転記していない）。
- 手順は `docs/perf/cuda-tensor-core-tolerance-gb10-scale-sweep.md`（#995）と同一プロトコル（`wmma_tolerance_probe` の `--scales 0.1,1,10,100`・15 形状 × 5 シード・`fandhe_ai_backend_cpu::compare` 統計）を `--routes mma` へ拡張して踏襲する。
- **閾値定数（`RELATIVE_TOLERANCE`＝1e-3・`ABSOLUTE_RESCUE_THRESHOLD`＝1e-5）・判定式・テスト許容誤差は本ドキュメントでは一切変更していない**（`.claude/rules/coding-rust.md`・`.claude/rules/security.md` A08）。f64 参照行は診断行であり REQ-2 判定には使わない（`wmma_tolerance_probe.rs` ファイル冒頭コメント参照）。

## 2. 計測環境

**未実測**（GB10 実機不達）。実測時は以下の項目を `docs/perf/cuda-tensor-core-tolerance-gb10-scale-sweep.md` §2 と同じ形式で埋める。

| 項目 | 値 |
|---|---|
| GPU | 未実測 |
| driver | 未実測 |
| compute capability | 未実測 |
| CUDA SDK | 未実測 |
| rustc | 未実測 |
| 計測対象コミット | 未実測（`.rev-stamp`） |
| 計測日 | 未実測 |
| GPU 空き確認（計測前） | 未実測（`nvidia-smi --query-gpu=utilization.gpu` / `--query-compute-apps`） |
| GPU 空き確認（計測後） | 未実測 |
| `uptime` / load average（計測前後） | 未実測 |

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
      cargo build --release -p fandhe-ai-backend-cuda \
      --example wmma_tolerance_probe --example gemm_tf32x3_kernel_time_bench
'
```

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

## 4. 実行カーネル種別の実測

**未実測**。`f32 SIMT` 節の `kernel` 列は `internal-diagnostics` feature 有効時のみ `Classic`／`Pipeline` を報告し（`--tf32-kernel` と異なり `--routes mma` 自体は feature 非依存で動く）、無効時は `n/a (internal-diagnostics)` になる（`wmma_tolerance_probe.rs::f32_simt_kernel_kind` の設計判断）。`TF32 mma.sync`／`3xTF32` の `kernel` 列はそれぞれ `mma_tf32`／`mma_tf32x3` 固定（単一カーネルのため多段選択なし）。

## 5. 整列非対応形状のスキップ確認

**未実測**。`SHAPES` のうち `n % 4 == 0 && k % 4 == 0` を満たさない 5 形状（1x1x1・17x23x19・17x19x23・33x31x65・130x70x90）は `TF32 mma.sync`／`3xTF32` の 2 節で `(skipped: alignment n%4/k%4)` 行になる想定（`wmma_tolerance_probe.rs::is_mma_aligned`）。実測後、この期待どおりであることを生ログで確認しここに記録する。

## 6. 誤差分布表（対 f32fma。REQ-2 判定行）

**未実測**。列: 形状 | scale | fail/total（5 シード）| max_abs_diff | max_rel_err | max_fail_abs_diff。`f32 SIMT`／`TF32 mma.sync`／`3xTF32` の 3 経路を並記する。

## 7. 誤差分布表（対 f64。診断行）

**未実測**。§6 と同じ列構成。`matmul_reference_fma`（REQ-2 判定に使う f32 FMA 参照）自体の丸め誤差規模を可視化する目的で、`f32 SIMT` の対 f64 列と `TF32 mma.sync`／`3xTF32` の対 f64 列を比較する。

## 8. s² 比例・rel_err 不変性（P3。#995 §8/§9 と同じ判定基準）

**未実測**。`r(s) = max_fail_abs_diff(s)/(s²·max_fail_abs_diff(1))` が `|log10 r| ≤ 0.3` を満たすか、`max_rel_err` がスケール間でおおむね不変か（3×TF32・単発 TF32・f32 SIMT の 3 経路それぞれについて確認する）。fail=0 の形状（1x1x1 等）は `max_abs_diff`（対 f64）の `s²` 比で代替する（#995 と同じ扱い）。

## 9. 純カーネル時間（5 回計測中央値）

**未実測**。`gemm_tf32x3_kernel_time_bench` の出力（`route,m,n,k,kernel,median_ms,q1_ms,q3_ms,tflops`）を 5 run 集計し、`f32_simt`・`mma_tf32`・`mma_tf32x3`・（参考）`wmma_tf32` を形状ごとに並記する。`mma_tf32x3 / f32_simt`・`mma_tf32x3 / mma_tf32` の比も併記する。

| 形状 | f32_simt (TFLOPS) | mma_tf32 (TFLOPS) | mma_tf32x3 (TFLOPS) | wmma_tf32（参考） | tf32x3/f32_simt | tf32x3/mma_tf32 |
|---|---|---|---|---|---|---|
| 512×512×512 | 未実測 | 未実測 | 未実測 | 未実測 | 未実測 | 未実測 |
| 1024×1024×1024 | 未実測 | 未実測 | 未実測 | 未実測 | 未実測 | 未実測 |
| 2048×2048×2048 | 未実測 | 未実測 | 未実測 | 未実測 | 未実測 | 未実測 |
| 4096×4096×4096 | 未実測 | 未実測 | 未実測 | 未実測 | 未実測 | 未実測 |
| 256×256×4096 | 未実測 | 未実測 | 未実測 | 未実測 | 未実測 | 未実測 |

## 10. `#[ignore]` テスト結果

**未実測**。`tests/gemm_mma_tf32x3.rs::mma_tf32x3_matches_reference_across_shapes`／`mma_tf32x3_k4096_stress`（`assert_parity`。厳密ゼロ fail 判定）の GB10 実行結果を記録する。§6 の対 f32fma 誤差分布表（fail/total 列）と整合するはずだが、`assert_parity` は形状ごとに独立の 1 回実行のためシード条件が完全一致しない可能性がある点に注意する。

## 11. 採否（P1〜P4 総合判定。3 択語彙: 「opt-in 維持・推奨」／「opt-in 維持・条件付き推奨」／「opt-in 維持・非推奨」）

**未確定（実測後に確定する）**。判定基準は計測前に固定済み（下記）で、実測後に事後変更しない:

- P1（REQ-2 判定・厳密ゼロ fail）: `#[ignore]` テスト 2 件が GB10 で pass するか
- P2（対 f64 精度）: `max_abs_diff`（対 f64）が f32 SIMT の同指標の 2 倍以内か・mma_tf32 に対し 1 桁以上改善しているか
- P3（スケール依存）: §8 の判定
- P4（性能）: §9 の比率から 3 択（推奨／条件付き推奨／非推奨）を選ぶ

P1 が不成立の場合、baseline 行・`ParityPath::MmaTf32x3` は追加せず、実測 `fail_count`／`mean_abs_diff`／`max_abs_diff`／`max_rel_err` を「baseline 提案値（未承認）」として本節に追記し、ユーザー判断へ回す（本節「P1 が不成立の場合」の運用）。

## 12. 制約・スコープ外

- レジスタ圧・occupancy の定量化（`CudaFunction` が非公開のため `internal-diagnostics` 側の拡張が必要。`docs/cuda-tf32x3-split-single-decision.md` §6 参照）
- `gemm_bias_act`／`gemm_resident_*`／VJP への 3×TF32 適用
- framework-compare `--tf32` の 3×TF32 対応（ピン `fandhe-ai =0.7.0` は `set_cuda_gemm_precision` を持たない）
- baseline 行・`ParityPath::MmaTf32x3` の追加（P1 不成立時のみ提案値を記録。承認は別途）
- Metal 側の同種検討

## 13. 生ログ一覧

**未実測のため生ログなし**。実測後、`docs/perf/logs/cuda-gemm-tf32x3-1356/` に以下を保存する想定:

- `probe-mma-run1.md`／`probe-mma-run2.md`（誤差分布。2 回実行）
- `ignored-gemm_mma_tf32x3.log`（`#[ignore]` テスト実行ログ）
- `bench-run1.csv` 〜 `bench-run5.csv`（純カーネル時間。5 回プロセス起動）
- `env_info.txt`（GPU・driver・compute capability・CUDA SDK・rustc・`.rev-stamp`・GPU 空き確認前後・load average）
- `uptime_before_run.txt`／`uptime_after_run.txt`
