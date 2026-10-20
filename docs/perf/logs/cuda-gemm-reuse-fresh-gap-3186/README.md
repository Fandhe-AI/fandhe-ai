# #3186 reuse と fresh の差の原因特定（実測基盤）

`docs/perf/cuda-gemm-reuse-phase-breakdown.md` §14 の実測基盤。判定規則は `RULE.txt`
（実測前に固定。事後に緩めない）が正。診断のみで本番コードは変更しない（是正は #3225）。

## 状態

**GB10 実測済み（2026-10-20）**。判定は `gb10/aggregate.md`、結論は `docs/perf/cuda-gemm-reuse-phase-breakdown.md` §14。
ツリー同定は `rev-stamp-verification-3186.md`。#2107（`622594f15`）と #2108（HEAD）の実測も同イシューで実施した。

## 構成

| ファイル | 内容 |
|---|---|
| `RULE.txt` | 事前登録判定規則（判定 W・判定 F・二峰性条件・checksum 規則・専有ゲート） |
| `orchestrate.sh` | `<gb10\|x86> [--dry-run] [--out <dir>]`。テストバイナリ（`crates/backend-cuda/src/fresh_reuse_window_diag_tests_3186.rs`）を 1 回ビルドし (N, 腕) ごとに独立プロセスで 5 run＋同一セッション Layer A（bench-fandhe gemm fresh／reuse）。`.rev-stamp` と HEAD の不一致で停止。既存 run の上書きは拒否 |
| `aggregate.py` | `python3 aggregate.py <dir>` で `aggregate.md` を生成。`--self-test`（GPU 不要）。欠損・run 不足・checksum 不一致は exit 1 |
| `env_info.txt` | 実測前の空テンプレート（実測時の採取値は `gb10/env_info.txt`） |
| `rev-stamp-verification-3186.md` | 3 ツリーの `.rev-stamp` とツリー指紋の照合 |
| `gb10/` | 生ログ（`run{1..5}/`）・`aggregate.md`・`load_gate*`・`env_info.txt` |

## 実行手順（GB10。`.git` 付きツリーが必要）

`orchestrate.sh` は `.rev-stamp` と `git rev-parse HEAD` を照合するため、標準の `.git` なし rsync では
動かない。`git bundle` を転送してノード上でオフライン clone し、`git rev-parse HEAD > .rev-stamp` を置く
（先例: `docs/perf/logs/cpu-thread-policy-macmini-3178/README.md`）。

```
./orchestrate.sh gb10
python3 aggregate.py gb10
```
