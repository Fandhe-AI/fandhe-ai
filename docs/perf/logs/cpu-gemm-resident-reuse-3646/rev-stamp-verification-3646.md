# 計測ツリーの同定（`.rev-stamp` とツリー指紋。イシュー #3646）

GB10 ノード（`<cuda-node>`）の専用ディレクトリ（既存ディレクトリは触っていない）へ、`git archive` で固定コミットの tracked ファイルだけを展開して計測した
（rsync は使っていない。ノード側にネットワーク経由の取得はなく、`.git`・`docs/spec` 実体・`docs/real-hardware-verification-env.local.md` は転送していない。
ノード上に `*.local.md` が無いことを `find` で確認済み）。ルートに `.rev-stamp`（1 行目 HEAD の sha・2 行目 `dirty=0`）を置いた。

| 項目 | 値 |
|---|---|
| `.rev-stamp` 1 行目 | `449ae15814514d7e64a62c6e0bd573c7f422391f`（`RULE.v2.txt` を含むコミット。`RULE.txt` は前コミット以前から不変） |
| `.rev-stamp` 2 行目 | `dirty=0` |
| `RULE.txt` の sha256 | `cb2a2c109c7539d6b3298ee4291767ece7d12ffc20d4cb2275df0b0cf0ecc2a2`（4 系列の `env_info.txt` の `rule_sha256` と一致） |
| `RULE.v2.txt` の sha256 | `eb71245f841c8586b08253a63a953c73f8833065945a78bcd788e103fcfebb65` |

`RULE.v2.txt` を別コミットにした理由と範囲（ブリッジ節のみの置換）は `RULE.v2.txt` 冒頭を正とする。計測ツリーに含まれるのは
`RULE.v2.txt` を作ったコミットであり、`session-scripts/session-3646.sh`・`gb10-bridge/bridge_judge.py`・`gb10-gcpuinf/gcpuinf_judge.py`・`blas_probe/` は
計測後に追加した（セッションスクリプトはノードへ個別に転送して実行した。sha256 は下表）。

## ツリー指紋

```
対象: crates と scripts/bench/framework-compare の tracked ファイル 2512 件（Cargo.lock と target* を除く）
指紋: LC_ALL=C sort した相対パス順に sha256 を取り（`<hash>  <path>` 形式）、その出力全体をもう一度 sha256 にかけた値
```

| 側 | 指紋 |
|---|---|
| ノード（展開ツリー。計測・ビルド後に取得） | `5e8b85d2ffbfa4f7a022a28b1acddf6427cc88e53fbd3df341904969e5c37680` |
| Mac（同コミットの worktree。`git archive` と同内容） | `5e8b85d2ffbfa4f7a022a28b1acddf6427cc88e53fbd3df341904969e5c37680` |
| 一致 | 一致 |

## 計測ツリーの等価性（0.12.0 の bench との同一コード確認）

`git diff --stat v0.12.0 HEAD -- crates/backend-cpu/src crates/autodiff/src/optim/resident_input.rs crates/autodiff/src/optim/device_store.rs crates/facade/src/lib.rs crates/tensor-core/src/alloc.rs`
は実測前の HEAD で空（出力なし）。bench-fandhe は registry の `fandhe-ai =0.12.0`（patch なし・`--locked`）でビルドした（行の `version` が `0.12.0`）。

## 実行物の sha256

| ファイル | sha256 |
|---|---|
| `session-scripts/session-3646.sh`（ノード上の実行物と同一） | `e23cf786923b2f44fed64c49dfae03d62033a3d0b01a843544cb14ac1217c64a` |
| `gb10-gcpuinf/bench_py.sha256`（`bench_py.py` の複写） | `9fd2c08f37d66b5bb41ececab8a5b3daaa36b55609770502ed3b35ae2ce2409b`（`opp_kernel_torch_diag.py` の固定値と一致） |

## 実行順序・ロック

`session-scripts/steps.log`: 実機ロック（`mkdir` 方式）取得 02:12:45Z → 診断主系列 → minflt → tunables → taskset → ブリッジ → G-CPU-INF → 解放 03:43:12Z（UTC）。
ビルドはロック取得前に完了済み（計測中のビルド並走なし）。実行前にノードで bench-fandhe の N=256 スモーク（gemm fresh・reuse `--phases`・infer reuse `--phases` 各 1 回。出力は削除済み・系列に数えない）を行った。

## 専有系列（`gb10-excl/`。3 回目の reopen）

手順は上と同じ（固定コミットの tracked ファイルだけを `git archive` で新しい作業ディレクトリへ展開。`.git`・`docs/spec` 実体・`*.local.md` は転送せず、ノード上に `*.local.md` が無いことを `find` で確認。rsync は使っていない）。
既存の回収物と作業ディレクトリには触れていない。実測前に手順（`session-scripts/session-3646-excl.sh`・`excl_check.py`）を別コミットで固定した。

| 項目 | 値 |
|---|---|
| `.rev-stamp` 1 行目 | `555a445a6d6a139e9d554cc951abc64d7c24c591`（手順固定コミット。`gb10-excl/gb10/env_info.txt` の `rev_stamp_head` と一致） |
| `.rev-stamp` 2 行目 | `dirty=0` |
| `RULE.txt` の sha256 | `cb2a2c109c7539d6b3298ee4291767ece7d12ffc20d4cb2275df0b0cf0ecc2a2`（`env_info.txt` の `rule_sha256` と一致。不変） |
| `RULE.v2.txt` の sha256 | `eb71245f841c8586b08253a63a953c73f8833065945a78bcd788e103fcfebb65`（不変） |
| ツリー指紋（ノード・Mac の `git archive` 展開物） | どちらも `5e8b85d2ffbfa4f7a022a28b1acddf6427cc88e53fbd3df341904969e5c37680`（2512 件。上と同じ指紋。一致） |
| 等価性 | `git diff --stat v0.12.0 555a445a…`（上と同じパス群）と `git diff --stat 449ae158… 555a445a…`（`crates`・`scripts/bench/framework-compare`・RULE 2 本・`orchestrate.sh`・`aggregate.py`・`cpu-predict-resident-fixedcost-2105`・`bench_py.py`）は、どちらも空。bench-fandhe は registry の `fandhe-ai =0.12.0`（patch なし・`--locked`） |

| ファイル | sha256 |
|---|---|
| `session-scripts/session-3646-excl.sh` | `66bba41cfe8af626cce8281231ada30370bfcdcb53470cd4feebb15e01324bbe` |
| `excl_check.py` | `699eb1bdbdd88eaf3b4f1f9fb3afda1183b8b883c047b1a1028008824289ad72（レビュー指摘で観測間隔 60 秒以下の検査を追加。判定閾値・実測の判定結果は不変）` |
| `gb10-bridge/bridge_judge.py` | `254a7a7557d53c96ef154deed8ae608ee27cf0f67f98c77d4882b4786bcaca7f`（不変） |
| `gb10-gcpuinf/gcpuinf_judge.py` | `88b6f84ef76792c988473805a926448409fa33eb5cd1a8e6ba933cb5de4738a3`（不変） |
| `aggregate.py` | `6ed4bb45b8393cfb61c7cde8c165084382927c18a91b9d4a91522b65dcffec96`（不変） |
| `orchestrate.sh` | `f508373827081dc74c015b239f1c4b3115053e6ae6ef9e528766ee000be961b5`（不変） |

実行: 実機ロック取得 05:34:04Z → 診断主系列 → minflt → tunables → taskset → ブリッジ → G-CPU-INF → 解放 07:00:11Z（UTC。`gb10-excl/session-scripts/steps.log`）。
ビルドはロック取得前に完了。ロック取得前に、計数しないスモーク（診断テスト 60 秒・bench-fandhe gemm fresh／reuse／reuse --phases の N=256・infer reuse --phases・PyTorch 純カーネル・PyTorch infer 窓。
1 秒間隔 64 回の compute-apps はすべて 0 件。出力は削除済み）を行い、自プロセス由来の compute-apps が出ないことを確かめた。
G-CPU-INF (iii) の追加確認（shim）はセッション後に別途実行した（ロックを取得・compute-apps 前後 0 件。計測外）。

