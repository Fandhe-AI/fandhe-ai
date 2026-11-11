# #4161 GB10 の推論だけに jpw1 を効かせる A/B

状態: **2026-11-11 に実測済み。判定は REJECT**（正式系列 5/5・前提成立・checksum 一致）。RULE は `RULE.txt`（PR #4246 で固定）。結果は下の「結果」。

## 解釈

- 「jpw1」は `jobs_per_worker = 1`。Issue 引用値 0.82〜0.94 は `docs/perf/cpu-gemm-m64-jobgrid-diagnosis.md` §4 の GB10 `jpw1` 腕と一致する。
- A の pack 共有（#3199 の G1）は含めない。`GB10_AFFINITY_ENABLED` は false のまま。
- 設計は `docs/cpu-gemm-small-m-job-merge-prepack-decision.md` §14。

## 仕組み

- ゲート `INFER_SCOPE_JPW1_ENABLED`（`crates/backend-cpu/src/gemm_blis/infer_scope.rs`、既定 OFF）。
- 推論本体（`predict` 系・`predict_resident`・`predict_resident_input`）がスコープに入る間だけ jpw を 1 に下げる。学習は対象外。
- rayon ワーカー上の GEMM はスコープを継がず既定の jpw になる。

## 腕・セル

| 腕 | 内容 |
|---|---|
| `off` | 対象 tree そのまま |
| `inf-jpw1` | `ab_patch_inf_jpw1.diff`（定数を true にする 1 行）を複製ツリーに当てる |

セルは bench-fandhe の train／infer × fresh／reuse、size=64、device=cpu。単体 6 形状と gemm タスクは推論スコープに入らないため置かない。

## 手順

1. マージ後、`RULE.draft.txt` を `RULE.txt` にコピーして記入し、単独コミットで固定する。
2. `aggregate.py` の sha256 を `RULE.txt` に書く（`python3 -I aggregate.py --self-test` が通ること）。
3. `<cuda-node>` で `orchestrate.sh` を実行する（環境変数はファイル冒頭を参照）。計測中はビルド・集計・転送を並走させない。
4. `python3 -I aggregate.py <ログディレクトリ>` で集計し、結果欄に記録する。ホスト名は `<cuda-node>`、`$HOME` は `<home>` に置換する。

## 対象 tree の差分確認

- tree_sha は `136705650`（RULE 項 11 (a)）。送ったリビジョンはノード上の `.rev-stamp`（`env/rev-stamp.txt`）で確かめた。
- `git diff --name-status v0.14.0 136705650 -- crates/` は 37 ファイル（`env/diff-v0.14.0-crates.txt`）。機構に関わるのは `crates/backend-cpu/src/gemm_blis/infer_scope.rs`（新規）・`small_m.rs`・`mod.rs` と、bit 一致テスト `crates/facade/tests/infer_scope_jpw1_4161.rs`。
- 腕ツリーの定数は `tree_verify.txt` で確かめた（off は `false`、inf-jpw1 は `true`）。patch の sha256 は `patch_sha256.txt`。

## 結果

### 計測の条件（実測）

- 系列: 2026-11-11T03:18:58Z〜03:28:53Z（`env/status.log`）。`orchestrate.sh` は rc=0（2 腕のビルド → 前提 report → 5 ラウンド。`env/orch.log`）。
- 実機ロック: owner は `#4161 measure 2026-11-11T03:18:58Z`（`env/lock-owner.txt`）。ビルドはロック取得後に、新しい展開先で行った。系列中はほかの計測・ビルド・転送を並走させていない。
- ゲート: 5/5 PASS（`round{1..5}/gate.txt`）。前提（RULE 項 6）: inf-jpw1 は infer_fresh・infer_reuse で applied>0、train の 2 パスで applied=0。off は 4 パスとも applied=0。両腕とも `gb10_affinity enabled=false`（`prereq/`）。
- 計測に使った bench-fandhe の fandhe-ai は、腕ツリーの `crates/facade` を path で参照している（`tree-off.txt`・`tree-inf-jpw1.txt`）。torch は 2.14.0+cu130（`env/torch-version.txt`）。
- マスク: `$HOME` は `<home>`、ホスト名は `<cuda-node>` に置換し、漏れがないことを確かめた。

### 集計（`aggregate.md`・`aggregate.tsv`。比 = inf-jpw1 / off、< 1.00 で inf-jpw1 が速い）

| セル | 中央値比 | ラウンド別 | inf-jpw1 の勝ち | NB-RANGE5（記録のみ） |
|---|---|---|---|---|
| train fresh | 0.9816 | 1.0034, 1.0147, 0.9816, 0.9783, 0.9147 | 3/5 | 0.9147〜1.0147 |
| train reuse | **1.0056** | 0.8691, 0.9581, 1.1111, 1.0056, 1.0139 | 2/5 | 0.8691〜1.1111 |
| infer fresh | 0.8977 | 0.8107, 0.9529, 0.7931, 0.8977, 0.9011 | 5/5 | 0.7931〜0.9529 |
| infer reuse | 0.8636 | 0.9645, 0.9050, 0.8636, 0.8596, 0.8337 | 5/5 | 0.8337〜0.9645 |

- **判定: REJECT**（RULE 項 8）。train reuse の中央値比 1.0056 が上限 1.00 を超えたため。推論 2 セルは ADOPT の条件（< 1.00）を満たしている。
- 事実の補足（判定は変えない）:
  - train の 2 パスは前提の probe で inf-jpw1 でも applied=0 で、機構は発火していない。
  - train reuse のラウンド別の比は 0.8691〜1.1111 にばらつき、NB-RANGE5 の幅は 0.2407 ある。
- NB-SEP（記録のみ。比 = PyTorch / fandhe、> 1.0 で fandhe が速い）: infer fresh・infer reuse とも不成立。
  - 変更後の比は infer fresh 0.8503〜0.9181、infer reuse 0.7970〜0.9620 で、PyTorch より遅いまま。
- 既定の反転（#4165）は ADOPT のときだけ行う。REJECT なので反転の根拠にはならない。
- 試走（RULE 項 11 (f)）の判定語も REJECT だった。
