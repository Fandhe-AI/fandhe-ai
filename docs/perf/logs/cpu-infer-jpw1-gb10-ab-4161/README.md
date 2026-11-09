# #4161 GB10 の推論だけに jpw1 を効かせる A/B

状態: **道具立ての時点。実測なし**（未検証）。実測と判定の記録を含む最後の PR にだけ `Closes #4161` を付ける。

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

`git diff v0.14.0 <tree_sha> -- crates/` の一覧と `.rev-stamp` を収録する（実測時に記入）。

## 結果

未実測。
