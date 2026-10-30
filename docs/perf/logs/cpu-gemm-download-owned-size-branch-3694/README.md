# #3694: matmul_resident のサイズ分岐（D'）の A/B

## 状態

**未実測**。事前登録規則（`RULE.txt`）の固定と、サイズ分岐の実装・単体テスト・小 N 記録セルの診断関数までは済んだ。
GB10 と Mac mini の A/B（各機体 3 系列以上・系列間 30 分以上・hw-lock）は、本作業の実行環境から実機へ到達できず行っていない。
判定器の出力（`judge-3694.md`）・マスク済み生ログ・rev-stamp 照合は存在しない。他機体（M4 Max・Linux）の値で代替していない。
RULE の「結論後」節に従い、未実測のため本番のサイズ分岐は main へ入れず、パッチ `size-branch.patch` として残した。

## 中身

| ファイル | 内容 |
|---|---|
| `RULE.txt` | 事前登録規則（単独コミットで固定済み。編集しない） |
| `size-branch.patch` | 本番のサイズ分岐（閾値 262144 B 以下は `download`・超えると `download_owned`）と単体・CPU bit 一致テストの差分 |

小 N（64・128）の記録セル用に `crates/facade/tests/cpu_gemm_resident_reuse_diag.rs` へ `cpu_gemm_resident_reuse_phases_small_n`（`#[ignore]`）を足した。既存の関数・反復数は変えていない。

## 検証済み（実機なし）

- サイズ分岐を当てた状態で `cargo test -p fandhe-ai-autodiff --lib size_branch`・`cargo test -p fandhe-ai --test resident_input_inference`（閾値を超える形状 (256, 8, 257) を含む CPU の bit 一致）が通った
- `cargo clippy -p fandhe-ai-autodiff -p fandhe-ai --all-targets -- -D warnings` が通った
- 診断テストバイナリ（小 N 関数を含む）のビルドが通った
- 未検証: 腕の木の作成（`make_before_tree.sh`）、`build_ab_arms.sh` のハード条件、セル集合の一致、合成判定器、実機の A/B

## 次にやること

1. 所有者が A/B の実施方針（機体到達・時間枠）を確認する
2. `make_before_tree.sh`・`orchestrate.sh`・`session-3694.sh`・`judge_3694.py`（#3681 の写しと合成判定）を用意する
3. 両機体で 3 系列以上を測り、判定が ADOPT なら `size-branch.patch` を適用して本番へ入れる
