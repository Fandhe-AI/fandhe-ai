# Mac mini Metal 常駐チェーンの時間の内訳（イシュー #3612）

## 状態

- **計測部品と事前登録規則の登録のみ。Mac mini 実機の計測は未実測**。結果は `docs/compat-resident-train-step-decision.md` §17.6 へ追記する。
- 背景: #3257 の参考系列で Mac mini の Metal 常駐チェーンが torch に届かなかった（train +14.3 us・infer +6.9 us）。§17.3 の 2 行は「Mac mini の内訳は未実測」のまま。

## 収録物

| ファイル | 内容 |
|---|---|
| `RULE.txt` | 事前登録規則（区間定義・5 round・負荷ゲート・hard 条件・突き合わせ・振り分け・仮説・限界） |
| `crates/backend-metal/src/resident_chain_breakdown_diag_tests_3612.rs` | B0／B1 の `#[ignore]` 診断テスト（`cfg(test)`。lib には入らない） |
| `scripts/bench/framework-compare/run_breakdown_resident_chain_metal.sh` | Mac mini 用ランナー（5 round。テストバイナリを計測前にビルド） |
| `scripts/bench/framework-compare/summarize_breakdown_resident_chain_metal.py` | RULE の hard 条件と振り分けを fail-closed に適用する集計（`..._test.py` は CI で実行） |

## 実行手順（実測時）

1. HEAD を bundle で Mac mini の専用ディレクトリへ置き、`.rev-stamp` を作って `.git/info/exclude` へ足す。`docs/real-hardware-verification-env.md` §7 の機体ロックを取る。
2. `bash scripts/bench/framework-compare/run_breakdown_resident_chain_metal.sh <label>`（nohup サブシェルで切り離し、生存確認）。
3. A0 は `scripts/bench/framework-compare/run_a0_resident_chain_metal_3612.sh`（計測前にコミット済み。末尾「A0 の手順」）で取る。旧記述: A0 は #3257 の `bench-resident-chain.patch` を当てた bench-fandhe（`--device metal --mode chain`）を別ランで取り、step 中央値 5 値を控える。
4. 生ログを回収し、ホスト名・`$HOME`・ユーザー名をマスクする。`python3 -I scripts/bench/framework-compare/summarize_breakdown_resident_chain_metal.py --dir <dir> --label <label> --a0-train-us ... --a0-infer-us ...` を終了コードつきで保存する。

## 注意

- 開発機（M4 Max）の値は記録に使わない（煙試験のみ）。B2（群別 GPU 時間）・A1・torch 同席は未実装・未実施（RULE.txt の「腕」節）。
- 原因が分かってもコードは直さない（是正の採否は所有者判断）。

## 計測ツリー（実測時の固定）

- B0／B1: `3e5d41016`（#3627。RULE を含み、`crates/` の `.rs`・`.metal` の `a8e91f0c9` との差は診断テストと `lib.rs` の `mod` 1 行だけ）。後続コミットは条件外のファイルを触るため使わない。
- A0: `a8e91f0c9` に `docs/perf/logs/resident-chain-ab-3257/bench-resident-chain.patch` を当てたツリー。`3e5d41016` は bench-fandhe のピン（`=0.11.0`）と facade 0.12.0 が食い違い path patch に解決できないため（RULE [限界]）。bench-fandhe・bench-common は #3257 のランナー `786d18b50` と同一で、本番の `crates/` は B0／B1 と同一。

## A0 の手順

1. `a8e91f0c9` を bundle で clone し、`.rev-stamp`（HEAD／`dirty=0`）を作って `.git/info/exclude` へ足し、パッチを適用する。
2. `A0_TREE=<abs> bash run_a0_resident_chain_metal_3612.sh <label>`（スクリプトはどちらのツリーにも入れず別に置く）。HEAD・パッチ完全一致を検証し、ビルド後に 5 round（cooldown と load1<3.0 ゲート。train→infer 固定）で `--device metal --mode chain` を取り、`a0-3612-<label>-us.txt` に `train=…`／`infer=…`（us・5 値）を出す。
3. その 5 値を集計の `--a0-train-us`／`--a0-infer-us` へ渡す。checksum と PARAM_DIGEST は記録のみ（判定条件にしない）。
