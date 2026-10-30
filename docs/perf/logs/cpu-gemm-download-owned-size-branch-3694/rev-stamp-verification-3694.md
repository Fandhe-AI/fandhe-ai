# #3694 計測ツリーの同定（GB10 専有 3 系列・Mac mini 3 系列・2026-10-30）

計測機は `<cuda-node>`（GB10・aarch64・compute プロセス 0 件の専有機体）と `<metal-node>`（Mac mini・Apple M4・aarch64）。
`/home/<user>`・`/Users/<user>` 配下のパスは `<home>` へマスクした（数値は変えていない）。ホスト名・ユーザー名は回収後の grep で 0 件を確認した。
マスク前後で `judge_3694.py` の出力（`judge-3694.md`。判定・ratio・全表）は完全に一致した（`diff` が空。系列の指定パスを同じ相対パスにして比較した）。

## 使わなかった系列（記録）

GB10 の最初の試行（s1 のみ）は、専有機体ではない別の機体（vLLM 2 件・ComfyUI 1 件が常駐し compute プロセス 3 件）に誤って投入した。
`excl_check` が `main excl=fail` を返し、RULE のゲート（compute プロセス 0 件）を満たさないため判定にも記録にも使っていない（値は見ていない）。
その機体の連鎖は止め、専有の機体へ木・事前ビルド・3 系列をすべてやり直した。以降の GB10 の記録はやり直した専有機体のものだけ。
やり直した 3 系列はいずれも s1 から数え直しており、「ゲートを通った最初の 3 系列」の規則（RULE）に反しない（誤投入の系列はゲート不通過）。

## ツリー

| 項目 | 値 |
|---|---|
| RULE.txt の固定 | PR #3704 の最初のコミット `7078523ffb`（2026-10-30T04:58:19Z。squash のため main の祖先には無い）。RULE.txt の sha256 は main の版と一致（下記）。最初の計測開始（Mac mini s1・05:45:27Z）はこれより後 |
| runner | `207eceb5274e79539f75d4218681389c9e89c03f`（after と同じ木。部品〈orchestrate・session・judge〉を含む） |
| after | `207eceb5274e79539f75d4218681389c9e89c03f`（`19d22f948` に `size-branch.patch`〈sha256 `7fc5c018b1dfb7140e6179b69e9f036351eaa2b2f24fd551001d319409f424b8`〉を当てたコミット。PR は squash されるため main に同じ sha は残らない。木はパッチの再適用で再現できる） |
| before | `a7aac2f0227add243845418e0897ac0675c5b622+harness-overlay(19426afd4ef2)`（a7aac2f0 の木に、`scripts/bench/framework-compare/`〈`target*` 除く〉と `crates/facade/tests/cpu_gemm_resident_reuse_diag.rs` を after の木から置換した木） |
| `.rev-stamp` 2 行目（3 本とも） | `dirty=0` |
| 置換内容の指紋（12 桁の元） | 置換した全ファイルを相対パスで `LC_ALL=C sort` し `<sha256>  <path>` の一覧全体を sha256。全体値は `19426afd4ef291b2d1ea5a180c7ec68be3bebede483d8a410e7e725d3a0c70d6`（`make_before_tree.sh` の出力） |
| 転送方式 | `git archive` で作った木から `make_before_tree.sh` で before を作り、ssh の tar へ流して展開（rsync は不使用）。GB10・Mac mini とも 3 本（runner・before・after） |
| ノード上の `*.local.md`・`.env*`・`settings.local.json` | `find` で両機体とも 0 件 |
| `7078523ffb` と after の差 | `git diff --name-only 7078523ffb <after> -- crates ':!crates/facade/tests'` は `crates/autodiff/src/optim/resident_input.rs` だけ（RULE の条件。外れていない） |
| after をマージ前に外した理由 | 判定が REJECT のため、本 PR は `size-branch.patch` と結果だけを残し、`crates/` を main のままにする（RULE「結論後」） |

## ツリー指紋（`crates` と `scripts/bench/framework-compare` の通常ファイル。`Cargo.lock`・`target*` を除く。相対パスを `LC_ALL=C sort` し `<sha256>  <path>` の出力全体を sha256）

ローカルの展開木と両ノードの展開木で一致した。

| 木 | 指紋 |
|---|---|
| runner | `d32ca0cf3caae6ff1e2c0dcd9d322c6728aa013329914610d393a7bf4a488569` |
| after | `d32ca0cf3caae6ff1e2c0dcd9d322c6728aa013329914610d393a7bf4a488569` |
| before | `f77a82a2332d5b283fad429bcbbeeb7ddf72f3066ba396036fd2f2fd7a4e579d` |

## ビルドと腕のバイナリ

- ローカル（M4 Max）でも `build_ab_arms.sh` を 1 回通し、ハード条件（両腕で診断テストと framework-compare が同一・両 `.rev-stamp` が dirty=0 で sha が異なる）と両腕のビルド成否を確かめた。値は記録にも判定にも使っていない
- ローカルで両腕の診断（本体 130 セル・小 N 52 セル）と bench（`--size 64`・`128`・`256` の phase 5 種）を 1 回ずつ流し、出力セル集合（arm・n・phase の組と bench の phase 名）が腕の間で同じであることを確かめた（値は捨てた）。`bench-fandhe` は `--size 64`・`128` を受け付ける
- 事前ビルドは各機体で `build_ab_arms.sh` を 1 回だけ実行（hw-lock を取る前。計測中はビルドしない）。出力は `arms/gb10-arms.txt`・`arms/macmini-arms.txt`

| 機体 | diag-before sha256 | diag-after sha256 | bench-before sha256 | bench-after sha256 | rustc |
|---|---|---|---|---|---|
| gb10 | `264841a5d2f77...` | `c2d04ec5f77c6...` | `29b03913555c1...` | `3c6067d29a7e8...` | 1.97.1 |
| macmini | `6f2734d1a5c43...` | `bdcaf92c963b6...` | `b3003f468bc34...` | `3d8d4dffa9429...` | 1.99.0 |

（全桁は `arms/*-arms.txt`。）

## 規則・スクリプトの同定

- 6 系列すべての `env_info.txt` の `rule_sha256` が `c461a59cf74994cc0e67b96d8510ac8c7879ffdf687cbbcfa865448ed8dd9046`（`7078523ffb` の RULE.txt・main の版と一致。`judge_3694.py` も検査する）、`rule_3681_sha256` が `48b3da16e68c53cf111a14e0fb431d3cb149f5816705e43c04339002dcf7614e`
- 流用元の sha256（編集していない）: `path_judge.py` `1afa378d9dfd84ec278b0ffb2c684ea518f604c4fd7f8dfeedea9337ba4ad535`・`ab_judge.py` `748deb3bf7bf3cb55244aaf4961031db9b18fdbea9917ff7b867133491252af4`・`excl_check.py` `ab9dcae3bc53fdf29ab31b7be84cd56764dff70791df0285e8d5e26f488d6ab9`・`build_ab_arms.sh` `07fb2d64864b5c3519de4d70283ac3ed49a4885758bd94ba328ecd60964d0fc3`
- 本 PR の部品: `orchestrate.sh` `bdbd87b547f951f42f39cb55513cb36a3a1b7cd1c4ad7c50e2f90725541c1665`・`session-scripts/session-3694.sh` `b7908561c1dc56595c0757ddce4a8470fc676ade198a92b7db0f9ad07eeefd3c`・`make_before_tree.sh` `f9f466e8f89c02e7d85edbc0d65b5c3f2e0f23fbfe668abe264eecb246f7fd51`（計測に使った版と同一）
- `judge_3694.py`: 計測の runner 木（`207eceb52`）の版は sha256 `0455bf9a817e46e3865e9f3236fc51bd87ed2a4eb4b35efeeb6d8b0e6fd29ca8`。計測後に表示名の関数 `series_name`（系列名に 1 つ上の階層名を添える。全系列が `gb10`・`macmini` と同名で区別できなかったため）だけを足した版（`e6c0af9100b3d792e59694b46f7e11133181bebb6eec8e43658d4ef1d2da2280`）で `judge-3694.md` を出した。差分は `series_name` の追加とその呼び出しへの置換だけで、判定ロジックは変えていない（self-test PASS。runner の版での再出力は未実施）
- 系列内の参考表示（各系列の `judge-<機体>.md`）は runner の版の `--single` 出力

## 系列の時刻と順（judge が検査しない項目を手で確認）

起動は各機体で 3 系列を `s1`（before 先頭）・`s2`（after 先頭）・`s3`（before 先頭）の順に連鎖させた。間隔は前の系列の終端行の後に 1860 秒あけて次を起動した。

| 機体 | 系列 | 先頭腕 | hw-lock 取得（UTC） | 解放（UTC） | 前の解放からの間隔 | ゲート |
|---|---|---|---|---|---|---|
| gb10 | s1 | before | 06:22:58 | 07:01:19 | - | 本体 5/5・小 N 5/5 run pass・`excl=pass` |
| gb10 | s2 | after | 07:32:19 | 08:07:36 | 31 分 00 秒 | 本体 5/5・小 N 5/5 run pass・`excl=pass` |
| gb10 | s3 | before | 08:38:36 | 09:13:16 | 31 分 00 秒 | 本体 5/5・小 N 5/5 run pass・`excl=pass` |
| macmini | s1 | before | 05:45:27 | 06:31:33 | - | 本体 5/5・小 N 5/5 run pass |
| macmini | s2 | after | 07:02:47 | 07:45:52 | 31 分 14 秒 | 本体 5/5・小 N 5/5 run pass |
| macmini | s3 | before | 08:16:52 | 08:59:57 | 31 分 00 秒 | 本体 5/5・小 N 5/5 run pass |

6 系列とも `orchestrate` の終了コードは 0（`steps.log` の `done. rc=0`）。ゲートを通らなかった系列はなく、s4・s5 は不要だった。
