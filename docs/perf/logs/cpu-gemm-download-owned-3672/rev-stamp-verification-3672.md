# #3672 腕の同定（.rev-stamp・ツリー指紋・バイナリ sha256）

| 木 | sha | `.rev-stamp` 2 行目 | ファイル数 | ツリー指紋（.rev-stamp を除く全ファイルの sha256 一覧の sha256） |
|---|---|---|---|---|
| runner（計測部品のコミット） | `a36f2efecbac9894114d0e6f4d61c986fe2f6c37` | `dirty=0` | 15550 | `43cc50306131b065ea7bd69cb967e978117d2fb28cec7c5234d6a485099472e7` |
| before（#3677 のマージベース） | `a7aac2f0227add243845418e0897ac0675c5b622` | `dirty=0` | 15543 | `5daf70be835a1d65cded437fb7de259e8a1d213d108286a7251d35cae33e1b10` |
| after（#3677 の PR head。tree は `e03f81957` と同一 `0e3514cd…`） | `f7ce20fc91730216e8f3f60514b181f32ebfee48` | `dirty=0` | 15544 | `1078d1e2943d4bc3ac5cf0ec4f8f0a7035c083a4e5a26b14b8c36ba3c625f4a1` |

- 指紋はローカルの `git archive` 展開物（Mac 手元）と、`<cuda-node>`・`<metal-node>` の展開木で完全一致した（ビルド後に再計算しても不変＝framework-compare の Cargo.lock は復元済み）。
- 腕の取得は `git archive` の tar を gzip で転送して別ディレクトリへ展開（rsync は使わない）。`.local.md` は送っていない。
- ハード条件（`git diff a7aac2f02 f7ce20fc9 -- crates/facade/tests/cpu_gemm_resident_reuse_diag.rs scripts/bench/framework-compare/` が空）は実行前にローカルで確認し、`build_ab_arms.sh` が両木の `diff` で再度強制した。両腕とも `crates/facade` の version は 0.12.0（`version.workspace`）で path patch が適用される。
- bench-fandhe は両腕とも各木の `crates/facade` への path patch（registry の `fandhe-ai =0.12.0` は不使用。`cargo tree` で解決先を確認してからビルド）。

## 規則ファイルの sha256

| ファイル | sha256 |
|---|---|
| `RULE.txt` | `03d4c5f4f3dd08d9f076bfa879ccb8ca9c983005bf5ea5e06663cb1f2ed793be` |
| `RULE.v2.txt` | `f07439fd88ee335028dee3d6a2bb1b495281099c4ab7a1d82fa804538a9d1ddc` |

各機体の `env_info.txt` に同じ値が記録されている。RULE.v2.txt の先頭の `fixed-at-utc: 2026-10-28T13:30:00Z` は記入時の誤り（実際のコミットは 2026-10-28T14:00Z 頃。コミット時刻は `git log`）。計測の開始（14:15Z）より前に固定した事実は変わらない。事後に書き換えると sha256 が変わるため、そのままにして本節で訂正する。

## バイナリ sha256

| バイナリ | `<cuda-node>` | `<metal-node>` |
|---|---|---|
| bench-before | `a970e5bccb69777054db3e7e5ab31372331e6c3b61deaf28ceb89b0d3edabec8` | `5eb3b38f456a449f2d1450f631424b41b43f2964082c0766e038be2b0c6322c2` |
| bench-after | `e47373308941fb1fd576322a54bb69e7a2e975ecf04bb255881d19475b7019ad` | `2b47c47719cc586f00d464bd34eca375c2c55b855e7eb48317ac7340a22209b2` |
| diag-before | `103ad827ac2af8feff47d710845d87d95adf15b6950922d7a8d414bef9b53733` | `da0c8b82c95cfe360ec6270a8ebaff8cfc0cf86e95730e526975fe0f8e2e880c` |
| diag-after | `5796acc84e60b5a50d0d79dccf37faaa353b76fb56ccdf3b605e58150fd2e9ed` | `6152f48de487065d00da69fc91549a0287393f57c3f3ffd24187df0c892afcd7` |

`orchestrate.sh` は各 run の起動前にバイナリの sha256 が `arms.txt` と一致することを確認している。

## 機体と toolchain

| 機体 | OS | CPU | rustc |
|---|---|---|---|
| `<cuda-node>`（GB10） | Linux 6.17 aarch64 | 20 コア（Cortex-X925 ＋ A725） | 1.97.1 |
| `<metal-node>`（Mac mini） | Darwin 27.0 arm64 | Apple M4 10 コア（P4・E6） | 1.99.0 |

## 実行の経緯

- 計測は各機体 1 回の起動（ラベル `s1`）。起動のやり直し・run の差し替え・追加・セル除外はない。
- GB10: hw-lock 取得 14:15:47Z〜終了 14:46:10Z。5 run 全てで load1 < 1.0 のゲートに pass。専有の poller は 30 秒間隔で 61 件、全件 compute プロセス 0 件（`excl_check.py --series main` は `main excl=pass`）。
- Mac mini: 14:16:03Z〜14:57:10Z。5 run 全てで load1 < 3.0 に pass（直前の自分の run の負荷が下がるまで待つため、run 4 は load1 11.13 から待機した）。run の起動順は ABBA（奇数 run は before→after、偶数 run は after→before）。
- 計測後に `ab_judge.py` を 1 点だけ修正した（Mac mini の診断 JSONL で、判定に使わない区間行〈`tape_build` 等〉の `median_s` がタイマー分解能で 0.0 になり、全行の正値検査で入力不備になったため。判定に使う行〈iter_total・download〉の検査は維持し、使わない行の検査だけを外した。閾値・判定式は不変。self-test に 0.0 行の fixture を追加）。修正前の判定器は GB10 に対して REJECT（同じ項目）を出していた。
