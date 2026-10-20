# 計測ツリーの同定（`.rev-stamp` とツリー指紋。イシュー #3186）

GB10 ノードには bundle（`git bundle`）から作成したオフライン clone を置き、各ツリーに HEAD の sha を
`.rev-stamp` として置いて計測した（ノード側でネットワーク経由の取得は行っていない）。ツリー指紋は次の手順で取り、
ノードの作業ツリー（tracked ファイルの内容）と Mac 側の `git archive <sha>` 展開物を突き合わせた。

```
対象: crates と scripts/bench/framework-compare の全ファイル（Cargo.lock と target* を除く）
指紋: LC_ALL=C sort した相対パス順に sha256sum を取り、その出力全体をもう一度 sha256 にかけた値
```

| # | ツリー | `.rev-stamp` | ファイル数 | ノード側指紋 | Mac 側指紋（git archive） | 一致 |
|---|---|---|---|---|---|---|
| (i) | #2107 計測（detached） | `622594f15994e7c21641c5dd4497b764a334ade5` | 1785 | `5cdaf5deba4a92b2d723ff41cd4ef3567088323c738155caa0b4dbca148cc114` | `5cdaf5deba4a92b2d723ff41cd4ef3567088323c738155caa0b4dbca148cc114` | 一致 |
| (ii) | #2108 計測（作業ブランチ HEAD） | `0889838c8f2537141b21430913fcc6f469f6cc42` | 2409 | `54369ac4c6ec6873daadcefb2835b297459a24294855336b9f66256308763534` | `54369ac4c6ec6873daadcefb2835b297459a24294855336b9f66256308763534` | 一致 |
| (iii) | #3186 計測（同上） | `0889838c8f2537141b21430913fcc6f469f6cc42` | 2409 | （(ii) と同一ツリー） | （同上） | 一致 |

補足:

- (ii)・(iii) は同じ clone を使った。(iii) の `orchestrate.sh` は実行時に `.rev-stamp` と HEAD の sha の一致を検査する
  （`gb10/env_info.txt` の `git_head` と `rev_stamp` が同値）。(i) の `git_head` は
  `docs/perf/logs/cuda-gemm-readback-attribution-2107/gb10/env_info.txt`。
- (iii) の Layer A は `--config patch.crates-io.fandhe-ai.path=<HEAD の crates/facade>` で HEAD のコードを計測した
  （ビルドログで `fandhe-ai v0.11.0` が clone 内の `crates/facade` から Compiling されることを確認）。
- (iii) の bench-fandhe ビルドで framework-compare の `Cargo.lock` が書き換わる（path patch による）。(ii) の A/B
  スクリプトは tracked ツリーのクリーンを要求するため、(iii) の後・(ii) の前に `Cargo.lock` だけを元に戻した。
  指紋が `Cargo.lock` を除く理由はこれ。
- ノード側の作業ツリーには `target/` と `results/raw/` が存在する。ノード側の指紋は tracked ファイルの一覧だけを
  対象にして作業ツリー上の内容を実測した。
- 実行順序: (iii) → (i) → (ii)。(i) は (ii) より前のコミットで先に実行するという #2108 RULE の順序制約を、
  コミットの前後関係（`622594f15` は #2108 導入コミットより前）と実行の前後関係（#2107 の判定が出てから #2108 を実行）の
  両方で満たす。
