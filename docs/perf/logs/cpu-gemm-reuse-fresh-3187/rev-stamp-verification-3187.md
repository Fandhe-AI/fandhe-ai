# 計測ツリーの同定（`.rev-stamp` とツリー指紋。イシュー #3187）

GB10 ノードには bundle（`git bundle`）から作成したオフライン clone を 3 つ置いて計測した（ノード側でネットワーク経由の取得は行っていない）。
どのツリーにも `.rev-stamp`（1 行目 HEAD の sha・2 行目 `dirty=0`）を置いた。`docs/real-hardware-verification-env.local.md` は転送していない。

| ツリー | 用途 | `.rev-stamp`（1 行目） | 差分 |
|---|---|---|---|
| runner | 帰属 5 run（`orchestrate.sh`）・補助 1 run・A/B スクリプトの実行元 | `d56c64ba87e825e0fa6b6e0958c456214d9083f6` | なし |
| before | affinity A/B の before 腕（`GB10_AFFINITY_ENABLED = false`） | 同上 | なし |
| after | affinity A/B の after 腕 | 同上 | `on-arm.patch` のみ（コミットしない。`GB10_AFFINITY_ENABLED` 宣言 1 行） |

`.rev-stamp` は before／after では `.git/info/exclude` に追記した。`run_ab_gb10_affinity_cpu.sh` の
`verify_no_untracked_build_inputs` は未追跡ファイルが 1 件でもあると停止するが、`ls-files --others --exclude-standard` は
`info/exclude` を尊重するため未追跡一覧に出ず、ignored 側の検査（`.cargo/`・`rust-toolchain`・`Cargo.*`・`build.rs`・`*.rs` のみが対象）にも当たらない。
実際にスクリプトの `verify_same_commit_and_diff` を通過している（A/B は開始した）。

ドリフトガード（RULE.txt）: `gb10/env_info.txt` の `rev_stamp_head` は `d56c64ba87e825e0fa6b6e0958c456214d9083f6`（RULE.txt を含むコミット）、
`rule_sha256` は `d57ba0221ef9f20fbcb4f128c36ecbf0c5071929a6f7a6d600ce5a1cc8bf26f3` で、`shasum -a 256 RULE.txt` の値と一致する。

## ツリー指紋

```
対象: crates と scripts/bench/framework-compare の tracked ファイル（Cargo.lock と target* を除く）
指紋: LC_ALL=C sort した相対パス順に sha256sum を取り、その出力全体をもう一度 sha256 にかけた値
```

Mac 側は `git archive d56c64ba8` の展開物（after 相当は展開物へ `on-arm.patch` を `patch -p1` で適用）で同じ手順を取った
（ノード・Mac とも同一スクリプト。`session-scripts/` には含めない一時スクリプト）。

| ツリー | ファイル数 | ノード側指紋 | Mac 側指紋 | 一致 |
|---|---|---|---|---|
| runner | 2415 | `87cde29ed5f2f2ae7c1190e66fe50daaefa2b9bac2e1576bca354008d49e9640` | `87cde29ed5f2f2ae7c1190e66fe50daaefa2b9bac2e1576bca354008d49e9640`（git archive） | 一致 |
| before | 2415 | `87cde29ed5f2f2ae7c1190e66fe50daaefa2b9bac2e1576bca354008d49e9640` | 同上 | 一致 |
| after | 2415 | `e8179beae2ddeaf2a66b2b7f4bcd3576ffe0809815dde733da178e0f5eb842cb` | `e8179beae2ddeaf2a66b2b7f4bcd3576ffe0809815dde733da178e0f5eb842cb`（git archive + patch） | 一致 |

補足:

- 指紋を `Cargo.lock` で除く理由: bench-fandhe のビルドで framework-compare の `Cargo.lock` が path patch により書き換わるため（`bench_fandhe_lock_restore.sh` が復元する）。
- ノード側の `tree-2117-{before,after}-*.txt` は A/B スクリプトが取った `cargo tree` の出力で、両腕のビルドが各 checkout の `crates/facade` を指していることを示す。
- 実行順序: 帰属 5 run → 補助 1 run → affinity A/B（`session-scripts/steps.log`。ビルドの並走なし）。
