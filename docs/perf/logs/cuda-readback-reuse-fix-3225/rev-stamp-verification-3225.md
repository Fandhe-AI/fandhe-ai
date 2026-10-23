# .rev-stamp 突合（イシュー #3225）

- 計測 sha: `73768af83c0f45b482f1a209d4539a4f06b8e0bc`
- ノード側 `.rev-stamp`: `73768af83c0f45b482f1a209d4539a4f06b8e0bc`（git bundle から作った clone の `git rev-parse HEAD`。
  manifest の `script_repo_head_sha`／`facade_head_sha` とも一致）
- 指紋: `crates` と `scripts/bench/framework-compare` の tracked ファイルから `Cargo.lock` と `target*` を除き、
  `LC_ALL=C` ソートした相対パス順に sha256 を取り、その一覧全体の sha256 をとった値

| 側 | ファイル数 | 指紋 |
|---|---|---|
| GB10 ノード（clone） | 2453 | `d01805bc72bb477c32525729a52a64e0e4fd14fe55e32fa7a1d0d8a23248335d` |
| 手元（`git archive 73768af` 展開） | 2453 | `d01805bc72bb477c32525729a52a64e0e4fd14fe55e32fa7a1d0d8a23248335d` |

結果: 一致。ノード側の tracked ツリーは A/B 終了後も clean（未追跡は `.rev-stamp` と results/raw の生成物のみ）。
