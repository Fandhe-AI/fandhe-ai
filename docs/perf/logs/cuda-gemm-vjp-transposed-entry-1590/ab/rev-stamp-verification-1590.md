# #1590 train A/B 両腕の `.rev-stamp` とツリー内容指紋の独立検証記録

## 目的

PR #1909 の codex-review P2 指摘（comment 4022716791）への対応。
`docs/perf/cuda-gemm-vjp-transposed-entry.md` §3.3「腕の同定」は当初、
before／after 両腕のコミット同定が実行担当者の申告のみで、README
「実行手順 1」が要求する `.rev-stamp` 等の独立検証記録を成果物に含めて
いなかった。本記録は DGX Spark GB10 ノードに残存していた展開ツリーから
`.rev-stamp` を回収し、さらにツリー内容そのものを `git archive` の出力と
ファイル単位で突き合わせることで、§3.3 の計測が事前登録どおり
`82058501`（before）／`ab0b77d0`（after）の比較であったことを独立に裏付ける。

## 1. `.rev-stamp` の回収（2026-09-16）

| 腕 | 展開ツリー（DGX Spark GB10 ノード） | `.rev-stamp` |
|----|----|----|
| before | `/home/<user>/work/ab-trees/before-1590/.rev-stamp` | `820585014a6d4493fc7be3711177af55660f7a02` |
| after | `/home/<user>/work/ab-trees/after-1590/.rev-stamp` | `ab0b77d0b23369603c02ee9ee2335d8488fde791` |

いずれも Mac 側 clone での `git rev-parse 82058501`／`git rev-parse
ab0b77d0` の結果と一致する。値の写しは同ディレクトリの
`rev-stamp-before-1590.txt`／`rev-stamp-after-1590.txt`。

`.rev-stamp` は展開時に手で書かれたファイルであるため、これ単独では
「申告」の域を出ない。そこで次節のとおりツリー内容自体を突き合わせた。

## 2. ツリー内容指紋の突合（独立検証）

### 2.1 DGX 側（各展開ツリーのルートで実行）

```sh
find crates scripts/bench/framework-compare -type f -print0 \
  | LC_ALL=C sort -z | xargs -0 sha256sum
```

出力: `tree-hashes-dgx-before-1590.txt`／`tree-hashes-dgx-after-1590.txt`
（相対パスと SHA-256 のみ。ビルド成果物 `target-ab-*` 配下のエントリも
そのまま含む）。

### 2.2 archive 側（任意の clone で再導出可能）

```sh
for pair in 82058501:before ab0b77d0:after; do
  sha="${pair%%:*}"; arm="${pair##*:}"
  d="$(mktemp -d)"
  git archive "$sha" crates scripts/bench/framework-compare | tar -x -C "$d"
  ( cd "$d" && find crates scripts/bench/framework-compare -type f -print0 \
      | LC_ALL=C sort -z | xargs -0 sha256sum ) > "tree-hashes-archive-${sha}.txt"
  rm -rf "$d"
done
```

出力: `tree-hashes-archive-82058501.txt`／`tree-hashes-archive-ab0b77d0.txt`。
（macOS で `sha256sum` が無い場合は `shasum -a 256` に読み替える。
出力形式は同一）

### 2.3 除外と突合

DGX 側の一覧から次の 2 種類だけを除外し、archive 側と一覧全体を比較した。

| 除外対象 | 理由 |
|----|----|
| `scripts/bench/framework-compare/Cargo.lock` | `run_ab_vjp_transposed_cuda.sh` は各腕のツリー内で `crates/facade` への path patch を当てて cargo build するため、cargo が lock を書き換える。ソースツリーの同一性判定とは無関係 |
| `scripts/bench/framework-compare/target-ab-vjp-transposed-1590-{before,after}/` | 同ビルドの成果物ディレクトリ（`CARGO_TARGET_DIR`）。`git archive` には存在しない |

上記以外（`crates/` 配下の fixture `Cargo.lock` を含む全ファイル）は
除外していない。突合コマンド:

```sh
norm() { grep -v 'target-ab-vjp-transposed-1590-' "$1" \
         | grep -v 'scripts/bench/framework-compare/Cargo.lock$' \
         | sed 's/  */ /'; }
for arm in before after; do
  sha=$([ "$arm" = before ] && echo 82058501 || echo ab0b77d0)
  diff <(norm tree-hashes-dgx-$arm-1590.txt) <(norm tree-hashes-archive-$sha.txt) \
    && echo "$arm: MATCH"
  norm tree-hashes-dgx-$arm-1590.txt | wc -l
  norm tree-hashes-dgx-$arm-1590.txt | sha256sum
done
```

（`sed 's/  */ /'` は `sha256sum` の区切り 2 空白を正規化するだけで、
パス・ハッシュ値には触れない）

### 2.4 結果（2026-09-16・Mac 側で実施）

| 腕 | 対象コミット | 除外後ファイル数 | 一覧全体の SHA-256（DGX 側） | 一覧全体の SHA-256（archive 側） | 判定 |
|----|----|----|----|----|----|
| before | `82058501` | 725 | `55f6f7d98e3c496b8c6c61d23c7c5e78839e0052a87b36aa32163ec9d79bf421` | 同値 | **MATCH** |
| after | `ab0b77d0` | 727 | `a24ddedeb633abc99baa2fa08db8fa3f574bbe157d8bea293cb2d7638dabaa90` | 同値 | **MATCH** |

`diff` の出力は両腕とも空（ファイル集合・各ファイルの SHA-256 が完全一致）。
after の +2 ファイルは #1214 で新規追加された
`crates/backend-cuda/tests/gemm_transposed_parity.rs`・
`gemm_transposed_perf.rs` であり、`git diff --stat 82058501 ab0b77d0 --
crates`（8 files changed・新規 2 件）と整合する。

## 3. 結論

- §3.3 の計測に使われた before／after ツリーは、`.rev-stamp` の値と
  ツリー内容指紋の両方で、事前登録した `82058501`／`ab0b77d0` と一致する
- したがって §3.3 の差分は #1214（VJP 転置入口）単独の効果として扱える
  （既存記録 `ab/tree-{before,after}-1590.txt` の `cargo tree` 表示・
  `ab/sha-{before,after}-1590.txt` のバイナリ SHA-256 を補完する）
- 本記録は同定の裏付けのみを追加するものであり、§3.3 の数値・§4 の
  verdict（ADOPT）・tolerance・baseline は変更しない

## 4. 本ディレクトリの関連ファイル

| ファイル | 内容 |
|----|----|
| `rev-stamp-before-1590.txt`／`rev-stamp-after-1590.txt` | DGX 展開ツリーから回収した `.rev-stamp` の値 |
| `tree-hashes-dgx-before-1590.txt`／`tree-hashes-dgx-after-1590.txt` | DGX 側の全ファイル SHA-256 一覧（除外前・`target-ab-*` 含む） |
| `tree-hashes-archive-82058501.txt`／`tree-hashes-archive-ab0b77d0.txt` | `git archive` 側の全ファイル SHA-256 一覧 |

内部ホスト名・ユーザー名は含めない（パスは `/home/<user>/...` でマスク）。
