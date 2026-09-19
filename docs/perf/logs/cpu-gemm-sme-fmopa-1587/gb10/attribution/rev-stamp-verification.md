# 腕同定の独立検証（イシュー #2053）

- 展開元: Mac 側 `git archive main`（main `ac3e0639fedf7ba9e5621174530d108f139595da`）を
  `rsync -a --delete --exclude target --exclude 'target-*' --exclude .git` で GB10 の
  `<home>/work/rust-ai-library-run/` へ転送し、`.rev-stamp` に同 sha を書き込んだ。
- 一覧: 展開ディレクトリの `find . -type f`（`.rev-stamp` を除く）5,598 行を
  `<home>/work/filelist-2053-gb10.txt` として転送（`git ls-files` ではなく展開実体から生成）。
- Mac 側指紋: 展開ディレクトリで `shasum -a 256` を一覧に対して実行し、`./` 接頭辞を除いて
  `fp-mac.txt` へ保存（5,598 行）。
- GB10 側指紋: `orchestrate_attribution_gb10.sh` が実測前に同一一覧へ `sha256sum` を実行した
  `fp-before.txt`（5,598 行。行数が一覧と一致することをスクリプトが assert）。
- 突合: `diff fp-mac.txt fp-before.txt` → 差分 0 行（Mac 側で実行・2026-09-19 UTC）。
- 腕: `rev_stamp_before=ac3e0639…`・`rev_stamp_prime=ac3e0639…+on-arm-prime.patch`・
  `rev_stamp_after=ac3e0639…+on-arm.patch`（`env_info.txt`）。各パッチ腕と before の差分は
  `fp-diff-{prime,after}.txt` のとおり `crates/backend-cpu/src/gemm_blis/mod.rs` の 1 件のみ。
- パッチ sha256: `patch_sha256.txt` の `on-arm-prime.patch` 値 `024d4fe2…` はリポジトリに
  コミット済みの `gb10/on-arm-prime.patch` の sha256 と一致（Mac 側で確認）。
