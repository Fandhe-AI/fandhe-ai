#!/usr/bin/env bash
# #3187 GB10 ノード上の計測ツリー作成（bundle から 3 clone。after のみ on-arm.patch を適用）
set -eu
SHA=d56c64ba87e825e0fa6b6e0958c456214d9083f6
mkdir -p ~/work/r3187 && cd ~/work/r3187
for d in before after runner; do
  [ -e $d ] && { echo "exists $d"; exit 1; }
  git clone -q ~/work/b3187.bundle $d
  git -C $d checkout -q --detach $SHA
  printf '%s\ndirty=0\n' "$SHA" > $d/.rev-stamp
done
for d in before after; do echo ".rev-stamp" >> $d/.git/info/exclude; done
git -C after apply docs/perf/logs/cpu-gb10-affinity-ab-2117/on-arm.patch
git -C after status --porcelain --untracked-files=no
for d in before after; do echo "== $d"; git -C $d ls-files --others --exclude-standard; git -C $d ls-files --others --ignored --exclude-standard | head -3; done
git -C runner status --short
echo setup-done
