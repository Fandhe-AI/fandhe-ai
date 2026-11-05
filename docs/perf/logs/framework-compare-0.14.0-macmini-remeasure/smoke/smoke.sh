export ROOT=$HOME/work/r0140/tree D=$HOME/work/r0140 EXPECT_SHA=18a7f4b87fe61029a179cb1922cc82928440f2c9
BUILD_ONLY=1 bash $D/macmini-run-0140.sh $D/smoke-build || { echo BUILD-FAILED; exit 1; }
SKIP_BUILD=1 bash $D/macmini-run-0140.sh $D/smoke
echo "smoke rc=$?"
