#!/usr/bin/env bash
D="<scratch>/remeasure"
for i in 1 2 3 4 5; do mkdir -p "$D/m4max-090/run$i"
  if [ "$i" = 1 ]; then bash "$D/m4max-run-090.sh" "$D/m4max-090/run$i" > "$D/m4max-090/run$i/run.log" 2>&1
  else SKIP_BUILD=1 bash "$D/m4max-run-090.sh" "$D/m4max-090/run$i" > "$D/m4max-090/run$i/run.log" 2>&1; fi
  echo "run$i: $(tail -1 "$D/m4max-090/run$i/run.log")"
done; echo ALL-DONE
