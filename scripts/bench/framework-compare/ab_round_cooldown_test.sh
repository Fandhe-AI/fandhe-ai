#!/usr/bin/env bash
# ab_round_cooldown.sh の self-test（イシュー #3184 再開分）。
# 検査: 未設定時 no-op かつファイル非作成／pass／timeout 境界／unavailable／非数値・範囲外閾値の拒否／既存ログ拒否。
# 実行: bash scripts/bench/framework-compare/ab_round_cooldown_test.sh
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
T=$(mktemp -d) || exit 1
trap 'rm -rf "$T"' EXIT
fail=0
ok() { echo "ok: $1"; }
ng() { echo "FAIL: $1"; fail=1; }

# 子 bash で hook を source して本文を実行する（cwd は一時ディレクトリ。副作用の有無を検査するため）
run_sub() {
  ( cd "$T" && bash -c ". '$HERE/ab_round_cooldown.sh'; $1" )
}

# 1. 未設定・空文字: no-op・ファイル非作成
( unset AB_ROUND_COOLDOWN_MAX_LOAD1; run_sub "ab_round_cooldown_init '$T/off.log'; ab_round_cooldown 1" ) \
  && [[ ! -e "$T/off.log" ]] && ok "未設定は no-op" || ng "未設定は no-op"
AB_ROUND_COOLDOWN_MAX_LOAD1= run_sub "ab_round_cooldown_init '$T/off2.log'; ab_round_cooldown 1" \
  && [[ ! -e "$T/off2.log" ]] && ok "空文字は no-op" || ng "空文字は no-op"

# 2. pass（3 回目の読取で下がる = 60 秒待ち）
AB_ROUND_COOLDOWN_MAX_LOAD1=3.0 run_sub "
  echo 0 >'$T/n'; ab_cooldown_read_load1() { n=\$((\$(cat '$T/n')+1)); echo \$n >'$T/n'; if [ \$n -lt 3 ]; then printf 5.0; else printf 2.5; fi; }
  ab_cooldown_sleep() { :; }
  ab_round_cooldown_init '$T/pass.log'; ab_round_cooldown 1"
grep -q '^round1 cooldown=pass threshold=3.0 load1=2.5 waited_s=60 ' "$T/pass.log" && ok "pass" || { ng "pass"; cat "$T/pass.log"; }

# 3. 閾値ちょうどは pass ではない（< 判定）。上がり続ければ 1800 秒で timeout
AB_ROUND_COOLDOWN_MAX_LOAD1=3.0 run_sub "
  ab_cooldown_read_load1() { printf 3.0; }; ab_cooldown_sleep() { :; }
  ab_round_cooldown_init '$T/eq.log'; ab_round_cooldown 1"
grep -q 'cooldown=timeout .*load1=3.0 waited_s=1800 ' "$T/eq.log" && ok "閾値ちょうどは timeout" || { ng "閾値ちょうどは timeout"; cat "$T/eq.log"; }

# 4. timeout 境界: 1800 秒目（61 回目）の読取で下がれば pass
AB_ROUND_COOLDOWN_MAX_LOAD1=3.0 run_sub "
  echo 0 >'$T/n'; ab_cooldown_read_load1() { n=\$((\$(cat '$T/n')+1)); echo \$n >'$T/n'; if [ \$n -lt 61 ]; then printf 9.0; else printf 1.0; fi; }
  ab_cooldown_sleep() { :; }
  ab_round_cooldown_init '$T/edge.log'; ab_round_cooldown 2"
grep -q 'round2 cooldown=pass .*waited_s=1800 ' "$T/edge.log" && ok "1800 秒境界で pass" || { ng "1800 秒境界"; cat "$T/edge.log"; }
# 4b. 62 回目で下がっても timeout（1800 秒を超えて待たない）
AB_ROUND_COOLDOWN_MAX_LOAD1=3.0 run_sub "
  echo 0 >'$T/n'; ab_cooldown_read_load1() { n=\$((\$(cat '$T/n')+1)); echo \$n >'$T/n'; if [ \$n -lt 62 ]; then printf 9.0; else printf 1.0; fi; }
  ab_cooldown_sleep() { :; }
  ab_round_cooldown_init '$T/edge2.log'; ab_round_cooldown 2"
grep -q 'round2 cooldown=timeout .*waited_s=1800 ' "$T/edge2.log" && ok "1800 秒超は timeout" || { ng "1800 秒超"; cat "$T/edge2.log"; }

# 5. unavailable（読取不能・非数値）
AB_ROUND_COOLDOWN_MAX_LOAD1=3.0 run_sub "
  ab_cooldown_read_load1() { printf ''; }; ab_cooldown_sleep() { :; }
  ab_round_cooldown_init '$T/na.log'; ab_round_cooldown 1"
grep -q 'cooldown=unavailable .*load1=NA ' "$T/na.log" && ok "unavailable" || ng "unavailable"

# 6. 閾値の拒否（非数値・0・負・上限超・注入文字列・指数表記・空白）
for bad in abc 0 0.0 -1 9.0 '3;touch injected' '1e1' ' '; do
  rm -f "$T/bad.log"
  AB_ROUND_COOLDOWN_MAX_LOAD1=$bad run_sub "ab_round_cooldown_init '$T/bad.log'" >/dev/null 2>&1
  rc=$?
  [[ $rc -ne 0 && ! -e "$T/bad.log" ]] && ok "拒否: '$bad'" || ng "拒否: '$bad' rc=$rc"
done
[[ ! -e "$T/injected" ]] && ok "注入なし" || ng "注入された"
# 上限ちょうど 8.0 は受理
AB_ROUND_COOLDOWN_MAX_LOAD1=8.0 run_sub "ab_round_cooldown_init '$T/up.log'" >/dev/null 2>&1 \
  && [[ -e "$T/up.log" ]] && ok "上限 8.0 は受理" || ng "上限 8.0 は受理"

# 7. 既存ログの拒否
echo keep >"$T/exist.log"
AB_ROUND_COOLDOWN_MAX_LOAD1=3.0 run_sub "ab_round_cooldown_init '$T/exist.log'" >/dev/null 2>&1
rc=$?
[[ $rc -ne 0 && "$(cat "$T/exist.log")" == keep ]] && ok "既存ログ拒否" || ng "既存ログ拒否"

if [[ $fail -eq 0 ]]; then echo "ALL OK"; else echo "FAILED"; exit 1; fi
