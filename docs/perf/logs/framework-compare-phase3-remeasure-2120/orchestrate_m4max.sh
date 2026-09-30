#!/usr/bin/env bash
# イシュー #2120: M4 Max 用ラッパー。本体は orchestrate.sh（HEAD_TREE／PRE_TREE／LOGD を環境変数で渡す。SMOKE=1 で疎通確認）。
exec bash "$(dirname "$0")/orchestrate.sh" m4max
