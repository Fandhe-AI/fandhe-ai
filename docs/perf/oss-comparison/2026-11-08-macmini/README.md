# エンジン単体の対照の Mac mini 測り直し（v0.14.0・2026-11-08）

スコアボード 0.14.0（`docs/perf/logs/framework-compare-0.14.0-remeasure/scoreboard/`）の「エンジン単体の対照」節に
残っていた M4 Max（開発機）の 2 カードを、Mac mini（Apple M4・10 コア・16 GB）で測り直した記録。
集計表は `docs/perf/oss-gemm-comparison-baseline.md`「Mac mini（Apple M4）での測り直し」節。

- 規則: `RULE.txt`（2026-11-08T10:02Z 固定・コミット 830641d0a で計測前に push）
- 対象: タグ v0.14.0 の git archive（`.rev-stamp`・ツリー指紋 `b80eadc9…` は `env/tree-fingerprint-*.txt` の 3 本で一致）
- 実行: `scripts/orchestrate.sh`（prepare → series）。ゲートは 5/5 通過（`out/gate.log`・`out/round*/gate.txt`）、
  round1 の前だけビルド直後の負荷で 60 秒待機（load1 4.97 → 2.59）
- 集計: `scripts/aggregate.py out 5` → `aggregate.out`（`--self-test` で過去 M4 Max データの baseline 値を再現）
- oss-gemm-compare は全ラウンド exit=1（出力不一致による。規則どおり記録のみ）。他のハーネスは全ラウンド exit=0
- `env/build-*.log` のホームディレクトリのパスは `~` に置き換えた
- 測らなかったもの: candle 比ゲート（正式系列の Mac mini 版は別 issue と事前登録が要る）
