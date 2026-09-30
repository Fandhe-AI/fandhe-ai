# x86 パイプライン疎通 smoke（系列ではない。イシュー #2117）

`scripts/bench/framework-compare/run_ab_gb10_affinity_cpu.sh` の end-to-end 疎通確認（x86_64 ホスト。GB10 ではない）。
**判定の根拠にはならない。** 本 smoke から ADOPT／REJECT／undetermined のいずれも導かない。

- 実行条件: `AB_AFFINITY_PRECHECK=report-only`（機構発火を assert しない）・`AB_LOAD_GATE=64`（他プロセスの高負荷下で実行したため。
  `series=reference`・`gate_ok=0`）。before = 作業ブランチ、after = 同一コミットに `on-arm.patch` を当てた一時 worktree。
- 確認できたこと: 両腕のビルド・path patch 解決（`tree-*`）・差分検証・機構発火確認の出力（`affinity-report-*`。x86 では
  `detected_big_core_ids=None`・`pool_active=false` のため機構は発火せず、after も実質 before と同一動作）・
  N=256 を含む 14 セル × 5 round の JSONL 出力（+1 行検証）・checksum 完全一致・`compare_gemm_ab.py --sizes affinity` の比較表。
- `compare-*.md` の ratio・「後退」表示は x86 上のノイズ（load1 が 11〜20）であり意味を持たない（両腕は同一動作のため本来 1.00 近傍）。
- 負の検証（別途実施済み）: `RAYON_NUM_THREADS` 設定時・after が false のまま・after に定数以外の差分・
  `AB_AFFINITY_PRECHECK` の不正値・既定の assert モード（x86 では大コア未検出で precheck 失敗）は、いずれもベンチ前に停止する。
- ホスト名・`$HOME`・絶対パスはマスク済み。生 JSONL・バイナリは収録しない。
