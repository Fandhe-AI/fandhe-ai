# SME `fmopa` マイクロカーネル A/B（イシュー #1587／#1978）実測ログ置き場

正式記録は `docs/perf/cpu-gemm-sme-fmopa-microkernel.md`（§5.4 が
本ディレクトリの正式実測を転記した節）を参照。

## 状態（2026-09-18 更新）

- **R3**（数値契約: run-to-run bit 同一・SME vs scalar 参照の有限値／
  非正規化数 bit 一致・NaN 混入で panic なし・本番入口 bit 一致）は
  Apple M4 Max 実機で完全実施・全 PASS 済み（`docs/perf/
  cpu-gemm-sme-fmopa-microkernel.md` §4）。`cargo test` の標準出力
  そのもの（決定的・非タイミング系のため生ログの保存は不要と判断し
  本ディレクトリには含めない）。
- **R1（framework-compare 非後退）・R2（checksum）・R4（しきい値の
  正式 5-run 独立プロセス起動スイープ）は Apple M4 Max で実施済み**
  （イシュー #1978。総合判定 undetermined〈初版は ADOPT と記録したが、PR #2016 レビュー指摘により train size=64 を到達セルへ再分類し、train reuse の round 1・5 が 1.0 超のため ADOPT 候補条件不成立〉。詳細・数値は
  `docs/perf/cpu-gemm-sme-fmopa-microkernel.md` §5.4）。DGX Spark
  GB10 側は RULE.txt が明記するとおり本セッションの対象外（SME 非
  対応）。`SME_PRODUCTION_ENABLED` の本番切替・しきい値確定は
  #1979 のユーザー承認事項として未実施のまま。
- **GB10（DGX Spark GB10・Grace CPU）側の「既存経路の非後退確認」は
  2026-09-18（UTC）に `gb10/` で実施済み**（イシュー #1978 残。事前登録
  `gb10/RULE-gb10.txt`）。R0（`sme_report()` が両腕とも
  `kernel_enabled: false`）成立・R2-GB10（10 セル checksum 完全一致）
  成立・RT は after 腕で定数ドリフトガード
  `sme_production_enabled_is_false_pending_measurement` の 1 件のみ FAIL
  （637 pass）・R1-GB10 は gemm cpu 1024/reuse が 5/5 round 一貫の後退
  （1.0195〜1.0554・中央値 1.0363）。事前登録規則により総合判定は
  **「GB10 非後退確認: 後退あり」**（是正・緩和なし。原因帰属は未検証。
  `docs/perf/cpu-gemm-sme-fmopa-microkernel.md` §5.5）。
- **GB10 側の「非 SME 環境で bit 同一のフォールバック」の全出力 bit 同一
  実証は 2026-09-19（UTC）に `gb10/bitdump/` で実施済み**（イシュー #2050。
  事前登録 `gb10/RULE-bitdump.txt`）。R0 成立・RB は before／after とも
  6,726,847 行・全体 sha256 一致・`cmp`／`diff` 差分 0・5 ラベル
  （gemm512／1024／2048・train・infer）すべて一致・RR（after 2 回目）も
  一致。総合判定 **「bit 同一: 成立」**（同 doc §5.6。
  `SME_PRODUCTION_ENABLED=false` は不変）。

## ディレクトリ構成

- `RULE.txt` — 事前登録判定規則（固定日時 2026-09-17T16:32:14Z。一次
  ソースは issue #1587 の事前登録コメント）
- `orchestrate_m4max.sh` — R4 用 5 プロセス独立起動スクリプト
  （`--dry-run` 対応）
- `aggregate.py` — R4 集計スクリプト（python3 標準ライブラリのみ・
  `--self-test` 付き。計測後に `cargo test` 出力の 1 行目パース条件
  のみ是正済み。判定規則は不変）
- `aggregate_r4.md` — R4（16 格子点 × 5 run）の集計結果
- `load_gate_r4.log` — R4 の負荷ゲート記録（5/5 通過）
- `sme_r4_grid_run{1..5}.log` — R4 各 run の抽出ログ（`variant=`／`test `／`SME ` 行）。
  PR #2016 の是正後の `orchestrate_m4max.sh` は未加工出力を
  `sme_r4_grid_run{1..5}.raw.log` へ併せて保存し、計測プロセスの非ゼロ終了・
  抽出行 0 件を非ゼロ終了で伝播する（本ディレクトリの実測は是正前の
  スクリプトで取得したため `.raw.log` は存在しない）
- `on-arm.patch` — after 腕の差分（main `a1c50f61` に対し
  `SME_PRODUCTION_ENABLED` のみ `false` → `true` へ反転。計測専用
  worktree の変更で main へはコミットしない）
- `env_info.txt` — 実行環境・時刻の記録（内部ホスト名は含めない）
- `gb10/` — GB10 側の非後退確認（イシュー #1978 残・2026-09-18 UTC）。
  `RULE-gb10.txt`（事前登録）・`orchestrate_gb10.sh`（ノード上で実行。
  after ツリーの複製と `on-arm.patch` 適用・指紋採取・外側専有ゲート・
  R0 プローブ・RT・R1/R2 を一括実行）・`sme-probe-{Cargo.toml,main.rs}`
  （R0 用の使い捨てプローブ crate。ツリー外に配置し `fandhe-ai-backend-cpu`
  を path 依存）・`sme_report.txt`・`cargo_test_after.summary.log`・
  `load_gate_outer.log`・`gate_constant.txt`・`patch_sha256.txt`・
  `fp-before.txt`／`fp-diff.txt`・`rev-stamp-verification.md`・`env_info.txt`・
  `run_ab.log`／`orchestrate.log`・`r1r2/`（Mac 側と同じ構成。LABEL は
  `1978-gb10`）。絶対パスは `<home>` へマスク済み・内部ホスト名は含めない
  `run_bitdump.sh`（イシュー #2049 新設。`SME_PRODUCTION_ENABLED`
  on/off の before/after 2 ツリー間で `cpu_sme_gate_bit_dump.rs::
  dump_cpu_sme_gate_bits`〈gemm512/1024/2048・train size=64（reuse・
  L1 d_weight のみ到達）・infer size=64（非到達対照）〉の出力を
  `diff`／`sha256sum` で bit 完全一致確認する診断スクリプト。
  `--dry-run` 対応。実測は #2050 で実施）・
  `RULE-bitdump.txt`（イシュー #2050 の事前登録規則。R0／RB／RR と
  `run_bitdump.sh` の exit code 対応を実測前に固定）・
  `orchestrate_bitdump_gb10.sh`（ノード上で実行。after ツリー複製と
  `on-arm.patch` 適用・指紋差分が `mod.rs` 1 件であることの assert・
  外側専有ゲート〈記録のみ〉・R0 プローブ・`run_bitdump.sh`・after 再実行）・
  `bitdump/`（#2050 の実測成果物。`orchestrate.log`・`run_bitdump.log`・
  `sme_report.txt`・`sme_probe_{before,after}.log`・`gate_constant.txt`・
  `patch_sha256.txt`・`patch_apply.log`・`fp-{before,after}.txt`／
  `fp-diff.txt`・`load_gate_outer.log`・`line_counts.txt`・`summary.txt`
  〈ラベル別行数・sha256〉・`bitdump_cmp.txt`／`bitdump_diff.txt`〈とも空〉・
  `rerun_after_sha256.txt`・`{before,after,after_rerun}_raw.summary.log`
  〈`^out\[` 行を除いた cargo 出力〉・`uptime_{before,after}.txt`・
  `env_info.txt`。`*_bits.txt`／`*_raw.log` の dump 本体〈各約 269 MB〉は
  コミット対象外——`.gitignore` は設けず本ファイルと `run_bitdump.sh`
  冒頭コメントの明記のみで管理する。絶対パスは `<home>` へマスク済み・
  内部ホスト名は含めない）
- `r1r2/` — R1（framework-compare gemm／train／infer cpu）・R2
  （checksum）の実測一式
  - `compare-{gemm,train,infer}-1978-cpu.md` — before/after 比較表（是正前の
    `run_ab_sme_cpu.sh` が生成した名前。PR #2016 是正後は LABEL を含む
    `compare-{task}-1978-cpu-<LABEL>.md`／`.err` へ出力する）
    （`scripts/bench/framework-compare/compare_gemm_ab.py
    --require-checksum-exact` の出力）
  - `results-{before,after}-1978-cpu-{gemm,train,infer}.jsonl` —
    生の計測結果（5 round）
  - `load-gate-1978-cpu-1978.log` — R1／R2 の負荷ゲート記録（5/5 通過）
  - `uptime-1978-cpu-1978.log`・`compare-exit-1978-cpu-1978.log`・
    `skipped-1978-cpu-1978.log`（空＝スキップなし）
  - `sha-1978-{before,after}-1978.txt` — before／after 各バイナリの
    sha256（ビルド成果物の再現性確認用。ホスト情報は含まない）
  - `tree-1978-{before,after}-1978.txt` — before／after 各ツリーの
    指紋記録

## 事前登録判定規則（issue #1587／#1978 コメントの転記）

正式版・一次ソースは GitHub issue #1587 の実装着手前コメント
（`https://github.com/Fandhe-AI/fandhe-ai/issues/1587#issuecomment-5648848287`）
および本ディレクトリの `RULE.txt`（イシュー #1978 用に固定した運用
規則）を正とする。本 README は要旨のみを転記し、規則自体はそちら側を
正とする（事後の緩和はコメント側を編集せず新規コメントで記録する規約）。
