# Metal GEMM N=1024/2048/4096 reuse candle 比再計測と #1037 ゲート判定の確定（イシュー #1147）

## 状態: Apple M4 Max 実機実測完了。#1037（reuse candle 超え）は正式系列・参考系列（#1167/#1168 反映後 HEAD）のいずれも未達成と判定した

## 1. 位置づけ

親 #1037「N=1024/2048/4096 reuse で candle 超え（各 5 回計測の中央値）」の受け入れ判定を、
Metal GEMM 転置ロード拡張（#1138。NN 経路はビット同一のまま維持）・N=4096 カーネル純境界の
candle 比ギャップ調査（#1143。新候補 `(32,64,16,1,2)` は不採用・選択ロジック不変）を踏まえた
最新既定経路で再計測し確定する。CUDA 側の同型判定は #1142（`docs/perf/
cuda-gemm-candle-gate-remeasurement.md`）で、`run_gemm_gate_cuda.sh`／`compare_gemm_gate.py`
を本 Issue で device 汎用化（`run_gemm_gate.sh <device> <label>` + device 別薄い wrapper
`run_gemm_gate_cuda.sh`／`run_gemm_gate_metal.sh` + `compare_gemm_gate.py --device`）した上で
Metal へ同一プロトコルを適用した。本ドキュメントはその一次記録（プロトコル・実測値・#1037
突合・判定・ユーザー判断事項）。tolerance・baseline・依存ピンは一切変更しない
（`.claude/rules/coding-rust.md`「テスト・ベンチ」節。本 PR は docs(perf) 区分）。

## 2. 計測環境・プロトコル

- 実機: Apple M4 Max（64GB・macOS 26.6.2。詳細は
  `docs/perf/logs/metal-gemm-candle-gate-1147/env_info.txt`）
- worktree HEAD（origin/main 由来）: `bb7e35a`（#1167〈転置ロード拡張〉・#1168〈N=4096
  カーネルギャップ調査。選択ロジック不変〉のマージ後）
- 集計ツール: `scripts/bench/framework-compare/run_gemm_gate.sh <device> <label>`（本 Issue で
  #1142 の CUDA 専用実装〈`run_gemm_gate_cuda.sh`〉を device 汎用化。呼び出し面は device 別
  薄い wrapper `run_gemm_gate_cuda.sh`／`run_gemm_gate_metal.sh`〈新規〉）／
  `compare_gemm_gate.py --device metal`（`--device` 追加。既定 `cuda` で #1142 と後方互換）。
  `README.md`「GEMM ゲート 5 回計測」節参照
- N=1024/2048/4096 それぞれ fandhe-ai（`gemm metal <N> reuse`）・candle（`gemm metal <N>
  fresh`。reuse 非対応）を run 内で交互に 5 回起動し、run 間中央値で判定（coding-rust.md
  「ベンチは 5 回計測の中央値」）
- **2 系列を独立に計測・記録する**（#1142 と同じ理由。承認済みピンで再計測しても #1138/#1143
  反映前とほぼ同値になり「最新既定経路」の値を得られないため。詳細は §3）:
  - **正式系列**（`0.6.0`。#1037 の正式判定に用いる）: `bench-fandhe/Cargo.toml` の承認済みピン
    `fandhe-ai =0.6.0`（crates.io 公開版。2026-09-02 公開）のまま計測。コミット済み
    manifest・`Cargo.lock` は変更していない
  - **参考系列**（`head-bb7e35a`。次回 crates.io 公開後の正式再計測で確定すべき見込み値）:
    ノード側のみで `cargo build --release -p bench-fandhe --config
    'patch.crates-io.fandhe-ai.path="<facade 絶対パス>"'` により `crates/facade`（worktree
    HEAD）へ path 差し替えてビルド。`[patch]` セクション・`.cargo/config.toml` は一切
    コミットしていない（CLI 引数のみ）
- 熱・電源状態確認: 各系列の計測前後で `pmset -g therm`（thermal/performance warning なし）・
  `uptime`（負荷平均。複数エージェントが並列稼働する共有マシンのため計測専有ではない旨を
  明記）を記録（`docs/perf/metal-bench-noise-protocol.md`「熱・電源状態の記録」節準拠。
  `sudo` 必須の `powermetrics` は使わない）
- 生データ:
  `scripts/bench/framework-compare/results/raw/results-m4max-gemm-gate-0.6.0.jsonl`・
  `results-m4max-gemm-gate-head-bb7e35a.jsonl`（各 30 行）、失敗記録は両系列とも空
  （`skipped-m4max-gemm-gate-*.log`）

## 3. なぜ 2 系列が必要か

- `fandhe-ai =0.6.0` の crates.io 公開（2026-09-02）は #1167（PR #1167。転置ロード拡張。
  NN 経路はビット同一のまま維持・自動ルーティング未結線）・#1168（PR #1168。N=4096 カーネル
  純境界のギャップ調査。新候補不採用・選択ロジック不変）より**前**であり、正式系列（承認済み
  ピンのまま）は #1167/#1168 の変更を反映しない。ただし §0 で確認したとおり、#1167 は NN
  経路自体をビット同一に保ち自動ルーティングを結線していないため、framework-compare の
  `gemm metal` タスク（NN 正方 GEMM）が通る本番経路は HEAD でも v0.6.0 と実質同一である
  （選択ロジックへの影響は無し。#1168 も同様に選択ロジック不変）
- そのため、参考系列（HEAD へ path 差し替え）の値は正式系列とほぼ同水準になると見込まれた。
  実測結果（§4）はこの見込みと整合する
- 参考系列は**正式なゲート判定には用いない**（§6）。#1142 と同じ運用（次回ピン更新後の正式
  再計測で確定する見込み値としての位置づけ）を踏襲する

## 4. 実測結果

### 4.1 正式系列（`fandhe-ai =0.6.0`）

| N | fandhe-ai reuse 中央値（min–max, n=5） | candle fresh 中央値（n=5） | candle/fandhe | GFLOP/s（fandhe） | 判定 |
|---|---|---|---|---|---|
| 1024 | 2.854 ms（2.673–2.966 ms） | 2.071 ms | 0.726 | 752.4 | 未達 |
| 2048 | 10.295 ms（9.225–12.090 ms） | 6.151 ms | 0.598 | 1668.8 | 未達 |
| 4096 | 38.941 ms（38.576–43.842 ms） | 22.948 ms | 0.589 | 3529.4 | 未達 |

`results/summary.md` 環境 11（v0.6.0 単発計測。計測日 2026-09-02）の同一 (task,device,size,
mode) との比較: N=1024 0.70 倍 → 0.726 倍・N=2048 0.65 倍 → 0.598 倍・N=4096 0.66 倍 → 0.589
倍。単発計測との差は主に run 間ばらつき（min–max 幅。特に N=2048 は 9.225–12.090 ms と
ばらつきが大きい）によるもので、**5 回計測に拡張しても環境 11 の単発計測から「未達」という
結論自体に変化はない**（承認済みピンが #1167/#1168 を含まないため。§3）。

### 4.2 参考系列（`head-bb7e35a`。#1167/#1168 反映後）

| N | fandhe-ai reuse 中央値（min–max, n=5） | candle fresh 中央値（n=5） | candle/fandhe | GFLOP/s（fandhe） | 判定 |
|---|---|---|---|---|---|
| 1024 | 2.915 ms（2.366–3.058 ms） | 2.115 ms | 0.726 | 736.7 | 未達 |
| 2048 | 9.424 ms（9.070–9.946 ms） | 6.265 ms | 0.665 | 1823.0 | 未達 |
| 4096 | 38.763 ms（38.673–39.459 ms） | 22.698 ms | 0.586 | 3545.6 | 未達 |

正式系列比: N=1024 は 0.726→0.726 倍（横ばい）、N=2048 は 0.598→0.665 倍（改善したが
未達のまま）、N=4096 は 0.589→0.586 倍（横ばい）。§3 で見込んだとおり、#1167/#1168 は
`gemm metal` の NN 正方 GEMM 本番経路を変更していないため、正式系列との差は run 間ばらつきの
範囲内にとどまり、系統的な性能改善は確認されなかった。

### 4.3 CUDA 側（#1142）との対比

CUDA 側（`docs/perf/cuda-gemm-candle-gate-remeasurement.md`）は #1137（cp.async 多段パイプ
ライン結線）反映後の参考系列で N=4096 が 0.824→0.898 倍まで改善し（カーネル単体では 1.5 倍
高速化したが reuse 計測境界の固定費に希釈された）、Metal 側より 1.0 倍に近い水準だった。
Metal 側は本計測時点で該当するカーネル最適化の本番結線（自動ルーティング）がまだ行われて
おらず（#1138 は NN 経路をビット同一に維持したまま。`docs/perf/
metal-gemm-transpose-tiled.md`「5. 性能実測（ベンチマーク A/B）と結線判断」節）、改善余地は
カーネル側にまだ残っている。

## 5. データ有効性

- 両系列とも全 30 run で `parity_fail_count=0`・`parity_total` が期待要素数（N=1024:
  1,048,576／N=2048: 4,194,304／N=4096: 16,777,216）と一致し、fandhe-ai/candle 間の checksum
  が同一 N で一致（`-1855.597736`／`-6016.774008`／`-25768.747284`）することを確認した。
  CUDA 側（#1142）で見られた N=2048 candle 無効データ（`parity_fail_count=2`）は Metal では
  再現しなかった（バックエンド固有の丸め誤差挙動の差。原因分析は本 PR の対象外）
- `compare_gemm_gate.py --device metal` の集計は両系列とも「判定不能」を出さず全 size で
  確定判定（未達）を返した（`exit code 3`＝未達あり・判定不能なし）

## 6. #1037 受け入れ条件との突合

| # | #1037 の受け入れ条件 | 正式系列（0.6.0） | 参考系列（head-bb7e35a） | 出典 |
|---|---|---|---|---|
| 1 | N=1024 reuse で candle 超え | 未達（0.726 倍） | 未達（0.726 倍） | §4.1・§4.2 |
| 2 | N=2048 reuse で candle 超え | 未達（0.598 倍） | 未達（0.665 倍。改善したが未達） | §4.1・§4.2 |
| 3 | N=4096 reuse で candle 超え | 未達（0.589 倍） | 未達（0.586 倍） | §4.1・§4.2 |
| 4 | parity 0 fail（fandhe-ai 側） | 達成（全 5 run `parity_fail_count=0`） | 達成（同上） | §4.1・§4.2・§5 |

**総合判定: #1037 は正式系列・参考系列のいずれにおいても未達成（未達 3 件）。**
`crate::precision`（TF32 等の精度緩和経路）は Metal には存在せず本計測の対象外。

## 7. `results/summary.md`・`performance-targets.md` への反映

- `results/summary.md` 環境 13 節・「目標達成ゲート総括」への追補は本 PR に含む
  （`scripts/bench/framework-compare/results/summary.md` 参照）
- `docs/performance-targets.md` §8.3「#1147 追補」（§2 段階的下限表・§3 丸め規則は不変）
- `docs/perf/gemm-optimization-baseline.md` §6 に本ドキュメントへの参照 1 行を追記済み
- `docs/perf/metal-gemm-bottleneck-rediagnosis.md` §8・`docs/perf/metal-gemm-n4096-kernel-gap.md`
  末尾に本ドキュメントへの参照 1 行を追記済み

## 8. スコープ外事項（本 PR では対応しない）

- **reuse 計測境界の転送・同期固定費削減**: `metal-gemm-bottleneck-rediagnosis.md` が既に
  指摘している fandhe-ai 自系列内の転送（アップロード＋readback）寄与。カーネル最適化のみでは
  解消できない構造要因の可能性がある。対処には CUDA 側 #1142 §8 と同型の `Tensor<f32>`
  デバイス常駐化等、別スコープの設計変更が必要（後続 issue 化の要否は §9 ユーザー判断）
  **実測確定（#1189）**: `docs/perf/metal-gemm-reuse-phase-breakdown.md` が reuse 1 反復を
  upload／encode／commit_wait／readback／host_copy へ分解実測した結果、CUDA（#1182）とは
  非対称な結論となった——N=1024/2048 は `matmul` 単体（転送＋カーネル＋同期）が candle
  fresh とほぼ同等（0.90〜0.99 倍）まで縮まるが、**N=4096 は `matmul` 単体が candle fresh
  より 1.52 倍遅く**、reuse 計測境界の再定義（ハーネス `host_copy`／`checksum` の除去）
  では N=4096 の未達は解消しない。Metal のギャップは GPU 実行自体（統合メモリ転送・
  カーネル・同期の合算）に起因する構造的なものであり、上記デバイス常駐化のような別スコープ
  の設計変更が必要という結論を裏付ける（同ドキュメント §7・§9）
- **#1138 自動ルーティング（`dispatch_strided_bias_act_prepared` 委譲）の性能 A/B**: NN 経路
  への転置ロード拡張の結線・計測は #1138 のスコープ外として持ち越されており、本 PR でも
  実施しない
- **`metal-gemm-n4096-kernel-gap.md` の残調査（レジスタ圧・カーネル生成側）**: 同ドキュメント
  §5「スコープ外（今後の切り出し候補）」で持ち越し済みの追跡事項であり本 PR の対象外
- **crates.io v0.7.0 公開・framework-compare ピン `=0.7.0` 更新**: 正式系列で #1138/#1143 系の
  改善（結線された場合）を反映した判定を得るために必要（ユーザー承認事項。deps-policy.md
  第 9 区分）。#1138 は本計測時点で自動ルーティング未結線のため、v0.7.0 が出ても本計測の
  結論（NN 経路は不変）は変わらない可能性が高い
- **CUDA 側 `run_gemm_gate.sh` cuda 経路の実機再検証**: 本 Issue の device 汎用化は「純粋な
  移設」（cuda 分岐のロジック変更なし。§検証方法参照）であり、DGX Spark 実機未接続のため
  cuda 経路の実行再検証はしていない
- `docs/perf/performance-floor-decision.md`（REQ-8 の PyTorch 比下限）は変更しない
  （candle 比とは別軸のため）

## 9. ユーザー判断事項

- **#1037 のクローズ可否**: 本計測により正式系列・参考系列いずれでも未達成が確定した。
  クローズせず残課題として維持するか、達成条件・スコープの見直し（例: reuse 計測境界の
  再定義、転送・同期を除いたカーネル専有時間での判定への変更）を検討するかはユーザー判断
- **後続 issue 化の要否**: reuse 転送・同期固定費削減（§8）・#1138 自動ルーティングの性能
  A/B（§8）・`metal-gemm-n4096-kernel-gap.md` の残調査（§8）を追跡する新規 issue を起票する
  かはユーザー判断（`out-of-scope-tracking.md` に従い、本 PR では Issue 操作を行わない）
- **crates.io 次回公開のタイミング**: v0.7.0 想定の正式ピン更新の要否・時期（#1138 自動
  ルーティング結線後に意味を持つ。§8）

## 10. 関連ドキュメント

- `docs/perf/cuda-gemm-candle-gate-remeasurement.md`（CUDA 側の同型判定。#1142）
- `docs/perf/metal-gemm-transpose-tiled.md`（#1138 の転置ロード拡張・自動ルーティング未結線の
  明記）
- `docs/perf/metal-gemm-n4096-kernel-gap.md`（#1143 の N=4096 カーネル純境界ギャップ調査）
- `docs/perf/metal-gemm-bottleneck-rediagnosis.md`（reuse でも残る転送・同期寄与の設計根拠）
- `docs/perf/metal-bench-noise-protocol.md`（熱・電源状態記録プロトコル）
- `scripts/bench/framework-compare/README.md`「GEMM ゲート 5 回計測」節
- `scripts/bench/framework-compare/results/summary.md` 環境 11/13 節
- `docs/performance-targets.md` §8/§8.1/§8.3
- `docs/perf/logs/metal-gemm-candle-gate-1147/`（実行ログ・env_info）
