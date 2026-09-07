# Metal GEMM N=1024/2048/4096 reuse candle 比再計測と #1037 ゲート判定の確定（イシュー #1147）

## 状態: Apple M4 Max 実機実測完了。#1037（reuse candle 超え）は正式系列・参考系列（#1167/#1168 反映後 HEAD）のいずれも未達成と判定した。#1185 で正式系列 `fandhe-ai =0.7.0` を 2026-09-06 に再計測し未達成を確定（§11）。#1337 で借用ビュー readout（既定 OFF feature）切替前後を 2026-09-07 に再計測（§12）。共有負荷下・全 3 形状で後退したが片方向の負荷差と切り分けられておらず、正式判定（§11）は不変

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
  （2026-09-06 更新: v0.7.0 公開・ピン更新〈PR #1233〉により解消済み。§11）
- **2026-09-06 更新（イシュー #1185）**: 正式系列 `fandhe-ai =0.7.0` でも未達成が確定した
  （§11）ことを受け、ユーザー指示（2026-09-06）「未達の場合は後継ツリーを新規起票し現 issue は
  クローズ」に従い、上記の残課題（reuse 転送・同期固定費削減・#1138 自動ルーティングの性能
  A/B・N=4096 カーネル純境界ギャップ・達成条件の見直し）は後継ツリー
  #1269（E2〜E5 級カーネル変更トラッキング）・#1242（転置タイル variant A/B 判定不能の解消トラッキング） へ引き継ぎ、現 issue #1037 はクローズする

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
- `docs/perf/logs/gemm-candle-gate-0.7.0-1185/`（イシュー #1185。=0.7.0 正式系列再計測の
  実行ログ・env_info。CUDA / Metal / CPU 共通）

## 11. 2026-09-06 追補: 正式系列 `fandhe-ai =0.7.0` 再計測（イシュー #1185）

### 11.1 位置づけ・プロトコル

- v0.7.0 の crates.io 公開と framework-compare の承認ピン `fandhe-ai =0.7.0` 更新（PR #1233）
  を受け、**正式系列のみ**で N=1024/2048/4096 reuse の 5 回計測中央値を Apple M4 Max で
  再取得した（CUDA 側 #1185 と同一プロトコルの Metal 版。参考系列は計測していない）
- プロトコルは §2 と同一（`run_gemm_gate_metal.sh 0.7.0`・`compare_gemm_gate.py --device metal`。
  manifest で `fandhe_ai_source=registry`・`candle_core_source=registry` を確認済み）
- 計測環境: M4 Max・macOS 26.6.2・rustc 1.96.0。`pmset -g therm` は計測前後とも thermal /
  performance warning なし
- **負荷状態の注意書き**: 計測中、同一マシンで他セッションの cargo ビルドが並走する**共有マシン
  状態**だった（`uptime` load average: 計測前 3.39 → Metal ゲート開始時 6.40 → 完了時 7.09 →
  後続 CPU ゲート完了時 7.96。`docs/perf/logs/gemm-candle-gate-0.7.0-1185/env_info.txt`）。
  fandhe-ai と candle は同一負荷下で run 内交互起動しているため候補比の方向性（未達）の結論には
  影響しないと判断するが、絶対値・0.6.0 系列との差分には背景負荷のノイズが乗る（§11.3）
- 生データ: `scripts/bench/framework-compare/results/raw/results-m4max-gemm-gate-0.7.0.jsonl`
  （30 行）・`skipped-m4max-gemm-gate-0.7.0.log`（空）・`manifest-m4max-gemm-gate-0.7.0.json`。
  実行ログ: `docs/perf/logs/gemm-candle-gate-0.7.0-1185/run_gemm_gate_metal-m4max-0.7.0.log`

### 11.2 実測結果（正式系列 `0.7.0`）

| N | fandhe-ai reuse 中央値（min–max, n=5） | candle fresh 中央値（n=5） | candle/fandhe | GFLOP/s（fandhe） | 判定 |
|---|---|---|---|---|---|
| 1024 | 2.536 ms（2.136–2.974 ms） | 2.120 ms | 0.836 | 846.7 | 未達 |
| 2048 | 10.506 ms（10.113–11.271 ms） | 6.698 ms | 0.638 | 1635.3 | 未達 |
| 4096 | 45.374 ms（41.004–47.129 ms） | 23.100 ms | 0.509 | 3029.0 | 未達 |

fandhe-ai・candle とも全 30 run で `parity_fail_count=0`・checksum が同一 N で一致（データ
有効性は §5 と同じ。tolerance は緩めていない）。

### 11.3 0.6.0 正式系列（§4.1）との差

| N | 0.6.0 正式系列（#1147） | 0.7.0 正式系列（§11.2） | 差の方向 |
|---|---|---|---|
| 1024 | 0.726（fandhe 2.854 ms / candle 2.071 ms） | 0.836（fandhe 2.536 ms / candle 2.120 ms） | 改善 |
| 2048 | 0.598（fandhe 10.295 ms / candle 6.151 ms） | 0.638（fandhe 10.506 ms / candle 6.698 ms） | 改善（candle 側の悪化が主因） |
| 4096 | 0.589（fandhe 38.941 ms / candle 22.948 ms） | 0.509（fandhe 45.374 ms / candle 23.100 ms） | 低下 |

- N=1024/2048 は改善、N=4096 は低下したが、いずれも**共有負荷下の観測**であり、N=4096 の
  fandhe-ai reuse は min–max 幅が 41.004–47.129 ms と広い（0.6.0 系列の 38.576–43.842 ms より
  上振れ）。N=2048 の比改善は candle 側 6.151 → 6.698 ms の悪化が主因で fandhe-ai 側は
  10.295 → 10.506 ms とほぼ同じ。**これらの差分をコード変更（0.6.0 → 0.7.0）に帰属させる
  根拠にはしない**（0.6.0 正式系列も非専有計測。#1147 §2・
  `docs/perf/logs/metal-gemm-candle-gate-1147/env_info.txt` の load average 3.44〜6.35）
- v0.6.0 → v0.7.0 の `crates/backend-metal/src` 変更（`git log v0.6.0..v0.7.0`: #1167/#1168・
  #1227 `linear_forward_device`・#1228 NT/TN strided の VJP 結線）はいずれも転置〈VJP〉経路・
  学習 forward 向けで、NN 正方 GEMM の reuse 経路を対象とした性能変更は含まれない。系統的な
  改善が観測されないことは §4.2 の参考系列判定とも整合する

### 11.4 #1037 ゲート判定（確定）

| # | #1037 の受け入れ条件 | 正式系列（0.7.0） | 出典 |
|---|---|---|---|
| 1 | N=1024 reuse で candle 超え | 未達（0.836 倍） | §11.2 |
| 2 | N=2048 reuse で candle 超え | 未達（0.638 倍） | §11.2 |
| 3 | N=4096 reuse で candle 超え | 未達（0.509 倍） | §11.2 |
| 4 | parity 0 fail（fandhe-ai 側） | 達成（全 15 run `parity_fail_count=0`） | §11.2 |

**総合判定: #1037 は正式系列 `fandhe-ai =0.7.0` においても未達成（未達 3 件）。#1147 の 2 系列
判定を正式系列単独で確定した。** 共有負荷下の計測ではあるが、最良形状 N=1024 でも 0.836 倍で
1.0 倍との差は負荷ノイズで説明できる範囲を超える（N=1024 の fandhe min 2.136 ms でも candle
中央値 2.120 ms に届かない）。達成条件の見直し要否・後継ツリーへの引き継ぎは §9
「2026-09-06 更新」を参照。

## 12. 2026-09-07 追補: 借用ビュー readout 切替前後の再計測（イシュー #1337）

### 12.1 位置づけ・プロトコル

- 実装本体は PR #1411（`3d5e833`。#1334 ツリー）で完了済み。`bench-fandhe` の `readout_var`／
  `checksum_var`／`checksum_tensor` を cargo feature `host-view-readout`（既定 OFF）で
  借用ビュー（`fandhe_ai::VarHostView`／`Tensor::host_slice`。#1335/#1336）readout 経路へ
  切替可能にした。本節はその効果を Metal（M4 Max）で切替前後 5 回中央値として実測した記録
- crates.io の承認ピンは §11 時点から更新されていないため（`host-view-readout` が使う API は
  `fandhe-ai =0.7.0` に未収録）、本節も §11 と同じ **参考系列**（`GEMM_GATE_PATCH_FACADE_PATH`
  による path 差し替え HEAD ビルド）のみで計測する。正式判定（registry ピン）は §11 のまま不変
- 転送元 sha: `1c298ff5641b948dae3c1c65699930054af8f747`（PR #1411 より後、PR #1420 まで含む
  HEAD）。ラベル: `head-1c298ff-readout-off`（feature 無効・legacy 経路）／
  `head-1c298ff-readout-on`（`host-view-readout` 有効・借用ビュー経路）
- **off/on 間でソースが揃っていない（重要な限定）**: manifest 実測（
  `docs/perf/logs/gemm-candle-gate-readout-1337/run_gemm_gate_metal-readout-{off,on}.log`
  の `依存元検証 OK` 行）で確認すると、off 腕は `fandhe_ai_source=registry`（
  `GEMM_GATE_PATCH_FACADE_PATH` 未使用。crates.io `fandhe-ai =0.7.0`）、on 腕のみ
  `fandhe_ai_source=path:<facade 絶対パス>`（HEAD `1c298ff...`）＋
  `GEMM_GATE_BENCH_FANDHE_FEATURES=host-view-readout` であり、上記「本節も §11 と同じ
  参考系列のみで計測する」という記述は正確ではない（off 腕のみ正式系列相当の registry
  ビルド）。off/on の差分には readout feature の効果に加え、v0.7.0 公開後にマージされた
  コード差分（HEAD と registry の乖離）が混入しており、**§12.2〜§12.3 の off/on 比較単独
  からは readout（#1337）への効果を分離帰属できない**。§12.4 の「readout-on 自体が Metal
  で純粋に遅いと断定しない」という保留判断はこの限定のもとでも変わらないが、原因を負荷
  ノイズのみに限定せず、ソース差（HEAD 対 registry）も未分離の交絡要因として扱う。
  同一 HEAD source（path 差し替え）での off 腕再計測は本イシューのスコープ内で追加実施
  していない
- プロトコルは §2 と同一（`run_gemm_gate_metal.sh`）に加え、
  `GEMM_GATE_PATCH_FACADE_PATH=<crates/facade 絶対パス>`（on 腕はさらに
  `GEMM_GATE_BENCH_FANDHE_FEATURES=host-view-readout`）を付与。効果分離には
  `compare_gemm_ab.py --device metal --sizes gate --modes reuse`
  （`jq -c 'select(.framework == "fandhe-ai")'` で抽出した `*.fandhe-only.jsonl` が入力）を
  追加使用
- **負荷状態の注意書き（重要）**: 本節の計測は §11 よりさらに高負荷な共有マシン状態で実施した
  （`uptime` load average: off 開始前 5.13/6.11/8.71 → off 完了時 10.08/7.63/9.07 →
  on 完了時 21.46/13.16/11.05。詳細 `docs/perf/logs/gemm-candle-gate-readout-1337/env_info.txt`）。
  off 腕より on 腕のほうが一貫して高負荷という**片方向のノイズ**のため、後述 §12.2 の後退が
  readout 経路そのものに起因するか負荷ノイズに起因するかを本節単独では切り分けられない
  （§12.4 のユーザー判断事項）
- 生データ: `scripts/bench/framework-compare/results/raw/{results,skipped,manifest}-m4max-
  gemm-gate-head-1c298ff-readout-{off,on}.{jsonl,log,json}`（各 30 行・`skipped-*.log` 空）。
  実行ログ・env_info: `docs/perf/logs/gemm-candle-gate-readout-1337/`

### 12.2 実測結果

| N | readout-off 中央値（min–max, n=5） | readout-on 中央値（min–max, n=5） | candle fresh 中央値（n=5） | off の candle 比 | on の candle 比 | off 判定 | on 判定 |
|---|---|---|---|---|---|---|---|
| 1024 | 2.291 ms（2.041–2.955 ms） | 3.570 ms（2.351–5.498 ms） | 2.102 / 2.337 ms | 0.918 | 0.655 | 未達 | 未達 |
| 2048 | 10.054 ms（9.233–10.570 ms） | 13.077 ms（10.846–15.814 ms） | 7.491 / 9.276 ms | 0.745 | 0.709 | 未達 | 未達 |
| 4096 | 41.511 ms（40.111–48.830 ms） | 55.336 ms（42.258–83.371 ms） | 23.620 / 35.790 ms | 0.569 | 0.647 | 未達 | 未達 |

fandhe-ai 側は全 30 run で `parity_fail_count=0`。off/on の checksum は全セル完全一致
（`compare_gemm_ab.py` 出力。§12.3）。CUDA（#1360）・DGX CPU（§15）で観測された「大形状で
改善」というパターンは Metal では再現せず、**全 3 形状で後退**した。

### 12.3 readout 切替効果（`compare_gemm_ab.py --device metal --sizes gate --modes reuse`）

```
| size/mode | before median | after median | after/before | checksum | 判定 |
|---|---|---|---|---|---|
| 1024/reuse | 2.291 ms (min 2.041 ms / max 2.955 ms) | 3.570 ms (min 2.351 ms / max 5.498 ms) | 1.5582 | 完全一致 | 後退 |
| 2048/reuse | 10.054 ms (min 9.233 ms / max 10.570 ms) | 13.077 ms (min 10.846 ms / max 15.814 ms) | 1.3007 | 完全一致 | 後退 |
| 4096/reuse | 41.511 ms (min 40.111 ms / max 48.830 ms) | 55.336 ms (min 42.258 ms / max 83.371 ms) | 1.3330 | 完全一致 | 後退 |
```

- checksum 完全一致・parity 0 fail のため数値精度への影響はなく、純粋に readout 経路のコスト
  差（＋計測時の負荷差。§12.1）の問題
- `docs/perf/metal-gemm-reuse-phase-breakdown.md` §6 は `host_copy` が Metal reuse の
  `iter_total` に占める割合を 7.8〜10.7% と見積もっており、CUDA（#1182 のフェーズ分解で
  `host_copy` がより支配的）ほど借用ビュー化の恩恵が大きくないことを事前に示唆していた。
  本節の後退はその見立てと方向としては矛盾しない（Metal では readout 削減の恩恵よりも
  負荷ノイズ・その他の固定費の影響が上回った可能性がある）が、**§12.1 の片方向負荷差のため
  「readout-on 自体が Metal で純粋に遅い」と断定はしない**

### 12.4 公正性の論点・スコープ・ユーザー判断事項（親 #1334 受け入れ条件）

- **公正性**: candle 側ハーネス（`bench-candle`）は本イシューで変更していない（`to_vec2` の
  まま）。読み出し経路は各ライブラリの公開 API の一部であり、fandhe-ai 側の feature 切替は
  ハーネスの偏向ではなく製品側実装の測定である。一方、candle にはこれに相当する借用 API が
  無い／使用していないため、iter_total 境界の比較が「読み出し方式の差」を含むことは限界として
  明記する
- **#1336 の非到達**: `Var::matmul` 出力は `gemm` 内部の readback で既にホスト常駐
  `Tensor` であるため、CUDA 向け pinned host staging（`MemoryOps::with_host_view`。#1336）は
  この readout 経路を通らない（Metal には同種の staging 実装自体がない）。本節の効果は
  `#1337`（borrowed-view readout そのもの）に帰属する
- **`host-view-readout` 既定化の可否**: 本節では判断しない（Metal で 3 形状とも後退・かつ
  負荷ノイズおよび off/on 間のソース差〈§12.1〉のいずれとも切り分けられていないため、
  既定化の根拠にはできない）
- **低負荷環境での再計測**: 本節の計測は片方向の負荷差（on 腕がより高負荷）を伴う共有マシン
  状態下で実施した。低負荷環境での再計測は本イシューのスコープ外とし、ユーザー判断で
  新規 issue を起票するかを決める
- **#1037 ゲート判定への影響**: readout-off／on いずれも §11 の正式系列判定（未達 3 件）を
  覆さない。正式判定（registry ピン）は §11 のまま不変
