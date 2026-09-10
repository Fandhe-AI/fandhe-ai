# Metal GEMM N=1024/2048/4096 reuse candle 比再計測と #1037 ゲート判定の確定（イシュー #1147）

## 状態: Apple M4 Max 実機実測完了。#1037（reuse candle 超え）は正式系列・参考系列（#1167/#1168 反映後 HEAD）のいずれも未達成と判定した。#1185 で正式系列 `fandhe-ai =0.7.0` を 2026-09-06 に再計測し未達成を確定（§11）。#1337 で借用ビュー readout（既定 OFF feature）切替前後を 2026-09-07 に再計測（§12）。共有負荷下・全 3 形状で後退したが片方向の負荷差と切り分けられておらず、正式判定（§11）は不変。#1438 で同一 facade ソース下の借用ビュー readout 既定化 before/after を M4 Max 実機実測し、全 3 形状非後退・checksum 完全一致を確認したが、before/after を連続実行しており負荷差の影響を排除できていないため暫定の参考結果とする（交互実行または負荷を揃えた再計測まで最終確定しない。正式判定は §11 のまま不変。§13）。#1309 で Phase 3（#1280・#1302・#1308・#1334・#1368 反映後）の正式系列 `fandhe-ai =0.7.0` を 2026-09-09 に再計測し §11 の未達成判定を再確認（§14。N=1024 0.700 倍・N=2048 0.969 倍・N=4096 0.710 倍。共有負荷下）。参考系列は負荷ゲート（1 分 load average < 4.0 を 2 回連続）が計測時間内に安定通過せず未計測のまま（§14.6）。#1477 で `--readout <legacy|borrowed>`（同一バイナリの run 単位 interleave override）を実装し §13.5 の暫定判定解消を試みたが、1 回目の専有ゲート試行（最大 4 試行）は成立せず undetermined のまま終了（計測未実施。§15）。#1490 で正式系列 `fandhe-ai =0.8.0` を 2026-09-10 に再計測し §16 追記: N=1024 0.743 倍（未達）・**N=2048 1.002 倍（達成。正式系列として初めて #1037 の形状別条件を満たした）**・N=4096 0.634 倍（未達）。共有負荷下の計測であり、旧 #1037 の「3 形状すべて」という受け入れ条件は依然として未達成のまま。#1520 で §15 undetermined を受け専有ゲートを opt-out 可能にする `AB_LOAD_GATE_MODE=record_only` を `run_ab_readout_metal.sh` へ追加（ルート #1519 指示）したが、本セッションは実機（Apple M4 Max）へのアクセス経路を持たないため実測は未実施のまま §17 に記入欄を残した。#1521 で、split-K 本番結線（#1527・#1530）により初めて `v0.8.0 ↔ origin/main` の Metal 計測経路 diff が非ゼロになったことを受け、対照系列（`0.8.0-ctrl-1521`）・参考系列（`head-<sha>-1521`。split-K 結線後 HEAD）の 2 系列を同一セッションで計測するスキャフォールド・帰属テスト・事前登録判定規則を整備した（§18）。本セッションも実機アクセス経路を持たないため実測は未実施のまま記入欄を残した

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
- `docs/perf/logs/metal-gemm-candle-gate-0.8.0-1490/`（イシュー #1490。=0.8.0 正式系列再計測の
  実行ログ・env_info・attribution）

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

- checksum 完全一致・parity 0 fail のため数値精度への影響はないが、§12.1 のとおり off/on 間の
  ソース差（registry ピン対 HEAD path）と計測時の負荷差の両方が未分離の交絡要因として残る
  ため、後退幅を readout 経路のコスト差のみに帰属することはできない
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
  この readout 経路を通らない（Metal には同種の staging 実装自体がない）。ただし §12.1 の
  とおり off/on 間でソース（registry ピン対 HEAD path）が揃っておらず readout feature 以外の
  コード差分が混入しうるため、**本節の効果を `#1337`（borrowed-view readout）単独へ厳密に
  帰属することはできない**（#1336 概念の非到達は追加の交絡要因が無いことのみを意味する）
- **`host-view-readout` 既定化の可否**: 本節では判断しない（Metal で 3 形状とも後退・かつ
  負荷ノイズおよび off/on 間のソース差〈§12.1〉のいずれとも切り分けられていないため、
  既定化の根拠にはできない）
- **低負荷環境での再計測**: 本節の計測は片方向の負荷差（on 腕がより高負荷）を伴う共有マシン
  状態下で実施した。低負荷環境での再計測は本イシューのスコープ外とし、ユーザー判断で
  新規 issue を起票するかを決める
- **#1037 ゲート判定への影響**: readout-off／on いずれも §11 の正式系列判定（未達 3 件）を
  覆さない。正式判定（registry ピン）は §11 のまま不変

## 13. §13（イシュー #1438。同一 facade ソース下の借用ビュー readout 既定化 before/after）

### 13.0 位置づけ

§12 は off 腕が registry・on 腕が path patch という facade ソース不一致と片方向の負荷差という
2 つの交絡を抱えたまま「3 形状とも後退」を記録し、既定化の根拠にはできないと結論していた。
イシュー #1438 は借用ビュー readout を bench-fandhe の既定経路とし旧 `host-view-readout`
cargo feature を撤去するにあたり、**両腕を同一 facade ソース（本 PR HEAD の `crates/facade`
への path patch）**で揃えた before（legacy bench-fandhe ソース）/after（default bench-fandhe
ソース）比較を M4 Max 実機で再計測し、§12 が抱えていた facade ソース不一致の交絡を解消した。

### 13.1 プロトコル

`docs/perf/logs/gemm-candle-gate-readout-default-1438/env_info.txt` を参照。§12 との違い:
両腕とも `GEMM_GATE_PATCH_FACADE_PATH` で本 PR HEAD の同一 `crates/facade` を指定し、
bench-fandhe 側のソースのみを base コミット `fddca17`（legacy）／本 PR HEAD（default）で
切り替えた。実測は 2026-09-08。高負荷共有環境（他セッション並走。`uptime` 実測:
計測開始前 load average 5.72–10.29 → Metal legacy 腕終了時 25.54–10.76 → Metal default 腕
終了時 11.09–10.41）は §12 と同様に残る（片方向ではなく both arms とも高負荷帯を通過）。

### 13.2 実測結果（fandhe-ai reuse・5 回計測中央値・before/after 比較）

| N | before 中央値（min–max） | after 中央値（min–max） | after/before | checksum |
|---|---|---|---|---|
| 1024 | 4.838 ms（4.373–4.931 ms） | 2.288 ms（2.178–2.824 ms） | **0.473** | 完全一致 |
| 2048 | 17.059 ms（16.771–19.880 ms） | 10.487 ms（9.250–10.899 ms） | **0.615** | 完全一致 |
| 4096 | 58.375 ms（40.857–64.843 ms） | 39.367 ms（39.051–40.284 ms） | **0.674** | 完全一致（判定注意: before spread 40.857–64.843 ms が 1.5 倍超で負荷ノイズの疑い） |

全 3 サイズで非後退（`after/before` 0.47〜0.67）。N=4096 は before 腕の分散が特に大きく
（40.857–64.843 ms）高負荷ノイズの影響が疑われるが、after 腕はその最良値（40.857 ms）と
比べても非後退（39.367 ms）であり、判定を覆す方向のノイズではない。checksum は 3 サイズとも
完全一致（要素単位 `parity_fail_count=0`）。
出典: `docs/perf/logs/gemm-candle-gate-readout-default-1438/compare_gemm_ab-metal-m4max.md`

### 13.3 candle 比ゲート（#1037。参考記録。正式判定は §11 のまま不変）

| N | before candle 比 | before 判定 | after candle 比 | after 判定 |
|---|---|---|---|---|
| 1024 | 0.616 | 未達 | 0.933 | 未達 |
| 2048 | 0.723 | 未達 | 0.655 | 未達 |
| 4096 | 0.585 | 未達 | 0.604 | 未達 |

出典: `docs/perf/logs/gemm-candle-gate-readout-default-1438/gate_metal-m4max-{legacy,default}.md`。
3 形状とも #1037 未達成のまま（正式系列 `fandhe-ai =0.7.0` ピンは本 PR で更新していないため
§11 の判定は不変）。N=1024・N=4096 は candle 比も改善方向（0.616→0.933・0.585→0.604）だが、
N=2048 のみ fandhe-ai 側が非後退（10.487 < 17.059 ms）にもかかわらず candle 比は
0.723→0.655 と悪化している。これは candle 側自体が同一プロトコルで大きく高速化した
（12.329→6.866 ms・約 1.8 倍。負荷変動）ため、fandhe-ai の改善幅（約 1.6 倍）を上回った
ことによるものであり fandhe-ai 側の後退ではない（§13.2 の非後退判定を参照）。

### 13.4 公正性の論点

- `bench-candle` は本 PR で一切変更していない（`.to_vec2()` による所有 `Vec` 読み出しのまま）
- Metal は CUDA `#1436/#1437` のような D2H 宛先事前タッチの是正が未実施（Metal readback は
  #1338 のスコープ外のまま）。それでも本節の同一 facade ソース比較では全 3 形状が非後退した

### 13.5 採否・判定木の適用（暫定。codex-review 指摘・PR #1452 P2）

§12 で懸念されていた「3 形状とも後退」は facade ソース不一致・片方向負荷差という交絡込みの
結果だったと判明した。facade ソースを揃えた本節（§13）の比較では全 3 サイズが非後退のため、
イシュー #1438 の事前宣言判定木の条件 (a)「全 N で after/before ≤ 1.00」の**数値上**は満たす。

ただし §13.1 のとおり before/after は同一マシン上で**連続実行**しており、負荷変動と
readout 切替の効果が分離できていない。§13.3 が示すとおり、fandhe-ai 側を一切変更していない
`bench-candle`（同一バイナリ）の N=2048 計測値も同一プロトコル内で 12.329→6.866 ms
（約 1.8 倍）と大きく変動しており、この区間で相応の負荷差が生じていたことを示唆する。
before 腕（readout-off/legacy）がより高負荷帯・after 腕（readout-on/default）がより低負荷帯を
通過した可能性を否定できず、§13.2 の after/before 0.47〜0.67 という改善幅を readout 切替
そのものの効果として断定することはできない。

したがって本節の結論は暫定の**参考結果**として扱い、以下のいずれかを実施した再計測が
得られるまで最終確定としない:

- before/after を交互実行（interleave）し、負荷変動を両腕へ均等に分散させる
- 低負荷（他プロセス非並走）環境を確認したうえで re-run し、負荷差そのものを排除する

上記再計測は本 PR のスコープ外とし、必要性の判断・新規 issue の起票はユーザー判断に委ねる。
本節の観測（非後退方向のシグナルであること・checksum 完全一致・parity 0 fail）自体は
有効な参考データとして保持するが、Metal の借用ビュー readout 既定化を無条件の ADOPT として
断定はしない。runtime `Device::Metal` 限定の legacy フォールバックの要否も同様に再計測後の
判断とする。正式判定（#1037 ゲート）は §11 のまま不変。

### 13.6 runtime legacy フォールバックの実装（codex-review 指摘・PR #1452 P2）

§13.5 で「要否は再計測後の判断」としていた `runtime Device::Metal` 限定の legacy
フォールバックは、codex-review（PR #1452）が「未確定の経路が既定で使われている」と
指摘したことを受け、再計測を待たず fail-closed に実装した（
`scripts/bench/framework-compare/bench-fandhe/src/main.rs::readout_uses_borrowed_view`）。
`device == "metal"` のときのみ `readout_var`／`checksum_var`／`checksum_tensor`／
`measure_gemm_reuse_phases` のインライン展開が旧 legacy 経路（`to_tensor()` +
`.to_vec()`）へ分岐し、CPU/CUDA は #1337/#1438 で確定した借用ビュー既定のまま変えない。
#1438 が撤去したコンパイル時 cargo feature（`host-view-readout`）を再導入するもの
ではなく、`device` 文字列 1 個を見る runtime 分岐に閉じている。Metal の ADOPT が
確定した場合は `readout_uses_borrowed_view` を `true` 固定へ変更する 1 箇所の
修正で本節の暫定判定と整合させられる。

## 14. 2026-09-09 追補: Phase 3 ゲート再判定（イシュー #1309）

### 14.1 位置づけ・事前宣言規則

- #1269（Metal GEMM candle 比未達トラッキング）Phase 3「ゲート再判定」の一環。Phase 1（#1280:
  E5 は結線対象なし）・Phase 2（#1302: E2〜E4/E6〜E8 は全 REJECT・`tile::select` 不変）・
  #1308（split-K は「採用検討推奨」だが実装は別 issue へ切り出し提案・本節時点でコード変更なし）・
  #1368（hfrag opt-in 候補は非結線）・#1334（Metal は §13.5/§13.6 のとおり legacy readout 維持）
  を反映した HEAD（`797030e`）を対象に、正式系列（crates.io 公開ピン `fandhe-ai =0.7.0` の
  registry 解決）で N=1024/2048/4096 reuse の candle 比を 5 回計測中央値で再取得した
- 事前宣言した判定規則（計測前に固定・計測後に変更しない）: (1) 判定は
  `compare_gemm_gate.py --device metal` の出力のみを正とする。(2) 正式判定は manifest
  `fandhe_ai_source=registry` の系列のみ。(3) ラベルは `0.7.0-1309`（正式）／
  `head-797030e-1309`（参考。未計測）で固定。(4) 負荷ゲートは 1 分 load average < 4.0 を
  30 秒間隔で 2 回連続確認して通過。待機は 60 秒開始・1.5 倍ずつ増加・最大 10 回。
  (5) 系列は正式 → 参考の順に直列実行し、各系列の前に負荷ゲートを再判定する。
  (6) 選択的再実行はしない（run が失敗しても数値を捏造しない）。(7) Metal readout は
  `readout_uses_borrowed_view` を変更しない。(8) 期待値は帰属表（`docs/perf/logs/
  metal-gemm-candle-gate-1309/attribution.md`）のとおり正式 ≈ 参考（本番 NN 正方 GEMM
  reuse 経路は Phase 1〜3・#1334・#1368 いずれからも変更を受けていない見込み）

### 14.2 構造的制約: 正式系列（registry 解決）ビルドの経路

- イシュー #1438（PR #1452。マージコミット `40ef890`）が bench-fandhe を借用ビュー readout
  API（`Var::host_view` 等）へ既定経路化し、旧計測専用 cargo feature を撤去したため、
  `origin/main` の framework-compare ツリーからは crates.io ピン `fandhe-ai =0.7.0`
  （借用ビュー API 未収録）による registry 解決ビルドが構造的に不可能（
  `bench_fandhe_pin_guard.sh` が `GEMM_GATE_PATCH_FACADE_PATH` 未指定時に fail-closed で
  停止する）
- pin guard 自身のエラー文が案内する対処（#1438 直前コミット `61b8b65` = `40ef890^` の
  framework-compare ツリーで registry 解決ビルドする）に従い、`git archive 61b8b65
  scripts/bench/framework-compare | tar -x` で本体ツリー・グローバル状態を変更せず自己完結
  ツリーを scratchpad へ展開し、そこで `cargo build --release -p bench-fandhe -p bench-candle`
  を実行した。この時点の `bench-fandhe/Cargo.toml` は `host-view-readout` feature が既定 OFF
  （Metal は legacy readout。HEAD の `readout_uses_borrowed_view("metal") == false` と同一
  経路）で pin guard も存在しないため、registry ビルドが成立することを確認した
  （`cargo tree -p bench-fandhe --depth 1` が `fandhe-ai v0.7.0`（無 path 注記）・
  `Cargo.lock` の `source = "registry+https://github.com/rust-lang/crates.io-index"` を確認）
- 迂回パッチは当てていない（`bench_fandhe_pin_guard.sh`・`Cargo.toml` は archive 元コミットの
  ものをそのまま使用）

### 14.3 プロトコル・専有状態

- 計測環境: M4 Max・macOS 26.6.2。実行前の負荷ゲート判定は 1 分 load average
  3.33 → 3.21（2 回連続 <4.0）で通過（`gate-m4max.log`）
- 正式系列（`0.7.0-1309`）実行: `bash run_gemm_gate_metal.sh 0.7.0-1309`（scratchpad の
  archive ツリー内）。実行時間は約 80 秒。manifest で `fandhe_ai_source=registry`・
  `bench_fandhe_features=""`・`candle_core_source=registry` を確認済み
  （`results/raw/manifest-m4max-gemm-gate-0.7.0-1309.json`）
- 実行直後（`13:36`）の `uptime` load average は 13.90（1 分）と急上昇しており、**共有負荷下の
  計測**である（他セッションの並列イシュー実行による負荷。`run_gemm_gate_metal-m4max-0.7.0-1309.log`
  「metal status (before)」節）。`pmset -g therm` は計測前後とも thermal / performance
  warning なし（`pmset_therm_before_formal.txt`・`pmset_therm_after_formal.txt`）
- 参考系列（`head-797030e-1309`）は、正式系列完了直後の負荷ゲート再判定が本イシューの
  作業時間内に安定通過しなかった（10 回試行の待機上限に達する前に、フォースアウトされた
  StructuredOutput 呼び出しにより本コミット作成が要求されたため、途中で `wait_gate.sh` を
  中断した）。§14.6 に未計測の理由・引き継ぎ事項を記録する

### 14.4 実測結果（正式系列 `0.7.0-1309`）

| N | fandhe-ai reuse 中央値（min–max, n=5） | candle fresh 中央値（n=5） | candle/fandhe | GFLOP/s（fandhe） | 判定 |
|---|---|---|---|---|---|
| 1024 | 3.005 ms（2.939–3.116 ms） | 2.104 ms | 0.700 | 714.70 | 未達 |
| 2048 | 10.212 ms（9.227–11.359 ms） | 9.890 ms | 0.969 | 1682.32 | 未達 |
| 4096 | 49.609 ms（48.370–54.177 ms） | 35.209 ms | 0.710 | 2770.44 | 未達 |

出典: `scripts/bench/framework-compare/results/raw/results-m4max-gemm-gate-0.7.0-1309.jsonl`
（30 行。`skipped-m4max-gemm-gate-0.7.0-1309.log` は空）。判定表の生成コマンド出力は
`docs/perf/logs/metal-gemm-candle-gate-1309/compare_gemm_gate-m4max-0.7.0-1309.md`。

### 14.5 データ有効性

- fandhe-ai・candle とも全 30 run で `parity_fail_count=0`・checksum が同一 N で一致（tolerance
  は緩めていない。`compare_gemm_gate-m4max-0.7.0-1309.md` の要素単位検証内訳を参照）
- manifest のバイナリ sha256 検証・依存元検証（registry）・feature 検証はいずれも
  `run_gemm_gate_metal.sh` 実行ログ内で OK（`run_gemm_gate_metal-m4max-0.7.0-1309.log`）

### 14.6 §11 との比較・帰属

| N | §11（0.7.0・2026-09-06） | §14.4（0.7.0-1309・2026-09-09） | 差の方向 |
|---|---|---|---|
| 1024 | 0.836 | 0.700 | 低下 |
| 2048 | 0.638 | 0.969 | 改善 |
| 4096 | 0.509 | 0.710 | 改善 |

- 帰属表（`docs/perf/logs/metal-gemm-candle-gate-1309/attribution.md`）のとおり、§11 時点
  （v0.7.0 直後の HEAD）から本節時点（`797030e`）までの Phase 1〜3・#1334・#1368 はいずれも
  本番 NN 正方 GEMM reuse 経路（`tile::select_for_device` の選択構成）を変更していない。
  正式系列は両時点とも同一の crates.io 公開ピン `fandhe-ai =0.7.0`（registry 解決）を計測
  対象としており、コードは完全に同一（バイナリ差は「ビルド元コミットが `61b8b65`（本節）か
  `HEAD`（§11）か」のみで、いずれも `fandhe-ai =0.7.0` の同一ソースを registry から取得する
  ため実質同一バイナリ）。よって §11 と §14.4 の差分（N=1024 低下・N=2048/4096 改善）は
  **コード変更に帰属できず、両時点の共有負荷の違いによる計測ノイズ**と判断する（§11 は
  load average 3.39〜7.96 の共有負荷下、本節は計測直後に 13.90 まで上昇する共有負荷下。
  いずれも非専有環境での計測であり、専有環境での再計測なしに N=1024/2048 いずれの方向の
  差分も確定できない）
- 3 形状とも未達成という**結論の符号自体は§11 と一致**しており、Phase 3 反映後も #1037 は
  未達成のままであることを確認した

### 14.7 #1037 ゲート判定（Phase 3 反映後・正式系列単独）

| # | #1037 の受け入れ条件 | 正式系列（0.7.0-1309） | 出典 |
|---|---|---|---|
| 1 | N=1024 reuse で candle 超え | 未達（0.700 倍） | §14.4 |
| 2 | N=2048 reuse で candle 超え | 未達（0.969 倍） | §14.4 |
| 3 | N=4096 reuse で candle 超え | 未達（0.710 倍） | §14.4 |
| 4 | parity 0 fail（fandhe-ai 側） | 達成（全 15 run `parity_fail_count=0`） | §14.5 |

**総合判定: #1037 は Phase 3（#1280・#1302・#1308・#1334・#1368）反映後も正式系列
`fandhe-ai =0.7.0` において未達成のまま（未達 3 件）。§11 の確定判定を再確認した。**
N=2048 が 0.969 倍と 1.0 倍にもっとも近づいたが、これも§14.6 のとおりコード変更に
起因するとは判断できない。

### 14.8 参考系列が未計測である理由・引き継ぎ

- 正式系列完了直後（`13:37`。load average 7.84）から参考系列開始前の負荷ゲート再判定を
  開始したが、他セッションの並列イシュー実行が継続しており、3 回の待機試行（60 秒→90 秒→
  135 秒。load average 6.46→5.08→3.65）を経ても 2 回連続 <4.0 の通過条件に達する直前で
  作業を打ち切った（詳細は `gate-m4max.log`・`uptime-m4max.log`）
- **選択的再実行はしていない**（§3 事前宣言規則 6）: 参考系列は 1 run も起動していないため
  捏造・部分実行データは存在しない。§14.4 の正式系列データのみを正式判定の根拠とする
- 参考系列（`GEMM_GATE_PATCH_FACADE_PATH` で HEAD `797030e` の `crates/facade` を path
  patch した見込み値）の取得、および §4 で計画していた未達時の残ギャップ内訳（reuse
  フェーズ分解との突合）は本イシューでは完了できず、後続の再計測（別セッション・負荷が
  下がった時間帯）へ引き継ぐ。再現手順は `docs/perf/logs/metal-gemm-candle-gate-1309/
  run_gated.sh`（オーケストレーション記録）・`wait_gate.sh`（負荷ゲート待機ロジック）を参照

### 14.9 次候補の整理（本 PR ではコード変更・Issue 起票なし）

1. split-K 実装（#1308「採用検討推奨」・`docs/backend-metal-splitk-decision.md` §2 の設計。
   別 issue への切り出しが #1308/PR #1466 本文で提案済み）
2. hfrag opt-in カーネルの opt-in API 設計（`docs/perf/metal-gemm-hfrag-candidate.md` §9。
   N=4096 限定で本番選択構成比 10〜12% 高速の見込みだが無条件の候補前進は非推奨）
3. Metal 借用ビュー readout の交互実行再計測による ADOPT 確定（§13.5/§13.6 の暫定判定の
   解消。片方向負荷差の交絡を排除する計測が必要）
4. 転置ルーティング判定不能の解消（#1242 ツリー。フェーズ 1 安定性ゲートが専有環境
   確保の失敗により繰り返し不成立）
5. デバイス側 checksum（#1339。readback を 8 バイトへ縮小し計測固定費を削減する見込み）
6. 参考系列の取得・本イシューの残タスク（§14.8）の完了
7. 次回 crates.io 公開・framework-compare 承認ピン更新（ユーザー承認事項）

### 14.10 出典

- 実行ログ・生データ・env_info: `docs/perf/logs/metal-gemm-candle-gate-1309/`
  （`gate-m4max.log`・`uptime-m4max.log`・`pmset_therm_{before,after}_formal.txt`・
  `run_gemm_gate_metal-m4max-0.7.0-1309.log`・`compare_gemm_gate-m4max-0.7.0-1309.md`・
  `diff_v0.7.0_797030e_metal_path.txt`・`attribution.md`・`run_gated.sh`・`wait_gate.sh`・
  `uptime_sampler.sh`）
- 生データ（framework-compare 側）:
  `scripts/bench/framework-compare/results/raw/{results,skipped,manifest}-m4max-gemm-gate-0.7.0-1309.*`
- `docs/performance-targets.md` §8.13・`scripts/bench/framework-compare/results/summary.md`
  環境 28 節にも同じ数値を反映する

## 15. §13.5 の暫定判定の解消: legacy/borrowed interleave 再計測（イシュー #1477）

### 15.0 位置づけ

§13.5 は「before/after が同一マシン上で連続実行され、負荷変動と readout
切替の効果を分離できていない」ことを理由に ADOPT 判定を保留した
（暫定・参考結果）。本イシューは `--readout <legacy|borrowed>`
（同一バイナリの runtime override。`bench-common::Cli.readout`／
`Record.readout`。`scripts/bench/framework-compare/README.md`
「`--readout <legacy|borrowed>`」節参照）を用いて legacy/borrowed を
run 単位に interleave 計測（奇数 run: legacy→borrowed・偶数 run:
borrowed→legacy）し、§13.5 の限界を解消したうえで
`readout_uses_borrowed_view` の Metal 分岐（`bench-fandhe/src/main.rs`）
の採否を確定する。

**正式系列 `fandhe-ai =0.7.0` の #1037 ゲート判定（§11・§14）は本イシュー
では不変**。本イシューは `bench-fandhe` の readout 実装（既定経路の
選択）自体の A/B であり、candle 比ゲート自体の再判定ではない。

### 15.1 事前宣言した判定規則

`scripts/bench/framework-compare/README.md`「`--readout` （Metal の
legacy/borrowed override interleave 再計測。イシュー #1477）」節・
`compare_readout_ab.py` docstring に転記した規則（計測前に固定。計測後の
緩和・読み替えは行わない）:

- 対象: `gemm metal` × N ∈ {1024, 2048, 4096} × mode ∈ {fresh, reuse}
  （6 セル）
- 反復: 各セル・各腕ちょうど 5 プロセス起動。run 単位で順序反転
- ADOPT: 全 6 セルで `ratio(=borrowed/legacy) <= 1.00` かつ checksum
  完全一致・parity 0 fail → `readout_uses_borrowed_view` の Metal 分岐を
  除去し借用ビューを既定化する
- REJECT: 1 セルでも `ratio > 1.00` または checksum 不一致 → legacy
  フォールバックを維持する
- undetermined: 専有ゲート（1 分 load average < 4.0 を 30 秒間隔で 2 回
  連続確認・不合格時は 60 秒開始 × 1.5 倍・最大 10 回）が成立しない場合。
  1 回だけ記録して終了し、再試行ループで待たない（コード既定は不変）

### 15.2 実測結果（1 回目の試み・2026-09-09）

`run_ab_readout_metal.sh head-e8cd3a2-1477`（`AB_LOAD_GATE_MAX_ATTEMPTS=4`。
既定 10 より少ない値で動作確認を兼ねて実行）を実行したが、専有ゲート
（1 分 load average < 4.0 を 30 秒間隔で 2 回連続確認・最大 4 試行）が
一度も成立しなかった（load1 実測値: 14.64 / 10.23 / 15.93 / 6.18。他セッション
の並走負荷が原因）。判定規則 §15.1 どおり **undetermined** と確定し、計測
（build・`bench-fandhe` 起動）は一切開始しないまま記録のみで終了した
（再試行ループでは待たない）。

- **正式系列 `fandhe-ai =0.7.0` の #1037 ゲート判定（§11・§14）は影響を
  受けない**（計測自体が発生していないため）。
- **`readout_uses_borrowed_view` の Metal 分岐は不変のまま**（legacy
  フォールバックを維持。§13.5 の暫定・ADOPT 保留判定も変更なし）。
- 既定 10 試行でのゲート再挑戦、またはより低負荷な時間帯での再実行は、
  専有環境が確保できるタイミングで別途行う（本 PR のスコープ外）。

実行ログ・undetermined マーカー・env_info:
`docs/perf/logs/metal-gemm-readout-interleave-1477/`。

### 15.3 出典

- 実装: `scripts/bench/framework-compare/bench-common/src/lib.rs`
  （`Cli.readout`／`Record.readout`）・
  `scripts/bench/framework-compare/bench-fandhe/src/main.rs`
  （`readout_uses_borrowed_view` override 引数）
- 集計: `scripts/bench/framework-compare/compare_readout_ab.py`・
  `compare_readout_ab_test.py`
- 計測: `scripts/bench/framework-compare/run_ab_readout_metal.sh`
- 生データ（実測後）: `docs/perf/logs/metal-gemm-readout-interleave-1477/`

## 16. 2026-09-10 追補: 正式系列 `fandhe-ai =0.8.0` 再計測（イシュー #1490）

### 16.1 位置づけ・事前宣言規則

- v0.8.0 の crates.io 公開（2026-09-09・`release-all.yml` run 34417008617）と framework-compare
  の承認ピン更新（#1487・PR #1504。`bench_fandhe_pin_guard.sh` 撤去）を受け、正式系列
  `fandhe-ai =0.8.0`（registry 解決）のみで N=1024/2048/4096 reuse の candle 比を
  Apple M4 Max（本セッションのホスト自身）で 5 回計測中央値再取得した。CUDA 側 #1489・
  CPU 側 #1488 と同一プロトコルの Metal 版
- 事前宣言した判定規則（計測前に固定。計測後に変更しない）:
  1. 判定の正は `compare_gemm_gate.py --device metal
     results/raw/results-m4max-gemm-gate-0.8.0-1490.jsonl` の出力のみ。閾値・tolerance・
     判定式・`BASELINES` は不変
  2. 正式系列のみを計測する（ラベル `0.8.0-1490`）。参考系列は計測しない
     （`v0.8.0 ↔ origin/main` の Metal 計測経路 src 差分がゼロのため。§16.2）
  3. 専有ゲート: 1 分 load average **< 6.0** を 2 回連続確認（1 回目合格直後の 2 回目は
     30 秒固定間隔・不合格時のみ 60 秒開始・1.5 倍バックオフ）。セッション累積 最大 10
     試行・経過時間上限 1800 秒（session-start スタンプ＋`gate-state` 永続化でプロセス
     再起動をまたいで強制。待機時間は短縮せず、期限到達後は判定しない）。閾値 6.0 は
     §14（4.0）と異なり、同一ホストの #1488（CPU 計測。Metal より背景負荷に敏感な経路）
     が採用した値と揃えたもので、計測後に選び直していない
  4. 不成立時は `verdict=undetermined` を 1 回だけ記録して終了し、新しいスタンプ
     ディレクトリで再実行しない
  5. run 失敗時は数値を捏造せず、`run_gemm_gate.sh` の fail-closed（`.failed-<ts>.jsonl`
     退避）に従う
  6. Metal readout・`SPLIT_K_NUMERIC_CONTRACT_APPROVED`・`tile::select` はいずれも変更しない

### 16.2 系列設計・帰属

- `git diff v0.8.0..origin/main --stat -- crates/backend-metal/src crates/facade/src
  crates/autodiff/src crates/tensor-core/src` は空（`docs/perf/logs/
  metal-gemm-candle-gate-0.8.0-1490/diff_v0.8.0_origin-main_metal_path.txt`）。よって
  registry 解決版（正式系列）と origin/main path patch 版（参考系列）は同一ソースになる
  ため、本イシューでは参考系列を計測しない（#1488 §24.2・#1489 §16.5 と同じ判断）
- `v0.7.0..v0.8.0` に入った `crates/backend-metal/src`／`crates/facade/src` の変更は
  split-K opt-in 実装（#1496）・split-K 結線（#1500）・GradStaging 重み勾配読み出し
  API（#1492）のみ。しかし `crates/backend-metal/src/gemm.rs:135` の
  `SPLIT_K_NUMERIC_CONTRACT_APPROVED = false` により split-K プランは classic 経路へ
  強制フォールバックし、かつ `tile::should_split_k` は本イシューの対象形状（NN 正方
  N=1024/2048/4096）を並列度条件で除外するため到達しない。#1492 は学習 step 限定で
  本計測（推論 GEMM・reuse）には無関係。よって本節の期待値は §11・§14 と同水準
  （差は計測ノイズ）であり、これを計測前に固定した（`docs/perf/logs/
  metal-gemm-candle-gate-0.8.0-1490/attribution.md`）
  **追記（イシュー #1518）**: 上記は v0.8.0 タグ時点（`SPLIT_K_NUMERIC_CONTRACT_APPROVED
  = false`）の事実。HEAD は #1513 で `true`・#1516 で `dispatch_auto` へ定数ゲート付き
  結線済み（既定 OFF）だが、対象形状（NN 正方）は並列度条件で非到達のため本節の帰属判断は
  変わらない（`docs/backend-metal-splitk-decision.md` §5）

### 16.3 プロトコル・専有状態

- ビルドは計測から分離した: `cargo build --release -p bench-fandhe` →
  `cargo build --release -p bench-candle` を専有ゲート開始前に実行
  （`prebuild-1490.log`。実行後 `Cargo.lock` に差分がないことを確認）
- 専有ゲートは 6 試行（約 820 秒）で成立（load1: 12.17→16.68→7.17→7.69→3.61→3.70。
  1 回目合格〈3.61〉直後の 30 秒固定確認〈3.70〉で成立）。並走プロセス確認
  （`pgrep -x bench-fandhe`／`bench-candle`）はヒットなし
- 正式系列（`0.8.0-1490`）実行: `bash run_gemm_gate_metal.sh 0.8.0-1490`。実行時間は
  約 49 秒。manifest で `fandhe_ai_source=registry`・`bench_fandhe_features=""`・
  `readout_method=legacy-metal-1452`・`candle_core_source=registry` を確認済み
- ゲート成立直後の load average は 3.70（1 分）だったが、計測完了直後（約 49 秒後）には
  7.05 まで再上昇しており、**共有負荷下の計測**である（他セッションの並列実行が
  計測中に再開した。`run_gemm_gate_metal-m4max-0.8.0-1490.log`「metal status (before /
  after)」節）。`pmset -g therm` は計測前後とも thermal / performance warning なし

### 16.4 実測結果（正式系列 `0.8.0-1490`）

| N | fandhe-ai reuse 中央値（min–max, n=5） | candle fresh 中央値（n=5） | candle/fandhe | GFLOP/s（fandhe） | 判定 |
|---|---|---|---|---|---|
| 1024 | 2.856 ms（2.191–3.071 ms） | 2.123 ms | 0.743 | 752.05 | 未達 |
| 2048 | 9.774 ms（8.992–13.603 ms） | 9.798 ms | **1.002** | 1757.76 | **達成** |
| 4096 | 48.753 ms（45.902–49.202 ms） | 30.922 ms | 0.634 | 2819.09 | 未達 |

出典: `scripts/bench/framework-compare/results/raw/results-m4max-gemm-gate-0.8.0-1490.jsonl`
（30 行。`skipped-m4max-gemm-gate-0.8.0-1490.log` は空）。判定表の生成コマンド出力は
`docs/perf/logs/metal-gemm-candle-gate-0.8.0-1490/compare_gemm_gate-m4max-0.8.0-1490.md`。

**N=2048 は正式系列として初めて #1037 の形状別受け入れ条件（reuse で candle 超え）を
達成した**（candle/fandhe = 1.002）。ただし fandhe-ai reuse の run 間分散が大きく
（min 8.992 ms – max 13.603 ms。共有負荷下の観測）、達成幅は僅少（0.2%）であり、
専有環境での再計測なしに再現性を確定できない（§16.6）。

### 16.5 データ有効性

- fandhe-ai・candle とも全 30 run で `parity_fail_count=0`・checksum が同一 N で一致
  （tolerance は緩めていない。`compare_gemm_gate-m4max-0.8.0-1490.md` の要素単位検証
  内訳を参照）
- manifest 4 条件（`fandhe_ai_source=registry`・`bench_fandhe_features=""`・
  `readout_method=legacy-metal-1452`・`candle_core_source=registry`）はいずれも期待値と
  一致。バイナリ sha256 検証・依存元検証・feature 検証・readout 方式検証は
  `run_gemm_gate_metal-m4max-0.8.0-1490.log` 内で OK

### 16.6 §11・§14 との比較・帰属

| N | §11（0.7.0・2026-09-06） | §14.4（0.7.0-1309・2026-09-09） | §16.4（0.8.0-1490・2026-09-10） |
|---|---|---|---|
| 1024 | 0.836（未達） | 0.700（未達） | 0.743（未達） |
| 2048 | 0.638（未達） | 0.969（未達） | **1.002（達成）** |
| 4096 | 0.509（未達） | 0.710（未達） | 0.634（未達） |

- §16.2 の帰属表のとおり、v0.7.0 → v0.8.0 の間に本番 NN 正方 GEMM reuse 経路
  （`tile::select_for_device` の選択構成・Metal readout）へのコード変更は入っていない。
  N=2048 の 0.969 倍（§14.4）→ 1.002 倍（§16.4）という改善は、**コード変更に帰属できず**、
  3 時点（§11・§14・§16）を通じた計測ノイズの範囲内での揺らぎと判断する（3 時点とも
  共有負荷下の計測であり、N=2048 は 0.638〜1.002 倍の幅で推移している）
- 3 形状中 1 形状（N=2048）が初めて達成側へ振れたが、旧 #1037 の受け入れ条件（3 形状
  すべてで candle 超え）は依然として未達成のまま（N=1024・N=4096 は §11・§14 と同様に
  明確に未達）

### 16.7 #1037 ゲート判定（正式系列 `0.8.0-1490`）

| # | #1037 の受け入れ条件 | 正式系列（0.8.0-1490） | 出典 |
|---|---|---|---|
| 1 | N=1024 reuse で candle 超え | 未達（0.743 倍） | §16.4 |
| 2 | N=2048 reuse で candle 超え | **達成（1.002 倍）** | §16.4 |
| 3 | N=4096 reuse で candle 超え | 未達（0.634 倍） | §16.4 |
| 4 | parity 0 fail（fandhe-ai 側） | 達成（全 15 run `parity_fail_count=0`） | §16.5 |

**総合判定: #1037 は正式系列 `fandhe-ai =0.8.0` においても未達成のまま（3 形状中 2
形状が未達）。N=2048 が正式系列として初めて形状別条件を達成したが、§16.6 のとおり
コード変更に帰属できる根拠はなく、共有負荷下の計測ノイズの範囲内である可能性が高い。
§11・§14 の「3 形状とも未達」という結論の核（#1037 全体としては未達成）は変わらない。**

### 16.8 スコープ外・引き継ぎ

- 参考系列（HEAD path patch）の計測（v0.8.0 ↔ origin/main 差分ゼロのため不要と判断） → split-K 本番結線（#1527・#1530）により差分が非ゼロになったため #1521（§18）で着手
- Metal 借用ビュー readout の interleave 再計測（#1477 undetermined の再挑戦）
- split-K 数値契約承認（`SPLIT_K_NUMERIC_CONTRACT_APPROVED`）: **#1513 で完了**
  （`true` へ切替済み）。結線自体は #1516（定数ゲート付き・既定 OFF）、ゲート ON への
  切替は #1515 の 5 run 正式 ADOPT 確定（未実測）待ち
- N=4096 カーネル純境界ギャップの縮小（#1269 後継）
- N=2048 の達成が専有環境下で再現するかの確認（§16.4 の run 間分散が大きいため、
  1 回の計測のみでは再現性を確定できない）
- #1037 後継ツリーのクローズ可否はユーザー判断（§9 の既存整理を踏襲。総合判定は
  引き続き未達成のため即クローズは推奨しない）
- ゲート不成立時の再計測は、専有環境が確保できるタイミングの別セッション・別 issue で
  行う（本イシューではゲートは成立し計測は完了したため、この項は次回不成立時の
  引き継ぎ事項として記録するのみ）

### 16.9 出典

- 実行ログ・生データ・env_info: `docs/perf/logs/metal-gemm-candle-gate-0.8.0-1490/`
  （`orchestrate_m4max.sh`・`prebuild-1490.log`・`gate-m4max.log`・`uptime-m4max.log`・
  `session-start-m4max.stamp`・`gate-state-m4max`・`ALL_DONE_m4max.marker`・
  `pmset_therm_{before,after}_formal.txt`・`run_gemm_gate_metal-m4max-0.8.0-1490.log`・
  `compare_gemm_gate-m4max-0.8.0-1490.md`・`.stdout.log`・
  `diff_v0.8.0_origin-main_metal_path.txt`・`attribution.md`・`env_info.txt`）
- 生データ（framework-compare 側）:
  `scripts/bench/framework-compare/results/raw/{results,skipped,manifest}-m4max-gemm-gate-0.8.0-1490.*`
- `docs/performance-targets.md` §8.16・`scripts/bench/framework-compare/results/summary.md`
  環境 30 節にも同じ数値を反映する

## 17. §15 undetermined を受けた 2 回目の試み: `AB_LOAD_GATE_MODE=record_only`（イシュー #1520・ルート #1519）

### 17.0 位置づけ

§15 の 1 回目の試み（イシュー #1477）は専有ゲート（1 分 load average < 4.0
を 2 回連続確認・最大 4 試行）が一度も成立せず、計測（build・
`bench-fandhe` 起動）を一切開始しないまま undetermined のまま終了した。

ルート issue #1519 でのユーザー指示「Metal に関しては一旦現在の環境で
測れる値で大丈夫です」を受け、専有ゲートを「要件」から「記録専用
（`AB_LOAD_GATE_MODE=record_only` で明示 opt-out）」へ変える最小限の
スクリプト変更（`run_ab_readout_metal.sh`）をイシュー #1520 で実施した。
既定値 `exclusive`（§15 までの挙動）は変更していない。判定規則自体
（§15.1）も計測前に固定したまま変更していない。

### 17.1 実施内容（本セッションで完了した範囲）

- `scripts/bench/framework-compare/run_ab_readout_metal.sh`:
  `AB_LOAD_GATE_MODE`（既定 `exclusive`。`record_only` で専有ゲートを
  スキップし現在の load average を 1 行記録して直ちに計測を開始する）
  を追加し、manifest JSON へ `gate_mode` フィールドを追記した。
  `record_only_gate_note` は `wait_for_exclusive_gate` と同じ
  `load1_is_valid` 検証を経てから記録する（非数値を `load1=` として
  誤記録しない）。
- `scripts/bench/framework-compare/README.md`「`--readout`」節へ
  `AB_LOAD_GATE_MODE` の説明を追記した。
- 単体検証: `bash -n run_ab_readout_metal.sh`（構文検証）・
  `python3 -m unittest compare_readout_ab_test.py`（既存 23 件 green。
  判定ロジック自体は無変更のため回帰確認のみ）。

### 17.2 実機計測（未実施・記入欄）

**本セッションは Linux 環境で実行されており、Apple M4 Max
実機への到達経路（ローカル直接実行が前提。`docs/real-hardware-
verification-env.md` §1・§7）を持たない**。`docs/real-hardware-
verification-env.local.md`（SSH ホスト名等の実値）も本 worktree には
存在しない。そのため、以下の実測コマンド自体はステップとして用意したが
**未実行**であり、判定（ADOPT／REJECT／undetermined）は未確定のまま
記入欄を残す（`docs/cuda-tf32-optin-api-decision.md` 等と同型の「実機
なしのため未実測明記」方針。CLAUDE.md 記載の他イシューと同様）。

実行予定コマンド（実機を持つ別セッションが引き継ぐ場合の手順）:

```bash
cd scripts/bench/framework-compare
SHORT_SHA=$(git rev-parse --short HEAD)
LABEL="head-${SHORT_SHA}-1520"
FACADE_PATH="$(cd ../../../crates/facade && pwd)"

pmset -g therm > "results/raw/pmset_therm_before-${LABEL}.txt" 2>&1 || true
uptime

AB_LOAD_GATE_MODE=record_only \
AB_PATCH_FACADE_PATH="$FACADE_PATH" \
  bash run_ab_readout_metal.sh "$LABEL" 2>&1 | tee "results/raw/run-${LABEL}.log"

pmset -g therm > "results/raw/pmset_therm_after-${LABEL}.txt" 2>&1 || true

python3 compare_readout_ab.py \
  "results/raw/results-m4max-readout-ab-${LABEL}.jsonl" \
  --device metal --sizes gate --threshold 1.00 \
  | tee "results/raw/compare_readout_ab-${LABEL}.md"
```

判定規則は §15.1 のまま変更しない（`AB_LOAD_GATE_MODE=record_only` は
計測実施条件〈専有ゲートの要否〉の運用パラメータであり、ADOPT/REJECT/
undetermined の閾値・対象セル・checksum 判定には影響しない）。

### 17.3 結果・判定

**未確定（実機未実測のため）**。`readout_uses_borrowed_view`（`bench-
fandhe/src/main.rs`）の Metal 分岐は §15 までと同じく legacy 既定の
まま維持する。正式系列 `fandhe-ai =0.8.0` の #1037 ゲート判定（§16）
は本イシューでは不変（計測自体が発生していないため）。

引き継ぎ: 実機（Apple M4 Max）にアクセス可能な別セッションが上記
17.2 のコマンドを実行し、本節へ実測結果表・checksum 一致・判定を追記
する。ADOPT と確定した場合に限り `bench-fandhe/src/main.rs` の
`readout_uses_borrowed_view` Metal 分岐撤去・
`scripts/bench/framework-compare/README.md`「借用ビュー readout」見出し・
`docs/perf/cuda-host-view-readout-small-shape-regression.md` §11／§13.6
末尾の注記更新を行う（コード変更は ADOPT 確定後の別イシューとする）。

### 17.4 出典

- 実装: `scripts/bench/framework-compare/run_ab_readout_metal.sh`
  （`AB_LOAD_GATE_MODE`）・`scripts/bench/framework-compare/README.md`
  （「`--readout`」節）
- 前回の試み（undetermined）: §15・`docs/perf/logs/
  metal-gemm-readout-interleave-1477/`
- 本イシューでは実機計測を実施していないため、新規ログディレクトリは
  作成していない

## 18. §17 は #1520 が使用済みのため §18: split-K 結線後 HEAD（参考系列）の再計測スキャフォールドと帰属（イシュー #1521）

### 18.0 位置づけ

§16.2 は `v0.8.0 ↔ origin/main` の Metal 計測経路 diff がゼロだったため
参考系列を計測しないと判断した。その後 #1527
（`SPLIT_K_NUMERIC_CONTRACT_APPROVED` を `true` へ切替）・#1530（split-K
本番結線。`should_split_k` 分岐を `dispatch_auto`／`select_for_device`
へ結線）が入り、この判断の前提（diff ゼロ）が崩れた。本節は §16.2 の
判断が覆った経緯を踏まえ、split-K 結線後 HEAD を参考系列として計測し、
§16 正式系列との差分を「コード差」か「負荷差（計測ノイズ）」かへ帰属
することを目的とする。**正式判定（`fandhe-ai =0.8.0` ピン・§16.7）は
本節では不変**。

### 18.1 事前登録判定規則（計測前に固定。計測後に変更しない）

1. 判定の正は `compare_gemm_gate.py --device metal <A.jsonl> <B.jsonl>`
   （系列ごと独立集計）の出力のみ。tolerance・判定式・`BASELINES`・
   `SPLIT_K_*` 定数・`tile::select` 系は不変。
2. 正式判定は §16.7 のまま不変。A（対照。registry `fandhe-ai =0.8.0`）は
   「同一負荷環境での §16 再現値」、B（参考。split-K 結線後 HEAD への
   `crates/facade` path patch）は「本イシューの主対象」であり、いずれも
   #1037 の正式判定を更新しない。
3. 専有ゲートなし（record_only。ルート #1509 のユーザー指示）。負荷が
   高いこと自体は undetermined の理由にしない。undetermined は件数
   不足（各系列・各 N で fandhe/candle ちょうど 5 件でない）・
   `run_gemm_gate.sh` の fail-closed 停止・manifest 不一致・
   `parity_fail_count>0` のみ。
4. 帰属分類（N ごと・fandhe-ai reuse 中央値）: `r = median_B / median_A`
   として、`|r − 1| ≤ 0.05` **または** `median_B` が A の 5 run
   min–max 範囲内 → 「負荷差（ノイズ帯）・構造分析と整合」。それ以外
   → 「構造分析と矛盾・原因未確定」（コード差確定ではない）。§16.4
   の run 内分散（N=1024 で 2.191–3.071 ms ≈ ±20%）を踏まえ 5% 帯単独
   では誤検知するため OR 条件とする。
5. §16 との差（A/§16・B/§16）は「セッション間の負荷ドリフト指標」
   として記録のみ（verdict を付けない）。
6. run の差し替え禁止。失敗時は数値を捏造せず `.failed-<ts>.jsonl`
   退避に従う。同 label の再実行は成果物退避後・新 label で行う。
7. 実行前に v0.8.0..HEAD の `--stat`（Metal 経路 4 パス）を再取得し、
   想定外の差分があれば帰属表を再導出する（規則 4 の帯域は変更しない）。

### 18.2 系列設計・帰属

- `run_gemm_gate.sh <device> <label>` は 1 起動で N=1024/2048/4096 の
  5 回計測を内部ループするため、run 単位の interleave は構造的に
  不可能。系列単位で A（対照 `0.8.0-ctrl-1521`）→ B（参考
  `head-<short sha>-1521`）の固定順に計測する。
- v0.8.0..origin/main の `--stat -- crates/backend-metal/src
  crates/facade/src crates/autodiff/src crates/tensor-core/src` は
  4 files・+583 −68（`gemm.rs`・`lib.rs`・`ops.rs`・`tile.rs`）。
  対応するコミットは `5b2d5060`（#1530）・`ef613b9b`（#1527）の
  2 コミット（`docs/perf/logs/metal-gemm-candle-gate-head-1521/
  diff_v0.8.0_origin-main_metal_path.txt`）。
- **構造分析（結論：計測前の期待値。`attribution.md` に事前登録）**:
  独立した 2 つの理由により、GEMM ゲート対象形状（NN 正方
  N=1024/2048/4096）は split-K 結線後も classic 経路のまま変わらない。
  1. 本番既定の `MetalGemm::new` は `split_k_auto_enabled =
     tile::SPLIT_K_DISPATCH_AUTO_PRODUCTION_ENABLED = false` のため、
     `dispatch_auto_with_route_impl` の `if self.split_k_auto_enabled
     && SPLIT_K_NUMERIC_CONTRACT_APPROVED` が偽となり split-K 分岐へ
     一切入らない（`SPLIT_K_NUMERIC_CONTRACT_APPROVED` が #1527 で
     `true` になった今も不変）。
  2. 対象形状自体が `tile::should_split_k` の並列度条件で `None`
     （`tile.rs::should_split_k_rejects_large_square_and_wide_shapes`
     が正方 512〜4096 を対象に回帰確認済み）。
  本 PR に含む Linux 実行可能テスト
  （`crates/backend-metal/tests/splitk_gemm_gate_shape_attribution.rs`）
  が両方を機械的に固定した（`print_attribution_table` 出力）:

  ```
  | shape | (m,n,k) | should_split_k | select_route_for_device |
  |---|---|---|---|
  | N=1024 | (1024,1024,1024) | should_split_k=None | select_route_for_device=Classic |
  | N=2048 | (2048,2048,2048) | should_split_k=None | select_route_for_device=Classic |
  | N=4096 | (4096,4096,4096) | should_split_k=None | select_route_for_device=Classic |
  ```

- readout 方式（`readout_uses_borrowed_view("metal", None)` は `false`。
  legacy 経路）は両系列とも不変。manifest で
  `readout_method=legacy-metal-1452` を両系列とも検証する。

### 18.3 プロトコル・共有負荷下の記録項目

- `docs/perf/logs/metal-gemm-candle-gate-head-1521/orchestrate_m4max.sh`
  （record_only。専有ゲートなし）が A→B の固定順で
  `run_gemm_gate_metal.sh` を実行する。
- 系列 A はビルドを計測から分離する（`cargo build --release -p
  bench-fandhe` → `cargo build --release -p bench-candle` を事前実行。
  #1490 §16.3 と同じ設計判断）。系列 B は `GEMM_GATE_PATCH_FACADE_PATH`
  経由のビルドと計測が不可分（#1166 の設計。プレビルドしない）。
- 10 秒間隔の uptime サンプラー・サーマル状態（前後）・watchlist
  プロセス件数を記録する（負荷を record_only で可視化するのみ。合否
  判定はしない）。
- 実行前に実行時点の v0.8.0..HEAD の `--stat`（Metal 経路）を
  再取得し、コミット済み diff（§18.2）と突き合わせる（規則 7）。

### 18.4 実測結果（未実測）

| N | A: fandhe-ai reuse 中央値（min–max, n=5） | A: candle fresh 中央値 | B: fandhe-ai reuse 中央値（min–max, n=5） | B: candle fresh 中央値 | B/A | 分類 |
|---|---|---|---|---|---|---|
| 1024 | 未実測 | 未実測 | 未実測 | 未実測 | 未実測 | 未実測 |
| 2048 | 未実測 | 未実測 | 未実測 | 未実測 | 未実測 | 未実測 |
| 4096 | 未実測 | 未実測 | 未実測 | 未実測 | 未実測 | 未実測 |

出典（記入予定）:
`scripts/bench/framework-compare/results/raw/results-m4max-gemm-gate-
0.8.0-ctrl-1521.jsonl`・`results-m4max-gemm-gate-head-<sha>-1521.jsonl`。

### 18.5 データ有効性（未実測）

- parity 0 fail（fandhe-ai・candle 両系列とも全 run）: 未確認
- manifest 4 条件（A: `fandhe_ai_source=registry`、B:
  `fandhe_ai_source=path:<facade>`、両系列とも
  `bench_fandhe_features=""`・`readout_method=legacy-metal-1452`・
  `candle_core_source=registry`）: 未確認

### 18.6 帰属表（§16 ↔ A ↔ B）

`docs/perf/logs/metal-gemm-candle-gate-head-1521/attribute.py` の出力を
転記する欄（未実測）。

| N | A中央値 | B中央値 | B/A | 分類 | candle/A | candle/B | §16参照値 | §16比 |
|---|---|---|---|---|---|---|---|---|
| 1024 | 未実測 | 未実測 | 未実測 | 未実測 | 未実測 | 未実測 | 0.743 | 未実測 |
| 2048 | 未実測 | 未実測 | 未実測 | 未実測 | 未実測 | 未実測 | 1.002 | 未実測 |
| 4096 | 未実測 | 未実測 | 未実測 | 未実測 | 未実測 | 未実測 | 0.634 | 未実測 |

### 18.7 #1037 ゲート判定はユーザー判断事項（読み替え可否）

**正式判定は §16.7 のまま不変**（#1037 は 3 形状中 2 形状が未達成の
まま）。本節は以下のみを整理し、判定自体は本 PR では行わない:

- 旧 #1037 の受け入れ条件は「N=1024/2048/4096 の**3 形状すべて**で
  reuse が candle を超える」というものだった。§16.4 は N=2048 のみ
  正式系列として初めて形状別条件を達成した（3 形状すべてではない）。
- この「3 形状すべて」条件を「形状ごとの個別条件」へ読み替えるか
  （読み替えた場合 N=2048 のみ達成という整理になる）は、旧 #1037 の
  受け入れ条件そのものの再定義に当たり、**spec（`docs/spec/`）の変更
  ではなくリポ側 issue の受け入れ条件の再定義**である。
- N=2048 の達成が再現するかは §16.8 のとおり未確認（run 間分散が
  大きく、専有環境での再計測なしに再現性を確定できない）。
- 上記を踏まえ、読み替え可否・#1037 系ツリーのクローズ可否は本 PR では
  決定しない（ユーザー判断事項として記録するのみ）。

### 18.8 スコープ外・引き継ぎ

- `SPLIT_K_DISPATCH_AUTO_PRODUCTION_ENABLED` の `true` 切替（#1515
  ADOPT 後の判断）とその後の A/B（#1517 が別途担当）
- §18.4〜18.6 の実機実測（Mac セッションへ引き継ぎ）
- `docs/performance-targets.md`・`scripts/bench/framework-compare/
  results/summary.md` への実測値反映（実測後・別途判断）
- 帰属表（§18.6）が「構造分析と矛盾・原因未確定」に分類された場合の
  原因調査（別イシュー起票をユーザーへ提案）
- 旧 #1037 受け入れ条件の読み替え（§18.7。ユーザー判断）

### 18.9 出典

- 実装（Linux 側スキャフォールド）: `docs/perf/logs/
  metal-gemm-candle-gate-head-1521/`（`orchestrate_m4max.sh`・
  `attribute.py`・`attribution.md`・`README.md`・`env_info.txt`・
  `diff_v0.8.0_origin-main_metal_path.txt`）
- 帰属テスト: `crates/backend-metal/tests/
  splitk_gemm_gate_shape_attribution.rs`
- v0.8.0 タグ時点の非到達根拠: `docs/perf/logs/
  metal-gemm-candle-gate-0.8.0-1490/attribution.md`
- split-K 本番結線の設計判断: `docs/backend-metal-splitk-decision.md`
