# CPU elementwise 融合効果の計測記録（#167・TASK-12.2a）

イシュー [#167](https://github.com/Fandhe-AI/fandhe-ai/issues/167)「test(fusion): TASK-12.2a 融合効果の実測（連鎖・fan-out・transpose 混在）」の実測記録。
受け入れ条件「実測記録が残されている」に対応する。`docs/kernel-fusion.md` §5 が本記録を実測正本として参照する。

## 0-a. 実測値の再取得（RNG 二重スケーリング修正、Bugbot 指摘対応）

初版計測時、`seeded_tensor_unit_range`（`crates/backend-cpu/tests/
fusion_effect_perf.rs`）が `Xorshift64Star::next_f32()`（既に `[-1, 1)` を
返す仕様。`crates/bench-harness/src/rng.rs` 参照）へさらに `* 2.0 - 1.0`
を適用しており、実際の入力範囲が意図した `[-1, 1)` ではなく `[-3, 1)`
になっていた（Cursor Bugbot 指摘、PR #405）。二重スケーリングを除去した
うえで全パターンを 5 回再計測し、本記録の §4〜§6 を実測値・中央値方式
（`coding-rust.md`「5 回計測の中央値」準拠）で更新した。連鎖・fan-out・
`ew_2d_contig` の傾向・速度比は初版と有意差なし。`ew_transpose` のみ、
単発計測では 0.707x（融合側が悪化）だったが、5 回中央値では 0.981x
（ほぼ互角）に変化した——後述 §5 参照。

## 0. 前提: `CpuBackendOps::run_fused` の結線（本イシューで実施）

TASK-12.1 系列は融合 IR（#163・PR #400）・CPU 単一パス融合カーネル
（#164・PR #403）ともにマージ済みだったが、#400 は「CPU 融合実行への
結線は #164 のスコープ」、#403 は「`run_fused` オーバーライドの提供元は
backend-cpu 側（#163 のスコープ）」としており、双方が相手に委ねた結果
`CpuBackendOps`（`crates/backend-cpu/src/ops.rs`）は `BackendOps::run_fused`
をオーバーライドしておらず、デフォルト実装（`Unsupported` fail-safe）の
まま融合カーネルが一度も起動しない状態だった。この状態では融合条件と
非融合条件が区別不能で「融合効果の実測」が構造的に不可能なため、本
イシューの前提ステップとして `CpuBackendOps::run_fused` を
`fused_elementwise::run_fused_elementwise` へ結線した（`crates/backend-cpu/src/ops.rs`
の `run_fused` オーバーライド。数行の委譲のみで新規設計を含まない）。

結線の直接検証は `crates/backend-cpu/tests/backend_ops_integration.rs`
`cpu_run_fused_via_backend_ops_is_wired_and_matches_sequential_composition`
（`&dyn BackendOps` 経由で `run_fused` が `Ok` を返すこと・per-op 逐次
合成との数値一致を固定する非 `#[ignore]` テスト）。

## 1. 計測環境

| 項目 | 値 |
|------|-----|
| CPU | QEMU Virtual CPU version 2.5+（`/proc/cpuinfo` 実測。物理ハードウェアではなく仮想化環境） |
| 論理コア数 | 12（`nproc`） |
| OS | Linux 7.0.0-28-generic |
| rustc | 1.96.0 (ac68faa20 2026-05-25) |
| ビルド条件 | `RUSTFLAGS="-C target-feature=+avx2,+fma" cargo test -p fandhe-ai-backend-cpu --release --test fusion_effect_perf -- --ignored --nocapture` |
| 計測プロトコル | `bench-harness::protocol::run`（warmup 20 回・計測 20 回・中央値/Q1/Q3 記録。TASK-8.1 準拠）を **5 プロセス独立実行し、パターンごとに速度比の中央値を採用**（coding-rust.md「5 回計測の中央値」準拠。§0-a の再計測から） |
| 計測バイナリ | `crates/backend-cpu/tests/fusion_effect_perf.rs`（`#[ignore]` 分離） |
| 比較対象（非融合） | `NonFusedCpuOps`（同ファイル定義）——`CpuBackendOps` の全 per-op メソッドへ委譲しつつ `run_fused` はオーバーライドせずデフォルト `Unsupported` のまま残すラッパー。`autodiff::Tape` が per-op フォールバックへ倒れる |
| 比較対象（融合） | `CpuBackendOps`（`run_fused` オーバーライド込み。本イシューで結線）。`autodiff::Tape` の遅延評価 2 層（`push_lazy`／`materialize_*`）が 4 段以上の elementwise 連鎖を `run_fused` へ回す |
| 実行方式 | `autodiff::Tape::new_with_ops(ops)` → `Var` 連鎖 → `to_tensor()` 実体化を 1 イテレーション内で完結（両条件で同一構造。CPU 同期は閉包リターンで完了） |
| 入力 | `bench_harness::rng::Xorshift64Star`（決定的シード）で `[-1, 1)` へスケールした一様分布（`exp`／`tanh` のオーバーフロー回避） |

**サイズについての注記**: 計画（実装計画 §3）は PoC-9 主サイズ（N=4e7、f32 160MB/テンソル）を主サイズとして想定していたが、共有ホスト（QEMU 仮想 CPU・複数エージェント並列実行中）での実行時間を抑えるため、計画が許容する縮小サイズ N=1e7（f32 40MB/テンソル）を主サイズとして採用した。2D パターンは D=2048（計画どおり）。

## 2. 再現コマンド

```bash
RUSTFLAGS="-C target-feature=+avx2,+fma" cargo test -p fandhe-ai-backend-cpu --release \
  --test fusion_effect_perf -- --ignored --nocapture
```

## 3. ワークロード定義

v2 の遅延対象 5 演算（`add`／`mul`／`relu`／`exp`／`tanh`）で構成する。PoC-9 の `ew4` は sigmoid を含むが、sigmoid は v2 では遅延対象外の組込み複合演算（`docs/kernel-fusion.md` 限界表）のため tanh へ置換した。

| # | パターン | 内容 | PoC-9 対応 |
|---|---------|------|------|
| 1 | `ew4`（連鎖 4 段） | `add → mul → tanh → mul` | `ew4`（sigmoid→tanh 置換） |
| 2 | `ew6`（連鎖 6 段） | `add → mul → tanh → mul → add → tanh` | `ew6` |
| 3 | `ew_fanout` | `a = x + y; b = a * a; c = b + x; tanh(c)`（中間 `a`・葉 `x` を 2 回消費） | `ew_fanout` |
| 4 | `ew_transpose`（transpose 混在） | 2D `D×D`: `ew4` 相当の 4 段連鎖を、葉 `x` を転置 view（非 contiguous）として実行 | `ew_reshape` 対応（挙動は大きく異なる。§5 参照） |
| 5 | `ew_2d_contig`（対照） | #4 と同一連鎖を contiguous 葉で実行（transpose の影響を分離する対照条件） | — |

サニティ検査（融合条件・非融合条件の最終値を REQ-2 統一複合判定「相対誤差 1e-3 未満 または 絶対誤差 1e-5 未満」で照合）を全パターンで実行し、いずれも一致を確認した（許容誤差は変更していない）。

## 4. 実測結果（5 回計測の中央値。§0-a 参照）

5 プロセス独立実行それぞれの median/Q1/Q3・速度比を計測し、パターンごとに速度比が中央値となった回の median/Q1/Q3 を代表値として採録する。

| パターン | 非融合 median (s) | 非融合 Q1〜Q3 (s) | 融合 median (s) | 融合 Q1〜Q3 (s) | 速度比 5 回分（非融合/融合） | 速度比中央値 |
|---|---|---|---|---|---|---|
| `ew4`（N=1e7） | 0.019340 | 0.018886〜0.019611 | 0.012724 | 0.012438〜0.013395 | 1.489, 1.498, **1.520**, 1.580, 1.587 | **1.520x** |
| `ew6`（N=1e7） | 0.030885 | 0.030141〜0.032324 | 0.019659 | 0.019256〜0.019806 | 1.444, 1.569, **1.571**, 1.581, 1.604 | **1.571x** |
| `ew_fanout`（N=1e7） | 0.019431 | 0.018625〜0.019757 | 0.012699 | 0.012400〜0.012972 | 1.488, 1.497, **1.530**, 1.572, 1.602 | **1.530x** |
| `ew_transpose`（D=2048、転置葉） | 0.068826 | 0.061556〜0.074268 | 0.070162 | 0.062397〜0.073074 | 0.865, 0.875, **0.981**, 1.271, 1.581 | **0.981x（ほぼ互角）** |
| `ew_2d_contig`（D=2048、対照） | 0.006963 | 0.006740〜0.007220 | 0.004538 | 0.004448〜0.004655 | 1.396, 1.426, **1.534**, 1.537, 1.557 | **1.534x** |

`ew4`／`ew6`／`ew_fanout` はいずれも 5 回とも融合側が高速（速度比中央値 1.52〜1.57 倍、5 回のレンジも常に 1.0 超）であり、連鎖・fan-out で融合が有意に効くという PoC-9 の定性的傾向（v1 実測 2.25〜3.19 倍。`docs/kernel-fusion.md` §3「v1 PoC-9 実測の位置づけ」）と整合する（数値の直接比較はしない。v1 は Metal 実機・Burn/CubeCL 前提の参考値であり v2 の保証値ではない）。

`ew_2d_contig`（対照条件）も速度比中央値 1.53 倍の改善が確認でき、transpose の有無が唯一の差分であることを裏付ける。

`ew_transpose` は 5 回中央値では 0.981x（ほぼ互角）だが、5 回のレンジが 0.865〜1.581x と 1.0 を挟んで大きくばらつく（詳細は §5）。

## 5. transpose 混在の挙動（`ew_transpose`）

計画（実装計画 §3 #4）は「速度比 ≈ 1.0（フォールバック）」を予測していた。RNG 二重スケーリング修正後の 5 回計測では、速度比が 0.865／0.875／0.981／1.271／1.581x と **1.0 を挟んで大きくばらつき**、中央値は 0.981x（ほぼ互角）——当初予測どおりの結果になった（§0-a）。数値は全 5 回とも両条件で REQ-2 複合判定を満たし一致している（サニティ検査で確認済み）ため、遅延評価そのものの誤りではない。

`ew_transpose` の Q1〜Q3 幅は `ew_2d_contig` より 1 桁以上広く（例: 中央値回で融合側 0.062397〜0.073074s に対し `ew_2d_contig` は 0.004448〜0.004655s）、共有ホスト（QEMU 仮想 CPU・複数エージェント並列実行中）のノイズの影響を強く受けており、単発計測では容易に 0.7x 台〜1.6x 台まで振れる。このばらつきの大きさ自体が、5 回中央値を採る必要性を裏付けている。

理論上のオーバーヘッド要因（実測が示す差の説明候補であり、5 回のレンジの広さからみて支配的とは断定しない）:

- `run_fused_elementwise`（`crates/backend-cpu/src/fused_elementwise.rs`）は非 contiguous な葉（転置 view）を検出すると `BackendError::Unsupported` を返し、`autodiff::tape` の materialize 層が per-op フォールバックへ再実行する（`docs/kernel-fusion.md` 限界表 4 行目の設計どおりの動作）。
- **融合条件は「`run_fused` を試みて `Unsupported` で失敗 → per-op フォールバックを実行」という 2 段の経路を踏む**のに対し、非融合条件（`NonFusedCpuOps`）は最初から `run_fused` を呼ばず直接 per-op 経路を実行する。融合条件は失敗した `run_fused` 呼び出し分のオーバーヘッド（`FusionPlan` の走査・`leaf.as_slice()` の non-contiguous 判定・エラー構築）を追加で払うはずだが、5 回中 2 回は融合側が非融合側より速い結果も観測されており（1.271x・1.581x）、この理論上のオーバーヘッドは QEMU 仮想 CPU のノイズに埋もれる程度の大きさである可能性が高い。

**結論**: 「transpose 混在連鎖では融合が有意な性能改善をもたらさない」という判断サマリ（`docs/kernel-fusion.md` §1 (e)）自体は実測で裏付けられた（融合は速度改善の根拠にならない）。ただし当初のコード不具合（RNG 二重スケーリング）に基づく単発計測が示していた「融合側が明確に悪化する（0.707x）」という結論は、5 回計測の中央値では再現せず、当初計画どおりの「速度比 ≈ 1.0（ほぼ互角、ノイズ幅内）」に訂正する。この知見は `docs/kernel-fusion.md` の限界表（表 4 行目）へ反映する。

## 6. PoC-9（v1）参考値との対比

v1（Metal 実機・Burn/CubeCL 前提）と v2（CPU 自作・本記録）は実行系が異なるため、速度比の**定性的傾向のみ**を比較し数値の直接比較はしない（`docs/kernel-fusion.md` の「参考値として扱う」方針に整合）。

| パターン | v1 速度比（PoC-9・Metal 実機） | v2 速度比中央値（本記録・CPU） | 定性的傾向の一致 |
|---|---|---|---|
| `ew4` | 2.25 倍 | 1.520 倍 | 一致（融合が有意に効く） |
| `ew6` | 2.61 倍 | 1.571 倍 | 一致（融合が有意に効く） |
| `ew_fanout` | 3.19 倍 | 1.530 倍 | 一致（融合が有意に効く） |
| `ew_reshape`（transpose 混在。v1 は融合有効時にメタデータ変換として取り込む） | 13.89 倍 | 0.981 倍（ほぼ互角、5 回のレンジ 0.865〜1.581） | **不一致**（v1 はストライド付きビューを融合セグメント内で扱う機構を持つため、v2 のような「融合を試みて拒否されフォールバックする」オーバーヘッドが発生しない構造上の違い。v2 の融合 IR は現時点で非 contiguous 葉を拒否する設計のため速度上の恩恵はないが、§5 のとおり明確な悪化でもない。`docs/kernel-fusion.md` §3 に記載済みの既知の受け入れコスト） |

`ew4`／`ew6`／`ew_fanout` は v1・v2 とも融合が有意に効くという傾向が一致する（絶対値は実行系の違いにより異なる）。`ew_reshape`／`ew_transpose` は v1・v2 で挙動が明確に異なり、v2 は§5 で述べたとおり v1 の性能水準（13.89 倍の改善）を達成しない（`docs/kernel-fusion.md` §3 で既知の受け入れコストとして明示済みの制約が実測で確認された）。ただし v2 側の速度比自体は「融合側が悪化する」のではなく「ほぼ互角（ノイズ幅内）」である点に注意（§5 参照）。

## 7. 数値一致

全パターンで融合条件・非融合条件の最終値が REQ-2 統一複合判定（相対誤差 1e-3 未満 または 絶対誤差 1e-5 未満）を満たすことを確認した（`crates/backend-cpu/tests/fusion_effect_perf.rs` 内のサニティ検査。`backend_cpu::parity::assert_parity` を使用。許容誤差は変更していない）。

## 8. スコープ外（本記録で対応しない事項）

- **GPU（CUDA NVRTC・Metal MSL）での融合実測**: 両バックエンドとも GEMM カーネルのみ実装済みで elementwise（`add`/`relu` 等）が未実装のため `run_fused` は到達しない。GPU 融合実測は実機（DGX Spark GB10・Metal 実機）検証が前提であり、`docs/kernel-fusion.md` §6「将来拡張・スコープ外」に既存スコープ外事項として記録済み（新規 Issue 起票はユーザー承認事項のため本イシューでは行わない）。
- **連鎖長 4〜6 段の上限に達した場合の実体化・再開挙動の実測**: 本記録の全パターンは想定連鎖長（6 段）以内に収まる。なお上限適用自体が `crates/autodiff/src/tape.rs::build_lazy_plan`（ライブ経路）では未実装であることが判明した（`docs/kernel-fusion.md` §3 表 5 行目、実装追跡は #404）。上限到達時の実体化・再開挙動は現状発生し得ない（無制限に連鎖が伸びる）ため、本記録では未計測。
- **`ew_reduce`（連鎖 + reduction 境界）の計測**: 実装計画（§3 #6）で「時間が許せば実施」とした補助パターン。共有ホストでの実行時間を抑える判断（§1 サイズについての注記）と合わせ、本イシューでは計測を見送った。reduction エピローグが融合対象外であること自体は `docs/kernel-fusion.md` 限界表 1 行目の設計事項として既に記録済みであり、本記録が新たに検証すべき論点ではないと判断した。
