# GPU `run_fused` elementwise allowlist 融合（区分 B-1・イシュー #2085）

## 背景

`docs/autodiff-graph-optimization-scope-decision.md` §5 区分 B-1（GPU
`run_fused` の elementwise allowlist）を実装した。CPU 側
（`backend-cpu::fused_elementwise::run_fused_elementwise`）は既に
`Input`／`Add`／`Mul`／`Relu`／`Exp`／`Tanh` の連鎖（最大 6 段。
`MAX_FUSED_CHAIN_LEN`）を単一パスで実行するが、CUDA・Metal の
`run_fused` は canonical RMSNorm／softmax プラン一致時のみ専用カーネル
へルーティングし、それ以外の elementwise-only プランは
`Unsupported` として呼び出し元の per-op フォールバック（`Tape::
materialize_fallible`）へ落ちていた。GPU では per-op ごとに
H2D→起動→D2H が発生するため、連鎖長 N なら N 往復が生じる。

F5（`docs/autodiff-graph-optimization-scope-decision.md` §2 F5）の
ホスト葉 I/O 固定費により、デバイス常駐化なしでは効果が限定的である
見込みを実装前に明記していた（REJECT を許容する事前登録。#1583 先例）。

## 設計

設計判断・数値契約・opt-in ゲートの詳細は
`docs/autodiff-graph-optimization-scope-decision.md`「#2085 追補」を正
とする（本 doc では重複記述しない）。要点のみ再掲する:

- **allowlist**（denylist 化しない）: `Input`／`Add`／`Mul`／`Relu`／
  `Exp`／`Tanh` のみ受理。CPU 版と同一。
- **既定 OFF の opt-in ゲート**: `backend-cuda::fused_elementwise::
  set_gpu_elementwise_fusion_enabled`／`backend-metal::fused_elementwise::
  set_gpu_elementwise_fusion_enabled`（各クレート内 `pub`。`facade` への
  再公開は対象外）。
- **数値契約**: 同一バックエンドの per-op 経路・CPU 融合カーネルと bit
  完全一致を目標とする。CUDA は非縮約 intrinsic（`__fadd_rn`／
  `__fmul_rn`）で FMA 縮約を遮断。Metal は既存の `MathMode::Safe` に
  依拠し、新規 pragma は追加しない。
- **キャッシュ上限**: プランごとに動的コンパイルされるカーネルの
  プロセス内キャッシュに 256 エントリの上限を設け、上限到達時は
  `Unsupported`（per-op フォールバック）。

## 事前登録判定規則（実測前コミット時点で固定）

- **比較腕**: 同一バイナリ。before = ゲート OFF、after = ゲート ON。
  OFF 経路は本 PR 導入前と atomic load 1 回を除き同一挙動
  （`Unsupported` → per-op）のため、同一バイナリ A/B が成立する。
- **ベンチ**: `crates/facade/tests/gpu_elementwise_fusion_bench.rs`
  （`#[ignore]`・`FANDHE_BENCH_DEVICE=cpu|cuda|metal`・
  `FANDHE_BENCH_GPU_EW_FUSION=0|1`）。ケース: (A) 4 段連鎖
  `add→relu→exp→tanh`、(B) 6 段連鎖、(C) fan-out、(D) fan-in、(E) matmul
  境界を挟む 2 連鎖。`numel ∈ {16384, 65536, 1048576}`。
- **run**: 各腕 5 プロセス起動、run 単位で before/after の起動順を反転。
  record_only（専有ゲートなし。#1583 と同方針）。
- **判定**: バックエンドごとに全セル `ratio(after/before) <= 1.00` かつ
  出力・勾配の checksum（`to_bits()` 折り込み）完全一致 → ADOPT；1 セル
  でも `ratio > 1.00` または checksum 不一致 → REJECT（既定 OFF 維持・
  機構は保持）。cpu は control（ゲート無関係。ratio≈1.00 の確認のみ）。
  事後のセル除外・run 追加・規則緩和は行わない。F5 により REJECT は
  事前登録済みの許容結果である。
- **出荷規則**: ADOPT のバックエンドでも既定 ON 化はユーザー承認事項
  （`docs/autodiff-graph-optimization-scope-decision.md`「#2085 追補」）。
  承認前は既定 OFF のまま。

## 正しさの検証（性能とは独立。Linux〈CI〉で実行可能）

- allowlist matcher・ソース証跡（非縮約 intrinsic・`metal::precise::
  exp`／`tanh`・境界ガード・引数個数）: `crates/backend-cuda/src/
  {fused_elementwise.rs, kernels_fused_elementwise.rs}`・
  `crates/backend-metal/src/{fused_elementwise.rs,
  fused_elementwise_source.rs}` の `#[cfg(test)]`。
- ゲート既定 OFF・OFF 時のデバイス非依存 `Unsupported`:
  `crates/backend-cuda/tests/fused_elementwise_parity.rs::
  fused_elementwise_gate_off_is_unsupported_without_device_access`・
  `crates/backend-metal/tests/fused_elementwise_parity.rs::
  fused_elementwise_gate_off_is_unsupported`（後者は `#![cfg(target_os =
  "macos")]` のため Linux では `cargo check --target aarch64-apple-darwin`
  でのみ型検査）。
- ホスト逐語モデル対 CPU 融合カーネルの bit 一致・勾配の bit 完全一致
  （`Tape::new_with_ops` フィクスチャ経由）:
  `crates/facade/tests/gpu_elementwise_fusion_grad_bit_identity.rs`
  （`forward_output_matches_bit_exact_between_host_model_and_cpu_fused_kernel`・
  `gradient_matches_bit_exact_between_host_model_and_cpu_fused_kernel`。
  いずれも pass 済み）。

## 実測結果

**未実測**（本エージェント実行環境に CUDA〈DGX Spark GB10〉・Metal
〈Apple Silicon〉実機への到達手段がなく、`#[ignore]` 実機テスト・A/B
ベンチは未実行のまま Mac／GB10 セッションへ申し送る）。実行コマンド・
記入欄は `docs/perf/logs/gpu-elementwise-fusion-2085/README.md` を参照。

## verdict

**保留（未実測のため判定不能）**。実測後、上記事前登録規則に従って
ADOPT／REJECT を判定し、本節を更新する。

## スコープ外・後続への引き継ぎ

- `facade` への opt-in setter 追加（公開面拡張。ユーザー承認事項）。
- ADOPT 時の既定 ON 化（ユーザー承認事項）。
- 区分 B-2 以降（`Sigmoid` 等 allowlist 拡張・XLA 相当のクロス演算
  融合）。
- CUDA／Metal 実機での A/B 計測・parity 検証（本 doc「実測結果」節）。
