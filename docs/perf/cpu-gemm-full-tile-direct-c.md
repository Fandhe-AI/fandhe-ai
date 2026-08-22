# CPU GEMM 完全タイル C 直接ロード/ストア 計測記録（#557）

イシュー #557「perf(backend-cpu): 非端タイルで C への直接ロード/ストアに切り替える（コピー往復の削減）」の実測記録。

## 背景

`gemm_blis_region`（`crates/backend-cpu/src/gemm_blis/mod.rs`）は変更前、全ての MR×NR タイル（完全タイル・端タイルの両方）について `MAX_TILE` 固定長スタックバッファへ C の現在値をコピーしてからマイクロカーネルを呼び、結果を再度 C へコピーバックしていた。参照実装（matrixmultiply）は完全タイルをカーネルが C へ直接ロード/ストアし、端タイルのみマスク付きバッファ経由とする。本変更は `mr_eff == MR && nr_eff == NR` の完全タイルのみ C 直接経路へ切り替え、端タイルは既存のコピー方式（境界検査込み）を維持する。

## 変更内容

- [`Microkernel`](../../crates/backend-cpu/src/gemm_blis/microkernel.rs) trait に行ストライド `ldc` を受け取る新規メソッド [`run_with_ldc`](../../crates/backend-cpu/src/gemm_blis/microkernel.rs) を追加し、`c[i*ldc+j]`（`i in 0..MR`・`j in 0..NR`）のみを読み書きする契約へ一般化した（scalar／neon／avx2／avx512 の全マイクロカーネルが `run_with_ldc` を直接実装）。既存の公開メソッド [`Microkernel::run`](../../crates/backend-cpu/src/gemm_blis/microkernel.rs) はシグネチャ変更なしで維持しており（公開 API 非破壊。#691 レビュー指摘への対応）、組み込みカーネルの `run` は `run_with_ldc` へは委譲せず各カーネルの従来入口（`scalar::kernel`／`neon::kernel` 等）へ直接委譲する実装のまま維持している（`run` は `Result` を返せない必須メソッドのため、`run_with_ldc` が返す `Result<(), TileBoundsError>` を `run` 内部で `panic!` へ変換する経路を新設しないための設計判断。#691 レビュー P1 再指摘 `PRRT_kwDOTuUCJc6ZrQZG` 対応）。`run_with_ldc` はデフォルト実装（`ldc == NR` なら `run` へ委譲、それ以外はギャザー/スキャッタでフォールバック）を持つため、`run` のみを実装するクレート外部実装も無変更でコンパイル可能
- 完全タイル: `gemm_blis_region` が C の実バッファから `&mut c[row0..row0+(mr-1)*n+nr]`・`ldc=n` を渡す（コピーなし）
- 端タイル: 従来どおり `MAX_TILE` スタックバッファへコピーイン → `ldc=nr` でカーネル実行 → 有効部のみコピーバック
- 各カーネル入口の境界検査（`ldc < NR`・`(MR-1)*ldc` のオーバーフロー・`c` の実長不足）は `check_c_tile_bounds` が行い、契約違反時は `assert!`／`panic!` ではなく型付きエラー [`TileBoundsError`](../../crates/backend-cpu/src/gemm_blis/microkernel.rs) を `Result::Err` として呼び出し元へ返す（本番経路 panic 禁止。AGENTS.md・#691 レビュー P1 再指摘対応）。REQ-8 の境界検査自体は維持・強化しつつ、検出結果の通知方式を `assert!` から型付きエラーへ変更した

## コピー削減量（コード変更から導かれる確定事実）

完全タイル 1 回あたり、変更前は C タイルのコピーイン（`mr*nr` 要素）・コピーアウト（同）の計 `2*MR*NR` 要素の往復が発生していたが、変更後はゼロになる（直接ロード/ストア）。

M=N=K=2048（AVX2 カーネル MR=6・NR=16 の場合）を例に取ると、完全タイル反復回数は概ね `(M/MR)*(N/NR)*(K/KC)` のオーダー（端数分を除く）で、2048/6≈341・2048/16=128・2048/256=8 より概算 341*128*8 ≈ 349,184 回。1 回あたり `2*6*16=192` 要素のコピーが消滅するため、全体で概算 6,700 万要素ぶんのコピー往復が削減される（端タイルはこの概算から除く。実際の反復回数はブロック境界の端数処理により若干変動する）。

## 数値一致

`ldc` の導入で変わるのはロード/ストアのアドレッシング（`i*NR` → `i*ldc`）のみで、FMA 連鎖（p 昇順・レーン間縮約なし）・累積順序・丸めは一切変更していない。したがって `gemm_naive` との bit 完全一致契約（REQ-2）は成立し続ける。既存 parity テスト（`tests/gemm_blis_parity.rs`・`tests/gemm_epilogue_parity.rs`・`tests/fma_contract.rs`）は変更なしで全て green（`cargo test -p fandhe-ai-backend-cpu` 実測）。加えて、`ldc > NR` でも `ldc = NR` と bit 完全一致しギャップ列（隣接領域）を破壊しないことを検証する回帰テストを scalar／avx2／avx512／neon の各マイクロカーネルへ新設し、全タイルが完全タイル（直接経路のみ）となる形状（M=256, N=512, K=300）で `gemm_naive` と bit 完全一致することを確認するドライバレベルテストも追加した（いずれも green）。

## 計測環境

| 項目 | 値 |
|------|-----|
| CPU | QEMU Virtual CPU version 2.5+（`/proc/cpuinfo` 実測。物理ハードウェアではなく仮想化環境。複数エージェント並列実行中の共有ホスト） |
| 論理コア数 | 12（`nproc`） |
| OS | Linux 7.0.0-29-generic |
| rustc | 1.96.0 (ac68faa20 2026-05-25) |
| ビルド条件 | `RUSTFLAGS="-C target-feature=+avx2,+fma" cargo test -p fandhe-ai-backend-cpu --release`（`cpu-gemm-packing-buffer-reuse.md` と同一条件） |
| 計測プロトコル | `bench-harness::protocol::run`（warmup 20 回・計測 20 回・中央値/Q1/Q3 記録。`crates/backend-cpu/tests/gemm_blis_perf.rs`） |

## 再現コマンド

```bash
RUSTFLAGS="-C target-feature=+avx2,+fma" cargo test -p fandhe-ai-backend-cpu --release \
  -- --ignored gemm_blis_perf --nocapture
```

## 実測結果（変更前 origin/main `e962c41` vs 変更後 本 PR HEAD、同一環境・独立 5 回計測の中央値。#691 レビュー P1 再指摘 `PRRT_kwDOTuUCJc6ZsKRa` 対応で再計測）

各条件（変更前／変更後 × 各形状）を、`cargo test` プロセスを独立に 5 回起動して個別計測し（プロセス内の warmup 20 回・計測 20 回はそのまま。プロセスをまたいだ独立試行を 5 回積み重ねる二重の計測）、5 回分の `gemm_blis_parallel` median を中央値で集約した（`.claude/rules/coding-rust.md`「ベンチは 5 回計測の中央値を採用」準拠）。変更前は `git worktree add` で `origin/main e962c41`（本記録の従前版と同一コミット）を別ディレクトリへ checkout し、変更後と同一の `RUSTFLAGS`・同一コマンドで計測した。

| 形状（M=N=K） | 変更前 5 回の median (s) | 変更前 中央値 (s) | 変更後 5 回の median (s) | 変更後 中央値 (s) | 差分（中央値ベース） |
|---|---|---|---|---|---|
| 512 | 0.001446 / 0.000811 / 0.000765 / 0.000818 / 0.000775 | 0.000811 | 0.000764 / 0.000706 / 0.000738 / 0.000772 / 0.000755 | 0.000755 | -6.9% |
| 1024 | 0.014934 / 0.004755 / 0.004886 / 0.005016 / 0.004894 | 0.004894 | 0.004761 / 0.004775 / 0.004756 / 0.004718 / 0.004691 | 0.004756 | -2.8% |
| 2048 | 0.034863 / 0.028899 / 0.036017 / 0.030078 / 0.029210 | 0.030078 | 0.028935 / 0.027644 / 0.027058 / 0.032492 / 0.030951 | 0.028935 | -3.8% |

変更前の各系列は 1 回目（512: 0.001446s・1024: 0.014934s・2048: 0.034863s）が残る 4 回に比べ大きく外れているが、本環境（QEMU 仮想 CPU・複数エージェント並列実行中の共有ホスト。11 個の並行 worktree が同時稼働。実測は本文書「計測環境」節参照）の実際のノイズであり、恣意的な除外は行わず中央値の算出にそのまま含めた（外れ値を中央値が自然に減衰させる）。全 3 形状で変更後の中央値が変更前を下回り、コピー削減（上記「コピー削減量」節）と整合する方向の結果が得られた。

## 採否判断

計画 §7 の判断基準（(a) 改善またはノイズ幅内の差 → 採用／(b) 全形状で中央値が IQR を明確に超えて悪化 → 不採用）に照らし、全形状で変更後中央値が変更前中央値を下回った（-2.8%〜-6.9%）ため **(a) 採用**と判断した。コピー削減量自体はコード変更から直接導かれる確定事実（上記節）であり、独立 5 回計測でもその方向と整合する結果が得られている。より安定した計測環境（専有ハードウェア等）での再計測は行っていない。

NEON（aarch64 実機）の実測はローカル環境（x86_64）では実行不可のため、`cargo check --target aarch64-unknown-linux-gnu -p fandhe-ai-backend-cpu --all-targets` によるコンパイル検証に留め、実機ベンチは実機検証ツリー（#408 系）の枠組みへ委ねる。

## スコープ外（本イシューで対応しない事項）

- 端タイル向けマスク付き SIMD カーネル（matrixmultiply の `masked_kernel` 相当の SIMD 化）— 端タイルは既存コピー方式維持が本イシューの要件
- MR/NR/MC/KC/NC の再チューニング（#24 スコープ）・B packing のスレッド間共有（#556 で記録済みの既知事項）
- aarch64（NEON）実機での性能実測（実機検証の枠組みへ委ねる）
- より安定した計測環境（専有ハードウェア等）での再計測（`cpu-gemm-packing-buffer-reuse.md` と同じ理由でスコープ外）
