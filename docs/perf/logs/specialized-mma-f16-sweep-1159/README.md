# イシュー #1159 GB10 sweep 証跡

`specialized_mma_parity.rs::specialized_mma_f16_matches_default_and_reference_across_shapes`
は `check_cpu_reference=true` の形状で `fandhe_ai_backend_cpu::assert_parity`
（厳密ゼロ fail 判定・fail で panic）を呼ぶため、1 形状でも FAIL すると
そこでテストが止まる。#1134 の GB10 sweep（main `1a32082`）で
`(256,512,1024)` の `DYNAMIC_ALL` が FAIL し、残りの (形状, プリセット)
が未評価のまま残っていた。本ディレクトリは、`assert_parity` の panic を
`catch_unwind` で捕捉して継続評価する一時診断テスト
（`crates/backend-cuda/tests/specialized_mma_f16_sweep_1159.rs`。当初は
**git 管理外**で実行し、実行ログのみを本ディレクトリへ証跡として残した
まま破棄していた）を GB10 実機で 2 回実行した証跡である。

**追記（codex-review P2 対応。本 PR 内）**: 上記の一時診断テストは
ソースが保存されておらず、再現手順がイシュー #1159 のコメントのみを
参照していたため、コメントへアクセスできない環境では再現・レビューが
できない状態だった。同名パス
`crates/backend-cuda/tests/specialized_mma_f16_sweep_1159.rs` へ、
`assert_parity` の代わりに（panic しない `Result` 版の）
`fandhe_ai_backend_cpu::compare` を直接呼ぶ形で機能的に同等な診断テストを
`#[ignore]` 付きで追加し git 管理下に置いた。`specialized_mma_parity.rs`
の形状表・生成規則・複合判定は変更せず再利用しているため、同一環境で
再実行すれば同一の判定が再現される想定である（追加した診断テスト自身は
未実測。既存の `assert_parity` ベースの `specialized_mma_parity.rs` 本体
・tolerance 定数・カーネル・`ParityBaseline` は本 PR で一切変更していない
点は変わらない）。

tolerance 定数（`RELATIVE_TOLERANCE`/`ABSOLUTE_RESCUE_THRESHOLD`）・
カーネル・`specialized_mma_parity.rs` 本体・`ParityBaseline` は本 PR で
一切変更していない（上記の診断テスト追加・`Cargo.toml` の `[[test]]`
登録を除き、`git diff --stat origin/main...HEAD -- '*.rs' '*.toml'` に
差分がないことを検証条件とする）。

## ファイル一覧

| ファイル | 内容 |
|---|---|
| `env_info.txt` | 計測環境（GPU・driver・CUDA・rustc/cargo・転送元コミット・実行コマンド・GPU 排他性確認結果） |
| `original_test_before.log` | ブランチ HEAD での元テスト（`specialized_mma_f16_matches_default_and_reference_across_shapes`）の実行ログ。before 状態の記録 |
| `sweep_run1.log`・`sweep_run2.log` | 一時診断テストの 2 回分の生ログ（`--nocapture`）。`TRIAGE_ROW`/`SWEEP_BITMATCH`/`SWEEP_SUMMARY` 行を diff し完全一致を確認済み |

## before 状態（元テストの再現確認）

`original_test_before.log` は `(256,512,1024)` の `DYNAMIC_ALL` で
複合判定 FAIL し panic することを示す:

```
fail_count=30/131072 max_abs_diff=1.562e-2 max_rel_err=3.341e-2 mean_abs_diff=1.011e-5
```

#1134 記録値（`fail_count=30/131072`・`max_abs_diff=1.562e-2`・
`max_rel_err=3.341e-2`・`mean_abs_diff=1.011e-5`）と完全一致。

## 厳密ゼロ fail が成立する (形状, プリセット)（21 組）

`check_cpu_reference=true` の 8 形状のうち 7 形状 × 3 プリセット:

- `(64,128,32)` × {DYNAMIC_ALL, STATIC_NK, STATIC_MNK}
- `(128,256,128)` × {DYNAMIC_ALL, STATIC_NK, STATIC_MNK}
- `(40,24,72)` × {DYNAMIC_ALL, STATIC_NK, STATIC_MNK}
- `(65,136,40)` × {DYNAMIC_ALL, STATIC_NK, STATIC_MNK}
- `(63,120,24)` × {DYNAMIC_ALL, STATIC_NK, STATIC_MNK}
- `(200,264,104)` × {DYNAMIC_ALL, STATIC_NK, STATIC_MNK}
- `(1,136,40)` × {DYNAMIC_ALL, STATIC_NK, STATIC_MNK}

## 厳密ゼロ fail が成立しない (形状, プリセット)（3 組）

- `(256,512,1024)` × {DYNAMIC_ALL, STATIC_NK, STATIC_MNK}
  （全プリセットで統計完全一致: `fail_count=30/131072`・
  `max_abs_diff=1.562500e-2`・`max_rel_err=3.341149e-2`・
  `mean_abs_diff=1.010961e-5`・`max_fail_abs_diff=3.051758e-5`・
  `p999_abs_diff=3.906250e-3`）

## bit 一致検査（全 10 形状 × 3 プリセット = 30 行、CPU 参照の有無に関わらず）

全 30 行 `bit_match=true`（`4096x4096x4096`・`512x64x4096`〈CPU 参照
なし・bit 一致検査のみ対象〉を含む）。特化カーネルは既定カーネルと
演算命令列・アキュムレート順序を変えない契約（`kernels_mma.rs` 冒頭
コメント）が本 sweep でも成立していることを確認した。

`(256,512,1024)` の 3 プリセットで CPU 参照との複合判定統計が完全に
同一（上記）なのは、bit 一致契約により比較対象バイト列が 3 プリセット
で同一だからである。したがって本形状の判定は実質「1 通り
（DYNAMIC_ALL 相当）」であり、プリセット間の統計差異は観測されなかった
（bit 一致契約の破れは観測されなかった）。

## 再現性

`sweep_run1.log`・`sweep_run2.log` の `TRIAGE_ROW`/`SWEEP_BITMATCH`/
`SWEEP_SUMMARY` 行（各 55 行）を `diff` し、完全一致（差分ゼロ）を確認
した。2 回とも `.rev-stamp`（転送元コミット）は
`c72d6649f46d5348b2be56c694d9572273e91493` のまま変化しておらず、他
セッションによる書き換えが計測ウィンドウ中に発生していないことも確認
した。

## 再現手順（同じ tolerance・カーネルでの再実行方法）

上記の一時診断テストの実行ログそのもの（`TRIAGE_ROW`/`SWEEP_BITMATCH`
行）は `docs/perf/logs/specialized-mma-f16-sweep-1159/` 掲載のログ本文
（`sweep_run1.log`・`sweep_run2.log`）を一次証跡として参照する。

同一環境（GB10・同一 tolerance 定数・同一カーネル）での再実行は、本 PR
で追加した `crates/backend-cuda/tests/specialized_mma_f16_sweep_1159.rs`
（`#[ignore]`・`internal-diagnostics` feature 必須。`SWEEP_BITMATCH`/
`SWEEP_ROW`/`SWEEP_SUMMARY` 行を出力する）で行う:

```bash
cargo test -p fandhe-ai-backend-cuda --test specialized_mma_f16_sweep_1159 \
  --all-features -- --ignored --nocapture
```

前身の一時診断テスト（本ファイルと同名パスで実行したが git 管理外の
まま破棄された版）のソース全文は「追記」節のとおり保存されていない。

## 申し送り（スコープ外・本 PR では対応しない）

- `ParityBaseline` への baseline 行追加（`(256,512,1024)` の 3 プリセット
  分）→ #1161（ユーザー承認必須）
- `docs/perf/cuda-parity-baseline.md` への正式記録追記 → #1162
- `(256,512,1024)` の f16 丸め由来／機能欠陥の切り分け（#1155 の実機
  切り分けは PR #1180 の時点で未実施と明記されている）→ 親 #1134 の
  議論に委ねる
