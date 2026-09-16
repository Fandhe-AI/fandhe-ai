# Metal readout legacy 後退の 4 腕診断（イシュー #1696）実測ログ

`crates/backend-metal/src/readout_regression_diag_tests_1695.rs`（イシュー
#1695）の 4 腕診断ハーネスを M4 Max 実機で実行した結果の置き場。

**2026-09-16 実測済み**: M4 Max 実機で `orchestrate.sh` を実行し、生成物
（`aggregate.md`・主系列 60 ログ・副系列 3 ログ・`uptime_*`・
`pmset_therm_*`・`env_info.txt`）を本ディレクトリへ収めた（record_only・
共有負荷下。転記先は `docs/perf/metal-readout-legacy-regression-four-arm-
diag.md` §6・§7・§8・§10）。以下は実測前の記述をそのまま残す。

**現状（本 PR 時点）**: `orchestrate.sh`・本 README・
`env_info.txt.example`・`aggregate.py` は用意済みだが、実測ログ（生成物
自体）は未生成。本 PR を書いた実行環境に Apple Silicon 実機へのアクセス
経路がないため、下記の実測は未実施のまま記入欄として残す（`docs/perf/
metal-readout-legacy-regression-four-arm-diag.md` §6・§7・§8・§10 の記入欄
と対応）。Mac セッションで `orchestrate.sh` を実行し、生成物をこの
ディレクトリへ収める。

## 実行手順

```sh
cd docs/perf/logs/metal-readout-regression-diag-1696
sh orchestrate.sh
# ドライラン（コマンド列の確認のみ・実行しない）:
sh orchestrate.sh --dry-run
```

実行後:

```sh
python3 aggregate.py
# 自己検証（固定 fixture 文字列で集計ロジックを検証する。実測ログは不要）:
python3 aggregate.py --self-test
```

## 単一テスト名フィルタの注意（#1436 README の失敗モードを転記）

`--exact` はテストバイナリ内の完全修飾パス（モジュール名を含む）に対して
一致判定するため、モジュール名を省いた短いテスト名（例:
`readout_regression_diag_n1024`）を渡すと `--exact` の完全一致が成立せず
対象テストが 0 件のまま終了する（診断データが取得できない）。加えて、
モジュール名を省いた部分一致フィルターのまま `--exact` なしで実行すると、
同名 prefix を持つ単腕 4 関数（`readout_regression_diag_n1024_legacy_to_vec`
等）にも一致してしまい、4 腕をまとめて実行する
`readout_regression_diag_n1024` と単腕 4 関数の計 5 関数が同一プロセス内で
連続実行され、記録時（4 腕一括のみ、または単腕 1 関数のみ）と異なる
allocator 状態の引き継ぎが起きる。このため、モジュール名付き完全修飾名
（`readout_regression_diag_tests_1695::readout_regression_diag_n1024_
legacy_to_vec` 等）を `--exact` と併用する（`orchestrate.sh` は既にこの
形式でコマンドを組み立てている）。

## ファイル一覧

- `env_info.txt`: `uname -srm`・`sw_vers`・`rustc -V`・
  `sysctl machdep.cpu.brand_string`・`pagesize`（または
  `getconf PAGESIZE`）・実行コミットの `git rev-parse HEAD`（内部ホスト名は
  含めない）
- `uptime_before.txt`／`uptime_after.txt`: 実行前後の 1 回計測
- `uptime_sampler.log`: 実行中の 30 秒間隔 `uptime` サンプル（load average
  の推移を把握するための参考記録。`record_only` 運用のため専有ゲートには
  使わない）
- `pmset_therm_before.txt`／`pmset_therm_after.txt`: `pmset -g therm`
  （サーマルスロットリングの有無の参考記録）
- `layerB-n{1024,2048,4096}-{legacy_to_vec,borrowed_keep_alive,
  borrowed_with_dummy_alloc_free,pretouched_reused_dest}-run{1..5}.log`:
  主系列（単一腕・単一サイズ。5 プロセス起動）の実行ログ。計 12 テスト
  × 5 run = 60 ログ
- `inprocess-n{1024,2048,4096}-run1.log`: 副系列（in-process 4 腕。各 1
  起動）の実行ログ
- `aggregate.md`: `aggregate.py` が生成する集計表（腕×N の 5 起動中央値・
  起動間 spread・腕差・checksum 一致確認）

## 事前登録判定規則

`docs/perf/metal-readout-legacy-regression-four-arm-diag.md` §5 を正とし、
本 README では二重管理しない。実測完了後は同ドキュメント §6・§7・§8・§10
の記入欄へ転記する。

## 関連

- `docs/perf/metal-readout-legacy-regression-four-arm-diag.md`（本
  イシューの記録本文・事前登録判定規則）
- `crates/backend-metal/src/readout_regression_diag_arms.rs`・
  `readout_regression_diag_tests_1695.rs`（実装。イシュー #1695）
- `docs/perf/logs/cuda-host-view-readout-regression-1436/`（CUDA 側同型
  診断のログ置き場。同じ構成方針）
- 親イシュー #1574 → 本イシュー #1696（兄弟イシュー #1695 がハーネス実装）
