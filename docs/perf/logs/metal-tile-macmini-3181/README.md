# Mac mini の Metal GEMM タイル選択の追跡と全候補計測（イシュー #3181）

結論は `docs/perf/metal-gemm-tile-select-macmini.md`。本ディレクトリは生ログ一式。**診断のみ**（選択ロジック・tolerance・
split-K parity baseline・相手側ベンチのコードは不変）。規則の正は `RULE.txt`（計測前に固定）。

| パス | 内容 |
|---|---|
| `RULE.txt` | 事前登録規則 |
| `orchestrate_macmini.sh` | Mac mini 側の計測本体（ビルド → parity ゲート → 5 run × {trace・cand・candle}） |
| `aggregate.py` | 厳格な集計（標準ライブラリのみ。`python3 -I aggregate.py --self-test`／`python3 -I aggregate.py . --expect-rev <SHA> > aggregate.md`） |
| `trace_run{1..5}.log` | 本番 `dispatch_auto` の経路・タイル追跡（`TRACE`／`DEVICE` 行） |
| `cand_run{1..5}.log` | 全 11 候補の hash と純カーネル時間（`CAND` 行） |
| `candle_run{1..5}.jsonl`・`.err` | `bench-candle --task gemm-transfer-split`（`gemm_transfer_excl` を使用。`.err` は空） |
| `parity.log` | 全候補 × 5 N の CPU 参照 parity（前提ゲート） |
| `load_gate.log`・`env_info.txt`・`rev-stamp.txt`・`candle_tree.txt`・`build_*.log` | 負荷ゲート・機体情報・腕の同定・ビルドログ（パスは `<WORK>`／`<HOME>` にマスク。ホスト名は含めない） |
| `aggregate.md` | 機械集計 |

## 再現（Mac mini 上、展開ツリーの本ディレクトリから）

```sh
CARGO_TARGET_DIR=$HOME/work/target-fandhe-ai OUT=<出力ディレクトリ> sh orchestrate_macmini.sh
```

計測前コミット（`bc5b42bcf66ca32caa3c8bad4b1fdc392117660f`）を `git archive` して `.rev-stamp` を置き、rsync で転送した
（`docs/real-hardware-verification-env.md` §7.1）。ビルドと計測は並走させていない。

## 注意

- 開発機（M4 Max）での動作確認は追跡テストの smoke のみで、ここの値は Mini の実測だけ。
- `aggregate.py` は計測後に 1 点だけ直した（candle の JSONL のスキーマに合わせた）。値・判定には影響しない。
