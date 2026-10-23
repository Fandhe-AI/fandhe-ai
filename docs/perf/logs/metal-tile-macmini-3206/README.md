# Mac mini 向けタイル表の追加前後と candle 比（イシュー #3206）

結論は `docs/perf/metal-gemm-tile-select-macmini.md` §9。本ディレクトリは生ログ一式。**診断・記録のみ**（`crates/`・tolerance・
split-K parity baseline・相手側ベンチのコードは不変）。規則の正は `RULE.txt`（計測前に固定）。

| パス | 内容 |
|---|---|
| `RULE.txt` | 事前登録規則 |
| `orchestrate_macmini.sh` | Mac mini 側の計測本体（ビルド → 前提ゲート 2 本 → 5 run × {select・bit・ab・candle}） |
| `aggregate.py` | 厳格な集計（標準ライブラリのみ。`python3 -I aggregate.py --self-test`／`python3 -I aggregate.py . --expect-rev <SHA> > aggregate.md`） |
| `mini_run.log` | PR #3525 時点の 1 プロセス記録（参考。正式系列ではない） |
| `select_run{1..5}.log` | 本番経路が選ぶタイルの証明（`SELECT` 行） |
| `bit_run{1..5}.log`・`bitgate.log` | before／after の bit 一致と CPU 参照 parity（`BITMATCH` 行）。`bitgate.log` は系列前の前提ゲート |
| `ab_run{1..5}.log` | 腕別の kernel_gpu／壁時計合計の中央値（`AB` 行） |
| `candle_run{1..5}.jsonl`・`.err` | `bench-candle --task gemm-transfer-split`（`gemm_transfer_excl` を使用） |
| `precheck.log` | `tile::` 単体テスト（AC2 の前提ゲート） |
| `load_gate.log`・`env_info.txt`・`rev-stamp.txt`・`tree_fingerprint.txt`・`binaries.sha256`・`candle_tree.txt`・`build_*.log` | 負荷ゲート・機体情報・腕の同定（ツリー指紋・実行バイナリ sha256）・ビルドログ（パスは `<WORK>`／`<HOME>` にマスク。ホスト名は含めない） |
| `aggregate.md` | 機械集計 |

## 再現（Mac mini 上、展開ツリーの本ディレクトリから）

```sh
CARGO_TARGET_DIR=$HOME/work/target-fandhe-ai OUT=<出力ディレクトリ> sh orchestrate_macmini.sh
```

計測前コミットを `git archive` して `.rev-stamp` を置き、rsync で転送する（`docs/real-hardware-verification-env.md` §7.1）。
ビルドと計測は並走させない。
