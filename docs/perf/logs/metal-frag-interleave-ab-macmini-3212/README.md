# simdgroup 交互配置候補の Mac mini A/B（イシュー #3212）

結論は `docs/perf/metal-gemm-steel-candidates.md` §10.8・§10.9。本ディレクトリは事前登録規則・計測基盤・実測ログ。
**診断のみ**（`tile.rs` の定数・選択ロジック・tolerance・parity baseline・FMA 契約・シェーダは不変）。

## 状態: Mac mini で実測済み（正式系列）

- 候補 `FRAG_INTERLEAVE_ENABLED` の総合判定は **REJECT**（全 N で改善は 0.5〜1.6% と、事前登録した 2% の帯内）。
- 規則の固定: `RULE.txt` の sha256 = `50e53980ddaab3a891548a5df362e4b3c90da9bab075f131bfc973375e6cbd46`、
  規則を固定したコミット = `bf2dbe7c8`（計測基盤はそのあとの別コミット。規則本文は固定後に変更していない）。
- 計測対象リビジョン（`rev-stamp.txt`）= `3d5a08b318b13ac54925766e34c8823337b51498`。ツリー指紋は開発機の `git archive` と一致（`tree-fingerprint.txt`）。
- 機体: Mac mini（Apple M4・GPU 10 コア・16 GB）、`host_kind=macmini`。ホスト名・ユーザー名・`$HOME` はマスク済み。
- 純カーネル計測は load1 < 3.0 ゲートを全 38 プロセスが待機なしで通過。gemm セル A/B も 5 round 全て pass。

## 収録物

| パス | 内容 |
|---|---|
| `RULE.txt` | 事前登録規則（計測前に単独コミット） |
| `orchestrate_macmini.sh` | 純カーネル計測の起動（ビルド → 前提ゲート → 5 run × {trace・kern} と candle excl。load1 ゲート・上書き禁止・`.rev-stamp` 必須） |
| `aggregate.py` | 厳格な集計（標準ライブラリのみ。`python3 -I aggregate.py --self-test`／`python3 -I aggregate.py <dir> --expect-rev <SHA> > aggregate.md`）。ab/ が無ければセル A/B は unavailable として総合を undetermined にする |
| `trace_run{1..5}.log`・`kern_run{1..5}.log` | `DEVICE`／`TRACE`／`ARM`／`MEAS` 行 |
| `candle_run{1..5}.jsonl`・`candle_tree.txt` | candle excl（`gemm-transfer-split`）と candle-core の版 |
| `prereq_bitident.log`・`bitident_report.log`・`parity.log` | 前提ゲート（既存 bit 一致 7 件・ON/OFF bit 一致・CPU 参照 parity） |
| `load_gate.log`・`env_info.txt`・`rev-stamp.txt`・`tree-fingerprint.txt`・`build_*.log` | 負荷ゲート・機体情報・腕の同定・ビルドログ |
| `ab/` | セル A/B runner の出力（`results-*.jsonl`・`sha-*.txt`・`tree-*.txt`・`compare-*`・`load-gate-*.log`・`uptime-*.log`・`skipped-*.log`）。バイナリは収録しない |
| `aggregate.md` | 機械集計 |

診断テストは `crates/backend-metal/src/gemm_frag_interleave_ab_diag_tests.rs`（出力行の書式は `aggregate.py` の正規表現と 1 対 1）。
セル A/B の runner は `scripts/bench/framework-compare/run_ab_frag_interleave_gemm_metal.sh`。
`compare_gemm_ab.py` には N=256 を含む `--sizes metal-square5` を追加した（既存の集合は不変）。

## 実測時の注意（再現・追試向け）

- `compare_gemm_ab.py --threshold 1.00` は中央値比が 1.00 を超えるセルが 1 つでもあると exit 3 になる（本実測も `compare exit=3`。
  `compare-exit-3212-metal-3212.log`）。checksum 不一致ではない（全 10 セル完全一致）。RULE 項 11 の判定（5/5 round で 2% 超後退）は
  `aggregate.py` が JSONL から直接計算する。
- N=256 の run 2 は cur・fi とも約 2 倍遅い外れ値（同 run の candle も遅い）。除外せず run 別の表に残している。
- 補助腕（`CANDIDATES[1..3]` の ON/OFF）は記録のみ。判定に使わない。

## 再現（Mac mini 上。`docs/real-hardware-verification-env.md` §7）

1. 計測前コミットを `git archive` して `.rev-stamp` を置き、rsync で転送する（`--delete-excluded` は使わない。openrsync は protect フィルタを無視する）。
   ツリー指紋（`.rev-stamp`・`.git` を除く全ファイルの sha256 を整列して再ハッシュ）を両側で突合する。
2. セル A/B 用に before／after の 2 clone を作る（本実測は diag tree を `git init` で 1 コミット化して before とし、その clone を after にして
   `crates/backend-metal/src/tile.rs` の `FRAG_INTERLEAVE_ENABLED` を false→true の 1 行だけ書き換えた。after は計測専用でコミットしない）。
3. 純カーネル計測（ビルドは計測前に全部済む）:
   `CARGO_TARGET_DIR=$HOME/work/target-fandhe-ai OUT=<出力ディレクトリ> sh orchestrate_macmini.sh`
4. セル A/B（3 を終えてから。同時に走らせない）:
   `AB_BEFORE_FACADE_PATH=<abs>/before/crates/facade AB_AFTER_FACADE_PATH=<abs>/after/crates/facade bash run_ab_frag_interleave_gemm_metal.sh 3212`
5. 回収・マスク・集計（`aggregate.py --expect-rev <SHA>`）。いずれも nohup のサブシェルで切り離して起動し、生存を確認する。
