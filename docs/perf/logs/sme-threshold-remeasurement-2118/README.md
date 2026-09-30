# SME しきい値候補（`SME_MIN_K` = 64／128／256）再実測ログ置き場（イシュー #2118）

正式記録の転記先は `docs/perf/cpu-gemm-sme-fmopa-microkernel.md` §5.8。判定規則は
`RULE.txt`（事前登録。実測前に単独コミット済み）が正で、事後に緩和しない。

## 状態（2026-09-30）

- **未実測**。本ディレクトリを作成した実行ホストは x86_64 Linux で、Apple M4 Max
  にも DGX Spark GB10 にも届かない。#2117 と同じ運用（Linux 側で事前登録規則・
  実行基盤・記入欄まで作り、実測は実機セッションへ申し送る）に従い、
  **実測値・推定値はどこにも書いていない**。総合判定は「未確定（実測未実施）」。
- 受け入れ条件 AC1（M4 Max）・AC2（GB10）は未達。イシューは open のまま。
- `SME_PRODUCTION_ENABLED=false`・`SME_MIN_M/N=256`・`SME_MIN_K=64` は main で不変。
  候補値は計測専用ツリーへ当てるパッチでのみ使う。採否と定数切替は #2119 の
  ユーザー承認事項（本イシューのスコープ外）。
- `.rs`・`Cargo.toml`・`Cargo.lock`・tolerance・baseline・`docs/spec/` は変更していない。

## 収録物

| ファイル | 内容 |
|---|---|
| `RULE.txt` | 事前登録判定規則（候補別の到達セル表・R4・R1・R2・総合判定・GB10 語彙・RT 既知 FAIL） |
| `on-arm-k{64,128,256}.patch` | 計測専用パッチ（下表）。`patch -p1` で適用 |
| `lib_trees.sh` | 事前登録 sha（`SME2118_REGISTERED_BASE`）を `git archive` して before／after ツリーを作り、指紋差分 1 件・定数行を assert する共有ヘルパー |
| `orchestrate_m4max.sh` | M4 Max 用（`r4`・`r1 <K>`・`all`・`--dry-run`） |
| `gb10/orchestrate_gb10.sh` | GB10 用（`<K>`・`--dry-run`）。R0 → RT → R1／R2 |
| `aggregate.py` | 集計（python3 標準ライブラリのみ・`--self-test` 付き）。`aggregate.md` を生成 |
| （実測時に生成）`m4max/`・`gb10/k{K}/` | 生ログ・JSONL・compare 表・load_gate・env_info（マスク済み） |

### パッチ（基準 main HEAD `0b25525fa4026b951021a5d0da3a9d613b507502`）

| パッチ | 変更 | sha256 |
|---|---|---|
| `on-arm-k64.patch` | `SME_PRODUCTION_ENABLED` false→true のみ | `b0b4c5d1db02039dcc6e63c63f4d528d8e049938b65618f6fb83c7617e18da97` |
| `on-arm-k128.patch` | 上記 + `SME_MIN_K` 64→128 | `0e61051c1ae195cc2be11e5080d936068db0ec30bd7cfaa5cf5e63953309adf8` |
| `on-arm-k256.patch` | 上記 + `SME_MIN_K` 64→256 | `16c7aa29d7c009ed4e948ee8538d5acc52da16f7a6dce3fe835fed0313b34875` |

パッチ対象は `crates/backend-cpu/src/gemm_blis/mod.rs` の 2 行のみ。この HEAD 以降に
`mod.rs` が変わって適用できない場合は `patch_apply.log` で停止するので、再生成して
sha256 と HEAD を追記（訂正は追記方式）したうえで実測する。

Linux 側で確認済み: 3 本とも `git archive` の展開ツリーへ `patch -p1 --forward` で適用でき、
差分ファイルは `mod.rs` の 1 件のみ、`cargo check -p fandhe-ai-backend-cpu --lib --tests` が
`aarch64-unknown-linux-gnu`・`aarch64-apple-darwin` の両ターゲットで通る（コンパイル確認のみ。
実行・性能の確認ではない）。

## 実機セッション向け手順

### M4 Max（順序固定: R4 → R1 k64 → k128 → k256。RULE.txt §10）

```sh
cd docs/perf/logs/sme-threshold-remeasurement-2118
./orchestrate_m4max.sh --dry-run all      # 計画の確認
./orchestrate_m4max.sh all                # 数時間かかる。計測中はビルド等を並走させない
python3 aggregate.py . > aggregate.md
```

- R4 は `m4max/sme_r4_grid_run{1..5}.log`（`.raw.log` 併存・`load_gate_r4.log`）。
- R1／R2 は `m4max/r1r2/k{K}/`（LABEL `2118-m4max-k{K}`）。
- 各 run／round の前に load1 < 8.0 を最大 30 分待つ。5/5 通過の系列だけが正式で、
  それ以外は record_only（`aggregate.md` の「系列」欄に出る）。
- 既存ログ・既存 LABEL があると停止する（差し替え禁止）。再実行は別 LABEL で行う。

### GB10（順序固定: k64 → k128 → k256）

```sh
cd docs/perf/logs/sme-threshold-remeasurement-2118
gb10/orchestrate_gb10.sh --dry-run 64
gb10/orchestrate_gb10.sh 64 && gb10/orchestrate_gb10.sh 128 && gb10/orchestrate_gb10.sh 256
python3 aggregate.py . > aggregate.md
```

- 収録先は `gb10/k{K}/`（LABEL `2118-gb10-k{K}`）。外側専有ゲート（load1<1.0・GPU 0%）
  は記録のみで、不通過の系列は「参考」と明記する。
- RT の既知 FAIL は `gemm_blis::tests::sme_production_enabled_is_false_pending_measurement`
  の 1 件のみ（`rt_result.txt` の `rt_verdict`）。それ以外が出たら「後退あり相当（要調査）」。

### 転記

`aggregate.md` の値を `docs/perf/cpu-gemm-sme-fmopa-microkernel.md` §5.8 の記入欄へ転記する。
verdict は実測後に限り更新する。定数の切替・本番化の判断は #2119。

## 注意

- 収録前に `$HOME` は `<home>`・作業ディレクトリは `<work>` へ置換される（`lib_trees.sh` の
  `sme2118_mask`）。ホスト名は収録テキストへ出さず env_info に `hostname=masked` と書く
  （短い名前の内容置換が LABEL 等を壊すため）。bench バイナリはコミットしない。
- before 腕は実行時 HEAD ではなく、RULE.txt ヘッダの登録 sha（`lib_trees.sh` の `SME2118_REGISTERED_BASE`）を `git archive` して固定する（当該 sha が無ければ停止・HEAD へのフォールバックなし。実行時 HEAD は `current_head` として記録のみ）。
  RULE.txt 記載の基準 HEAD から `mod.rs` 以外が変わっていても、指紋差分の対象は
  before／after の 2 ツリー間なので判定は成立する（基準 HEAD からの変化は env_info で確認）。
- checksum は丸めた集約値で、bit 同一の証拠ではない（RULE.txt §5）。
