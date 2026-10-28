# GB10 CPU gemm 常駐入力経路の reuse>fresh 逆転の切り分け（#3646）

0.12.0 の 6 フレームワーク同席計測（#3273・`../framework-compare-0.12.0-remeasure/README.md`）で、
bench-fandhe の `gemm --mode reuse` を常駐入力経路（#3272。`Tape::upload_resident` → `matmul_resident`）へ
切り替えたところ、GB10 の CPU gemm だけ reuse が同 round の fresh より 1.23〜1.79 倍遅くなった。
本ディレクトリはその原因を GB10 実測で切り分けるための事前登録規則・診断基盤・集計器を置く。
結論・帰属・修正方針の提案は `docs/perf/cpu-gemm-reuse-fresh-inversion.md` §9 に書く。

## 状態（2026-10-28 GB10 実測済み。参考記録・帰属保留）

**RULE.txt の専有条件（GB10 の compute プロセス 0）を満たせなかったため、本ディレクトリの実測はすべて参考記録とする。正式系列としての判定・帰属の確定は行わず保留する。確定には、専有した GB10 での独立系列、または実測前に条件変更を登録した独立系列の再計測が必要（未実施）。以下の「正式系列」「帰属」「分類」の語はこの留保つきで読むこと。**

| 項目 | 状態 |
|---|---|
| 診断テスト・RULE.txt・RULE.v2.txt・orchestrate.sh・aggregate.py | 整備済み。`RULE.v2.txt`（ブリッジ節のみの置換。理由は同ファイル冒頭）を実測前に別コミットで固定 |
| GB10 系列（5 run。参考記録） | **実測済み**。全 run が負荷ゲート通過（`gb10/load_gate_status.txt`）・checksum bit 一致・`aggregate.py` exit 0（`gb10/aggregate.md`） |
| 補助系列（minflt 1 run・tunables 5 run・taskset 5 run） | **実測済み**（判定外。`gb10-minflt/`・`gb10-tunables/`・`gb10-taskset/` の `aggregate.md`） |
| ブリッジ（bench-fandhe 5 run × 5 N × 3 モード） | **実測済み**。N=4096 のみ pass、N=256〜2048 は fail（`gb10-bridge/bridge_judge.md`） |
| G-CPU-INF の分類（要件 2） | **実測済み**（`gb10-gcpuinf/`）。分類は doc §9.4 を参照 |
| 修正（要件 3） | 実装しない（所有者への方針提示のみ。doc §9.5） |

## 結果の要約（詳細・解釈・限界は `docs/perf/cpu-gemm-reuse-fresh-inversion.md` §9）

- bench 側の逆転は GB10 で**再現**した（bench-fandhe の reuse iter_total / fresh = N=256〜4096 で 1.71・1.64・1.47・1.57・1.25）
- 診断の `resident_public` / `fresh` は N=256 が非再現（0.970）、N=512〜2048 が 1.04〜1.06、N=4096 が 1.214。**参考値として帰属候補が出たのは N=4096 だけ（保留）**
  （rho_DL 0.78 → H1: `download` の追加確保＋コピー。rho_DROP 0.36・rho_K 0.00・rho_IN 0.02 は未達）
- N=256〜2048 は診断が bench の逆転幅の 1/12〜1/15 しか写せておらずブリッジ fail。**この範囲の原因は未特定**
- 補助系列（判定外）: tunables は N=256・512 で ratio <= 1.05、N=1024 以上で > 1.05。taskset は ratio をほぼ動かさない（N=4096 で 1.201）

## 実行環境と逸脱（RULE.txt との差）

- ノードは `<cuda-node>`（GB10）。**RULE.txt の「GB10 専有（compute プロセス 0）の別途確認」は満たせなかった（このため参考記録へ降格）**。常駐 GPU プロセス 3 件
  （LLM 推論 2・画像生成 1。停止していない）が計測前後とも同一で、ps の %CPU は最大 3.9・`utilization.gpu` は 0 %、計測中に新規プロセスは現れていない
  （`session-scripts/gpu_procs.txt` の 9 時点）。aggregate.py が機械的に強制するのは load1 < 1.0（全 run 通過）だが、RULE.txt は専有確認も要求しており、これを欠くため正式判定には用いない
- 転送は固定コミットの tracked ファイルだけを tar で展開する方式（rsync の `--delete-excluded` は使っていない）。ビルドは実機ロック取得前に完了。実機ロックは外部の `mkdir` 方式
- `RULE.v2.txt` でブリッジ節だけを差し替えた（bench-fandhe が `gemm --mode fresh --phases` を MEASURE_ERROR で拒否するため RULE.txt のブリッジは実行不能）。
  `aggregate.py`・閾値・帰属規則は不変
- perf は `perf_event_paranoid=4` で拒否（`gb10-gcpuinf/perf_record.log`）。sudo では再試行していない。代わりに `LD_PRELOAD` の観測専用 shim で
  PyTorch の mm 実行時の呼び出し先を確認した（`gb10-gcpuinf/blas_probe/`。計測外・時間は測らない）。これは RULE 固定の手順（LD_DEBUG・perf）に
  対する追加確認で、(iii) の判定根拠として用いるのはこの追加確認である
- `session-scripts/session-3646.sh` の `blas-path rc=` は直前の echo の終了コードで意味を持たない（shellcheck SC2320）。実行物と同一のため修正していない
- 回収物のホスト名・`/home` パスは `<home>` に置換済み（数値は書き換えていない）

## 回収物の見方

| ディレクトリ・ファイル | 内容 |
|---|---|
| `gb10/` | 正式系列。`run{1..5}.jsonl`・`env_info.txt`・`load_gate*`・`bridge_status.txt`・`aggregate.md`（`python3 aggregate.py gb10` の出力） |
| `gb10-minflt/`・`gb10-tunables/`・`gb10-taskset/` | 補助系列（判定外）と各 `aggregate.md` |
| `gb10-bridge/` | bench-fandhe の生 JSONL（fresh／reuse／reuse --phases × N × 5 run）・`rc.log`・`bridge_judge.py`・`bridge_judge.md` |
| `gb10-gcpuinf/` | G-CPU-INF: PyTorch 純カーネル・PyTorch 窓・fandhe 窓・#2105 層別区間（`fixedcost-2105/`）・BLAS 経路（`blas_path.log`・`blas_bindings.txt`・`blas_probe/`）・`gcpuinf_judge.py`／`gcpuinf_judge.md` |
| `session-scripts/` | 実行したセッションスクリプト・`steps.log`・`gpu_procs.txt`・`load_gate_bench.log`・`orch-*.log`・`versions.txt` |
| `rev-stamp-verification-3646.md` | `.rev-stamp`・ツリー指紋・RULE の sha256 |

## 構成

- `RULE.txt`: 事前登録規則（腕・区間・代表値・checksum 規則・再現判定・帰属 rho・補助系列・ゲート・G-CPU-INF 分類）
- `orchestrate.sh <m4max|gb10> [--aux minflt|tunables|taskset] [--out <dir>]`: 1 回ビルドして独立 5 プロセスで実行
  （`--aux minflt` のみ 1 run）。`.rev-stamp` 必須・`RAYON_NUM_THREADS` 拒否・既存 run 上書き禁止・
  GB10 は load1 < 1.0 ゲート・ホスト名／`$HOME`／ユーザー名をマスク
- `aggregate.py <dir>`: 標準ライブラリのみ。欠落・n_samples 不一致・checksum bit 不一致・ゲート／系列種別欠落は fail-closed。
  判定は `series_kind=main` かつ全 run `gate=pass` の系列のみ

## 腕（窓は `iter_total`）

| 腕 | 内容 | 分離する仮説 |
|---|---|---|
| fresh | bench `run_gemm` と同じ列（Tape drop は窓の外） | 基準 |
| resident_public | facade `matmul_resident` → contiguous 借用 → checksum（bench 新腕の写し） | 再現対象 |
| resident_split | `gemm_device`／`download`／`out_drop`／readout／checksum の区間分解 | 区間の内訳 |
| gemm_device_hold | `gemm_device` のみ（drop は窓の外） | カーネル＋確保の基準 |
| gemm_device_drop_in | `gemm_device` ＋窓内 drop | H1b（munmap・TLB） |
| gemm_device_download | `gemm_device` ＋ `download`（drop は窓の外） | H1（`to_vec` の確保とコピー） |
| resident_reupload | 毎反復 窓の外で upload し直す | H2・H3（入力の寿命・配置） |
| backend_drop | `BackendOps::gemm` 直接（#3187 の対照） | カーネル基準 |

## 計測ツリーの等価性（実装時の確認）

0.12.0 の bench（registry `fandhe-ai =0.12.0`）と HEAD の診断テストが同じコードを測ることを、
`git diff --stat v0.12.0 HEAD -- crates/backend-cpu/src crates/autodiff/src/optim/resident_input.rs crates/autodiff/src/optim/device_store.rs crates/facade/src/lib.rs crates/tensor-core/src/alloc.rs`
が空であることで確かめる（実装時点の結果は §9 に記録）。差分が出た場合は計測ツリーを `v0.12.0` tag に固定する。

## 実行手順（GB10。`docs/real-hardware-verification-env.md` の手順に従う）

1. RULE.txt を含むコミットを計測ツリーとして rsync し、ルートに `.rev-stamp`（1 行目 HEAD・2 行目 `dirty=<件数>`）を置く。
   ツリー指紋を `git archive` の展開物と突合して `rev-stamp-verification-3646.md` に残す
2. `setsid nohup` で切り離し hw-lock を取得、PATH を明示、計測中はビルド並走禁止
3. `bash orchestrate.sh gb10`
4. ブリッジ（集計の前に行う。**RULE.v2.txt が正**）: bench-fandhe の fresh 通常行・reuse 通常行・`--mode reuse --phases` を N=256〜4096 各 5 run（`--out` を必ず明示）。`python3 gb10-bridge/bridge_judge.py .` が `G_bench`／`G_diag` の比で判定し `gb10/bridge_status.txt` へ `<N> bridge=pass|fail` を 1 行ずつ書く（未記録・fail の N は aggregate.py が帰属を保留する）
5. `python3 aggregate.py gb10`。補助: `--aux minflt`・`--aux tunables`・`--aux taskset`（別ディレクトリへ出力。判定外）
6. G-CPU-INF: RULE.txt 末尾の測定 (i)〜(iii)（既存の承認済み venv のみ。新規パッケージ導入はしない）
