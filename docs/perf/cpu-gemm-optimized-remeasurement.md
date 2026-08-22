# CPU GEMM 最適化後 対 PyTorch 比 確定計測 記録（#574・Phase F）【状態: 実測完了（2026-08-18・Apple M4 Max 実機）】

イシュー #574「bench(backend-cpu): 最適化後の CPU 対 PyTorch 比を確定計測」の実測記録。GEMM 最適化
ツリー（ルート #479）の Phase F（親 #569「GEMM 最適化後の全バックエンド再計測と REQ-8 下限再確定」）
の CPU 担当タスクであり、Phase E（NEON マイクロカーネル・packing・キャッシュブロッキング再設計。
親 #551、closed）適用後の CPU GEMM について、`docs/performance-targets.md` §4 の計測プロトコルで
対 PyTorch 比を**確定計測**した。2026-08-18 に Apple M4 Max 実機（ローカル直接実行）で計測を実施し、
下記「対 PyTorch 比 結果」節に実測値を記録した（#574 の受け入れ条件を満たす）。

計測結果（判定対象形状 2048/4096 の比率最小値）は Phase F の人間承認タスク #577（REQ-8 下限の
再確定）への入力となる。**本ドキュメントでは REQ-8 下限値を変更しない**（`cpu-gemm-phase-e-remeasurement.md`
§「REQ-8 下限値との関係」・Metal `metal-floor-remeasurement.md` §「REQ-8 下限値の扱い」と同方針）。

依存 #567（E-10）は closed 済みで、成果物 `docs/perf/cpu-gemm-phase-e-remeasurement.md` に計測
プロトコル・ベースライン/HEAD SHA 規則・記録枠が整備済み（ただし M4 Max 実機未到達のため実測値は
未記入のまま申し送り）。

## 目的・受け入れ条件対応

| 受け入れ条件 | 対応 |
|---|---|
| `docs/performance-targets.md` §4 の計測プロトコル準拠（warmup 20 回以上・計測 20 回以上の中央値・Q1/Q3、決定的シード、判定対象形状 2048/4096） | 既存ハーネス `gemm_blis_baseline_pytorch_square_512_to_4096`（`MeasurementConfig::default()` = warmup 20・iters 20）をそのまま用いる。§「計測対象・計測境界」参照 |
| 既存 `docs/perf/` と同形式で記録 | `metal-floor-remeasurement.md`（#572）・`cpu-gemm-phase-e-remeasurement.md`（#567）と同型の構成（目的→環境ゲート→計測手順→記録表→候補下限→共通契約→状態） |
| REQ-8 下限値は変更しない | §「REQ-8 下限値の扱い」参照。変更は #577（人間承認）のみが行う |

## 実行環境ゲート判定

### 旧セッション（Linux x86_64・実測未達）

計測は Apple M4 Max 実機（`docs/real-hardware-verification-env.md` §7: Mac はローカル直接実行）で
のみ有効という前提のもと、実装セッション開始時に以下を判定した（#567・#572 と同一のゲート判定手順）。

1. `uname -sm` → `Linux x86_64`（実測。本 worktree の開発環境）。
2. `docs/real-hardware-verification-env.local.md`（Git 管理外のローカル用ファイル）→ 不存在（実測）。
   M4 Max への到達経路が定義されていないため代替経路もなし。
3. `git log b96c3b3..main -- crates/backend-cpu` → 空（実測）。#567 が定めた規則（同ドキュメント
   §「ベースライン基準コミットの確定」）に従い、実測時点の `main` tip を HEAD として使用できることを
   確認した。本セッションの `main` tip は `2ae2959`。

**結論**: 当該セッションでは M4 Max 実機に到達できないため、確定計測の記録ドキュメント（手順・記録枠・
候補下限導出手順）の整備までを実施し、**実測値の捏造・placeholder 値での完了扱いは行わない**
（fail-closed。#567・#572 先例と同方針）。実測は M4 Max 実機へ到達可能な後続セッション・Agent
（`bench-runner` 委譲想定）が引き継いで実施する方針とした。

### 実測セッション（2026-08-18・Apple M4 Max ローカル直接実行）

1. `uname -sm` → `Darwin arm64`（実測。M4 Max 実機に到達）。
2. 環境ゲート成立を確認し、下記「SHA 規則」節の再判定を経て実測を実施した。

## 計測対象・計測境界

- 計測対象カーネル: `gemm_blis_parallel`（`crates/backend-cpu/src/gemm_blis/mod.rs`。BLIS 5-loop、
  ブロッキングパラメータ `MC=128`/`KC=256`/`NC=512`、aarch64 では `dispatch_region` が無条件に
  `NeonKernel`〈`MR=8`/`NR=12`、k=4 アンロール＋ソフトウェアパイプライン〉を選択）。
- ハーネス: `crates/backend-cpu/tests/gemm_blis_perf.rs::gemm_blis_baseline_pytorch_square_512_to_4096`
  （`#[ignore]`。A-8・#567 で使用した既存ハーネスをそのまま再利用。ハーネス改修は不要 —
  §4 準拠の中央値・Q1/Q3 出力を既に備えており、受け入れ基準「既存 docs/perf と同形式で記録」に対し
  ハーネス側の不足はない）。
- 計測境界: `gemm_blis_parallel` 呼び出し単体（C バッファの事前ゼロクリアを計測区間から除外済み。
  ハーネス側コメント参照）。数値正当性確認（`gemm_parallel` との `assert_eq!`）は計測区間の外側で
  独立して実行される。
- 出力形式: `kernel=gemm_blis_parallel size=<n> median_tflops=<v> q1_tflops=<v> q3_tflops=<v>
  median_secs=<v>`（TFLOPS = 2·N³ / median_secs / 1e12）。

## SHA 規則

HEAD SHA は #567 の規則（`cpu-gemm-phase-e-remeasurement.md` §「ベースライン基準コミットの確定」）
を参照し、本ドキュメントでは二重管理しない。

**実測セッション（2026-08-18）での再判定**: `git log b96c3b3..origin/main -- crates/backend-cpu` が
**非空**（`b054530` #718「融合 RMSNorm/softmax NEON 実装」を含む）だったため、規則
（`cpu-gemm-phase-e-remeasurement.md` §「ベースライン基準コミットの確定」）に従い main tip
（`12736c4`）ではなく **`b96c3b3` を HEAD として使用した**（Phase E 帰属の改善率が後続変更の効果と
混ざるのを防ぐため）。ベースライン SHA は `1cb2938`（A-8・E-10 と同一）。

ベースライン比較（対 A-8・対 Phase E 改善率）は `cpu-gemm-phase-e-remeasurement.md` 側の責務であり、
本ドキュメントでは**対 PyTorch 比のみ**を扱う。

## 計測手順（Apple Silicon 実機）

```bash
git fetch origin
git checkout bench/574-cpu-optimized-remeasurement   # 本イシューの実装ブランチ
git log b96c3b3..main -- crates/backend-cpu           # 空であることを再確認（空でなければ b96c3b3 を使う）

# 1. 数値一致確認を先に行う（既存 parity テスト群。閾値は緩和しない）
cargo test -p fandhe-ai-backend-cpu --release --test gemm_blis_parity

# 2. Rust 側ベンチを 5 回独立実行し、size ごとに 5 run の中央値の中央値を採用する
#    （MeasurementConfig::default() 自体が warmup 20・iters 20・中央値を内包するため、
#    5 プロセス独立実行との組み合わせで「5 回計測の中央値」下限
#    〈.claude/rules/coding-rust.md〉を二重に満たす。#567 先例と同方式）
cargo test -p fandhe-ai-backend-cpu --release --test gemm_blis_perf \
    -- --ignored gemm_blis_baseline_pytorch_square_512_to_4096 --nocapture
```

PyTorch 側は一時 venv（リポジトリ管理外）で実行する。A-8（#488）・E-10（#567）と同一実機・同一
PyTorch 版（`torch==2.13.0`）で計測済みの値がある場合は再利用し二重計測を避ける（PyTorch 側の
演算経路〈`torch.matmul`〉は Rust 側の変更と無関係のため、ベースライン・Phase E・本確定計測の
3 ドキュメントで共用してよい）。

```bash
python3 -m venv .venv-gemm-optimized
source .venv-gemm-optimized/bin/activate
pip install torch==2.13.0 numpy
git submodule update --init docs/spec  # 未初期化の場合のみ
python3 docs/spec/03-poc/poc-v2-1-tensor-cpu-gemm/code/pytorch/gemm_bench_torch_cpu.py <size> 20 20
```

`<size>` ∈ {512, 1024, 2048, 4096} それぞれについて 5 回実行しログを保存する（既存記録の再利用が
ない場合）。Rust 側と同様に size ごとに 5 run の中央値の中央値を採用する。

計測衛生（#488・#567 先例と同方式）: 他プロセス負荷の混入に注意し、異常 run（外れ値の明確な原因が
あるもの）は破棄・取り直しを記録に残す。

## parity 事前確認

計測前に既存数値一致テスト（`tests/gemm_blis_parity.rs`）を実行する。ハーネス内 `assert_eq!`（bit
一致契約）が実機で失敗した場合、それは「計測ハーネスの不具合」ではなく「NEON 経路の bit 一致契約に
関する新規の parity 上の発見」として扱う（#567 §「計測手順」と同方針）。**assert 削除や tolerance
緩和で回避しない**（`.claude/rules/coding-rust.md`「バックエンド間数値一致テストの許容誤差を単独で
緩和しない」に準ずる）。発見した場合は本ドキュメントの Appendix・イシューコメントに記録して報告する。

## 計測環境（実測値）

| 項目 | 値 |
|------|-----|
| チップ | Apple M4 Max |
| OS | macOS 26.6.1 |
| 論理コア数 | 16 |
| rustc | 1.96.0 (ac68faa20 2026-05-25) |
| torch | 2.13.0（macOS arm64） |
| PyTorch `torch.get_num_threads()` | 12 |
| PyTorch BLAS | `BLAS_INFO=accelerate`（Apple Accelerate/vecLib。AMX 経路を含みうる。`torch.__config__.show()` 実測） |
| venv 構成 | `python3 -m venv` + `pip install torch==2.13.0 numpy`（リポジトリ管理外） |
| 計測コミット SHA（ベースライン） | `1cb2938` |
| 計測コミット SHA（HEAD） | `b96c3b3`（上記「SHA 規則」参照） |
| parity（`gemm_blis_parity`） | `cargo test -p fandhe-ai-backend-cpu --release --test gemm_blis_parity` → 16 passed; 0 failed |
| 計測プロトコル | `bench_harness::protocol::run`（warmup 20・iters 20・中央値/Q1/Q3。TASK-8.1）を 5 回独立実行し size ごとに中央値の中央値を採用（Rust・PyTorch 双方） |
| 決定的シード | `Xorshift64Star`（`bench_harness::rng`。ハーネス内固定シード `3000+size`/`4000+size`） |
| 同期境界 | Rust: `gemm_blis_parallel` 呼び出し完了（CPU 同期処理のため追加の完了待ちは不要）／PyTorch: スクリプト内計測ループ完了 |

Rust 側は rayon 既定（16 論理コア）で並列化される。PyTorch 側（Accelerate・12 threads）とはスレッド数
が異なるが、いずれも実機の既定構成（追加のスレッド数チューニングなし）での計測である。

## 対 PyTorch 比 結果

| size | Rust median TFLOPS（5 run の中央値） | Rust Q1/Q3 TFLOPS | PyTorch median TFLOPS（5 run の中央値） | PyTorch Q1/Q3 TFLOPS | 対 PyTorch 比 |
|------|------|------|------|------|------|
| 512（参考値） | 0.470527 | 0.460701 / 0.482219 | 2.6040 | 2.5434 / 2.8506 | 18.1% |
| 1024（参考値） | 0.711176 | 0.703219 / 0.739089 | 3.1666 | 3.0805 / 3.1789 | 22.5% |
| 2048 | 0.800964 | 0.779322 / 0.834401 | 3.2375 | 3.2029 / 3.2446 | 24.7% |
| 4096 | 0.931332 | 0.903771 / 0.960642 | 3.0101 | 2.8946 / 3.0404 | 30.9% |

判定対象形状（REQ-8「判定対象形状」節）は **M=N=K=2048・4096 の実測比率の最小値**。512/1024 は参考値。
**判定対象最小値は 24.7%（size=2048）**。

異常 run の扱い: 明確な原因のある外れ値はなく、全 5 run を採用した（head 512 run2=0.386215、
head 2048 run2=0.725492、head 4096 run1=1.034018 のばらつきは原因不明のため保持。head→baseline の
実行順でサーマルドリフトによる改善率の水増しを示す証拠はない）。

## 計測条件の非対称性（開示）

本計測の対 PyTorch 比 24.7%（size=2048）は、**自作 NEON マイクロカーネル（`gemm_blis_parallel`・
rayon 16 論理コア並列）と、Apple Accelerate（vecLib・AMX 経路を含みうる、12 threads）による高度に
最適化された BLAS 実装との比較**である。PyTorch 側は Apple 純正の行列積アクセラレータ（AMX、
Accelerate 経由）を使用しうる一方、Rust 側は NEON SIMD に留まり AMX は未使用である。この非対称性は、
#577（REQ-8 下限の再確定）における「実測比率をどこまで REQ-8 下限へ反映
するか」の判断材料として明記する（本ドキュメントでは下限を変更しない）。

## 候補下限値導出手順（参考算出）

1. 上表の判定対象形状（2048/4096）の対 PyTorch 比率のうち最小値を求める。
2. `bench_harness::floor_lower_bound(measured_percent: f64) -> Result<u32, RoundingError>`
   （`crates/bench-harness/src/rounding.rs:88`）に、上記最小値をパーセント表記（例: 32.7%
   なら `32.7`）で渡し、候補下限値（整数パーセント、切り下げ丸め）を得る。
3. 得られた候補下限値と現行の最適化後下限 30%（`docs/performance-targets.md` §2 CPU 行。
   `crates/bench-harness/src/threshold.rs::floor_spec` の `CpuF32`/`Optimized` 分岐）を比較し、
   本節に記録する。
4. `bench_harness::threshold::judge`（`own: &BenchReport, pytorch: &BenchReport, backend_dtype,
   stage`）を用いた自動判定を行う場合は、上表の median/Q1/Q3 TFLOPS から `BenchReport` を構成する
   （型定義は `crates/bench-harness/src/threshold.rs` 参照）。

**値の反映判断（下限の最終確定・`docs/performance-targets.md` §2・`docs/spec/04-requirements.md`
への反映）は #577（人間承認）へ引き継ぐ。本ドキュメントでは下限を変更しない。**

候補下限値（参考算出）: 判定対象最小値 24.7%（size=2048）を `floor_lower_bound(24.7)` へ渡すと、
24.7% は 10% 以上のため 5% 刻み切り下げが適用され `floor(24.7 / 5.0) * 5.0 = floor(4.94) * 5.0 = 20`
となる。**候補下限値は 20%**。

現行の最適化後下限 30%（`docs/performance-targets.md` §2 CPU 行・`crates/bench-harness/src/threshold.rs::floor_spec`
の `CpuF32`/`Optimized` 分岐）と比較すると、**実測 24.7% に基づく候補下限値 20% は現行下限 30% を
下回る**。上記「計測条件の非対称性（開示）」節のとおり、PyTorch 側が Accelerate/AMX 系の高度最適化
BLAS であることが要因の一部と考えられる。反映判断（下限を実測値に合わせて引き下げるか、非対称性を
理由に現行 30% を維持するか）は #577（人間承認）へ引き継ぐ。本ドキュメントでは下限を変更しない。

## REQ-8 下限値の扱い

**REQ-8 下限値（初期リリース 20%／最適化後 30%）は本ドキュメントでは変更しない。** 変更は Phase F
の人間承認タスク #577 のみが行う。本ドキュメントは候補下限値の参考算出（上記「候補下限値導出手順」
節）を提供するに留め、下限の最終確定・`docs/spec/04-requirements.md` への反映判断は行わない
（`docs/spec/` は本リポでは編集しない）。

## 共通契約の遵守

- **境界チェック不省略**: 本イシューは計測・ドキュメントのみで `gemm_blis_parallel`・NEON マイクロ
  カーネルの手動境界チェックは変更していない。
- **tolerance 不緩和**: バックエンド間数値一致テストの許容誤差は変更していない。既存ハーネスの
  `assert_eq!`（bit 一致契約）もそのまま。
- **依存追加なし**: `Cargo.toml`／`Cargo.lock` に変更なし（PyTorch venv はリポジトリ管理外・計測
  専用）。
- **`docs/spec/` 不編集**: PoC-v2-1 の PyTorch スクリプトは読み取り実行のみで submodule への変更は
  行わない。
- **REQ-8 下限値不変更**: `docs/performance-targets.md` §2 の値は本ドキュメントでは変更していない。

## 状態: 実測完了（2026-08-18・Apple M4 Max 実機）

本ドキュメントは当初 Linux worktree で作成され、M4 Max 実機が同一セッションで使用できないため計測
手順・記録テンプレートのみを整備した（#567・#572 先例と同方式）。2026-08-18 に Apple M4 Max 実機
（ローカル直接実行）で「計測手順」節の手順に従い実測を実施し、上記「対 PyTorch 比 結果」「計測環境」
「候補下限値導出手順」の各節を実測値で記入した。

内部ホスト名等の実値は書かない（#461 のプレースホルダ方針。実測時の原文は
`docs/real-hardware-verification-env.local.md` へ記録する）。

## 動作確認（Linux セッションで実施済み）

- `cargo fmt --all --check`
- `cargo clippy --workspace --all-targets --all-features -- -D warnings`
- `cargo test --workspace`（Linux 実行分。実機依存・`#[ignore]` テストは除外）
- `cargo test -p fandhe-ai-backend-cpu --test gemm_blis_perf`（`--ignored` を付けない通常実行でハーネスの
  コンパイル成立を確認。x86_64 上で `--ignored` 実測は行わない）
- `git diff --stat` で `crates/bench-harness/src/threshold.rs`・数値一致 tolerance 定数・
  `docs/spec/`・`guardrail.toml`・`Cargo.toml`／`Cargo.lock` に差分がないことを確認

## 未実施・後続作業

- **M4 Max 実機での実測**: 「状態」節のとおり本セッション（2026-08-18）で実施済み。
  `cpu-gemm-baseline-remeasurement.md`（A-8）・`cpu-gemm-phase-e-remeasurement.md`（E-10）の記録枠
  も同一セッションで併せて記入した（Rust 側は同一ハーネス・PyTorch 側は同一スクリプトのため、
  ベースライン SHA・HEAD SHA の 2 セットの計測で複数ドキュメントの表を同時に満たした）。
- **候補下限値の最終確定・REQ-8 反映判断**: #577（人間承認）が対応する。本ドキュメントは候補下限値
  20%（現行下限 30% を下回る）を判断材料として提供する。
- **E-7（明示プリフェッチ）の採否・E-8（MC/KC/NC 実機スイープ）**: 既存の保留判断のまま本イシュー
  では扱わない。
- **`docs/performance-targets.md` 本体の更新**: #579（最終整合タスク）の担当範囲。

## Appendix: 全 run 生ログ

`kernel=gemm_blis_parallel size=<n> median_tflops=<v> q1_tflops=<v> q3_tflops=<v> median_secs=<v>`
（HEAD SHA `b96c3b3`。ベースライン SHA `1cb2938` 分の生ログは `cpu-gemm-baseline-remeasurement.md`・
`cpu-gemm-phase-e-remeasurement.md` の Appendix を参照）。

### Rust 側（HEAD、5 run）

```text
run1: size=512 median_tflops=0.469395 q1_tflops=0.438710 q3_tflops=0.491040 median_secs=0.000572
run1: size=1024 median_tflops=0.712188 q1_tflops=0.705462 q3_tflops=0.743975 median_secs=0.003015
run1: size=2048 median_tflops=0.839325 q1_tflops=0.802042 q3_tflops=0.856456 median_secs=0.020469
run1: size=4096 median_tflops=1.034018 q1_tflops=0.995078 q3_tflops=1.063882 median_secs=0.132917
run2: size=512 median_tflops=0.386215 q1_tflops=0.350457 q3_tflops=0.429325 median_secs=0.000695
run2: size=1024 median_tflops=0.684030 q1_tflops=0.666248 q3_tflops=0.693268 median_secs=0.003139
run2: size=2048 median_tflops=0.725492 q1_tflops=0.714098 q3_tflops=0.773227 median_secs=0.023680
run2: size=4096 median_tflops=0.912519 q1_tflops=0.837805 q3_tflops=0.963546 median_secs=0.150615
run3: size=512 median_tflops=0.470527 q1_tflops=0.460701 q3_tflops=0.482219 median_secs=0.000571
run3: size=1024 median_tflops=0.711176 q1_tflops=0.703219 q3_tflops=0.739089 median_secs=0.003020
run3: size=2048 median_tflops=0.800964 q1_tflops=0.779322 q3_tflops=0.834401 median_secs=0.021449
run3: size=4096 median_tflops=0.931332 q1_tflops=0.903771 q3_tflops=0.960642 median_secs=0.147572
run4: size=512 median_tflops=0.481031 q1_tflops=0.469293 q3_tflops=0.488842 median_secs=0.000558
run4: size=1024 median_tflops=0.730064 q1_tflops=0.695288 q3_tflops=0.764671 median_secs=0.002942
run4: size=2048 median_tflops=0.820975 q1_tflops=0.793633 q3_tflops=0.884352 median_secs=0.020926
run4: size=4096 median_tflops=0.957944 q1_tflops=0.931804 q3_tflops=0.997889 median_secs=0.143473
run5: size=512 median_tflops=0.472217 q1_tflops=0.466034 q3_tflops=0.495230 median_secs=0.000568
run5: size=1024 median_tflops=0.705635 q1_tflops=0.685987 q3_tflops=0.735387 median_secs=0.003043
run5: size=2048 median_tflops=0.786558 q1_tflops=0.771323 q3_tflops=0.812389 median_secs=0.021842
run5: size=4096 median_tflops=0.881967 q1_tflops=0.848707 q3_tflops=0.934798 median_secs=0.155832
```

### PyTorch 側（5 run。ベースライン・HEAD・E-10 の 3 ドキュメントで共用）

```text
run1: torch=2.13.0 numpy=2.5.2 threads=12 size=512 median_tflops=2.8108 q1=2.7913 q3=2.8994
run1: size=1024 median_tflops=3.1711 q1=3.1557 q3=3.1915
run1: size=2048 median_tflops=3.2440 q1=3.2368 q3=3.2452
run1: size=4096 median_tflops=3.0432 q1=2.9049 q3=3.0597
run2: size=512 median_tflops=2.6178 q1=2.5515 q3=2.8748
run2: size=1024 median_tflops=3.1691 q1=3.1486 q3=3.1953
run2: size=2048 median_tflops=3.3803 q1=3.3511 q3=3.3920
run2: size=4096 median_tflops=2.9889 q1=2.8987 q3=3.0596
run3: size=512 median_tflops=2.5915 q1=2.5195 q3=2.7461
run3: size=1024 median_tflops=3.1473 q1=3.0840 q3=3.1666
run3: size=2048 median_tflops=3.2091 q1=3.1392 q3=3.2380
run3: size=4096 median_tflops=3.0101 q1=2.8946 q3=3.0404
run4: size=512 median_tflops=2.5978 q1=2.5565 q3=2.8419
run4: size=1024 median_tflops=3.1666 q1=3.0805 q3=3.1789
run4: size=2048 median_tflops=3.2255 q1=3.2074 q3=3.2396
run4: size=4096 median_tflops=2.9914 q1=2.9072 q3=3.0045
run5: size=512 median_tflops=2.6040 q1=2.5434 q3=2.8506
run5: size=1024 median_tflops=3.1621 q1=3.1511 q3=3.1674
run5: size=2048 median_tflops=3.2375 q1=3.2029 q3=3.2446
run5: size=4096 median_tflops=3.0115 q1=2.9043 q3=3.0152
```

### parity（`gemm_blis_parity`、HEAD SHA `b96c3b3`）

```text
running 16 tests
（全 16 test ok）
test result: ok. 16 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out
```
