# CPU GEMM 最適化後 対 PyTorch 比 確定計測 記録（#574・Phase F）【状態: 未計測。M4 Max 実機セッションへ引き継ぎ】

イシュー #574「bench(backend-cpu): 最適化後の CPU 対 PyTorch 比を確定計測」の実測記録。GEMM 最適化
ツリー（ルート #479）の Phase F（親 #569「GEMM 最適化後の全バックエンド再計測と REQ-8 下限再確定」）
の CPU 担当タスクであり、Phase E（NEON マイクロカーネル・packing・キャッシュブロッキング再設計。
親 #551、closed）適用後の CPU GEMM について、`docs/performance-targets.md` §4 の計測プロトコルで
対 PyTorch 比を**確定計測**し、既存 `docs/perf/` と同形式で記録することを目的とする。**ただし本
セッションは M4 Max 実機に到達できないため（下記「実行環境ゲート判定」参照）、実測値は未記入の
まま整備した計測手順・記録枠のみを成果物とする。確定計測そのものの完了は実機到達可能なセッションへ
引き継ぐ（本ドキュメント単体では #574 の受け入れ条件を満たさない）。**

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

## 実行環境ゲート判定（本イシュー実装セッション時点）

計測は Apple M4 Max 実機（`docs/real-hardware-verification-env.md` §7: Mac はローカル直接実行）で
のみ有効という前提のもと、実装セッション開始時に以下を判定した（#567・#572 と同一のゲート判定手順）。

1. `uname -sm` → `Linux x86_64`（実測。本 worktree の開発環境）。
2. `docs/real-hardware-verification-env.local.md`（Git 管理外のローカル用ファイル）→ 不存在（実測）。
   M4 Max への到達経路が定義されていないため代替経路もなし。
3. `git log b96c3b3..main -- crates/backend-cpu` → 空（実測）。#567 が定めた規則（同ドキュメント
   §「ベースライン基準コミットの確定」）に従い、実測時点の `main` tip を HEAD として使用できることを
   確認した。本セッションの `main` tip は `2ae2959`。

**結論**: 本セッションでは M4 Max 実機に到達できないため、確定計測の記録ドキュメント（手順・記録枠・
候補下限導出手順）の整備までを実施し、**実測値の捏造・placeholder 値での完了扱いは行わない**
（fail-closed。#567・#572 先例と同方針）。実測は M4 Max 実機へ到達可能な後続セッション・Agent
（`bench-runner` 委譲想定）が引き継いで実施する。

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
を参照し、本ドキュメントでは二重管理しない。本セッション時点の再判定結果は上記「実行環境ゲート判定」
節のとおり `git log b96c3b3..main -- crates/backend-cpu` が空のため、実測時点の `main` tip
（本セッション: `2ae2959`。実機セッションでの計測実行時に再確認すること）を HEAD として使用する。

ベースライン比較（対 A-8・対 Phase E 改善率）は `cpu-gemm-phase-e-remeasurement.md` 側の責務であり、
本ドキュメントでは**対 PyTorch 比のみ**を扱う。

## 計測手順（Apple Silicon 実機）

```bash
git fetch origin
git checkout bench/574-cpu-optimized-remeasurement   # 本イシューの実装ブランチ
git log b96c3b3..main -- crates/backend-cpu           # 空であることを再確認（空でなければ b96c3b3 を使う）

# 1. 数値一致確認を先に行う（既存 parity テスト群。閾値は緩和しない）
cargo test -p backend-cpu --release --test gemm_blis_parity

# 2. Rust 側ベンチを 5 回独立実行し、size ごとに 5 run の中央値の中央値を採用する
#    （MeasurementConfig::default() 自体が warmup 20・iters 20・中央値を内包するため、
#    5 プロセス独立実行との組み合わせで「5 回計測の中央値」下限
#    〈.claude/rules/coding-rust.md〉を二重に満たす。#567 先例と同方式）
cargo test -p backend-cpu --release --test gemm_blis_perf \
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

## 計測環境（実測時に記入）

| 項目 | 値 |
|------|-----|
| チップ | （未計測） |
| OS | （未計測） |
| rustc | （未計測） |
| torch | （未計測） |
| 計測コミット SHA（HEAD） | （未計測。上記「SHA 規則」に従い実測時点で再確認） |
| 計測プロトコル | `bench_harness::protocol::run`（warmup 20・iters 20・中央値/Q1/Q3。TASK-8.1）を 5 回独立実行し size ごとに中央値の中央値を採用（Rust・PyTorch 双方） |
| 決定的シード | `Xorshift64Star`（`bench_harness::rng`。ハーネス内固定シード `3000+size`/`4000+size`） |
| 同期境界 | Rust: `gemm_blis_parallel` 呼び出し完了（CPU 同期処理のため追加の完了待ちは不要）／PyTorch: スクリプト内計測ループ完了 |

## 対 PyTorch 比 結果

| size | Rust median TFLOPS（5 run の中央値） | Rust Q1/Q3 TFLOPS | PyTorch median TFLOPS（5 run の中央値） | 対 PyTorch 比 |
|------|------|------|------|------|
| 512（参考値） | （未計測） | （未計測） | （未計測） | （未計測） |
| 1024（参考値） | （未計測） | （未計測） | （未計測） | （未計測） |
| 2048 | （未計測） | （未計測） | （未計測） | （未計測） |
| 4096 | （未計測） | （未計測） | （未計測） | （未計測） |

判定対象形状（REQ-8「判定対象形状」節）は **M=N=K=2048・4096 の実測比率の最小値**。512/1024 は参考値。

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

候補下限値（参考算出）: （未計測）

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

## 状態: 未計測。実機セッションで消化

本ドキュメントは Linux worktree で作成され、M4 Max 実機が同一セッションで使用できないため計測手順・
記録テンプレートのみを整備した（#567・#572 先例と同方式）。実機到達可能なセッションが「計測手順」
節の手順で計測し、上記「対 PyTorch 比 結果」「計測環境」「候補下限値導出手順」の各節を実測値で
埋めること。

内部ホスト名等の実値は書かない（#461 のプレースホルダ方針。実測時の原文は
`docs/real-hardware-verification-env.local.md` へ記録する）。

## 動作確認（Linux セッションで実施済み）

- `cargo fmt --all --check`
- `cargo clippy --workspace --all-targets --all-features -- -D warnings`
- `cargo test --workspace`（Linux 実行分。実機依存・`#[ignore]` テストは除外）
- `cargo test -p backend-cpu --test gemm_blis_perf`（`--ignored` を付けない通常実行でハーネスの
  コンパイル成立を確認。x86_64 上で `--ignored` 実測は行わない）
- `git diff --stat` で `crates/bench-harness/src/threshold.rs`・数値一致 tolerance 定数・
  `docs/spec/`・`guardrail.toml`・`Cargo.toml`／`Cargo.lock` に差分がないことを確認

## 未実施・後続作業

- **M4 Max 実機での実測**: 「状態」節のとおり本イシューでは未実施。実機セッションへ申し送る。
  `cpu-gemm-baseline-remeasurement.md`（A-8）・`cpu-gemm-phase-e-remeasurement.md`（E-10）の記録枠
  も同一セッションで併せて記入するのが効率的（Rust 側は同一ハーネス・PyTorch 側は同一スクリプトの
  ため、1 セットの追加計測〈Phase E HEAD 分〉で複数ドキュメントの表を同時に満たせる可能性がある）。
- **候補下限値の最終確定・REQ-8 反映判断**: #577（人間承認）が実測完了後に対応する。
- **E-7（明示プリフェッチ）の採否・E-8（MC/KC/NC 実機スイープ）**: 既存の保留判断のまま本イシュー
  では扱わない。
- **`docs/performance-targets.md` 本体の更新**: #579（最終整合タスク）の担当範囲。

## Appendix: 全 run 生ログ

実測完了後、Rust 側 5 run × 4 形状、PyTorch 側 5 run × 4 形状（既存値の再利用がない場合）の生ログを
ここへ記録する。
