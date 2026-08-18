# CPU GEMM Phase E 完了時点の対 PyTorch 比 再計測・記録（#567）

イシュー #567「Phase E 完了時点の対 PyTorch 比を再計測・記録」の実測記録。GEMM 最適化ツリー
（ルート #479）の Phase E（NEON マイクロカーネル・packing・キャッシュブロッキング再設計。
E-1〜E-8: #552・#554・#556・#557・#559・#561・#562・#564）が全て closed した時点でのスナップショット
計測であり、A-8（#488。`docs/perf/cpu-gemm-baseline-remeasurement.md`）で確定した基準系列に対する
改善率と、A-1（#481。`docs/perf/gemm-optimization-baseline.md`）由来の対 PyTorch 比の再確定を行う。

**本ドキュメントは REQ-8 の下限値（`docs/performance-targets.md` §2）を一切変更しない**（下限値の
変更は Phase F の人間承認タスク #577 へ申し送る。`cpu-gemm-baseline-remeasurement.md` §0 と同方針）。

## 状態: 実測完了（2026-08-18・Apple M4 Max 実機）

### 実行環境ゲート判定

#### 旧セッション（Linux x86_64・実測未達）

計測は Apple M4 Max 実機（`docs/real-hardware-verification-env.md` §7: Mac はローカル直接実行）
でのみ有効という前提のもと、実装セッション開始時に以下を判定した（A-8・Metal #547 と同一のゲート
判定手順）:

1. `uname -sm` → `Linux x86_64`（実測。本 worktree の開発環境）。
2. `docs/real-hardware-verification-env.local.md`（Git 管理外のローカル用ファイル）→ 不存在（実測）。
   M4 Max への到達経路が定義されていないため代替経路もなし。

**結論**: 当該セッションでは M4 Max 実機に到達できないため、Phase E 反映状況の切り分け・計測手順と
記録テンプレートの整備までを実施し、**実測値の捏造・placeholder 値での完了扱いは行わない**
（fail-closed。A-8・Metal #547 先例と同方針）。実測は M4 Max 実機へ到達可能な後続セッション・Agent
（`bench-runner` 委譲想定）が引き継いで実施する方針とした。

#### 実測セッション（2026-08-18・Apple M4 Max ローカル直接実行）

1. `uname -sm` → `Darwin arm64`（実測。M4 Max 実機に到達）。
2. SHA 規則の再判定（`git log b96c3b3..origin/main -- crates/backend-cpu`）は非空（`b054530` #718）
   だったため、`b96c3b3` を HEAD SHA として使用した（下記「ベースライン基準コミットの確定」節参照）。
3. ベースライン SHA `1cb2938`・HEAD SHA `b96c3b3` の 2 点 × Rust 5 run × 4 形状、PyTorch 5 run ×
   4 形状の実測を実施した。

## Phase E 変更の本番経路反映状況（ソースコード実測による切り分け）

`git log --oneline main -- crates/backend-cpu` と `crates/backend-cpu/src/gemm_blis/` の現状
（本ドキュメント作成コミット時点。x86_64 環境でのソース確認）を突合した表。

| 子イシュー | 内容 | 本番経路への適用状態 | 根拠 |
|---|---|---|---|
| E-1（#552） | NEON マイクロカーネルの A オペランドを `vfmaq_laneq_f32` 化（broadcast 排除） | **適用済み** | PR #686・コミット `9ba451d`（`neon.rs` の既定 `kernel`） |
| E-2（#554） | `pack_a`/`pack_b` の中間 `Vec` 確保を廃し panel バッファへ直接書き込み | **適用済み（A-8 ベースラインに既に含まれる）** | PR #644・コミット `76d89fe`。A-8 計測コミット（`1cb2938`・PR #650）より**前**にマージ済み |
| E-3（#556） | packing バッファを gemm 呼び出し全体で 1 回確保して再利用 | **適用済み（A-8 ベースラインに既に含まれる）** | PR #645・コミット `58f5d5d`。同上、A-8 計測より前にマージ済み |
| E-4（#557） | 非端タイルで C への直接ロード/ストアに切り替え | **適用済み** | PR #691・コミット `a74f28d` |
| E-5（#559） | NEON マイクロカーネルを MR=8, NR=12（24 accumulator）へ拡張 | **適用済み** | PR #693・コミット `82d86e1`。`neon.rs` の既定 `pub const MR: usize = 8; pub const NR: usize = 12;`（`microkernel::NeonKernel` が使用）で確認済み。12×8 対抗変種（`Neon12x8Kernel`／`MR_12X8=12, NR_12X8=8`）は `#[cfg(test)]` A/B 比較専用でありディスパッチ経路には接続されていない |
| E-6（#561） | NEON マイクロカーネルに k=4 アンロール＋ソフトウェアパイプラインを導入 | **適用済み** | PR #695・コミット `bd216cc`。`neon.rs::compute` の `k_main = kc_len - (kc_len % 4)` 主ループ・端数ループ分離で確認済み |
| E-7（#562） | 明示プリフェッチの導入（A-9 の結論に従う） | **未適用（承認ゲート未成立）** | PR #697・コミット `26eba5c` はドキュメントのみ（`docs/cpu-gemm-prefetch-decision.md`）。stable rustc に `core::arch::aarch64` の安定プリフェッチ intrinsic が存在せず、唯一到達可能な inline asm（`unsafe`）導入の採否をユーザー承認事項として保留中。`grep -rn prefetch crates/backend-cpu` は 0 件（実測） |
| E-8（#564） | MC/KC/NC を参照実装値の近傍で実機スイープして再選定 | **スイープ基盤のみ適用・選定値は現行維持** | PR #701・コミット `bae1f0f`。`crates/backend-cpu/src/gemm_blis/mod.rs:99-103` で `MC=128`/`KC=256`/`NC=512`（変更なし）を確認済み。実機スイープテスト自体は `#[ignore]` で追加済み |

**まとめ**: 本計測（HEAD）は A-8 ベースライン（E-2・E-3 適用済み状態）に対し、E-1・E-4・E-5・E-6 の
4 件が追加適用された経路の実測となる（E-7 は未適用、E-8 は選定値変更なしのため経路に影響しない）。

## 計測プロトコル

- 各計測は `bench_harness::protocol::run`（`MeasurementConfig::default()` = warmup 20・iters 20・
  中央値／Q1/Q3）を用いる（A-8・`docs/performance-targets.md` §4 と同一プロトコル）。
- **同一コマンドを 5 回実行（run1〜run5）し、形状ごとに 5 run の中央値の中央値を採用値とする**
  （イシューの「5 回計測中央値」・`.claude/rules/coding-rust.md`「ベンチは 5 回計測の中央値」・
  A-8 §「計測プロトコル」と同一方式）。生ログは Appendix に全 run 記録する。
- 判定・改善率の主対象は **2048/4096**（512 は起動オーバーヘッド支配のため参考値、1024 は中間
  参考値。A-8・`docs/perf/gemm-optimization-baseline.md` §1 の整理を踏襲）。
- 決定的シード（xorshift64*・`crates/bench-harness/src/rng.rs`）を用いる。
- 既存ハーネス `crates/backend-cpu/tests/gemm_blis_perf.rs::gemm_blis_baseline_pytorch_square_512_to_4096`
  （A-8・PR #650 で整備済み。`#[ignore]`）をそのまま再利用する。本イシューでは演算経路・ハーネスとも
  無変更（受け入れ基準の「既存出力形式で不足がある場合のみ最小追記」に該当する不足は確認されな
  かった）。

## ベースライン基準コミットの確定

**ベースライン SHA**: `1cb2938`（`bench(backend-cpu): 現行 NEON 経路の対 PyTorch ベースライン
再計測（A-1 確定実機） (#650)`。2026-08-16 マージ）。

選定根拠: A-8 ハーネス導入コミットそのものであり、`git log --reverse --oneline -- crates/backend-cpu`
で確認した時点で「E-2・E-3 は適用済み、E-1・E-4・E-5・E-6・E-8 はいずれも未適用」の状態と一致する
（上表参照）。`1cb2938` の直後（`8ce844f` 等）には backend-cpu 以外の変更のみが挟まり、最初の
Phase E 本番コード変更（E-1・PR #686・`9ba451d`）はその後に続く。よって `1cb2938` を「Phase E 適用前
（A-8 ベースライン相当）」の代表コミットとして採用する。

**HEAD SHA（Phase E 完了時点）**: `b96c3b3`（本ドキュメント作成コミットの親。本イシュー実装セッション
の作業ブランチ分岐元 `origin/main` の tip）。

再現用の worktree 分離コマンド例（実機セッションでの利用を想定）:

```bash
git worktree add /tmp/cpu-gemm-e-baseline 1cb2938
git worktree add /tmp/cpu-gemm-e-head b96c3b3
```

`b96c3b3` を正典の HEAD SHA とする。実測セッション開始時点の `main` が `b96c3b3` より進んでいる
場合は `git log b96c3b3..main -- crates/backend-cpu` を実行し、**空である場合に限り**その時点の
`main` を代わりに使ってよい（backend-cpu に無関係な変更のみが積まれている場合）。空でない場合
（Phase F 等の後続 backend-cpu 変更が既に main に入っている場合）は `b96c3b3` を使う（Phase E
帰属の改善率が後続変更の効果と混ざるのを防ぐため）。

## 計測手順・再現コマンド

### Rust 側（ベースライン SHA・HEAD の 2 点）

各 worktree で以下を 5 回実行する:

```bash
cargo test -p backend-cpu --release --test gemm_blis_perf \
    -- --ignored gemm_blis_baseline_pytorch_square_512_to_4096 --nocapture
```

出力は 1 行形式（TFLOPS = 2·N³ / median_secs / 1e12）。

計測ループ外で 1 回だけ `gemm_parallel`（参照実装）との `assert_eq!`（bit 一致契約）が実行される。
**この `assert_eq!` が M4 Max 実機・NEON 経路で失敗した場合、それは「計測ハーネスの不具合」ではなく
「NEON 経路の bit 一致契約に関する新規の parity 上の発見」として扱う**（A-8 §「計測手順」と同方針。
本ドキュメントの Appendix・イシューコメントに記録して報告する。assert 削除や tolerance 緩和で回避
しない。`.claude/rules/coding-rust.md`「バックエンド間数値一致テストの許容誤差を単独で緩和しない」
に準ずる）。

### PyTorch 側

旧セッション（Linux x86_64）の worktree では `docs/spec` submodule が未初期化（`git submodule status`
が `-44c5e6271ad679e7f4822528b9dec616768ceeaa docs/spec` を返し、
`docs/spec/03-poc/poc-v2-1-tensor-cpu-gemm/code/pytorch/` が実在しない）のため、下記コマンド例の
引数順序は未検証だった。**実測セッション（2026-08-18）で submodule 初期化後にスクリプトを実行し、
`<size> <warmup> <iters>` の位置引数順序で想定どおり動作することを確認した。**

```bash
python3 -m venv .venv-gemm-phase-e
source .venv-gemm-phase-e/bin/activate
pip install torch==2.13.0 numpy
git submodule update --init docs/spec  # 未初期化の場合のみ
python3 docs/spec/03-poc/poc-v2-1-tensor-cpu-gemm/code/pytorch/gemm_bench_torch_cpu.py <size> 20 20
```

`<size>` ∈ {512, 1024, 2048, 4096} それぞれについて 5 回実行しログを保存する。PyTorch 側は
ベースライン・HEAD で演算経路が変わらない（`torch.matmul` のみ）ため、A-8 のセッションと同一実機・
同一 PyTorch 版であれば **セッション内 1 回の計測で両方の比率算出に共用してよい**（A-8 で既に
記録済みの値がある場合はそれを再利用し、二重計測を避ける）。

計測中は他負荷の混入に注意し、異常 run（外れ値の明確な原因があるもの）は破棄・取り直しを記録に
残す。

## 実測結果（2026-08-18・Apple M4 Max 実機。計測環境は下記「計測環境」参照）

### 計測環境

| 項目 | 値 |
|---|---|
| チップ | Apple M4 Max |
| OS | macOS 26.6.1 |
| 論理コア数 | 16 |
| rustc | 1.96.0 (ac68faa20 2026-05-25) |
| torch | 2.13.0（macOS arm64） |
| `torch.get_num_threads()` | 12 |
| PyTorch BLAS | `BLAS_INFO=accelerate`（Apple Accelerate/vecLib。AMX 経路を含みうる） |
| venv 構成 | `python3 -m venv` + `pip install torch==2.13.0 numpy`（リポジトリ管理外） |
| ベースライン SHA | `1cb2938` |
| HEAD SHA | `b96c3b3` |
| parity（`gemm_blis_parity`、HEAD） | 16 passed; 0 failed（HEAD `b96c3b3` の worktree でのみ実行。ベースライン SHA `1cb2938` 側は未実行） |

ベースライン側（`1cb2938`）の数値正当性は、各 perf run 内でハーネスが計測区間外に実行する
`gemm_parallel`（参照実装）との bit 一致 `assert_eq!` で担保する。この `assert_eq!` はベースライン
5 run・HEAD 5 run の**全 10 run で pass** し、A-8 §「計測手順」が予告していた M=N=K=4096・NEON 経路
の bit 一致（未実施だった組合せ）が実機で初めて確認された（詳細は `cpu-gemm-baseline-remeasurement.md`
§「実測結果」参照）。

### 対 A-8 ベースライン改善率

| 形状 | ベースライン（`1cb2938`）median TFLOPS | Phase E（HEAD）median TFLOPS | 改善率（HEAD ÷ ベースライン） |
|---|---|---|---|
| 512（参考値） | 0.355661 | 0.470527 | 132.3% |
| 1024（参考値） | 0.530036 | 0.711176 | 134.2% |
| 2048 | 0.577056 | 0.800964 | 138.8% |
| 4096 | 0.663931 | 0.931332 | 140.3% |

### 対 PyTorch 比（REQ-8 分子・分母定義）

| 形状 | Rust median TFLOPS（Phase E・HEAD） | PyTorch median TFLOPS | 対 PyTorch 比率 |
|---|---|---|---|
| 512（参考値） | 0.470527 | 2.6040 | 18.1% |
| 1024（参考値） | 0.711176 | 3.1666 | 22.5% |
| 2048 | 0.800964 | 3.2375 | 24.7% |
| 4096 | 0.931332 | 3.0101 | 30.9% |

判定主対象（2048/4096）の比率最小値を Phase E 完了時点の対 PyTorch 比として本節に明記する
（`gemm-optimization-baseline.md` §1・A-8 と同方針）。**Phase E 完了時点の対 PyTorch 比は 24.7%
（size=2048）**。

異常 run の扱い: 明確な原因のある外れ値はなく、全 5 run を採用した（生ログは Appendix 参照）。

### PoC-v2-1（5.3%）との対比

Phase E 完了時点の対 PyTorch 比 24.7%（size=2048）は、PoC-v2-1 旧経路（SIMD 未適用）の 5.3%
（`docs/performance-targets.md:25`）に対し**約 4.7 倍**（24.7 / 5.3 ≈ 4.66）の改善である。SIMD 適用
（NEON マイクロカーネル・BLIS 5-loop）と Phase E 各改善（E-1・E-4・E-5・E-6）の累積効果により、
対 PyTorch 比が旧経路比で大幅に向上したことを確認した。

## REQ-8 下限値との関係（変更しない）

**本節は REQ-8 下限値（`docs/performance-targets.md` §2 の CPU 行）を一切変更しない**。改善率・
対 PyTorch 比の算出結果を下限値へ反映する判断は Phase F の人間承認タスク（#577）へ申し送る
（`cpu-gemm-baseline-remeasurement.md` §0、Metal #547 §「REQ-8 下限値との関係」と同方針）。

## 共通契約の遵守

- **境界チェック不省略**: 本イシューは計測・ドキュメントのみで `gemm_blis_parallel`・NEON
  マイクロカーネルの手動境界チェックは変更していない。
- **tolerance 不緩和**: バックエンド間数値一致テストの許容誤差は変更していない。既存ハーネスの
  `assert_eq!`（bit 一致契約）もそのまま。
- **依存追加なし**: `Cargo.toml`／`Cargo.lock` に変更なし（PyTorch venv はリポジトリ管理外・計測専用）。
- **`docs/spec/` 不編集**: PoC-v2-1 の PyTorch スクリプトは読み取り実行のみで submodule への変更は
  行っていない。
- **REQ-8 下限値不変更**: `docs/performance-targets.md` §2 の値は本イシューでは変更していない。

## スコープ外・申し送り

- **M4 Max 実機での実測**（ベースライン SHA `1cb2938`・HEAD SHA `b96c3b3` の 2 点 × Rust 5 run ×
  4 形状、PyTorch 5 run × 4 形状）: 2026-08-18 に実施済み。
- **`cpu-gemm-baseline-remeasurement.md` の実測値記入**: 本イシューと同一実機セッションで併せて
  記入済み（両ドキュメントとも Rust 側は同一ハーネス・PyTorch 側は同一スクリプトのため、
  ベースライン SHA 側の 1 セットの計測で両ドキュメントの表を同時に満たした）。
- **`docs/perf/cpu-gemm-optimized-remeasurement.md`（#574・Phase F）の実測値記入**: 本ドキュメントの
  HEAD 計測と対 PyTorch 比の判定対象形状が重なるため、同一 M4 Max 実機セッションで併せて記入済み
  （Rust・PyTorch とも同一ハーネス・同一スクリプトのため二重計測を回避した）。
- **REQ-8 下限値の変更**: 本イシューでは行わない。Phase F の人間承認タスク（#577）へ申し送る。
- **E-7（明示プリフェッチ）の採否判断**: `docs/cpu-gemm-prefetch-decision.md` の保留判断のまま。
  本イシューのスコープ外。
- **E-8（MC/KC/NC）の実機スイープ・選定**: #564 の申し送り事項であり本イシューでは扱わない
  （現行値 MC=128/KC=256/NC=512 のまま計測した）。

## Appendix: 全 run 生ログ

ベースライン SHA `1cb2938`・HEAD SHA `b96c3b3` それぞれについて Rust 側 5 run × 4 形状、PyTorch 側
5 run × 4 形状の生ログ（1 行形式）。PyTorch 側はベースライン・HEAD で共用（`torch.matmul` のみで
演算経路が変わらないため）。

### Rust 側（ベースライン `1cb2938`、5 run）

```text
run1: size=512 median_tflops=0.354760 q1_tflops=0.350762 q3_tflops=0.362649 median_secs=0.000757
run1: size=1024 median_tflops=0.551721 q1_tflops=0.530254 q3_tflops=0.560250 median_secs=0.003892
run1: size=2048 median_tflops=0.594819 q1_tflops=0.561821 q3_tflops=0.629246 median_secs=0.028882
run1: size=4096 median_tflops=0.648539 q1_tflops=0.633239 q3_tflops=0.682169 median_secs=0.211921
run2: size=512 median_tflops=0.355661 q1_tflops=0.336210 q3_tflops=0.368434 median_secs=0.000755
run2: size=1024 median_tflops=0.540146 q1_tflops=0.519526 q3_tflops=0.549205 median_secs=0.003976
run2: size=2048 median_tflops=0.578491 q1_tflops=0.549575 q3_tflops=0.600689 median_secs=0.029698
run2: size=4096 median_tflops=0.664804 q1_tflops=0.643260 q3_tflops=0.675550 median_secs=0.206736
run3: size=512 median_tflops=0.367049 q1_tflops=0.359491 q3_tflops=0.375325 median_secs=0.000731
run3: size=1024 median_tflops=0.529252 q1_tflops=0.513942 q3_tflops=0.537403 median_secs=0.004058
run3: size=2048 median_tflops=0.577056 q1_tflops=0.558697 q3_tflops=0.593423 median_secs=0.029772
run3: size=4096 median_tflops=0.671800 q1_tflops=0.648174 q3_tflops=0.682826 median_secs=0.204583
run4: size=512 median_tflops=0.371515 q1_tflops=0.356528 q3_tflops=0.378634 median_secs=0.000723
run4: size=1024 median_tflops=0.467522 q1_tflops=0.463632 q3_tflops=0.520266 median_secs=0.004593
run4: size=2048 median_tflops=0.571542 q1_tflops=0.565606 q3_tflops=0.600703 median_secs=0.030059
run4: size=4096 median_tflops=0.663931 q1_tflops=0.639499 q3_tflops=0.674484 median_secs=0.207008
run5: size=512 median_tflops=0.342994 q1_tflops=0.334551 q3_tflops=0.353069 median_secs=0.000783
run5: size=1024 median_tflops=0.530036 q1_tflops=0.515468 q3_tflops=0.537919 median_secs=0.004052
run5: size=2048 median_tflops=0.567919 q1_tflops=0.534839 q3_tflops=0.584703 median_secs=0.030251
run5: size=4096 median_tflops=0.650659 q1_tflops=0.630356 q3_tflops=0.667101 median_secs=0.211231
```

### Rust 側（HEAD `b96c3b3`、5 run）

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

（`docs/perf/cpu-gemm-optimized-remeasurement.md` の Appendix「Rust 側（HEAD、5 run）」と同一の計測。
同一ハーネス実行の再利用のため二重計測ではない。）

### PyTorch 側（5 run。ベースライン・HEAD 双方に共用）

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

（同一実機・同一 venv・同一 `torch==2.13.0` での 1 回の計測をベースライン・Phase E（HEAD）・A-8 の
3 ドキュメントで共用。`docs/perf/cpu-gemm-optimized-remeasurement.md`・
`docs/perf/cpu-gemm-baseline-remeasurement.md` の Appendix と同一値。）

### parity（HEAD `b96c3b3`）

```text
running 16 tests
（全 16 test ok）
test result: ok. 16 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out
```
