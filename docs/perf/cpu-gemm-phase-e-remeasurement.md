# CPU GEMM Phase E 完了時点の対 PyTorch 比 再計測・記録（#567）

イシュー #567「Phase E 完了時点の対 PyTorch 比を再計測・記録」の実測記録。GEMM 最適化ツリー
（ルート #479）の Phase E（NEON マイクロカーネル・packing・キャッシュブロッキング再設計。
E-1〜E-8: #552・#554・#556・#557・#559・#561・#562・#564）が全て closed した時点でのスナップショット
計測であり、A-8（#488。`docs/perf/cpu-gemm-baseline-remeasurement.md`）で確定した基準系列に対する
改善率と、A-1（#481。`docs/perf/gemm-optimization-baseline.md`）由来の対 PyTorch 比の再確定を行う。

**本ドキュメントは REQ-8 の下限値（`docs/performance-targets.md` §2）を一切変更しない**（下限値の
変更は Phase F の人間承認タスク #577 へ申し送る。`cpu-gemm-baseline-remeasurement.md` §0 と同方針）。

## 状態: 計測手順・Phase E 反映状況の切り分け表・記録枠の整備まで完了。M4 Max 実機での実測は未実施（環境ゲートで未達）

### 実行環境ゲート判定（本イシュー実装セッション時点）

計測は Apple M4 Max 実機（`docs/real-hardware-verification-env.md` §7: Mac はローカル直接実行）
でのみ有効という前提のもと、実装セッション開始時に以下を判定した（A-8・Metal #547 と同一のゲート
判定手順）:

1. `uname -sm` → `Linux x86_64`（実測。本 worktree の開発環境）。
2. `docs/real-hardware-verification-env.local.md`（Git 管理外のローカル用ファイル）→ 不存在（実測）。
   M4 Max への到達経路が定義されていないため代替経路もなし。

**結論**: 本セッションでは M4 Max 実機に到達できないため、Phase E 反映状況の切り分け・計測手順と
記録テンプレートの整備までを実施し、**実測値の捏造・placeholder 値での完了扱いは行わない**
（fail-closed。A-8・Metal #547 先例と同方針）。実測（ベースライン SHA・HEAD の 2 点 × Rust 5 run ×
4 形状、PyTorch 5 run × 4 形状）は M4 Max 実機へ到達可能な後続セッション・Agent（`bench-runner`
委譲想定）が引き継いで実施する。

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

A-8 §「PyTorch 側」と同じ未検証事項が残る: 本セッションの worktree では `docs/spec` submodule が
未初期化（`git submodule status` が `-44c5e6271ad679e7f4822528b9dec616768ceeaa docs/spec` を返し、
`docs/spec/03-poc/poc-v2-1-tensor-cpu-gemm/code/pytorch/` が実在しない）。実機セッションで submodule
初期化後にスクリプト冒頭（`argparse`／`sys.argv` 定義箇所）を確認し、下記コマンド例の引数順序が
異なる場合は訂正すること。

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

## 実測結果（未実施。M4 Max 実機セッションで記入する）

### 対 A-8 ベースライン改善率

| 形状 | ベースライン（`1cb2938`）median TFLOPS | Phase E（HEAD）median TFLOPS | 改善率（HEAD ÷ ベースライン） |
|---|---|---|---|
| 512（参考値） | 実測待ち | 実測待ち | 実測待ち |
| 1024（参考値） | 実測待ち | 実測待ち | 実測待ち |
| 2048 | 実測待ち | 実測待ち | 実測待ち |
| 4096 | 実測待ち | 実測待ち | 実測待ち |

### 対 PyTorch 比（REQ-8 分子・分母定義）

| 形状 | Rust median TFLOPS（Phase E・HEAD） | PyTorch median TFLOPS | 対 PyTorch 比率 |
|---|---|---|---|
| 512（参考値） | 実測待ち | 実測待ち | 実測待ち |
| 1024（参考値） | 実測待ち | 実測待ち | 実測待ち |
| 2048 | 実測待ち | 実測待ち | 実測待ち |
| 4096 | 実測待ち | 実測待ち | 実測待ち |

判定主対象（2048/4096）の比率最小値を Phase E 完了時点の対 PyTorch 比として本節に明記する
（`gemm-optimization-baseline.md` §1・A-8 と同方針）。

### PoC-v2-1（5.3%）との対比（未実施）

実測完了後、本ドキュメントの Phase E 完了時点の値と PoC-v2-1 旧経路の 5.3%
（`docs/performance-targets.md:25`）を対比し、SIMD 適用（NEON マイクロカーネル・BLIS 5-loop）と
Phase E 各改善（E-1・E-4・E-5・E-6）による累積改善の程度を記録する。

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

- **M4 Max 実機での実測**（ベースライン SHA `1cb2938`・HEAD の 2 点 × Rust 5 run × 4 形状、PyTorch
  5 run × 4 形状）: 本セッションでは環境ゲート未達のため未実施。後続の実機到達可能セッション
  （`bench-runner` 委譲想定）へ引き継ぐ。
- **`cpu-gemm-baseline-remeasurement.md` の実測値記入**: 本イシューと同一実機セッションで併せて
  埋めるのが効率的（両ドキュメントとも Rust 側は同一ハーネス・PyTorch 側は同一スクリプトのため、
  ベースライン SHA 側の 1 セットの計測で両ドキュメントの表を同時に満たせる）。
- **`docs/perf/cpu-gemm-optimized-remeasurement.md`（#574・Phase F）の実測値記入**: 本ドキュメントの
  HEAD 計測と対 PyTorch 比の判定対象形状が重なるため、同一 M4 Max 実機セッションで併せて消化する
  のが効率的（Rust・PyTorch とも同一ハーネス・同一スクリプトのため二重計測を避けられる）。
- **REQ-8 下限値の変更**: 本イシューでは行わない。Phase F の人間承認タスク（#577）へ申し送る。
- **E-7（明示プリフェッチ）の採否判断**: `docs/cpu-gemm-prefetch-decision.md` の保留判断のまま。
  本イシューのスコープ外。
- **E-8（MC/KC/NC）の実機スイープ・選定**: #564 の申し送り事項であり本イシューでは扱わない
  （現行値 MC=128/KC=256/NC=512 のまま計測する）。

## Appendix: 全 run 生ログ

実測完了後、ベースライン SHA・HEAD それぞれについて Rust 側 5 run × 4 形状、PyTorch 側 5 run ×
4 形状の生ログをここへ記録する。
