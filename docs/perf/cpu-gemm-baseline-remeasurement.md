# CPU GEMM 現行 NEON 経路の対 PyTorch ベースライン再計測（#488・A-8）

イシュー #488「現行 NEON 経路の対 PyTorch ベースライン再計測（A-1 確定実機）」の実測記録。
`docs/perf/gemm-optimization-baseline.md` §1 CPU 行が指摘する通り、REQ-8 実測比率 5.3%
（`docs/performance-targets.md` §2）は **PoC-v2-1 時点の SIMD 未適用の旧経路**の値であり、現行の
本番演算経路 `gemm_blis_parallel`（BLIS 5-loop・NEON/AVX2/AVX-512 マイクロカーネル dispatch。
`crates/backend-cpu/src/ops.rs:67` から呼ばれる）の対 PyTorch 比は本ドキュメント作成時点で未確定
だった。本ドキュメントは A-1（#481・マージ済み PR #633）が確定した基準実機（Apple M4 Max・
PyTorch 2.13.0 macOS arm64）でこれを再計測し、Phase E（#564 等）の改善率の分母を確定する。

**本ドキュメントは REQ-8 の下限値を一切変更しない**（下限値の変更は Phase F #577 の人間承認スコープ）。

## 状態: 計測ハーネス整備・x86 スモーク検証まで完了。M4 Max 実機での実測は未実施（環境ゲートで未達）。
## PyTorch 側再現コマンドの引数順序も docs/spec submodule 未初期化のため未検証（下記「PyTorch 側」節参照）

### 実行環境ゲート判定（本イシュー実装セッション時点）

計測は Apple M4 Max 実機（`docs/real-hardware-verification-env.md` §7: Mac はローカル直接実行）でのみ
有効という前提のもと、実装セッション開始時に以下を判定した:

1. `uname -sm` → `Linux x86_64`（QEMU 仮想環境。`Darwin arm64` ではない）。
2. `docs/real-hardware-verification-env.local.md`（Git 管理外のローカル用ファイル）→ 不存在。Mac への
   到達経路が定義されていないため、代替経路も不可。

**結論**: 本セッションでは M4 Max 実機に到達できないため、計測ハーネス整備・本ドキュメントの手順／
環境節整備・x86 でのスモーク検証（フォーマット確認）までを実施し、**実測値の捏造・placeholder 値での
完了扱いは行わない**（fail-closed。共通契約 §7「秘密情報」に近接する fail-safe 原則を踏襲）。
実測（Rust 側 5 run・PyTorch 側 5 run × 4 形状）は M4 Max 実機へアクセス可能な後続セッション・
Agent（`bench-runner` 委譲想定）が引き継いで実施する。

Phase A 親 #480 は本来「implement-issue-tree に載せず main が手動で消化する」区分であることも
併記する。

## 計測対象

- **Rust 側**: `gemm_blis_parallel`（公開入口。aarch64 では `dispatch_region`（
  `crates/backend-cpu/src/gemm_blis/mod.rs:360-365`）が無条件に `NeonKernel` を選択するため、
  M4 Max では追加の `RUSTFLAGS` 指定なしで NEON 経路になる）。
- **PyTorch 側**: `torch.matmul`（CPU・f32）。PoC-v2-1 と同一スクリプト
  `docs/spec/03-poc/poc-v2-1-tensor-cpu-gemm/code/pytorch/gemm_bench_torch_cpu.py` を**読み取り実行
  のみ**で再利用する（`docs/spec/` submodule は編集しない）。

## 計測環境（実測時に記入）

| 項目 | 値 |
|---|---|
| CPU | Apple M4 Max（実測待ち） |
| macOS 版 | 実測待ち |
| 論理コア数 | 実測待ち |
| rustc 版 | 実測待ち |
| PyTorch 版（`torch.__version__`） | 実測待ち（想定: 2.13.0 macOS arm64） |
| `torch.get_num_threads()` | 実測待ち |
| venv 構成 | `python3 -m venv` + `pip install torch numpy`（リポジトリ管理外・コミットしない） |

## 計測プロトコル

- 各計測は `bench_harness::protocol::run`（`MeasurementConfig::default()` = warmup 20・iters 20・
  中央値／Q1/Q3）を用いる（`docs/performance-targets.md` §4 準拠。PoC-v2-1 分母と同一プロトコルで
  比較可能性を確保する）。
- そのうえで**同一コマンドを 5 回実行（run1〜run5）し、形状ごとに 5 run の中央値の中央値を採用値と
  する**（イシューの「5 回計測中央値」と `.claude/rules/coding-rust.md`「ベンチは 5 回計測の中央値」
  を同時に満たす。#381 の run1/run2/run3 方式の拡張）。生ログは Appendix に全 run 記録する。
- 判定・Phase E 改善率の分母は §4 に従い **2048/4096 を主対象**とする（512 は起動オーバーヘッド支配の
  ため参考値。1024 は中間参考値。A-1 表の整理を踏襲）。
- 決定的シード（xorshift64*・`crates/bench-harness/src/rng.rs`）を用いる。

## 計測手順・再現コマンド

### Rust 側

```bash
cargo test -p backend-cpu --release --test gemm_blis_perf \
    -- --ignored gemm_blis_baseline --nocapture
```

`crates/backend-cpu/tests/gemm_blis_perf.rs` の `gemm_blis_baseline_pytorch_square_512_to_4096`
（本イシューで追加。`#[ignore]` のため通常 CI・既定 `cargo test` 実行からは除外される）が
M=N=K ∈ {512, 1024, 2048, 4096} で `gemm_blis_parallel` を計測し、以下の 1 行形式で出力する:

```text
kernel=gemm_blis_parallel size=<n> median_tflops=<v> q1_tflops=<v> q3_tflops=<v> median_secs=<v>
```

TFLOPS = 2·N³ / median_secs / 1e12（正方形状 GEMM の浮動小数点演算数）。計測ループ外で 1 回だけ
`gemm_parallel`（TASK-1.6a 参照実装）と `assert_eq!`（bit 一致契約）を行い、計測対象取り違えの
検出保険とする。**この `assert_eq!` は M=N=K=4096・NEON 経路では本イシュー実装セッション時点で
未実施の組み合わせである**（`tests/gemm_blis_parity.rs` の K=4096 ケース〈`gemm_blis_uses_mul_add_fma_contract`〉
は M=N=8・K=4096 の細長い形状であり、4096 立方の bit 一致は同ファイルの
`gemm_blis_matches_naive_bit_exact_shape_grid`／`gemm_blis_parallel_matches_naive_bit_exact` 等でも
カバーされていない。加えて aarch64/NEON 経路自体、本セッションが x86_64 のため一度も実行されていない）。
**M4 Max 実機でこの `assert_eq!` が失敗した場合、それは「計測ハーネスの不具合」ではなく「NEON 経路の
bit 一致契約に関する新規の parity 上の発見」として扱う**（本ドキュメントの Appendix・Issue コメントに
記録して報告する。assert を削除する・許容誤差を緩めることで回避しない。`.claude/rules/coding-rust.md`
「バックエンド間数値一致テストの許容誤差を単独で緩和しない」の趣旨に準ずる）。

上記コマンドを 5 回実行しログを保存する。

### PyTorch 側

**注意（本イシュー実装セッション時点で未検証）**: `docs/spec` submodule は本セッションの worktree では
未初期化（`git submodule status` が `-44c5e6271ad679e7f4822528b9dec616768ceeaa docs/spec` を返し、
`docs/spec/03-poc/poc-v2-1-tensor-cpu-gemm/code/pytorch/` が実在しない）。そのため以下のコマンド例の
引数順序（`<size> <warmup> <iters>` の位置引数と仮定）・`torch.__version__` や
`torch.get_num_threads()` の出力有無はスクリプト本体を読めておらず未検証。M4 Max 実機セッションで
submodule 初期化後にスクリプト冒頭（`argparse`／`sys.argv` 定義箇所）を確認し、実際の呼び出し形式が
下記と異なる場合はこの節を実測値に合わせて訂正すること。

```bash
python3 -m venv .venv-gemm-baseline
source .venv-gemm-baseline/bin/activate
pip install torch==2.13.0 numpy
git submodule update --init docs/spec  # 未初期化の場合のみ
python3 docs/spec/03-poc/poc-v2-1-tensor-cpu-gemm/code/pytorch/gemm_bench_torch_cpu.py <size> 20 20
```

`<size>` ∈ {512, 1024, 2048, 4096} それぞれについて上記コマンドを 5 回実行しログを保存する。
スクリプトが `torch.__version__`・`torch.get_num_threads()` を出力しない場合は、同一 venv 内で
別途 `python3 -c "import torch; print(torch.__version__, torch.get_num_threads())"` 等で取得し
記録する。venv はリポジトリ管理外（`.venv*/` はコミットしない）。

計測中は他負荷の混入に注意し、異常 run（外れ値の明確な原因があるもの）は破棄・取り直しを記録に
残す。

## 実測結果（未実施。M4 Max 実機セッションで記入する）

| 形状 | Rust median TFLOPS（5 run 中央値・Q1/Q3） | PyTorch median TFLOPS（5 run 中央値・Q1/Q3） | 対 PyTorch 比率 |
|---|---|---|---|
| 512（参考値） | 実測待ち | 実測待ち | 実測待ち |
| 1024（参考値） | 実測待ち | 実測待ち | 実測待ち |
| 2048 | 実測待ち | 実測待ち | 実測待ち |
| 4096 | 実測待ち | 実測待ち | 実測待ち |

## Phase E 分母の確定（未実施）

実測完了後、2048/4096 の比率最小値を Phase E（#564 等）改善率の分母として本節に明記する。512 は
起動オーバーヘッド支配のため参考値扱いとし分母に使わない（`docs/perf/gemm-optimization-baseline.md`
§1 表と同方針）。

## PoC-v2-1（5.3%）との対比（未実施）

実測完了後、本ドキュメントの現行経路の値と PoC-v2-1 旧経路の 5.3%（`docs/performance-targets.md:25`）
を対比し、SIMD 適用（NEON マイクロカーネル・BLIS 5-loop）による改善の程度を記録する。**REQ-8 下限値・
`docs/performance-targets.md` §2 表の実測比率は本ドキュメントでは一切変更しない**（下限変更は
Phase F #577 の人間承認スコープ）。

## 共通契約の遵守

- **境界チェック不省略**: `gemm_blis_parallel`・NEON マイクロカーネルの手動境界チェックは変更していない
  （計測ハーネス追加のみで演算経路自体は無変更）。
- **tolerance 不緩和**: バックエンド間数値一致テストの許容誤差は変更していない。本ハーネスの
  `assert_eq!` は既存の bit 一致契約（`tests/gemm_blis_parity.rs` が網羅検証済み）を計測対象取り違え
  検出の保険として使うのみ。
- **依存追加なし**: `Cargo.toml`／`Cargo.lock` に変更なし（PyTorch venv はリポジトリ外・計測専用）。
- **`docs/spec/` 不編集**: PoC-v2-1 の PyTorch スクリプトは読み取り実行のみで、submodule への変更は
  行っていない。
- **REQ-8 下限値不変更**: `docs/performance-targets.md` §2 の値は本イシューでは変更していない。

## Appendix: 全 run 生ログ

実測完了後、Rust 側・PyTorch 側それぞれ 5 run × 4 形状の生ログをここへ記録する。
