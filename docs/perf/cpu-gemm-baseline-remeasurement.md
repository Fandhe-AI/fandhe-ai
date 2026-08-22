# CPU GEMM 現行 NEON 経路の対 PyTorch ベースライン再計測（#488・A-8）

イシュー #488「現行 NEON 経路の対 PyTorch ベースライン再計測（A-1 確定実機）」の実測記録。
`docs/perf/gemm-optimization-baseline.md` §1 CPU 行が指摘する通り、REQ-8 実測比率 5.3%
（`docs/performance-targets.md` §2）は **PoC-v2-1 時点の SIMD 未適用の旧経路**の値であり、現行の
本番演算経路 `gemm_blis_parallel`（BLIS 5-loop・NEON/AVX2/AVX-512 マイクロカーネル dispatch。
`crates/backend-cpu/src/ops.rs:67` から呼ばれる）の対 PyTorch 比は本ドキュメント作成時点で未確定
だった。本ドキュメントは A-1（#481・マージ済み PR #633）が確定した基準実機（Apple M4 Max・
PyTorch 2.13.0 macOS arm64）でこれを再計測し、Phase E（#564 等）の改善率の分母を確定する。

**本ドキュメントは REQ-8 の下限値を一切変更しない**（下限値の変更は Phase F #577 の人間承認スコープ）。

## 状態: 実測完了（2026-08-18・Apple M4 Max 実機）

### 実行環境ゲート判定

#### 旧セッション（Linux x86_64・実測未達）

計測は Apple M4 Max 実機（`docs/real-hardware-verification-env.md` §7: Mac はローカル直接実行）でのみ
有効という前提のもと、実装セッション開始時に以下を判定した:

1. `uname -sm` → `Linux x86_64`（QEMU 仮想環境。`Darwin arm64` ではない）。
2. `docs/real-hardware-verification-env.local.md`（Git 管理外のローカル用ファイル）→ 不存在。Mac への
   到達経路が定義されていないため、代替経路も不可。

**結論**: 当該セッションでは M4 Max 実機に到達できないため、計測ハーネス整備・本ドキュメントの手順／
環境節整備・x86 でのスモーク検証（フォーマット確認）までを実施し、**実測値の捏造・placeholder 値での
完了扱いは行わない**（fail-closed。共通契約 §7「秘密情報」に近接する fail-safe 原則を踏襲）。
実測は M4 Max 実機へアクセス可能な後続セッション・Agent（`bench-runner` 委譲想定）が引き継いで
実施する方針とした。

#### 実測セッション（2026-08-18・Apple M4 Max ローカル直接実行）

1. `uname -sm` → `Darwin arm64`（実測。M4 Max 実機に到達）。
2. PyTorch 側再現コマンドの引数順序（`<size> <warmup> <iters>`）を実機で検証し、想定どおりであることを
   確認した。Rust 側 5 run・PyTorch 側 5 run × 4 形状の実測を実施した。

Phase A 親 #480 は本来「implement-issue-tree に載せず main が手動で消化する」区分であることも
併記する。

## 計測対象

- **Rust 側**: `gemm_blis_parallel`（公開入口。aarch64 では `dispatch_region`（
  `crates/backend-cpu/src/gemm_blis/mod.rs:360-365`）が無条件に `NeonKernel` を選択するため、
  M4 Max では追加の `RUSTFLAGS` 指定なしで NEON 経路になる）。
- **PyTorch 側**: `torch.matmul`（CPU・f32）。PoC-v2-1 と同一スクリプト
  `docs/spec/03-poc/poc-v2-1-tensor-cpu-gemm/code/pytorch/gemm_bench_torch_cpu.py` を**読み取り実行
  のみ**で再利用する（`docs/spec/` submodule は編集しない）。

## 計測環境（実測値）

| 項目 | 値 |
|---|---|
| CPU | Apple M4 Max |
| macOS 版 | 26.6.1 |
| 論理コア数 | 16 |
| rustc 版 | 1.96.0 (ac68faa20 2026-05-25) |
| PyTorch 版（`torch.__version__`） | 2.13.0（macOS arm64） |
| `torch.get_num_threads()` | 12 |
| PyTorch BLAS | `BLAS_INFO=accelerate`（Apple Accelerate/vecLib。AMX 経路を含みうる。`torch.__config__.show()` 実測） |
| venv 構成 | `python3 -m venv` + `pip install torch==2.13.0 numpy`（リポジトリ管理外・コミットしない） |

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
cargo test -p fandhe-ai-backend-cpu --release --test gemm_blis_perf \
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
検出保険とする。**この `assert_eq!` は M=N=K=4096・NEON 経路では旧セッション（Linux x86_64）時点で
未実施の組み合わせだった**（`tests/gemm_blis_parity.rs` の K=4096 ケース〈`gemm_blis_uses_mul_add_fma_contract`〉
は M=N=8・K=4096 の細長い形状であり、4096 立方の bit 一致は同ファイルの
`gemm_blis_matches_naive_bit_exact_shape_grid`／`gemm_blis_parallel_matches_naive_bit_exact` 等でも
カバーされていない）。**M4 Max 実機でこの `assert_eq!` が失敗した場合、それは「計測ハーネスの不具合」
ではなく「NEON 経路の bit 一致契約に関する新規の parity 上の発見」として扱う**方針だった（本ドキュメント
の Appendix・Issue コメントに記録して報告する。assert を削除する・許容誤差を緩めることで回避しない。
`.claude/rules/coding-rust.md`「バックエンド間数値一致テストの許容誤差を単独で緩和しない」の趣旨に
準ずる）。**実測セッション（2026-08-18）でベースライン 5 run・HEAD 5 run の全 10 run で pass を確認し
（下記「実測結果」節）、この 4096 立方・NEON 経路の bit 一致契約が実機で初めて実証された（新規の
parity 上の発見であり、ハーネスの不具合ではない）。**

上記コマンドを 5 回実行しログを保存する。

### PyTorch 側

旧セッション（Linux x86_64）の worktree では `docs/spec` submodule が未初期化（`git submodule status`
が `-44c5e6271ad679e7f4822528b9dec616768ceeaa docs/spec` を返し、
`docs/spec/03-poc/poc-v2-1-tensor-cpu-gemm/code/pytorch/` が実在しない）のため、下記コマンド例の
引数順序（`<size> <warmup> <iters>` の位置引数と仮定）・`torch.__version__` や
`torch.get_num_threads()` の出力有無は未検証だった。**実測セッション（2026-08-18）で submodule
初期化後にスクリプトを実行し、引数順序（想定どおり）・`torch.__version__`（2.13.0）・
`torch.get_num_threads()`（12）の出力を確認した。**

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

## 実測結果（2026-08-18・Apple M4 Max 実機。ベースライン SHA `1cb2938`）

| 形状 | Rust median TFLOPS（5 run 中央値・Q1/Q3） | PyTorch median TFLOPS（5 run 中央値・Q1/Q3） | 対 PyTorch 比率 |
|---|---|---|---|
| 512（参考値） | 0.355661（0.336210 / 0.368434） | 2.6040（2.5434 / 2.8506） | 13.7% |
| 1024（参考値） | 0.530036（0.515468 / 0.537919） | 3.1666（3.0805 / 3.1789） | 16.7% |
| 2048 | 0.577056（0.558697 / 0.593423） | 3.2375（3.2029 / 3.2446） | 17.8% |
| 4096 | 0.663931（0.639499 / 0.674484） | 3.0101（2.8946 / 3.0404） | 22.1% |

`gemm_blis_parity`（16 passed; 0 failed）は HEAD SHA `b96c3b3` の worktree でのみ実行した（ベースライン
SHA `1cb2938` 側の別 worktree では未実行）。ベースライン側の数値正当性は、各 perf run 内でハーネスが
計測区間外に実行する `gemm_parallel`（参照実装）との bit 一致 `assert_eq!` で担保する。この
`assert_eq!` はベースライン 5 run・HEAD 5 run の**全 10 run で pass** し、A-8 §「計測手順」が
予告していた M=N=K=4096・NEON 経路の bit 一致（本イシュー実装セッション時点で未実施だった組合せ）が
実機で初めて確認された。異常 run（明確な原因のある外れ値）はなし。

## Phase E 分母の確定

2048/4096 の比率最小値 **17.8%（size=2048）** を Phase E（#564 等）改善率の**対 PyTorch 比の分母**
として本節に明記する（`cpu-gemm-phase-e-remeasurement.md` の「改善率（HEAD ÷ ベースライン）」列は
これとは別に baseline median TFLOPS〈0.577056 等〉を分母とする点に注意。「対 PyTorch 比の分母」と
「改善率の分母」は異なる量である）。512 は起動オーバーヘッド支配のため参考値扱いとし分母に使わない
（`docs/perf/gemm-optimization-baseline.md` §1 表と同方針）。

**関連ドキュメント（#567）**: Phase E（E-1〜E-8）完了時点の対 PyTorch 比再計測は
`docs/perf/cpu-gemm-phase-e-remeasurement.md` に記録する（本ドキュメントと同じく Linux x86_64
セッションでは環境ゲート未達のため未実施）。同ドキュメントはベースライン基準コミットを本ドキュメント
の計測ハーネス導入コミット（PR #650・`1cb2938`）と同一に定めているため、**実機セッションでは
本ドキュメントの表と `cpu-gemm-phase-e-remeasurement.md` の「対 A-8 ベースライン改善率」表を同一の
ベースライン計測（Rust 5 run × 4 形状・PyTorch 5 run × 4 形状）1 セットで同時に埋められる**（二重
計測を避けるため、実機セッションでは両ドキュメントをまとめて更新することを推奨する）。

## PoC-v2-1（5.3%）との対比

本ドキュメントの現行経路の値（対 PyTorch 比 17.8%、size=2048）は、PoC-v2-1 旧経路の 5.3%
（`docs/performance-targets.md:25`）に対し**約 3.4 倍**（17.8 / 5.3 ≈ 3.36）の改善である。SIMD 適用
（NEON マイクロカーネル・BLIS 5-loop）による改善が確認できた。**REQ-8 下限値・
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

ベースライン SHA `1cb2938`。Rust 側 5 run × 4 形状、PyTorch 側 5 run × 4 形状（1 行形式）。

### Rust 側（5 run）

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

### PyTorch 側（5 run。`cpu-gemm-phase-e-remeasurement.md`・`cpu-gemm-optimized-remeasurement.md` と共用）

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
