# 候補判定の fandhe-ai 側 parity 非後退契約（ParityBaseline）への影響

## 1. 位置づけ

イシューツリー #1234（ルート）→ #1236（親）配下、前段イシュー #1237
（`docs/perf/candle-parity-tolerance-candidates.md`）に続く本 issue #1238
の成果物。#1237 は framework-compare の N=2048 candle 側 GEMM parity
「判定不能」（イシュー #1184 の fail 4 要素）に対し、候補判定
（スケール付き絶対誤差・ULP ベース）を現行複合判定へ OR 追加した場合の
fail 数を、要素単位のダンプ実値から算出した。本 issue はその**同じ候補
定義**を、fandhe-ai 本体側の parity 非後退契約
（`crates/backend-cuda/tests/common/parity_baseline.rs::BASELINES`・
`assert_no_parity_regression`。#491 系）へ適用した場合の影響を机上で
確認する。

**本 issue の範囲は事実（分類・fail_count の値域・同時更新箇所）の算出
までであり、推奨案・採否は #1239（決定記録 draft）が扱う**。tolerance
契約（`RELATIVE_TOLERANCE`／`ABSOLUTE_RESCUE_THRESHOLD`）・`BASELINES` の
値は本 issue では一切変更しない（変更にはユーザー承認が必須。イシュー
#1241）。

## 2. #1237 との違い（重要）

| | #1237（N=2048 candle 側 fail） | #1238（本 issue。`BASELINES`） |
|---|---|---|
| 入力データ | 要素単位のダンプ実値（fail した 2〜4 要素の `d`・`max_ab` 等） | 行単位の集計値（`fail_count`・`mean_abs_diff_ceiling`・`max_abs_diff_ceiling` 等）のみ。要素単位のダンプは存在しない |
| 入力分布 | U[-0.5,0.5)（`M ≈ 0.25`） | `Xorshift64Star::fill_vec`/`fill_vec_f16` の `[-1,1)` 一様分布（`M` は行ごとに実測） |
| 判定可能な粒度 | 個々の fail 要素が救済されるか否かを直接判定できる | 行単位の 3 クラス分類（no-op／全救済／部分・未確定）と `fail_count` の値域までしか判定できない |
| 対象行数 | 1 条件（N=2048 正方・candle cuda/cpu の 2 device） | 45 行（`BASELINES` 全行） |

## 3. 入力データと評価モデル

### 3.1 入力

- `crates/backend-cuda/tests/common/parity_baseline.rs::BASELINES`（45 行。
  `ParityPath` 別内訳: `WmmaTf32`×9・`WmmaTf32Opt`×9・`WmmaTf32Staged`×8・
  `MmaTf32`×11・`MmaTf32VsWmmaStaged`×2・`SpecializedMmaF16`×3・`WmmaF16`×2・
  `MmaF16`×1。K の集合 = {4, 8, 17, 19, 20, 32, 33, 36, 64, 65, 72, 128,
  256, 512, 1024, 4096}）
- **注意（単純な grep 出現数カウントとの食い違い）**: `grep -c
  "ParityBaseline {" crates/backend-cuda/tests/common/parity_baseline.rs`
  は 46 を返すが、これは `pub struct ParityBaseline { … }`（型定義自体）
  の 1 件を含んだ数である。実際の `BASELINES` 配列のエントリ数（実行時に
  意味を持つ行数）は **45**（上記内訳の合計）であり、本スクリプトは配列
  区間限定の構造的パースでこれを確認している（`parity_baseline_impact.py`
  の fail-closed 自己検査・`parity_baseline_impact_test.py::
  RealFileSmokeTest`）
- 入力生成コード（`fill_vec`/`fill_vec_f16` の呼び出しを file:line で確認
  済み）: `crates/backend-cuda/tests/gemm_wmma_tf32.rs:27-29`・
  `tests/gemm_wmma_tf32_opt.rs`（同型）・`tests/parity_nonregression.rs:535-536,566-567`
  （`assert_mma_tf32_baseline`/`check_mma_f16_baseline`。`BASELINES` 全行
  共通で `baseline.seed` から 1 本の `Xorshift64Star` を `new` し、`a =
  rng.fill_vec(m*k)` に続けて同一 rng で `b = rng.fill_vec(k*n)` を生成する
  ——A・B は独立したシードではなく連続したストリームである）・
  `tests/cpu_cuda_wmma_parity.rs:63-66`・`tests/specialized_mma_parity.rs:80-81`
  （いずれも `fill_vec_f16`）

### 3.2 dtype の割当（f32 か f16 か）

| `ParityPath` | dtype | 確認箇所 |
|---|---|---|
| `WmmaTf32`・`WmmaTf32Opt`・`WmmaTf32Staged`・`MmaTf32`・`MmaTf32VsWmmaStaged` | f32（`fill_vec`） | `tests/gemm_wmma_tf32.rs:28-29`・`tests/parity_nonregression.rs:535-536` |
| `MmaF16`・`WmmaF16`・`SpecializedMmaF16` | f16（`fill_vec_f16`） | `tests/parity_nonregression.rs:566-567`・`tests/cpu_cuda_wmma_parity.rs:64-65`・`tests/specialized_mma_parity.rs:80-81` |

### 3.3 `M`（入力規模）の導出（#1237 §7 引き継ぎ項目 1 への回答）

#1237 の候補定義（`parity_tolerance_candidates.py::build_candidates_a()`）
は `M=0.25`（固定。U[-0.5,0.5) の事前上界）と `M=metric.max_ab`（実測。
`m_mode="actual"`。A-2 系列）の 2 通りを持つが、いずれも #1237 対象データ
（U[-0.5,0.5)）向けであり、本データセット（`[-1,1)`）にそのまま転用する
と `M` を過小評価する。本スクリプト（`parity_baseline_impact.py`）は
候補の係数（`c`・`eps_value`・`k_mode`）のみを `build_candidates_a()` から
再利用し、`M` は `--scale-mode`（既定 `exact`）で行ごとに独自導出する:

- `exact`（既定）: `Xorshift64Star` の Python 移植で実際に `A`・`B` を
  生成し `S_A = max|A|`・`S_B = max|B|`・`M = S_A・S_B` を厳密に求める。
  `next_f32(bits) = |2*bits/2^24 - 1|` は `bits` の**凸関数**（絶対値関数
  の合成）であるため、集合全体の絶対値最大は集合内の `bits` の最小値・
  最大値のいずれかで達成される（凸関数は区間の端点で最大化されるという
  一般性質。証明: `a=min(集合)`・`b=max(集合)` とすると集合内の任意の
  `x` は `a<=x<=b` を満たし、凸性 `f(θa+(1-θ)b) <= θf(a)+(1-θ)f(b)` より
  `f(x) <= max(f(a),f(b))`。`a`・`b` 自身も集合の要素なので等号が成立し
  `max_x f(x) = max(f(a), f(b))`）。よって全要素を保持せず `bits` の
  min/max の 2 値だけを追跡すれば `O(count)` 時間・`O(1)` 空間で
  `max|A|`・`max|B|` を求められる（`parity_baseline_impact.py::
  bits_extreme_scan`。素朴な全要素保持実装との一致は
  `parity_baseline_impact_test.py::BitsExtremeScanTest` で検証済み）。
  f16 行（`fill_vec_f16`）は raw f32 の丸めが単調（round-to-nearest）で
  あるため、上記 2 値の raw f32 を `half::f16::from_f32` 相当（Python
  `struct` の `'e'` フォーマット）で丸めた絶対値の大きい方を使う
  （`round_f32_to_f16_abs`）
- `upper-bound`: `M=1`（`[-1,1)` の事前上界）。計算コストゼロだが
  no-op 判定にのみ使え、全救済判定には使えない（上界を過大評価するため）

45 行中に同一 `(seed, m, k, n, dtype)` の組が複数回登場する（例:
`seed=0xBEEF, m=n=k=4096` は `WmmaTf32Opt`・`WmmaTf32Staged` の 2 行で
共有）ため、プロセス内メモ化で重複計算を避ける（40 個の unique 組。
実測実行時間は約 40 秒。約 1.2 億要素の生成を伴う。詳細は
`docs/perf/logs/candle-parity-tolerance-baseline-impact-1238/env_info.txt`）。

実測結果: `exact` モードでの `S_A`・`S_B` は多くの行で 0.99〜1.00 に達する
（K が小さい行 — `k=4,8,17` 等 — では 0.78〜0.99 程度とやや低い。詳細は
`docs/perf/logs/candle-parity-tolerance-baseline-impact-1238/baseline-impact.md`
「行別 入力規模」表）。`upper-bound`（`M=1`）は `exact` の近似として妥当
であることが確認できる（`exact` の `M` は概ね 0.78〜1.00 の範囲）。

## 4. 机上判定の論理

前提: 候補は現行判定への **OR 追加（単調緩和）**としてのみ評価する
（#1237 §2.2 と同じ制約。「置き換え」判定は要素単位のダンプが `BASELINES`
に存在しないため評価不能・スコープ外とする）。

### 4.1 構造的に成立する事実（全候補・全行共通）

- `fail_count` は単調非増加（OR 追加で pass→fail は生じない）→ 契約項目
  「`fail_count <= baseline_fail_count`」は**恒常成立**（後退方向の false
  positive を候補追加が引き起こすことはない）
- `total` 不変、`mean_abs_diff`／`max_abs_diff`／`max_rel_err` は全セル
  集計のため **bit 同一**（候補は判定式のみを変え、計算対象の値は変えない）
  → ceiling 3 項目（`mean_abs_diff <= ceiling`・`max_abs_diff <= ceiling`・
  `max_rel_err <= ceiling`）は無影響
- `baseline_provenance_unconfirmed` の fail-closed 契約（`true` の行は
  無条件 panic）は候補追加とは独立で不変
- 厳密ゼロ fail 判定（`fandhe_ai_backend_cpu::assert_parity`）を使う既存
  テストは OR 追加で fail に転じない

### 4.2 行 × 候補ごとの 3 クラス分類

`BASELINES` には要素単位ダンプが無く行単位の集計値（`baseline_fail_count`・
`baseline_max_abs_diff_ceiling`）しかないため、確定値ではなく**値域**として
分類する:

- **no-op**（`bound < 1e-5`。厳密不等号）: 候補の pass 条件は `d <= bound`、
  既存救済は `d < ABSOLUTE_RESCUE_THRESHOLD(=1e-5)` なので、`bound < 1e-5`
  のとき `d <= bound ⇒ d < 1e-5` が成立し既存救済に完全に包含される。
  よって `fail_count` は**厳密に不変**。`bound == 1e-5` は `d == 1e-5` の
  要素が候補側でのみ救済されうるため no-op に含めない
- **全救済**（`baseline_max_abs_diff_ceiling <= bound`）: ceiling は
  「表示桁の最終桁 +1」で切り上げた保守的な値のため、これを下回るなら
  行内の全要素が候補で救済されうる → `fail_count → 0`
- **部分／未確定**（上記以外）: 行内の一部要素のみ救済されるか確定でき
  ない。確定値には要素単位ダンプ（GB10 実機実測）が必要（#1256 へ引き
  継ぎ）
- **分類不能（ceiling 未実測）**: `baseline_max_abs_diff_ceiling` が
  `None` の行（`BASELINES` 中 1 行のみ。`WmmaTf32Opt 512x512x512
  seed=0x7A0`）は全救済判定ができない

**注記（誤読防止）**: 候補の `eps_f32 ∈ {2^-23（machine epsilon）,
2^-24（unit roundoff）}` は f32 SIMT 出力と candle の比較を想定した係数
であり、TF32（仮数 10bit）・f16 経路の baseline 行へ適用するのは「機械的
な OR 追加の影響確認」であって、当該経路の誤差モデルとして妥当という
主張ではない。

B-1（`max_partial` 基準）・B-3（`exact` 基準）は部分和トレースまたは
厳密真値が本データセットには存在しないため、全 45 行「机上分類不能
（実行時適用不可／要トレース）」とする（bound 自体を計算しない）。

B-2（`sum_abs_ab` 基準）は `Σ|a_k・b_k| <= K・S_A・S_B`（上界）で代用し、
`bound = t・ulp(K・S_A・S_B)` として同じ 3 クラス分類を適用する（真の
`Σ|ab|` は行内で要素依存のため、この代用は no-op 側には安全〈上界なので
実際の bound は代用値以上〉だが全救済判定は参考値扱いとする）。

A-4（`K・u・Σ|ab|` の上界近似。`u=2^-24`）は#1237 と同じく緩すぎる参考値
として扱う。

## 5. 結果（`--scale-mode exact`。生表は logs 参照）

集計（45 行に対する分類件数。生成コマンドは §7）:

| 候補 | no-op | 全救済 | 部分／未確定 | 分類不能（ceiling未実測） |
|---|---:|---:|---:|---:|
| A c=0.125 eps=2^-23 K*M | 30 | 0 | 14 | 1 |
| A c=0.25 eps=2^-23 K*M | 25 | 0 | 19 | 1 |
| A c=0.5 eps=2^-23 K*M | 22 | 0 | 22 | 1 |
| A c=1.0 eps=2^-23 K*M | 17 | 0 | 27 | 1 |
| A c=2.0 eps=2^-23 K*M | 10 | 1 | 33 | 1 |
| A c=0.125 eps=2^-24 K*M | 33 | 0 | 11 | 1 |
| A c=0.25 eps=2^-24 K*M | 30 | 0 | 14 | 1 |
| A c=0.5 eps=2^-24 K*M | 25 | 0 | 19 | 1 |
| A c=1.0 eps=2^-24 K*M | 22 | 0 | 22 | 1 |
| A c=2.0 eps=2^-24 K*M | 17 | 0 | 27 | 1 |
| A c=1.0 eps=2^-23 sqrtK*M（A-3） | 44 | 0 | 0 | 1 |

代表的な観察:

- **A-3（√K スケール）は 44/45 行が no-op**（唯一の非 no-op 行は
  `baseline_max_abs_diff_ceiling` が `None` の分類不能行）——線形 K 版
  （A-1）より √K 版（A-3）の方が bound がはるかに小さく、大半の行で
  既存救済に包含される
- **A-1 系列で「全救済」が生じるのは `c=2.0 eps=2^-23` の 1 行のみ**:
  `MmaTf32VsWmmaStaged 512x512x512 seed=6002`（`baseline_max_abs_diff_ceiling
  = 1.220704e-3` が `bound = 1.221e-4` を下回る、すなわち `ceiling <=
  bound`）。他の全 A-1 系列・全行では「no-op」か「部分／未確定」のいずれか
  であり、**行を丸ごと救済してしまう組み合わせはほぼ存在しない**
- K が大きい行（K=4096。4 行）は A-1 系列のほとんどの候補で「部分／未確定」
  に分類される（bound が大きくなるため no-op の対象から外れるが、
  `ceiling` の方がさらに大きいため全救済にも届かない）
- B-2（`t=1,2,4`）・A-4 も同様の傾向（省略。生表参照）

**「置き換え」判定はスコープ外**（§4 前提）。「全救済」に分類された行
であっても、現行 pass 要素が候補判定の下で新たに fail に転じうるか（＝
tolerance を緩めることで見逃す真の回帰が増えないか）は本分析からは
評価できない。

生表（行別 `S_A`／`S_B`・候補 A 全系列・A-4・B-2 の 45 行分の bound と
分類）は
`docs/perf/logs/candle-parity-tolerance-baseline-impact-1238/baseline-impact.md`
を参照（読者は `python3 scripts/bench/framework-compare/
parity_baseline_impact.py --scale-mode exact` を実行して同じ表を再生成
できる。同一コマンドの再実行で bit 完全一致することを確認済み）。

## 6. 契約 5 項目への影響

| `assert_no_parity_regression` の検査項目 | 影響 |
|---|---|
| `baseline_provenance_unconfirmed` の fail-closed 契約 | 無影響（候補追加とは独立） |
| `total` 完全一致 | 無影響（要素数は不変） |
| `fail_count <= baseline_fail_count` | **恒常成立**（OR 追加の単調性。§4.1） |
| `mean_abs_diff <= baseline_mean_abs_diff_ceiling` | 無影響（bit 同一） |
| `max_abs_diff`/`max_rel_err <= ceiling`（`Some` のみ） | 無影響（bit 同一） |

契約自体は全候補で成立し続ける（非後退検査が誤って fail する行は無い）
が、全救済／部分クラスの行は `baseline_fail_count` が緩い上限になる
（＝実際の候補判定を実装した場合、回帰検出力が低下する方向の変化）。
これは「下方更新」ではなく「判定式自体の変更」であるため、`BASELINES`
の値そのものは本 issue では一切変更していない。

## 7. 同時更新が必要な箇所一覧

候補判定を本体（`compare`／`assert_parity`／`ParityBaseline`）へ実際に
組み込む場合に整合を取る必要がある箇所（file:line は本 issue 実装時点の
実測）。**本 issue はこれらを一切変更していない**（一覧の作成までがスコープ）。

### 7.1 定数ピン（定数のみを検査し判定式は検査しない）

| 検査 | file:line |
|---|---|
| `assert_tolerance_constants_pinned`（本体） | `crates/backend-cuda/tests/common/parity_baseline.rs:1290` |
| 呼び出し箇所（13 箇所） | `crates/backend-cuda/tests/{mse_parity.rs:109,132, rmsnorm_parity.rs:123,144,233, rmsnorm_backward_parity.rs:347,403,583,711, softmax_parity.rs:86,107,329, gemm_bias_act_parity.rs:92,128}`・`parity_nonregression.rs:84`・`gemm_tf32_optin.rs:305,327,413` |
| `parity_tolerances_match_backend_cpu_contract`（`extract_f64_const` で `pub const NAME: f64 = …;` をパース） | `scripts/bench/framework-compare/bench-common/src/parity.rs:1084`（`extract_f64_const` は `:1119`） |
| `test_summarize_tolerances_match_backend_cpu_contract`（`_extract_f64_const`） | `scripts/bench/framework-compare/summarize_test.py:3359`（`_extract_f64_const` は `:3393`） |

**重要な事実**: 上記 3 系統はいずれも `RELATIVE_TOLERANCE`／
`ABSOLUTE_RESCUE_THRESHOLD` の**定数値**を bit 等値で検査するのみで、
**判定式（`rel < RELATIVE_TOLERANCE || diff < ABSOLUTE_RESCUE_THRESHOLD`）
自体は検査しない**。よって定数を変えずに判定式へ候補判定を OR 追加した
場合、全ピンは green のまま通過してしまう（判定式変更の検知漏れ）。
候補判定を実装する場合は判定式そのものをピン止めするテストの追加が
必要（#1254／#1252 のスコープ）。

### 7.2 判定式レプリカ（定数を参照しつつ式を再実装している箇所）

| 箇所 | file:line |
|---|---|
| `element_error`（判定式そのものを再実装） | `scripts/bench/framework-compare/bench-common/src/parity.rs:118` |
| `wmma_tolerance_probe.rs` の複合判定レプリカ（`pass = rel < RELATIVE_TOLERANCE \|\| diff < ABSOLUTE_RESCUE_THRESHOLD`） | `crates/backend-cuda/examples/wmma_tolerance_probe.rs:680`（診断表示用の `margin()` 呼び出しは `:500-501,551,597-598`） |
| `specialized_mma_f16_triage.rs` | `crates/backend-cuda/tests/specialized_mma_f16_triage.rs:245` |
| `gemm_mma_tf32_triage.rs` | `crates/backend-cuda/tests/gemm_mma_tf32_triage.rs:118-119` |
| `gemm_mma_block_tile_bench.rs` | `crates/backend-cuda/examples/gemm_mma_block_tile_bench.rs:147-148` |
| `gemm_mma_tf32_block_tile_bench.rs` | `crates/backend-cuda/examples/gemm_mma_tf32_block_tile_bench.rs:108-109` |
| `summarize.py` の `CHECKSUM_*`（`PARITY_*` の別名） | `scripts/bench/framework-compare/summarize.py:850` |
| `compare_gemm_gate.py::_parity_check`（判定不能条件） | `scripts/bench/framework-compare/compare_gemm_gate.py:219` |

これらは定数こそ本体から参照するが判定式ロジック自体を各ファイルで
再実装しているため、候補判定を追加する場合は**個別に**式を更新しない
限り、本体側の判定変更から取り残される。

### 7.3 リテラル閾値レプリカ（定数を参照せず `1e-3`／`1e-5` を直書き）

`crates/autodiff/tests/{poc_v2_2_parity.rs, fusion_backend_integration.rs,
fusion_chain_limit.rs}` の `composite_close`・`optim_sgd.rs`・
`nn_optim_adamw.rs`・`mse_loss_fusion.rs`・`crates/backend-metal/tests/
{sgd_device_parity.rs, command_batching_bench.rs, gemm_resident_parity.rs,
command_batching.rs}`・`crates/backend-cuda/tests/{gemm_resident_real_device.rs,
sgd_device_real_device.rs, async_ordering_real_device.rs}`・
`examples/gemm_wmma_tf32_staged_stages_bench.rs`・`crates/facade/tests/
{device_param_store_backend_parity.rs, device_param_store_train.rs}`・
`crates/backend-cpu/tests/sgd_device_parity.rs`。

これらは定数ピンの検査対象にすら入っておらず、候補判定の導入有無に
関わらず既存の「単独緩和検知漏れ」のブラインドスポットである
（`.claude/rules/coding-rust.md`「バックエンド間数値一致テストの許容誤差を
単独で緩和しない」の対象。本 issue の範囲外。#1254 が扱う）。

### 7.4 文言

`.claude/rules/coding-rust.md`（バックエンド構成節・テスト節）・
`AGENTS.md`（該当節）・`.github/codex/prompts/review.md`・
`docs/cuda-tensor-core-parity-judgment-decision.md` §4・
`docs/perf/cuda-parity-baseline.md` §2-1・§6 末尾・
`crates/backend-cuda/tests/common/parity_baseline.rs` の定数 doc コメント・
`scripts/bench/framework-compare/README.md`・本リポジトリの CLAUDE.md。
spec（`docs/spec/04-requirements.md` REQ-2）の変更は #1240 経由のみ
（本リポの `docs/spec/` は編集しない。CLAUDE.md「委譲方針」節）。

### 7.5 新係数定数を追加する場合の追加対応

候補（`c`・`eps`・`t` 等）を新しい定数として本体に追加する場合、
`extract_f64_const`（`bench-common/src/parity.rs:1119`）と
`_extract_f64_const`（`summarize_test.py:3393`）の**両方**に抽出対象を
追加する必要がある（片方のみの更新は定数ドリフト検知の片翼を失う）。

## 8. Phase 2（実装）への設計制約

- `fandhe_ai_backend_cpu::compare`／`assert_parity` は現状 `(a, b)` の
  2 スライスのみを受け取り、K・入力スケール（`M`）を受け取らない。候補
  A/B を実装するには新しい入口（別関数）を設けるか、シグネチャ変更
  （破壊的変更）が必要
- `assert_no_parity_regression` は比較器（`compare`）が生成した
  `CompareReport`（`fail_count`・`mean_abs_diff` 等）を受けて非後退判定
  するため、`BASELINES` の記録値は**現行比較器で記録された値**である。
  候補判定を追加した新しい比較器を使う場合、`fail_count` 等の意味が
  変わるため `BASELINES` 自体の実機再測定が必要（本 issue の§5〜6が示す
  値域はあくまで「OR 追加した場合の理論上の値域」であり、確定値ではない）
- REQ-2「2026-09-02 追記・限定救済項（案 1′）」（`diff < max(1e-5,
  τ_path·S_A·S_B·√K)`）は候補 A-3（√K スケール）と同型だが、
  `internal-diagnostics` 限定・一般契約への昇格条件付きという spec 上の
  制約がある。候補 A-3 を一般契約へ組み込む場合はこの spec 条件との
  整合確認が必要（#1240 経由）
- B-1（`max_partial` 基準）を実行時に使うには GPU カーネル側で部分和を
  トレースする機構が必要（既存カーネルは最終結果のみを返す）。実装可否・
  コストの検討自体を本 issue では行っていない（#1254 へ引き継ぎ）

## 9. 未変更事項の確認

- `git diff --stat origin/main -- crates/ docs/spec/` は空（`crates/`・
  spec submodule を変更していない）
- `crates/backend-cuda/tests/common/parity_baseline.rs`（`BASELINES`・
  `RELATIVE_TOLERANCE`／`ABSOLUTE_RESCUE_THRESHOLD` 定数）は無変更
- `docs/perf/cuda-parity-baseline.md` の変更は本 issue への参照 1 行の
  追記のみ（§6「ベースライン更新規約」・数値自体は不変）
- イシュー #1356（3×TF32 baseline 提案値。未承認・`BASELINES` 未収録）は
  本分析の対象外（`ParityPath::MmaTf32` 等の既存 45 行のみを対象とした）

## 10. 再現手順

```bash
cd scripts/bench/framework-compare

# 単体テスト
python3 -m unittest parity_baseline_impact_test.py

# 生出力の再現（約 40 秒。exact モード）
python3 parity_baseline_impact.py --scale-mode exact \
  > ../../../docs/perf/logs/candle-parity-tolerance-baseline-impact-1238/baseline-impact.md

# 即時（upper-bound モード。no-op 判定の参考値のみ）
python3 parity_baseline_impact.py --scale-mode upper-bound
```

実行ログ・env_info（内部ホスト名なし）は
`docs/perf/logs/candle-parity-tolerance-baseline-impact-1238/` を参照。
実機（GB10・M4 Max）実行は不要・実施していない（本分析は `BASELINES` に
既に記録されている集計値と、コミット済みソースから決定的に導出できる
入力規模のみを使う机上計算のため）。
