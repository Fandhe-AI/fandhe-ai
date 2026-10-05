# 順序統計・NaN 無視縮約（`median`・`kthvalue`・`quantile`・`nanmean`・`nansum`）の CPU 実装記録（イシュー #2637）

親: #2625（Phase 4）／ルート: #2499。`docs/autodiff-cumulative-ops-decision.md`（#2636）・
`docs/autodiff-fft-ops-decision.md`（#2631）の「共有カーネルを `tensor-core` に置く」方式を再適用した実装記録であり、
**承認記録ではない**（facade 公開形の承認は #2677 で依頼中。公開自体は承認後の #2678）。

## 0. 結論

- 順序統計 3 種（`median`・`kthvalue`・`quantile`）と NaN 無視縮約 2 種（`nansum`・`nanmean`）を CPU 参照実装として
  内部クレートへ追加した。
  - 共有カーネル: `fandhe_ai_tensor_core::stat_reduce`（`kthvalue_host`／`median_dim_host`／`median_all_host`／
    `quantile_host`／`nansum_host`／`nanmean_host`・VJP 3 種・`stat_layout`・`QuantileInterpolation`・`StatReduceError`）。
    順序規則・NaN 規則・アキュムレータ契約の単一情報源。
  - `BackendOps` に既定 `Unsupported` のメソッド 6 件（`stat_kthvalue`／`stat_median_dim`／`stat_median_all`／
    `stat_quantile`／`stat_nansum`／`stat_nanmean`）を追加（非破壊拡張）。CPU は共有カーネルを呼ぶだけの override。
    CUDA／Metal は変更なし。
  - 入口は自由関数モジュール `fandhe_ai_autodiff::stat_reduce_ops`（`median`／`median_with_indices`／`kthvalue`／
    `quantile`／`nansum`／`nanmean`）。専用 `Op` 5 variant（`Op::OrderSelect`／`Op::MedianAll`／`Op::Quantile`／
    `Op::Nansum`／`Op::Nanmean`）と VJP は `grad.rs`。
- facade 公開は行わない。`StatReduceOpsHoldDoctestGuard`（`crates/facade/src/lib.rs`）と
  `crates/facade/tests/api_surface.rs` の否定ガードで機械固定した。
- 依存・`unsafe`・tolerance・baseline・`docs/spec/`・ガードレール閾値は変更していない。
- CUDA／Metal の専用カーネルは対象外。既定 `Unsupported` → 共有ホストカーネルへのフォールバックで動作する。
  実機テストは `#[ignore]` のまま未実測。

## 1. 着手時の判定（事実のみ）

- 5 演算は REQ-9 Tier の列挙に名前がない。
- 本実装はルート #2499 の Phase 4 方針（内部実装＋保留ガードまで先行し、facade 公開は承認後）に基づき、#2637 の受入条件
  （内部実装・決定記録・保留ガード）に限って行った。公開面・対象範囲の拡張について承認済みとは記録しない
  （承認依頼 #2677 は別途）。

## 2. 実装方式・命名規律

| 層 | 置き場所 | 内容 |
|---|---|---|
| 共有カーネル | `tensor-core/src/stat_reduce.rs` | レイアウト検査（`StatLayout`）・forward 6 種・VJP 3 種・`QuantileInterpolation`・`StatReduceError` |
| バックエンド抽象 | `tensor-core/src/backend_ops.rs` | `stat_*` 6 件（既定 `Unsupported`） |
| CPU | `backend-cpu/src/ops.rs` | 共有カーネルを呼ぶだけの override（`contiguous()` 済みスライスを渡す） |
| autodiff | `autodiff/src/stat_reduce_ops.rs`・`tape.rs`・`grad.rs`・`error.rs` | 自由関数 6 件・`Op` 5 variant・VJP・`From<StatReduceError>` |

- 既存演算の合成（`masked_fill`＋`sum` 等）は採らず共有カーネルにした。理由: 5 演算で `Unsupported` フォールバック契約を
  一様にテストできる・`nanmean` の丸めを 1 回にできる・tape ノードが 1 個で済む。
- 命名規律: 素の `fn median`／`median_with_indices`／`kthvalue`／`quantile`／`nanmean`／`nansum` は
  `autodiff/src/stat_reduce_ops.rs` の各 1 件のみ（workspace インベントリが固定。既存の無関係な `fn median` は §9）。
  trait メソッドは `stat_*`、共有カーネルは `*_host`。
- `Var`／`Tape`／`Tensor` に inherent メソッドは足していない（足すと facade 公開面が広がる）。
- 非融合（`push_eager`）・非 checkpoint・高階微分（`create_graph`）非対応。f64 自動微分経路は対象外。
- 上流 shape の検査は `check_fft_upstream_shape`（FFT 用の名称）を流用している（#2636 と同じ。兄弟 issue が編集する
  FFT の腕に触れないため）。

## 3. 数値契約と意味論

PyTorch 2.14.0 の実行値 fixture で実測して確定した（実装前の仮説と異なった点は §5 に明記）。

順序の共通規則（本リポ独自の契約）: lane 内は `BackendOps::sort` と同じ昇順・**安定**（同値は元添字昇順）・NaN は
任意の非 NaN より大きい・`±0` は同値。lane ごとの添字バッファの `sort_by`（安定）で実現し、`select_nth_unstable` は
使わない（タイの索引が非決定的になるため）。

| 演算 | 値 | 索引 | NaN |
|---|---|---|---|
| `kthvalue(k, dim)`（`k` は 1 始まり） | 安定昇順の `k-1` 番目（選択のみ。入力要素と bit 一致） | その要素の元添字 | NaN は最大として並ぶだけ |
| `median(Some(dim))`／`median_with_indices` | **下側中央値**（偶数個でも平均しない。位置 `(n-1)/2`） | 同上 | lane に NaN があれば値は NaN・索引は最初の NaN |
| `median(None)` | 全要素の下側中央値 | なし | NaN があれば NaN。要素数 0 は NaN |
| `quantile(q, dim, interp)` | `rank = f64(q) * (n-1)`。`Linear`: lerp・`Lower`: floor・`Higher`: ceil・`Midpoint`: lerp(0.5)・`Nearest`: 偶数丸め | なし | lane に NaN があれば NaN |
| `nansum(dim)` | NaN を 0 とみなした和。全 NaN・空 lane は `0.0` | なし | 無視 |
| `nanmean(dim)` | 非 NaN の和 ÷ 非 NaN の個数。全 NaN・空 lane は NaN | なし | 無視 |

アキュムレータ・丸め:

- 選択系は算術を含まず FMA 契約・`f64` 契約の対象外。
- `nansum`／`nanmean` は lane ごとに要素を `f64` へ昇格して添字昇順に逐次加算し、最後に 1 回だけ `f32` へ downcast
  （coding-rust.md の長軸縮約契約）。`nanmean` は `f64` のまま `和 ÷ 個数` を計算してから 1 回 downcast する
  （既存 `mean` の軸指定経路は `f32` の和を除算するが、本演算は丸め 1 回を意図的に選ぶ）。
- `quantile` の rank は **`f64` で `f64::from(q) * (n - 1)`**（§5。PyTorch 2.14.0 の実測）。補間は `f64` で PyTorch の
  `lerp` と同じ 2 分岐（`w < 0.5` なら `a + w(b-a)`、それ以外は `b - (b-a)(1-w)`）を計算して 1 回 downcast。
  `Lower`／`Higher`／`Nearest` は算術を通さず選択のみ。`mul_add` は使わず matmul 系 FMA 契約には触れない。
- 非有限入力は拒否せず伝播する（`±inf` の補間は `inf - inf = NaN` になりうる。PyTorch と forward のクラスが一致した）。

VJP:

- `kthvalue`／`median(Some(dim))`: 選ばれた 1 要素へ upstream をそのまま流す（`scatter_with_fallback`・`Add`。lane ごとに
  出力 1 個のため索引は重複しない）。
- `median(None)`: 中央値と**等しい要素（値の等価 `==`。`-0.0` と `0.0` は同じ組）**へ `g / 個数` を均等分配する。中央値が
  NaN のときは NaN 要素へ均等分配する（PyTorch 2.14.0 の実測）。**軸指定版（1 要素へ全量）と規則が異なる**。
- `quantile`: 下側へ `g·(1-w)`・上側へ `g·w`（`Lower`／`Higher`／`Nearest` は選択 1 要素へ `g`・`Midpoint` は 0.5 ずつ）。
  入力から lane を再ソートして導出する（`Op::Logcumsumexp` と同じ「入力から再計算」方式）。
- `nansum`: 非 NaN 位置へ `g`、NaN 位置へ `0 × g`。**要素選択でなく乗算**（実測: `g` に `inf`／NaN を流すと NaN 位置が
  NaN になる）。`nanmean`: 非 NaN 位置へ `g / 個数`、NaN 位置へ `0 × (g / 個数)`。全 NaN lane は `g / 0` の NaN が
  全位置に出る（実測）。`f64` で計算して 1 回 downcast。

## 4. 境界検査

- `stat_layout` が確保・実体化より前に `dim` 範囲（rank 0 に `Some(_)` を含む。`dim = None` は rank 0 も受理）と、
  入力要素数・バイト数・lane ソート用作業バッファの `checked_mul`・`isize::MAX` 超過を型付きエラー
  （`ShapeError::AxisOutOfRange`／`ElementCountOverflow`）で拒否する。要素数 0 の shape は入力側の部分積を計算する前に
  空レイアウトへ倒す（`[usize::MAX, usize::MAX, 0]` でも panic しない）。
- 索引を返す 2 演算（`kthvalue`・`median_with_indices`）は軸長 − 1 が `i32::MAX` 以下であることを 2 層で検査する
  （autodiff 入口が実体化より前に `check_i32_indices`、カーネルが `i32::try_from`。無検査 `as i32` なし）。
- `k`（1..=軸長）・`q`（有限かつ `[0, 1]`）・順序統計 3 種の空 lane は `InvalidArgument`
  （`AutodiffError::InvalidArgument`／`BackendError::InvalidArgument`）。`nansum`／`nanmean`／`median(None)` は空入力を受理する。
- カーネルは入力スライス長を再検査する（`ElementCountMismatch`）。`unsafe`／`get_unchecked`／本番経路の `unwrap`／
  `expect` は使わない。
- フォールバック条件は `BackendError::Unsupported` のみ。それ以外のバックエンドエラーは握りつぶさず伝播し、
  バックエンドの戻り値 shape（値・索引とも）が期待と異なる場合は `BackendError::ShapeMismatch` で拒否する。

## 5. PyTorch 2.14.0 との差分・実測で確定した点

fixture は実 PyTorch 2.14.0+cpu の実行値（`crates/autodiff/tests/fixtures/stat-reduce-pytorch-reference/`。f32 は u32
ビットパターンで保存）。有限（タイなし）436 件で値（選択系は bit 一致・補間系と `nan*` は REQ-2 判定）・索引完全一致・
勾配 REQ-2 判定が一致した。NaN 141 件・タイ 129 件・±inf 186 件・非有限 upstream 12 件・エラー 28 件も §3 の規則のとおり
一致（比較方針は下表と §6）。

実装前の仮説から**実測で修正した点**:

| 項目 | 計画時の仮説 | PyTorch 2.14.0 の実測 | 本実装 |
|---|---|---|---|
| `quantile` の rank 精度 | `f32` の積 `q * (n-1)` | **`f64`**（`n=11, q=0.1f` の `Higher` → 2、`n=11, q=0.7f` の `Lower` → 6、`n=6, q=0.1f` の `Nearest` → 1。`f32` の積なら 1／7／0） | `f64` の rank に合わせた |
| `quantile` の lane 長上限 | `2^24` 超を拒否 | `f32` rank 前提の上限だった。実測では 16,777,217 要素でも例外にならない | rank を `f64` にしたため上限を設けない |
| `median(None)` の空入力 | `InvalidArgument` | 例外にならず NaN | NaN（実測に合わせた） |
| `nansum`／`nanmean` の NaN 位置の勾配 | 要素選択 | 乗算（`g=inf` で NaN 位置が NaN） | 乗算に合わせた |
| 全 NaN lane の `nanmean` 勾配 | 未確定 | 全位置 NaN | 同じ |

その他の差分:

| 項目 | PyTorch 2.14.0 | 本実装 | 扱い |
|---|---|---|---|
| 索引の型 | `int64` | `Tensor<i32>`（`Var::sort`／`topk` と同じ慣例） | 軸長 − 1 が `i32::MAX` 超は型付きエラー |
| `dim` | 負の添字可 | `usize` | 差分。受け入れ範囲外 |
| `keepdim`・複数軸・1 次元 `q`・`nanmedian`／`nanquantile` | あり | 非対応 | 対象外（§8） |
| 0 次元入力への `dim=0` | 受理 | `AxisOutOfRange` で拒否 | 差分。`dim=None` は両者とも受理 |
| 軸長 0 の `median`／`kthvalue`／`quantile` | `IndexError`／`RuntimeError` | `InvalidArgument` | 一致（拒否） |
| 軸長 0 の `nansum`／`nanmean`、`median(None)` | 受理（0／NaN／NaN） | 受理 | 一致 |
| `k=0`・`k>n`・`q<0`・`q>1`・`q=NaN`・`dim` 範囲外 | 例外 | 型付きエラー | 一致（拒否） |
| タイ時の索引（`kthvalue`／`median`） | 規定なし（実測は非安定。例 `[2,1,2,2,3]` の中央値は添字 2、`k=3` は添字 0） | 安定昇順で決定的 | **受入条件にしない**。タイなしの入力のみ索引の完全一致を要求し、タイ入力は `x[index] == value` と独自契約の手計算期待値で検証 |
| NaN が 2 個以上ある lane の `kthvalue` の索引・選択系の勾配 | 並びが不定（`[1,nan,2,nan,0,7]` の `k=5` → 添字 3・`k=6` → 添字 1） | 安定順（`kthvalue` は NaN 同士を元添字順に並べる）。`quantile` の勾配は安定順で最後の NaN へ流す（独自契約。NaN が 1 つなら PyTorch と同じ） | 比較しない（単一 NaN の lane は一致を確認）。`median` の索引は最初の NaN で一致 |
| ±inf を含む入力の補間値 | `inf - inf` で NaN | 同じ（IEEE 伝播） | forward 186 件がクラス一致。勾配の PyTorch 完全一致は受入条件にしない |

tolerance・baseline は変更していない。

## 6. テスト構成

- `crates/tensor-core/src/stat_reduce.rs`（単体）: 下側中央値・安定順のタイ索引・NaN 規則・`±0`・非末尾 `dim`・
  `dim=None`・軸長 1・空入力（受理／拒否の別）・`k`／`q` 検査・5 補間と端点・`Nearest` の偶数丸め・`f64` rank・
  `nansum`／`nanmean` の `f64` 蓄積（`f32` 逐次和では落ちる並び）・境界エラー（`[usize::MAX, usize::MAX, 0]` で panic
  しない等）・`i32` 上限・スライス長不一致・VJP の手計算一致・run-to-run bit 決定性。`backend_ops.rs`: 既定 `Unsupported`。
- `crates/autodiff/tests/stat_reduce_parity.rs`: fixture 突合（有限・NaN・タイ・±inf・非有限 upstream・エラー）、中心差分
  （6 関数）・手計算オラクル、タイの独自契約（安定順の索引・`median` の軸指定／全要素の勾配規則差）、モック `BackendOps` による
  `Unsupported` フォールバックと他エラーの伝播と誤 shape（値・索引）、引数・`dim`・空軸の型付きエラー、巨大 broadcast view の
  確保前拒否、テープ記録数（各 1 ノード）、`create_graph` が型付きエラーであること、run-to-run 決定性。
- `crates/backend-cpu/tests/stat_reduce_parity.rs`・`backend_ops_dispatch.rs`: `CpuBackendOps::stat_*` の直接呼び出し
  （解析値・strided 入力・型付きエラー・決定性）、CUDA（macOS では Metal も）の `stat_*` が `Unsupported` を返し panic しないこと。
- `crates/facade/tests/stat_reduce_ops_backend_parity.rs`: CPU tape と NaiveOps tape の突合（属性なし）。CUDA／Metal 実機は
  `#[ignore]`。

## 7. facade 公開形の推奨案（未承認）

推奨は 1 つ。`Var` の inherent メソッドとして `stat_reduce_ops` への 1 行委譲で公開する。

- `Var::median(&self, dim: Option<usize>)`（`-> Result<Var<'t>, AutodiffError>`）
- `Var::median_with_indices(&self, dim: usize)`・`Var::kthvalue(&self, k: usize, dim: usize)`
  （`-> Result<(Var<'t>, Tensor<i32>), AutodiffError>`。戻り値形は `Var::topk`／`Var::sort` と同形）
- `Var::quantile(&self, q: f32, dim: Option<usize>, interpolation: QuantileInterpolation)`
  （`-> Result<Var<'t>, AutodiffError>`）
- `Var::nanmean(&self, dim: Option<usize>)`・`Var::nansum(&self, dim: Option<usize>)`（`-> Result<Var<'t>, AutodiffError>`。
  `Var::sum` と同形）
- 引数型 `QuantileInterpolation` だけをクレートルートから再エクスポートする（`FftNorm` と同じ扱い）。`stat_reduce_ops`／
  `stat_reduce` モジュールと `StatReduceError` は再エクスポートしない。
- inherent メソッドと型 1 件の追加のみで非破壊。承認依頼は #2677、公開は承認後の #2678。
- 承認後は `StatReduceOpsHoldDoctestGuard` と否定ガードを承認形の正ガード（委譲本体の固定を含む）へ反転する。

承認事項（**すべて未承認**）: 上記 6 メソッドの公開と `QuantileInterpolation` の再エクスポート、およびメソッド名・引数形
（`dim: Option<usize>`／`usize`・`k` は 1 始まり・索引 `Tensor<i32>`・`q` はスカラー `f32`）。

## 8. スコープ外

- facade 公開（#2677 の承認後に #2678・#2679）。
- `docs/compat-api-scope.md` 1 節の対象範囲表の拡張・`docs/compat-feature-gap.md` の判定変更・spec（REQ-9）の改定。
- CUDA／Metal の GPU 専用カーネルと実機計測（§10）。
- `nanmedian`・`nanquantile`・1 次元 `q`・`keepdim`・負の `dim`・複数軸・索引の `int64` 化・0 次元入力への軸指定。
- `create_graph`（高階微分）・activation checkpoint 対象化・f64／f16／bf16 自動微分経路。
- 選択アルゴリズムの高速化（quickselect・並列化。現状は lane ごと O(n log n) の安定ソート）。
- 非有限入力時の補間値・勾配の PyTorch 完全一致。
- 既存 `fn median`（`backend-cpu` のテスト内・`guardrail`）のリネーム、`check_fft_upstream_shape` の改名、
  `MIN_KNOWN_PROBE_BLOCKS` の更新。

## 9. 多層防御（保留ガード）

| ガード | 内容 |
|---|---|
| `StatReduceOpsHoldDoctestGuard`（`crates/facade/src/lib.rs`） | 全 `pub mod` glob import 下で、同名の関数・メソッド・型・モジュールが `Var`／`Tape`／`Tensor<f32>`／facade ルートに公開されるとコンパイルが失敗する正のプローブ |
| `stat_reduce_ops_hold_doctest_globs_all_pub_modules` | doctest の glob 集合と `pub mod` 宣言集合の一致 |
| `stat_reduce_ops_hold_doctest_probe_body_matches_fixed_contract` | プローブ本文の固定文言一致 |
| `facade_does_not_reexport_or_declare_stat_reduce_ops`（＋自己テスト） | facade src の再エクスポート・型の独自宣言・`pub mod stat_reduce_ops`／`pub mod stat_reduce`・6 名の `fn` 宣言の否定検査 |
| `workspace_declares_stat_reduce_ops_fn_names_only_in_allowed_locations` | workspace 全体で 6 名の `fn` 宣言が `autodiff/src/stat_reduce_ops.rs` の各 1 件と、**本イシュー以前から存在する無関係な `fn median`**（`backend-cpu/src/gemm_blis/mod.rs` のテスト内ローカル関数 5 件・`guardrail/src/report.rs` の 1 件）だけであること |

既存の `fn median` はリネームせず期待値へ実測件数で列挙した（リネームは本イシューの対象外）。これら以外への追加は引き続き
fail-closed に検出する。stable rustdoc は `compile_fail` のコードを照合しないため、否定ガードは正のプローブ＋インベントリで
組んでいる。保留ガードの有効性は、一時的に facade へ `pub use fandhe_ai_tensor_core::QuantileInterpolation;` を足して doctest
（E0659）と `facade_does_not_reexport_or_declare_stat_reduce_ops` が落ちることを確認したうえで元に戻した。

## 10. 実機申し送り

CUDA（DGX Spark GB10）・Metal（Apple Silicon）の実機テスト（`cuda_stat_reduce_ops_match_cpu_reference`・
`metal_stat_reduce_ops_match_cpu_reference`）は `#[ignore]` のまま未実測。手順は
`docs/perf/logs/stat-reduce-ops-2637/README.md`。

## 11. 出典

- `docs/autodiff-cumulative-ops-decision.md`（#2636）・`docs/autodiff-fft-ops-decision.md`（#2631）・
  `docs/autodiff-nonfinite-ops-decision.md`（#2635）・`docs/autodiff-topk-unique-ops-decision.md`（sort／topk の索引慣例）
- `docs/compat-api-scope.md` 5 節（適用記録）・`.claude/rules/coding-rust.md`（REQ-2 判定・f64 長軸縮約契約・
  カーネル境界検査）
- PyTorch 2.14.0 実行値: `crates/autodiff/tests/fixtures/stat-reduce-pytorch-reference/README.md`
