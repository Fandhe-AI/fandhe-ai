# rfft・irfft の内部実装記録（FFT 第 1 弾）

イシュー #2631（親 #2630「FFT」・ルート #2499 Phase 4）。基準コミット
`7711a3ac`（main HEAD）。設計の正は `docs/autodiff-fft-design.md`（#2151。案 B）で、
本書はその実装記録である。`fft`／`ifft` は #2632、`stft`／`istft` は #2633 が本書へ追記する。

## §0 結論

`rfft`（実 → 半スペクトル）と `irfft`（半スペクトル → 実）を、**内部クレート限定**で
CPU 参照実装と VJP 付きで実装した。複素数は末尾次元 2 の `f32` 実テンソル対
`(re, im)`（`torch.view_as_real` と同レイアウト）で表し、complex dtype は非目標のまま。

- **facade 公開面は追加していない**（`Var` の inherent メソッドも `pub use` も無し）。
  公開形は未承認で、承認依頼は #2677、公開自体は承認後の #2678。
- 内部実装はルート #2499 の Phase 4 方針（内部実装＋保留ガードまでを先行し、公開は承認後）
  に基づく。`docs/compat-api-scope.md` §1 の対象範囲表・`docs/compat-feature-gap.md` の
  判定列は変更していない（§1）。
- 依存追加なし・`unsafe` なし・tolerance／baseline／ガードレール閾値／`docs/spec/` 不変。

## §1 着手時の判定（事実のみ）

`docs/autodiff-fft-design.md` §2・§10 は、FFT が REQ-9 の Tier 1／Tier 2 に未列挙であり、
実装着手には `docs/compat-api-scope.md` §5 の手続きが前提と記す。一方ルート #2499 の
「承認範囲」節は、Phase 4 で新たな公開面を作るものについて「内部実装＋保留ガードまで先行し、
公開は承認後の #2678・#2679」と定める。本 issue はその Phase 4 配下に起票されている。

そのため本 issue の範囲を「内部クレートでの実装・`BackendOps` の既定 `Unsupported`
メソッド追加・`pub(crate) enum Op` の variant 追加・保留ガード・本決定記録」までとした
（先例: #2150 のコミット `0d3da152`）。facade 公開面と §1 の対象範囲表の拡張は含まない。
**公開面や対象範囲の拡張について、承認済みとは記録していない**（§7）。

## §2 実装方式

| 層 | 置き場所 | 内容 |
|---|---|---|
| 共有カーネル（単一情報源） | `crates/tensor-core/src/fft.rs` | `FftNorm`・`FftError`・形状検査 `rfft_layout`／`irfft_layout`・twiddle 表・forward `rfft_host`／`irfft_host`・VJP `rfft_vjp_host`／`irfft_vjp_host` |
| バックエンド抽象 | `crates/tensor-core/src/backend_ops.rs` | `BackendOps::fft_rfft`／`fft_irfft`（既定 `Unsupported`） |
| CPU | `crates/backend-cpu/src/ops.rs` | 上記 2 メソッドのオーバーライド（共有カーネルを呼ぶだけ） |
| CUDA／Metal | 変更なし | 既定 `Unsupported` のまま（GPU 専用カーネルは対象外） |
| autodiff | `crates/autodiff/src/fft_ops.rs`・`tape.rs`・`grad.rs` | 自由関数 `rfft`／`irfft`・`Op::Rfft`／`Op::Irfft`・VJP |
| facade | `crates/facade/src/lib.rs`・`tests/api_surface.rs` | 保留ガードのみ（公開なし） |

- 共有カーネルは autodiff のホストフォールバックと backend-cpu の両方が同じ関数を呼ぶ
  （`interpolate.rs` と同じ単一情報源方式。linalg のように複製しない）。
- `fft_ops::{rfft, irfft}` は ①形状・確保サイズ検査 → ②入力の実体化 → ③`BackendOps` 呼び出し
  （`Unsupported` のときだけホストカーネルへフォールバック。他のエラーは伝播）→ ④`Op` 記録。
- `Op::Rfft`／`Op::Irfft` は解決済みの `n`／`dim`／`norm` を保持する。非融合・非 checkpoint
  （`is_checkpoint_eligible` は `false`）・高階微分非対応（`supports_create_graph` は `false`）。
  いずれもワイルドカード腕に頼らず網羅 `match` へ明示的に足した。
- **命名規律**: 素の `fn rfft`／`fn irfft` の宣言は `autodiff/src/fft_ops.rs` の各 1 件のみ。
  trait メソッドは `fft_rfft`／`fft_irfft`、共有カーネルは `*_host` と別名にして、workspace
  走査のインベントリ（§9）と衝突させない。#2632 が `fft`／`ifft` を足すときも同じ規律。

## §3 数値契約

- 内部は `f64` で**逐次・固定順序**に蓄積し、出力時に 1 回だけ `f32` へ downcast する。
  蓄積は素の `acc += a * b`（`mul_add` を使わない）。`f64` 内部精度のため FMA の有無は
  REQ-2 統一複合判定の範囲内で、matmul 系の FMA 契約には触れない。
- twiddle は長さ `n` の `(cos, sin)` 表を 1 変換につき 1 回作る。1/4 turn・1/2 turn の添字は
  `sin_cos` を使わず厳密値（`0`／`±1`）へ上書きする。添字は `idx = (idx + k) % n` の増分更新
  （`k·j` の乗算オーバーフローを避ける。設計 §4 の `(k·j) mod n` と同値）。
- **DC と（`n` 偶数の）Nyquist の虚部は、蓄積結果ではなくリテラル `+0.0` を書き込む**
  （rfft forward・irfft VJP）。厳密 twiddle に加えた実装上の補強で、`-0.0` と非有限入力での
  `inf·0 = NaN` を避ける。irfft forward は Hermitian 折り畳み形で、これらの虚部を構造的に読まない。
- 非有限入力は事前に拒否せず伝播する（設計 §6。linalg 系の「非有限は `InvalidArgument`」とは異なる）。
- `sin_cos` は `libm` 依存のため、クレート間 bit 同一は受入条件にしない（REQ-2 判定で比較）。同一入力の
  run-to-run は bit 一致。
- 計算量は 1 レーンあたり O(n²)（直接 DFT）。O(n log n) 化は対象外。恣意的な `n` 上限は新設せず、
  確保サイズだけを fail-closed に検査する。

## §4 境界検査（REQ-8・OWASP A03）

`rfft_layout`／`irfft_layout` が確保・実体化より前に次を検査し、型付きエラーで拒否する。

| 拒否理由 | エラー |
|---|---|
| rank 不足（rfft は 1 未満・irfft は 2 未満） | `Shape(RankMismatch)` |
| irfft の末尾次元 ≠ 2 | `Shape(ShapeMismatch)` |
| 入出力の要素数・バイト数の `checked_mul` オーバーフロー・`isize::MAX` 超過 | `Shape(ElementCountOverflow)` |
| `dim` 範囲外（irfft は複素軸を除く）・解決後 `n == 0`（`n = 0` 明示、irfft で `m == 1` かつ `n` 省略、rfft で長さ 0 かつ `n` 省略）・irfft の入力 bin 数 0 | `InvalidArgument` |

autodiff 側は `AutodiffError::{Shape, InvalidArgument}` へ、バックエンド側は
`BackendError::{ShapeMismatch, InvalidArgument}` へ写像する。カーネルはスライス長も再検査し、
`unsafe`／`get_unchecked` を使わない。フォールバック条件は `BackendError::Unsupported` のみで、
それ以外のエラーを握りつぶさない（A08）。

## §5 PyTorch 2.14.0 との差分

- 複素数は `view_as_real` 形式の実テンソルで受け渡す（complex dtype は非目標）。
- `dim` は `usize`（負の添字は受けない）。irfft の `dim` は複素軸を除いた実軸の添字。
- 実測（`tests/fixtures/fft-pytorch-reference/` の `error_cases`）: torch が例外を出すのは
  `n = 0`・`dim` 範囲外・`rfft` で長さ 0 かつ `n` 省略・`irfft` で `m = 1` かつ `n` 省略。
  いずれも本実装も拒否する。torch が受理する `rfft`（長さ 0 の軸で `n` 明示）も本実装は受理する。
- 本実装が拒否し torch の挙動を未実測のもの: `irfft` の入力 bin 数 `m = 0`（カーネルが先頭 bin を
  読むため拒否する安全側）。差分として記録するのみで、必要なら #2632 以降で実測する。
- 非有限入力は拒否せず伝播する（torch と同じ）。

## §6 テスト構成

- `crates/tensor-core/src/fft.rs`（単体）: twiddle 厳密値・デルタ → 定数・`cos` → スペクトル線・
  n = 1／2／3（奇数）・ゼロ詰め／切り詰め・DC／Nyquist 虚部の `to_bits() == 0`・随伴恒等式
  （norm 3 種・`n` が入力より大／小）・往復・決定性・境界エラー。
- `crates/backend-cpu/tests/fft_parity.rs`・`backend_ops_dispatch.rs`: `CpuBackendOps` 直接呼び出しの
  解析解・決定性・型付きエラー、CUDA／Metal の `Unsupported`（panic しない）。
- `crates/autodiff/tests/fft_parity.rs`: **実 PyTorch 2.14.0 の実行値 fixture 33 ケース**の forward・
  勾配を REQ-2 統一複合判定（`common::req2_close`。tolerance 定数は新設しない）で突合。DFT 行列積
  オラクル（設計 §4 案 A）との forward・backward 突合・中心差分・往復・DC／Nyquist 虚部の bit 0・
  決定性・境界エラー・巨大 `n` の確保前拒否・非有限入力の伝播・`Unsupported` 以外のバックエンド
  エラーの伝播（モック ops）。
- `crates/facade/tests/fft_ops_backend_parity.rs`: CPU tape 対 naive tape の REQ-2 突合。CUDA／Metal は
  `#[ignore]`（§10）。
- fixture の出自（`torch.__version__`・シード・sha256・再生成手順）は
  `crates/autodiff/tests/fixtures/fft-pytorch-reference/README.md`。numpy や手計算値で代替していない。

## §7 facade 公開形の推奨案（未承認）

**推奨案（1 つ）**: `Var::rfft`／`Var::irfft` を `fft_ops` への 1 行委譲メソッドとして公開し、
`FftNorm` を `fandhe_ai` ルートへ再エクスポートする（`fft_ops` モジュール自体は再エクスポートしない）。
`InterpolateMode`・`MatrixNormOrd` の再エクスポートと linalg 5 演算の委譲メソッド（#2515）と同型で、
facade が唯一のサポート公開面であるという方針（`docs/compat-api-scope.md` §0）に沿う。

**承認事項（すべて未承認。本 issue では実施しない）**:

1. 上記の公開形（`Var::rfft`／`Var::irfft`・`FftNorm` の再エクスポート）。承認依頼は #2677、公開は承認後の #2678。
2. `docs/compat-api-scope.md` §1 の対象範囲への FFT 行の追加（同 §5 の手続き）。
3. 将来の GPU バタフライカーネルで baseline 方式を採る場合の baseline 値（実機実測値のみ・人間承認必須）。

## §8 スコープ外

- `fft`／`ifft`（#2632）・`stft`／`istft`（#2633）
- facade 公開と `compat-api-scope.md` §1／`compat-feature-gap.md` の判定変更、spec（REQ-9）改定
- GPU 専用 FFT カーネル・O(n log n) 化（radix-2・Bluestein）・`fft2`／`fftn`／`hfft`／`ihfft`／`fftshift`／`fftfreq`・ONNX `DFT`
- `create_graph`（高階微分）・activation checkpoint 対象化・f64 自動微分経路での FFT
- CUDA（DGX Spark GB10）・Metal（M4 Max）実機計測（§10 の申し送りのみ）

## §9 多層防御（facade 非公開の機械固定）

1. 正のプローブ: `crates/facade/src/lib.rs` の `FftOpsHoldDoctestGuard`（ローカル `__fandhe_fft_hold_probe`
   と全 `pub mod` glob import の共存・`Var`／`Tape` への `Type::method` 形式のプローブ）。
2. `crates/facade/tests/api_surface.rs`:
   `fft_ops_hold_doctest_globs_all_pub_modules`（glob 集合が `pub mod` 宣言集合と一致）・
   `fft_ops_hold_doctest_probe_body_matches_fixed_contract`（本文が `FFT_OPS_HOLD_PROBE_BODY` と 1 行単位で一致）・
   `facade_does_not_reexport_or_declare_fft_ops`（`pub use` に `fft_ops`／`FftNorm`／`FftError`／`fft` を識別子
   単位で含まない・内部クレートの glob 再エクスポートなし・同名 `fn`／`pub mod` 宣言なし）と
   `..._detects_each_category`（合成ソースで検出できることの恒久固定）・
   `workspace_declares_fft_ops_fn_names_only_in_allowed_locations`（`rfft`／`irfft` の `fn` 宣言が
   `autodiff/src/fft_ops.rs` の各 1 件のみ）。
3. stable rustdoc は `compile_fail` のコードを照合しないため、否定ガードは正のプローブ＋インベントリで構成した。
4. 承認後（#2678）は本ガードを正ガードへ反転する（`linalg_ops` の #2515 と同じ運用）。

## §10 実機申し送り

CUDA（DGX Spark GB10）・Metal（Apple Silicon）実機は本実装エージェント実行環境から到達できず未実測。
`crates/facade/tests/fft_ops_backend_parity.rs` の `cuda_fft_matches_cpu_reference`／
`metal_fft_matches_cpu_reference` は `#[ignore]` のまま。測定コマンド・期待結果・記入欄は
`docs/perf/logs/fft-rfft-irfft-2631/README.md`。CUDA／Metal は FFT の GPU カーネルを持たず既定
`Unsupported` からホスト計算へフォールバックするため、この比較は GPU カーネルの parity ではない。

## §11 出典

| 出典 | 内容 |
|---|---|
| `docs/autodiff-fft-design.md` | 設計判断記録（案 B・VJP 式・数値契約・境界検査） |
| `docs/autodiff-linalg-ops-decision.md`・コミット `0d3da152` | 同型の先例（内部実装＋保留ガード） |
| `docs/compat-api-scope.md` §0・§5 | 公開面の方針・範囲拡張手続き |
| `crates/tensor-core/src/interpolate.rs` | 共有カーネル方式の先例 |
| `.claude/rules/coding-rust.md` | REQ-2 統一複合判定・境界検査・FMA 契約 |
