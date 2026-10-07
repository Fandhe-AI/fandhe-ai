# rfft・irfft の内部実装記録（FFT 第 1 弾）

イシュー #2631（親 #2630「FFT」・ルート #2499 Phase 4）。基準コミット
`7711a3ac`（main HEAD）。設計の正は `docs/autodiff-fft-design.md`（#2151。案 B）で、
本書はその実装記録である。`fft`／`ifft` は #2632（§12 に追記済み）、`stft`／`istft` は #2633（§13 に追記済み）。

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

- `fft`／`ifft`（#2632。§12 で実装済み）・`stft`／`istft`（#2633。§13 で実装済み）
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

## §12 fft・ifft の追記（#2632）

イシュー #2632（親 #2630・ルート #2499 Phase 4）。基準コミット `b3c5df45`（#2631 マージ後の main）。
`fft`／`ifft`（複素 → 複素の c2c。入出力とも末尾次元 2 の `f32` 実テンソル `(re, im)`）を、`rfft`／`irfft`
（§0〜§11）と同じ内部クレート限定の方式で追加した。着手根拠は §1 と同じ整理（ルート #2499 Phase 4 の
「内部実装＋保留ガードまで先行・公開は承認後」）で、facade 公開の承認は得ていない。

### 12.1 PyTorch 相当・API

| 演算 | PyTorch 相当 | 入力 → 出力 |
|---|---|---|
| `fft_ops::fft(x, n, dim, norm)` | `view_as_real(torch.fft.fft(view_as_complex(x), n, dim, norm))` | `[..., L, ..., 2]` → `[..., n, ..., 2]` |
| `fft_ops::ifft(x, n, dim, norm)` | 同 `torch.fft.ifft` | 同上 |

`n` 省略時は `L`、`dim` は複素軸を除いた実軸の添字（既定 rank-2・負の添字は受けない）。変換前に `dim` 軸を
長さ `n` へ切り詰め／ゼロ詰めする。

### 12.2 実装方式（単一情報源）

`crates/tensor-core/src/fft.rs` の 4 カーネル（`fft_host`／`ifft_host`／`fft_vjp_host`／`ifft_vjp_host`）は、
符号 σ とスケール c だけが異なる 1 本の非公開コア `c2c_core` で実装した。

| 公開関数 | σ | c | src_len → dst_len |
|---|---|---|---|
| `fft_host` | -1 | `norm.forward_scale(n)` | `L` → `n` |
| `ifft_host` | +1 | `norm.inverse_scale(n)` | `L` → `n` |
| `fft_vjp_host` | +1 | `norm.forward_scale(n)` | `n` → `L` |
| `ifft_vjp_host` | -1 | `norm.inverse_scale(n)` | `n` → `L` |

随伴は共役転置（σ 反転・長さ入替）で、`n` 倍は掛けず、`ortho`／`forward` でも二重スケールしない。
ゼロ詰めの VJP は切り詰め、切り詰めの VJP はゼロ詰めになる（設計 §5）。経路は `rfft` と同じ
（`fft_layout` 検査 → 実体化 → `BackendOps::fft_fft`／`fft_ifft`〈既定 `Unsupported` のときだけ
ホストカーネルへフォールバック〉→ `Op::Fft`／`Op::Ifft`〈非融合・非 checkpoint・高階微分非対応〉）。
`backend-cpu` の `CpuBackendOps::fft_fft`／`fft_ifft` は `fft_layout` で再検査して共有カーネルを呼ぶだけ。
CUDA／Metal は既定 `Unsupported`（GPU カーネルなし）。

### 12.3 数値契約（§3 との差分）

- `f64` 逐次・固定順序（`s` 昇順）の素の加算で蓄積し、最後に 1 回だけ `f32` へ downcast する。`mul_add` は使わない。
- **c2c には DC／Nyquist の構造的ゼロが無い**ため、rfft のリテラル `+0.0` 書き込み規則は持ち込まない。
- 非有限入力は拒否せず伝播する（直接 DFT のため `inf·0 = NaN` もそのまま出る）。NaN／inf の配置が
  PyTorch（FFT アルゴリズム由来）と一致することは保証しない。
- 空要素（出力 0 要素）は twiddle 表の確保より前に早期 return する。

### 12.4 境界検査（`fft_layout`。確保・実体化より前）

| 拒否理由 | エラー |
|---|---|
| rank < 2 | `Shape(RankMismatch)` |
| 末尾次元 ≠ 2 | `Shape(ShapeMismatch)` |
| `dim >= rank-1` | `InvalidArgument` |
| 解決後 `n == 0`（`n = 0` 明示／`L == 0` で `n` 省略） | `InvalidArgument` |
| 入出力の要素数・バイト数の `checked_mul` 超過・`isize::MAX` 超過 | `Shape(ElementCountOverflow)` |

カーネルはスライス長も再検査し、`unsafe`／`get_unchecked` を使わない。計算量は 1 レーン O(n²)（直接 DFT）。
恣意的な `n` 上限は新設せず、表現可能だが巨大な `n` での確保失敗・長時間計算は 4 演算共通の既知の性質。

### 12.5 PyTorch 2.14.0 との差分

- 実入力 `[..., L]` の虚部 0 への自動昇格はしない（呼び出し側の責務。設計 §3）。
- `dim` は `usize`（負の添字は受けない）。
- `error_cases` の実測（torch 2.14.0+cpu）: `n=0`・`dim` 範囲外・空軸で `n` 省略は torch が例外、
  **空軸（長さ 0）で `n=4` 明示は torch が受理**（ゼロ詰め）。`fft`／`ifft` とも本実装は同じ判定。
  rank < 2・末尾 ≠ 2 は torch 側で表現できない（複素テンソルを前提とする）ため Rust 専用の拒否。
- 非有限入力の NaN／inf の配置は 12.3 のとおり一致を保証しない。

### 12.6 テスト構成

- fixture: `crates/autodiff/tests/fixtures/fft-pytorch-reference/`（実 PyTorch 2.14.0 実行値。rfft 18＋irfft 15＋
  fft 18＋ifft 18 = 計 69 ケース）。既存 33 ケースは再生成前後で完全一致することを機械比較して確認した
  （c2c は乱数列の末尾側で生成）。REQ-2 統一複合判定（`common::req2_close`）で forward・入力勾配を突合。
  tolerance 定数は新設・変更していない。
- `crates/autodiff/tests/fft_parity.rs`: DFT 行列積オラクル（cos・sin 行列 2 本の `matmul` 合成。norm 3 種・`n ≠ L`）、
  中心差分、`fft → ifft` 往復、bit 決定性、非有限入力の伝播、境界エラー、巨大 `n` の確保前拒否、
  モック `BackendOps` による `Unsupported` 以外のエラー伝播・フォールバック・誤 shape 戻り値の拒否。
- `crates/tensor-core/src/fft.rs` 単体: 随伴恒等式（norm 3 種 × `n` 省略／`n>L`／`n<L`）・解析解・往復・
  rfft との先頭 bin 一致・非末尾 `dim`・型付きエラー。
- `crates/backend-cpu/tests/fft_parity.rs`・`backend_ops_dispatch.rs`、
  `crates/facade/tests/fft_ops_backend_parity.rs`（CPU tape 対 naive tape。実機 2 件は `#[ignore]`）。

### 12.7 facade 公開形の推奨案（未承認）

**推奨案（1 つ）**: §7 と同一方針。`Var::fft`／`Var::ifft` を `fft_ops` への 1 行委譲メソッドとして公開し、
`FftNorm` を `fandhe_ai` ルートへ再エクスポートする（`fft_ops` モジュール自体は再エクスポートしない）。
`Sequential::add_*` は層ではないため対象外。**未承認**で、承認依頼は #2677、公開自体は承認後の #2678。

### 12.8 保留ガードの更新点

`FftOpsHoldDoctestGuard` のプローブへ `fft`／`ifft`（`fft_ops` 内の自由関数・`Var`／`Tape` の `Type::method`
呼び出し）を追加し、`api_surface.rs` の `FFT_OPS_HOLD_PROBE_BODY`・`FFT_OPS_FN_NAMES`
（`["rfft", "irfft", "fft", "ifft"]`）・自己テスト・workspace インベントリ期待値
（`autodiff/src/fft_ops.rs` の 4 関数が各 1 件）を更新した。素の `fn fft`／`fn ifft` の宣言は
`fft_ops.rs` の各 1 件だけで、trait メソッドは `fft_fft`／`fft_ifft`、共有カーネルは `*_host`、
形状検査は `fft_layout` と命名して衝突させない。公開面の追加は一切ない。

### 12.9 スコープ外・実機申し送り

- スコープ外: facade 公開（#2678）・`compat-api-scope.md` §1／`compat-feature-gap.md` の判定変更・
  `stft`／`istft`（#2633）・`fft2`／`fftn` 等・GPU 専用カーネル・O(n log n) 化・高階微分・checkpoint・f64 経路。
- 実機（CUDA／Metal）は未実測のまま `#[ignore]`（`cuda_fft_c2c_matches_cpu_reference`／
  `metal_fft_c2c_matches_cpu_reference`）。申し送りは `docs/perf/logs/fft-fft-ifft-2632/README.md`。

## §13 stft・istft の追記（#2633）

イシュー #2633（親 #2630・ルート #2499 Phase 4）。基準コミット `b41b73af`（#2632 マージ後の main）。
`stft`／`istft`（窓付き短時間フーリエ変換とその逆変換）を、§0〜§12 と同じ内部クレート限定の方式で追加した。
`docs/autodiff-fft-design.md` §9 は STFT を「スコープ外」とし設計判断を記録していないため、本節が STFT の
設計判断の記録である。着手根拠は §1 と同じ整理（ルート #2499 Phase 4 の「内部実装＋保留ガードまで先行・
公開は承認後」）で、facade 公開の承認は得ていない。

### 13.1 PyTorch 相当・API

| 演算 | PyTorch 相当 | 入力 → 出力 |
|---|---|---|
| `fft_ops::stft(x, n_fft, window, &StftOptions)` | `view_as_real(torch.stft(x, n_fft, hop_length, win_length, window, center, pad_mode, normalized, onesided, return_complex=True))` | 実 `[L]`／`[B, L]` → `[N, T, 2]`／`[B, N, T, 2]`（**周波数軸がフレーム軸より前**） |
| `fft_ops::istft(x, n_fft, window, &IstftOptions)` | `torch.istft(view_as_complex(x), n_fft, hop_length, win_length, window, center, normalized, onesided, length)` | `[N, T, 2]`／`[B, N, T, 2]` → 実 `[L_out]`／`[B, L_out]` |

`N = n_fft/2+1`（片側）または `n_fft`、`T = 1 + (L_pad − n_fft)/hop`（`L_pad = L + 2·(n_fft/2)`〈`center`〉）。
`StftOptions`（`hop_length`〈既定 `n_fft/4`〉・`win_length`〈既定 `n_fft`〉・`center = true`・`pad_mode = Reflect`・
`normalized = false`・`onesided = true`）と `IstftOptions`（`hop_length`・`win_length`・`center = true`・
`normalized = false`・`onesided: Option<bool>`〈`None` は入力 bin 数 ≠ `n_fft` なら片側と推定〉・`length`）は
`#[non_exhaustive]`＋`Default`（PyTorch 既定）＋`with_*` ビルダ（`TopkOptions` の先例）。`StftPadMode`
（`Reflect`／`Constant`。`#[non_exhaustive]`）・`StftParams`／`IstftParams`（検査付きコンストラクタ経由でのみ
生成）は `tensor-core::fft` に置く。`normalized` は順変換 `1/√n_fft`・逆変換 `1/√n_fft`（否なら順変換は
スケールなし・逆変換は `1/n_fft`）で、fixture が確認している。

### 13.2 実装方式（専用 `Op`＋共有ホストカーネル）

- 共有カーネルは `crates/tensor-core/src/fft/stft.rs`（`fft.rs` の子モジュール）。フレームごとの変換は既存の
  `rfft_host`／`rfft_vjp_host`／`irfft_host`／`irfft_vjp_host` を `[B·T, n_fft]` レイアウト（`dim = 1`）で呼ぶ
  だけで、**マージ済みカーネルの演算順序は変えていない**（既存 69 ケースの fixture 値・sha256 は不変）。
  新規の数式は「フレーム切り出し・端パディング写像・窓掛け・重畳加算・包絡除算」の添字計算だけ。
- `BackendOps::fft_stft`／`fft_istft`（既定 `Unsupported`）→ `Unsupported` のときだけ `stft_host`／`istft_host`
  へフォールバック（他のエラーは伝播・`Ok` は shape 検証）→ `Op::Stft`／`Op::Istft`（非融合・非 checkpoint・
  高階微分非対応）。`backend-cpu` の `CpuBackendOps` は `*_layout` で再検査して共有カーネルを呼ぶだけ。
  CUDA／Metal は既定 `Unsupported`（GPU カーネルなし）。
- 不採用: 既存 `Var` 演算の合成（`index_select`→`mul`→`rfft`→`permute`、逆は `irfft`→`mul`→`scatter_add`→`div`）。
  技術的には可能だが、(i) 「CUDA／Metal は既定 `Unsupported`→ホスト計算」の契約から外れ GPU 上で gather／scatter の
  実カーネルが走る未実測の組み合わせになる、(ii) 単一情報源のカーネルを `backend-cpu` と共有できない、
  (iii) テープに多数のノードと `B·T·n_fft` 要素の添字が載る、ため採らない。テストでは独立実装の `f64` 直接 DFT
  オラクル（フレーム化・端パディング・重畳加算を別の書き方で再実装）で突合する（合成そのものは使っていない）。

### 13.3 数値契約

- フレームの窓掛けは `f32` 積（PyTorch と同じ丸め位置）→ 既存カーネルの `f64` 逐次・固定順序の変換。
- `istft` の重畳加算・窓包絡・除算と、`stft` VJP の散布加算（反射パディング・重畳の随伴）は **`f64` アキュムレータに
  `t` 昇順 → `j` 昇順で蓄積し最後に 1 回だけ `f32` へ downcast**（勾配の長軸縮約の f64 契約と同方針）。`mul_add` は
  使わない。matmul 系 FMA 契約には触れない。
- 同一入力は run-to-run で bit 一致。クレート間 bit 同一は受入条件にしない（`sin_cos` が libm 依存。REQ-2 で判定）。
- 非有限入力は拒否せず伝播する（§3 と同じ）。DC／Nyquist 虚部のリテラル `+0.0` は `rfft_host` 由来でそのまま保たれる。
- 計算量はフレームあたり O(`n_fft`²)（直接 DFT）。恣意的な上限は新設せず、確保サイズだけを fail-closed に検査する。

### 13.4 境界検査（確保・実体化より前）

| 拒否理由 | エラー | 検査箇所 |
|---|---|---|
| `n_fft = 0`・`hop_length = 0`（省略時の既定 `n_fft/4` が 0 になる `n_fft < 4` を含む） | `InvalidArgument` | `StftParams::new`／`IstftParams::new` |
| `win_length` が 0 または `n_fft` 超・窓長 ≠ `win_length`・窓が rank 1 でない | `InvalidArgument`／`Shape(RankMismatch)` | `stft_window`・`fft_ops` |
| `istft` の `hop_length > win_length`・`length = 0` | `InvalidArgument` | `IstftParams::new` |
| `stft` 入力 rank が 1・2 以外・空次元 | `Shape(RankMismatch)`／`InvalidArgument` | `stft_layout` |
| 反射パディング幅 `n_fft/2 >= L`・`n_fft > L_pad` | `InvalidArgument` | `stft_layout` |
| `istft` 入力 rank が 3・4 以外・末尾次元 ≠ 2・空次元 | `Shape`／`InvalidArgument` | `istft_layout` |
| bin 数が `onesided` の期待値（`n_fft/2+1` または `n_fft`）と不一致 | `InvalidArgument` | `istft_layout` |
| 期待長・`start + length`・フレーム数・作業／出力バッファの `checked_add`／`checked_mul` 超過・バイト数 `> isize::MAX`（`f64` 幅で保守的に評価） | `Shape(ElementCountOverflow)` | 両 `*_layout` |
| 出力長 0 以下（`center` かつ `T = 1`・偶数 `n_fft` で `length` 省略 等） | `InvalidArgument` | `istft_layout` |
| NOLA 違反（窓二乗の重畳和の最小絶対値 `< 1e-11`） | `InvalidArgument` | `istft_check_nola` |

NOLA 検査はカーネル内と `fft_ops::istft`（バックエンド呼び出し前）の両方で行い、バックエンド実装が迂回できない
（バックエンドが `Ok` を返す構成でも拒否されることをテストで固定）。カーネルはスライス長も再検査し、
添字写像の結果は境界検査付きアクセスで読む（`unsafe`／`get_unchecked` 不使用）。

**NOLA しきい値 `1e-11`（`fft::ISTFT_NOLA_MIN_ENVELOPE`）**: 演算の意味論を PyTorch 2.14.0 の実測に合わせて
定めた**新設定数**であり、REQ-2 の tolerance・ガードレール閾値・テスト許容誤差ではない（既存定数は一切変更していない）。
実測の境界ペア（矩形窓・`hop = n_fft = 8`・`center = false`・窓先頭値のみ変更）で、先頭値 `3.1e-6`（包絡 9.61e-12）は
torch が拒否・`3.2e-6`（包絡 1.024e-11）は受理だった。Rust 側のしきい値 `1e-11` はこの区間に収まり、
`nola_threshold_boundary_matches_pytorch_pair` が同じペアで同じ判定になることを固定している（区間内の厳密な
境界値は実測で特定していない。fixture が確認しているのはこの区間と定性的な拒否まで）。

### 13.5 PyTorch 2.14.0 との差分（実測ベース）

| 項目 | 本実装 | 備考 |
|---|---|---|
| 窓への勾配 | **流れない** | 窓は非追跡の `Tensor<f32>`（`Op` の payload）。torch は窓も微分対象にできる。追跡窓は将来拡張 |
| `pad_mode` | `Reflect`／`Constant` のみ | torch の `replicate`／`circular` は非対応（`#[non_exhaustive]` のため後から足せる） |
| `istft` の `onesided = false` | 先頭 `n_fft/2+1` bin だけを読む c2r | **fixture 実測で確定**（非 Hermitian 入力の forward と勾配が torch と一致することを fixture で確認。内部実装の推測はしない）。非 Hermitian 入力でも forward が一致し、上位 bin の勾配は 0。DC／Nyquist 虚部は読まない |
| 入力 rank | `stft` は 1・2、`istft` は 3・4（末尾 2） | torch の複素 rank 1・2／2・3 に対応。それ以外は torch も例外 |
| 空入力 | 0 要素の次元は拒否 | torch も `L = 0`・`B = 0` は例外 |
| `hop_length > win_length` | `stft` は受理・`istft` は拒否 | torch の実測と同じ |
| `istft` で `n_fft = 1`・`center = true`・`length` 省略 | 受理（長さ `L`） | torch は終端 `-(n_fft/2) = -0` を 0 と解釈して空出力となり例外。`stft_parity.rs` の明示的な許可リストに載せている |
| 複素入力の `stft`・`return_complex = false` の `istft` 相当・`align_to_window`・窓生成関数（`hann_window` 等） | 非対応 | スコープ外 |
| 非有限入力 | 拒否せず伝播 | NaN／inf の配置が torch と一致することは保証しない |

`error_cases`（37 件）で torch の拒否有無を実測し、上記の 1 件を除いて Rust 側の拒否有無が一致する
（`error_cases_follow_pytorch_rejections`）。実測結果の詳細は `fixtures/fft-pytorch-reference/README.md`。

### 13.6 テスト構成

- fixture: `crates/autodiff/tests/fixtures/fft-pytorch-reference/stft_reference.json`（実 PyTorch 2.14.0 実行値。
  stft 21＋istft 21〈torch の `stft` 出力を入れる往復 1 件を含む〉= 計 42 ケース。生成は `gen_stft_reference.py`・
  sha256 は README に記録）。**既存の `fft_reference.json`／`gen_reference.py`（69 ケース）は変更していない**。
  REQ-2 統一複合判定（`common::req2_close`）で forward・入力勾配を突合。tolerance 定数は新設・変更していない。
- `crates/autodiff/tests/stft_parity.rs`: fixture 突合・`error_cases` 突合・独立実装の `f64` 直接 DFT オラクル
  （`center`×`pad_mode`×`normalized`×`onesided`×`hop`×`win_length`、`length` 有無）・中心差分・`stft → istft` 往復・
  bit 決定性・DC／Nyquist 虚部の bit 0・境界エラー・NOLA 境界ペア・確保前拒否・非有限入力の伝播・モック
  `BackendOps`（`Unsupported` 以外のエラー伝播・フォールバック・誤 shape 拒否・NOLA 違反の迂回拒否）。
- `crates/tensor-core/src/fft/stft.rs` 単体: 検査表の各行・添字写像（反射・定数）・解析解（`hop = n_fft`・
  `center = false` はブロックごとの `rfft_host` と一致）・**随伴恒等式** `⟨A x, g⟩ = ⟨x, Aᵀ g⟩`
  （`center`×`pad_mode`×`onesided`×`normalized`×`hop`×`win_length < n_fft`×奇数 `n_fft`、`stft`・`istft` 両方）・
  `istft(stft(x)) ≈ x`・NOLA 拒否・共役対称・巨大サイズの確保前拒否。
- `crates/backend-cpu/tests/fft_parity.rs`・`backend_ops_dispatch.rs`（CPU 直接呼び出し・CUDA／Metal の `Unsupported`）、
  `crates/facade/tests/fft_ops_backend_parity.rs`（CPU tape 対 naive tape。実機 2 件は `#[ignore]`）。

### 13.7 facade 公開形の推奨案（未承認）

**推奨案（1 つ）**: §7・§12.7 と同一方針。`Var::stft`／`Var::istft` を `fft_ops` への 1 行委譲メソッドとして公開し、
`StftOptions`／`IstftOptions`／`StftPadMode` を `fandhe_ai` ルートへ再エクスポートする（`fft_ops` モジュール自体は
再エクスポートしない）。`Sequential::add_*` は層ではないため対象外。**未承認**で、承認依頼は #2677、公開自体は
承認後の #2678。本 issue では一切公開していない。

### 13.8 保留ガードの更新点

`FftOpsHoldDoctestGuard` のプローブへ `stft`／`istft`（`fft_ops` 内の自由関数・`Var`／`Tape` の `Type::method`
呼び出し）と型 `StftOptions`／`IstftOptions`／`StftPadMode`（`__probe_free_fns` の引数型として実際に使用）を追加し、
`api_surface.rs` の `FFT_OPS_HOLD_PROBE_BODY`・`FFT_OPS_FN_NAMES`（6 件）・`FFT_OPS_IDENTS`（`StftOptions`・
`IstftOptions`・`StftPadMode`・`StftParams`・`IstftParams` を追加）・型宣言検出の対象名リスト・自己テスト・
workspace インベントリ期待値（`autodiff/src/fft_ops.rs` の `stft`／`istft` が各 1 件）を更新した。素の
`fn stft`／`fn istft` の宣言は `fft_ops.rs` の各 1 件だけで、trait メソッドは `fft_stft`／`fft_istft`、共有カーネルは
`stft_host`／`istft_host` 等、形状検査は `stft_layout`／`istft_layout` と命名して衝突させない。公開面の追加は一切ない。

### 13.9 スコープ外・実機申し送り

- スコープ外: facade 公開（#2677→#2678）・`compat-api-scope.md` §1／`compat-feature-gap.md` の判定変更・spec 改定・
  追跡窓（窓への勾配）・複素入力 `stft`／複素出力 `istft`・`pad_mode` の `replicate`／`circular`・`align_to_window`・
  窓生成関数・GPU 専用カーネル・O(n log n) 化・`create_graph`・checkpoint・f64 自動微分経路・CUDA／Metal 実機計測。
- 実機（CUDA／Metal）は未実測のまま `#[ignore]`（`cuda_stft_matches_cpu_reference`／`metal_stft_matches_cpu_reference`）。
  申し送りは `docs/perf/logs/fft-stft-istft-2633/README.md`。

## 付録. #2678 実装記録（Phase 4 の facade 公開）

- 状態: **§7 の公開形を #2678 で承認形どおり公開した。** 承認根拠はルート #2499 の 2026-10-07 ユーザー承認コメント（issuecomment-6033824965。「Phase 4（#2625）」節で `docs/compat-api-scope.md` §5.1 の行 1を各決定記録の推奨形で承認）。本書中の「未承認」「承認依頼は #2677」の記述は、#2678 時点で当該コメントの承認に更新された（承認は推奨形に限り、記録に形が書かれていない点は実装せず承認依頼へ戻す条件つき）。
- 公開した識別子: `Var::{rfft,irfft,fft,ifft}(&self, n: Option<usize>, dim: Option<usize>, norm: FftNorm)`・`Var::stft(&self, n_fft, window: Option<&Tensor<f32>>, options: &StftOptions)`・`Var::istft(.., options: &IstftOptions)`、クレートルートへ `FftNorm`・`StftPadMode`・`StftOptions`・`IstftOptions` の再エクスポート（`fft_ops`／`fft` モジュールは再エクスポートしない）。本体は `crate::<module>::<fn>` への 1 行委譲（`Var`）／`&self.0` を渡すだけの 1 行委譲（`Tape`）に固定し、新規 `Op`・`BackendOps` メソッド・`AutodiffError` variant・`unsafe` は追加していない。
- ガードの反転・縮小: `FftOpsHoldDoctestGuard` から `Var` の impl・UFCS 行と公開した 4 型のローカル定義を外し、`fft_ops`／`fft` モジュール名・裸の自由関数・`Tape` 上の同名メソッドのプローブだけを残した（先例 #2516）。`api_surface.rs` の否定ガードは、承認済みの型名を識別子表から外し、`Tape` の承認済みメソッドを `fn` 宣言走査から除外したうえで、承認形だけを許す正ガードへ反転した。宣言場所インベントリには `autodiff/src/var.rs`（`Tape` 分は `facade/src/lib.rs`）の各 1 件を追加した。
- テスト: `crates/facade/tests/phase4_ops_facade.rs`（fft_methods_roundtrip。`fandhe_ai::` だけを import し、fn ポインタ型でシグネチャを固定して厳密に決まる値を確認）と、`crates/facade/tests/api_surface.rs` の正ガード（`var_phase4_ops_methods_are_thin_delegations`・`facade_reexports_phase4_ops_types_only_in_approved_shape`・`facade_tape_phase4_methods_are_thin_delegations`・各 `workspace_declares_*`）。
- 依存・tolerance・baseline・ガードレール閾値・`docs/spec` は変更していない。`fandhe-ai =0.10.0` の既存公開 API・`pub use` 行・署名は変更せず、追加のみ。CUDA／Metal 実機 parity は未実測で、`docs/perf/logs/phase4-ops-autodiff-exposure-2678/README.md` へ申し送る（新しい数値経路はなく、1 行委譲のため既存の各 `*_backend_parity.rs` の結果がそのまま適用される）。
