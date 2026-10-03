# ROCm 格上げ条件表の v2 再定義（(b) 形式 spec 提案文案）と条件 (c) 充足計画（#2128）

基準コミット: `4be494be`（origin/main。v0.10.0 リリースサイクル後。#2614 で最新化。初版は `e083e609`＝#2125〜#2127 マージ後）。`docs/spec` submodule ポインタは `2e998dd77117814f4af8ed160394ad1d6a8f888a`（`git ls-tree HEAD docs/spec` で確認。初版時点から不変）。除外事項の ROCm bullet は `docs/spec/04-requirements.md:367`（`grep -n "ROCm バックエンドの正式対応" docs/spec/04-requirements.md` で再確認すること。初版の `:365-369` は格上げ条件表を含む範囲の記録で、bullet 本体は `:367`。`docs/chip-optimization-roadmap.md` §7 の食い違い記録と同じ）。submodule 更新で行番号はずれうるため、参照時は再確認すること。

更新履歴: #2128 で初版 → #2614 で v0.10.0 時点へ最新化（§3b 状況節・§3c 依存方式の推奨・§3d #2129／#2499 との関係を追加し、§5 を 2 段構成化、§6・§8 を整合）。**本 doc の推奨はすべて未承認**であり、承認済みの記述はない。

## §0 結論（最初に読む。#2129 が引用する安定した結論表）

- **コード変更なし**（`crates/**`・`Cargo.toml`／`Cargo.lock`・`docs/spec/`〈正本 submodule〉・tolerance／baseline・ガードレール閾値・依存は一切変更しない）。
- §5 の (b) 形式 spec 提案文案は**未起票**（承認事項は §6）。#2614 で **2 段構成**へ改めた: **段 1（Won't→Could の (a)'(b)(c)）は (d)' の閾値と独立に起票できる**（ユーザー承認後）。**段 2（Could→Should の (d)'(e)'(f)'）は軸のみを示し数値閾値は提案しない**（Could 到達後の別提案で確定）。
- **Could 昇格条件の確定案（未承認）**: 段 1 の (a)'(b)(c)（§5）。HIP FFI 依存方式は **B（`libloading` を新区分の直接依存にし、実行時解決する関数ポインタ型宣言を使う）を推奨**（§3c。未承認）。
- #2129 の「ROCm は Non-Goal 確定」推奨との関係は §3d。選ぶのはユーザーであり、本 doc は #2129 の推奨を覆さない。
- 格上げ判定の現状は **Won't（条件付き）のまま据え置く**。v2 の Could 条件のうち (b) は PoC-10 §2 の費用試算に基づく「達成見込み」であり、機材の利用可能性と実施時単価を確認するまで達成済みとは扱わない（要再確認）。
- 条件ごとの v1 → v2 の扱い:

| 条件 | v1（PoC-10・spec 現行） | v2 での扱い | 現状 |
|---|---|---|---|
| (a) | `burn-rocm`／`cubecl-hip` の experimental 解消 | **文言は削除・意図は軽度化して置換**（(a)'） | 未達 |
| (b) | 対象 GPU を約 1 万円以下で確保 | 維持 | **達成見込み（要再確認）**（2026-07-29 時点の費用試算のみ。単価・機材の利用可能性は実施時に再確認） |
| (c) | 実機ビルド・動作確認と出力一致（相対 1e-3 以内） | 意味は維持。判定式を REQ-2 統一複合判定へ揃える | 未達 |
| (d) | CUDA 実測比 70% 以上 | **軸を再定義**（同一 AMD 機材上の基準。数値閾値は確定しない） | 未達 |
| (e) | rocWMMA 経由の行列演算ユニット活用証跡 | 「行列演算ユニット命令の発行証跡」へ一般化 | 未達 |
| (f) | `burn-rocm` の未対応オペの影響なし | 自作 ROCm バックエンドの `BackendOps` 被覆へ置換 | 未達 |

- タイムラインは §8 のとおり**日付ではなく発火条件**で示す。提案の採択（spec 取り込み）と ROCm の Could 格上げは別の判断である。

## §1 位置づけ

イシュー #2128 の成果物。入力:

- `docs/spec/04-requirements.md:365-369`（除外事項「ROCm バックエンドの正式対応」と格上げ条件表）
- `docs/spec/03-poc/poc-10-rocm-promotion/README.md` §2（調達条件）・§3（検証項目）・§4（判断基準）
- `docs/backend-abstraction-amd-readiness-decision.md`（§6b #2125・§6c #2126）
- `docs/rocm-cooperative-hiprtc-unsafe-readiness-decision.md`（#2127。§4 互換性表・§6.3 依存方式・§8 起票案）
- `docs/backend-matrix.md` §3.4

「(b) 形式」とは、実装リポの doc を出典として spec 側には短い規定だけを置く提案形式（先例: `docs/ddp-grade-up-conditions.md`・`docs/tokenizer-non-target-spec-proposal.md`・`docs/spec-proposal-req2-candle-parity-tolerance.md` §2）。spec 本体の要件・判定式を改定する案 (a) 形式とは対になる。

### イシュー記載との不一致（安全側に確定）

| 項目 | イシューの記載 | spec 正本／PoC-10 の実際 | 本 doc での確定 |
|---|---|---|---|
| 条件 (c) | 標準ライブラリ化・言語機能 | (c) は実機でのビルド・動作確認と CPU/Metal/CUDA との出力一致の実測 | **spec の記号を正とする**。言語機能の論点は (c) を実測可能にする前提条件（§3）として扱う |
| 条件 (f) | 実機テスト | (f) は代表ワークロードの実装範囲に未対応オペの影響がないこと | spec の記号を正とする。実機テストは (c)・(d)・(e) そのものに当たり、拠点と機材は §4 にまとめる |
| 想定機材 | Radeon RX 6900 XT | RDNA2（gfx1030）。PoC-10 §2 の 1 行目は RDNA2 以前を非公式扱いとして**除外** | 最低ラインは CDNA（MI200/MI300 系）または RDNA3 |
| ROCm 版 | 5.6〜6.x | PoC-10 §2 の 2 行目は 2026-07 時点の最新安定を 7.2 系とする | 範囲を固定せず「実機確保時点の最新安定系列（対象 gfx の Compatibility Matrix 掲載を確認）」 |

§5 の提案文案でも条件記号は spec の意味に固定する。イシューの記号を持ち込むと編集対象の表そのものと矛盾する。

## §2 v1 → v2 格上げ条件の対照表

v2 前提: REQ-1 v2 で `burn` 系一式・`cubecl` は依存禁止であり、ROCm を扱うなら自作コア上の自作バックエンドとなる。

### (a) 削除か軽度化か

- v1 文言（`burn-rocm`／`cubecl-hip` の experimental 解消）は前提が消えたため**削除する**。
- v1 の意図（プラットフォームの成熟度・依存リスク）は**軽度化して残す**。(a)' 案は次の 2 点:
  1. 対象 GPU 世代（gfx）が AMD 公式 ROCm Compatibility Matrix に掲載されていること。
  2. HIP FFI の依存方式が承認されていること（選択肢は `docs/rocm-cooperative-hiprtc-unsafe-readiness-decision.md` §6.3 の「新依存区分でバインディング採用」と「手書き `unsafe extern`＋dlopen 手段の承認」。#2614 で比較して **B 案を推奨**に定めた。§3c）。
- 本節は提案でありユーザー承認前である。

### (b) 維持・達成見込み（要再確認）

クラウド AMD GPU で数千円〜1 万円程度という試算（PoC-10 §2）に基づく達成見込みである。単価は変動し、§4 の機材・実施拠点も仮定のため、実機スパイクの実施時点で機材の利用可能性と費用を確認できるまで「達成済み」とは扱わない（本イシューでは再調査しない）。

### (c) 意味維持・判定式の明確化

意味は維持する。判定式は REQ-2 の統一複合判定「相対誤差 1e-3 未満 または 絶対誤差 1e-5 未満」へ揃える旨を追記する（v1 は相対 1e-3 以内のみ）。tolerance 自体は変更しない。充足経路は §3。

### (d) 軸の再定義

v1 の「CUDA 実測比 70%」は、v2 の実機構成（GB10 と AMD 機材）ではハードウェアをまたぐ比較になり不適切である。PoC-10 §3 の 3 行目自身が「GPU クラスを揃えて再設定可」としている。再定義の**軸**として、同一 AMD 機材上の参照実装比（PyTorch ROCm 比、または rocBLAS／hipBLASLt 比。REQ-8 の同一機材上の参照実装比と同じ構造）を提案する。**新しい数値閾値は本 doc で確定しない**（ユーザー承認事項）。

### (e) 一般化

「rocWMMA 経由」を「行列演算ユニット命令（CDNA は MFMA、RDNA3 以降は WMMA）の発行証跡（rocprof 等）と、有効時・無効時の中央値比較」へ置き換える。rocWMMA ヘッダを HIPRTC で include する経路か `__builtin_amdgcn_*` 直接使用かは実装時に決める。builtin 名は `.claude/skills/amd-rocm/references/hip/hardware-features.md` の範囲では確認できないため**要出典確認**。

### (f) 置換

「`burn-rocm` の未対応オペ」を「自作 ROCm バックエンドの `BackendOps` 被覆が代表ワークロード（小型 Transformer 相当）の実装範囲を満たすこと」へ置き換える。

## §3 条件 (c) 充足計画（言語機能の論点を含む前提条件）

### 完了済み（readiness）

- #2125: `tensor-core::DeviceInfo::warp_width`（`with_warp_width`）。
- #2126: `crates/backend-cuda/src/warp_geometry.rs` によるレンダリング時注入（`WARP_SIZE`／`WARP_HALF`／`WARP_FULL_MASK`／`WARP_SHFL_XOR`／`WARP_SYNC`）。rmsnorm／softmax／mse へ適用。width≠32 は `InvalidKernelConfig` で fail-closed 拒否する。
- #2127: cooperative 起動の制約 C1〜C9、HIPRTC/NVRTC 互換性表、unsafe 分類 U1〜U7 と SAFETY 根拠案。

### 未了（(c) の実測に足りないもの）

1. HIP FFI の依存方式の承認（deps-policy の許容 10 区分の外）。
2. `backend-rocm`（仮称）クレート新設の承認（想定クレート数・公開区分が変わる）。
3. wave64 用マクロ本体（HIP の `__shfl_xor` はマスク引数なし。`__syncwarp` 相当は**要出典確認**）。
4. `docs/backend-abstraction-amd-readiness-decision.md` §6c の後続候補（`kernels_bce/huber/nll/reduce/norm_backward.rs` の 8 warp 固定、`kernels_mma*.rs`・`kernels_wmma_opt.rs`・`kernels.rs` のレーン導出リテラル）。
5. HIP-Clang の FP contraction 既定値の実測と、CPU 参照 `f32::mul_add` との FMA 契約統一の判定（tolerance は緩めない。統一できない場合の扱いはユーザー承認事項）。
6. ROCm toolkit 非搭載環境でのビルド成立（`build-no-cuda-toolkit` と同型の CI ジョブ）。
7. 実機依存テストの `#[ignore]` 分離と、実機ジョブを追加する場合の runner-policy 例外整理（`.claude/rules/ci.md`「実機依存」節・`scripts/check-workflow-runner-policy.py` の allowlist 更新）。

### 言語機能の論点

- **`#[derive]`／コード生成と手書きの比較**: HIP バインディングを bindgen（build-dependency）で生成するか、手書き `unsafe extern "C"` にするか。PoC-v2-6 で `prost-build`（`protoc` ビルド時依存）を退け手書き derive を選んだ先例（deps-policy の相互運用区分 `prost` の条件）と同じ論理で、手書き宣言を優先候補とする（決定は承認事項）。
- **HIPRTC と NVRTC の仕様差**: `docs/rocm-cooperative-hiprtc-unsafe-readiness-decision.md` §4 の表を参照し、ここでは複製しない。要点は、成果物が PTX ではなくコードオブジェクトになること、arch 指定が `--gpu-architecture=gfxNNN` であること、定義マクロ・pragma・最適化レベル・include 解決は要出典確認であること、FMA 契約が最重要であること。
- **warpSize が実行時値であること**: RDNA は 32、CDNA は 64。#2126 のレンダリング時 `#define` 注入で吸収するが、width≠32 は現状 fail-closed。
- **cooperative 起動**: 初期スコープでは不要。現行 CUDA にも使用箇所がない（#2127 §2）。

## §3b v0.10.0 時点の状況（#2614 で追加）

確認コマンドと結果（基準コミット `4be494be`）:

| 項目 | 確認コマンド | 結果 |
|---|---|---|
| 抽象層（完了済み） | `grep -n "warp_width" crates/tensor-core/src/device.rs` | `DeviceInfo::warp_width`（#2125）あり。#2125・#2126・#2127 のみ完了で、§3 の未了 1〜7 は変わっていない |
| WARP_SIZE 注入範囲 | `grep -rln WARP_SIZE crates/backend-cuda/src` | `warp_geometry.rs`・`device.rs`・`lib.rs`・rmsnorm／softmax／mse の各カーネルに限られる。§3 未了 4（8 warp 固定のカーネル群）は未着手 |
| `LaunchKind` | `grep -rn LaunchKind crates` | 0 件（未実装） |
| `BackendOps` 規模 | `awk '/pub trait BackendOps/{f=1} f&&/^    fn /{n++} END{print n}' crates/tensor-core/src/backend_ops.rs` | 195（粗い数え上げ。(f)' の被覆評価時に再計測する） |
| ROCm 関連の動き | `gh issue list --search rocm --state all`／`gh pr list --search rocm --state all` | 検索結果（2026-10-03 時点）: #2125〜#2129 の readiness・提案群は全て close／マージ済み。2026-10-01 以降にスパイク・承認・spec 起票はない。open は本 issue #2614 と Phase 親 #2606 のみ |
| 実機 | — | 実機拠点は DGX Spark GB10 と M4 Max のみで AMD GPU はない。クラウドスパイクは未実施。(b) の価格は本 doc で再調査せず、要再確認のまま残す |
| Windows | v0.10.0（#2489〜#2496） | CUDA NVRTC の `CUDA_PATH` 探索と onnx-interop の Windows 経路が入った。ROCm のスパイクは引き続き Linux のみを対象とする。Windows の HIP SDK の扱いは**要出典確認**（断定しない） |

## §3c HIP FFI 依存方式の比較と推奨（#2614 で追加。未承認）

現行の許容依存 10 区分に HIP FFI の担い手はない（`.claude/rules/deps-policy.md`。第 10 区分 `libc` は onnx-interop の external data 用途限定）。`Cargo.lock` の `libloading` は `0.9.0`（`grep -n -A2 'name = "libloading"' Cargo.lock`）で、cudarc の推移的依存にすぎず直接依存ではない（`docs/license-matrix.md` §6 で ISC・適合と判断済み）。

| 比較軸 | A: 新区分でバインディングクレート | **B: `libloading` を新区分の直接依存＋実行時解決の関数ポインタ型宣言（推奨）** | C: 手書き宣言＋`libc::dlopen`／`dlsym`（第 10 区分の用途拡張） |
|---|---|---|---|
| 1. toolkit 非搭載でビルド成立 | クレート次第（ビルド時リンクの有無は**要出典確認**） | 成立（ビルド時にリンクしない。cudarc dynamic-loading と `build-no-cuda-toolkit` が先例） | 成立（同左） |
| 2. `Cargo.lock` 差分 | 新規 crate が入る | 既存エントリのみ（直接依存化するだけ） | 既存エントリのみ |
| 3. deps-policy への影響 | 新区分を新設 | 新区分（第 11 区分案）を新設 | 第 10 区分の用途条件を拡張 |
| 4. U1（ハンドル寿命内でのみ呼ぶ）の担保 | クレート任せ | `Symbol<'lib, T>` は借用中のみ型が担保するが、関数ポインタを取り出して保持すると外れる。そのため、ライブラリハンドルと解決済み関数ポインタを 1 つの所有構造に同居させ、全呼び出しをその構造のメソッド経由に限る不変条件で担保（型だけでは担保しない） | `*mut c_void` からの `transmute` で手動。`dlerror` のスレッド安全性の問題もある |
| 5. Windows | クレート次第 | LoadLibrary を含み拡張可能 | unix に限られる |
| 6. ライセンス | 要調査 | ISC・判断済み | MIT／Apache（libc。判断済み） |
| 7. 禁止リストとの衝突 | `cubecl-hip-sys` 等は `FORBIDDEN_CRATES_ALT` の `cubecl-[a-z0-9-]+` で不可 | なし | なし |
| 8. 保守性・cudarc との同一機構 | 外部クレートの保守状況は**要出典確認** | cudarc と同じ機構 | unsafe の面積が最大 |

**推奨は B（1 つに確定。未承認）。** 根拠: ビルド時リンクがない／新規 crate がゼロ／U1 をハンドルとポインタの一体所有で担保できる／cudarc と同じ機構／Windows にも拡張できる。

- **実装形**: `#[link]` 付きの `extern "C"` ブロックはビルド時に `libamdhip64` へリンクし、toolkit 非搭載でのビルドを壊す。B でいう「手書き宣言」は、実行時に解決する `unsafe extern "C" fn` の関数ポインタ型を指す。
- **ピン版**: 実装時点の `Cargo.lock` にある版を `=x.y.z` で固定する（現時点は `0.9.0`）。cudarc が libloading を更新すると 2 つの版が並ぶリスクがある（実装 issue で `cargo tree -d` により確認）。
- **不採用理由**: A は外部クレートのバインディングの有無・ビルド時リンク・保守状況が要出典確認のままで、新規依存を増やす。C は unsafe の面積が増え、unix に限られる。
- 依存の実追加・deps-policy の改定・license-matrix の更新は、**承認後の別 issue** で行う（本 doc は依存を変えない。区分新設案は本 doc 内の提案文にとどめる）。

## §3d #2129 の Non-Goal 推奨・#2499 の目標との関係（#2614 で追加）

- #2129（`docs/chip-optimization-roadmap.md` §0.3）は「ROCm は Non-Goal 確定（v1.0 のスコープ外）」を推奨している（未承認）。一方ルート #2499 は PyTorch／TF 水準への到達を目標にしており、スコアボードの「バックエンド・ハード」行が「部分的」なのは AMD がないことも理由である（`docs/perf/framework-compare-feature-matrix-0.9.0.md:54`）。
- 分岐（ユーザーの選択）:
  - **(i)** 段 1 を承認して Could への道を開く。後続の承認事項: クラウドスパイクの予算・runner-policy 例外・`backend-rocm` クレート新設（§6）。
  - **(ii)** #2129 §0.3 に従い Non-Goal を確定する。spec 側に条件なし Won't への追加提案が必要になり、スコアボードの AMD 項目は「対象外」が確定する。
- 本 doc は本 issue の受け入れ条件に従い (i) の提案を推奨として示すが、#2129 の推奨を覆す決定ではない。選ぶのはユーザーである。

## §4 実機テストの実施拠点・機材の仮定

- **主案**: クラウドの MI300X（gfx942、CDNA3、**wave64**）で 1〜2 セッションのスパイク検証（PoC-10 の費用試算は数千円〜1 万円程度）。wave64 は #2126 の width≠32 fail-closed 経路の検証対象として唯一の現実的な機材である。
- **副案**: RDNA3（gfx1100 系、Radeon RX 7900 系等、wave32）の購入またはクラウド。
- **除外**: RDNA2（RX 6900 XT／gfx1030）。PoC-10 §2 の 1 行目に従う。
- **OS・ROCm 版**: Linux（Ubuntu 24.04 LTS 系）、実機確保時点の最新安定 ROCm。macOS／Windows は非対応。
- **拠点**: 現行の実機拠点（DGX Spark GB10・M4 Max）に AMD GPU はない。実測は一時的なクラウドインスタンスで行い、CI には常設しない。ログは `docs/perf/logs/*/README.md` の申し送り形式に従う（内部ホスト名・認証情報・インスタンス識別子は書かない）。

## §5 spec (b) 形式提案文案（起票用 draft。未起票・2 段構成）

タイトル案: `docs(requirements): 除外事項「ROCm バックエンドの正式対応」の格上げ条件表を v2（完全自作コア）前提へ再定義する（実装リポ Fandhe-AI/fandhe-ai#2128・#2614 提案）`

起票の単位: **起票文案（コードブロック）は段 1（Won't→Could）のみ**で、(d)' の閾値と独立に起票できる（ユーザー承認後）。**段 2（Could→Should）はコードブロック外の将来提案として軸のみを示し、数値閾値は提案しない**。(d)' の数値は Could 到達後の別提案で、ユーザー承認を得て確定する。

````markdown
## 背景

除外事項「ROCm バックエンドの正式対応」の格上げ条件表（Won't→Could の (a)〜(c)、Could→Should の (d)〜(f)）は PoC-10（2026-07-29）が v1（Burn/CubeCL 前提）で定義した。REQ-1 v2 で `burn` 系一式と `cubecl` は依存禁止となり、(a)・(f) の前提が失われている。(d) の「CUDA 実測比 70%」も、v2 の実機構成ではハードウェアをまたぐ比較となり意味が崩れる。実装リポ側では #2125〜#2127 で readiness が完了している。

## 提案: 格上げ条件表の v2 再定義（該当箇所の bullet と表をその場で置き換える。新しい Won't 項目は立てない。置換対象は Won't→Could の行のみ）

### Won't → Could（本提案で確定を求める範囲。置換文案は本行のみ）

| 現状 | 格上げ先 | 格上げ条件（すべて満たすこと） |
|------|---------|------------------------------|
| Won't | Could | (a)' 対象 GPU 世代が AMD 公式 ROCm Compatibility Matrix に掲載され、HIP FFI の依存方式が承認されていること。(b) 対象 GPU 世代（MI200/MI300 系または RDNA3）を実機かクラウドで 1 万円程度以下の費用で確保できること（達成見込み・要再確認）。(c) 実機でのビルド・動作確認と、CPU/Metal/CUDA との出力一致（REQ-2 統一複合判定: 相対誤差 1e-3 未満 または 絶対誤差 1e-5 未満）の実測。 |

## 承認時に固定する契約

- REQ-2 の統一複合判定・tolerance・baseline と FMA 契約は変更しない。
- REQ-8 の既存下限値は変更しない。
- カーネル側の手動境界チェックを省略しない。
- 依存の追加・新クレートの新設は個別にユーザー承認を得る。

## 実装リポ側との取り決め

spec で承認されるまで、実装リポは `backend-rocm` 本体と HIP FFI を起票・実装しない（readiness 範囲の変更は可）。

## 不変事項

除外事項の Won't 判断自体、Phase 4 判定追補のスコープ件数（Won't 11）、REQ-2 受け入れ基準の「ROCm は本要件の対象外」は変更しない。
````

### 将来提案（段 2: Could → Should。§5 の起票文案には含めない）

段 2 は数値閾値が未確定のため、上記の置換用文案（コードブロック）には含めない。Could 到達後に別提案として、(d)' の閾値を含めてユーザー承認を得てから起票する。以下は議論用の軸のみである。

v1 の `burn-rocm`／「CUDA 実測比 70%」／rocWMMA の文言は前提を失っているため、別提案で置き換える。

| 現状 | 格上げ先 | 格上げ条件の軸（数値閾値は未確定） |
|------|---------|---------------|
| Could | Should | Could の条件に加え、(d)' 同一 AMD 機材上の参照実装（PyTorch ROCm または rocBLAS／hipBLASLt）比の相対性能。(e)' 行列演算ユニット命令（MFMA／WMMA）の発行証跡と、有効時が無効時を優位に上回る実測。(f)' 自作 ROCm バックエンドの演算被覆が代表ワークロード（小型 Transformer 相当）の実装範囲を満たすこと。 |

## §6 ユーザー承認事項（未実施。いずれも未承認）

1. 段 1（§5）を spec リポジトリへ起票するか（(d)' の閾値とは独立に判断できる）。
2. HIP FFI の依存方式: 推奨 B（`libloading` を新区分〈第 11 区分案〉の直接依存にする。§3c）を採るか、代案（A／C）か。
3. `backend-rocm` クレートの新設。
4. ROCm 実機を得るためのクラウドスパイクの実施と予算。
5. 実機 CI ジョブを追加する場合の runner-policy 例外。
6. 段 2 の (d)' の比較軸と数値閾値（Could 到達後の別提案で確定する。段 1 の起票・採択の前提ではない）。
7. (i)／(ii) の選択（§3d）。(ii) を選ぶ場合は段 1 の起票は不要になる。

## §7 スコープ外・申し送り

- ROCm 実装への着手、spec リポジトリへの投稿、spec の既存 issue との調整。
- HIP-Clang の FP contraction 実測、#2126 の後続カーネル群の注入化。
- #2129 への結論引き渡し（§0 の表を正とする）。

## §8 結論とタイムライン目安（発火条件で示す）

- **現状**: Won't（条件付き）で据え置く。v0.x の間に ROCm へ着手しない。
- **Could 再評価の発火**: (a)' の承認（段 1 の採択を含む）、クラウドスパイクでの (c) 実測、および実施時の (b) 費用・機材の確認（対象 GPU を約 1 万円以下で確保できること。§0 の「達成見込み・要再確認」を実測価格で確定する）の**三者が全て**成立した時点で、Phase 4 の要件見直しにかける。(b) は (c) 実測のためのクラウド利用時に単価と機材の利用可能性を記録して確認する。
- **Non-Goal 確定の推奨条件**: #2129 の総括、または次のメジャーマイルストーン（v1.0 候補）の判定までに (a)' の承認もスパイクも実施されない場合、#2129 で「ROCm は Non-Goal 確定（v1.0 のスコープ外）」を推奨する。その場合は spec 側で Won't の条件記述を条件なしの Won't へ改める追加提案が必要。
- 「提案の採択（v2 再定義文案の spec 取り込み）」のうち **段 1 は (d)' の閾値と独立に**、ユーザー承認後に起票できる（§6 項目 1）。**段 2 は軸のみの記載**とし、(d)' の数値閾値は Could 到達後の別提案でユーザー承認を得て確定する（未確定の数値を spec に書かない）。「ROCm の Could 格上げ」は発火条件待ちであり、提案の採択とは別の判断である。

## §9 セキュリティ観点（OWASP）

- **A03**: 将来 HIPRTC ソースを組み立てる際は静的テンプレートと検証済みの数値・enum のみを使う。`--gpu-architecture` や include パスはオプション文字列としてのみ渡し、シェル展開・ソース連結に使わない（#2126 の数値のみ `#define` 注入を踏襲）。
- **A06**: HIP FFI 依存を追加する場合は `=x.y.z` 完全固定・`docs/license-matrix.md` 更新・ユーザー承認が必須。本 doc は依存を変えない。
- **A08**: コードオブジェクトのディスクキャッシュは NVRTC キャッシュの対策（`O_NOFOLLOW`・fd 相対解決・キーに toolchain 版と gfx target を含める）を踏襲する。cooperative 起動の通常起動への黙示代替は禁止（fail-closed）。
- **B 採用時の注意（§3c）**: dlopen するライブラリ名とシンボル名は固定の `&'static str` に限り、環境変数など外部入力からパスを組み立てない。U1（ハンドル寿命内でのみ呼ぶ）は `Symbol<'lib, T>` のライフタイムだけでは担保できない（関数ポインタをコピーして保持すれば制約を外せる）。解決済み関数ポインタはライブラリハンドルと同じ所有構造に保持し、呼び出し時までハンドルを生存させる不変条件（ポインタを構造の外へ出さず、呼び出しはメソッド経由に限る）を設ける。
- **秘密情報**: クラウドスパイクの手順・ログに認証情報・インスタンス識別子・内部ホスト名を書かない。

## §10 出典

- `docs/spec/04-requirements.md:365-369`、`docs/spec/03-poc/poc-10-rocm-promotion/README.md` §2〜§4
- `docs/backend-abstraction-amd-readiness-decision.md` §6・§6b・§6c
- `docs/rocm-cooperative-hiprtc-unsafe-readiness-decision.md` §4・§6.3・§8
- `docs/backend-matrix.md` §3.4
- `crates/backend-cuda/src/warp_geometry.rs`
- #2614 の確認に使ったファイル・コマンド: §3b・§3c の表、`scripts/check-forbidden-deps.sh`（`FORBIDDEN_CRATES_ALT`）、`Cargo.lock`、`docs/license-matrix.md` §6、`docs/chip-optimization-roadmap.md` §0.3・§7、`docs/perf/framework-compare-feature-matrix-0.9.0.md:54`
- 公式 ROCm ドキュメントは本 doc では未参照（「要出典確認」項目は実装イシューで確認する）。
