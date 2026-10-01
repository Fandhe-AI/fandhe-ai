# ROCm 格上げ条件表の v2 再定義（(b) 形式 spec 提案文案）と条件 (c) 充足計画（#2128）

基準コミット: `e083e609`（origin/main。#2125〜#2127 マージ後）。`docs/spec` submodule ポインタは `2e998dd77117814f4af8ed160394ad1d6a8f888a`。spec への `file_path:line` は submodule 読解時点（除外事項の ROCm bullet は `docs/spec/04-requirements.md:365-369`）のもので、submodule 更新で行番号はずれうるため、参照時は再確認すること。

## §0 結論（最初に読む。#2129 が引用する安定した結論表）

- **コード変更なし**（`crates/**`・`Cargo.toml`／`Cargo.lock`・`docs/spec/`〈正本 submodule〉・tolerance／baseline・ガードレール閾値・依存は一切変更しない）。
- §5 の (b) 形式 spec 提案文案は**未起票**（承認事項は §6）。
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
  2. HIP FFI の依存方式が承認されていること（選択肢は `docs/rocm-cooperative-hiprtc-unsafe-readiness-decision.md` §6.3 の「新依存区分でバインディング採用」と「手書き `unsafe extern`＋dlopen 手段の承認」）。
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

## §4 実機テストの実施拠点・機材の仮定

- **主案**: クラウドの MI300X（gfx942、CDNA3、**wave64**）で 1〜2 セッションのスパイク検証（PoC-10 の費用試算は数千円〜1 万円程度）。wave64 は #2126 の width≠32 fail-closed 経路の検証対象として唯一の現実的な機材である。
- **副案**: RDNA3（gfx1100 系、Radeon RX 7900 系等、wave32）の購入またはクラウド。
- **除外**: RDNA2（RX 6900 XT／gfx1030）。PoC-10 §2 の 1 行目に従う。
- **OS・ROCm 版**: Linux（Ubuntu 24.04 LTS 系）、実機確保時点の最新安定 ROCm。macOS／Windows は非対応。
- **拠点**: 現行の実機拠点（DGX Spark GB10・M4 Max）に AMD GPU はない。実測は一時的なクラウドインスタンスで行い、CI には常設しない。ログは `docs/perf/logs/*/README.md` の申し送り形式に従う（内部ホスト名・認証情報・インスタンス識別子は書かない）。

## §5 spec (b) 形式提案文案（起票用 draft。未起票）

タイトル案: `docs(requirements): 除外事項「ROCm バックエンドの正式対応」の格上げ条件表を v2（完全自作コア）前提へ再定義する（実装リポ Fandhe-AI/fandhe-ai#2128 提案）`

````markdown
## 背景

除外事項「ROCm バックエンドの正式対応」の格上げ条件表（Won't→Could の (a)〜(c)、Could→Should の (d)〜(f)）は PoC-10（2026-07-29）が v1（Burn/CubeCL 前提）で定義した。REQ-1 v2 で `burn` 系一式と `cubecl` は依存禁止となり、(a)・(f) の前提が失われている。(d) の「CUDA 実測比 70%」も、v2 の実機構成ではハードウェアをまたぐ比較となり意味が崩れる。実装リポ側では #2125〜#2127 で readiness が完了している。

## 提案: 格上げ条件表の v2 再定義（該当箇所の bullet と表をその場で置き換える。新しい Won't 項目は立てない）

| 現状 | 格上げ先 | 格上げ条件（すべて満たすこと） |
|------|---------|------------------------------|
| Won't | Could | (a)' 対象 GPU 世代が AMD 公式 ROCm Compatibility Matrix に掲載され、HIP FFI の依存方式が承認されていること。(b) 対象 GPU 世代（MI200/MI300 系または RDNA3）を実機かクラウドで 1 万円程度以下の費用で確保できること（達成見込み・要再確認）。(c) 実機でのビルド・動作確認と、CPU/Metal/CUDA との出力一致（REQ-2 統一複合判定: 相対誤差 1e-3 未満 または 絶対誤差 1e-5 未満）の実測。 |
| Could | Should | Could の条件に加え、(d)' 同一 AMD 機材上の参照実装（PyTorch ROCm または rocBLAS／hipBLASLt）比の相対性能（軸のみ本提案で定め、数値閾値は別途ユーザー承認で確定）。(e)' 行列演算ユニット命令（MFMA／WMMA）の発行証跡と、有効時が無効時を優位に上回る実測。(f)' 自作 ROCm バックエンドの演算被覆が代表ワークロード（小型 Transformer 相当）の実装範囲を満たすこと。 |

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

## §6 ユーザー承認事項（未実施）

1. spec リポジトリへ §5 を起票するか。
2. (a)' の方式（HIP FFI 依存方式の選択と許容依存区分の拡張有無）。
3. (d)' の比較軸と閾値の数値。
4. `backend-rocm` クレートの新設。
5. ROCm 実機を得るためのクラウドスパイクの実施と予算。
6. 実機 CI ジョブを追加する場合の runner-policy 例外。

## §7 スコープ外・申し送り

- ROCm 実装への着手、spec リポジトリへの投稿、spec の既存 issue との調整。
- HIP-Clang の FP contraction 実測、#2126 の後続カーネル群の注入化。
- #2129 への結論引き渡し（§0 の表を正とする）。

## §8 結論とタイムライン目安（発火条件で示す）

- **現状**: Won't（条件付き）で据え置く。v0.x の間に ROCm へ着手しない。
- **Could 再評価の発火**: (a)' の承認、クラウドスパイクでの (c) 実測、および実施時の (b) 費用・機材の確認（対象 GPU を約 1 万円以下で確保できること。§0 の「達成見込み・要再確認」を実測価格で確定する）の**三者が全て**成立した時点で、Phase 4 の要件見直しにかける。(b) は (c) 実測のためのクラウド利用時に単価と機材の利用可能性を記録して確認する。
- **Non-Goal 確定の推奨条件**: #2129 の総括、または次のメジャーマイルストーン（v1.0 候補）の判定までに (a)' の承認もスパイクも実施されない場合、#2129 で「ROCm は Non-Goal 確定（v1.0 のスコープ外）」を推奨する。その場合は spec 側で Won't の条件記述を条件なしの Won't へ改める追加提案が必要。
- 「提案の採択（v2 再定義文案の spec 取り込み）」は今すぐ可能、「ROCm の Could 格上げ」は発火条件待ちであり、両者は別の判断である。

## §9 セキュリティ観点（OWASP）

- **A03**: 将来 HIPRTC ソースを組み立てる際は静的テンプレートと検証済みの数値・enum のみを使う。`--gpu-architecture` や include パスはオプション文字列としてのみ渡し、シェル展開・ソース連結に使わない（#2126 の数値のみ `#define` 注入を踏襲）。
- **A06**: HIP FFI 依存を追加する場合は `=x.y.z` 完全固定・`docs/license-matrix.md` 更新・ユーザー承認が必須。本 doc は依存を変えない。
- **A08**: コードオブジェクトのディスクキャッシュは NVRTC キャッシュの対策（`O_NOFOLLOW`・fd 相対解決・キーに toolchain 版と gfx target を含める）を踏襲する。cooperative 起動の通常起動への黙示代替は禁止（fail-closed）。
- **秘密情報**: クラウドスパイクの手順・ログに認証情報・インスタンス識別子・内部ホスト名を書かない。

## §10 出典

- `docs/spec/04-requirements.md:365-369`、`docs/spec/03-poc/poc-10-rocm-promotion/README.md` §2〜§4
- `docs/backend-abstraction-amd-readiness-decision.md` §6・§6b・§6c
- `docs/rocm-cooperative-hiprtc-unsafe-readiness-decision.md` §4・§6.3・§8
- `docs/backend-matrix.md` §3.4
- `crates/backend-cuda/src/warp_geometry.rs`
- 公式 ROCm ドキュメントは本 doc では未参照（「要出典確認」項目は実装イシューで確認する）。
