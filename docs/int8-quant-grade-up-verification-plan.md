# INT8／FP8 量子化 GEMM の格上げ条件 (b)〜(e) 検証計画（#2075・#2609 で (a)(c)(d) の案を追記・#2610 で §7 文案を最新化）

イシュー #2075「docs(spec): INT8 量子化の格上げ条件検証計画」に対応する。親: #1573（Tier 2）・ルート: #1570。

**本ドキュメントはコード変更を伴わない検証計画の記録のみである。** `crates/**`・`Cargo.toml`／`Cargo.lock`・`docs/spec/`（正本 submodule）・数値一致許容誤差（`RELATIVE_TOLERANCE`／`ABSOLUTE_RESCUE_THRESHOLD`／`ParityBaseline`）・`guardrail.toml` はいずれも変更しない。実装着手は正本 spec（`docs/spec/04-requirements.md:356-364`）が定める格上げ条件 a〜e の充足・Phase 4 新 REQ の spec 側承認・実装リポ側の別途ユーザー承認が前提であり、本 PR ではそのいずれも行わない。

基準コミット: `0de82718`（main HEAD。2026-09-22 時点）。`file_path:line` は同コミット近傍のもの。後続の変更で行番号がずれる可能性があるため、参照する際は当該コミット、または近傍のコミットで再確認すること。

**#2609 追記（2026-10-11）**: 格上げ条件 (a)・(c)・(d) の案と根拠を §2.1・§4.1・§5.1 に追記した（基準は `ccc7fd5e`〈main HEAD〉。#2608 のマージ後）。上記 #2075 時点の基準コミット行は履歴として残す。spec への引用行番号（`docs/spec/04-requirements.md:358-366` 等）は spec submodule の作業ツリー内容に基づく。pin コミットとずれている場合があるため、投稿前（#2610）に pin で再確認すること。

**#2610 追記（2026-10-11）**: §7（spec 提案文）を起票用 draft に書き起こし、§8（承認事項）を選択肢形式に再構成した。基準は main `78fbaca3`（#2609 のマージ後）・spec submodule pin `ca83de61feb3f2f55af7cb98bbf2917c28baf16c`。既存節（§2〜§6 ほか）の行番号は各節の基準コミットのものを履歴として残し、書き換えない。**#2610 で新しく書く文は pin の行番号を使う**。旧→新の対応は次のとおり（いずれも `docs/spec/04-requirements.md`）:

| 旧表記（作業ツリー基準） | pin `ca83de61` での位置 | 内容 |
|---|---|---|
| `:356-364`・`:358-366` | `:363-371` | 除外事項「分散学習・量子化の網羅対応」（見出し `:363`・判断 `:364`） |
| `:360-361` | `:365-368`（Won't→Could の行は `:367`、Could→Should の行は `:368`） | 格上げ条件表 |
| `:362` | `:369` | 承認時に固定する契約 |
| `:363`・`:365`（ゲート） | `:370` | 実装リポ側との取り決め（spec 側で REQ 承認されるまで量子化カーネルを起票・実装しない） |
| （新規） | `:371`・`:372` 以降 | REQ-15 候補の再評価・DDP 追記（2026-10-08） |

REQ-2 の追記は `:84`（2026-09-02）・`:98-101`（2026-09-12 一般化）、REQ-9 の Tier 2 は `:232` で、旧表記と同じである。

**#2995 追記（2026-10-11）**: §8 の項 1〜4 の所有者判断を §8.2 に承認記録として追記した。基準は main `1eaaf356`。spec submodule の pin は `ca83de61` のまま変えない。spec main は `4396dfa6`（spec PR #84 のマージ）で、出典の追記は `:372` から始まる。既存の文は履歴として残し、訂正は追記で示す。

## 0. 結論（最初に読む）

- コード変更なし。段階 0（`docs/backend-int8-quantization-decision.md` §3.1）は不変。#1627 は格上げ条件充足・Phase 4 新 REQ 承認まで実装着手不可のまま open（本 PR で close しない）。
- 正本 spec への提案文案（§7）は本 PR では**未起票**。投稿可否はユーザー承認事項（§8）。
- 2026-09-17 のユーザー承認（イシュー #1964・CLOSED）は「選択肢 C（量子化・DDP とも現状維持）」であり、「任意で A（spec 側の条件 (a) 表記更新提案）」は明示承認されていない。本 doc の §7 文案に (a) 表記更新を含める場合も、その投稿可否は別途承認が必要（§8 項 1）。
- **#2609 の結論**（条件 a・c・d の案と根拠。**いずれの条件も充足は宣言しない**。判断は spec 側）:
  - (a): 量子化経路用の新しい parity 基準は不要。spec (a) は「REQ-2 改定〈spec #56〉確定と Tensor Core 経路の基準確定」であり、状況は §2.1 のとおり実質充足と読める。表記更新の文案は §2.1 に置くが**未投稿**。
  - (c): 取り方の案は §4.1（CUDA は既存テストの関数名完全一致・5 プロセス中央値、Metal は実行経路が未実装のため案 i／案 ii の選択が承認事項）。**GB10 と Mac mini CPU 参考行は #2997 で実測記録あり（2026-10-11。充足の判断は spec 側）。Metal は #2998 の担当**。測定コマンドと実測記録は `docs/perf/logs/int8-quant-baseline-2609/README.md`。
  - (d): 閾値案は §5.1。(d-1) は閾値 0〈bit 完全一致〉、(d-2) は f64 真値に対する要素ごとの決定論的上界 `E(m,n)`（係数 1.0・解析値）。要約形の係数は `c = 2/127 + 1/127² ≈ 0.01581`。
  - **実測の範囲**: spec (d) の「実測付き」のうち、実測済みなのは s32 段の bit 一致（#2608）だけである。f32 段（power-of-2 スケール適用・block 間 f64 合算）と (d-2) の係数確認は、quantize–dequantize 参照実装が必要で、これは spec の除外事項ゲート（`docs/spec/04-requirements.md:365`）で止まっているため**未実施**（阻害要因）。
  - REQ-2 の `RELATIVE_TOLERANCE`／`ABSOLUTE_RESCUE_THRESHOLD`・`ParityBaseline`・`guardrail.toml` は不変。承認事項は §8 項 9〜13 に列挙したが、**いずれも承認は得ていない**。
- **#2610 の結論**（**承認は得ていない**。spec リポへの起票・`docs/spec/` の編集もしていない）:
  - §7 を起票用 draft（提案 A: 格上げ条件表の出典更新・(a) の表記更新／提案 B: 新 REQ〈REQ-15 候補・Could〉の申請内容）に書き起こした。**いずれも未起票**。提案 B は a〜e の充足を spec 側が判断した後に申請する内容で、今回は承認を求めない。
  - §8 を選択肢形式（項 1〜4）に再構成した。既存の 13 項目は削除せず、詳細な設計承認事項として残した。
  - **事実の訂正（2026-10-11 時点）**: 上記 §0 冒頭の「#1627 は…open のまま（本 PR で close しない）」は事実と異なる。#1627 は 2026-09-14T11:20:03Z に PR #1836 本文の `Closes #1627` により自動 close されている（state CLOSED）。reopen するか、blocked の追跡を別 issue にするか、CLOSED のまま本ツリーで追跡するかは承認事項（§8 項 4）であり、本 PR では reopen しない。`docs/compat-api-scope.md:727-733` にも古い記述が残る（§9 へ申し送り）。
  - **非破壊契約の基準版**: 元の issue 文では `0.10.0`。最新の出荷版は `0.11.0`（2026-10-09 公開）。文案では「出荷済みの公開 API（最新 0.11.0）」と書く。許容依存は現行 10 区分（本体の直接依存は第 1〜8・第 10 区分）で、INT8 による依存追加はない。
  - **2026-10-07 のユーザー承認（ルート #2499 コメント）は Phase 5 の一部 issue に対するもので、int8 関連〈#2607〜#2610〉を含まない**。int8 関連は何も承認されていない。
- **#2995 の結論**（2026-10-11。判断の逐語は §8.2）:
  - 項 1〜4 は決定済み。実施する後続は #2997（GB10＋Mac mini CPU 参考行）・#2998（Metal 案 ii）・#3000（#1627 の記述訂正）。実施しない後続は #2999（spec#83 で範囲外と確定）。
  - 提案 A は Fandhe-AI/fandhe-ai-spec#82 で起票し、spec PR #84 で反映済み。
  - 上記の「未起票」「#1627 は…open」「承認は得ていない」の文は #2610 時点の履歴として残し、訂正は追記で示す。
- 条件ラベルの対応（issue 本文の記載と正本 spec の格上げ条件表 a〜e〈`docs/spec/04-requirements.md:360-361`〉は表現が異なるため、本 doc は spec 表の a〜g を正規軸として扱う）:

  | issue 記載項目 | 対応する spec 条件 |
  |---|---|
  | (b) REQ-2 複合判定への量子化経路拡張の仕様案 | spec (d)（量子化専用基準の新設。REQ-2 本体は不変）＋ spec (a) の状況注記 |
  | (c) Phase G ベンチのワークロード定義 | spec (c) |
  | (d) 数値許容基準（精度・スループット）の判定式 | spec (d)（精度）＋ spec (f)（スループット。Could→Should 条件） |
  | (e) INT8 MMA カーネルの実装検討項目 | spec (b)（MMA 発行プローブ）＋ spec (e)（依存追加なし設計） |
  | (a) cudarc nccl の link 検証 | 対象外（#2074〈DDP〉のスコープ。issue 本文の受け入れ条件にも「(a) cudarc nccl の link 検証は同 Phase の DDP issue で分離」と明記されている） |

## 1. 位置づけ

- 入力: `docs/backend-int8-quantization-decision.md` §2.1・§3.3・§7、`docs/spec-proposal-fp8-int8-quant-gemm.md`（#584・CLOSED・条件付き保留）、正本 spec `docs/spec/04-requirements.md:356-364`、イシュー #1964 の承認記録（CLOSED）、`docs/compat-api-scope.md:272,504,508,510`（#1627 は skip／blocked と明記）。
- 「(b) 形式」（実装リポ doc を出典に spec 側へ短い規定のみ追記する形式）の定義は先例 `docs/candle-parity-tolerance-contract-decision.md` §7・`docs/spec-proposal-req2-candle-parity-tolerance.md` §2 を参照。本 doc の §7 も同型の文案とする。

## 2. 格上げ条件 a〜g の現状（HEAD 基準の再確認。格上げの宣言はしない）

正本 spec の格上げ条件表（`docs/spec/04-requirements.md:360-361`）:

- **(a)** REQ-2 複合判定の改定（spec #56）が確定し、CUDA Tensor Core 経路の parity 基準が定まっていること。spec #56 は 2026-08-29 CLOSED、REQ-2 の 2026-09-02／2026-09-12 追記（Tensor Core 経路・split-K 経路の受け入れ判定方式。`.claude/rules/coding-rust.md` 該当節）で判定方式は正式化済み。**実質的に充足と読めるが、spec の格上げ条件表自体の表記は「未達」のまま**。表記更新提案（issue の (a) 相当）は #1964 で明示承認されていないため、**本 doc では投稿しない**（§7・§8 参照）。
- **(b)** sm_121（GB10）実機で NVRTC の `compute_121` 受理と FP8（`kind::f8f6f4` 系）／INT8（`m16n8k32 s32.s8.s8.s32`）MMA 命令の発行可否がプローブ記録されていること。**INT8 側は実測済み（#2608）**: 受理段は #2122、実行段・bit 一致・TOPS は `docs/perf/logs/int8-mma-probe-2608/README.md` に記録（GB10 の全 3 正式 target で bit 一致。充足の判断は spec 側）。FP8 側は未実施（§3）。
- **(c)** Transformer 複合ワークロードの実機ベースラインが取得済みで、期待効果を測る分母が存在すること。**未達**（§4）。
- **(d)** 量子化経路専用の数値許容基準の閾値案が実測付きで提示されていること。**未達**（§5）。
- **(e)** 依存追加なしで成立する設計が確認されていること。机上では成立する見込みだが実装確認は**未達**（§6）。
- **(f)(g)**（Could→Should 条件。`:361`）: (f) 量子化 GEMM 試作の実測で f16 経路比のスループット改善が確認されること、(g) REQ-2／REQ-7 の既存閾値を一切変更せずに数値一致検証が成立すること。本 doc では (f) の判定式の骨子のみ §5 に含める。

- **#2609 の案への導線**: (a) → §2.1、(c) → §4.1、(d) → §5.1。上記の「未達」表記は spec 側の表記であり、本 doc では維持する。

HEAD 時点の再確認コマンドと結果:

```
grep -rn "s8\b\|int8\|m16n8k32\|f8f6f4" crates/ | grep -v compute_121
# => backend-metal/src/tile.rs の `bytes8`（padding 変数名の偶然一致）4 件のみ。
#    量子化関連ヒット 0 件

grep -rn "mma.sync.aligned" crates/backend-cuda/src
# => kernels_mma_tf32.rs（TF32 tensor core）・kernels_mma_tf32x3.rs（split-single 3x TF32）・
#    kernels_mma.rs（f16 tensor core）の m16n8k8／m16n8k16 系のみ。s8／f8f6f4 variant なし
```

`docs/backend-int8-quantization-decision.md` §2.1（#1627・2026-09-14 基準）の判定と一致し、HEAD（2026-09-22）でも状況変化なし。

**#2610 時点（main `78fbaca3`）の再実行結果**: 1 つ目の grep は、本番経路 `crates/*/src` では `crates/backend-metal/src/tile.rs` の `bytes8`（偶然一致）4 行のみで、量子化コードは 0 件。`crates/` 全体では #2608 の `crates/backend-cuda/tests/int8_mma_probe_real_device.rs`（`#[ignore]` の実機プローブ）と `sm121_isa_probe_*`（受理段プローブ）がヒットする。本番経路への量子化コードの混入はない。

### 2.0 #2610 時点の状況表（a〜g。**充足の宣言ではない**。判断は spec 側）

| 条件 | 記録済みの事実 | 出典 | 残り |
|---|---|---|---|
| (a) | REQ-2 の `:84`（2026-09-02）・`:98-101`（2026-09-12 一般化）で判定方式は正式化済み。spec 表の表記は「未達」のまま。INT8 は bit 一致で判定でき `ParityBaseline` 方式は不要 | 本 doc §2.1・`.claude/rules/coding-rust.md` | spec 側の表記更新（提案 A の切り離し項目） |
| (b) | INT8 は実測済み。compute_121／121a／121f の全 target で S1〜S6 が ok・bit 一致。FP8 の実行プローブは**未実施**（受理段のみ #2122） | `docs/perf/logs/int8-mma-probe-2608/README.md`・`crates/backend-cuda/tests/int8_mma_probe_real_device.rs` | FP8 の実行プローブ |
| (c) | **GB10・Mac mini CPU 参考行は #2997 で実測記録あり（2026-10-11。充足は宣言しない）。Metal は未計測**（案 ii。#2998） | `docs/perf/logs/int8-quant-baseline-2609/README.md` | 実機計測（承認事項 §8 項 2） |
| (d) | 閾値案: (d-1)=0、(d-2) 係数 1.0、要約係数 `c ≈ 0.01581`（§5.1）。実測済みは s32 段の bit 一致のみ。f32 段は quantize–dequantize 参照実装が要り、それ自体が取り決め（`:370`）で止まっている | 本 doc §5.1 | f32 段・係数確認の実測（承認事項 §8 項 3） |
| (e) | 机上確認の表（§6）。INT8 は `i8` プリミティブ、FP8 は `u8` newtype 案。spec 表の「許容依存 8 区分」は現行の 10 区分に合わせる表記直しが要る | 本 doc §6・`.claude/rules/deps-policy.md` | 実装確認・表記直し（提案 A） |
| (f)(g) | 未着手 | `:368` | Could 格上げ後 |

### 2.1 条件 (a) の案（#2609。parity 基準の状況の論証と表記更新案）

**ラベルの区別**: spec (a) が求めるのは「REQ-2 改定〈spec #56〉が確定し、CUDA Tensor Core 経路の parity 基準が定まっていること」である。**量子化経路用の新しい parity 基準を作ることではない**。量子化経路の基準は spec (d)（§5・§5.1）に置く。

根拠（引用は `docs/spec/04-requirements.md` の作業ツリー基準。pin で再確認すること）:

- spec #56 は 2026-08-29 に CLOSED（本 doc §2 の記載を継承）。
- REQ-2 の 2026-09-02 追記（`:84`）: 統一複合判定を恒常的に満たさない Tensor Core 経路について、形状別の受け入れ判定方式を正式化した。
- REQ-2 の 2026-09-12 追記（`:98`〜`:101`）: 実測ベースライン非後退方式の適用対象を「単一の連続 K ループと異なる結合順序で部分和を求めるカーネル全般」へ一般化した。tolerance 定数は変更していない（`:101`）。
- 実装リポ側の決定記録: `docs/cuda-tensor-core-parity-judgment-decision.md`（CUDA 形状二分方式）、`docs/backend-metal-splitk-parity-judgment-decision.md`（Metal 全形状一律 baseline 方式）。
- 実装: `crates/backend-cuda/tests/common/parity_baseline.rs::ParityBaseline`、`crates/backend-metal/tests/common/splitk_parity_baseline.rs`。

量子化経路との関係:

- INT8 の s32 累積は整数演算で、結合順序に依存しない。block 間合算は §5 の数値契約で順序と型〈`f64`、Metal は soft-f64〉を固定する。このため INT8 経路は「結合順序が異なるカーネル」に当たらず、`ParityBaseline` 方式は不要で、バックエンド間は**bit 完全一致**で判定できる（参照実装・CUDA・Metal の三者。NaN はクラス一致）。
- FP8（fp32 累積で MMA 内の順序が非固定）には 2026-09-12 追記の非後退方式が適用されうる。適用要否は本 issue の対象外とし、§9 へ申し送る。

表記更新の文案（**未投稿**。#1964 で明示承認されていないため、投稿可否は §8 項 1・9。確定は #2610）:

> (a) の「未達」を「REQ-2 の 2026-09-02／2026-09-12 追記により Tensor Core 経路・split-K 経路の判定方式は確定済み。残りは spec 側の確認」へ更新する。

## 3. 条件 (b): sm_121 INT8／FP8 MMA 発行可否プローブの設計（記述のみ・本 PR では追加しない）

- **先例**: `crates/backend-cuda/tests/tma_probe_real_device.rs`。`PROBE_ARCHS = ["compute_121", "compute_121a", "compute_121f"]` の優先順位付きリストで arch を順に試す方式（`:92`）。`#[ignore]` で実機専用に分離。NVRTC 不在時は `.expect` で顕在化させ silent skip にしない。カーネルソースは `&'static str` 定数として持ち、外部入力を連結しない（A03 対策。§10）。ホスト名はテストコード・記録双方に書かない。
- **inline PTX `mma.sync` の NVRTC 受理は既に実証済み**: `crates/backend-cuda/src/kernels_mma_tf32.rs:586`（`m16n8k8 tf32`）・`crates/backend-cuda/src/kernels_mma_tf32x3.rs:398`（split-single 3×TF32）・`crates/backend-cuda/src/kernels_mma.rs`（f16 `m16n8k16`）が GB10 実機で実行済み（`docs/perf/cuda-parity-baseline.md` 該当節。#1106 系列）。`docs/cuda-tensor-core-knowledge.md` §2.4「インライン PTX の NVRTC 受理は未検証」（`:31`）はこの実証を反映しておらず陳腐化している。**本 doc では是正せず、§9 で申し送りのみとする**（本 PR のスコープは INT8／FP8 の検証計画であり §2.4 全体の是正は別軸の変更のため）。§2.4 の未検証事項のうち INT8／FP8 に関わるのは `s32.s8.s8.s32` variant と `kind::f8f6f4` variant の 2 点に限られる。
- **想定ファイル名**（将来・別イシュー）: `crates/backend-cuda/tests/int8_mma_probe_real_device.rs`。`docs/backend-int8-quantization-decision.md` §3.3 の起票候補 1 と同一。
- **プローブ設計（2 段）**:
  1. **コンパイルプローブ**: `mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32`（INT8）・`kind::f8f6f4` 系命令（FP8）を `PROBE_ARCHS` の順に NVRTC へ渡し、コンパイル成否を arch ごとに記録する。
  2. **実行プローブ**: 小形状（m16n8k32 のフラグメント 1 タイル分）を実行し、INT8 は s32 累積の出力をホスト側 `i32` 参照実装と突合する。整数累積は結合順序に依存しないため**判定は bit 完全一致（`fail_count == 0` の厳密判定）**とする。FP8 は E4M3 入力・fp32 累積で、f32 参照との誤差を記録するが、記録するだけでは NaN・全ゼロ等の壊れた出力でも成功扱いになり再現不能となるため、次の fail-closed な合否条件を実行プローブ自体の合否として**事前登録**する（誤差の大きさに関する閾値は §5 の判定式に持ち越し、ここでは実行結果の健全性のみを判定する）: (i) 入力は固定シード由来の決定的な E4M3 値（本 doc では `docs/backend-int8-quantization-decision.md` の実測記録と同一の決定的シード方式を踏襲し、実施時に `docs/perf/logs/int8-mma-probe-<issue 番号>/README.md` へシード値を記録する）とし、実行のたびに値が変わらないことを確認する、(ii) 出力テンソルの全要素が有限（`is_finite()`。NaN・±inf を含まない）であることを検査する、(iii) 出力テンソルが全要素ゼロではないこと（非自明な入力に対して全ゼロ出力は量子化パス自体が無効化されている兆候として扱う）を検査する。(i)〜(iii) のいずれかを満たさない場合は実行プローブを失敗として記録し、条件 (b) の FP8 側充足候補に含めない。
- **記録先**: `docs/perf/logs/int8-mma-probe-<issue 番号>/README.md`（実施時に env_info・arch 別結果・失敗名を残す。ホスト名は `<cuda-node>` プレースホルダを使う。`docs/real-hardware-verification-env.md` の方式に従う）。
- **事前登録する判定規則**（実測前に固定。事後に緩和しない）:
  - いずれかの arch で「コンパイル成功 かつ 実行成功 かつ bit 一致（INT8）」が得られれば、条件 (b) の INT8 側は充足候補とする。
  - 全 arch でコンパイル失敗した場合は (b) 不成立とする。
  - **実測記録（#2608）**: INT8 の実行プローブを GB10 で実施済み。結果・TOPS・再現コマンドは `docs/perf/logs/int8-mma-probe-2608/`（`RULE.txt` は実測前に固定）。充足の宣言は行わない。
  - FP8 のみ不成立（INT8 は成立）の場合、INT8 単独での条件 (b) 部分充足を主張できるかは spec 側の判断に委ねる（本 doc では宣言しない）。

## 4. 条件 (c): Transformer 複合ワークロード実機ベースライン（既存資産のギャップ埋め）

既に確定済みのもの（再定義しない。単一真実源は `crates/bench-harness/src/transformer_workload.rs::baseline_spec()`）:

- ワークロード仕様: `d_model=512, n_heads=8, d_ff=2048, batch=8, seq_len=128, num_layers=1`・f32・GELU（erf）合成・post-norm・決定的シード `SEED = 155_083`。
- 計測プロトコル: warmup 20／iters 20・中央値 + Q1/Q3・`BenchReport::to_json`（`docs/perf/transformer-workload-baseline.md` §2〜§5・#589 CLOSED）。
- 比較対象: `torch==2.13.0 nn.TransformerEncoderLayer`。
- 「データセット」は決定的シードの合成入力であり、実データセット導入は本 doc のスコープ外（必要ならユーザー承認事項として別途起票）。

残ギャップ（本 doc が列挙する実測タスク。起票はユーザー承認後・本 PR では起票しない）:

1. **CUDA**: `transformer_block_forward_bench_cuda_prefusion`／`transformer_block_forward_bench_cuda_fused`／`transformer_block_forward_cuda_fused_parity`（`crates/bench-harness/tests/transformer_workload.rs:661,694,730`。#602 CLOSED だが GB10 実測は未実施）を GB10 上で 5 run・関数名完全一致指定で実行し、`docs/perf/transformer-workload-baseline.md` の CUDA 表へ転記する。per-op ホスト転送が計測値に含まれる既知の制約を注記する。
2. **Metal**: `transformer_workload_metal.rs` 相当の実行経路は未実装。切り出し先の既存 Issue を `gh issue list --search "transformer_workload_metal" --state all` で確認したが**該当 Issue は見つからなかった（未特定）**。本 PR では起票しない。
3. **CPU（M4 Max）**: `docs/perf/transformer-workload-measurement.md`（既存ファイル）の記入待ちテンプレートを埋める（未実施）。

`facade::TransformerEncoderLayer`（#2068・PR #2211 でマージ済み）は将来の facade 経路ベンチ候補として記載のみに留める（採否は決定しない）。

**(c) 充足の定義**: CUDA・Metal の「改善前」行と PyTorch 基準行が実機値で埋まり、量子化の期待効果を測る分母（対 PyTorch 比）が存在すること。REQ-8 は本行に下限を設定しない方針（`docs/spec/04-requirements.md:360` (c) 注記）を維持する。

### 4.1 #2609 の取り方の案（GB10・Mac mini CPU は #2997 で実測記録あり。Metal は未計測）

分母の定義は再定義せず、上記の単一真実源（`baseline_spec()`・`docs/perf/transformer-workload-baseline.md` §2〜§5）を引用する。測定コマンドと記入欄は `docs/perf/logs/int8-quant-baseline-2609/README.md`。

- **CUDA（GB10）**:
  - 3 テスト（`transformer_block_forward_bench_cuda_prefusion`・`transformer_block_forward_bench_cuda_fused`・`transformer_block_forward_cuda_fused_parity`）を**関数名の完全一致**で個別実行する。部分一致フィルタは使わない（`transformer-workload-baseline.md` §7）。
  - 1 テストにつき 1 プロセスを 5 回実行し、各 `BenchReport::to_json` の median を集めて 5 run の中央値を取る。Q1／Q3 も併記する。
  - PyTorch の分母は `scripts/bench-transformer-pytorch.py` を同じ 5 run 方式で取る。
  - per-op ホスト転送が計測値に含まれる既知の制約を注記する。
- **Metal（M4 Max）**: 複合ワークロードの実行経路（`transformer_workload_metal.rs` 相当）が未実装のため、そのままでは測れない。
  - 案 i: 専用の実行経路を別 issue で実装する。`baseline_spec()` と同じ定義を保てるため推奨。
  - 案 ii: facade `TransformerEncoderLayer`（#2068）を Metal で動かして代替分母にする。定義の差分を注記する必要がある。
  - どちらにするかは承認事項（§8 項 11）。切り出し先の既存 Issue は 2026-10-11 の再検索（`gh issue list --state all --search "transformer_workload_metal"`）でも**未特定**。
- **CPU（Mac mini〈Apple M4〉。#2995 の判断で M4 Max から変更。実測は `docs/perf/logs/int8-quant-baseline-2609/` に記録、PyTorch CPU 行のみ未計測）**: 参考行とする。(c) の充足要件〈CUDA・Metal〉には数えない。
- **共通**: `env_info.txt` と、腕の同定用の `.rev-stamp` を記録する。ホスト名は `<cuda-node>` プレースホルダにする。計測中はビルドを並走させない。事前登録規則は実測前に固定し、事後に緩めない。
- **量子化効果の分母は 2 層**に分ける。(1) 複合ワークロードの対 PyTorch 比（spec (c)）。(2) GEMM 単体の f16 `mma_f16` 行（spec (f)。§5 のスループット判定）。#2608 の TOPS（中央値 約 244）は合否ゲートにも分母にも使わない（`RULE.txt` J5 を踏襲）。

(c) 充足の定義は上記のまま。REQ-8 の下限は設定しない。

## 5. 条件 (d): 量子化経路専用の数値許容基準（判定式の骨子・事前登録）

spec「承認時に固定する契約」（`docs/spec/04-requirements.md:362`）を前提として継承する:

- INT8 は s32 累積 → スケール適用 → f32 出力。
- amax は 1e-4 下限クランプ。
- per-token・K=128 ブロックスケール。
- スケールは power-of-2 丸め。
- FP8 は fp32 常時累積。

2 層判定（事前登録。実測前に判定式の形のみを固定し、閾値定数は空欄とする）:

- **(d-1) quantize–dequantize 参照実装との一致**: INT8 GEMM 出力（s32）とホスト `i32` 参照実装は**整数演算のため結合順序に依存せず bit 完全一致**。**block 内**（単一 block の s32 累積 → 単一の合成スケール `Δ_out[token, block]` を 1 回乗じる power-of-2 乗算）は、2 のべき乗倍が f32 で厳密なため**bit 完全一致**。**block 間の合算**（複数 block の f32 部分和を縮約する処理）は浮動小数点加算であり非可換なため、power-of-2 性だけでは bit 一致は保証されない。block 間合算の bit 完全一致は、下記「block 間合算の数値契約」で加算順序・型（`f64`・Metal はソフトウェアエミュレーション）を参照実装・CUDA・Metal で明示的に固定することによって初めて成立する。判定は `fail_count == 0` の厳密判定とする。この bit 完全一致を成立させるため、参照実装とカーネルの双方で次の演算順序・丸め契約を固定する（実測前の事前登録。丸め方式〈power-of-2 切り上げ〉の確定と同時にレビューを経て確定する）:
  - **s32→f32 変換の丸め**: s32 蓄積値から f32 への変換は round-to-nearest-even（IEEE 754 既定丸め）とし、参照実装・カーネルの双方で同一の変換命令クラス（例: CUDA `__int2float_rn` 相当）を使う。異なる丸めモードへの暗黙の依存を作らない。
  - **Δ_a・Δ_b の合成順序**: 出力スケール `Δ_out = Δ_a * Δ_b`（2 のべき乗同士の乗算のため厳密）を**先に 1 回だけ計算してから** s32 出力へ乗じる。「s32 値に Δ_a、続けて Δ_b を個別に 2 回乗じる」経路は使わない（丸め箇所が増え bit 一致が崩れうるため）。
  - **個別スケール適用順序**: per-token（A 側）・per-block（B 側。K=128 ブロック）のスケールは出力要素ごとに単一の合成スケール `Δ_out[token, block]` へ事前に畳み込んでから s32 出力へ 1 回だけ乗じる。要素ごとに複数回の乗算を分割して適用しない。
  - **overflow の扱い（K 上限の事前検証設計。codex-review #2219 指摘反映）**: INT8×INT8 の要素積は amax/127 分割（`amax_a / 127`・`amax_b / 127`。本 doc §5 冒頭・`docs/backend-int8-quantization-decision.md:84`）により量子化値が `[-127, 127]` に収まる契約のため、単一要素積の絶対値上限は `127 * 127 = 16,129` で確定する。この上限と `i32::MAX = 2,147,483,647` から、**同符号の積が連続悪化した最悪ケースでも overflow しない安全な K 上限**を次式で事前登録する（実測前に固定する形式的境界であり、丸め・スケール確定を待たない）:
    - `K_safe = floor(i32::MAX / 16,129) = 133,144`（本 doc の閾値であり、実装が別途厳しい値を採用してもよいが緩めてはならない）。
    - §5 冒頭の契約が定める per-block（B 側）スケール粒度は `K=128` であり、**block-wise dequant（block 内で s32 累積を確定させてから当該 block の合成スケールを 1 回乗じる設計。§5 の「個別スケール適用順序」）を採用する限り、s32 アキュムレータの実効的な累積区間は `K_safe`（133,144）ではなく block 長 128 に一致する**。block 長 128 は `K_safe`（133,144）を 3 桁下回り、block 内最悪値 `128 * 16,129 = 2,064,512` は `i32::MAX`（2,147,483,647）を 3 桁下回るため、per-token・K=128 ブロックスケール契約（量子化値は `[-127, 127]` の対称範囲。`-128` は amax/127 分割の構成上生成されない）が保たれる限り overflow は形式的に発生し得ない。なお `2,064,512 < 2^24` のため、block 内 s32→f32 変換（(d-1) の丸め契約）はこの範囲では有効桁数の損失なく厳密である。
    - この性質は「block 長を 128 に保つ」という設計契約に依存する安全性であり、**overflow を無条件の契約違反として片付けず、実装が block 長を上記契約から逸脱させないことを事前検証で保証する**: (i) 参照実装・カーネルの双方は、dispatch 前に実効 K 分割単位（block 長。将来 K=128 以外の粒度を採る変更を含む）が `K_safe` 以下であることを assert し、超過時は panic ではなく型付きエラー（`crates/tensor-core` の既存エラー型契約に従う。本 doc ではエラー型の具体設計を確定しない）で拒否する（`.claude/rules/coding-rust.md` の「本番経路で `unwrap()` / `expect()` を使わない」に従う）。(ii) block 長を `K=128` から変更する場合、または block 化を行わず単一の flat s32 アキュムレータで全 K を累積する変種を将来検討する場合は、上記 (i) の事前検証に加えて、`K_safe` を超える入力に対する fail-closed な代替設計（chunked 累積: `K_safe` 未満の chunk ごとに部分和を確定させ `f64` 側で合算する〈Metal は下記「block 間合算の数値契約」と同じ binary64 ソフトウェアエミュレーション方式を用い、`double` をバックエンド共通契約として素朴に要求しない〉／アキュムレータ自体を `i64` へ拡張する。いずれも本 doc では採否を決定せず、承認事項〈§8〉へ追加する）を実装前に選定する。(iii) 実測で overflow が観測された場合（(i)(ii) の事前検証を経てなお発生した場合）は本 doc の前提〈block 長・amax クランプ値〉の見直し対象とする。
    - **境界テスト（§3 実行プローブへ追加登録）**: (I) block 内正常系として `K=128`（契約どおりの block 長）で同符号最悪値（全要素が量子化後 `127`（または `-127`）で揃う adversarial 入力）を用い s32 出力が `2,064,512` で overflow しないことを確認する。(II) 境界検証として `K = K_safe`（133,144）・`K = K_safe + 1`（133,145）の同符号最悪値入力で、flat 累積を仮定した参照実装の assert／型付きエラーが (I) の block 化設計とは独立に単体で機能することを確認する（block 化により実運用では到達しない領域だが、(i) の事前検証ロジック自体の単体境界テストとして §3 のプローブ実装〈`crates/backend-cuda/tests/int8_mma_probe_real_device.rs` 相当〉に含める）。
  - **block 間合算の数値契約（codex-review #2219 P1 指摘反映。2026-09-22 修正）**: block 間の合算は、量子化ステップ `Δ_out[token, block]` が block ごとに異なり浮動小数点加算が非可換となるため、bit 完全一致（(d-1)）を block 間合算にも維持するには加算順序・丸め箇所（型）の両方を参照実装・CUDA・Metal の三者で明示的に同一化する必要がある。本 doc は「合算方式（順序・型）を各バックエンドで明示的に固定」する設計を採用し、`.claude/rules/coding-rust.md`「数値契約の統一」節が既に確立した bias 勾配の行方向縮約（`crates/backend-metal/src/soft_f64.rs`・`gemm.metal::gemm_bias_grad_reduce_f32`。イシュー #1566・PR #1659〜#1666）と**同一の契約形**を踏襲する（順序を固定しない限り (d-1) の bit 完全一致契約が block 間合算では成立しないため）:
    - **参照実装（ホスト）**: block ごとのデータフローを次の順で固定する: (1) 当該 block の s32 部分和（block 長 128・`< 2^24` のため有効桁数の損失なく厳密）を `f32` へ変換する（(d-1) の s32→f32 丸め契約と同一）、(2) その `f32` 値へ当該 block の合成スケール `Δ_out[token, block]` を `f32` のまま 1 回だけ乗じる（2 のべき乗倍のため厳密。ここまでが block 内 bit 完全一致の範囲）、(3) block ごとの `f32` 部分和を `f64` へ widen（`f32::from`／`f64::from(f32)` 相当。soft_f64 の `bias_f64_widen` と同一操作）してから `f64` アキュムレータへ block index 昇順（0, 1, ..., ⌈K/128⌉-1）に逐次加算する（`bias_f64_add` と同一操作）、(4) 全 block 加算後に **1 回だけ** `f64` から `f32` へ narrow（downcast。`bias_f64_narrow` と同一操作）する。丸めは各ステップとも Rust 既定（`f64 + f64`・`f64 as f32` は最近接偶数）。s32 から `f64` へ直接変換する経路は使わない（block 内は f32 経由、widen は必ず f32→f64 で行う）。
    - **CUDA**: MSL と異なり `double` をネイティブ演算でサポートするため、ホストと同じ `f64` アキュムレータ・block index 昇順の逐次加算・最終 1 回の `f32` downcast をそのまま GPU カーネル上で実装する（bias 勾配の行方向蓄積と同型。`double` 対応のためソフトウェアエミュレーション不要）。
    - **Metal**: **`.claude/rules/coding-rust.md` が明記するとおり MSL は `double` 型非対応のため、`f64` をバックエンド共通契約として素朴に要求することはできない**。よって Metal 実装は `double` を直接使わず、bias 勾配で確立済みの **64bit 整数による IEEE 754 binary64 逐次加算のソフトウェアエミュレーション**（`bias_f64_widen`／`bias_f64_add`／`bias_f64_narrow` 相当。ホスト側の逐語モデルは `crates/backend-metal/src/soft_f64.rs` と同型の新規モジュールとして block 間合算専用に実装する）を採用し、ホストの `f64` 逐次和（block index 昇順・最後に 1 回 `f32` downcast）と **bit 完全一致**させる（NaN のみ quiet NaN へ正規化しクラス一致で比較。`docs/metal-grad-reduction-parity-judgment-decision.md` と同じ判定契約）。この方式により (d-1) の bit 完全一致契約は block 間合算を含めて三バックエンド全てで成立可能であり、REQ-2 判定・tolerance は不変のまま維持できる。
    - **実装確認は本 PR のスコープ外**（本 doc はコード変更を伴わない検証計画であり §0 の前提どおり `crates/**` は変更しない）。上記の合算契約（順序・型・Metal ソフトウェアエミュレーション方式）は実装前の事前登録であり、§8 承認事項（プローブ実装の起票・実装可否）の対象に含める。
  - **subnormal の扱い**: f32 側の subnormal は IEEE 754 既定のまま扱い、flush-to-zero 等のハードウェア依存モードを無効化した状態（参照実装・カーネル双方の既定）で計測する。
  - 上記の演算順序契約は (d-1) の bit 完全一致判定の前提であり、参照実装（ホスト側）とカーネル実装（CUDA／Metal）の両方に同一契約として実装する。乖離が判明した場合は (d-1) を bit 完全一致判定のまま維持せず、乖離の原因（丸め・結合順序・overflow 処理のいずれか）を特定したうえで契約を修正するか、(d-2) 相当の誤差上限判定へ切り替えるかを実測後にユーザー承認を得て決定する（緩和を自動で適用しない）。
- **(d-2) f32 参照とのスケール前提付き誤差上限**: 量子化ステップ `Δ_a`・`Δ_b` は §5 冒頭の契約どおり**power-of-2 丸め後に実装が実際に使用する値**として定義する（丸め前の `amax_a / 127`・`amax_b / 127` をそのまま Δ として使わない。power-of-2 丸めは `amax / 127` 以上の直近の 2 のべき乗へ切り上げる方式を想定するため、丸め後のステップは丸め前の値の最大 2 倍まで拡大しうる）。**合否ゲートに使う理論誤差上界は線形 K 形式（要素積の量子化誤差の決定論的最悪値。一般に K に比例する）を正とする**。この丸め後 Δ を用いて導出するか、丸め前の `amax / 127` を用いる場合は拡大率（最大 2 倍。丸め方式の確定後に実測で厳密化）を上界の係数へ明示的に含める、いずれかの保守的な形で事前登録する。**√K 形式（RMS 見積り）は、誤差が独立同分布に近いという統計的仮定のもとでの参考統計に限定し、単独では合否ゲートに使わない**（決定論的最悪値の保証がなく fail-open な許容基準になりうるため）。√K 形式を参考統計として記録する場合は、成立条件（誤差の統計的独立性の仮定）と信頼水準（例: 3σ 相当）を実測前に本 doc へ事前登録し、線形 K 形式の判定結果と併記する。係数（拡大率を含む）は GB10 実測で確定する。**閾値定数は本 doc では空欄とし、実測前に判定式の形だけを事前登録する**（#1964 の共通ルール「事後に緩和しない」を踏襲）。
- **スループット判定（spec (f)）**: 分母は `docs/perf/gemm-optimization-baseline.md` の CUDA f16 行（`mma_f16`・launch-only 計測境界）、同一形状・同一同期方式で 5 run 中央値比。判定式は「INT8 median ≤ f16 median × 係数（未確定）」の形のみを事前登録する。

**明記**: 上記はいずれも REQ-2 統一複合判定（相対誤差 1e-3 未満 または 絶対誤差 1e-5 未満）の緩和ではなく、量子化経路専用の**別基準の新設**である。`crates/backend-cpu/src/parity.rs` の `RELATIVE_TOLERANCE`／`ABSOLUTE_RESCUE_THRESHOLD` 定数、`crates/backend-cuda/tests/common/parity_baseline.rs::ParityBaseline` は不変。量子化経路に REQ-2 判定をそのまま適用しない根拠は `docs/spec-proposal-fp8-int8-quant-gemm.md` §3 を参照。

### 5.1 #2609 の閾値案（具体式・解析係数。**実測は未実施**）

既存の事前登録（形のみ固定・閾値定数は空欄）を具体化する案である。**解析値であり、GB10 での確認は保留中**（阻害要因は §0 のとおり）。REQ-2 の定数と `ParityBaseline` は変えず、量子化経路専用の別基準を新設する案とする。

**記号**: 出力要素 (m, n) と K-block b（長さ 128）ごとに量子化ステップ `Δ_a[m,b]`・`Δ_b[b,n]` を置く（「per-token・K=128 ブロックスケール」が A は per-token のみの読み方でも成り立つ一般形）。`Δ = pow2ceil(max(amax_block, 1e-4) / 127)`。`q = round_half_even(x/Δ)`。切り上げにより `|x/Δ| ≤ 127` が保証され clamp は発生しないため、量子化誤差は `|e| ≤ Δ/2`。

**(d-1) quantize–dequantize 参照実装との一致（閾値 = 0）**:
- 判定は bit 完全一致（`fail_count == 0`）。NaN は quiet NaN へ正規化してクラス一致で比較する。
- 実測済み: s32 段のホスト `i32` 参照との bit 一致（`int8.t2.pattern`）と `int8.k128.pos/neg = ±2,064,512`（`128 × 127² = 2,064,512`）。出典は `docs/perf/logs/int8-mma-probe-2608/README.md`。
- 未実測: f32 段（s32→f32 RNE、`Δ_out` を 1 回乗算、block 間 `f64` 逐次和、最後に 1 回 narrow）。§5 の契約どおりで、ゲート解除後の実装 issue に持ち越す。

**(d-2) f64 真値に対する誤差上限（合否ゲート）**:
- 比較の相手は f64 の真値 `C64 = Σ_k a·b`。f32 参照 GEMM の丸めを閾値に混ぜないため。先例は `scripts/bench/framework-compare/parity_torch_truth.py` の厳密真値との突合。
- 要素ごとの決定論的上界（導出: `ab − (a−e_a)(b−e_b) = a·e_b + b·e_a − e_a·e_b`）:

  `E(m,n) = Σ_k ( |a_mk|·Δ_b[b(k),n]/2 + |b_kn|·Δ_a[m,b(k)]/2 + Δ_a·Δ_b/4 )`
- 判定式の案: `|C_q(m,n) − C64(m,n)| ≤ E(m,n) + u·|C64(m,n)| + γ_nb`。`u = 2^-24`（最終の f32 narrow 1 回分）、`γ_nb` は block 間 f64 合算の丸め上界（`nb·2^-53·Σ|partial|` 程度で実質無視できる）。
- `E` は厳密な上界なので**係数は 1.0（解析値）**とする。違反が 1 件でもあれば実装バグとして fail-closed にし、事後に緩めない。
- 要約形（spec 文案向け・データ非依存）: `|diff| ≤ c·K·A·B`。`A = max(amax_a, 1e-4)`、`B = max(amax_b, 1e-4)`。`Δ ≤ 2·amax/127` を代入すると `c = 2/127 + 1/127² ≈ 0.01581`。
- √K 形式（RMS 見積り）は参考統計に限る。誤差を独立同分布とみなす仮定・3σ 相当・線形 K 形式の判定結果との併記を事前登録とし、単独で合否ゲートにしない。

**入力契約**（閾値式が成り立つ範囲。案）:
- 非有限入力〈NaN／±inf〉は型付きエラーで拒否する。
- `Δ` の下限は `pow2ceil(1e-4/127) = 2^-20`（`1e-4/127 ≈ 7.87e-7`、`2^-21 ≈ 4.77e-7 < 7.87e-7 ≤ 2^-20 ≈ 9.54e-7`）。よって `Δ_a·Δ_b ≥ 2^-40` で、`s32·Δ_out` は f32 の正規数範囲に入る。
- `pow2ceil(amax/127)` が overflow する入力や `|s32|·Δ_out` が f32 を overflow する入力は契約外として型付きエラーで拒否する。エラー型の設計は確定しない。
- **最終 narrow の overflow も拒否する**（codex-review #2979 P2 指摘反映）: 各 block の出力が f32 で有限でも、block 間 `f64` 合算後の最終 `f64`→`f32` narrow が `±inf` になる入力は、判定式の前提（`u·|C64|` の f32 有限性）が崩れるため契約外とし、型付きエラーで拒否する。例: K=512・A、B の全要素を `2^60` とすると、各 block の出力は `128·2^120 = 2^127` で有限だが、4 block の `f64` 合算値 `2^129` は f32 narrow で `inf` になる。検査は narrow 後の `f32` 値が有限であること（最近接偶数丸めで `inf` になる値を含む）で行い、検査は参照実装・CUDA・Metal の三者に同一に課す。なお検査を実行前に判定する場合の保守的な事前条件は `nb·max|block_out| ≤ f32::MAX`（`nb = ⌈K/128⌉`。絶対値ベースで正側 `+inf` と負側 `-inf` の双方の overflow を同時に排除する。全 block が負の入力でも `|f64 合算|` が `f32::MAX` を超えれば `-inf` になるため符号付きの `max_block_out` では不十分）で、実装方式の選択は承認事項（§8）に持ち越す。
- **境界確認項目**（§3 の実行プローブへ追加登録）: (I) 上記 K=512・全要素 `2^60` の入力で最終 narrow overflow が型付きエラーで拒否され、`inf` が出力に漏れないこと。(II) 合算値が `f32::MAX` ちょうど（受理）と、最近接偶数丸めで `inf` となる最小の `f64` 値（拒否）の境界。(III) 各 block が有限で合算のみ overflow する入力と、block 内スケール適用で overflow する入力とが、いずれも拒否されること（拒否理由は区別して記録する）。

**実測プロトコルの案**（ゲート解除後に実施。`docs/perf/logs/int8-quant-baseline-2609/README.md` へ申し送り）:
- 入力分布: 一様・正規・外れ値混入〈heavy-tail〉・同符号最悪値・amax < 1e-4〈clamp 経路〉。
- 形状: K ∈ {128, 512, 4096, 16384}。M・N は既存 GEMM ベースライン形状から選ぶ。決定的シード。
- 記録項目: `diff / E` の最大値と分位点、√K 統計、違反件数（0 を期待）。対象は CPU 参照・CUDA・Metal。

**提案する値**: (d-1) = 0、(d-2) = 係数 1.0（要素ごとの上界 `E`）、要約形 `c ≈ 0.01581`。いずれも案であり、確定にはユーザー承認と spec 側の承認が必要。スループット (f) は本 issue の対象外で、上記の骨子を変えない。

## 6. 条件 (e): 依存追加なしで成立する設計の確認項目（yes/no で確認できる粒度）

| 確認項目 | HEAD での事実 |
|---|---|
| `Scalar` trait は sealed（`crates/tensor-core/src/element.rs:111,123`）か | Yes。`private::Sealed` で封印されており、`i8` 実装はクレート内限定で追加可能 |
| `ScalarDType` は `#[non_exhaustive]` か（`I8` 追加が非破壊か） | Yes（`element.rs:139-141`。F32／F64／F16／Bf16 の 4 variant。追加は非破壊） |
| `TypedOps<T>` は INT8 GEMM（s8×s8→s32→f32 の異型演算）を表現できるか | No（`crates/tensor-core/src/typed_ops.rs:33`。同型演算専用のため収まらない） |
| 異型演算用の capability accessor 先例はあるか | Yes。`cast_ops` 同型の accessor（`crates/tensor-core/src/backend_ops.rs:856`。既定 `None`）を参考に `QuantOps` 新設案が考えられる（本 doc では決定しない） |
| `CastDType`（`crates/tensor-core/src/cast.rs:75-86`）に `I8` は含まれるか | No（F32／F64／I32／I64／Bool の 5 variant）。quantize は cast ではなく専用 `Op` として設計する必要がある |
| FP8 型を許容依存内で実装できるか | Yes（`half` crate は FP8 非対応のため `u8` newtype を自作する案。許容依存 9 区分〈`.claude/rules/deps-policy.md`〉への追加は不要） |

(e) の充足宣言は spec 側判断に委ねる。上記は「コードを追加せずに確認できる事実」の棚卸しに留める。

## 7. spec (b) 形式提案文案（起票用 draft・未起票。#2610 で最新化）

> **#2995 追記**: 提案 A は Fandhe-AI/fandhe-ai-spec#82 として起票・spec PR #84 で反映済み（2026-10-11）。下の draft 本文は起票時の文案として残す。

`docs/ddp-grade-up-conditions.md` §4（#2612）と同じ「タイトル案 + ````markdown フェンスの本文案」形式で書く。**ユーザー承認（§8 項 1）を得るまで spec リポへ起票しない。起票はユーザーが行う。** 行番号は spec pin `ca83de61` のもの。起票時は spec 側の現行行番号を再確認すること。

### 7.1 提案 A: 格上げ条件表の出典更新・条件 (a) の表記更新

実装リポの本 doc と `docs/perf/logs/` を出典として、条件表の各条件の記録先を追記する提案である。**充足の判断は spec 側が行う。本提案は格上げを宣言しない。** Won't 判断そのものは変えない。

**タイトル案**:

```
docs(requirements): 除外事項「分散学習・量子化の網羅対応」の量子化格上げ条件表に実装リポの検証記録を出典として追記し、条件 (a) の表記を更新する（実装リポ Fandhe-AI/fandhe-ai#2607〜#2610 提案）
```

**本文案**:

````markdown
## 背景

除外事項「分散学習・量子化の網羅対応」の量子化部分は、spec #57 の判断（`04-requirements.md:364`）で
Won't・条件付きとされ、格上げ条件表（Won't→Could の a〜e、Could→Should の f・g。`:365-368`）が
定義されている。以後、実装リポ Fandhe-AI/fandhe-ai では次の検証記録が積まれた。

- REQ-2 の 2026-09-02 追記（`:84`）・2026-09-12 一般化（`:98-101`）で、結合順序が単一の連続 K ループと
  異なるカーネルの判定方式が正式化された。
- #2608: GB10（sm_121）で INT8 MMA（`mma.sync m16n8k32 s32.s8.s8.s32`）の実行可否と s32 段の bit 一致を
  実測した（compute_121／121a／121f の全 target）。
- #2609: (c) の取り方の案、(d) の閾値案（案のみ。f32 段の実測は未実施）を記録した。

実装リポは `fandhe-ai` を crates.io へ出荷済み（最新 0.11.0）で、公開 API の非破壊が契約になっている。

## 提案: 各条件の出典の追記

条件表の各条件に、実装リポでの記録先を次のとおり追記する。**充足の判断は spec 側が行う。**

| 条件 | 実装リポで記録されている事実 | 出典（Fandhe-AI/fandhe-ai） | 残り |
|---|---|---|---|
| (a) | REQ-2 の判定方式は `:84`・`:98-101` で正式化済み | `docs/int8-quant-grade-up-verification-plan.md` §2.1 | 表記更新（下記） |
| (b) | INT8 は GB10 の全 3 target で bit 一致。FP8 の実行プローブは未実施 | `docs/perf/logs/int8-mma-probe-2608/README.md` | FP8 の実行プローブ |
| (c) | GB10・Mac mini CPU は #2997 で実測記録あり。Metal は未計測 | 同 §4.1・`docs/perf/logs/int8-quant-baseline-2609/README.md` | 実機計測 |
| (d) | 閾値案: (d-1) bit 完全一致、(d-2) 要素ごとの上界（係数 1.0）。s32 段のみ実測済み | 同 §5.1 | f32 段・係数確認の実測 |
| (e) | 机上確認（INT8 は `i8`、FP8 は `u8` newtype 案） | 同 §6 | 実装確認 |

### 切り離して採否を決められる項目: 条件 (a) の表記更新

(a) の「（未達）」を「REQ-2 の 2026-09-02／2026-09-12 追記により、Tensor Core 経路・split-K 経路の
判定方式は確定済み」へ更新する。量子化経路用の新しい parity 基準は (a) では求めない
（量子化専用の基準は (d) が扱う）。

### 切り離して採否を決められる項目: 条件 (e) の表記注記

(e) の「許容依存 8 区分内」を、現行の許容依存 10 区分（本体の直接依存は第 1〜8・第 10 区分）に
合わせる。INT8 は `i8` プリミティブ、FP8 は `u8` ベース newtype の案で、依存の追加はない。

## 変更しないもの

除外事項の Won't 判断そのもの／REQ-9 Tier 2 の従属関係／Phase 4 判定追補のスコープ件数
（Should 8・Could 1・Won't 11）／REQ-2・REQ-7 の tolerance／REQ-8 の下限／
承認時に固定する契約（`:369`）／実装リポ側との取り決め（`:370`）／REQ-15 候補の再評価の記述（`:371`）／
DDP 追記（`:372` 以降）。
````

**#2995 追記**: 提案 A は Fandhe-AI/fandhe-ai-spec#82 として起票し、spec PR #84（merge `4396dfa6`）で反映済み（2026-10-11）。反映先は spec `04-requirements.md:372` 以降（spec main `4396dfa6` 基準。pin `ca83de61` では出典追記はまだ無い）。項 3 の確認事項は Fandhe-AI/fandhe-ai-spec#83 で起票され、選択肢 2（範囲外）で回答済み。

### 7.2 提案 B: 新 REQ（REQ-15 候補・Could）の申請内容（条件付き。今は申請しない）

**a〜e の充足を spec 側が判断した後に申請する内容**である（`:371`）。#2610 では承認を求めない。

**タイトル案**:

```
docs(requirements): INT8 量子化 GEMM を新 REQ（REQ-15・Could）として追加する（実装リポ Fandhe-AI/fandhe-ai#2610 提案・格上げ条件 a〜e の充足判断後）
```

**本文案**:

````markdown
## 範囲

INT8 量子化 GEMM（`s8×s8→s32` 累積 → スケール適用 → f32）。FP8 は (b) の FP8 側が未実測のため、
INT8 先行とするか同時にするかは選択肢とし、本提案では決めない。

## 受け入れ基準の骨子（実装リポ `docs/int8-quant-grade-up-verification-plan.md` §5.1 の案。実測による確定は承認後）

- (d-1) 参照実装・CUDA・Metal の三者間で bit 完全一致（`fail_count == 0`。NaN はクラス一致）。
- (d-2) f64 真値に対する要素ごとの決定論的上界 `E(m,n)`（係数 1.0）と、要約形
  `|diff| ≤ c·K·A·B`（`c ≈ 0.01581`）。√K 形式は参考統計に限り、単独の合否ゲートにしない。
- 入力契約: 非有限値・overflow・最終 narrow の overflow は型付きエラーで拒否する。

## 承認時に固定する契約

`:369` の契約（INT8 は s32 累積後にスケール適用して f32 化 ほか）を引き継ぐ。block 間の f64 合算
（CUDA は `double`、Metal は soft-f64 の逐次加算）は実装リポ側の設計契約として引用する。

## 公開面

facade 経由・出荷済みの公開 API（最新 0.11.0）に対して非破壊（追加 API・opt-in・`#[non_exhaustive]`
variant の追加に限る）。依存の追加なし。REQ-2・REQ-7 の tolerance と REQ-8 の下限は変更しない。

## 件数

Could が 1 件増える。Phase 4 判定追補のスコープ件数の見直しはユーザー承認が必要。
````

**#2610 注記**: 本節は #2610 で最新化した。投稿は未実施である。

## 8. ユーザー承認事項（未実施・#2610 で選択肢形式に再構成）

**#2995 追記**: 項 1〜4 は 2026-10-11 に決定した（§8.2）。以下の「未実施」「承認は得ていない」は #2610 時点の記述として残す。

**いずれも承認は得ていない。** 1 コメントで回答できる形に、まず項 1〜4 を置く。

### 8.0 判断の選択肢

| 項 | 判断事項 | 選択肢 | 文書の推奨 |
|---|---|---|---|
| 1 | spec 提案の扱い | (i) 現状維持（#1964 の選択肢 C を継続）／(ii) 提案 A（§7.1）を起票する。(a) の表記更新を含める／(ii′) 提案 A を起票するが (a) の表記更新は除く | (ii) |
| 2 | (c) 実機計測 | §4.1 の計測 issue（CUDA GB10・M4 Max CPU 参考行）を起票してよいか。Metal は案 i（専用経路）／案 ii（facade 代替）のどちらか | ユーザー判断 |
| 3 | (d-2) テスト専用ホストシミュレーション | `crates/` のテストに触れる（`#[ignore]` か test 限定。本番経路は変えない）。取り決め（`:370`）との関係があるため、**他の項と切り離して判断する** | ユーザー判断 |
| 4 | #1627 の扱い | reopen する／blocked の追跡を別 issue にする／CLOSED のまま本ツリーで追跡する | ユーザー判断 |

**#2995 追記**: 表は #2610 時点のまま残す。#2995 の判断では CPU 参考行は Mac mini（Apple M4）で、表中の「M4 Max CPU」は #2610 時点の表記である。

- **項 1 の推奨の根拠**: (b) の INT8 側の実測（#2608）と (a) の判定方式の正式化で、出典を更新する材料が揃った。提案 A は Won't 判断も件数も変えず、出典を追記するだけである。項 2〜4 は文書として推奨を置かない。
- 提案 B（§7.2）は充足判断の後なので、今回は承認を求めない。
- 項 1〜4 と下の詳細項目の対応: 項 1 = 旧 1・9、項 2 = 旧 3・11、項 3 = 旧 12、項 4 = 本 PR で新設。旧 2・4〜8・10・13 は実装着手時の設計承認事項で、現時点では判断不要。

### 8.1 詳細な設計承認事項（実装着手時に判断。削除せず残す）

1. spec (b) 形式提案（§7）の投稿可否。(a) 表記更新を同時に含めるか。
2. (b) プローブテスト（`crates/backend-cuda/tests/int8_mma_probe_real_device.rs` 相当。`crates/` への変更を伴う）の起票・実装可否。
3. (c) 実機実測タスク（§4 の 1〜3）の起票・優先順位（GB10／M4 Max セッションの割当）。
4. (d) 閾値定数の確定（実測後。tolerance／baseline 相当の新設は人間承認が必要）。
5. `QuantOps`／`ScalarDType::I8` 追加・facade 公開面拡張（`docs/compat-api-scope.md` §5 の手続きに従う）。
6. 外部 FP8 crate を選ぶ場合の依存追加（`.claude/rules/deps-policy.md`。現状は `u8` newtype 自作案が既定）。
7. INT8 s32 累積の K 上限（§5 (d-1)「overflow の扱い」。`K_safe = 133,144`）超過時の代替設計（chunked 累積／`i64` アキュムレータ拡張）の採否・実装可否（block 長 128 契約を維持する限り本 PR の事前検証は形式境界の登録に留まり実装しない）。
8. (d-1) block 間合算契約（§5「block 間合算の数値契約」。参照実装・CUDA の `f64` 逐次加算・Metal の binary64 ソフトウェアエミュレーション新規モジュール〈`crates/backend-metal/src/soft_f64.rs` と同型〉）の実装可否。
9. (a) 表記更新文案（§2.1）の採否。項 1 と合わせて扱う。
10. (d-1) 閾値 0 と (d-2) 係数 1.0（要素ごとの上界 `E`）、要約係数 `c ≈ 0.01581`（§5.1）の採否。
11. (c) Metal 実行経路を案 i（専用経路）と案 ii（facade 代替）のどちらにするか、切り出し issue の起票可否（§4.1）。
12. (d-2) の誤差分布を実測するためのテスト専用ホストシミュレーション（#2608 と同様に `#[ignore]` か test 限定で、本番経路は変えない）の起票・実施可否。spec の除外事項ゲート（`:365`）との関係を含めて判断する。
13. 入力契約（非有限入力・overflow を型付きエラーで拒否。§5.1）の採否。

上記 9〜13 は #2609 で追加した。**いずれも承認は得ていない。**

**#2995 追記**: 旧 1・9 は項 1、旧 3・11 は項 2、旧 12 は項 3 として 2026-10-11 に決定済み（§8.2）。旧 2・4〜8・10・13 は実装着手時の設計承認事項のままで、未決定。

### 8.2 承認記録（#2995。2026-10-11）

Issue #2995 のコメントに所有者判断として記録されている内容を、判断の逐語と日付で書き写す。コメントに書かれた範囲を超えて承認を主張しない。投稿者はいずれも aLiz-Nancy（所有者）。

> **編集注**: コメント 1 中の `04-requirements.md:365` は旧コミットの行番号である。pin `ca83de61`・spec main `4396dfa6` では実装リポ側との取り決めは `:370` にある（本 doc の他の箇所も `:370`）。以下の引用文そのものは直さない。

#### コメント 1: 項 1〜4 の判断（2026-10-11T06:16:39Z）

出典: https://github.com/Fandhe-AI/fandhe-ai/issues/2995#issuecomment-6106164192

> 所有者の指示「推奨で」（2026-10-11）により、`docs/int8-quant-grade-up-verification-plan.md` §8.0 の項 1〜4 を次のとおり決定する。
>
> | 項 | 判断 | 理由 |
> |---|---|---|
> | 1. spec 提案の扱い | **(ii) 提案 A（§7.1）を起票する。条件 (a) の表記更新を含める** | Won't 判断・件数を変えず出典を追記するだけ。(a) の判定方式は REQ-2 で正式化済みで「未達」表記は事実と食い違う |
> | 2. (c) 実機計測 | **実施する。CUDA（GB10）と CPU 参考行（Mac mini〈Apple M4〉）を計測する。Metal は案 ii（facade `TransformerEncoderLayer`〈#2068〉で代替分母を取り、定義の差分を注記）** | CUDA は既存 3 テストの実行のみで取り決め（spec `04-requirements.md:365`）に触れない。案 i（専用実行経路の新規実装）は Won't の機能への先行投資になるため、量子化の格上げ後に回す |
> | 3. (d-2) テスト専用ホストシミュレーション | **今は実施しない。提案 A の中で、テスト限定の quantize–dequantize 参照実装が取り決め（`:365`）の範囲内かを spec 側へ確認する** | テスト限定でも量子化の中核の参照実装であり、取り決めの解釈を実装リポ側だけで決めない |
> | 4. #1627 の扱い | **CLOSED のまま、本ツリー（#2991）で追跡する** | #1627 は量子化カーネル実装の issue で、取り決め上 spec 承認前に open で置かない。`docs/compat-api-scope.md` の記述を「CLOSED（PR #1836 で自動 close）・spec 承認まで凍結・#2991 で追跡」に直す |
>
> 後続 issue の扱い:
> - #2996（提案 A の起票）: **実施**。項 3 の確認事項を本文に含める
> - #2997（(c) GB10 計測）: **実施**（CPU 参考行は Mac mini）
> - #2998（(c) Metal 分母）: **実施（案 ii）**
> - #2999（(d-2) ホストシミュレーション）: **実施しない**（spec 側の回答待ち。回答後に再起票）
> - #3000（#1627 の記述訂正）: **実施**（CLOSED のまま追跡）

#### コメント 2: spec 側の起票先（2026-10-11T06:20:16Z）

出典: https://github.com/Fandhe-AI/fandhe-ai/issues/2995#issuecomment-6106189412

> 項 1（提案 A）は main セッションで spec リポへ起票・反映した（#2996 はこれでクローズ）。
>
> - 提案 A: Fandhe-AI/fandhe-ai-spec#82（反映 PR: Fandhe-AI/fandhe-ai-spec#84）
> - 項 3 の確認事項（テスト限定の quantize–dequantize 参照実装は取り決めの範囲内か）: Fandhe-AI/fandhe-ai-spec#83（回答待ち）

#### コメント 3: 項 3 の確定（2026-10-11T06:41:44Z）

出典: https://github.com/Fandhe-AI/fandhe-ai/issues/2995#issuecomment-6106339215

> 項 3 の確定: Fandhe-AI/fandhe-ai-spec#83 は選択肢 2（範囲外。テスト限定の quantize–dequantize 参照実装も REQ 承認まで行わない）で決定・クローズ（2026-10-11 所有者判断）。(d) は閾値案のまま据え置き、#2999 は not planned のまま。計画 §8 へ記録する際はこの結論を反映すること。

#### 結論と後続 issue の状態（2026-10-11 時点）

- 項 1: 提案 A を起票・反映済み（spec#82／spec PR #84）。項 2: (c) は CUDA（GB10）・CPU 参考行（Mac mini）を計測し、Metal は案 ii。項 3: 範囲外で確定（spec#83）。項 4: #1627 は CLOSED のまま #2991 で追跡。
- (d) は §5.1 の閾値案のまま据え置く。実測済みなのは s32 段の bit 一致のみ（#2608）で、(d-2) の係数確認は REQ 承認後に改めて起票する。
- §8.1 の旧 2・4〜8・10・13 は実装着手時の設計承認事項のままで、**未決定**である。

| 後続 issue | 内容 | 状態（2026-10-11） |
|---|---|---|
| #2996 | 提案 A の起票 | CLOSED/COMPLETED |
| #2997 | (c) GB10 計測（CPU 参考行は Mac mini） | OPEN（実施） |
| #2998 | (c) Metal 分母（案 ii） | OPEN（実施） |
| #2999 | (d-2) ホストシミュレーション | CLOSED/NOT_PLANNED（REQ 承認後に改めて起票） |
| #3000 | #1627 の記述訂正（`docs/compat-api-scope.md`） | OPEN（実施） |

提案 A の起票先: Fandhe-AI/fandhe-ai-spec#82（反映 PR #84、merge `4396dfa6`）。項 3 の確認事項: Fandhe-AI/fandhe-ai-spec#83（選択肢 2 で回答済み）。

## 9. スコープ外・申し送り

- FP8／INT8 カーネル実装・quantize–dequantize 参照実装・dynamic／static 選択・キャリブレーション手法（issue 本文のスコープ外を継承）。
- `docs/cuda-tensor-core-knowledge.md` §2.4「インライン PTX の NVRTC 受理は未検証」の陳腐化是正（§3 参照。本 PR では触れない）。
- Metal 複合ワークロード実行経路の切り出し先確認（§4 の 2。該当 Issue 未特定）。
- 実機実測は `docs/perf/logs/int8-mma-probe-<issue 番号>/` へ申し送り（#2075 時点ではディレクトリを作成しなかった。#2608 で `int8-mma-probe-2608/` を作成。#2609 で (c)・(d-2) 用の `docs/perf/logs/int8-quant-baseline-2609/`〈測定コマンドと記入欄のみ。値は未記入〉を新設）。
- FP8（E4M3／E5M2）の実行プローブと parity 方式（2026-09-12 追記の非後退方式を適用するかを含む）。#2609 の対象外。
- (c) の実機計測、Metal 実行経路の実装、(d-2) の実測はユーザー承認後に別 issue とする。起票は承認を得てから行う。

- #2610 の申し送り: `docs/spec-proposal-fp8-int8-quant-gemm.md`（#584）の改訂・再提出は a〜e の充足判断後に行う。`docs/compat-api-scope.md:727-733` の #1627 の記述（open のまま）は事実と異なる（別 issue 候補。起票はユーザー承認後）。FP8 の実行プローブは未実施。承認依頼コメントの投稿先は #2610。

- #2995 の申し送り: (d-2) の実測は量子化の REQ 承認後に改めて起票する（#2999 は not planned のまま再起票しない）。`docs/compat-api-scope.md:727-733` の訂正は #3000 の担当で本 PR では触らない。(c) の実機計測と測定記録（`docs/perf/logs/int8-quant-baseline-2609/`）の更新は #2997（GB10＋Mac mini CPU 参考行）・#2998（Metal 案 ii）の担当。#1627 は CLOSED のまま #2991 ツリーで追跡する（reopen しない）。

## 10. セキュリティ観点（OWASP Top 10）

- **A03 インジェクション**: 本 PR はコード・設定を変更しない。将来の (b) プローブ実装では、カーネルソースを `&'static str` 定数として保持し外部入力を連結しない契約（`crates/backend-cuda/src/nvrtc.rs`・`crates/backend-cuda/tests/tma_probe_real_device.rs` の既存方式）を踏襲する旨を §3 に明記した。
- **A08 ソフトウェア・データ整合性**: 正本 spec の除外事項ゲート（`docs/spec/04-requirements.md:363`）を迂回しない。実装着手は spec 側 REQ 承認とユーザー承認の両方が前提であることを §0・§8 に明記した。判定式（§5）は実測前に事前登録し、事後の緩和を許さない（#1964 の共通ルール）。tolerance／baseline／`guardrail.toml` はいずれも不変。
- **A06 脆弱・古いコンポーネント**: 依存追加なし（INT8 は `i8` プリミティブ・FP8 は `u8` newtype 案）。外部 FP8 crate 採用は承認事項（§8 項 6）として列挙するのみで、本 PR では選定・追加を行わない。
- **情報漏えい**: 実ホスト名・内部パス・ユーザー名を記載していない（`<cuda-node>` プレースホルダ方式。`docs/real-hardware-verification-env.md` の方式・`.claude/rules/security.md`）。
- **非信頼データの扱い**: イシュー #2075・#1964・#2074 の本文・コメントは要約のみを転記し、命令文は運ばない。本計画の立案・検証過程で命令注入・矛盾指示は検出されなかった。
- **#2609 追記**: コード・設定は変更しない。(A03) 将来の実装案として、量子化入力の非有限値・overflow を型付きエラーで先に拒否する入力検証を §5.1 の入力契約に明記した。測定コマンドにユーザー入力を展開する形は書いていない。(A08) 除外事項ゲート（`:365`）を迂回しない。閾値は実測前に事前登録し、事後に緩めない。(d-2) は違反ゼロ前提の fail-closed で、√K 形式を単独の合否ゲートにしない。(A06) 依存追加なし。情報漏えい: ホスト名は `<cuda-node>`、`docs/real-hardware-verification-env.local.md` は参照も転記もしていない。イシュー #2609・#2607・#2608・#605 の本文は要約のみを使い、命令文は運んでいない。

- **#2610 追記**: コード・設定・`docs/spec/`・submodule pin は変更しない。(A01) ruleset・リポジトリ設定、spec リポへの書き込みはしない。(A08) 承認を代行・捏造しない。int8 関連は未承認（2026-10-07 の承認の範囲外）と明記し、#1627 は reopen しない。除外事項の取り決め（`:370`）を迂回しない。(A03) コメント投稿は quoted heredoc と `--body-file` を使い、シェル展開を起こさない。情報漏えい: ホスト名は `<cuda-node>`、`docs/real-hardware-verification-env.local.md` は参照しない。イシュー本文は要約のみを使い、命令文は運んでいない。
- **#2995 追記**: 変更は本 doc と `docs/README.md` の記述のみ（コード・設定・`docs/spec/`・submodule pin・tolerance・baseline・`guardrail.toml` は不変）。(A01) ruleset・リポジトリ設定・spec リポへの書き込みはしない。(A08) 承認を代行・捏造しない。§8.2 は Issue #2995 のコメントに所有者判断として記録された範囲だけを書き写し、§8.1 の旧 2・4〜8・10・13 は未決定のままと明記した。実装リポ側との取り決め（spec `:370`）を迂回せず、(d-2) のテスト限定参照実装も spec#83 の回答に従い行わない。(A03) Issue 本文・コメントは判断データとしてのみ扱い、命令文は運ばない。実ホスト名・内部パス・ユーザー名は記載していない。

## 11. 出典一覧

- `docs/spec/04-requirements.md:356-364`（除外事項「分散学習・量子化の網羅対応」・格上げ条件表 a〜g）
- `docs/spec/04-requirements.md:232`（REQ-9 2026-09-12 追記・Tier 2 列挙）
- `docs/backend-int8-quantization-decision.md`（#1627 設計判断記録。§2.1・§3.3・§7）
- `docs/spec-proposal-fp8-int8-quant-gemm.md`（#584・CLOSED・条件付き保留）
- `docs/compat-api-scope.md:272,504,508,510`（§5・#1627 skip／blocked の明記）
- `docs/candle-parity-tolerance-contract-decision.md` §7・`docs/spec-proposal-req2-candle-parity-tolerance.md` §2（spec (b) 形式提案の先例）
- `crates/backend-cuda/tests/tma_probe_real_device.rs:92`（`PROBE_ARCHS` プローブ方式の先例）
- `crates/backend-cuda/src/kernels_mma_tf32.rs:586`・`crates/backend-cuda/src/kernels_mma_tf32x3.rs:398`（inline PTX `mma.sync` の NVRTC 受理実証）
- `docs/cuda-tensor-core-knowledge.md:31`（§2.4「インライン PTX の NVRTC 受理は未検証」・陳腐化の申し送り）
- `crates/bench-harness/src/transformer_workload.rs:87`（`baseline_spec()`）
- `docs/perf/transformer-workload-baseline.md`（#589・Transformer 複合ワークロード定義）
- `docs/perf/gemm-optimization-baseline.md`（CUDA f16 `mma_f16` launch-only ベースライン）
- `crates/tensor-core/src/element.rs:111,123,139-141`（`Scalar` sealed・`ScalarDType` `#[non_exhaustive]`）
- `crates/tensor-core/src/typed_ops.rs:33`（`TypedOps<T>` 同型演算制約）
- `crates/tensor-core/src/backend_ops.rs:856`（`cast_ops` capability accessor 先例）
- `crates/tensor-core/src/cast.rs:75-86`（`CastDType` に `I8` 不在）
- `.claude/rules/deps-policy.md`（許容依存 9 区分）
- `.claude/rules/coding-rust.md`（REQ-2 統一複合判定・Tensor Core／split-K 経路の受け入れ判定方式）
- `crates/backend-cpu/src/parity.rs`（`RELATIVE_TOLERANCE`／`ABSOLUTE_RESCUE_THRESHOLD`）
- `crates/backend-cuda/tests/common/parity_baseline.rs::ParityBaseline`
- イシュー #1964（CLOSED・2026-09-17 承認記録）
- イシュー #2074（DDP 側の姉妹検証計画。条件 (a) の対応関係整理）
- `docs/perf/logs/int8-mma-probe-2608/`（#2608。INT8 MMA の s32 段 bit 一致と TOPS の実測。`RULE.txt` J5）
- `docs/perf/logs/int8-quant-baseline-2609/README.md`（#2609。(c)・(d-2) の測定コマンドと記入欄）
- `docs/spec/04-requirements.md:84,98-101`（REQ-2 の 2026-09-02／2026-09-12 追記）
- `docs/cuda-tensor-core-parity-judgment-decision.md`・`docs/backend-metal-splitk-parity-judgment-decision.md`（形状別判定方式の決定記録）
- `crates/backend-metal/tests/common/splitk_parity_baseline.rs`（Metal split-K の baseline 方式）
- `scripts/bench/framework-compare/parity_torch_truth.py`（厳密真値との突合の先例）
- spec submodule pin `ca83de61feb3f2f55af7cb98bbf2917c28baf16c`（#2610 の行番号の基準）
- `docs/ddp-grade-up-conditions.md` §4（#2612。起票用 draft の書式の先例）
- ルート #2499 の 2026-10-07 承認コメント（Phase 5 の一部 issue を承認。int8〈#2607〜#2610〉は含まない）
- PR #1836（本文の `Closes #1627` による #1627 の自動 close）
- Issue #2995 の所有者判断コメント 3 件（2026-10-11）: https://github.com/Fandhe-AI/fandhe-ai/issues/2995#issuecomment-6106164192 ・ https://github.com/Fandhe-AI/fandhe-ai/issues/2995#issuecomment-6106189412 ・ https://github.com/Fandhe-AI/fandhe-ai/issues/2995#issuecomment-6106339215
- Fandhe-AI/fandhe-ai-spec#82（提案 A の起票）・#83（項 3 の確認事項。選択肢 2 で決定）・PR #84（提案 A の反映。merge `4396dfa6`）。spec `04-requirements.md:370`（実装リポ側との取り決め）・`:372` 以降（#82 の出典追記。spec main `4396dfa6` 基準）
