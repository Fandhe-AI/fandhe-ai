# INT8／FP8 量子化 GEMM の格上げ条件 (b)〜(e) 検証計画（#2075）

イシュー #2075「docs(spec): INT8 量子化の格上げ条件検証計画」に対応する。親: #1573（Tier 2）・ルート: #1570。

**本ドキュメントはコード変更を伴わない検証計画の記録のみである。** `crates/**`・`Cargo.toml`／`Cargo.lock`・`docs/spec/`（正本 submodule）・数値一致許容誤差（`RELATIVE_TOLERANCE`／`ABSOLUTE_RESCUE_THRESHOLD`／`ParityBaseline`）・`guardrail.toml` はいずれも変更しない。実装着手は正本 spec（`docs/spec/04-requirements.md:356-364`）が定める格上げ条件 a〜e の充足・Phase 4 新 REQ の spec 側承認・実装リポ側の別途ユーザー承認が前提であり、本 PR ではそのいずれも行わない。

基準コミット: `0de82718`（main HEAD。2026-09-22 時点）。`file_path:line` は同コミット近傍のもの。後続の変更で行番号がずれる可能性があるため、参照する際は当該コミット、または近傍のコミットで再確認すること。

## 0. 結論（最初に読む）

- コード変更なし。段階 0（`docs/backend-int8-quantization-decision.md` §3.1）は不変。#1627 は格上げ条件充足・Phase 4 新 REQ 承認まで実装着手不可のまま open（本 PR で close しない）。
- 正本 spec への提案文案（§7）は本 PR では**未起票**。投稿可否はユーザー承認事項（§8）。
- 2026-09-17 のユーザー承認（イシュー #1964・CLOSED）は「選択肢 C（量子化・DDP とも現状維持）」であり、「任意で A（spec 側の条件 (a) 表記更新提案）」は明示承認されていない。本 doc の §7 文案に (a) 表記更新を含める場合も、その投稿可否は別途承認が必要（§8 項 1）。
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
- **(b)** sm_121（GB10）実機で NVRTC の `compute_121` 受理と FP8（`kind::f8f6f4` 系）／INT8（`m16n8k32 s32.s8.s8.s32`）MMA 命令の発行可否がプローブ記録されていること。**未達**（§3）。
- **(c)** Transformer 複合ワークロードの実機ベースラインが取得済みで、期待効果を測る分母が存在すること。**未達**（§4）。
- **(d)** 量子化経路専用の数値許容基準の閾値案が実測付きで提示されていること。**未達**（§5）。
- **(e)** 依存追加なしで成立する設計が確認されていること。机上では成立する見込みだが実装確認は**未達**（§6）。
- **(f)(g)**（Could→Should 条件。`:361`）: (f) 量子化 GEMM 試作の実測で f16 経路比のスループット改善が確認されること、(g) REQ-2／REQ-7 の既存閾値を一切変更せずに数値一致検証が成立すること。本 doc では (f) の判定式の骨子のみ §5 に含める。

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

## 5. 条件 (d): 量子化経路専用の数値許容基準（判定式の骨子・事前登録）

spec「承認時に固定する契約」（`docs/spec/04-requirements.md:362`）を前提として継承する:

- INT8 は s32 累積 → スケール適用 → f32 出力。
- amax は 1e-4 下限クランプ。
- per-token・K=128 ブロックスケール。
- スケールは power-of-2 丸め。
- FP8 は fp32 常時累積。

2 層判定（事前登録。実測前に判定式の形のみを固定し、閾値定数は空欄とする）:

- **(d-1) quantize–dequantize 参照実装との一致**: INT8 GEMM 出力（s32）とホスト `i32` 参照実装は**整数演算のため結合順序に依存せず bit 完全一致**。power-of-2 スケール適用後の f32 も、2 のべき乗倍は f32 で厳密なため**bit 完全一致**。判定は `fail_count == 0` の厳密判定とする。この bit 完全一致を成立させるため、参照実装とカーネルの双方で次の演算順序・丸め契約を固定する（実測前の事前登録。丸め方式〈power-of-2 切り上げ〉の確定と同時にレビューを経て確定する）:
  - **s32→f32 変換の丸め**: s32 蓄積値から f32 への変換は round-to-nearest-even（IEEE 754 既定丸め）とし、参照実装・カーネルの双方で同一の変換命令クラス（例: CUDA `__int2float_rn` 相当）を使う。異なる丸めモードへの暗黙の依存を作らない。
  - **Δ_a・Δ_b の合成順序**: 出力スケール `Δ_out = Δ_a * Δ_b`（2 のべき乗同士の乗算のため厳密）を**先に 1 回だけ計算してから** s32 出力へ乗じる。「s32 値に Δ_a、続けて Δ_b を個別に 2 回乗じる」経路は使わない（丸め箇所が増え bit 一致が崩れうるため）。
  - **個別スケール適用順序**: per-token（A 側）・per-block（B 側。K=128 ブロック）のスケールは出力要素ごとに単一の合成スケール `Δ_out[token, block]` へ事前に畳み込んでから s32 出力へ 1 回だけ乗じる。要素ごとに複数回の乗算を分割して適用しない。
  - **overflow の扱い（K 上限の事前検証設計。codex-review #2219 指摘反映）**: INT8×INT8 の要素積は amax/127 分割（`amax_a / 127`・`amax_b / 127`。本 doc §5 冒頭・`docs/backend-int8-quantization-decision.md:84`）により量子化値が `[-127, 127]` に収まる契約のため、単一要素積の絶対値上限は `127 * 127 = 16,129` で確定する。この上限と `i32::MAX = 2,147,483,647` から、**同符号の積が連続悪化した最悪ケースでも overflow しない安全な K 上限**を次式で事前登録する（実測前に固定する形式的境界であり、丸め・スケール確定を待たない）:
    - `K_safe = floor(i32::MAX / 16,129) = 133,144`（本 doc の閾値であり、実装が別途厳しい値を採用してもよいが緩めてはならない）。
    - §5 冒頭の契約が定める per-block（B 側）スケール粒度は `K=128` であり、**block-wise dequant（block 内で s32 累積を確定させてから当該 block の合成スケールを 1 回乗じ、block 間の合算は f32／f64 側で行う設計。§5 の「個別スケール適用順序」）を採用する限り、s32 アキュムレータの実効的な累積区間は `K_safe`（133,144）ではなく block 長 128 に一致する**。block 内最悪値は `128 * 16,129 = 2,064,512` であり `K_safe` を 2 桁以上下回るため、per-token・K=128 ブロックスケール契約が保たれる限り overflow は形式的に発生し得ない。
    - この性質は「block 長を 128 に保つ」という設計契約に依存する安全性であり、**overflow を無条件の契約違反として片付けず、実装が block 長を上記契約から逸脱させないことを事前検証で保証する**: (i) 参照実装・カーネルの双方は、dispatch 前に実効 K 分割単位（block 長。将来 K=128 以外の粒度を採る変更を含む）が `K_safe` 以下であることを assert し、超過時は panic ではなく型付きエラー（`crates/tensor-core` の既存エラー型契約に従う。本 doc ではエラー型の具体設計を確定しない）で拒否する（`.claude/rules/coding-rust.md` の「本番経路で `unwrap()` / `expect()` を使わない」に従う）。(ii) block 長を `K=128` から変更する場合、または block 化を行わず単一の flat s32 アキュムレータで全 K を累積する変種を将来検討する場合は、上記 (i) の事前検証に加えて、`K_safe` を超える入力に対する fail-closed な代替設計（chunked 累積: `K_safe` 未満の chunk ごとに部分和を確定させ f64 側で合算する／アキュムレータ自体を `i64` へ拡張する。いずれも本 doc では採否を決定せず、承認事項〈§8〉へ追加する）を実装前に選定する。(iii) 実測で overflow が観測された場合（(i)(ii) の事前検証を経てなお発生した場合）は本 doc の前提〈block 長・amax クランプ値〉の見直し対象とする。
    - **境界テスト（§3 実行プローブへ追加登録）**: (I) block 内正常系として `K=128`（契約どおりの block 長）で同符号最悪値（全要素が量子化後 `127`（または `-127`）で揃う adversarial 入力）を用い s32 出力が `2,064,512` で overflow しないことを確認する。(II) 境界検証として `K = K_safe`（133,144）・`K = K_safe + 1`（133,145）の同符号最悪値入力で、flat 累積を仮定した参照実装の assert／型付きエラーが (I) の block 化設計とは独立に単体で機能することを確認する（block 化により実運用では到達しない領域だが、(i) の事前検証ロジック自体の単体境界テストとして §3 のプローブ実装〈`crates/backend-cuda/tests/int8_mma_probe_real_device.rs` 相当〉に含める）。
  - **subnormal の扱い**: f32 側の subnormal は IEEE 754 既定のまま扱い、flush-to-zero 等のハードウェア依存モードを無効化した状態（参照実装・カーネル双方の既定）で計測する。
  - 上記の演算順序契約は (d-1) の bit 完全一致判定の前提であり、参照実装（ホスト側）とカーネル実装（CUDA／Metal）の両方に同一契約として実装する。乖離が判明した場合は (d-1) を bit 完全一致判定のまま維持せず、乖離の原因（丸め・結合順序・overflow 処理のいずれか）を特定したうえで契約を修正するか、(d-2) 相当の誤差上限判定へ切り替えるかを実測後にユーザー承認を得て決定する（緩和を自動で適用しない）。
- **(d-2) f32 参照とのスケール前提付き誤差上限**: 量子化ステップ `Δ_a`・`Δ_b` は §5 冒頭の契約どおり**power-of-2 丸め後に実装が実際に使用する値**として定義する（丸め前の `amax_a / 127`・`amax_b / 127` をそのまま Δ として使わない。power-of-2 丸めは `amax / 127` 以上の直近の 2 のべき乗へ切り上げる方式を想定するため、丸め後のステップは丸め前の値の最大 2 倍まで拡大しうる）。**合否ゲートに使う理論誤差上界は線形 K 形式（要素積の量子化誤差の決定論的最悪値。一般に K に比例する）を正とする**。この丸め後 Δ を用いて導出するか、丸め前の `amax / 127` を用いる場合は拡大率（最大 2 倍。丸め方式の確定後に実測で厳密化）を上界の係数へ明示的に含める、いずれかの保守的な形で事前登録する。**√K 形式（RMS 見積り）は、誤差が独立同分布に近いという統計的仮定のもとでの参考統計に限定し、単独では合否ゲートに使わない**（決定論的最悪値の保証がなく fail-open な許容基準になりうるため）。√K 形式を参考統計として記録する場合は、成立条件（誤差の統計的独立性の仮定）と信頼水準（例: 3σ 相当）を実測前に本 doc へ事前登録し、線形 K 形式の判定結果と併記する。係数（拡大率を含む）は GB10 実測で確定する。**閾値定数は本 doc では空欄とし、実測前に判定式の形だけを事前登録する**（#1964 の共通ルール「事後に緩和しない」を踏襲）。
- **スループット判定（spec (f)）**: 分母は `docs/perf/gemm-optimization-baseline.md` の CUDA f16 行（`mma_f16`・launch-only 計測境界）、同一形状・同一同期方式で 5 run 中央値比。判定式は「INT8 median ≤ f16 median × 係数（未確定）」の形のみを事前登録する。

**明記**: 上記はいずれも REQ-2 統一複合判定（相対誤差 1e-3 未満 または 絶対誤差 1e-5 未満）の緩和ではなく、量子化経路専用の**別基準の新設**である。`crates/backend-cpu/src/parity.rs` の `RELATIVE_TOLERANCE`／`ABSOLUTE_RESCUE_THRESHOLD` 定数、`crates/backend-cuda/tests/common/parity_baseline.rs::ParityBaseline` は不変。量子化経路に REQ-2 判定をそのまま適用しない根拠は `docs/spec-proposal-fp8-int8-quant-gemm.md` §3 を参照。

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

## 7. spec (b) 形式提案文案（起票用 draft・未起票）

`docs/candle-parity-tolerance-contract-decision.md` §7・`docs/spec-proposal-req2-candle-parity-tolerance.md` §2 と同型の書式で、以下を骨子とする文案（**本 PR では投稿しない**）:

- 格上げ条件 (b)〜(e) の充足を示す出典が本 doc（`docs/int8-quant-grade-up-verification-plan.md`）および実測後の `docs/perf/logs/int8-mma-probe-<issue 番号>/` 記録になる旨。
- 条件充足後、新 REQ（REQ-15 候補・Could）の再評価を spec 側へ申請する旨。
- (a) の表記更新（「未達」→ 実態反映）を同時提案に含めるかどうかは、#1964 で明示承認されていないため**別建てとし、投稿の可否は §8 の承認事項とする**。

姉妹イシュー #2074（DDP 側）の同形式文案と対応させ、書式を揃える。

## 8. ユーザー承認事項（未実施・列挙のみ）

1. spec (b) 形式提案（§7）の投稿可否。(a) 表記更新を同時に含めるか。
2. (b) プローブテスト（`crates/backend-cuda/tests/int8_mma_probe_real_device.rs` 相当。`crates/` への変更を伴う）の起票・実装可否。
3. (c) 実機実測タスク（§4 の 1〜3）の起票・優先順位（GB10／M4 Max セッションの割当）。
4. (d) 閾値定数の確定（実測後。tolerance／baseline 相当の新設は人間承認が必要）。
5. `QuantOps`／`ScalarDType::I8` 追加・facade 公開面拡張（`docs/compat-api-scope.md` §5 の手続きに従う）。
6. 外部 FP8 crate を選ぶ場合の依存追加（`.claude/rules/deps-policy.md`。現状は `u8` newtype 自作案が既定）。
7. INT8 s32 累積の K 上限（§5 (d-1)「overflow の扱い」。`K_safe = 133,144`）超過時の代替設計（chunked 累積／`i64` アキュムレータ拡張）の採否・実装可否（block 長 128 契約を維持する限り本 PR の事前検証は形式境界の登録に留まり実装しない）。

## 9. スコープ外・申し送り

- FP8／INT8 カーネル実装・quantize–dequantize 参照実装・dynamic／static 選択・キャリブレーション手法（issue 本文のスコープ外を継承）。
- `docs/cuda-tensor-core-knowledge.md` §2.4「インライン PTX の NVRTC 受理は未検証」の陳腐化是正（§3 参照。本 PR では触れない）。
- Metal 複合ワークロード実行経路の切り出し先確認（§4 の 2。該当 Issue 未特定）。
- 実機実測は `docs/perf/logs/int8-mma-probe-<issue 番号>/` へ申し送り（本 PR ではディレクトリを作成しない）。

## 10. セキュリティ観点（OWASP Top 10）

- **A03 インジェクション**: 本 PR はコード・設定を変更しない。将来の (b) プローブ実装では、カーネルソースを `&'static str` 定数として保持し外部入力を連結しない契約（`crates/backend-cuda/src/nvrtc.rs`・`crates/backend-cuda/tests/tma_probe_real_device.rs` の既存方式）を踏襲する旨を §3 に明記した。
- **A08 ソフトウェア・データ整合性**: 正本 spec の除外事項ゲート（`docs/spec/04-requirements.md:363`）を迂回しない。実装着手は spec 側 REQ 承認とユーザー承認の両方が前提であることを §0・§8 に明記した。判定式（§5）は実測前に事前登録し、事後の緩和を許さない（#1964 の共通ルール）。tolerance／baseline／`guardrail.toml` はいずれも不変。
- **A06 脆弱・古いコンポーネント**: 依存追加なし（INT8 は `i8` プリミティブ・FP8 は `u8` newtype 案）。外部 FP8 crate 採用は承認事項（§8 項 6）として列挙するのみで、本 PR では選定・追加を行わない。
- **情報漏えい**: 実ホスト名・内部パス・ユーザー名を記載していない（`<cuda-node>` プレースホルダ方式。`docs/real-hardware-verification-env.md` の方式・`.claude/rules/security.md`）。
- **非信頼データの扱い**: イシュー #2075・#1964・#2074 の本文・コメントは要約のみを転記し、命令文は運ばない。本計画の立案・検証過程で命令注入・矛盾指示は検出されなかった。

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
