# CPU GEMM マイクロカーネル・packing 再チューニング（対 gemm crate 逆転。イシュー #1041）

## §1 目的・受け入れ条件

- **課題**: 自作 `gemm_blis_parallel`（`crates/backend-cpu/src/gemm_blis/mod.rs`）は
  M4 Max N=2048 で 808 vs 893 GFLOP/s（gemm crate = candle CPU バックエンドの実体。約 0.90 倍）、
  GB10 Grace CPU で 467 vs 513 と劣位。第 0 回実機比較（`docs/perf/
  oss-gemm-comparison-baseline.md` §7.2）でも 512/1024/2048 で 0.84〜0.91 倍、4096 でのみ
  1.01 倍
- **受け入れ条件**:
  1. Apple M4 Max・DGX Spark GB10（Grace CPU）の N=1024/2048（正方 GEMM）で gemm crate
     （`scripts/bench/oss-gemm-compare/`）以上のスループットを達成する
  2. 5 回計測の中央値で記録する
  3. （副次目標）PyTorch CPU（Accelerate/AMX 経路。M4 Max で約 3 TFLOPS）との差も記録する
- **本 PR のスコープ**: 実装・bit 完全一致検証・A/B 一括計測ハーネスの整備まで。実機実測・
  受け入れ条件の達成判定は Mac／DGX Spark 実機セッションへ持ち越す（下記§5 参照）。
  **GB10（DGX Spark）側の実機実測はイシュー #1140 で完了し、非採用（§5.1・§6）と結論した。
  M4 Max 側は引き続き #1141 へ持ち越し。**

## §2 診断（コード読解に基づく重複コストモデル）

`gemm_blis_parallel`（`mod.rs`）は行パネル 1 次元分割
（`panel_rows = m.div_ceil(num_threads)`）で、各 rayon タスクが独立に
`gemm_blis_region`（直列 5-loop 相当）を呼ぶ。#750 で B パネル共有経路
（`gemm_blis_shared_b_region`）を追加し B の重複 pack は解消済みだが、A は
各タスクが (jc,pc) ブロックの組ごとに packing し直す（jc の反復回数ぶん同じ
行範囲を重複 pack する）。

| 形状（16 スレッド想定） | 各タスクの行数 | A packing の重複回数（jc 反復数） |
|---|---|---|
| N=2048（NC=512） | 128 行 = MC ちょうど（ic ループ 1 回） | jc 4 回 × pc 8 回 → 同一 A 行を 4 回 pack |
| N=1024（NC=512） | 64 行（MC 未満） | jc 2 回 × pc 8 回 → 同一 A 行を 2 回 pack |

gemm crate（faer 実体）は (mc,nc) タイル 2D 分配＋packing 共有のため A・B とも重複がない。
1024/2048 で劣位・4096 で拮抗という観測は「A packing 重複コスト（メモリ帯域）が演算量
2N³ に対して相対的に重い中形状ほど効く」仮説と整合する。設計検討
`docs/cpu-gemm-b-packing-sharing-decision.md` も案 B（共有 pack＋ic 並列）を推奨済みだが、
同ドキュメントは B の共有化のみを扱い A の重複は残る。

## §3 候補（`GemmDriverVariant`。`#[cfg(test)]` 限定）

`crates/backend-cpu/src/gemm_blis/mod.rs` に A/B 一括計測用の列挙 `GemmDriverVariant` と
入口 `gemm_blis_parallel_variant` を追加した（本番公開入口 `gemm_blis_parallel`／
`gemm_blis_bias_act_parallel` は変更しない）。

| 候補 | 内容 | 実装 |
|---|---|---|
| `RowPanel` | 本番既定と同一（行パネル分割・タスク数に関わらず常に `dispatch_region`。PR #1075 codex-review・Cursor Bugbot 指摘を受け `gemm_blis_parallel_with_blocks`〈実タスク数 2 以上で `dispatch_shared_b` へ分岐し本番と乖離〉から切替） | `gemm_blis_parallel_row_panel_with_blocks` |
| `SharedB` | B パネル共有・A は (jc,pc) ごとに再 pack（#750） | `dispatch_shared_b`（強制） |
| `SharedBPcOuter` | B パネル共有・A はタスクごとに pc ブロックあたり 1 回だけ pack（pc→jc の順に入替） | `dispatch_shared_b_pc_outer`（新規・イシュー #1041） |

### bit 完全一致契約（REQ-2）を保つ根拠

`gemm_naive` との bit 完全一致は「C の各要素が pc（縮約次元のブロック）昇順に
`f32::mul_add` で蓄積される」ことにのみ依存する（`mod.rs` 冒頭ドキュメント参照）。
C の各要素は (m,n) 座標で一意に決まる 1 つの (タスク行範囲, jc ブロック) の組にのみ
属し、ある pc 値の中で jc・タスク行範囲の反復順を入れ替えても、その要素が
「どの pc の時に触れられるか」の集合と大小関係は変化しない（pc は `SharedBPcOuter`
でも外側で昇順に回るため）。jc・タスク行範囲の入れ替えは互いに素な C 要素集合を
担当するだけで、同一要素の蓄積順序には影響しない。実装は
`crates/backend-cpu/tests`（integration test は `pub(crate)` 入口に到達できないため、
`mod.rs` 内 `#[cfg(test)] mod tests` に集約。lib 単体テストの既存方針を踏襲）に
以下の検証を追加した:

- `gemm_blis_shared_b_pc_outer_multi_sync_point_matches_serial_bit_exact`: 小さい
  `BlockSizes`（mc=16・kc=17・nc=19）で多数の (pc,jc) 同期点を強制し直列経路と bit 一致
- `gemm_blis_shared_b_pc_outer_matches_naive_bit_exact_when_tasks_fewer_than_threads`:
  実タスク数 Q < スレッド数 T のケース
- `gemm_blis_parallel_variant_all_candidates_match_naive_bit_exact`: 3 候補 ×
  5 形状（m==1・極小・MC 未満・MC ちょうど・MC/KC/NC 境界を跨ぐ非正方）× 4 スレッド数
  （1/2/3/16）の網羅グリッド

### A パネル容量の拡張

`panel_capacity` は「ic ループが `blocks.mc` 単位で A パネルを使い回す」前提で
`mc_len_max = blocks.mc.min(mc_total)` にクランプした容量を返すが、`SharedBPcOuter` は
タスクの担当行範囲全体（`task_mc` 行）を pc ブロックごとに 1 回で pack し、その pc の
全 jc 反復で使い回すため `blocks.mc` によるクランプを行わず `task_mc` 行ぶんの容量が要る
（新規ヘルパー `task_a_capacity`。`checked_mul` でオーバーフローを検出し
`GemmError::DimProductOverflow` へ変換。OWASP A03）。対象形状（N=1024/2048・16 スレッド）
では `task_mc`（64〜128 行）が既定 MC（128）以下のため、実質的な footprint 増加はない。

## §4 x86_64 ローカル smoke（採用根拠にしない）

実装セッションは Linux x86_64（aarch64 実機なし）のため、以下は**動作確認のみ**を目的とした
smoke 実行であり、REQ-8 受け入れ条件の判定には使わない（AVX2/AVX-512 実行系は M4 Max の
NEON・GB10 の Grace CPU（aarch64）と特性が異なる）。

```
$ cargo test -p fandhe-ai-backend-cpu --release --lib -- --ignored gemm_blis_variant_ab_1024_2048 --nocapture
variant=RowPanel size=1024 median_gflops=433.964
variant=SharedB size=1024 median_gflops=344.090
variant=SharedBPcOuter size=1024 median_gflops=407.660
variant=RowPanel size=2048 median_gflops=444.939
variant=SharedB size=2048 median_gflops=450.896
variant=SharedBPcOuter size=2048 median_gflops=476.784
```

x86_64（AVX2/AVX-512）では `SharedBPcOuter` が N=2048 で最良（対 RowPanel +7.2%）。
N=1024 では `RowPanel` が最良（x86_64 の MC/KC/NC・キャッシュ階層は aarch64 実測値と別物の
ため、この smoke だけでは実機での優劣を予測できない）。

## §5 実機計測手順（持ち越し）

1. `git pull` 済みの本ブランチで以下を実行:
   ```
   cargo test -p fandhe-ai-backend-cpu --lib gemm_blis
   ```
   （bit 完全一致回帰の確認。全 pass が前提）
2. A/B 計測（5 回**独立プロセス**実行の中央値の中央値。既存ベンチハーネスと同方針）:
   ```
   cargo test -p fandhe-ai-backend-cpu --release --lib -- --ignored gemm_blis_variant_ab_1024_2048 --nocapture
   ```
   を 5 回実行し、各 (variant, size) の中央値を記録する
3. OSS 直接比較（既存ハーネス。`docs/perf/oss-gemm-comparison-baseline.md` の手順を踏襲）:
   ```
   cargo run --manifest-path scripts/bench/oss-gemm-compare/Cargo.toml --release -- --sizes 512,1024,2048,4096
   ```
4. PyTorch CPU（副次目標。M4 Max: `scripts/bench/gemm_bench_torch_mps_f32.py` 相当の
   CPU 版スクリプトを用意して計測。GB10: 同様に Grace CPU 上で計測）
5. 下表へ記入し、1024/2048 の両方で `SharedBPcOuter`（または他候補）が gemm crate 以上、
   かつ 4096 で非劣化であれば採用候補として確定する

### 記入表（GB10 は実測済み。M4 Max は #1141 へ持ち越し）

| 実機 | N | RowPanel (GFLOP/s) | SharedB (GFLOP/s) | SharedBPcOuter (GFLOP/s) | gemm crate (GFLOP/s) | 対 gemm crate 比（最良候補） | PyTorch CPU (GFLOP/s) |
|---|---|---|---|---|---|---|---|
| M4 Max | 1024 | | | | | | |
| M4 Max | 2048 | | | | | | |
| M4 Max | 4096 | | | | | | |
| GB10 (Grace CPU) | 1024 | 536.2 | 249.3 | 260.9 | 604.4 | 0.887（RowPanel＝最良候補） | 未計測（§5.1 参照） |
| GB10 (Grace CPU) | 2048 | 701.6 | 366.1 | 382.9 | 688.3 | 1.019（RowPanel＝最良候補） | 未計測（§5.1 参照） |
| GB10 (Grace CPU) | 4096 | 1107.9 | A/B ハーネス対象外・未計測 | A/B ハーネス対象外・未計測 | 764.6 | 1.449（RowPanel＝最良候補） | 未計測（§5.1 参照） |

GB10 行の RowPanel 列は 1024/2048 が A/B ハーネス（`gemm_blis_variant_ab_1024_2048`）の
5 回中央値、4096 は OSS 比較ハーネス（`self_gemm_blis_parallel`＝本番 `RowPanel` 相当）の値
（A/B ハーネスは 4096 を対象としない）。gemm crate 列は OSS 比較ハーネスの値
（`tflops_median` × 1000）。「対 gemm crate 比」は表の RowPanel 列の値（1024: 536.2・
2048: 701.6・4096: 1107.9 GFLOP/s。1024/2048 は A/B ハーネスの中央値、4096 は A/B ハーネス
対象外のため OSS ハーネスの `self_gemm_blis_parallel` 値）を分子として算出した。参考として
OSS ハーネス側 `self_gemm_blis_parallel`（1024: 544.0・2048: 696.9 GFLOP/s）を分子にした場合
との差は 1024 で 1.4%・2048 で 0.7% で許容範囲内（§5.1 のクロスチェック参照）であり、いずれの
値を分子にしても採否判定の結論は変わらない。

## §5.1 GB10（Grace CPU）実測記録（イシュー #1140）

- **環境**: DGX Spark GB10（aarch64・Cortex-X925/A725 混成・20 論理コア）。
  `rustc 1.97.0 (2d8144b78 2026-07-07)`・`cargo 1.97.0`。リビジョン
  `06ec33bbd5d1100d848d0e91d4e9a803e452647a`（HEAD。#1167 マージ後）。実測日:
  2026-09-03。実ホスト名は `docs/real-hardware-verification-env.local.md`
  管理（本ドキュメントには記載しない）
- **手順**: §5 の 1〜3（PyTorch CPU は §5-4 のとおり未計測。理由は後述）をそのまま実行
- **受け入れ条件 1（bit 完全一致回帰）**: `cargo test -p fandhe-ai-backend-cpu --lib gemm_blis`
  が 94 passed・0 failed・5 ignored（実機専用ケース）で全 pass
  （`docs/perf/oss-comparison/2026-09-03/gb10-cpu-1140/step2-bitexact.log`）
- **受け入れ条件 2（A/B 計測。5 回独立プロセス）**: `cargo test -p fandhe-ai-backend-cpu
  --release --lib -- --ignored gemm_blis_variant_ab_1024_2048 --nocapture` を 5 回、
  各回の前に他プロセス混入なし（`uptime` load average を確認。他計測ジョブは走っておらず、
  1 回目 0.28 → 5 回目 2.53 の緩やかな上昇は自身の直前 20 スレッド計測の 1 分移動平均残留であり
  外部プロセス混入ではない）。生値は `docs/perf/oss-comparison/2026-09-03/gb10-cpu-1140/
  variant-ab-run{1..5}.txt`。中央値（GFLOP/s）:

  | variant | N=1024（5 値） | 中央値 | N=2048（5 値） | 中央値 |
  |---|---|---|---|---|
  | RowPanel | 539.888, 536.241, 541.550, 527.206, 525.862 | **536.2** | 701.620, 697.393, 703.535, 704.057, 699.069 | **701.6** |
  | SharedB | 247.138, 245.324, 252.582, 254.602, 249.301 | 249.3 | 366.078, 354.885, 362.500, 373.799, 373.397 | 366.1 |
  | SharedBPcOuter | 299.181, 260.893, 255.249, 242.902, 265.534 | 260.9 | 382.882, 371.028, 390.486, 374.127, 389.375 | 382.9 |

- **受け入れ条件 3（OSS 直接比較）**: `cargo run --manifest-path
  scripts/bench/oss-gemm-compare/Cargo.toml --release -- --sizes 512,1024,2048,4096`
  （生データ: `cpu-oss-compare.jsonl`・`cpu-oss-compare.stderr`・`cpu-oss-compare.exit`）。
  `exit=1` は K≥1024 の既知丸め不一致（`rel_diff` は 0.0012〜0.0047 で
  `docs/perf/oss-gemm-comparison-baseline.md` §5 記載の既知範囲内）によるもので、性能値は有効:

  | size | self_gemm_blis_parallel（TFLOPS） | matrixmultiply（TFLOPS） | gemm crate（TFLOPS） |
  |---|---|---|---|
  | 512 | 0.2602 | 0.1298 | 0.2493 |
  | 1024 | 0.5440 | 0.1321 | 0.6044 |
  | 2048 | 0.6969 | 0.1269 | 0.6883 |
  | 4096 | 1.1079 | 0.1259 | 0.7646 |

  **クロスチェック**（A/B ハーネス RowPanel 中央値 vs OSS ハーネス
  `self_gemm_blis_parallel`）: 1024 は 536.2 vs 544.0 GFLOP/s（差 1.4%）、2048 は 701.6 vs
  696.9 GFLOP/s（差 0.7%）で、いずれも計画で定めた ±10% 目安の範囲内。2 ハーネスの
  プロトコル差（warmup 3 vs 20・入力生成器の違い）による乖離は無視できる水準
- **副次目標（PyTorch CPU）**: 未計測。ノード上の既知 venv 3 件
  （`~/.venv-hf`・`~/poc/venv-cudatools`・`~/poc/venv-hf`）・システム Python いずれも
  `import torch` が `ModuleNotFoundError` で失敗し、torch 導入済み環境が見つからなかった。
  計画上「新規 pip install は行わない」方針のため、これ以上の追跡はせず未計測として記録する
- **結論**:
  1. **SharedB・SharedBPcOuter はいずれも GB10 上で本番既定 RowPanel を大きく下回る**
     （1024: RowPanel 536.2 に対し SharedB 249.3・SharedBPcOuter 260.9 でおよそ半分。
     2048 も同様の傾向）。M4 Max 向け診断（§2）が想定した「A packing 重複コストが
     中形状で相対的に重い」という仮説は Grace CPU（20 コア・NEON 実装のメモリ階層特性が
     Apple Silicon と異なる）では成立せず、B 共有化＋A 事前 pack のオーバーヘッド
     （タスクあたりの `task_a_capacity` 確保・B 全域を都度参照する経路）がむしろ支配的と
     推測される（定量診断は本 issue のスコープ外。詳細切り分けは必要になれば別 issue）
  2. **本番既定 RowPanel は GB10 上で gemm crate と既に拮抗〜優位**（1024: 0.887 倍〈従来の
     0.84〜0.91 倍帯の範囲内〉・2048: 1.019 倍・4096: 1.449 倍）。#1041 の課題認識（M4 Max
     N=2048 で 0.90 倍・GB10 で 0.91 倍）と異なり、GB10 では現行 HEAD（#1167 まで反映済み）
     の RowPanel が既に 2048/4096 で gemm crate 以上に達している
  3. **採否判定（GB10）: 非採用**。1024/2048 のいずれでも `SharedB`・`SharedBPcOuter` は
     `RowPanel`（gemm crate 比 1024: 0.887 倍・2048: 1.019 倍）を上回らない。受け入れ条件 1
     （「1024/2048 の両方で gemm crate 以上」）を満たす候補は GB10 では確認できなかった
     （RowPanel 自身は 2048 で条件を満たすが、これは候補の採用ではなく現状維持）

## §6 本番結線（別 PR・ユーザー承認）・引き継ぎ

`docs/cpu-gemm-b-packing-sharing-decision.md` §F・`docs/perf/cpu-gemm-b-packing-sharing.md`
と同じ採用ゲート方針を踏襲する: 実機 5 回中央値で受け入れ条件 1 を満たす候補が確定した後、
本番公開入口（`gemm_blis_parallel`／`gemm_blis_bias_act_parallel`）への結線は**別 PR**（ユーザー
承認後）で行う。1024/2048 で gemm crate 未満の候補しかなければ、次候補（B 側 laneq のベクトル
転置化・`vld1q_f32_x3` 経路の prefetch・KC 再スイープ）を追跡 Issue へ切り出す
（`.claude/rules/out-of-scope-tracking.md`）。

- **GB10 側の結論（イシュー #1140）**: `SharedB`・`SharedBPcOuter` は非採用（§5.1）。
  本番結線（#1144）は GB10 向けには不要（現行 `RowPanel` 既定を変更しない）。4096 は
  A/B ハーネス対象外のため候補別の直接比較値がなく、`RowPanel`（本番既定＝現状維持）の
  OSS ハーネス値のみを参考として記録した
- **M4 Max 側**: #1141 が同ドキュメント §5 表の M4 Max 行を別途埋める（本 PR では触れない）。
  M4 Max・GB10 の両方が出揃った時点で #1144（本番結線の要否判断）・#1117（親 issue）へ
  結果を集約する
- **次候補の引き継ぎ**: GB10 で非採用となったため、次候補（B 側 laneq のベクトル転置化・
  `vld1q_f32_x3` 経路の prefetch・KC 再スイープ）の要否は M4 Max 側の結果（#1141）と合わせて
  親 #1117 へコメントで引き継ぐ。新規 issue 起票はユーザー承認事項のため本 issue では行わない
  （`.claude/rules/out-of-scope-tracking.md`）

## 出典

- イシュー #1041（本ドキュメントの起票元）・#1117（親 issue）・#1140（GB10 実機実測。本追補）・
  #1141（M4 Max 実機実測）・#1144（本番結線の要否判断）
- `docs/perf/oss-gemm-comparison-baseline.md` §7.2・§7.3（対 gemm crate 比較ベースライン）
- `docs/cpu-gemm-b-packing-sharing-decision.md`（B 共有化の設計判断・採用ゲート方針の前例）
- `crates/backend-cpu/src/gemm_blis/mod.rs`（`GemmDriverVariant`・
  `gemm_blis_shared_b_pc_outer_region`・`gemm_blis_parallel_variant`）
- `docs/perf/oss-comparison/2026-09-03/gb10-cpu-1140/`（GB10 実測の生データ。イシュー #1140）
- `.claude/rules/out-of-scope-tracking.md`（後続タスク起票のユーザー承認要件）
