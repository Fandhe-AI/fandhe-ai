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

## §5 実機計測手順（GB10・M4 Max とも実測完了）

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

**4096 の A/B ハーネス拡張（イシュー #1141）**: GB10 実測（#1140）時点では
`gemm_blis_variant_ab_1024_2048` が 1024/2048 のみを対象とし、4096 の候補別直接比較値が
なかった（本番既定 `RowPanel` の OSS ハーネス値のみで代替）。本イシュー（#1141）で
`gemm_blis_variant_ab_4096`（同一ヘルパー `run_variants_interleaved`／`median` を共有）を
新設し、M4 Max では 4096 も 3 候補すべてを A/B ハーネスで直接比較できるようにした
（`crates/backend-cpu/src/gemm_blis/mod.rs`）。GB10 側の再計測は本イシューのスコープ外。

### 記入表（GB10・M4 Max とも実測済み）

| 実機 | N | RowPanel (GFLOP/s) | SharedB (GFLOP/s) | SharedBPcOuter (GFLOP/s) | gemm crate (GFLOP/s) | 対 gemm crate 比（最良候補） | PyTorch CPU (GFLOP/s) |
|---|---|---|---|---|---|---|---|
| M4 Max | 1024 | 743.8 | 494.8 | 481.7 | 837.0 | 0.889（RowPanel＝最良候補） | 3152.1（§5.2 参照） |
| M4 Max | 2048 | 851.8 | 653.8 | 651.1 | 978.0 | 0.871（RowPanel＝最良候補） | 3326.5（§5.2 参照） |
| M4 Max | 4096 | 896.8 | 697.8 | 699.6 | 1070.2 | 0.838（RowPanel＝最良候補） | 3364.4（§5.2 参照） |
| GB10 (Grace CPU) | 1024 | 536.2 | 249.3 | 260.9 | 617.8 | 0.868（RowPanel＝最良候補） | 未計測（§5.1 参照） |
| GB10 (Grace CPU) | 2048 | 701.6 | 366.1 | 382.9 | 693.4 | 1.012（RowPanel＝最良候補） | 未計測（§5.1 参照） |
| GB10 (Grace CPU) | 4096 | 1107.4 | A/B ハーネス対象外・未計測 | A/B ハーネス対象外・未計測 | 764.1 | 1.449（RowPanel＝最良候補） | 未計測（§5.1 参照） |

GB10 行の RowPanel 列は 1024/2048 が A/B ハーネス（`gemm_blis_variant_ab_1024_2048`）の
5 回中央値、4096 は OSS 比較ハーネスの `self_gemm_blis_parallel`（＝本番 `RowPanel` 相当）の
**5 独立プロセス実行の中央値**（A/B ハーネスは 4096 を対象としない。#1141 で新設した
`gemm_blis_variant_ab_4096` は GB10 では未実行）。**gemm crate 列も同じく OSS 比較ハーネスを
5 独立プロセス実行した `tflops_median` の run 間中央値 × 1000**（codex-review P1 指摘対応・
§5.1「受け入れ条件 3」参照。従来は OSS ハーネス 1 回実行内の中央値のみを記録しており候補側
〈5 独立プロセス中央値〉と非対称だったため、gemm crate 側も同一プロトコル〈5 独立プロセス・
各回の run 内中央値〉へ揃えた）。「対 gemm crate 比」は表の RowPanel 列の値（GB10:
1024: 536.2・2048: 701.6・4096: 1107.4 GFLOP/s。1024/2048 は A/B ハーネスの中央値、4096 は
A/B ハーネス対象外のため OSS ハーネス 5 回実行の `self_gemm_blis_parallel` run 間中央値）を
分子、gemm crate 列（OSS ハーネス 5 回実行の run 間中央値）を分母として算出した。参考として
A/B ハーネス（1024: 536.2・2048: 701.6 GFLOP/s）と OSS ハーネス 5 回実行側
`self_gemm_blis_parallel`（1024: 531.4・2048: 698.2 GFLOP/s）の差は 1024 で 0.9%・2048 で
0.5%（§5.1 のクロスチェック参照）で、いずれの値を分子にしても採否判定の結論は変わらない。

M4 Max 行の RowPanel／SharedB／SharedBPcOuter 列はいずれも `gemm_blis_variant_ab_1024_2048`
（1024/2048）・`gemm_blis_variant_ab_4096`（4096。#1141 新設）の **5 回独立プロセス実行の
中央値の中央値**、gemm crate 列は OSS 比較ハーネスを 5 独立プロセス実行した `tflops_median`
の run 間中央値 × 1000（GB10 と同一プロトコル）。「対 gemm crate 比」は A/B ハーネスの
RowPanel 列（最良候補）を分子、OSS ハーネスの gemm crate 列を分母として算出した。詳細な
生値・較正差クロスチェック・PyTorch CPU 結果は §5.2 を参照。

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

- **受け入れ条件 3（OSS 直接比較。5 独立プロセス実行）**: `cargo run --manifest-path
  scripts/bench/oss-gemm-compare/Cargo.toml --release -- --sizes 512,1024,2048,4096`
  を**独立プロセスとして 5 回実行**し、各実装・形状ごとの run 間中央値を採用する
  （codex-review P1 指摘対応。従来は 1 回実行内の中央値のみを記録しており、候補側
  〈自作カーネル A/B ハーネス〉が既に 5 独立プロセス中央値だったのに対し gemm crate
  側と非対称だったため是正した）。5 回とも他計測ジョブの混入なし（`loadavg-per-run.txt`。
  各回の load average は 2.5〜3.6 で計測プロセス自身〈20 スレッド〉の残留のみ）。生データ:
  `docs/perf/oss-comparison/2026-09-03/gb10-cpu-1140-ossx5/oss-run{1..5}.jsonl`
  （`.stderr`・`.exit` も同ディレクトリに同梱）。各レコードの `commit` フィールドは
  `"unknown"` になっている（`docs/real-hardware-verification-env.md` §3 の rsync 転送
  手順どおりノード側に `.git` を置いていないため、`oss-gemm-compare` バイナリの
  `git_commit_short()` がフォールバック値を返す既知の制約。ハーネス側の不具合ではない）。
  代わりに転送前に記録した rev-stamp（`git rev-parse HEAD`）を同ディレクトリの
  `env.txt`（`=== rev-stamp ===` 節）へ同梱し、5 回の実行がいずれも
  `06ec33bbd5d1100d848d0e91d4e9a803e452647a`（本ドキュメント冒頭の環境節と同一
  コミット）に対する計測であることを紐付けた（codex-review 指摘対応。イシュー #1140）。
  5 回とも `exit=1` は、ハーネスが**意図的に持つ既定 fail-closed 挙動そのもの**
  （`docs/perf/oss-gemm-comparison-baseline.md` §5「全サイズの JSON Lines 出力を終えたうえで、
  突合 NG を 1 件でも検出していれば非 0 終了する」。イシュー #755 で確定した仕様であり、
  既知の限界を理由に非 fatal へ戻さない方針）であって、ハーネスの正当性検査そのものが
  破綻・中断したことを意味しない。スループット値は非 0 終了より**前**に全サイズぶん
  出力済みであり（同 §5 の記載どおり）、打ち切りによる欠損値ではない（codex-review P2
  指摘対応。イシュー #1140）。

  この `output_match=false` は**候補実装自身の正しさ**ではなく `matrixmultiply`・
  `gemm` crate との**縮約順序差に由来する丸め誤差**（クロス実装間出力突合）にのみ現れる
  ことを生データで確認した（`docs/perf/oss-comparison/2026-09-03/gb10-cpu-1140-ossx5/
  oss-run1.jsonl` 他 4 ファイル。`self_gemm_blis_parallel` レコードは全サイズ・全 5 回で
  `output_match=true`）。候補自身の正しさは、本ドキュメントが依拠する別軸の
  bit 完全一致契約（受け入れ条件 1・`gemm_naive` との bit 完全一致。全 94 テスト pass。
  `step2-bitexact.log`）で既に検証済みであり、この OSS ハーネスの `exit=1` とは独立
  である。`rel_diff` 自体も 0.0012〜0.0047 で `docs/perf/oss-gemm-comparison-baseline.md`
  §5 記載の既知範囲内（同 §7.2 の 2026-08-23 M4 Max 正式ベースラインでも同様の
  `output_match=false`・`exit=1` のもとで性能値を採用済みであり、本ドキュメントは
  その既承認プロトコルを踏襲している。全 5 回で症状が同一のため実行間で不安定化しては
  いない）。したがって、この `exit=1` を理由に「既知誤差許容モード」を新設する、または
  correctness gate 通過を前提に再計測することは、#755 で確定した fail-closed 方針を
  覆す変更（`.claude/rules/coding-rust.md` の許容誤差単独緩和禁止の対象）に該当し、
  本ドキュメントのスコープ外・ユーザー承認事項である。性能値は有効:

  | size | 実装 | 5 回の値（TFLOPS） | 中央値（TFLOPS） | 中央値（GFLOP/s） |
  |---|---|---|---|---|
  | 512 | self_gemm_blis_parallel | 0.2815, 0.2896, 0.2705, 0.2801, 0.2616 | 0.2801 | 280.1 |
  | 512 | matrixmultiply | 0.1298, 0.1319, 0.1318, 0.1299, 0.1321 | 0.1318 | 131.8 |
  | 512 | gemm crate | 0.2532, 0.3300, 0.2922, 0.2701, 0.4996 | 0.2922 | 292.2 |
  | 1024 | self_gemm_blis_parallel | 0.5236, 0.5379, 0.5314, 0.5158, 0.5399 | 0.5314 | 531.4 |
  | 1024 | matrixmultiply | 0.1321, 0.1320, 0.1326, 0.1281, 0.1322 | 0.1321 | 132.1 |
  | 1024 | gemm crate | 0.5956, 0.6991, 0.6178, 0.6186, 0.6127 | 0.6178 | 617.8 |
  | 2048 | self_gemm_blis_parallel | 0.7009, 0.7080, 0.6982, 0.6962, 0.6965 | 0.6982 | 698.2 |
  | 2048 | matrixmultiply | 0.1299, 0.1261, 0.1303, 0.1268, 0.1299 | 0.1299 | 129.9 |
  | 2048 | gemm crate | 0.6795, 0.6986, 0.6934, 0.6963, 0.6822 | 0.6934 | 693.4 |
  | 4096 | self_gemm_blis_parallel | 1.1130, 1.1071, 1.1052, 1.1092, 1.1074 | 1.1074 | 1107.4 |
  | 4096 | matrixmultiply | 0.1258, 0.1289, 0.1263, 0.1258, 0.1289 | 0.1263 | 126.3 |
  | 4096 | gemm crate | 0.7635, 0.7656, 0.7649, 0.7628, 0.7641 | 0.7641 | 764.1 |

  512 の gemm crate 列は 5 回目（`oss-run5.jsonl` の 0.4996。`oss-run1.jsonl` は 0.2532）が
  他 4 回（0.25〜0.33 帯）から外れた外れ値だが、中央値採用のため結論への影響はない（512 は
  受け入れ条件の対象形状〈1024/2048〉外であり参考値）。

  **クロスチェック**（A/B ハーネス RowPanel 中央値 vs OSS ハーネス 5 回実行
  `self_gemm_blis_parallel` の run 間中央値）: 1024 は 536.2 vs 531.4 GFLOP/s（差 0.9%）、
  2048 は 701.6 vs 698.2 GFLOP/s（差 0.5%）で、いずれも計画で定めた ±10% 目安の範囲内。
  2 ハーネスのプロトコル差（warmup 3 vs 20・入力生成器の違い）による乖離は無視できる水準
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
  2. **本番既定 RowPanel は GB10 上で gemm crate と既に拮抗〜優位**（1024: 0.868 倍〈従来の
     0.84〜0.91 倍帯の範囲内〉・2048: 1.012 倍・4096: 1.449 倍。gemm crate 側も 5 独立
     プロセス実行の run 間中央値へ是正済み。§5.1 受け入れ条件 3）。#1041 の課題認識（M4 Max
     N=2048 で 0.90 倍・GB10 で 0.91 倍）と異なり、GB10 では現行 HEAD（#1167 まで反映済み）
     の RowPanel が既に 2048/4096 で gemm crate 以上に達している。2048 の 1.012 倍は僅差
     ながら 5 独立プロセス同士の比較で再現し、境界付近の結論として成立する
  3. **採否判定（GB10）: 非採用**。1024/2048 のいずれでも `SharedB`・`SharedBPcOuter` は
     `RowPanel`（gemm crate 比 1024: 0.868 倍・2048: 1.012 倍）を上回らない。受け入れ条件 1
     （「1024/2048 の両方で gemm crate 以上」）を満たす候補は GB10 では確認できなかった
     （RowPanel 自身は 2048 で条件を満たすが、これは候補の採用ではなく現状維持）

## §5.2 Apple M4 Max 実測記録（イシュー #1141）

- **環境**: Apple M4 Max（16 論理コア）・64 GiB・macOS 26.6.2（BuildVersion 25G83）。
  `rustc 1.96.0 (ac68faa20 2026-05-25)`・`cargo 1.96.0 (30a34c682 2026-05-25)`。リビジョン
  `49516af`（HEAD。DGX Spark CPU 実測 PR #1170 マージ後）。実測日: 2026-09-03。ローカル
  直接実行（`docs/real-hardware-verification-env.md` §1。SSH・rsync 不要）。実ホスト名は
  記載しない
- **並列実行の注意**: 同一 Mac 上で他の並列イシュー（別 worktree）が実行中で、計測開始前に
  1 コアを専有するプロセスを検出した（`docs/perf/oss-comparison/2026-09-03/cpu-retune-1041/
  hygiene.txt`）。16 論理コアのうち 1 コア分の競合であり、5 回とも候補間の相対順位
  （RowPanel が SharedB・SharedBPcOuter を常に大きく上回る）は安定して再現したため、
  破棄対象の異常runとは判定しなかった。計測後の再確認では競合プロセスは解消していた
- **受け入れ条件 1（bit 完全一致回帰）**: `cargo test -p fandhe-ai-backend-cpu --lib gemm_blis`
  が 94 passed・0 failed・6 ignored（実機専用ケース。#1141 で追加した
  `gemm_blis_variant_ab_4096` を含む）で全 pass
  （`docs/perf/oss-comparison/2026-09-03/cpu-retune-1041/unit-test.txt`）
- **受け入れ条件 2（A/B 計測。5 回独立プロセス）**: `gemm_blis_variant_ab_1024_2048`
  （1024/2048）・`gemm_blis_variant_ab_4096`（4096。#1141 新設）をそれぞれ 5 回実行。
  生値は `docs/perf/oss-comparison/2026-09-03/cpu-retune-1041/ab-{1024-2048,4096}-run
  {1..5}.txt`。中央値（GFLOP/s）:

  | variant | N=1024（5 値） | 中央値 | N=2048（5 値） | 中央値 | N=4096（5 値） | 中央値 |
  |---|---|---|---|---|---|---|
  | RowPanel | 731.069, 748.002, 743.835, 730.893, 763.901 | **743.8** | 895.329, 881.652, 851.813, 846.297, 818.775 | **851.8** | 942.979, 873.643, 896.765, 885.690, 905.828 | **896.8** |
  | SharedB | 516.672, 494.783, 486.310, 498.328, 479.340 | 494.8 | 652.436, 657.117, 653.766, 656.312, 651.763 | 653.8 | 733.895, 703.901, 697.769, 694.844, 689.734 | 697.8 |
  | SharedBPcOuter | 486.880, 485.554, 472.758, 481.746, 462.903 | 481.7 | 648.896, 651.124, 656.563, 654.143, 648.724 | 651.1 | 733.393, 702.680, 698.327, 699.642, 696.726 | 699.6 |

  いずれのサイズ・いずれの run でも `RowPanel` が `SharedB`・`SharedBPcOuter` を明確に
  上回った（1024: SharedB 比 +50.4%・SharedBPcOuter 比 +54.4%／2048: +30.3%／+30.8%／
  4096: +28.5%／+28.2%）。GB10（§5.1）と同じ傾向であり、M4 Max 向け診断（§2）が想定した
  「A packing 重複コストが中形状で相対的に重い」という仮説は本 HEAD 時点の M4 Max 実測でも
  再現しない
- **受け入れ条件 3（OSS 直接比較。5 独立プロセス実行）**: `cargo run --manifest-path
  scripts/bench/oss-gemm-compare/Cargo.toml --release -- --sizes 512,1024,2048,4096` を
  5 回実行。3 回目実行は `Bash` ツール既定タイムアウト（120s）で打ち切られたため不完全出力
  （`ossx5/oss-run3.jsonl` 途中まで）を破棄し、同一コマンドをタイムアウト制約のない単発
  呼び出しで再実行して置き換えた（`hygiene.txt` に記録。他 4 回は正常終了）。5 回とも
  `exit=1` は既知の fail-closed 仕様（`docs/perf/oss-gemm-comparison-baseline.md` §5・
  §5.1 と同じ理由。1024/2048/4096 の `matrixmultiply`・4096 の `gemm` crate で
  `output_match=false`・`rel_diff` は 0.0012〜0.0047 で既知範囲内）。生データ:
  `docs/perf/oss-comparison/2026-09-03/cpu-retune-1041/ossx5/oss-run{1..5}.jsonl`
  （`.stderr`・`.exit` も同ディレクトリ）。`commit` フィールドはローカル直接実行のため
  `49516af`（本節冒頭の環境節と同一）。

  | size | 実装 | 5 回の値（TFLOPS） | 中央値（TFLOPS） | 中央値（GFLOP/s） |
  |---|---|---|---|---|
  | 512 | self_gemm_blis_parallel | 0.4099, 0.4333, 0.4244, 0.4402, 0.4790 | 0.4333 | 433.3 |
  | 512 | matrixmultiply | 0.1113, 0.1072, 0.1100, 0.1098, 0.1112 | 0.1100 | 110.0 |
  | 512 | gemm crate | 0.5281, 0.5347, 0.5138, 0.5188, 0.4986 | 0.5188 | 518.8 |
  | 1024 | self_gemm_blis_parallel | 0.7390, 0.7033, 0.7949, 0.7863, 0.8026 | 0.7863 | 786.3 |
  | 1024 | matrixmultiply | 0.1104, 0.1091, 0.1119, 0.1120, 0.1117 | 0.1117 | 111.7 |
  | 1024 | gemm crate | 0.8571, 0.7768, 0.8494, 0.8370, 0.8151 | 0.8370 | 837.0 |
  | 2048 | self_gemm_blis_parallel | 0.8346, 0.7425, 0.8970, 0.8767, 0.8742 | 0.8742 | 874.2 |
  | 2048 | matrixmultiply | 0.1104, 0.1088, 0.1110, 0.1109, 0.1111 | 0.1109 | 110.9 |
  | 2048 | gemm crate | 0.8923, 0.7844, 0.9893, 0.9780, 0.9790 | 0.9780 | 978.0 |
  | 4096 | self_gemm_blis_parallel | 0.9369, 0.8652, 1.0504, 1.0408, 1.0676 | 1.0408 | 1040.8 |
  | 4096 | matrixmultiply | 0.1079, 0.1082, 0.1102, 0.1103, 0.1105 | 0.1102 | 110.2 |
  | 4096 | gemm crate | 0.7190, 0.9138, 1.0747, 1.0702, 1.0792 | 1.0702 | 1070.2 |

  **クロスチェック**（A/B ハーネス RowPanel 中央値 vs OSS ハーネス 5 回実行
  `self_gemm_blis_parallel` の run 間中央値）: 1024 は 743.8 vs 786.3 GFLOP/s（差 5.7%）、
  2048 は 851.8 vs 874.2 GFLOP/s（差 2.6%）、4096 は 896.8 vs 1040.8 GFLOP/s（差 16.1%）。
  GB10 実測（§5.1・差 0.9%／0.5%）と比べて M4 Max は run 間の分散が大きく（とくに
  `self_gemm_blis_parallel` 4096 は 0.87〜1.07 TFLOPS の範囲で 2 群に分かれる挙動が
  見られた）、4096 の較正差は計画で目安とした ±10% を超える。この分散は §5.2 冒頭の
  並列実行注意に記載した外部プロセス競合、または周波数ブースト・サーマル挙動の run 間差
  に起因すると推測されるが、定量診断は本イシューのスコープ外とする（`.claude/rules/
  out-of-scope-tracking.md`）。ただし候補間の相対順位（`RowPanel` が他 2 候補を常に大きく
  上回る）は 5 回とも安定して再現しており、後述の採否判定への影響はない
- **対 gemm crate 比（A/B ハーネス RowPanel 分子・OSS ハーネス gemm crate 分母）**:
  1024: 743.8 ÷ 837.0 = **0.889**、2048: 851.8 ÷ 978.0 = **0.871**、
  4096: 896.8 ÷ 1070.2 = **0.838**。参考として、較正差を考慮した保守値（OSS ハーネス内
  同一プロトコルの `self_gemm_blis_parallel` ÷ gemm crate。最良候補が `RowPanel` 自身の
  ため A/B 側の較正係数は 1）は 1024: 0.939・2048: 0.894・4096: 0.973 で、いずれのプロトコル
  を採っても 1024/2048/4096 のすべてで 1.0 を下回る
- **副次目標（PyTorch CPU）**: 計測完了（`.venv-mps-bench`〈`.gitignore` 対象・リポジトリ
  管理外〉に `pip install torch` して実行）。`torch=2.14.0`・`device=cpu`・
  `at::get_num_threads()=12`（ATen parallel backend: OpenMP。MKLDNN は未検出）。
  生ログ: `docs/perf/oss-comparison/2026-09-03/cpu-retune-1041/torch-cpu-f32.txt`

  | size | PyTorch CPU f32（TFLOPS） | GFLOP/s |
  |---|---|---|
  | 512 | 2.5404 | 2540.4 |
  | 1024 | 3.1521 | 3152.1 |
  | 2048 | 3.3265 | 3326.5 |
  | 4096 | 3.3644 | 3364.4 |

  PyTorch CPU（Accelerate/AMX 経由と推測。`MKLDNN not found` のログから OpenMP 経路である
  ことのみ確認でき、AMX 到達性の直接確認は本イシューのスコープ外）は本番既定 `RowPanel`
  （0.90〜1.04 TFLOPS 帯）の 3〜4 倍のスループットで、gemm crate（0.84〜1.07 TFLOPS 帯）
  との比較でも参考値として大きく上回る。本イシューの採否判定（対 gemm crate 比較）には
  使わない参考記録
- **結論**:
  1. **SharedB・SharedBPcOuter はいずれも M4 Max 上で本番既定 RowPanel を大きく下回る**
     （1024: RowPanel 743.8 に対し SharedB 494.8・SharedBPcOuter 481.7 でおよそ 65〜66%・
     2048/4096 も RowPanel の 78〜80% 程度）。GB10（§5.1）と同一の傾向であり、M4 Max
     向け診断（§2）が想定した仮説はいずれの実機でも支持されなかった
  2. **本番既定 RowPanel は M4 Max 上で gemm crate を 1024/2048/4096 のいずれでも
     下回る**（対 gemm crate 比 0.838〜0.889。§5.2 対 gemm crate比参照）。GB10（1024:
     0.868・2048: 1.012・4096: 1.449）と異なり M4 Max では 4096 でも gemm crate に届いて
     いない。CLAUDE.md 冒頭に記載の v0.6.0 再計測（PR #1127）時点の 0.76 倍（1024）・
     0.82 倍（2048）と比べるとやや改善しているが、依然としてギャップが残る
  3. **採否判定（M4 Max）: 非採用**。1024/2048/4096 のいずれでも `SharedB`・
     `SharedBPcOuter` は `RowPanel`（gemm crate 比 0.838〜0.889）を上回らない。受け入れ
     条件 1（「1024/2048 の両方で gemm crate 以上、かつ 4096 で非劣化」）を満たす候補は
     M4 Max では確認できなかった（`RowPanel` 自身も 3 サイズいずれも gemm crate 未達の
     ため、GB10 の 2048 のような部分達成もない）

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
- **M4 Max 側の結論（イシュー #1141）**: `SharedB`・`SharedBPcOuter` は非採用（§5.2）。
  本番結線（#1144）は M4 Max 向けにも不要（現行 `RowPanel` 既定を変更しない）。#1141 で
  `gemm_blis_variant_ab_4096` を新設したため 4096 も候補別直接比較値が揃っている
- **GB10・M4 Max の両方が出揃った結果（#1144 への引き継ぎ）**: いずれの実機でも
  `SharedB`・`SharedBPcOuter` は本番既定 `RowPanel` を大きく下回り不採用。次候補（B 側
  laneq のベクトル転置化・`vld1q_f32_x3` 経路の prefetch・KC 再スイープ）の要否は
  #1144（本番結線の要否判断）・親 #1117 で最終判断する。新規 issue 起票はユーザー承認
  事項のため本 issue では行わない（`.claude/rules/out-of-scope-tracking.md`。§7 参照）

## §7 採否（M4 Max。#1141）

- **結論: 非採用**（`SharedB`・`SharedBPcOuter` いずれも `RowPanel` を 1024/2048/4096 の
  全サイズで下回る。§5.2「結論」3 参照）
- **根拠**: 受け入れ条件（§1・§5 手順 5「1024/2048 の両方で gemm crate 以上、かつ 4096 で
  非劣化」）を満たす候補は確認できなかった。本番既定 `RowPanel` 自身が対 gemm crate 比
  0.838〜0.889（1024/2048/4096 のいずれも 1.0 未満）であり、`SharedB`・`SharedBPcOuter`
  はさらにその 65〜80% 程度のスループットしか出ない（§5.2 受け入れ条件 2）
- **GB10 側の結論（#1140・参照）との異同**: GB10 は 2048/4096 で `RowPanel` が既に gemm
  crate 以上（1.012 倍／1.449 倍）に達していたのに対し、M4 Max は 3 サイズいずれも
  gemm crate 未達（0.838〜0.889 倍）という違いがあるが、**「`SharedB`・
  `SharedBPcOuter` は不採用」という結論自体は両実機で完全に一致**する
- **次アクション**: 本番結線（#1144）は GB10・M4 Max とも不要（現行 `RowPanel` 既定を
  維持）。次候補（B 側 laneq のベクトル転置化・`vld1q_f32_x3` 経路の prefetch・KC
  再スイープ）の切り出しは、本イシューでは行わずユーザー承認を経て親 #1117・#1144 側で
  判断する（`.claude/rules/out-of-scope-tracking.md`）。M4 Max 側の対 gemm crate 比が
  1.0 未満のまま残っている点（RowPanel 自身の性能ギャップ）は、次候補の要否検討時に
  合わせて考慮する参考情報として記録する

## 出典

- イシュー #1041（本ドキュメントの起票元）・#1117（親 issue）・#1140（GB10 実機実測。本追補）・
  #1141（M4 Max 実機実測。本追補）・#1144（本番結線の要否判断）
- `docs/perf/oss-gemm-comparison-baseline.md` §7.2・§7.3（対 gemm crate 比較ベースライン）
- `docs/cpu-gemm-b-packing-sharing-decision.md`（B 共有化の設計判断・採用ゲート方針の前例）
- `crates/backend-cpu/src/gemm_blis/mod.rs`（`GemmDriverVariant`・
  `gemm_blis_shared_b_pc_outer_region`・`gemm_blis_parallel_variant`）
- `docs/perf/oss-comparison/2026-09-03/gb10-cpu-1140/`（GB10 実測の生データ。イシュー #1140）
- `docs/perf/oss-comparison/2026-09-03/gb10-cpu-1140-ossx5/`（OSS 直接比較の 5 独立プロセス
  実行分の生データ。codex-review P1 指摘対応の追補）
- `docs/perf/oss-comparison/2026-09-03/cpu-retune-1041/`（M4 Max 実測の生データ。イシュー
  #1141。単体テスト・A/B 5 回×2 ハーネス・OSS 比較 5 回・PyTorch CPU 計測ログを含む）
- `scripts/bench/gemm_bench_torch_cpu_f32.py`（PyTorch CPU f32 GEMM 計測スクリプト。
  イシュー #1141 新設）
- `.claude/rules/out-of-scope-tracking.md`（後続タスク起票のユーザー承認要件）
