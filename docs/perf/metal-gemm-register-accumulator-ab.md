# Metal GEMM レジスタ常駐アキュムレータ構造の検証・出力タイル拡大 A/B 計測記録（#745）

イシュー #745「perf(backend-metal): レジスタ常駐アキュムレータ構造の検証と出力タイル拡大（AI 頭打ち対策）」の
構造診断確定記録・A/B 計測手順・記録テンプレート・判断基準。`docs/perf/metal-gemm-bottleneck-diagnosis.md`（#487）が
報告した「`gemm_simdgroup_tiled` の arithmetic intensity（AI）が size に依らず約 15 FLOP/byte で頭打ち」に対し、
(a) K ループ内のレジスタ常駐構造が MLX steel classic 相当かの診断・是正、(b) 出力タイル（TM×TN）拡大の効果計測、
の 2 点を扱う。

## 1. 背景

- 基準系列は MLX classic の f32 大型 GEMM パラメータ **BM64/BN64/BK16・WM2×WN2**。現行 `crates/backend-metal/src/tile.rs`
  の `CANDIDATES[0]`（`bm:64, bn:64, bk:16, wm:2, wn:2, staged:true`）と一致し、SMEM は pad=4 込みで
  `(64*20 + 16*68)*4 = 9,472` バイト（≈ 9.25KiB）。基準系列は既に本番選択構成である（`crate::tile::select` の
  大形状・正方分岐）。
- `docs/perf/metal-gemm-bottleneck-diagnosis.md` §3.2 の実測（下表転記）: AI が 15.06〜15.88 FLOP/byte で
  size に依らずほぼ一定。

  | size | tile (bm×bn×bk, wm×wn) | arithmetic_intensity (FLOP/byte) |
  |---|---|---|
  | 512  | 64×64×16, 2×2 | 15.0588 |
  | 1024 | 64×64×16, 2×2 | 15.5152 |
  | 2048 | 64×64×16, 2×2 | 15.7538 |
  | 4096 | 64×64×16, 2×2 | 15.8760 |

## 2. 構造診断（受け入れ条件 1。Linux 側で完結）

`crates/backend-metal/src/shaders/gemm.metal` の `gemm_simdgroup_tiled`（staged 経路）を精査した結果:

| 診断観点 | 是正前の現状 | MLX classic（`steel/gemm/mma.h` 型の構造） | 差分 |
|---|---|---|---|
| アキュムレータ確保位置 | `simdgroup_float8x8 acc[MAX_ACC][MAX_ACC]` を K タイルループ**外**で確保・ゼロ初期化 | 同じ | **なし**（既に K 全域レジスタ常駐） |
| K 反復中の device 直読 | staged 経路の MMA フェーズは threadgroup メモリからの `simdgroup_load` のみ | 同じ | **なし** |
| barrier 配置 | K タイルごとに 2 回（協調ロード後・MMA 完了後） | 同じ | **なし** |
| **K 内側（kk）反復のフラグメントロード構造** | `r` ループ内で `a_tile` を 1 回ロード後、`(r, ci)` の**内側で毎回 `b_tile` を再ロード**。1 kk ステップあたり TM + TM×TN 回（TM=TN=4 で 4+16=20 回） | A フラグメント TM 個・B フラグメント TN 個を**先にレジスタ配列へロードしてから** TM×TN 回の外積 MMA を発行。1 kk ステップあたり TM + TN 回（4+4=8 回） | **あり（是正対象）** |

つまり「アキュムレータのレジスタ常駐」自体は是正前から成立しており、是正すべき差分は「B（および A）フラグメントの
kk ステップ内レジスタ常駐化（ロードの巻き上げ。20→8 ロード/kk）」であった。

### 是正内容（実装済み）

`gemm_simdgroup_tiled` の staged 経路の kk 内側ループを、MLX steel `mma.h` 型（フラグメント配列を kk ステップ先頭で
一括ロードしてからレジスタ常駐のまま TM×TN の外積 MMA を発行する構造）へ再構成した:

```
simdgroup_float8x8 a_frag[MAX_ACC];
simdgroup_float8x8 b_frag[MAX_ACC];
for (uint r = 0; r < acc_rows; r++) {
    simdgroup_load(a_frag[r], ...);
}
for (uint c_ = 0; c_ < acc_cols; c_++) {
    simdgroup_load(b_frag[c_], ...);
}
for (uint r = 0; r < acc_rows; r++) {
    for (uint c_ = 0; c_ < acc_cols; c_++) {
        simdgroup_multiply_accumulate(acc[r][c_], a_frag[r], b_frag[c_], acc[r][c_]);
    }
}
```

旧構造が採っていた蛇行（serpentine）走査（イシュー #536。`c_ = (r % 2 == 1) ? (acc_cols - 1 - ci) : ci`）は
「`b_tile` を `(r, ci)` の内側で毎回再ロードしていたためアクセス局所性が問題になっていた」ことが前提だったが、
本巻き上げによりロード自体が kk ステップ先頭の 1 回にまとまるため、その前提が構造的に消滅する。よって staged 経路の
蛇行走査は撤去し MMA 発行順を行優先へ戻した（direct-load 経路は引き続きフラグメント再ロードが残る構造のため
蛇行走査を維持。`crates/backend-metal/tests/shader_source_evidence.rs` の
`gemm_simdgroup_tiled_source_uses_serpentine_scan_order` が出現数 1 = direct-load 経路のみへロック済み）。

`acc[r][c_]` ごとの K 方向累算オペランド列（値・順序）はロードスケジューリングを変えても不変のため、結果は
ビット単位で従来と一致する（#536・#538 と同じ論法。tolerance には触れていない）。

### 機械検査（Linux CI）

`crates/backend-metal/tests/shader_source_evidence.rs`:

- `gemm_simdgroup_tiled_source_uses_register_resident_fragment_arrays`: `a_frag`/`b_frag` 配列宣言・分離した
  ロードループ・`a_frag[r]`/`b_frag[c_]` を直接引数に取る `simdgroup_multiply_accumulate` 呼び出しの実在をロック
- `gemm_simdgroup_tiled_source_uses_serpentine_scan_order`: 蛇行走査式の出現数が 1（direct-load 経路のみ）へ
  変わったことをロック
- `gemm_simdgroup_tiled_source_uses_tgp_padding_stride`: フラグメント変数名変更後も TGP_PAD 込みストライド
  （`lda`/`ldb`）でのロードが維持されていることを再確認

## 3. 出力タイル（TM×TN）拡大の A/B 候補系列（受け入れ条件 2・3。実機セッションで消化）

すべて `TileConfig::validate` の制約（`bm % (wm*8) == 0`・`bn % (wn*8) == 0`・`bk % 8 == 0`・
`acc_rows`/`acc_cols` ≤ `MAX_ACC`=8・スレッド数 ≤ 1024・SMEM ≤ 32KiB）を満たすことを事前確認済み。

| ラベル | bm/bn/bk/wm/wn | TM×TN | SMEM（pad=4 込み） | AI @2048 | AI @4096 | 位置づけ |
|---|---|---|---|---|---|---|
| baseline（`bm64_bn64_bk16_staged`） | 64/64/16/2/2 staged | 4×4 | 9,472 B | 15.75 | 15.88 | MLX classic 基準系列（現行 `CANDIDATES[0]`） |
| `tm8_tn4_bm128_bn64_bk16_staged` | 128/64/16/2/2 staged | 8×4 | 14,592 B | ≈23.6 | ≈23.9 | 行方向拡大 |
| `tm4_tn8_bm64_bn128_bk16_staged` | 64/128/16/2/2 staged | 4×8 | 13,568 B | ≈23.6 | ≈23.9 | 列方向拡大 |
| `tm8_tn8_bm128_bn128_bk16_staged` | 128/128/16/2/2 staged | 8×8 | 18,688 B | ≈31.3 | ≈31.5 | 両方向拡大（`MAX_ACC` 上限） |

AI 列は `crates/backend-metal/examples/gemm_diagnosis.rs::analytics::analyze` と同一式
（`flops / (load_bytes_total + store_bytes_total)`。`load_bytes_total = actual_groups * k_tile_count *
(bm*bk + bk*bn) * 4`）で `size=2048`/`4096` について手計算した参考値（`tile::select` を経由しない構成のため
`analyze` 関数自体は直接使えず、本ドキュメント作成時に同式で算出した近似値。実機ベンチとは独立の理論値であり
性能の主張ではない）。

レジスタ圧迫（acc 64 個 + フラグメント 16 個 @8×8）による occupancy 低下が拡大のリスクであり、これは実機 A/B で
確定させる事項である。

## 4. 計測手順（Apple Silicon 実機）

`docs/real-hardware-verification-env.md` の接続・転送手順に従う（実ホスト名はローカル管理外ファイル参照）。

```sh
git fetch origin

# base（是正前 main）
git checkout main
cargo run -p backend-metal --example gemm_bench --release > /tmp/gemm_bench_745_base.txt

# head（本イシューの実装ブランチ）
git checkout perf/745-metal-register-accumulator-tile
cargo run -p backend-metal --example gemm_bench --release > /tmp/gemm_bench_745_head.txt
```

`examples/gemm_bench.rs` の「候補構成の明示比較」節（size ∈ {2048, 4096}）が baseline・拡大候補 3 種すべてを
`size=<N> candidate=<label> tflops=<T> requested=(...) resolved=(...) resolved_matches_requested=<bool>` 形式で
出力する。`resolved_matches_requested=false` の行は `pipeline_for_tile`（デバイス上限超過等）がフォールバックした
ことを意味し、その候補は意図した構成では計測できていないため判断基準から除外する。

### 数値一致確認（採否判断より前に必須）

```sh
cargo test -p backend-metal --release -- --ignored --nocapture
```

`gemm_dynamic_tile_parity`・`cpu_metal_parity`・`cpu_metal_f16_parity` 等が green であること（tolerance は
変更しない。`.claude/rules/coding-rust.md`）。

### REQ-8 下限非後退確認（タイル拡大採用時のみ）

```sh
# docs/performance-targets.md・docs/perf/metal-floor-remeasurement.md の計測系列に従う
```

## 5. 判断基準

- **構造是正（フラグメントレジスタ常駐化）**: base に対し head の候補比較（baseline 構成 `bm64_bn64_bk16_staged`）
  の中央値 TFLOPS が改善していれば「採用」（変更は既にコミット済みのため、この場合は追加対応不要）。改善が
  確認できなければ、構造是正自体は revert PR で撤去し、判断・実測値を本ドキュメントへ記録する（変更は staged
  経路の MMA フェーズに閉じているため revert は最小差分で可能）
- **出力タイル拡大**: 拡大候補（`tm8_tn4`・`tm4_tn8`・`tm8_tn8`）のいずれかが baseline 比で中央値 TFLOPS 改善を
  示し、かつ `resolved_matches_requested=true`（フォールバックなし）であれば採用候補とする。採用時は
  `crates/backend-metal/src/tile.rs` の `CANDIDATES`/`select` へ組み込む変更を**別コミット**で行い、REQ-8 下限
  非後退（Metal f32 対 MPS 10%・Metal f16 対 MPS f16 15%）を再実測確認してから確定する。改善がない、または
  `resolved_matches_requested=false`（フォールバック発生）の場合は不採用とし、その判断と実測値を本ドキュメントへ
  記録する（`examples/gemm_bench.rs` の当該候補・本ドキュメントは残置してよい。bench 候補としての記録は有用な
  ため撤去は必須としない）

## 6. 実測結果

（未計測。実機セッションで本節へ追記する。base/head の候補ごと `size=2048`/`4096` の中央値 TFLOPS・
`resolved_matches_requested`・数値一致確認結果・REQ-8 下限確認結果〈拡大採用時のみ〉を記録する）
