# Metal `tile::select` 正方大形状候補是正（#744）

イシュー #744「fix(backend-metal): tile::select の候補選択是正（2048 で最良候補比 2.8 倍の逸失を実測）」の判断根拠・実測記録。

## 状態: ロジック是正・単体テスト・docs 記録は完了。**実機確認（受け入れ条件 1〜3）は未計測（Mac 実機セッションで実施）**

本 PR はロジック是正・Linux で回る単体テスト更新・本記録の作成までを行う。是正の妥当性は
2026-08-19 の M4 Max 実機実測（親イシュー #737 のセッション実測。イシュー #744 本文記載）に基づく。
本 PR 自体（Linux worktree・Metal 実機なし）では新規の実機計測は行わない。実機確認結果は
「実機確認結果（記入欄）」節へ Mac 実機セッションで追記する。

## 背景: 是正前の選択ロジックと実測の乖離

`crates/backend-metal/src/tile.rs::select_with_occupancy` の段 1（形状判定）は、是正前は次の分岐だった:

```rust
let large = m >= LARGE /* 512 */ && n >= LARGE;
let shape_cfg = match (large, tall, wide) {
    (_, true, _) => CANDIDATES[1], // 64x32（縦長）
    (_, _, true) => CANDIDATES[2], // 32x64（横長）
    (true, _, _) => CANDIDATES[0], // 64x64（大形状・正方）
    _ => CANDIDATES[3],            // 32x32（中形状・正方）
};
```

正方形状（縦長・横長どちらにも該当しない）かつ `m, n >= 512` のとき `CANDIDATES[0]`（64x64/bk16/staged）を
選んでいた。この閾値は #188 導入当時の `docs/perf/metal-gemm-dynamic-tile.md`（#381 計測）で
`size=2048` の 64x64 staged が 2.36 TFLOPS・32x32 staged が 2.40 TFLOPS とほぼ同等だったことに基づく
一応の根拠があった。

その後、staged 経路に対する複数の変更（#533 float4 協調ロード・#538 TGP パディング・#572 prepared
境界確立など）を経た現 main では、この関係が大きく逆転している。

## 実測（2026-08-19・M4 Max・5 回中央値。イシュー #744/#737 記載）

`size=2048` の候補別 TFLOPS:

| 候補 | TFLOPS |
|------|--------|
| 32x32/bk16/staged（`CANDIDATES[3]`） | ≈ 3.31 |
| 32x32/bk16/direct | ≈ 1.99 |
| 64x64/bk16/staged（`CANDIDATES[0]`。是正前の自動選択結果） | ≈ 1.18 |

自動選択（`tile::select` → `dynamic-tile-auto`）は `size=2048` で ≈ 1.20 TFLOPS と最下位相当（64x64
staged）を選んでおり、最良候補（32x32 staged ≈ 3.31）比で約 2.8 倍の性能逸失があった。さらに
512〜4096 の正方形状全帯域で自動選択は単一 simdgroup カーネル `gemm_simdgroup`（`size=2048` で
≈ 1.67 TFLOPS）をも下回っていた。

## 判断式（是正後）

正方形状（`SMALL`（64）以上・縦長・横長のいずれにも該当しない形状）は、サイズによらず一律
`CANDIDATES[3]`（32x32/bk16/wm2/wn2/staged）を返す:

```rust
let shape_cfg = match (tall, wide) {
    (true, _) => CANDIDATES[1], // 64x32（縦長）
    (_, true) => CANDIDATES[2], // 32x64（横長）
    _ => CANDIDATES[3],         // 32x32（正方。#744 是正後の一律選択）
};
```

- **`LARGE`（512）閾値・「正方大形状 → `CANDIDATES[0]`」分岐は撤去した**。実測のある全サイズ
  （512/1024/2048/4096）で `CANDIDATES[3]` が一貫して最良のため、この帯域でサイズ別テーブルを
  持つ必要がない（サイズ帯条件分岐が必要になった場合の再導入は #747 のスコープ）
- **候補追加は行っていない**: 実測済みの 32x32 direct（≈1.99）は staged（≈3.31）に劣後しており、
  追加すべき優位候補が実測上存在しない
- **縦長・横長の分岐は変更していない**。2026-08-19 実測は正方形状のみを対象としており、非正方の
  実測根拠なしに変更するのは安全側の判断に反する
- **`CANDIDATES[0]`（64x64 staged）自体は候補配列から削除していない**。`select` の添字依存
  （`CANDIDATES[0..=3]` の並び順・個数固定）・縦長/横長経路の occupancy 縮退 fail-safe 比較・
  `fallback_chain`・#747 での再利用のため維持する。`select` の形状判定からは選ばれなくなった
  だけである

## 影響範囲

- `crate::gemm::MetalGemm::dispatch_auto`（本番ディスパッチ経路）が `tile::select` を使用するため、
  正方形状の自動ディスパッチが直接改善される
- `examples/gemm_f32_prepared_bench.rs`（floor bench）も `tile::select` を使用するため、REQ-8
  MetalF32 10% 下限に対する比率の改善に直結する（下限値の正式再判定は本 PR のスコープ外。
  `docs/perf/performance-floor-decision.md` 参照）
- `select_with_occupancy(..., Some(params))`（occupancy 縮退込み経路。イシュー #542。現時点で
  `dispatch_auto` の入口ではない）は、段 1 が正方形状に対し大タイル系（`CANDIDATES[0..=2]`）を
  返さなくなった結果、縮退判定の対象が実質的に縦長・横長のみになる（自然な帰結。#747 のスコープで
  再検討しうる）

## 計測ノイズに関する注意

イシュー #746（ベンチ計測ノイズ対策）の実測によれば `size=256`/`512` は対照カーネルで最大 70% の
変動があるのに対し `size=2048` は 2〜4% と小さい。本是正の根拠とした `size=2048` の最良候補比 2.8
倍差はこのノイズ床を大きく上回るため、判定材料として十分である。

## 再逆転時の再計測手順

もしカーネル側の変更（アキュムレータ構造・出力タイル拡大等。#745）で 64x64 staged が再度優位に
転じた場合は、`examples/gemm_bench.rs` の候補比較セクション（`size=<N> candidate=<label>` 行）で
512/1024/2048/4096 を再計測し、本ファイルと `crates/backend-metal/src/tile.rs` の判断式を追補
修正する。

## 実機確認結果（記入欄）

以下は Mac 実機セッション（M4 Max）で記入する。実行コマンドは各節見出しに記載。

### 数値一致（受け入れ条件 3）

```sh
cargo test -p backend-metal --release -- --ignored
```

結果: （未計測）

### 候補・自動選択ベンチ（受け入れ条件 1・2）

```sh
cargo run -p backend-metal --release --example gemm_bench
```

| size | 是正前 auto TFLOPS | 是正後 auto TFLOPS | 候補中最良 TFLOPS | 是正後 auto/最良 |
|------|--------------------|--------------------|--------------------|-------------------|
| 512  | （未計測） | （未計測） | （未計測） | （未計測） |
| 1024 | （未計測） | （未計測） | （未計測） | （未計測） |
| 2048 | （未計測） | （未計測） | （未計測） | （未計測） |
| 4096 | （未計測） | （未計測） | （未計測） | （未計測） |

- 受け入れ条件 1（2048/4096 で最良候補比 ±ノイズ範囲）: （未計測）
- 受け入れ条件 2（512/1024 の劣化中央値 5% 以内）: （未計測）

### floor bench 参考記録

```sh
cargo run -p backend-metal --release --example gemm_f32_prepared_bench
```

結果: （未計測。REQ-8 下限判定の正式更新は別スコープ）
