# Mac mini（Apple M4）での SME カーネル ON／OFF A/B（イシュー #3176）

## 状態

**計測基盤と事前登録規則まで完了。実測は未実施（判定 undetermined）**。本 PR を作った実行環境から Mac mini へ到達していないため、
値は推定で埋めていない。実測後に「結果」節を埋める。`SME_PRODUCTION_ENABLED` は `false` のまま（ドリフトガード
`sme_production_enabled_is_false_pending_measurement` が機械的に担保）。本番切替は #3230、しきい値の確定は #3231。

## 目的

- Mac mini で SME を ON にすると、相手（PyTorch＝Accelerate）の水準にどこまで近づくかを、同じ run 内の比で示す。
- Mac mini 上の R4 格子と MLP 長方形 4 形状で、SME と NEON のどちらが速いかを示す。
- Mac mini と M4 Max の倍率を機体別に並べる。

M4 Max の実測（`cpu-gemm-sme-fmopa-microkernel.md` §5.4）では、gemm 到達セルが after/before 0.57〜0.74 倍に縮み、checksum も一致した。
Mac mini は SME ユニットの構成（P クラスタに 1 基）と P4＋E6 のコア構成が異なるため、別に測る必要がある。

## 方法（規則の正は `logs/sme-macmini-3176/RULE.txt`）

- 腕: before = main、after = `SME_PRODUCTION_ENABLED` 1 行のみ true にした計測専用ツリー（`on-arm.patch`）。
- A/B: `scripts/bench/framework-compare/run_ab_sme_macmini_cpu.sh`。5 round・起動順を round ごとに反転・同じ round 内で PyTorch も測る。
  `run_ab_sme_cpu.sh`（#1978／#2118）は未実測の事前登録記録が依存するため変更せず、Mac mini 派生版を新設した（理由は RULE §1）。
- セル: gemm 512／1024／2048 と train／infer（size=64）の fresh／reuse の 10 セル。到達は 8、非到達は infer の 2。
- R4 格子: 既存 `sme_vs_neon_ab_r4_grid` を 5 プロセス独立に起動。
- 長方形 4 形状: 新規 `sme_vs_neon_ab_mlp_rect`（(64,256,784)・(784,256,64)・(784,64,256)・(64,10,256)）。SME と NEON の bit 一致を計時前に assert する。
  現行 `sme_shape_eligible` が可を返すのは (784,256,64) のみ。
- 前提ゲート: P-TREE（差分が `mod.rs` の 1 件だけ）・P-R0（両腕で `kernel_enabled: true`）ほか。実行時のディスパッチ報告 API は無いため、この指紋検査を発火の代わりの証拠とする。
- 負荷ゲート: load1 < 3.0。5/5 が通過した系列のみ正式。

## 結果

未実測（`logs/sme-macmini-3176/macmini/` 未生成）。実測後に次の表を埋める。

1. R1 の 10 セル表（after/before 中央値比・round 別・5/5・checksum）と、M4 Max（§5.4.2）の倍率を機体別に並べた表
2. R4 の 16 点表（Mac mini と M4 Max §5.4.1 の中央値を併記）・パレート候補・K 別の参考領域
3. 長方形 4 形状の SME／NEON 比
4. PyTorch 比（before→after で何倍の差が何倍になったか。0.11.0 系列 B の reuse 比 0.89／0.69 を参考値として併記）
5. 総合判定と AC1 直接判定欄

## #3175 を踏まえた解釈の枠

`cpu-sme-thread-scaling-macmini.md`（#3175）では、(2048,2048,1024) の SME は T=10 まで伸び（S(10)=3.07）、頭打ちは見えなかった。
一方で第三者ピークの 54% 止まりで、micro-call の固定費は 44.5〜89.8 ns だった。したがって本計測の after で相手に届かない場合、
その原因を「スレッド頭打ち」とは言えない。固定費や多タイル化の不足を疑う余地は残るが、本計測だけでは切り分けない（推定）。

## スコープ外

- `SME_PRODUCTION_ENABLED` の本番化と unsafe 監査記録の更新（#3230）、しきい値の確定（#3231）、#2118 の M4 Max・GB10 実測。
- `run_ab_sme_cpu.sh` の既知の弱点（jq 依存、`RULE_LOAD_GATE=8.0` 固定、compare の exit 3 が後退と checksum 不一致を区別しない、行数検査なし）の改修。
  #2118 が将来実行されるときに影響するため、別イシューの候補。
- MLP 長方形の R4 格子への組み込み（しきい値の規則変更にあたる）。
- 多タイル化・streaming mode の維持（#3234・#3238）、E コアの分離計測、GB10（SME 非対応）。
