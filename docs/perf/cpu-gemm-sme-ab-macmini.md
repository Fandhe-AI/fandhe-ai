# Mac mini（Apple M4）での SME カーネル ON／OFF A/B（イシュー #3176）

## 状態

**実測済み。総合判定は ADOPT 候補**（`logs/sme-macmini-3176/aggregate.md` の機械判定をそのまま採用。事後の再解釈なし）。

- 計測コミット（before = main）: `1a9031537c54692b9bf143920ad72e8815c20a7e`（`macmini/rev-stamp.txt`・env_info の `head=`）
- 計測日: 2026-10-21（Mac mini・Apple M4・P4＋E6・macOS 27.0.1・rustc 1.99.0。`macmini/r1r2/env_info-3176-sme-3176-r1.txt`）
- 負荷ゲート（load1 < 3.0）: R4 5 run・長方形 5 run・A/B 5 round のすべて通過（`macmini/load_gate.log`・`macmini/r1r2/load_gate-3176-sme-cpu-3176-r1.log`）。正式系列
- PyTorch: torch 2.14.0・numpy 2.5.4（リポ外の venv。torch のスレッド数 4）
- P-TREE（差分は `mod.rs` の 1 件のみ・patch sha256 一致）・P-R0（両腕 `kernel_enabled: true`）は成立（`macmini/tree_verify.txt`・`macmini/sme_report.txt`）
- `SME_PRODUCTION_ENABLED` は main で `false` のまま（after 腕は Mac mini 上の計測専用ツリーのみ。ドリフトガード
  `sme_production_enabled_is_false_pending_measurement` が機械的に担保）。本番切替は #3230、しきい値の確定は #3231

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

値の出典は `logs/sme-macmini-3176/aggregate.md`（`aggregate.py` の出力）と `macmini/` の生ログ。M4 Max は `cpu-gemm-sme-fmopa-microkernel.md` §5.4。

### 1. R1（10 セル。after/before 中央値比）

| セル | 到達 | Mac mini 比 | round 別 | <=1.00 | checksum | M4 Max 比（§5.4.2） |
|---|---|---:|---|---|---|---:|
| gemm 512 fresh | 到達 | 0.5636 | 0.628 0.570 0.559 0.567 0.558 | 5/5 | 一致 | 0.7378 |
| gemm 512 reuse | 到達 | 0.5982 | 0.585 0.603 0.595 0.580 0.621 | 5/5 | 一致 | 0.7363 |
| gemm 1024 fresh | 到達 | 0.4687 | 0.463 0.464 0.467 0.473 0.482 | 5/5 | 一致 | 0.6005 |
| gemm 1024 reuse | 到達 | 0.4960 | 0.497 0.500 0.496 0.499 0.484 | 5/5 | 一致 | 0.6251 |
| gemm 2048 fresh | 到達 | 0.4723 | 0.475 0.467 0.473 0.473 0.470 | 5/5 | 一致 | 0.5725 |
| gemm 2048 reuse | 到達 | 0.4806 | 0.478 0.479 0.487 0.482 0.483 | 5/5 | 一致 | 0.5821 |
| train 64 fresh | 到達 | 0.9009 | 0.899 0.921 0.899 0.901 0.891 | 5/5 | 一致 | （本記録では未転記） |
| train 64 reuse | 到達 | 0.9117 | 0.909 0.912 0.923 0.901 0.937 | 5/5 | 一致 | （同上） |
| infer 64 fresh | 非到達 | 0.9821 | 0.948 1.008 0.969 0.993 0.984 | 4/5 | 一致 | （同上） |
| infer 64 reuse | 非到達 | 1.0100 | 1.019 1.020 1.007 0.978 1.010 | 1/5 | 一致 | （同上） |

- gemm の到達 6 セルは Mac mini の方が M4 Max より縮みが大きい（0.47〜0.60 倍 対 0.57〜0.74 倍）。
- train（size=64）は到達セルとして 0.90〜0.91 倍に縮み、5/5 で <=1.00。
- infer（size=64）は SME に届かない m=64 を含む非到達セルで、比は 0.98／1.01 と 1.0 近傍。非到達セルは判定に使わない（RULE の分類）。
  infer reuse の 4/5 round が 1.00 を超える点は、SME が効いていない経路の揺らぎと見られる（推定）。
- AC1 直接判定（到達セルの中央値 <=1.00 かつ 5/5、checksum 一致）: **成立**。全 10 セルで checksum 完全一致。

### 2. R4 格子（SME/NEON 中央値比。5/5 で >=1.0 の点に *）

| min(m,n) \ k | 32 | 64 | 128 | 256 |
|---|---|---|---|---|
| 64 | 1.144*（M4 Max 1.416） | 1.061（1.098） | 1.122*（1.132） | 1.223*（1.405） |
| 128 | 1.031（0.935） | 1.050（0.956） | 1.129*（1.018） | 1.257*（1.139） |
| 256 | 1.122（1.027） | 1.404*（1.161） | 1.607*（1.320） | 1.602*（1.291） |
| 512 | 1.325*（1.161） | 1.641*（1.310） | 1.930*（1.363） | 1.983*（1.616） |

括弧内は M4 Max（§5.4.1）の中央値。M4 Max では 5/5 を満たさない点が多かったが、Mac mini は 16 点中 12 点が 5/5 で >=1.0。

- パレート極小の採用候補 (min(m,n), k): (64, 128)・(256, 64)・(512, 32)。
- K∈{64,128,256} 別の参考領域（K>=Kmin かつ min(m,n)∈{256,512} の全点が SME>=NEON）: Kmin=64・128・256 のいずれも成立。
- 現行しきい値の領域（min>=256 かつ k>=64）は全点成立。M4 Max では境界点 (256, 64) が 5/5 を満たさなかったが、Mac mini では 1.404* で 5/5。
- しきい値は本イシューで変更しない（#3231 の前提）。

### 3. MLP 長方形 4 形状（記録のみ。判定に使わない）

| 形状 (m,n,k) | SME/NEON 中央値比 | 5 run 中 >=1.0 | 現行 `sme_shape_eligible` |
|---|---:|---|---|
| (64, 256, 784) | 1.326 | 5/5 | 否 |
| (784, 256, 64) | 1.672 | 5/5 | 可 |
| (784, 64, 256) | 1.582 | 5/5 | 否 |
| (64, 10, 256) | 0.939 | 2/5 | 否 |

- 現行しきい値が可を返すのは (784,256,64) のみ。ほか 2 形状（(64,256,784)・(784,64,256)）は SME が 1.3〜1.6 倍速いが現行しきい値では NEON に回る。
  (64,10,256) だけは SME が遅い（0.939）。しきい値の見直しは #3231。

### 4. PyTorch 比（同じ round 内の torch median ÷ fandhe median の 5 round 中央値。1.0 超で fandhe が速い）

| セル | before | after |
|---|---:|---:|
| gemm 512 fresh | 0.737 | 1.307 |
| gemm 512 reuse | 0.742 | 1.251 |
| gemm 1024 fresh | 0.650 | 1.406 |
| gemm 1024 reuse | 0.619 | 1.246 |
| gemm 2048 fresh | 0.525 | 1.109 |
| gemm 2048 reuse | 0.494 | 1.030 |
| train 64 fresh | 0.189 | 0.211 |
| train 64 reuse | 0.197 | 0.215 |
| infer 64 fresh | 0.159 | 0.161 |
| infer 64 reuse | 0.160 | 0.160 |

- gemm は after で全 6 セルが PyTorch（Accelerate）を上回る（1.03〜1.41）。特に 2048 reuse は 1.030 で僅差。
- 0.11.0 系列 B の reuse 比 0.89／0.69 は**別 run の参考値**（本計測の before 側とは機体の状態・run が異なる）。
- train／infer は PyTorch に大差で届かない（0.16〜0.22）。SME の効果は train で 0.90 倍止まり、infer ではほぼ無い。
  原因の切り分けは本計測では行わない。

### 5. 総合判定

**ADOPT 候補**（本番切替は #3230、しきい値は #3231。本イシューでは切り替えない）。AC1 直接判定は成立。
前提ゲートの不成立理由はなし。

## #3175 を踏まえた解釈の枠

`cpu-sme-thread-scaling-macmini.md`（#3175）では、(2048,2048,1024) の SME は T=10 まで伸び（S(10)=3.07）、頭打ちは見えなかった。
一方で第三者ピークの 54% 止まりで、micro-call の固定費は 44.5〜89.8 ns だった。したがって本計測の after で相手に届かない場合、
その原因を「スレッド頭打ち」とは言えない。固定費や多タイル化の不足を疑う余地は残るが、本計測だけでは切り分けない（推定）。
実測では gemm の after が PyTorch を 1.03〜1.41 倍上回ったため、gemm では相手の水準に届いた。届かなかったのは train／infer（0.16〜0.22 倍）で、
SME に届かない小形状（m=64 など）が支配的なためと見られる（推定。本計測では切り分けていない）。

## スコープ外

- `SME_PRODUCTION_ENABLED` の本番化と unsafe 監査記録の更新（#3230）、しきい値の確定（#3231）、#2118 の M4 Max・GB10 実測。
- `run_ab_sme_cpu.sh` の既知の弱点（jq 依存、`RULE_LOAD_GATE=8.0` 固定、compare の exit 3 が後退と checksum 不一致を区別しない、行数検査なし）の改修。
  #2118 が将来実行されるときに影響するため、別イシューの候補。
- MLP 長方形の R4 格子への組み込み（しきい値の規則変更にあたる）。
- 多タイル化・streaming mode の維持（#3234・#3238）、E コアの分離計測、GB10（SME 非対応）。
