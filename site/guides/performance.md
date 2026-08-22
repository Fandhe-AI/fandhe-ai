# 性能の考え方

## REQ-8 段階的下限

`rust-ai-library` の性能要件（REQ-8）は「一発で理想的な性能を出す」
ことを目標にせず、**段階的な下限**を確定させながら積み上げる方針を
取っています。全バックエンド横断の下限一覧は
[`docs/performance-targets.md`](https://github.com/Fandhe-AI/rust-ai-library/blob/main/docs/performance-targets.md)
にまとまっています。GEMM 最適化の実測記録（CPU・CUDA・Metal 各
バックエンドのボトルネック診断・OSS 直接比較）は `docs/perf/` 配下に
蓄積されています。

## カーネル境界検査を省略しない

**性能下限・最適化の達成を理由に、シェーダ・カーネル側の手動境界
チェックを省略しません。** ベクトル化ロード・タイル端の分岐削減等、
境界検査を無効化しうる最適化を適用する場合でも、シェーダ側で手動境界
チェックを維持したうえで行う方針です。この規約は CPU（intrinsics）・
CUDA（NVRTC/mma）・Metal（simdgroup）の全カーネルに適用されます。
性能のための近道が境界外アクセス（メモリ安全性の欠陥）を生む余地を
構造的に排除する狙いです。

## 計測規約: 5 回計測の中央値

ベンチ計測は**5 回計測の中央値**を採用します。単発計測は外れ値
（OS のスケジューリング揺らぎ・初回のキャッシュコールドスタート等）の
影響を受けやすく、中央値を取ることでその影響を緩和します。
[GEMM ベンチ example](/examples/gemm-bench/)もこの規約に沿って
ウォームアップ 1 回＋ 5 回計測の中央値を計算する構成にしています。

学習系の回帰テストには決定的シード設定ユーティリティ
（`bench_harness::rng::Xorshift64Star` 等）を使い、実行のたびに結果が
揺れないようにします。[training-loop example](/examples/training-loop/)
も同じ理由で重み初期化・データ生成の双方を固定シードで駆動しています。

## 本格計測は `bench-harness`（criterion）の領分

`std::time::Instant` による簡易計測（[GEMM ベンチ example](/examples/gemm-bench/)
がその最小デモです）は、コードがどう動くかを手早く確認する用途には
向きますが、性能下限（REQ-8）判定・性能回帰検出を目的とした本格計測
ではありません。本格計測は `criterion`（`dev-dependencies` 限定）を
使う `bench-harness` クレートの領分です。両者を混同しないよう、
example 側にも同じ注記を残しています。
