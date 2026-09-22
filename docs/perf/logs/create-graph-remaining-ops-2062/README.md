# イシュー #2062: 高階微分（create_graph）残り Op 拡張の実機実測申し送り

## 背景

イシュー #2062（`Op::supports_create_graph()` を `ScalarUnary`・
`ScalarBinary`・`Transpose`・`Permute`・`Narrow`・`Concat`・
`Contiguous`・`Where` へ拡張。設計記録は
`docs/autodiff-higher-order-grad-decision.md` §15）は、新規カーネルを
一切追加せず既存の `Var` 公開メソッド（`add`／`mul`／`sub`／`div`／
`pow`／`where_cond`／`transpose`／`permute`／`narrow`／`cat`／
`contiguous`／各種 `scalar_unary` 系）の合成のみで実装した。このため
REQ-2 の統一複合判定は機構上そのまま非後退（新しい GPU カーネル経路は
存在しない）と判断しているが、Linux CI（GitHub ホステッド
`ubuntu-latest`）環境では CUDA・Metal 実機を利用できないため、CPU
バックエンド（`naive_ops`／`backend-cpu`）でのテストのみを実施済みで、
CUDA（DGX Spark GB10）・Metal 実機でのバックエンド固有 `BackendOps`
実装を経由した検証は未実施である。

## 実施手順（Mac／GB10 セッション向け）

```bash
# CPU（naive_ops。既存の統合テスト。Linux CI で実施済み）
cargo test -p fandhe-ai-autodiff --test create_graph

# Metal 実機（`crates/backend-metal` の BackendOps 実装を child/parent
# 双方の Tape に渡した経路で同じ有限差分突合が成立することを確認する。
# 既存テストは `naive_ops()` 固定のため、実機検証には
# `Tape::new_with_ops(fandhe_ai_backend_metal::ops())`（または相当）
# へ差し替えた別テスト・ベンチが必要。現時点では未整備。

# CUDA 実機（GB10）も同様に `fandhe_ai_backend_cuda::ops()` 相当へ
# 差し替えて検証する。
```

## 対象外（引き継ぎ）

- `MaskedFill`・`Gather`・`Scatter`・`Pad`・`MseLoss`・
  `CrossEntropyLoss` は本イシューのスコープ外（`Op::
  supports_create_graph()` は `false` のまま）。後続イシューで対応する。
- `ScalarUnaryOp::Gelu`（erf 版）・`GeluTanh` も対象外のまま。
