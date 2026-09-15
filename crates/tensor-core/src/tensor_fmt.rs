//! `Tensor<T>` の値表示（`Debug`／`Display`）実装。
//!
//! `tensor.rs` の `#[derive(Debug)]` は `shape`/`strides`/`offset`/
//! `Storage { len }` の構造情報のみを出力し値が見えないため、facade
//! 利用者（`fandhe_ai::Tensor` は `pub use` の再エクスポート経由で本
//! モジュールの実装へそのまま到達する。facade 側に新規公開アイテムは
//! 追加しない）向けに PyTorch の `print(t)` 相当の debuggability を
//! 補うモジュール（イシュー #1754）。
//!
//! `Tape: Debug` は既存の公開契約（`docs/public-api-design.md` §7）で
//! あり、`Tape`/`Var`/`Op` は多数の `Tensor` を保持しうるため、値の
//! 出力は必ず要素数上限つきで打ち切る（下記 [`FMT_MAX_ELEMS`]）。
//! これにより `format!("{:?}", tape)` のような呼び出しが巨大テンソル
//! 保持時でも出力サイズが入力サイズに比例して無限増加しない
//! （`.claude/rules/security.md` A04 DoS 観点）。
//!
//! 値の走査は `Tensor::get(&[usize])` による論理インデックス順（stride
//! 走査）で行う。これにより `transpose`/`narrow`/`broadcast_to`
//! （stride 0）等の view も、対応する `contiguous()` と同一の値列で
//! 表示され、表示のためだけの中間バッファを確保しない。

use crate::element::Element;
use crate::tensor::Tensor;

/// 打ち切りを発動する要素数の上限（この値を超えると各軸を省略表示する）。
/// PyTorch `torch.set_printoptions()` の既定値（`threshold=1000`）に
/// 合わせた値（無限に大きい既定を避けデフォルトで DoS 耐性を持たせる）。
const FMT_MAX_ELEMS: usize = 1000;

/// 打ち切り時に各軸の先頭・末尾に残す要素数。
/// PyTorch `torch.set_printoptions()` の既定値（`edgeitems=3`）と同じ。
const FMT_EDGE_ITEMS: usize = 3;

/// 現在の走査位置（多次元インデックス）から `Tensor` の指定軸以降を
/// 再帰的にレンダリングする。
///
/// `axis == shape.len()`（末端。スカラー要素 1 個）で `T::fmt` へ委譲し、
/// それ以外の軸では `[` `]` の入れ子と `, ` 区切りで descend する。
/// `truncate` は `numel() > FMT_MAX_ELEMS` のときに `true` となり、各軸の
/// 長さが `2 * FMT_EDGE_ITEMS` を超える場合に先頭・末尾のみ表示し中間を
/// `...` で省略する（該当軸の子孫がすべて省略されるため、打ち切りは
/// 出力サイズを軸ごとに独立して抑える）。
fn render_axis<T>(
    tensor: &Tensor<T>,
    index: &mut Vec<usize>,
    axis: usize,
    truncate: bool,
    f: &mut std::fmt::Formatter<'_>,
) -> std::fmt::Result
where
    T: Element,
{
    let shape = tensor.shape();
    if axis == shape.len() {
        // 末端: 多次元インデックスが 1 要素を指す。`Tensor::get` は
        // 論理インデックス境界内なら常に `Some` を返す契約だが、
        // 呼び出し元がその契約を破って到達不能な状態になった場合でも
        // 本番経路 panic を避けるため `?` を出力する
        // （`.claude/rules/coding-rust.md` の panic 禁止方針）。
        return match tensor.get(index) {
            Some(v) => std::fmt::Debug::fmt(&FmtElem(v), f),
            None => f.write_str("?"),
        };
    }

    let len = shape[axis];
    f.write_str("[")?;
    if truncate && len > 2 * FMT_EDGE_ITEMS {
        for i in 0..FMT_EDGE_ITEMS {
            if i > 0 {
                f.write_str(", ")?;
            }
            index.push(i);
            render_axis(tensor, index, axis + 1, truncate, f)?;
            index.pop();
        }
        f.write_str(", ..., ")?;
        for i in (len - FMT_EDGE_ITEMS)..len {
            index.push(i);
            render_axis(tensor, index, axis + 1, truncate, f)?;
            index.pop();
            if i + 1 < len {
                f.write_str(", ")?;
            }
        }
    } else {
        for i in 0..len {
            if i > 0 {
                f.write_str(", ")?;
            }
            index.push(i);
            render_axis(tensor, index, axis + 1, truncate, f)?;
            index.pop();
        }
    }
    f.write_str("]")
}

/// 要素 1 個の `Debug` 出力を、呼び出し元 `Formatter` の precision/flags
/// を伝播させたまま行うための薄いラッパー。`T::fmt` へそのまま委譲する
/// ことで `{:.3}` 等の書式指定が要素へ届く（PyTorch 風の自動桁揃えは
/// 対象外。決定的・有界な出力を優先する設計判断は issue #1754 実装計画
/// §3.2 参照）。
struct FmtElem<T>(T);

impl<T: std::fmt::Debug> std::fmt::Debug for FmtElem<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        std::fmt::Debug::fmt(&self.0, f)
    }
}

impl<T: std::fmt::Display> std::fmt::Display for FmtElem<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        std::fmt::Display::fmt(&self.0, f)
    }
}

/// `data` フィールドへ差し込む値プレビュー（打ち切り付き）の `Debug`
/// 実装。`Tensor` 本体の `Debug` から `f.debug_struct` 経由で使われる。
struct DataPreview<'a, T: Element>(&'a Tensor<T>);

impl<'a, T: Element> std::fmt::Debug for DataPreview<'a, T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let tensor = self.0;
        let truncate = tensor.numel() > FMT_MAX_ELEMS;
        if tensor.rank() == 0 {
            // rank 0（`Tensor::scalar`）は括弧なしの単一値。
            return match tensor.get(&[]) {
                Some(v) => std::fmt::Debug::fmt(&FmtElem(v), f),
                None => f.write_str("?"),
            };
        }
        let mut index = Vec::with_capacity(tensor.rank());
        render_axis(tensor, &mut index, 0, truncate, f)
    }
}

/// `Tensor<T>` の手書き `Debug` 実装。
///
/// 構造情報（`shape`/`strides`/`offset`/`storage_len`）を
/// `f.debug_struct` で維持しつつ、`data` フィールドに打ち切り付きの
/// 値プレビューを追加する。`{:#?}`（pretty-print）でも
/// `f.debug_struct` 経由のため自然に整形される。
///
/// `Element: Debug` は既存のトレイト境界（`crate::element::Element`）
/// のみを使い、新規境界は追加しない（`Element` は unsealed 公開 trait
/// のため必須メソッド追加は破壊的変更になる。既存の `Debug` 境界内で
/// 完結させる設計判断は issue #1754 実装計画 §3.2 参照）。
impl<T: Element> std::fmt::Debug for Tensor<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Tensor")
            .field("shape", &self.shape())
            .field("strides", &self.strides())
            .field("offset", &self.offset())
            .field("storage_len", &self.storage_len())
            .field("data", &DataPreview(self))
            .finish()
    }
}

/// `Tensor<T>` の `Display` 実装（`Element: Display` を持つ型限定）。
///
/// PyTorch の `print(t)` に寄せた `tensor([[1, 2], [3, 4]])` 形式。
/// rank 0 は `tensor(3)`、空テンソル（いずれかの軸が 0）は値から shape
/// を復元できないため `tensor([], shape=[2, 0])` のように `shape=` を
/// 付す。`Element` へ `Display` 境界は追加しない（`impl` 側の境界に
/// 限定することでトレイト自体は非破壊のまま）。
impl<T: Element + std::fmt::Display> std::fmt::Display for Tensor<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("tensor(")?;
        if self.is_empty() {
            // 空テンソル（rank 0 は `numel() == 1` のため到達しない）は
            // 値から shape を復元できないため、常に `shape=` を付す
            // （1 次元・多次元とも同じ規則。実装計画 §3.2 参照）。
            f.write_str("[]")?;
            write!(f, ", shape={:?}", self.shape())?;
        } else if self.rank() == 0 {
            match self.get(&[]) {
                Some(v) => std::fmt::Display::fmt(&FmtElem(v), f)?,
                None => f.write_str("?")?,
            }
        } else {
            let truncate = self.numel() > FMT_MAX_ELEMS;
            let mut index = Vec::with_capacity(self.rank());
            render_axis_display(self, &mut index, 0, truncate, f)?;
        }
        f.write_str(")")
    }
}

/// [`render_axis`] の `Display` 版（要素を `T::fmt`（`Display`）へ委譲する
/// 点のみ異なる。打ち切り・入れ子ロジックは共通）。
fn render_axis_display<T>(
    tensor: &Tensor<T>,
    index: &mut Vec<usize>,
    axis: usize,
    truncate: bool,
    f: &mut std::fmt::Formatter<'_>,
) -> std::fmt::Result
where
    T: Element + std::fmt::Display,
{
    let shape = tensor.shape();
    if axis == shape.len() {
        return match tensor.get(index) {
            Some(v) => std::fmt::Display::fmt(&FmtElem(v), f),
            None => f.write_str("?"),
        };
    }

    let len = shape[axis];
    f.write_str("[")?;
    if truncate && len > 2 * FMT_EDGE_ITEMS {
        for i in 0..FMT_EDGE_ITEMS {
            if i > 0 {
                f.write_str(", ")?;
            }
            index.push(i);
            render_axis_display(tensor, index, axis + 1, truncate, f)?;
            index.pop();
        }
        f.write_str(", ..., ")?;
        for i in (len - FMT_EDGE_ITEMS)..len {
            index.push(i);
            render_axis_display(tensor, index, axis + 1, truncate, f)?;
            index.pop();
            if i + 1 < len {
                f.write_str(", ")?;
            }
        }
    } else {
        for i in 0..len {
            if i > 0 {
                f.write_str(", ")?;
            }
            index.push(i);
            render_axis_display(tensor, index, axis + 1, truncate, f)?;
            index.pop();
        }
    }
    f.write_str("]")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn display_rank0() {
        let t = Tensor::scalar(3.0f32);
        assert_eq!(format!("{}", t), "tensor(3)");
    }

    #[test]
    fn debug_rank0() {
        let t = Tensor::scalar(3.0f32);
        let s = format!("{:?}", t);
        assert!(s.starts_with("Tensor {"), "{s}");
        assert!(s.contains("shape: []"), "{s}");
        assert!(s.contains("data: 3"), "{s}");
    }

    #[test]
    fn display_rank1() {
        let t = Tensor::new(vec![1.0f32, 2.0, 3.0], &[3]).unwrap();
        assert_eq!(format!("{}", t), "tensor([1, 2, 3])");
    }

    #[test]
    fn display_rank2() {
        let t = Tensor::new(vec![1.0f32, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
        assert_eq!(format!("{}", t), "tensor([[1, 2], [3, 4]])");
    }

    #[test]
    fn display_rank3() {
        let t = Tensor::new((0..8).map(|v| v as f32).collect(), &[2, 2, 2]).unwrap();
        assert_eq!(
            format!("{}", t),
            "tensor([[[0, 1], [2, 3]], [[4, 5], [6, 7]]])"
        );
    }

    #[test]
    fn display_empty_1d() {
        let t: Tensor<f32> = Tensor::new(vec![], &[0]).unwrap();
        assert_eq!(format!("{}", t), "tensor([], shape=[0])");
    }

    #[test]
    fn display_empty_2d_reports_shape() {
        let t: Tensor<f32> = Tensor::new(vec![], &[2, 0]).unwrap();
        assert_eq!(format!("{}", t), "tensor([], shape=[2, 0])");
    }

    #[test]
    fn debug_empty_2d_data_field_reports_shape() {
        // shape [2, 0]: 2 行 × 各行 0 要素なので入れ子表現は `[[], []]`
        // （outer 軸長 2 を素通しし各行が空、という一貫した規則）。
        let t: Tensor<f32> = Tensor::new(vec![], &[2, 0]).unwrap();
        let s = format!("{:?}", t);
        assert!(s.contains("data: [[], []]"), "{s}");
    }

    #[test]
    fn view_transpose_matches_contiguous_display() {
        let t = Tensor::new(vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]).unwrap();
        let tr = t.transpose_2d().unwrap();
        assert!(!tr.is_contiguous());
        let tr_contig = tr.contiguous();
        assert_eq!(format!("{}", tr), format!("{}", tr_contig));
        assert_eq!(
            format!("{}", tr),
            "tensor([[1, 4], [2, 5], [3, 6]])".to_string()
        );
    }

    #[test]
    fn view_narrow_matches_contiguous_display() {
        let t = Tensor::new((0..10).map(|v| v as f32).collect(), &[10]).unwrap();
        let n = t.narrow(0, 2, 3).unwrap();
        let n_contig = n.contiguous();
        assert_eq!(format!("{}", n), format!("{}", n_contig));
        assert_eq!(format!("{}", n), "tensor([2, 3, 4])");
    }

    #[test]
    fn view_broadcast_stride0_matches_contiguous_display() {
        let t = Tensor::new(vec![7.0f32], &[1]).unwrap();
        let b = t.broadcast_to(&[3]).unwrap();
        let b_contig = b.contiguous();
        assert_eq!(format!("{}", b), format!("{}", b_contig));
        assert_eq!(format!("{}", b), "tensor([7, 7, 7])");
    }

    #[test]
    fn truncation_boundary_full_at_1000() {
        let t = Tensor::new((0..1000).map(|v| v as f32).collect(), &[1000]).unwrap();
        let s = format!("{}", t);
        assert!(!s.contains("..."), "{s}");
        // 全要素表示: 先頭・末尾の値が両方含まれる。
        assert!(s.starts_with("tensor([0, 1, 2"), "{s}");
        assert!(s.ends_with("997, 998, 999])"), "{s}");
    }

    #[test]
    fn truncation_triggers_at_1001() {
        let t = Tensor::new((0..1001).map(|v| v as f32).collect(), &[1001]).unwrap();
        let s = format!("{}", t);
        assert!(s.contains("..."), "{s}");
        assert_eq!(s, "tensor([0, 1, 2, ..., 998, 999, 1000])");
    }

    #[test]
    fn truncation_output_length_bounded_for_large_tensor() {
        // 100x100（numel=10000）でも出力は有界（各軸が打ち切られるため
        // 総出力は軸数に対して指数的ではなく多項式的に抑えられる）。
        let t = Tensor::new(vec![0.0f32; 10_000], &[100, 100]).unwrap();
        let s = format!("{}", t);
        assert!(s.len() < 2000, "output too long: {} bytes", s.len());
        assert!(s.contains("..."), "{s}");
    }

    #[test]
    fn precision_flag_propagates_to_elements() {
        let t = Tensor::new(vec![1.0f32, 2.5], &[2]).unwrap();
        assert_eq!(format!("{:.2}", t), "tensor([1.00, 2.50])");
    }

    #[test]
    fn debug_pretty_print_does_not_panic() {
        let t = Tensor::new(vec![1.0f32, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
        let s = format!("{:#?}", t);
        assert!(s.contains("Tensor"), "{s}");
    }

    #[test]
    fn all_element_types_format_without_panicking() {
        use half::{bf16, f16};

        let a = Tensor::new(vec![true, false], &[2]).unwrap();
        let _ = format!("{} {:?}", a, a);

        let b = Tensor::new(vec![f16::from_f32(1.5), f16::from_f32(2.5)], &[2]).unwrap();
        let _ = format!("{} {:?}", b, b);

        let c = Tensor::new(vec![bf16::from_f32(1.5), bf16::from_f32(2.5)], &[2]).unwrap();
        let _ = format!("{} {:?}", c, c);

        let d = Tensor::new(vec![1i32, 2], &[2]).unwrap();
        let _ = format!("{} {:?}", d, d);

        let e = Tensor::new(vec![1i64, 2], &[2]).unwrap();
        let _ = format!("{} {:?}", e, e);

        let f = Tensor::new(vec![1.0f64, 2.0], &[2]).unwrap();
        let _ = format!("{} {:?}", f, f);

        let g = Tensor::new(vec![1.0f32, 2.0], &[2]).unwrap();
        let _ = format!("{} {:?}", g, g);
    }
}
