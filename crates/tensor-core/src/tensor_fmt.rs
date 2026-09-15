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
//! 出力は必ず要素数上限つきで打ち切る。打ち切りは 2 段構えで、総出力
//! サイズと走査コストの双方を shape に依らず有界にする
//! （`.claude/rules/security.md` A04 DoS 観点。レビュー指摘で判明した
//! 元設計の穴と是正内容はコードレビュー #1754 追加修正を参照）:
//!
//! 1. 軸ごとの打ち切り（[`FMT_MAX_ELEMS`]／[`FMT_EDGE_ITEMS`]。可読性
//!    目的）: `numel() > FMT_MAX_ELEMS` のとき、長さが
//!    `2 * FMT_EDGE_ITEMS` を超える軸だけを先頭・末尾のみへ省略する。
//!    **この基準だけでは軸ごとの長さが全て `2 * FMT_EDGE_ITEMS` 以下の
//!    高階テンソル（例: `shape = [2; 20]`。`numel` は 100 万超だが全軸
//!    長 2）で 1 軸も打ち切られず、出力が無制限に増大しうる**（元設計
//!    の穴。当初は「有界」と誤って記載していた）。
//! 2. 総出力要素数のグローバル上限（本修正で追加。[`render_axis`]／
//!    [`render_axis_display`] の各呼び出し先頭で共有カウンタを検査し、
//!    軸ごとの打ち切りが効かない形状でも総リーフ出力数を
//!    `FMT_MAX_ELEMS` 以下に強制する。カウンタ枯渇後の呼び出しは
//!    サブツリー全体を再帰せず即座に `"..."` を書いて返るため、走査
//!    コストも `O(FMT_MAX_ELEMS * rank)` に有界となる）。
//!
//! **さらに、空テンソル（`numel() == 0`）は 2 の総量予算だけでは
//! 塞げない別経路を持つ**: `budget` は末端（leaf）到達時のみ減算する
//! 設計のため、非末端軸に長さ 0 の軸を含む shape
//! （例: `shape = [1_000_000_000, 0]`）では末端へ一度も到達せず
//! `budget` が枯渇しないまま、長さ 0 の軸へ descend する直前の軸の
//! 要素数だけ空の内側配列 `"[]"` を書き続ける（本例では 10 億回。
//! `Display` は `is_empty()` を再帰前に判定して打ち切っていたが
//! `Debug`（[`DataPreview::fmt`]）側に同等のガードが無かったのが
//! 原因。コードレビュー #1754 追加修正で判明・是正: `DataPreview::fmt`
//! の先頭で `is_empty()` を判定し、非末端軸への descend 自体を発生
//! させずに打ち切る）。
//!
//! 加えて、rank 自体が極端に大きいテンソル（例: `shape = [1; 100_000]`。
//! `numel == 1` のため上記どちらの打ち切りも発動しないが、再帰の深さが
//! rank に比例しスタックオーバーフローしうる）に対しては
//! [`FMT_MAX_RENDER_RANK`] で rank 自体を検査し、超過時は再帰へ入らず
//! 代替テキストを返す（`Tensor::new`／`shape` に rank 上限はないため
//! 表示側で独立に守る必要がある。#1681 の checkpoint 反復化と同種の
//! 「非信頼な深さでの再帰を避ける」設計判断）。
//!
//! 値の走査は `Tensor::get(&[usize])` による論理インデックス順（stride
//! 走査）で行う。これにより `transpose`/`narrow`/`broadcast_to`
//! （stride 0）等の view も、対応する `contiguous()` と同一の値列で
//! 表示され、表示のためだけの中間バッファを確保しない。

use crate::element::Element;
use crate::tensor::Tensor;

/// 打ち切りを発動する要素数の上限。2 つの役割を持つ:
/// (1) `numel() > FMT_MAX_ELEMS` のとき軸ごとの打ち切り判定
///     （[`FMT_EDGE_ITEMS`]）を有効化する閾値、
/// (2) 全軸を通じて実際に出力するリーフ要素数の総量上限（グローバル
///     予算。モジュール doc 参照）。
///     いずれも PyTorch `torch.set_printoptions()` の既定値
///     （`threshold=1000`）に合わせた値。
const FMT_MAX_ELEMS: usize = 1000;

/// 打ち切り時に各軸の先頭・末尾に残す要素数。
/// PyTorch `torch.set_printoptions()` の既定値（`edgeitems=3`）と同じ。
const FMT_EDGE_ITEMS: usize = 3;

/// 表示のために再帰的に descend する rank の上限。`Tensor::new`／
/// `checked_numel` は shape の rank に上限を課さないため（要素数積が
/// `usize` に収まりさえすれば任意 rank を構築可能。`numel == 0` や
/// `1` になる shape なら rank を極端に大きくしても安価に構築できる）、
/// 表示側で独立に守らないと `render_axis`／`render_axis_display` の
/// 再帰深さが rank に比例してスタックオーバーフローしうる
/// （`.claude/rules/coding-rust.md` の panic 禁止方針・
/// `.claude/rules/security.md` A04 DoS 観点。#1681 で同種のスタック
/// オーバーフロー対策として反復化した前例があるが、本モジュールは
/// 再帰の深さ自体を通常利用で想定される範囲に制限する軽量な対策を
/// 採る）。PyTorch/NumPy が実務上使用する次元数（NumPy
/// `NPY_MAXDIMS` 系・PyTorch の実務上の次元数）を通常上回り、かつ
/// default スレッドスタックでも安全な余裕を持つ値として 64 を採用する
/// （具体的な上限値は PyTorch/NumPy のバージョンにより変わりうるため
/// 「十分大きい」という比較の断定はしない）。
const FMT_MAX_RENDER_RANK: usize = 64;

/// 現在の走査位置（多次元インデックス）から `Tensor` の指定軸以降を
/// 再帰的にレンダリングする。
///
/// `axis == shape.len()`（末端。スカラー要素 1 個）で `T::fmt` へ委譲し、
/// それ以外の軸では `[` `]` の入れ子と `, ` 区切りで descend する。
/// `truncate` は `numel() > FMT_MAX_ELEMS` のときに `true` となり、各軸の
/// 長さが `2 * FMT_EDGE_ITEMS` を超える場合に先頭・末尾のみ表示し中間を
/// `...` で省略する（該当軸の子孫がすべて省略される）。
///
/// `budget` は全軸を通じて共有する残りリーフ出力予算（呼び出し元が
/// `FMT_MAX_ELEMS` で初期化する）。`truncate` による軸ごとの打ち切り
/// だけでは軸長が全て `2 * FMT_EDGE_ITEMS` 以下の高階テンソル（例:
/// `shape = [2; 20]`）を打ち切れない（モジュール doc 参照）ため、
/// リーフを 1 個出力するたびに `budget` を 1 減らし、既に枯渇して
/// いる場合はこの呼び出し自体（葉・部分木の別なく）が構造へ descend
/// せず `"..."` を書いて即座に返る。これにより総出力要素数と走査
/// コストの双方を shape に依らず有界にする。
fn render_axis<T>(
    tensor: &Tensor<T>,
    index: &mut Vec<usize>,
    axis: usize,
    truncate: bool,
    budget: &mut usize,
    f: &mut std::fmt::Formatter<'_>,
) -> std::fmt::Result
where
    T: Element,
{
    if *budget == 0 {
        // グローバル予算が既に枯渇: 葉か部分木かを問わずこの呼び出しの
        // 構造へは descend せず単一のプレースホルダで打ち切る
        // （葉のみでチェックすると軸長が小さい高階形状で部分木ごと
        // 再帰し続け、出力サイズ・走査コストとも有界にならないため）。
        return f.write_str("...");
    }

    let shape = tensor.shape();
    if axis == shape.len() {
        // 末端: 多次元インデックスが 1 要素を指す。`Tensor::get` は
        // 論理インデックス境界内なら常に `Some` を返す契約だが、
        // 呼び出し元がその契約を破って到達不能な状態になった場合でも
        // 本番経路 panic を避けるため `?` を出力する
        // （`.claude/rules/coding-rust.md` の panic 禁止方針）。
        *budget -= 1;
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
            render_axis(tensor, index, axis + 1, truncate, budget, f)?;
            index.pop();
        }
        f.write_str(", ..., ")?;
        for i in (len - FMT_EDGE_ITEMS)..len {
            index.push(i);
            render_axis(tensor, index, axis + 1, truncate, budget, f)?;
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
            render_axis(tensor, index, axis + 1, truncate, budget, f)?;
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
        if tensor.is_empty() {
            // 空テンソル（`numel() == 0`。rank 0 は `numel() == 1` のため
            // 到達しない）は `Display` 実装（本ファイル下部）と同じく
            // 再帰前に打ち切る。`truncate` 判定（`numel() > FMT_MAX_ELEMS`）
            // は `numel() == 0` では常に `false` になり、かつ
            // `render_axis` の `budget` 減算は末端（leaf）到達時のみ
            // 発生するため、shape の非末端軸に 0 長がある場合
            // （例: `shape = [1_000_000_000, 0]`）は末端へ一度も到達
            // せず budget が枯渇しないまま非末端軸の要素数（本例では
            // 10 億）だけ `render_axis` が空の内側配列 `"[]"` を書き
            // 続ける（DoS 経路。`.claude/rules/security.md` A04
            // 観点。コードレビュー #1754 追加修正で判明）。
            return f.write_str("[]");
        }
        let truncate = tensor.numel() > FMT_MAX_ELEMS;
        if tensor.rank() == 0 {
            // rank 0（`Tensor::scalar`）は括弧なしの単一値。
            return match tensor.get(&[]) {
                Some(v) => std::fmt::Debug::fmt(&FmtElem(v), f),
                None => f.write_str("?"),
            };
        }
        if tensor.rank() > FMT_MAX_RENDER_RANK {
            // rank が表示上限を超える（`numel` は小さくても `shape` に
            // 上限がないため構築しうる。モジュール doc・
            // `FMT_MAX_RENDER_RANK` 参照）。再帰へ入らずプレースホルダ
            // のみ書いてスタックオーバーフローを避ける。
            return write!(
                f,
                "<rank {} exceeds display limit {}; numel={}>",
                tensor.rank(),
                FMT_MAX_RENDER_RANK,
                tensor.numel()
            );
        }
        let mut index = Vec::with_capacity(tensor.rank());
        let mut budget = FMT_MAX_ELEMS;
        render_axis(tensor, &mut index, 0, truncate, &mut budget, f)
    }
}

/// `shape`／`strides` フィールド表示用の打ち切り付きラッパー。
///
/// デフォルトの `Debug for Vec<usize>`／`Debug for Vec<isize>` は全
/// 要素を出力するため、`Tensor::new` が rank に上限を課さないことと
/// 合わさると、空テンソル（`data` フィールドは [`DataPreview::fmt`]
/// が `is_empty()` で即座に打ち切る）であっても `shape`／`strides`
/// フィールド自体は無条件に全要素を出力し、rank に比例して出力サイズ
/// が増大する（例: 先頭軸 0・残り 100,000 軸長 1 の shape で約 30 万
/// 文字。コードレビュー #1754 追加指摘・P2）。`data` フィールドの
/// 打ち切り判定（[`FMT_MAX_RENDER_RANK`]・グローバル予算）とは独立に
/// 発生する経路のため、`data` と同じ軸ごとの先頭・末尾省略
/// （[`FMT_EDGE_ITEMS`]）を要素数のみで判定して適用し、rank に依らず
/// 出力サイズを有界にする。
struct SlicePreview<'a, V>(&'a [V]);

impl<'a, V: std::fmt::Display> std::fmt::Debug for SlicePreview<'a, V> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let s = self.0;
        f.write_str("[")?;
        if s.len() > 2 * FMT_EDGE_ITEMS {
            for (i, v) in s[..FMT_EDGE_ITEMS].iter().enumerate() {
                if i > 0 {
                    f.write_str(", ")?;
                }
                write!(f, "{v}")?;
            }
            f.write_str(", ..., ")?;
            let tail = &s[s.len() - FMT_EDGE_ITEMS..];
            for (i, v) in tail.iter().enumerate() {
                if i > 0 {
                    f.write_str(", ")?;
                }
                write!(f, "{v}")?;
            }
        } else {
            for (i, v) in s.iter().enumerate() {
                if i > 0 {
                    f.write_str(", ")?;
                }
                write!(f, "{v}")?;
            }
        }
        f.write_str("]")
    }
}

/// `Tensor<T>` の手書き `Debug` 実装。
///
/// 構造情報（`shape`/`strides`/`offset`/`storage_len`）を
/// `f.debug_struct` で維持しつつ、`data` フィールドに打ち切り付きの
/// 値プレビューを追加する。`{:#?}`（pretty-print）でも
/// `f.debug_struct` 経由のため自然に整形される。`shape`／`strides`
/// フィールドは [`SlicePreview`] 経由で出力し、`data` の打ち切りとは
/// 独立に rank に依らず出力サイズを有界にする（P2 是正）。
///
/// `Element: Debug` は既存のトレイト境界（`crate::element::Element`）
/// のみを使い、新規境界は追加しない（`Element` は unsealed 公開 trait
/// のため必須メソッド追加は破壊的変更になる。既存の `Debug` 境界内で
/// 完結させる設計判断は issue #1754 実装計画 §3.2 参照）。
impl<T: Element> std::fmt::Debug for Tensor<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Tensor")
            .field("shape", &SlicePreview(self.shape()))
            .field("strides", &SlicePreview(self.strides()))
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
        } else if self.rank() > FMT_MAX_RENDER_RANK {
            // rank が表示上限を超える（`DataPreview::fmt` と同じ理由。
            // `FMT_MAX_RENDER_RANK` doc 参照）。再帰へ入らずプレースホルダ
            // のみ書いてスタックオーバーフローを避ける。
            write!(
                f,
                "<rank {} exceeds display limit {}; numel={}>",
                self.rank(),
                FMT_MAX_RENDER_RANK,
                self.numel()
            )?;
        } else {
            let truncate = self.numel() > FMT_MAX_ELEMS;
            let mut index = Vec::with_capacity(self.rank());
            let mut budget = FMT_MAX_ELEMS;
            render_axis_display(self, &mut index, 0, truncate, &mut budget, f)?;
        }
        f.write_str(")")
    }
}

/// [`render_axis`] の `Display` 版（要素を `T::fmt`（`Display`）へ委譲する
/// 点・`budget` によるグローバル打ち切り契約のみ共通。[`render_axis`]
/// の doc コメント参照）。
fn render_axis_display<T>(
    tensor: &Tensor<T>,
    index: &mut Vec<usize>,
    axis: usize,
    truncate: bool,
    budget: &mut usize,
    f: &mut std::fmt::Formatter<'_>,
) -> std::fmt::Result
where
    T: Element + std::fmt::Display,
{
    if *budget == 0 {
        return f.write_str("...");
    }

    let shape = tensor.shape();
    if axis == shape.len() {
        *budget -= 1;
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
            render_axis_display(tensor, index, axis + 1, truncate, budget, f)?;
            index.pop();
        }
        f.write_str(", ..., ")?;
        for i in (len - FMT_EDGE_ITEMS)..len {
            index.push(i);
            render_axis_display(tensor, index, axis + 1, truncate, budget, f)?;
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
            render_axis_display(tensor, index, axis + 1, truncate, budget, f)?;
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
        // shape [2, 0]（`numel() == 0`）は `DataPreview::fmt` が
        // `is_empty()` を再帰前に判定して打ち切るため、outer 軸長を
        // 素通しした入れ子表現（`[[], []]`）ではなくフラットな `[]`
        // になる（`Display` と同じ規約。`data` フィールドとは別に
        // `shape` フィールドが既に構造情報を持つため、shape の再掲は
        // 不要）。outer 軸長が巨大（例: `shape = [1_000_000_000, 0]`）
        // な場合に非末端軸への descend 自体を発生させないための設計
        // （コードレビュー #1754 追加修正。モジュール doc 参照）。
        let t: Tensor<f32> = Tensor::new(vec![], &[2, 0]).unwrap();
        let s = format!("{:?}", t);
        assert!(s.contains("data: []"), "{s}");
    }

    #[test]
    fn debug_empty_huge_leading_axis_does_not_hang() {
        // レビュー指摘の再現条件そのもの: shape の先頭軸が巨大
        // （10 億）でも 2 軸目が 0 長のため `numel() == 0` となり、
        // 実データ確保は発生しない。`is_empty()` の早期打ち切りが
        // 無ければ非末端軸（axis=0）の `for i in 0..len` ループが
        // 末端へ一度も到達せず `budget` を消費しないまま「軸長 0 の
        // 内側配列 `[]`」を 10 億回書き続ける DoS 経路になっていた
        // （`.claude/rules/security.md` A04 観点）。本テストは出力が
        // 定数時間・定数サイズで完了することを固定する回帰テスト。
        let t: Tensor<f32> = Tensor::new(vec![], &[1_000_000_000, 0]).unwrap();
        let s = format!("{:?}", t);
        assert!(s.contains("data: []"), "{s}");
        let d = format!("{}", t);
        assert_eq!(d, "tensor([], shape=[1000000000, 0])");
    }

    #[test]
    fn debug_empty_huge_rank_shape_field_is_bounded() {
        // コードレビュー #1754 追加指摘（P2）の再現条件: 空テンソル
        // （先頭軸 0）で残り軸数が極端に大きい shape（本例では
        // 100,000 軸・rank 100,001）でも、`shape`／`strides` フィールド
        // 自体は `SlicePreview` により先頭・末尾のみへ打ち切られ、
        // 出力サイズが rank に依らず有界であることを固定する。
        // （`data` フィールドは `is_empty()` の早期打ち切りにより
        // 既に `[]` へ収まる。`debug_empty_huge_leading_axis_does_not_hang`
        // とは異なり shape フィールド自体の打ち切りを検証する）。
        let mut shape = vec![0usize];
        shape.extend(std::iter::repeat_n(1usize, 100_000));
        let t: Tensor<f32> = Tensor::new(vec![], &shape).unwrap();
        let s = format!("{:?}", t);
        assert!(s.contains("data: []"), "{s}");
        assert!(s.contains("..."), "{s}");
        assert!(
            s.len() < 2_000,
            "shape/strides field not bounded: {} bytes",
            s.len()
        );
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
        // 100x100（numel=10000）は各軸長（100）が `2 * FMT_EDGE_ITEMS`
        // を超えるため軸ごとの打ち切りだけでも十分有界（総リーフ出力数
        // は `FMT_EDGE_ITEMS` の 2 乗のオーダーに収まる）。
        let t = Tensor::new(vec![0.0f32; 10_000], &[100, 100]).unwrap();
        let s = format!("{}", t);
        assert!(s.len() < 2000, "output too long: {} bytes", s.len());
        assert!(s.contains("..."), "{s}");
    }

    #[test]
    fn truncation_bounds_output_when_no_single_axis_exceeds_edge_items() {
        // レビュー指摘（イシュー #1754）の再現ケース: 軸ごとの打ち切り
        // （`len > 2 * FMT_EDGE_ITEMS`）は各軸長が 2 の rank 20 テンソル
        // （numel = 2^20 = 1,048,576）では 1 度も発動しない
        // （`2 <= 2 * FMT_EDGE_ITEMS`）。修正前はここで出力が
        // 5MB 超・打ち切りマーカーなしに膨張していた。グローバル予算
        // （`FMT_MAX_ELEMS`）がこのケースを打ち切ることを確認する。
        let t = Tensor::new(vec![0.0f32; 1 << 20], &[2; 20]).unwrap();
        let s = format!("{}", t);
        assert!(s.contains("..."), "{s}");
        // 総リーフ出力数は `FMT_MAX_ELEMS`（1000）以下に抑えられるため、
        // 要素・カンマ・ネストの記号を含めても数万バイト程度で収まる
        // （5MB という修正前の実測値とは桁が異なることを確認する）。
        assert!(s.len() < 50_000, "output too long: {} bytes", s.len());

        let dbg = format!("{:?}", t);
        assert!(dbg.contains("..."), "{dbg}");
        assert!(
            dbg.len() < 50_000,
            "debug output too long: {} bytes",
            dbg.len()
        );
    }

    #[test]
    fn truncation_bounds_output_via_global_budget_only() {
        // 各軸長 6（`== 2 * FMT_EDGE_ITEMS`。軸ごとの打ち切り条件
        // `len > 2 * FMT_EDGE_ITEMS` は非成立）・rank 4・
        // numel = 6^4 = 1296（`FMT_MAX_ELEMS` = 1000 超）。
        // 軸ごとの打ち切りが一度も発動せず、グローバル予算のみで
        // 打ち切られることを判別する最小ケース。
        let t = Tensor::new(vec![0.0f32; 1296], &[6, 6, 6, 6]).unwrap();
        let s = format!("{}", t);
        assert!(s.contains("..."), "{s}");
        assert!(s.len() < 20_000, "output too long: {} bytes", s.len());
    }

    #[test]
    fn extreme_rank_does_not_overflow_stack() {
        // `Tensor::new`／`checked_numel` は rank に上限を課さないため、
        // 全軸長 1（numel == 1）にすれば安価に極端な rank（100,000）の
        // テンソルを構築できる。`FMT_MAX_RENDER_RANK` によるガードが
        // なければ `render_axis`／`render_axis_display` が rank に比例
        // した深さまで再帰しスタックオーバーフローしうる
        // （`.claude/rules/security.md` A04 DoS 観点）。
        let t = Tensor::new(vec![0.0f32], &[1; 100_000]).unwrap();
        let s = format!("{}", t);
        assert!(s.contains("exceeds display limit"), "{s}");
        let dbg = format!("{:?}", t);
        assert!(dbg.contains("exceeds display limit"), "{dbg}");
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
