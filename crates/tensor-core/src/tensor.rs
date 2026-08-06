//! テンソル型本体（TASK-1.4a）。
//!
//! `Tensor<T>` は `autodiff`（`Var`/`Tape` が非追跡の葉ノードとして保持
//! する）・各バックエンド（`backend-cpu`/`backend-cuda`/`backend-metal`
//! が演算グラフのノードをカーネルへ変換する際の入出力表現）から共通して
//! 参照される、本ライブラリの最も基礎的なデータ構造である
//! （spec 根拠: `docs/spec/05-tasks.md` TASK-1.4、`docs/public-api-design.md`
//! §2）。
//!
//! PoC-v2-1（`docs/spec/03-poc/poc-v2-1-tensor-cpu-gemm/`）で確定した
//! 「行優先連続バッファ」のメモリレイアウト方針を維持しつつ、所有構造は
//! `Vec<T>` 直接所有から `Arc<Storage<T>>` + `offset` + `strides` へ
//! 変更する。これは PoC-v2-1 確定事項の明示的変更としてユーザー承認済み
//! （issue #182 コメント 2026-08-05、`docs/public-api-design.md` §2.1）。
//! `Arc` 共有により `transpose`/`narrow`/`reshape`（contiguous ケース）が
//! データコピーなしで成立し、`autodiff` のテープが同一データを複数ノード
//! から参照する際の複製コストも避けられる。

use std::sync::Arc;

use crate::element::Element;
use crate::error::ShapeError;

/// 実データを保持する共有バッファ。`Tensor` 間で `Arc` 共有される。
///
/// 非公開: 外部から直接構築・可変参照させると `Arc` 共有の前提
/// （複数 `Tensor` が同一データを指しうる）が壊れるため、公開 API は
/// すべて `Tensor` 経由に限定する。単一所有かつ非共有であることが
/// 分かっている経路（学習ループの勾配バッファ等）での `Arc::get_mut`
/// による in-place 更新最適化は、本イシューでは行わず将来へ保留する
/// （`docs/public-api-design.md` §2.1 コメント）。
struct Storage<T: Element> {
    data: Vec<T>,
}

/// テンソル本体。`storage` を複数の `Tensor` が共有することで
/// view 系操作（transpose/narrow/reshape の contiguous ケース）を
/// データコピーなしで表現する。
///
/// メモリレイアウト（行優先 + strides）は PoC-v2-1 の確定事項を
/// 維持する。`strides` は将来の stride 0 ブロードキャスト（#12）・
/// 負 stride 拡張に備え `isize` とする（`docs/public-api-design.md`
/// §2.1）。
///
/// `Clone` は `Arc` のポインタ複製のみで安価（データコピーを伴わない）。
/// `PartialEq` は意図的に derive しない: `offset`/`strides` が異なる
/// 同値 view（例: 同一データを異なる `narrow` で参照した 2 つの
/// `Tensor`）を構造的に不一致と誤判定するため。論理的な値の等価判定が
/// 必要になった場合は要素比較の別 API として後続イシューで導入する。
#[derive(Clone, Debug)]
pub struct Tensor<T: Element> {
    storage: Arc<Storage<T>>,
    offset: usize,
    shape: Vec<usize>,
    strides: Vec<isize>,
}

impl<T: Element> std::fmt::Debug for Storage<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Storage")
            .field("len", &self.data.len())
            .finish()
    }
}

/// shape から行優先（row-major）の strides を計算する。
///
/// 例: `[2, 3, 4]` → `[12, 4, 1]`。PoC-v2-1 の同名関数を `isize` 化して
/// 移植したもの（`docs/spec/03-poc/poc-v2-1-tensor-cpu-gemm/code/rust/src/tensor.rs`）。
/// サイズ 0 の軸を含む shape（空テンソル）にも対応する。
///
/// オーバーフロー方針: 本関数はすべての呼び出し元
/// （`Tensor::new`/`transpose`/`reshape`/`contiguous`）で事前に
/// `checked_numel` が `usize` 範囲の要素数積を検証済みの shape に
/// のみ適用されるため、通常到達しない。それでも `isize` は `usize` より
/// 表現範囲が狭く理論上超過しうる巨大 shape（実運用ではメモリ確保が
/// 先に失敗し到達不能）に備え、`checked_mul`（`ElementCountOverflow` を
/// 返す）ではなく `saturating_mul`（`isize::MAX` へクランプ）を用いる。
/// 本関数は `Result` を返さない内部ヘルパーであり、strides 計算単独の
/// オーバーフローを呼び出し元へ伝播する経路を持たないための妥協である。
fn row_major_strides(shape: &[usize]) -> Vec<isize> {
    let mut strides = vec![0isize; shape.len()];
    let mut acc: isize = 1;
    for i in (0..shape.len()).rev() {
        strides[i] = acc;
        // shape[i] が 0 の場合でも acc は 0 になるが、空テンソルは
        // numel() が 0 のため後続アクセスは発生せず問題ない。
        acc = acc.saturating_mul(shape[i] as isize);
    }
    strides
}

/// shape の要素数積を `checked_mul` の畳み込みで計算する。
///
/// アロケーション前の要素数計算に整数オーバーフローが混入すると
/// 過小アロケーション・境界検査の不整合を招く（OWASP A03 相当の
/// 不正入力対策。`.claude/rules/security.md`）ため、`usize` の
/// オーバーフローを検査付きで検出する。
fn checked_numel(shape: &[usize]) -> Result<usize, ShapeError> {
    shape
        .iter()
        .try_fold(1usize, |acc, &dim| acc.checked_mul(dim))
        .ok_or(ShapeError::ElementCountOverflow)
}

impl<T: Element> Tensor<T> {
    /// 実行時 shape 検査を行うコンストラクタ（PoC-v2-1 で確定した方式）。
    /// `data.len()` が `shape` の要素数積と一致しない場合 `ShapeError` を
    /// 返す。要素数積のオーバーフローも生成前に検査する。
    pub fn new(data: Vec<T>, shape: &[usize]) -> Result<Tensor<T>, ShapeError> {
        let expected = checked_numel(shape)?;
        if data.len() != expected {
            return Err(ShapeError::ElementCountMismatch {
                expected,
                actual: data.len(),
            });
        }
        let strides = row_major_strides(shape);
        Ok(Tensor {
            storage: Arc::new(Storage { data }),
            offset: 0,
            shape: shape.to_vec(),
            strides,
        })
    }

    /// `data` をコピーして `new` と同様に構築する。
    pub fn from_slice(data: &[T], shape: &[usize]) -> Result<Tensor<T>, ShapeError> {
        Tensor::new(data.to_vec(), shape)
    }

    /// 全要素 `T::zero()` で埋めたテンソルを生成する。
    /// shape の要素数積のオーバーフロー時に `Err` を返す
    /// （PoC の `expect` を排除し `Result` 化。本番経路で `unwrap`/
    /// `expect` を使わない方針。`.claude/rules/coding-rust.md`）。
    pub fn zeros(shape: &[usize]) -> Result<Tensor<T>, ShapeError> {
        Tensor::full(shape, T::zero())
    }

    /// 全要素 `T::one()` で埋めたテンソルを生成する。
    pub fn ones(shape: &[usize]) -> Result<Tensor<T>, ShapeError> {
        Tensor::full(shape, T::one())
    }

    /// 全要素を `value` で埋めたテンソルを生成する。
    pub fn full(shape: &[usize], value: T) -> Result<Tensor<T>, ShapeError> {
        let numel = checked_numel(shape)?;
        let data = vec![value; numel];
        let strides = row_major_strides(shape);
        Ok(Tensor {
            storage: Arc::new(Storage { data }),
            offset: 0,
            shape: shape.to_vec(),
            strides,
        })
    }

    /// shape（各軸のサイズ）を返す。
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    /// strides（各軸を 1 進めた際の要素インデックス増分）を返す。
    pub fn strides(&self) -> &[isize] {
        &self.strides
    }

    /// storage 先頭からの要素オフセットを返す。
    pub fn offset(&self) -> usize {
        self.offset
    }

    /// 次元数（rank）を返す。
    pub fn rank(&self) -> usize {
        self.shape.len()
    }

    /// 全要素数を返す。
    pub fn numel(&self) -> usize {
        self.shape.iter().product()
    }

    /// 要素数が 0 か判定する。
    pub fn is_empty(&self) -> bool {
        self.numel() == 0
    }

    /// 多次元インデックスから要素を取得する。
    ///
    /// `offset + Σ idx[i] * strides[i]` で対象要素の位置を解決する
    /// （stride ベースアクセスの受け入れ条件検証に用いる基本 API）。
    /// rank 不一致・軸範囲外は `None` を返す（型付きエラーではなく
    /// `Option` とするのは、ホットパスでの要素アクセスに `Result` の
    /// 分岐コストを強制しないため。範囲検査を伴う低頻度呼び出しの
    /// shape 検査は `ShapeError` を用いる view 系 API 側で行う）。
    pub fn get(&self, index: &[usize]) -> Option<T> {
        if index.len() != self.shape.len() {
            return None;
        }
        let mut pos: isize = self.offset as isize;
        for (i, &idx) in index.iter().enumerate() {
            if idx >= self.shape[i] {
                return None;
            }
            pos += idx as isize * self.strides[i];
        }
        self.storage.data.get(usize::try_from(pos).ok()?).copied()
    }

    /// contiguous な場合のみ、storage 全体を指すスライスを返す。
    ///
    /// 後続のカーネル実装（backend-cpu 等、#21 以降）がデータへ
    /// 直接アクセスする際の受け渡し口として想定する。非 contiguous な
    /// テンソル（transpose/narrow 後）は連続領域を保証できないため
    /// `None` を返す。
    pub fn as_slice(&self) -> Option<&[T]> {
        if !self.is_contiguous() {
            return None;
        }
        let start = self.offset;
        let end = start + self.numel();
        self.storage.data.get(start..end)
    }

    /// 2 軸の strides を入れ替えるのみ。常に zero-copy。
    /// 転置後は非 contiguous になりうる（`is_contiguous()` で判定）。
    pub fn transpose(&self, dim0: usize, dim1: usize) -> Result<Tensor<T>, ShapeError> {
        let rank = self.rank();
        if dim0 >= rank {
            return Err(ShapeError::AxisOutOfRange { axis: dim0, rank });
        }
        if dim1 >= rank {
            return Err(ShapeError::AxisOutOfRange { axis: dim1, rank });
        }
        let mut shape = self.shape.clone();
        let mut strides = self.strides.clone();
        shape.swap(dim0, dim1);
        strides.swap(dim0, dim1);
        Ok(Tensor {
            storage: Arc::clone(&self.storage),
            offset: self.offset,
            shape,
            strides,
        })
    }

    /// 指定軸の `[start, start+len)` をスライスする。offset/shape の
    /// 調整のみで常に zero-copy。
    pub fn narrow(&self, dim: usize, start: usize, len: usize) -> Result<Tensor<T>, ShapeError> {
        let rank = self.rank();
        if dim >= rank {
            return Err(ShapeError::AxisOutOfRange { axis: dim, rank });
        }
        let dim_size = self.shape[dim];
        let in_bounds = start.checked_add(len).is_some_and(|end| end <= dim_size);
        if !in_bounds {
            return Err(ShapeError::NarrowOutOfBounds {
                dim,
                start,
                len,
                dim_size,
            });
        }
        let mut shape = self.shape.clone();
        shape[dim] = len;
        // 現行実装では strides は常に非負（row_major_strides が生成する
        // 値のみを保持し、transpose は要素の入れ替えのみで負値を導入しない）
        // ため、new_offset は常に非負になる。将来 stride 0 ブロードキャスト
        // （#12）・負 stride を導入する際は本計算式の見直しが必要になる。
        let new_offset = self.offset as isize + start as isize * self.strides[dim];
        debug_assert!(new_offset >= 0, "narrow produced a negative offset");
        Ok(Tensor {
            storage: Arc::clone(&self.storage),
            offset: new_offset.max(0) as usize,
            shape,
            strides: self.strides.clone(),
        })
    }

    /// 現在のテンソルが行優先で連続配置されているか判定する。
    /// サイズ 0・1 の軸は strides に依らず標準的に許容する
    /// （NumPy 等の慣習に合わせ、当該軸の stride 値は判定に用いない）。
    ///
    /// `numel() == 0`（空テンソル）は常に連続とみなす（NumPy 方式）。
    /// `transpose`／内側軸を長さ 0 まで `narrow` した後の空テンソルは
    /// 残った軸に非ゼロの stride を保持しうるが、`row_major_strides` は
    /// サイズ 0 の軸が存在すると外側の stride を 0 に潰す
    /// （`row_major_strides` の実装コメント参照）ため、要素アクセスが
    /// 発生しない空テンソルの軸ごと比較では偽陰性（誤って非連続と判定）
    /// になりうる。`reshape`/`as_slice` はいずれも本関数の結果を経由する
    /// ため、ここで早期 `true` を返すことで空テンソル同士の
    /// 有効な reshape・スライス取得を誤って弾かない。
    pub fn is_contiguous(&self) -> bool {
        if self.numel() == 0 {
            return true;
        }
        let expected = row_major_strides(&self.shape);
        for (i, &dim) in self.shape.iter().enumerate() {
            if dim <= 1 {
                continue;
            }
            if self.strides[i] != expected[i] {
                return false;
            }
        }
        true
    }

    /// 新しい shape へ再解釈する。contiguous な場合のみ zero-copy
    /// （storage を共有し新しい strides を割り当てるのみ）。
    ///
    /// 非 contiguous な場合は `ShapeError::NonContiguousReshape` を
    /// 返す（案 A。`docs/public-api-design.md` §2.2.1 が推奨する
    /// 安全側の方針。自動運転モードのため安全側に倒して採用した。
    /// 最終決定はユーザー承認事項として残っており、案 B（暗黙コピー）
    /// へ変更する場合はこの分岐のみを差し替えればよい）。
    pub fn reshape(&self, shape: &[usize]) -> Result<Tensor<T>, ShapeError> {
        let expected = checked_numel(shape)?;
        if expected != self.numel() {
            return Err(ShapeError::ElementCountMismatch {
                expected,
                actual: self.numel(),
            });
        }
        if !self.is_contiguous() {
            return Err(ShapeError::NonContiguousReshape);
        }
        let strides = row_major_strides(shape);
        Ok(Tensor {
            storage: Arc::clone(&self.storage),
            offset: self.offset,
            shape: shape.to_vec(),
            strides,
        })
    }

    /// 非 contiguous な場合に、行優先連続バッファへ実体化した新しい
    /// `Tensor` を返す（常にコピーを伴う明示 API）。contiguous な場合は
    /// 自身の複製（`Arc` 共有のまま、コピーなし）を返す。
    pub fn contiguous(&self) -> Tensor<T> {
        if self.is_contiguous() {
            return self.clone();
        }
        // 非 contiguous: 多次元インデックスを行優先順に走査して実体化する。
        // 走査する index は shape 範囲内であることをループ自体が保証する
        // （各軸 0..shape[axis] の範囲でのみ繰り上げる）ため、本番経路で
        // `unwrap`/`expect` を使わない方針（`.claude/rules/coding-rust.md`）
        // に沿い、`get` の `None` 分岐は `T::zero()` にフォールバックする
        // 到達不能パスとして扱う（到達すれば shape 走査ロジックのバグ）。
        // ただし黙って `T::zero()` へ落ちるとリグレッションを検知しづらい
        // ため、debug ビルドでは `debug_assert!` で到達不能パスを明示的に
        // panic させ検知可能にする（release ビルドは安全側フォールバックを維持）。
        let shape = self.shape.clone();
        let numel = self.numel();
        let mut data = Vec::with_capacity(numel);
        let mut index = vec![0usize; shape.len()];
        for _ in 0..numel {
            let value = self.get(&index);
            debug_assert!(
                value.is_some(),
                "contiguous(): shape 走査ロジックのバグにより index {index:?} が範囲外になった"
            );
            data.push(value.unwrap_or_else(T::zero));
            // 行優先順で index をインクリメント（最終軸から繰り上げ）。
            for axis in (0..shape.len()).rev() {
                index[axis] += 1;
                if index[axis] < shape[axis] {
                    break;
                }
                index[axis] = 0;
            }
        }
        let strides = row_major_strides(&shape);
        Tensor {
            storage: Arc::new(Storage { data }),
            offset: 0,
            shape,
            strides,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn row_major_strides_basic() {
        assert_eq!(row_major_strides(&[2, 3, 4]), vec![12, 4, 1]);
    }

    #[test]
    fn new_element_count_mismatch() {
        let err = Tensor::<f32>::new(vec![1.0, 2.0, 3.0], &[2, 2]).unwrap_err();
        assert!(matches!(
            err,
            ShapeError::ElementCountMismatch {
                expected: 4,
                actual: 3
            }
        ));
    }

    #[test]
    fn new_element_count_overflow() {
        let err = Tensor::<f32>::zeros(&[usize::MAX, 2]).unwrap_err();
        assert!(matches!(err, ShapeError::ElementCountOverflow));
    }

    #[test]
    fn zeros_ones_full_values() {
        let z = Tensor::<f32>::zeros(&[2, 2]).unwrap();
        assert!(z.get(&[0, 0]).unwrap() == 0.0);
        assert_eq!(z.shape(), &[2, 2]);

        let o = Tensor::<f32>::ones(&[2, 2]).unwrap();
        assert_eq!(o.get(&[1, 1]).unwrap(), 1.0);

        let f = Tensor::<i32>::full(&[3], 7).unwrap();
        assert_eq!(f.get(&[2]).unwrap(), 7);

        let hz = Tensor::<half::f16>::zeros(&[2]).unwrap();
        assert_eq!(hz.get(&[0]).unwrap(), half::f16::ZERO);
        let ho = Tensor::<half::f16>::ones(&[2]).unwrap();
        assert_eq!(ho.get(&[0]).unwrap(), half::f16::ONE);
    }

    #[test]
    fn empty_tensor() {
        let t = Tensor::<f32>::zeros(&[0, 3]).unwrap();
        assert!(t.is_empty());
        assert_eq!(t.numel(), 0);
    }

    #[test]
    fn transpose_strides_and_contiguity() {
        let t = Tensor::<f32>::new((0..6).map(|v| v as f32).collect(), &[2, 3]).unwrap();
        let tt = t.transpose(0, 1).unwrap();
        assert_eq!(tt.shape(), &[3, 2]);
        assert_eq!(tt.strides(), &[1, 3]);
        assert!(!tt.is_contiguous());
        // 転置後も要素アクセスは元データと整合する: t[i][j] == tt[j][i]
        for i in 0..2 {
            for j in 0..3 {
                assert_eq!(t.get(&[i, j]), tt.get(&[j, i]));
            }
        }
    }

    #[test]
    fn transpose_axis_out_of_range() {
        let t = Tensor::<f32>::zeros(&[2, 3]).unwrap();
        let err = t.transpose(0, 5).unwrap_err();
        assert!(matches!(
            err,
            ShapeError::AxisOutOfRange { axis: 5, rank: 2 }
        ));
    }

    #[test]
    fn narrow_offset_and_shape() {
        let t = Tensor::<f32>::new((0..6).map(|v| v as f32).collect(), &[2, 3]).unwrap();
        let n = t.narrow(1, 1, 2).unwrap();
        assert_eq!(n.shape(), &[2, 2]);
        assert_eq!(n.get(&[0, 0]).unwrap(), 1.0);
        assert_eq!(n.get(&[1, 1]).unwrap(), 5.0);
    }

    #[test]
    fn narrow_out_of_bounds() {
        let t = Tensor::<f32>::zeros(&[2, 3]).unwrap();
        let err = t.narrow(1, 2, 2).unwrap_err();
        assert!(matches!(
            err,
            ShapeError::NarrowOutOfBounds {
                dim: 1,
                start: 2,
                len: 2,
                dim_size: 3
            }
        ));
    }

    #[test]
    fn narrow_out_of_bounds_overflow_display_does_not_panic() {
        // `start.checked_add(len)` がオーバーフローするケース
        // （`start` が `usize::MAX` 付近）でも `ShapeError` の
        // `Display` 実装（error.rs）が panic せず整形できることを
        // 検証する（`saturating_add` によるオーバーフロー回避の回帰テスト）。
        let t = Tensor::<f32>::zeros(&[2, 3]).unwrap();
        let err = t.narrow(0, usize::MAX, 1).unwrap_err();
        assert!(matches!(
            err,
            ShapeError::NarrowOutOfBounds {
                dim: 0,
                start: usize::MAX,
                len: 1,
                dim_size: 2
            }
        ));
        let _ = err.to_string();
    }

    #[test]
    fn narrow_axis_out_of_range() {
        let t = Tensor::<f32>::zeros(&[2, 3]).unwrap();
        let err = t.narrow(5, 0, 1).unwrap_err();
        assert!(matches!(
            err,
            ShapeError::AxisOutOfRange { axis: 5, rank: 2 }
        ));
    }

    #[test]
    fn reshape_contiguous_is_zero_copy() {
        let t = Tensor::<f32>::new((0..6).map(|v| v as f32).collect(), &[2, 3]).unwrap();
        let r = t.reshape(&[3, 2]).unwrap();
        assert_eq!(r.shape(), &[3, 2]);
        assert!(Arc::ptr_eq(&t.storage, &r.storage));
    }

    #[test]
    fn reshape_element_count_mismatch() {
        let t = Tensor::<f32>::zeros(&[2, 3]).unwrap();
        let err = t.reshape(&[4, 4]).unwrap_err();
        assert!(matches!(
            err,
            ShapeError::ElementCountMismatch {
                expected: 16,
                actual: 6
            }
        ));
    }

    #[test]
    fn reshape_non_contiguous_errors_and_contiguous_recovers() {
        let t = Tensor::<f32>::new((0..6).map(|v| v as f32).collect(), &[2, 3]).unwrap();
        let tt = t.transpose(0, 1).unwrap();
        let err = tt.reshape(&[6]).unwrap_err();
        assert!(matches!(err, ShapeError::NonContiguousReshape));

        let c = tt.contiguous();
        assert!(c.is_contiguous());
        let r = c.reshape(&[6]).unwrap();
        assert_eq!(r.shape(), &[6]);
        // contiguous() 経由の値は転置後の論理順（行優先で tt を読んだ順）と一致する。
        let expected: Vec<f32> = (0..3)
            .flat_map(|j| (0..2).map(move |i| (i * 3 + j) as f32))
            .collect();
        let got: Vec<f32> = (0..6).map(|i| r.get(&[i]).unwrap()).collect();
        assert_eq!(got, expected);
    }

    #[test]
    fn clone_shares_storage() {
        let t = Tensor::<f32>::zeros(&[2, 2]).unwrap();
        let v = t.narrow(0, 0, 1).unwrap();
        let v_clone = v.clone();
        assert!(Arc::ptr_eq(&v.storage, &v_clone.storage));
    }

    #[test]
    fn as_slice_contiguous_only() {
        let t = Tensor::<f32>::new((0..6).map(|v| v as f32).collect(), &[2, 3]).unwrap();
        assert!(t.as_slice().is_some());
        let tt = t.transpose(0, 1).unwrap();
        assert!(tt.as_slice().is_none());
    }

    // Bugbot 指摘（PR #215）: transpose／内側軸を長さ 0 まで narrow した後の
    // 空テンソル（numel == 0）が is_contiguous で誤って非連続と判定され、
    // reshape が NonContiguousReshape を返す・as_slice が None を返す不具合の
    // 回帰テスト。NumPy 方式では空テンソルは常に連続とみなす。
    #[test]
    fn empty_tensor_after_transpose_is_contiguous() {
        let t = Tensor::<f32>::zeros(&[0, 3]).unwrap();
        let tt = t.transpose(0, 1).unwrap();
        assert_eq!(tt.shape(), &[3, 0]);
        assert!(tt.numel() == 0);
        assert!(tt.is_contiguous());
        assert!(tt.as_slice().is_some());
        let r = tt.reshape(&[0]).unwrap();
        assert_eq!(r.shape(), &[0]);
    }

    #[test]
    fn empty_tensor_after_inner_narrow_is_contiguous() {
        let t = Tensor::<f32>::zeros(&[2, 3]).unwrap();
        let n = t.narrow(1, 0, 0).unwrap();
        assert_eq!(n.shape(), &[2, 0]);
        assert!(n.numel() == 0);
        assert!(n.is_contiguous());
        assert!(n.as_slice().is_some());
        let r = n.reshape(&[0]).unwrap();
        assert_eq!(r.shape(), &[0]);
    }
}
