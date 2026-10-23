//! Rank-2 shapes with an optional symbolic batch dimension.
//!
//! Every tensor in BoundML is a dense, row-major, f32 matrix. Scalars are `[1, 1]`
//! and vectors are `[1, n]`. A dimension is either fixed or the runtime batch size
//! `B`, which must lie in `1..=max_batch` declared when a plan is built.

use crate::error::{CoreError, Result};
use serde::{Deserialize, Serialize};

/// Hard ceiling on any single dimension; protects shape arithmetic from hostile metadata.
pub const MAX_DIM: usize = 1 << 24;
/// Hard ceiling on elements in any single tensor (256 Mi elements = 1 GiB of f32).
pub const MAX_ELEMENTS: usize = 1 << 28;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Dim {
    Batch,
    Fixed(usize),
}

impl Dim {
    pub fn resolve(self, batch: usize) -> usize {
        match self {
            Dim::Batch => batch,
            Dim::Fixed(n) => n,
        }
    }
}

impl std::fmt::Display for Dim {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Dim::Batch => write!(f, "B"),
            Dim::Fixed(n) => write!(f, "{n}"),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct Shape {
    pub rows: Dim,
    pub cols: Dim,
}

impl Shape {
    pub const SCALAR: Shape = Shape {
        rows: Dim::Fixed(1),
        cols: Dim::Fixed(1),
    };

    pub fn new(rows: Dim, cols: Dim) -> Result<Shape> {
        for d in [rows, cols] {
            if let Dim::Fixed(n) = d {
                if n == 0 || n > MAX_DIM {
                    return Err(CoreError::Shape(format!(
                        "dimension {n} outside 1..={MAX_DIM}"
                    )));
                }
            }
        }
        Ok(Shape { rows, cols })
    }

    pub fn fixed(rows: usize, cols: usize) -> Result<Shape> {
        Shape::new(Dim::Fixed(rows), Dim::Fixed(cols))
    }

    pub fn batch(cols: usize) -> Result<Shape> {
        Shape::new(Dim::Batch, Dim::Fixed(cols))
    }

    pub fn transposed(self) -> Shape {
        Shape {
            rows: self.cols,
            cols: self.rows,
        }
    }

    /// Concrete `(rows, cols)` for a given batch size.
    pub fn dims(self, batch: usize) -> (usize, usize) {
        (self.rows.resolve(batch), self.cols.resolve(batch))
    }

    /// Checked element count for a given batch size.
    pub fn elements(self, batch: usize) -> Result<usize> {
        let (r, c) = self.dims(batch);
        let n = r
            .checked_mul(c)
            .ok_or(CoreError::Overflow("tensor elements"))?;
        if n > MAX_ELEMENTS {
            return Err(CoreError::Shape(format!(
                "tensor of {r}x{c} exceeds {MAX_ELEMENTS} elements"
            )));
        }
        Ok(n)
    }

    /// Checked byte size (f32) for a given batch size.
    pub fn bytes(self, batch: usize) -> Result<u64> {
        let n = self.elements(batch)? as u64;
        n.checked_mul(4).ok_or(CoreError::Overflow("tensor bytes"))
    }

    pub fn is_scalar(self) -> bool {
        self == Shape::SCALAR
    }
}

impl std::fmt::Display for Shape {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "[{}, {}]", self.rows, self.cols)
    }
}

/// Broadcasting rule for binary elementwise ops: `b` may equal `a`, or have a
/// dimension of fixed size 1 that broadcasts over the corresponding dimension of `a`.
pub fn broadcastable(a: Shape, b: Shape) -> bool {
    let ok = |da: Dim, db: Dim| da == db || db == Dim::Fixed(1);
    ok(a.rows, b.rows) && ok(a.cols, b.cols)
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    #[test]
    fn rejects_zero_and_huge_dims() {
        assert!(Shape::fixed(0, 3).is_err());
        assert!(Shape::fixed(3, MAX_DIM + 1).is_err());
        assert!(Shape::fixed(MAX_DIM, MAX_DIM).unwrap().elements(1).is_err());
    }

    #[test]
    fn batch_resolution() {
        let s = Shape::batch(7).unwrap();
        assert_eq!(s.dims(5), (5, 7));
        assert_eq!(s.transposed().dims(5), (7, 5));
        assert_eq!(s.bytes(3).unwrap(), 84);
    }

    proptest! {
        #[test]
        fn elements_never_panic(r in 1usize..=MAX_DIM, c in 1usize..=MAX_DIM, b in 1usize..=MAX_DIM) {
            let s = Shape::new(Dim::Batch, Dim::Fixed(c)).unwrap();
            match s.elements(b) {
                Ok(n) => prop_assert_eq!(n, b * c),
                Err(_) => prop_assert!((b as u128) * (c as u128) > MAX_ELEMENTS as u128),
            }
            let f = Shape::fixed(r, c).unwrap();
            if let Ok(bytes) = f.bytes(1) { prop_assert_eq!(bytes, (r * c * 4) as u64); }
        }
    }
}
