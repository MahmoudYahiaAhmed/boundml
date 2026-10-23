//! Disjoint borrows inside one arena.
//!
//! This is the only module in the crate that contains `unsafe`. It hands out one
//! mutable slice and several shared slices of the same `Vec<f32>` arena after
//! *checking at runtime* (in release builds too) that every range is in bounds and
//! that the mutable range does not intersect any shared arena range. Shared ranges
//! may overlap each other. Parameter reads come from a different buffer and cannot
//! alias the arena.

use std::ops::Range;

/// Where an operand's data lives.
#[derive(Debug, Clone)]
pub(crate) enum Src {
    Arena(Range<usize>),
    Param(Range<usize>),
}

/// Borrow `R` read-only operands and one writable arena range simultaneously.
///
/// # Panics
/// If a range is out of bounds or the write range intersects an arena read range.
/// The planner guarantees neither happens (`plan::verify_no_conflicts`), so a panic
/// indicates an internal bug rather than bad user input.
pub(crate) fn split<'a, const R: usize>(
    arena: &'a mut [f32],
    params: &'a [f32],
    reads: [Src; R],
    write: Range<usize>,
) -> ([&'a [f32]; R], &'a mut [f32]) {
    let len = arena.len();
    assert!(
        write.start <= write.end && write.end <= len,
        "arena write out of bounds"
    );
    for r in &reads {
        match r {
            Src::Arena(r) => {
                assert!(r.start <= r.end && r.end <= len, "arena read out of bounds");
                assert!(
                    r.end <= write.start
                        || write.end <= r.start
                        || r.is_empty()
                        || write.is_empty(),
                    "arena read {r:?} aliases write {write:?}"
                );
            }
            Src::Param(r) => {
                assert!(
                    r.start <= r.end && r.end <= params.len(),
                    "param read out of bounds"
                )
            }
        }
    }
    let base = arena.as_mut_ptr();
    // SAFETY: all ranges are within `arena` (checked above); the write range does not
    // intersect any arena read range (checked above), so the `&mut` slice never aliases
    // a shared slice. Shared slices may overlap each other, which is permitted. The
    // returned borrows are tied to `'a`, the lifetime of the exclusive `arena` borrow
    // consumed by this call, so no other access to the arena can coexist with them.
    let out = unsafe { std::slice::from_raw_parts_mut(base.add(write.start), write.len()) };
    let views = reads.map(|r| match r {
        // SAFETY: see above; in bounds and disjoint from `out`.
        Src::Arena(r) => unsafe {
            std::slice::from_raw_parts(base.add(r.start) as *const f32, r.len())
        },
        Src::Param(r) => &params[r],
    });
    (views, out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn disjoint_views_work() {
        let mut arena = vec![1.0, 2.0, 3.0, 4.0];
        let params = vec![10.0];
        let ([a, p], w) = split(
            &mut arena,
            &params,
            [Src::Arena(0..2), Src::Param(0..1)],
            2..4,
        );
        w[0] = a[0] + p[0];
        w[1] = a[1];
        assert_eq!(arena, vec![1.0, 2.0, 11.0, 2.0]);
    }

    #[test]
    #[should_panic(expected = "aliases")]
    fn aliasing_is_rejected() {
        let mut arena = vec![0.0; 4];
        let _ = split(&mut arena, &[], [Src::Arena(0..3)], 2..4);
    }

    #[test]
    #[should_panic(expected = "out of bounds")]
    fn out_of_bounds_is_rejected() {
        let mut arena = vec![0.0; 4];
        let _ = split::<0>(&mut arena, &[], [], 2..5);
    }
}
