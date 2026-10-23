//! Static tensor-buffer planning.
//!
//! A [`Plan`] fixes, for a graph, a mode and a maximum batch size, where every
//! runtime tensor lives inside one f32 arena. Planning happens *before* any large
//! buffer is allocated so that budgets can be enforced up front.
//!
//! Schedule: active nodes run forward at steps `0..A` in topological order. In
//! training mode, backward for the node at forward step `t` runs at step `2A-1-t`.
//! Each buffer has an inclusive live interval `[start, end]`. Buffers whose intervals
//! intersect never share memory. Because intervals are inclusive, an operator's
//! output never aliases one of its own operands.
//!
//! The memory claim is limited to *planned tensor-buffer storage*: weights, arena
//! (inputs, outputs, activations, retained values, gradients, parameter gradients,
//! alignment padding) and optimizer state. Loader memory, graph metadata, parsing,
//! thread stacks, allocator bookkeeping, HTTP buffers and process overhead are
//! outside the bound and listed in [`MemoryReport::excluded`].

use crate::error::{Contributor, CoreError, Result};
use crate::graph::{Graph, NodeId, Op};
use serde::{Deserialize, Serialize};

/// Offsets are aligned to 16 f32 elements (64 bytes).
pub const ALIGN_ELEMS: usize = 16;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Mode {
    Inference,
    Training,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum PlannerKind {
    /// Every buffer gets its own region (no reuse). Simple and easy to audit.
    Conservative,
    /// Lifetime-based reuse: buffers whose live intervals do not intersect share memory.
    #[default]
    Reuse,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Category {
    Input,
    Output,
    Activation,
    /// An activation kept alive into the backward pass.
    Retained,
    Gradient,
    ParamGrad,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BufKind {
    Value(NodeId),
    Grad(NodeId),
    ParamGrads,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Buffer {
    pub kind: BufKind,
    pub category: Category,
    pub name: String,
    /// Capacity in f32 elements at `max_batch`.
    pub elems: usize,
    pub start: usize,
    pub end: usize,
    /// Offset in elements inside the arena.
    pub offset: usize,
}

impl Buffer {
    pub fn aligned(&self) -> usize {
        align(self.elems)
    }
    pub fn overlaps_in_time(&self, o: &Buffer) -> bool {
        self.start <= o.end && o.start <= self.end
    }
    pub fn overlaps_in_memory(&self, o: &Buffer) -> bool {
        self.offset < o.offset + o.aligned() && o.offset < self.offset + self.aligned()
    }
}

fn align(n: usize) -> usize {
    n.div_ceil(ALIGN_ELEMS) * ALIGN_ELEMS
}

/// A validated execution plan. Contains no tensor data.
#[derive(Debug, Clone)]
pub struct Plan {
    pub mode: Mode,
    pub planner: PlannerKind,
    pub max_batch: usize,
    pub buffers: Vec<Buffer>,
    pub(crate) value_buf: Vec<Option<usize>>,
    pub(crate) grad_buf: Vec<Option<usize>>,
    pub(crate) param_grad_buf: Option<usize>,
    pub(crate) forward: Vec<NodeId>,
    pub(crate) backward: Vec<NodeId>,
    /// For backward position `i`, `zero_list[zero_ranges[i].0..zero_ranges[i].1]` are
    /// gradient buffers that become live at that step and must be zeroed first.
    pub(crate) zero_ranges: Vec<(usize, usize)>,
    pub(crate) zero_list: Vec<usize>,
    pub arena_elems: usize,
}

impl Plan {
    pub fn new(graph: &Graph, mode: Mode, max_batch: usize, planner: PlannerKind) -> Result<Plan> {
        if max_batch == 0 || max_batch > crate::shape::MAX_DIM {
            return Err(CoreError::Shape(format!(
                "max_batch {max_batch} outside 1..={}",
                crate::shape::MAX_DIM
            )));
        }
        let root = match mode {
            Mode::Inference => graph.output(),
            Mode::Training => graph.loss().ok_or_else(|| {
                CoreError::Graph("graph has no loss; cannot plan training".into())
            })?,
        };
        if mode == Mode::Training && graph.target().is_none() {
            return Err(CoreError::Graph("training requires a target node".into()));
        }
        let active = graph.ancestors(root);
        let forward: Vec<NodeId> = (0..graph.nodes().len()).filter(|&i| active[i]).collect();
        let a = forward.len();
        let mut fstep = vec![usize::MAX; graph.nodes().len()];
        for (t, &n) in forward.iter().enumerate() {
            fstep[n] = t;
        }
        let bstep = |n: NodeId| 2 * a - 1 - fstep[n];
        let training = mode == Mode::Training;
        let runs_backward = |n: NodeId| training && graph.requires_grad(n);
        let backward: Vec<NodeId> = if training {
            forward
                .iter()
                .rev()
                .copied()
                .filter(|&n| runs_backward(n))
                .collect()
        } else {
            Vec::new()
        };
        let end_of_time = if training { 2 * a } else { a };

        let mut buffers = Vec::new();
        let mut value_buf = vec![None; graph.nodes().len()];
        let mut grad_buf = vec![None; graph.nodes().len()];

        // Value buffers.
        for &n in &forward {
            let op = graph.op(n);
            if matches!(op, Op::Param { .. }) {
                continue;
            }
            let elems = graph.shape(n).elements(max_batch)?;
            let start = match op {
                Op::Input { .. } | Op::Target { .. } => 0,
                _ => fstep[n],
            };
            let mut end = fstep[n];
            for &c in &forward {
                let cop = graph.op(c);
                for (i, operand) in cop.input_list().enumerate() {
                    if operand != n {
                        continue;
                    }
                    end = end.max(fstep[c]);
                    if runs_backward(c) && cop.backward_reads_input(i) {
                        end = end.max(bstep(c));
                    }
                }
            }
            if runs_backward(n) && op.backward_reads_output() {
                end = end.max(bstep(n));
            }
            let is_root = n == root;
            if is_root {
                end = end_of_time;
            }
            let category = match op {
                Op::Input { .. } | Op::Target { .. } => Category::Input,
                _ if is_root => Category::Output,
                _ if training && end >= a => Category::Retained,
                _ => Category::Activation,
            };
            value_buf[n] = Some(buffers.len());
            buffers.push(Buffer {
                kind: BufKind::Value(n),
                category,
                name: graph.nodes()[n].name.clone(),
                elems,
                start,
                end,
                offset: 0,
            });
        }

        let mut param_grad_buf = None;
        if training {
            // Activation gradients.
            for &n in &forward {
                if !graph.requires_grad(n) || matches!(graph.op(n), Op::Param { .. }) {
                    continue;
                }
                let mut start = bstep(n);
                for &c in &forward {
                    if runs_backward(c) && graph.op(c).input_list().any(|o| o == n) {
                        start = start.min(bstep(c));
                    }
                }
                grad_buf[n] = Some(buffers.len());
                buffers.push(Buffer {
                    kind: BufKind::Grad(n),
                    category: Category::Gradient,
                    name: format!("grad.{}", graph.nodes()[n].name),
                    elems: graph.shape(n).elements(max_batch)?,
                    start,
                    end: bstep(n),
                    offset: 0,
                });
            }
            if graph.param_count() > 0 {
                param_grad_buf = Some(buffers.len());
                buffers.push(Buffer {
                    kind: BufKind::ParamGrads,
                    category: Category::ParamGrad,
                    name: "grad.params".into(),
                    elems: graph.param_count(),
                    start: 0,
                    end: end_of_time,
                    offset: 0,
                });
            }
        }

        let arena_elems = match planner {
            PlannerKind::Conservative => assign_sequential(&mut buffers)?,
            PlannerKind::Reuse => assign_reuse(&mut buffers)?,
        };
        verify_no_conflicts(&buffers)?;

        // Zero lists for activation gradients, keyed by backward position.
        let mut zero_ranges = Vec::with_capacity(backward.len());
        let mut zero_list = Vec::new();
        for &n in &backward {
            let s = bstep(n);
            let from = zero_list.len();
            for (bi, b) in buffers.iter().enumerate() {
                if matches!(b.kind, BufKind::Grad(_)) && b.start == s {
                    zero_list.push(bi);
                }
            }
            zero_ranges.push((from, zero_list.len()));
        }

        Ok(Plan {
            mode,
            planner,
            max_batch,
            buffers,
            value_buf,
            grad_buf,
            param_grad_buf,
            forward,
            backward,
            zero_ranges,
            zero_list,
            arena_elems,
        })
    }

    pub fn arena_bytes(&self) -> u64 {
        self.arena_elems as u64 * 4
    }

    /// Element range of a buffer for a given runtime batch size.
    pub(crate) fn range(&self, buf: usize, elems: usize) -> std::ops::Range<usize> {
        let b = &self.buffers[buf];
        debug_assert!(elems <= b.elems);
        b.offset..b.offset + elems
    }

    /// Build the byte accounting for this plan.
    pub fn report(&self, graph: &Graph, sessions: usize, optimizer_slots: usize) -> MemoryReport {
        let by = |c: Category| -> u64 {
            self.buffers
                .iter()
                .filter(|b| b.category == c)
                .map(|b| b.elems as u64 * 4)
                .sum()
        };
        let padding: u64 = self
            .buffers
            .iter()
            .map(|b| (b.aligned() - b.elems) as u64 * 4)
            .sum();
        let unshared: u64 = self.buffers.iter().map(|b| b.aligned() as u64 * 4).sum();
        let weights = graph.param_count() as u64 * 4;
        let optimizer = if self.mode == Mode::Training {
            weights * optimizer_slots as u64
        } else {
            0
        };
        let arena = self.arena_bytes();
        let sessions_u = sessions.max(1) as u64;
        let total = weights
            .saturating_add(arena.saturating_mul(sessions_u))
            .saturating_add(optimizer);
        let mut largest: Vec<Contributor> = self
            .buffers
            .iter()
            .map(|b| Contributor {
                name: b.name.clone(),
                bytes: b.aligned() as u64 * 4,
            })
            .collect();
        largest.sort_by(|a, b| b.bytes.cmp(&a.bytes).then(a.name.cmp(&b.name)));
        largest.truncate(5);
        MemoryReport {
            mode: self.mode,
            planner: self.planner,
            max_batch: self.max_batch,
            sessions: sessions.max(1),
            shared_weights_bytes: weights,
            input_bytes: by(Category::Input),
            output_bytes: by(Category::Output),
            activation_bytes: by(Category::Activation),
            retained_for_backward_bytes: by(Category::Retained),
            gradient_bytes: by(Category::Gradient),
            param_grad_bytes: by(Category::ParamGrad),
            optimizer_state_bytes: optimizer,
            kernel_scratch_bytes: 0,
            alignment_padding_bytes: padding,
            sum_of_buffers_bytes: unshared,
            arena_bytes_per_session: arena,
            total_planned_bytes: total,
            largest_buffers: largest,
            excluded: EXCLUDED.iter().map(|s| s.to_string()).collect(),
        }
    }
}

const EXCLUDED: &[&str] = &[
    "package/checkpoint loading buffers",
    "graph and manifest metadata",
    "CSV parsing and data loading buffers",
    "thread stacks",
    "allocator bookkeeping and fragmentation",
    "HTTP request/response buffers",
    "process and runtime overhead",
];

/// Byte accounting for one plan. Category fields are sums of buffer sizes *before*
/// reuse; `arena_bytes_per_session` is the actual arena capacity *after* reuse.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MemoryReport {
    pub mode: Mode,
    pub planner: PlannerKind,
    pub max_batch: usize,
    pub sessions: usize,
    pub shared_weights_bytes: u64,
    pub input_bytes: u64,
    pub output_bytes: u64,
    pub activation_bytes: u64,
    pub retained_for_backward_bytes: u64,
    pub gradient_bytes: u64,
    pub param_grad_bytes: u64,
    pub optimizer_state_bytes: u64,
    pub kernel_scratch_bytes: u64,
    pub alignment_padding_bytes: u64,
    pub sum_of_buffers_bytes: u64,
    pub arena_bytes_per_session: u64,
    /// `shared weights + sessions * arena + optimizer state`.
    pub total_planned_bytes: u64,
    pub largest_buffers: Vec<Contributor>,
    pub excluded: Vec<String>,
}

impl MemoryReport {
    /// Reject the plan if its planned tensor storage exceeds `budget` bytes.
    pub fn check_budget(&self, budget: u64) -> Result<()> {
        if self.total_planned_bytes <= budget {
            return Ok(());
        }
        let mut contributors = vec![
            Contributor {
                name: format!("arena x {} session(s)", self.sessions),
                bytes: self.arena_bytes_per_session * self.sessions as u64,
            },
            Contributor {
                name: "shared weights".into(),
                bytes: self.shared_weights_bytes,
            },
        ];
        if self.optimizer_state_bytes > 0 {
            contributors.push(Contributor {
                name: "optimizer state".into(),
                bytes: self.optimizer_state_bytes,
            });
        }
        contributors.sort_by(|a, b| b.bytes.cmp(&a.bytes));
        contributors.extend(self.largest_buffers.iter().take(3).cloned());
        Err(CoreError::BudgetExceeded {
            requested: self.total_planned_bytes,
            available: budget,
            contributors,
        })
    }
}

fn assign_sequential(buffers: &mut [Buffer]) -> Result<usize> {
    let mut off: usize = 0;
    for b in buffers.iter_mut() {
        b.offset = off;
        off = off
            .checked_add(b.aligned())
            .ok_or(CoreError::Overflow("arena size"))?;
    }
    Ok(off)
}

/// Greedy offset assignment: largest buffers first, each placed at the lowest offset
/// that does not collide with an already-placed buffer whose interval intersects.
fn assign_reuse(buffers: &mut [Buffer]) -> Result<usize> {
    let mut order: Vec<usize> = (0..buffers.len()).collect();
    order.sort_by(|&x, &y| {
        buffers[y]
            .aligned()
            .cmp(&buffers[x].aligned())
            .then(buffers[x].start.cmp(&buffers[y].start))
            .then(x.cmp(&y))
    });
    let mut placed: Vec<usize> = Vec::with_capacity(buffers.len());
    let mut arena: usize = 0;
    for &i in &order {
        let mut conflicts: Vec<(usize, usize)> = placed
            .iter()
            .filter(|&&p| buffers[p].overlaps_in_time(&buffers[i]))
            .map(|&p| (buffers[p].offset, buffers[p].offset + buffers[p].aligned()))
            .collect();
        conflicts.sort_unstable();
        let need = buffers[i].aligned();
        let mut candidate = 0usize;
        for (lo, hi) in conflicts {
            let fits_before = candidate
                .checked_add(need)
                .ok_or(CoreError::Overflow("arena size"))?
                <= lo;
            if fits_before {
                break;
            }
            candidate = candidate.max(hi);
        }
        buffers[i].offset = candidate;
        arena = arena.max(
            candidate
                .checked_add(need)
                .ok_or(CoreError::Overflow("arena size"))?,
        );
        placed.push(i);
    }
    Ok(arena)
}

/// Defensive check run on every plan: live buffers never share memory.
pub fn verify_no_conflicts(buffers: &[Buffer]) -> Result<()> {
    for (i, a) in buffers.iter().enumerate() {
        for b in &buffers[i + 1..] {
            if a.overlaps_in_time(b) && a.overlaps_in_memory(b) {
                return Err(CoreError::Graph(format!(
                    "internal planner error: '{}' and '{}' overlap while both live",
                    a.name, b.name
                )));
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    fn buf(elems: usize, start: usize, end: usize) -> Buffer {
        Buffer {
            kind: BufKind::ParamGrads,
            category: Category::Activation,
            name: format!("b{elems}_{start}_{end}"),
            elems,
            start,
            end,
            offset: 0,
        }
    }

    proptest! {
        #[test]
        fn reuse_planner_never_overlaps_live_buffers(
            specs in proptest::collection::vec((1usize..500, 0usize..40, 0usize..20), 1..60)
        ) {
            let mut bufs: Vec<Buffer> =
                specs.iter().map(|&(e, s, len)| buf(e, s, s + len)).collect();
            let arena = assign_reuse(&mut bufs).unwrap();
            prop_assert!(verify_no_conflicts(&bufs).is_ok());
            let total: usize = bufs.iter().map(|b| b.aligned()).sum();
            prop_assert!(arena <= total);
            for b in &bufs {
                prop_assert!(b.offset + b.aligned() <= arena);
                prop_assert_eq!(b.offset % ALIGN_ELEMS, 0);
            }
            // Lower bound: the peak simultaneous live size.
            let horizon = bufs.iter().map(|b| b.end).max().unwrap();
            for t in 0..=horizon {
                let live: usize = bufs.iter().filter(|b| b.start <= t && t <= b.end).map(|b| b.aligned()).sum();
                prop_assert!(arena >= live);
            }
        }
    }

    #[test]
    fn disjoint_lifetimes_share_memory() {
        let mut bufs = vec![buf(100, 0, 1), buf(100, 2, 3)];
        assert_eq!(assign_reuse(&mut bufs).unwrap(), align(100));
        let mut bufs = vec![buf(100, 0, 2), buf(100, 2, 3)];
        assert_eq!(assign_reuse(&mut bufs).unwrap(), 2 * align(100));
    }
}
