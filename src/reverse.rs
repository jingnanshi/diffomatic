use std::borrow::BorrowMut;
use std::ops::{Add, Div, Mul, Neg, Sub, Index, AddAssign, MulAssign};
use std::cell::{Ref, RefCell};
use std::fmt::{Debug, Formatter};
use std::rc::Rc;

/// Node in the computation graph
#[derive(Clone, Copy)]
pub struct Node {
    /// partials of the two parents with respect to this node
    pub partials: [f64; 2],
    /// Parents to this node on the computation graph
    pub parents: [usize; 2],
}

/// Tape holding the computation graph
pub struct Tape {
    pub vars: RefCell<Vec<Node>>,
}

impl Tape {
    /// Create a new tape
    pub fn new() -> Tape {
        Tape {
            vars: RefCell::new(Vec::<Node>::new())
        }
    }

    /// Add a new (input) variable on the tape
    pub fn var<'t>(&'t self, value: f64) -> Var<'t> {
        let len = self.vars.borrow().len();
        self.vars.borrow_mut().push(
            Node {
                partials: [0.0, 0.0],
                // for a single (input) variable, we point the parents to itself
                parents: [len, len],
            }
        );
        Var {
            tape: self,
            index: len,
            v: value,
        }
    }
}

/// Variable for computations
#[derive(Clone, Copy)]
pub struct Var<'t> {
    /// Pointer to the tape holding the corresponding node
    pub tape: &'t Tape,
    /// Index of the node in the tape
    pub index: usize,
    /// Value
    pub v: f64,
}

impl<'t> Add for Var<'t> {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        // add values
        // add entry (of the new variable) on tape
        todo!()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn addition() {
        todo!()
        // building the computational graph

        // back prop

        // get gradients
    }
}
