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
    pub fn var(&self, value: f64) -> Var {
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

    /// Add a new node to the tape, where the node represents
    /// the result from a unary operation
    pub fn unary_op(&self, partial: f64,
                    index: usize, new_value: f64) -> Var {
        let len = self.vars.borrow().len();
        self.vars.borrow_mut().push(
            Node {
                partials: [partial, 0.0],
                // only the left index matters; the right index points to itself
                parents: [index, len],
            }
        );
        Var {
            tape: self,
            index: len,
            v: new_value,
        }
    }

    /// Add a new node to the tape, where the node represents
    /// the result from a binary operation
    pub fn binary_op(&self, lhs_partial: f64, rhs_partial: f64,
                     lhs_index: usize, rhs_index: usize, new_value: f64) -> Var {
        let len = self.vars.borrow().len();
        self.vars.borrow_mut().push(
            Node {
                partials: [lhs_partial, rhs_partial],
                // for a single (input) variable, we point the parents to itself
                parents: [lhs_index, rhs_index],
            }
        );
        Var {
            tape: self,
            index: len,
            v: new_value,
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
        self.tape.binary_op(1.0, 1.0,
                            self.index, rhs.index, self.v + rhs.v)
    }
}

impl<'t> Mul for Var<'t> {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        self.tape.binary_op(rhs.v, self.v,
                            self.index, rhs.index, self.v * rhs.v)
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
