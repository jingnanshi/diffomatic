use std::borrow::{Borrow, BorrowMut};
use std::ops::{Add, Div, Mul, Neg, Sub, Index, AddAssign, MulAssign};
use std::cell::{Ref, RefCell};
use std::fmt::{Debug, Formatter};
use std::rc::Rc;

/// Node in the computation graph
#[derive(Clone, Copy)]
pub struct Node {
    /// partials of the two parents with respect to this node
    pub partials: [f64; 2],
    /// Parents to this node on the computation graph. Parents
    /// in the sense that during forward pass, this node depends
    /// on the parents' nodes.
    pub parents: [usize; 2],
}

/// Tape holding the computation graph
pub struct Tape {
    pub nodes: RefCell<Vec<Node>>,
}

impl Tape {
    /// Create a new tape
    pub fn new() -> Tape {
        Tape {
            nodes: RefCell::new(Vec::<Node>::new())
        }
    }

    /// Add a new (input) variable on the tape
    pub fn var(&self, value: f64) -> Var {
        let len = self.nodes.borrow().len();
        self.nodes.borrow_mut().push(
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
        let len = self.nodes.borrow().len();
        self.nodes.borrow_mut().push(
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
        let len = self.nodes.borrow().len();
        self.nodes.borrow_mut().push(
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

impl Var<'_> {
    /// Perform back propagation
    pub fn backprop(&self) -> Grad {
        // vector storing the gradients
        let tape_len = self.tape.nodes.borrow().len();
        let mut grad = vec![0.0; tape_len];
        grad[self.index] = 1.0;

        // iterate through the tape from back to front
        // because during forward pass, we always store new nodes at the end
        // of the tape, when we do the backward pass we can
        // just incrementally add partial * adjoint
        for i in (0..tape_len).rev() {
            let node = self.tape.nodes.borrow()[i];
            // increment gradient contribution to the left parent
            let lhs_dep = node.parents[0];
            let lhs_partial = node.partials[0];
            grad[lhs_dep] += lhs_partial * grad[i];

            // increment gradient contribution to the right parent
            // note that in cases of unary operations, because
            // partial was set to zero, it won't affect the computation
            let rhs_dep = node.parents[1];
            let rhs_partial = node.partials[1];
            grad[rhs_dep] += rhs_partial * grad[i];
        }

        Grad { grad }
    }
}

impl<'t> Add for Var<'t> {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        self.tape.binary_op(1.0, 1.0,
                            self.index, rhs.index, self.v + rhs.v)
    }
}

impl<'t> Sub for Var<'t> {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        self.tape.binary_op(1.0, -1.0,
                            self.index, rhs.index, self.v - rhs.v)
    }
}

impl<'t> Neg for Var<'t> {
    type Output = Self;

    fn neg(self) -> Self::Output {
        self.tape.unary_op(-1.0, self.index, - self.v)
    }
}


impl<'t> Mul for Var<'t> {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        self.tape.binary_op(rhs.v, self.v,
                            self.index, rhs.index, self.v * rhs.v)
    }
}

impl<'t> Mul<Var<'t>> for f64 {
    type Output = Var<'t>;

    fn mul(self, rhs: Var<'t>) -> Self::Output {
        rhs.tape.unary_op(self, rhs.index, self * rhs.v)
    }
}

impl<'t> Div for Var<'t> {
    type Output = Self;

    fn div(self, rhs: Self) -> Self::Output {
        self.tape.binary_op(1.0 / rhs.v, -self.v / (rhs.v * rhs.v),
                            self.index, rhs.index, self.v / rhs.v)
    }
}

/// Struct holding gradients
#[derive(Debug)]
pub struct Grad {
    pub grad: Vec<f64>,
}

impl Grad {
    /// Get the gradient with respect to a variable
    pub fn wrt(&self, var: Var) -> f64 {
        self.grad[var.index]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use float_cmp::*;

    #[test]
    fn addition() {
        let tape = Tape::new();
        let x = tape.var(1.0);
        let y = tape.var(4.0);
        let z = x + y;
        let grad = z.backprop();
        assert!(approx_eq!(f64, grad.wrt(x), 1.0, ulps=5));
        assert!(approx_eq!(f64, grad.wrt(y), 1.0, ulps=5));
    }

    #[test]
    fn mul() {
        let tape = Tape::new();
        let x = tape.var(1.0);
        let y = tape.var(4.0);
        let z = x * y;
        let grad = z.backprop();
        assert!(approx_eq!(f64, grad.wrt(x), y.v, ulps=5));
        assert!(approx_eq!(f64, grad.wrt(y), x.v, ulps=5));
    }

    #[test]
    fn neg() {
        let tape = Tape::new();
        let x = tape.var(1.0);
        let z = -x;
        let grad = z.backprop();
        assert!(approx_eq!(f64, grad.wrt(x), -1.0, ulps=5));
    }

    #[test]
    fn multiple_multiplications() {
        let tape = Tape::new();
        let x = tape.var(1.0);
        let y = tape.var(1.0);
        let z = -2.0 * x + x * x * x * y;
        let grad = z.backprop();
        assert!(approx_eq!(f64, grad.wrt(x), 1.0, ulps=5));
    }
}
