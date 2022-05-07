use std::borrow::BorrowMut;
use std::ops::{Add, Div, Mul, Neg, Sub, Index, AddAssign, MulAssign};
use std::cell::RefCell;
use std::fmt::{Debug, Formatter};
use std::rc::Rc;

// https://ricardomartins.cc/2016/06/08/interior-mutability
// https://github.com/Rufflewind/revad/blob/eb3978b3ccdfa8189f3ff59d1ecee71f51c33fd7/revad.py

type VarRef = Rc<RefCell<Var>>;

#[derive(Clone)]
/// Node for reverse mode
pub struct Var {
    /// Value
    pub v: f64,
    /// Adjoint of this Var
    pub dv: f64,
    /// References to the parent nodes, with a weight (partial) value
    pub parents: Vec<(f64, VarRef)>,
}

impl Debug for Var {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "v: {}, dv: {:?}, parents: {:?}", self.v, self.dv, self.parents)
    }
}

impl Add for Var {
    type Output = Var;

    fn add(self, rhs: Self) -> Self::Output {
        let new_v = self.v + rhs.v;
        let self_rc = Rc::new(RefCell::new(self));
        let rhs_rc = Rc::new(RefCell::new(rhs));

        Var {
            v: new_v,
            dv: 0.0,
            parents: vec![(1.0, self_rc), (1.0, rhs_rc)],
        }
    }
}

impl Var {
    /// Create a new node
    pub fn new(v: f64) -> Var {
        Var {
            v,
            dv: 0.0,
            parents: Vec::<(f64, VarRef)>::new(),
        }
    }

    /// Backprop
    pub fn backprop(&mut self, adjoint: f64) {
        // backprop
        self.dv += adjoint;
        for parent in self.parents.iter() {
            println!("parent: {:?}", &parent);
            (*parent.1).borrow_mut().backprop(parent.0 * adjoint)
        }
    }

    /// Call this to get gradients
    pub fn grad(&mut self) {
        self.backprop(1.0);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn addition() {
        let x: Var = Var::new(1.0);
        let y: Var = Var::new(2.0);
        let mut z = x + y;
        z.grad();
        println!("z value: {:?}", &z);
    }
}
