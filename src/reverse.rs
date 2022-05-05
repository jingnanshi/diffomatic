use std::borrow::BorrowMut;
use std::ops::{Add, Div, Mul, Neg, Sub, Index, AddAssign, MulAssign};
use std::cell::RefCell;
use std::fmt::{Debug, Formatter};
use std::rc::Rc;

// https://ricardomartins.cc/2016/06/08/interior-mutability
// https://github.com/Rufflewind/revad/blob/eb3978b3ccdfa8189f3ff59d1ecee71f51c33fd7/revad.py

type VarRef = Rc<RefCell<Var>>;

/// Node for reverse mode
pub struct Var {
    /// Value
    pub v: f64,
    /// Adjoint of this Var
    pub dv: Option<f64>,
    /// References to the parent nodes, with a weight (partial) value
    pub parents: Vec<(f64, VarRef)>,
}

impl Debug for Var {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "v: {}, dv: {:?}", self.v, self.dv)
    }
}

impl Add for Var {
    type Output = Var;

    fn add(self, rhs: Self) -> Self::Output {
        let new_v =self.v + rhs.v;
        let self_rc = Rc::new(RefCell::new(self));
        let rhs_rc = Rc::new(RefCell::new(rhs));

        Var {
            v: new_v,
            dv: None,
            parents: vec![(1.0, self_rc), (1.0, rhs_rc)],
        }
    }
}

impl Var {
    /// Create a new node
    pub fn new(v : f64) -> Var {
        Var {
            v,
            dv: None,
            parents: Vec::<(f64, VarRef)>::new()
        }
    }

    /// Calculate adjoint
    pub fn grad(&self) -> f64 {
        match self.dv {
            None => {
                // backprop
                0.0
            },
            Some(grad_value) => grad_value,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn addition() {
        let x : Var = Var::new(1.0);
        let y : Var = Var::new(2.0);
        let z = x + y;
        println!("z value: {:?}", z);
    }
}
