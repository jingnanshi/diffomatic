use std::ops::{Add, Div, Mul, Neg, Sub, Index};
use num_traits::Float;

extern crate nalgebra as na;

/// A scalar dual number type
#[derive(Clone, Copy)]
pub struct DualScalar {
    pub v: f64,
    pub dv: f64,
}

// Other functions
impl DualScalar {
    // Get derivative from dual number
    pub fn deriv(&self) -> f64 {
        self.dv.clone()
    }
}

// Addition Rules
impl Add for DualScalar {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        DualScalar {
            v: self.v + rhs.v,
            dv: self.dv + rhs.dv,
        }
    }
}

// Subtraction Rules
impl Sub for DualScalar {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        DualScalar {
            v: self.v - rhs.v,
            dv: self.dv - rhs.dv,
        }
    }
}

// Multiplication Rules
impl Mul for DualScalar {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        DualScalar {
            v: self.v * rhs.v,
            dv: self.dv * rhs.v + self.v * rhs.dv,
        }
    }
}

// Division Rules
impl Div for DualScalar {
    type Output = Self;

    fn div(self, rhs: Self) -> Self::Output {
        DualScalar {
            v: self.v / rhs.v,
            dv: (self.dv * rhs.v - self.v * rhs.dv) / (rhs.v * rhs.v),
        }
    }
}

/// Evaluate the derivative
pub fn derivative<F>(func: F, x0: f64) -> f64
    where F: FnOnce(DualScalar) -> DualScalar,
{
    func(DualScalar { v: x0, dv: 1.0 }).deriv()
}

/// Evaluate the gradient
pub fn gradient<F>(func: F, x0: &[f64]) -> Vec<f64>
    where F: Fn(&[DualScalar]) -> DualScalar,
{
    // To get all the partials, we set each var to have dv=1
    // and the others dv=0, and pass them through the function
    let mut inputs: Vec<DualScalar> = x0.iter().map(|&v| DualScalar { v: v, dv: 0. }).collect();
    (0..x0.len()).map(
        |i| {
            inputs[i].dv = 1.;
            let partial = func(&inputs).deriv();
            inputs[i].dv = 0.;
            partial
        }
    ).collect()
}