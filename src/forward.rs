use std::fmt::{Debug, Formatter};
use std::ops::{Add, Div, Mul, Neg, Sub, Index, AddAssign, MulAssign};
use na::{Dim, OMatrix, OVector, DefaultAllocator, DMatrix, SMatrix};
use na::allocator::Allocator;
use num_traits::{Float, Zero};

extern crate nalgebra as na;

/// A scalar dual number type
#[derive(Clone, Copy)]
pub struct DualScalar {
    pub v: f64,
    pub dv: f64,
}

// Traits
impl PartialEq for DualScalar {
    fn eq(&self, other: &Self) -> bool {
        self.v == other.v
    }
}

impl Debug for DualScalar {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "(v, dv) = ({:?}, {:?})", self.v, self.dv)
    }
}

impl Zero for DualScalar {
    fn zero() -> Self {
        DualScalar {
            v: 0.0,
            dv: 0.0,
        }
    }

    fn is_zero(&self) -> bool {
        self.v.is_zero()
    }
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

impl AddAssign for DualScalar {
    fn add_assign(&mut self, rhs: Self) {
        *self = Self {
            v: self.v + rhs.v,
            dv: self.dv + rhs.dv,
        };
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

impl MulAssign for DualScalar {
    fn mul_assign(&mut self, rhs: Self) {
        *self = Self {
            v: self.v * rhs.v,
            dv: self.dv * rhs.v + self.v * rhs.dv,
        };
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

/// Evaluate the Jacobian of function f: R^N -> R^M
/// The Jacobian will be a M-by-N matrix
pub fn jacobian<F, const N: usize, const M: usize>(func: F, x0: &[f64]) -> SMatrix<f64, M, N>
    where F: Fn(&[DualScalar]) -> Vec<DualScalar>,
{
    // To get all the partials, we set each var to have dv=1
    // and the others dv=0, and pass them through the function
    let mut jacobian: SMatrix<f64, M, N> = SMatrix::zeros();
    let mut inputs: Vec<DualScalar> = x0.iter().map(|&v| DualScalar { v: v, dv: 0. }).collect();

    // every time we call the func we can get one column of partials
    for (i, mut col) in jacobian.column_iter_mut().enumerate() {
        inputs[i].dv = 1.;
        let col_result = func(&inputs);
        for j in 0..M {
            col[j] = col_result[j].dv;
        }
        inputs[i].dv = 0.;
    }

    return jacobian;
}

#[cfg(test)]
mod tests {
    use super::*;
    use float_cmp::*;

    #[test]
    fn derivative_test() {
        let f1_test = |x: DualScalar| x * x;
        let f1_result: f64 = derivative(f1_test, 2.0);
        assert!(approx_eq!(f64, f1_result, 4.0, ulps=5));
    }

    #[test]
    fn gradient_test() {
        let f_test = |x: &[DualScalar]| x[0] + x[1];
        let f_result: Vec<f64> = gradient(f_test, vec![1., 2.].as_slice());
        assert!(approx_eq!(f64, f_result[0], 1.0, ulps=5));
        assert!(approx_eq!(f64, f_result[1], 1.0, ulps=5));
    }

    #[test]
    fn jacobian_test() {
        let f_test = |x: &[DualScalar]| {
            vec![x[0] * x[0] * x[1], x[0] + x[1]]
        };
        let f_result: SMatrix<f64, 2, 2> = jacobian(f_test, vec![1., 2.].as_slice());
        assert!(approx_eq!(f64, f_result[(0,0)], 4.0, ulps=5));
        assert!(approx_eq!(f64, f_result[(0,1)], 1.0, ulps=5));
        assert!(approx_eq!(f64, f_result[(1,0)], 1.0, ulps=5));
        assert!(approx_eq!(f64, f_result[(1,1)], 1.0, ulps=5));
    }
}
