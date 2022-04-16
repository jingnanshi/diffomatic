use std::ops::{Add, Div, Mul, Neg, Sub};
use num_traits::Float;

extern crate nalgebra as na;

use na::allocator::Allocator;
use na::dimension::Dim;
use na::{DefaultAllocator, OVector};

/// A scalar dual number type
#[derive(Clone, Copy)]
pub struct DualScalar {
    pub v: f64,
    pub dv: f64,
}

/// A static column-vector dual number type
///
/// D: number of rows
/// S: storage type
#[derive(Clone)]
pub struct DualVector<D: Dim>
    where DefaultAllocator: Allocator<f64, D>,
{
    pub v: nalgebra::OVector<f64, D>,
    pub dv: nalgebra::OVector<f64, D>,
}

/// A static matrix dual number type
///
/// R: number of orows
/// C: number of columns
#[derive(Clone, Copy)]
pub struct DualSMatrix<const R: usize, const C: usize>
{
    pub v: nalgebra::SMatrix<f64, R, C>,
    pub dv: nalgebra::SMatrix<f64, R, C>,
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

impl<R : Dim> Add for DualVector<R>
    where DefaultAllocator: Allocator<f64, R>,
{
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        DualVector {
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

impl<R: Dim> Sub for DualVector<R>
    where DefaultAllocator: Allocator<f64, R>,
{
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        DualVector {
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

// Other functions
impl DualScalar {
    // Get derivative from dual number
    pub fn deriv(&self) -> f64 {
        self.dv.clone()
    }
}

/// Evaluate the derivative
pub fn derivative<F>(func: F, x0: f64) -> f64
    where F: FnOnce(DualScalar) -> DualScalar,
{
    func(DualScalar { v: x0, dv: 1.0 }).deriv()
}
