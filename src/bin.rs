use diff_lib::{DualScalar, derivative, gradient};


pub fn main() {
    // API
    //
    // derivative(f, x)
    let f1_test = |x : DualScalar| x * x;
    let f1_result : f64 = derivative(f1_test, 2.0);
    println!("Derivative of f(x) = x^2 at {} is {}", 2.0, f1_result);

    // gradient(f, x)
    let f2_test = | x : &[DualScalar] | x[0] + x[1];
    let f2_result : Vec<f64> = gradient( f2_test, vec![1., 2.].as_slice());
    println!("Gradient of f(x,y) = x + y at ({}, {}) is {:?}", 1.0, 2.0, f2_result);

    // another gradient
    let f2_test = | x : &[DualScalar] | x[0] * x[1] + x[1] * x[1];
    let f2_result : Vec<f64> = gradient( f2_test, vec![1., 2.].as_slice());
    println!("Gradient of f(x,y) = x * y + y^2 at ({}, {}) is {:?}", 1.0, 2.0, f2_result);
}