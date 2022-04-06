use diff_lib::{DualScalar, derivative};


pub fn main() {
    // API
    //
    // gradient(f, x)
    let f_test = |x : DualScalar| x * x;
    let result : f64 = derivative(f_test, 2.0);
    println!("{}", result);
}