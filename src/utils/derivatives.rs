//! This submodule contains various function for computing the finite
//! difference derivatives of a discrete vector:
//!     df/dq     ~ sum(a[i] f[i])
//!     d^2f/dq^2 ~ sum(b[i] f[i])
//! where the coefficients `a` and `b` are determined by the stencil and
//! derivative order. We use the following stencils:
//!
//! for first derivatives:
//!     left-most point:          s = [0, 1, 2, 3]      (forward)
//!     next-to-left-most point:  s = [-1, 0, 1, 2]     (forward)
//!     next-to-right-most point: s = [-2, -1, 0, 1]    (backward)
//!     right-most point:         s = [-3, -2, -1, 0]   (backward)
//!     all other:                s = [-2, -1, 0, 1, 2] (central)
//! for second derivatives:
//!     left-most point:          s = [0, 1, 2, 3, 4]     (forward)
//!     next-to-left-most point:  s = [-1, 0, 1, 2, 3]    (forward)
//!     next-to-right-most point: s = [-3, -2, -1, 0, 1]  (backward)
//!     right-most point:         s = [-4, -3, -2, -1, 0] (backward)
//!     all other:                s = [-2, -1, 0, 1, 2]   (central)
//! These stencils lead to the following finite-difference coefficients:
//!     left-most point:          a[0]=-11/6, a[1]=3, a[2]=3/2, a[3]=1/3
//!     next-to-left-most point:  a[0]=-1/3, a[1]=-1/2, a[2]=1, a[3]=-1/6
//!     next-to-right-most point:
//!     right-most point:         
//!     all other:                
//!     
//!
use ndarray::prelude::*;

/// Construct the first derivative vector from `v` using a 5-pt stencil.
pub fn first_deriv_vec(v: ArrayView1<f64>, h: f64) -> Array1<f64> {
    let n = v.len();
    let mut dv = Array1::<f64>::zeros(n);
    let den = 6.0 * h;

    // Use forward derivatives for first two entries
    dv[0] = array![-11.0, 18.0, -9.0, 2.0].dot(&v.slice(s![..4])) / den;
    dv[1] = array![-2.0, -3.0, 6.0, -1.0].dot(&v.slice(s![..4])) / den;

    // Use backwards derivatives for last two entries
    dv[n - 2] = array![1.0, -6.0, 3.0, 2.0].dot(&v.slice(s![-4..])) / den;
    dv[n - 1] = array![-2.0, 9.0, -18.0, 11.0].dot(&v.slice(s![-4..])) / den;

    // use central difference of all other points
    for i in 2..(n - 2) {
        dv[i] = array![1.0, -8.0, 0.0, 8.0, -1.0].dot(&v.slice(s![i - 2..i + 3])) / (2.0 * den);
    }

    dv
}

/// Construct the first derivative vector from `v` using a 5-pt stencil.
pub fn jac_first_deriv_vec(n: usize, h: f64) -> Array2<f64> {
    let mut dv = Array2::<f64>::zeros((n, n));
    let den = 6.0 * h;

    // Use forward derivatives for first two entries
    dv.slice_mut(s![0, ..4])
        .assign(&(array![-11.0, 18.0, -9.0, 2.0] / den));
    dv.slice_mut(s![1, ..4])
        .assign(&(array![-2.0, -3.0, 6.0, -1.0] / den));

    // Use backwards derivatives for last two entries
    dv.slice_mut(s![n - 2, -4..])
        .assign(&(array![1.0, -6.0, 3.0, 2.0] / den));
    dv.slice_mut(s![n - 1, -4..])
        .assign(&(array![-2.0, 9.0, -18.0, 11.0] / den));

    // use central difference of all other points
    for i in 2..(n - 2) {
        dv.slice_mut(s![i, i - 2..i + 3])
            .assign(&(array![1.0, -8.0, 0.0, 8.0, -1.0] / (2.0 * den)));
    }

    dv
}

/// Construct the second derivative vector from `v` using a 5-pt stencil.
pub fn second_deriv_vec(v: ArrayView1<f64>, h: f64) -> Array1<f64> {
    let n = v.len();

    let mut dv = Array1::<f64>::zeros(n);
    let den = 12.0 * h * h;
    // Use forward derivatives for first two entries
    dv[0] = array![35.0, -104.0, 114.0, -56.0, 11.0,].dot(&v.slice(s![..5])) / den;
    dv[1] = array![11.0, -20.0, 6.0, 4.0, -1.0,].dot(&v.slice(s![..5])) / den;

    // Use backwards derivatives for last two entries
    dv[n - 2] = array![-1.0, 4.0, 6.0, -20.0, 11.0].dot(&v.slice(s![-5..])) / den;
    dv[n - 1] = array![11.0, -56.0, 114.0, -104.0, 35.0,].dot(&v.slice(s![-5..])) / den;

    // use central difference of all other points
    for i in 2..(n - 2) {
        dv[i] = array![-1.0, 16.0, -30.0, 16.0, -1.0,].dot(&v.slice(s![i - 2..i + 3])) / den;
    }

    dv
}

/// Construct the second derivative vector from `v` using a 5-pt stencil.
pub fn jac_second_deriv_vec(n: usize, h: f64) -> Array2<f64> {
    let mut dv = Array2::<f64>::zeros((n, n));
    let den = 12.0 * h * h;
    // Use forward derivatives for first two entries
    dv.slice_mut(s![0, ..5])
        .assign(&(array![35.0, -104.0, 114.0, -56.0, 11.0,] / den));
    dv.slice_mut(s![0, ..5])
        .assign(&(array![11.0, -20.0, 6.0, 4.0, -1.0,] / den));

    // Use backwards derivatives for last two entries
    dv.slice_mut(s![n - 2, -5..])
        .assign(&(array![-1.0, 4.0, 6.0, -20.0, 11.0] / den));
    dv.slice_mut(s![n - 1, -5..])
        .assign(&(array![11.0, -56.0, 114.0, -104.0, 35.0,] / den));

    // use central difference of all other points
    for i in 2..(n - 2) {
        dv.slice_mut(s![i, i - 2..i + 3])
            .assign(&(array![-1.0, 16.0, -30.0, 16.0, -1.0,] / den));
    }

    dv
}
