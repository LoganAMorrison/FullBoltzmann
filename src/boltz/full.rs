use crate::utils::derivatives::*;
use cyphus_diffeq::prelude::*;
use haliax_constants::cosmology::M_PLANK;
use haliax_thermal_functions::prelude::*;
use ndarray::prelude::*;
use ndarray::Zip;
use std::f64::consts::PI;

use super::traits::FullBoltzmann;

pub fn gefft(temp: f64) -> f64 {
    sm_sqrt_gstar(temp) * sm_geff(temp).sqrt() / sm_heff(temp) - 1.0
}

pub fn hubblet(temp: f64) -> f64 {
    let h = (4.0 * PI.powi(3) * sm_geff(temp) / 45.0).sqrt() * temp * temp / M_PLANK;
    h / (1.0 + gefft(temp))
}

pub fn compute_dfi<T: FullBoltzmann>(
    i: usize,
    n: usize,
    x: f64,
    f: ArrayView1<f64>,
    feq: ArrayView1<f64>,
    qs: ArrayView1<f64>,
    dq: f64,
    dfi: f64,
    d2fi: f64,
    pre: f64,
    gam: f64,
    gt: f64,
    p: &T,
) -> f64 {
    let qi = qs[i];
    let fi = f[i];
    let feqi = feq[i];

    let mut deriv = 0.0;
    // Integrate the scattering matrix using trapizoid rule
    for k in 0..n {
        let qk = qs[k];
        let feqk = feq[k];
        let fk = f[k];
        let wgt = if k == 0 || k == n - 1 { 0.5 } else { 1.0 };
        deriv += wgt * qk * qk * p.sigmav(x, qi, qk) * (feqi * feqk - fi * fk);
    }
    deriv *= pre * dq;
    // We skip these terms at the end since df/dx(qf) = 0.0;
    if i != n - 1 {
        // Compute the elastic scattering term
        let xq = (x * x + qi * qi).sqrt();
        let dfi = dfi;
        deriv += gam / (2.0 * x) * (xq * d2fi + (qi + 2.0 * xq / qi + qi / xq) * dfi + 3.0 * fi);
        // Compute the expansion term
        deriv += gt * qi / x * dfi;
    }
    deriv
}

pub fn compute_j_dfi<T: FullBoltzmann>(
    i: usize,
    j: usize,
    n: usize,
    x: f64,
    f: ArrayView1<f64>,
    qs: ArrayView1<f64>,
    dq: f64,
    j_df: f64,
    j_d2f: f64,
    pre: f64,
    gam: f64,
    gt: f64,
    p: &T,
) -> f64 {
    let mut jac = 0.0;
    let qi = qs[i];
    // Integrate the jacobian of the scattering matrix using
    // trapizoid rule
    for k in 0..n {
        jac += -qs[k].powi(2)
            * p.sigmav(x, qi, qs[k])
            * if k == 0 || k == n - 1 { 0.5 } else { 1.0 }
            * pre
            * dq
            * if i == j && i == k {
                f[k] + f[i]
            } else if i == j {
                f[k]
            } else if i == k {
                f[i]
            } else {
                0.0
            }
    }
    if i != n - 1 {
        let xq = (x * x + qi * qi).sqrt();
        jac += gam / (2.0 * x)
            * (xq * j_d2f
                + (qi + 2.0 * xq / qi + qi / xq) * j_df
                + 3.0 * if i == j { 1.0 } else { 0.0 });
        jac += gt * qi / x * j_df;
    }
    jac
}

pub fn integrate_full_boltzmann<T: FullBoltzmann + Sync>(
    model: T,
    n: usize,
    xspan: (f64, f64),
) -> OdeSolution {
    let qs: Array1<f64> = Array::linspace(1e-6, 50.0, n);
    let dq = qs[1] - qs[0];
    // Weight vector for integration. We will use trapizoid rule.
    let mut wgts = Array1::<f64>::ones(n);
    wgts[0] /= 0.5;
    wgts[n - 1] /= 0.5;

    // Extract parameters that don't change
    let mx = model.dm_mass();
    let g = model.g();
    // Construct the jacobian of the df
    let j_df = jac_first_deriv_vec(n, dq);
    let j_d2f = jac_second_deriv_vec(n, dq);

    // Construct function for RHS of ODE.
    let dudt = |deriv: ArrayViewMut1<f64>, f: ArrayView1<f64>, x: f64, p: &T| {
        let temp = mx / x;
        let ht = hubblet(temp);
        let pre = mx.powi(3) * g / (ht * x.powi(4) * 2.0 * std::f64::consts::PI.powi(2));
        let feq = qs.mapv(|q| p.feq(x, q));
        let gam = p.gamma_hinv(x);
        let gt = gefft(temp);
        let df = first_deriv_vec(f.view(), dq);
        let d2f = second_deriv_vec(f.view(), dq);

        // Construct the derivative in parallel
        Zip::indexed(deriv).par_apply(|i, d| {
            *d = compute_dfi(
                i,
                n,
                x,
                f.view(),
                feq.view(),
                qs.view(),
                dq,
                df[i],
                d2f[i],
                pre,
                gam,
                gt,
                p,
            );
        });
    };

    // Construct function for the Jacobian of the RHS of ODE.
    let dfdu = |jac: ArrayViewMut2<f64>, f: ArrayView1<f64>, x: f64, p: &T| {
        let temp = mx / x;
        let ht = hubblet(temp);
        let gt = gefft(temp);
        let pre = mx.powi(3) * g / (ht * x.powi(4) * 2.0 * std::f64::consts::PI.powi(2));
        let gam = p.gamma_hinv(x);

        Zip::indexed(jac).par_apply(|(i, j), d| {
            *d = compute_j_dfi(
                i,
                j,
                n,
                x,
                f.view(),
                qs.view(),
                dq,
                j_df[[i, j]],
                j_d2f[[i, j]],
                pre,
                gam,
                gt,
                p,
            );
        });
    };

    // Construct the initial condition (i.e. the initial phase space)
    let mut finit = Array1::<f64>::zeros(n);
    for (i, q) in qs.iter().enumerate() {
        finit[i] = model.feq(xspan.0, *q);
    }

    let mut integrator = OdeIntegratorBuilder::default(&dudt, finit, xspan, Radau5, model)
        .abstol(1e-100)
        .reltol(1e-6)
        .dfdu(&dfdu)
        .build();
    integrator.integrate();
    integrator.sol
}
