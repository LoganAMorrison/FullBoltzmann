use super::traits::SimpleBoltzmann;
use cyphus_diffeq::prelude::*;
use haliax_constants::prelude::*;
use haliax_thermal_functions::prelude::*;
use ndarray::prelude::*;

pub fn integrate_simple_boltzmann<T: SimpleBoltzmann>(
    model: T,
    xmin: f64,
    xmax: f64,
) -> OdeSolution {
    let mx = model.mass();
    let dudt = |mut dw: ArrayViewMut1<f64>, w: ArrayView1<f64>, logx: f64, p: &T| {
        let x: f64 = logx.exp();
        let temp: f64 = mx / x;
        let s: f64 = sm_entropy_density(temp);
        let n = neq(temp, mx, 2.0, 1);
        let weq: f64 = (n / s).ln();
        let ww: f64 = w[0];

        let pf: f64 = -(std::f64::consts::PI / 45.0).sqrt() * M_PLANK * sm_sqrt_gstar(temp) * temp;
        let sigmav: f64 = p.thermal_cross_section(x);

        // dW_e / dlogx
        dw[0] = pf * sigmav * (ww.exp() - (2.0 * weq - ww).exp());
    };
    let dfdu = |mut df: ArrayViewMut2<f64>, w: ArrayView1<f64>, logx: f64, p: &T| {
        let x: f64 = logx.exp();
        let temp: f64 = mx / x;
        let s: f64 = sm_entropy_density(temp);

        let n = neq(temp, mx, 2.0, 1);
        let weq: f64 = (n / s).ln();
        let ww: f64 = w[0];

        let pf: f64 = -(std::f64::consts::PI / 45.0).sqrt() * M_PLANK * sm_sqrt_gstar(temp) * temp;
        let sigmav: f64 = p.thermal_cross_section(x);

        // dW_e / dlogx
        df[[0, 0]] = pf * sigmav * (ww.exp() + (2.0 * weq - ww).exp());
    };
    let temp = mx / xmin;
    let n = neq(temp, mx, 2.0, 1);
    let uinit = array![(n / sm_entropy_density(temp)).ln()];
    let tspan = (xmin.ln(), xmax.ln());

    let mut integrator = OdeIntegratorBuilder::default(&dudt, uinit, tspan, Radau5, model)
        .dfdu(&dfdu)
        .reltol(1e-7)
        .abstol(1e-7)
        .build();
    integrator.integrate();
    integrator.sol
}
