pub mod gamma;
pub mod sigma;
pub mod width;

use super::DipoleDm;
use crate::boltz::helper::hubblet;
use crate::boltz::traits::FullBoltzmann;
use cyphus_integration::prelude::*;
use gamma::*;

impl DipoleDm {
    pub fn new(mx: f64, dm: f64, lam: f64, ce: f64, cm: f64) -> DipoleDm {
        let gk_gamma = GaussKronrodIntegratorBuilder::default()
            .epsrel(1e-8)
            .epsabs(0.0)
            .key(2)
            .singular_points(vec![1.0])
            .build();
        let gk_sig = GaussKronrodIntegratorBuilder::default()
            .epsrel(1e-8)
            .epsabs(0.0)
            .key(2)
            .build();

        let udm = dm / mx;
        let ulam = lam / mx;
        let width_h = DipoleDm::compute_width_h(mx, udm, ulam, ce, cm);

        DipoleDm {
            mx,
            dm,
            lam,
            ce,
            cm,
            width_h,
            gam_coeff_ss: gamma_integrand_ss_coeff(mx, udm, ulam, ce, cm),
            gam_coeff_tt: gamma_integrand_ss_coeff(mx, udm, ulam, ce, cm),
            gam_coeff_st: gamma_integrand_ss_coeff(mx, udm, ulam, ce, cm),
        }
    }
}

impl FullBoltzmann for DipoleDm {
    fn gamma_hinv(&self, x: f64) -> f64 {
        let pre = 1.0 / (48.0 * (std::f64::consts::PI * self.mx).powi(3) * 2.0 * self.mx / x);
        let f = |w: f64| self.gamma_integrand(w, x);
        let gk_gamma = GaussKronrodIntegratorBuilder::default()
            .epsrel(1e-8)
            .epsabs(0.0)
            .key(2)
            .singular_points(vec![1.0])
            .build();
        let gam = gk_gamma.integrate(&f, 0.0, f64::INFINITY).val;

        pre * gam / hubblet(self.mx / x)
    }
    fn feq(&self, x: f64, q: f64) -> f64 {
        let e = (q * q + x * x).sqrt(); // energy / temperature
        1.0 / (e.exp() + 1.0)
    }
    fn sigmav(&self, x: f64, q: f64, qt: f64) -> f64 {
        let temp = self.mx / x;
        let k = temp * q;
        let kt = temp * qt;
        let gk_sig = GaussKronrodIntegratorBuilder::default()
            .epsrel(1e-8)
            .epsabs(0.0)
            .key(2)
            .build();
        let f = |theta: f64| -> f64 {
            let e1 = (k * k + self.mx * self.mx).sqrt();
            let e2 = (kt * kt + self.mx * self.mx).sqrt();
            // (E1;k).(E2;kt)
            let dot = e1 * e2 - k * kt * theta;

            let vmol = (dot * dot - self.mx.powi(4)).sqrt() / (e1 * e2);
            let cme = 2.0 * self.mx * self.mx + 2.0 * dot;
            self.sigma_11_to_gg(cme) * vmol
        };
        gk_sig.integrate(f, -1.0, 1.0).val / 2.0
    }
    fn dm_mass(&self) -> f64 {
        self.mx
    }
    fn g(&self) -> f64 {
        2.0
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_gamma() {
        let gk_gamma = GaussKronrodIntegratorBuilder::default()
            .epsrel(1e-8)
            .epsabs(0.0)
            .key(2)
            .singular_points(vec![1.0])
            .build();
        let model = DipoleDm::new(100.0, 0.1, 1e4, 1.0, 1.0);
        let x = 1.0;
        //let pre = 1.0 / (48.0 * (std::f64::consts::PI * model.ml).powi(3) * 2.0 * model.ml / x);
        let f = |w: f64| model.gamma_integrand(w, x);
        let gam = gk_gamma.integrate(&f, 0.0, f64::INFINITY);
        println!("{:e}", model.width_h);
        println!("{:e}", gam.val);
    }
}
