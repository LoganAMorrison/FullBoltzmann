pub mod gamma;
pub mod sigma;

use super::ScalarSinglet;
use crate::boltz::helper::hubblet;
use crate::boltz::traits::{FullBoltzmann, SimpleBoltzmann};
use crate::utils::integration::*;
use cyphus_integration::prelude::*;
use cyphus_specfun::bessel::CylBesselK;
use haliax_constants::masses::HIGGS_MASS;
use std::num::FpCategory;

impl ScalarSinglet {
    pub fn new(ms: f64, lam: f64) -> ScalarSinglet {
        ScalarSinglet { ms, lam_hs: lam }
    }
}

impl FullBoltzmann for ScalarSinglet {
    fn dm_mass(&self) -> f64 {
        self.ms
    }
    fn g(&self) -> f64 {
        1.0
    }
    fn feq(&self, x: f64, q: f64) -> f64 {
        let e = (q * q + x * x).sqrt(); // energy / temperature
        1.0 / (e.exp() - 1.0)
    }
    /// Compute the momentum exchange rate between the DM and SM.
    fn gamma_hinv(&self, x: f64) -> f64 {
        let gk_gamma = GaussKronrodIntegratorBuilder::default()
            .epsrel(1e-8)
            .epsabs(0.0)
            .key(2)
            .build();
        let f = |w: f64| self.gamma_integrand(w, x);
        let pre = 1.0 / (48.0 * (self.ms * std::f64::consts::PI).powi(3)) / 8.0;
        let int = gk_gamma.integrate(f, 0.0, f64::INFINITY).val;
        let ht = hubblet(self.ms / x);
        pre * int / ht
    }
    /// Compute sigma*vmol averaged over angles of two incoming DM particles with
    /// three-momenta which have magnitudes k1 and k2.
    fn sigmav(&self, x: f64, q: f64, qt: f64) -> f64 {
        let temp = self.ms / x;
        let k1 = q * temp;
        let k2 = qt * temp;
        let ms2 = self.ms * self.ms;
        let e1 = (k1 * k1 + ms2).sqrt();
        let e2 = (k2 * k2 + ms2).sqrt();

        let mut sum = 0.0;

        for (theta, wgt) in (*GAUSS_LEG_NS).iter().zip((*GAUSS_LEG_WS).iter()) {
            // p1.p2
            let dot = e1 * e2 - k1 * k2 * theta;
            // Moler velocity
            let vmol = (dot * dot - self.ms.powi(4)).sqrt() / (e1 * e2);
            // s = (p1 + p2)^2
            let cme = (2.0 * ms2 + 2.0 * dot).sqrt();
            let t = wgt * vmol * self.sigma_ss(cme);
            sum += if t.classify() == FpCategory::Nan {
                0.0
            } else {
                t
            };
        }
        sum / 2.0
    }
}

impl SimpleBoltzmann for ScalarSinglet {
    fn mass(&self) -> f64 {
        self.ms
    }
    fn thermal_cross_section(&self, x: f64) -> f64 {
        let m = self.ms;
        let denom = 2.0 * x.cyl_bessel_kn_scaled(2);
        let pf = x / (denom * denom);
        let resonance = HIGGS_MASS / m;
        let threshold = 2.0 * HIGGS_MASS / m;

        let singular_points = if threshold > 2.0 {
            if resonance > 2.0 {
                vec![resonance, threshold]
            } else {
                vec![threshold]
            }
        } else {
            vec![]
        };
        let gk_tcs = GaussKronrodIntegratorBuilder::default()
            .singular_points(singular_points)
            .epsrel(1e-8)
            .epsabs(0.0)
            .key(2)
            .build();

        let integrand = |z: f64| -> f64 {
            let z2 = z * z;
            let sig = self.sigma_ss(m * z);
            let kernal = z2 * (z2 - 4.0) * (x * z).cyl_bessel_k1_scaled() * (-x * (z - 2.0)).exp();
            sig * kernal
        };

        pf * gk_tcs.integrate(integrand, 2.0, f64::INFINITY).val
    }
}
