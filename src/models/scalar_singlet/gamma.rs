use super::ScalarSinglet;
use haliax_constants::prelude::*;

impl ScalarSinglet {
    pub(super) fn gamma_integrand_sf_sf(&self, w: f64, mf: f64, ncol: f64) -> f64 {
        let ms = self.ms;
        let mh = HIGGS_MASS;

        if w >= mf {
            let temp1: f64 = mf.powi(2);
            let temp2: f64 = mh.powi(2);
            let temp3: f64 = ms.powi(2);
            let temp4: f64 = mh.powi(-2);
            let temp5: f64 = 2.0 * w;
            let temp6: f64 = ms + temp5;
            let temp7: f64 = -4.0 * temp3;
            let temp8: f64 = temp2 + temp7;
            let temp9: f64 = temp1 * temp8;
            let temp10: f64 = w.powi(2);
            let temp11: f64 = 4.0 * ms * temp10;
            let temp12: f64 = temp2 * temp6;
            let temp13: f64 = temp11 + temp12;
            let temp14: f64 = ms * temp13;
            let temp15: f64 = temp14 + temp9;

            (self.lam_hs.powi(2)
                * ncol
                * temp1
                * ((4.0 * (-4.0 * temp1 + temp2) * temp3 * temp4 * (mf - w) * (mf + w)) / temp15
                    + ((temp15 * temp4) / (temp1 + ms * temp6)).ln()))
                / 2.0
        } else {
            0.0
        }
    }
    pub(super) fn gamma_integrand(&self, w: f64, x: f64) -> f64 {
        let temp = x / self.ms;
        let temp_fac = 1.0 / (2.0 * temp) / (1.0 + (w / temp).cosh());
        let mut sum = 0.0;

        sum += self.gamma_integrand_sf_sf(w, TOP_QUARK_MASS, 3.0);
        sum += self.gamma_integrand_sf_sf(w, CHARM_QUARK_MASS, 3.0);
        sum += self.gamma_integrand_sf_sf(w, UP_QUARK_MASS, 3.0);
        sum += self.gamma_integrand_sf_sf(w, DOWN_QUARK_MASS, 3.0);
        sum += self.gamma_integrand_sf_sf(w, STRANGE_QUARK_MASS, 3.0);
        sum += self.gamma_integrand_sf_sf(w, BOTTOM_QUARK_MASS, 3.0);
        sum += self.gamma_integrand_sf_sf(w, ELECTRON_MASS, 0.0);
        sum += self.gamma_integrand_sf_sf(w, MUON_MASS, 0.0);
        sum += self.gamma_integrand_sf_sf(w, TAU_MASS, 0.0);
        temp_fac * sum
    }
}
