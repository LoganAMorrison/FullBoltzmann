use super::ScalarSinglet;
use haliax_constants::prelude::*;

impl ScalarSinglet {
    #[allow(dead_code)]
    pub(super) fn sigma_ss_ff(&self, cme: f64, mf: f64, ncol: f64) -> f64 {
        if cme > 2.0 * mf && cme > 2.0 * self.ms {
            let s = cme * cme;
            let mh = HIGGS_MASS;
            let temp1: f64 = mf.powi(2);
            let temp2: f64 = 1.0 / s;
            let temp3: f64 = mh.powi(2);
            (self.lam_hs.powi(2)
                * ncol
                * (s - 4.0 * temp1)
                * temp1
                * temp2
                * (1.0 - 4.0 * temp1 * temp2).sqrt())
                / (16.0
                    * std::f64::consts::PI
                    * (1.0 - 4.0 * self.ms * self.ms * temp2).sqrt()
                    * ((-s + temp3).powi(2) + temp3 * HIGGS_WIDTH.powi(2)))
        } else {
            0.0
        }
    }
    #[allow(dead_code)]
    pub(super) fn sigma_ss_ww(&self, cme: f64) -> f64 {
        if cme > 2.0 * W_BOSON_MASS && cme > 2.0 * self.ms {
            let s = cme * cme;
            let mw = W_BOSON_MASS;
            let mh = HIGGS_MASS;

            let temp1: f64 = 1.0 / s;
            let temp2: f64 = mw.powi(2);
            let temp3: f64 = mh.powi(2);
            (ALPHA_EM.powi(2)
                * self.lam_hs.powi(2)
                * std::f64::consts::PI
                * temp1
                * (12.0 * mw.powi(4) + s.powi(2) - 4.0 * s * temp2)
                * (1.0 - 4.0 * temp1 * temp2).sqrt()
                * HIGGS_VEV.powi(4))
                / (16.0
                    * mw.powi(4)
                    * SIN_THETA_WEAK_SQRD.powi(2)
                    * (1.0 - 4.0 * self.ms * self.ms * temp1).sqrt()
                    * ((-s + temp3).powi(2) + temp3 * HIGGS_WIDTH.powi(2)))
        } else {
            0.0
        }
    }
    #[allow(dead_code)]
    pub(super) fn sigma_ss_zz(&self, cme: f64) -> f64 {
        if cme > 2.0 * Z_BOSON_MASS && cme > 2.0 * self.ms {
            let s = cme * cme;
            let temp1: f64 = 1.0 / s;
            let temp2: f64 = Z_BOSON_MASS.powi(2);
            let temp3: f64 = HIGGS_MASS.powi(2);
            (ALPHA_EM.powi(2)
                * self.lam_hs.powi(2)
                * std::f64::consts::PI
                * temp1
                * (12.0 * Z_BOSON_MASS.powi(4) + s.powi(2) - 4.0 * s * temp2)
                * (1.0 - 4.0 * temp1 * temp2).sqrt()
                * HIGGS_VEV.powi(4))
                / (16.0
                    * COS_THETA_WEAK.powi(4)
                    * Z_BOSON_MASS.powi(4)
                    * SIN_THETA_WEAK_SQRD.powi(2)
                    * (1.0 - 4.0 * self.ms * self.ms * temp1).sqrt()
                    * ((-s + temp3).powi(2) + temp3 * HIGGS_WIDTH.powi(2)))
        } else {
            0.0
        }
    }
    #[allow(dead_code)]
    pub(super) fn sigma_ss_hh(&self, cme: f64) -> f64 {
        if cme > 2.0 * HIGGS_MASS && cme > 2.0 * self.ms {
            let s: f64 = cme * cme;
            let temp1: f64 = 1.0 / s;
            let temp2: f64 = self.lam_hs.powi(2);
            let temp3: f64 = HIGGS_MASS.powi(2);
            let temp4: f64 = self.ms.powi(2);
            let temp5: f64 = HIGGS_MASS.powi(4);
            let temp6: f64 = -s;
            let temp7: f64 = temp3 + temp6;
            let temp8: f64 = temp7.powi(2);
            let temp9: f64 = HIGGS_WIDTH.powi(2);
            let temp10: f64 = temp3 * temp9;
            let temp11: f64 = temp10 + temp8;
            let temp12: f64 = 1.0 / temp11;
            let temp13: f64 = -temp3;
            let temp14: f64 = s + temp13;
            let temp15: f64 = HIGGS_VEV.powi(2);
            let temp16: f64 = 2.0 * temp3;
            let temp17: f64 = -4.0 * temp3;
            let temp18: f64 = s + temp17;
            let temp19: f64 = -4.0 * temp4;
            let temp20: f64 = s + temp19;
            let temp21: f64 = temp18 * temp20;
            let temp22: f64 = 1.0 / temp21.sqrt();
            return (temp1
                * temp2
                * (1.0 - 4.0 * temp1 * temp3).sqrt()
                * (1.0
                    + 3.0 * temp12 * (2.0 * s * temp3 + temp5)
                    + (2.0 * temp2 * HIGGS_VEV.powi(4))
                        / (s * temp4 - 4.0 * temp3 * temp4 + temp5)
                    - (8.0
                        * self.lam_hs
                        * temp12
                        * temp15
                        * temp22
                        * (temp14 * (-s * s + self.lam_hs * temp14 * temp15 + 4.0 * temp5)
                            + temp3 * (self.lam_hs * temp15 + temp16 + temp6) * temp9)
                        * (1.0 / (temp22 * (s - 2.0 * temp3))).atanh())
                        / (temp16 + temp6)))
                / (16.0 * std::f64::consts::PI * (1.0 - 4.0 * temp1 * temp4).sqrt());
        } else {
            0.0
        }
    }
    pub fn sigma_ss(&self, cme: f64) -> f64 {
        self.sigma_ss_ff(cme, TOP_QUARK_MASS, 3.0)
            + self.sigma_ss_ff(cme, CHARM_QUARK_MASS, 3.0)
            + self.sigma_ss_ff(cme, UP_QUARK_MASS, 3.0)
            + self.sigma_ss_ff(cme, BOTTOM_QUARK_MASS, 3.0)
            + self.sigma_ss_ff(cme, STRANGE_QUARK_MASS, 3.0)
            + self.sigma_ss_ff(cme, DOWN_QUARK_MASS, 3.0)
            + self.sigma_ss_ff(cme, TAU_MASS, 1.0)
            + self.sigma_ss_ff(cme, MUON_MASS, 1.0)
            + self.sigma_ss_ff(cme, ELECTRON_MASS, 1.0)
            + self.sigma_ss_zz(cme)
            + self.sigma_ss_ww(cme)
            + self.sigma_ss_hh(cme)
    }
}
