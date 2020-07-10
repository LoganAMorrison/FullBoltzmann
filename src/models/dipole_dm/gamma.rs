use super::DipoleDm;

pub(super) fn gamma_integrand_ss_coeff(mx: f64, udm: f64, ulam: f64, ce: f64, cm: f64) -> f64 {
    (8.0 * (ce * ce + cm * cm).powi(2) * mx.powi(4) * udm.powi(6)) / (3.0 * ulam.powi(4))
}
pub(super) fn gamma_integrand_tt_coeff(mx: f64, udm: f64, ulam: f64, ce: f64, cm: f64) -> f64 {
    (2.0 * (ce * ce + cm * cm).powi(2) * mx.powi(4) * udm.powi(4)) / (3.0 * ulam.powi(4))
}

pub(super) fn gamma_integrand_st_coeff(mx: f64, udm: f64, ulam: f64, ce: f64, cm: f64) -> f64 {
    (2.0 * (ce * ce + cm * cm).powi(2) * mx.powi(6) * udm.powi(2) * (1.0 + (1.0 + udm).powi(2)))
        / (3.0 * ulam.powi(4))
}
impl DipoleDm {
    fn gamma_integrand_ss(&self, w: f64) -> f64 {
        let udm = self.dm / self.mx;
        self.gam_coeff_ss * (w.powi(8) * (6.0 + udm * (-2.0 - udm + 14.0 * w)))
            / ((2.0 + udm - 2.0 * w).powi(2) * (1.0 + 2.0 * udm * w).powi(3))
    }
    fn gamma_integrand_tt(&self, w: f64) -> f64 {
        let udm = self.dm / self.mx;
        let t1 = w.powi(2);
        let t2 = 2.0 + udm;
        let t3 = t2.powi(2);
        let t4 = 2.0 * w * udm;
        let t5 = 1.0 + t4;
        let t6 = 5.0 * t1 * udm;
        let t7 = 1.0 + udm;
        let t8 = t7.powi(2);
        let t9 = 2.0 * t8 * w;
        let t10 = 2.0 + t6 + t9 + udm;
        self.gam_coeff_tt
            * (t1
                * udm
                * (-12.0 * t2.powi(4)
                    + 70.0 * w.powi(7) * udm.powi(3)
                    + w.powi(6) * udm.powi(2) * (58.0 - 37.0 * t2 * udm)
                    - 12.0 * t2.powi(3) * w * (1.0 + 6.0 * t2 * udm)
                    - 6.0 * t1 * t3 * (2.0 + 3.0 * t2 * udm) * (-1.0 + 8.0 * t2 * udm)
                    - 6.0
                        * t3
                        * w.powi(3)
                        * udm
                        * (-13.0 + 4.0 * t2 * udm * (7.0 + 4.0 * t2 * udm))
                    - 2.0
                        * t2
                        * w.powi(4)
                        * udm
                        * (3.0 + 5.0 * t2 * udm * (-17.0 + 12.0 * t2 * udm))
                    + 2.0 * w.powi(5) * udm * (6.0 + t2 * udm * (-15.0 + 62.0 * t2 * udm))))
            / (t10 * t5.powi(3))
            + 6.0
                * t3
                * (-t1 + t3 + t2 * w)
                * (t10.powi(2) / (t5.powi(2) * (2.0 + 2.0 * w + udm).powi(2))).ln()
    }
    fn gamma_integrand_st(&self, w: f64) -> f64 {
        let udm = self.dm / self.mx;
        let t1 = 1.0 + udm;
        let t2 = t1.powi(2);
        let t3 = 2.0 + udm;
        let t4 = w.powi(2);
        let t5 = 2.0 * w;
        let t6 = 2.0 + t5 + udm;
        let t7 = 2.0 * w * udm;
        let t8 = 1.0 + t7;
        let t9 = t8.powi(-2);
        let t10 = 2.0 * t2 * w;
        self.gam_coeff_st
            * (-(t4
                * t9
                * udm
                * (6.0 * t3.powi(2)
                    + 24.0 * t2 * t3 * w
                    + 42.0 * t2 * w.powi(3) * udm
                    + 2.0 * w.powi(4) * udm.powi(2)
                    + 3.0 * t4 * (8.0 + t3 * udm * (23.0 + 8.0 * t3 * udm))))
                + 3.0
                    * t6.powi(2)
                    * (2.0 + t10 + udm)
                    * ((t9 * (2.0 + t10 + udm + 5.0 * t4 * udm).powi(2)) / t6.powi(2)).ln())
            / (-2.0 + t5 - udm)
    }
    #[allow(dead_code)]
    pub(super) fn gamma_integrand(&self, w: f64, x: f64) -> f64 {
        let udm = self.dm / self.mx;
        let ss = self.gamma_integrand_ss(w);
        let tt = 0.0; //self.gamma_integrand_tt(w);
        let st = 0.0; //self.gamma_integrand_st(w);
        let temp_fac = x / (2.0 * self.mx) / ((w * x * udm).cosh() - 1.0);
        let jac = udm * self.mx;

        (ss + tt + st) * temp_fac * jac
    }
}
