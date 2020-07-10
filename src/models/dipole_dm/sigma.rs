use super::DipoleDm;
use haliax_constants::electroweak::ALPHA_EM;
use haliax_constants::masses::W_BOSON_MASS;

impl DipoleDm {
    /// Compute the annihilation cross-section for dark matter into photons.
    pub fn sigma_11_to_gg(&self, cme: f64) -> f64 {
        let q = cme / self.mx;
        let udm = self.dm / self.mx;
        let ulam = self.lam / self.mx;
        let temp1 = q * q;
        let temp2 = 2.0 + udm;
        let temp3 = -4.0 + temp1;
        let temp4 = 2.0 * temp2 * udm;
        let temp5 = temp3.sqrt();
        let temp6 = -(q * temp5);
        let temp7 = temp1 + temp4 + temp6;
        let temp8 = 1.0 / temp7;
        let temp9 = udm.powi(4);
        let temp10 = temp2.powi(4);
        let temp11 = udm.powi(5);
        let temp12 = temp2.powi(5);
        let temp13 = q * temp5;
        let temp14 = temp1 + temp13 + temp4;
        let temp15 = 1.0 / temp14;
        let temp16 = temp1 + temp4;
        let temp17 = udm.powi(2);
        let temp18 = q.powi(4);
        let temp19 = q.powi(3);
        let temp20 = temp2.powi(2);
        let temp21 = -36.0 * temp17 * temp20;
        let temp22 = 6.0 * udm;
        let temp23 = 3.0 * temp17;
        let temp24 = 7.0 + temp22 + temp23;
        let temp25 = 4.0 * temp1 * temp24;
        let temp26 = -(temp19 * temp5);
        let temp27 = 2.0 * udm;
        let temp28 = -1.0 + temp17 + temp27;
        let temp29 = temp19 * temp5;
        let temp30 = 2.0 * temp17 * temp20;
        let temp31 = 2.0 * temp1 * temp28;
        return ((self.ce * self.ce + self.cm + self.cm).powi(2)
            * (96.0 * temp11 * temp12 * temp15
                + q * temp16 * (temp18 + temp21 + temp25 + temp26) * (-q + temp5)
                + q * temp16 * (temp18 + temp21 + temp25 + temp29) * (q + temp5)
                - 96.0 * temp11 * temp12 * temp8
                + 48.0 * temp1 * temp10 * temp15 * temp9
                - 48.0 * temp1 * temp10 * temp8 * temp9
                + 12.0
                    * (4.0 * temp1 * temp17 * temp20 * temp28 + 8.0 * temp10 * temp9
                        - temp18
                            * (4.0 + 10.0 * temp17 + temp9 + 12.0 * udm + 4.0 * udm.powi(3)))
                    * ((temp18 + temp29 + temp30 + temp31 + 2.0 * q * temp2 * temp5 * udm)
                        / (temp18 + temp26 + temp30 + temp31 - 2.0 * q * temp2 * temp5 * udm))
                        .ln()))
            / (192.
                * self.mx
                * self.mx
                * std::f64::consts::PI
                * q
                * q
                * temp16
                * temp3
                * ulam.powi(4));
    }
    /// Compute the annihilation cross-section for the heavy-dark matter into
    /// photons.
    pub fn sigma_22_to_gg(&self, cme: f64) -> f64 {
        let q = cme / self.mx;
        let udm = self.dm / self.mx;
        let ulam = self.lam / self.mx;
        let temp1 = 1.0 + udm;
        let temp2 = q * q;
        let temp3 = 2.0 + udm;
        let temp4 = udm.powi(2);
        let temp5 = temp3.powi(2);
        let temp6 = self.mx.powi(6);
        let temp7 = udm.powi(4);
        let temp8 = temp3.powi(4);
        let temp9 = 2.0 * temp3 * udm;
        let temp10 = q.powi(4);
        let temp11 = temp3 * udm;
        let temp12 = -2.0 * temp3 * udm;
        let temp13 = temp1.powi(2);
        let temp14 = -4.0 * temp13;
        let temp15 = temp14 + temp2;
        let temp16 = temp15.sqrt();
        return ((self.ce * self.ce + self.cm * self.cm).powi(2)
            * ((q
                * temp6
                * ((q - 2.0 * temp1) * (2.0 + q + 2.0 * udm)).sqrt()
                * (q.powi(6) - 24.0 * temp7 * temp8
                    + 4.0 * temp2 * temp4 * temp5 * (-1.0 + temp9)
                    + temp10 * (14.0 + (8.0 + temp11) * temp3 * udm)))
                / (3.0 * (temp2 + temp4 * temp5))
                + (temp6
                    * (8.0 * temp7 * temp8 - 4.0 * temp2 * temp4 * temp5 * (1.0 + temp9)
                        + temp10 * (-4.0 + (-2.0 + temp11) * temp3 * udm))
                    * ((temp12 + q * temp16 + temp2).powi(2)
                        / (temp12 - q * temp16 + temp2).powi(2))
                    .ln())
                    / (temp12 + temp2)))
            / (16.0 * self.mx.powi(8) * std::f64::consts::PI * q.powi(2) * temp15 * ulam.powi(4));
    }
    /// Compute the annihilation cross-section for the light and heavy dark
    /// matter into photons.
    pub fn sigma_12_to_12(&self, cme: f64) -> f64 {
        let q = cme / self.mx;
        let udm = self.dm / self.mx;
        let ulam = self.lam / self.mx;

        let temp1 = q * q;
        let temp2 = udm * udm;
        let temp3 = temp1 * temp1;
        let temp4 = 41.0 * temp3;
        let temp5 = 28.0 * temp2;
        let temp6 = udm.powi(3);
        let temp7 = udm.powi(4);
        let temp8 = 9.0 * temp7;
        let temp9 = 37.0 * temp2;
        let temp10 = 1.0 + udm;
        let temp11 = temp10 * temp10;
        let temp12 = -4.0 * temp11;
        let temp13 = temp1 + temp12;
        let temp14 = 2.0 * udm;
        let temp15 = 2.0 + temp14 + temp2;
        let temp16 = self.ce * self.ce;
        let temp17 = self.cm * self.cm;
        let temp18 = temp16 * temp16;
        let temp19 = 2.0 * temp3;
        let temp20 = temp17 * temp17;
        let temp21 = 2.0 + udm;
        let temp22 = temp21.powi(2);
        let temp23 = 5.0 * temp2;
        let temp24 = -4.0 + temp1;
        let temp25 = 4.0 * udm;
        let temp26 = temp13 * temp24;
        let temp27 = temp26.sqrt();
        ((2.0
            * temp16
            * temp17
            * (-74.0 * temp1 * temp15
                + temp4
                + 8.0 * (-8.0 + temp5 + 36.0 * temp6 + temp8 - 16.0 * udm))
            + temp18
                * (temp4 + 8.0 * (10.0 + temp5 + 18.0 * temp6 + temp8 + 20.0 * udm)
                    - 2.0 * temp1 * (56.0 + temp9 + 56.0 * udm))
            + temp20
                * (temp4 - 2.0 * temp1 * (92.0 + temp9 + 2.0 * udm)
                    + 8.0 * (82.0 + 136.0 * temp2 + 54.0 * temp6 + temp8 + 164.0 * udm)))
            / (temp13 / temp24).sqrt()
            + (12.0
                * temp2
                * temp22
                * (2.0
                    * temp16
                    * temp17
                    * (-5.0 * temp1 * temp15 + 6.0 * temp15 * temp15 + temp19)
                    + temp18 * (temp19 + 6.0 * temp15 * temp2 - temp1 * (4.0 + temp23 + temp25))
                    + temp20
                        * (temp19 + 6.0 * temp15 * temp22 - temp1 * (16.0 + temp23 + 16.0 * udm)))
                * ((4.0 - temp1 + 2.0 * temp2 + temp25 + temp27).powi(2)
                    / (-4.0 + temp1 - 2.0 * temp2 + temp27 - 4.0 * udm).powi(2))
                .ln())
                / (temp13 * (temp1 - 2.0 * temp15)))
            / (192.0 * self.mx * self.mx * std::f64::consts::PI * q * q * ulam.powi(4))
    }
    /// Compute the cross section for chi1 + chi2 -> w+ + w-.
    pub fn sigma_12_to_ww(&self, cme: f64) -> f64 {
        let q = cme / self.mx;
        let mw = W_BOSON_MASS / self.mx;
        let udm = self.dm / self.mx;
        let ulam = self.lam / self.mx;

        let temp1 = q * q;
        let temp2 = mw * mw;
        let temp3 = temp1 * temp1;
        let temp4 = ulam.powi(2);
        let temp5 = 2.0 + ulam;
        let temp6 = temp5 * temp5;
        let temp7 = -2.0 * temp4 * temp6;
        -(ALPHA_EM
            * (temp1 - 4.0 * temp2).sqrt()
            * (48.0 * mw.powi(6) - q.powi(6) + 68.0 * mw.powi(4) * temp1 - 16.0 * temp2 * temp3)
            * (self.ce * self.ce * (temp3 + temp7 + temp1 * (-4.0 + temp4 - 4.0 * ulam))
                + self.cm * self.cm * (temp3 + temp7 + temp1 * (8.0 + temp4 + 8.0 * ulam))))
            / (96.0
                * self.mx
                * self.mx
                * mw.powi(4)
                * q.powi(5)
                * (temp3 + temp4 * temp6 - 2.0 * temp1 * (2.0 + temp4 + 2.0 * ulam)).sqrt()
                * ulam.powi(2))
    }
    /// Compute the cross section for chi1 + chi2 -> f + fbar.
    pub fn sigma_12_to_ff(&self, cme: f64, massf: f64, ncol: f64, qf: f64) -> f64 {
        let q = cme / self.mx;
        let mf = massf / self.mx;

        let udm = self.dm / self.mx;
        let ulam = self.lam / self.mx;
        let temp1 = mf * mf;
        let temp2 = q * q;
        let temp3 = udm.powi(2);
        let temp4 = temp2 * temp2;
        let temp5 = 2.0 + udm;
        let temp6 = temp5 * temp5;
        let temp7 = -2.0 * temp3 * temp6;
        (ALPHA_EM
            * (-4.0 * temp1 + temp2).sqrt()
            * (2.0 * temp1 + temp2)
            * (self.ce * self.ce * (temp4 + temp7 + temp2 * (-4.0 + temp3 - 4.0 * udm))
                + self.cm * self.cm * (temp4 + temp7 + temp2 * (8.0 + temp3 + 8.0 * udm))))
            / (6.0
                * self.mx.powi(2)
                * q.powi(5)
                * (temp4 + temp3 * temp6 - 2.0 * temp2 * (2.0 + temp3 + 2.0 * udm)).sqrt()
                * ulam.powi(2))
    }
}
