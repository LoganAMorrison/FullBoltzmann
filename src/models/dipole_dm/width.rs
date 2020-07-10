use super::DipoleDm;

impl DipoleDm {
    #[allow(dead_code)]
    pub(super) fn compute_width_h(mx: f64, udm: f64, ulam: f64, ce: f64, cm: f64) -> f64 {
        ((ce * ce + cm * cm)
            * mx
            * ((udm * udm * (2.0 + udm).powi(2)) / (1.0 + udm).powi(2)).powf(1.5))
            / (8. * std::f64::consts::PI * ulam * ulam)
    }
}
