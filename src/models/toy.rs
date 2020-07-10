use super::ToyModel;
use crate::boltz::traits::FullBoltzmann;

impl FullBoltzmann for ToyModel {
    fn feq(&self, x: f64, q: f64) -> f64 {
        (-(x * x + q * q).sqrt()).exp()
    }
    fn gamma_hinv(&self, temp: f64) -> f64 {
        let x = self.mx / temp;
        100.0 * (10.0 / x).powi(3)
    }
    fn sigmav(&self, x: f64, q: f64, qt: f64) -> f64 {
        let t = (x * x + q * q) * (x * x + qt * qt);
        self.c0 + self.c1 * (1.0 - x.powi(4) / (q * qt * t.sqrt()) * (q * qt / t.sqrt()).atanh())
    }
    fn dm_mass(&self) -> f64 {
        self.mx
    }
    fn g(&self) -> f64 {
        1.0
    }
}
