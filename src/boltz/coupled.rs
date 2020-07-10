pub trait CoupledBoltzmann {
    /// Compute the lab velocity times cross section for chi+chibar -> anything
    fn sigma_vlab(&self, s: f64) -> f64;
}
