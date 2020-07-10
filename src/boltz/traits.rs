pub trait FullBoltzmann {
    /// Equillibrium phase-space distribution evaluated at momentum `q` and
    /// x = m / T.
    fn feq(&self, x: f64, q: f64) -> f64;
    /// Momentum exchange rate divided by ht
    fn gamma_hinv(&self, x: f64) -> f64;
    /// velocity-weighted  cross  section  averaged  over angles
    fn sigmav(&self, x: f64, q: f64, qt: f64) -> f64;
    /// Dark matter mass
    fn dm_mass(&self) -> f64;
    /// Dark matter d.o.f.
    fn g(&self) -> f64;
}

pub trait SimpleBoltzmann {
    fn thermal_cross_section(&self, x: f64) -> f64;
    fn mass(&self) -> f64;
}

pub trait CoupledBoltzmann {
    fn sigmav(&self, x: f64) -> f64;
    fn sigmav2(&self, x: f64) -> f64;
    fn sigmav_neq(&self, x: f64) -> f64;
    fn sigmav2_neq(&self, x: f64) -> f64;
}
