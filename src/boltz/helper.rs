use haliax_constants::cosmology::M_PLANK;
use haliax_thermal_functions::prelude::*;
use std::f64::consts::PI;

pub fn gefft(temp: f64) -> f64 {
    sm_sqrt_gstar(temp) * sm_geff(temp).sqrt() / sm_heff(temp) - 1.0
}

pub fn hubblet(temp: f64) -> f64 {
    let h = (4.0 * PI.powi(3) * sm_geff(temp) / 45.0).sqrt() * temp * temp / M_PLANK;
    h / (1.0 + gefft(temp))
}
