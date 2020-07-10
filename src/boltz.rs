//! The boltz module contains various traits and functions for solving different
//! forms of the Boltzmann equation:
//!
//! # `boltz::full`
//! This module contains the trait `FullBoltzmann` and allows any type that
//! implements it to solve the full Boltzmann equation for the DM phase-space
//! distribution.
//!
//! # `boltz::coupled`
//! This module contains the trait `CoupledBoltzmann` and allows any type that
//! implements it to solve a coupled pair of Boltzmann equations for the DM
//! comoving number density and the DM temperature.
//!
//! # `boltz::simple`
//! This module contains the trait `SimpleBoltzmann` and allows any type that
//! implements it to solve the standard Boltzmann equation for the DM comoving
//! number density.

pub mod coupled;
pub mod full;
pub mod helper;
pub mod simple;
pub mod traits;

pub use coupled::*;
pub use full::*;
pub use helper::*;
pub use simple::*;
pub use traits::*;
