pub mod dipole_dm;
pub mod scalar_singlet;
pub mod toy;

/// Toy model that is used to check that the implementation of the full
/// Boltzmann equation is valid.
pub struct ToyModel {
    /// Mass of the dark-matter particle.
    pub mx: f64,
    /// Coefficient of the temperature-independent piece of the scattering term.
    pub c0: f64,
    /// Coefficient of the temperature-dependent piece of the scattering term.
    pub c1: f64,
}

/// BSM model where the SM is altered by adding a single scalar gauge singlet
/// which interacts with the SM through a Higgs interaction of the form
/// SSHH.
pub struct ScalarSinglet {
    /// Mass of the new scalar.
    pub ms: f64,
    /// Coefficient of the SSHH term in the scalar potential.
    pub lam_hs: f64,
}

/// Effective field theory with two dark matter particles chi1 and chi2 which
/// interact with the SM via a electic+magnetic dipole operator:
///     Lint ~ chi1bar.sigma_mn.chi2 F^mn
/// The chi2 particle has a mass which is larger than chi1 by an amount `dm`.
pub struct DipoleDm {
    /// Mass of the lighest dark particle
    pub mx: f64,
    /// Mass splitting between lighest and heavy dark particles
    pub dm: f64,
    /// Cut-off scale.
    pub lam: f64,
    /// Electric dipole coefficient.
    pub ce: f64,
    /// Magnetic dipole coefficient.
    pub cm: f64,
    /// Decay width of heavy dark particle
    pub width_h: f64,
    /// Coefficient of the s-channel contribution to gamma
    pub(super) gam_coeff_ss: f64,
    /// Coefficient of the t-channel contribution to gamma
    pub(super) gam_coeff_tt: f64,
    /// Coefficient of the s-t interference contribution to gamma
    pub(super) gam_coeff_st: f64,
}
