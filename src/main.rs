pub mod boltz;
pub mod models;
pub mod utils;

use boltz::*;
use haliax_constants::prelude::*;
use models::*;
use std::io::prelude::*;
use std::time::Instant;

#[allow(dead_code)]
fn simple() -> std::io::Result<()> {
    let ms = HIGGS_MASS / 2.0 - 1.0;
    let lam = 1e-3;
    let model = ScalarSinglet::new(ms, lam);

    let mut file = std::fs::File::create("analysis/simple_boltz_data.dat")?;
    let sol = integrate_simple_boltzmann(model, 1.0, 1000.0);
    let rd = sol.us[sol.us.len() - 1][0].exp() * ms * S_TODAY / RHO_CRIT;
    println!("retcode = {:?}", sol.retcode);
    println!("rd = {}", rd);
    for (t, u) in sol {
        let mut string = format!("{} {}\n", t, u).to_string();
        string.retain(|c| !r#"(),"[]"#.contains(c));
        file.write(string.as_bytes())?;
    }

    Ok(())
}

#[allow(dead_code)]
fn full() -> std::io::Result<()> {
    let ms = HIGGS_MASS / 2.0 - 1.0;
    let lam = 1e-3;
    let model = ScalarSinglet::new(ms, lam);

    let sol = integrate_full_boltzmann(model, 100, (15.0, 100.0));
    println!("retcode = {:?}", sol.retcode);
    let mut file = std::fs::File::create("analysis/full_boltz_data.dat")?;
    for (t, u) in sol {
        let mut string = format!("{} {}\n", t, u).to_string();
        string.retain(|c| !r#"(),"[]"#.contains(c));
        file.write(string.as_bytes())?;
    }
    Ok(())
}

#[allow(dead_code)]
fn full_toy() -> std::io::Result<()> {
    let model = ToyModel {
        mx: 100.0,
        c0: 1e-9,
        c1: 1e-8,
    };

    let sol = integrate_full_boltzmann(model, 100, (15.0, 100.0));
    println!("retcode = {:?}", sol.retcode);
    let mut file = std::fs::File::create("analysis/full_boltz_data.dat")?;
    for (t, u) in sol {
        let mut string = format!("{} {}\n", t, u).to_string();
        string.retain(|c| !r#"(),"[]"#.contains(c));
        file.write(string.as_bytes())?;
    }
    Ok(())
}
#[allow(dead_code)]
fn full_dipole() -> std::io::Result<()> {
    let model = DipoleDm::new(100.0, 1.0, 1e6, 1.0, 1.0);

    let sol = integrate_full_boltzmann(model, 100, (1.0, 100.0));
    println!("retcode = {:?}", sol.retcode);
    let mut file = std::fs::File::create("analysis/full_boltz_data.dat")?;
    for (t, u) in sol {
        let mut string = format!("{} {}\n", t, u).to_string();
        string.retain(|c| !r#"(),"[]"#.contains(c));
        file.write(string.as_bytes())?;
    }
    Ok(())
}

fn main() -> std::io::Result<()> {
    let now = Instant::now();
    let ret = {
        //simple()
        //full()
        full_toy()
        //full_dipole()
    };
    println!("time = {}", now.elapsed().as_secs_f64());
    ret
}
