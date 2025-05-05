use core::cmp::{Ordering, Ordering::*};
use core::ops::{Deref, Neg};
use statrs::distribution::{DiscreteCDF, Poisson};
use std::f64::consts::{FRAC_1_SQRT_2, PI};

#[derive(Clone, Copy, Debug)]
pub struct Ordf64 {
    pub f: f64,
}

impl Ordf64 {
    pub fn new(value: f64) -> Self {
        Ordf64 { f: value }
    }
}
impl Deref for Ordf64 {
    type Target = f64;
    fn deref(&self) -> &Self::Target {
        &self.f
    }
}
impl Neg for Ordf64 {
    type Output = Self;
    fn neg(self) -> Self::Output {
        Ordf64::new(-*self)
    }
}
impl PartialOrd for Ordf64 {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some((*self).cmp(other))
    }
}
impl PartialEq for Ordf64 {
    fn eq(&self, rhs: &Ordf64) -> bool {
        (*self).cmp(rhs) == Equal
    }
}
impl Ord for Ordf64 {
    fn cmp(&self, other: &Self) -> Ordering {
        (*self).total_cmp(other)
    }
}
impl Eq for Ordf64 {}

pub fn neg_ln_poisson_tail(k: f64, lambda: f64) -> f64 {
    // ----- Small λ: exact ----------------------------
    if lambda < 20.0 {
        let p = Poisson::new(lambda).unwrap().sf(k.ceil() as u64);
        -p.max(1e-308).ln()
    } else if k < lambda || (k - lambda).abs() <= 4. * lambda.sqrt() {
        // ---- Normal tail with continuity correction ----
        let z = (k - lambda + 0.5) / lambda.sqrt();
        let p = 0.5 * (z * FRAC_1_SQRT_2).erfc();
        -p.max(1e-308).ln()
    } else {
        // ---- Saddle-point (Lugannani–Rice) ----
        let s = k / lambda;
        let t = (2.0 * lambda * (s - 1.0 - s.ln())).sqrt();
        let w = t + (1.0 / s - 1.0) / t;
        let ln_sf = -lambda * (s - 1.0 - s.ln()) - w.ln() - 0.5 * (2.0 * PI * k).ln();
        -ln_sf
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use statrs::distribution::{DiscreteCDF, Poisson};

    fn exact_neg_ln_sf(k: f64, lambda: f64) -> f64 {
        let p = Poisson::new(lambda).unwrap().sf(k.ceil() as u64);
        -p.max(1e-308).ln()
    }

    fn err(est: f64, exact: f64) -> f64 {
        let p_est = (-est).exp();
        let p_exact = (-exact).exp();
        (p_est - p_exact).abs()
    }

    #[test]
    fn small_lambda_branch() {
        let k = 5.0;
        let lam = 7.3; // still < 20
        let est = neg_ln_poisson_tail(k, lam);
        let true_ = exact_neg_ln_sf(k, lam);
        assert!(
            err(est, true_) < 1e-5,
            "small-λ branch failed: est={est}, exact={true_}",
        );
    }

    #[test]
    fn normal_approx_branch() {
        let lam = 50.0;
        let k = 20.0;
        let est = neg_ln_poisson_tail(k, lam);
        let true_ = exact_neg_ln_sf(k, lam);
        assert!(
            err(est, true_) < 1e-3,
            "normal-approx branch failed: est={est}, exact={true_} ~ {}",
            err(est, true_),
        );
    }

    #[test]
    fn saddle_point_branch() {
        let lam = 45.0;
        let k = 20.0;
        let est = neg_ln_poisson_tail(k, lam);
        let true_ = exact_neg_ln_sf(k, lam);
        assert!(
            err(est, true_) < 1e-3,
            "saddle-point branch failed: est={est}, exact={true_} ~ {}",
            err(est, true_),
        );
    }

    #[test]
    fn non_negative() {
        let lam = 28.9148008977483;
        let k = 30.0;
        let est = neg_ln_poisson_tail(k, lam);
        let true_ = exact_neg_ln_sf(k, lam);
        assert!(
            err(est, true_) < 1e-1,
            "non-negative failed: est={est}, exact={true_} ~ {}",
            err(est, true_),
        );
    }
}
