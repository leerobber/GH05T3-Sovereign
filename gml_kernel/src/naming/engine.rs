use rand::seq::SliceRandom;
use rand::thread_rng;

use super::roots::*;

pub struct TitleEngine;

impl TitleEngine {
    pub fn generate_agent_title() -> String {
        let mut rng = thread_rng();
        let role = ["Architect", "Warden", "Navigator", "Sentinel", "Executor"]
            .choose(&mut rng)
            .unwrap();
        let domain = ["Sovereign Lattice", "Glyph Catacomb", "Ternary Sea", "Umbra Continuum"]
            .choose(&mut rng)
            .unwrap();
        format!("{role} of the {domain}")
    }

    pub fn generate_name_from_roots() -> String {
        let mut rng = thread_rng();
        let r1 = SOVEREIGN_ROOTS.choose(&mut rng).unwrap();
        let r2 = COSMIC_ROOTS.choose(&mut rng).unwrap();
        format!("{r1}{r2}")
    }

    pub fn generate_kernel_title() -> String {
        let mut rng = thread_rng();
        let concept = [
            "Silent Convergence",
            "Aethric Convergence",
            "Umbra Matrix",
            "Glyphor Crucible",
        ]
        .choose(&mut rng)
        .unwrap();
        concept.to_string()
    }
}
