use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// v2 MODEL_CALL contract — the JSON envelope exchanged between the kernel
/// and Python. `meta` carries anything in the glyph's params.Map beyond
/// backend/prompt/version.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelCallPayload {
    pub backend: String,
    pub prompt: String,
    pub version: String,
    #[serde(default)]
    pub meta: HashMap<String, String>,
}

impl ModelCallPayload {
    pub fn to_json(&self) -> String {
        serde_json::to_string(self).expect("ModelCallPayload always serializes")
    }
}

/// v4 multi-model MODEL_CALL contract — requests several backends and a
/// strategy for combining their outputs. Kept as a distinct struct from
/// ModelCallPayload (v2) rather than adding optional fields to it, so v2's
/// single-backend shape stays unchanged for existing callers.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MultiModelCallPayload {
    pub backends: Vec<String>,
    pub prompt: String,
    pub version: String,
    pub blend_strategy: String,
    #[serde(default)]
    pub meta: HashMap<String, String>,
}

impl MultiModelCallPayload {
    pub fn to_json(&self) -> String {
        serde_json::to_string(self).expect("MultiModelCallPayload always serializes")
    }
}

/// JSON envelope for gh05t3_run_core_loop_json — the full-run summary,
/// replacing the old Debug-formatted "ticks=...,short_term=..." string.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KernelRunSummary {
    pub tick: u64,
    pub short_term: Vec<String>,
}

impl KernelRunSummary {
    pub fn to_json(&self) -> String {
        serde_json::to_string(self).expect("KernelRunSummary always serializes")
    }
}
