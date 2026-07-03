#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GlyphClass {
    Perception,
    Model,
    Planning,
    Action,
    Reflection,
    Alignment,
    Memory,
    Evolution,
}

#[derive(Debug, Clone)]
pub struct Glyph {
    pub id: u16,
    pub code: &'static str,
    pub title: &'static str,
    pub class: GlyphClass,
}

#[derive(Debug, Clone)]
pub enum GlyphParams {
    None,
    Text(String),
    Number(f64),
    Bool(bool),
    Map(std::collections::HashMap<String, String>),
}

#[derive(Debug, Clone)]
pub struct GlyphInstance {
    pub glyph: &'static Glyph,
    pub params: GlyphParams,
}

pub static GLYPHS: &[Glyph] = &[
    // Perception
    Glyph { id: 1, code: "SENSE_IN", title: "Aperture of the Void", class: GlyphClass::Perception },
    Glyph { id: 2, code: "PARSE_FORM", title: "Fracture of Meaning", class: GlyphClass::Perception },
    Glyph { id: 3, code: "FEATURE_EXTRACT", title: "Spectral Unveiling", class: GlyphClass::Perception },

    // Model
    Glyph { id: 10, code: "MODEL_CALL", title: "Oracle Invocation", class: GlyphClass::Model },
    Glyph { id: 11, code: "MODEL_BLEND", title: "Chorus of Oracles", class: GlyphClass::Model },
    Glyph { id: 12, code: "MODEL_DISTILL", title: "Crucible of Insight", class: GlyphClass::Model },
    Glyph { id: 13, code: "MODEL_ERROR", title: "Oracle Fracture", class: GlyphClass::Model },
    Glyph { id: 14, code: "MODEL_EVAL", title: "Oracle Reckoning", class: GlyphClass::Model },

    // Planning
    Glyph { id: 20, code: "PLAN_CHAIN", title: "Lattice of Intent", class: GlyphClass::Planning },
    Glyph { id: 21, code: "PLAN_BRANCH", title: "Fork of Futures", class: GlyphClass::Planning },
    Glyph { id: 22, code: "PLAN_SELECT", title: "Judgment of Paths", class: GlyphClass::Planning },

    // Action
    Glyph { id: 30, code: "ACT_TOOL", title: "Hand of the Machine", class: GlyphClass::Action },
    Glyph { id: 31, code: "ACT_ENV", title: "World-Touch", class: GlyphClass::Action },
    Glyph { id: 32, code: "ACT_COMMIT", title: "Seal of Execution", class: GlyphClass::Action },

    // Reflection
    Glyph { id: 40, code: "REFLECT_SELF", title: "Mirror of the Core", class: GlyphClass::Reflection },
    Glyph { id: 41, code: "REFLECT_TRACE", title: "Path of Echoes", class: GlyphClass::Reflection },
    Glyph { id: 42, code: "REFLECT_REWRITE", title: "Blade of Revision", class: GlyphClass::Reflection },
    Glyph { id: 43, code: "DEPENDENCY_CHECK", title: "Sentinel of Foundations", class: GlyphClass::Reflection },
    Glyph { id: 44, code: "SENTINEL_DEP", title: "Ward of Systems", class: GlyphClass::Reflection },

    // Alignment
    Glyph { id: 50, code: "ALIGN_CHECK", title: "Ward of Boundaries", class: GlyphClass::Alignment },
    Glyph { id: 51, code: "ALIGN_BLOCK", title: "Lock of Prohibition", class: GlyphClass::Alignment },
    Glyph { id: 52, code: "ALIGN_ESCALATE", title: "Beacon of Review", class: GlyphClass::Alignment },

    // Memory
    Glyph { id: 60, code: "MEM_STORE", title: "Vault of Echoes", class: GlyphClass::Memory },
    Glyph { id: 61, code: "MEM_RECALL", title: "Summoning of Threads", class: GlyphClass::Memory },
    Glyph { id: 62, code: "MEM_COMPRESS", title: "Gravity of Knowledge", class: GlyphClass::Memory },

    // Evolution
    Glyph { id: 70, code: "EVO_MUTATE", title: "Spiral of Change", class: GlyphClass::Evolution },
    Glyph { id: 71, code: "EVO_EVAL", title: "Tribunal of Fitness", class: GlyphClass::Evolution },
    Glyph { id: 72, code: "EVO_SELECT", title: "Crown of Survivors", class: GlyphClass::Evolution },
    Glyph { id: 73, code: "EVO_ARCHIVE", title: "Catacomb of Code", class: GlyphClass::Evolution },
];
