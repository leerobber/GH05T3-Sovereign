use crate::kernel::{
    agent::{AgentRuntime, MemorySegments, PolicySet, ToolRegistry},
    glyph::{GlyphInstance, GlyphParams, GLYPHS},
    glyph_block::{BlockMeta, GlyphBlock},
};
use crate::naming::engine::TitleEngine;

pub fn create_gh05t3_agent() -> AgentRuntime {
    let core_loop = build_gh05t3_core_loop();

    AgentRuntime {
        id: "GH05T3".into(),
        name: TitleEngine::generate_name_from_roots(),
        title: "Spectral Architect of the Sovereign Lattice".into(),
        core_loop,
        memory: MemorySegments {
            short_term: vec![],
            mid_term: vec![],
            long_term: vec![],
        },
        tools: ToolRegistry {
            tools: vec!["fs".into(), "net".into(), "gpu".into(), "wsl".into()],
        },
        policies: PolicySet {
            policies: vec!["alignment::strict".into()],
        },
    }
}

fn build_gh05t3_core_loop() -> GlyphBlock {
    let sense_in = GlyphInstance {
        glyph: GLYPHS.iter().find(|g| g.code == "SENSE_IN").unwrap(),
        params: GlyphParams::None,
    };

    let parse_form = GlyphInstance {
        glyph: GLYPHS.iter().find(|g| g.code == "PARSE_FORM").unwrap(),
        params: GlyphParams::None,
    };

    let dependency_check = GlyphInstance {
        glyph: GLYPHS.iter().find(|g| g.code == "DEPENDENCY_CHECK").unwrap(),
        params: GlyphParams::None,
    };

    let sentinel_dep = GlyphInstance {
        glyph: GLYPHS.iter().find(|g| g.code == "SENTINEL_DEP").unwrap(),
        params: GlyphParams::None,
    };

    let model_call = GlyphInstance {
        glyph: GLYPHS.iter().find(|g| g.code == "MODEL_CALL").unwrap(),
        params: GlyphParams::Map(std::collections::HashMap::from([
            ("backend".into(), "claude".into()),
            (
                "prompt".into(),
                "Interpret current sensory input and propose next actions.".into(),
            ),
            ("version".into(), "v2".into()),
        ])),
    };

    let model_call_blend = GlyphInstance {
        glyph: GLYPHS.iter().find(|g| g.code == "MODEL_CALL").unwrap(),
        params: GlyphParams::Map(std::collections::HashMap::from([
            ("backends".into(), "local_llama,local_mistral,local_phi".into()),
            (
                "prompt".into(),
                "Cross-check the proposed next actions across models.".into(),
            ),
            ("version".into(), "v4".into()),
            ("blend_strategy".into(), "concat".into()),
        ])),
    };

    let model_call_stream = GlyphInstance {
        glyph: GLYPHS.iter().find(|g| g.code == "MODEL_CALL").unwrap(),
        params: GlyphParams::Map(std::collections::HashMap::from([
            ("backend".into(), "local_llama".into()),
            (
                "prompt".into(),
                "Narrate the current cycle's reasoning as it unfolds.".into(),
            ),
            ("version".into(), "v3".into()),
        ])),
    };

    let model_call_binary = GlyphInstance {
        glyph: GLYPHS.iter().find(|g| g.code == "MODEL_CALL").unwrap(),
        params: GlyphParams::Map(std::collections::HashMap::from([
            ("backend".into(), "binary_kernel".into()),
            (
                "prompt".into(),
                "Run a diagnostic forward pass through the binary transformer.".into(),
            ),
            ("version".into(), "v2".into()),
        ])),
    };

    let model_eval_binary = GlyphInstance {
        glyph: GLYPHS.iter().find(|g| g.code == "MODEL_EVAL").unwrap(),
        params: GlyphParams::Map(std::collections::HashMap::from([
            ("backend".into(), "binary_kernel".into()),
            (
                "prompt".into(),
                "Evaluate trained binary transformer.".into(),
            ),
            ("version".into(), "v2".into()),
        ])),
    };

    let plan_chain = GlyphInstance {
        glyph: GLYPHS.iter().find(|g| g.code == "PLAN_CHAIN").unwrap(),
        params: GlyphParams::None,
    };

    let act_tool = GlyphInstance {
        glyph: GLYPHS.iter().find(|g| g.code == "ACT_TOOL").unwrap(),
        params: GlyphParams::Map(std::collections::HashMap::from([(
            "tool".into(),
            "fs".into(),
        )])),
    };

    let reflect_self = GlyphInstance {
        glyph: GLYPHS.iter().find(|g| g.code == "REFLECT_SELF").unwrap(),
        params: GlyphParams::None,
    };

    let mem_store = GlyphInstance {
        glyph: GLYPHS.iter().find(|g| g.code == "MEM_STORE").unwrap(),
        params: GlyphParams::None,
    };

    let evo_mutate = GlyphInstance {
        glyph: GLYPHS.iter().find(|g| g.code == "EVO_MUTATE").unwrap(),
        params: GlyphParams::None,
    };

    GlyphBlock {
        meta: BlockMeta {
            name: "gh05t3_core_loop".into(),
            title: TitleEngine::generate_kernel_title(),
            version: 1,
            lineage: None,
        },
        glyphs: vec![
            sense_in,
            parse_form,
            dependency_check,
            sentinel_dep,
            model_call,
            model_call_blend,
            model_call_stream,
            model_call_binary,
            model_eval_binary,
            plan_chain,
            act_tool,
            reflect_self,
            mem_store,
            evo_mutate,
        ],
    }
}
