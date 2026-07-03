use super::{
    agent::AgentRuntime,
    glyph::{GlyphInstance, GlyphParams},
    glyph_block::GlyphBlock,
    KernelState,
};

#[derive(Debug, Clone)]
pub struct GlyphTrace {
    pub tick: u64,
    pub code: String,
    pub title: String,
    pub params: String,
}

pub fn execute_block(block: &GlyphBlock, agent: &mut AgentRuntime, kernel: &mut KernelState) {
    for instance in &block.glyphs {
        execute_glyph(instance, agent, kernel);
        kernel.tick += 1;
    }
}

fn execute_glyph(instance: &GlyphInstance, agent: &mut AgentRuntime, kernel: &mut KernelState) {
    let params_str = match &instance.params {
        GlyphParams::None => "None".into(),
        GlyphParams::Text(t) => format!("Text({})", t),
        GlyphParams::Number(n) => format!("Number({})", n),
        GlyphParams::Bool(b) => format!("Bool({})", b),
        GlyphParams::Map(m) => format!("Map({:?})", m),
    };

    kernel.trace.push(GlyphTrace {
        tick: kernel.tick,
        code: instance.glyph.code.into(),
        title: instance.glyph.title.into(),
        params: params_str,
    });

    match instance.glyph.code {
        "SENSE_IN" => sense_in(instance, agent, kernel),
        "MODEL_CALL" => model_call(instance, agent, kernel),
        "MODEL_ERROR" => model_error(instance, agent, kernel),
        "PLAN_CHAIN" => plan_chain(instance, agent, kernel),
        "ACT_TOOL" => act_tool(instance, agent, kernel),
        "REFLECT_SELF" => reflect_self(instance, agent, kernel),
        "MEM_STORE" => mem_store(instance, agent, kernel),
        "EVO_MUTATE" => evo_mutate(instance, agent, kernel),
        "DEPENDENCY_CHECK" => dependency_check(instance, agent, kernel),
        "SENTINEL_DEP" => sentinel_dep(instance, agent, kernel),
        _ => {}
    }
}

fn sense_in(_: &GlyphInstance, _: &mut AgentRuntime, _: &mut KernelState) {}

fn model_call(instance: &GlyphInstance, agent: &mut AgentRuntime, _: &mut KernelState) {
    let mut backend_name = "claude".to_string();
    let mut backends: Option<Vec<String>> = None;
    let mut blend_strategy: Option<String> = None;
    let mut prompt = "Default prompt from GH05T3 core loop".to_string();
    let mut version = "v2".to_string();
    let mut meta = std::collections::HashMap::new();

    if let GlyphParams::Map(m) = &instance.params {
        for (k, v) in m {
            match k.as_str() {
                "backend" => backend_name = v.clone(),
                "backends" => {
                    backends = Some(v.split(',').map(|s| s.trim().to_string()).collect())
                }
                "blend_strategy" => blend_strategy = Some(v.clone()),
                "prompt" => prompt = v.clone(),
                "version" => version = v.clone(),
                _ => {
                    meta.insert(k.clone(), v.clone());
                }
            }
        }
    }

    // v4: "backends" + "blend_strategy" present -> multi-model payload.
    // Otherwise: v2 single-backend payload (unchanged behavior).
    let response = match (backends, blend_strategy) {
        (Some(backends), Some(strategy)) => {
            crate::ffi::model_call_blend_summary(backends, &prompt, &version, &strategy, meta)
        }
        _ => crate::ffi::model_call_summary(&backend_name, &prompt, &version, meta),
    };
    agent.memory.short_term.push(response);
}

fn model_error(instance: &GlyphInstance, agent: &mut AgentRuntime, _: &mut KernelState) {
    if let GlyphParams::Text(msg) = &instance.params {
        agent.memory.short_term.push(format!("[MODEL_ERROR] {}", msg));
    }
}

fn plan_chain(_: &GlyphInstance, _: &mut AgentRuntime, _: &mut KernelState) {}
fn act_tool(_: &GlyphInstance, _: &mut AgentRuntime, _: &mut KernelState) {}
fn reflect_self(_: &GlyphInstance, _: &mut AgentRuntime, _: &mut KernelState) {}
fn mem_store(_: &GlyphInstance, _: &mut AgentRuntime, _: &mut KernelState) {}
fn evo_mutate(_: &GlyphInstance, _: &mut AgentRuntime, _: &mut KernelState) {}

fn dependency_check(_: &GlyphInstance, agent: &mut AgentRuntime, _: &mut KernelState) {
    agent
        .memory
        .short_term
        .push("[DEPENDENCY_CHECK: pending Python evaluation]".into());
}

fn sentinel_dep(_: &GlyphInstance, agent: &mut AgentRuntime, _: &mut KernelState) {
    agent
        .memory
        .short_term
        .push("[SENTINEL_DEP: Python dependency check required]".into());
}
