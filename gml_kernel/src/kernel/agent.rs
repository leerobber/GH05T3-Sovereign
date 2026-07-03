use super::glyph_block::GlyphBlock;

#[derive(Debug)]
pub struct MemorySegments {
    pub short_term: Vec<String>,
    pub mid_term: Vec<String>,
    pub long_term: Vec<String>,
}

#[derive(Debug)]
pub struct ToolRegistry {
    pub tools: Vec<String>,
}

#[derive(Debug)]
pub struct PolicySet {
    pub policies: Vec<String>,
}

#[derive(Debug)]
pub struct AgentRuntime {
    pub id: String,
    pub name: String,
    pub title: String,
    pub core_loop: GlyphBlock,
    pub memory: MemorySegments,
    pub tools: ToolRegistry,
    pub policies: PolicySet,
}
