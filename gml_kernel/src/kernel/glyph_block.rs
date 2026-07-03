use super::glyph::GlyphInstance;

#[derive(Debug, Clone)]
pub struct BlockMeta {
    pub name: String,
    pub title: String,
    pub version: u32,
    pub lineage: Option<String>,
}

#[derive(Debug, Clone)]
pub struct GlyphBlock {
    pub meta: BlockMeta,
    pub glyphs: Vec<GlyphInstance>,
}
