pub mod glyph;
pub mod glyph_block;
pub mod executor;
pub mod memory;
pub mod agent;
pub mod policy;
pub mod evolution;
pub mod io;
pub mod model;
pub mod payload;

use executor::GlyphTrace;
use memory::KernelMemory;

pub struct KernelState {
    pub memory: KernelMemory,
    pub tick: u64,
    pub trace: Vec<GlyphTrace>,
}

impl KernelState {
    pub fn new() -> Self {
        Self {
            memory: KernelMemory::new(),
            tick: 0,
            trace: Vec::new(),
        }
    }
}
