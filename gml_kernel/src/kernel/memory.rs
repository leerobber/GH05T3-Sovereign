#[derive(Debug)]
pub struct KernelMemory {
    pub global: std::collections::HashMap<String, String>,
}

impl KernelMemory {
    pub fn new() -> Self {
        Self {
            global: std::collections::HashMap::new(),
        }
    }
}
