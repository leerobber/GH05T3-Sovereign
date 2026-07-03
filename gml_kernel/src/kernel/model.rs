#[derive(Debug, Clone)]
pub enum ModelBackend {
    Claude,
    GPT,
    LocalLlama,
    LocalFallback,
    Unknown(String),
}

#[derive(Debug)]
pub struct ModelEngine;

impl ModelEngine {
    pub fn parse_backend(name: &str) -> ModelBackend {
        match name {
            "claude" => ModelBackend::Claude,
            "gpt" => ModelBackend::GPT,
            "local_llama" => ModelBackend::LocalLlama,
            "local_fallback" => ModelBackend::LocalFallback,
            other => ModelBackend::Unknown(other.to_string()),
        }
    }

    pub fn call(backend: ModelBackend, prompt: &str) -> String {
        match backend {
            ModelBackend::Claude => format!("[CLAUDE] {}", prompt),
            ModelBackend::GPT => format!("[GPT] {}", prompt),
            ModelBackend::LocalLlama => format!("[LLAMA] {}", prompt),
            ModelBackend::LocalFallback => format!("[LOCAL_FALLBACK] {}", prompt),
            ModelBackend::Unknown(name) => format!("[UNKNOWN BACKEND: {}] {}", name, prompt),
        }
    }
}
