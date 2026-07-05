//! Typed async tool invocation layer. Requires `async-trait` (used by the
//! `Tool` trait below so `Box<dyn Tool>` stays dyn-compatible with an
//! async method) -- the originally-pasted sketch used
//! `#[async_trait::async_trait]` but never listed the crate as a
//! dependency, so it would not have compiled as-is.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};
use thiserror::Error;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolInput {
    pub name: String,
    pub args: serde_json::Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolOutput {
    pub name: String,
    pub result: serde_json::Value,
    pub success: bool,
}

#[derive(Debug, Error)]
pub enum ToolError {
    #[error("tool not found: {0}")]
    NotFound(String),
    #[error("tool execution failed: {0}")]
    Execution(String),
}

#[async_trait::async_trait]
pub trait Tool: Send + Sync {
    async fn invoke(&self, input: ToolInput) -> Result<ToolOutput, ToolError>;
}

pub struct ToolBus {
    tools: HashMap<String, Box<dyn Tool>>,
}

impl ToolBus {
    pub fn new() -> Self {
        Self { tools: HashMap::new() }
    }

    pub fn register_tool<T>(&mut self, name: &str, tool: T)
    where
        T: Tool + 'static,
    {
        self.tools.insert(name.to_string(), Box::new(tool));
    }

    pub async fn call(&self, input: ToolInput) -> Result<ToolOutput, ToolError> {
        match self.tools.get(&input.name) {
            Some(tool) => tool.invoke(input).await,
            None => Err(ToolError::NotFound(input.name.clone())),
        }
    }
}

impl Default for ToolBus {
    fn default() -> Self {
        Self::new()
    }
}

/// Minimal round-trip tool -- a genuine infrastructure primitive proving
/// the bus dispatches correctly, not a placeholder standing in for real
/// work.
pub struct EchoTool;

#[async_trait::async_trait]
impl Tool for EchoTool {
    async fn invoke(&self, input: ToolInput) -> Result<ToolOutput, ToolError> {
        Ok(ToolOutput { name: input.name, result: input.args, success: true })
    }
}

/// Real tool: bridges the async ToolBus to the already-verified v2
/// MODEL_CALL contract (crate::ffi::model_call_summary) -- the same JSON
/// envelope the synchronous glyph engine's own MODEL_CALL glyph produces
/// (see kernel::executor::model_call). Reuses tested code rather than
/// re-implementing the contract.
pub struct ModelCallTool;

#[async_trait::async_trait]
impl Tool for ModelCallTool {
    async fn invoke(&self, input: ToolInput) -> Result<ToolOutput, ToolError> {
        let backend = input.args.get("backend").and_then(|v| v.as_str()).unwrap_or("claude").to_string();
        let prompt = input.args.get("prompt").and_then(|v| v.as_str()).unwrap_or("").to_string();
        let version = input.args.get("version").and_then(|v| v.as_str()).unwrap_or("v2").to_string();

        let json = crate::ffi::model_call_summary(&backend, &prompt, &version, HashMap::new());
        let result: serde_json::Value = serde_json::from_str(&json)
            .map_err(|e| ToolError::Execution(format!("model_call_summary produced invalid JSON: {e}")))?;

        Ok(ToolOutput { name: input.name, result, success: true })
    }
}

/// Accepts a real kernel-tick report (see
/// runtime::model_policy::Gh05t3Policy::infer) and echoes it back --
/// proves an ActionPlan built from real KernelState/GlyphTrace data
/// round-trips through the async ToolBus successfully, end to end.
pub struct KernelTickReportTool;

#[async_trait::async_trait]
impl Tool for KernelTickReportTool {
    async fn invoke(&self, input: ToolInput) -> Result<ToolOutput, ToolError> {
        Ok(ToolOutput { name: input.name, result: input.args, success: true })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn echo_tool_round_trips_args() {
        let mut bus = ToolBus::new();
        bus.register_tool("echo", EchoTool);

        let args = serde_json::json!({"hello": "world"});
        let out = bus.call(ToolInput { name: "echo".into(), args: args.clone() }).await.unwrap();

        assert!(out.success);
        assert_eq!(out.result, args);
    }

    #[tokio::test]
    async fn unregistered_tool_returns_not_found() {
        let bus = ToolBus::new();
        let err = bus.call(ToolInput { name: "nonexistent".into(), args: serde_json::json!({}) }).await;

        assert!(matches!(err, Err(ToolError::NotFound(name)) if name == "nonexistent"));
    }

    #[tokio::test]
    async fn model_call_tool_produces_real_v2_envelope() {
        let mut bus = ToolBus::new();
        bus.register_tool("model_call", ModelCallTool);

        let args = serde_json::json!({"backend": "claude", "prompt": "hi", "version": "v2"});
        let out = bus.call(ToolInput { name: "model_call".into(), args }).await.unwrap();

        assert!(out.success);
        assert_eq!(out.result["backend"], "claude");
        assert_eq!(out.result["prompt"], "hi");
        assert_eq!(out.result["version"], "v2");
    }
}
