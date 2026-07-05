//! Async episode-loop runtime: a typed Observation/ActionPlan/ToolBus
//! layer that drives the existing synchronous glyph engine (kernel::*,
//! gh05t3::core_loop) through model_policy::Gh05t3Policy, rather than
//! replacing it. See model_policy.rs for why Gh05t3Policy is real (not a
//! stub) and tool_bus.rs for why `async-trait` is a required dependency.

pub mod agent;
pub mod episode;
pub mod model_policy;
pub mod tool_bus;
