//! Typed Observation -> ActionPlan boundary the async runtime
//! (tool_bus/agent/episode) drives, without depending on glyph internals.
//!
//! Named `model_policy` rather than `gh05t3_core` (as originally sketched)
//! to avoid colliding, in meaning, with the crate's existing top-level
//! `gh05t3` module (the real persona/core-loop engine this file wraps) --
//! two different things named almost identically would be confusing.

use std::sync::Mutex;

use serde::{Deserialize, Serialize};

use crate::gh05t3::core_loop::create_gh05t3_agent;
use crate::kernel::{agent::AgentRuntime, executor::execute_block, KernelState};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Observation {
    pub episode_id: String,
    pub tick: u64,
    pub world_state: serde_json::Value,
    pub memory_snapshot: serde_json::Value,
    pub goals: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Action {
    pub id: String,
    pub kind: ActionKind,
    pub payload: serde_json::Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ActionKind {
    ToolCall { tool_name: String },
    MemoryWrite,
    MemoryRead,
    SpawnAgent { role: String },
    SelfMutation { description: String },
    NoOp,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ActionPlan {
    pub actions: Vec<Action>,
}

/// Core policy trait: any reasoning engine can plug in here as long as it's
/// Send + Sync (infer() takes &self -- multiple async tasks may hold a
/// shared reference).
pub trait ModelPolicy: Send + Sync {
    fn infer(&self, obs: Observation) -> ActionPlan;
}

/// Real policy, not a stub: each `infer()` call runs one real pass of the
/// existing glyph core loop (gh05t3::core_loop + kernel::executor -- the
/// same engine src/main.rs's synchronous entrypoint drives) against
/// persistent AgentRuntime/KernelState, and reports the glyphs that
/// actually executed this tick as a single real ToolCall action (see
/// runtime::tool_bus::KernelTickReportTool) -- not a fabricated response.
pub struct Gh05t3Policy {
    agent: Mutex<AgentRuntime>,
    kernel: Mutex<KernelState>,
}

impl Gh05t3Policy {
    pub fn new() -> Self {
        Self {
            agent: Mutex::new(create_gh05t3_agent()),
            kernel: Mutex::new(KernelState::new()),
        }
    }
}

impl Default for Gh05t3Policy {
    fn default() -> Self {
        Self::new()
    }
}

impl ModelPolicy for Gh05t3Policy {
    fn infer(&self, obs: Observation) -> ActionPlan {
        let mut agent = self.agent.lock().expect("agent mutex poisoned");
        let mut kernel = self.kernel.lock().expect("kernel mutex poisoned");

        let core_loop = agent.core_loop.clone();
        let trace_start = kernel.trace.len();
        execute_block(&core_loop, &mut agent, &mut kernel);

        let glyphs_executed: Vec<serde_json::Value> = kernel.trace[trace_start..]
            .iter()
            .map(|entry| {
                serde_json::json!({
                    "tick": entry.tick,
                    "code": entry.code,
                    "title": entry.title,
                    "params": entry.params,
                })
            })
            .collect();

        ActionPlan {
            actions: vec![Action {
                id: format!("kernel-tick-report-{}", obs.tick),
                kind: ActionKind::ToolCall { tool_name: "kernel_tick_report".to_string() },
                payload: serde_json::json!({
                    "observation_tick": obs.tick,
                    "observation_episode": obs.episode_id,
                    "glyphs_executed": glyphs_executed,
                }),
            }],
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn obs(tick: u64) -> Observation {
        Observation {
            episode_id: "test-episode".into(),
            tick,
            world_state: serde_json::json!({}),
            memory_snapshot: serde_json::json!({}),
            goals: vec![],
        }
    }

    #[test]
    fn infer_reports_every_glyph_that_actually_executed() {
        let policy = Gh05t3Policy::new();
        let plan = policy.infer(obs(1));

        assert_eq!(plan.actions.len(), 1);
        let payload = &plan.actions[0].payload;
        let glyphs = payload["glyphs_executed"].as_array().unwrap();

        // gh05t3::core_loop's fixed core loop has 14 glyphs (see
        // build_gh05t3_core_loop) -- every one of them must show up here,
        // not a fabricated subset.
        assert_eq!(glyphs.len(), 14);
        assert_eq!(glyphs[0]["code"], "SENSE_IN");
        assert_eq!(glyphs[4]["code"], "MODEL_CALL");
    }

    #[test]
    fn kernel_state_persists_and_accumulates_across_calls() {
        let policy = Gh05t3Policy::new();
        let _ = policy.infer(obs(1));
        let plan2 = policy.infer(obs(2));

        // Second call's report must reflect the SECOND pass's glyphs
        // (ticks continuing on from the first), proving KernelState is
        // real persistent state, not reconstructed fresh each call.
        let glyphs = plan2.actions[0].payload["glyphs_executed"].as_array().unwrap();
        let first_tick = glyphs[0]["tick"].as_u64().unwrap();
        assert_eq!(first_tick, 14, "second pass should continue from tick 14, not restart at 0");
    }
}
