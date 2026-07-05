//! Closed-loop runtime: ticks an Agent, dispatches whatever ToolCall
//! actions it returns through the ToolBus, and feeds each real tool
//! result back to the agent (AgentMessage::ToolResult) rather than
//! dropping it -- the originally-pasted sketch left that as a TODO.

use anyhow::Result;
use tokio::time::{sleep, Duration};
use uuid::Uuid;

use super::agent::{Agent, AgentMessage, AgentResponse};
use super::model_policy::{ActionKind, ActionPlan, Observation};
use super::tool_bus::{ToolBus, ToolInput};

pub struct EpisodeManager<A> {
    pub id: String,
    pub tick: u64,
    pub agent: A,
    pub tools: ToolBus,
    max_ticks: u64,
    tick_interval: Duration,
}

impl<A> EpisodeManager<A>
where
    A: Agent,
{
    /// max_ticks/tick_interval are explicit constructor params (not
    /// hardcoded constants, as in the original sketch) so tests can run
    /// short, fast episodes instead of always paying a fixed 100ms*N.
    pub fn new(agent: A, tools: ToolBus, max_ticks: u64, tick_interval: Duration) -> Self {
        Self { id: Uuid::new_v4().to_string(), tick: 0, agent, tools, max_ticks, tick_interval }
    }

    pub async fn run_episode(&mut self) -> Result<()> {
        loop {
            self.tick += 1;

            let obs = Observation {
                episode_id: self.id.clone(),
                tick: self.tick,
                world_state: serde_json::json!({ "tick": self.tick }),
                memory_snapshot: serde_json::json!({}),
                goals: vec!["explore".to_string()],
            };

            let resp = self.agent.handle(AgentMessage::Observe(obs), &self.tools).await?;

            match resp {
                AgentResponse::ActionPlan(plan) => {
                    self.execute_plan(plan).await?;
                }
                AgentResponse::Idle => {}
                AgentResponse::Terminated => {
                    println!("Episode {} terminated", self.id);
                    break;
                }
            }

            if self.tick >= self.max_ticks {
                println!("Episode {} reached tick limit ({})", self.id, self.max_ticks);
                break;
            }

            sleep(self.tick_interval).await;
        }

        Ok(())
    }

    async fn execute_plan(&mut self, plan: ActionPlan) -> Result<()> {
        for action in plan.actions {
            match action.kind {
                ActionKind::ToolCall { tool_name } => {
                    let input = ToolInput { name: tool_name.clone(), args: action.payload.clone() };
                    match self.tools.call(input).await {
                        Ok(out) => {
                            println!("Tool {} -> {:?}", tool_name, out.result);
                            // Feed the real result back to the agent, closing
                            // the loop the original sketch left as a TODO.
                            self.agent.handle(AgentMessage::ToolResult(out), &self.tools).await?;
                        }
                        Err(e) => {
                            // A tool lookup failing doesn't abort the episode
                            // (e.g. a policy naming a tool that isn't
                            // registered in this particular runtime) -- logged,
                            // not silently swallowed, not fatal either.
                            println!("Tool {} failed: {e}", tool_name);
                        }
                    }
                }
                ActionKind::MemoryWrite => {
                    println!("MemoryWrite (not implemented in this runtime): {:?}", action.payload)
                }
                ActionKind::MemoryRead => {
                    println!("MemoryRead (not implemented in this runtime): {:?}", action.payload)
                }
                ActionKind::SpawnAgent { role } => {
                    println!("SpawnAgent (not implemented in this runtime): role={}", role)
                }
                ActionKind::SelfMutation { description } => {
                    println!("SelfMutation (not implemented in this runtime): {}", description)
                }
                ActionKind::NoOp => {}
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::runtime::agent::PlannerAgent;
    use crate::runtime::model_policy::Gh05t3Policy;
    use crate::runtime::tool_bus::KernelTickReportTool;

    #[tokio::test]
    async fn episode_runs_to_tick_limit_with_real_policy_and_tools() {
        let planner = PlannerAgent::new(Gh05t3Policy::new());
        let mut tools = ToolBus::new();
        tools.register_tool("kernel_tick_report", KernelTickReportTool);

        let mut manager = EpisodeManager::new(planner, tools, 3, Duration::from_millis(1));
        manager.run_episode().await.unwrap();

        assert_eq!(manager.tick, 3);
    }

    #[tokio::test]
    async fn episode_does_not_abort_when_a_tool_is_unregistered() {
        // Gh05t3Policy::infer always targets "kernel_tick_report" -- an
        // episode with an EMPTY ToolBus must still run to completion
        // (logged failure, not a crash), proving execute_plan's error
        // handling actually works rather than just being untested code.
        let planner = PlannerAgent::new(Gh05t3Policy::new());
        let tools = ToolBus::new();

        let mut manager = EpisodeManager::new(planner, tools, 2, Duration::from_millis(1));
        let result = manager.run_episode().await;

        assert!(result.is_ok());
        assert_eq!(manager.tick, 2);
    }
}
