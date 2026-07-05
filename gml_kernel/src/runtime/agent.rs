//! Cognitive actors around a ModelPolicy -- typed message passing between
//! the episode loop and whatever policy an Agent wraps.

use serde::{Deserialize, Serialize};
use uuid::Uuid;

use super::model_policy::{ActionPlan, ModelPolicy, Observation};
use super::tool_bus::{ToolBus, ToolOutput};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AgentMessage {
    Observe(Observation),
    ToolResult(ToolOutput),
    Shutdown,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AgentResponse {
    ActionPlan(ActionPlan),
    Idle,
    Terminated,
}

#[async_trait::async_trait]
pub trait Agent: Send + Sync {
    async fn handle(&mut self, msg: AgentMessage, tools: &ToolBus) -> anyhow::Result<AgentResponse>;
}

/// Wraps any ModelPolicy (real or otherwise) as an Agent the episode loop
/// can drive via typed messages.
pub struct PlannerAgent<P> {
    pub id: String,
    pub policy: P,
}

impl<P> PlannerAgent<P> {
    pub fn new(policy: P) -> Self {
        Self { id: Uuid::new_v4().to_string(), policy }
    }
}

#[async_trait::async_trait]
impl<P> Agent for PlannerAgent<P>
where
    P: ModelPolicy + Send + Sync,
{
    async fn handle(&mut self, msg: AgentMessage, _tools: &ToolBus) -> anyhow::Result<AgentResponse> {
        match msg {
            AgentMessage::Observe(obs) => Ok(AgentResponse::ActionPlan(self.policy.infer(obs))),
            AgentMessage::ToolResult(_) => Ok(AgentResponse::Idle),
            AgentMessage::Shutdown => Ok(AgentResponse::Terminated),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::runtime::model_policy::Gh05t3Policy;

    fn obs() -> Observation {
        Observation {
            episode_id: "test".into(),
            tick: 1,
            world_state: serde_json::json!({}),
            memory_snapshot: serde_json::json!({}),
            goals: vec![],
        }
    }

    #[tokio::test]
    async fn observe_returns_action_plan_from_real_policy() {
        let mut planner = PlannerAgent::new(Gh05t3Policy::new());
        let tools = ToolBus::new();

        let resp = planner.handle(AgentMessage::Observe(obs()), &tools).await.unwrap();
        assert!(matches!(resp, AgentResponse::ActionPlan(plan) if !plan.actions.is_empty()));
    }

    #[tokio::test]
    async fn shutdown_terminates() {
        let mut planner = PlannerAgent::new(Gh05t3Policy::new());
        let tools = ToolBus::new();

        let resp = planner.handle(AgentMessage::Shutdown, &tools).await.unwrap();
        assert!(matches!(resp, AgentResponse::Terminated));
    }

    #[tokio::test]
    async fn tool_result_is_idle() {
        let mut planner = PlannerAgent::new(Gh05t3Policy::new());
        let tools = ToolBus::new();

        let dummy_result = ToolOutput { name: "x".into(), result: serde_json::json!(null), success: true };
        let resp = planner.handle(AgentMessage::ToolResult(dummy_result), &tools).await.unwrap();
        assert!(matches!(resp, AgentResponse::Idle));
    }
}
