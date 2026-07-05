//! Standalone entrypoint for the async episode-loop runtime (see
//! gml_kernel::runtime). Separate binary from gml_kernel_main (src/main.rs,
//! the existing synchronous single-shot core-loop demo) -- this one runs
//! the same real engine repeatedly, through the typed Observation/
//! ActionPlan/ToolBus layer, on a real tokio runtime.

use std::time::Duration;

use gml_kernel::runtime::agent::PlannerAgent;
use gml_kernel::runtime::episode::EpisodeManager;
use gml_kernel::runtime::model_policy::Gh05t3Policy;
use gml_kernel::runtime::tool_bus::{EchoTool, KernelTickReportTool, ModelCallTool, ToolBus};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    println!("=== GH05T3 Sovereign Runtime (async episode loop) ===");

    let policy = Gh05t3Policy::new();
    let planner = PlannerAgent::new(policy);
    println!("Agent ID: {}", planner.id);

    let mut tools = ToolBus::new();
    tools.register_tool("echo", EchoTool);
    tools.register_tool("model_call", ModelCallTool);
    tools.register_tool("kernel_tick_report", KernelTickReportTool);

    let mut manager = EpisodeManager::new(planner, tools, 5, Duration::from_millis(50));
    println!("Episode ID: {}", manager.id);
    println!("----------------------------------");

    manager.run_episode().await?;

    println!("----------------------------------");
    println!("Episode finished after {} ticks", manager.tick);
    Ok(())
}
