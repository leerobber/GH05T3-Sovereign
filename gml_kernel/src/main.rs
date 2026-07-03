use gml_kernel::kernel::{KernelState, executor::execute_block};
use gml_kernel::gh05t3::core_loop::create_gh05t3_agent;

fn main() {
    let mut kernel = KernelState::new();
    let mut agent = create_gh05t3_agent();

    println!("=== GH05T3 Sovereign Substrate ===");
    println!("Agent ID: {}", agent.id);
    println!("Name: {}", agent.name);
    println!("Title: {}", agent.title);
    println!("----------------------------------");

    let core_loop = agent.core_loop.clone();
    execute_block(&core_loop, &mut agent, &mut kernel);

    println!("Execution complete. Kernel tick = {}", kernel.tick);

    println!("\n=== Glyph Trace ===");
    for entry in &kernel.trace {
        println!(
            "[tick {}] {} — {} | params: {}",
            entry.tick, entry.code, entry.title, entry.params
        );
    }

    println!("\n=== Short-term Memory ===");
    for (i, m) in agent.memory.short_term.iter().enumerate() {
        println!("[{}] {}", i, m);
    }
}
