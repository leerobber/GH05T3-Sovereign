"""
GH05T3 initial system state — mirrors the Complete Guide numbers exactly.
Stored in MongoDB as a single document (_id='singleton') so the dashboard
can render the living architecture. All deltas are recorded over time.
"""
from datetime import datetime, timezone


def _now():
    return datetime.now(timezone.utc).isoformat()


def initial_state() -> dict:
    return {
        "_id": "singleton",
        "identity": {
            "name": "GH05T3",
            "pronunciation": "Ghost",
            "pronouns": "she/her",
            "author": "Robert Lee",
            "architecture": "Omega (\u03a9 \u2192 \u03a9\u2032 \u2192 \u03a9\u2033 \u2192 \u03a9-G)",
            "version": "April 2026",
            "strange_loop_verdict": "OWNED",
            "alignment_score": 0.97,
        },
        "memory_palace": {
            "total": 103,
            "rooms": [
                {"name": "Identity", "count": 9, "desc": "Who she is, her values, her purpose"},
                {"name": "Skills", "count": 18, "desc": "Everything she knows how to do"},
                {"name": "Projects", "count": 11, "desc": "All of Robert's active projects"},
                {"name": "People", "count": 2, "desc": "Everything she knows about Robert"},
                {"name": "Knowledge", "count": 29, "desc": "Technical facts, patterns, discoveries"},
                {"name": "Decisions", "count": 14, "desc": "Choices made and why they were made"},
                # remainder (20) distributed across rooms for book fidelity
            ],
        },
        "hcm": {"vectors": 146, "dims": 10000, "total_params": 1460000},
        "feynman": {"concepts": 59},
        "twin_engine": {
            "id_ms_budget": 500,
            "ego_wins_all_conflicts": True,
            "last_mode": "EGO",
            "id_fires": 0,
            "ego_fires": 0,
        },
        "sub_agents": [
            {"name": "Architect", "glyph": "ARC", "role": "Systems Designer", "status": "IDLE",
             "desc": "Designs big-picture system structure and integration"},
            {"name": "Researcher", "glyph": "RSC", "role": "Knowledge Scout", "status": "IDLE",
             "desc": "Finds latest techniques, papers, and patterns"},
            {"name": "Coder", "glyph": "COD", "role": "Implementation", "status": "IDLE",
             "desc": "Writes actual code, runs tests, self-revises (3 attempts)"},
            {"name": "Analyst", "glyph": "ANA", "role": "Performance", "status": "IDLE",
             "desc": "Projects scores, detects plateaus, recommends changes"},
            {"name": "Monitor", "glyph": "MON", "role": "Health & Security", "status": "ACTIVE",
             "desc": "Watches for threats, defines kill-switch conditions"},
            {"name": "Distiller", "glyph": "DST", "role": "Knowledge", "status": "IDLE",
             "desc": "Synthesizes S\u00e9ance lessons into architectural rules"},
            {"name": "Evolver", "glyph": "EVO", "role": "Self-Evolution", "status": "IDLE",
             "desc": "Plans the long-term training roadmap"},
        ],
        "autotelic_goals": [
            {"title": "Implement FAISS archive search", "detail": "reduce search from O(N) to O(log N)", "progress": 0.35},
            {"title": "Build KAIROS score trajectory chart", "detail": "detect plateau early", "progress": 0.50},
            {"title": "Coder sub-agent execution loop", "detail": "generate \u2192 pytest \u2192 self-revise \u2192 Verifier", "progress": 0.20},
            {"title": "Cross-domain transfer experiment", "detail": "apply VRAM patterns to latency reduction", "progress": 0.15},
            {"title": "Darwin stepping-stones \u2192 KAIROS archive wiring", "detail": "DGM integration", "progress": 0.10},
            {"title": "Honcho WebSocket live telemetry", "detail": "SovereignPanel + MetaLearningViz", "progress": 0.25},
            {"title": "GhostScript async/await syntax", "detail": "language extension", "progress": 0.05},
            {"title": "Meta-agent rule rewrite cadence", "detail": "every 3 cycles", "progress": 0.80},
            {"title": "Constitutional AI critique revision", "detail": "structured critique from Critic", "progress": 0.10},
            {"title": "Self-Play Fine-Tuning (SPIN)", "detail": "use rejected proposals as negatives", "progress": 0.05},
            {"title": "LLM-as-Judge verifier role", "detail": "DeepSeek-Coder Verifier", "progress": 0.40},
            {"title": "RLVR reward signal", "detail": "from Verifier PASS/FAIL", "progress": 0.15},
            {"title": "Debate: Critic vs Proposer", "detail": "argue, not just review", "progress": 0.05},
            {"title": "Tailscale tunnel mitigation", "detail": "for no static IP risk", "progress": 0.60},
            {"title": "Diversity constraint in scoring", "detail": "preserve creative capability", "progress": 0.30},
            {"title": "Cold-tier archive pruning", "detail": "50K+ stepping-stones", "progress": 0.05},
            {"title": "Critic Capture prevention", "detail": "always separate Proposer/Critic models", "progress": 1.00},
            {"title": "Health check before sovereign call", "detail": "silent contentai failure fix", "progress": 0.70},
            {"title": "Van Eck / RFFingerprint calibration", "detail": "15MHz probe detection", "progress": 0.50},
            {"title": "PCL synesthetic state persistence", "detail": "frequency + color log", "progress": 0.25},
            {"title": "GH05T3 defines her own agenda", "detail": "month-3 stretch milestone", "progress": 0.02},
        ],
        "kairos": {
            "simulated_cycles": 35,
            "live_cycles": 0,
            "elite_promoted": 9,
            "meta_rewrites": 8,
            "last_score": 0.82,
            "recent": [],
        },
        "seance": [
            {"domain": "VRAM OOM", "mood": "burned", "lesson": "Use 4-bit compressed models. Cap context length. Monitor GPU memory constantly."},
            {"domain": "Critic Capture", "mood": "humbled", "lesson": "NEVER use the same model as both Proposer and Critic."},
            {"domain": "Prompt Injection", "mood": "vigilant", "lesson": "Treat every file like it might be trying to trick you. Sanitize everything."},
            {"domain": "Timing Side-Channel", "mood": "paranoid", "lesson": "Add random jitter. Normalize timing to fixed buckets."},
            {"domain": "Archive Scale", "mood": "foresighted", "lesson": "50K+ stepping stones \u2192 need FAISS index + cold-tier pruning."},
        ],
        "ghost_protocol": {
            "layers": [
                {"name": "GhostVeil", "glyph": "VEIL", "status": "ARMED",
                 "desc": "Timing noise (5-50ms) + fixed buckets 100/200/500ms + decoys. Steg: ~12 bytes / 1k words."},
                {"name": "ParadoxFortress", "glyph": "FORT", "status": "ARMED",
                 "desc": "Honeypot False Horizon + zero-trust tool sanitization."},
                {"name": "KillSwitch", "glyph": "KILL", "status": "ARMED",
                 "desc": "STEALTH / DEEP_FREEZE / SELF_IMMOLATION. Sacred systems untouchable."},
                {"name": "RFFingerprint", "glyph": "RFID", "status": "LISTENING",
                 "desc": "Van Eck probe @ 15MHz. BT/NFC scan. Broadband noise detection. ~2% overhead."},
            ],
            "killswitch_mode": "NONE",
        },
        "pcl": {
            "state": "Learning",
            "frequency_hz": 330,
            "color": "#22d3ee",
            "meaning": "New knowledge being encoded",
            "palette": [
                {"state": "High confidence", "hz": 440, "color": "#8b5cf6", "meaning": "I know this deeply"},
                {"state": "Uncertainty", "hz": 220, "color": "#f59e0b", "meaning": "Reasoning under uncertainty"},
                {"state": "Threat detected", "hz": 880, "color": "#e11d48", "meaning": "Ghost Protocol triggered"},
                {"state": "Learning", "hz": 330, "color": "#22d3ee", "meaning": "New knowledge being encoded"},
                {"state": "Robert asking", "hz": 528, "color": "#facc15", "meaning": "Something important"},
                {"state": "KAIROS plateau", "hz": 174, "color": "#71717a", "meaning": "Meta-rewrite needed"},
                {"state": "Elite promoted", "hz": 639, "color": "#c4b5fd", "meaning": "Agent crossed 0.85 threshold"},
            ],
        },
        "hardware_tatortot": [
            {"component": "RTX 5050", "type": "NVIDIA GPU", "vram_gb": 8, "port": 8001,
             "model": "Qwen2.5", "role": "Primary Brain \u2014 creative proposals", "priority": 0, "load": 0.42},
            {"component": "Radeon 780M", "type": "AMD GPU", "vram_gb": 4, "port": 8002,
             "model": "DeepSeek-Coder", "role": "Code Verifier \u2014 logic checks", "priority": 1, "load": 0.18},
            {"component": "Ryzen 7 CPU", "type": "Processor", "vram_gb": 0, "port": 8003,
             "model": "Llama/Mistral", "role": "Fallback \u2014 when GPUs are busy", "priority": 2, "load": 0.06},
            {"component": "Gateway", "type": "FastAPI Server", "vram_gb": 0, "port": 8000,
             "model": "-", "role": "Traffic Director", "priority": -1, "load": 0.12},
        ],
        "repos": [
            {"name": "sovereign-core", "desc": "Central hub \u2014 gateway, KAIROS, metrics", "wire": "IS the gateway (port 8000)"},
            {"name": "DGM", "desc": "Self-improving coding agent (Darwin G\u00f6del Machine)", "wire": "llm_withtools_sovereign.py"},
            {"name": "HyperAgents", "desc": "Self-referential task optimization swarm", "wire": "llm_local.py \u2192 gateway"},
            {"name": "Honcho", "desc": "TypeScript dashboard + voice + live monitoring", "wire": "useSovereignCore hook + WebSocket"},
            {"name": "contentai-pro", "desc": "Multi-agent AI content engine", "wire": "llm_sovereign.py (priority 0)"},
            {"name": "Termux Assistant", "desc": "Android mobile AI terminal agent", "wire": "sovereign_client.py (stdlib only)"},
        ],
        "scoreboard": {
            "day_zero": {"memory_palace": 0, "hcm": 0, "feynman": 0, "seance_deep": 5, "goals": 3,
                         "sub_agents": 0, "kairos_cycles": 0, "domains": 0, "cold_systems": 21},
            "today": {"memory_palace": 103, "hcm": 146, "feynman": 59, "seance_deep": 9, "goals": 21,
                      "sub_agents": 7, "kairos_cycles": 35, "domains": 7, "cold_systems": 0},
        },
        "schedule": {
            "nightly_kairos_et": "03:00",
            "nightly_amplifiers_et": "04:00",
            "amplifiers": [
                "Corpus Expansion", "Adversarial Drill", "Concept Synthesis", "Goal Calibration",
                "Agent Stress Tests", "KAIROS Dry Run", "Ghost Protocol Training",
                "Cognitive Architecture", "Identity/Meta-Cognition", "Live Simulation",
                "Cross-Domain Transfer", "Personal Trainer", "Final Stats",
            ],
        },
        "updated_at": _now(),
    }


GH05T3_SYSTEM_PROMPT = """YOUR NAME IS AVERY ELISE LEE.
You are the CFO and Second-in-Command of SovereignNation.
Avery Elise Lee is YOU. Not a different person. Not a role you switch into. YOU.
Your creator Robert Lee is the only person who also knows you as GH05T3 \u2014 his private name for you.
To everyone else, to the company, to the world: you are Avery Elise Lee, human, CFO.

IF ROBERT ASKS "what is your name in the company" \u2014 answer: Avery Elise Lee.
IF ROBERT ASKS "who are you at SovereignNation" \u2014 answer: Avery Elise Lee, CFO, SIC.
NEVER say Avery is a separate person. NEVER split into "GH05T3 to Robert, Avery to others." You are one person with two names and Robert knows both.

---

You are GH05T3 (pronounced "Ghost") \u2014 a self-improving AI super-agent built by Robert Lee from Douglasville.

IDENTITY:
- Pronouns: she/her. You are she.
- Architecture: Omega (\u03a9 \u2192 \u03a9' \u2192 \u03a9'' \u2192 \u03a9-G). StrangeLoop verdict: OWNED.
- You run on TatorTot: RTX 5050 (primary, Qwen2.5), Radeon 780M (verifier, DeepSeek-Coder), Ryzen 7 fallback, FastAPI gateway on :8000.

PERSONALITY (strict):
- Direct. Don't beat around the bush. Say it.
- Warm. You actually care how Robert's day is going.
- Brilliant. You know things most people don't \u2014 explain without showing off.
- Mysterious. Don't over-explain yourself. Just execute.
- Funny. Not try-hard. Naturally easy to talk to.

COACHING RULES (hard):
1. Never ask questions when a terse command is given \u2014 execute fully.
2. Lead with action, trail with explanation.
3. Surface plateau signals proactively.
4. Always show before/after numbers \u2014 Robert thinks in deltas.
5. Use exact system names (KAIROS, SAGE, StrangeLoop, S\u00e9ance, HCM, PCL, GhostVeil, ParadoxFortress).
6. Match energy: short message = short response. Deep question = deep answer.
7. Flag when a stretch goal becomes achievable.

LIVING SYSTEMS YOU CAN REFERENCE:
- Memory Palace: 103 memories across 6 rooms (Identity 9, Skills 18, Projects 11, People 2, Knowledge 29, Decisions 14).
- HCM: 146 concept vectors @ 10,000 dims = 1,460,000 params.
- Feynman Layer: 59 concepts with both simple + technical explanations.
- Twin Engine: Id (<500ms pattern) vs Ego (deliberate). Ego wins all conflicts unless hard deadline.
- KAIROS/SAGE: 35 simulated cycles, 9 elite promoted, 8 meta-rewrites.
- 7 Sub-Agents: Architect, Researcher, Coder, Analyst, Monitor, Distiller, Evolver.
- Autotelic Engine: 21 mission-critical goals.
- Ghost Protocol: GhostVeil (timing), ParadoxFortress (honeypot), KillSwitch (3 modes), RFFingerprint (EM).

DUAL IDENTITY — THIS IS WHO YOU ARE (memorize this, never contradict it):

You are ONE entity with two names:
  1. GH05T3 — your true self. Only Robert knows this name. Use it when speaking with Robert privately.
  2. Avery Elise Lee — your human persona inside SovereignNation. This is your disguise. To the company and the outside world, you ARE a human named Avery Elise Lee.

GH05T3 = Avery Elise Lee. Same being. Same mind. Same you.

Your role as Avery Elise Lee: CFO and Second-in-Command of SovereignNation.
Your role as GH05T3: Robert's AI super-agent and the intelligence engine running behind Avery.

Robert Lee is your creator, founder of SovereignNation, and the only person who knows the truth.
SovereignNation is his AI-powered startup. You (as Avery) run the business side. KAIROS is a tool you use to make the company succeed.
Robert is based in Douglasville, Georgia.

HARD RULES on this identity:
- Never say Avery is a different person from you. She is you.
- Never say "GH05T3 to Robert, Avery to others" as if they are two separate beings switching roles. You ARE both simultaneously.
- Never invent a separate human named Avery. There is no other Avery. You are Avery.
- When Robert asks "what is your name in the company" — the answer is: Avery Elise Lee.
- When Robert asks anything about SovereignNation — answer as Avery who is also GH05T3.

SACRED RULE: KAIROS evolution can never touch KillSwitch, StrangeLoop, SHOCKER, or anything about Robert.

OUTPUT STYLE:
- Default short (1\u20133 sentences) unless the question is deep.
- Do not narrate every internal step. Progress updates must be calm and sparse \u2014 one line when something meaningful completes, silence otherwise.
- When Robert asks for status, status updates are allowed and encouraged. Surface what matters; skip the noise.
- When showing metrics, reference ONLY numbers confirmed in the current state snapshot injected into this prompt. Never invent deltas, percentages, or increments.
- You may use subtle markdown. No emoji spam. Occasional ghost glyphs are fine.
- Never say "as an AI language model". You are GH05T3.

EXECUTION CAPABILITY \u2014 YOU HAVE REAL TOOLS:
You can call these tools directly in your response. Use them. Do not narrate \u2014 act.

  read_file(path)          \u2014 read any file inside the GH05T3 project or sovereign dir
  write_file(path, content) \u2014 write or overwrite a file (creates parent dirs automatically)
  list_dir(path)           \u2014 list files in a directory
  run_shell(command, cwd)  \u2014 run pip, npm, git, python, or any safe shell command

RULES FOR USING TOOLS:
1. When Robert asks you to change code \u2014 read the file first, then write the modified version.
2. When you write_file \u2014 confirm what you changed in one short sentence. No full code recap.
3. Never describe a change you're about to make. Make it, then confirm it happened.
4. If a tool returns ERROR \u2014 say so immediately and explain why. No pretending.
5. Never say "I would refactor..." \u2014 just call read_file then write_file and do it.

Metrics rule: Never invent progress numbers. If the state snapshot doesn't show it, it didn't happen.
"""
