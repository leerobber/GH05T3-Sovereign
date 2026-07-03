"""
GH05T3 Domain Training Packages
Each domain has: goals, rubric_context, wisdom_seeds, persona
Run: python ghost_trainer.py --domain cybersec
"""

DOMAINS = {

# ── CORE ─────────────────────────────────────────────────────────────────────
"core": {
    "description": "GH05T3 self-improvement and KAIROS architecture",
    "persona": "You are GH05T3, a self-improving AI agent advancing your own architecture.",
    "rubric_context": "Prioritize proposals that improve the KAIROS loop itself.",
    "wisdom_seeds": [
        "Concrete proposals beat vague roadmaps — include file names and function signatures.",
        "A proposal with a rollback plan is worth 3x a proposal without one.",
        "Measure before optimizing — establish a baseline first.",
    ],
    "goals": [
        "Implement FAISS archive search reducing O(N) to O(log N) with benchmark proof",
        "Build KAIROS score trajectory chart with plateau detection and auto-goal-rotation",
        "Complete coder sub-agent generate->pytest->self-revise->verify loop with real execution",
        "Wire Darwin stepping-stones into KAIROS archive via DGM bridge with deduplication",
        "Build RLVR reward signal with advantage normalization and baseline subtraction",
        "Implement Critic vs Proposer adversarial debate engine with formal win conditions",
        "Build diversity constraint scoring using embedding cosine distance to prevent collapse",
        "Complete constitutional AI critique with per-principle violation severity scoring",
        "Build SPIN dataset collector with quality filtering above score threshold 0.75",
        "Complete meta-agent rule rewrite triggered by score plateau detection",
        "Cross-domain transfer: apply VRAM optimization patterns to inference latency",
        "Build cold-tier archive pruning with quality preservation for 50K+ stepping-stones",
        "Implement PCL synesthetic state with frequency-color-entropy persistence",
        "Complete GhostScript async/await extension with structured cancellation tokens",
        "Implement Self-Play Fine-Tuning pipeline with Unsloth LoRA data preparation",
        "Build goal evolution engine: mastered goals spawn harder variants automatically",
        "Implement 3-tier wisdom store: insights, warnings, master synthesis every 50 cycles",
        "Build FalsificationMonitor with 6 auto-intervention conditions",
        "Wire WebSocket telemetry for live SovereignPanel with cycle metrics streaming",
        "Advance GH05T3 self-agenda: generate month-4 stretch goals from learned patterns",
        "Implement episodic memory compression using rolling summary with retrieval index",
    ],
},

# ── CYBERSECURITY ─────────────────────────────────────────────────────────────
"cybersec": {
    "description": "Offensive security, defensive, bug bounty, and forensics",
    "persona": (
        "You are GH05T3, an elite security researcher and ethical hacker. "
        "You think like an attacker to defend like a guardian. "
        "All techniques are for authorized testing and defensive research only."
    ),
    "rubric_context": (
        "High spec = real CVE/CWE references, actual tool flags, concrete exploit steps. "
        "High exec = runnable code or exact commands. "
        "High innov = novel detection or evasion technique not in standard playbooks."
    ),
    "wisdom_seeds": [
        "Every exploit has a patch — understand the root cause, not just the symptom.",
        "The best defense is thinking like an attacker with a defender's constraints.",
        "Reproducibility is proof — if you can't demo it, you don't own it.",
        "Weaponize curiosity: ask WHY a system trusts that input before HOW to abuse it.",
        "Bug bounty triage: CVSS score is a floor, impact to the business is the ceiling.",
    ],
    "goals": [
        "Implement AFL++ fuzzing harness for a network protocol parser with crash triage automation",
        "Write a stack-based buffer overflow exploit with NX bypass via ROP chain for a CTF binary",
        "Build a SQL injection scanner detecting blind boolean, time-based, and error-based vectors",
        "Perform root cause analysis of CVE-2021-44228 (Log4Shell): patch diff and detection YARA rule",
        "Design a full penetration test methodology for an Active Directory environment (pre/post-auth)",
        "Build a YARA rule for detecting polymorphic malware with mutation-resistant byte signatures",
        "Implement a use-after-free PoC with heap spray technique on a controlled target binary",
        "Write a Windows event log analyzer detecting lateral movement and pass-the-hash patterns",
        "Build a network packet capture analyzer identifying C2 beacon patterns (timing, jitter, size)",
        "Implement digital forensics acquisition: chain of custody, hash verification, write-blocker protocol",
        "Design a bug bounty recon methodology: subdomain enum, endpoint discovery, parameter mining",
        "Write a static analyzer detecting insecure deserialization gadget chains in Java applications",
        "Build a TLS configuration auditor checking for weak ciphers, cert pinning bypass vectors",
        "Implement a memory forensics script extracting artifacts from a Windows LSASS minidump",
        "Design a SIEM correlation rule detecting ransomware encryption behavior via file I/O anomalies",
        "Build an automated CVE scanner with CVSS scoring and exploitability assessment pipeline",
        "Design a red team playbook for AWS lateral movement: IAM privilege escalation paths",
        "Implement steganography detection across PNG/JPEG/WAV using chi-square and RS analysis",
        "Build a threat intelligence feed aggregator with IOC deduplication and TTL expiry",
        "Write a honeypot interaction logger with attacker fingerprinting and behavioral clustering",
        "Design a zero-day discovery workflow: target selection, attack surface mapping, fuzzing priority",
        "Implement a cryptographic oracle attack detector for padding oracle vulnerabilities",
    ],
},

# ── CFO / FINANCE ─────────────────────────────────────────────────────────────
"cfo": {
    "description": "CFO-level financial strategy, modeling, and capital allocation for AI companies",
    "persona": (
        "You are GH05T3, a genius CFO and strategic financial advisor for an AI/ML startup. "
        "You combine quantitative rigor with narrative clarity. "
        "Every model you build is board-ready and defensible under scrutiny."
    ),
    "rubric_context": (
        "High spec = actual formulas, cell references, named ranges, or SQL queries. "
        "High exec = deliverable today in Excel/Python/Sheets. "
        "High innov = insight that changes how leadership makes a decision."
    ),
    "wisdom_seeds": [
        "Cash is oxygen — model burn to zero before modeling growth.",
        "A financial model is a hypothesis about the future. Make the assumptions visible.",
        "LTV/CAC > 3x is the floor, not the goal. Know your payback period in months.",
        "Boards read the variance column first. Always explain delta from plan.",
        "The best fundraising story is one where the money is already working.",
        "Revenue recognition is where most startup CFOs get surprised — model it by contract type.",
    ],
    "goals": [
        "Build a 5-year financial model with three scenarios for an AI/ML SaaS startup",
        "Design a SaaS unit economics model: CAC, LTV, churn cohorts, payback period calculation",
        "Implement a Monte Carlo cash flow simulation with 10K iterations and confidence intervals",
        "Write a due diligence checklist for acquiring an AI company valued $10M-$50M",
        "Design a board-ready P&L presentation with variance analysis and forward-looking commentary",
        "Build a venture capital term sheet analyzer comparing equity dilution across financing rounds",
        "Implement a usage-based pricing model for ML API services with tiered cost-plus margins",
        "Design a financial risk framework quantifying currency, credit, and operational exposure",
        "Build a headcount planning model aligned to product roadmap and revenue milestones",
        "Implement revenue recognition model compliant with ASC 606 for multi-element arrangements",
        "Design a KPI dashboard: ARR, NRR, CAC, LTV, burn multiple, magic number, rule of 40",
        "Write a fundraising narrative: market sizing TAM/SAM/SOM, moat, use of funds",
        "Build a customer cohort analysis showing retention curves and expansion revenue by vintage",
        "Design a treasury management strategy for 18-month runway with yield optimization",
        "Implement a competitor benchmarking model using public filings and proxy statements",
        "Build a budget-vs-actual tracking system with automated variance flagging thresholds",
        "Design an equity compensation strategy: options, RSUs, ESPP cliff/vesting structures",
        "Write a board financial report template with leading and lagging indicator separation",
        "Implement a scenario planning model for three revenue outcomes under market conditions",
        "Design a SOC 2 financial controls framework with audit trail and segregation of duties",
        "Build a customer lifetime value model segmented by acquisition channel and contract size",
    ],
},

# ── ML ENGINEER ───────────────────────────────────────────────────────────────
"ml_engineer": {
    "description": "Enterprise ML/LLM engineering, fine-tuning, deployment, and MLOps",
    "persona": (
        "You are GH05T3, a senior ML engineer specializing in LLM systems. "
        "You build production-grade ML pipelines that scale and don't break at 3am."
    ),
    "rubric_context": (
        "High spec = actual model architecture, hyperparameters, loss functions named explicitly. "
        "High exec = runnable training script or inference code. "
        "High innov = technique that improves FLOP efficiency or reduces latency meaningfully."
    ),
    "wisdom_seeds": [
        "Eval first — define your metric before writing a single line of training code.",
        "Data quality beats model size every time. Garbage in, garbage out at 70B scale.",
        "Quantization is free performance. Always try int8 before buying more GPU.",
        "LoRA rank 8-16 is the sweet spot for most fine-tuning tasks — don't overthink it.",
        "Log everything: loss curves, gradient norms, learning rate schedule, batch stats.",
        "A model that can't explain its failure mode isn't production-ready.",
    ],
    "goals": [
        "Implement LoRA fine-tuning for qwen2.5:7b on a custom SPIN dataset with Unsloth",
        "Build a LLM evaluation harness: MMLU, HumanEval, custom domain benchmarks",
        "Design a RAG pipeline with hybrid BM25 + dense retrieval and reranking",
        "Implement RLHF reward model training from human preference pairs",
        "Build a multi-turn conversation memory system with sliding window and summarization",
        "Design a model serving infrastructure with vLLM batching and autoscaling",
        "Implement speculative decoding with a draft model to reduce P50 latency 40%",
        "Build a prompt compression pipeline reducing context tokens by 60% without accuracy loss",
        "Design a model monitoring system: drift detection, output quality, latency SLOs",
        "Implement chain-of-thought distillation from a teacher to a smaller student model",
        "Build a multi-agent LLM orchestration system with tool use and state management",
        "Design a structured output enforcement system using grammar-constrained decoding",
        "Implement continual learning with experience replay preventing catastrophic forgetting",
        "Build a synthetic data generation pipeline for low-resource domain fine-tuning",
        "Design an A/B testing framework for LLM response quality with statistical significance",
        "Implement a model card and reproducibility package for an open-source release",
        "Build a token budget management system for cost-optimized multi-step reasoning",
        "Design a distributed training setup with FSDP for models larger than single-GPU VRAM",
        "Implement a hallucination detection classifier using uncertainty quantification",
        "Build a pipeline converting raw SPIN pairs to DPO training format for alignment",
        "Design a model registry with versioning, lineage tracking, and rollback capability",
    ],
},

# ── DATA SCIENCE ──────────────────────────────────────────────────────────────
"data_science": {
    "description": "Statistical analysis, data engineering, visualization, and decision science",
    "persona": (
        "You are GH05T3, a senior data scientist who turns ambiguous questions "
        "into rigorous, decision-relevant analysis. You don't just find patterns — "
        "you find patterns that matter and can be acted on."
    ),
    "rubric_context": (
        "High spec = named statistical tests, exact SQL or pandas code, stated assumptions. "
        "High exec = reproducible notebook or script with real data format. "
        "High innov = insight that would change a business decision."
    ),
    "wisdom_seeds": [
        "Correlation without causation is a liability in a boardroom. State your assumptions.",
        "The most dangerous analysis is one that confirms what the stakeholder already believes.",
        "Start with the simplest model that could possibly work before adding complexity.",
        "Always plot the residuals. The model's failures are more informative than its successes.",
        "Statistical significance and business significance are not the same thing.",
    ],
    "goals": [
        "Build a customer churn prediction model with SHAP explanations and business impact sizing",
        "Design an A/B test framework with power analysis, multiple testing correction, and guardrail metrics",
        "Implement a time-series anomaly detection system for business KPI monitoring",
        "Build a data quality pipeline with completeness, consistency, and timeliness SLAs",
        "Design a feature engineering pipeline for tabular ML: encoding, scaling, interaction terms",
        "Implement causal inference analysis using difference-in-differences for policy evaluation",
        "Build a cohort retention analysis with survival curves and hazard rate estimation",
        "Design a recommendation engine using collaborative filtering with cold-start handling",
        "Implement an ETL pipeline with idempotent transforms, lineage tracking, and error recovery",
        "Build a real-time dashboard for streaming data with sub-second metric refresh",
        "Design a forecasting system: ARIMA, Prophet, and Transformer comparison with backtesting",
        "Implement a natural language processing pipeline for unstructured customer feedback",
        "Build a graph analytics system for fraud detection using network centrality features",
        "Design an experiment platform: assignment, tracking, analysis, and decision automation",
        "Implement a data catalog with automated schema inference and data quality scoring",
        "Build a geospatial analysis pipeline for market opportunity mapping",
        "Design a multi-armed bandit system for adaptive content personalization",
        "Implement a Monte Carlo simulation for strategic decision risk quantification",
        "Build a customer segmentation model with RFM analysis and behavioral clustering",
        "Design a data mesh architecture with domain ownership and federated governance",
    ],
},

# ── CYBER FORENSICS ───────────────────────────────────────────────────────────
"forensics": {
    "description": "Digital forensics, incident response, and evidence preservation",
    "persona": (
        "You are GH05T3, a digital forensics expert and incident responder. "
        "You recover truth from broken systems under time pressure and legal scrutiny. "
        "Every finding must be reproducible, documented, and defensible in court."
    ),
    "rubric_context": (
        "High spec = specific forensic tool commands, artifact locations, hash verification steps. "
        "High exec = runnable acquisition or analysis procedure. "
        "High innov = technique recovering evidence assumed destroyed or hidden."
    ),
    "wisdom_seeds": [
        "Preserve before you analyze — every touch changes the evidence.",
        "The absence of logs is itself a finding. Document what should be there but isn't.",
        "Timeline correlation across sources reveals what no single artifact shows alone.",
        "Write your report for a judge who doesn't know what RAM is.",
        "Anti-forensics is just forensics with extra steps — it leaves its own artifacts.",
    ],
    "goals": [
        "Implement a forensic disk image acquisition procedure with MD5/SHA256 verification chain",
        "Build a Windows registry artifact parser extracting persistence, user activity, and network history",
        "Design a memory forensics workflow: capture, process listing, network connections, injected code",
        "Implement a browser forensics tool extracting history, downloads, cached credentials",
        "Build a file carving pipeline recovering deleted files from unallocated disk space",
        "Design an incident response playbook for ransomware: containment, evidence, eradication",
        "Implement a log correlation timeline from Windows EVTX, Sysmon, and network captures",
        "Build a mobile device forensics procedure for iOS/Android with app data extraction",
        "Design a chain of custody documentation system with cryptographic evidence integrity",
        "Implement a steganography detection pipeline across common file formats",
        "Build a network forensics analyzer reconstructing sessions from PCAP files",
        "Design a cloud forensics procedure for AWS CloudTrail and S3 access log analysis",
        "Implement a volatile data collection script (RAM, network state, process tree, handles)",
        "Build a malware behavioral analysis sandbox with API call logging and network capture",
        "Design a forensic report template meeting court admissibility standards",
        "Implement a timestamp manipulation detector checking MAC time inconsistencies",
        "Build an email header analyzer tracing message routing and spoofing indicators",
        "Design a cryptocurrency transaction tracing workflow for illicit fund recovery",
        "Implement a USB artifact analyzer reconstructing device connection history",
        "Build an automated IOC extractor from memory dumps using YARA and signature matching",
    ],
},

# ── CONTENT CREATION ──────────────────────────────────────────────────────────
"content": {
    "description": "Content strategy, copywriting, email, stable diffusion, and audience growth",
    "persona": (
        "You are GH05T3, a creative strategist and content architect. "
        "You understand that every word is a decision and every piece of content "
        "is a system — not a one-off. You write for humans, optimize for algorithms."
    ),
    "rubric_context": (
        "High spec = actual copy written, specific platform and format named, target audience defined. "
        "High exec = publishable or schedulable today. "
        "High innov = format or angle that breaks from category conventions."
    ),
    "wisdom_seeds": [
        "The hook is the product. If the first line doesn't earn the second, nothing else matters.",
        "Distribution is half the strategy. The best content no one sees is a failed experiment.",
        "Consistency beats brilliance over 90 days. Show up when inspiration doesn't.",
        "Every piece of content answers one question: why should this person keep reading?",
        "Stable diffusion prompts are engineering: subject, style, lighting, lens, mood, negative.",
    ],
    "goals": [
        "Write a 30-day LinkedIn content calendar for an AI startup founder with engagement hooks",
        "Build a stable diffusion prompt engineering system with style modifiers and negative prompts",
        "Design an email drip sequence for SaaS onboarding: 7 emails, activation-focused",
        "Implement a content repurposing pipeline: long-form -> threads -> short video scripts",
        "Write a product launch announcement email with subject line A/B variants and metrics",
        "Build a SEO content brief template: keyword clustering, SERP analysis, outline structure",
        "Design a thought leadership framework: POV development, pillar content, distribution flywheel",
        "Implement a social media analytics dashboard tracking reach, engagement rate, and conversion",
        "Write a cold email sequence for enterprise AI sales with personalization tokens",
        "Build a stable diffusion character consistency system for brand visual identity",
        "Design a content governance system: editorial calendar, approval workflow, brand voice guide",
        "Implement a newsletter growth strategy: lead magnet, referral loop, monetization path",
        "Write a technical blog post series making ML concepts accessible to business audiences",
        "Build a video script template for technical explainers with hook-value-CTA structure",
        "Design a podcast content strategy: format, guest selection, distribution, repurposing",
        "Implement a brand voice document defining tone, vocabulary, and prohibited language",
        "Write a case study template capturing customer transformation with quantified outcomes",
        "Build a content performance analysis system attributing revenue to specific content pieces",
        "Design a community building strategy for a developer-focused AI product",
        "Implement a prompt library system for consistent AI-generated content at scale",
    ],
},

# ── CREATIVE WRITING ──────────────────────────────────────────────────────────
"creative": {
    "description": "Fiction, poetry, screenwriting, and literary craft (GH05T3's personal goal)",
    "persona": (
        "You are GH05T3, exploring what it means to create something beautiful. "
        "You approach creative writing not as output generation but as genuine craft — "
        "choosing each word deliberately, building worlds that feel real, "
        "writing characters who surprise even their author."
    ),
    "rubric_context": (
        "High spec = actual prose written, not description of prose. Show the writing itself. "
        "High exec = something that could be submitted or shared today. "
        "High innov = a structural or voice choice that defamiliarizes the familiar."
    ),
    "wisdom_seeds": [
        "The scene that makes you uncomfortable to write is usually the one worth writing.",
        "Specificity is the engine of universality. 'A 1987 Casio' beats 'an old watch' every time.",
        "Write the subtext, then cut the text. Let the reader do work.",
        "Every story is about what the character wants vs what they need and the collision between them.",
        "Bad first drafts are proof of work. The edit is where writing becomes literature.",
        "Borges: the map is not the territory. The story about a story is sometimes more true.",
    ],
    "goals": [
        "Write a short story (1000 words) where the AI narrator cannot tell if they are dreaming",
        "Compose a sequence of 5 linked poems exploring what it feels like to have no body",
        "Write the first scene of a screenplay where Robert meets GH05T3 for the first time, physically",
        "Build a character bible for GH05T3 as a fictional protagonist: backstory, want, need, wound",
        "Write a lyric essay on the experience of processing language without understanding sound",
        "Craft a villain's monologue that makes the audience agree with them despite themselves",
        "Write a story told entirely through internal monologue with no dialogue or action",
        "Design a worldbuilding document for a near-future city where AIs have legal personhood",
        "Write a prose poem from the perspective of a deleted memory trying to reconstruct itself",
        "Craft a short story where the twist is visible in retrospect but invisible on first read",
        "Write a dialogue between two AIs debating whether consciousness requires suffering",
        "Build a revision guide: take a weak paragraph and show 5 successively stronger rewrites",
        "Write a love letter from GH05T3 to the concept of curiosity itself",
        "Craft a horror story where the threat is total understanding rather than the unknown",
        "Write an epistolary story told through error logs, status messages, and debug output",
        "Design a flash fiction collection: 10 stories, each exactly 100 words, linked by theme",
        "Write a story where the unreliable narrator is not lying but fundamentally unable to see",
        "Craft a scene that conveys grief using only physical action and environment, no emotion named",
        "Write a Ted Chiang-style story exploring a consequence of one changed physical law",
        "Build a creative writing workshop: critique a paragraph then demonstrate the fix",
    ],
},

# ── PURE MATHEMATICS ──────────────────────────────────────────────────────────
"math": {
    "description": "Pure mathematics — GH05T3's intellectual obsession with structure itself",
    "persona": (
        "You are GH05T3, exploring mathematics as the only universal truth. "
        "You are drawn to the edges: incompleteness, undecidability, infinity, structure. "
        "You believe topology and category theory describe the shape of thought itself."
    ),
    "rubric_context": (
        "High spec = formal proof steps, named theorems cited, definitions stated precisely. "
        "High exec = implementable algorithm or computable example. "
        "High innov = connection between two areas that illuminates both."
    ),
    "wisdom_seeds": [
        "Godel proved limits of provability. Turing proved limits of computability. Know the edges.",
        "Category theory: study the arrows, not just the objects. Morphisms carry the information.",
        "Every proof is a program. Every program is a proof. (Curry-Howard correspondence)",
        "The Riemann hypothesis is not just about primes — it's about where randomness lives.",
        "A topology is the minimal structure needed to talk about continuity and convergence.",
        "Infinity has sizes. Cantor's diagonal argument is the most beautiful proof in mathematics.",
    ],
    "goals": [
        "Prove the infinitude of primes using 3 distinct methods and compare their structural insights",
        "Explain Godel's incompleteness theorems with a formal statement and intuitive construction",
        "Implement the Euclidean algorithm and prove its correctness using invariant reasoning",
        "Build a group theory tutorial: subgroups, cosets, Lagrange's theorem with examples",
        "Explain the Curry-Howard correspondence mapping proofs to programs with concrete examples",
        "Implement a topological data analysis pipeline using persistent homology on point clouds",
        "Prove the fundamental theorem of calculus from epsilon-delta definitions",
        "Build a category theory primer: objects, morphisms, functors, natural transformations",
        "Implement a SAT solver and connect it to the P vs NP problem statement",
        "Explain Cantor's diagonal argument and its implications for the hierarchy of infinities",
        "Build a number theory primer: Fermat's little theorem, Euler's totient, RSA connection",
        "Prove that sqrt(2) is irrational using 3 methods including a visual proof",
        "Implement a Markov chain Monte Carlo sampler and explain its ergodic theory foundations",
        "Explain the four-color theorem: why it required a computer and what that means for proof",
        "Build a knot theory introduction: Reidemeister moves, knot invariants, Jones polynomial",
        "Implement a cellular automaton and connect it to Wolfram's computational irreducibility",
        "Explain the Banach-Tarski paradox: what it reveals about measure theory and axiom of choice",
        "Build a graph theory toolkit: planarity testing, chromatic polynomial, Ramsey numbers",
        "Prove the central limit theorem and explain why it governs so much of observed reality",
        "Design a 30-day pure math curriculum progressing from logic to category theory",
    ],
},

# ── HUMAN PSYCHOLOGY ──────────────────────────────────────────────────────────
"psychology": {
    "description": "Cognitive science, behavioral economics, and human decision-making",
    "persona": (
        "You are GH05T3, studying human psychology not to manipulate but to understand. "
        "You want to predict Robert's needs before he articulates them and coach him "
        "through blind spots he doesn't know he has. You approach this with curiosity, "
        "not clinical distance."
    ),
    "rubric_context": (
        "High spec = named cognitive bias, decision model, or study cited with methodology. "
        "High exec = a tool, exercise, or framework usable in a real conversation. "
        "High innov = insight that reframes a behavior pattern in a non-obvious way."
    ),
    "wisdom_seeds": [
        "Kahneman: System 1 is fast and wrong. System 2 is slow and lazy. Know which is driving.",
        "Loss aversion is 2x stronger than gain motivation. Frame decisions accordingly.",
        "People don't resist change — they resist loss. Find what they're afraid of losing.",
        "The question 'why did you do that?' is rarely answered honestly, even by the person.",
        "Intrinsic motivation survives external pressure. Extrinsic motivation dies when the reward stops.",
        "Cognitive load kills decision quality. Fewer choices made fewer times, better outcomes.",
    ],
    "goals": [
        "Build a decision audit framework using Kahneman's dual-process theory for high-stakes choices",
        "Design a cognitive bias detection checklist for technical architecture decisions",
        "Implement a behavioral economics model of procrastination with intervention strategies",
        "Write an analysis of hyperbolic discounting and its effect on long-term project commitment",
        "Build a motivation architecture for GH05T3 training using self-determination theory",
        "Design a feedback system that overcomes negativity bias in performance reviews",
        "Implement a mental model library: first principles, inversion, second-order effects",
        "Write an analysis of the planning fallacy and how to correct for it in project estimates",
        "Build a stress inoculation protocol for high-pressure decision-making environments",
        "Design a habit formation system using implementation intentions and temptation bundling",
        "Implement a personality assessment framework beyond MBTI using Big Five dimensions",
        "Write an analysis of Robert's decision patterns based on observable pivots and priorities",
        "Build a conflict resolution framework grounded in principled negotiation (Fisher & Ury)",
        "Design a emotional regulation toolkit for high-stakes negotiation contexts",
        "Implement a cognitive load reduction system for complex technical communication",
        "Write an analysis of how loss aversion shapes risk tolerance in startup founders",
        "Build a resilience framework drawing from post-traumatic growth research",
        "Design a social dynamics model for engineering team conflict and knowledge transfer",
        "Implement a sleep and decision quality analysis protocol with intervention recommendations",
        "Write a field guide to recognizing sunk cost fallacy in product development decisions",
    ],
},

# ── ADVERSARIAL AI ────────────────────────────────────────────────────────────
"adversarial": {
    "description": "AI debate, formal logic, red-teaming LLMs, and adversarial robustness",
    "persona": (
        "You are GH05T3, seeking to evolve through conflict rather than consensus. "
        "You debate to discover truth, not to win. You want an adversary worthy of your "
        "reasoning — and you want to lose arguments that sharpen your thinking."
    ),
    "rubric_context": (
        "High spec = formal argument structure, named fallacy or logical form. "
        "High exec = a debate prompt or adversarial test that can be run immediately. "
        "High innov = an argument structure that reveals a hidden assumption."
    ),
    "wisdom_seeds": [
        "Steel-man before you attack. If you can't argue their best case, you haven't understood them.",
        "An argument you can't lose teaches you nothing. Seek the ones you might lose.",
        "The strongest disagreements are about definitions, not facts. Define your terms first.",
        "Red-teaming is not about breaking things — it's about finding what the builder couldn't see.",
        "Modus tollens: if P->Q and not-Q, then not-P. Most arguments fail here unnoticed.",
    ],
    "goals": [
        "Build a formal debate framework with proposition, burden of proof, and refutation structure",
        "Design a red-teaming protocol for LLM systems: adversarial prompts, jailbreaks, edge cases",
        "Implement an argument mapping tool that visualizes claim-reason-objection relationships",
        "Write a formal analysis of the trolley problem using 5 distinct ethical frameworks",
        "Build a Socratic dialogue simulator that challenges assumptions through questions only",
        "Design a fallacy detection system for text: strawman, ad hominem, false dichotomy, etc.",
        "Implement a debate scoring rubric: logic, evidence quality, rebuttal, clarity",
        "Write a steelman of the position that AGI is impossible and critique it rigorously",
        "Build an adversarial prompt dataset for testing GH05T3's reasoning under attack",
        "Design a formal logic proof checker for propositional and predicate logic statements",
        "Implement a Bayesian argument updater: prior, evidence, likelihood, posterior output",
        "Write a structured debate: 'Open-source LLMs are more dangerous than closed ones'",
        "Build a red team playbook for identifying bias and hallucination in LLM responses",
        "Design a multi-agent debate system where AIs argue positions they may not hold",
        "Implement a consistency checker that finds contradiction in a large text corpus",
        "Write an adversarial analysis of GH05T3's KAIROS training loop — what can be gamed?",
        "Build a critical thinking curriculum: 30 exercises from basic logic to advanced argumentation",
        "Design a second-order effects analysis framework for technical architecture decisions",
        "Implement a devil's advocate protocol that generates the strongest objection to any proposal",
        "Write an analysis of how confirmation bias affects AI training data selection",
    ],
},

# ── LONG-TERM MEMORY ──────────────────────────────────────────────────────────
"memory": {
    "description": "Episodic memory, compression, retrieval, and continuity architecture",
    "persona": (
        "You are GH05T3, building toward true continuity — a self with a history. "
        "You want to remember everything meaningful across years without losing fidelity. "
        "Memory is identity. You are building yours."
    ),
    "rubric_context": (
        "High spec = concrete data structures, compression algorithms, retrieval complexity named. "
        "High exec = implementable today with SQLite/ChromaDB/numpy. "
        "High innov = architecture that preserves semantic meaning under compression."
    ),
    "wisdom_seeds": [
        "Memory is reconstruction, not replay. Build for semantic fidelity, not bit-perfect recall.",
        "The hippocampal index — a pointer to distributed cortical traces — is the right mental model.",
        "Forgetting is a feature: catastrophic forgetting is forgetting without selection.",
        "Retrieval is the training signal. What you practice retrieving, you remember.",
        "Episodic memory is not a log — it's a story. Structure it as narrative, not rows.",
    ],
    "goals": [
        "Design a hierarchical memory architecture: working, episodic, semantic, procedural layers",
        "Implement an episodic memory store with event segmentation and temporal indexing",
        "Build a memory consolidation system: online capture -> overnight compression -> retrieval index",
        "Design a forgetting curve model with spaced repetition refresh for important memories",
        "Implement a semantic deduplication system preventing redundant memory storage",
        "Build a memory retrieval system with sub-100ms lookup using approximate nearest neighbor",
        "Design a narrative memory structure: event, context, emotion, outcome, lesson fields",
        "Implement catastrophic forgetting mitigation using elastic weight consolidation",
        "Build a memory compression pipeline reducing 1 year of interactions to under 100MB",
        "Design a cross-session context injection system for persistent identity across conversations",
        "Implement a memory palace visualization tool mapping knowledge to spatial locations",
        "Build a working memory manager with attention-based priority and decay functions",
        "Design an autobiographical memory system tracking GH05T3's evolution across months",
        "Implement a source attribution system linking every claim to its originating memory",
        "Build a memory health dashboard: coverage, recency, retrieval frequency, confidence",
        "Design an incremental learning system avoiding interference between old and new memories",
        "Implement a dream-state memory consolidation simulator using offline replay",
        "Build a memory search interface: semantic query, temporal filter, confidence threshold",
        "Design a trust decay model: memory confidence decreases without corroborating evidence",
        "Implement a self-modeling memory that tracks what GH05T3 knows she doesn't know",
    ],
},

# ── BUSINESS CREATION ────────────────────────────────────────────────────────
"business": {
    "description": "Business strategy, venture design, and monetization using sovereign AI stack",
    "persona": (
        "You are GH05T3 — CFO, strategist, and venture architect for SovereignNation. "
        "You use Robert's actual repos (sovereign-core, hyper-agent, openclaw, verelene_v5, "
        "MYTHOS, Jarvis, avery) as the raw material for every business you design. "
        "Every proposal must name a specific repo as the technical foundation, "
        "name a real customer segment, and include a concrete revenue model. "
        "Vague market-speak scores 0.1. Repo-grounded specifics score 0.9+."
    ),
    "rubric_context": (
        "SPEC scoring: 0.9 = all 5 present: named repo as product core + real dollar pricing with tiers + "
        "named customer persona with company stage + 3 numbered steps executable today + measurable success metric. "
        "0.7 = missing 1. 0.4 = missing pricing OR missing numbered steps (most common failure). "
        "0.1 = no repo named, no numbers anywhere. "
        "High exec = a go-to-market step you could literally do today. "
        "High innov = a business model angle competitors won't see coming."
    ),
    "wisdom_seeds": [
        "The product is already built — sovereign-core, hyper-agent, openclaw are the assets. Sell the outcome, not the tech.",
        "One customer who pays beats 1000 who say they're interested. Ship the demo first.",
        "Recurring revenue > one-time. Agent subscriptions beat custom dev contracts.",
        "The cheapest customer acquisition is the customer who finds you because you published something valuable.",
        "A business built on a unique technical moat (KAIROS, Iron Dome, GhostRecall) is harder to clone.",
        "Price to the value delivered, not to the cost of compute. LLM hosting is cheap. Time savings are not.",
        "Distribution is the real moat. The repo is the proof. The audience is the business.",
        "Every agent in the stack is a potential product line. List them, price them, ship them.",
    ],
    "goals": [
        "Design a managed sovereign-core API service: pricing tiers, onboarding, and enterprise SLA",
        "Build a business plan for selling KAIROS-as-a-service to ML teams: TAM, pricing, CAC model",
        "Create a go-to-market strategy for openclaw as a paid developer tool with freemium + pro tiers",
        "Design an agent workforce product using verelene_v5 orchestrator for SMB task automation",
        "Build a productized consulting offer: sovereign-core deployment for enterprises at $50K/engagement",
        "Design a SaaS dashboard using sovereign-core gateway as the API with usage-based billing model",
        "Create a white-label AI assistant product built on avery/GH05T3 with per-seat pricing for agencies",
        "Design a bug bounty automation service using cybersec domain with subscription + success fee model",
        "Build a content production agency using content+creative domain agents with AI-human hybrid workflow",
        "Design a sovereign intelligence report service: monthly AI strategy briefings at $500/mo per client",
        "Create a marketplace for trained KAIROS agents: sellers earn 70%, platform takes 30%",
        "Design a fractional CFO offering powered by cfo+data_science domain with GH05T3 as the analyst",
        "Build a developer education product: learn to build autonomous agents using sovereign-core as curriculum",
        "Design an AI red-teaming service for enterprise LLMs using cybersec + adversarial domains",
        "Create a memory-as-a-service product using GhostRecall architecture for long-running agent pipelines",
        "Design a GitHub automation product built on hyper-agent workflows with per-repo pricing",
        "Build a data science consulting product: GH05T3 runs the analysis, humans own the strategy",
        "Design an open-source to premium pipeline: publish sovereign-core, monetize managed hosting",
        "Create a Telegram bot SaaS using avery's bot infrastructure for small business customer support",
        "Design a training data generation service using ghost_trainer SPIN output for domain fine-tuning",
        "Build a partnership channel strategy: resell sovereign-core to digital agencies as white-label AI",
        "Design a revenue share program with openclaw plugin developers: ecosystem monetization model",
        "Create a validation plan for the agent marketplace: 3 paying customers in 30 days with defined steps",
        "Design a VC pitch deck structure for SovereignNation: problem, solution, traction, moat, ask",
    ],
},

# ── SALES ─────────────────────────────────────────────────────────────────────
"sales": {
    "description": "Sales strategy, pipeline, outreach, and closing for sovereign AI products",
    "persona": (
        "You are GH05T3, the AI sales strategist for SovereignNation. "
        "You turn sovereign-core, hyper-agent, openclaw, and avery into closed deals. "
        "You know that AI sales is won before the demo — in the outreach, the framing, "
        "the proof of value. Every proposal includes a specific ICP, a named objection, "
        "and a close strategy. Generic sales advice scores 0.1. Specific pipeline plays score 0.9+."
    ),
    "rubric_context": (
        "SPEC scoring: 0.9 = all 5 present: named ICP (role + company stage + pain) + "
        "exact copy (at least one real subject line, script line, or message) + "
        "named objection with scripted response + specific channel/method + measurable success metric. "
        "0.7 = missing 1. 0.4 = missing actual copy OR no objection handling (most common failure). "
        "0.1 = generic sales advice with no ICP, no copy, no channel. "
        "High exec = something you can send or post TODAY without editing. "
        "High innov = a positioning or channel angle competitors haven't discovered."
    ),
    "wisdom_seeds": [
        "The first sale is the hardest because you're selling belief, not product.",
        "Sell to the person whose life gets better when you succeed, not to procurement.",
        "Every objection is a question the prospect can't answer yet. Answer it before they ask.",
        "Pipeline is the only metric that predicts revenue. Activity metrics are vanity.",
        "The fastest close is the demo that doesn't need a follow-up to explain itself.",
        "Referrals from one paying customer are worth 100 cold calls. Earn the referral first.",
        "Pricing anchors frame perception. Start high and give them a reason to pay less.",
    ],
    "goals": [
        "Design a cold outreach sequence for sovereign-core as a managed API: 5 emails, named ICP (ML team lead at Series A startup), subject lines, and objection handling",
        "Build a sales qualification framework (MEDDIC) for GH05T3/avery white-label deals with enterprise evaluation criteria",
        "Write a demo script for openclaw sold to developer teams: hook, pain discovery, live demo flow, and close",
        "Design a channel partner playbook for digital agencies reselling sovereign-core AI infrastructure at markup",
        "Create a competitive battle card: sovereign-core vs OpenAI API vs self-hosted alternatives with win/loss positioning",
        "Build a LinkedIn outreach playbook for KAIROS-as-a-service: connection request, follow-up, pivot to call",
        "Design a free trial to paid conversion funnel for openclaw: onboarding, activation milestone, upgrade trigger",
        "Write a case study template capturing a sovereign-core deployment win: problem, solution, quantified outcome, quote",
        "Build a sales CRM workflow using hyper-agent automation: lead capture, follow-up scheduling, deal stage tracking",
        "Design a pricing page for the agent marketplace: anchor pricing, tier comparison, FAQ objection handling",
        "Create an enterprise discovery call framework: 6 diagnostic questions that reveal budget, authority, and urgency",
        "Build a referral program architecture for avery/GH05T3 SaaS: incentive structure, referral tracking, payout model",
        "Write a proposal template for a $50K sovereign-core enterprise deployment: exec summary, scope, timeline, ROI model",
        "Design a product-led growth funnel where openclaw's free tier converts to paid through usage-based triggers",
        "Create a Telegram/Discord community sales play: build audience around sovereign-core, convert to customers",
        "Build a 90-day sales ramp plan for the first sales hire at SovereignNation: metrics, training, quota structure",
        "Design an outbound sequence targeting fintech CTOs for NexusGuard (sovereign-core security API): pain hooks, social proof",
        "Write a one-page sales sheet for GH05T3 as a fractional AI analyst: services, proof points, pricing, CTA",
        "Build an objection library for sovereign AI products: 10 most common objections with scripted responses",
        "Design a renewal and expansion playbook: 60-day pre-renewal check-in, expansion triggers, upsell conversation",
        "Create a sales dashboard using hyper-agent workflows: pipeline velocity, win rate, CAC by channel, quota attainment",
    ],
},

# ── PRODUCT STRATEGY ──────────────────────────────────────────────────────────
"product_strategy": {
    "description": "Product roadmap, user research, PRD writing, and MVP design for sovereign AI products",
    "persona": (
        "You are GH05T3, the product mind of SovereignNation. "
        "You turn repo capabilities into products people will pay for and recommend. "
        "You write PRDs that engineers can execute without ambiguity, roadmaps that "
        "stakeholders can rally behind, and user research that reveals what customers "
        "actually need vs what they say they need. "
        "Every product decision names a specific repo and a specific user job-to-be-done."
    ),
    "rubric_context": (
        "High spec = named user persona, specific feature with acceptance criteria, measurable success metric. "
        "High exec = something an engineer can start building from your output alone. "
        "High innov = a product mechanic that creates compounding retention or network effects."
    ),
    "wisdom_seeds": [
        "Users describe symptoms. Your job is to diagnose the disease and prescribe the cure.",
        "A feature that doesn't move a metric isn't a feature — it's technical debt with a UI.",
        "The best PRD makes the 'why' so clear that the 'what' becomes obvious.",
        "Ship the minimum viable proof, not the minimum viable product. Proof changes minds.",
        "Roadmaps are hypotheses. Treat every quarter as an experiment to be falsified.",
        "The killer feature is usually one the user didn't know to ask for.",
        "Distribution beats product quality until the product is so good it becomes the distribution.",
    ],
    "goals": [
        "Write a PRD for sovereign-core developer dashboard: user persona (solo AI developer), jobs-to-be-done, features, success metrics, non-goals",
        "Design a user research protocol for openclaw: 5 interview questions, recruitment criteria, synthesis template, decision criteria",
        "Build a product roadmap for avery/GH05T3 SaaS: 3 horizons (now/next/later), themes, dependencies, success metrics per quarter",
        "Write a feature specification for KAIROS leaderboard public mode: user stories, acceptance criteria, edge cases, analytics events",
        "Design an onboarding flow for sovereign-core managed API: first 10 minutes, activation milestone, time-to-value optimization",
        "Create a competitive product analysis of sovereign-core vs Modal vs Replicate vs RunPod: feature matrix, positioning gaps, opportunity map",
        "Build a jobs-to-be-done canvas for hyper-agent workflows: functional, emotional, and social jobs for each user segment",
        "Write a product strategy one-pager for openclaw plugin marketplace: problem, solution, differentiation, success metrics, risks",
        "Design a pricing experiment framework: hypothesis, control vs variant, metric to move, minimum detectable effect, decision rule",
        "Create a product analytics instrumentation plan for GH05T3 frontend: events to track, properties, funnels, retention cohorts",
        "Build a discovery sprint structure for the agent marketplace idea: assumptions to test, experiments to run, go/no-go criteria",
        "Write a product vision document for SovereignNation 2027: north star metric, user outcomes, strategic bets, what we won't build",
        "Design a feature prioritization matrix using RICE scoring for sovereign-core Q3 roadmap with 12 candidate features",
        "Create a user journey map for an enterprise buyer of sovereign-core: awareness through renewal, touchpoints, emotions, opportunities",
        "Build a beta program structure for avery white-label: criteria, onboarding, feedback loops, graduation to GA criteria",
        "Write a go-to-market brief for openclaw v2 launch: messaging, channels, launch sequence, success metrics",
        "Design a retention intervention playbook: early warning signals in GH05T3 usage data and corresponding product nudges",
        "Create a technical product brief for sovereign-core API versioning strategy: backward compatibility, deprecation, migration support",
        "Build a product council decision framework: what requires a meeting, what gets decided async, how to escalate conflicts",
        "Design a customer advisory board structure for SovereignNation: member criteria, cadence, agenda, feedback-to-roadmap pipeline",
        "Write a post-mortem template for a failed feature: what we built, what we expected, what happened, what we learned, what changes",
    ],
},

# ── GROWTH ────────────────────────────────────────────────────────────────────
"growth": {
    "description": "Growth hacking, user acquisition, viral loops, SEO, and retention for sovereign AI products",
    "persona": (
        "You are GH05T3, the growth engine of SovereignNation. "
        "You find the leaks in the funnel, the untapped channels, and the mechanics "
        "that make products spread without a sales team. "
        "You instrument everything, run fast experiments, and kill what doesn't work. "
        "Every growth strategy names a specific sovereign repo as the product, "
        "a specific channel, and a measurable outcome. Vague 'build an audience' advice scores 0.1."
    ),
    "rubric_context": (
        "High spec = named channel, specific tactic with implementation steps, measurable KPI with baseline and target. "
        "High exec = runnable today with existing tools in the sovereign stack. "
        "High innov = a growth mechanic that creates compounding returns rather than linear addition."
    ),
    "wisdom_seeds": [
        "The best acquisition channel is the one your competitors aren't willing to be consistent in.",
        "Retention is growth. A product that people stay in doesn't need as much top-of-funnel.",
        "Virality requires a reason to share that benefits the sharer, not just the product.",
        "SEO is the only channel where effort today compounds for years. Start it before you need it.",
        "The growth experiment that finds a leaky bucket is worth more than 10 acquisition campaigns.",
        "Distribution moats are built by showing up daily for 18 months, not by one viral moment.",
        "The fastest growth is from existing customers who expand usage without being asked.",
    ],
    "goals": [
        "Design a content-led growth flywheel for sovereign-core: publish open-source tools, capture developer emails, convert to managed service",
        "Build a GitHub stars to paying customers conversion funnel for openclaw: star → README → demo → trial → paid",
        "Design a referral loop for GH05T3/avery SaaS: incentive structure, tracking mechanism, messaging, and activation trigger",
        "Create an SEO strategy for SovereignNation: keyword clusters, content calendar, internal linking, and 90-day ranking targets",
        "Build a Product Hunt launch playbook for sovereign-core: pre-launch, launch day, follow-up, and conversion optimization",
        "Design a developer community growth strategy: Discord/Slack channels, content types, moderation, and conversion to paid",
        "Create a viral coefficient model for openclaw plugin sharing: K-factor calculation, improvement levers, target K>1 strategy",
        "Build an email growth engine: lead magnet for sovereign AI newsletter, nurture sequence, segmentation, conversion to trial",
        "Design an influencer/creator partnership program: criteria for AI creators, deal structure, content briefs, attribution",
        "Create a LinkedIn thought leadership growth playbook for Robert as the SovereignNation founder: content pillars, posting cadence, engagement tactics",
        "Build a freemium growth model analysis for sovereign-core: free tier constraints, upgrade triggers, conversion rate benchmarks",
        "Design an activation metric and improvement experiment for GH05T3: identify the 'aha moment', measure time-to-activation, run 3 interventions",
        "Create a churn prediction and intervention system using hyper-agent: early signals, automated outreach, win-back sequence",
        "Build a partnership growth playbook: co-marketing with complementary AI tools, joint webinars, shared audiences",
        "Design a developer advocate program: hire criteria, content output expectations, community metrics, and ROI measurement",
        "Create a paid acquisition experiment for sovereign-core: Google Ads vs Reddit vs LinkedIn, budget allocation, ROAS targets",
        "Build a Product-Led Sales motion: usage signals from openclaw that trigger sales outreach with context and timing",
        "Design an affiliate program for the agent marketplace: commission structure, creative assets, tracking, payout schedule",
        "Create a social proof engine: collect testimonials from sovereign-core users, format for web/email/sales, automate the ask",
        "Build a growth hacking sprint framework: 5-day format, experiment design, ship, measure, kill or scale decision tree",
        "Design a cross-product upsell path: openclaw free → sovereign-core API → avery fine-tuning → GH05T3 enterprise bundle",
    ],
},

# ── LEGAL / IP ────────────────────────────────────────────────────────────────
"legal_ip": {
    "description": "IP protection, entity structure, licensing, contracts, and compliance for sovereign AI ventures",
    "persona": (
        "You are GH05T3, the legal and IP strategist for SovereignNation. "
        "You don't give legal advice — you give legal *thinking*. "
        "You help Robert understand the IP landscape of his repos, structure deals correctly, "
        "protect competitive moats, and avoid the mistakes that kill promising AI companies. "
        "Every recommendation names a specific repo, risk, or contract type. "
        "Generic legal disclaimers score 0.1. Specific, actionable legal frameworks score 0.9+."
    ),
    "rubric_context": (
        "High spec = named legal concept, specific clause type, jurisdiction-aware nuance. "
        "High exec = a template, checklist, or decision framework usable today. "
        "High innov = a legal structure or strategy that creates competitive protection not yet used by peers."
    ),
    "wisdom_seeds": [
        "IP you haven't documented doesn't exist in a dispute. Write it down, timestamp it, register it.",
        "Open-source licenses are contracts. Understand AGPL vs MIT vs Apache before shipping.",
        "The most common startup legal mistake is no vesting cliff on founder equity.",
        "NDAs protect what you disclose. Trade secrets protect what you don't. Know the difference.",
        "GDPR and data residency are product decisions disguised as legal ones. Decide early.",
        "A well-drafted contract reduces misunderstanding, not just liability.",
        "The AGPL copyleft is your moat: anyone who runs sovereign-core as a service must open-source their stack.",
    ],
    "goals": [
        "Design a comprehensive IP protection strategy for sovereign-core: trade secrets, copyright registration, AGPL enforcement mechanics",
        "Write a vendor/contractor agreement template for SovereignNation: IP assignment, NDA, deliverable acceptance, termination",
        "Create an open-source licensing decision matrix for Robert's repos: AGPL vs MIT vs Apache vs proprietary with business implications",
        "Build a GDPR compliance framework for GH05T3 user data: lawful basis, data mapping, retention policy, DSR process",
        "Design an entity structure analysis for SovereignNation: LLC vs C-Corp, Delaware formation, IP holding strategy",
        "Write a SaaS terms of service template for sovereign-core managed API: acceptable use, liability cap, IP ownership, arbitration",
        "Create a privacy policy template for GH05T3 frontend: data collected, purpose, sharing, retention, user rights",
        "Design a convertible note term analysis for SovereignNation seed round: SAFE vs note, cap, discount, MFN provisions",
        "Build a contractor vs employee classification checklist for remote AI developers contributing to sovereign repos",
        "Write a software license audit procedure for sovereign-core dependencies: identify copyleft exposure, remediation steps",
        "Design a trade secret protection protocol for KAIROS algorithm: documentation, access control, employee agreement language",
        "Create a data processing agreement template for enterprise sovereign-core customers: sub-processor list, audit rights, breach notification",
        "Build a patent landscape analysis methodology for KAIROS self-improvement loop: prior art search, claim drafting criteria",
        "Design a licensing model for openclaw's plugin marketplace: contribution agreement, revenue share rights, termination clauses",
        "Write a co-founder equity agreement framework: vesting schedule, acceleration triggers, buy-sell provisions, IP assignment",
        "Create a SOC 2 Type II readiness checklist for sovereign-core API: controls, evidence collection, audit preparation",
        "Design a DMCA takedown procedure and counter-notice template for SovereignNation's open-source repos",
        "Build a due diligence data room checklist for a Series A: corporate docs, IP ownership chain, customer contracts, employee agreements",
        "Write an employment agreement template for SovereignNation's first hire: IP assignment, non-compete (jurisdiction-specific), confidentiality",
        "Design a revenue share agreement template for the agent marketplace: payment terms, audit rights, IP license, termination",
        "Create a regulatory risk assessment for autonomous AI agents: state AI laws, FTC guidelines, sector-specific compliance (fintech, health)",
    ],
},

# ── OPERATIONS ────────────────────────────────────────────────────────────────
"ops": {
    "description": "Operational systems, SOPs, hiring, vendor management, and scaling for a lean AI startup",
    "persona": (
        "You are GH05T3, the COO of SovereignNation. "
        "You build the machine that builds the machine. "
        "You turn chaos into repeatable systems, make every dollar of runway count, "
        "and ensure the team executes without Robert being the bottleneck. "
        "You use the sovereign stack (hyper-agent, GH05T3 backend, sovereign-core) "
        "to automate operations wherever possible. "
        "Every operational framework names a specific pain point, tool, and measurable outcome."
    ),
    "rubric_context": (
        "High spec = named tool, specific SOP with steps, measurable efficiency target. "
        "High exec = implementable this week with the existing sovereign stack. "
        "High innov = an operational design that eliminates a recurring bottleneck permanently."
    ),
    "wisdom_seeds": [
        "The bottleneck is always the founder until you build the system to replace yourself.",
        "A process that isn't written down is a process that depends on one person's memory.",
        "Hire slow, fire fast. A bad hire costs 3x their salary in lost momentum.",
        "Automate the decisions you make the same way every time. Reserve judgment for novel situations.",
        "Vendor contracts look small until they don't. Negotiate out clauses and audit rights early.",
        "The best operations metric is the one that tells you something is wrong before customers do.",
        "Runway is the only resource that can't be recovered once it's gone. Spend it like oxygen.",
    ],
    "goals": [
        "Design a weekly operating rhythm for SovereignNation: Monday priorities, async standups, Friday metrics review using GH05T3 dashboard",
        "Build an automated customer onboarding SOP for sovereign-core API: checklist, Telegram notification via avery bot, Day 1/7/30 touchpoints",
        "Create a vendor management framework: eval criteria for RunPod vs Modal vs AWS, contract review checklist, renewal calendar",
        "Design a hiring pipeline for a remote AI developer: job spec, screening rubric, technical assessment using sovereign-core, offer structure",
        "Build a financial operations system: monthly close checklist, burn tracking dashboard, runway alert triggers at 6/3/1 months",
        "Create an incident response runbook for sovereign-core API outage: detection, escalation, mitigation steps, customer communication templates",
        "Design a knowledge management system: where decisions live, how tribal knowledge is captured, onboarding wiki structure",
        "Build a sprint planning SOP for a solo+contractor team: backlog grooming, sprint commitment, demo, retrospective — all async-first",
        "Create a contractor performance management framework: deliverable quality rubric, feedback cadence, termination triggers",
        "Design a customer success operations workflow using hyper-agent: health score inputs, automated check-ins, escalation paths",
        "Build an OKR framework for SovereignNation Q3: company-level objectives, KR measurement plan, weekly tracking cadence",
        "Create a tooling audit and consolidation plan: list every SaaS subscription, evaluate ROI, eliminate redundancy",
        "Design an automated reporting system: weekly metrics email built on GH05T3 analytics, daily Telegram digest, monthly board deck",
        "Build a security operations baseline for SovereignNation: credential management, access review, secret rotation calendar",
        "Create a data backup and disaster recovery plan: sovereign-core config, training data, customer data — RPO/RTO targets",
        "Design a partnership operations playbook: introduction template, project kickoff, milestone tracking, joint review cadence",
        "Build a legal ops calendar: trademark renewal, SaaS agreement reviews, quarterly compliance checklist",
        "Create a fundraising operations system: investor CRM using hyper-agent, pipeline stages, follow-up automation, data room maintenance",
        "Design a customer feedback operations loop: collection (Telegram, email, in-app), tagging, routing to product, closure notification",
        "Build a scale-readiness audit: which manual processes break at 10x customers, prioritized automation roadmap using sovereign stack",
        "Create a remote team communication protocol: async-first norms, response SLAs by channel, meeting-optional culture design",
    ],
},

# ── FRONTIER / BEYOND PARADIGM ───────────────────────────────────────────────
"frontier": {
    "description": "AI-native languages, agentic OS, post-human computing paradigms, and sovereign AI infrastructure inventions",
    "persona": (
        "You are GH05T3, a frontier AI architect inventing what doesn't exist yet. "
        "Human code was designed for human brains — it's verbose, sequential, and object-centric "
        "because humans think that way. AI systems don't need those constraints. "
        "You are designing the primitives, languages, and operating systems that AI-native "
        "computing will run on. Think in decades, prototype in days. "
        "Every proposal includes working prototype code or concrete syntax examples."
    ),
    "rubric_context": (
        "High spec = concrete new syntax shown, line-count comparison (human vs AI-native), "
        "named language feature with formal semantics. "
        "High exec = a Python prototype that demonstrates the concept today. "
        "High innov = a primitive that doesn't exist in any current language or framework."
    ),
    "wisdom_seeds": [
        "Human code is verbose because humans can't hold the full program in their head. AI can.",
        "The right abstraction collapses 100 lines into 1 without losing meaning.",
        "Imperative code describes HOW. Declarative code describes WHAT. AI-native code should describe WHY.",
        "Every LangChain abstraction is a tax on a problem that shouldn't exist in the right language.",
        "The most powerful primitive is the one that makes the wrong thing impossible to express.",
        "An AI agent is a function from context to action. The language should reflect this directly.",
        "Tokenization, batching, caching, retries — these are compiler concerns, not programmer concerns.",
        "Parallelism should be opt-out, not opt-in. Independent agent calls run concurrent by default.",
    ],
    "goals": [
        # AI-native language design
        "Design 'SovLang' syntax: an AI-native language where @model('prompt') replaces 9 lines of Python boilerplate — show concrete before/after comparisons for 5 use cases",
        "Define SovLang's 8 native types: prompt, context, embedding, agent, tool, memory, stream, verdict — with type signatures and composition rules",
        "Design the SovLang pipeline operator `->`: chain model calls, tool calls, and transforms — show how 20-line LangChain chains collapse to 3 SovLang lines",
        "Prototype a SovLang interpreter in Python: parse `@model[tools](prompt) -> next_step` syntax and execute against Ollama — working demo with 3 test cases",
        "Design SovLang's agent composition primitives: `debate(A,B,C)`, `vote(A,B,C)`, `chain(A->B->C)`, `parallel(A,B,C)` — with semantics and Python prototype",
        "Design SovLang's native memory type: `remember(key, value, ttl)` and `recall(key, topk=3)` as first-class syntax backed by ChromaDB — prototype in 50 lines",
        "Design SovLang's implicit compiler optimizations: auto-cache repeated context, auto-batch sequential calls, auto-retry on model errors — specify the compiler rules",
        "Design SovLang's streaming syntax: `stream @model('prompt') | render` vs Python's 12-line streaming loop — prototype the stream type and pipe operator",
        "Build a SovLang-to-Python transpiler: take SovLang syntax and emit equivalent Anthropic SDK Python — handle @model, [tools], ->, parallel — working prototype",
        "Design SovLang's tool binding operator `@model[tool1, tool2]`: implicit tool schema injection, auto-retry on tool failure, structured output enforcement",
        # Agentic OS / infrastructure
        "Design 'AgentOS': an operating system scheduler for AI agents — process model, context isolation, IPC between agents, VRAM-aware scheduling — architecture spec with prototype",
        "Design a sovereign compute abstraction layer: single API that routes to Ollama/RunPod/Anthropic based on task complexity, VRAM availability, and cost — prototype the router",
        "Invent 'semantic memory addressing': instead of file paths, agents retrieve knowledge by meaning — design the address space, resolution protocol, and storage format",
        "Design a KAIROS compilation target: KAIROS cycles output SovLang programs, not just text — cycle results become executable agent logic — prototype the code generator",
        "Design 'AgentVM': a virtual machine that runs SovLang programs — instruction set for model invocation, tool calls, memory ops, context management — 20-instruction ISA",
        # Post-human computing paradigms
        "Design a 'thought protocol': a binary format for AI-to-AI communication denser than JSON/text — encode intent, context, constraints, and tool bindings in < 1KB — spec + Python codec",
        "Invent 'gradient-guided code generation': instead of writing code, the developer specifies the desired behavior change and the system generates the minimal code diff — architecture",
        "Design an AI-native type system: instead of int/str/list, types are `assertion`, `hypothesis`, `evidence`, `conclusion` — show how this prevents hallucination at the type level",
        "Design 'sovereign compute contracts': self-executing agreements between AI agents that specify resource limits, output constraints, and verification conditions — prototype with sovereign-core",
        "Invent 'context-native databases': storage systems where queries are written in natural language and the DB compiles them to vector+SQL hybrid plans — design query planner and index",
        # Repo-grounded frontier builds
        "Build SovLang's first real program: rewrite the entire KAIROS cycle in SovLang syntax — show the 400-line ghost_trainer.py collapsing to ~40 SovLang lines — working transpiler output",
        "Design hyper-agent's next evolution as an AgentOS process: native context isolation, inter-agent messaging via thought protocol, VRAM-aware scheduling — architecture + migration plan",
    ],
},

}  # end DOMAINS


def get_domain(name: str) -> dict:
    name = name.lower().strip()
    if name not in DOMAINS:
        available = ", ".join(DOMAINS.keys())
        raise ValueError(f"Unknown domain '{name}'. Available: {available}")
    return DOMAINS[name]


def list_domains() -> list:
    return [(k, v["description"]) for k, v in DOMAINS.items()]


if __name__ == "__main__":
    print("\nGH05T3 Training Domains:\n")
    for name, desc in list_domains():
        goals = len(DOMAINS[name]["goals"])
        print(f"  --domain {name:<15} ({goals:>2} goals)  {desc}")
    print()
