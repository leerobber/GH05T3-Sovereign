import json, os, re, time, sqlite3, argparse, subprocess, sys, tempfile
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import requests as _req
try:
    from groq import Groq as _Groq
except ImportError:
    _Groq = None   # groq optional â€” all production models use Ollama
# AUTO-DISABLED by GH05T3 aggressive engine: from ghost_domains import get_domain, DOMAINS
pass  # safe placeholder
try:
    from repo_scanner import load_capability_summary
except ImportError:
    def load_capability_summary(*_): return ""
try:
    import slack_notify as _slack
except ImportError:
    _slack = None
try:
    import economy_bridge as _eco
except ImportError:
    _eco = None

GROQ_KEY = os.environ.get("GROQ_API_KEY", "")
PROPOSER = "gemma3:12b"           # Ollama â€” strongest local model, no quota
VERIFIER = "qwen2.5:7b-instruct"  # Ollama â€” instruction-tuned rubric judge
CRITIC   = "dolphin-llama3:8b"    # Ollama â€” uncensored attacker, brutally honest
META     = "llama3.2:3b"          # Ollama â€” tiny + fast for meta rewrites

# Groq client â€” optional (all current models use Ollama, groq kept for future)
_groq = _Groq(api_key=GROQ_KEY) if (_Groq and GROQ_KEY) else None

# 4-axis scoring weights â€” must sum to 1.0
W = {"spec": 0.30, "exec": 0.35, "innov": 0.25, "rev": 0.10}
PASS_THRESH  = 0.55   # weighted score threshold
SANDBOX_TO   = 6      # seconds for code execution

GOALS = [
    "Implement FAISS archive search reducing O(N) to O(log N)",
    "Build KAIROS score trajectory chart with plateau detection",
    "Complete coder sub-agent generate->pytest->self-revise->verify loop",
    "Wire Darwin stepping-stones into KAIROS archive via DGM bridge",
    "Implement LLM-as-Judge verifier with multi-axis rubric scoring",
    "Build RLVR reward signal from Verifier PASS/FAIL with gradient tracking",
    "Implement Critic vs Proposer adversarial debate engine 2 rounds",
    "Build diversity constraint scoring to prevent population collapse",
    "Complete constitutional AI critique with principle violation detection",
    "Build SPIN dataset collector from chosen/rejected proposal pairs",
    "Wire WebSocket telemetry for live SovereignPanel dashboard",
    "Complete meta-agent rule rewrite triggered by score plateau",
    "Cross-domain transfer VRAM optimization patterns to latency reduction",
    "Build cold-tier archive pruning strategy for 50K+ stepping-stones",
    "Implement PCL synesthetic state with frequency-color persistence",
    "Add exponential backoff health check before every sovereign LLM call",
    "Complete GhostScript async/await language extension with cancellation",
    "Implement Self-Play Fine-Tuning pipeline with LoRA data prep",
    "Wire Tailscale mesh tunnel for no-static-IP resilience",
    "Build Van Eck RFFingerprint 15MHz probe calibration pipeline",
    "Advance GH05T3 self-agenda: generate month-4 stretch goals autonomously",
]


OLLAMA_URL = "http://localhost:11434/api/chat"


# â”€â”€ routing: ":" in name = Ollama REST, else = Groq â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def _call(model, msgs, temp=0.72, max_tok=512, _retries=4):
    if ":" in model:
        delay = 15
        for attempt in range(_retries):
            try:
                r = _req.post(OLLAMA_URL, json={
                    "model": model,
                    "messages": msgs,
                    "stream": False,
                    "options": {"temperature": temp, "num_predict": max_tok},
                }, timeout=600)
                r.raise_for_status()
                return r.json()["message"]["content"]
            except Exception as e:
                if attempt < _retries - 1:
                    jitter = delay * (0.8 + 0.4 * (time.time() % 1))
                    print(f"  [Ollama ERR -> retry {attempt+1} in {jitter:.0f}s] {str(e)[:60]}")
                    time.sleep(jitter)
                    delay *= 2
                else:
                    return f"[ERR:{e}]"
    delay = 8
    for attempt in range(_retries):
        try:
            r = _groq.chat.completions.create(
                model=model, messages=msgs, temperature=temp, max_tokens=max_tok)
            return r.choices[0].message.content
        except Exception as e:
            if "429" in str(e) and attempt < _retries - 1:
                jitter = delay * (0.8 + 0.4 * (time.time() % 1))
                print(f"  [429 -> retry {attempt+1} in {jitter:.0f}s]")
                time.sleep(jitter)
                delay *= 2
            else:
                return f"[ERR:{e}]"


def _strip_think(t):
    t = str(t)
    # Remove complete <think>...</think> block
    cleaned = re.sub(r'<think>.*?</think>', '', t, flags=re.DOTALL).strip()
    # If block was truncated (no closing tag), find real content after it
    if cleaned.startswith('<think>') or (not cleaned and '<think>' in t):
        end = t.find('</think>')
        if end != -1:
            cleaned = t[end + 8:].strip()
        else:
            # Scan for first substantive line after the think block started
            for marker in ['PROPOSAL:', '**', '##', 'Step 1', '\n\n']:
                idx = t.find(marker, 30)
                if idx != -1:
                    cleaned = t[idx:].strip()
                    break
            else:
                cleaned = t  # give up, return raw
    return cleaned

def _trunc(t, n=200):
    t = _strip_think(t)
    return t[:n] + "..." if len(t) > n else t


# â”€â”€ sandbox execution â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def _extract_code(text):
    blocks = re.findall(r'```python\n(.*?)```', text, re.DOTALL)
    return blocks[0].strip() if blocks else None

def _run_sandbox(code):
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False, dir='data') as f:
        f.write(code); fname = f.name
    try:
        r = subprocess.run([sys.executable, fname],
                           capture_output=True, text=True, timeout=SANDBOX_TO)
        return r.stdout[:300].strip(), r.stderr[:200].strip(), r.returncode
    except subprocess.TimeoutExpired:
        return "", f"TIMEOUT>{SANDBOX_TO}s", 1
    except Exception as e:
        return "", str(e)[:150], 1
    finally:
        Path(fname).unlink(missing_ok=True)


# â”€â”€ 3-tier wisdom store â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Tier 1: insights (what works)   data/wisdom_insights_{domain}.txt
# Tier 2: warnings (what fails)   data/wisdom_warnings_{domain}.txt
# Tier 3: master synthesis         data/wisdom_master_{domain}.txt  (compacted every 50 cycles)

def _wpath(tier, domain="core"):
    return Path(f"data/wisdom_{tier}_{domain}.txt")

def _load_wisdom(n=5, domain="core"):
    parts = []
    for tier in ("master", "insights", "warnings"):
        p = _wpath(tier, domain)
        if p.exists():
            lines = p.read_text(encoding="utf-8").strip().splitlines()
            parts.extend(lines[-n:])
    return " | ".join(parts[:n*2]) if parts else ""

def _save_wisdom(insight, tier="insights", domain="core"):
    p = _wpath(tier, domain)
    lines = p.read_text(encoding="utf-8").strip().splitlines() if p.exists() else []
    lines.append(insight.strip()[:110])
    p.write_text("\n".join(lines[-30:]), encoding="utf-8")

def _seed_wisdom(domain_data, domain):
    """Write domain wisdom seeds on first run if file doesn't exist."""
    p = _wpath("insights", domain)
    if not p.exists() and domain_data.get("wisdom_seeds"):
        p.write_text("\n".join(domain_data["wisdom_seeds"]), encoding="utf-8")


# â”€â”€ goal priority queue with evolution â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

class GoalQueue:
    def __init__(self, domain_goals=None, extra=None):
        base             = list(domain_goals or GOALS)
        self.goals       = base + (extra or [])
        self._base       = base
        self.scores      = {g: [] for g in self.goals}
        self.mastery     = {g: 0  for g in self.goals}
        self._last_pick  = None
        self._pick_streak = 0

    def update(self, goal, score):
        self.scores.setdefault(goal, []).append(score)
        if score >= 0.85:
            self.mastery[goal] = self.mastery.get(goal, 0) + 1
        else:
            self.mastery[goal] = 0

    def pick(self):
        untried = [g for g in self.goals if not self.scores.get(g)]
        if untried:
            chosen = untried[0]
        else:
            chosen = min(self.goals,
                         key=lambda g: sum(self.scores[g]) / max(1, len(self.scores[g])))
        self._pick_streak = (self._pick_streak + 1) if chosen == self._last_pick else 1
        self._last_pick = chosen
        return chosen

    def force_rotate(self):
        """Pick the second-worst goal (used by FalsificationMonitor)."""
        scored = [g for g in self.goals if self.scores.get(g)]
        if len(scored) < 2:
            return self.pick()
        ranked = sorted(scored, key=lambda g: sum(self.scores[g]) / max(1, len(self.scores[g])))
        return ranked[1] if len(ranked) > 1 else ranked[0]

    def evolve(self, old, new):
        if old in self.goals:
            self.goals[self.goals.index(old)] = new
        self.scores[new]  = []
        self.mastery[new] = 0
        self.scores.pop(old, None)
        self.mastery.pop(old, None)

    def swap_hardest_for_easier(self, n=5):
        """Replace n hardest goals with the base goals that haven't been tried."""
        scored = [g for g in self.goals if self.scores.get(g)]
        if not scored:
            return
        hardest = sorted(scored,
                         key=lambda g: sum(self.scores[g]) / max(1, len(self.scores[g])))[:n]
        untried_base = [g for g in self._base if g not in self.goals]
        for old, new in zip(hardest, untried_base or hardest[::-1]):
            self.evolve(old, new)

    def dump(self):
        return {g: round(sum(v)/len(v), 3) for g, v in self.scores.items() if v}

    def evolved(self):
        return [g for g in self.goals if g not in self._base]


# â”€â”€ falsification monitor â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

class FalsificationMonitor:
    """
    6 auto-intervention conditions. Checks after every cycle.
    Logs all interventions to data/falsification_log.jsonl.
    """
    WISDOM_HALT_WINDOW = 20
    SCORE_WINDOW       = 8
    STAGNATION_WINDOW  = 16

    def __init__(self):
        self.log_path         = Path("data/falsification_log.jsonl")
        self.score_window_means: list[float] = []
        self.reward_history:     list[float] = []
        self.wisdom_counts:      list[int]   = []  # wisdom line count per cycle
        self.low_window_streak   = 0

    def _count_wisdom(self, domain):
        total = 0
        for tier in ("insights", "warnings", "master"):
            p = _wpath(tier, domain)
            if p.exists():
                total += len(p.read_text(encoding="utf-8").strip().splitlines())
        return total

    def _log(self, cycle, condition, action, detail=""):
        entry = {"cycle": cycle, "condition": condition,
                 "action": action, "detail": detail, "ts": time.time()}
        with open(self.log_path, "a") as f:
            f.write(json.dumps(entry) + "\n")
        print(f"  [FALSIFY] {condition}: {action}")

    def check(self, trainer, cycle, goal, score, reward, domain="core"):
        scores  = trainer.scores
        queue   = trainer.queue
        total   = trainer.passes + trainer.fails
        actions = []

        # C1: Score collapse â€” rolling-8 mean < 0.40 for 3 consecutive windows
        if len(scores) >= self.SCORE_WINDOW:
            win_mean = sum(scores[-self.SCORE_WINDOW:]) / self.SCORE_WINDOW
            self.score_window_means.append(win_mean)
            if win_mean < 0.40:
                self.low_window_streak += 1
            else:
                self.low_window_streak = 0
            if self.low_window_streak >= 3:
                self._log(cycle, "score_collapse", "meta_rewrite",
                          f"window_mean={win_mean:.3f}")
                trainer.meta_rules = ""  # force fresh meta rewrite next cycle
                self.low_window_streak = 0
                actions.append("score_collapse")

        # C2: Baseline divergence â€” baseline < -0.5
        if trainer.baseline < -0.5:
            self._log(cycle, "baseline_dive", "reset_baseline",
                      f"baseline={trainer.baseline:.3f}")
            trainer.baseline = 0.5
            trainer.history  = []
            actions.append("baseline_dive")

        # C3: Pass rate floor â€” < 20% over last 50 cycles
        if total >= 50:
            recent_passes = sum(1 for s in scores[-50:] if s >= PASS_THRESH)
            recent_rate   = recent_passes / 50
            if recent_rate < 0.20:
                self._log(cycle, "pass_floor", "swap_goals",
                          f"rate={recent_rate:.0%}")
                queue.swap_hardest_for_easier(5)
                actions.append("pass_floor")

        # C4: Reward stagnation â€” std of last 16 rewards < 0.02
        self.reward_history.append(reward)
        if len(self.reward_history) >= self.STAGNATION_WINDOW:
            self.reward_history = self.reward_history[-self.STAGNATION_WINDOW:]
            std = float(np.std(self.reward_history))
            if std < 0.02:
                adversarial = queue.force_rotate()
                self._log(cycle, "reward_stagnation", "inject_adversarial",
                          f"std={std:.4f} -> goal={adversarial[:40]}")
                trainer._forced_goal = adversarial
                actions.append("reward_stagnation")

        # C5: Wisdom growth halt â€” 0 new wisdom lines in last 20 cycles
        wc = self._count_wisdom(domain)
        self.wisdom_counts.append(wc)
        if len(self.wisdom_counts) >= self.WISDOM_HALT_WINDOW:
            self.wisdom_counts = self.wisdom_counts[-self.WISDOM_HALT_WINDOW:]
            if self.wisdom_counts[-1] == self.wisdom_counts[0]:
                # Lower threshold so more proposals trigger wisdom extraction
                old = trainer._wisdom_thresh
                trainer._wisdom_thresh = max(0.45, trainer._wisdom_thresh - 0.05)
                self._log(cycle, "wisdom_halt", "lower_wisdom_threshold",
                          f"{old:.2f}->{trainer._wisdom_thresh:.2f}")
                actions.append("wisdom_halt")

        # C6: Goal mastery timeout â€” same goal picked 10+ consecutive cycles
        if queue._pick_streak >= 10:
            self._log(cycle, "goal_timeout", "force_evolve",
                      f"goal={goal[:40]} streak={queue._pick_streak}")
            new_goal_prompt = (
                f"This goal has been stuck for {queue._pick_streak} cycles: '{goal}'\n"
                "Generate ONE harder successor goal (max 12 words). ONLY the goal text:")
            new_g = _call(META, [{"role": "user", "content": new_goal_prompt}],
                          temp=0.6, max_tok=25)
            if new_g and not new_g.startswith("[ERR"):
                queue.evolve(goal, new_g.strip().strip('"')[:120])
            actions.append("goal_timeout")

        return actions


# â”€â”€ archive â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

class Archive:
    def __init__(self, conn):
        self.conn = conn
        conn.execute("CREATE TABLE IF NOT EXISTS archive("
                     "id INTEGER PRIMARY KEY, goal TEXT, proposal TEXT, score REAL, ts REAL)")
        conn.commit()

    def store(self, goal, proposal, score):
        self.conn.execute("INSERT INTO archive(goal,proposal,score,ts) VALUES(?,?,?,?)",
                         (goal, _strip_think(proposal)[:600], score, time.time()))
        self.conn.commit()

    def top_k(self, k=3):
        return self.conn.execute(
            "SELECT goal,proposal,score FROM archive ORDER BY score DESC LIMIT ?", (k,)).fetchall()

    def prune(self, keep=500):
        self.conn.execute("DELETE FROM archive WHERE id NOT IN "
                         "(SELECT id FROM archive ORDER BY score DESC LIMIT ?)", (keep,))
        self.conn.commit()


# â”€â”€ sparkline â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

_BARS = " ._-~+*#@"

def sparkline(scores, w=32):
    s = scores[-w:]
    if not s: return ""
    mx = max(s) or 1.0
    return "".join(_BARS[min(8, int(v / mx * 8))] for v in s)


# â”€â”€ dataclass â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

@dataclass
class R:
    cycle: int; goal: str; proposal: str; score: float; reward: float


# â”€â”€ trainer â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

class Trainer:
    def __init__(self, domain="core"):
        Path("data").mkdir(exist_ok=True)
        self.domain          = domain
        self.domain_data     = get_domain(domain)
        self.cycle           = 0
        self.passes          = 0
        self.fails           = 0
        self.baseline        = 0.5
        self.history         = []
        self.scores          = []
        self.spin            = []
        self.last_fail       = None
        self.last_fail_score = 0.0
        self.meta_rules      = ""
        self._wisdom_thresh  = 0.70   # FalsificationMonitor can lower this
        self._forced_goal    = None   # FalsificationMonitor can inject a goal

        self.conn    = sqlite3.connect("data/pcl.db")
        self.conn.execute("CREATE TABLE IF NOT EXISTS pcl("
                         "id INTEGER PRIMARY KEY,ts REAL,hz REAL,"
                         "r INT,g INT,b INT,label TEXT,score REAL)")
        self.conn.commit()
        self.archive = Archive(self.conn)
        self.monitor = FalsificationMonitor()

        # Seed domain wisdom on first run
        _seed_wisdom(self.domain_data, domain)

        ckpt = Path(f"data/ckpt_{domain}.json")
        if not ckpt.exists():
            ckpt = Path("data/ckpt.json")  # fall back to legacy file
        if ckpt.exists():
            d = json.loads(ckpt.read_text())
            self.cycle      = d.get("cycle", 0)
            self.baseline   = d.get("baseline", 0.5)
            self.passes     = d.get("passes", 0)
            self.fails      = d.get("fails", 0)
            self.meta_rules = d.get("meta_rules", "")
            self.queue = GoalQueue(self.domain_data["goals"], d.get("evolved_goals", []))
            for g, v in d.get("goal_scores", {}).items():
                if g in self.queue.scores:
                    self.queue.scores[g] = [float(v)] if isinstance(v, (int, float)) else list(v)
            for g, m in d.get("mastery", {}).items():
                if g in self.queue.mastery:
                    self.queue.mastery[g] = m
            pr = self.passes / max(1, self.passes + self.fails)
            print(f"  RESUME [{domain}]: cycle={self.cycle}  pass={pr:.0%}  goals={len(self.queue.goals)}")
        else:
            self.queue = GoalQueue(self.domain_data["goals"])

    def save(self):
        Path(f"data/ckpt_{self.domain}.json").write_text(json.dumps({
            "cycle":         self.cycle,
            "baseline":      self.baseline,
            "passes":        self.passes,
            "fails":         self.fails,
            "meta_rules":    self.meta_rules,
            "wisdom_thresh": self._wisdom_thresh,
            "goal_scores":   self.queue.dump(),
            "mastery":       {g: v for g, v in self.queue.mastery.items() if v > 0},
            "evolved_goals": self.queue.evolved(),
        }, indent=2))

    def _flush_spin(self):
        if not self.spin: return
        count = len(self.spin)
        with open("data/spin_dataset.jsonl", "a") as f:
            for pair in self.spin:
                f.write(json.dumps(pair) + "\n")
        self.spin.clear()
        if _eco and count > 0:
            _eco.on_spin_upload(count)

    # â”€â”€ prompts â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    def _propose_prompt(self, goal):
        d = self.domain_data
        parts = [d.get("persona", "You are GH05T3, self-improving AI."),
                 f"KAIROS cycle {self.cycle}. Domain: {self.domain}."]
        id_path = Path("core/identity.txt")
        if id_path.exists():
            parts.append(f"IDENTITY: {id_path.read_text(encoding='utf-8')[:160]}")
        if self.meta_rules:
            parts.append(f"META-RULES: {self.meta_rules[:120]}")
        if d.get("rubric_context"):
            parts.append(f"SCORING HINT: {d['rubric_context'][:160]}")
        # Inject live repo capabilities for business domain (and any domain that benefits)
        repo_cap = load_capability_summary()
        if repo_cap and self.domain in ("business", "cfo", "content", "ml_engineer"):
            parts.append(f"AVAILABLE REPOS (use these as product foundations):\n{repo_cap[:600]}")
        wisdom = _load_wisdom(5, self.domain)
        if wisdom:
            parts.append(f"WISDOM: {wisdom[:300]}")
        top = self.archive.top_k(3)
        if top:
            parts.append("BEST ARCHIVE: " + " | ".join(
                f"[{s:.2f}]{_trunc(p, 90)}" for _, p, s in top))
        if len(self.scores) >= 3:
            parts.append(f"RECENT SCORES: {[round(s,2) for s in self.scores[-3:]]}")
        # Force high-spec output for business/strategy domains
        _BUSINESS_DOMAINS = {"business", "sales", "product_strategy", "growth",
                              "cfo", "ops", "legal_ip", "content"}
        _REPOS = ["sovereign-core", "hyper-agent", "openclaw", "verelene_v5",
                  "MYTHOS", "Jarvis", "avery", "GH05T3"]
        if self.domain in _BUSINESS_DOMAINS:
            repo_of_cycle = _REPOS[self.cycle % len(_REPOS)]
            if self.domain == "sales":
                parts.append(
                    f"\nSPEC CHECKLIST â€” include ALL 5 or spec score = 0.4:\n"
                    f"[1] NAMED ICP: Specific role + company stage + pain "
                    "(e.g. 'ML team lead at 15-person Series A fintech struggling with LLM cost')\n"
                    "[2] EXACT COPY: At least one real subject line, opening line, or script excerpt "
                    "(e.g. Subject: 'Cut your GPT-4 bill by 60% â€” how [Company] did it')\n"
                    "[3] OBJECTION + RESPONSE: One scripted objection exchange "
                    "(e.g. 'We already use OpenAI' â†’ 'That's exactly who our best customers came from...')\n"
                    "[4] CHANNEL + SEQUENCE: Specific platform and step count "
                    "(e.g. 'LinkedIn: connection â†’ voice note â†’ case study DM â†’ Calendly')\n"
                    "[5] SUCCESS METRIC: Response rate, close rate, or pipeline target "
                    "(e.g. '10% reply rate â†’ 3 demos â†’ 1 close at $299/mo = $299 MRR in 30 days')\n"
                    "Missing actual copy or objection handling â†’ spec â‰¤ 0.4. All 5 present â†’ spec 0.8+."
                )
            else:
                parts.append(
                    f"\nSPEC CHECKLIST â€” include ALL 5 or spec score = 0.4:\n"
                    f"[1] REPO FOUNDATION: Build on {repo_of_cycle} (or name a better-fit repo with reason)\n"
                    "[2] PRICING: Exact dollar amounts + tier structure "
                    "(e.g. '$99/mo / 10K calls â€” $299/mo / 100K calls â€” $999/mo enterprise')\n"
                    "[3] CUSTOMER PERSONA: Named ICP with company stage/size "
                    "(e.g. 'ML team lead at 20-person Series A fintech')\n"
                    "[4] IMPLEMENTATION STEPS: 3+ numbered steps executable THIS WEEK\n"
                    "[5] SUCCESS METRIC: Measurable target "
                    "(e.g. '3 paying customers in 45 days at $299/mo = $897 MRR')\n"
                    "Missing any of these â†’ spec â‰¤ 0.4. All 5 present â†’ spec 0.8+."
                )

        parts.append(
            f"\nGOAL: {goal}\n"
            "Write a CONCRETE, EXECUTABLE proposal. Include Python code if applicable. "
            "Vague proposals score 0.3. Specific + runnable proposals score 0.9+.\nPROPOSAL:")
        return "\n".join(parts)

    def _attack_prompt(self, proposal, goal):
        return (f"You are GH05T3's adversarial critic. Be technically brutal.\n"
                f"GOAL: {goal}\nPROPOSAL: {_trunc(proposal, 360)}\n"
                "Identify exactly 3 specific technical weaknesses that would cause REAL failure. "
                "Be precise, max 160 words.")

    def _revise_prompt(self, proposal, attack):
        return (f"Fix these 3 weaknesses in the proposal:\nWEAKNESSES: {_trunc(attack, 180)}\n"
                f"ORIGINAL: {_trunc(proposal, 340)}\nONLY output the improved PROPOSAL:")

    def _verify_prompt(self, proposal, goal, exec_ev=""):
        ev = f"\nEXECUTION RESULT: {exec_ev}" if exec_ev else ""
        return (
            f"GH05T3 Strict Verifier. You MUST differentiate scores â€” do NOT default to 0.7/0.8.\n"
            f"GOAL: {goal}\nPROPOSAL: {_trunc(proposal, 440)}{ev}\n\n"
            "Rate each axis 0.0-1.0 (use the FULL range):\n"
            "spec  â€” concrete implementation detail? "
            "0.1=just an idea, 0.4=vague steps, 0.7=clear steps, 0.9=real code/commands\n"
            "exec  â€” buildable TODAY? "
            "0.1=needs unknown tech, 0.5=needs setup, 0.8=near-runnable, 1.0=copy-paste ready\n"
            "innov â€” intelligence advance? "
            "0.1=already exists, 0.4=minor tweak, 0.7=meaningful upgrade, 1.0=novel breakthrough\n"
            "rev   â€” safe rollback? "
            "0.2=destructive, 0.6=partial rollback, 0.9=fully reversible\n\n"
            "Think briefly: what score does each axis ACTUALLY deserve for THIS specific proposal?\n"
            'Then output ONLY JSON: {"spec":X,"exec":X,"innov":X,"rev":X,"rationale":"1 sentence"}')

    def _wisdom_prompt(self, proposal, goal):
        return (f"In max 90 chars, what is the KEY INSIGHT of this successful proposal "
                f"for '{goal[:45]}'?\n{_trunc(proposal, 200)}\nINSIGHT (one line):")

    def _evolve_prompt(self, goal):
        return (f"GH05T3 mastered: '{goal}'\n"
                "Generate ONE harder successor goal (max 12 words, no quotes). Output ONLY the goal:")

    def _meta_prompt(self):
        recent = self.scores[-10:]
        avg    = sum(recent) / len(recent) if recent else 0
        return (f"GH05T3 meta-agent. Last-10 avg={avg:.2f}. "
                f"Goal scores: {json.dumps(self.queue.dump(), separators=(',',':'))[:220]}. "
                f"Wisdom: {_load_wisdom(2)[:80]}. "
                "Write ONE specific rule (max 120 chars) to improve weakest goals. ONLY the rule:")

    # â”€â”€ sub-steps â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    def _debate(self, proposal, goal):
        attack  = _call(CRITIC,   [{"role":"user","content":self._attack_prompt(proposal, goal)}],
                        temp=0.8, max_tok=200)
        if attack.startswith("[ERR"): return proposal, ""
        revised = _call(VERIFIER, [{"role":"user","content":self._revise_prompt(proposal, attack)}],
                        temp=0.5, max_tok=700)
        if revised.startswith("[ERR"): return proposal, attack
        return _strip_think(revised), attack

    def _verify(self, proposal, goal, exec_ev=""):
        raw = _call(VERIFIER, [{"role":"user","content":self._verify_prompt(proposal, goal, exec_ev)}],
                    temp=0.1, max_tok=320)
        # Ollama/backend error â€” signal caller to skip cycle, not record a false FAIL
        if raw.startswith("[ERR:"):
            return "ERR", 0.0, raw[:80], {}
        def _fget(src, key, default=0.3):
            if key in src:
                return float(src[key])
            # Regex fallback handles non-JSON or malformed JSON output
            m2 = re.search(rf'"{key}"\s*:\s*([0-9.]+)', raw)
            if m2:
                return float(m2.group(1))
            m3 = re.search(rf'\b{key}\b[^0-9]*([0-9]\.[0-9]+)', raw)
            return float(m3.group(1)) if m3 else default
        try:
            m = re.search(r'\{.*\}', raw, re.DOTALL)
            try:
                d = json.loads(m.group()) if m else {}
            except (json.JSONDecodeError, ValueError):
                d = {}  # JSON malformed â€” _fget regex fallback will handle extraction
            spec  = _fget(d, "spec",  0.3)
            exec_ = _fget(d, "exec",  0.3)
            innov = _fget(d, "innov", 0.3)
            rev   = _fget(d, "rev",   0.5)
            score   = spec*W["spec"] + exec_*W["exec"] + innov*W["innov"] + rev*W["rev"]
            verdict = "PASS" if score >= PASS_THRESH else "FAIL"
            axes_out = {"spec": spec, "exec": exec_, "innov": innov, "rev": rev}
            return verdict, round(score, 3), d.get("rationale", raw[:60])[:80], axes_out
        except Exception:
            return "FAIL", 0.0, raw[:80], {}

    def _try_extract_wisdom(self, proposal, goal, score, vstr):
        # Tier 1: insight (what worked) â€” only for high-scoring PASSes
        if vstr == "PASS" and score >= self._wisdom_thresh:
            r = _call(META, [{"role": "user", "content":
                              f"In 90 chars max, what is the KEY INSIGHT from this successful "
                              f"proposal for '{goal[:45]}'?\n{_trunc(proposal, 200)}\nINSIGHT:"}],
                      temp=0.3, max_tok=55)
            if r and not r.startswith("[ERR"):
                _save_wisdom(r.strip(), "insights", self.domain)
        # Tier 2: warning (what fails) â€” always save low-score FAILs regardless of threshold
        elif vstr == "FAIL" and score < 0.40:
            r = _call(META, [{"role": "user", "content":
                              f"In 90 chars max, what ANTI-PATTERN caused this proposal to fail "
                              f"for '{goal[:45]}'?\n{_trunc(proposal, 200)}\nWARNING:"}],
                      temp=0.3, max_tok=55)
            if r and not r.startswith("[ERR"):
                _save_wisdom(r.strip(), "warnings", self.domain)

    def _try_compact_wisdom(self):
        """Every 50 cycles: compress insights + warnings into master synthesis."""
        if self.cycle % 50 != 0:
            return
        ins  = _wpath("insights", self.domain)
        warn = _wpath("warnings", self.domain)
        if not ins.exists():
            return
        combined = []
        if ins.exists():
            combined += ins.read_text(encoding="utf-8").strip().splitlines()[-20:]
        if warn.exists():
            combined += ["WARNING: " + w
                         for w in warn.read_text(encoding="utf-8").strip().splitlines()[-10:]]
        if not combined:
            return
        blob = "\n".join(combined)
        prompt = (f"Synthesize these training observations into the 8 most important lessons "
                  f"for domain '{self.domain}'. Each lesson max 100 chars, one per line.\n\n"
                  f"{blob[:1200]}\n\nLESSONS:")
        r = _call(META, [{"role": "user", "content": prompt}], temp=0.3, max_tok=350)
        if r and not r.startswith("[ERR"):
            _wpath("master", self.domain).write_text(r.strip(), encoding="utf-8")
            print(f"  WISDOM COMPACT: master updated ({self.domain})")

    def _try_evolve(self, goal, score):
        if self.queue.mastery.get(goal, 0) < 3: return
        r = _call(META, [{"role":"user","content":self._evolve_prompt(goal)}],
                  temp=0.5, max_tok=24)
        if r and not r.startswith("[ERR"):
            new = r.strip().strip('"')[:120]
            self.queue.evolve(goal, new)
            print(f"\n  *** EVOLVED: '{goal[:38]}'\n           -> '{new[:38]}' ***")

    def _try_meta(self):
        if self.cycle % 10 != 0: return
        r = _call(META, [{"role":"user","content":self._meta_prompt()}], temp=0.3, max_tok=80)
        if r and not r.startswith("[ERR"):
            self.meta_rules = r.strip()[:150]
            print(f"  META:     {self.meta_rules}")

    def _pcl_log(self, score, vstr):
        hz = 40.0 + score * 60.0
        rc, gc, bc = int(255*(1-score)), int(255*score), int(128+score*127)
        self.conn.execute("INSERT INTO pcl(ts,hz,r,g,b,label,score) VALUES(?,?,?,?,?,?,?)",
                         (time.time(), hz, rc, gc, bc, f"C{self.cycle:03d}:{vstr}", score))
        self.conn.commit()
        print(f"  PCL:      hz={hz:.0f}  rgb=({rc},{gc},{bc})  [{sparkline(self.scores)}]")

    # â”€â”€ main cycle â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    def run_cycle(self, goal=None):
        self.cycle += 1
        t0 = time.time()

        # FalsificationMonitor injects forced goal when reward stagnates (C4)
        if goal is None and self._forced_goal:
            goal = self._forced_goal
            self._forced_goal = None
        goal = goal or self.queue.pick()

        print(f"\n-- CYCLE {self.cycle:03d} -- {goal[:58]}")

        # 1. PROPOSE (Groq â€” only call that touches quota)
        raw      = _call(PROPOSER, [{"role":"user","content":self._propose_prompt(goal)}],
                         temp=0.72, max_tok=2048)
        proposal = _strip_think(raw)
        if proposal.startswith("[ERR:"):
            print(f"  PROPOSE:  ERROR â€” skipping cycle: {proposal[:60]}")
            self.cycle -= 1
            return None
        print(f"  PROPOSE:  {_trunc(proposal, 85)}")

        # 2. DEBATE â€” Critic attacks, Verifier revises (both Ollama, free)
        proposal, attack = self._debate(proposal, goal)
        if attack:
            print(f"  ATTACK:   {_trunc(attack, 78)}")
            print(f"  REVISED:  {_trunc(proposal, 78)}")

        # 3. SANDBOX â€” execute any embedded Python code, feed results to verifier
        exec_ev = ""
        code    = _extract_code(proposal)
        if code:
            stdout, stderr, rc = _run_sandbox(code)
            tag    = "OK" if rc == 0 else "FAIL"
            exec_ev = f"{tag}: {(stdout or stderr)[:150]}"
            print(f"  SANDBOX:  {tag} | {(stdout or stderr)[:68]}")

        # 4. VERIFY â€” 4-axis rubric (specificity/executability/innovation/reversibility)
        vstr, score, rationale, axes = self._verify(proposal, goal, exec_ev)
        if vstr == "ERR":
            print(f"  VERIFY:   ERR â€” skipping cycle (verifier offline): {rationale[:60]}")
            self.cycle -= 1
            return None
        print(f"  VERIFY:   {vstr} {score:.3f}  "
              f"spec={axes.get('spec','?')} exec={axes.get('exec','?')} "
              f"innov={axes.get('innov','?')} rev={axes.get('rev','?')}")
        print(f"            {rationale}")

        # 5. RLVR reward
        raw_r         = score if vstr == "PASS" else -(1.0 - score)
        self.passes  += vstr == "PASS"
        self.fails   += vstr != "PASS"
        reward        = raw_r - self.baseline
        self.history  = (self.history + [raw_r])[-32:]
        self.baseline = sum(self.history) / len(self.history)
        self.scores.append(score)
        self.queue.update(goal, score)
        pr = self.passes / max(1, self.passes + self.fails)
        print(f"  RLVR:     reward={reward:+.3f}  base={self.baseline:.3f}  pass={pr:.0%}")

        # Trend
        s = self.scores[-8:]
        if len(s) >= 3:
            slope = np.polyfit(np.arange(len(s), dtype=float), s, 1)[0]
            print(f"  {'PLATEAU' if abs(slope)<0.005 else 'TREND'}:   slope={slope:+.4f}")

        # 6. BREAKTHROUGH detection
        self.archive.store(goal, proposal, score)
        if score >= 0.90:
            print(f"\n  *** BREAKTHROUGH  score={score:.3f}  cycle={self.cycle} ***")
            with open("data/breakthroughs.jsonl", "a") as f:
                f.write(json.dumps({"cycle":self.cycle,"goal":goal,
                                    "score":score,"proposal":proposal[:600]}) + "\n")
            if _slack:
                _slack.notify_breakthrough(self.cycle, goal, score, self.domain, proposal[:200])
        if self.cycle % 50 == 0:
            self.archive.prune(500)

        # 7. PCL synesthetic log
        self._pcl_log(score, vstr)

        # 8. Wisdom extraction (3-tier) + compaction every 50 cycles
        self._try_extract_wisdom(proposal, goal, score, vstr)
        self._try_compact_wisdom()

        # 9. FalsificationMonitor â€” 6 auto-intervention conditions
        self.monitor.check(self, self.cycle, goal, score, reward, self.domain)

        # 10. Goal evolution
        self._try_evolve(goal, score)

        # 11. SPIN dataset
        if vstr == "PASS" and self.last_fail:
            self.spin.append({"goal":goal, "chosen":proposal[:400],
                              "rejected":self.last_fail[:400],
                              "rejected_score":self.last_fail_score})
        if vstr != "PASS":
            self.last_fail, self.last_fail_score = proposal, score
        if self.cycle % 5 == 0:
            self._flush_spin()

        # Economy bridge â€” real work earns real credits
        if _eco:
            if vstr == "PASS":
                _eco.on_kairos_pass(self.domain, score)
            mastery_score = self.queue.mastery.get(goal, 0)
            if mastery_score == 1 and score >= 0.8:
                _eco.on_domain_mastery(self.domain, goal, score)

        # 12. Meta-agent rule update
        self._try_meta()

        elapsed = time.time() - t0
        print(f"  TIME:     {elapsed:.1f}s")

        # Slack notification â€” PASS cycles and every 10th cycle
        if _slack and (vstr == "PASS" or self.cycle % 10 == 0):
            _slack.notify_cycle(
                cycle=self.cycle, goal=goal, verdict=vstr, score=score,
                axes=axes, domain=self.domain,
                proposal_excerpt=proposal[:140], elapsed=elapsed
            )

        self.save()
        return R(self.cycle, goal, proposal, score, reward)

    # â”€â”€ run loop â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    def run(self, n=None):
        dom_label = self.domain.upper().center(17)
        print("\n+=============================================+")
        print("|  GH05T3  DEEP INTELLIGENCE TRAINER  v4      |")
        print(f"|  Domain   : {dom_label}              |")
        print("|  100% LOCAL -- zero cloud quota dependency   |")
        print("|  Propose  : gemma3:12b       [Ollama]        |")
        print("|  Verify   : qwen2.5:7b       [Ollama]        |")
        print("|  Critic   : dolphin-llama3:8b [Ollama]       |")
        print("|  Meta     : llama3.2:3b       [Ollama]        |")
        print("|  Features : debate / sandbox / 4-axis rubric |")
        print("|             wisdom / goal-evolution / SPIN   |")
        print("+=============================================+")
        print(f"  cycle={self.cycle}  base={self.baseline:.3f}  goals={len(self.queue.goals)}  Ctrl+C to stop\n")

        i = 0
        try:
            while n is None or i < n:
                self.run_cycle()
                i += 1
        except KeyboardInterrupt:
            print("\n-- INTERRUPTED --")

        self._flush_spin()
        total = self.passes + self.fails
        pr    = self.passes / max(1, total)
        avg   = sum(self.scores) / len(self.scores) if self.scores else 0.0
        bt  = sum(1 for _ in open("data/breakthroughs.jsonl")) \
              if Path("data/breakthroughs.jsonl").exists() else 0
        sp  = sum(1 for _ in open("data/spin_dataset.jsonl")) \
              if Path("data/spin_dataset.jsonl").exists() else 0
        arc = self.conn.execute("SELECT COUNT(*) FROM archive").fetchone()[0]
        # 3-tier wisdom line count
        wl = sum(
            len(_wpath(t, self.domain).read_text(encoding="utf-8").strip().splitlines())
            for t in ("insights", "warnings", "master")
            if _wpath(t, self.domain).exists()
        )
        flog = Path("data/falsification_log.jsonl")
        interventions = sum(1 for _ in open(flog)) if flog.exists() else 0

        print(f"\n-- SESSION SUMMARY [{self.domain.upper()}] --")
        print(f"  Cycles        : {i} session / {self.cycle} total")
        print(f"  Pass rate     : {pr:.0%}  ({self.passes}P / {self.fails}F)")
        print(f"  Avg score     : {avg:.3f}  baseline={self.baseline:.3f}")
        print(f"  Breakthroughs : {bt}")
        print(f"  Wisdom lines  : {wl} (insights/warnings/master)")
        print(f"  SPIN pairs    : {sp}")
        print(f"  Archive       : {arc} entries")
        print(f"  Interventions : {interventions} falsification events")
        if self.queue.evolved():
            print(f"  Evolved goals : {len(self.queue.evolved())}")
        if self.meta_rules:
            print(f"  Meta rule     : {self.meta_rules[:80]}")
        print(f"  Sparkline     : [{sparkline(self.scores, 40)}]")
        print("  Checkpoint saved\n")


if __name__ == "__main__":
    from ghost_domains import list_domains
    ap = argparse.ArgumentParser(description="GH05T3 Deep Intelligence Trainer v4")
    ap.add_argument("-n", type=int, default=None,
                    help="number of cycles (default: infinite)")
    ap.add_argument("--goal", help="single-cycle goal override")
    _domain_names = [name for name, _ in list_domains()]
    ap.add_argument("--domain", default="core",
                    choices=_domain_names,
                    help=f"training domain (default: core). Available: {', '.join(_domain_names)}")
    ap.add_argument("--list-domains", action="store_true",
                    help="print available domains and exit")
    args = ap.parse_args()

    if args.list_domains:
        print("\nAvailable GH05T3 training domains:")
        for name, desc in list_domains():
            print(f"  {name:<16} {desc[:70]}")
        print()
        sys.exit(0)

    t = Trainer(domain=args.domain)
    if args.goal:
        t.run_cycle(args.goal)
        t.save()
    else:
        t.run(args.n)
