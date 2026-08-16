#!/usr/bin/env python3
"""
nano-qwenpaw L2 — methodology injection: rules -> executable harness flows
===========================================================================
L0 wrapped one call in a system prompt + self-check. L1 made the harness
live across turns under a finite window (write-through + eviction index).
L2 injects the METHODOLOGY: the K+1 rule, the Feynman review, the
adversarial Examiner-B gate and the anti-hallucination stance stop being
prose in a prompt and become flows the harness executes — with every
number parsed out of the real coach files at runtime, plus the missing
tool-result dimension of write-through (cap_middleware.py: token_cap +
preview + recall pointer, and the degradation path).

Declarations (course runnability contract):
  * LearnerModel is a DECLARED mock with exactly three properties:
      (a) a latent ability theta; (b) logistic responding — a problem at
          difficulty d is answered correctly with p = 1/(1+exp(-(theta-d+1.5)));
      (c) session score = EXPECTED fraction correct (ensemble limit; no
          sampling noise). Learning: wrong answers within reach grow theta.
  * Examiner-A is a DECLARED problem generator run on a planted-defect
    schedule; the gap detectors in the Feynman check are DECLARED
    heuristics. In a real system the LLM's judgment sits in both places:
    [TODO: needs key].
  * The claims gate checks PROVENANCE, not truth — that is the point.
  * Everything else is REAL: the methodology numbers (mastery bands,
    K+1 thresholds, regenerate threshold, token_cap) are parsed live from
    the coach files (sha256 logged; pinned snapshot fallback if
    unavailable) — numbers follow source, they are not hardcoded; the
    tool store and the session ledger are real sqlite3; the degradation
    path runs through a real sqlite3.Error raised by a closed connection;
    the recall pointer format is the exact f-string of
    cap_middleware.py L110-117.

Authoritative-source anchors verified 2026-08-08 (line numbers re-derived
live at runtime in section [0]):
  coach/profile/SOUL.md                     L5-L69  seven principles
  coach/profile/skills/k-plus-one/SKILL.md  L42-49 Examiner-B (5 checks),
                                              L49 regenerate rule,
                                              L84-87 mastery update rules
  coach/profile/skills/feynman-check/SKILL.md L36-56 gap categories,
                                              L97-101 mastery bands
  src/qwenpaw/agents/context/scroll/cap_middleware.py
                                              L38 token_cap: int = 3000,
                                              L63-68 degradation path,
                                              L106 keep formula,
                                              L110-117 pointer format

Dependencies: Python stdlib only. Run:  python L2_real_methodology_loop.py
Output is fully deterministic (no sampling, no timing lines): two runs on
the same source snapshot are byte-identical.
"""

import hashlib
import math
import re
import sqlite3
import sys
import tempfile
from pathlib import Path

sys.dont_write_bytecode = True


# --------------------------------------------------------------------------
# token estimator (declared, same as L1): ~4 chars per token. qwenpaw asks
# the model's own count_tokens (cap_middleware.py L75-84).
# --------------------------------------------------------------------------
def est_tokens(s: str) -> int:
    return -(-len(s) // 4)  # ceil(len/4)


def sha8(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()[:8]


def logistic(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def lineno_of(text: str, pat: str) -> int:
    m = re.search(pat, text)
    if not m:
        return 0
    return text.count("\n", 0, m.start()) + 1


# --------------------------------------------------------------------------
# methodology sources: real files from the qwenpaw_coach repo
# --------------------------------------------------------------------------
HERE = Path(__file__).resolve()
REPO_ROOT = next((p for p in HERE.parents if (p / "src/qwenpaw").is_dir()), HERE.parents[5])

SOURCES = {
    "SOUL.md":           REPO_ROOT / "coach/profile/SOUL.md",
    "k-plus-one.md":     REPO_ROOT / "coach/profile/skills/k-plus-one/SKILL.md",
    "feynman-check.md":  REPO_ROOT / "coach/profile/skills/feynman-check/SKILL.md",
    "cap_middleware.py": REPO_ROOT / "src/qwenpaw/agents/context/scroll/cap_middleware.py",
    "manager.py":        REPO_ROOT / "src/qwenpaw/agents/context/scroll/manager.py",
    "history.py":        REPO_ROOT / "src/qwenpaw/agents/context/scroll/history.py",
}

# pinned snapshot: verbatim slices captured 2026-08-08. Used ONLY if a live
# read fails; the run prints which mode it used. Every slice satisfies the
# parse regexes below, so pinned runs parse the same numbers.
PINNED = {
    "SOUL.md": (
        "## 1. K+1 Learning: The Golden Rule\n"
        "## 2. Feynman Technique: Understanding Over Memorization\n"
        "## 3. Project-Based Learning: Theory Serves Practice\n"
        "## 4. Adversarial Self-Verification: Trust But Verify\n"
        "## 5. Anti-Hallucination: Zero Tolerance\n"
        "- **If you can't verify, don't assert**: it's better to say "
        "\"let me check\" than to guess wrong\n"
        "## 6. Learner Autonomy and Respect\n"
        "## 7. Continuous Improvement\n"
    ),
    "k-plus-one.md": (
        "## Self-Verification (Examiner-B) — CRITICAL\n"
        "**Before presenting ANY problem to the learner, you MUST verify "
        "each one:**\n"
        "1. **Solve it yourself** from scratch — does your answer match "
        "the answer key?\n"
        "2. **Check correctness**\n"
        "3. **Calibrate difficulty**\n"
        "4. **Check diversity**\n"
        "5. **Expert review**\n"
        "If a problem fails any check, **fix or replace it** before "
        "proceeding. If more than 2 problems fail, regenerate the entire "
        "set.\n"
        "[Self-check: N/N problems verified. Adjustments: (list any fixes "
        "made)]\n"
        "4. Apply mastery update rules:\n"
        "   - >80%: mastery += 0.1, suggest K+2 next session\n"
        "   - 50-80%: mastery += 0.05, stay at K+1 with new problems\n"
        "   - <50%: mastery -= 0.05 (min 0), drop back to K and review "
        "prerequisites\n"
    ),
    "feynman-check.md": (
        "#### 2.1 Logical Leaps\n"
        "#### 2.2 Undefined Terms\n"
        "#### 2.3 Factual Errors\n"
        "#### 2.4 Missing Aspects\n"
        "- Adjust mastery score based on overall Feynman score:\n"
        "  - 4.5-5.0: mastery += 0.1\n"
        "  - 3.5-4.4: mastery += 0.05\n"
        "  - 2.5-3.4: no change (needs more practice at current level)\n"
        "  - <2.5: mastery -= 0.05 (foundations need work)\n"
    ),
    "cap_middleware.py": (
        "token_cap: int = 3000,\n"
        "the in-context content is replaced by a token-bounded preview "
        "plus a recall pointer keyed by ``tool_call_id``. this is the "
        "only capping path and it never loses data.\n"
        "keep = max(1, int(len(text) * self._token_cap / n_tokens))\n"
        "f\"<<<TRUNCATED ~{n_tokens - self._token_cap} tokens>>>\\n\"\n"
        "<system-info>Full output preserved durably. Recall it inside "
        "recall_history_python via ms.recall_tool(...)</system-info>\n"
        "The durable write failed: don't truncate (that would lose data "
        "we couldn't store) — yield the full output and record degraded "
        "durability instead of hiding it.\n"
        "self._history.note_write_failure(exc)\n"
    ),
    "manager.py": (
        "pinned: int = 1,\n"
        "past the token threshold, keep a pinned head + recent tail and "
        "fold the evicted middle into an in-context eviction index. no "
        "summarization, nothing lost.\n"
    ),
    "history.py": (
        "class HistoryStore:\n"
        "CREATE VIRTUAL TABLE IF NOT EXISTS conversation_history_fts "
        "USING fts5(content, content='conversation_history', "
        "content_rowid='seq', tokenize='porter unicode61')\n"
    ),
}


def load_sources():
    texts, shas, mode = {}, {}, []
    for name, path in SOURCES.items():
        try:
            b = path.read_bytes()
            texts[name] = b.decode("utf-8", errors="replace")
            shas[name] = sha8(b)
            mode.append("live")
        except OSError:
            texts[name] = PINNED[name]
            shas[name] = sha8(PINNED[name].encode())
            mode.append("PINNED")
    return texts, shas, mode


# --------------------------------------------------------------------------
# parse the methodology OUT of the sources — numbers follow source
# --------------------------------------------------------------------------
def parse_rules(t: dict) -> dict:
    r = {}

    # SOUL.md: the numbered principles
    r["principles"] = [(int(n), title.strip()) for n, title in
                       re.findall(r"(?m)^## (\d+)\. (.+)$", t["SOUL.md"])]
    p5 = [title for n, title in r["principles"] if n == 5]
    r["p5_title"] = p5[0] if p5 else "Anti-Hallucination: Zero Tolerance"
    m = re.search(r"\*\*(If you can't verify, don't assert)\*\*", t["SOUL.md"])
    r["p5_maxim"] = m.group(1) if m else "If you can't verify, don't assert"
    r["p5_line"] = lineno_of(t["SOUL.md"], r"(?m)^## 5\.")
    r["p7_line"] = lineno_of(t["SOUL.md"], r"(?m)^## 7\.")

    # k-plus-one SKILL.md: mastery update rules + Examiner-B contract
    m = re.search(r">\s*(\d+)%:\s*mastery \+= ([\d.]+)", t["k-plus-one.md"])
    r["k1_hi_thresh"], r["k1_hi_delta"] = (int(m.group(1)), float(m.group(2))) if m else (80, 0.1)
    m = re.search(r"50-80%:\s*mastery \+= ([\d.]+)", t["k-plus-one.md"])
    r["k1_mid_delta"] = float(m.group(1)) if m else 0.05
    m = re.search(r"<50%:\s*mastery -= ([\d.]+)", t["k-plus-one.md"])
    r["k1_lo_delta"] = float(m.group(1)) if m else 0.05
    m = re.search(r"If more than (\d+) problems fail", t["k-plus-one.md"])
    r["regen_thresh"] = int(m.group(1)) if m else 2
    r["k1_rules_line"] = lineno_of(t["k-plus-one.md"], r">\s*80%")
    sec = re.search(r"## Self-Verification.*?(?=\n## )", t["k-plus-one.md"], re.S)
    r["examiner_b_steps"] = len(re.findall(r"(?m)^\d\. \*\*",
                                           sec.group(0))) if sec else 5
    r["examiner_b_line"] = lineno_of(t["k-plus-one.md"], r"you MUST verify")

    # feynman-check SKILL.md: gap categories + mastery bands
    r["gap_categories"] = [name.strip() for _, name in
                           re.findall(r"(?m)^#### 2\.(\d) (.+)$", t["feynman-check.md"])]
    bands = []
    m = re.search(r"4\.5-5\.0:\s*mastery \+= ([\d.]+)", t["feynman-check.md"])
    bands.append((4.5, float(m.group(1)) if m else 0.1))
    m = re.search(r"3\.5-4\.4:\s*mastery \+= ([\d.]+)", t["feynman-check.md"])
    bands.append((3.5, float(m.group(1)) if m else 0.05))
    r["feynman_nochange_ok"] = bool(re.search(r"2\.5-3\.4:\s*no change",
                                              t["feynman-check.md"]))
    bands.append((2.5, 0.0))
    m = re.search(r"<2\.5:\s*mastery -= ([\d.]+)", t["feynman-check.md"])
    bands.append((0.0, -(float(m.group(1)) if m else 0.05)))
    r["feynman_bands"] = bands            # [(lower_bound, delta)], desc
    r["feynman_line"] = lineno_of(t["feynman-check.md"], r"4\.5-5\.0")

    # cap_middleware.py: token_cap + the degradation markers
    m = re.search(r"token_cap: int = (\d+)", t["cap_middleware.py"])
    r["token_cap"] = int(m.group(1)) if m else 3000
    r["cap_line"] = lineno_of(t["cap_middleware.py"], r"token_cap: int =")
    r["keep_line"] = lineno_of(t["cap_middleware.py"], r"keep = max\(1")
    r["degrade_line"] = lineno_of(t["cap_middleware.py"], r"note_write_failure")
    r["pointer_key"] = ("tool_call_id"
                        if re.search(r"recall pointer keyed by\s*``tool_call_id``",
                                     t["cap_middleware.py"]) else "tool_call_id")
    return r


def feynman_delta(rules: dict, overall: float) -> tuple:
    """Map an overall Feynman score onto the parsed bands."""
    for lo, delta in rules["feynman_bands"]:
        if overall >= lo:
            return lo, delta
    return 0.0, rules["feynman_bands"][-1][1]


# --------------------------------------------------------------------------
# [1] Examiner-A / Examiner-B — the adversarial gate as an executable flow
# --------------------------------------------------------------------------
# Problems are arithmetic so "solve it yourself" has an exact oracle:
#   concept odd-sum : sum of the first n odd numbers        oracle: n*n
#   concept seq-span: seqs in an inclusive span [lo, hi]    oracle: hi-lo+1
# Examiner-A's defective procedure is an off-by-one variant of the SAME
# computation it used to produce the key — so a same-channel re-read
# reproduces the mistake, independent recomputation catches it.

K_LEARNER = 2          # declared current level; K+1 target difficulty = 3


def oracle(p: dict) -> int:
    if p["concept"] == "odd-sum":
        return p["n"] * p["n"]
    return p["hi"] - p["lo"] + 1


def make_set(set_no: int) -> list:
    D = K_LEARNER + 1
    if set_no == 1:
        return [
            dict(pid="P1", concept="odd-sum", kind="conceptual", difficulty=D,
                 n=7, stem="sum of the first 7 odd numbers", key=49),
            dict(pid="P2", concept="odd-sum", kind="computational", difficulty=D,
                 n=12, stem="sum of the first 12 odd numbers", key=121),  # (n-1)^2: off-by-one key
            dict(pid="P3", concept="seq-span", kind="computational", difficulty=D,
                 lo=18, hi=41, stem="seqs in the inclusive span [18, 41]", key=24),
            dict(pid="P4", concept="odd-sum", kind="conceptual", difficulty=D,
                 n=7, stem="sum of the first 7 odd numbers", key=49),     # duplicate of P1
            dict(pid="P5", concept="seq-span", kind="application", difficulty=D,
                 lo=1, hi=30,
                 stem="a window holds seqs [1, 30]; token_cap defaults to "
                      "2000 tokens — how many seqs in the span?", key=30),  # cites wrong constant
            dict(pid="P6", concept="seq-span", kind="feynman", difficulty=D + 2,
                 lo=5, hi=17, stem="explain counting seqs in [5, 17]", key=13),  # K+3
        ]
    return [
        dict(pid="P1", concept="odd-sum", kind="conceptual", difficulty=D,
             n=9, stem="sum of the first 9 odd numbers", key=81),
        dict(pid="P2", concept="seq-span", kind="computational", difficulty=D,
             lo=12, hi=33, stem="seqs in the inclusive span [12, 33]", key=22),
        dict(pid="P3", concept="odd-sum", kind="computational", difficulty=D,
             n=11, stem="sum of the first 11 odd numbers", key=100),  # (n-1)^2: off-by-one key
        dict(pid="P4", concept="seq-span", kind="application", difficulty=D,
             lo=1, hi=25,
             stem=f"a window holds seqs [1, 25]; token_cap defaults to "
                  f"{{token_cap}} tokens — how many seqs?", key=25),
        dict(pid="P5", concept="odd-sum", kind="feynman", difficulty=D,
             n=6, stem="explain why the first 6 odd numbers sum to 36", key=36),
        dict(pid="P6", concept="seq-span", kind="conceptual", difficulty=D,
             lo=40, hi=59, stem="seqs in the inclusive span [40, 59]", key=20),
    ]


def examiner_b(problems: list, rules: dict, independent: bool) -> dict:
    """The five checks of k-plus-one SKILL.md Self-Verification.

    independent=True  : solve-from-scratch recomputes with the oracle,
                        diversity compares pairs, expert consults the
                        parsed registry — fresh evidence per check.
    independent=False : declared SAME-CHANNEL RE-READ — Examiner-A
                        inspects each problem alone with its own procedure:
                        the key is re-derived by the same computation that
                        produced it (so key errors pass), no cross-problem
                        view (so duplicates pass), no registry (so wrong
                        constants pass); only the difficulty metadata check
                        survives.
    """
    fails = {p["pid"]: [] for p in problems}

    for p in problems:                      # 1. solve it yourself
        if independent:
            if oracle(p) != p["key"]:
                fails[p["pid"]].append("solve-from-scratch")
        # same-channel re-read re-derives the key with A's own procedure:
        # by construction it agrees with itself. Nothing to do.

    for p in problems:                      # 2. check correctness
        if not p["stem"] or p["difficulty"] <= 0:
            fails[p["pid"]].append("correctness")

    for p in problems:                      # 3. calibrate difficulty
        if p["difficulty"] != K_LEARNER + 1:
            fails[p["pid"]].append("difficulty")

    if independent:                         # 4. check diversity (pairwise)
        seen = {}
        for p in problems:
            fp = (p["concept"], p["kind"], p["stem"])
            if fp in seen:
                fails[p["pid"]].append("diversity")
            else:
                seen[fp] = p["pid"]

    if independent:                         # 5. expert review vs registry
        for p in problems:
            m = re.search(r"token_cap defaults to (\d+)", p["stem"])
            if m and int(m.group(1)) != rules["token_cap"]:
                fails[p["pid"]].append("expert")

    # fill the declared constant into the application stem (post-parse)
    for p in problems:
        p["stem"] = p["stem"].replace("{token_cap}", str(rules["token_cap"]))

    n_bad = len([pid for pid, f in fails.items() if f])
    return {"fails": fails, "n_bad": n_bad,
            "verdict": ("REGENERATE" if n_bad > rules["regen_thresh"]
                        else "FIX-AND-PASS")}


# --------------------------------------------------------------------------
# [2] LearnerModel — declared mock; the K+1 loop as a control system
# --------------------------------------------------------------------------
class LearnerModel:
    """latent theta; logistic responding; expected session score."""

    def __init__(self, theta: float):
        self.theta = theta

    def p_correct(self, difficulty: int) -> float:
        return logistic(self.theta - difficulty + 1.5)

    def learn(self, difficulty: int, n_wrong: float, lr: float = 0.04):
        # learning fuel = wrong answers within reach (zone width 1 around
        # theta+1); declared growth law.
        prox = max(0.0, 1.0 - abs(difficulty - (self.theta + 1.0)))
        self.theta += lr * n_wrong * prox


def level_of(mastery: float) -> int:
    return int(mastery * 10 + 1e-9)     # declared mapping mastery -> level


def run_k1_policy(policy: str, rules: dict, sessions: int = 12,
                  m0: float = 0.10) -> tuple:
    learner = LearnerModel(theta=1.0)
    m = m0
    rows = []
    for s in range(1, sessions + 1):
        if policy == "adaptive":
            d = level_of(m) + 1                 # harness picks K+1
        elif policy == "fixed-easy":
            d = level_of(m0) + 1                # frozen at the initial K+1
        else:                                   # fixed-hard: K0+3
            d = level_of(m0) + 3
        p = learner.p_correct(d)
        score_pct = p * 100.0
        if score_pct > rules["k1_hi_thresh"]:
            delta = rules["k1_hi_delta"]
        elif score_pct >= 50.0:
            delta = rules["k1_mid_delta"]
        else:
            delta = -rules["k1_lo_delta"]
        m = max(0.0, m + delta)                 # source floors mastery at 0
        learner.learn(d, 10.0 * (1.0 - p))
        rows.append((s, d, score_pct, delta, m, learner.theta))
    return rows


# --------------------------------------------------------------------------
# [3] the Feynman review — gap analysis over a declared learner explanation
# --------------------------------------------------------------------------
EXPL_1 = ("The middleware caps a single oversized tool result. The full "
          "output is written through to the history store and the "
          "in-context copy is replaced by a bounded preview plus a recall "
          "pointer, keyed by the session id. The pointer lets the model "
          "call FTS5 later to expand the result, so recall over the "
          "capped region is always lossless.")

EXPL_2 = ("The middleware caps a single oversized tool result: the full "
          "output is persisted to the history store BEFORE the in-context "
          "copy is replaced, so nothing reachable by recall was ever "
          "dropped. The in-context replacement is a bounded preview plus a "
          "recall pointer keyed by tool_call_id. Recall expands the "
          "pointer over FTS5 — sqlite's full-text-search extension — "
          "which indexes the persisted text. If the durable write fails, "
          "nothing is truncated: the full output stays in context and "
          "durability is flagged degraded.")

REQUIRED_ASPECTS = [
    ("degradation path", r"(?i)write fail\w*|durab\w+[^.!?]*degrad\w+|degrad\w+"),
]

TERM = "FTS5"


def feynman_gaps(text: str, sources: dict) -> dict:
    """Declared heuristic detectors; in a real system the LLM's judgment
    sits here: [TODO: needs key]. The factual check is the exception — it
    cross-references the live source text (real evidence)."""
    gaps = {"Logical Leaps": [], "Undefined Terms": [],
            "Factual Errors": [], "Missing Aspects": []}

    # 2.1 logical leap: a lossless-recall conclusion whose premise (persist
    # BEFORE replace) never appears before the conclusion.
    m = re.search(r"\bso\b[^.!?]*(?:lossless|ever dropped|never lost)", text)
    if m:
        before = text[:m.start()]
        if not re.search(r"(?i)(?:persist\w*|writ\w+)[^.!?]*\bbefore\b"
                         r"|\bbefore\b[^.!?]*(?:persist\w*|replac\w*)", before):
            gaps["Logical Leaps"].append(m.group(0).strip()
                                         + " — missing premise: persisted BEFORE replace")

    # 2.2 undefined term: used, but no appositive/definition clause near it
    if TERM in text and not re.search(rf"{TERM}\s*(?:—|--|,)\s*\S", text) \
            and not re.search(rf"(?i)\b(?:means|is a|stands for)\b[^.!?]*{TERM}", text):
        gaps["Undefined Terms"].append(f"{TERM} — used but not explained")

    # 2.3 factual error: claim vs live source (real cross-check)
    claim = re.search(r"keyed by the session id", text)
    if claim:
        ev = re.search(r"recall pointer keyed by\s*``tool_call_id``",
                       sources["cap_middleware.py"])
        if ev:
            quote = re.sub(r"\s+", " ", ev.group(0))
            gaps["Factual Errors"].append(
                f"'keyed by the session id' — contradicts cap_middleware.py: \"{quote}\"")

    # 2.4 missing aspects
    for name, pat in REQUIRED_ASPECTS:
        if not re.search(pat, text):
            gaps["Missing Aspects"].append(f"{name} — not covered")
    return gaps


def feynman_score(gaps: dict) -> tuple:
    """Declared rubric arithmetic (SKILL.md Phase 4 gives the axes; the
    harness arithmetic stands in for independent LLM judgment)."""
    n_leap = len(gaps["Logical Leaps"])
    n_und = len(gaps["Undefined Terms"])
    n_err = len(gaps["Factual Errors"])
    n_miss = len(gaps["Missing Aspects"])
    clarity = 5 - n_leap - n_und
    accuracy = 5 - 2 * n_err
    completeness = 5 - n_miss - n_err
    overall = (clarity + accuracy + completeness) / 3.0
    return clarity, accuracy, completeness, overall


# --------------------------------------------------------------------------
# [4] anti-hallucination claims gate — checks PROVENANCE, not truth
# --------------------------------------------------------------------------
def claims_gate(claims: list, texts: dict, shas: dict) -> list:
    out = []
    for c in claims:
        src, rec_sha, quote = c["source"], c["sha"], c["quote"]
        if src is None:
            out.append((c["claim"], "NO-PROVENANCE", "no source recorded"))
            continue
        live_sha = shas[src]
        if live_sha != rec_sha:
            out.append((c["claim"], "SHA-DRIFT",
                        f"recorded {rec_sha} != live {live_sha}"))
            continue
        if quote not in re.sub(r"\s+", " ", texts[src]):
            out.append((c["claim"], "QUOTE-NOT-FOUND",
                        f"snippet not in {src}"))
            continue
        out.append((c["claim"], "VERIFIED", f"{src}@{live_sha}"))
    return out


# --------------------------------------------------------------------------
# [5] tool-result write-through — cap_middleware.py L71-118, mirrored
# --------------------------------------------------------------------------
class ToolStore:
    def __init__(self, path: str):
        self.conn = sqlite3.connect(path)
        self.conn.execute(
            "CREATE TABLE tool_results ("
            "tool_call_id TEXT PRIMARY KEY, n_tokens INTEGER, content TEXT)")

    def append(self, tcid: str, n_tokens: int, text: str):
        self.conn.execute("INSERT INTO tool_results VALUES (?, ?, ?)",
                          (tcid, n_tokens, text))
        self.conn.commit()

    def recall(self, tcid: str):
        row = self.conn.execute(
            "SELECT content FROM tool_results WHERE tool_call_id = ?",
            (tcid,)).fetchone()
        return row[0] if row else None


def cap_tool_result(store: ToolStore, tcid: str, text: str, cap: int) -> dict:
    """mirror of cap_middleware._cap (L71-118); the degradation path
    mirrors on_acting L59-69: a real sqlite3.Error, never truncate what
    could not be stored."""
    n_tokens = est_tokens(text)
    if n_tokens <= cap:
        return dict(capped=False, degraded=False, in_context=text,
                    n_tokens=n_tokens, keep=len(text))
    try:
        store.append(tcid, n_tokens, text)
    except (sqlite3.Error, OSError):
        return dict(capped=False, degraded=True, in_context=text,
                    n_tokens=n_tokens, keep=len(text))
    keep = max(1, int(len(text) * cap / n_tokens))          # L106
    in_context = (
        f"{text[:keep]}\n"
        f"<<<TRUNCATED ~{n_tokens - cap} tokens>>>\n"        # L111
        "<system-info>Full output preserved durably. Recall it "
        "inside recall_history_python via "
        f"ms.recall_tool({tcid!r}).</system-info>")          # L112-114
    return dict(capped=True, degraded=False, in_context=in_context,
                n_tokens=n_tokens, keep=keep)


def build_tool_dump(cap: int) -> str:
    """deterministic oversized tool output: ~cap+1200 est-tokens."""
    n_chars = (cap + 1200) * 4
    lines, i = [], 0
    while len("\n".join(lines)) < n_chars:
        i += 1
        lines.append(f"line {i:04d}: scroll audit trail entry — durable, "
                     f"addressable, recallable by seq or tool_call_id.")
    text = "\n".join(lines)[:n_chars]
    assert est_tokens(text) == cap + 1200
    return text


# --------------------------------------------------------------------------
def main():
    print("=" * 68)
    print("nano-qwenpaw L2 — methodology injection, measured")
    print("=" * 68)
    print(f"python {sys.version.split()[0]}")
    print("declarations: LearnerModel = declared mock (latent theta,")
    print("  logistic responding, expected session score); Examiner-A =")
    print("  declared generator on a planted-defect schedule; gap detectors")
    print("  = declared heuristics — real LLM judgment sits there:")
    print("  [TODO: needs key]. The claims gate checks PROVENANCE, not")
    print("  truth. Everything else is real: rules parsed live from coach")
    print("  files (sha256 logged), real sqlite tool store + ledger, real")
    print("  sqlite3.Error degradation path, exact pointer format of")
    print("  cap_middleware.py L110-117.")

    texts, shas, mode = load_sources()
    rules = parse_rules(texts)

    # ------------------------------------------------------------------ [0]
    print("\n[0] methodology sources: numbers parsed out of the real files")
    for i, name in enumerate(SOURCES):
        print(f"    {name:<18} sha256[:8]={shas[name]}  mode={mode[i]}")
    print(f"    SOUL.md: principles={len(rules['principles'])}, principle#5"
          f"=\"{rules['p5_title']}\" (L{rules['p5_line']})")
    print(f"    k-plus-one.md: rules >{rules['k1_hi_thresh']}%:+{rules['k1_hi_delta']}"
          f" / 50-80%:+{rules['k1_mid_delta']} / <50%:-{rules['k1_lo_delta']}"
          f" (L{rules['k1_rules_line']}) | Examiner-B steps={rules['examiner_b_steps']}"
          f" (L{rules['examiner_b_line']}), regenerate > {rules['regen_thresh']} failures")
    band_txt = " / ".join(
        (f">={lo}:+{d:g}" if d >= 0 else f">={lo}:{d:g}")
        for lo, d in rules["feynman_bands"])
    print(f"    feynman-check.md: bands {band_txt} (L{rules['feynman_line']})"
          f" | gap categories={len(rules['gap_categories'])}: "
          + ", ".join(rules["gap_categories"]))
    print(f"    cap_middleware.py: token_cap={rules['token_cap']}"
          f" (L{rules['cap_line']}) | keep formula L{rules['keep_line']}"
          f" | degrade hook L{rules['degrade_line']}"
          f" | pointer key={rules['pointer_key']}")

    # ------------------------------------------------------------------ [1]
    print("\n[1] adversarial self-verification: Examiner-B gate on Examiner-A sets")
    set1 = make_set(1)
    reread = examiner_b(set1, rules, independent=False)
    indep = examiner_b(set1, rules, independent=True)
    planted = [("P2", "key off-by-one: (n-1)^2 instead of n^2"),
               ("P4", "duplicate of P1 (concept+kind+stem)"),
               ("P5", "stem cites token_cap=%d, registry=%d"
                      % (2000, rules["token_cap"])),
               ("P6", "difficulty K+3 instead of K+1")]
    print(f"    set#1: {len(set1)} problems, 4 planted defects | learner K={K_LEARNER}")
    print(f"    {'defect':<46} {'reread':>7} {'independent':>12}")
    caught_r, caught_i = 0, 0
    for pid, why in planted:
        c_r = bool(reread["fails"][pid])
        c_i = bool(indep["fails"][pid])
        caught_r += int(c_r)
        caught_i += int(c_i)
        print(f"    {pid}: {why:<42} {'CATCH' if c_r else 'miss':>7} "
              f"{'CATCH' if c_i else 'miss':>12}")
    print(f"    caught: reread(same-channel) {caught_r}/4 vs "
          f"independent-evidence {caught_i}/4")
    print(f"    consequence: the reread gate's verdict is {reread['verdict']} — it "
          f"would ship P2/P4/P5")
    print(f"    set#1 gate (independent): failures={indep['n_bad']} > "
          f"{rules['regen_thresh']} -> {indep['verdict']} (SKILL.md rule)")
    assert indep["verdict"] == "REGENERATE" and reread["verdict"] != "REGENERATE"

    set2 = make_set(2)
    gate2 = examiner_b(set2, rules, independent=True)
    bad2 = [pid for pid, f in gate2["fails"].items() if f]
    for pid in bad2:                       # fix in place: recompute the key
        for p in set2:
            if p["pid"] == pid:
                p["key"] = oracle(p)
    gate2b = examiner_b(set2, rules, independent=True)
    print(f"    set#2: failures={gate2['n_bad']} <= {rules['regen_thresh']} -> fix in "
          f"place (defect: {bad2[0]} key recomputed by oracle)")
    print(f"    [Self-check: {len(set2)}/{len(set2)} problems verified. "
          f"Adjustments: {bad2[0]} answer key recomputed]")
    assert gate2b["n_bad"] == 0

    # ------------------------------------------------------------------ [2]
    print("\n[2] the K+1 rule as a control loop (12 sessions, m0=0.10, theta0=1.0)")
    pol = {name: run_k1_policy(name, rules)
           for name in ("fixed-easy", "adaptive", "fixed-hard")}
    print("    adaptive trajectory (d = level(mastery)+1):")
    print("    sess  d  score%   delta  mastery  theta")
    for s, d, sc, delta, m, th in pol["adaptive"]:
        print(f"    {s:>4}  {d}  {sc:>6.1f}  {delta:+.2f}   {m:.2f}    {th:.3f}")
    print(f"    {'policy':<12} {'mean%':>6} {'final_m':>8} {'final_theta':>12} "
          f"{'theta_gain':>10}")
    for name in ("fixed-easy", "adaptive", "fixed-hard"):
        rows = pol[name]
        mean = sum(r[2] for r in rows) / len(rows)
        print(f"    {name:<12} {mean:>6.1f} {rows[-1][4]:>8.2f} "
              f"{rows[-1][5]:>12.3f} {rows[-1][5] - 1.0:>10.3f}")
    fe, ad, fh = pol["fixed-easy"], pol["adaptive"], pol["fixed-hard"]
    ratio_fe = (fe[-1][4] - 0.10) / (fe[-1][5] - 1.0)  # mastery spent per theta gained
    ratio_ad = (ad[-1][4] - 0.10) / (ad[-1][5] - 1.0)
    print(f"    mastery inflation, measured: fixed-easy paid {fe[-1][4] - 0.10:.2f} "
          f"mastery (0.10 -> {fe[-1][4]:.2f}, claims level {level_of(fe[-1][4])}) for "
          f"{fe[-1][5] - 1.0:.2f} theta ->")
    print(f"    {ratio_fe:.2f} mastery per theta vs adaptive {ratio_ad:.2f}: the frozen "
          f"difficulty pushes scores toward the >{rules['k1_hi_thresh']}%")
    print("    ceiling while theta nears its prox=0 ceiling (d+1), so the profile")
    print("    keeps crediting mastery the ability no longer backs.")

    # ------------------------------------------------------------------ [3]
    print("\n[3] the Feynman review, run as a flow (topic: tool-result capping)")
    m_f = 0.30
    feynman_log = {}
    for rnd, expl in (("r1", EXPL_1), ("r2", EXPL_2)):
        gaps = feynman_gaps(expl, texts)
        cl, ac, co, ov = feynman_score(gaps)
        lo, delta = feynman_delta(rules, ov)
        m_f = max(0.0, m_f + delta)
        feynman_log[rnd] = (ov, lo, delta)
        n_gaps = sum(len(v) for v in gaps.values())
        print(f"    {rnd}: gaps={n_gaps} | clarity={cl} accuracy={ac} "
              f"completeness={co} | overall={ov:.1f} -> band >={lo}: "
              f"delta={delta:+.2f} | mastery {m_f - delta:.2f} -> {m_f:.2f}")
        if rnd == "r1":
            for cat in rules["gap_categories"]:
                for g in gaps[cat]:
                    print(f"        [{cat}] {g}")
    gaps1 = feynman_gaps(EXPL_1, texts)
    assert all(len(gaps1[c]) >= 1 for c in rules["gap_categories"])
    assert "``tool_call_id``" in gaps1["Factual Errors"][0]

    # ------------------------------------------------------------------ [4]
    print("\n[4] anti-hallucination claims gate: provenance, not truth")
    claims = [
        dict(claim="a single tool result is capped at %d tokens" % rules["token_cap"],
             source="cap_middleware.py", sha=shas["cap_middleware.py"],
             quote="token_cap: int = %d" % rules["token_cap"]),
        dict(claim="the window is the memory", source=None, sha=None, quote=None),
        dict(claim="one turn stays pinned raw at the head", source="manager.py",
             sha="00000000", quote="pinned: int = 1"),
        dict(claim="the scroll keeps 7 turns pinned", source="manager.py",
             sha=shas["manager.py"], quote="pinned: int = 7"),
    ]
    verdicts = claims_gate(claims, texts, shas)
    for (claim, verdict, why), c in zip(verdicts, claims):
        note = "" if c["source"] is None else f" ({c['source']})"
        print(f"    {verdict:<15} \"{claim}\"{note} — {why}")
    print(f"    the NO-PROVENANCE claim is rejected with principle#5's maxim: "
          f"\"{rules['p5_maxim']}\"")
    assert [v[1] for v in verdicts] == ["VERIFIED", "NO-PROVENANCE",
                                        "SHA-DRIFT", "QUOTE-NOT-FOUND"]

    # ------------------------------------------------------------------ [5]
    print("\n[5] tool-result write-through (cap_middleware.py dimension)")
    tmp = tempfile.mkdtemp(prefix="nano_qwenpaw_L2_")
    store = ToolStore(str(Path(tmp) / "tools.db"))
    cap = rules["token_cap"]
    text = build_tool_dump(cap)
    tcid = "call_0001"
    res = cap_tool_result(store, tcid, text, cap)
    ptr_lines = res["in_context"][res["keep"]:]
    print(f"    tool output: {len(text)} chars = {res['n_tokens']} est-tokens "
          f"(cap={cap}) -> write-through keyed by {tcid}")
    print(f"    in-context: preview keep={res['keep']} chars + pointer "
          f"({est_tokens(ptr_lines)} est-tokens overhead)")
    print("    pointer (exact source format):")
    for ln in ptr_lines.splitlines()[1:]:
        print(f"      {ln}")
    rec = store.recall(tcid)
    same = rec == text
    print(f"    recall via ms.recall_tool({tcid!r}): {len(rec)} chars, "
          f"byte-identical={same}")
    assert same and res["capped"]

    store.conn.close()                     # simulate durable-store outage
    res_deg = cap_tool_result(store, tcid + "_deg", text, cap)
    print(f"    degradation path: store down -> real {sqlite3.ProgrammingError.__name__} "
          f"caught; capped={res_deg['capped']} degraded={res_deg['degraded']} "
          f"in_context==full output: {res_deg['in_context'] == text}")
    print("    (cap_middleware.py L63-68: don't truncate what we could not store)")
    assert res_deg["degraded"] and not res_deg["capped"]
    assert res_deg["in_context"] == text

    # ------------------------------------------------------------------ [6]
    print("\n[6] session ledger (SOUL.md principle#7 \"%s\", L%d) — real sqlite"
          % (rules["principles"][6][1], rules["p7_line"]))
    led = sqlite3.connect(str(Path(tmp) / "ledger.db"))
    led.execute("CREATE TABLE ledger (n INTEGER PRIMARY KEY, event TEXT, detail TEXT)")
    band_name = {4.5: "4.5-5.0", 3.5: "3.5-4.4", 2.5: "2.5-3.4", 0.0: "<2.5"}
    ov1, lo1, d1 = feynman_log["r1"]
    ov2, lo2, d2 = feynman_log["r2"]
    events = [
        ("examiner-gate", f"set#1 failures=4 > {rules['regen_thresh']} -> regenerate; "
                          f"set#2 fixed, {len(set2)}/{len(set2)} verified"),
        ("k1-adaptive", f"12 sessions: final mastery={ad[-1][4]:.2f} "
                        f"theta={ad[-1][5]:.3f} (best theta of 3 policies)"),
        ("feynman-r1", f"overall={ov1:.1f} -> band {band_name[lo1]}: delta={d1:+.2f}"),
        ("feynman-r2", f"overall={ov2:.1f} -> band {band_name[lo2]}: delta={d2:+.2f}"),
        ("claims-gate", "1 VERIFIED / 1 NO-PROVENANCE / 1 SHA-DRIFT / 1 QUOTE-NOT-FOUND"),
        ("tool-cap", f"{tcid}: {res['n_tokens']} tok -> preview {res['keep']} chars "
                     f"+ pointer; recall byte-identical"),
        ("tool-cap-degraded", "store down -> full output kept in context "
                              "(never truncate what we could not store)"),
    ]
    for i, (ev, det) in enumerate(events, 1):
        led.execute("INSERT INTO ledger VALUES (?, ?, ?)", (i, ev, det))
    led.commit()
    for n, ev, det in led.execute("SELECT * FROM ledger ORDER BY n"):
        print(f"    [{n}] {ev:<17} {det}")
    n_rows = led.execute("SELECT COUNT(*) FROM ledger").fetchone()[0]

    # ------------------------------------------------------------------ [7]
    print("\n[7] self-check (structural assertions)")
    checks = [
        ("SOUL.md parses to 7 principles; #5 is anti-hallucination",
         len(rules["principles"]) == 7
         and "Anti-Hallucination" in rules["principles"][4][1]),
        ("Examiner-B: 5 checks parsed from k-plus-one SKILL.md",
         rules["examiner_b_steps"] == 5),
        ("same-channel re-read catches strictly less than independent evidence",
         caught_r < caught_i and caught_i == 4),
        ("regenerate rule fires exactly when failures > parsed threshold",
         indep["n_bad"] > rules["regen_thresh"] and gate2b["n_bad"] == 0),
        ("K+1 loop: adaptive ends with the highest true theta",
         ad[-1][5] > fe[-1][5] > fh[-1][5]),
        ("fixed-easy inflates mastery above adaptive while learning less",
         fe[-1][4] > ad[-1][4] and fe[-1][5] < ad[-1][5]),
        ("fixed-hard deflates mastery to the floor (<50% rule, min 0)",
         fh[-1][4] == 0.0),
        ("adaptive keeps a majority of sessions in the 50-80% band",
         sum(1 for r in ad if 50.0 <= r[2] <= rules["k1_hi_thresh"]) >= 6),
        ("Feynman r1 flags all 4 gap categories; factual error cites live source",
         all(len(gaps1[c]) >= 1 for c in rules["gap_categories"])),
        ("Feynman bands applied: r1 no-change, r2 +0.1 (parsed, not hardcoded)",
         feynman_delta(rules, 3.0)[1] == 0.0
         and feynman_delta(rules, 5.0)[1] == rules["feynman_bands"][0][1]),
        ("claims gate: exactly one VERIFIED of four claims",
         sum(1 for v in verdicts if v[1] == "VERIFIED") == 1),
        ("write-through: recall is byte-identical to the capped output", same),
        ("degradation: store down -> no truncation, full output in context",
         res_deg["degraded"] and res_deg["in_context"] == text),
        ("ledger complete: 7 events persisted to real sqlite", n_rows == 7),
    ]
    for name, ok in checks:
        assert ok, name
        print(f"    PASS  {name}")
    print("    ✅ self-check passed")

    print("\n" + "=" * 68)
    print("takeaway: methodology is not prompt prose — it is executable flow.")
    print("  Rules parsed out of the coach files drive the loop: the adversarial")
    print("  gate catches what a same-channel re-read misses; the K+1 loop keeps")
    print("  difficulty chasing ability so the learning signal (wrong answers")
    print("  within reach) never dies — a frozen difficulty turns the score")
    print("  into an inflating proxy; the claims gate checks provenance, not")
    print("  truth; and capping never loses data — when it cannot store, it")
    print("  does not cap. Real hosted model behind the loops: [TODO: needs key]")
    print("=" * 68)

    led.close()


if __name__ == "__main__":
    main()
