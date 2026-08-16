#!/usr/bin/env python3
"""
nano-qwenpaw L1 — memory & context management under a finite window
====================================================================
L0 wrapped ONE task in system prompt + self-check, stateless across tasks.
L1 makes the harness live across MANY turns under a FINITE context window:
state grows without bound, the window does not — something must give.

The experiment measures three policies on the SAME real conversation:

  none         append-only; the model's hard window cut drops the head;
  summarize    when over budget, fold old turns into an extractive summary
               (real TF-IDF salience, lossy, irreversible);
  evict-index  the qwenpaw scroll way: write every turn through to a real
               sqlite store as it enters the window (write-through); when
               over budget, evict old turns from context but leave ONE
               in-context index line per eviction ([seq N] headline); the
               model recalls a seq span on demand. No summarization —
               nothing is lost (the full turns stay in the store).

Declarations (course runnability contract):
  * WindowModel is a DECLARED mock with exactly two properties:
      (a) a hard window cut — it sees only the last W estimated tokens;
      (b) extractive answers — it answers by picking the best-scoring
          sentence (TF-IDF cosine) from what it can see.
  * Recall agency is DECLARED too, mediated by the harness: a question
    overlapping an in-context index line (>= 2 content tokens) recalls
    that seq span; with no map hit, the fallback is an FTS5 query over
    the durable store — the qwenpaw REPL / ms.sql_query analog. In a real
    system the LLM's judgment sits here: [TODO: needs key].
  * Everything else is REAL: real source files read at runtime (sha256
    logged; pinned snapshot fallback if unavailable), a real sqlite3
    write-through store with an FTS5 recall index (the same storage
    technology as qwenpaw's history.db), real token bookkeeping under a
    declared estimator (ceil(chars/4) — qwenpaw asks the model's own
    count_tokens), and real extractive summarization.

Dependencies: Python stdlib only. Run:  python L1_real_memory_loop.py
Output is fully deterministic (seeded, no timing lines): two runs on the
same source snapshot are byte-identical.
"""

import hashlib
import math
import random
import re
import sqlite3
import sys
import tempfile
from pathlib import Path

sys.dont_write_bytecode = True

SEED = 42

# --------------------------------------------------------------------------
# token estimator (declared): ~4 chars per token
# --------------------------------------------------------------------------
def est_tokens(s: str) -> int:
    return -(-len(s) // 4)  # ceil(len/4)


STOP = {
    "the", "a", "an", "of", "in", "on", "at", "to", "for", "and", "or",
    "is", "are", "was", "be", "by", "with", "that", "this", "it", "as",
    "how", "many", "what", "which", "when", "where", "who", "does", "do",
    "can", "one", "two", "its", "their", "there", "than", "into", "about",
    "over", "from", "very", "most", "part", "parts", "use", "used", "using",
}


def stem(t: str) -> str:
    # minimal suffix stripping so "keyed"~"key", "turns"~"turn" (FTS5 side
    # uses a porter tokenizer; this keeps the two sides in agreement)
    for suf in ("ing", "ed", "es", "s"):
        if t.endswith(suf) and len(t) - len(suf) >= 3:
            return t[: -len(suf)]
    return t


def toks(s: str) -> list:
    return [stem(t) for t in re.findall(r"[a-z0-9_]+", s.lower())
            if t not in STOP]


def split_sentences(text: str) -> list:
    parts = re.split(r"(?<=[.!?。！？])\s+|\n+", text)
    return [p.strip() for p in parts if len(p.strip()) >= 8]


# --------------------------------------------------------------------------
# tiny TF-IDF cosine (real math, stdlib only)
# --------------------------------------------------------------------------
def build_idf(docs: list) -> dict:
    n = len(docs)
    df = {}
    for d in docs:
        for t in set(d):
            df[t] = df.get(t, 0) + 1
    return {t: math.log((n + 1) / (c + 1)) + 1.0 for t, c in df.items()}


def cosine(q: list, d: list, idf: dict) -> float:
    if not q or not d:
        return 0.0
    def vec(ts):
        v = {}
        for t in ts:
            w = idf.get(t, 1.0)
            v[t] = v.get(t, 0.0) + w
        return v
    vq, vd = vec(q), vec(d)
    num = sum(w * vd.get(t, 0.0) for t, w in vq.items())
    nq = math.sqrt(sum(w * w for w in vq.values()))
    nd = math.sqrt(sum(w * w for w in vd.values()))
    return num / (nq * nd) if nq and nd else 0.0


# --------------------------------------------------------------------------
# corpus: real files from the qwenpaw repo (facts) + LLM-PBL prose (padding)
# --------------------------------------------------------------------------
HERE = Path(__file__).resolve()
REPO_ROOT = next((p for p in HERE.parents if (p / "src/qwenpaw").is_dir()), HERE.parents[5])
PBL_ROOT = HERE.parents[4]

SOURCES = {
    "eviction_index.py": REPO_ROOT / "src/qwenpaw/agents/context/scroll/eviction_index.py",
    "cap_middleware.py": REPO_ROOT / "src/qwenpaw/agents/context/scroll/cap_middleware.py",
    "manager.py":        REPO_ROOT / "src/qwenpaw/agents/context/scroll/manager.py",
    "history.py":        REPO_ROOT / "src/qwenpaw/agents/context/scroll/history.py",
    "SOUL.md":           REPO_ROOT / "coach/profile/SOUL.md",
}
PADDING_SOURCES = [PBL_ROOT / "README.md", PBL_ROOT / "tutorial/material/README.md",
                   REPO_ROOT / "README.md"]

# pinned snapshot: verbatim slices captured 2026-08-06 (sha256 in tutorial
# §11). Used ONLY if a live read fails; the run prints which mode it used.
PINNED = {
    "eviction_index.py": (
        "_TIER_CAP = 10\n"
        "Nothing is lost — every line carries a ``seq`` span and the full "
        "turns stay in ``conversation_history``; a collapsed line is a "
        "zoomed-out view the model re-expands with one ``ms.sql_query`` "
        "over its span.\n"
    ),
    "cap_middleware.py": (
        "token_cap: int = 3000,\n"
        "the in-context content is replaced by a token-bounded preview plus "
        "a recall pointer keyed by ``tool_call_id``. this is the only "
        "capping path and it never loses data.\n"
    ),
    "manager.py": (
        "pinned: int = 1,\n"
        "past the token threshold, keep a pinned head + recent tail and "
        "fold the evicted middle into an in-context eviction index. no "
        "summarization, nothing lost — every node points to a ``seq`` span "
        "recallable via the sandboxed REPL.\n"
    ),
    "history.py": (
        "class HistoryStore:\n"
        "self._conn = sqlite3.connect(str(self._path))\n"
        "CREATE TABLE IF NOT EXISTS conversation_history (\n"
        "CREATE VIRTUAL TABLE IF NOT EXISTS conversation_history_fts "
        "USING fts5(content, content='conversation_history', "
        "content_rowid='seq', tokenize='porter unicode61')\n"
        "every live turn is persisted to the durable conversation_history "
        "as it enters the window (write-through).\n"
    ),
    "SOUL.md": (
        "seven numbered principles govern every interaction: K+1 learning, "
        "feynman technique, project-based learning, adversarial "
        "self-verification, anti-hallucination zero tolerance, learner "
        "autonomy, continuous improvement.\n"
        "principle 5: anti-hallucination — never fabricate facts, formulas, "
        "citations, or references.\n"
    ),
}


def sha8(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()[:8]


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


def build_facts(texts: dict) -> list:
    """Each fact: a study-note turn grounded in a real source line.

    detail_ans is an exact substring of the fact text; gist_kw is the phrase
    a gist question targets. Both sit in the fact sentence so one visible
    sentence answers both question types.
    """
    t = texts
    n_principles = len(re.findall(r"(?m)^## \d+\.", t["SOUL.md"])) or 7
    m = re.search(r"_TIER_CAP = (\d+)", t["eviction_index.py"])
    tier_cap = m.group(1) if m else "10"
    m = re.search(r"token_cap: int = (\d+)", t["cap_middleware.py"])
    token_cap = m.group(1) if m else "3000"
    m = re.search(r"pinned: int = (\d+)", t["manager.py"])
    pinned = m.group(1) if m else "1"

    facts = [
        dict(file="eviction_index.py",
             text=f"eviction_index.py: `_TIER_CAP = {tier_cap}` — each tier of the "
                  f"eviction index holds at most {tier_cap} blocks before it carries up.",
             dq="how many blocks can one tier of the eviction index hold?",
             detail_ans=f"_TIER_CAP = {tier_cap}",
             gq="what happens when a tier of the eviction index fills up?",
             gist_kw="carries up"),
        dict(file="eviction_index.py",
             text="eviction_index.py: nothing is lost — every index line carries a "
                  "seq span and the full turns stay in conversation_history.",
             dq="where do the full evicted turns stay?",
             detail_ans="the full turns stay in conversation_history",
             gq="is anything lost when turns are folded into the index?",
             gist_kw="nothing is lost"),
        dict(file="cap_middleware.py",
             text=f"cap_middleware.py: `token_cap: int = {token_cap}` — a single tool "
                  f"result is capped at {token_cap} tokens in context; the full output "
                  "is written through and replaced by a preview plus a recall pointer.",
             dq="what is the default token_cap for one tool result?",
             detail_ans=f"token_cap: int = {token_cap}",
             gq="what replaces an oversized tool result in context?",
             gist_kw="recall pointer"),
        dict(file="cap_middleware.py",
             text="cap_middleware.py: the recall pointer is keyed by tool_call_id, "
                  "so capping never loses data.",
             dq="what key does the recall pointer use?",
             detail_ans="tool_call_id",
             gq="does capping ever lose data, given the pointer's key?",
             gist_kw="never loses data"),
        dict(file="manager.py",
             text=f"manager.py: `pinned: int = {pinned}` — {pinned} turn stays pinned "
                  "raw at the head of the window.",
             dq="how many turns stay pinned raw at the head?",
             detail_ans=f"pinned: int = {pinned}",
             gq="which part of the conversation stays raw at the head?",
             gist_kw="pinned"),
        dict(file="manager.py",
             text="manager.py: past the threshold, keep a pinned head plus a recent "
                  "tail and fold the evicted middle into an in-context index; no "
                  "summarization, nothing lost.",
             dq="what does the scroll manager keep raw past the token threshold?",
             detail_ans="pinned head plus a recent",
             gq="does the manager summarize the evicted middle, or fold it?",
             gist_kw="no summarization"),
        dict(file="history.py",
             text="history.py: HistoryStore owns a sqlite3 connection; every turn is "
                  "a row in the conversation_history table, persisted as it enters "
                  "the window (write-through).",
             dq="HistoryStore owns a connection to which storage engine?",
             detail_ans="sqlite3",
             gq="when is a turn persisted to the durable store?",
             gist_kw="write-through"),
        dict(file="history.py",
             text="history.py: recall runs over an fts5 virtual table with a porter "
                  "unicode61 tokenizer.",
             dq="which virtual table extension does recall run over?",
             detail_ans="fts5",
             gq="which tokenizer does the recall index use?",
             gist_kw="porter"),
        dict(file="SOUL.md",
             text=f"SOUL.md: {n_principles} numbered principles govern every "
                  "interaction, from K+1 learning to continuous improvement.",
             dq="how many numbered principles does SOUL.md define?",
             detail_ans=f"{n_principles} numbered principles",
             gq="which two endpoints name the principles' range, from K+1 learning onward?",
             gist_kw="k+1 learning"),
        dict(file="SOUL.md",
             text="SOUL.md principle 5: anti-hallucination, zero tolerance — never "
                  "fabricate facts, formulas, citations, or references.",
             dq="which principle number is anti-hallucination?",
             detail_ans="principle 5",
             gq="what does anti-hallucination forbid?",
             gist_kw="never fabricate"),
    ]
    for i, f in enumerate(facts):
        f["id"] = i + 1
        # ground-truth guard: the answer must literally sit in the fact text
        assert f["detail_ans"].lower() in f["text"].lower(), f"fact {i+1} broken"
        assert f["gist_kw"].lower() in f["text"].lower(), f"fact {i+1} broken"
    return facts


def build_padding(rng: random.Random, facts: list) -> list:
    """Real prose paragraphs (LLM-PBL README/learning navigation). Purity rule: no
    padding paragraph may contain a fact's answer/keyword PHRASE verbatim
    (token-level overlap is TF-IDF's designed behavior, not leakage)."""
    pool = []
    for path in PADDING_SOURCES:
        try:
            text = path.read_text()
        except OSError:
            continue
        for para in re.split(r"\n\s*\n", text):
            para = para.strip()
            if len(para) < 200 or para.startswith("|") or para.startswith("#"):
                continue
            # prose only: code blocks / directory trees are not paragraphs
            # (their rare ascii lines hijack salience ranking — a real trap)
            if any(c in para for c in "├└│▼```"):
                continue
            if para.startswith(">"):
                para = para.replace("> ", "")
            pool.append(para)
    clean = []
    for p in pool:
        low = p.lower()
        if any(f["detail_ans"].lower() in low or f["gist_kw"].lower() in low
               for f in facts):
            continue
        clean.append(p)
    assert len(clean) >= 14, f"padding pool too small: {len(clean)}"
    return [clean[i] for i in rng.sample(range(len(clean)), 14)]


# --------------------------------------------------------------------------
# WindowModel — declared mock: hard window cut + extractive answer
# --------------------------------------------------------------------------
ANSWER_THRESH = 0.10


class WindowModel:
    def __init__(self, window_tokens: int):
        self.window_tokens = window_tokens

    def visible(self, prompt: str) -> str:
        budget = self.window_tokens * 4  # chars, per the declared estimator
        return prompt[-budget:] if len(prompt) > budget else prompt

    def answer(self, prompt: str, question: str):
        """Extractive: best-scoring visible CONTENT sentence, or UNKNOWN.
        Returns (kind, sentence, score). The trailing 'question: ...' line
        is an instruction, not context; harness structural lines (index
        lines, section headers) are not content either."""
        marker = "\nquestion: "
        ctx = prompt.rsplit(marker, 1)[0] if marker in prompt else prompt
        structural = ("[seq ", "evicted turns (recall", "summary so far",
                      "recalled seq", "system:", "question:")
        sents = [s for s in split_sentences(self.visible(ctx))
                 if not s.lower().startswith(structural)]
        if not sents:
            return ("UNKNOWN", None, 0.0)
        doc_toks = [toks(s) for s in sents]
        idf = build_idf(doc_toks)
        qt = toks(question)
        scored = [(cosine(qt, dt, idf), s) for dt, s in zip(doc_toks, sents)]
        score, sent = max(scored)
        if score < ANSWER_THRESH:
            return ("UNKNOWN", None, score)
        return ("ANSWER", sent, score)


# --------------------------------------------------------------------------
# HistoryStore — real sqlite3, write-through, FTS5 recall (qwenpaw pattern:
# history.py conversation_history + conversation_history_fts)
# --------------------------------------------------------------------------
class HistoryStore:
    def __init__(self, path: str):
        self.conn = sqlite3.connect(path)
        self.conn.execute(
            "CREATE TABLE conversation_history ("
            "seq INTEGER PRIMARY KEY, role TEXT, content TEXT)")
        self.conn.execute(
            "CREATE VIRTUAL TABLE conversation_history_fts USING fts5("
            "content, content='conversation_history', content_rowid='seq', "
            "tokenize='porter unicode61')")
        self.n = 0

    def put(self, role: str, content: str) -> int:
        self.n += 1
        self.conn.execute(
            "INSERT INTO conversation_history VALUES (?, ?, ?)",
            (self.n, role, content))
        self.conn.execute(
            "INSERT INTO conversation_history_fts(rowid, content) VALUES (?, ?)",
            (self.n, content))
        return self.n

    def span(self, lo: int, hi: int) -> list:
        return self.conn.execute(
            "SELECT seq, content FROM conversation_history "
            "WHERE seq BETWEEN ? AND ? ORDER BY seq", (lo, hi)).fetchall()

    def match(self, query: str, k: int = 8) -> list:
        rows = self.conn.execute(
            "SELECT rowid FROM conversation_history_fts "
            "WHERE conversation_history_fts MATCH ? LIMIT ?", (query, k)).fetchall()
        return [r[0] for r in rows]

    def content_bytes(self) -> int:
        return self.conn.execute(
            "SELECT COALESCE(SUM(LENGTH(content)), 0) FROM conversation_history"
        ).fetchone()[0]


# --------------------------------------------------------------------------
# the three policies
# --------------------------------------------------------------------------
SYSTEM = ("system: You are a study assistant. Answer strictly from the "
          "conversation context; if the context lacks the answer, say UNKNOWN.")

WINDOW = 1000   # model hard cut (estimated tokens)
BUDGET = 1000   # harness budget for managed policies — SAME budget as the
                # window, so the comparison isolates the policy, not budget
IDX_CAP = 8     # index lines before the oldest two collapse into a span line
SUM_KEEP_TURNS = 3  # whole turns kept by the extractive summary


def headline_of(text: str, fact=None) -> str:
    # fact headlines keep enough tokens to stay addressable (qwenpaw Leaf
    # carries a milestone headline); padding headlines are short by design.
    return fact["text"][:120] if fact is not None else text[:48].replace("\n", " ")


def index_text(lines: list) -> str:
    out = []
    for ln in lines:
        if ln["seq_lo"] == ln["seq_hi"]:
            span, head = f"{ln['seq_lo']}", ln["head"]
        else:
            # span line shows endpoints only (qwenpaw Line: head/tail)
            span = f"{ln['seq_lo']}-{ln['seq_hi']}"
            head = ln["head"][:60] + " ... " + ln["tail"][:60]
        out.append(f"[seq {span}] {head}")
    return "\n".join(out)


def run_none(turns: list, model: WindowModel):
    """Append-only: prompt grows; the model's window cut drops the head."""
    transcript = SYSTEM + "\n" + "\n".join(t["text"] for t in turns)

    def ask(q):
        return model.answer(transcript + "\nquestion: " + q, q)
    return ask, {"final_ctx_tokens": est_tokens(transcript)}


def run_summarize(turns: list, model: WindowModel):
    """Over budget -> fold the oldest half; the summary keeps the top-K
    most salient TURNS (salience = mean IDF of a turn's tokens, IDF over
    candidate turns). Whole turns survive or vanish — which facts survive
    is decided by token rarity, not by importance."""
    summary_turns = []
    ctx = []
    compressions = 0
    folded_tokens = 0

    def total_tokens():
        s = SYSTEM + "\n"
        if summary_turns:
            s += ("summary so far (kept turns):\n"
                  + "\n".join(t["text"] for t in summary_turns) + "\n")
        s += "\n".join(t["text"] for t in ctx)
        return est_tokens(s), s

    for t in turns:
        ctx.append(t)
        tok, _ = total_tokens()
        while tok > BUDGET and len(ctx) > 1:
            half = max(1, len(ctx) // 2)
            folded, ctx = ctx[:half], ctx[half:]
            folded_tokens += sum(est_tokens(x["text"]) for x in folded)
            pool = summary_turns + folded
            doc_toks = [toks(x["text"]) for x in pool]
            idf = build_idf(doc_toks)
            scores = []
            for dt in doc_toks:
                sc = sum(idf.get(t2, 1.0) for t2 in dt) / len(dt) if dt else 0.0
                scores.append(sc)
            order = sorted(range(len(pool)), key=lambda i: (-scores[i], i))
            summary_turns = [pool[i] for i in sorted(order[:SUM_KEEP_TURNS])]
            compressions += 1
            tok, _ = total_tokens()

    _, prompt_head = total_tokens()

    def ask(q):
        return model.answer(prompt_head + "\nquestion: " + q, q)
    return ask, {
        "final_ctx_tokens": est_tokens(prompt_head),
        "compressions": compressions,
        "summary_tokens": est_tokens("\n".join(t["text"] for t in summary_turns)),
        "folded_tokens": folded_tokens,
        "kept_turns": [t.get("fact", {}).get("id", "-") for t in summary_turns],
    }


def run_evict_index(turns: list, model: WindowModel, store: HistoryStore):
    """Write-through + eviction index + recall (the qwenpaw scroll way)."""
    pinned_turn = None
    tail = []
    index = []
    seq_of_turn = {}
    recalls = {"map": 0, "fts": 0, "both": 0}

    def ctx_tokens():
        s = SYSTEM + "\n"
        if pinned_turn:
            s += pinned_turn["text"] + "\n"
        if index:
            s += "evicted turns (recall by seq):\n" + index_text(index) + "\n"
        s += "\n".join(t["text"] for t in tail)
        return est_tokens(s), s

    for t in turns:
        seq = store.put("turn", t["text"])     # write-through ON ENTRY
        seq_of_turn[id(t)] = seq
        if pinned_turn is None:
            pinned_turn = t                     # pinned: int = 1
            continue
        tail.append(t)
        tok, _ = ctx_tokens()
        while tok > BUDGET and tail:
            old = tail.pop(0)
            index.append({
                "seq_lo": seq_of_turn[id(old)],
                "seq_hi": seq_of_turn[id(old)],
                "head": headline_of(old["text"], old.get("fact")),
                "tail": headline_of(old["text"], old.get("fact")),
            })
            if len(index) > IDX_CAP:
                # collapse oldest two into a span line carrying ENDPOINTS only
                # (qwenpaw Line: head/tail of the span) — bounded by design
                a, b = index[0], index[1]
                index = [{"seq_lo": a["seq_lo"], "seq_hi": b["seq_hi"],
                          "head": a["head"], "tail": b["tail"]}] + index[2:]
            tok, _ = ctx_tokens()

    _, prompt_head = ctx_tokens()

    def ask(q):
        # declared recall agency, mediated by the harness: map candidate
        # (index-line overlap) and store candidate (FTS5 content overlap —
        # the qwenpaw REPL / ms.sql_query analog). Content evidence is a
        # superset of headline evidence, so when BOTH fire we fetch the
        # union in one round-trip; telemetry records which source(s)
        # contributed. (Declared boundary: this agency does not know what
        # the tail already shows, so in-tail facts also pay one recall —
        # part of the measured cost.)
        qt = set(toks(q))
        map_ln, map_ov = None, 1
        for ln in index:
            visible_headline = (ln["head"] if ln["seq_lo"] == ln["seq_hi"]
                                else ln["head"] + " " + ln["tail"])
            ov = len(qt & set(toks(visible_headline)))
            if ov > map_ov or (ov == map_ov and map_ln is not None
                               and ln["seq_lo"] < map_ln["seq_lo"]):
                map_ln, map_ov = ln, ov
        fts_seq, fts_ov = None, 1
        query = " OR ".join(sorted(t for t in qt if len(t) >= 3))
        for s in (store.match(query) if query else []):
            ov = len(qt & set(toks(store.span(s, s)[0][1])))
            if ov > fts_ov or (ov == fts_ov and fts_seq is not None
                               and s < fts_seq):
                fts_seq, fts_ov = s, ov
        if map_ln is None and fts_seq is None:
            return model.answer(prompt_head + "\nquestion: " + q, q)
        if map_ln is not None and fts_seq is not None:
            recalls["both"] += 1
        elif map_ln is not None:
            recalls["map"] += 1
        else:
            recalls["fts"] += 1
        wanted = set()
        if map_ln is not None:
            wanted.update(range(map_ln["seq_lo"], map_ln["seq_hi"] + 1))
        if fts_seq is not None:
            wanted.add(fts_seq)
        rows = store.span(min(wanted), max(wanted))
        rows = [r for r in rows if r[0] in wanted]
        lo, hi = rows[0][0], rows[-1][0]
        expanded = (prompt_head + f"\nrecalled seq {lo}-{hi}:\n"
                    + "\n".join(c for _, c in rows))
        return model.answer(expanded + "\nquestion: " + q, q)

    return ask, {
        "final_ctx_tokens": est_tokens(prompt_head),
        "index_lines": len(index),
        "seq_of_turn": seq_of_turn,
        # live reference: read AFTER evaluate() has asked the questions
        "recalls": recalls,
    }


def evaluate(ask, facts: list) -> dict:
    detail_hits, gist_hits, per_fact = 0, 0, []
    for f in facts:
        kind, sent, _ = ask(f["dq"])
        d_ok = kind == "ANSWER" and sent and f["detail_ans"].lower() in sent.lower()
        kind, sent, _ = ask(f["gq"])
        g_ok = kind == "ANSWER" and sent and f["gist_kw"].lower() in sent.lower()
        detail_hits += int(d_ok)
        gist_hits += int(g_ok)
        per_fact.append(d_ok)
    return {
        "detail": detail_hits / len(facts),
        "gist": gist_hits / len(facts),
        "per_fact": per_fact,
    }


# --------------------------------------------------------------------------
def main():
    print("=" * 68)
    print("nano-qwenpaw L1 — memory & context management, measured")
    print("=" * 68)
    print(f"python {sys.version.split()[0]}")
    print("declarations: WindowModel = declared mock (hard window cut +")
    print("  extractive answer); recall agency declared (index map, FTS")
    print("  fallback); corpus/store/tokens/summary = real (files+sha256,")
    print("  sqlite3+FTS5, declared estimator, TF-IDF). Real hosted model")
    print("  behind the loop: [TODO: needs key]")

    rng = random.Random(SEED)
    texts, shas, mode = load_sources()
    facts = build_facts(texts)
    padding = build_padding(rng, facts)

    print("\n[0] corpus: real sources -> facts + padding")
    for i, name in enumerate(SOURCES):
        print(f"    {name:<20} sha256[:8]={shas[name]}  mode={mode[i]}")
    print(f"    facts={len(facts)} (detail+gist Q each) | "
          f"padding paragraphs={len(padding)}")

    # conversation: 24 turns = fact/padding interleaved + padding tail
    turns = []
    fi = 0
    for i in range(20):
        if i % 2 == 1:
            f = facts[fi]
            turns.append({"text": f"study note {f['id']}: {f['text']}", "fact": f})
            fi += 1
        else:
            turns.append({"text": f"reading note: {padding[i // 2]}"})
    for i in range(4):
        turns.append({"text": f"reading note: {padding[10 + i]}"})
    total_tokens = est_tokens(SYSTEM + "\n" + "\n".join(t["text"] for t in turns))
    print(f"    conversation: {len(turns)} turns | transcript ~{total_tokens} est-tokens"
          f" | model window={WINDOW} | harness budget={BUDGET}")
    assert total_tokens > WINDOW, "construction must overflow the window"

    # padding purity: pressure must be volume, not verbatim answers
    for t in turns:
        if "fact" not in t:
            low = t["text"].lower()
            assert not any(f["detail_ans"].lower() in low
                           or f["gist_kw"].lower() in low for f in facts), \
                "padding leaked a fact phrase"

    # ------------------------------------------------------------------ [1]
    print("\n[1] append-only (none): recall of fact#1 as the conversation grows")
    curve = []
    for t in range(2, len(turns) + 1):   # fact#1 enters at turn 2
        ask, _ = run_none(turns[:t], WindowModel(WINDOW))
        kind, sent, _ = ask(facts[0]["dq"])
        ok = kind == "ANSWER" and sent and facts[0]["detail_ans"].lower() in sent.lower()
        curve.append(int(ok))
    first_zero = curve.index(0) if 0 in curve else -1
    print("    recall curve (probe fact#1 after each turn once it entered): "
          + "".join(map(str, curve)))
    print(f"    fact#1 falls out of the window after turn {first_zero + 2}"
          f" (window={WINDOW} est-tokens)")
    assert curve[0] == 1
    assert all(curve[i] >= curve[i + 1] for i in range(len(curve) - 1))
    assert curve[-1] == 0 and first_zero > 0

    # ------------------------------------------------------------------ [2]
    print("\n[2] three policies, same 24-turn conversation, 20 probe questions")
    results = {}

    ask, stats = run_none(turns, WindowModel(WINDOW))
    results["none"] = evaluate(ask, facts)
    results["none"].update(stats)

    ask, stats = run_summarize(turns, WindowModel(WINDOW))
    results["summarize"] = evaluate(ask, facts)
    results["summarize"].update(stats)

    tmp = tempfile.mkdtemp(prefix="nano_qwenpaw_L1_")
    store = HistoryStore(str(Path(tmp) / "history.db"))
    ask, stats = run_evict_index(turns, WindowModel(WINDOW), store)
    results["evict-index"] = evaluate(ask, facts)
    results["evict-index"].update(stats)
    results["evict-index"]["store_rows"] = store.n
    results["evict-index"]["store_bytes"] = store.content_bytes()

    print(f"    {'policy':<12} {'detail':>7} {'gist':>7} {'ctx_tok':>8}  extras")
    for name in ("none", "summarize", "evict-index"):
        r = results[name]
        if name == "none":
            extras = f"transcript overflows window={WINDOW} -> head dropped"
        elif name == "summarize":
            extras = (f"compressions={r['compressions']} "
                      f"kept_turns={r['kept_turns']}")
        else:
            rc = r["recalls"]
            extras = (f"recalls map/fts/both={rc['map']}/{rc['fts']}/{rc['both']} "
                      f"index_lines={r['index_lines']} "
                      f"store={r['store_rows']}rows/{r['store_bytes']}B")
        print(f"    {name:<12} {r['detail']:>7.1%} {r['gist']:>7.1%} "
              f"{r['final_ctx_tokens']:>8}  {extras}")

    # ------------------------------------------------------------------ [3]
    print("\n[3] where the losses sit (per-fact detail recall under each policy)")
    print("    fact   none  summarize  evict-index   source")
    for i, f in enumerate(facts):
        row = []
        for name in ("none", "summarize", "evict-index"):
            row.append("hit" if results[name]["per_fact"][i] else "MISSED")
        print(f"    F{i + 1:<3} {row[0]:>6}  {row[1]:>9}  {row[2]:>11}   {f['file']}")
    s = results["summarize"]
    ratio = s["summary_tokens"] / max(1, s["folded_tokens"])
    print(f"    summarize: summary {s['summary_tokens']} tok <- folded "
          f"{s['folded_tokens']} tok (ratio {ratio:.2f}, lossy & irreversible)")

    # ------------------------------------------------------------------ [4]
    print("\n[4] the qwenpaw invariant, checked on real storage")
    ok_lost = store.n == len(turns)
    print(f"    write-through: store rows == turns entered: "
          f"{store.n} == {len(turns)} -> {ok_lost}")
    seqs = store.match("tier cap blocks")
    f1_seq = results["evict-index"]["seq_of_turn"][id(turns[1])]
    print(f"    FTS5 recall: MATCH 'tier cap blocks' -> seq {seqs} "
          f"(fact#1 seq = {f1_seq})")
    assert ok_lost and seqs and seqs[0] == f1_seq

    # ------------------------------------------------------------------ [5]
    print("\n[5] self-check (structural assertions)")
    n, sm, ev = results["none"], results["summarize"], results["evict-index"]
    checks = [
        ("append-only: early facts lost, late facts visible",
         0.0 < n["detail"] < 1.0),
        ("append-only: per-fact recall monotone in recency",
         all(n["per_fact"][i] <= n["per_fact"][i + 1]
             for i in range(len(facts) - 1))),
        ("summarize: lossy — detail recall below evict-index",
         sm["detail"] < ev["detail"]),
        ("summarize: losses NOT recency-monotone (unpredictable, unlike none)",
         not all(sm["per_fact"][i] <= sm["per_fact"][i + 1]
                 for i in range(len(facts) - 1))),
        ("summarize: salience kept padding over facts (failure mode measured)",
         "-" in sm["kept_turns"] and any(isinstance(k, int)
                                         for k in sm["kept_turns"])),
        ("evict-index: detail recall == 100% (nothing lost)",
         ev["detail"] == 1.0),
        ("evict-index: gist recall == 100%", ev["gist"] == 1.0),
        ("evict-index: final context within budget",
         ev["final_ctx_tokens"] <= BUDGET),
        ("evict-index: recall agency actually fired",
         sum(ev["recalls"].values()) > 0),
        ("evict-index: store complete (write-through)",
         store.n == len(turns)),
        ("summarize: compression is lossy (summary < folded)",
         s["summary_tokens"] < s["folded_tokens"]),
    ]
    for name, ok in checks:
        assert ok, name
        print(f"    PASS  {name}")
    print("    ✅ self-check passed")

    print("\n" + "=" * 68)
    print("takeaway: the window is a cache, not the memory. Append-only makes")
    print("  the cache the whole truth and silently loses the head; summary")
    print("  trades detail for space and cannot be undone; write-through +")
    print("  eviction index + recall keeps context bounded while the store")
    print("  stays complete — memory reliability is a storage design, not a")
    print("  model property. (qwenpaw scroll: manager.py / eviction_index.py")
    print("  / history.py; real hosted model [TODO: needs key])")
    print("=" * 68)

    store.conn.close()


if __name__ == "__main__":
    main()
