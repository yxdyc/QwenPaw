"""
nano-agentscope L2 — planner + executor: message contracts & termination
==========================================================================
L1 put a real (tiny) model inside a SINGLE-agent loop and measured how the
harness buys reliability. L2 zooms out one level: when "one agent" is
decomposed into planner + executor, new failure modes appear that have
nothing to do with model intelligence — they are properties of the PROTOCOL
between agents. L2 isolates exactly that layer:

  [0] a typed message contract (who speaks, what each kind must carry),
      enforced at the boundary — not by sender goodwill;
  [1] real tools, L1 discipline: frozen list_dir + sandboxed read_file;
  [2] planner + executor as declared rule-based agents (NOT models);
  [3] an orchestrator whose termination conditions are explicit DATA:
      answered / aborted / no_progress (livelock guard) /
      replans_exhausted / budget_exhausted — failure is a produced state,
      never an exception;
  [4] experiments: happy path / contract violation at the boundary /
      livelock with and without the progress guard / budget sweep /
      failure isolation + replan + the T4 backstop;
  [5] the coordination ledger: what decomposition costs (messages) and what
      it buys (validated crossings, isolated failures, liveness).

Dependencies: Python standard library only. Run:  python L2_planner_executor.py

Declarations (course runnability contract):
  * PlannerAgent / ExecutorAgent / StuckExecutor / CorruptingExecutor /
    RecklessPlanner are declared rule-based test agents, NOT models — the
    same test-vector discipline as L1's Playback/FaultModel. Every failure
    below is a PROTOCOL failure by construction; that is the point. Putting
    a real model behind the planner is L3's job.
  * L2 deliberately does NOT import L1: L1's TinyReActLM memorizes one fixed
    task prefix and cannot follow planner-composed subtask prompts, so
    importing it would add torch + ~2 min of training and teach nothing true
    about coordination (tutorial_L2 §4). Tools are re-implemented under the
    same discipline instead.
  * list_dir is FROZEN at the L2 finalization moment (eight files), same
    reproducibility hardening as L1: the observation
    feeds the fallback heuristic and the printed trace, so freezing decouples
    this level's anchors from future ladder growth (L3 files).
  * There is NO randomness anywhere in L2: outputs are byte-identical
    across runs.
"""

import ast
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))

TASK = ("Read corpus.txt in this directory and answer: according to the "
        "title line of that file, what two things does ReAct synergize?")
TASK_NOTES = ("Read notes.txt in this directory and answer: according to "
              "the title line of that file, what two things does ReAct "
              "synergize?")
EXPECTED_ANSWER = "reasoning and acting"


# ===========================================================================
# [A] real tools — L1 discipline: frozen listing + sandboxed live reads
# ===========================================================================

class Tool:
    """name + callable + arg schema (same shape as L0/L1)."""

    def __init__(self, name, fn, schema):
        self.name, self.fn, self.schema = name, fn, schema

    def run(self, kwargs):
        missing = [k for k in self.schema if k not in kwargs]
        if missing:
            raise ValueError(f"missing args for {self.name}: {missing}")
        return str(self.fn(**kwargs))


def list_dir() -> str:
    """Module directory listing, FROZEN at the L2 finalization moment
    (eight files). Same reproducibility hardening as L1's list_dir
    (independently checked): this observation feeds the printed
    trace AND the planner's fallback heuristic [4e], so freezing it
    decouples L2's deterministic anchors from directory state — the ladder
    will grow again at L3. read_file below stays live."""
    return str(["L0_react_loop.py", "L1_real_agent_loop.py",
                "L2_planner_executor.py", "README.md", "corpus.txt",
                "tutorial_L0.md", "tutorial_L1.md", "tutorial_L2.md"])


def read_file(path: str, max_chars: int = 300) -> str:
    """Real file read, head-style, realpath-sandboxed to the module
    directory — verbatim discipline of L1 [A] (least privilege)."""
    root = os.path.realpath(HERE)
    target = os.path.realpath(os.path.join(HERE, path))
    if not (target == root or target.startswith(root + os.sep)):
        raise PermissionError(f"sandbox blocked path outside module: {path!r}")
    if not os.path.isfile(target):
        raise FileNotFoundError(f"no such file in module dir: {path!r}")
    with open(target, encoding="utf-8") as f:
        content = f.read()
    if len(content) > max_chars:
        return content[:max_chars] + "\n[... truncated]"
    return content


TOOLS = {
    "list_dir": Tool("list_dir", list_dir, {}),
    "read_file": Tool("read_file", read_file, {"path": "str"}),
}


# ===========================================================================
# [B] the message contract — typed kinds, validated at the boundary
# ===========================================================================

ROLES = ("planner", "executor")

# kind -> required metadata fields. Every crossing must carry these.
CONTRACT = {
    "plan":    ("plan_id", "steps"),
    "subtask": ("plan_id", "subtask_id", "tool", "args"),
    "result":  ("plan_id", "subtask_id", "status"),
    "clarify": ("plan_id", "subtask_id"),
    "answer":  ("plan_id",),
    "abort":   ("plan_id", "reason"),
}


def msg(name, role, content, **metadata):
    """The nano Msg: four fields, mirroring AgentScope's Msg split between
    context (name/role/content) and control (metadata)."""
    return {"name": name, "role": role, "content": content,
            "metadata": dict(metadata)}


def validate_msg(m):
    """Boundary enforcement. Returns ('ok', None) or (violation, detail)
    with TYPED violation kinds — L1's parse_block idea, lifted from
    single-block parsing to inter-agent crossings:
      bad_shape / bad_role / bad_kind / missing_field / bad_status
    A rejected message never touches the shared log or any agent's state."""
    if not isinstance(m, dict):
        return "bad_shape", "message must be a dict"
    for f in ("name", "role", "content", "metadata"):
        if f not in m:
            return "bad_shape", f"missing top-level field {f!r}"
    if m["role"] not in ROLES:
        return "bad_role", f"role {m['role']!r} not in {list(ROLES)}"
    md = m["metadata"]
    if not isinstance(md, dict) or "kind" not in md:
        return "bad_shape", "metadata.kind is required"
    kind = md["kind"]
    if kind not in CONTRACT:
        return "bad_kind", f"unknown kind {kind!r}"
    for f in CONTRACT[kind]:
        if f not in md:
            return "missing_field", f"{kind} requires metadata[{f!r}]"
    if kind == "result" and md["status"] not in ("ok", "error"):
        return "bad_status", f"status {md['status']!r} not in ['ok', 'error']"
    if not isinstance(m["content"], str) or not m["content"]:
        return "bad_shape", "content must be a non-empty string"
    return "ok", None


def brief(m):
    """One-line rendering of an accepted message (for traces)."""
    k = m["metadata"]["kind"]
    if k == "subtask":
        md = m["metadata"]
        tag = " [replan]" if md.get("replan") else ""
        return f"{md['subtask_id']} {md['tool']} {json.dumps(md['args'])}{tag}"
    if k == "result":
        md = m["metadata"]
        return f"{md['status']} {md['subtask_id']} <- {m['content'][:56]}"
    return m["content"][:72]


# ===========================================================================
# [C] declared rule-based agents (NOT models — see module declarations)
# ===========================================================================

class PlannerAgent:
    """Declared rule-based planner. Policy, explicit and naive ON PURPOSE
    where noted — L2 stress-tests the protocol, not the planner's wit:
      P1 decompose: list_dir -> read_file(<believed file>); the answer is
         extracted locally from the read result (real string computation);
      P2 dispatch one subtask at a time, in order;
      P3 on result(ok): advance; when the read result is in, answer;
      P4 on result(error): if replan budget remains, fall back to the first
         '*.txt' named in the list_dir observation that has not already
         failed (declared heuristic, real parse of real tool output);
         otherwise emit abort — give-up as a produced state;
      P5 on clarify: re-dispatch the same subtask (naive on purpose —
         liveness must be owned by the orchestrator's guard, see [4c]);
      P6 on critique of an own message: re-send it (the planner is
         well-behaved; only executors are corrupted in these experiments).
    """

    def __init__(self, task=TASK, believed_file="corpus.txt",
                 replan_budget=1, reckless=False):
        self.task, self.believed_file = task, believed_file
        self.replan_budget, self.reckless = replan_budget, reckless
        self.steps, self.next_i = [], 0
        self.results, self.listing = {}, None
        self.failed_paths, self.replans = set(), 0
        self.last_msg, self.guess_n = None, 1

    def _plan(self):
        self.steps = [
            {"subtask_id": "s1", "tool": "list_dir", "args": {}},
            {"subtask_id": "s2", "tool": "read_file",
             "args": {"path": self.believed_file}},
        ]
        text = (f"{len(self.steps)} steps: s1=list_dir, "
                f"s2=read_file({self.believed_file})")
        return self._emit("plan", text, plan_id="p1", steps=list(self.steps))

    def _emit(self, kind, content, **md):
        self.last_msg = msg("planner", "planner", content, kind=kind, **md)
        return self.last_msg

    def _dispatch(self, step, replan=False):
        md = dict(plan_id="p1", subtask_id=step["subtask_id"],
                  tool=step["tool"], args=step["args"])
        if replan:
            md["replan"] = True
        text = f"do {step['subtask_id']}: {step['tool']} {json.dumps(step['args'])}"
        return self._emit("subtask", text, **md)

    def _fallback_candidate(self):
        if self.reckless:                      # declared wrong guesses [4e]
            self.guess_n += 1
            return f"corpus{self.guess_n}.txt"
        for name in self.listing or []:
            if name.endswith(".txt") and name not in self.failed_paths:
                return name
        return None

    def compose(self, pending):
        kind, payload = pending
        if kind == "start":
            return self._plan()
        if kind == "critique":                 # P6
            return self.last_msg
        if kind == "dispatch_next":
            return self._dispatch(self.steps[self.next_i])
        if kind == "clarify":                  # P5 (naive on purpose)
            sid = payload["metadata"]["subtask_id"]
            step = next((s for s in self.steps if s["subtask_id"] == sid),
                        {"subtask_id": sid, "tool": "read_file",
                         "args": {"path": self.believed_file}})
            return self._dispatch(step)
        if kind == "result":                   # P3 / P4
            md = payload["metadata"]
            sid, status = md["subtask_id"], md["status"]
            if status == "ok":
                self.results[sid] = payload["content"]
                if sid == "s1":
                    self.listing = ast.literal_eval(payload["content"])
                if sid in {s["subtask_id"] for s in self.steps
                           if s["tool"] == "read_file"}:
                    answer = extract_answer(payload["content"])
                    return self._emit("answer", answer, plan_id="p1")
                self.next_i += 1
                return self._dispatch(self.steps[self.next_i])
            # error result:
            failed = next((s for s in self.steps if s["subtask_id"] == sid),
                          None)
            if failed and failed["tool"] == "read_file":
                self.failed_paths.add(failed["args"]["path"])
            if not self.reckless and self.replans >= self.replan_budget:
                return self._emit(
                    "abort",
                    f"read failed ({payload['content'][:48]}); "
                    "no replan budget",
                    plan_id="p1", reason="replan_budget_exhausted")
            cand = self._fallback_candidate()
            if cand is None:
                return self._emit("abort", "no fallback candidate",
                                  plan_id="p1", reason="no_candidate")
            self.replans += 1
            step = {"subtask_id": f"s{len(self.steps) + 1}",
                    "tool": "read_file", "args": {"path": cand}}
            self.steps.append(step)
            return self._dispatch(step, replan=True)
        raise AssertionError(f"planner: unknown pending {kind!r}")


def extract_answer(text):
    """Declared extraction rule — real string computation on real file
    text (no model): title line, cut 'Synergizing ' ... ' in Language
    Models', lowercase. Mirrors what L1's trained model 'answered'."""
    m = re.search(r"Synergizing (.+?) in Language Models",
                  text.splitlines()[0])
    return m.group(1).lower() if m else None


class ExecutorAgent:
    """Declared rule-based executor: run the named tool with the given
    args; exceptions become status=error results (L1 defense #3, now
    crossing an agent boundary)."""

    def __init__(self, name="executor-1"):
        self.name = name
        self.last_valid = None

    def compose(self, subtask):
        md = subtask["metadata"]
        try:
            obs = TOOLS[md["tool"]].run(md["args"])
            status = "ok"
        except Exception as e:                 # KeyError (unknown tool) too
            obs = f"{type(e).__name__}: {e}"
            status = "error"
        self.last_valid = msg(self.name, "executor", obs, kind="result",
                              plan_id="p1", subtask_id=md["subtask_id"],
                              status=status)
        return self.last_valid

    def resend(self, critique):
        """After a boundary rejection: re-send the valid payload."""
        return self.last_valid


class CorruptingExecutor(ExecutorAgent):
    """Declared test vector (L1 FaultModel discipline): the FIRST result
    message drops metadata['subtask_id'] — a real contract violation,
    which the boundary must catch in [4b]. The stored last_valid stays
    clean, so resend() complies."""

    def __init__(self, name="executor-1"):
        super().__init__(name)
        self.corrupted = False

    def compose(self, subtask):
        good = super().compose(subtask)
        if self.corrupted:
            return good
        self.corrupted = True
        bad = dict(good)
        bad["metadata"] = dict(good["metadata"])
        bad["metadata"].pop("subtask_id")
        return bad


class StuckExecutor(ExecutorAgent):
    """Declared test vector: ALWAYS asks for clarification, never produces
    a result — the livelock constructor for [4c]."""

    def compose(self, subtask):
        md = subtask["metadata"]
        return msg(self.name, "executor", "which file exactly?",
                   kind="clarify", plan_id="p1", subtask_id=md["subtask_id"])


class RecklessPlanner(PlannerAgent):
    """Declared test vector: ignores its replan budget and keeps guessing
    wrong file names (corpus2.txt, corpus3.txt, ...) — the construction
    that forces the orchestrator's T4 backstop to fire in [4e]."""

    def __init__(self, task=TASK_NOTES):
        super().__init__(task=task, believed_file="notes.txt", reckless=True)


# ===========================================================================
# [D] the orchestrator — termination conditions as explicit data
# ===========================================================================

class Orchestrator:
    """The message loop between planner and executor.

    Termination conditions, checked every turn — failure is a PRODUCED
    STATE (status + full log), never an exception:
      T1 budget_exhausted   global crossing budget — the last fuse;
      T2 no_progress        livelock guard: guard_window consecutive turns
                            without a progress event (plan / first dispatch
                            of a subtask / any result); rejected crossings
                            count as no-progress;
      T3 answered           a validated answer message was accepted;
      T4 replans_exhausted  backstop on replan-flagged subtasks — the
                            orchestrator does NOT trust the planner's own
                            bookkeeping (defense in depth, [4e]);
      (+) aborted           the planner itself emits abort (P4): semantic
                            give-up with partial results in the log.

    Progress event = accepted plan | first dispatch of a subtask_id |
    accepted result. Re-dispatching an already-seen subtask (the naive P5
    reaction to clarify) is NOT progress — that is what makes the guard
    see through the livelock.
    """

    def __init__(self, planner, executor, budget=24, guard_window=4,
                 max_replans=1, guard_on=True):
        self.planner, self.executor = planner, executor
        self.budget, self.guard_window = budget, guard_window
        self.max_replans, self.guard_on = max_replans, guard_on

    def _finish(self, status, answer, attempts, log, violations, replans):
        return {"answer": answer, "status": status, "attempts": attempts,
                "messages": len(log), "violations": len(violations),
                "replans": replans, "log": log,
                "violation_log": list(violations)}

    def run(self, task):
        log, violations = [], []
        attempts = replans = streak = 0
        dispatched, answer = set(), None
        pending, speaker = ("start", task), "planner"
        while True:
            if attempts >= self.budget:                        # T1
                return self._finish("budget_exhausted", answer, attempts,
                                    log, violations, replans)
            if speaker == "planner":
                m = self.planner.compose(pending)
            else:
                kind, payload = pending
                m = (self.executor.resend(payload) if kind == "critique"
                     else self.executor.compose(payload))
            attempts += 1
            v, detail = validate_msg(m)
            if v != "ok":                                      # boundary
                violations.append((m.get("name", "?"), v, detail))
                pending = ("critique",
                           f"contract violation ({v}: {detail}); re-send")
                speaker = m.get("role", "executor")
                streak += 1                                    # no progress
            else:
                log.append(m)
                k = m["metadata"]["kind"]
                progress = False
                if k == "plan":
                    progress = True
                    pending, speaker = ("dispatch_next", None), "planner"
                elif k == "subtask":
                    sid = m["metadata"]["subtask_id"]
                    if m["metadata"].get("replan"):
                        replans += 1
                    if sid not in dispatched:
                        progress = True
                    dispatched.add(sid)
                    pending, speaker = ("subtask", m), "executor"
                elif k == "result":
                    progress = True
                    pending, speaker = ("result", m), "planner"
                elif k == "clarify":
                    pending, speaker = ("clarify", m), "planner"
                elif k == "answer":                            # T3
                    return self._finish("answered", m["content"], attempts,
                                        log, violations, replans)
                elif k == "abort":
                    return self._finish("aborted", None, attempts,
                                        log, violations, replans)
                streak = 0 if progress else streak + 1
            if self.guard_on and streak >= self.guard_window:  # T2
                return self._finish("no_progress", answer, attempts,
                                    log, violations, replans)
            if replans > self.max_replans:                     # T4
                return self._finish("replans_exhausted", answer, attempts,
                                    log, violations, replans)


def trace(r, max_lines=12):
    for i, m in enumerate(r["log"][:max_lines], 1):
        print(f"    #{i} {m['role']:<8} {m['metadata']['kind']:<8} {brief(m)}")
    if len(r["log"]) > max_lines:
        print(f"    ... ({len(r['log']) - max_lines} more messages)")


def status_line(r, prefix="    "):
    print(f"{prefix}status={r['status']} | attempts={r['attempts']} | "
          f"messages={r['messages']} | violations={r['violations']} | "
          f"replans={r['replans']} | answer={r['answer']!r}")


# ===========================================================================
# main
# ===========================================================================

def main():
    print("=" * 69)
    print("nano-agentscope L2 — planner + executor: contracts & termination")
    print("=" * 69)
    print(f"python {sys.version.split()[0]} | stdlib only, no randomness")
    print("declarations: all agents are declared rule-based test vectors")
    print("  (NOT models) — every failure below is a protocol failure by")
    print("  construction; tools are real disk I/O (L1 discipline).")

    # ------------------------------------------------------------------ [0]
    print("\n[0] the message contract (enforced at the boundary)")
    for kind, fields in CONTRACT.items():
        print(f"    {kind:<8} requires metadata: {', '.join(fields)}")
    print(f"    roles: {', '.join(ROLES)} | every crossing is validated "
          "BEFORE it may touch the shared log")

    # ------------------------------------------------------------------ [1]
    print("\n[1] happy path: planner + executor on the L1 task")
    r = Orchestrator(PlannerAgent(), ExecutorAgent()).run(TASK)
    trace(r)
    status_line(r)
    assert r["status"] == "answered" and r["answer"] == EXPECTED_ANSWER
    assert r["messages"] == 6 and r["violations"] == 0 and r["attempts"] == 6

    # ------------------------------------------------------------------ [2]
    print("\n[2] contract violation at the boundary (CorruptingExecutor)")
    r = Orchestrator(PlannerAgent(), CorruptingExecutor()).run(TASK)
    for name, v, detail in r["violation_log"]:
        print(f"    REJECTED {name}: {v} — {detail}")
    status_line(r)
    assert r["status"] == "answered" and r["violations"] == 1
    assert r["violation_log"][0][1] == "missing_field"
    assert r["attempts"] == 7 and r["messages"] == 6

    # ------------------------------------------------------------------ [3]
    print("\n[3] livelock: StuckExecutor, with and without the guard")
    g = Orchestrator(PlannerAgent(), StuckExecutor(),
                     guard_window=4, guard_on=True).run(TASK)
    status_line(g, prefix="    guard ON  (window=4):  ")
    ng = Orchestrator(PlannerAgent(), StuckExecutor(),
                      budget=24, guard_on=False).run(TASK)
    status_line(ng, prefix="    guard OFF (budget=24): ")
    tail = [f"{m['metadata']['kind']}" for m in ng["log"][-4:]]
    print(f"    guard-off tail: {' / '.join(tail)}  <- clarify/re-dispatch "
          "forever; only the budget fuse stopped it, and it mislabeled")
    print(f"    the failure. The guard stopped {ng['attempts'] - g['attempts']} "
          "crossings earlier WITH the correct diagnosis.")
    assert g["status"] == "no_progress" and g["answer"] is None
    assert g["attempts"] == 6 and g["attempts"] < 24
    assert ng["status"] == "budget_exhausted" and ng["attempts"] == 24
    assert ng["answer"] is None

    # ------------------------------------------------------------------ [4]
    print("\n[4] budget sweep: a fuse, not a controller (happy path)")
    print("    budget   status            attempts  messages  answer")
    for B in [2, 4, 6, 8, 16]:
        r = Orchestrator(PlannerAgent(), ExecutorAgent(), budget=B).run(TASK)
        print(f"    {B:<8} {r['status']:<17} {r['attempts']:<9} "
              f"{r['messages']:<9} {r['answer']!r}")
        if B >= 6:
            assert r["status"] == "answered" and r["attempts"] == 6
        else:
            assert r["status"] == "budget_exhausted" and r["answer"] is None
    print("    below the task's message distance (6) the budget amputates;")
    print("    at/above it, it never fires — healthy flows are governed by")
    print("    semantics (T3), the fuse only bounds the pathological cases.")

    # ------------------------------------------------------------------ [5]
    print("\n[5] failure isolation + replan (real FileNotFoundError)")
    r = Orchestrator(PlannerAgent(task=TASK_NOTES, believed_file="notes.txt"),
                     ExecutorAgent(), max_replans=1).run(TASK_NOTES)
    trace(r)
    status_line(r)
    assert r["status"] == "answered" and r["answer"] == EXPECTED_ANSWER
    assert r["replans"] == 1 and r["violations"] == 0
    errs = [m for m in r["log"]
            if m["metadata"]["kind"] == "result"
            and m["metadata"]["status"] == "error"]
    assert len(errs) == 1 and errs[0]["content"].startswith("FileNotFoundError")
    print(f"    the failure stayed inside ONE subtask; the rest of the plan")
    print(f"    survived, and the fallback parsed the REAL list_dir output.")
    # b) planner with zero replan budget -> aborts itself (P4): failure as
    #    a produced state, partial results kept in the log
    r0 = Orchestrator(PlannerAgent(task=TASK_NOTES, believed_file="notes.txt",
                                   replan_budget=0),
                      ExecutorAgent()).run(TASK_NOTES)
    print(f"    zero replan budget: status={r0['status']} | "
          f"abort reason={r0['log'][-1]['metadata']['reason']!r} | "
          f"partial log kept={r0['messages']} msgs")
    assert r0["status"] == "aborted" and r0["answer"] is None
    # c) T4 backstop: a planner that ignores its own budget
    r4 = Orchestrator(RecklessPlanner(), ExecutorAgent(),
                      max_replans=1).run(TASK_NOTES)
    print(f"    RecklessPlanner (ignores budget): status={r4['status']} | "
          f"replans={r4['replans']} — T4 fired even though the planner")
    print(f"    kept going: the orchestrator does not trust agents with "
          f"liveness.")
    assert r4["status"] == "replans_exhausted" and r4["replans"] == 2

    # ------------------------------------------------------------------ [6]
    print("\n[6] the coordination ledger")
    print("    L1 single agent : 3 model calls, zero boundaries to validate.")
    print("    L2 decomposition: 6 messages for the same task — every one")
    print("    crossed a validated boundary. What the extra messages bought,")
    print("    measured above: a corrupted crossing caught BEFORE it touched")
    print("    shared state [2]; a livelock diagnosed, not just truncated [3];")
    print("    a failed subtask replanned locally while the plan survived [5].")
    print("    In real systems each planner/executor turn is itself a model")
    print("    call — L1's calls/success ledger generalizes to messages/task.")

    # ------------------------------------------------------------------ end
    print("\n" + "=" * 69)
    print("✅ self-check passed:")
    print("   happy path answers correctly in exactly 6 validated messages /")
    print("   boundary rejects the corrupted crossing, then recovers /")
    print("   livelock guard: no_progress at 6 attempts vs budget fuse at 24 /")
    print("   budget sweep: fuse never fires on the healthy flow /")
    print("   real FileNotFoundError isolated + replanned / abort is a state /")
    print("   T4 backstop fires on a budget-ignoring planner")
    print("=" * 69)
    print("\ntakeaway: put two reliable agents in a room and they can still")
    print("          loop forever, shout past each other, or hand each other")
    print("          malformed notes. Reliability of the PARTS does not imply")
    print("          reliability of the WHOLE — that takes a typed contract at")
    print("          every crossing and termination conditions owned by the")
    print("          orchestrator, not by the agents.")


if __name__ == "__main__":
    main()
