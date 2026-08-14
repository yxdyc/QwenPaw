"""
nano-agentscope L3 — typed messages x broadcast wiring x a real model
==========================================================================
L1 put a real (tiny) model inside a single-agent loop; L2 built the
protocol layer (contracts + termination) with rule-based agents. L3 fuses
the two and upgrades the message abstraction itself, against TWO snapshots
of the authoritative implementation:

  * AgentScope v2.0.6 (main, 2026-08-10 tarball): messages carry TYPED
    content blocks, are validated AT CONSTRUCTION, and tool calls are
    blocks with a state machine that must pass a permission step;
    termination lives IN the message (finished_reason).
  * AgentScope v1.0.0 (tag): the orchestration pattern is BROADCAST group
    chat — a MsgHub rewires every participant's subscriber list so a reply
    fans out to the whole room; SequentialPipeline is the linear contrast.
    (v2.0.6 removed pipeline/msghub from its core — the orchestration layer
    evolves fast, the contract layer evolves slow. L3 reproduces the
    pattern, which is what survives.)

Experiments:
  [0] train the planner's model — L1's recipe verbatim (real ~94K-param
      char-LM; params/loss must match L1's anchored values bit-for-bit);
  [1] the typed message layer: construction-time validation (5 typed
      errors) + the tool-call state machine (legal path vs permission
      skip);
  [2] the orchestration pattern: hub vs point-to-point vs sequential
      pipeline — SAME agents, SAME contract, three wirings, three
      epistemic states for the verifier;
  [3] happy path with the real model in the planner seat (greedy,
      deterministic): every model output becomes a typed crossing;
  [4] reliability algebra at the PROTOCOL level: per-crossing violation p
      measured at T=0.7, task success vs closed form across retry budgets;
  [5] premature completion — a failure mode only a third role can catch
      (broadcast gave the verifier the room's knowledge for free);
  [6] termination lives in the message: finished_reason=exceed_max_iters
      on the message itself, and the verifier still renders a verdict;
  [7] the coordination ledger: L1 -> L2 -> L3, live numbers.

Dependencies: torch (CPU) via the L1 import. Run:  python L3_typed_msghub.py
Total runtime ~3-5 min (dominated by the ~2 min training, L1 precedent).

Declarations (ROADMAP §3 contract):
  * The planner backend is L1's TinyReActLM, retrained HERE on the SAME
    recipe (real learned distribution, sampled outputs — it memorizes
    trajectories, it does not reason). The hosted-model planner path
    (HostedBackend below) is code-ready but needs a key [TODO: needs key];
    the deterministic fallback is the local tiny model (L1 precedent —
    L1 [6] verified the OpenAI-compatible client against a real HTTP
    contract server).
  * ExecutorAgent / VerifierAgent / RulePlanner / PrematurePlanner /
    WrongAnswerPlanner are declared rule-based test vectors, NOT models —
    the L1/L2 test-vector discipline. Every failure below is constructed,
    and each construction is labeled.
  * Tools are L1's real disk-I/O layer imported WHOLESALE: frozen
    list_dir (six files, L1's 2026-08-06 finalization moment) +
    sandboxed live read_file. L3 deliberately does NOT re-freeze its own
    listing: the model was trained on L1's observation strings, so the
    planner must be fed observations from the world it was trained in —
    a real constraint of model-backed agents (the model carries its own
    world snapshot; the harness must honor it).
  * Determinism: everything is seeded; there are NO timing lines anywhere
    in this file, so outputs are byte-identical across runs (L2 anchor
    discipline).
"""

import json
import os
import random
import sys

# Correct installation site for the bytecode flag (复现运行 machine
# experiment): set BEFORE importing L1/L2 — the flag only protects modules
# imported after it is set, never the module that sets it.
sys.dont_write_bytecode = True

import L1_real_agent_loop as L1          # noqa: E402  (needs torch)
import L2_planner_executor as L2         # noqa: E402  (stdlib only)

torch = L1.torch                          # re-export for the version line

EXPECTED_ANSWER = L1.EXPECTED_ANSWER      # "reasoning and acting"
TASK = L1.TASK


# ===========================================================================
# [A] the typed message layer (snapshot: AgentScope v2.0.6 main)
# ===========================================================================
# v2 Msg: content is a list of typed blocks; validate_role_content runs at
# CONSTRUCTION (pydantic model_validator); workflow-control fields
# (finished_reason & co.) live on the message itself. The nano mirror makes
# validation raise typed errors — a malformed message cannot be born.

BLOCK_TYPES = ("text", "tool_call", "tool_result")

# v2 ToolCallState, full vocabulary (message/_block.py): the nano flow
# exercises pending -> allowed -> finished; 'asking' (human-in-the-loop
# confirmation) and 'submitted' (external execution) are declared not
# exercised here.
TOOL_CALL_STATES = ("pending", "asking", "allowed", "submitted", "finished")
# Legal transitions, mirrored from the v2 docstring transition diagram:
#   pending  -> asking (permission ASK) | allowed (ALLOW) | finished (DENY)
#   asking   -> allowed (approved) | finished (denied)
#   allowed  -> finished (local exec) | submitted (external tool)
#   submitted-> finished (external result event)
#   finished -> (terminal)
TOOL_CALL_TRANSITIONS = {
    "pending": {"asking", "allowed", "finished"},
    "asking": {"allowed", "finished"},
    "allowed": {"finished", "submitted"},
    "submitted": {"finished"},
    "finished": set(),
}
# v2 ToolResultState, full vocabulary; nano produces success/error directly
# (no streaming: v2's default RUNNING state is declared out of scope).
TOOL_RESULT_STATES = ("success", "error", "interrupted", "denied", "running")

# v2 ReplyFinishedReason (types/_reply.py StrEnum), verbatim values.
REPLY_FINISHED_REASONS = ("completed", "interrupted", "exceed_max_iters",
                          "error")

# Role x block legality — the nano analog of v2's validate_role_content
# (per-role assertions on content blocks; v2 restricts user messages to
# text/data, system to text, at _base.py L33-48). Separation of privilege
# BY CONSTRUCTION: the executor cannot even construct a tool_call, the
# planner cannot fabricate a tool_result.
ROLE_BLOCKS = {
    "user": {"text"},
    "planner": {"text", "tool_call"},
    "executor": {"text", "tool_result"},
    "verifier": {"text"},
}


class MsgValidationError(ValueError):
    """Typed construction failure. Kinds (continuing the L1/L2 typed-
    violation lineage, now at message birth rather than at the gate):
      empty_content / bad_block_type / bad_block_field /
      role_block_mismatch / bad_finished_reason
    """

    def __init__(self, kind, detail):
        super().__init__(f"{kind}: {detail}")
        self.kind, self.detail = kind, detail


def text_block(text):
    return {"type": "text", "text": text}


def tool_call_block(call_id, name, args, state="pending"):
    # v2 ToolCallBlock.input is the RAW JSON STRING (accumulated during
    # streaming) — mirrored here: input is a string, not a dict.
    return {"type": "tool_call", "id": call_id, "name": name,
            "input": json.dumps(args), "state": state}


def tool_result_block(call_id, name, output, state):
    return {"type": "tool_result", "id": call_id, "name": name,
            "output": output, "state": state}


class Msg:
    """The nano Msg (v2.0.6 mirror): name/role/content(list of typed
    blocks)/metadata + the workflow-control field finished_reason.
    Validation happens HERE, at construction — not at a gate downstream."""

    def __init__(self, name, role, content, metadata=None,
                 finished_reason=None):
        if not isinstance(content, list) or len(content) == 0:
            raise MsgValidationError(
                "empty_content", "content must be a non-empty list of blocks")
        for b in content:
            if not isinstance(b, dict) or b.get("type") not in BLOCK_TYPES:
                raise MsgValidationError(
                    "bad_block_type",
                    f"block type {b.get('type') if isinstance(b, dict) else b!r}"
                    f" not in {list(BLOCK_TYPES)}")
            t = b["type"]
            if t == "text" and not (isinstance(b.get("text"), str)
                                    and b["text"]):
                raise MsgValidationError(
                    "bad_block_field", "text block needs a non-empty 'text'")
            if t == "tool_call" and not (
                    isinstance(b.get("id"), str) and b["id"]
                    and isinstance(b.get("name"), str) and b["name"]
                    and isinstance(b.get("input"), str)
                    and b.get("state") in TOOL_CALL_STATES):
                raise MsgValidationError(
                    "bad_block_field",
                    f"tool_call block malformed: {sorted(b.keys())}")
            if t == "tool_result" and not (
                    isinstance(b.get("id"), str) and b["id"]
                    and isinstance(b.get("name"), str) and b["name"]
                    and isinstance(b.get("output"), str)
                    and b.get("state") in TOOL_RESULT_STATES):
                raise MsgValidationError(
                    "bad_block_field",
                    f"tool_result block malformed: {sorted(b.keys())}")
            if t not in ROLE_BLOCKS[role]:
                raise MsgValidationError(
                    "role_block_mismatch",
                    f"role {role!r} may not carry block type {t!r} "
                    f"(allowed: {sorted(ROLE_BLOCKS[role])})")
        if finished_reason not in (None,) + REPLY_FINISHED_REASONS:
            raise MsgValidationError(
                "bad_finished_reason",
                f"{finished_reason!r} not in {list(REPLY_FINISHED_REASONS)}")
        self.name, self.role, self.content = name, role, content
        self.metadata = dict(metadata or {})
        self.finished_reason = finished_reason

    def get_text(self, sep=" "):
        parts = [b["text"] for b in self.content if b["type"] == "text"]
        return sep.join(parts) if parts else None

    def blocks(self, btype):
        return [b for b in self.content if b["type"] == btype]

    def brief(self):
        k = self.metadata.get("kind", "?")
        if k == "subtask":
            call = self.blocks("tool_call")[0]
            return (f"{self.metadata['subtask_id']} {call['name']} "
                    f"{call['input']} [{call['state']}]")
        if k == "result":
            res = self.blocks("tool_result")[0]
            return (f"{self.metadata['subtask_id']} <- "
                    f"{res['output'][:52]} [{res['state']}]")
        if k == "answer":
            fr = f" [finished_reason={self.finished_reason}]" \
                if self.finished_reason else ""
            return f"{(self.get_text() or '')[:60]}{fr}"
        if k == "verdict":
            return (f"{self.metadata['status']} "
                    f"({self.metadata['reason']})")
        return (self.get_text() or "")[:60]


def transition(call_block, to_state):
    """Tool-call state machine (v2 ToolCallState mirror). Only legal edges
    pass; 'pending -> submitted' (skipping the permission system) and any
    edge out of 'finished' are typed errors."""
    frm = call_block["state"]
    if to_state not in TOOL_CALL_TRANSITIONS[frm]:
        raise MsgValidationError(
            "illegal_transition",
            f"tool_call {call_block['id']}: {frm} -> {to_state} is not a "
            f"legal edge (legal from {frm!r}: "
            f"{sorted(TOOL_CALL_TRANSITIONS[frm])})")
    call_block["state"] = to_state


# ===========================================================================
# [B] orchestration combinators (snapshot: AgentScope v1.0.0 tag)
# ===========================================================================
# v1 mechanism, verified in source: MsgHub does NOT route messages. On
# entry it rewires every participant's subscriber list (reset_subscribers
# excludes self); thereafter each agent's reply auto-fans-out to its
# subscribers. broadcast() is the explicit variant (observe for ALL
# participants, sender included).

class AgentBase:
    """Nano AgentBase: memory + subscribers (v1 _agent_base.py mirror)."""

    def __init__(self, name, role):
        self.name, self.role = name, role
        self.memory = []
        self._subscribers = []
        self.observations = 0

    def observe(self, msg):
        self.memory.append(msg)
        self.observations += 1

    def reset_subscribers(self, subscribers):
        # v1 L447: an agent never subscribes to itself.
        self._subscribers = [s for s in subscribers if s is not self]

    def speak(self, msg):
        """Emit a message: fan out to subscribers (reply-broadcast, v1
        __call__ mirror). Returns the message for orchestrator logging."""
        for s in self._subscribers:
            s.observe(msg)
        return msg


class MsgHub:
    """v1 MsgHub mirror: context manager rewiring the subscription graph."""

    def __init__(self, participants, announcement=None):
        self.participants = participants
        self.announcement = announcement

    def __enter__(self):
        for p in self.participants:
            p.reset_subscribers(self.participants)
        if self.announcement is not None:
            for p in self.participants:
                p.observe(self.announcement)
        return self

    def __exit__(self, *exc):
        for p in self.participants:
            p.reset_subscribers([])

    def broadcast(self, msg):
        """Explicit broadcast: ALL participants observe, sender included
        (v1 broadcast L115-123 — note the difference from reply-broadcast,
        which excludes the sender)."""
        for p in self.participants:
            p.observe(msg)


def sequential_pipeline(agents, msg):
    """v1 sequential_pipeline mirror (pipeline/_functional.py): a fold —
    each agent's output becomes the next agent's ONLY input."""
    for a in agents:
        msg = a(msg)
    return msg


# ===========================================================================
# [C] agents — one real model in the planner seat, the rest declared
# ===========================================================================

TOOLS = L1.make_tools()   # frozen list_dir (six files) + sandboxed read_file


class ModelPlanner(AgentBase):
    """Planner backed by a real (tiny) model. Each turn: sample the backend
    with the ReAct prefix, parse the raw text (L1's parse_block — layer 1),
    and, if compliant, let the orchestrator wrap it into a typed crossing.

    The prefix is built EXACTLY as L1.Harness builds it, because the model
    was trained on exactly that format — character-level memory has zero
    slack for prompt drift. The plan lives in the model's weights (a
    memorized trajectory); the protocol only ever sees the crossings."""

    def __init__(self, backend, task=TASK):
        super().__init__("planner", "planner")
        self.backend, self.task = backend, task
        self.prefix = f"Task: {task}\nTools: list_dir, read_file\n"
        self.calls, self.n_subtask = 0, 0
        self.last_action = None

    def next_turn(self, critique=None):
        prompt = self.prefix
        if critique is not None:
            kind, payload = critique
            text = L1.CRITIQUE.format(kind=kind, payload=payload)
            # The critique goes BACK INTO the model's prompt, so it must
            # stay inside the model's input alphabet. Real parser errors
            # carry arbitrary text — e.g. json.JSONDecodeError's
            # 'Unterminated string starting at...' — and a char-level LM
            # with a fixed vocabulary chokes on out-of-alphabet chars
            # (KeyError in the embedding lookup). L1's seeded sweep never
            # drew such a payload; L3's wider T=0.7 sweep does. Hosted
            # tokenized backends expose no .stoi and need no sanitizing.
            alpha = getattr(self.backend, "stoi", None)
            if alpha is not None:
                text = "".join(c if c in alpha else "?" for c in text)
            prompt += text
        raw = self.backend(prompt)
        self.calls += 1
        kind, payload = L1.parse_block(raw, TOOLS)
        if kind == "final":
            return raw, "answer", payload
        if kind == "compliant":
            name, args = payload
            if (name, json.dumps(args, sort_keys=True)) == self.last_action:
                return raw, "loop", None            # L1 loop guard
            return raw, "subtask", (name, args)
        return raw, "violation", (kind, payload)

    def observe_result(self, tool_name, args, obs, raw_block):
        # L1.Harness prefix update, verbatim shape.
        self.prefix += raw_block.rstrip("\n") + f"\nObservation: {obs}\n"
        self.last_action = (tool_name, json.dumps(args, sort_keys=True))

    def subtask_msg(self, tool_name, args):
        self.n_subtask += 1
        sid = f"s{self.n_subtask}"
        call = tool_call_block(f"tc{self.n_subtask}", tool_name, args)
        return Msg(self.name, self.role, [call],
                   metadata={"kind": "subtask", "subtask_id": sid,
                             "tool": tool_name, "args": args}), call, sid

    def answer_msg(self, answer, finished_reason=None):
        return Msg(self.name, self.role, [text_block(str(answer))],
                   metadata={"kind": "answer", "answer": answer},
                   finished_reason=finished_reason)

    def partial_msg(self, raw):
        """Retry budget exhausted: the reply ENDS, but a message still
        exists — carrying finished_reason=exceed_max_iters and the last raw
        text (v2: structured_output stays None when the reply ends before
        it is generated; the reason lives ON the message)."""
        return Msg(self.name, self.role,
                   [text_block(f"<last raw> {raw.strip()[:60]}")],
                   metadata={"kind": "answer", "answer": None},
                   finished_reason="exceed_max_iters")


class RulePlanner(AgentBase):
    """Declared rule-based planner (NOT a model) — the wiring experiments
    [2] need agents held constant while the wiring varies. Policy:
    s1=list_dir, s2=read_file(corpus.txt), then answer extracted from the
    REAL read result by L2's declared extraction rule (real string
    computation, no model)."""

    def __init__(self, final_text=None):
        super().__init__("planner", "planner")
        self.final_text = final_text        # WrongAnswerPlanner hook
        self.n_subtask, self.read_obs = 0, None
        self.calls = 0                      # rule planners make no model calls

    def next_turn(self, critique=None):
        if self.n_subtask == 0:
            return "<rule>", "subtask", ("list_dir", {})
        if self.n_subtask == 1:
            return "<rule>", "subtask", ("read_file", {"path": "corpus.txt"})
        answer = self.final_text or L2.extract_answer(self.read_obs)
        return "<rule>", "answer", answer

    def observe_result(self, tool_name, args, obs, raw_block):
        # n_subtask is advanced by subtask_msg (inherited); here we only
        # record the evidence the answer will be extracted from.
        if tool_name == "read_file":
            self.read_obs = obs

    subtask_msg = ModelPlanner.subtask_msg
    answer_msg = ModelPlanner.answer_msg

    def partial_msg(self, raw):
        raise AssertionError("rule planner never exhausts retries")


class PrematurePlanner(RulePlanner):
    """Declared test vector: emits the CORRECT answer immediately, before
    any subtask/result crossing. The failure is not ignorance — it is
    declaring completion without evidence in the shared log. This is the
    classic real-LLM failure mode L2 §9 deferred to L3."""

    def next_turn(self, critique=None):
        return "<rule>", "answer", EXPECTED_ANSWER


class ExecutorAgent(AgentBase):
    """Declared rule-based executor: run the named tool on the ALLOWED
    call block; real exceptions become state=error results. Refuses to
    touch a call that has not passed the permission step."""

    def __init__(self):
        super().__init__("executor-1", "executor")

    def compose(self, subtask_msg, call):
        if call["state"] != "allowed":
            raise MsgValidationError(
                "illegal_transition",
                f"executor refuses call {call['id']} in state "
                f"{call['state']!r} — permission step not passed")
        try:
            obs = TOOLS[call["name"]].run(json.loads(call["input"]))
            state = "success"
        except Exception as e:                  # KeyError (unknown tool) too
            obs = f"{type(e).__name__}: {e}"
            state = "error"
        transition(call, "finished")            # allowed -> finished
        sid = subtask_msg.metadata["subtask_id"]
        return Msg(self.name, self.role,
                   [tool_result_block(call["id"], call["name"], obs, state)],
                   metadata={"kind": "result", "subtask_id": sid})


class VerifierAgent(AgentBase):
    """Declared rule-based verifier: judges an answer ONLY from what it
    observed. Under hub wiring that is the whole room (broadcast); under
    point-to-point wiring it is nothing — and the verdict says so.
    Verdict rule (declared, machine-checkable):
      incomplete_reply  the answer message carries no well-formed answer;
      no_evidence       no tool_result block was ever observed;
      unsupported       evidence exists but does not contain the answer
                        (case-insensitive substring — deliberately crude);
      verified          answer found in the observed evidence."""

    def __init__(self):
        super().__init__("verifier-1", "verifier")

    def compose(self, answer_msg):
        answer = answer_msg.metadata.get("answer")
        if answer is None:
            status, reason = "not_verified", "incomplete_reply"
            return self._verdict(status, reason)
        evidence = " ".join(
            b["output"] for m in self.memory for b in m.blocks("tool_result"))
        if not evidence:
            return self._verdict("not_verified", "no_evidence")
        if str(answer).lower() in evidence.lower():
            return self._verdict("verified", "supported_by_evidence")
        return self._verdict("not_verified", "unsupported")

    def _verdict(self, status, reason):
        return Msg(self.name, self.role,
                   [text_block(f"verdict: {status} ({reason})")],
                   metadata={"kind": "verdict", "status": status,
                             "reason": reason})


class HostedBackend:
    """Hosted planner backend, code-ready [TODO: needs key]. Reuses L1's
    OpenAI-compatible client, which L1 [6] round-tripped through a real
    HTTP contract server. Without a key, make_backend() falls back to the
    local tiny model — the deterministic fallback (L1 precedent)."""

    def __init__(self, model="qwen-turbo",
                 base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"):
        key = os.environ.get("DASHSCOPE_API_KEY") \
            or os.environ.get("OPENAI_API_KEY")
        if not key:
            raise RuntimeError("[TODO: needs key] set DASHSCOPE_API_KEY or "
                               "OPENAI_API_KEY to use the hosted planner")
        self.client = L1.OpenAICompatChat(base_url, key, model)

    def __call__(self, prompt):
        return self.client(prompt)


def make_backend():
    """Hosted model if a key exists, else None -> caller uses the locally
    trained TinyReActLM (deterministic fallback, L1 precedent)."""
    key = os.environ.get("DASHSCOPE_API_KEY") \
        or os.environ.get("OPENAI_API_KEY")
    if key:
        return HostedBackend()
    return None


# ===========================================================================
# [D] the hub orchestrator — crossings, permission, verdicts
# ===========================================================================

class HubOrchestrator:
    """The message loop over the hub.

    Wiring modes:
      hub   every accepted crossing is spoken through the hub: all other
            participants observe it (subscriber rewiring, [B]);
      p2p   each crossing is delivered only to its intended recipient;
            the verifier observes nothing unless forward_to_verifier=True
            adds an EXPLICIT send per crossing (the cost broadcast avoids).

    Termination (session-level status; the message-level finished_reason
    is a separate channel carried by the messages themselves):
      verified / not_verified / exceed_max_iters / loop_detected /
      budget_exhausted
    """

    def __init__(self, planner, executor, verifier, wiring="hub",
                 max_retries=1, budget=24, forward_to_verifier=False):
        self.planner, self.executor, self.verifier = planner, executor, verifier
        self.wiring, self.max_retries, self.budget = wiring, max_retries, budget
        self.forward_to_verifier = forward_to_verifier
        self.extra_sends = 0

    def _deliver(self, msg, speaker, intended):
        if self.wiring == "hub":
            speaker.speak(msg)
        else:
            intended.observe(msg)
            if (self.forward_to_verifier and intended is not self.verifier
                    and speaker is not self.verifier):
                self.verifier.observe(msg)
                self.extra_sends += 1

    def run(self, task):
        task_msg = Msg("user", "user", [text_block(task)],
                       metadata={"kind": "task"})
        log, violations = [], []
        attempts = 0
        critique = None
        retries_this_crossing = 0
        participants = [self.planner, self.executor, self.verifier]
        hub = MsgHub(participants, announcement=task_msg) if \
            self.wiring == "hub" else None
        if hub:
            hub.__enter__()
        try:
            while True:
                if attempts >= self.budget:
                    return self._finish("budget_exhausted", None, attempts,
                                        log, violations)
                raw, kind, payload = self.planner.next_turn(critique)
                critique = None
                attempts += 1
                if kind == "violation":
                    vkind, detail = payload
                    violations.append(("planner", vkind, str(detail)))
                    if retries_this_crossing < self.max_retries:
                        retries_this_crossing += 1
                        critique = (vkind, detail)
                        continue
                    partial = self.planner.partial_msg(raw)
                    self._deliver(partial, self.planner, self.verifier)
                    log.append(partial)
                    verdict = self.verifier.compose(partial)
                    self._deliver(verdict, self.verifier, self.planner)
                    log.append(verdict)
                    return self._finish("exceed_max_iters", None, attempts,
                                        log, violations)
                if kind == "loop":
                    return self._finish("loop_detected", None, attempts,
                                        log, violations)
                if kind == "subtask":
                    retries_this_crossing = 0
                    tool_name, args = payload
                    msg, call, sid = self.planner.subtask_msg(tool_name, args)
                    log.append(msg)
                    self._deliver(msg, self.planner, self.executor)
                    transition(call, "allowed")       # permission step
                    result = self.executor.compose(msg, call)
                    log.append(result)
                    self._deliver(result, self.executor, self.planner)
                    res = result.blocks("tool_result")[0]
                    self.planner.observe_result(tool_name, args,
                                                res["output"], raw)
                    continue
                if kind == "answer":
                    msg = self.planner.answer_msg(payload)
                    msg.finished_reason = "completed"   # well-formed reply
                    log.append(msg)
                    self._deliver(msg, self.planner, self.verifier)
                    verdict = self.verifier.compose(msg)
                    log.append(verdict)
                    self._deliver(verdict, self.verifier, self.planner)
                    status = "verified" if \
                        verdict.metadata["status"] == "verified" else \
                        "not_verified"
                    return self._finish(status, payload, attempts, log,
                                        violations)
        finally:
            if hub:
                hub.__exit__()

    def _finish(self, status, answer, attempts, log, violations):
        return {"status": status, "answer": answer, "attempts": attempts,
                "model_calls": self.planner.calls,
                "crossings": len(log), "log": log,
                "violations": list(violations),
                "extra_sends": self.extra_sends,
                "verifier_knowledge": len(
                    [m for m in self.verifier.memory
                     if m.metadata.get("kind") != "task"])}


# ===========================================================================
# main
# ===========================================================================

def train_planner_model():
    """L1's recipe, verbatim: same transcript builders, same rng stream,
    same seed — the resulting weights are L1's weights."""
    rng = random.Random(L1.SEED)
    dir_obs = L1.list_dir()
    read_obs = L1.read_file("corpus.txt")
    transcripts = L1.make_transcripts(rng, dir_obs, read_obs)
    n_clean = len(transcripts)
    transcripts += L1.make_critique_transcripts(rng, dir_obs, read_obs,
                                                TOOLS)
    model, stoi, itos, UNK, PAD, loss, nparams, _secs = \
        L1.train_tiny_lm(transcripts)
    return (model, stoi, itos, UNK, PAD, loss, nparams,
            len(transcripts), n_clean)


def sample_kind(model, stoi, itos, unk, pad, prefix, T, seed):
    block = L1.generate(model, stoi, itos, prefix, temperature=T,
                        max_new=250, seed=seed, unk=unk, pad=pad)
    kind, _ = L1.parse_block(block, TOOLS)
    return kind


def main():
    print("=" * 69)
    print("nano-agentscope L3 — typed messages x broadcast wiring x model")
    print("=" * 69)
    print(f"python {sys.version.split()[0]} | torch {torch.__version__}")
    print("declarations: planner backend = L1's TinyReActLM retrained on")
    print("  the SAME recipe (real ~94K-param char-LM; memorizes, does not")
    print("  reason); hosted planner path code-ready [TODO: needs key];")
    print("  executor/verifier/Rule/Premature/WrongAnswer planners are")
    print("  declared rule-based test vectors; tools = L1's real disk I/O")
    print("  (frozen list_dir + sandboxed read_file), imported wholesale.")

    # ------------------------------------------------------------------ [0]
    print("\n[0] train the planner's model (L1 recipe, verbatim, ~2 min)")
    (model, stoi, itos, UNK, PAD, loss, nparams, ntrans, nclean) = \
        train_planner_model()
    print(f"    transcripts={ntrans} ({nclean} clean + {ntrans - nclean} "
          f"critique-repair) | params={nparams:,} | final loss={loss:.4f}")
    assert nparams == 93731 and f"{loss:.4f}" == "0.0218", \
        "L3's model must BE L1's model (same recipe, same seed)"
    print("    cross-level anchor: params/loss match L1's anchored values")
    print("    bit-for-bit — L3's planner IS L1's model, no re-derivation.")
    backend = make_backend() or L1.TinyReActLM(model, stoi, itos, UNK, PAD,
                                               temperature=0.0)
    hosted = "hosted (key found)" if make_backend() else \
        "local TinyReActLM fallback [TODO: needs key for hosted]"
    print(f"    planner backend: {hosted}")

    # ------------------------------------------------------------------ [1]
    print("\n[1] the typed message layer: validation at construction")
    demos = [
        ("empty_content",
         lambda: Msg("planner", "planner", [])),
        ("bad_block_type",
         lambda: Msg("planner", "planner", [{"type": "image"}])),
        ("bad_block_field",
         lambda: Msg("planner", "planner",
                     [{"type": "tool_call", "id": "", "name": "x",
                       "input": "{}", "state": "pending"}])),
        ("role_block_mismatch (executor forging a tool_call)",
         lambda: Msg("executor-1", "executor",
                     [tool_call_block("tc9", "read_file", {})])),
        ("role_block_mismatch (planner fabricating a tool_result)",
         lambda: Msg("planner", "planner",
                     [tool_result_block("tc1", "list_dir", "x", "success")])),
        ("bad_finished_reason",
         lambda: Msg("planner", "planner", [text_block("x")],
                     finished_reason="done-ish")),
    ]
    for label, fn in demos:
        try:
            fn()
            raise AssertionError(f"construction should have failed: {label}")
        except MsgValidationError as e:
            print(f"    REJECTED at birth  {label:<52} -> {e.kind}")
    call = tool_call_block("tc1", "list_dir", {})
    transition(call, "allowed")
    transition(call, "finished")
    print(f"    state machine legal path: pending -> allowed -> finished"
          f"  (final state: {call['state']})")
    try:
        skip = tool_call_block("tc2", "read_file", {"path": "corpus.txt"})
        transition(skip, "submitted")
        raise AssertionError("permission skip should have failed")
    except MsgValidationError as e:
        print(f"    REJECTED transition: pending -> submitted "
              f"({e.kind}) — the permission system cannot be skipped")
    print("    two validation layers: [layer 1] raw model text -> typed")
    print("    crossing (L1 parse kinds); [layer 2] message construction")
    print("    (the five kinds above). A malformed message cannot be born.")

    # ------------------------------------------------------------------ [2]
    print("\n[2] the orchestration pattern: same agents, three wirings")
    rows = []
    for wiring, fwd in (("hub", False), ("p2p", False), ("p2p+forward", True)):
        o = HubOrchestrator(RulePlanner(), ExecutorAgent(), VerifierAgent(),
                            wiring="p2p" if wiring.startswith("p2p") else "hub",
                            max_retries=1, forward_to_verifier=fwd)
        r = o.run(TASK)
        rows.append((wiring, r, o))
        print(f"    {wiring:<12} verdict={r['status']:<13} "
              f"verifier knowledge={r['verifier_knowledge']} msgs | "
              f"extra sends={o.extra_sends}")
    seq_seen = []
    planner_once = lambda m: Msg("planner", "planner",
                                 [tool_call_block("tc1", "read_file",
                                                  {"path": "corpus.txt"})],
                                 metadata={"kind": "subtask",
                                           "subtask_id": "s1"})
    def executor_once(m):
        call = m.blocks("tool_call")[0]
        transition(call, "allowed")
        obs = TOOLS[call["name"]].run(json.loads(call["input"]))
        return Msg("executor-1", "executor",
                   [tool_result_block("tc1", "read_file", obs, "success")],
                   metadata={"kind": "result", "subtask_id": "s1"})
    def verifier_once(m):
        seq_seen.append(m)
        v = VerifierAgent()
        v.observe(m)
        ans = Msg("planner", "planner", [text_block(EXPECTED_ANSWER)],
                  metadata={"kind": "answer", "answer": EXPECTED_ANSWER})
        return v.compose(ans)
    sequential_pipeline([planner_once, executor_once, verifier_once],
                        Msg("user", "user", [text_block(TASK)],
                            metadata={"kind": "task"}))
    print(f"    sequential    each stage sees exactly the previous stage's "
          f"output (verifier saw {len(seq_seen)} msg: the result, never the "
          f"task; a fold has no shared log)")
    assert rows[0][1]["status"] == "verified" and rows[0][1]["verifier_knowledge"] == 5
    assert rows[1][1]["status"] == "not_verified" and \
        rows[1][1]["verifier_knowledge"] == 1      # the answer, no evidence
    assert rows[2][1]["status"] == "verified" and rows[2][0] == "p2p+forward" \
        and rows[2][1]["extra_sends"] == 4
    print("    broadcast buys the verifier the room's knowledge at ZERO")
    print("    extra sends; point-to-point pays one explicit send per")
    print("    crossing to get the same epistemic state.")

    # ------------------------------------------------------------------ [3]
    print("\n[3] happy path: the real model in the planner seat (greedy)")
    o = HubOrchestrator(ModelPlanner(backend), ExecutorAgent(),
                        VerifierAgent(), wiring="hub", max_retries=1)
    r = o.run(TASK)
    for i, m in enumerate(r["log"], 1):
        print(f"    #{i} {m.role:<8} {m.metadata.get('kind', '?'):<8} "
              f"{m.brief()}")
    print("    (block states shown as of end-of-run: every tool_call has")
    print("    traversed pending -> allowed -> finished, [1] legal path)")
    print(f"    status={r['status']} | model_calls={r['model_calls']} | "
          f"crossings={r['crossings']} | violations={len(r['violations'])} | "
          f"answer={r['answer']!r}")
    obs_total = sum(a.observations for a in
                    (o.planner, o.executor, o.verifier))
    print(f"    observations delivered by broadcast: {obs_total} "
          f"(task announcement to 3 + 6 crossings x 2 others = 15);")
    print(f"    finished_reason on the answer message: "
          f"{r['log'][4].finished_reason!r} — termination is message data.")
    assert r["status"] == "verified" and r["answer"] == EXPECTED_ANSWER
    assert r["model_calls"] == 3 and r["crossings"] == 6
    assert len(r["violations"]) == 0 and obs_total == 15
    assert r["log"][4].finished_reason == "completed"

    # ------------------------------------------------------------------ [4]
    print("\n[4] reliability algebra at the protocol level (T=0.7)")
    dir_obs = L1.list_dir()
    read_obs = L1.read_file("corpus.txt")
    P0 = f"Task: {TASK}\nTools: list_dir, read_file\n"
    P1 = P0 + L1.SCRIPT[0].rstrip("\n") + f"\nObservation: {dir_obs}\n"
    P2 = P1 + L1.SCRIPT[1].rstrip("\n") + f"\nObservation: {read_obs}\n"
    ps = {}
    ps_final0 = 0
    for label, prefix, good in (("step0", P0, "compliant"),
                                ("step1", P1, "compliant"),
                                ("answer", P2, "final")):
        counts = {}
        for i in range(200):
            k = sample_kind(model, stoi, itos, UNK, PAD, prefix, 0.7, 1000 + i)
            counts[k] = counts.get(k, 0) + 1
        ps[label] = 1.0 - counts.get(good, 0) / 200
        if label == "step0":
            ps_final0 = counts.get("final", 0)
        detail = ", ".join(f"{k}:{v}" for k, v in sorted(counts.items()))
        print(f"    p({label}) = {ps[label]:.3f}   [{detail}]")
    formula = lambda k: ((1 - ps["step0"] ** (k + 1))
                         * (1 - ps["step1"] ** (k + 1))
                         * (1 - ps["answer"] ** (k + 1)))
    print("    task runs (200 each, fresh seeded model per task, hub wiring):")
    print("     k   measured   formula   mean_calls")
    measured = {}
    for k in (0, 1, 2):
        ok, calls_tot = 0, 0
        for i in range(200):
            lm = L1.TinyReActLM(model, stoi, itos, UNK, PAD, temperature=0.7,
                                seed_base=60000 + 37 * i)
            rr = HubOrchestrator(ModelPlanner(lm), ExecutorAgent(),
                                 VerifierAgent(), wiring="hub",
                                 max_retries=k).run(TASK)
            ok += (rr["status"] == "verified")
            calls_tot += rr["model_calls"]
        measured[k] = ok / 200
        print(f"    {k}    {measured[k]:6.1%}    {formula(k):6.1%}   "
              f"{calls_tot / 200:6.2f}")
    assert measured[1] > measured[0] and measured[2] >= measured[1]
    assert abs(measured[0] - formula(0)) < 0.10
    assert measured[1] < formula(1) - 0.05      # iid is an UPPER bound here
    print("    k=0: per-site failure probabilities COMPOSE across boundaries")
    print("    — measured within sampling noise of prod_i (1-p_i). The algebra")
    print("    that priced L1's single loop prices the protocol too.")
    print("    k>=1: the iid formula is an upper BOUND, not an equality —")
    print("    retries are not fresh iid draws (failures are sticky, L1) and")
    print("    repair only works where critique-repair was trained (the answer")
    print("    position was not). Retries still buy monotone uplift; the iid")
    print("    magnitude overpromises. Same lesson as L1, one level up:")
    print("    rewiring the loop does not restore independence.")
    print(f"    note: step0 spectrum 'final:{ps_final0}' — organic premature")
    print("    completion: at T=0.7 the real model jumps to Final Answer with")
    print("    no evidence; every such draw is rejected by the verifier below.")

    # ------------------------------------------------------------------ [5]
    print("\n[5] premature completion: only a third role can catch it")
    r = HubOrchestrator(PrematurePlanner(), ExecutorAgent(), VerifierAgent(),
                        wiring="hub").run(TASK)
    print(f"    hub + verifier:   status={r['status']} | verdict="
          f"{r['log'][-1].metadata['reason']} | crossings={r['crossings']}")
    assert r["status"] == "not_verified"
    assert r["log"][-1].metadata["reason"] == "no_evidence"

    class L2PrematurePlanner:
        """The same vector expressed in L2's two-party protocol."""
        def compose(self, pending):
            kind, _ = pending
            assert kind == "start"
            return L2.msg("planner", "planner", EXPECTED_ANSWER,
                          kind="answer", plan_id="p1")
    r2 = L2.Orchestrator(L2PrematurePlanner(), L2.ExecutorAgent()).run(TASK)
    print(f"    L2 two-party:     status={r2['status']} | answer="
          f"{r2['answer']!r} — ACCEPTED: with no verifier role and no")
    print("    evidence check, this failure mode cannot even be expressed.")
    assert r2["status"] == "answered" and r2["answer"] == EXPECTED_ANSWER
    rw = HubOrchestrator(RulePlanner(final_text="reasoning and retrieval"),
                         ExecutorAgent(), VerifierAgent(),
                         wiring="hub").run(TASK)
    print(f"    wrong answer:     status={rw['status']} | verdict="
          f"{rw['log'][-1].metadata['reason']} — evidence existed but did")
    print("    not support the claim (no_evidence vs unsupported are "
          "different diagnoses).")
    assert rw["status"] == "not_verified"
    assert rw["log"][-1].metadata["reason"] == "unsupported"

    # ------------------------------------------------------------------ [6]
    print("\n[6] termination lives in the message (T=1.3, zero retries)")
    lm13 = L1.TinyReActLM(model, stoi, itos, UNK, PAD, temperature=1.3,
                          seed_base=61000)
    r = HubOrchestrator(ModelPlanner(lm13), ExecutorAgent(), VerifierAgent(),
                        wiring="hub", max_retries=0).run(TASK)
    partial = r["log"][-2]
    print(f"    partial message text: {partial.get_text()[:72]!r}")
    print(f"    message.finished_reason = {partial.finished_reason!r} | "
          f"session status = {r['status']} | verdict = "
          f"{r['log'][-1].metadata['reason']}")
    print("    the verifier rendered a verdict ON the failed message — any")
    print("    downstream consumer reads termination off the message itself")
    print("    (v2 ReplyFinishedReason), no session context needed.")
    assert partial.finished_reason == "exceed_max_iters"
    assert r["status"] == "exceed_max_iters"
    assert r["log"][-1].metadata["reason"] == "incomplete_reply"

    # ------------------------------------------------------------------ [7]
    print("\n[7] the coordination ledger (live numbers)")
    r_l2 = L2.Orchestrator(L2.PlannerAgent(), L2.ExecutorAgent()).run(L2.TASK)
    print("    level  model_calls  crossings  parties  verifier  observations")
    print(f"    L1     3            0          1        -         -")
    print(f"    L2     0 (rules)    {r_l2['messages']}          2        "
          f"-         -")
    print(f"    L3     3            6          3        yes       15")
    print("    L2's explicit plan message became implicit (the plan lives")
    print("    in the model's weights; the protocol sees only crossings).")
    print("    What L3's extra wiring bought, measured above: evidence-based")
    print("    acceptance instead of trust [5], knowledge without forwarding")
    print("    [2], termination as message data [6]. What it costs: every")
    print("    observation is tokens in the observer's context — the budget")
    print("    nano-vllm-sglang's KV cache pays for.")

    # ------------------------------------------------------------------ end
    print("\n" + "=" * 69)
    print("✅ self-check passed:")
    print("   retrained model matches L1's anchor bit-for-bit (93,731 / 0.0218) /")
    print("   six typed construction errors + the permission-skip rejected /")
    print("   hub verifies at zero extra sends; p2p starves the verifier /")
    print("   greedy real-model run: 3 calls, 6 crossings, verified /")
    print("   retry algebra: composition holds at k=0, iid upper bound at k>=1 /")
    print("   premature completion caught ONLY when a verifier sees the room /")
    print("   exceed_max_iters lives on the message, verdict still rendered")
    print("=" * 69)
    print("\ntakeaway: contracts become typed, orchestration becomes wiring.")
    print("          A message that cannot be born malformed, a tool call")
    print("          that cannot execute itself, a room where everyone hears")
    print("          everything, and a verifier whose knowledge IS the room's")
    print("          — reliability is no longer a property of any participant.")
    print("          It is a property of the wiring. The wiring is the part")
    print("          that evolves fast (v1's pipelines are gone in v2); the")
    print("          contract is the part you invest in.")


if __name__ == "__main__":
    main()
