"""
nano-agentscope L1 — a real (tiny) model behind the ReAct loop:
where unreliability comes from, and what the harness buys
=====================================================================
L0 used a rule mock that NEVER fails, hiding the core problem of agents:
a real model is a distribution, not a function — its output violates the
format contract with measurable probability. L1 makes this real:

  [0] train a real char-level tiny LM (93,731 params, torch, CPU ~2 min)
      on ReAct transcripts whose observations come from REAL files here;
  [1] greedy decode: the real model completes a 3-step task through two
      REAL tools (disk I/O, sandboxed) — trajectory is deterministic;
  [2] sampling decode: measured format-compliance vs temperature —
      the raw material of the reliability problem, real numbers;
  [3] harness defenses (parse-validate / critique-retry / tool-error
      feedback / loop guard) under declared fault-injection vectors;
  [4] reliability algebra: organic uplift on the real tiny model +
      controlled iid vs sticky (correlated) fault sweeps, measured
      success vs closed-form prediction;
  [5] cost ledger: reliability is bought with extra model calls;
  [6] the real-API path: an OpenAI-compatible client (urllib, key via
      env) verified against a local contract server. Calling a real
      hosted model needs a key: [TODO: needs key].

Dependencies: torch only (CPU). Run:  python L1_real_agent_loop.py

Declarations (course runnability contract):
  * TinyReActLM IS a real language model (learned distribution, sampled
    outputs), but a ~94K-param char-level one — it memorizes trajectories,
    it does not reason. Real hosted-model behavior: [TODO: needs key].
  * Playback / FaultModel are NOT models. They are declared fault-
    injection test vectors (unit-test fixtures) to control failure rates.
  * The local contract server in [6] verifies OUR client code against the
    OpenAI-compatible JSON contract; it is not a real LLM endpoint.
"""

import json
import os
import random
import re
import sys
import time

try:
    import torch
    import torch.nn as nn
except ImportError:
    sys.exit("[error] this script needs torch (CPU is enough):  pip install torch")

SEED = 42
HERE = os.path.dirname(os.path.abspath(__file__))

TASK = ("Read corpus.txt in this directory and answer: according to the "
        "title line of that file, what two things does ReAct synergize?")
EXPECTED_ANSWER = "reasoning and acting"

CRITIQUE = ("System: your last output violated the contract "
            "({kind}: {payload}). Reply with exactly one "
            "Thought/Action/Action Input block, or a Final Answer line.\n")


# ===========================================================================
# [A] REAL tools — disk I/O with a sandbox (least privilege)
# ===========================================================================

class Tool:
    """name + callable + arg schema. The harness turns exceptions into
    observations; tools themselves just raise."""

    def __init__(self, name, fn, schema):
        self.name, self.fn, self.schema = name, fn, schema

    def run(self, kwargs):
        missing = [k for k in self.schema if k not in kwargs]
        if missing:
            raise ValueError(f"missing args for {self.name}: {missing}")
        return str(self.fn(**kwargs))


def list_dir() -> str:
    """Module directory listing, FROZEN at the 2026-08-06 finalization
    moment (six files; taken verbatim from the anchored run output in
    tutorial_L1.md §2, not from memory).

    Declared reproducibility hardening (2026-08-08, independent
    arbitration, conflict option (ii)): this observation feeds BOTH the
    training corpus ([0]) and the runtime obs line ([1]), so any file
    added to the module directory — the L2/L3 deliverables this ladder
    will grow — would otherwise silently shift every downstream number
    of this PASS material. Freezing the observation list decouples L1's
    deterministic anchors from directory state; the mechanism taught in
    [A] (real disk I/O) is unchanged — read_file stays live. Same
    discipline as nano-qwenpaw L1's PADDING_SOURCES fixed explicit list
    (no glob/listdir dependency)."""
    return str(["L0_react_loop.py", "L1_real_agent_loop.py", "README.md",
                "corpus.txt", "tutorial_L0.md", "tutorial_L1.md"])


def read_file(path: str, max_chars: int = 300) -> str:
    """Real file read, head-style (observations eat context budget).
    SANDBOX: paths must stay inside the module directory — realpath
    containment defeats `../` traversal and symlink escapes."""
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


def make_tools():
    return {
        "list_dir": Tool("list_dir", list_dir, {}),
        "read_file": Tool("read_file", read_file, {"path": "str"}),
    }


# ===========================================================================
# [B] strict parser — typed violation categories (L0 returned (None, None))
# ===========================================================================

def parse_block(text, tools):
    """Classify one model output block. Returns (kind, payload):
      final        -> payload = answer string
      compliant    -> payload = (tool_name, args)
      no_action / unknown_tool / bad_json / missing_args -> payload = detail
    """
    m = re.search(r"Final Answer:\s*([^\n]+)", text)
    if m:
        return "final", m.group(1).strip()
    am = re.search(r"Action:\s*(\w+)", text)
    im = re.search(r"Action Input:\s*(\{.*?\})", text, re.DOTALL)
    if not am or not im:
        return "no_action", "cannot find Action / Action Input lines"
    name = am.group(1)
    if name not in tools:
        return "unknown_tool", name
    try:
        args = json.loads(im.group(1))
    except json.JSONDecodeError as e:
        return "bad_json", str(e)[:60]
    missing = [k for k in tools[name].schema if k not in args]
    if missing:
        return "missing_args", str(missing)
    return "compliant", (name, args)


# ===========================================================================
# [C] TinyReActLM — a REAL (tiny) language model, trained below
# ===========================================================================

THOUGHTS = [
    ["I should first see which files are in this directory.",
     "Let me list the directory to find the file.",
     "First I will check what files exist here."],
    ["Now I will read corpus.txt to see its title line.",
     "corpus.txt is there; I read it next.",
     "I should read corpus.txt and look at the first line."],
    ["The title line names the two things directly.",
     "The answer is in the title line I just read.",
     "I have the title line, so I can answer now."],
]


def make_transcripts(rng, dir_obs, read_obs, n=60):
    """Synthetic trajectories (declared): varied thoughts, fixed action
    skeleton, observations taken from the REAL tools above."""
    out = []
    for _ in range(n):
        t = [rng.choice(ts) for ts in THOUGHTS]
        out.append(
            f"Task: {TASK}\nTools: list_dir, read_file\n"
            f"[Step 0] Thought: {t[0]}\nAction: list_dir\nAction Input: {{}}\n"
            f"Observation: {dir_obs}\n"
            f"[Step 1] Thought: {t[1]}\nAction: read_file\n"
            f'Action Input: {{"path": "corpus.txt"}}\n'
            f"Observation: {read_obs}\n"
            f"[Step 2] Thought: {t[2]}\nFinal Answer: {EXPECTED_ANSWER}\n")
    return out


def make_critique_transcripts(rng, dir_obs, read_obs, tools, n=15):
    """Trajectories of the form (clean context) + critique + repair.
    The critique string is built EXACTLY as the harness builds it (same
    template, same parser payload from the violating block), and the
    violating block itself is NOT part of the context — because the
    harness retry prefix is `previous context + critique`, failed block
    excluded. Train prefix == inference prefix, character for character.
    Vocab side effect: critique punctuation ('/', '(', ':' ...) becomes
    part of the model's alphabet."""
    out = []
    for _ in range(n):
        viol = rng.choice(["unknown_tool", "no_action",
                           "bad_json", "missing_args"])
        head = f"Task: {TASK}\nTools: list_dir, read_file\n"
        t0 = rng.choice(THOUGHTS[0])
        if viol == "unknown_tool":
            bad = f"[Step 0] Thought: {t0}\nAction: list_dirx\nAction Input: {{}}"
            ctx = head
        elif viol == "no_action":
            bad = f"[Step 0] Thought: {t0}"
            ctx = head
        elif viol == "bad_json":
            bad = (f"[Step 0] Thought: {t0}\nAction: list_dir\n"
                   "Action Input: {broken: json}")
            ctx = head
        else:  # missing_args at step 1: context already contains step 0
            bad = ("[Step 1] Thought: I will read the file.\n"
                   "Action: read_file\nAction Input: {}")
            ctx = (head
                   + f"[Step 0] Thought: {t0}\nAction: list_dir\n"
                   + "Action Input: {}\n"
                   + f"Observation: {dir_obs}\n")
        kind, payload = parse_block(bad, tools)
        critique = CRITIQUE.format(kind=kind, payload=payload)
        if viol == "missing_args":
            repair = ("[Step 1] Thought: I will read the file.\n"
                      "Action: read_file\n"
                      f'Action Input: {{"path": "corpus.txt"}}')
            tail = (f"\nObservation: {read_obs}\n"
                    f"[Step 2] Thought: {rng.choice(THOUGHTS[2])}\n"
                    f"Final Answer: {EXPECTED_ANSWER}\n")
        else:
            repair = (f"[Step 0] Thought: {t0}\nAction: list_dir\n"
                      "Action Input: {}")
            tail = (f"\nObservation: {dir_obs}\n"
                    f"[Step 1] Thought: {rng.choice(THOUGHTS[1])}\n"
                    "Action: read_file\n"
                    f'Action Input: {{"path": "corpus.txt"}}\n'
                    f"Observation: {read_obs}\n"
                    f"[Step 2] Thought: {rng.choice(THOUGHTS[2])}\n"
                    f"Final Answer: {EXPECTED_ANSWER}\n")
        out.append(ctx + critique + repair + tail)
    return out


class TinyLM(nn.Module):
    def __init__(self, vocab, emb=32, hid=128):
        super().__init__()
        self.emb = nn.Embedding(vocab, emb)
        self.lstm = nn.LSTM(emb, hid, batch_first=True)
        self.out = nn.Linear(hid, vocab)

    def forward(self, x, h=None):
        o, h = self.lstm(self.emb(x), h)
        return self.out(o), h


def train_tiny_lm(transcripts, epochs=150, lr=2e-3, seed=SEED):
    """Char-level LM. PAD gets its own index OUTSIDE the char vocab —
    letting pad collide with a real char (e.g. '\\n') silently removes
    that char from supervision; the first prototype died exactly there."""
    torch.manual_seed(seed)
    chars = sorted(set(c for s in transcripts for c in s))
    stoi = {c: i for i, c in enumerate(chars)}
    itos = {i: c for c, i in stoi.items()}
    V = len(chars) + 2
    UNK, PAD = V - 2, V - 1
    seqs = [torch.tensor([stoi[c] for c in s], dtype=torch.long)
            for s in transcripts]
    model = TinyLM(V)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    rng = random.Random(seed)
    t0 = time.time()
    final_loss = float("nan")
    for ep in range(epochs):
        rng.shuffle(seqs)
        tot, nbat = 0.0, 0
        for i in range(0, len(seqs), 16):
            batch = nn.utils.rnn.pad_sequence(
                seqs[i:i + 16], batch_first=True, padding_value=PAD)
            inp, tgt = batch[:, :-1], batch[:, 1:]
            logits, _ = model(inp)
            loss = nn.functional.cross_entropy(
                logits.reshape(-1, V), tgt.reshape(-1), ignore_index=PAD)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item(); nbat += 1
        final_loss = tot / nbat
    nparams = sum(p.numel() for p in model.parameters())
    return model, stoi, itos, UNK, PAD, final_loss, nparams, time.time() - t0


def generate(model, stoi, itos, prefix, temperature=0.0, max_new=400,
             seed=None, unk=None, pad=None):
    """Sample (or greedy-decode) a continuation; stop at the observation
    boundary or after a complete Final Answer line. `pad` is masked out of
    every step's distribution — sampling can otherwise emit a special index.
    Prefix characters outside the training vocabulary map to UNK instead of
    crashing the critique/retry path."""
    if seed is not None:
        torch.manual_seed(seed)
    if unk is None:
        missing = sorted(set(prefix) - set(stoi))
        if missing:
            raise ValueError(f"prefix contains out-of-vocabulary chars: {missing}")
    x = torch.tensor([[stoi.get(c, unk) for c in prefix]], dtype=torch.long)
    with torch.no_grad():
        logits, h = model(x)
        nxt = logits[0, -1].clone()
        for special in (unk, pad):
            if special is not None:
                nxt[special] = float("-inf")
        out = []
        for _ in range(max_new):
            if temperature == 0.0:
                idx = int(nxt.argmax())
            else:
                p = torch.softmax(nxt / temperature, dim=-1)
                idx = int(torch.multinomial(p, 1))
            out.append(itos[idx])
            logits, h = model(torch.tensor([[idx]]), h)
            nxt = logits[0, -1].clone()
            for special in (unk, pad):
                if special is not None:
                    nxt[special] = float("-inf")
            text = "".join(out)
            if "\nObservation:" in text:
                return text.split("\nObservation:")[0]
            if re.search(r"Final Answer:[^\n]*\n", text):
                return text
        return "".join(out)


class TinyReActLM:
    """Adapter: prompt -> text, same interface as a real API client."""

    def __init__(self, model, stoi, itos, unk, pad, temperature=0.0,
                 seed_base=70000):
        self.model, self.stoi, self.itos = model, stoi, itos
        self.unk, self.pad = unk, pad
        self.temperature = temperature
        self._seed = seed_base

    def __call__(self, prompt):
        self._seed += 1
        return generate(self.model, self.stoi, self.itos, prompt,
                        temperature=self.temperature, seed=self._seed,
                        unk=self.unk, pad=self.pad)


# ===========================================================================
# [D] the harness — defenses between a distribution and a task
# ===========================================================================

class Harness:
    """ReAct loop with explicit defenses:
      1. parse-validate every block (typed categories, [B]);
      2. critique + retry on violation (max_retries per step);
      3. tool exceptions become observations, not crashes;
      4. loop guard: same (action, args) twice in a row -> abort;
      5. max_steps budget;
      6. full trajectory record (this is what flows back as training data).
    """

    def __init__(self, tools, max_steps=6, max_retries=1):
        self.tools = tools
        self.max_steps = max_steps
        self.max_retries = max_retries

    def _exec(self, name, args):
        try:                                            # defense #3
            return self.tools[name].run(args)
        except Exception as e:
            return f"error - {type(e).__name__}: {e}"

    def run(self, model, task, verbose=False):
        traj = []                                       # defense #6
        prefix = f"Task: {task}\nTools: {', '.join(self.tools)}\n"
        calls = 0
        last_action = None
        for step in range(self.max_steps):
            block = model(prefix)
            calls += 1
            kind, payload = parse_block(block, self.tools)
            traj.append({"step": step, "role": "model", "kind": kind,
                         "text": block.strip()})
            if verbose:
                print(f"    [step {step}] ({kind}) "
                      + block.strip().replace("\n", " | ")[:110])
            if kind == "final":
                return {"answer": payload, "status": "answered",
                        "calls": calls, "traj": traj}
            if kind == "compliant":
                name, args = payload
                key = (name, json.dumps(args, sort_keys=True))
                if key == last_action:                  # defense #4
                    traj.append({"step": step, "role": "harness",
                                 "kind": "loop_guard"})
                    return {"answer": None, "status": "loop_detected",
                            "calls": calls, "traj": traj}
                last_action = key
                obs = self._exec(name, args)
                prefix += block.rstrip("\n") + f"\nObservation: {obs}\n"
                traj.append({"step": step, "role": "tool", "kind": name,
                             "text": obs[:110]})
                if verbose:
                    print(f"             obs: {obs[:100]}")
                continue
            # violation -> defense #2: critique and retry, else give up
            fixed = False
            for r in range(self.max_retries):
                critique = CRITIQUE.format(kind=kind, payload=payload)
                block = model(prefix + critique)
                calls += 1
                kind, payload = parse_block(block, self.tools)
                traj.append({"step": step, "role": "model",
                             "kind": f"{kind}(retry{r + 1})",
                             "text": block.strip()})
                if verbose:
                    print(f"    [retry {r + 1}] ({kind}) "
                          + block.strip().replace("\n", " | ")[:100])
                if kind == "final":
                    return {"answer": payload, "status": "answered",
                            "calls": calls, "traj": traj}
                if kind == "compliant":
                    name, args = payload
                    obs = self._exec(name, args)
                    prefix += block.rstrip("\n") + f"\nObservation: {obs}\n"
                    traj.append({"step": step, "role": "tool",
                                 "kind": name, "text": obs[:110]})
                    fixed = True
                    break
            if not fixed:
                return {"answer": None, "status": f"gave_up:{kind}",
                        "calls": calls, "traj": traj}
        return {"answer": None, "status": "max_steps",
                "calls": calls, "traj": traj}


# ===========================================================================
# [E] declared fault-injection test vectors (NOT models)
# ===========================================================================

SCRIPT = [
    "[Step 0] Thought: I should first see which files are in this "
    "directory.\nAction: list_dir\nAction Input: {}",
    "[Step 1] Thought: Now I will read corpus.txt to see its title "
    'line.\nAction: read_file\nAction Input: {"path": "corpus.txt"}',
    "[Step 2] Thought: The title line names the two things directly.\n"
    f"Final Answer: {EXPECTED_ANSWER}",
]


class Playback:
    """Plays canned outputs turn by turn (declared test vector)."""

    def __init__(self, turns):
        self.turns, self.i = list(turns), 0

    def __call__(self, prompt):
        out = self.turns[min(self.i, len(self.turns) - 1)]
        self.i += 1
        return out


class FaultModel:
    """Plays scripted compliant blocks, corrupting each call with
    probability p (seeded). Every corruption is a REAL violation category
    (the ones measured in [2]). sticky=s: after a failure, the next call
    fails with prob s + (1-s)*p (correlated failures — hard prompts stay
    hard). A critique turn replays the same step with a fresh draw."""

    def __init__(self, script, p=0.0, sticky=0.0, seed=0):
        self.script, self.p, self.sticky = script, p, sticky
        self.rng = random.Random(seed)
        self.pos, self.last_failed = -1, False

    def _corrupt(self, block):
        if "Final Answer:" in block:
            modes = ["no_action"]
        elif '"path"' in block:
            modes = ["bad_json", "unknown_tool", "missing_args", "no_action"]
        else:
            modes = ["unknown_tool", "no_action"]
        mode = self.rng.choice(modes)
        if mode == "bad_json":
            return block.replace('"corpus.txt"', "corpus.txt")
        if mode == "missing_args":
            return block.replace('"path": "corpus.txt"', "")
        if mode == "unknown_tool":
            return re.sub(r"Action: (\w+)", r"Action: \1x", block, count=1)
        if "Final Answer:" in block:
            return block.replace("Final Answer:", "Finale Answer:")
        return block.split("\nAction")[0]               # thought only

    def __call__(self, prompt):
        if "violated the contract" not in prompt:       # a new turn
            self.pos += 1
        idx = min(self.pos, len(self.script) - 1)
        q = (self.p if not self.last_failed
             else self.sticky + (1 - self.sticky) * self.p)
        fail = self.rng.random() < q
        self.last_failed = fail
        block = self.script[idx]
        return self._corrupt(block) if fail else block


# ===========================================================================
# [F] real-API client (OpenAI-compatible) + local contract server
# ===========================================================================

class OpenAICompatChat:
    """Minimal OpenAI-compatible chat client, stdlib urllib only.
    Key via env (never hardcoded); base_url overridable for testing."""

    def __init__(self, base_url, api_key, model, timeout=30):
        self.base_url = base_url.rstrip("/")
        self.api_key, self.model, self.timeout = api_key, model, timeout

    def __call__(self, prompt):
        import urllib.request
        body = json.dumps({
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.0,
        }).encode()
        req = urllib.request.Request(
            self.base_url + "/chat/completions", data=body,
            headers={"Authorization": f"Bearer {self.api_key}",
                     "Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            data = json.loads(r.read())
        return data["choices"][0]["message"]["content"]


def verify_api_client_against_local_server(tools):
    """Stand up a local HTTP server implementing the chat-completions
    contract; run ONE full agent task through the real HTTP client.
    Verifies our request building / auth header / response parsing —
    NOT a real LLM ([TODO: needs key] for the real endpoint)."""
    import threading
    from http.server import BaseHTTPRequestHandler, HTTPServer
    captured = {"k": 0}
    canned = list(SCRIPT)

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            n = int(self.headers.get("Content-Length", 0))
            captured["path"] = self.path
            captured["auth"] = self.headers.get("Authorization", "")
            captured["body"] = json.loads(self.rfile.read(n))
            content = canned[min(captured["k"], len(canned) - 1)]
            captured["k"] += 1
            payload = json.dumps({"choices": [{"message": {
                "role": "assistant", "content": content}}]}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, *a):
            pass

    srv = HTTPServer(("127.0.0.1", 0), Handler)
    port = srv.server_address[1]
    th = threading.Thread(target=srv.serve_forever, daemon=True)
    th.start()
    try:
        client = OpenAICompatChat(
            f"http://127.0.0.1:{port}/compatible-mode/v1",
            api_key="test-key-redacted", model="qwen-turbo")
        result = Harness(tools, max_steps=6, max_retries=0).run(client, TASK)
    finally:
        srv.shutdown()
    return result, captured


# ===========================================================================
# main
# ===========================================================================

def run_batch(factory, n_runs, max_retries):
    """Task success rate + mean model calls over n_runs fresh tasks."""
    ok, calls_tot = 0, 0
    for i in range(n_runs):
        r = Harness(make_tools(), max_steps=8, max_retries=max_retries).run(
            factory(i), TASK)
        ok += (r["answer"] == EXPECTED_ANSWER)
        calls_tot += r["calls"]
    return ok / n_runs, calls_tot / n_runs


def main():
    print("=" * 68)
    print("nano-agentscope L1 — real tiny model + real tools + harness")
    print("=" * 68)
    print(f"python {sys.version.split()[0]} | torch {torch.__version__}")
    print("declarations: TinyReActLM = real ~94K-param char-LM (memorizes,")
    print("  does not reason); Playback/FaultModel = declared fault vectors;")
    print("  real hosted LLM path needs a key [TODO: needs key].")

    tools = make_tools()

    # ------------------------------------------------------------------ [0]
    print("\n[0] train TinyReActLM (synthetic trajectories, real file obs)")
    rng = random.Random(SEED)
    dir_obs = list_dir()
    read_obs = read_file("corpus.txt")
    transcripts = make_transcripts(rng, dir_obs, read_obs)
    n_clean = len(transcripts)
    transcripts += make_critique_transcripts(rng, dir_obs, read_obs, tools)
    mean_len = sum(len(s) for s in transcripts) / len(transcripts)
    model, stoi, itos, UNK, PAD, loss, nparams, secs = train_tiny_lm(transcripts)
    print(f"    transcripts={len(transcripts)} ({n_clean} clean + "
          f"{len(transcripts) - n_clean} violation->critique->repair) "
          f"mean_len={mean_len:.0f} chars | vocab={len(stoi)}+UNK+PAD")
    print(f"    params={nparams:,} | final loss={loss:.4f} | "
          f"train time={secs:.1f}s (CPU)")

    # ------------------------------------------------------------------ [1]
    print("\n[1] greedy decode: real model x real tools, full task")
    lm = TinyReActLM(model, stoi, itos, UNK, PAD, temperature=0.0)
    res = Harness(tools, max_steps=6, max_retries=1).run(lm, TASK,
                                                         verbose=True)
    print(f"    answer={res['answer']!r} | status={res['status']} | "
          f"model calls={res['calls']}")
    assert res["answer"] == EXPECTED_ANSWER, res
    assert res["status"] == "answered" and res["calls"] == 3
    for bad in ["../../etc/passwd", "no_such_file.txt"]:
        try:
            read_file(bad)
            raise AssertionError(f"sandbox failed for {bad}")
        except (PermissionError, FileNotFoundError) as e:
            print(f"    sandbox check {bad!r}: blocked -> "
                  f"{type(e).__name__}")

    # ------------------------------------------------------------------ [2]
    print("\n[2] model-side reality: format compliance vs temperature")
    print("    (first block from task prefix, 200 samples each, seeded)")
    prefix = f"Task: {TASK}\nTools: list_dir, read_file\n"
    sweep = {}
    for T in [0.3, 0.7, 1.0, 1.3]:
        counts = {}
        for i in range(200):
            block = generate(model, stoi, itos, prefix, temperature=T,
                             max_new=250, seed=1000 + i, unk=UNK, pad=PAD)
            kind, _ = parse_block(block, tools)
            counts[kind] = counts.get(kind, 0) + 1
        ok = counts.get("compliant", 0) + counts.get("final", 0)
        sweep[T] = ok / 200
        detail = ", ".join(f"{k}:{v}" for k, v in sorted(counts.items()))
        print(f"    T={T}: compliant {ok}/200 = {ok / 200:.1%}   [{detail}]")
    assert sweep[0.3] >= 0.95 and sweep[0.3] > sweep[1.3] + 0.5
    p_real = 1.0 - sweep[0.7]
    print(f"    => a real model gives per-call failure p ~= {p_real:.2f} "
          f"at T=0.7. This p is what the harness has to live with.")

    # ------------------------------------------------------------------ [3]
    print("\n[3] harness defenses under declared fault injection")
    # a) bad_json -> critique -> retry recovers
    bad_json_block = SCRIPT[0].replace("Action Input: {}",
                                       "Action Input: {broken: json}")
    r = Harness(tools, max_retries=1).run(
        Playback([bad_json_block, SCRIPT[0], SCRIPT[1], SCRIPT[2]]), TASK)
    kinds = [t["kind"] for t in r["traj"] if t["role"] == "model"]
    print(f"    a) bad_json -> critique -> retry: status={r['status']} | "
          f"kinds={kinds} | calls={r['calls']}")
    assert r["status"] == "answered" and kinds[0] == "bad_json" \
        and "compliant(retry1)" in kinds
    # b) unknown tool (typo) -> critique names the violation -> fixed
    typo_block = SCRIPT[1].replace("read_file", "read_flie")
    r = Harness(tools, max_retries=1).run(
        Playback([SCRIPT[0], typo_block, SCRIPT[1], SCRIPT[2]]), TASK)
    kinds = [t["kind"] for t in r["traj"] if t["role"] == "model"]
    print(f"    b) unknown_tool 'read_flie' -> critique(unknown_tool: "
          f"read_flie) -> retry: status={r['status']} | kinds={kinds}")
    assert r["status"] == "answered" and kinds[1] == "unknown_tool"
    # c) tool exception (real FileNotFoundError) becomes an observation
    wrong_file = SCRIPT[1].replace("corpus.txt", "no_such.txt")
    r = Harness(tools, max_retries=0, max_steps=8).run(
        Playback([SCRIPT[0], wrong_file, SCRIPT[1], SCRIPT[2]]), TASK)
    errs = [t["text"] for t in r["traj"]
            if t["role"] == "tool" and t["text"].startswith("error")]
    print(f"    c) tool exception as observation: {errs[0][:72]!r}")
    assert r["status"] == "answered" and len(errs) == 1
    # d) loop guard — same action twice in a row
    r = Harness(tools, max_retries=0).run(
        Playback([SCRIPT[0], SCRIPT[0], SCRIPT[2]]), TASK)
    print(f"    d) loop (same action twice): status={r['status']} "
          f"(guard fired, budget saved)")
    assert r["status"] == "loop_detected"

    # ------------------------------------------------------------------ [4]
    print("\n[4] reliability algebra: measured task success vs formula")
    n_steps = 3
    # 4a: organic — the real tiny model at T=0.7, harness retry on/off
    organic = {}
    for k in [0, 1]:
        ok, mc = run_batch(
            lambda i: TinyReActLM(model, stoi, itos, UNK, PAD, temperature=0.7,
                                  seed_base=50000 + 37 * i),
            200, max_retries=k)
        organic[k] = (ok, mc)
        print(f"    organic (real LM, T=0.7): retries={k} -> success "
              f"{ok:.1%} | mean calls {mc:.2f}")
    assert organic[1][0] >= organic[0][0]
    # 4b: controlled iid sweep vs closed form  q = [1 - p^(k+1)]^n
    # (p = per-call FAILURE rate: a step passes if any of its k+1
    #  attempts is compliant; the task needs all n steps to pass)
    print("    controlled iid (FaultModel, 400 runs each):")
    print("     p     k   measured   formula   |diff|")
    max_dev = 0.0
    for p in [0.05, 0.10, 0.20, 0.30]:
        for k in [0, 1, 2]:
            ok, _ = run_batch(
                lambda i, p=p: FaultModel(SCRIPT, p=p, seed=10000 + i),
                400, max_retries=k)
            formula = (1 - p ** (k + 1)) ** n_steps
            max_dev = max(max_dev, abs(ok - formula))
            print(f"    {p:.2f}  {k}    {ok:6.1%}    {formula:6.1%}   "
                  f"{abs(ok - formula):5.1%}")
    assert max_dev < 0.08, max_dev
    # 4c: sticky (correlated) failures defeat naive retry arithmetic
    p, s, k = 0.20, 0.75, 1
    ok_sticky, _ = run_batch(
        lambda i: FaultModel(SCRIPT, p=p, sticky=s, seed=20000 + i),
        400, max_retries=k)
    f_iid = (1 - p ** (k + 1)) ** n_steps
    f_sticky = (1 - p * (s + (1 - s) * p)) ** n_steps
    print(f"    sticky s={s} @ p={p}, k=1: measured {ok_sticky:.1%} vs "
          f"iid formula {f_iid:.1%} (sticky formula {f_sticky:.1%})")
    assert ok_sticky < f_iid - 0.05

    # ------------------------------------------------------------------ [5]
    print("\n[5] cost ledger: reliability is bought with model calls")
    print("     p     k   success   mean_calls   calls/success")
    for p in [0.10, 0.30]:
        for k in [0, 1, 2]:
            ok, mc = run_batch(
                lambda i, p=p: FaultModel(SCRIPT, p=p, seed=30000 + i),
                400, max_retries=k)
            eff = mc / ok if ok > 0 else float("inf")
            print(f"    {p:.2f}  {k}    {ok:6.1%}     {mc:6.2f}       "
                  f"{eff:6.2f}")
    print("    (each extra call = tokens = latency = $; retries consume the")
    print("     throughput headroom nano-vllm-sglang works so hard for; and")
    print("     every recorded trajectory is itself training data -> 03/01)")

    # ------------------------------------------------------------------ [6]
    print("\n[6] real-API path: OpenAI-compatible client vs local contract "
          "server")
    result, captured = verify_api_client_against_local_server(tools)
    print(f"    POST {captured['path']} | auth header present: "
          f"{captured['auth'].startswith('Bearer ')}")
    print(f"    request model={captured['body']['model']!r} | response "
          f"parsed -> agent status={result['status']}, "
          f"answer={result['answer']!r}")
    assert captured["path"] == "/compatible-mode/v1/chat/completions"
    assert result["answer"] == EXPECTED_ANSWER
    assert captured["auth"] == "Bearer test-key-redacted"
    print("    client code verified against the JSON contract locally.")
    print("    real endpoint (DASHSCOPE_API_KEY / OPENAI_API_KEY): "
          "[TODO: needs key]")

    # ------------------------------------------------------------------ end
    print("\n" + "=" * 68)
    print("✅ self-check passed:")
    print("   greedy real-model trajectory correct (3 calls) /")
    print("   sandbox blocks traversal & missing files /")
    print("   compliance falls with temperature (measured) /")
    print("   defenses recover or abort under fault injection /")
    print(f"   measured success matches iid formula (max dev {max_dev:.1%})"
          " / sticky < iid /")
    print("   API client round-trips through real HTTP")
    print("=" * 68)
    print("\ntakeaway: L0's mock never failed, so the harness had nothing")
    print("          to do. Put a real distribution behind the loop and the")
    print("          harness BECOMES the product: parse-validate, retry with")
    print("          critique, tool errors as observations, loop guards, and")
    print("          a budget — reliability you can measure and price.")


if __name__ == "__main__":
    main()
