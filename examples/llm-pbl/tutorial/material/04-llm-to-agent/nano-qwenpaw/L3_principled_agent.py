#!/usr/bin/env python3
"""
nano-qwenpaw L3 — the principled agent: skills as data, assembly under SOUL
============================================================================
L0 wrapped one call in a harness. L1 made the harness live across turns
under a finite window. L2 turned the methodology into executable flows —
but the flows were still HARDCODED in the harness script. L3 closes the
last step: the methodology becomes DATA. A skill is a directory bearing
SKILL.md; a builder discovers, filters and injects skills PER REQUEST;
and the seven SOUL principles govern not only execution but ASSEMBLY
itself — which capability reaches which request is a decision made under
the same principles that run the session.

What is reproduced (mirrors, stdlib-only):
  * registry.resolve_effective_skills  (manifest: enabled + channels)
  * builder._resolve_skill_loader_dirs (the SKILL.md admission gate,
                                        with the exact log message)
  * builder.build_toolkit's injection  (skills_or_loaders — skills are
                                        prompt-level capabilities, NOT
                                        tools on the tools= list)
  * store.get_workspace_skills_dir    (skills/ preferred, legacy skill/
                                        renamed in place)
  * a SOUL-governed session in which all seven principles take a form:
    #1 K+1, #2 Feynman, #4 Examiner-B, #5 claims gate, #7 ledger were
    flows in L2; #3 PBL and #6 learner autonomy are new here, and the
    capability itself (gap categories, mastery bands, thresholds) is
    parsed OUT OF THE INJECTED SKILL TEXTS, not out of harness memory.

Declarations (ROADMAP §3 contract):
  * The workspace under tempfile is a DECLARED test fixture (removed at
    the end of a successful run; left in place on failure for forensics).
    Two skill documents (k-plus-one, feynman-check) are copied VERBATIM
    from the real coach profile at runtime (byte-identity asserted by
    sha256[:8]); the others are synthetic edge-case plantings, each labeled.
  * The base model, Examiner-A and the gap detectors are DECLARED
    deterministic stand-ins; in a real system the LLM's judgment sits
    there: [TODO: needs key]. Everything structural is real: real file
    I/O, a real JSON manifest, real frontmatter parsing, a real sqlite
    ledger, live sha256 of every source.
  * The feynman-check entry carries channels=["console"] in the fixture
    manifest — a DECLARED divergence from the real coach manifest (where
    it is ["all"]) — planted so one workspace yields two different
    agents per request channel.

Authoritative-source anchors verified 2026-08-10 (line numbers re-derived
live at runtime in section [0]):
  src/qwenpaw/runtime/builder.py        L2 per-request assembly docstring,
                                        L94 Toolkit(tools=...,
                                        skills_or_loaders=skill_dirs),
                                        L97-121 _resolve_skill_loader_dirs,
                                        L116-120 the not-injected log
  src/qwenpaw/agents/skill_system/registry.py
                                        L1186-1201 resolve_effective_skills
  src/qwenpaw/agents/skill_system/store.py
                                        L65-76 get_workspace_skills_dir
  coach/profile/SOUL.md                 L5-L69 seven principles
  coach/profile/skills/{k-plus-one,feynman-check}/SKILL.md
  coach/profile/skill.json              real workspace manifest
  (+ the four scroll sources carried over from L1/L2)

Dependencies: Python stdlib only + import of L2_real_methodology_loop.py
(same directory; no pyc written — sys.dont_write_bytecode is set before
the import). Run:  python L3_principled_agent.py
Output is fully deterministic (no sampling, no timing lines): two runs on
the same source snapshot are byte-identical.
"""

import copy
import hashlib
import json
import re
import shutil
import sqlite3
import sys
import tempfile
from pathlib import Path

sys.dont_write_bytecode = True          # must precede the L2 import
import L2_real_methodology_loop as L2   # noqa: E402  (est_tokens, sha8,
#                                       lineno_of, parse_rules, oracle,
#                                       examiner_b, LearnerModel)


def level_of(mastery: float) -> int:    # declared mapping (mirror of L2)
    return int(mastery * 10 + 1e-9)


# ---------------------------------------------------------------------------
# sources: the six coach sources of L1/L2 + the skills-architecture three
# ---------------------------------------------------------------------------
HERE = Path(__file__).resolve()
REPO_ROOT = next((p for p in HERE.parents if (p / "src/qwenpaw").is_dir()), HERE.parents[5])

ARCH_SOURCES = {
    "builder.py":  REPO_ROOT / "src/qwenpaw/runtime/builder.py",
    "registry.py": REPO_ROOT / "src/qwenpaw/agents/skill_system/registry.py",
    "store.py":    REPO_ROOT / "src/qwenpaw/agents/skill_system/store.py",
}
REAL_MANIFEST   = REPO_ROOT / "coach/profile/skill.json"
REAL_SKILLS_DIR = REPO_ROOT / "coach/profile/skills"

# pinned snapshot slices: verbatim substrings captured 2026-08-10, used ONLY
# if a live read fails (the run prints which mode it used). The six coach
# sources reuse L2's pinned set; the three architecture sources pin the
# exact lines this section anchors on.
PINNED_ARCH = {
    "builder.py": (
        '"""Per-request agent assembly.\n'
        '    """Compose an agent for each request.\n'
        "        return Toolkit(tools=tools, skills_or_loaders=skill_dirs)\n"
        "    def _resolve_skill_loader_dirs(\n"
        '                    "skill \'%s\' has no SKILL.md at %s; not injected",\n'
    ),
    "registry.py": (
        "def resolve_effective_skills(\n"
        '        channels = entry.get("channels") or ["all"]\n'
        '        if "all" in channels or channel_name in channels:\n'
    ),
    "store.py": (
        "def get_workspace_skills_dir(workspace_dir: Path) -> Path:\n"
        '    legacy = workspace_dir / "skill"\n'
        "            legacy.rename(preferred)\n"
    ),
}

# the two real skill documents copied verbatim into the fixture workspace
VERBATIM_SKILLS = {
    "k-plus-one":    REPO_ROOT / "coach/profile/skills/k-plus-one/SKILL.md",
    "feynman-check": REPO_ROOT / "coach/profile/skills/feynman-check/SKILL.md",
}

# synthetic SKILL.md bodies for the fixture (declared plantings)
SYNTHETIC_SKILLS = {
    "daily-review": (
        "---\n"
        "name: daily-review\n"
        "description: \"End-of-day review: summarize sessions, flag "
        "stalled topics, propose tomorrow's K+1 target.\"\n"
        "version: 1.0.0\n"
        "---\n"
        "\n"
        "# Daily Review\n"
        "\n"
        "Review today's sessions from `.local/logs/`. For each topic "
        "studied, record the mastery delta and flag any topic attempted "
        "3+ times without progress (SOUL.md principle 7). Propose "
        "tomorrow's single K+1 target — one target, not a list.\n"
    ),
    "voice-brief": (
        "---\n"
        "name: voice-brief\n"
        "description: \"Voice-channel briefing: answer in at most three "
        "short spoken sentences; defer deep verification to the console "
        "channel.\"\n"
        "version: 1.0.0\n"
        "---\n"
        "\n"
        "# Voice Brief\n"
        "\n"
        "The voice channel cannot display tables or verification trails. "
        "Answer in at most three short sentences. If the request needs "
        "adversarial verification, say so and defer it to the console "
        "channel — do not silently skip the check.\n"
    ),
    "codex-delegate": (
        "---\n"
        "name: codex-delegate\n"
        "description: \"Delegate a coding task to an external coding "
        "agent (fixture planting: on disk, but never listed in the "
        "manifest).\"\n"
        "version: 0.1.0\n"
        "---\n"
        "\n"
        "# Codex Delegate\n"
        "\n"
        "Hand the coding task to the external agent and report back its "
        "diff summary. (This fixture skill exists on disk but is absent "
        "from skill.json, mirroring the real coach profile.)\n"
    ),
}


def load_arch_sources():
    texts, shas, mode = {}, {}, []
    for name, path in ARCH_SOURCES.items():
        try:
            b = path.read_bytes()
            texts[name] = b.decode("utf-8", errors="replace")
            shas[name] = L2.sha8(b)
            mode.append("live")
        except OSError:
            texts[name] = PINNED_ARCH[name]
            shas[name] = L2.sha8(PINNED_ARCH[name].encode())
            mode.append("PINNED")
    return texts, shas, mode


def load_verbatim_skills():
    """Copy the two real skill documents; fall back to L2's pinned slices."""
    bodies, shas, mode = {}, {}, []
    for name, path in VERBATIM_SKILLS.items():
        try:
            b = path.read_bytes()
            bodies[name] = b.decode("utf-8", errors="replace")
            shas[name] = L2.sha8(b)
            mode.append("live")
        except OSError:
            key = name + ".md"
            bodies[name] = L2.PINNED[key]
            shas[name] = L2.sha8(L2.PINNED[key].encode())
            mode.append("PINNED")
    return bodies, shas, mode


def load_real_profile():
    """The real coach profile's funnel, parsed live (names + counts only)."""
    try:
        manifest = json.loads(REAL_MANIFEST.read_text())
        entries = {n: e for n, e in manifest.get("skills", {}).items()}
        enabled = sorted(n for n, e in entries.items()
                         if e.get("enabled", False))
        dirs = sorted(p.name for p in REAL_SKILLS_DIR.iterdir() if p.is_dir())
        return dirs, enabled, "live"
    except (OSError, ValueError):
        # pinned snapshot of the real profile, captured 2026-08-10
        dirs = ["checkup", "codex-delegate", "daily-review", "feynman-check",
                "k-plus-one", "meta-review", "onboard", "progress-tracker",
                "spaced-repetition"]
        enabled = ["checkup", "daily-review", "feynman-check", "k-plus-one",
                   "meta-review", "onboard", "progress-tracker",
                   "spaced-repetition"]
        return dirs, enabled, "PINNED"


# ---------------------------------------------------------------------------
# mirrors of the skills architecture (registry.py / builder.py / store.py)
# ---------------------------------------------------------------------------
def get_workspace_skills_dir(ws: Path) -> tuple:
    """mirror of store.py L65-76: skills/ preferred, legacy skill/ renamed."""
    preferred = ws / "skills"
    legacy = ws / "skill"
    if preferred.exists():
        return preferred, False
    if legacy.exists():
        try:
            legacy.rename(preferred)          # L73
        except OSError:
            return legacy, False
        return preferred, True
    return preferred, False


def resolve_effective_skills(ws: Path, channel: str) -> list:
    """mirror of registry.py L1186-1201 (manifest: enabled + channels)."""
    manifest = json.loads((ws / "skill.json").read_text())
    resolved = []
    for name, entry in sorted(manifest.get("skills", {}).items()):
        if not entry.get("enabled", False):
            continue
        channels = entry.get("channels") or ["all"]
        if "all" in channels or channel in channels:
            if (get_workspace_skills_dir(ws)[0] / name).exists():
                resolved.append(name)
    return resolved


def resolve_skill_loader_dirs(ws: Path, effective: list) -> tuple:
    """mirror of builder.py L97-121: the SKILL.md admission gate."""
    base = get_workspace_skills_dir(ws)[0]
    dirs, not_injected = [], []
    for name in effective:
        skill_dir = base / name
        if (skill_dir / "SKILL.md").exists():
            dirs.append(name)
        else:
            # builder.py L116-120 logs exactly this (workspace-relative
            # path here, so the output does not leak the tempfile dir):
            not_injected.append(
                "skill '%s' has no SKILL.md at %s; not injected"
                % (name, Path("skills") / name))
    return dirs, not_injected


def funnel(ws: Path, channel: str) -> dict:
    """the whole resolution funnel with a reason per manifest entry."""
    manifest = json.loads((ws / "skill.json").read_text())
    base = get_workspace_skills_dir(ws)[0]
    steps, effective = [], []
    for name, entry in sorted(manifest.get("skills", {}).items()):
        if not entry.get("enabled", False):
            steps.append((name, "DISABLED"))
            continue
        channels = entry.get("channels") or ["all"]
        if not ("all" in channels or channel in channels):
            steps.append((name, "CHANNEL[%s]" % ",".join(channels)))
            continue
        if not (base / name).exists():
            steps.append((name, "NO-DIR"))
            continue
        effective.append(name)
        steps.append((name, "PASS"))
    injected, not_injected = resolve_skill_loader_dirs(ws, effective)
    return dict(steps=steps, effective=effective,
                injected=injected, not_injected=not_injected)


def parse_frontmatter(text: str) -> dict:
    """stdlib YAML-frontmatter reader (the real frontmatter is flat
    key: value; no yaml dependency needed)."""
    m = re.match(r"^---\n(.*?)\n---\n", text, re.S)
    if not m:
        return {}
    fm = {}
    for line in m.group(1).splitlines():
        k, _, v = line.partition(":")
        fm[k.strip()] = v.strip().strip('"')
    return fm


def build_prompt(soul_text: str, injected: list, bodies: dict) -> tuple:
    """SOUL verbatim + one block per injected skill (verbatim body).
    Skills ride the system prompt — skills_or_loaders, not tools=."""
    parts = [soul_text.rstrip("\n"), ""]
    parts.append("# Active skills (injected per-request; "
                 "verbatim from SKILL.md)")
    blocks = {}
    for name in injected:
        fm = parse_frontmatter(bodies[name])
        block = ("\n## skill: %s (version %s)\n> %s\n\n%s"
                 % (name, fm.get("version", "?"),
                    fm.get("description", ""), bodies[name].rstrip("\n")))
        blocks[name] = block
        parts.append(block)
    return "\n".join(parts) + "\n", blocks


# ---------------------------------------------------------------------------
# fixture workspace (declared test fixture under tempfile)
# ---------------------------------------------------------------------------
def build_workspace(verbatim: dict) -> Path:
    ws = Path(tempfile.mkdtemp(prefix="nano_qwenpaw_L3_"))
    skills = ws / "skills"
    for name, body in sorted(verbatim.items()):       # real, verbatim
        d = skills / name
        d.mkdir(parents=True, exist_ok=True)
        (d / "SKILL.md").write_text(body)
    for name, body in sorted(SYNTHETIC_SKILLS.items()):
        if name == "codex-delegate":
            continue                                  # handled below
        d = skills / name
        d.mkdir(parents=True, exist_ok=True)
        (d / "SKILL.md").write_text(body)
    # planted: enabled + channel-passing, but NO SKILL.md (only a README)
    # — this is the directory that reaches the gate and gets gated out
    ob = skills / "onboard"
    ob.mkdir(parents=True, exist_ok=True)
    (ob / "README.txt").write_text("onboarding notes (no SKILL.md)\n")
    # planted: on disk with SKILL.md, but absent from the manifest —
    # mirrors codex-delegate in the real coach profile.
    cd = skills / "codex-delegate"
    cd.mkdir(parents=True, exist_ok=True)
    (cd / "SKILL.md").write_text(SYNTHETIC_SKILLS["codex-delegate"])
    manifest = {
        "schema_version": "workspace-skill-manifest.v1",   # verbatim key
        "skills": {
            "checkup":       {"enabled": False, "channels": ["all"]},
            "daily-review":  {"enabled": True,  "channels": ["all"]},
            "feynman-check": {"enabled": True,  "channels": ["console"]},
            "k-plus-one":    {"enabled": True,  "channels": ["all"]},
            "onboard":       {"enabled": True,  "channels": ["all"]},
            "voice-brief":   {"enabled": True,  "channels": ["voice"]},
        },
    }
    (ws / "skill.json").write_text(json.dumps(manifest, indent=2,
                                              sort_keys=True) + "\n")
    return ws


def build_legacy_workspace() -> Path:
    ws = Path(tempfile.mkdtemp(prefix="nano_qwenpaw_L3_legacy_"))
    d = ws / "skill" / "daily-review"                   # legacy singular dir
    d.mkdir(parents=True)
    (d / "SKILL.md").write_text(SYNTHETIC_SKILLS["daily-review"])
    return ws


# ---------------------------------------------------------------------------
# the feynman review, driven by the INJECTED skill text (L3's point):
# categories and bands are parsed out of whatever reached this request.
# Detectors are declared heuristics — real LLM judgment sits there:
# [TODO: needs key] — except the factual check, which cross-references
# the live source (real evidence, as in L2).
# ---------------------------------------------------------------------------
EXPL = ("The dashboard caps a single oversized tool result: the "
        "in-context copy is replaced by a bounded preview plus a recall "
        "pointer keyed by the session id, so the dashboard never loses "
        "data. The pointer lets the model call FTS5 later to expand the "
        "result.")


def feynman_gaps_l3(text: str, categories: list, cap_src: str) -> dict:
    gaps = {c: [] for c in categories}
    for cat in categories:
        if cat == "Logical Leaps":
            m = re.search(r"\bso\b[^.!?]*(?:never loses|no data loss)", text)
            if m:
                before = text[:m.start()]
                if not re.search(r"(?i)(?:persist\w*|writ\w+)[^.!?]*\bbefore\b"
                                 r"|\bbefore\b[^.!?]*(?:persist\w*|replac\w*)",
                                 before):
                    gaps[cat].append(m.group(0).strip()
                                     + " — missing premise: persisted BEFORE replace")
        elif cat == "Undefined Terms":
            if "FTS5" in text and not re.search(r"FTS5\s*(?:—|--|,)\s*\S", text) \
                    and not re.search(r"(?i)\b(?:means|is a|stands for)\b[^.!?]*FTS5",
                                      text):
                gaps[cat].append("FTS5 — used but not explained")
        elif cat == "Factual Errors":
            claim = re.search(r"keyed by the session id", text)
            if claim:
                ev = re.search(r"recall pointer keyed by\s*``tool_call_id``",
                               cap_src)
                if ev:
                    quote = re.sub(r"\s+", " ", ev.group(0))
                    gaps[cat].append("'keyed by the session id' — contradicts "
                                     "cap_middleware.py: \"%s\"" % quote)
        elif cat == "Missing Aspects":
            if not re.search(r"(?i)write fail\w*|durab\w+[^.!?]*degrad\w+"
                             r"|degrad\w+", text):
                gaps[cat].append("degradation path — not covered")
    return gaps


# ---------------------------------------------------------------------------
# the PBL problem set (declared Examiner-A; difficulty K+1 = 3, concepts
# reuse L2's oracle vocabulary so "solve it yourself" stays exact)
# ---------------------------------------------------------------------------
def make_pbl_set(cap: int) -> list:
    D = L2.K_LEARNER + 1
    return [
        dict(pid="P1", concept="seq-span", kind="computational", difficulty=D,
             lo=12, hi=33,
             stem="your cap-dashboard audits tool spans [12, 33]; how many "
                  "seqs in the inclusive span?", key=21),   # planted: hi-lo
        dict(pid="P2", concept="seq-span", kind="application", difficulty=D,
             lo=1, hi=25,
             stem=f"your dashboard window holds seqs [1, 25]; token_cap "
                  f"defaults to {{token_cap}} tokens — how many seqs in "
                  f"the span?", key=25),
        dict(pid="P3", concept="odd-sum", kind="conceptual", difficulty=D,
             n=6,
             stem="the dashboard's savings counter sums the first 6 odd "
                  "numbers; what does it show?", key=36),
    ]


# ---------------------------------------------------------------------------
def main():
    print("=" * 68)
    print("nano-qwenpaw L3 — the principled agent: skills as data,")
    print("assembly under SOUL")
    print("=" * 68)
    print(f"python {sys.version.split()[0]}")
    print("declarations: fixture workspace under tempfile (two skill docs")
    print("  copied verbatim from the coach profile, byte-identity checked")
    print("  by sha256[:8]; the rest are labeled plantings); base model,")
    print("  Examiner-A and gap detectors = declared deterministic")
    print("  stand-ins — real LLM judgment sits there: [TODO: needs key].")
    print("  Real: file I/O, JSON manifest, frontmatter parsing, sqlite")
    print("  ledger, live sha256 of every source. feynman-check carries")
    print("  channels=[console] here — a declared divergence from the real")
    print("  manifest (which says [all]) — so one workspace yields two")
    print("  different agents, one per request channel.")

    texts6, shas6, mode6 = L2.load_sources()
    rules = L2.parse_rules(texts6)
    arch, arch_shas, arch_mode = load_arch_sources()
    verb, verb_shas, verb_mode = load_verbatim_skills()

    # ------------------------------------------------------------------ [0]
    print("\n[0] sources & freshness (line anchors re-derived live)")
    for i, name in enumerate(L2.SOURCES):
        print(f"    {name:<18} sha256[:8]={shas6[name]}  mode={mode6[i]}")
    for i, name in enumerate(ARCH_SOURCES):
        print(f"    {name:<18} sha256[:8]={arch_shas[name]}  "
              f"mode={arch_mode[i]}")
    b = arch["builder.py"]
    b_skills = L2.lineno_of(b, r'skills_or_loaders=skill_dirs')
    b_resolve = L2.lineno_of(b, r'def _resolve_skill_loader_dirs')
    b_log = L2.lineno_of(b, r'has no SKILL\.md at')
    r = arch["registry.py"]
    r_resolve = L2.lineno_of(r, r'def resolve_effective_skills')
    r_channel = L2.lineno_of(r, r'in channels or channel_name in channels')
    s = arch["store.py"]
    s_getdir = L2.lineno_of(s, r'def get_workspace_skills_dir')
    s_rename = L2.lineno_of(s, r'legacy\.rename\(preferred\)')
    derived = {"skills_or_loaders": b_skills,
               "_resolve_skill_loader_dirs": b_resolve,
               "not-injected log": b_log,
               "resolve_effective_skills": r_resolve,
               "channel test": r_channel,
               "get_workspace_skills_dir": s_getdir,
               "legacy rename": s_rename}
    for anchor_name, ln in derived.items():
        assert ln > 0, (f"anchor '{anchor_name}' failed to match — a "
                        f"double-escaped regex here once printed L0 silently")
    print(f"    builder.py anchors: skills_or_loaders L{b_skills}"
          f" / _resolve_skill_loader_dirs L{b_resolve}"
          f" / not-injected log L{b_log}")
    print(f"    registry.py anchors: resolve_effective_skills L{r_resolve}"
          f" / channel test L{r_channel}")
    print(f"    store.py anchors: get_workspace_skills_dir L{s_getdir}"
          f" / legacy rename L{s_rename}")

    # ------------------------------------------------------------------ [1]
    print("\n[1] the real coach profile, parsed live: the funnel starts at")
    print("    the manifest, not at the directory listing")
    real_dirs, real_enabled, real_mode = load_real_profile()
    print(f"    mode={real_mode}: {len(real_dirs)} skill dirs on disk vs "
          f"{len(real_enabled)} enabled manifest entries")
    missing = sorted(set(real_dirs) - set(real_enabled))
    for name in missing:
        print(f"    {name:<16} on disk WITH SKILL.md, absent from the "
              f"manifest -> never effective")
    print("    (a directory is not a skill: enablement lives in skill.json)")

    # ------------------------------------------------------------------ [2]
    print("\n[2] fixture workspace (declared): verbatim copies + plantings")
    ws = build_workspace(verb)
    manifest = json.loads((ws / "skill.json").read_text())
    for name, entry in sorted(manifest["skills"].items()):
        ch = entry["channels"]
        print(f"    manifest  {name:<15} enabled={str(entry['enabled']):<5} "
              f"channels={ch}")
    on_disk = sorted(p.name for p in (ws / "skills").iterdir() if p.is_dir())
    for name in sorted(set(on_disk) - set(manifest["skills"])):
        print(f"    disk-only {name:<15} has SKILL.md, NOT in the manifest")
    for name in sorted(verb):
        copy_sha = L2.sha8(((ws / "skills") / name / "SKILL.md").read_bytes())
        print(f"    verbatim  {name:<15} workspace copy sha256[:8]={copy_sha} "
              f"== coach source {verb_shas[name]}")
        assert copy_sha == verb_shas[name]

    # ------------------------------------------------------------------ [3]
    print("\n[3] resolution funnel per request channel (registry + builder")
    print("    mirrors; same workspace, one build per request)")
    funnels = {}
    for channel in ("console", "voice"):
        f = funnel(ws, channel)
        funnels[channel] = f
        print(f"    channel={channel}:")
        for name, status in f["steps"]:
            if status == "PASS":
                print(f"      {name:<15} -> effective")
            elif status == "DISABLED":
                print(f"      {name:<15} DROP  (manifest: disabled)")
            elif status.startswith("CHANNEL"):
                print(f"      {name:<15} DROP  (manifest: {status.lower()})")
            else:
                print(f"      {name:<15} DROP  ({status})")
        for line in f["not_injected"]:
            print(f"      {line}")
        print(f"      -> effective={len(f['effective'])}, injected="
              f"{len(f['injected'])}: {', '.join(f['injected'])}")
    assert funnels["console"]["injected"] == \
        ["daily-review", "feynman-check", "k-plus-one"]
    assert funnels["voice"]["injected"] == \
        ["daily-review", "k-plus-one", "voice-brief"]

    # ------------------------------------------------------------------ [4]
    print("\n[4] frontmatter contract (name must equal the directory name)")
    for name in funnels["console"]["injected"]:
        body = ((ws / "skills") / name / "SKILL.md").read_text()
        fm = parse_frontmatter(body)
        ok = fm.get("name") == name
        print(f"    {name:<15} name={fm.get('name'):<15} "
              f"version={fm.get('version')}  "
              f"{'OK' if ok else 'MISMATCH'}")
        assert ok
    # planted counter-example: a frontmatter name that lies about its dir
    bad_dir = ws / "skills" / "misnamed"
    bad_dir.mkdir()
    (bad_dir / "SKILL.md").write_text(
        "---\nname: other-name\ndescription: \"planted mismatch\"\n"
        "version: 0.0.1\n---\n\n# Misnamed\n")
    fm_bad = parse_frontmatter((bad_dir / "SKILL.md").read_text())
    print(f"    {'misnamed':<15} name={fm_bad.get('name'):<15} -> would "
          f"fail the contract (planted; never enabled, never injected)")

    # ------------------------------------------------------------------ [5]
    print("\n[5] prompt assembly: skills ride the system prompt, verbatim")
    soul = texts6["SOUL.md"]
    bodies = {n: ((ws / "skills") / n / "SKILL.md").read_text()
              for n in sorted(set(funnels["console"]["injected"])
                              | set(funnels["voice"]["injected"]))}
    prompts, blocks = {}, {}
    for channel in ("console", "voice"):
        p, blk = build_prompt(soul, funnels[channel]["injected"], bodies)
        prompts[channel], blocks[channel] = p, blk
        print(f"    channel={channel}: SOUL {L2.est_tokens(soul)} + skills "
              f"{sum(L2.est_tokens(v) for v in blk.values())} est-tokens "
              f"-> prompt {L2.est_tokens(p)} est-tokens "
              f"({len(funnels[channel]['injected'])} skills)")
    only_console = sorted(set(funnels["console"]["injected"])
                          - set(funnels["voice"]["injected"]))
    only_voice = sorted(set(funnels["voice"]["injected"])
                        - set(funnels["console"]["injected"]))
    delta = (L2.est_tokens(prompts["console"])
             - L2.est_tokens(prompts["voice"]))
    block_delta = (L2.est_tokens(blocks["console"]["feynman-check"])
                   - L2.est_tokens(blocks["voice"]["voice-brief"]))
    print(f"    prompt delta: console - voice = {delta} est-tokens; "
          f"block-only estimate = {block_delta} (whole-prompt integer "
          f"rounding may differ by 1)")
    print(f"    provenance (principle#5 at assembly time):")
    for channel in ("console", "voice"):
        prov = " ".join("%s@%s" % (n, L2.sha8(bodies[n].encode()))
                        for n in funnels[channel]["injected"])
        print(f"      {channel}: SOUL@{shas6['SOUL.md']} + {prov}")
    assert only_console == ["feynman-check"] and only_voice == ["voice-brief"]

    # ------------------------------------------------------------------ [6]
    M0 = 0.25                       # declared initial mastery (fixture)
    print("\n[6] one SOUL-governed session (channel=console; learner mastery")
    print(f"    {M0:.2f} on 'token-budgeting', project 'cap-dashboard')")
    m = M0
    cap = rules["token_cap"]
    print(f"    [#1 K+1]      d = level({m:.2f})+1 = {level_of(m) + 1} "
          f"(rule parsed from the injected k-plus-one SKILL.md)")
    print(f"    [#6 autonomy] harness suggests next topic 'feynman practice'")
    print(f"                  learner declines: 'stay on token-budgeting — "
          f"my dashboard ships this week'")
    print(f"                  -> discussed, not overridden; mastery stays "
          f"{m:.2f}; suggestion logged declined")
    assert m == M0

    problems = make_pbl_set(cap)
    gate = L2.examiner_b(copy.deepcopy(problems), rules, independent=True)
    bad = [pid for pid, f in gate["fails"].items() if f]
    for pid in bad:
        for p in problems:
            if p["pid"] == pid:
                p["key"] = L2.oracle(p)
    gate2 = L2.examiner_b(copy.deepcopy(problems), rules, independent=True)
    print(f"    [#3 PBL]      3 problems at d={L2.K_LEARNER + 1}, stems seeded "
          f"with the learner's project (cap-dashboard) + live token_cap={cap}")
    print(f"    [#4 ExaminerB] planted defect {bad[0]} (key=hi-lo instead of "
          f"hi-lo+1) -> failures={gate['n_bad']} <= {rules['regen_thresh']} "
          f"-> fix in place; re-gate failures={gate2['n_bad']}")
    print(f"                  [Self-check: {len(problems)}/{len(problems)} "
          f"problems verified. Adjustments: {bad[0]} answer key recomputed]")
    assert gate2["n_bad"] == 0

    key = {p["pid"]: p["key"] for p in problems}
    submitted = {"P1": key["P1"], "P2": key["P2"], "P3": 30}   # P3 wrong
    n_correct = sum(1 for pid, a in submitted.items() if a == key[pid])
    score_pct = 100.0 * n_correct / len(submitted)
    if score_pct > rules["k1_hi_thresh"]:
        delta_k = rules["k1_hi_delta"]
    elif score_pct >= 50.0:
        delta_k = rules["k1_mid_delta"]
    else:
        delta_k = -rules["k1_lo_delta"]
    m = max(0.0, m + delta_k)
    print(f"    [#1 grading]  learner answers P1 ok, P2 ok, P3=30 (wrong) "
          f"-> {n_correct}/{len(submitted)} = {score_pct:.1f}% -> "
          f"50-80% rule: +{rules['k1_mid_delta']} -> mastery {m:.2f}")

    cats = rules["gap_categories"]          # parsed from the INJECTED text
    gaps = feynman_gaps_l3(EXPL, cats, texts6["cap_middleware.py"])
    cl = 5 - len(gaps["Logical Leaps"]) - len(gaps["Undefined Terms"])
    ac = 5 - 2 * len(gaps["Factual Errors"])
    co = 5 - len(gaps["Missing Aspects"]) - len(gaps["Factual Errors"])
    ov = (cl + ac + co) / 3.0
    lo, delta_f = L2.feynman_delta(rules, ov)
    m = max(0.0, m + delta_f)
    n_gaps = sum(len(v) for v in gaps.values())
    print(f"    [#2 Feynman]  gaps={n_gaps} | clarity={cl} accuracy={ac} "
          f"completeness={co} | overall={ov:.1f} -> band >={lo}: "
          f"delta={delta_f:+.2f} | mastery {m:.2f}")
    for cat in cats:
        for g in gaps[cat]:
            print(f"        [{cat}] {g}")

    claims = [
        dict(claim="a single tool result is capped at %d tokens" % cap,
             source="cap_middleware.py", sha=shas6["cap_middleware.py"],
             quote="token_cap: int = %d" % cap),
        dict(claim="the window is the memory", source=None, sha=None,
             quote=None),
        dict(claim="one turn stays pinned raw at the head",
             source="manager.py", sha="00000000", quote="pinned: int = 1"),
    ]
    verdicts = L2.claims_gate(claims, texts6, shas6)
    n_ver = sum(1 for v in verdicts if v[1] == "VERIFIED")
    print(f"    [#5 claims]   {n_ver} VERIFIED / "
          + " / ".join(v[1] for v in verdicts if v[1] != "VERIFIED"))

    led = sqlite3.connect(str(Path(ws) / "ledger.db"))
    led.execute("CREATE TABLE ledger (n INTEGER PRIMARY KEY, event TEXT, "
                "detail TEXT)")
    events = [
        ("autonomy", "suggested 'feynman practice' -> declined; discussed, "
                     "not overridden; mastery unchanged"),
        ("problem-set", f"3 problems at d=3 for project cap-dashboard; "
                        f"{bad[0]} key fixed by oracle; "
                        f"{len(problems)}/{len(problems)} verified"),
        ("grading", f"{n_correct}/{len(submitted)} = {score_pct:.1f}% -> "
                    f"+{rules['k1_mid_delta']} -> mastery {m - delta_f:.2f} "
                    f"-> {m:.2f}"),
        ("feynman", f"overall={ov:.1f} -> band >={lo}: delta={delta_f:+.2f}; "
                    f"4 gap categories from the injected SKILL.md"),
        ("claims", f"{n_ver} VERIFIED / 1 NO-PROVENANCE / 1 SHA-DRIFT"),
        ("session-end", f"mastery={m:.2f} = {M0:.2f} {delta_k:+.2f} grading "
                        f"{delta_f:+.2f} feynman (verified credits only); "
                        f"channel=console; skills="
                        + ",".join(funnels["console"]["injected"])),
    ]
    for i, (ev, det) in enumerate(events, 1):
        led.execute("INSERT INTO ledger VALUES (?, ?, ?)", (i, ev, det))
    led.commit()
    print(f"    [#7 ledger]   {len(events)} events -> real sqlite")
    for n, ev, det in led.execute("SELECT * FROM ledger ORDER BY n"):
        print(f"        [{n}] {ev:<12} {det}")
    n_rows = led.execute("SELECT COUNT(*) FROM ledger").fetchone()[0]
    led.close()
    assert n_rows == 6

    # ------------------------------------------------------------------ [7]
    print("\n[7] behavior delta: the SAME session replayed on the voice")
    print("    channel (no feynman-check injected) vs the console agent")
    print(f"    console: grading +{delta_k:.2f} (k-plus-one), feynman-check "
          f"runs: gaps={n_gaps},")
    print(f"             overall={ov:.1f}, delta={delta_f:+.2f} -> mastery "
          f"{m:.2f} (verified credit only)")
    # declared stand-in: with no gap-analysis document in the prompt, the
    # unverified explanation is credited the mid-band magnitude — parsed
    # from the injected k-plus-one, not hardcoded
    naive = rules["k1_mid_delta"]
    m_voice = M0 + delta_k + naive
    print("    voice:   k-plus-one IS injected (channels=[all]) -> the same")
    print(f"             answers grade first: {M0:.2f} -> {M0 + delta_k:.2f}; "
          f"no feynman-check")
    print(f"             -> explanation accepted WITHOUT gap analysis; naive")
    print(f"             credit +{naive:.2f} -> mastery {m_voice:.2f}  "
          f"[TODO: needs key — a real")
    print("             model would still judge, but nothing in that prompt")
    print("             makes it verify)")
    proj_console = M0 + 10 * (delta_k + delta_f)
    proj_voice = M0 + 10 * (delta_k + naive)
    print(f"    10 such turns, projected (linear; the parsed rules floor")
    print(f"    at 0 and name no cap): console {proj_console:.2f} vs voice "
          f"{proj_voice:.2f} —")
    print("    the L2 mastery-inflation mechanism, now caused by a missing")
    print("    skill rather than a frozen difficulty.")
    print("    capability is not in the harness code — it is the injected")
    print("    document; the harness only executes what the skill says.")
    assert m_voice > m

    # ------------------------------------------------------------------ [8]
    print("\n[8] legacy dir + cross-checks")
    ws2 = build_legacy_workspace()
    base2, renamed = get_workspace_skills_dir(ws2)
    # (the legacy workspace carries no manifest; the point is the rename)
    print(f"    workspace with only legacy 'skill/': get_workspace_skills_dir")
    print(f"    renamed it in place: renamed={renamed}, base={base2.name}/, "
          f"'skill/' exists={( ws2 / 'skill').exists()}")
    assert renamed and not (ws2 / "skill").exists()
    inj2, _ = resolve_skill_loader_dirs(ws2, ["daily-review"])
    print(f"    the legacy skill still resolves through the gate: "
          f"injected={inj2}")
    assert inj2 == ["daily-review"]

    injected_texts = {
        "SOUL.md": soul,
        "k-plus-one.md": bodies["k-plus-one"],
        "feynman-check.md": bodies["feynman-check"],
        "cap_middleware.py": texts6["cap_middleware.py"],
        "manager.py": texts6["manager.py"],
        "history.py": texts6["history.py"],
    }
    rules_from_injected = L2.parse_rules(injected_texts)
    same = (rules_from_injected["k1_hi_thresh"] == rules["k1_hi_thresh"]
            and rules_from_injected["k1_hi_delta"] == rules["k1_hi_delta"]
            and rules_from_injected["k1_mid_delta"] == rules["k1_mid_delta"]
            and rules_from_injected["k1_lo_delta"] == rules["k1_lo_delta"]
            and rules_from_injected["regen_thresh"] == rules["regen_thresh"]
            and rules_from_injected["examiner_b_steps"] == rules["examiner_b_steps"]
            and rules_from_injected["gap_categories"] == rules["gap_categories"]
            and rules_from_injected["feynman_bands"] == rules["feynman_bands"]
            and rules_from_injected["token_cap"] == rules["token_cap"])
    print(f"    numbers follow the injected document: rules parsed from the")
    print(f"    workspace copies == rules parsed from the coach files "
          f"({len(rules['gap_categories'])} gap categories, bands "
          f"{len(rules['feynman_bands'])}, thresh "
          f"{rules['k1_hi_thresh']}%/{rules['regen_thresh']}) -> {same}")
    assert same

    # ------------------------------------------------------------------ [9]
    print("\n[9] self-check (structural assertions)")
    checks = [
        ("[0] all 7 line anchors re-derived live and > 0 (no silent L0)",
         all(ln > 0 for ln in derived.values())),
        ("real profile funnel: 9 dirs vs 8 enabled, codex-delegate outside",
         len(real_dirs) == 9 and len(real_enabled) == 8
         and missing == ["codex-delegate"]),
        ("manifest funnel (console): 2 dropped, 1 gated out, 3 injected",
         funnels["console"]["injected"]
         == ["daily-review", "feynman-check", "k-plus-one"]
         and len(funnels["console"]["effective"]) == 4
         and len(funnels["console"]["not_injected"]) == 1),
        ("manifest funnel (voice): voice-brief in, feynman-check out",
         funnels["voice"]["injected"]
         == ["daily-review", "k-plus-one", "voice-brief"]),
        ("SKILL.md gate logs the builder's exact message",
         funnels["console"]["not_injected"][0]
         == "skill 'onboard' has no SKILL.md at skills/onboard; not injected"),
        ("a dir with SKILL.md but no manifest entry is never effective",
         "codex-delegate" not in funnels["console"]["effective"]
         and "codex-delegate" not in funnels["voice"]["effective"]),
        ("verbatim injection: workspace copies byte-identical to coach files",
         all(L2.sha8(((ws / 'skills') / n / 'SKILL.md').read_bytes())
             == verb_shas[n] for n in verb)),
        ("frontmatter contract: parsed name == directory name (3 injected)",
         True),   # asserted inline in [4]; kept here for the printed record
        ("skills ride the prompt: prompt delta matches block delta within "
         "integer-estimator rounding",
         abs(delta - block_delta) <= 1),
        ("per-request assembly: same workspace, channel decides the skill set",
         only_console == ["feynman-check"] and only_voice == ["voice-brief"]),
        (f"#6 autonomy: a declined suggestion changes nothing "
         f"(mastery {M0:.2f})",
         True),   # asserted inline in [6]
        ("#4 gate: planted defect fixed in place; re-gate clean",
         gate["n_bad"] == 1 and gate2["n_bad"] == 0),
        (f"#1 grading: 2/3 = 66.7% lands in the parsed 50-80% band "
         f"(+{rules['k1_mid_delta']})",
         abs(score_pct - 66.7) < 0.05 and delta_k == rules["k1_mid_delta"]),
        ("#2 feynman: 4 gaps across the parsed categories; band = no change",
         n_gaps == 4 and delta_f == 0.0 and ov == 3.0),
        ("#5 claims gate: exactly one VERIFIED of three",
         n_ver == 1),
        ("#7 ledger: 6 events persisted to real sqlite", n_rows == 6),
        ("behavior delta: unverified voice credit > verified console credit",
         m_voice > m),
        ("legacy skill/ renamed in place and still resolves through the gate",
         renamed and inj2 == ["daily-review"]),
        ("rules parsed from injected texts == rules from the coach files",
         same),
    ]
    for name, ok in checks:
        assert ok, name
        print(f"    PASS  {name}")
    print("    ✅ self-check passed")

    print("\n" + "=" * 68)
    print("takeaway: the principled agent's principles are not prompt prose")
    print("  and not harness code — they are loadable artifacts. SKILL.md is")
    print("  the admission gate, the manifest owns enablement, the channel")
    print("  filters the reach, and the builder re-assembles per request, so")
    print("  the same workspace yields different agents for different")
    print("  requests. Capability lives in the injected document: the same")
    print("  explanation is verified on one channel and waved through on the")
    print("  other — and the waved-through credit is exactly the inflation")
    print("  L2 measured. SOUL governs assembly too: verbatim injection with")
    print("  provenance is principle#5 applied at build time, and a declined")
    print("  suggestion is logged, never overridden (principle#6). Real")
    print("  hosted model behind the judgment seats: [TODO: needs key]")
    print("=" * 68)

    # fixture hygiene: a successful run removes its own workspaces (this
    # prints nothing — output stays byte-identical; a failed run leaves
    # the fixtures in place for forensics)
    shutil.rmtree(ws)
    shutil.rmtree(ws2)


if __name__ == "__main__":
    main()
