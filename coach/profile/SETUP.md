# Coach System Setup Guide

This directory is a reusable learning-coach profile, not a place to commit a
live learner workspace. QwenPaw creates `agent.json` and other runtime state in
the workspace; `coach/bootstrap.sh` moves those files under the Git-ignored
`coach/.local/` directory and leaves transparent symlinks behind.

## 1. Create the Agent

Run from the repository root. The command below matches the current
`upstream/main` CLI (`agents` is plural; `agent` remains an alias):

```bash
qwenpaw agents create \
  --name "Personal Learning Coach" \
  --agent-id coach \
  --workspace-dir "$(pwd)/coach/profile" \
  --language zh
```

Then relocate the generated runtime state and seed optional demo data:

```bash
bash coach/bootstrap.sh

mkdir -p coach/.local/profile coach/.local/projects
cp -n coach/local-demo/profile/learner.json.sample \
  coach/.local/profile/learner.json
cp -n coach/local-demo/projects/sample-project.md.sample \
  coach/.local/projects/sample-project.md
cp -n coach/local-demo/projects/coding-project.md.sample \
  coach/.local/projects/coding-project.md
```

`cp -n` does not overwrite existing private data. Run `git status` afterwards:
runtime state, `agent.json`, logs, credentials, and learner data must not appear.

## 2. Start and Verify

Start QwenPaw with the normal application command, select the `coach` agent in
the console, and begin with `/onboard`.

Useful checks:

```bash
qwenpaw agents list
qwenpaw cron list --agent-id coach
```

Suggested first interactions:

- `/onboard` — configure goals and consented private profile data
- `/progress` — inspect mastery state and evidence quality
- `/k1 <topic>` — generate a calibrated K+1 exercise set
- `/feynman <concept>` — test an explanation for gaps
- `/review morning` — generate a daily learning plan
- `/meta-review` — review the coaching system after enough evidence exists

## 3. Scheduled Reviews Are Opt-in

Do not create cron jobs before the learner confirms the schedule and target
conversation. First obtain valid routing values rather than guessing them:

```bash
qwenpaw chats list --agent-id coach --channel console
```

One current-CLI example is:

```bash
qwenpaw cron create \
  --agent-id coach \
  --type agent \
  --schedule-type cron \
  --name "Morning Study Plan" \
  --cron "0 8 * * *" \
  --timezone "Asia/Shanghai" \
  --channel console \
  --target-user "CHANGEME" \
  --target-session "CHANGEME" \
  --text "/review morning" \
  --timeout 600
```

Replace the timezone, channel, user, and session with confirmed values. Other
optional schedules can use `/review evening`, `/recall`, or `/meta-review`.
Prefer the QwenPaw console if you do not need scriptable setup.

See the repository's [cron documentation](../../website/public/docs/cron.zh.md)
and [heartbeat documentation](../../website/public/docs/heartbeat.zh.md) for
the current lifecycle and delivery semantics.

## 4. Codex Delegation Is Optional

The `codex-delegate` skill is a learning workflow layered on QwenPaw's ACP
integration. It is disabled unless all of the following are true:

1. An ACP-compatible Codex runner is installed and can start outside QwenPaw.
2. Authentication is configured outside this Git-tracked profile.
3. The `codex` runner and `delegate_external_agent` tool are explicitly enabled.
4. The learner explicitly asks to delegate a coding task and confirms the
   target working directory.

Use QwenPaw's ACP settings to inspect `command`, `args`, `env`, and `trusted`.
Do not paste API keys into this repository, and do not mark a runner trusted
until its command, package provenance, permissions, and working-directory
boundary have been reviewed. See the current
[ACP integration guide](../../website/public/docs/acp-integration.zh.md).

## 5. Public and Private Files

```text
coach/
├── README.md                     # Showcase overview and data boundary
├── bootstrap.sh                  # Moves generated runtime state to .local/
├── profile/                      # Public, reusable template
│   ├── AGENTS.md                 # Workflow routing contract
│   ├── PROFILE.md                # Generic coach persona; no user identity
│   ├── SOUL.md                   # Teaching principles
│   ├── HEARTBEAT.md              # Opt-in proactive behavior
│   ├── SETUP.md                  # This guide
│   ├── skills/                   # Reusable coaching workflows
│   └── knowledge/                # Generic example roadmaps
└── .local/                       # Private runtime state; Git-ignored
    ├── agent.json                # Live agent and ACP configuration
    ├── profile/learner.json      # Consented learner profile
    ├── projects/                 # Private project descriptions/data
    ├── logs/                     # Session logs
    └── evolution/                # Meta-review reports
```

Never put credentials, private infrastructure, employer/project codenames,
financial records, or collaboration ledgers in the public template.

## 6. Customization

- Add a generic domain roadmap under `profile/knowledge/{domain}/ROADMAP.md`.
- Store learner-specific mastery under `.local/profile/learner.json`.
- Add a skill under `profile/skills/{skill-name}/SKILL.md` only when its
  behavior and write boundary are reusable and documented.
- Keep schedule, channel, ACP, and provider configuration in the ignored live
  `agent.json`, not in source-controlled examples.
