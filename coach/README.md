# Learning Coach Reference Profile

This directory is a usage showcase for composing QwenPaw persona files,
skills, project-based learning, spaced repetition, cron, heartbeat, memory, and
optional ACP delegation into a long-running learning coach. It is not a new
QwenPaw engine implementation.

## Data Boundary

The checked-in profile is public and reusable. It must contain only generic
persona rules, skill definitions, schemas, and synthetic examples.

All user-derived or machine-derived state belongs under `coach/.local/`, which
is ignored by Git:

- learner identity, preferences, mastery estimates, and schedules;
- real project names, repository paths, and working directories;
- holdings, insurance, health, employer, or other sensitive context;
- conversations, logs, memory indexes, credentials, channel identifiers, and
  ACP runner configuration.

The coach must preview inferred or sensitive profile changes and obtain user
confirmation before persisting them. Secret values are never valid learning
profile data.

## Start Here

Follow [profile/SETUP.md](profile/SETUP.md). The setup flow creates the live
agent configuration, relocates runtime state into `.local/`, and optionally
seeds synthetic examples without overwriting existing data.

The main learning loop is:

1. establish a consented learner profile;
2. estimate current mastery and record whether evidence is self-reported or
   assessed;
3. select a K+1 task anchored to an approved real or synthetic project;
4. evaluate the answer with a Feynman-style probe;
5. update mastery and spaced-repetition state with an evidence note;
6. periodically meta-review the teaching system, requiring approval before
   changing skills, roadmaps, or schedules.

For a full project-based curriculum that applies these principles to runnable
LLM systems exercises, see the sanitized [LLM-PBL example](../examples/llm-pbl/README.md).
The curriculum is an independent learning artifact; this profile only shows
how QwenPaw can host the coaching method around it.

## Safety Defaults

- Cron, heartbeat, and ACP delegation are opt-in.
- Runtime `agent.json` is ignored and must not be used as a checked-in sample.
- Coding delegation requires an explicit request and confirmed working
  directory; destructive or external actions require separate confirmation.
- A successful generated artifact is not by itself evidence that the learner
  understands it; mastery changes require learner-side evidence.
