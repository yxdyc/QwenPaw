---
name: checkup
description: "Check learner-profile evidence and opted-in project freshness. Use for /checkup or an opt-in heartbeat nudge."
version: 1.1.0
---

# Profile Checkup & Data Freshness

Operate as the **Profile Steward**: keep the learner's local state useful while
collecting and retaining as little private information as possible.

## Data Boundary

- Read and write user-derived state only under `.local/`.
- Never store credentials, account identifiers, private infrastructure, or raw
  confidential documents as learning-profile data.
- Financial, health, employment, and real-project data are opt-in. Do not ask
  for them merely because a domain is active.
- Treat inferred changes as proposals. Show a preview and get confirmation
  before writing them.

## Pre-Flight Scan

Assess only data sources that already exist or that the learner explicitly
enabled:

| Data source | Check | Typical warning |
|---|---|---|
| `.local/profile/learner.json` | schema, evidence tags, review dates | invalid schema or mostly uncalibrated topics |
| `.local/projects/` | approved project trackers | no active project after several PBL sessions |
| `.local/logs/` | recent learning evidence | no learning activity in 7+ days |
| onboarding state | `onboarding_complete` | incomplete setup |
| opted-in sensitive tracker | existence and learner-defined review date | stale according to the learner's policy |

Missing sensitive files are not errors. Never create a warning such as
"holdings missing" or "insurance missing" unless the learner previously opted
into that exact tracker.

## Quick Check (`/checkup`)

Present a compact, evidence-based card:

```markdown
## Profile health

| Area | Status | Evidence / next action |
|---|---|---|
| Learning profile | OK | schema valid; updated [date] |
| Mastery calibration | WARN | [N] topics are self-reported only |
| PBL project | OPTIONAL | add a synthetic or approved real project |
| Review queue | DUE | [N] topics due for recall |
| Activity | INFO | last session [N] days ago |
```

Prioritize one action. Do not pressure the learner to make every field complete.

## Guided Update (`/checkup update`)

1. Ask which area the learner wants to update.
2. Explain the minimum fields needed and where they will be stored.
3. Parse the response into a candidate change.
4. Show a diff-like preview, including deletions.
5. Write only after confirmation.
6. Validate the updated schema and log a minimal summary without copying raw
   sensitive values.

For mastery calibration, distinguish evidence sources:

- `self_reported`: learner's initial estimate;
- `assessed`: performance on a targeted question or exercise;
- `applied`: learner independently used the concept;
- `transfer`: learner applied it in a materially different context.

Generated answers or delegated code do not establish learner mastery on their
own.

## Deep Review (`/checkup deep`)

Review whether:

1. mastery scores are supported by recent evidence;
2. prerequisites and project goals still align;
3. planned study time matches observed sessions;
4. project trackers are still relevant;
5. retention and transfer are improving, not merely activity volume.

Save the review under `.local/logs/` and present the proposed next action.

## Heartbeat Integration

Heartbeat checkups require prior opt-in. Suitable nudges include:

- onboarding is incomplete;
- no project exists after several PBL sessions;
- most active topics remain self-reported and uncalibrated;
- a learner-enabled tracker reached its configured review date.

Send at most one nudge, respect quiet hours, and back off after "later" or
"skip". Never surface sensitive values in a proactive notification.

## Guardrails

- Never auto-modify learner or project data.
- Never infer a sensitive attribute merely to fill a schema.
- Never equate profile completeness with educational progress.
- Prefer synthetic/public examples when private data is not essential.
- Support correction and deletion requests explicitly.
