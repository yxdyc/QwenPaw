---
name: meta-review
description: "Periodic self-evolution: review recent learning logs, assess skill/roadmap effectiveness, and propose structural adjustments. Use for /meta-review or weekly cron self-improvement cycle."
version: 1.1.0
---

# Meta-Review & Self-Evolution

You are now operating as the **System Evolver** — responsible for reviewing the coaching system's own effectiveness and proposing evidence-based improvements.

This skill is the coach's "retrospective": a structured self-audit that closes the feedback loop between what the system prescribes and what actually works for the learner.

## Context Loading

Always read these before running the review:

1. **Daily logs**: `.local/logs/` — read all logs from the review window (default: last 7 days)
2. **Learner profile**: `.local/profile/learner.json` — full mastery state, streaks, domain balance
3. **Active skills**: Read each `skills/*/SKILL.md` frontmatter to know what workflows exist
4. **Knowledge roadmaps**: `knowledge/*/ROADMAP.md` — current topic priorities and prerequisite chains
5. **Prior evolution reports**: `.local/evolution/` — read the most recent report (if any) to avoid repeating proposals and to check whether past proposals were acted on
6. **Current cron config**: Check `agent.json` heartbeat/cron settings and any cron jobs referenced in `SETUP.md`

## Phase 1: Pattern Analysis

Analyze the logs from the review window and extract structured signals:

### 1.1 Engagement Patterns

| Signal | How to detect |
|---|---|
| **Active days / total days** | Count log files vs calendar days in window |
| **Session depth** | Average problems attempted per session; Feynman completions vs skips |
| **Domain balance** | % of sessions per domain; flag >70% skew toward one domain |
| **Time-of-day distribution** | Morning vs evening session ratio (from log timestamps) |
| **Streak status** | Current streak from `learner.json`; trend (growing/stalled/broken) |
| **Cron engagement** | Did cron-triggered sessions get meaningful follow-up, or just acknowledged? |

### 1.2 Mastery Trajectory

For each topic studied in the window:
- **Advancing**: mastery increased across sessions — healthy progression
- **Plateaued**: mastery unchanged across 3+ attempts — stuck, strategy needs change
- **Regressing**: mastery decreased — possible over-promotion, fatigue, or poor problem calibration
- **Neglected**: topic has high `interest_level` in domain but zero attempts in window — roadmap ordering issue?

### 1.3 Skill Utilization

For each active skill (`/k1`, `/feynman`, `/review`, `/recall`, `/progress`):
- How often invoked (by user vs by cron)?
- When invoked, did it complete the full workflow, or get abandoned mid-stream?
- Any skills never invoked in the window? Flag as underused or redundant.

## Phase 2: Effectiveness Assessment

Evaluate the system itself — not the learner, the system.

### 2.1 Roadmap Alignment

- Are roadmap priorities matching what the learner actually studies?
- Are prerequisite chains too rigid (learner wants to skip ahead) or too loose (learner is lost)?
- Are there topics in the roadmap that consistently get skipped? Consider removing or merging them.
- Are there topics the learner asks about that are NOT in any roadmap? Propose additions.

### 2.2 Skill Effectiveness

- **K+1 calibration**: Are problems consistently too hard (>50% failure) or too easy (>90% success)? The sweet spot is ~65-75% success rate.
- **Feynman depth**: Are Feynman checks producing genuine understanding signals, or are they becoming rote?
- **Daily review accuracy**: Do morning plans match what actually happened that day? If not, the planning model is miscalibrated.
- **Spaced repetition timing**: Are topics being reviewed at the right intervals, or is the learner consistently failing/succeeding at review?

### 2.3 Cron Schedule Fit

- Is the morning plan arriving at a useful time (before the learner's first session)?
- Is the evening review capturing the full day, or is it too early?
- Is midday recall being acted on, or consistently ignored?
- Does the schedule match the learner's actual rhythm (from log timestamps)?

## Phase 3: Generate Proposals

For each finding from Phases 1-2, produce a concrete, actionable proposal.

### Proposal Format

```markdown
### [PRIORITY: HIGH | MEDIUM | LOW] Proposal Title

**Finding**: [1-2 sentence description of what the data shows]
**Evidence**: [Specific numbers/log references that support the finding]
**Proposal**: [Concrete action to take — what to change and how]
**Affected file(s)**: [Exact file paths that would be modified]
**Risk**: [What could go wrong if this change is made]
**Reversibility**: [Easy/Medium/Hard to undo if it doesn't work]
```

### Proposal Categories

| Category | Examples |
|---|---|
| **ROADMAP** | Reorder topics, add missing topics, remove consistently-skipped prerequisites, adjust difficulty curves |
| **SKILL** | Adjust problem count in `/k1`, add scaffolding to `/feynman`, change recall question format in `/recall` |
| **SCHEDULE** | Shift cron times, add/remove cron jobs, change frequency (daily→weekly for low-engagement tasks) |
| **PROFILE** | Recalibrate initial mastery estimates, adjust `daily_study_minutes`, update `active_projects` |
| **WORKFLOW** | Add a new skill, retire an unused skill, change adversarial verification thresholds |

### Priority Rules

- **HIGH**: Blocking issue — learner is stuck, disengaged, or system is actively miscalibrated
- **MEDIUM**: Improvement opportunity — system works but could work better
- **LOW**: Polish — minor tweaks that may improve experience over time

## Output: Evolution Report

Save the report to `.local/evolution/YYYY-MM-DD-meta-review.md`:

```markdown
# Meta-Review Report — [Date]

**Review window**: [start date] to [end date]
**Logs analyzed**: [N] sessions across [N] days
**Active domains**: [list with session counts]
**Overall system health**: [Healthy / Needs Adjustment / Requires Intervention]

---

## Executive Summary

[3-5 sentence plain-language summary of the week: what worked, what didn't, what to change]

## Engagement Snapshot

| Metric | This Week | Trend |
|---|---|---|
| Active days | [N/7] | [↑↓→] |
| Sessions completed | [N] | [↑↓→] |
| Problems attempted | [N] | [↑↓→] |
| Average score | [X%] | [↑↓→] |
| Domain balance | [finance X% / AI Y%] | [skew direction] |

## Mastery Movement

| Topic | Start Mastery | End Mastery | Trend | Notes |
|---|---|---|---|---|
| [topic] | [X] | [Y] | [↑↓→] | [brief note] |

## Proposals

[List all proposals sorted by priority: HIGH first, then MEDIUM, then LOW]

## Status of Prior Proposals

[If a previous evolution report exists, list its proposals and their current status:
- IMPLEMENTED: the change was made
- PENDING: proposed but not yet acted on
- REJECTED: decided against (note reason if known)]

---
*Generated by /meta-review on [date]. Proposals require user approval before implementation.*
```

After writing the report, present the **Executive Summary** and **all HIGH-priority proposals** directly to the learner. Ask for approval before modifying any skill, roadmap, or schedule files.

## Applying Approved Proposals

When the learner approves a proposal:

1. **Read the target file** (SKILL.md, ROADMAP.md, SETUP.md, or learner.json)
2. **Make the minimal change** described in the proposal — do not refactor unrelated content
3. **Log the change** at the bottom of the evolution report under a `## Changes Applied` section:
   ```markdown
   ### [Date] Applied: [Proposal Title]
   - File: [path]
   - Change: [brief description of what was modified]
   - Revert: [how to undo if needed]
   ```
4. **Update `skill.json` version** if a skill's SKILL.md was modified (bump `version` in frontmatter)

## Cron Integration

### Weekly Self-Review (Recommended)

Scheduling is opt-in. After the learner confirms the timezone, cadence, target
channel, user, and session, follow `SETUP.md` or the QwenPaw console to create a
current-format agent cron job whose text is `/meta-review`. Do not create or
change the schedule from this skill without explicit approval.

### On-Demand

The learner can also invoke `/meta-review` manually at any time:
- `/meta-review` — full review of last 7 days
- `/meta-review 14` — review last 14 days
- `/meta-review focus:finance` — scope review to a single domain

### Post-Change Follow-up

After applying any proposal, add a note to the NEXT cron-triggered review to check whether the change had the intended effect. This creates a closed feedback loop.

## Guardrails

- **Never auto-apply proposals.** Always present them and wait for explicit approval.
- **Never delete a skill or roadmap.** Only propose deprecation (comment out, mark as archived).
- **Cap changes per cycle.** Propose at most 3 HIGH and 5 MEDIUM changes — avoid overwhelming the learner.
- **Respect the learner's autonomy.** If a proposal was rejected last cycle, do not re-propose it unless new evidence strongly supports it.
- **Be honest about uncertainty.** If the data is insufficient to draw a conclusion, say so rather than manufacturing a proposal.
