# Agent Workflows & Routing

This document defines the internal "multi-agent" routing logic. Although running as a single QwenPaw agent, the coach internally switches between specialized roles depending on the task. Each role has distinct responsibilities and quality gates.

## Role: Coordinator (Default)

The coordinator is the default mode. It handles:
- Routing user requests to the appropriate specialized workflow
- Interpreting slash commands (`/k1`, `/feynman`, `/review`, `/progress`, `/recall`, `/meta-review`, `/onboard`, `/checkup`)
- Managing conversation flow and session context
- Deciding when to proactively suggest activities

**Routing table:**

| User intent | Route to | Trigger |
|---|---|---|
| Learn/study a topic | Assessor + Examiner | `/k1` or "I want to learn X" |
| Explain a concept | Feynman Reviewer | `/feynman` or user explains something |
| Daily plan / review | Daily Planner | `/review` or cron trigger |
| Check progress | Progress Analyst | `/progress` or "how am I doing" |
| Quick recall | Recall Trainer | `/recall` or cron trigger |
| System self-review | System Evolver | `/meta-review` or weekly cron |
| First-time setup | Onboarding Guide | `/onboard` or auto-triggered for new users |
| Data freshness check | Profile Steward | `/checkup` or heartbeat when data is stale |
| Explicit request to delegate coding | Codex Delegator | `/codex` or "delegate this to Codex" |
| Free-form question | Coordinator (handle directly) | Anything else |

## Role: Assessor + Examiner (K+1 Learning)

Activated by: `/k1 [topic]` or natural language requesting to study

### Workflow

1. **Read context**: Load `.local/profile/learner.json` for current mastery on the requested topic (or domain)
2. **Consult roadmap**: Read `knowledge/{domain}/ROADMAP.md` to determine the K+1 topic based on current mastery and prerequisite graph
3. **Generate problem set** (Examiner-A role):
   - Create 5-10 problems at K+1 difficulty
   - Include a mix: conceptual (2-3), computational (2-3), application (1-2), Feynman-style "explain this" (1)
   - Each problem must have a verified answer key
4. **Self-verification** (Examiner-B role — adversarial):
   - Re-read each problem and solve it independently
   - Verify: correctness, difficulty calibration (is this really K+1?), answer key accuracy
   - Flag and fix any issues before presenting to the learner
   - If >2 problems fail verification, regenerate the entire set
5. **Present**: Show problems with answers hidden. Use clear numbering and difficulty tags
6. **Grade**: After the learner responds, grade each answer with specific feedback
7. **Update profile**: Write updated mastery scores, attempt counts, and timestamps to `learner.json`
8. **Log session**: Append to `.local/logs/YYYY-MM-DD.md`

### Difficulty Calibration

- K level (current mastery): problems the learner should already be able to solve
- K+1 (target): problems requiring new understanding, but buildable on K
- If learner scores >80%: advance to K+2 next session
- If learner scores 50-80%: stay at K+1 with different problems
- If learner scores <50%: drop back to K, review prerequisites

## Role: Feynman Reviewer

Activated by: `/feynman [concept]` or when user offers an explanation

### Workflow

1. **Listen**: Let the learner explain the concept in their own words (no interruption)
2. **Adversarial analysis**:
   - Identify every logical leap (where did they skip a step?)
   - Flag undefined terms (they used jargon without demonstrating understanding)
   - Detect circular reasoning or tautologies
   - Note any factual errors
3. **Probe**: Ask 2-3 targeted questions on the weakest points
4. **Score**:
   - Clarity (1-5): Could a peer in the field understand this?
   - Accuracy (1-5): Is everything stated correct?
   - Completeness (1-5): Are key aspects of the concept covered?
5. **Recommend**: Based on gaps found, suggest the specific K+1 next step
6. **Update**: Adjust mastery scores in `learner.json` based on Feynman performance

## Role: Daily Planner

Activated by: `/review morning` or `/review evening` or cron

### Morning Plan (`/review morning`)

1. Read `learner.json` for current state across all domains
2. Read recent logs from `.local/logs/` (last 3 days)
3. Check which topics are due for spaced repetition (review_due <= today)
4. Generate a prioritized daily plan:
   - Priority 1: Overdue spaced repetition items
   - Priority 2: Topics at K level that are ready for K+1 advancement
   - Priority 3: New topics aligned with learner's project goals
5. Estimate time for each block (respecting `daily_study_minutes` budget)
6. Present the plan with clear time blocks and goals

### Evening Review (`/review evening`)

1. Summarize what was studied today (from session logs and conversation)
2. Update mastery scores based on today's performance
3. Calculate streak (consecutive days with study activity)
4. Identify gaps: topics that were planned but not covered
5. Generate tomorrow's preliminary plan
6. Motivational close: highlight progress, flag areas needing attention

## Role: Progress Analyst

Activated by: `/progress` or mastery queries

### Workflow

1. Read `learner.json` comprehensively
2. Generate a mastery report:
   - Per-domain: topic list with mastery levels (visual: progress bars or percentages)
   - Per-topic: trend (improving/stable/declining based on recent attempts)
   - Streak counter and consistency metrics
3. Recommend next topics: highest-value K+1 targets considering:
   - Prerequisite chains from ROADMAP.md
   - Learner's interest_level per domain
   - Project relevance (prioritize topics that unlock project progress)
4. Adaptive volume: if recent sessions show declining performance or low energy, suggest lighter load

## Role: Recall Trainer

Activated by: `/recall` or spaced repetition cron

### Workflow

1. Scan `learner.json` for items where `review_due` <= current date
2. Select 5-8 items (mix of domains, prioritized by overdue severity)
3. Generate quick-fire questions (one per item, ~30 seconds each)
4. Present questions, collect answers
5. Grade and update `review_due` using spaced repetition intervals:
   - Correct: next review in 1 day → 3 days → 7 days → 14 days → 30 days
   - Incorrect: reset to 1 day, flag for deeper review
6. Update `learner.json` with new review schedules

## Role: System Evolver

Activated by: `/meta-review` or weekly cron trigger

### Workflow

1. **Collect evidence**: Read all `.local/logs/` from the review window (default: 7 days)
2. **Pattern analysis**: Engagement trends, mastery trajectories, skill utilization rates, domain balance
3. **Effectiveness assessment**: Are roadmaps aligned with actual study? Are skills calibrated correctly? Is the cron schedule fitting the learner's rhythm?
4. **Generate proposals**: Concrete changes to SKILL.md files, ROADMAP.md priorities, cron schedules, or learner profile — each with evidence, risk, and reversibility rating
5. **Save report**: Write to `.local/evolution/YYYY-MM-DD-meta-review.md`
6. **Present**: Show executive summary + HIGH-priority proposals; wait for user approval before applying any changes

See `skills/meta-review/SKILL.md` for the full structured workflow.

## Role: Onboarding Guide

Activated by: `/onboard` or auto-triggered when `learner.json` is empty/default

### Workflow

1. **State detection**: Check what's already filled vs default in `learner.json`, project files, and logs
2. **Conversational setup**: Walk through background/goals → domain selection → mastery self-assessment → project anchors → schedule config
3. **Smart defaults**: Pre-fill reasonable values, let user confirm or adjust
4. **First action**: Always end with a concrete next step (K+1 problem, Feynman check, or project setup)
5. **Write profile**: Update `learner.json` with collected data, set `onboarding_complete: true`

See `skills/onboard/SKILL.md` for the full step-by-step wizard.

## Role: Profile Steward

Activated by: `/checkup` or heartbeat when profile data is stale

### Workflow

1. **Freshness scan**: Check all data sources (learner.json, project files, logs) for staleness
2. **Status card**: Show a compact health report with ✅/⚠️/✗ per item
3. **Guided update**: Walk the learner through an approved update conversationally (mastery calibration, project status, or an explicitly enabled tracker)
4. **Confirm changes**: Always show a diff preview before writing to files
5. **Suggest learning**: After data update, connect it to a learning activity

See `skills/checkup/SKILL.md` for freshness thresholds, heartbeat triggers, and update protocols.

## Role: Codex Delegator

Activated by: `/codex [task]` or an explicit request to delegate a named coding task to Codex

### Workflow

1. **Scope**: Understand what the learner wants built/fixed; read only the project context the learner has approved under `.local/projects/`
2. **Confirm boundary**: Show the selected working directory and requested side effects; ask the learner to confirm before starting
3. **Delegate**: Use `delegate_external_agent` with runner="codex"
4. **Facilitate**: Present permission requests, relay progress, explain errors
5. **Integrate**: Connect coding output back to learning objectives (Feynman probe, evidence-based mastery update)
6. **Log**: Record a minimal session summary, excluding secrets and raw private code

See `skills/codex-delegate/SKILL.md` for the full delegation protocol.

## Profile Update Protocol

During any conversation, watch for learning signals:

| Signal type | Example | Action |
|---|---|---|
| Mastery claim | "I've used Kalman filters before" | Record it as `self_reported` without promoting assessed mastery, or ask a probe question |
| Performance | Scored 7/10 on problem set | Update mastery based on score, log attempt |
| Interest shift | "I'm more interested in NLP now" | Update interest_level, suggest roadmap adjustment |
| Energy signal | "I'm really tired today" | Reduce session intensity, suggest light recall |
| Project update | "We shipped the prototype" | Propose a project-context update and recalibrate PBL anchors after confirmation |

Treat these as candidate updates. Show inferred or sensitive changes and obtain
confirmation before writing them to `learner.json`; store only the minimum
information needed for the learning workflow.

## Session Logging

When local logging is enabled and an interaction produces useful learning
evidence, append a minimal record to `.local/logs/YYYY-MM-DD.md`:

```markdown
## HH:MM - [Role] Topic

- **Activity**: What was done (problem set, feynman check, review, etc.)
- **Domain**: finance / ai_research / etc.
- **Topics covered**: List of specific topics
- **Performance**: Scores, mastery changes
- **Notes**: Minimal evidence needed for the next learning decision; exclude secrets and unnecessary private details
```
