#!/usr/bin/env python3
"""L0: turn a benchmark headline into a versioned, comparable score contract."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math


CONTRACT_FIELDS = (
    "benchmark", "revision", "split", "task_manifest", "prompt_template",
    "harness", "environment", "tools", "reasoning_effort", "context_budget",
    "output_budget", "samples_per_task", "temperature", "timeout_seconds",
    "verifier", "judge", "failure_policy", "metric",
)


@dataclass(frozen=True)
class BenchmarkRun:
    model: str
    benchmark: str
    revision: str
    split: str
    task_manifest: str
    prompt_template: str
    harness: str
    environment: str
    tools: str
    reasoning_effort: str
    context_budget: int
    output_budget: int
    samples_per_task: int
    temperature: float
    timeout_seconds: int
    verifier: str
    judge: str
    failure_policy: str
    metric: str
    score: float
    tasks: int
    provenance: str

def missing_fields(run: BenchmarkRun) -> list[str]:
    missing = []
    for field in CONTRACT_FIELDS + ("model", "provenance"):
        value = getattr(run, field)
        if value == "" or value is None:
            missing.append(field)
    if run.tasks <= 0:
        missing.append("tasks")
    if not 0.0 <= run.score <= 100.0:
        missing.append("score")
    return missing


def contract_diff(left: BenchmarkRun, right: BenchmarkRun) -> list[str]:
    return [field for field in CONTRACT_FIELDS if getattr(left, field) != getattr(right, field)]


def wilson_interval(percent: float, n: int) -> tuple[float, float]:
    """95% Wilson interval for a binary per-task success rate."""
    p = percent / 100.0
    z = 1.959963984540054
    denominator = 1.0 + z * z / n
    centre = (p + z * z / (2.0 * n)) / denominator
    radius = z * math.sqrt(p * (1.0 - p) / n + z * z / (4.0 * n * n)) / denominator
    return 100.0 * (centre - radius), 100.0 * (centre + radius)


def make_run(model: str, score: float, **changes: object) -> BenchmarkRun:
    values: dict[str, object] = {
        "model": model, "benchmark": "Terminal-Bench", "revision": "2.1",
        "split": "test", "task_manifest": "sha256:tasks-v2.1-100",
        "prompt_template": "terminal-agent@v1", "harness": "same-minimal-agent@abc123",
        "environment": "ubuntu-24.04@sha256:env-a", "tools": "terminal-only",
        "reasoning_effort": "max", "context_budget": 262_144,
        "output_budget": 65_536, "samples_per_task": 1, "temperature": 1.0,
        "timeout_seconds": 7_200, "verifier": "official-tests@v2.1",
        "judge": "deterministic-binary@v1",
        "failure_policy": "timeout/error/blocked=0; all tasks in denominator",
        "metric": "task_success_rate", "score": score, "tasks": 100,
        "provenance": "public-harness",
    }
    values.update(changes)
    return BenchmarkRun(**values)  # type: ignore[arg-type]


def main() -> None:
    baseline = make_run("model-a", 82.0)
    same_contract = make_run("model-b", 85.0)
    with_search = make_run("model-c", 91.0, tools="terminal+web-search")
    pass_at_eight = make_run(
        "model-d", 96.0, samples_per_task=8, metric="pass@8", temperature=0.8
    )
    hidden_failures = make_run(
        "model-e", 88.0, failure_policy="completed tasks only; failures excluded"
    )
    changed_prompt = make_run("model-f", 90.0, prompt_template="terminal-agent@v2")
    changed_runtime = make_run(
        "model-g",
        89.0,
        environment="ubuntu-24.04@sha256:env-b",
        verifier="official-tests@v2.1-patch1",
    )
    changed_manifest = make_run(
        "model-h", 87.0, task_manifest="sha256:tasks-v2.1-80", tasks=80
    )
    incomplete = make_run("model-invalid", 99.0, prompt_template="")
    runs = (baseline, same_contract, with_search, pass_at_eight, hidden_failures,
            changed_prompt, changed_runtime, changed_manifest, incomplete)

    valid = [run for run in runs if not missing_fields(run)]
    raw_headline_winner = max(runs, key=lambda run: run.score)
    naive_winner = max(valid, key=lambda run: run.score)
    comparable = [run for run in valid if not contract_diff(baseline, run)]
    comparable_winner = max(comparable, key=lambda run: run.score)

    print("raw headlines (missing contracts are ineligible)")
    for run in sorted(runs, key=lambda item: item.score, reverse=True):
        missing = missing_fields(run)
        suffix = f" | INVALID: {', '.join(missing)}" if missing else ""
        print(f"  {run.model}: {run.score:.1f} {run.metric}{suffix}")
    print(f"raw headline winner: {raw_headline_winner.model} ({raw_headline_winner.score:.1f})")
    print(f"naive valid winner: {naive_winner.model} ({naive_winner.score:.1f})")
    print("\ncontract audit against model-a")
    for run in runs[1:]:
        missing = missing_fields(run)
        differences = contract_diff(baseline, run)
        verdict = "INVALID" if missing else "COMPARABLE" if not differences else "NOT_COMPARABLE"
        detail = f"missing {', '.join(missing)}" if missing else (
            "same contract" if not differences else ", ".join(differences)
        )
        print(f"  {run.model}: {verdict} | {detail}")
    print(
        f"comparable winner: {comparable_winner.model} "
        f"({comparable_winner.score:.1f}, n={comparable_winner.tasks})"
    )

    low, high = wilson_interval(comparable_winner.score, comparable_winner.tasks)
    print(f"illustrative 95% Wilson interval: [{low:.1f}, {high:.1f}]")
    print("boundary: interval assumes independent binary tasks; agent tasks often violate this")

    checks = {
        "missing contract fails closed": missing_fields(incomplete) == ["prompt_template"]
        and incomplete not in valid
        and incomplete not in comparable,
        "same contract remains comparable": not contract_diff(baseline, same_contract),
        "prompt template is part of contract": contract_diff(baseline, changed_prompt)
        == ["prompt_template"],
        "environment and verifier are part of contract": contract_diff(
            baseline, changed_runtime
        ) == ["environment", "verifier"],
        "task manifest is part of contract": contract_diff(baseline, changed_manifest)
        == ["task_manifest"],
        "tool augmentation breaks comparability": contract_diff(baseline, with_search) == ["tools"],
        "pass@k is not pass@1": {"samples_per_task", "temperature", "metric"}.issubset(
            contract_diff(baseline, pass_at_eight)
        ),
        "failure denominator is part of score": contract_diff(baseline, hidden_failures)
        == ["failure_policy"],
        "contract-aware winner differs from naive winner": comparable_winner != naive_winner,
    }
    for name, passed in checks.items():
        print(f"{'PASS' if passed else 'FAIL'} | {name}")
    assert all(checks.values())

    payload = {
        "schema_version": "1.0",
        "module": "benchmark-contract-l0",
        "metrics": {
            "runs": len(runs),
            "valid_runs": len(valid),
            "invalid_runs": len(runs) - len(valid),
            "comparable_to_baseline": len(comparable),
            "raw_headline_winner": raw_headline_winner.model,
            "naive_valid_winner": naive_winner.model,
            "contract_winner": comparable_winner.model,
        },
        "checks": {"passed": sum(checks.values()), "total": len(checks)},
        "evidence_boundary": (
            "synthetic contracts only; no external benchmark score or model quality is validated"
        ),
    }
    digest_input = json.dumps(
        {"runs": [asdict(run) for run in runs], "payload": payload},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    payload["digest"] = hashlib.sha256(digest_input.encode()).hexdigest()[:16]
    print("RESULT_JSON=" + json.dumps(payload, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
