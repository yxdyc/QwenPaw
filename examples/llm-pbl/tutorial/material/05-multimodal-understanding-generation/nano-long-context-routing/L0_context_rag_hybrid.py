"""Deterministic evidence-selection surrogate for long context, RAG, and hybrid."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    doc: str
    position: int
    text: str


@dataclass(frozen=True)
class Case:
    name: str
    question: str
    required: tuple[str, ...]


CHUNKS = (
    Chunk("contract/cover", "contract", 0, "Master service agreement for ACME and billing contacts."),
    Chunk("contract/sla", "contract", 1, "Service credits apply after monthly availability falls below target."),
    Chunk("contract/privacy", "contract", 2, "The processor deletes customer backups after account closure."),
    Chunk("contract/exit", "contract", 3, "ACME may terminate for convenience with 45 days written notice."),
    Chunk("meeting/0", "meeting", 0, "The launch discussion opened with design status."),
    Chunk("meeting/1", "meeting", 1, "The product owner made a commitment to launch on Friday."),
    Chunk("meeting/2", "meeting", 2, "The finance team reviewed an unrelated travel budget."),
    Chunk("meeting/3", "meeting", 3, "The security review failed after a critical incident."),
    Chunk("meeting/4", "meeting", 4, "Therefore that earlier commitment was withdrawn and release is paused."),
    Chunk("repo/0", "repo", 0, "The API router change requires rollback plan R1."),
    Chunk("repo/1", "repo", 1, "The schema note updates field descriptions only."),
    Chunk("repo/2", "repo", 2, "The worker retry change requires rollback plan R2."),
    Chunk("repo/3", "repo", 3, "The dashboard copy change is documentation only."),
    Chunk("repo/4", "repo", 4, "The database migration requires rollback plan R3."),
)

CASES = (
    Case("sparse_clause", "What notice period lets ACME terminate for convenience?", ("contract/exit",)),
    Case(
        "temporal_reversal",
        "Did the Friday launch commitment survive the security review?",
        ("meeting/1", "meeting/3", "meeting/4"),
    ),
    Case(
        "exhaustive_audit",
        "Which changes require a rollback plan?",
        ("repo/0", "repo/2", "repo/4"),
    ),
)


def words(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def token_cost(chunks: tuple[Chunk, ...]) -> int:
    return sum(len(re.findall(r"[a-z0-9]+", chunk.text.lower())) for chunk in chunks)


def rag(question: str, top_k: int = 2) -> tuple[Chunk, ...]:
    query = words(question)
    ranked = sorted(
        enumerate(CHUNKS),
        key=lambda item: (-len(query & words(item[1].text)), item[0]),
    )
    return tuple(chunk for _, chunk in ranked[:top_k])


def hybrid(question: str) -> tuple[Chunk, ...]:
    anchors = rag(question)
    wanted = {
        chunk.chunk_id
        for anchor in anchors
        for chunk in CHUNKS
        if chunk.doc == anchor.doc and abs(chunk.position - anchor.position) <= 1
    }
    return tuple(chunk for chunk in CHUNKS if chunk.chunk_id in wanted)


def evaluate(selected: tuple[Chunk, ...], case: Case) -> dict[str, object]:
    selected_ids = tuple(chunk.chunk_id for chunk in selected)
    found = len(set(selected_ids) & set(case.required))
    recall = found / len(case.required)
    return {
        "correct_if_oracle_readout": recall == 1.0,
        "evidence_recall": round(recall, 6),
        "input_tokens_surrogate": token_cost(selected),
        "selected": list(selected_ids),
    }


def main() -> None:
    selectors = {
        "whole_context": lambda _question: CHUNKS,
        "rag_top2": rag,
        "hybrid_neighbor": hybrid,
    }
    records: dict[str, dict[str, dict[str, object]]] = {}
    print("LONG-CONTEXT ROUTING L0")
    for case in CASES:
        records[case.name] = {}
        for method, selector in selectors.items():
            result = evaluate(selector(case.question), case)
            records[case.name][method] = result
            print(
                f"case={case.name} method={method} "
                f"recall={result['evidence_recall']:.3f} "
                f"correct={str(result['correct_if_oracle_readout']).lower()} "
                f"tokens={result['input_tokens_surrogate']} "
                f"selected={','.join(result['selected'])}"
            )

    metrics = {}
    for method in selectors:
        rows = [records[case.name][method] for case in CASES]
        metrics[method] = {
            "accuracy": round(sum(bool(row["correct_if_oracle_readout"]) for row in rows) / len(rows), 6),
            "mean_evidence_recall": round(sum(float(row["evidence_recall"]) for row in rows) / len(rows), 6),
            "mean_input_tokens_surrogate": round(
                sum(int(row["input_tokens_surrogate"]) for row in rows) / len(rows), 3
            ),
        }
    checks = {
        "sparse_rag_saves_tokens": records["sparse_clause"]["rag_top2"]["correct_if_oracle_readout"]
        and records["sparse_clause"]["rag_top2"]["input_tokens_surrogate"]
        < records["sparse_clause"]["whole_context"]["input_tokens_surrogate"],
        "hybrid_recovers_temporal_chain": records["temporal_reversal"]["hybrid_neighbor"]["correct_if_oracle_readout"]
        and not records["temporal_reversal"]["rag_top2"]["correct_if_oracle_readout"],
        "global_view_needed_for_exhaustive_case": records["exhaustive_audit"]["whole_context"]["correct_if_oracle_readout"]
        and not records["exhaustive_audit"]["hybrid_neighbor"]["correct_if_oracle_readout"],
    }
    payload: dict[str, object] = {
        "schema_version": "1.0",
        "module": "nano-long-context-routing/L0",
        "metrics": metrics,
        "checks": checks,
        "evidence_boundary": "selection surrogate only; oracle readout, no tokenizer, attention, retrieval model, or LLM",
    }
    canonical = json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    payload["digest"] = hashlib.sha256(canonical.encode()).hexdigest()[:16]
    print("RESULT_JSON=" + json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":")))
    if not all(checks.values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
