from __future__ import annotations

from dataclasses import dataclass
from typing import Any


SLOT_ORDER = ("primary", "linked", "ambient", "evidence")


@dataclass
class MemoryContextPacket:
    version: str
    run_id: str
    step_index: int | None
    query: str
    slots: dict[str, list[dict[str, Any]]]
    evidence: list[dict[str, Any]]
    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "run_id": self.run_id,
            "step_index": self.step_index,
            "query": self.query,
            "slots": self.slots,
            "evidence": self.evidence,
            "metadata": self.metadata,
        }


def build_memory_context_packet(
    *,
    run_id: str,
    step_index: int | None,
    query: str,
    working_memory: list[dict[str, Any]],
    evidence_snippets: list[str] | None = None,
    metadata: dict[str, Any] | None = None,
    limit: int = 4,
) -> dict[str, Any]:
    slots: dict[str, list[dict[str, Any]]] = {slot: [] for slot in SLOT_ORDER}
    for item in working_memory[: max(limit, 0)]:
        slot = str(item.get("slot") or item.get("source", {}).get("slot") or "primary")
        if slot not in slots:
            slot = "ambient"
        slots[slot].append(_packet_item(item))
    evidence = [
        {"content": snippet, "content_type": "support", "source": {"slot": "evidence"}}
        for snippet in evidence_snippets or []
    ]
    packet = MemoryContextPacket(
        version="memory-context-packet-v1",
        run_id=run_id,
        step_index=step_index,
        query=query,
        slots=slots,
        evidence=evidence,
        metadata=metadata or {},
    )
    return packet.to_dict()


def render_memory_context_packet(packet: dict[str, Any]) -> str:
    slots = packet.get("slots") or {}
    evidence = packet.get("evidence") or []
    lines: list[str] = ["<memoria_context>"]
    section_labels = {
        "primary": "primary_memory",
        "linked": "linked_context",
        "ambient": "ambient_memory",
        "evidence": "evidence_memory",
    }
    for slot in SLOT_ORDER:
        items = list(slots.get(slot) or [])
        if not items:
            continue
        lines.append(f"<{section_labels[slot]}>")
        for item in items:
            label = str(item.get("node_class") or item.get("content_type") or "memory").replace("_", " ")
            reason = str(item.get("reason") or "").strip()
            suffix = f" ({reason})" if reason else ""
            lines.append(f"- [{label}] {item.get('content', '')}{suffix}")
        lines.append(f"</{section_labels[slot]}>")
    if evidence:
        lines.append("<evidence>")
        for item in evidence:
            lines.append(f"- [support] {item.get('content', '')}")
        lines.append("</evidence>")
    if len(lines) == 1:
        return ""
    lines.append("Use only the memory items that are directly relevant to the current task.")
    lines.append("</memoria_context>")
    return "\n".join(lines)


def _packet_item(item: dict[str, Any]) -> dict[str, Any]:
    source = dict(item.get("source") or {})
    return {
        "node_key": item.get("node_key"),
        "content_type": item.get("content_type"),
        "content": item.get("content", ""),
        "score": item.get("score"),
        "slot": item.get("slot") or source.get("slot"),
        "reason": item.get("reason") or source.get("reason"),
        "evidence_ids": item.get("evidence_ids") or source.get("evidence_ids") or [],
        "node_class": item.get("node_class") or source.get("node_class"),
        "source": source,
    }
