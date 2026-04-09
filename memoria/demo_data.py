from __future__ import annotations


def demo_catalog() -> dict[str, dict]:
    return {
        "personal_assistant": {
            "namespace_id": "demo.personal",
            "user_id": "alex",
            "agent_id": "pa",
            "session_id": "sess-personal",
            "episodes": [
                {"role": "user", "source_type": "user_message", "content": "I prefer oat milk in my coffee and I dislike loud restaurants."},
                {"role": "user", "source_type": "user_message", "content": "I am working on the Orion garden renovation project with Maya."},
                {"role": "agent", "source_type": "agent_message", "content": "We noted Alex's coffee preference and the Orion project."},
                {"role": "user", "source_type": "user_message", "content": "For travel, I like window seats and quiet boutique hotels."},
            ],
            "steps": [
                {"step_type": "user_input", "text": "Help me plan a cafe meeting for the Orion project."},
                {"step_type": "reasoning_step", "text": "Which personal preferences matter for a cafe recommendation?"},
                {"step_type": "answer_drafting_step", "text": "Draft an answer that uses only relevant preferences."},
            ],
            "expected": ["SummaryNode", "Fact", "Entity"],
        },
        "research_agent": {
            "namespace_id": "demo.research",
            "user_id": "rina",
            "agent_id": "researcher",
            "session_id": "sess-research",
            "episodes": [
                {"role": "user", "source_type": "user_message", "content": "We are researching temporal knowledge graphs for agent memory."},
                {"role": "user", "source_type": "imported_doc", "content": "Paper note: provenance links help explain where facts came from and when they were valid."},
                {"role": "tool", "source_type": "tool_result", "content": "Graphiti-style episode centric graphs preserve source episodes and allow incremental updates."},
                {"role": "tool", "source_type": "tool_result", "content": "Mem0-style systems keep explicit add, search, update, and delete operations with scoped memory."},
            ],
            "steps": [
                {"step_type": "planner_step", "text": "Plan a comparison between explicit memory CRUD and temporal graph memory."},
                {"step_type": "reasoning_step", "text": "What evidence do we already have about provenance and incremental updates?"},
                {"step_type": "answer_drafting_step", "text": "Prepare a compact synthesis for the architecture section."},
            ],
            "expected": ["Fact", "SummaryNode"],
        },
        "sports_query": {
            "namespace_id": "demo.sports",
            "user_id": "sam",
            "agent_id": "sportsdesk",
            "session_id": "sess-sports",
            "episodes": [
                {"role": "tool", "source_type": "tool_result", "content": "Toronto Maple Leafs finished the 2025-26 NHL season with 48 wins, 26 losses, and 8 overtime losses."},
                {"role": "tool", "source_type": "tool_result", "content": "Toronto Maple Leafs were eliminated in the second round of the 2025-26 Stanley Cup Playoffs."},
                {"role": "user", "source_type": "user_message", "content": "Last month we discussed the NHL standings and Toronto Maple Leafs playoff chances."},
                {"role": "tool", "source_type": "tool_result", "content": "NHL procedure note: season summaries should cite record, playoff outcome, and notable context."},
            ],
            "steps": [
                {"step_type": "user_input", "text": "I've been asked to see how the Toronto Maple Leafs performed in the 2025-26 NHL season.", "metadata": {"include_provenance": True}},
                {"step_type": "planner_step", "text": "Find season record, playoff outcome, and any reusable NHL reporting guidance."},
                {"step_type": "answer_drafting_step", "text": "Draft a concise sports summary with provenance-aware support."},
            ],
            "expected": ["Entity", "Fact", "SummaryNode", "Episode"],
        },
    }


def load_demo(name: str) -> dict:
    demos = demo_catalog()
    if name not in demos:
        raise KeyError(f"Unknown demo {name}")
    return demos[name]
