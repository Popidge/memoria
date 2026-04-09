# Memoria

Memoria is a local Python prototype for agent memory. It tests whether an episode-centric temporal graph plus an activation controller can keep working memory compact while still surfacing the right context at the right step.

The design is inspired by:
- Mem0-style explicit memory lifecycle operations and scoped memory layers
- Graphiti-style episode-first knowledge graphs with provenance and temporal facts

What is novel here is the activation layer. Instead of treating retrieval as a one-shot search, each agent step updates a live salience field over the graph. Only nodes that become hot enough are promoted into working memory, and the runtime prefers compact summaries before raw snippets.

## What It Stores

Episodes are the source of truth. Consolidation derives:
- entities
- temporal facts
- summary nodes
- provenance links back to episodes
- graph edges for traversal and spreading activation

Key persisted tables:
- `Episode`
- `Entity`
- `Fact`
- `ProvenanceLink`
- `SummaryNode`
- `GraphEdge`
- `ActivationState`
- `WorkingMemoryItem`

Facts preserve `valid_from`, optional `valid_to`, confidence, and optional supersession. Activation and working memory rows are persisted per run so the prototype is easy to debug after the fact.

## How Activation Works

For each agent step, the engine:
1. Builds a signal from the step text using embeddings, keywords, and detected entities.
2. Retrieves direct candidates by semantic similarity, keyword overlap, and scope match.
3. Pulls in one-hop neighbours of previously hot nodes.
4. Scores candidates with configurable weights.
5. Applies decay, recency, importance priors, and inhibition penalties.
6. Persists the full activation trace.
7. Promotes the hottest nodes into working memory with this preference order:
   `SummaryNode -> Fact -> Entity -> Episode snippet`

The default scoring formula is:

```text
activation =
  0.45 * semantic_score +
  0.20 * keyword_score +
  0.15 * spread_score +
  0.10 * recency_score +
  0.10 * importance_score -
  inhibition_penalty
```

## Quick Start

```bash
uv sync
uv run tame init
uv run tame ingest-demo sports_query
uv run tame run-demo sports_query
uv run pytest
```

That should get you from empty repo to a working local prototype in under five minutes.

## CLI

```bash
uv run tame init
uv run tame ingest-demo personal_assistant
uv run tame ingest-demo research_agent
uv run tame ingest-demo sports_query
uv run tame run-demo sports_query
uv run tame search "Toronto Maple Leafs 2025-26 season" --namespace-id demo.sports
uv run tame consolidate --namespace-id demo.sports --force
uv run tame export-graph --namespace-id demo.sports --out sports-graph.json
uv run tame evaluate --name all
```

## Demos

`personal_assistant`
- recalls preferences and ongoing projects only when they matter

`research_agent`
- lights up different memory regions across planning, reasoning, and synthesis

`sports_query`
- shows direct activation of Leafs / NHL / season context, spread to related summaries, and provenance-aware support

## Architecture Notes

- SQLite is the only database.
- SQLAlchemy manages the schema.
- `sentence-transformers` is supported, but tests and local demos work with a deterministic hashed-text fallback so the core flow stays offline-friendly.
- Consolidation is synchronous in the MVP, but kept as a separate module boundary.
- Retrieval is intentionally simple and explainable rather than optimized.

More detail lives in [docs/architecture.md](/home/jamie/Dev/memoria/docs/architecture.md), [docs/scoring.md](/home/jamie/Dev/memoria/docs/scoring.md), and [docs/demo_notes.md](/home/jamie/Dev/memoria/docs/demo_notes.md).

## Limitations

- extraction is heuristic and narrow on purpose
- graph summaries are simple rollups, not deep abstractions
- there is no LLM requirement for core tests
- baseline retrieval is small-scale and in-memory over SQLite rows
- this is not production infra, multi-tenant infra, or a serving system

## Next Steps

- richer fact extraction and better contradiction handling
- better entity merging with reference-safe rewrites
- learned or task-specific activation policies
- stronger hybrid retrieval indexing
- explicit agent-facing APIs for provenance requests and step annotations
