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
uv run memoria init
uv run memoria ingest-demo sports_query
uv run memoria run-demo sports_query
uv run pytest
```

That should get you from empty repo to a working local prototype in under five minutes.

## CLI

```bash
uv run memoria init
uv run memoria ingest-demo personal_assistant
uv run memoria ingest-demo research_agent
uv run memoria ingest-demo sports_query
uv run memoria run-demo sports_query
uv run memoria search "Toronto Maple Leafs 2025-26 season" --namespace-id demo.sports
uv run memoria consolidate --namespace-id demo.sports --force
uv run memoria export-graph --namespace-id demo.sports --out sports-graph.json
uv run memoria evaluate --name all
```

## Workbench

Memoria now includes a standalone local workbench for memory-system development.
It is intentionally independent from OpenClaw and gives you:

- a local web UI for chat, traces, graph snapshots, corpus inspection, and MemoryArena browsing
- matching CLI commands over the same runtime APIs
- replay mode for deterministic local testing
- OpenAI-compatible API support for live LLM-backed runs

Start the local app:

```bash
uv run memoria workbench-serve
```

Then open `http://127.0.0.1:8080`.

Useful CLI entrypoints:

```bash
uv run memoria workbench-runs
uv run memoria workbench-create-run --provider-type replay --replay-response "Stored."
uv run memoria workbench-chat --provider-type replay --replay-response "Stored."
uv run memoria workbench-chat \
  --provider-type openai-compatible \
  --model-name openai/gpt-4.1-mini \
  --api-base-url https://openrouter.ai/api/v1 \
  --api-key-env OPENROUTER_API_KEY
uv run memoria workbench-trace <run-id>
uv run memoria workbench-corpus <run-id>
uv run memoria workbench-memoryarena
```

The workbench persists run history in the local SQLite database, alongside the existing memory graph and activation trace tables.

## MemoryArena

Memoria now has a [MemoryArena](https://memoryarena.github.io/) benchmark suite for memory-system development.
The dataset is transformed into two derived families:

- `snapshot_lookup`: preload the full task corpus, then test retrieval from pre-existing memory
- `learn_as_you_act`: reveal task state turn by turn, then test whether Memoria writes and later recalls it

Each derived case is tagged with a strand such as `incremental_state_tracking`, `paper_context_recall`, or `structured_plan_continuity`, so runs can target a specific memory behavior.

Install the optional dataset dependency:

```bash
uv sync --extra benchmark
```

Build the derived benchmark files:

```bash
uv run memoria memoryarena-build
```

Run the fast local offline regression path:

```bash
uv run memoria memoryarena-sync
uv run memoria memoryarena-eval --smoke
uv run memoria memoryarena-eval --family learn_as_you_act --strand paper_context_recall
```

Run the v2 experiment scorecard wrapper:

```bash
uv run memoria memoryarena-experiment --label v2-baseline --family all --jobs auto
```

Run the same derived suite through the local workbench agent loop:

```bash
uv run memoria memoryarena-agent-eval --provider-type replay --smoke
```

Run one live OpenClaw mode once your local sidecar and plugin are up:

```bash
uv run memoria serve
uv run memoria memoryarena-openclaw --memory-mode native --suite group_travel_planner --limit 1 --local
```

Run the full baseline/native/prefetch comparison matrix:

```bash
uv run memoria serve
uv run memoria memoryarena-compare --full --local
```

Reproduce the sampled live comparison published in this repo:

```bash
uv run memoria serve
uv run memoria memoryarena-compare \
  --suite group_travel_planner \
  --suite formal_reasoning_math \
  --suite formal_reasoning_phys \
  --limit 1 \
  --local
```

The live runner now supports three comparison modes:

- `baseline`: OpenClaw only
- `native`: OpenClaw plus the Memoria context engine
- `prefetch`: runner-managed prompt injection for sanity/debug comparison

Published benchmark summaries live in [docs/benchmarks/memoryarena-latest.json](/home/jamie/Dev/memoria/docs/benchmarks/memoryarena-latest.json). Full raw artifacts stay local under `artifacts/benchmarks/memoryarena/`.

Benchmark notes live in [docs/memoryarena.md](/home/jamie/Dev/memoria/docs/memoryarena.md).

## Benchmarking

Benchmark comparisons are run in an isolated OpenClaw harness:

- temporary OpenClaw config
- temporary OpenClaw state directory
- temporary benchmark-only workspace
- copied provider auth so the model stack stays identical across modes

That keeps everyday Garland workspace/session memory out of the published numbers.

Current caveat: native step-time benchmark runs rely on the locally patched OpenClaw install used during development, so any published stats should say that explicitly.

## Latest Snapshot

Checked-in benchmark summary:
- Date: `2026-04-09`
- Runtime: OpenClaw `2026.4.9 (0512059)`
- Model: OpenRouter `openrouter/minimax/minimax-m2.7`
- Live sample: 3 tasks / 16 turns across `group_travel_planner`, `formal_reasoning_math`, and `formal_reasoning_phys`

| Mode | Token F1 | Travel field coverage | Ok | Blocked | Timeout | Failed | Avg latency |
|---|---:|---:|---:|---:|---:|---:|---:|
| Baseline | 0.042 | 0.397 | 12 | 2 | 1 | 1 | 37.7s |
| Native | 0.102 | 0.583 | 13 | 2 | 1 | 0 | 44.6s |
| Prefetch | 0.234 | 0.556 | 13 | 2 | 1 | 0 | 30.6s |

Native improves over baseline on this sampled live run by `+0.061` token F1 and `+0.187` travel field coverage. Prefetch performs best here, but it remains a debug ceiling rather than the target architecture because the goal is first-class memory inside the OpenClaw turn loop, not prompt stuffing from outside the loop.

The previous full offline proxy sweep covered all 5 suites and 4,850 turns. On the `2026-04-09` run it reached `0.583` support recall@5, `0.341` support precision@5, `0.588` prompt support coverage, and an average prompt size of about `108` tokens. The current local offline suite supersedes that path with derived `snapshot_lookup` and `learn_as_you_act` families plus write/deferred-recall metrics.

One formal-reasoning physics turn timed out across all live modes on this model/provider stack. Later turns in that same task are marked `blocked` in the benchmark instead of being counted as extra failures, so the summary reflects the real provider/runtime limitation more honestly.

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

More detail lives in [docs/architecture.md](/home/jamie/Dev/memoria/docs/architecture.md), [docs/scoring.md](/home/jamie/Dev/memoria/docs/scoring.md), [docs/demo_notes.md](/home/jamie/Dev/memoria/docs/demo_notes.md), and [docs/memoryarena.md](/home/jamie/Dev/memoria/docs/memoryarena.md).

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
