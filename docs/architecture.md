# Architecture

Memoria has one canonical local engine and five modules:

## Ingestion

Ingestion writes raw episodes with scopes:
- namespace
- user
- agent
- session

The episode row stores the raw content, a lightweight summary, metadata, and an embedding.

## Graph Model

The graph is stored in ordinary SQLite tables:
- `Episode` holds documentary memory
- `Entity` and `Fact` hold derived semantic memory
- `SummaryNode` holds compact rollups
- `ProvenanceLink` ties facts back to source episodes
- `GraphEdge` gives traversal structure for activation spread

Node identity is practical and explicit:
- `Entity:<id>`
- `Fact:<id>`
- `SummaryNode:<id>`
- `Episode:<id>`

## Retrieval

Retrieval is explainable rather than fancy:
- semantic similarity
- keyword overlap
- scope bonus
- confidence filtering

This is used both as a standalone baseline and as the candidate generator for activation.

## Activation

Each step produces a text signal. The engine scores candidates, spreads activation over one-hop edges, applies decay and inhibition, then promotes compact content into working memory. Activation state and working memory are persisted for every step so runs are inspectable after the fact.

## Consolidation

Consolidation is the dreaming pass. It:
- summarises new episodes
- extracts entities
- resolves or creates canonical entities
- creates or strengthens facts
- preserves provenance
- refreshes summary nodes
- links graph edges

It runs synchronously in the MVP, but the code boundary is already separate enough to move later.
