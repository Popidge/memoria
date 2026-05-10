# Architecture

Memoria has one canonical local engine and five modules:

## Ingestion

Ingestion writes raw episodes with scopes:
- namespace
- user
- agent
- session

The episode row stores the raw content, a lightweight summary, metadata, and an embedding.
Each episode also gets `EpisodeChunk` memory atoms. Episodes remain the documentary source of truth; chunks carry precise evidence text, chunk type, summary, embedding, salience seed, and metadata for retrieval.

## Graph Model

The graph is stored in ordinary SQLite tables:
- `Episode` holds documentary memory
- `Entity` and `Fact` hold derived semantic memory
- `SummaryNode` holds compact rollups
- `EpisodeChunk` holds atom-sized evidence beneath episodes
- `ProvenanceLink` ties facts back to source episodes
- `GraphEdge` gives traversal structure for activation spread
- `NodeDescriptor` and `EdgeDescriptor` add v2 classes, relation classes, confidence, facets, and evidence counts without replacing the base graph

Node identity is practical and explicit:
- `Entity:<id>`
- `Fact:<id>`
- `SummaryNode:<id>`
- `Episode:<id>`
- `EpisodeChunk:<id>`

## Retrieval

Retrieval is explainable rather than fancy:
- semantic similarity
- keyword overlap
- scope bonus
- confidence filtering

This searches summaries, facts, entities, chunks, and episodes. Chunks are preferred when they provide precise evidence, while episodes remain available for broader documentary recall.

## Activation

Each step produces a text signal. The engine scores candidates, spreads activation over one-hop edges after a primary memory is hot enough, applies reinforcement-aware decay and inhibition, then promotes compact content into working memory slots:
- `primary`
- `linked`
- `ambient`
- `evidence`

Activation state and working memory are persisted for every step so runs are inspectable after the fact. Sidecar and workbench recall now return a structured `MemoryContextPacket`; `prompt_addition` is rendered from that packet for compatibility with OpenClaw.

## Workbench

The local workbench sits on top of the canonical engine. It persists experiment runs and turns, records prompt additions and provider payloads, and exposes the same runtime through CLI commands and a small HTTP UI. Replay and OpenAI-compatible providers share the same `ProviderConfig` contract so benchmark and manual runs exercise the same code path.

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
