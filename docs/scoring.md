# Scoring

Default activation weights live in `memoria.toml`.

## Inputs

Each step builds:
- a query embedding
- a keyword set
- linked entities detected from capitalised phrases

## Candidate Generation

Candidates come from:
- direct retrieval hits
- one-hop neighbours of hot nodes from the previous step

## Formula

```text
activation =
  0.45 * semantic_score +
  0.20 * keyword_score +
  0.15 * spread_score +
  0.10 * recency_score +
  0.10 * importance_score -
  inhibition_penalty
```

The implementation also carries forward a decayed slice of previous activation to make multi-step runs feel temporal instead of stateless.

## Inhibition

Current penalties:
- recently surfaced nodes
- high-degree generic nodes
- near-duplicate summaries
- per-type promotion caps

## Promotion Policy

Working memory prefers:
1. summary nodes
2. fact summaries
3. entity summaries
4. raw episode snippets

Episode snippets appear only at the higher threshold or when provenance is explicitly requested.
