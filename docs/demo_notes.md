# Demo Notes

## Personal Assistant

Purpose:
- show that not every remembered preference should surface on every turn

Expected behavior:
- cafe planning should activate coffee and restaurant preference facts
- project context should surface because Orion is part of the prompt
- unrelated travel preferences should be less prominent

## Research Agent

Purpose:
- show step-wise movement across memory regions during planning and synthesis

Expected behavior:
- planning step lights up explicit memory CRUD and temporal graph summaries
- reasoning step shifts toward provenance and incremental update facts
- drafting step should keep the working memory compact

## Sports Query

Purpose:
- show the clearest “light-up” trace

Expected behavior:
- direct hits for `Toronto Maple Leafs`, `NHL`, and `2025-26`
- spread to related fact and summary nodes
- promotion of season record and playoff outcome
- provenance-aware episode snippets available on request
