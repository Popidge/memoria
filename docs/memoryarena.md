# MemoryArena

Memoria exposes two benchmark paths for the MemoryArena dataset:

- `memoryarena_proxy`
  - fast local regression for tuning the Memoria engine directly
- `memoryarena_openclaw`
  - live model-in-the-loop execution through OpenClaw

There is also a comparison wrapper:

- `memoryarena_compare`
  - runs `baseline`, `native`, and `prefetch` OpenClaw modes and writes a compact public summary

## Setup

Install the optional benchmark dependency:

```bash
uv sync --extra benchmark
```

The default dataset source is `ZexueHe/memoryarena` on Hugging Face. For offline development, benchmark commands also accept `--data-root` pointing at a local fixture tree with `<suite>/data.jsonl`.

## Benchmark Modes

`baseline`
- OpenClaw only, no Memoria memory path

`native`
- OpenClaw with the Memoria context engine active
- recall happens through real `assemble` and step-time memory updates

`prefetch`
- runner-managed Memoria recall injected into the prompt before the turn
- useful as a sanity/debug comparison, not the target architecture

## Clean Run Protocol

Published comparisons should use the isolated OpenClaw mode now built into the CLI:

- temporary OpenClaw config
- temporary OpenClaw state directory
- temporary benchmark-only workspace
- copied auth profiles from the active OpenClaw install so provider auth still works

This avoids leaking personal workspace memory, old sessions, or Garland-specific day-to-day state into benchmark results.

## Commands

Cache the dataset and print suite counts:

```bash
uv run memoria memoryarena-sync
```

Run the fast local proxy path:

```bash
uv run memoria memoryarena-eval --smoke
uv run memoria memoryarena-eval --suite formal_reasoning_math --full
```

Run one live OpenClaw mode:

```bash
uv run memoria serve
uv run memoria memoryarena-openclaw --memory-mode native --suite group_travel_planner --limit 1 --local
```

Run the full comparison matrix and write the compact public summary:

```bash
uv run memoria serve
uv run memoria memoryarena-compare --full --local
```

Reproduce the sampled live comparison currently checked into the repo:

```bash
uv run memoria serve
uv run memoria memoryarena-compare \
  --suite group_travel_planner \
  --suite formal_reasoning_math \
  --suite formal_reasoning_phys \
  --limit 1 \
  --local
```

For validation/debugging, you can also enable isolated debug toggles:

```bash
uv run memoria memoryarena-compare --smoke --local --raw-stream --debug-commands
```

## Outputs

Each benchmark mode writes local raw artifacts under:

- `artifacts/benchmarks/memoryarena/<timestamp>/summary.json`
- `artifacts/benchmarks/memoryarena/<timestamp>/cases.jsonl`
- `artifacts/benchmarks/memoryarena/<timestamp>/config.json`

The checked-in public summary lives at:

- `docs/benchmarks/memoryarena-latest.json`

That summary is intended for README/docs use. The current checked-in file is a sampled live comparison covering 3 tasks / 16 turns, not a full live sweep of all supported suites. Full raw artifacts stay local and are gitignored.

## Latest Results

Date:
- `2026-04-09`

Live sample setup:
- OpenClaw `2026.4.9 (0512059)`
- OpenRouter `openrouter/minimax/minimax-m2.7`
- Suites: `group_travel_planner`, `formal_reasoning_math`, `formal_reasoning_phys`
- Sample size: 1 task per suite, 16 total turns

Live comparison summary:

| Mode | Token F1 | Travel field coverage | Ok | Blocked | Timeout | Failed |
|---|---:|---:|---:|---:|---:|---:|
| Baseline | 0.0415 | 0.3968 | 12 | 2 | 1 | 1 |
| Native | 0.1022 | 0.5833 | 13 | 2 | 1 | 0 |
| Prefetch | 0.2341 | 0.5556 | 13 | 2 | 1 | 0 |

Full proxy regression summary:
- 4,850 turns across all 5 suites
- support recall@5: `0.5828`
- support precision@5: `0.3411`
- prompt support coverage: `0.5877`
- average prompt size: `107.9` tokens

Methodology note:
- A timed-out live turn now blocks later turns in the same sampled task instead of repeatedly hammering the same OpenClaw session lock. That keeps the summary honest about one underlying provider/runtime failure rather than inflating it into several synthetic failures.

## Notes

- `memoryarena_proxy` is for regression and tuning, not leaderboard claims.
- The live runner currently supports:
  - `group_travel_planner`
  - `formal_reasoning_math`
  - `formal_reasoning_phys`
- `bundled_shopping` and `progressive_search` remain proxy-only for now.
- Native OpenClaw benchmark runs currently depend on the locally patched OpenClaw install on this machine; document that caveat anywhere results are published.
