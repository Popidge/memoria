# Memoria OpenClaw Integration

This package is the in-repo OpenClaw integration for Memoria. It now supports a
first-class `memoria` context engine so recall happens during OpenClaw turn
assembly and step updates, instead of depending on the model to call a tool.

Important caveat: the native step-time benchmark path currently relies on the
locally patched OpenClaw install on this machine. The Memoria repo documents
that requirement; it does not yet ship a portable OpenClaw patch helper.

## What is implemented

- `memoria` context engine
- step-aware recall updates wired into the OpenClaw runner
- `memoria_recall`, `memoria_store`, and `memoria_trace` as debug tools
- sidecar client for `memoria serve`

## Local setup

1. Start the Memoria sidecar:

```bash
uv run memoria serve
```

2. Link the plugin into a local OpenClaw checkout:

```bash
openclaw plugins install -l /home/jamie/Dev/memoria/integrations/openclaw
```

3. Enable and configure the plugin in `openclaw.json`:

```json5
{
  plugins: {
    slots: {
      contextEngine: "memoria",
    },
    entries: {
      "memoria-openclaw": {
        enabled: true,
        config: {
          baseUrl: "http://127.0.0.1:18733",
          namespaceId: "openclaw.default",
          promptLimit: 4,
        },
      },
    },
  },
}
```

4. Restart the OpenClaw gateway and verify the plugin is loaded:

```bash
openclaw plugins list --enabled
openclaw plugins inspect memoria-openclaw
```

## Benchmark note

For publishable MemoryArena runs, prefer the isolated benchmark harness in this
repo instead of your everyday OpenClaw profile:

```bash
uv run memoria serve
uv run memoria memoryarena-compare --full --local
```

That path creates a temporary OpenClaw config, state directory, and benchmark
workspace so Garland's normal workspace/session memory does not contaminate the
results.

## First live test

Use the personal-assistant seed flow:

- ingest the seeded preference/project messages into Memoria
- ask OpenClaw for help planning a cafe meeting for the Orion project
- verify the answer favors oat milk, quiet venue preference, and Orion project context
- verify travel preferences stay out of the answer
