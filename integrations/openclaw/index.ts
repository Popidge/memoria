import { Type } from "@sinclair/typebox";
import { definePluginEntry } from "openclaw/plugin-sdk/core";

import {
  MemoriaClient,
  buildSessionEnvelope,
  formatRecallSummary,
  normaliseOpenClawMessage,
  resolvePluginConfig,
} from "./src/client.ts";

const recallParams = Type.Object({
  text: Type.String({ minLength: 1 }),
  step_type: Type.Optional(Type.String()),
  linked_tool_name: Type.Optional(Type.String()),
  prompt_limit: Type.Optional(Type.Number({ minimum: 1, maximum: 8 })),
});

const storeParams = Type.Object({
  content: Type.String({ minLength: 1 }),
  role: Type.Optional(Type.String()),
  source_type: Type.Optional(Type.String()),
  metadata: Type.Optional(Type.Record(Type.String(), Type.Any())),
});

export default definePluginEntry({
  id: "memoria-openclaw",
  name: "Memoria",
  description: "First-class Memoria integration for OpenClaw via a local sidecar and context engine.",
  kind: "memory",
  register(api) {
    const pluginConfig = resolvePluginConfig(api.pluginConfig);
    api.registerMemoryCapability({
      promptBuilder: () => [],
    });

    api.registerTool(
      (ctx: any) => {
        const pluginConfig = resolvePluginConfig(api.pluginConfig);
        const client = new MemoriaClient(pluginConfig.baseUrl);
        const session = buildSessionEnvelope(ctx, pluginConfig);
        return {
          name: "memoria_recall",
          description: "Inspect what Memoria would currently recall for the active session.",
          parameters: recallParams,
          async execute(_id: string, params: any) {
            await client.bootstrap(session);
            const result = await client.recall(session, {
              text: params.text,
              step_type: params.step_type ?? "user_input",
              linked_tool_name: params.linked_tool_name,
              prompt_limit: params.prompt_limit ?? pluginConfig.promptLimit,
            });
            return {
              content: [{ type: "text", text: formatRecallSummary(result) }],
            };
          },
        };
      },
      { names: ["memoria_recall"] },
    );

    api.registerTool(
      (ctx: any) => {
        const pluginConfig = resolvePluginConfig(api.pluginConfig);
        const client = new MemoriaClient(pluginConfig.baseUrl);
        const session = buildSessionEnvelope(ctx, pluginConfig);
        return {
          name: "memoria_store",
          description: "Manually persist an important message into Memoria for debugging or curation.",
          parameters: storeParams,
          async execute(_id: string, params: any) {
            await client.bootstrap(session);
            const result = await client.ingest(session, [
              {
                role: params.role ?? "assistant",
                source_type: params.source_type ?? "agent_message",
                content: params.content,
                metadata: params.metadata ?? {},
              },
            ]);
            return {
              content: [{ type: "text", text: `Stored ${result.stored_count} message(s) in Memoria.` }],
            };
          },
        };
      },
      { names: ["memoria_store"] },
    );

    api.registerTool(
      (ctx: any) => {
        const pluginConfig = resolvePluginConfig(api.pluginConfig);
        const client = new MemoriaClient(pluginConfig.baseUrl);
        const session = buildSessionEnvelope(ctx, pluginConfig);
        return {
          name: "memoria_trace",
          description: "Inspect the latest activation trace for the current Memoria session.",
          parameters: Type.Object({}),
          async execute() {
            const bootstrap = await client.bootstrap(session);
            const trace = await client.trace(String(bootstrap.session_key));
            return {
              content: [{ type: "text", text: JSON.stringify(trace, null, 2) }],
            };
          },
        };
      },
      { names: ["memoria_trace"] },
    );

    api.on("agent_end", async (event: any) => {
      const pluginConfig = resolvePluginConfig(api.pluginConfig);
      if (!pluginConfig.autoStoreAssistantTurns) {
        return;
      }
      const session = buildSessionEnvelope(event, pluginConfig);
      const client = new MemoriaClient(pluginConfig.baseUrl);
      const assistantMessages = Array.isArray(event?.messages)
        ? event.messages
            .map((message: any) => normaliseOpenClawMessage(message))
            .filter((message: any) => message?.role === "assistant")
        : [];
      if (assistantMessages.length === 0) {
        return;
      }
      await client.afterTurn(session, assistantMessages);
    });
  },
});
