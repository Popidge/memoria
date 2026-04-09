import { delegateCompactionToRuntime } from "openclaw/plugin-sdk/core";

import {
  MemoriaClient,
  buildSessionEnvelope,
  collectTurnMessages,
  extractPromptText,
  openClawContentToText,
  resolvePluginConfig,
} from "./src/client.ts";

function stepTypeForKind(stepKind: string): string {
  return stepKind === "tool_result" ? "tool_result" : "assistant_message";
}

export function createMemoriaContextEngine(openClawConfig: any) {
  const pluginConfig = resolvePluginConfig(openClawConfig?.plugins?.entries?.["memoria-openclaw"]?.config);
  const client = new MemoriaClient(pluginConfig.baseUrl);

  return {
    info: {
      id: "memoria",
      name: "Memoria",
      version: "0.2.0",
    },

    async bootstrap(ctx: any) {
      const session = buildSessionEnvelope(ctx, pluginConfig);
      await client.bootstrap(session);
    },

    async assemble(ctx: any) {
      const session = buildSessionEnvelope(ctx, pluginConfig);
      await client.bootstrap(session);
      const promptText = extractPromptText(ctx.prompt, Array.isArray(ctx.messages) ? ctx.messages : []);
      if (!promptText) {
        return {
          messages: ctx.messages,
          estimatedTokens: 0,
        };
      }
      const result = await client.recall(session, {
        text: promptText,
        step_type: "user_input",
        metadata: {
          phase: "assemble",
          message_count: Array.isArray(ctx.messages) ? ctx.messages.length : 0,
        },
        prompt_limit: pluginConfig.promptLimit,
      });
      return {
        messages: ctx.messages,
        estimatedTokens: 0,
        ...(result.prompt_addition ? { systemPromptAddition: result.prompt_addition } : {}),
      };
    },

    async onStep(ctx: any) {
      const session = buildSessionEnvelope(ctx, pluginConfig);
      await client.bootstrap(session);
      const newMessages = Array.isArray(ctx.newMessages) ? ctx.newMessages : [];
      const newestMessage = newMessages.at(-1);
      const text = openClawContentToText(newestMessage?.content);
      if (!text) {
        return;
      }
      const linkedToolName =
        typeof newestMessage?.toolName === "string" && newestMessage.toolName.trim()
          ? newestMessage.toolName.trim()
          : typeof ctx.toolName === "string" && ctx.toolName.trim()
            ? ctx.toolName.trim()
            : undefined;
      const result = await client.recall(session, {
        text,
        step_type: stepTypeForKind(String(ctx.stepKind ?? "assistant_message")),
        linked_tool_name: linkedToolName,
        metadata: {
          phase: "step",
          step_index: ctx.stepIndex ?? null,
          step_kind: ctx.stepKind ?? "assistant_message",
        },
        prompt_limit: pluginConfig.promptLimit,
      });
      if (!result.prompt_addition) {
        return;
      }
      return {
        systemPromptAddition: result.prompt_addition,
      };
    },

    async afterTurn(ctx: any) {
      const session = buildSessionEnvelope(ctx, pluginConfig);
      const messages = collectTurnMessages(Array.isArray(ctx.messages) ? ctx.messages : [], Number(ctx.prePromptMessageCount ?? 0));
      if (messages.length === 0) {
        return;
      }
      await client.afterTurn(session, messages, {
        consolidate: true,
        force: false,
      });
    },

    async compact(ctx: any) {
      return delegateCompactionToRuntime(ctx);
    },

    async dispose() {},
  };
}
