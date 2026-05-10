export type MemoriaSessionEnvelope = {
  namespace_id: string;
  user_id?: string | null;
  agent_id?: string | null;
  session_id?: string | null;
};

export type MemoriaMessage = {
  role?: string | null;
  source_type?: string;
  content: string;
  timestamp?: string;
  metadata?: Record<string, unknown>;
};

export type MemoriaRecallResponse = {
  session_key: string;
  run_id: string;
  trace_id: string;
  step_index: number | null;
  prompt_addition: string;
  memory_context_packet?: Record<string, unknown>;
  working_memory: Array<{
    node_key: string;
    content_type: string;
    content: string;
    score: number;
    slot?: string;
    reason?: string;
    evidence_ids?: Array<number | string>;
    node_class?: string | null;
    source: Record<string, unknown>;
  }>;
};

export type MemoriaPluginConfig = {
  baseUrl?: string;
  namespaceId?: string;
  autoStoreAssistantTurns?: boolean;
  promptLimit?: number;
};

function trimTrailingSlash(value: string): string {
  return value.endsWith("/") ? value.slice(0, -1) : value;
}

export class MemoriaClient {
  private readonly baseUrl: string;

  constructor(baseUrl: string) {
    this.baseUrl = trimTrailingSlash(baseUrl);
  }

  async health(): Promise<Record<string, unknown>> {
    return this.request("GET", "/health");
  }

  async bootstrap(session: MemoriaSessionEnvelope): Promise<Record<string, unknown>> {
    return this.request("POST", "/sessions/bootstrap", session);
  }

  async ingest(
    session: MemoriaSessionEnvelope,
    messages: MemoriaMessage[],
    options?: { consolidate?: boolean; force?: boolean },
  ): Promise<Record<string, unknown>> {
    return this.request("POST", "/sessions/ingest", {
      ...session,
      messages,
      consolidate: options?.consolidate ?? true,
      force: options?.force ?? false,
    });
  }

  async recall(
    session: MemoriaSessionEnvelope,
    params: {
      text: string;
      step_type?: string;
      linked_tool_name?: string;
      metadata?: Record<string, unknown>;
      prompt_limit?: number;
    },
  ): Promise<MemoriaRecallResponse> {
    return this.request("POST", "/sessions/recall", {
      ...session,
      ...params,
    });
  }

  async afterTurn(
    session: MemoriaSessionEnvelope,
    messages: MemoriaMessage[],
    options?: { consolidate?: boolean; force?: boolean },
  ): Promise<Record<string, unknown>> {
    return this.request("POST", "/sessions/after-turn", {
      ...session,
      messages,
      consolidate: options?.consolidate ?? true,
      force: options?.force ?? false,
    });
  }

  async trace(sessionKey: string): Promise<Record<string, unknown>> {
    const encoded = encodeURIComponent(sessionKey);
    return this.request("GET", `/sessions/${encoded}/trace`);
  }

  private async request(method: string, path: string, body?: unknown): Promise<any> {
    const response = await fetch(`${this.baseUrl}${path}`, {
      method,
      headers: {
        "Content-Type": "application/json",
      },
      body: body === undefined ? undefined : JSON.stringify(body),
    });
    const payload = await response.json();
    if (!response.ok) {
      const message = typeof payload?.error === "string" ? payload.error : `Memoria request failed: ${response.status}`;
      throw new Error(message);
    }
    return payload;
  }
}

export function resolvePluginConfig(rawConfig: unknown): Required<MemoriaPluginConfig> {
  const config = (rawConfig ?? {}) as MemoriaPluginConfig;
  return {
    baseUrl: config.baseUrl ?? "http://127.0.0.1:18733",
    namespaceId: config.namespaceId ?? "openclaw.default",
    autoStoreAssistantTurns: config.autoStoreAssistantTurns ?? false,
    promptLimit: config.promptLimit ?? 4,
  };
}

function agentIdFromSessionKey(sessionKey: unknown): string | null {
  if (typeof sessionKey !== "string" || !sessionKey.trim()) {
    return null;
  }
  const parts = sessionKey.split(":");
  if (parts.length >= 2 && parts[0] === "agent" && parts[1]) {
    return parts[1];
  }
  return null;
}

export function buildSessionEnvelope(context: any, pluginConfig: Required<MemoriaPluginConfig>): MemoriaSessionEnvelope {
  return {
    namespace_id: context?.namespaceId ?? pluginConfig.namespaceId,
    user_id: context?.userId ?? context?.senderId ?? context?.session?.userId ?? null,
    agent_id:
      context?.agentId ??
      context?.session?.agentId ??
      agentIdFromSessionKey(context?.sessionKey) ??
      "openclaw",
    session_id: context?.sessionId ?? context?.sessionKey ?? context?.session?.id ?? "default-session",
  };
}

export function formatRecallSummary(result: MemoriaRecallResponse): string {
  if (result.working_memory.length === 0) {
    return "No relevant Memoria recall found for this turn.";
  }
  const lines = result.working_memory.map((item) => `- [${item.content_type}] ${item.content}`);
  return [`Memoria recall (trace ${result.trace_id}):`, ...lines].join("\n");
}

function stringifyJson(value: unknown): string {
  try {
    return JSON.stringify(value);
  } catch {
    return String(value ?? "");
  }
}

function appendNonEmpty(parts: string[], value: unknown): void {
  if (typeof value === "string" && value.trim()) {
    parts.push(value.trim());
    return;
  }
  if (value == null) {
    return;
  }
  const rendered = stringifyJson(value).trim();
  if (rendered) {
    parts.push(rendered);
  }
}

export function openClawContentToText(content: unknown): string {
  if (typeof content === "string") {
    return content.trim();
  }
  if (!Array.isArray(content)) {
    return "";
  }
  const parts: string[] = [];
  for (const item of content) {
    if (!item || typeof item !== "object") {
      continue;
    }
    const record = item as Record<string, unknown>;
    if (record.type === "text") {
      appendNonEmpty(parts, record.text);
      continue;
    }
    if (record.type === "toolCall") {
      const name = typeof record.name === "string" ? record.name.trim() : "tool";
      const argumentsText =
        typeof record.arguments === "string" ? record.arguments.trim() : stringifyJson(record.arguments ?? {});
      parts.push(`Tool call ${name}: ${argumentsText}`);
      continue;
    }
    if (record.type === "thinking") {
      continue;
    }
    appendNonEmpty(parts, record.text ?? record.content);
  }
  return parts.join("\n").trim();
}

export function extractPromptText(prompt: unknown, messages: unknown[]): string {
  if (typeof prompt === "string" && prompt.trim()) {
    return prompt.trim();
  }
  for (let index = messages.length - 1; index >= 0; index -= 1) {
    const message = messages[index] as Record<string, unknown> | undefined;
    if (!message || message.role !== "user") {
      continue;
    }
    const text = openClawContentToText(message.content);
    if (text) {
      return text;
    }
  }
  return "";
}

function inferSourceType(role: string): string {
  const normalizedRole = role.trim().toLowerCase();
  if (normalizedRole === "assistant") {
    return "agent_message";
  }
  if (normalizedRole === "toolresult" || normalizedRole === "tool") {
    return "tool_result";
  }
  if (normalizedRole === "system") {
    return "system_note";
  }
  return "user_message";
}

export function normaliseOpenClawMessage(message: any, metadata?: Record<string, unknown>): MemoriaMessage | null {
  if (!message || typeof message !== "object") {
    return null;
  }
  const roleRaw = typeof message.role === "string" ? message.role : "";
  if (!roleRaw) {
    return null;
  }
  const role = roleRaw === "toolResult" ? "tool" : roleRaw;
  const text = openClawContentToText(message.content);
  if (!text) {
    return null;
  }
  const nextMetadata: Record<string, unknown> = {
    ...(metadata ?? {}),
  };
  if (typeof message.phase === "string" && message.phase) {
    nextMetadata.phase = message.phase;
  }
  if (typeof message.toolName === "string" && message.toolName) {
    nextMetadata.tool_name = message.toolName;
  }
  if (typeof message.toolCallId === "string" && message.toolCallId) {
    nextMetadata.tool_call_id = message.toolCallId;
  }
  return {
    role,
    source_type: inferSourceType(roleRaw),
    content: text,
    metadata: Object.keys(nextMetadata).length > 0 ? nextMetadata : undefined,
  };
}

export function collectTurnMessages(messages: any[], prePromptMessageCount: number): MemoriaMessage[] {
  const collected: MemoriaMessage[] = [];
  const maybeUserIndex = prePromptMessageCount - 1;
  if (maybeUserIndex >= 0) {
    const maybeUser = normaliseOpenClawMessage(messages[maybeUserIndex], { turn_role: "trigger_user" });
    if (maybeUser?.role === "user") {
      collected.push(maybeUser);
    }
  }
  for (const message of messages.slice(prePromptMessageCount)) {
    const normalised = normaliseOpenClawMessage(message);
    if (normalised) {
      collected.push(normalised);
    }
  }
  return collected;
}
