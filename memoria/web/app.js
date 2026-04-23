const state = {
  activeRunId: null,
  runs: [],
};

const elements = {
  runForm: document.querySelector("#run-form"),
  runsList: document.querySelector("#runs-list"),
  refreshRuns: document.querySelector("#refresh-runs"),
  activeRunTitle: document.querySelector("#active-run-title"),
  activeRunMeta: document.querySelector("#active-run-meta"),
  runStatus: document.querySelector("#run-status"),
  chatLog: document.querySelector("#chat-log"),
  chatForm: document.querySelector("#chat-form"),
  chatInput: document.querySelector("#chat-input"),
  sendButton: document.querySelector("#send-button"),
  traceView: document.querySelector("#trace-view"),
  graphView: document.querySelector("#graph-view"),
  corpusView: document.querySelector("#corpus-view"),
  memoryarenaView: document.querySelector("#memoryarena-view"),
  refreshTrace: document.querySelector("#refresh-trace"),
  refreshGraph: document.querySelector("#refresh-graph"),
  refreshCorpus: document.querySelector("#refresh-corpus"),
  refreshMemoryArena: document.querySelector("#refresh-memoryarena"),
};

async function request(url, options = {}) {
  const response = await fetch(url, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  const payload = await response.json();
  if (!response.ok) {
    throw new Error(payload.error || `Request failed: ${response.status}`);
  }
  return payload;
}

function pretty(value) {
  return JSON.stringify(value, null, 2);
}

function setBusy(label) {
  elements.runStatus.textContent = label;
}

function clearBusy() {
  elements.runStatus.textContent = state.activeRunId ? "ready" : "idle";
}

function renderRuns() {
  elements.runsList.innerHTML = "";
  for (const run of state.runs) {
    const item = document.createElement("button");
    item.type = "button";
    item.className = `run-item ${run.id === state.activeRunId ? "active" : ""}`;
    item.innerHTML = `
      <strong>${escapeHtml(run.title)}</strong>
      <div class="muted">${escapeHtml(run.provider_type)} · ${escapeHtml(run.model_name)}</div>
      <div class="muted">${run.turn_count} turn(s)</div>
    `;
    item.addEventListener("click", () => selectRun(run.id));
    elements.runsList.appendChild(item);
  }
}

function renderChat(turns) {
  elements.chatLog.innerHTML = "";
  if (!turns.length) {
    elements.chatLog.innerHTML = `<p class="muted">No turns yet. Send a message to start tracing memory behavior.</p>`;
    return;
  }
  for (const turn of turns) {
    appendMessage("user", turn.user_message, `Turn ${turn.turn_index} · user`);
    appendMessage("assistant", turn.assistant_message, `Turn ${turn.turn_index} · assistant`);
  }
}

function appendMessage(kind, text, label) {
  const message = document.createElement("article");
  message.className = `message ${kind}`;
  message.innerHTML = `<span class="label">${escapeHtml(label)}</span>${escapeHtml(text || "")}`;
  elements.chatLog.appendChild(message);
}

async function loadRuns() {
  const payload = await request("/api/runs");
  state.runs = payload.runs;
  renderRuns();
  if (!state.activeRunId && state.runs.length) {
    await selectRun(state.runs[0].id);
  }
}

async function selectRun(runId) {
  state.activeRunId = runId;
  renderRuns();
  await Promise.all([
    loadRun(),
    loadTurns(),
    loadTrace(),
    loadGraph(),
    loadCorpus(),
  ]);
}

async function loadRun() {
  if (!state.activeRunId) {
    return;
  }
  const run = await request(`/api/runs/${state.activeRunId}`);
  elements.activeRunTitle.textContent = run.title;
  elements.activeRunMeta.textContent = `${run.provider_type} · ${run.model_name} · namespace ${run.namespace_id}`;
  elements.runStatus.textContent = run.status;
  elements.chatInput.disabled = false;
  elements.sendButton.disabled = false;
}

async function loadTurns() {
  if (!state.activeRunId) {
    return;
  }
  const payload = await request(`/api/runs/${state.activeRunId}/turns`);
  renderChat(payload.turns);
}

async function loadTrace() {
  if (!state.activeRunId) {
    return;
  }
  const payload = await request(`/api/runs/${state.activeRunId}/trace`);
  const latestStep = payload.steps[payload.steps.length - 1] || null;
  elements.traceView.textContent = pretty({
    step_count: payload.steps.length,
    latest_step: latestStep,
  });
}

async function loadGraph() {
  if (!state.activeRunId) {
    return;
  }
  const payload = await request(`/api/runs/${state.activeRunId}/graph`);
  elements.graphView.textContent = pretty({
    namespace_id: payload.namespace_id,
    node_count: payload.node_count,
    edge_count: payload.edge_count,
    latest_step_index: payload.latest_step_index,
    top_nodes: payload.nodes.slice(0, 20),
  });
}

async function loadCorpus() {
  if (!state.activeRunId) {
    return;
  }
  const payload = await request(`/api/runs/${state.activeRunId}/corpus`);
  elements.corpusView.textContent = pretty(payload);
}

async function loadMemoryArena() {
  try {
    const suites = await request("/api/memoryarena/suites");
    elements.memoryarenaView.textContent = pretty(suites);
  } catch (error) {
    elements.memoryarenaView.textContent = String(error.message || error);
  }
}

elements.runForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  const form = new FormData(elements.runForm);
  const payload = {
    title: form.get("title"),
    system_prompt: form.get("system_prompt"),
    prompt_limit: Number(form.get("prompt_limit") || 4),
    provider: {
      provider_type: form.get("provider_type"),
      model_name: form.get("model_name"),
      api_base_url: form.get("api_base_url") || null,
      api_key_env: form.get("api_key_env") || null,
    },
  };
  try {
    setBusy("creating");
    const run = await request("/api/runs", {
      method: "POST",
      body: JSON.stringify(payload),
    });
    state.activeRunId = run.id;
    await loadRuns();
    await selectRun(run.id);
  } catch (error) {
    alert(error.message || error);
  } finally {
    clearBusy();
  }
});

elements.chatForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  if (!state.activeRunId) {
    return;
  }
  const message = elements.chatInput.value.trim();
  if (!message) {
    return;
  }
  elements.chatInput.value = "";
  try {
    setBusy("running");
    const payload = await request(`/api/runs/${state.activeRunId}/turns`, {
      method: "POST",
      body: JSON.stringify({ message }),
    });
    renderChat((await request(`/api/runs/${state.activeRunId}/turns`)).turns);
    elements.traceView.textContent = pretty(payload.latest_trace);
    await loadRuns();
    await loadGraph();
    await loadCorpus();
  } catch (error) {
    alert(error.message || error);
  } finally {
    clearBusy();
  }
});

elements.refreshRuns.addEventListener("click", () => loadRuns().catch(handleError));
elements.refreshTrace.addEventListener("click", () => loadTrace().catch(handleError));
elements.refreshGraph.addEventListener("click", () => loadGraph().catch(handleError));
elements.refreshCorpus.addEventListener("click", () => loadCorpus().catch(handleError));
elements.refreshMemoryArena.addEventListener("click", () => loadMemoryArena().catch(handleError));

function handleError(error) {
  alert(error.message || error);
}

function escapeHtml(text) {
  return String(text ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;");
}

await loadRuns().catch(handleError);
await loadMemoryArena().catch(handleError);
