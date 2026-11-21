const state = {
  sessionId: localStorage.getItem("kb-session-id") || crypto.randomUUID().replaceAll("-", ""),
  activeTaskId: null,
  selectedFile: null,
  pollTimer: null,
};

const queryPipeline = [
  ["node_item_name_confirm", "产品识别"], ["node_multi_search", "多路检索"],
  ["node_search_embedding", "向量检索"], ["node_search_embedding_hyde", "HyDE 检索"],
  ["node_web_search_mcp", "网络搜索"], ["node_join", "检索合并"],
  ["node_rrf", "RRF 融合"], ["node_rerank", "GPT 精排"],
  ["node_answer_output", "答案生成"],
];

const importPipeline = [
  ["node_entry", "文件检查"], ["node_pdf_to_md", "PDF 解析"],
  ["node_md_img", "图片处理"], ["node_document_split", "文档切分"],
  ["node_item_name_recognition", "主体识别"], ["node_bge_embedding", "向量生成"],
  ["node_import_milvus", "写入 Milvus"],
];

const $ = (selector) => document.querySelector(selector);
const elements = {
  body: document.body, questionForm: $("#questionForm"), questionInput: $("#questionInput"),
  sendButton: $("#sendButton"), conversation: $("#conversation"), hero: $("#chatHero"),
  pipeline: $("#pipeline"), traceState: $("#traceState"), duration: $("#durationMetric"),
  sourceMetric: $("#sourceMetric"), sessionLabel: $("#sessionLabel"), systemBadge: $("#systemBadge"),
  fileInput: $("#fileInput"), selectedFile: $("#selectedFile"), uploadForm: $("#uploadForm"),
  startImport: $("#startImportButton"), importPipeline: $("#importPipeline"), importState: $("#importState"),
  importResult: $("#importResult"), toastRegion: $("#toastRegion"),
};

function escapeHtml(value = "") {
  return String(value).replace(/[&<>'"]/g, (char) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", "'": "&#39;", '"': "&quot;" }[char]));
}

function formatAnswer(text) {
  const safe = escapeHtml(text).replace(/\[资料(\d+)\]/g, '<button class="citation" data-citation="$1">资料$1</button>');
  const lines = safe.split(/\r?\n/);
  let html = ""; let listType = null;
  for (const line of lines) {
    const numbered = line.match(/^\s*\d+[.、]\s*(.+)$/);
    const bullet = line.match(/^\s*[-*]\s+(.+)$/);
    const nextType = numbered ? "ol" : bullet ? "ul" : null;
    if (listType && nextType !== listType) { html += `</${listType}>`; listType = null; }
    if (nextType) {
      if (!listType) { listType = nextType; html += `<${listType}>`; }
      html += `<li>${numbered?.[1] || bullet?.[1]}</li>`;
    } else if (line.trim()) {
      html += `<p>${line}</p>`;
    }
  }
  if (listType) html += `</${listType}>`;
  return html;
}

function renderPipeline(container, steps, completed = [], current = null, compact = false) {
  container.innerHTML = steps.map(([key, label], index) => {
    const status = completed.includes(key) ? "done" : current === key ? "running" : "";
    if (compact) return `<div class="import-step ${status}"><i></i><strong>${label}</strong><small>${status === "done" ? "完成" : status === "running" ? "处理中" : "等待"}</small></div>`;
    return `<div class="pipeline-step ${status}"><span class="step-dot">${status === "done" ? "✓" : String(index + 1).padStart(2, "0")}</span><strong>${label}</strong><small>${status === "done" ? "完成" : status === "running" ? "运行中" : "等待"}</small></div>`;
  }).join("");
}

function showToast(message, type = "info") {
  const toast = document.createElement("div");
  toast.className = `toast ${type}`; toast.textContent = message;
  elements.toastRegion.append(toast);
  setTimeout(() => toast.remove(), 4200);
}

function createMessage(role, content, options = {}) {
  elements.hero.hidden = true;
  const wrapper = document.createElement("article");
  wrapper.className = `message ${role}`;
  wrapper.innerHTML = `
    <span class="message-avatar">${role === "user" ? "YOU" : "AI"}</span>
    <div class="message-body">
      <p class="message-label">${role === "user" ? "你" : "Knowledge Assistant"}</p>
      <div class="bubble">${options.loading ? '<div class="thinking"><span></span><span></span><span></span><em>正在理解问题并检索资料…</em></div>' : role === "assistant" ? formatAnswer(content) : escapeHtml(content)}</div>
    </div>`;
  elements.conversation.append(wrapper);
  wrapper.scrollIntoView({ behavior: "smooth", block: "end" });
  return wrapper;
}

function renderSources(messageElement, sources = [], images = []) {
  if (!sources.length && !images.length) return;
  const body = messageElement.querySelector(".message-body");
  if (sources.length) {
    const list = document.createElement("div"); list.className = "source-list";
    list.innerHTML = sources.map((source, index) => {
      const url = /^https?:\/\//.test(source.url || "") ? source.url : "";
      const tag = url ? "a" : "div";
      const attrs = url ? `href="${escapeHtml(url)}" target="_blank" rel="noopener noreferrer"` : "";
      return `<${tag} class="source-card" data-source="${index + 1}" ${attrs}><span class="source-number">${index + 1}</span><span><strong>${escapeHtml(source.title || `参考资料 ${index + 1}`)}</strong><small>${escapeHtml((source.content || source.snippet || "").slice(0, 120))}</small></span><span class="source-score">${source.score != null ? `${Math.round(source.score * 100)}%` : source.source === "web" ? "WEB" : "KB"}</span></${tag}>`;
    }).join("");
    body.append(list);
  }
  if (images.length) {
    const grid = document.createElement("div"); grid.className = "image-grid";
    grid.innerHTML = images.map((url) => `<a href="${escapeHtml(url)}" target="_blank" rel="noopener noreferrer"><img src="${escapeHtml(url)}" alt="答案相关资料图片" loading="lazy"></a>`).join("");
    body.append(grid);
  }
}

async function api(path, options = {}) {
  const response = await fetch(path, options);
  let data = null;
  try { data = await response.json(); } catch { data = {}; }
  if (!response.ok) throw new Error(data.detail || "请求失败");
  return data;
}

async function pollTask(taskId, kind, loadingMessage = null) {
  const task = await api(`/api/tasks/${taskId}`);
  if (kind === "query") {
    renderPipeline(elements.pipeline, queryPipeline, task.completed_nodes, task.current_node);
    elements.traceState.textContent = task.status === "processing" ? "执行中" : task.status === "completed" ? "已完成" : task.status === "failed" ? "失败" : "排队中";
    elements.duration.textContent = task.duration_ms < 1000 ? `${task.duration_ms} ms` : `${(task.duration_ms / 1000).toFixed(1)} s`;
    elements.sourceMetric.textContent = task.sources?.length ? `${task.sources.length} 条` : "—";
  } else {
    renderPipeline(elements.importPipeline, importPipeline, task.completed_nodes, task.current_node, true);
    elements.importState.textContent = task.status === "processing" ? "处理中" : task.status === "completed" ? "构建完成" : task.status === "failed" ? "处理失败" : "排队中";
  }

  if (["pending", "processing"].includes(task.status)) {
    state.pollTimer = setTimeout(() => pollTask(taskId, kind, loadingMessage).catch(handlePollError), 900);
    return;
  }

  if (task.status === "failed") {
    if (loadingMessage) loadingMessage.remove();
    throw new Error(task.error || "任务执行失败");
  }

  if (kind === "query") {
    loadingMessage.remove();
    const answerMessage = createMessage("assistant", task.answer);
    renderSources(answerMessage, task.sources, task.image_urls);
    elements.sendButton.disabled = false;
    state.activeTaskId = null;
  } else {
    elements.importResult.hidden = false;
    const result = task.result || {};
    elements.importResult.innerHTML = `<strong>知识构建完成</strong><br>文件：${escapeHtml(result.file_title || "—")}<br>识别主体：${escapeHtml(result.item_name || "—")}<br>切片数量：${result.chunk_count ?? 0}`;
    elements.startImport.disabled = false;
  }
}

function handlePollError(error) {
  elements.sendButton.disabled = false; elements.startImport.disabled = false;
  showToast(error.message, "error");
}

async function submitQuestion(question) {
  if (!question || state.activeTaskId) return;
  createMessage("user", question);
  const loading = createMessage("assistant", "", { loading: true });
  elements.questionInput.value = ""; resizeTextarea(); elements.sendButton.disabled = true;
  elements.traceState.textContent = "启动中"; elements.sourceMetric.textContent = "—"; elements.duration.textContent = "—";
  renderPipeline(elements.pipeline, queryPipeline);
  try {
    const result = await api("/api/query", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question, session_id: state.sessionId }),
    });
    state.sessionId = result.session_id; state.activeTaskId = result.task_id;
    localStorage.setItem("kb-session-id", state.sessionId);
    elements.sessionLabel.textContent = state.sessionId.slice(0, 12);
    await pollTask(result.task_id, "query", loading);
  } catch (error) {
    loading.remove(); elements.sendButton.disabled = false; state.activeTaskId = null;
    createMessage("assistant", `请求未完成：${error.message}`); showToast(error.message, "error");
  }
}

function resizeTextarea() {
  elements.questionInput.style.height = "auto";
  elements.questionInput.style.height = `${Math.min(elements.questionInput.scrollHeight, 150)}px`;
}

function switchView(name) {
  document.querySelectorAll(".view").forEach((view) => view.classList.toggle("active", view.id === `${name}View`));
  document.querySelectorAll(".nav-item").forEach((button) => button.classList.toggle("active", button.dataset.view === name));
  $("#viewTitle").textContent = name === "chat" ? "智能问答" : "知识入库";
  document.body.classList.remove("menu-open");
}

function selectFile(file) {
  if (!file) return;
  const suffix = file.name.split(".").pop()?.toLowerCase();
  if (!["pdf", "md"].includes(suffix)) return showToast("仅支持 PDF 和 Markdown 文件", "error");
  if (file.size > 50 * 1024 * 1024) return showToast("文件不能超过 50 MB", "error");
  state.selectedFile = file; elements.selectedFile.hidden = false; elements.startImport.disabled = false;
  $("#selectedFileName").textContent = file.name;
  $("#selectedFileSize").textContent = `${(file.size / 1024 / 1024).toFixed(2)} MB`;
}

async function loadHistory() {
  try {
    const data = await api(`/api/history/${state.sessionId}`);
    if (!data.messages?.length) return;
    elements.hero.hidden = true;
    data.messages.forEach((message) => {
      const element = createMessage(message.role === "assistant" ? "assistant" : "user", message.text || "");
      if (message.role === "assistant") renderSources(element, [], message.image_urls || []);
    });
  } catch { /* 新会话或 Mongo 暂不可用时保留空界面。 */ }
}

elements.questionForm.addEventListener("submit", (event) => { event.preventDefault(); submitQuestion(elements.questionInput.value.trim()); });
elements.questionInput.addEventListener("input", resizeTextarea);
elements.questionInput.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey) { event.preventDefault(); elements.questionForm.requestSubmit(); }
});
document.querySelectorAll("[data-question]").forEach((button) => button.addEventListener("click", () => submitQuestion(button.dataset.question)));
document.querySelectorAll(".nav-item").forEach((button) => button.addEventListener("click", () => switchView(button.dataset.view)));
$("#uploadShortcut").addEventListener("click", () => switchView("library"));
$("#mobileMenuButton").addEventListener("click", () => document.body.classList.toggle("menu-open"));
$("#newChatButton").addEventListener("click", () => {
  state.sessionId = crypto.randomUUID().replaceAll("-", ""); localStorage.setItem("kb-session-id", state.sessionId);
  elements.sessionLabel.textContent = "新会话"; elements.conversation.innerHTML = ""; elements.hero.hidden = false;
  renderPipeline(elements.pipeline, queryPipeline); switchView("chat");
});
$("#clearHistoryButton").addEventListener("click", async () => {
  try { await api(`/api/history/${state.sessionId}`, { method: "DELETE" }); elements.conversation.innerHTML = ""; elements.hero.hidden = false; showToast("当前会话已清空"); }
  catch (error) { showToast(error.message, "error"); }
});
elements.conversation.addEventListener("click", (event) => {
  const citation = event.target.closest("[data-citation]"); if (!citation) return;
  const source = document.querySelector(`[data-source="${citation.dataset.citation}"]`);
  source?.classList.add("highlight"); source?.scrollIntoView({ behavior: "smooth", block: "center" });
  setTimeout(() => source?.classList.remove("highlight"), 1800);
});

elements.fileInput.addEventListener("change", () => selectFile(elements.fileInput.files[0]));
const dropZone = $("#dropZone");
["dragenter", "dragover"].forEach((name) => dropZone.addEventListener(name, (event) => { event.preventDefault(); dropZone.classList.add("dragover"); }));
["dragleave", "drop"].forEach((name) => dropZone.addEventListener(name, (event) => { event.preventDefault(); dropZone.classList.remove("dragover"); }));
dropZone.addEventListener("drop", (event) => selectFile(event.dataTransfer.files[0]));
$("#removeFileButton").addEventListener("click", () => { state.selectedFile = null; elements.fileInput.value = ""; elements.selectedFile.hidden = true; elements.startImport.disabled = true; });
elements.uploadForm.addEventListener("submit", async (event) => {
  event.preventDefault(); if (!state.selectedFile) return;
  const formData = new FormData(); formData.append("file", state.selectedFile);
  elements.startImport.disabled = true; elements.importResult.hidden = true; elements.importState.textContent = "上传中";
  try { const result = await api("/api/import", { method: "POST", body: formData }); await pollTask(result.task_id, "import"); }
  catch (error) { elements.startImport.disabled = false; showToast(error.message, "error"); }
});

async function init() {
  localStorage.setItem("kb-session-id", state.sessionId);
  elements.sessionLabel.textContent = state.sessionId.slice(0, 12);
  renderPipeline(elements.pipeline, queryPipeline); renderPipeline(elements.importPipeline, importPipeline, [], null, true);
  try { await api("/api/health"); elements.systemBadge.textContent = "运行中"; }
  catch { elements.systemBadge.textContent = "离线"; elements.systemBadge.style.color = "#ff8da3"; }
  await loadHistory();
}

init();
