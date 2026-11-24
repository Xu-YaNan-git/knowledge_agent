const state = {
  sessionId: localStorage.getItem("kb-session-id") || crypto.randomUUID().replaceAll("-", ""),
  activeTaskId: null,
  selectedFile: null,
  pollTimer: null,
  activeEventSource: null,
  knowledgeBases: [],
  selectedKnowledgeBaseId: "",
  authEnabled: false,
  currentUser: null,
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
  knowledgeBaseSelect: $("#knowledgeBaseSelect"), queryKnowledgeBase: $("#queryKnowledgeBase"),
  documentList: $("#documentList"), documentCount: $("#documentCount"),
  evaluationState: $("#evaluationState"), evaluationResult: $("#evaluationResult"),
  evaluationSection: $("#evaluationSection"), evaluationHistory: $("#evaluationHistory"),
  stopButton: $("#stopButton"), sessionList: $("#sessionList"),
  operationsMetrics: $("#operationsMetrics"), taskList: $("#taskList"),
  authOverlay: $("#authOverlay"), loginError: $("#loginError"),
  userManagementSection: $("#userManagementSection"), userList: $("#userList"),
  auditSection: $("#auditSection"), auditList: $("#auditList"), auditTotal: $("#auditTotal"),
  queryQuotaStatus: $("#queryQuotaStatus"),
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
      const content = source.content || source.snippet || "";
      const link = url ? `<a href="${escapeHtml(url)}" target="_blank" rel="noopener noreferrer">打开原始来源 ↗</a>` : "";
      return `<details class="source-card source-detail" data-source="${index + 1}"><summary><span class="source-number">${index + 1}</span><span><strong>${escapeHtml(source.title || source.file_title || `参考资料 ${index + 1}`)}</strong><small>${escapeHtml(content.slice(0, 120))}</small></span><span class="source-score">${source.score != null ? `${Math.round(source.score * 100)}%` : source.source === "web" ? "WEB" : "KB"}</span></summary><div class="source-full-content">${escapeHtml(content)}${link}</div></details>`;
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
  if (response.status === 401 && state.authEnabled) {
    elements.authOverlay.hidden = false;
    elements.loginError.textContent = data.detail || "登录已失效，请重新登录";
  }
  if (!response.ok) throw new Error(data.detail || "请求失败");
  return data;
}

function applyUserState(user) {
  state.currentUser = user;
  const roleNames = { admin: "管理员", editor: "编辑者", viewer: "只读用户" };
  $("#currentUsername").textContent = user?.username || "本地工作区";
  $("#currentUserRole").textContent = user ? roleNames[user.role] || user.role : "Python 3.11 · FastAPI";
  $("#logoutButton").hidden = !state.authEnabled || !user;
  document.body.dataset.role = user?.role || (state.authEnabled ? "anonymous" : "admin");
  $("#operationsSection").hidden = state.authEnabled && user?.role === "viewer";
  elements.userManagementSection.hidden = !state.authEnabled || user?.role !== "admin";
  elements.auditSection.hidden = state.authEnabled && user?.role !== "admin";
  elements.evaluationSection.hidden = state.authEnabled && user?.role === "viewer";
}

async function initializeAuth() {
  const config = await api("/api/auth/config");
  state.authEnabled = Boolean(config.enabled);
  if (!state.authEnabled) { applyUserState(null); return true; }
  try {
    const result = await api("/api/auth/me");
    applyUserState(result.user); elements.authOverlay.hidden = true; return true;
  } catch {
    elements.authOverlay.hidden = false; return false;
  }
}

function updateTaskView(task, kind) {
  if (kind === "query") {
    renderPipeline(elements.pipeline, queryPipeline, task.completed_nodes, task.current_node);
    elements.traceState.textContent = task.status === "processing" ? "执行中" : task.status === "completed" ? "已完成" : task.status === "failed" ? "失败" : task.status === "cancelled" ? "已取消" : "排队中";
    elements.duration.textContent = task.duration_ms < 1000 ? `${task.duration_ms} ms` : `${(task.duration_ms / 1000).toFixed(1)} s`;
    elements.sourceMetric.textContent = task.sources?.length ? `${task.sources.length} 条` : "—";
  } else if (kind === "import") {
    renderPipeline(elements.importPipeline, importPipeline, task.completed_nodes, task.current_node, true);
    elements.importState.textContent = task.status === "processing" ? "处理中" : task.status === "completed" ? "构建完成" : task.status === "failed" ? "处理失败" : "排队中";
  }
}

async function finishTaskView(task, kind, loadingMessage = null) {
  updateTaskView(task, kind);
  if (task.status === "failed") {
    if (loadingMessage) loadingMessage.remove();
    throw new Error(task.error || "任务执行失败");
  }
  if (task.status === "cancelled") {
    if (loadingMessage) loadingMessage.remove();
    elements.sendButton.disabled = false; elements.stopButton.hidden = true; state.activeTaskId = null;
    if (kind === "query") createMessage("assistant", "本次生成已停止。");
    await loadOperations().catch(() => {});
    return;
  }

  if (kind === "query") {
    loadingMessage?.remove();
    const answerMessage = createMessage("assistant", task.answer);
    renderSources(answerMessage, task.sources, task.image_urls);
    elements.sendButton.disabled = false;
    elements.stopButton.hidden = true;
    state.activeTaskId = null;
    await loadSessions();
  } else if (kind === "import") {
    elements.importResult.hidden = false;
    const result = task.result || {};
    elements.importResult.innerHTML = `<strong>知识构建完成</strong><br>文件：${escapeHtml(result.file_title || "—")}<br>识别主体：${escapeHtml(result.item_name || "—")}<br>切片数量：${result.chunk_count ?? 0}`;
    elements.startImport.disabled = false;
    await loadDocuments();
  } else if (kind === "evaluation") {
    elements.evaluationState.textContent = "评测完成";
    elements.evaluationResult.hidden = false;
    const result = task.result || {};
    const gate = result.comparison ? `<br>回归门禁：<b>${result.comparison.passed ? "通过" : `发现 ${result.comparison.regressions.length} 项退化`}</b>` : "<br>回归门禁：尚未设置同数据集基线";
    elements.evaluationResult.innerHTML = `<strong>评测完成</strong><br>用例：${result.case_count ?? 0}<br>通过率：${Math.round((result.pass_rate || 0) * 100)}%<br>关键词召回：${Math.round((result.keyword_recall || 0) * 100)}%<br>平均耗时：${result.average_latency_ms ?? 0} ms${gate}`;
    await loadEvaluationRuns();
  } else if (kind === "export") {
    const result = task.result || {};
    showToast(`导出完成：${result.archived_file_count || 0} 个归档文件`);
    if (result.download_url) window.location.assign(result.download_url);
  }
  await loadOperations().catch(() => {});
  await loadUsage().catch(() => {});
}

async function pollTask(taskId, kind, loadingMessage = null) {
  const task = await api(`/api/tasks/${taskId}`);
  updateTaskView(task, kind);
  if (["pending", "processing", "retrying"].includes(task.status)) {
    state.pollTimer = setTimeout(() => pollTask(taskId, kind, loadingMessage).catch(handlePollError), 900);
    return;
  }
  await finishTaskView(task, kind, loadingMessage);
}

function streamTask(taskId, kind, loadingMessage = null) {
  return new Promise((resolve, reject) => {
    const source = new EventSource(`/api/tasks/${taskId}/events`);
    let settled = false;
    state.activeEventSource = source;
    const handleEvent = async (event, isFinal) => {
      try {
        const task = JSON.parse(event.data); updateTaskView(task, kind);
        if (isFinal) {
          settled = true; source.close(); state.activeEventSource = null;
          await finishTaskView(task, kind, loadingMessage); resolve(task);
        }
      } catch (error) { settled = true; source.close(); state.activeEventSource = null; reject(error); }
    };
    source.addEventListener("progress", (event) => handleEvent(event, false));
    source.addEventListener("final", (event) => handleEvent(event, true));
    source.onerror = () => {
      if (settled) return;
      settled = true;
      source.close(); state.activeEventSource = null;
      pollTask(taskId, kind, loadingMessage).then(resolve).catch(reject);
    };
  });
}

function handlePollError(error) {
  elements.sendButton.disabled = false; elements.startImport.disabled = false;
  elements.stopButton.hidden = true; state.activeTaskId = null;
  state.activeEventSource?.close(); state.activeEventSource = null;
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
      body: JSON.stringify({
        question,
        session_id: state.sessionId,
        knowledge_base_ids: elements.queryKnowledgeBase.value ? [elements.queryKnowledgeBase.value] : [],
      }),
    });
    state.sessionId = result.session_id; state.activeTaskId = result.task_id;
    localStorage.setItem("kb-session-id", state.sessionId);
    elements.sessionLabel.textContent = state.sessionId.slice(0, 12);
    elements.stopButton.hidden = false;
    await streamTask(result.task_id, "query", loading);
  } catch (error) {
    loading.remove(); elements.sendButton.disabled = false; elements.stopButton.hidden = true; state.activeTaskId = null;
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

async function loadHistory(clearExisting = false) {
  try {
    const data = await api(`/api/history/${state.sessionId}`);
    if (clearExisting) { elements.conversation.innerHTML = ""; elements.hero.hidden = false; }
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
  if (state.activeTaskId) return showToast("请先停止当前任务", "error");
  state.activeEventSource?.close(); state.activeEventSource = null;
  state.sessionId = crypto.randomUUID().replaceAll("-", ""); localStorage.setItem("kb-session-id", state.sessionId);
  elements.sessionLabel.textContent = "新会话"; elements.conversation.innerHTML = ""; elements.hero.hidden = false;
  renderPipeline(elements.pipeline, queryPipeline); switchView("chat");
});
elements.stopButton.addEventListener("click", async () => {
  if (!state.activeTaskId) return;
  elements.stopButton.disabled = true;
  try { await api(`/api/tasks/${state.activeTaskId}/cancel`, { method: "POST" }); showToast("正在安全停止任务"); }
  catch (error) { showToast(error.message, "error"); }
  finally { elements.stopButton.disabled = false; }
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

function renderSessions(items) {
  elements.sessionList.innerHTML = items.map((session) => `
    <div class="session-list-item ${session.session_id === state.sessionId ? "active" : ""}" data-session-id="${escapeHtml(session.session_id)}">
      <button type="button" data-open-session="${escapeHtml(session.session_id)}"><strong>${escapeHtml(session.title || "未命名会话")}</strong><small>${escapeHtml((session.updated_at || "").slice(0, 16).replace("T", " "))}</small></button>
      <span><button type="button" title="重命名" data-rename-session="${escapeHtml(session.session_id)}">✎</button><button type="button" title="删除" data-delete-session="${escapeHtml(session.session_id)}">×</button></span>
    </div>`).join("") || "<small>暂无会话</small>";
}

async function loadSessions() {
  const data = await api("/api/sessions"); renderSessions(data.items || []);
}

elements.sessionList.addEventListener("click", async (event) => {
  const openButton = event.target.closest("[data-open-session]");
  const renameButton = event.target.closest("[data-rename-session]");
  const deleteButton = event.target.closest("[data-delete-session]");
  try {
    if (openButton) {
      if (state.activeTaskId) return showToast("请先停止当前任务", "error");
      state.sessionId = openButton.dataset.openSession; localStorage.setItem("kb-session-id", state.sessionId);
      elements.sessionLabel.textContent = state.sessionId.slice(0, 12); await loadHistory(true); await loadSessions(); switchView("chat");
    } else if (renameButton) {
      const title = prompt("输入新的会话标题"); if (!title?.trim()) return;
      await api(`/api/sessions/${renameButton.dataset.renameSession}`, { method: "PATCH", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ title: title.trim() }) });
      await loadSessions();
    } else if (deleteButton) {
      if (!confirm("确认删除该会话及全部消息？")) return;
      const deletedId = deleteButton.dataset.deleteSession;
      if (state.activeTaskId && deletedId === state.sessionId) return showToast("请先停止当前任务", "error");
      await api(`/api/sessions/${deletedId}`, { method: "DELETE" });
      if (deletedId === state.sessionId) $("#newChatButton").click();
      await loadSessions();
    }
  } catch (error) { showToast(error.message, "error"); }
});
$("#refreshSessionsButton").addEventListener("click", () => loadSessions().catch((error) => showToast(error.message, "error")));

elements.fileInput.addEventListener("change", () => selectFile(elements.fileInput.files[0]));
const dropZone = $("#dropZone");
["dragenter", "dragover"].forEach((name) => dropZone.addEventListener(name, (event) => { event.preventDefault(); dropZone.classList.add("dragover"); }));
["dragleave", "drop"].forEach((name) => dropZone.addEventListener(name, (event) => { event.preventDefault(); dropZone.classList.remove("dragover"); }));
dropZone.addEventListener("drop", (event) => selectFile(event.dataTransfer.files[0]));
$("#removeFileButton").addEventListener("click", () => { state.selectedFile = null; elements.fileInput.value = ""; elements.selectedFile.hidden = true; elements.startImport.disabled = true; });
elements.uploadForm.addEventListener("submit", async (event) => {
  event.preventDefault(); if (!state.selectedFile) return;
  if (!state.selectedKnowledgeBaseId) return showToast("请先创建并选择知识库", "error");
  const formData = new FormData(); formData.append("file", state.selectedFile);
  elements.startImport.disabled = true; elements.importResult.hidden = true; elements.importState.textContent = "上传中";
  try {
    const result = await api(`/api/knowledge-bases/${state.selectedKnowledgeBaseId}/documents`, { method: "POST", body: formData });
    if (result.duplicate) { elements.startImport.disabled = false; showToast("相同内容已存在，无需重复导入"); return; }
    await streamTask(result.task_id, "import");
  }
  catch (error) { elements.startImport.disabled = false; showToast(error.message, "error"); }
});

function renderKnowledgeBases() {
  const options = state.knowledgeBases.map((item) => `<option value="${escapeHtml(item.knowledge_base_id)}">${escapeHtml(item.name)}（${item.document_count || 0}）</option>`).join("");
  elements.knowledgeBaseSelect.innerHTML = options || '<option value="">暂无知识库</option>';
  elements.queryKnowledgeBase.innerHTML = '<option value="">全部知识库</option>' + options;
  if (!state.selectedKnowledgeBaseId || !state.knowledgeBases.some((item) => item.knowledge_base_id === state.selectedKnowledgeBaseId)) {
    state.selectedKnowledgeBaseId = state.knowledgeBases[0]?.knowledge_base_id || "";
  }
  elements.knowledgeBaseSelect.value = state.selectedKnowledgeBaseId;
}

async function loadKnowledgeBases() {
  const data = await api("/api/knowledge-bases");
  state.knowledgeBases = data.items || [];
  renderKnowledgeBases();
  await loadDocuments();
}

async function loadDocuments() {
  if (!state.selectedKnowledgeBaseId) {
    elements.documentList.innerHTML = '<p class="empty-hint">请选择或创建知识库。</p>';
    elements.documentCount.textContent = "0 个文档";
    return;
  }
  const data = await api(`/api/knowledge-bases/${state.selectedKnowledgeBaseId}/documents`);
  elements.documentCount.textContent = `${data.total || 0} 个文档`;
  elements.documentList.innerHTML = (data.items || []).map((doc) => `
    <article class="document-row">
      <div><strong>${escapeHtml(doc.filename)}</strong><small>v${doc.version} · ${doc.chunk_count || 0} 个切片 · ${escapeHtml(doc.item_name || "未识别主体")}</small></div>
      <span class="document-status ${escapeHtml(doc.status)}">${escapeHtml(doc.status)}</span>
      <span class="document-actions"><button type="button" data-view-versions="${escapeHtml(doc.document_id)}">版本</button>${doc.source_archived ? `<a href="/api/documents/${escapeHtml(doc.document_id)}/download">下载原文</a>` : ""}${doc.status === "failed" ? `<button type="button" data-retry-document="${escapeHtml(doc.document_id)}">重试</button>` : doc.is_current ? `<button type="button" data-delete-document="${escapeHtml(doc.document_id)}">删除</button>` : ""}</span>
      <div class="document-versions" data-versions-for="${escapeHtml(doc.document_id)}" hidden></div>
    </article>`).join("") || '<p class="empty-hint">该知识库还没有文档。</p>';
}

async function loadOperations() {
  const [summary, tasks] = await Promise.all([api("/api/operations/summary"), api("/api/tasks?limit=12")]);
  const taskCounts = summary.task_status_counts || {};
  elements.operationsMetrics.innerHTML = `
    <div><span>知识库</span><strong>${summary.knowledge_base_count || 0}</strong></div>
    <div><span>会话</span><strong>${summary.session_count || 0}</strong></div>
    <div><span>运行中</span><strong>${taskCounts.processing || 0}</strong></div>
    <div><span>失败</span><strong>${taskCounts.failed || 0}</strong></div>
    <div><span>评测</span><strong>${summary.evaluation_count || 0}</strong></div>`;
  const typeNames = { query: "问答", import: "入库", evaluation: "评测", export: "导出" };
  elements.taskList.innerHTML = (tasks.items || []).map((task) => `
    <div class="task-row"><span>${typeNames[task.type] || escapeHtml(task.type)}</span><strong>${escapeHtml(task.status)}</strong><small>${escapeHtml((task.created_at || "").slice(0, 19).replace("T", " "))}</small><em>${task.duration_ms < 1000 ? `${task.duration_ms}ms` : `${(task.duration_ms / 1000).toFixed(1)}s`}</em></div>`).join("") || '<p class="empty-hint">暂无任务记录。</p>';
}

async function loadUsage() {
  const data = await api("/api/usage");
  const queryUsage = (data.items || []).find((item) => item.action === "query");
  elements.queryQuotaStatus.textContent = !queryUsage?.enabled
    ? "不限制"
    : `${queryUsage.remaining}/${queryUsage.limit}`;
}

async function loadUsers() {
  if (!state.authEnabled || state.currentUser?.role !== "admin") return;
  const data = await api("/api/users");
  elements.userList.innerHTML = (data.items || []).map((user) => `
    <div class="user-row" data-user-id="${escapeHtml(user.user_id)}">
      <strong>${escapeHtml(user.username)}</strong>
      <select data-user-role><option value="viewer" ${user.role === "viewer" ? "selected" : ""}>只读用户</option><option value="editor" ${user.role === "editor" ? "selected" : ""}>编辑者</option><option value="admin" ${user.role === "admin" ? "selected" : ""}>管理员</option></select>
      <label><input data-user-active type="checkbox" ${user.is_active ? "checked" : ""}> 启用</label>
      <button type="button" data-save-user>保存</button>
    </div>`).join("") || '<p class="empty-hint">暂无账号。</p>';
}

async function loadAuditLogs() {
  if (state.authEnabled && state.currentUser?.role !== "admin") return;
  const params = new URLSearchParams({ limit: "50" });
  const action = $("#auditActionFilter").value;
  const outcome = $("#auditOutcomeFilter").value;
  if (action) params.set("action", action);
  if (outcome) params.set("outcome", outcome);
  const data = await api(`/api/audit-logs?${params}`);
  const actionNames = {
    "auth.login": "账号登录", "auth.logout": "账号退出", "user.create": "创建账号", "user.update": "更新账号",
    "knowledge_base.create": "创建知识库", "knowledge_base.update": "更新知识库", "knowledge_base.delete": "删除知识库",
    "document.import": "文档入库", "document.delete": "删除文档", "document.retry": "重试文档",
    "query.create": "发起问答", "evaluation.create": "运行评测", "evaluation.baseline": "更新评测基线", "task.cancel": "取消任务", "task.retry": "重试任务",
    "session.rename": "重命名会话", "session.delete": "删除会话", "history.delete": "清除历史",
  };
  const outcomeNames = { success: "成功", failure: "失败", denied: "拒绝" };
  elements.auditTotal.textContent = `${data.total || 0} 条记录`;
  elements.auditList.innerHTML = (data.items || []).map((item) => `
    <div class="audit-row">
      <strong>${escapeHtml(actionNames[item.action] || item.action)}</strong>
      <span class="audit-outcome ${escapeHtml(item.outcome)}">${escapeHtml(outcomeNames[item.outcome] || item.outcome)}</span>
      <span>${escapeHtml(item.actor_username || "匿名/本地模式")}<small>${escapeHtml(item.actor_role || "—")}</small></span>
      <span>${escapeHtml(item.target_id || item.target_type || "—")}<small>${escapeHtml(item.path || "")}</small></span>
      <time>${escapeHtml((item.created_at || "").slice(0, 19).replace("T", " "))}</time>
    </div>`).join("") || '<p class="empty-hint">暂无审计记录。</p>';
}

async function loadEvaluationRuns() {
  if (state.authEnabled && state.currentUser?.role === "viewer") return;
  const data = await api("/api/evaluations?limit=20");
  elements.evaluationHistory.innerHTML = (data.items || []).map((run) => {
    const summary = run.summary || {};
    const gate = run.is_baseline ? "基线" : run.comparison ? (run.comparison.passed ? "门禁通过" : "发生退化") : "未比较";
    return `<div class="evaluation-row" data-evaluation-id="${escapeHtml(run.evaluation_id)}">
      <div><strong>${escapeHtml(run.dataset_name || "评测数据集")}</strong><small>${escapeHtml((run.created_at || "").slice(0, 19).replace("T", " "))}</small></div>
      <span>通过率 <b>${Math.round((summary.pass_rate || 0) * 100)}%</b></span>
      <span>召回 <b>${Math.round((summary.keyword_recall || 0) * 100)}%</b></span>
      <em class="${run.comparison && !run.comparison.passed ? "regressed" : ""}">${gate}</em>
      <div class="evaluation-actions"><button type="button" data-set-baseline>设为基线</button>${run.is_baseline ? "" : '<button type="button" data-compare-evaluation>对比</button>'}</div>
    </div>`;
  }).join("") || '<p class="empty-hint">暂无评测记录。</p>';
}

$("#knowledgeBaseForm").addEventListener("submit", async (event) => {
  event.preventDefault();
  try {
    const created = await api("/api/knowledge-bases", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ name: $("#knowledgeBaseName").value.trim(), description: $("#knowledgeBaseDescription").value.trim() }),
    });
    state.selectedKnowledgeBaseId = created.knowledge_base_id;
    event.target.reset(); await loadKnowledgeBases(); showToast("知识库已创建");
  } catch (error) { showToast(error.message, "error"); }
});

elements.knowledgeBaseSelect.addEventListener("change", async () => {
  state.selectedKnowledgeBaseId = elements.knowledgeBaseSelect.value;
  await loadDocuments().catch((error) => showToast(error.message, "error"));
});
$("#refreshDocumentsButton").addEventListener("click", () => loadKnowledgeBases().catch((error) => showToast(error.message, "error")));
$("#exportKnowledgeBaseButton").addEventListener("click", async () => {
  if (!state.selectedKnowledgeBaseId) { showToast("请先选择知识库", "error"); return; }
  try {
    const result = await api(`/api/knowledge-bases/${state.selectedKnowledgeBaseId}/exports`, { method: "POST" });
    showToast("知识库导出任务已创建");
    await streamTask(result.task_id, "export");
  } catch (error) { showToast(error.message, "error"); }
});
$("#refreshOperationsButton").addEventListener("click", () => loadOperations().catch((error) => showToast(error.message, "error")));
$("#refreshUsersButton").addEventListener("click", () => loadUsers().catch((error) => showToast(error.message, "error")));
$("#refreshAuditButton").addEventListener("click", () => loadAuditLogs().catch((error) => showToast(error.message, "error")));
$("#auditActionFilter").addEventListener("change", () => loadAuditLogs().catch((error) => showToast(error.message, "error")));
$("#auditOutcomeFilter").addEventListener("change", () => loadAuditLogs().catch((error) => showToast(error.message, "error")));
$("#userForm").addEventListener("submit", async (event) => {
  event.preventDefault();
  try {
    await api("/api/users", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ username: $("#newUsername").value.trim(), password: $("#newUserPassword").value, role: $("#newUserRole").value }) });
    event.target.reset(); await loadUsers(); showToast("账号已创建");
  } catch (error) { showToast(error.message, "error"); }
});
elements.userList.addEventListener("click", async (event) => {
  const button = event.target.closest("[data-save-user]"); if (!button) return;
  const row = button.closest("[data-user-id]");
  try {
    await api(`/api/users/${row.dataset.userId}`, { method: "PATCH", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ role: row.querySelector("[data-user-role]").value, is_active: row.querySelector("[data-user-active]").checked }) });
    await loadUsers(); showToast("账号权限已更新");
  } catch (error) { showToast(error.message, "error"); }
});
elements.evaluationHistory.addEventListener("click", async (event) => {
  const baselineButton = event.target.closest("[data-set-baseline]");
  const compareButton = event.target.closest("[data-compare-evaluation]");
  if (!baselineButton && !compareButton) return;
  const row = event.target.closest("[data-evaluation-id]");
  try {
    if (baselineButton) {
      await api(`/api/evaluations/${row.dataset.evaluationId}/baseline`, { method: "PATCH" });
      await loadEvaluationRuns(); showToast("评测基线已更新");
    } else {
      const comparison = await api(`/api/evaluations/${row.dataset.evaluationId}/comparison`);
      elements.evaluationResult.hidden = false;
      elements.evaluationResult.innerHTML = `<strong>${comparison.passed ? "回归门禁通过" : "检测到效果退化"}</strong><br>退化指标：${comparison.regressions.length}<br>通过率变化：${Math.round((comparison.deltas.pass_rate || 0) * 10000) / 100}%<br>平均耗时变化：${comparison.deltas.average_latency_ms || 0} ms`;
    }
  } catch (error) { showToast(error.message, "error"); }
});
elements.documentList.addEventListener("click", async (event) => {
  const deleteButton = event.target.closest("[data-delete-document]");
  const versionsButton = event.target.closest("[data-view-versions]");
  const retryButton = event.target.closest("[data-retry-document]");
  if (!deleteButton && !versionsButton && !retryButton) return;
  if (deleteButton && !confirm("确认删除该文档及其检索数据？")) return;
  try {
    if (deleteButton) {
      await api(`/api/documents/${deleteButton.dataset.deleteDocument}`, { method: "DELETE" }); await loadKnowledgeBases(); showToast("文档已删除");
    } else if (retryButton) {
      const result = await api(`/api/documents/${retryButton.dataset.retryDocument}/retry`, { method: "POST" });
      showToast("已重新提交文档处理"); await streamTask(result.task_id, "import"); await loadKnowledgeBases();
    } else {
      const documentId = versionsButton.dataset.viewVersions;
      const container = elements.documentList.querySelector(`[data-versions-for="${documentId}"]`);
      if (!container.hidden) { container.hidden = true; return; }
      const document = await api(`/api/documents/${documentId}`);
      container.innerHTML = document.versions.map((version) => `<div><b>v${version.version}</b><span>${escapeHtml(version.status)}</span><small>${escapeHtml((version.updated_at || "").slice(0, 19).replace("T", " "))}${version.error ? ` · ${escapeHtml(version.error)}` : ""}</small>${version.source_archived ? `<a href="/api/documents/${escapeHtml(version.document_id)}/download">下载</a>` : ""}</div>`).join("");
      container.hidden = false;
    }
  }
  catch (error) { showToast(error.message, "error"); }
});

$("#evaluationForm").addEventListener("submit", async (event) => {
  event.preventDefault(); const file = $("#evaluationFile").files[0]; if (!file) return;
  const formData = new FormData(); formData.append("file", file);
  if (state.selectedKnowledgeBaseId) formData.append("knowledge_base_ids", state.selectedKnowledgeBaseId);
  elements.evaluationState.textContent = "运行中"; elements.evaluationResult.hidden = true;
  try { const result = await api("/api/evaluations", { method: "POST", body: formData }); await streamTask(result.task_id, "evaluation"); }
  catch (error) { elements.evaluationState.textContent = "失败"; showToast(error.message, "error"); }
});

$("#loginForm").addEventListener("submit", async (event) => {
  event.preventDefault(); elements.loginError.textContent = ""; $("#loginButton").disabled = true;
  try {
    const result = await api("/api/auth/login", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ username: $("#loginUsername").value.trim(), password: $("#loginPassword").value }),
    });
    applyUserState(result.user); elements.authOverlay.hidden = true; event.target.reset(); await loadApplicationData();
  } catch (error) { elements.loginError.textContent = error.message; }
  finally { $("#loginButton").disabled = false; }
});

$("#logoutButton").addEventListener("click", async () => {
  try { await api("/api/auth/logout", { method: "POST" }); }
  finally { applyUserState(null); elements.authOverlay.hidden = false; elements.conversation.innerHTML = ""; }
});

async function loadApplicationData() {
  const health = await api("/api/health");
  elements.systemBadge.textContent = health.status === "ok" ? "运行中" : "部分依赖异常";
  await loadUsage();
  await loadKnowledgeBases();
  await loadSessions();
  if (!state.authEnabled || ["admin", "editor"].includes(state.currentUser?.role)) await loadOperations();
  if (state.currentUser?.role === "admin") await loadUsers();
  if (!state.authEnabled || state.currentUser?.role === "admin") await loadAuditLogs();
  if (!state.authEnabled || ["admin", "editor"].includes(state.currentUser?.role)) await loadEvaluationRuns();
  await loadHistory(true);
}

async function init() {
  localStorage.setItem("kb-session-id", state.sessionId);
  elements.sessionLabel.textContent = state.sessionId.slice(0, 12);
  renderPipeline(elements.pipeline, queryPipeline); renderPipeline(elements.importPipeline, importPipeline, [], null, true);
  try {
    if (await initializeAuth()) await loadApplicationData();
  }
  catch { elements.systemBadge.textContent = "离线"; elements.systemBadge.style.color = "#ff8da3"; }
}

init();
