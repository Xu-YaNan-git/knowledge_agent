const state = {
  sessionId: localStorage.getItem("kb-session-id") || crypto.randomUUID().replaceAll("-", ""),
  activeTaskId: null,
  selectedFile: null,
  pollTimer: null,
  activeEventSource: null,
  knowledgeBases: [],
  selectedKnowledgeBaseId: "",
  documentPage: 1,
  documentPageSize: 20,
  documentTotal: 0,
  evaluationPage: 1,
  evaluationPageSize: 20,
  evaluationTotal: 0,
  evaluationRequestId: 0,
  auditPage: 1,
  auditPageSize: 20,
  auditTotal: 0,
  auditRequestId: 0,
  authEnabled: false,
  currentUser: null,
  // 知识内容状态
  chunkPage: 1,
  chunkPageSize: 20,
  chunkTotal: 0,
  chunkKeyword: "",
  chunkDocumentFilter: "",
  chunkDocuments: [], // 用于下拉过滤
  chunkRequestId: 0,
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
  documentPagination: $("#documentPagination"), documentPageInfo: $("#documentPageInfo"),
  documentPrevPage: $("#documentPrevPage"), documentNextPage: $("#documentNextPage"),
  chunkList: $("#chunkList"),
  chunkCount: $("#chunkCount"),
  chunkKeyword: $("#chunkKeyword"),
  chunkDocumentFilter: $("#chunkDocumentFilter"),
  chunkSearchBtn: $("#chunkSearchBtn"),
  chunkRefreshBtn: $("#chunkRefreshBtn"),
  chunkPagination: $("#chunkPagination"),
  chunkPageInfo: $("#chunkPageInfo"),
  chunkRange: $("#chunkRange"), chunkPageSize: $("#chunkPageSize"),
  chunkPageNumbers: $("#chunkPageNumbers"),
  chunkFirstPage: $("#chunkFirstPage"), chunkLastPage: $("#chunkLastPage"),
  chunkPrevPage: $("#chunkPrevPage"),
  chunkNextPage: $("#chunkNextPage"),
  evaluationState: $("#evaluationState"), evaluationResult: $("#evaluationResult"),
  evaluationSection: $("#evaluationSection"), evaluationHistory: $("#evaluationHistory"),
  evaluationTotal: $("#evaluationTotal"), evaluationPagination: $("#evaluationPagination"),
  evaluationPageInfo: $("#evaluationPageInfo"), evaluationPrevPage: $("#evaluationPrevPage"),
  evaluationNextPage: $("#evaluationNextPage"), evaluationPageSize: $("#evaluationPageSize"),
  stopButton: $("#stopButton"), sessionList: $("#sessionList"),
  operationsMetrics: $("#operationsMetrics"), taskList: $("#taskList"),
  authOverlay: $("#authOverlay"), loginError: $("#loginError"),
  userManagementSection: $("#userManagementSection"), userList: $("#userList"),
  auditSection: $("#auditSection"), auditList: $("#auditList"), auditTotal: $("#auditTotal"),
  auditPagination: $("#auditPagination"), auditPageInfo: $("#auditPageInfo"),
  auditPrevPage: $("#auditPrevPage"), auditNextPage: $("#auditNextPage"), auditPageSize: $("#auditPageSize"),
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
  document.querySelector('[data-view="audit"]').hidden = state.authEnabled && user?.role !== "admin";
  document.querySelector('[data-view="evaluation"]').hidden = state.authEnabled && user?.role === "viewer";
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
    await loadKnowledgeBases();
  } else if (kind === "evaluation") {
    elements.evaluationState.textContent = "评测完成";
    elements.evaluationResult.hidden = false;
    const result = task.result || {};
    const gate = result.comparison ? `<br>回归门禁：<b>${result.comparison.passed ? "通过" : `发现 ${result.comparison.regressions.length} 项退化`}</b>` : "<br>回归门禁：尚未设置同数据集基线";
    elements.evaluationResult.innerHTML = `<strong>评测完成</strong><br>用例：${result.case_count ?? 0}<br>通过率：${Math.round((result.pass_rate || 0) * 100)}%<br>关键词召回：${Math.round((result.keyword_recall || 0) * 100)}%<br>平均耗时：${result.average_latency_ms ?? 0} ms${gate}`;
    await loadEvaluationRuns(true);
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
  const pages = {
    library: ["知识库概览", "KNOWLEDGE WORKSPACE", "管理知识库，查看文档与切片，让每条答案都能回到原始资料。"],
    import: ["资料入库", "INGEST DOCUMENTS", "上传资料并跟踪解析、切片、向量写入的每一步。"],
    documents: ["知识管理", "KNOWLEDGE MANAGEMENT", "集中查看文档版本、知识切片和运行状态。"],
    evaluation: ["RAG 效果评测", "RAG EVALUATION", "运行标准问答集，追踪检索与答案质量。"],
    audit: ["安全审计日志", "SECURITY AUDIT", "按操作、结果与时间检查访问和变更记录。"],
  };
  if (name === "operations") name = "documents";
  if (name === "audit" && state.authEnabled && state.currentUser?.role !== "admin") name = "library";
  if (name === "evaluation" && state.authEnabled && !["admin", "editor"].includes(state.currentUser?.role)) name = "library";
  const activePage = name === "chunks" ? "documents" : name;
  if (activePage !== "chat" && !pages[activePage]) { name = "library"; }
  const pageName = name === "chunks" ? "documents" : name;
  const enteredPage = state.currentView !== name;
  state.currentView = name;
  document.querySelectorAll(".view").forEach((view) => view.classList.toggle("active", view.id === (pageName === "chat" ? "chatView" : "libraryView")));
  document.querySelectorAll("[data-library-page]").forEach((page) => { page.hidden = page.dataset.libraryPage !== pageName; });
  document.querySelectorAll(".nav-item").forEach((button) => button.classList.toggle("active", button.dataset.view === pageName));
  $("#viewTitle").textContent = pageName === "chat" ? "智能问答" : pages[pageName][0];
  if (pageName !== "chat") {
    $("#libraryHeading").textContent = pages[pageName][0];
    $("#libraryEyebrow").textContent = pages[pageName][1];
    $("#libraryDescription").textContent = pages[pageName][2];
  }
  if (location.hash !== `#/${name}`) location.hash = `/${name}`;
  document.body.dataset.view = pageName;
  if (pageName === "documents") {
    const tab = name === "chunks" ? "chunks" : "documents";
    $("#documentsPanel").hidden = tab !== "documents";
    $("#chunksPanel").hidden = tab !== "chunks";
    document.querySelectorAll(".kb-content-tabs [data-tab]").forEach((button) => {
      const selected = button.dataset.tab === tab;
      button.classList.toggle("active", selected);
      button.setAttribute("aria-selected", String(selected));
    });
  }
  if (name === "chunks" && enteredPage) {
    loadChunkDocumentFilter().then(() => loadChunks()).catch((error) => showToast(error.message, "error"));
  }
  if (name === "evaluation" && enteredPage) loadEvaluationRuns().catch((error) => showToast(error.message, "error"));
  if (name === "audit" && enteredPage) loadAuditLogs().catch((error) => showToast(error.message, "error"));
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
$("#uploadShortcut").addEventListener("click", () => switchView("import"));
document.querySelectorAll("[data-go-page]").forEach((button) => button.addEventListener("click", () => switchView(button.dataset.goPage)));
window.addEventListener("hashchange", () => switchView(location.hash.replace(/^#\//, "") || "chat"));
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

function formatSessionTime(value) {
  const date = new Date(value);
  if (!value || Number.isNaN(date.getTime())) return "时间未知";
  const now = new Date();
  const sameDay = now.toDateString() === date.toDateString();
  const time = new Intl.DateTimeFormat("zh-CN", { hour: "2-digit", minute: "2-digit", hour12: false }).format(date);
  if (sameDay) return `今天 ${time}`;
  const day = new Intl.DateTimeFormat("zh-CN", { month: "2-digit", day: "2-digit" }).format(date);
  return date.getFullYear() === now.getFullYear() ? `${day} ${time}` : `${date.getFullYear()}/${day} ${time}`;
}

function renderSessions(items) {
  elements.sessionList.innerHTML = items.map((session) => {
    const title = String(session.title || "未命名会话").trim() || "未命名会话";
    const active = session.session_id === state.sessionId;
    const id = escapeHtml(session.session_id);
    return `<article class="session-list-item ${active ? "active" : ""}" data-session-id="${id}">
      <button class="session-open" type="button" data-open-session="${id}" title="${escapeHtml(title)}" ${active ? 'aria-current="true"' : ""}>
        <strong>${escapeHtml(title)}</strong><time datetime="${escapeHtml(session.updated_at || "")}">${escapeHtml(formatSessionTime(session.updated_at))}</time>
      </button>
      <div class="session-actions">
        <button type="button" title="重命名" aria-label="重命名：${escapeHtml(title)}" data-rename-session="${id}">✎</button>
        <button type="button" title="删除" aria-label="删除：${escapeHtml(title)}" data-delete-session="${id}">×</button>
      </div>
    </article>`;
  }).join("") || '<div class="session-empty"><span>◌</span><strong>暂无会话</strong><small>发起一次提问后会显示在这里</small></div>';
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
  catch (error) {
    elements.startImport.disabled = false;

    if (error.detail && error.detail.message && error.detail.message.includes("同一文档已有版本正在处理中")) {
      const docId = error.detail.document_id;
      elements.importResult.hidden = false;
      elements.importResult.innerHTML = `
        <div class="import-result error">
          <p>检测到该文件有卡住的处理任务。</p>
          <button type="button" id="clearStuckTaskBtn" class="primary-button">清除卡住任务并重试</button>
        </div>
      `;

      $("#clearStuckTaskBtn").addEventListener("click", async () => {
        if (!docId) {
          showToast("无法获取文档ID，请手动清理", "error");
          return;
        }
        try {
          elements.startImport.disabled = true;
          await api(`/api/documents/${docId}/discard-processing`, { method: "POST" });
          showToast("已清除卡住任务，正在重新上传...");
          // 重新发起上传
          const formData = new FormData();
          formData.append("file", state.selectedFile);
          const result = await api(`/api/knowledge-bases/${state.selectedKnowledgeBaseId}/documents`, { method: "POST", body: formData });
          await streamTask(result.task_id, "import");
        } catch (e) {
          elements.startImport.disabled = false;
          showToast(e.message || "清除失败", "error");
        }
      });
      return;
    }

    showToast(error.message, "error");
  }
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
  await loadKnowledgeOverview();
}

async function loadKnowledgeOverview() {
  const target = $("#knowledgeOverviewMetrics");
  const selected = state.knowledgeBases.find((item) => item.knowledge_base_id === state.selectedKnowledgeBaseId);
  if (!selected) {
    target.innerHTML = '<div class="overview-empty">还没有知识库。创建一个知识库后即可导入资料。</div>';
    return;
  }
  try {
    const stats = await api(`/api/knowledge-bases/${state.selectedKnowledgeBaseId}/stats`);
    target.innerHTML = `
      <div class="overview-stat"><span>知识库</span><strong>${escapeHtml(selected.name)}</strong><small>${escapeHtml(selected.description || "当前工作空间")}</small></div>
      <div class="overview-stat"><span>可用文档</span><strong>${Number(stats.total_documents) || 0}</strong><small>已完成入库</small></div>
      <div class="overview-stat"><span>知识切片</span><strong>${Number(stats.total_chunks) || 0}</strong><small>可检索内容</small></div>
      <div class="overview-stat"><span>最近更新</span><strong class="stat-date">${escapeHtml(stats.last_updated ? String(stats.last_updated).slice(0, 10) : "—")}</strong><small>切片写入日期</small></div>`;
  } catch (error) {
    target.innerHTML = `<div class="overview-empty">概览加载失败：${escapeHtml(error.message)}</div>`;
  }
}

async function loadDocuments(resetPage = false) {
  if (resetPage) state.documentPage = 1;
  if (!state.selectedKnowledgeBaseId) {
    elements.documentList.innerHTML = '<p class="empty-hint">请选择或创建知识库。</p>';
    elements.documentCount.textContent = "0 个文档";
    elements.documentPagination.hidden = true;
    return;
  }
  const skip = (state.documentPage - 1) * state.documentPageSize;
  const data = await api(`/api/knowledge-bases/${state.selectedKnowledgeBaseId}/documents?skip=${skip}&limit=${state.documentPageSize}`);
  state.documentTotal = data.total || 0;
  const pageCount = Math.max(1, Math.ceil(state.documentTotal / state.documentPageSize));
  if (state.documentPage > pageCount) return loadDocuments(true);
  elements.documentCount.textContent = `${data.total || 0} 个文档`;
  elements.documentList.innerHTML = (data.items || []).map((doc) => `
    <article class="document-row">
      <div><strong>${escapeHtml(doc.filename)}</strong><small>v${doc.version} · ${doc.chunk_count || 0} 个切片 · ${escapeHtml(doc.item_name || "未识别主体")}</small></div>
      <span class="document-status ${escapeHtml(doc.status)}">${escapeHtml(doc.status)}</span>
      <span class="document-actions">${doc.status === "ready" && doc.is_current ? `<button type="button" data-show-document-chunks="${escapeHtml(doc.document_id)}">查看切片</button>` : ""}<button type="button" data-view-versions="${escapeHtml(doc.document_id)}">版本</button>${doc.source_archived ? `<a href="/api/documents/${escapeHtml(doc.document_id)}/download">下载原文</a>` : ""}${doc.status === "failed" ? `<button type="button" data-retry-document="${escapeHtml(doc.document_id)}">重试</button>` : doc.is_current ? `<button type="button" data-delete-document="${escapeHtml(doc.document_id)}">删除</button>` : ""}</span>
      <div class="document-versions" data-versions-for="${escapeHtml(doc.document_id)}" hidden></div>
    </article>`).join("") || '<p class="empty-hint">该知识库还没有文档。</p>';
  elements.documentPageInfo.textContent = `第 ${state.documentPage} 页 / 共 ${pageCount} 页`;
  elements.documentPrevPage.disabled = state.documentPage <= 1;
  elements.documentNextPage.disabled = state.documentPage >= pageCount;
  elements.documentPagination.hidden = pageCount <= 1;
}

// ==================== 知识内容（Chunks）功能 ====================

async function loadChunkDocumentFilter() {
  if (!state.selectedKnowledgeBaseId) {
    elements.chunkDocumentFilter.innerHTML = '<option value="">全部文档</option>';
    return;
  }
  try {
    const knowledgeBaseId = state.selectedKnowledgeBaseId;
    const documents = [];
    let total = 0;
    do {
      const data = await api(`/api/knowledge-bases/${knowledgeBaseId}/documents?skip=${documents.length}&limit=100`);
      documents.push(...(data.items || []));
      total = data.total || 0;
      if (!data.items?.length) break;
    } while (documents.length < total);
    if (knowledgeBaseId !== state.selectedKnowledgeBaseId) return;
    state.chunkDocuments = documents.filter((doc) => doc.is_current && doc.status === "ready");
    const select = elements.chunkDocumentFilter;
    select.innerHTML = '<option value="">全部文档</option>';
    state.chunkDocuments.forEach(doc => {
      const opt = document.createElement("option");
      opt.value = doc.document_id;
      opt.textContent = `${doc.filename} (v${doc.version})`;
      select.appendChild(opt);
    });
    if (state.chunkDocumentFilter && !state.chunkDocuments.some((doc) => doc.document_id === state.chunkDocumentFilter)) state.chunkDocumentFilter = "";
    select.value = state.chunkDocumentFilter;
  } catch (e) {
    showToast(`文档筛选加载失败：${e.message}`, "error");
  }
}

function renderChunkPagination() {
  const totalPages = Math.max(1, Math.ceil(state.chunkTotal / state.chunkPageSize));
  const first = state.chunkTotal ? (state.chunkPage - 1) * state.chunkPageSize + 1 : 0;
  const last = Math.min(state.chunkPage * state.chunkPageSize, state.chunkTotal);
  elements.chunkRange.textContent = state.chunkTotal
    ? `显示 ${first}–${last} 条，共 ${state.chunkTotal} 条切片`
    : "共 0 条切片";
  elements.chunkPageInfo.textContent = `第 ${state.chunkPage} 页 / 共 ${totalPages} 页`;
  elements.chunkFirstPage.disabled = state.chunkPage <= 1;
  elements.chunkPrevPage.disabled = state.chunkPage <= 1;
  elements.chunkNextPage.disabled = state.chunkPage >= totalPages;
  elements.chunkLastPage.disabled = state.chunkPage >= totalPages;
  const start = Math.max(1, Math.min(state.chunkPage - 2, totalPages - 4));
  const end = Math.min(totalPages, start + 4);
  elements.chunkPageNumbers.innerHTML = Array.from(
    { length: end - start + 1 }, (_, index) => start + index
  ).map((page) => `<button type="button" data-chunk-page="${page}" aria-label="第 ${page} 页" ${page === state.chunkPage ? 'aria-current="page"' : ""}>${page}</button>`).join("");
  elements.chunkPagination.hidden = state.chunkTotal === 0;
}

async function loadChunks(resetPage = false) {
  if (!state.selectedKnowledgeBaseId) {
    elements.chunkList.innerHTML = '<p class="empty-hint">请选择知识库。</p>';
    elements.chunkCount.textContent = "0 条知识";
    state.chunkTotal = 0;
    elements.chunkRange.textContent = "共 0 条切片";
    elements.chunkPagination.hidden = true;
    return;
  }
  if (resetPage) state.chunkPage = 1;
  const requestId = ++state.chunkRequestId;
  elements.chunkList.innerHTML = '<p class="empty-hint">正在加载切片…</p>';
  elements.chunkRange.textContent = "正在加载切片…";
  elements.chunkPagination.hidden = true;

  const params = new URLSearchParams({
    page: state.chunkPage,
    page_size: state.chunkPageSize,
  });
  if (state.chunkKeyword) params.set("keyword", state.chunkKeyword);
  if (state.chunkDocumentFilter) params.set("document_id", state.chunkDocumentFilter);

  try {
    const data = await api(`/api/knowledge-bases/${state.selectedKnowledgeBaseId}/chunks?${params.toString()}`);
    if (requestId !== state.chunkRequestId) return;
    state.chunkTotal = data.total || 0;
    elements.chunkCount.textContent = `${state.chunkTotal} 条知识`;
    const totalPages = Math.max(1, Math.ceil(state.chunkTotal / state.chunkPageSize));
    if (state.chunkPage > totalPages) {
      state.chunkPage = totalPages;
      return loadChunks();
    }

    if (!data.items || data.items.length === 0) {
      const filtered = Boolean(state.chunkKeyword || state.chunkDocumentFilter);
      elements.chunkList.innerHTML = `<div class="chunk-empty"><strong>${filtered ? "没有匹配的切片" : "这里还没有切片"}</strong><p>${filtered ? "试试更短的关键词，或切换来源文档。" : "上传资料并完成入库后，切片会显示在这里。"}</p>${filtered ? '<button type="button" data-clear-chunk-search>清除筛选</button>' : '<button type="button" data-go-page="import">去导入资料</button>'}</div>`;
      renderChunkPagination();
      return;
    }

    elements.chunkList.innerHTML = data.items.map((chunk, index) => {
      const content = chunk.content || "";
      const number = (state.chunkPage - 1) * state.chunkPageSize + index + 1;
      return `<article class="chunk-row" data-chunk-id="${escapeHtml(chunk._id)}" tabindex="0" role="button" aria-label="查看第 ${number} 条切片">
        <div class="chunk-index">${String(number).padStart(2, "0")}</div>
        <div class="chunk-preview"><div class="chunk-title-line"><strong>${escapeHtml(chunk.title || chunk.parent_title || "未命名切片")}</strong><span>${content.length} 字</span></div>
          <p>${escapeHtml(content.slice(0, 240))}${content.length > 240 ? "…" : ""}</p>
          <div class="chunk-meta"><span>▤ ${escapeHtml(chunk.source_document || "未知来源")}</span><span>版本 v${Number(chunk.version) || 1}</span><span>段落 ${Number(chunk.part) || 0}</span></div></div>
        <span class="chunk-open" aria-hidden="true">↗</span></article>`;
    }).join("");

    renderChunkPagination();

  } catch (error) {
    if (requestId !== state.chunkRequestId) return;
    elements.chunkList.innerHTML = `<p class="empty-hint error">加载失败：${escapeHtml(error.message)}</p>`;
    elements.chunkRange.textContent = "切片加载失败";
    elements.chunkPagination.hidden = true;
  }
}

function showChunkDetail(chunkId) {
  if (!state.selectedKnowledgeBaseId) return;

  api(`/api/knowledge-bases/${state.selectedKnowledgeBaseId}/chunks/${chunkId}`)
    .then(chunk => {
      const modal = document.createElement("div");
      modal.className = "modal-overlay";
      modal.innerHTML = `
        <div class="modal">
          <div class="modal-header">
            <div><p class="eyebrow">CHUNK DETAIL</p><h3>${escapeHtml(chunk.title || chunk.parent_title || "知识切片")}</h3></div>
            <button class="close-btn" type="button" aria-label="关闭详情">&times;</button>
          </div>
          <div class="modal-body">
            <div class="chunk-detail-meta">
              <div><strong>来源文档：</strong> ${escapeHtml(chunk.source_document || "-")}</div>
              <div><strong>版本：</strong> v${chunk.version || 1}</div>
              <div><strong>段落：</strong> ${Number(chunk.part) || 0}</div>
              <div><strong>字数：</strong> ${(chunk.content || "").length}</div>
            </div>
            <div class="chunk-content-box">
              <pre>${escapeHtml(chunk.content || "")}</pre>
            </div>
            <div class="chunk-meta-extra">
              <small>Chunk ID: ${escapeHtml(chunk._id || chunk.chunk_id)}</small>
            </div>
          </div>
          <div class="modal-footer">
            <button class="primary-button close-btn">关闭</button>
          </div>
        </div>
      `;
      document.body.appendChild(modal);
      const previousFocus = document.activeElement;
      modal.querySelector(".close-btn").focus();
      const close = () => { modal.remove(); previousFocus?.focus(); document.removeEventListener("keydown", onKeydown); };
      const onKeydown = (event) => { if (event.key === "Escape") close(); };
      document.addEventListener("keydown", onKeydown);

      modal.querySelectorAll(".close-btn").forEach(btn => {
        btn.onclick = close;
      });
      modal.onclick = (e) => {
        if (e.target === modal) close();
      };
    })
    .catch(err => showToast("加载详情失败: " + err.message, "error"));
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

function renderPagedControls(kind, page, pageSize, total) {
  const pageCount = Math.max(1, Math.ceil(total / pageSize));
  elements[`${kind}Total`].textContent = kind === "audit" ? `${total} 条记录` : `${total} 条评测记录`;
  elements[`${kind}PageInfo`].textContent = `第 ${page} 页 / 共 ${pageCount} 页`;
  elements[`${kind}PrevPage`].disabled = page <= 1;
  elements[`${kind}NextPage`].disabled = page >= pageCount;
  elements[`${kind}Pagination`].hidden = pageCount <= 1;
  return pageCount;
}

async function loadAuditLogs(resetPage = false) {
  if (state.authEnabled && state.currentUser?.role !== "admin") return;
  if (resetPage) state.auditPage = 1;
  const requestId = ++state.auditRequestId;
  const params = new URLSearchParams({
    skip: String((state.auditPage - 1) * state.auditPageSize), limit: String(state.auditPageSize),
  });
  const action = $("#auditActionFilter").value;
  const outcome = $("#auditOutcomeFilter").value;
  if (action) params.set("action", action);
  if (outcome) params.set("outcome", outcome);
  const data = await api(`/api/audit-logs?${params}`);
  if (requestId !== state.auditRequestId) return;
  state.auditTotal = data.total || 0;
  const pageCount = Math.max(1, Math.ceil(state.auditTotal / state.auditPageSize));
  if (state.auditPage > pageCount) { state.auditPage = pageCount; return loadAuditLogs(); }
  const actionNames = {
    "auth.login": "账号登录", "auth.logout": "账号退出", "user.create": "创建账号", "user.update": "更新账号",
    "knowledge_base.create": "创建知识库", "knowledge_base.update": "更新知识库", "knowledge_base.delete": "删除知识库",
    "document.import": "文档入库", "document.delete": "删除文档", "document.retry": "重试文档",
    "query.create": "发起问答", "evaluation.create": "运行评测", "evaluation.baseline": "更新评测基线", "task.cancel": "取消任务", "task.retry": "重试任务",
    "session.rename": "重命名会话", "session.delete": "删除会话", "history.delete": "清除历史",
  };
  const outcomeNames = { success: "成功", failure: "失败", denied: "拒绝" };
  elements.auditList.innerHTML = (data.items || []).map((item) => `
    <div class="audit-row">
      <strong>${escapeHtml(actionNames[item.action] || item.action)}</strong>
      <span class="audit-outcome ${escapeHtml(item.outcome)}">${escapeHtml(outcomeNames[item.outcome] || item.outcome)}</span>
      <span>${escapeHtml(item.actor_username || "匿名/本地模式")}<small>${escapeHtml(item.actor_role || "—")}</small></span>
      <span>${escapeHtml(item.target_id || item.target_type || "—")}<small>${escapeHtml(item.path || "")}</small></span>
      <time>${escapeHtml((item.created_at || "").slice(0, 19).replace("T", " "))}</time>
    </div>`).join("") || '<p class="empty-hint">暂无审计记录。</p>';
  renderPagedControls("audit", state.auditPage, state.auditPageSize, state.auditTotal);
}

async function loadEvaluationRuns(resetPage = false) {
  if (state.authEnabled && state.currentUser?.role === "viewer") return;
  if (resetPage) state.evaluationPage = 1;
  const requestId = ++state.evaluationRequestId;
  const params = new URLSearchParams({
    skip: String((state.evaluationPage - 1) * state.evaluationPageSize),
    limit: String(state.evaluationPageSize),
  });
  const data = await api(`/api/evaluations?${params}`);
  if (requestId !== state.evaluationRequestId) return;
  state.evaluationTotal = data.total || 0;
  const pageCount = Math.max(1, Math.ceil(state.evaluationTotal / state.evaluationPageSize));
  if (state.evaluationPage > pageCount) { state.evaluationPage = pageCount; return loadEvaluationRuns(); }
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
  renderPagedControls("evaluation", state.evaluationPage, state.evaluationPageSize, state.evaluationTotal);
}

$("#knowledgeBaseForm").addEventListener("submit", async (event) => {
  event.preventDefault();
  try {
    const created = await api("/api/knowledge-bases", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ name: $("#knowledgeBaseName").value.trim(), description: $("#knowledgeBaseDescription").value.trim() }),
    });
    state.selectedKnowledgeBaseId = created.knowledge_base_id;
    state.documentPage = 1;
    event.target.reset(); await loadKnowledgeBases(); showToast("知识库已创建");
  } catch (error) { showToast(error.message, "error"); }
});

elements.knowledgeBaseSelect.addEventListener("change", async () => {
  state.selectedKnowledgeBaseId = elements.knowledgeBaseSelect.value;
  state.documentPage = 1;
  state.chunkRequestId++;
  state.chunkDocumentFilter = "";
  state.chunkPage = 1;
  await loadDocuments().catch((error) => showToast(error.message, "error"));
  await loadKnowledgeOverview();
  if (!$("#chunksPanel").hidden && !document.querySelector('[data-library-page="documents"]').hidden) {
    await loadChunkDocumentFilter();
    await loadChunks(true);
  }
});
elements.documentPrevPage.addEventListener("click", () => {
  if (state.documentPage > 1) { state.documentPage--; loadDocuments().catch((error) => showToast(error.message, "error")); }
});
elements.documentNextPage.addEventListener("click", () => {
  if (state.documentPage * state.documentPageSize < state.documentTotal) {
    state.documentPage++; loadDocuments().catch((error) => showToast(error.message, "error"));
  }
});
$("#refreshDocumentsButton").addEventListener("click", async () => {
  try {
    await loadKnowledgeBases();
    if (!$("#chunksPanel").hidden && !document.querySelector('[data-library-page="documents"]').hidden) {
      await loadChunkDocumentFilter();
      await loadChunks();
    }
    showToast("数据已刷新");
  } catch (error) { showToast(error.message, "error"); }
});
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
$("#auditActionFilter").addEventListener("change", () => loadAuditLogs(true).catch((error) => showToast(error.message, "error")));
$("#auditOutcomeFilter").addEventListener("change", () => loadAuditLogs(true).catch((error) => showToast(error.message, "error")));
elements.auditPageSize.addEventListener("change", () => {
  state.auditPageSize = Number(elements.auditPageSize.value);
  loadAuditLogs(true).catch((error) => showToast(error.message, "error"));
});
elements.auditPrevPage.addEventListener("click", () => {
  if (state.auditPage > 1) { state.auditPage--; loadAuditLogs().catch((error) => showToast(error.message, "error")); }
});
elements.auditNextPage.addEventListener("click", () => {
  if (state.auditPage * state.auditPageSize < state.auditTotal) {
    state.auditPage++; loadAuditLogs().catch((error) => showToast(error.message, "error"));
  }
});
elements.evaluationPageSize.addEventListener("change", () => {
  state.evaluationPageSize = Number(elements.evaluationPageSize.value);
  loadEvaluationRuns(true).catch((error) => showToast(error.message, "error"));
});
elements.evaluationPrevPage.addEventListener("click", () => {
  if (state.evaluationPage > 1) { state.evaluationPage--; loadEvaluationRuns().catch((error) => showToast(error.message, "error")); }
});
elements.evaluationNextPage.addEventListener("click", () => {
  if (state.evaluationPage * state.evaluationPageSize < state.evaluationTotal) {
    state.evaluationPage++; loadEvaluationRuns().catch((error) => showToast(error.message, "error"));
  }
});
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
  const chunksButton = event.target.closest("[data-show-document-chunks]");
  if (chunksButton) {
    state.chunkDocumentFilter = chunksButton.dataset.showDocumentChunks;
    state.chunkKeyword = ""; elements.chunkKeyword.value = "";
    switchView("chunks");
    return;
  }
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

// ==================== 知识内容事件绑定 ====================
function bindChunkEvents() {
  const goToPage = (page) => {
    const lastPage = Math.max(1, Math.ceil(state.chunkTotal / state.chunkPageSize));
    if (!Number.isInteger(page) || page < 1 || page > lastPage || page === state.chunkPage) return;
    state.chunkPage = page;
    loadChunks();
  };
  if (elements.chunkSearchBtn) {
    elements.chunkSearchBtn.addEventListener("click", () => {
      state.chunkKeyword = elements.chunkKeyword?.value.trim() || "";
      state.chunkDocumentFilter = elements.chunkDocumentFilter?.value || "";
      loadChunks(true);
    });
  }

  if (elements.chunkRefreshBtn) {
    elements.chunkRefreshBtn.addEventListener("click", () => {
      state.chunkKeyword = "";
      if (elements.chunkKeyword) elements.chunkKeyword.value = "";
      state.chunkDocumentFilter = "";
      if (elements.chunkDocumentFilter) elements.chunkDocumentFilter.value = "";
      loadChunks(true);
    });
  }

  if (elements.chunkKeyword) {
    elements.chunkKeyword.addEventListener("keydown", (e) => {
      if (e.key === "Enter") {
        e.preventDefault();
        state.chunkKeyword = elements.chunkKeyword.value.trim();
        state.chunkDocumentFilter = elements.chunkDocumentFilter?.value || "";
        loadChunks(true);
      }
    });
  }

  if (elements.chunkDocumentFilter) {
    elements.chunkDocumentFilter.addEventListener("change", () => {
      state.chunkDocumentFilter = elements.chunkDocumentFilter.value;
      loadChunks(true);
    });
  }

  elements.chunkPageSize.addEventListener("change", () => {
    state.chunkPageSize = Number(elements.chunkPageSize.value);
    loadChunks(true);
  });
  elements.chunkFirstPage.addEventListener("click", () => goToPage(1));
  elements.chunkLastPage.addEventListener("click", () => goToPage(Math.ceil(state.chunkTotal / state.chunkPageSize)));
  elements.chunkPageNumbers.addEventListener("click", (event) => {
    const button = event.target.closest("[data-chunk-page]");
    if (button) goToPage(Number(button.dataset.chunkPage));
  });

  if (elements.chunkPrevPage) {
    elements.chunkPrevPage.addEventListener("click", () => goToPage(state.chunkPage - 1));
  }

  if (elements.chunkNextPage) {
    elements.chunkNextPage.addEventListener("click", () => goToPage(state.chunkPage + 1));
  }

  // 点击或键盘打开切片详情。
  if (elements.chunkList) {
    elements.chunkList.addEventListener("click", (e) => {
      if (e.target.closest("[data-clear-chunk-search]")) {
        state.chunkKeyword = ""; elements.chunkKeyword.value = "";
        state.chunkDocumentFilter = ""; elements.chunkDocumentFilter.value = "";
        loadChunks(true); return;
      }
      const go = e.target.closest("[data-go-page]");
      if (go) { switchView(go.dataset.goPage); return; }
      const row = e.target.closest(".chunk-row");
      if (row?.dataset.chunkId) showChunkDetail(row.dataset.chunkId);
    });
    elements.chunkList.addEventListener("keydown", (event) => {
      if (!["Enter", " "].includes(event.key) || !event.target.matches(".chunk-row")) return;
      event.preventDefault(); showChunkDetail(event.target.dataset.chunkId);
    });
  }
}

async function init() {
  localStorage.setItem("kb-session-id", state.sessionId);
  state.chunkPageSize = Number(elements.chunkPageSize.value);
  state.auditPageSize = Number(elements.auditPageSize.value);
  state.evaluationPageSize = Number(elements.evaluationPageSize.value);
  elements.sessionLabel.textContent = state.sessionId.slice(0, 12);
  renderPipeline(elements.pipeline, queryPipeline); renderPipeline(elements.importPipeline, importPipeline, [], null, true);
  bindChunkEvents(); // 绑定知识内容事件
  document.querySelectorAll(".kb-content-tabs [data-tab]").forEach((button) => {
    button.addEventListener("click", () => switchView(button.dataset.tab));
  });

  // 智能问答页快速跳转到知识内容
  const jumpBtn = $("#jumpToChunksBtn");
  if (jumpBtn) {
    jumpBtn.addEventListener("click", () => {
      const kbId = elements.queryKnowledgeBase.value;
      if (!kbId) {
        showToast("请先选择一个知识库");
        return;
      }
      state.selectedKnowledgeBaseId = kbId;
      elements.knowledgeBaseSelect.value = kbId;
      switchView("chunks");
    });
  }
  try {
    if (await initializeAuth()) await loadApplicationData();
  }
  catch { elements.systemBadge.textContent = "离线"; elements.systemBadge.style.color = "#ff8da3"; }
  switchView(location.hash.replace(/^#\//, "") || "chat");
}

init();
