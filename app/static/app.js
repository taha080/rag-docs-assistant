const KEY_STORAGE = "rag-docs-assistant.groqApiKey";

const healthEl = document.querySelector("#health");
const keyInput = document.querySelector("#api-key");
const keyHint = document.querySelector("#key-hint");
const forgetButton = document.querySelector("#forget-key");
const fileInput = document.querySelector("#file-input");
const dropzone = document.querySelector("#dropzone");
const documentsEl = document.querySelector("#documents");
const libraryCount = document.querySelector("#library-count");
const threadEl = document.querySelector("#thread");
const emptyEl = document.querySelector("#empty");
const noticeEl = document.querySelector("#notice");
const askForm = document.querySelector("#ask-form");
const questionEl = document.querySelector("#question");
const askButton = document.querySelector("#ask-button");
const clearButton = document.querySelector("#clear-chat");

const state = { documents: [], messages: [], busy: false };

keyInput.value = readStoredKey();
keyInput.addEventListener("input", () => storeKey(keyInput.value));
forgetButton.addEventListener("click", () => {
  keyInput.value = "";
  storeKey("");
  keyInput.focus();
});

fileInput.addEventListener("change", () => {
  uploadFiles(fileInput.files);
  fileInput.value = "";
});

["dragenter", "dragover"].forEach((name) => {
  dropzone.addEventListener(name, (event) => {
    event.preventDefault();
    dropzone.classList.add("dragover");
  });
});

["dragleave", "drop"].forEach((name) => {
  dropzone.addEventListener(name, (event) => {
    event.preventDefault();
    dropzone.classList.remove("dragover");
  });
});

dropzone.addEventListener("drop", (event) => {
  uploadFiles(event.dataTransfer.files);
});

askForm.addEventListener("submit", (event) => {
  event.preventDefault();
  ask();
});

questionEl.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    askForm.requestSubmit();
  }
});

document.querySelectorAll("[data-prompt]").forEach((button) => {
  button.addEventListener("click", () => {
    questionEl.value = button.dataset.prompt;
    questionEl.focus();
  });
});

clearButton.addEventListener("click", () => {
  state.messages = [];
  renderThread();
  setNotice("");
});

refreshHealth();
refreshDocuments();

async function refreshHealth() {
  try {
    const response = await fetch("/api/health");
    if (!response.ok) throw new Error("Health check failed");
    const data = await response.json();
    healthEl.textContent = data.groq_configured ? "Server key configured" : "Paste a free Groq key";
    keyHint.textContent = data.groq_configured
      ? "A key is already configured on the server. Paste one here only to override it for this tab."
      : "Kept in this browser tab only. Sent with each question, never written to the server.";
  } catch {
    healthEl.textContent = "Server unreachable";
  }
}

async function refreshDocuments() {
  const response = await fetch("/api/documents");
  if (!response.ok) {
    setNotice(await readError(response));
    return;
  }
  state.documents = await response.json();
  renderDocuments();
}

function renderDocuments() {
  documentsEl.replaceChildren();
  const count = state.documents.length;
  libraryCount.textContent = count === 0 ? "No documents yet" : `${count} document${count === 1 ? "" : "s"}`;
  state.documents.forEach((doc) => {
    const item = document.createElement("li");
    const copy = document.createElement("div");
    const name = document.createElement("div");
    name.className = "doc-name";
    name.textContent = doc.filename;
    const meta = document.createElement("div");
    meta.className = "doc-meta";
    const passages = doc.chunk_count === 1 ? "1 passage" : `${doc.chunk_count} passages`;
    meta.textContent = `${passages} · ${formatBytes(doc.size_bytes)} · ${formatDate(doc.created_at)}`;
    copy.append(name, meta);
    const remove = document.createElement("button");
    remove.type = "button";
    remove.className = "danger";
    remove.textContent = "Delete";
    remove.setAttribute("aria-label", `Delete ${doc.filename}`);
    remove.addEventListener("click", () => removeDocument(doc));
    item.append(copy, remove);
    documentsEl.append(item);
  });
}

async function uploadFiles(fileList) {
  const files = Array.from(fileList || []);
  if (!files.length || state.busy) return;
  state.busy = true;
  setNotice(files.length === 1 ? `Uploading ${files[0].name}` : `Uploading ${files.length} files`);
  try {
    for (const file of files) {
      const body = new FormData();
      body.append("file", file, file.name);
      const response = await fetch("/api/documents", { method: "POST", body });
      if (!response.ok) {
        setNotice(`${file.name}: ${await readError(response)}`);
        break;
      }
      setNotice(`Added ${file.name}`);
    }
    await refreshDocuments();
  } catch {
    setNotice("Upload failed before the server answered.");
  } finally {
    state.busy = false;
  }
}

async function removeDocument(doc) {
  if (!window.confirm(`Delete ${doc.filename}?`)) return;
  const response = await fetch(`/api/documents/${doc.id}`, { method: "DELETE" });
  if (!response.ok && response.status !== 204) {
    setNotice(await readError(response));
    return;
  }
  setNotice(`Deleted ${doc.filename}`);
  await refreshDocuments();
}

async function ask() {
  const question = questionEl.value.trim();
  if (!question || state.busy) return;
  state.messages.push({ role: "user", content: question });
  renderThread();
  questionEl.value = "";
  state.busy = true;
  askButton.disabled = true;
  askButton.textContent = "Reading";
  setNotice("");
  try {
    const headers = { "Content-Type": "application/json" };
    const key = keyInput.value.trim();
    if (key) headers["X-Groq-Api-Key"] = key;
    const response = await fetch("/api/query", {
      method: "POST",
      headers,
      body: JSON.stringify({ question }),
    });
    if (!response.ok) {
      state.messages.push({ role: "assistant", content: await readError(response), sources: [] });
    } else {
      const data = await response.json();
      state.messages.push({
        role: "assistant",
        content: data.answer,
        model: data.model,
        sources: data.sources || [],
      });
    }
  } catch {
    state.messages.push({
      role: "assistant",
      content: "The server could not be reached.",
      sources: [],
    });
  } finally {
    state.busy = false;
    askButton.disabled = false;
    askButton.textContent = "Ask";
    renderThread();
  }
}

function renderThread() {
  threadEl.replaceChildren();
  emptyEl.hidden = state.messages.length > 0;
  state.messages.forEach((message) => {
    const item = document.createElement("li");
    item.className = `bubble ${message.role}`;
    if (message.role === "user") {
      const paragraph = document.createElement("p");
      paragraph.textContent = message.content;
      item.append(paragraph);
    } else {
      item.append(renderAnswer(message));
    }
    threadEl.append(item);
  });
  threadEl.lastElementChild?.scrollIntoView({ block: "nearest" });
}

function renderAnswer(message) {
  const card = document.createElement("article");
  card.className = "answer";
  const text = document.createElement("p");
  text.className = "answer-text";
  text.textContent = message.content;
  card.append(text);
  if (message.model) {
    const model = document.createElement("p");
    model.className = "model-name";
    model.textContent = message.model;
    card.append(model);
  }
  const sources = message.sources || [];
  if (sources.length) {
    const list = document.createElement("ol");
    list.className = "sources";
    sources.forEach((source) => list.append(renderSource(source)));
    card.append(list);
  }
  return card;
}

function renderSource(source) {
  const item = document.createElement("li");
  const details = document.createElement("details");
  const summary = document.createElement("summary");
  summary.textContent = source.label;
  const quote = document.createElement("blockquote");
  quote.textContent = source.text;
  const score = document.createElement("p");
  score.className = "score";
  const percent = Math.round((source.score || 0) * 100);
  score.textContent = `Relevance ${percent}%`;
  details.append(summary, quote, score);
  item.append(details);
  return item;
}

function setNotice(message) {
  if (!message) {
    noticeEl.hidden = true;
    noticeEl.textContent = "";
    return;
  }
  noticeEl.hidden = false;
  noticeEl.textContent = message;
}

async function readError(response) {
  try {
    const data = await response.json();
    if (data.message) return data.message;
    if (data.details && data.details[0] && data.details[0].msg) return data.details[0].msg;
    if (typeof data.detail === "string") return data.detail;
  } catch {
    /* The body was not JSON. */
  }
  return response.statusText || "Request failed";
}

function readStoredKey() {
  try {
    return sessionStorage.getItem(KEY_STORAGE) || "";
  } catch {
    return "";
  }
}

function storeKey(value) {
  try {
    if (value) sessionStorage.setItem(KEY_STORAGE, value);
    else sessionStorage.removeItem(KEY_STORAGE);
  } catch {
    /* Private mode can reject storage. The field still sends the key. */
  }
}

function formatBytes(bytes) {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

function formatDate(iso) {
  const date = new Date(iso);
  if (Number.isNaN(date.getTime())) return iso;
  return date.toLocaleString(undefined, {
    month: "short",
    day: "numeric",
    hour: "numeric",
    minute: "2-digit",
  });
}
