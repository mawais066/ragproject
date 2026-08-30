/**
 * Frontend JavaScript for PDF RAG Assistant
 */

document.addEventListener("DOMContentLoaded", () => {
  // Initialize Lucide Icons
  lucide.createIcons();

  // Dynamic API Base URL
  const API_BASE = (window.location.protocol.startsWith("http") && window.location.port === "8000") 
    ? "" 
    : "http://127.0.0.1:8000";

  // State
  let isDocumentLoaded = false;
  let activeFileName = "";
  let isAsking = false;

  // DOM Elements
  const dropZone = document.getElementById("dropZone");
  const fileInput = document.getElementById("pdfFileInput");
  const uploadOverlay = document.getElementById("uploadOverlay");
  const uploadOverlayText = document.getElementById("uploadOverlayText");

  const fileInfoCard = document.getElementById("fileInfoCard");
  const infoFileName = document.getElementById("infoFileName");
  const infoPages = document.getElementById("infoPages");
  const infoChunks = document.getElementById("infoChunks");
  const infoChars = document.getElementById("infoChars");
  const resetDocBtn = document.getElementById("resetDocBtn");

  const apiStatusBadge = document.getElementById("apiStatusBadge");
  const apiStatusText = document.getElementById("apiStatusText");
  const docStatusBadge = document.getElementById("docStatusBadge");
  const docStatusText = document.getElementById("docStatusText");
  const chatContextSubtitle = document.getElementById("chatContextSubtitle");

  const chatMessages = document.getElementById("chatMessages");
  const welcomeBox = document.getElementById("welcomeBox");
  const chatForm = document.getElementById("chatForm");
  const questionInput = document.getElementById("questionInput");
  const sendBtn = document.getElementById("sendBtn");
  const charCount = document.getElementById("charCount");
  const clearChatBtn = document.getElementById("clearChatBtn");
  const toastContainer = document.getElementById("toastContainer");

  // Initial Status Check
  checkServerStatus();

  // -------------------------------------------------------------
  // Server Status & Health Check
  // -------------------------------------------------------------
  async function checkServerStatus() {
    try {
      const res = await fetch(`${API_BASE}/api/status`);
      if (!res.ok) throw new Error("Status check failed");
      const data = await res.json();

      // Update API Key badge
      if (data.has_api_key) {
        apiStatusBadge.className = "status-badge ready";
        apiStatusText.textContent = `API Ready (${data.llm_model})`;
      } else {
        apiStatusBadge.className = "status-badge warning";
        apiStatusText.textContent = "API Key Missing (.env)";
        showToast("Warning: LLM_API_KEY is not set in your .env file.", "warning");
      }

      // Update Document status
      if (data.is_document_loaded) {
        setDocumentActiveState({
          filename: data.loaded_filename || "Active Document",
          total_pages: "-",
          total_chunks: "-",
          total_characters: "-",
        });
      } else {
        setDocumentEmptyState();
      }
    } catch (err) {
      console.warn("Could not reach backend status endpoint:", err);
      apiStatusBadge.className = "status-badge warning";
      apiStatusText.textContent = "Connecting to Server...";
    }
  }

  // -------------------------------------------------------------
  // File Upload Handlers (Drag & Drop + Click)
  // -------------------------------------------------------------
  ["dragenter", "dragover"].forEach((eventName) => {
    dropZone.addEventListener(eventName, (e) => {
      e.preventDefault();
      e.stopPropagation();
      dropZone.classList.add("dragover");
    });
  });

  ["dragleave", "drop"].forEach((eventName) => {
    dropZone.addEventListener(eventName, (e) => {
      e.preventDefault();
      e.stopPropagation();
      dropZone.classList.remove("dragover");
    });
  });

  dropZone.addEventListener("drop", (e) => {
    const files = e.dataTransfer.files;
    if (files.length > 0) {
      handleFileUpload(files[0]);
    }
  });

  fileInput.addEventListener("change", (e) => {
    if (fileInput.files.length > 0) {
      handleFileUpload(fileInput.files[0]);
    }
  });

  async function handleFileUpload(file) {
    if (!file.name.toLowerCase().endsWith(".pdf")) {
      showToast("Please select a valid PDF document (.pdf)", "error");
      return;
    }

    // Set uploading state
    dropZone.classList.add("uploading");
    uploadOverlayText.textContent = `Reading & chunking "${file.name}"...`;

    const formData = new FormData();
    formData.append("file", file);

    try {
      uploadOverlayText.textContent = "Generating embeddings & building vector index...";
      const response = await fetch(`${API_BASE}/upload-pdf`, {
        method: "POST",
        body: formData,
      });

      const result = await response.json();

      if (!response.ok) {
        throw new Error(result.detail || "Failed to process PDF.");
      }

      showToast(`Successfully indexed "${file.name}"!`, "success");
      setDocumentActiveState(result.stats);
    } catch (error) {
      showToast(error.message, "error");
    } finally {
      dropZone.classList.remove("uploading");
      fileInput.value = ""; // Reset input so same file can be re-uploaded if needed
    }
  }

  function setDocumentActiveState(stats) {
    isDocumentLoaded = true;
    activeFileName = stats.filename || "Uploaded PDF";

    // Update File info card
    infoFileName.textContent = activeFileName;
    infoPages.textContent = stats.total_pages ?? "-";
    infoChunks.textContent = stats.total_chunks ?? "-";
    infoChars.textContent = stats.total_characters ? stats.total_characters.toLocaleString() : "-";
    fileInfoCard.classList.remove("hidden");

    // Update Header badge
    docStatusBadge.className = "status-badge has-doc";
    docStatusText.textContent = `Indexed: ${activeFileName}`;

    // Update Chat subtitle
    chatContextSubtitle.textContent = `Asking questions against "${activeFileName}"`;

    lucide.createIcons();
  }

  function setDocumentEmptyState() {
    isDocumentLoaded = false;
    activeFileName = "";

    fileInfoCard.classList.add("hidden");
    docStatusBadge.className = "status-badge no-doc";
    docStatusText.textContent = "No PDF Uploaded";
    chatContextSubtitle.textContent = "Upload a PDF document to begin asking questions.";
  }

  // -------------------------------------------------------------
  // Document Reset Handler
  // -------------------------------------------------------------
  resetDocBtn.addEventListener("click", async () => {
    try {
      const res = await fetch(`${API_BASE}/reset`, { method: "POST" });
      if (res.ok) {
        setDocumentEmptyState();
        showToast("PDF document cleared. You can now upload a new one.", "info");
      }
    } catch (e) {
      showToast("Failed to reset document.", "error");
    }
  });

  // -------------------------------------------------------------
  // Chat Interaction & Q&A
  // -------------------------------------------------------------
  questionInput.addEventListener("input", () => {
    // Auto resize
    questionInput.style.height = "auto";
    questionInput.style.height = Math.min(questionInput.scrollHeight, 120) + "px";
    charCount.textContent = `${questionInput.value.length}/1000`;
  });

  questionInput.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      chatForm.dispatchEvent(new Event("submit"));
    }
  });

  chatForm.addEventListener("submit", async (e) => {
    e.preventDefault();
    const question = questionInput.value.trim();

    if (!question) return;

    if (!isDocumentLoaded) {
      showToast("Please upload a PDF document first before asking questions!", "warning");
      return;
    }

    if (isAsking) return;

    // Reset input
    questionInput.value = "";
    questionInput.style.height = "auto";
    charCount.textContent = "0/1000";

    // Hide welcome box if visible
    if (welcomeBox) welcomeBox.style.display = "none";

    // Append user message
    appendUserMessage(question);

    // Append loading typing bubble
    const typingElement = appendTypingIndicator();
    isAsking = true;
    sendBtn.disabled = true;

    try {
      const response = await fetch(`${API_BASE}/ask`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question }),
      });

      const data = await response.json();

      // Remove typing indicator
      typingElement.remove();

      if (!response.ok) {
        throw new Error(data.detail || "An error occurred while answering.");
      }

      appendAssistantMessage(data.answer, data.sources);
    } catch (error) {
      typingElement.remove();
      appendAssistantMessage(
        `**Error:** ${error.message}\n\n*Please ensure your LLM API credentials are configured in the .env file.*`,
        []
      );
      showToast(error.message, "error");
    } finally {
      isAsking = false;
      sendBtn.disabled = false;
      questionInput.focus();
    }
  });

  function appendUserMessage(text) {
    const row = document.createElement("div");
    row.className = "message-row user";
    row.innerHTML = `
      <div class="avatar user">
        <i data-lucide="user"></i>
      </div>
      <div class="message-bubble">
        <p>${escapeHtml(text)}</p>
      </div>
    `;
    chatMessages.appendChild(row);
    lucide.createIcons();
    scrollToBottom();
  }

  function appendTypingIndicator() {
    const row = document.createElement("div");
    row.className = "message-row assistant";
    row.innerHTML = `
      <div class="avatar assistant">
        <i data-lucide="bot"></i>
      </div>
      <div class="message-bubble typing-bubble">
        <div class="typing-dot"></div>
        <div class="typing-dot"></div>
        <div class="typing-dot"></div>
      </div>
    `;
    chatMessages.appendChild(row);
    lucide.createIcons();
    scrollToBottom();
    return row;
  }

  function appendAssistantMessage(markdownAnswer, sources = []) {
    const row = document.createElement("div");
    row.className = "message-row assistant";

    // Parse Markdown safely
    const parsedHtml = marked.parse(markdownAnswer);

    let sourcesHtml = "";
    if (sources && sources.length > 0) {
      const sourceItems = sources
        .map(
          (s) => `
          <div class="source-item">
            <div class="source-item-header">
              <span>Page ${s.page} (Chunk ${s.chunk_id})</span>
            </div>
            <div class="source-item-snippet">"${escapeHtml(s.content)}"</div>
          </div>
        `
        )
        .join("");

      sourcesHtml = `
        <div class="sources-container">
          <button class="sources-toggle-btn" onclick="toggleSources(this)">
            <i data-lucide="chevron-down"></i>
            <span>Retrieved Sources (${sources.length} chunks)</span>
          </button>
          <div class="sources-list">
            ${sourceItems}
          </div>
        </div>
      `;
    }

    row.innerHTML = `
      <div class="avatar assistant">
        <i data-lucide="bot"></i>
      </div>
      <div class="message-bubble">
        <div>${parsedHtml}</div>
        ${sourcesHtml}
      </div>
    `;

    chatMessages.appendChild(row);
    lucide.createIcons();
    scrollToBottom();
  }

  function scrollToBottom() {
    chatMessages.scrollTop = chatMessages.scrollHeight;
  }

  // Clear Chat Messages
  clearChatBtn.addEventListener("click", () => {
    chatMessages.innerHTML = "";
    if (welcomeBox) {
      welcomeBox.style.display = "flex";
      chatMessages.appendChild(welcomeBox);
    }
  });

  // Sample Questions Quick Click
  window.applySampleQuestion = function (text) {
    questionInput.value = text;
    questionInput.focus();
    // Dispatch input to resize
    questionInput.dispatchEvent(new Event("input"));
  };

  // Toggle Sources dropdown
  window.toggleSources = function (btn) {
    btn.classList.toggle("open");
    const list = btn.nextElementSibling;
    list.classList.toggle("show");
  };

  // -------------------------------------------------------------
  // Toast Notifications
  // -------------------------------------------------------------
  function showToast(message, type = "info") {
    const toast = document.createElement("div");
    toast.className = `toast ${type}`;

    let iconName = "info";
    if (type === "error") iconName = "alert-circle";
    if (type === "success") iconName = "check-circle-2";
    if (type === "warning") iconName = "alert-triangle";

    toast.innerHTML = `
      <i data-lucide="${iconName}"></i>
      <span>${escapeHtml(message)}</span>
    `;

    toastContainer.appendChild(toast);
    lucide.createIcons();

    setTimeout(() => {
      toast.style.opacity = "0";
      toast.style.transform = "translateX(30px)";
      toast.style.transition = "all 0.3s ease";
      setTimeout(() => toast.remove(), 300);
    }, 4500);
  }

  function escapeHtml(str) {
    if (!str) return "";
    return str
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;")
      .replace(/'/g, "&#039;");
  }
});
