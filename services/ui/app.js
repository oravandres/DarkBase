/**
 * MiMi AI Platform — Frontend Application
 * Vanilla JS, API-driven, with SSE streaming for chat.
 */

(() => {
    'use strict';

    // ─── Config ───────────────────────────────────────────────────────
    const API_BASE = '';  // Same origin via ingress
    const HEALTH_INTERVAL = 5000;
    const POLL_INTERVAL = 3000;
    const HISTORY_KEY = 'mimi-ai-chat-history';

    // ─── State ────────────────────────────────────────────────────────
    let chatHistory = [];
    let selectedModel = '';
    let isGenerating = false;
    let activeJobs = new Map();  // jobId → jobInfo

    // ─── DOM refs ─────────────────────────────────────────────────────
    const $ = (sel) => document.querySelector(sel);
    const chatMessages = $('#chatMessages');
    const chatInput = $('#chatInput');
    const sendBtn = $('#sendBtn');
    const modelSelect = $('#modelSelect');
    const clearChatBtn = $('#clearChatBtn');
    const chatTabBtn = $('#chatTabBtn');
    const imagesTabBtn = $('#imagesTabBtn');
    const historyTabBtn = $('#historyTabBtn');
    const chatTabContent = $('#chatTab');
    const imagesTabContent = $('#imagesTab');
    const historyTabContent = $('#historyTab');
    // Img2Img controls
    const imgUpload = $('#imgUpload');
    const imgUploadClear = $('#imgUploadClear');
    const imgPreviewContainer = $('#imgPreviewContainer');
    const imgPreview = $('#imgPreview');
    const denoiseContainer = $('#denoiseContainer');
    const imgDenoise = $('#imgDenoise');
    const denoiseVal = $('#denoiseVal');
    const imagePrompt = $('#imagePrompt');
    const generateBtn = $('#generateBtn');
    const imageGallery = $('#imageGallery');
    const queuePanel = $('#queuePanel');
    const queueInfo = $('#queueInfo');
    const imageModal = $('#imageModal');
    const modalImage = $('#modalImage');
    const modalMeta = $('#modalMeta');
    const modalClose = $('#modalClose');
    const llmStatusChip = $('#llmStatus');
    const imageStatusChip = $('#imageStatus');

    // ─── Tab switching ────────────────────────────────────────────────
    let historyLoaded = false;

    function switchTab(tab) {
        document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
        document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
        if (tab === 'chat') {
            chatTabBtn.classList.add('active');
            chatTab.classList.add('active');
        } else if (tab === 'images') {
            imagesTabBtn.classList.add('active');
            imagesTab.classList.add('active');
        } else if (tab === 'history') {
            historyTabBtn.classList.add('active');
            historyTab.classList.add('active');
            if (!historyLoaded) {
                historyLoaded = true;
                loadHistory();
            }
        }
    }

    chatTabBtn.addEventListener('click', () => switchTab('chat'));
    imagesTabBtn.addEventListener('click', () => switchTab('images'));
    historyTabBtn.addEventListener('click', () => switchTab('history'));

    // ─── Status polling ───────────────────────────────────────────────
    function updateStatusChip(chip, state) {
        const dot = chip.querySelector('.status-dot');
        const value = chip.querySelector('.status-value');
        dot.className = 'status-dot ' + state;
        value.textContent = state;
    }

    async function pollHealth() {
        try {
            const llmRes = await fetch(`${API_BASE}/api/v1/chat/health`);
            if (llmRes.ok) {
                const data = await llmRes.json();
                updateStatusChip(llmStatusChip, data.state);
            } else {
                updateStatusChip(llmStatusChip, 'error');
            }
        } catch {
            updateStatusChip(llmStatusChip, 'error');
        }

        try {
            const imgRes = await fetch(`${API_BASE}/api/v1/images/health`);
            if (imgRes.ok) {
                const data = await imgRes.json();
                let state = data.state;
                if (data.queue_depth > 0 && state !== 'busy') state = 'queued';
                updateStatusChip(imageStatusChip, state);
            } else {
                updateStatusChip(imageStatusChip, 'error');
            }
        } catch {
            updateStatusChip(imageStatusChip, 'error');
        }
    }

    setInterval(pollHealth, HEALTH_INTERVAL);
    pollHealth();

    // ─── Load models ──────────────────────────────────────────────────
    async function loadModels() {
        try {
            const res = await fetch(`${API_BASE}/api/v1/chat/models`);
            if (!res.ok) throw new Error('Failed to load models');
            const data = await res.json();
            modelSelect.innerHTML = '';
            data.models.forEach((m, i) => {
                const opt = document.createElement('option');
                opt.value = m.id;
                opt.textContent = m.name;
                if (i === 0) opt.selected = true;
                modelSelect.appendChild(opt);
            });
            selectedModel = data.models[0]?.id || '';
        } catch {
            modelSelect.innerHTML = '<option value="">Models unavailable</option>';
        }
    }

    modelSelect.addEventListener('change', () => {
        selectedModel = modelSelect.value;
    });

    loadModels();

    // ─── Chat history ─────────────────────────────────────────────────
    function loadHistory() {
        try {
            const saved = localStorage.getItem(HISTORY_KEY);
            if (saved) {
                chatHistory = JSON.parse(saved);
                renderChat();
            }
        } catch { /* ignore */ }
    }

    function saveHistory() {
        try {
            // Keep last 50 messages
            const toSave = chatHistory.slice(-50);
            localStorage.setItem(HISTORY_KEY, JSON.stringify(toSave));
        } catch { /* ignore */ }
    }

    function renderChat() {
        if (chatHistory.length === 0) {
            chatMessages.innerHTML = `
                <div class="welcome-message">
                    <div class="welcome-icon">◆</div>
                    <h2>MiMi AI Chat</h2>
                    <p>Powered by DeepSeek R1 on DarkBase</p>
                </div>`;
            return;
        }

        chatMessages.innerHTML = '';
        chatHistory.forEach(msg => {
            appendMessageBubble(msg.role, msg.content, false);
        });
        scrollToBottom();
    }

    function appendMessageBubble(role, content, animate = true) {
        const div = document.createElement('div');
        div.className = `message ${role}`;
        if (!animate) div.style.animation = 'none';

        // Parse thinking blocks from DeepSeek-style <think>...</think>
        let displayContent = content;
        const thinkMatch = content.match(/<think>([\s\S]*?)<\/think>/);
        if (thinkMatch) {
            const thinking = thinkMatch[1].trim();
            const rest = content.replace(/<think>[\s\S]*?<\/think>/, '').trim();
            displayContent = '';
            if (thinking) {
                displayContent += `<div class="thinking">${escapeHtml(thinking)}</div>`;
            }
            displayContent += escapeHtml(rest);
        } else {
            displayContent = escapeHtml(content);
        }

        div.innerHTML = `
            <div class="message-role ${role}">${role}</div>
            <div class="message-content">${displayContent}</div>`;
        chatMessages.appendChild(div);
        return div;
    }

    function escapeHtml(str) {
        const el = document.createElement('span');
        el.textContent = str;
        return el.innerHTML;
    }

    function scrollToBottom() {
        chatMessages.scrollTop = chatMessages.scrollHeight;
    }

    clearChatBtn.addEventListener('click', () => {
        chatHistory = [];
        localStorage.removeItem(HISTORY_KEY);
        renderChat();
    });

    loadHistory();

    // ─── Chat send ────────────────────────────────────────────────────
    async function sendMessage() {
        const text = chatInput.value.trim();
        if (!text || isGenerating) return;
        if (!selectedModel) {
            alert('No model selected');
            return;
        }

        // Remove welcome
        const welcome = chatMessages.querySelector('.welcome-message');
        if (welcome) welcome.remove();

        // Add user message
        chatHistory.push({ role: 'user', content: text });
        appendMessageBubble('user', text);
        chatInput.value = '';
        chatInput.style.height = 'auto';
        scrollToBottom();

        isGenerating = true;
        sendBtn.disabled = true;

        // Create assistant placeholder
        const assistantDiv = appendMessageBubble('assistant', '');
        const contentEl = assistantDiv.querySelector('.message-content');
        contentEl.textContent = '…';

        try {
            const res = await fetch(`${API_BASE}/api/v1/chat/completions`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    model: selectedModel,
                    messages: chatHistory.map(m => ({ role: m.role, content: m.content })),
                    stream: true,
                }),
            });

            if (!res.ok) {
                const errData = await res.json().catch(() => ({}));
                throw new Error(errData.error?.message || `HTTP ${res.status}`);
            }

            // Check if response is SSE
            const ct = res.headers.get('content-type') || '';
            if (ct.includes('text/event-stream')) {
                let fullContent = '';
                const reader = res.body.getReader();
                const decoder = new TextDecoder();
                let buffer = '';

                while (true) {
                    const { done, value } = await reader.read();
                    if (done) break;

                    buffer += decoder.decode(value, { stream: true });
                    const lines = buffer.split('\n');
                    buffer = lines.pop() || '';

                    for (const line of lines) {
                        if (line.startsWith('data: ')) {
                            const data = line.slice(6).trim();
                            if (data === '[DONE]') continue;
                            try {
                                const parsed = JSON.parse(data);
                                const delta = parsed.choices?.[0]?.delta?.content || '';
                                fullContent += delta;
                                // Re-render with thinking support
                                updateAssistantContent(contentEl, fullContent);
                                scrollToBottom();
                            } catch { /* skip parse errors */ }
                        }
                    }
                }

                chatHistory.push({ role: 'assistant', content: fullContent });
            } else {
                // Non-streaming response
                const data = await res.json();
                const content = data.choices?.[0]?.message?.content || '(empty response)';
                updateAssistantContent(contentEl, content);
                chatHistory.push({ role: 'assistant', content });
            }
        } catch (err) {
            contentEl.className = 'message-content message-error';
            contentEl.textContent = `Error: ${err.message}`;
        } finally {
            isGenerating = false;
            sendBtn.disabled = false;
            saveHistory();
            scrollToBottom();
        }
    }

    function updateAssistantContent(el, content) {
        const thinkMatch = content.match(/<think>([\s\S]*?)<\/think>/);
        if (thinkMatch) {
            const thinking = thinkMatch[1].trim();
            const rest = content.replace(/<think>[\s\S]*?<\/think>/, '').trim();
            let html = '';
            if (thinking) html += `<div class="thinking">${escapeHtml(thinking)}</div>`;
            html += escapeHtml(rest);
            el.innerHTML = html;
        } else if (content.includes('<think>')) {
            // Partial thinking block (still streaming)
            const parts = content.split('<think>');
            let html = escapeHtml(parts[0]);
            if (parts[1]) html += `<div class="thinking">${escapeHtml(parts[1])}</div>`;
            el.innerHTML = html;
        } else {
            el.textContent = content;
        }
    }

    sendBtn.addEventListener('click', sendMessage);
    chatInput.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            sendMessage();
        }
    });

    // Auto-resize textarea
    chatInput.addEventListener('input', () => {
        chatInput.style.height = 'auto';
        chatInput.style.height = Math.min(chatInput.scrollHeight, 200) + 'px';
    });

    // ─── Image generation ─────────────────────────────────────────────
    let galleryImages = [];  // { id, prompt, image_url, seed, width, height, steps, duration }

    function renderGallery() {
        if (galleryImages.length === 0 && activeJobs.size === 0) {
            imageGallery.innerHTML = `
                <div class="gallery-empty">
                    <div class="gallery-empty-icon">
                        <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1"><rect x="3" y="3" width="18" height="18" rx="2"/><circle cx="8.5" cy="8.5" r="1.5"/><path d="M21 15l-5-5L5 21"/></svg>
                    </div>
                    <p>Generated images will appear here</p>
                </div>`;
            return;
        }

        let html = '<div class="gallery-grid">';

        // Active/processing jobs first
        activeJobs.forEach((job) => {
            if (job.status === 'queued' || job.status === 'processing') {
                html += `
                    <div class="gallery-card processing">
                        <div class="processing-placeholder">
                            <div class="spinner"></div>
                            <span>${job.status === 'processing' ? 'Generating…' : `Queued (#${job.queue_position || '?'})`}</span>
                        </div>
                        <div class="gallery-card-meta">
                            <div class="gallery-card-prompt">${escapeHtml(job.prompt)}</div>
                        </div>
                    </div>`;
            }
        });

        // Completed images
        galleryImages.forEach((img) => {
            html += `
                <div class="gallery-card" data-id="${img.id}" onclick="window._openImage('${img.id}')">
                    <img src="${img.image_url}" alt="${escapeHtml(img.prompt)}" loading="lazy">
                    <div class="gallery-card-meta">
                        <div class="gallery-card-prompt">${escapeHtml(img.prompt)}</div>
                        <div class="gallery-card-info">
                            <span>${img.width}×${img.height}</span>
                            <span>seed: ${img.seed}</span>
                            ${img.duration_seconds ? `<span>${img.duration_seconds}s</span>` : ''}
                        </div>
                    </div>
                </div>`;
        });

        html += '</div>';
        imageGallery.innerHTML = html;
    }

    // Modal
    window._openImage = function (id) {
        const img = galleryImages.find(i => i.id === id);
        if (!img) return;
        modalImage.src = img.image_url;
        modalMeta.innerHTML = `
            <div>Prompt: <span>${escapeHtml(img.prompt)}</span></div>
            <div>Size: <span>${img.width}×${img.height}</span></div>
            <div>Steps: <span>${img.steps}</span></div>
            <div>Seed: <span>${img.seed}</span></div>
            ${img.duration_seconds ? `<div>Time: <span>${img.duration_seconds}s</span></div>` : ''}`;
        imageModal.style.display = 'flex';
    };

    modalClose.addEventListener('click', () => { imageModal.style.display = 'none'; });
    imageModal.addEventListener('click', (e) => {
        if (e.target === imageModal) imageModal.style.display = 'none';
    });
    document.addEventListener('keydown', (e) => {
        if (e.key === 'Escape') imageModal.style.display = 'none';
    });

    // Img2Img Event Listeners
    imgUpload.addEventListener('change', (e) => {
        const file = e.target.files[0];
        if (file) {
            const reader = new FileReader();
            reader.onload = (e) => {
                imgPreview.src = e.target.result;
                imgPreviewContainer.style.display = 'block';
                imgUploadClear.style.display = 'inline';
                denoiseContainer.style.display = 'block';
            };
            reader.readAsDataURL(file);
        } else {
            clearImageUpload();
        }
    });

    imgUploadClear.addEventListener('click', clearImageUpload);

    function clearImageUpload() {
        imgUpload.value = '';
        imgPreview.src = '';
        imgPreviewContainer.style.display = 'none';
        imgUploadClear.style.display = 'none';
        denoiseContainer.style.display = 'none';
    }

    imgDenoise.addEventListener('input', (e) => {
        denoiseVal.textContent = parseFloat(e.target.value).toFixed(2);
    });

    // Generate
    generateBtn.addEventListener('click', async () => {
        const prompt = imagePrompt.value.trim();
        if (!prompt) return;

        generateBtn.disabled = true;

        const width = parseInt($('#imgWidth').value) || 1024;
        const height = parseInt($('#imgHeight').value) || 1024;
        const steps = parseInt($('#imgSteps').value) || 20;
        const seedVal = $('#imgSeed').value.trim();
        const seed = seedVal ? parseInt(seedVal) : null;
        const denoise = parseFloat(imgDenoise.value) || 1.0;
        const file = imgUpload.files[0];

        try {
            const formData = new FormData();
            formData.append('prompt', prompt);
            formData.append('width', width);
            formData.append('height', height);
            formData.append('steps', steps);
            if (seed !== null) formData.append('seed', seed);
            formData.append('denoise', denoise);
            if (file) {
                formData.append('image', file);
            }

            const res = await fetch(`${API_BASE}/api/v1/images/generate`, {
                method: 'POST',
                body: formData, // fetch will automatically set multipart/form-data boundary
            });

            if (!res.ok) {
                const err = await res.json().catch(() => ({}));
                throw new Error(err.error?.message || `HTTP ${res.status}`);
            }

            const job = await res.json();
            activeJobs.set(job.id, job);
            renderGallery();
            updateQueuePanel();

            // Start polling this job
            pollJob(job.id);
        } catch (err) {
            alert(`Failed to queue image: ${err.message}`);
        } finally {
            generateBtn.disabled = false;
        }
    });

    async function pollJob(jobId) {
        while (true) {
            await sleep(POLL_INTERVAL);
            try {
                const res = await fetch(`${API_BASE}/api/v1/images/${jobId}`);
                if (!res.ok) break;
                const job = await res.json();
                activeJobs.set(jobId, job);

                if (job.status === 'completed') {
                    activeJobs.delete(jobId);
                    galleryImages.unshift({
                        id: job.id,
                        prompt: job.prompt,
                        image_url: job.image_url,
                        seed: job.seed,
                        width: job.width,
                        height: job.height,
                        steps: job.steps,
                        duration_seconds: job.duration_seconds,
                    });
                    renderGallery();
                    updateQueuePanel();
                    break;
                } else if (job.status === 'error') {
                    activeJobs.delete(jobId);
                    alert(`Image generation failed: ${job.error_message}`);
                    renderGallery();
                    updateQueuePanel();
                    break;
                }

                renderGallery();
                updateQueuePanel();
            } catch {
                break;
            }
        }
    }

    function updateQueuePanel() {
        const active = [...activeJobs.values()].filter(
            j => j.status === 'queued' || j.status === 'processing'
        );
        if (active.length === 0) {
            queuePanel.style.display = 'none';
            return;
        }

        queuePanel.style.display = 'block';
        queueInfo.innerHTML = active.map(j => `
            <div class="queue-item">
                <span class="queue-item-status ${j.status}">${j.status}</span>
                <span>${escapeHtml(j.prompt.slice(0, 30))}${j.prompt.length > 30 ? '…' : ''}</span>
            </div>
        `).join('');
    }

    function sleep(ms) {
        return new Promise(r => setTimeout(r, ms));
    }

    renderGallery();

    // ─── History ──────────────────────────────────────────────────────
    const historyTimeline = $('#historyTimeline');
    const historyLoadMore = $('#historyLoadMore');
    const loadMoreBtn = $('#loadMoreBtn');
    const filterAll = $('#filterAll');
    const filterChats = $('#filterChats');
    const filterImages = $('#filterImages');

    let historyEntries = [];
    let historyFilter = 'all';  // 'all' | 'chat' | 'image'
    let historyOffset = 0;
    const HISTORY_PAGE_SIZE = 30;
    let historyTotalChats = 0;
    let historyTotalImages = 0;
    let expandedEntryId = null;

    // Filter buttons
    [filterAll, filterChats, filterImages].forEach(btn => {
        btn.addEventListener('click', () => {
            document.querySelectorAll('.filter-btn').forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            historyFilter = btn.dataset.filter;
            renderHistory();
        });
    });

    async function loadHistory() {
        historyEntries = [];
        historyOffset = 0;
        await fetchHistory();
    }

    async function fetchHistory() {
        try {
            const [chatRes, imgRes] = await Promise.all([
                fetch(`${API_BASE}/api/v1/chat/history?limit=${HISTORY_PAGE_SIZE}&offset=${historyOffset}`).then(r => r.ok ? r.json() : { conversations: [], total: 0 }),
                fetch(`${API_BASE}/api/v1/images/history?limit=${HISTORY_PAGE_SIZE}&offset=${historyOffset}`).then(r => r.ok ? r.json() : { images: [], total: 0 }),
            ]);

            historyTotalChats = chatRes.total;
            historyTotalImages = imgRes.total;

            // Merge
            const chats = (chatRes.conversations || []).map(c => ({ ...c, type: 'chat' }));
            const images = (imgRes.images || []).map(i => ({ ...i, type: 'image' }));
            const combined = [...chats, ...images];

            // Sort by date descending
            combined.sort((a, b) => {
                const da = new Date(a.created_at || 0);
                const db = new Date(b.created_at || 0);
                return db - da;
            });

            if (historyOffset === 0) {
                historyEntries = combined;
            } else {
                historyEntries.push(...combined);
            }

            renderHistory();

            // Show/hide load more
            const totalLoaded = historyEntries.length;
            const totalAvailable = historyTotalChats + historyTotalImages;
            historyLoadMore.style.display = totalLoaded < totalAvailable ? 'block' : 'none';
        } catch (err) {
            console.error('Failed to fetch history:', err);
        }
    }

    loadMoreBtn.addEventListener('click', () => {
        historyOffset += HISTORY_PAGE_SIZE;
        fetchHistory();
    });

    function renderHistory() {
        const filtered = historyFilter === 'all'
            ? historyEntries
            : historyEntries.filter(e => e.type === historyFilter);

        if (filtered.length === 0) {
            historyTimeline.innerHTML = `
                <div class="history-empty">
                    <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1"><circle cx="12" cy="12" r="10"/><polyline points="12 6 12 12 16 14"/></svg>
                    <p>No ${historyFilter === 'all' ? '' : historyFilter + ' '}history yet</p>
                </div>`;
            return;
        }

        let html = '';
        filtered.forEach(entry => {
            if (entry.type === 'chat') {
                html += renderChatEntry(entry);
            } else {
                html += renderImageEntry(entry);
            }
        });
        historyTimeline.innerHTML = html;

        // Attach click handlers
        historyTimeline.querySelectorAll('.history-entry[data-id]').forEach(el => {
            el.addEventListener('click', () => onEntryClick(el.dataset.id, el.dataset.type));
        });
    }

    function formatTime(isoStr) {
        if (!isoStr) return '';
        try {
            const d = new Date(isoStr);
            const now = new Date();
            const diffMs = now - d;
            const diffMin = Math.floor(diffMs / 60000);
            if (diffMin < 1) return 'just now';
            if (diffMin < 60) return `${diffMin}m ago`;
            const diffHr = Math.floor(diffMin / 60);
            if (diffHr < 24) return `${diffHr}h ago`;
            const diffDay = Math.floor(diffHr / 24);
            if (diffDay < 7) return `${diffDay}d ago`;
            return d.toLocaleDateString();
        } catch { return isoStr; }
    }

    function renderChatEntry(entry) {
        const isExpanded = expandedEntryId === entry.id;
        return `
            <div class="history-entry" data-id="${entry.id}" data-type="chat">
                <div class="history-entry-header">
                    <span class="history-type-badge chat">
                        <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M21 15a2 2 0 01-2 2H7l-4 4V5a2 2 0 012-2h14a2 2 0 012 2z"/></svg>
                        Chat
                    </span>
                    <span class="history-entry-time">${formatTime(entry.created_at)}</span>
                </div>
                <div class="history-entry-preview">${escapeHtml(entry.preview || '(empty)')}</div>
                <div class="history-entry-meta">
                    <span>📊 ${entry.model || 'unknown'}</span>
                    <span>💬 ${entry.message_count || 0} messages</span>
                    ${entry.duration_seconds ? `<span>⏱ ${entry.duration_seconds}s</span>` : ''}
                </div>
                ${isExpanded ? '<div class="history-expanded" id="expanded-' + entry.id + '">Loading…</div>' : ''}
            </div>`;
    }

    function renderImageEntry(entry) {
        return `
            <div class="history-entry" data-id="${entry.id}" data-type="image">
                <div class="history-entry-header">
                    <span class="history-type-badge image">
                        <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="3" y="3" width="18" height="18" rx="2"/><circle cx="8.5" cy="8.5" r="1.5"/><path d="M21 15l-5-5L5 21"/></svg>
                        Image
                    </span>
                    <span class="history-entry-time">${formatTime(entry.created_at)}</span>
                </div>
                <div class="history-entry-thumb">
                    ${entry.image_url ? `<img src="${entry.image_url}" alt="Generated image" loading="lazy">` : ''}
                    <div class="history-entry-thumb-info">
                        <div class="history-entry-preview">${escapeHtml(entry.prompt || '')}</div>
                        <div class="history-entry-meta">
                            <span>${entry.width || '?'}×${entry.height || '?'}</span>
                            <span>seed: ${entry.seed || '?'}</span>
                            ${entry.duration_seconds ? `<span>⏱ ${entry.duration_seconds}s</span>` : ''}
                        </div>
                    </div>
                </div>
            </div>`;
    }

    async function onEntryClick(id, type) {
        if (type === 'image') {
            // Open image in modal
            const entry = historyEntries.find(e => e.id === id && e.type === 'image');
            if (entry && entry.image_url) {
                modalImage.src = entry.image_url;
                modalMeta.innerHTML = `
                    <div>Prompt: <span>${escapeHtml(entry.prompt || '')}</span></div>
                    <div>Size: <span>${entry.width}×${entry.height}</span></div>
                    <div>Seed: <span>${entry.seed}</span></div>
                    ${entry.duration_seconds ? `<div>Time: <span>${entry.duration_seconds}s</span></div>` : ''}`;
                imageModal.style.display = 'flex';
            }
            return;
        }

        // Toggle expanded chat
        if (expandedEntryId === id) {
            expandedEntryId = null;
            renderHistory();
            return;
        }

        expandedEntryId = id;
        renderHistory();

        // Fetch full conversation
        const expandedEl = document.getElementById(`expanded-${id}`);
        if (!expandedEl) return;

        try {
            const res = await fetch(`${API_BASE}/api/v1/chat/history/${id}`);
            if (!res.ok) throw new Error('Failed to load conversation');
            const data = await res.json();

            let html = '';
            (data.messages || []).forEach(msg => {
                html += `
                    <div class="history-msg">
                        <div class="history-msg-role ${msg.role}">${msg.role}</div>
                        <div class="history-msg-content">${escapeHtml(msg.content || '')}</div>
                    </div>`;
            });
            expandedEl.innerHTML = html || '<p style="color:var(--text-muted)">No messages</p>';
        } catch (err) {
            expandedEl.innerHTML = `<p style="color:var(--error)">Error: ${err.message}</p>`;
        }
    }

})();
