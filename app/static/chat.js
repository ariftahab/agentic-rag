// Sohbet sayfası. Durum (geçmiş, oturum) tarayıcıda tutulur; API stateless.

const state = {
  history: [],                     // API'ye giden sade geçmiş: [{role, content}], kaynak listesi yok
  sessionId: crypto.randomUUID(),  // log'da bu sohbetin sorularını gruplar
  busy: false,
};

const $ = (id) => document.getElementById(id);
const els = {};

function scrollToBottom() {
  els.messages.scrollTop = els.messages.scrollHeight;
}

function addEl(className, html) {
  els.empty.hidden = true;
  const div = document.createElement("div");
  div.className = className;
  div.innerHTML = html;
  els.inner.appendChild(div);
  scrollToBottom();
  return div;
}

function addUserMessage(text) {
  const div = addEl("msg-user", "");
  div.textContent = text; // textContent: kullanıcının yazdığı HTML olarak yorumlanmaz
}

function addBotMessage(body) {
  const chips = body.sources.map((s) =>
    `<button class="source-chip" data-chunk="${s.chunk_id}" title="${escapeHtml(s.label)}">${escapeHtml(s.label)}</button>`
  ).join("");
  const rewrite = body.rewritten_query
    ? `<span class="rewrite" title="Takip sorusu, geçmişe bakılarak bağımsız hale getirildi (query rewriting)">Aranan: ${escapeHtml(body.rewritten_query)}</span>`
    : "";
  const div = addEl("msg-bot", `
    <div class="answer">${renderAnswer(body.answer, body.sources)}</div>
    <div class="sources">${chips}</div>
    <div class="meta">
      <span>${(body.latency_ms / 1000).toFixed(1)} sn</span>
      ${rewrite}
      <span class="fb">
        <button data-rating="1" title="Faydalı" aria-label="Faydalı">👍</button>
        <button data-rating="-1" title="Faydalı değil" aria-label="Faydalı değil">👎</button>
      </span>
    </div>`);
  div.querySelectorAll(".fb button").forEach((btn) =>
    btn.addEventListener("click", () => sendFeedback(body.id, btn, div)));
}

async function sendFeedback(answerId, btn, msgEl) {
  const rating = Number(btn.dataset.rating);
  try {
    await api("/feedback", { method: "POST", body: { id: answerId, rating } });
    msgEl.querySelectorAll(".fb button").forEach((b) => b.classList.toggle("on", b === btn));
  } catch (e) {
    alert(`Geri bildirim kaydedilemedi: ${e.message}`);
  }
}

async function ask(question) {
  if (state.busy || !question.trim()) return;
  state.busy = true;
  els.send.disabled = true;
  addUserMessage(question);
  const typing = addEl("typing", "Cevap hazırlanıyor");
  try {
    const body = await api("/ask", {
      method: "POST",
      body: { question, history: state.history, session_id: state.sessionId },
    });
    typing.remove();
    addBotMessage(body);
    state.history.push({ role: "user", content: question }, { role: "assistant", content: body.answer });
  } catch (e) {
    // Hata ekranda görünür ama geçmişe eklenmez: sonraki soruların rewrite'ını bozmasın.
    typing.remove();
    addEl("msg-error", escapeHtml(e.message));
  } finally {
    state.busy = false;
    els.send.disabled = false;
    els.input.focus();
  }
}

function newChat() {
  state.history = [];
  state.sessionId = crypto.randomUUID();
  els.inner.querySelectorAll(".msg-user, .msg-bot, .msg-error").forEach((el) => el.remove());
  els.empty.hidden = false;
  panel.close();
  els.input.focus();
}

// Metin kutusu yazdıkça uzar (en fazla CSS'teki max-height kadar).
function autoGrow() {
  els.input.style.height = "auto";
  els.input.style.height = `${els.input.scrollHeight}px`;
}

document.addEventListener("DOMContentLoaded", () => {
  Object.assign(els, {
    messages: $("messages"), inner: $("messages-inner"), empty: $("empty"),
    form: $("composer"), input: $("question"), send: $("send"),
  });

  els.form.addEventListener("submit", (e) => {
    e.preventDefault(); // formun sayfayı yeniden yüklemesini engelle
    const q = els.input.value;
    els.input.value = "";
    autoGrow();
    ask(q);
  });
  els.input.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      els.form.requestSubmit();
    }
  });
  els.input.addEventListener("input", autoGrow);
  $("new-chat").addEventListener("click", newChat);
  els.empty.querySelectorAll(".examples button").forEach((b) =>
    b.addEventListener("click", () => ask(b.textContent)));
  els.input.focus();
});
