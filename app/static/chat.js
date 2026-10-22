// Sohbet sayfası. Durum (geçmiş, oturum) tarayıcıda tutulur; API stateless.

const state = {
  history: [],                     // API'ye giden sade geçmiş: [{role, content}], kaynak listesi yok
  sessionId: crypto.randomUUID(),  // log'da bu sohbetin sorularını gruplar
  busy: false,
  // Transkript ve eşdeğerlik teyitleri yalnız bu değişkende durur (localStorage/sessionStorage yok): sayfa
  // yenilenince ya da kapanınca kaybolur. Sunucu da saklamaz; her /agent/ask isteğine eklenir.
  transcript: null,                // POST /transcript'in döndürdüğü kişisel bilgisiz JSON
  confirmations: {},               // {plan dışı ders kodu: plan ders kodu | null}; agent günceller, biz geri göndeririz
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

// Agent cevabı: kaynak düğmeleri ve 👍/👎 yok (agent cevapları soru-cevap log'una yazılmaz); altında çağrılan
// tool'lar görünür ki kullanıcı cevabın hangi hesaptan ya da aramadan geldiğini bilsin.
function addAgentMessage(body) {
  const tools = body.tool_calls.map((c) => {
    const args = Object.values(c.args).map((v) => JSON.stringify(v)).join(", ");
    return `<code>${escapeHtml(c.name)}(${escapeHtml(args)})</code>`;
  }).join(" → ");
  // Anlık tool (request_new_tool): kodu o an bir LLM yazdı, test edilmedi; kullanıcı bunu bilsin (#18).
  const generated = body.tool_calls.some((c) => c.name === "request_new_tool")
    ? `<div class="tools">⚠ Anlık üretilmiş hesap, doğrulanmamış</div>` : "";
  addEl("msg-bot", `
    <div class="answer">${renderAnswer(body.answer)}</div>
    <div class="tools">${tools ? `Kullanılan tool'lar: ${tools}` : "Tool kullanılmadı"}</div>${generated}
    <div class="meta"><span>${(body.latency_ms / 1000).toFixed(1)} sn</span><span>danışmanlık agent'ı</span></div>`);
}

// --- Transkript ---

function showTranscript() {
  const loaded = state.transcript !== null;
  $("transcript-badge").hidden = !loaded;
  $("transcript-hint").hidden = loaded;
  $("transcript-pick").textContent = loaded ? "Başka transkript yükle" : "Transkript yükle";
  if (loaded) $("transcript-label").textContent = `Transkript yüklü (${state.transcript.dersler.length} ders)`;
}

// PDF istek gövdesinin kendisi olarak gider (multipart değil): sunucu dosyayı yalnız bellekte okur.
async function uploadTranscript(file) {
  let res;
  try {
    res = await fetch("/transcript", { method: "POST", headers: { "Content-Type": "application/pdf" }, body: file });
  } catch {
    throw new Error("Sunucuya ulaşılamadı. Sunucu çalışıyor mu?");
  }
  const data = await res.json().catch(() => null);
  if (!res.ok) throw new Error((data && typeof data.detail === "string" && data.detail) ||
    "Lütfen UBYS'den tarihsel transkripti yükleyin.");
  return data;
}

async function onTranscriptFile(file) {
  if (!file) return;
  const note = addEl("typing", "Transkript okunuyor");
  try {
    const body = await uploadTranscript(file);
    state.transcript = body.transcript;
    state.confirmations = {}; // teyitler yüklenen transkripte aittir
    note.remove();
    addEl("msg-info", `Transkript yüklendi (${body.course_count} ders). Sorularınız artık transkriptle cevaplanacak.`);
  } catch (e) {
    note.remove();
    addEl("msg-error", escapeHtml(`Transkript yüklenemedi: ${e.message}`));
  }
  showTranscript();
}

function removeTranscript() {
  state.transcript = null;
  state.confirmations = {};
  showTranscript();
  addEl("msg-info", "Transkript kaldırıldı. Sorular yalnız mevzuattan cevaplanacak.");
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
    // Her soru danışmanlık agent'ına gider; transkript yoksa null gider ve agent bunu "yüklü değil" olarak görür.
    // Ders programı, açılan seçmeliler ve mevzuat tool'ları transkriptsiz de çalışır. Klasik /ask (addBotMessage)
    // deney ve geriye uyumluluk için sunucuda duruyor, arayüz kullanmıyor.
    const body = await api("/agent/ask", {
      method: "POST",
      body: { question, history: state.history, session_id: state.sessionId,
              transcript: state.transcript, confirmations: state.confirmations },
    });
    state.confirmations = body.confirmations;
    typing.remove();
    addAgentMessage(body);
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

// Yeni sohbet geçmişi siler; yüklü transkript ve teyitler kalır (terminaldeki chat.py'de de böyle).
function newChat() {
  state.history = [];
  state.sessionId = crypto.randomUUID();
  els.inner.querySelectorAll(".msg-user, .msg-bot, .msg-error, .msg-info").forEach((el) => el.remove());
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
  const fileInput = $("transcript-file");
  $("transcript-pick").addEventListener("click", () => fileInput.click());
  fileInput.addEventListener("change", () => {
    onTranscriptFile(fileInput.files[0]);
    fileInput.value = ""; // aynı dosya tekrar seçilirse de change olayı gelsin
  });
  $("transcript-remove").addEventListener("click", removeTranscript);
  els.empty.querySelectorAll(".examples button").forEach((b) =>
    b.addEventListener("click", () => ask(b.textContent)));
  els.input.focus();
});
