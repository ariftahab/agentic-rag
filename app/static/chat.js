// Sohbet sayfası. Durum (geçmiş, oturum) tarayıcıda tutulur; API stateless.
// Kenar çubuğu, tema ve "Son sohbetler" listesi shell.js'te; araç adları arac_bilgisi.js'te.

const state = {
  history: [],                     // API'ye giden sade geçmiş: [{role, content}], kaynak listesi yok
  sessionId: crypto.randomUUID(),  // log'da bu sohbetin sorularını gruplar
  busy: false,
  chat: null,                      // açık sohbet (Shell.chats içindeki nesne); ilk soruyla oluşur
  // Transkript ve eşdeğerlik teyitleri yalnız bu değişkende durur (localStorage/sessionStorage yok): sayfa
  // yenilenince ya da kapanınca kaybolur. Sunucu da saklamaz; her /agent/ask isteğine eklenir.
  transcript: null,                // POST /transcript'in döndürdüğü kişisel bilgisiz JSON
  confirmations: {},               // {plan dışı ders kodu: plan ders kodu | null}; agent günceller, biz geri göndeririz
  cost: { usd: 0, tokens: 0 },     // açık sohbetin toplamı
  unread: false,                   // başka görünümdeyken cevap geldi: sohbete dönünce en alta kaydır
};

// Sohbet görünümü gizlendiğinde DOM'da ve bu durumda kalır; cevap arka planda gelip sohbete eklenir.
Shell.onShow("sohbet", () => {
  if (state.unread) { state.unread = false; scrollToBottom(); }
});

const EXAMPLES = [
  "Yatay geçiş başvurusu için not ortalaması şartı nedir?",
  "Bugün hangi dersler var?",
  "Bu dönem hangi seçmeli dersler açılıyor?",
  "Mezun olmak için kaç AKTS eksiğim var?",
];

// --- Token maliyeti ---
// Anthropic fiyat listesi, 2026-10. Model: claude-haiku-5-5, ABD doları / 1M token, ≤100K token'lık prompt'lar.
// Prompt 100K token'ı aşarsa fiyat değişir; bu hesap o farkı göz ardı eder. Cache okuma input'un 0,1 katı, cache
// yazma 1,25 katı. Sunucu ham token sayılarını (usage) döndürür, dolar burada hesaplanır. Anthropic'te "input"
// cache'ten okunan ve cache'e yazılan token'ları içermez; dört kalem toplanır, çift sayım olmaz.
const PRICING = {
  model: "claude-haiku-5-5",
  perMillion: { input: 0.10, output: 0.50, cache_read: 0.01, cache_write: 0.125 },
};

// usage: {input, output, reasoning?, cache_read?, cache_write?}. reasoning output'un içinde sayıldığı için eklenmez.
function usageCost(usage) {
  const n = (k) => Number(usage[k]) || 0;
  const tokens = n("input") + n("output") + n("cache_read") + n("cache_write");
  const usd = Object.entries(PRICING.perMillion).reduce((sum, [k, price]) => sum + n(k) * price, 0) / 1e6;
  return { usd, tokens };
}

function formatTokens(t) {
  return t >= 1000 ? `${(t / 1000).toFixed(1)}k token` : `${t} token`;
}

function formatUsd(usd) {
  return `$${usd.toFixed(4)}`;
}

function showTotalCost() {
  const el = $("cost");
  el.hidden = false;
  el.textContent = `Bu sohbet: ${formatUsd(state.cost.usd)} · ${formatTokens(state.cost.tokens)}`;
  el.title = `${PRICING.model} fiyatlarıyla tahmini maliyet (yalnız LLM token'ları)`;
}

function scrollToBottom() {
  $("scroll").scrollTop = $("scroll").scrollHeight;
}

// --- Görünüm ---

// Boş durumda yazı kutusu karşılamanın ortasında (slot), sohbet başlayınca altta (dock) durur.
function setEmpty(empty) {
  $("view-chat").classList.toggle("is-empty", empty);
  ($("view-chat").classList.contains("is-empty") ? $("slot") : $("dock")).prepend($("composer"));
}

function setTitle(title) {
  Shell.setChatTitle(title); // üst çubuk başlığını yalnız sohbet görünümü açıkken değiştirir
}

function updateSend() {
  $("send").disabled = state.busy || !$("q").value.trim();
}

// Metin kutusu yazdıkça uzar (en fazla CSS'teki max-height kadar).
function autoGrow() {
  const q = $("q");
  q.style.height = "auto";
  q.style.height = `${Math.min(q.scrollHeight, 180)}px`;
}

function addMsg(cls, html) {
  const div = document.createElement("div");
  div.className = cls;
  div.innerHTML = html;
  $("msgs").appendChild(div);
  return div;
}

function appendUser(text) {
  const div = addMsg("msg user", `<div class="bubble"></div>`);
  div.firstChild.textContent = text; // textContent: kullanıcının yazdığı HTML olarak yorumlanmaz
}

function appendError(text) {
  addMsg("msg", `<div class="msg-error">${escapeHtml(text)}</div>`);
}

function appendTyping() {
  return addMsg("msg bot", `<div class="typing" role="status"><i></i><i></i><i></i><span>Cevap hazırlanıyor</span></div>`);
}

// --- Cevap ---

// Kullanılan araçlar: kod adı değil arac_bilgisi.js'teki insan okunur ad (toolLabel). Satırın altında aracın aldığı
// argümanlar (arama sorgusu, ders adı vb.) küçük gri yazılır; kapalıyken yalnız sayı ve adlar görünür.

function stepDetail(call) {
  const vals = Object.values(call.args || {})
    .filter((v) => v !== "" && v !== null && v !== undefined && !(Array.isArray(v) && !v.length))
    .map((v) => (typeof v === "string" ? v : JSON.stringify(v)));
  const text = vals.join(" · ");
  return text.length > 160 ? `${text.slice(0, 157)}…` : text;
}

function toolsHtml(calls) {
  if (!calls.length) return `<div class="used-none">Araç kullanılmadı</div>`;
  const rows = calls.map((c) => {
    const card = toolCard(c.name);
    const detail = stepDetail(c);
    return `<div class="step"><div><b>${escapeHtml(toolLabel(c.name))}</b>` +
      (detail ? `<span class="detail">${escapeHtml(detail)}</span>` : "") + `</div>` +
      (card ? `<span class="tag ${card.rozet === "code" ? "code" : ""}">${TOOL_KINDS[card.rozet].done}</span>` : "") + `</div>`;
  }).join("");
  return `<details class="used"><summary><svg class="ico"><use href="#i-chev"/></svg><span>Kullanılan araçlar</span>` +
    `<span class="tag">${calls.length}</span><span class="names">${escapeHtml(calls.map((c) => toolLabel(c.name)).join(" · "))}</span></summary>${rows}</details>`;
}

// Agent cevabın sonuna bazen "Kaynak: …" / "Kaynaklar: …" paragrafı yazıyor; aynı bilgi altındaki Kaynaklar bölümünde
// zaten var. Yalnız en sondaki böyle paragraflar gösterilmez (metin içindeki [..] atıflar kalır).
function stripSourceTrailer(text) {
  const parts = text.trimEnd().split(/\n\s*\n/);
  while (parts.length > 1 && /^\s*(\*\*)?Kaynak(lar)?(\*\*)?\s*:/i.test(parts[parts.length - 1])) parts.pop();
  return parts.join("\n\n");
}

const THUMBS = `<svg class="ico"><use href="#i-up2"/></svg>`;
const THUMBS_DOWN = `<svg class="ico"><use href="#i-down2"/></svg>`;

function metaHtml(a) {
  const cost = a.usage ? usageCost(a.usage) : null; // sunucu usage döndürmezse (eski sunucu) satır gösterilmez
  return `<span class="meta"><span>${(a.latency_ms / 1000).toFixed(1)} sn</span>` +
    (cost ? `<span class="cost" title="${PRICING.model} fiyatlarıyla tahmini maliyet">${formatUsd(cost.usd)} · ${formatTokens(cost.tokens)}</span>` : "") + `</span>`;
}

// m: {r: "b", a: {id, answer, tool_calls, sources, latency_ms, usage}, fb?}. Eski sürümde transkriptli cevaplar
// {r: "b", gizli: true, a: {latency_ms, usage}} olarak (metinsiz) saklanıyordu; o kayıtlar için not gösterilir.
function appendBot(m) {
  const a = m.a;
  if (m.gizli) {
    addMsg("msg bot", `<div class="restore">Bu cevabın metni eski bir sürümde saklanmamıştı. ` +
      `Görmek için transkripti yükleyip soruyu yeniden sorun.</div><div class="rate">${metaHtml(a)}</div>`);
    return;
  }
  // Anlık araç: kodu o an bir LLM yazdı, test edilmedi; kullanıcı bunu bilsin.
  const live = a.tool_calls.some((c) => isLiveTool(c.name))
    ? `<div class="live-warn"><span class="tag warn">Anlık üretilmiş hesap, doğrulanmamış</span></div>` : "";
  const chips = a.sources.length
    ? `<div class="sources"><span class="sources-head">İncelenen belgeler</span>${a.sources.map(sourceChip).join("")}</div>` : "";
  const div = addMsg("msg bot", `${toolsHtml(a.tool_calls)}${live}
    <div class="answer">${renderAnswer(stripSourceTrailer(a.answer), a.sources)}</div>${chips}
    <div class="rate">
      <button type="button" data-rating="1" title="Faydalı" aria-label="Faydalı" class="${m.fb === 1 ? "on" : ""}">${THUMBS}</button>
      <button type="button" data-rating="-1" title="Faydalı değil" aria-label="Faydalı değil" class="${m.fb === -1 ? "on" : ""}">${THUMBS_DOWN}</button>
      ${metaHtml(a)}
    </div>`);
  div.dataset.aid = a.id;
}

async function sendFeedback(btn, msgEl) {
  const rating = Number(btn.dataset.rating);
  try {
    await api("/feedback", { method: "POST", body: { id: msgEl.dataset.aid, rating } });
    msgEl.querySelectorAll(".rate button").forEach((b) => b.classList.toggle("on", b === btn));
    const stored = state.chat && state.chat.msgs.find((m) => m.r === "b" && m.a && m.a.id === msgEl.dataset.aid);
    if (stored) { stored.fb = rating; Shell.saveChats(); }
  } catch (e) {
    alert(`Geri bildirim kaydedilemedi: ${e.message}`);
  }
}

// --- Transkript ---

function showTranscript() {
  const loaded = state.transcript !== null;
  $("tr-btn").classList.toggle("on", loaded);
  $("tr-label").textContent = loaded ? `Transkript yüklü (${state.transcript.dersler.length} ders)` : "Transkript yükle";
  $("tr-btn").setAttribute("aria-label", loaded ? "Başka transkript yükle" : "Transkript yükle");
  $("tr-remove").hidden = !loaded;
}

function trNote(text, isError = false) {
  const el = $("tr-note");
  el.hidden = !text;
  el.textContent = text || "";
  el.classList.toggle("err", isError);
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
  trNote("Transkript okunuyor…");
  try {
    const body = await uploadTranscript(file);
    state.transcript = body.transcript;
    state.confirmations = {}; // teyitler yüklenen transkripte aittir
    trNote(`Transkript yüklendi (${body.course_count} ders). Sorularınız artık transkriptle cevaplanacak.`);
  } catch (e) {
    trNote(`Transkript yüklenemedi: ${e.message}`, true);
  }
  showTranscript();
}

function removeTranscript() {
  state.transcript = null;
  state.confirmations = {};
  showTranscript();
  trNote("Transkript kaldırıldı. Sorular yalnız mevzuattan cevaplanacak.");
}

// --- Sohbet: soru sorma, saklama ---

// Yalnız localStorage (bu tarayıcı); sunucuya hiçbir sohbet verisi gitmez. Transkriptin KENDİSİ (ders listesi, notlar)
// hiçbir zaman yazılmaz, yalnız "transkriptle yapıldı" bayrağı konur. Transkriptli sorunun cevabı ise (kullanıcı kararı,
// 2026-11-02: sohbete dönünce cevabı görmek istiyor) soru ve cevap metniyle birlikte saklanır; cevap transkriptten
// türetilmiş hesapları içerebilir. Ortak bilgisayar riski için "Tümünü sil" ve tekil silme vardır.
function storedAnswer(body) {
  return { r: "b", a: { id: body.id, answer: body.answer, tool_calls: body.tool_calls, sources: body.sources,
                        latency_ms: body.latency_ms, usage: body.usage || null } };
}

async function send(text) {
  const question = (typeof text === "string" ? text : $("q").value).trim();
  if (!question || state.busy) return;
  state.busy = true;
  $("q").value = "";
  autoGrow();
  updateSend();
  trNote("");
  setEmpty(false);

  if (!state.chat) {
    state.chat = { id: Date.now().toString(36) + Math.random().toString(36).slice(2, 6), ts: Date.now(),
                   title: question.length > 48 ? `${question.slice(0, 48)}…` : question,
                   sid: state.sessionId, transcript: false, msgs: [] };
    Shell.chats.unshift(state.chat);
    Shell.currentId = state.chat.id;
    setTitle(state.chat.title);
  }
  const chat = state.chat;
  const withTranscript = state.transcript !== null;
  if (withTranscript) chat.transcript = true; // yalnız bayrak; transkriptin kendisi saklanmaz
  chat.msgs.push({ r: "u", t: question });
  Shell.saveChats();
  Shell.renderRecents();
  appendUser(question);
  const typing = appendTyping();
  scrollToBottom();

  try {
    // Her soru danışmanlık agent'ına gider; transkript yoksa null gider ve agent bunu "yüklü değil" olarak görür.
    // Ders programı, açılan seçmeliler ve mevzuat araçları transkriptsiz de çalışır. Klasik /ask deney ve geriye
    // uyumluluk için sunucuda duruyor, arayüz kullanmıyor.
    const body = await api("/agent/ask", {
      method: "POST",
      body: { question, history: state.history, session_id: chat.sid,
              transcript: state.transcript, confirmations: state.confirmations },
    });
    chat.msgs.push(storedAnswer(body));
    Shell.saveChats();
    if (state.chat === chat) { // soru beklerken başka sohbete geçildiyse cevap yalnız kaydedilir
      state.confirmations = body.confirmations;
      if (body.usage) {
        const c = usageCost(body.usage);
        state.cost.usd += c.usd;
        state.cost.tokens += c.tokens;
      }
      showTotalCost();
      typing.remove();
      appendBot({ r: "b", a: body });
      state.history.push({ role: "user", content: question }, { role: "assistant", content: body.answer });
    }
  } catch (e) {
    // Hata ekranda görünür ama geçmişe eklenmez: sonraki soruların rewrite'ını bozmasın.
    chat.msgs.push({ r: "e", t: e.message });
    Shell.saveChats();
    if (state.chat === chat) { typing.remove(); appendError(e.message); }
  } finally {
    state.busy = false;
    updateSend();
    if (state.chat === chat) {
      if (Shell.view === "sohbet") { scrollToBottom(); $("q").focus(); } else state.unread = true; // arka planda geldi
    }
  }
}

// Saklanmış mesajlardan API geçmişi: yalnız cevabı saklanmış soru-cevap çiftleri (transkriptli ve hatalı olanlar yok).
function rebuildHistory(chat) {
  const history = [];
  for (let i = 0; i < chat.msgs.length - 1; i++) {
    const u = chat.msgs[i];
    const b = chat.msgs[i + 1];
    if (u.r === "u" && b.r === "b" && b.a && typeof b.a.answer === "string") {
      history.push({ role: "user", content: u.t }, { role: "assistant", content: b.a.answer });
    }
  }
  return history;
}

function sumCost(chat) {
  return chat.msgs.reduce((sum, m) => {
    if (m.r !== "b" || !m.a || !m.a.usage) return sum;
    const c = usageCost(m.a.usage);
    return { usd: sum.usd + c.usd, tokens: sum.tokens + c.tokens };
  }, { usd: 0, tokens: 0 });
}

function openChat(id) {
  const chat = Shell.chats.find((c) => c.id === id);
  if (!chat) return;
  state.chat = chat;
  Shell.currentId = id;
  chat.sid ||= crypto.randomUUID();
  state.sessionId = chat.sid;
  state.history = rebuildHistory(chat);
  state.cost = sumCost(chat);
  $("msgs").innerHTML = "";
  // Transkript hiçbir yerde saklanmadığı için eski bir transkriptli sohbette yeni sorular için yeniden yüklenmeli.
  if (chat.transcript && state.transcript === null) {
    addMsg("restore", "Bu sohbet bir transkriptle yapılmıştı. Transkript saklanmadığı için transkriptli yeni sorular " +
      "için transkripti yeniden yükleyin.");
  }
  for (const m of chat.msgs) {
    if (m.r === "u") appendUser(m.t);
    else if (m.r === "b") appendBot(m);
    else if (m.r === "e") appendError(m.t);
  }
  trNote("");
  setEmpty(false);
  setTitle(chat.title);
  showTotalCost();
  Shell.renderRecents();
  panel.close();
  if (matchMedia("(max-width:860px)").matches) Shell.setCollapsed(true);
  scrollToBottom();
}

// Yeni sohbet geçmişi siler; yüklü transkript ve teyitler kalır (terminaldeki chat.py'de de böyle).
function newChat() {
  state.chat = null;
  Shell.currentId = null;
  state.history = [];
  state.sessionId = crypto.randomUUID();
  state.cost = { usd: 0, tokens: 0 };
  $("msgs").innerHTML = "";
  trNote("");
  setEmpty(true);
  setTitle("");
  showTotalCost();
  Shell.renderRecents();
  panel.close();
  if (matchMedia("(max-width:860px)").matches) Shell.setCollapsed(true);
  $("q").focus();
}

document.addEventListener("DOMContentLoaded", () => {
  Shell.chatPage = { open: openChat, reset: newChat, send };
  setEmpty(true); // HTML'de yazı kutusu altta (dock) durur; karşılamada ortaya (slot) alınır

  const q = $("q");
  $("composer").addEventListener("submit", (e) => { e.preventDefault(); send(); });
  q.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey && !e.isComposing) { e.preventDefault(); $("composer").requestSubmit(); }
  });
  q.addEventListener("input", () => { autoGrow(); updateSend(); });

  $("tr-btn").addEventListener("click", () => $("tr-file").click());
  $("tr-file").addEventListener("change", (e) => {
    onTranscriptFile(e.target.files[0]);
    e.target.value = ""; // aynı dosya tekrar seçilirse de change olayı gelsin
  });
  $("tr-remove").addEventListener("click", removeTranscript);

  $("chips").innerHTML = EXAMPLES.map((t) => `<button class="chip" type="button">${escapeHtml(t)}</button>`).join("");
  $("chips").addEventListener("click", (e) => { if (e.target.classList.contains("chip")) send(e.target.textContent); });

  $("msgs").addEventListener("click", (e) => {
    const btn = e.target.closest(".rate button[data-rating]");
    if (btn) sendFeedback(btn, btn.closest(".msg"));
  });

  showTotalCost();
  showTranscript();

  // /?sohbet=<id>: admin sayfasındaki sohbet listesinden gelen istek. Adres hemen temizlenir.
  const params = new URLSearchParams(location.search);
  if (params.has("sohbet")) {
    history.replaceState({ view: "sohbet" }, "", "/");
    openChat(params.get("sohbet"));
  } else if (Shell.view === "sohbet") q.focus();
});
