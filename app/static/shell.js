// Ortak kabuk: tema, kenar çubuğu (aç/kapa, gezinme sayıları, durum satırı), "Son sohbetler" ve görünüm yönlendirmesi.
// Sohbet, Belgeler ve Araçlar TEK dokümandır (index.html): bağlantılara tıklamak history.pushState ile adresi değiştirir
// ve görünümü açar, sayfa yeniden yüklenmez; böylece yalnız bellekte duran transkript ve sohbet durumu korunur.
// Gizlenen görünüm DOM'da kalır. /admin ayrı sayfadır (yeni sekmede açılır).
// Her sayfa bunu <head>'de senkron yükler (tema sayfa çizilmeden uygulansın, yanıp sönmesin).
// Sohbet saklama yalnız bu tarayıcının localStorage'ındadır; sunucuya hiçbir sohbet verisi gönderilmez.

const THEME_KEY = "danisman.tema";

// localStorage özel pencerede, site verisi kapalıyken ya da doluyken hata fırlatabilir: her erişim try/catch içinde,
// sayfa depo olmadan da çalışır.
function storageGet(key) {
  try { return localStorage.getItem(key); } catch { return null; }
}
function storageSet(key, value) {
  try { localStorage.setItem(key, value); return true; } catch { return false; }
}

(function applyTheme() {
  const saved = storageGet(THEME_KEY);
  const dark = saved ? saved === "dark" : matchMedia("(prefers-color-scheme: dark)").matches;
  document.documentElement.dataset.theme = dark ? "dark" : "light";
})();

const Shell = {
  CHATS_KEY: "danisman.chats.v1",
  MAX_CHATS: 50,
  chats: [],       // en yeni başta; her biri {id, title, ts, sid, transcript, msgs} (bkz. chat.js)
  currentId: null, // açık sohbetin id'si (sohbet sayfasında)
  // chat.js {open(id), reset(), send(text)} verir. Admin sayfasında null kalır; orada sohbet açmak
  // sohbet sayfasına gitmek demektir (/?sohbet=<id>).
  chatPage: null,

  // --- Görünümler ---
  VIEWS: {
    sohbet: { path: "/", title: "" },   // başlığı açık sohbetin adı (chatTitle)
    belgeler: { path: "/belgeler", title: "Belgeler" },
    araclar: { path: "/araclar", title: "Araçlar" },
  },
  view: null,
  chatTitle: "",
  chatScroll: 0,
  hooks: {},       // görünüm adı -> [{fn, once, done}]

  // Görünüm gösterilince çalışacak iş: once=true ise yalnız ilk gösterimde (belgeler/araçlar verisi böyle lazy yüklenir).
  onShow(view, fn, once = false) {
    (this.hooks[view] ||= []).push({ fn, once, done: false });
  },

  viewFromPath(path) {
    return Object.keys(this.VIEWS).find((v) => this.VIEWS[v].path === path.replace(/\/+$/, "") || (path === "/" && v === "sohbet")) || "sohbet";
  },

  setChatTitle(title) {
    this.chatTitle = title || "";
    if (this.view === "sohbet") this.paintTitle();
  },

  paintTitle() {
    const v = this.VIEWS[this.view];
    document.getElementById("page-title").textContent = this.view === "sohbet" ? (this.chatTitle || "Yeni sohbet") : v.title;
    document.title = this.view === "sohbet" ? "Danışman Asistanı" : `${v.title} · Danışman Asistanı`;
  },

  show(view, push = false) {
    if (!this.VIEWS[view]) view = "sohbet";
    const scroll = document.getElementById("scroll");
    if (this.view === "sohbet" && scroll) this.chatScroll = scroll.scrollTop; // gizlenince kaydırma konumu kaybolmasın
    const changed = this.view !== view;
    this.view = view;
    document.getElementById("app").dataset.view = view;
    document.querySelectorAll(".nav a[data-nav]").forEach((a) => {
      const on = a.dataset.nav === view;
      a.classList.toggle("active", on);
      if (on) a.setAttribute("aria-current", "page"); else a.removeAttribute("aria-current");
    });
    this.paintTitle();
    if (push && changed) history.pushState({ view }, "", this.VIEWS[view].path);
    if (view === "sohbet" && scroll) scroll.scrollTop = this.chatScroll;
    for (const h of this.hooks[view] || []) {
      if (h.once && h.done) continue;
      h.done = true;
      h.fn();
    }
    if (matchMedia("(max-width:860px)").matches) this.setCollapsed(true);
  },

  // Başka görünümden (Belgeler, Araçlar) soru gönderir: sayfa yenilenmeden sohbet görünümüne geçer.
  ask(text) {
    this.show("sohbet", true);
    if (this.chatPage) this.chatPage.send(text);
  },

  // --- Sohbet deposu ---

  loadChats() {
    try {
      const data = JSON.parse(storageGet(this.CHATS_KEY) || "[]");
      this.chats = Array.isArray(data) ? data.filter((c) => c && typeof c.id === "string" && Array.isArray(c.msgs)) : [];
    } catch {
      this.chats = [];
    }
  },

  // Transkript içeriği ve ondan çıkan veri buraya hiç gelmez (chat.js yalnız "transkriptle yapıldı" bayrağını ve
  // transkriptten bağımsız cevapları verir). Depo dolarsa en eski sohbet atılıp yeniden denenir.
  saveChats() {
    this.chats = this.chats.slice(0, this.MAX_CHATS);
    while (!storageSet(this.CHATS_KEY, JSON.stringify(this.chats)) && this.chats.length > 1) this.chats.pop();
  },

  deleteChat(id) {
    this.chats = this.chats.filter((c) => c.id !== id);
    this.saveChats();
    if (this.currentId === id && this.chatPage) this.newChat();
    else this.renderRecents();
  },

  clearChats() {
    if (!this.chats.length || !confirm("Bu tarayıcıda saklanan bütün sohbetler silinsin mi?")) return;
    this.chats = [];
    this.saveChats();
    if (this.chatPage) this.newChat();
    else this.renderRecents();
  },

  renderRecents() {
    const term = document.getElementById("side-search").value.trim().toLocaleLowerCase("tr");
    const list = this.chats.filter((c) => c.title.toLocaleLowerCase("tr").includes(term));
    document.getElementById("recents").innerHTML = list.length
      ? list.map((c) => `<div class="rc${c.id === this.currentId ? " on" : ""}" data-id="${escapeHtml(c.id)}">` +
          `<button class="open" type="button" title="${escapeHtml(c.title)}">${escapeHtml(c.title)}</button>` +
          `<button class="del" type="button" title="Sohbeti sil" aria-label="Sohbeti sil"><svg class="ico"><use href="#i-x"/></svg></button></div>`).join("")
      : `<div class="none">${this.chats.length ? "Sonuç yok" : "Henüz sohbet yok"}</div>`;
  },

  openChat(id) {
    if (!this.chatPage) { location.href = `/?sohbet=${encodeURIComponent(id)}`; return; }
    this.show("sohbet", true);
    this.chatPage.open(id);
  },

  newChat() {
    if (!this.chatPage) { location.href = "/"; return; }
    this.show("sohbet", true);
    this.chatPage.reset();
  },

  // --- Kenar çubuğu ---

  setCollapsed(on) { document.getElementById("app").classList.toggle("collapsed", on); },

  init() {
    const $ = (id) => document.getElementById(id);
    // Dar ekranda kenar çubuğu kapalı başlar (üstüne biner); geniş ekranda açık.
    this.setCollapsed(matchMedia("(max-width:860px)").matches);
    $("close-side").addEventListener("click", () => this.setCollapsed(true));
    $("open-side").addEventListener("click", () => this.setCollapsed(false));
    $("scrim").addEventListener("click", () => this.setCollapsed(true));
    $("theme").addEventListener("click", () => {
      const next = document.documentElement.dataset.theme === "dark" ? "light" : "dark";
      document.documentElement.dataset.theme = next;
      storageSet(THEME_KEY, next);
    });

    this.loadChats();
    this.renderRecents();
    $("recents").addEventListener("click", (e) => {
      const row = e.target.closest(".rc");
      if (!row) return;
      if (e.target.closest(".del")) this.deleteChat(row.dataset.id);
      else this.openChat(row.dataset.id);
    });
    $("clear-all").addEventListener("click", () => this.clearChats());
    $("side-search").addEventListener("input", () => this.renderRecents());
    $("new-chat").addEventListener("click", () => this.newChat());
    // Sohbet / Belgeler / Araçlar bağlantıları sayfa yüklemeden görünüm değiştirir (değiştirici tuşlu tıklamalar
    // ve /admin normal bağlantı olarak kalır). Admin sayfasında görünümler yoktur, bağlantılar normal çalışır.
    if (document.getElementById("view-chat")) {
      document.querySelector(".nav").addEventListener("click", (e) => {
        const a = e.target.closest("a[data-nav]");
        if (!a || !this.VIEWS[a.dataset.nav] || e.metaKey || e.ctrlKey || e.shiftKey || e.button) return;
        e.preventDefault();
        this.show(a.dataset.nav, true);
      });
      addEventListener("popstate", () => this.show(this.viewFromPath(location.pathname)));
      const first = this.viewFromPath(location.pathname);
      history.replaceState({ view: first }, "");
      this.show(first);
    }

    if (typeof TOOL_CARDS !== "undefined") $("tool-count").textContent = TOOL_CARDS.length;
    this.loadStatus();
  },

  // Sunucu durumu: alt satırda "deney · commit · belge sayısı"; belge sayısı gezinmedeki sayıya da yazılır.
  async loadStatus() {
    try {
      const h = await api("/health");
      document.getElementById("status").textContent = `${h.experiment} · ${h.commit} · ${h.documents} belge`;
      document.getElementById("doc-count").textContent = h.documents;
    } catch {
      document.getElementById("status").textContent = "sunucuya ulaşılamıyor";
    }
  },
};

document.addEventListener("DOMContentLoaded", () => Shell.init());
