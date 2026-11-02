// Araçlar sayfası: kartlar arac_bilgisi.js'teki TOOL_CARDS'tan çizilir (sohbetteki "Kullanılan araçlar" ile aynı adlar).
// "Dene" örnek soruyu (sayfa yenilemeden) sohbet görünümüne gönderir.

function toolCardHtml(t) {
  const kind = TOOL_KINDS[t.rozet];
  return `<div class="card"><div class="card-top"><div class="doc-ico"><svg class="ico"><use href="#i-wrench"/></svg></div><div><h3>${escapeHtml(t.ad)}</h3></div></div>` +
    `<p>${escapeHtml(t.aciklama)}</p>` +
    `<div class="row"><span class="tag ${t.rozet === "code" ? "code" : ""}">${kind.label}</span>` +
    (t.transkript ? `<span class="tag">Transkript gerekir</span>` : "") +
    `<button class="try" type="button" data-q="${escapeHtml(t.ornek)}">Dene →</button></div></div>`;
}

Shell.onShow("araclar", () => {
  const tools = $("tools");
  tools.innerHTML = TOOL_GROUPS.map((g) => {
    const cards = TOOL_CARDS.filter((t) => t.grup === g);
    return cards.length ? `<div class="group"><h2>${escapeHtml(g)}</h2></div><div class="grid">${cards.map(toolCardHtml).join("")}</div>` : "";
  }).join("");
  tools.addEventListener("click", (e) => {
    const btn = e.target.closest(".try");
    if (btn) Shell.ask(btn.dataset.q);
  });
}, true);
