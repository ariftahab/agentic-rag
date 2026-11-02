// Belgeler sayfası: gerçek belge listesi API'den (GET /documents) gelir; başlık, açıklama, konu grubu ve kapsam
// rozeti /static/belge_basliklari.json'dan dosya adına göre eklenir. Eşlemesi olmayan belge dosya adından türetilen
// başlıkla "Diğer" grubuna düşer. Bölüm verileri (plan, program, personel, duyuru) belge olarak aranmadığı için
// madde listesi olmayan açıklama kartlarıdır; içerikleri API'den doğrulanmaz.

const KIND = { mevzuat: "Mevzuat", takvim: "Takvim", veri: "Bölüm verisi" };
const SCOPE = { uni: "Üniversite geneli", fac: "Mühendislik Fakültesi", dept: "Bölüm", grad: "Lisansüstü" };
const OTHER_GROUP = "Diğer";

let items = [];        // [{baslik, aciklama, grup, tur, kapsam, source?, apiTitle?, chunks?}]
let groupOrder = [];
let kindFilter = "all";

async function loadItems() {
  const [docs, titles] = await Promise.all([api("/documents"), loadDocTitles()]);
  const known = titles.belgeler || {};
  items = docs.map((d) => {
    const info = known[d.source] || {};
    return { baslik: info.baslik || titleFromFilename(d.source), aciklama: info.aciklama || "",
             grup: info.grup || OTHER_GROUP, tur: info.tur || d.kind || "mevzuat", kapsam: info.kapsam || null,
             source: d.source, apiTitle: d.title, chunks: d.chunk_count };
  });
  const data = titles.bolum_verileri;
  if (data) {
    for (const k of data.kartlar) items.push({ baslik: k.baslik, aciklama: k.aciklama, grup: data.grup, tur: "veri", kapsam: k.kapsam });
    groupOrder = [...(titles.gruplar || []).filter((g) => g !== OTHER_GROUP), data.grup, OTHER_GROUP];
  } else {
    groupOrder = titles.gruplar || [OTHER_GROUP];
  }
}

function card(d, i) {
  const tags = `<span class="tag">${KIND[d.tur] || escapeHtml(d.tur)}</span>` +
    (d.kapsam ? `<span class="tag">${SCOPE[d.kapsam]}</span>` : "") +
    (d.chunks ? `<span class="tag">${d.chunks} chunk</span>` : "");
  const open = d.source ? `<button class="try" type="button" data-act="open" data-i="${i}">Maddeleri aç</button>` : "";
  return `<div class="card"><div class="card-top"><div class="doc-ico"><svg class="ico"><use href="#i-${d.tur === "veri" ? "chart" : "doc"}"/></svg></div>` +
    `<div><h3>${escapeHtml(d.baslik)}</h3></div></div>` +
    `<p>${escapeHtml(d.aciklama)}</p>` +
    `<div class="row">${tags}${open}<button class="try" type="button" data-act="ask" data-i="${i}">Bu belgeyle ilgili sor →</button></div></div>`;
}

function render() {
  const term = norm($("doc-q").value);
  const shown = items.map((d, i) => [d, i])
    .filter(([d]) => (kindFilter === "all" || d.tur === kindFilter) && (!term || norm(`${d.baslik} ${d.aciklama}`).includes(term)));
  const groups = [...new Set([...groupOrder, ...shown.map(([d]) => d.grup)])];
  const html = groups.map((g) => {
    const rows = shown.filter(([d]) => d.grup === g);
    if (!rows.length) return "";
    return `<div class="group"><h2>${escapeHtml(g)}</h2><small>${rows.length} belge</small></div>` +
      `<div class="grid">${rows.map(([d, i]) => card(d, i)).join("")}</div>`;
  }).join("");
  $("docs").innerHTML = html || `<p class="lead">Aramanıza uyan belge yok.</p>`;
}

// Görünüm ilk kez gösterilince bir kez kurulur ve veri yüklenir (shell.js onShow, once).
Shell.onShow("belgeler", async () => {
  $("doc-seg").innerHTML = [["all", "Tümü"], ["mevzuat", "Mevzuat"], ["takvim", "Takvim"], ["veri", "Bölüm verisi"]]
    .map(([k, l]) => `<button type="button" data-k="${k}" class="${k === "all" ? "on" : ""}">${l}</button>`).join("");
  $("doc-seg").addEventListener("click", (e) => {
    const k = e.target.dataset.k;
    if (!k) return;
    kindFilter = k;
    document.querySelectorAll("#doc-seg button").forEach((b) => b.classList.toggle("on", b.dataset.k === k));
    render();
  });
  $("doc-q").addEventListener("input", render);
  $("docs").addEventListener("click", (e) => {
    const btn = e.target.closest(".try");
    if (!btn) return;
    const d = items[Number(btn.dataset.i)];
    if (btn.dataset.act === "open") openDocument(d.source, d.apiTitle, d.baslik);
    else Shell.ask(`“${d.baslik}” hakkında bilgi verir misin?`);
  });
  try {
    await loadItems();
    render();
  } catch (e) {
    $("docs").innerHTML = `<div class="msg-error">${escapeHtml(e.message)}</div>`;
  }
}, true);
