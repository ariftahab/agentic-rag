// Değerlendirme sayfası: kayıtları listeler, her cevap için elle etiket formu gösterir.

const LABELS = {
  answer: { dogru: "Doğru", kismen: "Kısmen", yanlis: "Yanlış", cevapsiz: "Cevapsız (doğru)" },
  source: { evet: "Evet", hayir: "Hayır", emin_degil: "Emin değilim" },
  stage: { rewrite: "Rewrite", retrieval: "Retrieval", generation: "Generation", corpus_yok: "Corpus'ta yok" },
};
const NEEDS_STAGE = ["yanlis", "kismen"]; // hata aşaması yalnız bu etiketlerde sorulur

let records = [];

function pct(x) { return x == null ? "—" : `%${Math.round(x * 100)}`; }

async function loadSummary() {
  const s = await api("/admin/summary");
  const l = s.cevap_etiketleri;
  const stages = Object.entries(s.hata_asamalari).map(([k, v]) => `${LABELS.stage[k]} ${v}`).join(" · ") || "—";
  const srcTotal = Object.values(s.kaynak_dogru).reduce((a, b) => a + b, 0);
  const stats = [
    [s.toplam, `soru (${s.hatali_istek} hatalı istek)`],
    [`${s.degerlendirilen}`, "değerlendirildi"],
    [pct(s.dogruluk), `doğruluk (doğru ${l.dogru} + cevapsız ${l.cevapsiz})`],
    [`${l.kismen} / ${l.yanlis}`, "kısmen / yanlış"],
    [srcTotal ? pct((s.kaynak_dogru.evet || 0) / srcTotal) : "—", "doğru kaynak geldi"],
    [`${s.geri_bildirim.olumlu} / ${s.geri_bildirim.olumsuz}`, "👍 / 👎"],
  ];
  document.getElementById("stats").innerHTML =
    stats.map(([v, label]) => `<div class="stat"><div class="v">${v}</div><div class="l">${label}</div></div>`).join("");
  document.getElementById("stages").textContent = `Hata aşamaları: ${stages}`;
}

function choices(name, options, selected, disabled = false) {
  return Object.entries(options).map(([value, text]) => `
    <label class="choice">
      <input type="radio" name="${name}" value="${value}" ${value === selected ? "checked" : ""} ${disabled ? "disabled" : ""}>
      <span>${text}</span>
    </label>`).join("");
}

// Silme iki adımlı: "Sil" → kartın içinde onay. Tarayıcının confirm() penceresi yerine sayfa içi onay
// kullanılıyor (telefonda da aynı görünür, yanlışlıkla iki kez dokunmak tek başına silmez).
function deleteControls() {
  return `<span class="delete-box">
      <button class="link-btn danger" type="button" data-action="ask-delete">Sil</button>
      <span class="delete-confirm" hidden>
        Kalıcı olarak silinsin mi?
        <button class="btn btn-danger" type="button" data-action="delete">Evet, sil</button>
        <button class="btn" type="button" data-action="cancel-delete">Vazgeç</button>
      </span>
    </span>`;
}

function recordHtml(r) {
  const e = r.evaluation;
  const fb = r.feedback === 1 ? "👍" : r.feedback === -1 ? "👎" : "";
  const evalBadge = e
    ? `<span class="badge ${["dogru", "cevapsiz"].includes(e.answer_label) ? "ok" : "bad"}">${LABELS.answer[e.answer_label]}</span>`
    : `<span class="badge">değerlendirilmedi</span>`;
  const sources = r.sources.map((s) =>
    `<button class="source-chip" data-chunk="${s.chunk_id}" title="${escapeHtml(s.label)}">${escapeHtml(s.label)}</button>`
  ).join("");

  if (r.error) {
    return `<div class="record" data-id="${r.id}">
      <div class="q">${escapeHtml(r.question)}</div>
      <div class="msg-error" style="margin-top:8px">${escapeHtml(r.error)}</div>
      <div class="info"><span>${r.time.replace("T", " ")}</span><span>${r.experiment}</span></div>
      <div class="eval-actions" style="margin-top:10px">${deleteControls()}</div>
    </div>`;
  }
  const stageDisabled = !NEEDS_STAGE.includes(e?.answer_label);
  return `<div class="record" data-id="${r.id}">
    <div class="q">${escapeHtml(r.question)} ${fb}</div>
    ${r.rewritten_query ? `<div class="rq">Aranan: ${escapeHtml(r.rewritten_query)}</div>` : ""}
    <div class="answer">${renderAnswer(r.answer, r.sources)}</div>
    <div class="sources">${sources}</div>
    <div class="info">
      ${evalBadge}
      <span>${r.time.replace("T", " ")}</span>
      <span>${r.experiment} · ${r.commit || "?"}</span>
      <span>${r.latency_ms != null ? (r.latency_ms / 1000).toFixed(1) + " sn" : ""}</span>
    </div>
    <form class="eval-form">
      <div class="eval-row"><span>Cevap</span>${choices(`a-${r.id}`, LABELS.answer, e?.answer_label)}</div>
      <div class="eval-row"><span>Doğru kaynak geldi mi?</span>${choices(`s-${r.id}`, LABELS.source, e?.source_ok)}</div>
      <div class="eval-row"><span>Hata nerede?</span>${choices(`h-${r.id}`, LABELS.stage, e?.error_stage, stageDisabled)}</div>
      <textarea name="note" placeholder="Not (isteğe bağlı): ör. doğru madde hangisiydi, cevapta ne eksikti">${escapeHtml(e?.note || "")}</textarea>
      <div class="eval-actions"><button class="btn btn-primary" type="submit">Kaydet</button><span class="saved"></span>${deleteControls()}</div>
    </form>
  </div>`;
}

function render() {
  const onlyPending = document.getElementById("only-pending").checked;
  const rows = onlyPending ? records.filter((r) => !r.evaluation && !r.error) : records;
  document.getElementById("shown").textContent = `${rows.length} kayıt gösteriliyor`;
  document.getElementById("records").innerHTML = rows.length
    ? rows.map(recordHtml).join("")
    : `<p class="lead">Gösterilecek kayıt yok.</p>`;
}

async function reload() {
  records = await api("/admin/records");
  render();
  await loadSummary();
}

document.addEventListener("DOMContentLoaded", async () => {
  const container = document.getElementById("records");

  // Cevap etiketi değişince "Hata nerede?" yalnız yanlış / kısmen için açılır.
  container.addEventListener("change", (e) => {
    if (!e.target.name?.startsWith("a-")) return;
    const form = e.target.closest("form");
    const enable = NEEDS_STAGE.includes(e.target.value);
    form.querySelectorAll('input[name^="h-"]').forEach((i) => { i.disabled = !enable; if (!enable) i.checked = false; });
  });

  container.addEventListener("submit", async (e) => {
    e.preventDefault();
    const form = e.target;
    const id = form.closest(".record").dataset.id;
    const value = (prefix) => form.querySelector(`input[name="${prefix}-${id}"]:checked`)?.value ?? null;
    const body = {
      id,
      answer_label: value("a"),
      source_ok: value("s"),
      error_stage: value("h"),
      note: form.note.value.trim() || null,
    };
    const saved = form.querySelector(".saved");
    if (!body.answer_label || !body.source_ok) {
      saved.style.color = "var(--danger)";
      saved.textContent = "Cevap ve kaynak seçilmeli.";
      return;
    }
    try {
      await api("/admin/evaluations", { method: "POST", body });
      // Kaydı yerinde güncelle; "yalnız değerlendirilmemişler" açıksa kart bir sonraki çizimde listeden çıkar.
      const r = records.find((x) => x.id === id);
      r.evaluation = { ...body, time: new Date().toISOString() };
      saved.style.color = "";
      saved.textContent = "Kaydedildi ✓";
      loadSummary();
    } catch (err) {
      saved.style.color = "var(--danger)";
      saved.textContent = err.message;
    }
  });

  // Silme: soru-cevap ve ona bağlı geri bildirim / değerlendirme log'lardan kalıcı olarak silinir.
  container.addEventListener("click", async (e) => {
    const action = e.target.dataset.action;
    if (!["ask-delete", "cancel-delete", "delete"].includes(action)) return;
    const card = e.target.closest(".record");
    const confirmBox = card.querySelector(".delete-confirm");
    const askButton = card.querySelector('[data-action="ask-delete"]');
    if (action !== "delete") {
      confirmBox.hidden = action === "cancel-delete";
      askButton.hidden = action === "ask-delete";
      return;
    }
    e.target.disabled = true;
    try {
      await api(`/admin/records/${encodeURIComponent(card.dataset.id)}`, { method: "DELETE" });
      records = records.filter((x) => x.id !== card.dataset.id);
      render();
      loadSummary();
    } catch (err) {
      e.target.disabled = false;
      confirmBox.textContent = err.message;
    }
  });

  document.getElementById("only-pending").addEventListener("change", render);

  try {
    await reload();
  } catch (err) {
    container.innerHTML = `<div class="msg-error">${escapeHtml(err.message)}</div>`;
  }
});
