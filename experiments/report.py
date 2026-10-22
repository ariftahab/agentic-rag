"""experiments/experiment_log.md dosyasını configs/ ve results/ klasörlerinden üretir.

  python -m experiments.report

Log kısa tutulur: hangi yapılandırma (hyperparameter'lar) kullanıldı, sonuç ne oldu, tek satırlık yorum.
Soru başına sonuçlar, istatistik ayrıntıları ve ortam bilgisi results/EXP-*.json içindedir.
Log elle düzenlenmez; tek satırlık yorum config'teki "kisa_not" alanından gelir.
"""
import json

import config
from experiments import evaluation as ev
from experiments import run

LOG_FILE = config.EXPERIMENTS_DIR / "experiment_log.md"

# Setup tablosunda gösterilen hyperparameter'lar (kısa ad -> RagConfig alanı)
SETUP_COLUMNS = [
    ("mod", "mod"), ("corpus", "corpus"), ("device", "device"), ("parser", "pdf_parser"), ("chunking", "chunking"), ("chunk_size", "chunk_size"),
    ("overlap", "chunk_overlap"), ("embedding", "embedding_model"), ("normalize", "normalize_embeddings"),
    ("retrieval", "retrieval"), ("reranker", "reranker_model"), ("final_k", "final_k"), ("prompt", "prompt_version"),
    ("llm", "llm_model"),
]
SHORT_VALUES = {
    "paraphrase-multilingual-MiniLM-L12-v2": "MiniLM", "BAAI/bge-m3": "bge-m3",
    "BAAI/bge-reranker-v2-m3": "bge-reranker", None: "–", True: "✓", False: "✗",
    "gemini-2.5-flash": "2.5-flash", "gemini-3.1-flash-lite": "3.1-flash-lite", "gemini-3.5-flash-lite": "3.5-flash-lite",
}
METRIC_NAMES = {"mrr@10": "MRR@10", "context_chars": "context chars", "answer_acc": "answer acc.",
                "tool_secimi_dogrulugu": "tool seçimi", "tool_adi_dogrulugu": "tool adı", "hesap_dogrulugu": "hesap",
                "gerekli_tool_dogrulugu": "gerekli tool", "yan_etkisiz_orani": "yan etkisiz"}
# Agent tablosu: başlık -> llm.metrics anahtarı
AGENT_COLUMNS = [("Answer acc.", "answer_acc"), ("Tool seçimi", "tool_secimi_dogrulugu"),
                 ("Tool adı", "tool_adi_dogrulugu"), ("Gerekli tool", "gerekli_tool_dogrulugu"),
                 ("Yan etkisiz", "yan_etkisiz_orani"), ("Hesap (cevap)", "hesap_dogrulugu"),
                 ("Hesap (tool)", "arac_hesap_dogrulugu"), ("Ort. tool çağrısı", "ort_tool_cagrisi"),
                 ("Ort. tur", "ort_tur")]


def _is_agent(res):
    return res.get("retrieval") is None and bool(res.get("llm"))


def _dataset_line(name):
    dataset, _ = ev.load_dataset(name)
    questions = dataset["sorular"]
    if any("beklenen_tool_lar" in q for q in questions):  # agent soru seti
        n_tool = sum(1 for q in questions if "beklenen_tool_lar" in q)
        n_calc = sum(1 for q in questions if "beklenen_sayilar" in q)
        return (f"`eval/{name}.json` v{dataset['surum']} ({len(questions)} soru: {n_tool} tool beklentili, "
                f"{n_calc} hesaplı)")
    n_ret = sum(1 for q in questions if ev.answerable(q))
    return (f"`eval/{name}.json` v{dataset['surum']} ({n_ret} cevaplı + "
            f"{len(questions) - n_ret} cevapsız/kapsam dışı)")


def _agent_lines(configs, results):
    agent_results = {e: r for e, r in results.items() if e in configs and _is_agent(r)}
    if not agent_results:
        return []
    lines = [
        "Tool seçimi: beklenen tool'lar doğru argümanlarla çağrıldı, beklenmeyen tool yok. Tool adı: aynısı, "
        "argümanlara bakmadan. Gerekli tool: beklenen tool'lar doğru argümanlarla çağrıldı, fazla çağrı serbest. Yan "
        "etkisiz: oturum durumunu değiştiren tool (teyit kaydı) beklenmeden çağrılmadı. Hesap (cevap): elle hesaplanan AKTS/GNO cevapta tam olarak geçiyor. Hesap (tool): "
        "aynı sayılar deterministik modülün sonucuyla aynı (LLM'den bağımsız). Answer acc. anlamca ayrıca okunur.",
        "",
        "| Exp | Set | " + " | ".join(c for c, _ in AGENT_COLUMNS) + " | n | Tamamlandı |",
        "|---|---|" + "---|" * len(AGENT_COLUMNS) + "---|---|",
    ]
    for exp_id, res in agent_results.items():
        # Sonradan eklenen metrikler eski sonuç dosyalarında yok; soru satırlarından türetilir.
        m = {**run.derived_agent_metrics(res["llm"]["per_question"]), **res["llm"]["metrics"]}
        cells = [(_num(m[k], 2) if k.startswith("ort_") else _num(m[k])) if m.get(k) is not None else "–"
                 for _, k in AGENT_COLUMNS]
        done = "✓" if res["llm"].get("tamamlandi", True) else "✗ (kota)"
        lines.append(f"| {exp_id} | {res['dataset']['surum']} | " + " | ".join(cells) + f" | {m['n']} | {done} |")
    return lines


def _load(folder):
    data = {p.stem: json.loads(p.read_text(encoding="utf-8"))
            for p in sorted((config.EXPERIMENTS_DIR / folder).glob("EXP-*.json"))}
    for d in data.values():  # sonradan eklenen parametreler eski dosyalarda yok; varsayılanla doldurulur
        d["params"] = config.full_params(d["params"])
    return data


def _num(value, digits=3):
    return f"{float(value):.{digits}f}"


def _short(value):
    return SHORT_VALUES.get(value, str(value)) if isinstance(value, (str, bool, type(None))) else str(value)


def _verdict(res, exp):
    """Birincil metrik için kısa karar: yön işareti, p-value ve hypothesis sonucu."""
    comp = (res or {}).get("karsilastirma")
    if not res:
        return "koşulmadı", ""
    if not comp:
        return "baseline", ""
    if not comp.get("ayni_veri_seti", True):
        return "referans (yeni soru seti, karşılaştırma yok)", "–"
    metric = exp["birincil_metrik"]
    m = comp["metrikler"].get(metric)
    if not m:
        return f"{METRIC_NAMES.get(metric, metric)}: ölçülmedi", "beklemede"
    # Ok değerin gerçek yönünü gösterir; iyi mi kötü mü olduğu ayrıca yazılır (context chars'ta azalma iyidir).
    if abs(m["fark"]) < ev.NO_CHANGE_EPS:
        change = "= değişmedi"
    else:
        good = (m["fark"] > 0) == (ev.METRIC_DIRECTION[metric] > 0)
        change = ("↑ " if m["fark"] > 0 else "↓ ") + ("iyileşti" if good else "kötüleşti")
    p = "p<0.01" if m["p"] < 0.01 else f"p={_num(m['p'], 2)}"
    sig = "anlamlı" if m["p"] < 0.05 else "anlamlı değil"
    hyp = {"Doğrulandı": "✓", "Doğrulanmadı": "✗", "Yön doğru, anlamlı değil": "~"}.get(comp["karar"]["hipotez"], "?")
    return f"{METRIC_NAMES.get(metric, metric)} {change} ({p}, {sig})", hyp


def build():
    configs, results = _load("configs"), _load("results")
    dataset_lines = [_dataset_line(name) for name in sorted({c.get("veri_seti", ev.DEFAULT_DATASET)
                                                             for c in configs.values()})]
    constant = {k: v for k, v in next(iter(configs.values()))["params"].items()
                if all(c["params"][k] == v for c in configs.values())}

    lines = [
        "# Experiment Log",
        "",
        f"> Otomatik üretilir (`python -m experiments.report`). Production: **{config.PRODUCTION_EXPERIMENT}**. "
        "Ayrıntılar: `experiments/results/EXP-*.json`.",
        "",
        "**Dataset:** " + " · ".join(dataset_lines) + ". Bir chunk, doğru kaynak belgelerden birine aitse ve o "
        "belgede birebir geçen evidence cümlesini içeriyorsa relevant sayılır. Farklı setlerle koşulan deneyler "
        "birbiriyle karşılaştırılmaz.  ",
        "**Kural:** Her deney parent'ından tek bir parametre farklı. Hypothesis deneyden önce yazılır. "
        "p < 0,05 = anlamlı (McNemar / Wilcoxon, aynı sorular üzerinde).  ",
        "**Sabit parametreler:** " + ", ".join(f"`{k}={_short(v)}`" for k, v in constant.items()),
        "",
        "## 1. Setup (hyperparameters)",
        "",
        "Kalın hücre = parent'a göre değişen parametre.",
        "",
        "| Exp | Parent | " + " | ".join(name for name, _ in SETUP_COLUMNS) + " | #chunks |",
        "|---|---|" + "---|" * len(SETUP_COLUMNS) + "---|",
    ]
    for exp_id, exp in configs.items():
        parent = configs.get(exp["ust_deney"]) if exp["ust_deney"] else None
        cells = []
        for _, field in SETUP_COLUMNS:
            value = _short(exp["params"][field])
            changed = parent and parent["params"][field] != exp["params"][field]
            cells.append(f"**{value}**" if changed else value)
        chunks = results[exp_id]["index"]["chunks"] if exp_id in results else "–"
        lines.append(f"| {exp_id} | {exp['ust_deney'] or '–'} | " + " | ".join(cells) + f" | {chunks} |")

    lines += [
        "",
        "## 2. Results",
        "",
        "| Exp | Değişiklik | Set | MRR@10 | Hit@1 | Hit@5 | Context hit | Context chars | Latency | Primary metric | Hyp. |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for exp_id, exp in configs.items():
        res = results.get(exp_id)
        verdict, hyp = _verdict(res, exp)
        if not res:
            lines.append(f"| {exp_id} | {exp['baslik']} | | | | | | | | {verdict} | |")
            continue
        if _is_agent(res):  # retrieval değerlendirmesi yok; metrikler agent tablosunda
            lines.append(f"| {exp_id} | {exp['baslik']} | {res['dataset']['surum']} | – | – | – | – | – | – | "
                         f"{verdict} | {hyp} |")
            continue
        m = res["retrieval"]["metrics"]
        latency = f"{m['latency_ms'] / 1000:.1f} s" if m["latency_ms"] >= 1000 else f"{m['latency_ms']:.0f} ms"
        lines.append(
            f"| {exp_id} | {exp['baslik']} | {res['dataset']['surum']} | {_num(m['mrr@10'])} | {_num(m['hit@1'])} | "
            f"{_num(m['hit@5'])} | "
            f"{_num(m['context_hit'])} | {m['context_chars']:.0f} | {latency} | {verdict} | {hyp} |")
    lines += ["", "Hyp.: ✓ doğrulandı · ✗ doğrulanmadı · ~ yön doğru ama anlamlı değil"]
    agent = _agent_lines(configs, results)
    if agent:  # Results'ın alt bölümü: RAG deneylerinin bölüm numaraları değişmesin
        lines += ["", "### Agent", "", *agent]

    staged = {e: r["retrieval"]["asamalar"] for e, r in results.items()
              if r.get("retrieval") and r["retrieval"].get("asamalar")}
    if staged:
        lines += [
            "",
            "## 3. Retrieval stages",
            "",
            "Her ara listede MRR@10. Rerank input recall: ilgili chunk reranker'a giden adaylar içinde mi "
            "(reranker'ın düzeltebileceğinin üst sınırı). Aşama verisi her koşuda kaydediliyor.",
            "",
            "| Exp | Vector | BM25 | RRF | Rerank input recall | Final |",
            "|---|---|---|---|---|---|",
        ]
        for exp_id, s in staged.items():
            cells = [s.get(k) for k in ("vector_mrr@10", "bm25_mrr@10", "rrf_mrr@10", "rerank_input_recall",
                                       "final_mrr@10")]
            lines.append(f"| {exp_id} | " + " | ".join(_num(c) if c is not None else "–" for c in cells) + " |")

    grouped = {e: r["retrieval"]["alt_kirilimlar"] for e, r in results.items()
               if r.get("retrieval") and r["retrieval"].get("alt_kirilimlar")}
    if grouped:
        cols = [("Eski sorular", "set:eski"), ("Yeni sorular", "set:yeni"), ("Belge dili", "stil:belge"),
                ("Öğrenci dili", "stil:ogrenci"), ("Fakülteye özel", "kategori:fakulteye_ozel"),
                ("Çok kapsamlı", "kategori:coklu_kapsam")]
        lines += ["", "## 4. Soru grupları (MRR@10)", "",
                  "Set 3.0'dan itibaren. Parantez içi soru sayısı; küçük gruplarda ortalama gürültülüdür.", "",
                  "| Exp | " + " | ".join(c for c, _ in cols) + " |", "|---|" + "---|" * len(cols)]
        for exp_id, g in grouped.items():
            cells = [f"{_num(g[k]['mrr@10'])} ({g[k]['n']})" if k in g else "–" for _, k in cols]
            lines.append(f"| {exp_id} | " + " | ".join(cells) + " |")

    lines += ["", "## 5. Notes", ""]
    for exp_id, exp in configs.items():
        if exp.get("kisa_not"):
            lines.append(f"- **{exp_id}:** {exp['kisa_not']}")
    lines.append("")
    LOG_FILE.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    build()
    print(f"Yazıldı: {LOG_FILE}")
