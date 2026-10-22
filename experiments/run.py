"""Deney koşturucu.

  python -m experiments.run EXP-003            # tek deney
  python -m experiments.run EXP-001 EXP-002    # sırayla birkaç deney
  python -m experiments.run --all              # tüm deneyler (sonucu olanlar atlanır)
  python -m experiments.run EXP-008 --llm      # cevap kalitesini de ölç (Gemini kotası harcar)
  python -m experiments.run EXP-003 --yeniden  # var olan sonucu yeniden üret

Her deney experiments/configs/EXP-*.json dosyasıyla tanımlanır. Sonuç experiments/results/EXP-*.json'a
yazılır ve experiments/experiment_log.md yeniden üretilir.

params.mod = "agent" olan deneyler retrieval değerlendirmesi yerine agent'ı koşar (yalnız --llm ile): her soru
AdvisorAgent'a sorulur, çağrılan tool'lar ve argümanları sonuca yazılır, tool seçimi ve hesap doğruluğu ölçülür.
"""
import argparse
import json
import re
import statistics
import sys
import time
from datetime import datetime

import config
from experiments import evaluation as ev
from experiments import report

CONFIGS_DIR = config.EXPERIMENTS_DIR / "configs"
RESULTS_DIR = config.EXPERIMENTS_DIR / "results"
REQUIRED_FIELDS = ("id", "baslik", "ust_deney", "neden", "hipotez", "birincil_metrik", "beklenen_yon", "params")


def load_result(exp_id):
    path = RESULTS_DIR / f"{exp_id}.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def dataset_name(exp):
    """Deneyin soru seti (eval/<ad>.json). Alan yoksa varsayılan set (evaluation.DEFAULT_DATASET)."""
    return exp.get("veri_seti", ev.DEFAULT_DATASET)


def param_diff(exp):
    """Üst deneye göre değişen parametreler: {ad: [eski, yeni]}.

    Soru seti RAG parametresi değil ama değişirse deneyin tek değişkeni odur; bu yüzden farka dahil edilir.
    """
    if not exp["ust_deney"]:
        return {}
    parent_exp = config.load_experiment(exp["ust_deney"])
    parent = {**config.full_params(parent_exp["params"]), "veri_seti": dataset_name(parent_exp)}
    child = {**config.full_params(exp["params"]), "veri_seti": dataset_name(exp)}
    return {k: [parent[k], v] for k, v in child.items() if parent.get(k) != v}


def check_experiment(exp):
    missing = [f for f in REQUIRED_FIELDS if f not in exp]
    if missing:
        raise SystemExit(f"{exp.get('id')}: eksik alanlar {missing}")
    if exp["birincil_metrik"] not in ev.METRIC_DIRECTION:
        raise SystemExit(f"{exp['id']}: bilinmeyen birincil metrik {exp['birincil_metrik']}")
    if exp["beklenen_yon"] not in ("artis", "azalis", "degisim_yok"):
        raise SystemExit(f"{exp['id']}: beklenen_yon artis | azalis | degisim_yok olmalı")
    cfg = config.RagConfig.from_dict(exp["params"])  # parametre doğrulaması
    if cfg.mod == "agent":
        from agent import AGENT_PROMPTS
        if cfg.prompt_version not in AGENT_PROMPTS:
            raise SystemExit(f"{exp['id']}: agent modunda prompt_version {sorted(AGENT_PROMPTS)} anahtarlarından biri olmalı")
    if not ev.dataset_path(dataset_name(exp)).exists():
        raise SystemExit(f"{exp['id']}: soru seti bulunamadı: {ev.dataset_path(dataset_name(exp))}")
    diff = param_diff(exp)
    if exp["ust_deney"] and not diff:
        raise SystemExit(f"{exp['id']}: üst deneyden hiçbir parametre farklı değil.")
    # Tek değişken ilkesi: birden fazla parametre değişiyorsa gerekçe yazılmış olmalı.
    if len(diff) > 1 and not exp.get("coklu_degisiklik_gerekcesi"):
        raise SystemExit(f"{exp['id']}: birden fazla parametre değişiyor {sorted(diff)}; etkiyi izole etmek için "
                         f"ayrı deneylere bölün ya da 'coklu_degisiklik_gerekcesi' yazın.")
    return diff


def ensure_index(exp_id, cfg):
    """İndeks yoksa ayrı bir süreçte kurar.

    Aynı süreçte kurulursa embedding hesabının GPU belleği (PyTorch önbelleği) retrieval ölçümü sırasında
    da tutulur; 6 GB'lık kartta VRAM dolup Windows belleği sistem RAM'ine taşıyınca latency birkaç kat
    şişiyordu (campus-chatbot EXP-016: 2220 ms, ayrı süreçte ~400 ms). Süreç bitince belleğin tamamı geri verilir, böylece
    latency her zaman "indeks hazır, temiz süreç" koşulunda ölçülür.
    """
    import subprocess
    import ingest
    if not (ingest.index_dir(cfg) / "manifest.json").exists():
        subprocess.run([sys.executable, "ingest.py", exp_id], cwd=config.ROOT, check=True)


REWRITE_CACHE = config.EXPERIMENTS_DIR / "rewrite_cache.json"
STAGES = ("vector", "bm25", "rrf", "rerank_input", "final")  # trace'teki chunk listeleri (diğer anahtarlar bilgi)


def make_rewriter(cfg, retriever):
    """Sohbet geçmişli sorular için production'daki rewrite adımı (RagBot.rewrite), önbellekli.

    Rewrite bir LLM çağrısıdır ve sıcaklık > 0 olduğu için her koşuda biraz farklı sorgu üretebilir. Retrieval
    parametresini değiştiren bir deneyde bu fark gürültüdür; bu yüzden sonuç; LLM modeli, sıcaklık, rewrite prompt'u,
    geçmiş ve soru aynı kaldıkça önbellekten okunur. Yalnız rewrite'ı etkileyen bir değişiklik yeni sorgu üretir.
    """
    import hashlib
    from langchain_google_genai import ChatGoogleGenerativeAI
    import rag
    bot = rag.RagBot(cfg, retriever, ChatGoogleGenerativeAI(model=cfg.llm_model, temperature=cfg.llm_temperature,
                                                            max_retries=2))
    cache = json.loads(REWRITE_CACHE.read_text(encoding="utf-8")) if REWRITE_CACHE.exists() else {}

    def rewrite(item):
        key = hashlib.sha256(json.dumps([cfg.llm_model, cfg.llm_temperature, bot.prompt["history"], rag.REWRITE_SYSTEM,
                                         item["gecmis"], item["soru"]], ensure_ascii=False).encode()).hexdigest()[:16]
        if key not in cache:
            cache[key] = bot.rewrite(item["soru"], item["gecmis"])
            REWRITE_CACHE.write_text(json.dumps(cache, ensure_ascii=False, indent=1), encoding="utf-8")
        return cache[key]
    return rewrite


def evaluate_retrieval(cfg, retriever, items, rewrite=None):
    retriever.rank(items[0]["soru"])  # ısınma: ilk çağrıdaki model/önbellek yüklemesi süreye katılmasın
    rows = []
    for item in items:
        # Sohbet geçmişli (takip) sorularda retrieval, production'daki gibi rewrite edilmiş sorguyla yapılır.
        query = rewrite(item) if item.get("gecmis") else item["soru"]
        trace = {}
        started = time.perf_counter()
        ranking = retriever.rank(query, depth=ev.EVAL_DEPTH, trace=trace)
        latency = (time.perf_counter() - started) * 1000
        top = ranking[:ev.EVAL_DEPTH]
        relevant = [ev.is_relevant(item, ev.norm(c.page_content), c.metadata["source"]) for c in top]
        first = next((i + 1 for i, r in enumerate(relevant) if r), None)
        context = ranking[:trace.get("context_n", cfg.final_k)]  # kapsam tamamlama context'i büyütebilir
        row = {
            "id": item["id"],
            "rank": first,
            "rr": 1 / first if first else 0.0,
            "context_hit": int(any(relevant[:cfg.final_k])),
            "context_chars": sum(len(c.page_content) for c in context),
            "latency_ms": round(latency, 1),
            "top5": [f"{c.metadata['source']} s.{c.metadata['page']}"
                     + (f" m.{c.metadata['madde']}" if "madde" in c.metadata else "") for c in top[:5]],
        }
        for k in ev.HIT_KS:
            row[f"hit@{k}"] = int(bool(first and first <= k))
        if query != item["soru"]:
            row["query"] = query
        if trace.get("scope_added"):
            row["scope_added"] = trace["scope_added"]
        row["asamalar"] = stage_ranks(item, retriever.chunks, trace)
        rows.append(row)
    return rows


def stage_ranks(item, chunks, trace):
    """Her ara listede ilk ilgili parçanın sırası (1'den başlar; listede yoksa None).

    Böylece bir hatanın aday üretiminde mi (ilgili parça hiçbir listeye girmemiş: recall hatası)
    yoksa sıralamada mı (aday listesinde var ama aşağıda kalmış: ranking hatası) oluştuğu görülür.
    """
    cache = {}

    def relevant(cid):
        if cid not in cache:
            c = chunks[cid]
            cache[cid] = ev.is_relevant(item, ev.norm(c.page_content), c.metadata["source"])
        return cache[cid]

    return {stage: next((i + 1 for i, cid in enumerate(ids) if relevant(cid)), None)
            for stage, ids in trace.items() if stage in STAGES}


def stage_metrics(rows):
    """Aşama başına MRR@10; reranker varsa ilgili parçanın reranker'a giden adaylar içinde olma oranı."""
    out = {}
    for stage in rows[0]["asamalar"]:
        ranks = [r["asamalar"][stage] for r in rows]
        if stage == "rerank_input":
            out["rerank_input_recall"] = round(statistics.mean(int(x is not None) for x in ranks), 4)
        else:
            out[f"{stage}_mrr@10"] = round(statistics.mean(
                1 / x if x and x <= ev.EVAL_DEPTH else 0.0 for x in ranks), 4)
    return out


def summarize(rows, key):
    return round(statistics.mean(r[key] for r in rows), 4)


def retrieval_metrics(rows):
    m = {"mrr@10": summarize(rows, "rr")}
    for k in ev.HIT_KS:
        m[f"hit@{k}"] = summarize(rows, f"hit@{k}")
    m["context_hit"] = summarize(rows, "context_hit")
    m["context_chars"] = round(statistics.mean(r["context_chars"] for r in rows), 1)
    m["latency_ms"] = round(statistics.mean(r["latency_ms"] for r in rows), 1)
    m["latency_ms_median"] = round(statistics.median(r["latency_ms"] for r in rows), 1)
    return m


def madde_index(chunks):
    """Fıkra doğruluğu için: {(belge, madde): maddenin bütün metni} ve {belge: başlık} (chunk'lar sırayla birleştirilir)."""
    texts, titles = {}, {}
    for c in chunks:
        m = c.metadata
        titles[m["source"]] = m["title"]
        if "madde" in m:
            texts[(m["source"], m["madde"])] = texts.get((m["source"], m["madde"]), "") + "\n" + c.page_content
    return texts, titles


def answer_extras(item, answer, maddeler):
    """Cevap biçimi değerleri: uzunluk, birleştirme (varsa), fıkralı atıf oranı ve fıkra doğruluğu (varsa)."""
    out = {"uzunluk": len(answer)}
    if item.get("birlestirme"):
        out["birlestirme"] = int(ev.merged(item, answer))
    cited = [c for c in ev.citations(answer) if c["madde"]]
    if cited:
        out["fikra_atif"] = round(statistics.mean(c["fikra"] is not None for c in cited), 4)
    checked = [ev.fikra_correct(item, c, *maddeler) for c in cited]
    checked = [x for x in checked if x is not None]
    if checked:
        out["fikra_dogru"] = round(statistics.mean(checked), 4)
    return out


def answer_rows(bot, items, wait):
    rows = []
    maddeler = madde_index(bot.retriever.chunks)
    for n, item in enumerate(items):
        try:
            out = bot.ask(item["soru"], item.get("gecmis"))  # takip sorularında sohbet geçmişiyle
            answer, llm_ms, usage = out["answer"], round(out["generation_ms"], 1), out.get("usage")
        except Exception as e:  # kota dolarsa kalan soruları boşuna deneme
            print(f"  LLM hatası ({item['id']}): {type(e).__name__}: {str(e)[:200]}")
            if "ResourceExhausted" in type(e).__name__ or "429" in str(e):
                print("  Kota dolu; cevap değerlendirmesi yarıda kesildi.")
                return rows, False
            answer, llm_ms, out, usage = f"<hata: {type(e).__name__}>", None, None, None
        rows.append({"id": item["id"], "answer": answer, "answer_ok": int(answer_ok(item, answer)),
                     "netlestirme": int(asks_unit(answer)), "llm_ms": llm_ms,
                     **answer_extras(item, answer, maddeler)})
        if usage:  # sabit cevaplarda (yaz okulu) LLM çağrılmaz, token yok
            rows[-1]["tokens"] = usage
        print(f"  {item['id']} cevap {'OK' if rows[-1]['answer_ok'] else 'YANLIŞ'}")
        if n + 1 < len(items):
            time.sleep(wait)
    return rows, True


def make_agent(exp_id, cfg):
    """Deneyin agent'ı: config'teki retrieval hattı (search_regulations), LLM modeli ve agent prompt'u.

    Model agent.AGENT_LLM_MODEL sabitinden değil config'ten alınır; böylece kullanılan model sonuçta kayıtlıdır.
    Dönüş: (agent, index manifest'i)."""
    from langchain_google_genai import ChatGoogleGenerativeAI
    import graduation
    import ingest
    from agent import REQUEST_TIMEOUT_S, AdvisorAgent
    from rag import Retriever
    ensure_index(exp_id, cfg)
    store, chunks, manifest = ingest.load_index(cfg)
    llm = ChatGoogleGenerativeAI(model=cfg.llm_model, temperature=cfg.llm_temperature, max_retries=2,
                                 timeout=REQUEST_TIMEOUT_S)
    agent = AdvisorAgent(Retriever(cfg, store, chunks), llm, graduation.load_plan(), prompt_version=cfg.prompt_version)
    return agent, manifest


def agent_rows(agent, items, wait, plan, transcripts_dir=None):
    """Her soruyu agent'a sorar; cevap, tool çağrıları (ad + argümanlar), tur sayısı ve agent metrikleri.

    Soru alanları: 'gecmis' (takip sorusu), 'transkript' (fixture adı; oturumdaki transkript), 'teyitler'
    (oturumun eşdeğerlik teyitleri, her soruda yeni kopya), 'beklenen_tool_lar', 'beklenen_sayilar', 'beklenen'.
    Bir metriğin alanı soruda yoksa o metriğin değeri satıra yazılmaz (ortalamaya girmez).
    agent_ms: ask() çağrısının tamamı (LLM turları + tool'ların çalışması), sorular arası bekleme hariç.
    """
    import graduation
    transcripts = {}
    rows = []
    for n, item in enumerate(items):
        transcript = None
        if item.get("transkript"):
            name = item["transkript"]
            if name not in transcripts:
                transcripts[name] = ev.load_transcript(name, transcripts_dir)
            transcript = transcripts[name]
        started = time.perf_counter()
        try:
            out = agent.ask(item["soru"], item.get("gecmis"), transcript=transcript,
                            confirmations=dict(item.get("teyitler") or {}))
            row = {"id": item["id"], "answer": out["answer"], "tool_calls": out["tool_calls"], "turns": out["turns"],
                   "agent_ms": round((time.perf_counter() - started) * 1000, 1)}
            if out.get("usage"):
                row["tokens"] = out["usage"]
        except Exception as e:  # kota dolarsa kalan soruları boşuna deneme
            print(f"  Agent hatası ({item['id']}): {type(e).__name__}: {str(e)[:200]}")
            if "ResourceExhausted" in type(e).__name__ or "429" in str(e):
                print("  Kota dolu; agent değerlendirmesi yarıda kesildi.")
                return rows, False
            row = {"id": item["id"], "answer": f"<hata: {type(e).__name__}>", "tool_calls": [], "turns": None,
                   "agent_ms": None}
        if "beklenen" in item or item.get("cevapsiz"):
            row["answer_ok"] = int(answer_ok(item, row["answer"]))
        if "beklenen_tool_lar" in item:
            row.update(ev.tool_selection(item["beklenen_tool_lar"], row["tool_calls"]))
        if "beklenen_sayilar" in item:
            reference = graduation.evaluate(transcript, plan, confirmations=dict(item.get("teyitler") or {}))
            calc = ev.calculation(item["beklenen_sayilar"], row["answer"], reference)
            row.update(hesap_dogru=calc["hesap_dogru"], sayilar=calc["sayilar"])
            if calc["arac_hesap_dogru"] is not None:
                row["arac_hesap_dogru"] = calc["arac_hesap_dogru"]
        rows.append(row)
        marks = [f"{label} {'OK' if row[key] else 'YANLIŞ'}" for key, label in
                 (("tool_dogru", "tool"), ("hesap_dogru", "hesap"), ("answer_ok", "cevap")) if key in row]
        print(f"  {item['id']} " + (", ".join(marks) or "değerlendirme alanı yok"))
        if n + 1 < len(items):
            time.sleep(wait)
    return rows, True


# Agent özet metrikleri: ad -> soru başına anahtar (yalnız anahtarı olan sorular üzerinden ortalama)
AGENT_METRICS = (("answer_acc", "answer_ok", "answer_n"), ("tool_secimi_dogrulugu", "tool_dogru", "tool_secimi_n"),
                 ("tool_adi_dogrulugu", "tool_adi_dogru", None), ("hesap_dogrulugu", "hesap_dogru", "hesap_n"),
                 ("arac_hesap_dogrulugu", "arac_hesap_dogru", "arac_hesap_n"),
                 ("gerekli_tool_dogrulugu", "gerekli_tool_dogru", None), ("yan_etkisiz_orani", "yan_etkisiz", None))
DERIVED_TOOL_METRICS = {"gerekli_tool_dogrulugu": "gerekli_tool_dogru", "yan_etkisiz_orani": "yan_etkisiz"}


def derived_agent_metrics(rows):
    """Sonradan eklenen tool metrikleri (gerekli tool, yan etki) sonuç dosyasına dokunmadan, satırlardan."""
    out = {}
    for metric in DERIVED_TOOL_METRICS:
        values = [v for v in (per_question_value(r, metric, None) for r in rows) if v is not None]
        if values:
            out[metric] = round(statistics.mean(values), 4)
    return out


def agent_metrics(rows):
    out = {"n": len(rows)}
    for metric, key, count in AGENT_METRICS:
        values = [r[key] for r in rows if key in r]
        if values:
            out[metric] = round(statistics.mean(values), 4)
            if count:
                out[count] = len(values)
    if rows:
        out["ort_tool_cagrisi"] = round(statistics.mean(len(r["tool_calls"]) for r in rows), 2)
    turns = [r["turns"] for r in rows if r["turns"] is not None]
    if turns:
        out["ort_tur"] = round(statistics.mean(turns), 2)
    timed = [r["agent_ms"] for r in rows if r["agent_ms"] is not None]
    if timed:
        out.update(agent_ms=round(statistics.mean(timed), 1), agent_ms_median=round(statistics.median(timed), 1))
    used = [r["tokens"] for r in rows if r.get("tokens")]
    if used:
        out.update({f"token_{k}": round(statistics.mean(t[k] for t in used), 1) for k in ("input", "output", "reasoning")})
    return out


def agent_subset_metrics(rows, items):
    """Soru grupları ('grup', 'kategori') için agent metrikleri; küçük gruplarda ortalama gürültülüdür."""
    meta = {q["id"]: q for q in items}
    groups = {}
    for r in rows:
        for key in ("grup", "kategori"):
            if key in meta[r["id"]]:
                groups.setdefault(f"{key}:{meta[r['id']][key]}", []).append(r)
    out = {}
    for name, members in sorted(groups.items()):
        out[name] = {"n": len(members)}
        for metric, key, _ in AGENT_METRICS:
            values = [r[key] for r in members if key in r]
            if values:
                out[name][metric] = round(statistics.mean(values), 4)
    return out


def run_agent(exp_id, exp, cfg, diff, dataset, dataset_hash, wait):
    """Agent modundaki deneyin koşusu ve sonuç dosyası (retrieval değerlendirmesi yok: 'retrieval' None).

    Sonuçta agent satırları 'llm' altında durur; böylece answer_acc ve agent metrikleri üst deneyle aynı yoldan
    (compare) karşılaştırılır. Transkript fixture'larının, öğretim planının ve kuralların hash'i de kaydedilir:
    bunlar değişirse aynı soru seti farklı doğru cevaplar demektir.
    """
    import graduation
    from agent import MAX_TURNS
    agent, manifest = make_agent(exp_id, cfg)
    items = dataset["sorular"]
    names = sorted({q["transkript"] for q in items if q.get("transkript")})
    started = time.perf_counter()
    rows, complete = agent_rows(agent, items, wait, graduation.load_plan(), ev.TRANSCRIPTS_DIR)
    agent_s = time.perf_counter() - started

    env = ev.environment(cfg)
    if env["git_dirty"]:
        print("  UYARI: çalışma dizininde commit edilmemiş değişiklik var; sonuç tam tekrar üretilemeyebilir.")
    result = {
        "id": exp_id,
        "calisma_zamani": datetime.now().isoformat(timespec="seconds"),
        "dataset": {"dosya": f"eval/{dataset_name(exp)}.json", "surum": dataset["surum"], "hash": dataset_hash,
                    "n_agent": len(items), "n_toplam": len(dataset["sorular"]),
                    "transkriptler": {n: ev.file_hash(ev.transcript_path(n)) for n in names}},
        "params": exp["params"],
        "degisen_parametreler": diff,
        "index": manifest,
        "environment": env,
        "agent": {"prompt_version": cfg.prompt_version, "max_tur": MAX_TURNS,
                  # Agent bugünün tarihini prompt'a ve ders programı tool'una verir ("yarın" sorularında cevap değişir).
                  "bugun": datetime.now().date().isoformat(),
                  "plan_hash": ev.file_hash(config.PLAN_PATH), "kurallar_hash": ev.file_hash(graduation.RULES_PATH)},
        "retrieval": None,
        "llm": {"tamamlandi": complete, "metrics": agent_metrics(rows), "per_question": rows,
                "alt_kirilimlar": agent_subset_metrics(rows, items)},
        # agent: bütün soruların ask() süresi + sorular arası --bekle beklemeleri (index yükleme hariç)
        "sure": {"agent_sn": round(agent_s, 1)},
    }
    result["sure"]["toplam_sn"] = result["sure"]["agent_sn"]
    parent = load_result(exp["ust_deney"]) if exp["ust_deney"] else None
    result["karsilastirma"] = compare(exp, result, parent) if parent else None

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / f"{exp_id}.json").write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
    m = result["llm"]["metrics"]
    print("  " + " ".join(f"{k}={m[k]:.3f}" for k in ("tool_secimi_dogrulugu", "tool_adi_dogrulugu", "gerekli_tool_dogrulugu", "yan_etkisiz_orani", "hesap_dogrulugu",
                                                      "arac_hesap_dogrulugu", "answer_acc") if k in m))
    if result["karsilastirma"] and "karar" in result["karsilastirma"]:
        k = result["karsilastirma"]["karar"]
        print(f"  Karar ({k['metrik']}): {k['sonuc']} | Hipotez: {k['hipotez']}")


def _normalize_answer(text):
    return re.sub(r"\b0(\d)", r"\1", ev.tr_lower(text))  # "05 Ocak" == "5 Ocak"


# Cevap öğrencinin fakültesini/bölümünü soruyor mu? (set 4.1, beklenen_davranis: netlestirme). Yalnız soru ifadeleri:
# "kayıtlı olduğunuz birime göre değişir" gibi açıklamalar sayılmaz (campus-chatbot EXP-023'ün 139 cevabında hiç eşleşme yok).
NETLESTIRME = re.compile(
    r"hangi (fakülte|bölüm|program|birim)|(fakülte|bölüm|program|birim)\w* (belirt(ir|ebilir|in|men)|"
    r"yaz(ar|abilir|ın|man)|paylaş(ır|abilir|ın)|söyle(r|yebilir|yin)|bildir(ir|ebilir|in))")


def asks_unit(answer):
    return bool(NETLESTIRME.search(ev.tr_lower(answer)))


def clarification_metrics(items, answers):
    """netlestirme_isabeti: netleştirme beklenen sorularda sorma oranı; gereksiz_netlestirme: diğer sorularda."""
    expected = {q["id"] for q in items if q.get("beklenen_davranis") == "netlestirme"}
    if not expected:
        return {}
    hit = [a["netlestirme"] for a in answers if a["id"] in expected]
    extra = [a["netlestirme"] for a in answers if a["id"] not in expected]
    out = {}
    if hit:
        out.update(netlestirme_isabeti=round(statistics.mean(hit), 4), netlestirme_n=len(hit))
    if extra:
        out["gereksiz_netlestirme"] = round(statistics.mean(extra), 4)
    return out


def answer_ok(item, answer):
    normalize = _normalize_answer
    answer = normalize(answer)
    if item.get("cevapsiz"):
        # 'kabul': belge tutarı/saati vermiyor ama kimin belirlediğini söylüyor; bunu söyleyen cevap da doğru.
        return any(normalize(x) in answer for x in [config.NO_ANSWER, *item.get("kabul", [])])
    return all(any(normalize(alt) in answer for alt in kw.split("|")) for kw in item["beklenen"])


def per_question_value(row, metric, key):
    """Soru başına metrik değeri; rerank_input_recall: ilgili parça reranker'a giden adaylar içinde mi (0/1).

    Uzunluklar cevap metninden hesaplanır (bu alanlardan önceki sonuçlarla da karşılaştırılabilsin).
    Değeri olmayan soru (ör. netleştirme açılmamış, birleştirme sorusu değil) karşılaştırmaya girmez.
    """
    if metric == "rerank_input_recall":
        stages = row.get("asamalar", {})
        return int(stages["rerank_input"] is not None) if "rerank_input" in stages else None
    if metric == "cevap_uzunlugu":
        return len(row["answer"])
    if metric == "netlestirme_uzunlugu":
        return len(row["answer"]) if row.get("netlestirme_tetik") else None
    if metric in DERIVED_TOOL_METRICS:
        if "eksik_tool_lar" not in row:  # sorunun beklenen tool'u yok
            return None
        return ev.tool_values(row["eksik_tool_lar"], row["fazla_tool_lar"])[DERIVED_TOOL_METRICS[metric]]
    return row.get(key)


def compare(exp, result, parent):
    """Üst deneyle aynı sorular üzerinde eşleştirilmiş karşılaştırma."""
    # Transkript senaryolarında fixture'lar da veri setinin parçası (RAG sonuçlarında alan yok: ikisi de None).
    same_data = (parent["dataset"]["hash"] == result["dataset"]["hash"]
                 and parent["dataset"].get("transkriptler") == result["dataset"].get("transkriptler"))
    out = {"ust_deney": exp["ust_deney"], "ayni_veri_seti": same_data, "metrikler": {}}
    if not same_data:
        out["uyari"] = "Üst deney farklı bir veri seti sürümüyle koşulmuş; karşılaştırma geçersiz."
        return out
    for metric, key in ev.PER_QUESTION_KEY.items():
        source = "llm" if metric in ev.LLM_METRICS else "retrieval"
        a, b = parent.get(source), result.get(source)
        if not a or not b or not a.get("tamamlandi", True) or not b.get("tamamlandi", True):
            continue
        before_rows = {r["id"]: per_question_value(r, metric, key) for r in a["per_question"]}
        after_rows = {r["id"]: per_question_value(r, metric, key) for r in b["per_question"]}
        ids = [i for i in after_rows if before_rows.get(i) is not None and after_rows[i] is not None]
        if not ids:
            continue
        before = [before_rows[i] for i in ids]
        after = [after_rows[i] for i in ids]
        before_mean, after_mean = statistics.mean(before), statistics.mean(after)
        test = ev.paired_test(metric, before, after)
        out["metrikler"][metric] = {
            "once": round(before_mean, 4), "sonra": round(after_mean, 4),
            "fark": round(after_mean - before_mean, 4), **test,
        }
    primary = exp["birincil_metrik"]
    if primary in out["metrikler"]:
        m = out["metrikler"][primary]
        outcome, hypothesis = ev.verdict(primary, m["fark"], m["p"], exp["beklenen_yon"])
        out["karar"] = {"metrik": primary, "beklenen_yon": exp["beklenen_yon"],
                        "sonuc": outcome, "hipotez": hypothesis}
    else:
        out["karar"] = {"metrik": primary, "beklenen_yon": exp["beklenen_yon"],
                        "sonuc": "Ölçülmedi", "hipotez": "Beklemede (birincil metrik bu koşuda ölçülmedi)"}
    if not parent.get("retrieval") or not result.get("retrieval"):  # agent koşusunda retrieval değerlendirmesi yok
        return out
    # Sıralaması değişen sorular (tezde hata analizi için)
    before_rank = {r["id"]: r["rank"] for r in parent["retrieval"]["per_question"]}
    out["degisen_sorular"] = [
        {"id": r["id"], "once": before_rank.get(r["id"]), "sonra": r["rank"]}
        for r in result["retrieval"]["per_question"] if before_rank.get(r["id"]) != r["rank"]
    ]
    return out


def format_metrics(answers):
    """Cevap biçimi özetleri (değeri olan sorular üzerinden ortalama)."""
    out = {"cevap_uzunlugu": round(statistics.mean(len(a["answer"]) for a in answers), 1)}
    used = [a for a in answers if a.get("tokens") and a["tokens"]["output"] and a.get("llm_ms")]
    if used:
        # Üretim hızı: (çıktı + reasoning) token / saniye; latency farkının modelden mi sunucudan mı geldiğine ipucu.
        out.update({f"token_{k}": round(statistics.mean(a["tokens"][k] for a in used), 1)
                    for k in ("input", "output", "reasoning")})
        out["token_per_sn"] = round(statistics.median(
            (a["tokens"]["output"] + a["tokens"]["reasoning"]) / (a["llm_ms"] / 1000) for a in used), 1)
    clarified = [len(a["answer"]) for a in answers if a.get("netlestirme_tetik")]
    if clarified:
        out.update(netlestirme_uzunlugu=round(statistics.mean(clarified), 1), netlestirme_uzunlugu_n=len(clarified))
    for metric, key in (("birlestirme_isabeti", "birlestirme"), ("fikra_atif_orani", "fikra_atif"),
                        ("fikra_dogrulugu", "fikra_dogru")):
        values = [a[key] for a in answers if key in a]
        if values:
            out.update({metric: round(statistics.mean(values), 4), f"{metric}_n": len(values)})
    return out


def subset_metrics(rows, items, answers=None):
    """Soru alt grupları için ortalamalar: eski/yeni set, stil (belge/öğrenci dili), kategori.

    Set 2.0'da bu alanlar yok; o zaman boş döner. Grup küçükse (birkaç soru) ortalama gürültülüdür.
    """
    meta = {q["id"]: q for q in items}
    groups = {}
    for key in ("stil", "kategori", "grup"):
        for r in rows:
            if key in meta[r["id"]]:
                groups.setdefault(f"{key}:{meta[r['id']][key]}", []).append(r["id"])
    if any("stil" in q for q in items):
        for r in rows:
            groups.setdefault("set:eski" if meta[r["id"]].get("eski_set") else "set:yeni", []).append(r["id"])
    by_id = {r["id"]: r for r in rows}
    ok = {a["id"]: a["answer_ok"] for a in (answers or [])}
    asked = {a["id"]: a["netlestirme"] for a in (answers or []) if "netlestirme" in a}
    out = {}
    for name, ids in sorted(groups.items()):
        out[name] = {"n": len(ids), "mrr@10": round(statistics.mean(by_id[i]["rr"] for i in ids), 4),
                     "hit@5": round(statistics.mean(by_id[i]["hit@5"] for i in ids), 4)}
        answered = [ok[i] for i in ids if i in ok]
        if answered:
            out[name]["answer_acc"] = round(statistics.mean(answered), 4)
        clarified = [asked[i] for i in ids if i in asked]
        if clarified:
            out[name]["netlestirme"] = round(statistics.mean(clarified), 4)  # fakülteyi/bölümü sorma oranı
    return out


def run(exp_id, use_llm, wait, rerun, datasets):
    exp = config.load_experiment(exp_id)
    diff = check_experiment(exp)
    dataset, dataset_hash = datasets(dataset_name(exp))
    if load_result(exp_id) and not rerun:
        print(f"{exp_id}: sonuç zaten var, atlanıyor (yeniden üretmek için --yeniden).")
        return
    if exp["ust_deney"] and not load_result(exp["ust_deney"]):
        print(f"{exp_id}: üst deney {exp['ust_deney']} henüz koşulmamış; karşılaştırma yapılmayacak.")

    cfg = config.RagConfig.from_dict(exp["params"])
    print(f"\n=== {exp_id}: {exp['baslik']}")
    for k, (old, new) in diff.items():
        print(f"    {k}: {old} -> {new}")
    if cfg.mod == "agent":
        if not use_llm:  # agent'ın ölçülecek her şeyi LLM çağrısı gerektirir; --all koşusu bu yüzden durmaz
            print(f"  {exp_id}: agent modu yalnız --llm ile koşulur (soru başına birkaç LLM çağrısı); atlandı.")
            return
        return run_agent(exp_id, exp, cfg, diff, dataset, dataset_hash, wait)

    from rag import RagBot, Retriever
    import ingest
    ensure_index(exp_id, cfg)
    store, chunks, manifest = ingest.load_index(cfg)
    retriever = Retriever(cfg, store, chunks)
    items = [q for q in dataset["sorular"] if ev.answerable(q)]
    # Takip soruları varsa rewrite için LLM gerekir (retrieval-only koşuda da; sonuçlar önbelleğe alınır).
    rewrite = make_rewriter(cfg, retriever) if any(q.get("gecmis") for q in items) else None
    started = time.perf_counter()
    rows = evaluate_retrieval(cfg, retriever, items, rewrite)
    retrieval_s = time.perf_counter() - started

    env = ev.environment(cfg)
    if env["git_dirty"]:
        print("  UYARI: çalışma dizininde commit edilmemiş değişiklik var; sonuç tam tekrar üretilemeyebilir.")
    result = {
        "id": exp_id,
        "calisma_zamani": datetime.now().isoformat(timespec="seconds"),
        "dataset": {"dosya": f"eval/{dataset_name(exp)}.json", "surum": dataset["surum"], "hash": dataset_hash,
                    "n_getirme": len(items),
                    "n_toplam": len(dataset["sorular"])},
        "params": exp["params"],
        "degisen_parametreler": diff,
        "index": manifest,
        "environment": env,
        "retrieval": {"metrics": retrieval_metrics(rows), "asamalar": stage_metrics(rows), "per_question": rows},
        "llm": None,
        # Koşu süresi (tez için). getirme: index yükleme hariç bütün soruların retrieval'ı (rewrite dahil);
        # llm: yalnız modelin cevap üretme süresi (retrieval, rewrite ve sorular arası --bekle beklemeleri hariç).
        "sure": {"getirme_sn": round(retrieval_s, 1)},
    }
    if use_llm:
        from langchain_google_genai import ChatGoogleGenerativeAI
        llm = ChatGoogleGenerativeAI(model=cfg.llm_model, temperature=cfg.llm_temperature, max_retries=2)
        bot = RagBot(cfg, retriever, llm)  # getirme için yüklenen retriever yeniden kullanılır
        answers, complete = answer_rows(bot, dataset["sorular"], wait)
        timed = [a["llm_ms"] for a in answers if a.get("llm_ms") is not None]
        result["sure"]["llm_sn"] = round(sum(timed) / 1000, 1)
        result["llm"] = {
            "tamamlandi": complete,
            "metrics": {"answer_acc": summarize(answers, "answer_ok") if answers else None,
                        "n": len(answers), **clarification_metrics(dataset["sorular"], answers),
                        **(format_metrics(answers) if answers else {}),
                        **({"llm_ms": round(statistics.mean(timed), 1), "llm_ms_median": round(statistics.median(timed), 1)}
                           if timed else {})},
            "per_question": answers,
        }
    result["sure"]["toplam_sn"] = round(sum(result["sure"].values()), 1)
    llm_rows = result["llm"]["per_question"] if result["llm"] else None
    result["retrieval"]["alt_kirilimlar"] = subset_metrics(rows, items, llm_rows)
    parent = load_result(exp["ust_deney"]) if exp["ust_deney"] else None
    result["karsilastirma"] = compare(exp, result, parent) if parent else None

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / f"{exp_id}.json").write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
    m = result["retrieval"]["metrics"]
    print(f"  MRR@10={m['mrr@10']:.3f} Hit@1={m['hit@1']:.3f} Hit@5={m['hit@5']:.3f} "
          f"bağlam isabeti={m['context_hit']:.3f} gecikme={m['latency_ms']:.0f} ms")
    print("  Aşamalar: " + " ".join(f"{k}={v:.3f}" for k, v in result["retrieval"]["asamalar"].items()))
    if result["karsilastirma"] and "karar" in result["karsilastirma"]:
        k = result["karsilastirma"]["karar"]
        print(f"  Karar ({k['metrik']}): {k['sonuc']} | Hipotez: {k['hipotez']}")


def late_compare(exp_id):
    """Üst deneyi sonradan koşulan bir sonucun boş kalan karşılaştırmasını tamamlar (LLM çağrısı yok).

    Örnek: model deneyi o modelin kotasıyla, üst deney (baseline) ise diğer modelin kotası yenilenince koşuldu.
    Ölçümler (retrieval, llm, per_question) değişmez; yalnız boş 'karsilastirma' alanı doldurulur ve ne zaman
    doldurulduğu yazılır. Karşılaştırması zaten olan sonuca dokunulmaz.
    """
    exp, result = config.load_experiment(exp_id), load_result(exp_id)
    if not result:
        raise SystemExit(f"{exp_id}: sonuç yok.")
    if result.get("karsilastirma"):
        print(f"{exp_id}: karşılaştırma zaten var, değiştirilmedi.")
        return
    parent = load_result(exp["ust_deney"]) if exp["ust_deney"] else None
    if not parent:
        raise SystemExit(f"{exp_id}: üst deney {exp['ust_deney']} hâlâ koşulmamış.")
    result["karsilastirma"] = compare(exp, result, parent)
    result["karsilastirma_sonradan"] = datetime.now().isoformat(timespec="seconds")
    (RESULTS_DIR / f"{exp_id}.json").write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
    k = result["karsilastirma"].get("karar")
    if k:
        print(f"  {exp_id} karar ({k['metrik']}): {k['sonuc']} | Hipotez: {k['hipotez']}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("ids", nargs="*")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--llm", action="store_true", help="cevap kalitesini de ölç (Gemini kotası harcar)")
    parser.add_argument("--bekle", type=float, default=4.0, help="LLM çağrıları arası saniye")
    parser.add_argument("--yeniden", action="store_true", help="var olan sonucun üzerine yaz")
    parser.add_argument("--karsilastir", action="store_true",
                        help="koşmadan, boş kalan karşılaştırmayı üst deneyin sonucuyla tamamla")
    args = parser.parse_args()

    ids = sorted(p.stem for p in CONFIGS_DIR.glob("EXP-*.json")) if args.all else args.ids
    if not ids:
        parser.error("deney kimliği ya da --all verin")

    cache = {}

    def datasets(name):
        """Soru setini bir kez yükleyip doğrular (her deney kendi setini seçebilir)."""
        if name not in cache:
            dataset, dataset_hash = ev.load_dataset(name)
            problems = ev.validate_dataset(dataset)
            if problems:
                raise SystemExit(f"Soru seti ({name}) doğrulaması başarısız:\n  " + "\n  ".join(problems))
            cache[name] = (dataset, dataset_hash)
        return cache[name]

    for exp_id in ids:
        if args.karsilastir:
            late_compare(exp_id)
        else:
            run(exp_id, args.llm, args.bekle, args.yeniden, datasets)
    report.build()
    print(f"\nGünlük güncellendi: {report.LOG_FILE}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
