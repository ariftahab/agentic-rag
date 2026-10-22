"""Deneylerde kullanılan ölçüm fonksiyonları: veri seti, ilgililik, metrikler, istatistik testleri, ortam bilgisi.

Metrik tanımları (hepsi soru başına hesaplanıp ortalaması alınır):
  hit@k        : ilk k parçada en az bir ilgili parça var mı (0/1)
  rr           : 1 / (ilk ilgili parçanın sırası), ilk 10'da yoksa 0; ortalaması MRR@10
  context_hit  : LLM'e giden ilk final_k parçada ilgili parça var mı (0/1)
  context_chars: LLM'e giden bağlamın toplam karakter sayısı (maliyet göstergesi)
  latency_ms   : yalnız getirme süresi (LLM hariç), milisaniye
  answer_ok    : LLM cevabı beklenen anahtar kelimeleri içeriyor mu (0/1), yalnız --llm ile
  netlestirme  : cevap öğrencinin fakültesini/bölümünü soruyor mu (0/1), yalnız --llm ile (run.NETLESTIRME)
  netlestirme_uzunlugu: netleştirme prompt'unun açıldığı sorularda cevabın karakter sayısı
  birlestirme  : birden fazla belgede aynı olan değer cevapta tam bir kez geçiyor mu (0/1; set 4.2 'birlestirme')
  fikra_atif   : cevaptaki madde atıflarından fıkra numarası taşıyanların oranı
  fikra_dogru  : fıkralı atıflardan kanıt cümlesini gerçekten o fıkrada bulunanların oranı (değerlendirilebilenler)

Agent metrikleri (params.mod = "agent"; yalnız ilgili alanı tanımlı sorular üzerinden ortalama):
  tool_dogru      : beklenen her tool doğru argümanlarla çağrıldı ve beklenmeyen bir tool çağrılmadı (0/1);
                    ortalaması tool_secimi_dogrulugu ('beklenen_tool_lar'), bkz. tool_selection
  tool_adi_dogru  : aynısı, argümanlara bakmadan (0/1); ortalaması tool_adi_dogrulugu. İkisinin farkı argüman hatası.
  hesap_dogru     : beklenen her sayı (AKTS, GNO) cevap metninde tam olarak geçiyor (0/1); ortalaması
                    hesap_dogrulugu ('beklenen_sayilar', transkript senaryoları), bkz. calculation
  arac_hesap_dogru: elle hesaplanan sayılar deterministik modülün (graduation.evaluate) sonucuyla aynı (0/1);
                    LLM'den bağımsız kontrol, ortalaması arac_hesap_dogrulugu
"""
import hashlib
import importlib.metadata
import json
import os
import platform
import re
import subprocess
from pathlib import Path

from scipy.stats import binomtest, wilcoxon

import config
from rag import tr_lower

EVAL_DEPTH = 10  # sıralama metrikleri ilk 10 parça üzerinden
HIT_KS = (1, 3, 5, 10)

# Metriklerin "iyi" yönü: +1 büyük iyi, -1 küçük iyi
METRIC_DIRECTION = {
    "mrr@10": +1, "hit@1": +1, "hit@3": +1, "hit@5": +1, "hit@10": +1,
    "context_hit": +1, "context_chars": -1, "latency_ms": -1, "answer_acc": +1,
    "rerank_input_recall": +1,
    "cevap_uzunlugu": -1, "netlestirme_uzunlugu": -1, "birlestirme_isabeti": +1,
    "fikra_atif_orani": +1, "fikra_dogrulugu": +1,
    "tool_secimi_dogrulugu": +1, "tool_adi_dogrulugu": +1, "hesap_dogrulugu": +1,
}
PER_QUESTION_KEY = {  # özet metrik -> soru başına değer anahtarı
    "mrr@10": "rr", "hit@1": "hit@1", "hit@3": "hit@3", "hit@5": "hit@5", "hit@10": "hit@10",
    "context_hit": "context_hit", "context_chars": "context_chars", "latency_ms": "latency_ms",
    "answer_acc": "answer_ok",
    "rerank_input_recall": None,  # soru başına değer aşama sırasından türetilir (run.per_question_value)
    # Cevaptan türetilen metrikler (yalnız --llm); uzunluklar eski sonuçlarda da cevap metninden hesaplanır.
    "cevap_uzunlugu": None, "netlestirme_uzunlugu": None, "birlestirme_isabeti": "birlestirme",
    "fikra_atif_orani": "fikra_atif", "fikra_dogrulugu": "fikra_dogru",
    # Agent metrikleri (yalnız mod=agent; RAG sonuçlarında bu alanlar yok, karşılaştırmaya girmez).
    "tool_secimi_dogrulugu": "tool_dogru", "tool_adi_dogrulugu": "tool_adi_dogru", "hesap_dogrulugu": "hesap_dogru",
}
LLM_METRICS = {"answer_acc", "cevap_uzunlugu", "netlestirme_uzunlugu", "birlestirme_isabeti", "fikra_atif_orani",
               "fikra_dogrulugu", "tool_secimi_dogrulugu", "tool_adi_dogrulugu", "hesap_dogrulugu"}
BINARY_METRICS = {"hit@1", "hit@3", "hit@5", "hit@10", "context_hit", "answer_acc", "rerank_input_recall",
                  "birlestirme_isabeti", "tool_secimi_dogrulugu", "tool_adi_dogrulugu", "hesap_dogrulugu"}
NO_CHANGE_EPS = 1e-9


# --- Veri seti ---

DEFAULT_DATASET = "sorular_v1"  # veri_seti alanı olmayan deneyler bu setle koşulur (set henüz yok)


def dataset_path(name=DEFAULT_DATASET):
    return config.ROOT / "eval" / f"{name}.json"


def load_dataset(name=DEFAULT_DATASET):
    raw = dataset_path(name).read_bytes()
    data = json.loads(raw.decode("utf-8"))
    return data, hashlib.sha256(raw).hexdigest()[:16]


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:16]


# Transkript senaryolarının fixture'ları: sorudaki 'transkript' alanı dosya adıdır (eval/transkriptler/<ad>.json).
# Biçim transcript.parse_transcript çıktısıyla aynı; kişisel bilgi alanı (ad, numara, TC) olamaz.
TRANSCRIPTS_DIR = config.ROOT / "eval" / "transkriptler"
TRANSCRIPT_KEYS = {"bolum", "egitim_duzeyi", "kayit_yili", "kayit_donemi", "dersler", "obs_ozet"}
COURSE_KEYS = {"yil", "donem", "kod", "ad", "kredi", "akts", "not", "erasmus"}


def transcript_path(name, folder=None):
    return Path(folder or TRANSCRIPTS_DIR) / f"{name}.json"


def load_transcript(name, folder=None):
    return json.loads(transcript_path(name, folder).read_text(encoding="utf-8"))


def norm(text):
    """Karşılaştırma için: NFC + Türkçe küçük harf + harf/rakam dışı her şey silinir.

    Böylece "ö ğretim" (pypdf boşluk hatası), satır sonları, tire ve iki nokta farkları eşleşmeyi bozmaz.
    """
    return re.sub(r"[\W_]+", "", tr_lower(text))


def evidence_alternatives(kanit):
    """Her alternatif, aynı parçada birlikte bulunması gereken ifadelerin listesidir."""
    return [[alt] if isinstance(alt, str) else alt for alt in kanit]


def sources(item):
    """Sorunun doğru kaynakları: [(belge, kanıt alternatifleri)].

    Set 2.0'da tek kaynak vardı ('kaynak' + 'kanit'); 3.0'da 'kaynaklar' listesi. Aynı kural birden fazla
    belgede geçebildiği için (ör. not tablosu birçok yönetmelikte aynı), soruyu sorulduğu haliyle doğru
    cevaplamaya yeten her belge ayrı kaynak olarak yazılır.
    """
    if "kaynaklar" in item:
        return [(s["belge"], evidence_alternatives(s["kanit"])) for s in item["kaynaklar"]]
    return [(item["kaynak"], evidence_alternatives(item["kanit"]))]


def answerable(item):
    """Retrieval değerlendirmesine girer mi (kanıtı olan soru)."""
    return "kaynaklar" in item or "kanit" in item


def matched_sources(item, chunk_text_norm, chunk_source):
    """Parçanın karşıladığı kaynakların sıra numaraları (sources(item) listesindeki konumları)."""
    return {i for i, (belge, alts) in enumerate(sources(item))
            if chunk_source == belge and any(all(norm(part) in chunk_text_norm for part in alt) for alt in alts)}


def is_relevant(item, chunk_text_norm, chunk_source):
    return bool(matched_sources(item, chunk_text_norm, chunk_source))


def validate_dataset(data):
    """Her kanıtın kaynak PDF'in ham metninde (her iki okuyucuda da) geçtiğini doğrular.

    Kanıt bir okuyucunun metninde yoksa o okuyucuyla yapılan deneyler haksız yere cezalanır;
    bu yüzden eksik kanıt hata sayılır.
    """
    import ingest
    texts = {}
    # Yalnız soruların kaynak gösterdiği belgeler okunur (139 belgenin hepsini iki okuyucuyla okumak yavaş).
    paths = {config.DATA_DIR / belge for item in data["sorular"] if answerable(item) for belge, _ in sources(item)}
    for path in paths:
        for parser in ("pypdf", "pdfplumber"):
            texts[(path.name, parser)] = norm(" ".join(ingest.read_pages(path, parser))) if path.exists() else ""
    problems = validate_agent_items(data, TRANSCRIPTS_DIR)
    for item in data["sorular"]:
        if not answerable(item):
            continue
        for belge, alts in sources(item):
            for alt in alts:
                for part in alt:
                    for parser in ("pypdf", "pdfplumber"):
                        if norm(part) not in texts.get((belge, parser), ""):
                            problems.append(f"{item['id']}: '{part}' {belge} ({parser}) metninde bulunamadı")
    return problems


def _transcript_problem(name, folder):
    """Fixture yoksa ya da parser çıktısında olmayan (kişisel bilgi olabilecek) bir alan taşıyorsa sorun metni."""
    path = transcript_path(name, folder)
    if not path.exists():
        return f"transkript fixture'ı bulunamadı: {path}"
    data = json.loads(path.read_text(encoding="utf-8"))
    extra = set(data) - TRANSCRIPT_KEYS
    extra |= {k for c in data.get("dersler", []) for k in set(c) - COURSE_KEYS}
    if extra or not data.get("dersler"):
        return f"transkript fixture'ı ({name}) parser biçiminde değil; fazla alanlar: {sorted(extra)}"
    return None


def validate_agent_items(data, transcripts_dir=None):
    """Agent alanlarının biçimi: 'beklenen_tool_lar' [{"name", "args"?}], 'transkript' (var olan fixture),
    'beklenen_sayilar' ({ad: sayı}, transkript senaryosu gerektirir). Her soru için en fazla bir sorun yazılır."""
    problems, checked = [], {}
    for item in data["sorular"]:
        problem = None
        expected = item.get("beklenen_tool_lar")
        numbers = item.get("beklenen_sayilar")
        if expected is not None and not (isinstance(expected, list) and all(
                isinstance(e, dict) and isinstance(e.get("name"), str) and isinstance(e.get("args", {}), dict)
                for e in expected)):
            problem = "'beklenen_tool_lar' [{\"name\": ..., \"args\": {...}}] listesi olmalı"
        elif item.get("transkript"):
            name = item["transkript"]
            if name not in checked:
                checked[name] = _transcript_problem(name, transcripts_dir)
            problem = checked[name]
        elif numbers is not None:
            problem = "'beklenen_sayilar' için 'transkript' fixture'ı gerekli"
        if problem is None and numbers is not None and not (isinstance(numbers, dict) and numbers and all(
                isinstance(v, (int, float)) and not isinstance(v, bool) for v in numbers.values())):
            problem = "'beklenen_sayilar' {ad: sayı} sözlüğü olmalı"
        if problem:
            problems.append(f"{item['id']}: {problem}")
    return problems


# --- Agent: tool seçimi ve hesap doğruluğu ---

NUMBER = re.compile(r"(?<![\d.,])\d+(?:[.,]\d+)?")


def _norm_value(value):
    """Argüman karşılaştırması için: metin Türkçe küçük harf ve tek boşluk, sayısal metin sayı, liste sırasız."""
    if isinstance(value, bool) or value is None:
        return value
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        text = " ".join(tr_lower(value).split())
        return float(text.replace(",", ".")) if NUMBER.fullmatch(text) else text
    if isinstance(value, (list, tuple)):
        return sorted((_norm_value(v) for v in value), key=repr)
    if isinstance(value, dict):
        return {k: _norm_value(v) for k, v in value.items()}
    return value


def args_match(expected, actual):
    """Beklenen argümanlar gerçek argümanların alt kümesi mi (yazılmayan argüman serbest)."""
    return all(k in actual and _norm_value(actual[k]) == _norm_value(v) for k, v in expected.items())


def tool_selection(expected, calls):
    """Sorunun beklenen tool çağrılarıyla agent'ın çağrılarını karşılaştırır.

    expected: [{"name", "args"?}], calls: [{"name", "args"}] (sonuç dosyasındaki 'tool_calls').
    Beklenen her çağrı, adı aynı ve argümanları beklenenleri içeren en az bir çağrıyla karşılanmalı; sıra önemsiz,
    beklenen bir tool'un tekrar çağrılması serbest (ör. iki arama). Beklenenler dışında bir tool çağrılırsa seçim
    yanlış sayılır: gereksiz tool hem latency hem yanlış bilgi riski. Boş liste "tool çağırmamalı" demektir.
    """
    names = {e["name"] for e in expected}
    missing = [e["name"] for e in expected
               if not any(c["name"] == e["name"] and args_match(e.get("args", {}), c["args"]) for c in calls)]
    missing_names = [e["name"] for e in expected if not any(c["name"] == e["name"] for c in calls)]
    extra = list(dict.fromkeys(c["name"] for c in calls if c["name"] not in names))
    return {"tool_dogru": int(not missing and not extra), "tool_adi_dogru": int(not missing_names and not extra),
            "eksik_tool_lar": missing, "fazla_tool_lar": extra}


def numbers_in_text(text):
    """Metindeki sayılar (ondalık ayırıcı virgül ya da nokta): "2,47" ve "2.47" -> 2.47."""
    return [float(m.replace(",", ".")) for m in NUMBER.findall(text)]


def _same_number(a, b):
    return a is not None and b is not None and abs(float(a) - float(b)) < NO_CHANGE_EPS


def calculation(expected, answer, reference=None):
    """Hesap doğruluğu: elle hesaplanmış sayılar kullanıcının gördüğü cevapta tam olarak geçiyor mu?

    Neden cevap metni: hesap deterministik koddadır ve kendi testleri vardır; agent'ın bozabileceği şey sayının
    kullanıcıya ulaşmasıdır (tool'u hiç çağırmamak, yanlış senaryonun sayısını aktarmak, sayıyı kendisi
    "hesaplamak"). Bu yüzden ölçülen, cevaptaki sayıdır. Eşleşme tam eşitliktir (yuvarlama kabul edilmez;
    "2,50" ile 2.5 aynı sayıdır). Sınır: sayının cevapta geçmesi, doğru etiketle geçtiğini garanti etmez (ör. 14
    başka bir bağlamda yazılmış olabilir); bu yüzden anlamca kontrol cevap okunarak yapılır.

    reference: deterministik modülün aynı fixture için sonucu (graduation.evaluate raporu). Verilirse beklenen
    sayıların modülün sonucuyla aynı olup olmadığı ayrıca yazılır (arac_hesap_dogru). Böylece yanlış bir cevabın
    kaynağı ayrılır: modül de farklıysa elle hesap ya da kod hatası, modül doğruysa LLM'in aktarma hatası.
    Raporda karşılığı olmayan sayılar (ör. what-if sonucu) bu kontrole girmez; hiçbiri yoksa None.
    """
    found = numbers_in_text(answer)
    numbers = {k: {"beklenen": v, "cevapta": any(_same_number(x, v) for x in found)} for k, v in expected.items()}
    judged = [k for k in expected if reference and k in reference]
    for k in judged:
        numbers[k]["aracta"] = reference[k]
    return {"hesap_dogru": int(all(n["cevapta"] for n in numbers.values())), "sayilar": numbers,
            "arac_hesap_dogru": int(all(_same_number(reference[k], expected[k]) for k in judged)) if judged else None}


# --- Cevap biçimi: birleştirme ve fıkralı atıf ---

CITATION = re.compile(r"\[([^\[\]]+)\]")
MADDE_REF = re.compile(r"madde\s*(\d+)(?:\s*/\s*(\d+)|\s*,?\s*(?:fıkra\s*\(?(\d+)\)?|\(?(\d+)\)?\s*\.?\s*fıkra))?")
# Fıkra başı: "MADDE 6- (1)" ya da cümle sonundan sonra "(2)". "(Değişik ...)(1)" gibi dipnot işaretleri sıra
# numarası tutmadığı için ayıklanır (fikra_spans yalnız 1, 2, 3... sırasıyla gelenleri alır).
FIKRA_START = re.compile(r"(?:MADDE\s*\d+\s*[–—-]\s*|[.:;]\s+|^\s*)\((\d+)\)", re.MULTILINE)


def citations(answer):
    """Cevaptaki köşeli parantezli atıflar: [{"metin", "madde", "fikra"}]; bir parantezde ';' ile birden çok atıf."""
    out = []
    for group in CITATION.findall(answer):
        for part in group.split(";"):
            m = MADDE_REF.search(tr_lower(part))
            fikra = next((int(g) for g in m.groups()[1:] if g), None) if m else None
            out.append({"metin": part.strip(), "madde": int(m.group(1)) if m else None, "fikra": fikra})
    return out


def fikra_spans(text):
    """Madde metnini fıkralara böler: {fıkra no: metin}. Fıkra işareti yoksa boş döner."""
    starts, expected = [], 1
    for m in FIKRA_START.finditer(text):
        if int(m.group(1)) == expected:
            starts.append((expected, m.start(1) - 1))
            expected += 1
    return {n: text[s:(starts[i + 1][1] if i + 1 < len(starts) else len(text))] for i, (n, s) in enumerate(starts)}


def fikra_correct(item, citation, madde_texts, titles):
    """Fıkralı atıf, sorunun kanıtını içeren fıkrayı mı gösteriyor? Değerlendirilemezse None.

    Değerlendirilebilir atıf: belgesi sorunun kaynaklarından biri, maddesi o kaynağın kanıtını içeren madde.
    madde_texts: {(belge, madde): madde metni}, titles: {belge: başlık} (atıfta belge başlığıyla anılır).
    """
    if not citation["fikra"] or not answerable(item):
        return None
    for belge, alts in sources(item):
        if norm(titles.get(belge, belge)) not in norm(citation["metin"]):
            continue
        text = madde_texts.get((belge, citation["madde"]))
        if not text:
            continue
        evidence = [norm(part) for alt in alts for part in alt]
        if not any(e in norm(text) for e in evidence):
            continue
        span = fikra_spans(text).get(citation["fikra"])
        return int(bool(span) and any(e in norm(span) for e in evidence))
    return None


def strip_citations(answer):
    return CITATION.sub(" ", answer)


def merged(item, answer):
    """Birleştirme: her ortak değer (alternatifleri '|' ile) atıflar dışındaki metinde tam bir kez geçiyor mu."""
    text = tr_lower(strip_citations(answer))
    for value in item["birlestirme"]:
        alts = sorted((re.escape(tr_lower(a)) for a in value.split("|")), key=len, reverse=True)
        if len(re.findall(r"(?<![\w,.])(?:" + "|".join(alts) + r")(?![\w])", text)) != 1:
            return False
    return True


# --- İstatistik ---

def paired_test(metric, before, after):
    """Aynı sorular üzerinde eşleştirilmiş test.

    İkili metrikler (hit, context_hit, answer) için kesin McNemar testi (uyuşmayan çiftler üzerinde
    binom testi); sürekli metrikler (rr, karakter, gecikme) için Wilcoxon işaretli sıra testi.
    Tüm farklar sıfırsa p = 1.
    """
    if metric in BINARY_METRICS:
        b = sum(1 for x, y in zip(before, after) if x > y)
        c = sum(1 for x, y in zip(before, after) if x < y)
        if b + c == 0:
            return {"test": "McNemar (kesin)", "p": 1.0, "kotulesen": b, "iyilesen": c}
        return {"test": "McNemar (kesin)", "p": binomtest(min(b, c), b + c, 0.5).pvalue,
                "kotulesen": b, "iyilesen": c}
    diffs = [y - x for x, y in zip(before, after)]
    if all(abs(d) < NO_CHANGE_EPS for d in diffs):
        return {"test": "Wilcoxon", "p": 1.0}
    return {"test": "Wilcoxon", "p": float(wilcoxon(before, after, zero_method="wilcox").pvalue)}


def verdict(metric, delta, p, expected):
    """Önceden tanımlı karar kuralı (experiment_log.md başında açıklanır)."""
    better = METRIC_DIRECTION[metric]
    if abs(delta) < NO_CHANGE_EPS:
        observed = "degisim_yok"
        # Ortalama aynı kalıp bazı sorular iyileşip bazıları kötüleşmiş olabilir.
        outcome = "Değişim yok" if p >= 1.0 else "Ortalamada değişim yok (bazı sorularda sıra değişti)"
    else:
        observed = "artis" if delta > 0 else "azalis"
        good = (delta > 0) == (better > 0)
        word = "iyileşme" if good else "kötüleşme"
        outcome = f"Anlamlı {word} (p<0,05)" if p < 0.05 else f"Sayısal {word} (istatistiksel olarak anlamlı değil)"
    if expected == observed:
        hypothesis = "Doğrulandı" if (observed == "degisim_yok" or p < 0.05) else "Yön doğru, anlamlı değil"
    else:
        hypothesis = "Doğrulanmadı"
    return outcome, hypothesis


# --- Ortam ---

KEY_PACKAGES = ["torch", "sentence-transformers", "transformers", "faiss-cpu", "rank-bm25", "pdfplumber",
                "pypdf", "langchain-core", "langchain-community", "langchain-text-splitters",
                "langchain-huggingface", "langchain-google-genai", "scipy"]


def _git(*args):
    try:
        return subprocess.run(["git", *args], cwd=config.ROOT, capture_output=True, text=True,
                              check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def model_revision(name):
    """Hugging Face önbelleğindeki model anlık görüntüsünün (snapshot) commit kimliği."""
    if not name:
        return None
    from huggingface_hub import try_to_load_from_cache
    repo = name if "/" in name else f"sentence-transformers/{name}"
    path = try_to_load_from_cache(repo, "config.json")
    return Path(path).parent.name if isinstance(path, str) else None


def environment(cfg):
    import torch
    return {
        "git_commit": _git("rev-parse", "HEAD"),
        # Koşucunun kendi çıktıları (sonuçlar, günlük, rewrite önbelleği) "commit edilmemiş değişiklik" sayılmaz.
        "git_dirty": bool(_git("status", "--porcelain", "--", ".", ":!experiments/results",
                               ":!experiments/experiment_log.md", ":!experiments/rewrite_cache.json")),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "cpu": platform.processor(),
        "cpu_count": os.cpu_count(),
        "torch_device": cfg.device,  # kullanılan cihaz (config'den); CUDA'nın var olması kullanıldığı anlamına gelmez
        "torch_version": torch.__version__,
        "gpu": torch.cuda.get_device_name(0) if cfg.device == "cuda" else None,
        "packages": {p: _version(p) for p in KEY_PACKAGES},
        "model_revisions": {
            "embedding": model_revision(cfg.embedding_model),
            "reranker": model_revision(cfg.reranker_model),
        },
    }


def _version(package):
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return None
