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
}
PER_QUESTION_KEY = {  # özet metrik -> soru başına değer anahtarı
    "mrr@10": "rr", "hit@1": "hit@1", "hit@3": "hit@3", "hit@5": "hit@5", "hit@10": "hit@10",
    "context_hit": "context_hit", "context_chars": "context_chars", "latency_ms": "latency_ms",
    "answer_acc": "answer_ok",
    "rerank_input_recall": None,  # soru başına değer aşama sırasından türetilir (run.per_question_value)
    # Cevaptan türetilen metrikler (yalnız --llm); uzunluklar eski sonuçlarda da cevap metninden hesaplanır.
    "cevap_uzunlugu": None, "netlestirme_uzunlugu": None, "birlestirme_isabeti": "birlestirme",
    "fikra_atif_orani": "fikra_atif", "fikra_dogrulugu": "fikra_dogru",
}
LLM_METRICS = {"answer_acc", "cevap_uzunlugu", "netlestirme_uzunlugu", "birlestirme_isabeti", "fikra_atif_orani",
               "fikra_dogrulugu"}
BINARY_METRICS = {"hit@1", "hit@3", "hit@5", "hit@10", "context_hit", "answer_acc", "rerank_input_recall",
                  "birlestirme_isabeti"}
NO_CHANGE_EPS = 1e-9


# --- Veri seti ---

DEFAULT_DATASET = "sorular_v1"  # veri_seti alanı olmayan deneyler bu setle koşulur (set henüz yok)


def dataset_path(name=DEFAULT_DATASET):
    return config.ROOT / "eval" / f"{name}.json"


def load_dataset(name=DEFAULT_DATASET):
    raw = dataset_path(name).read_bytes()
    data = json.loads(raw.decode("utf-8"))
    return data, hashlib.sha256(raw).hexdigest()[:16]


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
    problems = []
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
