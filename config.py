"""Genel ayarlar ve RAG hattı yapılandırması.

RAG bileşenlerinin parametreleri (parçalama, embedding, getirme, reranker, LLM) kodda değil,
experiments/configs/EXP-*.json deney dosyalarında tutulur. Sohbet uygulaması
PRODUCTION_EXPERIMENT deneyinin parametreleriyle çalışır; böylece kullanılan her yapılandırma
kayıtlı ve ölçülmüş bir deneye karşılık gelir.
"""
import json
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).parent
DATA_DIR = ROOT / "data"
PLAN_PATH = DATA_DIR / "plan" / "bilgisayar_2024-2025.json"  # öğretim planı (mezuniyet hesabı)
INDEX_ROOT = ROOT / "vectorstore"  # her yapılandırma kendi alt klasöründe
EXPERIMENTS_DIR = ROOT / "experiments"

# Duyurular (scrape_site.py yazar, agent'ın get_announcements tool'u okur) ve Bilgisayar Mühendisliği akademik
# personeli (sabit liste, get_academic_staff). Corpus'a ve index'e girmez.
ANNOUNCEMENTS_DB = DATA_DIR / "duyurular.db"
ANNOUNCEMENTS_EXPORT_DIR = DATA_DIR / "duyurular"  # kaynak başına okunabilir JSON kopyası
ANNOUNCEMENTS_URL = "https://muhendislik.ibu.edu.tr/tr/news"
BM_ANNOUNCEMENTS_URL = "http://bm.muhendislik.ibu.edu.tr/tr/news"  # Bilgisayar Mühendisliği Bölümü (https yok)
GENERAL_ANNOUNCEMENTS_URL = "https://www.ibu.edu.tr/Website/Contents.aspx?PageID=1349&LangID=1"
STAFF_URL = "https://muhendislik.ibu.edu.tr/tr/akademikpersonel"  # personel listesinin kaynağı (taranmaz)
STAFF_PATH = DATA_DIR / "personel_bilgisayar.json"
# Duyuru özetleme modeli: "claude-..." Anthropic API'siyle, "gemini-..." Gemini'yle (agent'ın ücretsiz kotasından düşer).
ANNOUNCEMENT_SUMMARY_MODEL = "claude-haiku-5-5"

# Sohbet uygulamasının kullandığı deney. Yeni bir deney üretime alınacaksa burası değişir.
PRODUCTION_EXPERIMENT = "EXP-007"  # EXP-001 (campus-chatbot EXP-038 hattı, corpus v1) + gemini-3.5-flash-lite (EXP-006)

# Belge adları kaynak gösteriminde ve parçaların başlığında kullanılır.
# Listede olmayan bir PDF eklenirse dosya adı başlık olarak kullanılır.
# tur: "takvim" -> tablo satırları okunur, "mevzuat" -> madde madde bölünür (yapısal parçalamada).
DOCUMENTS = {
    "akademik-takvim-2026-2027.pdf": {"title": "BAİBÜ 2026-2027 Akademik Takvimi", "tur": "takvim"},
    "ogrenci-kayıt-yonerge.pdf": {"title": "BAİBÜ Öğrenci Kayıt Kabul İşleri Yönetmeliği", "tur": "mevzuat"},
    "yatay-gecis-yonergesi.pdf": {"title": "BAİBÜ Yatay Geçiş Yönergesi", "tur": "mevzuat"},
}

# Belge kümeleri (corpus parametresi). Her küme sabit bir dosya listesidir; data/ klasörüne sonradan
# eklenen bir PDF eski deneylerin belge kümesini değiştirmez.
#   v1: campus-chatbot corpus v4'ten (129 belge) Mühendislik Fakültesi öğrencisinin danışmanlığında gereken belgeler
#       (24 belge, 2026-10-21): eğitim-öğretim ve kayıt mevzuatı, geçişler, çift anadal/yandal, mühendislik staj
#       yönergesi, makine mühendisliği bitirme projesi esasları, akademik takvim, lisansüstü.
CORPORA = {
    "v1": DATA_DIR / "corpus_v1.txt",
}


def corpus_paths(name):
    files = CORPORA[name]
    if isinstance(files, Path):
        files = [l.strip() for l in files.read_text(encoding="utf-8").splitlines() if l.strip()]
    paths = [DATA_DIR / f for f in files]
    missing = [p.name for p in paths if not p.exists()]
    if missing:
        raise SystemExit(f"'{name}' belge kümesindeki dosyalar data/ içinde yok: {missing}")
    return sorted(paths)


# Sonradan eklenen parametreler. Eski config dosyaları değiştirilmediği için bu alanlar onlarda yoktur;
# eksikse buradaki değer kullanılır (eski deneyler bu değerle koşulmuştu).
PARAM_DEFAULTS = {"corpus": "v1", "device": "cpu", "reranker_dtype": "fp32", "fixed_answers": None, "citation_style": "madde",
                  "mod": "rag"}


def full_params(params):
    return {**PARAM_DEFAULTS, **params}


# Soru yeniden yazma ve cevapta kullanılacak son mesaj sayısı (soru+cevap = 2 mesaj).
HISTORY_MESSAGES = 6
NO_ANSWER = "Bu konuda bilgim yok."


@dataclass(frozen=True)
class RagConfig:
    # Belge işleme
    corpus: str                # CORPORA anahtarı: hangi belgeler indekslenir
    device: str                # "cpu" | "cuda": embedding ve reranker'ın çalıştığı donanım
    pdf_parser: str            # "pypdf" | "pdfplumber"
    chunking: str              # "recursive" (sabit boyut) | "structural" (madde / takvim tablosu)
                               # | "structural_v2" (genişletilmiş madde deseni) | "structural_v3" (+ belge bütünü fallback)
    chunk_size: int            # recursive: parça boyutu; structural: bu boyutu aşan madde bölünür (karakter)
    chunk_overlap: int         # karakter
    # Embedding
    embedding_model: str
    normalize_embeddings: bool  # True -> kosinüs benzerliği, False -> ham L2 uzaklığı
    # Getirme
    retrieval: str             # "vector" | "bm25" | "hybrid"
    vector_k: int              # vektör aramasından alınan aday sayısı
    bm25_k: int                # BM25'ten alınan aday sayısı
    rrf_k: int                 # Reciprocal Rank Fusion sabiti (yalnız hybrid)
    reranker_model: Optional[str]  # None -> reranker yok
    reranker_max_length: int   # reranker'ın okuduğu en fazla token
    reranker_dtype: str        # "fp32" | "fp16" (yarım hassasiyet: ~3× hızlı, daha az bellek; yalnız cuda)
    rerank_candidates: int     # reranker'a giden aday sayısı
    final_k: int               # LLM'e bağlam olarak verilen parça sayısı
    fixed_answers: Optional[str]  # None | "summer_v1": yaz okulu/yaz öğretimi sorularına retrieval yapmadan sabit cevap
                                  # (yönetmelik yürürlükten kalktı; başka belgelerdeki eski atıflar yanlış cevap üretiyordu)
    citation_style: str           # "madde": [Belge, Madde N] | "fikra": [Belge, Madde N/F]
    # Üretim
    mod: str                   # "rag": RagBot (retrieval + tek LLM çağrısı) | "agent": AdvisorAgent (LLM tool seçer)
    llm_model: str
    llm_temperature: float
    prompt_version: str        # rag.PROMPTS anahtarı (mod=rag) | agent.AGENT_PROMPTS anahtarı (mod=agent)

    # İndeksi (parçaları ve embedding'leri) etkileyen parametreler
    INDEX_FIELDS = ("corpus", "device", "pdf_parser", "chunking", "chunk_size", "chunk_overlap",
                    "embedding_model", "normalize_embeddings")

    @classmethod
    def from_dict(cls, params):
        params = full_params(params)
        names = {f.name for f in fields(cls)}
        unknown = set(params) - names
        missing = names - set(params)
        if unknown or missing:
            raise ValueError(f"Geçersiz parametreler. Bilinmeyen: {sorted(unknown)}, eksik: {sorted(missing)}")
        cfg = cls(**params)
        if cfg.mod not in ("rag", "agent"):
            raise ValueError(f"Bilinmeyen mod: {cfg.mod} (seçenekler: rag, agent)")
        if cfg.device not in ("cpu", "cuda"):
            raise ValueError(f"Bilinmeyen device: {cfg.device}")
        if cfg.corpus not in CORPORA:
            raise ValueError(f"Bilinmeyen corpus: {cfg.corpus} (seçenekler: {sorted(CORPORA)})")
        if cfg.fixed_answers not in (None, "summer_v1"):
            raise ValueError(f"Bilinmeyen fixed_answers: {cfg.fixed_answers}")
        if cfg.citation_style not in ("madde", "fikra"):
            raise ValueError(f"Bilinmeyen citation_style: {cfg.citation_style}")
        if cfg.reranker_dtype not in ("fp32", "fp16"):
            raise ValueError(f"Bilinmeyen reranker_dtype: {cfg.reranker_dtype}")
        if cfg.reranker_dtype == "fp16" and cfg.device != "cuda":
            raise ValueError("reranker_dtype=fp16 yalnız cuda'da desteklenir.")
        if cfg.chunking.startswith("structural") and cfg.pdf_parser != "pdfplumber":
            raise ValueError("structural parçalama pdfplumber gerektirir (takvim tabloları için).")
        return cfg

    def to_dict(self):
        return asdict(self)

    def index_params(self):
        return {k: getattr(self, k) for k in self.INDEX_FIELDS}


def load_experiment(exp_id):
    path = EXPERIMENTS_DIR / "configs" / f"{exp_id}.json"
    return json.loads(path.read_text(encoding="utf-8"))


def production_config():
    return RagConfig.from_dict(load_experiment(PRODUCTION_EXPERIMENT)["params"])
