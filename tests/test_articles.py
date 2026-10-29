"""Mevzuatta madde düzeyinde gezinme (agent_h8, tools/articles.py) ve agent'taki tool'ları."""
from langchain_core.documents import Document

from tools import graduation
from tools.articles import ArticleIndex

LISANS = "önlisans lisans eğitim öğretim.pdf"
YANDAL = "yandal programı.pdf"


def chunk(chunk_id, source, madde, text, heading="", section=""):
    return Document(page_content=text, metadata={"chunk_id": chunk_id, "source": source, "title": source[:-4],
                                                 "page": 1, "madde": madde, "heading": heading, "section": section})


CHUNKS = [
    chunk(1, LISANS, 12, "MADDE 12- Yan dal programı...", "Yan dal", "İKİNCİ BÖLÜM"),
    chunk(2, LISANS, 13, "MADDE 13- Çift ana dal...", "Çift ana dal", "İKİNCİ BÖLÜM"),
    chunk(3, LISANS, 23, "MADDE 23- (3) Ders yükü en fazla 15 AKTS arttırılabilir. 12 nci ve 13 üncü maddeler "
                         "kapsamına giren öğrenciler için yönetim kurulu karar verir.", "Ders yükü", "ÜÇÜNCÜ BÖLÜM"),
    chunk(4, LISANS, 23, "(4) 21 inci maddede tanımlanan kredi değeri.", "Ders yükü", "ÜÇÜNCÜ BÖLÜM"),
    chunk(5, LISANS, 21, "MADDE 21- Kredi değeri (AKTS)...", "Kredi", "ÜÇÜNCÜ BÖLÜM"),
    chunk(6, LISANS, 3, "MADDE 3- Bu Yönetmelik, 2547 sayılı Yükseköğretim Kanununun 14 üncü maddesine dayanılarak "
                        "hazırlanmıştır.", "Dayanak"),
    chunk(7, LISANS, 14, "MADDE 14- ...", "Kontenjan"),
    chunk(8, YANDAL, 3, "MADDE 3- Bu Yönerge, Bolu Abant İzzet Baysal Üniversitesi Önlisans, Lisans Eğitim-Öğretim "
                        "Yönetmeliğinin 12 nci maddesinin üçüncü fıkrasına dayanarak hazırlanmıştır.", "Dayanak"),
    Document(page_content="takvim", metadata={"chunk_id": 9, "source": "takvim.pdf", "title": "Takvim", "page": 1}),
]
INDEX = ArticleIndex(CHUNKS)


def test_references_follow_same_document_and_named_documents_but_not_laws():
    assert INDEX.references(CHUNKS[2]) == [(LISANS, 12), (LISANS, 13)]
    assert INDEX.references(CHUNKS[3]) == [(LISANS, 21)]
    assert INDEX.references(CHUNKS[5]) == []                  # 2547 sayılı Kanunun 14 üncü maddesi korpusta değil
    assert INDEX.references(CHUNKS[7]) == [(LISANS, 12)]     # başka belgenin adıyla yapılan atıf


def test_article_text_joins_all_parts_in_order():
    text, parts = INDEX.article_text(LISANS, 23)
    assert [p.metadata["chunk_id"] for p in parts] == [3, 4] and text.index("15 AKTS") < text.index("21 inci")
    assert INDEX.article_text(LISANS, 99) == (None, [])


def test_find_document_by_title_file_name_or_part_of_the_title():
    assert INDEX.find_document("önlisans lisans eğitim öğretim") == LISANS
    assert INDEX.find_document("Önlisans Lisans Eğitim Öğretim.pdf".replace(".pdf", "")) == LISANS
    assert INDEX.find_document("yandal") == YANDAL
    assert INDEX.find_document("Takvim") is None              # madde yapısı yok
    assert INDEX.find_document("") is None


def test_table_of_contents_lists_articles_in_order():
    assert [m for m, _, _ in INDEX.table_of_contents(LISANS)] == [3, 12, 13, 14, 21, 23]
    assert INDEX.table_of_contents(LISANS)[-1] == (23, "Ders yükü", "ÜÇÜNCÜ BÖLÜM")


class FakeRetriever:
    chunks = CHUNKS

    def search(self, query):
        return [CHUNKS[2]]


def h8_tools(found):
    from agent import make_tools
    tools = make_tools(FakeRetriever(), graduation.load_plan(), None, {}, None, found=found, navigation=INDEX)
    return {t.name: t for t in tools}


def test_h8_search_results_show_references_and_get_article_reads_them():
    found = []
    tools = h8_tools(found)
    result = tools["search_regulations"].invoke({"query": "ders yükü"})
    assert "Atıf yapılan maddeler: önlisans lisans eğitim öğretim, Madde 12; önlisans lisans eğitim öğretim, Madde 13" in result
    article = tools["get_article"].invoke({"belge": "önlisans lisans eğitim öğretim", "madde": 23})
    assert "15 AKTS" in article and "21 inci" in article and "Madde 21" in article
    assert [d.metadata["chunk_id"] for d in found] == [3, 3, 4]   # okunan madde cevabın kaynaklarına girer
    assert "bulunamadı" in tools["get_article"].invoke({"belge": "yok böyle belge", "madde": 1})


def test_h8_list_articles_gives_contents_or_document_names():
    tools = h8_tools([])
    contents = tools["list_articles"].invoke({"belge": "yandal"})
    assert "Madde 3: Dayanak" in contents
    names = tools["list_articles"].invoke({"belge": ""})
    assert "önlisans lisans eğitim öğretim" in names and "Takvim" not in names


def test_navigation_tools_only_in_h8():
    from agent import AGENT_PROMPTS, AdvisorAgent
    assert "get_article" in AGENT_PROMPTS["agent_h8"] and "get_article" not in AGENT_PROMPTS["agent_h6"]
    plan = graduation.load_plan()
    assert AdvisorAgent(FakeRetriever(), None, plan, prompt_version="agent_h8").navigation is not None
    assert AdvisorAgent(FakeRetriever(), None, plan, prompt_version="agent_h6").navigation is None


# --- agent_h9: belge düzeyi ve tarihleri ---

from tools.articles import document_line, load_document_info  # noqa: E402


def test_document_info_covers_every_regulation_in_the_corpus():
    import config
    info = load_document_info()
    regulations = [p.name for p in config.corpus_paths("v1") if "takvim" not in p.name]
    assert sorted(info) == sorted(regulations)
    assert all(1 <= d["duzey_sira"] <= 4 for d in info.values())
    assert all(d.get("kabul") or d.get("son_degisiklik") or d.get("tarih_notu") for d in info.values())


def test_document_line_shows_level_and_dates():
    info = {YANDAL: {"duzey": "Senato yönergesi", "duzey_sira": 4, "kabul": "2023-01-12", "son_degisiklik": "2026-02-25"}}
    line = document_line(info, YANDAL)
    assert "düzey 4/4" in line and "Senato yönergesi" in line and "kabul 2023-01-12" in line and "2026-02-25" in line
    assert document_line(info, "takvim.pdf") == ""


def test_h9_search_results_carry_document_level():
    from agent import make_tools
    info = {LISANS: {"duzey": "üniversite yönetmeliği", "duzey_sira": 3, "kabul": "2019-10-14"}}
    tools = {t.name: t for t in make_tools(FakeRetriever(), graduation.load_plan(), None, {}, None, found=[],
                                           navigation=INDEX, document_info=info)}
    result = tools["search_regulations"].invoke({"query": "ders yükü"})
    assert "(Belge: düzey 3/4" in result and "Atıf yapılan maddeler" in result
    assert "(Belge: düzey 3/4" in tools["get_article"].invoke({"belge": "önlisans lisans eğitim öğretim", "madde": 21})


def test_h9_is_h8_plus_the_conflict_rule():
    from agent import AGENT_PROMPTS, AdvisorAgent
    h8, h9 = AGENT_PROMPTS["agent_h8"].splitlines(), AGENT_PROMPTS["agent_h9"].splitlines()
    added = [line for line in h9 if line not in h8]
    assert len(added) == 1 and "düzeyi üst olan" in added[0]
    agent = AdvisorAgent(FakeRetriever(), None, graduation.load_plan(), prompt_version="agent_h9")
    assert agent.document_info and agent.navigation is not None
    assert AdvisorAgent(FakeRetriever(), None, graduation.load_plan(), prompt_version="agent_h8").document_info is None
