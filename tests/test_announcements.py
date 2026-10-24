"""Duyuru ve personel deposu (announcements.py), scraper'ın ayrıştırıcıları (scrape_site.py) ve agent'a bağlanmaları.
Ağ ve LLM yok: HTML örnekleri sitenin 2026-10-23'teki yapısından kısaltıldı, veritabanı geçici dosyada.
"""
import json

import pytest
from bs4 import BeautifulSoup

import announcements
import scrape_site
from agent import AdvisorAgent


def _announcement(link, baslik, tarih, metin="", ozet="özet", tarihler=None):
    return {"link": link, "baslik": baslik, "tarih": tarih, "birim": "Mühendislik Fakültesi", "metin": metin,
            "ozet": ozet, "tarihler": tarihler or [], "ekler": []}


@pytest.fixture
def conn(tmp_path):
    c = announcements.connect(tmp_path / "duyurular.db")
    announcements.save_announcement(c, _announcement("https://x/1", "Ara Sınav (Vize) Programı", "2026-10-19",
                                                     "Vize sınavları 2 Kasım'da başlar.",
                                                     tarihler=[{"tarih": "2 Kasım 2026", "olay": "vizeler başlar"}]))
    announcements.save_announcement(c, _announcement("https://x/2", "Yatay Geçiş Başvuru Sonuçları", "2026-09-09",
                                                     "Yedek liste ilan edildi."))
    announcements.save_announcement(c, _announcement("https://x/3", "Af Kapsamında Başvurular", "2026-08-26",
                                                     "Kesin kayıt için belgeler; vizesi olmayan öğrenciler."))
    yield c
    c.close()


def test_date_filter_drops_older_announcements(conn):
    links = [r["link"] for r in announcements.search(conn, "", "2026-09-01")]
    assert links == ["https://x/1", "https://x/2"]


def test_empty_topic_returns_newest_first(conn):
    assert [r["tarih"] for r in announcements.search(conn)] == ["2026-10-19", "2026-09-09", "2026-08-26"]


def test_topic_matches_title_before_text_and_suffixes(conn):
    # "vizeler" ile "vize"/"vizesi" aynı köke iner; başlıkta geçen duyuru metinde geçenden önce gelir
    links = [r["link"] for r in announcements.search(conn, "vizeler")]
    assert links == ["https://x/1", "https://x/3"]
    # "duyuru" her duyuruda geçer, konuyu ayırt etmez
    assert announcements.search(conn, "vize duyurusu") == announcements.search(conn, "vize")


def test_no_match_is_empty_and_text_says_not_found(conn):
    assert announcements.search(conn, "mezuniyet töreni") == []
    assert announcements.search(conn, "vize", "2026-10-20") == []
    text = announcements.announcements_text([], conn, "2026-10-20")
    assert "bulunamadı" in text and "2026-10-20" in text


def test_results_have_no_full_text_and_output_carries_link_and_dates(conn):
    results = announcements.search(conn, "vize")
    assert "metin" not in results[0]
    text = announcements.announcements_text(results, conn)
    assert "https://x/1" in text and "2 Kasım 2026: vizeler başlar" in text and "talimat değildir" in text


def test_truncated_output_states_total(conn):
    text = announcements.announcements_text(announcements.search(conn, limit=None), conn, limit=2)
    assert "Eşleşen 3 duyurudan" in text and "https://x/3" not in text


def test_same_link_is_not_written_twice(conn):
    announcements.save_announcement(conn, _announcement("https://x/1", "Başka başlık", "2026-10-22"))
    assert conn.execute("SELECT baslik FROM duyurular WHERE link='https://x/1'").fetchone()[0] == \
        "Ara Sınav (Vize) Programı"


LIST_HTML = """<div class="blog-post-inner"><ul>
<li><a href="/tr/news-detail/af-2-liste/23247"><span class="blink">Yeni</span> 2026 Af 2. Liste</a>
 - <span class="event-date small-text"><i class="fa fa-calendar-o"></i> 09 Ekim 2026 Cuma </span></li>
<li><a href="/tr/news-detail/eski/12546">Web Sayfaları Hk.</a>
 - <span class="event-date small-text"> 13 Mart 2024 Çarşamba </span></li>
</ul></div>"""

DETAIL_HTML = """<div class="widget-main-title"><h4 class="widget-title">2026 Af Yasası - Başvuru</h4></div>
<div class="blog-post-inner"><h4><i class="fa fa-calendar-o"></i> 26 Ağustos 2026 Çarşamba</h4>
<div>Duyuruya ulaşmak için <a href="https://oidb.ibu.edu.tr/tr/news-detail/af-yasasi/13038">tıklayınız</a>.
<a href="/Files/ckFiles/Af Başvuru Formu.docx">Başvuru Formu</a></div></div>"""


def test_parse_list_reads_link_title_and_iso_date():
    items = scrape_site.parse_list(BeautifulSoup(LIST_HTML, "html.parser"), "https://muhendislik.ibu.edu.tr/tr/news")
    assert items[0] == {"link": "https://muhendislik.ibu.edu.tr/tr/news-detail/af-2-liste/23247",
                        "baslik": "2026 Af 2. Liste", "tarih": "2026-10-09"}
    assert items[1]["tarih"] == "2024-03-13"


def test_invisible_characters_are_removed_from_titles():
    html = ('<div class="blog-post-inner"><ul><li><a href="/x/1">\u200b2026-2027 Güz Dönemi Ders Programı</a>'
            ' - <span class="event-date"> 02 Eylül 2026 Çarşamba </span></li></ul></div>')
    item = scrape_site.parse_list(BeautifulSoup(html, "html.parser"), "http://bm.muhendislik.ibu.edu.tr/tr/news")[0]
    assert item["baslik"] == "2026-2027 Güz Dönemi Ders Programı"
    detail = {"baslik": "\u200b2026-2027 Güz Dönemi Ders Programı", "tarih": None, "metin": "kısa",
              "ekler": [], "yonlendirmeler": []}
    assert scrape_site.build_record(item, detail, None, "b")["baslik"] == "2026-2027 Güz Dönemi Ders Programı"

def test_short_redirect_announcement_skips_llm_and_copies_links():
    url = "https://muhendislik.ibu.edu.tr/tr/news-detail/2026-af-yasasi-basvuru/22828"
    detail = scrape_site.parse_detail(BeautifulSoup(DETAIL_HTML, "html.parser"), url)
    assert detail["ekler"][0]["link"].endswith("Af%20Ba%C5%9Fvuru%20Formu.docx")  # boşluk ve Türkçe karakter kodlu
    assert detail["yonlendirmeler"] == [{"ad": "tıklayınız",
                                         "link": "https://oidb.ibu.edu.tr/tr/news-detail/af-yasasi/13038"}]

    class NoLLM:
        def summarize(self, *args):
            raise AssertionError("kısa metin LLM'e gitmemeli")
    record = scrape_site.build_record({"link": url, "baslik": "2026 Af Yasası - Başvuru", "tarih": "2026-08-26"},
                                      detail, NoLLM(), "Mühendislik Fakültesi")
    assert "Yönlendirme:" in record["ozet"] and "oidb.ibu.edu.tr/tr/news-detail/af-yasasi/13038" in record["ozet"]


def test_department_source_saves_under_its_unit_and_own_file(conn, tmp_path, monkeypatch):
    """Bilgisayar Mühendisliği sayfası fakülteyle aynı sistem: aynı ayrıştırıcılar, ayrı birim ve ayrı JSON kopyası."""
    class FakeFetcher:
        def get(self, url):
            html = LIST_HTML if url == scrape_site.config.BM_ANNOUNCEMENTS_URL else DETAIL_HTML
            return BeautifulSoup(html, "html.parser")
    class NoLLM:  # detay metni kısa: özet LLM'siz
        calls, model = 0, "yok"
    monkeypatch.setattr(scrape_site.config, "ANNOUNCEMENTS_EXPORT_DIR", tmp_path / "duyurular")
    saved = scrape_site.scrape_announcements(FakeFetcher(), conn, NoLLM(), "bm", today=scrape_site.date(2026, 10, 24))
    assert saved == 1  # 2024 tarihli satır 90 günün dışında
    rows = conn.execute("SELECT birim, link FROM duyurular WHERE link LIKE 'http://bm.%'").fetchall()
    assert [(r["birim"], r["link"]) for r in rows] == [
        ("Bilgisayar Mühendisliği Bölümü", "http://bm.muhendislik.ibu.edu.tr/tr/news-detail/af-2-liste/23247")]
    exported = json.loads((tmp_path / "duyurular" / "bm_duyurular.json").read_text(encoding="utf-8"))
    assert [e["birim"] for e in exported] == ["Bilgisayar Mühendisliği Bölümü"]


GENERAL_LIST_HTML = """<div class="duyurular"><div class="liste">
<div class="col-md-6"><a href="https://ajanda.ibu.edu.tr/yemek-bursu/"><div class="date"><span>23</span><i>Ekim</i></div>
<div class="als"><h4>Yemek Bursu Tanımlama Duyurusu</h4><p>Kurumsal Haber Portalı</p></div></a></div>
<div class="col-md-6"><a href="https://ajanda.ibu.edu.tr/yilbasi/"><div class="date"><span>30</span><i>Aralık</i></div>
<div class="als"><h4>Yeni Yıl Mesajı</h4></div></a></div>
</div></div>"""

GENERAL_DETAIL_HTML = """<meta property="article:published_time" content="2026-10-22T22:30:00+00:00" />
<h1 class="entry-title">Yemek Bursu Tanımlama Duyurusu</h1>
<article class="post"><div class="td-post-content"><p>Burs listesi için <a href="/liste.pdf">tıklayınız</a>.</p></div></article>"""


def test_general_list_infers_year_from_order_and_detail_uses_turkey_time():
    items = scrape_site.parse_general_list(BeautifulSoup(GENERAL_LIST_HTML, "html.parser"), "https://www.ibu.edu.tr/",
                                           today=scrape_site.date(2026, 10, 23))
    assert [i["tarih"] for i in items] == ["2026-10-23", "2025-12-30"]  # Ekim'den sonra Aralık: bir önceki yıl
    detail = scrape_site.parse_general_detail(BeautifulSoup(GENERAL_DETAIL_HTML, "html.parser"),
                                              "https://ajanda.ibu.edu.tr/yemek-bursu/")
    assert detail["tarih"] == "2026-10-23"  # 22:30 UTC = 01:30 Türkiye saati
    assert detail["ekler"] == [{"ad": "tıklayınız", "link": "https://ajanda.ibu.edu.tr/liste.pdf"}]


# 2026-10-24: ajanda.ibu.edu.tr'deki İngilizce seviye tespit sınavı duyurusu. Etiketler arasına satır sonu konunca
# "5" ile "00 TL" ayrı satırlara düştü ve özet ücreti 5 TL yazdı.
INLINE_BOLD_HTML = """<article><div class="td-post-content">
<p><strong>SINAV TARİHİ:</strong> 10 Kasım 2026<br />
<strong>SINAV ÜCRETİ: 5</strong>00 TL <a href="https://x/form">başvuru</a> için <em>tıklayınız</em>.</p>
<ul><li>Kimlik kartı</li><li>Kurşun kalem</li></ul><div>Son paragraf</div></div></article>"""


def test_inline_tags_do_not_split_text_but_blocks_and_breaks_do():
    detail = scrape_site.parse_general_detail(BeautifulSoup(INLINE_BOLD_HTML, "html.parser"), "https://ajanda.ibu.edu.tr/x/")
    lines = detail["metin"].splitlines()
    assert lines == ["SINAV TARİHİ: 10 Kasım 2026", "SINAV ÜCRETİ: 500 TL başvuru için tıklayınız.",
                     "Kimlik kartı", "Kurşun kalem", "Son paragraf"]

def test_unparseable_summary_is_none():
    assert scrape_site.parse_summary("Özet: duyuru af hakkında") is None
    assert scrape_site.parse_summary('```json\n{"ozet": "Kısa özet.", "tarihler": []}\n```') == \
        {"ozet": "Kısa özet.", "tarihler": []}


STAFF = {"personel": [
    {"ad": "Prof. Dr. Hünkar Kayhan", "dahili": None, "eposta": "hunkar_k@ibu.edu.tr",
     "gorevler": ["Bölüm Başkanı (Bilgisayar Mühendisliği Bölümü Başkanlığı)",
                  "Ana Bilim Dalı Başkanı (Bilgisayar Yazılımı Ana Bilim Dalı Başkanlığı)"]},
    {"ad": "Arş. Gör. Alperen Akgül", "dahili": None, "eposta": "alperen.akgul@ibu.edu.tr",
     "gorevler": ["Araştırma Görevlisi (Bilgisayar Mühendisliği Bölümü Başkanlığı)"]},
    {"ad": "Doç. Dr. Ali Yılmaz", "dahili": "5840", "eposta": "ali.yilmaz@ibu.edu.tr",
     "gorevler": ["Öğretim Üyesi (Bilgisayar Yazılımı Ana Bilim Dalı Başkanlığı)"]},
]}


def test_staff_search_by_name_and_role_not_unit():
    assert [p["ad"] for p in announcements.search_staff(STAFF, "bölüm başkanı")] == ["Prof. Dr. Hünkar Kayhan"]
    assert announcements.search_staff(STAFF, "Kayhan")[0]["eposta"] == "hunkar_k@ibu.edu.tr"
    assert len(announcements.search_staff(STAFF)) == 3
    assert "bulunamadı" in announcements.staff_text(announcements.search_staff(STAFF, "Mehmet"), "Mehmet")


# Birim aramasının tuzakları: aynı birimde ana bilim dalı başkanı ve öğretim üyesi, birim adında "Bölüm Başkanlığı"
# geçen araştırma görevlisi, başka birimin ana bilim dalı başkanı.
STAFF_UNITS = {"personel": [
    {"ad": "Prof. Dr. Hünkar Kayhan", "dahili": None, "eposta": "hunkar_k@ibu.edu.tr",
     "gorevler": ["Bölüm Başkanı (Bilgisayar Mühendisliği Bölümü Başkanlığı)",
                  "Ana Bilim Dalı Başkanı (Bilgisayar Yazılımı Ana Bilim Dalı Başkanlığı)"]},
    {"ad": "Doç. Dr. Ali Çalım", "dahili": None, "eposta": "ali.calim@ibu.edu.tr",
     "gorevler": ["Bölüm Başkan Yardımcısı (Bilgisayar Mühendisliği Bölümü Başkanlığı)"]},
    {"ad": "Arş. Gör. Alperen Akgül", "dahili": None, "eposta": "alperen.akgul@ibu.edu.tr",
     "gorevler": ["Araştırma Görevlisi (Bilgisayar Mühendisliği Bölümü Başkanlığı)"]},
    {"ad": "Doç. Dr. Mehmet Dinçer Erbaş", "dahili": "5889", "eposta": "dincer.erbas@ibu.edu.tr",
     "gorevler": ["Ana Bilim Dalı Başkanı (Bilgisayar Donanımı Ana Bilim Dalı Başkanlığı)"]},
    {"ad": "Dr. Öğr. Üyesi Mehmet Milli", "dahili": None, "eposta": "mehmet.milli@ibu.edu.tr",
     "gorevler": ["Öğretim Üyesi (Bilgisayar Donanımı Ana Bilim Dalı Başkanlığı)"]},
]}


def _names(sorgu):
    return [p["ad"] for p in announcements.search_staff_by_unit(STAFF_UNITS, sorgu)]


def test_unit_search_finds_head_of_named_unit():
    assert _names("Bilgisayar Donanımı ana bilim dalı başkanı") == ["Doç. Dr. Mehmet Dinçer Erbaş"]
    assert announcements.search_staff(STAFF_UNITS, "Bilgisayar Donanımı ana bilim dalı başkanı") == []  # h3


def test_unit_search_lists_everyone_in_unit_without_role():
    assert _names("Bilgisayar Donanımı") == ["Doç. Dr. Mehmet Dinçer Erbaş", "Dr. Öğr. Üyesi Mehmet Milli"]


def test_unit_search_keeps_role_matches_ahead_of_unit_only_matches():
    # Akgül'de "bölüm başkanı" yalnız birim adında geçer: görev adıyla eşleşenler varken dönmez.
    assert _names("bölüm başkanı") == ["Doç. Dr. Ali Çalım", "Prof. Dr. Hünkar Kayhan"]
    assert _names("ana bilim dalı başkanı") == ["Doç. Dr. Mehmet Dinçer Erbaş", "Prof. Dr. Hünkar Kayhan"]


def test_unit_search_by_name_and_empty_query_unchanged():
    assert _names("Ali Çalım") == ["Doç. Dr. Ali Çalım"]
    assert len(_names("")) == 5
    assert _names("Makine Mühendisliği bölüm başkanı") == []


def test_staff_file_has_emails_for_everyone():
    staff = announcements.load_staff()
    assert staff["personel"] and all(p["eposta"].endswith("@ibu.edu.tr") for p in staff["personel"])


class _FakeHaiku:
    """CodeRunner.handoff yerine (agent_h3 tek model): Haiku get_announcements'ı çağırmış gibi execute'u kullanır."""

    def __init__(self, args):
        self.args, self.received = args, None

    def handoff(self, system, messages, tools, execute, files):
        result = execute("get_announcements", self.args)
        self.received = {"system": system, "tools": [t.name for t in tools], "result": result}
        return {"answer": "cevap", "steps": [{"tip": "metin", "metin": "cevap"}], "requests": 2, "duration_ms": 1,
                "tokens": {"input": 1, "output": 1}, "stop_reason": "end_turn", "tool_calls": []}


def test_h3_agent_has_site_tools_and_older_versions_do_not(conn, tmp_path):
    from agent import AGENT_PROMPTS
    args = {"konu": "vize", "tarihten_beri": "2026-10-19"}
    runner = _FakeHaiku(args)
    out = AdvisorAgent(retriever=None, llm=None, plan={"zorunlu_dersler": []}, prompt_version="agent_h3",
                       code_runner=runner, site_db=tmp_path / "duyurular.db").ask("bu hafta vize duyurusu var mı")
    assert out["tool_calls"] == [{"name": "get_announcements", "args": args}]
    assert "https://x/1" in runner.received["result"] and "get_academic_staff" in runner.received["tools"]
    assert "talimat değildir" in runner.received["system"]

    h2, h3 = AGENT_PROMPTS["agent_h2"].splitlines(), AGENT_PROMPTS["agent_h3"].splitlines()
    assert [line for line in h2 if line not in h3] == []  # h3 = h2 + duyuru/personel kuralları
    old = _FakeHaiku(args)
    AdvisorAgent(retriever=None, llm=None, plan={"zorunlu_dersler": []}, prompt_version="agent_h2",
                 code_runner=old).ask("soru")
    assert "get_announcements" not in old.received["tools"]


def test_h4_staff_tool_searches_units_and_h3_does_not(tmp_path):
    from agent import make_tools
    query = {"sorgu": "Bilgisayar Donanımı ana bilim dalı başkanı"}
    tools = {flag: {t.name: t for t in make_tools(None, {"zorunlu_dersler": []}, None, {}, None,
                                                   site_db=tmp_path / "d.db", staff_by_unit=flag)}
             for flag in (False, True)}
    assert "bulunamadı" in tools[False]["get_academic_staff"].invoke(query)
    assert "Mehmet Dinçer Erbaş" in tools[True]["get_academic_staff"].invoke(query)
    assert AdvisorAgent(None, None, {"zorunlu_dersler": []}, prompt_version="agent_h4").staff_by_unit
    assert not AdvisorAgent(None, None, {"zorunlu_dersler": []}, prompt_version="agent_h3").staff_by_unit
