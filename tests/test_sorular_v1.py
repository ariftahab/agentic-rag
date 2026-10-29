"""İlk agentic soru seti (eval/sorular_v1.json): biçim, kanıtlar ve elle hesaplanan sayılar. Ağ ve LLM yok.

Beklenen sayılar fixture'lardan elle hesaplandı; bu test onları deterministik modülle (graduation) karşılaştırır.
Fark çıkarsa önce hangisinin yanlış olduğu bulunur, sayı koddan kopyalanmaz. Ayrıca beklenen tool doğru
argümanlarla çağrıldığında anahtar kelimelerin tool çıktısında bulunduğu kontrol edilir: çıktıda olmayan bir
anahtar kelimeyi agent'tan beklemek, agent'ı haksız yere cezalandırır.
"""
import importlib.util
import re
from datetime import date

import pytest

import config
from tools import graduation
import ingest
from agent import make_tools
from experiments import evaluation as ev
from experiments import run

DATASET, _ = ev.load_dataset("sorular_v1")
ITEMS = DATASET["sorular"]
PLAN = graduation.load_plan()
# Ders programı soruları gün adıyla; çıktı ders döneminde bir pazartesiden (2026-11-02) hesaplanır.
TODAY = date(2026, 11, 2)
KUME = re.compile(r"kume(\d+)_akts")


def _by_id(item_id):
    return next(q for q in ITEMS if q["id"] == item_id)


def test_ids_are_unique_and_every_item_has_group_and_category():
    ids = [q["id"] for q in ITEMS]
    assert len(ids) == len(set(ids))
    assert all(q.get("grup") and q.get("kategori") for q in ITEMS)


def test_agent_fields_and_transcript_fixtures_are_valid():
    assert ev.validate_agent_items(DATASET, ev.TRANSCRIPTS_DIR) == []


def test_regulation_evidence_exists_in_the_source_documents():
    if importlib.util.find_spec("pypdf"):
        assert ev.validate_dataset(DATASET) == []
        return
    # pypdf kurulu değilse (validate_dataset iki okuyucuyu da ister) en azından pdfplumber metni kontrol edilir.
    texts, missing = {}, []
    for item in filter(ev.answerable, ITEMS):
        for belge, alts in ev.sources(item):
            if belge not in texts:
                texts[belge] = ev.norm(" ".join(ingest.read_pages(config.DATA_DIR / belge, "pdfplumber")))
            missing += [(item["id"], part) for alt in alts for part in alt if ev.norm(part) not in texts[belge]]
    assert missing == []


def _confirmations(item, transcript):
    """Sorunun oturum teyitleri: 'teyitler' + beklenen confirm_equivalence çağrısının kaydedeceği eşleme."""
    confirmations = dict(item.get("teyitler") or {})
    for call in item.get("beklenen_tool_lar", []):
        if call["name"] == "confirm_equivalence":
            args = call["args"]
            course = next(c for c in transcript["dersler"] if c["ad"] == args["transcript_course"])
            target = next(c for c in graduation.plan_courses(PLAN) if c["ad"] == args["plan_course"])
            confirmations[course["kod"]] = target["kod"]
    return confirmations


def _module_value(key, item, report, transcript, confirmations):
    m = KUME.fullmatch(key)
    if m:
        sets = next(c["args"]["course_sets"] for c in item["beklenen_tool_lar"] if c["name"] == "simulate_courses")
        simulated = graduation.simulate(transcript, PLAN, sets[int(m.group(1)) - 1], confirmations=confirmations)
        return simulated["devam_edenler_gecilirse_akts"]
    rules = {r["id"]: r for r in report["kurallar"]}
    values = {
        "gecilen_akts": report["gecilen_akts"],
        "devam_edenler_gecilirse_akts": report["devam_edenler_gecilirse_akts"],
        "gno": report["gno"],
        "secmeli_gecilen_akts": report["secmeli"]["gecilen_akts"],
        "secmeli_acik_akts": report["secmeli"]["acik_akts"],
        "tamamlanan_zorunlu_ders": rules["zorunlu_dersler"]["mevcut"],
    }
    return values[key]  # bilinmeyen anahtar: test hata verir, her sayı bir yerde kontrol edilmeli


@pytest.mark.parametrize("item", [q for q in ITEMS if "beklenen_sayilar" in q], ids=lambda q: q["id"])
def test_hand_calculated_numbers_match_the_graduation_module(item):
    transcript = ev.load_transcript(item["transkript"])
    confirmations = _confirmations(item, transcript)
    report = graduation.evaluate(transcript, PLAN, confirmations=confirmations)
    actual = {k: _module_value(k, item, report, transcript, confirmations) for k in item["beklenen_sayilar"]}
    assert actual == item["beklenen_sayilar"]


TOOL_CHECKED = [q for q in ITEMS if q.get("beklenen") and q.get("beklenen_tool_lar")
                and q["beklenen_tool_lar"][0]["name"] != "search_regulations"
                and (q["beklenen_tool_lar"][0]["name"] != "simulate_courses" or "args" in q["beklenen_tool_lar"][0])]


@pytest.mark.parametrize("item", TOOL_CHECKED, ids=lambda q: q["id"])
def test_expected_keywords_are_in_the_expected_tool_output(item):
    transcript = ev.load_transcript(item["transkript"]) if item.get("transkript") else None
    tools = {t.name: t for t in make_tools(None, PLAN, transcript, dict(item.get("teyitler") or {}), TODAY)}
    call = item["beklenen_tool_lar"][0]
    output = tools[call["name"]].invoke(call.get("args", {}))
    assert run.answer_ok(item, output), output


def test_unknown_course_in_what_if_is_reported_as_not_in_the_plan():
    item = _by_id("W03")
    tools = {t.name: t for t in make_tools(None, PLAN, ev.load_transcript(item["transkript"]), {}, TODAY)}
    output = tools["simulate_courses"].invoke({"course_sets": [["Oyun Programlama"]]})
    assert run.answer_ok(item, output) and "bulunamayan" in output
