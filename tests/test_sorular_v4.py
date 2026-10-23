"""Soru seti v4 (eval/sorular_v4.json): v3 + rapordan cevaplanan transkript soruları + devir soruları. Ağ ve LLM yok.

agent_v12'de analyze_transcript ders geçmişini de verir (graduation.history_text). Burada iki şey kontrol edilir:
beklentisi analyze_transcript'e taşınan soruların cevabı raporda gerçekten var; yeni devir sorularının cevabı raporda
yok (ek hesap gerekir) ve beklenen sayılar ham transkriptten yeniden hesaplanınca aynı çıkar.
"""
from datetime import date

import pytest

import gpa
import graduation
from agent import make_tools
from experiments import evaluation as ev
from experiments import run

V4, _ = ev.load_dataset("sorular_v4")
V3, _ = ev.load_dataset("sorular_v3")
ITEMS = {q["id"]: q for q in V4["sorular"]}
PLAN = graduation.load_plan()
RULES = graduation.load_rules(regulation_only=True)
REPORT_IDS = ["A06", "A07", "A08", "A09", "A10"]
HANDOFF_IDS = ["D01", "D02", "D03", "D04"]


def transcript(item_id):
    return ev.load_transcript(ITEMS[item_id]["transkript"])


def report(item_id):
    tools = {t.name: t for t in make_tools(None, PLAN, transcript(item_id), {}, date(2026, 11, 2), rules=RULES,
                                           with_history=True)}
    return tools["analyze_transcript"].invoke({})


def average(courses):
    graded = [c for c in courses if c["not"] in gpa.COEFF]
    return gpa._truncate(sum(c["akts"] * gpa.COEFF[c["not"]] for c in graded) / sum(c["akts"] for c in graded))


def test_only_the_report_questions_change_and_handoff_questions_are_added():
    v3 = {q["id"]: q for q in V3["sorular"]}
    assert sorted(set(ITEMS) - set(v3)) == HANDOFF_IDS and not set(v3) - set(ITEMS)
    assert sorted(i for i in v3 if ITEMS[i] != v3[i]) == REPORT_IDS
    for i in REPORT_IDS:
        assert ITEMS[i]["beklenen_tool_lar"] == [{"name": "analyze_transcript"}]
        same = ("beklenen", "beklenen_sayilar", "soru", "transkript")
        assert {k: ITEMS[i].get(k) for k in same} == {k: v3[i].get(k) for k in same}
    for i in HANDOFF_IDS:
        assert [e["name"] for e in ITEMS[i]["beklenen_tool_lar"]] == ["analyze_transcript", run.CODE_TOOL]
    assert ev.validate_agent_items(V4, ev.TRANSCRIPTS_DIR) == []


@pytest.mark.parametrize("item_id", REPORT_IDS)
def test_report_questions_are_answered_by_the_report(item_id):
    item, output = ITEMS[item_id], report(item_id)
    if "beklenen" in item:
        assert run.answer_ok(item, output), output
    for value in (item.get("beklenen_sayilar") or {}).values():
        assert value in ev.numbers_in_text(output), output


@pytest.mark.parametrize("item_id", HANDOFF_IDS)
def test_handoff_answers_are_not_in_the_report(item_id):
    found = ev.numbers_in_text(report(item_id))
    for value in ITEMS[item_id]["beklenen_sayilar"].values():
        if item_id == "D02":  # 3.48 raporda 4. ve 5. dönemin YANO'su (tesadüf); birinci yılın ortalaması yok
            lines = [line for line in report(item_id).splitlines() if "3.48" in line]
            assert all(line.startswith(("- 4. dönem", "- 5. dönem")) for line in lines)
            continue
        assert value not in found


def test_handoff_answers_from_the_raw_transcript():
    third = transcript("D01")["dersler"]
    assert average([c for c in third if c["ad"] in ("Matematik I", "Matematik II", "Lineer Cebir")]) == \
        ITEMS["D01"]["beklenen_sayilar"]["ortalama"]
    erasmus = transcript("D02")
    first_year = [c for c in erasmus["dersler"] if c["yil"] == erasmus["kayit_yili"]]  # güz + bahar, tekrar yok
    assert average(first_year) == ITEMS["D02"]["beklenen_sayilar"]["ortalama"]
    last = {}
    for c in sorted(erasmus["dersler"], key=graduation._term_key):
        if c["not"] in gpa.COEFF:
            last[c["kod"]] = c  # Madde 28/3: tekrar edilen derste son not
    assert average(last.values()) == erasmus["obs_ozet"]["gno"] == 3.5
    assert average([c for c in last.values() if not c["erasmus"]]) == ITEMS["D04"]["beklenen_sayilar"]["ortalama"]
    passing = set(graduation._rule(RULES, "gecer_notlar")["deger"])
    first = {}
    for c in sorted(transcript("D03")["dersler"], key=graduation._term_key):
        first.setdefault(c["kod"], c)
    failed_first = sorted(c["ad"] for c in first.values() if c["not"] not in passing)
    assert failed_first == ["Algoritma Analizi", "Biyoenformatik", "Matematik I"]
    assert len(first) - len(failed_first) == ITEMS["D03"]["beklenen_sayilar"]["ders"]
