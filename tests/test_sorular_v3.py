"""Soru seti v3 (eval/sorular_v3.json): v2'den tek fark ortalama sorularının beklenen tool'u. Ağ ve LLM yok.

Sayıların veriden hesabı tests/test_sorular_v2.py'de (sorular aynı); burada beklenen tool'un doğru argümanlarla
çağrıldığında beklenen sayıları ve anahtar kelimeleri gerçekten döndürdüğü kontrol edilir.
"""
from datetime import date

import pytest

from tools import graduation
from agent import make_tools
from experiments import evaluation as ev
from experiments import run

V3, _ = ev.load_dataset("sorular_v3")
V2, _ = ev.load_dataset("sorular_v2")
ITEMS = {q["id"]: q for q in V3["sorular"]}
PLAN = graduation.load_plan()
GRADE_IDS = ["A01", "A02", "A03", "A04", "A05"]

# Agent'ın bu sorularda göndermesi beklenen argümanlar (soru metninden).
ARGS = {
    "A01": {"dersler": [{"ad": "Algoritma Analizi", "puan": 62}]},
    "A02": {"dersler": [{"vize": 40}]},
    "A03": {"dersler": [{"ad": "Algoritma Analizi", "vize": 50, "final": 60}], "hedef_gno": 2.0},
    "A04": {"dersler": [{"ad": "Matematik I", "puan": 85}, {"ad": "Fizik I", "puan": 72}, {"ad": "Kimya", "puan": 58}]},
    "A05": {"dersler": [{"ad": "Algoritma Analizi"}], "hedef_gno": 3.0},
}


def test_only_the_grade_questions_change():
    v2 = {q["id"]: q for q in V2["sorular"]}
    changed = sorted(i for i, q in ITEMS.items() if q != v2[i])
    assert changed == GRADE_IDS
    for i in GRADE_IDS:
        assert ITEMS[i]["beklenen_tool_lar"] == [{"name": "simulate_grades"}]
        assert {k: v for k, v in ITEMS[i].items() if k not in ("grup", "kategori", "beklenen_tool_lar")} == \
               {k: v for k, v in v2[i].items() if k not in ("grup", "kategori", "beklenen_tool_lar")}
    assert ev.validate_agent_items(V3, ev.TRANSCRIPTS_DIR) == []


def test_code_tool_is_no_longer_expected_in_grade_questions():
    expected = [q["id"] for q in V3["sorular"] if any(e["name"] == run.CODE_TOOL for e in q.get("beklenen_tool_lar") or [])]
    assert not set(GRADE_IDS) & set(expected) and len(expected) == 9


@pytest.mark.parametrize("item_id", GRADE_IDS)
def test_grades_tool_returns_the_expected_answer(item_id):
    item = ITEMS[item_id]
    transcript = ev.load_transcript(item["transkript"]) if item.get("transkript") else None
    tools = {t.name: t for t in make_tools(None, PLAN, transcript, {}, date(2026, 11, 2), with_grades=True)}
    output = tools["simulate_grades"].invoke(ARGS[item_id])
    if "beklenen" in item:
        assert run.answer_ok(item, output), output
    for value in (item.get("beklenen_sayilar") or {}).values():
        assert value in ev.numbers_in_text(output), output
