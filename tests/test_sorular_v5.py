"""Soru seti v5 (eval/sorular_v5.json): v4 + duyuru ve personel soruları. Ağ ve LLM yok.

Duyuru cevapları eval/duyurular_v1.json anlık görüntüsünden gelir; burada beklenen cevapların o görüntüden kurulan
deponun tool çıktısında gerçekten bulunduğu ve v4'ün geri kalanının değişmediği kontrol edilir.
"""
from datetime import date

import pytest

import graduation
from agent import make_tools
from experiments import evaluation as ev
from experiments import run

V5, _ = ev.load_dataset("sorular_v5")
V4, _ = ev.load_dataset("sorular_v4")
ITEMS = {q["id"]: q for q in V5["sorular"]}
NEW_IDS = [q["id"] for q in V5["sorular"] if q["grup"] in ("duyuru", "personel")]


@pytest.fixture(scope="module")
def tools():
    db = run.site_snapshot(V5["duyuru_deposu"])
    return {t.name: t for t in make_tools(None, graduation.load_plan(), None, {}, date.fromisoformat(V5["bugun"]),
                                          site_db=db)}


def test_v4_questions_unchanged_except_department_head():
    old = {q["id"]: q for q in V4["sorular"]}
    assert set(old) <= set(ITEMS)
    assert [i for i in old if old[i] != ITEMS[i]] == ["C03"]
    assert ITEMS["C03"]["beklenen"] == ["Hünkar Kayhan"] and not ITEMS["C03"].get("cevapsiz")


def test_snapshot_has_38_announcements(tools):
    assert "38 duyuru" in tools["get_announcements"].invoke({"konu": "uzay asansörü"})


@pytest.mark.parametrize("item_id", ["N01", "N02", "N03", "N04", "N06", "N07"])
def test_announcement_answer_is_in_tool_output(tools, item_id):
    topics = {"N01": "yemek bursu", "N02": "seviye tespit sınavı", "N03": "öğretim üyesi alımı", "N04": "2209",
              "N06": "kimya mühendisliği", "N07": "muafiyet sınavı"}
    out = run._normalize_answer(tools["get_announcements"].invoke({"konu": topics[item_id]}))
    for kw in ITEMS[item_id]["beklenen"]:
        assert any(run._normalize_answer(alt) in out for alt in kw.split("|")), (item_id, kw)


def test_this_week_starts_on_monday_of_fixed_day(tools):
    since = ITEMS["N08"]["beklenen_tool_lar"][0]["args"]["tarihten_beri"]
    assert date.fromisoformat(since).weekday() == 0
    assert (date.fromisoformat(V5["bugun"]) - date.fromisoformat(since)).days < 7
    assert "Yemek Bursu" in tools["get_announcements"].invoke({"tarihten_beri": since})


def test_spring_transfer_is_not_announced(tools):
    out = tools["get_announcements"].invoke({"konu": "bahar yatay geçiş"})
    assert "Bahar" not in out


@pytest.mark.parametrize("item_id,query", [("C03", "bölüm başkanı"), ("K01", "bölüm başkan yardımcısı"),
                                           ("K02", "ana bilim dalı başkanı"), ("K03", "İsmail Hakkı Parlak")])
def test_staff_answer_is_in_tool_output(tools, item_id, query):
    out = tools["get_academic_staff"].invoke({"sorgu": query})
    assert ITEMS[item_id]["beklenen"][0] in out


def test_new_questions_expect_site_tools():
    for i in NEW_IDS:
        names = {t["name"] for t in ITEMS[i]["beklenen_tool_lar"]}
        assert names & {"get_announcements", "get_academic_staff"}, i
