"""Anlık tool soru seti (eval/sorular_v2.json): biçim, v1'den devralınan sorular ve elle hesaplanan cevaplar. Ağ ve LLM yok.

Anlık tool sorularının cevabını deterministik bir modül vermez (mevcut tool'ların yetmediği hesaplar); bu yüzden
sayılar burada mevzuattaki formülle ve ham veriden (transkript fixture'ları, öğretim planı, ders programı) bağımsız
olarak yeniden hesaplanır. Ortalama formülü, fixture'ların UBYS özetindeki GNO'yu tutturarak doğrulanır.
"""
import math
from datetime import date

import pytest

import graduation
import schedule
from agent import make_tools
from experiments import evaluation as ev
from experiments import run

DATASET, _ = ev.load_dataset("sorular_v2")
V1, _ = ev.load_dataset("sorular_v1")
ITEMS = {q["id"]: q for q in DATASET["sorular"]}
PLAN = graduation.load_plan()
SCHEDULE = schedule.load_schedule()["dersler"]
TODAY = date(2026, 11, 2)

# Önlisans-Lisans Eğitim-Öğretim Yönetmeliği Madde 27 (puan aralıkları ve katsayılar) ve Madde 26 (vize %40, final %60).
GRADES = [(90, "AA", 4.0), (85, "BA", 3.75), (80, "BB", 3.5), (75, "CB", 3.25), (65, "CC", 3.0), (60, "DC", 2.5),
          (55, "DD", 2.0), (50, "FD", 1.0), (0, "FF", 0.0)]
COEFF = {letter: c for _, letter, c in GRADES}


def letter(score):
    return next(g for low, g, _ in GRADES if score >= low)


def term_key(course):
    return course["yil"], 0 if course["donem"] == "Güz" else 1


def gno(transcript, new_grades=None):
    """Madde 28: AKTS ağırlıklı, tekrar edilen derste son not, iki haneye kesilir (yuvarlama yok). new_grades bu
    dönemin derslerine (ya da tekrar alınan bir derse) verilen varsayımsal harf notları."""
    last = {}
    for c in sorted(transcript["dersler"], key=term_key):
        grade = c["not"] if c["not"] is not None else (new_grades or {}).get(c["ad"])
        if grade in COEFF:
            last[c["ad"]] = (c["akts"], grade)
    for name, grade in (new_grades or {}).items():  # tekrar (not yükseltme): son not geçerli
        if name in last:
            last[name] = (last[name][0], grade)
    total = sum(a for a, _ in last.values())
    return math.floor(sum(a * COEFF[g] for a, g in last.values()) / total * 100) / 100


def transcript(item_id):
    return ev.load_transcript(ITEMS[item_id]["transkript"])


def hours(rows):
    return sum(len(r["saatler"]) for r in rows)


def test_v1_items_are_kept_except_two_corrected_expectations():
    v2 = {q["id"]: q for q in DATASET["sorular"]}
    changed = [q["id"] for q in V1["sorular"] if v2[q["id"]] != q]
    assert changed == ["B01", "T14"]
    assert v2["B01"]["beklenen_tool_lar"] == [{"name": "get_graduation_requirements"}]
    assert v2["B01"]["beklenen"] == ["240", "2,00|2.00"]  # mevzuattaki alt sınır, v1 ile aynı
    assert v2["T14"]["beklenen_tool_lar"] == []


def test_ids_are_unique_and_agent_fields_are_valid():
    ids = [q["id"] for q in DATASET["sorular"]]
    assert len(ids) == len(set(ids)) == 66
    assert all(q.get("grup") and q.get("kategori") for q in DATASET["sorular"])
    assert ev.validate_agent_items(DATASET, ev.TRANSCRIPTS_DIR) == []


@pytest.mark.parametrize("name", ["ucuncu_sinif_2024", "uzatma_2021", "erasmus_2023"])
def test_gno_formula_reproduces_the_transcript_summary(name):
    t = ev.load_transcript(name)
    assert gno(t) == t["obs_ozet"]["gno"]


def test_score_to_grade_and_average_questions():
    assert letter(62) == "DC" and gno(transcript("A01"), {"Algoritma Analizi": "DC"}) == ITEMS["A01"]["beklenen_sayilar"]["ortalama"]
    # Vize 40: 0,4·40 + 0,6·final ≥ 55 (DD) → final ≥ 65
    assert min(f for f in range(101) if letter(0.4 * 40 + 0.6 * f) == "DD") == 65
    score = 0.4 * 50 + 0.6 * 60
    assert letter(score) == "DD" and gno(transcript("A03"), {"Algoritma Analizi": "DD"}) == 2.02 > 2.0
    first = {"Matematik I": letter(85), "Fizik I": letter(72), "Kimya": letter(58)}
    assert first == {"Matematik I": "BA", "Fizik I": "CC", "Kimya": "DD"}
    assert gno(transcript("A04"), first) == ITEMS["A04"]["beklenen_sayilar"]["ortalama"]


def test_minimum_score_to_keep_the_average_at_three():
    t = transcript("A05")
    minimum = min(s for s in range(101) if gno(t, {"Algoritma Analizi": letter(s)}) >= 3.0)
    assert minimum == ITEMS["A05"]["beklenen_sayilar"]["puan"]
    assert gno(t, {"Algoritma Analizi": letter(minimum - 1)}) == 2.99


def test_transcript_query_answers():
    dd = [c["ad"] for c in transcript("A06")["dersler"] if c["not"] == "DD"]
    assert sorted(dd) == ["Elektrik Devreleri", "Kimya"] and len(dd) == ITEMS["A06"]["beklenen_sayilar"]["dd_ders"]
    failed = {c["ad"] for c in transcript("A07")["dersler"] if c["not"] in ("FF", "FD")}
    assert failed == {"Fizik II", "Diferansiyel Denklemler"}
    attempts = [c["not"] for c in transcript("A08")["dersler"] if c["ad"] == "Algoritma Analizi"]
    assert attempts == ["FF", "FD", "FF", None] and len(attempts) == ITEMS["A08"]["beklenen_sayilar"]["alis"]
    terms = {(c["yil"], c["donem"]) for c in transcript("A09")["dersler"]}
    assert len(terms) == ITEMS["A09"]["beklenen_sayilar"]["donem"]


def test_courses_left_after_this_term():
    t = transcript("A10")
    report = graduation.evaluate(t, PLAN)
    mandatory = [c for c in report["eksik_zorunlu"] if c["durum"] != "devam_ediyor"]
    pools = {s["ad"]: {c["ad"]: c["akts"] for c in s["havuz"]} for s in PLAN["secmeli_yuvalar"]}
    electives = 0
    for slot in report["secmeli"]["yuvalar"]:
        if not slot["devam_edenler_gecilirse_doldu"]:
            missing = slot["akts"] - sum(pools[slot["ad"]][c] for c in slot["devam_edenler_gecilirse_dersler"])
            electives += math.ceil(missing / min(pools[slot["ad"]].values()))
    assert (len(mandatory), electives) == (9, 12)
    assert len(mandatory) + electives == ITEMS["A10"]["beklenen_sayilar"]["ders"]


def test_schedule_answers():
    current = {c["ad"] for c in transcript("A11")["dersler"] if c["not"] is None}
    rows = [r for r in SCHEDULE if r["ders"] in current]
    friday = {(r["ders"], r["sube"]) for r in rows if r["gun"] == "Cuma"}
    assert friday == {("Makine Öğrenmesine Giriş", "B")}
    assert {r["sube"] for r in rows if r["ders"] == "Makine Öğrenmesine Giriş" and r["gun"] != "Cuma"} == {"A"}

    def slots(rows_):
        return {(r["gun"], s) for r in rows_ for s in r["saatler"]}
    os_b = slots(r for r in rows if r["ders"] == "İşletim Sistemleri I" and r["sube"] == "B")
    clashes = {(r["ders"], r["sube"]) for r in rows if r["ders"] != "İşletim Sistemleri I" and slots([r]) & os_b}
    assert clashes == {("Mikroişlemciler", "A")}

    first = {c["ad"] for c in transcript("A13")["dersler"] if c["not"] is None}
    by_section = {s: hours(r for r in SCHEDULE if r["ders"] in first and r["sube"] in (s, None)) for s in "AB"}
    assert by_section == {"A": 27, "B": 27} and ITEMS["A13"]["beklenen_sayilar"]["saat"] == 27

    room = {r["ders"] for r in SCHEDULE if r["derslik"] == "327" and r["gun"] == "Pazartesi"}
    assert room == {"Algoritma Analizi"}


def test_code_tool_is_expected_only_in_the_code_tool_group():
    for q in DATASET["sorular"]:
        expected = any(e["name"] == run.CODE_TOOL for e in q.get("beklenen_tool_lar") or [])
        assert expected == (q["grup"] == "anlik_tool" and q["kategori"] != "transkript_yok"), q["id"]


# Sınırdaki soruların cevabı mevcut tool'un çıktısında (G06'da sayı LLM'in saymasıyla çıkar, çıktıda yazmaz).
BOUNDARY = [q for q in DATASET["sorular"] if q["grup"] == "sinir_tool" and q["id"] != "G06"]


@pytest.mark.parametrize("item", BOUNDARY, ids=lambda q: q["id"])
def test_boundary_answers_are_in_the_existing_tool_output(item):
    t = ev.load_transcript(item["transkript"]) if item.get("transkript") else None
    tools = {tl.name: tl for tl in make_tools(None, PLAN, t, {}, TODAY, with_requirements=True,
                                              transcript_detail=True)}
    call = next(c for c in item["beklenen_tool_lar"] if c["name"] != "search_regulations")
    output = tools[call["name"]].invoke(call.get("args", {}))
    if "beklenen" in item and item["id"] != "G02":  # şeref eşiği mevzuat aramasından gelir, transkript raporunda yok
        assert run.answer_ok(item, output), output
    for value in (item.get("beklenen_sayilar") or {}).values():
        assert value in ev.numbers_in_text(output)


def test_offered_seventh_elective_count():
    tools = {tl.name: tl for tl in make_tools(None, PLAN, None, {}, TODAY)}
    output = tools["get_offered_electives"].invoke({"yuva": "Seçmeli VII"})
    offered = {r["ders"] for r in SCHEDULE if r["yuva"] == "Seç VII"}
    assert len(offered) == 7 and all(name in output for name in offered)
