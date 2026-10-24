"""agent_h6'nın tool'ları: transkriptin bir kısmından ortalama (gpa.subset_average), bütün haftada ders programı
araması (schedule.find_lessons) ve şube planı / çakışma kontrolü (schedule.plan_sections). Ağ ve LLM yok.

Haiku bu hesapları deneylerde (EXP-005 - EXP-013) her seferinde kodla yeniden yazdı; beklenen sayılar sorular_v5'teki
ilgili soruların cevapları.
"""
from datetime import date

import gpa
import graduation
import schedule
from agent import AdvisorAgent, make_tools
from experiments import evaluation as ev

PLAN = graduation.load_plan()
DATA = schedule.load_schedule()


def _course(ad, yil, donem, akts, not_, erasmus=False):
    return {"yil": yil, "donem": donem, "kod": ad, "ad": ad, "kredi": 3, "akts": akts, "not": not_, "erasmus": erasmus}


# --- Transkriptin bir kısmından ortalama ---

def test_average_of_named_courses():
    result = gpa.subset_average(ev.load_transcript("ucuncu_sinif_2024"),
                                courses=["Matematik I", "Matematik II", "Lineer Cebir"])
    assert result["gno"] == 2.9
    assert sorted(c["ad"] for c in result["dersler"]) == ["Lineer Cebir", "Matematik I", "Matematik II"]


def test_course_names_with_digits_match_roman_numerals_only_exactly():
    result = gpa.subset_average(ev.load_transcript("ucuncu_sinif_2024"), courses=["matematik 1", "Fizik 9"])
    assert [c["ad"] for c in result["dersler"]] == ["Matematik I"]
    assert result["bulunamayan"] == ["Fizik 9"]


def test_average_at_the_end_of_first_year():
    assert gpa.subset_average(ev.load_transcript("erasmus_2023"), years=1)["gno"] == 3.48


def test_average_without_erasmus_courses():
    assert gpa.subset_average(ev.load_transcript("erasmus_2023"), exclude_erasmus=True)["gno"] == 3.49


def test_repeated_course_counts_with_its_last_grade_within_the_subset():
    transcript = {"dersler": [_course("Fizik I", 2023, "Güz", 5, "FF"), _course("Fizik I", 2024, "Güz", 5, "BB"),
                              _course("Kimya", 2023, "Güz", 5, "AA")]}
    assert gpa.subset_average(transcript)["gno"] == 3.75                # (3,50 + 4,00) / 2
    assert gpa.subset_average(transcript, years=1)["gno"] == 2.0       # ilk yılın sonunda Fizik I hâlâ FF


def test_regulation_coefficients_count_incomplete_as_ff():
    transcript = {"dersler": [_course("Fizik I", 2023, "Güz", 5, "E/I"), _course("Kimya", 2023, "Güz", 5, "AA")]}
    assert gpa.subset_average(transcript, coeff=gpa.COEFF_REGULATION)["gno"] == 2.0


def test_average_text_cites_regulation_and_lists_courses():
    text = gpa.subset_text(gpa.subset_average(ev.load_transcript("erasmus_2023"), exclude_erasmus=True))
    assert "3,49" in text and "Madde 28" in text and "Erasmus" in text


# --- Bütün haftada ders programı araması ---

def test_find_lessons_by_room_and_day():
    rows = schedule.find_lessons(DATA, room="327", day="Pazartesi")
    assert "Algoritma Analizi" in {r["ders"] for r in rows}
    assert all(r["gun"] == "Pazartesi" and r["derslik"] == "327" for r in rows)


def test_find_lessons_by_course_over_the_whole_week():
    rows = schedule.find_lessons(DATA, courses=["yapay zeka"])
    assert rows and {r["ders"] for r in rows} == {"Yapay Zeka"}
    text = schedule.week_text(rows, DATA)
    assert "Çarşamba" in text and "15:30" in text


def test_find_lessons_room_name_and_section():
    rows = schedule.find_lessons(DATA, room="bilgisayar laboratuvarı", section="B")
    assert rows and all(r["derslik"] == "B.L." and r["sube"] == "B" for r in rows)


# --- Şube planı ve çakışma ---

def test_fixed_section_reports_the_clashing_course():
    transcript = ev.load_transcript("ucuncu_sinif_2024")
    plan = schedule.plan_sections(DATA, codes=schedule.current_course_codes(transcript),
                                  choices=[{"ders": "İşletim Sistemleri I", "sube": "B"}])
    clashes = {c["ders"] for c in plan["cakismalar"]}
    assert "Mikroişlemciler" in clashes
    assert "Mikroişlemciler" in schedule.plan_text(plan, DATA)


def test_free_friday_keeps_only_combinations_without_friday_lessons():
    transcript = ev.load_transcript("ucuncu_sinif_2024")
    plan = schedule.plan_sections(DATA, codes=schedule.current_course_codes(transcript), free_days=["cuma"])
    assert plan["uygun_sayisi"] > 0
    for combo in plan["ornekler"]:
        assert all(lesson["gun"] != "Cuma" for lesson in combo["dersler"])
    assert plan["uygun_subeler"]["Makine Öğrenmesine Giriş"] == ["A"]   # B şubesi Cuma
    text = schedule.plan_text(plan, DATA)
    assert "Makine Öğrenmesine Giriş: yalnız A şubesi" in text


def test_weekly_hours_count_each_lesson_once():
    plan = schedule.plan_sections(DATA, codes=schedule.current_course_codes(ev.load_transcript("birinci_sinif_2026")))
    assert {combo["ders_saati"] for combo in plan["ornekler"]} == {27}


def test_course_not_in_schedule_is_reported():
    plan = schedule.plan_sections(DATA, codes=set(), choices=[{"ders": "Uzay Mühendisliği", "sube": ""}])
    assert plan["programda_yok"] == ["Uzay Mühendisliği"]


# --- agent_h6 ---

def test_h6_has_the_new_tools_and_h5_does_not():
    transcript = ev.load_transcript("erasmus_2023")
    names = {flag: {t.name: t for t in make_tools(None, PLAN, transcript, {}, date(2026, 10, 24), h6_tools=flag)}
             for flag in (False, True)}
    new = {"get_transcript_average", "search_course_schedule", "plan_course_sections"}
    assert new <= set(names[True]) and not new & set(names[False])
    assert "3,49" in names[True]["get_transcript_average"].invoke({"erasmus_haric": True})
    assert AdvisorAgent(None, None, PLAN, prompt_version="agent_h6").h6_tools
    assert not AdvisorAgent(None, None, PLAN, prompt_version="agent_h5").h6_tools
