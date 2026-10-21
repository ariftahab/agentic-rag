"""Ders programı: veri PDF'le karşılaştırılır, gün ve filtreler elle hesaplanmış değerlerle sınanır. LLM yok.

Beklenen bloklar kaynak PDF'ten (data/2026_27_Güz_Lisans_v2.pdf) elle okunmuştur; her sınıftan bir örnek gün.
"""
from datetime import date

import pytest

import schedule

DATA = schedule.load_schedule()
WEDNESDAY = date(2026, 10, 21)


def blocks(day, grade, name):
    return [(b["saatler"][0][:5] + "-" + b["saatler"][-1][-5:], b["sube"], b["yuva"], b["derslik"])
            for b in DATA["dersler"] if b["gun"] == day and b["sinif"] == grade and b["ders"] == name]


def test_saved_schedule_matches_the_pdf():
    assert DATA["dersler"] == schedule.parse_pdf()


def test_first_year_monday_with_aliased_name():
    assert blocks("Pazartesi", 1, "Matematik I") == [("13:30-18:15", None, None, "227")]


def test_second_year_wednesday_room_change_splits_the_block():
    assert blocks("Çarşamba", 2, "Elektrik Devreleri") == [("13:30-15:15", "B", None, "D.L."),
                                                          ("15:30-18:15", "B", None, "326")]


def test_third_year_tuesday_sections():
    assert blocks("Salı", 3, "Mikroişlemciler") == [("13:30-16:15", "A", None, "327"),
                                                    ("16:30-18:15", "A", None, "D.L.")]
    assert blocks("Salı", 3, "İşletim Sistemleri I") == [("08:30-13:15", "A", None, "326"),
                                                         ("13:30-18:15", "B", None, "326")]


def test_fourth_year_thursday_elective_slot():
    assert blocks("Perşembe", 4, "Bilişim Hukuku") == [("12:30-15:15", None, "Seç VII", "326")]


def test_fifth_day_is_friday():
    assert blocks("Cuma", 2, "Veri Yapıları") == [("08:30-13:15", "A", None, "B.L."), ("13:30-18:15", "B", None, "B.L.")]


@pytest.mark.parametrize("text, expected", [
    ("yarın", date(2026, 10, 22)),
    ("bugün", WEDNESDAY),
    ("PERŞEMBE", date(2026, 10, 22)),
    ("çarşamba", WEDNESDAY),
    ("salı", date(2026, 10, 27)),
    ("2026-10-29", date(2026, 10, 29)),
])
def test_day_is_resolved_from_today(text, expected):
    assert schedule.resolve_day(text, WEDNESDAY) == expected


def test_unknown_day_is_reported():
    with pytest.raises(ValueError):
        schedule.resolve_day("geçen hafta", WEDNESDAY)


def test_republic_day_is_a_holiday():
    result = schedule.lookup(DATA, date(2026, 10, 29))
    assert (result["durum"], result["dersler"]) == ("tatil", [])


def test_republic_day_eve_drops_lessons_from_13_00():
    result = schedule.lookup(DATA, date(2026, 10, 28), grade=1)
    assert [(b["ders"], b["saatler"][0][:5], b["saatler"][-1][-5:]) for b in result["dersler"]] == [
        ("Bilgisayar Mühendisliğine Giriş", "08:30", "10:15"),
        ("Kariyer Planlama", "11:30", "12:15"),
        ("Atatürk İlkeleri ve Inkilap Tarihi I", "12:30", "13:15"),
    ]


def test_weekend_and_outside_the_teaching_period():
    assert schedule.lookup(DATA, date(2026, 10, 24))["durum"] == "hafta_sonu"
    assert schedule.lookup(DATA, date(2027, 1, 5))["durum"] == "donem_disi"


def test_filter_by_course_name_accepts_the_schedule_spelling():
    result = schedule.lookup(DATA, date(2026, 10, 26), courses=["Sensör Ağları", "bilişim hukuku"])
    assert [b["ders"] for b in result["dersler"]] == ["Sensor Ağları"]      # Bilişim Hukuku perşembe


def test_my_courses_from_the_transcript():
    transcript = {"dersler": [
        {"yil": 2025, "donem": "Bahar", "kod": "1906003032015", "ad": "Algoritma Analizi", "akts": 6, "not": "BB"},
        {"yil": 2026, "donem": "Güz", "kod": "1906004232018", "ad": "Bilişim Hukuku", "akts": 4, "not": None},
        {"yil": 2026, "donem": "Güz", "kod": "1906004112015", "ad": "Yapay Zeka", "akts": 4, "not": None},
    ]}
    codes = schedule.current_course_codes(transcript)
    assert codes == {"1906004232018", "1906004112015"}
    thursday = schedule.lookup(DATA, date(2026, 10, 22), codes=codes)
    assert [b["ders"] for b in thursday["dersler"]] == ["Bilişim Hukuku"]


def test_llm_text_has_time_section_slot_and_room():
    text = schedule.schedule_text(schedule.lookup(DATA, date(2026, 10, 27), grade=3, courses=["Mikroişlemciler"]), DATA)
    assert "- 13:30-16:15 Mikroişlemciler (A şubesi), 3. sınıf, 327" in text
    assert "- 16:30-18:15 Mikroişlemciler (A şubesi), 3. sınıf, Donanım Laboratuvarı" in text
