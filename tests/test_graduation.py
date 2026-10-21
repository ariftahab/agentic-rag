"""Mezuniyet değerlendirmesi: sahte transkript ve küçük bir plan, LLM yok.

Beklenen değerler elle hesaplanmıştır; plan ve transkript bilerek küçük tutulur.
"""
import graduation

PLAN = {
    "bolum": "Bilgisayar Mühendisliği",
    "katalog_yili": "2024-2025",
    "kaynak": "test planı",
    "zorunlu_dersler": [
        {"kod": "Z1", "ad": "Matematik I", "yariyil": 1, "kredi": 4, "akts": 6},
        {"kod": "Z2", "ad": "Fizik I", "yariyil": 1, "kredi": 4, "akts": 6},
        {"kod": "Z3", "ad": "Veri Yapıları", "yariyil": 3, "kredi": 4, "akts": 4},
        {"kod": "Z4", "ad": "Bitirme Projesi I", "yariyil": 7, "kredi": 3, "akts": 5},
        {"kod": "Z5", "ad": "Staj I", "yariyil": 5, "kredi": 0, "akts": 5},
    ],
}


def course(year, term, code, name, ects, grade):
    return {"yil": year, "donem": term, "kod": code, "ad": name, "kredi": 3, "akts": ects,
            "not": grade, "erasmus": False}


def transcript(*courses):
    return {"bolum": "Bilgisayar Mühendisliği", "egitim_duzeyi": "Lisans", "kayit_yili": 2023,
            "kayit_donemi": "Güz", "dersler": list(courses),
            "obs_ozet": {"birikimli_alinan_akts": 0, "birikimli_tamamlanan_akts": 0, "gno": 3.0}}


SAMPLE = transcript(
    course(2023, "Güz", "Z1", "Matematik I", 6, "CC"),
    course(2023, "Güz", "Z2", "Fizik I", 6, "FF"),
    course(2024, "Güz", "Z3", "Veri Yapıları", 4, "DD"),
    course(2026, "Güz", "Z4", "Bitirme Projesi I", 5, None),
    course(2026, "Güz", "S1", "Yapay Zeka", 4, None),
)


def test_passed_ects_and_ects_if_current_term_passed():
    report = graduation.evaluate(SAMPLE, PLAN)
    assert report["gecilen_akts"] == 10                    # Matematik I 6 + Veri Yapıları 4 (DD geçer, FF geçmez)
    assert report["devam_edenler_gecilirse_akts"] == 19    # + Bitirme Projesi I 5 + Yapay Zeka 4


def test_missing_compulsory_courses_with_status_in_semester_order():
    report = graduation.evaluate(SAMPLE, PLAN)
    assert [(c["ad"], c["durum"]) for c in report["eksik_zorunlu"]] == [
        ("Fizik I", "basarisiz"),
        ("Staj I", "alinmadi"),
        ("Bitirme Projesi I", "devam_ediyor"),
    ]


def test_spring_follows_fall_of_the_same_academic_year():
    # Yıl akademik yılın başlangıcı: "2025 Bahar", "2025 Güz"den sonradır. Yalnız en son dönemin
    # notsuz dersi bu dönem alınıyor sayılır.
    report = graduation.evaluate(transcript(
        course(2025, "Güz", "Z1", "Matematik I", 6, None),
        course(2025, "Bahar", "Z3", "Veri Yapıları", 4, None),
    ), PLAN)
    assert report["devam_edenler_gecilirse_akts"] == 4


# --- Seçmeli yuvaları -------------------------------------------------------------------------
# Rektörlük seçmelileri iki yuvada aynı havuzu paylaşır; Siber Güvenlik iki bölüm yuvasının havuzunda.
ELECTIVE_PLAN = {
    **PLAN,
    "secmeli_yuvalar": [
        {"ad": "Rek. Seçmeli I", "kod": "R1", "yariyil": 1, "akts": 4, "tur": "rektorluk",
         "havuz": [{"kod": "GIR", "ad": "Girişimcilik", "akts": 4},
                   {"kod": "ETK", "ad": "Mühendislik Etiği", "akts": 4}]},
        {"ad": "Rek. Seçmeli II", "kod": "R2", "yariyil": 2, "akts": 4, "tur": "rektorluk",
         "havuz": [{"kod": "GIR", "ad": "Girişimcilik", "akts": 4},
                   {"kod": "ETK", "ad": "Mühendislik Etiği", "akts": 4}]},
        {"ad": "Seçmeli VI", "kod": "S6", "yariyil": 6, "akts": 4, "tur": "bolum",
         "havuz": [{"kod": "PDO", "ad": "Python ile Derin Öğrenme", "akts": 4},
                   {"kod": "SIB", "ad": "Siber Güvenlik", "akts": 4}]},
        {"ad": "Seçmeli VII", "kod": "S7", "yariyil": 7, "akts": 8, "tur": "bolum",
         "havuz": [{"kod": "YZ", "ad": "Yapay Zeka", "akts": 4},
                   {"kod": "YSA", "ad": "Yapay Sinir Ağları", "akts": 4},
                   {"kod": "SIB", "ad": "Siber Güvenlik", "akts": 4}]},
    ],
}


def classification(report, code):
    return next(c for c in report["siniflandirma"] if c["kod"] == code)


def test_rectorate_elective_is_classified_into_its_slots():
    report = graduation.evaluate(transcript(course(2023, "Bahar", "GIR", "Girişimcilik", 4, "AA")), ELECTIVE_PLAN)
    c = classification(report, "GIR")
    assert (c["sinif"], c["yuvalar"]) == ("secmeli", ["Rek. Seçmeli I", "Rek. Seçmeli II"])


def test_department_elective_is_classified_into_its_slot():
    report = graduation.evaluate(transcript(course(2024, "Bahar", "PDO", "Python ile Derin Öğrenme", 4, "CB")),
                                 ELECTIVE_PLAN)
    c = classification(report, "PDO")
    assert (c["sinif"], c["yuvalar"]) == ("secmeli", ["Seçmeli VI"])


def test_course_outside_the_plan_is_classified_as_such():
    report = graduation.evaluate(transcript(course(2025, "Güz", "2FI1", "Operating Systems", 5, "DC")),
                                 ELECTIVE_PLAN)
    c = classification(report, "2FI1")
    assert (c["sinif"], c["yuvalar"]) == ("planda_yok", [])


def test_compulsory_course_is_classified_as_compulsory():
    report = graduation.evaluate(transcript(course(2023, "Güz", "Z1", "Matematik I", 6, "CC")), ELECTIVE_PLAN)
    assert classification(report, "Z1")["sinif"] == "zorunlu"


ELECTIVE_SAMPLE = transcript(
    course(2023, "Güz", "ETK", "Mühendislik Etiği", 4, "CC"),
    course(2023, "Bahar", "GIR", "Girişimcilik", 4, "AA"),
    course(2024, "Bahar", "PDO", "Python ile Derin Öğrenme", 4, "CB"),
    course(2026, "Güz", "YZ", "Yapay Zeka", 4, None),
)


def test_elective_ects_gap_in_both_scenarios():
    s = graduation.evaluate(ELECTIVE_SAMPLE, ELECTIVE_PLAN)["secmeli"]
    assert s["gereken_akts"] == 20                          # 4 + 4 + 4 + 8
    assert (s["gecilen_akts"], s["acik_akts"]) == (12, 8)
    assert (s["devam_edenler_gecilirse_akts"], s["devam_edenler_gecilirse_acik_akts"]) == (16, 4)
    assert s["fazla_akts"] == 0


def test_filled_slots_in_both_scenarios():
    slots = graduation.evaluate(ELECTIVE_SAMPLE, ELECTIVE_PLAN)["secmeli"]["yuvalar"]
    assert [(y["ad"], y["dersler"], y["doldu"]) for y in slots] == [
        ("Rek. Seçmeli I", ["Mühendislik Etiği"], True),
        ("Rek. Seçmeli II", ["Girişimcilik"], True),
        ("Seçmeli VI", ["Python ile Derin Öğrenme"], True),
        ("Seçmeli VII", [], False),
    ]
    vii = slots[-1]
    assert (vii["devam_edenler_gecilirse_dersler"], vii["devam_edenler_gecilirse_doldu"]) == (["Yapay Zeka"], False)


# --- Not durumları ----------------------------------------------------------------------------
def test_failed_elective_does_not_count():
    s = graduation.evaluate(transcript(course(2023, "Bahar", "GIR", "Girişimcilik", 4, "FF")), ELECTIVE_PLAN)["secmeli"]
    assert (s["gecilen_akts"], s["yuvalar"][0]["doldu"]) == (0, False)


def test_course_failed_then_passed_counts_once_and_is_not_missing():
    report = graduation.evaluate(transcript(
        course(2023, "Güz", "Z1", "Matematik I", 6, "FF"),
        course(2024, "Güz", "Z1", "Matematik I", 6, "CC"),
    ), PLAN)
    assert report["gecilen_akts"] == 6
    assert "Matematik I" not in [c["ad"] for c in report["eksik_zorunlu"]]


def test_course_repeated_to_raise_the_grade_counts_once():
    report = graduation.evaluate(transcript(
        course(2023, "Güz", "Z1", "Matematik I", 6, "DD"),
        course(2024, "Güz", "Z1", "Matematik I", 6, "BB"),
    ), PLAN)
    assert report["gecilen_akts"] == 6


def test_latest_grade_of_a_repeated_course_is_what_counts():
    # Yönetmelik: tekrarlanan derste önceki not dikkate alınmadan son not geçerlidir.
    report = graduation.evaluate(transcript(
        course(2023, "Güz", "Z1", "Matematik I", 6, "CC"),
        course(2024, "Güz", "Z1", "Matematik I", 6, "FF"),
    ), PLAN)
    assert report["gecilen_akts"] == 0
    assert ("Matematik I", "basarisiz") in [(c["ad"], c["durum"]) for c in report["eksik_zorunlu"]]


def test_passed_course_retaken_this_term_counts_once_in_both_scenarios():
    report = graduation.evaluate(transcript(
        course(2023, "Güz", "Z1", "Matematik I", 6, "DD"),
        course(2026, "Güz", "Z1", "Matematik I", 6, None),
    ), PLAN)
    assert (report["gecilen_akts"], report["devam_edenler_gecilirse_akts"]) == (6, 6)
    assert report["eksik_zorunlu"][0]["ad"] != "Matematik I"


def test_dd_and_dc_are_passing_grades():
    report = graduation.evaluate(transcript(
        course(2023, "Güz", "Z1", "Matematik I", 6, "DD"),
        course(2023, "Güz", "Z2", "Fizik I", 6, "DC"),
        course(2024, "Bahar", "PDO", "Python ile Derin Öğrenme", 4, "DD"),
    ), ELECTIVE_PLAN)
    assert report["gecilen_akts"] == 16
    assert {"Matematik I", "Fizik I"}.isdisjoint(c["ad"] for c in report["eksik_zorunlu"])
    assert report["secmeli"]["gecilen_akts"] == 4


def test_extra_failed_elective_does_not_block_once_electives_are_complete():
    # Yönetmelik: seçmeli yükümlülüğü tamamlandıktan sonra fazladan alınıp kalınan seçmeli engel değildir.
    s = graduation.evaluate(transcript(
        course(2023, "Güz", "ETK", "Mühendislik Etiği", 4, "CC"),
        course(2023, "Bahar", "GIR", "Girişimcilik", 4, "AA"),
        course(2024, "Bahar", "PDO", "Python ile Derin Öğrenme", 4, "CB"),
        course(2025, "Güz", "YZ", "Yapay Zeka", 4, "BB"),
        course(2025, "Güz", "YSA", "Yapay Sinir Ağları", 4, "BA"),
        course(2025, "Bahar", "SIB", "Siber Güvenlik", 4, "FF"),
    ), ELECTIVE_PLAN)["secmeli"]
    assert (s["gecilen_akts"], s["acik_akts"], s["fazla_akts"]) == (20, 0, 0)
    assert all(y["doldu"] for y in s["yuvalar"])


def test_extra_passed_elective_ects_is_reported_as_surplus():
    s = graduation.evaluate(transcript(
        course(2023, "Güz", "ETK", "Mühendislik Etiği", 4, "CC"),
        course(2023, "Bahar", "GIR", "Girişimcilik", 4, "AA"),
        course(2024, "Bahar", "PDO", "Python ile Derin Öğrenme", 4, "CB"),
        course(2025, "Güz", "YZ", "Yapay Zeka", 4, "BB"),
        course(2025, "Güz", "YSA", "Yapay Sinir Ağları", 4, "BA"),
        course(2025, "Bahar", "SIB", "Siber Güvenlik", 4, "AA"),
    ), ELECTIVE_PLAN)["secmeli"]
    assert (s["gecilen_akts"], s["acik_akts"], s["fazla_akts"]) == (24, 0, 4)


def test_course_in_two_pools_is_kept_for_the_slot_that_needs_it():
    # Siber Güvenlik hem Seçmeli VI hem Seçmeli VII havuzunda; Python ile Derin Öğrenme yalnız VI'da.
    # Siber Güvenlik önce alınmış olsa da VI'yı Python ile Derin Öğrenme doldurur, Siber Güvenlik VII'ye kalır.
    slots = graduation.evaluate(transcript(
        course(2024, "Güz", "SIB", "Siber Güvenlik", 4, "AA"),
        course(2024, "Bahar", "PDO", "Python ile Derin Öğrenme", 4, "CB"),
    ), ELECTIVE_PLAN)["secmeli"]["yuvalar"]
    assert [(y["ad"], y["dersler"]) for y in slots[2:]] == [
        ("Seçmeli VI", ["Python ile Derin Öğrenme"]),
        ("Seçmeli VII", ["Siber Güvenlik"]),
    ]
