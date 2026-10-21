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


def transcript(*courses, enrolled=2023, gpa=3.0):
    return {"bolum": "Bilgisayar Mühendisliği", "egitim_duzeyi": "Lisans", "kayit_yili": enrolled,
            "kayit_donemi": "Güz", "dersler": list(courses),
            "obs_ozet": {"birikimli_alinan_akts": 0, "birikimli_tamamlanan_akts": 0, "gno": gpa}}


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


# --- Yönetmelik ve kayıt yılı kuralları -------------------------------------------------------
RULES_PLAN = {
    **ELECTIVE_PLAN,
    "secmeli_yuvalar": ELECTIVE_PLAN["secmeli_yuvalar"] + [
        {"ad": "Alan Dışı Seçmeli Ders", "kod": "A6", "yariyil": 6, "akts": 2, "tur": "alan_disi",
         "havuz": [{"kod": "ADS1", "ad": "Diksiyon ve Hitabet", "akts": 2}]},
    ],
}


def rule(report, rule_id):
    return next(r for r in report["kurallar"] if r["id"] == rule_id)


def test_every_rule_result_carries_a_source():
    report = graduation.evaluate(ELECTIVE_SAMPLE, RULES_PLAN)
    assert {r["id"] for r in report["kurallar"]} == {
        "toplam_akts", "zorunlu_dersler", "secmeli_akts", "alan_disi_secmeli", "gno_alt_siniri"}
    for r in report["kurallar"]:
        assert r["kaynak"] and all(k["belge"] and k["madde"] for k in r["kaynak"]), r["id"]


def test_out_of_field_elective_is_optional_for_2023_enrollees():
    r = rule(graduation.evaluate(transcript(course(2023, "Güz", "Z1", "Matematik I", 6, "CC"), enrolled=2023),
                                 RULES_PLAN), "alan_disi_secmeli")
    assert (r["zorunlu"], r["karsilandi"]) == (False, True)
    assert r["kaynak"][0]["madde"] == "Geçici Madde 1, fıkra 1"


def test_out_of_field_elective_is_missing_for_2025_enrollees():
    r = rule(graduation.evaluate(transcript(course(2025, "Güz", "Z1", "Matematik I", 6, "CC"), enrolled=2025),
                                 RULES_PLAN), "alan_disi_secmeli")
    assert (r["zorunlu"], r["karsilandi"], r["devam_edenler_gecilirse_karsilandi"]) == (True, False, False)


def test_out_of_field_elective_taken_this_term_completes_it_if_passed():
    r = rule(graduation.evaluate(transcript(
        course(2025, "Güz", "Z1", "Matematik I", 6, "CC"),
        course(2026, "Güz", "ADS1", "Diksiyon ve Hitabet", 2, None),
        enrolled=2025), RULES_PLAN), "alan_disi_secmeli")
    assert (r["karsilandi"], r["devam_edenler_gecilirse_karsilandi"]) == (False, True)


def test_gpa_below_two_fails_the_gpa_rule():
    report = graduation.evaluate(transcript(course(2023, "Güz", "Z1", "Matematik I", 6, "DD"), gpa=1.95),
                                 RULES_PLAN)
    r = rule(report, "gno_alt_siniri")
    assert (r["gereken"], r["mevcut"], r["karsilandi"]) == (2.0, 1.95, False)


def test_gpa_of_two_meets_the_gpa_rule():
    r = rule(graduation.evaluate(transcript(course(2023, "Güz", "Z1", "Matematik I", 6, "DD"), gpa=2.0),
                                 RULES_PLAN), "gno_alt_siniri")
    assert r["karsilandi"] is True


BIG_PLAN = {**RULES_PLAN, "zorunlu_dersler": [{"kod": "Z1", "ad": "Matematik I", "yariyil": 1, "kredi": 4, "akts": 230}]}


def test_total_ects_uses_240_or_the_larger_plan_total_depending_on_enrollment_year():
    # Küçük plan: yönetmelikteki 240 bağlayıcı. Büyük plan: 230 zorunlu + 20 seçmeli + 2 Alan Dışı = 252;
    # 2023 girişliye Alan Dışı Seçmeli zorunlu değil, 250.
    assert rule(graduation.evaluate(ELECTIVE_SAMPLE, RULES_PLAN), "toplam_akts")["gereken"] == 240
    first = course(2025, "Güz", "Z1", "Matematik I", 230, "AA")
    assert rule(graduation.evaluate(transcript(first, enrolled=2025), BIG_PLAN), "toplam_akts")["gereken"] == 252
    assert rule(graduation.evaluate(transcript(first, enrolled=2023), BIG_PLAN), "toplam_akts")["gereken"] == 250


def test_all_requirements_met_only_when_every_rule_is_met():
    done = transcript(
        course(2023, "Güz", "Z1", "Matematik I", 230, "AA"),
        course(2023, "Güz", "ETK", "Mühendislik Etiği", 4, "CC"),
        course(2023, "Bahar", "GIR", "Girişimcilik", 4, "AA"),
        course(2024, "Bahar", "PDO", "Python ile Derin Öğrenme", 4, "CB"),
        course(2025, "Güz", "YZ", "Yapay Zeka", 4, "BB"),
        course(2026, "Güz", "YSA", "Yapay Sinir Ağları", 4, None),
        enrolled=2023, gpa=3.1)
    report = graduation.evaluate(done, BIG_PLAN)
    assert report["tum_sartlar_karsilandi"] is False
    assert report["devam_edenler_gecilirse_tum_sartlar_karsilandi"] is True


def test_requirements_summary_without_a_transcript():
    summary = graduation.graduation_requirements(RULES_PLAN)
    assert (summary["plan_toplam_akts"], summary["zorunlu_akts"], summary["secmeli_akts"]) == (48, 26, 20)  # 6+6+4+5+5 zorunlu, 4+4+4+8 seçmeli, 2 Alan Dışı
    ids = [r["id"] for r in summary["kurallar"]]
    assert {"toplam_akts", "gno_alt_siniri", "gecer_notlar", "fazla_basarisiz_secmeli",
            "alan_disi_secmeli", "alan_disi_secmeli_istisna"} <= set(ids)
    assert all(r["kaynak"] for r in summary["kurallar"])
    assert next(r for r in summary["kurallar"] if r["id"] == "toplam_akts")["deger"] == 240
