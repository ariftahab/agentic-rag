"""Mezuniyet değerlendirmesi: sahte transkript ve küçük bir plan, LLM yok.

Beklenen değerler elle hesaplanmıştır; plan ve transkript bilerek küçük tutulur.
"""
import pytest

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
    assert (s["gecilen_akts"], s["acik_akts"], s["fazla_akts"]) == (20, 0, 4)
    assert s["devam_edenler_gecilirse_acik_akts"] == 0                 # fazlalık açığı eksiye düşürmez


def test_surplus_in_one_slot_does_not_fill_another_slot():
    # Her yuva kendi havuzundan, kendi AKTS'si kadar dolar (kullanıcı kararı 2026-10-22): Seçmeli VII'de fazladan
    # alınan ders, Seçmeli VIII'in açığını kapatmaz.
    plan = {**PLAN, "secmeli_yuvalar": [
        {"ad": "Seçmeli VII", "kod": "S7", "yariyil": 7, "akts": 4, "tur": "bolum",
         "havuz": [{"kod": "YZ", "ad": "Yapay Zeka", "akts": 4}, {"kod": "YSA", "ad": "Yapay Sinir Ağları", "akts": 4}]},
        {"ad": "Seçmeli VIII", "kod": "S8", "yariyil": 8, "akts": 4, "tur": "bolum",
         "havuz": [{"kod": "GI", "ad": "Görüntü İşleme", "akts": 4}]},
    ]}
    report = graduation.evaluate(transcript(
        course(2025, "Güz", "YZ", "Yapay Zeka", 4, "BB"),
        course(2025, "Güz", "YSA", "Yapay Sinir Ağları", 4, "BA"),
    ), plan)
    s = report["secmeli"]
    assert (s["gecilen_akts"], s["acik_akts"], s["fazla_akts"]) == (4, 4, 4)
    assert rule_values(report, "secmeli_akts") == (8, 4, 4, False, False)


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


def test_llm_summary_starts_with_both_scenarios():
    report = graduation.evaluate(SAMPLE, PLAN)
    lines = graduation.summary_text(report, PLAN).splitlines()
    assert lines[3].startswith("1) Şu an: 10 / ")
    assert lines[4].startswith("2) Bu dönem alınan dersler geçilirse: 19 / ")


def test_llm_summary_lists_missing_courses_with_their_status():
    text = graduation.summary_text(graduation.evaluate(SAMPLE, PLAN), PLAN)
    assert "- Fizik I (1. yarıyıl, 6 AKTS, başarısız)" in text
    assert "- Bitirme Projesi I (7. yarıyıl, 5 AKTS, bu dönem alınıyor)" in text


def test_llm_summary_leaves_out_course_classification():
    report = graduation.evaluate(SAMPLE, PLAN)
    assert "Matematik I" in str(report["siniflandirma"])         # geçilmiş zorunlu ders
    assert "Matematik I" not in graduation.summary_text(report, PLAN)


def test_llm_summary_lists_current_term_courses_with_kind_and_ects():
    # "Bu dönem hangi seçmelileri alıyorum?" sorusu için: dolan yuvalar özette olmadığından agent cevaplayamıyordu.
    report = graduation.evaluate(transcript(
        course(2025, "Bahar", "PDO", "Python ile Derin Öğrenme", 4, "CB"),
        course(2026, "Güz", "Z4", "Bitirme Projesi I", 5, None),
        course(2026, "Güz", "YZ", "Yapay Zeka", 4, None),
        course(2026, "Güz", "SIB", "Siber Güvenlik", 4, None),
        course(2026, "Güz", "2FI1", "Operating Systems", 5, None),
    ), ELECTIVE_PLAN)
    lines = graduation.summary_text(report, ELECTIVE_PLAN).splitlines()
    start = lines.index("BU DÖNEM ALINAN DERSLER")
    assert lines[start + 1:start + 5] == [
        "- Bitirme Projesi I (5 AKTS, zorunlu)",
        "- Yapay Zeka (4 AKTS, seçmeli: Seçmeli VII)",
        "- Siber Güvenlik (4 AKTS, seçmeli: Seçmeli VII)",
        "- Operating Systems (5 AKTS, planda yok)",
    ]


# --- Plan dışı derslerin eşlenmesi (Erasmus, intibak) -----------------------------------------
# Adlar gerçek transkriptteki Erasmus dersleri ve öğretim planındaki İngilizce adlardır; eşik bunlarla seçildi.
MATCH_PLAN = {
    **PLAN,
    "zorunlu_dersler": [
        {"kod": "OS1", "ad": "İşletim Sistemleri I", "ad_en": "Operating Systems I", "yariyil": 5, "kredi": 4, "akts": 5},
        {"kod": "VTY", "ad": "Veritabanı Yönetim Sistemleri", "ad_en": "Database Management Systems", "yariyil": 6,
         "kredi": 4, "akts": 6},
        {"kod": "GOM", "ad": "Gömülü Sistemler", "ad_en": "Embedded Systems", "yariyil": 6, "kredi": 4, "akts": 5},
        {"kod": "MIK", "ad": "Mikroişlemciler", "ad_en": "Microprocessors", "yariyil": 5, "kredi": 4, "akts": 5},
        {"kod": "AG", "ad": "Veri İletişimi ve Bilgisayar Ağları", "ad_en": "Data Communication and Computer Networks",
         "yariyil": 8, "kredi": 4, "akts": 5},
    ],
    "secmeli_yuvalar": [
        {"ad": "Seçmeli V", "kod": "S5", "yariyil": 5, "akts": 4, "tur": "bolum",
         "havuz": [{"kod": "OS2", "ad": "İşletim Sistemleri II", "ad_en": "Operating Systems II", "akts": 4},
                   {"kod": "YZ", "ad": "Yapay Zeka", "ad_en": "Artificial Intelligence", "akts": 4}]},
    ],
}


def erasmus(code, name, ects, grade, term="Güz"):
    return {**course(2025, term, code, name, ects, grade), "erasmus": True}


ERASMUS_SAMPLE = transcript(
    erasmus("E1", "Operating Systems", 5, "DC"),
    erasmus("E2", "Microcomputer Systems", 5, "AA"),
    erasmus("E3", "Database Systems", 6, "CB", "Bahar"),
    erasmus("E4", "Computer Networks", 5, "FF", "Bahar"),
    erasmus("E5", "Embedded Computer Systems", 5, "DC", "Bahar"),
)


def matched(report):
    return {m["ad"]: (m["plan_ad"], m["yontem"]) for m in report["eslesmeler"]}


def test_erasmus_courses_with_similar_english_names_are_matched_automatically():
    report = graduation.evaluate(ERASMUS_SAMPLE, MATCH_PLAN)
    assert matched(report) == {
        "Operating Systems": ("İşletim Sistemleri I", "otomatik"),      # compulsory wins the tie with II
        "Database Systems": ("Veritabanı Yönetim Sistemleri", "otomatik"),
        "Computer Networks": ("Veri İletişimi ve Bilgisayar Ağları", "otomatik"),
        "Embedded Computer Systems": ("Gömülü Sistemler", "otomatik"),
    }


def test_matched_courses_count_as_the_plan_course():
    missing = {c["ad"]: c["durum"] for c in graduation.evaluate(ERASMUS_SAMPLE, MATCH_PLAN)["eksik_zorunlu"]}
    assert missing == {"Mikroişlemciler": "alinmadi", "Veri İletişimi ve Bilgisayar Ağları": "basarisiz"}


def test_unmatched_course_is_listed_and_still_counts_towards_total_ects():
    report = graduation.evaluate(ERASMUS_SAMPLE, MATCH_PLAN)
    assert [(c["ad"], c["durum"]) for c in report["eslesmeyen"]] == [("Microcomputer Systems", "gecti")]
    assert report["gecilen_akts"] == 5 + 5 + 6 + 5                     # FF olan Computer Networks hariç


def test_confirmed_equivalence_completes_the_plan_course():
    report = graduation.evaluate(ERASMUS_SAMPLE, MATCH_PLAN, confirmations={"E2": "MIK"})
    assert matched(report) == {
        "Operating Systems": ("İşletim Sistemleri I", "otomatik"),
        "Microcomputer Systems": ("Mikroişlemciler", "teyit"),     # teyit sonraki dersleri atlatmıyor
        "Database Systems": ("Veritabanı Yönetim Sistemleri", "otomatik"),
        "Computer Networks": ("Veri İletişimi ve Bilgisayar Ağları", "otomatik"),
        "Embedded Computer Systems": ("Gömülü Sistemler", "otomatik"),
    }
    assert "Mikroişlemciler" not in [c["ad"] for c in report["eksik_zorunlu"]]
    assert report["eslesmeyen"] == []


def test_confirmed_no_equivalent_is_not_matched_again():
    report = graduation.evaluate(ERASMUS_SAMPLE, MATCH_PLAN, confirmations={"E1": None})
    assert "Operating Systems" not in matched(report)
    assert [(c["ad"], c["teyit_edildi"]) for c in report["eslesmeyen"]] == [
        ("Operating Systems", True), ("Microcomputer Systems", False)]


def test_matched_elective_fills_its_slot():
    report = graduation.evaluate(transcript(erasmus("E9", "Artificial Intelligence", 4, "BB")), MATCH_PLAN)
    assert matched(report) == {"Artificial Intelligence": ("Yapay Zeka", "otomatik")}
    assert report["secmeli"]["gecilen_akts"] == 4


def test_plan_course_already_passed_is_not_a_match_target():
    report = graduation.evaluate(transcript(course(2024, "Güz", "OS1", "İşletim Sistemleri I", 5, "BB"),
                                            erasmus("E1", "Operating Systems", 5, "DC")), MATCH_PLAN)
    assert matched(report) == {"Operating Systems": ("İşletim Sistemleri II", "otomatik")}


def test_tie_between_two_compulsory_courses_is_left_to_the_user():
    plan = {**MATCH_PLAN, "zorunlu_dersler": MATCH_PLAN["zorunlu_dersler"] + [
        {"kod": "OS3", "ad": "İşletim Sistemleri III", "ad_en": "Operating Systems III", "yariyil": 7, "kredi": 4,
         "akts": 5}]}
    report = graduation.evaluate(transcript(erasmus("E1", "Operating Systems", 5, "DC")), plan)
    assert [c["ad"] for c in report["eslesmeyen"]] == ["Operating Systems"]


def test_llm_summary_shows_matches_and_unmatched_courses():
    text = graduation.summary_text(graduation.evaluate(ERASMUS_SAMPLE, MATCH_PLAN), MATCH_PLAN)
    assert "Operating Systems (5 AKTS) → İşletim Sistemleri I" in text
    assert "intibak komisyonu" in text
    assert "- Microcomputer Systems (5 AKTS, geçti)" in text


def test_llm_summary_asks_about_unconfirmed_courses_right_after_the_scenarios():
    lines = graduation.summary_text(graduation.evaluate(ERASMUS_SAMPLE, MATCH_PLAN), MATCH_PLAN).splitlines()
    ask = next(i for i, line in enumerate(lines) if "hangi derse sayıldı" in line)
    assert "Microcomputer Systems" in lines[ask]
    assert ask < lines.index("KURALLAR (şu an → bu dönem geçilirse)")


def test_llm_summary_does_not_ask_again_after_a_no_equivalent_confirmation():
    report = graduation.evaluate(ERASMUS_SAMPLE, MATCH_PLAN, confirmations={"E2": None})
    assert "hangi derse sayıldı" not in graduation.summary_text(report, MATCH_PLAN)


# --- Mutasyon testinde (cosmic-ray) yakalanmayan davranışlar -----------------------------------
def test_summer_term_follows_spring_of_the_same_year():
    # Transkriptte sıra karışık olsa da Yaz, Bahar'dan sonradır: Bahar'da FF, Yaz okulunda CC -> geçti.
    report = graduation.evaluate(transcript(
        course(2024, "Yaz", "Z1", "Matematik I", 6, "CC"),
        course(2024, "Bahar", "Z1", "Matematik I", 6, "FF"),
    ), PLAN)
    assert report["gecilen_akts"] == 6


def test_repeated_course_is_classified_once_and_later_courses_still_are():
    report = graduation.evaluate(transcript(
        course(2023, "Güz", "Z1", "Matematik I", 6, "FF"),
        course(2024, "Güz", "Z1", "Matematik I", 6, "CC"),
        course(2024, "Güz", "Z2", "Fizik I", 6, "CC"),
    ), PLAN)
    assert [c["kod"] for c in report["siniflandirma"]] == ["Z1", "Z2"]


def rule_values(report, rule_id):
    r = rule(report, rule_id)
    return r["gereken"], r["mevcut"], r["devam_edenler_gecilirse"], r["karsilandi"], r["devam_edenler_gecilirse_karsilandi"]


def test_rule_values_when_requirements_are_not_met():
    # PLAN: 5 zorunlu (6+6+4+5+5 = 26 AKTS). Geçilen 6 + 4 = 10; Bitirme Projesi I geçilirse 15, Yapay Zeka planda
    # yok ama AKTS'si sayılır: 19. Eksik 3 zorunlu (Fizik I, Staj I, Bitirme Projesi I); biri devam ediyor.
    report = graduation.evaluate(SAMPLE, PLAN)
    assert rule_values(report, "toplam_akts") == (240, 10, 19, False, False)
    assert rule_values(report, "zorunlu_dersler") == (5, 2, 3, False, False)
    assert rule_values(report, "secmeli_akts") == (0, 0, 0, True, True)


def test_rule_values_when_requirements_are_met_this_term():
    # BIG_PLAN, 2023 girişli: gereken 230 + 20 = 250 (Alan Dışı isteğe bağlı). Geçilen 230 + 16 seçmeli = 246;
    # Yapay Sinir Ağları (4) geçilirse 250.
    report = graduation.evaluate(transcript(
        course(2023, "Güz", "Z1", "Matematik I", 230, "AA"),
        course(2023, "Güz", "ETK", "Mühendislik Etiği", 4, "CC"),
        course(2023, "Bahar", "GIR", "Girişimcilik", 4, "AA"),
        course(2024, "Bahar", "PDO", "Python ile Derin Öğrenme", 4, "CB"),
        course(2025, "Güz", "YZ", "Yapay Zeka", 4, "BB"),
        course(2026, "Güz", "YSA", "Yapay Sinir Ağları", 4, None),
        enrolled=2023, gpa=3.1), BIG_PLAN)
    assert rule_values(report, "toplam_akts") == (250, 246, 250, False, True)
    assert rule_values(report, "zorunlu_dersler") == (1, 1, 1, True, True)
    assert rule_values(report, "secmeli_akts") == (20, 16, 20, False, True)
    assert rule_values(report, "alan_disi_secmeli") == (0, 0, 0, True, True)
    assert rule_values(report, "gno_alt_siniri") == (2.0, 3.1, 3.1, True, True)


def test_out_of_field_elective_is_required_from_the_first_year_it_applies():
    # Yönerge: 2024-2025 ve sonrasında kaydolanlar için zorunlu (sınır değer 2024).
    r = rule(graduation.evaluate(transcript(course(2024, "Güz", "Z1", "Matematik I", 6, "CC"), enrolled=2024),
                                 RULES_PLAN), "alan_disi_secmeli")
    assert r["zorunlu"] is True


def test_out_of_field_rule_counts_only_the_out_of_field_slot():
    # Rektörlük seçmelisi dolu, Alan Dışı boş: Alan Dışı kuralı karşılanmaz. Diksiyon geçilince 1 / 1.
    other = graduation.evaluate(transcript(course(2025, "Bahar", "GIR", "Girişimcilik", 4, "AA"), enrolled=2025),
                                RULES_PLAN)
    assert rule_values(other, "alan_disi_secmeli") == (1, 0, 0, False, False)
    done = graduation.evaluate(transcript(course(2025, "Bahar", "ADS1", "Diksiyon ve Hitabet", 2, "AA"), enrolled=2025),
                               RULES_PLAN)
    assert rule_values(done, "alan_disi_secmeli") == (1, 1, 1, True, True)


def test_only_the_out_of_field_slot_is_optional_for_early_enrollees():
    slots = graduation.evaluate(transcript(course(2023, "Güz", "Z1", "Matematik I", 6, "CC"), enrolled=2023),
                                RULES_PLAN)["secmeli"]["yuvalar"]
    assert [(y["ad"], y["zorunlu"]) for y in slots] == [
        ("Rek. Seçmeli I", True), ("Rek. Seçmeli II", True), ("Seçmeli VI", True),
        ("Alan Dışı Seçmeli Ders", False), ("Seçmeli VII", True)]


def test_course_smaller_than_the_slot_does_not_fill_it():
    slots = graduation.evaluate(transcript(course(2023, "Bahar", "GIR", "Girişimcilik", 3, "AA")),
                                ELECTIVE_PLAN)["secmeli"]["yuvalar"]
    assert (slots[0]["dersler"], slots[0]["doldu"]) == (["Girişimcilik"], False)


def test_course_larger_than_the_slot_fills_it_alone():
    # Mühendislik Etiği (6) Rek. Seçmeli I'i (4) tek başına doldurur; Girişimcilik Rek. Seçmeli II'ye kalır.
    slots = graduation.evaluate(transcript(
        course(2023, "Güz", "ETK", "Mühendislik Etiği", 6, "CC"),
        course(2023, "Bahar", "GIR", "Girişimcilik", 4, "AA"),
    ), ELECTIVE_PLAN)["secmeli"]["yuvalar"]
    assert [(y["dersler"], y["doldu"]) for y in slots[:2]] == [(["Mühendislik Etiği"], True), (["Girişimcilik"], True)]


def test_elective_gap_if_this_term_passes():
    # Gereken 20; geçilen 4 + 4 = 8, Python ile Derin Öğrenme geçilirse 12 -> açık 12 ve 8.
    s = graduation.evaluate(transcript(
        course(2023, "Güz", "ETK", "Mühendislik Etiği", 4, "CC"),
        course(2023, "Bahar", "GIR", "Girişimcilik", 4, "AA"),
        course(2026, "Güz", "PDO", "Python ile Derin Öğrenme", 4, None),
    ), ELECTIVE_PLAN)["secmeli"]
    assert (s["acik_akts"], s["devam_edenler_gecilirse_acik_akts"]) == (12, 8)


def test_requirements_summary_lists_each_rule_once_with_the_out_of_field_ects():
    summary = graduation.graduation_requirements(RULES_PLAN)
    ids = [r["id"] for r in summary["kurallar"]]
    assert len(ids) == len(set(ids))
    assert next(r for r in summary["kurallar"] if r["id"] == "alan_disi_secmeli")["akts"] == 2


def test_requirements_text_gives_every_rule_with_its_source():
    text = graduation.requirements_text(graduation.graduation_requirements(RULES_PLAN))
    assert "48 AKTS" in text                                                     # planın bağlayıcı toplamı
    assert "- Seçmeli VII (7. yarıyıl, 8 AKTS, bölüm seçmelisi)" in text         # yuva bazlı seçmeliler
    assert "2,00" in text and "[BAİBÜ Önlisans, Lisans Eğitim-Öğretim Yönetmeliği, Madde 35, fıkra 1, bent b]" in text
    assert "2024" in text and "Geçici Madde 1" in text                           # ADSD ve kayıt yılı istisnası
    rule_lines = [line for line in text.splitlines() if line.startswith("- ") and "yarıyıl" not in line]
    assert rule_lines and all(line.endswith("]") for line in rule_lines)         # her kuralın kaynağı var


def test_requirements_text_names_the_internships_of_the_real_plan():
    text = graduation.requirements_text(graduation.graduation_requirements())
    assert "Staj I (5. yarıyıl, 5 AKTS)" in text and "Staj II (7. yarıyıl, 5 AKTS)" in text


@pytest.mark.parametrize("a, b, expected", [
    ("Operating Systems", "Operating Systems I", 2 / 3),
    ("Computer Networks", "Data Communication and Computer Networks", 0.5),   # "and" sayılmaz
    ("Microcomputer Systems", "Microprocessors", 0.0),
    ("Operating Systems", "", 0.0),
])
def test_name_similarity(a, b, expected):
    assert graduation.name_similarity(a, b) == pytest.approx(expected)


def test_automatic_match_reports_its_similarity():
    report = graduation.evaluate(ERASMUS_SAMPLE, MATCH_PLAN)
    assert {m["ad"]: m["benzerlik"] for m in report["eslesmeler"]}["Operating Systems"] == 0.67


def test_plan_course_without_english_name_is_not_matched_by_its_empty_name():
    plan = {**PLAN, "zorunlu_dersler": [{"kod": "Z1", "ad": "Matematik I", "yariyil": 1, "kredi": 4, "akts": 6}]}
    report = graduation.evaluate(transcript(erasmus("E1", "Operating Systems", 5, "DC")), plan)
    assert report["eslesmeler"] == []


def test_erasmus_courses_against_the_real_plan():
    # Yalnız ad benzerliği katmanı (eşdeğerlik tablosu ayrı testte).
    report = graduation.evaluate(ERASMUS_SAMPLE, graduation.load_plan(), equivalences=[])
    assert matched(report) == {
        "Operating Systems": ("İşletim Sistemleri I", "otomatik"),
        "Database Systems": ("Veritabanı Yönetim Sistemleri", "otomatik"),
        "Computer Networks": ("Veri İletişimi ve Bilgisayar Ağları", "otomatik"),
        "Embedded Computer Systems": ("Gömülü Sistemler", "otomatik"),
    }
    assert [c["ad"] for c in report["eslesmeyen"]] == ["Microcomputer Systems"]


def test_llm_summary_states_requirements_rules_and_open_slots():
    report = graduation.evaluate(ELECTIVE_SAMPLE, RULES_PLAN)
    text = graduation.summary_text(report, RULES_PLAN)
    assert "mezuniyet şartları karşılanmıyor" in text
    assert "- Mezuniyet için genel ağırlıklı not ortalaması (GNO) en az 2,00 olur. gereken 2.0, şu an 3.0" in text
    assert "- Seçmeli VII (7. yarıyıl, 8 AKTS): bu dönem geçilse de açık" in text
    assert "Rek. Seçmeli I (" not in text                       # dolu yuva listelenmez


def test_report_text_for_the_terminal():
    text = graduation.report_text(graduation.evaluate(ELECTIVE_SAMPLE, RULES_PLAN), RULES_PLAN)
    assert "  - 6. yarıyıl  Alan Dışı Seçmeli Ders (2 AKTS): boş/eksik (isteğe bağlı)  [-]" in text
    assert "  - 7. yarıyıl  Seçmeli VII (8 AKTS): boş/eksik  [Yapay Zeka]" in text
    assert "  - toplam_akts: 12 / 240 -> karşılanmadı; 16 -> karşılanmadı" in text


@pytest.mark.parametrize("ects, met", [(239, False), (240, True), (300, True)])
def test_total_ects_boundary(ects, met):
    # PLAN toplamı 26 < 240: yönetmelikteki 240 bağlayıcı; 240 tam sınır, 300 > 256 (küçük tamsayı önbelleği dışı).
    report = graduation.evaluate(transcript(course(2023, "Güz", "Z1", "Matematik I", ects, "AA")), PLAN)
    assert rule_values(report, "toplam_akts") == (240, ects, ects, met, met)


def test_elective_rule_when_the_gap_remains_this_term():
    # ELECTIVE_SAMPLE: seçmeli 4 + 4 + 4 = 12, Yapay Zeka geçilirse 16; gereken 20.
    report = graduation.evaluate(ELECTIVE_SAMPLE, RULES_PLAN)
    assert rule_values(report, "secmeli_akts") == (20, 12, 16, False, False)


def test_llm_summary_says_requirements_are_met():
    done = transcript(
        course(2023, "Güz", "Z1", "Matematik I", 230, "AA"),
        course(2023, "Güz", "ETK", "Mühendislik Etiği", 4, "CC"),
        course(2023, "Bahar", "GIR", "Girişimcilik", 4, "AA"),
        course(2024, "Bahar", "PDO", "Python ile Derin Öğrenme", 4, "CB"),
        course(2025, "Güz", "YZ", "Yapay Zeka", 4, "BB"),
        course(2025, "Bahar", "YSA", "Yapay Sinir Ağları", 4, "BA"),
        enrolled=2023, gpa=3.1)
    text = graduation.summary_text(graduation.evaluate(done, BIG_PLAN), BIG_PLAN)
    assert "mezuniyet şartları karşılanıyor" in text
    assert "karşılanmıyor" not in text


def test_report_text_lists_unmatched_out_of_plan_courses():
    text = graduation.report_text(graduation.evaluate(ERASMUS_SAMPLE, MATCH_PLAN), MATCH_PLAN)
    assert "Planda olmayan dersler (1):\n  - Microcomputer Systems (E2)\n" in text


# --- What-if (simulate) -----------------------------------------------------------------------
# BIG_PLAN, 2023 girişli: gereken 230 + 20 seçmeli = 250 (Alan Dışı isteğe bağlı). Geçilen 230 + 12 seçmeli = 242;
# bu dönem Yapay Zeka (4) geçilirse 246, seçmeli 16 / 20.
WHAT_IF_SAMPLE = transcript(
    course(2023, "Güz", "Z1", "Matematik I", 230, "AA"),
    course(2023, "Güz", "ETK", "Mühendislik Etiği", 4, "CC"),
    course(2023, "Bahar", "GIR", "Girişimcilik", 4, "AA"),
    course(2024, "Bahar", "PDO", "Python ile Derin Öğrenme", 4, "CB"),
    course(2026, "Güz", "YZ", "Yapay Zeka", 4, None),
    enrolled=2023, gpa=3.1)


def test_what_if_set_that_completes_the_requirements():
    report = graduation.simulate(WHAT_IF_SAMPLE, BIG_PLAN, ["Yapay Sinir Ağları"])
    assert rule_values(report, "toplam_akts") == (250, 242, 250, False, True)
    assert rule_values(report, "secmeli_akts") == (20, 12, 20, False, True)
    assert report["devam_edenler_gecilirse_tum_sartlar_karsilandi"] is True


def test_what_if_set_that_does_not_complete_the_requirements():
    # Alan Dışı Seçmeli Ders toplam AKTS'ye sayılır ama bölüm seçmelisi açığını kapatmaz: 248 / 250, seçmeli 16 / 20.
    report = graduation.simulate(WHAT_IF_SAMPLE, BIG_PLAN, ["Diksiyon ve Hitabet"])
    assert rule_values(report, "toplam_akts") == (250, 242, 248, False, False)
    assert rule_values(report, "secmeli_akts") == (20, 12, 16, False, False)
    assert report["devam_edenler_gecilirse_tum_sartlar_karsilandi"] is False


def test_what_if_without_courses_equals_the_this_term_scenario():
    assert graduation.simulate(WHAT_IF_SAMPLE, BIG_PLAN, []) == {
        **graduation.evaluate(WHAT_IF_SAMPLE, BIG_PLAN), "varsayimsal": []}


def test_what_if_course_already_passed_or_ongoing_counts_once():
    report = graduation.simulate(WHAT_IF_SAMPLE, BIG_PLAN, ["Girişimcilik", "Yapay Zeka", "Yapay Sinir Ağları"])
    assert [(c["ad"], c["durum"]) for c in report["varsayimsal"]] == [
        ("Girişimcilik", "gecti"), ("Yapay Zeka", "devam_ediyor"), ("Yapay Sinir Ağları", "yeni")]
    assert report["devam_edenler_gecilirse_akts"] == 250


def test_what_if_failed_compulsory_course_is_no_longer_missing():
    report = graduation.simulate(SAMPLE, PLAN, ["Fizik I"])
    assert [c["ad"] for c in report["eksik_zorunlu"] if c["durum"] != "devam_ediyor"] == ["Staj I"]
    assert report["devam_edenler_gecilirse_akts"] == 25     # 19 + Fizik I 6


def test_what_if_resolves_names_case_insensitively_in_english_and_by_code():
    plan = {**BIG_PLAN, "secmeli_yuvalar": [
        {**s, "havuz": [{**m, "ad_en": "Neural Networks"} if m["kod"] == "YSA" else m for m in s["havuz"]]}
        for s in BIG_PLAN["secmeli_yuvalar"]]}
    for name in ("yapay sinir ağları", "Neural Networks", "YSA"):
        assert graduation.simulate(WHAT_IF_SAMPLE, plan, [name])["varsayimsal"][0]["kod"] == "YSA", name


def test_what_if_gpa_stays_the_transcript_value():
    report = graduation.simulate({**WHAT_IF_SAMPLE, "obs_ozet": {"gno": 1.9}}, BIG_PLAN, ["Yapay Sinir Ağları"])
    assert rule_values(report, "gno_alt_siniri") == (2.0, 1.9, 1.9, False, False)
    assert report["devam_edenler_gecilirse_tum_sartlar_karsilandi"] is False


def test_what_if_with_unknown_course_name_names_it():
    with pytest.raises(graduation.UnknownCourseError) as e:
        graduation.simulate(WHAT_IF_SAMPLE, BIG_PLAN, ["Yapay Sinir Ağları", "Kuantum Fiziği"])
    assert e.value.names == ["Kuantum Fiziği"]
    assert "Kuantum Fiziği" in str(e.value)


def test_what_if_text_compares_the_sets():
    reports = [graduation.simulate(WHAT_IF_SAMPLE, BIG_PLAN, names)
               for names in (["Yapay Sinir Ağları"], ["Diksiyon ve Hitabet"])]
    text = graduation.simulation_text(graduation.evaluate(WHAT_IF_SAMPLE, BIG_PLAN), reports, BIG_PLAN)
    assert "Varsayımsal ders olmadan: 246 / 250 AKTS" in text
    assert "1. küme (Yapay Sinir Ağları): mezuniyet şartları karşılanıyor" in text
    assert "2. küme (Diksiyon ve Hitabet): mezuniyet şartları karşılanmıyor" in text
    assert "248 / 250 AKTS" in text and "seçmeli 16 / 20 AKTS (4 AKTS açık)" in text
    assert "GNO 3.1" in text and "transkriptteki güncel" in text
    assert "danışman" in text


def test_what_if_text_notes_courses_that_add_no_ects():
    report = graduation.simulate(WHAT_IF_SAMPLE, BIG_PLAN, ["Girişimcilik"])
    text = graduation.simulation_text(graduation.evaluate(WHAT_IF_SAMPLE, BIG_PLAN), [report], BIG_PLAN)
    assert "Girişimcilik zaten geçilmiş" in text


# --- What-if ve bu dönem açılan seçmeliler ----------------------------------------------------
# schedule.offered_electives çıktısının biçimi: bölüm yuvalarında açılan / açılmayan, rektörlük ve alan dışı
# yuvalarında "bilinmiyor" (onları başka birimler açar).
def _pool(*codes):
    names = {"PDO": "Python ile Derin Öğrenme", "SIB": "Siber Güvenlik", "YZ": "Yapay Zeka",
             "YSA": "Yapay Sinir Ağları"}
    return [{"kod": c, "ad": names[c], "akts": 4} for c in codes]


OFFERED = {"donem": "2026-2027 Güz", "kaynak": "test programı", "yuvalar": [
    {"ad": "Rek. Seçmeli I", "durum": "bilinmiyor", "acilan": [], "acilmayan": []},
    {"ad": "Rek. Seçmeli II", "durum": "bilinmiyor", "acilan": [], "acilmayan": []},
    {"ad": "Seçmeli VI", "durum": "acilmiyor", "acilan": [], "acilmayan": _pool("PDO", "SIB")},
    {"ad": "Seçmeli VII", "durum": "acilan_var", "acilan": _pool("YZ", "YSA"), "acilmayan": _pool("SIB")},
    {"ad": "Alan Dışı Seçmeli Ders", "durum": "bilinmiyor", "acilan": [], "acilmayan": []},
]}


def offered_text(names, offered=OFFERED):
    report = graduation.simulate(WHAT_IF_SAMPLE, BIG_PLAN, names)
    return graduation.simulation_text(graduation.evaluate(WHAT_IF_SAMPLE, BIG_PLAN), [report], BIG_PLAN,
                                      offered=offered)


def test_what_if_marks_a_course_not_offered_this_term():
    text = offered_text(["Siber Güvenlik"])
    assert "- Siber Güvenlik bu dönem açılmıyor (2026-2027 Güz ders programında yok)" in text


def test_what_if_does_not_mark_offered_courses():
    assert "açılmıyor" not in offered_text(["Yapay Sinir Ağları"]).split("Açık kalan")[0]


def test_what_if_course_offered_in_one_of_its_slots_counts_as_offered():
    yuvalar = [{**s, "acilan": _pool("SIB"), "acilmayan": [], "durum": "acilan_var"} if s["ad"] == "Seçmeli VII" else s
               for s in OFFERED["yuvalar"]]
    assert "Siber Güvenlik bu dönem açılmıyor" not in offered_text(["Siber Güvenlik"], {**OFFERED, "yuvalar": yuvalar})


def test_what_if_university_wide_course_is_unknown_not_closed():
    text = offered_text(["Diksiyon ve Hitabet"])
    assert "- Diksiyon ve Hitabet bölüm ders programında yer almıyor" in text
    assert "Diksiyon ve Hitabet bu dönem açılmıyor" not in text


def test_what_if_open_slots_list_only_offered_courses_not_yet_taken():
    # Seçmeli VII açık kalıyor (Yapay Zeka bu dönem alınıyor, 4 / 8): önerilebilecek tek açılan ders Yapay Sinir Ağları.
    text = offered_text(["Diksiyon ve Hitabet"])
    assert "Açık kalan seçmeli yuvaları: Seçmeli VII (8 AKTS; bu dönem açılanlar: Yapay Sinir Ağları)" in text
    assert "yalnız bu dönem açılan" in text and "Kaynak: test programı" in text


def test_what_if_open_slot_that_is_not_offered_says_so():
    yuvalar = [{**s, "acilan": [], "acilmayan": _pool("YZ", "YSA"), "durum": "acilmiyor"} if s["ad"] == "Seçmeli VII"
               else s for s in OFFERED["yuvalar"]]
    text = offered_text(["Diksiyon ve Hitabet"], {**OFFERED, "yuvalar": yuvalar})
    assert "Seçmeli VII (8 AKTS; bu dönem açılmıyor)" in text


def test_what_if_without_offered_electives_is_unchanged():
    assert "açılmıyor" not in graduation.simulation_text(
        graduation.evaluate(WHAT_IF_SAMPLE, BIG_PLAN), [graduation.simulate(WHAT_IF_SAMPLE, BIG_PLAN, ["Siber Güvenlik"])],
        BIG_PLAN)


def test_llm_summary_states_missing_gpa_in_words():
    # İlk dönem öğrencisinin notlu dersi yok; "GNO None" LLM'i şaşırtıyordu.
    report = graduation.evaluate(transcript(course(2026, "Güz", "Z1", "Matematik I", 6, None), gpa=None), PLAN)
    text = graduation.summary_text(report, PLAN)
    assert "None" not in text
    assert "GNO henüz yok (notlu ders yok)" in text


@pytest.mark.parametrize("enrolled, status", [(2023, "isteğe bağlı"), (2025, "zorunlu")])
def test_summary_can_state_the_enrollment_year_and_out_of_field_status(enrolled, status):
    report = graduation.evaluate(transcript(course(enrolled, "Güz", "Z1", "Matematik I", 6, "CC"), enrolled=enrolled),
                                 RULES_PLAN)
    text = graduation.summary_text(report, RULES_PLAN, enrollment=True)
    assert text.splitlines()[1] == (f"Kayıt yılı: {enrolled}-{enrolled + 1} "
                                    f"(Alan Dışı Seçmeli Ders bu öğrenci için {status}).")
    assert "Kayıt yılı" not in graduation.summary_text(report, RULES_PLAN)  # varsayılan özet değişmez


def test_equivalence_table_matches_a_known_erasmus_course_with_its_source():
    # Benzerlikle eşlenmeyen ders ("Microcomputer Systems" ~ "Microprocessors": 0) bölümün tablosundan eşlenir.
    plan = graduation.load_plan()
    erasmus = {**course(2025, "Güz", "E9", "Microcomputer Systems", 5, "AA"), "erasmus": True}
    report = graduation.evaluate(transcript(erasmus), plan)
    [m] = report["eslesmeler"]
    assert (m["plan_ad"], m["yontem"]) == ("Mikroişlemciler", "tablo") and m["kaynak"]
    assert report["eslesmeyen"] == []
    assert "eşdeğerlik tablosu" in graduation.summary_text(report, plan)


def test_user_confirmation_overrides_the_equivalence_table():
    plan = graduation.load_plan()
    erasmus = {**course(2025, "Güz", "E9", "Microcomputer Systems", 5, "AA"), "erasmus": True}
    report = graduation.evaluate(transcript(erasmus), plan, confirmations={"E9": None})
    assert report["eslesmeler"] == [] and report["eslesmeyen"][0]["teyit_edildi"] is True
