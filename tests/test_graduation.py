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
