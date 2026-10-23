"""Not ortalaması hesabı (gpa.py): puan -> harf notu, vize/final, AKTS ağırlıklı GNO, hedef not. LLM ve ağ yok.

Beklenen değerler yönetmelikten elle hesaplandı (sorular_v2'nin ortalama soruları ve fixture'ların UBYS özetindeki GNO).
"""
import pytest

import gpa
import graduation
from experiments import evaluation as ev

PLAN = graduation.load_plan()


def t(name):
    return ev.load_transcript(name)


@pytest.mark.parametrize("score,letter", [(100, "AA"), (90, "AA"), (89, "BA"), (85, "BA"), (80, "BB"), (75, "CB"),
                                          (74, "CC"), (65, "CC"), (62, "DC"), (60, "DC"), (59, "DD"), (55, "DD"),
                                          (54, "FD"), (50, "FD"), (49, "FF"), (0, "FF")])
def test_score_to_letter_follows_article_27(score, letter):
    assert gpa.letter(score) == letter


def test_exam_score_uses_default_weights():
    assert gpa.exam_score(50, 60) == 56
    assert gpa.exam_score(50, 60, final_weight=70) == 57
    assert gpa.exam_score(40, 65) == 55


def test_final_weight_outside_article_26_limits_is_rejected():
    with pytest.raises(ValueError):
        gpa.exam_score(50, 60, final_weight=80)


@pytest.mark.parametrize("name", ["ucuncu_sinif_2024", "uzatma_2021", "erasmus_2023"])
def test_current_gpa_reproduces_the_transcript_summary(name):
    report = gpa.simulate(t(name), PLAN, [])
    assert report["mevcut_gno"] == t(name)["obs_ozet"]["gno"]


def test_score_for_an_ongoing_course_updates_the_gpa():
    report = gpa.simulate(t("ucuncu_sinif_2024"), PLAN, [{"ad": "Algoritma Analizi", "puan": 62}])
    course = report["dersler"][0]
    assert (course["harf"], course["akts"], course["gecer"]) == ("DC", 6, True)
    assert report["yeni_gno"] == 3.06


def test_midterm_and_final_on_a_repeated_course_replace_the_last_grade():
    report = gpa.simulate(t("uzatma_2021"), PLAN, [{"ad": "Algoritma Analizi", "vize": 50, "final": 60}],
                          target_gpa=2.0)
    course = report["dersler"][0]
    assert (course["puan"], course["harf"]) == (56, "DD")
    assert report["yeni_gno"] == 2.02 and report["hedef_saglandi"] is True


def test_several_scores_take_credits_from_the_plan():
    courses = [{"ad": "Matematik I", "puan": 85}, {"ad": "Fizik I", "puan": 72}, {"ad": "Kimya", "puan": 58}]
    report = gpa.simulate(t("birinci_sinif_2026"), PLAN, courses)
    assert [(c["harf"], c["akts"]) for c in report["dersler"]] == [("BA", 6), ("CC", 6), ("DD", 6)]
    assert report["mevcut_gno"] is None and report["yeni_gno"] == 2.91  # 2,9166: kesilir, yuvarlanmaz
    assert report["verilen_derslerin_ortalamasi"] == 2.91


def test_minimum_grade_for_a_target_gpa():
    report = gpa.simulate(t("ucuncu_sinif_2024"), PLAN, [{"ad": "Algoritma Analizi"}], target_gpa=3.0)
    options = {o["harf"]: o for o in report["secenekler"]}
    assert options["DD"]["en_az_puan"] == 55 and options["DD"]["yeni_gno"] == 3.03
    assert options["FD"]["yeni_gno"] == 2.99 and not options["FD"]["hedef"]
    assert report["hedefi_saglayan_en_dusuk"]["harf"] == "DD"


def test_target_reached_only_with_a_failing_grade_is_reported_as_such():
    report = gpa.simulate(t("uzatma_2021"), PLAN, [{"ad": "Algoritma Analizi"}], target_gpa=2.0)
    lowest = report["hedefi_saglayan_en_dusuk"]
    assert lowest["harf"] == "FD" and lowest["gecer"] is False
    assert report["hedefi_saglayan_en_dusuk_gecer"]["harf"] == "DD"


def test_midterm_only_gives_the_minimum_final_for_each_letter():
    report = gpa.simulate(None, PLAN, [{"vize": 40}])
    options = {o["harf"]: o for o in report["secenekler"]}
    assert options["DD"]["en_az_final"] == 65
    assert options["AA"]["en_az_final"] is None  # 0,4·40 + 0,6·100 = 76 < 90: mümkün değil
    assert report["mevcut_gno"] is None and report["yeni_gno"] is None


def test_letter_grade_input_and_explicit_credits():
    report = gpa.simulate(None, PLAN, [{"ad": "Uydurma Ders", "harf": "bb", "akts": 5}])
    assert report["dersler"][0]["harf"] == "BB" and report["verilen_derslerin_ortalamasi"] == 3.5


def test_unknown_course_without_credits_is_an_error():
    with pytest.raises(ValueError, match="AKTS"):
        gpa.simulate(None, PLAN, [{"ad": "Uydurma Ders", "puan": 70}])


def test_more_than_one_course_without_a_grade_is_an_error():
    with pytest.raises(ValueError, match="bir ders"):
        gpa.simulate(t("ucuncu_sinif_2024"), PLAN, [{"ad": "Algoritma Analizi"}, {"ad": "Mikroişlemciler"}])


def test_report_text_cites_the_articles():
    text = gpa.report_text(gpa.simulate(t("ucuncu_sinif_2024"), PLAN, [{"ad": "Algoritma Analizi", "puan": 62}]))
    assert "DC" in text and "3.06" in text
    for madde in ("Madde 27", "Madde 28"):
        assert f"BAİBÜ Önlisans, Lisans Eğitim-Öğretim Yönetmeliği, {madde}" in text
