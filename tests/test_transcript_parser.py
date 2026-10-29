"""Transkript parser'ı: gerçek UBYS transkriptleriyle, yalnız yerelde çalışır.

Gerçek transkriptler ve onlardaki kişisel değerler tests/private/ altında durur ve git'e girmez:
kisisel.json dosya adlarını ve çıktıda bulunmaması gereken değerleri (ad, numara, TC...) tutar.
Dosyalar yoksa testler atlanır.
"""
import json
from pathlib import Path

import pytest

from tools import transcript

PRIVATE = Path(__file__).parent / "private"
PERSONAL_FILE = PRIVATE / "kisisel.json"
PERSONAL = json.loads(PERSONAL_FILE.read_text(encoding="utf-8")) if PERSONAL_FILE.exists() else {}
HISTORICAL = PRIVATE / PERSONAL.get("tarihsel_transkript", "-")
SEMESTER_LAYOUT = PRIVATE / PERSONAL.get("yariyil_transkript", "-")

needs_historical = pytest.mark.skipif(not HISTORICAL.is_file(), reason="gerçek tarihsel transkript yok")
needs_semester_layout = pytest.mark.skipif(not SEMESTER_LAYOUT.is_file(), reason="gerçek yarıyıl transkripti yok")


@needs_historical
def test_current_term_ungraded_courses_are_read():
    t = transcript.parse_transcript(HISTORICAL)
    current = [c for c in t["dersler"] if (c["yil"], c["donem"]) == (2026, "Güz")]
    assert sorted(c["ad"] for c in current) == sorted([
        "Bitirme Projesi I", "Yapay Zeka", "Yapay Sinir Ağları",
        "Sensor Ağları", "Bilişim Hukuku", "Nesnelerin İnterneti",
    ])
    assert all(c["not"] is None for c in current)
    assert sum(c["akts"] for c in current) == 25


@needs_historical
def test_no_course_row_is_skipped():
    # UBYS'nin son "Birikimli" satırındaki alınan AKTS, okunan bütün derslerin toplamına eşit olmalı.
    t = transcript.parse_transcript(HISTORICAL)
    assert t["obs_ozet"]["birikimli_alinan_akts"] == 192
    assert sum(c["akts"] for c in t["dersler"]) == 192


@needs_historical
def test_erasmus_course_is_flagged_and_code_cleaned():
    t = transcript.parse_transcript(HISTORICAL)
    networks = next(c for c in t["dersler"] if c["ad"] == "Computer Networks")
    assert networks == {"yil": 2025, "donem": "Bahar", "kod": "2FI102721", "ad": "Computer Networks",
                        "kredi": 4, "akts": 5, "not": "FF", "erasmus": True}


@needs_historical
def test_enrollment_info_is_read_and_personal_data_left_out():
    t = transcript.parse_transcript(HISTORICAL)
    assert t["bolum"] == "Bilgisayar Mühendisliği"
    assert t["egitim_duzeyi"] == "Lisans"
    assert (t["kayit_yili"], t["kayit_donemi"]) == (2023, "Güz")
    text = json.dumps(t, ensure_ascii=False)
    assert PERSONAL["kisisel_degerler"]
    for value in PERSONAL["kisisel_degerler"]:
        assert value not in text


@needs_semester_layout
def test_semester_layout_transcript_is_rejected():
    # Bu biçim bu dönemin seçmelilerini göstermediği için desteklenmez.
    with pytest.raises(transcript.TranscriptFormatError, match="tarihsel transkript"):
        transcript.parse_transcript(SEMESTER_LAYOUT)
