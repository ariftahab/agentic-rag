"""UBYS tarihsel transkriptini okur: dönem dönem dersler, notlar ve UBYS özeti.

Çıktıda kişisel bilgi (ad, soyad, öğrenci numarası, TC) bulunmaz; yalnız mezuniyet hesabının
ihtiyaç duyduğu alanlar tutulur. Transkriptte bir dönem satırı beş sütundur (kod, ad,
"kredi | AKTS", harf notu, açıklama); iki dönem yan yana basıldığında satır on sütun olur.

Dönem başlığındaki yıl akademik yılın başlangıcıdır: 2025-2026 baharı "2025 Yılı Bahar Dönemi"
olarak basılır ve `{"yil": 2025, "donem": "Bahar"}` diye tutulur.
"""
import re

import pdfplumber

TERM_RE = re.compile(r"(\d{4}) Yılı (Güz|Bahar|Yaz) Dönemi")
CREDIT_ECTS_RE = re.compile(r"^\s*(\d+)\s*\|\s*(\d+)\s*$")
TERM_COLUMNS = 5  # bir dönemin tablo genişliği
SEMESTER_LAYOUT_RE = re.compile(r"\d+\s*\.\s*Yarıyıl")
ASK_FOR_HISTORICAL = "Lütfen UBYS'den tarihsel transkripti yükleyin."


class TranscriptFormatError(ValueError):
    """Belge desteklenen transkript biçiminde değil; mesaj kullanıcıya gösterilebilir."""


def _ects(cell):
    return int(CREDIT_ECTS_RE.match(cell).group(2))


def _course(cells, year, term):
    if len(cells) < TERM_COLUMNS:
        return None
    code, name, credit_ects, grade, _note = cells
    m = CREDIT_ECTS_RE.match(credit_ects or "")
    if not code or not m:
        return None
    code = code.strip()
    return {
        "yil": year,
        "donem": term,
        "kod": code.removesuffix("(*)").strip(),
        "ad": " ".join((name or "").split()),
        "kredi": int(m.group(1)),
        "akts": int(m.group(2)),
        "not": (grade or "").strip() or None,
        "erasmus": code.endswith("(*)"),
    }


def _number(text):
    return float(text.replace(",", ".")) if text and text.strip() not in ("", "-") else None


def _cumulative(row):
    """Özet tablosundaki "Birikimli :" satırı: alınan, tamamlanan (kredi | AKTS), ..., GNO."""
    for i, cell in enumerate(row):
        if (cell or "").startswith("Birikimli"):
            taken, completed, _calculated, _points, _term_gpa, gpa = row[i + 1:i + 7]
            return {
                "birikimli_alinan_akts": _ects(taken),
                "birikimli_tamamlanan_akts": _ects(completed),
                "gno": _number(gpa),
            }
    return None


def _enrollment(text):
    """Başlıktan yalnız hesap için gerekenler; ad, numara, TC ve tam kayıt tarihi alınmaz."""
    department = re.search(r"Bölüm\s*:\s*(.+?)(?:\s+Bölümü)?\s*$", text, re.M)
    level = re.search(r"Eğitim Düzeyi\s*:\s*(\S+)", text)
    enrolled = re.search(r"Kayıt Tarihi\s*:\s*\S+\s*/\s*(\d{4})\s*/\s*(\S+)", text)
    return {
        "bolum": department.group(1).strip() if department else None,
        "egitim_duzeyi": level.group(1) if level else None,
        "kayit_yili": int(enrolled.group(1)) if enrolled else None,
        "kayit_donemi": enrolled.group(2) if enrolled else None,
    }


def parse_transcript(source):
    """`source`: PDF yolu ya da dosya nesnesi. Dönüş: transkript sözlüğü (bkz. modül açıklaması)."""
    courses = []
    summary = None  # belgedeki son "Birikimli" satırı = en güncel dönem
    with pdfplumber.open(source) as pdf:
        first_page = pdf.pages[0].extract_text() or ""
        if SEMESTER_LAYOUT_RE.search(first_page):
            raise TranscriptFormatError(
                "Bu, yarıyıl düzenli transkript; bu dönem alınan seçmelileri göstermiyor. " + ASK_FOR_HISTORICAL)
        enrollment = _enrollment(first_page)
        for page in pdf.pages:
            for table in page.extract_tables():
                terms = {}  # sütun başlangıcı -> (yıl, dönem)
                for row in table:
                    summary = _cumulative(row) or summary
                    for start in range(0, len(row) - TERM_COLUMNS + 1, TERM_COLUMNS):
                        m = TERM_RE.search(row[start] or "")
                        if m:
                            terms[start] = (int(m.group(1)), m.group(2))
                    for start, (year, term) in terms.items():
                        course = _course(row[start:start + TERM_COLUMNS], year, term)
                        if course:
                            courses.append(course)
    if not courses or summary is None:
        raise TranscriptFormatError("Belgede transkript tablosu bulunamadı. " + ASK_FOR_HISTORICAL)
    return {**enrollment, "dersler": courses, "obs_ozet": summary}
