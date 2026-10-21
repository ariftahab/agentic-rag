"""Mezuniyet değerlendirmesi: transkripti öğretim planıyla karşılaştırır. LLM kullanılmaz.

Agent'ın transkript tool'ları (analiz, what-if) bu modülün üzerine kurulur; hesap burada,
cümleye dökmek LLM'de.

Komut satırı (gerçek transkriptle elle doğrulamak için):
    python graduation.py <tarihsel_transkript.pdf>
"""
import json
import sys

import config

# Önlisans ve Lisans Eğitim-Öğretim Yönetmeliği'nin not tablosu: DD ve üstü geçer, B (başarılı)
# not ortalamasına katılmadan geçer.
PASSING_GRADES = {"AA", "BA", "BB", "CB", "CC", "DC", "DD", "B"}
# Bir akademik yıl içindeki sıra; yıl akademik yılın başlangıcı olduğu için (yıl, sıra) kronolojiktir.
TERM_ORDER = {"Güz": 0, "Bahar": 1, "Yaz": 2}
STATUS_TEXT = {"alinmadi": "alınmadı", "basarisiz": "başarısız", "devam_ediyor": "bu dönem alınıyor"}


def _term_key(course):
    return course["yil"], TERM_ORDER[course["donem"]]


def evaluate(transcript, plan):
    courses = transcript["dersler"]
    latest_term = max(map(_term_key, courses))
    passed = [c for c in courses if c["not"] in PASSING_GRADES]
    # Notu boş dersler yalnız en son dönemdeyse "bu dönem alınıyor" sayılır.
    ongoing = [c for c in courses if c["not"] is None and _term_key(c) == latest_term]
    passed_ects = sum(c["akts"] for c in passed)

    passed_codes = {c["kod"] for c in passed}
    ongoing_codes = {c["kod"] for c in ongoing}
    taken_codes = {c["kod"] for c in courses}
    missing = []
    for course in sorted(plan["zorunlu_dersler"], key=lambda c: c["yariyil"]):
        if course["kod"] in passed_codes:
            continue
        if course["kod"] in ongoing_codes:
            status = "devam_ediyor"
        elif course["kod"] in taken_codes:
            status = "basarisiz"
        else:
            status = "alinmadi"
        missing.append({**course, "durum": status})

    return {
        "gecilen_akts": passed_ects,
        "devam_edenler_gecilirse_akts": passed_ects + sum(c["akts"] for c in ongoing),
        "eksik_zorunlu": missing,
    }


def load_plan(path=config.PLAN_PATH):
    return json.loads(path.read_text(encoding="utf-8"))


def report_text(report, plan):
    lines = [
        f"Öğretim planı: {plan['kaynak']}",
        f"Şu an geçilen AKTS: {report['gecilen_akts']}",
        f"Bu dönemin dersleri geçilirse AKTS: {report['devam_edenler_gecilirse_akts']}",
        f"Eksik zorunlu dersler ({len(report['eksik_zorunlu'])}):",
    ]
    lines += [f"  - {c['yariyil']}. yarıyıl  {c['ad']} ({c['akts']} AKTS): {STATUS_TEXT[c['durum']]}"
              for c in report["eksik_zorunlu"]]
    return "\n".join(lines)


if __name__ == "__main__":
    import transcript

    try:
        parsed = transcript.parse_transcript(sys.argv[1])
    except transcript.TranscriptFormatError as e:
        sys.exit(str(e))
    plan = load_plan()
    print(report_text(evaluate(parsed, plan), plan))
