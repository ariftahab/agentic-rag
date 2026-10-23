"""Not ortalaması hesabı (LLM'siz): puan -> harf notu, vize/final -> başarı puanı, AKTS ağırlıklı GNO ve hedef not.

EXP-005 ve iki teşhiste (agent_v8 aktarım, agent_v9 devir) ortalama soruları her seferinde LLM'in formülü yeniden
kurmasına kaldı: standart YÖK katsayıları (BAİBÜ'de BB 3,50, DC 2,50), unutulan vize/final oranı, "AKTS bilinmiyor"
ve UBYS'deki GNO'yu tutturamayan yeniden hesaplar. Kural sabit ve sık soruluyor; bu yüzden hesap burada, deterministik.

    simulate(transcript, plan, courses, target_gpa=None, final_weight=60) -> rapor
    report_text(rapor) -> agent'a dönen metin (kaynaklı)

Kaynak: BAİBÜ Önlisans, Lisans Eğitim-Öğretim Yönetmeliği, Madde 26 (başarı notu), 27 (puan aralıkları, katsayılar),
28 (AKTS ağırlıklı ortalama, iki hane, yuvarlamasız, tekrar edilen derste son not).
"""
import math

import graduation

REGULATION = "BAİBÜ Önlisans, Lisans Eğitim-Öğretim Yönetmeliği"
SOURCES = {
    "agirlik": f"{REGULATION}, Madde 26, fıkra 1-2",
    "not": f"{REGULATION}, Madde 27, fıkra 1",
    "ortalama": f"{REGULATION}, Madde 28, fıkra 2-3",
}
# Madde 27: (en düşük puan, harf notu, katsayı); FD ve FF başarısız.
GRADES = [(90, "AA", 4.0), (85, "BA", 3.75), (80, "BB", 3.5), (75, "CB", 3.25), (65, "CC", 3.0), (60, "DC", 2.5),
          (55, "DD", 2.0), (50, "FD", 1.0), (0, "FF", 0.0)]
COEFF = {g: c for _, g, c in GRADES}
LOWEST_SCORE = {g: low for low, g, _ in GRADES}
FAILING = {"FD", "FF"}
DEFAULT_FINAL_WEIGHT = 60          # Madde 26/2: bilgi girilmemişse ara sınav %40, dönem sonu %60
FINAL_WEIGHT_RANGE = (40, 70)      # Madde 26/1: dönem sonu sınavının etkisi en az %40, en çok %70


def letter(score):
    return next(g for low, g, _ in GRADES if score >= low)


def exam_score(midterm, final, final_weight=DEFAULT_FINAL_WEIGHT):
    """Ders başarı puanı (Madde 26): vize ve finalin ağırlıklı toplamı; Madde 27'deki aralıkla doğrudan karşılaştırılır."""
    low, high = FINAL_WEIGHT_RANGE
    if not low <= final_weight <= high:
        raise ValueError(f"Dönem sonu sınavının etkisi %{low} ile %{high} arasında olmalı (Madde 26/1).")
    score = (midterm * (100 - final_weight) + final * final_weight) / 100
    return int(score) if score == int(score) else score


def _truncate(value):
    return math.floor(value * 100 + 1e-9) / 100  # Madde 28/2: virgülden sonra iki hane, yuvarlama yok


def _term_key(course):
    return course["yil"], 0 if course["donem"] == "Güz" else 1


def _same(a, b):
    return graduation.name_similarity(a, b) == 1.0


def _base(transcript):
    """Transkriptteki her dersin son harf notu: {ders adı: (akts, harf)} (Madde 28/3: tekrar edilen derste son not;
    B/S gibi ortalamaya katılmayan notlar ve devam eden dersler dışarıda)."""
    last = {}
    for c in sorted(transcript["dersler"] if transcript else [], key=_term_key):
        if c["not"] in COEFF:
            last[c["ad"]] = (c["akts"], c["not"])
    return last


def _gpa(grades):
    total = sum(a for a, _ in grades.values())
    return _truncate(sum(a * COEFF[g] for a, g in grades.values()) / total) if total else None


def term_average(courses):
    """Yarıyıl ağırlıklı not ortalaması (Madde 28/2): o dönem kayıtlanılan derslerin AKTS ağırlıklı katsayı
    ortalaması; Madde 27'deki harf notları esas (B ve notu henüz girilmemiş dersler dışarıda)."""
    graded = [c for c in courses if c["not"] in COEFF]
    total = sum(c["akts"] for c in graded)
    return _truncate(sum(c["akts"] * COEFF[c["not"]] for c in graded) / total) if total else None


def _credits(course, transcript, plan):
    """AKTS: kullanıcı verdiyse o; yoksa transkriptteki aynı ders, yoksa öğretim planı. (ders adı, akts)"""
    name = course.get("ad", "").strip()
    if course.get("akts"):
        return name, course["akts"]
    if not name:
        return name, None
    for c in sorted(transcript["dersler"] if transcript else [], key=_term_key, reverse=True):
        if _same(c["ad"], name):
            return c["ad"], c["akts"]
    match, _ = graduation._best_match(name, graduation.plan_courses(plan))
    if match is None:
        raise ValueError(f'"{name}" transkriptte ve öğretim planında bulunamadı; dersin AKTS\'sini ver.')
    return match["ad"], match["akts"]


def _known_grade(course, final_weight):
    """(puan, harf) ya da bilinmiyorsa (None, None)."""
    if course.get("harf"):
        grade = course["harf"].strip().upper()
        if grade not in COEFF:
            raise ValueError(f"Geçersiz harf notu: {course['harf']} (geçerli: {', '.join(COEFF)}).")
        return None, grade
    if course.get("puan") is not None:
        return course["puan"], letter(course["puan"])
    if course.get("vize") is not None and course.get("final") is not None:
        score = exam_score(course["vize"], course["final"], final_weight)
        return score, letter(score)
    return None, None


def simulate(transcript, plan, courses, target_gpa=None, final_weight=DEFAULT_FINAL_WEIGHT):
    """courses: [{"ad", "puan"?, "vize"?, "final"?, "harf"?, "akts"?}]. Notu bilinmeyen en fazla bir ders olabilir
    (adı ya da yalnız vizesi verilmiş): onun için her harf notunun puan/final eşiği ve yol açacağı GNO hesaplanır.

    GNO transkriptteki son notlardan yeniden hesaplanır (UBYS özetini tutturur, testli); verilen notlar aynı dersin
    önceki notunun yerine geçer (bu dönem alınan ya da tekrar edilen ders) veya yeni ders olarak eklenir."""
    base = _base(transcript)
    rows, unknown = [], None
    for course in courses:
        name, credits = _credits(course, transcript, plan)
        score, grade = _known_grade(course, final_weight)
        row = {"ad": name, "akts": credits, "puan": score, "harf": grade, "vize": course.get("vize"),
               "tekrar": name in base}
        if grade is None:
            if unknown is not None:
                raise ValueError("Notu bilinmeyen yalnız bir ders olabilir; diğer derslerin puanını ya da notunu ver.")
            unknown = row
        else:
            row["gecer"] = grade not in FAILING
        rows.append(row)

    def with_grades(extra):
        grades = dict(base)
        for r in rows:
            g = extra if r is unknown else r["harf"]
            if g is not None and r["akts"]:
                grades[r["ad"] or "?"] = (r["akts"], g)
        return grades

    known = {r["ad"] or "?": (r["akts"], r["harf"]) for r in rows if r["harf"] and r["akts"]}
    report = {"dersler": rows, "final_agirligi": final_weight, "hedef_gno": target_gpa,
              "transkript": transcript is not None, "mevcut_gno": _gpa(base),
              "ozet_gno": (transcript or {}).get("obs_ozet", {}).get("gno"),
              "verilen_derslerin_ortalamasi": _gpa(known) if unknown is None else None,
              "yeni_gno": None, "secenekler": None}
    if unknown is None:
        report["yeni_gno"] = _gpa(with_grades(None)) if transcript is not None else None
        if target_gpa is not None and report["yeni_gno"] is not None:
            report["hedef_saglandi"] = report["yeni_gno"] >= target_gpa
        return report

    options = []
    for low, grade, _ in GRADES:
        option = {"harf": grade, "gecer": grade not in FAILING, "en_az_puan": low}
        if unknown["vize"] is not None:  # yalnız vize verilmiş: bu harf için gereken en düşük final
            finals = [f for f in range(101) if exam_score(unknown["vize"], f, final_weight) >= low]
            option["en_az_final"] = finals[0] if finals else None
        new = _gpa(with_grades(grade)) if transcript is not None and unknown["akts"] else None
        option["yeni_gno"] = new
        option["hedef"] = None if target_gpa is None or new is None else new >= target_gpa
        options.append(option)
    report["secenekler"] = options
    reachable = [o for o in options if o["hedef"] and o.get("en_az_final", 0) is not None]
    if reachable:
        report["hedefi_saglayan_en_dusuk"] = reachable[-1]
        passing = [o for o in reachable if o["gecer"]]
        report["hedefi_saglayan_en_dusuk_gecer"] = passing[-1] if passing else None
    return report


def _num(value):
    return "–" if value is None else f"{value:.2f}" if isinstance(value, float) else str(value)


def report_text(report):
    """Agent'a dönen metin: sonuç, kullanılan kural ve varsayımlar, her kural kaynağıyla."""
    lines = ["NOT ORTALAMASI HESABI (yönetmelikteki tabloyla, deterministik)"]
    for r in report["dersler"]:
        name = r["ad"] or "Ders"
        if r["harf"]:
            score = f"{r['puan']} puan → " if r["puan"] is not None else ""
            status = "geçer" if r["gecer"] else "başarısız (ders tekrar alınır)"
            repeat = "; önceki notunun yerine geçer" if r["tekrar"] else ""
            lines.append(f"- {name} ({_num(r['akts'])} AKTS): {score}{r['harf']} (katsayı {COEFF[r['harf']]:.2f}), "
                         f"{status}{repeat} [{SOURCES['not']}]")
        else:
            given = f", vize {r['vize']}" if r["vize"] is not None else ""
            lines.append(f"- {name} ({_num(r['akts'])} AKTS{given}): notu bilinmiyor; seçenekler aşağıda.")
    if any(r["vize"] is not None for r in report["dersler"]):
        lines.append(f"Başarı puanı = vize × %{100 - report['final_agirligi']} + final × %{report['final_agirligi']}; "
                     f"dersin hocası başka oran duyurmadıysa varsayılan budur [{SOURCES['agirlik']}].")
    if report["transkript"]:
        current = report["mevcut_gno"]
        note = "" if report["ozet_gno"] in (None, current) else f" (UBYS özetinde {report['ozet_gno']})"
        lines.append(f"Şu anki GNO: {_num(current)}{note}." if current is not None
                     else "Şu anki GNO: henüz yok (transkriptte notlu ders yok).")
    else:
        lines.append("Transkript yüklü değil: GNO hesaplanmadı, yalnız verilen dersler değerlendirildi.")
    if report["verilen_derslerin_ortalamasi"] is not None and len(report["dersler"]) > 1:
        lines.append(f"Verilen derslerin AKTS ağırlıklı ortalaması: {_num(report['verilen_derslerin_ortalamasi'])}.")
    if report["yeni_gno"] is not None:
        lines.append(f"Bu notlarla yeni GNO: {_num(report['yeni_gno'])}.")
        if report.get("hedef_saglandi") is not None:
            verdict = "sağlanıyor" if report["hedef_saglandi"] else "sağlanmıyor"
            lines.append(f"Hedef {report['hedef_gno']:.2f}: {verdict}.")
    if report["secenekler"]:
        with_final = any("en_az_final" in o for o in report["secenekler"])
        lines.append("SEÇENEKLER (harf notu: " + ("gereken en düşük final" if with_final else "en düşük puan") +
                     ("; yeni GNO" if report["transkript"] else "") + ")")
        for o in report["secenekler"]:
            need = o.get("en_az_final", o["en_az_puan"]) if with_final else o["en_az_puan"]
            need = "mümkün değil (100 yetmez)" if need is None else need
            gpa_part = f"; GNO {_num(o['yeni_gno'])}" if o["yeni_gno"] is not None else ""
            mark = "" if o["hedef"] is None else (" ✓ hedef" if o["hedef"] else " ✗ hedef")
            lines.append(f"- {o['harf']}{'' if o['gecer'] else ' (başarısız)'}: {need}{gpa_part}{mark}")
        lowest = report.get("hedefi_saglayan_en_dusuk")
        if report["hedef_gno"] is not None:
            if not lowest:
                lines.append(f"Hedef {report['hedef_gno']:.2f} hiçbir notla sağlanmıyor.")
            else:
                lines.append(f"Hedef {report['hedef_gno']:.2f} için en düşük not: {lowest['harf']}"
                             f" (en az {lowest['en_az_puan']} puan).")
                safe = report.get("hedefi_saglayan_en_dusuk_gecer")
                if not lowest["gecer"] and safe:
                    lines.append(f"{lowest['harf']} başarısız bir not; dersi geçip hedefi sağlayan en düşük not "
                                 f"{safe['harf']} (en az {safe['en_az_puan']} puan).")
    lines.append(f"GNO AKTS ağırlıklıdır, virgülden sonra iki hane alınır (yuvarlama yok), tekrar edilen derste son not "
                 f"geçerlidir [{SOURCES['ortalama']}]. Harf notu ve katsayılar [{SOURCES['not']}].")
    return "\n".join(lines)
