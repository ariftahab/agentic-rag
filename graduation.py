"""Mezuniyet değerlendirmesi: transkripti öğretim planıyla karşılaştırır. LLM kullanılmaz.

Agent'ın transkript tool'ları (analiz, what-if) bu modülün üzerine kurulur; hesap burada,
cümleye dökmek LLM'de.

Komut satırı (gerçek transkriptle elle doğrulamak için):
    python graduation.py <tarihsel_transkript.pdf>
"""
import json
import sys

import config

# Yönetmelik ve yönerge kuralları (geçer notlar, 240 AKTS, GNO, Alan Dışı Seçmeli Ders...);
# her kural kaynağını (belge + madde/fıkra) veri olarak taşır.
RULES_PATH = config.DATA_DIR / "plan" / "yonetmelik_kurallari.json"
# Bir akademik yıl içindeki sıra; yıl akademik yılın başlangıcı olduğu için (yıl, sıra) kronolojiktir.
TERM_ORDER = {"Güz": 0, "Bahar": 1, "Yaz": 2}
STATUS_TEXT = {"alinmadi": "alınmadı", "basarisiz": "başarısız", "devam_ediyor": "bu dönem alınıyor"}


def _term_key(course):
    return course["yil"], TERM_ORDER[course["donem"]]


def _latest_status(courses, passing):
    """Ders başına son durum: (şu an geçilmiş deneme, bu dönem devam eden deneme).

    Tekrarlanan derste son not geçerlidir; bu dönem yeniden alınan ve daha önce geçilmiş ders,
    şu an geçilmiş sayılır ve ikinci senaryoda bir kez daha sayılmaz."""
    latest_term = max(map(_term_key, courses))
    passed, ongoing = [], []
    by_code = {}
    for c in sorted(courses, key=_term_key):
        by_code.setdefault(c["kod"], []).append(c)
    for attempts in by_code.values():
        graded = [c for c in attempts if c["not"] is not None]
        if graded and graded[-1]["not"] in passing:
            passed.append(graded[-1])
            continue
        # Notu boş dersler yalnız en son dönemdeyse "bu dönem alınıyor" sayılır.
        current = [c for c in attempts if c["not"] is None and _term_key(c) == latest_term]
        if current:
            ongoing.append(current[-1])
    return passed, ongoing


def evaluate(transcript, plan, rules=None):
    """Transkripti plan ve kurallarla karşılaştırır; iki senaryo: şu an ve bu dönemin dersleri geçilirse.

    `rules` verilmezse yönetmelik kuralları `RULES_PATH`'ten okunur."""
    rules = rules or load_rules()
    courses = transcript["dersler"]
    passed, ongoing = _latest_status(courses, set(_rule(rules, "gecer_notlar")["deger"]))
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

    adsd_required = _adsd_required(transcript, rules)
    report = {
        "gecilen_akts": passed_ects,
        "devam_edenler_gecilirse_akts": passed_ects + sum(c["akts"] for c in ongoing),
        "eksik_zorunlu": missing,
        "siniflandirma": _classify(courses, plan),
        "secmeli": _electives(passed, passed + ongoing, plan, adsd_required),
        "gno": transcript.get("obs_ozet", {}).get("gno"),
        "kayit_yili": transcript.get("kayit_yili"),
    }
    report["kurallar"] = _rule_results(report, plan, rules, adsd_required)
    report["tum_sartlar_karsilandi"] = all(r["karsilandi"] for r in report["kurallar"])
    report["devam_edenler_gecilirse_tum_sartlar_karsilandi"] = all(
        r["devam_edenler_gecilirse_karsilandi"] for r in report["kurallar"])
    return report


def _rule(rules, rule_id):
    return next(r for r in rules["kurallar"] if r["id"] == rule_id)


def _plan_source(plan, section):
    return {"belge": plan["kaynak"], "madde": section}


def _adsd_required(transcript, rules):
    """Alan Dışı Seçmeli Ders kayıt yılına göre zorunlu mu? Kayıt yılı akademik yılın başlangıcıdır
    (2023 = 2023-2024). Kayıt yılı bilinmiyorsa zorunlu sayılır: eksik göstermek, gizlemekten güvenli."""
    year = transcript.get("kayit_yili")
    return year is None or year >= _rule(rules, "alan_disi_secmeli")["zorunlu_ilk_kayit_yili"]


def _plan_total(plan, adsd_required):
    slots = [s for s in plan.get("secmeli_yuvalar", []) if adsd_required or s["tur"] != "alan_disi"]
    return sum(c["akts"] for c in plan["zorunlu_dersler"]) + sum(s["akts"] for s in slots)


def _result(rule, sources, required, now, later, met, met_later, **extra):
    return {"id": rule["id"], "aciklama": rule["aciklama"], "kaynak": sources, "gereken": required,
            "mevcut": now, "devam_edenler_gecilirse": later, "karsilandi": met,
            "devam_edenler_gecilirse_karsilandi": met_later, **extra}


def _rule_results(report, plan, rules, adsd_required):
    """Her mezuniyet şartı için sonuç ve kaynağı; iki senaryo ayrı ayrı."""
    results = []

    # Toplam AKTS: yönetmelikteki alt sınır ile öğretim planının toplamından büyük olanı bağlayıcıdır.
    rule = _rule(rules, "toplam_akts")
    required = max(rule["deger"], _plan_total(plan, adsd_required))
    now, later = report["gecilen_akts"], report["devam_edenler_gecilirse_akts"]
    results.append(_result(rule, rule["kaynak"] + [_plan_source(plan, "yarıyıl toplamları")],
                           required, now, later, now >= required, later >= required))

    rule = _rule(rules, "zorunlu_dersler")
    missing = report["eksik_zorunlu"]
    total = len(plan["zorunlu_dersler"])
    still_missing = [c for c in missing if c["durum"] != "devam_ediyor"]
    results.append(_result(rule, [_plan_source(plan, "yarıyıl tablolarındaki zorunlu dersler")] + rule["kaynak"],
                           total, total - len(missing), total - len(still_missing),
                           not missing, not still_missing))

    rule = _rule(rules, "secmeli_akts")
    s = report["secmeli"]
    results.append(_result(
        rule, [_plan_source(plan, "seçmeli ders yuvaları ve havuzları")] + rule["kaynak"]
        + _rule(rules, "fazla_basarisiz_secmeli")["kaynak"],
        s["gereken_akts"], s["gecilen_akts"], s["devam_edenler_gecilirse_akts"],
        s["acik_akts"] == 0, s["devam_edenler_gecilirse_acik_akts"] == 0))

    rule = _rule(rules, "alan_disi_secmeli")
    adsd = [y for y in s["yuvalar"] if y["tur"] == "alan_disi"]
    done = any(y["doldu"] for y in adsd)
    done_later = any(y["devam_edenler_gecilirse_doldu"] for y in adsd)
    if adsd_required:
        results.append(_result(rule, rule["kaynak"] + [_plan_source(plan, "Alan Dışı Seçmeli Ders yuvası")],
                               1, int(done), int(done_later), done, done_later, zorunlu=True))
    else:
        results.append(_result({**rule, "aciklama": rule["istisna"]["aciklama"]}, rule["istisna"]["kaynak"],
                               0, int(done), int(done_later), True, True, zorunlu=False))

    # What-if GNO hesabı yok: ikinci senaryoda da transkriptteki güncel GNO kullanılır.
    rule = _rule(rules, "gno_alt_siniri")
    gpa = report["gno"]
    met = gpa is not None and gpa >= rule["deger"]
    results.append(_result(rule, rule["kaynak"], rule["deger"], gpa, gpa, met, met))
    return results


def graduation_requirements(plan=None, rules=None):
    """Transkript olmadan mezuniyet şartlarının kaynaklı özeti (get_graduation_requirements tool'u için)."""
    plan = plan or load_plan()
    rules = rules or load_rules()
    slots = sorted(plan.get("secmeli_yuvalar", []), key=lambda s: s["yariyil"])
    adsd = _rule(rules, "alan_disi_secmeli")
    return {
        "bolum": plan["bolum"],
        "katalog_yili": plan["katalog_yili"],
        "ogretim_plani": plan["kaynak"],
        "plan_toplam_akts": _plan_total(plan, True),
        "zorunlu_akts": sum(c["akts"] for c in plan["zorunlu_dersler"]),
        "zorunlu_ders_sayisi": len(plan["zorunlu_dersler"]),
        "secmeli_akts": sum(s["akts"] for s in slots if s["tur"] != "alan_disi"),
        "secmeli_yuvalar": [{k: s[k] for k in ("ad", "yariyil", "akts", "tur")} for s in slots],
        "kurallar": [
            {k: r[k] for k in ("id", "aciklama", "deger", "kaynak") if k in r}
            for r in rules["kurallar"] if r["id"] != "alan_disi_secmeli"
        ] + [
            {"id": adsd["id"], "aciklama": adsd["aciklama"], "kaynak": adsd["kaynak"],
             "zorunlu_ilk_kayit_yili": adsd["zorunlu_ilk_kayit_yili"],
             "akts": sum(s["akts"] for s in slots if s["tur"] == "alan_disi")},
            {"id": "alan_disi_secmeli_istisna", **adsd["istisna"]},
        ],
    }


def _fill_slots(slots, passed):
    """Geçilen havuz derslerini yuvalara dağıtır: yuvalar yarıyıl sırasıyla, her yuvaya önce en az
    yuvada geçerli olan ders (ör. iki havuzda bulunan ders sona kalır), sonra en eski ders."""
    pools = [{m["kod"] for m in slot["havuz"]} for slot in slots]
    used, filled = set(), []
    for slot, pool in zip(slots, pools):
        candidates = [c for c in passed if c["kod"] in pool and c["kod"] not in used]
        candidates.sort(key=lambda c: (sum(c["kod"] in p for p in pools), _term_key(c)))
        chosen, ects = [], 0
        for c in candidates:
            if ects >= slot["akts"]:
                break
            chosen.append(c)
            used.add(c["kod"])
            ects += c["akts"]
        filled.append((chosen, ects >= slot["akts"]))
    return filled


def _electives(passed_now, passed_if_ongoing, plan, adsd_required=True):
    """Seçmeli şartı AKTS toplamıyla; yuvalar yalnız hangilerinin dolduğunu göstermek için.

    Alan Dışı Seçmeli Ders yuvası bu toplamın dışındadır (ayrı kural)."""
    slots = sorted(plan.get("secmeli_yuvalar", []), key=lambda s: s["yariyil"])
    counted = [s for s in slots if s["tur"] != "alan_disi"]
    pool_codes = {m["kod"] for s in counted for m in s["havuz"]}
    required = sum(s["akts"] for s in counted)
    now = sum(c["akts"] for c in passed_now if c["kod"] in pool_codes)
    if_ongoing = sum(c["akts"] for c in passed_if_ongoing if c["kod"] in pool_codes)
    yuvalar = [
        {"ad": slot["ad"], "yariyil": slot["yariyil"], "akts": slot["akts"], "tur": slot["tur"],
         "zorunlu": adsd_required or slot["tur"] != "alan_disi",
         "dersler": [c["ad"] for c in chosen_now], "doldu": done_now,
         "devam_edenler_gecilirse_dersler": [c["ad"] for c in chosen_later],
         "devam_edenler_gecilirse_doldu": done_later}
        for slot, (chosen_now, done_now), (chosen_later, done_later)
        in zip(slots, _fill_slots(slots, passed_now), _fill_slots(slots, passed_if_ongoing))
    ]
    return {
        "gereken_akts": required,
        "gecilen_akts": now,
        "acik_akts": max(0, required - now),
        "devam_edenler_gecilirse_akts": if_ongoing,
        "devam_edenler_gecilirse_acik_akts": max(0, required - if_ongoing),
        "fazla_akts": max(0, now - required),
        "yuvalar": yuvalar,
    }


def _classify(courses, plan):
    """Her ders (kod başına bir kez): zorunlu mu, hangi seçmeli yuvalarının havuzunda mı, planda yok mu.

    Eşleme ders koduyla yapılır; yuvalar adlarıyla tanımlanır (yuva kodları yıllar içinde değişti).
    """
    compulsory = {c["kod"] for c in plan["zorunlu_dersler"]}
    slots_of = {}
    for slot in plan.get("secmeli_yuvalar", []):
        for member in slot["havuz"]:
            slots_of.setdefault(member["kod"], []).append(slot["ad"])
    result, seen = [], set()
    for c in courses:
        if c["kod"] in seen:
            continue
        seen.add(c["kod"])
        if c["kod"] in compulsory:
            kind = "zorunlu"
        elif c["kod"] in slots_of:
            kind = "secmeli"
        else:
            kind = "planda_yok"
        result.append({"kod": c["kod"], "ad": c["ad"], "sinif": kind, "yuvalar": slots_of.get(c["kod"], [])})
    return result


def load_plan(path=config.PLAN_PATH):
    return json.loads(path.read_text(encoding="utf-8"))


def load_rules(path=RULES_PATH):
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
    s = report["secmeli"]
    lines += [
        f"Seçmeli AKTS: {s['gecilen_akts']} / {s['gereken_akts']} (açık {s['acik_akts']}, fazla {s['fazla_akts']}); "
        f"bu dönemin dersleri geçilirse {s['devam_edenler_gecilirse_akts']} (açık {s['devam_edenler_gecilirse_acik_akts']})",
        "Seçmeli yuvaları:",
    ]
    for y in s["yuvalar"]:
        state = "doldu" if y["doldu"] else ("bu dönem dolar" if y["devam_edenler_gecilirse_doldu"] else "boş/eksik")
        if not y["zorunlu"]:
            state += " (isteğe bağlı)"
        courses = y["devam_edenler_gecilirse_dersler"] or ["-"]
        lines.append(f"  - {y['yariyil']}. yarıyıl  {y['ad']} ({y['akts']} AKTS): {state}  [{', '.join(courses)}]")
    outside = [c for c in report["siniflandirma"] if c["sinif"] == "planda_yok"]
    lines.append(f"Planda olmayan dersler ({len(outside)}):")
    lines += [f"  - {c['ad']} ({c['kod']})" for c in outside]
    lines.append("Mezuniyet şartları (şu an / bu dönemin dersleri geçilirse):")
    yes_no = {True: "karşılandı", False: "karşılanmadı"}
    for r in report["kurallar"]:
        sources = "; ".join(f"{k['belge']}, {k['madde']}" for k in r["kaynak"])
        lines.append(f"  - {r['id']}: {r['mevcut']} / {r['gereken']} -> {yes_no[r['karsilandi']]}; "
                     f"{r['devam_edenler_gecilirse']} -> {yes_no[r['devam_edenler_gecilirse_karsilandi']]}")
        lines.append(f"      kaynak: {sources}")
    lines.append(f"Bütün şartlar: {yes_no[report['tum_sartlar_karsilandi']]}; bu dönemin dersleri geçilirse "
                 f"{yes_no[report['devam_edenler_gecilirse_tum_sartlar_karsilandi']]}")
    return "\n".join(lines)


if __name__ == "__main__":
    import transcript

    try:
        parsed = transcript.parse_transcript(sys.argv[1])
    except transcript.TranscriptFormatError as e:
        sys.exit(str(e))
    plan = load_plan()
    print(report_text(evaluate(parsed, plan), plan))
