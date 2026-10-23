"""Mezuniyet değerlendirmesi: transkripti öğretim planıyla karşılaştırır. LLM kullanılmaz.

Agent'ın transkript tool'ları (analiz, what-if) bu modülün üzerine kurulur; hesap burada,
cümleye dökmek LLM'de.

Komut satırı (gerçek transkriptle elle doğrulamak için):
    python graduation.py <tarihsel_transkript.pdf>
"""
import json
import math
import re
import sys

import config

# Yönetmelik ve yönerge kuralları (geçer notlar, 240 AKTS, GNO, Alan Dışı Seçmeli Ders...);
# her kural kaynağını (belge + madde/fıkra) veri olarak taşır.
RULES_PATH = config.DATA_DIR / "plan" / "yonetmelik_kurallari.json"
# Ad benzerliğiyle eşlenemeyen plan dışı derslerin kesinleşmiş karşılıkları (ör. "Microcomputer Systems" →
# Mikroişlemciler): teyit her oturumda kaybolduğu için agent aynı dersi tekrar tekrar soruyordu.
EQUIVALENCES_PATH = config.DATA_DIR / "plan" / "esdegerlikler.json"
# Bir akademik yıl içindeki sıra; yıl akademik yılın başlangıcı olduğu için (yıl, sıra) kronolojiktir.
TERM_ORDER = {"Güz": 0, "Bahar": 1, "Yaz": 2}
STATUS_TEXT = {"alinmadi": "alınmadı", "basarisiz": "başarısız", "devam_ediyor": "bu dönem alınıyor"}
UNMATCHED_STATUS_TEXT = {"gecti": "geçti", "basarisiz": "başarısız", "devam_ediyor": "bu dönem alınıyor"}

# Plan dışı ders (Erasmus, intibak) ile plan dersinin adları arasında otomatik eşleme eşiği (kelime kümelerinin
# Jaccard benzerliği). Gerçek transkriptle seçildi: "Operating Systems" ~ "Operating Systems I" 0,67,
# "Computer Networks" ~ "Data Communication and Computer Networks" 0,5 eşlenir; "Microcomputer Systems" en fazla
# 0,33 alır ve kullanıcıya sorulur.
MATCH_THRESHOLD = 0.5
STOPWORDS = {"and", "of", "the", "in", "to", "for", "with", "a", "an", "ve", "ile", "için"}


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


def _words(name):
    name = name.replace("I", "ı").replace("İ", "i").lower()
    return {w for w in re.findall(r"\w+", name) if w not in STOPWORDS}


def name_similarity(a, b):
    """İki ders adının kelime kümelerinin Jaccard benzerliği (ortak kelime / toplam farklı kelime)."""
    wa, wb = _words(a), _words(b)
    return len(wa & wb) / len(wa | wb) if wa and wb else 0.0


def plan_courses(plan):
    """Plandaki bütün dersler (zorunlular ve seçmeli havuzları), kod başına bir kez; "zorunlu" alanıyla."""
    result = {c["kod"]: {**c, "zorunlu": True} for c in plan["zorunlu_dersler"]}
    for slot in plan.get("secmeli_yuvalar", []):
        for c in slot["havuz"]:
            result.setdefault(c["kod"], {**c, "zorunlu": False})
    return list(result.values())


def _best_match(name, candidates):
    """Adı en benzer plan dersi; eşik altındaysa ya da eşitlik zorunlu ders tercihiyle de çözülmüyorsa None."""
    scored = [(max(name_similarity(name, c["ad"]), name_similarity(name, c.get("ad_en", ""))), c)
              for c in candidates]
    best = max((s for s, _ in scored), default=0.0)
    if best < MATCH_THRESHOLD:
        return None, best
    top = [c for s, c in scored if s == best]
    if len(top) > 1:
        top = [c for c in top if c["zorunlu"]] or top
    return (top[0], best) if len(top) == 1 else (None, best)


def _match_unplanned(courses, plan, passed, ongoing, confirmations, equivalences):
    """Plan dışı dersleri plan derslerine eşler: önce kullanıcı teyidi, sonra eşdeğerlik tablosu (ad birebir aynı),
    sonra İngilizce/Türkçe ad benzerliği.

    Dönüş: (eşlemeler, eşlenemeyenler). Hedef adayları, transkriptte kendi koduyla geçilmiş plan dersleri dışındaki
    plan dersleridir."""
    by_code = {c["kod"]: c for c in plan_courses(plan)}
    passed_codes = {c["kod"] for c in passed}
    status = {**{c["kod"]: "devam_ediyor" for c in ongoing}, **{c["kod"]: "gecti" for c in passed}}
    candidates = [c for c in by_code.values() if c["kod"] not in passed_codes]
    matches, unmatched, seen = [], [], set()
    for c in courses:
        if c["kod"] in by_code or c["kod"] in seen:
            continue
        seen.add(c["kod"])
        base = {"kod": c["kod"], "ad": c["ad"], "akts": c["akts"], "durum": status.get(c["kod"], "basarisiz")}
        if c["kod"] in confirmations:
            target = by_code.get(confirmations[c["kod"]])
            if target is None:
                unmatched.append({**base, "teyit_edildi": True})
                continue
            matches.append({**base, "plan_kod": target["kod"], "plan_ad": target["ad"], "yontem": "teyit"})
            continue
        known = next((e for e in equivalences
                      if name_similarity(e["ad"], c["ad"]) == 1.0 and e["plan_kod"] in by_code), None)
        if known is not None:
            target = by_code[known["plan_kod"]]
            matches.append({**base, "plan_kod": target["kod"], "plan_ad": target["ad"], "yontem": "tablo",
                            "kaynak": known["kaynak"]})
            continue
        target, score = _best_match(c["ad"], candidates)
        if target is None:
            unmatched.append({**base, "teyit_edildi": False})
        else:
            matches.append({**base, "plan_kod": target["kod"], "plan_ad": target["ad"], "yontem": "otomatik",
                            "benzerlik": round(score, 2)})
    return matches, unmatched


def evaluate(transcript, plan, rules=None, confirmations=None, equivalences=None):
    """Transkripti plan ve kurallarla karşılaştırır; iki senaryo: şu an ve bu dönemin dersleri geçilirse.

    `rules` verilmezse yönetmelik kuralları `RULES_PATH`'ten okunur. `confirmations` kullanıcının teyit ettiği
    eşlemelerdir: {plan dışı ders kodu: plan ders kodu ya da eşdeğeri yoksa None}. Eşlenen plan dışı ders,
    bütün hesaplarda plan dersinin koduyla sayılır. `equivalences` verilmezse EQUIVALENCES_PATH'ten okunur."""
    rules = rules or load_rules()
    equivalences = load_equivalences() if equivalences is None else equivalences
    passing = set(_rule(rules, "gecer_notlar")["deger"])
    matches, unmatched = _match_unplanned(transcript["dersler"], plan, *_latest_status(transcript["dersler"], passing),
                                          confirmations or {}, equivalences)
    target_of = {m["kod"]: m["plan_kod"] for m in matches}
    courses = [{**c, "kod": target_of.get(c["kod"], c["kod"])} for c in transcript["dersler"]]
    passed, ongoing = _latest_status(courses, passing)
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
        "eslesmeler": matches,
        "eslesmeyen": unmatched,
        "secmeli": _electives(passed, passed + ongoing, plan, adsd_required),
        "bu_donem": [{"ad": c["ad"], "kod": c["kod"], "akts": c["akts"]} for c in ongoing],
        "gno": transcript.get("obs_ozet", {}).get("gno"),
        "kayit_yili": transcript.get("kayit_yili"),
    }
    report["kurallar"] = _rule_results(report, plan, rules, adsd_required)
    report["tum_sartlar_karsilandi"] = all(r["karsilandi"] for r in report["kurallar"])
    report["devam_edenler_gecilirse_tum_sartlar_karsilandi"] = all(
        r["devam_edenler_gecilirse_karsilandi"] for r in report["kurallar"])
    return report


class UnknownCourseError(ValueError):
    """What-if derslerinden biri öğretim planında yok; `names` bulunamayan adlardır."""

    def __init__(self, names):
        self.names = names
        super().__init__("Öğretim planında bulunamayan ders: " + ", ".join(names))


def _resolve_plan_courses(names, plan):
    """Ders adlarını (Türkçe ya da İngilizce, büyük/küçük harf fark etmez) veya kodlarını plan derslerine çevirir."""
    courses = plan_courses(plan)
    resolved, unknown = [], []
    for name in names:
        match = next((c for c in courses if c["kod"] == name.strip()
                      or name_similarity(name, c["ad"]) == 1.0 or name_similarity(name, c.get("ad_en", "")) == 1.0),
                     None)
        if match is None:
            unknown.append(name)
        else:
            resolved.append(match)
    if unknown:
        raise UnknownCourseError(unknown)
    return resolved


def simulate(transcript, plan, courses, rules=None, confirmations=None):
    """What-if: `courses` (plan ders adları ya da kodları) bu dönemin dersleriyle birlikte geçilmiş varsayılır.

    Varsayımsal dersler transkriptin en son dönemine notsuz eklenir; böylece `evaluate`'in ikinci senaryosu
    ("devam_edenler_gecilirse_*") "bu dönemin dersleri ve varsayımsal dersler geçilirse" olur, ilk senaryo değişmez.
    Yalnız AKTS ve ders şartları değişir; GNO transkriptteki güncel değerdir (varsayımsal notla hesap yok).
    Zaten geçilmiş ya da bu dönem alınan ders bir kez sayılır.

    Dönüş: `evaluate` raporu + "varsayimsal": [{kod, ad, akts, durum: "yeni" | "gecti" | "devam_ediyor"}].
    Plan dışı ad → UnknownCourseError."""
    rules = rules or load_rules()
    targets = _resolve_plan_courses(courses, plan)
    taken = transcript["dersler"]
    latest = max(taken, key=_term_key) if taken else {"yil": transcript.get("kayit_yili") or 0, "donem": "Güz"}
    base = evaluate(transcript, plan, rules, confirmations)
    # Varsayımsal dersin şu anki durumu: eşlemeler (Erasmus) uygulanmış transkriptte geçilmiş mi, bu dönem mi alınıyor?
    target_of = {m["kod"]: m["plan_kod"] for m in base["eslesmeler"]}
    mapped = [{**c, "kod": target_of.get(c["kod"], c["kod"])} for c in taken]
    passed, ongoing = _latest_status(mapped, set(_rule(rules, "gecer_notlar")["deger"])) if mapped else ([], [])
    state = {**{c["kod"]: "devam_ediyor" for c in ongoing}, **{c["kod"]: "gecti" for c in passed}}
    hypothetical = [{"yil": latest["yil"], "donem": latest["donem"], "kod": c["kod"], "ad": c["ad"],
                     "kredi": c.get("kredi"), "akts": c["akts"], "not": None, "erasmus": False} for c in targets]
    report = evaluate({**transcript, "dersler": taken + hypothetical}, plan, rules, confirmations)
    report["varsayimsal"] = [{"kod": c["kod"], "ad": c["ad"], "akts": c["akts"], "durum": state.get(c["kod"], "yeni")}
                             for c in targets]
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

    # Toplam AKTS: Madde 22/1'deki alt sınır. Eski kurallarda (plan_toplami_bagla) öğretim planının toplamı daha büyükse
    # o kullanılıyordu; bu mevzuatta yok, yalnız eski sürümler için duruyor (load_rules(regulation_only=True)).
    rule = _rule(rules, "toplam_akts")
    sources = rule["kaynak"]
    required = rule["deger"]
    if rule.get("plan_toplami_bagla"):
        required = max(required, _plan_total(plan, adsd_required))
        sources = sources + [_plan_source(plan, "yarıyıl toplamları")]
    now, later = report["gecilen_akts"], report["devam_edenler_gecilirse_akts"]
    results.append(_result(rule, sources, required, now, later, now >= required, later >= required))

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
    summary = {
        "bolum": plan["bolum"],
        "katalog_yili": plan["katalog_yili"],
        "ogretim_plani": plan["kaynak"],
        "zorunlu_akts": sum(c["akts"] for c in plan["zorunlu_dersler"]),
        "zorunlu_ders_sayisi": len(plan["zorunlu_dersler"]),
        "stajlar": [{k: c[k] for k in ("ad", "yariyil", "akts")} for c in plan["zorunlu_dersler"]
                    if c["ad"].lower().startswith("staj")],
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
    if _rule(rules, "toplam_akts").get("plan_toplami_bagla"):  # eski kurallar: planın toplamı da yazılır
        summary["plan_toplam_akts"] = _plan_total(plan, True)
    return summary


SLOT_KIND_TEXT = {"bolum": "bölüm seçmelisi", "rektorluk": "rektörlük seçmelisi", "alan_disi": "alan dışı seçmeli"}


def requirements_text(summary):
    """get_graduation_requirements tool'unun LLM'e verdiği metin: planın toplamları, stajlar, seçmeli yuvaları ve
    her biri kaynağıyla mezuniyet kuralları. Transkript gerekmez; kişiye özel durum analyze_transcript'tedir."""
    adsd = next(r for r in summary["kurallar"] if r["id"] == "alan_disi_secmeli")
    lines = [
        f"Öğretim planı: {summary['ogretim_plani']}",
        "",
        f"MEZUNİYET ŞARTLARI ({summary['bolum']}, {summary['katalog_yili']} öğretim planı)",
        (f"Öğretim planının toplamı {summary['plan_toplam_akts']} AKTS: " if "plan_toplam_akts" in summary
         else "Öğretim planındaki dersler: ") +
        f"{summary['zorunlu_ders_sayisi']} zorunlu ders "
        f"{summary['zorunlu_akts']} AKTS (stajlar dahil), bölüm ve rektörlük seçmelileri {summary['secmeli_akts']} AKTS, "
        f"Alan Dışı Seçmeli Ders {adsd['akts']} AKTS.",
        "",
        "STAJLAR (zorunlu ders olarak planda)",
    ]
    lines += [f"- {c['ad']} ({c['yariyil']}. yarıyıl, {c['akts']} AKTS)" for c in summary["stajlar"]]
    lines += ["", "SEÇMELİ YUVALARI (her yuva kendi havuzundaki derslerle, kendi AKTS'si kadar dolar)"]
    lines += [f"- {s['ad']} ({s['yariyil']}. yarıyıl, {s['akts']} AKTS, {SLOT_KIND_TEXT[s['tur']]})"
              for s in summary["secmeli_yuvalar"]]
    lines += ["", "KURALLAR"]
    for r in summary["kurallar"]:
        text = r["aciklama"]
        if r["id"] == "alan_disi_secmeli":
            text += f" Bu zorunluluk {r['zorunlu_ilk_kayit_yili']} ve sonrasında kaydolan öğrenciler içindir."
        sources = "; ".join(f"{k['belge']}, {k['madde']}" for k in r["kaynak"])
        lines.append(f"- {text} [{sources}]")
    return "\n".join(lines)


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
    """Seçmeli şartı yuva yuva: her yuva kendi havuzundaki derslerle, kendi AKTS'si kadar dolar (kullanıcı kararı
    2026-10-22). Sayılan AKTS, yuvaların doldurulan AKTS'lerinin toplamıdır (yuva başına en fazla yuvanın AKTS'si);
    bir yuvadaki fazlalık başka yuvanın açığını kapatmaz, "fazla_akts" olarak raporlanır.

    Alan Dışı Seçmeli Ders yuvası bu toplamın dışındadır (ayrı kural)."""
    slots = sorted(plan.get("secmeli_yuvalar", []), key=lambda s: s["yariyil"])
    counted = [s for s in slots if s["tur"] != "alan_disi"]
    pool_codes = {m["kod"] for s in counted for m in s["havuz"]}
    required = sum(s["akts"] for s in counted)
    filled_now, filled_later = _fill_slots(slots, passed_now), _fill_slots(slots, passed_if_ongoing)

    def slot_ects(filled):
        return sum(min(slot["akts"], sum(c["akts"] for c in chosen))
                   for slot, (chosen, _) in zip(slots, filled) if slot["tur"] != "alan_disi")

    now, if_ongoing = slot_ects(filled_now), slot_ects(filled_later)
    pool_ects = sum(c["akts"] for c in passed_now if c["kod"] in pool_codes)
    yuvalar = [
        {"ad": slot["ad"], "yariyil": slot["yariyil"], "akts": slot["akts"], "tur": slot["tur"],
         "zorunlu": adsd_required or slot["tur"] != "alan_disi",
         "dersler": [c["ad"] for c in chosen_now], "doldu": done_now,
         "devam_edenler_gecilirse_dersler": [c["ad"] for c in chosen_later],
         "devam_edenler_gecilirse_doldu": done_later}
        for slot, (chosen_now, done_now), (chosen_later, done_later)
        in zip(slots, filled_now, filled_later)
    ]
    return {
        "gereken_akts": required,
        "gecilen_akts": now,
        "acik_akts": max(0, required - now),
        "devam_edenler_gecilirse_akts": if_ongoing,
        "devam_edenler_gecilirse_acik_akts": max(0, required - if_ongoing),
        "fazla_akts": pool_ects - now,
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


def load_rules(path=RULES_PATH, regulation_only=False):
    """regulation_only=True: kuralların mevzuattan alıntısı ('alinti') kullanılır, mevzuatta olmayan eklemeler
    ('plan_toplami_bagla') çıkarılır (agent_v11 ve sonrası). Varsayılan eski davranış (eski deneyler için)."""
    rules = _load_rules(path)
    if not regulation_only:  # eski kurallar: mevzuat dışı eklemeler kuralların içinde, ayrı kayıtlar yok
        rules["kurallar"] = [r for r in rules["kurallar"] if not r.get("yalniz_mevzuat_modu")]
        return rules
    for r in rules["kurallar"]:
        if "alinti" in r:
            r["aciklama"] = r["alinti"]
        r.pop("plan_toplami_bagla", None)
    return rules


def _load_rules(path):
    return json.loads(path.read_text(encoding="utf-8"))


def load_equivalences(path=EQUIVALENCES_PATH):
    return json.loads(path.read_text(encoding="utf-8"))["esdegerlikler"] if path.exists() else []


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
    lines.append(f"Plan derslerine eşlenen plan dışı dersler ({len(report['eslesmeler'])}):")
    lines += [f"  - {m['ad']} ({m['kod']}) -> {m['plan_ad']}: {m['yontem']}" for m in report["eslesmeler"]]
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


def _value(v):
    """Kural değeri LLM metninde; boş değer (ilk dönemde GNO) "None" yerine "yok"."""
    return "yok" if v is None else v


def _gpa_text(report):
    """GNO; ilk dönem öğrencisinde notlu ders olmadığı için transkriptte yoktur."""
    return "GNO henüz yok (notlu ders yok)" if report["gno"] is None else f"GNO {report['gno']}"


def enrollment_text(report):
    """Kayıt yılı ve Alan Dışı Seçmeli Ders'in bu öğrenci için durumu (kayıt yılı akademik yılın başlangıcıdır)."""
    year = report["kayit_yili"]
    adsd = next(r for r in report["kurallar"] if r["id"] == "alan_disi_secmeli")
    status = "zorunlu" if adsd["zorunlu"] else "isteğe bağlı"
    if year is None:
        return f"Kayıt yılı: transkriptte yok (Alan Dışı Seçmeli Ders bu öğrenci için {status} sayıldı)."
    return f"Kayıt yılı: {year}-{year + 1} (Alan Dışı Seçmeli Ders bu öğrenci için {status})."


def summary_text(report, plan, enrollment=False):
    """analyze_transcript tool'unun LLM'e verdiği kısa özet: önce iki senaryo, sonra kaynaklı kurallar, eksik
    zorunlular, bu dönemin dersleri ve açık seçmeli yuvaları. Geçilmiş derslerin sınıflandırması ve dolu yuvalar
    dışarıda kalır; tam rapor (~11 bin karakter) verildiğinde LLM "bu dönem geçilirse" senaryosunu ve açık seçmeli
    AKTS'yi çoğu cevapta atlıyordu.

    enrollment=True: planın altına kayıt yılı ve ADSD'nin bu öğrenci için zorunlu olup olmadığı yazılır (agent_v6;
    önceki özette yalnız kuralın sonucu vardı, LLM "kayıt yılınızı kontrol edin" diyordu)."""
    s = report["secmeli"]
    rules = {r["id"]: r for r in report["kurallar"]}
    total, compulsory = rules["toplam_akts"], rules["zorunlu_dersler"]
    met = {True: "karşılanıyor", False: "karşılanmıyor"}
    lines = [f"Öğretim planı: {plan['kaynak']}"]
    if enrollment:
        lines.append(enrollment_text(report))
    lines += [
        "",
        "İKİ SENARYO",
        f"1) Şu an: {report['gecilen_akts']} / {total['gereken']} AKTS; zorunlu dersler {compulsory['mevcut']} / "
        f"{compulsory['gereken']}; seçmeli {s['gecilen_akts']} / {s['gereken_akts']} AKTS ({s['acik_akts']} AKTS açık); "
        f"{_gpa_text(report)}; mezuniyet şartları {met[report['tum_sartlar_karsilandi']]}.",
        f"2) Bu dönem alınan dersler geçilirse: {report['devam_edenler_gecilirse_akts']} / {total['gereken']} AKTS; "
        f"zorunlu dersler {compulsory['devam_edenler_gecilirse']} / {compulsory['gereken']}; seçmeli "
        f"{s['devam_edenler_gecilirse_akts']} / {s['gereken_akts']} AKTS ({s['devam_edenler_gecilirse_acik_akts']} AKTS "
        f"açık); mezuniyet şartları {met[report['devam_edenler_gecilirse_tum_sartlar_karsilandi']]}.",
    ]
    # Eşlenemeyen dersler senaryoların hemen altında: özetin sonundayken agent kullanıcıya sormuyordu.
    asked = [c for c in report["eslesmeyen"] if not c["teyit_edildi"]]
    if report["eslesmeyen"]:
        lines += ["", "EŞLENEMEYEN PLAN DIŞI DERSLER (Erasmus, intibak; AKTS'leri toplama sayıldı, plandaki "
                      "karşılıkları bilinmiyor)"]
        for c in report["eslesmeyen"]:
            note = "; kullanıcı plandaki bir derse karşılık gelmediğini teyit etti" if c["teyit_edildi"] else ""
            lines.append(f"- {c['ad']} ({c['akts']} AKTS, {UNMATCHED_STATUS_TEXT[c['durum']]}{note})")
    if asked:
        names = ", ".join(c["ad"] for c in asked)
        lines.append(f"Cevabın sonunda kullanıcıya şunu sor: \"{names} öğretim planındaki hangi derse sayıldı?\" "
                     "Cevabını confirm_equivalence ile kaydet.")
    lines += ["", "KURALLAR (şu an → bu dönem geçilirse)"]
    for r in report["kurallar"]:
        sources = "; ".join(f"{k['belge']}, {k['madde']}" for k in r["kaynak"])
        lines.append(f"- {r['aciklama']} gereken {r['gereken']}, şu an {_value(r['mevcut'])}, "
                     f"geçilirse {_value(r['devam_edenler_gecilirse'])}. [{sources}]")
    lines += ["", "EKSİK ZORUNLU DERSLER"]
    lines += [f"- {c['ad']} ({c['yariyil']}. yarıyıl, {c['akts']} AKTS, {STATUS_TEXT[c['durum']]})"
              for c in report["eksik_zorunlu"]]
    # Bu dönemin dersleri: "bu dönem hangi seçmelileri alıyorum" sorusu için (dolan yuvalar özette yok).
    kinds = {c["kod"]: c["sinif"] for c in report["siniflandirma"]}
    slot_of = {name: y["ad"] for y in s["yuvalar"] for name in y["devam_edenler_gecilirse_dersler"]}
    kind_text = {"zorunlu": "zorunlu", "planda_yok": "planda yok"}
    lines += ["", "BU DÖNEM ALINAN DERSLER"]
    for c in report["bu_donem"]:
        kind = (f"seçmeli: {slot_of.get(c['ad'], 'yuvalar dolu, fazla seçmeli')}" if kinds[c["kod"]] == "secmeli"
                else kind_text[kinds[c["kod"]]])
        lines.append(f"- {c['ad']} ({c['akts']} AKTS, {kind})")
    lines += ["", "AÇIK SEÇMELİ YUVALARI"]
    for y in s["yuvalar"]:
        if not y["doldu"]:
            then = "bu dönem geçilirse doluyor" if y["devam_edenler_gecilirse_doldu"] else "bu dönem geçilse de açık"
            lines.append(f"- {y['ad']} ({y['yariyil']}. yarıyıl, {y['akts']} AKTS): {then}")
    if report["eslesmeler"]:
        lines += ["", "PLAN DIŞI DERSLERİN EŞLEMESİ (Erasmus, intibak; kesin eşdeğerlik intibak komisyonu "
                      "kararına bağlıdır, cevapta bunu belirt)"]
        how = {"otomatik": "ders adının çevirisine göre otomatik eşlendi", "teyit": "kullanıcı teyit etti",
               "tablo": "bölümün eşdeğerlik tablosundan"}
        lines += [f"- {m['ad']} ({m['akts']} AKTS) → {m['plan_ad']}: {how[m['yontem']]}"
                  + (f" (eşdeğerlik tablosu: {m['kaynak']})" if m["yontem"] == "tablo" else "")
                  for m in report["eslesmeler"]]
    return "\n".join(lines)


def _remaining_electives(report, plan, transcript, later):
    """Açık seçmeli yuvalarını dolduracak ders sayısı, yuva başına (en az, en çok): açık AKTS havuzdaki derslerin en
    yüksek ve en düşük AKTS'sine bölünür. İsteğe bağlı Alan Dışı Seçmeli Ders yuvası sayılmaz."""
    ects_of = {c["ad"]: c["akts"] for c in transcript["dersler"]}
    pools = {s["ad"]: [m["akts"] for m in s["havuz"]] for s in plan.get("secmeli_yuvalar", [])}
    done, taken = ("devam_edenler_gecilirse_doldu", "devam_edenler_gecilirse_dersler") if later else ("doldu", "dersler")
    out = []
    for y in report["secmeli"]["yuvalar"]:
        if y[done] or not y["zorunlu"]:
            continue
        open_ects = y["akts"] - sum(ects_of.get(name, 0) for name in y[taken])
        out.append((y["ad"], math.ceil(open_ects / max(pools[y["ad"]])), math.ceil(open_ects / min(pools[y["ad"]]))))
    return out


def _count_text(low, high):
    return str(low) if low == high else f"{low}-{high}"


def history_text(transcript, report, plan, rules=None):
    """analyze_transcript'e eklenen ders geçmişi (agent_v12): dönem dönem bütün dersler ve harf notları, yarıyıl
    ortalamaları, harf notu dağılımı, başarısız olunan ve tekrarlanan dersler, mezuniyete kalan ders sayısı. EXP-008'de
    özette olmayan bilgiyi agent "yok" diye okudu ("kaldığın ders yok", kayıt yılından tahmin edilen dönem)."""
    import gpa  # gpa graduation'ı içe aktarır

    rules = rules or load_rules()
    passing = set(_rule(rules, "gecer_notlar")["deger"])
    courses = sorted(transcript["dersler"], key=_term_key)
    terms = {}
    for c in courses:
        terms.setdefault((c["yil"], c["donem"]), []).append(c)
    regular = [t for t in terms if t[1] != "Yaz"]
    current = report["bu_donem"] and max(terms)
    number = {t: i for i, t in enumerate(regular, 1)}

    lines = ["", "DERS GEÇMİŞİ (transkriptteki bütün dersler, dönem dönem harf notlarıyla; YANO: yarıyıl ağırlıklı not "
                 f"ortalaması) [{gpa.SOURCES['ortalama']}]",
             f"Transkriptte ders alınan dönem sayısı: {len(regular)} (güz ve bahar; yaz okulu ayrı yazılır)."]
    if current and current in number:
        lines.append(f"Şu anki dönem: {number[current]}. dönem ({current[0]}-{current[0] + 1} {current[1]}; notlar "
                     "henüz girilmedi).")
    for term, items in terms.items():
        label = f"{number[term]}. dönem" if term in number else "Yaz okulu"
        avg = gpa.term_average(items)
        head = f"- {label}, {term[0]}-{term[0] + 1} {term[1]}" + (f", YANO {avg:.2f}" if avg is not None else "")
        if term == current:
            head += " (bu dönem)"
        lines.append(head + ": " + ", ".join(f"{c['ad']} {c['not'] or 'not yok'}" for c in items))

    attempts = {}
    for c in courses:
        attempts.setdefault(c["kod"], []).append(c)
    last = [a[-1] for a in attempts.values() if a[-1]["not"] is not None]
    counts = {}
    for c in last:
        counts[c["not"]] = counts.get(c["not"], 0) + 1
    order = [g for _, g, _ in gpa.GRADES]
    keys = [g for g in order if g in counts] + sorted(g for g in counts if g not in order)
    lines += ["", "HARF NOTU DAĞILIMI (her dersin son notu)",
              ", ".join(f"{g}: {counts[g]}" for g in keys) or "Notlu ders yok."]

    failed = [a for a in attempts.values() if any(c["not"] is not None and c["not"] not in passing for c in a)]
    lines += ["", "BAŞARISIZ OLUNAN DERSLER (geçer not dışındaki notlar; şu anki durumuyla)"]
    for a in failed:
        fails = ", ".join(f"{c['not']} ({c['yil']}-{c['yil'] + 1} {c['donem']})" for c in a
                          if c["not"] is not None and c["not"] not in passing)
        end = a[-1]
        if end["not"] is None:
            status = "bu dönem tekrar alınıyor"
        elif end["not"] in passing:
            status = f"{end['yil']}-{end['yil'] + 1} {end['donem']} döneminde {end['not']} ile geçildi"
        else:
            status = "hâlâ başarısız"
        lines.append(f"- {end['ad']}: {fails} → {status}")
    if not failed:
        lines.append("- Başarısız olunan ders yok.")

    repeated = [a for a in attempts.values() if len(a) > 1]
    lines += ["", "TEKRARLANAN DERSLER (kaç kez alındığı ve notları)"]
    lines += [f"- {a[-1]['ad']}: {len(a)} kez ({', '.join(c['not'] or 'bu dönem' for c in a)})" for a in repeated]
    if not repeated:
        lines.append("- Tekrarlanan ders yok.")

    compulsory_now = len(report["eksik_zorunlu"])
    compulsory_later = sum(c["durum"] != "devam_ediyor" for c in report["eksik_zorunlu"])
    lines += ["", "MEZUNİYETE KALAN DERS SAYISI (eksik zorunlu dersler + açık seçmeli yuvalarını dolduracak dersler; "
                  "seçmeli sayısı havuzdaki derslerin AKTS'sine göre)"]
    for title, n, later in (("Şu an", compulsory_now, False),
                            ("Bu dönem alınan dersler geçilirse", compulsory_later, True)):
        slots = _remaining_electives(report, plan, transcript, later)
        low, high = sum(s[1] for s in slots), sum(s[2] for s in slots)
        detail = ", ".join(f"{name} {_count_text(lo, hi)}" for name, lo, hi in slots)
        lines.append(f"- {title}: {n} zorunlu + {_count_text(low, high)} seçmeli = {_count_text(n + low, n + high)} "
                     "ders" + (f" ({detail})" if detail else ""))
    return "\n".join(lines)


def _if_passed_line(report):
    """İkinci senaryonun tek satırlık özeti (what-if metninde her küme için)."""
    rules = {r["id"]: r for r in report["kurallar"]}
    total, compulsory, s = rules["toplam_akts"], rules["zorunlu_dersler"], report["secmeli"]
    return (f"{report['devam_edenler_gecilirse_akts']} / {total['gereken']} AKTS; zorunlu dersler "
            f"{compulsory['devam_edenler_gecilirse']} / {compulsory['gereken']}; seçmeli "
            f"{s['devam_edenler_gecilirse_akts']} / {s['gereken_akts']} AKTS ({s['devam_edenler_gecilirse_acik_akts']} "
            f"AKTS açık); {_gpa_text(report)} (transkriptteki güncel değer)")


def _offer_status(offered):
    """Havuz dersi kodu -> "acilan" | "acilmayan" (rektörlük ve alan dışı dersleri, programda yoksa, burada yok).
    İki havuzda bulunan ders, birinde açılıyorsa açılıyor sayılır."""
    status = {c["kod"]: "acilmayan" for slot in offered["yuvalar"] for c in slot["acilmayan"]}
    status.update({c["kod"]: "acilan" for slot in offered["yuvalar"] for c in slot["acilan"]})
    return status


def simulation_text(base, reports, plan, offered=None):
    """simulate_courses tool'unun LLM'e verdiği metin: önce varsayımsal ders olmadan "bu dönem geçilirse" durumu
    (`base` = evaluate raporu), sonra her küme (`reports` = simulate raporları) için sonuç, karşılanmayan şartlar
    (kaynaklı), eksik kalan zorunlular ve açık seçmeli yuvaları. Seçim danışmana bırakılır.

    `offered` (schedule.offered_electives çıktısı) verilirse bu dönem açılmayan varsayımsal seçmeliler işaretlenir,
    rektörlük ve alan dışı seçmelileri "bölüm programında yok" diye belirtilir ve açık yuvaların yanında yalnız bu
    dönem açılan, henüz alınmamış dersler listelenir."""
    status = _offer_status(offered) if offered else {}
    unknown_slots = {s["ad"] for s in offered["yuvalar"] if s["durum"] == "bilinmiyor"} if offered else set()
    pools = {s["ad"]: s for s in plan.get("secmeli_yuvalar", [])}
    met = {True: "karşılanıyor", False: "karşılanmıyor"}
    lines = [
        f"Öğretim planı: {plan['kaynak']}",
        "VARSAYIM: Bu dönem alınan dersler ve kümedeki dersler geçilirse. Yalnız AKTS ve ders şartları hesaplanır; "
        "GNO varsayımsal notlarla hesaplanmaz, transkriptteki güncel değerdir.",
        "",
        f"Varsayımsal ders olmadan: {_if_passed_line(base)}; mezuniyet şartları "
        f"{met[base['devam_edenler_gecilirse_tum_sartlar_karsilandi']]}.",
    ]
    for i, report in enumerate(reports, 1):
        names = ", ".join(c["ad"] for c in report["varsayimsal"])
        lines += ["", f"{i}. küme ({names}): mezuniyet şartları "
                      f"{met[report['devam_edenler_gecilirse_tum_sartlar_karsilandi']]}",
                  f"- {_if_passed_line(report)}"]
        for c in report["varsayimsal"]:
            if c["durum"] == "gecti":
                lines.append(f"- {c['ad']} zaten geçilmiş; ek AKTS getirmez.")
            elif c["durum"] == "devam_ediyor":
                lines.append(f"- {c['ad']} bu dönem zaten alınıyor; ek AKTS getirmez.")
            elif status.get(c["kod"]) == "acilmayan":
                lines.append(f"- {c['ad']} bu dönem açılmıyor ({offered['donem']} ders programında yok); "
                             "bu dönem alınamaz.")
            elif offered and c["kod"] not in status and any(
                    c["kod"] in {m["kod"] for m in pools[name]["havuz"]} for name in unknown_slots):
                lines.append(f"- {c['ad']} bölüm ders programında yer almıyor (rektörlük ya da alan dışı seçmelisi); "
                             "açılıp açılmadığı ilgili birimden teyit edilmeli.")
        for r in report["kurallar"]:
            if not r["devam_edenler_gecilirse_karsilandi"]:
                sources = "; ".join(f"{k['belge']}, {k['madde']}" for k in r["kaynak"])
                lines.append(f"- Karşılanmayan şart: {r['aciklama']} gereken {r['gereken']}, sonuç "
                             f"{_value(r['devam_edenler_gecilirse'])}. [{sources}]")
        missing = [c["ad"] for c in report["eksik_zorunlu"] if c["durum"] != "devam_ediyor"]
        if missing:
            lines.append(f"- Eksik kalan zorunlu dersler: {', '.join(missing)}")
        taken = {c["kod"] for c in report["siniflandirma"]}
        empty = [f"{y['ad']} ({y['akts']} AKTS{_offered_note(y['ad'], offered, taken)})"
                 for y in report["secmeli"]["yuvalar"] if y["zorunlu"] and not y["devam_edenler_gecilirse_doldu"]]
        if empty:
            lines.append(f"- Açık kalan seçmeli yuvaları: {', '.join(empty)}")
    if offered:
        lines += ["", f"Ders önerirken yalnız bu dönem açılan dersleri öner. Kaynak: {offered['kaynak']}"]
    lines += ["", "Cevapta her kümenin sonucunu ve hesabını aktar; hangi kümenin seçileceği danışman hocanın "
                  "kararıdır, soranı danışman hocasına yönlendir."]
    return "\n".join(lines)


def _offered_note(slot_name, offered, taken):
    """Açık yuvanın yanına: bu dönem açılan ve transkriptte (varsayımsal dersler dahil) olmayan dersler."""
    slot = next((s for s in offered["yuvalar"] if s["ad"] == slot_name), None) if offered else None
    if slot is None:
        return ""
    names = ", ".join(c["ad"] for c in slot["acilan"] if c["kod"] not in taken)
    if slot["durum"] == "bilinmiyor":
        return (f"; bölüm ders programında: {names}; diğerleri için ilgili birime sor" if names
                else "; bölüm ders programında yok, ilgili birime sor")
    if slot["durum"] == "acilmiyor":
        return "; bu dönem açılmıyor"
    return f"; bu dönem açılanlar: {names}" if names else "; bu dönem açılan dersler zaten alınmış"


if __name__ == "__main__":
    import transcript

    try:
        parsed = transcript.parse_transcript(sys.argv[1])
    except transcript.TranscriptFormatError as e:
        sys.exit(str(e))
    plan = load_plan()
    print(report_text(evaluate(parsed, plan), plan))
