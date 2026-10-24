"""Haftalık ders programı: PDF'ten bir kez yapılandırılmış veriye çevrilir, agent'ın get_course_schedule tool'u
buradan okur. RAG'a girmez; LLM kullanılmaz.

Komut satırı (veriyi yeniden üretmek için):
    python schedule.py
"""
import itertools
import json
import re
from datetime import date, timedelta

import pdfplumber

import config
import gpa
import graduation

PDF_PATH = config.DATA_DIR / "2026_27_Güz_Lisans_v2.pdf"
SCHEDULE_PATH = config.DATA_DIR / "schedule" / "bilgisayar_2026-2027_guz.json"
SOURCE = "BAİBÜ Bilgisayar Mühendisliği Bölümü 2026-2027 Güz Dönemi Ders Programı (data/2026_27_Güz_Lisans_v2.pdf)"
CALENDAR_SOURCE = "BAİBÜ 2026-2027 Eğitim-Öğretim Yılı Akademik Takvimi (data/akademik-takvim-2026-2027.pdf)"
HOLIDAY_SOURCE = "2429 sayılı Ulusal Bayram ve Genel Tatiller Hakkında Kanun, Madde 1"

DAYS = ["Pazartesi", "Salı", "Çarşamba", "Perşembe", "Cuma", "Cumartesi", "Pazar"]
# Tablo sütunları: sınıf -> (ders adı, derslik) sütun çiftleri. 2. ve 3. sınıfta aynı saatte iki ders olabiliyor;
# 4. sınıfta ders ile derslik arasında öğretim elemanı sütunu var (kullanılmıyor).
COLUMNS = {1: [(3, 4)], 2: [(5, 6), (7, 8)], 3: [(9, 10), (11, 12)], 4: [(13, 15)]}
# Programdaki ad -> öğretim planındaki ad (yalnız farklı yazılanlar).
ALIASES = {
    "Matematik-1": "Matematik I",
    "Fizik-1": "Fizik I",
    "A.I.I.T. - I": "Atatürk İlkeleri ve Inkilap Tarihi I",
    "Türk Dili - I": "Türk Dili I",
    "Yabancı Dil - I": "Yabancı Dil I",
    "Sensör Ağları": "Sensor Ağları",
    "İleri Veri Tabanı Yön. Sis.": "İleri Veri Tabanı Yönetim Sistemleri",
    "Sektör Kampüste Veri Yönetişimi": "Veri Yönetişimi",
}
ROOMS = {
    "A.L.": "Sistem Ağları Laboratuvarı",
    "B.L.": "Bilgisayar (Yazılım) Laboratuvarı",
    "D.L.": "Donanım Laboratuvarı",
}
CELL_RE = re.compile(r"^(?P<ad>.+?)(?:\s*\((?P<yuva>Seç [IVX]+)\))?(?:\s*\((?P<sube>[A-Z])\))?$")


def _room(text):
    text = text.strip()
    return text + "." if re.fullmatch(r"[A-Z]\.[A-Z]", text) else text  # "D.L" gibi eksik noktalı yazımlar


def parse_pdf(path=PDF_PATH, plan=None):
    """PDF tablosunu satır listesine çevirir: her satır bir gün, sınıf ve dersin peş peşe saatleri (blok)."""
    plan = plan or graduation.load_plan()
    codes = {c["ad"]: c["kod"] for c in graduation.plan_courses(plan)}
    with pdfplumber.open(path) as pdf:
        table = pdf.pages[0].extract_tables()[0]
    slots, day = [], -1
    for row in table:
        if row[1] == "Ders 1":
            day += 1
        if not (row[1] or "").startswith("Ders "):
            continue
        for grade, pairs in COLUMNS.items():
            for name_col, room_col in pairs:
                cell = (row[name_col] or "").strip()
                if not cell:
                    continue
                m = CELL_RE.match(cell)
                name = ALIASES.get(m["ad"].strip(), m["ad"].strip())
                if name not in codes:
                    raise ValueError(f"Programdaki ders öğretim planında yok: {cell!r}")
                slots.append({"gun": DAYS[day], "sinif": grade, "ders": name, "kod": codes[name],
                              "sube": m["sube"], "yuva": m["yuva"], "saat": row[2],
                              "derslik": _room(row[room_col] or "")})
    return _blocks(slots)


def _blocks(slots):
    """Aynı gün, sınıf, ders, şube ve derslikteki peş peşe saatleri tek blokta birleştirir."""
    blocks = []
    for s in slots:
        key = (s["gun"], s["sinif"], s["kod"], s["sube"], s["derslik"])
        last = next((b for b in reversed(blocks) if b["_key"] == key), None)
        if last and _next_slot(last["saatler"][-1], s["saat"]):
            last["saatler"].append(s["saat"])
            continue
        blocks.append({**{k: v for k, v in s.items() if k != "saat"}, "saatler": [s["saat"]], "_key": key})
    for b in blocks:
        del b["_key"]
    return blocks


def _next_slot(previous, current):
    """Ders saatleri saat başı 15 dakika arayla: 08:30-09:15'ten sonra 09:30-10:15 gelir."""
    return int(current[:2]) == int(previous[:2]) + 1


def write_schedule(path=SCHEDULE_PATH):
    data = {
        "kaynak": SOURCE,
        "donem": "2026-2027 Güz",
        "ders_baslangic": "2026-09-21",
        "ders_bitis": "2026-12-25",
        "takvim_kaynak": CALENDAR_SOURCE,
        "tatiller": [
            {"tarih": "2026-10-28", "baslangic": "13:00", "aciklama": "Cumhuriyet Bayramı arifesi (13.00'ten sonra)",
             "kaynak": HOLIDAY_SOURCE},
            {"tarih": "2026-10-29", "aciklama": "Cumhuriyet Bayramı", "kaynak": HOLIDAY_SOURCE},
        ],
        "derslikler": ROOMS,
        "dersler": parse_pdf(),
    }
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return data


def load_schedule(path=SCHEDULE_PATH):
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_day(text, today):
    """"bugün", "yarın", gün adı ("perşembe": bugünden itibaren ilk perşembe) ya da YYYY-AA-GG -> tarih."""
    t = text.strip().replace("I", "ı").replace("İ", "i").lower()
    if t in ("", "bugün"):
        return today
    if t == "yarın":
        return today + timedelta(days=1)
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", t):
        return date.fromisoformat(t)
    lowered = [d.replace("I", "ı").replace("İ", "i").lower() for d in DAYS]
    if t in lowered:
        return today + timedelta(days=(lowered.index(t) - today.weekday()) % 7)
    raise ValueError(f'Gün anlaşılamadı: "{text}". "bugün", "yarın", bir gün adı ya da YYYY-AA-GG kullan.')


def current_course_codes(transcript):
    """Transkriptte en son dönemin notsuz dersleri: öğrencinin bu dönem aldığı dersler."""
    latest = max((c["yil"], graduation.TERM_ORDER[c["donem"]]) for c in transcript["dersler"])
    return {c["kod"] for c in transcript["dersler"]
            if c["not"] is None and (c["yil"], graduation.TERM_ORDER[c["donem"]]) == latest}


def _same(a, b):
    norm = lambda s: re.sub(r"\W+", " ", s.replace("I", "ı").replace("İ", "i").lower()).strip()  # noqa: E731
    return norm(a) == norm(b)


def lookup(data, day, grade=None, courses=None, codes=None):
    """Bir tarihin dersleri. Filtreler: sınıf, ders adları (programdaki ya da plandaki ad), plan ders kodları.

    Dönüş: {"tarih", "gun", "durum": ders_var | ders_yok | tatil | hafta_sonu | donem_disi, "aciklama", "dersler"}."""
    result = {"tarih": day.isoformat(), "gun": DAYS[day.weekday()], "aciklama": None, "dersler": []}
    start, end = date.fromisoformat(data["ders_baslangic"]), date.fromisoformat(data["ders_bitis"])
    if not start <= day <= end:
        result.update(durum="donem_disi", aciklama=f"{data['donem']} dersleri {start.isoformat()} ile "
                                                   f"{end.isoformat()} arasında [{data['takvim_kaynak']}]")
        return result
    if day.weekday() >= 5:
        result["durum"] = "hafta_sonu"
        return result
    holiday = next((h for h in data["tatiller"] if h["tarih"] == day.isoformat()), None)
    if holiday and "baslangic" not in holiday:
        result.update(durum="tatil", aciklama=f"{holiday['aciklama']} [{holiday['kaynak']}]")
        return result
    rows = [b for b in data["dersler"] if b["gun"] == result["gun"]]
    if grade:
        rows = [b for b in rows if b["sinif"] == grade]
    if courses:
        names = [ALIASES.get(c.strip(), c) for c in courses]
        rows = [b for b in rows if any(_same(b["ders"], n) for n in names)]
    if codes is not None:
        rows = [b for b in rows if b["kod"] in codes]
    if holiday:  # yarım gün tatil: başlangıç saatinden sonraki ders saatleri düşer
        rows = [{**b, "saatler": [s for s in b["saatler"] if s[:5] < holiday["baslangic"]]} for b in rows]
        rows = [b for b in rows if b["saatler"]]
        result["aciklama"] = f"{holiday['aciklama']} [{holiday['kaynak']}]"
    result["dersler"] = sorted(rows, key=lambda b: (b["saatler"][0], b["sinif"]))
    result["durum"] = "ders_var" if rows else "ders_yok"
    return result


def schedule_text(result, data):
    """LLM için kısa metin: tarih ve durum, sonra her ders bloğu saat aralığı, şube ve dersliğiyle."""
    head = f"{result['tarih']} {result['gun']}"
    status = {"donem_disi": "ders dönemi dışında", "hafta_sonu": "hafta sonu, ders yok", "tatil": "resmi tatil, ders yok",
              "ders_yok": "bu filtreyle ders yok", "ders_var": "dersler"}[result["durum"]]
    lines = [f"{head}: {status}." + (f" {result['aciklama']}" if result["aciklama"] else "")]
    for b in result["dersler"]:
        time = f"{b['saatler'][0][:5]}-{b['saatler'][-1][-5:]}"
        room = data["derslikler"].get(b["derslik"], b["derslik"]) or "derslik belirtilmemiş"
        extra = ", ".join(x for x in (f"{b['sube']} şubesi" if b["sube"] else "", b["yuva"] or "") if x)
        lines.append(f"- {time} {b['ders']}" + (f" ({extra})" if extra else "") + f", {b['sinif']}. sınıf, {room}")
    if result["dersler"]:
        lines.append(f"Kaynak: {data['kaynak']}")
    return "\n".join(lines)


# --- Bu dönem açılan seçmeliler ---
# Programda bölüm seçmelileri "(Seç III)" gibi yuva etiketiyle yazılır; etiket plandaki "Seçmeli III" yuvasıdır.
# Ders adları programdan okunurken (parse_pdf) plan derslerine ve kodlarına çevrildiği için eşleme kodla yapılır.
# Rektörlük ve Alan Dışı seçmelilerini başka birimler açar; bölüm programında olmamaları açılmadıkları anlamına
# gelmediği için bu yuvalar "bilinmiyor" sayılır.
ROMAN = {"i": 1, "ii": 2, "iii": 3, "iv": 4, "v": 5, "vi": 6, "vii": 7, "viii": 8}


def label_slot_name(label):
    """Programdaki yuva etiketi -> plandaki yuva adı ("Seç III" -> "Seçmeli III")."""
    return re.sub(r"^Seç\s+", "Seçmeli ", label.strip())


def _ascii_words(text):
    """Büyük/küçük harf ve noktalı/noktasız i farkı olmadan kelimeler ("SEÇMELİ III" -> ["seçmeli", "iii"])."""
    return re.findall(r"\w+", text.replace("İ", "i").replace("I", "i").replace("ı", "i").lower())


def _slot_number(words):
    return next((int(w) if w.isdigit() else ROMAN[w] for w in words if w.isdigit() or w in ROMAN), None)


def find_slot(text, plan):
    """Kullanıcının yazdığı yuva ("Seçmeli III", "seç 3", "rektörlük seçmeli 1", "alan dışı") -> plan yuvası.

    Numara aynı olan yuvalardan, "rek" yazılmadıysa bölüm yuvası tercih edilir. Bulunamazsa ya da belirsizse
    yuva adlarını listeleyen ValueError."""
    slots = sorted(plan.get("secmeli_yuvalar", []), key=lambda s: s["yariyil"])
    words = _ascii_words(text)
    exact = [s for s in slots if _ascii_words(s["ad"]) == words]
    if "alan" in words:
        found = [s for s in slots if s["tur"] == "alan_disi"]
    elif exact:
        found = exact
    else:
        number = _slot_number(words)
        found = [s for s in slots if s["tur"] != "alan_disi" and _slot_number(_ascii_words(s["ad"])) == number]
        rectorate = any(w.startswith("rek") for w in words)
        preferred = [s for s in found if (s["tur"] == "rektorluk") == rectorate]
        found = preferred or found
    if len(found) != 1:
        names = ", ".join(s["ad"] for s in slots)
        raise ValueError(f'Seçmeli yuvası anlaşılamadı: "{text}". Yuvalar: {names}.')
    return found[0]


def _offered_course(course, blocks):
    return {"kod": course["kod"], "ad": course["ad"], "akts": course["akts"], "sinif": blocks[0]["sinif"],
            "dersler": [{k: b[k] for k in ("gun", "saatler", "sube", "derslik")} for b in blocks]}


def offered_electives(data, plan):
    """Plandaki her seçmeli yuvası için bu dönem açılan ve açılmayan dersler (LLM'siz).

    Bölüm yuvası: açılanlar programda o yuvanın etiketiyle geçen dersler, açılmayanlar havuzun geri kalanı;
    durum "acilan_var" ya da "acilmiyor". Rektörlük ve Alan Dışı yuvası: durum "bilinmiyor"; açılanlar yalnız
    havuzdan bölüm programında geçen dersler (ör. Mühendislik Etiği), açılmayan listesi boş.

    Dönüş: {"donem", "kaynak", "yuvalar": [{"ad", "yariyil", "akts", "tur", "durum", "acilan", "acilmayan"}]};
    açılan ders: {"kod", "ad", "akts", "sinif", "dersler": [{"gun", "saatler", "sube", "derslik"}]}."""
    courses = {c["kod"]: c for c in graduation.plan_courses(plan)}
    labelled, scheduled = {}, {}
    for b in data["dersler"]:
        scheduled.setdefault(b["kod"], []).append(b)
        if b["yuva"]:
            labelled.setdefault(label_slot_name(b["yuva"]), {}).setdefault(b["kod"], []).append(b)
    slots = []
    for slot in sorted(plan.get("secmeli_yuvalar", []), key=lambda s: s["yariyil"]):
        pool = [m["kod"] for m in slot["havuz"]]
        if slot["tur"] == "bolum":
            here = labelled.get(slot["ad"], {})
            codes = pool + [k for k in here if k not in pool and k in courses]  # havuzda olmayan etiketli ders
            offered = [_offered_course(courses[k], here[k]) for k in codes if k in here]
            closed = [{k: m[k] for k in ("kod", "ad", "akts")} for m in slot["havuz"] if m["kod"] not in here]
            status = "acilan_var" if offered else "acilmiyor"
        else:
            offered = [_offered_course(courses[k], scheduled[k]) for k in pool if k in scheduled]
            closed, status = [], "bilinmiyor"
        slots.append({"ad": slot["ad"], "yariyil": slot["yariyil"], "akts": slot["akts"], "tur": slot["tur"],
                      "durum": status, "acilan": offered, "acilmayan": closed})
    return {"donem": data["donem"], "kaynak": data["kaynak"], "yuvalar": slots}


def _lesson_text(lesson):
    time = f"{lesson['gun']} {lesson['saatler'][0][:5]}-{lesson['saatler'][-1][-5:]}"
    extra = [f"{lesson['sube']} şubesi"] if lesson["sube"] else []
    room = ROOMS.get(lesson["derslik"], lesson["derslik"])
    return f"{time} ({', '.join(extra + [room or 'derslik belirtilmemiş'])})"


def _slot_lines(slot):
    head = f"{slot['ad']} ({slot['yariyil']}. yarıyıl, {slot['akts']} AKTS)"
    courses = [f"- {c['ad']} ({c['akts']} AKTS, {c['sinif']}. sınıf): " + "; ".join(map(_lesson_text, c["dersler"]))
               for c in slot["acilan"]]
    closed = [f"Bu dönem açılmayanlar: {', '.join(c['ad'] for c in slot['acilmayan'])}"] if slot["acilmayan"] else []
    if slot["durum"] == "acilan_var":
        return [f"{head}: bu dönem açılanlar:"] + courses + closed
    if slot["durum"] == "acilmiyor":
        return [f"{head}: bu dönem açılmıyor; havuzundaki derslerin hiçbiri ders programında yok."] + closed
    if courses:
        return ([f"{head}: havuzdaki dersleri rektörlük ya da ilgili birimler açar; bölüm ders programında "
                 "yer alanlar:"] + courses + ["Diğer havuz derslerinin açılıp açılmadığını ilgili birimden teyit et."])
    return [f"{head}: havuzdaki dersleri rektörlük ya da ilgili birimler açar, bölüm ders programında yer almıyor; "
            "açılıp açılmadığını ilgili birimden teyit et."]


def offered_electives_text(offered, slot_name=None):
    """get_offered_electives tool'unun LLM'e verdiği metin: tek yuva (`slot_name`) ya da bütün yuvalar."""
    if slot_name:
        slot = next(s for s in offered["yuvalar"] if s["ad"] == slot_name)
        lines = _slot_lines(slot)
        lines[0] = f"{offered['donem']} dönemi, {lines[0]}"
    else:
        lines = [f"{offered['donem']} dönemi seçmeli yuvaları:"]
        for slot in offered["yuvalar"]:
            lines += ["", *_slot_lines(slot)]
    return "\n".join(lines + ["", f"Kaynak: {offered['kaynak']}"])


# --- Bütün haftada arama ve şube planı (agent_h6) ---
# EXP-005 - EXP-013'te "Yapay Zeka hangi gün?", "327'de pazartesi hangi dersler var?", "İşletim Sistemleri I'i B
# şubesinden alırsam neyle çakışır?" ve "Cuma boş kalsın, hangi şubeler?" sorularında Haiku program dosyasını her
# seferinde kodla süzdü; lookup() tek bir günü ve yalnız sınıf/ders filtresini biliyor.
MAX_EXAMPLES = 3


def _words(text):
    return set(gpa.course_key(text).split())


def _course_matches(lesson_name, wanted):
    key = gpa.course_key(lesson_name)
    return any(gpa.course_key(ALIASES.get(w.strip(), w)) == key for w in wanted)


def _day_name(text):
    t = gpa.course_key(text)
    return next((d for d in DAYS if gpa.course_key(d) == t), None)


def _order(lesson):
    return DAYS.index(lesson["gun"]), lesson["saatler"][0], lesson["sinif"]


def find_lessons(data, courses=None, room=None, section=None, grade=None, day=None):
    """Bütün haftanın ders blokları; filtreler: ders adları, derslik (numara, kısaltma ya da adı), şube, sınıf, gün adı."""
    rows = data["dersler"]
    if courses:
        rows = [b for b in rows if _course_matches(b["ders"], courses)]
    if room:
        want = _words(room)
        rows = [b for b in rows if b["derslik"] and (gpa.course_key(b["derslik"]) == gpa.course_key(room)
                                                     or want <= _words(ROOMS.get(b["derslik"], b["derslik"])))]
    if section:
        rows = [b for b in rows if (b["sube"] or "") == section.strip().upper()]
    if grade:
        rows = [b for b in rows if b["sinif"] == grade]
    if day:
        rows = [b for b in rows if b["gun"] == _day_name(day)]
    return sorted(rows, key=_order)


def _block_line(b, data):
    time = f"{b['gun']} {b['saatler'][0][:5]}-{b['saatler'][-1][-5:]}"
    room = data["derslikler"].get(b["derslik"], b["derslik"]) or "derslik belirtilmemiş"
    extra = ", ".join(x for x in (f"{b['sube']} şubesi" if b["sube"] else "", b["yuva"] or "") if x)
    return f"- {time} {b['ders']}" + (f" ({extra})" if extra else "") + f", {b['sinif']}. sınıf, {room}"


def week_text(rows, data):
    if not rows:
        return f"Bu filtreyle {data['donem']} ders programında ders yok."
    return "\n".join([f"{data['donem']} ders programından:"] + [_block_line(b, data) for b in rows]
                     + [f"Kaynak: {data['kaynak']}"])


def _slots(lessons):
    return [(b["gun"], s) for b in lessons for s in b["saatler"]]


def plan_sections(data, codes, choices=None, free_days=None):
    """Öğrencinin dersleri (plan kodları) + istenen dersler/şubeler için şube kombinasyonları.

    choices: [{"ders", "sube"}]; şube verilirse o derste yalnız o şube denenir, ders listede yoksa eklenir.
    free_days: boş kalması istenen gün adları. Bir ders saati iki derste birden olmamalı.
    Dönüş: {"dersler": {ders: [şube]}, "sabit": {ders: şube}, "cakismalar": sabit şubeyle çakışan diğer ders blokları,
    "uygun_sayisi", "ornekler": [{"secim", "dersler", "ders_saati"}], "programda_yok", "programda_olmayan_kodlar"}."""
    by_course = {}
    for b in data["dersler"]:
        if b["kod"] in codes:
            by_course.setdefault(b["ders"], []).append(b)
    missing_codes = sorted(set(codes) - {b["kod"] for b in data["dersler"]})
    fixed, missing = {}, []
    for choice in choices or []:
        lessons = [b for b in data["dersler"] if _course_matches(b["ders"], [choice["ders"]])]
        section = (choice.get("sube") or "").strip().upper()
        if not lessons or (section and not any((b["sube"] or "") == section for b in lessons)):
            missing.append(choice["ders"] + (f" ({section} şubesi)" if section and lessons else ""))
            continue
        name = lessons[0]["ders"]
        by_course[name] = lessons
        if section:
            fixed[name] = section
    options = {}
    for name, lessons in sorted(by_course.items()):
        sections = sorted({b["sube"] or "" for b in lessons})
        if name in fixed:
            sections = [fixed[name]]
        options[name] = {s: [b for b in lessons if (b["sube"] or "") == s] for s in sections}

    clashes = []
    for name, section in fixed.items():
        taken = set(_slots(options[name][section]))
        for other, by_section in options.items():
            if other == name:
                continue
            hits = [(s, b) for s, bs in by_section.items() for b in bs if taken & set(_slots([b]))]
            every = all(taken & set(_slots(bs)) for bs in by_section.values())
            for s, b in hits:
                clashes.append({"ders": other, "sube": s, "gun": b["gun"], "saatler": b["saatler"],
                                "kesin": every, "sabit_ders": name, "sabit_sube": section})

    free = {_day_name(d) for d in free_days or []} - {None}
    names = list(options)
    valid, examples, usable = 0, [], {n: set() for n in names}
    for combo in itertools.product(*(options[n].items() for n in names)):
        lessons = [b for _, bs in combo for b in bs]
        slots = _slots(lessons)
        if len(slots) != len(set(slots)) or any(b["gun"] in free for b in lessons):
            continue
        valid += 1
        for n, (s, _) in zip(names, combo):
            usable[n].add(s)
        if len(examples) < MAX_EXAMPLES:
            examples.append({"secim": {n: s for n, (s, _) in zip(names, combo)},
                             "dersler": sorted(lessons, key=_order), "ders_saati": len(slots)})
    return {"dersler": {n: list(options[n]) for n in names}, "sabit": fixed, "cakismalar": clashes,
            "bos_gunler": sorted(free, key=DAYS.index), "uygun_sayisi": valid, "ornekler": examples,
            "uygun_subeler": {n: sorted(usable[n]) for n in names},
            "programda_yok": missing, "programda_olmayan_kodlar": missing_codes}


def plan_text(plan, data):
    """plan_course_sections tool'unun metni: çakışmalar, uygun kombinasyon sayısı, örnekler ve haftalık ders saati."""
    lines = [f"ŞUBE PLANI ({data['donem']}, deterministik)"]
    if plan["programda_yok"]:
        lines.append(f"Ders programında bulunamayan: {', '.join(plan['programda_yok'])}.")
    courses = [f"{n} ({', '.join(s + ' şubesi' if s else 'tek şube' for s in ss)})" for n, ss in plan["dersler"].items()]
    lines.append("Dersler: " + ("; ".join(courses) if courses else "yok") + ".")
    if plan["bos_gunler"]:
        lines.append(f"Boş kalması istenen gün: {', '.join(plan['bos_gunler'])}.")
    for name, section in plan["sabit"].items():
        mine = [c for c in plan["cakismalar"] if c["sabit_ders"] == name]
        if not mine:
            lines.append(f"{name} {section} şubesi diğer derslerin hiçbir şubesiyle çakışmıyor.")
            continue
        lines.append(f"{name} {section} şubesiyle çakışanlar:")
        for c in mine:
            sure = "bu dersin her şubesi çakışıyor" if c["kesin"] else "başka şubesi seçilebilir"
            label = f"{c['sube']} şubesi" if c["sube"] else "tek şube"
            lines.append(f"- {c['ders']} ({label}): {c['gun']} {c['saatler'][0][:5]}-{c['saatler'][-1][-5:]} ({sure})")
    if plan["uygun_sayisi"] == 0:
        lines.append("Bu koşullarla çakışmasız bir şube kombinasyonu yok.")
    else:
        lines.append("Uygun kombinasyonlarda ders başına seçilebilecek şubeler:")
        for name, sections in plan["uygun_subeler"].items():
            if len(plan["dersler"][name]) > 1:
                which = (f"yalnız {sections[0]} şubesi" if len(sections) == 1
                         else " ya da ".join(s + " şubesi" for s in sections))
                lines.append(f"- {name}: {which}")
        lines.append(f"Çakışmasız uygun kombinasyon sayısı: {plan['uygun_sayisi']}"
                     + (f" (ilk {len(plan['ornekler'])} tanesi aşağıda)" if plan["uygun_sayisi"] > len(plan["ornekler"])
                        else "") + ".")
        for i, combo in enumerate(plan["ornekler"], 1):
            picks = ", ".join(f"{n} {s} şubesi" if s else n for n, s in combo["secim"].items())
            lines.append(f"{i}. {picks}; haftada {combo['ders_saati']} ders saati")
            lines += ["   " + _block_line(b, data) for b in combo["dersler"]]
    lines.append("Ders saati: programdaki her 45 dakikalık ders bir ders saati. Kaynak: " + data["kaynak"])
    return "\n".join(lines)


if __name__ == "__main__":
    written = write_schedule()
    print(f"{len(written['dersler'])} ders bloğu yazıldı: {SCHEDULE_PATH}")
