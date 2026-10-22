"""Haftalık ders programı: PDF'ten bir kez yapılandırılmış veriye çevrilir, agent'ın get_course_schedule tool'u
buradan okur. RAG'a girmez; LLM kullanılmaz.

Komut satırı (veriyi yeniden üretmek için):
    python schedule.py
"""
import json
import re
from datetime import date, timedelta

import pdfplumber

import config
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


if __name__ == "__main__":
    written = write_schedule()
    print(f"{len(written['dersler'])} ders bloğu yazıldı: {SCHEDULE_PATH}")
