"""Mevzuatta madde düzeyinde gezinme (agent_h8, LLM'siz): belgelerin madde başlıkları (içindekiler), bir maddenin tam
metni ve bir parçadaki madde atıfları. Kaynak indeksin chunk'larıdır (structural chunking'in `madde` ve `heading`
metadata'sı); ayrı bir dosya ya da index gerekmez.

EXP-016/017'de (agent_h6) hataların çoğu, cevabın dayandığı ikinci maddenin aramada sıralamaya girmemesiydi: bulunan
madde başka bir maddeye atıf yapıyordu ya da agent doğru belgeyi bildiği hâlde ilgili madde getirilmedi.
"""
import json
import re
import unicodedata

import config

# agent_h9: belgelerin düzeyi ve tarihleri (elle çıkarıldı; bkz. dosyanın 'aciklama' alanı).
DOCUMENT_INFO_PATH = config.DATA_DIR / "mevzuat_belgeleri.json"
LEVELS = "1 kanun, 2 YÖK düzenlemesi, 3 üniversite yönetmeliği, 4 Senato yönergesi"

# Başka belgelere yapılan atıflarda belgenin metindeki adı → corpus dosyası. Yalnız korpusta olan belgeler; kanunlara ve
# korpus dışı yönetmeliklere yapılan atıflar izlenmez.
REFERENCED_DOCUMENTS = {
    "önlisans, lisans eğitim-öğretim yönetmeliği": "önlisans lisans eğitim öğretim.pdf",
    "önlisans, lisans eğitim öğretim yönetmeliği": "önlisans lisans eğitim öğretim.pdf",
    "ön lisans, lisans eğitim öğretim yönetmeliği": "önlisans lisans eğitim öğretim.pdf",
    "yan dal programı yönergesi": "yandal programı.pdf",
    "çift ana dal programı yönergesi": "çift anadal program yönergesi.pdf",
    "kredi transferi yapılması esaslarına ilişkin yönetmeliğ": "kurumlar arası kredi transferi.pdf",
    "uzaktan öğretime ilişkin usul ve esasları": "uzaktan-ogretime-iliskin-usul-esas.pdf",
}
_ORDINAL = r"(?:\s*['’]?\s*(?:inci|ıncı|nci|ncı|üncü|uncu|ncü|ncu|ünci|unci|ıncı|ının|inin))?"
_NUMBERS = rf"(\d+){_ORDINAL}((?:\s*(?:,|ve|ile|-|–)\s*\d+{_ORDINAL})*)"
_REFERENCE = re.compile(rf"\b{_NUMBERS}\s*madde(?:s|ler|de|y|ye)", re.IGNORECASE)
# Atıftan hemen önce bunlardan biri geçiyorsa atıf korpus dışı bir metne (kanun, başka yönetmelik) gider.
_EXTERNAL = re.compile(r"(sayılı|kanun\w*|yönetmeliğ\w*|yönerge\w*|esas\w*|karar\w*)(?:['’]\w+)?\s*(?:\([^)]*\)\s*)?$",
                       re.IGNORECASE)
# "bu Yönetmeliğin 11 inci maddesi": belgenin kendisine atıf.
_SELF = re.compile(r"\bbu\s+(yönetmeliğin|yönergenin|usul ve esasların)\s*(?:\([^)]*\)\s*)?$", re.IGNORECASE)


def load_document_info(path=DOCUMENT_INFO_PATH):
    """Kaynak dosya → {"duzey", "duzey_sira", "kabul"?, "son_degisiklik"?, "tarih_notu"?}."""
    return json.loads(path.read_text(encoding="utf-8"))["belgeler"]


def document_line(info, source):
    """agent_h9: parçanın kaynak etiketinin altına belgenin düzeyi ve tarihleri; belge tabloda yoksa (takvim) boş."""
    d = info.get(source)
    if d is None:
        return ""
    parts = [f"düzey {d['duzey_sira']}/4 ({LEVELS}): {d['duzey']}"]
    if d.get("kabul"):
        parts.append(f"kabul {d['kabul']}")
    if d.get("son_degisiklik"):
        parts.append(f"son değişiklik {d['son_degisiklik']}")
    if d.get("tarih_notu"):
        parts.append(d["tarih_notu"])
    return f"(Belge: {'; '.join(parts)})\n"


def normalize(text):
    """Belge adlarını karşılaştırmak için: küçük harf, Türkçe karakterler ve noktalama sadeleştirilmiş."""
    text = text.replace("İ", "i").replace("I", "ı").lower()
    text = unicodedata.normalize("NFKD", text)
    text = "".join(c for c in text if not unicodedata.combining(c)).replace("ı", "i")
    return re.sub(r"[^a-z0-9]+", " ", text).strip()


class ArticleIndex:
    """Chunk'lardan madde dizini: (kaynak dosya, madde no) → o maddenin chunk'ları (sırasıyla)."""

    def __init__(self, chunks):
        self.articles, self.titles = {}, {}
        for c in chunks:
            m = c.metadata
            self.titles.setdefault(m["source"], m["title"])
            if m.get("madde") is not None:
                self.articles.setdefault((m["source"], m["madde"]), []).append(c)
        for parts in self.articles.values():
            parts.sort(key=lambda c: c.metadata.get("chunk_id", 0))

    def documents(self):
        """Madde yapısı olan belgelerin başlıkları (agent'ın arama sonuçlarında gördüğü adlar), alfabetik."""
        return sorted({self.titles[s] for s, _ in self.articles})

    def find_document(self, name):
        """Belge adı (arama sonucundaki başlık, dosya adı ya da başlığın bir kısmı) → kaynak dosya; bulunamazsa None."""
        wanted = normalize(name)
        if not wanted:
            return None
        sources = {s for s, _ in self.articles}
        exact = [s for s in sources if wanted in (normalize(self.titles[s]), normalize(s.rsplit(".", 1)[0]))]
        if exact:
            return exact[0]
        partial = [s for s in sources if wanted in normalize(self.titles[s]) or normalize(self.titles[s]) in wanted]
        return min(partial, key=lambda s: len(self.titles[s])) if partial else None

    def table_of_contents(self, source):
        """[(madde no, başlık, bölüm)] madde sırasıyla."""
        rows = []
        for (s, madde), parts in self.articles.items():
            if s == source:
                m = parts[0].metadata
                rows.append((madde, m.get("heading") or "", m.get("section") or ""))
        return sorted(rows)

    def article_text(self, source, madde):
        """Maddenin bütün chunk'ları birleştirilmiş; madde yoksa None. Dönüş: (metin, chunk listesi)."""
        parts = self.articles.get((source, madde))
        if not parts:
            return None, []
        return "\n".join(c.page_content for c in parts), parts

    def references(self, chunk):
        """Parçanın korpustaki başka maddelere yaptığı atıflar: [(kaynak dosya, madde no)], parçanın kendi maddesi hariç.
        Atıf öncesinde bir korpus belgesinin adı geçiyorsa o belgeye, kanun ya da korpus dışı bir metin adı geçiyorsa
        hiçbir yere (izlenmez), ad geçmiyorsa parçanın kendi belgesine gider."""
        text, own = chunk.page_content, chunk.metadata["source"]
        found = []
        for match in _REFERENCE.finditer(text):
            before = text[max(0, match.start() - 160):match.start()].split(". ")[-1].rstrip()  # aynı cümle
            target = own
            lowered = before.replace("İ", "i").lower()
            named = [(lowered.rfind(alias), source) for alias, source in REFERENCED_DOCUMENTS.items() if alias in lowered]
            if named:
                target = max(named)[1]
            elif _EXTERNAL.search(before) and not _SELF.search(before):
                continue
            numbers = [int(match.group(1))] + [int(n) for n in re.findall(r"\d+", match.group(2) or "")]
            for n in numbers:
                ref = (target, n)
                if ref != (own, chunk.metadata.get("madde")) and ref in self.articles and ref not in found:
                    found.append(ref)
        return found

    def label(self, ref):
        source, madde = ref
        return f"{self.titles[source]}, Madde {madde}"
