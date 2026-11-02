// Agent araçlarının insan okunur adları: TEK kaynak. Hem /araclar sayfası hem sohbetteki "Kullanılan araçlar"
// bölümü buradan okur; kod adları (search_regulations vb.) kullanıcıya gösterilmez.
// `names`: agent.py'deki gerçek tool adları (API'nin tool_calls[].name değeri). Bir kart birden çok tool'u kapsayabilir.
// `rozet`: aracın nasıl çalıştığı. `transkript`: transkript yüklü olmadan çalışmaz.
// "code_execution" agent.py'nin anlık kod çalıştırdığında tool_calls'a eklediği sanal addır.

const TOOL_GROUPS = ["Mevzuat", "Transkript ve hesaplar", "Ders programı", "Güncel bilgi", "Gelişmiş"];

const TOOL_KINDS = {
  search: { label: "Belgelerde arar", done: "Belgelerde arandı" },
  code: { label: "Kodla hesaplanır", done: "Kodla hesaplandı" },
  db: { label: "Veri tabanı", done: "Veri tabanından alındı" },
  live: { label: "Anlık kod", done: "Anlık kod çalıştırıldı" },
};

const TOOL_CARDS = [
  { names: ["search_regulations"], grup: "Mevzuat", ad: "Mevzuat arama", rozet: "search",
    aciklama: "Yönetmelik ve yönergelerde ilgili maddeleri bulur; cevabı madde numarasıyla verir.",
    ornek: "Staj için kaç iş günü gerekiyor?" },
  { names: ["list_articles", "get_article"], grup: "Mevzuat", ad: "Madde okuma", rozet: "search",
    aciklama: "Bir belgenin maddelerini listeler ve istenen maddeyi tam metniyle açar.",
    ornek: "Yatay Geçiş Yönergesi’nin başvuru şartları maddesini göster." },
  { names: ["get_graduation_requirements"], grup: "Mevzuat", ad: "Mezuniyet şartları", rozet: "code",
    aciklama: "Öğretim planı ve yönetmelikten gelen mezuniyet şartlarını kaynaklarıyla özetler.",
    ornek: "Mezuniyet şartları neler?" },

  { names: ["analyze_transcript"], grup: "Transkript ve hesaplar", ad: "Mezuniyet durumu analizi", rozet: "code", transkript: true,
    aciklama: "Yüklediğiniz transkriptten tamamlanan AKTS’yi, eksik zorunlu dersleri, seçmeli açığını ve not ortalamasını çıkarır.",
    ornek: "Mezun olmak için neyim eksik?" },
  { names: ["confirm_equivalence"], grup: "Transkript ve hesaplar", ad: "Ders eşdeğerliği teyidi", rozet: "code", transkript: true,
    aciklama: "Planla eşleşmeyen dersler (Erasmus, intibak) için hangi derse sayıldığını sorar ve hesaba katar.",
    ornek: "Bu Erasmus dersi hangi derse sayılır?" },
  { names: ["simulate_courses"], grup: "Transkript ve hesaplar", ad: "Ders alırsam ne olur?", rozet: "code", transkript: true,
    aciklama: "Seçtiğiniz derslerle AKTS’nin tamamlanıp tamamlanmayacağını gösterir; kararı danışmana bırakır.",
    ornek: "Şu iki seçmeliyi alırsam AKTS’m tamamlanır mı?" },
  { names: ["simulate_grades"], grup: "Transkript ve hesaplar", ad: "Not senaryosu", rozet: "code",
    aciklama: "Vize ve final puanlarından harf notunu, hedef not ortalaması için gereken en düşük notu hesaplar.",
    ornek: "Ortalamam 2,50 olsun diye finalden kaç almalıyım?" },
  { names: ["get_transcript_average"], grup: "Transkript ve hesaplar", ad: "Transkript ortalaması", rozet: "code", transkript: true,
    aciklama: "Transkriptin bir kısmının (belirli dersler, ilk yıllar, Erasmus hariç) not ortalamasını yönetmelikteki tabloyla hesaplar.",
    ornek: "Birinci sınıfın sonundaki ortalamam neydi?" },

  { names: ["get_course_schedule"], grup: "Ders programı", ad: "Günün programı", rozet: "db",
    aciklama: "Bir günün derslerini saat, şube ve derslikle verir; tatil ve hafta sonunu takvime göre söyler.",
    ornek: "Bugün hangi dersler var?" },
  { names: ["search_course_schedule"], grup: "Ders programı", ad: "Ders ve derslik arama", rozet: "db",
    aciklama: "Bir dersin, dersliğin veya şubenin hafta içinde ne zaman ve nerede olduğunu bulur.",
    ornek: "Yapay Zeka dersi hangi gün ve derslikte?" },
  { names: ["plan_course_sections"], grup: "Ders programı", ad: "Şube planı", rozet: "code",
    aciklama: "Derslerin şubelerinden çakışmasız kombinasyonlar çıkarır; boş kalmasını istediğiniz günleri ve haftalık ders saatini hesaba katar.",
    ornek: "Cuma günü dersim olmayacak şekilde şubeleri nasıl seçebilirim?" },
  { names: ["get_offered_electives"], grup: "Ders programı", ad: "Bu dönem açılan seçmeliler", rozet: "db",
    aciklama: "Seçmeli yuvalarına göre bu dönem gerçekten açılan dersleri listeler.",
    ornek: "Bu dönem hangi seçmeli dersler açılıyor?" },

  { names: ["get_announcements"], grup: "Güncel bilgi", ad: "Duyurular", rozet: "db",
    aciklama: "Bölüm, fakülte ve üniversite duyurularında konu ve tarihe göre arar.",
    ornek: "Staj ile ilgili son duyurular neler?" },
  { names: ["get_academic_staff"], grup: "Güncel bilgi", ad: "Akademik personel", rozet: "db",
    aciklama: "Öğretim elemanlarını ad, unvan veya görev (ör. bölüm başkanı) ile bulur.",
    ornek: "Bölüm başkanı kim?" },

  { names: ["request_new_tool", "code_execution"], grup: "Gelişmiş", ad: "Anlık hesap", rozet: "live",
    aciklama: "Hazır aracı olmayan bir hesap için o anda kod yazıp çalıştırır. Sonuç doğrulanmamıştır; cevapta uyarı gösterilir.",
    ornek: "Bu dönemin haftalık toplam ders saatini hesapla." },
];

// Gerçek tool adından kartı bulur; tanımadığı ad için null (çağıran "Diğer araç" der, kod adı gösterilmez).
function toolCard(name) {
  return TOOL_CARDS.find((c) => c.names.includes(name)) || null;
}

// Görünen ad; tanımadığı tool için kod adı değil "Diğer araç".
function toolLabel(name) {
  const card = toolCard(name);
  return card ? card.ad : "Diğer araç";
}

// Cevabın altında uyarı gösterilen anlık araçlar: kodu o an bir LLM yazdı, test edilmedi.
function isLiveTool(name) {
  const card = toolCard(name);
  return !!card && card.rozet === "live";
}
