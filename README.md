# agentic-rag

Bolu Abant İzzet Baysal Üniversitesi Mühendislik Fakültesi öğrencileri için agentic RAG chatbot.
[campus-chatbot](https://github.com/ariftahab/campus-chatbot) mimarisinden (commit `0004309`, production EXP-038)
türetildi; belge kümesi Mühendislik Fakültesi öğrencisinin danışmanlığında gereken belgelerle sınırlı (24 belge).
RAG bileşenlerindeki her değişiklik tekrar üretilebilir bir deney olarak kaydedilir
(bkz. `experiments/experiment_log.md`, `python -m experiments.report` üretir).

## Kurulum

```bash
python -m venv venv
venv\Scripts\activate          # Windows
pip install -r requirements.txt
```

Proje köküne `.env` dosyası oluşturup Gemini anahtarını ekleyin:

```
GOOGLE_API_KEY=...
```

## Kullanım

```bash
python chat.py                          # sohbet ('çık' ile çıkılır, 'yeni' geçmişi temizler)
python ingest.py [EXP-ID]               # bir yapılandırmanın indeksini elle kur (normalde otomatik)
python -m experiments.run EXP-002       # bir deneyi koş (retrieval metrikleri, API çağrısı yok)
python -m experiments.run --all         # sonucu olmayan tüm deneyleri koş
python -m experiments.run EXP-002 --llm # cevap doğruluğunu da ölç (Gemini kotası harcar)
python -m experiments.report            # deney günlüğünü yeniden üret
python experiments/teshis_alt_kume.py   # bir değişikliği yalnız seçilen sorularda dene (deney değil)
```

Sohbet uygulaması `config.PRODUCTION_EXPERIMENT` ile seçilen deneyin parametreleriyle çalışır.
İlk çalıştırmada embedding ve reranker modelleri (~6.5 GB) indirilir.

## Yeni deney ekleme

1. `experiments/configs/` altındaki bir deneyi kopyalayıp yeni kimlik verin (`EXP-002.json`).
2. `ust_deney`'i karşılaştırılacak deney yapın ve `params` içinde **yalnız bir** parametreyi değiştirin.
3. `neden`, `hipotez`, `birincil_metrik` ve `beklenen_yon` alanlarını deneyi koşmadan önce doldurun.
4. Kodu commit edin, sonra `python -m experiments.run EXP-002` çalıştırın.
5. Sonucu yorumlayıp `notlar` alanına yazın, `python -m experiments.report` ile günlüğü güncelleyin ve commit edin.

## Yeni belge ekleme

PDF'i `data/` klasörüne koyup `config.py` içindeki `DOCUMENTS`'a başlığını ve türünü yazın.
Belgeler değişince indeksler kendiliğinden yeniden kurulur. Ancak eski deney sonuçları eski belgelerle
ölçüldüğü için yeni belgelerle yapılan ölçümlerle karşılaştırılamaz.
