---
status: accepted
---

# Agent döngüsü elle yazılır, LangGraph kullanılmaz

Agent döngüsü (model → tool → model, tur sınırı, hata yakalama, kaynak toplama) framework'süz, elle yazılır: main'de
`agent.py` (langchain `bind_tools` + Gemini), `exp/haiku-agent` branch'inde `codeact.py` (Anthropic SDK + Haiku).
LangGraph, model, tool'lar ve prompt aynı kaldıkça cevapları değiştirmez; getirdiği yetenekler (sunucuda durum,
kesintiye uğrayıp devam eden akış, çoklu agent) bu mimarinin bilinçli olarak kullanmadığı şeylerdir. (2026-10-26)

## Considered Options

- **LangGraph'a geçmek (ayrı branch'te).** Reddedildi: ölçülebilir bir iyileşme beklenmeyen bir refactor; deney değil,
  yine de karşılaştırma için tam koşu kotası ister. Özelliklerinin bizdeki karşılığı:
  - Checkpointer (sohbet durumu sunucuda): API bilinçli olarak stateless; geçmiş ve transkript istemcide, kişisel veri
    sunucuda kalmaz.
  - Interrupt (kullanıcı onayı için durma): `confirm_equivalence` bunu stateless yapıyor; agent sorar, kullanıcı sonraki
    mesajda cevaplar.
  - LangSmith tracing: `logs/agent.jsonl` ve `logs/anlik_tool.jsonl` yeterli; veri dış servise gitmez.
  - Prebuilt ReAct agent: kendi döngümüz birkaç düzine satır.
- **Hazır framework döngüsü yerine elle yazılı döngü (seçilen).** Haiku branch'i Anthropic'e özgü özellikler kullanıyor
  (sunucu tarafı `code_execution`, container'a dosya yükleme, `pause_turn` devamı, prompt caching); bunlar SDK'yla
  doğrudan yazılı, ara katmanın desteğine bağlı değil. Tezde döngünün her adımı kendi kodumuzda gösterilebilir.

## Ne zaman yeniden açılır

- Akış gerçekten dallanırsa: sabit, çok adımlı ve arada kullanıcının beklediği bir iş akışı (ör. transkript yükle →
  doğrula → öğrenciye onaylat → analiz → danışmana rapor).
- Birden fazla agent iş bölümü yaparsa.
- Sohbet durumunun sunucuda tutulması gerekirse (bu da kişisel veri kararını yeniden açar).
