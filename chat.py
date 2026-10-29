"""Terminalde sohbet arayüzü.

Kullanım:
    python chat.py                          # klasik RAG (RagBot, production yapılandırması)
    python chat.py --agent                  # agent (tool calling), transkriptsiz
    python chat.py --transkript dosya.pdf   # agent, UBYS tarihsel transkriptiyle
    python chat.py --agent --prompt agent_v7  # agent, belirli bir prompt sürümüyle (v7: anlık tool)
"""
import argparse
import os
import sys

import config
from rag import RagBot, source_label, tr_lower

EXIT_COMMANDS = {"çık", "çik", "cık", "cik", "exit", "quit", "q"}
RESET_COMMANDS = {"yeni", "/yeni", "sıfırla"}


def print_tool_call(name, args):
    shown = ", ".join(f"{k}={v!r}" for k, v in args.items())
    print(f"  [tool] {name}({shown})")


def main():
    parser = argparse.ArgumentParser(description="BAİBÜ Mühendislik Fakültesi danışmanlık asistanı")
    parser.add_argument("--agent", action="store_true", help="tool calling yapan agent ile başlat")
    parser.add_argument("--transkript", metavar="PDF", help="UBYS tarihsel transkripti (agent ile başlatır)")
    parser.add_argument("--prompt", metavar="SÜRÜM", help="agent prompt sürümü (ör. agent_v7; agent ile başlatır)")
    args = parser.parse_args()
    use_agent = args.agent or args.transkript is not None or args.prompt is not None

    # Anahtar yoksa modelleri yüklemeye hiç başlama.
    if not os.getenv("GOOGLE_API_KEY"):
        print("Hata: GOOGLE_API_KEY .env dosyasında bulunamadı.")
        return 1

    transcript = None  # parse edilmiş, kişisel bilgilerden arındırılmış sözlük; yalnız bu süreç boyunca bellekte
    if args.transkript:
        from tools import transcript as transcript_parser
        try:
            transcript = transcript_parser.parse_transcript(args.transkript)
        except transcript_parser.TranscriptFormatError as e:
            print(f"Hata: transkript okunamadı. {e}")
            return 1
        except OSError as e:
            print(f"Hata: dosya açılamadı ({e}).")
            return 1
        print(f"Transkript okundu: {len(transcript['dersler'])} ders.")

    print(f"Asistan hazırlanıyor (yapılandırma: {config.PRODUCTION_EXPERIMENT}, "
          f"{'agent' if use_agent else 'RAG'}, modeller yükleniyor)...")
    if use_agent:
        from agent import AGENT_PROMPT_VERSION, AdvisorAgent
        bot = AdvisorAgent.from_config(config.production_config(), prompt_version=args.prompt or AGENT_PROMPT_VERSION)
    else:
        bot = RagBot.from_config(config.production_config())
    print("Hazır. Çıkmak için 'çık', yeni konuya geçmek için 'yeni' yazın.")

    history = []  # bot stateless; sohbet geçmişini burada tutuyoruz
    confirmations = {}  # plan dışı derslerin teyit edilen eşdeğerleri; transkriptle birlikte yaşar, "yeni" silmez
    while True:
        try:
            question = input("\nSoru: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not question:
            continue
        command = tr_lower(question)
        if command in EXIT_COMMANDS:
            break
        if command in RESET_COMMANDS:
            history = []
            print("Sohbet geçmişi temizlendi.")
            continue

        print("Cevap düşünülüyor...")
        try:
            if use_agent:
                result = bot.ask(question, history, transcript=transcript, on_tool_call=print_tool_call,
                                 confirmations=confirmations)
            else:
                result = bot.ask(question, history)
        except Exception as e:  # kota, ağ, API hataları döngüyü öldürmesin
            print(f"Hata: cevap alınamadı ({type(e).__name__}: {e})")
            continue
        history += [{"role": "user", "content": question}, {"role": "assistant", "content": result["answer"]}]

        print(f"\n{result['answer']}")
        if use_agent:
            for item in result.get("generated_code", []):  # anlık tool'un hamleleri (doğrulanmamış)
                print(f"\n[anlık tool] {item['gorev']}")
                if "hata" in item:
                    print(f"  Hata: {item['hata']}")
                    continue
                print(f"  {item['istek_sayisi']} istek, {item['sure_ms'] / 1000:.1f} sn, {item['tokens']}")
                for step in item["adimlar"]:
                    text = step["kod"] if step["tip"] == "kod" else step["metin"]
                    print(f"  [{step['tip']}] " + text.replace("\n", "\n    "))
                    if step["tip"] == "kod":
                        print("    → " + str(step["cikti"]).replace("\n", "\n      "))
            continue
        if result["query"] != question:
            print(f"\n(Aranan soru: {result['query']})")
        print("\nKaynaklar:")
        for doc in result["docs"]:
            print(f"  - {source_label(doc)}")

    print("Görüşmek üzere!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
