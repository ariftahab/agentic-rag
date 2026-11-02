"""API ve arayüzü tek sunucuda birleştirir.

Çalıştırma:  venv/Scripts/python.exe -m uvicorn app.main:app --port 8000
  http://localhost:8000/          sohbet (app/static/index.html; /belgeler ve /araclar aynı doküman, farklı açılış görünümü)
  http://localhost:8000/belgeler  corpus'taki belgeler
  http://localhost:8000/araclar   agent'ın kullandığı araçlar
  http://localhost:8000/admin     değerlendirme (elle etiketleme, deney sonuçları)
  http://localhost:8000/docs      API dokümantasyonu (Swagger)

Sayfalar statik dosyalardır (HTML/CSS/JS); veriyi tarayıcıdan API'ye fetch ile istek atarak alırlar.
Sunucu sayfa üretmez; yalnız ortak kabuğu (kenar çubuğu + üst çubuk, app/kabuk.html) sayfa dosyalarına yerleştirir,
dosyaları ve API'yi servis eder.
"""
import hashlib
import re
from pathlib import Path

from fastapi import Depends
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from app.api import app, require_admin

STATIC = Path(__file__).parent / "static"

app.mount("/static", StaticFiles(directory=STATIC), name="static")


# Sayfalar ve statik dosyalar için "no-cache": tarayıcı dosyayı saklar ama her kullanımda sunucuya değişip
# değişmediğini sorar (ETag / Last-Modified; değişmediyse 304 döner, dosya tekrar inmez). Bu header olmadan
# tarayıcı kendi tahminiyle eski CSS/JS'i bir süre sorgusuz kullanır ve arayüz güncellemesi telefona geç ulaşır.
@app.middleware("http")
async def no_cache_ui(request, call_next):
    response = await call_next(request)
    if request.url.path.startswith("/static/") or request.url.path in ("/", "/belgeler", "/araclar", "/admin"):
        response.headers["Cache-Control"] = "no-cache"
    return response


# Cache busting: HTML'deki /static/ adreslerine dosya içeriğinin hash'i eklenir (style.css?v=3f2a91c0).
# Dosya değişince adres değişir; tarayıcı yeni adresi cache'inde bulamaz ve dosyayı mutlaka indirir.
# Gerekçe: telefon tarayıcısı CSS/JS'i hiç sormadan eski cache'ten kullanıyordu (sunucu log'unda istek yoktu);
# no-cache tek başına yetmedi. HTML her istekte okunur, böylece sunucu yeniden başlatılmadan da güncel kalır.
_ASSET = re.compile(r"""(["'])/static/([\w.-]+)(?=["'])""")


def _versioned(match):
    path = STATIC / match[2]
    if not path.is_file():
        return match[0]
    digest = hashlib.sha256(path.read_bytes()).hexdigest()[:8]
    return f"{match[1]}/static/{match[2]}?v={digest}"


# Ortak kabuk: her sayfada aynı kenar çubuğu. Sayfa dosyası içeriğini <!--KABUK--> ve <!--/KABUK--> arasına yazar;
# kabuk.html, <!--ICERIK--> işaretinden bölünüp bu iki işaretin yerine konur. Etkin gezinme bağlantısı sunucuda
# işaretlenir (JS kapalıyken de doğru görünür).
SHELL = Path(__file__).parent / "kabuk.html"


def _with_shell(html, active, title):
    head, tail = SHELL.read_text(encoding="utf-8").split("<!--ICERIK-->")
    head = head.replace("{{BASLIK}}", title).replace("{{GORUNUM}}", active)
    head = head.replace(f'data-nav="{active}"', f'data-nav="{active}" class="active" aria-current="page"')
    return html.replace("<!--KABUK-->", head).replace("<!--/KABUK-->", tail)


def _page(name, active, title):
    def handler():
        html = _with_shell((STATIC / name).read_text(encoding="utf-8"), active, title)
        return HTMLResponse(_ASSET.sub(_versioned, html))
    return handler


# Sohbet, belgeler ve araçlar TEK dokümandır (index.html'de üç görünüm): gezinme sayfa yüklemeden görünüm değiştirir,
# böylece yalnız bellekte duran transkript kaybolmaz. Yol yalnız açılışta hangi görünümün etkin olacağını belirler.
for route, active, title in [("/", "sohbet", "Yeni sohbet"), ("/belgeler", "belgeler", "Belgeler"),
                             ("/araclar", "araclar", "Araçlar")]:
    app.add_api_route(route, _page("index.html", active, title), include_in_schema=False)
# Panel sayfası da korunur: tarayıcı şifreyi burada bir kez sorar, sonra /admin/* fetch isteklerine kendisi ekler.
app.add_api_route("/admin", _page("admin.html", "admin", "Değerlendirme"), include_in_schema=False, dependencies=[Depends(require_admin)])
