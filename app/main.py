"""API ve arayüzü tek sunucuda birleştirir.

Çalıştırma:  venv/Scripts/python.exe -m uvicorn app.main:app --port 8000
  http://localhost:8000/          sohbet (app/static/index.html)
  http://localhost:8000/belgeler  corpus'taki belgeler
  http://localhost:8000/admin     elle değerlendirme
  http://localhost:8000/docs      API dokümantasyonu (Swagger)

Sayfalar statik dosyalardır (HTML/CSS/JS); veriyi tarayıcıdan API'ye fetch ile istek atarak alırlar.
Sunucu sayfa üretmez, yalnız dosyaları ve API'yi servis eder.
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
    if request.url.path.startswith("/static/") or request.url.path in ("/", "/belgeler", "/admin"):
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


def _page(name):
    def handler():
        return HTMLResponse(_ASSET.sub(_versioned, (STATIC / name).read_text(encoding="utf-8")))
    return handler


for route, page in [("/", "index.html"), ("/belgeler", "belgeler.html")]:
    app.add_api_route(route, _page(page), include_in_schema=False)
# Panel sayfası da korunur: tarayıcı şifreyi burada bir kez sorar, sonra /admin/* fetch isteklerine kendisi ekler.
app.add_api_route("/admin", _page("admin.html"), include_in_schema=False, dependencies=[Depends(require_admin)])
