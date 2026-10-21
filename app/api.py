"""API katmanı: FastAPI endpoint'leri.

Çalıştırma:  venv/Scripts/python.exe -m uvicorn app.api:app --port 8000
Dokümantasyon (Swagger UI):  http://localhost:8000/docs

Bu katman yalnız HTTP ile ilgilenir: isteği doğrular (Pydantic), service'i çağırır, sonucu JSON'a ve
hataları uygun HTTP durum kodlarına çevirir. İş mantığı (kayıt, geri bildirim) app/service.py'dedir.
"""
import os
import secrets
from contextlib import asynccontextmanager
from typing import Literal

from fastapi import Depends, FastAPI, HTTPException, Request, status
from fastapi.responses import FileResponse
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from pydantic import BaseModel, Field

from app.service import ChatService


# --- Şemalar (API sözleşmesi) ---
# İstemcinin göreceği alan adları burada tanımlanır. Log dosyasının iç biçimi değişse bile bu sözleşme
# sabit kalabilir; dışarıya açılan API'lerde alan adları yaygın olarak İngilizcedir.

class Message(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=1000, examples=["Yatay geçiş başvurusu ne zaman yapılır?"])
    history: list[Message] = Field(default_factory=list, description="Önceki mesajlar (stateless: istemci tutar)")
    session_id: str | None = Field(default=None, description="Aynı sohbetin kayıtlarını gruplamak için")


class Source(BaseModel):
    chunk_id: int
    label: str
    # Bu alanlar 2026-10-06'dan önceki log kayıtlarında yok; bu yüzden isteğe bağlı.
    source: str | None = Field(default=None, description="PDF dosya adı (GET /files/{source})")
    title: str | None = None
    page: int | None = None
    madde: int | str | None = None


def _source(s):
    return Source(chunk_id=s["chunk_id"], label=s["etiket"], source=s.get("source"), title=s.get("title"),
                  page=s.get("page"), madde=s.get("madde"))


class AskResponse(BaseModel):
    id: str = Field(description="Bu cevabın kimliği; geri bildirimde kullanılır")
    answer: str
    rewritten_query: str | None = Field(description="Geçmişe göre yeniden yazılan soru; rewrite yoksa null")
    sources: list[Source]
    latency_ms: int


class FeedbackRequest(BaseModel):
    id: str
    rating: Literal[1, -1] = Field(description="1 = 👍, -1 = 👎")
    note: str | None = Field(default=None, max_length=1000)


class DocumentInfo(BaseModel):
    source: str
    title: str
    kind: str | None
    chunk_count: int


class ChunkInfo(BaseModel):
    chunk_id: int
    label: str
    text: str


class ChunkDetail(Source):
    text: str


class EvaluationRequest(BaseModel):
    id: str
    answer_label: Literal["dogru", "kismen", "yanlis", "cevapsiz"] = Field(
        description="cevapsiz: bilgi corpus'ta yokken doğru şekilde 'bilgim yok' dedi")
    source_ok: Literal["evet", "hayir", "emin_degil"] = Field(description="LLM'e doğru kaynak gitti mi?")
    error_stage: Literal["rewrite", "retrieval", "generation", "corpus_yok"] | None = Field(
        default=None, description="Yalnız yanlis / kismen için: hata hangi aşamada?")
    note: str | None = Field(default=None, max_length=2000)


class Evaluation(BaseModel):
    answer_label: str
    source_ok: str
    error_stage: str | None
    note: str | None
    time: str


class RecordOut(BaseModel):
    id: str
    time: str
    session_id: str | None
    question: str
    rewritten_query: str | None
    answer: str | None
    sources: list[Source]
    experiment: str
    commit: str | None
    latency_ms: int | None
    error: str | None
    feedback: int | None = Field(description="Son 👍 (1) / 👎 (-1); yoksa null")
    evaluation: Evaluation | None


class Health(BaseModel):
    status: str
    experiment: str
    commit: str | None
    documents: int


# --- Uygulama ---

@asynccontextmanager
async def lifespan(app: FastAPI):
    # yield'dan önceki kısım sunucu açılırken bir kez çalışır: modeller (index, bge-m3, reranker) burada
    # yüklenir, her istekte değil. yield'dan sonrası kapanışta çalışır (şimdilik temizlenecek bir şey yok).
    app.state.service = ChatService.from_experiment()
    yield


app = FastAPI(
    title="BAİBÜ Kampüs Asistanı API",
    description="Bolu Abant İzzet Baysal Üniversitesi mevzuatı üzerinde RAG tabanlı soru-cevap.",
    version="0.1.0",
    lifespan=lifespan,
)


def get_service(request: Request) -> ChatService:
    """Endpoint'lere service'i veren dependency. Testlerde app.dependency_overrides ile sahtesi verilir."""
    return request.app.state.service


# Endpoint'ler `async def` değil düz `def`: bot.ask() GPU ve Gemini'yi bekleyen bloklayan bir çağrı.
# FastAPI düz def endpoint'leri ayrı bir thread havuzunda çalıştırır; böylece bir soru cevaplanırken
# sunucu diğer istekleri (ör. /health) karşılamaya devam eder.

@app.get("/health", response_model=Health, tags=["sistem"])
def health(service: ChatService = Depends(get_service)):
    """Sunucu ayakta mı, hangi config ve kod sürümüyle çalışıyor?"""
    return Health(status="ok", experiment=service.experiment_id, commit=service.commit,
                  documents=len(service.documents()))


@app.post("/ask", response_model=AskResponse, tags=["sohbet"])
def ask(req: AskRequest, service: ChatService = Depends(get_service)):
    """Soruyu cevaplar. Her soru-cevap logs/sohbet.jsonl'e kaydedilir."""
    try:
        record = service.ask(req.question, [m.model_dump() for m in req.history], req.session_id)
    except Exception as e:
        # Hata service'te zaten kaydedildi; burada yalnız istemciye uygun HTTP cevabına çevrilir.
        if "ResourceExhausted" in type(e).__name__ or "429" in str(e):
            raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE,
                                "LLM kotası doldu (Gemini ücretsiz kota). Daha sonra tekrar deneyin.")
        raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, f"Cevap üretilemedi: {type(e).__name__}")
    return AskResponse(
        id=record["id"],
        answer=record["cevap"],
        rewritten_query=record["rewritten_query"],
        sources=[_source(s) for s in record["kaynaklar"]],
        latency_ms=record["latency_ms"],
    )


@app.post("/feedback", status_code=status.HTTP_201_CREATED, tags=["sohbet"])
def feedback(req: FeedbackRequest, service: ChatService = Depends(get_service)):
    """Bir cevaba 👍/👎 verir (logs/geri_bildirim.jsonl)."""
    try:
        service.feedback(req.id, req.rating, req.note)
    except KeyError:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"Böyle bir cevap kaydı yok: {req.id}")
    return {"ok": True}


@app.get("/documents", response_model=list[DocumentInfo], tags=["data"])
def documents(service: ChatService = Depends(get_service)):
    """Corpus'taki belgeler ve chunk sayıları."""
    return service.documents()


@app.get("/documents/{source}/chunks", response_model=list[ChunkInfo], tags=["data"])
def document_chunks(source: str, service: ChatService = Depends(get_service)):
    """Bir belgenin chunk'ları. `source` dosya adıdır (ör. yatay-gecis-yonergesi.pdf)."""
    try:
        rows = service.document_chunks(source)
    except KeyError:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"Corpus'ta böyle bir belge yok: {source}")
    return [ChunkInfo(chunk_id=r["chunk_id"], label=r["etiket"], text=r["metin"]) for r in rows]


@app.get("/chunks/{chunk_id}", response_model=ChunkDetail, tags=["data"])
def chunk(chunk_id: int, service: ChatService = Depends(get_service)):
    """Tek bir chunk'ın tam metni ve kaynağı (arayüzdeki kaynak paneli)."""
    try:
        c = service.chunk(chunk_id)
    except KeyError:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"Böyle bir chunk yok: {chunk_id}")
    return ChunkDetail(**_source(c).model_dump(), text=c["metin"])


@app.get("/files/{source}", tags=["data"], response_class=FileResponse)
def pdf_file(source: str, service: ChatService = Depends(get_service)):
    """Corpus'taki bir belgenin orijinal PDF'i. Belirli sayfa için URL'ye #page=N eklenir."""
    try:
        path = service.pdf_path(source)
    except KeyError:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"Corpus'ta böyle bir belge yok: {source}")
    # inline: tarayıcı dosyayı indirmek yerine kendi PDF görüntüleyicisinde açar.
    return FileResponse(path, media_type="application/pdf", filename=source, content_disposition_type="inline")


# --- Yönetici (elle değerlendirme) ---
# /admin altındaki her şey HTTP Basic Auth ile korunur: kullanıcıların soruları kişisel veridir ve linki bilen
# herkes değerlendirme verisine etiket yazabilmemeli. Şifre .env'deki ADMIN_PASSWORD'dan okunur (kullanıcı adı
# ADMIN_USER, varsayılan "admin"). ADMIN_PASSWORD tanımlı değilse panel tamamen kapalıdır (fail closed).
# Not: Basic Auth şifreyi yalnız base64 ile kodlar, şifrelemez; bu yüzden yalnız HTTPS üzerinden güvenlidir
# (Cloudflare Tunnel linki HTTPS'tir; localhost'ta trafik makineden çıkmaz).

_basic = HTTPBasic(realm="BAIBU admin")


def require_admin(credentials: HTTPBasicCredentials = Depends(_basic)):
    password = os.getenv("ADMIN_PASSWORD")
    if not password:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "Yönetici paneli kapalı: ADMIN_PASSWORD tanımlı değil.")
    # compare_digest: karşılaştırma süresi eşleşen karakter sayısına bağlı değildir (timing attack'a karşı).
    user_ok = secrets.compare_digest(credentials.username.encode(), os.getenv("ADMIN_USER", "admin").encode())
    pass_ok = secrets.compare_digest(credentials.password.encode(), password.encode())
    if not (user_ok and pass_ok):
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Kullanıcı adı veya şifre yanlış.",
                            headers={"WWW-Authenticate": 'Basic realm="BAIBU admin"'})
    return credentials.username


def _record(r):
    e = r["degerlendirme"]
    return RecordOut(
        id=r["id"], time=r["zaman"], session_id=r["oturum"], question=r["soru"],
        rewritten_query=r["rewritten_query"], answer=r["cevap"], sources=[_source(s) for s in r["kaynaklar"]],
        experiment=r["config"], commit=r["commit"], latency_ms=r["latency_ms"], error=r["hata"],
        feedback=r["geri_bildirim"],
        evaluation=Evaluation(answer_label=e["cevap_etiketi"], source_ok=e["kaynak_dogru"],
                              error_stage=e["hata_asamasi"], note=e["not"], time=e["zaman"]) if e else None,
    )


@app.get("/admin/records", response_model=list[RecordOut], tags=["yönetici"], dependencies=[Depends(require_admin)])
def records(service: ChatService = Depends(get_service)):
    """Bütün soru-cevap kayıtları (yeniden eskiye), son geri bildirim ve son değerlendirmeyle."""
    return [_record(r) for r in service.records()]


@app.post("/admin/evaluations", status_code=status.HTTP_201_CREATED, tags=["yönetici"], dependencies=[Depends(require_admin)])
def evaluate(req: EvaluationRequest, service: ChatService = Depends(get_service)):
    """Bir cevabı elle değerlendirir (logs/degerlendirme.jsonl). Tekrar gönderilirse en sonuncusu geçerli."""
    try:
        service.evaluate(req.id, req.answer_label, req.source_ok, req.error_stage, req.note)
    except KeyError:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"Böyle bir cevap kaydı yok: {req.id}")
    except ValueError as e:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, str(e))
    return {"ok": True}


@app.delete("/admin/records/{answer_id}", status_code=status.HTTP_204_NO_CONTENT, tags=["yönetici"],
            dependencies=[Depends(require_admin)])
def delete_record(answer_id: str, service: ChatService = Depends(get_service)):
    """Bir soru-cevabı ve ona bağlı geri bildirim / değerlendirmeleri log'lardan kalıcı olarak siler."""
    try:
        service.delete(answer_id)
    except KeyError:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"Böyle bir cevap kaydı yok: {answer_id}")


@app.get("/admin/summary", tags=["yönetici"], dependencies=[Depends(require_admin)])
def summary(service: ChatService = Depends(get_service)):
    """Değerlendirme özeti: etiket dağılımı, doğruluk, kaynak doğruluğu, hata aşamaları, 👍/👎 sayıları."""
    return service.summary()
