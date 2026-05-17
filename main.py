import ast
import hashlib
import os
import tempfile
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from itertools import chain
from pathlib import Path

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, File, HTTPException, Query, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel
from sqlalchemy import func
from sqlalchemy.orm import Session

from auth.utils import send_email
from auth.dependencies import get_current_user
from auth.utils import decode_access_token
from auth.router import router as auth_router
from dependencies.database import DATABASE_URL, SessionLocal, engine, get_db
from models import Base, ChatHistory, Document, User

load_dotenv()

os.environ.setdefault("PDF_LOADER_LIGHTWEIGHT", "1")

BASE_DIR = Path(__file__).resolve().parent
PDFS_DIR = BASE_DIR / "pdfs"
DEFAULT_QDRANT_DIR = Path("/tmp/qdrant") if os.name != "nt" else (BASE_DIR / "qdrant_local_data")
QDRANT_DIR = Path(os.getenv("QDRANT_LOCAL_PATH", str(DEFAULT_QDRANT_DIR)))
COLLECTION_NAME = "learning-rag"
STORAGE_BACKEND = os.getenv("STORAGE_BACKEND", "local").strip().lower()
DISABLE_FREE_LIMITS = os.getenv("DISABLE_FREE_LIMITS", "0").strip().lower() in {
    "1",
    "true",
    "yes",
}

PDFS_DIR.mkdir(exist_ok=True)
QDRANT_DIR.mkdir(exist_ok=True)

# Initialize database tables on startup (can be done once safely)
try:
    Base.metadata.create_all(bind=engine)
except Exception as e:
    print(f"Warning: Database initialization failed: {e}")


def ensure_users_schema():
    if not DATABASE_URL.startswith("sqlite"):
        return
    try:
        with engine.connect() as conn:
            cols = {
                row[1]
                for row in conn.exec_driver_sql("PRAGMA table_info(users)").fetchall()
            }
            migrations = [
                (
                    "name",
                    "ALTER TABLE users ADD COLUMN name VARCHAR(255)",
                ),
                (
                    "phone_number",
                    "ALTER TABLE users ADD COLUMN phone_number VARCHAR(20)",
                ),
                (
                    "password_hash",
                    "ALTER TABLE users ADD COLUMN password_hash VARCHAR(255)",
                ),
                (
                    "subscription_plan",
                    "ALTER TABLE users ADD COLUMN subscription_plan VARCHAR(50) NOT NULL DEFAULT 'free'",
                ),
                (
                    "chat_count",
                    "ALTER TABLE users ADD COLUMN chat_count INTEGER NOT NULL DEFAULT 0",
                ),
            ]
            for col_name, sql in migrations:
                if col_name not in cols:
                    conn.exec_driver_sql(sql)
                    conn.commit()
                    print(f"Added users.{col_name} column")
                    cols.add(col_name)
    except Exception as exc:
        print(f"Warning: users schema migration skipped: {exc}")


ensure_users_schema()

# Lazy-loaded to avoid blocking imports
embedding_model = None
qdrant_client = None
vector_store = None
text_splitter = None
openai_client = None
s3_client = None
s3_bucket = os.getenv("S3_BUCKET", "").strip()
s3_region = os.getenv("AWS_REGION", "").strip() or None
is_hosted = any(
    os.getenv(name)
    for name in ("RENDER", "RENDER_SERVICE_ID", "RAILWAY_ENVIRONMENT", "FLY_APP_NAME")
)
if is_hosted and STORAGE_BACKEND != "s3":
    print(
        "Warning: Hosted deployment is not using S3 storage. Set STORAGE_BACKEND=s3 to persist PDFs."
    )


def _storage_key(user_id: str, file_name: str) -> str:
    return f"pdfs/{user_id}/{file_name}"


def _local_storage_path(user_id: str, file_name: str) -> Path | None:
    user_file = PDFS_DIR / user_id / file_name
    if user_file.exists():
        return user_file
    legacy_file = PDFS_DIR / file_name
    if legacy_file.exists():
        return legacy_file
    try:
        matches = [p for p in PDFS_DIR.rglob(file_name) if p.is_file()]
        if matches:
            return matches[0]
    except Exception:
        pass
    return None


def _file_exists_for_user(user_id: str, file_name: str) -> bool:
    if STORAGE_BACKEND == "s3":
        if not s3_bucket:
            return False
        try:
            client = _build_s3_client()
            client.head_object(Bucket=s3_bucket, Key=_storage_key(user_id, file_name))
            return True
        except Exception:
            return False
    return _local_storage_path(user_id, file_name) is not None


def _build_s3_client():
    global s3_client
    if s3_client is not None:
        return s3_client
    import boto3

    s3_client = boto3.client("s3", region_name=s3_region)
    return s3_client


def _upload_to_storage(user_id: str, file_name: str, content: bytes):
    if STORAGE_BACKEND == "s3":
        if not s3_bucket:
            raise RuntimeError("S3_BUCKET is required when STORAGE_BACKEND=s3")
        client = _build_s3_client()
        client.put_object(
            Bucket=s3_bucket,
            Key=_storage_key(user_id, file_name),
            Body=content,
            ContentType="application/pdf",
        )
        return

    user_dir = PDFS_DIR / user_id
    user_dir.mkdir(parents=True, exist_ok=True)
    target_path = user_dir / file_name
    with open(target_path, "wb") as out_file:
        out_file.write(content)


def _delete_from_storage(user_id: str, file_name: str):
    if STORAGE_BACKEND == "s3":
        if not s3_bucket:
            return
        client = _build_s3_client()
        client.delete_object(Bucket=s3_bucket, Key=_storage_key(user_id, file_name))
        return

    user_file = PDFS_DIR / user_id / file_name
    if user_file.exists():
        user_file.unlink()


def init_ai_components():
    """Initialize OpenAI and Qdrant components"""
    global embedding_model, qdrant_client, vector_store, text_splitter, openai_client

    if embedding_model is not None:
        return  # Already initialized

    print("Initializing AI components...")

    # Lazy import to avoid hanging on startup
    from openai import OpenAI
    from langchain_openai import OpenAIEmbeddings
    from langchain_qdrant import QdrantVectorStore
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from qdrant_client import QdrantClient
    from qdrant_client.models import (
        Distance,
        Filter,
        FieldCondition,
        MatchValue,
        VectorParams,
    )

    embedding_model = OpenAIEmbeddings(model="text-embedding-3-large")
    use_remote_qdrant = os.getenv("USE_REMOTE_QDRANT", "0").strip() == "1"
    qdrant_client = None
    if use_remote_qdrant:
        qdrant_url = os.getenv("QDRANT_URL", "").strip()
        qdrant_api_key = os.getenv("QDRANT_API_KEY", "").strip() or None
        if qdrant_url:
            try:
                qdrant_client = QdrantClient(
                    url=qdrant_url,
                    api_key=qdrant_api_key,
                    timeout=30,
                    check_compatibility=False,
                )
                qdrant_client.get_collections()
                print(f"Using remote Qdrant: {qdrant_url}")
            except Exception:
                qdrant_client = None

    if qdrant_client is None:
        qdrant_client = QdrantClient(path=str(QDRANT_DIR))
        print(f"Using local Qdrant: {QDRANT_DIR}")

    existing = [c.name for c in qdrant_client.get_collections().collections]
    if COLLECTION_NAME not in existing:
        qdrant_client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=3072, distance=Distance.COSINE),
        )

    vector_store = QdrantVectorStore(
        client=qdrant_client,
        collection_name=COLLECTION_NAME,
        embedding=embedding_model,
    )
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    openai_client = OpenAI()
    print("✓ AI components initialized")


def sync_existing_documents_to_vectors():
    """Rebuild vectors for PDFs already stored on disk."""
    if vector_store is None or qdrant_client is None:
        return

    from pdf_loader import split_file_into_chunks
    from qdrant_client.models import Filter, FieldCondition, MatchValue

    db = SessionLocal()
    try:
        rows = db.query(Document).all()
        if not rows:
            return

        for row in rows:
            file_path = PDFS_DIR / row.user_id / row.file_name
            if not file_path.exists():
                file_path = PDFS_DIR / row.file_name
            if not file_path.exists():
                continue

            try:
                qdrant_client.delete(
                    collection_name=COLLECTION_NAME,
                    points_selector=Filter(
                        must=[
                            FieldCondition(
                                key="metadata.file_id",
                                match=MatchValue(value=row.vector_file_id),
                            )
                        ]
                    ),
                )
            except Exception as exc:  # noqa: BLE001
                print(f"Skipping old vector cleanup for {row.file_name}: {exc}")

            chunks = split_file_into_chunks(str(file_path))
            if not chunks:
                print(f"Skipping reindex for {row.file_name}: no extractable text found")
                continue

            for chunk in chunks:
                chunk.metadata["file_id"] = row.vector_file_id
                chunk.metadata["source"] = row.file_name
                chunk.metadata["user_id"] = row.user_id

            if chunks:
                vector_store.add_documents(chunks)
                row.chunks_count = len(chunks)
                row.file_hash = get_hash(file_path)
                print(f"Reindexed: {row.file_name} ({len(chunks)} chunks)")

        db.commit()
    finally:
        db.close()


def prepare_document_chunks(file_path: str, *, file_id: str, source: str, user_id: str):
    """Load a document, split it into chunks, and attach storage metadata."""
    from pdf_loader import split_file_into_chunks

    chunks = split_file_into_chunks(file_path)
    if not chunks:
        raise HTTPException(
            status_code=422,
            detail=f"No extractable text found in {source}. The file may be image-only or corrupted.",
        )

    for chunk in chunks:
        chunk.metadata["file_id"] = file_id
        chunk.metadata["source"] = source
        chunk.metadata["user_id"] = user_id

    return chunks


app = FastAPI(title="Fuzragion RAG API", version="2.0.0")
app.include_router(auth_router)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def startup_reindex_documents():
    try:
        auto_reindex_raw = os.getenv("AUTO_REINDEX_ON_STARTUP", "").strip().lower()
        auto_reindex_enabled = auto_reindex_raw in {"1", "true", "yes"}
        auto_reindex_disabled = auto_reindex_raw in {"0", "false", "no"}

        # On hosted platforms, default to reindex ON unless explicitly disabled.
        if is_hosted and not auto_reindex_raw:
            auto_reindex_enabled = True
            print(
                "AUTO_REINDEX_ON_STARTUP not set; defaulting to ON for hosted deployment."
            )
        elif auto_reindex_disabled:
            auto_reindex_enabled = False

        if auto_reindex_enabled:
            init_ai_components()
            sync_existing_documents_to_vectors()
            print("Startup document sync complete. PDFs are indexed and ready.")
            return

        print(
            "Startup reindex skipped. Set AUTO_REINDEX_ON_STARTUP=1 to reindex existing PDFs on boot."
        )
    except Exception as exc:  # noqa: BLE001
        print(f"Startup document sync skipped: {exc}")


class AskRequest(BaseModel):
    query: str
    file_name: str | None = None
    mode: str = "normal"


class FeedbackRequest(BaseModel):
    type: str
    message: str
    rating: int | None = None


class SaveChatRequest(BaseModel):
    chat_id: str
    title: str
    messages: list[dict]


class LoadChatRequest(BaseModel):
    chat_id: str


def get_hash(path: Path) -> str:
    digest = hashlib.md5()
    with open(path, "rb") as file_obj:
        for chunk in iter(lambda: file_obj.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def detect_language(query: str) -> str:
    response = openai_client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[
            {
                "role": "system",
                "content": "Detect the language and return only language name in English.",
            },
            {"role": "user", "content": query},
        ],
        temperature=0,
    )
    return response.choices[0].message.content.strip()


def breakdown_query(query: str) -> list[str]:
    response = openai_client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[
            {
                "role": "system",
                "content": "Break into max 3 concise English search sub-queries. Return Python list only.",
            },
            {"role": "user", "content": query},
        ],
        temperature=0.1,
    )
    try:
        parsed = ast.literal_eval(response.choices[0].message.content.strip())
        if isinstance(parsed, list) and parsed:
            return [str(item) for item in parsed[:3]]
    except (SyntaxError, ValueError):
        pass
    return [query]


def retrieve_chunks(query: str, vector_file_id: str | None = None):
    if vector_file_id:
        return vector_store.similarity_search(
            query,
            k=6,
            filter=Filter(
                must=[
                    FieldCondition(
                        key="metadata.file_id", match=MatchValue(value=vector_file_id)
                    )
                ]
            ),
        )
    return vector_store.similarity_search(query, k=6)


def _vector_file_id(user_id: str, file_name: str) -> str:
    return f"{user_id}:{file_name}"


def is_user_premium(user: User, now: datetime) -> bool:
    return bool(
        user.subscription_plan == "pro"
        and user.subscription_expiry
        and user.subscription_expiry > now
    )


@app.on_event("shutdown")
async def shutdown_event():
    if qdrant_client is not None:
        qdrant_client.close()


def _frontend_file_response():
    frontend_file = BASE_DIR / "market_frontend.html"
    if frontend_file.exists():
        return FileResponse(frontend_file)
    return None


@app.get("/")
def root():
    frontend = _frontend_file_response()
    if frontend is not None:
        return frontend
    return {"status": "running"}


@app.get("/app", include_in_schema=False)
def serve_frontend():
    """Serve the frontend application"""
    frontend = _frontend_file_response()
    if frontend is not None:
        return frontend
    raise HTTPException(status_code=404, detail="Frontend not found")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/documents")
def list_documents(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    rows = (
        db.query(Document)
        .filter(Document.user_id == current_user.id)
        .order_by(Document.created_at.desc())
        .all()
    )
    return {
        "documents": [
            {
                "id": row.id,
                "file_name": row.file_name,
                "chunks_count": row.chunks_count,
                "created_at": row.created_at,
                "file_available": _file_exists_for_user(row.user_id, row.file_name),
            }
            for row in rows
        ]
    }


@app.get("/files")
def list_files(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    rows = (
        db.query(Document.file_name)
        .filter(Document.user_id == current_user.id)
        .order_by(Document.created_at.desc())
        .all()
    )
    return {"files": [row[0] for row in rows]}


@app.post("/upload")
async def upload(
    file: UploadFile = File(...),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    is_premium = is_user_premium(current_user, now)
    if DISABLE_FREE_LIMITS:
        is_premium = True

    init_ai_components()  # Initialize AI components on first use

    from qdrant_client.models import Filter, FieldCondition, MatchValue

    incoming_name = (file.filename or "").strip()
    safe_name = Path(incoming_name).name if incoming_name else "uploaded_document"

    file_bytes = await file.read()
    if not file_bytes:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")

    content_type = (file.content_type or "").lower().strip()
    has_pdf_extension = safe_name.lower().endswith(".pdf")
    has_pdf_signature = file_bytes.startswith(b"%PDF-")
    is_pdf_content_type = "application/pdf" in content_type

    # Mobile browsers may strip extensions or send generic mime types.
    if not has_pdf_extension and has_pdf_signature:
        safe_name = f"{safe_name}.pdf"
        has_pdf_extension = True

    if not (has_pdf_extension or has_pdf_signature or is_pdf_content_type):
        raise HTTPException(
            status_code=400,
            detail="Only PDF files are supported. Please select a valid PDF document.",
        )

    existing_doc = (
        db.query(Document)
        .filter(
            Document.user_id == current_user.id, Document.file_name == safe_name
        )
        .first()
    )
    pdf_count = (
        db.query(Document.id).filter(Document.user_id == current_user.id).count()
    )
    if not is_premium and not existing_doc and pdf_count >= 1:
        raise HTTPException(
            status_code=403,
            detail="Free plan includes 1 PDF upload. Delete your existing PDF or upgrade to Pro for unlimited uploads.",
        )

    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp_file:
        tmp_file.write(file_bytes)
        temp_path = Path(tmp_file.name)

    file_hash = get_hash(temp_path)
    vector_file_id = _vector_file_id(current_user.id, safe_name)

    try:
        chunks = prepare_document_chunks(
            str(temp_path),
            file_id=vector_file_id,
            source=safe_name,
            user_id=current_user.id,
        )
        if existing_doc:
            qdrant_client.delete(
                collection_name=COLLECTION_NAME,
                points_selector=Filter(
                    must=[
                        FieldCondition(
                            key="metadata.file_id",
                            match=MatchValue(value=existing_doc.vector_file_id),
                        )
                    ]
                ),
            )
        vector_store.add_documents(chunks)
        _upload_to_storage(current_user.id, safe_name, file_bytes)
    except HTTPException:
        raise
    except Exception as exc:
        # Prevent generic 500 and show actionable message for PDF parsing/chunking failures.
        raise HTTPException(
            status_code=422,
            detail=f"PDF process nahi ho paya: {safe_name}. File password-protected, scanned image-only, ya corrupt ho sakti hai. Error: {exc}",
        ) from exc
    finally:
        if temp_path.exists():
            temp_path.unlink()

    if not existing_doc:
        existing_doc = Document(
            user_id=current_user.id,
            file_name=safe_name,
            file_hash=file_hash,
            vector_file_id=vector_file_id,
            chunks_count=len(chunks),
        )
        db.add(existing_doc)
    else:
        existing_doc.file_hash = file_hash
        existing_doc.vector_file_id = vector_file_id
        existing_doc.chunks_count = len(chunks)
    db.commit()

    return {"message": f"Uploaded: {file.filename}", "chunks": len(chunks)}


@app.delete("/delete-document/{file_name}")
def delete_document(
    file_name: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    from qdrant_client.models import Filter, FieldCondition, MatchValue

    row = (
        db.query(Document)
        .filter(Document.user_id == current_user.id, Document.file_name == file_name)
        .first()
    )
    if not row:
        raise HTTPException(status_code=404, detail="File not found")

    qdrant_client.delete(
        collection_name=COLLECTION_NAME,
        points_selector=Filter(
            must=[
                FieldCondition(
                    key="metadata.file_id", match=MatchValue(value=row.vector_file_id)
                )
            ]
        ),
    )

    _delete_from_storage(current_user.id, file_name)

    db.delete(row)
    db.commit()
    return {"message": f"Deleted: {file_name}"}


@app.delete("/delete/{file_name}")
def delete_document_alias(
    file_name: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    return delete_document(file_name=file_name, current_user=current_user, db=db)


def _resolve_user_from_token(
    request: Request,
    access_token: str | None,
    db: Session,
) -> User:
    token = None
    auth_header = request.headers.get("authorization", "")
    if auth_header.lower().startswith("bearer "):
        token = auth_header.split(" ", 1)[1].strip()
    if not token and access_token:
        token = access_token.strip()
    if not token:
        raise HTTPException(status_code=401, detail="Missing token")

    try:
        payload = decode_access_token(token)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=401, detail="Invalid token") from exc

    user_id = payload.get("sub")
    if not user_id:
        raise HTTPException(status_code=401, detail="Invalid token payload")

    current_user = (
        db.query(User).filter(User.id == user_id, User.is_active.is_(True)).first()
    )
    if not current_user:
        raise HTTPException(status_code=401, detail="User not found or inactive")
    return current_user


def _build_document_response(current_user: User, file_name: str, db: Session):
    wanted_name = str(file_name or "").strip()
    row = (
        db.query(Document)
        .filter(Document.user_id == current_user.id, Document.file_name == wanted_name)
        .first()
    )
    if not row and wanted_name:
        # Fallback for case/whitespace differences from UI/client.
        row = (
            db.query(Document)
            .filter(
                Document.user_id == current_user.id,
                func.lower(func.trim(Document.file_name))
                == wanted_name.strip().lower(),
            )
            .first()
        )
    if not row:
        raise HTTPException(status_code=404, detail="File not found")

    if STORAGE_BACKEND == "s3":
        if not s3_bucket:
            raise HTTPException(status_code=500, detail="S3 bucket is not configured")
        try:
            client = _build_s3_client()
            obj = client.get_object(
                Bucket=s3_bucket, Key=_storage_key(current_user.id, row.file_name)
            )
            data = obj["Body"].read()
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=404, detail="File not found in storage") from exc

        headers = {"Content-Disposition": f'inline; filename="{row.file_name}"'}
        return Response(content=data, media_type="application/pdf", headers=headers)

    user_file = _local_storage_path(current_user.id, row.file_name)
    if user_file is None or not user_file.exists():
        raise HTTPException(
            status_code=404,
            detail="File not found in storage. Please re-upload this PDF.",
        )

    return FileResponse(
        path=user_file,
        media_type="application/pdf",
        filename=row.file_name,
        headers={"Content-Disposition": f'inline; filename="{row.file_name}"'},
    )


@app.get("/document/{file_name}")
def view_document(
    file_name: str,
    request: Request,
    access_token: str | None = Query(default=None),
    db: Session = Depends(get_db),
):
    current_user = _resolve_user_from_token(request, access_token, db)
    return _build_document_response(current_user, file_name, db)


@app.get("/document")
def view_document_query(
    request: Request,
    file_name: str = Query(...),
    access_token: str | None = Query(default=None),
    db: Session = Depends(get_db),
):
    current_user = _resolve_user_from_token(request, access_token, db)
    return _build_document_response(current_user, file_name, db)


def _run_answer(req: AskRequest, current_user: User, db: Session):
    init_ai_components()  # Initialize AI components on first use

    MAX_FREE_CHATS = 5
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    subscription_active = is_user_premium(current_user, now)
    if (
        current_user.subscription_plan == "pro"
        and current_user.subscription_expiry
        and current_user.subscription_expiry <= now
    ):
        current_user.subscription_plan = "free"
        db.commit()
    days_left = (
        max(0, (current_user.subscription_expiry - now).days)
        if subscription_active
        else 0
    )
    chats_used = int(current_user.chat_count or 0)
    limit_reached_message = (
        "⚠️ Free limit reached!\n\n"
        "You’ve used all 5 free chats.\n\n"
        "🚀 Upgrade to Premium to unlock:\n\n"
        "* Unlimited chat\n"
        "* PDF document analysis\n"
        "* High-quality expert answers\n\n"
        "💎 Activate 1 Month Premium now and continue instantly."
    )

    if not subscription_active and not DISABLE_FREE_LIMITS and chats_used >= MAX_FREE_CHATS:
        raise HTTPException(
            status_code=403,
            detail=limit_reached_message,
        )

    if not subscription_active and not DISABLE_FREE_LIMITS:
        current_user.chat_count += 1
        db.commit()
        chats_used = int(current_user.chat_count or 0)

    lang = detect_language(req.query)
    sub_queries = breakdown_query(req.query)

    vector_file_id = None
    if req.file_name:
        row = (
            db.query(Document)
            .filter(
                Document.user_id == current_user.id, Document.file_name == req.file_name
            )
            .first()
        )
        if not row:
            raise HTTPException(
                status_code=404, detail="Requested file not found for this user"
            )
        vector_file_id = row.vector_file_id

    with ThreadPoolExecutor() as executor:
        all_chunks = list(
            executor.map(lambda q: retrieve_chunks(q, vector_file_id), sub_queries)
        )

    unique_chunks = list(
        {doc.page_content: doc for doc in chain.from_iterable(all_chunks)}.values()
    )
    if not unique_chunks:
        return {
            "answer": "This information is not available in the provided documents.",
            "language": lang,
            "sources": [],
        }

    context = "\n\n".join(
        [
            f"[Source: {doc.metadata.get('source', 'unknown')} | Page: {doc.metadata.get('page', '?')}]\n{doc.page_content}"
            for doc in unique_chunks
        ]
    )

    # Mode-based system prompts
    mode_prompts = {
        "normal": "You are an expert document analyst. Reply in {lang} using Roman script. Answer only from context and cite source tags.",
        "business": "You are a strategic business consultant analyzing documents. Focus on ROI, scalability, market analysis, competition, and actionable business insights. Reply in {lang} using Roman script. Answer from context and cite sources.",
        "legal": "You are a legal expert analyzing documents. Provide accurate information and cite relevant laws/sections where possible. Always add disclaimer: 'This is not legal advice, consult a lawyer.' Reply in {lang} using Roman script. Answer from context and cite sources.",
        "research": "You are a research assistant analyzing documents. Provide detailed, cited, academic-level responses with latest knowledge. Suggest further reading. Reply in {lang} using Roman script. Answer from context and cite sources.",
    }

    base_prompt = mode_prompts.get(req.mode, mode_prompts["normal"])
    system_prompt = base_prompt.format(lang=lang) + f"\n\nContext:\n{context}"

    # Tier-based model and settings
    if subscription_active:
        model = "gpt-4.1"
        temperature = 0.5
        max_tokens = 2200
    else:
        model = "gpt-4.1-mini"
        temperature = 0.9
        max_tokens = 1000

    answer = openai_client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": req.query},
        ],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    answer_text = answer.choices[0].message.content
    if subscription_active and days_left <= 3:
        reminder = (
            f"⏳ Your Premium plan will expire in {days_left} day(s).\n\n"
            "To avoid interruption, please renew your plan.\n\n"
        )
        answer_text = reminder + answer_text

    return {
        "answer": answer_text,
        "language": lang,
        "sources": list(
            {
                f"{doc.metadata.get('source', '?')} - Page {doc.metadata.get('page', '?')}"
                for doc in unique_chunks
            }
        ),
        "tier": "premium" if subscription_active else "free",
        "chats_remaining": (
            max(0, MAX_FREE_CHATS - chats_used) if not subscription_active else None
        ),
        "chats_used": chats_used,
        "subscription_active": subscription_active,
        "days_left": days_left,
    }


@app.post("/ask")
def ask(
    req: AskRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    return _run_answer(req, current_user, db)


@app.post("/chat")
def chat(
    req: AskRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    return _run_answer(req, current_user, db)


@app.post("/query")
def query(
    req: AskRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    return _run_answer(req, current_user, db)


@app.post("/feedback")
def submit_feedback(
    req: FeedbackRequest,
    current_user: User = Depends(get_current_user),
):
    feedback_to = os.getenv("FEEDBACK_TO_EMAIL", "fuzailshaik42@gmail.com")
    kind = req.type.strip().lower()
    if kind not in {"feedback", "report"}:
        raise HTTPException(status_code=400, detail="Invalid feedback type")
    if not req.message.strip():
        raise HTTPException(status_code=400, detail="Message is required")

    subject_prefix = "Feedback" if kind == "feedback" else "Bug Report"
    subject = f"Fuzragion {subject_prefix} | {current_user.email}"
    rating_text = (
        f"<p><strong>Rating:</strong> {req.rating}/5</p>" if req.rating else ""
    )
    html_body = (
        f"<h3>New {subject_prefix}</h3>"
        f"<p><strong>From:</strong> {current_user.email}</p>"
        f"{rating_text}"
        f"<p><strong>Type:</strong> {kind}</p>"
        f"<p><strong>Message:</strong></p>"
        f"<pre style='white-space:pre-wrap;font-family:inherit'>{req.message.strip()}</pre>"
    )

    try:
        send_email(
            to_email=feedback_to,
            subject=subject,
            html_body=html_body,
            text_body=f"From: {current_user.email}\nType: {kind}\nRating: {req.rating}\n\n{req.message.strip()}",
        )
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Feedback email failed: {exc}",
        ) from exc

    return {"success": True, "message": "Feedback sent successfully."}


@app.post("/chat/save")
def save_chat(
    req: SaveChatRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    import json

    existing = (
        db.query(ChatHistory)
        .filter(
            ChatHistory.user_id == current_user.id, ChatHistory.chat_id == req.chat_id
        )
        .first()
    )

    messages_json = json.dumps(req.messages)

    if existing:
        existing.title = req.title
        existing.messages = messages_json
        existing.updated_at = datetime.utcnow()
    else:
        chat_history = ChatHistory(
            user_id=current_user.id,
            chat_id=req.chat_id,
            title=req.title,
            messages=messages_json,
        )
        db.add(chat_history)

    db.commit()
    return {"success": True}


@app.get("/chat/history")
def get_chat_history(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    import json

    histories = (
        db.query(ChatHistory)
        .filter(ChatHistory.user_id == current_user.id)
        .order_by(ChatHistory.updated_at.desc())
        .all()
    )

    result = []
    for h in histories:
        result.append(
            {
                "id": h.chat_id,
                "title": h.title,
                "date": h.updated_at.isoformat(),
                "msgs": json.loads(h.messages),
            }
        )

    return {"histories": result}


@app.delete("/chat/{chat_id}")
def delete_chat(
    chat_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    chat_history = (
        db.query(ChatHistory)
        .filter(ChatHistory.user_id == current_user.id, ChatHistory.chat_id == chat_id)
        .first()
    )

    if not chat_history:
        raise HTTPException(status_code=404, detail="Chat not found")

    db.delete(chat_history)
    db.commit()
    return {"success": True}

