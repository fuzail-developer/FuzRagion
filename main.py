import ast
import hashlib
import os
import tempfile
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from itertools import chain
from pathlib import Path

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from sqlalchemy.orm import Session

from auth.utils import send_email
from auth.dependencies import get_current_user
from auth.router import router as auth_router
from dependencies.database import DATABASE_URL, engine, get_db
from models import Base, Document, User

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
PDFS_DIR = BASE_DIR / "pdfs"
QDRANT_DIR = Path("/tmp/qdrant")
COLLECTION_NAME = "learning-rag"
STORAGE_BACKEND = os.getenv("STORAGE_BACKEND", "local").strip().lower()

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
    qdrant_url = os.getenv("QDRANT_URL", "").strip()
    qdrant_api_key = os.getenv("QDRANT_API_KEY", "").strip() or None
    if is_hosted and not qdrant_url:
        print(
            "Warning: QDRANT_URL is not set on hosted deployment. Falling back to local qdrant path (non-persistent)."
        )

    if qdrant_url:
        qdrant_client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key, timeout=30)
    else:
        qdrant_client = QdrantClient(path=str(QDRANT_DIR))

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


app = FastAPI(title="Fuzragion RAG API", version="2.0.0")
app.include_router(auth_router)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


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


@app.get("/")
def root():
    return {"status": "running"}


@app.get("/app", include_in_schema=False)
def serve_frontend():
    """Serve the frontend application"""
    frontend_file = BASE_DIR / "market_frontend.html"
    if frontend_file.exists():
        return FileResponse(frontend_file)
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
    if not is_premium:
        raise HTTPException(
            status_code=403,
            detail="PDF upload is available on Pro plan. Please upgrade to unlock unlimited PDF uploads.",
        )

    init_ai_components()  # Initialize AI components on first use

    # Lazy imports for PDF processing
    from langchain_community.document_loaders import PyPDFLoader
    from qdrant_client.models import Filter, FieldCondition, MatchValue

    if not file.filename:
        raise HTTPException(status_code=400, detail="Invalid file name")
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=400, detail="Only PDF upload is currently supported"
        )

    file_bytes = await file.read()
    if not file_bytes:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")

    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp_file:
        tmp_file.write(file_bytes)
        temp_path = Path(tmp_file.name)

    file_hash = get_hash(temp_path)
    vector_file_id = _vector_file_id(current_user.id, file.filename)

    existing_doc = (
        db.query(Document)
        .filter(
            Document.user_id == current_user.id, Document.file_name == file.filename
        )
        .first()
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

    try:
        docs = PyPDFLoader(str(temp_path)).load()
        chunks = text_splitter.split_documents(docs)
        for chunk in chunks:
            chunk.metadata["file_id"] = vector_file_id
            chunk.metadata["source"] = file.filename
            chunk.metadata["user_id"] = current_user.id
        vector_store.add_documents(chunks)
        _upload_to_storage(current_user.id, file.filename, file_bytes)
    finally:
        if temp_path.exists():
            temp_path.unlink()

    if not existing_doc:
        existing_doc = Document(
            user_id=current_user.id,
            file_name=file.filename,
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

    if not subscription_active and chats_used >= MAX_FREE_CHATS:
        raise HTTPException(
            status_code=403,
            detail=limit_reached_message,
        )

    if not subscription_active:
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

