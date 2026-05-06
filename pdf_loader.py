import atexit
import hashlib
import os
import sqlite3
import threading
from pathlib import Path
from typing import List, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydantic import BaseModel, Field
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    VectorParams,
)

load_dotenv()

COLLECTION_NAME = "learning-rag"
TRACKER_DB = "pdf_tracker.db"
SOURCE_PATH = "./pdfs/"
LOCAL_QDRANT_PATH = "./qdrant_local_data"
SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md"}
BASE_DIR = Path(__file__).resolve().parent
TRACKER_DB_PATH = BASE_DIR / TRACKER_DB
SOURCE_ROOT = BASE_DIR / "pdfs"
QDRANT_ROOT = BASE_DIR / "qdrant_local_data"
FRONTEND_ORIGINS = os.getenv("FRONTEND_ORIGINS", "")
ALLOWED_ORIGINS = [
    origin.strip() for origin in FRONTEND_ORIGINS.split(",") if origin.strip()
]


def get_hash(file_path: str) -> str:
    digest = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_qdrant_client() -> QdrantClient:
    qdrant_url = os.getenv("QDRANT_URL")
    qdrant_api_key = os.getenv("QDRANT_API_KEY")

    if qdrant_url:
        cloud_client = QdrantClient(
            url=qdrant_url,
            api_key=qdrant_api_key,
            check_compatibility=False,
        )
        try:
            cloud_client.get_collections()
            print(f"Using cloud Qdrant: {qdrant_url}")
            return cloud_client
        except Exception as exc:  # noqa: BLE001
            print(f"Cloud Qdrant unavailable ({exc}). Falling back to local Qdrant.")

    QDRANT_ROOT.mkdir(parents=True, exist_ok=True)
    local_client = QdrantClient(path=str(QDRANT_ROOT))
    print(f"Using local Qdrant: {QDRANT_ROOT}")
    return local_client


embedding_model = OpenAIEmbeddings(model="text-embedding-3-large")
qdrant_client = build_qdrant_client()

existing = [c.name for c in qdrant_client.get_collections().collections]
if COLLECTION_NAME not in existing:
    qdrant_client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=3072, distance=Distance.COSINE),
    )
    print(f"Collection created: {COLLECTION_NAME}")
else:
    print(f"Collection already exists: {COLLECTION_NAME}")

vector_store = QdrantVectorStore(
    client=qdrant_client,
    collection_name=COLLECTION_NAME,
    embedding=embedding_model,
)
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)

conn = sqlite3.connect(str(TRACKER_DB_PATH), check_same_thread=False, timeout=30)
db_lock = threading.Lock()
conn.execute(
    """
    CREATE TABLE IF NOT EXISTS files (
        file_id   TEXT PRIMARY KEY,
        file_name TEXT UNIQUE,
        file_hash TEXT
    )
    """
)
conn.commit()


def close_resources() -> None:
    try:
        conn.close()
    except Exception:  # noqa: BLE001
        pass
    try:
        qdrant_client.close()
    except Exception:  # noqa: BLE001
        pass


atexit.register(close_resources)


def db_fetchall(query: str, params=()):
    with db_lock:
        return conn.execute(query, params).fetchall()


def db_fetchone(query: str, params=()):
    with db_lock:
        return conn.execute(query, params).fetchone()


def db_execute(query: str, params=()):
    with db_lock:
        conn.execute(query, params)
        conn.commit()


def list_all_files():
    rows = db_fetchall("SELECT file_name FROM files ORDER BY file_name")
    return [row[0] for row in rows]


def load_docs(file_path: str):
    lowered = file_path.lower()
    if lowered.endswith(".pdf"):
        return PyPDFLoader(file_path).load()
    if lowered.endswith(".txt") or lowered.endswith(".md"):
        return TextLoader(file_path, encoding="utf-8").load()
    raise ValueError(f"Unsupported file type: {file_path}")


def delete_vectors_by_file_id(file_id: str) -> None:
    try:
        # Try deleting by file_id
        print(f"Attempting to delete vectors for file_id: {file_id}")
        qdrant_client.delete(
            collection_name=COLLECTION_NAME,
            points_selector=Filter(
                must=[
                    FieldCondition(
                        key="metadata.file_id",
                        match=MatchValue(value=file_id),
                    )
                ]
            ),
        )
        print(f"✓ Deleted by file_id: {file_id}")

        # Also try deleting by source name (filename) as fallback
        qdrant_client.delete(
            collection_name=COLLECTION_NAME,
            points_selector=Filter(
                must=[
                    FieldCondition(
                        key="metadata.source",
                        match=MatchValue(value=file_id),
                    )
                ]
            ),
        )
        print(f"✓ Deleted by source: {file_id}")

    except Exception as exc:
        print(f"⚠️ Error deleting vectors: {exc}")
        # If filters fail, this might mean LangChain stored metadata differently
        # In that case, the vectors might persist but this is a known limitation
        import traceback

        traceback.print_exc()


def upload_file(file_path: str) -> None:
    file_name = os.path.basename(file_path)
    file_hash = get_hash(file_path)
    file_id = file_name

    existing_row = db_fetchone(
        "SELECT file_id FROM files WHERE file_name = ?",
        (file_name,),
    )
    if existing_row:
        print(f"File already exists. Deleting old vectors...")
        delete_vectors_by_file_id(existing_row[0])

    docs = load_docs(file_path)
    chunks = text_splitter.split_documents(docs)
    for chunk in chunks:
        chunk.metadata["file_id"] = file_id
        chunk.metadata["source"] = file_name

    print(f"Adding {len(chunks)} chunks with file_id: {file_id}")
    vector_store.add_documents(chunks)
    db_execute(
        "INSERT OR REPLACE INTO files (file_id, file_name, file_hash) VALUES (?, ?, ?)",
        (file_id, file_name, file_hash),
    )
    print(f"Uploaded: {file_name} ({len(chunks)} chunks)")


def delete_file(file_name: str) -> bool:
    row = db_fetchone("SELECT file_id FROM files WHERE file_name = ?", (file_name,))
    if not row:
        print(f"File not found in database: {file_name}")
        return False

    file_id = row[0]
    print(f"Deleting file: {file_name} (file_id: {file_id})")
    delete_vectors_by_file_id(file_id)
    db_execute("DELETE FROM files WHERE file_name = ?", (file_name,))
    print(f"Deleted: {file_name}")
    return True


def sync_single_file(file_path: str) -> None:
    file_name = os.path.basename(file_path)
    current_hash = get_hash(file_path)
    row = db_fetchone("SELECT file_hash FROM files WHERE file_name = ?", (file_name,))

    if not row:
        print(f"New file: {file_name}")
        upload_file(file_path)
    elif row[0] != current_hash:
        print(f"Changed: {file_name} - re-indexing...")
        delete_file(file_name)
        upload_file(file_path)
    else:
        print(f"No changes: {file_name} - skipping")


def sync_path(path_value: str) -> None:
    if not os.path.exists(path_value):
        print(f"Path not found: {path_value}")
        return

    if os.path.isfile(path_value):
        ext = os.path.splitext(path_value)[1].lower()
        if ext not in SUPPORTED_EXTENSIONS:
            print(f"Unsupported file type: {path_value}")
            return
        sync_single_file(path_value)
        print("Sync complete.")
        return

    files = []
    for name in os.listdir(path_value):
        full_path = os.path.join(path_value, name)
        ext = os.path.splitext(name)[1].lower()
        if os.path.isfile(full_path) and ext in SUPPORTED_EXTENSIONS:
            files.append(full_path)

    if not files:
        print(f"No supported files found in: {path_value}")
        return

    for file_path in sorted(files):
        sync_single_file(file_path)
    print("Sync complete.")


def search(query: str, file_name: Optional[str] = None, top_k: int = 5) -> List[str]:
    if file_name:
        results = vector_store.similarity_search(
            query,
            k=top_k,
            filter=Filter(
                must=[
                    FieldCondition(
                        key="metadata.file_id",
                        match=MatchValue(value=file_name),
                    )
                ]
            ),
        )
    else:
        results = vector_store.similarity_search(query, k=top_k)
    return [r.page_content for r in results]


def resolve_user_file_path(user_input: str) -> Path:
    incoming = Path(user_input).expanduser()
    if incoming.is_absolute():
        incoming = incoming.resolve()
    else:
        incoming = (SOURCE_ROOT / incoming).resolve()

    try:
        incoming.relative_to(SOURCE_ROOT)
    except ValueError as exc:
        raise HTTPException(
            status_code=400, detail="file_path must be inside ./pdfs/"
        ) from exc

    if not incoming.exists() or not incoming.is_file():
        raise HTTPException(status_code=400, detail="File not found")

    if incoming.suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=400, detail=f"Unsupported extension: {incoming.suffix.lower()}"
        )

    return incoming


app = FastAPI(title="RAG Chunking and Search API")

if ALLOWED_ORIGINS:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=ALLOWED_ORIGINS,
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    print(f"CORS enabled for origins: {ALLOWED_ORIGINS}")
else:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    print("CORS enabled for all origins (development mode).")


class SearchRequest(BaseModel):
    query: str
    file_name: Optional[str] = None
    top_k: int = Field(default=5, ge=1, le=20)


class SearchResponse(BaseModel):
    results: List[str]


@app.on_event("startup")
def on_startup() -> None:
    SOURCE_ROOT.mkdir(parents=True, exist_ok=True)
    sync_path(str(SOURCE_ROOT))


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/files", response_model=List[str])
async def get_files():
    return list_all_files()


@app.post("/sync")
async def sync_files(path_value: str = Query(default=SOURCE_PATH)):
    if path_value == SOURCE_PATH:
        sync_path(str(SOURCE_ROOT))
    else:
        sync_path(path_value)
    return {"message": "Sync complete", "tracked_files": list_all_files()}


@app.post("/upload")
async def upload_fastapi(
    file_path: str = Query(..., description="Path inside ./pdfs/, example: RAG.PDF")
):
    safe_path = resolve_user_file_path(file_path)
    upload_file(str(safe_path))
    return {"message": "Uploaded successfully", "file_name": safe_path.name}


@app.post("/upload-file")
async def upload_binary_file(file: UploadFile = File(...)):
    original_name = os.path.basename(file.filename or "")
    if not original_name:
        raise HTTPException(status_code=400, detail="Invalid file name")

    ext = Path(original_name).suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise HTTPException(status_code=400, detail=f"Unsupported extension: {ext}")

    SOURCE_ROOT.mkdir(parents=True, exist_ok=True)
    target_path = SOURCE_ROOT / original_name

    try:
        content = await file.read()
        with open(target_path, "wb") as f:
            f.write(content)
        upload_file(str(target_path))
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=str(exc))
    finally:
        await file.close()

    return {"message": "Uploaded and indexed", "file_name": original_name}


@app.delete("/delete/{file_name}")
async def delete_fastapi(file_name: str):
    if not delete_file(file_name):
        raise HTTPException(status_code=404, detail="File not found")
    return {"message": "Deleted", "file_name": file_name}


@app.post("/search", response_model=SearchResponse)
async def search_fastapi(req: SearchRequest):
    try:
        return SearchResponse(results=search(req.query, req.file_name, req.top_k))
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=str(exc))


if __name__ == "__main__":
    import sys
    import uvicorn

    args = set(sys.argv[1:])
    Path(SOURCE_PATH).mkdir(parents=True, exist_ok=True)

    if "--serve" in args:
        host = os.getenv("API_HOST", "127.0.0.1")
        port = int(os.getenv("API_PORT", "8000"))
        uvicorn.run(app, host=host, port=port)
    else:
        target = SOURCE_PATH
        for value in sys.argv[1:]:
            if not value.startswith("--"):
                target = value
                break
        sync_path(target)
        print("Tracked files:", list_all_files())
