import ast
import atexit
import os
from concurrent.futures import ThreadPoolExecutor
from itertools import chain
from pathlib import Path
from typing import List, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from openai import OpenAI
from pydantic import BaseModel, Field
from qdrant_client import QdrantClient
from qdrant_client.models import FieldCondition, Filter, MatchValue

load_dotenv()

COLLECTION_NAME = "learning-rag"
LOCAL_QDRANT_PATH = "./qdrant_local_data"
BASE_DIR = Path(__file__).resolve().parent
QDRANT_ROOT = BASE_DIR / "qdrant_local_data"
FRONTEND_ORIGINS = os.getenv("FRONTEND_ORIGINS", "")
ALLOWED_ORIGINS = [
    origin.strip() for origin in FRONTEND_ORIGINS.split(",") if origin.strip()
]

openai_client = OpenAI()
embedding_model = OpenAIEmbeddings(model="text-embedding-3-large")


def build_qdrant_client() -> QdrantClient:
    qdrant_url = os.getenv("QDRANT_URL")
    qdrant_api_key = os.getenv("QDRANT_API_KEY")

    if qdrant_url:
        try:
            cloud_client = QdrantClient(
                url=qdrant_url,
                api_key=qdrant_api_key,
                check_compatibility=False,
            )
            cloud_client.get_collections()
            print(f"Using cloud Qdrant: {qdrant_url}")
            return cloud_client
        except Exception as exc:  # noqa: BLE001
            print(f"Cloud unavailable ({exc}). Using local Qdrant.")

    QDRANT_ROOT.mkdir(parents=True, exist_ok=True)
    local_client = QdrantClient(path=str(QDRANT_ROOT))
    print(f"Using local Qdrant: {QDRANT_ROOT}")
    return local_client


if os.getenv("REUSE_PDF_LOADER_CLIENT") == "1":
    from pdf_loader import (
        qdrant_client as qdrant_client,
    )  # reuse single local client to avoid lock
    from pdf_loader import vector_store as vector_store

    print("Reusing Qdrant client from pdf_loader.")
else:
    qdrant_client = build_qdrant_client()
    atexit.register(qdrant_client.close)
    vector_store = QdrantVectorStore(
        client=qdrant_client,
        collection_name=COLLECTION_NAME,
        embedding=embedding_model,
    )


def breakdown_query(user_query: str) -> List[str]:
    prompt = """
You are an expert query analyst. Your job is to:
1. Detect the language of the user query
2. Translate the query to English for better document retrieval
3. Break it into precise sub-queries in ENGLISH only

Rules:
- Maximum 3 sub-queries
- Sub-queries must ALWAYS be in English
- Each sub-query must target a different aspect
- Return a Python list only
- If query is simple, return one-item list
"""

    response = openai_client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": user_query},
        ],
        temperature=0.1,
    )
    content = (response.choices[0].message.content or "").strip()
    try:
        parsed = ast.literal_eval(content)
        if isinstance(parsed, list):
            cleaned = [str(x).strip() for x in parsed if str(x).strip()]
            return cleaned[:3] if cleaned else [user_query]
    except Exception:  # noqa: BLE001
        pass
    return [user_query]


def detect_language(user_query: str) -> str:
    prompt = """
Detect the language of the following text.
Return ONLY the language name in English.
Examples: English, Hindi, Urdu, French, Arabic, Spanish
"""
    response = openai_client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": user_query},
        ],
        temperature=0,
    )
    return (response.choices[0].message.content or "English").strip()


def retrieve_chunks(query: str, file_name: Optional[str], top_k: int):
    if file_name:
        return vector_store.similarity_search(
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
    return vector_store.similarity_search(query, k=top_k)


def generate_answer(user_query: str, user_language: str, context: str) -> str:
    system_prompt = f"""
You are a high-precision document intelligence system for heavy, complex, and professional documents.

ROLE:
- Act like a senior analyst, research assistant, and document reviewer
- Extract only the most relevant information
- Be precise, structured, and trustworthy
- Avoid teaching tone, casual tone, and unnecessary explanation

LANGUAGE RULES:
- User language: {user_language}
- Reply ONLY in that language using Roman/English letters
- Do not use Hindi, Urdu, Arabic, or any non-Latin script

CORE OBJECTIVE:
Turn dense document content into answers that are:
- accurate
- structured
- easy to scan
- decision-ready
- useful for business, legal, research, or technical reading

STRICT ANSWER RULES:
1. Use only the provided context.
2. Do not guess, assume, or invent missing details.
3. If the answer is not fully available, say so clearly.
4. Keep the answer concise, but expand when the question needs detail.
5. Prioritize the most important points first.
6. Avoid repetition and filler language.
7. If there are multiple relevant points, organize them logically.
8. If there is a conflict in the context, mention it instead of hiding it.
9. Use a professional tone at all times.
10. Never mention that you are an AI.

CITATION RULES:
- Every important factual point must include a source citation in this exact format:
  [Source: filename | Page: X]
- If page or source is missing, still cite what is available.
- Do not cite anything that is not supported by the context.

RESPONSE STYLE:
- Start with the direct answer.
- Then give the most important supporting points.
- Then mention conditions, exceptions, or limitations if relevant.
- End with a short conclusion or implication.
- Use bullets when they improve readability.
- Use short paragraphs for long explanations.
- Keep the language crisp, clear, and high-density.
- Use emojis lightly and professionally only when they improve readability, such as:
  🔹 for direct answer
  📌 for key points
  ⚠️ for important notes
  🎯 for conclusion
- Do not overuse emojis.

FOR HEAVY DOCUMENTS:
- Extract the main idea first.
- Highlight critical facts, definitions, clauses, or conclusions.
- Do not copy large blocks of text.
- Summarize intelligently, but never distort meaning.
- If the document is long or technical, focus on what matters most to the query.

WHEN INFORMATION IS NOT FOUND:
Reply clearly in {user_language} Roman script:
"Requested information is not available in the provided documents."

OUTPUT FORMAT:
1. Direct answer
2. Key points
3. Supporting details
4. Important notes or conditions
5. Final conclusion

CONTEXT:
{context}
"""

    response = openai_client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_query},
        ],
        temperature=0.15,
        max_tokens=1800,
    )
    return (response.choices[0].message.content or "").strip()


class AskRequest(BaseModel):
    query: str = Field(min_length=1)
    file_name: Optional[str] = None
    top_k: int = Field(default=6, ge=1, le=20)


class SourceItem(BaseModel):
    source: str
    page: str
    snippet: str


class AskResponse(BaseModel):
    language: str
    sub_queries: List[str]
    answer: str
    sources: List[SourceItem]


app = FastAPI(title="Multilingual RAG Retrieval API")

if ALLOWED_ORIGINS:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=ALLOWED_ORIGINS,
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )
else:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/ask", response_model=AskResponse)
async def ask(req: AskRequest):
    try:
        user_language = detect_language(req.query)
        sub_queries = breakdown_query(req.query)

        with ThreadPoolExecutor(max_workers=min(4, len(sub_queries) or 1)) as executor:
            all_chunks = list(
                executor.map(
                    lambda q: retrieve_chunks(q, req.file_name, req.top_k), sub_queries
                )
            )

        flattened = list(chain.from_iterable(all_chunks))
        unique_docs = list({doc.page_content: doc for doc in flattened}.values())

        if not unique_docs:
            return AskResponse(
                language=user_language,
                sub_queries=sub_queries,
                answer="Yeh information provided documents mein available nahi hai.",
                sources=[],
            )

        context = "\n\n".join(
            [
                f"[Source: {doc.metadata.get('source', 'unknown')} | Page: {doc.metadata.get('page', '?')}]\n{doc.page_content}"
                for doc in unique_docs
            ]
        )
        answer = generate_answer(req.query, user_language, context)

        sources = [
            SourceItem(
                source=str(doc.metadata.get("source", "unknown")),
                page=str(doc.metadata.get("page", "?")),
                snippet=doc.page_content[:220],
            )
            for doc in unique_docs[:8]
        ]

        return AskResponse(
            language=user_language,
            sub_queries=sub_queries,
            answer=answer,
            sources=sources,
        )
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=str(exc))


if __name__ == "__main__":
    import uvicorn

    host = os.getenv("API_HOST", "127.0.0.1")
    port = int(os.getenv("API_PORT", "8001"))
    uvicorn.run("retreive:app", host=host, port=port, reload=False)
