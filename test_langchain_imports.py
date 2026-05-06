#!/usr/bin/env python3
import sys

print("1. Importing PyPDFLoader...", flush=True)
try:
    from langchain_community.document_loaders import PyPDFLoader

    print("  ✓ OK", flush=True)
except Exception as e:
    print(f"  ✗ Error: {e}", flush=True)

print("2. Importing OpenAIEmbeddings...", flush=True)
try:
    from langchain_openai import OpenAIEmbeddings

    print("  ✓ OK", flush=True)
except Exception as e:
    print(f"  ✗ Error: {e}", flush=True)

print("3. Importing QdrantVectorStore...", flush=True)
try:
    from langchain_qdrant import QdrantVectorStore

    print("  ✓ OK", flush=True)
except Exception as e:
    print(f"  ✗ Error: {e}", flush=True)

print("4. Importing QdrantClient...", flush=True)
try:
    from qdrant_client import QdrantClient

    print("  ✓ OK", flush=True)
except Exception as e:
    print(f"  ✗ Error: {e}", flush=True)

print("Done!", flush=True)
