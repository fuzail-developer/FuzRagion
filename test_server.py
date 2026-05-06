import sys
sys.path.insert(0, "c:/Users/fuzai/OneDrive/Desktop/genai_with_python/RAG")
import uvicorn
from main import app

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
