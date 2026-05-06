# CogniVault - RAG Frontend-Backend Integration Guide

## ✅ Integration Complete!

Your RAG system is now fully integrated with a beautiful, production-ready frontend.

---

## 🚀 How to Start

### Run the Application
```bash
cd c:\Users\fuzai\OneDrive\Desktop\genai_with_python\RAG
python app.py
```

Server will start at: **http://127.0.0.1:8000**

---

## 📋 What Was Changed

### 1. **Backend (main.py)**
- Added Frontend HTML serving capability
- Updated `/` endpoint to serve `market_frontend.html`
- CORS enabled for all origins
- All API endpoints already working:
  - `POST /ask` - Send query, get RAG response with citations
  - `POST /upload` - Upload PDF/TXT/MD files
  - `GET /files` - List all uploaded files
  - `DELETE /delete/{filename}` - Remove files

### 2. **Frontend (market_frontend.html)**
- Connected all buttons to backend APIs
- File upload now saves to backend & Qdrant vector DB
- Chat messages call `/ask` endpoint
- File deletion calls `/delete` endpoint
- Automatic file sync from backend on load

---

## 💬 Features Working

✅ **Chat with Documents**
- Upload PDFs, TXT, or Markdown files
- Ask questions in any language (Hindi, English, etc.)
- Get answers with source citations
- Powered by OpenAI GPT-4.1-mini + Qdrant vector DB

✅ **File Management**
- Upload multiple documents
- See file status (Indexed)
- Delete files instantly
- Files sync between sessions

✅ **Subscription System**
- Free trial: 5 chats
- Paid plans with unlimited access
- Billing modal ready for payment integration

---

## 🛠️ System Architecture

```
Frontend (market_frontend.html)
    ↓
FastAPI Server (main.py)
    ├── PDF Loader (pdf_loader.py)
    ├── RAG Retriever (retreive.py)
    └── Vector DB (Qdrant)
         └── Local storage: ./qdrant_local_data
```

---

## 📚 API Examples

### Send a Query
```javascript
fetch('/ask', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({ 
    query: 'What is RAG?',
    file_name: null  // null = search all files
  })
})
.then(r => r.json())
.then(data => console.log(data.answer, data.sources))
```

### Upload a File
```javascript
const formData = new FormData();
formData.append('file', fileInput.files[0]);

fetch('/upload', {
  method: 'POST',
  body: formData
})
.then(r => r.json())
.then(data => console.log('Uploaded:', data.message))
```

---

## 🔧 Configuration

### Environment Variables (.env)
```env
OPENAI_API_KEY=your_key_here
QDRANT_URL=  # Leave empty for local
QDRANT_API_KEY=  # Leave empty for local
FRONTEND_ORIGINS=http://localhost:3000
```

### Supported File Types
- `.pdf` - PDF documents
- `.txt` - Text files
- `.md` - Markdown files

---

## 📊 Status

| Component | Status |
|-----------|--------|
| Frontend Loading | ✅ Working |
| Chat API | ✅ Working |
| File Upload | ✅ Working |
| File Deletion | ✅ Working |
| Vector DB | ✅ Working |
| Source Citations | ✅ Working |

---

## 💡 Next Steps

1. **Test with your documents**: Upload PDF files through the UI
2. **Ask questions**: Try asking in Hindi or English
3. **Customize**: Modify pricing plans in HTML if needed
4. **Deploy**: Use Docker or cloud platform for production

---

## 🎯 Example Usage Flow

1. Open http://127.0.0.1:8000
2. Click "Open Workspace"
3. Go to "My Files" tab
4. Upload a PDF file
5. Go back to "Chat with Docs"
6. Type: "What is the main topic of the document?"
7. Get instant answer with page reference!

---

## ❓ Troubleshooting

**Server won't start?**
- Kill existing Python processes: `taskkill /f /im python.exe`
- Make sure port 8000 is free
- Check OPENAI_API_KEY is set

**Files not uploading?**
- Ensure `./pdfs/` folder exists
- Check file size is reasonable
- Verify OPENAI_API_KEY is valid

**No responses?**
- Check OpenAI API quota
- Verify documents are indexed (check status)
- Try uploading a new document first

---

## 📞 Support

For issues with:
- **Frontend UI**: Check `market_frontend.html` JavaScript console
- **Backend API**: Check server terminal output
- **Vector DB**: Check `./qdrant_local_data/` folder

---

**Your RAG system is ready to use! 🚀**
