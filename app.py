import os
from fastapi import FastAPI, Request, UploadFile, File
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Fuzragion")

# Store uploaded files (temporary)
uploaded_files = []

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ====================== Serve Frontend ======================
@app.get("/app", response_class=HTMLResponse)
def serve_market_frontend():
    try:
        with open("market_frontend.html", "r", encoding="utf-8") as file:
            return HTMLResponse(content=file.read())
    except FileNotFoundError:
        return HTMLResponse("<h1>market_frontend.html not found!</h1>", 404)


# ====================== Auth Routes ======================
@app.post("/auth/login")
async def login(request: Request):
    try:
        data = await request.json()
        print(f"✅ Login: {data.get('email')}")
        return {"success": True, "message": "Login successful", "user": {"email": data.get("email"), "name": "Fuzail"}}
    except:
        return {"success": False, "message": "Invalid request"}, 400


@app.get("/auth/me")
async def auth_me():
    return {"success": True, "user": {"email": "fuzailshaik42@gmail.com", "name": "Fuzail"}}


# ====================== Main Routes ======================
@app.get("/health")
def health():
    return {"status": "healthy", "online": True}

@app.get("/chat/history")
async def chat_history():
    return {"success": True, "history": []}

@app.get("/files")
async def get_files():
    return {
        "success": True,
        "files": uploaded_files,           # Clean list
        "message": f"{len(uploaded_files)} file(s) found"
    }


# ====================== Upload Route ======================
@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    try:
        content = await file.read()
        
        file_info = {
            "id": len(uploaded_files) + 1,
            "filename": file.filename,
            "original_name": file.filename,
            "size": len(content),
            "type": file.content_type or "application/pdf",
            "uploaded_at": "Just now",
            "status": "indexed"
        }
        
        uploaded_files.append(file_info)
        print(f"✅ File Uploaded: {file.filename} ({len(content)} bytes)")
        
        return {
            "success": True,
            "message": f"{file.filename} uploaded successfully",
            "file": file_info
        }
    except Exception as e:
        return JSONResponse(status_code=500, content={"success": False, "message": str(e)})


# ====================== Local Development ======================
if __name__ == "__main__":
    import socket
    import uvicorn

    def get_ip():
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            s.connect(("8.8.8.8", 80))
            return s.getsockname()[0]
        except:
            return "127.0.0.1"
        finally:
            s.close()

    port = int(os.environ.get("PORT", 8001))
    ip = get_ip()

    print("\n" + "="*90)
    print("🚀 Fuzragion Server Started Successfully!")
    print("="*90)
    print(f"🌐 Frontend : http://127.0.0.1:{port}/app")
    print(f"🌐 Network  : http://{ip}:{port}/app")
    print("="*90 + "\n")

    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info", access_log=True)