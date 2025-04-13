from fastapi import FastAPI, File, UploadFile, Form, Request, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import mimetypes
from PIL import Image
import numpy as np
from numpy.linalg import svd
import os

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

original_path = "static/original.png"
compressed_path = "static/compressed.png"


def compress_image(k: int) -> bool:
    if not os.path.exists(original_path):
        return False

    img = Image.open(original_path).convert("RGB")
    img_np = np.array(img)

    def compress_channel(channel, k):
        U, S, Vt = svd(channel, full_matrices=False)
        S_k = np.diag(S[:k])
        return np.clip(U[:, :k] @ S_k @ Vt[:k, :], 0, 255)

    compressed_channels = []
    for i in range(3):
        compressed = compress_channel(img_np[:, :, i], k)
        compressed_channels.append(compressed.astype(np.uint8))

    result = np.stack(compressed_channels, axis=2)
    compressed_img = Image.fromarray(result)
    compressed_img.save(compressed_path)
    return True

@app.get("/", response_class=HTMLResponse)
async def show_form(request: Request):
    return templates.TemplateResponse("index.html", {
        "request": request,
        "original_image": None,
        "compressed_image": None,
        "k": None
    })

@app.post("/", response_class=HTMLResponse)
async def handle_form(
    request: Request,
    file: UploadFile = File(None),
    k: int = Form(...)
):
    if file and file.filename:

        mime_type, _ = mimetypes.guess_type(file.filename)

        if mime_type not in ['image/jpeg', 'image/png', 'image/gif']:
            raise HTTPException(status_code=400, detail="Invalid file type")

        img = Image.open(file.file).convert("RGB")

        # Уменьшаем ширину до 1000px, если нужно
        max_width = 1000
        if img.width > max_width:
            scale = max_width / img.width
            new_size = (max_width, int(img.height * scale))
            img = img.resize(new_size, Image.LANCZOS)

        img.save(original_path)

    success = compress_image(k)

    return templates.TemplateResponse("index.html", {
        "request": request,
        "original_image": original_path if os.path.exists(original_path) else None,
        "compressed_image": compressed_path if success else None,
        "k": k
    })
