from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, FileResponse
import os
import live_face as yolo


app = FastAPI(debug=True)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
async def read_root():
    file_path = os.path.join(os.path.dirname(__file__), 'index.html')
    if os.path.exists(file_path):
        return FileResponse(file_path)
    return HTMLResponse(content="<h1>Файл index.html не найден</h1>", status_code=404)


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    queue = await yolo.start()
    try:
        async for frame in yolo.process_frames_async(queue):
            await websocket.send_bytes(frame)
    except WebSocketDisconnect:
        print("WebSocket отключен")
    except Exception as e:
        print(f"WebSocket ошибка: {e}")
    finally:
        await websocket.close()
