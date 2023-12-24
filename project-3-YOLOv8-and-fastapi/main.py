'''
Instantiating a new FastAPI app and we added our yolo.router to it which has our two, predefined endpoints.
'''

from uvicorn import Server, Config
from fastapi import FastAPI
from starlette.middleware.cors import CORSMiddleware
import os
from routers import yolo

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_credentials=False,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(yolo.router)
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 80))
    server = Server(Config(app, host = "0.0.0.0", port = port, lifespan = "on"))
    server.run()