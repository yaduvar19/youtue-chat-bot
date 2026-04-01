# src/main.py
import uvicorn
from dotenv import load_dotenv

load_dotenv()

from .api.routes import app

from fastapi import FastAPI
from src.api.chat import router as chat_router

app = FastAPI()

app.include_router(chat_router)
def main():
    uvicorn.run(
        "src.api.routes:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )


if __name__ == "__main__":
    main()
