# src/main.py
import uvicorn
from dotenv import load_dotenv

load_dotenv()

from .api.routes import app


def main():
    uvicorn.run(
        "src.api.routes:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )


if __name__ == "__main__":
    main()
