import os
import sys
import shutil
from pathlib import Path

# Ensure UTF-8 output on Windows consoles
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass

import uvicorn

def main():
    root_dir = Path(__file__).resolve().parent
    backend_dir = root_dir / "backend"
    env_file = backend_dir / ".env"
    env_example = backend_dir / ".env.example"

    print("=" * 60)
    print(" [STARTING] Full-Stack PDF RAG Assistant ")
    print("=" * 60)

    # Automatically create .env from .env.example if missing
    if not env_file.exists() and env_example.exists():
        shutil.copy(env_example, env_file)
        print(">> Created initial 'backend/.env' from '.env.example'.")
        print(">> NOTE: Make sure to check LLM_API_KEY in backend/.env!")
        print("-" * 60)

    # Add backend directory to Python sys.path so imports resolve
    if str(backend_dir) not in sys.path:
        sys.path.insert(0, str(backend_dir))

    print(">> Serving application at: http://127.0.0.1:8000")
    print(">> OpenAPI / Swagger Docs: http://127.0.0.1:8000/docs")
    print("=" * 60)

    uvicorn.run("main:app", app_dir=str(backend_dir), host="127.0.0.1", port=8000, reload=False)

if __name__ == "__main__":
    main()

