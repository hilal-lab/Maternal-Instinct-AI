"""
Maternal Instinct AI — FastAPI Application Entry Point

Architecture:
  routes/       -> API endpoint definitions
  controllers/  -> Request handling & validation
  services/     -> Business logic & DB interaction
  models/       -> Pydantic schemas & database setup
  core/         -> 4-Layer AI Architecture
  middleware/   -> Logging, error handling
  utils/        -> Helper functions
"""
import sys
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Ensure project root is importable
ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR))

from backend.models.database import init_db
from backend.middleware.logging import RequestLoggingMiddleware, setup_logging
from backend.routes.chat_routes import router as chat_router
from backend.routes.schedule_routes import router as schedule_router
from backend.routes.document_routes import router as document_router
from backend.routes.analytics_routes import router as analytics_router
from backend.routes.learning_routes import router as learning_router


# ─── Lifecycle ───────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    setup_logging()
    print("  Initializing database...")
    await init_db()

    from backend.models.seed import seed
    await seed()

    print("  Backend ready!")
    yield
    print("  Shutting down...")


# ─── App ─────────────────────────────────────────────────

app = FastAPI(
    title="Maternal Instinct AI",
    description="Layered Agentic AI Framework for Proactive Stress Management",
    version="2.0.0",
    lifespan=lifespan,
)

# Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(RequestLoggingMiddleware)

# Routes
app.include_router(chat_router, prefix="/api", tags=["Chat"])
app.include_router(schedule_router, prefix="/api", tags=["Schedule"])
app.include_router(document_router, prefix="/api", tags=["Documents"])
app.include_router(analytics_router, prefix="/api", tags=["Analytics"])
app.include_router(learning_router, prefix="/api", tags=["Learning"])


@app.get("/", tags=["Health"])
async def root():
    return {
        "name": "Maternal Instinct AI",
        "version": "2.0.0",
        "status": "running",
        "docs": "/docs",
    }


# ─── Run ─────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.app:app", host="0.0.0.0", port=8000, reload=True)
