"""Main FastAPI application module for VeriTrust AI."""

from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import AsyncGenerator
from fastapi import Depends, FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.core.config import get_settings
from backend.app.db.session import async_engine, get_db, init_db
from backend.app.schemas.health import HealthResponse

settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan context manager handling startup and shutdown."""
    # Startup: Ensure database tables are created
    await init_db()
    yield
    # Shutdown: Cleanly dispose async engine connections
    await async_engine.dispose()


def create_app() -> FastAPI:
    """Factory function for creating and configuring the FastAPI application."""
    application = FastAPI(
        title="VeriTrust AI — Hallucination Guardrail System",
        description=(
            "Dual-Agent Maker & Judge compliance filter preventing unsupported, "
            "contradictory, or fabricated AI answers from reaching customers."
        ),
        version="1.0.0",
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
    )

    # Configure CORS for local development and frontend clients
    application.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Suitable for local development & cross-origin frontend
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # --------------------------------------------------------------------------
    # Root & Health Endpoints
    # --------------------------------------------------------------------------

    @application.get(
        "/",
        tags=["System"],
        summary="API Root Information",
    )
    async def root():
        return {
            "name": settings.APP_NAME,
            "version": "1.0.0",
            "environment": settings.APP_ENV,
            "status": "online",
            "docs": "/docs",
            "health": "/health",
        }

    @application.get(
        "/health",
        response_model=HealthResponse,
        tags=["System"],
        summary="System and Database Health Check",
    )
    async def health_check(db: AsyncSession = Depends(get_db)):
        db_status = "disconnected"
        system_status = "healthy"

        try:
            # Verify SQLite connectivity via a lightweight query
            result = await db.execute(text("SELECT 1"))
            val = result.scalar()
            if val == 1:
                db_status = "connected"
        except Exception:
            db_status = "error"
            system_status = "degraded"

        return HealthResponse(
            status=system_status,
            app_name=settings.APP_NAME,
            environment=settings.APP_ENV,
            database=db_status,
            llm_provider=settings.LLM_PROVIDER,
            timestamp=datetime.now(timezone.utc),
        )

    # --------------------------------------------------------------------------
    # API Routers
    # --------------------------------------------------------------------------
    from backend.app.api.v1.router import api_v1_router
    application.include_router(api_v1_router)

    return application


# Global application instance for uvicorn
app = create_app()
