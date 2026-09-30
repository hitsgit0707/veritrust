"""API v1 router aggregating all endpoint routers."""

from fastapi import APIRouter
from backend.app.api.v1.endpoints.documents import router as documents_router
from backend.app.api.v1.endpoints.maker import router as maker_router

api_v1_router = APIRouter()
api_v1_router.include_router(documents_router)
api_v1_router.include_router(maker_router)
