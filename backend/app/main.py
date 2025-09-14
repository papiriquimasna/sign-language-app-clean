"""
Aplicación principal de reconocimiento de lenguaje de señas
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import uvicorn

from app.core.config import settings
from app.core.logger import logger
from app.api.v1.hand_routes import router as hand_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Gestionar el ciclo de vida de la aplicación
    """
    # Startup
    logger.info(f"Iniciando {settings.app_name} v{settings.app_version}")
    logger.info(f"Modo debug: {settings.debug}")
    
    yield
    
    # Shutdown
    logger.info("Cerrando aplicación...")


# Crear aplicación FastAPI
app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="API para reconocimiento de lenguaje de señas usando 21 puntos de la mano",
    docs_url="/docs" if settings.debug else None,
    redoc_url="/redoc" if settings.debug else None,
    lifespan=lifespan
)

# Configurar CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["*"],
)

# Incluir routers
app.include_router(hand_router)


@app.get("/")
async def root():
    """
    Endpoint raíz de la aplicación
    """
    return {
        "message": f"Bienvenido a {settings.app_name}",
        "version": settings.app_version,
        "status": "running",
        "docs": "/docs" if settings.debug else "Documentación no disponible en producción"
    }


@app.get("/health")
async def health():
    """
    Endpoint de salud simple
    """
    return {
        "status": "healthy",
        "version": settings.app_version
    }


if __name__ == "__main__":
    # Ejecutar servidor de desarrollo
    uvicorn.run(
        "app.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
        log_level=settings.log_level.lower()
    )