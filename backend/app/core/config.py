"""
Configuración central de la aplicación
"""
from pydantic_settings import BaseSettings
from typing import List


class Settings(BaseSettings):
    """Configuración de la aplicación"""
    
    # Configuración de la aplicación
    app_name: str = "Sign Language Recognition API"
    app_version: str = "1.0.0"
    debug: bool = False
    
    # Configuración del servidor
    host: str = "0.0.0.0"
    port: int = 8000
    
    # Configuración de CORS
    cors_origins: List[str] = [
        "http://localhost:3000",
        "http://localhost:5173",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:5173",
        "https://sign-language-app-frontend.vercel.app",  # Frontend en Vercel
        "https://*.vercel.app",  # Permitir cualquier subdominio de Vercel
        "https://*.netlify.app"  # Permitir cualquier subdominio de Netlify
    ]
    
    # Configuración de logging
    log_level: str = "INFO"
    
    class Config:
        env_file = ".env"


settings = Settings()