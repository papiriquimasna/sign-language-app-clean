"""
Configuración de logging
"""
import logging
import sys
from .config import settings


def setup_logging():
    """Configurar el sistema de logging"""
    
    # Configurar el formato de los logs
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    # Configurar el nivel de logging
    logging.basicConfig(
        level=getattr(logging, settings.log_level.upper()),
        format=log_format,
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler("app.log", encoding="utf-8")
        ]
    )
    
    # Crear logger para la aplicación
    logger = logging.getLogger("sign_language_app")
    logger.info("Sistema de logging configurado correctamente")
    
    return logger


# Inicializar el logger
logger = setup_logging()