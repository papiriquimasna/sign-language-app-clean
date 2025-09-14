"""
Rutas para el reconocimiento de lenguaje de señas
"""
from fastapi import APIRouter, HTTPException, Depends
from fastapi.responses import JSONResponse
import time
from app.models.hand_data import (
    PredictionRequest, 
    PredictionResponse, 
    HealthResponse,
    HandLandmarks
)
from app.services.advanced_letter_predictor import advanced_letter_predictor
from app.services.personalized_predictor import personalized_predictor
from app.services.image_predictor import image_predictor
from app.core.logger import logger
from app.core.config import settings

# Crear router para las rutas de la mano
router = APIRouter(prefix="/api/v1", tags=["sign-language"])


@router.get("/", response_model=HealthResponse)
async def health_check():
    """
    Endpoint de salud de la API
    """
    return HealthResponse(
        status="healthy",
        version=settings.app_version,
        message="API de reconocimiento de lenguaje de señas funcionando correctamente"
    )


@router.post("/predict", response_model=PredictionResponse)
async def predict_sign_language(request: PredictionRequest):
    """
    Endpoint principal para predecir palabras del lenguaje de señas
    
    Recibe 21 puntos de la mano (3 coordenadas por punto) y devuelve
    la palabra predicha con su nivel de confianza.
    
    Args:
        request: Datos de la mano con 21 landmarks y umbral de confianza
        
    Returns:
        PredictionResponse: Palabra predicha, confianza y tiempo de procesamiento
    """
    try:
        logger.info("Nueva solicitud de predicción recibida")
        
        # Validar que se proporcionen exactamente 21 puntos
        if len(request.hand_landmarks.landmarks) != 21:
            raise HTTPException(
                status_code=400,
                detail="Debe proporcionar exactamente 21 puntos de la mano"
            )
        
        # Prioridad: Modelo de imágenes > Modelo personalizado > Modelo avanzado
        if image_predictor.is_available():
            logger.info("Usando modelo entrenado con dataset de imágenes")
            prediction = image_predictor.predict(
                hand_landmarks=request.hand_landmarks,
                confidence_threshold=request.confidence_threshold
            )
        elif personalized_predictor.is_available():
            logger.info("Usando modelo personalizado del usuario")
            prediction = personalized_predictor.predict(
                hand_landmarks=request.hand_landmarks,
                confidence_threshold=request.confidence_threshold
            )
        else:
            logger.info("Usando modelo avanzado preentrenado")
            prediction = advanced_letter_predictor.predict(
                hand_landmarks=request.hand_landmarks,
                confidence_threshold=request.confidence_threshold
            )
        
        logger.info(f"Predicción exitosa: {prediction.predicted_word}")
        
        return prediction
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error interno en predicción: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Error interno del servidor durante la predicción"
        )


@router.get("/letters")
async def get_letters():
    """
    Obtener las letras disponibles para predicción
    """
    try:
        if image_predictor.is_available():
            letters = image_predictor.get_available_letters()
            model_version = image_predictor.model_version
            model_type = "image_dataset"
        elif personalized_predictor.is_available():
            letters = personalized_predictor.get_available_letters()
            model_version = personalized_predictor.model_version
            model_type = "personalized"
        else:
            letters = advanced_letter_predictor.get_letters()
            model_version = advanced_letter_predictor.model_version
            model_type = "advanced_predefined"
            
        return {
            "letters": letters,
            "total_letters": len(letters),
            "model_version": model_version,
            "model_type": model_type
        }
    except Exception as e:
        logger.error(f"Error obteniendo letras: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Error obteniendo letras"
        )




@router.get("/model/info")
async def get_model_info():
    """
    Obtener información sobre el modelo de ML
    """
    if image_predictor.is_available():
        # Información del modelo de imágenes
        available_letters = image_predictor.get_available_letters()
        metadata = image_predictor.metadata or {}
        
        return {
            "model_type": "image_dataset",
            "model_version": image_predictor.model_version,
            "status": "image_trained_model",
            "description": f"Modelo entrenado con dataset de imágenes ({len(available_letters)} letras)",
            "available_letters": available_letters,
            "total_letters": len(available_letters),
            "accuracy": metadata.get('accuracy', 'N/A'),
            "total_samples": metadata.get('total_samples', 'N/A'),
            "samples_per_letter": metadata.get('samples_per_letter', 'N/A'),
            "features": {
                "input_points": 21,
                "coordinates_per_point": 3,
                "total_features": 63
            }
        }
    elif personalized_predictor.is_available():
        # Información del modelo personalizado
        available_letters = personalized_predictor.get_available_letters()
        metadata = personalized_predictor.metadata or {}
        
        return {
            "model_type": "personalized",
            "model_version": personalized_predictor.model_version,
            "status": "personalized_model",
            "description": f"Modelo personalizado entrenado con {len(available_letters)} letras",
            "available_letters": available_letters,
            "total_letters": len(available_letters),
            "accuracy": metadata.get('accuracy', 'N/A'),
            "total_samples": metadata.get('total_samples', 'N/A'),
            "features": {
                "input_points": 21,
                "coordinates_per_point": 3,
                "total_features": 63
            }
        }
    else:
        # Información del modelo avanzado
        model_info = advanced_letter_predictor.get_model_info()
        return {
            **model_info,
            "model_type": "advanced_predefined",
            "status": "trained_model",
            "description": "Modelo avanzado preentrenado para reconocimiento de letras A-Z del alfabeto de señas.",
            "features": {
                "input_points": 21,
                "coordinates_per_point": 3,
                "total_features": 63
            }
        }