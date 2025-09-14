"""
Rutas para predicción de palabras en ASL.
"""
from fastapi import APIRouter, HTTPException

from ...models.hand_data import (
    WordPredictionRequest,
    WordPredictionResponse,
)
from ...services.word_predictor import word_predictor
from ...core.logger import logger


router = APIRouter(prefix="/api/v1/word", tags=["sign-language-word"])


@router.post("/predict", response_model=WordPredictionResponse)
async def predict_word(request: WordPredictionRequest):
    try:
        # Validar frames (pydantic ya valida estructura)
        prediction = word_predictor.predict(
            sequence=request.sequence,
            confidence_threshold=request.confidence_threshold,
        )
        return prediction
    except Exception as exc:
        logger.error(f"Error en /word/predict: {exc}")
        raise HTTPException(status_code=500, detail="Error interno del servidor")


@router.get("/model/info")
async def get_word_model_info():
    return {
        "status": "stub",
        "model_version": word_predictor.model_version,
        "description": "Modelo stub temporal que agrega letras frame a frame",
        "expects": {
            "frames": "lista de frames con 21 puntos (x,y,z)",
            "min_frames": 1,
        },
    }





