"""
Modelos de datos para los puntos de la mano
"""
from pydantic import BaseModel, Field, validator
from typing import List, Tuple
import numpy as np


class HandPoint(BaseModel):
    """Representa un punto 3D de la mano"""
    x: float = Field(..., description="Coordenada X del punto")
    y: float = Field(..., description="Coordenada Y del punto")
    z: float = Field(..., description="Coordenada Z del punto")
    
    @validator('x', 'y', 'z')
    def validate_coordinates(cls, v):
        """Validar que las coordenadas estén en un rango válido"""
        if not isinstance(v, (int, float)):
            raise ValueError("Las coordenadas deben ser números")
        
        # Validar que las coordenadas estén en un rango razonable
        if not (-10.0 <= float(v) <= 10.0):
            raise ValueError(f"Coordenada {v} fuera del rango válido [-10.0, 10.0]")
        
        return float(v)


class HandLandmarks(BaseModel):
    """Representa los 21 puntos de la mano"""
    landmarks: List[HandPoint] = Field(..., min_items=21, max_items=21)
    
    @validator('landmarks')
    def validate_landmarks_count(cls, v):
        """Validar que se proporcionen exactamente 21 puntos"""
        if len(v) != 21:
            raise ValueError(f"Debe proporcionar exactamente 21 puntos de la mano, se proporcionaron {len(v)}")
        
        # Validar que todos los puntos sean válidos
        for i, point in enumerate(v):
            if not isinstance(point, HandPoint):
                raise ValueError(f"El punto {i} no es un HandPoint válido")
        
        return v
    
    def to_numpy(self) -> np.ndarray:
        """Convertir los landmarks a un array de NumPy"""
        points = []
        for landmark in self.landmarks:
            points.append([landmark.x, landmark.y, landmark.z])
        return np.array(points, dtype=np.float32)
    
    @classmethod
    def from_numpy(cls, array: np.ndarray) -> 'HandLandmarks':
        """Crear HandLandmarks desde un array de NumPy"""
        if array.shape != (21, 3):
            raise ValueError("El array debe tener forma (21, 3)")
        
        landmarks = []
        for point in array:
            landmarks.append(HandPoint(x=point[0], y=point[1], z=point[2]))
        
        return cls(landmarks=landmarks)


class PredictionRequest(BaseModel):
    """Request para predicción de lenguaje de señas"""
    hand_landmarks: HandLandmarks = Field(..., description="21 puntos de la mano")
    confidence_threshold: float = Field(default=0.5, ge=0.0, le=1.0, description="Umbral de confianza")
    
    @validator('confidence_threshold')
    def validate_confidence_threshold(cls, v):
        """Validar umbral de confianza"""
        if not isinstance(v, (int, float)):
            raise ValueError("El umbral de confianza debe ser un número")
        
        if not (0.0 <= float(v) <= 1.0):
            raise ValueError("El umbral de confianza debe estar entre 0.0 y 1.0")
        
        return float(v)


class PredictionResponse(BaseModel):
    """Response de predicción de lenguaje de señas"""
    predicted_word: str = Field(..., description="Palabra predicha")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Nivel de confianza de la predicción")
    processing_time_ms: float = Field(..., description="Tiempo de procesamiento en milisegundos")
    model_version: str = Field(default="1.0.0", description="Versión del modelo utilizado")


class HealthResponse(BaseModel):
    """Response del endpoint de salud"""
    status: str = Field(..., description="Estado de la aplicación")
    version: str = Field(..., description="Versión de la aplicación")
    message: str = Field(..., description="Mensaje de estado")


# ===== Modelos para predicción a nivel palabra =====

class WordSequence(BaseModel):
    """Secuencia de landmarks para una palabra (lista de frames)."""
    frames: List[HandLandmarks] = Field(..., min_items=1, description="Frames con 21 puntos por frame")

    @validator('frames')
    def validate_frames(cls, v):
        if len(v) < 1:
            raise ValueError("Debe incluir al menos 1 frame")
        return v

    def to_numpy(self) -> np.ndarray:
        """Devuelve array con forma (T, 21, 3)."""
        arrays = [frame.to_numpy() for frame in self.frames]
        return np.stack(arrays, axis=0)


class WordPredictionRequest(BaseModel):
    """Request para predicción de palabras en ASL (secuencia de frames)."""
    sequence: WordSequence = Field(..., description="Secuencia de frames con landmarks")
    confidence_threshold: float = Field(default=0.5, ge=0.0, le=1.0)


class WordPredictionResponse(BaseModel):
    """Response de predicción de palabra."""
    predicted_word: str = Field(...)
    confidence: float = Field(..., ge=0.0, le=1.0)
    processing_time_ms: float = Field(...)
    model_version: str = Field(default="1.0.0")
    sequence_length: int = Field(..., description="Número de frames procesados")