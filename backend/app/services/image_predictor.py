#!/usr/bin/env python3
"""
Predictor que usa el modelo entrenado con dataset de imágenes
"""

import joblib
import json
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional
from app.models.hand_data import HandLandmarks, PredictionResponse
from app.core.logger import logger

class ImagePredictor:
    """Predictor que usa el modelo entrenado con dataset de imágenes"""
    
    def __init__(self):
        self.model = None
        self.scaler = None
        self.metadata = None
        self.model_version = "image_2.0.0"
        self._load_model()
    
    def _load_model(self):
        """Cargar el modelo entrenado con imágenes"""
        try:
            models_dir = Path("models")
            model_path = models_dir / "image_sample_model.pkl"
            scaler_path = models_dir / "image_sample_scaler.pkl"
            metadata_path = models_dir / "image_sample_metadata.json"
            
            if not model_path.exists() or not scaler_path.exists():
                logger.warning("Modelo de imágenes no encontrado")
                return
            
            # Cargar modelo y escalador
            self.model = joblib.load(model_path)
            self.scaler = joblib.load(scaler_path)
            
            # Cargar metadatos
            if metadata_path.exists():
                with open(metadata_path, 'r') as f:
                    self.metadata = json.load(f)
            
            logger.info(f"Modelo de imágenes cargado exitosamente")
            logger.info(f"Letras disponibles: {self.metadata.get('letters', [])}")
            logger.info(f"Precisión del modelo: {self.metadata.get('accuracy', 'N/A')}")
            
        except Exception as e:
            logger.error(f"Error cargando modelo de imágenes: {e}")
            self.model = None
            self.scaler = None
    
    def is_available(self) -> bool:
        """Verificar si el modelo de imágenes está disponible"""
        return self.model is not None and self.scaler is not None
    
    def get_available_letters(self) -> list:
        """Obtener las letras disponibles en el modelo de imágenes"""
        if self.metadata:
            return self.metadata.get('letters', [])
        return []
    
    def predict(self, hand_landmarks: HandLandmarks, confidence_threshold: float = 0.1) -> PredictionResponse:
        """Realizar predicción usando el modelo de imágenes"""
        if not self.is_available():
            return PredictionResponse(
                predicted_word="MODELO_NO_DISPONIBLE",
                confidence=0.0,
                processing_time_ms=0.0,
                model_version=self.model_version,
                error="Modelo de imágenes no está disponible"
            )
        
        try:
            import time
            start_time = time.time()
            
            # Convertir landmarks a array
            landmarks_array = self._landmarks_to_array(hand_landmarks)
            
            # Escalar características
            landmarks_scaled = self.scaler.transform([landmarks_array])
            
            # Realizar predicción
            prediction = self.model.predict(landmarks_scaled)[0]
            confidence = self.model.predict_proba(landmarks_scaled)[0].max()
            
            processing_time = (time.time() - start_time) * 1000
            
            # Verificar umbral de confianza
            if confidence < confidence_threshold:
                return PredictionResponse(
                    predicted_word="CONFIANZA_BAJA",
                    confidence=confidence,
                    processing_time_ms=processing_time,
                    model_version=self.model_version,
                    error=f"Confianza muy baja: {confidence:.3f}"
                )
            
            return PredictionResponse(
                predicted_word=prediction,
                confidence=confidence,
                processing_time_ms=processing_time,
                model_version=self.model_version
            )
            
        except Exception as e:
            logger.error(f"Error en predicción de imágenes: {e}")
            return PredictionResponse(
                predicted_word="ERROR",
                confidence=0.0,
                processing_time_ms=0.0,
                model_version=self.model_version,
                error=str(e)
            )
    
    def _landmarks_to_array(self, hand_landmarks: HandLandmarks) -> np.ndarray:
        """Convertir landmarks a array de características"""
        landmarks_array = []
        for landmark in hand_landmarks.landmarks:
            landmarks_array.extend([landmark.x, landmark.y, landmark.z])
        return np.array(landmarks_array)

# Instancia global del predictor de imágenes
image_predictor = ImagePredictor()
