"""
Servicio de predicción de palabras en ASL basado en secuencias de landmarks.

Este stub utiliza un enfoque simplificado: agrega características por frame,
promedia a lo largo del tiempo y aplica una heurística para inferir una "palabra"
(en esta versión, devuelve una letra dominante como palabra). En el futuro,
debería reemplazarse por un modelo temporal (p. ej. LSTM/Transformer) entrenado
con datasets como How2Sign o WLASL.
"""

import time
from typing import List, Tuple
import numpy as np

from ..models.hand_data import (
    WordSequence,
    WordPredictionResponse,
    HandLandmarks,
)
from ..services.letter_predictor import letter_predictor
from ..core.logger import logger


class WordPredictor:
    def __init__(self) -> None:
        self.model_version = "0.1.0"
        self.is_trained = True  # Stub listo para responder

    def predict(self, sequence: WordSequence, confidence_threshold: float = 0.5) -> WordPredictionResponse:
        start_time = time.time()
        try:
            frames_np = sequence.to_numpy()  # (T, 21, 3)
            num_frames = frames_np.shape[0]

            # Predicción frame a frame usando el predictor de letras existente
            letters: List[str] = []
            confidences: List[float] = []
            for t in range(num_frames):
                frame_landmarks = HandLandmarks.from_numpy(frames_np[t])
                letter_resp = letter_predictor.predict(frame_landmarks, confidence_threshold=0.0)
                letters.append(letter_resp.predicted_word)
                confidences.append(float(letter_resp.confidence))

            # Heurística: palabra = letra más frecuente (ignorando '?')
            filtered = [l for l in letters if l != '?']
            if len(filtered) == 0:
                predicted = "?"
                conf = 0.0
            else:
                values, counts = np.unique(np.array(filtered), return_counts=True)
                idx = int(np.argmax(counts))
                predicted = str(values[idx])
                # confianza combinando frecuencia y promedio de conf
                freq_conf = counts[idx] / max(1, len(letters))
                avg_conf = float(np.mean(confidences)) if len(confidences) > 0 else 0.0
                conf = float(0.5 * freq_conf + 0.5 * avg_conf)

            if conf < confidence_threshold:
                predicted = "?"
                conf = 0.0

            processing_time = (time.time() - start_time) * 1000.0
            logger.info(f"Predicción de palabra completada: '{predicted}' (conf: {conf:.2f}, frames: {num_frames})")

            return WordPredictionResponse(
                predicted_word=predicted,
                confidence=conf,
                processing_time_ms=processing_time,
                model_version=self.model_version,
                sequence_length=num_frames,
            )
        except Exception as exc:
            processing_time = (time.time() - start_time) * 1000.0
            logger.error(f"Error en WordPredictor: {exc}")
            return WordPredictionResponse(
                predicted_word="?",
                confidence=0.0,
                processing_time_ms=processing_time,
                model_version=self.model_version,
                sequence_length=sequence.to_numpy().shape[0] if sequence else 0,
            )


# Instancia global
word_predictor = WordPredictor()





