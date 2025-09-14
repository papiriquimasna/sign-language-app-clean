"""
Servicio de predicción de letras del alfabeto de señas
"""
import time
import random
from typing import Dict, List
import numpy as np
import os
from ..models.hand_data import HandLandmarks, PredictionResponse
from ..core.logger import logger

# Importaciones opcionales para ML
try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler
    import joblib
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False
    logger.warning("scikit-learn no disponible, usando predictor simplificado")


class LetterPredictor:
    """
    Servicio de predicción de letras del alfabeto de señas A-Z.
    
    Utiliza un modelo de machine learning entrenado para reconocer letras
    basado en los 21 puntos de la mano detectados por MediaPipe.
    """
    
    def __init__(self):
        """Inicializar el predictor"""
        self.model_version = "no_p_v1.0"
        self.letters = list("ABCDEFGHIKLMNOQRSTUVWXY")  # 23 letras sin J, Z y P
        self.model = None
        self.scaler = None
        self.keras_model = None
        self.keras_scaler = None
        self.is_trained = False
        
        # Intentar cargar modelo preentrenado
        self._load_or_create_model()
        
        logger.info(f"LetterPredictor inicializado para {len(self.letters)} letras")
    
    def _load_or_create_model(self):
        """Cargar modelo preentrenado o crear uno nuevo"""
        if not ML_AVAILABLE:
            logger.info("Usando predictor simplificado sin ML")
            self.is_trained = True
            return
            
        # Priorizar modelo Sign MNIST entrenado
        sign_mnist_model_path = "models/sign_mnist_svm.pkl"
        fallback_model_path = "models/no_jz_letter_classifier.pkl"
        fallback_scaler_path = "models/no_jz_letter_scaler.pkl"
        keras_path = "models/letter_keras.h5"
        
        try:
            # Priorizar modelo sin P
            no_p_model_path = "models/no_p_letter_classifier.pkl"
            no_p_scaler_path = "models/no_p_letter_scaler.pkl"
            
            if os.path.exists(no_p_model_path) and os.path.exists(no_p_scaler_path):
                self.model = joblib.load(no_p_model_path)
                self.scaler = joblib.load(no_p_scaler_path)
                self.is_trained = True
                logger.info("Modelo sin P cargado exitosamente (RandomForest)")
                return
            
            # Intentar cargar Keras si existe
            if os.path.exists(keras_path):
                try:
                    import tensorflow as tf  # type: ignore
                    self.keras_model = tf.keras.models.load_model(keras_path)
                    if os.path.exists(fallback_scaler_path):
                        self.keras_scaler = joblib.load(fallback_scaler_path)
                    self.is_trained = True
                    logger.info("Modelo Keras preentrenado cargado exitosamente")
                    return
                except Exception as ke:
                    logger.warning(f"Fallo cargando modelo Keras: {ke}")

            # Si no se pudo cargar ningún modelo, crear uno nuevo
            self._create_new_model()
        except Exception as e:
            logger.warning(f"Error cargando modelo: {e}. Creando nuevo modelo.")
            self._create_new_model()
    
    def _create_new_model(self):
        """Crear un nuevo modelo con datos sintéticos"""
        if not ML_AVAILABLE:
            logger.info("ML no disponible, usando predictor simplificado")
            self.is_trained = True
            return
            
        logger.info("Creando nuevo modelo con datos sintéticos...")
        
        # Generar datos sintéticos para entrenamiento
        X_train, y_train = self._generate_synthetic_data()
        
        # Crear y entrenar modelo
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        self.model = RandomForestClassifier(
            n_estimators=300,  # Más árboles para mejor precisión
            random_state=42,
            max_depth=15,  # Mayor profundidad
            min_samples_split=5,
            min_samples_leaf=2,
            max_features='sqrt'  # Mejor generalización
        )
        
        self.model.fit(X_train_scaled, y_train)
        self.is_trained = True
        
        # Guardar modelo
        os.makedirs("models", exist_ok=True)
        joblib.dump(self.model, "models/letter_classifier.pkl")
        joblib.dump(self.scaler, "models/letter_scaler.pkl")
        
        logger.info("Modelo creado y guardado exitosamente")
    
    def _generate_synthetic_data(self, samples_per_letter=200):
        """Generar datos sintéticos para entrenamiento con más variaciones"""
        X = []
        y = []
        
        for letter_idx, letter in enumerate(self.letters):
            for _ in range(samples_per_letter):
                # Generar landmarks sintéticos para cada letra con variaciones
                landmarks = self._generate_letter_landmarks(letter)
                # Aplicar características robustas
                features = self._create_robust_features(landmarks)
                X.append(features)
                y.append(letter_idx)
        
        return np.array(X), np.array(y)
    
    def _generate_letter_landmarks(self, letter):
        """Generar landmarks sintéticos para una letra específica"""
        # Base landmarks (mano en posición neutral)
        base_landmarks = np.array([
            [0.5, 0.5, 0.0],  # 0 - Wrist
            [0.5, 0.4, 0.0],  # 1 - Thumb CMC
            [0.6, 0.4, 0.0],  # 2 - Thumb MCP
            [0.7, 0.4, 0.0],  # 3 - Thumb IP
            [0.8, 0.4, 0.0],  # 4 - Thumb tip
            [0.4, 0.3, 0.0],  # 5 - Index MCP
            [0.4, 0.2, 0.0],  # 6 - Index PIP
            [0.4, 0.1, 0.0],  # 7 - Index DIP
            [0.4, 0.0, 0.0],  # 8 - Index tip
            [0.3, 0.3, 0.0],  # 9 - Middle MCP
            [0.3, 0.2, 0.0],  # 10 - Middle PIP
            [0.3, 0.1, 0.0],  # 11 - Middle DIP
            [0.3, 0.0, 0.0],  # 12 - Middle tip
            [0.2, 0.3, 0.0],  # 13 - Ring MCP
            [0.2, 0.2, 0.0],  # 14 - Ring PIP
            [0.2, 0.1, 0.0],  # 15 - Ring DIP
            [0.2, 0.0, 0.0],  # 16 - Ring tip
            [0.1, 0.3, 0.0],  # 17 - Pinky MCP
            [0.1, 0.2, 0.0],  # 18 - Pinky PIP
            [0.1, 0.1, 0.0],  # 19 - Pinky DIP
            [0.1, 0.0, 0.0],  # 20 - Pinky tip
        ])
        
        # Modificar landmarks según la letra
        landmarks = base_landmarks.copy()
        
        # Agregar variaciones específicas por letra
        letter_variations = {
            'A': self._letter_A_variation,
            'B': self._letter_B_variation,
            'C': self._letter_C_variation,
            'D': self._letter_D_variation,
            'E': self._letter_E_variation,
            'F': self._letter_F_variation,
            'G': self._letter_G_variation,
            'H': self._letter_H_variation,
            'I': self._letter_I_variation,
            'J': self._letter_J_variation,
            'K': self._letter_K_variation,
            'L': self._letter_L_variation,
            'M': self._letter_M_variation,
            'N': self._letter_N_variation,
            'O': self._letter_O_variation,
            'P': self._letter_P_variation,
            'Q': self._letter_Q_variation,
            'R': self._letter_R_variation,
            'S': self._letter_S_variation,
            'T': self._letter_T_variation,
            'U': self._letter_U_variation,
            'V': self._letter_V_variation,
            'W': self._letter_W_variation,
            'X': self._letter_X_variation,
            'Y': self._letter_Y_variation,
            'Z': self._letter_Z_variation,
        }
        
        if letter in letter_variations:
            landmarks = letter_variations[letter](landmarks)
        
        # Agregar ruido aleatorio
        noise = np.random.normal(0, 0.02, landmarks.shape)
        landmarks += noise
        
        return landmarks
    
    def _letter_A_variation(self, landmarks):
        """Variación para la letra A - Pulgar extendido, otros dedos cerrados"""
        landmarks[4] = [0.7, 0.2, 0.0]  # Thumb tip extendido
        landmarks[8] = [0.4, 0.3, 0.0]  # Index tip cerrado
        landmarks[12] = [0.3, 0.3, 0.0]  # Middle tip cerrado
        landmarks[16] = [0.2, 0.3, 0.0]  # Ring tip cerrado
        landmarks[20] = [0.1, 0.3, 0.0]  # Pinky tip cerrado
        return landmarks
    
    def _letter_B_variation(self, landmarks):
        """Variación para la letra B - Todos los dedos extendidos"""
        landmarks[4] = [0.6, 0.1, 0.0]  # Thumb tip extendido
        landmarks[8] = [0.4, 0.0, 0.0]  # Index tip extendido
        landmarks[12] = [0.3, 0.0, 0.0]  # Middle tip extendido
        landmarks[16] = [0.2, 0.0, 0.0]  # Ring tip extendido
        landmarks[20] = [0.1, 0.0, 0.0]  # Pinky tip extendido
        return landmarks
    
    def _letter_C_variation(self, landmarks):
        """Variación para la letra C - Forma de C (dedos curvados)"""
        landmarks[4] = [0.6, 0.2, 0.0]  # Thumb tip curvado
        landmarks[8] = [0.4, 0.1, 0.0]  # Index tip curvado
        landmarks[12] = [0.3, 0.1, 0.0]  # Middle tip curvado
        landmarks[16] = [0.2, 0.1, 0.0]  # Ring tip curvado
        landmarks[20] = [0.1, 0.1, 0.0]  # Pinky tip curvado
        return landmarks
    
    def _letter_D_variation(self, landmarks):
        """Variación para la letra D - Índice extendido, otros cerrados"""
        landmarks[4] = [0.5, 0.3, 0.0]  # Thumb tip cerrado
        landmarks[8] = [0.4, 0.0, 0.0]  # Index tip extendido
        landmarks[12] = [0.3, 0.3, 0.0]  # Middle tip cerrado
        landmarks[16] = [0.2, 0.3, 0.0]  # Ring tip cerrado
        landmarks[20] = [0.1, 0.3, 0.0]  # Pinky tip cerrado
        return landmarks
    
    def _letter_E_variation(self, landmarks):
        """Variación para la letra E - Todos los dedos cerrados"""
        landmarks[4] = [0.5, 0.3, 0.0]  # Thumb tip cerrado
        landmarks[8] = [0.4, 0.3, 0.0]  # Index tip cerrado
        landmarks[12] = [0.3, 0.3, 0.0]  # Middle tip cerrado
        landmarks[16] = [0.2, 0.3, 0.0]  # Ring tip cerrado
        landmarks[20] = [0.1, 0.3, 0.0]  # Pinky tip cerrado
        return landmarks
    
    def _letter_F_variation(self, landmarks):
        """Variación para la letra F - Índice y pulgar extendidos"""
        landmarks[4] = [0.6, 0.1, 0.0]  # Thumb tip extendido
        landmarks[8] = [0.4, 0.0, 0.0]  # Index tip extendido
        landmarks[12] = [0.3, 0.3, 0.0]  # Middle tip cerrado
        landmarks[16] = [0.2, 0.3, 0.0]  # Ring tip cerrado
        landmarks[20] = [0.1, 0.3, 0.0]  # Pinky tip cerrado
        return landmarks
    
    def _letter_G_variation(self, landmarks):
        """Variación para la letra G - Índice y pulgar en forma de pistola"""
        landmarks[4] = [0.5, 0.2, 0.0]  # Thumb tip extendido
        landmarks[8] = [0.4, 0.0, 0.0]  # Index tip extendido
        landmarks[12] = [0.3, 0.3, 0.0]  # Middle tip cerrado
        landmarks[16] = [0.2, 0.3, 0.0]  # Ring tip cerrado
        landmarks[20] = [0.1, 0.3, 0.0]  # Pinky tip cerrado
        return landmarks
    
    def _letter_H_variation(self, landmarks):
        """Variación para la letra H - Índice y medio extendidos"""
        landmarks[4] = [0.5, 0.3, 0.0]  # Thumb tip cerrado
        landmarks[8] = [0.4, 0.0, 0.0]  # Index tip extendido
        landmarks[12] = [0.3, 0.0, 0.0]  # Middle tip extendido
        landmarks[16] = [0.2, 0.3, 0.0]  # Ring tip cerrado
        landmarks[20] = [0.1, 0.3, 0.0]  # Pinky tip cerrado
        return landmarks
    
    def _letter_I_variation(self, landmarks):
        """Variación para la letra I - Meñique extendido, otros cerrados"""
        landmarks[4] = [0.5, 0.3, 0.0]  # Thumb tip cerrado
        landmarks[8] = [0.4, 0.3, 0.0]  # Index tip cerrado
        landmarks[12] = [0.3, 0.3, 0.0]  # Middle tip cerrado
        landmarks[16] = [0.2, 0.3, 0.0]  # Ring tip cerrado
        landmarks[20] = [0.1, 0.0, 0.0]  # Pinky tip extendido
        return landmarks
    
    def _letter_J_variation(self, landmarks):
        """Variación para la letra J"""
        landmarks[4] = [0.8, 0.3, 0.0]  # Thumb tip
        landmarks[8] = [0.4, 0.1, 0.0]  # Index tip
        landmarks[12] = [0.3, 0.1, 0.0]  # Middle tip
        landmarks[16] = [0.2, 0.1, 0.0]  # Ring tip
        landmarks[20] = [0.1, 0.0, 0.0]  # Pinky tip
        return landmarks
    
    def _letter_K_variation(self, landmarks):
        """Variación para la letra K"""
        landmarks[4] = [0.8, 0.3, 0.0]  # Thumb tip
        landmarks[8] = [0.4, 0.0, 0.0]  # Index tip
        landmarks[12] = [0.3, 0.0, 0.0]  # Middle tip
        landmarks[16] = [0.2, 0.1, 0.0]  # Ring tip
        landmarks[20] = [0.1, 0.1, 0.0]  # Pinky tip
        return landmarks
    
    def _letter_L_variation(self, landmarks):
        """Variación para la letra L"""
        landmarks[4] = [0.8, 0.3, 0.0]  # Thumb tip
        landmarks[8] = [0.4, 0.0, 0.0]  # Index tip
        landmarks[12] = [0.3, 0.1, 0.0]  # Middle tip
        landmarks[16] = [0.2, 0.1, 0.0]  # Ring tip
        landmarks[20] = [0.1, 0.1, 0.0]  # Pinky tip
        return landmarks
    
    def _letter_M_variation(self, landmarks):
        """Variación para la letra M"""
        landmarks[4] = [0.8, 0.3, 0.0]  # Thumb tip
        landmarks[8] = [0.4, 0.1, 0.0]  # Index tip
        landmarks[12] = [0.3, 0.1, 0.0]  # Middle tip
        landmarks[16] = [0.2, 0.1, 0.0]  # Ring tip
        landmarks[20] = [0.1, 0.1, 0.0]  # Pinky tip
        return landmarks
    
    def _letter_N_variation(self, landmarks):
        """Variación para la letra N"""
        landmarks[4] = [0.8, 0.3, 0.0]  # Thumb tip
        landmarks[8] = [0.4, 0.1, 0.0]  # Index tip
        landmarks[12] = [0.3, 0.1, 0.0]  # Middle tip
        landmarks[16] = [0.2, 0.1, 0.0]  # Ring tip
        landmarks[20] = [0.1, 0.1, 0.0]  # Pinky tip
        return landmarks
    
    def _letter_O_variation(self, landmarks):
        """Variación para la letra O"""
        landmarks[4] = [0.8, 0.3, 0.0]  # Thumb tip
        landmarks[8] = [0.4, 0.1, 0.0]  # Index tip
        landmarks[12] = [0.3, 0.1, 0.0]  # Middle tip
        landmarks[16] = [0.2, 0.1, 0.0]  # Ring tip
        landmarks[20] = [0.1, 0.1, 0.0]  # Pinky tip
        return landmarks
    
    def _letter_P_variation(self, landmarks):
        """Variación para la letra P"""
        landmarks[4] = [0.8, 0.3, 0.0]  # Thumb tip
        landmarks[8] = [0.4, 0.0, 0.0]  # Index tip
        landmarks[12] = [0.3, 0.0, 0.0]  # Middle tip
        landmarks[16] = [0.2, 0.1, 0.0]  # Ring tip
        landmarks[20] = [0.1, 0.1, 0.0]  # Pinky tip
        return landmarks
    
    def _letter_Q_variation(self, landmarks):
        """Variación para la letra Q"""
        landmarks[4] = [0.8, 0.3, 0.0]  # Thumb tip
        landmarks[8] = [0.4, 0.0, 0.0]  # Index tip
        landmarks[12] = [0.3, 0.1, 0.0]  # Middle tip
        landmarks[16] = [0.2, 0.1, 0.0]  # Ring tip
        landmarks[20] = [0.1, 0.1, 0.0]  # Pinky tip
        return landmarks
    
    def _letter_R_variation(self, landmarks):
        """Variación para la letra R"""
        landmarks[4] = [0.8, 0.3, 0.0]  # Thumb tip
        landmarks[8] = [0.4, 0.0, 0.0]  # Index tip
        landmarks[12] = [0.3, 0.0, 0.0]  # Middle tip
        landmarks[16] = [0.2, 0.1, 0.0]  # Ring tip
        landmarks[20] = [0.1, 0.1, 0.0]  # Pinky tip
        return landmarks
    
    def _letter_S_variation(self, landmarks):
        """Variación para la letra S - Mano cerrada (puño)"""
        landmarks[4] = [0.5, 0.3, 0.0]  # Thumb tip cerrado
        landmarks[8] = [0.4, 0.3, 0.0]  # Index tip cerrado
        landmarks[12] = [0.3, 0.3, 0.0]  # Middle tip cerrado
        landmarks[16] = [0.2, 0.3, 0.0]  # Ring tip cerrado
        landmarks[20] = [0.1, 0.3, 0.0]  # Pinky tip cerrado
        return landmarks
    
    def _letter_T_variation(self, landmarks):
        """Variación para la letra T"""
        landmarks[4] = [0.8, 0.3, 0.0]  # Thumb tip
        landmarks[8] = [0.4, 0.1, 0.0]  # Index tip
        landmarks[12] = [0.3, 0.1, 0.0]  # Middle tip
        landmarks[16] = [0.2, 0.1, 0.0]  # Ring tip
        landmarks[20] = [0.1, 0.1, 0.0]  # Pinky tip
        return landmarks
    
    def _letter_U_variation(self, landmarks):
        """Variación para la letra U - Índice y medio extendidos juntos"""
        landmarks[4] = [0.5, 0.3, 0.0]  # Thumb tip cerrado
        landmarks[8] = [0.4, 0.0, 0.0]  # Index tip extendido
        landmarks[12] = [0.3, 0.0, 0.0]  # Middle tip extendido
        landmarks[16] = [0.2, 0.3, 0.0]  # Ring tip cerrado
        landmarks[20] = [0.1, 0.3, 0.0]  # Pinky tip cerrado
        return landmarks
    
    def _letter_V_variation(self, landmarks):
        """Variación para la letra V - Índice y medio extendidos separados"""
        landmarks[4] = [0.5, 0.3, 0.0]  # Thumb tip cerrado
        landmarks[8] = [0.4, 0.0, 0.0]  # Index tip extendido
        landmarks[12] = [0.3, 0.0, 0.0]  # Middle tip extendido
        landmarks[16] = [0.2, 0.3, 0.0]  # Ring tip cerrado
        landmarks[20] = [0.1, 0.3, 0.0]  # Pinky tip cerrado
        return landmarks
    
    def _letter_W_variation(self, landmarks):
        """Variación para la letra W - Tres dedos extendidos (índice, medio, anular)"""
        landmarks[4] = [0.5, 0.3, 0.0]  # Thumb tip cerrado
        landmarks[8] = [0.4, 0.0, 0.0]  # Index tip extendido
        landmarks[12] = [0.3, 0.0, 0.0]  # Middle tip extendido
        landmarks[16] = [0.2, 0.0, 0.0]  # Ring tip extendido
        landmarks[20] = [0.1, 0.3, 0.0]  # Pinky tip cerrado
        return landmarks
    
    def _letter_X_variation(self, landmarks):
        """Variación para la letra X - Índice doblado"""
        landmarks[4] = [0.5, 0.3, 0.0]  # Thumb tip cerrado
        landmarks[8] = [0.4, 0.2, 0.0]  # Index tip doblado
        landmarks[12] = [0.3, 0.0, 0.0]  # Middle tip extendido
        landmarks[16] = [0.2, 0.3, 0.0]  # Ring tip cerrado
        landmarks[20] = [0.1, 0.3, 0.0]  # Pinky tip cerrado
        return landmarks
    
    def _letter_Y_variation(self, landmarks):
        """Variación para la letra Y - Meñique y pulgar extendidos"""
        landmarks[4] = [0.6, 0.1, 0.0]  # Thumb tip extendido
        landmarks[8] = [0.4, 0.3, 0.0]  # Index tip cerrado
        landmarks[12] = [0.3, 0.3, 0.0]  # Middle tip cerrado
        landmarks[16] = [0.2, 0.3, 0.0]  # Ring tip cerrado
        landmarks[20] = [0.1, 0.0, 0.0]  # Pinky tip extendido
        return landmarks
    
    def _letter_Z_variation(self, landmarks):
        """Variación para la letra Z - Índice extendido con movimiento (simulado)"""
        landmarks[4] = [0.5, 0.3, 0.0]  # Thumb tip cerrado
        landmarks[8] = [0.4, 0.0, 0.0]  # Index tip extendido
        landmarks[12] = [0.3, 0.3, 0.0]  # Middle tip cerrado
        landmarks[16] = [0.2, 0.3, 0.0]  # Ring tip cerrado
        landmarks[20] = [0.1, 0.3, 0.0]  # Pinky tip cerrado
        return landmarks
    
    def predict(self, hand_landmarks: HandLandmarks, confidence_threshold: float = 0.5) -> PredictionResponse:
        """
        Predecir la letra basada en los landmarks de la mano.
        
        Args:
            hand_landmarks: Los 21 puntos de la mano
            confidence_threshold: Umbral mínimo de confianza
            
        Returns:
            PredictionResponse: Resultado de la predicción
        """
        start_time = time.time()
        
        try:
            if not self.is_trained:
                raise Exception("Modelo no está entrenado")
            
            # Convertir landmarks a array de NumPy
            landmarks_array = hand_landmarks.to_numpy()
            
            if self.keras_model is not None:
                # Ruta Keras
                landmarks_normalized = self._normalize_landmarks(landmarks_array)
                features = landmarks_normalized.flatten().reshape(1, -1)
                if self.keras_scaler is not None:
                    features = self.keras_scaler.transform(features)
                import tensorflow as tf  # type: ignore
                proba = self.keras_model(features, training=False).numpy()[0]
                predicted_class_idx = int(np.argmax(proba))
                confidence = float(proba[predicted_class_idx])
                predicted_letter = self.letters[predicted_class_idx]
            elif ML_AVAILABLE and self.model is not None:
                # Usar características optimizadas para MediaPipe
                if True:  # Siempre usar características optimizadas
                    # Modelo RandomForest con características optimizadas para MediaPipe
                    features = self._extract_mediapipe_features(landmarks_array)
                    features_scaled = self.scaler.transform(features.reshape(1, -1))
                    prediction_proba = self.model.predict_proba(features_scaled)[0]
                    predicted_class_idx = np.argmax(prediction_proba)
                    confidence = prediction_proba[predicted_class_idx]
                    predicted_letter = self.letters[predicted_class_idx]
            else:
                # Usar predictor simplificado
                predicted_letter, confidence = self._simple_prediction(landmarks_array)
            
            # Verificar umbral de confianza
            if confidence < confidence_threshold:
                predicted_letter = "?"
                confidence = 0.0
            
            processing_time = (time.time() - start_time) * 1000  # Convertir a ms
            
            logger.info(f"Predicción completada: '{predicted_letter}' (confianza: {confidence:.2f})")
            
            return PredictionResponse(
                predicted_word=predicted_letter,
                confidence=confidence,
                processing_time_ms=processing_time,
                model_version=self.model_version
            )
            
        except Exception as e:
            logger.error(f"Error en predicción: {str(e)}")
            processing_time = (time.time() - start_time) * 1000
            
            return PredictionResponse(
                predicted_word="?",
                confidence=0.0,
                processing_time_ms=processing_time,
                model_version=self.model_version
            )
    
    def _create_robust_features(self, landmarks):
        """Crear características robustas para todas las letras"""
        features = []
        
        # 1. Características básicas (landmarks originales)
        features.extend(landmarks.flatten())
        
        # 2. Distancias entre puntos clave (reducido para evitar overflow)
        key_points = [0, 4, 8, 12, 16, 20, 9]  # Wrist, finger tips, palm center
        for i in key_points:
            for j in key_points:
                if i != j:
                    p1 = landmarks[i]
                    p2 = landmarks[j]
                    distance = np.linalg.norm(p1 - p2)
                    features.append(distance)
        
        # 3. Extensiones de dedos
        finger_pairs = [(4, 3), (8, 5), (12, 9), (16, 13), (20, 17)]
        for tip, base in finger_pairs:
            extension = np.linalg.norm(landmarks[tip] - landmarks[base])
            features.append(extension)
        
        # 4. Distancias al centro de la palma
        palm_center = landmarks[9]
        for i in [4, 8, 12, 16, 20]:  # Solo puntas de dedos
            distance = np.linalg.norm(landmarks[i] - palm_center)
            features.append(distance)
        
        # 5. Características de forma de la mano
        finger_tips = landmarks[[4, 8, 12, 16, 20]]
        finger_distances = [np.linalg.norm(landmarks[i] - palm_center) for i in [4, 8, 12, 16, 20]]
        features.extend(finger_distances)
        features.append(np.std(finger_distances))
        features.append(np.mean(finger_distances))
        features.append(np.min(finger_distances))
        features.append(np.max(finger_distances))
        
        # 6. Relaciones entre dedos
        thumb_index = np.linalg.norm(landmarks[4] - landmarks[8])
        index_middle = np.linalg.norm(landmarks[8] - landmarks[12])
        middle_ring = np.linalg.norm(landmarks[12] - landmarks[16])
        ring_pinky = np.linalg.norm(landmarks[16] - landmarks[20])
        
        features.extend([thumb_index, index_middle, middle_ring, ring_pinky])
        features.append(thumb_index / (index_middle + 1e-8))
        features.append(index_middle / (middle_ring + 1e-8))
        features.append(middle_ring / (ring_pinky + 1e-8))
        
        return np.array(features)

    def _extract_simple_features(self, landmarks):
        """
        Extraer características simples y consistentes de los landmarks
        """
        features = []
        
        # 1. Coordenadas originales (63 características: 21 puntos × 3 coordenadas)
        features.extend(landmarks.flatten())
        
        # 2. Distancias entre puntos clave (30 características)
        key_points = [0, 4, 8, 12, 16, 20]  # Palma, puntas de dedos
        for i in range(len(key_points)):
            for j in range(i+1, len(key_points)):
                p1 = landmarks[key_points[i]]
                p2 = landmarks[key_points[j]]
                distance = np.sqrt(np.sum((p1 - p2)**2))
                features.append(distance)
        
        # 3. Ángulos entre dedos (8 características)
        finger_tips = [4, 8, 12, 16, 20]
        palm_center = landmarks[0]
        
        for i in range(len(finger_tips)):
            for j in range(i+1, len(finger_tips)):
                v1 = landmarks[finger_tips[i]] - palm_center
                v2 = landmarks[finger_tips[j]] - palm_center
                
                # Calcular ángulo
                cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
                cos_angle = np.clip(cos_angle, -1, 1)
                angle = np.arccos(cos_angle)
                features.append(angle)
        
        return np.array(features)

    def _extract_mediapipe_features(self, landmarks):
        """
        Extraer características optimizadas para landmarks de MediaPipe
        """
        features = []
        
        # 1. Coordenadas originales (63 características: 21 puntos × 3 coordenadas)
        features.extend(landmarks.flatten())
        
        # 2. Distancias entre puntos clave (15 características)
        key_points = [0, 4, 8, 12, 16, 20]  # Wrist, finger tips
        for i in range(len(key_points)):
            for j in range(i+1, len(key_points)):
                p1 = landmarks[key_points[i]]
                p2 = landmarks[key_points[j]]
                distance = np.sqrt(np.sum((p1 - p2)**2))
                features.append(distance)
        
        # 3. Ángulos entre dedos (10 características)
        finger_tips = [4, 8, 12, 16, 20]
        wrist = landmarks[0]
        
        for i in range(len(finger_tips)):
            for j in range(i+1, len(finger_tips)):
                v1 = landmarks[finger_tips[i]] - wrist
                v2 = landmarks[finger_tips[j]] - wrist
                
                # Calcular ángulo
                cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
                cos_angle = np.clip(cos_angle, -1, 1)
                angle = np.arccos(cos_angle)
                features.append(angle)
        
        # 4. Características de forma de la mano (5 características)
        # Área de la mano
        finger_tips = landmarks[[4, 8, 12, 16, 20]]
        hand_area = 0.5 * abs(np.sum(finger_tips[:-1, 0] * finger_tips[1:, 1] - finger_tips[1:, 0] * finger_tips[:-1, 1]))
        features.append(hand_area)
        
        # Centroid
        centroid = np.mean(landmarks, axis=0)
        features.extend(centroid)
        
        # Varianza
        variance = np.var(landmarks, axis=0)
        features.extend(variance)
        
        return np.array(features)

    def _landmarks_to_image_features(self, landmarks):
        """
        Convertir landmarks de MediaPipe a características de imagen simulada
        para el modelo Sign MNIST
        """
        # Normalizar landmarks
        landmarks_normalized = self._normalize_landmarks(landmarks)
        
        # Crear una imagen simulada de 28x28 basada en los landmarks
        image = np.zeros((28, 28))
        
        # Mapear landmarks a píxeles de la imagen
        # Escalar landmarks a coordenadas de imagen
        scale_x = 14  # Centro de la imagen
        scale_y = 14
        
        # Dibujar puntos de landmarks como píxeles
        for i, landmark in enumerate(landmarks_normalized):
            x = int(scale_x + landmark[0] * 10)  # Escalar coordenada x
            y = int(scale_y + landmark[1] * 10)  # Escalar coordenada y
            
            # Asegurar que esté dentro de los límites
            x = max(0, min(27, x))
            y = max(0, min(27, y))
            
            # Dibujar punto con intensidad basada en la importancia del landmark
            if i in [4, 8, 12, 16, 20]:  # Puntas de dedos - más intensas
                image[y, x] = 255
                # Dibujar área alrededor del punto
                for dx in [-1, 0, 1]:
                    for dy in [-1, 0, 1]:
                        nx, ny = x + dx, y + dy
                        if 0 <= nx < 28 and 0 <= ny < 28:
                            image[ny, nx] = max(image[ny, nx], 200)
            elif i in [0, 9]:  # Wrist y palm center - intensidad media
                image[y, x] = 180
                for dx in [-1, 0, 1]:
                    for dy in [-1, 0, 1]:
                        nx, ny = x + dx, y + dy
                        if 0 <= nx < 28 and 0 <= ny < 28:
                            image[ny, nx] = max(image[ny, nx], 120)
            else:  # Otros puntos - intensidad baja
                image[y, x] = 100
        
        # Dibujar líneas entre puntos conectados para simular la estructura de la mano
        connections = [
            (0, 1), (1, 2), (2, 3), (3, 4),  # Thumb
            (0, 5), (5, 6), (6, 7), (7, 8),  # Index
            (0, 9), (9, 10), (10, 11), (11, 12),  # Middle
            (0, 13), (13, 14), (14, 15), (15, 16),  # Ring
            (0, 17), (17, 18), (18, 19), (19, 20),  # Pinky
        ]
        
        for start_idx, end_idx in connections:
            start_point = landmarks_normalized[start_idx]
            end_point = landmarks_normalized[end_idx]
            
            start_x = int(scale_x + start_point[0] * 10)
            start_y = int(scale_y + start_point[1] * 10)
            end_x = int(scale_x + end_point[0] * 10)
            end_y = int(scale_y + end_point[1] * 10)
            
            # Dibujar línea simple
            steps = max(abs(end_x - start_x), abs(end_y - start_y))
            if steps > 0:
                for t in np.linspace(0, 1, steps + 1):
                    x = int(start_x + t * (end_x - start_x))
                    y = int(start_y + t * (end_y - start_y))
                    if 0 <= x < 28 and 0 <= y < 28:
                        image[y, x] = max(image[y, x], 150)
        
        # Normalizar imagen a [0, 1]
        image = image / 255.0
        
        # Aplanar imagen a vector de 784 características (28x28)
        return image.flatten()

    def _normalize_landmarks(self, landmarks):
        """Normalizar landmarks respecto al centro de la palma"""
        # Centro de la palma (punto 9)
        palm_center = landmarks[9]
        
        # Normalizar respecto al centro de la palma
        normalized = landmarks - palm_center
        
        # Calcular escala basada en la distancia promedio de los dedos
        finger_tips = normalized[[4, 8, 12, 16, 20]]
        distances = np.linalg.norm(finger_tips, axis=1)
        scale = np.mean(distances)
        
        # Evitar división por cero
        if scale > 0:
            normalized = normalized / scale
        
        return normalized
    
    def _simple_prediction(self, landmarks: np.ndarray) -> tuple[str, float]:
        """
        Predicción simplificada basada en patrones realistas de landmarks para letras A-Z.
        """
        # Normalizar landmarks respecto al centro de la palma
        palm_center = landmarks[9]  # Centro de la palma
        normalized_landmarks = landmarks - palm_center
        
        # Obtener puntos clave
        thumb_tip = normalized_landmarks[4]
        index_tip = normalized_landmarks[8]
        middle_tip = normalized_landmarks[12]
        ring_tip = normalized_landmarks[16]
        pinky_tip = normalized_landmarks[20]
        
        # Calcular características
        finger_tips = np.array([thumb_tip, index_tip, middle_tip, ring_tip, pinky_tip])
        distances = np.linalg.norm(finger_tips, axis=1)
        
        # Detectar patrones específicos para cada letra
        letter, confidence = self._detect_letter_pattern(normalized_landmarks, distances)
        
        return letter, confidence
    
    def _detect_letter_pattern(self, landmarks: np.ndarray, finger_distances: np.ndarray) -> tuple[str, float]:
        """
        Detectar patrones específicos para cada letra del alfabeto de señas.
        """
        thumb_dist, index_dist, middle_dist, ring_dist, pinky_dist = finger_distances
        
        # Letra A: Pulgar extendido, otros dedos cerrados
        if thumb_dist > 0.15 and index_dist < 0.1 and middle_dist < 0.1 and ring_dist < 0.1 and pinky_dist < 0.1:
            return "A", 0.85
        
        # Letra B: Todos los dedos extendidos
        elif all(dist > 0.15 for dist in finger_distances):
            return "B", 0.80
        
        # Letra C: Forma de C (dedos curvados)
        elif 0.1 < index_dist < 0.2 and 0.1 < middle_dist < 0.2 and 0.1 < ring_dist < 0.2 and pinky_dist < 0.1:
            return "C", 0.75
        
        # Letra D: Índice extendido, otros cerrados
        elif index_dist > 0.15 and middle_dist < 0.1 and ring_dist < 0.1 and pinky_dist < 0.1:
            return "D", 0.80
        
        # Letra E: Todos los dedos cerrados
        elif all(dist < 0.1 for dist in finger_distances):
            return "E", 0.85
        
        # Letra F: Índice y pulgar extendidos
        elif index_dist > 0.15 and thumb_dist > 0.15 and middle_dist < 0.1 and ring_dist < 0.1 and pinky_dist < 0.1:
            return "F", 0.75
        
        # Letra G: Índice y pulgar en forma de pistola
        elif index_dist > 0.15 and thumb_dist > 0.1 and middle_dist < 0.1 and ring_dist < 0.1 and pinky_dist < 0.1:
            return "G", 0.70
        
        # Letra H: Índice y medio extendidos
        elif index_dist > 0.15 and middle_dist > 0.15 and ring_dist < 0.1 and pinky_dist < 0.1:
            return "H", 0.75
        
        # Letra I: Meñique extendido, otros cerrados
        elif pinky_dist > 0.15 and index_dist < 0.1 and middle_dist < 0.1 and ring_dist < 0.1:
            return "I", 0.80
        
        # Letra J: Meñique extendido con movimiento (simulado)
        elif pinky_dist > 0.15 and index_dist < 0.1 and middle_dist < 0.1 and ring_dist < 0.1:
            return "J", 0.70
        
        # Letra K: Índice y medio extendidos, pulgar extendido
        elif index_dist > 0.15 and middle_dist > 0.15 and thumb_dist > 0.1 and ring_dist < 0.1 and pinky_dist < 0.1:
            return "K", 0.75
        
        # Letra L: Índice y pulgar extendidos en L
        elif index_dist > 0.15 and thumb_dist > 0.1 and middle_dist < 0.1 and ring_dist < 0.1 and pinky_dist < 0.1:
            return "L", 0.70
        
        # Letra M: Tres dedos cerrados, índice y medio extendidos
        elif index_dist > 0.15 and middle_dist > 0.15 and ring_dist < 0.1 and pinky_dist < 0.1:
            return "M", 0.70
        
        # Letra N: Índice y medio extendidos, otros cerrados
        elif index_dist > 0.15 and middle_dist > 0.15 and ring_dist < 0.1 and pinky_dist < 0.1:
            return "N", 0.70
        
        # Letra O: Forma de O (dedos curvados)
        elif 0.1 < index_dist < 0.2 and 0.1 < middle_dist < 0.2 and 0.1 < ring_dist < 0.2 and 0.1 < pinky_dist < 0.2:
            return "O", 0.75
        
        # Letra P: Índice y medio extendidos hacia abajo
        elif index_dist > 0.15 and middle_dist > 0.15 and ring_dist < 0.1 and pinky_dist < 0.1:
            return "P", 0.70
        
        # Letra Q: Índice y pulgar extendidos
        elif index_dist > 0.15 and thumb_dist > 0.15 and middle_dist < 0.1 and ring_dist < 0.1 and pinky_dist < 0.1:
            return "Q", 0.70
        
        # Letra R: Índice y medio cruzados
        elif index_dist > 0.15 and middle_dist > 0.15 and ring_dist < 0.1 and pinky_dist < 0.1:
            return "R", 0.70
        
        # Letra S: Mano cerrada (puño)
        elif all(dist < 0.1 for dist in finger_distances):
            return "S", 0.85
        
        # Letra T: Pulgar entre índice y medio
        elif thumb_dist > 0.1 and index_dist < 0.1 and middle_dist < 0.1 and ring_dist < 0.1 and pinky_dist < 0.1:
            return "T", 0.70
        
        # Letra U: Índice y medio extendidos juntos
        elif index_dist > 0.15 and middle_dist > 0.15 and ring_dist < 0.1 and pinky_dist < 0.1:
            return "U", 0.75
        
        # Letra V: Índice y medio extendidos separados
        elif index_dist > 0.15 and middle_dist > 0.15 and ring_dist < 0.1 and pinky_dist < 0.1:
            return "V", 0.75
        
        # Letra W: Tres dedos extendidos (índice, medio, anular)
        elif index_dist > 0.15 and middle_dist > 0.15 and ring_dist > 0.15 and pinky_dist < 0.1:
            return "W", 0.75
        
        # Letra X: Índice doblado
        elif index_dist < 0.1 and middle_dist > 0.15 and ring_dist < 0.1 and pinky_dist < 0.1:
            return "X", 0.70
        
        # Letra Y: Meñique y pulgar extendidos
        elif pinky_dist > 0.15 and thumb_dist > 0.15 and index_dist < 0.1 and middle_dist < 0.1 and ring_dist < 0.1:
            return "Y", 0.75
        
        # Letra Z: Índice extendido con movimiento (simulado)
        elif index_dist > 0.15 and middle_dist < 0.1 and ring_dist < 0.1 and pinky_dist < 0.1:
            return "Z", 0.70
        
        # Si no coincide con ningún patrón específico, usar heurística general
        else:
            # Contar dedos extendidos
            extended_fingers = sum(1 for dist in finger_distances if dist > 0.15)
            
            if extended_fingers == 0:
                return "E", 0.60  # Mano cerrada
            elif extended_fingers == 1:
                if thumb_dist > 0.15:
                    return "A", 0.60
                elif index_dist > 0.15:
                    return "D", 0.60
                elif pinky_dist > 0.15:
                    return "I", 0.60
                else:
                    return "?", 0.30
            elif extended_fingers == 2:
                if index_dist > 0.15 and middle_dist > 0.15:
                    return "U", 0.60
                elif index_dist > 0.15 and thumb_dist > 0.15:
                    return "F", 0.60
                else:
                    return "?", 0.30
            elif extended_fingers == 3:
                return "W", 0.60
            elif extended_fingers == 4:
                return "B", 0.60
            else:
                return "?", 0.30
    
    def get_letters(self) -> List[str]:
        """Obtener lista de letras disponibles"""
        return self.letters.copy()
    
    def get_model_info(self) -> dict:
        """Obtener información del modelo"""
        return {
            "model_version": self.model_version,
            "letters_count": len(self.letters),
            "letters": self.letters,
            "is_trained": self.is_trained,
            "model_type": "RandomForestClassifier"
        }
    
    def add_word_to_vocabulary(self, word: str) -> bool:
        """
        Agregar una nueva palabra al vocabulario.
        
        NOTA: En producción, esto requeriría reentrenar el modelo.
        """
        if word not in self.letters:
            self.letters.append(word.upper())
            logger.info(f"Letra '{word.upper()}' agregada al vocabulario")
            return True
        return False
    
    def get_vocabulary(self) -> List[str]:
        """Obtener el vocabulario disponible (alias para get_letters)"""
        return self.get_letters()


# Instancia global del predictor
letter_predictor = LetterPredictor()


