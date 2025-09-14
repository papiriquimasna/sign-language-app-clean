"""
Servicio avanzado de predicción de letras con múltiples modelos
"""
import time
import random
from typing import Dict, List, Tuple
import numpy as np
import os
from ..models.hand_data import HandLandmarks, PredictionResponse
from ..core.logger import logger

# Importaciones para ML avanzado
try:
    from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier, VotingClassifier
    from sklearn.svm import SVC
    from sklearn.neural_network import MLPClassifier
    from sklearn.preprocessing import StandardScaler, RobustScaler
    from sklearn.model_selection import cross_val_score
    from sklearn.metrics import classification_report, accuracy_score
    import joblib
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False
    logger.warning("scikit-learn no disponible, usando predictor simplificado")


class AdvancedLetterPredictor:
    """
    Servicio avanzado de predicción de letras con múltiples modelos y ensemble
    """
    
    def __init__(self):
        """Inicializar el predictor avanzado"""
        self.model_version = "3.0.0"
        self.letters = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
        self.models = {}
        self.scalers = {}
        self.ensemble_model = None
        self.is_trained = False
        
        # Intentar cargar modelos preentrenados
        self._load_or_create_models()
        
        logger.info(f"AdvancedLetterPredictor inicializado para {len(self.letters)} letras")
    
    def _load_or_create_models(self):
        """Cargar modelos preentrenados o crear nuevos"""
        if not ML_AVAILABLE:
            logger.info("ML no disponible, usando predictor simplificado")
            self.is_trained = True
            return
        
        try:
            # Intentar cargar modelos existentes
            if (os.path.exists("models/advanced_ensemble.pkl") and 
                os.path.exists("models/advanced_scaler.pkl")):
                self.ensemble_model = joblib.load("models/advanced_ensemble.pkl")
                self.scalers['main'] = joblib.load("models/advanced_scaler.pkl")
                self.is_trained = True
                logger.info("Modelos avanzados cargados exitosamente")
            else:
                logger.info("Modelos no encontrados, creando nuevos modelos avanzados...")
                self._create_advanced_models()
        except Exception as e:
            logger.warning(f"Error cargando modelos: {e}. Creando nuevos modelos.")
            self._create_advanced_models()
    
    def _create_advanced_models(self):
        """Crear múltiples modelos avanzados con ensemble"""
        logger.info("🚀 Creando modelos avanzados con ensemble...")
        
        # Generar datos sintéticos mejorados
        X, y = self._generate_enhanced_synthetic_data()
        
        # Dividir datos
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Crear múltiples escaladores
        self.scalers['standard'] = StandardScaler()
        self.scalers['robust'] = RobustScaler()
        
        X_train_std = self.scalers['standard'].fit_transform(X_train)
        X_test_std = self.scalers['standard'].transform(X_test)
        X_train_rob = self.scalers['robust'].fit_transform(X_train)
        X_test_rob = self.scalers['robust'].transform(X_test)
        
        # Crear múltiples modelos (versión optimizada)
        models_config = {
            'random_forest_1': {
                'model': RandomForestClassifier(
                    n_estimators=200,
                    max_depth=15,
                    min_samples_split=3,
                    min_samples_leaf=2,
                    max_features='sqrt',
                    random_state=42,
                    n_jobs=-1
                ),
                'scaler': 'standard'
            },
            'random_forest_2': {
                'model': RandomForestClassifier(
                    n_estimators=150,
                    max_depth=12,
                    min_samples_split=4,
                    min_samples_leaf=2,
                    max_features='log2',
                    random_state=123,
                    n_jobs=-1
                ),
                'scaler': 'robust'
            },
            'extra_trees': {
                'model': ExtraTreesClassifier(
                    n_estimators=150,
                    max_depth=12,
                    min_samples_split=3,
                    min_samples_leaf=2,
                    max_features='sqrt',
                    random_state=42,
                    n_jobs=-1
                ),
                'scaler': 'standard'
            }
        }
        
        # Entrenar cada modelo
        trained_models = []
        for name, config in models_config.items():
            logger.info(f"🔄 Entrenando {name}...")
            
            # Seleccionar escalador
            if config['scaler'] == 'standard':
                X_train_scaled = X_train_std
                X_test_scaled = X_test_std
            else:
                X_train_scaled = X_train_rob
                X_test_scaled = X_test_rob
            
            # Entrenar modelo
            start_time = time.time()
            config['model'].fit(X_train_scaled, y_train)
            training_time = time.time() - start_time
            
            # Evaluar modelo
            y_pred = config['model'].predict(X_test_scaled)
            accuracy = accuracy_score(y_test, y_pred)
            
            logger.info(f"✅ {name} - Precisión: {accuracy:.3f} - Tiempo: {training_time:.2f}s")
            
            # Guardar modelo entrenado
            self.models[name] = config['model']
            trained_models.append((name, config['model']))
        
        # Crear ensemble con los mejores modelos
        self.ensemble_model = VotingClassifier(
            estimators=trained_models,
            voting='soft'  # Usar probabilidades
        )
        
        # Entrenar ensemble
        logger.info("🔄 Entrenando ensemble final...")
        start_time = time.time()
        self.ensemble_model.fit(X_train_std, y_train)
        training_time = time.time() - start_time
        
        # Evaluar ensemble
        y_pred = self.ensemble_model.predict(X_test_std)
        accuracy = accuracy_score(y_test, y_pred)
        
        logger.info(f"✅ Ensemble entrenado exitosamente!")
        logger.info(f"⏱️ Tiempo de entrenamiento: {training_time:.2f} segundos")
        logger.info(f"🎯 Precisión final: {accuracy:.3f}")
        
        # Mostrar reporte detallado
        logger.info("📊 Reporte de clasificación:")
        report = classification_report(y_test, y_pred, target_names=self.letters)
        logger.info(f"\n{report}")
        
        # Guardar modelos
        self._save_models()
        self.is_trained = True
        
        return True
    
    def _generate_enhanced_synthetic_data(self, samples_per_letter=100):
        """Generar datos sintéticos mejorados con más variaciones"""
        logger.info(f"📊 Generando {samples_per_letter} muestras por letra...")
        
        X = []
        y = []
        
        for letter_idx, letter in enumerate(self.letters):
            logger.info(f"Generando datos para letra {letter}...")
            for i in range(samples_per_letter):
                # Generar landmarks sintéticos con múltiples variaciones
                landmarks = self._generate_enhanced_letter_landmarks(letter, i)
                # Aplicar características avanzadas
                features = self._create_advanced_features(landmarks)
                X.append(features)
                y.append(letter_idx)
        
        logger.info(f"✅ Generados {len(X)} ejemplos de entrenamiento")
        return np.array(X), np.array(y)
    
    def _generate_enhanced_letter_landmarks(self, letter, variation_idx):
        """Generar landmarks sintéticos mejorados para una letra específica"""
        # Base landmarks más realistas
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
        
        # Modificar landmarks según la letra con múltiples variaciones
        landmarks = base_landmarks.copy()
        
        # Aplicar variaciones específicas por letra
        letter_variations = {
            'A': self._enhanced_letter_A,
            'B': self._enhanced_letter_B,
            'C': self._enhanced_letter_C,
            'D': self._enhanced_letter_D,
            'E': self._enhanced_letter_E,
            'F': self._enhanced_letter_F,
            'G': self._enhanced_letter_G,
            'H': self._enhanced_letter_H,
            'I': self._enhanced_letter_I,
            'J': self._enhanced_letter_J,
            'K': self._enhanced_letter_K,
            'L': self._enhanced_letter_L,
            'M': self._enhanced_letter_M,
            'N': self._enhanced_letter_N,
            'O': self._enhanced_letter_O,
            'P': self._enhanced_letter_P,
            'Q': self._enhanced_letter_Q,
            'R': self._enhanced_letter_R,
            'S': self._enhanced_letter_S,
            'T': self._enhanced_letter_T,
            'U': self._enhanced_letter_U,
            'V': self._enhanced_letter_V,
            'W': self._enhanced_letter_W,
            'X': self._enhanced_letter_X,
            'Y': self._enhanced_letter_Y,
            'Z': self._enhanced_letter_Z,
        }
        
        if letter in letter_variations:
            landmarks = letter_variations[letter](landmarks, variation_idx)
        
        # Agregar ruido realista
        noise_scale = 0.02
        noise = np.random.normal(0, noise_scale, landmarks.shape)
        landmarks += noise
        
        # Asegurar que los landmarks estén en el rango válido
        landmarks = np.clip(landmarks, 0, 1)
        
        return landmarks
    
    def _create_advanced_features(self, landmarks):
        """Crear características avanzadas para el modelo"""
        features = []
        
        # 1. Coordenadas originales (63 características)
        features.extend(landmarks.flatten())
        
        # 2. Distancias entre puntos clave (distances)
        key_points = [0, 4, 8, 12, 16, 20]  # Wrist, thumb tip, index tip, middle tip, ring tip, pinky tip
        for i in range(len(key_points)):
            for j in range(i + 1, len(key_points)):
                p1, p2 = key_points[i], key_points[j]
                dist = np.linalg.norm(landmarks[p1] - landmarks[p2])
                features.append(dist)
        
        # 3. Ángulos entre dedos
        finger_tips = [4, 8, 12, 16, 20]  # Tips de todos los dedos
        for i in range(len(finger_tips)):
            for j in range(i + 1, len(finger_tips)):
                for k in range(j + 1, len(finger_tips)):
                    p1, p2, p3 = finger_tips[i], finger_tips[j], finger_tips[k]
                    v1 = landmarks[p1] - landmarks[p2]
                    v2 = landmarks[p3] - landmarks[p2]
                    cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
                    cos_angle = np.clip(cos_angle, -1, 1)
                    angle = np.arccos(cos_angle)
                    features.append(angle)
        
        # 4. Extensiones de dedos (distancia desde base hasta tip)
        finger_bases = [1, 5, 9, 13, 17]  # Bases de dedos
        for base, tip in zip(finger_bases, finger_tips):
            extension = np.linalg.norm(landmarks[tip] - landmarks[base])
            features.append(extension)
        
        # 5. Área de la mano (aproximada)
        hull_points = [0, 4, 8, 12, 16, 20]  # Puntos del contorno
        hull_coords = landmarks[hull_points][:, :2]  # Solo x, y
        area = 0.5 * abs(sum(hull_coords[i][0] * (hull_coords[(i+1) % len(hull_coords)][1] - hull_coords[i-1][1]) 
                            for i in range(len(hull_coords))))
        features.append(area)
        
        # 6. Centroide de la mano
        centroid = np.mean(landmarks, axis=0)
        features.extend(centroid)
        
        # 7. Varianza de posiciones
        variance = np.var(landmarks, axis=0)
        features.extend(variance)
        
        # 8. Distancias desde el centroide
        for landmark in landmarks:
            dist_from_centroid = np.linalg.norm(landmark - centroid)
            features.append(dist_from_centroid)
        
        return np.array(features)
    
    def predict(self, landmarks: HandLandmarks, confidence_threshold: float = 0.1) -> PredictionResponse:
        """Realizar predicción con el ensemble de modelos"""
        if not self.is_trained or not self.ensemble_model:
            return PredictionResponse(
                predicted_word="?",
                confidence=0.0,
                processing_time_ms=0.0,
                model_version=self.model_version
            )
        
        start_time = time.time()
        
        try:
            # Convertir landmarks a array
            landmarks_array = landmarks.to_numpy()
            
            # Crear características avanzadas
            features = self._create_advanced_features(landmarks_array)
            features = features.reshape(1, -1)
            
            # Escalar características
            features_scaled = self.scalers['standard'].transform(features)
            
            # Realizar predicción con ensemble
            prediction_proba = self.ensemble_model.predict_proba(features_scaled)[0]
            predicted_class_idx = np.argmax(prediction_proba)
            confidence = prediction_proba[predicted_class_idx]
            
            # Obtener letra predicha
            predicted_letter = self.letters[predicted_class_idx]
            
            # Aplicar umbral de confianza
            if confidence < confidence_threshold:
                predicted_letter = "?"
                confidence = 0.0
            
            processing_time = (time.time() - start_time) * 1000
            
            return PredictionResponse(
                predicted_word=predicted_letter,
                confidence=float(confidence),
                processing_time_ms=processing_time,
                model_version=self.model_version
            )
            
        except Exception as e:
            logger.error(f"Error en predicción: {e}")
            return PredictionResponse(
                predicted_word="?",
                confidence=0.0,
                processing_time_ms=(time.time() - start_time) * 1000,
                model_version=self.model_version
            )
    
    def _save_models(self):
        """Guardar modelos entrenados"""
        os.makedirs("models", exist_ok=True)
        
        # Guardar ensemble principal
        joblib.dump(self.ensemble_model, "models/advanced_ensemble.pkl")
        joblib.dump(self.scalers['standard'], "models/advanced_scaler.pkl")
        
        # Guardar modelos individuales
        for name, model in self.models.items():
            joblib.dump(model, f"models/advanced_{name}.pkl")
        
        logger.info("Modelos avanzados guardados exitosamente")
    
    # Métodos para generar variaciones específicas de letras mejoradas
    def _enhanced_letter_A(self, landmarks, variation):
        """Letra A mejorada - Pulgar extendido, otros cerrados"""
        landmarks[4] = [0.8, 0.3, 0.0]  # Thumb tip extendido
        landmarks[3] = [0.7, 0.3, 0.0]  # Thumb IP
        landmarks[2] = [0.6, 0.3, 0.0]  # Thumb MCP
        
        # Otros dedos cerrados con variaciones
        for i in range(5, 21):
            if i % 4 == 0:  # Tips
                landmarks[i] = [landmarks[i-3][0], landmarks[i-3][1] + 0.05, 0.0]
            else:
                landmarks[i] = [landmarks[i-1][0], landmarks[i-1][1] + 0.05, 0.0]
        
        return landmarks
    
    def _enhanced_letter_B(self, landmarks, variation):
        """Letra B mejorada - Todos los dedos extendidos"""
        # Todos los dedos extendidos con variaciones
        finger_tips = [4, 8, 12, 16, 20]
        for tip in finger_tips:
            base_idx = tip - 3
            landmarks[tip] = [landmarks[base_idx][0], landmarks[base_idx][1] - 0.3, 0.0]
            landmarks[tip-1] = [landmarks[base_idx][0], landmarks[base_idx][1] - 0.2, 0.0]
            landmarks[tip-2] = [landmarks[base_idx][0], landmarks[base_idx][1] - 0.1, 0.0]
        
        return landmarks
    
    def _enhanced_letter_C(self, landmarks, variation):
        """Letra C mejorada - Forma de C"""
        # Crear forma de C con todos los dedos
        for i in range(4, 21):
            angle = (i - 4) * 0.1 + variation * 0.05
            radius = 0.15 + variation * 0.02
            landmarks[i] = [
                0.5 + radius * np.cos(angle),
                0.4 + radius * np.sin(angle),
                0.0
            ]
        return landmarks
    
    def _enhanced_letter_D(self, landmarks, variation):
        """Letra D mejorada - Índice extendido"""
        # Índice extendido
        landmarks[8] = [0.4, 0.0, 0.0]  # Index tip
        landmarks[7] = [0.4, 0.1, 0.0]  # Index DIP
        landmarks[6] = [0.4, 0.2, 0.0]  # Index PIP
        
        # Otros dedos cerrados
        for i in [4, 12, 16, 20]:  # Tips de otros dedos
            landmarks[i] = [landmarks[i-3][0], landmarks[i-3][1] + 0.05, 0.0]
        
        return landmarks
    
    def _enhanced_letter_E(self, landmarks, variation):
        """Letra E mejorada - Mano cerrada"""
        # Todos los dedos cerrados
        for i in range(4, 21):
            landmarks[i] = [landmarks[i-3][0], landmarks[i-3][1] + 0.05, 0.0]
        return landmarks
    
    def _enhanced_letter_F(self, landmarks, variation):
        """Letra F mejorada - Índice y pulgar"""
        # Pulgar e índice extendidos
        landmarks[4] = [0.8, 0.3, 0.0]  # Thumb tip
        landmarks[8] = [0.4, 0.0, 0.0]  # Index tip
        
        # Otros dedos cerrados
        for i in [12, 16, 20]:
            landmarks[i] = [landmarks[i-3][0], landmarks[i-3][1] + 0.05, 0.0]
        
        return landmarks
    
    def _enhanced_letter_G(self, landmarks, variation):
        """Letra G mejorada - Forma de pistola"""
        # Índice extendido, pulgar hacia arriba
        landmarks[8] = [0.4, 0.0, 0.0]  # Index tip
        landmarks[4] = [0.6, 0.2, 0.0]  # Thumb tip
        
        # Otros dedos cerrados
        for i in [12, 16, 20]:
            landmarks[i] = [landmarks[i-3][0], landmarks[i-3][1] + 0.05, 0.0]
        
        return landmarks
    
    def _enhanced_letter_H(self, landmarks, variation):
        """Letra H mejorada - Índice y medio"""
        # Índice y medio extendidos
        landmarks[8] = [0.4, 0.0, 0.0]  # Index tip
        landmarks[12] = [0.3, 0.0, 0.0]  # Middle tip
        
        # Otros dedos cerrados
        for i in [4, 16, 20]:
            landmarks[i] = [landmarks[i-3][0], landmarks[i-3][1] + 0.05, 0.0]
        
        return landmarks
    
    def _enhanced_letter_I(self, landmarks, variation):
        """Letra I mejorada - Meñique extendido"""
        # Meñique extendido
        landmarks[20] = [0.1, 0.0, 0.0]  # Pinky tip
        
        # Otros dedos cerrados
        for i in [4, 8, 12, 16]:
            landmarks[i] = [landmarks[i-3][0], landmarks[i-3][1] + 0.05, 0.0]
        
        return landmarks
    
    def _enhanced_letter_J(self, landmarks, variation):
        """Letra J mejorada - Meñique con movimiento"""
        # Meñique extendido con curva
        landmarks[20] = [0.1, 0.0, 0.0]  # Pinky tip
        landmarks[19] = [0.1, 0.1, 0.0]  # Pinky DIP
        landmarks[18] = [0.1, 0.2, 0.0]  # Pinky PIP
        
        # Otros dedos cerrados
        for i in [4, 8, 12, 16]:
            landmarks[i] = [landmarks[i-3][0], landmarks[i-3][1] + 0.05, 0.0]
        
        return landmarks
    
    def _enhanced_letter_K(self, landmarks, variation):
        """Letra K mejorada - Tres dedos extendidos"""
        # Índice, medio y anular extendidos
        landmarks[8] = [0.4, 0.0, 0.0]  # Index tip
        landmarks[12] = [0.3, 0.0, 0.0]  # Middle tip
        landmarks[16] = [0.2, 0.0, 0.0]  # Ring tip
        
        # Pulgar y meñique cerrados
        landmarks[4] = [0.6, 0.3, 0.0]  # Thumb tip
        landmarks[20] = [0.1, 0.2, 0.0]  # Pinky tip
        
        return landmarks
    
    def _enhanced_letter_L(self, landmarks, variation):
        """Letra L mejorada - Forma de L"""
        # Índice y pulgar en L
        landmarks[8] = [0.4, 0.0, 0.0]  # Index tip
        landmarks[4] = [0.6, 0.2, 0.0]  # Thumb tip
        
        # Otros dedos cerrados
        for i in [12, 16, 20]:
            landmarks[i] = [landmarks[i-3][0], landmarks[i-3][1] + 0.05, 0.0]
        
        return landmarks
    
    def _enhanced_letter_M(self, landmarks, variation):
        """Letra M mejorada - Tres dedos extendidos"""
        # Índice, medio y anular extendidos
        landmarks[8] = [0.4, 0.0, 0.0]  # Index tip
        landmarks[12] = [0.3, 0.0, 0.0]  # Middle tip
        landmarks[16] = [0.2, 0.0, 0.0]  # Ring tip
        
        # Pulgar y meñique cerrados
        landmarks[4] = [0.6, 0.3, 0.0]  # Thumb tip
        landmarks[20] = [0.1, 0.2, 0.0]  # Pinky tip
        
        return landmarks
    
    def _enhanced_letter_N(self, landmarks, variation):
        """Letra N mejorada - Índice y medio"""
        # Índice y medio extendidos
        landmarks[8] = [0.4, 0.0, 0.0]  # Index tip
        landmarks[12] = [0.3, 0.0, 0.0]  # Middle tip
        
        # Otros dedos cerrados
        for i in [4, 16, 20]:
            landmarks[i] = [landmarks[i-3][0], landmarks[i-3][1] + 0.05, 0.0]
        
        return landmarks
    
    def _enhanced_letter_O(self, landmarks, variation):
        """Letra O mejorada - Forma de O"""
        # Crear forma de O con todos los dedos
        for i in range(4, 21):
            angle = (i - 4) * 0.2 + variation * 0.05
            radius = 0.1 + variation * 0.02
            landmarks[i] = [
                0.5 + radius * np.cos(angle),
                0.4 + radius * np.sin(angle),
                0.0
            ]
        return landmarks
    
    def _enhanced_letter_P(self, landmarks, variation):
        """Letra P mejorada - Índice y medio hacia abajo"""
        # Índice y medio hacia abajo
        landmarks[8] = [0.4, 0.1, 0.0]  # Index tip
        landmarks[12] = [0.3, 0.1, 0.0]  # Middle tip
        
        # Otros dedos cerrados
        for i in [4, 16, 20]:
            landmarks[i] = [landmarks[i-3][0], landmarks[i-3][1] + 0.05, 0.0]
        
        return landmarks
    
    def _enhanced_letter_Q(self, landmarks, variation):
        """Letra Q mejorada - Forma de Q"""
        # Forma de Q con pulgar e índice
        landmarks[4] = [0.6, 0.2, 0.0]  # Thumb tip
        landmarks[8] = [0.4, 0.0, 0.0]  # Index tip
        
        # Otros dedos cerrados
        for i in [12, 16, 20]:
            landmarks[i] = [landmarks[i-3][0], landmarks[i-3][1] + 0.05, 0.0]
        
        return landmarks
    
    def _enhanced_letter_R(self, landmarks, variation):
        """Letra R mejorada - Dedos cruzados"""
        # Índice y medio cruzados
        landmarks[8] = [0.3, 0.0, 0.0]  # Index tip
        landmarks[12] = [0.4, 0.0, 0.0]  # Middle tip
        
        # Otros dedos cerrados
        for i in [4, 16, 20]:
            landmarks[i] = [landmarks[i-3][0], landmarks[i-3][1] + 0.05, 0.0]
        
        return landmarks
    
    def _enhanced_letter_S(self, landmarks, variation):
        """Letra S mejorada - Puño cerrado"""
        # Todos los dedos cerrados
        for i in range(4, 21):
            landmarks[i] = [landmarks[i-3][0], landmarks[i-3][1] + 0.05, 0.0]
        return landmarks
    
    def _enhanced_letter_T(self, landmarks, variation):
        """Letra T mejorada - Pulgar entre dedos"""
        # Pulgar entre índice y medio
        landmarks[4] = [0.35, 0.1, 0.0]  # Thumb tip
        
        # Otros dedos cerrados
        for i in [8, 12, 16, 20]:
            landmarks[i] = [landmarks[i-3][0], landmarks[i-3][1] + 0.05, 0.0]
        
        return landmarks
    
    def _enhanced_letter_U(self, landmarks, variation):
        """Letra U mejorada - Índice y medio juntos"""
        # Índice y medio juntos
        landmarks[8] = [0.4, 0.0, 0.0]  # Index tip
        landmarks[12] = [0.35, 0.0, 0.0]  # Middle tip
        
        # Otros dedos cerrados
        for i in [4, 16, 20]:
            landmarks[i] = [landmarks[i-3][0], landmarks[i-3][1] + 0.05, 0.0]
        
        return landmarks
    
    def _enhanced_letter_V(self, landmarks, variation):
        """Letra V mejorada - Índice y medio separados"""
        # Índice y medio separados
        landmarks[8] = [0.4, 0.0, 0.0]  # Index tip
        landmarks[12] = [0.3, 0.0, 0.0]  # Middle tip
        
        # Otros dedos cerrados
        for i in [4, 16, 20]:
            landmarks[i] = [landmarks[i-3][0], landmarks[i-3][1] + 0.05, 0.0]
        
        return landmarks
    
    def _enhanced_letter_W(self, landmarks, variation):
        """Letra W mejorada - Tres dedos extendidos"""
        # Índice, medio y anular extendidos
        landmarks[8] = [0.4, 0.0, 0.0]  # Index tip
        landmarks[12] = [0.3, 0.0, 0.0]  # Middle tip
        landmarks[16] = [0.2, 0.0, 0.0]  # Ring tip
        
        # Pulgar y meñique cerrados
        landmarks[4] = [0.6, 0.3, 0.0]  # Thumb tip
        landmarks[20] = [0.1, 0.2, 0.0]  # Pinky tip
        
        return landmarks
    
    def _enhanced_letter_X(self, landmarks, variation):
        """Letra X mejorada - Índice doblado"""
        # Índice doblado
        landmarks[8] = [0.4, 0.15, 0.0]  # Index tip doblado
        landmarks[7] = [0.4, 0.2, 0.0]  # Index DIP
        landmarks[6] = [0.4, 0.25, 0.0]  # Index PIP
        
        # Otros dedos cerrados
        for i in [4, 12, 16, 20]:
            landmarks[i] = [landmarks[i-3][0], landmarks[i-3][1] + 0.05, 0.0]
        
        return landmarks
    
    def _enhanced_letter_Y(self, landmarks, variation):
        """Letra Y mejorada - Meñique y pulgar"""
        # Meñique y pulgar extendidos
        landmarks[20] = [0.1, 0.0, 0.0]  # Pinky tip
        landmarks[4] = [0.8, 0.3, 0.0]  # Thumb tip
        
        # Otros dedos cerrados
        for i in [8, 12, 16]:
            landmarks[i] = [landmarks[i-3][0], landmarks[i-3][1] + 0.05, 0.0]
        
        return landmarks
    
    def _enhanced_letter_Z(self, landmarks, variation):
        """Letra Z mejorada - Índice con movimiento"""
        # Índice extendido con curva
        landmarks[8] = [0.4, 0.0, 0.0]  # Index tip
        landmarks[7] = [0.4, 0.1, 0.0]  # Index DIP
        landmarks[6] = [0.4, 0.2, 0.0]  # Index PIP
        
        # Otros dedos cerrados
        for i in [4, 12, 16, 20]:
            landmarks[i] = [landmarks[i-3][0], landmarks[i-3][1] + 0.05, 0.0]
        
        return landmarks


# Instancia global del predictor avanzado
advanced_letter_predictor = AdvancedLetterPredictor()
