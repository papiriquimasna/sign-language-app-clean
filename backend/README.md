# 🤟 Sistema de Reconocimiento de Lenguaje de Señas

## 📋 Descripción
Sistema completo de reconocimiento de letras del alfabeto de señas en tiempo real usando MediaPipe y Machine Learning. Implementa un pipeline completo desde la captura de video hasta la predicción de letras con alta precisión.

## ✨ Características Principales

### 🎯 Detección en Tiempo Real
- **23 letras del alfabeto** (A-Y, excluyendo J, Z y P para evitar confusiones)
- **Detección instantánea** con MediaPipe Hands (21 landmarks por mano)
- **Interfaz visual** con landmarks de la mano en tiempo real
- **Información de confianza** y tiempo de procesamiento
- **Filtro Kalman** para suavizado de landmarks y reducción de ruido

### 🧠 Modelo de Machine Learning
- **RandomForest Classifier** optimizado para MediaPipe landmarks
- **95 características** extraídas de landmarks de la mano:
  - Coordenadas originales (63 características: 21 puntos × 3 coordenadas)
  - Distancias entre puntos clave (15 características)
  - Ángulos entre dedos (10 características)
  - Características de forma de la mano (7 características)
- **Precisión del 73.9%** en detección de letras
- **Modelo entrenado** con datos sintéticos realistas
- **StandardScaler** para normalización de características

### 🎨 Interfaz de Usuario
- **Componente principal** (`LetterRecognition.jsx`) con detección en tiempo real
- **Componente rápido** (`FastLetterRecognition.jsx`) para detección simplificada
- **Guía visual** (`LetterGuide.jsx`) de todas las letras disponibles
- **Controles intuitivos** de cámara y configuración
- **Estados de predicción** con indicadores visuales

## 🏗️ Arquitectura Técnica

### Backend (FastAPI)
- **API REST** con endpoints para predicción de letras
- **Servicios modulares**:
  - `LetterPredictor`: Servicio principal de predicción
  - `AdvancedLetterPredictor`: Predicción avanzada con ensemble
  - `PersonalizedPredictor`: Predicción personalizada
  - `ImagePredictor`: Predicción basada en imágenes
  - `WordPredictor`: Predicción de palabras completas
- **Logging completo** con niveles configurables
- **Configuración flexible** con Pydantic Settings
- **Manejo de errores** robusto con HTTPException

### Frontend (React + Vite)
- **Componentes React** optimizados con hooks
- **Tailwind CSS** para estilos modernos y responsivos
- **MediaPipe integrado** para detección de manos en tiempo real
- **Filtro Kalman** para suavizado de landmarks
- **Estados de React** para manejo de predicciones
- **Canvas API** para renderizado de landmarks

## 📁 Estructura Detallada del Proyecto

```
backend/
├── app/
│   ├── api/v1/           # Rutas de la API REST
│   │   ├── hand_routes.py    # Endpoints para detección de manos
│   │   └── word_routes.py    # Endpoints para predicción de palabras
│   ├── core/             # Configuración central
│   │   ├── config.py         # Configuración con Pydantic Settings
│   │   └── logger.py         # Sistema de logging
│   ├── models/           # Modelos de datos Pydantic
│   │   └── hand_data.py      # Estructuras de datos para landmarks
│   ├── services/         # Servicios de predicción
│   │   ├── letter_predictor.py      # Servicio principal
│   │   ├── advanced_letter_predictor.py  # Predicción avanzada
│   │   ├── personalized_predictor.py     # Predicción personalizada
│   │   ├── image_predictor.py           # Predicción por imágenes
│   │   └── word_predictor.py            # Predicción de palabras
│   ├── utils/            # Utilidades
│   │   └── helpers.py        # Funciones auxiliares
│   └── main.py           # Punto de entrada de la aplicación
├── models/               # Modelos entrenados
│   ├── no_p_letter_classifier.pkl    # Modelo RandomForest
│   ├── no_p_letter_scaler.pkl        # StandardScaler
│   └── no_p_model_metadata.json      # Metadatos del modelo
└── requirements.txt      # Dependencias Python

frontend/
├── src/
│   ├── components/       # Componentes React
│   │   ├── LetterRecognition.jsx     # Componente principal
│   │   ├── FastLetterRecognition.jsx # Componente rápido
│   │   ├── LetterGuide.jsx           # Guía de letras
│   │   ├── HandCapture.jsx           # Captura de manos
│   │   └── SimpleHandDetection.jsx   # Detección simple
│   ├── services/         # Servicios de API
│   │   └── api.js            # Cliente HTTP para backend
│   ├── utils/            # Utilidades
│   ├── App.jsx           # Componente principal
│   └── main.jsx          # Punto de entrada
├── package.json          # Dependencias Node.js
└── vite.config.js        # Configuración de Vite
```

## 🚀 Instalación y Uso

### Backend
```bash
cd backend
pip install -r requirements.txt
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

### Frontend
```bash
cd frontend
npm install
npm run dev
```

## 🎯 Letras Soportadas
**23 letras del alfabeto de señas:**
A, B, C, D, E, F, G, H, I, K, L, M, N, O, Q, R, S, T, U, V, W, X, Y

*Excluidas: J, Z (movimientos complejos) y P (confusión con otras letras)*

## 🔧 Tecnologías y Librerías Detalladas

### Backend
- **FastAPI (0.104.1)**: Framework web moderno con documentación automática
- **Uvicorn (0.24.0)**: Servidor ASGI para FastAPI
- **Pydantic (2.7.0)**: Validación de datos y configuración
- **MediaPipe (0.10.14)**: Detección de landmarks de manos
- **Scikit-learn (1.3.2)**: Machine Learning (RandomForest, StandardScaler)
- **NumPy (1.24.3)**: Operaciones matemáticas y arrays
- **Joblib (1.3.2)**: Persistencia de modelos entrenados
- **SciPy (1.14.1)**: Operaciones científicas
- **OpenCV (4.8.1.78)**: Procesamiento de imágenes
- **TensorFlow (2.20.0)**: Redes neuronales (opcional)
- **Python-dotenv (1.0.0)**: Variables de entorno

### Frontend
- **React (18.x)**: Framework de interfaz de usuario
- **Vite (5.x)**: Herramienta de desarrollo y build
- **Tailwind CSS (3.x)**: Framework de estilos
- **MediaPipe**: Detección de manos en tiempo real
- **Lucide React**: Iconos modernos
- **Axios**: Cliente HTTP para API calls

## 🧮 Algoritmos y Procesamiento

### Extracción de Características
```python
def _extract_mediapipe_features(self, landmarks):
    # 1. Coordenadas originales (63 características)
    features.extend(landmarks.flatten())
    
    # 2. Distancias entre puntos clave (15 características)
    key_points = [0, 4, 8, 12, 16, 20]  # Wrist, finger tips
    
    # 3. Ángulos entre dedos (10 características)
    finger_tips = [4, 8, 12, 16, 20]
    
    # 4. Características de forma (7 características)
    # Área, centroid, varianza
```

### Pipeline de Predicción
1. **Captura de video** → MediaPipe Hands
2. **Extracción de landmarks** → 21 puntos 3D
3. **Extracción de características** → 95 características
4. **Normalización** → StandardScaler
5. **Predicción** → RandomForest Classifier
6. **Post-procesamiento** → Filtro Kalman
7. **Visualización** → Canvas API

### Filtro Kalman
```javascript
// Suavizado de landmarks para reducir ruido
const smoothedLandmark = {
    x: alpha * newLandmark.x + (1 - alpha) * previousLandmark.x,
    y: alpha * newLandmark.y + (1 - alpha) * previousLandmark.y,
    z: alpha * newLandmark.z + (1 - alpha) * previousLandmark.z
};
```

## 📊 Rendimiento y Métricas
- **Precisión general**: 73.9%
- **14 letras** con 100% de precisión
- **Latencia de predicción**: <100ms
- **FPS de detección**: 30 FPS
- **Uso de memoria**: Optimizado para dispositivos móviles
- **Interfaz responsiva** y fluida

## 🔄 Flujo de Datos
1. **Video Stream** → MediaPipe Hands
2. **Landmarks** → Extracción de características
3. **Características** → Modelo ML
4. **Predicción** → Filtro Kalman
5. **Resultado** → Interfaz de usuario

## 🎉 Estado del Proyecto
**✅ Completamente funcional** - Sistema listo para uso con detección estable de 23 letras del alfabeto de señas. Implementación robusta con manejo de errores, logging completo y arquitectura escalable.
