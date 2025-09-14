"""
Utilidades y funciones auxiliares
"""
import numpy as np
from typing import List, Tuple
from app.models.hand_data import HandLandmarks, HandPoint


def normalize_landmarks(landmarks: HandLandmarks) -> HandLandmarks:
    """
    Normalizar los landmarks de la mano para mejorar la precisión del modelo.
    
    Args:
        landmarks: Landmarks originales de la mano
        
    Returns:
        HandLandmarks: Landmarks normalizados
    """
    # Convertir a array de NumPy
    points = landmarks.to_numpy()
    
    # Calcular el centro de la palma (punto de referencia)
    palm_center = points[9]  # Centro de la palma según MediaPipe
    
    # Normalizar respecto al centro de la palma
    normalized_points = points - palm_center
    
    # Calcular la escala basada en la distancia promedio de los dedos
    finger_tips = normalized_points[[4, 8, 12, 16, 20]]
    distances = np.linalg.norm(finger_tips, axis=1)
    scale = np.mean(distances)
    
    # Evitar división por cero
    if scale > 0:
        normalized_points = normalized_points / scale
    
    # Convertir de vuelta a HandLandmarks
    normalized_landmarks = []
    for point in normalized_points:
        normalized_landmarks.append(HandPoint(x=point[0], y=point[1], z=point[2]))
    
    return HandLandmarks(landmarks=normalized_landmarks)


def calculate_hand_features(landmarks: HandLandmarks) -> dict:
    """
    Calcular características adicionales de la mano para mejorar la predicción.
    
    Args:
        landmarks: Landmarks de la mano
        
    Returns:
        dict: Diccionario con características calculadas
    """
    points = landmarks.to_numpy()
    
    features = {}
    
    # Distancias entre puntos clave
    palm_center = points[9]
    finger_tips = points[[4, 8, 12, 16, 20]]
    
    # Distancia promedio de las puntas de los dedos al centro de la palma
    distances = np.linalg.norm(finger_tips - palm_center, axis=1)
    features['avg_finger_distance'] = float(np.mean(distances))
    features['finger_distance_std'] = float(np.std(distances))
    
    # Ángulos entre dedos
    thumb_tip = points[4]
    index_tip = points[8]
    middle_tip = points[12]
    
    # Ángulo entre pulgar e índice
    thumb_to_palm = palm_center - thumb_tip
    index_to_palm = palm_center - index_tip
    
    if np.linalg.norm(thumb_to_palm) > 0 and np.linalg.norm(index_to_palm) > 0:
        cos_angle = np.dot(thumb_to_palm, index_to_palm) / (
            np.linalg.norm(thumb_to_palm) * np.linalg.norm(index_to_palm)
        )
        cos_angle = np.clip(cos_angle, -1, 1)  # Evitar errores de precisión
        features['thumb_index_angle'] = float(np.arccos(cos_angle))
    
    # Extensión de los dedos (distancia de la punta a la base)
    finger_bases = points[[3, 6, 10, 14, 18]]  # Bases de los dedos
    
    finger_extensions = []
    for i in range(5):
        extension = np.linalg.norm(finger_tips[i] - finger_bases[i])
        finger_extensions.append(float(extension))
    
    features['finger_extensions'] = finger_extensions
    features['avg_finger_extension'] = float(np.mean(finger_extensions))
    
    return features


def validate_hand_landmarks(landmarks: HandLandmarks) -> bool:
    """
    Validar que los landmarks de la mano sean realistas.
    
    Args:
        landmarks: Landmarks a validar
        
    Returns:
        bool: True si los landmarks son válidos
    """
    try:
        points = landmarks.to_numpy()
        
        # Verificar que no haya valores NaN o infinitos
        if not np.all(np.isfinite(points)):
            return False
        
        # Verificar que las coordenadas estén en un rango razonable
        # MediaPipe típicamente devuelve valores entre -1 y 1
        if np.any(points < -2) or np.any(points > 2):
            return False
        
        # Verificar que la mano no esté completamente colapsada
        palm_center = points[9]
        finger_tips = points[[4, 8, 12, 16, 20]]
        distances = np.linalg.norm(finger_tips - palm_center, axis=1)
        
        # Al menos un dedo debe estar extendido
        if np.all(distances < 0.01):
            return False
        
        return True
        
    except Exception:
        return False


def smooth_landmarks(landmarks_list: List[HandLandmarks], window_size: int = 3) -> HandLandmarks:
    """
    Suavizar una secuencia de landmarks usando promedio móvil.
    
    Args:
        landmarks_list: Lista de landmarks a suavizar
        window_size: Tamaño de la ventana para el promedio móvil
        
    Returns:
        HandLandmarks: Landmarks suavizados
    """
    if len(landmarks_list) == 0:
        raise ValueError("La lista de landmarks no puede estar vacía")
    
    if len(landmarks_list) == 1:
        return landmarks_list[0]
    
    # Tomar los últimos landmarks para el suavizado
    recent_landmarks = landmarks_list[-window_size:]
    
    # Convertir a arrays de NumPy
    arrays = [landmarks.to_numpy() for landmarks in recent_landmarks]
    
    # Calcular promedio
    smoothed_array = np.mean(arrays, axis=0)
    
    # Convertir de vuelta a HandLandmarks
    smoothed_landmarks = []
    for point in smoothed_array:
        smoothed_landmarks.append(HandPoint(x=point[0], y=point[1], z=point[2]))
    
    return HandLandmarks(landmarks=smoothed_landmarks)


