import axios from 'axios'

// Configuración de la API
const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000'

// Crear instancia de axios
const api = axios.create({
  baseURL: API_BASE_URL,
  timeout: 30000, // 30 segundos para evitar timeouts
  headers: {
    'Content-Type': 'application/json',
  },
})

// Interceptor para manejar errores
api.interceptors.response.use(
  (response) => response,
  (error) => {
    console.error('Error de API:', error)

    if (error.code === 'ECONNABORTED') {
      throw new Error('Tiempo de espera agotado. Verifica tu conexión.')
    }

    if (error.response?.status === 500) {
      throw new Error('Error interno del servidor. Intenta nuevamente.')
    }

    if (error.response?.status === 400) {
      throw new Error('Datos inválidos enviados al servidor.')
    }

    if (!error.response) {
      throw new Error('No se pudo conectar con el servidor. Verifica que esté ejecutándose.')
    }

    throw error
  }
)

/**
 * Predecir palabra de lenguaje de señas basada en landmarks de la mano
 * @param {Array} landmarks - Array de 21 puntos con coordenadas x, y, z
 * @param {number} confidenceThreshold - Umbral mínimo de confianza (0-1)
 * @returns {Promise<Object>} Resultado de la predicción
 */
export const predictSignLanguage = async (landmarks, confidenceThreshold = 0.5) => {
  try {
    // Validar que se proporcionen exactamente 21 puntos
    if (!landmarks || landmarks.length !== 21) {
      throw new Error('Debe proporcionar exactamente 21 puntos de la mano')
    }

    // Validar estructura de los landmarks
    const validLandmarks = landmarks.map((point, index) => {
      if (!point || typeof point.x !== 'number' || typeof point.y !== 'number') {
        throw new Error(`Punto ${index} inválido: debe tener coordenadas x, y válidas`)
      }

      return {
        x: point.x,
        y: point.y,
        z: point.z || 0
      }
    })

    const requestData = {
      hand_landmarks: {
        landmarks: validLandmarks
      },
      confidence_threshold: confidenceThreshold
    }

    const response = await api.post('/api/v1/predict', requestData)

    return response.data
  } catch (error) {
    console.error('Error en predictSignLanguage:', error)
    throw error
  }
}

/**
 * Obtener el vocabulario disponible
 * @returns {Promise<Object>} Vocabulario y metadatos
 */
export const getVocabulary = async () => {
  try {
    const response = await api.get('/api/v1/vocabulary')
    return response.data
  } catch (error) {
    console.error('Error obteniendo vocabulario:', error)
    throw error
  }
}

/**
 * Agregar una nueva palabra al vocabulario
 * @param {string} word - Palabra a agregar
 * @returns {Promise<Object>} Resultado de la operación
 */
export const addWordToVocabulary = async (word) => {
  try {
    if (!word || typeof word !== 'string' || !word.trim()) {
      throw new Error('La palabra no puede estar vacía')
    }

    const response = await api.post('/api/v1/vocabulary', null, {
      params: { word: word.trim().toLowerCase() }
    })

    return response.data
  } catch (error) {
    console.error('Error agregando palabra:', error)
    throw error
  }
}

/**
 * Obtener información del modelo
 * @returns {Promise<Object>} Información del modelo
 */
export const getModelInfo = async () => {
  try {
    const response = await api.get('/api/v1/model/info')
    return response.data
  } catch (error) {
    console.error('Error obteniendo información del modelo:', error)
    throw error
  }
}

/**
 * Predicción de palabra (secuencia de frames de 21 puntos)
 * @param {Array<Array<{x:number,y:number,z:number}>>} frames - Lista de frames
 * @param {number} confidenceThreshold - Umbral de confianza
 */
export const predictWord = async (frames, confidenceThreshold = 0.5) => {
  try {
    if (!Array.isArray(frames) || frames.length === 0) {
      throw new Error('Debe proporcionar al menos 1 frame')
    }

    const normalizedFrames = frames.map((landmarks, frameIdx) => {
      if (!Array.isArray(landmarks) || landmarks.length !== 21) {
        throw new Error(`Frame ${frameIdx} inválido: se requieren 21 puntos`)
      }
      return {
        landmarks: landmarks.map((p, i) => {
          if (typeof p?.x !== 'number' || typeof p?.y !== 'number') {
            throw new Error(`Punto ${i} inválido en frame ${frameIdx}`)
          }
          return { x: p.x, y: p.y, z: p.z || 0 }
        })
      }
    })

    const body = {
      sequence: { frames: normalizedFrames },
      confidence_threshold: confidenceThreshold,
    }

    const response = await api.post('/api/v1/word/predict', body)
    return response.data
  } catch (error) {
    console.error('Error en predictWord:', error)
    throw error
  }
}

export const getWordModelInfo = async () => {
  try {
    const response = await api.get('/api/v1/word/model/info')
    return response.data
  } catch (error) {
    console.error('Error obteniendo info del modelo de palabras:', error)
    throw error
  }
}

/**
 * Verificar el estado de salud de la API
 * @returns {Promise<Object>} Estado de la API
 */
export const checkApiHealth = async () => {
  try {
    const response = await api.get('/health')
    return response.data
  } catch (error) {
    console.error('Error verificando salud de la API:', error)
    throw error
  }
}

/**
 * Obtener información de la API raíz
 * @returns {Promise<Object>} Información de la API
 */
export const getApiInfo = async () => {
  try {
    const response = await api.get('/')
    return response.data
  } catch (error) {
    console.error('Error obteniendo información de la API:', error)
    throw error
  }
}

/**
 * Predecir letra del alfabeto de señas (alias para predictSignLanguage)
 * @param {Array} landmarks - Array de 21 puntos con coordenadas x, y, z
 * @param {number} confidenceThreshold - Umbral mínimo de confianza (0-1)
 * @returns {Promise<Object>} Resultado de la predicción
 */
export const predictLetter = async (landmarks, confidenceThreshold = 0.5) => {
  return predictSignLanguage(landmarks, confidenceThreshold)
}

/**
 * Obtener letras disponibles
 * @returns {Promise<Object>} Letras disponibles
 */
export const getLetters = async () => {
  try {
    const response = await api.get('/api/v1/letters')
    return response.data
  } catch (error) {
    console.error('Error obteniendo letras:', error)
    throw error
  }
}

export default api