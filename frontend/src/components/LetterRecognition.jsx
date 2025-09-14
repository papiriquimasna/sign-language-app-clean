import React, { useRef, useEffect, useState, useCallback } from 'react'
import { Hands } from '@mediapipe/hands'
import { Camera } from '@mediapipe/camera_utils'
import { predictLetter } from '../services/api'
import { Play, Pause, RotateCcw, AlertCircle, Type, Settings, Eye, EyeOff } from 'lucide-react'

/**
 * Componente de reconocimiento de letras en tiempo real
 * Integra MediaPipe Hands con el modelo de ML del backend
 * para detectar y mostrar letras del alfabeto de señas A-Z
 */

const LetterRecognition = ({ onLetterDetected }) => {
  const videoRef = useRef(null)
  const canvasRef = useRef(null)
  const handsRef = useRef(null)
  const cameraRef = useRef(null)

  // Estados principales
  const [isDetecting, setIsDetecting] = useState(false)
  const [isCameraOn, setIsCameraOn] = useState(false)
  const [prediction, setPrediction] = useState(null)
  const [error, setError] = useState(null)
  const [fps, setFps] = useState(0)
  const [lastPredictionTime, setLastPredictionTime] = useState(0)
  const [detectedLetters, setDetectedLetters] = useState([])
  const [history, setHistory] = useState([]) // [{letter, confidence, ts}]
  const [showLandmarks, setShowLandmarks] = useState(true)
  const [showSettings, setShowSettings] = useState(false)
  const [confidenceThreshold, setConfidenceThreshold] = useState(0.1)

  // Estados para predicción mejorada
  const [predictionHistory, setPredictionHistory] = useState([])
  const [stablePrediction, setStablePrediction] = useState(null)
  const [predictionConfidence, setPredictionConfidence] = useState(0)
  const [isGestureStable, setIsGestureStable] = useState(false)
  const [lastStableTime, setLastStableTime] = useState(0)
  const [consecutivePredictions, setConsecutivePredictions] = useState(0)

  // Configuración optimizada para mejor detección
  const predictionInterval = 100 // Frecuencia aún más rápida para mejor responsividad
  const smoothWindow = 3 // Ventana más pequeña para respuesta más rápida
  const stableMinCount = 1 // Menos predicciones para detección más rápida
  const [adaptiveThreshold, setAdaptiveThreshold] = useState(0.03) // Umbral aún más bajo

  // Sistema de filtrado avanzado para landmarks
  const [landmarkHistory, setLandmarkHistory] = useState([])
  const [smoothedLandmarks, setSmoothedLandmarks] = useState(null)
  const [gestureStability, setGestureStability] = useState(0)
  const [lightingQuality, setLightingQuality] = useState(1.0)

  /**
   * Inicializar MediaPipe Hands con configuración optimizada
   * para detección en tiempo real de alta precisión
   */
  const initializeHands = useCallback(() => {
    if (handsRef.current) {
      console.log('⚠️ MediaPipe ya está inicializado, evitando reinicialización')
      return
    }

    console.log('🤖 Inicializando MediaPipe Hands...')

    const hands = new Hands({
      locateFile: (file) => {
        return `https://cdn.jsdelivr.net/npm/@mediapipe/hands/${file}`
      }
    })

    // Configuración optimizada para mejor detección
    hands.setOptions({
      maxNumHands: 1,
      modelComplexity: 1, // Mayor complejidad para mejor precisión
      minDetectionConfidence: 0.2, // Aún menos estricto para detectar más letras
      minTrackingConfidence: 0.2 // Mejor seguimiento
    })

    hands.onResults(onResults)
    handsRef.current = hands
    console.log('✅ MediaPipe Hands inicializado correctamente')
  }, [])

  /**
   * Procesar resultados de MediaPipe Hands
   * Optimizado para rendimiento en tiempo real
   */
  const onResults = useCallback((results) => {
    const canvas = canvasRef.current
    const video = videoRef.current

    if (!canvas || !video) return

    const ctx = canvas.getContext('2d')
    canvas.width = video.videoWidth
    canvas.height = video.videoHeight

    // Limpiar y dibujar frame actual
    ctx.save()
    ctx.clearRect(0, 0, canvas.width, canvas.height)
    ctx.drawImage(results.image, 0, 0, canvas.width, canvas.height)

    // Verificar si se detectó una mano
    if (results.multiHandLandmarks && results.multiHandLandmarks.length > 0) {
      const landmarks = results.multiHandLandmarks[0]
      console.log('✋ Mano detectada! Landmarks:', landmarks.length)

      // Dibujar landmarks si está habilitado
      if (showLandmarks) {
        drawHandLandmarks(ctx, landmarks)
      }

      // SOLUCIÓN DEFINITIVA: Realizar predicción directamente sin depender del estado
      console.log('🔍 Realizando predicción directa...')

      if (Date.now() - lastPredictionTime > predictionInterval) {
        console.log('🚀 Iniciando predicción...')
        performPrediction(landmarks)
        setLastPredictionTime(Date.now())
      } else {
        console.log('⏰ Esperando intervalo de predicción')
      }
    } else {
      console.log('❌ No se detectaron manos')
    }

    // Dibujar letra detectada sobre el video (si la hay) - Versión mejorada
    if (stablePrediction && stablePrediction.predicted_word && stablePrediction.predicted_word !== '?') {
      const text = `${stablePrediction.predicted_word}`
      const x = canvas.width / 2
      const y = 100

      // Fondo con gradiente para mejor legibilidad
      const gradient = ctx.createLinearGradient(x - 80, y - 80, x + 80, y + 80)
      gradient.addColorStop(0, 'rgba(16, 185, 129, 0.9)')
      gradient.addColorStop(1, 'rgba(5, 150, 105, 0.9)')

      ctx.fillStyle = gradient
      ctx.fillRect(x - 80, y - 80, 160, 100)

      // Borde
      ctx.strokeStyle = '#ffffff'
      ctx.lineWidth = 3
      ctx.strokeRect(x - 80, y - 80, 160, 100)

      // Texto de la letra principal
      ctx.fillStyle = '#ffffff'
      ctx.font = 'bold 64px Arial'
      ctx.textAlign = 'center'
      ctx.shadowColor = 'rgba(0, 0, 0, 0.5)'
      ctx.shadowBlur = 4
      ctx.fillText(text, x, y - 10)

      // Información de confianza
      if (stablePrediction.confidence) {
        ctx.fillStyle = '#ffffff'
        ctx.font = '16px Arial'
        ctx.fillText(`${Math.round(stablePrediction.confidence * 100)}%`, x, y + 20)
      }

      // Indicador de estabilidad
      if (isGestureStable) {
        ctx.fillStyle = '#22c55e'
        ctx.font = '14px Arial'
        ctx.fillText('ESTABLE', x, y + 40)
      }

      // Resetear sombra
      ctx.shadowBlur = 0
    } else if (prediction && prediction.predicted_word && prediction.predicted_word !== '?') {
      // Indicador de predicción en proceso
      const text = `${prediction.predicted_word}`
      const x = canvas.width / 2
      const y = 100

      // Fondo amarillo para indicar procesamiento
      ctx.fillStyle = 'rgba(245, 158, 11, 0.8)'
      ctx.fillRect(x - 60, y - 40, 120, 60)

      // Borde
      ctx.strokeStyle = '#ffffff'
      ctx.lineWidth = 2
      ctx.strokeRect(x - 60, y - 40, 120, 60)

      // Texto
      ctx.fillStyle = '#ffffff'
      ctx.font = 'bold 36px Arial'
      ctx.textAlign = 'center'
      ctx.fillText(text, x, y)

      // Texto de procesamiento
      ctx.fillStyle = '#fef3c7'
      ctx.font = '12px Arial'
      ctx.fillText('ANALIZANDO...', x, y + 20)
    } else if (isDetecting) {
      // Mostrar estado de análisis
      ctx.font = 'bold 28px Arial'
      ctx.textAlign = 'center'
      ctx.fillStyle = 'rgba(255,255,255,0.9)'
      ctx.strokeStyle = 'rgba(0,0,0,0.8)'
      ctx.lineWidth = 4
      const statusText = results.multiHandLandmarks && results.multiHandLandmarks.length > 0
        ? 'Analizando...'
        : 'Mano no detectada'
      ctx.strokeText(statusText, canvas.width / 2, 60)
      ctx.fillText(statusText, canvas.width / 2, 60)
    }

    ctx.restore()
  }, [isDetecting, lastPredictionTime, predictionInterval, showLandmarks, prediction])

  /**
   * Dibujar landmarks de la mano en el canvas
   * Muestra los 21 puntos de MediaPipe con conexiones y colores distintivos
   */
  const drawHandLandmarks = (ctx, landmarks) => {
    // Definir conexiones entre puntos de la mano
    const connections = [
      [0, 1], [1, 2], [2, 3], [3, 4], // Pulgar
      [0, 5], [5, 6], [6, 7], [7, 8], // Índice
      [5, 9], [9, 10], [10, 11], [11, 12], // Medio
      [9, 13], [13, 14], [14, 15], [15, 16], // Anular
      [13, 17], [17, 18], [18, 19], [19, 20], // Meñique
      [0, 17] // Base de la palma
    ]

    // Dibujar conexiones entre puntos
    ctx.strokeStyle = '#00ff00' // Verde brillante
    ctx.lineWidth = 3
    connections.forEach(([start, end]) => {
      const startPoint = landmarks[start]
      const endPoint = landmarks[end]

      ctx.beginPath()
      ctx.moveTo(startPoint.x * canvasRef.current.width, startPoint.y * canvasRef.current.height)
      ctx.lineTo(endPoint.x * canvasRef.current.width, endPoint.y * canvasRef.current.height)
      ctx.stroke()
    })

    // Dibujar puntos individuales con colores distintivos
    landmarks.forEach((landmark, index) => {
      const x = landmark.x * canvasRef.current.width
      const y = landmark.y * canvasRef.current.height

      // Color según el tipo de punto
      ctx.fillStyle = getPointColor(index)
      ctx.beginPath()
      ctx.arc(x, y, 4, 0, 2 * Math.PI) // Puntos más pequeños (4px en lugar de 8px)
      ctx.fill()

      // Borde blanco para mejor visibilidad
      ctx.strokeStyle = '#ffffff'
      ctx.lineWidth = 1 // Borde más delgado
      ctx.stroke()

      // Número del punto para identificación (más pequeño)
      ctx.fillStyle = '#ffffff'
      ctx.font = 'bold 10px Arial' // Texto más pequeño
      ctx.textAlign = 'center'
      ctx.strokeStyle = '#000000'
      ctx.lineWidth = 1
      ctx.strokeText(index.toString(), x, y - 12)
      ctx.fillText(index.toString(), x, y - 12)
    })
  }

  /**
   * Obtener color distintivo para cada punto de la mano
   * Cada dedo tiene un color único para facilitar la identificación
   */
  const getPointColor = (index) => {
    if (index >= 0 && index <= 4) return '#ef4444' // Pulgar - Rojo
    if (index >= 5 && index <= 8) return '#f59e0b' // Índice - Naranja
    if (index >= 9 && index <= 12) return '#10b981' // Medio - Verde
    if (index >= 13 && index <= 16) return '#8b5cf6' // Anular - Morado
    if (index >= 17 && index <= 20) return '#ec4899' // Meñique - Rosa
    return '#6b7280' // Palma - Gris
  }

  /**
   * Filtro de Kalman simple para suavizar landmarks
   * Reduce el ruido y mejora la estabilidad de la detección
   */
  const applyKalmanFilter = (newLandmarks) => {
    if (!smoothedLandmarks) {
      setSmoothedLandmarks(newLandmarks)
      return newLandmarks
    }

    const alpha = 0.6 // Factor de suavizado aún menos restrictivo para mejor detección
    const smoothed = newLandmarks.map((landmark, index) => {
      const prev = smoothedLandmarks[index]
      return {
        x: alpha * landmark.x + (1 - alpha) * prev.x,
        y: alpha * landmark.y + (1 - alpha) * prev.y,
        z: alpha * landmark.z + (1 - alpha) * prev.z
      }
    })

    setSmoothedLandmarks(smoothed)
    return smoothed
  }

  /**
   * Analizar estabilidad del gesto basado en variación de landmarks
   * Calcula qué tan estable está la mano en la posición actual
   */
  const analyzeGestureStability = (landmarks) => {
    setLandmarkHistory(prev => {
      const next = [...prev, landmarks]
      if (next.length > 10) next.shift() // Mantener solo últimos 10 frames

      if (next.length < 3) return next

      // Calcular variación promedio de posición
      let totalVariation = 0
      let pointCount = 0

      for (let i = 0; i < landmarks.length; i++) {
        let pointVariation = 0
        for (let j = 1; j < next.length; j++) {
          const prev = next[j - 1][i]
          const curr = next[j][i]
          pointVariation += Math.sqrt(
            Math.pow(curr.x - prev.x, 2) +
            Math.pow(curr.y - prev.y, 2) +
            Math.pow(curr.z - prev.z, 2)
          )
        }
        totalVariation += pointVariation / (next.length - 1)
        pointCount++
      }

      const avgVariation = totalVariation / pointCount
      const stability = Math.max(0, 1 - avgVariation * 10) // Convertir a 0-1
      setGestureStability(stability)

      return next
    })
  }

  /**
   * Analizar calidad de iluminación basada en variación de landmarks
   * Detecta si hay suficiente luz y contraste para buena detección
   */
  const analyzeLightingQuality = (landmarks) => {
    // Calcular dispersión de landmarks (mayor dispersión = mejor iluminación)
    let minX = Infinity, maxX = -Infinity
    let minY = Infinity, maxY = -Infinity

    landmarks.forEach(landmark => {
      minX = Math.min(minX, landmark.x)
      maxX = Math.max(maxX, landmark.x)
      minY = Math.min(minY, landmark.y)
      maxY = Math.max(maxY, landmark.y)
    })

    const spread = Math.sqrt(Math.pow(maxX - minX, 2) + Math.pow(maxY - minY, 2))
    const quality = Math.min(1.0, spread * 2) // Normalizar a 0-1
    setLightingQuality(quality)
  }

  /**
   * Realizar predicción de letra basada en landmarks de la mano
   * Versión mejorada con predicción más exacta y estable
   */
  const performPrediction = async (landmarks) => {
    try {
      console.log('🔍 Iniciando predicción mejorada...', landmarks.length, 'landmarks')

      // Aplicar filtro de Kalman para suavizar landmarks
      const smoothed = applyKalmanFilter(landmarks)

      // Analizar estabilidad del gesto
      analyzeGestureStability(smoothed)

      // Analizar calidad de iluminación
      analyzeLightingQuality(smoothed)

      console.log('📊 Análisis de calidad:', {
        estabilidad: Math.round(gestureStability * 100) + '%',
        iluminación: Math.round(lightingQuality * 100) + '%'
      })

      // Convertir landmarks suavizados a formato requerido por el backend
      const handData = smoothed.map(landmark => ({
        x: landmark.x,
        y: landmark.y,
        z: landmark.z || 0
      }))

      console.log('📤 Enviando datos suavizados al backend:', handData.slice(0, 3), '...')

      // Realizar predicción con umbral muy bajo para mejor detección
      const raw = await predictLetter(handData, 0.0)

      console.log('📥 Respuesta del backend:', raw)

      const letter = raw?.predicted_word && raw.predicted_word !== '?' ? raw.predicted_word : '?'
      const baseConf = typeof raw?.confidence === 'number' ? raw.confidence : 0

      // Aplicar factores de calidad para ajustar confianza
      const stabilityFactor = Math.max(0.5, gestureStability)
      const lightingFactor = Math.max(0.7, lightingQuality)
      const adjustedConf = baseConf * stabilityFactor * lightingFactor

      console.log('🎯 Letra procesada:', letter, 'Confianza base:', baseConf, 'Ajustada:', adjustedConf)

      // Actualizar historial de predicciones para análisis de estabilidad
      setPredictionHistory(prev => {
        const next = [...prev, { letter, confidence: adjustedConf, ts: Date.now() }]
        if (next.length > 8) next.shift() // Ventana más pequeña para respuesta más rápida

        // Calcular predicción estable
        const valid = next.filter(x => x.letter !== '?')

        if (valid.length >= 1) { // Mostrar inmediatamente con solo 1 predicción válida
          const counts = {}
          valid.forEach(x => { counts[x.letter] = (counts[x.letter] || 0) + 1 })
          const best = Object.entries(counts).sort((a, b) => b[1] - a[1])[0]
          const bestLetter = best ? best[0] : null
          const avgConf = valid.reduce((s, x) => s + x.confidence, 0) / valid.length

          // ULTRA SIMPLIFICADO: Mostrar letra inmediatamente
          const isStable = best && best[1] >= 1 // Sin umbral de confianza para mejor detección

          if (isStable && bestLetter) {
            const stable = {
              predicted_word: bestLetter,
              confidence: avgConf,
              processing_time_ms: raw?.processing_time_ms || 0,
              model_version: raw?.model_version || 'improved',
              stability: gestureStability,
              lighting_quality: lightingQuality,
              consecutive_count: best[1]
            }

            console.log('🎉 Letra detectada:', stable)
            setStablePrediction(stable)
            setPrediction(stable)
            setPredictionConfidence(avgConf)
            setIsGestureStable(true)
            setLastStableTime(Date.now())
            setError(null)

            // Actualizar lista de letras detectadas
            if (!detectedLetters.includes(bestLetter)) {
              setDetectedLetters(prev => [...prev, bestLetter].slice(-10))
            }

            // Notificar al componente padre INMEDIATAMENTE
            if (onLetterDetected) {
              console.log('📢 Notificando al componente padre:', bestLetter, avgConf)
              onLetterDetected(bestLetter, avgConf)
            }
          } else {
            // Solo limpiar si no hay predicción por mucho tiempo
            if (Date.now() - lastStableTime > 2000) {
              setPrediction(null)
              setStablePrediction(null)
              setIsGestureStable(false)
            }
          }
        }

        return next
      })

    } catch (err) {
      console.error('💥 Error en predicción:', err)
      setError('Error al procesar la predicción: ' + err.message)
    }
  }

  /**
   * Iniciar cámara y configurar MediaPipe Hands
   * Optimizado para mejor rendimiento y manejo de errores
   */
  const startCamera = async () => {
    try {
      console.log('📹 Iniciando cámara...')
      setError(null)

      if (!videoRef.current) {
        console.log('❌ videoRef no disponible')
        setError('Error: Referencia de video no disponible')
        return
      }

      const camera = new Camera(videoRef.current, {
        onFrame: async () => {
          if (handsRef.current) {
            console.log('📸 Enviando frame a MediaPipe...')
            await handsRef.current.send({ image: videoRef.current })
          }
        },
        width: 640, // Resolución optimizada para balance rendimiento/precisión
        height: 480
      })

      await camera.start()
      cameraRef.current = camera
      setIsCameraOn(true)
      console.log('✅ Cámara iniciada correctamente')

      // Inicializar MediaPipe si no está inicializado
      if (!handsRef.current) {
        console.log('🤖 Inicializando MediaPipe...')
        initializeHands()
      } else {
        console.log('⚠️ MediaPipe ya está inicializado')
      }
    } catch (err) {
      console.error('💥 Error iniciando cámara:', err)
      setError('Error al acceder a la cámara. Verifica los permisos.')
    }
  }

  /**
   * Detener cámara y limpiar recursos
   */
  const stopCamera = () => {
    if (cameraRef.current) {
      cameraRef.current.stop()
      cameraRef.current = null
    }
    setIsCameraOn(false)
    setIsDetecting(false)
    setPrediction(null)
    setError(null)
  }

  // Monitorear cambios en el estado de detección
  useEffect(() => {
    console.log('🔄 Estado de detección actualizado:', isDetecting)
  }, [isDetecting])

  // Función global para activar detección desde la consola
  useEffect(() => {
    window.activateDetection = () => {
      console.log('🚨 ACTIVACIÓN MANUAL desde consola')
      setIsDetecting(true)
    }
    window.deactivateDetection = () => {
      console.log('🚨 DESACTIVACIÓN MANUAL desde consola')
      setIsDetecting(false)
    }
    console.log('🔧 Funciones globales disponibles: window.activateDetection() y window.deactivateDetection()')
  }, [])

  // Limpiar recursos al desmontar el componente
  useEffect(() => {
    return () => {
      stopCamera()
    }
  }, [])

  return (
    <div className="space-y-6">
      {/* Video Container */}
      <div className="relative bg-slate-900 rounded-xl overflow-hidden shadow-2xl">
        <div className="relative">
          <video
            ref={videoRef}
            className="w-full h-auto"
            style={{ transform: 'scaleX(-1)' }}
            playsInline
            muted
          />
          <canvas
            ref={canvasRef}
            className="absolute top-0 left-0 w-full h-full"
            style={{ transform: 'scaleX(-1)' }}
          />

          {/* Overlay de estado */}
          <div className="absolute top-4 left-4 flex items-center space-x-2">
            <div className={`w-3 h-3 rounded-full ${isCameraOn ? 'bg-emerald-400' : 'bg-red-400'} animate-pulse`}></div>
            <span className="text-sm font-medium text-white bg-black/60 backdrop-blur-sm px-3 py-1 rounded-lg">
              {isCameraOn ? 'Cámara Activa' : 'Cámara Inactiva'}
            </span>
          </div>

          {/* Indicador de detección */}
          {isCameraOn && (
            <div className="absolute top-4 right-4">
              <div className="flex items-center space-x-2">
                <div className={`w-3 h-3 rounded-full ${isDetecting ? 'bg-blue-400 animate-pulse' : 'bg-slate-400'}`}></div>
                <span className="text-sm font-medium text-white bg-black/60 backdrop-blur-sm px-3 py-1 rounded-lg">
                  {isDetecting ? 'IA Detectando' : 'IA Inactiva'}
                </span>
              </div>
            </div>
          )}

          {/* Indicador de estado de detección mejorado */}
          <div className="absolute bottom-4 left-4">
            <div className={`px-4 py-2 rounded-lg text-white font-bold ${isDetecting
              ? 'bg-green-500 animate-pulse'
              : 'bg-red-500'
              }`}>
              {isDetecting ? '🟢 DETECTANDO' : '🔴 DETECCIÓN INACTIVA'}
            </div>
            {/* Indicadores de calidad removidos */}
          </div>

          {/* Letra detectada mejorada y más exacta */}
          {stablePrediction && stablePrediction.predicted_word && stablePrediction.predicted_word !== '?' && (
            <div className="absolute top-4 right-4 bg-gradient-to-br from-green-500/90 to-emerald-600/90 text-white p-6 rounded-2xl backdrop-blur-sm border-2 border-green-300/50 shadow-2xl">
              <div className="text-center">
                {/* Letra principal */}
                <div className="text-8xl font-black text-white mb-3 drop-shadow-lg">
                  {stablePrediction.predicted_word}
                </div>

                {/* Información de confianza */}
                {stablePrediction.confidence && (
                  <div className="text-2xl font-bold text-green-200 mb-2">
                    {Math.round(stablePrediction.confidence * 100)}% confianza
                  </div>
                )}

                {/* Tiempo de procesamiento */}
                {stablePrediction.processing_time_ms && (
                  <div className="text-sm text-green-100">
                    {stablePrediction.processing_time_ms}ms
                  </div>
                )}

                {/* Indicadores de calidad removidos */}

                {/* Tiempo de procesamiento removido */}

                {/* Indicador de estabilidad */}
                {isGestureStable && (
                  <div className="mt-2 flex items-center justify-center">
                    <div className="w-2 h-2 bg-green-300 rounded-full animate-pulse mr-2"></div>
                    <span className="text-xs text-green-100 font-medium">Gesto Estable</span>
                  </div>
                )}
              </div>
            </div>
          )}

          {/* Indicador de predicción en proceso */}
          {prediction && !stablePrediction && (
            <div className="absolute top-4 right-4 bg-yellow-500/80 text-white p-4 rounded-xl backdrop-blur-sm border border-yellow-300/50">
              <div className="text-center">
                <div className="text-4xl font-bold text-white mb-2">
                  {prediction.predicted_word}
                </div>
                <div className="text-sm text-yellow-100">
                  Analizando estabilidad...
                </div>
                <div className="text-xs text-yellow-200 mt-1">
                  {consecutivePredictions}/2 confirmaciones
                </div>
              </div>
            </div>
          )}

          {/* FPS Counter */}
          {fps > 0 && (
            <div className="absolute bottom-4 right-4">
              <span className="text-sm font-medium text-white bg-black/60 backdrop-blur-sm px-3 py-1 rounded-lg">
                {fps} FPS
              </span>
            </div>
          )}
        </div>
      </div>

      {/* Controles */}
      <div className="flex items-center justify-between">
        <div className="flex items-center space-x-3">
          {!isCameraOn ? (
            <button
              onClick={startCamera}
              className="bg-gradient-to-r from-emerald-500 to-emerald-600 hover:from-emerald-600 hover:to-emerald-700 text-white font-semibold py-3 px-6 rounded-xl transition-all duration-200 shadow-lg hover:shadow-xl flex items-center space-x-2"
            >
              <Play className="h-5 w-5" />
              <span>Iniciar Cámara</span>
            </button>
          ) : (
            <button
              onClick={stopCamera}
              className="bg-gradient-to-r from-red-500 to-red-600 hover:from-red-600 hover:to-red-700 text-white font-semibold py-3 px-6 rounded-xl transition-all duration-200 shadow-lg hover:shadow-xl flex items-center space-x-2"
            >
              <Pause className="h-5 w-5" />
              <span>Detener Cámara</span>
            </button>
          )}

          {isCameraOn && (
            <button
              onClick={() => {
                const newDetectingState = !isDetecting
                console.log('🔄 Cambiando estado de detección:', newDetectingState)
                console.log('🔄 Estado actual:', isDetecting)
                setIsDetecting(newDetectingState)
                console.log('🔄 Estado después del setState:', newDetectingState)
              }}
              className={`flex items-center space-x-3 font-bold py-4 px-8 rounded-xl transition-all duration-200 shadow-lg hover:shadow-xl text-lg ${isDetecting
                ? 'bg-gradient-to-r from-emerald-500 to-emerald-600 hover:from-emerald-600 hover:to-emerald-700 text-white animate-pulse'
                : 'bg-gradient-to-r from-blue-500 to-blue-600 hover:from-blue-600 hover:to-blue-700 text-white'
                }`}
              style={{ zIndex: 1000 }}
            >
              <div className={`w-3 h-3 rounded-full ${isDetecting ? 'bg-emerald-300 animate-pulse' : 'bg-blue-300'}`}></div>
              <span>{isDetecting ? '🟢 DETECTANDO' : '🚀 INICIAR DETECCIÓN'}</span>
            </button>
          )}
        </div>

        <div className="flex items-center space-x-3">
          {/* Botón de Emergencia para Activar Detección */}
          <button
            onClick={() => {
              console.log('🚨 BOTÓN DE EMERGENCIA: Activando detección')
              setIsDetecting(true)
            }}
            className="flex items-center space-x-2 px-4 py-2 rounded-lg bg-red-500 text-white hover:bg-red-600 transition-all duration-200"
          >
            <span className="text-sm font-medium">🚨 ACTIVAR DETECCIÓN</span>
          </button>

          {/* Toggle Landmarks */}
          <button
            onClick={() => setShowLandmarks(!showLandmarks)}
            className={`flex items-center space-x-2 px-4 py-2 rounded-lg transition-all duration-200 ${showLandmarks
              ? 'bg-blue-100 text-blue-700 hover:bg-blue-200'
              : 'bg-slate-100 text-slate-600 hover:bg-slate-200'
              }`}
          >
            {showLandmarks ? <Eye className="h-4 w-4" /> : <EyeOff className="h-4 w-4" />}
            <span className="text-sm font-medium">Landmarks</span>
          </button>

          {/* Settings */}
          <button
            onClick={() => setShowSettings(!showSettings)}
            className="flex items-center space-x-2 px-4 py-2 rounded-lg bg-slate-100 text-slate-600 hover:bg-slate-200 transition-all duration-200"
          >
            <Settings className="h-4 w-4" />
            <span className="text-sm font-medium">Configuración</span>
          </button>

          {/* Clear */}
          <button
            onClick={() => {
              setPrediction(null)
              setError(null)
              setDetectedLetters([])
            }}
            className="flex items-center space-x-2 px-4 py-2 rounded-lg bg-slate-100 text-slate-600 hover:bg-slate-200 transition-all duration-200"
          >
            <RotateCcw className="h-4 w-4" />
            <span className="text-sm font-medium">Limpiar</span>
          </button>
        </div>
      </div>

      {/* Settings Panel - Simplificado */}
      {showSettings && (
        <div className="bg-white/80 backdrop-blur-sm rounded-xl p-6 border border-slate-200">
          <h3 className="text-lg font-bold text-slate-800 mb-4">Configuración de Detección</h3>
          <div className="space-y-4">
            <div className="text-center">
              <p className="text-slate-600 mb-4">
                La detección está optimizada para reconocer las 23 letras del alfabeto de señas (A-Y, sin P).
              </p>
              <div className="bg-blue-50 border border-blue-200 rounded-lg p-4">
                <h4 className="font-semibold text-blue-800 mb-2">💡 Consejos para mejor detección:</h4>
                <ul className="text-sm text-blue-700 space-y-1 text-left">
                  <li>• Mantén buena iluminación frontal</li>
                  <li>• Asegúrate de que toda la mano sea visible</li>
                  <li>• Haz las señas de forma clara y pausada</li>
                  <li>• Mantén la seña por 1-2 segundos</li>
                </ul>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Error Display */}
      {error && (
        <div className="bg-red-50 border border-red-200 rounded-xl p-4 flex items-center space-x-3">
          <AlertCircle className="h-5 w-5 text-red-500 flex-shrink-0" />
          <span className="text-red-700 font-medium">{error}</span>
        </div>
      )}

      {/* Instructions */}
      {!isCameraOn && (
        <div className="bg-gradient-to-r from-blue-50 to-indigo-50 rounded-xl p-6 border border-blue-200">
          <h4 className="text-lg font-bold text-slate-800 mb-4 flex items-center">
            <Type className="h-5 w-5 mr-2 text-blue-600" />
            Instrucciones de Uso
          </h4>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div className="space-y-3">
              <h5 className="font-semibold text-slate-700">Configuración Inicial:</h5>
              <ul className="text-slate-600 space-y-2 text-sm">
                <li className="flex items-start">
                  <span className="text-blue-500 mr-2">1.</span>
                  <span>Haz clic en <strong>"Iniciar Cámara"</strong></span>
                </li>
                <li className="flex items-start">
                  <span className="text-blue-500 mr-2">2.</span>
                  <span>Permite el acceso a la cámara</span>
                </li>
                <li className="flex items-start">
                  <span className="text-blue-500 mr-2">3.</span>
                  <span>Activa <strong>"INICIAR DETECCIÓN"</strong></span>
                </li>
              </ul>
            </div>
            <div className="space-y-3">
              <h5 className="font-semibold text-slate-700">Para Mejores Resultados:</h5>
              <ul className="text-slate-600 space-y-2 text-sm">
                <li className="flex items-start">
                  <span className="text-emerald-500 mr-2">•</span>
                  <span>Buena iluminación frontal</span>
                </li>
                <li className="flex items-start">
                  <span className="text-emerald-500 mr-2">•</span>
                  <span>Mano completa visible en cámara</span>
                </li>
                <li className="flex items-start">
                  <span className="text-emerald-500 mr-2">•</span>
                  <span>Señas claras del alfabeto A-Y (23 letras, sin P)</span>
                </li>
                <li className="flex items-start">
                  <span className="text-emerald-500 mr-2">•</span>
                  <span>Mantén la seña por 1-2 segundos</span>
                </li>
              </ul>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}

export default LetterRecognition


