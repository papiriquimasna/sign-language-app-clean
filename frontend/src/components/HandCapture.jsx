import React, { useRef, useEffect, useState, useCallback } from 'react'
import { Hands } from '@mediapipe/hands'
import { Camera } from '@mediapipe/camera_utils'
import { predictSignLanguage } from '../services/api'
import { Play, Pause, RotateCcw, AlertCircle } from 'lucide-react'

const HandCapture = () => {
  const videoRef = useRef(null)
  const canvasRef = useRef(null)
  const handsRef = useRef(null)
  const cameraRef = useRef(null)
  
  const [isDetecting, setIsDetecting] = useState(false)
  const [isCameraOn, setIsCameraOn] = useState(false)
  const [prediction, setPrediction] = useState(null)
  const [error, setError] = useState(null)
  const [fps, setFps] = useState(0)
  const [lastPredictionTime, setLastPredictionTime] = useState(0)
  
  const predictionInterval = 500 // Predicir cada 500ms
  const confidenceThreshold = 0.7

  // Inicializar MediaPipe Hands
  const initializeHands = useCallback(() => {
    if (handsRef.current) return

    const hands = new Hands({
      locateFile: (file) => {
        return `https://cdn.jsdelivr.net/npm/@mediapipe/hands/${file}`
      }
    })

    hands.setOptions({
      maxNumHands: 1,
      modelComplexity: 1,
      minDetectionConfidence: 0.5,
      minTrackingConfidence: 0.5
    })

    hands.onResults(onResults)
    handsRef.current = hands
  }, [])

  // Procesar resultados de MediaPipe
  const onResults = useCallback((results) => {
    const canvas = canvasRef.current
    const video = videoRef.current
    
    if (!canvas || !video) return

    const ctx = canvas.getContext('2d')
    canvas.width = video.videoWidth
    canvas.height = video.videoHeight

    // Limpiar canvas
    ctx.save()
    ctx.clearRect(0, 0, canvas.width, canvas.height)
    ctx.drawImage(results.image, 0, 0, canvas.width, canvas.height)

    if (results.multiHandLandmarks && results.multiHandLandmarks.length > 0) {
      const landmarks = results.multiHandLandmarks[0]
      
      // Dibujar landmarks
      drawLandmarks(ctx, landmarks)
      
      // Realizar predicción si está habilitada
      if (isDetecting && Date.now() - lastPredictionTime > predictionInterval) {
        performPrediction(landmarks)
        setLastPredictionTime(Date.now())
      }
    } else {
      // No se detectó mano
      if (prediction) {
        setPrediction(null)
      }
    }

    ctx.restore()
  }, [isDetecting, lastPredictionTime, predictionInterval])

  // Dibujar landmarks en el canvas
  const drawLandmarks = (ctx, landmarks) => {
    ctx.strokeStyle = '#0ea5e9'
    ctx.lineWidth = 2

    // Dibujar conexiones
    const connections = [
      [0, 1], [1, 2], [2, 3], [3, 4], // Pulgar
      [0, 5], [5, 6], [6, 7], [7, 8], // Índice
      [5, 9], [9, 10], [10, 11], [11, 12], // Medio
      [9, 13], [13, 14], [14, 15], [15, 16], // Anular
      [13, 17], [17, 18], [18, 19], [19, 20], // Meñique
      [0, 17] // Base de la palma
    ]

    connections.forEach(([start, end]) => {
      const startPoint = landmarks[start]
      const endPoint = landmarks[end]
      
      ctx.beginPath()
      ctx.moveTo(startPoint.x * canvasRef.current.width, startPoint.y * canvasRef.current.height)
      ctx.lineTo(endPoint.x * canvasRef.current.width, endPoint.y * canvasRef.current.height)
      ctx.stroke()
    })

    // Dibujar puntos
    landmarks.forEach((landmark, index) => {
      const x = landmark.x * canvasRef.current.width
      const y = landmark.y * canvasRef.current.height
      
      ctx.fillStyle = getPointColor(index)
      ctx.beginPath()
      ctx.arc(x, y, 4, 0, 2 * Math.PI)
      ctx.fill()
      
      // Borde blanco
      ctx.strokeStyle = '#ffffff'
      ctx.lineWidth = 1
      ctx.stroke()
    })
  }

  // Obtener color para cada punto
  const getPointColor = (index) => {
    if (index >= 0 && index <= 4) return '#ef4444' // Pulgar
    if (index >= 5 && index <= 8) return '#f59e0b' // Índice
    if (index >= 9 && index <= 12) return '#10b981' // Medio
    if (index >= 13 && index <= 16) return '#8b5cf6' // Anular
    if (index >= 17 && index <= 20) return '#ec4899' // Meñique
    return '#6b7280' // Palma
  }

  // Realizar predicción
  const performPrediction = async (landmarks) => {
    try {
      // Convertir landmarks a formato requerido por el backend
      const handData = landmarks.map(landmark => ({
        x: landmark.x,
        y: landmark.y,
        z: landmark.z || 0
      }))

      const result = await predictSignLanguage(handData, confidenceThreshold)
      
      if (result && result.predicted_word && result.predicted_word !== 'gesto_no_reconocido') {
        setPrediction(result)
        setError(null)
      } else {
        setPrediction(null)
      }
    } catch (err) {
      console.error('Error en predicción:', err)
      setError('Error al procesar la predicción')
    }
  }

  // Iniciar cámara
  const startCamera = async () => {
    try {
      setError(null)
      
      if (!videoRef.current) return

      const camera = new Camera(videoRef.current, {
        onFrame: async () => {
          if (handsRef.current) {
            await handsRef.current.send({ image: videoRef.current })
          }
        },
        width: 640,
        height: 480
      })

      await camera.start()
      cameraRef.current = camera
      setIsCameraOn(true)
      
      // Inicializar MediaPipe si no está inicializado
      if (!handsRef.current) {
        initializeHands()
      }
    } catch (err) {
      console.error('Error iniciando cámara:', err)
      setError('Error al acceder a la cámara. Verifica los permisos.')
    }
  }

  // Detener cámara
  const stopCamera = () => {
    if (cameraRef.current) {
      cameraRef.current.stop()
      cameraRef.current = null
    }
    setIsCameraOn(false)
    setIsDetecting(false)
    setPrediction(null)
  }

  // Limpiar recursos al desmontar
  useEffect(() => {
    return () => {
      stopCamera()
    }
  }, [])

  return (
    <div className="space-y-6">
      {/* Video Container */}
      <div className="video-container bg-dark-700 rounded-xl overflow-hidden">
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
            <div className={`w-3 h-3 rounded-full ${isCameraOn ? 'bg-green-400' : 'bg-red-400'}`}></div>
            <span className="text-sm font-medium text-white bg-black/50 px-2 py-1 rounded">
              {isCameraOn ? 'Cámara activa' : 'Cámara inactiva'}
            </span>
          </div>

          {/* FPS Counter */}
          {fps > 0 && (
            <div className="absolute top-4 right-4">
              <span className="text-sm font-medium text-white bg-black/50 px-2 py-1 rounded">
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
              className="btn-primary flex items-center space-x-2"
            >
              <Play className="h-4 w-4" />
              <span>Iniciar Cámara</span>
            </button>
          ) : (
            <button
              onClick={stopCamera}
              className="btn-secondary flex items-center space-x-2"
            >
              <Pause className="h-4 w-4" />
              <span>Detener Cámara</span>
            </button>
          )}

          {isCameraOn && (
            <button
              onClick={() => setIsDetecting(!isDetecting)}
              className={`flex items-center space-x-2 ${
                isDetecting 
                  ? 'bg-green-600 hover:bg-green-700' 
                  : 'bg-dark-600 hover:bg-dark-500'
              } text-white font-medium py-2 px-4 rounded-lg transition-colors duration-200`}
            >
              <div className={`w-2 h-2 rounded-full ${isDetecting ? 'bg-green-300' : 'bg-gray-400'}`}></div>
              <span>{isDetecting ? 'Detectando' : 'Detener Detección'}</span>
            </button>
          )}
        </div>

        <button
          onClick={() => {
            setPrediction(null)
            setError(null)
          }}
          className="btn-secondary flex items-center space-x-2"
        >
          <RotateCcw className="h-4 w-4" />
          <span>Limpiar</span>
        </button>
      </div>

      {/* Error Display */}
      {error && (
        <div className="bg-red-900/50 border border-red-700 rounded-lg p-4 flex items-center space-x-3">
          <AlertCircle className="h-5 w-5 text-red-400 flex-shrink-0" />
          <span className="text-red-200">{error}</span>
        </div>
      )}

      {/* Prediction Display */}
      {prediction && (
        <div className="bg-green-900/50 border border-green-700 rounded-lg p-6">
          <div className="text-center">
            <h3 className="text-lg font-semibold text-green-200 mb-2">
              Palabra Detectada
            </h3>
            <div className="text-3xl font-bold text-green-100 mb-2">
              {prediction.predicted_word}
            </div>
            <div className="flex items-center justify-center space-x-4 text-sm text-green-300">
              <span>Confianza: {(prediction.confidence * 100).toFixed(1)}%</span>
              <span>•</span>
              <span>Tiempo: {prediction.processing_time_ms.toFixed(0)}ms</span>
            </div>
          </div>
        </div>
      )}

      {/* Instructions */}
      {!isCameraOn && (
        <div className="bg-dark-700 rounded-lg p-4">
          <h4 className="text-sm font-medium text-white mb-2">Instrucciones:</h4>
          <ul className="text-sm text-dark-300 space-y-1">
            <li>• Haz clic en "Iniciar Cámara" para comenzar</li>
            <li>• Permite el acceso a la cámara cuando se solicite</li>
            <li>• Coloca tu mano frente a la cámara</li>
            <li>• Activa la detección para comenzar a reconocer señas</li>
          </ul>
        </div>
      )}
    </div>
  )
}

export default HandCapture