import React, { useRef, useEffect, useState, useCallback } from 'react'
import { Hands } from '@mediapipe/hands'
import { Camera } from '@mediapipe/camera_utils'
import { Play, Pause, RotateCcw } from 'lucide-react'

const SimpleHandDetection = () => {
  const videoRef = useRef(null)
  const canvasRef = useRef(null)
  const handsRef = useRef(null)
  const cameraRef = useRef(null)
  
  const [isCameraOn, setIsCameraOn] = useState(false)
  const [handDetected, setHandDetected] = useState(false)

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
      drawHandLandmarks(ctx, landmarks)
      setHandDetected(true)
    } else {
      setHandDetected(false)
    }

    ctx.restore()
  }, [])

  // Dibujar landmarks de la mano
  const drawHandLandmarks = (ctx, landmarks) => {
    // Dibujar conexiones
    const connections = [
      [0, 1], [1, 2], [2, 3], [3, 4], // Pulgar
      [0, 5], [5, 6], [6, 7], [7, 8], // Índice
      [5, 9], [9, 10], [10, 11], [11, 12], // Medio
      [9, 13], [13, 14], [14, 15], [15, 16], // Anular
      [13, 17], [17, 18], [18, 19], [19, 20], // Meñique
      [0, 17] // Base de la palma
    ]

    // Dibujar conexiones
    ctx.strokeStyle = '#00ff00'
    ctx.lineWidth = 2
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
      
      // Color según el tipo de punto
      ctx.fillStyle = getPointColor(index)
      ctx.beginPath()
      ctx.arc(x, y, 6, 0, 2 * Math.PI)
      ctx.fill()
      
      // Borde blanco
      ctx.strokeStyle = '#ffffff'
      ctx.lineWidth = 2
      ctx.stroke()
      
      // Número del punto
      ctx.fillStyle = '#ffffff'
      ctx.font = 'bold 12px Arial'
      ctx.textAlign = 'center'
      ctx.strokeStyle = '#000000'
      ctx.lineWidth = 1
      ctx.strokeText(index.toString(), x, y - 10)
      ctx.fillText(index.toString(), x, y - 10)
    })
  }

  // Obtener color para cada punto
  const getPointColor = (index) => {
    if (index >= 0 && index <= 4) return '#ff0000' // Pulgar - Rojo
    if (index >= 5 && index <= 8) return '#ff8800' // Índice - Naranja
    if (index >= 9 && index <= 12) return '#00ff00' // Medio - Verde
    if (index >= 13 && index <= 16) return '#8800ff' // Anular - Morado
    if (index >= 17 && index <= 20) return '#ff0088' // Meñique - Rosa
    return '#666666' // Palma - Gris
  }

  // Iniciar cámara
  const startCamera = async () => {
    try {
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
      alert('Error al acceder a la cámara. Verifica los permisos.')
    }
  }

  // Detener cámara
  const stopCamera = () => {
    if (cameraRef.current) {
      cameraRef.current.stop()
      cameraRef.current = null
    }
    setIsCameraOn(false)
    setHandDetected(false)
  }

  // Limpiar recursos al desmontar
  useEffect(() => {
    return () => {
      stopCamera()
    }
  }, [])

  return (
    <div className="max-w-4xl mx-auto p-6">
      <div className="bg-white rounded-lg shadow-lg overflow-hidden">
        {/* Header */}
        <div className="bg-blue-600 text-white p-4">
          <h1 className="text-2xl font-bold text-center">
            Detección de Puntos de la Mano
          </h1>
          <p className="text-center text-blue-100 mt-1">
            Proyecto de Estudiante - MediaPipe Hands
          </p>
        </div>

        {/* Video Container */}
        <div className="relative bg-gray-900">
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

          {/* Indicador de mano detectada */}
          {isCameraOn && (
            <div className="absolute top-4 right-4">
              <div className="flex items-center space-x-2">
                <div className={`w-3 h-3 rounded-full ${handDetected ? 'bg-green-400 animate-pulse' : 'bg-gray-400'}`}></div>
                <span className="text-sm font-medium text-white bg-black/50 px-2 py-1 rounded">
                  {handDetected ? 'Mano detectada' : 'Buscando mano...'}
                </span>
              </div>
            </div>
          )}
        </div>

        {/* Controles */}
        <div className="p-4 bg-gray-50">
          <div className="flex items-center justify-center space-x-4">
            {!isCameraOn ? (
              <button
                onClick={startCamera}
                className="flex items-center space-x-2 bg-green-600 hover:bg-green-700 text-white font-medium py-2 px-6 rounded-lg transition-colors"
              >
                <Play className="h-4 w-4" />
                <span>Iniciar Cámara</span>
              </button>
            ) : (
              <button
                onClick={stopCamera}
                className="flex items-center space-x-2 bg-red-600 hover:bg-red-700 text-white font-medium py-2 px-6 rounded-lg transition-colors"
              >
                <Pause className="h-4 w-4" />
                <span>Detener Cámara</span>
              </button>
            )}
          </div>
        </div>

        {/* Información */}
        <div className="p-4 bg-gray-100">
          <h3 className="text-lg font-semibold text-gray-800 mb-3">
            Información de los Puntos
          </h3>
          <div className="grid grid-cols-2 md:grid-cols-3 gap-4 text-sm">
            <div className="flex items-center space-x-2">
              <div className="w-4 h-4 bg-red-500 rounded-full"></div>
              <span>Pulgar (0-4)</span>
            </div>
            <div className="flex items-center space-x-2">
              <div className="w-4 h-4 bg-orange-500 rounded-full"></div>
              <span>Índice (5-8)</span>
            </div>
            <div className="flex items-center space-x-2">
              <div className="w-4 h-4 bg-green-500 rounded-full"></div>
              <span>Medio (9-12)</span>
            </div>
            <div className="flex items-center space-x-2">
              <div className="w-4 h-4 bg-purple-500 rounded-full"></div>
              <span>Anular (13-16)</span>
            </div>
            <div className="flex items-center space-x-2">
              <div className="w-4 h-4 bg-pink-500 rounded-full"></div>
              <span>Meñique (17-20)</span>
            </div>
            <div className="flex items-center space-x-2">
              <div className="w-4 h-4 bg-gray-500 rounded-full"></div>
              <span>Palma</span>
            </div>
          </div>
        </div>

        {/* Instrucciones */}
        <div className="p-4 bg-blue-50">
          <h3 className="text-lg font-semibold text-blue-800 mb-2">
            Instrucciones
          </h3>
          <ul className="text-sm text-blue-700 space-y-1">
            <li>• Haz clic en "Iniciar Cámara" para comenzar</li>
            <li>• Permite el acceso a la cámara cuando se solicite</li>
            <li>• Coloca tu mano frente a la cámara</li>
            <li>• Verás 21 puntos dibujados en tu mano</li>
            <li>• Cada punto tiene un número del 0 al 20</li>
            <li>• Los colores indican diferentes partes de la mano</li>
          </ul>
        </div>
      </div>
    </div>
  )
}

export default SimpleHandDetection

