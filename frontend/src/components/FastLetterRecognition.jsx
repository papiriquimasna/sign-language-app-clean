import React, { useRef, useEffect, useState, useCallback } from 'react'
import { Hands } from '@mediapipe/hands'
import { Camera } from '@mediapipe/camera_utils'
import { predictLetter } from '../services/api'
import { Play, Pause, RotateCcw, Type } from 'lucide-react'

/**
 * Componente de reconocimiento de letras RÁPIDO y SIMPLE
 * Diseño optimizado para detección inmediata
 */

const FastLetterRecognition = ({ onLetterDetected }) => {
    const videoRef = useRef(null)
    const canvasRef = useRef(null)
    const handsRef = useRef(null)
    const cameraRef = useRef(null)

    // Estados principales - SIMPLIFICADOS
    const [isDetecting, setIsDetecting] = useState(false)
    const [isCameraOn, setIsCameraOn] = useState(false)
    const [currentLetter, setCurrentLetter] = useState(null)
    const [confidence, setConfidence] = useState(0)
    const [error, setError] = useState(null)
    const [detectedLetters, setDetectedLetters] = useState([])

    // Configuración OPTIMIZADA
    const predictionInterval = 200 // Cada 200ms para evitar spam
    const confidenceThreshold = 0.1 // BAJO - acepta casi todo
    const lastPredictionTime = useRef(0)
    const isPredicting = useRef(false) // Evitar múltiples peticiones simultáneas

    // Inicializar MediaPipe Hands
    useEffect(() => {
        if (!videoRef.current) return

        const hands = new Hands({
            locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/hands/${file}`
        })

        hands.setOptions({
            maxNumHands: 1,
            modelComplexity: 0, // MÁS RÁPIDO
            minDetectionConfidence: 0.1, // MUY PERMISIVO
            minTrackingConfidence: 0.1 // MUY PERMISIVO
        })

        hands.onResults(onResults)
        handsRef.current = hands

        return () => {
            if (handsRef.current) {
                handsRef.current.close()
            }
        }
    }, [])

    // Función de resultados - SIMPLIFICADA
    const onResults = useCallback((results) => {
        if (!canvasRef.current || !videoRef.current) return

        const canvas = canvasRef.current
        const ctx = canvas.getContext('2d')
        const video = videoRef.current

        // Limpiar canvas
        ctx.save()
        ctx.clearRect(0, 0, canvas.width, canvas.height)

        // Dibujar video
        ctx.drawImage(results.image, 0, 0, canvas.width, canvas.height)

        // Si hay mano detectada
        if (results.multiHandLandmarks && results.multiHandLandmarks.length > 0) {
            const landmarks = results.multiHandLandmarks[0]

            // SIEMPRE dibujar landmarks cuando hay mano
            drawLandmarks(ctx, landmarks)

            // Solo hacer predicción si estamos detectando
            if (isDetecting) {
                const now = Date.now()
                if (now - lastPredictionTime.current > predictionInterval && !isPredicting.current) {
                    lastPredictionTime.current = now
                    performFastPrediction(landmarks)
                }
            }
        } else {
            // Mostrar mensaje cuando no hay mano detectada
            ctx.fillStyle = 'rgba(255, 255, 255, 0.8)'
            ctx.font = '20px Arial'
            ctx.textAlign = 'center'
            ctx.fillText('Coloca tu mano frente a la cámara', canvas.width / 2, 50)
        }

        ctx.restore()
    }, [isDetecting])

    // Dibujar landmarks - MÁS VISIBLES
    const drawLandmarks = (ctx, landmarks) => {
        ctx.strokeStyle = '#00FF00'
        ctx.lineWidth = 3

        // Dibujar puntos MÁS GRANDES
        landmarks.forEach((landmark, index) => {
            const x = landmark.x * canvasRef.current.width
            const y = landmark.y * canvasRef.current.height

            // Punto exterior (borde)
            ctx.beginPath()
            ctx.arc(x, y, 6, 0, 2 * Math.PI)
            ctx.fillStyle = '#FFFFFF'
            ctx.fill()

            // Punto interior
            ctx.beginPath()
            ctx.arc(x, y, 4, 0, 2 * Math.PI)
            ctx.fillStyle = '#00FF00'
            ctx.fill()
        })

        // Dibujar conexiones MÁS VISIBLES
        const connections = [
            [0, 1], [1, 2], [2, 3], [3, 4], // Pulgar
            [0, 5], [5, 6], [6, 7], [7, 8], // Índice
            [5, 9], [9, 10], [10, 11], [11, 12], // Medio
            [9, 13], [13, 14], [14, 15], [15, 16], // Anular
            [13, 17], [17, 18], [18, 19], [19, 20], // Meñique
            [0, 17] // Base
        ]

        connections.forEach(([start, end]) => {
            const startPoint = landmarks[start]
            const endPoint = landmarks[end]

            ctx.beginPath()
            ctx.moveTo(startPoint.x * canvasRef.current.width, startPoint.y * canvasRef.current.height)
            ctx.lineTo(endPoint.x * canvasRef.current.width, endPoint.y * canvasRef.current.height)
            ctx.stroke()
        })
    }

    // Predicción OPTIMIZADA - CON CONTROL DE CONCURRENCIA
    const performFastPrediction = (landmarks) => {
        if (isPredicting.current) return // Evitar múltiples peticiones

        isPredicting.current = true

        const handData = landmarks.map(landmark => ({
            x: landmark.x,
            y: landmark.y,
            z: landmark.z
        }))

        // Llamada asíncrona con control de concurrencia
        predictLetter({
            hand_landmarks: handData,
            confidence_threshold: confidenceThreshold
        }).then(response => {
            if (response.predicted_word && response.predicted_word !== 'N/A') {
                // MOSTRAR INMEDIATAMENTE
                setCurrentLetter(response.predicted_word)
                setConfidence(response.confidence)

                // Agregar a historial
                setDetectedLetters(prev => {
                    const newList = [...prev, {
                        letter: response.predicted_word,
                        confidence: response.confidence,
                        timestamp: Date.now()
                    }]
                    return newList.slice(-10) // Mantener solo las últimas 10
                })

                // Notificar al componente padre
                if (onLetterDetected) {
                    onLetterDetected(response.predicted_word, response.confidence)
                }
            }
        }).catch(error => {
            console.error('Error en predicción rápida:', error)
            setError('Error de predicción')
        }).finally(() => {
            isPredicting.current = false // Liberar el lock
        })
    }

    // Iniciar cámara
    const startCamera = async () => {
        try {
            setError(null)
            const stream = await navigator.mediaDevices.getUserMedia({
                video: {
                    width: { ideal: 640 },
                    height: { ideal: 480 },
                    facingMode: 'user'
                }
            })

            if (videoRef.current) {
                videoRef.current.srcObject = stream
                videoRef.current.play()

                const camera = new Camera(videoRef.current, {
                    onFrame: async () => {
                        if (handsRef.current) {
                            await handsRef.current.send({ image: videoRef.current })
                        }
                    },
                    width: 640,
                    height: 480
                })

                cameraRef.current = camera
                camera.start()
                setIsCameraOn(true)
                setIsDetecting(true) // AUTO-ACTIVAR DETECCIÓN
            }
        } catch (err) {
            setError('Error al acceder a la cámara')
            console.error('Error de cámara:', err)
        }
    }

    // Detener cámara
    const stopCamera = () => {
        if (cameraRef.current) {
            cameraRef.current.stop()
            cameraRef.current = null
        }
        if (videoRef.current && videoRef.current.srcObject) {
            const tracks = videoRef.current.srcObject.getTracks()
            tracks.forEach(track => track.stop())
            videoRef.current.srcObject = null
        }
        setIsCameraOn(false)
        setIsDetecting(false)
    }

    // Limpiar detecciones
    const clearDetections = () => {
        setDetectedLetters([])
        setCurrentLetter(null)
        setConfidence(0)
    }

    return (
        <div className="w-full max-w-4xl mx-auto p-4">
            {/* TÍTULO SIMPLE */}
            <div className="text-center mb-6">
                <h1 className="text-3xl font-bold text-white mb-2">
                    🖐️ Reconocimiento Rápido de Señas
                </h1>
                <p className="text-gray-300">
                    Haz una seña y ve la letra instantáneamente
                </p>
            </div>

            <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
                {/* CÁMARA - CENTRAL */}
                <div className="lg:col-span-2">
                    <div className="relative bg-gray-900 rounded-lg overflow-hidden">
                        <video
                            ref={videoRef}
                            className="w-full h-auto"
                            style={{ display: 'none' }}
                        />
                        <canvas
                            ref={canvasRef}
                            className="w-full h-auto bg-gray-800"
                            width={640}
                            height={480}
                        />

                        {/* LETRA DETECTADA - GRANDE Y CENTRAL */}
                        {currentLetter && (
                            <div className="absolute top-4 left-4 bg-green-500 text-white px-8 py-4 rounded-xl shadow-2xl border-4 border-white">
                                <div className="text-6xl font-black">{currentLetter}</div>
                                <div className="text-lg font-bold opacity-90">
                                    {/* Confianza removida */}
                                </div>
                            </div>
                        )}

                        {/* CONTROLES SIMPLES */}
                        <div className="absolute bottom-4 left-4 right-4 flex gap-2">
                            {!isCameraOn ? (
                                <button
                                    onClick={startCamera}
                                    className="flex items-center gap-2 bg-green-600 hover:bg-green-700 text-white px-6 py-3 rounded-lg transition-colors text-lg font-bold"
                                >
                                    <Play size={24} />
                                    INICIAR DETECCIÓN
                                </button>
                            ) : (
                                <>
                                    <button
                                        onClick={clearDetections}
                                        className="flex items-center gap-2 bg-blue-600 hover:bg-blue-700 text-white px-4 py-2 rounded-lg transition-colors"
                                    >
                                        <RotateCcw size={20} />
                                        Limpiar
                                    </button>

                                    <button
                                        onClick={stopCamera}
                                        className="flex items-center gap-2 bg-red-600 hover:bg-red-700 text-white px-4 py-2 rounded-lg transition-colors"
                                    >
                                        <Pause size={20} />
                                        Parar
                                    </button>
                                </>
                            )}
                        </div>
                    </div>

                    {/* ESTADO SIMPLE */}
                    <div className="mt-4 text-center">
                        {isCameraOn && (
                            <div className="text-white text-lg font-bold">
                                <span className="text-green-400">🟢 DETECTANDO SEÑAS EN TIEMPO REAL</span>
                            </div>
                        )}
                    </div>
                </div>

                {/* PANEL LATERAL - SIMPLE */}
                <div className="space-y-4">
                    {/* LETRA ACTUAL - GRANDE */}
                    <div className="bg-gray-800 rounded-lg p-6">
                        <h3 className="text-white font-bold mb-4 flex items-center gap-2 text-lg">
                            <Type size={24} />
                            LETRA DETECTADA
                        </h3>

                        {currentLetter ? (
                            <div className="text-center">
                                <div className="text-8xl font-black text-green-400 mb-4 drop-shadow-lg">
                                    {currentLetter}
                                </div>
                                <div className="text-xl text-white font-bold bg-green-600 rounded-lg px-4 py-2">
                                    {Math.round(confidence * 100)}% confianza
                                </div>
                            </div>
                        ) : (
                            <div className="text-center text-gray-400">
                                <div className="text-6xl mb-4">?</div>
                                <div className="text-lg">Haz una seña</div>
                            </div>
                        )}
                    </div>

                    {/* HISTORIAL SIMPLE */}
                    <div className="bg-gray-800 rounded-lg p-4">
                        <h3 className="text-white font-semibold mb-3">
                            Últimas Detecciones
                        </h3>

                        <div className="space-y-2 max-h-40 overflow-y-auto">
                            {detectedLetters.length > 0 ? (
                                detectedLetters.slice().reverse().map((item, index) => (
                                    <div key={index} className="flex justify-between items-center bg-gray-700 rounded px-3 py-2">
                                        <span className="text-white font-bold text-lg">{item.letter}</span>
                                        <span className="text-gray-300 text-sm">
                                            {Math.round(item.confidence * 100)}%
                                        </span>
                                    </div>
                                ))
                            ) : (
                                <div className="text-gray-400 text-center py-4">
                                    No hay detecciones aún
                                </div>
                            )}
                        </div>
                    </div>

                    {/* ERRORES */}
                    {error && (
                        <div className="bg-red-600 text-white p-3 rounded-lg">
                            {error}
                        </div>
                    )}
                </div>
            </div>
        </div>
    )
}

export default FastLetterRecognition
