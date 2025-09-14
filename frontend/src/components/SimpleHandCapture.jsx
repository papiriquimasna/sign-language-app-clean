import React, { useRef, useEffect, useState } from 'react'
import { Play, Pause, RotateCcw, AlertCircle, Camera, Bug } from 'lucide-react'
import { checkCameraSupport, testCameraAccess, logDiagnostics } from '../utils/cameraDiagnostics'

const SimpleHandCapture = () => {
  const videoRef = useRef(null)
  const canvasRef = useRef(null)
  
  const [isCameraOn, setIsCameraOn] = useState(false)
  const [error, setError] = useState(null)
  const [stream, setStream] = useState(null)
  const [diagnostics, setDiagnostics] = useState(null)

  // Iniciar cámara
  const startCamera = async () => {
    try {
      setError(null)
      
      // Solicitar acceso a la cámara
      const mediaStream = await navigator.mediaDevices.getUserMedia({
        video: {
          width: { ideal: 640 },
          height: { ideal: 480 },
          facingMode: 'user'
        }
      })
      
      if (videoRef.current) {
        // Limpiar cualquier fuente anterior
        videoRef.current.srcObject = null
        
        // Asignar el nuevo stream
        videoRef.current.srcObject = mediaStream
        
        // Esperar a que el video se cargue
        videoRef.current.onloadedmetadata = () => {
          videoRef.current.play()
          console.log('Video cargado y reproduciéndose')
        }
        
        setStream(mediaStream)
        setIsCameraOn(true)
        console.log('Cámara iniciada correctamente')
      }
    } catch (err) {
      console.error('Error al acceder a la cámara:', err)
      setError(`Error al acceder a la cámara: ${err.message}`)
    }
  }

  // Detener cámara
  const stopCamera = () => {
    if (stream) {
      stream.getTracks().forEach(track => track.stop())
      setStream(null)
    }
    if (videoRef.current) {
      videoRef.current.srcObject = null
    }
    setIsCameraOn(false)
    setError(null)
  }

  // Ejecutar diagnósticos al cargar el componente
  useEffect(() => {
    const runDiagnostics = async () => {
      const support = checkCameraSupport()
      const testResult = await testCameraAccess()
      
      setDiagnostics({
        support,
        testResult
      })
      
      // Log en consola para debugging
      logDiagnostics()
    }
    
    runDiagnostics()
  }, [])

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
            autoPlay
            controls={false}
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
        </div>

        <button
          onClick={() => {
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

      {/* Instructions */}
      {!isCameraOn && (
        <div className="bg-dark-700 rounded-lg p-4">
          <h4 className="text-sm font-medium text-white mb-2">Instrucciones:</h4>
          <ul className="text-sm text-dark-300 space-y-1">
            <li>• Haz clic en "Iniciar Cámara" para comenzar</li>
            <li>• Permite el acceso a la cámara cuando se solicite</li>
            <li>• Asegúrate de tener buena iluminación</li>
            <li>• Coloca tu mano frente a la cámara</li>
          </ul>
        </div>
      )}

      {/* Estado de la cámara */}
      {isCameraOn && (
        <div className="bg-green-900/50 border border-green-700 rounded-lg p-4">
          <div className="flex items-center space-x-3">
            <Camera className="h-5 w-5 text-green-400" />
            <div>
              <h4 className="text-sm font-medium text-green-200">Cámara funcionando</h4>
              <p className="text-sm text-green-300">La cámara está activa y lista para detectar manos</p>
            </div>
          </div>
        </div>
      )}

      {/* Panel de diagnósticos */}
      {diagnostics && (
        <div className="bg-dark-700 rounded-lg p-4">
          <div className="flex items-center space-x-2 mb-3">
            <Bug className="h-4 w-4 text-blue-400" />
            <h4 className="text-sm font-medium text-white">Diagnósticos de Cámara</h4>
          </div>
          
          <div className="space-y-2 text-xs">
            <div className="flex justify-between">
              <span className="text-dark-300">getUserMedia:</span>
              <span className={diagnostics.support.hasGetUserMedia ? 'text-green-400' : 'text-red-400'}>
                {diagnostics.support.hasGetUserMedia ? '✅ Disponible' : '❌ No disponible'}
              </span>
            </div>
            
            <div className="flex justify-between">
              <span className="text-dark-300">Contexto seguro:</span>
              <span className={diagnostics.support.isSecureContext ? 'text-green-400' : 'text-yellow-400'}>
                {diagnostics.support.isSecureContext ? '✅ Sí' : '⚠️ No (puede requerir HTTPS)'}
              </span>
            </div>
            
            <div className="flex justify-between">
              <span className="text-dark-300">Protocolo:</span>
              <span className="text-blue-400">{diagnostics.support.protocol}</span>
            </div>
            
            <div className="flex justify-between">
              <span className="text-dark-300">Prueba de acceso:</span>
              <span className={diagnostics.testResult.success ? 'text-green-400' : 'text-red-400'}>
                {diagnostics.testResult.success ? '✅ Exitoso' : '❌ Falló'}
              </span>
            </div>
            
            {!diagnostics.testResult.success && (
              <div className="mt-2 p-2 bg-red-900/30 rounded text-red-300">
                <strong>Error:</strong> {diagnostics.testResult.message}
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  )
}

export default SimpleHandCapture
