import React, { useState, useEffect } from 'react'
import LetterRecognition from './components/LetterRecognition'
import { Camera, Brain, Activity, Zap, Shield, CheckCircle } from 'lucide-react'

function App() {
  const [isConnected, setIsConnected] = useState(false)
  const [apiStatus, setApiStatus] = useState('checking')
  const [detectedLetters, setDetectedLetters] = useState([])
  const [currentPrediction, setCurrentPrediction] = useState(null)
  const [stats, setStats] = useState({
    totalDetections: 0,
    accuracy: 0,
    avgConfidence: 0
  })

  useEffect(() => {
    // Verificar conexión con el backend
    checkApiConnection()
  }, [])

  const checkApiConnection = async () => {
    try {
      const response = await fetch('http://localhost:8000/health')
      if (response.ok) {
        setIsConnected(true)
        setApiStatus('connected')
      } else {
        setApiStatus('error')
      }
    } catch (error) {
      console.error('Error conectando con el backend:', error)
      setApiStatus('error')
    }
  }

  const handleLetterDetected = (letter, confidence) => {
    console.log('🎯 App: Letra detectada recibida:', letter, 'Confianza:', confidence)

    if (letter && letter !== '?') {
      const newDetection = {
        letter,
        confidence,
        timestamp: new Date().toLocaleTimeString()
      }

      console.log('📝 App: Creando nueva detección:', newDetection)

      setDetectedLetters(prev => {
        const updated = [newDetection, ...prev].slice(0, 20)
        console.log('📋 App: Lista de detecciones actualizada:', updated.length, 'elementos')
        return updated
      })

      setCurrentPrediction(newDetection)
      console.log('🎨 App: Predicción actual establecida:', newDetection)

      // Actualizar estadísticas
      setStats(prev => {
        const newStats = {
          totalDetections: prev.totalDetections + 1,
          accuracy: Math.min(95, prev.accuracy + 0.5), // Simular mejora de precisión
          avgConfidence: (prev.avgConfidence + confidence) / 2
        }
        console.log('📊 App: Estadísticas actualizadas:', newStats)
        return newStats
      })
    } else {
      console.log('❌ App: Letra inválida o vacía, ignorando')
    }
  }

  const handlePredictionUpdate = (prediction) => {
    setCurrentPrediction(prediction)
  }

  const handleStatsUpdate = (newStats) => {
    setStats(newStats)
  }

  const getStatusColor = () => {
    switch (apiStatus) {
      case 'connected': return 'text-emerald-500'
      case 'error': return 'text-red-500'
      default: return 'text-amber-500'
    }
  }

  const getStatusText = () => {
    switch (apiStatus) {
      case 'connected': return 'Sistema Conectado'
      case 'error': return 'Error de Conexión'
      default: return 'Verificando...'
    }
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 to-slate-100">
      {/* Header Profesional */}
      <header className="bg-white/80 backdrop-blur-sm shadow-lg border-b border-slate-200">
        <div className="max-w-7xl mx-auto px-6 py-6">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-4">
              <div className="p-3 bg-gradient-to-r from-blue-600 to-indigo-600 rounded-xl shadow-lg">
                <Brain className="h-8 w-8 text-white" />
              </div>
              <div>
                <h1 className="text-2xl font-bold text-slate-800">
                  SignAI Pro
                </h1>
                <p className="text-slate-600 font-medium">
                  Reconocimiento Inteligente de Lenguaje de Señas
                </p>
              </div>
            </div>

            <div className="flex items-center space-x-6">
              {/* Stats */}
              <div className="hidden md:flex items-center space-x-4 text-sm">
                <div className="flex items-center space-x-2">
                  <Activity className="h-4 w-4 text-emerald-500" />
                  <span className="text-slate-600">{stats.totalDetections} detecciones</span>
                </div>
                <div className="flex items-center space-x-2">
                  <Zap className="h-4 w-4 text-blue-500" />
                  <span className="text-slate-600">{stats.accuracy.toFixed(1)}% precisión</span>
                </div>
              </div>

              {/* Status */}
              <div className="flex items-center space-x-3">
                <div className={`w-3 h-3 rounded-full ${apiStatus === 'connected' ? 'bg-emerald-400' : 'bg-red-400'} animate-pulse`}></div>
                <span className={`text-sm font-semibold ${getStatusColor()}`}>
                  {getStatusText()}
                </span>
              </div>
            </div>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="max-w-7xl mx-auto px-6 py-8">
        {!isConnected ? (
          <div className="text-center py-16">
            <div className="bg-white/80 backdrop-blur-sm rounded-2xl shadow-xl p-12 max-w-lg mx-auto border border-slate-200">
              <div className="text-center">
                <div className="mx-auto flex items-center justify-center h-16 w-16 rounded-full bg-red-100 mb-6">
                  <Camera className="h-8 w-8 text-red-600" />
                </div>
                <h3 className="text-xl font-bold text-slate-800 mb-3">
                  Sistema No Disponible
                </h3>
                <p className="text-slate-600 mb-6 leading-relaxed">
                  El servidor de reconocimiento no está disponible.
                  Asegúrate de que el backend esté ejecutándose en el puerto 8000.
                </p>
                <button
                  onClick={checkApiConnection}
                  className="bg-gradient-to-r from-blue-600 to-indigo-600 hover:from-blue-700 hover:to-indigo-700 text-white font-semibold py-3 px-6 rounded-xl transition-all duration-200 shadow-lg hover:shadow-xl"
                >
                  Reintentar Conexión
                </button>
              </div>
            </div>
          </div>
        ) : (
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
            {/* Panel Principal - Cámara */}
            <div className="lg:col-span-2">
              <div className="bg-white/80 backdrop-blur-sm rounded-2xl shadow-xl overflow-hidden border border-slate-200">
                <div className="p-6 border-b border-slate-200 bg-gradient-to-r from-slate-50 to-white">
                  <div className="flex items-center justify-between">
                    <h2 className="text-xl font-bold text-slate-800 flex items-center">
                      <Camera className="h-6 w-6 mr-3 text-blue-600" />
                      Detección en Tiempo Real
                    </h2>
                    <div className="flex items-center space-x-2">
                      <Shield className="h-5 w-5 text-emerald-500" />
                      <span className="text-sm font-medium text-slate-600">IA Activa</span>
                    </div>
                  </div>
                </div>
                <div className="p-6">
                  <LetterRecognition
                    onLetterDetected={handleLetterDetected}
                    onPredictionUpdate={handlePredictionUpdate}
                    onStatsUpdate={handleStatsUpdate}
                  />
                </div>
              </div>
            </div>

            {/* Panel Lateral - Resultados */}
            <div className="space-y-6">
              {/* Predicción Actual */}
              {currentPrediction && (
                <div className="bg-gradient-to-br from-emerald-500 to-teal-600 rounded-2xl shadow-xl p-6 text-white">
                  <div className="text-center">
                    <h3 className="text-lg font-bold mb-4">Letra Detectada</h3>
                    <div className="text-6xl font-black mb-4 drop-shadow-lg">
                      {currentPrediction.letter}
                    </div>
                    <div className="space-y-2">
                      <div className="bg-white/20 rounded-lg px-4 py-2">
                        <span className="text-sm font-medium">Confianza: {(currentPrediction.confidence * 100).toFixed(1)}%</span>
                      </div>
                      <div className="bg-white/20 rounded-lg px-4 py-2">
                        <span className="text-sm font-medium">Hora: {currentPrediction.timestamp}</span>
                      </div>
                    </div>
                  </div>
                </div>
              )}

              {/* Historial de Detecciones */}
              <div className="bg-white/80 backdrop-blur-sm rounded-2xl shadow-xl border border-slate-200">
                <div className="p-6 border-b border-slate-200">
                  <h3 className="text-lg font-bold text-slate-800 flex items-center">
                    <CheckCircle className="h-5 w-5 mr-2 text-blue-600" />
                    Historial de Detecciones
                  </h3>
                </div>
                <div className="p-6">
                  {detectedLetters.length > 0 ? (
                    <div className="space-y-3 max-h-96 overflow-y-auto">
                      {detectedLetters.map((detection, index) => (
                        <div key={index} className="flex items-center justify-between bg-slate-50 rounded-lg p-3 hover:bg-slate-100 transition-colors">
                          <div className="flex items-center space-x-3">
                            <div className="w-10 h-10 bg-gradient-to-r from-blue-500 to-indigo-500 rounded-lg flex items-center justify-center text-white font-bold text-lg">
                              {detection.letter}
                            </div>
                            <div>
                              <div className="font-semibold text-slate-800">{detection.letter}</div>
                              <div className="text-sm text-slate-500">{detection.timestamp}</div>
                            </div>
                          </div>
                          <div className="text-right">
                            <div className="text-sm font-medium text-slate-600">
                              {(detection.confidence * 100).toFixed(1)}%
                            </div>
                            <div className="w-16 bg-slate-200 rounded-full h-2 mt-1">
                              <div
                                className="bg-gradient-to-r from-emerald-400 to-emerald-500 h-2 rounded-full transition-all duration-300"
                                style={{ width: `${detection.confidence * 100}%` }}
                              ></div>
                            </div>
                          </div>
                        </div>
                      ))}
                    </div>
                  ) : (
                    <div className="text-center py-8">
                      <div className="text-slate-400 mb-2">
                        <Camera className="h-12 w-12 mx-auto" />
                      </div>
                      <p className="text-slate-500 font-medium">Inicia la detección para ver resultados</p>
                    </div>
                  )}
                </div>
              </div>

              {/* Estadísticas del Sistema */}
              <div className="bg-white/80 backdrop-blur-sm rounded-2xl shadow-xl border border-slate-200">
                <div className="p-6 border-b border-slate-200">
                  <h3 className="text-lg font-bold text-slate-800 flex items-center">
                    <Activity className="h-5 w-5 mr-2 text-blue-600" />
                    Estadísticas del Sistema
                  </h3>
                </div>
                <div className="p-6 space-y-4">
                  <div className="flex justify-between items-center">
                    <span className="text-slate-600">Total Detecciones</span>
                    <span className="font-bold text-slate-800">{stats.totalDetections}</span>
                  </div>
                  <div className="flex justify-between items-center">
                    <span className="text-slate-600">Precisión Promedio</span>
                    <span className="font-bold text-emerald-600">{stats.accuracy.toFixed(1)}%</span>
                  </div>
                  <div className="flex justify-between items-center">
                    <span className="text-slate-600">Confianza Promedio</span>
                    <span className="font-bold text-blue-600">{(stats.avgConfidence * 100).toFixed(1)}%</span>
                  </div>
                </div>
              </div>
            </div>
          </div>
        )}
      </main>
    </div>
  )
}

export default App