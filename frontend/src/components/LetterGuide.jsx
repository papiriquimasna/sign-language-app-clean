import React, { useState } from 'react'
import { ChevronLeft, ChevronRight, Hand } from 'lucide-react'

const LetterGuide = () => {
  const [currentLetter, setCurrentLetter] = useState(0)

  const letters = [
    { letter: 'A', description: 'Pulgar extendido, otros dedos cerrados', emoji: '👍' },
    { letter: 'B', description: 'Todos los dedos extendidos', emoji: '🖐️' },
    { letter: 'C', description: 'Forma de C (dedos curvados)', emoji: '🤏' },
    { letter: 'D', description: 'Índice extendido, otros cerrados', emoji: '👆' },
    { letter: 'E', description: 'Todos los dedos cerrados (puño)', emoji: '✊' },
    { letter: 'F', description: 'Índice y pulgar extendidos', emoji: '🤟' },
    { letter: 'G', description: 'Índice y pulgar en forma de pistola', emoji: '🔫' },
    { letter: 'H', description: 'Índice y medio extendidos', emoji: '✌️' },
    { letter: 'I', description: 'Meñique extendido, otros cerrados', emoji: '🤙' },
    { letter: 'K', description: 'Índice y medio extendidos, pulgar extendido', emoji: '✌️' },
    { letter: 'L', description: 'Índice y pulgar extendidos en L', emoji: '🤟' },
    { letter: 'M', description: 'Índice y medio extendidos, otros cerrados', emoji: '✌️' },
    { letter: 'N', description: 'Índice y medio extendidos, otros cerrados', emoji: '✌️' },
    { letter: 'O', description: 'Forma de O (dedos curvados)', emoji: '👌' },
    { letter: 'Q', description: 'Índice y pulgar extendidos', emoji: '🤟' },
    { letter: 'R', description: 'Índice y medio cruzados', emoji: '✌️' },
    { letter: 'S', description: 'Mano cerrada (puño)', emoji: '✊' },
    { letter: 'T', description: 'Pulgar entre índice y medio', emoji: '🤞' },
    { letter: 'U', description: 'Índice y medio extendidos juntos', emoji: '✌️' },
    { letter: 'V', description: 'Índice y medio extendidos separados', emoji: '✌️' },
    { letter: 'W', description: 'Tres dedos extendidos (índice, medio, anular)', emoji: '🤟' },
    { letter: 'X', description: 'Índice doblado', emoji: '🤞' },
    { letter: 'Y', description: 'Meñique y pulgar extendidos', emoji: '🤙' }
  ]

  const nextLetter = () => {
    setCurrentLetter((prev) => (prev + 1) % letters.length)
  }

  const prevLetter = () => {
    setCurrentLetter((prev) => (prev - 1 + letters.length) % letters.length)
  }

  return (
    <div className="bg-dark-700 rounded-xl p-6">
      <div className="flex items-center space-x-2 mb-4">
        <Hand className="h-5 w-5 text-blue-400" />
        <h3 className="text-lg font-semibold text-white">Guía de Letras del Alfabeto de Señas (23 letras)</h3>
      </div>

      <div className="bg-dark-600 rounded-lg p-6">
        <div className="text-center">
          <div className="text-6xl mb-4">{letters[currentLetter].emoji}</div>
          <div className="text-4xl font-bold text-blue-400 mb-2">
            Letra {letters[currentLetter].letter}
          </div>
          <div className="text-lg text-gray-300 mb-6">
            {letters[currentLetter].description}
          </div>
        </div>

        <div className="flex items-center justify-between">
          <button
            onClick={prevLetter}
            className="flex items-center space-x-2 bg-dark-500 hover:bg-dark-400 text-white px-4 py-2 rounded-lg transition-colors"
          >
            <ChevronLeft className="h-4 w-4" />
            <span>Anterior</span>
          </button>

          <div className="text-sm text-gray-400">
            {currentLetter + 1} de {letters.length}
          </div>

          <button
            onClick={nextLetter}
            className="flex items-center space-x-2 bg-dark-500 hover:bg-dark-400 text-white px-4 py-2 rounded-lg transition-colors"
          >
            <span>Siguiente</span>
            <ChevronRight className="h-4 w-4" />
          </button>
        </div>
      </div>

      <div className="mt-4 text-sm text-gray-400 text-center">
        💡 Coloca tu mano frente a la cámara y haz la seña correspondiente
      </div>
    </div>
  )
}

export default LetterGuide
