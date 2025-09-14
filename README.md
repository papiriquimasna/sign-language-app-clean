# SignAI Pro - Reconocimiento de Lenguaje de Señas

Sistema de reconocimiento de lenguaje de señas en tiempo real usando inteligencia artificial, MediaPipe y modelos de machine learning entrenados.

## 🚀 Características

- **Reconocimiento en tiempo real** de letras del alfabeto de señas (A-S)
- **Interfaz web moderna** con React y TailwindCSS
- **Múltiples modelos de IA** entrenados con datasets reales
- **Detección de manos** con MediaPipe Hands
- **Precisión del 86.9%** con modelo entrenado en imágenes
- **API REST** con FastAPI
- **Sistema de entrenamiento personalizado** incluido

## 📋 Requisitos

- Python 3.8+
- Node.js 16+
- Cámara web
- Navegador moderno (Chrome, Firefox, Edge)

### ⚠️ Importante: Entorno Virtual
**Es obligatorio crear y activar un entorno virtual de Python** antes de instalar las dependencias para evitar conflictos con otras versiones de paquetes en tu sistema.

## 🛠️ Instalación

### 1. Clonar el repositorio
```bash
git clone <tu-repositorio>
cd sign-language-app
```

### 2. Crear entorno virtual para Python
```bash
# Crear entorno virtual
python -m venv venv

# Activar entorno virtual
# En Windows:
venv\Scripts\activate
# En macOS/Linux:
source venv/bin/activate
```

### 3. Instalar dependencias del Backend
```bash
cd backend
pip install -r requirements.txt
```

### 4. Instalar dependencias del Frontend
```bash
cd frontend
npm install
```

## ▶️ Ejecución

### Opción 1: Ejecución Manual (Recomendada)

**Terminal 1 - Backend:**
```bash
# Activar entorno virtual
venv\Scripts\activate  # Windows
# source venv/bin/activate  # macOS/Linux

cd backend
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

**Terminal 2 - Frontend:**
```bash
cd frontend
npm run dev
```

### Opción 2: Ejecución desde Directorio Raíz

**Terminal 1 - Backend:**
```bash
# Activar entorno virtual
venv\Scripts\activate  # Windows
# source venv/bin/activate  # macOS/Linux

cd backend && python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

**Terminal 2 - Frontend:**
```bash
cd frontend && npm run dev
```

## 🌐 Acceso

Una vez iniciados ambos servidores:

- **Frontend**: http://localhost:5173 (o el puerto que muestre Vite)
- **Backend API**: http://localhost:8000
- **Documentación API**: http://localhost:8000/docs

## 📱 Uso de la Aplicación

1. **Abrir el navegador** en http://localhost:5173
2. **Permitir acceso a la cámara** cuando se solicite
3. **Presionar "Iniciar Cámara"** para activar la cámara
4. **Presionar "Iniciar Detección"** para comenzar el reconocimiento
5. **Colocar la mano** frente a la cámara
6. **Hacer señas** de las letras A-S del alfabeto
7. **Ver las letras detectadas** en tiempo real

## 🎯 Letras Soportadas

El sistema reconoce las siguientes letras del alfabeto de señas:
**A, B, C, D, E, F, G, H, I, K, L, M, N, O, P, Q, R, S**

## 🧠 Modelos de IA Disponibles

1. **Modelo de Imágenes** (Prioritario)
   - Entrenado con dataset real de imágenes
   - Precisión: 86.9%
   - 18 letras disponibles

2. **Modelo Personalizado**
   - Entrenado por el usuario
   - Letras personalizadas

3. **Modelo Avanzado**
   - Ensemble de múltiples algoritmos
   - Fallback cuando otros modelos no están disponibles

## 🔧 Entrenamiento Personalizado

Para entrenar tu propio modelo:

```bash
cd backend
python trainer.py
```

Sigue las instrucciones en pantalla para:
- Entrenar letras específicas
- Probar el modelo entrenado
- Ver estadísticas de precisión

## 📊 Estructura del Proyecto

```
sign-language-app/
├── backend/                 # API FastAPI
│   ├── app/
│   │   ├── api/            # Endpoints REST
│   │   ├── services/       # Servicios de ML
│   │   ├── models/         # Modelos de datos
│   │   └── utils/          # Utilidades
│   ├── models/             # Modelos entrenados
│   └── trainer.py          # Sistema de entrenamiento
├── frontend/               # Aplicación React
│   ├── src/
│   │   ├── components/     # Componentes React
│   │   ├── services/       # Cliente API
│   │   └── utils/          # Utilidades
│   └── package.json
└── models/                 # Modelos globales
```

## 🐛 Solución de Problemas

### La cámara no se activa
- Verificar permisos del navegador
- Asegurarse de que no hay otras aplicaciones usando la cámara
- Probar en modo incógnito

### No se detectan letras
- Verificar que el backend esté ejecutándose en puerto 8000
- Comprobar la consola del navegador (F12) para errores
- Asegurarse de tener buena iluminación
- Mantener la mano a una distancia adecuada de la cámara

### Error de conexión con el backend
- Verificar que el backend esté ejecutándose
- Comprobar que no hay firewall bloqueando el puerto 8000
- Revisar los logs del backend para errores

### Problemas con el entorno virtual
- **Error "python no se reconoce"**: Asegúrate de tener Python instalado y en el PATH
- **Error al activar venv**: Verifica que estés en el directorio correcto del proyecto
- **Error de permisos en Windows**: Ejecuta PowerShell como administrador
- **Dependencias no se instalan**: Verifica que el entorno virtual esté activado (debe aparecer `(venv)` al inicio de la línea de comandos)

### Las letras no aparecen en la interfaz
- Verificar que se presionó "Iniciar Detección"
- Comprobar la consola del navegador para mensajes de debug
- Asegurarse de que la mano esté bien iluminada y visible

## 📈 Rendimiento

- **FPS**: 30-60 FPS dependiendo del hardware
- **Latencia**: <100ms para predicciones
- **Precisión**: 86.9% con modelo de imágenes
- **Memoria**: ~200MB RAM para el backend

## 🔒 Seguridad

- Todas las comunicaciones son locales (localhost)
- No se envían datos a servidores externos
- Los modelos se ejecutan localmente
- No se almacenan imágenes de la cámara

## 📝 Logs y Debug

### Backend
Los logs se guardan en `backend/app.log` y se muestran en la consola.

### Frontend
Abrir DevTools (F12) para ver logs detallados del reconocimiento.

## 🤝 Contribuir

1. Fork el proyecto
2. Crear una rama para tu feature
3. Commit tus cambios
4. Push a la rama
5. Abrir un Pull Request

## 📄 Licencia

Este proyecto está bajo la Licencia MIT. Ver el archivo LICENSE para más detalles.

## 🆘 Soporte

Si encuentras problemas:
1. Revisar la sección de solución de problemas
2. Verificar los logs del backend y frontend
3. Comprobar que todos los requisitos estén instalados
4. Crear un issue en el repositorio

---

**¡Disfruta usando SignAI Pro!** 🎉
