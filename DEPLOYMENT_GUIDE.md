# 🚀 Guía de Despliegue en Render

Esta guía te llevará paso a paso para desplegar tu aplicación de reconocimiento de lenguaje de señas en Render.

## 📋 Prerrequisitos

- Cuenta en [Render.com](https://render.com)
- Proyecto subido a GitHub
- Docker instalado localmente (para pruebas)

## 🏗️ Estructura del Proyecto

```
sign-language-app/
├── backend/                 # API FastAPI
│   ├── Dockerfile
│   ├── render.yaml
│   └── ...
├── frontend/               # App React
│   ├── Dockerfile
│   ├── render.yaml
│   └── ...
├── docker-compose.yml      # Para desarrollo local
└── render.yaml            # Configuración global
```

## 🔧 Configuración Local (Opcional)

### 1. Probar con Docker Compose

```bash
# Construir y ejecutar en modo producción
docker-compose up --build

# O en modo desarrollo
docker-compose -f docker-compose.dev.yml up --build
```

### 2. Verificar que todo funciona

- Backend: http://localhost:8000
- Frontend: http://localhost:3000
- API Docs: http://localhost:8000/docs

## 🌐 Despliegue en Render

### Paso 1: Preparar el Repositorio

1. **Subir a GitHub:**
   ```bash
   git add .
   git commit -m "Add Docker configuration for Render deployment"
   git push origin main
   ```

2. **Verificar que el repositorio esté público** (Render necesita acceso)

### Paso 2: Crear Cuenta en Render

1. Ve a [render.com](https://render.com)
2. Haz clic en "Get Started for Free"
3. Conecta tu cuenta de GitHub
4. Autoriza el acceso a tu repositorio

### Paso 3: Desplegar el Backend

1. **Crear nuevo servicio:**
   - Haz clic en "New +"
   - Selecciona "Web Service"

2. **Configurar el servicio:**
   - **Connect a repository:** Selecciona tu repositorio
   - **Name:** `sign-language-backend`
   - **Environment:** `Docker`
   - **Dockerfile Path:** `./backend/Dockerfile`
   - **Docker Context:** `./backend`
   - **Plan:** `Starter` (gratuito)

3. **Variables de entorno:**
   ```
   PYTHONPATH=/app
   PYTHONUNBUFFERED=1
   ```

4. **Configuración avanzada:**
   - **Health Check Path:** `/health`
   - **Auto-Deploy:** `Yes`

5. **Crear servicio:**
   - Haz clic en "Create Web Service"
   - Espera a que se construya (5-10 minutos)

### Paso 4: Desplegar el Frontend

1. **Crear nuevo servicio:**
   - Haz clic en "New +"
   - Selecciona "Web Service"

2. **Configurar el servicio:**
   - **Connect a repository:** Selecciona tu repositorio
   - **Name:** `sign-language-frontend`
   - **Environment:** `Docker`
   - **Dockerfile Path:** `./frontend/Dockerfile`
   - **Docker Context:** `./frontend`
   - **Plan:** `Starter` (gratuito)

3. **Variables de entorno:**
   ```
   NODE_ENV=production
   ```

4. **Crear servicio:**
   - Haz clic en "Create Web Service"
   - Espera a que se construya (5-10 minutos)

### Paso 5: Configurar URLs

1. **Obtener URLs de los servicios:**
   - Backend: `https://sign-language-backend.onrender.com`
   - Frontend: `https://sign-language-frontend.onrender.com`

2. **Actualizar configuración del frontend:**
   - Ve a la configuración del servicio frontend
   - Agrega variable de entorno:
     ```
     REACT_APP_API_URL=https://sign-language-backend.onrender.com
     ```

3. **Redeploy el frontend:**
   - Haz clic en "Manual Deploy" → "Deploy latest commit"

## 🔄 Despliegue Automático

### Configuración con render.yaml

Si prefieres usar el archivo `render.yaml`, puedes:

1. **Usar el archivo global:**
   - Render detectará automáticamente `render.yaml` en la raíz
   - Creará ambos servicios automáticamente

2. **O usar archivos individuales:**
   - `backend/render.yaml` para el backend
   - `frontend/render.yaml` para el frontend

## 🧪 Verificar el Despliegue

### 1. Probar el Backend

```bash
# Health check
curl https://sign-language-backend.onrender.com/health

# API docs
# Visita: https://sign-language-backend.onrender.com/docs
```

### 2. Probar el Frontend

- Visita: `https://sign-language-frontend.onrender.com`
- Verifica que la aplicación cargue correctamente
- Prueba la funcionalidad de reconocimiento de señas

## 🚨 Solución de Problemas

### Problemas Comunes

1. **Build falla:**
   - Verifica que todos los archivos estén en el repositorio
   - Revisa los logs de build en Render
   - Asegúrate de que las dependencias estén correctas

2. **Frontend no puede conectar al backend:**
   - Verifica la variable `REACT_APP_API_URL`
   - Asegúrate de que el backend esté funcionando
   - Revisa la configuración de CORS

3. **Servicio se duerme:**
   - Render pone a dormir servicios gratuitos después de 15 minutos de inactividad
   - La primera petición puede tardar 30-60 segundos
   - Considera actualizar a un plan de pago para evitar esto

### Logs y Debugging

1. **Ver logs:**
   - Ve a tu servicio en Render
   - Haz clic en "Logs"
   - Revisa los logs de build y runtime

2. **Debug local:**
   ```bash
   # Probar localmente con las mismas configuraciones
   docker-compose up --build
   ```

## 💰 Costos

- **Plan Starter (Gratuito):**
  - 750 horas/mes
  - Servicios se duermen después de 15 min de inactividad
  - Perfecto para desarrollo y pruebas

- **Plan Standard ($7/mes por servicio):**
  - Siempre activo
  - Mejor rendimiento
  - Recomendado para producción

## 🔄 Actualizaciones

Para actualizar tu aplicación:

1. **Haz cambios en tu código local**
2. **Commit y push a GitHub:**
   ```bash
   git add .
   git commit -m "Update application"
   git push origin main
   ```
3. **Render detectará automáticamente los cambios y redeployará**

## 📞 Soporte

- **Render Docs:** [render.com/docs](https://render.com/docs)
- **Render Community:** [community.render.com](https://community.render.com)
- **Status Page:** [status.render.com](https://status.render.com)

## ✅ Checklist Final

- [ ] Repositorio subido a GitHub
- [ ] Backend desplegado y funcionando
- [ ] Frontend desplegado y funcionando
- [ ] URLs configuradas correctamente
- [ ] Aplicación probada end-to-end
- [ ] Logs revisados para errores

¡Tu aplicación de reconocimiento de lenguaje de señas ya está desplegada en Render! 🎉
