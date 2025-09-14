# 🐳 Docker Configuration

Esta documentación explica cómo usar Docker para tu aplicación de reconocimiento de lenguaje de señas.

## 📁 Archivos Docker Creados

### Backend
- `backend/Dockerfile` - Imagen de producción para FastAPI
- `backend/.dockerignore` - Archivos a ignorar en el build

### Frontend
- `frontend/Dockerfile` - Imagen de producción para React
- `frontend/Dockerfile.dev` - Imagen de desarrollo para React
- `frontend/nginx.conf` - Configuración de Nginx
- `frontend/.dockerignore` - Archivos a ignorar en el build

### Orquestación
- `docker-compose.yml` - Configuración de producción
- `docker-compose.dev.yml` - Configuración de desarrollo
- `.dockerignore` - Archivos a ignorar en el build global

## 🚀 Comandos Docker

### Desarrollo Local

```bash
# Construir y ejecutar en modo desarrollo
docker-compose -f docker-compose.dev.yml up --build

# Ejecutar en segundo plano
docker-compose -f docker-compose.dev.yml up -d --build

# Ver logs
docker-compose -f docker-compose.dev.yml logs -f

# Detener servicios
docker-compose -f docker-compose.dev.yml down
```

### Producción Local

```bash
# Construir y ejecutar en modo producción
docker-compose up --build

# Ejecutar en segundo plano
docker-compose up -d --build

# Ver logs
docker-compose logs -f

# Detener servicios
docker-compose down
```

### Comandos Individuales

```bash
# Construir solo el backend
docker build -t sign-language-backend ./backend

# Construir solo el frontend
docker build -t sign-language-frontend ./frontend

# Ejecutar backend individualmente
docker run -p 8000:8000 sign-language-backend

# Ejecutar frontend individualmente
docker run -p 3000:80 sign-language-frontend
```

## 🔧 Configuración de Servicios

### Backend (FastAPI)
- **Puerto:** 8000
- **Health Check:** `/health`
- **API Docs:** `/docs`
- **Variables de entorno:**
  - `PYTHONPATH=/app`
  - `PYTHONUNBUFFERED=1`

### Frontend (React)
- **Puerto:** 80 (producción) / 3000 (desarrollo)
- **Servidor:** Nginx (producción) / Vite (desarrollo)
- **Variables de entorno:**
  - `NODE_ENV=production`

## 🌐 URLs de Acceso

### Desarrollo
- Backend: http://localhost:8000
- Frontend: http://localhost:3000
- API Docs: http://localhost:8000/docs

### Producción
- Backend: http://localhost:8000
- Frontend: http://localhost:3000
- API Docs: http://localhost:8000/docs

## 🛠️ Troubleshooting

### Problemas Comunes

1. **Puerto ya en uso:**
   ```bash
   # Cambiar puertos en docker-compose.yml
   ports:
     - "8001:8000"  # Backend en puerto 8001
     - "3001:80"    # Frontend en puerto 3001
   ```

2. **Permisos de archivos:**
   ```bash
   # En Linux/Mac, cambiar permisos
   chmod +x backend/Dockerfile
   chmod +x frontend/Dockerfile
   ```

3. **Limpiar imágenes:**
   ```bash
   # Eliminar imágenes no utilizadas
   docker system prune -a
   
   # Eliminar volúmenes
   docker volume prune
   ```

### Logs y Debugging

```bash
# Ver logs de un servicio específico
docker-compose logs backend
docker-compose logs frontend

# Ver logs en tiempo real
docker-compose logs -f backend

# Entrar al contenedor
docker-compose exec backend bash
docker-compose exec frontend sh
```

## 📊 Monitoreo

### Verificar Estado de Servicios

```bash
# Ver estado de contenedores
docker-compose ps

# Ver uso de recursos
docker stats

# Ver información de imágenes
docker images
```

### Health Checks

```bash
# Verificar backend
curl http://localhost:8000/health

# Verificar frontend
curl http://localhost:3000
```

## 🔄 Actualizaciones

### Reconstruir Imágenes

```bash
# Reconstruir sin cache
docker-compose build --no-cache

# Reconstruir y ejecutar
docker-compose up --build --force-recreate
```

### Actualizar Dependencias

1. **Backend:**
   - Editar `backend/requirements.txt`
   - Reconstruir imagen

2. **Frontend:**
   - Editar `frontend/package.json`
   - Reconstruir imagen

## 🚀 Optimizaciones

### Multi-stage Builds
Los Dockerfiles ya están optimizados con multi-stage builds para reducir el tamaño de las imágenes.

### Caching
- Las dependencias se instalan antes de copiar el código
- Esto mejora el tiempo de build en cambios de código

### Security
- Imágenes base oficiales y actualizadas
- Usuarios no-root cuando es posible
- Configuración de seguridad en Nginx

## 📝 Notas Importantes

1. **Desarrollo vs Producción:**
   - Usa `docker-compose.dev.yml` para desarrollo
   - Usa `docker-compose.yml` para producción

2. **Volúmenes:**
   - En desarrollo, el código se monta como volumen
   - En producción, el código se copia a la imagen

3. **Variables de Entorno:**
   - Configura las variables según tu entorno
   - Usa archivos `.env` para desarrollo local

4. **Redes:**
   - Los servicios se comunican a través de la red Docker
   - El frontend puede acceder al backend usando el nombre del servicio

¡Tu aplicación está lista para Docker! 🎉
