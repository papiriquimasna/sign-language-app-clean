# Dockerfile para Frontend React
FROM node:18-alpine as build

# Establecer directorio de trabajo
WORKDIR /app

# Copiar archivos de dependencias
COPY frontend/package.json ./

# Instalar dependencias
RUN npm install --omit=dev

# Copiar código fuente
COPY frontend/ .

# Construir la aplicación
RUN npm run build

# Etapa de producción con nginx
FROM nginx:alpine

# Copiar archivos construidos desde la etapa anterior
COPY --from=build /app/dist /usr/share/nginx/html

# Copiar configuración personalizada de nginx
COPY frontend/nginx.conf /etc/nginx/conf.d/default.conf

# Exponer puerto
EXPOSE 80

# Comando para ejecutar nginx
CMD ["nginx", "-g", "daemon off;"]